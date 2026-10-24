"""Scripted hybrid defender: target assignment, collision-course guidance and interposition.

Pure torch with no Isaac imports, so the same code runs batched over envs in simulation and
on the flight laptop's CPU (one "env"). Inputs are positions and velocities in one shared
frame, z up, metres; outputs are velocity commands in that frame, capped at ``max_speed``.

Re-solved every control step:

1. threat      -- each live attacker's time until it could cross the protected cylinder
2. intercept   -- each defender's earliest collision-course intercept time per attacker
3. assignment  -- the defender -> attacker map (all A**D enumerated) that covers the most
                  attackers before they breach, most urgent first, then the least total time;
                  a defender's current target looks ``hysteresis`` seconds cheaper
4. guidance    -- fly at the predicted intercept point (the velocity-command form of
                  proportional navigation: hold a collision course); if the attacker cannot be
                  caught before the breach, or a teammate is already striking it, hold the
                  point on its line to the site instead (interposition)
5. separation  -- teammates closer than ``separation`` push apart
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

NEVER = 1.0e3  # [s] stands in for "no intercept" / "never breaches" in cost sums


@dataclass
class HybridDefenseCfg:
    max_speed: float = 1.0  # [m/s] defender speed the plan assumes (0.5 behind the real bridge)
    site_radius: float = 1.5  # [m] protected cylinder around the site axis
    guard_radius: float = 2.2  # [m] interposition point, horizontal distance from the site axis
    breach_margin: float = 0.5  # [s] an intercept must land this long before the breach to count
    terminal_range: float = 0.7  # [m] inside this, home straight at the attacker
    max_lead_time: float = 6.0  # [s] cap on how far ahead the intercept point is predicted
    hysteresis: float = 0.5  # [s] bonus for keeping the current target
    guard_gain: float = 2.0  # [1/s] proportional gain onto the interposition point
    separation: float = 0.7  # [m] teammates closer than this push apart
    separation_gain: float = 1.5  # [m/s per m] of overlap
    min_height: float = 0.5  # [m] floor clearance, in the frame's z
    path_guard_min: float = 0.5  # [m] site unknown: block at least this far ahead on the attacker's line
    always_block: bool = False  # zone defence: hold each attacker's line at guard_radius, strike only inside terminal_range
    # Collisions knock drones out (sim: a tilted Tello can't recover; real: it crashes). A capture
    # needs centres within capture_radius (0.35 m) but contact happens at ~0.1 m, so aim this far
    # to the side of the attacker instead of at it (0 = aim at its centre, the original behaviour)
    contact_offset: float = 0.0  # [m]
    obstacle_radius: float = 0.0  # [m] steer clear of downed attackers (they hover where caught); 0 = off
    obstacle_gain: float = 2.0  # [m/s per m] of intrusion
    idle_return: bool = False  # between waves, return to evenly spaced stations on the guard ring


def clamp_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Scale vectors (..., 3) down to at most ``max_norm``."""
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v * torch.clamp(max_norm / norm.clamp(min=1e-9), max=1.0)


def intercept_time(rel_pos: torch.Tensor, target_vel: torch.Tensor, speed: float) -> torch.Tensor:
    """Earliest t >= 0 with |rel_pos + target_vel * t| = speed * t, else ``inf``.

    ``rel_pos`` is target minus pursuer, (..., 3). A pursuer flying straight at the returned
    point meets a constant-velocity target there.
    """
    a = (target_vel * target_vel).sum(-1) - speed**2
    b = 2.0 * (rel_pos * target_vel).sum(-1)
    c = (rel_pos * rel_pos).sum(-1)
    inf = torch.full_like(c, math.inf)
    flat = a.abs() < 1e-9
    safe_a = torch.where(flat, torch.ones_like(a), a)
    disc = b * b - 4.0 * a * c
    root = torch.sqrt(disc.clamp(min=0.0))
    t1 = (-b - root) / (2.0 * safe_a)
    t2 = (-b + root) / (2.0 * safe_a)
    t1 = torch.where(t1 >= 0.0, t1, inf)
    t2 = torch.where(t2 >= 0.0, t2, inf)
    quadratic = torch.where(disc >= 0.0, torch.minimum(t1, t2), inf)
    # equal speeds: b t + c = 0
    linear = torch.where(b < 0.0, -c / b.clamp(max=-1e-9), inf)
    return torch.where(flat, linear, quadratic)


def time_to_breach(att_pos: torch.Tensor, att_vel: torch.Tensor, site: torch.Tensor, site_radius: float) -> torch.Tensor:
    """Worst-case time until each attacker could cross the site cylinder, (N, A).

    Assumes the attacker turns straight in at its current horizontal speed, which keeps the
    estimate stable for weaving attackers whose radial speed swings through zero.
    """
    offset = att_pos[..., :2] - site[:, None, :2]
    gap = (torch.linalg.norm(offset, dim=-1) - site_radius).clamp(min=0.0)
    speed = torch.linalg.norm(att_vel[..., :2], dim=-1).clamp(min=0.2)
    return gap / speed


def guard_point(att_pos: torch.Tensor, site: torch.Tensor, guard_radius: float) -> torch.Tensor:
    """Point on each attacker's straight line to the site centre, ``guard_radius`` out, (N, A, 3).

    An attacker already inside ``guard_radius`` is its own guard point.
    """
    offset = att_pos - site[:, None, :]
    radial = torch.linalg.norm(offset[..., :2], dim=-1, keepdim=True).clamp(min=1e-6)
    return site[:, None, :] + offset * torch.clamp(guard_radius / radial, max=1.0)


_TABLES: dict[tuple[int, int, str], torch.Tensor] = {}


def _assignments(num_defenders: int, num_attackers: int, device: torch.device) -> torch.Tensor:
    """Every defender -> attacker map, (A**D, D)."""
    key = (num_defenders, num_attackers, str(device))
    if key not in _TABLES:
        rows = list(itertools.product(range(num_attackers), repeat=num_defenders))
        _TABLES[key] = torch.tensor(rows, dtype=torch.long, device=device)
    return _TABLES[key]


class HybridDefender:
    """Stateful wrapper: keeps each defender's previous target for hysteresis."""

    def __init__(self, cfg: HybridDefenseCfg | None = None):
        self.cfg = cfg or HybridDefenseCfg()
        self.prev_target: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.prev_target is None or env_ids is None:
            self.prev_target = None
        else:
            self.prev_target[env_ids] = -1

    def __call__(
        self,
        def_pos: torch.Tensor,
        def_vel: torch.Tensor,
        att_pos: torch.Tensor,
        att_vel: torch.Tensor,
        att_alive: torch.Tensor,
        site: torch.Tensor,
        site_confident: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (velocity commands (N, D, 3), target index (N, D), interposing flag (N, D)).

        Shapes: def_* (N, D, 3), att_* (N, A, 3), att_alive (N, A) bool, site (N, 3) -- the
        centre of the protected cylinder. ``site_confident`` (N,) bool: where False the site is
        only a rough guess (asset position unknown). Nothing that beats plain pursuit is
        dropped: attackers are still ranked by urgency against the guess, assignment, lead and
        stagger stay, but the guess cannot rule an intercept out, and blocking points are taken
        on the attacker's own flight line (cut it off ahead) instead of its line to the site.
        """
        cfg = self.cfg
        n, num_def, _ = def_pos.shape
        num_att = att_pos.shape[1]
        device = def_pos.device
        if self.prev_target is None or self.prev_target.shape != (n, num_def):
            self.prev_target = torch.full((n, num_def), -1, dtype=torch.long, device=device)

        # 1-2: threat and intercept times
        t_breach = time_to_breach(att_pos, att_vel, site, cfg.site_radius)  # (N, A)
        trusted = torch.ones(n, dtype=torch.bool, device=device) if site_confident is None else site_confident
        # an untrusted site still ranks attackers, but cannot rule an intercept out
        t_cutoff = torch.where(trusted[:, None], t_breach, torch.full_like(t_breach, NEVER))
        rel = att_pos[:, None, :, :] - def_pos[:, :, None, :]  # (N, D, A, 3)
        t_int = intercept_time(rel, att_vel[:, None, :, :].expand_as(rel), cfg.max_speed)  # (N, D, A)
        alive = att_alive[:, None, :].expand(n, num_def, num_att)
        feasible = alive & (t_int < t_cutoff[:, None, :] - cfg.breach_margin)
        guards = guard_point(att_pos, site, cfg.guard_radius)  # (N, A, 3)
        t_guard = torch.linalg.norm(guards[:, None] - def_pos[:, :, None], dim=-1) / cfg.max_speed
        t_eff = torch.where(feasible, t_int, NEVER * 0.05 + t_guard)  # infeasible: cost of guarding
        if cfg.always_block:
            # zone defence: every live attacker is covered by whoever blocks its line, so the
            # assignment spreads defenders over attackers by how fast each can reach its point
            feasible = alive & torch.ones_like(feasible)
            t_eff = t_guard

        # 3: assignment over every defender -> attacker map
        table = _assignments(num_def, num_att, device)  # (K, D)
        k = table.shape[0]
        idx = table[None].expand(n, k, num_def)  # (N, K, D)
        pick = lambda x: x[:, None].expand(n, k, num_def, num_att).gather(-1, idx[..., None]).squeeze(-1)  # noqa: E731
        feas_k = pick(feasible)
        alive_k = att_alive[:, None, :].expand(n, k, num_att).gather(-1, idx)
        covered = (torch.nn.functional.one_hot(idx, num_att).bool() & feas_k[..., None]).any(dim=2)  # (N, K, A)
        urgency = 100.0 / (1.0 + t_breach)  # break count ties toward the attacker closest to breaching
        value = covered.float().sum(-1) * 1000.0 + (covered.float() * urgency[:, None, :]).sum(-1)
        cost = -value + pick(t_eff).sum(-1)
        cost = cost + (~alive_k).float().sum(-1) * 1.0e4
        cost = cost - (idx == self.prev_target[:, None, :]).float().sum(-1) * cfg.hysteresis
        target = table[cost.argmin(dim=-1)]  # (N, D)
        self.prev_target = target.clone()

        # stagger: when two defenders share a target, the slower one interposes instead
        t_sel = t_eff.gather(-1, target[..., None]).squeeze(-1)  # (N, D)
        same = target[:, :, None] == target[:, None, :]
        order = torch.arange(num_def, device=device)
        ahead = (t_sel[:, None, :] < t_sel[:, :, None]) | (
            (t_sel[:, None, :] == t_sel[:, :, None]) & (order[None, None, :] < order[None, :, None])
        )
        backup = (same & ahead & ~torch.eye(num_def, dtype=torch.bool, device=device)).any(dim=-1)

        # 4: guidance
        gather3 = lambda x: x.gather(1, target[..., None].expand(n, num_def, 3))  # noqa: E731
        tgt_pos, tgt_vel, tgt_guard = gather3(att_pos), gather3(att_vel), gather3(guards)
        feas_sel = feasible.gather(-1, target[..., None]).squeeze(-1)
        lead = t_int.gather(-1, target[..., None]).squeeze(-1)
        lead = torch.where(torch.isfinite(lead), lead.clamp(max=cfg.max_lead_time), torch.zeros_like(lead))
        aim = tgt_pos + tgt_vel * lead[..., None]
        close = torch.linalg.norm(tgt_pos - def_pos, dim=-1, keepdim=True) < cfg.terminal_range
        aim = torch.where(close, tgt_pos, aim)
        if cfg.contact_offset > 0.0:
            # pass beside the attacker: offset the aim point sideways (horizontal, perpendicular to
            # the line of sight), so the capture triggers before the bodies touch
            sight = aim - def_pos
            side = torch.stack([-sight[..., 1], sight[..., 0], torch.zeros_like(sight[..., 0])], dim=-1)
            side = side / torch.linalg.norm(side, dim=-1, keepdim=True).clamp(min=1e-6)
            aim = aim + side * cfg.contact_offset
        strike = aim - def_pos
        v_strike = strike / torch.linalg.norm(strike, dim=-1, keepdim=True).clamp(min=1e-6) * cfg.max_speed
        # site unknown: block on the attacker's own flight line, at the point ahead of it nearest
        # this defender (a stationary attacker is its own blocking point)
        heading = tgt_vel / torch.linalg.norm(tgt_vel, dim=-1, keepdim=True).clamp(min=1e-6)
        along = ((def_pos - tgt_pos) * heading).sum(-1, keepdim=True).clamp(min=cfg.path_guard_min)
        moving = torch.linalg.norm(tgt_vel[..., :2], dim=-1, keepdim=True) > 0.15
        path_guard = torch.where(moving, tgt_pos + heading * along, tgt_pos)
        tgt_guard = torch.where(trusted[:, None, None], tgt_guard, path_guard)
        v_guard = clamp_norm((tgt_guard - def_pos) * cfg.guard_gain, cfg.max_speed)
        interposing = ~feas_sel | backup
        if cfg.always_block:
            interposing = ~close.squeeze(-1)  # hold the line; strike only in the last metre
        v = torch.where(interposing[..., None], v_guard, v_strike)
        if cfg.idle_return:
            # between waves, go back to evenly spaced stations on the guard ring instead of
            # hovering wherever the last intercept left you
            stations = ring_stations(num_def, site, cfg.guard_radius, 0.0)
            stations[..., 2] = site[:, None, 2]
            idle = clamp_norm((stations - def_pos) * cfg.guard_gain, cfg.max_speed)
        else:
            idle = torch.zeros_like(v)  # hold
        v = torch.where(att_alive.any(dim=-1)[:, None, None], v, idle)

        # 5: separation, obstacle clearance and floor clearance
        if cfg.obstacle_radius > 0.0:
            # downed attackers keep hovering where they were caught: don't fly into them
            away = def_pos[:, :, None, :] - att_pos[:, None, :, :]  # (N, D, A, 3)
            gap = torch.linalg.norm(away, dim=-1, keepdim=True)
            intrusion = (cfg.obstacle_radius - gap).clamp(min=0.0) * (~att_alive)[:, None, :, None].float()
            v = v + (away / gap.clamp(min=1e-6) * intrusion * cfg.obstacle_gain).sum(dim=2)
        if num_def > 1:
            apart = def_pos[:, :, None, :] - def_pos[:, None, :, :]  # (N, D, D, 3)
            gap = torch.linalg.norm(apart, dim=-1, keepdim=True)
            overlap = (cfg.separation - gap).clamp(min=0.0)
            push = apart / gap.clamp(min=1e-6) * overlap * cfg.separation_gain
            v = v + push.sum(dim=2)
        low = def_pos[..., 2] < cfg.min_height
        v[..., 2] = torch.where(low, v[..., 2].clamp(min=0.0) + (cfg.min_height - def_pos[..., 2]), v[..., 2])
        return clamp_norm(v, cfg.max_speed), target, interposing


SITE_CONFIDENT = 0.2  # [s] SiteEstimator.confidence() above this -> trust the site estimate


class SiteEstimator:
    """Where the attackers are heading, for when the asset's position is unknown, (N, 2).

    Each live attacker moving faster than ``min_speed`` contributes the horizontal line through
    its position along its velocity; ``estimate`` is the least-squares meeting point of those
    lines over the last ``memory_s`` seconds (exponential forgetting). Averaging over time is
    what makes it usable on real drones: at 5 m, 10 degrees of heading noise in one instant
    moves a single-instant estimate by ~0.9 m, but every point of a straight attack run lies
    on the same line.

    ``believed`` is what to act on: ``estimate`` while ``confidence()`` is above
    ``SITE_CONFIDENT``, otherwise the last trusted estimate, or ``prior`` (set by ``reset``,
    e.g. the centroid of the defenders' start positions) before any. Nearly parallel lines --
    attackers arriving from one side -- pin the point sideways but not along their shared
    direction, where a few centimetres of heading wobble move it metres; the gate keeps that
    out of ``believed``.
    """

    def __init__(self, memory_s: float = 2.0, prior_weight: float = 0.02, min_speed: float = 0.25):
        self.memory_s = memory_s
        self.prior_weight = prior_weight
        self.min_speed = min_speed  # also skips attackers still accelerating, whose heading wobbles
        self.estimate: torch.Tensor | None = None
        self.believed: torch.Tensor | None = None
        self._lhs: torch.Tensor | None = None
        self._rhs: torch.Tensor | None = None

    def reset(self, prior: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Start over from ``prior`` (N, 2), or (len(env_ids), 2) for a subset of envs."""
        if env_ids is None or self.estimate is None:
            n = prior.shape[0]
            self.estimate = prior.clone()
            self.believed = prior.clone()
            self._lhs = torch.zeros(n, 2, 2, device=prior.device, dtype=prior.dtype)
            self._rhs = torch.zeros(n, 2, device=prior.device, dtype=prior.dtype)
        else:
            self.estimate[env_ids] = prior
            self.believed[env_ids] = prior
            self._lhs[env_ids] = 0.0
            self._rhs[env_ids] = 0.0

    def update(self, att_pos: torch.Tensor, att_vel: torch.Tensor, att_alive: torch.Tensor, dt: float) -> torch.Tensor:
        vel = att_vel[..., :2]
        speed = torch.linalg.norm(vel, dim=-1)
        direction = vel / speed.clamp(min=1e-6).unsqueeze(-1)
        weight = (att_alive & (speed > self.min_speed)).float()[..., None, None] * dt  # (N, A, 1, 1)
        eye = torch.eye(2, device=att_pos.device, dtype=att_pos.dtype)
        across = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)  # removes the along-line part
        decay = math.exp(-dt / self.memory_s)
        self._lhs = decay * self._lhs + (weight * across).sum(dim=1)
        self._rhs = decay * self._rhs + (weight * across @ att_pos[..., :2].unsqueeze(-1)).sum(dim=1).squeeze(-1)
        lhs = self.prior_weight * eye + self._lhs
        rhs = self.prior_weight * self.believed + self._rhs
        self.estimate = torch.linalg.solve(lhs, rhs)
        trusted = self.confidence() > SITE_CONFIDENT
        self.believed = torch.where(trusted[:, None], self.estimate, self.believed)
        return self.believed

    def confidence(self) -> torch.Tensor:
        """How well the estimate is pinned in its weakest direction, (N,).

        The smallest eigenvalue of the accumulated line evidence, in seconds: two lines w
        seconds old crossing at angle theta give about w * (1 - |cos theta|); one line, or
        parallel lines, give ~0. ``SITE_CONFIDENT`` (0.2) is roughly two lines 30 degrees
        apart for 1.5 s.
        """
        return torch.linalg.eigvalsh(self._lhs)[..., 0]


def ring_stations(num_defenders: int, site: torch.Tensor, radius: float, height: float) -> torch.Tensor:
    """Fixed stations evenly spaced on a ring around the site, (N, D, 3), z = ``height``.

    The no-sensing reference: what defenders achieve by holding a barrier without knowing
    where the attackers are.
    """
    angle = torch.arange(num_defenders, device=site.device, dtype=site.dtype) * (2.0 * math.pi / num_defenders)
    ring = torch.stack([torch.cos(angle) * radius, torch.sin(angle) * radius, torch.zeros_like(angle)], dim=-1)
    stations = site[:, None, :] + ring[None]
    stations[..., 2] = height
    return stations
