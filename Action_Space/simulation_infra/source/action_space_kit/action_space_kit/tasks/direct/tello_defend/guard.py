"""Scripted interceptor for the defend task: the base that a residual policy refines.

Greedy chase (scripts/evaluate.py) sends both defenders after the same nearest attacker and
leaves a captured one alone, so an attacker that holds where it was caught and retries is
already inside the defence. This guard instead:

1. assigns each defender its own attacker (minimum worst-case distance, with hysteresis so
   the pair does not swap every step),
2. flies at the lead-intercept point of an attacking target (where it will be, not where it is),
3. parks next to a target that is not attacking, on the side facing the site, so the moment
   it attacks again it is inside the capture radius. A target that teleports away simply
   moves the guard point, and the defender follows.

Everything is batched torch on world-frame positions, so the same code runs in the sim and
on the real drones (positions from the AprilTag map, velocities by differencing).
"""

from __future__ import annotations

import itertools

import torch


class InterceptGuard:
    def __init__(
        self,
        num_envs: int,
        num_defenders: int,
        num_attackers: int,
        device: str,
        speed: float = 1.2,
        gain: float = 2.5,
        guard_offset: float = 0.25,
        park_lead: float = 0.15,
        min_height: float = 0.5,
        lead_horizon: float = 3.0,
        hysteresis: float = 0.4,
        intercept_radius: float = 2.1,
        park_radius: float = 3.0,
        clearance: float = 0.7,
        repulsion: float = 5.0,
        dock_range: float = 1.5,
        dock_gap: float = 0.3,
        dock_speed: float = 1.0,
        goalkeep: bool = True,
    ):
        """speed: defender ground speed assumed for lead intercept [m/s]. gain: position gain
        [1/s]. guard_offset: distance kept from a parked target, towards the site [m].
        min_height: never aim below this, so a target sinking to the floor is not followed
        into it [m]. hysteresis: a new assignment must be this much better to replace the
        current one [m]. intercept_radius: never plan to meet an attacker closer to the site
        centre than this (site radius plus margin) [m]. park_radius: furthest out a parked
        target is guarded from [m]. clearance / repulsion: distance under which a defender is
        pushed away from drones it is not catching [m], and the push at contact [m/s]."""
        self.clearance = clearance
        self.repulsion = repulsion
        self.dock_range = dock_range  # [m] switch from intercepting to docking this close
        self.dock_gap = dock_gap  # [m] docking distance, inside the 0.35 m capture radius
        self.dock_speed = dock_speed  # [m/s] fastest approach relative to the target while docking
        self.goalkeep = goalkeep  # wait beside the path of an attacker coming at the defender        self.park_lead = park_lead  # [m] parked guard point is also this far towards the site
        self.intercept_radius = intercept_radius
        self.park_radius = park_radius
        self.speed = speed
        self.gain = gain
        self.guard_offset = guard_offset
        self.min_height = min_height
        self.lead_horizon = lead_horizon
        self.hysteresis = hysteresis
        self.device = device
        # every way to give each defender a distinct attacker; with more defenders than
        # attackers the extras double up
        if num_defenders <= num_attackers:
            perms = list(itertools.permutations(range(num_attackers), num_defenders))
        else:
            perms = list(itertools.product(range(num_attackers), repeat=num_defenders))
        self.perms = torch.tensor(perms, device=device, dtype=torch.long)  # (P, D)
        self.current = torch.zeros(num_envs, dtype=torch.long, device=device)  # index into perms

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.current.zero_()
        else:
            self.current[env_ids] = 0

    def aim_points(self, att_pos, att_vel, attacking, site, site_floor, def_pos):
        """Where each defender should go for each attacker: (E, D, A, 3)."""
        rel = att_pos.unsqueeze(1) - def_pos.unsqueeze(2)  # (E, D, A, 3)
        vel = att_vel.unsqueeze(1).expand_as(rel)
        # lead intercept: smallest t > 0 with |rel + vel t| = speed t
        a = (vel * vel).sum(-1) - self.speed**2
        b = 2.0 * (rel * vel).sum(-1)
        c = (rel * rel).sum(-1)
        disc = b * b - 4.0 * a * c
        sq = torch.sqrt(disc.clamp(min=0.0))
        safe_a = torch.where(a.abs() < 1e-6, torch.full_like(a, -1e-6), a)
        t1 = (-b - sq) / (2.0 * safe_a)
        t2 = (-b + sq) / (2.0 * safe_a)
        t = torch.where(t1 > 0, t1, t2)
        fallback = torch.sqrt(c) / self.speed  # no solution: the target is faster; aim ahead anyway
        t = torch.where((disc >= 0) & (t > 0), t, fallback).clamp(max=self.lead_horizon)
        lead = att_pos.unsqueeze(1) + vel * t.unsqueeze(-1)

        # Parked target: sit beside its path to the site, inside capture radius, so the moment
        # it attacks again it is recaptured. Beside, not in front: in front, the defender had to
        # keep clear of the attacker's run-up and was measured 0.4-0.7 m away when it got in.
        to_site = site.unsqueeze(1) - att_pos  # (E, A, 3)
        to_site = to_site / torch.clamp(torch.linalg.norm(to_site, dim=-1, keepdim=True), min=1e-6)
        # A little towards the site as well: a resuming attacker then flies past the defender
        # inside capture radius, instead of pulling away from a defender that reacts 0.3 s late.
        park = (
            att_pos.unsqueeze(1)
            + self.guard_offset * side_of(-rel, to_site.unsqueeze(1).expand_as(rel))
            + self.park_lead * to_site.unsqueeze(1)
        )

        # Perimeter rules, both on the target's straight line to the site centre (attackers
        # aim there, so the line shrinks radially):
        # - meeting an attacker inside intercept_radius is too late; wait on that ring instead
        # - guard a parked target from no further out than park_radius, so one that waits at
        #   the spawn ring (returned or teleported) does not drag its defender out there
        site_xy = site[:, :2].unsqueeze(1)  # (E, 1, 2)
        att_radial = torch.linalg.norm(att_pos[..., :2] - site_xy, dim=-1).clamp(min=1e-6)  # (E, A)

        def on_line(radius):
            frac = (radius / att_radial).clamp(max=1.0).unsqueeze(-1)
            return site.unsqueeze(1) + (att_pos - site.unsqueeze(1)) * frac  # (E, A, 3)

        lead_radial = torch.linalg.norm(lead[..., :2] - site_xy.unsqueeze(1), dim=-1)  # (E, D, A)
        crossing = on_line(self.intercept_radius).unsqueeze(1).expand_as(lead)
        late = (lead_radial < self.intercept_radius) & (att_radial > self.intercept_radius).unsqueeze(1)
        lead = torch.where(late.unsqueeze(-1), crossing, lead)
        park_radial = torch.linalg.norm(park[..., :2] - site_xy.unsqueeze(1), dim=-1)
        park = torch.where((park_radial > self.park_radius).unsqueeze(-1), on_line(self.park_radius).unsqueeze(1).expand_as(park), park)

        aim = torch.where(attacking.unsqueeze(1).unsqueeze(-1), lead, park).clone()
        aim[..., 2] = torch.maximum(aim[..., 2], (site_floor + self.min_height).view(-1, 1, 1))
        return aim

    def compute(self, def_pos, def_vel, att_pos, att_vel, attacking, site, site_floor, max_speed: float):
        """Velocity commands (E, D, 3) in [-1, 1] per axis (fraction of max_speed), the chosen
        target per defender (E, D) and the aim points (E, D, 3). site is the centre of the
        protected cylinder and site_floor (E,) the floor height below it."""
        aim = self.aim_points(att_pos, att_vel, attacking, site, site_floor, def_pos)  # (E, D, A, 3)
        dist = torch.linalg.norm(aim - def_pos.unsqueeze(2), dim=-1)  # (E, D, A)

        # cost of each assignment: the worst defender's distance, plus a little of the total
        num_d = def_pos.shape[1]
        d_idx = torch.arange(num_d, device=self.device)
        pair = dist[:, d_idx.unsqueeze(0), self.perms]  # (E, P, D)
        cost = pair.max(dim=-1).values + 0.1 * pair.sum(dim=-1)
        best = cost.argmin(dim=-1)
        keep = cost.gather(1, self.current.unsqueeze(1)).squeeze(1)
        switch = cost.gather(1, best.unsqueeze(1)).squeeze(1) < keep - self.hysteresis
        self.current = torch.where(switch, best, self.current)

        target = self.perms[self.current]  # (E, D)
        chosen = aim.gather(2, target.view(*target.shape, 1, 1).expand(-1, -1, 1, 3)).squeeze(2)
        # Close in on an attacking target: dock next to it at its own velocity instead of
        # ramming it, so the capture happens inside capture radius without contact. (Matching
        # a parked target's velocity is left out on purpose: once contact pushes the target,
        # following it pushes harder, and both were measured spinning off at 8-10 m/s.)
        tgt_pos = att_pos.gather(1, target.unsqueeze(-1).expand(-1, -1, 3))
        tgt_vel = att_vel.gather(1, target.unsqueeze(-1).expand(-1, -1, 3))
        tgt_attacking = attacking.gather(1, target)
        # Dock beside its path, not in front: head-on, the attacker's braking distance after
        # capture carried it into the defender (measured: contact at 0.07-0.2 m, both flipped).
        away = def_pos - tgt_pos
        gap = torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-6)
        docking = tgt_attacking.unsqueeze(-1) & (gap < self.dock_range)
        heading = tgt_vel / torch.linalg.norm(tgt_vel, dim=-1, keepdim=True).clamp(min=1e-3)
        side = side_of(away, heading)
        dock_point = tgt_pos + self.dock_gap * side
        rel = self.gain * (dock_point - def_pos)
        rel_len = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
        rel = rel * (self.dock_speed / rel_len).clamp(max=1.0)  # cap the closing speed
        follow = tgt_vel + rel
        # Coming at the defender: hold a point dock_gap beside its path and let it fly through
        # capture radius. Matching its velocity instead fails as soon as the attacker is the
        # faster drone (measured: 93% breaches against 0.8-1.4 m/s attackers).
        ahead = ((away * heading).sum(-1, keepdim=True) > 0.0) & self.goalkeep
        on_path = tgt_pos + (away * heading).sum(-1, keepdim=True) * heading + self.dock_gap * side
        goalie = self.gain * (on_path - def_pos)
        dock_point = torch.where(ahead, on_path, dock_point)
        chosen = torch.where(docking, dock_point, chosen)
        cmd = torch.where(docking, torch.where(ahead, goalie, follow), self.gain * (chosen - def_pos))

        # Keep clear of every drone it is not trying to catch right now (teammate, parked or
        # other attackers): touching knocks both out of the air, in the sim and for real.
        num_a = att_pos.shape[1]
        others = torch.cat([att_pos, def_pos], dim=1)  # (E, A + D, 3)
        rel = def_pos.unsqueeze(2) - others.unsqueeze(1)  # (E, D, A + D, 3)
        dist = torch.linalg.norm(rel, dim=-1).clamp(min=1e-3)
        is_target = torch.zeros(*target.shape, num_a, dtype=torch.bool, device=self.device)
        is_target.scatter_(2, target.unsqueeze(-1), True)
        not_self = ~torch.eye(num_d, dtype=torch.bool, device=self.device).unsqueeze(0).expand(def_pos.shape[0], -1, -1)
        avoid = torch.cat([~is_target, not_self], dim=-1) & (dist < self.clearance)
        push = ((self.clearance - dist) / self.clearance * avoid.float()).unsqueeze(-1) * rel / dist.unsqueeze(-1)
        cmd = cmd + self.repulsion * push.sum(dim=2)
        # Each rc channel is its own velocity set-point, so scale by the largest axis, not
        # the vector norm: flying diagonally then uses the full stick on two axes.
        cmd = cmd / max_speed
        peak = cmd.abs().max(dim=-1, keepdim=True).values.clamp(min=1.0)
        return cmd / peak, target, chosen


def side_of(offset, heading):
    """Unit vector perpendicular to heading, on the side offset points to (horizontal
    perpendicular when offset is dead ahead or behind)."""
    side = offset - (offset * heading).sum(-1, keepdim=True) * heading
    side_len = torch.linalg.norm(side, dim=-1, keepdim=True)
    flat = torch.stack([-heading[..., 1], heading[..., 0], torch.zeros_like(heading[..., 0])], dim=-1)
    flat = flat / torch.linalg.norm(flat, dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.where(side_len > 0.05, side / side_len.clamp(min=1e-6), flat)


def infer_attacking(att_pos, att_vel, site, threshold: float = 0.15):
    """Status when the game does not report it: moving towards the site horizontally."""
    to_site = site.unsqueeze(1) - att_pos
    to_site[..., 2] = 0.0
    to_site = to_site / torch.clamp(torch.linalg.norm(to_site, dim=-1, keepdim=True), min=1e-6)
    return (att_vel * to_site).sum(-1) > threshold
