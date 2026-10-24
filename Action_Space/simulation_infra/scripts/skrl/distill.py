# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Distil a scripted defender into the IPPO checkpoint that scripts/evaluate.py loads.

``scripts/evaluate.py`` scores a defender only as a skrl checkpoint, and PPO from scratch drifts
to doing nothing on AS-Defend-v0 (README, "Known issue"; its suggested fix is a behaviour-cloning
warm start from ``chase``). This builds the agent the way evaluate.py and train.py do, teaches
each defender's policy to reproduce the teacher with DAgger, fits the value function to the
student's own returns, and saves with skrl's ``agent.save()``, so both of these load it as is:

    python -u scripts/evaluate.py --task AS-Defend-v0 --checkpoint <run>/checkpoints/agent_distill.pt
    python -u scripts/skrl/train.py --task AS-Defend-v0 --algorithm IPPO --checkpoint <same> ...

Teachers (``--teacher``): ``chase``, evaluate.py's greedy-pursuit baseline, which uses only each
defender's own observation, so the student can copy it exactly; or ``hybrid``
(``action_space_kit/control/defense.py``), which also uses teammate positions the student cannot see.

DAgger: in round 0 the teacher flies; afterwards the student flies with probability 1 - 0.5**round,
and every visited state is labelled with the teacher's action, so the student learns to recover
from its own mistakes.

``--check chase|hybrid`` flies a scripted pilot on the default task and counts outcomes exactly as
evaluate.py does, so the teacher can be compared with chase before anything is distilled (run
``--check chase`` first: it must reproduce evaluate.py's chase numbers). For the hybrid it also
verifies every command: inside [-1, 1], yaw rate 0, speed under the cap, finite.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distil a scripted defender into an IPPO checkpoint.")
parser.add_argument("--task", type=str, default="AS-Defend-v0")
parser.add_argument("--teacher", type=str, default="chase", choices=["chase", "chase_live", "pursuit", "hybrid", "assigned"])
parser.add_argument("--lead", action="store_true", help="pursuit: aim at the predicted intercept point.")
parser.add_argument("--num_envs", type=int, default=None, help="Default: 64 for --check (as evaluate.py), 4096 to distil.")
parser.add_argument("--check", type=str, default=None, choices=["chase", "chase_live", "pursuit", "hybrid", "assigned"], help="Score a scripted pilot and exit.")
parser.add_argument("--episodes", type=int, default=128, help="Episodes for --check.")
parser.add_argument("--speed_cap", type=float, default=None, help="Teacher speed [m/s]; default the task's max.")
parser.add_argument("--rounds", type=int, default=12)
parser.add_argument("--steps", type=int, default=200, help="Env steps per DAgger round.")
parser.add_argument("--keep_every", type=int, default=2, help="Store every Nth step's samples.")
parser.add_argument("--grad_steps", type=int, default=600, help="Policy gradient steps per round.")
parser.add_argument("--batch", type=int, default=8192)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--log_std", type=float, default=-1.2, help="Saved exploration std = exp(this), for RL fine-tuning.")
parser.add_argument("--value_steps", type=int, default=700, help="Rollout length for fitting the value function.")
parser.add_argument("--value_fit", type=int, default=400, help="Leading steps of that rollout used as targets.")
parser.add_argument("--value_grad_steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--contact_offset", type=float, default=0.0, help="hybrid: aim this far beside the attacker [m].")
parser.add_argument("--obstacle_radius", type=float, default=0.0, help="hybrid: keep clear of downed attackers [m].")
parser.add_argument("--soft_closing", type=float, default=None, help="assigned: within 0.7 m match the attacker's velocity and close at this speed [m/s] (gentle contact).")
parser.add_argument("--terminal", type=float, default=0.7, help="assigned: terminal range [m].")
parser.add_argument("--block", type=float, default=None, help="assigned: when the intercept comes too late, wait on the attacker's line this far from the asset [m].")
parser.add_argument("--box_speed", action="store_true", help="assigned: use the per-axis action box (up to sqrt(2) faster diagonally) outside the terminal range.")
parser.add_argument("--urgent_eta", type=float, default=None, help="assigned: close faster once the attacker is this many seconds from breaching [s].")
parser.add_argument("--urgent_closing", type=float, default=0.5, help="assigned: closing speed when urgent [m/s].")
parser.add_argument("--escort", type=float, nargs=3, default=None, metavar=("UP", "AHEAD", "ENGAGE_ETA"), help="assigned: a free defender escorts its teammate's attacker instead of joining in [m, m, s].")
parser.add_argument("--cruise_up", type=float, default=0.0, help="assigned: outside the terminal range aim this far above the attacker [m].")
parser.add_argument("--helper_up", type=float, default=0.0, help="assigned: once its own attacker is down, a defender works this far above attacker level [m].")
parser.add_argument("--pair", type=str, default="slot", choices=["slot", "rescue", "nearest"], help="assigned: who takes which attacker (see assigned_actions).")
parser.add_argument("--pair_margin", type=float, default=0.0, help="assigned, --pair rescue: seconds of slack required to keep or take an attacker [s].")
parser.add_argument("--guard", type=float, nargs=2, default=None, metavar=("RADIUS", "ENGAGE_ETA"), help="assigned: a free defender waits on the live attacker's line this far from the asset until it is this close to breaching [m, s].")
parser.add_argument("--keeper", type=float, nargs=2, default=None, metavar=("RADIUS", "ENGAGE_ETA"), help="assigned: defenders past the attacker count hold a post this far from the asset on the most urgent attacker's line until it is this close to breaching [m, s].")
# RL fine-tuning prep (the hosts' recipe, SINGLE_DRONE_SIMULATION.md): small std, a critic fitted
# before the actor moves, no stale optimizer moments
parser.add_argument("--refit", type=str, default=None, help="Start from this checkpoint and skip DAgger: only re-set std and refit the critic.")
parser.add_argument("--gamma", type=float, default=None, help="Discount for the value targets; default the agent cfg's (0.99).")
parser.add_argument(
    "--preserve_policy",
    type=float,
    default=50.0,
    help="The policy and value share a trunk (models.separate: False). >0: fit the value through the trunk too, "
    "with this weight on keeping the policy's actions unchanged; 0: fit the value head only.",
)
parser.add_argument("--keep_optimizer", action="store_true", help="Keep optimizer state in the saved checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os
from datetime import datetime

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from action_space_kit.control.defense import SITE_CONFIDENT, HybridDefenseCfg, HybridDefender
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def teacher_actions(raw, hybrid: HybridDefender) -> dict[str, torch.Tensor]:
    """The hybrid's world-frame velocity commands as per-agent [vx, vy, vz, 0] actions."""
    origin = raw.scene.env_origins
    def_pos = torch.stack([d.data.root_pos_w - origin for d in raw._defenders], dim=1)
    def_vel = torch.stack([d.data.root_lin_vel_w for d in raw._defenders], dim=1)
    att_pos, att_vel = raw._attacker_states()
    site = torch.zeros_like(origin)
    site[:, 2] = raw.cfg.site_height * 0.5
    confident = None
    if hasattr(raw, "believed_site_xy"):  # Team1 variants: the asset's position is unknown, use the estimate
        site[:, :2] = raw.believed_site_xy()
        confident = raw.site_confidence() > SITE_CONFIDENT  # otherwise pursuit-only, as on the real drones
    v, _, _ = hybrid(def_pos, def_vel, att_pos - origin[:, None, :], att_vel, raw._attacker_alive, site, confident)
    zeros = torch.zeros(raw.num_envs, 1, device=raw.device)
    scale = raw.cfg.command.max_lin_speed
    return {a: torch.cat([v[:, i] / scale, zeros], dim=-1) for i, a in enumerate(raw.cfg.possible_agents)}


def chase_actions(raw, obs):
    """Copied from scripts/evaluate.py: head straight at the nearest tracked attacker."""
    actions = {}
    num_attackers = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * num_attackers].reshape(raw.num_envs, num_attackers, 3)
        distance = torch.linalg.norm(rel, dim=-1)
        target = rel[torch.arange(raw.num_envs, device=raw.device), distance.argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def chase_live_actions(raw, obs):
    """chase, but only at live attackers.

    evaluate.py's chase takes the nearest slot, and a downed attacker's slot is all zeros
    (distance 0), so after the first capture of a wave both defenders aim at nothing and
    stop. This uses the alive flags the observation already carries; with none alive, hover.
    """
    actions = {}
    num_attackers = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        o = obs[agent]
        rel = o[:, 9 : 9 + 3 * num_attackers].reshape(raw.num_envs, num_attackers, 3)
        alive = o[:, 9 + 6 * num_attackers : 9 + 7 * num_attackers] > 0.5
        distance = torch.where(alive, torch.linalg.norm(rel, dim=-1), torch.full_like(alive, math.inf, dtype=rel.dtype))
        target = rel[torch.arange(raw.num_envs, device=raw.device), distance.argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        direction = direction * alive.any(dim=-1, keepdim=True).float()
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def pursuit_actions(raw, obs, lead: bool, contact_offset: float, speed: float):
    """Decentralised pursuit from each defender's own 23 observations only, so a student can copy it.

    Target: the live attacker it can reach soonest (``lead``: on a collision course using the
    observed relative velocity) or the nearest live one. ``contact_offset``: aim that far beside
    the attacker so the capture (< capture_radius) triggers before the bodies touch.
    """
    from action_space_kit.control.defense import intercept_time

    actions = {}
    n = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        o = obs[agent]
        own_vel = o[:, 3:6]
        rel = o[:, 9 : 9 + 3 * n].reshape(raw.num_envs, n, 3)
        att_vel = o[:, 9 + 3 * n : 9 + 6 * n].reshape(raw.num_envs, n, 3) + own_vel[:, None, :]
        alive = o[:, 9 + 6 * n : 9 + 7 * n] > 0.5
        if lead:
            t = intercept_time(rel, att_vel, speed).clamp(max=6.0)
            t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
            aims = rel + att_vel * t.unsqueeze(-1)
            key = torch.where(alive, intercept_time(rel, att_vel, speed), torch.full_like(t, math.inf))
            key = torch.where(torch.isfinite(key), key, torch.linalg.norm(rel, dim=-1) + 1e3)  # unreachable: last
            key = torch.where(alive, key, torch.full_like(key, math.inf))
        else:
            aims = rel
            key = torch.where(alive, torch.linalg.norm(rel, dim=-1), torch.full_like(alive, math.inf, dtype=rel.dtype))
        pick = key.argmin(dim=-1)
        aim = aims[torch.arange(raw.num_envs, device=raw.device), pick]
        close = torch.linalg.norm(rel[torch.arange(raw.num_envs, device=raw.device), pick], dim=-1, keepdim=True) < 0.7
        aim = torch.where(close, rel[torch.arange(raw.num_envs, device=raw.device), pick], aim)  # terminal: home in
        if contact_offset > 0.0:
            side = torch.stack([-aim[:, 1], aim[:, 0], torch.zeros_like(aim[:, 0])], dim=-1)
            aim = aim + side / torch.linalg.norm(side, dim=-1, keepdim=True).clamp(min=1e-6) * contact_offset
        direction = aim / torch.clamp(torch.linalg.norm(aim, dim=-1, keepdim=True), min=1e-6)
        direction = direction * alive.any(dim=-1, keepdim=True).float()
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def assigned_actions(raw, obs, contact_offset: float, speed: float, terminal: float = 0.7, soft_closing: float | None = None, block: float | None = None, box_speed: bool = False, urgent_eta: float | None = None, urgent_closing: float = 0.5, escort: tuple | None = None, cruise_up: float = 0.0, helper_up: float = 0.0, pair: str = "slot", pair_margin: float = 0.0, guard: tuple | None = None, keeper: tuple | None = None):
    """Teamwork from each defender's own 23 observations only, so a student can copy it.

    No defender sees its teammate, so they coordinate by convention instead:
      - pairing: defender i takes attacker slot i while it is alive (they never converge on the
        same attacker, the usual way two defenders collide); once its attacker is down it helps
        on the other live one.
      - sides: defender i passes ``contact_offset`` beside its attacker, defender 0 on the
        attacker's left and defender 1 on its right (relative to the attacker's heading), so
        even two defenders on one attacker arrive ~2 x contact_offset apart, and neither touches
        it (capture triggers within capture_radius before the bodies meet).
      - lead: aim at the predicted intercept point at full speed; within ``terminal`` home on it.
      - soft contact (``soft_closing`` m/s): within ``terminal``, fly the attacker's own velocity
        plus a slow closing speed, so contact is a gentle bump and the defender stays airborne
        (a hard hit tips it over, and in this sim a tilted drone can't recover).
      - block (``block`` m): if the defender can't reach the intercept point before the attacker
        reaches the asset (a fast attacker it is behind), fly to where the attacker's straight line
        to the asset crosses ``block`` m from it instead, and meet it there. The asset position is
        in the observation (own position relative to it), so this stays copyable.
      - box speed (``box_speed``): the action space is a per-axis box, and the hosts' controller scales
        each axis by max_lin_speed, so outside the terminal range the direction is stretched until its
        largest component is 1 (up to sqrt(2) faster horizontally) instead of being a unit vector.
      - urgency (``urgent_eta`` s): a soft approach can shadow an attacker for seconds while it keeps
        flying at the asset, so once the attacker is within ``urgent_eta`` of breaching, close at
        ``urgent_closing`` instead (a firmer bump beats a breach).
      - escort (``escort`` = (up, ahead, engage_eta)): once its own attacker is down, a defender doesn't
        join its teammate on the other attacker (76% of crashes involve the teammate); it flies ``up`` m
        above that attacker and ``ahead`` m in front of it toward the asset, and only commits to contact
        when the attacker is within ``engage_eta`` s of breaching (its teammate may be down).
      - cruise (``cruise_up`` m): outside the terminal range aim this far above the attacker, so passes
        tend to go over downed attackers hovering where they were caught; the soft contact itself is level.
      - helper lane (``helper_up`` m): once its own attacker is down, a defender works ``helper_up`` above
        attacker level: it lifts off the attacker it just stopped (which hovers where it was caught) and
        joins its teammate on the other attacker from above rather than level with it (at flip onset, 48%
        of crashes are teammate contacts and 28% a just-downed attacker). Capture still triggers within
        capture_radius, so the lane must stay below it.
      - pairing (``pair``): ``slot`` is the convention above. ``rescue`` keeps it while the defender can
        reach its own attacker before that one could breach (intercept time + ``pair_margin`` s <= time to
        breach); otherwise it takes the other live attacker it can still reach in time. With close spawns a
        crossed pairing (each defender on the far side from its own attacker) then becomes a swap, which both
        defenders find on their own. ``nearest`` always takes the soonest-interceptable live attacker.
      - guard (``guard`` = (radius, engage_eta)): once its own attacker is down, a defender doesn't fly out to
        help; it waits on the live attacker's line to the asset, ``radius`` m from it at the attacker's height,
        and engages only when that attacker is within ``engage_eta`` s of breaching. With close spawns the next
        wave then finds one defender near the asset instead of both far out.
      - goalkeeper (``keeper`` = (radius, engage_eta)): with more defenders than attackers, each defender past
        the attacker count (defender 2 in 3 vs 2) has no slot. It always holds a post ``radius`` m from the asset
        on the line of the most urgent live attacker, at its height, and engages (lead, soft contact) only when
        that attacker is within ``engage_eta`` s of breaching. Without ``keeper`` it would double attacker i % n.
    """
    from action_space_kit.control.defense import intercept_time

    actions = {}
    n = raw.cfg.num_attackers
    rows = torch.arange(raw.num_envs, device=raw.device)
    up = torch.tensor([0.0, 0.0, 1.0], device=raw.device)
    for i, agent in enumerate(raw.cfg.possible_agents):
        o = obs[agent]
        own_vel = o[:, 3:6]
        rel = o[:, 9 : 9 + 3 * n].reshape(raw.num_envs, n, 3)
        att_vel = o[:, 9 + 3 * n : 9 + 6 * n].reshape(raw.num_envs, n, 3) + own_vel[:, None, :]
        alive = o[:, 9 + 6 * n : 9 + 7 * n] > 0.5
        mine = i % n
        is_keeper = keeper is not None and i >= n
        # own attacker if alive, else the live attacker it can reach soonest
        t_all = intercept_time(rel, att_vel, speed)
        key = torch.where(torch.isfinite(t_all), t_all, torch.linalg.norm(rel, dim=-1) + 1e3)
        key = torch.where(alive, key, torch.full_like(key, math.inf))
        pick = torch.where(alive[:, mine], torch.full_like(rows, mine), key.argmin(dim=-1))
        if pair != "slot":
            from action_space_kit.control.defense import time_to_breach

            att_site_all = o[:, None, 0:3] + rel  # every attacker relative to the asset
            eta_all = time_to_breach(att_site_all, att_vel, torch.zeros_like(o[:, 0:3]), raw.cfg.site_radius)
            if pair == "nearest":
                pick = key.argmin(dim=-1)
            else:  # rescue
                reach = alive & (key + pair_margin <= eta_all)
                first_reachable = torch.where(reach, key, torch.full_like(key, math.inf)).argmin(dim=-1)
                pick = torch.where(reach[:, mine], torch.full_like(rows, mine), torch.where(reach.any(dim=-1), first_reachable, pick))
        if is_keeper:
            from action_space_kit.control.defense import time_to_breach

            att_site_all = o[:, None, 0:3] + rel
            eta_all = time_to_breach(att_site_all, att_vel, torch.zeros_like(o[:, 0:3]), raw.cfg.site_radius)
            pick = torch.where(alive, eta_all, torch.full_like(eta_all, math.inf)).argmin(dim=-1)  # the most urgent live attacker
        r = rel[rows, pick]
        v = att_vel[rows, pick]
        if helper_up > 0.0:
            # helping (own attacker down): aim at a point helper_up above the attacker, in every phase
            r = r + up * helper_up * (~alive[:, mine]).float().unsqueeze(-1)
        t = t_all[rows, pick].clamp(max=6.0)
        t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
        aim = r + v * t.unsqueeze(-1)
        if block is not None:
            from action_space_kit.control.defense import time_to_breach

            own_site = o[:, 0:3]  # own position relative to the asset
            att_site = own_site + r  # the attacker's, relative to the asset
            eta = time_to_breach(att_site[:, None, :], v[:, None, :], torch.zeros_like(own_site), raw.cfg.site_radius)[:, 0]
            late = ~torch.isfinite(t_all[rows, pick]) | (t_all[rows, pick] > eta)
            flat = att_site[:, :2]
            far = torch.linalg.norm(flat, dim=-1, keepdim=True).clamp(min=1e-6)
            gate = torch.cat([flat / far * torch.clamp(far, max=block), att_site[:, 2:3]], dim=-1)  # on its line, block m out
            aim = torch.where(late[:, None], gate - own_site, aim)
        if cruise_up > 0.0:
            aim = aim + up * cruise_up
        if escort is not None:
            from action_space_kit.control.defense import time_to_breach

            e_up, e_ahead, e_eta = escort
            att_site = o[:, 0:3] + r
            eta = time_to_breach(att_site[:, None, :], v[:, None, :], torch.zeros_like(att_site), raw.cfg.site_radius)[:, 0]
            to_site = -att_site / torch.linalg.norm(att_site, dim=-1, keepdim=True).clamp(min=1e-6)
            station = r + to_site * e_ahead + up * e_up  # above and in front of the teammate's attacker
            escorting = ~alive[:, mine] & alive.any(dim=-1) & (eta > e_eta)
            aim = torch.where(escorting[:, None], station, aim)
        else:
            escorting = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        post_cfg = keeper if is_keeper else guard  # the keeper's post, or the free defender's guard post
        # if guard is not None:
        if post_cfg is not None:
            from action_space_kit.control.defense import guard_point, time_to_breach

            # g_radius, g_eta = guard
            g_radius, g_eta = post_cfg
            own_site = o[:, 0:3]
            att_site = own_site + r
            eta = time_to_breach(att_site[:, None, :], v[:, None, :], torch.zeros_like(own_site), raw.cfg.site_radius)[:, 0]
            post = guard_point(att_site[:, None, :], torch.zeros_like(own_site), g_radius)[:, 0]
            post = torch.cat([post[:, :2], att_site[:, 2:3]], dim=-1)  # at the attacker's height
            # guarding = ~alive[:, mine] & alive.any(dim=-1) & (eta > g_eta)
            free = alive.any(dim=-1) if is_keeper else (~alive[:, mine] & alive.any(dim=-1))  # a keeper is always free
            guarding = free & (eta > g_eta)
            aim = torch.where(guarding[:, None], post - own_site, aim)
            escorting = escorting | guarding  # no terminal home-in or soft contact while on the post
        aim = torch.where((torch.linalg.norm(r, dim=-1, keepdim=True) < terminal) & ~escorting[:, None], r, aim)
        if contact_offset > 0.0:
            heading = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)
            slow = torch.linalg.norm(v[:, :2], dim=-1, keepdim=True) < 0.05  # hovering: use the line of sight
            heading = torch.where(slow, r / torch.linalg.norm(r, dim=-1, keepdim=True).clamp(min=1e-6), heading)
            left = torch.linalg.cross(up.expand_as(heading), heading, dim=-1)
            left = left / torch.linalg.norm(left, dim=-1, keepdim=True).clamp(min=1e-6)
            aim = aim + left * (contact_offset if i % 2 == 0 else -contact_offset)
        direction = aim / torch.clamp(torch.linalg.norm(aim, dim=-1, keepdim=True), min=1e-6)
        # if guard is not None:
        if post_cfg is not None:
            # settle on the post (proportional inside 0.5 m) instead of circling it at full speed
            direction = torch.where(guarding[:, None], aim / torch.clamp(torch.linalg.norm(aim, dim=-1, keepdim=True), min=0.5), direction)
        if box_speed:
            direction = direction / direction.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        if soft_closing is not None:
            # terminal phase: match the attacker's velocity and close slowly (velocity / max speed = action)
            closing = torch.full_like(r[:, :1], soft_closing)
            if urgent_eta is not None:
                from action_space_kit.control.defense import time_to_breach

                att_site = o[:, 0:3] + r  # the attacker relative to the asset
                eta = time_to_breach(att_site[:, None, :], v[:, None, :], torch.zeros_like(att_site), raw.cfg.site_radius)[:, 0]
                closing = torch.where((eta < urgent_eta)[:, None], torch.full_like(closing, urgent_closing), closing)
            matched = (v + r / torch.linalg.norm(r, dim=-1, keepdim=True).clamp(min=1e-6) * closing) / raw.cfg.command.max_lin_speed
            matched = matched / torch.linalg.norm(matched, dim=-1, keepdim=True).clamp(min=1.0)
            direction = torch.where((torch.linalg.norm(r, dim=-1, keepdim=True) < terminal) & ~escorting[:, None], matched, direction)
        direction = direction * alive.any(dim=-1, keepdim=True).float()
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def check(raw, pilot: str, speed: float) -> None:
    """Fly a scripted pilot and count outcomes exactly as scripts/evaluate.py does."""
    hybrid = HybridDefender(HybridDefenseCfg(max_speed=speed, site_radius=raw.cfg.site_radius, contact_offset=args_cli.contact_offset, obstacle_radius=args_cli.obstacle_radius))
    obs, _ = raw.reset()
    episodes = breaches = 0
    intercepted = attacker_slots = 0
    intercept_times: list[float] = []
    alive_prev = raw._attacker_alive.clone()
    step_in_episode = torch.zeros(raw.num_envs, device=raw.device)
    episode_intercept_times: list[list[float]] = [[] for _ in range(raw.num_envs)]
    # command and flight constraints, hybrid only
    bad = {"outside [-1,1]": 0, "yaw rate != 0": 0, "over speed cap": 0, "non-finite": 0}
    commands = 0
    min_gap = torch.full((raw.num_envs,), math.inf, device=raw.device)
    episode_min_gaps: list[float] = []
    lowest = math.inf
    site_err_sum, site_err_n = 0.0, 0
    floor_steps, all_steps = 0.0, 0

    while episodes < args_cli.episodes:
        if pilot == "chase":
            actions = chase_actions(raw, obs)
        elif pilot == "chase_live":
            actions = chase_live_actions(raw, obs)
        elif pilot == "pursuit":
            actions = pursuit_actions(raw, obs, args_cli.lead, args_cli.contact_offset, speed)
        elif pilot == "assigned":
            # actions = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up)
            # actions = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up, args_cli.pair, args_cli.pair_margin)
            # actions = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up, args_cli.pair, args_cli.pair_margin, tuple(args_cli.guard) if args_cli.guard else None)
            actions = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up, args_cli.pair, args_cli.pair_margin, tuple(args_cli.guard) if args_cli.guard else None, tuple(args_cli.keeper) if args_cli.keeper else None)
        else:
            actions = teacher_actions(raw, hybrid)
            for a in actions.values():
                commands += a.shape[0]
                bad["non-finite"] += int((~torch.isfinite(a)).any(dim=-1).sum())
                bad["outside [-1,1]"] += int((a.abs() > 1.0 + 1e-6).any(dim=-1).sum())
                bad["yaw rate != 0"] += int((a[:, 3] != 0.0).sum())
                over = torch.linalg.norm(a[:, :3], dim=-1) * raw.cfg.command.max_lin_speed > speed + 1e-4
                bad["over speed cap"] += int(over.sum())

        obs, _, _, _, _ = raw.step(actions)
        step_in_episode += 1
        if hasattr(raw, "believed_site_xy"):
            site_err_sum += float(torch.linalg.norm(raw.believed_site_xy(), dim=-1).mean())
            site_err_n += 1
        pos = torch.stack([d.data.root_pos_w for d in raw._defenders], dim=1)
        lowest = min(lowest, float((pos[..., 2] - raw.scene.env_origins[:, None, 2]).min()))
        floor_steps += float(((pos[..., 2] - raw.scene.env_origins[:, None, 2]) < 0.3).sum())
        all_steps += pos.shape[0] * pos.shape[1]
        if pos.shape[1] > 1:
            gaps = torch.cdist(pos, pos) + torch.eye(pos.shape[1], device=raw.device) * 1e6
            min_gap = torch.minimum(min_gap, gaps.flatten(1).min(dim=-1).values)

        newly_down = alive_prev & ~raw._attacker_alive
        if newly_down.any():
            for row in newly_down.any(dim=-1).nonzero(as_tuple=False).flatten().tolist():
                episode_intercept_times[row].append(step_in_episode[row].item() * raw.step_dt)
        alive_prev = raw._attacker_alive.clone()

        finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        if len(finished):
            episodes += len(finished)
            breaches += int(raw.outcome_breached[finished].sum().item())
            intercepted += int(raw.outcome_captures[finished].sum().item())
            attacker_slots += len(finished) * raw.cfg.num_attackers
            for row in finished.tolist():
                intercept_times += episode_intercept_times[row]
                episode_intercept_times[row] = []
            step_in_episode[finished] = 0.0
            episode_min_gaps += min_gap[finished].tolist()
            min_gap[finished] = math.inf
            hybrid.reset(finished)

    print(f"\n=== check:{pilot} on {args_cli.task} (counted as scripts/evaluate.py; not an official score) ===", flush=True)
    print("episodes:            %d" % episodes, flush=True)
    print("interception rate:   %.1f%% of attackers" % (100.0 * intercepted / max(attacker_slots, 1)), flush=True)
    print("breach rate:         %.1f%% of episodes" % (100.0 * breaches / max(episodes, 1)), flush=True)
    if intercept_times:
        mean_t = sum(intercept_times) / len(intercept_times)
        print("time to intercept:   %.1f s (mean of %d)" % (mean_t, len(intercept_times)), flush=True)
    else:
        print("time to intercept:   n/a (nothing intercepted)", flush=True)
    if episode_min_gaps:
        near = sum(g < 0.3 for g in episode_min_gaps) / len(episode_min_gaps)
        print("teammates < 0.3 m:   %.1f%% of episodes (closest %.2f m)" % (100.0 * near, min(episode_min_gaps)), flush=True)
    print("lowest defender:     %.2f m above the floor" % lowest, flush=True)
    print("defender floor time: %.1f%% of defender-steps below 0.3 m (crashed)" % (100.0 * floor_steps / max(all_steps, 1)), flush=True)
    if site_err_n:
        print("site estimate error: %.2f m mean (asset position inferred from attackers)" % (site_err_sum / site_err_n), flush=True)
    if pilot == "hybrid":
        for name, count in bad.items():
            print("constraint %-15s %s" % (name + ":", "ok" if count == 0 else f"VIOLATED in {count} of {commands} commands"), flush=True)


def fresh_compute(model, inputs, role: str) -> torch.Tensor:
    """A role's output on these inputs.

    skrl's shared policy/value model caches the trunk output of each policy call and reuses
    it on the next value call (``_shared_output``), which would give values for stale inputs.
    """
    if hasattr(model, "_shared_output"):
        model._shared_output = None
    return model.compute(inputs, role=role)[0]


def param_ids(model, inputs, role: str) -> set[int]:
    """Parameters the given role's output depends on (shared trunk + that role's head)."""
    model.zero_grad(set_to_none=True)
    fresh_compute(model, inputs, role).sum().backward()
    ids = {id(p) for p in model.parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0)}
    model.zero_grad(set_to_none=True)
    return ids


def distil(env, raw, speed: float) -> None:
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")
    agent_cfg["seed"] = args_cli.seed
    log_root = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    suffix = "refit" if args_cli.refit else f"distill_{args_cli.teacher}"
    # the pid keeps runs started in the same second (parallel pipelines) in separate folders
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{suffix}_{os.getpid()}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    log_dir = os.path.join(log_root, run_name)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), raw.cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "distill_args.txt"), "w") as f:
        f.write(f"{vars(args_cli)}\nteacher speed {speed}\n")

    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent = Runner(wrapped, agent_cfg).agent
    if args_cli.refit:
        agent.load(args_cli.refit)
        print(f"[distil] refit: loaded {args_cli.refit}; DAgger skipped, observation scaler frozen", flush=True)
    uids = list(raw.cfg.possible_agents)
    policies = {u: agent.policies[u] for u in uids}
    values = {u: agent.values[u] for u in uids}
    scalers = {u: agent._observation_preprocessor[u] for u in uids}

    obs, _ = wrapped.reset()
    probe = {u: {"observations": scalers[u](obs[u][:8]), "states": None} for u in uids}
    policy_ids = {u: param_ids(policies[u], probe[u], "policy") for u in uids}
    value_all_ids = {u: param_ids(values[u], probe[u], "value") for u in uids}  # trunk + value head
    value_ids = {u: value_all_ids[u] - (policy_ids[u] if values[u] is policies[u] else set()) for u in uids}
    for u in uids:
        shared = values[u] is policies[u]
        print(f"[distil] {u}: policy params {len(policy_ids[u])}, value-only params {len(value_ids[u])}, shared model {shared}", flush=True)

    def student(u, o):
        return fresh_compute(policies[u], {"observations": scalers[u](o), "states": None}, "policy")

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir)  # curves next to skrl's own (empty) events file, under "Distill /"
    optim = {u: torch.optim.Adam([p for p in policies[u].parameters() if id(p) in policy_ids[u]], lr=args_cli.lr) for u in uids}
    hybrid = HybridDefender(HybridDefenseCfg(max_speed=speed, site_radius=raw.cfg.site_radius, contact_offset=args_cli.contact_offset, obstacle_radius=args_cli.obstacle_radius))
    data_obs = {u: [] for u in uids}
    data_act = {u: [] for u in uids}

    for rnd in range(0 if args_cli.refit else args_cli.rounds):
        beta = 1.0 if rnd == 0 else 0.5**rnd
        new_obs = {u: [] for u in uids}
        mismatch, captures, breaches, alive_prev = 0.0, 0, 0, raw._attacker_alive.clone()
        for t in range(args_cli.steps):
            if args_cli.teacher == "chase":
                teach = chase_actions(raw, obs)
            elif args_cli.teacher == "chase_live":
                teach = chase_live_actions(raw, obs)
            elif args_cli.teacher == "pursuit":
                teach = pursuit_actions(raw, obs, args_cli.lead, args_cli.contact_offset, speed)
            elif args_cli.teacher == "assigned":
                # teach = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up)
                # teach = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up, args_cli.pair, args_cli.pair_margin)
                # teach = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up, args_cli.pair, args_cli.pair_margin, tuple(args_cli.guard) if args_cli.guard else None)
                teach = assigned_actions(raw, obs, args_cli.contact_offset, speed, args_cli.terminal, args_cli.soft_closing, args_cli.block, args_cli.box_speed, args_cli.urgent_eta, args_cli.urgent_closing, tuple(args_cli.escort) if args_cli.escort else None, args_cli.cruise_up, args_cli.helper_up, args_cli.pair, args_cli.pair_margin, tuple(args_cli.guard) if args_cli.guard else None, tuple(args_cli.keeper) if args_cli.keeper else None)
            else:
                teach = teacher_actions(raw, hybrid)
            if beta < 1.0:
                with torch.no_grad():
                    stud = {u: student(u, obs[u]) for u in uids}
                use_teacher = torch.rand(raw.num_envs, 1, device=raw.device) < beta
                actions = {u: torch.where(use_teacher, teach[u], stud[u]) for u in uids}
                mismatch += sum(float(((stud[u] - teach[u]) ** 2).mean()) for u in uids) / len(uids)
            else:
                actions = teach
            if t % args_cli.keep_every == 0:
                for u in uids:
                    new_obs[u].append(obs[u].clone())
                    data_act[u].append(teach[u].clone())
            obs, _, terminated, truncated, _ = wrapped.step(actions)
            done = (terminated[uids[0]] | truncated[uids[0]]).view(-1)
            breaches += int(terminated[uids[0]].view(-1).sum())
            captures += int((alive_prev & ~raw._attacker_alive).sum())
            alive_prev = raw._attacker_alive.clone()
            if done.any():
                hybrid.reset(done.nonzero(as_tuple=False).flatten())

        losses = []
        for u in uids:
            fresh = torch.cat(new_obs[u])
            scalers[u](fresh, train=True)  # update the running mean/std the checkpoint carries
            data_obs[u].append(fresh)
            x_all, y_all = torch.cat(data_obs[u]), torch.cat(data_act[u])
            for _ in range(args_cli.grad_steps):
                idx = torch.randint(0, x_all.shape[0], (args_cli.batch,), device=raw.device)
                loss = torch.nn.functional.mse_loss(student(u, x_all[idx]), y_all[idx])
                optim[u].zero_grad(set_to_none=True)
                loss.backward()
                optim[u].step()
            losses.append(float(loss))
        seconds = args_cli.steps * raw.step_dt * raw.num_envs
        writer.add_scalar("Distill / student-vs-teacher action mse", mismatch / args_cli.steps, rnd)
        writer.add_scalar("Distill / fit mse", sum(losses) / len(losses), rnd)
        writer.add_scalar("Distill / teacher share (beta)", beta, rnd)
        writer.add_scalar("Distill / captures per env-minute", 60 * captures / seconds, rnd)
        writer.add_scalar("Distill / breaches per env-minute", 60 * breaches / seconds, rnd)
        print(
            f"[distil] round {rnd:2d}  beta {beta:.3f}  student-vs-teacher mse {mismatch / args_cli.steps:.4f}"
            f"  fit mse {sum(losses) / len(losses):.4f}  samples {x_all.shape[0]}"
            f"  captures/min {60 * captures / seconds:.2f}  breaches/min {60 * breaches / seconds:.3f}",
            flush=True,
        )

    # exploration noise for RL fine-tuning: small, so PPO starts near the teacher
    for u in uids:
        for name, p in policies[u].named_parameters():
            if "log_std" in name:
                with torch.no_grad():
                    p.fill_(args_cli.log_std)

    # value function: discounted returns of the (stochastic) student, as PPO will see them
    gamma = args_cli.gamma or agent_cfg["agent"]["discount_factor"]
    print(f"[distil] value targets: gamma {gamma}, {args_cli.value_steps}-step rollout, first {args_cli.value_fit} steps fitted", flush=True)
    roll_obs = {u: [] for u in uids}
    roll_rew = {u: [] for u in uids}
    roll_done = []
    for _ in range(args_cli.value_steps):
        with torch.no_grad():
            actions = {
                u: policies[u].act({"observations": scalers[u](obs[u]), "states": None}, role="policy")[0] for u in uids
            }
        for u in uids:
            roll_obs[u].append(obs[u].clone())
        obs, rewards, terminated, truncated, _ = wrapped.step(actions)
        for u in uids:
            roll_rew[u].append(rewards[u].view(-1).clone())
        roll_done.append((terminated[uids[0]] | truncated[uids[0]]).view(-1).float())
    for u in uids:
        rew = torch.stack(roll_rew[u])
        done = torch.stack(roll_done)
        ret = torch.zeros_like(rew)
        running = torch.zeros_like(rew[0])
        for t in reversed(range(rew.shape[0])):
            running = rew[t] + gamma * (1.0 - done[t]) * running
            ret[t] = running
        x_all = torch.stack(roll_obs[u])[: args_cli.value_fit].reshape(-1, obs[u].shape[-1])
        r_all = ret[: args_cli.value_fit].reshape(-1, 1)
        vpre = agent._value_preprocessor[u]
        vpre(r_all, train=True)
        target = vpre(r_all)
        # A value head alone on behaviour-cloned features is a linear probe (explained variance ~0.2-0.5).
        # With --preserve_policy the shared trunk learns too, penalised for moving the policy's actions.
        through_trunk = args_cli.preserve_policy > 0 and values[u] is policies[u]
        ids = value_all_ids[u] if through_trunk else value_ids[u]
        params = [p for p in values[u].parameters() if id(p) in ids]
        if not params:
            print(f"[distil] {u}: no value parameters; skipping value fit", flush=True)
            continue
        with torch.no_grad():
            old_mean = torch.cat([student(u, x_all[i : i + 65536]) for i in range(0, x_all.shape[0], 65536)])
        opt = torch.optim.Adam(params, lr=1e-3)
        for _ in range(args_cli.value_grad_steps):
            idx = torch.randint(0, x_all.shape[0], (args_cli.batch,), device=raw.device)
            pred = fresh_compute(values[u], {"observations": scalers[u](x_all[idx]), "states": None}, "value")
            loss = torch.nn.functional.mse_loss(pred, target[idx])
            if through_trunk:
                loss = loss + args_cli.preserve_policy * torch.nn.functional.mse_loss(student(u, x_all[idx]), old_mean[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = torch.cat(
                [fresh_compute(values[u], {"observations": scalers[u](x_all[i : i + 65536]), "states": None}, "value") for i in range(0, x_all.shape[0], 65536)]
            )
            explained = 1.0 - float(torch.var(target - pred) / torch.var(target).clamp(min=1e-8))
            drift = float(torch.cat([student(u, x_all[i : i + 65536]) for i in range(0, x_all.shape[0], 65536)]).sub(old_mean).pow(2).mean())
        print(
            f"[distil] {u}: value fit mse {float(loss):.4f}, explained variance {explained:.3f}, "
            f"policy drift {drift:.2e} ({'trunk + head' if through_trunk else 'head only'})",
            flush=True,
        )
        writer.add_scalar(f"Distill / policy drift from value fit ({u})", drift, args_cli.rounds)
        writer.add_scalar(f"Distill / value explained variance ({u})", explained, args_cli.rounds)
    writer.close()

    path = os.path.join(log_dir, "checkpoints", "agent_distill.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    agent.save(path)
    if not args_cli.keep_optimizer:
        # skrl's load() restores only the entries a file has, so without stale optimizer moments
        # the fine-tune starts its own optimizer at the learning rate it was given
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for u in saved:
            saved[u].pop("optimizer", None)
        torch.save(saved, path)
    print(f"[distil] saved {path}", flush=True)
    print(f"[distil] score it: python -u scripts/evaluate.py --task {args_cli.task} --checkpoint {path}", flush=True)


def main():
    num_envs = args_cli.num_envs or (64 if args_cli.check else 4096)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    max_speed = raw.cfg.command.max_lin_speed
    speed = min(args_cli.speed_cap, max_speed) if args_cli.speed_cap else max_speed
    if args_cli.check:
        check(raw, args_cli.check, speed)
    else:
        distil(env, raw, speed)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
