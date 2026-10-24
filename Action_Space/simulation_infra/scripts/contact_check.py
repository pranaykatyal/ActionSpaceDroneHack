# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Do our defenders actually touch the attackers they "capture"? A diagnostic, not an official score.

The hosts' capture fires at ``capture_radius`` (0.35 m centre to centre); a Tello's frame is 9.8 x 9.25 cm,
so frames or propeller guards touch only at about 0.10-0.12 m. For every capture this records the capturing
defender's distance and closing speed at that step, how close it gets to the downed attacker in the next
second, and whether it tips over (> 30 deg) in that second. It also reports any frame contact (< 0.12 m)
between a defender and a live attacker. Loads the checkpoint exactly as scripts/evaluate.py does.

    python -u scripts/contact_check.py --task Team1-Defend-3v2-v0 --checkpoint <agent.pt> --episodes 256
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Contact check for a defend checkpoint.")
parser.add_argument("--task", type=str, default="AS-Defend-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--pilot", type=str, default=None, choices=["chase"], help="Fly the hosts' chase baseline instead.")
parser.add_argument("--touch", type=float, default=0.12, help="Centre distance at which Tello frames touch [m].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def chase_actions(raw, obs):
    """As scripts/evaluate.py: head straight at the nearest tracked attacker."""
    actions = {}
    n = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * n].reshape(raw.num_envs, n, 3)
        target = rel[torch.arange(raw.num_envs, device=raw.device), torch.linalg.norm(rel, dim=-1).argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    raw = env
    uids = list(raw.cfg.possible_agents)
    agent = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        agent = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")).agent
        agent.load(args_cli.checkpoint)
        agent.set_running_mode("eval") if hasattr(agent, "set_running_mode") else agent.enable_training_mode(False)
    n, dev = raw.num_envs, raw.device
    follow = int(1.0 / raw.step_dt)  # one second after each capture
    # open follow-ups: (env, defender, attacker, steps left, min distance, max tilt)
    open_rows: list[list] = []
    cap_dist, cap_closing, after_min, after_tilt = [], [], [], []
    live_touch_steps = 0
    defender_steps = 0
    episodes = 0
    obs, _ = env.reset()
    while episodes < args_cli.episodes:
        alive_before = raw._attacker_alive.clone()
        with torch.inference_mode():
            if agent is not None:
                out = agent.act(obs, {u: None for u in uids}, timestep=0, timesteps=0)
                actions = {u: out[-1][u].get("mean_actions", out[0][u]) for u in uids}
            else:
                actions = chase_actions(raw, obs)
        obs, *_ = env.step(actions)
        d_pos = torch.stack([d.data.root_pos_w for d in raw._defenders], 1)
        d_vel = torch.stack([d.data.root_lin_vel_w for d in raw._defenders], 1)
        a_pos, a_vel = raw._attacker_states()
        gap = torch.cdist(d_pos, a_pos)  # (n, D, A)
        tilt = torch.stack([torch.rad2deg(torch.acos((-d.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))) for d in raw._defenders], 1)
        alive_now = raw._attacker_alive
        live = alive_before & alive_now
        live_touch_steps += int(((gap < args_cli.touch) & live[:, None, :]).any(dim=-1).sum())
        defender_steps += n * len(uids)
        # follow-ups of earlier captures (a reset ends them)
        reset = raw.episode_length_buf == 0
        keep = []
        for row in open_rows:
            e, d, a, left, mn, mt = row
            if bool(reset[e]):
                after_min.append(mn)
                after_tilt.append(mt)
                continue
            row[4] = min(mn, float(gap[e, d, a]))
            row[5] = max(mt, float(tilt[e, d]))
            row[3] = left - 1
            if row[3] <= 0:
                after_min.append(row[4])
                after_tilt.append(row[5])
            else:
                keep.append(row)
        open_rows = keep
        # new captures this step: the attacker went down without a reset
        downed = alive_before & ~alive_now & ~reset[:, None]
        for e, a in downed.nonzero(as_tuple=False).tolist():
            d = int(gap[e, :, a].argmin())
            r = a_pos[e, a] - d_pos[e, d]
            closing = float(torch.dot(d_vel[e, d] - a_vel[e, a], r / torch.linalg.norm(r).clamp(min=1e-6)))
            cap_dist.append(float(gap[e, d, a]))
            cap_closing.append(closing)
            open_rows.append([e, d, a, follow, float(gap[e, d, a]), float(tilt[e, d])])
        episodes += int(raw.outcome_valid.sum())

    label = args_cli.checkpoint or f"baseline:{args_cli.pilot}"
    total = len(cap_dist)
    print(f"\n=== contact check: {label} on {args_cli.task} (diagnostic) ===", flush=True)
    print(f"episodes {episodes}, captures {total}, capture radius {raw.cfg.capture_radius} m, frames touch at < {args_cli.touch} m", flush=True)
    if total:
        touched = sum(x < args_cli.touch for x in after_min)
        tipped = sum(t > 30.0 for t in after_tilt)
        print(f"distance at capture:  median {q(cap_dist, 0.5):.3f} m (p10 {q(cap_dist, 0.1):.3f}, p90 {q(cap_dist, 0.9):.3f})", flush=True)
        print(f"closing speed at capture: median {q(cap_closing, 0.5):+.2f} m/s (p10 {q(cap_closing, 0.1):+.2f}, p90 {q(cap_closing, 0.9):+.2f})", flush=True)
        print(f"within 1 s after: closest to the downed attacker median {q(after_min, 0.5):.3f} m; "
              f"frames touched in {100 * touched / max(len(after_min), 1):.0f}% of captures; "
              f"capturing defender tipped > 30 deg in {100 * tipped / max(len(after_tilt), 1):.0f}%", flush=True)
    print(f"frame contact with a live attacker: {100 * live_touch_steps / max(defender_steps, 1):.2f}% of defender-steps", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
