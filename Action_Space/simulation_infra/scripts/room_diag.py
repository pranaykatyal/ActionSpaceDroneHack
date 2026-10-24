# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Where and why defenders are lost in the room tasks (team1_defend_room_env.py). A diagnostic, not a score.

Flies a checkpoint (loaded as scripts/evaluate.py loads it) or chase, and for every defender the room's
wall check takes out it records: which wall or the ceiling, the room position, whether it was already
tipped past 30 deg (so it was falling or tumbling before it got there), and the time in the episode.
Also counts captures made within 1 m of the attackers' start box (the box every wave respawns in).

    python -u scripts/room_diag.py --task Team1-Defend-3v2-Room-v0 --checkpoint <agent.pt> --episodes 64
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Room diagnostic for a defend checkpoint.")
parser.add_argument("--task", type=str, default="Team1-Defend-3v2-Room-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--checkpoint", type=str, default=None, help="skrl IPPO checkpoint; chase if omitted")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from collections import Counter

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    uids = list(raw.cfg.possible_agents)
    agent = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        agent = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")).agent
        agent.load(args_cli.checkpoint)
        agent.set_running_mode("eval") if hasattr(agent, "set_running_mode") else agent.enable_training_mode(False)

    obs, _ = env.reset()
    raw._ensure_room()
    lo, hi, clear = raw._room_lo, raw._room_hi, raw._rcfg("wall_clearance")
    cx, cy = raw._rcfg("attacker_box_center")
    half = 0.5 * raw._rcfg("attacker_box_size")
    n = raw.cfg.num_attackers
    walls, tipped_first, times, where = Counter(), 0, [], []
    lost = 0
    episodes = 0
    captures = near_box = 0
    alive_prev = raw._attacker_alive.clone()
    crashed_prev = raw._room_crashed.clone()
    while episodes < args_cli.episodes:
        with torch.inference_mode():
            if agent is not None:
                out = agent.act(obs, {u: None for u in uids}, timestep=0, timesteps=0)
                actions = {u: out[-1][u].get("mean_actions", out[0][u]) for u in uids}
            else:
                actions = {}
                for u in uids:
                    rel = obs[u][:, 9 : 9 + 3 * n].reshape(raw.num_envs, n, 3)
                    target = rel[torch.arange(raw.num_envs, device=raw.device), torch.linalg.norm(rel, dim=-1).argmin(dim=-1)]
                    direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
                    actions[u] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
        # where every defender is right before the step that may take it out
        before = [d.data.root_pos_w - raw.scene.env_origins + raw._room_site for d in raw._defenders]
        tilt = [d.data.projected_gravity_b[:, 2] > -0.866 for d in raw._defenders]
        t_ep = raw.episode_length_buf.float() * raw.step_dt
        obs, *_ = env.step(actions)
        new = raw._room_crashed & ~crashed_prev
        for i in range(len(raw._defenders)):
            for e in new[:, i].nonzero(as_tuple=False).flatten().tolist():
                p = before[i][e]
                lost += 1
                tipped_first += int(tilt[i][e])
                times.append(float(t_ep[e]))
                where.append([round(float(v), 2) for v in p])
                if p[2] > hi[2] - clear - 0.1:
                    walls["ceiling"] += 1
                elif p[0] < lo[0] + clear + 0.1:
                    walls["tag-10 wall (behind the asset)"] += 1
                elif p[0] > hi[0] - clear - 0.1:
                    walls["far wall"] += 1
                elif p[1] < lo[1] + clear + 0.1:
                    walls["left wall"] += 1
                elif p[1] > hi[1] - clear - 0.1:
                    walls["right wall"] += 1
                else:
                    walls["inside (moved > 0.1 m in one step)"] += 1
        crashed_prev = raw._room_crashed.clone()
        # captures near the attackers' start box (the box every wave respawns in)
        downed = alive_prev & ~raw._attacker_alive
        if downed.any():
            pos = torch.stack([a.data.root_pos_w for a in raw._attackers], 1) - raw.scene.env_origins[:, None] + raw._room_site
            box = ((pos[..., 0] - cx).abs() < half + 1.0) & ((pos[..., 1] - cy).abs() < half + 1.0)
            captures += int(downed.sum())
            near_box += int((downed & box).sum())
        alive_prev = raw._attacker_alive.clone()
        fin = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        episodes += len(fin)
    print(f"=== {args_cli.checkpoint or 'chase'} on {args_cli.task}: {episodes} episodes ===", flush=True)
    print(f"defenders lost: {lost} ({lost / max(episodes, 1):.2f} per episode); already tipped > 30 deg: "
          f"{tipped_first} ({tipped_first / max(lost, 1):.0%})", flush=True)
    for k, v in walls.most_common():
        print(f"  {k}: {v} ({v / max(lost, 1):.0%})", flush=True)
    if times:
        times.sort()
        print(f"  time in episode: median {times[len(times) // 2]:.1f} s, first 1 s: "
              f"{sum(t < 1.0 for t in times)}", flush=True)
        print(f"  first positions (room X, Y, Z): {where[:8]}", flush=True)
    print(f"captures within 1 m of the attackers' start box: {near_box} of {captures} "
          f"({near_box / max(captures, 1):.0%})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
