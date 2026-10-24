# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""What the deploy-side layer (team1_defend_deploy_env.py) does in flight. A diagnostic, not a score.

Flies a checkpoint (loaded as scripts/evaluate.py loads it) and reports how often the separation
filter changes a command and by how much, how close the defenders get, and how often a defender
flips (tilt past 30 deg) or falls below 0.3 m, per episode.

    python -u scripts/deploy_diag.py --task Team1-Defend-RR-Sep-v0 --checkpoint <agent.pt> --episodes 64
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy-layer diagnostic for a defend checkpoint.")
parser.add_argument("--task", type=str, default="Team1-Defend-RR-Sep-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    agent = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")).agent
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval") if hasattr(agent, "set_running_mode") else agent.enable_training_mode(False)
    uids = list(raw.cfg.possible_agents)
    n, dev, speed = raw.num_envs, raw.device, raw.cfg.command.max_lin_speed

    obs, _ = env.reset()
    steps = active = 0
    corr_sum = torch.zeros((), device=dev)
    near03 = near05 = 0
    flipped = torch.zeros(n, len(uids), dtype=torch.bool, device=dev)
    fell = torch.zeros_like(flipped)
    flips = falls = episodes = breaches = 0
    while episodes < args_cli.episodes:
        with torch.inference_mode():
            out = agent.act(obs, {u: None for u in uids}, timestep=0, timesteps=0)
            actions = {u: out[-1][u].get("mean_actions", out[0][u]) for u in uids}
        obs, *_ = env.step(actions)
        # the command the env flew this step (after the filter) against the network's
        sent = torch.stack([raw._actions[u][:, :3] for u in uids], 1)
        net = torch.stack([actions[u][:, :3].clamp(-1.0, 1.0) for u in uids], 1)
        corr = torch.linalg.norm(sent - net, dim=-1) * speed  # [m/s]
        steps += corr.numel()
        active += int((corr > 1e-4).sum())
        corr_sum += corr.sum()
        d_pos = torch.stack([d.data.root_pos_w for d in raw._defenders], 1)
        gap = torch.linalg.norm(d_pos[:, 0] - d_pos[:, 1], dim=-1)
        near03 += int((gap < 0.3).sum())
        near05 += int((gap < 0.5).sum())
        tilt = torch.stack([d.data.projected_gravity_b[:, 2] for d in raw._defenders], 1) > -0.866  # past 30 deg
        low = d_pos[..., 2] - raw.scene.env_origins[:, None, 2] < 0.3
        flips += int((tilt & ~flipped).sum())
        falls += int((low & ~fell).sum())
        flipped |= tilt
        fell |= low
        fin = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        if len(fin):
            episodes += len(fin)
            breaches += int(raw.outcome_breached[fin].sum())
            flipped[fin] = False
            fell[fin] = False
    pairs = steps // len(uids)
    print(f"=== {args_cli.checkpoint} on {args_cli.task}: {episodes} episodes ===", flush=True)
    print(f"breach {breaches / episodes:.1%}; flips {flips / episodes:.2f} / episode; below 0.3 m {falls / episodes:.2f} / episode", flush=True)
    print(f"filter changed {active / steps:.1%} of defender commands, mean {float(corr_sum) / max(active, 1):.3f} m/s when it did", flush=True)
    print(f"defenders closer than 0.3 m: {near03 / pairs:.2%} of steps, 0.5 m: {near05 / pairs:.2%}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
