# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Parity check for a flight defend.py (deliverable/defend_*/defend.py) against the sim it was trained in.

Flies the skrl agent (loaded as scripts/evaluate.py loads it) in the 3v2 task and, every step, checks:
  1. defend.py's build_obs(), fed the sim's own states (world frame = room frame, asset = env origin + SITE_Z),
     reproduces the env's 23-number observation for every defender;
  2. defend.py's act() on that observation reproduces skrl's mean action (clipped to [-1, 1]), from the
     checkpoint file defend.py ships (defend_checkpoint.pt or .npz).

    python -u scripts/defend_parity.py --defend ../deliverable/defend_soft/defend.py --checkpoint policies/team1_defend_3v2_soft/agent_10.pt
"""

import argparse
import importlib.util

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="defend.py vs sim parity.")
parser.add_argument("--task", type=str, default="Team1-Defend-3v2-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--defend", type=str, required=True, help="path to the flight defend.py")
parser.add_argument("--checkpoint", type=str, required=True, help="the skrl checkpoint the sim agent flies")
parser.add_argument("--weights", type=str, default=None, help="checkpoint defend.py loads (default: its own CHECKPOINT)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import action_space_kit.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    spec = importlib.util.spec_from_file_location("flight_defend", args_cli.defend)
    dfd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dfd)
    w = dfd.load_weights(args_cli.weights or dfd.CHECKPOINT)
    print(f"[parity] defend.py {args_cli.defend}, weights {args_cli.weights or dfd.CHECKPOINT}, agents {w['agents']}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    agent = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")).agent
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval") if hasattr(agent, "set_running_mode") else agent.enable_training_mode(False)
    uids = list(raw.cfg.possible_agents)
    assert uids == list(dfd.AGENTS), (uids, dfd.AGENTS)
    assert abs(raw.cfg.site_height * 0.5 - dfd.SITE_Z) < 1e-9, (raw.cfg.site_height, dfd.SITE_Z)
    assert raw.cfg.obs_clip_pos == dfd.CLIP_POS and raw.cfg.obs_clip_vel == dfd.CLIP_VEL

    obs, _ = env.reset()
    obs_err = act_err = 0.0
    n_dead_slots = 0
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            out = agent.act(obs, {u: None for u in uids}, timestep=0, timesteps=0)
            ref = {u: out[-1][u].get("mean_actions", out[0][u]).clamp(-1.0, 1.0) for u in uids}
        # the sim's own states -> defend.py's observation builder
        site = (raw.scene.env_origins + torch.tensor([0.0, 0.0, dfd.SITE_Z], device=raw.device)).cpu().numpy()
        att_P = torch.stack([a.data.root_pos_w for a in raw._attackers], 1).cpu().numpy()
        att_V = torch.stack([a.data.root_lin_vel_w for a in raw._attackers], 1).cpu().numpy()
        alive = raw._attacker_alive.cpu().numpy()
        n_dead_slots += int((~alive).sum())
        for i, u in enumerate(uids):
            P = raw._defenders[i].data.root_pos_w.cpu().numpy()
            V = raw._defenders[i].data.root_lin_vel_w.cpu().numpy()
            g = raw._defenders[i].data.projected_gravity_b.cpu().numpy()
            env_obs = obs[u].cpu().numpy()
            for e in range(raw.num_envs):
                mine = dfd.build_obs(P[e], V[e], site[e], att_P[e], att_V[e], alive[e])
                mine[6:9] = g[e]  # defend.py assumes level flight (0, 0, -1); compare everything else exactly
                obs_err = max(obs_err, float(np.abs(mine - env_obs[e]).max()))
                act_err = max(act_err, float(np.abs(dfd.act(w, u, env_obs[e].astype(np.float64)) - ref[u][e].cpu().numpy()).max()))
        obs, *_ = env.step(ref)
    tilt = max(float(np.abs(d.data.projected_gravity_b.cpu().numpy() - [0, 0, -1]).max()) for d in raw._defenders)
    print(f"[parity] {args_cli.steps} steps x {raw.num_envs} envs x {len(uids)} defenders; dead attacker slots seen: {n_dead_slots}", flush=True)
    print(f"[parity] max |build_obs - env obs| = {obs_err:.2e} (gravity taken from the sim; level-flight gap now {tilt:.3f})", flush=True)
    print(f"[parity] max |defend.py act - skrl mean action| = {act_err:.2e}", flush=True)
    ok = obs_err < 1e-4 and act_err < 1e-4
    print("PARITY PASSED" if ok else "PARITY FAILED", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
