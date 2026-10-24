# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""How aggressive and jittery is a defend controller's command stream? A diagnostic, not an official score.

Policies trained on task reward alone have crashed real drones with aggressive commands (OPEN,
arXiv 2409.15866), and with the real Tello's ~0.5 s lag a jittery command stream oscillates. This
flies one controller on a task (meant for Team1-Defend-Fitted-v0) and reports, per defender:

  outcome         breach and interception rate counted like scripts/evaluate.py, floor time
  raw change      |a_t - a_t-1| of the controller's own velocity command per 60 Hz step [m/s]
  sent change     change of the command that reached the drone, per rc packet that changed it [m/s];
                  'jumps' = packets changing it by more than 0.5 m/s
  variation       total command change per second [m/s per s]
  reversals       sign flips per second of a horizontal command axis (|v| >= 0.1 m/s on both sides)

    python -u scripts/smoothness_check.py --task Team1-Defend-Fitted-v0 --checkpoint <agent.pt>
    python -u scripts/smoothness_check.py --task Team1-Defend-Fitted-v0 --pilot hybrid --contact_offset 0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Command smoothness of a defend controller.")
parser.add_argument("--task", type=str, default="Team1-Defend-Fitted-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--pilot", type=str, default=None, choices=["chase", "hybrid"])
parser.add_argument("--contact_offset", type=float, default=0.0, help="hybrid: aim this far beside the attacker [m].")
parser.add_argument("--hybrid_state", type=str, default="estimated", choices=["estimated", "true"],
                    help="hybrid: the laptop's Kalman estimates (as in the room) or the true state.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from action_space_kit.control.defense import SITE_CONFIDENT, HybridDefenseCfg, HybridDefender
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def chase_actions(raw, obs):
    """As scripts/evaluate.py."""
    actions = {}
    n = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * n].reshape(raw.num_envs, n, 3)
        target = rel[torch.arange(raw.num_envs, device=raw.device), torch.linalg.norm(rel, dim=-1).argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def hybrid_actions(raw, hybrid, estimated: bool):
    origin = raw.scene.env_origins
    n_def = raw.cfg.num_defenders
    if estimated and hasattr(raw, "_kf"):
        def_pos, def_vel = raw._kf.p[:, :n_def], raw._kf.v[:, :n_def]
        att_pos, att_vel = raw._kf.p[:, n_def:], raw._kf.v[:, n_def:]
    else:
        def_pos = torch.stack([d.data.root_pos_w - origin for d in raw._defenders], dim=1)
        def_vel = torch.stack([d.data.root_lin_vel_w for d in raw._defenders], dim=1)
        att_pos, att_vel = raw._attacker_states()
        att_pos = att_pos - origin[:, None, :]
    site = torch.zeros_like(origin)
    site[:, 2] = raw.cfg.site_height * 0.5
    confident = None
    if hasattr(raw, "believed_site_xy"):
        site[:, :2] = raw.believed_site_xy()
        confident = raw.site_confidence() > SITE_CONFIDENT
    v, _, _ = hybrid(def_pos, def_vel, att_pos, att_vel, raw._attacker_alive, site, confident)
    zeros = torch.zeros(raw.num_envs, 1, device=raw.device)
    scale = raw.cfg.command.max_lin_speed
    return {a: torch.cat([v[:, i] / scale, zeros], dim=-1) for i, a in enumerate(raw.cfg.possible_agents)}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    raw = env
    uids = list(raw.cfg.possible_agents)
    policy = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")
        policy = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), agent_cfg).agent
        policy.load(args_cli.checkpoint)
        policy.set_running_mode("eval") if hasattr(policy, "set_running_mode") else policy.enable_training_mode(False)
    hybrid = HybridDefender(
        HybridDefenseCfg(max_speed=raw.cfg.command.max_lin_speed, site_radius=raw.cfg.site_radius, contact_offset=args_cli.contact_offset)
    )
    speed = raw.cfg.command.max_lin_speed

    obs, _ = env.reset()
    episodes = breaches = 0
    captures = slots = 0.0
    prev_raw = prev_sent = None
    raw_changes, sent_changes = [], []
    variation = 0.0
    reversals = 0
    active_steps = 0  # defender-steps that count (not across a reset)
    floor_steps = 0
    sign_prev = None
    while episodes < args_cli.episodes:
        with torch.inference_mode():
            if policy is not None:
                out = policy.act(obs, {u: None for u in uids}, timestep=0, timesteps=0)
                actions = {u: out[-1][u].get("mean_actions", out[0][u]) for u in uids}
            elif args_cli.pilot == "hybrid":
                actions = hybrid_actions(raw, hybrid, args_cli.hybrid_state == "estimated")
            else:
                actions = chase_actions(raw, obs)
            same_episode = raw.episode_length_buf > 0  # this step continues the episode of the last one
            obs, *_ = env.step(actions)

            cmd = torch.stack([actions[u][:, :3].clamp(-1, 1) for u in uids], dim=1) * speed  # (N, D, 3) m/s
            sent = torch.stack([raw._actions[u][:, :3] for u in uids], dim=1) * speed
            if prev_raw is not None:
                keep = same_episode[:, None].expand(-1, len(uids))
                d_raw = torch.linalg.norm(cmd - prev_raw, dim=-1)[keep]
                d_sent = torch.linalg.norm(sent - prev_sent, dim=-1)[keep]
                raw_changes.append(d_raw)
                sent_changes.append(d_sent[d_sent > 1e-6])
                variation += float(d_sent.sum())
                active_steps += int(keep.sum())
                sign = torch.sign(sent[..., :2]) * (sent[..., :2].abs() >= 0.1)
                if sign_prev is not None:
                    flips = (sign * sign_prev < 0) & keep[..., None]
                    reversals += int(flips.sum())
                sign_prev = torch.where(sign != 0, sign, sign_prev if sign_prev is not None else sign)
            prev_raw, prev_sent = cmd, sent
            height = torch.stack([d.data.root_pos_w[:, 2] for d in raw._defenders], 1) - raw.scene.env_origins[:, 2:3]
            floor_steps += int((height < 0.3).sum())

            finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
            if len(finished):
                episodes += len(finished)
                breaches += int(raw.outcome_breached[finished].sum())
                captures += float(raw.outcome_captures[finished].sum())
                slots += len(finished) * raw.cfg.num_attackers
                if sign_prev is not None:
                    sign_prev[finished] = 0.0

    raw_all = torch.cat(raw_changes)
    sent_all = torch.cat(sent_changes) if sent_changes else torch.zeros(1, device=raw.device)
    seconds = active_steps * raw.step_dt  # defender-seconds
    label = args_cli.checkpoint or f"{args_cli.pilot} (contact_offset {args_cli.contact_offset}, {args_cli.hybrid_state} state)"
    print(f"\n=== {label} on {args_cli.task} (diagnostic, not an official score) ===", flush=True)
    print(f"tau {raw.cfg.command.response_time} s, top speed {speed} m/s, command latency {raw.cfg.command.command_latency_steps} physics steps", flush=True)
    print(f"outcome:     breach {100 * breaches / max(episodes, 1):.1f}% of {episodes} episodes, interception {100 * captures / max(slots, 1):.1f}%, "
          f"defender floor time {100 * floor_steps / max(active_steps, 1):.1f}%", flush=True)
    print(f"raw change:  mean {float(raw_all.mean()):.3f} m/s per 60 Hz step, p95 {float(raw_all.quantile(0.95)):.3f}, max {float(raw_all.max()):.2f}", flush=True)
    print(f"sent change: mean {float(sent_all.mean()):.3f} m/s per changed packet, p95 {float(sent_all.quantile(0.95)):.3f}, "
          f"jumps > 0.5 m/s: {100 * float((sent_all > 0.5).float().mean()):.1f}% of changes, {float((sent_all > 0.5).sum()) / max(seconds, 1e-6):.3f} per s", flush=True)
    print(f"variation:   {variation / max(seconds, 1e-6):.2f} m/s of command change per second", flush=True)
    print(f"reversals:   {reversals / max(seconds, 1e-6):.3f} per second per defender (horizontal axes)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
