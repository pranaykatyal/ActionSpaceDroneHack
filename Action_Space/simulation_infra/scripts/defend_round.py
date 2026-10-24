# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Score a defender over timed rounds (the final is 5 minutes): breaches and interceptions per round.

The official metric (scripts/evaluate.py) ends an episode at the first breach, so it can't say
how many attackers get through in five minutes. The Team1 round tasks keep the episode going:
a breach is counted, that wave is replaced, and only the clock ends the round. The policy is
loaded exactly as evaluate.py loads it. A diagnostic, not an official score.

    python -u scripts/defend_round.py --task Team1-Defend-Round-Official-v0 --baseline chase
    python -u scripts/defend_round.py --task Team1-Defend-Round-Realistic-v0 --checkpoint <agent.pt>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Score a defender over timed rounds.")
parser.add_argument("--task", type=str, default="Team1-Defend-Round-Official-v0")
parser.add_argument("--num_envs", type=int, default=64, help="Rounds played in parallel.")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--baseline", type=str, default=None, choices=["zero", "chase", "hybrid"])
parser.add_argument("--zone", action="store_true", help="hybrid: zone defence (always_block + idle_return).")
parser.add_argument("--idle_return", action="store_true", help="hybrid: return to the guard ring between waves.")
parser.add_argument("--guard_radius", type=float, default=2.2, help="hybrid: blocking distance from the site axis [m].")
parser.add_argument("--terminal_range", type=float, default=0.7, help="hybrid: strike inside this range [m].")
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from action_space_kit.control.defense import HybridDefenseCfg, HybridDefender
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


def hybrid_actions(raw, hybrid):
    """The scripted hybrid on the true state (the round tasks keep the asset position known)."""
    origin = raw.scene.env_origins
    def_pos = torch.stack([d.data.root_pos_w - origin for d in raw._defenders], dim=1)
    def_vel = torch.stack([d.data.root_lin_vel_w for d in raw._defenders], dim=1)
    att_pos, att_vel = raw._attacker_states()
    site = torch.zeros_like(origin)
    site[:, 2] = raw.cfg.site_height * 0.5
    v, _, _ = hybrid(def_pos, def_vel, att_pos - origin[:, None, :], att_vel, raw._attacker_alive, site)
    zeros = torch.zeros(raw.num_envs, 1, device=raw.device)
    return {a: torch.cat([v[:, i] / raw.cfg.command.max_lin_speed, zeros], dim=-1) for i, a in enumerate(raw.cfg.possible_agents)}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    raw = env
    policy = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        agent_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point")
        runner = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), agent_cfg)
        runner.agent.load(args_cli.checkpoint)
        runner.agent.set_running_mode("eval") if hasattr(runner.agent, "set_running_mode") else runner.agent.enable_training_mode(False)
        policy = runner.agent

    hybrid = HybridDefender(
        HybridDefenseCfg(
            max_speed=raw.cfg.command.max_lin_speed,
            site_radius=raw.cfg.site_radius,
            guard_radius=args_cli.guard_radius,
            terminal_range=args_cli.terminal_range,
            always_block=args_cli.zone,
            idle_return=args_cli.zone or args_cli.idle_return,
        )
    )
    obs, _ = env.reset()
    breaches, intercepts = [], []
    floor_steps = torch.zeros(raw.num_envs, device=raw.device)  # defender-steps spent below 0.3 m
    steps = 0
    down_at_end = []
    while len(breaches) < raw.num_envs:
        if policy is not None:
            with torch.inference_mode():
                out = policy.act(obs, {a: None for a in raw.cfg.possible_agents}, timestep=0, timesteps=0)
                actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in raw.cfg.possible_agents}
        elif args_cli.baseline == "chase":
            actions = chase_actions(raw, obs)
        elif args_cli.baseline == "hybrid":
            actions = hybrid_actions(raw, hybrid)
        else:
            actions = {a: torch.zeros(raw.num_envs, 4, device=raw.device) for a in raw.cfg.possible_agents}
        before = raw.round_breaches.clone()  # the reset after the final step zeroes it
        # heights before the step: a finished round is reset inside step(), so read them now
        prev_height = torch.stack([d.data.root_pos_w[:, 2] for d in raw._defenders], 1) - raw.scene.env_origins[:, 2:3]
        obs, _, _, _, _ = env.step(actions)
        steps += 1
        height = torch.stack([d.data.root_pos_w[:, 2] for d in raw._defenders], 1) - raw.scene.env_origins[:, 2:3]
        floor_steps += (height < 0.3).float().sum(dim=1)
        finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        for i in finished.tolist():
            breaches.append(float(torch.maximum(before[i], raw.round_breaches[i])))
            intercepts.append(float(raw.outcome_captures[i]))
            down_at_end.append(float((prev_height[i] < 0.3).sum()))

    minutes = raw.cfg.episode_length_s / 60.0
    n = len(breaches)

    def summary(xs):
        mean = sum(xs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(n - 1, 1))
        return mean, 1.96 * sd / math.sqrt(n)

    b, bci = summary(breaches)
    c, cci = summary(intercepts)
    label = args_cli.checkpoint or f"baseline:{args_cli.baseline or 'zero'}"
    if args_cli.baseline == "hybrid":
        label += f" (zone={args_cli.zone}, idle_return={args_cli.zone or args_cli.idle_return}, guard_radius={args_cli.guard_radius}, terminal_range={args_cli.terminal_range})"
    print(f"\n=== {label} on {args_cli.task}: {n} rounds of {minutes:.0f} min (diagnostic, not an official score) ===", flush=True)
    print(f"breaches per round:      {b:6.2f} +- {bci:.2f}   ({b / minutes:.2f} per min)", flush=True)
    print(f"interceptions per round: {c:6.2f} +- {cci:.2f}   ({c / minutes:.2f} per min)", flush=True)
    print(f"rounds with no breach:   {100.0 * sum(x == 0 for x in breaches) / n:.0f}%", flush=True)
    share = float(floor_steps.sum()) / (steps * raw.num_envs * raw.cfg.num_defenders)
    print(f"defender time on the floor (< 0.3 m): {100.0 * share:.1f}%; defenders down at the end: {sum(down_at_end) / n:.2f} of {raw.cfg.num_defenders} per round", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
