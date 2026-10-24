# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Measure how well a defender policy actually defends.

Training curves say little about the thing that matters on day two: how often attackers
are stopped before they reach the site. This script runs whole episodes and reports
interception rate, breach rate and time-to-intercept, for a trained checkpoint or for
either built-in baseline.

    python scripts/evaluate.py --task AS-Defend-v0 --baseline chase --episodes 20
    python scripts/evaluate.py --task AS-Defend-v0 --checkpoint logs/skrl/.../best_agent.pt

Always evaluate a policy against ``--baseline chase`` before trusting it: a policy that
cannot beat greedy pursuit is not yet worth flying.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a defender policy.")
parser.add_argument("--task", type=str, default="AS-Defend-v0", help="Task to evaluate.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments.")
parser.add_argument("--episodes", type=int, default=20, help="Episodes to finish before reporting.")
parser.add_argument("--checkpoint", type=str, default=None, help="Trained skrl checkpoint to load.")
parser.add_argument(
    "--baseline",
    type=str,
    default=None,
    choices=["zero", "chase"],
    help="Evaluate a scripted baseline instead of a checkpoint: zero does nothing, chase pursues the nearest attacker.",
)
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO"], help="Algorithm of the checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the run.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in steps.")
parser.add_argument("--video_dir", type=str, default="videos/evaluate", help="Where to write the video.")
parser.add_argument(
    "--set",
    nargs="*",
    default=[],
    metavar="KEY=VALUE",
    help="env cfg overrides, e.g. attacker_retry=True observe_status=False (Python literals)",
)
parser.add_argument("--diag", action="store_true", default=False, help="Print the situation just before each breach.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True  # rendering is needed to capture frames

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def chase_actions(raw, obs):
    """Head straight at the nearest tracked attacker -- the baseline to beat."""
    actions = {}
    num_attackers = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * num_attackers].reshape(raw.num_envs, num_attackers, 3)
        # Only chase attackers that are attacking. Downed ones are zeroed in the stock observation
        # (argmin would pick a dead slot at distance 0 and stop), and with attacker_retry they
        # stay visible while returning or holding. Read the env's truth so this still works when
        # observe_status hides the flags from the policy.
        distance = torch.linalg.norm(rel, dim=-1).masked_fill(~raw._attacker_alive, float("inf"))
        target = rel[torch.arange(raw.num_envs, device=raw.device), distance.argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    for item in args_cli.set:
        import ast

        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # plain string
        owner = env_cfg
        *path, key = key.split(".")  # dotted keys reach nested cfgs, e.g. command.response_time=0.3
        for name in path:
            owner = getattr(owner, name)
        setattr(owner, key, value)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        import os

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.abspath(args_cli.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[evaluate] recording {args_cli.video_length} steps to {args_cli.video_dir}", flush=True)
    env = env.unwrapped if not args_cli.video else env

    raw = env.unwrapped  # RecordVideo wraps the env; outcome buffers live on the raw one

    policy = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry

        entry = f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point"
        agent_cfg = load_cfg_from_registry(args_cli.task, entry)
        # the wrapper takes no algorithm argument; multi-agent envs stay multi-agent
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        runner = Runner(wrapped, agent_cfg)
        runner.agent.load(args_cli.checkpoint)
        if hasattr(runner.agent, "set_running_mode"):
            runner.agent.set_running_mode("eval")
        else:
            runner.agent.enable_training_mode(False)
        policy = runner.agent

    obs, _ = env.reset()
    episodes = breaches = 0
    intercepted = attacker_slots = 0
    intercept_times: list[float] = []
    alive_prev = raw._attacker_alive.clone()
    step_in_episode = torch.zeros(raw.num_envs, device=raw.device)
    episode_intercept_times: list[list[float]] = [[] for _ in range(raw.num_envs)]
    captures_so_far = torch.zeros(raw.num_envs, raw.cfg.num_attackers, device=raw.device)
    episode_crashed = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    crashes = 0

    while episodes < args_cli.episodes:
        if policy is not None:
            with torch.inference_mode():
                # multi-agent skrl agents take and return dicts keyed by agent, and
                # they index states[agent_id], so the states argument must be a dict too
                states = {agent: None for agent in raw.cfg.possible_agents}
                outputs = policy.act(obs, states, timestep=0, timesteps=0)
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in raw.cfg.possible_agents
                }
        elif args_cli.baseline == "chase":
            actions = chase_actions(raw, obs)
        else:
            actions = {a: torch.zeros(raw.num_envs, 4, device=raw.device) for a in raw.cfg.possible_agents}

        if args_cli.diag:
            # snapshot before the step: after it, a breached env has already been reset
            att = torch.stack([d.data.root_pos_w for d in raw._attackers], dim=1) - raw.scene.env_origins.unsqueeze(1)
            dfn = torch.stack([d.data.root_pos_w for d in raw._defenders], dim=1) - raw.scene.env_origins.unsqueeze(1)
            snap = {
                "radial": torch.linalg.norm(att[..., :2], dim=-1),
                "def_dist": torch.cdist(att, dfn).min(dim=-1).values,
                "def_radial": torch.linalg.norm(dfn[..., :2], dim=-1),
                "captures": captures_so_far.clone(),
                "speed": raw._attacker_speed.clone(),
            }
        # a defender on the floor (knocked down by a collision) stays there; count it
        def_z = torch.stack([d.data.root_pos_w[:, 2] for d in raw._defenders], dim=1) - raw.scene.env_origins[:, 2:3]
        crashed_now = (def_z < 0.15).any(dim=-1) & (step_in_episode > 5)
        obs, _, terminated, truncated, _ = env.step(actions)
        step_in_episode += 1
        episode_crashed |= crashed_now
        if args_cli.diag:
            captures_so_far += (alive_prev & ~raw._attacker_alive).float()

        newly_down = alive_prev & ~raw._attacker_alive
        if newly_down.any():
            for row in newly_down.any(dim=-1).nonzero(as_tuple=False).flatten().tolist():
                episode_intercept_times[row].append(step_in_episode[row].item() * raw.step_dt)
        alive_prev = raw._attacker_alive.clone()

        # outcomes are latched by the env before its auto-reset clears them
        finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        if len(finished):
            episodes += len(finished)
            breaches += int(raw.outcome_breached[finished].sum().item())
            intercepted += int(raw.outcome_captures[finished].sum().item())
            attacker_slots += len(finished) * raw.cfg.num_attackers
            crashes += int(episode_crashed[finished].sum().item())
            episode_crashed[finished] = False
            for row in finished.tolist():
                intercept_times += episode_intercept_times[row]
                episode_intercept_times[row] = []
                if args_cli.diag and raw.outcome_breached[row]:
                    a = int(snap["radial"][row].argmin())  # the attacker that got in
                    print(
                        "[diag] breach t=%.1fs attacker=%d prior_captures=%d speed=%.2f "
                        "nearest_defender=%.2fm defender_radials=%s"
                        % (step_in_episode[row].item() * raw.step_dt, a, int(snap["captures"][row, a]),
                           snap["speed"][row, a], snap["def_dist"][row, a],
                           [round(v, 1) for v in snap["def_radial"][row].tolist()]),
                        flush=True,
                    )
                if args_cli.diag:
                    captures_so_far[row] = 0.0
            step_in_episode[finished] = 0.0

    label = args_cli.checkpoint or f"baseline:{args_cli.baseline or 'zero'}"
    print("\n=== %s on %s ===" % (label, args_cli.task), flush=True)
    print("episodes:            %d" % episodes, flush=True)
    print("interception rate:   %.1f%% of attackers" % (100.0 * intercepted / max(attacker_slots, 1)), flush=True)
    print("breach rate:         %.1f%% of episodes" % (100.0 * breaches / max(episodes, 1)), flush=True)
    print("defender crash rate: %.1f%% of episodes (a defender ended up on the floor)" % (100.0 * crashes / max(episodes, 1)), flush=True)
    if intercept_times:
        mean_t = sum(intercept_times) / len(intercept_times)
        print("time to intercept:   %.1f s (mean of %d)" % (mean_t, len(intercept_times)), flush=True)
    else:
        print("time to intercept:   n/a (nothing intercepted)", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
