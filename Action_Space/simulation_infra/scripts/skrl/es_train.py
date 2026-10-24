# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Evolution strategies (OpenAI-ES) fine-tuning of the defend policy on the official outcome.

PPO fine-tunes peak after 25-40 iterations and then degrade (EXPERIMENTS.md): its critic has to
predict a breach 5-11 s ahead from 0.27 s rollouts, and can't. ES needs no critic and no reward.
Each generation flies a population of perturbed copies of the policy (antithetic pairs), scores
each copy by what scripts/evaluate.py counts (breach first, then how long the asset held, then
interceptions) and moves the weights toward the better copies. Only the policy path (trunk and
mean head of both defenders) is perturbed; the observation scaler stays frozen. Checkpoints are
skrl files written by ``agent.save()``, so evaluate.py and eval_sweep.sh load them as they are.

    python -u scripts/skrl/es_train.py --checkpoint <agent.pt> --calibrate        # action change per sigma
    python -u scripts/skrl/es_train.py --checkpoint <agent.pt> --generations 100
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenAI-ES fine-tuning on the official defend outcome.")
parser.add_argument("--task", type=str, default="AS-Defend-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="skrl IPPO checkpoint to start from.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--pop", type=int, default=240, help="Perturbed copies per generation (even: antithetic pairs).")
parser.add_argument("--center_envs", type=int, default=256, help="Envs flying the unperturbed policy (progress curve).")
parser.add_argument("--sigma", type=float, default=0.01, help="Perturbation std on the weights.")
parser.add_argument("--lr", type=float, default=0.003, help="Adam step on the weights.")
parser.add_argument("--generations", type=int, default=100)
parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--w_hold", type=float, default=0.5, help="Fitness per fraction of the episode held before a breach.")
parser.add_argument("--w_capture", type=float, default=0.05, help="Fitness per interception.")
parser.add_argument("--w_floor", type=float, default=0.0, help="Fitness penalty per fraction of defender-time on the floor (< 0.3 m): landing counts against us.")
parser.add_argument("--name", type=str, default="es")
parser.add_argument("--resume", type=str, default=None, help="es_state.pt of an earlier run to continue from.")
parser.add_argument("--crn", action="store_true", help="Common random numbers: each antithetic pair starts from identical first waves.")
parser.add_argument("--calibrate", action="store_true", help="Print the action change per sigma and exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn.functional as F

import action_space_kit.tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

POLICY_PARAMS = [
    "net_container.0.weight", "net_container.0.bias",
    "net_container.2.weight", "net_container.2.bias",
    "net_container.4.weight", "net_container.4.bias",
    "policy_layer.weight", "policy_layer.bias",
]


def forward(params: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """The policy's mean action for G weight sets at once: params[k] has a leading G, x is (G, E, 23)."""
    h = x
    for i in range(3):
        h = F.elu(torch.baddbmm(params[2 * i + 1].unsqueeze(1), h, params[2 * i].transpose(1, 2)))
    return torch.baddbmm(params[7].unsqueeze(1), h, params[6].transpose(1, 2))


def main():
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    raw = env
    dev = raw.device

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")
    log_root = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_es_{args_cli.name}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), raw.cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "es_args.txt"), "w") as f:
        f.write(f"{vars(args_cli)}\n")

    # the agent exactly as evaluate.py builds it
    agent = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), agent_cfg).agent
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval") if hasattr(agent, "set_running_mode") else agent.enable_training_mode(False)
    uids = list(raw.cfg.possible_agents)
    scalers = {u: agent._observation_preprocessor[u] for u in uids}
    named = {u: dict(agent.policies[u].named_parameters()) for u in uids}
    for u in uids:
        net = agent.policies[u].net_container
        assert len(net) == 6 and all(isinstance(net[i], torch.nn.ELU) and net[i].alpha == 1.0 for i in (1, 3, 5)), net
        assert all(n in named[u] for n in POLICY_PARAMS), sorted(named[u])

    # flat weight vector over both defenders' policy paths
    layout, off = [], 0
    for u in uids:
        for n in POLICY_PARAMS:
            p = named[u][n]
            layout.append((u, n, tuple(p.shape), off, p.numel()))
            off += p.numel()
    dim = off
    theta = torch.cat([named[u][n].detach().flatten() for u, n, *_ in layout]).clone()
    adam_m = torch.zeros_like(theta)
    adam_v = torch.zeros_like(theta)
    start_gen = 0
    if args_cli.resume:
        state = torch.load(args_cli.resume, map_location=dev)
        theta, adam_m, adam_v, start_gen = state["theta"], state["adam_m"], state["adam_v"], state["gen"]
        print(f"[es] resumed {args_cli.resume} at generation {start_gen}", flush=True)
    print(f"[es] {dim} weights ({dim // len(uids)} per defender), log dir {log_dir}", flush=True)

    def unpack(flat: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        out = {u: [] for u in uids}
        for u, _, shape, o, n in layout:
            out[u].append(flat[:, o : o + n].reshape(flat.shape[0], *shape))
        return out

    num_envs, pop, center = raw.num_envs, args_cli.pop, args_cli.center_envs
    per = (num_envs - center) // pop
    assert pop % 2 == 0 and per * pop + center == num_envs, "need num_envs = pop * envs_per_copy + center_envs"
    split = pop * per

    def act(obs, members, middle):
        actions = {}
        for u in uids:
            x = scalers[u](obs[u])
            a_members = forward(members[u], x[:split].reshape(pop, per, -1)).reshape(split, -1)
            a_center = forward(middle[u], x[split:].unsqueeze(0)).squeeze(0)
            actions[u] = torch.cat([a_members, a_center], dim=0)
        return actions

    # parity: our batched forward must give the actions evaluate.py takes from skrl
    obs, _ = env.reset()
    with torch.inference_mode():
        out = agent.act(obs, {u: None for u in uids}, timestep=0, timesteps=0)
        ref = {u: out[-1][u].get("mean_actions", out[0][u]) for u in uids}
        middle = unpack(theta.unsqueeze(0))
        for u in uids:
            mine = forward(middle[u], scalers[u](obs[u]).unsqueeze(0)).squeeze(0)
            err = float((mine - ref[u]).abs().max())
            print(f"[es] parity {u}: max |ours - skrl mean_actions| = {err:.2e}", flush=True)
            assert err < 1e-4, "batched forward differs from the skrl policy"

    if args_cli.calibrate:
        with torch.inference_mode():
            samples = []
            for t in range(300):
                obs, *_ = env.step({u: forward(middle[u], scalers[u](obs[u]).unsqueeze(0)).squeeze(0) for u in uids})
                if t % 30 == 0:
                    samples.append({u: obs[u][:256].clone() for u in uids})
            for sigma in (0.003, 0.005, 0.01, 0.02, 0.04):
                noisy = unpack(theta.unsqueeze(0) + sigma * torch.randn(16, dim, device=dev))
                diffs = []
                for s in samples:
                    for u in uids:
                        x = scalers[u](s[u])
                        base = forward(middle[u], x.unsqueeze(0)).clamp(-1, 1)
                        pert = forward(noisy[u], x.unsqueeze(0).expand(16, -1, -1)).clamp(-1, 1)
                        diffs.append((pert - base).square().mean().sqrt())
                print(f"[es] sigma {sigma:<6} action change RMS {float(torch.stack(diffs).mean()):.3f} (actions in [-1, 1])", flush=True)
        env.close()
        return

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir)
    max_len = raw.max_episode_length
    n_att = raw.cfg.num_attackers

    def save(gen: int) -> None:
        with torch.no_grad():
            for u, n, shape, o, k in layout:
                named[u][n].copy_(theta[o : o + k].view(shape))
        path = os.path.join(log_dir, "checkpoints", f"agent_{gen}.pt")
        agent.save(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for u in saved:
            saved[u].pop("optimizer", None)
        torch.save(saved, path)
        torch.save({"theta": theta, "adam_m": adam_m, "adam_v": adam_v, "gen": gen, "args": vars(args_cli)}, os.path.join(log_dir, "es_state.pt"))
        print(f"[es] saved {path}", flush=True)

    b1, b2 = 0.9, 0.999
    for gen in range(start_gen + 1, start_gen + args_cli.generations + 1):
        t0 = time.time()
        with torch.no_grad():
            half = torch.randn(pop // 2, dim, device=dev)
            eps = torch.cat([half, -half], dim=0)
            members = unpack(theta.unsqueeze(0) + args_cli.sigma * eps)
            middle = unpack(theta.unsqueeze(0))
        with torch.inference_mode():
            obs, _ = env.reset()
            if args_cli.crn:
                # member k and k + pop/2 carry +eps and -eps: give their envs the same first wave, so
                # the pair's difference reflects the weights rather than the scenarios they drew
                half_envs = (pop // 2) * per
                src = torch.arange(half_envs, device=dev)
                dst = src + half_envs
                shift = raw.scene.env_origins[dst] - raw.scene.env_origins[src]
                for drone in list(raw._defenders) + list(raw._attackers):
                    state = drone.data.root_state_w[src].clone()
                    state[:, :3] += shift
                    drone.write_root_pose_to_sim(state[:, :7], dst)
                    drone.write_root_velocity_to_sim(state[:, 7:], dst)
                raw._attacker_speed[dst] = raw._attacker_speed[src]
                obs = raw._get_observations()
            done = torch.zeros(num_envs, dtype=torch.bool, device=dev)
            breached = torch.zeros(num_envs, device=dev)
            captures = torch.zeros(num_envs, device=dev)
            held = torch.zeros(num_envs, device=dev)
            floor = torch.zeros(num_envs, device=dev)  # defender-steps below 0.3 m in the first episode
            for t in range(max_len):
                obs, *_ = env.step(act(obs, members, middle))
                height = torch.stack([d.data.root_pos_w[:, 2] for d in raw._defenders], 1) - raw.scene.env_origins[:, 2:3]
                floor += ((height < 0.3).float().sum(dim=1)) * (~done).float()
                # the first episode of every env: outcomes latched by the env before its auto-reset
                fin = raw.outcome_valid & ~done
                if fin.any():
                    breached[fin] = raw.outcome_breached[fin].float()
                    captures[fin] = raw.outcome_captures[fin]
                    held[fin] = (t + 1) / max_len
                    done |= fin
                if bool(done.all()):
                    break
            floor_frac = floor / (len(raw._defenders) * held.clamp(min=1.0 / max_len) * max_len)
            score = (1.0 - breached) + args_cli.w_hold * held + args_cli.w_capture * captures - args_cli.w_floor * floor_frac
            breached, captures, held, score, floor_frac = breached.clone(), captures.clone(), held.clone(), score.clone(), floor_frac.clone()

        with torch.no_grad():
            fitness = score[:split].view(pop, per).mean(dim=1)
            ranks = fitness.argsort().argsort().float()
            weights = ranks / (pop - 1) - 0.5  # centred ranks: robust to the noisy 0/1 breach outcomes
            grad = weights @ eps / (pop * args_cli.sigma)
            adam_m.mul_(b1).add_(grad, alpha=1 - b1)
            adam_v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            m_hat = adam_m / (1 - b1**gen)  # gen counts Adam steps, also across a resume
            v_hat = adam_v / (1 - b2**gen)
            theta.add_(args_cli.lr * m_hat / (v_hat.sqrt() + 1e-8))

        c = slice(split, num_envs)
        stats = {
            "center breach rate": float(breached[c].mean()),
            "center interception rate": float(captures[c].sum() / (center * n_att)),
            "center held fraction": float(held[c].mean()),
            "center fitness": float(score[c].mean()),
            "center floor fraction": float(floor_frac[c].mean()),
            "population breach rate": float(breached[:split].mean()),
            "population fitness mean": float(fitness.mean()),
            "population fitness best": float(fitness.max()),
            "seconds per generation": time.time() - t0,
        }
        for k, v in stats.items():
            writer.add_scalar(f"ES / {k}", v, gen)
        writer.flush()
        print(
            f"[es] gen {gen:4d}  center breach {100 * stats['center breach rate']:5.1f}%  intercept {100 * stats['center interception rate']:5.1f}%"
            f"  held {stats['center held fraction']:.3f}  | population breach {100 * stats['population breach rate']:5.1f}%"
            f"  fitness {stats['population fitness mean']:.3f} (best {stats['population fitness best']:.3f})  {stats['seconds per generation']:.0f} s",
            flush=True,
        )
        if gen % args_cli.save_every == 0:
            save(gen)
    save(start_gen + args_cli.generations)
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
