# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Why does a defend policy still breach? A diagnostic, not an official score.

Flies a checkpoint (loaded exactly as scripts/evaluate.py loads it) and, for every breached episode,
records the state one step before the breach: when it happened, which wave, the breaching attacker's
speed and slot, how far the nearest defender was, whether its paired defender (same slot index) or
any defender was on the floor, and how many defenders were within 1 m of it.

    python -u scripts/breach_autopsy.py --checkpoint <agent.pt> --episodes 1024
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Breach autopsy for a defend checkpoint.")
parser.add_argument("--task", type=str, default="AS-Defend-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--checkpoint", type=str, required=True)
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
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    raw = env
    agent = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), load_cfg_from_registry(args_cli.task, "skrl_ippo_cfg_entry_point")).agent
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval") if hasattr(agent, "set_running_mode") else agent.enable_training_mode(False)
    uids = list(raw.cfg.possible_agents)
    n = raw.num_envs
    dev = raw.device

    obs, _ = env.reset()
    # crash autopsy: closest approach to the teammate / a live attacker / a downed attacker over the
    # 1.5 s before a defender first drops below 0.3 m (a tilted drone falls for ~0.5-1 s)
    window = int(1.5 / raw.step_dt)
    n_def = len(raw._defenders)
    hist = torch.full((3, window, n, n_def), 1e9, device=dev)
    head = 0
    was_up = torch.ones(n, n_def, dtype=torch.bool, device=dev)
    crashes = Counter()
    # tight version (a Tello's collision box is ~0.1 m, so bodies touch only within ~0.13 m centre to centre)
    tight_window = int(0.3 / raw.step_dt)
    tight = Counter()
    drop_tilt, drop_vz = [], []
    # the flip, not the floor, is the event: a tipped drone's thrust (hosts' body-frame bug) dives it
    was_level = torch.ones(n, n_def, dtype=torch.bool, device=dev)
    flips = Counter()
    t_in_ep = torch.zeros(n, device=dev)
    caps_in_ep = torch.zeros(n, device=dev)
    episodes = breaches = 0
    rows = []
    while episodes < args_cli.episodes:
        # the state this step starts from: a breach latched after the step is judged on it
        origin = raw.scene.env_origins
        d_pos = torch.stack([d.data.root_pos_w for d in raw._defenders], 1) - origin[:, None, :]
        a_pos, a_vel = raw._attacker_states()
        a_pos = a_pos - origin[:, None, :]
        alive = raw._attacker_alive.clone()
        speed = raw._attacker_speed.clone()
        with torch.inference_mode():
            out = agent.act(obs, {u: None for u in uids}, timestep=0, timesteps=0)
            actions = {u: out[-1][u].get("mean_actions", out[0][u]) for u in uids}
        caps_before = raw._episode_captures.clone()
        gap_dd = torch.cdist(d_pos, d_pos) + torch.eye(n_def, device=dev) * 1e9
        gap_da = torch.cdist(d_pos, a_pos)  # (n, D, A)
        big = torch.full_like(gap_da, 1e9)
        hist[0, head] = gap_dd.min(dim=-1).values
        hist[1, head] = torch.where(alive[:, None, :], gap_da, big).min(dim=-1).values
        hist[2, head] = torch.where(~alive[:, None, :], gap_da, big).min(dim=-1).values
        head = (head + 1) % window
        up = d_pos[..., 2] >= 0.3
        fell = was_up & ~up
        for e, d in fell.nonzero(as_tuple=False).tolist():
            near = hist[:, :, e, d].min(dim=1).values  # closest approach in the window, per kind
            kinds = [k for k, g in zip(("teammate", "live attacker", "downed attacker"), near.tolist()) if g < 0.45]
            crashes[" + ".join(kinds) if kinds else "nothing within 0.45 m"] += 1
            recent = [(head - 1 - j) % window for j in range(tight_window)]  # the last 0.3 s
            near_t = hist[:, recent, e, d].min(dim=1).values
            kinds_t = [k for k, g in zip(("teammate", "live attacker", "downed attacker"), near_t.tolist()) if g < 0.15]
            tight[" + ".join(kinds_t) if kinds_t else "no contact (< 0.15 m) in the last 0.3 s"] += 1
            drone = raw._defenders[d]
            drop_tilt.append(float(torch.rad2deg(torch.acos((-drone.data.projected_gravity_b[e, 2]).clamp(-1.0, 1.0)))))
            drop_vz.append(float(drone.data.root_lin_vel_w[e, 2]))
        was_up = up
        tilt = torch.stack([torch.rad2deg(torch.acos((-dr.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))) for dr in raw._defenders], 1)
        level = tilt < 30.0
        for e, d in (was_level & ~level).nonzero(as_tuple=False).tolist():
            recent = [(head - 1 - j) % window for j in range(tight_window)]
            near_f = hist[:, recent, e, d].min(dim=1).values
            kinds_f = [k for k, g in zip(("teammate", "live attacker", "downed attacker"), near_f.tolist()) if g < 0.15]
            flips[" + ".join(kinds_f) if kinds_f else "no contact (< 0.15 m)"] += 1
        was_level = level
        obs, *_ = env.step(actions)
        t_in_ep += raw.step_dt
        fin = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        for i in fin.tolist():
            episodes += 1
            if bool(raw.outcome_breached[i]):
                breaches += 1
                radial = torch.linalg.norm(a_pos[i, :, :2], dim=-1)
                radial = torch.where(alive[i], radial, torch.full_like(radial, 1e9))
                k = int(radial.argmin())  # the live attacker closest to the asset breached
                gaps = torch.linalg.norm(d_pos[i] - a_pos[i, k], dim=-1)
                down = d_pos[i, :, 2] < 0.3
                paired = k % len(uids)
                rows.append({
                    "time": float(t_in_ep[i]),
                    "wave": int(caps_before[i] // raw.cfg.num_attackers) + 1,
                    "slot": k,
                    "speed": float(speed[i, k]),
                    "nearest": float(gaps.min()),
                    "paired_down": bool(down[paired]),
                    "any_down": bool(down.any()),
                    "within_1m": int((gaps < 1.0).sum()),
                    "other_alive": bool(alive[i].sum() > 1),
                })
            t_in_ep[i] = 0.0
            hist[:, :, i] = 1e9  # a new episode starts with a clean window
            was_up[i] = True
            was_level[i] = True
    print(f"\n=== breach autopsy: {args_cli.checkpoint} on {args_cli.task} (diagnostic) ===", flush=True)
    print(f"episodes {episodes}, breached {breaches} ({100 * breaches / max(episodes, 1):.1f}%, counted like evaluate.py)", flush=True)
    if not rows:
        return
    def share(pred):
        return f"{100 * sum(pred(r) for r in rows) / len(rows):.0f}%"
    print("breach time (s):   " + "  ".join(f"{a}-{a + 5}: {sum(a <= r['time'] < a + 5 for r in rows)}" for a in (0, 5, 10, 15)), flush=True)
    print("wave at breach:    " + "  ".join(f"{w}: {c}" for w, c in sorted(Counter(min(r['wave'], 5) for r in rows).items())), flush=True)
    print("attacker speed:    " + "  ".join(f"{lo:.1f}-{lo + 0.1:.1f}: {sum(lo <= r['speed'] < lo + 0.1 for r in rows)}" for lo in (0.4, 0.5, 0.6, 0.7, 0.8)), flush=True)
    print(f"paired defender on the floor: {share(lambda r: r['paired_down'])}; any defender on the floor: {share(lambda r: r['any_down'])}", flush=True)
    print(f"nearest defender: <0.5 m {share(lambda r: r['nearest'] < 0.5)}, 0.5-1.5 m {share(lambda r: 0.5 <= r['nearest'] < 1.5)}, >=1.5 m {share(lambda r: r['nearest'] >= 1.5)}", flush=True)
    print(f"both defenders within 1 m of it: {share(lambda r: r['within_1m'] >= 2)}; other attacker still alive: {share(lambda r: r['other_alive'])}", flush=True)
    print(f"breaching slot: " + "  ".join(f"{s}: {c}" for s, c in sorted(Counter(r['slot'] for r in rows).items())), flush=True)
    total = sum(crashes.values())
    print(f"defender crashes (first drop below 0.3 m): {total} in {episodes} episodes; closest approach within 1.5 s before:", flush=True)
    for kind, c in crashes.most_common():
        print(f"   {kind:40s} {c:5d}  ({100 * c / max(total, 1):.0f}%)", flush=True)
    print("tight: closest approach < 0.15 m within 0.3 s before the drop:", flush=True)
    for kind, c in tight.most_common():
        print(f"   {kind:40s} {c:5d}  ({100 * c / max(total, 1):.0f}%)", flush=True)
    total_f = sum(flips.values())
    print(f"flips (tilt first passes 30 deg): {total_f}; closest approach < 0.15 m within 0.3 s before the flip:", flush=True)
    for kind, c in flips.most_common():
        print(f"   {kind:40s} {c:5d}  ({100 * c / max(total_f, 1):.0f}%)", flush=True)
    if drop_tilt:
        tilt = sorted(drop_tilt)
        vz = sorted(drop_vz)
        q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
        print(f"at the drop: tilt median {q(tilt, 0.5):.0f} deg (p10 {q(tilt, 0.1):.0f}, p90 {q(tilt, 0.9):.0f}); "
              f"vz median {q(vz, 0.5):+.2f} m/s (p10 {q(vz, 0.1):+.2f}, p90 {q(vz, 0.9):+.2f}); "
              f"level (< 20 deg) drops: {100 * sum(t < 20 for t in tilt) / len(tilt):.0f}%", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
