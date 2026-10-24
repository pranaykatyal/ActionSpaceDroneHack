"""Export a skrl MAPPO/IPPO defender checkpoint to a NumPy .npz for the flight laptop.

Per defender: the observation scaler (running mean/variance) and the deterministic actor
(Linear-ELU-Linear-ELU-Linear-ELU-Linear). ``act`` below is the whole inference; the
flight script copies it, so it runs without torch or Isaac. With ``--check`` the exported
NumPy policy is flown in AS-Defend-Room-v0 and scored like evaluate.py.

    python scripts/export_defend_policy.py --checkpoint policies/defend_room_B7200/checkpoint.pt \
        --out policies/defend_room_B7200/policy.npz
"""

import argparse

import numpy as np

CLIP = 5.0  # skrl RunningStandardScaler clip_threshold
EPS = 1e-8  # skrl RunningStandardScaler epsilon


def act(npz, agent: str, obs: np.ndarray) -> np.ndarray:
    """obs (N, obs_dim) -> deterministic action (N, 4) in [-1, 1]."""
    x = np.clip((obs - npz[f"{agent}/mean"]) / np.sqrt(npz[f"{agent}/var"] + EPS), -CLIP, CLIP)
    for i in range(4):
        x = x @ npz[f"{agent}/W{i}"].T + npz[f"{agent}/b{i}"]
        if i < 3:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
    return np.clip(x, -1.0, 1.0)


def export(checkpoint: str, out: str) -> None:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arrays = {}
    for agent, parts in ck.items():
        pol, pre = parts["policy"], parts["observation_preprocessor"]
        arrays[f"{agent}/mean"] = pre["running_mean"].double().numpy()
        arrays[f"{agent}/var"] = pre["running_variance"].double().numpy()
        for i, layer in enumerate((0, 2, 4, 6)):
            arrays[f"{agent}/W{i}"] = pol[f"net_container.{layer}.weight"].double().numpy()
            arrays[f"{agent}/b{i}"] = pol[f"net_container.{layer}.bias"].double().numpy()
    arrays["agents"] = np.array(sorted(ck))
    np.savez(out, **arrays)
    print(f"[export] {sorted(ck)} -> {out}", flush=True)


def check(npz_path: str, episodes: int) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    import gymnasium as gym
    import torch

    import action_space_kit.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task = "AS-Defend-Room-v0"
    env = gym.make(task, cfg=parse_env_cfg(task, device="cuda:0", num_envs=200)).unwrapped
    npz = np.load(npz_path)
    obs, _ = env.reset()
    done = breaches = crashes = 0
    while done < episodes:
        actions = {a: torch.as_tensor(act(npz, a, obs[a].cpu().double().numpy()), dtype=torch.float32,
                                      device=env.device) for a in env.cfg.possible_agents}
        obs, *_ = env.step(actions)
        fin = env.outcome_valid.nonzero().flatten()
        done += len(fin)
        breaches += int(env.outcome_breached[fin].sum())
        crashes += int(env.outcome_crashed[fin].sum())
    print(f"[check] numpy policy: {done} episodes, breach {100 * breaches / done:.1f}%, "
          f"crash {100 * crashes / done:.1f}%", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=0, help="also fly the export in sim for this many episodes")
    a = ap.parse_args()
    export(a.checkpoint, a.out)
    if a.check:
        check(a.out, a.check)
