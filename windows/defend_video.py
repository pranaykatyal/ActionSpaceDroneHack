"""Top-down video of a Defend task: guard alone, greedy chase, or a trained MAPPO checkpoint.

Isaac's viewport crashes on this laptop (RTX + iGPU display), so this records drone positions
from a headless rollout and draws them with matplotlib: a 2x2 grid of environments, the goal
zone, defenders (blue), attackers (red while attacking, grey when hit or out), short trails,
and per-env counts of hits and breaches.

  conda activate isaaclab
  cd Action_Space\\simulation_infra
  python ..\\..\\windows\\defend_video.py --task AS-Defend-RoomReal-v0 --out outputs\\video_guard.mp4
  python ..\\..\\windows\\defend_video.py --task AS-Defend-RoomReal3-v0 --out outputs\\video_guard_3v2.mp4
  python ..\\..\\windows\\defend_video.py --task AS-Defend-Room-v0 --baseline chase --set defender_base=none
  python ..\\..\\windows\\defend_video.py --task AS-Defend-Room-v0 --checkpoint <run>\\checkpoints\\best_agent.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Top-down Defend video.")
parser.add_argument("--task", default="AS-Defend-RoomReal-v0")
parser.add_argument("--checkpoint", default=None, help="MAPPO checkpoint; default: zero residual (guard alone)")
parser.add_argument("--baseline", choices=["zero", "chase"], default="zero")
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--out", default="outputs/defend_video.mp4")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env cfg overrides (Python literals)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import ast  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

NUM = 4  # environments drawn


def chase_actions(raw, obs):
    """scripts/evaluate.py's greedy pursuit (importing it would re-run its app launch)."""
    actions, n = {}, raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * n].reshape(raw.num_envs, n, 3)
        dist = torch.linalg.norm(rel, dim=-1).masked_fill(~raw._attacker_alive, float("inf"))
        target = rel[torch.arange(raw.num_envs, device=raw.device), dist.argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def main():
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=NUM)
    for item in args_cli.set:
        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        owner = cfg
        *path, key = key.split(".")
        for name in path:
            owner = getattr(owner, name)
        setattr(owner, key, value)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    policy = None
    if args_cli.checkpoint:
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry
        from skrl.utils.runner.torch import Runner

        runner = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), load_cfg_from_registry(args_cli.task, "skrl_mappo_cfg_entry_point"))
        runner.agent.load(args_cli.checkpoint)
        runner.agent.set_running_mode("eval")
        policy = runner.agent
    obs, _ = env.reset()
    agents = raw.cfg.possible_agents
    o = raw.scene.env_origins.cpu().numpy()
    frames, hits, breaches = [], np.zeros(NUM, int), np.zeros(NUM, int)
    for _ in range(int(args_cli.seconds / raw.step_dt)):
        with torch.inference_mode():
            if policy is not None:
                out = policy.act(obs, {a: None for a in agents}, timestep=0, timesteps=0)
                actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in agents}
            elif args_cli.baseline == "chase":
                actions = chase_actions(raw, obs)
            else:
                actions = {a: torch.zeros(NUM, 4, device=raw.device) for a in agents}
        alive_before = raw._attacker_alive.clone()
        obs, _, _, _, _ = env.step(actions)
        hits += (alive_before & ~raw._attacker_alive).sum(-1).cpu().numpy()
        breaches += raw.outcome_breached.cpu().numpy() & raw.outcome_valid.cpu().numpy()
        frames.append((
            np.stack([d.data.root_pos_w.cpu().numpy() - o for d in raw._defenders], 1),
            np.stack([d.data.root_pos_w.cpu().numpy() - o for d in raw._attackers], 1),
            raw._attacker_alive.cpu().numpy().copy(), hits.copy(), breaches.copy(),
            raw.outcome_valid.cpu().numpy().copy(),
        ))
    env.close()
    render(frames, raw.cfg, raw.step_dt)


def render(frames, cfg, dt):
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    lane = getattr(cfg, "attacker_spawn", "ring") == "lane"
    lim_x = (cfg.lane_offset[0] - 0.8, 1.6) if lane else (-cfg.spawn_radius - 1, cfg.spawn_radius + 1)
    lim_y = (cfg.lane_offset[1] - 2.0, cfg.lane_offset[2] + 2.5) if lane else lim_x
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    title = fig.suptitle("")
    artists = []
    for k, ax in enumerate(axes.flat):
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.add_patch(plt.Circle((0, 0), cfg.site_radius, color="tab:green", alpha=0.25))
        ax.text(0, 0, "goal", ha="center", va="center", fontsize=8)
        if lane:
            ax.axvline(cfg.lane_offset[0] - 0.6, color="k", lw=3)  # the tag-10 wall the attackers launch from
        d = ax.scatter([], [], s=90, c="tab:blue", edgecolors="k", zorder=3)
        a = ax.scatter([], [], s=90, marker="^", edgecolors="k", zorder=3)
        trails = [ax.plot([], [], "-", lw=1, alpha=0.5, color=c)[0] for c in ["tab:blue"] * cfg.num_defenders + ["tab:red"] * cfg.num_attackers]
        label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
        artists.append((d, a, trails, label))
    step = max(1, int(round(0.1 / dt)))  # 10 fps of sim time, played at 10 fps = real time
    idx = list(range(0, len(frames), step))

    def update(n):
        i = idx[n]
        title.set_text(f"{cfg.__class__.__name__}   t = {i * dt:5.1f} s   (blue = defenders, red = attacking, grey = hit / out)")
        for k, (d, a, trails, label) in enumerate(artists):
            dp, ap, alive, hits, br, _ = frames[i]
            d.set_offsets(dp[k, :, :2])
            a.set_offsets(ap[k, :, :2])
            a.set_color(["tab:red" if x else "0.6" for x in alive[k]])
            a.set_edgecolor("k")
            lo = max(0, i - int(2.0 / dt))
            pts = [f[0][k] for f in frames[lo:i + 1]], [f[1][k] for f in frames[lo:i + 1]]
            for j, tr in enumerate(trails):
                src = pts[0] if j < cfg.num_defenders else pts[1]
                jj = j if j < cfg.num_defenders else j - cfg.num_defenders
                tr.set_data([p[jj, 0] for p in src], [p[jj, 1] for p in src])
            label.set_text(f"env {k}: hits {hits[k]}  breaches {br[k]}")
        return []

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(args_cli.out, writer=FFMpegWriter(fps=10, bitrate=2400))
    print(f"[video] wrote {args_cli.out} ({len(idx)} frames)", flush=True)


if __name__ == "__main__":
    main()
    app.close()
