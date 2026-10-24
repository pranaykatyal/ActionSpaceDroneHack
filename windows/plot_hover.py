"""Plot a trained AS-Tello-Hover-v0 policy without rendering (for machines where Isaac's RTX
viewport crashes). Runs one 10 s episode headless with the policy, zero actions and the
P-controller from the same seed, then saves a figure of flight paths and distance to goal.

    conda activate isaaclab
    cd Action_Space\\simulation_infra
    python ..\\..\\windows\\plot_hover.py --checkpoint logs\\skrl\\tello_hover\\<run>\\checkpoints\\best_agent.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="skrl checkpoint (best_agent.pt)")
parser.add_argument("--task", default="AS-Tello-Hover-v0", help="AS-Tello-Hover-v0 or AS-Tello-Waypoint-v0")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="env cfg overrides the policy was trained with, e.g. observe_goal_velocity=True. "
                    "Leave reward terms at their defaults so every run is scored on the stock reward.")
parser.add_argument("--num_envs", type=int, default=512, help="drones scored (same seeded starts for every pilot)")
parser.add_argument("--show", type=int, default=8, help="drones drawn in the plot and video")
parser.add_argument("--out", default="outputs/hover_policy.png")
parser.add_argument("--video", nargs="?", const="outputs/hover_policy.mp4", default=None,
                    help="also render an animated mp4 (default path outputs/hover_policy.mp4)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import action_space_kit.tasks  # noqa: F401


def load_policy(path, device):
    """Deterministic action (the Gaussian mean) of an skrl shared policy/value MLP checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    p, pre = ckpt["policy"], ckpt.get("observation_preprocessor")

    def act(obs):
        if pre:
            mean, var = pre["running_mean"].float(), pre["running_variance"].float()
            obs = ((obs - mean) / torch.sqrt(var + 1e-8)).clamp(-5, 5)
        h = torch.nn.functional.elu(obs @ p["net_container.0.weight"].T + p["net_container.0.bias"])
        h = torch.nn.functional.elu(h @ p["net_container.2.weight"].T + p["net_container.2.bias"])
        return (h @ p["policy_layer.weight"].T + p["policy_layer.bias"]).clamp(-1, 1)

    return act


def rollout(env, pilot, seed):
    """One full episode from a seeded reset.

    Returns (positions [T,N,3], goals [N,3], distances [T,N], died [N], returns [N]); a drone's return
    stops accumulating when it dies, the same as the episode return in training.
    """
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    obs = obs["policy"]
    u = env.unwrapped
    origins = u.scene.env_origins
    goals = (u._goal_pos_w - origins).clone()
    alive = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
    ret = torch.zeros(u.num_envs, device=u.device)
    pos, dist = [], []
    for _ in range(int(u.max_episode_length) - 2):  # stop before the time-out reset re-samples goals
        p = u._robot.data.root_pos_w - origins
        d = torch.linalg.norm(u._goal_pos_w - u._robot.data.root_pos_w, dim=-1)
        nan = torch.full_like(p, float("nan"))
        pos.append(torch.where(alive[:, None], p, nan).cpu())
        dist.append(torch.where(alive, d, torch.full_like(d, float("nan"))).cpu())
        obs, rew, term, _, _ = env.step(pilot(obs))
        obs = obs["policy"]
        ret += torch.where(alive, rew, torch.zeros_like(rew))
        alive &= ~term
    return torch.stack(pos), goals.cpu(), torch.stack(dist), (~alive).cpu(), ret.cpu()


cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
for item in args_cli.set:
    key, value = item.split("=", 1)
    setattr(cfg, key, {"true": True, "false": False}.get(value.lower(), value))
env = gym.make(args_cli.task, cfg=cfg)
dev = env.unwrapped.device
dt = env.unwrapped.step_dt

policy = load_policy(args_cli.checkpoint, dev)


def goal_velocity():
    """The goal's world velocity [m/s] (Waypoint moves it; Hover's goal is still)."""
    u = env.unwrapped
    return u._goal_vel_w if hasattr(u, "_goal_vel_w") else torch.zeros(u.num_envs, 3, device=dev)


def p_controller(obs, gain, feedforward=None):
    """Velocity command u = feedforward + gain * goal_error (obs[:, :3]), clipped to the stick range."""
    v = gain * obs[:, :3] + (feedforward if feedforward is not None else 0.0)
    return torch.cat([v.clamp(-1, 1), torch.zeros(obs.shape[0], 1, device=dev)], -1)


pilots = {
    "trained policy": policy,
    "do nothing": lambda obs: torch.zeros(obs.shape[0], 4, device=dev),
    "P-controller": lambda obs: p_controller(obs, gain=1.0),
    "P x3": lambda obs: p_controller(obs, gain=3.0),
    # CLF V = |e|^2/2 with u = v_goal + k e gives dV/dt = -2k V; uses the goal velocity, which the policy can't see
    "CLF (sees v_goal)": lambda obs: p_controller(obs, gain=3.0, feedforward=goal_velocity()),
}
with torch.inference_mode():
    runs = {name: rollout(env, pilot, seed=7) for name, pilot in pilots.items()}

fig = plt.figure(figsize=(15, 9))
colors = plt.cm.tab10.colors
show = min(args_cli.show, args_cli.num_envs)
pos, goals = runs["trained policy"][0][:, :show], runs["trained policy"][1][:show]

ax = fig.add_subplot(2, 3, 1)  # top view
for i in range(pos.shape[1]):
    c = colors[i % 10]
    ax.plot(pos[:, i, 0], pos[:, i, 1], color=c, lw=1.2)
    ax.plot(pos[0, i, 0], pos[0, i, 1], "o", color=c, ms=5, mfc="white")
    ax.plot(goals[i, 0], goals[i, 1], "x", color=c, ms=10, mew=2)
ax.set(title="Trained policy, top view (o start, x goal)", xlabel="x [m]", ylabel="y [m]", aspect="equal")
ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 2)  # side view
for i in range(pos.shape[1]):
    c = colors[i % 10]
    ax.plot(pos[:, i, 0], pos[:, i, 2], color=c, lw=1.2)
    ax.plot(pos[0, i, 0], pos[0, i, 2], "o", color=c, ms=5, mfc="white")
    ax.plot(goals[i, 0], goals[i, 2], "x", color=c, ms=10, mew=2)
ax.axhline(0, color="k", lw=1)
ax.set(title="Trained policy, side view", xlabel="x [m]", ylabel="height z [m]", aspect="equal")
ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 3, projection="3d")
for i in range(pos.shape[1]):
    c = colors[i % 10]
    ax.plot(pos[:, i, 0], pos[:, i, 1], pos[:, i, 2], color=c, lw=1.2)
    ax.scatter(*goals[i], color=c, marker="x", s=60)
ax.set(title="Trained policy, 3D", xlabel="x", ylabel="y", zlabel="z")

ax = fig.add_subplot(2, 1, 2)  # distance to goal, all pilots
styles = {"trained policy": "-", "do nothing": ":", "P-controller": "--", "P x3": "-.", "CLF (sees v_goal)": (0, (1, 1))}
for name, (_, _, dist, dead, ret) in runs.items():
    t = torch.arange(dist.shape[0]) * dt
    mean = torch.nanmean(dist, dim=1)
    ax.plot(t, mean, linestyle=styles[name], lw=2.5, label=f"{name}: return {ret.mean():.1f}, final {mean[-1]*100:.1f} cm, "
            f"crashed {int(dead.sum())}/{dist.shape[1]}")
    if name == "trained policy":
        for i in range(show):
            ax.plot(t, dist[:, i], color=colors[i % 10], lw=0.8, alpha=0.5)
ax.set(title="Distance to goal over one 10 s episode (same starts for every pilot)", xlabel="time [s]",
       ylabel="distance to goal [m]")
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(args_cli.out, dpi=110)
print(f"[plot] wrote {args_cli.out}", flush=True)
for name, (_, _, dist, dead, ret) in runs.items():
    mean = torch.nanmean(dist, dim=1)
    print(f"[plot] {name:15s} return {ret.mean():6.2f} +- {ret.std():4.2f} | distance start {mean[0]*100:5.1f} cm "
          f"-> 1 s {mean[int(1/dt)]*100:5.1f} -> 3 s {mean[int(3/dt)]*100:5.1f} -> end {mean[-1]*100:5.1f} cm "
          f"| crashed {int(dead.sum())}/{dist.shape[1]}", flush=True)


def render_video(path, fps=30):
    """Real-time mp4: 3D flight of the trained policy beside the distance-to-goal race of all pilots."""
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    pos, goals = runs["trained policy"][0][:, :show], runs["trained policy"][1][:show]
    n_steps, n = pos.shape[0], pos.shape[1]
    stride = max(1, round(1 / (dt * fps)))  # policy runs at 60 Hz; one video frame per `stride` steps
    t_all = torch.arange(n_steps) * dt

    vfig = plt.figure(figsize=(14, 6.5))
    ax3 = vfig.add_subplot(1, 2, 1, projection="3d")
    lo, hi = torch.nan_to_num(pos, nan=0.0).amin((0, 1)) - 0.1, torch.nan_to_num(pos, nan=0.0).amax((0, 1)) + 0.1
    ax3.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(0, max(2.0, float(hi[2]))),
            xlabel="x [m]", ylabel="y [m]", zlabel="height [m]")
    for i in range(n):
        ax3.scatter(*goals[i], color=colors[i % 10], marker="x", s=80, linewidths=2)
    trails = [ax3.plot([], [], [], color=colors[i % 10], lw=1.5, alpha=0.6)[0] for i in range(n)]
    drones = [ax3.plot([], [], [], "o", color=colors[i % 10], ms=8, mec="k")[0] for i in range(n)]
    title = ax3.set_title("")

    axd = vfig.add_subplot(1, 2, 2)
    curves = {}
    for name, (_, _, dist, _, _) in runs.items():
        mean = torch.nanmean(dist, dim=1)
        axd.plot(t_all, mean, linestyle=styles[name], lw=1, alpha=0.25, color="gray")
        curves[name] = (mean, axd.plot([], [], linestyle=styles[name], lw=2.5, label=name)[0])
    cursor = axd.axvline(0, color="k", lw=0.8)
    top = max(float(torch.nan_to_num(m, nan=0.0).max()) for m, _ in curves.values()) * 1.1
    axd.set(xlim=(0, float(t_all[-1])), ylim=(0, top), xlabel="time [s]", ylabel="mean distance to goal [m]",
            title="Distance to goal: trained policy vs baselines")
    axd.legend(loc="upper right")
    axd.grid(alpha=0.3)
    vfig.tight_layout()

    def frame(k):
        s = k * stride
        trail_from = max(0, s - int(1.5 / dt))  # last 1.5 s of path
        for i in range(n):
            seg = pos[trail_from : s + 1, i]
            trails[i].set_data(seg[:, 0], seg[:, 1])
            trails[i].set_3d_properties(seg[:, 2])
            drones[i].set_data(pos[s : s + 1, i, 0], pos[s : s + 1, i, 1])
            drones[i].set_3d_properties(pos[s : s + 1, i, 2])
        for mean, line in curves.values():
            line.set_data(t_all[: s + 1], mean[: s + 1])
        cursor.set_xdata([t_all[s]])
        d = curves["trained policy"][0][s]
        title.set_text(f"Trained policy, t = {t_all[s]:4.1f} s, mean distance {d * 100:4.1f} cm")
        ax3.view_init(elev=22, azim=-60 + 30 * s / n_steps)  # slow orbit for depth
        return [*trails, *drones, *[c[1] for c in curves.values()], cursor, title]

    anim = FuncAnimation(vfig, frame, frames=n_steps // stride, blit=False)
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    print(f"[plot] wrote {path}", flush=True)


if args_cli.video:
    render_video(args_cli.video)

env.close()
simulation_app.close()
