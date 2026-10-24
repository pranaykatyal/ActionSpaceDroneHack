"""Record the defend scenario with chase or a trained checkpoint, labelled, for side-by-side clips.

``scripts/record_clip.py`` always flies greedy pursuit (and ticks the app with bare updates, which step physics without the controller forces). This uses the same viewport capture, ticked through ``sim.render()`` as record_approach.py does,
(livestream mode keeps the viewport alive; frames are stitched before the app closes) but
flies either ``--pilot chase`` or a skrl ``--checkpoint`` (loaded as scripts/evaluate.py
does), and stamps a label plus running interception and breach counts on every frame.

    python -u scripts/record_defend.py --pilot chase --label "chase (hosts' baseline)" \
        --out outputs/videos/defend_chase.mp4 --frame_dir outputs/frames_chase
    python -u scripts/record_defend.py --checkpoint <agent.pt> --label "ours: distilled hybrid" \
        --out outputs/videos/defend_ours.mp4 --frame_dir outputs/frames_ours
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Record a labelled defend-scenario clip.")
parser.add_argument("--task", type=str, default="AS-Defend-v0")
parser.add_argument("--pilot", type=str, default="chase", choices=["chase", "zero"])
parser.add_argument("--checkpoint", type=str, default=None, help="skrl IPPO checkpoint; overrides --pilot.")
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--frames", type=int, default=900, help="900 x 3 steps = 45 s, two or more episodes.")
parser.add_argument("--capture_every", type=int, default=3, help="Env steps (1/60 s) between captures.")
parser.add_argument("--ticks_per_capture", type=int, default=3)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--eye", type=float, nargs=3, default=None, help="camera position (world); default: the arena view, or a side view of the room on room tasks")
parser.add_argument("--lookat", type=float, nargs=3, default=None, help="camera target (world)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--frame_dir", type=str, required=True)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

os.makedirs(args.frame_dir, exist_ok=True)
for stale in os.listdir(args.frame_dir):
    os.remove(os.path.join(args.frame_dir, stale))

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env_cfg.seed = args.seed
env_cfg.viewer.eye = (9.0, 9.0, 6.0)
env_cfg.viewer.lookat = (0.0, 0.0, 1.0)
env = gym.make(args.task, cfg=env_cfg).unwrapped

policy = None
if args.checkpoint:
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    agent_cfg = load_cfg_from_registry(args.task, "skrl_ippo_cfg_entry_point")
    runner = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), agent_cfg)
    runner.agent.load(args.checkpoint)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False)
    policy = runner.agent
label = args.label or (os.path.basename(args.checkpoint) if args.checkpoint else args.pilot)

obs, _ = env.reset()
viewport = get_active_viewport()
eye, lookat = [6.5, 6.5, 6.0], [0.0, 0.0, 0.8]
room_points = None
if hasattr(env, "_ensure_room"):
    # room tasks (team1_defend_room_env.py): outline the room and look at it from beside the left wall
    env._ensure_room()
    lo, hi, site = env._room_lo.tolist(), env._room_hi.tolist(), env._room_site.tolist()
    origin = env.scene.env_origins[0].tolist()

    def to_world(p):
        return [origin[0] + p[0] - site[0], origin[1] + p[1] - site[1], origin[2] + p[2]]

    mid_x, mid_y = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    eye, lookat = to_world([mid_x, lo[1] - 6.5, 6.0]), to_world([mid_x, mid_y, 0.5])
    pts = []
    for i in range(int((hi[0] - lo[0]) / 0.25) + 1):  # floor edges along X
        x = lo[0] + 0.25 * i
        pts += [to_world([x, lo[1], 0.02]), to_world([x, hi[1], 0.02])]
    for i in range(int((hi[1] - lo[1]) / 0.25) + 1):  # floor edges along Y
        y = lo[1] + 0.25 * i
        pts += [to_world([lo[0], y, 0.02]), to_world([hi[0], y, 0.02])]
    for x, y in ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])):  # corner posts up to the ceiling
        pts += [to_world([x, y, 0.25 * k]) for k in range(int(hi[2] / 0.25) + 1)]
    room_points = torch.tensor(pts, device=env.device)
    tag10 = torch.tensor([to_world([lo[0] + 0.1, 0.0, env._rcfg("tag10_height")])], device=env.device)
    room_marks = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/team1_room",
            markers={
                "edge": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75))),
                "tag10": sim_utils.CuboidCfg(size=(0.05, 0.57, 0.57), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95))),
            },
        )
    )
    room_marks.visualize(
        torch.cat([room_points, tag10]),
        marker_indices=torch.cat([torch.zeros(len(room_points), dtype=torch.long), torch.ones(1, dtype=torch.long)]).to(env.device),
    )
if args.eye is not None:
    eye = list(args.eye)
if args.lookat is not None:
    lookat = list(args.lookat)
env.sim.set_camera_view(eye=eye, target=lookat)


def _mat(color):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color)


# The drones are 10 cm across, too small to follow on camera: mark each one with a coloured
# sphere (blue defender, red attacker, grey once intercepted) and outline the protected
# cylinder with a gold disc and three rings of dots (RTX ignores opacity, so no solid shell).
beacons = VisualizationMarkers(
    VisualizationMarkersCfg(
        prim_path="/Visuals/team1_beacons",
        markers={
            "defender": sim_utils.SphereCfg(radius=0.14, visual_material=_mat((0.15, 0.45, 1.0))),
            "attacker": sim_utils.SphereCfg(radius=0.14, visual_material=_mat((1.0, 0.2, 0.15))),
            "down": sim_utils.SphereCfg(radius=0.09, visual_material=_mat((0.5, 0.5, 0.5))),
        },
    )
)
zone = VisualizationMarkers(
    VisualizationMarkersCfg(
        prim_path="/Visuals/team1_zone",
        markers={"disc": sim_utils.CylinderCfg(radius=env.cfg.site_radius, height=0.01, visual_material=_mat((0.95, 0.75, 0.2)))},
    )
)
rings = VisualizationMarkers(
    VisualizationMarkersCfg(
        prim_path="/Visuals/team1_rings",
        markers={"dot": sim_utils.SphereCfg(radius=0.035, visual_material=_mat((0.95, 0.75, 0.2)))},
    )
)
_angle = torch.linspace(0.0, 2.0 * math.pi, 41, device=env.device)[:-1]
_ring = torch.stack([torch.cos(_angle), torch.sin(_angle), torch.zeros_like(_angle)], dim=-1) * env.cfg.site_radius
_ring_points = torch.cat([_ring + torch.tensor([0.0, 0.0, h], device=env.device) for h in (0.6, 1.25, 1.9)])


def draw_markers() -> None:
    origin = env.scene.env_origins[0]
    defenders = torch.stack([d.data.root_pos_w[0] for d in env._defenders])
    attackers = torch.stack([a.data.root_pos_w[0] for a in env._attackers])
    kinds = torch.cat(
        [
            torch.zeros(len(env._defenders), dtype=torch.long, device=env.device),
            torch.where(env._attacker_alive[0], 1, 2).long(),
        ]
    )
    beacons.visualize(torch.cat([defenders, attackers]), marker_indices=kinds)
    zone.visualize((origin + torch.tensor([0.0, 0.0, 0.006], device=env.device)).unsqueeze(0))
    rings.visualize(origin.unsqueeze(0) + _ring_points)


draw_markers()
for _ in range(30):
    env.sim.render()  # pauses physics around the app tick; a bare app update would step physics without the controller forces


def chase_actions(obs):
    """As scripts/evaluate.py and record_clip.py: fly at the nearest attacker slot."""
    actions = {}
    n = env.cfg.num_attackers
    for agent in env.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * n].reshape(env.num_envs, n, 3)
        target = rel[torch.arange(env.num_envs, device=env.device), torch.linalg.norm(rel, dim=-1).argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
    return actions


def capture(index: int) -> str:
    path = os.path.join(args.frame_dir, f"raw_{index:05d}.png")
    capture_viewport_to_file(viewport, path)
    for _ in range(args.ticks_per_capture):
        env.sim.render()  # pauses physics around the app tick; a bare app update would step physics without the controller forces
    return path


intercepts = breaches = 0
alive_prev = env._attacker_alive.clone()
stats: list[tuple[int, int, float]] = []  # per captured frame: (intercepts, breaches, sim time)
captured = step = 0
print(f"[rec] {label}: capturing {args.frames} frames", flush=True)
while captured < args.frames:
    if policy is not None:
        with torch.inference_mode():
            outputs = policy.act(obs, {a: None for a in env.cfg.possible_agents}, timestep=0, timesteps=0)
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.cfg.possible_agents}
    elif args.pilot == "chase":
        actions = chase_actions(obs)
    else:
        actions = {a: torch.zeros(env.num_envs, 4, device=env.device) for a in env.cfg.possible_agents}
    obs, _, _, _, _ = env.step(actions)
    step += 1
    intercepts += int((alive_prev & ~env._attacker_alive).sum())
    alive_prev = env._attacker_alive.clone()
    if bool(env.outcome_valid[0]):
        breached = bool(env.outcome_breached[0])
        breaches += int(breached)
        print(
            f"[rec] episode ended at t={step * env.step_dt:5.1f} s: {'BREACH' if breached else 'timeout, no breach'},"
            f" {int(env.outcome_captures[0])} intercepted this episode",
            flush=True,
        )
    draw_markers()
    env.sim.render()
    if step % args.capture_every == 0:
        path = capture(captured)
        stats.append((intercepts, breaches, step * env.step_dt))
        captured += 1
        if captured == 1:
            for _ in range(60):
                env.sim.render()  # pauses physics around the app tick; a bare app update would step physics without the controller forces
            if not os.path.exists(path):
                print("[rec] first capture produced no file -- aborting", flush=True)
                env.close()
                simulation_app.close()
                sys.exit(1)

for _ in range(120):
    env.sim.render()  # pauses physics around the app tick; a bare app update would step physics without the controller forces

# stamp the label and running counts on every frame
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    small = ImageFont.truetype("DejaVuSans.ttf", 24)
except OSError:
    font = small = ImageFont.load_default()
written = 0
for i, (n_int, n_breach, t) in enumerate(stats):
    raw = os.path.join(args.frame_dir, f"raw_{i:05d}.png")
    if not os.path.exists(raw):
        continue
    img = Image.open(raw).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 92], fill=(0, 0, 0))
    draw.text((16, 8), label, font=font, fill=(255, 255, 255))
    draw.text((16, 52), f"t = {t:4.1f} s    intercepted {n_int}    breaches {n_breach}", font=small, fill=(255, 220, 120))
    # draw.text((img.width - 430, 52), "blue defender   red attacker   grey intercepted", font=small, fill=(200, 200, 200))  # ran off the right edge
    legend = "blue defender   red attacker   grey intercepted"
    legend_w = draw.textlength(legend, font=small) if hasattr(draw, "textlength") else 430
    draw.text((img.width - legend_w - 16, 52), legend, font=small, fill=(200, 200, 200))
    img.save(os.path.join(args.frame_dir, f"frame_{written:05d}.png"))
    written += 1
print(f"[rec] {label}: {written} frames, intercepted {intercepts}, breaches {breaches} in {step * env.step_dt:.1f} s", flush=True)

import imageio_ffmpeg  # noqa: E402

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(args.fps),
    "-i", os.path.join(args.frame_dir, "frame_%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.out,
]
proc = subprocess.run(cmd, capture_output=True, text=True)
ok = proc.returncode == 0 and os.path.exists(args.out)
print(f"[rec] wrote {args.out} ({os.path.getsize(args.out)} bytes)" if ok else f"[rec] ffmpeg failed: {proc.stderr[-600:]}", flush=True)

env.close()
simulation_app.close()  # exits the process
sys.exit(0 if ok else 1)
