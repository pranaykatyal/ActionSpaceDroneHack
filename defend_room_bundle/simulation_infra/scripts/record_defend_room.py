"""Record a video of a trained AS-Defend-Room-v0 policy (skrl MAPPO/IPPO checkpoint).

``evaluate.py --video`` fails headless (Replicator "Unable to write from unknown dtype"),
so this captures Kit viewport frames and stitches them with ffmpeg, like record_approach.py.
Blue = defenders, red = attackers, grey = caught attackers; the dark patch is the corridor
and the yellow wall the breach line at tag 6.

    python -u scripts/record_defend_room.py --checkpoint outputs/defend_best/checkpoint.pt \
        --episodes 3 --out outputs/videos/defend_room.mp4 > outputs/record_defend.log 2>&1
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--algorithm", default="MAPPO", choices=["MAPPO", "IPPO"])
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default="outputs/videos/defend_room.mp4")
parser.add_argument("--frame_dir", default="outputs/frames_defend")
parser.add_argument("--ticks_per_capture", type=int, default=3, help="App ticks after each capture.")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

# Livestream mode keeps the viewport pipeline alive in headless mode (see record_approach.py).
app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
from action_space_kit.tasks.direct.tello_defend_room.tello_defend_room_env_cfg import TelloDefendRoomEnvCfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

TASK = "AS-Defend-Room-v0"


def colour(rgb, opacity=1.0):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, opacity=opacity)


def main() -> int:
    os.makedirs(args.frame_dir, exist_ok=True)
    for stale in os.listdir(args.frame_dir):
        os.remove(os.path.join(args.frame_dir, stale))

    cfg = TelloDefendRoomEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.device = "cuda:0"
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped

    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from skrl.utils.runner.torch import Runner

    agent_cfg = load_cfg_from_registry(TASK, f"skrl_{args.algorithm.lower()}_cfg_entry_point")
    runner = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), agent_cfg)
    runner.agent.load(args.checkpoint)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False)
    policy = runner.agent
    print(f"[rec] policy: {args.checkpoint}", flush=True)

    # 0 defender, 1 attacker, 2 caught attacker, 3 corridor floor, 4 breach line
    c = u.cfg
    width = c.arena_y[1] - c.arena_y[0]
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/record",
            markers={
                "defender": sim_utils.SphereCfg(radius=0.12, visual_material=colour((0.1, 0.4, 1.0))),
                "attacker": sim_utils.SphereCfg(radius=0.12, visual_material=colour((1.0, 0.15, 0.1))),
                "caught": sim_utils.SphereCfg(radius=0.08, visual_material=colour((0.5, 0.5, 0.5))),
                "floor": sim_utils.CuboidCfg(size=(c.arena_x[1], width, 0.01), visual_material=colour((0.15, 0.15, 0.2))),
                "breach": sim_utils.CuboidCfg(size=(0.03, width, c.arena_height), visual_material=colour((1.0, 0.85, 0.1), 0.35)),
            },
        )
    )
    origin = u.scene.env_origins[0]
    mid_y = 0.5 * (c.arena_y[0] + c.arena_y[1])
    fixed = torch.tensor(
        [[0.5 * c.arena_x[1], mid_y, 0.005], [c.breach_x, mid_y, 0.5 * c.arena_height]], device=u.device
    ) + origin

    def draw():
        d = torch.stack([dr.data.root_pos_w[0] for dr in u._defenders])
        a = torch.stack([dr.data.root_pos_w[0] for dr in u._attackers])
        alive = u._attacker_alive[0]
        a_idx = torch.where(alive, torch.ones_like(alive, dtype=torch.long), torch.full_like(alive, 2, dtype=torch.long))
        pos = torch.cat([d, a, fixed])
        idx = torch.cat([torch.zeros(len(d), dtype=torch.long, device=u.device), a_idx,
                         torch.tensor([3, 4], device=u.device)])  # fmt: skip
        markers.visualize(translations=pos, marker_indices=idx)

    obs, _ = env.reset()
    viewport = get_active_viewport()
    look = (origin + torch.tensor([0.5 * c.breach_x, mid_y, 0.6], device=u.device)).tolist()
    eye = (origin + torch.tensor([0.5 * c.breach_x, mid_y - 6.5, 4.5], device=u.device)).tolist()
    u.sim.set_camera_view(eye=eye, target=look)
    draw()
    for _ in range(30):
        u.sim.render()  # renders without stepping physics

    frame = 0

    def capture():
        nonlocal frame
        capture_viewport_to_file(viewport, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
        for _ in range(args.ticks_per_capture):
            u.sim.render()
        frame += 1

    spans = []
    for ep in range(args.episodes):
        first = frame
        captures_before = 0.0
        while True:
            with torch.inference_mode():
                states = {agent: None for agent in c.possible_agents}
                outputs = policy.act(obs, states, timestep=0, timesteps=0)
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in c.possible_agents}
            obs, _, _, _, _ = env.step(actions)
            if bool(u.outcome_valid[0]):
                break  # already auto-reset: this frame would show the next episode
            draw()
            u.sim.render()
            capture()
        result = (
            f"episode {ep + 1}: caught {int(u.outcome_captures[0])}, "
            f"{'BREACH' if bool(u.outcome_breached[0]) else 'no breach'}, "
            f"{'CRASH' if bool(u.outcome_crashed[0]) else 'no crash'}"
        )
        print(f"[rec] {result}", flush=True)
        spans.append((first, frame - 1, result))

    for _ in range(120):
        u.sim.render()
    frames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".png"))
    print(f"[rec] wrote {len(frames)} frames", flush=True)
    if not frames:
        return 1

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    for a, b, result in spans:
        for i in range(a, b + 1):
            path = os.path.join(args.frame_dir, f"frame_{i:05d}.png")
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            draw_ = ImageDraw.Draw(img)
            lines = ["blue: defenders   red: attackers   yellow: breach line (tag 6)"]
            if i > b - int(1.5 / u.step_dt):
                lines.append(result)
            y = 14
            for line in lines:
                draw_.rectangle(draw_.textbbox((14, y), line, font=font), fill=(0, 0, 0))
                draw_.text((14, y), line, font=font, fill=(255, 255, 255))
                y += 36
            img.save(path)

    import imageio_ffmpeg

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fps = round(1.0 / u.step_dt)  # one frame per policy step: real time
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps),
        "-i", os.path.join(args.frame_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.out,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(args.out):
        print(f"[rec] ffmpeg failed: {proc.stderr[-600:]}", flush=True)
        return 1
    print(f"[rec] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        # with livestream on, simulation_app.close() can hang after the video is written
        sys.stdout.flush()
        os._exit(code)
