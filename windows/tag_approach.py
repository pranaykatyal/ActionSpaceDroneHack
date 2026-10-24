#!/usr/bin/env python3
"""Find an AprilTag, fly to a stand-off in front of it, and hover there, on one Tello from Windows.

Uses the same trained Hover-Real policy as hover_direct.py, but now with all three axes: the
goal is a point --standoff metres in front of the tag along the line of sight, measured from
the drone's own camera (tags.yaml sizes + config/camera/<drone>.yaml). Height comes from the
downward range sensor, its goal from the tag's height in the image. Yaw is a small P loop
that keeps the tag centred (the policy was trained with yaw locked).

  1. SEARCH   take off, climb to --height, turn slowly until --tag is seen
  2. APPROACH the policy flies toward the stand-off point, yaw keeps the tag centred
  3. HOLD     within --tol of the point for --hold seconds, then land

  conda activate tello
  python windows\\tag_approach.py --ssid TELLO-3 --tag 8 --dry     # no takeoff: carry it, watch the commands
  python windows\\tag_approach.py --ssid TELLO-3 --tag 8           # fly

The policy runs in numpy from windows/policies/hover_real_R1.npz, so this needs no torch
(the tello env has the camera and AprilTag packages). Ctrl-C or q in the video window
lands. It also lands on: search timeout, range reading lost for 1 s, above --ceiling,
--timeout overall, low battery. The tag is lost for more than 2 s -> back to SEARCH.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Action_Space" / "drone_deployment_infra" / "cal_and_map"))
sys.path.insert(0, str(HERE))

from tello_calibration.calib.io import camera_from_dict, load_yaml  # noqa: E402
from tello_calibration.tags.detect import TagDetector, load_tag_config  # noqa: E402
from tello_calibration.utils.config import CONFIG_DIR  # noqa: E402
from tello_check import TELLO_IP, join, local_ip_toward_tello  # noqa: E402

# m/s per rc unit (control/tello_motion_fit.yaml)
RIGHT_GAIN, FWD_GAIN, UP_GAIN = 0.008, 0.00731, 0.006


class NumpyPolicy:
    """hover_runner.HoverPolicy without torch: 12 -> 64 -> 64 (elu) -> 4, skrl's observation scaler."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.layers = [(d["w0"], d["b0"]), (d["w1"], d["b1"])]
        self.out = (d["wo"], d["bo"])
        self.mean, self.std = d["mean"], np.sqrt(d["var"] + 1e-8)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - self.mean) / self.std, -5.0, 5.0)
        for w, b in self.layers:
            x = x @ w.T + b
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # elu
        return np.clip(x @ self.out[0].T + self.out[1], -1.0, 1.0)


def rot(yaw_deg: float) -> np.ndarray:
    """2D rotation by yaw (counter-clockwise positive, i.e. minus the Tello's clockwise yaw)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s], [s, c]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssid", help="drone WiFi to join first, e.g. TELLO-3 (omit if already joined)")
    ap.add_argument("--iface", default="Wi-Fi 2", help="dongle adapter (netsh wlan show interfaces)")
    ap.add_argument("--tag", type=int, required=True, help="tag id to fly to")
    ap.add_argument("--camera", type=Path, help="default: config/camera/<drone>.yaml")
    ap.add_argument("--checkpoint", type=Path, default=HERE / "policies/hover_real_R1.npz")
    ap.add_argument("--standoff", type=float, default=0.5, help="hover this far in front of the tag [m]")
    ap.add_argument("--height", type=float, default=1.0, help="search height above the floor [m]")
    ap.add_argument("--hold", type=float, default=10.0, help="seconds to hold at the stand-off, then land")
    ap.add_argument("--tol", type=float, default=0.10, help="counts as arrived within this [m]")
    ap.add_argument("--act-scale", type=float, default=0.3, help="m/s at action 1 (the Real task's speed)")
    ap.add_argument("--max-speed", type=float, default=0.3, help="horizontal speed cap [m/s]")
    ap.add_argument("--max-climb", type=float, default=0.3, help="vertical speed cap [m/s]")
    ap.add_argument("--search-yaw", type=int, default=25, help="rc yaw while searching (+ = clockwise)")
    ap.add_argument("--search-time", type=float, default=40.0, help="land if the tag is not found in this [s]")
    ap.add_argument("--yaw-gain", type=float, default=1.2, help="rc yaw per degree of tag bearing")
    ap.add_argument("--ceiling", type=float, default=1.8, help="land if the range reading exceeds this [m]")
    ap.add_argument("--timeout", type=float, default=90.0, help="land after this long in the air [s]")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--min-battery", type=int, default=30)
    ap.add_argument("--no-window", action="store_true", help="no video window")
    ap.add_argument("--dry", action="store_true", help="no takeoff: video, detections and the commands it would send")
    args = ap.parse_args()

    import cv2

    if args.ssid:
        join(args.ssid, args.iface)
    elif not (local_ip_toward_tello() or "").startswith("192.168.10."):
        raise SystemExit("not on a Tello network: pass --ssid TELLO-3, or join it in Windows WiFi settings")
    key = (args.ssid or "tello").lower().replace("-", "_")
    camera_path = args.camera or CONFIG_DIR / "camera" / f"{key}.yaml"
    if not camera_path.exists():
        raise SystemExit(f"no camera calibration at {camera_path}; pass --camera")
    K, dist = camera_from_dict(load_yaml(camera_path))
    tags = load_tag_config(CONFIG_DIR / "tags.yaml")
    if tags.size_m(args.tag) is None:
        raise SystemExit(f"tag {args.tag} has no size in tags.yaml")
    detector = TagDetector(tags, K, dist)
    half_fov = math.degrees(math.atan(K[0, 2] / K[0, 0]))
    fit = tags.size_m(args.tag) / (2 * math.tan(math.radians(half_fov)))
    if args.standoff < 1.3 * fit:
        print(f"NOTE: tag {args.tag} ({tags.size_m(args.tag) * 1000:.0f} mm) only fits the image from "
              f"{fit:.2f} m; at --standoff {args.standoff} it may be cut off. Try --standoff {1.3 * fit:.1f}", flush=True)
    policy = NumpyPolicy(args.checkpoint)

    from djitellopy import Tello

    tello = Tello(TELLO_IP)
    tello.connect()
    time.sleep(0.5)
    tello.get_own_udp_object()["responses"].clear()  # drop a late duplicate "ok" (see tello_check.py)
    battery = tello.get_battery()
    print(f"[link] connected, battery {battery}%, camera {camera_path.name}", flush=True)
    if battery < args.min_battery and not args.dry:
        print(f"battery below {args.min_battery}%: charge first")
        return 1
    tello.streamon()
    frames = tello.get_frame_read()

    log_dir = HERE / "flight_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"tag_approach_{args.ssid or 'tello'}_tag{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry else ''}.jsonl"
    log = log_path.open("w")
    log.write(json.dumps({"kind": "meta", **{k: str(v) for k, v in vars(args).items()},
                          "note": "tag_b = tag in body frame [fwd, left, up] m; goal_b = error to the stand-off point"}) + "\n")
    print(f"logging to {log_path}", flush=True)

    def tof_m():
        cm = tello.get_distance_tof()
        return cm / 100.0 if 5 <= cm < 400 else None

    flying = False
    phase, phase_t = "SEARCH", time.time()
    z_goal = args.height
    seen_t, held_since = 0.0, None
    track: list[tuple[float, np.ndarray]] = []  # (t, tag position in the takeoff-yaw frame) for velocity
    z_hist: list[tuple[float, float]] = []
    t_start, last_print, last_tof_ok, period = time.time(), 0.0, time.time(), 1.0 / args.hz
    try:
        if not args.dry:
            tello.takeoff()  # it climbs to ~0.8 m by itself
            flying = True
            print("airborne: SEARCH", flush=True)
        while True:
            t0 = time.time()
            if t0 - t_start > args.timeout:
                print("timeout: landing", flush=True)
                break
            state = tello.get_current_state()
            yaw = float(state.get("yaw", 0))  # degrees, clockwise positive
            z = tof_m()
            if z is not None:
                last_tof_ok = t0
                z_hist = (z_hist + [(t0, z)])[-4:]
            elif flying and t0 - last_tof_ok > 1.0:
                print("range reading lost for 1 s: landing", flush=True)
                break
            if z is not None and z > args.ceiling:
                print(f"{z:.2f} m above the ceiling {args.ceiling} m: landing", flush=True)
                break
            vz = (z_hist[-1][1] - z_hist[0][1]) / max(z_hist[-1][0] - z_hist[0][0], 1e-3) if len(z_hist) > 1 else 0.0

            frame = frames.frame
            obs_tag = None
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                for o in detector.detect(gray):
                    if o.tag_id == args.tag and o.accepted and o.tvec is not None:
                        obs_tag = o
            tag_b = None
            if obs_tag is not None:
                x, y, zc = obs_tag.tvec  # camera: x right, y down, z forward
                tag_b = np.array([zc, -x, -y])  # body: forward, left, up
                seen_t = t0
                # velocity from the tag's track in a frame that does not turn with the drone
                fixed = rot(-yaw) @ tag_b[:2]
                track = [(t, p) for t, p in track if t0 - t < 0.4] + [(t0, fixed)]
                # height goal: bring the camera level with the tag (within the range sensor's limits)
                if z is not None:
                    z_goal = float(np.clip(0.8 * z_goal + 0.2 * (z + tag_b[2]), 0.5, args.ceiling - 0.2))
                if phase == "SEARCH":
                    phase, phase_t = "APPROACH", t0
                    print(f"tag {args.tag} found at {np.linalg.norm(tag_b[:2]):.2f} m: APPROACH", flush=True)
            elif phase != "SEARCH" and t0 - seen_t > 2.0:
                phase, phase_t, held_since, track = "SEARCH", t0, None, []
                print("tag lost for 2 s: SEARCH", flush=True)
            if phase == "SEARCH" and t0 - phase_t > args.search_time:
                print(f"tag {args.tag} not found in {args.search_time:.0f} s: landing", flush=True)
                break

            # velocity of the drone in the body frame = minus the tag's apparent motion
            v_b = np.zeros(2)
            if len(track) > 2 and track[-1][0] - track[0][0] > 0.1:
                v_fixed = -(track[-1][1] - track[0][1]) / (track[-1][0] - track[0][0])
                v_b = rot(yaw) @ v_fixed

            err = np.zeros(3)
            yaw_rc = 0
            if phase == "SEARCH":
                yaw_rc = args.search_yaw
            elif tag_b is not None:
                rng = float(np.linalg.norm(tag_b[:2]))
                goal_xy = tag_b[:2] * (1.0 - args.standoff / max(rng, 1e-3))  # stop short on the line of sight
                n = np.linalg.norm(goal_xy)
                err[:2] = goal_xy if n <= 1.0 else goal_xy / n  # the policy was trained on errors up to ~1 m
                bearing = math.degrees(math.atan2(-tag_b[1], tag_b[0]))  # + = tag to the right
                yaw_rc = int(np.clip(args.yaw_gain * bearing, -30, 30))
                if n < args.tol and abs(bearing) < 8:
                    held_since = held_since or t0
                    if phase == "APPROACH":
                        phase = "HOLD"
                        print(f"at the stand-off ({rng:.2f} m from the tag): HOLD {args.hold:.0f} s", flush=True)
                elif n > 2 * args.tol:
                    held_since = None
                if held_since and t0 - held_since > args.hold:
                    print("held: landing", flush=True)
                    break
            if z is not None:
                err[2] = z_goal - z
            act = policy(np.array([*err, *v_b, vz, 0, 0, 0, 0, 0, -1.0]))
            if phase == "SEARCH" or tag_b is None:
                act[:2] = 0.0  # no horizontal goal without the tag
            v_cmd = np.clip(act[:2] * args.act_scale, -args.max_speed, args.max_speed)
            if tag_b is not None and np.linalg.norm(tag_b[:2]) < 0.3:
                v_cmd[0] = min(v_cmd[0], -0.1)  # too close: back off whatever the policy says
            up_cmd = float(np.clip(act[2] * args.act_scale, -args.max_climb, args.max_climb)) if z is not None else 0.0
            rc = [int(np.clip(round(-v_cmd[1] / RIGHT_GAIN), -100, 100)),
                  int(np.clip(round(v_cmd[0] / FWD_GAIN), -100, 100)),
                  int(np.clip(round(up_cmd / UP_GAIN), -100, 100)),
                  int(yaw_rc)]
            if flying:
                tello.send_rc_control(*rc)

            log.write(json.dumps({"kind": "ctl", "t": t0, "phase": phase, "z": z, "z_goal": round(z_goal, 3),
                                  "tag_b": None if tag_b is None else tag_b.round(3).tolist(),
                                  "goal_b": err.round(3).tolist(), "v_b": v_b.round(3).tolist(), "vz": round(vz, 3),
                                  "act": act.round(4).tolist(), "rc": rc, "state": state}) + "\n")
            if t0 - last_print > 0.5:
                where = "no tag" if tag_b is None else (f"tag fwd {tag_b[0]:+.2f} left {tag_b[1]:+.2f} up {tag_b[2]:+.2f}")
                print(f"{phase:8s} {where}  z {z}  -> err {err.round(2).tolist()}  rc {rc}", flush=True)
                last_print = t0
            if not args.no_window and frame is not None:
                view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if obs_tag is not None:
                    cv2.polylines(view, [obs_tag.corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.putText(view, f"{phase} tag {args.tag} rc {rc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("tag_approach (q = land)", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("q: landing", flush=True)
                    break
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nCtrl-C", flush=True)
    finally:
        if flying:
            tello.send_rc_control(0, 0, 0, 0)
            try:
                tello.land()
                print("landed", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"LAND FAILED ({exc}); it auto-lands ~15 s after the last command", flush=True)
        log.close()
        try:
            tello.streamoff()
        except Exception:  # noqa: BLE001
            pass
        tello.end()
        cv2.destroyAllWindows()
        print(f"log: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
