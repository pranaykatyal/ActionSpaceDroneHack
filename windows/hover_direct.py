#!/usr/bin/env python3
"""Hover test with our trained policy on one Tello, straight from Windows (no drone_gui, no tags).

Joins the drone's WiFi on the USB dongle, takes off, flies height only with the Hover
policy (from the downward range sensor) to the ground reading + --rise metres, holds it for
--hold seconds, then lands. x/y sticks stay zero, so the Tello's own position hold (IMU +
downward camera) keeps it in place. Every tick logs the drone's full SDK state (attitude,
velocities, accelerations, tof, baro, battery) plus the policy's action and the sticks sent,
to windows/flight_logs/hover_direct_*.jsonl, for later motion fits and training.

  conda activate isaaclab
  python windows\\hover_direct.py --ssid TELLO-7 --dry     # connect + read sensors, no takeoff
  python windows\\hover_direct.py --ssid TELLO-7           # take off, hold +0.5 m for 15 s, land

Defaults fly windows/policies/hover_real_R1.pt (AS-Tello-Hover-Real-v0). For an AS-Tello-Hover-v0
policy pass --checkpoint <run>\\checkpoints\\best_agent.pt --act-scale 1.0 --vel-window 1.
See windows/HOVER_README.md.

Ctrl-C lands. It also lands if the range reading is lost for 1 s or the drone climbs past
--ceiling. Needs a clear 1 m around the drone and a textured floor (the Tello's position hold
uses its downward camera; a plain glossy floor makes it drift).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hover_runner import HoverPolicy  # noqa: E402
from tello_check import TELLO_IP, join, local_ip_toward_tello  # noqa: E402

UP_GAIN = 0.006  # m/s per rc unit, up channel (control/tello_motion_fit.yaml)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssid", help="drone WiFi to join first, e.g. TELLO-7 (omit if already joined)")
    ap.add_argument("--iface", default="Wi-Fi 2", help="dongle adapter (netsh wlan show interfaces)")
    ap.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "policies/hover_real_R1.pt")
    ap.add_argument("--rise", type=float, default=0.5, help="hold this far above the ground reading [m]")
    ap.add_argument("--hold", type=float, default=15.0, help="seconds to hold, then land")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--max-climb", type=float, default=0.3, help="vertical speed cap [m/s]")
    ap.add_argument("--act-scale", type=float, default=0.3,
                    help="m/s at action 1: 1.0 for AS-Tello-Hover-v0 policies, 0.3 for AS-Tello-Hover-Real-v0")
    ap.add_argument("--vel-window", type=int, default=3,
                    help="difference the height over this many ticks for the velocity (3 matches the Real task's noise)")
    ap.add_argument("--ceiling", type=float, default=1.6, help="land if the range reading exceeds this [m]")
    ap.add_argument("--min-battery", type=int, default=30)
    ap.add_argument("--dry", action="store_true", help="connect and print sensors and policy output; no takeoff")
    args = ap.parse_args()

    if args.ssid:
        join(args.ssid, args.iface)
    elif not (local_ip_toward_tello() or "").startswith("192.168.10."):
        raise SystemExit("not on a Tello network: pass --ssid TELLO-7, or join it in Windows WiFi settings")

    from djitellopy import Tello

    policy = HoverPolicy(args.checkpoint)
    tello = Tello(TELLO_IP)
    tello.connect()
    time.sleep(0.5)
    tello.get_own_udp_object()["responses"].clear()  # drop a late duplicate "ok" (see tello_check.py)
    battery = tello.get_battery()
    print(f"[link] connected, battery {battery}%", flush=True)
    if battery < args.min_battery and not args.dry:
        print(f"battery below {args.min_battery}%: charge first")
        return 1

    log_dir = Path(__file__).resolve().parent / "flight_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"hover_direct_{args.ssid or 'tello'}_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry else ''}.jsonl"
    log = log_path.open("w")
    log.write(json.dumps({"kind": "meta", "ssid": args.ssid, "checkpoint": str(args.checkpoint), "hz": args.hz,
                          "rise": args.rise, "hold": args.hold, "max_climb": args.max_climb, "up_gain": UP_GAIN,
                          "note": "state = raw Tello SDK state (tof cm, h cm, vgx/vgy/vgz dm/s, agx.. 0.001 g)"}) + "\n")
    print(f"logging to {log_path}", flush=True)

    def tof_m():
        cm = tello.get_distance_tof()
        return cm / 100.0 if 5 <= cm < 400 else None

    ground = tof_m()
    if ground is None:
        print("no valid range reading on the ground; is the drone flat on the floor?")
        return 1
    goal = ground + args.rise
    print(f"ground {ground:.2f} m -> hold at {goal:.2f} m for {args.hold:.0f} s", flush=True)

    flying = False
    try:
        if not args.dry:
            tello.takeoff()  # blocks until the Tello answers (it climbs to ~0.8 m by itself)
            flying = True
            print("airborne", flush=True)
        z_prev, t_prev, vz, last_ok = None, None, 0.0, time.time()
        history: list[tuple[float, float]] = []  # (t, z) for --vel-window
        t_end, last_print, period = time.time() + args.hold, 0.0, 1.0 / args.hz
        while time.time() < t_end:
            t0 = time.time()
            z = tof_m()
            up, act = 0, None
            if z is None:
                if t0 - last_ok > 1.0:
                    print("range reading lost for 1 s: landing", flush=True)
                    break
            else:
                last_ok = t0
                if z > args.ceiling:
                    print(f"{z:.2f} m above the ceiling {args.ceiling} m: landing", flush=True)
                    break
                if args.vel_window > 1:
                    history = (history + [(t0, z)])[-(args.vel_window + 1):]
                    if len(history) > 1:
                        vz = (history[-1][1] - history[0][1]) / max(history[-1][0] - history[0][0], 1e-3)
                else:
                    if z_prev is not None and z != z_prev:
                        vz = 0.6 * vz + 0.4 * (z - z_prev) / max(t0 - t_prev, 1e-3)
                    if z != z_prev:
                        z_prev, t_prev = z, t0
                act = policy(np.array([0.0, 0.0, goal - z, 0.0, 0.0, vz, 0, 0, 0, 0, 0, -1.0]))
                cmd = float(np.clip(act[2] * args.act_scale, -args.max_climb, args.max_climb))  # height only
                up = int(np.clip(round(cmd / UP_GAIN), -100, 100))
            if flying:
                tello.send_rc_control(0, 0, up, 0)
            log.write(json.dumps({"kind": "ctl", "t": t0, "z": z, "vz": round(vz, 4), "goal": goal,
                                  "act": None if act is None else act.round(4).tolist(), "rc": [0, 0, up, 0],
                                  "state": tello.get_current_state()}) + "\n")
            if t0 - last_print > 0.5:
                print(f"t-{t_end - t0:4.1f}s  z {z if z is None else round(z, 2)}  vz {vz:+.2f}  "
                      f"act_z {None if act is None else round(float(act[2]), 2)}  rc up {up}", flush=True)
                last_print = t0
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
            for _ in range(30):  # keep logging the touchdown
                log.write(json.dumps({"kind": "post", "t": time.time(), "state": tello.get_current_state()}) + "\n")
                time.sleep(0.1)
        log.close()
        tello.end()
        print(f"log: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
