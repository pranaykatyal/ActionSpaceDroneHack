#!/usr/bin/env python3
"""Fly a trained AS-Tello-Hover-v0 policy on a real Tello: the first sim-to-real check.

The policy holds a goal point. Here the goal is wherever the drone is when the runner first
sees a fresh pose (or --goal X Y Z in the guard frame), so after takeoff in drone_gui it
should just hold still and pull back when nudged. Same plumbing as guard_runner.py:
drone_gui telemetry -> map.yaml room frame -> policy -> rc sticks via drone_gui's API.
Dry run by default; --fly sends sticks.

  python hover_runner.py                      # dry run: prints obs and sticks
  python hover_runner.py --fly                # mission: take off, hold start + 0.5 m up for 15 s, land
  python hover_runner.py --fly --rise 0.5 --hold 15

Mission (--fly): with TELLO-7 on the floor facing the tag-10 wall, the runner reads its ground
pose, takes off through drone_gui, holds the goal = ground pose + --rise metres up for --hold
seconds, then lands. Ctrl-C lands too. A Tello's own takeoff climbs to ~0.8 m, so expect the
policy to bring it down to ~0.6 m. With no fresh pose for 0.5 s it hovers (zero sticks).

Observation (tello_hover_env.py): goal error (world, 3), lin vel (world, 3), ang vel (body, 3),
projected gravity (body, 3). Telemetry has no attitude, so the last two are fed their
level-hover values (0, and (0, 0, -1)); hover is near level, so this is what the policy saw
there. The policy acts at 60 Hz in the sim; here it runs at --hz with the drone's ~0.5 s
lag, so its velocity commands are capped at --max-speed.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_runner as gr  # noqa: E402

HOVER_RUN = gr.REPO / "Action_Space/simulation_infra/logs/skrl/tello_hover/2026-10-23_22-53-49_ppo_torch_tello_hover_ppo"


class HoverPolicy:
    """skrl Gaussian policy (shared net: 12 -> 64 -> 64, elu) and its observation scaler, in plain torch."""

    def __init__(self, path: Path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p, s = ck["policy"], ck["observation_preprocessor"]
        self.layers = [(p["net_container.0.weight"], p["net_container.0.bias"]),
                       (p["net_container.2.weight"], p["net_container.2.bias"])]
        self.out = (p["policy_layer.weight"], p["policy_layer.bias"])
        self.mean, self.var = s["running_mean"].float(), s["running_variance"].float()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.tensor(obs, dtype=torch.float32)
        x = ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-5.0, 5.0)  # skrl RunningStandardScaler
        for w, b in self.layers:
            x = torch.nn.functional.elu(x @ w.T + b)
        return (x @ self.out[0].T + self.out[1]).clamp(-1.0, 1.0).numpy()  # mean action, deterministic


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone", default="TELLO-7")
    ap.add_argument("--checkpoint", type=Path, default=HOVER_RUN / "checkpoints/best_agent.pt")
    ap.add_argument("--rise", type=float, default=0.5, help="goal height above the ground pose [m]")
    ap.add_argument("--hold", type=float, default=15.0, help="seconds to hold after takeoff, then land")
    ap.add_argument("--fly", action="store_true")
    ap.add_argument("--gui", default="http://127.0.0.1:8780")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--map", type=Path, default=gr.MAP)
    ap.add_argument("--tag10-height", type=float, default=1.22)
    ap.add_argument("--max-speed", type=float, default=0.3, help="cap on commanded speed [m/s]")
    ap.add_argument("--max-error", type=float, default=1.5, help="farther than this from the goal: hover and warn [m]")
    args = ap.parse_args()

    policy = HoverPolicy(args.checkpoint)
    # reuse the guard runner's tracking and stick conversion (heading hold, caps, geofence)
    gargs = gr.argparse.Namespace(goal_tag=6, tag10_height=args.tag10_height, max_speed=args.max_speed, max_climb=0.3,
                                  min_height=0.3, max_height=2.0, box=[0.6, 7.0, -2.0, 3.5], yaw_gain=0.5,
                                  defenders=[args.drone])
    pilot = gr.GuardPilot(gargs, gr.load_map(args.map))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)
    gui = gr.Gui(args.gui) if args.fly else None
    iface = None
    if gui:
        iface = {d["name"]: d["iface"] for d in gui.state()}.get(args.drone)
        if iface is None:
            print(f"drone_gui has no {args.drone}")
            return 1
    print(("FLYING " if gui else "dry run ") + f"{args.drone} with {args.checkpoint.name}", flush=True)
    msg, period = None, 1.0 / args.hz
    # Flight log for later motion fits and training: every telemetry entry of this drone
    # ("tel": pose, tof, h, vel, bat as drone_gui sent them) and every control tick ("ctl":
    # tracked state, goal, policy action, sticks sent), one JSON object per line.
    log_dir = Path(__file__).resolve().parent / "flight_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"hover_{args.drone}_{time.strftime('%Y%m%d_%H%M%S')}{'' if gui else '_dry'}.jsonl"
    log = log_path.open("w")
    log.write(json.dumps({"kind": "meta", "drone": args.drone, "checkpoint": str(args.checkpoint), "fly": bool(gui),
                          "hz": args.hz, "max_speed": args.max_speed, "rise": args.rise, "hold": args.hold,
                          "rc_gain_mps": pilot.gain.tolist(), "frame": "guard: X out of tag-10 wall, Y along, Z up"}) + "\n")
    print(f"logging to {log_path}", flush=True)

    def receive():
        nonlocal msg
        try:
            while True:
                m = json.loads(sock.recvfrom(65535)[0])
                if m.get("v") == 1:
                    msg = m
                    pilot.ingest(msg)
                    for d in m["drones"]:
                        if d["name"] == args.drone:
                            log.write(json.dumps({"kind": "tel", "t_recv": time.time(), "t": m["t"], "seq": m["seq"], **d}) + "\n")
        except BlockingIOError:
            pass

    def log_ctl(sticks, goal, act=None):
        tr = pilot.tracks.get(args.drone)
        log.write(json.dumps({
            "kind": "ctl", "t": time.time(), "sticks": list(sticks), "goal": goal.round(3).tolist(),
            "act": None if act is None else act.round(4).tolist(),
            "pos": None if tr is None or tr.pos is None else tr.pos.round(3).tolist(),
            "vel": None if tr is None else tr.vel.round(3).tolist(),
            "heading": None if tr is None else tr.heading,
        }) + "\n")

    def fresh_track():
        tr = pilot.tracks.get(args.drone)
        return tr if tr and tr.heading is not None and time.time() - tr.t < gr.STALE_S else None

    # 1. ground pose (the drone must see a wall tag, e.g. tag 10, from the floor)
    deadline = time.time() + 15.0
    while fresh_track() is None:
        receive()
        if time.time() > deadline:
            print(f"no fresh wall-tag pose for {args.drone} in 15 s: point it at tag 10 and retry")
            return 1
        time.sleep(0.05)
    goal = fresh_track().pos.copy()
    goal[2] += args.rise
    print(f"ground pose {fresh_track().pos.round(2)} -> goal {goal.round(2)} (guard frame: X out of the wall, Z up)", flush=True)

    # 2. take off through drone_gui and wait until it reports flying
    if gui:
        gui.post(iface, "takeoff")
        deadline = time.time() + 15.0
        while not next((d["flying"] and not d["busy"] for d in gui.state() if d["name"] == args.drone), False):
            if time.time() > deadline:
                print("takeoff did not complete in 15 s; landing")
                gui.post(iface, "land")
                return 1
            time.sleep(0.2)
        print("airborne", flush=True)

    # 3. hold the goal for --hold seconds, then 4. land
    t_end, last_print = time.time() + args.hold, 0.0
    try:
        while time.time() < t_end:
            t0 = time.time()
            receive()
            tr = fresh_track()
            sticks, note, act = (0, 0, 0, 0), "no fresh pose: hover", None
            if tr is not None:
                err = goal - tr.pos
                if np.linalg.norm(err) > args.max_error:
                    note = f"{np.linalg.norm(err):.2f} m from goal: hover (too far for this test)"
                else:
                    obs = np.concatenate([err, tr.vel, np.zeros(3), [0.0, 0.0, -1.0]])
                    act = policy(obs)
                    vel = act[:3] * 1.0  # sim max_lin_speed = 1.0 m/s per axis
                    vel = vel * min(1.0, args.max_speed / max(np.abs(vel).max(), 1e-6))
                    sticks = pilot.sticks(tr, vel)
                    note = f"err {err.round(2)} vel {tr.vel.round(2)} act {act.round(2)}"
            if gui:
                try:
                    gui.post(iface, "rc", {"sticks": list(sticks)})
                except OSError as exc:
                    print(f"rc post failed ({exc})", flush=True)
            log_ctl(sticks, goal, act)
            if t0 - last_print > 0.5:
                print(f"t-{t_end - t0:4.1f}s {note} -> sticks {sticks}", flush=True)
                last_print = t0
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nCtrl-C", flush=True)
    finally:
        if gui:
            try:
                gui.post(iface, "rc", {"sticks": [0, 0, 0, 0]})
                gui.post(iface, "land")
                print("landing", flush=True)
            except OSError as exc:
                print(f"LAND FAILED ({exc}): press LAND in drone_gui", flush=True)
            # keep logging the landing: it shows how the drone settles and touches down
            t_land = time.time() + 5.0
            while time.time() < t_land:
                receive()
                time.sleep(0.05)
        log.close()
        print(f"log: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
