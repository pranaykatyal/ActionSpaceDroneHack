#!/usr/bin/env python3
"""Fly real defenders with the scripted intercept guard (simulation_infra .../tello_defend/guard.py).

Reads drone_gui's fleet telemetry (UDP JSON, drone_telemetry/README.md), puts every drone into
one room frame with config/map.yaml, runs the guard, and steers each defender through
drone_gui's HTTP API (POST /api/<iface>/rc). drone_gui keeps doing takeoff, landing, the 20 Hz
rc keepalive and its own safety rules; this script only steers. Run it on the machine that
runs drone_gui (its server listens on 127.0.0.1), and start drone_gui with
TELLO_MANUAL_MAX=100, or the GUI caps the sticks at 40 (~0.3 m/s).

Dry run by default: it prints the sticks it would send. --fly sends them.

  python guard_runner.py --selftest                  # offline: synthetic attackers, no network
  python guard_runner.py --defenders TELLO-7         # dry run on live telemetry
  python guard_runner.py --defenders TELLO-7 --fly   # steer TELLO-7 (take it off in the GUI first)

Scenario (organisers, 2026-10-24): attackers start at tag 10 and fly to tag 6; drone teams are
reported in telemetry. How an attacker is dismissed is not written anywhere yet, so the
defender blocks: it holds a point `--capture-gap` beside the attacker's path, inside the
capture distance, and keeps `--clearance` from every drone it is not catching. It never aims
at contact; two touching Tellos fall.

Frames
  telemetry pose   camera in tag k's frame: x right, y down, z into the wall (pupil)
  map.yaml         tag k in the tag-10 frame: x right, y up, z out of the face
  guard (here)     the sim's frame: X out of the tag-10 wall, Y along it (= map x),
                   Z up from the floor (= map y + tag10_height). Right-handed, like the sim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import socket
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
MAP = REPO / "Action_Space/drone_deployment_infra/config/map.yaml"
GUARD = REPO / "Action_Space/simulation_infra/source/action_space_kit/action_space_kit/tasks/direct/tello_defend/guard.py"
MOTION_FIT = REPO / "Action_Space/simulation_infra/source/action_space_kit/action_space_kit/control/tello_motion_fit.yaml"

C_FLIP = np.diag([1.0, -1.0, -1.0])  # pupil (y down, z into) <-> map (y up, z out)
STALE_S = 0.5  # a pose older than this is last-known, not current


def load_guard_module():
    # guard.py only needs torch; importing it through action_space_kit would pull in Isaac Lab
    spec = importlib.util.spec_from_file_location("guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard_mod = load_guard_module()


# -- frames -------------------------------------------------------------------------------


def load_map(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{tag: (R, t)} taking tag-k pupil coordinates to tag-10 pupil coordinates."""
    raw = yaml.safe_load(path.read_text())
    if "y up" not in raw.get("frame", "") or "out of" not in raw.get("frame", ""):
        print(f"WARNING: map frame says {raw.get('frame')!r}; expected 'x right, y up, z out of the tag face'")
    poses = {}
    for key, item in raw["tags"].items():
        R, t = np.asarray(item["R"], float), np.asarray(item["t"], float)
        poses[int(key)] = (C_FLIP @ R @ C_FLIP, C_FLIP @ t)
    return poses


def map_to_guard(v: np.ndarray, tag10_height: float = 0.0) -> np.ndarray:
    """Point (tag10_height given) or direction (0) from the map frame to the guard frame."""
    return np.array([v[2], v[0], v[1] + tag10_height])


def guard_to_map(g: np.ndarray, tag10_height: float = 0.0) -> np.ndarray:
    return np.array([g[1], g[2] - tag10_height, g[0]])


def pose_to_guard(pose: dict, poses: dict, tag10_height: float):
    """(position, heading [rad, CCW from +X]) in the guard frame from a telemetry pose, or None."""
    if pose.get("tag") not in poses:
        return None
    R, t = poses[pose["tag"]]
    cam10 = R @ np.asarray(pose["xyz"], float) + t
    position = map_to_guard(C_FLIP @ cam10, tag10_height)
    # Heading only from wall tags: on a floor tag (4, 7: face normal points up) the pose yaw
    # says nothing about the body's heading (measured ~90 deg off). None = keep the last one.
    if abs((C_FLIP @ R)[:, 2][1]) > 0.5:
        return position, None
    yaw = math.radians(pose["yaw"])  # 0 = facing the tag, + = turned towards the tag's +x
    forward = map_to_guard(C_FLIP @ (R @ np.array([math.sin(yaw), 0.0, math.cos(yaw)])))
    return position, math.atan2(forward[1], forward[0])


def guard_to_pose(position: np.ndarray, heading: float, tag: int, poses: dict, tag10_height: float) -> dict:
    """Inverse of pose_to_guard (self-test only): what drone_gui would report."""
    R, t = poses[tag]
    cam10 = C_FLIP @ guard_to_map(position, tag10_height)
    xyz = R.T @ (cam10 - t)
    forward_k = R.T @ (C_FLIP @ guard_to_map(np.array([math.cos(heading), math.sin(heading), 0.0])))
    return {"tag": tag, "xyz": xyz.round(3).tolist(), "yaw": math.degrees(math.atan2(forward_k[0], forward_k[2])), "age": 0.0}


# -- tracking -----------------------------------------------------------------------------


class Track:
    """One drone's position, velocity (differenced, smoothed) and heading in the guard frame."""

    def __init__(self):
        self.pos = None
        self.vel = np.zeros(3)
        self.heading = None
        self.t = -1.0  # measurement time of pos
        self.pending = None  # a big jump waits for a second sighting (mirror-flipped poses)
        self.key = None  # the last pose as reported, to skip repeats

    def update(self, pos: np.ndarray, heading: float | None, t_meas: float, key=None) -> None:
        # The same old pose comes back in every datagram until a new one is measured; its
        # t - age wobbles by +-5 ms from rounding, so compare the pose itself, not the time.
        if key is not None and key == self.key or t_meas <= self.t + 0.02:
            return
        self.key = key
        if self.pos is not None:
            dt = t_meas - self.t
            jump = np.linalg.norm(pos - self.pos)
            if jump > 1.0 and dt < 0.5:
                if self.pending is None or np.linalg.norm(pos - self.pending) > 0.3:
                    self.pending = pos
                    return
            if dt < 1.0:
                raw = np.clip((pos - self.pos) / max(dt, 1e-3), -3.0, 3.0)
                self.vel = 0.5 * self.vel + 0.5 * raw
            else:
                self.vel = np.zeros(3)
        self.pending = None
        self.pos, self.t = pos, t_meas
        if heading is None:
            pass
        elif self.heading is None:
            self.heading = heading
        else:  # smooth on the circle: single-tag yaw can jump 10-20 deg
            d = math.atan2(math.sin(heading - self.heading), math.cos(heading - self.heading))
            self.heading += 0.3 * d


# -- the controller -----------------------------------------------------------------------


class GuardPilot:
    def __init__(self, args, poses):
        self.args = args
        self.poses = poses
        goal = poses[args.goal_tag][1]  # tag-6 centre, tag-10 pupil frame
        self.goal = map_to_guard(C_FLIP @ goal, args.tag10_height)
        self.goal[2] = max(self.goal[2], args.min_height)
        self.tracks: dict[str, Track] = {}
        self.floor_since: dict[str, float] = {}  # attacker name -> when it was first seen down
        self.guard = None
        self.num_attackers = -1
        fit = yaml.safe_load(MOTION_FIT.read_text())["channels"]
        # m/s per rc unit, measured on a real Tello (control/tello_motion_fit.yaml)
        self.gain = np.array([fit["right"]["gain"], fit["forward"]["gain"], fit["up"]["gain"]])
        self.rc_cap = np.minimum(100, np.array([args.max_speed, args.max_speed, args.max_climb]) / self.gain).astype(int)

    def _guard(self, num_defenders: int, num_attackers: int):
        if self.guard is None or num_attackers != self.num_attackers:
            a = self.args
            self.guard = guard_mod.InterceptGuard(
                1, num_defenders, num_attackers, "cpu", speed=a.max_speed, gain=a.gain,
                guard_offset=a.capture_gap * 0.7, park_lead=a.capture_gap * 0.7, min_height=a.min_height,
                intercept_radius=a.goal_radius + a.margin, park_radius=a.park_radius,
                clearance=a.clearance, dock_gap=a.capture_gap, dock_speed=a.max_speed,
            )
            self.num_attackers = num_attackers
        return self.guard

    def ingest(self, msg: dict) -> None:
        for d in msg["drones"]:
            pose = d.get("pose")
            if not pose:
                continue
            got = pose_to_guard(pose, self.poses, self.args.tag10_height)
            if got is None:
                continue
            pos, heading = got
            tof = d.get("tof")
            if tof is not None and 0.1 < tof < 4.0:
                pos[2] = tof  # single-tag height can mirror-flip; the range sensor does not
            key = (pose["tag"], tuple(pose["xyz"]), pose.get("yaw"))
            self.tracks.setdefault(d["name"], Track()).update(pos, heading, msg["t"] - pose.get("age", 0.0), key)

    def is_out(self, d: dict, now: float) -> bool:
        """Telemetry has no "crashed" or "dismissed" flag. After an impact the firmware cuts the
        motors but drone_gui keeps fly=True, so also call it out when it sits at the floor without
        moving for a while. Revocable: it counts again as soon as it climbs or takes off."""
        if not d.get("link", True) or not d.get("fly"):
            return True
        tof, vel = d.get("tof"), d.get("vel") or [0.0, 0.0, 0.0]
        on_floor = tof is not None and tof <= 0.15 and float(np.linalg.norm(vel)) < 0.1
        if not on_floor:
            self.floor_since.pop(d["name"], None)
            return False
        return now - self.floor_since.setdefault(d["name"], now) > 0.7

    def home(self, i: int, n: int) -> np.ndarray:
        """Waiting spot: between the goal and tag 10, spread across the attackers' lane."""
        to_wall = np.array([-self.goal[0], -self.goal[1], 0.0])
        to_wall /= max(np.linalg.norm(to_wall), 1e-6)
        across = np.array([-to_wall[1], to_wall[0], 0.0])
        spot = self.goal + self.args.home_distance * to_wall + (i - (n - 1) / 2) * 0.9 * across
        spot[2] = self.args.home_height
        return spot

    def step(self, msg: dict, now: float) -> dict[str, tuple[int, int, int, int] | None]:
        """Sticks (right, forward, up, yaw) per defender name; None = no fresh pose, hover."""
        a = self.args
        by_name = {d["name"]: d for d in msg["drones"]}
        # steerable: a fresh pose, and a heading from a wall tag at some point (else hover)
        fresh = lambda n: n in self.tracks and now - self.tracks[n].t < STALE_S and self.tracks[n].heading is not None  # noqa: E731
        defenders = [n for n in a.defenders if n in by_name]
        attackers = [
            d["name"] for d in msg["drones"]
            if d["team"] == "attack" and not self.is_out(d, now)
            and d["name"] in self.tracks and now - self.tracks[d["name"]].t < 2.0
        ]
        out = {n: None for n in defenders}
        live = [n for n in defenders if fresh(n)]
        if not live:
            return out
        def_pos = torch.tensor(np.stack([self.tracks[n].pos for n in live]), dtype=torch.float32).unsqueeze(0)
        def_vel = torch.tensor(np.stack([self.tracks[n].vel for n in live]), dtype=torch.float32).unsqueeze(0)
        if attackers:
            att_pos = torch.tensor(np.stack([self.tracks[n].pos for n in attackers]), dtype=torch.float32).unsqueeze(0)
            att_vel = torch.tensor(np.stack([self.tracks[n].vel for n in attackers]), dtype=torch.float32).unsqueeze(0)
            site = torch.tensor(self.goal, dtype=torch.float32).unsqueeze(0)
            # status: an attacker counts as attacking while it heads for the goal (telemetry has
            # teams but no "dismissed" flag yet); a dismissed one that lands drops out above
            attacking = guard_mod.infer_attacking(att_pos, att_vel, site, threshold=a.attack_speed)
            cmd, target, _ = self._guard(len(live), len(attackers)).compute(
                def_pos, def_vel, att_pos, att_vel, attacking, site, torch.zeros(1), max_speed=a.max_speed
            )
            vel = cmd[0].numpy() * a.max_speed
        else:
            spots = np.stack([self.home(i, len(live)) for i in range(len(live))])
            vel = np.clip(a.gain * (spots - def_pos[0].numpy()), -a.max_speed, a.max_speed)
        for i, name in enumerate(live):
            out[name] = self.sticks(self.tracks[name], vel[i])
        return out

    def sticks(self, track: Track, vel: np.ndarray) -> tuple[int, int, int, int]:
        a = self.args
        pos = track.pos
        # geofence: never push further out of the box than it already is
        lo = np.array([a.box[0], a.box[2], a.min_height])
        hi = np.array([a.box[1], a.box[3], a.max_height])
        vel = vel.copy()
        vel = np.where((pos < lo) & (vel < 0), 0.5, vel)
        vel = np.where((pos > hi) & (vel > 0), -0.5, vel)
        psi = track.heading
        forward = vel[0] * math.cos(psi) + vel[1] * math.sin(psi)
        right = vel[0] * math.sin(psi) - vel[1] * math.cos(psi)
        rc = np.clip(np.round(np.array([right, forward, vel[2]]) / self.gain), -self.rc_cap, self.rc_cap)
        # hold the heading facing the tag-10 wall (-X): the camera keeps seeing the wall tags,
        # and attackers launched from tag 10 come at it head-on. rc yaw + = clockwise = -psi.
        err = math.atan2(math.sin(math.pi - psi), math.cos(math.pi - psi))
        yaw = int(np.clip(-a.yaw_gain * math.degrees(err), -30, 30))
        return int(rc[0]), int(rc[1]), int(rc[2]), yaw


# -- I/O ----------------------------------------------------------------------------------


class Gui:
    """drone_gui's HTTP API (drone_gui.py: POST /api/<iface>/<action>)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def state(self) -> list[dict]:
        with urllib.request.urlopen(f"{self.url}/api/state", timeout=1.0) as r:
            return json.loads(r.read())["drones"]

    def post(self, iface: str, action: str, body: dict | None = None) -> None:
        req = urllib.request.Request(
            f"{self.url}/api/{iface}/{action}", data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=0.3).read()


def run_live(args, poses) -> int:
    pilot = GuardPilot(args, poses)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.setblocking(False)
    gui = Gui(args.gui) if args.fly else None
    iface = {}
    if gui:
        iface = {d["name"]: d["iface"] for d in gui.state()}
        missing = [n for n in args.defenders if n not in iface]
        if missing:
            print(f"drone_gui has no {missing}; it has {sorted(iface)}")
            return 1
        print("FLYING: steering", args.defenders, "- Ctrl-C hands control back (zero sticks)", flush=True)
    else:
        print("dry run: printing sticks only (add --fly to send them)", flush=True)
    print(f"goal (tag {args.goal_tag}) in guard frame: {pilot.goal.round(2)}; rc caps {pilot.rc_cap}", flush=True)
    msg, period, next_t, last_print = None, 1.0 / args.hz, time.time(), 0.0
    try:
        while True:
            try:  # drain: only the newest datagram matters
                while True:
                    data, _ = sock.recvfrom(65535)
                    m = json.loads(data)
                    if m.get("v") == 1:
                        msg = m
                        pilot.ingest(msg)
            except BlockingIOError:
                pass
            now = time.time()
            if msg is None or now - msg["t"] > 1.0:
                if now - last_print > 2.0:
                    print("waiting for telemetry ...", flush=True)
                    last_print = now
            else:
                sticks = pilot.step(msg, now)
                for name, s in sticks.items():
                    s = s or (0, 0, 0, 0)  # no fresh pose: hover in place
                    if gui:
                        try:
                            gui.post(iface[name], "rc", {"sticks": list(s)})
                        except OSError as exc:
                            print(f"{name}: rc post failed ({exc})", flush=True)
                if now - last_print > 0.5:
                    parts = []
                    for name, s in sticks.items():
                        tr = pilot.tracks.get(name)
                        where = tr.pos.round(2) if tr and tr.pos is not None else "no pose"
                        parts.append(f"{name} at {where} sticks {s or 'HOVER (stale pose)'}")
                    print(" | ".join(parts), flush=True)
                    last_print = now
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))
    except KeyboardInterrupt:
        if gui:
            for name in args.defenders:
                try:
                    gui.post(iface[name], "rc", {"sticks": [0, 0, 0, 0]})
                except OSError:
                    pass
        print("\nstopped; drone_gui still holds the drones (LAND in the GUI)")
    return 0


def selftest(args, poses) -> int:
    """Synthetic attackers fly tag 10 -> tag 6; point-mass defenders follow the sticks with the
    measured rc gains and a 0.4 s lag. Exercises the frames (poses reported in tags 10 and 5),
    tracking, the guard and the stick conversion end to end, without hardware."""
    rng = np.random.default_rng(0)
    pilot = GuardPilot(args, poses)
    goal = pilot.goal
    tags = [t for t in (10, 5) if t in poses]
    dt, tau, results = 1.0 / args.hz, 0.4, []
    for ep in range(args.episodes):
        pilot.tracks.clear()
        defs = {n: [pilot.home(i, len(args.defenders)) + rng.normal(0, 0.1, 3), np.zeros(3), math.pi + rng.normal(0, 0.2)]
                for i, n in enumerate(args.defenders)}
        atks = {}
        for j in range(args.attackers):
            start = np.array([0.6, rng.uniform(-1.0, 2.0), rng.uniform(1.0, 1.8)])  # in front of tag 10
            atks[f"ATK-{j}"] = {"pos": start, "speed": args.attacker_speed * rng.uniform(0.9, 1.1),
                                "delay": j * rng.uniform(0.0, 2.0), "alive": True}
        breached, caught, t = False, 0, 0.0
        while t < 25.0 and not breached and any(x["alive"] for x in atks.values()):
            drones = []
            for n, (p, v, h) in defs.items():
                drones.append({"name": n, "team": "defense", "fly": True, "tof": float(p[2]),
                               "pose": guard_to_pose(p, h, tags[int(t * 2) % len(tags)], poses, args.tag10_height)})
            for n, x in atks.items():
                if t > x["delay"] and x["alive"]:
                    to_goal = goal - x["pos"]
                    x["pos"] = x["pos"] + x["speed"] * dt * to_goal / max(np.linalg.norm(to_goal), 1e-6)
                drones.append({"name": n, "team": "attack", "fly": x["alive"], "tof": float(x["pos"][2]),
                               "pose": guard_to_pose(x["pos"], math.pi, 10, poses, args.tag10_height)})
            msg = {"v": 1, "t": t, "drones": drones}
            pilot.ingest(msg)
            sticks = pilot.step(msg, t)
            for n, s in sticks.items():
                p, v, h = defs[n]
                s = s or (0, 0, 0, 0)
                body = np.array(s[:3]) * pilot.gain  # right, forward, up in m/s
                target = np.array([body[1] * math.cos(h) + body[0] * math.sin(h),
                                   body[1] * math.sin(h) - body[0] * math.cos(h), body[2]])
                v = v + (target - v) * dt / tau
                h = h - math.radians(s[3] * 0.6) * dt  # rc yaw + = clockwise; ~0.6 deg/s per unit
                defs[n] = [p + v * dt, v, h]
            for n, x in atks.items():
                if not x["alive"]:
                    continue
                gaps = [np.linalg.norm(x["pos"] - defs[d][0]) for d in defs]
                if min(gaps) < args.capture_radius:
                    x["alive"] = False
                    caught += 1
                elif np.linalg.norm((x["pos"] - goal)[:2]) < args.goal_radius:
                    breached = True
            t += dt
        results.append((breached, caught))
    n_breach = sum(b for b, _ in results)
    print(f"selftest: {args.episodes} episodes, {args.attackers} attackers at {args.attacker_speed} m/s vs "
          f"{len(args.defenders)} defenders at {args.max_speed} m/s")
    print(f"  breached {n_breach}/{args.episodes}, attackers caught {sum(c for _, c in results)}/{args.episodes * args.attackers}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defenders", nargs="+", default=["TELLO-7"], help="drone names this runner steers")
    ap.add_argument("--fly", action="store_true", help="send sticks to drone_gui (default: dry run)")
    ap.add_argument("--gui", default="http://127.0.0.1:8780", help="drone_gui address")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9100, help="telemetry port (one listener per port)")
    ap.add_argument("--hz", type=float, default=15.0, help="control rate (drone_gui's deadman is 0.35 s)")
    ap.add_argument("--map", type=Path, default=MAP)
    ap.add_argument("--goal-tag", type=int, default=6)
    ap.add_argument("--tag10-height", type=float, default=1.22, help="tag 10 centre above the floor [m]")
    ap.add_argument("--goal-radius", type=float, default=0.75, help="goal area around the goal tag [m]")
    ap.add_argument("--margin", type=float, default=0.6, help="intercept no closer than goal radius + this [m]")
    ap.add_argument("--park-radius", type=float, default=3.0, help="guard a waiting attacker from at most this far from the goal [m]")
    ap.add_argument("--capture-gap", type=float, default=0.3, help="block this far beside the attacker's path [m]")
    ap.add_argument("--clearance", type=float, default=0.7, help="keep this far from drones it is not catching [m]")
    ap.add_argument("--max-speed", type=float, default=0.65, help="horizontal speed cap [m/s] (battery)")
    ap.add_argument("--max-climb", type=float, default=0.4, help="vertical speed cap [m/s]")
    ap.add_argument("--gain", type=float, default=1.5, help="position gain [1/s] (sim 2.5; real Tellos lag ~0.5 s)")
    ap.add_argument("--yaw-gain", type=float, default=0.5, help="rc yaw per degree of heading error")
    ap.add_argument("--attack-speed", type=float, default=0.15, help="moving at the goal faster than this = attacking [m/s]")
    ap.add_argument("--min-height", type=float, default=0.5)
    ap.add_argument("--max-height", type=float, default=2.0)
    ap.add_argument("--home-distance", type=float, default=2.0, help="waiting spot: this far from the goal towards tag 10 [m]")
    ap.add_argument("--home-height", type=float, default=1.2)
    ap.add_argument("--box", type=float, nargs=4, default=[0.6, 7.0, -2.0, 3.5], metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="geofence in the guard frame [m]: X out of the tag-10 wall, Y along it (tag-10 right = +)")
    ap.add_argument("--selftest", action="store_true", help="offline check with synthetic attackers")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--attackers", type=int, default=2)
    ap.add_argument("--attacker-speed", type=float, default=0.7)
    ap.add_argument("--capture-radius", type=float, default=0.35, help="selftest only: dismissal distance [m]")
    args = ap.parse_args()

    poses = load_map(args.map)
    if args.goal_tag not in poses:
        print(f"goal tag {args.goal_tag} is not in {args.map}")
        return 1
    return selftest(args, poses) if args.selftest else run_live(args, poses)


if __name__ == "__main__":
    sys.exit(main())
