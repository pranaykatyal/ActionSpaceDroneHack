#!/usr/bin/env python3
"""Team 1's defenders with camera localization: every defender finds itself from the AprilTags
(swarm_formation.py's video + tag pipeline, as in defend_room.py), the attackers come from the
umbrella's telemetry bus ("formation/drone<N>"), and policies/defend.py's intercept guard (our
simulation guard, numpy) decides. No dead reckoning: positions are measured every frame.

  python swarm_policies.py --run defend_cam:6,7,8 --run formation:4,5          # dry-run
  python swarm_policies.py --run defend_cam:6,7,8 --run formation:4,5 --fly
  python defend_live.py --drones 6,7,8 --attackers 4,5                         # standalone demo:
        # it also flies 4,5 as scripted attackers from the tag-10 wall to tag 6 (--attack-speed)

Placement: every drone FACING tag 10, defenders 0.3-1 m in front of the goal tag, 0.5 m+ apart.
DEFAULT IS DRY-RUN (no takeoff; prints positions and the rc it would send). --fly to fly.
Ends after --duration, or (standalone attackers) when both are caught or one breaches.
A defender that loses its tags hovers, and lands after 3 s. Logs: logs/defend_live_*.csv.
"""
import argparse
import csv
import math
import os
import sys
import termios
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_formation as sf  # noqa: E402  (drone link, video, tag localization)
from swarm_formation import CAMERA_DIR, DONGLES_PATH, MAP_PATH, TAGS_PATH  # noqa: E402
from policies.defend import GUARD, InterceptGuard, infer_attacking  # noqa: E402
from policies.telemetry import TelemetryClient  # noqa: E402

PROMPT = "Type FLY:"         # the umbrella answers this line (see swarm_policies.External)
PERIOD_S = 0.074             # one rc packet per tick (13.5 Hz)
GAIN = {"right": 0.008, "forward": 0.00731, "up": 0.006}   # m/s per rc unit (tello_motion_fit.yaml)
RC_DEADBAND = 5
KP_YAW, YAW_DEADBAND_DEG, MAX_RC_YAW = 1.0, 4.0, 30
VEL_ALPHA = 0.4
LOST_LAND_S = 3.0
READY_TIMEOUT_S = 20.0
MAX_AGE = 0.5                # s: bus attacker fixes older than this are ignored
SITE_RADIUS = 0.75           # m: the goal zone (breach for standalone attackers)
GEOFENCE_M = 0.3


def to_rc(v, psi_deg, max_rc):
    """World velocity in the guard frame (X out of the tag-10 wall, Y = map x, Z up) -> rc
    (right, forward, up, yaw) for a drone at yaw psi (0 = facing tag 10)."""
    vx_i, vz_i = v[1], -v[0]                          # internal x (right) / z (into tag 10)
    s, c = math.sin(math.radians(psi_deg)), math.cos(math.radians(psi_deg))
    fwd, right = vx_i * s + vz_i * c, vx_i * c - vz_i * s

    def stick(u, g):
        if abs(u) < 0.03:
            return 0
        return int(math.copysign(min(max_rc, RC_DEADBAND + abs(u) / g), u))

    yaw = 0 if abs(psi_deg) < YAW_DEADBAND_DEG else int(np.clip(-KP_YAW * psi_deg, -MAX_RC_YAW, MAX_RC_YAW))
    return stick(right, GAIN["right"]), stick(fwd, GAIN["forward"]), stick(v[2], GAIN["up"]), yaw


class Track:
    """Guard-frame position and smoothed velocity of one drone, from its tag fixes."""

    def __init__(self, floor_m):
        self.floor_m, self.P, self.V, self.t, self.psi = floor_m, None, np.zeros(3), None, 0.0

    def to_g(self, p):  # swarm_formation internal (x right, y down, z into tag 10) -> guard frame
        return np.array([-p[2], p[0], -p[1] + self.floor_m])

    def update(self, d, now):
        e = d.est(now)
        if e is None:
            return
        p, psi, _ = e
        fix_t = d.fix[2]
        P = self.to_g(p)
        if self.t is not None and fix_t > self.t:
            v = (P - self.P) / max(fix_t - self.t, 1e-3)
            self.V = VEL_ALPHA * np.clip(v, -3.0, 3.0) + (1 - VEL_ALPHA) * self.V
        if self.t is None or fix_t > self.t:
            self.P, self.t = P, fix_t
        self.psi = psi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drones", required=True, help="our defenders, e.g. 6,7,8")
    ap.add_argument("--attackers", default="", help="standalone demo: also fly these as scripted attackers")
    ap.add_argument("--config", default=str(DONGLES_PATH))
    ap.add_argument("--goal-tag", type=int, default=6)
    ap.add_argument("--attack-speed", type=float, default=0.3, help="scripted attacker speed, m/s")
    ap.add_argument("--speed", type=float, default=0.65, help="defender speed cap, m/s")
    ap.add_argument("--max-rc", type=int, default=45, help="defender stick limit")
    ap.add_argument("--capture", type=float, default=0.45, help="standalone: catch distance, m")
    ap.add_argument("--duration", type=float, default=40.0, help="seconds after GO, then land all")
    ap.add_argument("--fly", action="store_true")
    args = ap.parse_args()
    fly = args.fly

    def_nums = [int(x) for x in args.drones.split(",") if x.strip()]
    own_att = [int(x) for x in args.attackers.split(",") if x.strip()]
    print(f"MODE: {'FLY' if fly else 'DRY-RUN (no takeoff, no motion)'}", flush=True)

    origin, tag_map, sizes = sf.load_tag_files(str(MAP_PATH), str(TAGS_PATH))
    # floor = median height of the floor tags (face normal vertical), in the internal frame (y down)
    floor = [t[1] for R, t in tag_map.values() if abs(R[1, 2]) > 0.8]
    floor_m = float(np.median(floor)) if floor else 1.2
    goal = Track(floor_m).to_g(tag_map[args.goal_tag][1])
    site = np.array([goal[0], goal[1], 0.0])
    print(f"Map: tags {sorted(tag_map)}; floor {floor_m:.2f} m below tag 10; goal tag {args.goal_tag} at "
          f"X {goal[0]:.2f} Y {goal[1]:.2f} Z {goal[2]:.2f} (X out of the tag-10 wall, Y = map x)", flush=True)
    lo = np.min([Track(floor_m).to_g(t) for _, t in tag_map.values()], axis=0) - 0.2
    hi = np.max([Track(floor_m).to_g(t) for _, t in tag_map.values()], axis=0) + 0.2
    lo[2], hi[2] = 0.4, 2.0

    mapcheck = sf.MapCheck(tag_map)
    by_num = {d.num: d for d in sf.load_drones(args.config, set(def_nums + own_att), str(CAMERA_DIR), fly,
                                               tag_map, sizes, mapcheck)}
    defenders = [by_num[n] for n in def_nums]
    attackers = [by_num[n] for n in own_att]
    drones = defenders + attackers
    if not all(sf.parallel(drones, lambda d: sf.preflight(d, fly)).values()):
        for d in drones:
            d.close()
        sys.exit("Preflight FAILED -> nothing takes off.")
    bus = TelemetryClient.from_env()

    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logf = open(os.path.join(HERE, time.strftime("logs/defend_live_%Y%m%d_%H%M%S.csv")), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["t", "drone", "role", "X", "Y", "Z", "VX", "VY", "VZ", "psi", "fresh", "target", "a", "b", "c", "d"])

    tracks = {d.num: Track(floor_m) for d in drones}
    guard, guard_key = None, None
    alive = {a.num: True for a in attackers}
    bus_att = {}  # topic -> (P, V, t)
    result = None
    try:
        if fly:
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            answer = input(f"All PASS. Drones facing tag 10, defenders in front of tag {args.goal_tag}? {PROMPT} ").strip()
            if answer != "FLY":
                sys.exit(f"Aborted, no takeoff (got {answer!r}).")
            for d in drones:
                d.airborne = True
            ok = sf.parallel(drones, lambda d: (time.sleep(0.3 * drones.index(d)), d.send("takeoff", 20) == "ok")[1])
            if not all(ok.values()):
                print("A takeoff failed -> landing all", flush=True)
                return
        keys = sf.start_terminal_keys()
        t_ready = time.time()
        while True:
            now = time.time()
            for d in drones:
                tracks[d.num].update(d, now)
            if all(d.has_fix(now) and tracks[d.num].P is not None for d in drones):
                break
            if now - t_ready > READY_TIMEOUT_S:
                print("Not every drone sees a tag -> landing all: "
                      + ", ".join(f"{d.name}: {d.reason}" for d in drones if not d.has_fix(now)), flush=True)
                return
            if keys() in (ord("q"), ord(" ")):
                return
            time.sleep(0.05)
        for d in drones:
            t = tracks[d.num]
            print(f"[{d.name}] {'DEF' if d in defenders else 'ATK'} at X {t.P[0]:.2f} Y {t.P[1]:.2f} Z {t.P[2]:.2f}", flush=True)
        heights = {a.num: tracks[a.num].P[2] for a in attackers}

        t_go, last_print = time.time(), 0.0
        print(f">>> GO ({args.duration:.0f} s)" + ("" if fly else " (DRY: nothing is sent)"), flush=True)
        while True:
            t_loop = now = time.time()
            for d in drones:
                tracks[d.num].update(d, now)
            cmds, targets = {}, {}

            # attackers: from the bus (the organisers' formation), plus our own scripted ones
            att = []
            for topic, (data, age) in sorted(bus.since("formation/", max_age=MAX_AGE).items()):
                if not isinstance(data, dict) or data.get("p") is None or (data.get("height_cm") or 100) < 25:
                    continue
                m = np.asarray(data["p"], float)  # map frame: x right, y up, z out of tag 10
                P = np.array([m[2], m[0], m[1] + floor_m])
                t_fix = now - age - (data.get("fix_age") or 0.0)
                old = bus_att.get(topic)
                V = np.zeros(3) if old is None else old[1]
                if old is not None and t_fix - old[2] > 0.03:
                    V = VEL_ALPHA * np.clip((P - old[0]) / (t_fix - old[2]), -3, 3) + (1 - VEL_ALPHA) * old[1]
                    bus_att[topic] = (P, V, t_fix)
                elif old is None:
                    bus_att[topic] = (P, V, t_fix)
                att.append((topic.split("/")[-1], P, V))
            for a in attackers:
                if not alive[a.num]:
                    continue
                P = tracks[a.num].P
                if any(np.linalg.norm(tracks[d.num].P - P) < args.capture for d in defenders):
                    alive[a.num] = False
                    print(f"[{a.name}] CAUGHT", flush=True)
                    sf.land_async(a, "caught", fly)
                    continue
                if np.linalg.norm(P[:2] - site[:2]) < SITE_RADIUS:
                    alive[a.num] = False
                    result = "BREACH"
                    print(f"[{a.name}] BREACH (goal zone)", flush=True)
                    sf.land_async(a, "breach", fly)
                    continue
                if now - a.seen_t > LOST_LAND_S:
                    alive[a.num] = False
                    sf.land_async(a, f"no tag for {now - a.seen_t:.0f} s", fly)
                    continue
                att.append((a.name, P, tracks[a.num].V))
                if a.has_fix(now):
                    to = site - P
                    to[2] = 0.0
                    v = args.attack_speed * to / max(np.linalg.norm(to), 1e-6)
                    v[2] = float(np.clip(heights[a.num] - P[2], -0.2, 0.2))
                    cmds[a] = to_rc(v, tracks[a.num].psi, 100)
                else:
                    cmds[a] = (0, 0, 0, 0)
            if attackers and result is None and not any(alive.values()):
                result = "DEFENDED"

            # defenders: the guard, on measured positions
            live = [d for d in defenders if not (d.landing or d.landed)]
            for d in live:
                if now - d.seen_t > LOST_LAND_S:
                    sf.land_async(d, f"no tag for {now - d.seen_t:.0f} s", fly)
            live = [d for d in live if d.has_fix(now)]
            if att and live:
                key = (tuple(n for n, _, _ in att), tuple(d.num for d in live))
                if guard is None or key != guard_key:
                    guard, guard_key = InterceptGuard(1, len(live), len(att), **dict(GUARD, speed=args.speed)), key
                def_pos = np.stack([tracks[d.num].P for d in live])[None]
                def_vel = np.stack([tracks[d.num].V for d in live])[None]
                att_pos = np.stack([p for _, p, _ in att])[None]
                att_vel = np.stack([v for _, _, v in att])[None]
                attacking = infer_attacking(att_pos, att_vel, site[None])
                cmd, target, _ = guard.compute(def_pos, def_vel, att_pos, att_vel, attacking, site[None],
                                               np.zeros(1), args.speed)
                for i, d in enumerate(live):
                    v = cmd[0, i] * args.speed
                    P = tracks[d.num].P
                    v = np.where((P < lo + GEOFENCE_M) & (v < 0), 0.0, v)
                    v = np.where((P > hi - GEOFENCE_M) & (v > 0), 0.0, v)
                    cmds[d] = to_rc(v, tracks[d.num].psi, args.max_rc)
                    targets[d.num] = att[target[0, i]][0]
                    bus.publish(f"defend/{d.name}", {"p": np.round(P, 3).tolist(), "target": targets[d.num]}, now)
            for d in defenders:
                if d not in cmds:
                    cmds[d] = (0, 0, 0, 0)  # no fix, or nothing to defend against: hover

            for d, rc in cmds.items():
                if fly and not (d.landing or d.landed):
                    d.send_noreply(f"rc {rc[0]} {rc[1]} {rc[2]} {rc[3]}")
                d.last_rc = rc
            for d in drones:
                t = tracks[d.num]
                log.writerow([f"{now - t_go:.3f}", d.num, "defender" if d in defenders else "attacker",
                              *np.round(t.P, 3), *np.round(t.V, 3), f"{t.psi:.1f}", int(d.has_fix(now)),
                              targets.get(d.num, ""), *d.last_rc])
            if now - last_print >= 1.0:
                last_print = now
                print(f"attackers seen: {[n for n, _, _ in att]}", flush=True)
                for d in drones:
                    t = tracks[d.num]
                    role = "DEF" if d in defenders else ("ATK" if alive.get(d.num) else "down")
                    print(f"[{d.name}] {role} X {t.P[0]:.2f} Y {t.P[1]:.2f} Z {t.P[2]:.2f} psi {t.psi:+.0f} "
                          f"{'SEE' if d.has_fix(now) else 'LOST'} -> {targets.get(d.num, '-')}  rc {d.last_rc}"
                          + ("" if fly else " (DRY)"), flush=True)
            if result is not None:
                print(f">>> RESULT: {result} after {now - t_go:.1f} s -> landing all", flush=True)
                return
            if now - t_go > args.duration:
                print(f">>> RESULT: TIME UP ({args.duration:.0f} s) -> landing all", flush=True)
                return
            k = keys()
            if k == ord("q"):
                print("q -> landing all", flush=True)
                return
            if k == ord(" ") and fly:
                print("!!! EMERGENCY: motors off all", flush=True)
                for d in drones:
                    d.send_noreply("emergency")
                    d.landed = True
                return
            time.sleep(max(0.0, PERIOD_S - (time.time() - t_loop)))
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing all", flush=True)
    finally:
        if fly:
            for d in drones:
                if d.land_thread is not None:
                    d.land_thread.join(20)
            sf.parallel([d for d in drones if d.airborne and not d.landed], lambda d: d.land())
        for d in drones:
            d.close()
        logf.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
