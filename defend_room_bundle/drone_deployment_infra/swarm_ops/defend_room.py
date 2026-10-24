#!/usr/bin/env python3
"""Fly the trained room-defense policy (sim task AS-Defend-Room-v0): 3 defenders vs 2 attackers.

Every drone localizes itself from the AprilTags (swarm_formation.py's video + tag pipeline),
so this one process knows where everyone is. Each tick (74 ms):
  attackers: scripted, fly away from tag 10 towards tag 6 at --attack-speed, like the sim
  defenders: the trained policy (one network per defender) -> world velocity -> rc
An attacker within --capture of a defender is caught (it lands); one crossing the tag-6
line is a breach. Ends when both are caught, on a breach, or after --duration: land all.

Frames. swarm_formation internal (tag 10 origin): x right, y DOWN, z INTO the tag.
Policy (sim) frame: X = out of the tag-10 wall (map z), Y = map x, Z = height above the
floor = map y + FLOOR (floor at the lowest mapped tag, tag 4). The policy commands world
velocities; the real rc is in the drone's heading frame, so every command is rotated by
its yaw, and the yaw is held facing tag 10 (psi = 0), as in the sim (yaw locked).

Placement: all drones FACING tag 10. Attackers 1-2 m in front of the tag-10 wall, defenders
0.3-1 m in front of tag 6, spread across the room, 0.5 m+ apart.

DEFAULT IS DRY-RUN (no takeoff; prints what it would send). --fly to fly.
  python defend_room.py --drones 3,4,5,6,7                 # dry-run: 3,4,5 defend, 6,7 attack
  python defend_room.py --drones 3,4,5,6,7 --fly
  python swarm_policies.py --run defend:3,4,5,6,7 --fly    # same, through the umbrella
Keys: q + Enter = land all, e + Enter = EMERGENCY. Ctrl+C = land all.
"""
import argparse
import csv
import math
import os
import sys
import termios
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_formation as sf  # noqa: E402  (drone link, video, tag localization)
from swarm_formation import CAMERA_DIR, DONGLES_PATH, MAP_PATH, TAGS_PATH  # noqa: E402

DEFAULT_POLICY = os.path.join(HERE, "..", "..", "simulation_infra", "policies", "defend_room_B7200", "policy.npz")

# ======================= SETTINGS =======================
PERIOD_S = 0.074             # one policy step = one rc packet (13.5 Hz), as trained
FLOOR_M = 1.207              # tag 10 center above the floor (sim frame Z = map y + FLOOR_M)
ARENA_X = (0.0, 7.0)         # sim frame bounds, m (tag-10 wall .. past tag 6)
ARENA_Y = (-0.40, 2.00)
ARENA_Z = (0.15, 1.776)
BREACH_X = 6.483             # tag 6
GEOFENCE_M = 0.30            # no motion further outwards this close to a wall / floor / ceiling
SEPARATION_M = 0.5           # defenders closer than this are pushed apart
SEPARATION_MPS = 0.3
MAX_SPEED_MPS = 1.0          # policy action 1.0 = 1 m/s (sim max_lin_speed)
VZ_SCALE = 0.3               # policy up action scale, as trained (vz_scale)
GAIN = {"right": 0.008, "forward": 0.00731, "up": 0.006}   # m/s per rc unit (tello_motion_fit.yaml)
RC_DEADBAND = 5
KP_YAW, YAW_DEADBAND_DEG, MAX_RC_YAW = 1.0, 4.0, 30
VEL_ALPHA = 0.4              # velocity estimate smoothing (new fixes)
FIX_STALE_S = 0.5            # no fresh tag fix: hover in place
LOST_LAND_S = 3.0            # no fix this long while flying: that drone lands
READY_TIMEOUT_S = 20.0       # after takeoff, every drone must see a tag by then
CLIP_POS, CLIP_VEL = 12.0, 3.0   # sim observation clipping
SCALER_CLIP, SCALER_EPS = 5.0, 1e-8
# ========================================================


def act(npz, agent, obs):
    """Exported policy (simulation_infra/scripts/export_defend_policy.py): obs (35,) -> action (4,)."""
    x = np.clip((obs - npz[f"{agent}/mean"]) / np.sqrt(npz[f"{agent}/var"] + SCALER_EPS), -SCALER_CLIP, SCALER_CLIP)
    for i in range(4):
        x = x @ npz[f"{agent}/W{i}"].T + npz[f"{agent}/b{i}"]
        if i < 3:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
    return np.clip(x, -1.0, 1.0)


def to_sim(p):
    """swarm_formation internal position (x right, y down, z into tag 10) -> sim frame."""
    return np.array([-p[2], p[0], -p[1] + FLOOR_M])


def to_rc(v_sim, psi_deg, max_rc):
    """World velocity in the sim frame (m/s) -> rc (right, forward, up, yaw) for a drone at yaw psi."""
    vx_i, vz_i = v_sim[1], -v_sim[0]                  # back to internal x (right) / z (into tag)
    s, c = math.sin(math.radians(psi_deg)), math.cos(math.radians(psi_deg))
    fwd, right = vx_i * s + vz_i * c, vx_i * c - vz_i * s    # as swarm_formation.control()

    def stick(v, g):
        if abs(v) < 0.03:
            return 0
        return int(math.copysign(min(max_rc, RC_DEADBAND + abs(v) / g), v))

    yaw = 0 if abs(psi_deg) < YAW_DEADBAND_DEG else int(np.clip(-KP_YAW * psi_deg, -MAX_RC_YAW, MAX_RC_YAW))
    return stick(right, GAIN["right"]), stick(fwd, GAIN["forward"]), stick(v_sim[2], GAIN["up"]), yaw


def geofence(P, v):
    """Zero any velocity component that heads further out within GEOFENCE_M of a bound."""
    v = v.copy()
    for k, (lo, hi) in enumerate((ARENA_X, ARENA_Y, ARENA_Z)):
        if P[k] < lo + GEOFENCE_M and v[k] < 0 or P[k] > hi - GEOFENCE_M and v[k] > 0:
            v[k] = 0.0
    return v


class Track:
    """Sim-frame position and smoothed velocity of one drone, from its tag fixes."""

    def __init__(self):
        self.P, self.V, self.t, self.psi = None, np.zeros(3), None, 0.0

    def update(self, d, now):
        e = d.est(now)
        if e is None:
            return
        p, psi, _ = e
        fix_t = d.fix[2]
        P = to_sim(p)
        if self.t is not None and fix_t > self.t:
            v = (P - self.P) / max(fix_t - self.t, 1e-3)
            self.V = VEL_ALPHA * np.clip(v, -CLIP_VEL, CLIP_VEL) + (1 - VEL_ALPHA) * self.V
        if self.t is None or fix_t > self.t:
            self.P, self.t = P, fix_t
        self.psi = psi


def build_obs(i, defenders, attackers, tracks, alive):
    """35 numbers for defender i, in AS-Defend-Room-v0's order (tello_defend_room_env.py)."""
    me = tracks[defenders[i].num]
    rel_p, rel_v = [], []
    for k, a in enumerate(attackers):
        t = tracks[a.num]
        m = 1.0 if alive[k] else 0.0
        rel_p.append(np.clip(t.P - me.P, -CLIP_POS, CLIP_POS) * m)
        rel_v.append(np.clip(t.V - me.V, -CLIP_VEL, CLIP_VEL) * m)
    mates = [tracks[defenders[j].num] for j in range(len(defenders)) if j != i]
    mate_p = [np.clip(t.P - me.P, -CLIP_POS, CLIP_POS) for t in mates]
    mate_v = [np.clip(t.V - me.V, -CLIP_VEL, CLIP_VEL) for t in mates]
    return np.concatenate([me.P, me.V, [0.0, 0.0, -1.0], *rel_p, *rel_v, np.array(alive, float), *mate_p, *mate_v])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drones", required=True, help="5 drone numbers: 3 defenders, then 2 attackers")
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="exported .npz (export_defend_policy.py)")
    ap.add_argument("--config", default=str(DONGLES_PATH))
    ap.add_argument("--attack-speed", type=float, default=0.3, help="attacker speed, m/s (sim trained 0.4-0.9)")
    ap.add_argument("--max-rc", type=int, default=60, help="defender stick limit (100 = ~0.75 m/s)")
    ap.add_argument("--capture", type=float, default=0.45, help="defender-attacker distance counted as a catch, m")
    ap.add_argument("--duration", type=float, default=25.0, help="seconds after GO, then land all")
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    args = ap.parse_args()
    fly = args.fly

    nums = [int(x) for x in args.drones.split(",") if x.strip()]
    if len(nums) != 5 or len(set(nums)) != 5:
        sys.exit("--drones needs 5 different numbers: 3 defenders, then 2 attackers")
    npz = np.load(args.policy)
    agents = [str(a) for a in npz["agents"]]
    print(f"MODE: {'FLY' if fly else 'DRY-RUN (no takeoff, no motion)'}; policy {args.policy}", flush=True)

    origin, tag_map, sizes = sf.load_tag_files(str(MAP_PATH), str(TAGS_PATH))
    print(f"Map: tags {sorted(tag_map)}", flush=True)
    # the policy was trained for the room whose breach line is tag 6, 6.48 m out from tag 10
    tag6 = to_sim(tag_map[6][1]) if 6 in tag_map else None
    if tag6 is None or abs(tag6[0] - BREACH_X) > 0.3:
        msg = (f"map {MAP_PATH}: tag 6 is " + ("missing" if tag6 is None else f"{tag6[0]:.2f} m from tag 10")
               + f", the policy expects {BREACH_X} m. Wrong map.yaml for this room?")
        if fly:
            sys.exit("REFUSING TO FLY: " + msg)
        print("WARNING: " + msg, flush=True)
    mapcheck = sf.MapCheck(tag_map)
    by_num = {d.num: d for d in sf.load_drones(args.config, set(nums), str(CAMERA_DIR), fly, tag_map, sizes, mapcheck)}
    drones = [by_num[n] for n in nums]
    defenders, attackers = drones[:3], drones[3:]
    print(f"defenders {[d.name for d in defenders]}, attackers {[d.name for d in attackers]}", flush=True)
    if not all(sf.parallel(drones, lambda d: sf.preflight(d, fly)).values()):
        for d in drones:
            d.close()
        sys.exit("Preflight FAILED -> nothing takes off.")

    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logf = open(os.path.join(HERE, time.strftime("logs/defend_%Y%m%d_%H%M%S.csv")), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["t", "drone", "role", "X", "Y", "Z", "VX", "VY", "VZ", "psi", "fresh", "a", "b", "c", "d"])

    tracks = {d.num: Track() for d in drones}
    alive, result = [True, True], None
    try:
        if fly:
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            answer = input("All PASS. All drones facing tag 10, defenders by tag 6, attackers by tag 10? "
                           "Type FLY: ").strip()
            if answer != "FLY":
                sys.exit(f"Aborted, no takeoff (got {answer!r}).")
            for d in drones:
                d.airborne = True
            ok = sf.parallel(drones, lambda d: (time.sleep(0.3 * drones.index(d)), d.send("takeoff", 20) == "ok")[1])
            if not all(ok.values()):
                print("A takeoff failed -> landing all", flush=True)
                return
        keys = sf.start_terminal_keys()
        print(">>> keys: q + Enter = land all, e + Enter = EMERGENCY, Ctrl+C = land all", flush=True)

        # wait until every drone sees a tag, then fix the defender roles by position (left -> right)
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
        defenders.sort(key=lambda d: tracks[d.num].P[1])      # defender_0 = smallest Y (trained lanes)
        for i, d in enumerate(defenders):
            print(f"[{d.name}] -> {agents[i]} at X {tracks[d.num].P[0]:.2f} Y {tracks[d.num].P[1]:.2f} "
                  f"Z {tracks[d.num].P[2]:.2f}", flush=True)
        for k, a in enumerate(attackers):
            print(f"[{a.name}] -> attacker_{k} at X {tracks[a.num].P[0]:.2f} Y {tracks[a.num].P[1]:.2f}", flush=True)
        heights = {a.num: tracks[a.num].P[2] for a in attackers}

        t_go, last_print = time.time(), 0.0
        print(f">>> GO ({args.duration:.0f} s)" + ("" if fly else " (DRY: nothing is sent)"), flush=True)
        while True:
            t_loop = now = time.time()
            for d in drones:
                tracks[d.num].update(d, now)
            cmds = {}

            # attackers: caught / breach / fly on
            for k, a in enumerate(attackers):
                if not alive[k]:
                    continue
                P = tracks[a.num].P
                if any(np.linalg.norm(tracks[d.num].P - P) < args.capture for d in defenders):
                    alive[k] = False
                    print(f"[{a.name}] CAUGHT", flush=True)
                    sf.land_async(a, "caught", fly)
                elif P[0] >= BREACH_X:
                    alive[k] = False
                    result = "BREACH"
                    print(f"[{a.name}] BREACH (past tag 6)", flush=True)
                    sf.land_async(a, "breach", fly)
                elif now - a.seen_t > LOST_LAND_S:
                    alive[k] = False
                    sf.land_async(a, f"no tag for {now - a.seen_t:.0f} s", fly)
                elif not a.has_fix(now):
                    cmds[a] = (0, 0, 0, 0)
                else:
                    v = np.array([args.attack_speed, 0.0, np.clip(heights[a.num] - P[2], -0.2, 0.2)])
                    cmds[a] = to_rc(geofence(P, v), tracks[a.num].psi, 100)
            if result is None and not any(alive):
                result = "DEFENDED"

            # defenders: the trained policy
            for i, d in enumerate(defenders):
                if d.landing or d.landed:
                    continue
                if now - d.seen_t > LOST_LAND_S:
                    sf.land_async(d, f"no tag for {now - d.seen_t:.0f} s", fly)
                    continue
                if not d.has_fix(now):
                    cmds[d] = (0, 0, 0, 0)
                    continue
                a = act(npz, agents[i], build_obs(i, defenders, attackers, tracks, alive))
                v = np.array([a[0] * MAX_SPEED_MPS, a[1] * MAX_SPEED_MPS, a[2] * VZ_SCALE * MAX_SPEED_MPS])
                P = tracks[d.num].P
                for o in defenders:
                    gap = P - tracks[o.num].P
                    dist = float(np.linalg.norm(gap))
                    if o is not d and 1e-6 < dist < SEPARATION_M:
                        v += SEPARATION_MPS * gap / dist
                cmds[d] = to_rc(geofence(P, v), tracks[d.num].psi, args.max_rc)

            for d, rc in cmds.items():
                if fly and not (d.landing or d.landed):
                    d.send_noreply(f"rc {rc[0]} {rc[1]} {rc[2]} {rc[3]}")
                d.last_rc = rc
            for d in drones:
                t = tracks[d.num]
                role = "defender" if d in defenders else "attacker"
                log.writerow([f"{now - t_go:.3f}", d.num, role, *np.round(t.P, 3), *np.round(t.V, 3),
                              f"{t.psi:.1f}", int(d.has_fix(now)), *d.last_rc])
            if now - last_print >= 1.0:
                last_print = now
                for d in drones:
                    t = tracks[d.num]
                    role = "DEF" if d in defenders else ("ATK" if alive[attackers.index(d)] else "down")
                    print(f"[{d.name}] {role} X {t.P[0]:.2f} Y {t.P[1]:.2f} Z {t.P[2]:.2f} psi {t.psi:+.0f} "
                          f"{'SEE' if d.has_fix(now) else 'LOST'}  rc {d.last_rc}" + ("" if fly else " (DRY)"),
                          flush=True)

            if result is not None:
                print(f">>> RESULT: {result} after {now - t_go:.1f} s -> landing all", flush=True)
                return
            if now - t_go > args.duration:
                print(f">>> RESULT: TIME UP ({args.duration:.0f} s), attackers left {sum(alive)} -> landing all",
                      flush=True)
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
