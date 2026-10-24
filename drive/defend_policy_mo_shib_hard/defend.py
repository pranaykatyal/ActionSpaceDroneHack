"""defend: Mo's defend.py (cameras for all 5 drones, scripted attackers, fallback chaser) flying
Shibani's hard networks (normal_contact_es_g40, 23 inputs, trained on AS-Defend-v0) instead of
Rohan's 35-input room policy. 3 defenders vs 2 attackers.

  python swarm_policies.py --run defend:3,4,5,6,7 --duration 45          # dry-run
  python swarm_policies.py --run defend:3,4,5,6,7 --duration 45 --fly

The group is 5 drones: the first 3 defend, the last 2 attack. Attackers are scripted: they
fly away from tag 10 towards tag 6 at ATTACK_SPEED_MPS, like in the sim. Defenders run the
trained networks (one per defender) from CHECKPOINT: defend_checkpoint.pt next to
this file, or the path in $DEFEND_CHECKPOINT. Needs torch to read it.

Shibani's networks (what changed from Mo's file):
  observation  23 per defender, in her training frame (x = map x, y = -map z, z = height):
               own position - site, own velocity, gravity (0, 0, -1), attackers 0 and 1 position
               and velocity relative to the defender (zeros once down), their alive flags. The site
               is tag 6's floor spot, SITE_Z_M above the floor.
  action       world velocity in that frame, 1.0 = 1 m/s on every axis; climb capped MAX_CLIMB_MPS,
               horizontal capped MAX_HSPEED_MPS (her deployment's 0.65 m/s).
  roles        by --run order, as she trained: defender_0 takes attacker 0, defender_1 attacker 1,
               defender_2 is the goalkeeper (Mo's left-to-right sort was for Rohan's lanes).
  sticks       GAIN is the 0.021 m/s per rc unit Team 1 measured on TELLO-7 (Mo's 0.008 made the
               drones fly ~2.6x faster than commanded).

Positions: the umbrella's drones only report height, so this policy starts its own video +
AprilTag localization per drone (swarm_formation.py's functions, map.yaml + tags.yaml).
Room frame: tag 10 = origin; X = straight out of the tag-10 wall (map z), Y = map x, Z =
height above the floor (Tello tof sensor). Tag 6, from the map, is the breach line.
The policy commands world velocities; the rc is in each drone's heading frame, so every
command is rotated by its yaw, and the yaw is held facing tag 10 (yaw locked in the sim).

Episode: once every drone sees a tag, roles are fixed (defender lanes left -> right) and
GO. An attacker within CAPTURE_M of a defender is caught; one past tag 6 is a breach.
Ends on both caught, a breach, or EPISODE_S: then every drone lands.
Placement: all drones FACING tag 10; attackers 1-2 m in front of the tag-10 wall;
defenders 0.3-1 m in front of tag 6, spread across the room, 0.5 m+ apart.
"""
import math
import os
import socket
import sys
import threading
import time

import numpy as np

from .base import Policy

# ======================= SETTINGS =======================
# Shibani's networks in skrl checkpoint layout (normal_contact_es_g40): next to this file, or $DEFEND_CHECKPOINT
CHECKPOINT = os.environ.get("DEFEND_CHECKPOINT",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "defend_checkpoint.pt"))
PERIOD_S = 0.074            # the umbrella's tick = one policy step, as trained (13.5 Hz)
EPISODE_S = 25.0             # after GO, then land all
READY_TIMEOUT_S = 20.0       # every drone must see a tag by then, else land all
ATTACK_SPEED_MPS = 0.3       # scripted attackers (sim trained on 0.4-0.9)
CAPTURE_M = 0.45             # defender-attacker distance counted as a catch (sim 0.35)
MAX_RC_DEFENDER = 100        # defender stick limit (100 = ~0.75 m/s, full speed)
# Fallback: an attacker past FALLBACK_X (m from tag 10) or still alive FALLBACK_T after GO gets ONE chaser,
# the nearest defender not already chasing (most advanced attacker first), flying at it 0.5 s ahead at
# full speed; the others keep the policy. Sim (scripts/eval_defend_room_1v1.py --mode 3v2-hybrid, 256
# episodes): breaches 1.2% -> 0.4% (attackers 0.4-0.9 m/s) and 2.3% -> 0.4% (0.8-1.2 m/s), crashes unchanged.
# All three chasing is NOT this: they ram each other (99.6% crash in sim).
FALLBACK = True
FALLBACK_X = 4.5
FALLBACK_T = 8.0
ARENA_Y = (-0.40, 2.00)      # room bounds the policy was trained in (Y = map x), m
ARENA_Z = (0.15, 1.776)
BEHIND_BREACH_M = 0.5        # X bound = tag 6 + this
GEOFENCE_M = 0.30            # no motion further outwards this close to a bound
SEPARATION_M, SEPARATION_MPS = 0.5, 0.3   # defenders closer than this are pushed apart
FLOOR_M = 1.207              # tag 10 center above the floor, used only without a tof reading
MAX_SPEED_MPS = 1.0          # network action 1.0 = 1 m/s; also the fallback chaser's speed
MAX_HSPEED_MPS = 0.65        # horizontal cap on the network's command (her deployment)
MAX_CLIMB_MPS = 0.4
VZ_SCALE = 0.3               # the fallback chaser's climb scale (Mo's)
SITE_Z_M = 1.25              # her site centre: this far above the floor below tag 6
GAIN = {"right": 0.021, "forward": 0.021, "up": 0.006}     # m/s per rc unit (Team 1, TELLO-7)
RC_DEADBAND = 5
KP_YAW, YAW_DEADBAND_DEG, MAX_RC_YAW = 1.0, 4.0, 30
VEL_ALPHA = 0.4              # velocity estimate smoothing
POSE_ALPHA = 0.35            # position smoothing (as swarm_formation)
JUMP_MAX_M, JUMP_ACCEPT_S = 0.5, 1.0
FIX_STALE_S = 0.5            # no fresh tag fix: hover in place
LOST_LAND_S = 3.0            # no fix this long: that drone lands
CLIP_POS, CLIP_VEL = 12.0, 3.0
SCALER_CLIP, SCALER_EPS = 5.0, 1e-8
DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
# ========================================================


def _sf():
    """swarm_formation (numpy/cv2/apriltags), imported only when defend runs."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import swarm_formation
    return swarm_formation


def load_weights(path=CHECKPOINT):
    """skrl MAPPO checkpoint -> per defender: observation scaler mean/var, actor layers W0..W3/b0..b3."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = {"agents": sorted(ck)}
    for agent, parts in ck.items():
        pol, pre = parts["policy"], parts["observation_preprocessor"]
        w[f"{agent}/mean"] = pre["running_mean"].double().numpy()
        w[f"{agent}/var"] = pre["running_variance"].double().numpy()
        for i, layer in enumerate((0, 2, 4, 6)):              # Linear layers between the ELUs
            w[f"{agent}/W{i}"] = pol[f"net_container.{layer}.weight"].double().numpy()
            w[f"{agent}/b{i}"] = pol[f"net_container.{layer}.bias"].double().numpy()
    return w


def act(w, agent, obs):
    """obs (23,) -> deterministic action (4,) in [-1, 1]: observation scaler + actor MLP."""
    x = np.clip((obs - w[f"{agent}/mean"]) / np.sqrt(w[f"{agent}/var"] + SCALER_EPS), -SCALER_CLIP, SCALER_CLIP)
    for i in range(4):
        x = x @ w[f"{agent}/W{i}"].T + w[f"{agent}/b{i}"]
        if i < 3:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
    return np.clip(x, -1.0, 1.0)


def to_room(p):
    """swarm_formation position (x right, y down, z into tag 10) -> room (X out of the wall, Y, Z)."""
    return np.array([-p[2], p[0], -p[1] + FLOOR_M])


def room_to_sim(v):
    """Room frame (X out of the tag-10 wall, Y = map x, Z up) -> Shibani's sim frame (x = map x, y = -map z, z up)."""
    return np.array([v[1], -v[0], v[2]])


def sim_to_room(v):
    return np.array([-v[1], v[0], v[2]])


def to_rc(v, psi_deg, max_rc):
    """Room-frame velocity (m/s) -> rc (right, forward, up, yaw) for a drone at yaw psi."""
    vx_i, vz_i = v[1], -v[0]
    s, c = math.sin(math.radians(psi_deg)), math.cos(math.radians(psi_deg))
    fwd, right = vx_i * s + vz_i * c, vx_i * c - vz_i * s    # as swarm_formation.control()

    def stick(u, g):
        return 0 if abs(u) < 0.03 else int(math.copysign(min(max_rc, RC_DEADBAND + abs(u) / g), u))

    yaw = 0 if abs(psi_deg) < YAW_DEADBAND_DEG else int(np.clip(-KP_YAW * psi_deg, -MAX_RC_YAW, MAX_RC_YAW))
    return stick(right, GAIN["right"]), stick(fwd, GAIN["forward"]), stick(v[2], GAIN["up"]), yaw


class Eye:
    """One drone's video + tag localization; position/velocity in the room frame."""

    def __init__(self, d, sf, K, D, tag_map, sizes, mapcheck):
        self.d, self.sf, self.K, self.D = d, sf, K, D
        self.tag_map, self.sizes, self.mapcheck = tag_map, sizes, mapcheck
        self.stop = threading.Event()
        self.frame, self.frame_id = None, 0
        self.fix, self.seen_t, self.jump_since = None, 0.0, None   # fix = (p, psi, t)
        self.P, self.V, self.psi, self.t = None, np.zeros(3), 0.0, None
        self.reason = "no frames"

    def start(self):
        threading.Thread(target=self.video_loop, daemon=True).start()
        for _ in range(3):                                   # raw: the umbrella only allows rc/land
            self.d.sock.sendto(b"streamon", DRONE_ADDR)
            time.sleep(0.3)
        threading.Thread(target=self.perception_loop, daemon=True).start()

    def video_loop(self):
        import av
        vs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        vs.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        vs.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, self.d.iface.encode())
        vs.bind(("", 11111))
        vs.settimeout(1.0)
        codec = av.CodecContext.create("h264", "r")
        while not self.stop.is_set():
            try:
                data = vs.recv(65536)
            except socket.timeout:
                continue
            try:
                for pkt in codec.parse(data):
                    for fr in codec.decode(pkt):
                        self.frame, self.frame_id = fr.to_ndarray(format="bgr24"), self.frame_id + 1
            except Exception:
                pass
        vs.close()

    def perception_loop(self):
        import cv2
        from pupil_apriltags import Detector
        sf = self.sf
        det = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0, refine_edges=True, decode_sharpening=0.25)
        last = 0
        while not self.stop.is_set():
            if self.frame_id == last or self.frame is None:
                time.sleep(0.005)
                continue
            last, img, t = self.frame_id, self.frame, time.time()
            good = []
            for dt in det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):
                if dt.tag_id not in self.tag_map or not self.mapcheck.trusted(dt.tag_id):
                    continue
                c = np.array(dt.corners, np.float64)
                side = min(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
                if dt.decision_margin >= sf.MIN_DECISION_MARGIN and side >= sf.MIN_TAG_SIDE_PX:
                    good.append((dt.tag_id, c))
            if not good:
                self.reason = "no trusted tag in view"
                continue
            prior = (self.fix[0], self.fix[1]) if self.fix is not None and t - self.fix[2] < 2.0 else None
            p, psi, info, per_tag = sf.localize(good, self.K, self.D, self.tag_map, self.sizes, prior)
            if p is None:
                self.reason = str(info)
                continue
            self.mapcheck.observe(per_tag)
            if self.fix is not None and t - self.fix[2] < 1.0 and np.linalg.norm(p - self.fix[0]) > JUMP_MAX_M:
                self.jump_since = self.jump_since or t
                if t - self.jump_since < JUMP_ACCEPT_S:
                    continue                                 # position jump: ignore briefly
            self.jump_since = None
            if self.fix is not None and t - self.fix[2] < FIX_STALE_S:
                p = POSE_ALPHA * p + (1 - POSE_ALPHA) * self.fix[0]
                psi = sf.wrap_deg(self.fix[1] + POSE_ALPHA * sf.wrap_deg(psi - self.fix[1]))
            self.fix, self.seen_t, self.reason = (p, psi, t), t, f"tags {info['tags']}"

    def has_fix(self, now):
        return self.fix is not None and now - self.seen_t < FIX_STALE_S

    def update(self, now):
        """Room position (Z from the tof sensor when it reads), smoothed velocity, yaw."""
        if self.fix is None:
            return
        p, psi, fix_t = self.fix
        P = to_room(p)
        h = self.d.height_cm()
        if h is not None:
            P[2] = h / 100.0
        if self.t is not None and fix_t > self.t:
            v = np.clip((P - self.P) / max(fix_t - self.t, 1e-3), -CLIP_VEL, CLIP_VEL)
            self.V = VEL_ALPHA * v + (1 - VEL_ALPHA) * self.V
        if self.t is None or fix_t > self.t:
            self.P, self.t = P, fix_t
        self.psi = psi


class DefendPolicy(Policy):
    name = "defend"

    def setup(self, drones):
        super().setup(drones)
        if len(self.drones) != 5:
            raise SystemExit(f"defend needs 5 drones (3 defenders, then 2 attackers), got {len(self.drones)}")
        if not os.path.exists(CHECKPOINT):
            raise SystemExit(f"defend: no policy checkpoint at {CHECKPOINT} (copy it there or set DEFEND_CHECKPOINT)")
        sf = self.sf = _sf()
        _, tag_map, sizes = sf.load_tag_files(str(sf.MAP_PATH), str(sf.TAGS_PATH))
        if 6 not in tag_map:
            raise SystemExit(f"defend: tag 6 (the breach line) is not in {sf.MAP_PATH}")
        self.breach_x = float(to_room(tag_map[6][1])[0])
        self.site = to_room(tag_map[6][1])
        self.site[2] = SITE_Z_M
        self.arena = ((0.0, self.breach_x + BEHIND_BREACH_M), ARENA_Y, ARENA_Z)
        print(f"[defend] map tags {sorted(tag_map)}; breach line (tag 6) {self.breach_x:.2f} m from tag 10"
              + ("" if abs(self.breach_x - 6.48) < 0.5 else "  WARNING: trained for 6.48 m"), flush=True)
        mapcheck = sf.MapCheck(tag_map)
        self.eyes = {}
        for d in self.drones:
            K, D = sf.load_camera(str(sf.camera_file(d.num, sf.CAMERA_DIR)))
            self.eyes[d.num] = Eye(d, sf, K, D, tag_map, sizes, mapcheck)
        self.w = load_weights()
        self.agents = [str(a) for a in self.w["agents"]]
        self.defenders, self.attackers = list(self.drones[:3]), list(self.drones[3:])
        self.alive, self.phase, self.result = [True, True], "START", None
        self.cmds, self.tick_t, self.t0, self.last_print = {}, 0.0, None, 0.0

    def step(self, d, t):
        now = time.time()
        if now - self.tick_t > 0.5 * PERIOD_S:                 # once per umbrella tick, for everyone
            self.tick_t = now
            self._tick(now)
        return self.cmds.get(d.num, (0, 0, 0, 0))

    def describe(self, d):
        if d in self.attackers:
            return f"attacker_{self.attackers.index(d)}"
        if self.phase == "GO" and d.num in getattr(self, "_chasing", {}):
            return f"chase->atk{self._chasing[d.num]}"
        return self.agents[self.defenders.index(d)] if self.phase == "GO" else "defender"

    # ---- one tick for all 5 drones ----
    def _tick(self, now):
        if self.phase == "START":
            for e in self.eyes.values():
                e.start()
            self.phase, self.t0 = "WAIT", now
        eyes = self.eyes
        for e in eyes.values():
            e.update(now)
        if self.phase == "WAIT":
            self.cmds = {n: (0, 0, 0, 0) for n in eyes}
            if all(e.has_fix(now) and e.P is not None for e in eyes.values()):
                for i, d in enumerate(self.defenders):
                    print(f"[defend] {d.name} -> {self.agents[i]} at {np.round(eyes[d.num].P, 2)}", flush=True)
                for k, a in enumerate(self.attackers):
                    print(f"[defend] {a.name} -> attacker_{k} at {np.round(eyes[a.num].P, 2)}", flush=True)
                self.heights = {a.num: eyes[a.num].P[2] for a in self.attackers}
                self.phase, self.t0 = "GO", now
                print(f"[defend] >>> GO ({EPISODE_S:.0f} s)", flush=True)
            elif now - self.t0 > READY_TIMEOUT_S:
                print("[defend] not every drone sees a tag -> land all: " + ", ".join(
                    f"drone{n}: {e.reason}" for n, e in eyes.items() if not e.has_fix(now)), flush=True)
                self._end("NO TAGS")
            return
        if self.phase == "END":
            return

        cmds = {}
        for k, a in enumerate(self.attackers):
            e = eyes[a.num]
            if not self.alive[k]:
                cmds[a.num] = None
            elif any(np.linalg.norm(eyes[d.num].P - e.P) < CAPTURE_M for d in self.defenders):
                self.alive[k], cmds[a.num] = False, None
                print(f"[defend] {a.name} CAUGHT", flush=True)
            elif e.P[0] >= self.breach_x:
                self.alive[k], cmds[a.num], self.result = False, None, "BREACH"
                print(f"[defend] {a.name} BREACH (past tag 6)", flush=True)
            elif now - e.seen_t > LOST_LAND_S:
                self.alive[k], cmds[a.num] = False, None
                print(f"[defend] {a.name} lost its tags -> lands", flush=True)
            elif not e.has_fix(now):
                cmds[a.num] = (0, 0, 0, 0)
            else:
                v = np.array([ATTACK_SPEED_MPS, 0.0, np.clip(self.heights[a.num] - e.P[2], -0.2, 0.2)])
                cmds[a.num] = to_rc(self._fence(e.P, v), e.psi, 100)
        chasers = self._chasers(now) if FALLBACK else {}
        for i, d in enumerate(self.defenders):
            e = eyes[d.num]
            if self.cmds.get(d.num, 0) is None or now - e.seen_t > LOST_LAND_S:
                cmds[d.num] = None
                continue
            if not e.has_fix(now):
                cmds[d.num] = (0, 0, 0, 0)
                continue
            k = chasers.get(d.num)
            if k is not None:
                v = self._chase(e, eyes[self.attackers[k].num])
            else:
                a = act(self.w, self.agents[min(i, len(self.agents) - 1)], self._obs(i))
                v = sim_to_room(a[:3] * MAX_SPEED_MPS)
                hv = float(np.linalg.norm(v[:2]))
                if hv > MAX_HSPEED_MPS:
                    v[:2] *= MAX_HSPEED_MPS / hv
                v[2] = float(np.clip(v[2], -MAX_CLIMB_MPS, MAX_CLIMB_MPS))
            for o in self.defenders:
                gap = e.P - eyes[o.num].P
                dist = float(np.linalg.norm(gap))
                if o is not d and 1e-6 < dist < SEPARATION_M:
                    v += SEPARATION_MPS * gap / dist
            cmds[d.num] = to_rc(self._fence(e.P, v), e.psi, MAX_RC_DEFENDER)
        self.cmds = cmds

        if now - self.last_print >= 1.0:
            self.last_print = now
            for d in self.drones:
                e = eyes[d.num]
                print(f"[defend] {d.name} {self.describe(d):<11} X {e.P[0]:.2f} Y {e.P[1]:.2f} Z {e.P[2]:.2f} "
                      f"psi {e.psi:+.0f} {'SEE' if e.has_fix(now) else 'LOST'}  rc {cmds.get(d.num)}", flush=True)
        if self.result is None and not any(self.alive):
            self.result = "DEFENDED"
        if self.result is None and now - self.t0 > EPISODE_S:
            self.result = f"TIME UP, {sum(self.alive)} attacker(s) left"
        if self.result is not None:
            print(f"[defend] >>> RESULT: {self.result} after {now - self.t0:.1f} s -> land all", flush=True)
            self._end(self.result)

    def _end(self, why):
        self.phase, self.cmds = "END", {n: None for n in self.eyes}   # None: the umbrella lands it
        for e in self.eyes.values():
            e.stop.set()

    def _chasers(self, now):
        """{defender num: attacker index} for the fallback (see FALLBACK_X / FALLBACK_T)."""
        eyes = self.eyes
        trig = [k for k, a in enumerate(self.attackers) if self.alive[k]
                and (eyes[a.num].P[0] > FALLBACK_X or now - self.t0 > FALLBACK_T)]
        trig.sort(key=lambda k: -eyes[self.attackers[k].num].P[0])       # most advanced first
        out, free = {}, list(self.defenders)
        for k in trig:
            if not free:
                break
            ak = eyes[self.attackers[k].num].P
            d = min(free, key=lambda o: float(np.linalg.norm(eyes[o.num].P - ak)))
            out[d.num] = k
            free.remove(d)
        if out != getattr(self, "_chasing", {}):
            for n, k in out.items():
                if getattr(self, "_chasing", {}).get(n) != k:
                    print(f"[defend] FALLBACK: drone{n} chases attacker_{k} "
                          f"(X {eyes[self.attackers[k].num].P[0]:.2f} m, {now - self.t0:.1f} s)", flush=True)
            self._chasing = dict(out)
        return out

    def _chase(self, me, atk):
        """Room-frame velocity at the attacker 0.5 s ahead, full speed until within 0.3 m."""
        rel = (atk.P - me.P) + 0.5 * (atk.V - me.V)
        n = max(float(np.linalg.norm(rel[:2])), 1e-3)
        xy = rel[:2] / n * min(n / 0.3, 1.0) * MAX_SPEED_MPS
        return np.array([xy[0], xy[1], float(np.clip(rel[2] / 0.3, -1.0, 1.0)) * VZ_SCALE * MAX_SPEED_MPS])

    def _fence(self, P, v):
        v = v.copy()
        for k, (lo, hi) in enumerate(self.arena):
            if P[k] < lo + GEOFENCE_M and v[k] < 0 or P[k] > hi - GEOFENCE_M and v[k] > 0:
                v[k] = 0.0
        return v

    def _obs(self, i):
        """23 numbers for defender i, in Shibani's AS-Defend-v0 order (her sim frame)."""
        eyes = self.eyes
        me = eyes[self.defenders[i].num]
        rel_p, rel_v = [], []
        for k, a in enumerate(self.attackers):
            m = 1.0 if self.alive[k] else 0.0
            rel_p.append(np.clip(room_to_sim(eyes[a.num].P - me.P), -CLIP_POS, CLIP_POS) * m)
            rel_v.append(np.clip(room_to_sim(eyes[a.num].V - me.V), -CLIP_VEL, CLIP_VEL) * m)
        return np.concatenate([room_to_sim(me.P - self.site), room_to_sim(me.V), [0.0, 0.0, -1.0],
                               *rel_p, *rel_v, np.array(self.alive, float)])

