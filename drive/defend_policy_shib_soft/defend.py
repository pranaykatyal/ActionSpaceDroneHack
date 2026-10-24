"""defend: Shibani's trained 3v2 defender networks (Team 1, soft_contact_es_g10), 23 observations per defender.

  python swarm_policies.py --run defend:6,7,8 --run formation:4,5           # dry-run
  python swarm_policies.py --run defend:6,7,8 --run formation:4,5 --fly

Drop this file into swarm_ops/policies/ (replacing defend.py) with defend_checkpoint.pt next to
it, or point $DEFEND_CHECKPOINT at the checkpoint. The checkpoint is skrl-style: per defender,
policy net_container.{0,2,4,6} weights/biases and observation_preprocessor running_mean/variance
(a .npz export with defender_i/W0.. keys also loads). Needs numpy and pyyaml; torch only to read a .pt.

Attackers: self.bus "formation/drone<N>" (p in the map frame: x right, y up, z out of tag 10);
fixes older than MAX_AGE are ignored and a drone under 25 cm is out. Defenders: no camera feed in
this runner from the umbrella, so each runs its own video + AprilTag localization (Mo's idea, his
Eye class): every fresh fix resets its position; between fixes, and with no camera pipeline at all,
it is dead reckoned from the START spots printed at launch (all facing tag 10) and the velocity
commanded (DEFEND_HGAIN m/s per rc unit). Height from the tof. Fallback (Mo's, off by default): an attacker within
2 m of the goal, or still alive 8 s after the first attacker fix, gets one dedicated chaser at full
speed; the other defenders keep their networks.

Roles are fixed by --run order, as in training: the 1st drone is defender_0 (attacker slot 0),
the 2nd defender_1 (slot 1), the 3rd defender_2 (the goalkeeper). A 4th+ drone reuses the last
network. Attacker slots are sticky; an empty slot takes the free attacker nearest the goal. The
networks see 2 attackers at most.

Observation (23 per defender, sim frame x, y horizontal, z up; see map_to_sim):
  [0:3] own position - site centre (1.25 m above the floor below the goal tag)  [3:6] own velocity
  [6:9] projected gravity (0, 0, -1)  [9:15] attacker 0, 1 position - own (clip 12 m, 0 if empty)
  [15:21] attacker 0, 1 velocity - own (clip 3 m/s)  [21:23] attacker 0, 1 alive
Action: world velocity, 1.0 = 1 m/s; then teammate separation, geofence and speed caps.

Environment (--run takes no options):
  DEFEND_CHECKPOINT=<path>      default: defend_checkpoint.pt next to this file
  DEFEND_START="x,z;x,z;x,z"   floor spots in the map frame, one per defender in --run order
                                (the networks expect a ring about 2.5 m around the goal)
  DEFEND_HGAIN=0.021           horizontal m/s per rc unit (Team 1 measured 0.021 on TELLO-7)
  DEFEND_SPEED=0.65            horizontal speed cap, m/s
  DEFEND_GOAL_TAG=6
  DEFEND_CAMERA=1              0 = dead reckoning only (the cameras need cv2, av, pupil_apriltags)
  DEFEND_FALLBACK=0            1 = fallback chaser (no gain for this policy in the spoof tests)
"""
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from .base import Policy, hold_height

CHECKPOINT = os.environ.get("DEFEND_CHECKPOINT",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "defend_checkpoint.pt"))


def load_checkpoint(path):
    """skrl checkpoint (.pt) or numpy export (.npz) -> {"defender_i/mean|var|W0..W3|b0..b3": array}."""
    path = str(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = {}
    for agent, parts in ck.items():
        pol, pre = parts["policy"], parts["observation_preprocessor"]
        w[f"{agent}/mean"] = pre["running_mean"].float().numpy()
        w[f"{agent}/var"] = pre["running_variance"].float().numpy()
        for i, layer in enumerate((0, 2, 4, 6)):  # Linear layers between the ELUs
            w[f"{agent}/W{i}"] = pol[f"net_container.{layer}.weight"].float().numpy()
            w[f"{agent}/b{i}"] = pol[f"net_container.{layer}.bias"].float().numpy()
    return w


# -- cameras and fallback chaser (Mo's defend.py ideas) ------------------------------------
# CAMERAS: each defender runs its own video + AprilTag localization (swarm_formation.py's
# functions, as Mo's defend.py): every fresh fix resets its position, dead reckoning only bridges
# the gaps between fixes, so stick-gain and heading errors no longer pile up. Without a camera
# pipeline (no cv2 / av / pupil_apriltags / calibration) it falls back to dead reckoning alone.
# FALLBACK CHASER: an attacker within FALLBACK_R of the goal, or any attacker still alive
# FALLBACK_T after the first one was seen, gets ONE dedicated chaser: the nearest defender not
# already chasing (closest-to-goal attacker first) flies at it 0.5 s ahead at full speed.
CAMERA = os.environ.get("DEFEND_CAMERA", "1") != "0"
FALLBACK = os.environ.get("DEFEND_FALLBACK", "0") == "1"   # off by default: spoof tests found no gain here
FALLBACK_R = 2.0            # m from the goal, horizontal
FALLBACK_T = 8.0            # s after the first attacker fix
FIX_STALE_S = 0.5           # an older camera fix is not used
POSE_ALPHA = 0.35           # camera position smoothing (as swarm_formation)
JUMP_MAX_M, JUMP_ACCEPT_S = 0.5, 1.0
DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)


class Eye:
    """One drone's video + tag localization (Mo's defend.py); fix_map() = (map-frame position, t)."""

    def __init__(self, d, sf, K, D, tag_map, sizes, mapcheck):
        self.d, self.sf, self.K, self.D = d, sf, K, D
        self.tag_map, self.sizes, self.mapcheck = tag_map, sizes, mapcheck
        self.stop = threading.Event()
        self.frame, self.frame_id = None, 0
        self.fix, self.seen_t, self.jump_since = None, 0.0, None   # fix = (p internal, psi, t)
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

    def fix_map(self, now):
        """(position in the map frame: x right, y up, z out of tag 10; fix time), or None if stale."""
        if self.fix is None or now - self.seen_t > FIX_STALE_S:
            return None
        p = self.fix[0]                                      # swarm_formation: x right, y down, z into tag 10
        return np.array([p[0], -p[1], -p[2]], float), self.fix[2]


def make_eyes(drones):
    """{drone num: Eye} for every drone with a working camera pipeline; {} without one."""
    if not CAMERA:
        return {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import swarm_formation as sf
        _, tag_map, sizes = sf.load_tag_files(str(sf.MAP_PATH), str(sf.TAGS_PATH))
        mapcheck = sf.MapCheck(tag_map)
        eyes = {}
        for d in drones:
            K, D = sf.load_camera(str(sf.camera_file(d.num, sf.CAMERA_DIR)))
            eyes[d.num] = Eye(d, sf, K, D, tag_map, sizes, mapcheck)
        for e in eyes.values():  # streamon waits ~1 s per drone: don't hold up the control loop
            threading.Thread(target=e.start, daemon=True).start()
        print(f"[defend] cameras on for {sorted(eyes)}: positions from AprilTags, dead reckoning between fixes",
              flush=True)
        return eyes
    except Exception as exc:  # no video stack here: dead reckoning alone
        print(f"[defend] cameras off ({type(exc).__name__}: {exc}); dead reckoning only", flush=True)
        return {}


def chase_velocity(me_p, me_v, att_p, att_v, up, vmax):
    """Velocity at the attacker 0.5 s ahead, full speed until within 0.3 m (Mo's _chase).
    up = index of the vertical axis in this frame."""
    rel = (att_p - me_p) + 0.5 * (att_v - me_v)
    h = [k for k in range(3) if k != up]
    n = max(float(np.linalg.norm(rel[h])), 1e-3)
    v = np.zeros(3)
    v[h] = rel[h] / n * min(n / 0.3, 1.0) * vmax
    v[up] = float(np.clip(rel[up] / 0.3, -1.0, 1.0)) * 0.3
    return v


def pick_chasers(defenders, attackers, goal, up, since_first):
    """{defender num: attacker topic}. defenders: {num: position}; attackers: [(topic, p, v)]."""
    h = [k for k in range(3) if k != up]
    dist = {a[0]: float(np.linalg.norm((a[1] - goal)[h])) for a in attackers}
    trig = sorted((a for a in attackers if dist[a[0]] < FALLBACK_R or since_first > FALLBACK_T),
                  key=lambda a: dist[a[0]])
    out, free = {}, dict(defenders)
    for topic, p, _ in trig:
        if not free:
            break
        n = min(free, key=lambda k: float(np.linalg.norm(free[k] - p)))
        out[n] = topic
        free.pop(n)
    return out


MAP = Path(__file__).resolve().parents[2] / "config" / "map.yaml"
MAX_AGE = 0.5          # s: older attacker fixes are not steered on
SETTLE_S = 2.0         # s after takeoff: hold height only
GAIN = 2.0             # 1/s position gain
GOALIE_R = 1.35        # m from the goal centre, on the line to the most threatening attacker
LEAD_MAX = 2.0         # s of attacker motion to lead by
SEP_M = 0.6            # m: no closing on a teammate inside this
UP_GAIN = 0.006        # m/s per rc unit, up channel (Tello fit)
MAX_CLIMB = 0.4        # m/s
TAU = 0.35             # s: the drone's velocity lag, for the dead reckoning


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class GuardBase(Policy):
    """Shared plumbing (bus attackers, dead reckoning, rc mapping); tactics come from the networks below."""
    name = "defend"

    def setup(self, drones):
        super().setup(drones)
        m = yaml.safe_load(MAP.read_text())
        tags = {int(k): np.asarray(v["t"], float) for k, v in m["tags"].items()}
        normals = {int(k): np.asarray(v["R"], float)[:, 2] for k, v in m["tags"].items()}
        floor = [t[1] for k, t in tags.items() if abs(normals[k][1]) > 0.8]
        self.floor_y = float(np.median(floor)) if floor else tags[10][1] - 1.3
        goal_tag = int(_env_float("DEFEND_GOAL_TAG", 6))
        self.goal = tags[goal_tag].copy()
        self.h_gain = _env_float("DEFEND_HGAIN", 0.021)
        self.vmax = _env_float("DEFEND_SPEED", 0.65)
        pts = np.stack(list(tags.values()))
        self.lo = pts.min(0) - 0.2  # geofence: the box the tags span
        self.hi = pts.max(0) + 0.2
        self.lo[1], self.hi[1] = self.floor_y + 0.4, self.floor_y + 2.0
        n = len(self.drones)
        starts = os.environ.get("DEFEND_START")
        if starts:
            xz = [[float(v) for v in s.split(",")] for s in starts.split(";")]
        else:  # between the goal and tag 10, 2 m out from the goal, 0.9 m apart
            to10 = -self.goal.copy()
            to10[1] = 0.0
            to10 /= max(np.linalg.norm(to10), 1e-6)
            across = np.array([to10[2], 0.0, -to10[0]])
            spots = [self.goal + 2.0 * to10 + (i - (n - 1) / 2) * 0.9 * across for i in range(n)]
            xz = [[s[0], s[2]] for s in spots]
        self.p = {d.num: np.array([xz[i][0], self.floor_y, xz[i][1]]) for i, d in enumerate(self.drones)}
        self.v = {d.num: np.zeros(3) for d in self.drones}
        self.cmd = {d.num: np.zeros(3) for d in self.drones}
        self.hold = {}
        self.att = {}  # topic -> (p, v, t)
        self.roles, self.roles_t, self.last_t = {}, 0.0, {}
        self.eyes, self.fix_t, self.first_att, self.chasing = None, {}, None, {}
        print(f"[defend] goal tag {goal_tag} at {self.goal.round(2)} (map frame); place the defenders "
              f"facing tag 10 at START (x, z): " + ", ".join(f"{d.name} ({x:.2f}, {z:.2f})"
                                                                for d, (x, z) in zip(self.drones, xz)), flush=True)

    # -- state ------------------------------------------------------------------
    def _attackers(self, now):
        out = []
        for topic, (data, age) in self.bus.since("formation/", max_age=MAX_AGE).items():
            if not isinstance(data, dict) or data.get("p") is None:
                continue
            if (data.get("height_cm") or 100) < 25:  # down on the floor: out
                continue
            p = np.asarray(data["p"], float)
            t = now - (data.get("fix_age") or 0.0) - age
            old = self.att.get(topic)
            v = np.zeros(3)
            if old is not None and t - old[2] > 0.03:
                v = 0.6 * old[1] + 0.4 * (p - old[0]) / (t - old[2])
            elif old is not None:
                v = old[1]
            self.att[topic] = (p, v, t)
            out.append((topic, p, v))
        return out

    def _dead_reckon(self, d, now):
        dt = min(now - self.last_t.get(d.num, now), 0.2)
        self.last_t[d.num] = now
        self.v[d.num] += (self.cmd[d.num] - self.v[d.num]) * min(dt / TAU, 1.0)
        self.p[d.num] = self.p[d.num] + self.v[d.num] * dt
        h = d.height_cm()
        if h is not None:
            self.p[d.num][1] = self.floor_y + h / 100.0
        e = self.eyes.get(d.num) if self.eyes else None
        fix = e.fix_map(now) if e is not None else None
        if fix is not None and fix[1] > self.fix_t.get(d.num, 0.0):  # a fresh camera fix: reset
            self.fix_t[d.num] = fix[1]
            P = fix[0] + self.v[d.num] * max(now - fix[1], 0.0)  # brought forward to now
            if h is not None:
                P[1] = self.floor_y + h / 100.0
            self.p[d.num] = P

    # -- tactics ----------------------------------------------------------------
    def _assign(self, now):
        """Every defender's aim point and role, recomputed once per tick for all of them."""
        if now - self.roles_t < 0.03:
            return
        self.roles_t = now
        atts = self._attackers(now)
        nums = [d.num for d in self.drones]
        self.roles = {}
        if not atts:
            return
        # markers: cheapest defender for each attacker, attackers nearest the goal first
        free = set(nums)
        threat = sorted(atts, key=lambda a: np.linalg.norm((a[1] - self.goal)[[0, 2]]))
        for topic, p, v in threat:
            if not free:
                break
            best = min(free, key=lambda n: np.linalg.norm(self.p[n] - p))
            free.discard(best)
            lead = float(np.clip(np.linalg.norm(self.p[best] - p) / self.vmax, 0.0, LEAD_MAX))
            aim = p + v * lead
            # never lead past the goal: cap the aim at the attacker-goal segment
            if np.dot(aim - self.goal, p - self.goal) < 0:
                aim = self.goal + 0.3 * (p - self.goal)
            self.roles[best] = ("marker", topic, aim)
        # goalies: on the line goal -> most threatening attacker, GOALIE_R out
        topic, p, _ = threat[0]
        off = p - self.goal
        off[1] = 0.0
        r = np.linalg.norm(off)
        for i, n in enumerate(sorted(free)):
            spread = (i - (len(free) - 1) / 2) * 0.5
            unit = off / max(r, 1e-6)
            across = np.array([unit[2], 0.0, -unit[0]])
            aim = self.goal + min(GOALIE_R, r) * unit + spread * across
            aim[1] = p[1]
            if r < GOALIE_R + 0.6:  # it is through the first line: take it
                aim = p.copy()
            self.roles[n] = ("goalie", topic, aim)

    def step(self, d, t):
        now = time.time()
        if self.eyes is None:
            self.eyes = make_eyes(self.drones)
        if d.num not in self.hold:
            self.hold[d.num] = d.height_cm()
        self._dead_reckon(d, now)
        hover = (0, 0, hold_height(d, self.hold[d.num]), 0)
        if t < SETTLE_S:
            self.cmd[d.num] = np.zeros(3)
            return hover
        self._assign(now)
        role = self.roles.get(d.num)
        if role is None:  # no attacker in sight: hold the start spot
            self.cmd[d.num] = np.zeros(3)
            return hover
        p = self.p[d.num]
        v = GAIN * (role[2] - p)
        # separation: drop the closing component toward a teammate that is too near
        for n, q in self.p.items():
            if n == d.num:
                continue
            gap = q - p
            dist = np.linalg.norm(gap)
            if 1e-6 < dist < SEP_M:
                u = gap / dist
                v -= max(float(np.dot(v, u)), 0.0) * u
        # geofence: no further out of the box
        v = np.where((p < self.lo) & (v < 0), 0.3, v)
        v = np.where((p > self.hi) & (v > 0), -0.3, v)
        hv = np.linalg.norm(v[[0, 2]])
        if hv > self.vmax:
            v[[0, 2]] *= self.vmax / hv
        v[1] = float(np.clip(v[1], -MAX_CLIMB, MAX_CLIMB))
        self.cmd[d.num] = v
        # facing tag 10 (looking along -z): right = +x, forward = -z, up = +y
        rc = (int(np.clip(round(v[0] / self.h_gain), -100, 100)),
              int(np.clip(round(-v[2] / self.h_gain), -100, 100)),
              int(np.clip(round(v[1] / UP_GAIN), -100, 100)), 0)
        self.bus.publish(f"defend/{d.name}", {"p": [round(float(x), 3) for x in p], "role": role[0],
                                              "target": role[1]}, now)
        return rc

    def describe(self, d):
        role = getattr(self, "roles", {}).get(d.num)
        p = getattr(self, "p", {}).get(d.num)
        where = "" if p is None else f" @{p[0]:.1f},{p[2]:.1f}"
        return f"defend {role[0] if role else 'hold'}{where}"


# -- the networks ----------------------------------------------------------------------

CLIP_POS, CLIP_VEL = 12.0, 3.0      # obs_clip_pos / obs_clip_vel of tello_defend_env_cfg.py
SITE_HEIGHT = 2.5                    # the site centre is SITE_HEIGHT / 2 above the floor
MAX_LIN_SPEED = 1.0                  # m/s per action unit in training
EPS, CLIP = 1e-8, 5.0                # skrl RunningStandardScaler


def map_to_sim(v):
    """Organizers' map frame (x right, y up, z out of tag 10) -> sim frame (x, y horizontal, z up)."""
    v = np.asarray(v, float)
    return np.array([v[0], -v[2], v[1]])


def sim_to_map(v):
    v = np.asarray(v, float)
    return np.array([v[0], v[2], -v[1]])


def build_obs(own_p, own_v, site, attackers):
    """own_p, own_v, site: sim frame. attackers: two (pos, vel) in the sim frame, or None for an empty slot."""
    own_p, own_v = np.asarray(own_p, float), np.asarray(own_v, float)
    rel_p, rel_v, alive = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2)
    for k, att in enumerate(attackers[:2]):
        if att is None:
            continue
        rel_p[k] = np.clip(np.asarray(att[0], float) - own_p, -CLIP_POS, CLIP_POS)
        rel_v[k] = np.clip(np.asarray(att[1], float) - own_v, -CLIP_VEL, CLIP_VEL)
        alive[k] = 1.0
    grav = np.array([0.0, 0.0, -1.0])
    return np.concatenate([own_p - np.asarray(site, float), own_v, grav, rel_p.ravel(), rel_v.ravel(), alive]).astype(np.float32)


class DefendNets:
    """One MLP per defender slot (defender_0, defender_1, ...), loaded from export_npz.py's .npz."""

    def __init__(self, path):
        z = load_checkpoint(path)
        self.count = len({k.split("/")[0] for k in z})
        self.nets = []
        for i in range(self.count):
            a = f"defender_{i}"
            self.nets.append((z[f"{a}/mean"], np.sqrt(z[f"{a}/var"]) + EPS,
                              [(z[f"{a}/W{k}"], z[f"{a}/b{k}"]) for k in range(4)]))
        self.obs_dim = self.nets[0][0].shape[0]

    def __call__(self, i, obs):
        mean, std, layers = self.nets[i]
        h = np.clip((np.asarray(obs, np.float32) - mean) / std, -CLIP, CLIP)
        for k, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if k < 3:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0.0)))  # ELU
        return np.clip(h, -1.0, 1.0)


# -- the policy ------------------------------------------------------------------------

ROLE = ["slot0", "slot1", "keeper"]


class DefendPolicy(GuardBase):
    name = "defend"

    def setup(self, drones):
        super().setup(drones)
        w = Path(CHECKPOINT)
        if not w.exists():
            raise SystemExit(f"defend: no policy checkpoint at {w} (copy it there or set DEFEND_CHECKPOINT)")
        self.nets = DefendNets(w)
        self.site = self.goal.copy()
        self.site[1] = self.floor_y + SITE_HEIGHT / 2  # the sim's site centre: 1.25 m above the floor
        self.slots = [None, None]
        self.vnet = {}
        print(f"[defend] {w.name}: {self.nets.count} networks; roles by --run order: "
              + ", ".join(f"{d.name}={ROLE[min(i, self.nets.count - 1)] if self.nets.count == 3 else f'defender_{min(i, self.nets.count - 1)}'}"
                          for i, d in enumerate(self.drones)), flush=True)

    def _assign(self, now):
        """Every defender's network velocity, as an aim point defend.py's step flies at exactly."""
        if now - self.roles_t < 0.03:
            return
        self.roles_t = now
        atts = {topic: (p, v) for topic, p, v in self._attackers(now)}
        self.roles = {}
        self.slots = [t if t in atts else None for t in self.slots]
        for topic in sorted((t for t in atts if t not in self.slots),
                            key=lambda t: np.linalg.norm((atts[t][0] - self.goal)[[0, 2]])):
            if None not in self.slots:
                break
            self.slots[self.slots.index(None)] = topic
        if not atts:  # nothing in sight: defend.py holds position
            return
        site = map_to_sim(self.site)
        for i, d in enumerate(self.drones):
            n, k = d.num, min(i, self.nets.count - 1)
            slots = [None if t is None else (map_to_sim(atts[t][0]), map_to_sim(atts[t][1])) for t in self.slots]
            obs = build_obs(map_to_sim(self.p[n]), map_to_sim(self.v[n]), site, slots)
            v = sim_to_map(self.nets(k, obs)[:3] * MAX_LIN_SPEED)
            target = self.slots[k] if k < 2 and self.slots[k] else next((t for t in self.slots if t), None)
            # defend.py's step flies GAIN * (aim - p) = v, then separation, geofence and speed caps
            self.vnet[n] = v
            self.roles[n] = (ROLE[k] if self.nets.count == 3 else f"defender_{k}", target, self.p[n] + v / GAIN)
        if self.first_att is None:
            self.first_att = now
        self.chasing = pick_chasers({d.num: self.p[d.num] for d in self.drones},
                                    [(t, p, v) for t, (p, v) in atts.items()], self.goal, 1,
                                    now - self.first_att) if FALLBACK else {}
        for n, topic in self.chasing.items():
            v = chase_velocity(self.p[n], self.v[n], atts[topic][0], atts[topic][1], 1, self.vmax)
            self.vnet[n] = v
            self.roles[n] = ("chase", topic, self.p[n] + v / GAIN)

    def _dead_reckon(self, d, now):
        super()._dead_reckon(d, now)
        role = self.roles.get(d.num)
        if role is not None and d.num in self.vnet:  # keep the aim one GAIN-step ahead of the moved drone
            self.roles[d.num] = (role[0], role[1], self.p[d.num] + self.vnet[d.num] / GAIN)
