"""defend: Team 1's defenders. Stop the attackers before they reach the goal tag (6).

  python swarm_policies.py --run defend:6,7,8 --run formation:4,5           # dry-run
  python swarm_policies.py --run defend:6,7,8 --run formation:4,5 --fly

The controller is Team 1's scripted intercept guard, the one in our simulation videos
(simulation_infra/.../tello_defend/guard.py, github.com/pranaykatyal/ActionSpaceDroneHack),
ported line for line from torch to numpy (checked equal to the torch original), with the
settings of our 3 v 2 room task (AS-Defend-RoomReal3-v0):
  * every attacker gets its own MARKER (assignment minimising the worst defender's distance,
    with hysteresis); the marker flies at the attacker's lead-intercept point and rams it;
  * the spare defender is a GOALIE: it holds the crossing point of whichever attacker is
    closest to the goal, just outside the goal zone, and takes anything that gets past;
  * an attacker that stops attacking is guarded from beside its path to the goal;
  * defenders stop closing on each other inside 1 m (fully at 0.3 m) and keep clear of drones
    they are not catching. Heading is never changed (yaw rc 0).
Works for any number of defenders and attackers.

Where things are:
  attackers   self.bus "formation/drone<N>": p in the map frame (x right, y up, z out of
              tag 10); fixes older than MAX_AGE are ignored, a drone under 25 cm is down
  defenders   each runs its own video + AprilTag localization (Mo's idea, his Eye class): every
              fresh fix resets its position; between fixes, and with no camera pipeline at all,
              it is dead reckoned from the START spots printed at launch (facing tag 10) and the
              velocity commanded (H_GAIN m/s per rc unit). Height is the drone's range sensor.
  fallback    off by default (DEFEND_FALLBACK=1): an attacker within 2 m of the goal, or alive 8 s after the first
              attacker fix, gets one dedicated chaser at full speed (Mo's fallback); the rest stay guard.
Publishes "defend/drone<N>": {"p", "aim", "target"} (map frame) for anyone watching.

Environment (since --run takes no options):
  DEFEND_START="x,z;x,z;x,z"   floor spots in the map frame, one per defender in --run order
  DEFEND_HGAIN=0.021           horizontal m/s per rc unit (Team 1 measured 0.021 on TELLO-7)
  DEFEND_SPEED=0.65            horizontal speed cap, m/s (battery)
  DEFEND_GOAL_TAG=6
  DEFEND_CAMERA=1              0 = dead reckoning only
  DEFEND_FALLBACK=0            1 = fallback chaser (no gain for this policy in the spoof tests)
Needs numpy and pyyaml (in requirements.txt); the cameras also need cv2, av and pupil_apriltags.
"""
import itertools
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from .base import Policy, hold_height

MAP = Path(__file__).resolve().parents[2] / "config" / "map.yaml"
MAX_AGE = 0.5          # s: older attacker fixes are not steered on
SETTLE_S = 2.0         # s after takeoff: hold height only
UP_GAIN = 0.006        # m/s per rc unit, up channel (Tello fit)
MAX_CLIMB = 0.4        # m/s
TAU = 0.35             # s: the drone's velocity lag, for the dead reckoning
# AS-Defend-RoomReal3-v0 guard settings, tuned in the organisers' final room (3 v 2, random
# attacker paths, 256 episodes each: 49.6% breached episodes -> 35.2% with lead_horizon 1.5 and
# sweep_depth 0.2; the 6 m corridor before it: 25.7%)
GUARD = {"speed": 0.65, "intercept_radius": 1.35, "park_radius": 3.0, "dock_gap": 0.0,
         "guard_offset": 0.0, "park_lead": 0.0, "dock_range": 0.0, "gain": 4.0,
         "lead_horizon": 1.5, "sweep_depth": 0.2}


# the guard's settings ship as defend_checkpoint.pt (next to this file, or $DEFEND_CHECKPOINT), like
# the network policies' checkpoints: {"guard_kwargs": {...}}. Without it (or without torch) the
# GUARD settings above are used.
CHECKPOINT = os.environ.get("DEFEND_CHECKPOINT",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "defend_checkpoint.pt"))


def load_guard_settings(path=CHECKPOINT):
    """GUARD updated with the checkpoint's guard_kwargs; GUARD alone if there is none."""
    cfg = dict(GUARD)
    if not os.path.exists(path):
        print(f"[defend] no {os.path.basename(path)}: built-in guard settings", flush=True)
        return cfg
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg.update({k: float(v) for k, v in ck["guard_kwargs"].items()})
        print(f"[defend] guard settings from {path}", flush=True)
    except Exception as exc:
        print(f"[defend] could not read {path} ({type(exc).__name__}: {exc}): built-in guard settings", flush=True)
    return cfg


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


# -- the guard (numpy port of guard.py; arrays are batched (E, ...) like the original) -----

def _norm(x, keepdims=False):
    return np.linalg.norm(x, axis=-1, keepdims=keepdims)


def side_of(offset, heading):
    """Unit vector perpendicular to heading, on the side offset points to (horizontal
    perpendicular when offset is dead ahead or behind)."""
    side = offset - (offset * heading).sum(-1, keepdims=True) * heading
    side_len = _norm(side, True)
    flat = np.stack([-heading[..., 1], heading[..., 0], np.zeros_like(heading[..., 0])], axis=-1)
    flat = flat / np.maximum(_norm(flat, True), 1e-6)
    return np.where(side_len > 0.05, side / np.maximum(side_len, 1e-6), flat)


def infer_attacking(att_pos, att_vel, site, threshold=0.15):
    """Status when the game does not report it: moving towards the site horizontally."""
    to_site = site[:, None] - att_pos
    to_site[..., 2] = 0.0
    to_site = to_site / np.maximum(_norm(to_site, True), 1e-6)
    return (att_vel * to_site).sum(-1) > threshold


class InterceptGuard:
    def __init__(self, num_envs, num_defenders, num_attackers, speed=1.2, gain=2.5, guard_offset=0.25,
                 park_lead=0.15, min_height=0.5, lead_horizon=3.0, hysteresis=0.4, intercept_radius=2.1,
                 park_radius=3.0, clearance=0.7, repulsion=5.0, dock_range=1.5, dock_gap=0.3, dock_speed=1.0,
                 goalkeep=True, sep_range=1.0, sep_min=0.3, sweep_depth=0.45):
        self.clearance, self.repulsion = clearance, repulsion
        self.dock_range, self.dock_gap, self.dock_speed, self.goalkeep = dock_range, dock_gap, dock_speed, goalkeep
        self.park_lead, self.sep_range, self.sep_min, self.sweep_depth = park_lead, sep_range, sep_min, sweep_depth
        self.intercept_radius, self.park_radius, self.speed, self.gain = intercept_radius, park_radius, speed, gain
        self.guard_offset, self.min_height, self.lead_horizon, self.hysteresis = guard_offset, min_height, lead_horizon, hysteresis
        if num_defenders <= num_attackers:
            perms = list(itertools.permutations(range(num_attackers), num_defenders))
        else:  # every attacker covered; never all spares on one attacker
            perms = [p for p in itertools.product(range(num_attackers), repeat=num_defenders)
                     if len(set(p)) == num_attackers]
        self.perms = np.array(perms, dtype=np.int64)  # (P, D)
        self.current = np.zeros(num_envs, dtype=np.int64)

    def aim_points(self, att_pos, att_vel, attacking, site, site_floor, def_pos):
        rel = att_pos[:, None] - def_pos[:, :, None]  # (E, D, A, 3)
        vel = np.broadcast_to(att_vel[:, None], rel.shape)
        a = (vel * vel).sum(-1) - self.speed ** 2
        b = 2.0 * (rel * vel).sum(-1)
        c = (rel * rel).sum(-1)
        disc = b * b - 4.0 * a * c
        sq = np.sqrt(np.maximum(disc, 0.0))
        safe_a = np.where(np.abs(a) < 1e-6, -1e-6, a)
        t1 = (-b - sq) / (2.0 * safe_a)
        t2 = (-b + sq) / (2.0 * safe_a)
        t = np.where(t1 > 0, t1, t2)
        fallback = np.sqrt(c) / self.speed
        t = np.minimum(np.where((disc >= 0) & (t > 0), t, fallback), self.lead_horizon)
        lead = att_pos[:, None] + vel * t[..., None]

        to_site = site[:, None] - att_pos
        to_site = to_site / np.maximum(_norm(to_site, True), 1e-6)
        park = (att_pos[:, None] + self.guard_offset * side_of(-rel, np.broadcast_to(to_site[:, None], rel.shape))
                + self.park_lead * to_site[:, None])

        site_xy = site[:, None, :2]
        att_radial = np.maximum(_norm(att_pos[..., :2] - site_xy), 1e-6)  # (E, A)

        def on_line(radius):
            frac = np.minimum(radius / att_radial, 1.0)[..., None]
            return site[:, None] + (att_pos - site[:, None]) * frac

        lead_radial = _norm(lead[..., :2] - site_xy[:, None])
        crossing = np.broadcast_to(on_line(self.intercept_radius)[:, None], lead.shape)
        late = (lead_radial < self.intercept_radius) & (att_radial > self.intercept_radius)[:, None]
        lead = np.where(late[..., None], crossing, lead)
        park_radial = _norm(park[..., :2] - site_xy[:, None])
        park = np.where((park_radial > self.park_radius)[..., None],
                        np.broadcast_to(on_line(self.park_radius)[:, None], park.shape), park)
        aim = np.where(attacking[:, None, :, None], lead, park).copy()
        aim[..., 2] = np.maximum(aim[..., 2], (site_floor + self.min_height).reshape(-1, 1, 1))
        return aim

    def compute(self, def_pos, def_vel, att_pos, att_vel, attacking, site, site_floor, max_speed):
        aim = self.aim_points(att_pos, att_vel, attacking, site, site_floor, def_pos)  # (E, D, A, 3)
        dist = _norm(aim - def_pos[:, :, None])  # (E, D, A)
        E, num_d = def_pos.shape[:2]
        ar_e, ar_d = np.arange(E), np.arange(num_d)
        pair = dist[:, ar_d[None, :], self.perms]  # (E, P, D)
        cost = pair.max(-1) + 0.1 * pair.sum(-1)
        best = cost.argmin(-1)
        keep = cost[ar_e, self.current]
        switch = cost[ar_e, best] < keep - self.hysteresis
        self.current = np.where(switch, best, self.current)

        target = self.perms[self.current]  # (E, D)
        chosen = aim[ar_e[:, None], ar_d[None, :], target]  # (E, D, 3)

        dist_t = dist[ar_e[:, None], ar_d[None, :], target]  # (E, D)
        same = target[:, :, None] == target[:, None, :]
        nearer = dist_t[:, None, :] < dist_t[:, :, None]
        eye = np.eye(num_d, dtype=bool)[None]
        earlier = ~(np.cumsum(eye, axis=-1).astype(bool))
        sweeper = (same & (nearer | ((dist_t[:, None, :] == dist_t[:, :, None]) & earlier)) & ~eye).any(-1)
        att_radial = _norm(att_pos[..., :2] - site[:, None, :2])
        threat = np.where(attacking, att_radial, att_radial + 1e3).argmin(-1)  # (E,)
        tgt_now = np.repeat(att_pos[ar_e, threat][:, None], num_d, axis=1)
        radial = np.maximum(_norm(tgt_now[..., :2] - site[:, None, :2], True), 1e-6)
        frac = np.minimum((self.intercept_radius - self.sweep_depth) / radial, 1.0)
        crossing = site[:, None] + (tgt_now - site[:, None]) * frac
        crossing[..., 2] = np.maximum(crossing[..., 2], (site_floor + self.min_height).reshape(-1, 1))
        chosen = np.where(sweeper[..., None], crossing, chosen)

        tgt_pos = att_pos[ar_e[:, None], target]
        tgt_vel = att_vel[ar_e[:, None], target]
        tgt_attacking = attacking[ar_e[:, None], target]
        away = def_pos - tgt_pos
        gap = np.maximum(_norm(away, True), 1e-6)
        docking = tgt_attacking[..., None] & (gap < self.dock_range) & ~sweeper[..., None]
        heading = tgt_vel / np.maximum(_norm(tgt_vel, True), 1e-3)
        side = side_of(away, heading)
        dock_point = tgt_pos + self.dock_gap * side
        rel = self.gain * (dock_point - def_pos)
        rel_len = np.maximum(_norm(rel, True), 1e-6)
        rel = rel * np.minimum(self.dock_speed / rel_len, 1.0)
        follow = tgt_vel + rel
        ahead = ((away * heading).sum(-1, keepdims=True) > 0.0) & self.goalkeep
        on_path = tgt_pos + (away * heading).sum(-1, keepdims=True) * heading + self.dock_gap * side
        goalie = self.gain * (on_path - def_pos)
        dock_point = np.where(ahead, on_path, dock_point)
        chosen = np.where(docking, dock_point, chosen)
        cmd = np.where(docking, np.where(ahead, goalie, follow), self.gain * (chosen - def_pos))

        for i in range(num_d):
            for k in range(num_d):
                if i == k:
                    continue
                to_k = def_pos[:, k] - def_pos[:, i]
                gap_ik = np.maximum(_norm(to_k, True), 1e-6)
                u = to_k / gap_ik
                weight = np.clip((self.sep_range - gap_ik) / (self.sep_range - self.sep_min), 0.0, 1.0)
                closing = np.maximum((cmd[:, i] * u).sum(-1, keepdims=True), 0.0)
                cmd[:, i] = cmd[:, i] - weight * closing * u

        num_a = att_pos.shape[1]
        others = np.concatenate([att_pos, def_pos], axis=1)
        rel = def_pos[:, :, None] - others[:, None]
        dist = np.maximum(_norm(rel), 1e-3)
        is_target = np.zeros((E, num_d, num_a), dtype=bool)
        is_target[ar_e[:, None], ar_d[None, :], target] = True
        not_self = np.broadcast_to(~np.eye(num_d, dtype=bool)[None], (E, num_d, num_d))
        avoid = np.concatenate([~is_target, not_self], axis=-1) & (dist < self.clearance)
        push = ((self.clearance - dist) / self.clearance * avoid)[..., None] * rel / dist[..., None]
        cmd = cmd + self.repulsion * push.sum(2)
        cmd = cmd / max_speed
        peak = np.maximum(np.abs(cmd).max(-1, keepdims=True), 1.0)
        return cmd / peak, target, chosen


# -- the policy ---------------------------------------------------------------------------

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class DefendPolicy(Policy):
    name = "defend"

    def setup(self, drones):
        super().setup(drones)
        m = yaml.safe_load(MAP.read_text())
        tags = {int(k): np.asarray(v["t"], float) for k, v in m["tags"].items()}
        normals = {int(k): np.asarray(v["R"], float)[:, 2] for k, v in m["tags"].items()}
        floor = [t[1] for k, t in tags.items() if abs(normals[k][1]) > 0.8]
        self.floor_y = float(np.median(floor)) if floor else tags[10][1] - 1.3
        goal_tag = int(_env_float("DEFEND_GOAL_TAG", 6))
        self.goal_map = tags[goal_tag]
        self.site = self.to_g(self.goal_map)
        self.site[2] = 0.0  # the protected zone stands on the floor below the goal tag
        self.h_gain = _env_float("DEFEND_HGAIN", 0.021)
        self.vmax = _env_float("DEFEND_SPEED", 0.65)
        pts = np.stack([self.to_g(t) for t in tags.values()])
        self.lo, self.hi = pts.min(0) - 0.2, pts.max(0) + 0.2  # geofence: the box the tags span
        self.lo[2], self.hi[2] = 0.4, 2.0
        n = len(self.drones)
        starts = os.environ.get("DEFEND_START")
        if starts:
            xz = [[float(v) for v in s.split(",")] for s in starts.split(";")]
        else:  # between the goal and tag 10, 2 m out from the goal, 0.9 m apart
            to10 = -self.goal_map.copy()
            to10[1] = 0.0
            to10 /= max(np.linalg.norm(to10), 1e-6)
            across = np.array([to10[2], 0.0, -to10[0]])
            spots = [self.goal_map + 2.0 * to10 + (i - (n - 1) / 2) * 0.9 * across for i in range(n)]
            xz = [[s[0], s[2]] for s in spots]
        self.p = {d.num: self.to_g(np.array([xz[i][0], self.floor_y, xz[i][1]])) for i, d in enumerate(self.drones)}
        self.v = {d.num: np.zeros(3) for d in self.drones}
        self.cmd = {d.num: np.zeros(3) for d in self.drones}
        self.hold, self.last_t = {}, {}
        self.att = {}  # topic -> (p, v, t), guard frame
        self.guard, self.guard_topics = None, None
        self.out, self.out_t = {}, 0.0
        self.eyes, self.fix_t, self.first_att, self.chasing = None, {}, None, {}
        cfg = dict(load_guard_settings(), speed=self.vmax)
        self.guard_cfg = cfg
        print(f"[defend] goal tag {goal_tag} at {self.goal_map.round(2)} (map frame). Place the defenders "
              "facing tag 10 at START (map x, z): " + ", ".join(
                  f"{d.name} ({x:.2f}, {z:.2f})" for d, (x, z) in zip(self.drones, xz)), flush=True)

    def to_g(self, p_map):
        """Map frame (x right, y up, z out of tag 10) -> guard frame (X out of the tag-10 wall,
        Y along it, Z up from the floor), right-handed with Z up like the simulation."""
        return np.array([p_map[2], p_map[0], p_map[1] - self.floor_y], float)

    def to_map(self, g):
        return np.array([g[1], g[2] + self.floor_y, g[0]], float)

    def _attackers(self, now):
        out = []
        for topic, (data, age) in sorted(self.bus.since("formation/", max_age=MAX_AGE).items()):
            if not isinstance(data, dict) or data.get("p") is None:
                continue
            if (data.get("height_cm") or 100) < 25:  # on the floor: out
                continue
            p = self.to_g(np.asarray(data["p"], float))
            t = now - (data.get("fix_age") or 0.0) - age
            old = self.att.get(topic)
            v = np.zeros(3) if old is None else old[1]
            if old is not None and t - old[2] > 0.03:
                v = 0.6 * old[1] + 0.4 * (p - old[0]) / (t - old[2])
            self.att[topic] = (p, v, t) if old is None or t - old[2] > 0.03 else (old[0], v, old[2])
            out.append((topic, p, v))
        return out

    def _dead_reckon(self, d, now):
        dt = min(now - self.last_t.get(d.num, now), 0.2)
        self.last_t[d.num] = now
        self.v[d.num] += (self.cmd[d.num] - self.v[d.num]) * min(dt / TAU, 1.0)
        self.p[d.num] = self.p[d.num] + self.v[d.num] * dt
        h = d.height_cm()
        if h is not None:
            self.p[d.num][2] = h / 100.0
        e = self.eyes.get(d.num) if self.eyes else None
        fix = e.fix_map(now) if e is not None else None
        if fix is not None and fix[1] > self.fix_t.get(d.num, 0.0):  # a fresh camera fix: reset
            self.fix_t[d.num] = fix[1]
            P = self.to_g(fix[0]) + self.v[d.num] * max(now - fix[1], 0.0)  # brought forward to now
            if h is not None:
                P[2] = h / 100.0
            self.p[d.num] = P

    def _solve(self, now):
        """The guard's velocity for every defender, once per tick."""
        if now - self.out_t < 0.03:
            return
        self.out_t = now
        self.out = {}
        atts = self._attackers(now)
        if not atts:
            return
        topics = [a[0] for a in atts]
        if self.guard is None or topics != self.guard_topics:
            self.guard = InterceptGuard(1, len(self.drones), len(atts), **self.guard_cfg)
            self.guard_topics = topics
        nums = [d.num for d in self.drones]
        def_pos = np.stack([self.p[n] for n in nums])[None]
        def_vel = np.stack([self.v[n] for n in nums])[None]
        att_pos = np.stack([a[1] for a in atts])[None]
        att_vel = np.stack([a[2] for a in atts])[None]
        site = self.site[None]
        attacking = infer_attacking(att_pos, att_vel, site)
        cmd, target, aim = self.guard.compute(def_pos, def_vel, att_pos, att_vel, attacking, site,
                                              np.zeros(1), self.vmax)
        for i, n in enumerate(nums):
            self.out[n] = (cmd[0, i] * self.vmax, topics[target[0, i]], aim[0, i])
        if self.first_att is None:
            self.first_att = now
        self.chasing = pick_chasers({n: self.p[n] for n in nums}, atts, self.site, 2,
                                    now - self.first_att) if FALLBACK else {}
        for n, topic in self.chasing.items():
            _, p, v = next(a for a in atts if a[0] == topic)
            self.out[n] = (chase_velocity(self.p[n], self.v[n], p, v, 2, self.vmax), topic, p)

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
        self._solve(now)
        if d.num not in self.out:  # no attacker in sight: hold where it is
            self.cmd[d.num] = np.zeros(3)
            return hover
        v, topic, aim = self.out[d.num]
        p = self.p[d.num]
        v = np.where((p < self.lo) & (v < 0), 0.3, v)  # geofence: no further out of the box
        v = np.where((p > self.hi) & (v > 0), -0.3, v)
        v[2] = float(np.clip(v[2], -MAX_CLIMB, MAX_CLIMB))
        self.cmd[d.num] = v
        # facing tag 10 (-X): right = +Y, forward = -X, up = +Z
        rc = (int(np.clip(round(v[1] / self.h_gain), -100, 100)),
              int(np.clip(round(-v[0] / self.h_gain), -100, 100)),
              int(np.clip(round(v[2] / UP_GAIN), -100, 100)), 0)
        self.bus.publish(f"defend/{d.name}", {"p": [round(float(x), 3) for x in self.to_map(p)],
                                              "aim": [round(float(x), 3) for x in self.to_map(aim)],
                                              "target": topic}, now)
        return rc

    def describe(self, d):
        o = getattr(self, "out", {}).get(d.num)
        p = getattr(self, "p", {}).get(d.num)
        where = "" if p is None else f" @{p[0]:.1f},{p[1]:.1f}"
        how = "chase" if d.num in getattr(self, "chasing", {}) else "defend"
        return f"{how} -> {o[1].split('/')[-1] if o else 'hold'}{where}"
