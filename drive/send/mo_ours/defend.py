"""defend: Mo's defend.py (cameras for all 5 drones, scripted attackers, fallback chaser) with Team 1's
scripted intercept guard as the defenders' tactics instead of a trained network. 3 defenders vs 2 attackers.

  python swarm_policies.py --run defend:3,4,5,6,7 --duration 45          # dry-run
  python swarm_policies.py --run defend:3,4,5,6,7 --duration 45 --fly

The group is 5 drones: the first 3 defend, the last 2 attack. Attackers are scripted: they
fly away from tag 10 towards tag 6 at ATTACK_SPEED_MPS, like in the sim. Defenders run the
intercept guard (the numpy port of Team 1's sim guard, as in our defend.py): every attacker gets
its own MARKER flying at its lead-intercept point to ram it, the spare defender is a GOALIE on the
crossing point of the attacker closest to tag 6; no checkpoint, no torch.

What changed from Mo's file:
  tactics      the guard in the room frame (identical to the guard's own frame: X out of the
               tag-10 wall, Y = map x, Z up), with the AS-Defend-RoomReal3 settings (GUARD) and a
               horizontal speed cap of MAX_HSPEED_MPS; the protected site is tag 6's floor spot
  fallback     Mo's chaser, off by default here (it did not help the guard in spoof tests);
               DEFEND_FALLBACK=1 turns it on
  sticks       GAIN is the 0.021 m/s per rc unit Team 1 measured on TELLO-7 (Mo's 0.008 made the
               drones fly ~2.6x faster than commanded)

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
import itertools
import math
import os
import socket
import sys
import threading
import time

import numpy as np

from .base import Policy

# ======================= SETTINGS =======================
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
FALLBACK = os.environ.get("DEFEND_FALLBACK", "0") == "1"
FALLBACK_X = 4.5
FALLBACK_T = 8.0
ARENA_Y = (-0.40, 2.00)      # room bounds the policy was trained in (Y = map x), m
ARENA_Z = (0.15, 1.776)
BEHIND_BREACH_M = 0.5        # X bound = tag 6 + this
GEOFENCE_M = 0.30            # no motion further outwards this close to a bound
SEPARATION_M, SEPARATION_MPS = 0.5, 0.3   # defenders closer than this are pushed apart
FLOOR_M = 1.207              # tag 10 center above the floor, used only without a tof reading
MAX_SPEED_MPS = 1.0          # the fallback chaser's speed
MAX_HSPEED_MPS = 0.65        # the guard's horizontal speed cap (battery)
MAX_CLIMB_MPS = 0.4
VZ_SCALE = 0.3               # the fallback chaser's climb scale (Mo's)
GAIN = {"right": 0.021, "forward": 0.021, "up": 0.006}     # m/s per rc unit (Team 1, TELLO-7)
# AS-Defend-RoomReal3-v0 guard settings (as in our defend.py)
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


def to_room(p):
    """swarm_formation position (x right, y down, z into tag 10) -> room (X out of the wall, Y, Z)."""
    return np.array([-p[2], p[0], -p[1] + FLOOR_M])


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
        sf = self.sf = _sf()
        _, tag_map, sizes = sf.load_tag_files(str(sf.MAP_PATH), str(sf.TAGS_PATH))
        if 6 not in tag_map:
            raise SystemExit(f"defend: tag 6 (the breach line) is not in {sf.MAP_PATH}")
        self.breach_x = float(to_room(tag_map[6][1])[0])
        self.site = to_room(tag_map[6][1])
        self.site[2] = 0.0  # the protected zone stands on the floor below tag 6
        self.arena = ((0.0, self.breach_x + BEHIND_BREACH_M), ARENA_Y, ARENA_Z)
        print(f"[defend] map tags {sorted(tag_map)}; breach line (tag 6) {self.breach_x:.2f} m from tag 10"
              + ("" if abs(self.breach_x - 6.48) < 0.5 else "  WARNING: trained for 6.48 m"), flush=True)
        mapcheck = sf.MapCheck(tag_map)
        self.eyes = {}
        for d in self.drones:
            K, D = sf.load_camera(str(sf.camera_file(d.num, sf.CAMERA_DIR)))
            self.eyes[d.num] = Eye(d, sf, K, D, tag_map, sizes, mapcheck)
        self.guard, self.guard_alive, self.vguard, self.targets = None, None, {}, {}
        self.guard_cfg = dict(load_guard_settings(), speed=MAX_HSPEED_MPS)
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
        k = self.targets.get(d.num)
        return f"guard->atk{k}" if self.phase == "GO" and k is not None else "defender"

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
                for d in self.defenders:
                    print(f"[defend] {d.name} -> guard at {np.round(eyes[d.num].P, 2)}", flush=True)
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
        self._guard(now)
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
                if d.num not in self.vguard:  # no attacker left: hold
                    cmds[d.num] = (0, 0, 0, 0)
                    continue
                v = self.vguard[d.num].copy()
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

    def _guard(self, now):
        """The guard's velocity for every defender (room frame), from the live attackers' camera tracks."""
        eyes = self.eyes
        live = [k for k in range(len(self.attackers)) if self.alive[k]]
        self.vguard, self.targets = {}, {}
        if not live:
            return
        if self.guard is None or live != self.guard_alive:  # the assignment is per attacker count
            self.guard = InterceptGuard(1, len(self.defenders), len(live), **self.guard_cfg)
            self.guard_alive = live
        def_pos = np.stack([eyes[d.num].P for d in self.defenders])[None]
        def_vel = np.stack([eyes[d.num].V for d in self.defenders])[None]
        att_pos = np.stack([eyes[self.attackers[k].num].P for k in live])[None]
        att_vel = np.stack([eyes[self.attackers[k].num].V for k in live])[None]
        site = self.site[None]
        cmd, target, _ = self.guard.compute(def_pos, def_vel, att_pos, att_vel, infer_attacking(att_pos, att_vel, site),
                                            site, np.zeros(1), MAX_HSPEED_MPS)
        for i, d in enumerate(self.defenders):
            v = cmd[0, i] * MAX_HSPEED_MPS
            v[2] = float(np.clip(v[2], -MAX_CLIMB_MPS, MAX_CLIMB_MPS))
            self.vguard[d.num] = v
            self.targets[d.num] = live[int(target[0, i])]
