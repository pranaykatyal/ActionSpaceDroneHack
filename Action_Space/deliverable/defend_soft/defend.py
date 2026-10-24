"""defend: Team 1's trained 3v2 defense policy (soft contact, ES gen 10), 3 defenders vs 2 attackers.

  python swarm_policies.py --run defend:3,4,5,6,7 --duration 45          # dry-run
  python swarm_policies.py --run defend:3,4,5,6,7 --duration 45 --fly

Drop-in for swarm_ops/policies/defend.py (same class DefendPolicy, same "defend" name in POLICIES):
copy defend.py, defend_checkpoint.npz (or .pt), defend_map.yaml and defend_tags.yaml into
swarm_ops/policies/. The group is 5 drones: the first 3 defend, the last 2 attack. Attackers are
scripted: they fly at the asset at ATTACK_SPEED_MPS, as in training. Defenders run Team 1's network
(the hosts' skrl IPPO MLP 23-256-128-64-4, one per defender) from CHECKPOINT: defend_checkpoint.npz
(numpy only) or defend_checkpoint.pt (needs torch) next to this file, or $DEFEND_CHECKPOINT.

Trained on Team1-Defend-3v2-v0 (the hosts' AS-Defend-v0 with 3 defenders). Its 23 inputs per defender:
own position relative to the asset centre (SITE_Z up), own velocity, gravity (level: 0, 0, -1), each
attacker's position and velocity relative to the defender (zero once caught), the attackers' alive
flags. Output: world velocity [vx, vy, vz] x 1 m/s (+ yaw rate, unused). Hosts' evaluate.py, 3v2:
1.6% breach over ~3,000 episodes (chase 99.8%).

Map: localization and the asset both come from ONE map, MAP_PATH: $DEFEND_MAP / $DEFEND_TAGS if set,
else defend_map.yaml / defend_tags.yaml next to this file (the user's TELLO-3 map of 2026-10-24 09:56 and
its measured tag sizes), else the hosts' config/map.yaml + tags.yaml. Setup prints which, and refuses to
fly unless tag 6 is a far-wall tag (faces tag 10, 5.5-7.5 m out, inside the room).
Asset: centre at tag 6's (X, Y), SITE_Z up. Breach (the sim's rule): a live attacker within
SITE_RADIUS_M (horizontal) of the asset centre, about 1.5 m in front of the far wall on axis.
Far wall: the nearest mapped tag facing tag 10 (tags 6 and 9 here); nobody is driven closer than
FAR_WALL_MARGIN_M to it.

Inputs are moved to the present before the network sees them: every drone's last fix is extrapolated
by its age + VIDEO_DELAY_S + CMD_DELAY_S at its estimated velocity (the constant-velocity predictor
scored in the sim, Team1-Defend-*-PredictCV). Separation: commands are clamped to what MAX_RC_DEFENDER
can fly, then the closing speed towards any teammate inside SEP_SAFE_M is removed and a push apart is
added. After a catch, defenders within RETREAT_M of the caught attacker back off for RETREAT_S.

Roles: the network was trained with defender_0 on attacker slot 0, defender_1 on slot 1, defender_2
as the goalkeeper near the asset; at GO the defender nearest the asset becomes defender_2 and the
other two are paired with the attackers left -> right (Y).

Positions: the umbrella's drones only report height, so this policy starts its own video + AprilTag
localization per drone (swarm_formation.py's functions).
Room frame: tag 10 = origin; X = straight out of the tag-10 wall (map z), Y = map x, Z = height above
the floor (Tello tof sensor, checked against the tag height). The policy commands world velocities;
the rc is in each drone's heading frame, so every command is rotated by its yaw, and every drone
turns to face tag 10 (as swarm_formation.control()).

Episode: once every drone sees a tag, roles are fixed and GO. An attacker within CAPTURE_M of a
defender is caught; one inside the asset radius is a breach. Ends on both caught (DEFENDED), a breach,
or EPISODE_S: then every drone lands. Attackers that land because they lost their tags are reported
separately, not as caught.
Placement (asset = tag 6 on the far wall): all drones FACING tag 10.
- Attackers 1-2 m in front of the tag-10 wall (X), one on each side of the tag-10 axis: one at Y <= -0.9,
  one at Y >= +0.9, at least 1.5 m apart, and |Y| <= 1.2 x X so tag 10 stays within ~50 deg of its
  face (e.g. X 1.0-1.2 at Y -1.0..-1.2 and +1.0..+1.2). They both fly at the asset and converge, and
  this keeps them out of the defenders' sight lines to tag 10.
- The two flank defenders on the trained 2.5 m ring around the asset at about +/-35 deg: about 2.0 m
  in front of the far wall, 1.4 m either side of tag 6.
- The goalkeeper 1.6-2.0 m in front of tag 6 on its axis, so it is clearly the one nearest the asset.
- Defenders 0.5 m+ apart.
At the end (both caught, breach or time up) the attackers land at once; a defender that just made a
catch first backs away from it for up to RETREAT_S, then everyone lands.
Before flying: clear the floor of chairs and boxes (~0.45 m tall), at least in the attackers' start area
(X 0.8-2.0): the tof height reads low over them, and a slow crossing can get past the height check.
Hovering at each start spot, the printed Z should match the tag height.
"""
import math
import os
import socket
import sys
import threading
import time

import numpy as np

try:
    from .base import Policy
except ImportError:  # loaded on its own (parity test, tooling): no umbrella base class needed
    Policy = object

# ======================= SETTINGS =======================
# Team 1's checkpoint (skrl IPPO, Team1-Defend-3v2-v0): .npz (numpy) or .pt (torch) next to this file, or $DEFEND_CHECKPOINT
_HERE = os.path.dirname(os.path.abspath(__file__))


def _beside(env, names, default=None):
    """$env, else the first of names next to this file that exists, else default."""
    if os.environ.get(env):
        return os.environ[env]
    return next((os.path.join(_HERE, n) for n in names if os.path.exists(os.path.join(_HERE, n))), default)


CHECKPOINT = _beside("DEFEND_CHECKPOINT", ("defend_checkpoint.npz", "defend_checkpoint.pt"), os.path.join(_HERE, "defend_checkpoint.pt"))
MAP_PATH = _beside("DEFEND_MAP", ("defend_map.yaml",))      # None: the hosts' config/map.yaml
TAGS_PATH = _beside("DEFEND_TAGS", ("defend_tags.yaml",))   # None: the hosts' config/tags.yaml
PERIOD_S = 0.074            # the umbrella's tick = one policy step (13.5 Hz; trained at 60 Hz, sample-and-hold)
EPISODE_S = 25.0             # after GO, then land all
READY_TIMEOUT_S = 20.0       # every drone must see a tag by then, else land all
ATTACK_SPEED_MPS = 0.3       # scripted attackers, flying at the asset (sim trained on 0.4-0.9)
CAPTURE_M = 0.45             # defender-attacker distance counted as a catch (sim 0.35)
SITE_RADIUS_M = 1.5          # breach: a live attacker this close (horizontal) to the asset centre (sim site_radius)
SITE_Z = 1.25                # asset centre height in the sim (site_height 2.5 / 2), m
MAX_RC_DEFENDER = 60         # defender stick limit (100 = ~0.75 m/s)
ARENA_Y = (-2.00, 2.50)      # room walls (Y = map x): hosts' measured left wall, tag 8's right wall, m
SIDE_WALL_MARGIN_M = 0.30    # Y bounds = the walls moved in by this (the fence alone stopped drones 0-15 cm off a wall)
ARENA_Z = (0.30, 2.40)       # floor margin .. below the 2.74 m ceiling, m
TAG10_WALL_X = 0.0           # tag 10's wall, m
FAR_WALL_MARGIN_M = 0.20     # X bound = nearest far-wall tag - this
GEOFENCE_M = 0.30            # no motion further outwards when the predicted position is this close to a bound
FENCE_LOOKAHEAD_S = 0.8      # predicted position for the fence: P + V * this (stopping distance at ~0.4 m/s)
FENCE_PUSH_MPS = 0.15        # inside the margin: pushed back in at this speed
VIDEO_DELAY_S = 0.20         # camera frame -> tag fix (Tello H.264 + decode + detect); measure if you can
CMD_DELAY_S = 0.10           # rc packet -> motion
SEP_SAFE_M, SEP_STOP_M = 0.7, 0.4    # teammates: no closing speed inside SAFE, pushed apart; full stop inside STOP
SEP_K_MPS_PER_M = 1.0        # push-apart gain (0.3 m inside SAFE -> 0.3 m/s)
RETREAT_M, RETREAT_S, RETREAT_MPS = 0.7, 1.5, 0.4   # after a catch: back off from the caught attacker
BRAKE_S = 0.5                # a blind drone (told to hover) is taken to stop this far past its predicted position
FLOOR_M = 1.22               # tag 10 center above the floor (measured), used only without a tof reading
TOF_WIN_S = 0.4             # tof judged over this window (real crossings arrive over 2-3 readings)
TOF_MOVE_M = 0.25            # tof moved more than this in the window ...
TAG_STILL_M = 0.12           # ... while fresh tag heights moved less than this: flying over something; Z kept continuous
TOF_AGREE_S = 1.0            # a tof offset is dropped once raw tof has agreed with the tag height this long
TOF_OFF_RATE_MPS = 0.3       # the applied offset moves towards its target at most this fast (no Vz spike)
TOF_HOLD_S = 0.5             # no tof reading at all this long: use the tag height
MAX_SPEED_MPS = 1.0          # policy action 1.0 = 1 m/s (sim max_lin_speed, all three axes)
VZ_SCALE = 1.0               # policy up action scale (trained unscaled)
GAIN = {"right": 0.008, "forward": 0.00731, "up": 0.006}   # m/s per rc unit (tello_motion_fit.yaml)
RC_DEADBAND = 5
V_MAX_DEFENDER = min(GAIN["right"], GAIN["forward"]) * (MAX_RC_DEFENDER - RC_DEADBAND)  # what MAX_RC_DEFENDER flies, m/s
KP_YAW, YAW_DEADBAND_DEG, MAX_RC_YAW = 1.0, 4.0, 30
VEL_ALPHA = 0.4              # velocity estimate smoothing
POSE_ALPHA = 0.35            # position smoothing (as swarm_formation)
JUMP_MAX_M, JUMP_ACCEPT_S = 0.5, 1.0
FIX_STALE_S = 0.5            # no fresh tag fix: hover in place (turning to face tag 10)
LOST_LAND_S = 3.0            # no fix this long: that drone lands
CLIP_POS, CLIP_VEL = 12.0, 3.0   # the env's obs_clip_pos / obs_clip_vel (relative attacker terms only)
SCALER_CLIP, SCALER_EPS = 5.0, 1e-8   # skrl RunningStandardScaler
DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
AGENTS = ("defender_0", "defender_1", "defender_2")
LAYERS = ("net_container.0", "net_container.2", "net_container.4", "policy_layer")
# ========================================================


def _sf():
    """swarm_formation (numpy/cv2/apriltags), imported only when defend runs."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import swarm_formation
    return swarm_formation


def load_weights(path=CHECKPOINT):
    """skrl IPPO checkpoint -> per defender: observation scaler mean/var, policy layers W0..W3/b0..b3.

    .pt: the skrl checkpoint ({agent: {"policy": state_dict, "observation_preprocessor": ...}}), needs torch.
    .npz: the same arrays exported as "<agent>/mean", "<agent>/var", "<agent>/W0".."<agent>/b3" (numpy only).
    """
    if path.endswith(".npz"):
        z = np.load(path)
        w = {k: z[k].astype(np.float64) for k in z.files}
        w["agents"] = [a for a in AGENTS if f"{a}/W0" in w]
        return w
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = {"agents": [a for a in AGENTS if a in ck]}
    for agent in w["agents"]:
        pol, pre = ck[agent]["policy"], ck[agent]["observation_preprocessor"]
        w[f"{agent}/mean"] = pre["running_mean"].double().numpy()
        w[f"{agent}/var"] = pre["running_variance"].double().numpy()
        for i, layer in enumerate(LAYERS):                   # three ELU layers, then the mean-action layer
            w[f"{agent}/W{i}"] = pol[f"{layer}.weight"].double().numpy()
            w[f"{agent}/b{i}"] = pol[f"{layer}.bias"].double().numpy()
    return w


def act(w, agent, obs):
    """obs (23,) -> deterministic action (4,) in [-1, 1]: skrl's observation scaler + the policy's mean action."""
    x = np.clip((obs - w[f"{agent}/mean"]) / (np.sqrt(w[f"{agent}/var"]) + SCALER_EPS), -SCALER_CLIP, SCALER_CLIP)
    for i in range(4):
        x = x @ w[f"{agent}/W{i}"].T + w[f"{agent}/b{i}"]
        if i < 3:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
    return np.clip(x, -1.0, 1.0)


def build_obs(P, V, site, att_P, att_V, alive):
    """The 23 numbers of Team1-Defend-3v2-v0 (hosts' TelloDefendEnv._get_observations), in its order.

    P, V: the defender's room position / velocity; site: asset centre; att_P, att_V: (2, 3); alive: (2,) bool.
    """
    m = np.asarray(alive, float)[:, None]
    rel_p = np.clip(np.asarray(att_P) - P, -CLIP_POS, CLIP_POS) * m
    rel_v = np.clip(np.asarray(att_V) - V, -CLIP_VEL, CLIP_VEL) * m
    return np.concatenate([P - site, V, [0.0, 0.0, -1.0], rel_p.reshape(-1), rel_v.reshape(-1), m[:, 0]])


def to_room(p):
    """swarm_formation position (x right, y down, z into tag 10) -> room (X out of the wall, Y, Z)."""
    return np.array([-p[2], p[0], -p[1] + FLOOR_M])


def normal_room(R_int):
    """A tag's face normal (out of the print) in the room frame, from its swarm_formation pose.

    load_tag_files stores R_int = C R_map C with C = diag(1, -1, -1), so n_map = -C R_int[:, 2];
    room = (map z, map x, map y)."""
    n_map = -np.diag([1.0, -1.0, -1.0]) @ np.asarray(R_int)[:, 2]
    return np.array([n_map[2], n_map[0], n_map[1]])


def yaw_to_tag10(P, psi_deg):
    """rc yaw that turns a drone at room position P towards tag 10 (swarm_formation.face_tag_yaw)."""
    err = (math.degrees(math.atan2(-P[1], P[0])) - psi_deg + 180.0) % 360.0 - 180.0
    return 0 if abs(err) < YAW_DEADBAND_DEG else int(np.clip(KP_YAW * err, -MAX_RC_YAW, MAX_RC_YAW))


def to_rc(v, psi_deg, max_rc, P):
    """Room-frame velocity (m/s) -> rc (right, forward, up, yaw) for a drone at yaw psi, turning towards tag 10."""
    vx_i, vz_i = v[1], -v[0]
    s, c = math.sin(math.radians(psi_deg)), math.cos(math.radians(psi_deg))
    fwd, right = vx_i * s + vz_i * c, vx_i * c - vz_i * s    # as swarm_formation.control()

    def stick(u, g):
        return 0 if abs(u) < 0.03 else int(math.copysign(min(max_rc, RC_DEADBAND + abs(u) / g), u))

    return stick(right, GAIN["right"]), stick(fwd, GAIN["forward"]), stick(v[2], GAIN["up"]), yaw_to_tag10(P, psi_deg)


class Eye:
    """One drone's video + tag localization; position/velocity in the room frame."""

    def __init__(self, d, sf, K, D, tag_map, sizes, mapcheck):
        self.d, self.sf, self.K, self.D = d, sf, K, D
        self.tag_map, self.sizes, self.mapcheck = tag_map, sizes, mapcheck
        self.stop = threading.Event()
        self.frame, self.frame_id = None, 0
        self.fix, self.seen_t, self.jump_since = None, 0.0, None   # fix = (p, psi, t)
        self.P, self.V, self.psi, self.t = None, np.zeros(3), 0.0, None
        self.tof_t, self.tof_off, self.tof_target, self.agree_since = 0.0, 0.0, 0.0, None   # tof continuity state
        self.tof_win, self.h_t = [], None                    # [(t, tof, tag_z or None)], last height update time
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

    def _height(self, tag_z, fresh, now):
        """Z from tof plus an offset, so flying over an object (a chair, another drone) doesn't move Z.

        tof is trusted unless, over TOF_WIN_S, it moved more than TOF_MOVE_M while fresh tag heights moved less
        than TAG_STILL_M (a tag pose flip moves the tag height, not tof): then the offset target re-anchors so Z
        stays where it was. The target drops to 0 once raw tof agrees with fresh tag heights for TOF_AGREE_S.
        The applied offset follows its target at TOF_OFF_RATE_MPS. No tof reading for TOF_HOLD_S: the tag height.
        Returns (Z, hold_vz)."""
        h = self.d.height_cm()
        dt = 0.0 if self.h_t is None else max(now - self.h_t, 0.0)
        self.h_t = now
        if h is None:
            if self.P is not None and now - self.tof_t < TOF_HOLD_S:
                return self.P[2], True
            self.tof_win, self.tof_off, self.tof_target = [], 0.0, 0.0
            return tag_z, False
        tof = h / 100.0
        self.tof_win = [w for w in self.tof_win if now - w[0] <= TOF_WIN_S] + [(now, tof, tag_z if fresh else None)]
        tags = [w[2] for w in self.tof_win if w[2] is not None]
        t0, tof0 = self.tof_win[0][0], self.tof_win[0][1]
        if (now - t0 > 0.5 * TOF_WIN_S and abs(tof - tof0) > TOF_MOVE_M and len(tags) >= 2
                and max(tags) - min(tags) < TAG_STILL_M):
            self.tof_target = (tof0 + self.tof_target) - tof     # Z where it was before the window; ~0 when tof steps back
            if abs(self.tof_target) < 0.05:
                self.tof_target = 0.0
            self.tof_win, self.agree_since = [(now, tof, tag_z if fresh else None)], None
        elif self.tof_target != 0.0 and fresh:
            if abs(tof - tag_z) < 0.15:
                self.agree_since = self.agree_since or now
                if now - self.agree_since > TOF_AGREE_S:
                    self.tof_target, self.agree_since = 0.0, None
            else:
                self.agree_since = None
        step = TOF_OFF_RATE_MPS * dt
        self.tof_off += float(np.clip(self.tof_target - self.tof_off, -step, step))
        self.tof_t = now
        return tof + self.tof_off, False

    def update(self, now):
        """Room position (Z from the tof sensor, kept continuous), smoothed velocity, yaw."""
        if self.fix is None:
            return
        p, psi, fix_t = self.fix
        P = to_room(p)
        P[2], keep_vz = self._height(P[2], now - fix_t < TOF_WIN_S, now)
        if self.t is not None and fix_t > self.t:
            v = np.clip((P - self.P) / max(fix_t - self.t, 1e-3), -CLIP_VEL, CLIP_VEL)
            if keep_vz:
                v[2] = self.V[2]
            self.V = VEL_ALPHA * v + (1 - VEL_ALPHA) * self.V
        if self.t is None or fix_t > self.t:
            self.P, self.t = P, fix_t
        self.psi = psi

    def predicted(self, now):
        """Position moved to when a command sent now acts: fix age + video delay (XY), command delay (all).

        A blind drone (no fix for FIX_STALE_S) is being told to hover: taken on towards where it stops, up to
        BRAKE_S further, growing with the time it has been blind (a full step at once credits ghost catches)."""
        raw_age = max(now - self.t, 0.0) if self.t is not None else 0.0
        h_xy = min(raw_age, FIX_STALE_S) + VIDEO_DELAY_S + CMD_DELAY_S + min(max(raw_age - FIX_STALE_S, 0.0), BRAKE_S)
        h_z = CMD_DELAY_S                                    # Z comes from the live tof reading
        return self.P + self.V * np.array([h_xy, h_xy, h_z])


class DefendPolicy(Policy):
    name = "defend"

    def setup(self, drones):
        super().setup(drones)
        if len(self.drones) != 5:
            raise SystemExit(f"defend needs 5 drones (3 defenders, then 2 attackers), got {len(self.drones)}")
        if not os.path.exists(CHECKPOINT):
            raise SystemExit(f"defend: no policy checkpoint at {CHECKPOINT} (copy it there or set DEFEND_CHECKPOINT)")
        sf = self.sf = _sf()
        map_path, tags_path = MAP_PATH or str(sf.MAP_PATH), TAGS_PATH or str(sf.TAGS_PATH)
        _, tag_map, sizes = sf.load_tag_files(map_path, tags_path)
        src = lambda env, path: f" (from ${env})" if os.environ.get(env) else ""  # noqa: E731
        print(f"[defend] map {map_path}{src('DEFEND_MAP', map_path)}, tag sizes {tags_path}{src('DEFEND_TAGS', tags_path)}: "
              f"tags {sorted(tag_map)}", flush=True)
        if 6 not in tag_map:
            raise SystemExit(f"defend: tag 6 (the asset) is not in {map_path}")
        tag6, n6 = to_room(tag_map[6][1]), normal_room(tag_map[6][0])
        if not (n6[0] < -0.9 and 5.5 <= tag6[0] <= 7.5 and ARENA_Y[0] + 0.5 <= tag6[1] <= ARENA_Y[1] - 0.5):
            raise SystemExit(f"defend: tag 6 in {map_path} is not on the far wall (room X {tag6[0]:.2f}, Y {tag6[1]:.2f}, "
                             f"normal {np.round(n6, 2)}); this map does not match the room the policy is set up for. "
                             f"Set DEFEND_MAP to the map of the room as it is now.")
        far = [to_room(t)[0] for R, t in tag_map.values() if normal_room(R)[0] < -0.9]
        self.far_wall_x = float(min(far))
        self.site = np.array([float(tag6[0]), float(tag6[1]), SITE_Z])
        self.arena = ((TAG10_WALL_X, self.far_wall_x - FAR_WALL_MARGIN_M),
                      (ARENA_Y[0] + SIDE_WALL_MARGIN_M, ARENA_Y[1] - SIDE_WALL_MARGIN_M), ARENA_Z)
        print(f"[defend] asset centre {np.round(self.site, 2)} (tag 6), breach within {SITE_RADIUS_M} m of it "
              f"(X {self.site[0] - SITE_RADIUS_M:.2f} on axis); far wall {self.far_wall_x:.2f} (nearest tag facing tag 10), "
              f"arena X {self.arena[0][0]:.2f}..{self.arena[0][1]:.2f}, Y {self.arena[1][0]:.2f}..{self.arena[1][1]:.2f}, "
              f"Z {ARENA_Z}", flush=True)
        mapcheck = sf.MapCheck(tag_map)
        self.eyes = {}
        for d in self.drones:
            K, D = sf.load_camera(str(sf.camera_file(d.num, sf.CAMERA_DIR)))
            self.eyes[d.num] = Eye(d, sf, K, D, tag_map, sizes, mapcheck)
        self.w = load_weights()
        if list(self.w["agents"]) != list(AGENTS):
            raise SystemExit(f"defend: {CHECKPOINT} has agents {self.w['agents']}, need {list(AGENTS)} (a 3v2 checkpoint)")
        self.agents = list(AGENTS)
        print(f"[defend] policy {CHECKPOINT}", flush=True)
        self.defenders, self.attackers = list(self.drones[:3]), list(self.drones[3:])
        self.alive, self.lost, self.phase, self.result = [True, True], [False, False], "START", None
        self.caught_at = []                                  # [(room position, time)] for the retreat
        self.cmds, self.tick_t, self.t0, self.last_print = {}, 0.0, None, 0.0
        self.end_t = None                                    # when the result was set (ENDING phase)

    def step(self, d, t):
        now = time.time()
        if now - self.tick_t > 0.5 * PERIOD_S:                 # once per umbrella tick, for everyone
            self.tick_t = now
            self._tick(now)
        return self.cmds.get(d.num, (0, 0, 0, 0))

    def describe(self, d):
        if d in self.attackers:
            return f"attacker_{self.attackers.index(d)}"
        return self.agents[self.defenders.index(d)] if self.phase == "GO" else "defender"

    def _assign_roles(self):
        """Attacker slots left -> right (Y); defender_2 = the defender nearest the asset (goalkeeper),
        defender_0 / defender_1 = the other two left -> right, so each faces the attacker on its side."""
        eyes = self.eyes
        self.attackers.sort(key=lambda a: eyes[a.num].P[1])
        keeper = min(self.defenders, key=lambda d: np.linalg.norm(eyes[d.num].P[:2] - self.site[:2]))
        pair = sorted((d for d in self.defenders if d is not keeper), key=lambda d: eyes[d.num].P[1])
        self.defenders = pair + [keeper]

    def _active(self, d, now):
        """Still flying and not being landed (a blind drone still hovers: it stays in, at its predicted stop)."""
        return (self.cmds.get(d.num, 0) is not None and d.land_thread is None and not d.landed
                and self.eyes[d.num].t is not None)

    def _hover(self, e):
        """No fresh fix: hold position, keep turning towards tag 10 to find it again."""
        return (0, 0, 0, yaw_to_tag10(e.P, e.psi)) if e.P is not None else (0, 0, 0, 0)

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
                self._assign_roles()
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
                print("[defend] >>> RESULT: NO TAGS -> land all", flush=True)
                self._end("NO TAGS")
            elif now - self.last_print >= 1.0:
                self.last_print = now
                print("[defend] waiting for tags: " + ", ".join(
                    f"{d.name}: {eyes[d.num].reason}" for d in self.drones if not eyes[d.num].has_fix(now)), flush=True)
            return
        if self.phase == "END":
            return
        if self.phase == "ENDING":
            self._ending(now)
            return

        pred = {n: e.predicted(now) for n, e in eyes.items() if e.P is not None}
        live_def = [d for d in self.defenders if self._active(d, now)]
        cmds = {}
        for k, a in enumerate(self.attackers):
            e = eyes[a.num]
            if not self.alive[k]:
                cmds[a.num] = None
            elif a.land_thread is not None or a.landed:
                self.alive[k], self.lost[k], cmds[a.num] = False, True, None
                print(f"[defend] {a.name} landed (umbrella) -> out, not caught", flush=True)
            elif any(np.linalg.norm(pred[d.num] - pred[a.num]) < CAPTURE_M for d in live_def):
                self.alive[k], cmds[a.num] = False, None
                self.caught_at.append((pred[a.num].copy(), now))
                print(f"[defend] {a.name} CAUGHT", flush=True)
            elif np.linalg.norm(pred[a.num][:2] - self.site[:2]) < SITE_RADIUS_M:
                self.alive[k], cmds[a.num], self.result = False, None, "BREACH"
                print(f"[defend] {a.name} BREACH (inside {SITE_RADIUS_M} m of the asset)", flush=True)
            elif now - e.seen_t > LOST_LAND_S:
                self.alive[k], self.lost[k], cmds[a.num] = False, True, None
                print(f"[defend] {a.name} lost its tags -> lands, not caught", flush=True)
            elif not e.has_fix(now):
                cmds[a.num] = self._hover(e)
            else:
                to_site = self.site[:2] - pred[a.num][:2]
                v_xy = ATTACK_SPEED_MPS * to_site / max(float(np.linalg.norm(to_site)), 1e-6)
                v = np.array([v_xy[0], v_xy[1], np.clip(self.heights[a.num] - e.P[2], -0.2, 0.2)])
                cmds[a.num] = to_rc(self._fence(e, v, now), e.psi, 100, e.P)
        att_P = [pred[a.num] for a in self.attackers]
        att_V = [eyes[a.num].V for a in self.attackers]
        for i, d in enumerate(self.defenders):
            e = eyes[d.num]
            if self.cmds.get(d.num, 0) is None or d.land_thread is not None or d.landed or now - e.seen_t > LOST_LAND_S:
                cmds[d.num] = None
                continue
            if not e.has_fix(now):
                cmds[d.num] = self._hover(e)
                continue
            P = pred[d.num]
            a = act(self.w, self.agents[i], build_obs(P, e.V, self.site, att_P, att_V, self.alive))
            v = np.array([a[0], a[1], a[2] * VZ_SCALE]) * MAX_SPEED_MPS
            v = self._separate(d, P, v, live_def, pred, now)
            cmds[d.num] = to_rc(self._fence(e, v, now), e.psi, MAX_RC_DEFENDER, e.P)
        self.cmds = cmds

        if now - self.last_print >= 1.0:
            self.last_print = now
            for d in self.drones:
                e = eyes[d.num]
                if e.P is None:
                    continue
                print(f"[defend] {d.name} {self.describe(d):<11} X {e.P[0]:.2f} Y {e.P[1]:.2f} Z {e.P[2]:.2f} "
                      f"psi {e.psi:+.0f} {'SEE' if e.has_fix(now) else 'LOST'}  rc {cmds.get(d.num)}", flush=True)
        if self.result is None and not any(self.alive):
            caught = 2 - sum(self.lost)
            self.result = "DEFENDED" if not any(self.lost) else f"NO BREACH ({caught} caught, {sum(self.lost)} lost/landed)"
        if self.result is None and now - self.t0 > EPISODE_S:
            self.result = f"TIME UP, {sum(self.alive)} attacker(s) left"
        if self.result is not None:
            print(f"[defend] >>> RESULT: {self.result} after {now - self.t0:.1f} s -> attackers land; "
                  f"a defender that just caught one backs off first, then all land", flush=True)
            self.alive, self.phase, self.end_t = [False, False], "ENDING", now
            self._ending(now)

    def _ending(self, now):
        """After the result: attackers land; a defender within RETREAT_M of a catch younger than RETREAT_S keeps
        backing away from it (fenced, while it can see tags); everyone else lands. Then everything lands."""
        eyes, cmds, backing = self.eyes, {a.num: None for a in self.attackers}, False
        for d in self.defenders:
            e = eyes[d.num]
            cmds[d.num] = None
            if (now - self.end_t < RETREAT_S and self.cmds.get(d.num, 0) is not None and d.land_thread is None
                    and not d.landed and e.P is not None):         # blind too: it backs off from its predicted position
                P = e.predicted(now)
                for pos, t in self.caught_at:
                    g = (P - pos)[:2]
                    dist = float(np.linalg.norm(g))
                    if now - t < RETREAT_S and dist < RETREAT_M:
                        u = g / dist if dist > 1e-6 else np.array([-1.0, 0.0])
                        v = np.array([RETREAT_MPS * u[0], RETREAT_MPS * u[1], 0.0])
                        cmds[d.num], backing = to_rc(self._fence(e, v, now), e.psi, MAX_RC_DEFENDER, e.P), True
                        break
        self.cmds = cmds
        if not backing:
            self._end(self.result)

    def _separate(self, d, P, v, live_def, pred, now):
        """Clamp to the flyable speed, then keep teammates apart and back off from a just-caught attacker."""
        v = v.copy()
        n = float(np.hypot(v[0], v[1]))
        if n > V_MAX_DEFENDER:
            v[:2] *= V_MAX_DEFENDER / n
        for pos, t in self.caught_at:                        # retreat: the caught attacker hovers or drops in place
            g = (P - pos)[:2]
            dist = float(np.linalg.norm(g))
            if now - t < RETREAT_S and dist < RETREAT_M:
                u = g / dist if dist > 1e-6 else np.array([-1.0, 0.0])
                return np.array([RETREAT_MPS * u[0], RETREAT_MPS * u[1], 0.0])
        scale, rep = 1.0, np.zeros(3)
        for o in live_def:
            if o is d:
                continue
            g = P - pred[o.num]
            dist = float(np.linalg.norm(g))
            if dist < SEP_SAFE_M:
                u = g / dist if dist > 1e-6 else np.array([0.0, 1.0, 0.0])
                closing = float(v @ u)
                if closing < 0:
                    v = v - closing * u                      # no speed towards the teammate
                rep += SEP_K_MPS_PER_M * (SEP_SAFE_M - dist) * u
                scale = min(scale, max(0.0, (dist - SEP_STOP_M) / (SEP_SAFE_M - SEP_STOP_M)))
        return v * scale + rep

    def _end(self, why):
        self.phase, self.cmds = "END", {n: None for n in self.eyes}   # None: the umbrella lands it
        for e in self.eyes.values():
            e.stop.set()

    def _fence(self, e, v, now):
        """Room bounds on the position predicted FENCE_LOOKAHEAD_S ahead; pushed back in when inside the margin."""
        v = v.copy()
        ahead = e.predicted(now) + e.V * FENCE_LOOKAHEAD_S
        for k, (lo, hi) in enumerate(self.arena):
            if ahead[k] < lo + GEOFENCE_M and v[k] < 0 or ahead[k] > hi - GEOFENCE_M and v[k] > 0:
                v[k] = 0.0
            if e.P[k] < lo + GEOFENCE_M:
                v[k] = max(v[k], FENCE_PUSH_MPS)
            elif e.P[k] > hi - GEOFENCE_M:
                v[k] = min(v[k], -FENCE_PUSH_MPS)
        return v
