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
  defenders   these drones get no camera feed in this runner, so their position is dead
              reckoned: they start on the START spots printed at launch (facing tag 10) and
              integrate the velocity they were commanded (H_GAIN m/s per rc unit); height is
              the drone's own range sensor. Place them carefully; drift grows with time.
Publishes "defend/drone<N>": {"p", "aim", "target"} (map frame) for anyone watching.

Environment (since --run takes no options):
  DEFEND_START="x,z;x,z;x,z"   floor spots in the map frame, one per defender in --run order
  DEFEND_HGAIN=0.021           horizontal m/s per rc unit (Team 1 measured 0.021 on TELLO-7)
  DEFEND_SPEED=0.65            horizontal speed cap, m/s (battery)
  DEFEND_GOAL_TAG=6
Needs numpy and pyyaml (in requirements.txt).
"""
import itertools
import os
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
# AS-Defend-RoomReal3-v0 guard settings (3 v 2, random attacker paths: 25.7% breached episodes)
GUARD = {"speed": 0.65, "intercept_radius": 1.35, "park_radius": 3.0, "dock_gap": 0.0,
         "guard_offset": 0.0, "park_lead": 0.0, "dock_range": 0.0, "gain": 4.0}


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
        cfg = dict(GUARD, speed=self.vmax)
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

    def step(self, d, t):
        now = time.time()
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
        return f"defend -> {o[1].split('/')[-1] if o else 'hold'}{where}"
