"""defend: Team 1's defenders. Stop the attackers before they reach the goal tag (6).

  python swarm_policies.py --run defend:6,7,8 --run formation:4,5           # dry-run
  python swarm_policies.py --run defend:6,7,8 --run formation:4,5 --fly

Tactics (the scripted guard our simulation policies are built on, Team 1 repo
simulation_infra/.../tello_defend/guard.py, here in numpy for this infra):
  * one MARKER per attacker (cheapest pairing by distance): flies at the attacker's predicted
    position (lead intercept) and rams it -- capture is by contact;
  * every spare defender is a GOALIE: holds the line between the goal and the attacker
    closest to it, GOALIE_R from the goal, and takes over anything that gets past a marker;
  * defenders never close on each other inside SEP_M; heading is never changed (yaw rc 0).
Works for any number of defenders and attackers (2 v 3, 3 v 5, ...): the roles re-pair every tick.

Where things are:
  attackers   self.bus "formation/drone<N>": p in the map frame (x right, y up, z out of
              tag 10), from swarm_formation.py's tag localisation; stale (> MAX_AGE) = ignored
  defenders   these drones have no camera feed in this runner, so their position is dead
              reckoned: they start on the START spots printed at launch (facing tag 10), then
              integrate the velocity they were commanded (H_GAIN m/s per rc unit); height is
              the drone's own range sensor. Place them carefully; drift grows with time.
Publishes "defend/drone<N>": {"p", "role", "target"} for anyone watching.

Environment (since --run takes no options):
  DEFEND_START="x,z;x,z;x,z"   floor spots in the map frame, one per defender in --run order
  DEFEND_HGAIN=0.021           horizontal m/s per rc unit (Team 1 measured 0.021 on TELLO-7)
  DEFEND_SPEED=0.65            horizontal speed cap, m/s (battery)
  DEFEND_GOAL_TAG=6
Needs numpy and pyyaml (in requirements.txt).
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml

from .base import Policy, hold_height

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
