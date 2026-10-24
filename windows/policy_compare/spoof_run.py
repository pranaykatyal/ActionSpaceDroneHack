"""Spoofed swarm_ops run: a defend policy flies simulated Tellos against simulated attackers on the bus.

  python spoof_run.py <swarm_ops dir> [--policy defend_rl] [--episodes 100] [--svg out.html]

Everything the policy touches is the real thing (policies package, TelemetryBus, map.yaml, step()/rc);
only the drones, the attackers and the clock are fake:
  defenders  rc -> velocity with the measured TELLO-7 stick gain (0.021 m/s per unit), 0.35 s lag,
             0.2 s command delay; height_cm() from the true height
  attackers  start at the tag-10 end, fly to the goal tag at 0.3-0.5 m/s on curved / weaving /
             wandering paths; published on "formation/drone4|5" at 13.5 Hz with 3 cm noise
  outcome    caught: a defender within 0.35 m of an attacker (it drops out);
             breach: an attacker within 0.75 m (horizontal) of the goal tag
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("swarm_ops")
ap.add_argument("--policy", default="defend_rl")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seconds", type=float, default=40.0)
ap.add_argument("--true-gain", type=float, default=0.021, help="the simulated drones' real stick gain")
ap.add_argument("--svg", default="")
ap.add_argument("--att-speed", default="0.3,0.5", help="attacker speed range, m/s")
ap.add_argument("--scenario", default="goal", choices=["goal", "line"],
                help="goal: curved/weaving/wandering paths to tag 6, breach within 0.75 m of it; line: Rohan's "
                     "AS-Defend-Room task, straight out from the tag-10 wall, breach = crossing the tag-6 line")
ap.add_argument("--ideal", action="store_true", help="no sensing noise, 0.1 s response, no command delay (training-like)")
ap.add_argument("--fallback", action="store_true", help="defend_room: Mo's fallback chaser (X > 4.5 m or alive 8 s after GO) and full stick")
ap.add_argument("--messy", action="store_true", help="per-drone gain x U(0.7,1.3) per axis, heading error U(-8,8) deg, "
                "start placement error N(0, 0.15) m per axis, slow random drift (0.05 m/s)")
ap.add_argument("--cam", action="store_true", help="give policies with make_eyes() simulated cameras: "
                "true pose 0.1 s old + noise, 10% of frames dropped")
ap.add_argument("--room-gain", type=float, default=None, help="defend_room: override its GAIN right/forward")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

CLOCK = [1_000_000.0]
time.time = lambda: CLOCK[0]          # the bus and the policies read this
sys.path.insert(0, str(Path(args.swarm_ops).resolve()))
from policies import POLICIES          # noqa: E402
from policies.telemetry import TelemetryBus  # noqa: E402

HZ, TAU, DELAY, UP_GAIN = 13.5, 0.35, 0.2, 0.006
NOISE = 0.03
if args.ideal:
    TAU, DELAY, NOISE = 0.1, 0.0, 0.0
SPEED = [float(v) for v in args.att_speed.split(",")]


class RoomAdapter:
    """Rohan's defend_room.py main loop for one tick, as a Policy: his own build_obs / act / to_rc /
    geofence on camera-like fixes (true pose + 3 cm noise every tick) of all five drones."""
    name = "defend_room"
    bus = None

    def __init__(self):
        import types
        sys.modules.setdefault("termios", types.ModuleType("termios"))
        sf = types.ModuleType("swarm_formation")
        sf.CAMERA_DIR = sf.DONGLES_PATH = sf.MAP_PATH = sf.TAGS_PATH = ""
        sys.modules["swarm_formation"] = sf
        import defend_room as R
        if args.room_gain:
            R.GAIN.update(right=args.room_gain, forward=args.room_gain)
        self.R, self.npz = R, np.load(R.DEFAULT_POLICY)
        self.agents = [str(a) for a in self.npz["agents"]]

    def setup(self, drones):
        import yaml
        self.drones = list(drones)
        m = yaml.safe_load((Path(args.swarm_ops).parent / "config" / "map.yaml").read_text())["tags"]
        fl = [v["t"][1] for v in m.values() if abs(np.array(v["R"])[1, 2]) > 0.8]
        self.floor_y = float(np.median(fl))
        self.goal = np.array(m[6]["t"], float)
        # his placement: 0.3-1 m in front of tag 6, spread across the room, facing tag 10
        xs = np.linspace(-0.2, 1.8, 3)
        self.p = {d.num: np.array([x, self.floor_y, self.goal[2] - 0.6]) for d, x in zip(self.drones, xs)}
        self.tr, self.tick, self.order = {}, None, None

    def _track(self, key, P, now):
        t = self.tr.get(key)
        if t is None:
            t = self.tr[key] = self.R.Track()
            t.P = P
            t.t = now
            return
        if now > t.t:
            v = (P - t.P) / (now - t.t)
            t.V = self.R.VEL_ALPHA * np.clip(v, -3, 3) + (1 - self.R.VEL_ALPHA) * t.V
            t.P, t.t = P, now

    def _sim(self, p):  # map frame -> his sim frame (FLOOR_M is his hard-coded tag-10 height)
        return np.array([p[2], p[0], p[1] + self.R.FLOOR_M])

    def step(self, d, t):
        now = time.time()
        if self.tick != now:
            self.tick = now
            for dd in self.drones:
                self._track(dd.num, self._sim(dd.p + np.random.normal(0, NOISE, 3)), now)
            self.atts, self.alive = [], []
            for topic, (data, age) in sorted(self.bus.since("formation/").items()):
                self._track(topic, self._sim(data["p"]), now)
                self.atts.append(type("A", (), {"num": topic})())
                self.alive.append((data.get("height_cm") or 0) >= 25)
            if self.order is None:  # defender_0 = smallest Y (his trained lanes)
                self.order = sorted(self.drones, key=lambda x: self.tr[x.num].P[1])
        if t < 2.0 or len(self.atts) < 2:
            return (0, 0, 0, 0)
        R, i = self.R, self.order.index(d)
        k = self._chasers(now, t).get(d.num) if args.fallback else None
        if k is not None:  # Mo's defend.py _chase: at the attacker 0.5 s ahead, full speed until within 0.3 m
            me, atk = self.tr[d.num], self.tr[self.atts[k].num]
            rel = (atk.P - me.P) + 0.5 * (atk.V - me.V)
            n = max(float(np.linalg.norm(rel[:2])), 1e-3)
            xy = rel[:2] / n * min(n / 0.3, 1.0) * R.MAX_SPEED_MPS
            v = np.array([xy[0], xy[1], float(np.clip(rel[2] / 0.3, -1.0, 1.0)) * R.VZ_SCALE * R.MAX_SPEED_MPS])
        else:
            a = R.act(self.npz, self.agents[i], R.build_obs(i, self.order, self.atts, self.tr, self.alive))
            v = np.array([a[0] * R.MAX_SPEED_MPS, a[1] * R.MAX_SPEED_MPS, a[2] * R.VZ_SCALE * R.MAX_SPEED_MPS])
        P = self.tr[d.num].P
        for o in self.order:
            gap = P - self.tr[o.num].P
            dist = float(np.linalg.norm(gap))
            if o is not d and 1e-6 < dist < R.SEPARATION_M:
                v += R.SEPARATION_MPS * gap / dist
        return R.to_rc(R.geofence(P, v), 0.0, 100 if args.fallback else 60)

    def _chasers(self, now, t):
        """Mo's defend.py _chasers: one chaser per triggered attacker, most advanced first, nearest free defender."""
        if getattr(self, "_chase_tick", None) == now:
            return self._chase_out
        trig = [k for k, a in enumerate(self.atts) if self.alive[k]
                and (self.tr[a.num].P[0] > 4.5 or t - 2.0 > 8.0)]
        trig.sort(key=lambda k: -self.tr[self.atts[k].num].P[0])
        out, free = {}, list(self.order)
        for k in trig:
            if not free:
                break
            ak = self.tr[self.atts[k].num].P
            o = min(free, key=lambda x: float(np.linalg.norm(self.tr[x.num].P - ak)))
            out[o.num] = k
            free.remove(o)
        self._chase_tick, self._chase_out = now, out
        return out
CATCH, GOAL_R = 0.35, 0.75


class FakeDrone:
    def __init__(self, num, p):
        self.num, self.name = num, f"drone{num}"
        self.p, self.v = np.array(p, float), np.zeros(3)
        self.queue = []
        self.floor_y = 0.0

    def height_cm(self):
        return round((self.p[1] - self.floor_y) * 100)

    def battery(self):
        return 80

    def mess(self, rng):
        self.gx, self.gz = rng.uniform(0.7, 1.3, 2)
        self.yaw = math.radians(rng.uniform(-8, 8))
        self.rng, self.drift = rng, np.zeros(3)

    def fly(self, rc, dt):
        self.queue.append(rc)
        rc = self.queue.pop(0) if len(self.queue) > round(DELAY * HZ) else (0, 0, 0, 0)
        # facing tag 10: a = right (+x), b = forward (-z), c = up (+y)
        want = np.array([rc[0] * args.true_gain, rc[2] * UP_GAIN, -rc[1] * args.true_gain])
        if hasattr(self, "gx"):  # messy drone: its own gains, a heading error, a slow drift
            want[0] *= self.gx
            want[2] *= self.gz
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            want[0], want[2] = c * want[0] - s * want[2], s * want[0] + c * want[2]
            self.drift += -self.drift / 3.0 * dt + self.rng.normal(0, 0.05, 3) * math.sqrt(dt / 1.5) * np.array([1, 0.2, 1])
            want = want + self.drift
        self.v += (want - self.v) * min(dt / TAU, 1.0)
        self.p = self.p + self.v * dt
        self.hist = (getattr(self, "hist", []) + [(CLOCK[0], self.p.copy())])[-10:]


class FakeEye:
    """Stands in for the policies' Eye: the drone's pose 0.1 s ago plus noise; 10% of frames lost."""

    def __init__(self, d, rng):
        self.d, self.rng = d, rng

    def fix_map(self, now):
        old = [h for h in getattr(self.d, "hist", []) if h[0] <= now - 0.1]
        if not old or self.rng.random() < 0.1:
            return None
        t, p = old[-1]
        return p + self.rng.normal(0, max(NOISE, 0.03), 3), t


def attacker_path(rng, start, goal, speed):
    kind = "straight" if args.scenario == "line" else rng.choice(["curve", "weave", "wander"])
    mid = (start + goal) / 2 + np.array([rng.uniform(-1.5, 1.5), 0, rng.uniform(-0.8, 0.8)])
    amp, rate = rng.uniform(0.3, 0.8), rng.uniform(0.4, 0.9)
    state = {"p": start.copy(), "t": 0.0, "w": np.zeros(3), "kind": kind}

    def step(dt):
        p, state["t"] = state["p"], state["t"] + dt
        to = np.array([0.0, 0.0, 1.0]) if kind == "straight" else (mid if kind == "curve" and np.linalg.norm((p - goal)[[0, 2]]) > np.linalg.norm((mid - goal)[[0, 2]]) + 0.3 else goal) - p
        d = to / max(np.linalg.norm(to), 1e-6)
        if kind == "weave":
            side = np.array([d[2], 0, -d[0]])
            d = d + side * amp * math.sin(rate * 2 * math.pi * state["t"])
        elif kind == "wander":
            state["w"] += (-state["w"] / 2.0) * dt + rng.normal(0, 0.6, 3) * math.sqrt(dt) * np.array([1, 0.2, 1])
            d = d + state["w"]
        d /= max(np.linalg.norm(d), 1e-6)
        state["p"] = p + d * speed * dt
        return state["p"]
    return kind, step


def episode(rng, record=False):
    CLOCK[0] += 1000.0
    bus = TelemetryBus()
    pol = RoomAdapter() if args.policy == "defend_room" else POLICIES[args.policy]()
    pol.bus = bus
    defenders = [FakeDrone(n, (0, 0, 0)) for n in (6, 7, 8)]
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        pol.setup(defenders)
    if args.cam:
        mod = sys.modules[type(pol).__module__]
        if hasattr(mod, "make_eyes"):
            mod.make_eyes = lambda drones: {d.num: FakeEye(d, rng) for d in drones}
    floor_y, goal = pol.floor_y, np.array(pol.goal_map if hasattr(pol, "goal_map") else pol.goal, float)
    for i, d in enumerate(defenders):  # take off from the start spots the policy printed
        d.floor_y = floor_y
        start = pol.p[d.num] if np.ndim(pol.p[d.num]) else None
        xz = (start[1], start[0]) if hasattr(pol, "to_g") else (start[0], start[2])  # our guard keeps a guard frame
        d.p = np.array([xz[0], floor_y + 0.9, xz[1]])
        if args.messy:
            d.mess(rng)
            d.p += np.array([rng.normal(0, 0.15), 0.0, rng.normal(0, 0.15)])
    atts = []
    for k in range(2):
        s = np.array([rng.uniform(-0.2, 1.8) if args.scenario == "line" else rng.uniform(-0.3, 2.5),
                      floor_y + rng.uniform(0.9, 1.6), rng.uniform(0.4, 1.4) if args.scenario == "goal" else rng.uniform(1.0, 2.0)])
        kind, f = attacker_path(rng, s, np.array([goal[0], s[1], goal[2]]), rng.uniform(*SPEED))
        atts.append({"num": 4 + k, "p": s, "step": f, "kind": kind, "alive": True, "delay": rng.uniform(0, 3)})
    dt, t = 1.0 / HZ, 0.0
    trace = {"def": {d.num: [] for d in defenders}, "att": {a["num"]: [] for a in atts}, "events": []}
    result, min_gap = "held", 9.9
    while t < args.seconds:
        t += dt
        CLOCK[0] += dt
        for a in atts:
            if a["alive"] and t > a["delay"]:
                a["p"] = a["step"](dt)
            noisy = a["p"] + rng.normal(0, NOISE, 3)
            bus.publish(f"formation/drone{a['num']}", {"state": "fly", "p": [float(x) for x in noisy], "psi_deg": 0.0,
                                                         "fix_age": 0.05, "height_cm": round((a["p"][1] - floor_y) * 100) if a["alive"] else 0,
                                                         "rc": [0, 0, 0, 0]})
        for d in defenders:
            rc = pol.step(d, t)
            assert rc is not None and len(rc) == 4 and all(isinstance(x, int) and -100 <= x <= 100 for x in rc), rc
            d.fly(rc, dt)
        for a in atts:
            if not a["alive"]:
                continue
            for d in defenders:
                gap = np.linalg.norm(d.p - a["p"])
                min_gap = min(min_gap, gap)
                if gap < CATCH:
                    a["alive"] = False
                    trace["events"].append(("catch", a["p"].copy(), t))
                    break
            if a["alive"] and (a["p"][2] >= goal[2] if args.scenario == "line" else np.linalg.norm((a["p"] - goal)[[0, 2]]) < GOAL_R):
                result = "breach"
                trace["events"].append(("breach", a["p"].copy(), t))
        if record:
            for d in defenders:
                trace["def"][d.num].append(d.p.copy())
            for a in atts:
                trace["att"][a["num"]].append(a["p"].copy())
        if result == "breach" or not any(a["alive"] for a in atts):
            break
    caught = sum(not a["alive"] for a in atts)
    return result, caught, t, min_gap, [a["kind"] for a in atts], trace, pol, goal



res = [episode(np.random.default_rng([args.seed, i]), record=i < 6) for i in range(args.episodes)]
br = sum(r[0] == "breach" for r in res)
print(f"{args.policy} on {Path(args.swarm_ops).parent / 'config' / 'map.yaml'} | {args.episodes} episodes, "
      f"true stick gain {args.true_gain}")
print(f"  breach {br}/{args.episodes} = {100 * br / args.episodes:.1f}%   both attackers caught "
      f"{sum(r[1] == 2 for r in res)}   mean episode {np.mean([r[2] for r in res]):.1f} s")
for kind in ("curve", "weave", "wander"):
    sub = [r for r in res if kind in r[4]]
    print(f"  episodes with a {kind:6s} attacker: {len(sub):3d}, breach {sum(r[0] == 'breach' for r in sub)}")
if args.svg:
    import json
    keep = [{"result": r[0], "caught": r[1], "t": round(r[2], 1), "kinds": r[4], "goal": [float(r[7][0]), float(r[7][2])],
             "def": {k: [[round(float(p[0]), 2), round(float(p[2]), 2)] for p in v] for k, v in r[5]["def"].items()},
             "att": {k: [[round(float(p[0]), 2), round(float(p[2]), 2)] for p in v] for k, v in r[5]["att"].items()},
             "events": [[e[0], round(float(e[1][0]), 2), round(float(e[1][2]), 2)] for e in r[5]["events"]]} for r in res[:6]]
    Path(args.svg).write_text(json.dumps(keep))
