"""File-level spoof of Mo-style defend.py (5 drones: 3 defenders + 2 scripted attackers, own cameras).

  python test_mo_file.py <defend.py> <checkpoint.pt> [--speed lo,hi] [--messy] [--episodes N]

The policy file runs as written (setup / step / its own catch + breach rules and fallback); only these
are faked: swarm_formation (map: our TELLO-3 map.yaml), Eye (camera fix = true pose 0.1 s old + 3 cm
noise, 10% of frames lost), the drones (rc -> velocity, 0.021 m/s per rc unit, 0.35 s lag, 0.2 s
delay; --messy adds per-drone gain x U(0.7,1.3), heading error U(-8,8) deg, drift, placement error).
Attackers are flown by this harness straight out from the tag-10 wall at a per-episode speed (the
policy's own attacker commands only decide when one lands), so both files face identical attackers.
"""
import argparse
import importlib.util
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("policy")
ap.add_argument("checkpoint")
ap.add_argument("--speed", default="0.4,0.9")
ap.add_argument("--messy", action="store_true")
ap.add_argument("--episodes", type=int, default=200)
args = ap.parse_args()
LO, HI = (float(v) for v in args.speed.split(","))

CLOCK = [1_000_000.0]
time.time = lambda: CLOCK[0]
HERE = Path(__file__).parent
ROOT = HERE / "spoof/drone_deployment_infra/swarm_ops"
MAP = HERE / "spoof/drone_deployment_infra/config/map.yaml"
os.environ["DEFEND_CHECKPOINT"] = str(Path(args.checkpoint).resolve())

# load the policy file as policies.<name> so its "from .base import Policy" works
sys.path.insert(0, str(ROOT))
import policies  # noqa: E402,F401
spec = importlib.util.spec_from_file_location("policies.under_test", args.policy)
mod = importlib.util.module_from_spec(spec)
sys.modules["policies.under_test"] = mod
spec.loader.exec_module(mod)

m = yaml.safe_load(MAP.read_text())["tags"]
FLOOR_Y = float(np.median([v["t"][1] for v in m.values() if abs(np.array(v["R"])[1, 2]) > 0.8]))
GOAL = np.array(m[6]["t"], float)                           # map frame
HZ, TAU, DELAY, GAIN, UP = 13.5, 0.35, 0.2, 0.021, 0.006


def internal(p):  # map frame -> swarm_formation internal (x right, y down, z into tag 10)
    return np.array([p[0], -p[1], -p[2]])


sf = types.SimpleNamespace(
    MAP_PATH="map", TAGS_PATH="tags", CAMERA_DIR="",
    load_tag_files=lambda a, b: (None, {k: (np.eye(3), internal(np.array(v["t"], float))) for k, v in m.items()}, {}),
    MapCheck=lambda tag_map: None, load_camera=lambda path: (None, None), camera_file=lambda n, d: "cam")
mod._sf = lambda: sf


class Drone:
    def __init__(self, num, p, rng):
        self.num, self.name, self.p, self.v, self.q, self.hist = num, f"drone{num}", np.array(p, float), np.zeros(3), [], []
        self.down, self.rng = False, rng
        self.gx = self.gz = 1.0
        self.yaw, self.drift = 0.0, np.zeros(3)
        if args.messy:
            self.gx, self.gz = rng.uniform(0.7, 1.3, 2)
            self.yaw = math.radians(rng.uniform(-8, 8))

    def height_cm(self):
        return round((self.p[1] - FLOOR_Y) * 100)

    def fly(self, rc, dt):
        self.q.append(rc)
        rc = self.q.pop(0) if len(self.q) > round(DELAY * HZ) else (0, 0, 0, 0)
        want = np.array([rc[0] * GAIN * self.gx, rc[2] * UP, -rc[1] * GAIN * self.gz])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        want[0], want[2] = c * want[0] - s * want[2], s * want[0] + c * want[2]
        if args.messy:
            self.drift += -self.drift / 3.0 * dt + self.rng.normal(0, 0.05, 3) * math.sqrt(dt / 1.5) * np.array([1, 0.2, 1])
            want = want + self.drift
        self.v += (want - self.v) * min(dt / TAU, 1.0)
        self.p = self.p + self.v * dt

    def log(self):
        self.hist = (self.hist + [(CLOCK[0], self.p.copy())])[-10:]


class Eye:
    """The policy file's Eye, faked: room-frame pose from the true pose 0.1 s ago + noise."""

    def __init__(self, d, *rest):
        self.d, self.P, self.V, self.psi, self.t, self.seen_t, self.fix, self.reason = d, None, np.zeros(3), 0.0, None, 0.0, None, ""
        self.stop = types.SimpleNamespace(set=lambda: None)

    def start(self):
        pass

    def has_fix(self, now):
        return self.fix is not None and now - self.seen_t < 0.5

    def update(self, now):
        old = [h for h in self.d.hist if h[0] <= now - 0.1]
        if not old or self.d.rng.random() < 0.1:
            return
        t, p = old[-1]
        p = p + self.d.rng.normal(0, 0.03, 3)
        P = np.array([p[2], p[0], self.d.height_cm() / 100.0])
        if self.t is not None and t > self.t:
            self.V = 0.4 * np.clip((P - self.P) / (t - self.t), -3, 3) + 0.6 * self.V
        self.P, self.t, self.seen_t, self.fix = P, t, now, (p, 0.0, t)


mod.Eye = Eye


def episode(i):
    rng = np.random.default_rng([7, i])
    CLOCK[0] += 1000.0
    xs = np.linspace(-0.2, 1.8, 3) + rng.normal(0, 0.1, 3)
    defs = [Drone(n, (x, FLOOR_Y + rng.uniform(0.8, 1.0), GOAL[2] - rng.uniform(0.3, 1.0)), rng) for n, x in zip((3, 4, 5), xs)]
    atts = [Drone(n, (rng.uniform(-0.2, 1.8), FLOOR_Y + rng.uniform(0.9, 1.4), rng.uniform(1.0, 2.0)), rng) for n in (6, 7)]
    speed = [rng.uniform(LO, HI) for _ in atts]
    if args.messy:
        for d in defs:
            d.p += np.array([rng.normal(0, 0.15), 0, rng.normal(0, 0.15)])
    pol = mod.DefendPolicy()
    pol.bus = None
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        pol.setup(defs + atts)
    dt, t = 1.0 / HZ, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        while t < 40:
            t += dt
            CLOCK[0] += dt
            for d in defs + atts:
                d.log()
            for d in defs + atts:
                rc = pol.step(d, t)
                if rc is None:
                    d.down = True
                if d in atts:
                    if not d.down and pol.phase == "GO":  # straight out from the tag-10 wall
                        d.p = d.p + np.array([0, 0, speed[atts.index(d)] * dt])
                elif not d.down:
                    d.fly(rc, dt)
            if pol.phase == "END":
                break
    return str(pol.result), t


res = [episode(i) for i in range(args.episodes)]
br = sum(r[0] == "BREACH" for r in res)
print(f"{Path(args.policy).parent.name} + {Path(args.checkpoint).parent.name} | attackers {LO}-{HI} m/s"
      f"{' | messy' if args.messy else ''} | {args.episodes} episodes: breach {100 * br / args.episodes:.1f}%, "
      f"defended {sum(r[0] == 'DEFENDED' for r in res)}, time up {sum(r[0].startswith('TIME') for r in res)}, "
      f"other {sum(r[0] not in ('BREACH', 'DEFENDED') and not r[0].startswith('TIME') for r in res)}")
