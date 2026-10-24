"""Closed-loop check of defend_rl.py (and defend.py) in the organizers' swarm_ops, no drones needed.

  python check_defend_rl.py <path to drone_deployment_infra/swarm_ops> [--episodes 40]

Run it after installing (defend.py, defend_rl.py, defend_net.py and weights/ in swarm_ops/policies/).
A simulated bus and drones stand in for the runner: two attackers fly straight at the goal tag
from up to 6 m at 0.4-0.9 m/s (inside the tags' box), defenders execute the rc exactly as
defend.py's dead reckoning assumes (no drift, no noise, no delay), 13.5 Hz, 30 s episodes. A defender within
0.35 m captures an attacker (it then reports height 0, i.e. down); an attacker within 1.5 m of the goal
(horizontally) breaches. It checks frames, signs and roles, not real-world skill: drift, fix noise and
lag are absent, so real numbers will be worse.
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("swarm_ops")
parser.add_argument("--episodes", type=int, default=40)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
sys.path.insert(0, os.path.abspath(args.swarm_ops))

from policies import defend, defend_rl, telemetry  # noqa: E402
from policies.telemetry import TelemetryBus  # noqa: E402


class Clock:
    t = 1000.0

    def time(self):
        return self.t


CLOCK = Clock()
for mod in (defend, defend_rl, telemetry):
    mod.time = CLOCK  # the policies and the bus read time.time()

DT, EPISODE_S, CAPTURE, BREACH = 0.074, 30.0, 0.35, 1.5


class FakeDrone:
    def __init__(self, num):
        self.num, self.name, self.h = num, f"drone{num}", 100.0

    def height_cm(self):
        return self.h


def episode(make, weights, rng):
    if weights:
        os.environ["DEFEND_RL_WEIGHTS"] = weights
    pol = make()
    pol.bus = TelemetryBus()
    drones = [FakeDrone(n) for n in (6, 7, 8)]
    with contextlib.redirect_stdout(io.StringIO()):
        pol.setup(drones)
    goal, floor = pol.goal.copy(), pol.floor_y
    lo, hi = pol.lo, pol.hi
    truth = {d.num: pol.p[d.num].copy() for d in drones}
    vel = {d.num: np.zeros(3) for d in drones}
    for d in drones:
        truth[d.num][1] = floor + 1.0
        d.h = 100.0
    atts = []
    for _ in range(2):
        for _ in range(200):
            ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(3.0, 6.0)
            p = goal + np.array([r * np.cos(ang), 0.0, r * np.sin(ang)])
            if np.all(p[[0, 2]] > lo[[0, 2]]) and np.all(p[[0, 2]] < hi[[0, 2]]):
                break
        p[1] = floor + rng.uniform(1.0, 2.0)
        atts.append({"p": p, "speed": rng.uniform(0.4, 0.9), "down": False})
    t0 = CLOCK.t
    for step in range(int(EPISODE_S / DT)):
        CLOCK.t = t0 + step * DT
        for k, a in enumerate(atts):
            if not a["down"]:
                to = goal - a["p"]
                to[1] = 0.0
                a["p"] = a["p"] + a["speed"] * DT * to / max(np.linalg.norm(to), 1e-6)
            pol.bus.publish(f"formation/drone{k + 4}", {"p": a["p"].tolist(), "fix_age": 0.0,
                                                        "height_cm": 0 if a["down"] else 120}, CLOCK.t)
        for d in drones:
            rc = pol.step(d, step * DT)
            cmd = np.array([rc[0] * pol.h_gain, rc[2] * defend.UP_GAIN, -rc[1] * pol.h_gain])
            vel[d.num] += (cmd - vel[d.num]) * min(DT / defend.TAU, 1.0)
            truth[d.num] = truth[d.num] + vel[d.num] * DT
            d.h = (truth[d.num][1] - floor) * 100.0
        for a in atts:
            if a["down"]:
                continue
            if min(np.linalg.norm(truth[d.num] - a["p"]) for d in drones) < CAPTURE:
                a["down"] = True
            elif np.linalg.norm((a["p"] - goal)[[0, 2]]) < BREACH:
                return sum(x["down"] for x in atts), True
        if all(a["down"] for a in atts):
            break
    return sum(a["down"] for a in atts), False


here = os.path.dirname(os.path.abspath(defend_rl.__file__))
runs = [("defend.py (scripted)", defend.DefendPolicy, None)]
for f in sorted(os.listdir(os.path.join(here, "weights"))):
    if f.endswith(".npz"):
        runs.append((f"defend_rl {f}", defend_rl.DefendRLPolicy, os.path.join(here, "weights", f)))
for name, make, w in runs:
    rng = np.random.default_rng(args.seed)
    res = [episode(make, w, rng) for _ in range(args.episodes)]
    caught = sum(c for c, _ in res)
    breach = sum(b for _, b in res)
    print(f"{name:<40} captured {caught}/{2 * args.episodes} attackers, breach {breach}/{args.episodes} episodes")
