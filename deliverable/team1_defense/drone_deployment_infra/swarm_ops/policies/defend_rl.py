"""defend_rl: Team 1's trained 3v2 defender networks on the organizers' umbrella runner.

  python swarm_policies.py --run defend_rl:6,7,8 --run formation:4,5           # dry-run
  python swarm_policies.py --run defend_rl:6,7,8 --run formation:4,5 --fly

Everything except the tactics is defend.py's (DefendPolicy): attackers from the bus
("formation/drone<N>", map frame), dead-reckoned defenders from the printed START spots, 2 s
height hold after takeoff, teammate separation, geofence, speed caps, heading fixed facing tag 10,
rc mapping. The tactics are the networks: every tick each defender's 23 observations
(defend_net.py) go through its own network, and the velocity it outputs is flown.

Roles are fixed by --run order, as in training: the 1st drone is defender_0 (takes attacker
slot 0), the 2nd defender_1 (slot 1), the 3rd defender_2 (the goalkeeper). A 4th+ drone reuses the
last network. Attacker slots are sticky: an attacker keeps its slot while its fixes are fresh; an
empty slot takes the free attacker nearest the goal. The networks see 2 attackers at most.

Environment (besides defend.py's DEFEND_START / DEFEND_HGAIN / DEFEND_SPEED / DEFEND_GOAL_TAG):
  DEFEND_RL_WEIGHTS=weights/normal_contact_es_g40.npz   (relative to this file, or absolute)
Needs numpy and pyyaml.
"""
import os
import time
from pathlib import Path

import numpy as np

from .defend_basic import GAIN, DefendPolicy
from .defend_net import MAX_LIN_SPEED, SITE_HEIGHT, DefendNets, build_obs, map_to_sim, sim_to_map

DEFAULT_WEIGHTS = "weights/normal_contact_es_g40.npz"
ROLE = ["slot0", "slot1", "keeper"]


class DefendRLPolicy(DefendPolicy):
    name = "defend_rl"

    def setup(self, drones):
        super().setup(drones)
        w = Path(os.environ.get("DEFEND_RL_WEIGHTS", DEFAULT_WEIGHTS))
        if not w.is_absolute():
            w = Path(__file__).resolve().parent / w
        self.nets = DefendNets(w)
        self.site = self.goal.copy()
        self.site[1] = self.floor_y + SITE_HEIGHT / 2  # the sim's site centre: 1.25 m above the floor
        self.slots = [None, None]
        self.vnet = {}
        print(f"[defend_rl] {w.name}: {self.nets.count} networks; roles by --run order: "
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

    def _dead_reckon(self, d, now):
        super()._dead_reckon(d, now)
        role = self.roles.get(d.num)
        if role is not None and d.num in self.vnet:  # keep the aim one GAIN-step ahead of the moved drone
            self.roles[d.num] = (role[0], role[1], self.p[d.num] + self.vnet[d.num] / GAIN)
