"""defend: defend_room.py (the trained room-defense policy) on its drones, as its own process.

Like formation: it needs video + tag localization of every drone, so swarm_policies.py
launches it with --drones <its drones> (3 defenders, then 2 attackers), relays its output,
answers its FLY prompt once you have typed FLY, and sends it one SIGINT (= land all) when
the run ends. Its terminal keys don't work under the umbrella: use Ctrl+C there.
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DefendPolicy:
    name = "defend"
    external = True

    def __init__(self, extra=""):
        self.extra = shlex.split(extra)

    def command(self, nums, fly, config):
        return ([sys.executable, "-u", os.path.join(HERE, "defend_room.py"),
                 "--drones", ",".join(str(n) for n in nums), "--config", str(config)]
                + (["--fly"] if fly else []) + self.extra)
