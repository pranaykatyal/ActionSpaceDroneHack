"""defend_cam: Team 1's guard with camera localization (defend_live.py), as its own process.

Like formation: it needs video + tag localization of its drones, so swarm_policies.py launches
defend_live.py with --drones <its drones>, relays its output, answers its FLY prompt and sends
it one SIGINT (= land all) when the run ends. Attackers come from the bus (formation/drone<N>).
Extra arguments: DEFEND_CAM_ARGS="--speed 0.65 --max-rc 45".
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DefendCamPolicy:
    name = "defend_cam"
    external = True

    def __init__(self, extra=None):
        self.extra = shlex.split(os.environ.get("DEFEND_CAM_ARGS", "") if extra is None else extra)

    def command(self, nums, fly, config):
        return ([sys.executable, "-u", os.path.join(HERE, "defend_live.py"),
                 "--drones", ",".join(str(n) for n in nums), "--config", str(config)]
                + (["--fly"] if fly else []) + self.extra)
