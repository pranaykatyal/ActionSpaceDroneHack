"""defend_room: Rohan's trained 35-input room network (defend_room.py), as its own process.

It flies 3 defenders AND 2 scripted attackers of its own (--run defend_room:<3 defenders>,<2 attackers>),
with video + tag localization of all five. Trained for the corridor with tag 6 6.48 m out from
tag 10: with any other map.yaml it refuses to fly. Extra arguments: DEFEND_ROOM_ARGS="...".
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DefendRoomPolicy:
    name = "defend_room"
    external = True

    def __init__(self, extra=None):
        self.extra = shlex.split(os.environ.get("DEFEND_ROOM_ARGS", "") if extra is None else extra)

    def command(self, nums, fly, config):
        return ([sys.executable, "-u", os.path.join(HERE, "defend_room.py"),
                 "--drones", ",".join(str(n) for n in nums), "--config", str(config)]
                + (["--fly"] if fly else []) + self.extra)
