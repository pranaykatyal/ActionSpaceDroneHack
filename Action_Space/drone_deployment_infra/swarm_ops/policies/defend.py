"""defend: Team 1's defenders. Intercept the attackers before they reach the goal tag (6).

  python swarm_policies.py --run defend:6,7                           # dry-run
  python swarm_policies.py --run defend:6,7 --fly --duration 120
  python swarm_policies.py --run defend:6,7 --run formation:4,5 --fly # defend against an attack

Each tick it reads the fleet telemetry drone_gui publishes (UDP JSON, drone_telemetry/README.md:
every drone's team and tag pose), puts all drones in one room frame with config/map.yaml, and
runs our intercept guard (windows/guard_runner.py, the controller our simulation policies are
built on): one defender per attacker, the spare one as goalie, lead intercept onto the
attacker's path, separation between defenders, heading held facing the tag-10 wall. A
defender with no fresh pose, or no telemetry at all, holds its height in place.

Telemetry is matched to drones by WiFi name (TELLO-7), so drone_gui's names must be the SSIDs.
Settings come from the environment, since --run takes none:
  DEFEND_PORT=9100                                 telemetry port to listen on
  DEFEND_ARGS="--tag10-height 1.35 --max-speed 0.6"   any guard_runner.py option

Needs numpy, pyyaml and torch (CPU is enough): pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
import json
import os
import shlex
import socket
import threading
import time
from pathlib import Path

from .base import Policy, hold_height

REPO = Path(__file__).resolve().parents[4]
SETTLE_S = 2.0  # after takeoff, just hold height for this long


class DefendPolicy(Policy):
    name = "defend"

    def setup(self, drones):
        super().setup(drones)
        import importlib.util
        import sys

        path = REPO / "windows" / "guard_runner.py"
        spec = importlib.util.spec_from_file_location("guard_runner", path)
        gr = importlib.util.module_from_spec(spec)
        sys.modules["guard_runner"] = gr
        spec.loader.exec_module(gr)  # needs torch for the guard
        names = [d.ssid for d in self.drones]
        self.args = gr.make_parser().parse_args(["--defenders", *names, *shlex.split(os.environ.get("DEFEND_ARGS", ""))])
        self.pilot = gr.GuardPilot(self.args, gr.load_map(self.args.map))
        self.hold = {}
        self.sticks, self.sticks_t = {}, 0.0
        self.msg, self.lock = None, threading.Lock()
        port = int(os.environ.get("DEFEND_PORT", self.args.port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.5)
        threading.Thread(target=self._listen, daemon=True).start()
        print(f"[defend] {names}: telemetry on :{port}, goal tag {self.args.goal_tag} at "
              f"{self.pilot.goal.round(2)} (room frame)", flush=True)

    def _listen(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
                msg = json.loads(data)
            except (socket.timeout, ValueError):
                continue
            except OSError:
                return
            if msg.get("v") != 1:
                continue
            with self.lock:
                self.pilot.ingest(msg)
                self.msg = msg

    def step(self, d, t):
        if d.num not in self.hold:
            self.hold[d.num] = d.height_cm()  # the takeoff height, held whenever we cannot steer
        hover = (0, 0, hold_height(d, self.hold[d.num]), 0)
        now = time.time()
        if t < SETTLE_S:
            return hover
        with self.lock:
            msg = self.msg
            if msg is None or now - msg["t"] > 1.0:
                return hover
            if now - self.sticks_t > 0.03:  # once per tick for all defenders
                self.sticks, self.sticks_t = self.pilot.step(msg, now), now
            rc = self.sticks.get(d.ssid)
        return rc if rc is not None else hover

    def describe(self, d):
        track = getattr(self, "pilot", None) and self.pilot.tracks.get(d.ssid)
        if not track or track.pos is None:
            return "defend (no pose)"
        x, y, z = track.pos
        return f"defend @{x:.1f},{y:.1f},{z:.1f}"
