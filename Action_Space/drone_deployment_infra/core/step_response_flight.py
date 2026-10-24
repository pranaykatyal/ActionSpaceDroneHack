#!/usr/bin/env python3
"""Step-response flight: measure how a real Tello's speed follows rc steps, for the sim's motion model.

The defend sim's Tello command path (team1_defend_fitted_env.py) still runs on the hosts' PRIORS in
simulation_infra/source/.../control/tello_motion_fit.yaml (tau 0.5 s, gains, delay). This flies one
drone through short rc steps and logs every tick in the format the hosts' fitter reads, so the priors
can be replaced by measured values:

  1. take off, hold 2 s (rc 0);
  2. for each amplitude and channel (forward, right, up): +A for STEP_S, 0 for STEP_S, -A, 0;
  3. land.

One rc packet per 74 ms tick (13.5 Hz, the engine's rate). Each tick is a row of
recordings/step_<time>_<name>.train.jsonl:

  {t, rc: [right, forward, up, yaw], mode, fresh, h, vel: {r, f, u, yaw}, state: {raw SDK state}}

vel comes from the SDK state packet. Its units and axes were never verified on these drones
(teammates' logs: vgx/vgy/vgz in dm/s); --vel_map sets the mapping, the raw packet is kept in every
row, and --remap rewrites an existing log with another mapping without flying. Then fit (writes a
NEW yaml; the hosts' tello_motion_fit.yaml is not touched):

  python3 drone_deployment_infra/core/step_response_flight.py                 # dry-run: connect + status, no takeoff
  python3 drone_deployment_infra/core/step_response_flight.py --fly --name TELLO-3
  python3 simulation_infra/scripts/fit_tello_motion.py drone_deployment_infra/recordings/step_<...>.train.jsonl \
      --out simulation_infra/outputs/tello_motion_fit_measured.yaml
  python3 drone_deployment_infra/core/step_response_flight.py --selftest      # no drone: log format -> fitter

The fitter prints an rc-vs-speed correlation matrix: it should be diagonal with positive gains. If
forward/right are swapped or a sign is wrong, --remap the log with a different --vel_map and fit again.

DEFAULT IS DRY-RUN. --fly to fly. Needs ~2 m clear in every horizontal direction and 0.4-2.0 m of height.
Safety: lands on Ctrl+C, any error, no state packet for STATE_STALE_S, battery below LAND_BATTERY, or
height (tof) outside the allowed band. Stdlib only.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DD_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = DD_ROOT / "recordings"
FITTER = DD_ROOT.parent / "simulation_infra" / "scripts" / "fit_tello_motion.py"

# ======================= SETTINGS =======================
DRONE_ADDR = ("192.168.10.1", 8889)
PERIOD_S = 0.074             # one rc packet per tick (13.5 Hz)
HOLD_S = 2.0                 # hover before the first step
STEP_S = 1.5                 # each step and each pause
AMPLITUDES = (10, 25, 40)    # rc units; forward/right (10 sits near the deadband)
UP_MAX = 30                  # rc cap on the up channel (keeps the height in band)
MAX_AMPLITUDE = 60
MIN_BATTERY_FLY = 30
LAND_BATTERY = 15
STATE_STALE_S = 1.0
TOF_MIN_CM, TOF_MAX_CM = 30, 220
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
CHANNEL_INDEX = {"forward": 1, "right": 0, "up": 2}   # rc a b c d = right, forward, up, yaw


def plan(amplitudes):
    """[(mode, rc[4], seconds)] for the whole flight after takeoff."""
    steps = [("hold", [0, 0, 0, 0], HOLD_S)]
    for amp in amplitudes:
        for ch in ("forward", "right", "up"):
            a = min(amp, UP_MAX) if ch == "up" else amp
            for sign in (1, 0, -1, 0):
                rc = [0, 0, 0, 0]
                rc[CHANNEL_INDEX[ch]] = sign * a
                steps.append(("hold", rc, STEP_S))
    return steps


def parse_vel_map(spec):
    """'f=vgx,r=vgy,u=-vgz,scale=0.1' -> {'f': (+1, 'vgx'), ..., 'scale': 0.1}."""
    out = {"scale": 0.1}
    for part in spec.split(","):
        key, val = part.split("=")
        if key == "scale":
            out["scale"] = float(val)
        else:
            sign = -1.0 if val.startswith("-") else 1.0
            out[key] = (sign, val.lstrip("+-"))
    return out


def vel_from_state(state, vmap):
    def get(k):
        sign, field = vmap[k]
        return round(sign * float(state.get(field, 0.0)) * vmap["scale"], 3)

    return {"r": get("r"), "f": get("f"), "u": get("u"), "yaw": float(state.get("yaw", 0.0))}


def make_row(t, rc, mode, state, vmap):
    tof = float(state.get("tof", 0) or 0)
    h = tof / 100.0 if tof > 10 else float(state.get("h", 0) or 0) / 100.0
    return {"t": round(t, 3), "rc": [int(v) for v in rc], "mode": mode, "fresh": False, "h": round(h, 3),
            "vel": vel_from_state(state, vmap), "state": state}


class Tello:
    """One drone over the SDK: commands on 8889, state packets on 8890."""

    def __init__(self, iface=None):
        self.iface = iface
        self.cmd = self._sock(0)
        self.cmd.settimeout(10.0)
        self.state, self.state_t = {}, 0.0
        self._stop = False
        threading.Thread(target=self._state_loop, daemon=True).start()

    def _sock(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.iface:
            s.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, self.iface.encode())
        s.bind(("", port))
        return s

    def _state_loop(self):
        s = self._sock(8890)
        s.settimeout(0.5)
        while not self._stop:
            try:
                data = s.recv(1024).decode(errors="ignore")
            except OSError:
                continue
            fields = {}
            for item in data.strip().strip(";").split(";"):
                if ":" in item:
                    k, v = item.split(":", 1)
                    try:
                        fields[k] = float(v) if k != "mpry" else v
                    except ValueError:
                        fields[k] = v
            self.state, self.state_t = fields, time.time()

    def send(self, text, wait=True):
        self.cmd.sendto(text.encode(), DRONE_ADDR)
        if not wait:
            return None
        try:
            return self.cmd.recv(1024).decode(errors="ignore").strip()
        except socket.timeout:
            return None

    def rc(self, rc):
        self.send("rc %d %d %d %d" % tuple(rc), wait=False)

    def close(self):
        self._stop = True


def fly(args, vmap):
    d = Tello(args.iface)
    if d.send("command") != "ok":
        sys.exit("no 'ok' from the drone: connect the laptop to its Wi-Fi (or pass --iface) and retry")
    time.sleep(1.0)
    bat = d.state.get("bat")
    print(f"connected; battery {bat}%, state packets {'yes' if d.state else 'NO'}", flush=True)
    steps = plan(args.amplitudes)
    total = sum(s for _, _, s in steps)
    print(f"plan: {len(steps)} segments, {total:.0f} s after takeoff; amplitudes {args.amplitudes} (up capped at {UP_MAX})")
    if not args.fly:
        print("DRY-RUN: no takeoff. Add --fly to fly.")
        d.close()
        return None
    if bat is None or bat < MIN_BATTERY_FLY:
        sys.exit(f"battery {bat}% < {MIN_BATTERY_FLY}%: not flying")
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    out = RECORDINGS / f"step_{time.strftime('%Y%m%d_%H%M%S')}_{args.name}.train.jsonl"
    rows = []
    try:
        print("takeoff", d.send("takeoff"), flush=True)
        t0 = time.time()
        for mode, rc, seconds in steps:
            end = time.time() + seconds
            while time.time() < end:
                tick = time.time()
                st = dict(d.state)
                if tick - d.state_t > STATE_STALE_S:
                    raise RuntimeError("state stream lost")
                if st.get("bat", 100) < LAND_BATTERY:
                    raise RuntimeError(f"battery {st.get('bat')}%")
                tof = st.get("tof", 100)
                if tof and not (TOF_MIN_CM <= tof <= TOF_MAX_CM):
                    raise RuntimeError(f"height {tof} cm out of band")
                d.rc(rc)
                rows.append(make_row(tick - t0, rc, mode, st, vmap))
                time.sleep(max(0.0, PERIOD_S - (time.time() - tick)))
        print("steps done", flush=True)
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        d.rc([0, 0, 0, 0])
        print("land", d.send("land"), flush=True)
        d.close()
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"wrote {out} ({len(rows)} rows)")
    return out


def remap(path, vmap):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for r in rows:
        r["vel"] = vel_from_state(r["state"], vmap)
    out = Path(path).with_name(Path(path).name.replace(".train.jsonl", "") + "_remap.train.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out}")


def selftest(vmap):
    """No drone: a synthetic first-order Tello flies the plan, rows go through make_row, the hosts' fitter reads them."""
    import math

    gain, tau, delay_ticks, deadband = 0.012, 0.35, 2, 5   # m/s per rc unit, s, ticks, rc
    steps = plan(AMPLITUDES)
    rows, t, v = [], 0.0, [0.0, 0.0, 0.0]
    hist = []
    for _, rc, seconds in steps:
        for _ in range(int(round(seconds / PERIOD_S))):
            hist.append(rc)
            cmd = hist[-1 - delay_ticks] if len(hist) > delay_ticks else [0, 0, 0, 0]
            for k, idx in enumerate((1, 0, 2)):                 # forward, right, up
                u = 0.0 if abs(cmd[idx]) < deadband else gain * (cmd[idx] - math.copysign(deadband, cmd[idx]))
                v[k] += (u - v[k]) * (1.0 - math.exp(-PERIOD_S / tau))
            # SDK state: dm/s on vgx (forward), vgy (right), vgz (up, negated)
            state = {"vgx": round(v[0] * 10), "vgy": round(v[1] * 10), "vgz": round(-v[2] * 10), "yaw": 0.0,
                     "tof": 100.0, "h": 100.0, "bat": 80.0}
            rows.append(make_row(t, rc, "hold", state, vmap))
            t += PERIOD_S
    tmp = Path(tempfile.mkdtemp())
    log = tmp / "selftest.train.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = subprocess.run([sys.executable, str(FITTER), str(log), "--dry_run", "--out", str(tmp / "fit.yaml")],
                         capture_output=True, text=True)
    print(res.stdout[-2500:])
    if res.returncode != 0:
        print(res.stderr[-1500:])
    ok = res.returncode == 0 and "forward" in res.stdout and "right" in res.stdout and "up" in res.stdout
    print("SELFTEST PASSED (log format accepted by the fitter; check the gains above vs truth "
          f"{gain} m/s per rc, tau {tau} s, delay {delay_ticks * PERIOD_S:.2f} s, deadband {deadband})" if ok else "SELFTEST FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    ap.add_argument("--name", default="tello", help="drone name for the log file, e.g. TELLO-3")
    ap.add_argument("--iface", default=None, help="Linux: bind to this Wi-Fi interface (multi-dongle setups)")
    ap.add_argument("--amplitudes", type=int, nargs="+", default=list(AMPLITUDES), help=f"rc step sizes (max {MAX_AMPLITUDE})")
    ap.add_argument("--vel_map", default="f=vgx,r=vgy,u=-vgz,scale=0.1",
                    help="state packet -> m/s: which field is forward/right/up, sign, and scale (dm/s -> 0.1)")
    ap.add_argument("--remap", default=None, help="rewrite this log's vel with --vel_map (no flight)")
    ap.add_argument("--selftest", action="store_true", help="no drone: check the log format against the hosts' fitter")
    args = ap.parse_args()
    if any(abs(a) > MAX_AMPLITUDE for a in args.amplitudes):
        sys.exit(f"amplitudes must be <= {MAX_AMPLITUDE}")
    vmap = parse_vel_map(args.vel_map)
    if args.selftest:
        selftest(vmap)
    elif args.remap:
        remap(args.remap, vmap)
    else:
        fly(args, vmap)


if __name__ == "__main__":
    main()
