# Hover test with the trained policy (Windows, one Tello)

`hover_direct.py` takes off, climbs to **0.5 m above where the drone started**, holds that
height for **15 s** with our trained policy, then lands. It does not use AprilTags or the room
map. The policy flies height only, from the Tello's downward range sensor. The Tello's own
position hold (IMU plus downward camera) keeps it in place sideways.

The policy is `windows/policies/hover_real_R1.pt`, trained in Isaac Lab on
`AS-Tello-Hover-Real-v0`. That task models the real drone's control lag, slow climb response,
1 cm range steps and sensor noise. The script uses this policy by default.

## 1. One-time setup

You need Python with `torch`, `numpy`, `pyyaml` and `djitellopy`. If you already have the
`isaaclab` conda env, use it. Otherwise make a small env (a CPU build of torch is fine):

```powershell
conda create -y -n hover python=3.11
conda activate hover
pip install torch numpy pyyaml djitellopy opencv-python
```

Get the code:

```powershell
git clone https://github.com/pranaykatyal/ActionSpaceDroneHack.git
cd ActionSpaceDroneHack
```

(or `git pull` in your existing clone).

## 2. Connect to the drone

Turn the Tello on and wait for its LED to blink. Then either:

- pass `--ssid TELLO-7` and the script joins the drone's Wi-Fi on the USB dongle (adapter
  `Wi-Fi 2` by default; use `--iface "<name>"` if yours is named differently. Check the name
  with `netsh wlan show interfaces`), or
- join `TELLO-7` yourself in Windows Wi-Fi settings and leave `--ssid` out.

## 3. Dry run first (no take-off)

```powershell
conda activate hover        # or isaaclab
python windows\hover_direct.py --ssid TELLO-7 --dry
```

This connects, prints the battery and range readings, and shows what the policy would
command. A sane dry run reads roughly `ground 0.10 m -> hold 0.60 m` and shows a positive
"up" command while the drone is still on the ground. **Nothing flies in this mode.**

## 4. Fly

Put the drone on a **textured floor** with **1 m clear all around** and nothing overhead.
The Tello's position hold drifts on plain or glossy floors.

```powershell
python windows\hover_direct.py --ssid TELLO-7
```

- **Ctrl-C lands immediately.** Keep a hand on the keyboard.
- It also lands by itself if the range reading drops out for 1 s, if the drone goes above
  `--ceiling` (1.6 m), or when the hold time is up.
- It refuses to take off below 30% battery (`--min-battery`).

Useful options:

| Option | Default | What |
|---|---|---|
| `--rise` | 0.5 | metres above the starting range reading to hold |
| `--hold` | 15 | seconds to hold before landing |
| `--max-climb` | 0.3 | vertical speed cap, m/s |
| `--ceiling` | 1.6 | land if the range reading goes above this, m |
| `--checkpoint` | `windows/policies/hover_real_R1.pt` | the policy file |

The older policy (trained on `AS-Tello-Hover-v0`) bobbed about ±9 cm on the real drone. To
compare against it, pass `--checkpoint <run>\checkpoints\best_agent.pt --act-scale 1.0 --vel-window 1`.

## 5. Send back the log

Every flight writes `windows\flight_logs\hover_direct_<drone>_<time>.jsonl`. It records every
tick: attitude, velocities, accelerations, range, barometer, battery, the policy's action and
the sticks sent. Commit it or send it over; we use these logs to tune the sim. Good to note
with it: did the drone hold steady, bob up and down, or drift sideways?

## What a good flight looks like

The drone climbs smoothly for about 2 s, settles within a few centimetres of the hold height
without bouncing, stays there for 15 s, then lands. Up-and-down bobbing of about a second per
cycle means the policy is over-correcting. Send the log in that case.
