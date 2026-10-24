# Team 1 defense policy: `defend.py`

One file for the organizers' `swarm_ops` runner (`swarm_policies.py`, umbrella). It needs only
numpy and pyyaml, which are already in `swarm_ops/requirements.txt`.

## Install (2 lines)

1. Copy `defend.py` to `drone_deployment_infra/swarm_ops/policies/defend.py`.
2. In `drone_deployment_infra/swarm_ops/policies/__init__.py`, add:

```python
from .defend import DefendPolicy
POLICIES["defend"] = DefendPolicy
```

## Run

```bash
cd drone_deployment_infra/swarm_ops
python swarm_policies.py --run defend:6,7,8 --run formation:4,5          # dry run
python swarm_policies.py --run defend:6,7,8 --run formation:4,5 --fly
```

At launch it prints the goal (tag 6) and a **start spot (x, z) per defender**, in the map
frame. Place each defender there, **facing tag 10**. The policy never yaws.

## What it does

It runs **Team 1's scripted intercept guard**, the controller in our simulation videos
(`simulation_infra/.../tello_defend/guard.py`). It's ported line for line from torch to
numpy, and on random inputs over 200 steps it matches the torch original to 1e-14, with the
same target choices for 3v2, 2v2, 3v5 and 2v3. The settings are those of our 3 v 2 room task:

- **Assignment:** each attacker gets its own defender, chosen to minimise the worst
  defender's distance, with hysteresis so pairs don't swap every tick.
- **Marker:** flies at its attacker's lead-intercept point and rams it.
- **Goalie:** the spare defender holds the crossing point, just outside the goal zone, of
  whichever attacker is closest to the goal, and takes anything that gets past.
- **Hovering attacker:** an attacker that stops heading for the goal is guarded from beside
  its path to the goal.
- **Spacing:** defenders stop closing on each other inside 1 m (fully at 0.3 m) and keep
  clear of drones they aren't catching.
- **Safety:** a geofence around the mapped tags at 0.4–2.0 m height, 0.65 m/s horizontal
  and 0.4 m/s vertical caps, 2 s of height hold after take-off, and position hold when no
  attacker fix is fresh.
- **Attackers** come from the bus, `formation/drone<N>` (`p` in the map frame, `fix_age`).
  Fixes older than 0.5 s are ignored, and a drone below 25 cm counts as down.
- **Our own positions** come from dead reckoning, because in-process policies get no camera
  feed. It starts from the printed start spots and integrates the commanded velocity, with
  height from the drone's range sensor. Drift grows over time, so place the drones carefully.
- **Publishes:** `defend/drone<N>` = `{p, aim, target}`, in the map frame.

## Settings (environment variables, since `--run` takes no options)

| Variable | Default | Meaning |
|---|---|---|
| `DEFEND_START` | spots 2 m from the goal toward tag 10, 0.9 m apart | `"x,z;x,z;x,z"` start spots, in `--run` order |
| `DEFEND_HGAIN` | `0.021` | horizontal m/s per rc unit (we measured 0.021 on TELLO-7) |
| `DEFEND_SPEED` | `0.65` | horizontal speed cap, m/s |
| `DEFEND_GOAL_TAG` | `6` | the protected tag |

## Where the tactics come from

The same tactics as our Isaac Lab guard (`simulation_infra/.../tello_defend/guard.py`,
https://github.com/pranaykatyal/ActionSpaceDroneHack). In the 3 v 2 room simulation, with
sensing noise, 0.2 s lag, falling debris and curved, weaving, flanking and wandering attackers,
the ramming guard breaches **25.7%** of 30 s episodes, against 44.9% for the guard that blocks
beside the attacker's path and about 100% for greedy chase.

A closed-loop check in this infra (simulated bus and drones, 40 episodes, 2 weaving attackers
against 3 defenders, perfect dead reckoning) caught 74 of 80 attackers; 4 episodes had a breach.
