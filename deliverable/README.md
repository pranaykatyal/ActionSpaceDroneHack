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

- **Attackers:** read every tick from the bus, `formation/drone<N>` (`p` in the map frame,
  `fix_age`). Fixes older than 0.5 s are ignored, and a drone below 25 cm counts as down.
- **Roles, recomputed every tick:**
  - **Marker:** one per attacker, the cheapest pairing by distance, attackers nearest the
    goal first. It flies at the attacker's predicted position (lead intercept) to make contact.
  - **Goalie:** each spare defender. It holds the line between the goal and the most
    threatening attacker, 1.35 m out from the goal. It goes straight at any attacker that
    gets inside 1.95 m.
- **Safety:**
  - no closing on a teammate inside 0.6 m;
  - a geofence around the mapped tags, at 0.4–2.0 m above the floor;
  - speed capped at 0.65 m/s horizontal and 0.4 m/s vertical;
  - 2 s of height hold after take-off;
  - with no fresh attacker data, it holds position.
- **Our own positions:** dead reckoning, because in-process policies get no camera feed.
  It starts from the printed start spots and integrates the commanded velocity. Height comes
  from the drone's range sensor. Drift grows over time, so place the drones carefully.
- **Any team size:** works for 2 v 3, 3 v 5 and so on, since roles re-pair every tick.
- **Publishes:** `defend/drone<N>` = `{p, role, target}`.

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
against 3 defenders) caught 80 of 80 attackers. That check models no drift, so it is optimistic.
