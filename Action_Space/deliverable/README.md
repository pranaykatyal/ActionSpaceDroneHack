# Team 1 flight policies: `defend.py` for our trained 3v2 networks

Two drop-in replacements for `drone_deployment_infra/swarm_ops/policies/defend.py`. They keep the same class
(`DefendPolicy`), the same `defend` entry in `POLICIES` and the same command. Each runs one of our trained
3-defender networks on the real Tellos.

| Zip | Network | Hosts' `evaluate.py`, 3v2 official | Closest-to-flight sim variant |
|---|---|---|---|
| **`defend_policy_soft.zip`** | soft contact, ES gen 10 (`policies/team1_defend_3v2_soft/agent_10.pt`) | 1.2% breach (100 × 1,000) | **39.8% breach** |
| `defend_policy_normal.zip` | normal contact, ES gen 40 (`policies/team1_defend_3v2_normal/agent_40.pt`) | 4.9% breach | 70.9% breach |

On the same variant, the hosts' chase baseline breaches 91.6%. That variant is `Team1-Defend-3v2-FittedT024-Spawn6-PredictCVSep-v0`:
- the hosts' Tello motion priors;
- position fixes only, filtered;
- 6 m spawns;
- the laptop's delay compensation and separation.

## Use

```bash
unzip defend_policy_soft.zip -d drone_deployment_infra/swarm_ops/policies/   # defend.py + checkpoint + map + tag sizes
cd drone_deployment_infra/swarm_ops
python swarm_policies.py --run defend:3,4,5,6,7 --duration 45          # dry-run
python swarm_policies.py --run defend:3,4,5,6,7 --duration 45 --fly
```

Drones 3, 4 and 5 defend; drones 6 and 7 are the scripted attackers. Each zip holds:
- `defend.py`;
- `defend_checkpoint.npz`, which is numpy only and used first;
- `defend_checkpoint.pt`, which needs torch;
- `defend_map.yaml`: the user's room map from 2026-10-24 09:56;
- `defend_tags.yaml`: that map's measured tag sizes.

Placement, settings and every rule are documented in the `defend.py` docstring. Before flying:
- **Placement:** the asset is tag 6 on the far wall.
  - The two flank defenders go on a 2.5 m ring around it at ±35°.
  - The goalkeeper goes 1.6–2.0 m in front of tag 6.
  - The attackers go 1.0–1.2 m in front of tag 10, one at Y −1.0..−1.2 and one at +1.0..+1.2.
  - Every drone faces tag 10.
- **Clear the floor:** remove chairs and boxes, at least in the attackers' start area.
- **Wrong map:** if setup stops saying tag 6 is not on the far wall, the room has been re-mapped. Set `DEFEND_MAP` and `DEFEND_TAGS` to the current map and tag sizes.

## How it was checked

- **Simulator parity** (`simulation_infra/scripts/defend_parity.py`, 600 steps × 8 envs × 3 defenders in
  `Team1-Defend-3v2-v0`): `build_obs` reproduces the env's 23 inputs exactly (error 0), and `act` matches skrl's
  actions within 2e-6 (soft) and 1.1e-6 (normal).
- **Three review rounds, each finding adversarially verified:**
  - **Round 1** found 15 problems in the first version, including the wrong map, a breach line that couldn't
    fire, a fence too close to the walls, stale inputs, and yaw not facing tag 10.
  - **Rounds 2 and 3** ran the real `DefendPolicy` in closed-loop fake-drone harnesses: Tello lag and delays,
    noisy 15 Hz fixes, dropouts, bias, occlusion, about 7,000 games. They also replayed 12.7 drone-minutes of
    real flight logs through the height logic.
- **Final version, closed loop:** every game ends DEFENDED in nominal, combined-stress and dropout runs, with 0
  exceptions. No defender comes within 0.2 m of a wall. Teammates stay at least 1.09 m apart. The breach rule
  fires when the defenders are disabled.
- **Not covered:** none of this replaces a real flight. The motion model is the hosts' unfitted prior;
  `drone_deployment_infra/core/step_response_flight.py` measures the real one.
