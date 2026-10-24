# Team 1 defense: 3v2 policies and how to put them on the drones

Trained networks for **3 defenders vs 2 attackers**, their simulation results, and what you need
to fly them on the organizers' `swarm_ops` runner. Everything is also in
[`../3v2_policies.zip`](../3v2_policies.zip).

Integrating on the real Tellos (gains, heights, start spots, drift, anything hardware) is the
integrator's job. This folder gives you the networks, the exact input/output contract, a reference
plugin that plugs them into `defend.py`'s runner code, and an offline check. None of it has flown yet.

## What's here

| Path | What |
|---|---|
| `weights/normal_contact_es_g30.pt` | **Flight pick.** skrl IPPO checkpoint, 3 networks (`defender_0..2`). Full-speed contact, no slow-down near the attacker. |
| `weights/soft_contact_es_g10.pt` | Best simulation score. Same structure, but trained to slow to 0.15 m/s at contact (see below). |
| `weights/2v2_soft_guard_es_g20.pt` | The 2v2 submission (2 networks), in case you test with 2 defenders. |
| `weights/*.npz` | The same networks as plain numpy arrays (made by `export_npz.py`), so the laptop needs no torch. |
| `weights/skrl_ippo_cfg.yaml` | skrl agent config the `.pt` files were trained with (load them with it). |
| `defend_net.py` | numpy forward pass + observation builder + map↔sim frame conversion (the contract below, in code). |
| `defend_rl.py` | Reference plugin for `swarm_ops/policies/`: `defend.py`'s state handling and safety, our networks as the tactics. |
| `defend.py` | Pranay's scripted defense plugin (copy of `deliverable/defend.py`); `defend_rl.py` builds on it. |
| `check_defend_rl.py` | Offline plumbing check of the plugin in the organizers' runner (fake bus and drones). |
| `export_npz.py` | `.pt` → `.npz`, with a torch-vs-numpy parity check. |
| `results/` | The hosts' `evaluate.py` logs behind every number below (Isaac warnings filtered out). |
| `sim/` | 3v2 task configs and gym registration, to re-score in Isaac Lab. |

The `.pt` files are the original skrl IPPO checkpoints, byte-identical to the training runs'.
The hosts' `scripts/evaluate.py --checkpoint` loads them directly.

## Which network to fly

- **`normal_contact_es_g30`** flies straight into the attacker at full speed. The organizers count a real
  capture only when the attacker falls out of the air, and a soft 0.15 m/s bump probably won't
  knock a Tello down. This is the one to fly. In sim it breaches 6.4% of episodes.
- **`soft_contact_es_g10`** scores best in sim (1.2–2.2% breach) because the sim counts any approach
  within 0.35 m as a capture, and in this sim a fast hit flips the defender for good. Keep it as a
  comparison, not as the flight network.
- Normal-contact ES generation 40 was still being scored when this was packaged. If it beats g30,
  it will replace it here.

## Results (simulation)

All numbers are from the **hosts' unmodified `scripts/evaluate.py`** on our desktop. The team repo's
`evaluate.py` is modified, so re-score with the hosts' copy to compare. Task `Team1-Defend-3v2-v0`
is the official AS-Defend-v0 (geometry, attackers at 6 m and 0.4–0.9 m/s, 0.35 m capture, 20 s
episodes) with three defenders instead of two. "Interception" counts captures per attacker spawned;
attackers respawn in waves during an episode, so it goes above 100%.

| network | episodes (envs × eps) | interception | **breach** |
|---|---|---|---|
| hosts' `chase` baseline | 64 × 640 | 69.6% | 99.8% |
| normal contact, distilled (no ES) | 100 × 1000 | 257.2% | 25.5% |
| normal contact, ES gen 10 | 100 × 1000 | 312.7% | 11.2% |
| normal contact, ES gen 20 | 100 × 1000 | 333.6% | 7.3% |
| **normal contact, ES gen 30 (flight pick)** | 100 × 1000 | 342.9% | **6.4%** |
| soft contact, distilled (no ES) | 100 × 1000 | 364.1% | 5.5% |
| **soft contact, ES gen 10** | 100 × 1000 | 379.6% | **1.2%** |
| soft contact, ES gen 10 | 128 × 1000 | 378.1% | 2.2% |
| soft contact, ES gen 10 | 200 × 1000 | 381.1% | 1.5% |
| soft contact, ES gen 15 | 100 × 1000 | 379.9% | 1.8% |

The `*_64x20.log` files in `results/eval_3v2/` are `evaluate.py`'s defaults. They count only the first 20 finished
episodes, which are mostly the early breaches, so don't quote them.

Harder attackers, `Team1-Defend-3v2-Realistic-v0` (70% weave, heading wobble, 0.2–1.0 m/s), 64 × 640:
`chase` 92.2% breach, no defense 98.6%, normal contact gen 15 29.7%. Weaving attackers are the
main weakness of the normal-contact network. Gen 30/40 and the soft network on this task were still scoring.

Crashes (`results/crash_3v2/`, ~256 episodes, time defenders spend on the floor): soft ES gen 15
1.3%, soft distilled 2.7%, normal distilled 32.7%, `chase` 6.1%. In this sim a hard hit tips a
defender over and it can't recover (the hosts' controller applies thrust in the body frame), so
sim crash rates overstate what full-speed contact costs on a real Tello.

**Sim vs real:** the sim counts a capture at 0.35 m, centre to centre, and a captured attacker
hovers in place. Real capture is the attacker falling out of the air. The sim also assumes perfect
positions at 60 Hz with no delay. On the real drones you get tag-based fixes, about 13.5 Hz rc and
~0.1–0.2 s of latency. Sim tests with these effects (2v2 network): 13.5 Hz rc costs about 1 point of
breach; delay plus positions-only sensing costs much more. Treat the table as an upper bound.

## How it was trained

1. **Teacher** (scripted, `distill.py --teacher assigned`): each defender sees only its own 23
   inputs, so the roles are fixed by index. Defender 0 takes attacker slot 0 and defender 1 takes
   slot 1, flying at the lead intercept point. Defender 2 is a **goalkeeper**: it holds a post
   1.6 m from the site on the most urgent attacker's line and engages only when that attacker is
   0.8 s from breaching. Once its own attacker is down, a defender guards: it waits 2.2 m out on the
   live attacker's line and engages at 1.5 s. Flags: `--keeper 1.6 0.8 --guard 2.2 1.5
   --contact_offset 0` plus `--soft_closing 0.15` for the soft network only.
2. **Distillation** (DAgger, 12 rounds) into the hosts' skrl IPPO network, one MLP per defender.
3. **ES fine-tuning** (`es_train.py`, pop 120, sigma 0.003, lr 0.0005; fitness = breach,
   captures, defender floor time) directly on `evaluate.py`'s outcome counters.

The full sim pipeline (`distill.py`, `es_train.py`, task configs, write-up) is on branch `shibani`.

## Integrating with the organizers' `swarm_ops`

`defend_rl.py` is a drop-in policy for the organizers' current `drone_deployment_infra/swarm_ops`,
the version with the telemetry bus (`policies/telemetry.py`). It subclasses Pranay's
`deliverable/defend.py` (`DefendPolicy`, included here as `defend.py`) and replaces only the
tactics. Attackers still come from the bus, defenders are still dead-reckoned from start spots, and
the separation, geofence, speed caps and rc mapping are unchanged. Read defend.py's README first
(`deliverable/README.md` on the team repo, branch `main`); everything there applies.

### Install

Copy into `drone_deployment_infra/swarm_ops/policies/`:
`defend.py` (Pranay's, included here), `defend_rl.py`, `defend_net.py`, and the `weights/` folder
(only the `.npz` files are needed at runtime). Then register it in `policies/__init__.py`:

```python
from .defend import DefendPolicy
from .defend_rl import DefendRLPolicy
POLICIES["defend"] = DefendPolicy
POLICIES["defend_rl"] = DefendRLPolicy
```

### Run

```bash
cd drone_deployment_infra/swarm_ops
python swarm_policies.py --run defend_rl:6,7,8 --run formation:4,5            # dry run
python swarm_policies.py --run defend_rl:6,7,8 --run formation:4,5 --fly
DEFEND_RL_WEIGHTS=weights/soft_contact_es_g10.npz python swarm_policies.py ...   # other network
```

Environment variables: `DEFEND_RL_WEIGHTS` (default `weights/normal_contact_es_g30.npz`, relative
to `defend_rl.py` or absolute), plus defend.py's `DEFEND_START`, `DEFEND_HGAIN`, `DEFEND_SPEED`,
`DEFEND_GOAL_TAG`.

### Roles follow `--run` order

The networks are not interchangeable. In `--run defend_rl:6,7,8`, drone 6 runs `defender_0`
(attacker slot 0), drone 7 runs `defender_1` (slot 1) and drone 8 runs `defender_2` (goalkeeper).
A 4th defender would reuse the goalkeeper network. Each attacker keeps its slot while its bus fixes
are fresh (≤ 0.5 s old and above 25 cm), and an empty slot takes the free attacker nearest the goal.
The networks see two attackers at most.

### Offline check (no drones)

```bash
python check_defend_rl.py <path>/drone_deployment_infra/swarm_ops --episodes 40
```

This runs the scripted `defend.py` and each `.npz` network in a fake bus/drone loop: two attackers fly
straight at the goal tag from 3–6 m inside the tags' box, defenders execute the rc perfectly, at 13.5 Hz.
It checks that the frames, signs, roles and plumbing work end to end. It is **not** a skill score:
the goal (tag 6) sits in a corner of the mapped box, so attackers start much closer than the
sim's 6 m, and the defenders' 0.65 m/s cap is below the attackers' 0.9 m/s. Every policy breaches
often there, including the scripted one.

## The network contract (if you write your own runner)

The code version is `defend_net.py`. Per defender, every tick:

**Frames.** The sim's world frame has x and y horizontal and z up. The protected site is the
origin, and its centre is 1.25 m above the floor (`site_height / 2`). From the organizers' map
frame (x right, y up, z out of tag 10): `sim = (x, -z, y)`. Back: `map = (x_s, z_s, -y_s)`. This is
a proper rotation, and the networks were trained with attackers and defenders at random angles
around the site, so the room's orientation doesn't matter. Use the goal tag's floor point
plus 1.25 m as the site.

**Observation, 23 floats** (the hosts' AS-Defend-v0 layout for 2 attackers: 9 + 7 × 2):

| index | value | notes |
|---|---|---|
| 0–2 | own position − site centre | m |
| 3–5 | own velocity | m/s, world frame |
| 6–8 | projected gravity in the body frame | `(0, 0, -1)` when level; use that |
| 9–14 | attacker slot 0, slot 1: position − own position | clip ±12 m; zeros if the slot is empty/down |
| 15–20 | attacker slot 0, slot 1: velocity − own velocity | clip ±3 m/s; zeros if the slot is empty/down |
| 21–22 | attacker slot 0, slot 1 alive flag | 1 or 0 |

Use the same attacker in the same slot for all three defenders, and keep it there.

**Network.** Normalize with `clip((obs − running_mean) / (sqrt(running_variance) + 1e-8), −5, 5)`.
The scaler is per defender and comes inside the checkpoint. Then run the MLP
23 → 256 → 128 → 64 → 4 with ELU between layers, and fly the mean output clipped to [−1, 1].
`log_std` and the value network are for training only.

**Action, 4 floats in [−1, 1]:** `[vx, vy, vz, yaw_rate]`, a **world-frame** velocity setpoint at
**1.0 m/s per unit** (`max_lin_speed`), and yaw rate at 1.745 rad/s per unit. The observation has
no heading, so the yaw output can be ignored. `defend.py` and `defend_rl.py` keep facing tag 10
so the camera keeps seeing tags. To send rc, convert to the map frame, rotate into the drone's
heading, and scale by the measured gain. Facing tag 10, as in `defend.py`: `a = v_x / HGAIN`
(right +), `b = −v_z / HGAIN` (forward +), `c = v_y / UP_GAIN` (up +), `d = 0`.
Mind the signs: the SDK's `a` is right-positive and `d` is clockwise-positive, while the sim's `vy`
is left-positive and its yaw is counter-clockwise-positive. The hosts' `TelloVelocityController.to_rc`
labels `a` as left. `swarm_formation.py`'s `control()` has flight-proven signs.

**Rate and speed.** Training ran at 60 Hz with 1 m/s per axis and a 0.15 s velocity lag. The runner
sends rc at 13.5 Hz and `defend.py` caps horizontal speed at 0.65 m/s and vertical speed at
0.4 m/s. The cap slows the networks' intercepts; raising `DEFEND_SPEED` toward 1.0 is closer to
training if the battery and room allow.

**Optional latency layer.** `drone_deployment_infra/core/defend_deploy.py` (branch `shibani`)
has `predict_obs`, which moves the observation forward by the command delay plus the fix age, and
`separate`, a teammate-separation filter. In sim, at 6 m spawns with 0.1 s delay and a Tello-fitted
response, the two together cut the 2v2 network's breach rate from 62% to 40%. The 3v2 numbers
above don't use them.

## 23 vs 35 inputs

- **23 inputs (these networks):** the hosts' observation for 2 attackers. Each defender sees itself
  (position, velocity, gravity: 9) and the two attackers (relative position, relative velocity,
  alive flag: 7 each). **It has no information about teammates.** Teamwork is by convention:
  slot pairing and the goalkeeper role are fixed by network index. That kept the hosts' layout and
  their `evaluate.py` unchanged, and lets the same network run with any team size.
- **35 inputs (not trained):** the same 23 plus, for each of the 2 teammates, its relative position
  and relative velocity (2 × 6 = 12). A teacher that used teammate positions did well in testing, but a
  23-input network can't copy it. With 35 inputs, defenders could split attackers by geometry
  instead of by slot and keep clear of each other. This matters most in 3v2 and with close
  spawns. It needs a retrain and every teammate's position at runtime.
- For reference, the team repo's `TelloDefendRoomReal3EnvCfg` (Pranay's room 3v2) uses
  **38**: those 35 plus 3 for the scripted guard's aim point.

## Loading the `.pt` directly

```python
import torch
ck = torch.load("weights/normal_contact_es_g30.pt", map_location="cpu", weights_only=False)
# ck["defender_i"]["policy"]: net_container.{0,2,4}.{weight,bias}, policy_layer.{weight,bias}, log_std_parameter
# ck["defender_i"]["observation_preprocessor"]: running_mean, running_variance, current_count
```

In Isaac Lab, re-score with the hosts' evaluator after registering the task (see `sim/`):

```bash
python -u scripts/evaluate.py --task Team1-Defend-3v2-v0 --checkpoint <weights/...pt> --num_envs 100 --episodes 1000
```

To regenerate an `.npz`: `python export_npz.py weights/<name>.pt weights/<name>.npz`
(needs torch; prints `PARITY OK`).
