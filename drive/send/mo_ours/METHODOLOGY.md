# Team 1 defense, "Mo + ours": methodology

`defend.py` in this folder flies 3 defenders against 2 attackers in the organisers' room. It
combines two pieces of work:

- **Mo's runner** handles everything around the tactics: a camera and AprilTag localizer for
  each of the 5 drones, the scripted attackers, the episode referee, the geofence, and the
  conversion to rc commands.
- **Team 1's intercept guard** decides where each defender flies. It replaces the trained
  network in Mo's original file.

The defenders' tactics are a **scripted controller, not a neural network**. We trained RL
policies in simulation, but they did not beat the scripted guard (see section 5). The guard's
parameters were tuned in the same simulator, against the same randomized attackers. No
checkpoint and no torch are needed: the file uses only numpy.

---

## 1. What came from where

| Part | Source | Change in this file |
|---|---|---|
| Video, AprilTag localization, pose smoothing (`Eye`) | Mo | unchanged |
| Scripted attackers, capture and breach referee, episode phases | Mo | unchanged |
| Geofence and defender separation push | Mo | unchanged |
| Room-frame velocity to rc sticks (`to_rc`) | Mo | `GAIN` is now 0.021 m/s per rc unit (measured on TELLO-7), was 0.008 |
| Defender tactics | Mo: MAPPO network (`AS-Defend-Room-v0` checkpoint) | **Team 1's `InterceptGuard`**, a numpy port of our Isaac Lab guard |
| Fallback chaser | Mo | off by default; `DEFEND_FALLBACK=1` turns it on |

The stick gain matters a lot. With Mo's 0.008, a commanded 0.3 m/s became about 0.8 m/s on the
real drone (about 2.6x too fast). We measured 0.021 m/s per rc unit on TELLO-7 from our
tag-approach flight logs.

---

## 2. Room frame and roles

- **Origin:** tag 10.
  - **X:** straight out of the tag-10 wall.
  - **Y:** map x.
  - **Z:** height above the floor, from the Tello's ToF sensor.
- **Goal / breach line:** tag 6, read from `config/map.yaml`. The protected *site* is tag 6's
  spot on the floor. An attacker whose X passes tag 6 is a breach.
- **Drone order:** `--run defend:a,b,c,d,e` makes the first 3 drones defenders and the last 2
  attackers.
- **Placement:** every drone faces tag 10. Yaw is held at 0, as in the sim, where yaw was locked.

---

## 3. The control loop, one tick (13.5 Hz, `PERIOD_S = 0.074`)

`DefendPolicy.step()` runs `_tick()` once for each tick of the umbrella runner and hands each
drone its rc command.

1. **Perception (`Eye`, one per drone, in background threads).**
   - Each drone's H.264 stream is decoded with PyAV.
   - `pupil_apriltags` detects tag36h11 tags. Only tags that are in the map, trusted by
     `MapCheck`, have a good decision margin and are big enough on screen are used.
   - `swarm_formation.localize()` solves the camera pose from the tag map and tag sizes.
   - **Outlier gate:** a position jump larger than 0.5 m is ignored for up to 1 s.
   - **Smoothing:** pose is smoothed with α = 0.35.
   - **Velocity:** finite differences, smoothed with α = 0.4.
   - **Height:** Z is replaced by the ToF reading whenever there is one.
2. **WAIT phase.** All drones hover until every drone has a tag fix. They all land if that
   takes more than 20 s. Then the phase becomes **GO**, and each attacker's current height is
   stored as its cruise height.
3. **Attackers and referee.**
   - Each live attacker flies +X at `ATTACK_SPEED_MPS` (0.3 m/s) and holds its height.
   - It is **caught** if any defender is within `CAPTURE_M` (0.45 m).
   - It is a **breach** if its X passes tag 6.
   - It lands if it has had no tags for 3 s.
4. **Guard (`_guard`).** The live attackers' positions and velocities and the defenders'
   positions go into `InterceptGuard.compute()`, which returns one room-frame velocity per
   defender (section 4).
   - The horizontal speed cap is 0.65 m/s and the climb cap is 0.4 m/s.
   - The guard is rebuilt whenever an attacker is caught, because its assignment table
     depends on the number of attackers.
5. **Defender safety layer.**
   - Defenders closer than 0.5 m to each other get a 0.3 m/s push apart.
   - The **geofence** (`_fence`) zeroes any velocity that points out of the room box when the
     drone is within 0.3 m of a wall.
   - A defender with a stale fix (older than 0.5 s) hovers. One without a fix for 3 s lands.
6. **rc conversion (`to_rc`).**
   - The room-frame velocity is rotated into each drone's heading frame and divided by the
     stick gain.
   - A deadband of 5 is added, and the result is clipped to `MAX_RC_DEFENDER`.
   - A P-controller on yaw keeps the drone facing tag 10.
7. **End of episode.** The episode ends when both attackers are caught, on a breach, or after
   25 s. Then every drone lands.

---

## 4. The intercept guard (`InterceptGuard`)

This is a line-for-line numpy port of
`simulation_infra/.../tasks/direct/tello_defend/guard.py`. On random inputs over 200 steps it
matches the torch original to 1e-14, with the same target choices for 3v2, 2v2, 3v5 and 2v3.
Arrays keep the sim's batch dimension (E = 1 here).

1. **Is an attacker attacking?** The real game does not report attacker status, so
   `infer_attacking` decides from motion. An attacker counts as attacking if its horizontal
   velocity toward the site is above 0.15 m/s.
2. **Aim point for each defender–attacker pair (`aim_points`).**
   - **Attacking:** the *lead-intercept* point, which is where the attacker will be when a
     defender flying at `speed` can reach it. It comes from the quadratic
     |rel + v·t| = speed·t, capped at `lead_horizon` = 1.5 s.
     - If that point falls inside the inner ring (`intercept_radius` = 1.35 m around the site)
       while the attacker is still outside it, the aim moves to the attacker's crossing point
       on that ring. We meet the attacker before the goal, not at it.
   - **Not attacking (hovering or retreating):** park on the attacker, clamped to within
     `park_radius` = 3 m of the site.
3. **Assignment.**
   - Candidate assignments: every defender gets a target, and every attacker is covered (with
     3 defenders against 2 attackers, one attacker gets two defenders).
   - Cost of an assignment: the largest defender-to-aim distance + 0.1 × the sum of the
     distances.
   - Hysteresis: the guard switches to a new assignment only if it is 0.4 m better, so pairs
     don't swap every tick.
4. **Roles.**
   - **Marker:** the nearer defender on each attacker flies at its aim point. With
     `dock_gap = 0` and `dock_range = 0`, the marker rams the attacker instead of shadowing it,
     because in this game contact is what removes an attacker.
   - **Goalie (sweeper):** the spare defender holds the point
     `intercept_radius − sweep_depth` (1.15 m) from the site, on the line to whichever
     attacking drone is closest to the site. It takes anything that gets past the markers.
5. **Command.**
   - Base command: `gain` (4.0) × (aim − position).
   - Defenders stop closing on each other inside 1 m, and fully at 0.3 m.
   - Each defender is pushed away (repulsion 5) from any drone within 0.7 m that it is not
     trying to catch.
   - The result is normalized so that no axis exceeds the speed cap.

---

## 5. How we got here: simulation, training, tuning

All of this ran in Isaac Lab on our extended `tello_defend` task (`AS-Defend-*`), with 1024
parallel environments on a laptop GPU and on the organisers' H100 node.

### 5.1 Baselines and a simulator fix

- **Hosts' greedy `chase` baseline:** about **100% breaches** on `AS-Defend-v0`. Both
  defenders go after the same nearest attacker and ignore one that holds and retries.
- **Simulator bug:** the kit's velocity controller produced world-frame forces, but Isaac Lab
  applied them in the body frame. After any contact or yaw, the drones flipped. We added
  `world_frame_force = True` to every task we trained on.

### 5.2 Reinforcement learning (MAPPO, skrl)

- **Setup:**
  - One actor per defender: an MLP with layers [256, 128, 64] and ELU activations.
  - A centralized critic.
  - Rollouts of 16 steps, 5 epochs, 4 mini-batches.
  - γ = 0.99, λ = 0.95, learning rate 5e-4 with a KL-adaptive schedule.
  - Running-standard-scaler preprocessing on observations and values.
- **Observations (per defender):**
  - Its own position relative to the site, its velocity, and gravity.
  - For each attacker: relative position, relative velocity, and a visible flag.
  - Its teammates' relative positions and velocities.
  - The guard's aim point.
- **Action:** the 4 Tello rc channels in [-1, 1].
- **Reward per defender:**

  | Term | Scale |
  |---|---|
  | Closing on its target | +5 |
  | Proximity to the nearest attacker | +2 |
  | Capture | +30 |
  | Clearing the wave | +30 |
  | Breach (shared) | −20 |
  | Effort | −0.01 |
  | Alive | +0.1 |
  | Action rate (realistic tasks only) | −1 |

  We kept the breach penalty comparable to the shaping terms. A huge penalty made training
  drift below a do-nothing policy.
- **Fixes along the way:**
  - The breach penalty was paid on two steps; now it is paid once.
  - Target switches caused jumps in the closing reward; that is fixed.
  - Waves respawn instead of ending the episode, because ending on success truncated
    bootstrapping and the policy unlearned success.
  - `clip_actions: False`: with skrl 2.x, clipping actions corrupted the log-probabilities.
- **Residual RL on the guard** (`defender_base = "guard"`): the action is the guard's command
  plus `residual_scale` × the network output. This was the most promising variant.
  - We ran 2 variants of about 1050 iterations each on the H100 and scored them on 1000
    episodes:

    | Controller | Breaches |
    |---|---|
    | Guard alone | **7.6%** |
    | Residual, scale 0.3 | 11.1–12.3% |
    | Residual, scale 0.6 | 12.9–15.7% |

  - The learned residual made the guard worse. In shorter pilots it either broke the guard
    (50% breaches) or shrank to zero.
- **Conclusion:** we fly the guard. Our teammates' learned policies (Rohan's MAPPO room
  network, Shibani's ES-trained networks) are packaged separately as `mo_roh` and `mo_shib`
  on the same runner.

### 5.3 Making the simulator look like the real game, then tuning the guard

We kept adding realism and re-tuned the guard parameters at each stage, scoring breach rates
over hundreds of seeded episodes:

| Task | What it adds |
|---|---|
| `AS-Defend-Guard-v0` | 15 Hz control (the real rc rate is about 13.4 Hz). Defenders get the slower, laggier response we measured on TELLO-1. Attackers are faster and retry after a capture: they go out, return, hold or teleport. Park-point tuning: 8.8% vs 13.2% breaches (400 episodes). |
| `AS-Defend-Room-v0` | The organisers' room geometry: attackers launch from the tag-10 wall, and the goal is tag 6, 4.21 m out and 3.16 m to the right. Defenders are capped at 0.65 m/s (battery); attackers fly at 0.6–0.8 m/s. Capture means contact: 0.25 m. |
| `AS-Defend-RoomReal-v0` | What our real flights taught us: 5 cm of position noise, 0.1 m/s of velocity noise, 10% stale readings (tag out of view), about 0.2 s command latency, an action-rate penalty. Hit attackers fall with their momentum (landing in the goal counts as a breach) and respawn. Attacker paths are straight, weaving, curved via a random point, or wandering. |
| `AS-Defend-RoomReal3-v0` | 3 defenders vs 2 attackers. This file's `GUARD` settings come from here. |

Two tuning results in `RoomReal3` set this file's guard settings:

- **Ram instead of block beside the path:** 44.9% → 25.7% breached episodes (256 episodes).
  Contact is what removes an attacker.
- **`lead_horizon` 1.5 s and `sweep_depth` 0.2 m** for the organisers' shorter 4.2 m room:
  49.6% → 35.2%.

Breach here means *any* attacker got in during a 30 s episode, against attackers faster than
our defenders that weave, flank and respawn. That is much harder than the real run's straight
0.3 m/s attackers.

### 5.4 Checks outside Isaac

- **numpy vs torch:** the numpy port matches the torch guard to 1e-14.
- **Closed loop on the organisers' runner:** we tested our bus-based `defend.py` with simulated
  drones and perfect dead reckoning, over 40 episodes of 2 weaving attackers against 3
  defenders. It caught 74 of 80 attackers, and 4 episodes had a breach.
- **Mo's fallback chaser:** it did not improve the guard in our spoofed-input tests, so it is
  off by default here.

---

## 6. Data we used

| Data | Used for |
|---|---|
| **Simulation rollouts** (Isaac Lab, 1024 envs, randomized attacker speed, path type, height, spawn lane, retry mode, noise) | All RL training and every guard-tuning comparison. There is no fixed dataset: experience is generated online. |
| **Organisers' room map** (`config/map.yaml`, final 2026-10-24) and **tag sizes** (`config/tags.yaml`, measured black-square sizes) | Room geometry in the sim (`lane_offset`, site at tag 6), localization on the drones, and the breach line at runtime |
| **Per-drone camera calibrations** (`swarm_formation.camera_file`) | Pose from AprilTags |
| **Real flight logs** (`windows/flight_logs/*.jsonl`, `recordings/*.train.jsonl`) | Stick-to-speed gain, latency and response time (`scripts/fit_tello_motion.py` → `control/tello_motion_fit.yaml`), then corrected from TELLO-7 tag-approach flights to **0.021 m/s per rc unit**. Hover flights gave the noise and latency levels used in `RoomReal`. |
| **Measured drone limits** | 0.65 m/s defender speed cap (battery), about 13.4 Hz rc rate, 0.3 s response time |

---

## 7. Settings (top of `defend.py`)

| Setting | Value | Why |
|---|---|---|
| `PERIOD_S` | 0.074 s | Umbrella tick, which is also the sim control step |
| `EPISODE_S` / `READY_TIMEOUT_S` | 25 s / 20 s | Episode length / time allowed to see tags |
| `ATTACK_SPEED_MPS` | 0.3 | Scripted attackers for the demo |
| `CAPTURE_M` | 0.45 m | Catch distance (0.25–0.35 m in the sim) |
| `MAX_RC_DEFENDER` | 100 | Stick limit. The guard's own 0.65 m/s cap is what binds. |
| `MAX_HSPEED_MPS` / `MAX_CLIMB_MPS` | 0.65 / 0.4 m/s | Same caps as the sim |
| `GAIN` | 0.021 / 0.021 / 0.006 | m/s per rc unit: right / forward / up |
| `GUARD` | `RoomReal3` settings | See section 4 |
| `DEFEND_FALLBACK` (env var) | `0` | `1` turns on Mo's single-chaser fallback |

---

## 8. Limitations

- **Localization.** Positions come only from AprilTags. A defender that loses its tags hovers,
  then lands after 3 s. Camera pose noise is the main error source.
- **Inferred attacker status.** An attacker moving slower than 0.15 m/s toward the goal counts
  as not attacking, and the guard parks on it instead of leading it.
- **Speed.** Defenders at 0.65 m/s can only stop attackers that are slower or that start far
  enough away. In the sim, attackers at 0.8–1.4 m/s breached about 92% even against the guard.
  That is a physical limit, not a tactics problem.
- **Sim numbers.** They come from harder, randomized attackers. They show which tactics are
  better, not the breach rate to expect in the live run.

---

## 9. Run

```bash
cd drone_deployment_infra/swarm_ops
python swarm_policies.py --run defend:3,4,5,6,7 --duration 45          # dry run
python swarm_policies.py --run defend:3,4,5,6,7 --duration 45 --fly
```

- **Order:** the first 3 drones are defenders, the last 2 are attackers.
- **Placement:** all drones face tag 10.
  - Attackers: 1–2 m in front of the tag-10 wall.
  - Defenders: 0.3–1 m in front of tag 6, spread across the room, at least 0.5 m apart.
