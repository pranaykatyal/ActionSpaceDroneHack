# Team 1 plan: Action Space hackathon

**Goal:** score the most of the 100 judging points in the few hours left. The same defender has to
win in simulation **and** fly on the real Tellos.

| Rubric category | Points | How we earn it |
|---|---|---|
| Real-world deployment | 35 | Our defenders protect the asset from the attacking Tellos, with flight logs counting protected vs breached runs |
| Simulation | 25 | Our learned policy beats `chase` on the hosts' `scripts/evaluate.py`, default `AS-Defend-v0` |
| Video | 15 | Sim clip of chase failing next to ours succeeding, plus real-flight footage with the metric on screen |
| Explanation | 15 | One architecture page + results table + honest limitations |
| Creativity | 10 | Hybrid teacher → learned student; later an adversarial attacker pool, if time allows |

## Rules we follow

- **Only the hosts' scripts produce reported numbers:** `scripts/evaluate.py` (scoring) and `scripts/skrl/train.py` (training).
- **Never edit their files.** Everything we add is a new file.
- **Never change the action space.** Every command is `[vx, vy, vz, yaw_rate]` in [-1, 1], i.e. the Tello `rc` sticks.

## Where we stand (measured on this PC with `evaluate.py`, default `AS-Defend-v0`, 128 episodes)

| Pilot | Interceptions | Breaches | Time to intercept |
|---|---|---|---|
| `zero` (do nothing) | 2.3% | 100% | 4.9 s |
| `chase` (fly at the nearest attacker) | **55.5%** | **100%** | 3.1 s |
| distilled chase (step 2, `logs/skrl/tello_defend/2026-10-23_18-19-51_distill_chase/checkpoints/agent_distill.pt`) | 57.8% | 100% | 3.4 s |

| RL fine-tune of distilled chase (their `train.py --checkpoint`, 500 it.) | best 30.5%, final **0.4%** | 100% | — |
| **distilled hybrid** (`2026-10-23_18-25-37_distill_hybrid`) | **169.1%** | **77.3%** | 8.1 s |

### All official scores so far (`evaluate.py`, 128 episodes, 64 envs; interceptions / breaches)

| Policy (checkpoint under `logs/skrl/tello_defend/`) | Official `AS-Defend-v0` | `Team1-Defend-Robust-v0` | `Team1-Defend-Robust-v1` |
|---|---|---|---|
| `chase` (hosts' baseline) | 55.5% / 100% | 38.7% / 100% | — |
| distilled chase (`18-19-51_distill_chase`) | 57.8% / 100% | — | — |
| **distilled chase_live**, robust v0 (`18-35-06_distill_chase_live`) | **128.9% / 90.6%** | 46.9% / 100% | — |
| distilled chase_live, robust v1 (`18-41-47_distill_chase_live`) | 123.8% / 90.6% | — | 43.4% / 100% |
| distilled hybrid, official task (`18-25-37_distill_hybrid`) | **169.1% / 77.3%** | — | — |
| **distilled hybrid**, robust v0 (`18-35-00_distill_hybrid`) | 162.1% / 82.0% | **80.1% / 100%** | — |
| **distilled hybrid, robust v1** (`18-45-23_distill_hybrid`): asset unknown, inference off 30%, confidence gate | **172.3% / 77.3%** (best) | — | 75.8% / 100% |
| same, 3× gradient steps (`19-06-40_distill_hybrid`) | 160.5% / 82.8% | — | — |

**Copy-quality check:** the hybrid's fit MSE rises over DAgger rounds and plateaus near 0.05. Three times
the gradient steps lowered it only ~10% (0.054 → 0.048) and didn't raise the score. So the plateau is
mostly *missing information* (the network can't see its teammate, whose position drives the teacher's
target split), not undertraining. More compute won't fix it.

- **`chase_live`** = the hosts' chase with one fix: target the nearest *live* attacker. Distilling it is the main line, at 2.3× chase's interceptions.
- **Teammates within 0.3 m:** `chase_live` in 41% of episodes (both defenders chase the same attacker); hybrid in 9%. Matters for real flights.
- **Robust tasks** (closer spawns, 0.2 s latency, asset inferred from attackers): v1 adds 30% of episodes where inference is off.

Interception above 100% means more than one wave was cleared per episode. **With endless waves, breach
rate is near 100% for nearly everyone, so interception rate is the number that separates defenders.**

**Finding: RL fine-tuning destroys the distilled policy.** Its training reward fell from −1.6 to −64 and
interceptions from 57.8% to 0.4%. That's the README's "PPO learns to do nothing", even from a warm start.
Likely cause, from the reward code: `closing` pays for *any* shrinking distance, including attackers
flying into a defender that's sitting still, while each capture makes the "nearest attacker" jump
farther away (−5). Distillation doesn't use the reward, so it's unaffected. We don't spend more time
on RL unless we decide to.

**Decision: assume the asset's position is unknown.** Our variant `Team1-Defend-Robust-v0` (new files
under `tasks/direct/team1_defend/`) feeds the policy an asset position **inferred from where the attackers'
flight lines meet** (`control/defense.py` `SiteEstimator`, averaged over 2 s, starting from the defenders'
centroid), with ±0.2 m bias, attacker spawns at 3–6 m (room scale) and 0.2 s command latency. Unit test,
attackers at 0.5 m/s: exact with clean headings; 0.12 m median error with 10° heading noise after 3 s.
We distill on this variant and score on the official `AS-Defend-v0` with `evaluate.py`.

**Why chase is weak (from its code):** a downed attacker's slot in the observation is all zeros, and chase picks
the attacker at the smallest distance. So after the first capture of a wave, each defender "targets" the
dead slot at distance 0 and stops. The second attacker then breaches. The distilled copy inherits this,
which gives RL fine-tuning an obvious first thing to fix: keep chasing the live attacker.

**Note:** the README claims chase gets 100% / 0%. That doesn't reproduce. The task respawns a new wave every
time one is cleared (`respawn_waves=True`), so an episode only ends early on a breach. The breach
rate therefore means "did any wave get through within 20 s". **These are the numbers to beat.**

## The idea in one paragraph

Training an RL policy from scratch on this task fails. The README's "Known issue" says PPO learns to do
nothing: random flying almost never gets within the 0.35 m capture radius, so it never sees a reward
for catching anything. So we first **train the neural network to copy the hosts' `chase` defender**
(*distillation*; the README's own suggested fix, "warm-start from behaviour cloning on the `chase`
baseline"). Chase uses only each defender's own observation, so the network can copy it exactly. That
gives us a network in exactly the format `evaluate.py --checkpoint` loads, already intercepting at
chase's level. Then we **improve it with RL** using the hosts' own `train.py`, starting from that
working network, so the improvement over chase comes from RL. **Decision (2026-10-23): teacher = chase.**
The hand-written *hybrid* stays available as a second teacher (`--teacher hybrid`) and as the
real-flight fallback.

## Glossary

- **Hybrid (teacher):** hand-written controller in `simulation_infra/source/action_space_kit/action_space_kit/control/defense.py` (done, unit-tested). Every control step it:
  1. works out how soon each attacker could reach the site;
  2. works out how soon each defender could intercept each attacker, flying a collision course (aim where the attacker *will be*, the velocity-command version of proportional navigation);
  3. picks the defender → attacker assignment that stops the most attackers before they breach, most urgent first;
  4. if an attacker can't be caught in time, or a teammate is already on it, flies to the point on the attacker's line to the site and blocks it (*interposition*). Scripted attackers don't dodge, so blocking works.
  5. keeps teammates apart and keeps yaw fixed (so on real drones the camera stays on the AprilTag wall).
- **Distillation / behaviour cloning:** supervised learning. Show the network thousands of situations together with the teacher's action, and train it to output the same action. No reward is involved.
- **DAgger:** the distillation recipe we use. Round 1: the teacher flies and we record. Later rounds: the *student* flies, and the teacher labels what it *should* have done. This teaches the student to recover from its own mistakes, which plain copying doesn't.
- **RL fine-tune:** continue training the distilled network with PPO on the real reward (`train.py --checkpoint`), so it can beat the teacher.

## Steps, commands and time

Training throughput was measured on this PC (RTX A5500): **4096 envs ≈ 0.38 s per iteration**, so
300 iterations ≈ 2 min and 1000 ≈ 6.5 min. One `evaluate.py` run ≈ 1 min.

| # | Step | What runs | Compute | Our time |
|---|---|---|---|---|
| 0 | Setup and baselines | Done | — | done |
| 1 | Hybrid teacher | `control/defense.py`, done and unit-tested | — | done |
| 2 | **Distill chase** into their IPPO network | new `scripts/skrl/distill.py --teacher chase` builds the agent exactly like `evaluate.py`, runs DAgger, saves with skrl's own `agent.save()`. Also pre-trains the value network and lowers the exploration noise, so step 3 doesn't wreck it | ~5 min | running |
| 2b | Score it | `python -u scripts/evaluate.py --task AS-Defend-v0 --checkpoint <distilled.pt> --episodes 128` | ~1 min | — |
| 3 | **RL fine-tune** from the distilled network | `python -u scripts/skrl/train.py --task AS-Defend-v0 --algorithm IPPO --headless --num_envs 4096 --max_iterations 500 --checkpoint <distilled.pt>`, then `evaluate.py` on the result. Keep whichever scores best | ~4 min per run | ~30 min |
| 4 | **Sim-to-real hardening** | Fine-tune again with command latency (`env.command.command_latency_steps=…`, 0.1–0.3 s). Register *new* task IDs (new files, e.g. `AS-Defend-Room-v0`: room scale, latency) and score them with `evaluate.py --task …` | ~5 min per run | ~45 min |
| 5 | **Real-world flight** (35 pts) | New swarm_ops script; see below | — | ~1.5–2 h |
| 6 | Video + explanation | Sim clips (chase vs ours), real footage, architecture page, results table | few min | ~30–45 min |

**Decision points:**
- After 2b: the distilled student should score about the same as chase (it's a copy). If it's clearly worse, the copy failed; fix that before step 3.
- After 3: if RL makes it worse, we ship the distilled network and say so.

**Priority if time runs short:**
1. Step 2: a learned policy that beats chase (sim points).
2. Step 5: real flights with the hybrid (biggest points).
3. Step 6: video and explanation (cheap points).
4. Steps 3–4: RL fine-tune and latency.

**Step 5 can run in parallel** with the sim work: it flies the hybrid directly, with no training needed.

## Tunable parameters

What we may change, where it lives, and what it does. **The official score is always taken on
the default `AS-Defend-v0` with the hosts' `evaluate.py`, so layers 1 and 2 are never changed for
scoring.** They can only be changed for *training* (our variants, or `train.py env.*` overrides).

**1. The official scenario: fixed for scoring** (`tasks/direct/tello_defend/tello_defend_env_cfg.py`, hosts' file, read-only)

| Parameter | Default | What it does |
|---|---|---|
| `num_defenders` / `num_attackers` | 2 / 2 | team sizes (the real fleet is 2 vs 3) |
| `site_radius`, `site_height` | 1.5 m, 2.5 m | protected cylinder; a breach = a live attacker inside its radius |
| `spawn_radius` | 6.0 m | where attackers appear (random bearing each wave) |
| `defender_radius` | 2.5 m | defenders start on this ring (random bearings), 1–2 m up |
| `capture_radius` | 0.35 m | a defender this close downs an attacker |
| `attacker_speed` | [0.4, 0.9] m/s | per attacker, per wave |
| `attacker_evasion` | 0.0 | 0 = straight dive; up to 1 = sideways weave |
| `respawn_waves` | True | a cleared wave is replaced; only a breach ends an episode early |
| `episode_length_s` | 20 | |
| `command.max_lin_speed` / `max_yaw_rate` | 1.0 m/s / 1.745 rad/s | action ±1 = this speed |
| `command.response_time`, `max_lin_accel` | 0.15 s, 4 m/s² | the simple Tello velocity model |
| `command.command_latency_steps` | 0 | delay in physics steps (1/120 s each) |

**2. Rewards: only matter for RL** (`rew_scale_*` in the same file; distillation ignores them)

closing 5.0 · proximity 2.0 · capture +30 · wave cleared +30 · breach −20 · effort −0.01 · alive +0.1.
Likely cause of PPO's drift to passivity: `closing` pays for *any* shrinking distance, including
attackers flying into a defender that's sitting still.

**3. Our training variants: free to change** (`tasks/direct/team1_defend/team1_defend_env_cfg.py`)

| Parameter | v0 | v1 | What it does |
|---|---|---|---|
| `site_memory_s` | 2.0 | 2.0 | asset estimator's averaging window |
| `site_bias_xy` | ±0.2 m | ±0.2 m | per-episode error added to the inferred asset position |
| `site_inference_off_prob` | 0 | **0.3** | share of episodes where the asset can't be inferred |
| `site_bias_off_xy` | ±1.0 m | ±1.0 m | error of the starting guess when inference is off |
| `spawn_radius_range` | 3–6 m | 3–6 m | closer attacker starts (room scale) |
| `command.command_latency_steps` | 24 (0.2 s) | 24 | real SDK latency |

**4. The hybrid teacher** (`control/defense.py`, `HybridDefenseCfg`)

| Parameter | Default | What it does |
|---|---|---|
| `max_speed` | 1.0 m/s (sim) / ~0.4 real | the speed its intercept geometry assumes |
| `guard_radius` | 2.2 m | where a blocker stands on the attacker's line to the site |
| `breach_margin` | 0.5 s | an intercept must land this long before the breach to count |
| `terminal_range` | 0.7 m | inside this, fly straight at the attacker |
| `max_lead_time` | 6 s | how far ahead it predicts the attacker |
| `hysteresis` | 0.5 s | bonus for keeping the current target (stops flip-flopping) |
| `guard_gain` | 2.0 /s | how hard a blocker moves onto its point |
| `separation`, `separation_gain` | 0.7 m, 1.5 | teammate repulsion (raise for real flights) |
| `min_height` | 0.5 m | floor clearance |
| `path_guard_min` | 0.5 m | asset unknown: block at least this far ahead on the attacker's line |
| `SiteEstimator` `min_speed` / `prior_weight` / `SITE_CONFIDENT` | 0.25 m/s / 0.02 / 0.2 | which attackers count, the pull of the starting guess, the trust threshold |

**5. Distillation** (`scripts/skrl/distill.py` flags)

| Flag | Default | What it does |
|---|---|---|
| `--teacher` | chase | `chase`, `chase_live` or `hybrid` |
| `--task` | AS-Defend-v0 | which env to collect states in (our variants for robustness) |
| `--num_envs` | 4096 | parallel envs |
| `--rounds`, `--steps`, `--keep_every` | 12, 200, 2 | DAgger rounds × steps per round; ~4.9 M samples per defender |
| `--grad_steps`, `--batch`, `--lr` | 600, 8192, 1e-3 | supervised training per round |
| `--log_std` | −1.2 (std 0.3) | exploration noise saved for later RL |
| `--value_steps`, `--value_fit`, `--value_grad_steps` | 700, 400, 1500 | value-function fit, only used by RL fine-tuning |

**6. RL fine-tuning** (`agents/skrl_ippo_cfg.yaml`, hosts' file; override with `train.py agent.*=…`)

rollouts 16 · learning_epochs 5 · mini_batches 4 · discount 0.99 · λ 0.95 · lr 5e-4 with
KL-adaptive schedule (kl_threshold 0.016) · ratio_clip 0.2 · entropy 0 · value_loss_scale 2.
**Network 23→256→128→64→4 (ELU): fixed**, because `evaluate.py` builds it from this file.

**7. Real flights (to build):** speed cap 0.5 m/s · **contact is the capture** (hosts, 2026-10-24: drones are meant to collide; no stand-off, `contact_offset 0`, `obstacle_radius 0`) · room-scaled hybrid
(max_speed ≈ 0.4, site ≈ 0.6 m, guard ≈ 1.0 m, separation ≈ 0.9 m) · rc gain ≈ 0.0075 m/s per rc unit ·
geofence · land after 3 s of stale link.

**Where the score is likely to come from:** layer 4 (teacher quality: tune it, check it with `distill.py --check`, re-distill) and layer 5 (copy quality).
**Where sim-to-real comes from:** layers 3 and 7.

## Sim-to-real: what we account for from the start

| Real-world fact | What we do |
|---|---|
| Tello takes only `rc` velocity sticks | Same `[vx, vy, vz, yaw]` action; yaw rate held at 0 |
| Positions come only from AprilTags on one wall | Yaw locked toward the tag wall; asset placed near that wall, attackers fly toward it so they keep seeing tags; Kalman filter (`core/positional_kalman_filter.py`) gives position + velocity, which is the same observation the network uses |
| Indoor speed cap 0.5 m/s | Cap commands at 0.5 m/s at deploy; check performance at that cap in sim |
| 0.1–0.3 s command latency, 13.5 Hz rc rate | Fine-tune with latency (step 4); hold commands between rc packets |
| Sign conventions differ (sim `vy` left+, yaw CCW+; SDK `a` right+, `d` CW+) | Use the correctly-signed conversion as in `action_space_kit/control/rc_action.py`, **not** `bridge/tello_bridge.py`, whose `to_rc` doesn't negate |
| Sim frame ≠ map frame | Reuse the map → sim transform in `drone_deployment_infra/core/room_policy.py` |
| Room is smaller than the sim (6 m spawn) | Room-scale task variant for checking; the hybrid takes the geometry as parameters |

## Real-world flight (step 5)

- **Fleet:** defenders TELLO-1 and TELLO-2; attackers TELLO-3, 4 and 5 (`config/fleet.yaml`), all from one laptop.
- **New script** in `drone_deployment_infra/swarm_ops/` (a new file; `swarm_formation.py` is the template because it already localizes every drone and shares positions inside one process):
  - Attackers fly a scripted path toward the asset.
  - Defenders run the policy, or the hybrid as a fallback.
  - Every drone's position comes from its own AprilTag view + Kalman filter.
  - Safety: 0.5 m/s cap, geofence, land on lost link / Ctrl-C, CSV log of every flight.
- **Order:** dry run (props off) → 1 defender alone → 2 defenders vs 1 attacker → 2 vs 3.
- **Evidence:** count protected vs breached runs from the logs, which gives a real success rate.

### Real-drone code: what exists and what's missing (reviewed 2026-10-23)

**SF** = `drone_deployment_infra/swarm_ops/swarm_formation.py`.

| Need | Status |
|---|---|
| All 5 drones localized in one process | **Exists.** SF runs video + AprilTag + `localize()` per drone, and every drone can read the others (`est()`). 8–12 fixes/s per drone with 4 flying. Frame: x right, y down, z into the tag wall, m. |
| Velocity of each drone (intercept points, asset inference and the network all need it) | **Missing.** SF keeps only a smoothed position. `core/positional_kalman_filter.py` `KalmanCV` works on any frame, so feed SF's raw fixes into it (subclass `Drone`, hook `_accept`). Don't trust Tello `vgx/vgy/vgz`: units are unverified. |
| SF frame → our frame (z up) | **Exists** in `core/room_policy.py` (`P_MAP_TO_WORLD`, tag 10 at 1.22 m). From SF: p_w = (−p_z, p_x, 1.22 − p_y), yaw_w = 180° − psi. Derived from the code but **not yet run**; check on a drone. |
| World velocity → rc sticks with correct signs | **Missing as a function.** SF's `control()` (line 213) does the body-frame rotation correctly and is the template. **Don't use `bridge/tello_bridge.py`**: wrong `a` and `d` signs, and no world→body rotation. ~0.0075 m/s per rc unit, so 0.4 m/s ≈ rc 50, above SF's caps (15/35). |
| Keep the camera on the tag wall while moving | **Exists** (SF `yaw_to`, `face_tag_yaw`). Replaces the hybrid's yaw rate of 0. |
| Attackers fly straight at the asset | **Partial** (`tello_swarm_to_origin.py` flies to spots). Need: v = 0.3–0.5 m/s toward P, one height per attacker, facing the tag wall, so **attackers fly toward the tag wall and the asset sits near it**. |
| Safety | **Partial.** SF has all-or-nothing preflight, a typed FLY prompt, Ctrl-C / `q` lands all, 15% battery landing and a CSV log. **Missing:** landing on lost link or stale status (copy `swarm_policies.py`'s 3 s rule), a room geofence, and a speed cap in m/s. |
| Running the network on the laptop | **Partial.** skrl isn't needed: torch alone loads the checkpoint (MLP 23→256→128→64→4 + running mean/var scaler, `clamp(out, −1, 1)`). The network sees **exactly 2 attackers**, so with 3 each defender gets its nearest 2 live ones. The hybrid handles any number natively. |
| How a capture works | **Contact** (hosts, 2026-10-24: drones are meant to collide). Defenders aim straight at the attacker. Keep defender–defender separation, because two defenders colliding is a loss with no capture. A defender that goes down after contact is out (no relaunch assumed). |
| Venue tag map | Surveyed by a teammate with `cal_and_map/tello_map.py` (same schema and frame as SF reads). In the world frame (x out of the wall, y right, z up, tag 10 at 1.22 m): tag 4 (1.86, 0.16, 0.01) and tag 7 (2.70, 1.44, 0.04) are on the floor; tag 5 (0.57, 2.00, 1.49); **tag 6 (6.48, 1.12, 0.67) is suspect** (beyond the ~4.5 m room): tape-measure it or `--ignore-tags 6`. Needs the actual `map.yaml` (the printout has no rotations). Save it as a new file and pass `--map`; don't overwrite the tracked `config/map.yaml`. |

**New files to write (nothing existing modified):**
1. `defense_lib.py`: SF `Drone` subclass + Kalman velocity, frame conversion, world-velocity→rc with yaw hold, stale-link landing, geofence, CSV log.
2. `attacker_line.py`: straight run toward P, per-drone height, land when caught.
3. `defense_run.py`: one process for drones 1–5: SF preflight → FLY prompt → localize → 20 Hz loop: Kalman → `SiteEstimator` → `HybridDefender` scaled to the room (max speed ≈ 0.4 m/s, site ≈ 0.6 m, guard ≈ 1.0 m, separation ≈ 0.9 m) or the network → rc.
4. `skrl_cpu_policy.py`: load the checkpoint with torch, build the 23-number observation.
5. `replay_check.py`: run the defender offline on logged flights, before flying.

**Top real-world risks:**
- **Tags out of view:** chasing turns drones away from the tag wall, and SF lands after 4 s without a fix.
- **Collisions:** contact with attackers is intended. Collisions between defenders are not, so keep hard separation.
- **Room scale:** the room is about 4.5 × 3.4 m against the sim's 6 m spawns and 1 m/s.
- **Radio:** 5 drones on 2.4 GHz; non-EDU motor interference.
- **Sign and frame mistakes.**

## Risks and open questions

- **Teammate-blind observations:** in `AS-Defend-v0` each defender observes only itself and the attackers, never its teammate. The network can only partly copy the teacher's target-splitting. Each defender has its own network, so they can learn different roles; we'll measure how much survives.
- **Real attackers must localize themselves** (camera facing the tag wall) and run in the defenders' process, or the defenders can't know where they are. Only 3 tags are mapped, all on one wall, so the map may need extending (`map_tags.py`).
- **Radio:** 5 video streams on 2.4 GHz can pixelate. Non-EDU Tellos can drop the link shortly after the motors start (`FLYING.md`).
- **Still to ask the hosts:**
  - What counts as "protecting the asset" in the real test, and where is the asset?
  - May our defenders use positions from attacker drones we fly ourselves?
  - What are the video length, format and deadline?
