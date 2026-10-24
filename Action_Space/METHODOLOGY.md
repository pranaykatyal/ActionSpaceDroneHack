# How Team 1 defends 2 v 2 (`AS-Defend-v0`)

**Result:** the hosts' unchanged IPPO network, with our weights, holds the asset in **98.3% of
episodes (1.7% breach) with 389% interception**. That's about 3,000 episodes of the hosts' own
`scripts/evaluate.py`, over three scenario sets, and 1.6% breach at its defaults. The hosts' `chase`
baseline breaches 100% of episodes. Nothing in the hosts' environment, reward, observation, action
space or evaluator was changed. The policy file is
`simulation_infra/policies/team1_defend_guard20/agent_20.pt`.

The recipe in one line: **write a teacher that a network can copy, copy it, then fine-tune the
copy on the score itself.**

```
scripted teacher            DAgger distillation          evolution strategies       hosts' evaluate.py
(each defender's own   -->  into the hosts' IPPO    -->  on whole-episode       -->  389% / 1.7%
 23 observations)           network (unchanged)          breach / hold / capture     (~3,000 episodes)
 379% / 1.2% *              370% / 2.9%                  generation 20
```
\* our internal counter (`distill.py --check`, counted like `evaluate.py`), not an official score.

---

## 1. The problem

| | `AS-Defend-v0` (hosts' defaults) |
|---|---|
| Teams | 2 defenders vs 2 scripted attackers. Attackers start 6 m out, fly 0.4–0.9 m/s at a 1.5 m protected cylinder. Defenders start on a 2.5 m ring. |
| Capture | a defender within 0.35 m downs an attacker. A downed attacker stays hovering where it was caught. |
| Episode | 20 s. When both attackers are down, a new wave spawns at once. Only a breach ends an episode early. |
| Observation (per defender, 23 numbers) | own position relative to the asset, own velocity, gravity direction, each attacker's relative position and velocity, attacker alive flags. **The teammate is not observed**, and downed attackers are zeroed. |
| Action | `[vx, vy, vz, yaw_rate]` in [-1, 1] (velocity commands to a Tello-like controller). We never changed this. |
| Score | `evaluate.py`: **breach rate** (share of episodes the asset is entered) and **interception rate** = captures / (2 × episodes). Above 100% means more than one wave was cleared per episode. 389% ≈ 7.8 captures in 20 s. |

We rank by breach rate first, then interceptions.

## 2. Results (hosts' `evaluate.py`, AS-Defend-v0)

| Policy | Interception | Breach | Episodes |
|---|---|---|---|
| `zero` (do nothing) | 2.3% | 100% | 64 envs × 128 |
| `chase` (hosts' baseline: greedy pursuit of the nearest attacker) | 55.5% | 100% | 64 × 128 |
| Best plain PPO fine-tune we found (hosts' `train.py`, reward and network) | 204.8% | 69.7% | 100 × 300 |
| Distilled teacher, no guard (section 5) | 365.1% | 5.7% | 100 × 1,000 |
| ES gen 35, no guard (previous pick) | 388.0% | 3.9% | ~3,000 (three sets) |
| Distilled teacher with guard | 370.2% | 2.9% | 100 × 1,000 |
| **Guard ES gen 20 (final pick)** | **389.3%** | **1.7%** | ~3,000 (three sets) |
| Guard ES gen 20, `evaluate.py` with no arguments | 389.1% | 1.6% | defaults (64) |

The README lists `chase` at 100% / 0%. That doesn't reproduce today: waves keep respawning
until one gets through, so `chase` eventually breaches every episode.

## 3. Why plain RL didn't get there

- **Out of the box PPO learns to do nothing.** The hosts' README documents this.
- **Starting PPO from a good policy, it gets better for 25–40 iterations, then falls apart.** We found
  three reasons:
  - All 4096 envs start their episodes at the same moment, and a rollout covers only 0.27 s. The first ~19
    updates contain no breach at all. Staggering the starts fixed the early dips, but not the collapse.
  - The hosts' `closing` reward pays only for flying straight at the attacker's current position.
    So PPO un-learns leading the target, aiming beside it and blocking, which are the behaviours that win.
  - Policy and value share one trunk, so a poorly fitted critic pulls the policy's features around.
- Things that didn't help: longer horizons (γ 0.997 / 0.999), an events-only reward, and longer training
  (300 iterations).

## 4. The key insight: crashes, not misses, cause breaches

In long diagnostic runs, defenders spent **86–97% of the time on the floor.** A defender that hits
anything at full speed tips over. In this sim a tipped drone never recovers: the controller's
world-frame force is applied in the body frame. A downed defender leaves its attacker free.

So the teacher's job became **capture without crashing**:

| Teacher (internal counter, 256 episodes) | Interception | Breach | Defender time on floor |
|---|---|---|---|
| chase the nearest live attacker | 134.6% | 90.2% | 17.6% |
| assigned targets, full-speed contact | 155.3% | 98.8% | 36.8% |
| **assigned targets + soft contact (0.15 m/s)** | **373.2%** | **4.7%** | **1.5%** |

That one change, a gentle bump instead of a ram, takes breaches from 99% to 5%.

The second biggest cause was the two defenders hitting each other. When a defender tips over, 48% of
the time it has just touched its teammate, 28% a just-captured attacker, and 24% nothing (152 flips,
gen 35). That led to the guard post below.

## 5. The pipeline

### Step 1: a teacher built only from what the network can see (`scripts/skrl/distill.py --teacher assigned`)

The network sees only its own 23 numbers. So the teacher uses only those, and the defenders
coordinate by **convention** instead of communication:

- **Fixed pairing:** defender *i* takes attacker slot *i* while that attacker is alive. The two defenders
  never converge on one attacker, which was the usual way they collided.
- **Lead intercept:** fly at full speed to the predicted meeting point, not at the attacker's current position.
- **Soft contact:** inside 0.7 m, fly the attacker's own velocity plus 0.15 m/s closing speed, so contact
  is a slow bump and the defender stays airborne. The 0.7 m range is needed: at 0.5 m the velocity lag
  can't shed speed in time (teacher: 63.7% breach).
- **Guard post** (`--guard 1.8 1.5`): once its own attacker is down, a defender does *not* join its teammate.
  It waits on the live attacker's line to the asset, 1.8 m out at the attacker's height, and engages only
  when that attacker is within 1.5 s of breaching. On the official task (internal counter, 64 × 640) this
  cuts the teacher's breaches from 4.4% to 1.2%, and defender time on the floor from 1.5% to 0.3%.

Because every input is in the observation, a network can copy it almost exactly. Earlier teachers that
used the teammate's position couldn't be copied (copy error 0.07–0.08, distilled breach 77–83%).

### Step 2: distillation (DAgger) into the hosts' IPPO network

- Network: the hosts' skrl IPPO MLP 256-128-64 plus its running scaler, unchanged, so `evaluate.py`
  loads it as is.
- Training: 12 DAgger rounds on 4096 envs. The student flies, the teacher labels what it should have
  done, and the student trains on those labels.
- Result: copy error (mse) 0.0031.
- Official: **370.2% / 2.9%** (100 × 1,000). Without the guard rule: 365.1% / 5.7%.

### Step 3: evolution strategies on the score itself (`scripts/skrl/es_train.py`)

PPO broke good behaviour, so we fine-tuned without a critic or a per-step reward:

- **Method:** OpenAI-ES on both defenders' networks (95,112 weights; the scaler stays frozen). Each
  generation flies 120 perturbed copies (antithetic pairs) on 32 envs each, plus 256 envs on the
  unperturbed network. The weights move toward the better copies (centred ranks, Adam).
- **Settings:** σ 0.003, learning rate 0.0005, 60 generations.
- **Fitness per episode:** 1 if no breach + 0.5 × the fraction of the episode held + 0.01 × captures −
  0.5 × the fraction of defender time spent on the floor. This is close to what `evaluate.py` counts.

**How the final checkpoint was picked:**
- Only gens 10, 15 and 20 were scored, to avoid picking on noise.
- They were compared on the same 100 × 1,000 scenarios as the previous pick (gen 35, no guard), then
  confirmed on two scenario sets none of them was picked on:

| Hosts' `evaluate.py` | 100 × 1,000 | 128 × 1,000 (unseen) | 200 × 1,000 (unseen) | pooled ~3,000 | defaults (64) |
|---|---|---|---|---|---|
| **guard ES gen 20 (final)** | 388.6% / 1.5% | 387.1% / 2.5% | 392.1% / 1.0% | **1.7%** | 389.1% / 1.6% |
| guard ES gen 10 | 384.7% / 1.6% | 386.8% / 1.8% | 385.2% / 1.7% | 1.7% | 395.3% / 3.1% |
| ES gen 35, no guard (previous) | 390.5% / 3.2% | 385.3% / 5.2% | 388.2% / 3.3% | 3.9% | 381.2% / 4.7% |

On the unseen sets that's **36 breaches for gen 20 vs 86 for gen 35**. Gens 10 and 20 tie on breaches;
gen 20 wins on interceptions and on the defaults score.

## 6. Robustness (hosts' `evaluate.py --task` on our scenario variants)

Measured on ES gen 35, the same pipeline without the guard rule; not yet re-run on gen 20.

| Attacker speed (official everything else, 100 envs × 300 episodes) | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 | mixed 0.1–1.0 |
|---|---|---|---|---|---|---|
| breach | 0.0% | 0.3% | 1.7% | 1.7% | 7.6% | 11.7% |

| Other variants (64 envs × 640 episodes) | Interception | Breach |
|---|---|---|
| weaving, wobbling attackers at 0.2–1.0 m/s | 342.8% | 6.4% |
| attackers spawn at 5 m (official: 6 m) | 454.3% | 8.5% |
| attackers spawn at 4 m | 429.3% | 40.2% |
| attackers spawn at 3 m | 114.5% | 98.4% |

ES ended up far better than its teacher on fast attackers: 7.6% vs 34.3% breach at 1.0 m/s.

## 7. Three defenders vs two (in training)

The real demo flies 3 defenders against 2 attackers, so a separate 3-defender policy is being built
with the same steps (`Team1-Defend-3v2-v0`: the hosts' environment class and geometry with
`num_defenders=3`).

- The simplest extension, where the third defender doubles up on attacker 1, fails: two defenders on
  one attacker crash into each other (teacher: 287.0% / 46.6% breach, 21% floor time).
- So defenders 1 and 2 follow the guard recipe, and the third plays **goalkeeper**. It holds a post near
  the asset on the most urgent attacker's line and engages only when that attacker is within 1.5 s of
  breaching (`distill.py --keeper`).
- Then DAgger and 100 generations of ES. Results will be added when it finishes. It will be saved to
  `simulation_infra/policies/team1_defend_3v2/`.

## 8. Strengths

- **Scored entirely on the hosts' terms:** their environment, network, action space and `evaluate.py`,
  unchanged. Our scenario variants are new task IDs, not edits.
- **Decentralised by construction:** each defender acts on its own observation alone. They coordinate
  by fixed conventions (target pairing, guard post), with no communication or shared state.
- **The fine-tune optimises the score itself** (breach, time held, captures). It doesn't rely on a
  per-step reward that, as we found, punishes leading and blocking.
- **Robust beyond the training scenario:** weaving attackers, attacker speeds from 0.1 to 1.0 m/s, and
  5 m spawns stay under 12% breach (gen 35).
- **Picked honestly:** the final checkpoint was confirmed on ~2,000 episodes it wasn't selected on.
- **One code path from sim to flight:** the laptop layer is a single file that the sim loads and scores.

## 9. Limitations

- **The teammate is invisible to the network.** Conventions (pairing, guard post) stand in for seeing it.
  The guard post halved breaches, but the defenders still can't react to each other directly. The rest has
  to live outside the network (section 10).
- **Close spawns are a cliff** (gen 35). Breach goes from 8.5% at 5 m to 40.2% at 4 m and 98.4% at 3 m.
  - At 3 m, defenders starting on the 2.5 m ring can cover both attackers in only ~30% of waves, even with
    perfect pairing.
  - Starting the defenders 1.0 m from the asset cuts first-wave breaches from 72% to 17%. At close range,
    where the defenders start matters more than the network.
- **Delay hurts** (gen 35). With 3–6 m spawns, weaving attackers, 0.2 s command delay and an estimated
  asset position, breach is 80.2%; removing the delay alone gives 58.9%.
- **Soft contact is partly a sim artefact.** In the sim, a hard hit tips the defender for good. In real
  flights the hosts count contact as the capture. The slow final approach is still the safer way to touch.
- **The real-Tello command model is unfitted.** Our variant with the hosts' motion priors (τ 0.5 s,
  0.75 m/s, 0.1 s delay, 13.5 Hz rc) breaches 88.1% (gen 35), mostly because attackers there are as fast
  as the defenders and start close. A step-response flight would replace the priors.
- **Measurement caveat.** `evaluate.py` stops after N finished episodes. A breached env restarts and can
  breach again, so it counts several times. With `--episodes` near `--num_envs`, breach rates are inflated.
  - The headline sets run 5–10 × num_envs episodes (100 × 1,000, 128 × 1,000, 200 × 1,000), and the
    speed sweep 3 × (100 × 300). The bias only adds breaches, so these numbers err high, not low.
  - The 64 × 640 variants are 10 ×.
  - The defaults are seeded and score 64 episodes, about ±5 points.

## 10. From sim to the real Tellos

The submitted network is bare. For real flights we add a small layer on the laptop, around the
unchanged network (`drone_deployment_infra/core/defend_deploy.py`). The sim loads the same file, so what
we score is what flies:

1. **Delay compensation:** before the network sees its observation, every drone is moved forward by the
   command delay, assuming constant velocity.
2. **Teammate separation filter:** if two defenders within 1.5 m are closing too fast to stop before 0.3 m,
   their velocity commands get just enough push apart (a control-barrier-function filter). It never
   avoids attackers, because contact is the capture.

| Hosts' `evaluate.py --task`, ES gen 35, 64 × 640 | Interception | Breach |
|---|---|---|
| 3–6 m spawns + weaving attackers + 0.2 s delay + estimated asset | 197.2% | 80.2% |
| same, with no delay (the target) | 295.7% | 58.9% |
| same, + delay compensation | 259.8% | 57.3% |
| same, + delay compensation + separation filter | 280.3% | **52.0%** |
| official scenario | 385.2% | 4.1% |
| official scenario + separation filter | 390.4% | **2.5%** |

Delay compensation recovers the whole delay gap. The filter adds a small gain and removes defender pairs
closer than 0.3 m; the two always run together. These were measured on gen 35, before the guard post.

In training: a separate **flight network**, kept apart from the official submission. It is trained on the
realistic command path with the laptop layer in the loop. That path has positions-only sensing, 0.1 s
command delay, 13.5 Hz integer rc and a 0.75 m/s cap, with attackers on the official 6 m ring.

- Its teacher does **not** slow down near the attacker. In real flights the attacker has to be knocked
  out of the air, and a 0.15 m/s bump probably won't do that.
- A first, soft-contact ES run from guard gen 20 was stopped at generation 12: breach went from 39.5% to
  33.6%, within ±3 points.
- Guard gen 20's baseline on this path is 40.2% breach with the layer and 62.2% without it.

Still to do before flying:
- a step-response flight to fit τ and delay;
- rotating the room-frame velocity commands into each drone's heading frame for `rc`;
- the laptop flight loop that runs the layer.

## 11. Reproduce

From `simulation_infra/`, with the shell set up as in `CLAUDE.md` step 2:

```bash
# score the final network (hosts' evaluator, unchanged)
python -u scripts/evaluate.py --checkpoint policies/team1_defend_guard20/agent_20.pt
python -u scripts/evaluate.py --checkpoint policies/team1_defend_guard20/agent_20.pt --num_envs 200 --episodes 1000
python -u scripts/evaluate.py --baseline chase          # the hosts' baseline, for comparison

# rebuild it
python -u scripts/skrl/distill.py --task AS-Defend-v0 --teacher assigned --contact_offset 0 --soft_closing 0.15 --guard 1.8 1.5
python -u scripts/skrl/es_train.py --task AS-Defend-v0 --checkpoint <distill run>/checkpoints/agent_distill.pt \
  --num_envs 4096 --pop 120 --center_envs 256 --sigma 0.003 --lr 0.0005 --w_capture 0.01 --w_floor 0.5 --generations 60

# video (45 s, labelled, running interception and breach counts)
python -u scripts/record_defend.py --checkpoint policies/team1_defend_guard20/agent_20.pt \
  --label "Team 1: guard ES gen 20" --out outputs/videos/defend_final_guard20.mp4 --frame_dir outputs/frames_final_guard20
```

| File | What |
|---|---|
| `simulation_infra/policies/team1_defend_guard20/` | final network, sha256 and provenance |
| `simulation_infra/policies/team1_defend_es35/` | previous pick (no guard rule) |
| `simulation_infra/scripts/skrl/distill.py` | teachers (`assigned`, soft contact, guard, keeper) and DAgger |
| `simulation_infra/scripts/skrl/es_train.py` | evolution-strategies fine-tune, saves skrl checkpoints |
| `simulation_infra/scripts/record_defend.py` | labelled video of any checkpoint or baseline |
| `simulation_infra/source/.../tasks/direct/team1_defend/` | our scenario variants (new task IDs; hosts' files untouched) |
| `drone_deployment_infra/core/defend_deploy.py` | laptop layer: delay compensation + separation filter (self-test included) |
| `simulation_infra/outputs/videos/defend_final_side_by_side.mp4` | the demo: hosts' chase (left, 5 interceptions / 5 breaches in 45 s) vs our final pick (right, 17 / 0) |
| `simulation_infra/outputs/videos/defend_final_*.mp4` | each clip on its own: final pick, guard gen 10, previous pick, chase (seed 7: same first wave) |
| `EXPERIMENTS.md` | every run, including the ones that failed |
