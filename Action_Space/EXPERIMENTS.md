# Experiment log: team 1, `AS-Defend-v0`

Every run, its settings, its result and a verdict. Updated as runs finish.

**Conventions**
- **Scores:** the hosts' `scripts/evaluate.py` on the official default `AS-Defend-v0`, 64 envs, 128 episodes unless noted. Each score is **interception rate / breach rate / time to intercept**.
- **Interception above 100%** means more than one attack wave was cleared within a 20 s episode.
- **Seeds:** `evaluate.py` sets no seed ("Seed not set for the environment"), so every scoring run sees fresh random scenarios.
- **Run folders** are under `simulation_infra/logs/skrl/tello_defend/`; console logs are in `simulation_infra/outputs/`.

**Goal and ranking:** reduce **breach rate** first, then raise **interception rate**. Candidates are ranked by breach rate, ties broken by interceptions, and confirmed on fresh episodes (256) plus the Robust-v1 task, so we don't pick noise or overfit the official scenario.

**Current best (2026-10-24, final pick): `2026-10-24_07-52-17_es_guard_s003/checkpoints/agent_20.pt`** (copy: `simulation_infra/policies/team1_defend_guard20/agent_20.pt`), ES fine-tune of the distilled assigned + soft-contact + **guard** network. Hosts' `evaluate.py`, AS-Defend-v0: **defaults 389.1% / 1.6%**; 100 × 1,000 388.6% / 1.5%; unseen 128 × 1,000 387.1% / 2.5% and 200 × 1,000 392.1% / 1.0%; **pooled ~3,000 episodes: 1.7% breach**, vs 3.9% for gen 35 on the same three sets.

~~Previous pick (superseded 2026-10-24 by guard gen 20): `2026-10-24_01-25-32_es_assigned_s003/checkpoints/agent_35.pt`, ES fine-tune of the distilled assigned + soft-contact network. Official `evaluate.py`, 1,000 fresh episodes: **390.5% interception / 3.2% breach / 10.7 s**. Across Tello attacker speeds: 0.0–1.7% at 0.1–0.75 m/s, 7.6% at 1.0 m/s, 11.7% mixed 0.1–1.0. With weaving attackers (Realistic): ~~1.6%~~ **6.4%** (clean 64 × 640; the 1.6% was 128 episodes).~~

Previous best: `2026-10-24_01-15-18_distill_assigned_147307/checkpoints/agent_distill.pt`, the network distilled from the `assigned` + soft-contact (0.15 m/s) teacher on AS-Defend-v0. Official `evaluate.py`: **362.0% / 6.3%** (100 envs × 300 ep, fresh) and 367.6% / 7.0% / 10.8 s (64 × 128). The goal (≤ 20% breach) is met; now pushing lower.

~~Current best:** `2026-10-23_19-48-37_ippo_torch_ft2b_chase_live/checkpoints/agent_400.pt` (RL iteration 25 from distilled chase_live).~~ (superseded)

| Task | Interceptions / breaches |
|---|---|
| Official, **fresh** scenarios (100 envs × 300 episodes) | **204.8% / 69.7%** |
| Robust v1 | 82.0% / 100% |
| Weave 0.5 / weave 1.0 (iteration 30 checkpoint) | 185.2% / 71.1% / 231.4% / 39.4% |
| IRL | 86.3% / 94.5% |

**Determinism (found 2026-10-23):**
- The hosts' `train.py` is deterministic **and ignores `--seed`**: FT2 (seed 42) and FT2b (seed 1) produced byte-identical checkpoints. FT2b is FT2 with finer checkpoints, not a replicate.
- `evaluate.py` replays the same scenarios for the same `--num_envs`. For a fresh confirmation, change `--num_envs` (e.g. 100).

## Baselines

| ID | Pilot | Official result | Notes |
|---|---|---|---|
| B0 | `zero` | 2.3% / 100% / 4.9 s | |
| B1 | `chase` (hosts' baseline) | 55.5% / 100% / 3.1 s | The README's 100% / 0% does **not** reproduce: waves respawn until a breach |
| B1r | `chase` on Team1-Defend-Robust-v0 | 38.7% / 100% / 2.4 s | closer spawns + 0.2 s latency |
| B1s | `chase` on Team1-Defend-Single-v0 (1 defender vs 2 attackers) | 50.0% / 100% (64 episodes) | |

## Distillation (`scripts/skrl/distill.py`, DAgger)

12 rounds × 200 steps, 4096 envs, 600 gradient steps per round, std saved at 0.30 unless noted.

| ID | Teacher | Trained on | Run | Official | Other task | Verdict |
|---|---|---|---|---|---|---|
| D1 | chase | AS-Defend-v0 | `18-19-51_distill_chase` | 57.8% / 100% / 3.4 s | — | exact copy of chase (action error 0.005) |
| D2 | hybrid | AS-Defend-v0 | `18-25-37_distill_hybrid` | 169.1% / 77.3% / 8.1 s | — | partial copy (error 0.07: can't see teammate) but strong |
| D3 | chase_live | Robust-v0 | `18-35-06_distill_chase_live` | 128.9% / 90.6% / 6.9 s | Robust-v0 46.9% / 100% | fixing chase's dead-target flaw doubles it |
| D4 | hybrid | Robust-v0 | `18-35-00_distill_hybrid` | 162.1% / 82.0% / 7.8 s | Robust-v0 80.1% / 100% | robust training costs a little officially |
| D5 | chase_live | Robust-v1 | `18-41-47_distill_chase_live` | 123.8% / 90.6% / 6.3 s | Robust-v1 43.4% / 100% | **parent for the main-line RL** |
| D6 | hybrid (asset unknown, confidence gate) | Robust-v1 | `18-45-23_distill_hybrid` | 172.3% / 77.3% / 7.8 s | Robust-v1 75.8% / 100% | best distilled |
| D7 | hybrid, **3× gradient steps** (1800) | Robust-v1 | `19-06-40_distill_hybrid` | 160.5% / 82.8% / 7.3 s | — | **no gain**: the copy-error plateau (~0.05) is missing information, not undertraining |

## RL fine-tuning (hosts' `scripts/skrl/train.py`, IPPO, 4096 envs)

| ID | Parent | Settings | Run | Result | Verdict |
|---|---|---|---|---|---|
| FT1 | D1 | hosts' defaults (lr 5e-4), 500 iterations | `18-22-08_ippo_torch_tello_defend_ippo` | best_agent 30.5% / 100%; it. 250 4.3%; final 0.4%. Reward −1.6 → −64; lr collapsed to 1.5e-6 | **collapse**. Too-high lr, critic poorly fitted through the shared trunk |
| FT2 | D5 | lr 5e-5, 300 iterations | `19-19-37_ippo_torch_tello_defend_ippo` | **it. 30: 185.4% / 74.2%** (256 ep), Robust-v1 82.0%; it. 60 97.7%; it. 90 50.0%; it. 120–300 → 10.9%. Reward 16.7 → **161** (it. 90) → −35.5; value loss 0.004 → 0.67. The reward before it. 75 is a start-up artifact (see "Training curves") | **RL helps when stopped early**, then collapses. skrl's reward-picked `best_agent` (it. 60) was the wrong pick |
| ~~first attempt of FT3 / FT2b / S-A/B~~ | — | launched in pairs at the same moment | folders renamed `*_INVALID_collision` | **invalid**: the hosts' `train.py` names run folders by launch second, so FT2b's two runs and the single-drone A/B's two runs wrote into one folder each, and both FT3 preps wrote one refit checkpoint. Results discarded. Fix: unique `agent.agent.experiment.experiment_name` per run, plus the pid in `distill.py`'s folder names | — |
| FT3-cl | D5 | hosts' recipe: critic refit through the trunk with the policy held fixed (EV 0.41), std 0.10, γ 0.997, lr 1e-5, rollouts 64, 32 minibatches, KL stop 0.01, value loss ×0.5, time-limit bootstrap, 80 it., checkpoint every 10 | `19-48-46_*_ft3_chase_live` (approx.) | it.10 142.6/85.9 → **it.40 172.7/75.8** → it.80 155.9/77.3 (128 ep) | **stable, no collapse**, but a lower peak than FT2 |
| FT3-hy | D6 | same as FT3-cl | `*_ippo_torch_ft3_hybrid` | 165.6/74.2 → 101.6/91.4 → **32.0/99.2** | RL **destroys the hybrid's behaviour** even when gentle: the hosts' reward punishes blocking (moving away from the nearest attacker). Reward misaligned with preventing breaches |
| FT2b-cl | D5 | FT2 recipe, seed 1 (ignored, see Determinism), 60 it., checkpoint every 5 | `19-48-37_ippo_torch_ft2b_chase_live` | it.5 101.6/100, it.15 177.7/78.4, **it.25 193.8/69.4** (64 ep; fresh 204.8/69.7), it.30 170.7/80.0, it.35–55 ~151–180 / 74–77, it.60 97.7/100 | **current best = it.25** |
| FT2b-hy | D6 | same | `19-48-40_ippo_torch_ft2b_hybrid` | 53–92% / 100% at every checkpoint | RL on the hybrid parent fails |
| S-A | D5 (defender_0 only) | **single drone**, hosts' reward, lr 5e-5, 60 it., checkpoint every 5 | `*_ippo_torch_single_official` | parent 43.8/100 → peak it.25 76.6/100 → ~50/100 | improves, then decays |
| S-B | D5 (defender_0 only) | **single drone, shaped reward** (potential progress on the same target, breach paid once, no proximity) | `*_ippo_torch_single_shaped` | → peak it.25 68.0/100 → ~56–58/100 | lower peak, slower decay. Inconclusive; breach is always 100% with 1 defender vs 2 attackers |
| FT-IRL | FT2b-cl it.25 | RL on **Team1-Defend-IRL-v0**, lr 5e-5, 40 it., checkpoint every 5; scored on IRL **and** official | `*_ippo_torch_ft_irl` | IRL (64 ep, same scenarios as parent's 51.6/100): it.15 62.5, it.20 60.2, **it.25 74.2**, it.30–40 55–59 (all 100% breach). Official 64 ep: it.15 211.0/61.6, it.20 212.1/64.3 → **fresh (100 envs × 300 ep): it.15 201.8/68.0, it.20 200.0/69.0** vs parent fresh 204.8/69.7 | improves IRL on paired scenarios; official gain **did not survive fresh confirmation** (64-episode selection bias). IRL scores vary a lot between scenario samples (same parent: 51.6 on 64 vs 86.3 on 128) |
| Rounds (**withdrawn 2026-10-24**: the user says don't assume 5-minute rounds; use the sim's 20 s episodes) | — | **5-minute rounds** (`scripts/defend_round.py`, 64 rounds, breaches / interceptions per round) | — | Official attackers: chase 43.0 / 6.4; **best 39.0 / 14.7**; distilled hybrid 40.8 / 10.6. Realistic attackers: chase 36.4 / 5.4; best 33.9 / 14.2; hybrid **32.6 / 15.8** | **2–3× more interceptions but only ~10% fewer breaches**: most waves still get one attacker through. The official 20 s metric hides this. Points to positioning (defenders out of place after far-out intercepts) |

## Key finding: defender crashes drive breaches (2026-10-23)

5-minute rounds (`defend_round.py`, official attackers): **defenders spend 86–97% of the round on the floor**: chase 86.0%, best checkpoint 87.6%, scripted hybrid 97.3%. They collide with attackers (or caught ones left hovering, or each other), and the hosts' body-frame thrust bug keeps a tilted drone from recovering. The earlier round and zone comparisons therefore measured crash survival, not defending.

**Collision avoidance in the hybrid teacher** (`distill.py --check`, official task, 128 episodes; internal counter):

| Teacher | Interceptions / breaches | Defender floor time |
|---|---|---|
| chase | 59.0% / 99.2% | 7.4% |
| hybrid, aims at the attacker's centre (D2/D6 teacher) | 157.0% / 99.2% | 37.6% |
| hybrid, `contact_offset 0.2` (aim 0.2 m beside the attacker: capture at < 0.35 m, contact at ~0.1 m) | 300.4% / 38.0% | 17.1% |
| hybrid, `contact_offset 0.2` + `obstacle_radius 0.6` (steer clear of caught attackers) | 366.8% / 15.6% | 3.6% |
| **hybrid, `contact_offset 0.25` + `obstacle_radius 0.6`** | **378.7% / 10.1%** | **0.4%** |

More teacher tuning: `contact_offset 0.3` + `obstacle_radius 0.6` gives 379.7 / 11.7; `0.25` + `0.8` gives 378.5 / 7.8; **`0.3` + `0.8` gives 380.9 / 5.5 (floor time 0.2%)**.

**D8: distilled `0.25` + `0.6` teacher** (`21-39-27_distill_hybrid_10928`) scores officially **151.2% / 82.0%**, fresh 154.0% / 82.7%: worse than the current best. Its copy error was 0.079, the worst yet. Most of the teacher's gain comes from steering clear of caught attackers, which the network cannot observe (zeroed slots), so those labels look like noise to it. **Sideways-aim-only hybrid teachers** (no clearance step): teacher 0.3 m gives 330.9 / 29.7 (floor 7.9%). Distilled 0.25 m (`21-44-41_distill_hybrid_15181`): fresh **165.3 / 77.0**, copy error 0.075. Distilled 0.3 m (`21-44-45_distill_hybrid_15717`): fresh **157.3 / 77.7**, copy error 0.078. Still worse than the best: the hybrid's *assignment* uses the teammate's position, which the network can't see.

**Decentralized pursuit teachers** (`--teacher pursuit`, built only from each defender's own 23 observations, so fully copyable; internal counter, 128 ep):

| Pursuit variant | Interceptions / breaches | Floor time |
|---|---|---|
| nearest live (= chase_live, sanity) | 122.7 / 92.2 | 15.1% |
| nearest live + 0.3 m beside | 105.9 / 93.8 | 9.3% |
| lead (collision course from the observed relative velocity) | 133.6 / 100 | 38.3% |
| **lead + 0.3 m beside** | **199.2 / 70.3** | 10.7% |
| lead + 0.25 m beside | 200.0 / 72.7 | 10.0% |

**D9:** distilled `pursuit --lead --contact_offset 0.3` (`21-54-13_distill_pursuit_22413`). Copy error 0.0125; official fresh **176.7 / 72.0** (64-ep sweep scenarios 169.9 / 69.9).

**FT-lead:** RL with the hosts' reward from D9 (FT2 recipe) **collapsed at once**: it.5 70.3/100 → it.40 14.1/100. That's the third behaviour (after blocking and the hybrid) that the hosts' reward un-learns. Their `closing` pays only for heading straight at the attacker's *current* position, so leading or aiming beside earns less per step, PPO removes them, and the defenders crash again.

**FT-shaped:** the hosts' `train.py` on **`Team1-Defend-Shaped-Crash-v0`**, which adds to the shaped reward (potential-based) **−20 once when a defender drops below 0.3 m**. lr 5e-5, 40 it., checkpoint every 5; official sweep at 64 ep (same scenarios as FT2b's sweep).
- From D9 (`ft_lead_shaped`): **collapsed** too (it.5 79.7/100 → it.40 18.8/100). The lead-pursuit network is fragile under PPO whatever the reward. Line dropped.
- From D5 (`2026-10-23_22-00-35_ippo_torch_ft_cl_shaped`): it.15 177.0/74.3, it.25 197.2/70.8, it.30 177.6/75.0, **it.40 195.3/66.7** (lowest breach on these scenarios; FT2b at it.40 was 158.4/75.3 and collapses by it.60). More stable than the hosts' reward in the 64-ep sweep, but **fresh confirmation brings it down**: it.25 **179.5/78.7**, it.40 **194.8/72.0** (100 envs × 300 ep).
- **Continued 40 more iterations** (`22-06-32_ippo_torch_ft_cl_shaped2`, it.45–80): unstable. 105–186% swings, then 100% breach at it.70/80. The best-looking checkpoint (it.60: 186.2/65.8 on 64 ep) is **160.8/73.0 fresh**. Line closed.

## State of play (2026-10-23, late)

- **Best official (learned) policy:** FT2b `agent_400` (RL iteration 25 on distilled chase_live, hosts' reward): **204.8% / 69.7% fresh**. Nothing has beaten it on fresh scenarios.
- **64-episode sweeps overstate gains by ~5–15 points of breach rate** (selection bias on a fixed scenario set). Always confirm fresh.
- **Biggest lever found:** collisions. Scripted hybrid with `contact_offset 0.3` + `obstacle_radius 0.8` reaches **380.9% / 5.5%** (internal counter, floor time 0.2%). But a network can't copy it: the official observation hides the teammate and caught attackers.
- **RL with the hosts' reward un-learns** leading, blocking and aiming beside. Our shaped + crash reward is gentler but also destabilizes after ~40–60 iterations.
- **Recommendation:**
  - **Sim score:** present FT2b `agent_400` (official), and show the collision-aware scripted controller with the same metric as the "what the observation space can't express" result.
  - **Real flights:** ~~fly the collision-aware hybrid; avoiding contact is a safety requirement~~. **Corrected 2026-10-24: the hosts want drones to collide in real flights; contact is the capture.** Fly the hybrid with `contact_offset 0` and `obstacle_radius 0` (aim at the attacker), keeping defender–defender separation. Episode length for real flights: **as in the sim** (user, 2026-10-24: don't assume 5-minute rounds).

## Training curves: what they can and can't show (2026-10-24)

- **The logged reward is meaningless for the first 75 iterations.**
  - All 4096 envs start their episode at step 0 (`episode_length_buf` zeroed, never randomized).
  - skrl's `Reward / Total reward (mean)` averages the last 100 *finished* episodes.
  - A full episode is 20 s = 1200 steps = 75 iterations at rollouts 16.
  - So before iteration 75 the only finished episodes are the ones that breached, and later-finishing ones have banked more reward. The curve rises mechanically.
  - **Every fine-tune we ran (≤ 60 iterations) sits inside that window**, including FT2b (15.7 → 282.7 while breach got worse), the shaped run (−47 → +64) and FT2's 51 → 264.6.
  - skrl's `best_agent` is picked from this same number, which is why it kept being the wrong pick.
  - Our checkpoint choices were always made with `evaluate.py` sweeps, so they are unaffected.
- **After iteration 75 (FT2 only), the reward and the official score fall together:**
  - reward 161 → −35, interception 50% → 11%, value loss 0.29 → 0.71 → 0.33.
  - That's PPO destabilizing (value loss through the shared trunk), not the policy gaming the reward.
  - The evidence that the reward is misaligned comes from `evaluate.py`: lead pursuit and blocking are un-learned. It does not come from the curves.
- **300 iterations is not too few.** Every fine-tune peaked at iteration 25–40 on `evaluate.py` and then got worse. FT1 ran 500 iterations and collapsed.
- **Never isolated: the discount.**
  - γ 0.99 at 60 Hz is a ~1.7 s horizon, while an attacker takes 3–30 s to reach the asset.
  - FT3 (γ 0.997) was the one stable run, but it changed six things at once.
  - Test: `outputs/ft_gamma.sh` changes only γ (PPO and `shaping_gamma`, which must match) relative to `ft_cl_shaped` (D5 parent, shaped + crash, lr 5e-5, 40 iterations). 64-episode sweeps:

    | γ | it. 15 | it. 25 | it. 30 | it. 40 | best |
    |---|---|---|---|---|---|
    | 0.99 (`ft_cl_shaped`, control) | 177.0/74.3 | 197.2/70.8 | 177.6/75.0 | 195.3/66.7 | 195.3/66.7 (fresh 194.8/72.0) |
    | 0.997 (`00-15-33_ft_g997`) | 122.7/100 | 180.0/73.3 | 155.2/77.9 | 152.0/80.3 | 180.0/73.3 |
    | 0.999 (`00-15-53_ft_g999`) | 172.9/80.6 | 167.8/76.3 | 173.7/76.3 | 109.4/100 | 173.7/76.3 |

    **Negative: a longer horizon is worse, not better.** The discount isn't what holds RL back. Line closed.
- **Events-only reward** (`outputs/ft_events.sh`: shaped + crash task with `progress_scale 0`, `alive 0`, `effort 0`, i.e. capture +30, cleared +30, breach −20 once, crash −20; D5, 40 it.):
  - γ 0.99 (`ft_ev99`): it.15 **200.0/68.9**, it.25 193.8/70.8, then 73–80% breach. **Fresh, it.15: 177.7/78.0.**
  - γ 0.999 (`ft_ev999`): best 184.3/75.7, then 100% from it.35.
  - Dropping the dense shaping neither helps nor hurts. Same peak-then-decline as every other reward. **The reward is not the main bottleneck.**

## Why PPO peaks early: the rollouts are phase-locked (2026-10-24)

- **Mechanism:**
  - All 4096 envs start the same episode at the same instant.
  - Rollouts are 16 steps (0.27 s at 60 Hz), so update *k* trains only on episode time 0.27·(k−1) … 0.27·k s.
  - No attacker can reach the asset before (6 − 1.5 m) / 0.9 m/s = **5.0 s**; the slowest arrives at 11.25 s.
  - So **updates 1–19 contain no breach and no finished episode**. skrl logs the first finished episode at **timestep ~324 (iteration 20) in every run**.
  - Breach data exists only around iterations 19–42.
  - Envs that don't breach all time out together at iteration 75 and re-synchronize.
- **Evidence:** official breach rate (64 ep) by iteration, all runs from D5:

  | run | it5 | it10 | it15 | it20 | it25 | it30 | it35 | it40 |
  |---|---|---|---|---|---|---|---|---|
  | FT2b | 100 | 100 | 78.4 | **100** | 69.4 | 80.0 | 76.7 | 75.3 |
  | shaped+crash | 100 | 100 | 74.3 | 81.3 | 70.8 | 75.0 | 77.2 | 66.7 |
  | γ 0.997 | 100 | 100 | 100 | **100** | 73.3 | 77.9 | 100 | 80.3 |
  | γ 0.999 | 100 | 100 | 80.6 | **100** | 76.3 | 76.3 | 100 | 100 |
  | events γ 0.99 | 100 | 74.4 | 68.9 | **100** | 70.8 | 73.3 | 78.5 | 79.7 |
  | events γ 0.999 | 100 | 100 | 74.4 | **100** | 75.7 | 75.7 | 100 | 100 |

  - The early updates (approach-phase data only) wreck the parent: D5 alone is 90.6%, and every run is at 100% by iteration 5.
  - The best checkpoints fall in the breach window (it. 15–30). Iteration 20, the last update before breaches enter the data, is 100% in 5 of 6 runs.
  - The 40-iteration fine-tunes never trained on the second half of an episode.
- **So "train longer" was never a fair test.** FT2 ran 300 iterations, but on phase-locked, then re-synchronizing data.
- **Fix: `Team1-Defend-Stagger-v0`** (`team1_defend_stagger_env.py`).
  - Each env's first episode starts at a random elapsed time, with attackers moved as far along their line as they'd have flown by then.
  - Later episodes are official; scenario, reward, observations and actions are AS-Defend-v0's.
  - skrl 2.1.1 IPPO ignores `learning_starts`, so a warm-up via config isn't possible.
  - Test: `outputs/ft_stagger.sh 300 ft_stagger`, the FT2 recipe for the full 300 iterations, sweep every 20.
- **Result (`2026-10-24_00-32-09_ippo_torch_ft_stagger`, 64-episode sweeps every 20 iterations):**

  | it. | 20 | 40 | 60 | 80 | 100 | 140 | 200 | 300 |
  |---|---|---|---|---|---|---|---|---|
  | interception / breach | 195.9/75.7 | **198.6/63.0** | 173.7/68.4 | 90.2/100 | 75.0/100 | 46.9/100 | 23.4/100 | 14.8/100 |

  - **Staggering fixed the early dips:** iteration 20 is 75.7%, not 100%. Iteration 40 is the lowest 64-episode breach of any PPO run.
  - **It still collapses after iteration 60.** Once the first episodes have left the data, skrl's reward falls (≈250 → 25) together with the official score.
  - The KL-adaptive learning rate stays within 1e-6–1e-4 (no blow-up). Value loss climbs 0.3 → 0.5.
  - **Verdict: more iterations don't help PPO here, even with stationary data.** The update direction is wrong (poor critic through the shared trunk), not the amount of training. Plot: `simulation_infra/outputs/plots/ppo_stagger_vs_ft2.png`.
  - Not yet confirmed fresh: it.40.
- **Budget of 300 iterations, for reference:**
  - 16 × 4096 = 65,536 env-steps per iteration. 300 iterations = 19.7 M env-steps ≈ 16,400 episodes, ~2 min.
  - That's 5× the data of the hosts' 512-env × 500-iteration recipe, but only 300 policy updates (6,000 minibatch steps).
  - The number of updates, not the amount of data, is what limits how far PPO can move.

## Evolution strategies (`scripts/skrl/es_train.py`, 2026-10-24)

- **Setup:**
  - OpenAI-ES on both defenders' policy paths (95,112 weights; the scaler is frozen), starting from FT2b `agent_400`.
  - 240 perturbed copies (antithetic pairs) × 16 envs, plus 256 envs on the unperturbed policy (the progress curve).
  - Fitness per episode: no breach (1/0) + 0.5 × fraction of the episode held + 0.05 × interceptions, which is what `evaluate.py` counts. Centred ranks, Adam lr 0.001.
  - Parity with skrl's mean action: 4.3e-6.
  - σ calibration (RMS action change): 0.003 → 0.07, 0.005 → 0.12, 0.01 → 0.24, 0.02 → 0.44. **σ 0.005** chosen.
- **Run `2026-10-24_00-28-48_es_ft2b_s005`** (100 generations, about 60 s each):
  - Generation 1, unperturbed policy: 66.4% breach / 214.6% intercept (256 episodes).
  - Checkpoints every 5 generations, in skrl format for `evaluate.py` and `eval_sweep.sh`.

## Checks and diagnostics (internal, not official scores)

| What | Result |
|---|---|
| `distill.py --check chase` (our outcome counter) | 59.0% / 99.2% / 3.8 s, close to official 55.5% / 100% / 3.1 s (seed differs) |
| `--check hybrid`, official task | 157.0% / 99.2% / 5.5 s; constraints all ok (actions in [-1,1], yaw 0, under speed cap, finite) |
| `--check chase_live` | 122.7% / 92.2% / 6.3 s; teammates < 0.3 m in 41% of episodes (collision risk) |
| `--check hybrid`, Robust-v0 | 85.9% / 100%; asset estimate 0.96 m mean error (before the confidence gate) |
| Asset estimator, in env, after confidence gate | median error 1.71 m (start = prior) → 0.32 m at 2–4 s → 0.10 m at 4–8 s; 90th percentile stays at the prior when inference is impossible |
| Action variation (D6, official) | mean speed 0.83 m/s, per-axis std 0.5, yaw 0, two defenders act differently (0.67); env re-randomizes every episode |
| RL training throughput | 4096 envs ≈ 0.38 s per iteration (rollouts 16) |
| Weave robustness (official reward, `attacker_evasion` 0.5 / 1.0) | chase 56.2/100 and 59.8/100; best (FT2 it.30) 185.2/71.1 and **231.4/39.4**. Weaving slows attackers, so no overfit to straight paths |
| Attackers on the ground? (64 envs, ~320 single-wave episodes) | **never while alive**. Caught attackers fall (72 with chase defenders, 2 with hovering ones) because of the hosts' body-frame thrust bug. Breach rates are not inflated by knocked-down attackers |
| IRL task (`evaluate.py`, 128 ep) | chase 40.2/100; best (trained on perfect state) 86.3/94.5; hybrid `--check` 115.2/97.7 with asset estimate 1.67 m mean error |

## Team1 task variants (new files under `tasks/direct/team1_defend/`; scored with the hosts' `evaluate.py --task`)

| Task | What changes from `AS-Defend-v0` |
|---|---|
| `Team1-Defend-Robust-v0` / `-v1` | attackers spawn 3–6 m away, 0.2 s command latency, asset position **inferred** from the attackers' flight lines (v1: inference off in 30% of episodes) |
| `Team1-Defend-Single-v0` / `-Single-Shaped-v0` | 1 defender vs 2 attackers (same 23-wide observation); hosts' or shaped reward |
| `Team1-Defend-Shaped-v0` | official scenario, shaped reward |
| `Team1-Defend-Weave05-v0` / `-Weave10-v0` | official, attackers weave at evasion 0.5 / 1.0 (robustness tests) |
| `Team1-Defend-Realistic-v0` | **realistic attackers** (`team1_attackers.py`): 70% weave (random strength 0.2–1.0, period 2–6 s, phase), all get a slow heading wobble (~9°), speeds **0.2–1.0 m/s**, never below 0.7 m while alive |
| `Team1-Defend-Robust-Realistic-v0` | Robust-v1 + realistic attackers |
| **`Team1-Defend-IRL-v0`** | Robust-Realistic + **positions-only sensing** (`team1_sensing.py`): every drone's position as 15 Hz fixes with 4 cm noise and 10% dropouts; **all velocities from the deployed `KalmanCV`**, also feeding the asset estimator. No tags on attackers, no reported speed |
| `Team1-Defend-Stagger-v0` | official scenario and reward; each env's **first** episode starts at a random elapsed time with attackers pre-flown, so PPO rollouts aren't phase-locked (`team1_defend_stagger_env.py`) |
| **`Team1-Defend-Fitted-v0`** (+ `-Fitted-Tau03-v0`, `-Fitted-Tau015-v0`) | IRL task plus the real command path at the hosts' priors (`team1_defend_fitted_env.py`; replace with fitted values once the step-response flight exists). τ 0.5 s, 0.75 m/s at rc 100, 0.1 s command delay, 0.2 s fix age before the Kalman filter, rc at 13.5 Hz + 16 ms jitter, integer rc with deadband 5, 0–15% rc loss per episode. One task ID per τ, because `evaluate.py` takes no overrides |
| `Team1-Defend-Round-Official-v0` / `-Round-Realistic-v0` | **5-minute rounds**: a breach is counted and its wave replaced; only time ends the episode. Scored with `scripts/defend_round.py` (breaches and interceptions per round) |

## Bugs and issues found (hosts' files untouched)

- **Video:** `scripts/record_clip.py` ticks the app with bare `simulation_app.update()`, which steps physics without the drones' thrust, so the footage is wrong. Our `scripts/record_defend.py` uses `sim.render()` ticks (as the hosts' `record_approach.py` does).
- **Reward:**
  - `closing` jumps about −5 whenever the nearest target changes
  - `_prev_distance` resets to 6.0 on a new wave
  - a breach is paid twice (`_get_dones` runs before `_get_rewards`)
  - with γ 0.99 at 60 Hz the horizon is ~1.7 s
- **Shared network:** policy and value share a trunk (`models.separate: False`). A value head alone is a linear probe (explained variance 0.2–0.5), and in PPO the value loss pulls on the policy's features.
- **`bridge/tello_bridge.py` `to_rc`:** doesn't negate `vy` or yaw for the SDK.
- **`evaluate.py`** writes empty TensorBoard files into the tracked `simulation_infra/tello_defend/tello_defend_ippo/` (13+ untracked files).

- **Thrust frame:** the hosts' env applies the controller's world-frame thrust as a **body-frame** force (`set_external_force_and_torque` defaults to `is_global=False`). Harmless when level and yaw 0; once a collision tilts a drone it can't recover.
- **The hosts' `train.py` ignores `--seed`**: runs are deterministic.

## Not tried yet

- **Shaped reward in 2 vs 2** (Team1-Defend-Shaped-v0). Now well motivated: FT3-hy shows the hosts' reward un-learns blocking.
- **Kalman filter tuning:** for real flights, on logged flights; the IRL task already uses the deployed filter as-is.
- **MAPPO** (shared critic sees both defenders): the checkpoint format differs from IPPO's.
- **Curriculum** (README: capture 1.0 m, spawn 4 m, slow attackers, then step back).
- **Residual policy:** frozen distilled base plus a zero-initialized correction (hosts' SDS lines 810–817).
- **Tuning the hybrid teacher's settings** (guard radius, hysteresis, separation) before re-distilling.
- **Entropy bonus** (PM: 0.005, "raise if the policy collapses to hovering").
- **AprilTag-like observation noise variant (v2):** 3 cm per fix, ~10 Hz fixes with hold, freezes.
- **Real-flight variant:** 3 defenders vs 2 attackers, defenders 0.5 m/s, attackers 0.2–0.45 m/s.
- **Self-play / adversarial attacker pool.**
- **Separate actor and critic networks:** ruled out, because `evaluate.py` builds the shared network from the hosts' config.

## ES checkpoints, fresh (hosts' `evaluate.py`, 100 envs × 300 episodes)

| ES run `00-28-48_es_ft2b_s005` | interception | breach |
|---|---|---|
| start (FT2b agent_400) | 204.8% | 69.7% |
| gen 25 | 215.3% | **62.7%** |
| gen 40 | 209.7% | 67.3% |

- **`evaluate.py` over-counts early breaches.** It stops after N *finished* episodes, and breached episodes finish sooner. So its breach rate sits above the true per-episode probability; ES's own unbiased curve is about 58% at gen 25–50.
- To read 20% on `evaluate.py`, the true breach rate must be lower, roughly 15%.

## Contact is the capture (user, 2026-10-24): teachers with `contact_offset 0`

User rules: collide with the attacker (an attacker tipped over is a capture), but a defender on the ground is a penalty, and the offset stays at the code default (0).

`distill.py --check`, AS-Defend-v0, 256 episodes, counted like `evaluate.py`; not official:

| teacher | interception | breach | defender floor time |
|---|---|---|---|
| chase_live | 134.6% | 90.2% | 17.6% |
| pursuit, lead | 138.7% | 99.2% | 39.7% |
| **assigned**, hard contact | 155.3% | 98.8% | 36.8% |
| **assigned + soft contact 0.3 m/s** | 369.5% | **7.8%** | 1.8% |
| **assigned + soft contact 0.15 m/s** | 373.2% | **4.7%** | 1.5% |

- **`assigned`** (new, `distill.py`) uses only each defender's own 23 observations, so a network can copy it:
  - defender *i* takes attacker slot *i* while it's alive, then helps on the other;
  - lead intercept at full speed;
  - **soft contact**: within 0.7 m it flies the attacker's own velocity plus a slow closing speed, so contact is a gentle bump.
- **A full-speed hit tips the defender over, and in this sim it can't recover** (hosts' thrust-frame bug). A gentle bump downs the attacker and keeps the defender flying. That one change takes breaches from 99% to 5%.
- Teammates still come within 0.3 m in ~28% of episodes, but crashes stay rare.
- Next: distil it into the IPPO network (`distill.py --teacher assigned --soft_closing 0.15`), score with `evaluate.py`, then ES-fine-tune.

## Smoothness check (`scripts/smoothness_check.py`, 128 episodes each; `outputs/smooth/sweep.txt`)

| controller | Fitted τ 0.5: breach / intercept / floor | jumps > 0.5 m/s per s | reversals per s |
|---|---|---|---|
| FT2b agent_400 | 100 / 23.8 / 1.3% | 0.46 | 0.50 |
| D9 pursuit-lead | 100 / 17.6 / 0.4% | 0.39 | 0.39 |
| hybrid, Kalman state, offset 0 | 98.4 / 59.8 / 3.4% | 0.57 | 0.49 |
| chase | 100 / 14.1 / 0.3% | 0.45 | 0.27 |

- τ 0.3 and τ 0.15 give the same picture: about 100% breach for every controller. **Lag isn't what breaks Fitted.** The rest of the realistic setup is:
  - 0.75 m/s top speed for defenders and attackers alike;
  - spawns 3–6 m out;
  - 0.3 s of delay;
  - the asset position inferred, not known.
- Command streams are not jittery: 3.5–7.6% of changed packets jump more than 0.5 m/s; reversals run 0.3–0.7 per second. Smoothness is not the blocker for flying.
- On the official task, agent_400 sends 0.7% jumps and 0.50 reversals per s, with 10.4% floor time.

## Distilled assigned + soft contact: GOAL MET (2026-10-24)

- **Run:** `distill.py --task AS-Defend-v0 --teacher assigned --contact_offset 0 --soft_closing 0.15`, 12 DAgger rounds → `2026-10-24_01-15-18_distill_assigned_147307`.
- **Copy:** student-vs-teacher mse 0.0033; critic explained variance 0.86 (it was 0.2–0.5 before).
- **Official `evaluate.py` (hosts' code, unchanged):**
  - fresh 100 envs × 300 ep: **362.0% interception / 6.3% breach**;
  - 64 × 128: 367.6% / 7.0% / 10.8 s.
- **Versus before:** FT2b agent_400, 204.8% / 69.7%, and chase, 55.5% / 100%.
- The ES run from FT2b was stopped at gen 65, made obsolete by this result. Checkpoints agent_5 … agent_65 are kept.

## Tuning the assigned + soft-contact teacher (`distill.py --check assigned --contact_offset 0`, 256 ep)

| variant | interception | breach | floor |
|---|---|---|---|
| soft 0.15 (default terminal 0.7 m) | 373.2% | 4.7% | 1.5% |
| soft 0.10 | 370.0% | 5.4% | 1.7% |
| **soft 0.05** | 371.3% | **3.5%** | 1.5% |
| soft 0.15, terminal 1.0 m | 312.5% | 6.6% | 1.4% |
| soft 0.15, terminal 0.5 m | 241.4% | **63.7%** | 22.6% |
| soft 0.15 + `--block 2.2` (wait on the line when late) | 367.0% | 5.9% | 1.6% |

- The terminal range must be ≥ 0.7 m. At 0.5 m the defender can't slow down in time (lag), hits hard and falls.
- Blocking doesn't help. The option stays in the code, off by default.
- Soft 0.05–0.15 are within noise (±1.4 points at 256 ep).

**Across Tello attacker speeds** (soft 0.15; new tasks `Team1-Defend-Speed{010,025,050,075,100}-v0` and `-WideSpeed-v0`, on the hosts' env class):

| attacker speed | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 | 0.1–1.0 |
|---|---|---|---|---|---|---|
| breach | 0.0% | 0.8% | 2.0% | 3.5% | **34.3%** | 19.7% |
| with `--block 2.2` | | | | | 42.6% | 18.3% |

- Attackers at 1.0 m/s are as fast as the defenders, so once behind, chasing can't close the gap.

**On the realistic tasks** (soft 0.15): Fitted 60.2% / 96.1% (floor 5.9%); IRL 126.2% / 91.4% (floor 12.1%).

- Peer session's feasibility bound: even an oracle pairing can stop only 71% of Fitted first waves (85% on IRL and Robust). The 3 m spawns leave 1.5 m to breach, and on Fitted the attackers share the 0.75 m/s cap.
- Ablation tasks `Team1-Defend-Fitted-Spawn6-v0` and `-Fitted-Slow-v0` (attackers 0.2–0.5 m/s) are registered.

**ES from the distilled network** (`01-25-32_es_assigned_s003`): σ 0.003, lr 0.0005, 120 copies × 32 envs, `w_capture 0.01`, `w_floor 0.5`.

- The unperturbed policy stays at 4–6% breach (5-gen average; per-gen noise ±1.4).
- Interception rose 360% → 387% over 26 generations.

## ES from the distilled assigned network: 2.3% (2026-10-24)

Official `evaluate.py`, fresh, 100 envs × 300 ep:

| checkpoint (`01-25-32_es_assigned_s003`) | interception | breach |
|---|---|---|
| start: distilled soft 0.15 | 362.0% | 6.3% |
| **ES gen 15** | **383.7%** | **2.3%** |
| ES gen 25 | 389.0% | 3.3% |

- 300 episodes put a 95% CI of about ±1.7 points on 2.3%. A larger fresh confirmation (600 ep) is queued for the final pick.

Other runs:
- Distilled soft 0.05 (official speeds): 363.8% / 6.3%. Distilled soft 0.05 on WideSpeed (0.1–1.0 m/s): 358.0% / 7.7% official.
- Teacher, 512 ep: soft 0.05 → 371.3% / 4.1% (floor 1.7%).
- `--box_speed` (per-axis box, up to 1.41 m/s): official 12.3%, 1.0 m/s attackers 81.2%, wide 18.5%. **Worse**: too fast to slow down for soft contact.
- `--block 2.5` / `3.0`: official 7.8% / 11.3%; 1.0 m/s 46.9% / 48.0%; wide 19.1% / 19.0%. **Worse.**
- Fitted ablations (soft 0.15): 6 m spawns → 75.8% breach (floor 12.5%); attackers 0.2–0.5 m/s → 55.6% (floor 16.9%). The rest of the gap is sensing, latency and the inferred asset.

### ES checkpoints: 600-episode fresh confirmations, and across Tello speeds

| ES `01-25-32_es_assigned_s003` | official (100 envs × 600 ep) |
|---|---|
| gen 10 | 381.0% / 2.5% |
| gen 15 | 385.4% / 2.8% |
| gen 20 | 383.3% / 4.3% |
| gen 35 (300 ep) | 395.5% / 2.0% |

ES plateaus around 2–4% (`evaluate.py`), and its own curve (unperturbed policy, 256 ep/gen) runs 1–6%.

ES gen 35 on the speed tasks (hosts' `evaluate.py --task`, 300 ep):

| attacker speed | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 | 0.1–1.0 |
|---|---|---|---|---|---|---|
| interception / breach | 249.2/0.0 | 293.8/0.3 | 357.5/1.7 | 419.0/1.7 | 479.9/**7.6** | 335.7/11.7 |

- At 1.0 m/s the ES network is far better than its teacher (7.6% vs 34.3%).
- Mixed speeds (0.1–1.0) are hardest: a slow attacker keeps the wave open while a fast one comes in.

Urgency closing, 512 ep (soft 0.05 baseline 4.1%):

| `--urgent_eta` / `--urgent_closing` | breach |
|---|---|
| 2.0 s / 0.5 m/s | 5.3% |
| 3.0 s / 0.5 m/s | 6.1% |
| 2.0 s / 0.3 m/s | 6.0% |
| 2.0 s / 0.8 m/s | 8.7% |

All worse; dropped.

### Breach autopsy (`scripts/breach_autopsy.py`, ES gen 35, 1,538 ep, 3.7% breach)

- **91% of breaches come after the breaching attacker's paired defender is already on the floor.** The defenders can't see each other, so the survivor doesn't take over, and the next waves come 2-against-1.
- Breaches fall mostly in waves 2–3, at 10–20 s, from fast attackers (0.8–0.9 m/s: 26 of 57).
- Crashes (146 in 1,538 ep), by what was within 0.45 m in the 1.5 s before the fall:

  | within 0.45 m before the fall | share of crashes |
  |---|---|
  | **teammate** (alone 39%) | **76%** |
  | a downed attacker still hovering | 55% |
  | the live attacker | 43% |

- **Next lever: keep the two defenders apart.**
  - New teacher options: `--escort UP AHEAD ENGAGE_ETA` (a free defender escorts above and ahead of its teammate's attacker instead of joining in) and `--cruise_up` (approach from above, level only for the soft contact).
- New video: `simulation_infra/outputs/videos/defend_es35.mp4` (45 s: 17 interceptions, 1 breach); legend clipping fixed in `record_defend.py`.

### Keeping the defenders apart (teacher, soft 0.05, 512 ep; baseline 371.3% / 4.1%, teammates < 0.3 m in ~28% of episodes)

| variant | interception | breach | teammates < 0.3 m | floor |
|---|---|---|---|---|
| `--escort 0.6 1.0 2.5` | 370.8% | 4.5% | 20.3% | 1.3% |
| `--escort 0.6 0.5 2.0` | 368.1% | 4.1% | 16.0% | 1.3% |
| `--cruise_up 0.5` | 64.8% | **90.0%** | 21.1% | 0.6% |
| `--cruise_up 0.5 --escort 0.6 1.0 2.5` | 69.9% | 89.1% | 22.1% | 0.4% |

- Escort cuts close passes but not breaches.
- Cruising above breaks the capture: the 0.5 m vertical gap is closed at the 0.05 m/s soft speed, too slowly.

**ES plateau**, 10-generation means of the unperturbed policy:
- breach: 4.5, 5.1, 4.2, 4.0, 4.4, 4.1, 4.8, 3.9, 4.1%;
- floor: 1.6–1.9%;
- perturbed population: 20 → 15.5%, i.e. a flatter optimum, not a better one.

ES has saturated at about 4% true breach. The remaining failures are teammate collisions the observation can't see. Next: a CRN ES restart (`--crn`), and the final pick on 1,000 fresh episodes.

### ES gen 35 on the realistic variants (`evaluate.py --task`, 64 × 128)

| task | interception | breach |
|---|---|---|
| Team1-Defend-Realistic-v0 (weaving, 0.2–1.0 m/s, true state) | 355.1% | **1.6%** |
| Team1-Defend-IRL-v0 | 176.6% | 87.5% |
| Team1-Defend-Fitted-v0 | 30.9% | 100% |
| Team1-Defend-Fitted-Spawn6-v0 | 80.9% | 81.2% |
| Team1-Defend-Fitted-Slow-v0 (0.2–0.5 m/s) | 188.7% | 32.1% |

- Robust to attacker motion (weave, wobble, speeds).
- Not robust to the room's sensing and latency, or to the inferred asset. That is the open sim-to-real gap.
- The peer session's oracle bound says much of the Fitted gap is scenario harshness (3 m spawns, attackers no slower than the defenders).

### Final pick: 1,000 fresh episodes each (hosts' `evaluate.py`, 100 envs)

| checkpoint | interception | breach |
|---|---|---|
| distilled assigned soft 0.15 (`01-15-18_distill_assigned_147307`) | 365.1% | 5.7% |
| ES gen 10 (`01-25-32_es_assigned_s003/checkpoints/agent_10.pt`) | 380.4% | **3.1%** |
| ES gen 35 (`…/agent_35.pt`) | 390.5% | **3.2%** |
| ES gen 60 / 100 | pending | |

### Crash autopsy, corrected (peer review)

- A Tello's collision box is ~0.1 m, so bodies touch only within ~0.13 m. The first version (0.45 m / 1.5 s window) was ~3× too loose.
- At the floor (< 0.3 m) the drone is already **upside down**: tilt median 167°, vz −7.8 m/s, and 93% have no contact in the last 0.3 s. A tipped drone's thrust (hosts' body-frame bug) drives it down, so the floor is the wrong moment to look.
- **At the flip** (tilt first passes 30°), closest approach < 0.15 m within the previous 0.3 s, over 152 flips:

  | cause | share of flips |
  |---|---|
  | **teammate contact** | **48%** |
  | a just-downed attacker (likely the capture bump: it stops, the defender keeps its matched speed) | 28% |
  | no contact | 24% |

- New teacher option `--helper_up` (a free defender works above attacker level) is being tested.

### Splitting the realistic-task gap (ES gen 35, `evaluate.py --task`, 64 × 256)

| task | what it adds over official | interception / breach |
|---|---|---|
| Realistic | weaving attackers, 0.2–1.0 m/s, true state | 355.1 / **1.6** |
| Robust-Realistic | + 3–6 m spawns, 0.2 s latency, inferred asset | 197.7 / **82.4** |
| IRL | + Kalman sensing | 176.6 / 87.5 |
| Fitted τ 0.15 | + rc path, 0.75 m/s cap | 103.7 / 89.5 |

- **The jump is from geometry, latency and the inferred asset, not from sensing noise** (+5 points).
- The network was trained with the true asset position in its observation.

| ES gen 60 / 100 (1,000 ep) | 389.2% / 5.0% · 388.7% / 3.5% |
|---|---|

**Final pick: ES gen 35**, 390.5% / 3.2% (95% CI about ±1.1). Gen 10 (3.1%) is statistically tied; gen 35 has more interceptions.

### Last teacher and ES attempts

- Helper lane (`--helper_up`, soft 0.05, 512 ep; baseline 4.1%):

  | `--helper_up` | breach | teammates < 0.3 m | floor |
  |---|---|---|---|
  | 0.2 m | 3.7% | 27.5% | 1.2% |
  | 0.3 m | 3.5% | 27.1% | 1.3% |

  Within noise.
- **CRN ES restart** (`--crn`, from gen 35, 60 gens, `2026-10-24_03-22-37_es_es35_crn`): unperturbed-policy breach, 10-gen means 3.6, 3.7, 4.3, 4.7, 4.2, 4.4%. No gain over the plateau.

**Plateau: about 3–4% true breach.** Half the remaining flips are teammate contacts, which the 23-number observation can't see.

### Robust-Realistic split (ES gen 35, `evaluate.py --task`, 64 × 256; new `team1_defend_ablation_env.py`)

| Robust-Realistic (82.4%) minus one change | interception | breach |
|---|---|---|
| (a) official 6 m spawns (`Team1-Defend-RR-Spawn6-v0`) | 265.6% | **32.3%** |
| (b) no command delay (`-RR-NoDelay-v0`) | 282.4% | 62.5% |
| (c) true asset in obs (`-RR-TrueSite-v0`) | 200.2% | 84.4% |

- **Close spawns are the main driver, then latency. The inferred asset doesn't matter.**
- The peer session owns the laptop-side latency predictor and the teammate separation filter.

### Deploy-side layer: latency predictor + teammate separation filter (ES gen 35, `evaluate.py --task`, 64 × 64 unless noted)
The policy, its observation and its action are unchanged. The shared math is the laptop's file `drone_deployment_infra/core/defend_deploy.py` (numpy or torch; `python3 drone_deployment_infra/core/defend_deploy.py` gives SELFTEST PASSED). The sim loads it by path in `team1_defend/team1_defend_deploy_env.py`.
- **Predictor:** the network sees the observation moved forward by the command delay (+ fix age on Fitted). Mode `cv` assumes constant velocity; mode `model` replays the commands still in flight through the velocity lag.
- **Filter:** a second-order CBF on defender-defender distance (d_min 0.3 m). It acts on the world-frame velocity commands and uses the states at the moment the command takes effect. It never avoids attackers, because contact is the capture.

The baseline and the variants run at the same `--num_envs`, so they see the same scenarios: a paired comparison. Flips and teammate distances come from `scripts/deploy_diag.py`.

| Task | Interceptions / breaches | Flips / episode | Defenders < 0.3 m (steps) |
|---|---|---|---|
| Robust-Realistic (baseline) | 205.8% / 77.9% | 1.14 | 1.43% |
| RR-PredictCV | 259.2% / **55.3%** | **0.22** | 2.23% |
| RR-PredictModel | 241.5% / 62.2% | | |
| RR-Sep (filter alone) | 143.8% / **100%** | 1.17 | 0.05% |
| RR-PredictCVSep | **272.2% / 52.8%** | **0.06** | **0.00%** |
| AS-Defend-v0, 64 × 128 | 380.5% / 4.7% | 0.10 | 0.99% |
| Official-Sep, 64 × 128 | 387.1% / 3.1% | 0.06 | 0.01% |
| Fitted / Fitted-PredictCV / Fitted-PredictCVSep | 22.7 / 31.2 / 28.9% captures, all 100% breach | 0.09 / 0.16 / 0.11 | 0.72 / 0.85 / 0.12% |
| FittedT024 / -PredictCV / -PredictCVSep | 43.0 / 39.1 / 39.8% captures, all 100% breach | | |

Verdict:
- The 0.2 s command delay is what flips the defenders on Robust-Realistic. The predictor removes about 80% of the flips, and breach drops from 77.9% to 55.3%, better than RR-NoDelay's 62.5% (that one at 64 × 256).
- The filter must run together with the predictor. Alone, inside a delayed loop, it costs about a third of the captures and every episode breaches. The reason is not understood yet.
- With the predictor it is neutral or slightly positive, and it removes defender pairs closer than 0.3 m. On the official geometry it does no harm (4 vs 6 breaches in 128 episodes: noise).
- **Fitted's tau is about 2× too slow.** `TelloCommand` lags twice by `response_time` (63% rise in about 2.1 × response_time). The hosts' prior is a single lag (`rc_action.py`: kp_vel = 1/tau). So Fitted's `response_time=0.5` gives a 63% rise in 1.07 s, against 0.5 s for the prior; `response_time=0.235` matches it (FittedT024).
- Even so, Fitted breaches 100% and the predictor doesn't help, so Fitted's failure is not latency or tau. Untested suspects: speed parity (attackers capped at the defenders' 0.75 m/s) and the rc path.
- `deploy_diag.py`'s "filter changed" share is meaningless on Fitted, where the rc rounding and hold change every command.
- Next: 64 × 256 confirmations (peer). Untried: weighting the filter's correction toward the defender farther from its attacker.

**Correction (same day): the 64 × 64 table above is biased; don't use it.** `evaluate.py` counts the first N episodes to finish across all envs (`episodes += len(finished)`). A breached env restarts and can breach again, counting several times, while a 20 s survivor counts once. So when `--episodes` ≲ `--num_envs`, the breach rate is pushed toward 100%. Example: RR-NoDelay is 100% at 64 × 64 and 62.5% at 64 × 256. Use at least 10 × num_envs episodes for comparisons; the hosts' default of 64 envs × 20 episodes is extreme. Clean re-runs at 64 × 640: `simulation_infra/outputs/deploy_64x640_summary.txt`.

### Two measurement caveats (peer review, 2026-10-24)

- **`evaluate.py` over-counts breaches when `--episodes` is small relative to `--num_envs`.**
  - It stops after N *finished* episodes. A breached env restarts and can breach again, so it counts several times, while a 20 s survivor counts once.
  - Example, ES gen 35 on RR-NoDelay: 64 × 64 → 100%, 64 × 256 → 62.5%.
  - **Our headline numbers (100 × 300, 100 × 600, 100 × 1000) are fine.**
  - The 64 × 128 / 64 × 256 realistic-task numbers above overstate breaches somewhat. The peer session is re-running them at 64 × 640.
  - The hosts' own default, 64 envs × 20 episodes, is the extreme case.
- **Fitted's τ is ~2× slower than intended.**
  - The hosts' TelloVelocityController applies `response_time` twice (target lag, then `accel = error / τ`), so 63% rise ≈ 2.15 τ. Fitted at τ 0.5 behaves like ~1.07 s; official τ 0.15 is ~0.32 s.
  - The hosts' single-lag prior (0.5 s) corresponds to `response_time` 0.235: `Team1-Defend-FittedT024-v0` (peer file `team1_defend_deploy_env.py`).
  - Peer early result (biased 64 × 64): FittedT024 still ~100% breach, and its latency predictor doesn't help. So Fitted's failure is mostly the 3 m spawns and the speed parity (attackers capped at the defenders' 0.75 m/s), not delay or τ.

### Deploy-side layer, clean numbers (ES gen 35, `evaluate.py --task`, 64 envs × 640 episodes; supersedes the 64 × 64 table and its verdict)
Same `--num_envs`, so the scenarios are paired. At 640 episodes the binomial SE is about ±2 points at 55% breach and about ±0.8 at 4%.

| Task | Interceptions / breaches / time to intercept |
|---|---|
| Robust-Realistic (baseline) | 197.2% / 80.2% / 7.6 s |
| RR-NoDelay (the target: what zero delay gives) | 295.7% / 58.9% / 9.1 s |
| RR-PredictCV | 259.8% / 57.3% / 9.4 s |
| RR-PredictModel | 274.1% / 56.7% / 9.3 s |
| RR-Sep (filter alone) | 199.3% / 80.0% / 7.6 s |
| **RR-PredictCVSep** | **280.3% / 52.0%** / 9.6 s |
| AS-Defend-v0 | 385.2% / 4.1% / 10.7 s |
| Official-Sep | 390.4% / 2.5% / 10.7 s |
| Fitted | 52.0% / 95.8% / 7.3 s |
| FittedT024 (tau matched to the hosts' single-lag prior) | 106.1% / 88.1% / 8.0 s |
| FittedT024-PredictCVSep | 87.9% / 89.5% / 8.0 s |

Verdict:
- **The predictor recovers the whole delay gap.** RR breach goes from 80.2% to 57.3% (CV) or 56.7% (model), against 58.9% with zero delay. CV and model are equal, so ship CV: it has no tau to get wrong.
- **The filter is neutral alone** (80.0% vs 80.2%; the 100% in the 64 × 64 table was the counting bias). It gives about −5 points with the predictor (52.0% vs 57.3%, about 1.3 sigma), and −1.6 points on the official geometry (16 vs 26 breaches in 640 episodes, p ≈ 0.1). Both lean the same way: a small gain, no harm, and no defender pairs within 0.3 m.
- **The tau correction matters on Fitted:** captures double, 52% → 106%, and breach drops 95.8% → 88.1%. The deploy layer doesn't help there (89.5%). What's left of Fitted's gap is the 3 m spawns, speed parity (attackers capped at the defenders' 0.75 m/s) and sensing, not delay.
- None of this changes the official score (the layer isn't in AS-Defend-v0). It is what we fly: CV predictor with horizon = measured command delay + fix age, plus filter (d_min 0.3 m, lam = 0.9/tau), always used together.

### Clean re-runs at 64 × 640 (ES gen 35; supersedes the 64 × 128 / 64 × 256 rows above, which stay for the record)

| task | 64 × 128/256 (biased) | **64 × 640** interception / breach |
|---|---|---|
| Team1-Defend-Realistic-v0 | 355.1 / 1.6 | **342.8 / 6.4** |
| Team1-Defend-IRL-v0 | 176.6 / 87.5 | 188.4 / 79.1 |
| Team1-Defend-Fitted-Spawn6-v0 | 80.9 / 81.2 | 90.1 / 80.0 |
| Team1-Defend-Fitted-Slow-v0 | 188.7 / 32.1 | 190.7 / 32.7 |
| Team1-Defend-Fitted-Tau015-v0 | 103.7 / 89.5 | 113.5 / 88.0 |
| Team1-Defend-RR-Spawn6-v0 | 265.6 / 32.3 | 259.7 / 34.7 |
| Team1-Defend-RR-TrueSite-v0 | 200.2 / 84.4 | 205.3 / 81.6 |

The conclusions stand: spawn distance is the main driver, then latency (peer: RR 80.2% → RR-NoDelay 58.9%), and the inferred asset doesn't matter. Weaving attackers cost about 2–3 points over the official task (3.2–4.1%).

### Hosts' `evaluate.py` at its defaults (2026-10-24)

The user asked for the hosts' defaults: `python -u scripts/evaluate.py --checkpoint policies/team1_defend_es35/agent_35.pt` (AS-Defend-v0, 64 envs, 20 episodes, IPPO).

- ES gen 35: **381.2% interception / 4.7% breach** (3 of 64 episodes), 10.7 s. Three runs gave the same numbers: the defaults are seeded.
- At the defaults the loop stops at the first step with ≥ 20 finished episodes. A good policy finishes all 64 at the 20 s timeout, so this is 64 episodes (95% CI about ±5 points). A policy breaching in more than 20 envs before 20 s gets only its breaches counted (why chase reads 100%).
- Submission copy: `simulation_infra/policies/team1_defend_es35/agent_35.pt` (byte-identical, with a README of how it was made).

### Closer spawns, policy only (new `team1_defend_spawn_env.py`; official everything else)

The user can hand over only the network, and expects real attackers to start closer than 6 m (the room is about 4.5 × 3.4 m). New tasks `Team1-Defend-Spawn{25,3,4,5,6}-v0` (fixed radius) and `Team1-Defend-Close-v0` (3–6 m) change only the spawn radius; no delay, true asset, official speeds.

- Sanity: `Spawn6` at the defaults gives 389.8% / 1.6% (official 381.2% / 4.7%: same task, different random draws).
- ES gen 35 on Close-v0 (ES generation 1, 256 first episodes): **57.4% breach**.
- ~~ES fine-tune from gen 35 on Close-v0 (same recipe as `es_assigned_s003`, 60 generations): `2026-10-24_06-33-03_es_close_from35`, running.~~
- ES fine-tune from gen 35 on Close-v0 (same recipe as `es_assigned_s003`): `2026-10-24_06-33-03_es_close_from35`, **stopped at gen 10, flat** (population breach 62–64% every generation; checkpoints agent_5/10 and `es_state.pt` kept for a resume).

ES gen 35, hosts' `evaluate.py --task`, 64 × 640 (and at the defaults, 64 × 20):

| spawn | 6 m | 5 m | 4 m | 3 m | 2.5 m | 3–6 m (Close) |
|---|---|---|---|---|---|---|
| 64 × 640 | 385.4 / 3.9 | 454.3 / 8.5 | 429.3 / 40.2 | 114.5 / 98.4 | 35.8 / 100 | 302.8 / 61.1 |
| defaults | 389.8 / 1.6 | 451.6 / 7.8 | 115.0 / 100* | 20.0 / 100* | 10.0 / 100* | 50.0 / 100* |

\* the defaults stop at the first step with ≥ 20 finished episodes; at close spawns those are 20 early breaches, so they read 100%.

- **The cliff is between 5 and 4 m.** From the hosts' 2.5 m defender ring, a defender reaches a 3 m attacker before it crosses 1.5 m only if it starts within about 50–70° of the attacker's bearing, so even perfect pairing covers both attackers in only ~30% of waves. Waves at 3 m last ~3 s, so a 20 s episode holds many, and one leak ends it.
- Teacher variants on Close-v0 (`distill.py --check assigned --contact_offset 0 --soft_closing 0.15`, 64 × 640; new `--pair` option, default unchanged):

  | variant | interception | breach | floor |
  |---|---|---|---|
  | slot pairing (default) | 252.2% | 71.4% | 3.1% |
  | + `--urgent_eta 1.5` | 254.3% | 70.9% | 2.6% |
  | + `--block 2.2` | 228.5% | 77.2% | 2.3% |
  | + `--block 2.2 --urgent_eta 1.5` | 239.7% | 73.4% | 2.5% |
  | `--pair rescue` (switch when own attacker unreachable in time) | 243.5% | 74.5% | 2.3% |
  | `--pair rescue --pair_margin 0.3` | 184.5% | 87.7% | 2.3% |
  | `--pair nearest` | 160.9% | 94.4% | 12.0% |

  None beats slot pairing; nearest pairing sends both defenders at one attacker and they crash into each other. The ES network (61.1%) is already better than its teacher here (71.4%).
- Defenders starting 1.0 m from the asset (`Team1-Defend-Spawn3-Def1-v0`): 249.5% / 95.3% (64 × 640), vs 98.4% from the 2.5 m ring.
- Defenders starting 1.5 m out (`-Spawn3-Def15-v0`): 205.3% / 95.5%.
- **Half-scale arena** (`Team1-Defend-Half-v0`: asset zone 0.75 m, defenders start 1.25 m, attackers 3 m; speeds unchanged, so it's harder than a true scale model): **723.0% / 14.5%** (64 × 640), 735.9% / 12.5% at the defaults.

First waves (`scripts/breach_autopsy.py`, 256 envs × ~1,024 episodes, diagnostic; it counts like evaluate.py, so fast-breaching envs are over-represented and these are upper bounds):

| task | episode breach | first-wave breach | worst wave |
|---|---|---|---|
| 6 m (hosts' geometry) | 2.7% | 0.5% | 3 |
| 5 m | 9.4% | 0.8% | 3 |
| 4 m | 42.7% | 18% | 1 |
| 3 m, defenders from 2.5 m (hosts') | 99.5% | **72%** | 1 |
| 3 m, defenders from 1.0 m | 96.8% | **17%** | **2** (34%) |
| half-scale arena | 13.8% | 0.6% | 5+ |

- **Where the defenders start is the lever at close range, not the network:** from 1.0 m, the same policy loses 17% of first waves instead of 72%.
- From 1.0 m, wave 2 is the worst: after the first capture the free defender goes out to help, so both are far from the asset when the next wave spawns (waves respawn the moment both attackers are down).

### Guard fallback for the free defender (new `distill.py --guard RADIUS ENGAGE_ETA`, 2026-10-24)

Once its own attacker is down, the defender waits on the live attacker's line to the asset, RADIUS m out at the attacker's height, and engages only when that attacker is within ENGAGE_ETA s of breaching. Teacher `--check assigned --contact_offset 0 --soft_closing 0.15`, 64 × 640, counted like evaluate.py:

| task | no guard | `--guard 1.8 1.5` | `--guard 1.8 2.5` |
|---|---|---|---|
| AS-Defend-v0 | 372.7% / 4.4% (floor 1.5%) | **379.1% / 1.2%** (floor 0.3%) | 373.0% / 2.5% (floor 0.4%) |
| Spawn3-Def1 | 201.3% / 98.3% | 205.8% / 98.0% | 209.5% / 97.8% |

- **On the official task the guard cuts the teacher's breaches from 28 to 8 in 640 episodes**, and crashed time by 5×. The free defender no longer joins its teammate on the last attacker, and teammate contact caused about half the crashes (flip autopsy).
- No help at 3 m spawns.
- Next: distil `--guard 1.8 1.5` on AS-Defend-v0, score with the hosts' `evaluate.py`, then ES.
- Guard settings (teacher, AS-Defend-v0, 64 × 640): 1.8 m / 1.0 s → 371.6% / 4.1%; 2.2 m / 1.5 s → 370.5% / 3.4%; **1.5 m / 1.5 s → 376.2% / 1.7%** (floor 0.2%); 1.8 / 1.5 with soft 0.05 → 371.1% / 4.4%. Engage at 1.5 s and a post at 1.5–1.8 m is the good region; the 1.2% is partly luck.
- **Distilled `--guard 1.8 1.5`** (`2026-10-24_07-35-49_distill_assigned_363220`, copy mse 0.0031). Hosts' `evaluate.py`, AS-Defend-v0:

  | | defaults (64 × 20) | 64 × 640 | 100 × 1,000 |
  |---|---|---|---|
  | distilled guard | 378.9% / **1.6%** | 366.7% / 3.4% | 370.2% / **2.9%** |
  | ES gen 35 (pick) | 381.2% / 4.7% | — / 4.1% (peer) | 390.5% / 3.2% |
  | distilled, no guard | | | 365.1% / 5.7% |

  Before ES it ties gen 35 on 1,000 episodes and halves the no-guard network's breaches (29 vs 57).
- ~~ES from it (same recipe as `es_assigned_s003`, 60 generations): `outputs/es_guard_s003.log`, running.~~
- **ES from it** (`2026-10-24_07-52-17_es_guard_s003`, same recipe as `es_assigned_s003`). Hosts' `evaluate.py`, AS-Defend-v0, same 100 × 1,000 scenarios as gen 35's 3.2%:

  | ES guard checkpoint | 100 × 1,000 | defaults (64 × 20) |
  |---|---|---|
  | gen 10 | 384.7% / **1.6%** | 395.3% / 3.1% |
  | gen 15 | 387.6% / 2.4% | 383.6% / 1.6% |
  | gen 20 | 388.6% / **1.5%** | 389.1% / 1.6% |
  | ES gen 35 (previous pick) | 390.5% / 3.2% | 381.2% / 4.7% |

  Picked out of several checkpoints on the same scenarios, so it's being confirmed on unseen ones (128 × 1,000 and 200 × 1,000) against gen 35.

**Unseen-scenario confirmation, final pick (hosts' `evaluate.py`, AS-Defend-v0; a different env count gives a different seeded scenario set):**

| checkpoint | 100 × 1,000 | 128 × 1,000 | 200 × 1,000 | pooled | defaults |
|---|---|---|---|---|---|
| guard gen 10 | 384.7 / 1.6 | 386.8 / 1.8 | 385.2 / 1.7 | 1.7% | 395.3 / 3.1 |
| **guard gen 20 (final)** | 388.6 / 1.5 | 387.1 / 2.5 | 392.1 / 1.0 | **1.7%** | 389.1 / **1.6** |
| ES gen 35 (previous pick) | 390.5 / 3.2 | 385.3 / 5.2 | 388.2 / 3.3 | 3.9% | 381.2 / 4.7 |

- On the two unseen sets: 36 breaches (gen 20) vs 86 (gen 35) in ~2,000 episodes each, well beyond noise.
- Gens 10 and 20 tie on breaches; gen 20 has more interceptions and the better default-setting score (the tie-break that picked gen 35).
- The ES guard run finished all 60 generations; gens 25–60 were not scored, to avoid picking on noise.

### Flight network: ES on the realistic command path at 6 m (2026-10-24, running)

The user wants a separate network for the real Tellos, trained with delay; **contact rules stay the original ones** (official 0.35 m capture, same physics; nothing about contact changes). The official submission is untouched.

- New tasks (`team1_defend_flight_env.py`, registered in `team1_defend/__init__.py`), all with attackers on the official 6 m ring:
  - `Team1-Defend-FittedT024-Spawn6-v0`: FittedT024 (positions-only sensing, inferred asset, realistic attackers, 0.75 m/s cap, 0.1 s command delay, 0.2 s fix age, 13.5 Hz integer rc, single-lag response) with no deploy layer.
  - `Team1-Defend-FittedT024-Spawn6-PredictCVSep-v0`: the same plus the laptop's CV predictor and separation filter. **This is what we fly, and the ES training task.**
  - `Team1-Defend-RR-Spawn6-PredictCVSep-v0`: fallback (Robust-Realistic at 6 m plus the layer, official command path).
- Attackers here are pessimistic: 70% weave, 0.2–1.0 m/s capped at the defenders' 0.75 m/s. The hosts' own attack script flies ~0.2–0.52 m/s on smooth paths.
- **Run:** `es_train.py --task Team1-Defend-FittedT024-Spawn6-PredictCVSep-v0 --checkpoint logs/skrl/tello_defend/2026-10-24_07-52-17_es_guard_s003/checkpoints/agent_20.pt --num_envs 4096 --pop 120 --center_envs 256 --sigma 0.003 --lr 0.0005 --generations 60 --save_every 5 --w_hold 0.5 --w_capture 0.01 --w_floor 0.5 --name flight_g20` (guard run's recipe), log `simulation_infra/outputs/es_flight_g20.log`.
- Stop rule: if the center breach hasn't moved after ~15 generations, stop and restart on `RR-Spawn6-PredictCVSep`.
- Pending baselines (guard gen 20, `evaluate.py --task`, 64 × 640): FittedT024-Spawn6, FittedT024-Spawn6-PredictCVSep, Fitted-Spawn6 (gen 35 there: 80.0%).

### Write-up and final videos (2026-10-24)

- **`METHODOLOGY.md`** (repo root): how we win 2 v 2, for the judges' Explanation category. Covers the pipeline (teacher → DAgger → ES), the final pick (guard ES gen 20), why PPO failed, the crash insight, robustness, 3 v 2 (goalkeeper, in training), limitations, the laptop layer and reproduce commands. Numbers are the hosts' `evaluate.py` unless marked internal.
- **Videos** (`simulation_infra/scripts/record_defend.py`, 45 s each, seed 7: same first wave, scenarios diverge after the first breach), in `simulation_infra/outputs/videos/`:

  | clip | 45 s: intercepted / breaches |
  |---|---|
  | `defend_final_guard20.mp4`: **final pick**, guard ES gen 20 | 17 / 0 |
  | `defend_final_guard10.mp4`: guard ES gen 10 | 16 / 0 |
  | `defend_final_es35.mp4`: previous pick, ES gen 35 | 17 / 1 |
  | `defend_final_chase.mp4`: hosts' chase baseline | 5 / 5 |
  | `defend_final_side_by_side.mp4`: chase (left) vs guard gen 20 (right), 1920 × 540 | |

  Check frames: `simulation_infra/outputs/video_checks/`. Frame folders `outputs/frames_final_*` are kept.

### 3 defenders vs 2 attackers (user request, 2026-10-24): `Team1-Defend-3v2-v0`

New `team1_defend_3v2_env_cfg.py`: the hosts' TelloDefendEnv with `num_defenders = 3`; geometry, attackers and the 23-number observation unchanged (no teammate inputs, so it stays the hosts' layout). User's choices: the third defender is a **goalkeeper** (new `distill.py --keeper RADIUS ENGAGE_ETA`: holds a post on the most urgent attacker's line and engages only near breach), hosts' geometry, the guard recipe, ES 100 generations.

Teacher (`--check assigned --contact_offset 0 --soft_closing 0.15`, 64 × 640, counted like evaluate.py):

| variant | interception | breach | floor |
|---|---|---|---|
| double-team (defender 3 also takes attacker 1) + guard 1.8/1.5 | 287.0% | 46.6% | 21.0% |
| keeper 1.8/1.5, no guard | 356.5% | 11.2% | 5.0% |
| keeper 1.8/1.5 + guard 1.8/1.5 | 346.0% | 14.5% | 6.7% |
| keeper 1.8/1.5 + guard 1.8/1.0 | 347.8% | 13.3% | 6.4% |
| **keeper 1.6/1.5 + guard 2.2/1.5** | **358.2%** | **10.1%** | 4.4% |

- Double-teaming crashes constantly. Separating the keeper's post from the guard post helps.
- Three defenders still do worse than the 2v2 teacher (1.2–4.4%): the extra drone adds crashes near the asset.
- **Organizers' real capture rule (user, same day): the attacker has to fall out of the air.** The sim counts any approach within 0.35 m and the "downed" attacker hovers, so sim captures overstate real ones; a 0.15 m/s soft bump probably won't drop a Tello. This run still trains the intercept under delay and the 0.75 m/s cap; its capture numbers are sim-rule only. Proposed: a laptop-side ram override for the last ~0.7 m (not scorable in this sim: the thrust bug flips every rammer), with the ram speed set by a real test (one defender vs a hovering Tello at 0.3 / 0.5 / 0.75 m/s). The scripted hybrid (`control/defense.py`, `contact_offset` 0) already flies straight in.
- Progress: center breach 39.5% (gen 1) → 34.4% (gen 5) → 33.6% (gen 10); 256 episodes each, ±3 points.
- **Baselines, guard gen 20 at 6 m** (`evaluate.py --task`, 64 × 640, sim proximity capture): FittedT024-Spawn6 **148.1% / 62.2%** (9.4 s); FittedT024-Spawn6-PredictCVSep **185.2% / 40.2%** (10.4 s). The predictor + filter is worth 22 points at 6 m. Fitted-Spawn6 (old τ) pending.
- **Soft-contact ES run stopped at gen 12** (user: put the ram in the sim and retrain). Center breach 39.5 → 33.6–35.9% over gens 1–12 (±3). Checkpoints `2026-10-24_09-00-53_es_flight_g20/checkpoints/agent_5.pt`, `agent_10.pt`, `es_state.pt` kept.

### Knockdown capture: the organizers' real rule in the sim (new `team1_defend_knockdown_env.py`, 2026-10-24)

- `KnockdownCaptureMixin`: an attacker is captured only when knocked out of the air, i.e. tilted past 30° (the flip definition from the crash autopsy; the attacker controller holds it level, so only a hit tilts it) or below 0.3 m. Then its rotors are cut and it falls. `capture_radius` 0: proximity captures nothing. Counted in `_episode_captures`, so `evaluate.py` and `es_train.py` count it unchanged. Hosts' physics: a defender that tips in the hit is down. The hosts' capture reward and `cleared` bonus never fire (fine for ES and evaluate; not for PPO).
- Tasks: `Team1-Defend-Knockdown-v0` (official + knockdown), `Team1-Defend-Flight-Knockdown-v0` (FittedT024-Spawn6-PredictCVSep + knockdown, the new training task).
- First test (queued; the GPU is full of peer runs): teacher `--check assigned --contact_offset 0 --guard 1.8 1.5` on Knockdown-v0, hard vs `--soft_closing 0.15`. The soft teacher must capture far less than the hard one, or the 30° threshold doesn't encode the rule.
- Then: hard teacher on Flight-Knockdown → distill → ES on Flight-Knockdown. Defender floor time reported with every number.
- **Correction (user, same day): within 0.35 m still counts. "Add it and retrain" meant: don't slow down near the attacker (no soft contact), not a new capture rule.** The knockdown variant is not used; its file and two task IDs stay registered, unused. Its teacher checks were cancelled before finishing.
- Fitted-Spawn6 baseline (guard gen 20, old τ, no layer): **70.5% / 87.0%** (8.7 s). Against FittedT024-Spawn6's 62.2%, the τ correction is worth ~25 points at 6 m.
- Queued (GPU full of peer runs): no-slow-down teacher (`--check assigned --contact_offset 0 --guard 1.8 1.5`, no `--soft_closing`) on AS-Defend-v0 and FittedT024-Spawn6-PredictCVSep, 64 × 640, logs `simulation_infra/outputs/noslow/`. Known risk: the captured attacker hovers and a fast hit flips the defender for good (thrust bug), so the sim punishes not slowing down (pre-guard hard-contact teacher: 98.8% breach).
- **Retrain without the slow-down (user: "cancel and retrain")**: teacher checks cancelled. Chain (`simulation_infra/outputs/noslow/`): `distill.py --task Team1-Defend-FittedT024-Spawn6-PredictCVSep-v0 --teacher assigned --contact_offset 0 --guard 1.8 1.5` (guard network's settings minus `--soft_closing`; fresh network, 12 DAgger rounds) → `es_train.py` on the same task from the distilled checkpoint, guard recipe, 60 generations, `--name noslow`. Starts when ≥ 4 GB of GPU memory is free.
- **Switched to the official task (user: "i just want the behaviour the repo expects").** The flight-variant distill was stopped at round 3. The README's expected behaviour is pursuit into the attacker (`chase`), capture within 0.35 m, and its suggested levers are assigned targets, staying between attacker and site (our guard) and a behaviour-cloning warm start. Soft contact was ours, not the repo's. New chain (`simulation_infra/outputs/noslow_official/`): `distill.py --task AS-Defend-v0 --teacher assigned --contact_offset 0 --guard 1.8 1.5` (no `--soft_closing`) → `es_train.py --task AS-Defend-v0` from it, guard recipe, 60 generations, `--name noslow_official`. Scored with the hosts' `evaluate.py` on AS-Defend-v0 against guard gen 20.
