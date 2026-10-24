# Team 1 final policies: soft contact

The two winning soft-contact networks, their videos and their final scores in one place.

- **Network:** each is the hosts' unchanged skrl IPPO network: 23 observations in, `[vx, vy, vz, yaw_rate]`
  out, the hosts' `skrl_ippo_cfg.yaml`.
- **Scores:** every number below comes from the hosts' unmodified `scripts/evaluate.py`, as interception / breach.
- **Soft contact:** within 0.7 m of its attacker, a defender matches the attacker's velocity and closes at
  0.15 m/s. Contact is a gentle bump that downs the attacker without tipping the defender over.

| | 2 v 2 (official task) | 3 v 2 |
|---|---|---|
| Checkpoint | `2v2_guard_es_gen20/agent_20.pt` | `3v2_soft_es_gen10/agent_10.pt` |
| Task | `AS-Defend-v0` (hosts') | `Team1-Defend-3v2-v0` (hosts' env, 3 defenders) |
| **Breach, ~3,000 episodes** | **1.7%** | **1.6%** |
| Chase baseline breach | 100% | 99.8% |

Copies are byte-identical to `simulation_infra/policies/team1_defend_guard20/` and
`simulation_infra/policies/team1_defend_3v2_soft/`. Each policy folder also has the hosts' unmodified IPPO
config and the policy's own provenance README (`POLICY_README.md`).

**Evaluation logs:** each policy folder has an `evaluation/` folder with:
- the raw `evaluate.py` log behind every number below, copied unchanged;
- `evaluation/baselines/`: chase and do-nothing on the same tasks and settings;
- `evaluation/SUMMARY.md`: one row per log, read from the log itself (checkpoint, task, envs × episodes,
  interception, breach).

`simulation_infra/outputs/make_final_eval.py` rebuilds the summaries and adds new logs as runs finish.

## 2 v 2: guard ES gen 20 (the official submission)

| `evaluate.py`, `AS-Defend-v0` | Interception | Breach |
|---|---|---|
| defaults, no arguments (64 envs, seeded) | 389.1% | **1.6%** (1 of 64) |
| 100 envs × 1,000 | 388.6% | 1.5% |
| 128 envs × 1,000 (not used to pick it) | 387.1% | 2.5% |
| 200 envs × 1,000 (not used to pick it) | 392.1% | 1.0% |
| **pooled, ~3,000 episodes** | **~389%** | **1.7%** |
| chase baseline, defaults (same settings) | 47.5% | 100% |
| chase baseline, 100 envs × 1,000 (same settings) | 59.0% | 100% |
| do-nothing baseline, defaults | 0.0% | 100% |
| do-nothing baseline, 100 envs × 1,000 | 3.1% | 99.9% |

Robustness, same network, `evaluate.py --task` on our scenario variants, 64 × 640:

| Variant | Interception | Breach |
|---|---|---|
| weaving, wobbling attackers at 0.2–1.0 m/s | 351.2% | 3.6% |
| real Tello command rate, 13.5 Hz (10 Hz) | 387.0% (386.8%) | 2.7% (3.6%) |
| attackers start at 4 m instead of 6 m | 425.9% | 38.1% |
| realistic sensing: position fixes only + Kalman filter, estimated asset, 3–6 m spawns, 0.2 s delay | 185.4% | 80.3% |
| the above + the hosts' Tello motion priors | 88.8% | 92.0% |

Videos (`2v2_guard_es_gen20/videos/`, 45 s, same seed):

| Clip | Interceptions / breaches in 45 s |
|---|---|
| `defend_final_side_by_side.mp4`: hosts' chase (left) vs ours (right) | 5 / 5 vs 17 / 0 |
| `defend_final_guard20.mp4`: ours | 17 / 0 |
| `defend_final_chase.mp4`: hosts' chase baseline | 5 / 5 |

## 3 v 2: soft ES gen 10

Defenders 0 and 1 each take an attacker. A free defender guards 2.2 m out on the live attacker's line, and
defender 2 is a goalkeeper 1.6 m out. Built with the same pipeline as 2 v 2: scripted teacher, then DAgger
distillation, then evolution strategies.

| `evaluate.py --task Team1-Defend-3v2-v0` | Interception | Breach |
|---|---|---|
| 100 envs × 1,000 | 379.6% | **1.2%** |
| 128 envs × 1,000 (not used to pick it) | 378.1% | 2.2% |
| 200 envs × 1,000 (not used to pick it) | 381.1% | 1.5% |
| **pooled, ~3,000 episodes** | **~380%** | **1.6%** |
| defaults (64 envs, seeded) | 378.1% | 6.2% (4 of 64; a small sample) |
| chase baseline, defaults (same settings) | 50.0% | 100% |
| chase baseline, 100 envs × 1,000 (same settings) | 65.5% | 99.9% |
| chase baseline, 64 × 640 | 69.6% | 99.8% |
| do-nothing baseline, defaults | 0.0% | 100% |
| do-nothing baseline, 100 envs × 1,000 | 5.8% | 100% |

Defender time on the floor: 1.2% (chase 6.1%).

Other scenarios, 64 × 640:

| Task | Ours | Chase | Do nothing |
|---|---|---|---|
| weaving attackers (`Team1-Defend-3v2-Realistic-v0`) | **336.0% / 3.3%** | 68.7% / 92.2% | 6.2% / 98.6% |
| room from the organizers' map, official attackers (`-3v2-Room-v0`) | 1084.1% / 5.1% | 101.4% / 100% | |
| same room, weaving attackers (`-3v2-Room-Realistic-v0`) | 1035.6% / 8.3% | 106.1% / 97.8% | |
| room from the user's map (`-3v2-Room-UserMap-v0`) | pending (re-run after the far-wall fix) | | |
| **Tello motion priors + realistic sensing + 6 m spawns + laptop delay compensation and separation** (`-3v2-FittedT024-Spawn6-PredictCVSep-v0`; closest to a real flight) | **180.1% / 39.8%** | 62.3% / 91.6% | |
| the same without the laptop layer (`-3v2-FittedT024-Spawn6-v0`) | 170.0% / 51.6% | 48.8% / 92.8% | |
| realistic sensing: position fixes only + Kalman filter, estimated asset, 3–6 m spawns, 0.2 s delay (`-3v2-IRL-v0`) | 180.9% / 80.3% | 49.5% / 98.6% | |
| Tello motion priors, 3–6 m spawns (`-3v2-FittedT024-v0`) | 100.1% / 88.8% | 33.2% / 99.7% | |

**Read the room rows for breach only.**
- Every wave respawns instantly in the same start box. The defenders camp it, and 97% of captures happen
  there, so interception is inflated.
- The breach is optimistic too: only the first wave of each episode is a fair test.

Videos (`3v2_soft_es_gen10/videos/`, 45 s, same seed):

| Clip | Interceptions / breaches in 45 s |
|---|---|
| `defend3v2_side_by_side_chase_vs_soft.mp4`: hosts' chase (left) vs ours (right) | 8 / 5 vs 18 / 0 |
| `defend3v2_soft.mp4`: ours | 18 / 0 |
| `defend3v2_chase.mp4`: chase baseline | 8 / 5 |
| `defend3v2_zero.mp4`: do-nothing baseline | 1 / 5 |
| `defend3v2_usermap_soft.mp4`: ours in the room from the user's map (`-3v2-Room-UserMap-v0`) | 58 / 0 |
| `defend3v2_usermap_chase.mp4`: chase in that room | 10 / 7 |
| `defend3v2_usermap_side_by_side.mp4`: chase, normal contact (gen 40), soft contact, in that room | 10 / 7, 28 / 0, 58 / 0 |

The room clips show the room outline, tag 10 (white panel) and the protected zone in front of it. Interceptions
there are inflated by instant respawns at the attackers' start box.

## Reproduce

From `simulation_infra/`, with the shell set up as in `CLAUDE.md` step 2:

```bash
python -u scripts/evaluate.py --checkpoint ../FINAL_soft_contact/2v2_guard_es_gen20/agent_20.pt
python -u scripts/evaluate.py --checkpoint ../FINAL_soft_contact/2v2_guard_es_gen20/agent_20.pt --num_envs 200 --episodes 1000
python -u scripts/evaluate.py --task Team1-Defend-3v2-v0 --checkpoint ../FINAL_soft_contact/3v2_soft_es_gen10/agent_10.pt --num_envs 100 --episodes 1000
python -u scripts/evaluate.py --task Team1-Defend-3v2-v0 --baseline chase --num_envs 64 --episodes 640
```

Use at least 10× as many episodes as envs. `evaluate.py` stops after N finished episodes, and an env that
breaches early restarts and can breach again, so short runs over-count breaches.

## Limitations

- **Simulation scores only.** With realistic sensing and the hosts' Tello motion priors, the 2 v 2 network
  breaches 80–92% (table above).
  - The 3 v 2 soft network breaches 39.8% on its closest-to-flight variant with the laptop layer, against
    chase's 91.6% and the normal-contact network's 70.9%.
  - Expect real-world breach well above the official 1–2%. The motion priors are unfitted; `drone_deployment_infra/core/step_response_flight.py`
  is ready to measure the real values.
- **Soft contact may not capture in real flights.** The organizers count a capture only when the attacker falls
  out of the air, and a 0.15 m/s bump probably won't knock a Tello down. The normal-contact 3 v 2 network is in
  `simulation_infra/policies/team1_defend_3v2_normal/`: 347.3% / 4.9% official, 267.1% / 24.7% with weaving attackers.
- **3 v 2 in real flights:** the network takes the asset as the centre of the defenders' takeoff positions, so
  take off evenly spaced around it, 120° apart.

Pipeline and full write-up: `../METHODOLOGY.md`. Every run: `../EXPERIMENTS.md`.
