# Evaluation logs: `2v2_guard_es_gen20`

Hosts' unmodified `scripts/evaluate.py`, copied unchanged from `simulation_infra/outputs/`.
Each row is read from its log's own header and results. `baselines/` holds chase and zero on the same tasks.

| log | policy | task | envs × episodes | interception | breach | time to intercept |
|---|---|---|---|---|---|---|
| `official_defaults_64x20.log` | agent_20.pt | `AS-Defend-v0` | 64 × 64 | 389.1% | **1.6%** | 10.8 s |
| `official_100x1000.log` | agent_20.pt | `AS-Defend-v0` | 100 × 1001 | 388.6% | **1.5%** | 10.8 s |
| `official_128x1000_unseen.log` | agent_20.pt | `AS-Defend-v0` | 128 × 1024 | 387.1% | **2.5%** | 10.8 s |
| `official_200x1000_unseen.log` | agent_20.pt | `AS-Defend-v0` | 200 × 1000 | 392.1% | **1.0%** | 10.9 s |
| `weaving_attackers_64x640.log` | agent_20.pt | `Team1-Defend-Realistic-v0` | 64 × 640 | 351.2% | **3.6%** | 10.7 s |
| `tello_rate_13p5hz_64x640.log` | agent_20.pt | `Team1-Defend-Rate135-v0` | 64 × 640 | 387.0% | **2.7%** | 10.8 s |
| `tello_rate_10hz_64x640.log` | agent_20.pt | `Team1-Defend-Rate10-v0` | 64 × 640 | 386.8% | **3.6%** | 10.8 s |
| `spawn_4m_64x640.log` | agent_20.pt | `Team1-Defend-Spawn4-v0` | 64 × 640 | 425.9% | **38.1%** | 10.0 s |
| `realistic_sensing_IRL_64x640.log` | agent_20.pt | `Team1-Defend-IRL-v0` | 64 × 640 | 185.4% | **80.3%** | 7.9 s |
| `tello_motion_priors_FittedT024_64x640.log` | agent_20.pt | `Team1-Defend-FittedT024-v0` | 64 × 641 | 88.8% | **92.0%** | 7.3 s |
| `tello_motion_priors_spawn6_64x640.log` | agent_20.pt | `Team1-Defend-FittedT024-Spawn6-v0` | 64 × 640 | 148.1% | **62.2%** | 9.4 s |
| `tello_motion_priors_spawn6_laptop_layer_64x640.log` | agent_20.pt | `Team1-Defend-FittedT024-Spawn6-PredictCVSep-v0` | 64 × 640 | 185.2% | **40.2%** | 10.4 s |
| `baselines/chase_official_64x128.log` | baseline:chase | `AS-Defend-v0` | 64 × 128 | 55.5% | **100.0%** | 3.1 s |
| `baselines/zero_official_64x128.log` | baseline:zero | `AS-Defend-v0` | 64 × 128 | 2.3% | **100.0%** | 4.9 s |
| `baselines/chase_official_defaults_64x20.log` | baseline:chase | `AS-Defend-v0` | 64 × 20 | 47.5% | **100.0%** | 3.0 s |
| `baselines/chase_official_100x1000.log` | baseline:chase | `AS-Defend-v0` | 100 × 1000 | 59.0% | **100.0%** | 3.5 s |
| `baselines/zero_official_defaults_64x20.log` | baseline:zero | `AS-Defend-v0` | 64 × 20 | 0.0% | **100.0%** |  |
| `baselines/zero_official_100x1000.log` | baseline:zero | `AS-Defend-v0` | 100 × 1000 | 3.1% | **99.9%** | 5.4 s |
