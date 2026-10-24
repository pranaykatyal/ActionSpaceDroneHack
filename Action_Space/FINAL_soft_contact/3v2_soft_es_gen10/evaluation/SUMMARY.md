# Evaluation logs: `3v2_soft_es_gen10`

Hosts' unmodified `scripts/evaluate.py`, copied unchanged from `simulation_infra/outputs/`.
Each row is read from its log's own header and results. `baselines/` holds chase and zero on the same tasks.

| log | policy | task | envs × episodes | interception | breach | time to intercept |
|---|---|---|---|---|---|---|
| `official_defaults_64x20.log` | agent_10.pt | `Team1-Defend-3v2-v0` | 64 × 64 | 378.1% | **6.2%** | 10.8 s |
| `official_100x1000.log` | agent_10.pt | `Team1-Defend-3v2-v0` | 100 × 1000 | 379.6% | **1.2%** | 10.8 s |
| `official_128x1000_unseen.log` | agent_10.pt | `Team1-Defend-3v2-v0` | 128 × 1026 | 378.1% | **2.2%** | 10.8 s |
| `official_200x1000_unseen.log` | agent_10.pt | `Team1-Defend-3v2-v0` | 200 × 1000 | 381.1% | **1.5%** | 10.8 s |
| `distilled_before_es_100x1000.log` | agent_distill.pt | `Team1-Defend-3v2-v0` | 100 × 1001 | 364.1% | **5.5%** | 10.7 s |
| `weaving_attackers_64x640.log` | agent_10.pt | `Team1-Defend-3v2-Realistic-v0` | 64 × 642 | 336.0% | **3.3%** | 10.7 s |
| `room_hosts_map_64x640.log` | agent_10.pt | `Team1-Defend-3v2-Room-v0` | 64 × 642 | 1084.1% | **5.1%** | 10.5 s |
| `room_hosts_map_weaving_64x640.log` | agent_10.pt | `Team1-Defend-3v2-Room-Realistic-v0` | 64 × 641 | 1035.6% | **8.3%** | 10.5 s |
| `realistic_sensing_IRL_64x640.log` | agent_10.pt | `Team1-Defend-3v2-IRL-v0` | 64 × 640 | 180.9% | **80.3%** | 8.0 s |
| `tello_motion_priors_FittedT024_64x640.log` | agent_10.pt | `Team1-Defend-3v2-FittedT024-v0` | 64 × 640 | 100.1% | **88.8%** | 8.0 s |
| `tello_motion_priors_spawn6_64x640.log` | agent_10.pt | `Team1-Defend-3v2-FittedT024-Spawn6-v0` | 64 × 640 | 170.0% | **51.6%** | 9.8 s |
| `tello_motion_priors_spawn6_laptop_layer_64x640.log` | agent_10.pt | `Team1-Defend-3v2-FittedT024-Spawn6-PredictCVSep-v0` | 64 × 640 | 180.1% | **39.8%** | 10.4 s |
| `baselines/chase_official_64x640.log` | baseline:chase | `Team1-Defend-3v2-v0` | 64 × 641 | 69.6% | **99.8%** | 3.4 s |
| `baselines/chase_official_defaults_64x20.log` | baseline:chase | `Team1-Defend-3v2-v0` | 64 × 20 | 50.0% | **100.0%** | 2.9 s |
| `baselines/chase_official_100x1000.log` | baseline:chase | `Team1-Defend-3v2-v0` | 100 × 1000 | 65.5% | **99.9%** | 3.3 s |
| `baselines/zero_official_defaults_64x20.log` | baseline:zero | `Team1-Defend-3v2-v0` | 64 × 20 | 0.0% | **100.0%** |  |
| `baselines/zero_official_100x1000.log` | baseline:zero | `Team1-Defend-3v2-v0` | 100 × 1001 | 5.8% | **100.0%** | 5.4 s |
| `baselines/chase_weaving_attackers_64x640.log` | baseline:chase | `Team1-Defend-3v2-Realistic-v0` | 64 × 640 | 68.7% | **92.2%** | 3.7 s |
| `baselines/zero_weaving_attackers_64x640.log` | baseline:zero | `Team1-Defend-3v2-Realistic-v0` | 64 × 640 | 6.2% | **98.6%** | 6.3 s |
| `baselines/chase_room_hosts_map_64x640.log` | baseline:chase | `Team1-Defend-3v2-Room-v0` | 64 × 640 | 101.4% | **100.0%** | 2.8 s |
| `baselines/chase_room_hosts_map_weaving_64x640.log` | baseline:chase | `Team1-Defend-3v2-Room-Realistic-v0` | 64 × 640 | 106.1% | **97.8%** | 3.4 s |
| `baselines/chase_room_user_map_64x640.log` | baseline:chase | `Team1-Defend-3v2-Room-UserMap-v0` | 64 × 640 | 104.8% | **99.1%** | 2.2 s |
| `baselines/chase_room_user_map_weaving_64x640.log` | baseline:chase | `Team1-Defend-3v2-Room-UserMap-Realistic-v0` | 64 × 640 | 108.9% | **99.1%** | 2.5 s |
| `baselines/chase_realistic_sensing_IRL_64x640.log` | baseline:chase | `Team1-Defend-3v2-IRL-v0` | 64 × 640 | 49.5% | **98.6%** | 2.9 s |
| `baselines/chase_tello_motion_priors_FittedT024_64x640.log` | baseline:chase | `Team1-Defend-3v2-FittedT024-v0` | 64 × 640 | 33.2% | **99.7%** | 3.5 s |
| `baselines/chase_tello_motion_priors_spawn6_64x640.log` | baseline:chase | `Team1-Defend-3v2-FittedT024-Spawn6-v0` | 64 × 640 | 48.8% | **92.8%** | 4.9 s |
| `baselines/chase_tello_motion_priors_spawn6_laptop_layer_64x640.log` | baseline:chase | `Team1-Defend-3v2-FittedT024-Spawn6-PredictCVSep-v0` | 64 × 640 | 62.3% | **91.6%** | 5.1 s |

Pending (not run yet, or still running):

| log | source |
|---|---|
| `room_user_map_64x640.log` | `eval_3v2_usermap/soft_Room-UserMap.log` |
| `room_user_map_weaving_64x640.log` | `eval_3v2_usermap/soft_Room-UserMap-Realistic.log` |
