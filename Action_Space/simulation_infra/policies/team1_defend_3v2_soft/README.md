# Team 1, 3 defenders vs 2 attackers, soft contact: `agent_10.pt`

The hosts' unchanged skrl IPPO network (`skrl_ippo_cfg.yaml`, copied here unmodified), one per defender
(`defender_0`, `defender_1`, `defender_2`). Inputs per defender: the hosts' 23 numbers (own position relative to
the asset, own velocity, gravity in the body frame, then each attacker's relative position, relative velocity and
alive flag); no teammate inputs. Outputs `[vx, vy, vz, yaw_rate]` in [-1, 1], times 1 m/s.

- **Source:** `logs/skrl/tello_defend/2026-10-24_10-16-07_es_3v2_soft/checkpoints/agent_10.pt` (byte-identical), ES generation 10.
- **Task:** `Team1-Defend-3v2-v0`, the hosts' AS-Defend-v0 with `num_defenders = 3` (new file
  `team1_defend_3v2_env_cfg.py`): attackers spawn 6 m out at 0.4-0.9 m/s, 1.5 m zone, defenders start on the 2.5 m ring.
- **Recipe:**
  1. Scripted teacher (`distill.py --teacher assigned --contact_offset 0 --soft_closing 0.15 --guard 2.2 1.5 --keeper 1.6 0.8`):
     - defenders 0 and 1 each take one attacker, lead the intercept, and soft-contact it (inside 0.7 m: match its velocity + 0.15 m/s);
     - a free defender waits 2.2 m out on the live attacker's line until it is 1.5 s from breaching;
     - defender 2 is a goalkeeper: it holds a post 1.6 m out on the most urgent attacker's line and engages only 0.8 s before breach.
  2. DAgger distillation: `2026-10-24_10-08-37_distill_assigned_458395` (5.5% breach).
  3. ES fine-tune (σ 0.003, lr 0.0005, 120 copies × 32 envs, fitness = no breach + 0.5 × held + 0.01 × interceptions − 0.5 × floor time).
- **Hosts' `scripts/evaluate.py --task Team1-Defend-3v2-v0`:**

  | setting | interception | breach |
  |---|---|---|
  | 100 envs × 1,000 | 379.6% | 1.2% |
  | 128 envs × 1,000 (unseen scenarios) | 378.1% | 2.2% |
  | 200 envs × 1,000 (unseen scenarios) | 381.1% | 1.5% |
  | **pooled, ~3,000 episodes** | ~380% | **1.6%** |
  | defaults (64 × 20) | 378.1% | 6.2% (4 of 64) |
  | chase baseline (64 × 640) | 69.6% | 99.8% |

- **Real flights:** the asset is taken as the centre of the defenders' takeoff positions, so take off evenly spaced
  around it (120° apart).

```bash
python -u scripts/evaluate.py --task Team1-Defend-3v2-v0 --checkpoint policies/team1_defend_3v2_soft/agent_10.pt --num_envs 100 --episodes 1000
```
