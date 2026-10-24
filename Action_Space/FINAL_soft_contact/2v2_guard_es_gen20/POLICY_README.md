# Team 1 defend policy (final pick): `agent_20.pt`

The checkpoint we hand over for `AS-Defend-v0`, replacing `../team1_defend_es35/agent_35.pt` (kept for the record).
It is the hosts' unchanged skrl IPPO network (`tello_defend/agents/skrl_ippo_cfg.yaml`, 23 observations in,
`[vx, vy, vz, yaw_rate]` in [-1, 1] out) with nothing around it: no predictor, filter or scripted layer.

- **Source:** `logs/skrl/tello_defend/2026-10-24_07-52-17_es_guard_s003/checkpoints/agent_20.pt`
  (byte-identical copy, sha256 `a38f69c626b299583be088c84788ff8b103ab01f889a5561419162a0f9bdc008`).
- **How it was made** (the gen 35 pipeline plus one teacher rule, the guard):
  1. A scripted teacher built only from each defender's own 23 observations (`scripts/skrl/distill.py --teacher assigned
     --contact_offset 0 --soft_closing 0.15 --guard 1.8 1.5`):
     - defender *i* takes attacker slot *i* and leads the intercept at full speed;
     - inside 0.7 m it flies the attacker's velocity plus 0.15 m/s closing, so the contact downs the attacker without
       tipping the defender;
     - **guard:** once its own attacker is down, it doesn't join its teammate. It waits 1.8 m from the asset on the
       live attacker's line in, at the attacker's height, and engages only when that attacker is 1.5 s from breaching.
       Teammate contact caused about half the remaining crashes.
  2. DAgger distillation into the IPPO network: `2026-10-24_07-35-49_distill_assigned_363220` (copy mse 0.0031; 2.9% breach).
  3. Evolution-strategies fine-tune (`scripts/skrl/es_train.py`, σ 0.003, lr 0.0005, 120 copies × 32 envs,
     fitness = no breach + 0.5 × time held + 0.01 × interceptions − 0.5 × defender floor time), generation 20 of 60.
- **Hosts' `scripts/evaluate.py`, AS-Defend-v0:**

  | setting | interception | breach |
  |---|---|---|
  | defaults (64 envs, 20 episodes; seeded) | 389.1% | **1.6%** (1 of 64) |
  | 100 envs × 1,000 | 388.6% | 1.5% |
  | 128 envs × 1,000 (scenarios not used to pick it) | 387.1% | 2.5% |
  | 200 envs × 1,000 (scenarios not used to pick it) | 392.1% | 1.0% |
  | **pooled, ~3,000 episodes** | ~389% | **1.7%** |

  On the same three scenario sets, gen 35 scores 3.2 / 5.2 / 3.3% (pooled 3.9%).

```bash
python -u scripts/evaluate.py --checkpoint policies/team1_defend_guard20/agent_20.pt
```

Details and every other run: `EXPERIMENTS.md` at the repo root.
