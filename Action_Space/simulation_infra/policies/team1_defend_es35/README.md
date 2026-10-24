# Team 1 defend policy: `agent_35.pt`

The checkpoint we hand over for `AS-Defend-v0`. It is the hosts' unchanged skrl IPPO network
(`tello_defend/agents/skrl_ippo_cfg.yaml`, 23 observations in, `[vx, vy, vz, yaw_rate]` in [-1, 1] out)
with nothing around it: no predictor, filter or scripted layer.

- **Source:** `logs/skrl/tello_defend/2026-10-24_01-25-32_es_assigned_s003/checkpoints/agent_35.pt`
  (byte-identical copy, sha256 `fc02b85f3a86dfaf3c2d22ed783e1a843a0ffbba48e70606380df1e8c497e80f`).
- **How it was made:**
  1. A scripted teacher built only from each defender's own 23 observations (`scripts/skrl/distill.py --teacher assigned`):
     defender *i* takes attacker slot *i*, leads the intercept at full speed, and inside 0.7 m flies the attacker's
     velocity plus 0.15 m/s closing, so the contact downs the attacker without tipping the defender.
  2. DAgger distillation into the IPPO network: `2026-10-24_01-15-18_distill_assigned_147307` (6.3% breach).
  3. Evolution-strategies fine-tune (`scripts/skrl/es_train.py`, σ 0.003, lr 0.0005, 120 copies × 32 envs,
     fitness = no breach + 0.5 × time held + 0.01 × interceptions − 0.5 × defender floor time), generation 35.
- **Hosts' `scripts/evaluate.py`, AS-Defend-v0:**
  - defaults (64 envs, 20 episodes; seeded, so it gives the same result every run): **381.2% interception / 4.7% breach** (3 of 64 episodes);
  - 1,000 fresh episodes (100 envs): **390.5% / 3.2%**, 10.7 s to intercept.

```bash
python -u scripts/evaluate.py --checkpoint policies/team1_defend_es35/agent_35.pt
```

Details and every other run: `EXPERIMENTS.md` at the repo root.
