# Team 1, 3 defenders vs 2 attackers, normal (full-speed) contact: `agent_40.pt`

**The pick is `agent_40.pt` (ES generation 40, 11:55).** `agent_30.pt` was the pick at 11:53, before gen 40's score landed; it's kept for the record.

Same network, inputs, outputs and task as `../team1_defend_3v2_soft/` (see there), but the teacher hits attackers
at full speed: no soft-contact slowdown.

- **Source:** `logs/skrl/tello_defend/2026-10-24_10-20-11_es_3v2_hard/checkpoints/agent_40.pt` (byte-identical), ES generation 40 (`agent_30.pt`: same run, gen 30).
- **Recipe:**
  1. Teacher `distill.py --teacher assigned --contact_offset 0 --guard 2.2 1.5 --keeper 1.6 0.8` (no `--soft_closing`).
  2. DAgger distillation: `2026-10-24_10-15-47_distill_assigned_464286` (25.5% breach; defenders on the floor 32.7% of the time).
  3. ES as for the soft version, stopped at generation 40 for the 12:00 deadline; gen 40 scored best.
- **Hosts' `scripts/evaluate.py --task Team1-Defend-3v2-v0`, 100 envs × 1,000:**

  | network | interception | breach |
  |---|---|---|
  | distilled (gen 0) | 257.2% | 25.5% |
  | ES gen 10 | 312.7% | 11.2% |
  | ES gen 20 | 333.6% | 7.3% |
  | ES gen 30 | 342.9% | 6.4% |
  | **ES gen 40 (the pick)** | **347.3%** | **4.9%** |
  | chase baseline (64 × 640) | 69.6% | 99.8% |

- A full-speed hit tips the defender in the hosts' sim (thrust-frame bug), so this version crashes more and
  breaches about 4× more than soft contact.

```bash
python -u scripts/evaluate.py --task Team1-Defend-3v2-v0 --checkpoint policies/team1_defend_3v2_normal/agent_40.pt --num_envs 100 --episodes 1000
```

Weaving attackers (`Team1-Defend-3v2-Realistic-v0`, 64 × 640): gen 40 267.1% / 24.7%, gen 30 259.5% / 27.2%.
