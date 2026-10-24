# Team 1 defense: four policies for the organizers' `swarm_ops`

Copy this folder's `drone_deployment_infra/` and `simulation_infra/` over the organizers'
`Action_Space` repo. Two existing files change:

- `swarm_ops/policies/__init__.py`: registers our policies.
- `swarm_ops/swarm_policies.py`: one 3-line patch in `parse_runs`, so that any policy marked
  `external` runs as its own process, like `formation` does.

Everything else is new files. Only numpy and pyyaml are needed; no torch.

| `--run` name | What | Network inputs | Our drones' position |
|---|---|---|---|
| `defend` | **Team 1 scripted intercept guard**: the exact numpy port of our sim guard. One marker per attacker rams it; the spare defender is goalie. | none | dead reckoning from the printed start spots |
| `defend_cam` | the same guard, run by `defend_live.py` | none | **cameras + AprilTags**, every frame |
| `defend_rl` | Shibani's ES-trained 3v2 networks (`policies/weights/*.npz`, default `normal_contact_es_g40`) | 23 | dead reckoning (`policies/defend_basic.py`) |
| `defend_room` | Rohan's MAPPO room network (`defend_room.py`). Also flies 2 scripted attackers of its own. | 35 | cameras + AprilTags |

## Run (all drones facing tag 10)

```bash
cd drone_deployment_infra/swarm_ops
python swarm_policies.py --run defend:6,7,8 --run formation:4,5            # dry run; --fly to fly
python swarm_policies.py --run defend_cam:6,7,8 --run formation:4,5
python swarm_policies.py --run defend_rl:6,7,8 --run formation:4,5
python swarm_policies.py --run defend_room:3,4,5,6,7     # 3 defenders, then its own 2 attackers
```

## Notes

- **Map.** `defend`, `defend_cam` and `defend_rl` read the goal tag (6) from `config/map.yaml`,
  so they work with the organizers' final map. `defend_room` was trained for a room with tag 6
  at 6.48 m from tag 10 and refuses to fly with any other map.
- **`defend` / `defend_rl` (dead reckoning).** Place each defender on the start spot printed at
  launch (`DEFEND_START="x,z;x,z;x,z"` sets them). See `deliverable/README.md` for every
  setting. For `defend_rl` the networks expect a ring about 2.5 m around the goal (Shibani's
  README).
- **`defend_cam`.** Defenders 0.3–1 m in front of the goal tag, 0.5 m+ apart. Every defender
  must see a tag within 20 s of takeoff. A defender that loses its tags hovers, then lands after
  3 s. Options: `DEFEND_CAM_ARGS="--speed 0.65 --max-rc 45"`. Logs go to `logs/defend_live_*.csv`.
  For a standalone demo with our own scripted attackers:
  `python defend_live.py --drones 6,7,8 --attackers 4,5`.
- **None of these has flown on hardware yet.** Dry-run first and keep a hand on Ctrl+C.

## Simulation results (3 defenders vs 2 attackers, breach = share of episodes an attacker got in)

| Policy | Task | Breach |
|---|---|---|
| `defend` guard | our realistic 3v2 sim (sensing noise, 0.2 s lag, falling debris, curved/weaving/flanking/wandering attackers): organisers' room / 6 m corridor | 35.2% / 25.7% |
| `defend_rl` `normal_contact_es_g40` | hosts' AS-Defend-v0 with 3 defenders / Shibani's realistic weaving | 4.9% / 24.7% |
| `defend_room` B7200 | Rohan's AS-Defend-Room-v0 | 14.8% |
| hosts' `chase` baseline | AS-Defend-v0 | about 100% |
