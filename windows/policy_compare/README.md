# Policy comparison (spoofed swarm_ops runs)

How the numbers on the comparison page (`spoof_test.html`) were made. The policy files run as
shipped in the hosts' policy runner and telemetry bus; only the drones, attackers, cameras and the
clock are simulated (numpy, no Isaac).

Setup: a scratch folder holding `spoof/drone_deployment_infra` (the hosts' `swarm_ops` with
`drive/` policy files copied into `policies/`, and our TELLO-3 map as `config/map.yaml`) next to
these scripts. Use the `isaaclab` env for any `.pt` checkpoint (needs torch).

- `spoof_run.py <swarm_ops dir> --policy defend [--scenario goal|line] [--att-speed lo,hi]
  [--true-gain 0.014] [--messy] [--cam] [--ideal]`: one policy reading attackers from the bus
  (ours, Shibani's), 200 identical episodes per setting. `--policy defend_room [--fallback]` runs
  Rohan's network (Mo's chaser with `--fallback`) through its own functions.
- `test_mo_file.py <defend.py> <checkpoint.pt> [--speed lo,hi] [--messy]`: a Mo-style file as
  written (5 drones, its own cameras and scripted attackers, its own catch/breach rules).
- `mkcompare.py` + `compare_tpl.html` -> `spoof_test.html` (results table is filled in by hand
  from the runs).
