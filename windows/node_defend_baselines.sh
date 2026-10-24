#!/usr/bin/env bash
# Measure AS-Defend-v0 baselines (zero, chase) on the SF node, detached. Run node_sync.sh first.
#   bash windows/node_defend_baselines.sh [node] [episodes]
# Logs: /data/runs/defend_baseline_{zero,chase}.log on the node.
set -euo pipefail
N=${1:-team1}
EPISODES=${2:-50}
AUTOR=${AUTOR:-$HOME/.local/bin/autor}

"$AUTOR" run "$N" -d --hold-on-failure 10m \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/data/as/simulation_infra/source/action_space_kit \
  --env TELLO_USD_PATH=/data/assets/tello.usd \
  -- sh -c "cd /workspace/isaaclab && mkdir -p /data/runs && for b in zero chase; do
      echo \"== baseline \$b\";
      ./isaaclab.sh -p /data/as/simulation_infra/scripts/evaluate.py --task AS-Defend-v0 --baseline \$b \
        --episodes $EPISODES --headless > /data/runs/defend_baseline_\$b.log 2>&1;
      grep -vE 'Warning|^\s*\$' /data/runs/defend_baseline_\$b.log | tail -12;
    done"
