#!/usr/bin/env bash
# Tune the scripted intercept guard's settings on the SF node, detached (from WSL). Run
# node_sync.sh first. The job (node_guard_search_job.sh) random-searches the guard settings in a
# realistic room task with evaluate.py --baseline zero --set guard_kwargs={...} (500 episodes
# each, WORKERS at a time), then re-scores the 8 best and the current settings with 2000.
#   bash windows/node_guard_search.sh [node] [configs] [workers] [hours] [task]
#   e.g. bash windows/node_guard_search.sh team1 240 8 2.5 AS-Defend-RoomReal3-v0
# Results on the node: /data/runs/<tag>/search.csv and final.csv (breach %, crash %, settings).
set -euo pipefail
N=${1:-team1}
CONFIGS=${2:-240}
WORKERS=${3:-8}
HOURS=${4:-2.5}
TASK=${5:-AS-Defend-RoomReal3-v0}
AUTOR=${AUTOR:-$HOME/.local/bin/autor}
TAG=gsearch_$(date +%Y%m%d_%H%M)

"$AUTOR" cp "$(dirname "$0")/node_guard_search_job.sh" "$N:/data/node_guard_search_job.sh"
"$AUTOR" run "$N" -d --hold-on-failure 10m \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/data/as/simulation_infra/source/action_space_kit \
  --env TELLO_USD_PATH=/data/assets/tello.usd \
  -- bash -c "/workspace/isaaclab/isaaclab.sh -p -m pip install -q skrl==2.1.1;
              bash /data/node_guard_search_job.sh /data/runs/$TAG $CONFIGS $WORKERS $HOURS $TASK > /data/runs/$TAG.log 2>&1;
              tail -20 /data/runs/$TAG.log"
echo "TAG=$TAG"
