#!/usr/bin/env bash
# Long AS-Defend-Guard-v0 run (guard base + residual MAPPO) on the SF node, detached, then
# score the result against the guard alone. Run node_sync.sh first.
#   bash windows/node_train_guard.sh [node] [iterations] [num_envs] [hours] [extra hydra overrides...]
# Everything lands in /data/runs/<tag>/ on the node (train.log, eval_*.log, logs/skrl/...).
# The hours cap is a hard `timeout` on training, so a stuck run cannot burn the budget.
set -euo pipefail
N=${1:-team1}
ITERS=${2:-3000}
ENVS=${3:-4096}
HOURS=${4:-3}
shift $(( $# < 4 ? $# : 4 ))
EXTRA="$*"
AUTOR=${AUTOR:-$HOME/.local/bin/autor}
TAG=guard_$(date +%Y%m%d_%H%M)
K=/workspace/isaaclab/isaaclab.sh
S=/data/as/simulation_infra/scripts

"$AUTOR" run "$N" -d --hold-on-failure 10m \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/data/as/simulation_infra/source/action_space_kit \
  --env TELLO_USD_PATH=/data/assets/tello.usd \
  -- sh -c "
    set -x
    # the node image ships skrl 1.4.3; the configs and the clip_actions fix assume 2.1.1
    $K -p -m pip install -q skrl==2.1.1
    mkdir -p /data/runs/$TAG && cd /data/runs/$TAG
    timeout ${HOURS}h $K -p $S/skrl/train.py --task AS-Defend-Guard-v0 --algorithm MAPPO --headless \
        --num_envs $ENVS --max_iterations $ITERS $EXTRA > train.log 2>&1
    echo train exit \$?
    RUN=\$(ls -d logs/skrl/tello_defend/*guard* | tail -1)
    echo RUN=\$RUN
    $K -p $S/evaluate.py --task AS-Defend-Guard-v0 --baseline zero --num_envs 256 --episodes 1000 --headless > eval_guard_only.log 2>&1
    for c in \$(ls \$RUN/checkpoints/ | grep -E 'best_agent|agent_[0-9]+' | sort -t_ -k2 -n | tail -3); do
      $K -p $S/evaluate.py --task AS-Defend-Guard-v0 --algorithm MAPPO --checkpoint \$RUN/checkpoints/\$c \
          --num_envs 256 --episodes 1000 --headless > eval_\$c.log 2>&1
    done
    grep -H -E 'interception rate|breach rate|crash rate' eval_*.log
  "
echo "TAG=$TAG"
