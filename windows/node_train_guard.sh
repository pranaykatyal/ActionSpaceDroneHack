#!/usr/bin/env bash
# Long AS-Defend-Guard-v0 runs (guard base + residual MAPPO) on the SF node, detached: several
# variants train side by side on the one H100, then each is scored against the guard alone.
# Run node_sync.sh first.
#   bash windows/node_train_guard.sh [node] [iterations] [num_envs] [hours] "variant overrides" ...
#   e.g. bash windows/node_train_guard.sh team1 2000 2048 2.5 \
#          "agent.models.policy.initial_log_std=-2.0" "agent.models.policy.initial_log_std=-1.5 env.residual_scale=0.5"
# Everything lands in /data/runs/<tag>/v<i>/ on the node (train.log, eval_*.log, logs/skrl/...).
# The hours cap is a hard `timeout` on training, so a stuck run cannot burn the budget.
set -euo pipefail
N=${1:-team1}
ITERS=${2:-2000}
ENVS=${3:-2048}
HOURS=${4:-2.5}
shift $(( $# < 4 ? $# : 4 ))
[ $# -gt 0 ] || set -- ""
AUTOR=${AUTOR:-$HOME/.local/bin/autor}
TAG=${TASK_TAG:-guard}_$(date +%Y%m%d_%H%M)
K=/workspace/isaaclab/isaaclab.sh
S=/data/as/simulation_infra/scripts

TRAIN=""
EVAL=""
i=0
for v in "$@"; do
  D=/data/runs/$TAG/v$i
  # the eval needs the env overrides too (residual_scale etc.), as evaluate.py --set KEY=VALUE
  SETS=$(echo "$v" | tr ' ' '\n' | { grep '^env\.' || true; } | sed 's/^env\.//' | tr '\n' ' ')
  echo "v$i: $v"
  TRAIN="$TRAIN (mkdir -p $D && cd $D && echo '$v' > variant.txt &&
     timeout ${HOURS}h $K -p $S/skrl/train.py --task ${TASK:-AS-Defend-Guard-v0} --algorithm MAPPO --headless \
       --num_envs $ENVS --max_iterations $ITERS $v > train.log 2>&1; echo v$i train exit \$?) &"
  EVAL="$EVAL cd $D; RUN=\$(ls -d logs/skrl/tello_defend/*guard* | tail -1);
     for c in best_agent.pt \$(ls \$RUN/checkpoints/ | grep '^agent_' | sort -t_ -k2 -n | tail -2); do
       $K -p $S/evaluate.py --task ${TASK:-AS-Defend-Guard-v0} --algorithm MAPPO --checkpoint \$RUN/checkpoints/\$c \
         --num_envs 256 --episodes 1000 --headless ${SETS:+--set $SETS} > eval_\$c.log 2>&1;
     done;"
  i=$((i + 1))
done

"$AUTOR" run "$N" -d --hold-on-failure 10m \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/data/as/simulation_infra/source/action_space_kit \
  --env TELLO_USD_PATH=/data/assets/tello.usd \
  -- sh -c "
    # the node image ships skrl 1.4.3; the configs and the clip_actions fix assume 2.1.1
    $K -p -m pip install -q skrl==2.1.1
    mkdir -p /data/runs/$TAG && cd /data/runs/$TAG
    $K -p $S/evaluate.py --task ${TASK:-AS-Defend-Guard-v0} --baseline zero --num_envs 256 --episodes 1000 --headless \
      > /data/runs/$TAG/eval_guard_only.log 2>&1 &
    $TRAIN
    wait
    $EVAL
    cd /data/runs/$TAG && grep -r -H -E 'interception rate|breach rate|crash rate' --include='eval_*.log' .
  "
echo "TAG=$TAG"
