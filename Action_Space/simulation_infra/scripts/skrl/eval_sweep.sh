#!/bin/bash
# Score every checkpoint of a skrl run with the hosts' scripts/evaluate.py, to see whether
# training improves the defender or not (the reward curve alone can mislead).
#
#   scripts/skrl/eval_sweep.sh logs/skrl/tello_defend/<run> [episodes] [task]
#
# Writes <run>/eval_sweep.txt: one line per checkpoint (step, interceptions, breaches, time to intercept).
set -u
RUN=${1%/}
EPISODES=${2:-64}
TASK=${3:-AS-Defend-v0}
cd "$(dirname "$0")/../.."
unset CONDA_PREFIX CONDA_DEFAULT_ENV
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=${TELLO_USD_PATH:-$HOME/isaac/assets/tello.usd}
OUT="$RUN/eval_sweep.txt"
echo "# $TASK, $EPISODES episodes per checkpoint, scripts/evaluate.py" > "$OUT"
# paths may contain spaces ("Action space"): build the list without word splitting, sorted by step
shopt -s nullglob
mapfile -t CKS < <(for f in "$RUN"/checkpoints/agent_*.pt; do n=${f##*agent_}; echo "${n%.pt} $f"; done | sort -n | cut -d' ' -f2-)
CKS+=("$RUN/checkpoints/best_agent.pt")
for CK in "${CKS[@]}"; do
  [ -f "$CK" ] || continue
  LOG="$RUN/eval_$(basename "$CK" .pt).log"
  python -u scripts/evaluate.py --task "$TASK" --checkpoint "$CK" --episodes "$EPISODES" --num_envs 64 > "$LOG" 2>&1
  ROW=$(grep -E "interception rate|breach rate|time to intercept" "$LOG" | sed -E 's/ +/ /g' | tr '\n' ' ')
  printf "%-22s %s\n" "$(basename "$CK" .pt)" "$ROW" | tee -a "$OUT"
done
