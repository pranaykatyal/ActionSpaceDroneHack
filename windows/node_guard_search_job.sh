#!/usr/bin/env bash
# Runs ON the node (started by node_guard_search.sh): random search over the intercept guard's
# settings, then the best few re-scored with more episodes.
#   bash node_guard_search_job.sh <dir> <configs> <workers> <hours> <task>
set -uo pipefail
D=$1; CONFIGS=$2; WORKERS=$3; HOURS=$4; TASK=$5
K=/workspace/isaaclab/isaaclab.sh
S=/data/as/simulation_infra/scripts
DEADLINE=$(( $(date +%s) + $(awk "BEGIN{print int($HOURS * 3600)}") ))
mkdir -p "$D/logs" && cd "$D"

# the settings to try: the current AS-Defend-RoomReal3 tuning first, then random ones around it
"$K" -p - "$CONFIGS" > configs.txt <<'PY'
import json, random, sys
random.seed(1)
base = {"speed": 0.65, "intercept_radius": 1.35, "park_radius": 3.0, "dock_gap": 0.0, "guard_offset": 0.0,
        "park_lead": 0.0, "dock_range": 0.0, "gain": 4.0, "lead_horizon": 1.5, "sweep_depth": 0.2,
        "hysteresis": 0.4, "sep_range": 1.0, "clearance": 0.7}
print(json.dumps(base))
for _ in range(int(sys.argv[1]) - 1):
    c = dict(base)
    c.update(intercept_radius=round(random.uniform(0.9, 2.3), 2), park_radius=round(random.uniform(2.0, 4.0), 2),
             gain=round(random.uniform(2.0, 7.0), 2), lead_horizon=round(random.uniform(0.4, 3.0), 2),
             sweep_depth=round(random.uniform(0.0, 0.7), 2), hysteresis=round(random.uniform(0.1, 0.9), 2),
             sep_range=round(random.uniform(0.6, 1.5), 2), clearance=round(random.uniform(0.35, 0.9), 2))
    print(json.dumps(c))
PY

score() {  # score <episodes> <json settings> <log>  ->  breach %,crash %,settings
  timeout 20m "$K" -p "$S/evaluate.py" --task "$TASK" --baseline zero --num_envs 256 --episodes "$1" \
    --headless --set "guard_kwargs=$2" > "$3" 2>&1
  local b c
  b=$(grep -o 'breach rate: *[0-9.]*' "$3" | grep -o '[0-9.]*$')
  c=$(grep -o 'crash rate: *[0-9.]*' "$3" | grep -o '[0-9.]*$')
  echo "${b:-NA},${c:-NA},$2"
}

worker() {  # worker <w>: configs w, w + WORKERS, ... until the deadline
  local i=0 cfg
  while IFS= read -r cfg; do
    if [ $((i % WORKERS)) -eq "$1" ]; then
      [ "$(date +%s)" -ge "$DEADLINE" ] && break
      score 500 "$cfg" "logs/search_$i.log" >> "logs/search_w$1.csv"
    fi
    i=$((i + 1))
  done < configs.txt
}

for w in $(seq 0 $((WORKERS - 1))); do worker "$w" & sleep 20; done
wait
cat logs/search_w*.csv | grep -v '^NA' | sort -t, -k1 -g > search.csv
echo "== search: $(wc -l < search.csv) configs scored; best 8:"; head -8 search.csv

# stage 2: the 8 best and the current settings, 2000 episodes each, 4 at a time
{ head -8 search.csv | cut -d, -f3-; head -1 configs.txt; } | awk '!seen[$0]++' > top.txt
j=0
while IFS= read -r cfg; do
  score 2000 "$cfg" "logs/final_$j.log" >> logs/final.csv &
  j=$((j + 1)); [ $((j % 4)) -eq 0 ] && wait
done < top.txt
wait
sort -t, -k1 -g logs/final.csv > final.csv
echo "== final (breach %, crash %, settings; current tuning included)"; cat final.csv
