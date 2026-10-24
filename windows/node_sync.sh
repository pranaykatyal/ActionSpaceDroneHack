#!/usr/bin/env bash
# Upload the Action_Space kit to the SF Compute node and prepare it (PDF steps 5-6), from WSL.
#   bash windows/node_sync.sh [node]        # default node: team1
# Leaves /data/as (the kit) and /data/assets/tello.usd on the node; both survive node stops.
# Jobs then need, besides $E:  --env PYTHONPATH=/data/as/simulation_infra/source/action_space_kit
#                              --env TELLO_USD_PATH=/data/assets/tello.usd
set -euo pipefail
N=${1:-team1}
AUTOR=${AUTOR:-$HOME/.local/bin/autor}
REPO=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== packing $REPO/Action_Space"
tar -C "$REPO" --exclude=.git --exclude=logs --exclude=outputs --exclude=recordings \
    --exclude='*.egg-info' --exclude=__pycache__ -czf "$TMP/as.tgz" Action_Space
ls -lh "$TMP/as.tgz" | awk '{print "   " $5}'

echo "== waking $N"
until "$AUTOR" run "$N" --timeout 300 -- echo awake 2>/dev/null | grep -q awake; do sleep 10; done

echo "== uploading"
"$AUTOR" cp "$TMP/as.tgz" "$N:/data/as.tgz"
"$AUTOR" run "$N" --timeout 300 -- sh -c 'rm -rf /data/as && mkdir -p /data/as && tar -xzf /data/as.tgz -C /data/as --strip-components=1 && ls /data/as'

echo "== node versions"
"$AUTOR" run "$N" --timeout 300 -- sh -c 'cd /workspace/isaaclab && ./isaaclab.sh -p -m pip list 2>/dev/null | grep -iE "^(skrl|stable.baselines3|torch|isaacsim) "'

echo "== tello.usd"
"$AUTOR" run "$N" --timeout 300 --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1 -- sh -c '
  test -f /data/assets/tello.usd && echo "already there" && exit 0
  mkdir -p /data/assets && cd /workspace/isaaclab &&
  ./isaaclab.sh -p scripts/tools/convert_urdf.py /data/as/simulation_infra/assets/tello/tello.urdf \
      /data/assets/tello.usd --merge-joints --headless 2>&1 | tail -3 && ls -la /data/assets/tello.usd'
echo "== done"
