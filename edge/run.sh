#!/usr/bin/env bash
# Launch one site's edge agent.
#
# WHAT CHANGED
#   `cd /workspaces/video-intel` is gone, and so is the hardcoded six-camera
#   list per site. The camera set now comes from edge/roster.yaml, which is the
#   same file the agent, the ffmpeg recorders and tools/verify.py read, so
#   there is exactly one place where "which cameras exist" is written down.
#   A second list that drifts from the first is how a site ends up analysing
#   cameras that were never spawned.
set -euo pipefail

SITE="${1:?usage: edge/run.sh <site>   e.g. edge/run.sh a}"
VI_ROOT="${VI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VI_ROOT
export DATA="${DATA:-$VI_ROOT/data}"
export PYTHONPATH="${PYTHONPATH:-$VI_ROOT}"
export RUN_ID="${RUN_ID:-$(date +%s | tail -c 6)}"

cd "$VI_ROOT"
mkdir -p "$DATA/logs" "$DATA/seg/$SITE"

# Migrate before applying the schema. schema.sql is CREATE TABLE IF NOT EXISTS
# throughout, which is a silent no-op on the Round 1 tables, so the columns
# Round 2 needs would never appear without this step.
python tools/migrate_edge.py "$DATA/edge_${SITE}.db"

./edge/run_ffmpeg_recorders.sh "$SITE"

exec python -m edge.agent \
  --site "$SITE" \
  --db "$DATA/edge_${SITE}.db" \
  --segdir "$DATA/seg/${SITE}" \
  --log "$DATA/logs/edge_${SITE}.jsonl" \
  --api "${CLOUD_API:-http://127.0.0.1:8000}"
