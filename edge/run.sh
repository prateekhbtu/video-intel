#!/usr/bin/env bash
set -euo pipefail
SITE=$1
export RUN_ID=${RUN_ID:-$(date +%s | tail -c 6)}

cd /workspaces/video-intel

# Kick off the ffmpeg segmentation processes
./edge/run_ffmpeg_recorders.sh "$SITE"

# Explicitly list the cameras to avoid API race conditions
if [ "$SITE" == "a" ]; then
  CAMS="a_cam01,a_cam02,a_cam03,a_cam04,a_cam05,a_cam06"
else
  CAMS="b_cam01,b_cam02,b_cam03,b_cam04,b_cam05,b_cam06"
fi

# Start the agent python process
python -m edge.agent \
  --site "$SITE" \
  --cameras "$CAMS" \
  --db "$DATA/edge_${SITE}.db" \
  --segdir "$DATA/seg/${SITE}" \
  --log "$DATA/logs/edge_${SITE}.jsonl" \
  --api "${CLOUD_API:-http://127.0.0.1:8000}"
