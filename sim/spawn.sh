#!/usr/bin/env bash
# Publishes pre-encoded clips as live RTSP streams, split into two site cohorts.
set -euo pipefail
MTX="rtsp://127.0.0.1:8554"
SRC="${DATA}/sim"
LOGDIR="${DATA}/simlogs"
mkdir -p "$LOGDIR"

spawn () {
  local file="$1" cam="$2"
  ffmpeg -hide_banner -loglevel warning \
    -re -stream_loop -1 -i "$file" \
    -c copy -an \
    -f rtsp -rtsp_transport tcp "$MTX/$cam" \
    > "$LOGDIR/$cam.log" 2>&1 &
  echo "$cam -> $(basename "$file") pid $!"
}

n=0
for f in "$SRC"/*.mp4; do
  cam=$(basename "$f" .mp4)
  spawn "$f" "$cam"
  n=$((n+1))
done
echo "spawned $n cameras"
