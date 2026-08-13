#!/usr/bin/env bash
SITE=$1
CAMERAS=$(curl -s localhost:9997/v3/paths/list | jq -r ".items[].name | select(startswith(\"${SITE}_\"))")

for cam in $CAMERAS; do
    mkdir -p "$DATA/seg/$SITE/$cam"
    ffmpeg -rtsp_transport tcp -i "rtsp://127.0.0.1:8554/$cam" \
      -c copy -f segment -segment_time 10 \
      -segment_format mp4 -reset_timestamps 1 -strftime 1 \
      "$DATA/seg/$SITE/$cam/%Y%m%d_%H%M%S.mp4" -y </dev/null >/dev/null 2>&1 &
done
