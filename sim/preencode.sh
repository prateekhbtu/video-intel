#!/usr/bin/env bash
# Normalise dataset clips into a single controlled camera profile, once.
set -euo pipefail
SRC="${DATA}/datasets/shanghaitech/training/videos"
OUT="${DATA}/sim"
mkdir -p "$OUT"

i=0
for f in "$SRC"/*.avi; do
  i=$((i+1))
  [ $i -gt 12 ] && break
  if [ $i -le 6 ]; then cam=$(printf 'a_cam%02d' $i)
  else cam=$(printf 'b_cam%02d' $((i-6))); fi
  [ -f "$OUT/$cam.mp4" ] && { echo "$cam exists, skip"; continue; }
  ffmpeg -hide_banner -loglevel warning -y \
    -i "$f" \
    -c:v libx264 -preset veryfast \
    -b:v 900k -maxrate 900k -bufsize 1800k \
    -vf "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=25" \
    -g 50 -keyint_min 50 -sc_threshold 0 \
    -pix_fmt yuv420p -an \
    "$OUT/$cam.mp4"
  echo "encoded $cam <- $(basename "$f")"
done
du -sh "$OUT"
