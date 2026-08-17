#!/usr/bin/env bash
# Build a 4-camera site where cameras SHARE identities via time-offset crops.
#
# WHY: the Round 1 simulator runs 12 independent looping clips. No identity
# ever appears in two cameras, so cross-camera ReID has nothing to re-identify
# and PS-3 cannot be evaluated at all.
#
# HOW: one long source video feeds four cameras. cam01 shows the right half at
# t=0; cam02 shows the left half at t=12s, so a person leaving cam01 on the
# right "enters" cam02 on the left twelve seconds later. cam03 and cam04 add
# controlled illumination and resolution shift, which is how Phase 4 measures
# embedding drift instead of asserting it.
#
# Ground truth is free: you know the offsets because you built them.
set -euo pipefail

: "${DATA:?source .env first}"
SRCDIR="${1:-$DATA/datasets/shanghaitech/training/videos}"
OUT="$DATA/sim"
WORK="$DATA/work"
mkdir -p "$OUT" "$WORK"

# ---------------------------------------------------------------------------
# 1. Concatenate the source clips into one long take.
#    Individual ShanghaiTech clips run 20 to 32 seconds. A 12 second offset
#    against a 32 second clip leaves only 20 seconds of usable overlap, and the
#    loop wrap every 32 seconds is what resets PTS and corrupts segmentation.
#    Concatenating all 12 gives roughly 6 minutes and far fewer wraps.
# ---------------------------------------------------------------------------
echo "==> concatenating source clips from $SRCDIR"
: > "$WORK/concat.txt"
shopt -s nullglob
for f in "$SRCDIR"/*.avi "$SRCDIR"/*.mp4; do
  echo "file '$(readlink -f "$f")'" >> "$WORK/concat.txt"
done
shopt -u nullglob
test -s "$WORK/concat.txt" || { echo "no source video found in $SRCDIR"; exit 1; }
echo "    $(wc -l < "$WORK/concat.txt") clips"

ffmpeg -y -hide_banner -loglevel error \
  -f concat -safe 0 -i "$WORK/concat.txt" \
  -c:v libx264 -preset veryfast -crf 20 -an "$WORK/source_long.mp4"

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/source_long.mp4")
echo "    source_long.mp4  ${DUR}s"
awk -v d="$DUR" 'BEGIN{ if (d+0 < 90) print "    WARNING: under 90s, offsets will overlap poorly" }'

# ---------------------------------------------------------------------------
# 2. Derive four cameras with known offsets and known transforms.
#    Encode profile is identical to Round 1 (640x360, 900 kbps, GOP 50) so the
#    bandwidth numbers stay comparable to the baseline you captured in Phase 0.
# ---------------------------------------------------------------------------
enc() { ffmpeg -y -hide_banner -loglevel error "$@" \
          -c:v libx264 -preset veryfast -b:v 900k -maxrate 900k -bufsize 1800k \
          -g 50 -pix_fmt yuv420p -an; }

echo "==> a_cam01  right half, t+0        (person exits frame right)"
enc -i "$WORK/source_long.mp4" -vf "crop=iw/2:ih:iw/2:0,scale=640:360" "$OUT/a_cam01.mp4"

echo "==> a_cam02  left half,  t+12       (same person enters frame left)"
enc -ss 12 -i "$WORK/source_long.mp4" -vf "crop=iw/2:ih:0:0,scale=640:360" "$OUT/a_cam02.mp4"

echo "==> a_cam03  full,       t+30, dim  (illumination domain shift)"
enc -ss 30 -i "$WORK/source_long.mp4" \
    -vf "scale=640:360,eq=brightness=-0.12:contrast=1.15:saturation=0.85" "$OUT/a_cam03.mp4"

echo "==> a_cam04  full,       t+45, soft (resolution domain shift)"
enc -ss 45 -i "$WORK/source_long.mp4" -vf "scale=320:180,scale=640:360" "$OUT/a_cam04.mp4"

# ---------------------------------------------------------------------------
# 3. Record the offsets so tools/build_gt.py can derive ground truth.
#    This file IS the label source. Do not edit it by hand.
# ---------------------------------------------------------------------------
cat > "$OUT/topology.json" <<JSON
{
  "source": "$WORK/source_long.mp4",
  "source_duration_s": $DUR,
  "offsets_s": {"a_cam01": 0, "a_cam02": 12, "a_cam03": 30, "a_cam04": 45},
  "transforms": {
    "a_cam01": "crop_right_half",
    "a_cam02": "crop_left_half",
    "a_cam03": "illumination_shift",
    "a_cam04": "resolution_shift"
  },
  "built_at": "$(date -Iseconds)"
}
JSON

echo
printf "%-12s %10s %8s %s\n" CAMERA DURATION SIZE RESOLUTION
for f in "$OUT"/a_cam0[1-4].mp4; do
  printf "%-12s %10s %8s %s\n" \
    "$(basename "$f" .mp4)" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" | cut -c1-6)s" \
    "$(du -h "$f" | cut -f1)" \
    "$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$f")"
done
echo
echo "topology written to $OUT/topology.json"