#!/usr/bin/env bash
# Start one segmenting ffmpeg per camera for a site.
#
# Cameras come from edge/roster.yaml, not from MediaMTX's path list. The old
# version asked the RTSP server which paths existed, which meant a camera that
# had not published yet was silently never recorded — and "silently never
# recorded" is exactly the failure that left edge_b.db empty. Reading the
# roster makes the intended set explicit, so a camera that fails to appear
# shows up as a completeness fault instead of as nothing at all.
set -euo pipefail

SITE="${1:?usage: run_ffmpeg_recorders.sh <site>}"
VI_ROOT="${VI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-$VI_ROOT/data}"
ROSTER="${VI_ROSTER:-$VI_ROOT/edge/roster.yaml}"
RTSP_BASE="${VI_RTSP_BASE:-rtsp://127.0.0.1:8554}"
SEG_S="${VI_SEGMENT_SECONDS:-10}"

CAMERAS=$(python - "$ROSTER" "$SITE" <<'PY'
import sys, yaml
roster = yaml.safe_load(open(sys.argv[1])) or {}
site = sys.argv[2]
cams = (roster.get("sites", {}).get(site, {}) or {}).get("cameras")
if not cams:
    cams = [c for c in (roster.get("analyse") or []) if c.startswith(f"{site}_")]
print(" ".join(cams))
PY
)

test -n "$CAMERAS" || { echo "no cameras for site '$SITE' in $ROSTER" >&2; exit 1; }

for cam in $CAMERAS; do
  outdir="$DATA/seg/$SITE/$cam"
  mkdir -p "$outdir"
  # -c copy: no re-encode, so recording costs almost no CPU and cannot compete
  # with inference for the box. Recording and analysis must not share a budget.
  ffmpeg -nostdin -hide_banner -loglevel error \
    -rtsp_transport tcp -timeout 5000000 -i "$RTSP_BASE/$cam" \
    -c copy -f segment -segment_time "$SEG_S" \
    -segment_format mp4 -reset_timestamps 1 -strftime 1 \
    "$outdir/%Y%m%d_%H%M%S.mp4" -y \
    </dev/null >"$DATA/logs/rec_${cam}.log" 2>&1 &
  echo "recording $cam -> $outdir (pid $!)"
done
