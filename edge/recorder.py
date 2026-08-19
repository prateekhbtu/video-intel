"""
Segment indexing. REPLACES the Round 1 edge/recorder.py.

TWO BUGS, BOTH SILENT
    1. `seq` was a per-PROCESS counter starting at 0. Combined with
       `INSERT OR IGNORE` and a PRIMARY KEY of (camera_id, seq), every restart
       began overwriting sequence numbers that already existed, and the IGNORE
       swallowed each collision without a word. Restart the agent twice and
       you have silently dropped two segments' worth of index while the files
       sit on disk unreferenced, which is the worst of both worlds: you pay
       the storage and cannot retrieve the footage. seq is now seeded from
       MAX(seq) in the database.

    2. duration_s was hardcoded to 10 in the telemetry and never written to
       the table at all. It is the denominator completeness.py needs and the
       window retention.purge_subject() searches, so assuming it produced the
       2.33 coverage ratio and would have made a consent purge miss the
       segment containing the subject.

WHY THE DURATION IS PROBED AND NOT ASSUMED
    `ffmpeg -f segment -segment_time 10` cuts on the next keyframe, not at
    exactly 10 s, and the simulator's `-stream_loop -1` resets presentation
    timestamps at each wrap, which produces a short runt segment at every loop
    boundary. Assuming 10 s is how a metric ends up reading 233%.
"""
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime

from common import telem

_TS_RE = re.compile(r"(\d{8}_\d{6})")
SETTLE_S = 2.0            # a file still being written has a moving mtime


def _start_ts_from_name(path):
    """`-strftime 1` names segments %Y%m%d_%H%M%S, so the filename is a more
    trustworthy start time than any filesystem timestamp."""
    m = _TS_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def probe_duration(path, timeout=8):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            d = json.loads(out.stdout).get("format", {}).get("duration")
            if d is not None:
                return float(d)
    except Exception as e:
        telem.emit("probe_error", path=path, err=repr(e))
    return None


def _next_seq(conn, camera_id):
    row = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM segments WHERE camera_id=?",
                       (camera_id,)).fetchone()
    return (row[0] or 0) + 1


def watch(camera_id, site_id, segdir, conn, poll=2.0):
    seq = _next_seq(conn, camera_id)
    telem.emit("recorder_start", camera_id=camera_id, site_id=site_id,
               segdir=segdir, resume_seq=seq)
    seen = {r[0] for r in conn.execute(
        "SELECT path FROM segments WHERE camera_id=?", (camera_id,))}

    while True:
        try:
            for path in sorted(glob.glob(os.path.join(segdir, "*.mp4"))):
                if path in seen:
                    continue
                try:
                    if time.time() - os.path.getmtime(path) < SETTLE_S:
                        continue                       # still being written
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size == 0:
                    continue

                start_ts = _start_ts_from_name(path) or os.path.getmtime(path)
                dur = probe_duration(path)
                if dur is None:
                    dur = 0.0
                    telem.emit("segment_unprobed", camera_id=camera_id, path=path,
                               severity="warning")

                conn.execute(
                    "INSERT OR REPLACE INTO segments"
                    "(camera_id, seq, path, start_ts, end_ts, duration_s, bytes, "
                    " uploaded, legal_hold) "
                    "VALUES(?,?,?,?,?,?,?,"
                    "  COALESCE((SELECT uploaded   FROM segments WHERE path=?), 0),"
                    "  COALESCE((SELECT legal_hold FROM segments WHERE path=?), 0))",
                    (camera_id, seq, path, start_ts, start_ts + dur, dur, size,
                     path, path))
                conn.commit()
                seen.add(path)
                telem.emit("segment_write", camera_id=camera_id, site_id=site_id,
                           seq=seq, bytes=size, duration_s=round(dur, 3),
                           start_ts=round(start_ts, 3))
                seq += 1
        except Exception as e:
            telem.emit("recorder_error", camera_id=camera_id, err=repr(e))
        time.sleep(poll)
