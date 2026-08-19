"""
Recording coverage SLI. REPLACES the Round 1 edge/completeness.py.

THE MEASUREMENT WAS BROKEN, NOT THE SYSTEM
    Round 1 computed

        expected = window // seg_s          # 60 // 10 = 6, a constant
        actual   = COUNT(*) FROM segments WHERE start_ts > cutoff
        ratio    = actual / expected

    and reported 1.667 to 2.333 on all 168 readings. A coverage ratio above
    1.0 is not "extra good coverage", it is arithmetic that cannot detect
    loss: counting FILES against an assumed file length says nothing about
    how many SECONDS were captured. ffmpeg cuts on the next keyframe rather
    than at exactly 10 s, and the simulator's `-stream_loop -1` resets
    timestamps at every wrap, emitting a runt segment each time. Count those
    runts as whole segments and you manufacture coverage out of nothing.

WHAT IT MEASURES NOW
    seconds of media actually recorded, over seconds of wall clock elapsed,
    with each segment clipped to the observation window so a segment that
    straddles the boundary contributes only its overlapping part.

        ratio = sum(overlap(segment, window)) / window_seconds

    That is bounded by construction: you cannot record more seconds of one
    camera than have elapsed. So when it DOES exceed 1.0 the sensor itself is
    broken, and that is worth paging someone about rather than quietly
    reporting. Hence the explicit bounds assertion and the completeness_fault
    record, which is the alert PS-4 Q4.2a is really asking for: the one that
    fires when the MEASUREMENT is unhealthy, not when the system is.
"""
import time

from common import telem

LO, HI = 0.0, 1.05         # 5% slack for probe rounding and clock jitter


def coverage(conn, camera_id, t0, t1):
    """Seconds of media recorded inside [t0, t1], clipped at both ends."""
    rows = conn.execute(
        "SELECT start_ts, duration_s FROM segments "
        "WHERE camera_id=? AND start_ts < ? AND start_ts + COALESCE(duration_s,0) > ?",
        (camera_id, t1, t0)).fetchall()
    covered = 0.0
    for start, dur in rows:
        end = start + (dur or 0.0)
        covered += max(0.0, min(end, t1) - max(start, t0))
    return covered, len(rows)


def completeness_loop(camera_id, site_id, conn, window=60, interval=None):
    interval = interval or window
    # Anchor to a real start instant. The first window is skipped rather than
    # reported, because a partial window at boot is not a coverage failure.
    t_prev = time.time()
    time.sleep(interval)

    while True:
        try:
            t_now = time.time()
            elapsed = t_now - t_prev
            covered, n_seg = coverage(conn, camera_id, t_prev, t_now)
            ratio = covered / elapsed if elapsed > 0 else 0.0

            telem.emit("completeness", camera_id=camera_id, site_id=site_id,
                       ratio=round(ratio, 3),
                       covered_s=round(covered, 1),
                       window_s=round(elapsed, 1),
                       segments=n_seg)

            if not (LO <= ratio <= HI):
                # The invariant, asserted where it is cheapest to assert.
                telem.emit("completeness_fault", camera_id=camera_id,
                           site_id=site_id, ratio=round(ratio, 3),
                           covered_s=round(covered, 1),
                           window_s=round(elapsed, 1), segments=n_seg,
                           bound_lo=LO, bound_hi=HI, severity="critical",
                           note="coverage outside physical bounds: the SENSOR "
                                "is broken, not the recording")
            elif ratio < 0.95:
                telem.emit("recording_gap", camera_id=camera_id, site_id=site_id,
                           ratio=round(ratio, 3),
                           missing_s=round(elapsed - covered, 1),
                           severity="warning")

            t_prev = t_now
        except Exception as e:
            telem.emit("completeness_error", camera_id=camera_id, err=repr(e))
        time.sleep(interval)
