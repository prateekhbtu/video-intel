#!/usr/bin/env python3
"""
Cross-camera ground truth, derived from the topology rather than annotated.

THE IDEA (build plan section 1.3)
    Hand-annotating cross-camera identity is slow and is the reason most
    candidates answer PS-3 Q3.1 with adjectives instead of a Rank-1 number.
    We annotate BY CONSTRUCTION instead: sim/build_topology.sh derives four
    cameras from ONE source video at known time offsets, so two tracklets are
    the same person exactly when their appearance windows coincide once each
    is mapped back to source time. You know the labels because you built them.

TWO CORRECTIONS TO THE SNIPPET IN THE BUILD PLAN, BOTH DELIBERATE
    1. It reads first_ts / last_ts off `track_end` telemetry. edge/tracker.py
       emits camera_id, track_id, dwell_s and hits on that stage, so those
       keys do not exist and the original raises KeyError on the first record.
       Tracklet windows are read from the `sightings` table instead, where
       `ts` and `last_ts` are real columns, with a JSONL fallback that
       reconstructs first_ts as (record ts - dwell_s). edge/tracker.py is not
       modified: the defect is in the reader, not the writer.

    2. The offset sign is inverted. build_topology.sh builds a_cam02 with
       `-ss 12`, so that clip BEGINS twelve seconds into the source. At wall
       clock T0+W, a_cam01 shows source W while a_cam02 shows source W+12, so
       a person at source time S reaches a_cam02 twelve seconds EARLIER in
       wall clock, not later.

           a_cam01 wall = T0 + S           a_cam02 wall = T0 + S - 12
           source_time  = (wall - T0) + OFFSET

       Matching therefore compares (first_ts + OFFSET) across cameras. The
       plan subtracts, which places the two cameras 24 s apart and fails every
       comparison against TOL=2.5. The plan's prose ("enters cam02 twelve
       seconds later") agrees with its own sign; the ffmpeg command that
       actually produced the clips disagrees with both.

    Rather than assert which is right, --sign auto MEASURES it. Over all
    cross-camera tracklet pairs the delta distribution is broad noise except
    at the true offset, where genuine correspondences pile up. The mode of
    that histogram is the empirical offset, and it is reported next to the
    configured one whichever sign you choose.

WHAT THIS DELIBERATELY DOES NOT CLAIM
    The docstring in the plan says windows align "and their spatial handoff is
    consistent". The spatial half is NOT implemented, because `sightings` does
    not carry box geometry and inventing it would be exactly the kind of
    unmeasured claim this build plan exists to prevent. Temporal alignment
    under a known offset is what is verified here; the handoff direction is
    stated in topology.json and left as an assumption. Say so when citing it.

VALIDITY WINDOW
    Each camera's clip has a different duration (D, D-12, D-30, D-45), so
    `-stream_loop -1` wraps them at different periods and the phase
    relationship breaks after the SHORTEST clip wraps. Ground truth is only
    sound inside that first period, which this script enforces and reports
    rather than quietly emitting pairs it cannot justify.

USAGE
    python tools/build_gt.py                                  # DB, defaults
    python tools/build_gt.py --log data/logs/edge_a.jsonl     # JSONL fallback
    python tools/build_gt.py --sign auto --tol 3.0
"""
import argparse
import collections
import itertools
import json
import math
import pathlib
import sqlite3
import sys

# Offsets used when data/sim/topology.json is absent. These mirror the -ss
# values in sim/build_topology.sh; topology.json is preferred because it is
# written by the script that actually encoded the clips.
DEFAULT_OFFSETS = {"a_cam01": 0.0, "a_cam02": 12.0, "a_cam03": 30.0, "a_cam04": 45.0}


# --------------------------------------------------------------------------
# union-find. The plan's grouping (`gid.get(k1) or gid.get(k2) or next_gid`)
# silently drops a group when two tracklets that already belong to DIFFERENT
# groups are linked: it keeps one id and abandons the other's members. With
# four cameras a single identity spans up to six pairs, so that path is hit
# constantly and inflates the identity count. Proper union-find makes the
# transitive closure correct, which matters because n_identities is a
# denominator in the Phase 4 evaluation.
# --------------------------------------------------------------------------
class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return ra


def load_topology(path):
    p = pathlib.Path(path)
    if not p.exists():
        return DEFAULT_OFFSETS, None, f"{path} absent, using built-in defaults"
    d = json.loads(p.read_text())
    offs = {k: float(v) for k, v in (d.get("offsets_s") or {}).items()}
    return (offs or DEFAULT_OFFSETS), d.get("source_duration_s"), f"read {path}"


def tracklets_from_db(db_path, cameras=None):
    """`sightings` is the authoritative tracklet record: one row per confirmed
    tracklet that closed, written before the descriptor check, so tracklets
    with no usable crop are present too."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT camera_id, track_id, ts, last_ts FROM sightings "
        "WHERE ts IS NOT NULL AND last_ts IS NOT NULL").fetchall()
    conn.close()
    out = collections.defaultdict(list)
    for cam, tid, first_ts, last_ts in rows:
        if cameras and cam not in cameras:
            continue
        out[cam].append({"track_id": tid, "first_ts": float(first_ts),
                         "last_ts": float(last_ts)})
    return out


def tracklets_from_log(paths, cameras=None):
    """Fallback. `track_end` carries dwell_s and the record's own ts, which is
    emitted at close, so first_ts = ts - dwell_s reconstructs the window."""
    out = collections.defaultdict(list)
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("stage") != "track_end":
                    continue
                cam = d.get("camera_id")
                if not cam or (cameras and cam not in cameras):
                    continue
                last = float(d["ts"])
                out[cam].append({"track_id": d["track_id"],
                                 "first_ts": last - float(d.get("dwell_s", 0.0)),
                                 "last_ts": last})
    return out


def measure_offset(t1, t2, span=180.0, bin_s=1.0):
    """Empirical offset between two cameras, with no correspondence assumed.

    Genuine matches concentrate the (start_a - start_b) delta at the true
    offset; unrelated pairs spread out. Returns the modal delta, its count,
    and the ratio of that peak to the median bin, which is the honest measure
    of whether a peak exists at all."""
    hist = collections.Counter()
    for a in t1:
        for b in t2:
            delta = a["first_ts"] - b["first_ts"]
            if abs(delta) <= span:
                hist[round(delta / bin_s) * bin_s] += 1
    if not hist:
        return None, 0, 0.0
    mode, count = hist.most_common(1)[0]
    vals = sorted(hist.values())
    median = vals[len(vals) // 2] or 1
    return mode, count, round(count / median, 2)


def build(tracks, offsets, sign, tol, window=None, t0=None):
    """A pair is the same identity when BOTH ends of the window align in
    source time. Requiring both ends, not just the start, is what stops a
    long tracklet on one camera from absorbing several short ones on another."""
    norm = {}
    for cam, tl in tracks.items():
        off = offsets.get(cam, 0.0) * sign
        keep = []
        for t in tl:
            if window is not None and t0 is not None and t["first_ts"] > t0 + window:
                continue
            keep.append({**t, "s": t["first_ts"] + off, "e": t["last_ts"] + off})
        norm[cam] = keep

    dsu, pairs = DSU(), []
    for (c1, t1), (c2, t2) in itertools.combinations(sorted(norm.items()), 2):
        for a in t1:
            for b in t2:
                if abs(a["s"] - b["s"]) < tol and abs(a["e"] - b["e"]) < tol:
                    k1, k2 = (c1, a["track_id"]), (c2, b["track_id"])
                    dsu.union(k1, k2)
                    pairs.append({"cam_a": c1, "track_a": a["track_id"],
                                  "cam_b": c2, "track_b": b["track_id"],
                                  "d_start": round(a["s"] - b["s"], 3),
                                  "d_end": round(a["e"] - b["e"], 3)})

    roots = {}
    for p in pairs:
        r = dsu.find((p["cam_a"], p["track_a"]))
        p["gid"] = roots.setdefault(r, len(roots) + 1)
    return pairs, len(roots), norm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="JSONL logs (fallback source)")
    ap.add_argument("--db", default="data/edge_a.db")
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--topology", default="data/sim/topology.json")
    ap.add_argument("--out", default="results/identity/ground_truth.json")
    ap.add_argument("--tol", type=float, default=2.5,
                    help="seconds of slack for tracker start/stop jitter")
    ap.add_argument("--sign", choices=("auto", "plus", "minus"), default="auto")
    ap.add_argument("--cameras", default="a_cam01,a_cam02,a_cam03,a_cam04")
    ap.add_argument("--no-window", action="store_true",
                    help="do not restrict to the first loop period")
    a = ap.parse_args()

    cameras = {c.strip() for c in a.cameras.split(",") if c.strip()}
    offsets, src_dur, topo_note = load_topology(a.topology)

    log_paths = list(a.log) + list(a.logs)
    if log_paths:
        tracks, source = tracklets_from_log(log_paths, cameras), f"logs {log_paths}"
    elif pathlib.Path(a.db).exists():
        tracks, source = tracklets_from_db(a.db, cameras), f"sightings in {a.db}"
    else:
        sys.exit(f"no source: {a.db} absent and no --log given")

    tracks = {c: t for c, t in tracks.items() if t}
    n_tracks = sum(len(v) for v in tracks.values())
    print(f"  topology : {topo_note}")
    print(f"  offsets  : {offsets}")
    print(f"  source   : {source}")
    print(f"  tracklets: {n_tracks} across {len(tracks)} cameras "
          f"{ {c: len(v) for c, v in sorted(tracks.items())} }")

    if len(tracks) < 2:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(
            {"pairs": [], "n_identities": 0, "n_tracklets": n_tracks,
             "error": "fewer than two cameras produced tracklets"}, indent=2))
        sys.exit(f"\nFAIL: only {len(tracks)} camera(s) produced tracklets; "
                 f"cross-camera ground truth needs at least two. Check that "
                 f"roster.yaml analyses a_cam01..a_cam04.")

    # ---- validity window ------------------------------------------------
    all_starts = [t["first_ts"] for v in tracks.values() for t in v]
    t0 = min(all_starts)
    window = None
    if not a.no_window and src_dur:
        window = float(src_dur) - max(offsets.values())
        print(f"  window   : first {window:.0f}s after t0 (shortest clip wraps "
              f"there; ground truth is unsound past it)")

    # ---- empirical offsets, measured not assumed ------------------------
    print("\n  measured cross-camera offsets (mode of the delta histogram):")
    measured = {}
    for c1, c2 in itertools.combinations(sorted(tracks), 2):
        mode, count, peak = measure_offset(tracks[c1], tracks[c2])
        measured[f"{c1}|{c2}"] = {"mode_s": mode, "count": count, "peak_ratio": peak}
        exp_plus = -(offsets.get(c1, 0) - offsets.get(c2, 0))
        print(f"    {c1} vs {c2:9s} mode={mode:+7.1f}s  n={count:3d}  "
              f"peak/median={peak:4.1f}x   (+sign predicts {exp_plus:+.1f}s, "
              f"-sign predicts {-exp_plus:+.1f}s)")

    # ---- sign resolution -------------------------------------------------
    if a.sign == "auto":
        results = {}
        for name, s in (("plus", +1.0), ("minus", -1.0)):
            p, n, _ = build(tracks, offsets, s, a.tol, window, t0)
            results[name] = (len(p), n, s)
        sign_name = max(results, key=lambda k: results[k][0])
        sign = results[sign_name][2]
        print(f"\n  sign     : auto -> {sign_name} "
              f"(plus={results['plus'][0]} pairs, minus={results['minus'][0]} pairs)")
        if results["plus"][0] and results["minus"][0]:
            print("             WARNING: both signs yield pairs; the topology "
                  "may be ambiguous. Inspect the histogram above.")
    else:
        sign = +1.0 if a.sign == "plus" else -1.0
        sign_name = a.sign
        print(f"\n  sign     : {sign_name} (forced)")

    pairs, n_ids, norm = build(tracks, offsets, sign, a.tol, window, t0)

    # ---- tolerance sensitivity, so the number is not a lucky threshold ---
    print("\n  tolerance sensitivity:")
    for t in (1.0, 1.5, 2.5, 4.0, 6.0):
        p, n, _ = build(tracks, offsets, sign, t, window, t0)
        mark = "  <- chosen" if abs(t - a.tol) < 1e-9 else ""
        print(f"    tol={t:4.1f}s  pairs={len(p):4d}  identities={n:4d}{mark}")

    by_pair = collections.Counter(f"{p['cam_a']}|{p['cam_b']}" for p in pairs)
    out = {
        "pairs": [{k: p[k] for k in ("cam_a", "track_a", "cam_b", "track_b", "gid")}
                  for p in pairs],
        "n_identities": n_ids,
        "n_tracklets": n_tracks,
        "n_tracklets_in_window": sum(len(v) for v in norm.values()),
        "provenance": {
            "source": source, "topology": topo_note, "offsets_s": offsets,
            "sign": sign_name, "tol_s": a.tol,
            "window_s": window, "t0": t0,
            "measured_offsets": measured,
            "pairs_by_camera_pair": dict(by_pair),
            "spatial_handoff_verified": False,
            "note": "temporal alignment under known offsets only; the spatial "
                    "handoff asserted in topology.json is NOT verified here",
        },
    }
    p_out = pathlib.Path(a.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2))

    print(f"\n  {len(pairs)} cross-camera pairs, {n_ids} identities")
    print(f"  by camera pair: {dict(by_pair)}")
    print(f"  wrote {p_out}")

    if len(pairs) < 20:
        print(f"\n  GATE 1 NOT MET: {len(pairs)} pairs, need >= 20.")
        print("  Likely causes, in order: too few cameras analysed in "
              "roster.yaml; the run was too short; VI_TARGET_FPS so low that "
              "tracklets never reach min_hits=3; or the source video has few "
              "people crossing the crop boundary.")
        return 1
    print(f"\n  GATE 1 pair count met: {len(pairs)} >= 20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
