#!/usr/bin/env python3
"""
Cross-camera evaluation on our own topology. Build plan section 4.5.

Market-1501 tells you the MODEL is good. This tells you the SYSTEM is good,
which is a different and harder claim: it exercises the real detector, the
real tracker, the real crop quality, the real descriptor averaging and the
real resolver, against ground truth derived from camera offsets we control.

WHAT IS MEASURED
    correctly_linked     GT pairs the resolver put under one subject_id
    missed_links         GT pairs it left split (recall loss)
    over_merged          distinct GT identities collapsed into one subject
    link_recall          correctly_linked / GT pairs

KNOWN LIMITATION, STATED NOT HIDDEN
    The a_cam01|a_cam02 crop-split pair — the exact mechanism the topology was
    built to demonstrate — contributed ZERO ground-truth pairs on this
    footage. Ground truth comes from the other four camera pairs. So this
    measures cross-camera linking under time offset and illumination /
    resolution shift, but NOT the left-half/right-half spatial handoff. Any
    claim about handoff specifically is unsupported by this evidence.
"""
import argparse
import collections
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import psycopg2

GT = REPO / "results" / "identity" / "ground_truth.json"
OUT = REPO / "results" / "identity" / "crosscam_eval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ver", default="reid-osnet-x025-fp32.onnx")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    if not GT.exists():
        sys.exit(f"missing ground truth: {GT}")
    gt = json.loads(GT.read_text())
    pairs = gt["pairs"]

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    # Join on sighting_id, not (camera_id, track_id): track_id restarts at
    # zero on every agent start, so the composite key collides across runs and
    # silently matches tracklets that have nothing to do with each other.
    cur.execute("SELECT sighting_id, camera_id, track_id, subject_id FROM sightings "
                "WHERE subject_id IS NOT NULL AND model_ver = %s", (a.model_ver,))
    rows = cur.fetchall()
    assigned = {r[0]: r[3] for r in rows}
    assigned_bytrack = {(r[1], r[2]): r[3] for r in rows}

    def lookup(p, side):
        sid = p.get(f"sighting_{side}")
        if sid is not None:
            return assigned.get(sid)
        return assigned_bytrack.get((p[f"cam_{side}"], p[f"track_{side}"]))

    cur.execute("SELECT COUNT(*), COUNT(subject_id) FROM sightings WHERE model_ver=%s",
                (a.model_ver,))
    n_sight, n_assigned = cur.fetchone()

    tp = fn = missing = 0
    per_pair = collections.Counter()
    for p in pairs:
        sa, sb = lookup(p, "a"), lookup(p, "b")
        key = f"{p['cam_a']}|{p['cam_b']}"
        if sa is None or sb is None:
            missing += 1
            per_pair[f"{key}:unresolved"] += 1
            continue
        if sa == sb:
            tp += 1
            per_pair[f"{key}:linked"] += 1
        else:
            fn += 1
            per_pair[f"{key}:split"] += 1

    # Over-merge: one subject_id spanning tracklets that ground truth says are
    # DIFFERENT identities. This is the failure that matters most — a false
    # link corrupts the identity graph, where a missed link merely loses one.
    gtmap = {}
    for p in pairs:
        for side in ("a", "b"):
            key = p.get(f"sighting_{side}") or (p[f"cam_{side}"], p[f"track_{side}"])
            gtmap[key] = p["gid"]
    groups = collections.defaultdict(set)
    for key, subj in list(assigned.items()) + list(assigned_bytrack.items()):
        if key in gtmap:
            groups[subj].add(gtmap[key])
    over_merged = sum(1 for sid, gids in groups.items() if len(gids) > 1)

    cur.execute("SELECT COUNT(*) FROM identities")
    n_identities = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM adjudications WHERE kind='identity_review'")
    n_review = cur.fetchone()[0]

    keyed_by = "sighting_id" if any(p.get("sighting_a") for p in pairs) else "(camera,track_id)"
    res = {
        "label": a.label,
        "gt_joined_on": keyed_by,
        "model_ver": a.model_ver,
        "gt_pairs": len(pairs),
        "gt_identities": gt["n_identities"],
        "sightings_total": n_sight,
        "sightings_assigned": n_assigned,
        "unresolved_tracklets": missing,
        "correctly_linked": tp,
        "missed_links": fn,
        "over_merged_identities": over_merged,
        "link_recall": round(tp / max(1, tp + fn), 4),
        "link_recall_of_all_gt": round(tp / max(1, len(pairs)), 4),
        "identities_created": n_identities,
        "review_queue_depth": n_review,
        "per_camera_pair": dict(per_pair),
        "caveat": ("a_cam01|a_cam02 (the crop-split handoff) contributed 0 GT "
                   "pairs on this footage, so spatial handoff is NOT evaluated "
                   "here; only time-offset linking across the other pairs is."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Keep every run, so the operating-point comparison is auditable.
    hist_p = OUT.parent / "crosscam_eval_runs.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    hist.append(res)
    hist_p.write_text(json.dumps(hist, indent=2))
    OUT.write_text(json.dumps(res, indent=2))

    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
