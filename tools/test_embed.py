#!/usr/bin/env python3
"""
Embedding contract tests. Build plan section 3.2.

WHY THESE RUN BEFORE THE CASCADE IS TRUSTED
    An embedder that loads without error and returns 512 floats can still be
    useless. Three things have to hold before any downstream threshold means
    anything, and none of them are visible from "the model loaded":

      shape   normalised, 512-d, float32     -> cosine similarity is valid
      signal  same track scores above cross   -> the descriptor tracks identity
      margin  the gap is large enough to act  -> a threshold can separate them

    Property 6 is the one that matters. If same-track similarity is not
    clearly above cross-track similarity ON YOUR OWN FOOTAGE, the model has
    not transferred to your domain and no amount of threshold tuning
    downstream will fix it. Measuring this here is what stops Phase 4 from
    calibrating a threshold against noise.

    Properties 1-3 are cheap assertions that would have caught a wrong export
    (a model emitting logits instead of features still returns a float array
    of plausible size).
"""
import itertools
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("VI_ROOT", str(REPO))

import numpy as np
import cv2

from common import telem

telem.init(str(REPO / "data" / "logs" / "embed_test.jsonl"))

from edge import detect, embed, tracker

CAM = os.environ.get("VI_TEST_CAM", "a_cam01")
CLIP = REPO / "data" / "sim" / f"{CAM}.mp4"
OUT = REPO / "results" / "reid" / "embed_contract.json"
TARGET_VECS = 60
SEPARATION_MIN = 0.10


def main():
    if not CLIP.exists():
        sys.exit(f"clip absent: {CLIP}")

    d = detect.Detector(classes={1})
    e = embed.Embedder()
    t = tracker.Tracker(CAM, min_hits=3)

    print(f"  embedder kind={e.kind}  dim={e.dim}")
    print(f"  path={e.path}")

    cap = cv2.VideoCapture(str(CLIP))
    vecs, ids, quals = [], [], []
    frames = 0
    for i in itertools.count():
        ok, f = cap.read()
        if not ok or len(vecs) >= TARGET_VECS:
            break
        if i % 6:
            continue
        frames += 1
        conf, _ = d(f, CAM)
        active, _ended = t.update(conf, f.shape)
        for tr in active:
            v = e(f, tr)
            if v is not None:
                vecs.append(v)
                ids.append(tr.id)
                quals.append(e.crop_quality(tr, f.shape))
    cap.release()

    if len(vecs) < 4:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(
            {"status": "not measured",
             "reason": f"only {len(vecs)} embeddings from {frames} sampled "
                       f"frames of {CAM}; need at least 4 to compare "
                       f"within- and cross-track similarity"}, indent=2))
        sys.exit(f"\nFAIL: only {len(vecs)} vectors collected")

    V = np.stack(vecs)
    norms = np.linalg.norm(V, axis=1)

    same, diff = [], []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            (same if ids[i] == ids[j] else diff).append(float(V[i] @ V[j]))

    m_same = float(np.mean(same)) if same else float("nan")
    m_diff = float(np.mean(diff)) if diff else float("nan")
    sep = m_same - m_diff

    tests = [
        ("1. L2 norm == 1",
         bool(np.allclose(norms, 1.0, atol=1e-3)),
         f"min {norms.min():.6f} max {norms.max():.6f}"),
        ("2. dim == 512",
         V.shape[1] == 512, f"{V.shape[1]}"),
        ("3. dtype float32",
         V.dtype == np.float32, f"{V.dtype}"),
        ("4. same-track sim mean",
         len(same) > 0, f"{m_same:.3f} over {len(same)} pairs (want high)"),
        ("5. cross-track sim mean",
         len(diff) > 0, f"{m_diff:.3f} over {len(diff)} pairs (want lower)"),
        (f"6. separation > {SEPARATION_MIN}",
         bool(sep > SEPARATION_MIN),
         f"{sep:+.3f}  ({m_same:.3f} - {m_diff:.3f})"),
    ]

    print()
    for name, ok, detail in tests:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:26s} {detail}")

    n_pass = sum(1 for _, ok, _ in tests if ok)
    res = {
        "status": "measured",
        "camera": CAM,
        "embedder_kind": e.kind,
        "embedder_path": str(e.path),
        "frames_sampled": frames,
        "n_vectors": len(V),
        "n_tracks": len(set(ids)),
        "dim": int(V.shape[1]),
        "dtype": str(V.dtype),
        "l2_norm_min": round(float(norms.min()), 6),
        "l2_norm_max": round(float(norms.max()), 6),
        "same_track_pairs": len(same),
        "cross_track_pairs": len(diff),
        "same_track_sim_mean": round(m_same, 4),
        "cross_track_sim_mean": round(m_diff, 4),
        "separation": round(sep, 4),
        "separation_threshold": SEPARATION_MIN,
        "crop_quality_mean": round(float(np.mean(quals)), 4) if quals else None,
        "tests": [{"name": n, "pass": ok, "detail": det} for n, ok, det in tests],
        "tests_passed": n_pass,
        "tests_total": len(tests),
        "all_pass": n_pass == len(tests),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  {n_pass}/{len(tests)} contract tests passed")
    print(f"  wrote {OUT}")
    return 0 if n_pass == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
