#!/usr/bin/env python3
"""
Threshold calibration sweep. Build plan section 4.4.

THE OUTPUT IS THE ANSWER TO "HOW DID YOU CHOOSE YOUR THRESHOLD"
    tau and delta are not preferences. They are an operating point on a
    measured surface, and the surface is what makes the choice defensible.

    tau    accept floor on top-1 similarity
    delta  required gap between top-1 and top-2
    below tau_low -> NEW identity, everything else -> REVIEW

WHY DELTA EXISTS AT ALL
    A single threshold on top-1 cannot tell "this is David" from "this looks
    like David". Both score high. What separates them is the margin to the
    runner-up. Phase 2 measured margin_p05 = 0.0009 on this model, meaning
    one query in twenty is separated from the WRONG identity by under a
    thousandth of a cosine unit. Any system with one threshold answers those
    confidently and is wrong on a large share of them.

    So the sweep reports four rates, not one accuracy:
        precision         of the queries we auto-decided, how many were right
        recall            of ALL queries, how many did we get right
        auto_decide_rate  how much work the system did without a human
        review_rate       how much work it handed to a human

    Raising delta buys precision by moving queries into review. That trade is
    the product decision, and it is visible in the CSV rather than hidden in
    a constant.

PROTOCOL
    Standard Market-1501: a gallery item is excluded when it shares BOTH
    identity and camera with the query, because matching the same person on
    the same camera is trivial and inflates every number.
"""
import argparse
import csv
import itertools
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("VI_ROOT", str(REPO))

import glob
import numpy as np
import cv2
import onnxruntime as ort

from common import telem

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ROOT = REPO / "data" / "datasets" / "Market-1501-v15.09.15"
OUTDIR = REPO / "results" / "identity"


def meta(p):
    n = pathlib.Path(p).name.split("_")
    return int(n[0]), int(n[1][1])          # pid, camid


def embed_all(sess, inp, files, bs=32):
    out = []
    for i in range(0, len(files), bs):
        batch = []
        for f in files[i:i + bs]:
            im = cv2.imread(f)
            if im is None:
                continue
            im = cv2.resize(im, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
            batch.append(((im - MEAN) / STD).transpose(2, 0, 1))
        if not batch:
            continue
        x = np.ascontiguousarray(np.stack(batch), dtype=np.float32)
        v = sess.run(None, {inp: x})[0]
        out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12))
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=str(REPO / "data/models/registry/reid-osnet-x025-fp32.onnx"))
    ap.add_argument("--n-query", type=int, default=1200)
    ap.add_argument("--n-gallery", type=int, default=8000)
    ap.add_argument("--tau-low", type=float, default=0.55)
    ap.add_argument("--target-precision", type=float, default=0.99)
    a = ap.parse_args()

    telem.init(str(REPO / "data" / "logs" / "calibrate.jsonl"))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    q = sorted(glob.glob(str(ROOT / "query" / "*.jpg")))[:a.n_query]
    g = [f for f in sorted(glob.glob(str(ROOT / "bounding_box_test" / "*.jpg")))
         if not pathlib.Path(f).name.startswith("-1")][:a.n_gallery]
    if not q or not g:
        sys.exit(f"Market-1501 not found under {ROOT}")
    print(f"  model   {pathlib.Path(a.model).name}")
    print(f"  query   {len(q)}   gallery {len(g)}")

    t0 = time.time()
    Q = embed_all(sess, inp, q)
    G = embed_all(sess, inp, g)
    qm = np.array([meta(f) for f in q])
    gm = np.array([meta(f) for f in g])
    print(f"  embedded in {time.time()-t0:.1f}s")

    # Precompute top-2 per query once; the sweep is then pure arithmetic.
    tops = []
    for i in range(len(q)):
        sims = G @ Q[i]
        valid = ~((gm[:, 0] == qm[i, 0]) & (gm[:, 1] == qm[i, 1]))
        sv, gp = sims[valid], gm[valid, 0]
        o = np.argsort(-sv)[:2]
        tops.append((float(sv[o[0]]), float(sv[o[1]]),
                     int(gp[o[0]]) == int(qm[i, 0])))
    n = len(tops)
    margins = np.array([t1 - t2 for t1, t2, _ in tops])
    print(f"  margin: mean {margins.mean():.4f}  p05 {np.percentile(margins,5):.4f}"
          f"  p50 {np.percentile(margins,50):.4f}")

    rows = []
    for tau, delta in itertools.product(np.arange(0.50, 0.90, 0.025),
                                        np.arange(0.00, 0.20, 0.01)):
        tp = fp = rev = new = 0
        for t1, t2, correct in tops:
            if t1 >= tau and (t1 - t2) >= delta:
                tp += correct
                fp += not correct
            elif t1 < a.tau_low:
                new += 1
            else:
                rev += 1
        dec = tp + fp
        rows.append({"tau": round(float(tau), 3), "delta": round(float(delta), 3),
                     "precision": round(tp / dec, 4) if dec else None,
                     "recall": round(tp / n, 4),
                     "auto_decide_rate": round(dec / n, 4),
                     "review_rate": round(rev / n, 4),
                     "new_rate": round(new / n, 4),
                     "fp_count": fp, "tp_count": tp})

    csv_path = OUTDIR / "threshold_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {csv_path}  ({len(rows)} rows)")

    ok = [r for r in rows if r["precision"] and r["precision"] >= a.target_precision]
    chosen = max(ok, key=lambda r: r["recall"]) if ok else None
    print(f"  operating points meeting precision >= {a.target_precision}: {len(ok)}")

    if chosen is None:
        # Report the best achievable instead of silently lowering the bar.
        best = max((r for r in rows if r["precision"] is not None),
                   key=lambda r: r["precision"])
        chosen = {**best, "target_precision_met": False,
                  "note": f"NO operating point reaches precision "
                          f"{a.target_precision} on this model. Best achievable "
                          f"precision is {best['precision']} at tau={best['tau']} "
                          f"delta={best['delta']}, auto-deciding only "
                          f"{best['auto_decide_rate']:.1%} of queries. This is a "
                          f"consequence of cross-domain transfer (Phase 2: "
                          f"Rank-1 0.60, margin_p05 0.0009), not of the sweep."}
        print(f"  *** target precision NOT reachable; best = {best['precision']} "
              f"at tau={best['tau']} delta={best['delta']}")
    else:
        chosen = {**chosen, "target_precision_met": True}
        print(f"  chosen: tau={chosen['tau']} delta={chosen['delta']} "
              f"precision={chosen['precision']} recall={chosen['recall']}")

    chosen["model"] = a.model
    chosen["tau_low"] = a.tau_low
    chosen["n_query"] = n
    chosen["n_gallery"] = len(g)
    chosen["margin_mean"] = round(float(margins.mean()), 4)
    chosen["margin_p05"] = round(float(np.percentile(margins, 5)), 4)
    (OUTDIR / "chosen_operating_point.json").write_text(json.dumps(chosen, indent=2))
    print(f"  wrote {OUTDIR/'chosen_operating_point.json'}")
    print(json.dumps(chosen, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
