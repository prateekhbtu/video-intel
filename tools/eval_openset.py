#!/usr/bin/env python3
"""Precision and recall for an open-set class, with ZERO training steps.
Build plan 9.4, answering Q4.1b."""
import json, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, onnxruntime as ort
sys.path.insert(0, str(REPO/"tools"))
from add_class import crops_for, embed, MODEL

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "forklift"
    base = REPO/"data/datasets/objects/forklift_true"
    meta = json.loads((REPO/f"results/objects/class_{name}.json").read_text())
    proto = np.array(meta["prototype"], np.float32); thr = meta["suggested_threshold"]
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    pos = embed(sess, inp, crops_for(str(base/"valid"), 0, 80))   # forklift
    # NEGATIVES: the held-out split contains only ~1 usable person box, and a
    # precision computed against one negative is not a measurement. Open-set
    # precision is really "does this prototype reject arbitrary OTHER objects",
    # so negatives are drawn from the PPE dataset across several classes.
    ppe = REPO/"data/datasets/objects/ppe/valid"
    neg_crops, neg_src = [], {}
    for cid in (8, 2, 3, 10, 11, 24):        # Person, Hardhat, Ladder, Cone, Vest, wheel loader
        c = crops_for(str(ppe), cid, 25)
        if c: neg_src[cid] = len(c); neg_crops += c
    neg_crops += crops_for(str(base/"valid"), 1, 25)
    neg = embed(sess, inp, neg_crops)
    sp = pos @ proto if len(pos) else np.array([])
    sn = neg @ proto if len(neg) else np.array([])
    tp = int((sp >= thr).sum()); fn = int((sp < thr).sum())
    fp = int((sn >= thr).sum()); tn = int((sn < thr).sum())
    # sweep, because one threshold is a point and the curve is the answer
    sweep = []
    for t in np.arange(0.30, 0.95, 0.05):
        a = int((sp >= t).sum()); b = int((sn >= t).sum())
        sweep.append({"threshold": round(float(t), 2),
                      "recall": round(a/max(1, len(sp)), 4),
                      "precision": round(a/max(1, a+b), 4)})
    res = {"class": name, "threshold": thr, "n_pos": len(sp), "n_neg": len(sn),
           "negative_sources": {"ppe_class_ids": neg_src,
                                "forklift_valid_person": len(neg_crops)-sum(neg_src.values())},
           "true_positives": tp, "false_negatives": fn,
           "false_positives": fp, "true_negatives": tn,
           "recall": round(tp/max(1, len(sp)), 4),
           "precision": round(tp/max(1, tp+fp), 4),
           "training_steps": 0, "gpu_hours": 0.0,
           "pos_sim_mean": round(float(sp.mean()), 4) if len(sp) else None,
           "neg_sim_mean": round(float(sn.mean()), 4) if len(sn) else None,
           "separation": round(float(sp.mean()-sn.mean()), 4) if len(sp) and len(sn) else None,
           "threshold_sweep": sweep,
           "caveat": ("the embedder is OSNet, trained for PERSON re-id, so "
                      "forklift prototypes are out of domain. This measures "
                      "the open-set MECHANISM, not the ceiling a purpose-built "
                      "visual backbone would reach.")}
    (REPO/"results/objects/openset_eval.json").write_text(json.dumps(res, indent=2))
    print(f"  pos={len(sp)} neg={len(sn)}  recall={res['recall']} "
          f"precision={res['precision']}  separation={res['separation']}")
    best = max(sweep, key=lambda r: r["precision"]*r["recall"])
    print(f"  best F-ish point in sweep: thr={best['threshold']} "
          f"P={best['precision']} R={best['recall']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
