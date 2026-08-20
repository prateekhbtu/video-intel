#!/usr/bin/env python3
"""Rank-1 / Rank-5 / mAP on Market-1501 with the standard protocol:
a gallery item is excluded when it shares BOTH identity and camera with the
query, because matching the same person in the same camera is trivial."""
import sys, glob, json, time, pathlib
import numpy as np, cv2, onnxruntime as ort

MEAN = np.array([0.485,0.456,0.406], np.float32)
STD = np.array([0.229,0.224,0.225], np.float32)
ROOT = pathlib.Path("data/datasets/Market-1501-v15.09.15")

def meta(p):
    n = pathlib.Path(p).name.split("_")
    return int(n[0]), int(n[1][1])

def embed_all(sess, files, bs=32):
    inp = sess.get_inputs()[0].name
    out, lat = [], []
    for i in range(0, len(files), bs):
        batch = []
        for f in files[i:i+bs]:
            img = cv2.imread(f)
            img = cv2.resize(img, (128,256))[:,:,::-1].astype(np.float32)/255.0
            batch.append(((img-MEAN)/STD).transpose(2,0,1))
        x = np.ascontiguousarray(np.stack(batch), dtype=np.float32)
        t0 = time.perf_counter()
        v = sess.run(None, {inp: x})[0]
        lat.append((time.perf_counter()-t0)*1000/len(batch))
        out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12))
    return np.vstack(out), float(np.mean(lat))

model = sys.argv[1]
so = ort.SessionOptions(); so.intra_op_num_threads = 2
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])

q = sorted(glob.glob(str(ROOT/"query"/"*.jpg")))
g = sorted(f for f in glob.glob(str(ROOT/"bounding_box_test"/"*.jpg"))
           if not pathlib.Path(f).name.startswith("-1"))
print(f"query {len(q)} gallery {len(g)}")

Q, q_ms = embed_all(sess, q); G, g_ms = embed_all(sess, g)
qm = np.array([meta(f) for f in q]); gm = np.array([meta(f) for f in g])
r1 = r5 = 0; aps = []; margins = []
for i in range(len(q)):
    sims = G @ Q[i]
    valid = ~((gm[:,0] == qm[i,0]) & (gm[:,1] == qm[i,1]))
    s, gp = sims[valid], gm[valid,0]
    order = np.argsort(-s); ranked = gp[order]
    hit = ranked == qm[i,0]
    if not hit.any(): continue
    r1 += hit[0]; r5 += hit[:5].any()
    margins.append(float(s[order][0] - s[order][1]))
    cum = np.cumsum(hit)
    aps.append(float(np.sum(cum[hit] / (np.where(hit)[0] + 1)) / hit.sum()))

res = {"model": model,
       "rank1": round(r1/len(q), 4), "rank5": round(r5/len(q), 4),
       "mAP": round(float(np.mean(aps)), 4),
       "margin_mean": round(float(np.mean(margins)), 4),
       "margin_p05": round(float(np.percentile(margins, 5)), 4),
       "ms_per_crop_query": round(q_ms, 2),
       "ms_per_crop_gallery": round(g_ms, 2),
       "dim": int(Q.shape[1]), "n_query": len(q), "n_gallery": len(g)}
name = pathlib.Path(model).stem
pathlib.Path("results/reid").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"results/reid/eval_{name}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
