#!/usr/bin/env python3
"""
Add an open-set class from N reference crops. Build plan 9.2, PS-4 Q4.1c.

THE POINT, WHICH IS EASY TO MISS
    The class does not go into the detector at all. Stage 1 proposes regions
    class-agnostically; stage 2 embeds each crop; membership is a cosine
    comparison against a prototype held in a gallery. So "add a class" is a
    database write plus a control-plane directive, both of which take seconds,
    instead of a retrain measured in weeks.

    The acceptance threshold is DERIVED from intra-class spread, not
    hardcoded. A visually tight class earns a tight threshold; a diverse one
    earns a loose one. Hardcoding 0.7 for every class is how open-set systems
    end up with per-class accuracy nobody can explain.

HONEST LIMITATION
    The embedder is OSNet, trained for PERSON re-identification. Using it on
    forklifts is out of domain, and the numbers in eval_openset reflect that.
    The MECHANISM is what is being demonstrated here; a production deployment
    would use a general visual backbone for object prototypes.
"""
import glob, json, os, pathlib, sys, time
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, cv2, onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MODEL = REPO / "data/models/registry/reid-osnet-x025-fp32.onnx"

def crops_for(split_dir, class_id, limit):
    out = []
    for img_p in sorted(glob.glob(f"{split_dir}/images/*.jpg")):
        lbl = img_p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl): continue
        img = cv2.imread(img_p)
        if img is None: continue
        h, w = img.shape[:2]
        for line in open(lbl):
            p = line.split()
            if len(p) < 5 or int(p[0]) != class_id: continue
            cx, cy, bw, bh = [float(x) for x in p[1:5]]
            x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
            x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
            cr = img[max(0,y1):y2, max(0,x1):x2]
            if cr.size == 0 or (x2-x1) < 20 or (y2-y1) < 20: continue
            out.append(cr)
            if len(out) >= limit: return out
    return out

def embed(sess, inp, crops):
    if not crops: return np.zeros((0, 512), np.float32)
    b = []
    for cr in crops:
        im = cv2.resize(cr, (128, 256))[:, :, ::-1].astype(np.float32)/255.
        b.append(((im-MEAN)/STD).transpose(2,0,1))
    v = sess.run(None, {inp: np.ascontiguousarray(np.stack(b), dtype=np.float32)})[0]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "forklift"
    src  = sys.argv[2] if len(sys.argv) > 2 else str(REPO/"data/datasets/objects/forklift_true/train")
    k    = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    crops = crops_for(src, 0, k)
    if len(crops) < 5: sys.exit(f"only {len(crops)} crops found under {src}")
    V = embed(sess, inp, crops)
    proto = V.mean(0); proto /= np.linalg.norm(proto)
    sims = V @ proto
    out = {"class": name, "n_examples": len(V),
           "prototype": [round(float(x), 6) for x in proto],
           "exemplars": [[round(float(x), 6) for x in v] for v in V[:8]],
           "intra_sim_mean": round(float(sims.mean()), 4),
           "intra_sim_p05": round(float(np.percentile(sims, 5)), 4),
           "suggested_threshold": round(float(np.percentile(sims, 5)) - 0.05, 4),
           "embedder": MODEL.name, "created_at": time.time(),
           "threshold_rationale": "intra-class p05 minus 0.05, not a hardcoded constant"}
    d = REPO/"results"/"objects"; d.mkdir(parents=True, exist_ok=True)
    (d/f"class_{name}.json").write_text(json.dumps(out, indent=2))
    print(f"  {name}: {len(V)} examples  intra_sim mean={out['intra_sim_mean']} "
          f"p05={out['intra_sim_p05']}  threshold={out['suggested_threshold']}")
    print(f"  wrote {d/f'class_{name}.json'}")

if __name__ == "__main__":
    main()
