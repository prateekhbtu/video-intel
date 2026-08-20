#!/usr/bin/env python3
"""
Temporal detection caching, measured. Build plan 11.3, answering Q4.3c.

Caching in video is NOT memoisation of identical inputs -- consecutive frames
are never byte-identical. It is deciding when a previous result is still
VALID. The sweep finds where reuse stops being free.

THE TTL IS A SAFETY PARAMETER, NOT A TUNING KNOB
    An unbounded cache will happily hold a stale "no person present" across a
    genuine entry, because a person walking into frame at the far edge changes
    very few pixels. cache_age < MAX_AGE bounds that exposure to roughly one
    second at 4 fps. State the bound and the reason for it; an unbounded cache
    is how a surveillance system misses the event it exists to catch.
"""
import itertools, json, os, pathlib, sys, time
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("VI_ROOT", str(REPO))
import numpy as np, cv2
from common import telem
telem.init(str(REPO / "data" / "logs" / "cache_eval.jsonl"))
from edge import detect

MAX_AGE = 5          # frames; ~1.25 s at 4 fps
N_FRAMES = 120
STRIDE = 6

def sig(f):
    g = cv2.cvtColor(cv2.resize(f, (64, 36)), cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32) / 255.0

def main():
    d = detect.Detector(classes={1})
    clip = REPO / "data" / "sim" / "a_cam01.mp4"
    rows = []
    for thr in (0.0, 0.001, 0.002, 0.004, 0.008, 0.016):
        cap = cv2.VideoCapture(str(clip))
        prev_sig = None; cached = None; cache_age = 0
        runs = hits = total = agree = 0
        t0 = time.perf_counter()
        for i in itertools.count():
            ok, f = cap.read()
            if not ok or total >= N_FRAMES:
                break
            if i % STRIDE:
                continue
            s = sig(f)
            diff = 1.0 if prev_sig is None else float(np.mean(np.abs(s - prev_sig)))
            truth, _ = d(f, "cache_bench")          # always compute ground truth
            total += 1
            if diff < thr and cached is not None and cache_age < MAX_AGE:
                hits += 1; cache_age += 1
                agree += abs(len(cached) - len(truth)) <= 1
            else:
                runs += 1; cached = truth; cache_age = 0
                agree += 1
            prev_sig = s
        cap.release()
        rows.append({"threshold": thr, "frames": total, "inferences_run": runs,
                     "cache_hits": hits,
                     "hit_rate": round(hits / max(1, total), 4),
                     "compute_saved": round(1 - runs / max(1, total), 4),
                     "agreement_with_truth": round(agree / max(1, total), 4),
                     "max_cache_age_frames": MAX_AGE,
                     "wall_s": round(time.perf_counter() - t0, 1)})
        print(f"  thr={thr:<6} hits={hits:>3} saved={rows[-1]['compute_saved']:.3f} "
              f"agree={rows[-1]['agreement_with_truth']:.3f}")
    usable = [r for r in rows if r["compute_saved"] > 0.25
              and r["agreement_with_truth"] > 0.95]
    out = {"rows": rows, "max_cache_age_frames": MAX_AGE,
           "usable_operating_points": usable,
           "recommended": usable[-1] if usable else None,
           "note": ("agreement_with_truth compares cached vs freshly computed "
                    "object COUNT within +/-1. TTL bounds staleness to "
                    f"{MAX_AGE} frames (~{MAX_AGE/4:.1f}s at 4 fps).")}
    dd = REPO / "results" / "cost"; dd.mkdir(parents=True, exist_ok=True)
    (dd / "cache_eval.json").write_text(json.dumps(out, indent=2))
    print(f"  usable points (saving>25%, agreement>0.95): {len(usable)}")
    print(f"  wrote {dd/'cache_eval.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
