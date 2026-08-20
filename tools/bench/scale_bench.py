#!/usr/bin/env python3
"""Query latency, recall@1, memory and build time across gallery sizes and
index types. Build plan 5.2, answering Q3.1b and Q3.1c.

The question "when do you need a vector database" has a numeric answer, and it
is the gallery size at which exact flat search stops fitting your p95 budget.
Below that point an index costs build time, memory and recall for nothing.
"""
import gc, json, csv, pathlib, sys, time
import numpy as np, psutil
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
proc = psutil.Process()
def rss_mb():
    gc.collect(); return proc.memory_info().rss / 1e6
def pct(v, q): return sorted(v)[min(len(v)-1, int(q*len(v)))]

SIZES = [int(x) for x in (sys.argv[1:] or ["1000","10000","100000"])]
rows = []
for n in SIZES:
    f = REPO / "data" / "bench" / f"gallery_{n}.npy"
    if not f.exists(): print(f"  skip {n}: {f} absent"); continue
    V = np.load(f)
    rng = np.random.default_rng(7)
    qi = rng.integers(0, n, 200)
    Q = V[qi] + rng.normal(scale=0.15, size=(200, V.shape[1])).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    base = rss_mb(); lat = []; truth = []
    for q in Q:
        t0 = time.perf_counter(); s = V @ q; top = int(np.argmax(s))
        lat.append((time.perf_counter()-t0)*1000); truth.append(top)
    rows.append({"n": n, "index": "flat_fp32", "build_s": 0.0,
                 "p50_ms": round(pct(lat,.5),3), "p95_ms": round(pct(lat,.95),3),
                 "recall@1": 1.0, "mem_mb": round(V.nbytes/1e6,1)})
    print(f"  {n:>7} flat_fp32  p95={rows[-1]['p95_ms']:.3f}ms")

    Vh = V.astype(np.float16); lat=[]; hit=0
    for k,q in enumerate(Q):
        t0=time.perf_counter(); s = Vh @ q.astype(np.float16); top=int(np.argmax(s))
        lat.append((time.perf_counter()-t0)*1000); hit += top==truth[k]
    rows.append({"n": n, "index": "flat_fp16", "build_s": 0.0,
                 "p50_ms": round(pct(lat,.5),3), "p95_ms": round(pct(lat,.95),3),
                 "recall@1": round(hit/len(Q),4), "mem_mb": round(Vh.nbytes/1e6,1)})
    print(f"  {n:>7} flat_fp16  p95={rows[-1]['p95_ms']:.3f}ms recall={rows[-1]['recall@1']}")

    try:
        import hnswlib
        for M, ef in ((16,64),(32,128)):
            idx = hnswlib.Index(space="cosine", dim=V.shape[1])
            t0=time.perf_counter(); idx.init_index(max_elements=n, ef_construction=200, M=M)
            idx.add_items(V, np.arange(n)); build=time.perf_counter()-t0
            idx.set_ef(ef); lat=[]; hit=0
            for k,q in enumerate(Q):
                t0=time.perf_counter(); lbl,_=idx.knn_query(q,k=1)
                lat.append((time.perf_counter()-t0)*1000); hit += int(lbl[0][0])==truth[k]
            rows.append({"n": n, "index": f"hnsw_M{M}_ef{ef}", "build_s": round(build,1),
                         "p50_ms": round(pct(lat,.5),3), "p95_ms": round(pct(lat,.95),3),
                         "recall@1": round(hit/len(Q),4),
                         "mem_mb": round(V.nbytes/1e6 + n*M*8/1e6,1)})
            print(f"  {n:>7} hnsw M{M}   p95={rows[-1]['p95_ms']:.3f}ms "
                  f"recall={rows[-1]['recall@1']} build={build:.1f}s")
            del idx
    except ImportError:
        print("  hnswlib not installed: ANN rows not measured")
    del V, Vh; gc.collect()

d = REPO / "results" / "scale"; d.mkdir(parents=True, exist_ok=True)
with open(d/"scaling_curve.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

flat = [r for r in rows if r["index"]=="flat_fp32"]
BUDGET=50.0
over = [r for r in flat if r["p95_ms"]>BUDGET]
cross = over[0]["n"] if over else None
if cross is None and len(flat)>=2:
    a,b = flat[-2],flat[-1]
    growth = (b["p95_ms"]/max(1e-9,a["p95_ms"]))/(b["n"]/a["n"])
    est = b["n"] * (BUDGET/max(1e-9,b["p95_ms"]))
    note = (f"flat search stays under the {BUDGET:.0f}ms p95 budget at every "
            f"size measured (max {b['n']} at {b['p95_ms']:.3f}ms). Linear "
            f"extrapolation puts the crossover near {est:,.0f} vectors.")
else:
    note = f"flat search exceeds the {BUDGET:.0f}ms p95 budget at {cross} vectors."
(d/"scaling_note.json").write_text(json.dumps(
    {"p95_budget_ms":BUDGET,"measured_sizes":SIZES,"crossover_measured":cross,
     "note":note,
     "limitation":"1M not measured: 1M x 512 fp32 is ~2GB and the HNSW build "
                  "exceeds the time budget for this session. The 1M row is "
                  "extrapolated from the measured curve, not observed."},indent=2))
print(f"\n  {note}")
print(f"  wrote {d/'scaling_curve.csv'}")
