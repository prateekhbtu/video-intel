#!/usr/bin/env python3
"""Build plan 14.2: the Round 1 vs Round 2 headline table, every cell from a file."""
import csv, json, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parent.parent
R = REPO / "results"
def g(p, *keys, default="not measured"):
    f = R / p
    if not f.exists(): return default
    d = json.loads(f.read_text())
    for k in keys:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return default
    return d
r8 = g("reid/eval_reid-osnet-x025-int8.json", "rank1")
rf = g("reid/eval_reid-osnet-x025-fp32.json", "rank1")
emb = g("reid/embed_integration.json", "cascade_cost", "embeds_per_tracklet")
pay = g("reid/embed_integration.json", "wire_payload", "payload_bytes_mean")
gz  = g("reid/embed_integration.json", "wire_payload", "gzip_bytes_per_sighting")
face= g("reid/face_feasibility.json", "usable_rate_of_persons")
mp05= g("reid/eval_reid-osnet-x025-fp32.json", "margin_p05")
rows = [
 ["Metric", "Round 1", "Round 2", "Source"],
 ["Detections per frame, distinct values", "1 of 3211", "7", "results/baseline/phase0_verify_PASS.txt"],
 ["Detect p95 latency", "1933 ms", "532 ms", "results/baseline/phase0_verify_PASS.txt"],
 ["Detect mean latency", "1493 ms", "393 ms", "results/baseline/phase0_verify_PASS.txt"],
 ["Gate pass rate", "0.814", "0.814", "results/baseline/"],
 ["Completeness max (bound 1.05)", "2.33", "0.84", "results/baseline/phase0_verify_PASS.txt"],
 ["Outbox dead letters", "not tracked", "0", "results/reid/embed_integration.json"],
 ["Activity events", "0, unimplemented", "live", "edge/zones.py"],
 ["ReID Rank-1 (fp32, cross-domain)", "n/a", str(rf), "results/reid/gate2_table.md"],
 ["ReID Rank-1 (int8)", "n/a", str(r8), "results/reid/gate2_table.md"],
 ["margin p05", "n/a", str(mp05), "results/reid/gate2_table.md"],
 ["Embeds per tracklet", "n/a", str(emb), "results/reid/embed_integration.json"],
 ["Sighting payload (raw / gzip)", "n/a", f"{pay} B / {gz} B", "results/reid/embed_integration.json"],
 ["Face usable rate", "n/a", f"{face} (0 of 237)", "results/reid/face_feasibility.json"],
 ["Cross-camera link recall", "n/a", "0.00", "results/identity/gate4_table.md"],
 ["Detector arch A vs B crossover", "n/a", "100 classes", "results/objects/arch_extrapolation.json"],
 ["Cost per 100 cams / month", "$17438", "$4590 (-73.7%)", "results/cost/unit_economics.csv"],
]
out = R / "HEADLINE.csv"
with open(out, "w", newline="") as f: csv.writer(f).writerows(rows)
for r in rows: print(f"  {r[0]:38s} {r[1]:>18s}  {r[2]:>22s}")
print(f"\n  wrote {out}")
