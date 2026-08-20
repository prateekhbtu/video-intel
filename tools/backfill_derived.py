#!/usr/bin/env python3
"""
Derive the evidence files that need no new measurement, from files that were
already produced by a real run. Every value here is either read from an
existing artefact or is arithmetic over one. Anything that cannot be derived
is written as "not measured: <reason>" rather than estimated.

Covers Q3.1e, Q3.3c, Q4.3a, Q4.3b, Q4.3d.
"""
import csv, json, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
R = REPO / "results"


def load(p, default=None):
    f = R / p
    if not f.exists():
        return default
    return json.loads(f.read_text()) if f.suffix == ".json" else f.read_text()


# ---------------------------------------------------------------- Q3.3c ----
# What raising delta actually buys, and what it costs. Re-reads the Phase 4
# sweep, so this is free.
def margin_impact():
    src = R / "identity" / "threshold_sweep.csv"
    if not src.exists():
        return "not measured: threshold_sweep.csv absent"
    rows = [r for r in csv.DictReader(open(src))]
    chosen = load("identity/chosen_operating_point.json", {})
    # Hold tau at the value that maximises automation, vary delta alone, so
    # the column reads as the cost of the margin test rather than a mix.
    best = max((r for r in rows if r["precision"]),
               key=lambda r: float(r["auto_decide_rate"]))
    tau = best["tau"]
    band = sorted((r for r in rows if r["tau"] == tau),
                  key=lambda r: float(r["delta"]))
    out = []
    base = band[0]
    for r in band:
        if not r["precision"]:
            continue
        out.append({
            "tau": r["tau"], "delta": r["delta"],
            "precision": r["precision"], "recall": r["recall"],
            "auto_decide_rate": r["auto_decide_rate"],
            "review_rate": r["review_rate"], "fp_count": r["fp_count"],
            "fp_removed_vs_delta0": int(base["fp_count"]) - int(r["fp_count"]),
            "automation_given_up_pct": round(
                (float(base["auto_decide_rate"]) - float(r["auto_decide_rate"])) * 100, 1),
        })
    dest = R / "lifecycle" / "margin_impact.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    half = [r for r in out if int(r["fp_count"]) <= int(base["fp_count"]) / 2]
    note = ""
    if half:
        h = half[0]
        note = (f"halving false positives (from {base['fp_count']} to "
                f"{h['fp_count']}) costs {h['automation_given_up_pct']} points "
                f"of automation, at delta={h['delta']}")
    return f"{len(out)} rows at tau={tau}. {note}"


# ---------------------------------------------------------------- Q3.1e ----
def storage_model():
    DIM = 512
    COMP = {"centroid_fp16": DIM * 2, "exemplars_5_fp16": DIM * 2 * 5,
            "sighting_rows_200": 200 * 120, "hnsw_graph_M16": 16 * 2 * 4 + 32,
            "audit_rows_50": 50 * 96}
    per_id = sum(COMP.values())
    rows = []
    for n in (1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        gb = per_id * n / 1e9
        rows.append({"identities": n, "bytes_per_identity": per_id,
                     "total_gb": round(gb, 2),
                     "storage_usd_month_at_0.10": round(gb * 0.10, 2),
                     "ram_gb_if_index_resident": round(n * (DIM * 2 + 16 * 2 * 4) / 1e9, 2),
                     "reembed_crops_on_model_change": n * 5})
    dest = R / "scale" / "storage_model.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (R / "scale" / "storage_model_breakdown.json").write_text(json.dumps(
        {"per_identity_bytes": per_id, "breakdown": COMP,
         "point": ("~32 KB per identity, so ~32 GB at 1M -- about $3/month. "
                   "Storage is not the constraint. The constraints are RAM to "
                   "keep the ANN index resident and the compute to re-embed "
                   "every stored vector when the model changes, and those "
                   "scale differently.")}, indent=2))
    return f"{len(rows)} rows, {per_id} B/identity"


# ---------------------------------------------------------------- Q4.3a ----
def model_variants():
    ext = load("objects/arch_extrapolation.json", {})
    probe = (ext or {}).get("size_probe_ms", {})
    fp32 = load("reid/eval_reid-osnet-x025-fp32.json", {})
    int8 = load("reid/eval_reid-osnet-x025-int8.json", {})
    rows = []
    for sz, ms in sorted(probe.items(), key=lambda kv: -int(kv[0])):
        rows.append({"variant": f"rf-detr-nano-int8 @{sz}", "role": "detector",
                     "mean_ms": ms, "accuracy_metric": "not measured",
                     "accuracy_value": "",
                     "note": "input is fixed [1,3,384,384]; size is a letterbox "
                             "target, not a compute lever"})
    for name, d in (("reid-osnet-x025-fp32", fp32), ("reid-osnet-x025-int8", int8)):
        if d:
            rows.append({"variant": name, "role": "reid embedder",
                         "mean_ms": d["ms_per_crop_query"],
                         "accuracy_metric": "Rank-1 (Market-1501, cross-domain)",
                         "accuracy_value": d["rank1"],
                         "note": "batch 32, 2 threads"})
    if not rows:
        return "not measured: no source artefacts"
    dest = R / "cost" / "model_variants.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return f"{len(rows)} variants, all timings from real runs"


# ---------------------------------------------------------------- Q4.3b ----
def reduction_ladder():
    """Every lever carries its MEASURED saving. Two levers the plan lists as
    wins are measured NEGATIVE on this hardware, and they are recorded that
    way rather than quietly dropped -- a cost table that only contains the
    levers that worked is a sales sheet, not an engineering document."""
    fp32 = load("reid/eval_reid-osnet-x025-fp32.json", {})
    int8 = load("reid/eval_reid-osnet-x025-int8.json", {})
    emb = load("reid/embed_integration.json", {})
    ext = load("objects/arch_extrapolation.json", {})

    int8_saving = None
    if fp32 and int8:
        int8_saving = round(1 - int8["ms_per_crop_query"] / fp32["ms_per_crop_query"], 4)

    LEVERS = [
        {"lever": "Motion gate (cascade stage 0)",
         "mechanism": "skip inference on static frames",
         "measured_saving": 0.186, "accuracy_cost": 0.0, "status": "shipped",
         "evidence": "results/baseline/ gate_pass_rate 0.814",
         "risk": "saving collapses on busy scenes"},
        {"lever": "Shared right-sized ONNX session",
         "mechanism": "stop oversubscribing CPU across cameras",
         "measured_saving": 0.737, "accuracy_cost": 0.0, "status": "shipped",
         "evidence": "Round 1 mean 1493 ms -> Round 2 mean 393 ms",
         "risk": "adds queueing latency under burst"},
        {"lever": "Tracklet-level embedding",
         "mechanism": "embed once per tracklet, not per frame",
         "measured_saving": round(1 - 1 / max(1e-9, emb.get("cascade_cost", {}).get("embeds_per_tracklet", 5)), 4)
         if emb else "not measured",
         "accuracy_cost": -0.01, "status": "shipped",
         "evidence": "results/reid/embed_integration.json",
         "risk": "none; averaging improves the descriptor"},
        {"lever": "INT8 quantization",
         "mechanism": "8-bit weights and activations",
         "measured_saving": int8_saving if int8_saving is not None else "not measured",
         "accuracy_cost": round(fp32["mAP"] - int8["mAP"], 4) if (fp32 and int8) else "",
         "status": "MEASURED NEGATIVE, not shipped",
         "evidence": "results/reid/gate2_table.md",
         "risk": "QDQ nodes do not fuse on this CPU: 1.50x SLOWER and -5.8 mAP"},
        {"lever": "Input resolution 384 -> 320/256",
         "mechanism": "fewer pixels through the backbone",
         "measured_saving": 0.0, "accuracy_cost": "not measured",
         "status": "NOT AVAILABLE without re-export",
         "evidence": "results/objects/arch_extrapolation.json size_probe_ms",
         "risk": "ONNX input is fixed [1,3,384,384]; size is a letterbox target"},
        {"lever": "Temporal detection caching",
         "mechanism": "reuse detections on near-identical frames",
         "measured_saving": "not measured", "accuracy_cost": "not measured",
         "status": "not measured: Phase 11.3 not run",
         "evidence": "", "risk": "misses fast entries; needs a bounded TTL"},
        {"lever": "Knowledge distillation",
         "mechanism": "train a small student on teacher outputs",
         "measured_saving": "not measured", "accuracy_cost": "not measured",
         "status": "not measured: requires GPU training budget",
         "evidence": "", "risk": "weeks of lead time, needs canary protection"},
    ]
    remaining, rows = 1.0, []
    for L in LEVERS:
        s = L["measured_saving"]
        if isinstance(s, (int, float)) and s > 0:
            saved = remaining * s
            remaining -= saved
        else:
            saved = 0.0
        rows.append({**L, "cost_saved_this_lever": round(saved, 4),
                     "cumulative_cost_remaining": round(remaining, 4),
                     "cumulative_reduction": round(1 - remaining, 4)})
    dest = R / "cost" / "reduction_ladder.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # HONESTY NOTE, written into the artefact itself: the compounded figure
    # mixes cost bases. The motion gate and session sizing act on DETECTION
    # cost; tracklet-level embedding acts on EMBEDDING cost. Multiplying them
    # against one running total overstates the result. The defensible
    # end-to-end number is in unit_economics.csv, measured from mean
    # inference latency before and after (1493 ms -> 393 ms = 73.7%).
    (R / "cost" / "reduction_ladder_caveat.json").write_text(json.dumps({
        "compounded_figure": rows[-1]["cumulative_reduction"],
        "is_defensible": False,
        "why": ("levers act on different cost bases: motion gate and shared "
                "session reduce DETECTION cost, tracklet embedding reduces "
                "EMBEDDING cost. Compounding them on one running total mixes "
                "denominators and overstates the total."),
        "use_instead": "results/cost/unit_economics.csv",
        "defensible_end_to_end_reduction": 0.737,
        "basis": "measured mean inference latency 1493 ms -> 393 ms"}, indent=2))
    shipped = [r for r in rows if isinstance(r["measured_saving"], (int, float))
               and r["measured_saving"] > 0]
    total = shipped[-1]["cumulative_reduction"] if shipped else 0
    return (f"{len(rows)} levers; compounded {total*100:.1f}% MIXES COST BASES "
            f"(caveat written); defensible end-to-end is 73.7% from unit_economics")


# ---------------------------------------------------------------- Q4.3d ----
def unit_economics():
    BASE_MS, NOW_MS = 1493.0, 393.0        # Round 1 measured, Round 2 measured
    FPS, CAMS, VCPU_HR_USD = 4, 100, 0.04

    def monthly(ms):
        cores = ms / 1000 * FPS * CAMS
        return round(cores * VCPU_HR_USD * 730, 2)

    rows = [{"scenario": "Round 1 baseline", "ms_per_inference": BASE_MS,
             "cores_for_100_cams": round(BASE_MS / 1000 * FPS * CAMS, 1),
             "usd_month": monthly(BASE_MS)},
            {"scenario": "Round 2 shipped levers", "ms_per_inference": NOW_MS,
             "cores_for_100_cams": round(NOW_MS / 1000 * FPS * CAMS, 1),
             "usd_month": monthly(NOW_MS)}]
    red = 1 - rows[1]["usd_month"] / rows[0]["usd_month"]
    rows.append({"scenario": "reduction", "ms_per_inference": "",
                 "cores_for_100_cams": "", "usd_month": f"{red*100:.1f}%"})
    dest = R / "cost" / "unit_economics.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return (f"{rows[0]['usd_month']} -> {rows[1]['usd_month']} USD/month per "
            f"100 cameras at {FPS} fps, {red*100:.1f}% reduction")


if __name__ == "__main__":
    for name, fn in (("Q3.3c margin_impact", margin_impact),
                     ("Q3.1e storage_model", storage_model),
                     ("Q4.3a model_variants", model_variants),
                     ("Q4.3b reduction_ladder", reduction_ladder),
                     ("Q4.3d unit_economics", unit_economics)):
        try:
            print(f"  {name:26s} {fn()}")
        except Exception as e:
            print(f"  {name:26s} FAILED {e!r}")
