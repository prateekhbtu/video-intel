#!/usr/bin/env python3
"""
Evidence consolidator. Build plan section 14.1.

Collects every results/ artefact into one evidence file. A question with no
artefact prints MISSING, which is the to-do list, not a rounding error. This
is the mechanism that makes "every number is traceable to a file" checkable
instead of aspirational.
"""
import json, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent

MAP = {
 "Q3.1a": ("results/reid/eval_reid-osnet-x025-int8.json", "results/reid/face_feasibility.json"),
 "Q3.1b": ("results/scale/scaling_curve.csv",),
 "Q3.1c": ("results/scale/scaling_curve.csv", "results/scale/pgvector_explain.txt"),
 "Q3.1d": ("results/identity/domain_shift.txt",),
 "Q3.1e": ("results/scale/storage_model.csv",),
 "Q3.2a": ("results/privacy/deletion_receipt.json",),
 "Q3.2b": ("results/privacy/audit_sample.txt",),
 "Q3.2c": ("results/privacy/deletion_receipt.json",),
 "Q3.2d": ("results/privacy/deletion_receipt.json",),
 "Q3.3a": ("results/privacy/fp_diagnosis.json",),
 "Q3.3b": ("results/privacy/exclusion_latency.json",),
 "Q3.3c": ("results/lifecycle/margin_impact.csv",),
 "Q4.1a": ("results/objects/arch_benchmark.csv",),
 "Q4.1b": ("results/objects/openset_eval.json",),
 "Q4.1c": ("results/objects/new_class_timeline.md",),
 "Q4.1d": ("results/objects/arch_extrapolation.json",),
 "Q4.2a": ("results/baseline/round1_verify_FAIL.txt", "results/lifecycle/canary_log.json"),
 "Q4.2b": ("results/lifecycle/rollback_drill.json",),
 "Q4.2c": ("results/lifecycle/cohort_metrics.txt",),
 "Q4.2d": ("results/lifecycle/canary_log.json",),
 "Q4.3a": ("results/cost/model_variants.csv",),
 "Q4.3b": ("results/cost/reduction_ladder.csv",),
 "Q4.3c": ("results/cost/cache_eval.json",),
 "Q4.3d": ("results/cost/unit_economics.csv",),
}
TOPIC = {
 "Q3.1a":"Face recognition vs ReID","Q3.1b":"1K to 1M database scaling",
 "Q3.1c":"When a vector DB is necessary","Q3.1d":"Embedding drift / domain shift",
 "Q3.1e":"Storage cost per identity","Q3.2a":"Consent at scale",
 "Q3.2b":"Audit trail design","Q3.2c":"Time-based deletion",
 "Q3.2d":"Hard problems in deletion","Q3.3a":"Diagnosing a false match",
 "Q3.3b":"Immediate fix","Q3.3c":"Systemic change",
 "Q4.1a":"Monolithic vs modular","Q4.1b":"Classification vs embedding retrieval",
 "Q4.1c":"New class in under a day","Q4.1d":"Strategy A vs B",
 "Q4.2a":"Catching a bad deploy earlier","Q4.2b":"Per-customer rollback",
 "Q4.2c":"A/B testing in production","Q4.2d":"Cohort-scoped metrics",
 "Q4.3a":"Quantization / pruning","Q4.3b":"Multi-stage inference",
 "Q4.3c":"Caching strategy","Q4.3d":"50 percent cost reduction",
}

def main():
    report, missing = {}, []
    for q, paths in MAP.items():
        have = [p for p in paths if (REPO / p).exists()]
        gap  = [p for p in paths if not (REPO / p).exists()]
        report[q] = {"topic": TOPIC[q], "evidence": have, "missing": gap,
                     "status": "OK" if have else "GAP"}
        status = "OK  " if have else "GAP "
        print(f"  {status} {q}  {TOPIC[q]:38s} {len(have)}/{len(paths)}"
              + (f"   missing: {', '.join(gap)}" if gap else ""))
        if not have:
            missing.append(q)
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/EVIDENCE.json").write_text(json.dumps(report, indent=2))
    n = len(MAP) - len(missing)
    print(f"\n  {n}/{len(MAP)} questions have at least one evidence file")
    if missing:
        print(f"  NO evidence at all: {', '.join(missing)}")
    return 1 if missing else 0

if __name__ == "__main__":
    sys.exit(main())
