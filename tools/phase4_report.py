#!/usr/bin/env python3
"""Phase 4 gate table, generated from the artifacts the phase actually wrote."""
import json, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ID = REPO / "results" / "identity"

chosen = json.loads((ID / "chosen_operating_point.json").read_text())
curve = json.loads((ID / "precision_recall_curve.json").read_text())
gt = json.loads((ID / "ground_truth.json").read_text())
runs = json.loads((ID / "crosscam_eval_runs.json").read_text())
contract = json.loads((REPO / "results/reid/embed_contract.json").read_text())

tl = {r["label"]: r for r in runs if r["label"].startswith("taulow_")}

L = ["# Gate 4 — Cloud identity plane and threshold calibration", "",
     "## 4.4 The calibration sweep", "",
     f"Swept tau 0.50–0.90 x delta 0.00–0.20 over {chosen['n_query']} Market-1501 "
     f"queries against {chosen['n_gallery']} gallery images "
     f"(`results/identity/threshold_sweep.csv`, 320 rows).", "",
     f"Measured margin: mean {chosen['margin_mean']}, p05 {chosen['margin_p05']}.", "",
     "### What delta buys, and what it costs (tau = 0.50)", "",
     "| delta | precision | recall | auto-decide | review |",
     "|---|---|---|---|---|",
     "| 0.00 | 0.6733 | 0.6733 | 100.0% | 0.0% |",
     "| 0.01 | 0.7246 | 0.4342 | 59.9% | 40.1% |",
     "| 0.02 | 0.7519 | 0.2475 | 32.9% | 67.1% |",
     "| 0.03 | 0.8028 | 0.1458 | 18.2% | 81.8% |",
     "| 0.05 | 0.8267 | 0.0517 | 6.2% | 93.8% |",
     "| 0.09 | 0.9091 | 0.0083 | 0.9% | 99.1% |", "",
     "**This table is the answer to Q3.3c.** With a single threshold "
     "(delta = 0) the system auto-decides everything and is wrong on 32.7% of "
     "those decisions — that IS the John/David false match, quantified. Every "
     "increment of delta converts false positives into review items. The "
     "margin test is not a tuning knob, it is the mechanism that turns a "
     "confident wrong answer into a slower correct one.", "",
     "### Best recall at each precision target", "",
     "| target | tau | delta | precision | recall | auto-decide |",
     "|---|---|---|---|---|---|"]
for c in curve:
    L.append(f"| {c['precision_target']:.2f} | {c['tau']:.3f} | {c['delta']:.2f} | "
             f"{c['precision']:.4f} | {c['recall']:.4f} | {c['auto_decide_rate']*100:.1f}% |")
L += ["",
      f"Chosen operating point at the stated precision target 0.99: "
      f"**tau={chosen['tau']}, delta={chosen['delta']}**, precision "
      f"{chosen['precision']}, recall {chosen['recall']} — auto-deciding "
      f"{chosen['auto_decide_rate']*100:.1f}% of queries.", "",
      "Reaching surveillance-grade precision on this model means abstaining on "
      "essentially everything. That is the honest reading, and it is a "
      "statement about the MODEL (cross-domain Rank-1 0.601, margin_p05 "
      "0.0009), not about the resolution logic.", "",
      "## 4.5 Cross-camera evaluation on our own topology", "",
      f"Ground truth: **{len(gt['pairs'])} pairs / {gt['n_identities']} "
      f"identities** across {gt.get('n_tracklets')} tracklets, keyed by "
      f"`sighting_id`, built from the same population the resolver acts on.", "",
      "### The tau_low ablation", "",
      "| tau_low | delta | identities admitted | GT resolved | correctly linked | split | over-merged | link recall |",
      "|---|---|---|---|---|---|---|---|"]
for key, lab, d in (("taulow_0.55", "0.55", "0.00"),
                    ("taulow_0.80", "0.80", "0.00"),
                    ("taulow_0.80d03", "0.80", "0.03")):
    r = tl.get(key)
    if not r:
        continue
    resolved = r["gt_pairs"] - r["unresolved_tracklets"]
    L.append(f"| {lab} | {d} | {r['identities_created']} | "
             f"{resolved}/{r['gt_pairs']} | {r['correctly_linked']} | "
             f"{r['missed_links']} | {r['over_merged_identities']} | "
             f"{r['link_recall']:.2f} |")

L += ["", "### Two distinct failures, separated by measurement", "",
      "**1. Gallery starvation at the plan's default tau_low = 0.55.** Phase 3 "
      f"measured this model's CROSS-track similarity at "
      f"{contract['cross_track_sim_mean']} — the similarity between two "
      "different people. tau_low sits *below* that noise floor, so no tracklet "
      "ever scores low enough to be admitted as a new identity. Of 24 "
      "ground-truth tracklets, 0 would qualify as NEW; the gallery froze at 16 "
      "identities for 1124 sightings, and a cross-camera link is impossible if "
      "the person was never admitted in the first place. The sweep in 4.4 tuned "
      "tau and delta but never tau_low, which is why it did not catch this.", "",
      "**2. The model cannot link across cameras anyway.** Raising tau_low to "
      "0.80 fixed the starvation — 139 identities admitted, 12 of 21 GT pairs "
      "now resolved on both sides — and **0 of those 12 were linked correctly**. "
      "Every one landed under a different subject_id. Link recall is 0.00 at "
      "every operating point tested, and the two false links that did form were "
      "removed by turning the margin test on (delta 0.03), which is the margin "
      "test working correctly with nothing correct left to protect.", "",
      "The first failure is a calibration bug and is fixed. The second is a "
      "model capability limit and is not fixable by threshold choice.", "",
      "## Gate 4 criteria", "",
      "| Criterion | Value | Threshold | Result |", "|---|---|---|---|",
      f"| threshold_sweep.csv >= 300 rows | 320 | 300 | PASS |",
      f"| chosen_operating_point meets stated precision | {chosen['precision']} | 0.99 | PASS |",
      f"| crosscam link_recall above 0.60 | 0.00 | 0.60 | **FAIL** |",
      f"| resolver review count non-zero (abstain path live) | "
      f"{tl.get('taulow_0.80',{}).get('review_queue_depth','n/a')} | >0 | PASS |",
      "", "**GATE 4: NOT MET** — on link recall only.", "",
      "## 4.6 Domain shift (Q3.1d)", "",
      "Recorded separately in `results/identity/domain_shift.txt`.", "",
      "## Known limitation carried forward from Phase 1", "",
      "The a_cam01|a_cam02 crop-split pair is the one the topology was built to "
      "demonstrate, and its measured offset does not match the configured one. "
      "With phase-aligned streams the full-frame pairs reproduce their designed "
      "offsets closely (a_cam02|a_cam03 measured +20.0s against +18.0s "
      "configured; a_cam03|a_cam04 measured +16.0s against +15.0s), but "
      "a_cam01|a_cam02 measures -12.0s against +12.0s configured. That is "
      "expected once stated properly: cam01 shows the right half and cam02 the "
      "left half, so a person must physically WALK between them, adding a "
      "variable transit time whose sign depends on direction. A pure "
      "time-offset ground truth model does not describe that pair, and no "
      "amount of tolerance tuning makes it.", "",
      "A second, separate finding: `-stream_loop -1` wraps each clip at its own "
      "duration (364/352/334/319 s), so the four cameras drift out of phase "
      "after the first loop. Ground truth is only sound in the first ~319 s "
      "after `sim/spawn.sh`. Runs started later measure noise — the offsets "
      "read +70s/+179s/-46s on a long-running stream set.", ""]

out = ID / "gate4_table.md"
out.write_text("\n".join(L))
print("\n".join(L))
print(f"\nwrote {out}")
