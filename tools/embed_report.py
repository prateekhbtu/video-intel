#!/usr/bin/env python3
"""
Phase 3 evidence: cost of the tracklet-level embedding cascade (3.3) and the
bytes it actually puts on the wire (3.4), plus the Gate 3 table.

WHY THESE TWO NUMBERS TOGETHER
    They are the quantitative form of the same architectural claim. Embedding
    once per tracklet instead of once per frame is what makes edge ReID
    affordable, and shipping a vector instead of a crop is what makes it
    private. One is a compute argument (Q4.3b), the other an egress and
    privacy argument (Q3.1e, Q3.2). Neither is worth asserting without the
    measurement, because both are commonly claimed and rarely checked.
"""
import gzip
import json
import os
import pathlib
import sqlite3
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LOG = REPO / "data" / "logs" / "edge_a.jsonl"
DB = REPO / "data" / "edge_a.db"
OUT = REPO / "results" / "reid" / "embed_integration.json"
MD = REPO / "results" / "reid" / "gate3_table.md"

JPEG_CROP_BYTES = 15000          # a 640x360 person crop, the thing we do NOT send
PAYLOAD_BUDGET = 1500            # Gate 3
EMBEDS_PER_TRACKLET_BUDGET = 5   # Gate 3


def stage_counts(path):
    import collections
    c = collections.Counter()
    for line in open(path):
        try:
            c[json.loads(line).get("stage")] += 1
        except Exception:
            pass
    return c


def main():
    if not LOG.exists():
        sys.exit(f"missing {LOG}")
    c = stage_counts(LOG)
    frames = c["detect_result"]
    embeds = c["embed"]
    tracks = c["track_end"]
    sightings_emitted = c["sighting"]

    cascade = {
        "frames_inferred": frames,
        "embed_calls": embeds,
        "tracklets_closed": tracks,
        "sightings_emitted": sightings_emitted,
        "embeds_per_frame": round(embeds / max(1, frames), 3),
        "embeds_per_tracklet": round(embeds / max(1, tracks), 2),
        "budget_embeds_per_tracklet": EMBEDS_PER_TRACKLET_BUDGET,
    }

    # ---- 3.4 wire payload -----------------------------------------------
    # The field is `emb` (base64 float16), not `embedding`. Measuring the
    # wrong key silently reports zero, which is how a payload budget gets
    # declared met without ever being measured.
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT payload FROM outbox WHERE kind='sighting' ORDER BY id DESC LIMIT 1000"
    ).fetchall()
    payload = {}
    if rows:
        sizes = [len(r[0]) for r in rows]
        parsed = [json.loads(r[0]) for r in rows[:200]]
        emb_lens = {len(p["emb"]) for p in parsed if isinstance(p.get("emb"), str)}
        dims = {p.get("emb_dim") for p in parsed}
        dtypes = {p.get("emb_dtype") for p in parsed}
        meta_bytes = [len(json.dumps({k: v for k, v in p.items() if k != "emb"}))
                      for p in parsed]

        # The outbox POSTs batches, and HTTP can compress. Per-sighting cost on
        # the wire is the gzipped batch divided by its size, which is the
        # number an egress bill is actually computed from.
        batch = json.dumps([{"idem_key": p["sighting_id"], "kind": "sighting",
                             "payload": p} for p in parsed[:20]])
        gz = len(gzip.compress(batch.encode()))

        payload = {
            "sightings_sampled": len(sizes),
            "payload_bytes_mean": round(statistics.mean(sizes), 1),
            "payload_bytes_p95": sorted(sizes)[int(0.95 * len(sizes))],
            "payload_bytes_max": max(sizes),
            "emb_b64_chars": sorted(emb_lens),
            "emb_dim": sorted(d for d in dims if d),
            "emb_dtype": sorted(t for t in dtypes if t),
            "metadata_bytes_mean": round(statistics.mean(meta_bytes), 1),
            "batch_of_20_raw_bytes": len(batch),
            "batch_of_20_gzip_bytes": gz,
            "gzip_bytes_per_sighting": round(gz / 20, 1),
            "gzip_ratio": round(gz / len(batch), 3),
            "budget_bytes": PAYLOAD_BUDGET,
            "vs_jpeg_crop_raw": round(JPEG_CROP_BYTES / statistics.mean(sizes), 1),
            "vs_jpeg_crop_gzip": round(JPEG_CROP_BYTES / (gz / 20), 1),
        }

    # ---- gate ------------------------------------------------------------
    contract_p = REPO / "results" / "reid" / "embed_contract.json"
    contract = json.loads(contract_p.read_text()) if contract_p.exists() else {}

    pg = {}
    try:
        import psycopg2
        cx = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = cx.cursor()
        cur.execute("SELECT model_ver, COUNT(*), COUNT(embedding) FROM sightings "
                    "GROUP BY 1 ORDER BY 2 DESC")
        pg["by_model_ver"] = [{"model_ver": r[0], "rows": r[1], "with_embedding": r[2]}
                              for r in cur.fetchall()]
        pg["total"] = sum(r["rows"] for r in pg["by_model_ver"])
        pg["real_reid_rows"] = sum(r["rows"] for r in pg["by_model_ver"]
                                   if "x025" in (r["model_ver"] or ""))
        cx.close()
    except Exception as e:
        pg = {"error": repr(e)}

    dead = conn.execute("SELECT COALESCE(SUM(dead),0) FROM outbox").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM outbox WHERE sent=0").fetchone()[0]
    conn.close()

    crit = {
        "six_contract_tests_pass": {
            "pass": contract.get("all_pass") is True,
            "value": f"{contract.get('tests_passed')}/{contract.get('tests_total')}"},
        "embeds_per_tracklet_le_5": {
            "pass": cascade["embeds_per_tracklet"] <= EMBEDS_PER_TRACKLET_BUDGET,
            "value": cascade["embeds_per_tracklet"], "threshold": 5},
        "postgres_sightings_nonzero": {
            "pass": pg.get("real_reid_rows", 0) > 0,
            "value": pg.get("real_reid_rows", 0)},
        "mean_payload_under_1500B": {
            "pass": payload.get("payload_bytes_mean", 1e9) < PAYLOAD_BUDGET,
            "value": payload.get("payload_bytes_mean"), "threshold": PAYLOAD_BUDGET},
        "outbox_clean": {"pass": dead == 0 and pending == 0,
                         "value": f"dead={dead} pending={pending}"},
    }
    met = all(v["pass"] for v in crit.values())

    res = {"cascade_cost": cascade, "wire_payload": payload,
           "contract_tests": {"passed": contract.get("tests_passed"),
                              "total": contract.get("tests_total"),
                              "separation": contract.get("separation"),
                              "embedder_kind": contract.get("embedder_kind")},
           "postgres": pg, "outbox": {"dead": dead, "pending": pending},
           "gate3": {"criteria": crit, "met": met}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))

    # ---- markdown --------------------------------------------------------
    L = ["# Gate 3 — Edge embedding integration", "",
         f"Embedder: `{contract.get('embedder_kind')}`, "
         f"`{pathlib.Path(str(contract.get('embedder_path','?'))).name}`", "",
         "## 3.3 Cascade cost — embed once per tracklet, not per frame", "",
         "| Quantity | Value |", "|---|---|",
         f"| Frames inferred | {frames} |",
         f"| Embed calls | {embeds} |",
         f"| Tracklets closed | {tracks} |",
         f"| Sightings emitted | {sightings_emitted} |",
         f"| **Embeds per tracklet** | **{cascade['embeds_per_tracklet']}** "
         f"(budget ≤ {EMBEDS_PER_TRACKLET_BUDGET}) |", "",
         "The descriptor keeps the best 5 crops per tracklet, so ~4.6 is the "
         "cascade working as designed: it embeds only crops that improve the "
         "descriptor, then emits one averaged vector when the tracklet closes.", ""]

    if payload:
        L += ["## 3.4 What actually crosses the WAN", "",
              "| Quantity | Bytes |", "|---|---|",
              f"| Sighting JSON, mean | {payload['payload_bytes_mean']} |",
              f"| Sighting JSON, p95 | {payload['payload_bytes_p95']} |",
              f"| — of which base64 vector | {payload['emb_b64_chars']} chars |",
              f"| — of which metadata | {payload['metadata_bytes_mean']} |",
              f"| Batch of 20, raw | {payload['batch_of_20_raw_bytes']} |",
              f"| Batch of 20, gzipped | {payload['batch_of_20_gzip_bytes']} |",
              f"| **Per sighting, gzipped** | **{payload['gzip_bytes_per_sighting']}** |",
              "",
              f"A 640x360 JPEG crop is ~{JPEG_CROP_BYTES} B, so the vector is "
              f"{payload['vs_jpeg_crop_raw']}x smaller raw and "
              f"{payload['vs_jpeg_crop_gzip']}x smaller gzipped — and, more "
              "importantly, raw biometric pixels never leave the customer "
              "premises at all.", "",
              f"The vector is already float16 ({payload['emb_dim']} dims x 2 B = "
              f"1024 B); base64 inflates that by 4/3 to "
              f"{payload['emb_b64_chars'][0] if payload['emb_b64_chars'] else '?'} "
              "chars. That expansion is the entire reason the raw JSON mean sits "
              f"at {payload['payload_bytes_mean']} B rather than under the 1500 B "
              "budget.", ""]

    L += ["## Gate criteria", "", "| Criterion | Value | Threshold | Result |",
          "|---|---|---|---|"]
    for k, v in crit.items():
        L.append(f"| `{k}` | {v['value']} | {v.get('threshold','—')} | "
                 f"{'PASS' if v['pass'] else '**FAIL**'} |")
    L += ["", f"**GATE 3: {'MET' if met else 'NOT MET'}**", ""]

    if not crit["mean_payload_under_1500B"]["pass"] and payload:
        L += ["### On the payload budget", "",
              f"Raw JSON is {payload['payload_bytes_mean']} B against a 1500 B "
              f"budget — over by "
              f"{payload['payload_bytes_mean']-PAYLOAD_BUDGET:.0f} B. The vector "
              "is already at the plan's recommended float16; the overage is "
              "base64's 4/3 expansion, and the remaining honest levers are "
              "transport compression or a binary content type.", "",
              f"Measured with gzip on a batch of 20, the real cost is "
              f"**{payload['gzip_bytes_per_sighting']} B per sighting**, which is "
              f"{'under' if payload['gzip_bytes_per_sighting'] < PAYLOAD_BUDGET else 'still over'} "
              "the budget. The gate is reported against the raw figure because "
              "that is what the plan specifies; the gzip figure is the one an "
              "egress bill is computed from.", ""]

    MD.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {OUT}\nwrote {MD}")
    return 0 if met else 1


if __name__ == "__main__":
    sys.exit(main())
