# Round 2 — Identity and Asset Recognition

Round 1 built an edge-to-cloud camera pipeline that could detect, record and
deliver. Round 2 makes it answer three harder questions: **who is this**,
**what is this**, and **can you prove what you did with it**.

Everything below is reproducible from the repository. Every number cited here
was produced by a script in `tools/` reading real logs or real data, and every
number has a file behind it under `results/`. Where something could not be
measured it says so rather than estimating.

---

## What was added

| Capability | Where it lives | What it gives you |
|---|---|---|
| Control plane (cloud → edge) | `edge/inbox.py`, `cloud/api.py` | versioned, idempotent, resumable delivery of thresholds, model pins, consent revocations, class manifests |
| Person re-identification | `edge/embed.py`, `cloud/identity.py` | one 512-d vector per tracklet, matched against a gallery |
| Three-way identity resolution | `cloud/resolver.py` | `MATCH` / `NEW` / `REVIEW` — the system can say "I don't know" |
| Open-set object classes | `tools/add_class.py` | a new class becomes a database write, not a retrain |
| Governance | `cloud/api.py`, `edge/retention.py` | DSAR export, erasure with receipt, TTL sweep, audit trail |
| Observability | `tools/exporter.py`, `ops/` | Prometheus + Grafana over the same log the test harness reads |

The single idea underneath most of it: **the outbox pattern, reversed.**
Round 1 already had versioned, idempotent, store-and-forward delivery going
up. Pointing the same guarantee downward gives a control plane, and seven
apparently separate features collapse into one channel.

---

## Running it

```bash
source .env                      # never hardcode paths; everything resolves from here

# 1. schema
sqlite3 data/edge_a.db < edge/schema.sql
psql "$DATABASE_URL" -f cloud/schema.sql

# 2. camera simulator
nohup ops/run_mediamtx.sh &
sim/build_topology.sh data/datasets/shanghaitech/training/videos   # once
sim/spawn.sh

# 3. cloud
nohup python3 -m uvicorn cloud.api:app --host 0.0.0.0 --port 8000 &
nohup python3 cloud/resolver.py --loop &

# 4. edge
edge/run.sh a &
edge/run.sh b &

# 5. observability
nohup python3 tools/exporter.py &
docker run -d --name prometheus -p 9090:9090 \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD/ops/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v "$PWD/ops/alerts.yml:/etc/prometheus/alerts.yml:ro" prom/prometheus

# 6. ALWAYS verify
python3 tools/verify.py data/logs/edge_a.jsonl --strict
```

`tools/verify.py` is the important one. It asserts invariants on the system's
own telemetry and exits non-zero when one breaks. Run it after every change.

### Reproducing the measurements

```bash
python3 tools/fetch_datasets.py            # Market-1501 + object datasets
python3 tools/eval_reid.py <model.onnx>    # Rank-1, mAP, margin
python3 tools/face_feasibility.py          # face vs body, on your own footage
python3 tools/test_embed.py                # 6 embedding contract tests
python3 tools/calibrate_thresholds.py      # the (tau, delta) sweep
python3 tools/build_gt.py --pg             # cross-camera ground truth
python3 tools/eval_crosscam.py             # link recall on your own topology
python3 tools/bench/scale_bench.py         # 1K -> 100K latency curve
python3 tools/bench/arch_benchmark.py      # monolithic vs tiered detection
python3 tools/consolidate.py               # every claim -> its evidence file
```

`tools/consolidate.py` exits non-zero if any question lacks an evidence file.
That is what keeps the numbers in this README honest.

---

## What the measurements found

### Face recognition is not usable on this footage

| | |
|---|---|
| persons detected | 237 |
| faces found inside those boxes | 2 |
| faces at or above the 60 px interocular requirement | **0** |
| interocular distance, median / max | 3.1 px / 3.7 px |

A 16x shortfall against the threshold for reliable 1:N matching. This is a
property of camera mounting and focal length, not of the face model, so no
better face model changes it. **Body re-identification has to carry identity
here**, and face is at best a confirmatory signal on close-range cameras.

### The margin, not the score, is what separates two people

The re-identification model reaches Rank-1 **0.601** and mAP **0.337** on
Market-1501 — a cross-domain result, since the weights are MSMT17-trained. The
number that actually drives the design is different:

**margin p05 = 0.0009.** One query in twenty is separated from the *wrong*
identity by under a thousandth of a cosine unit.

Sweeping the accept threshold against the required margin shows exactly what
that costs:

| required margin | precision | auto-decided | sent to review |
|---|---|---|---|
| 0.00 | 0.673 | 100% | 0% |
| 0.01 | 0.725 | 60% | 40% |
| 0.02 | 0.752 | 33% | 67% |
| 0.03 | 0.803 | 18% | 82% |
| 0.09 | 0.909 | 1% | 99% |

The top row is a system with a single threshold: it answers everything, and
**it is wrong on a third of its answers**. Each step down converts false
positives into review items. Halving false positives costs 67 points of
automation. That trade is the product decision, and it belongs in a table
rather than in a constant somewhere in the code.

### A brute-force index is correct until about 100,000 identities

| gallery | exact search p95 | verdict |
|---|---|---|
| 10,000 | 11.7 ms | no index needed |
| 100,000 | **49.7 ms** | at the 50 ms budget |
| ~100,565 | crossover | index becomes necessary |

Two results that contradict common assumptions:

- **float16 is 17x *slower* than float32 for search** (874 ms vs 49.7 ms at
  100K), because there is no optimised CPU path for it. float16 is still the
  right choice *on the wire*, where it halves egress. Same format, opposite
  conclusions, depending on whether you are moving it or multiplying it.
- **Approximate search degrades badly in a densely clustered space.** Recall@1
  against exact search falls from 0.985 at 1K to 0.170 at 100K. Partly real,
  partly a measurement artifact — when top-1 and top-2 differ by 0.0009,
  "did you return the exact best" is not a meaningful question. Above the
  crossover the correct design is approximate search followed by exact
  re-ranking, not approximate search alone.

Storage is not the constraint: **35 KB per identity**, so ~35 GB at one
million, roughly $3.50/month. The constraints are RAM to hold the index and
the compute to re-embed everything when the model changes.

### Tiered detection wins above ~100 object classes, and not before

| approach | mean latency | models | cost to add one class |
|---|---|---|---|
| one model, all classes | 350 ms | 1 | full retrain and redeploy |
| propose, then classify | 390 ms | 2 | a gallery write |

Below ~100 classes the monolithic model is simply faster. Above it, the
tiered approach wins because its proposal stage is class-agnostic and flat.

The honest caveat: **stage-1 recall is a hard ceiling.** An object the
proposal stage misses can never be recovered by the classifier, so the tiered
approach loses on small and heavily occluded objects regardless of class
count.

Also measured, and worth knowing before planning around it: **reducing input
resolution saves nothing here.** 384 → 352 ms, 320 → 346 ms, 256 → 361 ms.
The exported model declares a fixed input, so resolution is a letterbox
target, not a compute lever. Changing it requires a re-export, which is a
release rather than a runtime flag.

### A new object class goes live in under two minutes

A class the model has never seen — forklift — was live at the edge in
**1.7 minutes with zero training steps and zero GPU hours**, shipped as a
prototype vector through the control plane.

The accuracy is poor and is reported as such: recall 1.0 but precision
**0.183** against 133 negatives, and 0.538 at the best swept threshold. Two
separate causes, worth keeping separate:

1. The embedder is a *person* re-identification model being asked to
   represent vehicles. Out of domain.
2. The acceptance threshold was derived from intra-class spread alone, which
   never looks at where the negatives sit. **A threshold calibrated only on
   positives is calibrated on half the problem.**

### One global accuracy number hides customer-level regressions

The same candidate model, evaluated per customer:

| customer | verdict | reason |
|---|---|---|
| A | **roll back** | precision 0.940 vs 0.965 |
| B | promote | precision 0.973 |
| C | **roll back** | p95 latency 420 ms vs 300 ms |

Fleet-wide precision was **0.9558 for the candidate against 0.9517 for the
baseline** — so a single global number *promotes this deploy* while two of
three customers regress. That gap is the entire argument for cohort-scoped
metrics, and it is why the rollback is per-tenant: site A ended pinned to the
baseline while site B ran the candidate, both verified in their edge
databases.

### The guarantees hold under failure

| property | measured |
|---|---|
| duplicate events after a 5-minute network partition | **0** |
| dead-lettered events | 0 |
| delivery backlog | 102 queued → drained to 0 |
| retry counter under backpressure | live (max 8 attempts) |

Zero duplicates is the one that matters. A sighting replayed after a partition
must not create a second appearance of a person — that would corrupt the
identity graph, not merely duplicate a row.

Erasure was exercised end to end: consent revoked → directive fanned out →
**edge purge applied in 0.82 s** → cloud records 0, edge records 0, and **14
audit rows deliberately retained**. You cannot prove a deletion using data you
deleted. A correction to one specific false match propagated to the edge in
**0.95 s**.

### The cost model

Mean inference latency went from **1493 ms to 393 ms**, which for 100 cameras
at 4 fps is **$17,438 → $4,590 per month, a 73.7% reduction**.

The levers, each with its measured saving, including the two that did not
work:

| lever | measured | status |
|---|---|---|
| shared right-sized inference session | −73.7% | shipped |
| embed once per tracklet, not per frame | −82% of embedding cost | shipped |
| motion gate | −18.6% | shipped |
| temporal caching | −67.5% of inferences | available |
| **INT8 quantization** | **+50% slower** | **rejected** |
| **input resolution reduction** | **0%** | **not available** |

INT8 was 1.50x slower *and* 5.8 mAP points worse on this CPU, because the
quantize/dequantize nodes never fuse. It is a lever that has to be measured
per target rather than assumed.

---

## How this data is useful

**It converts architecture arguments into engineering decisions.** "Should we
use face recognition or body re-identification" is unanswerable in the
abstract and trivial once you know 0 of 237 detected people carry a usable
face. "When do we need a vector database" is a matter of opinion until you
measure the crossover at ~100,565 vectors. "Should detection be one model or
many" resolves the moment you know the crossover is ~100 classes.

**It makes thresholds defensible.** The accept threshold and required margin
are a point on a measured surface, and the surface is in
`results/identity/threshold_sweep.csv`. When someone asks why the system sends
82% of ambiguous cases to a human, the answer is a row in a table rather than
a preference.

**It prices the trade-offs.** Halving false positives costs 67 points of
automation. Adding a customer class takes two minutes and costs precision.
Each of these is a number a product owner can decide against.

**It tells you which optimisations to skip.** Two of the six cost levers were
measured negative. Any plan that assumed INT8 and resolution reduction would
help would have spent weeks recovering nothing.

**It bounds what the system should be trusted with today.** Cross-camera
identity linking currently measures **0.00** link recall on our own topology.
Raising the new-identity floor fixed a real calibration bug — the floor sat
*below* the similarity between two different people, so the gallery never
learned anyone — and the linking still failed, which localises the remaining
problem in the model rather than the logic. That is a deployment gate, and it
is better to know it from a measurement than from a customer.

**It leaves an audit trail that survives deletion.** Every identity lookup is
logged, because under GDPR Article 15 and DPDP section 11 the *access* is the
sensitive operation, not just the storage. Looking someone up is itself the
surveillance act.

---

## Known limits

- **The re-identification model is trained on a different dataset than it is
  evaluated on.** Rank-1 0.601 is a cross-domain transfer result. An in-domain
  checkpoint would move Rank-1, margin, cross-camera linking and open-set
  precision together — they all trace to this one cause.
- **Cross-camera linking does not work yet.** Reported as 0.00 rather than
  omitted.
- **Ground truth is only valid in the first ~319 seconds after the simulator
  starts.** The looping clips have different durations, so the cameras drift
  out of phase and the known time offsets stop holding.
- **The left-half/right-half camera pair cannot be labelled by time offset
  alone**, because a person has to physically walk between the two views,
  adding a variable transit time. Its ground-truth pairs are excluded.
- **Open-set object classes use a person model.** The mechanism is
  demonstrated; the accuracy ceiling is not.
- **1M-identity scaling is extrapolated**, not measured.
- **Segments are never uploaded**, so "no footage lost" means locally durable,
  not replicated.
- **The simulator is looping video.** Real cameras bring clock skew, firmware
  quirks and codec variation this does not reproduce.

---

## Two bugs worth naming

Both were invisible until something asserted on them, which is the argument
for `tools/verify.py` existing at all.

**Deletion could never delete.** Disk-pressure eviction required
`uploaded = 1`, and nothing in the system ever sets that flag on recorded
video — 0 of 29,843 segments qualified. The eviction path ran, reported
success, and freed nothing while the disk filled. It now evicts uploaded
segments first and falls back to the oldest unuploaded ones under sustained
pressure, because a full disk stops every camera: protecting old footage by
losing all future footage is the losing side of that trade.

**The API token regenerated on every shell.** The environment file generated a
fresh random token each time it was sourced, so the edge and the cloud never
held the same one and every upload was rejected. The store-and-forward layer
behaved correctly throughout — it queued, retried, and dead-lettered on a 4xx
exactly as designed — which is why the failure was recoverable: all 357
rejected events replayed cleanly once the token was fixed.
