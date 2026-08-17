# Video Intelligence: Edge-to-Cloud Distributed Camera Network

**Author:** Prateek Srivastava

A distributed video surveillance prototype that runs AI inference on-premise,
buffers durably through network partitions, and synchronises telemetry to a
cloud aggregator. Built for sites with variable network quality (5 Mbps to
100 Mbps) where footage cannot be lost and bandwidth is a first-order cost.

The system is organised as **three planes**. Most surveillance prototypes build
only the first one.

| Plane | Direction | Carries |
|---|---|---|
| **Data** | edge to cloud | detections, activity events, sightings, health telemetry |
| **Control** | cloud to edge | model pins, thresholds, class manifests, consent revocations |
| **Governance** | cross-cutting | retention TTL, consent basis, append-only audit trail |

---

## Architecture

```
                    EDGE NODE (on-prem)                        CLOUD
  ┌───────────┐    ┌──────────────────────────────┐    ┌──────────────────┐
  │ IP cameras│    │  decode  (PyAV, 4 fps)       │    │ FastAPI ingest   │
  │ RTSP/ONVIF├───▶│     ↓                        │    │  idempotent      │
  │  MediaMTX │    │  motion gate  (OpenCV MOG2)  │    │     ↓            │
  └───────────┘    │     ↓                        │    │ Postgres / Neon  │
                   │  inference pool (shared ONNX)│    │     ↓            │
  ┌───────────┐    │     ↓                        │    │ identity resolve │
  │  ffmpeg   │    │  tracker  (IoU + confirm)    │    │  margin test     │
  │ segmenter ├───▶│     ↓                        │    │     ↓            │
  │ 10s chunks│    │  zones  (dwell/tripwire)     │    │ adjudication     │
  └───────────┘    │     ↓                        │    │  queue + canary  │
                   │  embedder (1 per tracklet)   │    └──────────────────┘
                   │     ↓                        │             │
                   │  SQLite WAL outbox ──────────┼──── data ───┘
                   │  control inbox      ◀────────┼─── control ─┐
                   │  retention sweeper           │             │
                   └──────────────────────────────┘             │
                                                        directives, versioned
```

### The pipeline is a cost cascade

Each stage is more expensive than the last and only sees what survived the
previous one:

```
gate  →  detect  →  track  →  zone  →  embed

cost_total = Σᵢ ( Πⱼ<ᵢ pass_rateⱼ ) × costᵢ
```

The motion gate rejects static frames before any tensor is allocated. The
tracker requires `min_hits` consecutive associations before a detection becomes
a reportable object. Zones require spatial *and* temporal validity. The embedder
runs once per closed tracklet rather than once per frame.

A cascade only pays when early-stage rejection is high. On busy retail or campus
footage where people are almost always moving, a motion gate earns much less
than the marketing suggests. Measured rejection on our test clips ranged from
**18.6% to 43.2%** depending on scene activity. That range is the honest answer,
not a single flattering number.

### Only vectors cross the WAN

Because detection and embedding both run on-prem, a person sighting leaves the
site as a 512-dimension float16 vector plus metadata (**~1.2 KB**), never as a
crop (~15 KB) and never as a frame. That is roughly 12x less egress, and raw
biometric pixels never leave customer premises, which turns most of the privacy
requirement into an architecture property rather than a policy document.

---

## Repository layout

```
edge/                    on-premise agent
  config.py              paths, tunables, pool sizing (no hardcoded paths)
  agent.py               entrypoint, supervised threads, heartbeat
  decode.py              RTSP → frames at target fps
  gate.py                MOG2 motion gate (cascade stage 0)
  infer_pool.py          ONE shared ONNX session, bounded queue, drop metric
  detect.py              RF-DETR decode, letterbox + ImageNet norm, two-band conf
  tracker.py             IoU-gated association, temporal confirmation, tracklets
  zones.py               dwell (loitering), tripwire (entry), crowd formation
  embed.py               per-tracklet ReID descriptor
  recorder.py            segment indexing
  completeness.py        recording coverage SLI, with bounds assertion
  outbox.py              store-and-forward up: backoff, dead-letter, retention
  inbox.py               control plane down: versioned, idempotent, resumable
  policy.py              per-tenant / per-camera versioned flags
  retention.py           TTL sweep, disk-pressure eviction, consent purge
  schema.sql             SQLite  (11 tables)   ← line 1 says "Edge schema"
  roster.yaml            which cameras get inference vs record-only
  zones.yaml             per-camera zone geometry
  run.sh                 launch a site

cloud/
  api.py                 FastAPI: /events, /control, /control/ack, /health/fleet
  identity.py            gallery, cosine search, margin test, three-way resolve
  adjudicate.py          review queue, hard-example mining, cohort canary gate
  schema.sql             Postgres (13 tables)  ← line 1 says "Cloud schema"

tools/
  verify.py              telemetry invariant harness (run this every time)

sim/                     RTSP camera simulator (ffmpeg + MediaMTX)
ops/                     MediaMTX supervisor
data/                    models, sim clips, segments, SQLite DBs, JSONL logs
docs/ui/                 Streamlit fleet dashboard
```

### File placement check

If files landed flat during export, fix them up before running:

```bash
mkdir -p cloud tools
mv identity.py adjudicate.py cloud/  2>/dev/null
mv api.py cloud/                     2>/dev/null
mv verify.py tools/                  2>/dev/null

# the two schema.sql files are NOT interchangeable
head -1 edge/schema.sql    # "Edge schema"  → SQLite,  stays in edge/
head -1 cloud/schema.sql   # "Cloud schema" → Postgres, stays in cloud/
```

`cloud/identity.py`, `cloud/adjudicate.py` and `tools/verify.py` have no
intra-repo imports, so a wrong directory will not raise an ImportError. It will
just quietly sit in the wrong architectural layer. The schema files are the only
placement error that fails loudly, and it fails at agent boot.

---

## Measured performance

All numbers from `data/logs/*.jsonl` on a 1 vCPU / 8 GB devcontainer, RF-DETR
Nano INT8 at 384x384 on CPU. Nothing here is estimated.

| Metric | Value |
|---|---|
| Detection latency, mean | 282 ms |
| Detection latency, p95 | 307 ms |
| Motion gate rejection | 18.6% to 43.2% (scene dependent) |
| Detections per frame | 1 to 4, varies with scene |
| Track IDs issued | 13 over 75 inferences (stable, low churn) |
| Segment throughput | ~1.7 GB per camera-hour at 640x360 / 900 kbps |
| Sighting payload | ~1.2 KB per tracklet |
| Outbox recovery | 934-deep backlog drained to 1 after partition |

### Inference pool sizing matters more than the model

One ONNX session per camera thread on a single core is pathological:

| Concurrent sessions | Mean latency |
|---|---|
| 1 | 619 ms |
| 3 | 1902 ms (3.07x) |

The agent therefore runs **one shared session** sized to the host, with a
bounded queue in front. When the queue saturates, frames are dropped **and
counted**. Dropping under load is correct behaviour for a real-time system.
Silently falling behind is not, and an uncounted drop is indistinguishable from
a healthy pipeline.

Enabling `ORT_ENABLE_ALL` graph optimisation independently took a single
session from 619 ms to 282 ms.

### Preprocessing is not a detail

RF-DETR expects ImageNet mean/std normalisation and an aspect-preserving
resize. Skipping either silently degrades every downstream number:

| Preprocessing | avg max score | detections at 0.5 |
|---|---|---|
| Plain resize + /255 | 0.646 | 1.0 |
| Plain resize + ImageNet | 0.832 | 1.3 |
| **Letterbox + ImageNet** | **0.827** | **1.7** |

---

## Verification: the invariant harness

`tools/verify.py` asserts physical invariants on the telemetry the system emits
about itself, and exits non-zero when one breaks.

```bash
python tools/verify.py data/logs/edge_a.jsonl --strict
```

Checks currently enforced:

| Check | Invariant |
|---|---|
| `detect_cap_not_always_hit` | the output cap must not bind on every frame |
| `detect_count_has_variance` | a constant detection count is a bug, never a scene |
| `completeness_within_bounds` | coverage ratio ∈ [0, 1.05] |
| `inference_keeps_up` | p95 detect latency ≤ the per-frame budget at target fps |
| `infer_drop_rate_bounded` | queue drop rate < 20% |
| `outbox_drains` | final depth bounded, zero dead letters |
| `all_configured_cameras_report` | every camera in `roster.yaml` produces detections |
| `no_thread_crashes` | supervised workers stayed up |
| `no_critical_events` | no critical-severity telemetry in the run |

**Why this file exists.** An earlier build of this pipeline shipped with the
ONNX outputs decoded in the wrong order. The model declares `dets [1,300,4]`
before `labels [1,300,91]`; the decoder assumed the reverse, so it thresholded a
sigmoid of box coordinates and read class logits as geometry. The failure was
fully visible in telemetry from the first run: `n_det` was exactly 10 in
**3,211 out of 3,211** detection records, because a hard cap was the only thing
limiting output. The completeness SLI reported coverage ratios up to **2.33**,
which is physically impossible, in 168 of 168 samples.

Nothing was hidden. There was a dashboard that faithfully displayed the number,
and no invariant that ever asked whether the number was possible. Dashboards
serve people who are already looking. Invariants serve the three days before
anyone looks.

Two habits came out of it and are now enforced in code:

1. **Resolve model outputs by tensor shape, never by index.** `detect.py`
   locates the box tensor by `shape[-1] == 4`, so a re-export with a different
   output order keeps working instead of producing confident garbage.
2. **Every SLI carries an assertion on its own range.** A metric that can read
   233% can also read 100% while a camera is dead.

---

## Setup

### Prerequisites

```bash
pip install opencv-python-headless av onnxruntime pyyaml scipy numpy \
            fastapi uvicorn psycopg2-binary requests streamlit pandas
```

MediaMTX at `/usr/local/bin/mediamtx` for the RTSP simulator. Linux or
GitHub Codespaces.

### Environment

No absolute paths are compiled in. Everything resolves from `VI_ROOT`.

```bash
export VI_ROOT="$PWD"
export DATA="$PWD/data"
export DATABASE_URL="postgresql://...neon.tech/..."
export VI_API_TOKEN="$(openssl rand -hex 24)"
export CLOUD_API="http://127.0.0.1:8000"
```

Optional tuning:

| Variable | Default | Purpose |
|---|---|---|
| `VI_INFER_WORKERS` | `cpu_count / 2` | shared inference workers |
| `VI_INFER_QUEUE_MAX` | `2 × workers` | backpressure depth before dropping |
| `VI_TARGET_FPS` | `4` | decode sampling rate |
| `VI_CONF_HIGH` | `0.50` | accept threshold |
| `VI_CONF_LOW` | `0.30` | abstain floor; between the two goes to review |
| `VI_RETAIN_DAYS` | `30` | retention TTL |
| `VI_TENANT` | `default` | tenant key for policy scoping |

### Schema

Both files are idempotent and additive, so they are safe to run against
existing databases.

```bash
sqlite3 data/edge_a.db < edge/schema.sql
sqlite3 data/edge_b.db < edge/schema.sql
psql "$DATABASE_URL" -f cloud/schema.sql
```

### Run

```bash
# clean slate
pkill -f "python -m edge.agent"; pkill -f ffmpeg; pkill -f mediamtx; pkill -f uvicorn

# 1. RTSP simulator
nohup ops/run_mediamtx.sh >/dev/null 2>&1 &
sleep 2
sim/spawn.sh                      # 12 looping RTSP streams, 2 site cohorts

# 2. cloud
nohup uvicorn cloud.api:app --host 0.0.0.0 --port 8000 >/dev/null 2>&1 &

# 3. edges
edge/run.sh a &
edge/run.sh b &

# 4. dashboard
streamlit run docs/ui/app.py --server.port 8501 --server.address 0.0.0.0

# 5. verify (always)
python tools/verify.py data/logs/edge_a.jsonl data/logs/edge_b.jsonl --strict
```

---

## Control plane

The control plane is the outbox reversed: versioned, idempotent, resumable
across partition, acknowledged. `directive_id` is the downward equivalent of
`idem_key`. Edges **pull**, because they sit behind customer NAT with no
inbound reachability, so long-poll is what survives a real firewall.

| Directive | Effect at the edge |
|---|---|
| `threshold` | retune confidence, dwell, or confirmation depth per camera |
| `model_pin` | stage weights, verify sha256, atomic swap; per-tenant version |
| `class_manifest` | new detection classes; embedding classes go live instantly |
| `gallery_delta` | refresh identity centroids as appearance drifts |
| `consent_revoke` | purge subject locally, including source segments |
| `retention` | change TTL window |
| `exclusion_pair` | hard never-match constraint on two specific identities |

These are not seven features. They are one channel with seven payload types.

## Identity resolution

Cross-camera identity uses a **margin test**, not a single similarity
threshold. A top-1 score of 0.83 against a runner-up of 0.81 is not a match, it
is a coin flip with a confident tone of voice.

```
MATCH   top1 ≥ τ  AND  (top1 − top2) ≥ δ
NEW     top1 < τ_low
REVIEW  everything else  →  adjudication queue
```

Three outcomes, never two. The review band converts a trust-destroying false
positive into a slightly slower correct answer, and it requires no model change.

Two co-visible tracks in the same frame cannot be the same person. That
constraint is free and is passed to `resolve()` as `known_not`.

Scale guidance for 512-d float32 on one core:

| Identities | Approach | Index size | Query |
|---|---|---|---|
| 1K | flat numpy | 2 MB | ~0.1 ms |
| 10K | flat numpy | 20 MB | ~1 ms |
| 100K | pgvector IVFFLAT | 200 MB | ~10 ms (the knee) |
| 1M | HNSW | 2 GB | ~1 to 5 ms |
| 10M+ | region shards + exact re-rank of top 100 | | |

Brute force is simply correct below roughly 100K identities. Storage per
identity at 1M scale is ~32 KB (centroid + 5 exemplars + ~200 sighting rows +
graph overhead), so ~32 GB total. Storage is not the cost. Resident RAM for the
ANN index and re-embedding on model upgrade are the costs.

## Privacy and retention

Retention is a **column on the record** (`retain_until`, `consent_basis`), not a
separate subsystem bolted on later. Enforcement runs at the edge because the
edge is what holds pixels; a cloud-only deletion removes the index and leaves
the evidence.

The audit log is append-only and is deliberately exempt from the TTL sweep,
because you must still be able to prove a deletion happened after the data is
gone. Every identity **read** is logged, not just every write: looking someone
up is itself the surveillance act.

Three hard problems this design acknowledges rather than papers over:

1. A deletion that a backup restore can undo is not a deletion.
2. A gallery centroid derived from a deleted sighting still encodes the
   subject, so revocation must trigger recomputation, not just row removal.
3. Honouring "delete everything about me" requires matching against the person
   who asked not to be matched. That needs a narrowly scoped, audited,
   time-boxed deletion index kept separate from the operational gallery.

---

## Known limitations

Stated plainly, because a prototype that claims no limits is a prototype nobody
should trust.

- **ReID model is a placeholder.** `edge/embed.py` uses a striped HSV histogram
  descriptor when no ONNX ReID model is present. That is enough to exercise the
  interfaces end to end; it is not person re-identification. Drop an OSNet or
  CLIP-ReID export at `data/models/reid-osnet-int8.onnx` for real results.
- **No cross-camera resolution running yet.** `cloud/identity.py` is
  implemented and unit-testable but is not yet wired into the ingest path.
- **Detector is COCO pretrained.** No domain adaptation to the deployment
  scenes, so accuracy will vary by site. Per-site cohort metrics exist to
  measure exactly that; the retraining loop is not yet automated.
- **Segments are never uploaded.** `segments.uploaded` stays 0. Evidence-clip
  upload on event is designed but not built, so "no footage lost" currently
  means locally durable, not replicated.
- **`cloud/adjudicate.py` has no UI.** The queue, verdict handling and canary
  gate are implemented as library functions with no reviewer front-end.
- **Single-region cloud.** Fine to roughly 100 cameras. See below.
- **Simulated cameras only.** ffmpeg looping pre-encoded clips over RTSP. Real
  cameras bring firmware quirks, clock skew and codec variation that this does
  not reproduce.

## Scale roadmap

| Scale | Ingest | Identity | Fails because |
|---|---|---|---|
| 100 cameras | FastAPI → Postgres (current) | flat numpy | direct writes contend under IOPS |
| 1,000 | Kafka between edge and DB | pgvector IVFFLAT | single cluster routing latency |
| 10,000+ | regional cell sharding | HNSW per region + global re-rank | global write locking, WAN traversal |

The edge tier does not change across these transitions. That is the point of
putting inference and durability on-premise: the edge scales by replication,
and only the aggregation tier has to be re-architected.

---

## Datasets

**Simulation clips.** `sim/preencode.sh` normalises source video into a single
controlled camera profile (H.264, 640x360, 900 kbps, GOP 50) and
`sim/spawn.sh` publishes them as looping RTSP streams. Current clips come from
ShanghaiTech Campus. VIRAT and MOT17/MOT20 are better choices for pedestrian
density and activity variety.

**Model weights.** RF-DETR Nano exported to ONNX and quantised to INT8, input
locked at 384x384. For domain adaptation, fine-tune with PyTorch on
site-specific data (Roboflow Universe, CrowdHuman for dense crowds), re-export
at the same input size, and ship it through the `model_pin` directive rather
than by redeploying the agent.

**Note on the looping simulator.** `-stream_loop -1` resets presentation
timestamps at each wrap, which produces short segments at the loop boundary and
distorts naive coverage metrics. `completeness.py` measures against elapsed wall
time and recorded media duration for this reason.
