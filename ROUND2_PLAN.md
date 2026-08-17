# video-intel: Round 1 fixes and Round 2 extension plan

## Verdict: yes, extend the same repo. Do not start over.

Your PS-1 architecture is not just reusable for PS-3 and PS-4, it is the
*correct base*, and three specific decisions you already made are what make it
correct. Say these out loud in Round 2, because they are the reason your
answers will have continuity that a fresh design cannot fake.

**1. Idempotency keys.** `outbox.idem_key UNIQUE` at the edge plus
`ON CONFLICT (idem_key) DO NOTHING` in the cloud. This is the single most
reusable thing in the repo. PS-3 needs exactly this property again so a
sighting replayed after a partition does not create a phantom second
appearance of a person, which would corrupt the identity graph rather than
just duplicating a row.

**2. Inference at the edge.** This pre-decides PS-3 Q3.1 in your favour.
Because detection already runs on-prem, embeddings can too, which means a
1 KB vector crosses the WAN instead of a 15 KB crop, and raw biometric pixels
never leave the customer's premises. That is roughly 12x less egress and it
answers most of PS-3 Q3.2 (privacy) with an architecture choice rather than a
policy document.

**3. Store-and-forward with local durability.** The outbox pattern generalises
directly into the control plane you are missing. Reverse its direction and you
have versioned, idempotent, resumable delivery of model updates, consent
revocations, and threshold changes. Same guarantee, same mental model.

What is genuinely missing is not the data plane. It is the other two planes:

| Plane | Direction | Status |
|---|---|---|
| Data | edge to cloud | Built, with the bugs listed below |
| Control | cloud to edge | **Does not exist.** Needed by 6 of 6 Round 2 questions |
| Governance | cross-cutting | **Does not exist.** Needed by PS-3 Q3.2 entirely |

---

## Part A: correctness fixes (do these first, roughly 4 hours)

### A1. The detector is wired backwards. This is the one that matters.

The ONNX graph declares its outputs as:

```
output[0] = "dets"   [1, 300, 4]     boxes
output[1] = "labels" [1, 300, 91]    class logits
```

`edge/detect.py` read them as `logits, boxes = out[0][0], out[1][0]`, so it
took a sigmoid of *box coordinates*, thresholded that at 0.70, and then read
the first four *class logits* as `cx, cy, w, h`.

Verified on your own model and footage:

| | Buggy path | Fixed path |
|---|---|---|
| Queries passing 0.70 | 116 of 300 | 1 of 300 |
| Top detection | meaningless | person, score 0.884 |
| `n_det` in your log | exactly 10, all 3211 times | varies |

Two further preprocessing defects, also measured over 5 frames:

| Preprocessing | avg max score | avg detections at 0.5 |
|---|---|---|
| Plain resize + /255 (yours) | 0.646 | 1.0 |
| Plain resize + ImageNet norm | 0.832 | 1.3 |
| **Letterbox + ImageNet norm** | **0.827** | **1.7** |

RF-DETR expects ImageNet mean/std, and squashing 640x360 into 384x384
distorts aspect ratio. Both are fixed in the new `edge/detect.py`.

**Consequence for your deck.** The "taming false alarms" slide describes
truncation, not thresholding. The 692-object spike and the 52-object
"stabilized output" are tracker ID churn caused by boxes being left in
normalized [0,1] space. Do not repeat those numbers in Round 2.

**How to use this instead.** This is a real production incident that you found
by reading your own telemetry. The scoring page says the Founder Round
"separates operators who've debugged production incidents from architects who
have only designed systems theoretically." You now have one. Lead with it.

### A2. Everything else that is wrong

| # | File | Symptom in your data | Fix |
|---|---|---|---|
| 1 | `edge/detect.py` | `n_det == 10` in 3211/3211 records | Resolve outputs by shape, ImageNet norm, letterbox, pixel-space boxes, two-band confidence |
| 2 | `edge/cascade.py` | 1493 ms mean, 1933 ms p95 | One ONNX session per camera on 1 vCPU. Measured: 619 ms alone, 1902 ms with 3. Shared bounded pool |
| 3 | (new) `edge/infer_pool.py` | no drop metric existed | Bounded queue, explicit drops, `drop_rate` as an SLI |
| 4 | `edge/tracker.py` | 692-object spike | Normalized coords into `cdist`, no gating, no confirmation. Rewritten in pixel space with IoU gate and `min_hits` |
| 5 | `edge/completeness.py` | 168/168 readings above 1.0, max 2.33 | Wall-clock elapsed, media duration, and a hard bounds assertion |
| 6 | `edge/outbox.py` | `attempts=0` on all 3771 rows | Never incremented; non-2xx hot-looped. Backoff, dead-letter, retention |
| 7 | `edge/agent.py` | `edge_b.db` empty, 14 log lines | Hardcoded `/workspaces` paths, unsupervised threads. Config module, supervised restart, heartbeat |
| 8 | `cloud/api.py` | n/a | Sync psycopg2 inside `async def`, new connection per request, row-by-row insert, no auth. Pool + `to_thread` + `execute_values` + bearer token |
| 9 | `edge/zones.py` | `zones.yaml` never opened | `cascade.py` took the path and ignored it. Dwell, tripwire, crowd now implemented |
| 10 | `edge/retention.py` | 12,237 segments, 9.85 GB, `uploaded=0`, no delete path | Fills a 32 GB disk in ~18 h. TTL sweep, disk-pressure eviction, consent purge |
| 11 | `cloud/schema.sql` | n/a | No index on `events.created_at`, which the dashboard orders by |
| 12 | `auto.key`, `auto.crt` | committed to a public repo | Rotate, delete, purge from git history |

### A3. The harness that catches all of it

`tools/verify.py` asserts invariants on your telemetry. Run against your
existing log, unmodified, it produces:

```
FAIL  detect_cap_not_always_hit
      cap hit on 100.0% of 3211 detections.
FAIL  detect_count_has_variance
      1 distinct detection count across 3211 frames (mode=(10, 3211)).
FAIL  completeness_within_bounds
      168/168 readings outside [0, 1.05], max=2.33.
WARN  inference_keeps_up
      p95 detect latency 1933 ms against a 250 ms budget at 4 fps.
```

Every bug was already visible in `data/logs/edge_a.jsonl`. Nothing was hidden.
The data was there and nobody asked it a question. That observation *is* your
answer to PS-4 Q4.2a: dashboards are for humans who are already looking,
invariants are for the three days before anyone looks.

---

## Part B: Round 2 extension (roughly 12 hours)

### B1. The one architectural addition: mirror the outbox

Seven separate Round 2 requirements collapse into one mechanism.

| Requirement | Question | Directive kind |
|---|---|---|
| Tune sensitivity per customer | PS-2 Q2.3c | `threshold` |
| Roll back Customer A, forward Customer C | PS-4 Q4.2b | `model_pin` |
| Introduce a customer-specific class | PS-4 Q4.1c | `class_manifest` |
| Update embeddings as people age | PS-3 Q3.1d | `gallery_delta` |
| Enforce consent at scale | PS-3 Q3.2a | `consent_revoke` |
| Time-based deletion at scale | PS-3 Q3.2c | `retention` |
| Stop matching John as David, today | PS-3 Q3.3b | `exclusion_pair` |

They are not seven features. They are one versioned control channel with seven
payload types. Say that sentence in the Round 2 answer for Q4.2 and it does
double duty for Q3.1, Q3.2 and Q3.3.

Semantics are the outbox, reversed: versioned, idempotent (`directive_id` is
the downward `idem_key`), resumable, acknowledged. **Pull, not push**, because
edges sit behind customer NAT with no inbound reachability. Long-poll costs one
idle connection per site and survives a real firewall.

### B2. New files

| File | Purpose | Answers |
|---|---|---|
| `edge/config.py` | kills hardcoded paths, sizes the pool to the box | Q1.1 (30-min deploy) |
| `edge/infer_pool.py` | shared session, bounded queue, drop metric | Q1.3, Q4.3 |
| `edge/inbox.py` | control plane, 7 appliers | Q2.3, Q3.1d, Q3.2, Q3.3, Q4.1c, Q4.2 |
| `edge/policy.py` | widens your unused `policies` table into per-tenant flags | Q4.2b |
| `edge/zones.py` | dwell, tripwire, crowd | Q2.1, Q2.3 |
| `edge/embed.py` | ReID embedding, one per tracklet | Q3.1a, Q3.1e |
| `edge/retention.py` | TTL, disk pressure, consent purge | Q3.2c, Q3.2d |
| `cloud/identity.py` | gallery, margin test, three-way resolve | Q3.1, Q3.3 |
| `cloud/adjudicate.py` | review queue, hard-example mining, canary gate | Q2.3c, Q3.3c, Q4.1c, Q4.2 |
| `tools/verify.py` | invariant harness | Q1.3, Q4.2a |

### B3. The two ideas worth building the answers around

**Idea 1: the abstain band.** Both of your worst bugs and the John/David
scenario are the same failure: the system produced an answer where the correct
output was "I do not know."

- Round 1 detector: `res[:10]` truncated instead of abstaining.
- PS-3 Q3.3: matched on top-1 similarity instead of abstaining.

The fix is the same shape in both places. Not one threshold, but two, with a
review band between them, and a **margin test** on top:

```
MATCH   top1 >= tau AND (top1 - top2) >= delta
NEW     top1 <  tau_low
REVIEW  everything else
```

John at 0.81 and David at 0.83 gives a margin of 0.02. A 0.02 margin is not a
match decision, it is a coin flip with a confident tone of voice. Abstaining
there converts a trust-destroying false positive into a slightly slower
correct answer, and it costs no model change.

**Idea 2: the cost cascade, with your measured numbers.**

```
gate (0.814 pass) -> detect -> track -> zone -> embed

cost_total = SUM_i ( PROD_{j<i} pass_rate_j ) * cost_i
```

One equation answers PS-1 Q1.2, PS-4 Q4.1d (Strategy A vs B), PS-4 Q4.3b, and
PS-3 Q3.1a (ReID as the cheap wide stage, face as the expensive narrow one).
Your gate rejects 18.6%, not the 21% in the deck, and on busy retail footage a
motion gate genuinely does not earn much. **Say that.** A cascade only pays
when early rejection is high, and naming the condition under which your own
optimisation fails is worth more than the optimisation.

### B4. Numbers to have ready

Storage per identity at 1M (PS-3 Q3.1e):

```
centroid, 512 x fp16                    1.0 KB
5 exemplar vectors                      5.0 KB
~200 sighting rows x 120 B             24.0 KB
HNSW graph overhead (M=16)              1.5 KB
                                       -------
                                       ~32 KB  ->  ~32 GB at 1M
```

Roughly $3.20/month at $0.10/GB. Storage is not the cost. RAM for a resident
ANN index and re-embedding on model upgrade are the costs.

When a vector DB becomes necessary (PS-3 Q3.1c), for 512-d float32 on one core:

| Identities | Approach | Size | Latency |
|---|---|---|---|
| 1K | flat numpy | 2 MB | ~0.1 ms |
| 10K | flat numpy | 20 MB | ~1 ms |
| 100K | pgvector IVFFLAT | 200 MB | ~10 ms, the knee |
| 1M | HNSW | 2 GB | ~1 to 5 ms |
| 10M+ | shard by region + exact re-rank | | |

Brute force is simply correct below ~100K. Reaching for a vector DB at 1K
identities signals that nobody measured brute force.

Bandwidth, from your own run: 12,237 segments, 9.85 GB, 6 cameras at 640x360
and 900 kbps over about 5.7 hours, so ~1.7 GB per camera-hour. Compare a
sighting at ~1.2 KB.

---

## Part C: order of work

| Phase | Hours | Work | Gate |
|---|---|---|---|
| 0 | 0.5 | Rotate `auto.key`, purge from history, add to `.gitignore` | `git log -p` shows no key |
| 1 | 1.5 | Drop in `config.py`, `detect.py`, `infer_pool.py`, `tracker.py` | `verify.py` cap and variance checks pass |
| 2 | 1.0 | `completeness.py`, `outbox.py`, schema migration | bounds check passes, `attempts` non-zero under fault |
| 3 | 1.0 | `agent.py`, `cloud/api.py`, indexes | site B produces detections, heartbeat clean |
| 4 | 0.5 | Re-run 30 min, capture new numbers, redo the deck slides | `verify.py` exits 0 |
| 5 | 3.0 | `zones.py` live, activity events flowing | loitering fires on real dwell |
| 6 | 3.0 | `inbox.py` + `policy.py` + cloud `/control` | threshold change reaches edge in one poll |
| 7 | 3.0 | `embed.py` + `cloud/identity.py` | sighting resolves to match / new / review |
| 8 | 2.0 | `adjudicate.py` + canary gate | a verdict emits an `exclusion_pair` directive |
| 9 | 1.0 | `retention.py` scheduled | disk stops growing, purge receipt in `audit_log` |

Phases 0 to 4 are the correctness recovery, about 4.5 hours, and are worth
doing before you write a single Round 2 answer, because they change the numbers
you would otherwise be quoting.

---

## Part D: migration commands

```bash
# 0. secrets first
git rm --cached auto.key auto.crt
printf 'auto.key\nauto.crt\n*.pem\n' >> .gitignore
# then rotate the key and purge history with git-filter-repo

# 1. back up the evidence before touching anything
cp -r data/logs data/logs.pre-fix
cp data/edge_a.db data/edge_a.db.pre-fix

# 2. drop in the patched files (all additive or in-place replacements)
cp -r patch/edge/*    edge/
cp -r patch/cloud/*   cloud/
mkdir -p tools && cp patch/tools/verify.py tools/

# 3. schema is idempotent, safe on the existing db
sqlite3 data/edge_a.db < edge/schema.sql
sqlite3 data/edge_b.db < edge/schema.sql
psql "$DATABASE_URL" -f cloud/schema.sql

# 4. env, replacing the hardcoded /workspaces paths
export VI_ROOT="$PWD" DATA="$PWD/data"
export VI_API_TOKEN="$(openssl rand -hex 24)"
export DATABASE_URL="..."

# 5. run and verify
edge/run.sh a & edge/run.sh b &
sleep 1800
python tools/verify.py data/logs/edge_a.jsonl data/logs/edge_b.jsonl --strict
```

## Part E: how to talk about the bug

Do not hide it and do not apologise for it. The framing that scores:

> "After Round 1 I audited my own telemetry rather than my architecture. Every
> detection record showed exactly 10 objects, which no real scene produces. The
> ONNX export declared `dets` before `labels` and my decode assumed the
> opposite, so I was thresholding a sigmoid of box coordinates. The fix was
> four lines. The lesson was that I had built a dashboard, not a monitor: it
> displayed the number faithfully and no invariant ever asked whether the
> number was possible. I now resolve model outputs by tensor shape rather than
> position, and I assert bounds on every SLI in CI."

That paragraph answers PS-4 Q4.2a, PS-1 Q1.3d, and most of the Founder Round
in one go.
