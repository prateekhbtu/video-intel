"""
Identity resolution. NEW FILE. The core of PS-3.

THE ONE IDEA THAT MATTERS (PS-3 Q3.3, and the same idea as your res[:10] bug)
    A single threshold on top-1 similarity cannot separate "this is David"
    from "this is someone who looks like David". Both produce a high score.
    What separates them is the MARGIN between the best and second best
    candidate.

        John matched as David 5 times last week.
        sim(query, David) = 0.83, threshold 0.80  -> match, and it is wrong.
        sim(query, John)  = 0.81
        margin = 0.02

    A 0.02 margin means the gallery cannot distinguish these two people at
    all. Accepting the top-1 there is not a matching decision, it is a coin
    flip with a confident tone of voice. This is structurally identical to
    the bug in the Round 1 detector: the system produced an answer where the
    correct output was "I do not know."

    So resolution has THREE outcomes, never two:
        MATCH    top1 >= tau AND margin >= delta
        NEW      top1 <  tau_low
        REVIEW   everything else -> human adjudication queue

    The review band is the product. It is what converts a trust destroying
    false positive into a slightly slower correct answer.

SCALE (PS-3 Q3.1b and Q3.1c), with honest numbers for a 512-d float32 index
    1K       flat numpy, 2 MB,      ~0.1 ms   brute force is simply correct
    10K      flat numpy, 20 MB,     ~1 ms     still no index needed
    100K     flat, 200 MB,          ~10 ms    the knee. pgvector IVFFLAT here
    1M       ANN required, 2 GB raw ~1-5 ms   HNSW, or IVFPQ if RAM bound
    10M+     shard by region, ANN + exact re-rank of the top 100

    A vector DB becomes necessary when brute force latency exceeds your p99
    budget, which for 512-d float32 on one core is around 100K vectors. Not
    at 1K. Saying "we would use a vector DB" at 1K identities is a signal
    that someone has not measured brute force.

STORAGE PER IDENTITY AT 1M (Q3.1e)
    centroid 512 x fp16                        1.0 KB
    up to 5 exemplar vectors                   5.0 KB
    sighting metadata, ~200 rows x 120 B      24.0 KB
    HNSW graph overhead (M=16)                 1.5 KB
    ------------------------------------------------
    ~32 KB per identity  ->  ~32 GB at 1M identities
    At roughly $0.10/GB/month that is ~$3.20/month for the whole gallery,
    which is negligible. The real cost is not storage, it is the RAM to keep
    the ANN index resident, and the compute to re-embed on model upgrade.
"""
import time
import numpy as np

MATCH, NEW, REVIEW = "match", "new", "review"


class ResolutionResult:
    __slots__ = ("decision", "subject_id", "top1", "top2", "margin",
                 "runner_up", "reason")

    def __init__(self, decision, subject_id, top1, top2, runner_up, reason):
        self.decision = decision
        self.subject_id = subject_id
        self.top1 = top1
        self.top2 = top2
        self.margin = top1 - top2
        self.runner_up = runner_up
        self.reason = reason

    def as_dict(self):
        return {"decision": self.decision, "subject_id": self.subject_id,
                "top1": round(self.top1, 4), "top2": round(self.top2, 4),
                "margin": round(self.margin, 4), "runner_up": self.runner_up,
                "reason": self.reason}


class Gallery:
    """Flat cosine index. Correct and fast below ~100K identities. Swap the
    search() body for hnswlib or pgvector above that; the decision logic
    around it does not change, which is the point of keeping them separate."""

    def __init__(self, dim=512, tau=0.75, tau_low=0.55, delta=0.10):
        self.dim = dim
        self.tau = tau            # accept floor
        self.tau_low = tau_low    # below this it is a new identity
        self.delta = delta        # required top1 - top2 margin
        self.ids = []
        self.mat = np.zeros((0, dim), np.float32)
        self.exclusions = set()   # frozenset({a, b}) pairs that must never match
        self.consent_blocked = set()

    # ---- index management ----------------------------------------------
    def upsert(self, subject_id, vec):
        v = np.asarray(vec, np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        if subject_id in self.ids:
            self.mat[self.ids.index(subject_id)] = v
        else:
            self.ids.append(subject_id)
            self.mat = np.vstack([self.mat, v[None]]) if self.mat.size else v[None]

    def delete(self, subject_id):
        if subject_id in self.ids:
            i = self.ids.index(subject_id)
            self.ids.pop(i)
            self.mat = np.delete(self.mat, i, axis=0)

    def apply_delta(self, upserts, deletes, version):
        for sid, vec in (upserts or {}).items():
            self.upsert(sid, vec)
        for sid in (deletes or []):
            self.delete(sid)

    def add_exclusion(self, a, b):
        self.exclusions.add(frozenset((a, b)))

    def block_consent(self, subject_id):
        self.consent_blocked.add(subject_id)
        self.delete(subject_id)

    # ---- resolution -----------------------------------------------------
    def search(self, vec, k=5):
        if not self.ids:
            return []
        v = np.asarray(vec, np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        sims = self.mat @ v
        idx = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def resolve(self, vec, coherence=1.0, known_not=None):
        """known_not: subject ids this query provably is not, from exclusions
        or from a co-occurrence constraint (two tracks visible in the SAME
        frame cannot be the same person, which is a free and very strong
        signal that most ReID systems throw away)."""
        t0 = time.perf_counter()
        cands = self.search(vec, k=5)
        cands = [(sid, s) for sid, s in cands
                 if sid not in (known_not or ()) and sid not in self.consent_blocked]

        if not cands:
            return ResolutionResult(NEW, None, 0.0, 0.0, None, "empty_gallery")

        (best_id, top1) = cands[0]
        top2 = cands[1][1] if len(cands) > 1 else 0.0
        runner = cands[1][0] if len(cands) > 1 else None
        margin = top1 - top2

        # Low tracklet coherence means the views disagreed with each other,
        # so tighten the bar rather than trusting a noisy descriptor.
        tau = self.tau + (1.0 - min(1.0, coherence)) * 0.10

        if runner and frozenset((best_id, runner)) in self.exclusions:
            return ResolutionResult(REVIEW, None, top1, top2, runner,
                                    "excluded_pair_in_contention")
        if top1 < self.tau_low:
            return ResolutionResult(NEW, None, top1, top2, runner, "below_tau_low")
        if top1 >= tau and margin >= self.delta:
            return ResolutionResult(MATCH, best_id, top1, top2, runner, "confident")
        if top1 >= tau and margin < self.delta:
            # The John / David case, caught instead of committed.
            return ResolutionResult(REVIEW, None, top1, top2, runner, "insufficient_margin")
        return ResolutionResult(REVIEW, None, top1, top2, runner, "ambiguous_band")


def sweep_expired(cur, now=None):
    """Retention enforcement in the cloud, mirroring edge/retention.py.
    Deleting the sighting is not enough: any centroid derived from it still
    encodes the subject, so affected identities are queued for recomputation."""
    now = now or time.time()
    cur.execute("SELECT DISTINCT subject_id FROM sightings "
                "WHERE retain_until < %s AND subject_id IS NOT NULL", (now,))
    affected = [r[0] for r in cur.fetchall()]
    cur.execute("DELETE FROM sightings WHERE retain_until < %s", (now,))
    if affected:
        cur.execute("UPDATE identities SET needs_recompute = true "
                    "WHERE subject_id = ANY(%s)", (affected,))
    return len(affected)
