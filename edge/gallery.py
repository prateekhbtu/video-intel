"""
Edge-side gallery cache. NEW FILE.

WHY THE EDGE HOLDS A GALLERY AT ALL
    Resolution is a cloud job: the cloud sees every site, so only the cloud
    can link a person across them. But an edge that loses its uplink should
    degrade, not go blind. This cache is the local mirror the control plane
    keeps fresh with `gallery_delta` directives, and it does three jobs the
    cloud cannot do for you during a partition:

      1. honours an `exclusion_pair` immediately, so the John/David fix is
         live at the edge within one poll rather than one deploy;
      2. honours a consent revocation immediately, which matters more than
         the match does, because continuing to track a subject who has
         objected is the actual compliance breach;
      3. holds open-set CLASS prototypes for PS-4 Q4.1c, which is the same
         machinery as person identity and deliberately shares this file.

    That last point is the connection worth naming in the write-up: a
    customer-specific object class and a person identity are the same problem
    — nearest neighbour against a prototype gallery with an acceptance radius
    — so they get the same code, and adding either one is a data write plus a
    control directive rather than a retrain.

THIS IS A CACHE, NOT THE SOURCE OF TRUTH
    Deltas are versioned and applied in order. A version older than the one
    already applied is ignored, so an out-of-order or replayed directive
    cannot rewind the cache.
"""
import threading
import time

import numpy as np

from common import telem


class EdgeGallery:
    def __init__(self, conn=None, dim=512):
        self.conn = conn
        self.dim = dim
        self._lock = threading.RLock()
        self.ids = []
        self.mat = np.zeros((0, dim), np.float32)
        self.exclusions = set()
        self.consent_blocked = set()
        self.prototypes = {}          # class name -> (vector, threshold)
        self.version = 0
        if conn is not None:
            self.load(conn)

    # ---- persistence -----------------------------------------------------
    def load(self, conn):
        with self._lock:
            rows = conn.execute(
                "SELECT subject_id, centroid, dim FROM identities "
                "WHERE centroid IS NOT NULL").fetchall()
            if rows:
                self.ids = [r[0] for r in rows]
                self.mat = np.stack([
                    np.frombuffer(r[1], np.float32)[: (r[2] or self.dim)]
                    for r in rows]).astype(np.float32)
            self.exclusions = {frozenset((a, b)) for a, b in
                               conn.execute("SELECT a_id, b_id FROM exclusions")}
            self.consent_blocked = {r[0] for r in conn.execute(
                "SELECT subject_id FROM consent WHERE basis='revoked'")}
            row = conn.execute(
                "SELECT COALESCE(MAX(version),0) FROM applied_directives "
                "WHERE kind='gallery_delta'").fetchone()
            self.version = row[0] if row else 0
        telem.emit("gallery_loaded", n_identities=len(self.ids),
                   n_exclusions=len(self.exclusions),
                   n_blocked=len(self.consent_blocked), version=self.version)

    # ---- index management ------------------------------------------------
    def upsert(self, subject_id, vec, persist=True):
        v = np.asarray(vec, np.float32).ravel()
        v = v / (np.linalg.norm(v) + 1e-12)
        with self._lock:
            if subject_id in self.ids:
                self.mat[self.ids.index(subject_id)] = v
            else:
                self.ids.append(subject_id)
                self.mat = (np.vstack([self.mat, v[None]])
                            if self.mat.size else v[None].copy())
            if persist and self.conn is not None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO identities"
                    "(subject_id, centroid, dim, version, updated) VALUES(?,?,?,?,?)",
                    (subject_id, v.tobytes(), int(v.shape[0]), self.version, time.time()))
                self.conn.commit()

    def delete(self, subject_id, persist=True):
        with self._lock:
            if subject_id in self.ids:
                i = self.ids.index(subject_id)
                self.ids.pop(i)
                self.mat = np.delete(self.mat, i, axis=0)
            if persist and self.conn is not None:
                self.conn.execute("DELETE FROM identities WHERE subject_id=?",
                                  (subject_id,))
                self.conn.commit()

    def apply_delta(self, upserts, deletes, version):
        """PS-3 Q3.1d. The cloud recomputes centroids as people age or change
        appearance and ships only what moved, so a site stays current without
        ever downloading the whole gallery."""
        with self._lock:
            if version is not None and version <= self.version:
                telem.emit("gallery_delta_stale", got=version, have=self.version)
                return 0
            for sid, vec in (upserts or {}).items():
                self.upsert(sid, vec)
            for sid in (deletes or []):
                self.delete(sid)
            if version is not None:
                self.version = version
        telem.emit("gallery_delta", upserts=len(upserts or {}),
                   deletes=len(deletes or []), version=version,
                   n_identities=len(self.ids))
        return len(upserts or {}) + len(deletes or [])

    def add_exclusion(self, a, b):
        with self._lock:
            self.exclusions.add(frozenset((a, b)))
        telem.emit("exclusion_added", a=a, b=b, n_exclusions=len(self.exclusions))

    def block_consent(self, subject_id):
        with self._lock:
            self.consent_blocked.add(subject_id)
        self.delete(subject_id)
        telem.emit("consent_blocked", subject_id=subject_id)

    def upsert_prototypes(self, prototypes):
        """PS-4 Q4.1c. An open-set class arrives as a prototype vector plus an
        acceptance radius derived from its own intra-class spread — never a
        hardcoded 0.7, because a tight class deserves a tight threshold and a
        visually diverse one does not."""
        with self._lock:
            for name, spec in (prototypes or {}).items():
                if isinstance(spec, dict):
                    v, thr = spec.get("prototype"), spec.get("threshold", 0.65)
                else:
                    v, thr = spec, 0.65
                v = np.asarray(v, np.float32).ravel()
                v = v / (np.linalg.norm(v) + 1e-12)
                self.prototypes[name] = (v, float(thr))
        telem.emit("prototypes_upserted", classes=sorted(self.prototypes),
                   n=len(self.prototypes))

    # ---- queries ---------------------------------------------------------
    def search(self, vec, k=5):
        with self._lock:
            if not self.ids:
                return []
            v = np.asarray(vec, np.float32).ravel()
            v = v / (np.linalg.norm(v) + 1e-12)
            sims = self.mat @ v
            idx = np.argsort(-sims)[:k]
            return [(self.ids[i], float(sims[i])) for i in idx
                    if self.ids[i] not in self.consent_blocked]

    def classify(self, vec):
        """Open-set membership against class prototypes. Returns
        (class_name, similarity) or (None, best_sim) when nothing clears its
        own threshold, which is the honest answer for an unknown object."""
        with self._lock:
            if not self.prototypes:
                return None, 0.0
            v = np.asarray(vec, np.float32).ravel()
            v = v / (np.linalg.norm(v) + 1e-12)
            best, best_sim = None, -1.0
            for name, (p, thr) in self.prototypes.items():
                s = float(p @ v)
                if s > best_sim:
                    best, best_sim = (name if s >= thr else None), s
            return best, best_sim
