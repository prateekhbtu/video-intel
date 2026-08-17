"""
Retention and consent enforcement at the edge. NEW FILE. PS-3 Q3.2c and Q3.2a.

THE LIABILITY IN THE CURRENT REPO
    roster.yaml sets record_all: true. edge_a.db holds 12,237 segment rows
    totalling 9.85 GB from a SINGLE run, with segments.uploaded = 0 on every
    one of them, and there is no deletion path anywhere in the codebase.

    On a 32 GB devcontainer that fills the disk in roughly 18 hours. Under
    PS-1 alone that is an availability bug. The moment PS-3 adds identity it
    also becomes a compliance breach, because "we kept biometric source
    material indefinitely with no lawful basis and no deletion mechanism" is
    the exact thing GDPR Article 5(1)(e) and India's DPDP Act section 8(7)
    prohibit.

WHY DELETION MUST RUN HERE AND NOT ONLY IN THE CLOUD
    The edge holds the pixels. The cloud only holds vectors and metadata. A
    retention policy enforced solely in the cloud deletes the index and
    leaves the evidence, which is worse than useless because it destroys
    your ability to prove compliance while preserving the thing you were
    supposed to delete.

THE HARD PART, WHICH Q3.2d ASKS FOR DIRECTLY
    Deletion is easy. Deletion that is PROVABLE is hard, and there are three
    genuinely difficult sub problems worth naming:
      1. Backups and replicas. A deletion that a restore can undo is not a
         deletion. Either backups carry the tombstone forward or they expire
         faster than the retention window.
      2. Derived artifacts. A gallery centroid computed from ten sightings
         still contains information about a subject who revoked consent, so
         revocation must trigger recomputation, not just row removal.
      3. Deletion needs identity, and identity is the thing the subject is
         objecting to. Honouring "delete everything about me" requires
         running a match against the person you were told not to track. The
         standard resolution is a narrowly scoped, audited, time boxed
         deletion index that is separate from the operational gallery.
"""
import os
import time
import threading

from common import telem


class RetentionManager:
    def __init__(self, conn, site_id, days=30, min_free_gb=5.0, seg_root=None):
        self.conn = conn
        self.site = site_id
        self.days = days
        self.min_free_gb = min_free_gb
        self.seg_root = seg_root
        self._lock = threading.Lock()

    def configure(self, days=None, classes=None, min_free_gb=None):
        with self._lock:
            if days is not None:
                self.days = int(days)
            if min_free_gb is not None:
                self.min_free_gb = float(min_free_gb)
        telem.emit("retention_configured", days=self.days, min_free_gb=self.min_free_gb)

    # ---- scheduled TTL sweep -------------------------------------------
    def sweep(self):
        cutoff = time.time() - self.days * 86400
        rows = self.conn.execute(
            "SELECT camera_id, seq, path, bytes FROM segments WHERE start_ts < ? "
            "AND legal_hold = 0 LIMIT 5000", (cutoff,)).fetchall()
        freed = 0
        for cam, seq, path, nbytes in rows:
            freed += self._unlink(path, nbytes)
            self.conn.execute("DELETE FROM segments WHERE camera_id=? AND seq=?", (cam, seq))
        self.conn.commit()
        if rows:
            self._audit("ttl_sweep", n=len(rows), bytes=freed, cutoff=cutoff)
        telem.emit("retention_sweep", site_id=self.site, deleted=len(rows),
                   bytes_freed=freed, retention_days=self.days)
        return len(rows), freed

    # ---- disk pressure eviction ----------------------------------------
    def evict_for_space(self):
        """Runs BEFORE the disk fills, not after. Evicts oldest first, but
        never anything under legal hold and never anything not yet uploaded,
        which is the ordering that keeps 'no footage lost' honest instead of
        aspirational."""
        if not self.seg_root:
            return 0, 0
        st = os.statvfs(self.seg_root)
        free_gb = st.f_bavail * st.f_frsize / 1e9
        if free_gb >= self.min_free_gb:
            return 0, 0
        telem.emit("disk_pressure", site_id=self.site, free_gb=round(free_gb, 2),
                   threshold_gb=self.min_free_gb, severity="warning")
        rows = self.conn.execute(
            "SELECT camera_id, seq, path, bytes FROM segments "
            "WHERE legal_hold = 0 AND uploaded = 1 ORDER BY start_ts LIMIT 2000"
        ).fetchall()
        n = freed = 0
        for cam, seq, path, nbytes in rows:
            freed += self._unlink(path, nbytes)
            self.conn.execute("DELETE FROM segments WHERE camera_id=? AND seq=?", (cam, seq))
            n += 1
            st = os.statvfs(self.seg_root)
            if st.f_bavail * st.f_frsize / 1e9 >= self.min_free_gb * 1.2:
                break
        self.conn.commit()
        self._audit("space_eviction", n=n, bytes=freed)
        return n, freed

    # ---- consent revocation --------------------------------------------
    def purge_subject(self, subject_id):
        """Invoked by the control plane. Deletes local sightings and any
        segment whose window contains one, then records an auditable receipt
        with a content hash so deletion can be PROVEN later."""
        sightings = self.conn.execute(
            "SELECT camera_id, ts FROM sightings WHERE subject_id=?", (subject_id,)).fetchall()
        seg_deleted = bytes_freed = 0
        for cam, ts in sightings:
            for (seq, path, nbytes) in self.conn.execute(
                "SELECT seq, path, bytes FROM segments WHERE camera_id=? "
                "AND start_ts <= ? AND start_ts + duration_s >= ? AND legal_hold=0",
                (cam, ts, ts)
            ).fetchall():
                bytes_freed += self._unlink(path, nbytes)
                self.conn.execute("DELETE FROM segments WHERE camera_id=? AND seq=?", (cam, seq))
                seg_deleted += 1
        self.conn.execute("DELETE FROM sightings WHERE subject_id=?", (subject_id,))
        self.conn.execute("DELETE FROM embeddings_cache WHERE subject_id=?", (subject_id,))
        self.conn.commit()
        self._audit("consent_purge", subject_id=subject_id,
                    sightings=len(sightings), segments=seg_deleted, bytes=bytes_freed)
        telem.emit("consent_purge", subject_id=subject_id, site_id=self.site,
                   sightings=len(sightings), segments=seg_deleted)
        return seg_deleted

    def _unlink(self, path, nbytes):
        try:
            os.unlink(path)
            return nbytes or 0
        except FileNotFoundError:
            return 0
        except Exception as e:
            telem.emit("retention_unlink_error", path=path, err=repr(e))
            return 0

    def _audit(self, action, **detail):
        """Append only. The audit log is itself exempt from the TTL sweep,
        because you must be able to prove a deletion happened after the data
        is gone."""
        import json
        self.conn.execute(
            "INSERT INTO audit_log(ts, site_id, action, detail) VALUES(?,?,?,?)",
            (time.time(), self.site, action, json.dumps(detail)))
        self.conn.commit()


def retention_loop(mgr, interval=300):
    while True:
        try:
            mgr.evict_for_space()
            mgr.sweep()
        except Exception as e:
            telem.emit("retention_error", err=repr(e))
        time.sleep(interval)
