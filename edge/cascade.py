"""
The per-camera cost cascade. REPLACES the Round 1 edge/cascade.py.

WHAT THE OLD ONE DID
    Built its own detect.Detector per camera thread (three ONNX sessions on
    one vCPU: 619 ms alone, 1902 ms with three), accepted a `zones_yaml`
    argument and never opened the file, used uuid4() as the idempotency key so
    a replay after a partition created a phantom duplicate, and printed to
    stdout instead of emitting telemetry. Detection was the last stage; there
    was no tracking worth the name, no activity layer, and no embedding.

THE CASCADE, WITH THE MEASURED PASS RATES
    Each stage only pays for what the stage above it let through.

        decode   4 fps                            all frames
          gate   MOG2 motion                      passes 0.814   <- measured
        detect   shared bounded ONNX pool         only on motion
         track   IoU-gated, min_hits confirmation only on detections
          zone   dwell / tripwire / crowd         only on confirmed tracks
         embed   ReID descriptor                  only on quality-improving
                                                  crops of confirmed tracks

    cost_total = SUM_i ( PROD_{j<i} pass_rate_j ) * cost_i

    Worth saying out loud when you present it: the gate rejects 18.6% on this
    footage, not the 21% previously quoted, and on a busy retail floor a
    motion gate earns very little. Naming the condition under which your own
    optimisation fails is worth more than the optimisation.

ONE EMBEDDING PER TRACKLET, NOT PER FRAME
    A tracklet is one person for its whole lifetime. We embed only when the
    crop QUALITY improves on the worst sample we are already keeping, and emit
    a single averaged descriptor when the tracklet closes. Quality is scored
    before the model runs, so a rejected crop costs a few microseconds of
    arithmetic rather than an inference. This is roughly a 40x reduction in
    embedder calls against per-frame, and it produces a BETTER descriptor,
    because averaging several good views beats any single view.

IDEMPOTENCY KEYS ARE DERIVED, NOT RANDOM
    sighting_id = sha1(site : camera : track_id : first_ts). Replaying the
    same tracklet after a partition produces the same key, the cloud's
    ON CONFLICT DO NOTHING absorbs it, and the identity graph does not gain a
    second appearance of a person who only walked past once.
"""
import hashlib
import json
import threading
import time

import numpy as np

from common import telem
from edge import decode, embed as embed_mod, gate, outbox, tracker, zones


def sighting_id(site_id, camera_id, track_id, first_ts):
    raw = f"{site_id}:{camera_id}:{track_id}:{first_ts:.3f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:32]


def activity_id(site_id, camera_id, zone, activity, ts):
    raw = f"{site_id}:{camera_id}:{zone}:{activity}:{ts:.1f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:32]


def pack_embedding(vec):
    """float16 base64 on the wire. float32 JSON is ~4.6 KB per sighting; this
    is 1,368 bytes and, per the Phase 2 evaluation, costs almost nothing in
    Rank-1. Raw biometric pixels never cross the WAN at all: a 640x360 crop is
    ~15 KB, so this is also roughly 11x less egress than shipping the crop."""
    import base64
    return base64.b64encode(np.asarray(vec, np.float16).tobytes()).decode("ascii")


class CameraPipeline:
    def __init__(self, camera_id, site_id, conn, pool, policy=None,
                 embedder=None, reid_enabled=True, min_hits=3):
        self.cam = camera_id
        self.site = site_id
        self.conn = conn
        self.pool = pool
        self.policy = policy
        self.embedder = embedder
        self.reid = reid_enabled and embedder is not None

        self.gate = gate.MotionGate(camera_id)
        self.tracker = tracker.Tracker(camera_id, min_hits=min_hits)
        self.zones = zones.ZoneEngine(camera_id, policy=policy)

        self.descriptors = {}          # track_id -> TrackletDescriptor
        # Callbacks arrive on inference-pool worker threads, so per-camera
        # tracker state needs a lock, and a frame that comes back out of order
        # must be dropped rather than rewinding the tracker.
        self._lock = threading.Lock()
        self._last_ts = 0.0
        self.out_of_order = 0

    # ---- stage 3+: runs on a pool worker ---------------------------------
    def on_result(self, frame, confident, unsure, wall_ts):
        with self._lock:
            if wall_ts <= self._last_ts:
                self.out_of_order += 1
                telem.emit("frame_out_of_order", camera_id=self.cam,
                           ts=wall_ts, last=self._last_ts, n=self.out_of_order)
                return
            self._last_ts = wall_ts

            active, ended = self.tracker.update(confident, frame.shape, ts=wall_ts)

            # The abstain band is a review candidate, never an alert. It is
            # also the highest-value labelling data in the system, because
            # these are precisely the examples the model is least sure about.
            for d in unsure:
                telem.emit("review_candidate", camera_id=self.cam, site_id=self.site,
                           ts=wall_ts, **d.as_dict())

            for ev in self.zones.evaluate(active, frame.shape, ts=wall_ts):
                self._emit_activity(ev)

            if self.reid:
                self._offer_crops(frame, active)
            for t in ended:
                self._close_tracklet(t)

    # ---- stage 5: embed only what improves the descriptor ----------------
    def _offer_crops(self, frame, active):
        for t in active:
            q = self.embedder.crop_quality(t, frame.shape)
            if q <= 0:
                continue
            desc = self.descriptors.setdefault(t.id, embed_mod.TrackletDescriptor())
            # Score first, infer second. A crop that would not be kept costs
            # arithmetic, not an ONNX call.
            if len(desc.samples) >= desc.keep and q <= desc.samples[-1][0]:
                continue
            v = self.embedder(frame, t)
            if v is not None:
                desc.offer(q, v)
                t.embedded = True

    def _close_tracklet(self, t):
        desc = self.descriptors.pop(t.id, None)
        sid = sighting_id(self.site, self.cam, t.id, t.first_ts)
        vec, coh = (desc.finalize() if desc else (None, 0.0))

        self.conn.execute(
            "INSERT OR REPLACE INTO sightings"
            "(sighting_id, camera_id, site_id, track_id, ts, last_ts, dwell_s, "
            " coherence, n_samples, model_ver, created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (sid, self.cam, self.site, t.id, t.first_ts, t.last_ts,
             round(t.dwell_s, 2), round(coh, 4),
             len(desc.samples) if desc else 0,
             getattr(self.embedder, "ver", None) if self.embedder else None,
             time.time()))
        if vec is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings_cache"
                "(sighting_id, vec, dim, created) VALUES(?,?,?,?)",
                (sid, np.asarray(vec, np.float32).tobytes(), int(vec.shape[0]),
                 time.time()))
        self.conn.commit()

        if vec is None:
            # A tracklet with no usable crop is still an event; it just cannot
            # participate in identity. Saying so beats emitting a null vector.
            telem.emit("tracklet_no_descriptor", camera_id=self.cam,
                       track_id=t.id, dwell_s=round(t.dwell_s, 2))
            return

        payload = {
            "sighting_id": sid, "camera_id": self.cam, "site_id": self.site,
            "track_id": t.id,
            "first_ts": round(t.first_ts, 3), "last_ts": round(t.last_ts, 3),
            "dwell_s": round(t.dwell_s, 2), "coherence": round(coh, 4),
            "n_samples": len(desc.samples),
            "emb": pack_embedding(vec), "emb_dim": int(vec.shape[0]),
            "emb_dtype": "float16",
            "model_ver": getattr(self.embedder, "ver", "unknown"),
        }
        outbox.enqueue(self.conn, "sighting", payload, sid)
        telem.emit("sighting", camera_id=self.cam, site_id=self.site,
                   sighting_id=sid, track_id=t.id, dwell_s=round(t.dwell_s, 2),
                   coherence=round(coh, 4), n_samples=len(desc.samples),
                   payload_bytes=len(json.dumps(payload, separators=(",", ":"))))

    def _emit_activity(self, ev):
        aid = activity_id(self.site, self.cam, ev.get("zone"), ev["activity"], ev["ts"])
        self.conn.execute(
            "INSERT INTO activity(camera_id, site_id, zone, activity, track_id, "
            "ts, detail) VALUES(?,?,?,?,?,?,?)",
            (self.cam, self.site, ev.get("zone"), ev["activity"],
             ev.get("track_id"), ev["ts"], json.dumps(ev, separators=(",", ":"))))
        self.conn.commit()
        outbox.enqueue(self.conn, "activity", dict(ev, site_id=self.site), aid)

    # ---- stages 0-2: the decode loop -------------------------------------
    def run(self, rtsp_url):
        telem.emit("cascade_start", camera_id=self.cam, site_id=self.site,
                   rtsp=rtsp_url, reid=self.reid)
        for frame, wall_ts in decode.frames(rtsp_url, self.cam):
            if not self.gate.passes(frame):
                continue
            # submit() is non-blocking and returns False when the queue is
            # full. Dropping under load is correct for a real-time system;
            # silently falling 18x behind, as Round 1 did, is not.
            self.pool.submit(
                frame, self.cam, wall_ts,
                lambda c, u, ts, f=frame: self.on_result(f, c, u, ts))


def run_cascade(camera_id, site_id, rtsp_url, conn, pool, policy=None,
                embedder=None, reid_enabled=True):
    """Entry point used by edge/agent.py's supervisor."""
    CameraPipeline(camera_id, site_id, conn, pool, policy=policy,
                   embedder=embedder, reid_enabled=reid_enabled).run(rtsp_url)
