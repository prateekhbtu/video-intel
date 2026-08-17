"""
Tracklet tracker. REPLACES the original edge/tracker.py.

WHAT WAS WRONG
    1. It received boxes in normalized [0,1] coordinates (because detect.py
       never scaled them back to pixels), so scipy.cdist distances were all
       in the range 0 to 1.41. Every object was "close" to every other
       object, association was effectively arbitrary, and IDs churned. That
       is the 692-object spike and the 52-object "stabilized output" in the
       deck: both are ID churn, not object counts.
    2. No gating. A track could be matched to a detection on the far side of
       the frame because it happened to be the nearest one.
    3. No confirmation. A track existed the instant one detection appeared,
       so a single frame of noise became a reported object. PS-2 Q2.1
       explicitly contrasts single stage against "detect -> classify ->
       temporal validation". The temporal validation stage did not exist.

WHAT THIS IS NOW
    Greedy association on a combined IoU + centroid cost, hard gated by both.
    Tracks carry hits, age, and a confirmed flag. Only confirmed tracks are
    reported upward. Each track keeps its history so zones.py can compute
    dwell time and tripwire crossings, and so embed.py can pick a good
    quality crop rather than embedding every frame.

    This object is the seconds-scale ancestor of the cross camera identity
    graph in PS-3. Same association problem, different time and space scale:
        tracklet   (this file, one camera, seconds)
        local id   (one site, hours)
        global id  (cloud, persistent)
"""
import time
import numpy as np

from common import telem


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def centroid(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


class Track:
    __slots__ = ("id", "box", "cls", "score", "hits", "age", "missed",
                 "first_ts", "last_ts", "history", "confirmed", "embedded",
                 "best_crop_score", "zone_state")

    def __init__(self, tid, det, ts):
        self.id = tid
        self.box = det.box
        self.cls = det.cls
        self.score = det.score
        self.hits = 1
        self.age = 1
        self.missed = 0
        self.first_ts = ts
        self.last_ts = ts
        self.history = [(ts, centroid(det.box))]
        self.confirmed = False
        self.embedded = False
        self.best_crop_score = det.score
        self.zone_state = {}          # zone name -> dict(entered_ts, inside)

    def update(self, det, ts):
        self.box = det.box
        self.score = det.score
        self.cls = det.cls
        self.hits += 1
        self.missed = 0
        self.last_ts = ts
        self.history.append((ts, centroid(det.box)))
        if len(self.history) > 300:
            self.history = self.history[-300:]

    @property
    def dwell_s(self):
        return self.last_ts - self.first_ts

    @property
    def area(self):
        return max(1.0, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    def as_dict(self):
        return {"track_id": self.id, "cls": int(self.cls),
                "box": [round(float(v), 1) for v in self.box],
                "score": round(float(self.score), 4),
                "hits": self.hits, "dwell_s": round(self.dwell_s, 2)}


class Tracker:
    def __init__(self, camera_id, max_missed=8, min_hits=3,
                 iou_gate=0.15, dist_gate_frac=0.12):
        self.cam = camera_id
        self.max_missed = max_missed
        self.min_hits = min_hits          # temporal validation depth
        self.iou_gate = iou_gate
        self.dist_gate_frac = dist_gate_frac
        self.tracks = {}
        self._next = 1

    def update(self, dets, frame_shape, ts=None):
        ts = ts or time.time()
        h, w = frame_shape[:2]
        dist_gate = self.dist_gate_frac * float(np.hypot(w, h))

        for t in self.tracks.values():
            t.age += 1

        tids = list(self.tracks.keys())
        if tids and dets:
            cost = np.full((len(tids), len(dets)), np.inf)
            for i, tid in enumerate(tids):
                tb = self.tracks[tid].box
                tc = centroid(tb)
                for j, d in enumerate(dets):
                    ov = iou(tb, d.box)
                    dist = float(np.linalg.norm(tc - centroid(d.box)))
                    # HARD GATE. This is what was missing.
                    if ov < self.iou_gate and dist > dist_gate:
                        continue
                    cost[i, j] = (1.0 - ov) + (dist / dist_gate) * 0.5

            used_t, used_d = set(), set()
            flat = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for i, j in flat:
                if not np.isfinite(cost[i, j]):
                    break
                if i in used_t or j in used_d:
                    continue
                self.tracks[tids[i]].update(dets[j], ts)
                used_t.add(i)
                used_d.add(j)

            for j, d in enumerate(dets):
                if j not in used_d:
                    self._spawn(d, ts)
            for i, tid in enumerate(tids):
                if i not in used_t:
                    self.tracks[tid].missed += 1
        elif dets:
            for d in dets:
                self._spawn(d, ts)
        else:
            for t in self.tracks.values():
                t.missed += 1

        ended = []
        for tid, t in list(self.tracks.items()):
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
                telem.emit("track_confirmed", camera_id=self.cam, track_id=tid,
                           hits=t.hits, cls=int(t.cls))
            if t.missed > self.max_missed:
                if t.confirmed:
                    ended.append(t)
                    telem.emit("track_end", camera_id=self.cam, track_id=tid,
                               dwell_s=round(t.dwell_s, 2), hits=t.hits)
                del self.tracks[tid]

        active = [t for t in self.tracks.values() if t.confirmed]
        telem.emit("track_update", camera_id=self.cam,
                   n_det=len(dets), n_active=len(active),
                   n_tentative=len(self.tracks) - len(active),
                   next_id=self._next)
        return active, ended

    def _spawn(self, det, ts):
        self.tracks[self._next] = Track(self._next, det, ts)
        self._next += 1
