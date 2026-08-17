"""
Edge person embedding for cross camera ReID. NEW FILE. Entry point to PS-3.

THE DESIGN DECISION THAT ALREADY GOT MADE FOR YOU
    Because PS-1 put inference at the edge, PS-3's answer is forced and you
    should say so out loud: embeddings are computed at the edge and only the
    VECTOR crosses the WAN. Never the crop, never the frame.

        512 float16 values          = 1024 bytes
        + track and zone metadata   ~  200 bytes
        one 640x360 JPEG crop       ~ 15000 bytes

    That is roughly 12x less egress than shipping crops, and it means raw
    biometric pixels never leave the customer premises, which is most of
    PS-3 Q3.2 answered by an architecture choice rather than a policy
    document.

ONE EMBEDDING PER TRACKLET, NOT PER FRAME
    A tracklet is one person for its whole lifetime, so embedding every
    frame is pure waste. We embed only when the crop quality improves, and
    emit one aggregated descriptor when the tracklet closes. On your logged
    numbers that is roughly a 40x reduction in embedding calls versus per
    frame, and it also produces a BETTER descriptor because averaging over
    several good views is more robust than any single view.

    This is the same cost cascade shape as your motion gate, one level up:
        gate (cheap, rejects 18.6%)
          -> detect (expensive, only on motion)
            -> embed (expensive, only on confirmed tracks, only on best crops)

MODEL
    Drop an OSNet or CLIP-ReID ONNX at data/models/reid-osnet-int8.onnx.
    If absent, the deterministic fallback descriptor below keeps the whole
    pipeline runnable end to end so you can demo the plumbing. Say clearly
    in any writeup that the fallback is a stand in: a colour histogram is
    not ReID, it is a placeholder that lets the interfaces be tested.
"""
import numpy as np
import cv2

from common import telem
from edge import config

DIM = 512
CROP_W, CROP_H = 128, 256
MIN_AREA = 40 * 80          # below this a crop carries no usable identity
MIN_SCORE = 0.5


class Embedder:
    def __init__(self, path=None):
        self.path = str(path or config.REID_MODEL)
        self.session = None
        self.inp = None
        self.dim = DIM
        self.kind = "fallback"
        try:
            import onnxruntime as ort
            import os
            if os.path.exists(self.path):
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                self.session = ort.InferenceSession(
                    self.path, so, providers=["CPUExecutionProvider"])
                self.inp = self.session.get_inputs()[0].name
                self.dim = self.session.get_outputs()[0].shape[-1]
                self.kind = "onnx"
        except Exception as e:
            telem.emit("embedder_init_error", err=repr(e))
        telem.emit("embedder_init", kind=self.kind, dim=self.dim, path=self.path)

    @staticmethod
    def crop_quality(track, frame_shape):
        """Cheap, explainable quality score. Bigger, more confident, more
        centred, less truncated crops win. Feeding a good crop to a mediocre
        model beats feeding a bad crop to a great one."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = track.box
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < MIN_AREA or track.score < MIN_SCORE:
            return 0.0
        aspect = bh / max(1.0, bw)
        aspect_ok = 1.0 if 1.5 <= aspect <= 4.0 else 0.4
        truncated = 0.5 if (x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2) else 1.0
        size = min(1.0, (bw * bh) / (0.08 * w * h))
        return float(track.score * aspect_ok * truncated * (0.4 + 0.6 * size))

    def _preprocess(self, frame, box):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        img = cv2.resize(crop, (CROP_W, CROP_H))
        return img

    def __call__(self, frame, track):
        img = self._preprocess(frame, track.box)
        if img is None:
            return None
        with telem.Timer("embed", camera_id=getattr(track, "cam", None),
                         kind=self.kind):
            if self.session is not None:
                x = img[:, :, ::-1].astype(np.float32) / 255.0
                x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
                    np.array([0.229, 0.224, 0.225], np.float32)
                x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
                v = self.session.run(None, {self.inp: x})[0][0]
            else:
                v = self._fallback(img)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else None

    @staticmethod
    def _fallback(img):
        """Three horizontal stripes of HSV histogram. Deterministic and fast.
        A PLACEHOLDER so the pipeline runs, not a ReID model."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        feats = []
        for band in np.array_split(hsv, 3, axis=0):
            hist = cv2.calcHist([band], [0, 1], None, [24, 12], [0, 180, 0, 256])
            feats.append(cv2.normalize(hist, hist).flatten())
        v = np.concatenate(feats)
        out = np.zeros(DIM, np.float32)
        out[:min(DIM, v.size)] = v[:DIM]
        return out


class TrackletDescriptor:
    """Accumulates the best K embeddings of one tracklet and emits their
    normalized mean when the tracklet closes. Averaging several good views
    is materially more robust than any single frame."""

    def __init__(self, keep=5):
        self.keep = keep
        self.samples = []       # list of (quality, vector)

    def offer(self, quality, vector):
        if vector is None or quality <= 0:
            return False
        self.samples.append((quality, vector))
        self.samples.sort(key=lambda s: -s[0])
        self.samples = self.samples[:self.keep]
        return True

    def finalize(self):
        if not self.samples:
            return None, 0.0
        w = np.array([q for q, _ in self.samples], np.float32)
        m = np.stack([v for _, v in self.samples])
        v = (m * w[:, None]).sum(0) / w.sum()
        n = np.linalg.norm(v)
        if n == 0:
            return None, 0.0
        # Spread of the kept samples is a usable confidence signal: a tracklet
        # whose views disagree is one you should not trust for identity.
        coh = float(np.mean(m @ (v / n)))
        return (v / n).astype(np.float32), coh
