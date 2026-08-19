"""
Detector. REPLACES the Round 1 edge/detect.py.

THE BUG THIS FILE EXISTS TO KILL
    The ONNX graph declares its outputs as

        output[0] = "dets"    [1, 300, 4]     box coordinates
        output[1] = "labels"  [1, 300, 91]    class logits

    The old code read them positionally, as `logits, boxes = out[0], out[1]`.
    So it took a sigmoid of BOX COORDINATES, thresholded that at 0.70, and
    then read the first four CLASS LOGITS as (cx, cy, w, h). It then truncated
    to `res[:10]`, which is why n_det was exactly 10 in 3,211 of 3,211 records:
    the threshold was inert and the slice was the only thing limiting output.

    The fix is not "swap the two indices". Positional unpacking is the defect.
    Exporters reorder outputs between versions and the failure is silent, so
    outputs are resolved BY SHAPE here: whichever tensor has a trailing
    dimension of 4 is the boxes, the other is the logits. Wrong order becomes
    impossible rather than merely currently-correct.

TWO BANDS, NOT ONE THRESHOLD
    Anything between CONF_LOW and CONF_HIGH is "I am not sure" and is returned
    separately for review instead of being silently accepted or dropped. This
    is the same abstain-band idea as the identity margin test in
    cloud/identity.py, one layer down. Measured on a_cam01: at 0.70 a clearly
    visible second person scoring 0.368 was discarded, so 0.70 was never a
    precision setting, it was a recall cliff.

THE CAP IS A SANITY BOUND, NOT A FILTER
    MAX_DETECTIONS still exists, because an unbounded list from a broken model
    should not OOM the box. But hitting it is now REPORTED as cap_hit rather
    than silently truncating, and tools/verify.py fails the run when the cap
    binds on more than 90% of frames. A cap that is load-bearing is a bug
    wearing a threshold's clothes.
"""
import numpy as np

from common import telem
from edge import config

# COCO-with-background exports carry a no-object slot at column 0. Excluding it
# is what makes argmax mean "best real class" instead of "usually background".
_BG_COLUMN_SIZES = (91, 92)


class Detection:
    """A detection in ORIGINAL FRAME PIXELS. Round 1 passed normalized [0,1]
    boxes into scipy.cdist, so every object was within 1.41 of every other one
    and association was effectively arbitrary. Coordinates carry their space
    in this codebase: everything downstream of here is pixels."""

    __slots__ = ("box", "cls", "score")

    def __init__(self, box, cls, score):
        self.box = box            # [x1, y1, x2, y2] float pixels
        self.cls = int(cls)
        self.score = float(score)

    @property
    def area(self):
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    def as_dict(self):
        return {"box": [round(float(v), 1) for v in self.box],
                "cls": self.cls, "score": round(self.score, 4)}

    def __repr__(self):
        return f"Detection(cls={self.cls}, score={self.score:.3f}, box={self.box})"


def letterbox(frame, size):
    """Resize preserving aspect ratio, pad to square with grey.

    The old path did a straight cv2.resize to a square, which anamorphically
    stretches a 640x360 frame by 1.78x vertically. A person becomes a
    different shape than anything in the training distribution, and the
    detector pays for it in both recall and box quality. Returns the padded
    image plus the (ratio, pad_x, pad_y) needed to map boxes back."""
    import cv2
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


def _split_outputs(outs):
    """Resolve boxes and logits BY SHAPE. This is the whole point of the file."""
    boxes = logits = None
    for o in outs:
        a = np.asarray(o)
        if a.ndim == 3:
            a = a[0]
        elif a.ndim != 2:
            continue
        if a.shape[-1] == 4 and boxes is None:
            boxes = a
        elif a.shape[-1] != 4:
            logits = a
    if boxes is None or logits is None:
        raise ValueError(
            f"cannot resolve detector outputs by shape: got "
            f"{[np.asarray(o).shape for o in outs]}; expected one [..., 4] "
            f"box tensor and one [..., n_classes] logit tensor")
    return boxes, logits


class Detector:
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path=None, size=None, conf_high=None, conf_low=None,
                 classes=None, threads=None, max_detections=None):
        import onnxruntime as ort

        self.path = str(path or config.DETECT_MODEL)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads or config.INFER_THREADS)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

        # The RF-DETR export has a FIXED input size baked into the graph, so
        # `size` is read from the model rather than trusted from the caller.
        # This matters for Q4.3a: input-resolution reduction is a re-export,
        # not a runtime flag, which is the difference between a lever you can
        # pull today and one that needs a release.
        shape = self.s.get_inputs()[0].shape
        graph_size = next((d for d in shape[2:] if isinstance(d, int) and d > 0), None)
        self.size = int(size or graph_size or 384)
        if graph_size and size and int(size) != graph_size:
            telem.emit("detect_size_override", requested=int(size),
                       graph=graph_size, note="graph input is fixed; re-export to change")
            self.size = graph_size

        self.conf_high = float(conf_high if conf_high is not None else config.CONF_HIGH)
        self.conf_low = float(conf_low if conf_low is not None else config.CONF_LOW)
        self.classes = set(classes) if classes else None
        self.max_detections = int(max_detections or config.MAX_DETECTIONS)
        self.ver = self.path.rsplit("/", 1)[-1]

        telem.emit("detector_init", model_ver=self.ver, size=self.size,
                   conf_high=self.conf_high, conf_low=self.conf_low,
                   classes=sorted(self.classes) if self.classes else "all",
                   threads=so.intra_op_num_threads)

    def set_thresholds(self, conf_high=None, conf_low=None):
        """Live retune from a control-plane `threshold` directive. PS-2 Q2.3c:
        the customer turns the sensitivity dial and the edge honours it within
        one poll, with no redeploy and no restart."""
        if conf_high is not None:
            self.conf_high = float(conf_high)
        if conf_low is not None:
            self.conf_low = float(conf_low)
        telem.emit("detect_thresholds_set", model_ver=self.ver,
                   conf_high=self.conf_high, conf_low=self.conf_low)

    def _preprocess(self, frame):
        img, r, px, py = letterbox(frame, self.size)
        x = img[:, :, ::-1].astype(np.float32) / 255.0      # BGR -> RGB
        x = (x - self.MEAN) / self.STD                       # ImageNet norm,
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # which the old
        return x, r, px, py                                   # path skipped

    def __call__(self, frame, camera_id):
        """Returns (confident, unsure). Both are lists of Detection in original
        frame pixels. `unsure` is the abstain band and is what feeds the
        review queue rather than the alert stream."""
        h, w = frame.shape[:2]
        x, ratio, pad_x, pad_y = self._preprocess(frame)

        with telem.Timer("detect", camera_id=camera_id, model_ver=self.ver):
            outs = self.s.run(None, {self.inp: x})

        boxes_n, logits = _split_outputs(outs)

        # DETR-family heads are trained with focal loss, so each class is an
        # independent sigmoid, not a softmax over classes.
        probs = logits if (logits.min() >= 0.0 and logits.max() <= 1.0) \
            else 1.0 / (1.0 + np.exp(-logits))

        start = 1 if probs.shape[-1] in _BG_COLUMN_SIZES else 0
        valid = probs[:, start:]
        cls_idx = valid.argmax(axis=-1) + start
        scores = valid.max(axis=-1)

        # cxcywh normalized against the LETTERBOXED square -> original pixels.
        cx, cy, bw, bh = (boxes_n[:, 0] * self.size, boxes_n[:, 1] * self.size,
                          boxes_n[:, 2] * self.size, boxes_n[:, 3] * self.size)
        x1 = (cx - bw / 2 - pad_x) / ratio
        y1 = (cy - bh / 2 - pad_y) / ratio
        x2 = (cx + bw / 2 - pad_x) / ratio
        y2 = (cy + bh / 2 - pad_y) / ratio
        x1, x2 = np.clip(x1, 0, w), np.clip(x2, 0, w)
        y1, y2 = np.clip(y1, 0, h), np.clip(y2, 0, h)

        keep = scores >= self.conf_low
        if self.classes is not None:
            keep &= np.isin(cls_idx, list(self.classes))
        keep &= (x2 - x1) > 1.0
        keep &= (y2 - y1) > 1.0

        idx = np.nonzero(keep)[0]
        idx = idx[np.argsort(-scores[idx])]          # best first, so a cap
        cap_hit = len(idx) > self.max_detections      # truncates the WEAKEST
        if cap_hit:
            idx = idx[:self.max_detections]

        confident, unsure = [], []
        for i in idx:
            d = Detection([float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                          cls_idx[i], scores[i])
            (confident if d.score >= self.conf_high else unsure).append(d)

        telem.emit("detect_result", camera_id=camera_id, model_ver=self.ver,
                   n_confident=len(confident), n_unsure=len(unsure),
                   n_det=len(confident),          # kept for log continuity
                   cap_hit=bool(cap_hit),
                   max_score=round(float(scores.max()), 4) if scores.size else 0.0,
                   conf_high=self.conf_high, conf_low=self.conf_low)
        return confident, unsure
