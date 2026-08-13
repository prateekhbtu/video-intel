import onnxruntime as ort, numpy as np
from common import telem

class Detector:
    def __init__(self, path, size=640, conf=0.45):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.size, self.conf = size, conf
        self.ver = path.split("/")[-1]

    def __call__(self, frame, camera_id):
        import cv2
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (self.size, self.size))[:, :, ::-1]
        x = (img.astype(np.float32)/255.0).transpose(2,0,1)[None]
        
        with telem.Timer("detect", camera_id=camera_id, model_ver=self.ver):
            out = self.s.run(None, {self.inp: x})
            logits, boxes = out[0][0], out[1][0]
        
        probs = logits if logits.max() <= 1.0 else 1 / (1 + np.exp(-logits))
        scores = probs.max(axis=-1)
        keep = scores > self.conf
        
        res = []
        for b in boxes[keep]:
            cx, cy, bw, bh = b
            res.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
        
        telem.emit("detect_result", camera_id=camera_id, n_det=len(res))
        return res
