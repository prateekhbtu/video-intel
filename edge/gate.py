import cv2, numpy as np
from common import telem

class MotionGate:
    def __init__(self, camera_id, min_area_frac=0.0015, scale=0.25):
        self.bs = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
        self.cam, self.min_area_frac, self.scale = camera_id, min_area_frac, scale
        self.k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    def passes(self, frame):
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        m = self.bs.apply(small)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k)
        frac = float(np.count_nonzero(m)) / m.size
        ok = frac > self.min_area_frac
        telem.emit("motion_gate", camera_id=self.cam, passed=ok, motion_frac=round(frac, 5))
        return ok
