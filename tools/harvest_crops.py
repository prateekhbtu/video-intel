#!/usr/bin/env python3
"""Dump person crops from the live simulator for quantization calibration."""
import os, sys, pathlib, cv2, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("VI_ROOT", os.getcwd())
from common import telem
telem.init("data/logs/harvest_crops.jsonl")   # stop flooding stdout
from edge import detect
TARGET = 200   # comfortably clears quantize_static's >=100 requirement
d = detect.Detector(classes={1})
n = 0
for cam in ("a_cam01", "a_cam02", "a_cam03", "a_cam04"):
    cap = cv2.VideoCapture(f"data/sim/{cam}.mp4")
    sampled = 0
    for i in itertools.count():
        ok, f = cap.read()
        if not ok or n >= TARGET: break
        if i % 15: continue
        sampled += 1
        conf, _ = d(f, cam)
        for det in conf:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            crop = f[max(0,y1):y2, max(0,x1):x2]
            if crop.size and (x2-x1) > 25 and (y2-y1) > 55:
                cv2.imwrite(f"data/calib/crops/{cam}_{n:04d}.jpg", crop); n += 1
        if sampled % 20 == 0:
            print(f"  {cam}: sampled {sampled} frames, {n}/{TARGET} crops so far", flush=True)
    print(f"{cam} done: {n}/{TARGET}")
    if n >= TARGET:
        break
print("crops:", n)
