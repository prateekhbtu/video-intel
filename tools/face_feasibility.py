#!/usr/bin/env python3
"""
Face recognition versus ReID, decided by measurement. Build plan section 2.4,
answering PS-3 Q3.1a.

THE ARGUMENT THIS REPLACES
    "Face recognition is more accurate than ReID" is true on a benchmark and
    frequently irrelevant on a camera. Reliable 1:N face recognition needs
    roughly 60 px between the eye centres. A camera mounted for scene
    coverage, at the heights and focal lengths surveillance actually uses,
    usually delivers far less than that on a walking subject.

    So the question is not "which model scores higher", it is "what fraction
    of the people my cameras detect carry a face big enough to recognise at
    all". That is a property of the deployment, not of the model, and it is
    measurable from the same footage the rest of the pipeline runs on.

WHAT IS MEASURED
    persons_detected      person boxes from the SAME detector the edge uses
    faces_detected        faces insightface finds inside those boxes
    usable                faces whose interocular distance >= THRESHOLD_PX
    usable_rate_of_persons  usable / persons  <- this is the answer

    A low usable rate is not a failure of the experiment. It is the finding:
    it says face recognition cannot be the primary identity signal on this
    deployment, and body ReID has to carry it, which is exactly why the
    architecture puts an embedding gallery at the centre instead of a face
    matcher.

HONESTY RULE
    If insightface is unavailable, this writes "not measured" plus the reason
    into the JSON rather than estimating. An absent number is a fact; an
    invented one is a defect.
"""
import json
import os
import pathlib
import sys
import time

# tools/ scripts are run by path (python3 tools/face_feasibility.py), not as a
# module, so the repo root is not on sys.path and `from edge import ...` fails.
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("VI_ROOT", str(REPO))

import numpy as np
import cv2

from common import telem

OUT = REPO / "results" / "reid" / "face_feasibility.json"
THRESHOLD_PX = 60.0          # industry guidance for reliable 1:N face matching
CAMERAS = ("a_cam01", "a_cam02", "a_cam03", "a_cam04")
MAX_FRAMES = 900             # per camera
STRIDE = 25                  # sample every Nth frame; faces do not change fast


def write(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def main():
    telem.init(str(REPO / "data" / "logs" / "face_feasibility.jsonl"))

    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        write({"status": "not measured",
               "reason": f"insightface unavailable: {e!r}",
               "threshold_px": THRESHOLD_PX})
        return 0

    from edge import detect

    try:
        fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        write({"status": "not measured",
               "reason": f"insightface model pack failed to load: {e!r}",
               "threshold_px": THRESHOLD_PX})
        return 0

    d = detect.Detector(classes={1})          # person only, same detector as the edge

    persons = faces = usable = 0
    eyedists, face_frac = [], []
    per_cam = {}
    t0 = time.time()

    for cam in CAMERAS:
        clip = REPO / "data" / "sim" / f"{cam}.mp4"
        if not clip.exists():
            per_cam[cam] = {"error": "clip absent"}
            continue
        cap = cv2.VideoCapture(str(clip))
        c_persons = c_faces = c_usable = 0
        i = 0
        while i < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % STRIDE:
                continue
            conf, _ = d(frame, cam)
            c_persons += len(conf)
            for det in conf:
                x1, y1, x2, y2 = [int(v) for v in det.box]
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                box_h = max(1, y2 - y1)
                for fc in fa.get(crop):
                    c_faces += 1
                    kp = fc.kps                      # 5 landmarks, eyes first two
                    ed = float(np.linalg.norm(kp[0] - kp[1]))
                    eyedists.append(ed)
                    fb = fc.bbox
                    face_frac.append(float((fb[3] - fb[1]) / box_h))
                    if ed >= THRESHOLD_PX:
                        c_usable += 1
            cap_ok = True
        cap.release()
        persons += c_persons
        faces += c_faces
        usable += c_usable
        per_cam[cam] = {"frames_sampled": i // STRIDE, "persons": c_persons,
                        "faces": c_faces, "usable": c_usable}
        telem.emit("face_feasibility_camera", camera_id=cam, **per_cam[cam])
        print(f"  {cam}: {i//STRIDE:3d} frames  {c_persons:4d} persons  "
              f"{c_faces:3d} faces  {c_usable:3d} usable")

    res = {
        "status": "measured",
        "persons_detected": persons,
        "faces_detected": faces,
        "face_detection_rate": round(faces / max(1, persons), 4),
        "usable_for_recognition": usable,
        "usable_rate_of_persons": round(usable / max(1, persons), 4),
        "usable_rate_of_faces": round(usable / max(1, faces), 4),
        "interocular_px_median": round(float(np.median(eyedists)), 1) if eyedists else 0,
        "interocular_px_p90": round(float(np.percentile(eyedists, 90)), 1) if eyedists else 0,
        "interocular_px_max": round(float(np.max(eyedists)), 1) if eyedists else 0,
        "face_height_frac_of_body_median":
            round(float(np.median(face_frac)), 4) if face_frac else 0,
        "threshold_px": THRESHOLD_PX,
        "per_camera": per_cam,
        "elapsed_s": round(time.time() - t0, 1),
        "interpretation": (
            "usable_rate_of_persons is the fraction of detected people who carry "
            "a face large enough for reliable 1:N recognition on THIS deployment. "
            "It bounds face recognition as a primary identity signal regardless "
            "of how good the face model is."),
    }
    write(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
