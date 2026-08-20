#!/usr/bin/env python3
import glob, numpy as np, cv2
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def prep(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

class Reader(CalibrationDataReader):
    def __init__(self, files):
        self.it = iter([{"images": prep(f)} for f in files])
    def get_next(self):
        return next(self.it, None)

from onnxruntime.quantization.shape_inference import quant_pre_process
PREPROC = "data/models/registry/reid-osnet-x025-fp32-preproc.onnx"
quant_pre_process("data/models/registry/reid-osnet-x025-fp32.onnx", PREPROC)
print("pre-processing done")

files = sorted(glob.glob("data/calib/crops/*.jpg"))
print(f"calibrating on {len(files)} crops")
assert len(files) >= 100, f"need >=100 calibration crops, found {len(files)}"

quantize_static(
    PREPROC,
    "data/models/registry/reid-osnet-x025-int8.onnx",
    Reader(files),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True)
print("quantized")
