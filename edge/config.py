"""
Single source of truth for paths and tunables.

WHY THIS EXISTS
    agent.py and docs/extract_metrics.py hardcoded /workspaces/video-intel/...
    which makes the repo Codespaces-only. PS-1 Q1.1 requires "operational
    within 30 minutes of deployment" on arbitrary hardware. A hardcoded
    absolute path is a direct contradiction of that requirement.

Everything resolves from REPO_ROOT, which is derived from this file's
location, with an env override for containerised deployments.
"""
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("VI_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("DATA", REPO_ROOT / "data"))

MODELS = DATA / "models"
LOGS = DATA / "logs"
SEG = DATA / "seg"
SIM = DATA / "sim"

DETECT_MODEL = Path(os.environ.get("VI_DETECT_MODEL", MODELS / "rf-detr-nano-int8.onnx"))
REID_MODEL = Path(os.environ.get("VI_REID_MODEL", MODELS / "reid-osnet-int8.onnx"))

ROSTER = Path(os.environ.get("VI_ROSTER", REPO_ROOT / "edge" / "roster.yaml"))
ZONES = Path(os.environ.get("VI_ZONES", REPO_ROOT / "edge" / "zones.yaml"))
EDGE_SCHEMA = REPO_ROOT / "edge" / "schema.sql"

CLOUD_API = os.environ.get("CLOUD_API", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("VI_API_TOKEN", "dev-token-change-me")
RTSP_BASE = os.environ.get("VI_RTSP_BASE", "rtsp://127.0.0.1:8554")

# ---- inference tunables -------------------------------------------------
# Derived from measurement, not guessed. On a 1 vCPU box a single ONNX
# session runs 619 ms; three concurrent sessions run 1902 ms (3.07x).
# So: ONE shared session, sized to the box, with a bounded queue in front.
CPU_COUNT = os.cpu_count() or 1
INFER_WORKERS = int(os.environ.get("VI_INFER_WORKERS", max(1, CPU_COUNT // 2)))
INFER_THREADS = int(os.environ.get("VI_INFER_THREADS", max(1, CPU_COUNT // INFER_WORKERS)))
INFER_QUEUE_MAX = int(os.environ.get("VI_INFER_QUEUE_MAX", 2 * INFER_WORKERS))

TARGET_FPS = float(os.environ.get("VI_TARGET_FPS", 4))
SEGMENT_SECONDS = int(os.environ.get("VI_SEGMENT_SECONDS", 10))

# ---- detection bands ----------------------------------------------------
# Two thresholds, not one. Anything between LOW and HIGH is "I am not sure"
# and is routed to review instead of being silently accepted or dropped.
# Measured on a_cam01: at 0.70 a clearly visible second person (0.368) is
# discarded, so 0.70 was never a precision setting, it was a recall cliff.
CONF_HIGH = float(os.environ.get("VI_CONF_HIGH", 0.50))
CONF_LOW = float(os.environ.get("VI_CONF_LOW", 0.30))
MAX_DETECTIONS = int(os.environ.get("VI_MAX_DETECTIONS", 100))  # sanity bound, not a filter

for _p in (LOGS, SEG, MODELS):
    _p.mkdir(parents=True, exist_ok=True)
