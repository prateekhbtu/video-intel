#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ffmpeg sqlite3 jq wget curl unzip bc \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev pkg-config

pip install --no-cache-dir \
  av onnxruntime opencv-python-headless requests \
  fastapi uvicorn psycopg2-binary prometheus-client \
  pyyaml pillow numpy

mkdir -p "$DATA"/{datasets,sim,seg,logs,simlogs,models}
echo "setup complete"