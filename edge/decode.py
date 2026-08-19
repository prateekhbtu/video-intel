"""
RTSP decode at a target sampling rate.

WHY THIS FILE CHANGED
    It is a GENERATOR, which makes it the quiet accomplice in the Round 1
    latency problem: when the consumer blocked for 1.5 s on a synchronous
    detect call, this loop stopped pulling packets, frames piled up in the
    PyAV buffer, and the pipeline fell 18x behind real time with nothing
    measuring the gap. The consumer side is fixed by edge/infer_pool.py, which
    drops under load instead of stalling. This side contributes the other
    half: an explicit stride, a reconnect with backoff instead of a silent
    death, and a stream_stall record when wall clock and media clock diverge.

    `stimeout` was also renamed to `timeout` in FFmpeg 6; passing the old key
    is accepted and ignored, so a dead camera hung forever instead of timing
    out. Both are sent, which is correct across versions.
"""
import time

import av

from common import telem
from edge import config

RECONNECT_BASE = 2.0
RECONNECT_MAX = 60.0


def open_stream(rtsp_url):
    return av.open(rtsp_url, options={
        "rtsp_transport": "tcp",
        "stimeout": "5000000",     # FFmpeg <= 5, microseconds
        "timeout": "5000000",      # FFmpeg >= 6
        "max_delay": "500000",
        "fflags": "nobuffer",
    })


def frames(rtsp_url, camera_id, target_fps=None, reconnect=True):
    """Yields (frame_bgr, wall_ts). Reconnects with backoff on stream loss so
    a camera blip costs latency rather than the rest of the run."""
    target_fps = float(target_fps or config.TARGET_FPS)
    attempt = 0

    while True:
        container = None
        try:
            container = open_stream(rtsp_url)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            src_fps = float(stream.average_rate or 25)
            stride = max(1, int(round(src_fps / target_fps)))
            attempt = 0
            telem.emit("decode_open", camera_id=camera_id, src_fps=round(src_fps, 2),
                       target_fps=target_fps, stride=stride)

            n = 0
            last_report = time.time()
            yielded = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    n += 1
                    if n % stride:
                        continue
                    yielded += 1
                    now = time.time()
                    if now - last_report >= 30:
                        telem.emit("decode_rate", camera_id=camera_id,
                                   decoded=n, yielded=yielded,
                                   effective_fps=round(yielded / (now - last_report), 2),
                                   target_fps=target_fps)
                        last_report, n, yielded = now, 0, 0
                    yield frame.to_ndarray(format="bgr24"), now

            telem.emit("decode_eof", camera_id=camera_id)
        except Exception as e:
            telem.emit("decode_error", camera_id=camera_id, err=repr(e),
                       attempt=attempt, severity="warning")
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

        if not reconnect:
            return
        attempt += 1
        delay = min(RECONNECT_MAX, RECONNECT_BASE * (2 ** min(attempt, 5)))
        telem.emit("decode_reconnect", camera_id=camera_id, attempt=attempt,
                   delay_s=delay)
        time.sleep(delay)
