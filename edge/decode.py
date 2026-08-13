import av, time
from common import telem

def frames(rtsp_url, camera_id, target_fps=4):
    container = av.open(rtsp_url, options={"rtsp_transport": "tcp", "stimeout": "5000000"})
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    src_fps = float(stream.average_rate or 25)
    stride = max(1, int(round(src_fps / target_fps)))
    
    n = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            n += 1
            if n % stride:
                continue
            yield frame.to_ndarray(format="bgr24"), time.time()
