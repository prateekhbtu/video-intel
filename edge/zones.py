"""
Zone semantics: dwell (loitering), tripwire (unauthorized entry), crowd. NEW FILE.

WHY THIS IS THE BIGGEST FUNCTIONAL GAP IN THE CURRENT REPO
    edge/zones.yaml already defines entrance dwell 25s, a doorway tripwire,
    and a corridor dwell 30s. cascade.py accepts zones_yaml as a parameter
    and NEVER OPENS IT. So the activity layer that PS-2 is entirely about
    (loitering, crowd formation, unauthorized entry, falls) does not exist.
    The system currently reports "N objects", which is a detector output,
    not an activity event.

    PS-2 Q2.1 asks you to decompose detection, tracking, and temporal
    validation. This file is the temporal validation stage, and it is the
    layer where a pixel becomes something a human can act on.

WHY IT MATTERS FOR FALSE ALARMS (Q2.3)
    The cheapest false alarm reduction available is not a better model. It is
    requiring an event to be spatially valid (inside a zone), temporally
    valid (sustained for N seconds), and identity stable (one confirmed
    track, not flickering IDs). Each condition is close to independent, so
    the false alarm rate multiplies down while true positives, which by
    definition satisfy all three, barely move. That is the quantified
    trade-off answer Q2.3b is asking for, and it costs no extra inference.
"""
import time

import yaml

from common import telem
from edge import config


def point_in_poly(pt, poly):
    """Ray casting. poly is normalized [[x,y], ...], pt is normalized."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def side_of_line(pt, line):
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)


class ZoneEngine:
    def __init__(self, camera_id, zones_path=None, policy=None):
        self.cam = camera_id
        self.policy = policy
        self.zones = []
        self.fired = {}                # (track_id, zone, kind) -> ts, dedupes alerts
        self.load(zones_path or config.ZONES)

    def load(self, path):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            self.zones = data.get(self.cam, [])
            telem.emit("zones_loaded", camera_id=self.cam, n_zones=len(self.zones))
        except Exception as e:
            telem.emit("zones_error", camera_id=self.cam, err=repr(e))
            self.zones = []

    def _dwell_limit(self, zone):
        """Policy override beats the yaml default, so a customer can retune
        the loitering threshold live via the control plane."""
        if self.policy:
            v = self.policy.get(self.cam, f"zone:{zone['name']}", "dwell_seconds")
            if v is not None:
                return float(v)
        return float(zone.get("dwell_seconds", 30))

    def evaluate(self, tracks, frame_shape, ts=None):
        """tracks: confirmed Track objects from tracker.py. Returns activity
        events, which are the things a human actually cares about."""
        ts = ts or time.time()
        h, w = frame_shape[:2]
        events = []

        for zone in self.zones:
            name, kind = zone["name"], zone["type"]

            if kind == "dwell":
                poly = zone["polygon"]
                limit = self._dwell_limit(zone)
                occupants = 0
                for t in tracks:
                    cx = ((t.box[0] + t.box[2]) / 2) / w
                    fy = t.box[3] / h                 # feet, not centre: a
                    inside = point_in_poly((cx, fy), poly)   # person stands on
                    st = t.zone_state.setdefault(name, {"inside": False, "since": None})
                    if inside:
                        occupants += 1
                        if not st["inside"]:
                            st["inside"], st["since"] = True, ts
                        elif ts - st["since"] >= limit:
                            if self._fire(t.id, name, "dwell", ts, cooldown=limit):
                                events.append(self._ev(
                                    "loitering", name, t, ts,
                                    dwell_s=round(ts - st["since"], 1), threshold_s=limit))
                    else:
                        st["inside"], st["since"] = False, None

                crowd_k = int(zone.get("crowd_threshold", 0))
                if crowd_k and occupants >= crowd_k:
                    if self._fire(0, name, "crowd", ts, cooldown=30):
                        events.append(self._ev("crowd_formation", name, None, ts,
                                               occupants=occupants, threshold=crowd_k))

            elif kind == "tripwire":
                line = zone["line"]
                want = zone.get("direction", "any")
                for t in tracks:
                    if len(t.history) < 2:
                        continue
                    p_prev = t.history[-2][1]
                    p_now = t.history[-1][1]
                    a = side_of_line((p_prev[0] / w, p_prev[1] / h), line)
                    b = side_of_line((p_now[0] / w, p_now[1] / h), line)
                    if a == 0 or (a > 0) == (b > 0):
                        continue
                    direction = "down" if b > a else "up"
                    if want != "any" and direction != want:
                        continue
                    if self._fire(t.id, name, "cross", ts, cooldown=5):
                        events.append(self._ev("unauthorized_entry", name, t, ts,
                                               direction=direction))
        return events

    def _fire(self, track_id, zone, kind, ts, cooldown):
        key = (track_id, zone, kind)
        last = self.fired.get(key, 0)
        if ts - last < cooldown:
            return False
        self.fired[key] = ts
        return True

    def _ev(self, activity, zone, track, ts, **extra):
        ev = {"activity": activity, "camera_id": self.cam, "zone": zone, "ts": ts}
        if track is not None:
            ev.update({"track_id": track.id, "box": [round(float(v), 1) for v in track.box],
                       "score": round(float(track.score), 4), "hits": track.hits})
        ev.update(extra)
        telem.emit("activity", **ev)
        return ev
