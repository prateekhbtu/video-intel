"""
Versioned per tenant / per camera policy cache. NEW FILE.

WHY THIS IS ALMOST FREE FOR YOU
    edge/schema.sql already declares:
        policies(camera_id, zone, field, value, version)
    and it has ZERO rows in edge_a.db. You built the table and never used it.

    Widened slightly, that table IS the per customer feature flag system that
    PS-4 Q4.2b asks for. "Roll back for specific customers while rolling
    forward for others" becomes a row write with a version bump, propagated
    by inbox.py. No redeploy, no restart, no per customer branch.

RESOLUTION ORDER (most specific wins)
    (tenant, camera, key) > (tenant, "*", key) > ("*", "*", key) > code default
"""
import json
import threading
import time

from common import telem


class PolicyStore:
    def __init__(self, conn, tenant_id="default"):
        self.tenant = tenant_id
        self._lock = threading.RLock()
        self._cache = {}
        self.reload(conn)

    def reload(self, conn):
        with self._lock:
            self._cache = {}
            for tenant, cam, scope, field, value, ver in conn.execute(
                "SELECT tenant_id, camera_id, zone, field, value, version FROM policies"
            ):
                self._cache[(tenant, cam, scope, field)] = (value, ver)
        telem.emit("policy_reload", n=len(self._cache), tenant=self.tenant)

    def set(self, conn, camera_id, scope, field, value, version):
        v = json.dumps(value) if not isinstance(value, str) else value
        with self._lock:
            key = (self.tenant, camera_id, scope, field)
            old = self._cache.get(key)
            if old and old[1] >= version:
                return False                      # stale directive, ignore
            conn.execute(
                "INSERT INTO policies(tenant_id,camera_id,zone,field,value,version,updated) "
                "VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(tenant_id,camera_id,zone,field) DO UPDATE SET "
                "value=excluded.value, version=excluded.version, updated=excluded.updated "
                "WHERE excluded.version > policies.version",
                (self.tenant, camera_id, scope, field, v, version, time.time()))
            conn.commit()
            self._cache[key] = (v, version)
        telem.emit("policy_set", camera_id=camera_id, scope=scope,
                   field=field, version=version)
        return True

    def get(self, camera_id, scope, field, default=None, cast=None):
        with self._lock:
            for t, c in ((self.tenant, camera_id), (self.tenant, "*"), ("*", "*")):
                hit = self._cache.get((t, c, scope, field))
                if hit:
                    v = hit[0]
                    if cast:
                        try:
                            return cast(json.loads(v) if v.startswith(("[", "{", '"')) else v)
                        except Exception:
                            return default
                    return v
        return default

    def version(self, scope, field):
        with self._lock:
            for t, c in ((self.tenant, "*"), ("*", "*")):
                hit = self._cache.get((t, c, scope, field))
                if hit:
                    return hit[1]
        return 0
