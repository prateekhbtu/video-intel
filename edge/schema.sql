CREATE TABLE IF NOT EXISTS segments (
    camera_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    path TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL,
    bytes INTEGER,
    uploaded INTEGER DEFAULT 0,
    PRIMARY KEY (camera_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_seg_up ON segments(camera_id, uploaded);

CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idem_key TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL,
    attempts INTEGER DEFAULT 0,
    sent INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_out_pending ON outbox(sent, id);

CREATE TABLE IF NOT EXISTS policies (
    camera_id TEXT NOT NULL,
    zone TEXT NOT NULL,
    field TEXT NOT NULL,
    value TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY (camera_id, zone, field)
);
