CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    idem_key VARCHAR(255) UNIQUE,
    camera_id VARCHAR(50),
    site_id VARCHAR(50),
    event_type VARCHAR(50),
    payload JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
