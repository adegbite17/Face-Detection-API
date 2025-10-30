CREATE TABLE IF NOT EXISTS mask_cache (
    id SERIAL PRIMARY KEY,
    phash TEXT UNIQUE,
    mask_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
