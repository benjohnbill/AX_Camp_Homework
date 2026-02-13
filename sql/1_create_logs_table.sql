-- Supabase/Postgres schema for Antigravity logs
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    meta_type TEXT DEFAULT 'Fragment',
    parent_id TEXT REFERENCES logs(id),
    action_plan TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    embedding VECTOR(1536),
    emotion TEXT,
    dimension TEXT,
    keywords JSONB DEFAULT '[]'::jsonb,
    linked_constitutions JSONB DEFAULT '[]'::jsonb,
    duration INTEGER,
    debt_repaid INTEGER DEFAULT 0,
    kanban_status TEXT,
    quality_score INTEGER DEFAULT 0,
    reward_points INTEGER DEFAULT 0
);

-- pgvector ANN index (ivfflat)
CREATE INDEX IF NOT EXISTS idx_logs_embedding_ivfflat
ON logs USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- FTS index (English parser; see README note for Korean caveat)
CREATE INDEX IF NOT EXISTS idx_logs_fts_en
ON logs USING GIN (to_tsvector('english', COALESCE(content, '')));

-- JSONB keyword index
CREATE INDEX IF NOT EXISTS idx_logs_keywords_gin
ON logs USING GIN (keywords jsonb_path_ops);
