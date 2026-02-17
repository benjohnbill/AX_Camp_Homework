-- Supabase/Postgres auxiliary schema for Antigravity runtime
-- Run this after: sql/1_create_logs_table.sql

-- Optional extension for better LIKE/ILIKE and fuzzy text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS user_stats (
    id INTEGER PRIMARY KEY DEFAULT 1,
    last_login DATE,
    current_streak INTEGER NOT NULL DEFAULT 0,
    longest_streak INTEGER NOT NULL DEFAULT 0,
    debt_count INTEGER NOT NULL DEFAULT 0,
    recovery_points INTEGER NOT NULL DEFAULT 0,
    last_log_at TIMESTAMPTZ,
    CONSTRAINT user_stats_singleton CHECK (id = 1),
    CONSTRAINT user_stats_nonnegative_streak CHECK (current_streak >= 0),
    CONSTRAINT user_stats_nonnegative_longest CHECK (longest_streak >= 0),
    CONSTRAINT user_stats_nonnegative_debt CHECK (debt_count >= 0),
    CONSTRAINT user_stats_nonnegative_recovery CHECK (recovery_points >= 0)
);

INSERT INTO user_stats (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS chat_history (
    id BIGSERIAL PRIMARY KEY,
    "timestamp" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_chat_id BIGINT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB
);

ALTER TABLE chat_history
ADD COLUMN IF NOT EXISTS source_chat_id BIGINT;

CREATE TABLE IF NOT EXISTS connections (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES logs(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES logs(id) ON DELETE CASCADE,
    type TEXT NOT NULL DEFAULT 'manual',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT connections_no_self_loop CHECK (source_id <> target_id),
    CONSTRAINT connections_unique_pair UNIQUE (source_id, target_id)
);

ALTER TABLE logs
ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'::jsonb;

-- Runtime indexes
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_meta_type ON logs (meta_type);
CREATE INDEX IF NOT EXISTS idx_logs_parent_id ON logs (parent_id);
CREATE INDEX IF NOT EXISTS idx_logs_kanban_status ON logs (kanban_status);
CREATE INDEX IF NOT EXISTS idx_logs_tags_gin ON logs USING GIN (tags jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history ("timestamp" DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_role ON chat_history (role);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_history_source_chat_id ON chat_history (source_chat_id);

CREATE INDEX IF NOT EXISTS idx_connections_source ON connections (source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON connections (target_id);
CREATE INDEX IF NOT EXISTS idx_connections_created_at ON connections (created_at DESC);

-- Optional trigram index for Korean/short text fallback search
CREATE INDEX IF NOT EXISTS idx_logs_content_trgm
ON logs USING GIN (content gin_trgm_ops);
