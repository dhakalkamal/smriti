CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    position INT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    token_count INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (conversation_id, position)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_position
    ON messages (conversation_id, position);

CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('message', 'summary')),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    range_start INT,
    range_end INT,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    last_accessed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT episodes_kind_shape_check CHECK (
        (kind = 'message'
            AND message_id IS NOT NULL
            AND range_start IS NULL
            AND range_end IS NULL)
        OR
        (kind = 'summary'
            AND message_id IS NULL
            AND range_start IS NOT NULL
            AND range_end IS NOT NULL)
    ),
    CONSTRAINT episodes_summary_range_order_check CHECK (
        kind = 'message' OR range_start <= range_end
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_message_episode_unique
    ON episodes (message_id)
    WHERE kind = 'message';

CREATE INDEX IF NOT EXISTS idx_episodes_conversation_created_at
    ON episodes (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS embedding_models (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO embedding_models (model_id, provider, dimensions, metadata)
VALUES (
    'nomic-embed-text',
    'ollama',
    768,
    '{"default": true, "notes": "default local embedding model"}'::jsonb
)
ON CONFLICT (model_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS embeddings_768 (
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    model_id INT NOT NULL REFERENCES embedding_models(id) ON DELETE RESTRICT,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (episode_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_768_model_id
    ON embeddings_768 (model_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_768_embedding_hnsw
    ON embeddings_768 USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS eval_scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID NOT NULL REFERENCES eval_scenarios(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_scenario_started_at
    ON eval_runs (scenario_id, started_at DESC);

CREATE TABLE IF NOT EXISTS eval_run_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, metric_name)
);
