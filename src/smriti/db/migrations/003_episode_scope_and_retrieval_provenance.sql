ALTER TABLE episodes
    ADD COLUMN IF NOT EXISTS scope_id UUID;

UPDATE episodes
SET scope_id = conversations.scope_id
FROM conversations
WHERE episodes.scope_id IS NULL
  AND episodes.conversation_id = conversations.id;

ALTER TABLE episodes
    ALTER COLUMN scope_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_episodes_scope_id
    ON episodes (scope_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'conversations_id_scope_id_key'
          AND conrelid = 'conversations'::regclass
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_id_scope_id_key UNIQUE (id, scope_id);
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'messages_id_conversation_id_key'
          AND conrelid = 'messages'::regclass
    ) THEN
        ALTER TABLE messages
            ADD CONSTRAINT messages_id_conversation_id_key UNIQUE (id, conversation_id);
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'episodes_id_scope_id_key'
          AND conrelid = 'episodes'::regclass
    ) THEN
        ALTER TABLE episodes
            ADD CONSTRAINT episodes_id_scope_id_key UNIQUE (id, scope_id);
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'episodes_scope_id_fkey'
          AND conrelid = 'episodes'::regclass
    ) THEN
        ALTER TABLE episodes
            ADD CONSTRAINT episodes_scope_id_fkey
            FOREIGN KEY (scope_id)
            REFERENCES scopes(id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'episodes_conversation_scope_fkey'
          AND conrelid = 'episodes'::regclass
    ) THEN
        ALTER TABLE episodes
            ADD CONSTRAINT episodes_conversation_scope_fkey
            FOREIGN KEY (conversation_id, scope_id)
            REFERENCES conversations(id, scope_id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'episodes_message_conversation_fkey'
          AND conrelid = 'episodes'::regclass
    ) THEN
        ALTER TABLE episodes
            ADD CONSTRAINT episodes_message_conversation_fkey
            FOREIGN KEY (message_id, conversation_id)
            REFERENCES messages(id, conversation_id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS message_retrievals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_message_id UUID NOT NULL,
    query_conversation_id UUID NOT NULL,
    scope_id UUID NOT NULL,
    episode_id UUID NOT NULL,
    embedding_model_id INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE RESTRICT,
    result_rank INTEGER NOT NULL CHECK (result_rank > 0),
    similarity DOUBLE PRECISION NOT NULL CHECK (similarity >= -1.0 AND similarity <= 1.0),
    recency_score DOUBLE PRECISION NOT NULL CHECK (recency_score >= 0.0 AND recency_score <= 1.0),
    access_score DOUBLE PRECISION NOT NULL CHECK (access_score >= 0.0 AND access_score <= 1.0),
    importance_score DOUBLE PRECISION NOT NULL CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    frequency_score DOUBLE PRECISION NOT NULL CHECK (frequency_score >= 0.0 AND frequency_score <= 1.0),
    score DOUBLE PRECISION NOT NULL,
    scoring_version TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    FOREIGN KEY (query_message_id, query_conversation_id)
        REFERENCES messages(id, conversation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (query_conversation_id, scope_id)
        REFERENCES conversations(id, scope_id)
        ON DELETE CASCADE,
    FOREIGN KEY (episode_id, scope_id)
        REFERENCES episodes(id, scope_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_message_retrievals_query_message_id
    ON message_retrievals (query_message_id, retrieved_at DESC);

CREATE INDEX IF NOT EXISTS idx_message_retrievals_episode_id
    ON message_retrievals (episode_id);

CREATE INDEX IF NOT EXISTS idx_message_retrievals_scope_id_retrieved_at
    ON message_retrievals (scope_id, retrieved_at DESC);
