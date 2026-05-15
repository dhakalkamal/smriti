CREATE TABLE IF NOT EXISTS scopes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_scopes_user_id
    ON scopes (user_id);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS scope_id UUID REFERENCES scopes(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_conversations_scope_id
    ON conversations (scope_id);
