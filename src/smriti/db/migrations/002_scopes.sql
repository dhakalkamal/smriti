CREATE TABLE IF NOT EXISTS scopes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (user_id, name),
    UNIQUE (id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_scopes_user_id
    ON scopes (user_id);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS scope_id UUID;

INSERT INTO scopes (user_id, name, system_prompt, metadata)
SELECT
    users.id,
    'Default',
    '',
    '{"created_by": "migration_002"}'::jsonb
FROM users
ON CONFLICT (user_id, name) DO NOTHING;

UPDATE conversations
SET scope_id = scopes.id
FROM scopes
WHERE conversations.scope_id IS NULL
  AND conversations.user_id = scopes.user_id
  AND scopes.name = 'Default';

ALTER TABLE conversations
    ALTER COLUMN scope_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'conversations_scope_user_fkey'
          AND conrelid = 'conversations'::regclass
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_scope_user_fkey
            FOREIGN KEY (scope_id, user_id)
            REFERENCES scopes(id, user_id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_conversations_scope_id
    ON conversations (scope_id);

CREATE INDEX IF NOT EXISTS idx_conversations_scope_id_updated_at
    ON conversations (scope_id, updated_at DESC);
