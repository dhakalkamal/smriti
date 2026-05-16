ALTER TABLE message_retrievals
    ADD COLUMN IF NOT EXISTS assistant_message_id UUID;

WITH next_assistant_message AS (
    SELECT DISTINCT ON (message_retrievals.id)
        message_retrievals.id AS message_retrieval_id,
        assistant_messages.id AS assistant_message_id
    FROM message_retrievals
    INNER JOIN messages AS query_messages
        ON query_messages.id = message_retrievals.query_message_id
    INNER JOIN messages AS assistant_messages
        ON assistant_messages.conversation_id = message_retrievals.query_conversation_id
       AND assistant_messages.role = 'assistant'
       AND assistant_messages.position > query_messages.position
    WHERE message_retrievals.assistant_message_id IS NULL
    ORDER BY
        message_retrievals.id,
        assistant_messages.position ASC,
        assistant_messages.id ASC
)
UPDATE message_retrievals
SET assistant_message_id = next_assistant_message.assistant_message_id
FROM next_assistant_message
WHERE message_retrievals.id = next_assistant_message.message_retrieval_id
  AND message_retrievals.assistant_message_id IS NULL;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM message_retrievals
        WHERE assistant_message_id IS NULL
    ) THEN
        RAISE EXCEPTION
            'migration 004 requires existing message_retrievals rows to have a later assistant message in the query conversation';
    END IF;
END;
$$;

ALTER TABLE message_retrievals
    ALTER COLUMN assistant_message_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'message_retrievals_assistant_message_id_fkey'
          AND conrelid = 'message_retrievals'::regclass
    ) THEN
        ALTER TABLE message_retrievals
            ADD CONSTRAINT message_retrievals_assistant_message_id_fkey
            FOREIGN KEY (assistant_message_id)
            REFERENCES messages(id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_message_retrievals_assistant_message_id
    ON message_retrievals (assistant_message_id);
