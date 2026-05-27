# Stage 10b: Read-only Retrieval Inspection

## Purpose
Observability before optimization. Expose existing
message_retrievals provenance via the UI so users can inspect
which memories were retrieved for each assistant response.
This stage adds no new behavior to the retrieval pipeline; it
only surfaces what is already recorded.

Prior stage contracts (9a, 9b, 10a) intentionally deferred
provenance visualization. Stage 10b delivers that deferred
capability; this is not a contradiction but the staged
delivery working as designed.

## Non-goals (locked)
- No feedback capture, no retrieval_feedback table
- No hybrid retrieval, no scoring algorithm changes
- No prompt changes, no memory or episode editing
- No eval harness
- No retrieval re-execution from inspection
- No side effects on episodes (access_count, last_accessed_at)
- No new dependencies (backend or frontend)
- No schema changes, no new migration
- No inspection of the in-flight streaming assistant draft

## Architectural invariants (preserved)
- Referential integrity: every FK on message_retrievals
  either cascades on source deletion (assistant_message_id,
  query_message_id, query_conversation_id, scope_id,
  episode_id) or restricts deletion of its target
  (embedding_model_id ON DELETE RESTRICT). Orphan
  provenance is structurally impossible; INNER JOINs
  throughout the read SQL are safe.
- Scope is the privacy boundary. Inspection surfaces source
  episode content from within the same scope. The current
  retrieval already enforces this and the inspection JOIN
  preserves it via episodes.scope_id.
- Routes remain SQL-free; all DB access goes through
  MemoryService.
- Routes do not raise HTTPException; they raise service
  exceptions mapped centrally in api/errors.py.
- retrieve_scoped_episodes is the only code path that mutates
  episodes.access_count and last_accessed_at. The new
  inspection method must NOT call it.

## Database
- No DDL changes.
- Existing index idx_message_retrievals_assistant_message_id
  is sufficient for the endpoint's primary lookup.

## Backend

### Endpoint
GET /conversations/{conversation_id}/messages/{message_id}/retrievals

Nested path follows the existing convention in messages.py.
No /api prefix.

### Behavior
1. Authorization (layered; follows existing house convention
   for nested message endpoints — see
   _conversation_scope_for_user and the
   ConversationAccessDeniedError mapping in api/errors.py)

   a. Conversation-level check (first): reuse the existing
      ownership check.
      - Conversation does not exist →
        ConversationNotFoundError → 404.
      - Conversation exists but belongs to another user →
        ConversationAccessDeniedError → 403.

   b. Message-level check (only after conversation
      ownership is established): verify the message exists
      in that conversation with role='assistant'.
      Otherwise raise AssistantMessageNotFoundError (new
      exception, mapped to 404 in api/errors.py). At this
      layer, not-found / wrong-role / mismatched-ID all
      collapse to 404, which leaks nothing because the
      user already owns the parent conversation.

   Implemented as two sequential checks before the
   retrieval SELECT, matching the two-query authorization
   pattern used elsewhere in MemoryService.

2. Fetch (single SELECT, INNER JOINs only)
     message_retrievals mr
       JOIN episodes e        ON e.id = mr.episode_id
                              AND e.scope_id = mr.scope_id
       JOIN conversations sc  ON sc.id = e.conversation_id
       JOIN scopes s          ON s.id = sc.scope_id
       JOIN messages qm       ON qm.id = mr.query_message_id
     WHERE mr.assistant_message_id = $1
     ORDER BY mr.result_rank ASC;

3. Read-only
   - No UPDATE statements anywhere in the handler.
   - Does NOT call retrieve_scoped_episodes,
     _update_retrieved_episode_access_metadata, or any
     scoring code.
   - Does NOT touch episodes.access_count or
     last_accessed_at.

4. 200 with empty retrievals[] when the message is a valid
   assistant message owned by the user but has no recorded
   provenance.

### Service layer
- New method on MemoryService:
    async def list_message_retrievals(
        self,
        user_id: UUID,
        conversation_id: UUID,
        assistant_message_id: UUID,
    ) -> list[RetrievalRecord]
  Placed near list_messages and the provenance writers
  (_insert_message_retrievals).
- New dataclasses in memory/models.py:
    RetrievalRecord
    RetrievalEpisodeSource
    RetrievalQueryMessage
  Exported from memory/__init__.py per existing convention.
- New exception AssistantMessageNotFoundError, mapped to 404
  in api/errors.py.

### Response shape
{
  "assistant_message_id": "uuid",
  "total": int,
  "retrievals": [
    {
      "rank": int,
      "similarity": float,
      "score": float,
      "recency_score": float,
      "access_score": float,
      "frequency_score": float,
      "importance_score": float,
      "scoring_version": "string",
      "retrieved_at": "iso8601",
      "query": {
        "message_id": "uuid",
        "content": "string"
      },
      "episode": {
        "id": "uuid",
        "kind": "message" | "summary",
        "content": "string",
        "source_conversation_id": "uuid",
        "source_conversation_title": "string" | null,
        "source_scope_id": "uuid",
        "source_scope_name": "string"
      }
    }
  ]
}

Notes:
- All score components non-nullable per schema.
- scoring_version per-row, not top-level.
- source_conversation_title nullable per schema; UI fallback.
- No content truncation server-side. Localhost; summary
  episodes may be long.

Intentionally omitted from the response (writer columns not
surfaced in 10b):
- embedding_model_id — single embedding model assumed; no UX
  for multi-model.
- message_retrievals.metadata — currently {} everywhere;
  surface when populated.

### Pydantic schemas
Added to api/schemas.py, following the existing inline-style
convention (BaseModel directly, model_config =
ConfigDict(extra="forbid"), snake_case fields, no shared base
model, no custom datetime serializer, from_record(...)
classmethods to construct from memory dataclasses):
- MessageRetrievalsResponse
- RetrievalEntry
- RetrievalQuery
- RetrievalEpisode

### Backend tests
- tests/test_api_app.py
    Update the asserted route set to include the new
    endpoint.
- tests/test_api_messages_retrieval.py
  (add tests to the existing file)
    - 200 + ordered retrievals[] for an owned assistant
      message
    - 200 + empty retrievals[] for an owned assistant
      message with no provenance
    - 404 for nonexistent conversation_id
    - 404 for nonexistent message_id within an owned
      conversation
    - 404 for user-role message within an owned
      conversation
    - 404 for system-role message within an owned
      conversation
    - 404 for message_id that belongs to a different
      (also owned) conversation
    - 403 for a conversation owned by a different user
- tests/test_memory_service.py
    - list_message_retrievals returns expected RetrievalRecord
      shape
    - Ordered by result_rank ASC
    - Read-only invariant: snapshot episodes.access_count and
      episodes.last_accessed_at before and after the call;
      assert both unchanged across the call
    - Raises AssistantMessageNotFoundError for invalid inputs

## Frontend

### Types
- Add the new endpoint to the backend OpenAPI surface.
- Regenerate frontend/src/api/types.ts via
  `pnpm generate:api`.
- Do not hand-write the response type.

### Typed path union (ConversationApiPath)
If frontend/src/api/client.ts exposes a typed path union
(e.g. ConversationApiPath) that constrains apiFetch paths
today, extend it with the new endpoint shape. If it is a
residual that no longer constrains anything, leave it alone.
Decision documented in the planning summary; do not modify
the file as part of contract writing.

### Hook
- File: frontend/src/features/messages/api/useMessageRetrievals.ts
- Signature: useMessageRetrievals(conversationId, messageId)
- Query key helper: messageRetrievalsQueryKey(conversationId,
  messageId) returning
    ["messageRetrievals", "list", conversationId, messageId]
      as const
  Matches messagesQueryKey style in useMessages.ts.
- queryFn: apiFetch<MessageRetrievalsResponse>(...) using the
  existing apiFetch wrapper.
- enabled: gated on panel expansion (lazy first-fetch).
- staleTime: Infinity. Provenance rows are immutable; they
  are cascade-deleted, never updated.

### UI
- New component:
    frontend/src/features/messages/components/MessageRetrievalsPanel.tsx
- New affordance in MessageItem.tsx, rendered only when
  message.role === "assistant".
- Toggle expands MessageRetrievalsPanel beneath the message.
- States: loading / error+retry / empty / success, using the
  existing inline-text conventions (muted text for
  loading/empty; text-danger with role="alert" for error,
  matching MessageList.tsx).
- Success rendering per row:
    - rank, similarity, final score (numeric, formatted)
    - score components (recency / access / frequency /
      importance) in an expandable subgroup
    - episode.kind badge
    - episode.content, with truncate+expand for long content
    - chip: scope_name › conversation_title
      (title fallback: "Untitled conversation")
    - retrieved_at (relative time)
    - scoring_version (small, muted)
- No edit, no rerun, no feedback affordance.

### Frontend tests
- Hook: loading / error / success.
- Panel: ordered rendering by rank.
- Panel: empty state.
- Panel: long-content truncate/expand.
- Panel: null conversation title fallback renders
  "Untitled conversation".
- MessageItem: affordance present on assistant messages,
  absent on user and system messages.

## Validation
- Backend: uv run ruff check . ; uv run ruff format --check . ;
  uv run mypy src/smriti/db src/smriti/config.py ;
  uv run pytest -q
- Frontend: pnpm generate:api ; pnpm typecheck ; pnpm lint ;
  pnpm test ; pnpm build

## Known acceptable limitations (documented, not fixed in 10b)
- Indistinguishable: "retrieval ran and returned 0 results"
  vs "no provenance recorded" (e.g., pre-10b assistant
  messages). Both render as empty. Sentinel rows would be
  required to disambiguate; out of scope.
- Retrievals into now-deleted source conversations are not
  shown (cascade-deleted with their source). Consistent with
  the Stage 10a privacy model.
- Single embedding model assumed; embedding_model_id is not
  surfaced.
- Inspection is unavailable during the streaming assistant
  draft; available only after the response is persisted and
  the message list refetches. The draft has no persisted
  message id and no provenance row yet.
