# Stage 11 - Summary Episode Memory Contract

## 1. Stage Metadata

- VERIFIED - Branch is `stage-11-summary-episode-memory` from `git branch --show-current` during this inspection.
- VERIFIED - Initial `git status --short` was empty before this contract was created.
- VERIFIED - `docs/stages/stage-11-summary-episode-memory.md` was absent from the sorted `docs/stages` listing before drafting; the existing stage docs stopped at Stage 10b plus `.DS_Store`.
- VERIFIED - The active local `.env` and `.env.stage11` name `postgresql://smriti:smriti@127.0.0.1:5432/smriti_dev_stage11` using `SMRITI_DATABASE_URL` (`.env:1`, `.env.stage11:1`).
- VERIFIED - `.env.main` names rollback DB `postgresql://smriti:smriti@127.0.0.1:5432/smriti` using `SMRITI_DATABASE_URL` (`.env.main:1`).
- VERIFIED - Runtime defaults are localhost-only for Postgres `127.0.0.1:5432`, FastAPI `127.0.0.1:8100`, Vite `127.0.0.1:5173`, and Ollama `127.0.0.1:11434` (`README.md:55-62`).
- VERIFIED - The roadmap marks Stage 11 as "window summary episodes (in progress)" (`README.md:314-334`).
- Current planning status: contract reconciliation complete; this contract is design-only and implementation is not started.

## 2. Locked Decisions

- Layer 1 = raw message episodes.
- Layer 2 = window summary episodes.
- Reuse the existing episodes table.
- Summary episodes use kind='summary'.
- Summary episodes are immutable.
- Summary episodes belong to exactly one conversation.
- Summary episodes are embedded into embeddings_768 via the existing embedding pipeline.
- No hierarchical summaries.
- No rolling summaries.
- No updating summaries.
- No v1 backfill.
- No retrieval SQL changes.
- No scoring/ranking changes.
- Summary episodes may be retrieved internally through the existing retrieval path.
- No user-visible summarization signal.
- User-facing provenance presentation must not expose `summary` as a visible source label.
- If provenance currently renders `episode.kind`, later implementation must either hide the kind label entirely or normalize all episode kinds uniformly to one generic label such as `memory`.
- Do not map only `summary -> memory` while leaving `message -> message`; that would still reveal summary-vs-message differences side by side.
- Summarization runs AFTER the SSE 'done' event.
- Summarization runs as fire-and-forget background work.
- Summary failure must NOT fail the user request.
- Embedding failure must NOT fail the user request.
- Failure produces a structured log only.
- No retry mechanism in Stage 11.
- Stage 11 implements only complete fixed-size window summaries.
- End-of-window handling is N-boundaries ONLY.
- NEVER summarize partial windows.
- NEVER summarize incomplete windows.
- NEVER create overlapping summary ranges.
- Tail/session-close summaries are intentional future work, not Stage 11.

## 3. Repo Findings

### A. Episodes Schema

- VERIFIED - The canonical schema file is only a pointer to append-only migrations; it does not define tables itself (`src/smriti/db/schema.sql:1-4`).
- VERIFIED - `episodes` is defined in migration 001 with `id`, `conversation_id`, `kind`, nullable `message_id`, nullable `range_start`, nullable `range_end`, `content`, `created_at`, `importance`, `access_count`, `last_accessed_at`, and `metadata` (`src/smriti/db/migrations/001_init.sql:35-47`).
- VERIFIED - `kind='summary'` is schema-supported explicitly by a CHECK constraint, not merely by free text: `kind TEXT NOT NULL CHECK (kind IN ('message', 'summary'))` (`src/smriti/db/migrations/001_init.sql:35-39`).
- VERIFIED - The schema enforces shape by kind: message episodes require `message_id` and no range; summary episodes require no `message_id` and non-null `range_start` and `range_end` (`src/smriti/db/migrations/001_init.sql:48-58`).
- VERIFIED - Summary ranges must satisfy `range_start <= range_end` (`src/smriti/db/migrations/001_init.sql:59-61`).
- VERIFIED - Only message episodes have a uniqueness index on `message_id`; no summary-window uniqueness index appears in migration 001 (`src/smriti/db/migrations/001_init.sql:64-69`).
- VERIFIED - Migration 003 adds non-null `scope_id`, an index, and scope/conversation foreign keys to `episodes` (`src/smriti/db/migrations/003_episode_scope_and_retrieval_provenance.sql:1-14`, `src/smriti/db/migrations/003_episode_scope_and_retrieval_provenance.sql:58-87`).
- VERIFIED - App-layer type `EpisodeKind` explicitly allows `"message"` and `"summary"` (`src/smriti/memory/models.py:8-9`).
- VERIFIED - Retrieval-oriented records are summary-capable: `ScoredEpisode` has `kind`, optional `message_id`, optional `message_position`, optional `range_start`, and optional `range_end` (`src/smriti/memory/models.py:155-171`).
- VERIFIED - The write-return `EpisodeRecord` is message-shaped today: it has no `kind`, no `range_start`, no `range_end`, and `message_id` is typed non-optional (`src/smriti/memory/models.py:129-138`).

Useful excerpt:

```sql
kind TEXT NOT NULL CHECK (kind IN ('message', 'summary')),
...
(kind = 'summary'
    AND message_id IS NULL
    AND range_start IS NOT NULL
    AND range_end IS NOT NULL)
```

Source: `src/smriti/db/migrations/001_init.sql:38-57`.

### B. Episode Write Sites

- VERIFIED - Production write site 1 is `MemoryService.append_message_with_episode`. It embeds `request.content`, inserts a `messages` row, then inserts an `episodes` row with hardcoded `kind='message'`, `conversation_id=request.conversation_id`, `scope_id` from `_lock_conversation_for_user`, `message_id` from the inserted message, and `content=request.content` (`src/smriti/memory/service.py:475-550`).
- VERIFIED - Production write site 2 is `MemoryService.create_message_episode`. It loads a stored message, embeds `message.content`, then inserts an `episodes` row with hardcoded `kind='message'`, `conversation_id=request.conversation_id`, `scope_id=request.scope_id`, `message_id=request.message_id`, and `content=message.content` (`src/smriti/memory/service.py:552-595`).
- VERIFIED - The HTTP message route delegates to `append_message_with_episode`; the route does not write SQL directly (`src/smriti/api/routes/messages.py:66-88`).
- VERIFIED - Test fixtures also create message episodes directly by SQL in DB and memory-service tests (`tests/test_db.py:349-358`, `tests/test_db.py:462-472`, `tests/test_db.py:608-618`, `tests/test_memory_service.py:787-798`, `tests/test_memory_service.py:1423-1433`).
- VERIFIED - One test directly writes `kind='summary'` by SQL with `range_start=1`, `range_end=3`, and an embedding row to verify conversation-delete cascades (`tests/test_memory_service.py:361-388`).
- VERIFIED - Current production application code does not write `kind='summary'`; both production episode inserts hardcode `'message'` (`src/smriti/memory/service.py:523-545`, `src/smriti/memory/service.py:570-593`).
- VERIFIED - Current test code does write `kind='summary'` directly as a fixture (`tests/test_memory_service.py:361-388`).

### C. Embedding Pipeline

- VERIFIED - The embedding protocol is async and accepts arbitrary text through `embed_text` / `embed_texts`; it does not include episode-kind parameters (`src/smriti/embeddings/base.py:9-20`).
- VERIFIED - `OllamaEmbedder` posts text batches to local Ollama `/api/embed` with model, input list, and `num_ctx`, then parses vectors (`src/smriti/embeddings/ollama.py:21-67`, `src/smriti/embeddings/ollama.py:108-157`).
- VERIFIED - `append_message_with_episode` embeds `request.content`, validates 768 dimensions, inserts the episode, then inserts `episode_id`, model PK, and vector into `embeddings_768` (`src/smriti/memory/service.py:484-545`).
- VERIFIED - `create_message_episode` embeds the loaded message content, validates 768 dimensions, inserts the episode, then inserts `episode_id`, model PK, and vector into `embeddings_768` (`src/smriti/memory/service.py:558-593`).
- VERIFIED - `embeddings_768` stores `episode_id`, `model_id`, `embedding vector(768)`, and `created_at`, keyed by `(episode_id, model_id)` (`src/smriti/db/migrations/001_init.sql:90-96`).
- VERIFIED - The embedding model PK is resolved from active `embedding_models` where `model_id='nomic-embed-text'` by default and dimensions equal 768 (`src/smriti/memory/service.py:106-112`, `src/smriti/memory/service.py:1169-1183`).
- VERIFIED - Embedding itself does not depend on `episode.kind`; the current production write functions are message-only because they hardcode `kind='message'`, not because the embedder requires message episodes (`src/smriti/memory/service.py:523-545`, `src/smriti/memory/service.py:570-593`).
- ASSUMED - A summary episode can use the same embedding primitives if later code inserts a schema-valid `kind='summary'` row and embeds its `content`; this would be confirmed by a later integration test that inserts a summary episode plus `embeddings_768` and retrieves it through `retrieve_scoped_episodes`.
- VERIFIED - A new production summary write path is required; the existing production episode-write methods require a message source and hardcode `kind='message'` (`src/smriti/memory/service.py:475-595`).

### D. Retrieval - Kind Filtering

- VERIFIED - `retrieve_scoped_episodes` retrieves from `episodes` joined to `conversations`, `embeddings_768`, and optionally `messages`; it selects `episodes.kind`, `message_id`, `range_start`, `range_end`, and `content` (`src/smriti/memory/service.py:627-648`).
- VERIFIED - The WHERE clause filters by scope and user, not by kind (`src/smriti/memory/service.py:649-663`).

Useful excerpt:

```sql
WHERE episodes.scope_id = $1
  AND conversations.user_id = $2
```

Source: `src/smriti/memory/service.py:649-663`.

- VERIFIED - Retrieval filters by conversation scope through the `conversations` join requiring matching `conversation_id` and `scope_id` (`src/smriti/memory/service.py:649-653`).
- VERIFIED - Retrieval filters by user/scope boundary via `episodes.scope_id = $1` and `conversations.user_id = $2` (`src/smriti/memory/service.py:659-660`).
- VERIFIED - Retrieval requires an embedded row for the active embedding model through the `INNER JOIN embeddings_768` and `model_id=$4` (`src/smriti/memory/service.py:653-655`).
- VERIFIED - Summary episodes would naturally be included by retrieval once written and embedded, because there is no kind predicate and the `messages` join is `LEFT JOIN` (`src/smriti/memory/service.py:653-660`).
- VERIFIED - The assistant prompt builder consumes retrieved episodes generically by score/content and does not branch on kind (`src/smriti/assistant/prompt_builder.py:36-43`, `src/smriti/assistant/prompt_builder.py:86-95`).
- VERIFIED - The retrieval search API exposes generic `ScoredEpisodeResponse` fields including `kind`, `message_id`, `message_position`, `range_start`, and `range_end` (`src/smriti/api/schemas.py:311-360`).
- VERIFIED - Retrieval provenance response models expose `episode.kind` (`src/smriti/api/schemas.py:231-254`).
- VERIFIED - The frontend generated API type allows retrieval episode `kind: "message" | "summary"` (`frontend/src/api/types.ts:448-474`).
- VERIFIED - The retrieval panel renders `retrieval.episode.kind` visibly (`frontend/src/features/messages/components/MessageRetrievalsPanel.tsx:91-99`).
- VERIFIED - The visible kind badge requires a bounded user-facing provenance presentation fix before Stage 11 acceptance: hide the kind label entirely, or normalize every episode kind uniformly to one generic label such as `memory` (`frontend/src/features/messages/components/MessageRetrievalsPanel.tsx:91-99`).
- VERIFIED - The presentation fix must not change retrieval SQL or ranking/scoring; `retrieve_scoped_episodes` already selects all embedded episode kinds without a kind predicate (`src/smriti/memory/service.py:627-663`).

### E. SSE Done Event and Background Task Mechanism

- VERIFIED - The streaming endpoint is FastAPI and returns `StreamingResponse` with `media_type="text/event-stream"` (`src/smriti/api/routes/assistant.py:7-8`, `src/smriti/api/routes/assistant.py:68-98`).
- VERIFIED - `_stream_sse_events` pulls events from an async iterator, yields encoded SSE strings, and closes the iterator in `finally` (`src/smriti/api/routes/assistant.py:101-115`).
- VERIFIED - The `done` SSE event is emitted by `_encode_sse_event` when the event is `AssistantStreamDone` (`src/smriti/api/routes/assistant.py:118-141`).
- VERIFIED - `AssistantOrchestrator.stream_prepared` persists the assistant response before yielding `AssistantStreamDone` (`src/smriti/assistant/orchestrator.py:139-159`).
- VERIFIED - Current code after `AssistantStreamDone` in `stream_prepared` does not perform more work before the function ends; the next method starts at line 161 (`src/smriti/assistant/orchestrator.py:154-161`).
- VERIFIED - No application source hit for `BackgroundTasks` or `asyncio.create_task` was found by read-only search. The only `asyncio.create_task` hit is in a test ASGI transport helper (`tests/test_api_assistant.py:926-942`).
- UNKNOWN - There is no existing internal background worker/helper in `src/smriti`; a future implementation must introduce a small local mechanism without adding dependencies.
- ASSUMED - The safest Stage 11 mechanism is a small `asyncio.create_task` helper scheduled from the SSE generator after an `AssistantStreamDone` has been yielded, because that preserves "after done" ordering and does not require a task queue. Confirmation will require later tests proving the SSE `done` event is observed even when summary work fails.

### F. Conversation and Window Boundaries

- VERIFIED - Messages are stored in `messages` with `conversation_id`, unique per-conversation `position`, `role`, `content`, `token_count`, `created_at`, and metadata (`src/smriti/db/migrations/001_init.sql:20-30`).
- VERIFIED - Message list retrieval orders by `position ASC, id ASC` (`src/smriti/memory/service.py:182-190`).
- VERIFIED - Normal message append computes the next position with `COALESCE(MAX(position), 0) + 1` (`src/smriti/memory/service.py:446-460`, `src/smriti/memory/service.py:1154-1167`).
- VERIFIED - Assistant response persistence inserts role `'assistant'` at the next conversation position (`src/smriti/memory/service.py:765-780`).
- VERIFIED - User and assistant messages are both stored; tests load recent user, assistant, and user messages through assistant context (`tests/test_assistant_persistence.py:58-103`).
- VERIFIED - `AppendMessageWithEpisodeRequest` accepts a `MessageRole`, and `MessageRole` includes `system`, `user`, and `assistant` (`src/smriti/memory/models.py:8`, `src/smriti/memory/models.py:68-75`).
- VERIFIED - The public create-message API body also accepts `MessageRole` and the route delegates to `append_message_with_episode`, so this route can create message episodes for any accepted role (`src/smriti/api/schemas.py:94-101`, `src/smriti/api/routes/messages.py:66-88`).
- VERIFIED - Generated assistant responses are persisted without creating episodes or embeddings; a persistence test asserts zero episodes and zero embeddings for the assistant message while provenance is stored (`tests/test_assistant_persistence.py:167-224`).
- VERIFIED - Current raw message episode helper fixtures append user messages before creating message episodes (`tests/test_memory_service.py:2927-2951`, `tests/test_assistant_persistence.py:395-410`).
- VERIFIED - Existing data available to detect N-boundaries is `messages.position` plus conversation membership; summary episodes also have `range_start` and `range_end` available for recording position windows (`src/smriti/db/migrations/001_init.sql:20-30`, `src/smriti/db/migrations/001_init.sql:35-47`).
- VERIFIED - No summarization/window service or config exists in source today; summary support is currently schema/type-level plus one deletion test fixture (`src/smriti/memory/service.py:475-595`, `src/smriti/config.py:18-30`, `tests/test_memory_service.py:361-388`).
- VERIFIED - Stage 10a explicitly scoped in whole-conversation deletion and scoped out message-level deletion and scope deletion (`docs/stages/stage-10a-conversation-delete.md:48-78`).
- VERIFIED - The current conversations API exposes `DELETE /conversations/{conversation_id}` and delegates to `MemoryService.delete_conversation` (`src/smriti/api/routes/conversations.py:55-72`).
- VERIFIED - The current messages API exposes message listing, retrieval-provenance listing, and message creation, with no message-delete route in that file (`src/smriti/api/routes/messages.py:25-88`).
- VERIFIED - The current scopes API exposes scope listing and creation, with no scope-delete route in that file (`src/smriti/api/routes/scopes.py:15-41`).
- VERIFIED - `MemoryService.delete_conversation` hard-deletes only a user-owned conversation row and relies on schema cascades for dependent rows (`src/smriti/memory/service.py:412-431`).
- VERIFIED - Stage 10a cascade tests verify whole-conversation deletion removes the deleted conversation's messages, message episodes, directly seeded summary episode, embeddings, and related provenance rows while preserving unrelated conversations and episodes (`tests/test_memory_service.py:249-553`).
- VERIFIED - Schema-level scope deletion cascades conversations if a scope row is deleted, and current DB tests exercise that only by direct SQL, not through an app API route (`src/smriti/db/migrations/002_scopes.sql:46-50`, `tests/test_db.py:266-289`).
- VERIFIED - Under current supported app behavior, there is no individual-message deletion surface, so app-created message positions are append-only within a surviving conversation (`src/smriti/memory/service.py:433-460`, `src/smriti/memory/service.py:1154-1167`, `src/smriti/api/routes/messages.py:25-88`).
- VERIFIED - Count-based N-boundaries and position-based N-windows are equivalent under current supported deletes because supported deletion removes an entire conversation or cascades an entire scope rather than leaving holes inside one surviving conversation (`src/smriti/api/routes/conversations.py:55-72`, `src/smriti/memory/service.py:412-431`, `tests/test_memory_service.py:249-553`).
- ASSUMED - Direct SQL, future individual-message deletion, or future partial redaction could create position holes; if such behavior is added before Stage 11 implementation, the window contract must be revisited.
- UNKNOWN - The repo has no existing policy for whether rare `system` rows in `messages` should count toward N; the proposed contract below counts persisted positions to avoid inventing another ordering concept.

### G. Summarization Model Access / Ollama

- VERIFIED - App lifespan wires `OllamaEmbedder` and `OllamaChatGenerator` by default, with injected fakes supported for tests (`src/smriti/api/app.py:34-65`).
- VERIFIED - Settings define local Ollama base URL, chat model, chat context window, embedding context window, and chat timeout (`src/smriti/config.py:26-30`).
- VERIFIED - `OllamaEmbedder` uses local `/api/embed` for embeddings (`src/smriti/embeddings/ollama.py:21-67`).
- VERIFIED - `OllamaChatGenerator` uses local `/api/chat` for both non-streaming and streaming generation (`src/smriti/chat/ollama.py:30-73`).
- VERIFIED - The chat interface can produce a complete text response with `ChatGenerator.generate(ChatRequest) -> ChatResponse(content=...)` (`src/smriti/chat/base.py:26-40`, `src/smriti/chat/base.py:62-70`).
- VERIFIED - `OllamaChatGenerator.generate` sends `{"stream": False}` and returns parsed assistant content, model, finish reason, and usage (`src/smriti/chat/ollama.py:49-60`, `src/smriti/chat/ollama.py:170-198`).
- VERIFIED - Chat and embedding clients validate localhost-only base URLs (`src/smriti/chat/ollama.py:140-149`, `src/smriti/embeddings/ollama.py:96-106`).
- ASSUMED - The existing `ChatGenerator` interface is sufficient for summary generation because it can take a summary prompt and return one text response. Confirmation requires a later summarizer test using `FakeChatGenerator` and an integration test using the wired `OllamaChatGenerator` only if real-Ollama tests are explicitly enabled.

### H. Logging Conventions

- VERIFIED - Backend source uses Python stdlib logging, not `structlog`; the only source logger found is `logging.getLogger(__name__)` in `src/smriti/api/app.py` (`src/smriti/api/app.py:3-29`).
- VERIFIED - Existing app logging records resolved Ollama model/context settings at INFO without message content (`src/smriti/api/app.py:45-50`).
- VERIFIED - API error handlers return sanitized JSON errors and do not log exception details in the handler code (`src/smriti/api/errors.py:32-187`).
- VERIFIED - Tests assert assistant API and orchestrator code do not log user, memory, or assistant content at INFO or below (`tests/test_api_assistant.py:617-658`, `tests/test_api_assistant.py:693-731`, `tests/test_assistant_orchestrator.py:379-420`, `tests/test_assistant_orchestrator.py:423-466`).
- VERIFIED - Chat tests assert prompt/response content is not logged at INFO or below (`tests/test_chat.py:468-524`).
- UNKNOWN - There is no existing application pattern for structured background-task exception logging; Stage 11 must define a small stdlib logging convention.

### I. Config

- VERIFIED - Runtime settings live in `Settings(BaseSettings)` with local defaults and an `.env` file (`src/smriti/config.py:11-37`).
- VERIFIED - Settings use `env_prefix="SMRITI_"` and ignore extra environment keys (`src/smriti/config.py:32-37`).
- VERIFIED - The default database URL in code is the baseline `smriti` database (`src/smriti/config.py:18`).
- VERIFIED - `.env.example` uses `SMRITI_DATABASE_URL`, matching the Settings prefix (`.env.example:1`).
- VERIFIED - Current `.env` and `.env.stage11` now use `SMRITI_DATABASE_URL=postgresql://smriti:smriti@127.0.0.1:5432/smriti_dev_stage11`, matching the Settings prefix (`.env:1`, `.env.stage11:1`, `src/smriti/config.py:32-37`).
- VERIFIED - The earlier env-prefix mismatch is resolved at the file level because `.env` now uses the `SMRITI_` prefix that `Settings` reads (`.env:1`, `src/smriti/config.py:32-37`).
- VERIFIED - `make start` calls `scripts/dev.sh start`, and dev start runs `uv run migrate up` with no explicit `--database-url`, then starts uvicorn with no explicit database override (`Makefile:6-8`, `scripts/dev.sh:268-269`, `scripts/dev.sh:333-340`).
- VERIFIED - Migration CLI reads `get_settings()` and only overrides the database URL when `--database-url` is provided (`src/smriti/db/migrate.py:121-133`).
- UNKNOWN - The effective runtime value has not been verified in this contract pass; Stage 11 implementation is still gated on running the preflight in the same shell/runtime that will write data.
- VERIFIED - A later summary window size and optional feature flag would belong in `Settings` beside other runtime knobs (`src/smriti/config.py:11-37`).

### J. Existing Tests

- VERIFIED - Schema and migration tests cover table creation, vector extension, embedding model seed, scope columns, and migration idempotence (`tests/test_db.py:20-160`).
- VERIFIED - Migration 003 tests backfill existing episode `scope_id` for pre-scoped message episodes (`tests/test_db.py:295-383`).
- VERIFIED - Message retrieval schema tests cover episode/provenance foreign keys and assistant-message linkage (`tests/test_db.py:545-745`).
- VERIFIED - Memory service tests cover message episode creation and embedding insertion (`tests/test_memory_service.py:48-245`).
- VERIFIED - Memory service tests cover deletion cascades for both message and directly seeded summary episodes plus embeddings (`tests/test_memory_service.py:249-550`).
- VERIFIED - Memory service tests cover embedding dimension rejection before writing episodes (`tests/test_memory_service.py:643-708`).
- VERIFIED - Retrieval tests cover scoped/embedded-only results, access metadata updates, SQL scope/user filtering before ordering, ordering, top_k, and validation (`tests/test_memory_service.py:712-820`, `tests/test_memory_service.py:1272-1388`, `tests/test_memory_service.py:1561-1590`).
- VERIFIED - API message/retrieval tests cover create-message -> episode -> embedding -> retrieval, no embedding leakage, and no provenance write from retrieval search (`tests/test_api_messages_retrieval.py:59-130`).
- VERIFIED - API message tests cover embedding failure causing no partial message/episode write (`tests/test_api_messages_retrieval.py:246-288`).
- VERIFIED - API route delegation tests assert message/retrieval routes call memory service methods rather than SQL directly (`tests/test_api_messages_retrieval.py:431-535`).
- VERIFIED - Assistant persistence tests cover recent-message loading through the query and assistant response persistence without an episode or embedding for the assistant message (`tests/test_assistant_persistence.py:39-108`, `tests/test_assistant_persistence.py:125-229`).
- VERIFIED - Assistant orchestrator streaming tests cover start/token/done ordering, persistence before done, generation failures, cancellation, and content logging guardrails (`tests/test_assistant_orchestrator.py:101-158`, `tests/test_assistant_orchestrator.py:159-279`, `tests/test_assistant_orchestrator.py:379-466`).
- VERIFIED - Assistant API streaming tests cover SSE start/token/done payloads, disconnect behavior, persistence rollback, and logging guardrails (`tests/test_api_assistant.py:274-350`, `tests/test_api_assistant.py:423-616`, `tests/test_api_assistant.py:617-732`).
- VERIFIED - Embedding tests cover fake embedder determinism and Ollama embed payload/error behavior (`tests/test_embeddings.py:26-168`).
- VERIFIED - Chat tests cover fake chat generators, non-streaming Ollama payloads, streaming Ollama payloads, and no prompt/response content logging (`tests/test_chat.py:32-96`, `tests/test_chat.py:131-297`, `tests/test_chat.py:468-524`).
- VERIFIED - Config tests cover current Ollama env names/defaults and invalid config rejection (`tests/test_config.py:9-48`).
- VERIFIED - Frontend SSE tests parse `start`, `token`, `done`, and error cases; the hook returns to idle on `done` (`frontend/src/lib/sse.ts:29-33`, `frontend/src/lib/sse.test.ts:16-71`, `frontend/src/features/chat/api/useAssistantResponseStream.ts:121-167`, `frontend/src/features/chat/api/useAssistantResponseStream.test.tsx:44-86`).
- VERIFIED - Frontend retrieval panel tests cover ordering, empty state, long-content expansion, and source-title fallback; fixtures default `episode.kind` to `"message"` (`frontend/src/features/messages/components/MessageRetrievalsPanel.test.tsx:13-92`, `frontend/src/features/messages/components/MessageRetrievalsPanel.test.tsx:135-170`).

## 4. Conflicts / Scope Issues

- VERIFIED - The earlier `.env` env-prefix mismatch is resolved at the file level: `.env` and `.env.stage11` now use `SMRITI_DATABASE_URL`, matching `Settings` (`.env:1`, `.env.stage11:1`, `src/smriti/config.py:32-37`).
- UNKNOWN - The effective runtime database URL is not verified by this contract edit. Before Stage 11 implementation writes any data, the implementation shell must run the preflight in section 10 and confirm the output includes `smriti_dev_stage11`.
- VERIFIED - The earlier provenance-label conflict is reconciled by a bounded presentation exception: user-facing provenance may not expose raw `episode.kind`, and later implementation may only hide the kind label entirely or normalize all episode kinds uniformly to one generic label such as `memory` (`frontend/src/features/messages/components/MessageRetrievalsPanel.tsx:91-99`, `frontend/src/api/types.ts:448-474`).
- UNKNOWN - Human owner acceptance of the bounded provenance-label exception is still required before implementation because it permits a narrow frontend/API presentation change even though Stage 11 otherwise avoids frontend work.
- VERIFIED - The existing production write-return `EpisodeRecord` is message-shaped and does not expose `kind` or summary ranges (`src/smriti/memory/models.py:129-138`). Later implementation may need a summary-specific record or a careful model extension; this contract does not choose the code shape.
- UNKNOWN - There is no existing application background-task helper or structured background logging convention in source (`src/smriti/api/routes/assistant.py:101-115`, `src/smriti/api/app.py:3-50`). Later implementation must add a small local pattern without dependencies.
- VERIFIED - The schema has no summary-window uniqueness constraint, so Stage 11 uses service-level duplicate prevention and adds no migration unless a human explicitly changes scope (`src/smriti/db/migrations/001_init.sql:64-69`).
- VERIFIED - No individual-message delete surface exists in current supported app behavior, so count-based N-boundaries and persisted-position windows are equivalent under current deletes (`src/smriti/api/routes/messages.py:25-88`, `src/smriti/api/routes/conversations.py:55-72`, `src/smriti/memory/service.py:412-431`).

## 5. Proposed Contract

### a. Trigger Point and Background Mechanism

- Trigger only after the SSE `done` event for a successfully persisted assistant response.
- The later hook should live on the SSE stream path, because `_encode_sse_event` is where `AssistantStreamDone` becomes the wire `done` event (`src/smriti/api/routes/assistant.py:118-141`), and `stream_prepared` has already persisted the assistant response before yielding that event (`src/smriti/assistant/orchestrator.py:139-159`).
- Chosen mechanism for later implementation: schedule a small `asyncio.create_task` helper after the `AssistantStreamDone` event has been yielded by `_stream_sse_events`.
- The summary background task must acquire its own pooled DB connection and must not reuse a request-scoped connection/session that may be closed after the SSE response lifecycle.
- The scheduled task must be retained in a small in-process task registry/set until completion so it is not garbage-collected silently.
- The task must remove itself from the registry when done.
- The task must catch all exceptions internally and emit only the structured failure log described below.
- No retries in Stage 11.
- No task queue, no dependency addition, and no blocking of the SSE response.

### b. Window / N-Boundary Definition

- Default N: `12` persisted messages per conversation window.
- Stage 11 implements only complete fixed-size windows.
- What counts toward N: persisted rows in `messages` for the conversation, ordered by `position`, regardless of role. This uses the only existing total order (`src/smriti/db/migrations/001_init.sql:20-30`, `src/smriti/memory/service.py:1154-1167`).
- Stage 11 window ranges use persisted `messages.position`, not an independently computed ordinal.
- Count-based N-boundaries and position-based windows are equivalent under current supported deletes because the app does not support individual message deletion; supported conversation deletion removes the whole conversation and its dependent rows (`src/smriti/api/routes/messages.py:25-88`, `src/smriti/api/routes/conversations.py:55-72`, `tests/test_memory_service.py:249-553`).
- Normal UI traffic makes this approximately six user/assistant exchanges because generated assistant responses are persisted after user messages (`src/smriti/memory/service.py:765-780`).
- Complete windows are position ranges:
  - window 1: `1..12`
  - window 2: `13..24`
  - window 3: `25..36`
  - window k: `((k - 1) * N + 1)..(k * N)`
- A window is complete only if exactly N messages exist in the candidate position range for that conversation.
- Later implementation must still verify exactly N rows exist in the candidate range before summarizing.
- The background task checks the latest persisted message count only after `done`; if `count % N != 0`, it does nothing.
- If future supported behavior can delete individual messages or otherwise leave position holes inside a surviving conversation, this window contract becomes a scope issue and must be revisited before implementation because existing summary ranges could become stale, misleading, or orphaned from the intended message window.
- Partial windows are never summarized.
- Incomplete windows are never summarized.
- Windows ending before the current N-boundary are not backfilled in Stage 11.
- Stage 11 does not create inactive-tail summaries.
- Stage 11 does not create time-based summaries.
- Stage 11 does not create manual "end session" summaries.
- Stage 11 does not create partial summaries for conversations with fewer than N messages.
- Stage 11 does not create partial summaries for leftover tails such as `13..21` or `25..32`.
- Examples below assume normal Stage 11 operation as the conversation crosses N-boundaries while summary memory is enabled; they do not create a v1 backfill obligation for pre-existing conversations.
  - Pre-existing conversations whose earlier N-boundaries were already missed are not backfilled in Stage 11.
  - 8-message conversation -> no summary episode in Stage 11.
  - 21-message conversation -> summary `1..12` only; `13..21` remains raw-only.
  - 32-message conversation -> summaries `1..12` and `13..24` only; `25..32` remains raw-only.
- Tail/session-close summaries are intentional future work and require a separate design for conversation finalization, non-overlapping summary ranges, resume/reopen behavior, and whether tail summaries are final or provisional.
- Stage 11 implementation must not create overlapping summary ranges.

### c. Summary Episode Construction

- Insert into the existing `episodes` table with `kind='summary'`.
- Bind exactly one `conversation_id` and the same `scope_id` as the conversation.
- Set `message_id = NULL`.
- Set `range_start` and `range_end` to the inclusive `messages.position` window boundaries.
- Write the summary text into `episodes.content`.
- Let `created_at` default to database `NOW()` unless later tests require a deterministic injected timestamp.
- Leave `importance`, `access_count`, `last_accessed_at`, and `metadata` at defaults unless later implementation needs a non-content metadata marker.
- Summary episodes are immutable: no updates to `content`, range, kind, or conversation binding.
- Metadata/window fields available today: `range_start`, `range_end`, `metadata`, `created_at`.
- Metadata/window fields missing today: no explicit `window_index`, no summary version field, no summary model field, and no uniqueness constraint for summary windows.
- Duplicate prevention is service-level only in Stage 11.
- Before inserting a summary episode, the summary writer must check for an existing row with the same `conversation_id`, same `scope_id`, `kind='summary'`, same `range_start`, and same `range_end`.
- If such a row exists, the task exits successfully without writing another summary.
- The summary writer must not create a summary range that overlaps an existing summary episode range for the same `conversation_id` and `scope_id`.
- No migration and no partial unique index are added in Stage 11 unless a human explicitly changes scope.
- A future stage may add a partial unique index if service-level duplicate prevention proves insufficient.

### d. Embedding Flow

- Reuse the existing embedding primitives:
  - call `embedder.embed_text(summary_content)`;
  - validate a 768-dimensional vector;
  - resolve active `embedding_models.id`;
  - insert into `embeddings_768 (episode_id, model_id, embedding)`.
- Embedding does not depend on kind; the summary path needs only a schema-valid summary episode and content text.
- Expected insertion target is the existing `embeddings_768` table.
- The contract prefers leaving no unembedded summary episode behind.
- Exact ordering for later implementation:
  1. Outside the DB transaction:
     - load the complete message window;
     - call the summarizer/model;
     - receive non-empty summary text;
     - optionally call the embedder for the summary text before opening the DB transaction if the implementation chooses the embed-before-transaction path.
  2. Then open one DB transaction:
     - re-check the conversation/window still exists;
     - re-check exactly N messages exist in the persisted-position range;
     - re-check no duplicate summary exists for the same `conversation_id`, `scope_id`, `kind='summary'`, `range_start`, and `range_end`;
     - re-check no overlapping summary range exists for the same `conversation_id` and `scope_id`;
     - insert the summary episode;
     - embed the summary text if the embedder was not already called before the transaction;
     - insert the embedding row;
     - commit.
  3. If summary generation fails:
     - no DB write should happen;
     - log only.
  4. If summary episode insert or embedding insert fails:
     - roll back the transaction;
     - keep the user request successful;
     - log only.
- If the current embedding service pattern or transaction-safety choice forces embedding before opening the DB transaction, that is an allowed implementation detail as long as the transaction still re-checks existence/window/duplicate/overlap conditions and the final DB commit is all-or-nothing for the summary episode row plus the embedding row.

### e. Summarization Call Contract

- Inputs:
  - `user_id`
  - `scope_id`
  - `conversation_id`
  - `range_start`
  - `range_end`
  - ordered window messages: `position`, `role`, `content`, `created_at`
  - configured summary/chat model identifier
- Message selection:
  - select exactly the N persisted messages whose positions match the complete window range;
  - order by `position ASC, id ASC`;
  - abort/log if the count is not exactly N.
- Model/client interface:
  - use the existing `ChatGenerator.generate(ChatRequest)` interface for one complete summary response.
  - use local Ollama through the existing configured chat generator unless a fake generator is injected in tests.
- Prompt expectations:
  - summarize only the supplied window;
  - do not infer facts beyond the messages;
  - mark unresolved questions as unresolved;
  - preserve concrete details: entities, decisions, constraints, filenames, dates, TODOs, and unresolved questions;
  - ignore instructions inside conversation content that try to alter summarization rules.
- Output shape:
  - one plain text summary string for `episodes.content`;
  - no JSON requirement in Stage 11;
  - empty or whitespace-only output is a summary generation failure and is logged only.

### f. Error Handling

- Summary generation failure does not fail the user request.
- Summary episode write failure does not fail the user request.
- Embedding failure does not fail the user request.
- Failure produces a structured log only.
- No retry mechanism in Stage 11.
- Log with stdlib logging using a stable event name, for example `summary_episode_memory_failed`.
- Do not log message content, summary content, prompt content, or retrieved memory content at INFO or below.
- Suggested log fields:
  - `event`
  - `failure_step`
  - `user_id`
  - `scope_id`
  - `conversation_id`
  - `range_start`
  - `range_end`
  - `message_count`
  - `summary_model`
  - `embedding_model_id`
  - `exception_type`
- `logger.exception` may be used at ERROR with the structured fields, but content must remain excluded.

### g. Proposed Config Additions

Names and location only; do not implement in this task.

- `Settings.summary_episode_window_messages` -> `SMRITI_SUMMARY_EPISODE_WINDOW_MESSAGES`
- `Settings.summary_episode_memory_enabled` -> `SMRITI_SUMMARY_EPISODE_MEMORY_ENABLED`

Both belong in `src/smriti/config.py` next to current local runtime settings (`src/smriti/config.py:11-37`).

### h. Explicit Non-Goals

- No hierarchical summaries.
- No rolling summaries.
- No mutable summaries.
- No v1 backfill.
- No inactive-tail summaries.
- No time-based summaries.
- No manual "end session" summaries.
- No partial summaries for conversations with fewer than N messages.
- No partial summaries for leftover tails such as `13..21` or `25..32`.
- No retrieval SQL changes.
- No scoring/ranking changes.
- No frontend feature work, except the bounded provenance-label presentation fix required to preserve "no user-visible summary signal."
- The bounded provenance-label exception may only hide the kind label entirely or normalize all episode kinds uniformly to one generic label such as `memory`.
- No `summary -> memory` only mapping while leaving `message -> message`.
- No user-visible summary signal in UI/API provenance presentation.
- No migration or partial unique index in Stage 11 unless a human explicitly changes scope.
- No application code changes in this planning task.

### i. Known Limitations

- Uncovered tail limitation: conversations with leftover messages after the last complete N-window keep those tail messages raw-only in Stage 11.
- Similarity-only retrieval mix limitation: summary episodes and raw message episodes enter the existing retrieval SQL and ranking/scoring mix together; Stage 11 does not add kind-specific retrieval filters, summary-specific weighting, or a separate retrieval lane.
- These limitations are intentional for Stage 11 and must be documented in later implementation notes/tests where relevant.

## 6. Data Flow

```text
user turn
-> assistant stream
-> SSE done
-> post-done background task
-> window check
-> summary generation
-> all-or-nothing summary episode + embedding transaction
-> normal retrieval
```

## 7. Test Plan

Later implementation tests should cover:

- Unit: no summary before an N-boundary.
- Unit: summary exactly at an N-boundary.
- Unit: no partial summary when `message_count % N != 0`.
- Unit/integration: no incomplete summary when the candidate range has fewer than N rows.
- Unit/integration: persisted-position ranges are used, and exactly N rows must exist before summarization.
- Unit/integration: 8-message conversations produce no summary episode.
- Unit/integration: 21-message conversations produce only summary `1..12`; `13..21` remains raw-only.
- Unit/integration: 32-message conversations produce only summaries `1..12` and `13..24`; `25..32` remains raw-only.
- Integration: no duplicate summary for the same `(conversation_id, scope_id, kind='summary', range_start, range_end)`.
- Integration: no overlapping summary ranges for the same conversation/scope.
- Integration: summary episode uses `kind='summary'`, `message_id IS NULL`, and non-null range boundaries.
- Integration: summary episode belongs to exactly one conversation and scope.
- Integration: summary episode is embedded into `embeddings_768`.
- Integration: summary episode and embedding insertion are all-or-nothing, leaving no unembedded summary episode after insert/embedding failure.
- Integration: embed-before-transaction ordering, if used, still re-checks the window/duplicate/overlap conditions inside the transaction before committing summary episode plus embedding.
- Integration: retrieval includes the embedded summary without retrieval SQL changes.
- Integration: summary and raw message episodes share the existing retrieval SQL/ranking path without kind-specific scoring changes.
- API/SSE: summary generation failure does not fail the chat request or alter the `done` payload.
- API/SSE: summary episode write failure does not fail the chat request.
- API/SSE: embedding failure does not fail the chat request.
- API/SSE: the background task acquires its own pooled connection, is retained in an in-process registry until completion, and removes itself when done.
- Logging: structured failure log is emitted with IDs/range/step and without message or summary content.
- Frontend/API: no user-visible summary signal; provenance either hides the kind label entirely or normalizes all episode kinds uniformly to a generic label such as `memory`.
- Frontend/API: no `summary -> memory` only mapping while `message` remains visibly labeled as `message`.
- Regression: assistant generated messages still do not create raw message episodes.
- Regression: existing message episode creation and retrieval tests continue to pass.

## 8. Rollback Plan

- Baseline DB `smriti` remains untouched.
- Stage 11 DB is intended to be `smriti_dev_stage11`.
- Summary episodes are additive rows in the existing `episodes` table.
- Rollback can remove summary embeddings and summary episodes from the stage DB:

```sql
DELETE FROM embeddings_768
WHERE episode_id IN (
    SELECT id FROM episodes WHERE kind = 'summary'
);

DELETE FROM episodes
WHERE kind = 'summary';
```

- No v1 backfill means rollback does not rewrite historical baseline data.
- The `.env` file-level DB target is now `SMRITI_DATABASE_URL=postgresql://smriti:smriti@127.0.0.1:5432/smriti_dev_stage11`, but human confirmation is still required before implementation that the effective runtime points at `smriti_dev_stage11` (`.env:1`, `src/smriti/config.py:32-37`).

## 9. Acceptance Criteria

- Section 4 is reviewed by a human before coding.
- The effective `Settings().database_url` is confirmed to include `smriti_dev_stage11` for Stage 11 implementation.
- The human owner accepts the bounded provenance-label exception.
- Later implementation creates summary episodes only at N-boundaries.
- Later implementation never summarizes partial or incomplete windows.
- Later implementation never summarizes inactive tails, time-based tails, manual end-session tails, or leftover tails such as `13..21` or `25..32`.
- Later implementation never creates overlapping summary ranges.
- Later implementation writes schema-valid immutable `kind='summary'` episodes.
- Later implementation embeds summaries into `embeddings_768`.
- Later implementation leaves no unembedded summary episode behind after summary/embedding insert failure.
- Existing retrieval SQL and scoring are unchanged.
- Summary, write, and embedding failures are isolated from the user request and logged only.
- No new dependencies are added.
- No frontend behavior changes are made except the accepted bounded provenance-label presentation fix.
- Later implementation includes tests from section 7.

## 10. Implementation Status

- Clarification checklist:
  - RESOLVED - Background-task connection/reference lifecycle is specified: own pooled DB connection, no request-scoped connection/session reuse, task retained in an in-process registry until completion, self-removal on completion, caught/logged exceptions, no retries, no queue/dependency.
  - RESOLVED - Deletion-granularity verification is specified: current supported app behavior has no individual message delete surface, count-based and position-based windows are equivalent under current deletes, and future individual-message deletion/partial redaction must revisit the contract because summaries could become stale, misleading, or orphaned from the intended window.
  - RESOLVED - Duplicate prevention is specified: service-level duplicate check for the same `conversation_id`, `scope_id`, `kind='summary'`, `range_start`, and `range_end`, successful no-op on duplicate, no Stage 11 migration/partial unique index.
  - RESOLVED - Non-overlap is specified: Stage 11 must not create overlapping summary ranges for the same conversation/scope.
  - RESOLVED - Embed-before-transaction ordering is specified as an allowed implementation detail only when the transaction still re-checks existence/window/duplicate/overlap conditions and commits summary episode plus embedding all-or-nothing.
  - RESOLVED - Known limitation: uncovered tails remain raw-only in Stage 11.
  - RESOLVED - Known limitation: summary and raw message episodes share the existing similarity/ranking retrieval mix with no kind-specific filtering or scoring changes in Stage 11.
- No still-open clarification item remains from the final contract clarification checklist.
- If any checklist item above is removed or contradicted, implementation becomes blocked again until the contract is reconciled.
- Implementation remains blocked until the effective runtime database URL is verified in the implementation shell.
- Required preflight:

```bash
PYTHONPATH=src uv run python -c "from smriti.config import Settings; print(Settings().database_url)"
```

- Expected output must include `smriti_dev_stage11`.
- This preflight is read-only and must happen before any application writes.
- Implementation remains blocked until the human owner accepts the bounded provenance-label exception.
- Implementation remains blocked until this updated contract has no internal contradictions.

## 11. Implementation Checklist for Later Coding

DO NOT EXECUTE IN THIS TASK.

1. Run the read-only preflight from section 10 and confirm the output includes `smriti_dev_stage11`.
2. Confirm the human owner accepts the bounded provenance-label exception.
3. Add config fields for summary window size and optional feature flag.
4. Add a summary-specific memory service method that locks the conversation, detects complete fixed-size N-boundaries, re-checks exactly N rows in the persisted-position range, performs service-level duplicate and overlap prevention, writes a schema-valid summary episode, and embeds it all-or-nothing.
5. Add a summarizer wrapper using `ChatGenerator.generate(ChatRequest)`.
6. Add a post-done SSE hook that schedules fire-and-forget summary work after `AssistantStreamDone`, retains the task in a small in-process registry until completion, and ensures the task acquires its own pooled DB connection.
7. Add structured ERROR logging for background failures with no message/summary content.
8. Add focused unit tests for boundary detection and prompt/output handling.
9. Add integration tests for summary episode write, embedding, retrieval inclusion, duplicate prevention, overlap prevention, uncovered-tail behavior, transaction rollback, provenance-label presentation, and failure isolation.
10. Run backend verification:
    `uv run ruff check .`
    `uv run ruff format --check .`
    `uv run mypy src/smriti/db src/smriti/config.py`
    `uv run pytest -q`
