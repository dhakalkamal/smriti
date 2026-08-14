# Smriti

Smriti is a localhost-only memory and chat application for local LLMs. It stores
conversation history in user-defined scopes, embeds user messages with Ollama,
retrieves relevant memories from the active scope, and uses those memories while
generating assistant responses through a local Ollama chat model.

This project is in active development. Contributions are welcome. 
## Current Status

Implemented:

- Python memory service using raw SQL, asyncpg, Postgres 16, and pgvector
- Append-only messages and message-backed retrieval episodes
- User-defined scopes with enforced scope-filtered retrieval
- Ollama embeddings with `nomic-embed-text`, 768 dimensions
- Weighted retrieval scoring using similarity, recency, access reinforcement,
  importance, and access frequency
- Used-memory provenance for assistant responses
- Local FastAPI API with narrow localhost CORS
- Non-streaming and SSE streaming assistant generation
- React, TypeScript, Vite, Tailwind, and TanStack Query frontend
- Scope creation and listing in the UI
- Conversation creation, listing, selection, and hard deletion in the UI
- Streaming chat UI with transient assistant drafts, Stop, and Retry
- Offline OpenAPI export and generated frontend API types
- Deterministic fake embedders and chat generators for tests
- Minimal retrieval eval helper
- Provenance visualization in the UI
- Read-only retrieval inspection panel 
- Explicit Ollama context window configuration 
- hybrid retrieval 


Not implemented:

- Rolling summarization
- Scope editing or deletion
- Conversation rename
- Message-level deletion
- Memory episode management UI
- Cross-scope retrieval
- Hybrid search, rerankers, or query rewriting
- Authentication or multi-user accounts

## Privacy Model

With the bundled UI, FastAPI backend, Postgres database, and Ollama running on
localhost, user data stays on the machine. Messages, embeddings, retrieved
memory context, assistant prompts, and generated responses are handled by local
services.

The current codebase does not include an MCP server. External MCP clients are
therefore not part of the current product path.

Smriti does not include telemetry, analytics, accounts, signup flows, CDN
assets, Google Fonts, or remote model providers. Runtime defaults bind services
to localhost:

- Postgres: `127.0.0.1:5432`
- FastAPI: `127.0.0.1:8100`
- Vite: `127.0.0.1:5173`
- Ollama: `127.0.0.1:11434`

## Architecture

```text
+------------------------------- User machine -------------------------------+
|                                                                            |
|  Browser                                                                   |
|  React app                                                                 |
|  127.0.0.1:5173                                                            |
|      |                                                                     |
|      | HTTP and fetch-based SSE                                            |
|      v                                                                     |
|  FastAPI local API                                                         |
|  127.0.0.1:8100                                                            |
|      |                                                                     |
|      | calls service layer only                                            |
|      v                                                                     |
|  Memory service and assistant orchestrator                                 |
|      |                                                                     |
|      | raw SQL                         local HTTP                          |
|      v                                 v                                   |
|  Postgres 16 + pgvector             Ollama                                 |
|  127.0.0.1:5432                     127.0.0.1:11434                       |
|  Docker                             embeddings and chat                    |
|                                                                            |
+----------------------------------------------------------------------------+
```

The memory service is the core boundary. FastAPI routes depend on it rather than
issuing memory SQL directly. The assistant orchestrator composes the memory
service with the local chat generator for prompt construction, generation,
persistence, and provenance.

## API Surface

The local FastAPI app currently exposes:

- `GET /health`
- `GET /scopes`
- `POST /scopes`
- `GET /conversations`
- `POST /conversations`
- `DELETE /conversations/{conversation_id}`
- `GET /conversations/{conversation_id}/messages`
- `POST /conversations/{conversation_id}/messages`
- `POST /retrieval/search`
- `POST /conversations/{conversation_id}/assistant-response`
- `POST /conversations/{conversation_id}/assistant-response/stream`

The app disables HTTP documentation and OpenAPI routes at runtime. OpenAPI JSON
for frontend type generation is exported offline through
`scripts/export_openapi.py`.

## Data Model

Key tables:

- `users`: local identity. The API bootstraps one configured local user.
- `scopes`: user-defined memory partitions with system prompts.
- `conversations`: chat threads that belong to one scope.
- `messages`: immutable user and assistant turns.
- `episodes`: retrieval units. Current UI-created user messages create
  `kind = 'message'` episodes. When summary episode memory is enabled, complete
  fixed-size conversation windows create immutable `kind = 'summary'` episodes.
- `embedding_models`: local embedding model registry.
- `embeddings_768`: pgvector embeddings for 768-dimensional episode vectors.
- `message_retrievals`: provenance records for memories used in assistant
  responses.
- `eval_*`: early eval harness scaffolding.

Generated assistant responses are persisted as messages with provenance, but
they are not embedded as raw message episodes. Stage 11 summary memory can add
embedded summary episodes after the SSE `done` event for complete 12-message
windows.

## Setup

### Prerequisites

- macOS or Linux
- make and Bash
- uv
- Python 3.13.x
- Docker with Compose v2
- Node.js
- pnpm 11.1.2
- Ollama, installed and started manually outside Smriti
- curl

Postgres runs through Docker Compose, so no local Postgres installation or
`psql` client is required. Ollama must be running at `127.0.0.1:11434` before
starting Smriti. Required Ollama models are:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

`qwen2.5:7b` is the default chat model. If `SMRITI_OLLAMA_CHAT_MODEL` is
overridden in `.env`, install that model instead.

Summary episode memory is controlled by:

```text
SMRITI_SUMMARY_EPISODE_MEMORY_ENABLED=false
SMRITI_SUMMARY_EPISODE_WINDOW_MESSAGES=12
```

The feature is disabled by default unless explicitly enabled in local config.
Stage 11 only summarizes complete fixed-size windows such as `1..12` and
`13..24`; it does not summarize partial tails.

### First-Time Setup

From a fresh clone, run:

```bash
make setup
```

`make setup` installs locked backend and frontend dependencies, creates `.env`
from `.env.example` only when it is missing, repairs the known macOS editable
install hidden-flag issue when needed, verifies the editable Python import, and
checks Ollama reachability and required models. It does not start services, run
migrations, overwrite `.env`, or pull Ollama models automatically.

### Daily Development

Start Smriti:

```bash
make start
```

`make start` starts Postgres through Docker Compose, applies migrations,
verifies Ollama and the required models, starts the backend and frontend, waits
for readiness, and opens the UI. Start Ollama manually outside Smriti before
running it.

Useful local development commands:

```bash
make status
make logs
make stop
make restart
```

The UI runs at:

```text
http://127.0.0.1:5173
```

### Optional Manual Troubleshooting

The Make targets are the canonical workflow. The equivalent lower-level
commands are available when diagnosing a local setup problem.

Install backend dependencies and repair macOS editable-install flags:

```bash
uv sync --python 3.13 --extra dev --locked
./scripts/fix-venv.sh
uv run python -c "import smriti; print(smriti.__file__)"
```

Start Postgres and apply migrations:

```bash
docker compose up -d --wait postgres
uv run migrate up
```

Run the backend:

```bash
uv run python -m uvicorn smriti.api:create_app --factory --host 127.0.0.1 --port 8100
```

Install and run the frontend:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

On macOS, if editable imports fail with `ModuleNotFoundError: No module named
'smriti'`, run `./scripts/fix-venv.sh` before changing packaging.

## Frontend API Types

Generated API types are committed at `frontend/src/api/types.ts`. Regenerate
them after backend API shape changes:

```bash
cd frontend
pnpm generate:api
```

This command runs the backend OpenAPI exporter offline. It should not require
Postgres, Ollama, or a running FastAPI server.

## Validation

Backend validation from the repository root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/smriti/db src/smriti/config.py
uv run pytest -q
```

Frontend validation from `frontend/`:

```bash
pnpm generate:api
pnpm typecheck
pnpm lint
pnpm test
pnpm build
```

Repository hygiene:

```bash
git diff --check
```

## Concepts

**Scopes.** A scope is a user-defined memory partition with its own system
prompt. Retrieval requires a `scope_id` and only returns episodes from that
scope.

**Conversations.** A conversation is a chat thread inside one scope. Multiple
conversations in the same scope share the same retrieval pool.

**Messages.** Messages are immutable conversation turns. User messages submitted
through the normal message endpoint are also archived as retrieval episodes.

**Episodes.** Episodes are the retrieval unit. Message episodes point to one
message. Summary episodes cover one complete fixed-size message-position window
inside a single conversation and are embedded into the same retrieval path.

**Retrieval.** Retrieval embeds the query, searches only episodes in the active
scope, scores candidates, updates access metadata for returned episodes, and
returns structured scored records.

**Assistant generation.** Assistant generation starts from a persisted user
message. The backend retrieves scoped memories, builds a prompt from the scope
prompt, fixed local instructions, selected memories, and recent messages, calls
local Ollama, then persists the assistant response and used-memory provenance.

**SSE streaming.** The streaming endpoint emits `start`, `token`, `done`, and
`error` events over a `POST` request. The frontend uses `fetch` with
`ReadableStream`, not `EventSource`, because the stream is initiated by `POST`.

**Conversation deletion.** Stage 10a implements hard deletion for conversations.
Deleting a conversation removes dependent messages, episodes, embeddings, and
provenance rows through schema cascades. The UI requires confirmation and does
not delete active conversations while a stream is running.

## Stage Roadmap

Completed:

- Stage 1 through 4: schema, migrations, database client, scopes, embeddings,
  and memory service core
- Stage 5: scoped retrieval, scoring, access metadata, provenance groundwork,
  and minimal eval helper
- Stage 6: local FastAPI API layer
- Stage 7: assistant generation, local Ollama chat, provenance persistence, and
  SSE backend route
- Stage 8: React frontend foundation
- Stage 9a: non-streaming chat UI
- Stage 9b: SSE streaming chat UI
- Stage 10a: conversation deletion and cleanup UI
- Stage 10b: read-only retrieval inspection panel
- Stage 10c: explicit Ollama context window configuration
- Stage 11: fixed-window summary episodes
- Stage 12: Eval harness created
- Stage 13: Scope management UI beyond create and list
- Stage 14: Expanded retrieval eval harness

Next planned areas:
- Tune in the rolling summaries, eval harness and keep the system improving

## License

MIT License. See [LICENSE](/Users/kamal/Desktop/projects/smriti/LICENSE).

Copyright (c) 2026 Kamal Dhakal.
