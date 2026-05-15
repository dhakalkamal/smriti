# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code, Cursor, etc.) working in this repo. Read this fully before making changes.

## What this project is

smriti gives local LLMs long-term conversational memory, organized into user-defined **scopes**. Privacy is a hard requirement, not a feature — design and code decisions must preserve the property that with the bundled UI, no user data leaves the machine.

The product has three components, all running on localhost:

1. A **Python backend** combining the memory service, a FastAPI HTTP/SSE layer, and an MCP server (FastMCP). All three sit on top of one memory service.
2. A **React + TypeScript frontend** that talks to the FastAPI backend over HTTP/SSE.
3. A **Postgres + pgvector** database in Docker.

Plus **Ollama** running on localhost as a separate service for embeddings and chat inference.

The Python package is named `smriti`. All imports look like `from smriti.memory.service import ...`.


## Environment requirements (critical)

This project currently supports:

- Python 3.13.x
- uv
- macOS/Linux
- editable installs via Hatchling + src/ layout

Do not:
- use Python 3.14+
- change package manager
- flatten the src/ layout
- replace Hatchling editable installs
- switch away from uv without explicit instruction

Bootstrap:

```bash
uv sync --python 3.13 --extra dev
chflags -R nouchg,nohidden .venv
```

Verification:

```bash
uv run python -c "import smriti; print(smriti.__file__)"
uv run ruff check .
uv run ruff format --check .
uv run mypy src/smriti/db src/smriti/config.py
uv run pytest -q
```

## Core concept: scopes

A **scope** is a user-defined memory partition with its own system prompt. Examples: "Family Companion," "Research Notes," "Coding Helper." A user can have many scopes.

- A scope contains many conversations.
- A conversation contains many messages.
- Retrieval is scoped: when searching for relevant memories, the query is filtered by `scope_id` and only returns episodes from conversations within that scope.
- Cross-scope retrieval is **not** supported by default. It is an opt-in feature considered for v1.1+.

This is the privacy boundary the user controls. Marriage stuff stays in marriage. Work stuff stays in work. Whatever the user wants separate, stays separate.

The schema supports scopes; existing v1 code must enforce scope-filtering in every retrieval query.

## Tech stack — non-negotiable

**Backend:**
- **Python 3.11+**, async-first. `asyncio` + `asyncpg`. **No ORM** — raw SQL is the source of truth.
- **Postgres 16 + pgvector**, HNSW indexes (not IVFFlat).
- **Ollama** for embeddings (`nomic-embed-text`, 768d) and chat (`qwen2.5:7b` as default; user-configurable). localhost only.
- **FastAPI** for the HTTP/SSE layer.
- **FastMCP** for the MCP server.
- **uv** for dependencies. **pytest** + **pytest-asyncio** for tests. **ruff** lint+format. **mypy** strict for `memory/` and `db/`.

**Frontend:**
- **React 18+** with **TypeScript** (strict mode).
- **Vite** as the build tool.
- **Tailwind CSS** + **shadcn/ui** for styling and components.
- **Zustand** or **TanStack Query** for state, picked based on need. Don't add Redux.
- All assets vendored locally. **No Google Fonts, no CDN imports, no third-party analytics SDKs.**

**Do not introduce:** LangChain, LlamaIndex, Chroma, Pinecone, Qdrant, any cloud SDK, MUI/Chakra/Ant Design, Redux, socket.io. If you think a different choice is better, raise it in the PR description — don't silently swap.

## Architectural rules

1. **The memory service is the core.** All memory operations — archive, retrieve, summarize, score — live in `src/smriti/memory/`. The MCP server, FastAPI routes, and tests all call into this service. They do not talk to the database directly.

2. **Raw SQL, not ORMs.** Schema lives in `src/smriti/db/schema.sql` and `db/migrations/`. Queries live as parameterized strings near where they're used, or in `db/queries.py` if reused.

3. **Messages are immutable.** Once a message is written, it is never edited or deleted except by explicit user action via export/forget. Summaries are derived and can be regenerated. The `messages` table is the audit trail.

4. **No fabrication.** Local models are small and prone to hallucination. Do not introduce features where the model writes "facts" about the user that get treated as ground truth. Summaries are okay (labeled as summaries; source messages preserved). Fact-extraction-as-memory is v2+ gated on an eval harness that catches hallucination.

5. **Privacy is structural, not aspirational.** No telemetry, no analytics, no auto-update, no error reporting that includes user content. Any third-party dependency should be checked for phone-home behavior. This applies to npm packages as much as Python ones.

6. **Localhost only by default.** `docker-compose.yml` binds Postgres to `127.0.0.1:5432`, never `0.0.0.0`. FastAPI binds to `127.0.0.1:8000`. Vite dev server binds to `127.0.0.1:5173`. CI should grep for `0.0.0.0` and fail.

7. **Embeddings table is partitioned by dim.** `embeddings_768`, `embeddings_1024`, etc. The `model_id` column discriminates within. pgvector indexes need fixed dims.

8. **Browser security:** FastAPI must serve a strict Content Security Policy header that only allows connections to localhost. No `unsafe-inline`, no remote script sources, no remote connections.

## Data model overview

The schema lives in `src/smriti/db/migrations/`. Key tables:

- **`users`** — identity. Single-user setup has one row.
- **`scopes`** — user-defined memory partitions (added in migration 002).
- **`conversations`** — chat threads, each belonging to one scope.
- **`messages`** — raw user/assistant turns. Immutable, append-only.
- **`episodes`** — the retrieval unit. Either `kind='message'` (one episode per message) or `kind='summary'` (one episode summarizing a range of messages). Embeddings reference episodes, not messages directly.
- **`embedding_models`** — registry of which embedding models exist.
- **`embeddings_768`** — vectors, one row per (episode, model) pair.
- **`eval_*`** — eval harness scaffolding.

The episode design unifies short messages and long summaries into one retrieval path. When a conversation gets long, oldest messages get summarized into a summary-kind episode, but the original message-kind episodes stay (for fine-grained retrieval). Retrieval queries `episodes` without caring which kind.

## Retrieval scoring (v1)

Pure cosine similarity is wrong for chat memory. The scoring function combines:

- **Similarity** (cosine over embeddings) — the obvious signal.
- **Recency** (exponential decay on `created_at`) — newer episodes weigh more.
- **Reinforcement** (decay on `last_accessed_at`) — episodes the user revisits stay warm.
- **Importance** (user-pinnable, 0..1) — explicit override.
- **Frequency** (`log1p(access_count)`) — repeatedly-accessed memories surface more easily.

Starting weights: `0.55·sim + 0.20·rec + 0.10·acc + 0.10·imp + 0.05·freq`. These are guesses. The eval harness in v1.1 exists to find better ones.

## Code style

**Python:**
- Type hints everywhere. `from __future__ import annotations` at the top.
- Async functions for anything touching the DB or network. Sync helpers fine for pure logic.
- Docstrings on public functions, especially in `memory/`. Explain *why*, not *what*.
- Prefer `dataclasses` or `pydantic` BaseModel. Don't pass dicts around.
- `logging`, not `print`. Never log message contents at INFO or below — DEBUG only, gated on a config flag.
- Names: verbs first, plural nouns. `archive_messages`, `retrieve_memories`. Don't abbreviate (`mem`, `conv`, `msg` are not okay).

**TypeScript:**
- `strict: true` in `tsconfig.json`. No `any` without an explicit comment justifying it.
- Functional components only. No class components.
- Hooks for state and effects. Custom hooks for reusable logic.
- API types generated from FastAPI's OpenAPI schema where possible (use `openapi-typescript`).
- File naming: `PascalCase` for components, `camelCase` for hooks and utilities.

## Testing requirements

- Every memory service function has a unit test against a real Postgres (use testcontainers). Mocked-DB tests aren't enough — pgvector behavior matters.
- Don't test against Ollama directly. Use a `FakeEmbedder` returning deterministic vectors based on a hash of the input.
- Real-Ollama tests live in `tests/integration/` and are opt-in.
- Frontend: component tests with Vitest + React Testing Library. End-to-end happy-path test with Playwright (one test that does scope creation → send message → receive response).
- Eval-harness runs live in `tests/eval/`, run on demand, not in CI.

## Build order — follow this strictly

1. ✅ **Schema + db client + migration runner** (`db/`)
2. **Migration 002: scopes table + scope_id on conversations**
3. **Embeddings**: `Embedder` protocol, `OllamaEmbedder`, `FakeEmbedder`
4. **Memory service core**: `archive_message`, `retrieve(query, k, scope_id)`, multi-component scoring, `pin`, `forget`
5. **MCP server** with `recall`, `remember`, `list_scopes`, `forget`. Verify via Claude Desktop or Cline. **First demo-able milestone.**
6. **FastAPI layer**: all endpoints (chat, conversations, scopes, memories). No streaming yet.
7. **SSE streaming for chat**: `POST /api/conversations/{id}/messages` streams Ollama tokens.
8. **React scaffolding**: Vite, Tailwind, shadcn/ui, base layout, routing.
9. **React chat UI**: send message, render streaming response, message history.
10. **React scope + conversation management**: sidebar, create/rename/delete.
11. **Rolling summarization**: summarize old messages when conversation exceeds threshold.
12. **Eval harness**: synthetic conversations with known needles, recall@k and MRR. **End of v1.**

After v1 is shipped and used: v1.1 (hybrid search, contextual retrieval, reranking), measured against the v1 baseline.

Don't skip ahead. Each step works and is tested before moving on.

## API surface (FastAPI)

```
GET    /api/health
GET    /api/scopes
POST   /api/scopes
PATCH  /api/scopes/{id}
DELETE /api/scopes/{id}

GET    /api/scopes/{id}/conversations
POST   /api/scopes/{id}/conversations
GET    /api/conversations/{id}
DELETE /api/conversations/{id}

GET    /api/conversations/{id}/messages
POST   /api/conversations/{id}/messages      (SSE stream)

POST   /api/memories/search
POST   /api/memories/{id}/pin
DELETE /api/memories/{id}
```

## What "done" looks like for a change

- `ruff check` and `ruff format --check` pass.
- `mypy` passes for memory/ and db/.
- Backend: `pytest tests/` passes.
- Frontend: `npm run lint`, `npm run typecheck`, `npm test` all pass.
- New behavior has a test. Bugfixes have a regression test.
- README updated if user-facing.
- Migrations are append-only — never edit a committed migration; add a new one.

## Stop and ask before doing

- Adding a new dependency (especially one that makes network calls).
- Changing the schema in a way that loses data.
- Adding a feature where the LLM writes data that's later treated as ground truth.
- Exposing any service on `0.0.0.0` or a non-loopback interface.
- Adding telemetry, analytics, or any outbound network call not explicitly initiated by the user.
- Changing the privacy claims in the README.
- Adding a npm package that imports remote assets or fonts.

## Things this project deliberately does NOT do (v1)

- **Cross-scope retrieval.** Memories stay within their scope. v1.1 may add an opt-in flag.
- **Agentic memory tools the LLM calls itself** (MemGPT-style). Local models are unreliable at tool-calling; auto-archiving is more robust.
- **Fact extraction as ground truth.** Local models hallucinate; storing fabricated "facts" is harmful.
- **Knowledge graphs.** v3 territory, contingent on v2 results.
- **Document RAG.** This is a memory layer for chat, not a doc-search tool. If a single message contains a huge paste, the orchestrator chunks it before persisting. That's the only place chunking happens.
- **Multi-user accounts.** Schema supports it (the `users` table is real), but no auth, no signup, no separate user UI in v1. Single user assumed.
- **WebSockets for chat streaming.** Use SSE — simpler, sufficient, standard for LLM token streams. Revisit only if bidirectional features are needed.

## Repo layout (target)

```
smriti/
├── README.md
├── AGENTS.md
├── pyproject.toml                  # backend deps
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── src/smriti/                     # Python backend
│   ├── config.py
│   ├── db/                         # schema.sql, migrations/, client.py, migrate.py
│   ├── memory/                     # service.py, scoring.py, summarizer.py
│   ├── embeddings/                 # base.py, ollama.py, fake.py
│   ├── chat/                       # orchestrator.py, ollama_client.py
│   ├── api/                        # FastAPI app, routes, schemas
│   ├── mcp/                        # FastMCP server
│   ├── eval/                       # dataset.py, metrics.py, runner.py
│   └── cli.py                      # entry points: `smriti serve`, `smriti mcp`, `migrate`
│
├── frontend/                       # React app
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/
│       ├── components/
│       ├── hooks/
│       ├── state/
│       └── styles/
│
├── experiments/
│   └── chunking/                   # chunking-strategy sandbox (separate study)
│
└── tests/
    ├── unit/
    ├── integration/                # real Ollama, opt-in
    └── eval/                       # eval harness, opt-in
```

Build files into this layout as you go. Don't pre-create empty packages.