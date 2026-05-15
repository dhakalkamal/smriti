# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code, Cursor, etc.) working in this repo. Read this fully before making changes.

## What this project is

smriti gives local LLMs long-term conversational memory, organized into user-defined **scopes**. Privacy is a hard requirement, not a feature. Design and code decisions must preserve the property that, with the bundled UI, no user data leaves the machine.

The product has three components, all running on localhost:

1. A **Python backend** combining the memory service, a FastAPI HTTP/SSE layer, and an MCP server using FastMCP. All three sit on top of one memory service.
2. A **React + TypeScript frontend** that talks to the FastAPI backend over HTTP/SSE.
3. A **Postgres + pgvector** database in Docker.

Plus **Ollama** running on localhost as a separate service for embeddings and chat inference.

The Python package is named `smriti`. All imports look like:

```python
from smriti.memory.service import ...
```

---

## Environment requirements

This project currently supports:

- Python 3.13.x
- uv
- macOS/Linux
- editable installs via Hatchling + `src/` layout

Do not:

- use Python 3.14+
- change package manager
- flatten the `src/` layout
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

If editable imports fail on macOS:

- inspect `.pth` hidden flags
- inspect `.venv` directory flags
- do not refactor packaging before checking filesystem flags

---

## Debugging philosophy

When debugging:

1. Verify assumptions before refactoring.
2. Prefer minimal diffs.
3. Environment/tooling bugs should not trigger architectural changes.
4. Inspect Python version, `sys.path`, editable install state, `.pth` files, macOS file flags, and uv environment state before changing package structure.

---

## Change scope discipline

Unless explicitly requested:

- avoid broad refactors
- avoid renaming modules
- avoid moving files
- avoid changing APIs
- avoid introducing abstractions “for future flexibility”

Prefer small, verifiable diffs.

---

## Core concept: scopes

A **scope** is a user-defined memory partition with its own system prompt.

Examples:

- Family Companion
- Research Notes
- Coding Helper

A user can have many scopes.

- A scope contains many conversations.
- A conversation contains many messages.
- Retrieval is scoped: when searching for relevant memories, the query is filtered by `scope_id` and only returns episodes from conversations within that scope.
- Cross-scope retrieval is not supported by default. It is an opt-in feature considered for v1.1+.

This is the privacy boundary the user controls.

The schema supports scopes. Existing v1 code must enforce scope-filtering in every retrieval query.

---

## Tech stack

### Backend

- Python 3.13.x, async-first
- `asyncio` + `asyncpg`
- No ORM. Raw SQL is the source of truth.
- Postgres 16 + pgvector
- HNSW indexes, not IVFFlat
- Ollama for embeddings using `nomic-embed-text`, 768d
- Chat model: `qwen2.5:7b` by default
- FastAPI for HTTP/SSE
- FastMCP for MCP server
- uv for dependency management
- pytest + pytest-asyncio
- ruff for lint and format
- mypy strict for `memory/` and `db/`

### Frontend

- React 18+
- TypeScript strict mode
- Vite
- Tailwind CSS
- shadcn/ui
- Zustand or TanStack Query
- No Redux
- All assets vendored locally
- No Google Fonts
- No CDN imports
- No analytics SDKs

### Do not introduce

- LangChain
- LlamaIndex
- Chroma
- Pinecone
- Qdrant
- cloud SDKs
- MUI
- Chakra
- Ant Design
- Redux
- socket.io

If you think another choice is better, explain it in the PR description. Do not silently replace core tooling.

---

## Architectural rules

1. **The memory service is the core.**

   All memory operations live in:

   ```text
   src/smriti/memory/
   ```

   The MCP server, FastAPI routes, and tests call into the memory service. They do not talk directly to the database.

2. **Raw SQL, not ORMs.**

   Schema lives in:

   ```text
   src/smriti/db/schema.sql
   src/smriti/db/migrations/
   ```

3. **Messages are immutable.**

   Messages are append-only unless explicitly forgotten/exported by the user.

4. **No fabrication.**

   Do not store hallucinated “facts” as truth.

5. **Privacy is structural.**

   No telemetry, analytics, auto-update, or hidden outbound network behavior.

6. **Localhost only by default.**

   - Postgres: `127.0.0.1:5432`
   - FastAPI: `127.0.0.1:8000`
   - Vite: `127.0.0.1:5173`

   Never expose services to `0.0.0.0` unless explicitly requested.

7. **Embeddings are partitioned by dimension.**

   Examples:

   ```text
   embeddings_768
   embeddings_1024
   ```

8. **Strict browser security.**

   No remote scripts, unsafe CSP, or external assets.

---

## Data model overview

The schema lives in `src/smriti/db/migrations/`.

Key tables:

- `users`: identity. Single-user setup has one row.
- `scopes`: user-defined memory partitions.
- `conversations`: chat threads, each belonging to one scope.
- `messages`: raw user/assistant turns. Immutable and append-only.
- `episodes`: the retrieval unit. Either `kind='message'` or `kind='summary'`.
- `embedding_models`: registry of embedding models.
- `embeddings_768`: vectors, one row per episode/model pair.
- `eval_*`: eval harness scaffolding.

The episode design unifies short messages and long summaries into one retrieval path.

---

## Retrieval scoring

Pure cosine similarity is not enough for chat memory. The scoring function combines:

- similarity
- recency
- reinforcement from `last_accessed_at`
- user-pinned importance
- access frequency

Starting weights:

```text
0.55 * similarity
+ 0.20 * recency
+ 0.10 * access reinforcement
+ 0.10 * importance
+ 0.05 * frequency
```

These are starting guesses. The eval harness should be used to tune them.

---

## Code style

### Python

- Type hints everywhere.
- Use `from __future__ import annotations`.
- Async functions for DB/network operations.
- Sync helpers are fine for pure logic.
- Docstrings on public functions, especially in `memory/`.
- Explain why, not just what.
- Prefer dataclasses or Pydantic models.
- Do not pass anonymous dictionaries across service boundaries.
- Use logging, not print.
- Never log message contents at INFO or below.
- Do not abbreviate names like `mem`, `conv`, or `msg`.

### TypeScript

- `strict: true` in `tsconfig.json`.
- No `any` without a comment explaining why.
- Functional components only.
- Hooks for state and effects.
- Custom hooks for reusable logic.
- Prefer API types generated from FastAPI OpenAPI schema.
- `PascalCase` for components.
- `camelCase` for hooks and utilities.

---

## Testing requirements

- Memory service functions require tests against real Postgres.
- Use `FakeEmbedder` for deterministic embedding tests.
- Real Ollama tests are opt-in.
- Frontend uses Vitest + React Testing Library.
- End-to-end tests use Playwright.
- Eval harness lives in `tests/eval/` and runs on demand, not by default in CI.

---

## Build order

Follow this strictly:

1. Schema + db client + migration runner
2. Migration 002: scopes table + `scope_id`
3. Embeddings: `Embedder` protocol, `OllamaEmbedder`, `FakeEmbedder`
4. Memory service core
5. MCP server
6. FastAPI layer
7. SSE streaming
8. React scaffolding
9. React chat UI
10. Scope management UI
11. Rolling summarization
12. Eval harness

Do not skip ahead.

---

## What done means

Before finishing a task, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/smriti/db src/smriti/config.py
uv run pytest -q
```

Requirements:

- tests pass
- lint passes
- typing passes
- new behavior has tests
- migrations are append-only
- README updated if user-facing

---

## Stop and ask before

- adding dependencies
- changing schema destructively
- introducing telemetry
- exposing services publicly
- changing privacy guarantees
- adding remote assets, fonts, or scripts
- broad architectural refactors

---

## Things this project deliberately does not do in v1

- Cross-scope retrieval
- Agentic memory tools called directly by the LLM
- Fact extraction as ground truth
- Knowledge graphs
- Document RAG
- Multi-user accounts
- WebSockets for chat streaming

Use SSE for token streaming.

---

## Repo layout

```text
smriti/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── src/smriti/
├── frontend/
├── tests/
└── experiments/
```

Build incrementally. Do not pre-create empty packages.