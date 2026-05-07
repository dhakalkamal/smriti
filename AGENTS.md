# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code, Cursor, etc.) working in this repo. Read this fully before making changes.

## What this project is

smriti gives local LLMs long-term conversational memory. Privacy is a hard requirement, not a feature — design and code decisions must preserve the property that with the bundled UI, no user data leaves the machine.

The product is a **memory service** with two front doors:
1. An **MCP server** (FastMCP) for use with external clients.
2. A **bundled chat UI** (Streamlit) wired to local Ollama for the strict-privacy path.

Both share one Postgres + pgvector backend and one Python memory service. Do not duplicate memory logic across them.

The Python package is named `smriti`. All imports look like `from smriti.memory.service import ...`.

## Tech stack — non-negotiable

- **Python 3.11+**, async-first. Use `asyncio` and `asyncpg`. **Do not introduce SQLAlchemy or any ORM** — raw SQL is the source of truth.
- **Postgres 16 + pgvector** for storage and vector search. **HNSW indexes**, not IVFFlat.
- **Ollama** for embeddings and chat inference, on localhost. Default embedding: `nomic-embed-text` (768d).
- **FastMCP** for the MCP server.
- **Streamlit** for the bundled UI. Keep it under ~300 lines; it's a thin wrapper.
- **uv** for dependencies. **pytest** + **pytest-asyncio** for tests. **ruff** for lint+format. **mypy** strict for `memory/` and `db/`.

**Do not introduce:** LangChain, LlamaIndex, Chroma, Pinecone, Qdrant, or any cloud SDK. The point of this project is to write the retrieval code directly, not to wrap a framework.

If you think a different choice is better, raise it in the PR description — don't silently swap.

## Architectural rules

1. **The memory service is the core.** All memory operations — archive, retrieve, summarize, score — live in `src/smriti/memory/`. The MCP server, the chat orchestrator, and tests all call into this service. They do not talk to the database directly.

2. **Raw SQL, not ORMs.** Schema lives in `src/smriti/db/schema.sql` and `db/migrations/`. Queries live as parameterized strings near where they're used (or in `db/queries.py` if reused).

3. **Messages are immutable.** Once a message is written, it is never edited or deleted except by explicit user action via export/forget. Summaries are derived and can be regenerated. The `messages` table is the audit trail.

4. **No fabrication.** The local model is small and prone to hallucination. Do not introduce features where the model writes "facts" about the user that get treated as ground truth. Summaries are okay (they're labeled, the user can inspect source messages). Fact-extraction-as-memory is a v2 feature gated on having an eval harness that catches hallucination.

5. **Privacy is structural, not aspirational.** No telemetry, no analytics, no auto-update, no error reporting that includes user content. If you add a third-party dependency, check it doesn't phone home.

6. **Localhost only by default.** `docker-compose.yml` binds Postgres to `127.0.0.1:5432`, never `0.0.0.0`. Same for any HTTP server. CI should grep for `0.0.0.0` and fail.

7. **Embeddings table is partitioned by dim.** `embeddings_768`, `embeddings_1024`, etc. The `model_id` column discriminates within. pgvector indexes need fixed dims.

## Data model overview

The schema lives in `src/smriti/db/migrations/001_init.sql`. The key concept is **episodes**:

- A `message` is the raw user/assistant turn — immutable, append-only.
- An `episode` is the unit of *retrieval*. Two kinds:
  - `kind='message'`: one episode per message.
  - `kind='summary'`: one episode summarizing a contiguous range of messages.
- Embeddings reference episodes, not messages directly.

This unifies short messages and long summaries into one retrieval path. When a conversation gets long, oldest messages get summarized into a summary-kind episode, but the original message-kind episodes stay (for fine-grained retrieval).

## Retrieval scoring

Pure cosine similarity is wrong for chat memory. The scoring function combines:

- **Similarity** (cosine over embeddings) — the obvious signal.
- **Recency** (exponential decay on `created_at`) — newer episodes weigh more.
- **Reinforcement** (decay on `last_accessed_at`) — episodes the user revisits stay warm.
- **Importance** (user-pinnable, 0..1) — explicit override.
- **Frequency** (`log1p(access_count)`) — repeatedly-accessed memories surface more easily.

Starting weights: `0.55·sim + 0.20·rec + 0.10·acc + 0.10·imp + 0.05·freq`. These are guesses. The eval harness exists to find better ones.

## Code style

- Type hints everywhere. `from __future__ import annotations` at the top.
- Async functions for anything that touches the DB or network. Sync helpers fine for pure logic.
- Docstrings on public functions, especially in `memory/`. Explain *why*, not *what*.
- Prefer `dataclasses` or `pydantic` BaseModel. Don't pass dicts around.
- `logging`, not `print`. Never log message contents at INFO or below — DEBUG only, gated on a config flag.
- Names: verbs first, plural nouns. `archive_messages`, `retrieve_memories`. Don't abbreviate (`mem`, `conv`, `msg` are not okay).

## Testing requirements

- Every memory service function has a unit test against a real Postgres (use testcontainers). Mocked-DB tests are not enough — pgvector behavior matters.
- Don't test against Ollama directly. Use a `FakeEmbedder` that returns deterministic vectors based on a hash of the input.
- Real-Ollama tests live in `tests/integration/` and are opt-in.
- Eval-harness runs live in `tests/eval/`, run on demand, not in CI.

## Build order — follow this strictly

If picking up work and unsure where to start:

1. `db/schema.sql` + `db/migrations/001_init.sql` + `db/client.py` (asyncpg pool wrapper) + `db/migrate.py` (CLI runner).
2. `embeddings/base.py` (Embedder protocol) + `embeddings/ollama.py` + `embeddings/fake.py`.
3. `memory/service.py` core API: `archive_message`, `retrieve(query, k)`. No summarization yet. Single-component scoring (cosine only) for now.
4. `mcp/server.py` with two tools: `recall` and `remember`. Verify against Claude Desktop or Cline. **First demo-able milestone.**
5. `chat/orchestrator.py` + `chat/ollama_client.py` + `ui/streamlit_app.py` for end-to-end local flow. **First shippable milestone.**
6. `memory/scoring.py` — multi-component scoring as described above.
7. `memory/summarizer.py` — rolling summaries when conversation exceeds threshold.
8. `eval/` harness with synthetic conversations and known "needles."
9. Hybrid search (BM25 + vector + RRF). Re-run eval, measure delta.
10. Multi-model embedding comparison (add bge-m3 or mxbai-embed-large at 1024d).

Don't skip ahead. Each step works and is tested before moving on.

## What "done" looks like for a change

- `ruff check` and `ruff format --check` pass.
- `mypy` passes for memory/ and db/.
- `pytest tests/` passes.
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

## Things this project deliberately does NOT do

- **Agentic memory tools the LLM calls itself** (MemGPT-style). Local models are unreliable at tool-calling; auto-archiving is more robust.
- **Fact extraction as ground truth.** Local models hallucinate; storing fabricated "facts" about the user is harmful.
- **Cross-conversation retrieval by default.** Privacy bleed risk. Opt-in only via a flag.
- **Knowledge graphs.** v2+ work.
- **Document RAG.** This is a memory layer for chat, not a doc-search tool. If a single message contains a huge paste (e.g., a medical report), the orchestrator chunks it before persisting. That's the only place chunking happens.

## Repo layout (target)

```
smriti/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── .gitignore
├── src/smriti/
│   ├── config.py
│   ├── db/                    # schema.sql, migrations/, client.py, migrate.py
│   ├── memory/                # service.py, scoring.py, summarizer.py
│   ├── chat/                  # orchestrator.py, ollama_client.py
│   ├── embeddings/            # base.py, ollama.py, fake.py
│   ├── mcp/                   # server.py
│   ├── ui/                    # streamlit_app.py
│   └── eval/                  # dataset.py, metrics.py, runner.py
├── experiments/
│   └── chunking/              # chunking-strategy comparison sandbox
└── tests/
```

Build files into this layout as you go. Don't pre-create empty packages.
