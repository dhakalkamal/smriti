# smriti

Long-memory companions you can shape to your purpose. Local LLMs on your machine, with memory that spans months instead of minutes.

*smriti* (स्मृति) is Sanskrit for "memory" or "remembrance."

> **Status:** early development.

## What it is

Local LLMs run well on Apple Silicon, but their effective context window is small. Once a chat passes ~8k–32k tokens, the model forgets. You can't have a long-running conversation with a local model the way you can with ChatGPT or Claude.

smriti fixes that by giving local LLMs a long-term memory layer. You define **scopes** — bounded memory spaces for whatever purpose you choose. A scope might be a journaling companion, a counselor persona, a research assistant, a coding partner. Each scope has its own system prompt and its own memory pool. Conversations within a scope share memory; conversations across scopes never bleed into each other.

The bot remembers what you said weeks ago. It surfaces relevant past moments when you ask about them. It can notice patterns over time. And none of your data ever leaves your machine.

## Privacy model

This is the part to read carefully.

**With the bundled chat UI + Ollama:** nothing leaves your machine. Messages, embeddings, summaries — all local. You can verify with `lsof -i` or Little Snitch.

**With external MCP clients (Claude Desktop, ChatGPT Desktop, Cline-with-cloud-models):** the MCP server itself is local, but those clients send your messages — and any memories retrieved on your behalf — to their respective cloud providers. This is fine if you trust them, but it is *not* the local-only mode. Use the bundled UI for the strict privacy guarantee.

No telemetry, no analytics, no auto-update, no account, no signup. Postgres binds to localhost. Ollama binds to localhost. The only outbound network calls the system makes are the ones *you* explicitly configure.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       User's Mac (M-series)                      │
│                                                                  │
│  ┌──────────────────┐         ┌─────────────────────────────┐   │
│  │ Browser          │         │ External MCP client          │   │
│  │ React app        │         │ (Claude Desktop, Cline, etc.)│   │
│  │ localhost:5173   │         │                              │   │
│  └────────┬─────────┘         └─────────────┬────────────────┘   │
│           │ HTTP/SSE                        │ MCP (stdio)        │
│           ▼                                 ▼                    │
│  ┌──────────────────┐         ┌─────────────────────────────┐   │
│  │ FastAPI          │         │ FastMCP server               │   │
│  │ localhost:8000   │         │                              │   │
│  │ + chat           │         │ Tools: recall, remember,     │   │
│  │   orchestrator   │         │ list_scopes, forget          │   │
│  └────────┬─────────┘         └─────────────┬────────────────┘   │
│           │                                 │                    │
│           └───────────────┬─────────────────┘                    │
│                           ▼                                      │
│           ┌───────────────────────────────────┐                  │
│           │   Memory service (Python)         │                  │
│           │   archive · retrieve · summarize  │                  │
│           │   score (multi-component)         │                  │
│           └───────────────┬───────────────────┘                  │
│                           │                                      │
│             ┌─────────────┴─────────────┐                        │
│             ▼                           ▼                        │
│   ┌──────────────────┐         ┌──────────────────┐              │
│   │ Postgres 16 +    │         │ Ollama           │              │
│   │ pgvector         │         │ localhost:11434  │              │
│   │ 127.0.0.1:5432   │         │                  │              │
│   │ (Docker)         │         │ - embeddings     │              │
│   └──────────────────┘         │ - chat           │              │
│                                └──────────────────┘              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

         No network egress. All boxes are on localhost.
```

The memory service is the core. FastAPI and the MCP server are both thin wrappers over it. One memory implementation, two ways to use it.

## Tech stack

**Backend:**
- Python 3.11+, async-first. The project environment is pinned to CPython 3.13; Python 3.14 is excluded until uv editable installs on macOS no longer hide `.pth` files from Python's startup checks.
- Postgres 16 + pgvector (HNSW indexes)
- asyncpg for database access (no ORM)
- Ollama for embeddings (`nomic-embed-text`, 768d) and chat (`qwen2.5:7b` by default)
- FastAPI for the HTTP/SSE layer
- FastMCP for the MCP server
- uv for dependency management

**Frontend:**
- React + TypeScript
- Vite as the build tool
- Tailwind + shadcn/ui for styling
- All assets vendored locally — no Google Fonts, no CDNs

## Quick start (once implemented)

```bash
# Bring up Postgres
docker compose up -d

# Install backend deps and apply migrations
uv sync --python 3.13 --extra dev
uv run migrate up

# Pull local models
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# Run backend (one terminal)
uv run smriti serve

# Run frontend (another terminal)
cd frontend
npm install
npm run dev

# Open browser at http://localhost:5173
```

## Concepts

**Scopes.** A scope is a user-defined memory partition with its own system prompt. You might create a "Family Companion" scope, a "Research Notes" scope, a "Coding Helper" scope. The bot only retrieves memories from within the current scope — never across scopes.

**Conversations.** A conversation is a single chat thread within a scope. You can have many concurrent threads in the same scope; they share memory.

**Messages.** Individual turns. Stored immutably — the audit trail.

**Episodes.** The unit of retrieval. Either an individual message or a summary of a range of older messages. Both kinds coexist in one table; retrieval searches both transparently.

**Memory retrieval.** When you send a message, smriti embeds it, searches the current scope's episodes for relevant matches, scores them by similarity + recency + importance + reinforcement + frequency, and injects the top results into the prompt alongside your recent live messages.

## Roadmap

**v1 — basic working memory**
- [x] Schema, migrations, db client
- [ ] Scopes (migration 002)
- [ ] Ollama embedder, fake embedder for tests
- [ ] Memory service: archive, retrieve, score
- [ ] MCP server with `recall` + `remember`
- [ ] FastAPI layer with SSE streaming
- [ ] React chat UI with scope and conversation management
- [ ] Rolling summarization

**v1.1 — confident upgrades (measure before/after with eval harness)**
- [ ] Eval harness with synthetic conversations
- [ ] Hybrid search (BM25 + vector + RRF)
- [ ] Contextual retrieval (Anthropic-style prefix embedding)
- [ ] Cross-encoder reranking (local model)

**v2 — pattern detection and richer memory**
- [ ] Batch clustering of embeddings → generated insight episodes
- [ ] Time-aware retrieval
- [ ] Query rewriting with the local LLM
- [ ] Multi-model embedding comparison

**v3 — experimental, contingent on v2 results**
- [ ] Lightweight knowledge graph for entity resolution
- [ ] Agentic retrieval (model issues follow-up queries)

## License

TBD.
