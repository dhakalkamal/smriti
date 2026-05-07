# smriti

Long-term memory for local LLMs. Have hour-long conversations with deepseek, qwen, llama, or any Ollama model — without losing context, and without your data leaving your machine.

*smriti* (स्मृति) is Sanskrit for "memory" or "remembrance."

> **Status:** early development.

## Why this exists

Local LLMs run great on Apple Silicon, but their effective context window is small. Once a chat passes ~8k–32k tokens, the model starts forgetting. People who care about privacy — therapy-adjacent conversations, health questions, family matters — can't comfortably use ChatGPT or Claude for these topics, but local models lose the thread within an hour.

smriti sits between you and a local LLM. It archives older messages, embeds them, and pulls the relevant ones back into context when you need them. The conversation feels long. Your data stays on your machine.

## Privacy model

**With the bundled chat UI + Ollama:** nothing leaves your machine. Messages, embeddings, summaries — all local.

**With external MCP clients (Claude Desktop, ChatGPT Desktop, Cline-with-cloud-models):** the MCP server itself is local, but those clients send your messages — and any memories retrieved on your behalf — to their respective cloud providers. Use the bundled UI for the strict privacy guarantee.

No telemetry, no analytics, no auto-update. Postgres binds to localhost. Ollama binds to localhost. The only outbound network call the system makes is when you explicitly point it at a remote model.

## Architecture

```
┌────────────────────────────┐    ┌────────────────────────────┐
│  Bundled chat UI           │    │  External MCP client       │
│  (Streamlit, local-only)   │    │  (Claude Desktop, etc.)    │
└─────────────┬──────────────┘    └─────────────┬──────────────┘
              │ HTTP                            │ MCP (stdio)
              ▼                                 ▼
┌─────────────────────────┐    ┌────────────────────────────────┐
│  Chat orchestrator      │    │  MCP server (FastMCP)          │
│  (context management,   │    │  Tools: recall, remember,      │
│   Ollama client)        │    │  list_sessions, export, forget │
└─────────────┬───────────┘    └─────────────┬──────────────────┘
              └──────────────┬───────────────┘
                             ▼
              ┌────────────────────────────┐
              │  Memory service            │
              │  archive · retrieve ·      │
              │  summarize · score         │
              └─────────────┬──────────────┘
                ┌───────────┴───────────┐
                ▼                       ▼
         ┌──────────────┐        ┌──────────────┐
         │  Postgres +  │        │   Ollama     │
         │  pgvector    │        │  (embeddings │
         │              │        │  + chat)     │
         └──────────────┘        └──────────────┘
```

The memory service is the core. The MCP server and the chat orchestrator are both thin wrappers over it. One memory implementation, two ways to use it.

## Tech stack

- Python 3.11+, async (asyncpg, httpx)
- Postgres 16 + pgvector (HNSW indexes)
- Ollama for embeddings and chat (default embedding: nomic-embed-text, 768d)
- FastMCP for the MCP server
- Streamlit for the bundled UI

## Quick start (once implemented)

```bash
docker compose up -d
uv sync
uv run python -m smriti.db.migrate up
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# Bundled UI (private):
uv run streamlit run src/smriti/ui/streamlit_app.py

# OR MCP server (for external clients):
uv run python -m smriti.mcp.server
```

## Roadmap

- [ ] Schema + migrations + asyncpg client
- [ ] Ollama embedder + fake embedder for tests
- [ ] Memory service (archive + retrieve)
- [ ] MCP server with `recall` + `remember`
- [ ] Chat orchestrator + Streamlit UI
- [ ] Rolling summarization
- [ ] Eval harness with synthetic conversations
- [ ] Hybrid search (BM25 + vector + RRF)
- [ ] Multi-model embedding comparison
- [ ] Optional: fact extraction (post-v1)

## License

TBD.
