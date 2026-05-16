# Stage 5: Retrieval and Eval Groundwork

This document defines the Stage 5 contract before retrieval code is written.

## 1. Scope

Stage 5 includes:

- scoped vector retrieval
- scoring composition
- retrieval provenance writes to `message_retrievals`
- synchronous reinforcement metadata updates
- groundwork for later eval harness

Stage 5 excludes:

- FastAPI
- MCP
- SSE
- frontend
- summarization
- memory rewriting
- agentic retrieval
- cross-scope retrieval
- hybrid BM25
- rerankers
- query rewriting
- caching layer
- full eval harness implementation unless explicitly approved later

## 2. Invariants

- Retrieval must filter by episodes.scope_id before vector similarity is applied.
- Global vector search followed by post-filtering is forbidden.
- Retrieval must require user_id and scope_id.
- Retrieval must return structured ScoredEpisode records, not list[str].
- Retrieval must update access_count and last_accessed_at synchronously for retrieved rows.
- message_retrievals records represent memories used for a generated assistant response, not every candidate considered.
- No retrieval path may bypass the memory service.

## 3. Important Distinction

"retrieved" means returned by the scoped retrieval method.

"used" means actually injected into a generated assistant response.

`access_count` and `last_accessed_at` update on retrieved rows. This is synchronous reinforcement metadata for the retrieval path itself.

`message_retrievals` should be written only for used rows, when the caller has a query/assistant message to attach provenance to. A retrieval call that returns candidates but does not inject them into an assistant response should not create provenance rows.

## 4. Non-Goals

- Summarization: deferred because Stage 5 retrieves existing episodes only, while summary episode creation belongs to the later rolling summarization stage.
- Memory rewriting: deferred because messages and episodes should remain append-oriented until explicit forget/export/rewrite semantics are designed.
- Reinforcement learning: deferred because Stage 5 only updates simple access metadata and does not train or adapt a model.
- Agentic retrieval: deferred because v1 retrieval should be a deterministic service operation, not a model-driven tool loop.
- Cross-scope retrieval: deferred because scope isolation is the product privacy boundary and cross-scope access must be an explicit later feature.
- Hybrid BM25: deferred because vector retrieval must be proven and tested before adding a second retrieval channel.
- Rerankers: deferred because reranking adds more model/runtime surface and should wait until baseline retrieval metrics exist.
- Query rewriting: deferred because rewritten queries can obscure provenance and should wait until the eval path can measure benefit.
- Caching: deferred because correctness, access metadata, and provenance semantics should be established before cached retrieval is introduced.
- FastAPI/MCP/SSE/frontend: deferred because Stage 5 should harden the memory service contract before exposing retrieval through external layers.

## 5. Open Decisions Before Implementation

### D1. Scoring Formula

Use the starting formula from `AGENTS.md`:

```text
0.55 similarity
0.20 recency
0.10 access reinforcement
0.10 importance
0.05 frequency
```

Recommendation: use named module-level constants for Stage 5 rather than runtime configuration. The weights should be easy to find and test, but configuration should wait until the eval harness can justify tuning.

### D2. Result Limiting

Recommendation: use `top_k` only for the initial Stage 5 retrieval contract. A score threshold should be deferred until there is enough eval data to choose a threshold without silently hiding useful memories.

Stage 5.2 accepts `candidate_limit = max(top_k * 5, 25)` as the first similarity-first candidate heuristic before Python reranking. This may miss some lower-similarity but high-recency or high-importance episodes until eval tuning improves candidate selection.

### D3. Provenance Write Trigger

Recommendation: confirm `message_retrievals` writes happen only when memories are used in assistant generation, not on every retrieval call. The explicit provenance method should require the query message, assistant message or response context if available, and the used `ScoredEpisode` rows.

### D4. Access Metadata Update

Recommendation: confirm `access_count` and `last_accessed_at` update for every returned retrieved row. Candidate rows considered by the database but not returned should not be updated.

### D5. Deterministic Ordering

Recommendation: use stable ordering by `score DESC`, `created_at DESC`, `episode_id ASC`. If similarity is exposed before full score composition, use the same shape with `similarity DESC`, `created_at DESC`, `episode_id ASC`.

## 6. Implementation Order

### Step 5.1: Implement Scoped Retrieval Only

- input: `user_id`, `scope_id`, query text/vector, `top_k`
- output: `list[ScoredEpisode]`
- strict scope filter
- no provenance writes yet

### Step 5.2: Add Scoring Composition

- similarity
- recency
- access
- importance
- frequency
- stable ordering

### Step 5.3: Add Synchronous Access Metadata Updates

Update `access_count` and `last_accessed_at` for returned rows in the same service-level retrieval flow.

### Step 5.4: Add Explicit Used-Retrieval Provenance Method

Add a dedicated memory service method to persist used retrievals into `message_retrievals`. This method should not be called by raw retrieval automatically.

### Step 5.5: Design Minimal Eval Harness

Only after real retrieval works, design the smallest useful eval harness around scoped retrieval quality and regression cases.

## 7. Testing Requirements

Stage 5 implementation must include tests that prove:

- retrieval never returns episodes from another scope
- wrong `user_id` cannot retrieve another user's episodes
- `top_k` is respected
- equal-score ordering is deterministic
- `access_count` / `last_accessed_at` update only for returned rows
- `message_retrievals` writes only through explicit used-memory provenance method
- no global post-filter retrieval shape is introduced
