# Stage 6: FastAPI Local API

This document defines the Stage 6 contract before FastAPI code is written.

Stage 6 moves the local FastAPI API layer ahead of MCP in the build order. MCP is optional and deferred until after the core local product works.

## 1. Purpose

Stage 6 exposes the existing `MemoryService` through a localhost-only FastAPI API for the future React frontend.

The API is an adapter over the memory service, not a second memory implementation. FastAPI routes must preserve the current service guarantees around scoped retrieval, append-only messages, embedding creation, and access metadata updates.

## 2. Scope

Stage 6 includes:

- FastAPI app creation
- lifespan setup and teardown
- dependency injection for the `asyncpg` pool, `Embedder`, and `MemoryService`
- local-only API routes for existing memory service functions
- health endpoint
- scope endpoints
- conversation and message endpoints
- retrieval endpoint
- typed request and response schemas

Candidate route surface:

- `GET /health`
- `POST /api/v1/scopes`
- `GET /api/v1/scopes`
- `POST /api/v1/scopes/{scope_id}/conversations`
- `POST /api/v1/scopes/{scope_id}/conversations/{conversation_id}/messages`
- `POST /api/v1/scopes/{scope_id}/conversations/{conversation_id}/messages/{message_id}/episode`
- `POST /api/v1/scopes/{scope_id}/retrieve`

Routes should call `MemoryService` methods rather than issuing memory SQL directly. The API may own dependency setup, request validation, response shaping, and HTTP error mapping.

Stage 6 must not expose `MemoryService.record_used_memories` through HTTP. Provenance snapshots must be server-trustworthy and must not accept client-invented `ScoredEpisode` scores or score components. Provenance API wiring is deferred until the chat/SSE stage, where the server controls retrieval, prompt construction, and assistant response flow.

## 3. Non-Goals

Stage 6 excludes:

- MCP
- SSE
- frontend
- chat generation
- summarization
- cross-scope retrieval
- authentication or multi-user accounts
- cloud deployment
- public hosting
- telemetry
- analytics
- used-memory provenance API wiring
- `message_retrievals` writes from API retrieval

## 4. Privacy Invariants

- The FastAPI ASGI app itself does not bind sockets.
- The app run command and settings must default to `127.0.0.1`.
- The app run command and settings must never default to `0.0.0.0`.
- No telemetry may be added.
- No analytics may be added.
- No user data may flow outbound beyond local Ollama.
- CORS must be narrow and localhost-only.
- CORS must not use wildcard origins.
- No remote assets, remote scripts, remote fonts, or CDN imports may be added.
- API responses and logs must not leak message contents unnecessarily.

## 5. API Design Decisions Before Implementation

### D1. Single-User Identity

Question: how should Stage 6 choose `user_id` before real auth exists?

Decision: Stage 6 uses a bootstrapped local default user. Do not invent auth or multi-user accounts in Stage 6.

The API should pass the bootstrapped default `user_id` into every `MemoryService` call that requires it. The local user helper must not accept arbitrary user switching from frontend requests unless a later auth design explicitly approves it.

### D2. Pydantic Schemas

Question: where should API request and response models live?

Recommendation: place API request and response models in `src/smriti/api/schemas.py`.

Schemas should be typed Pydantic models that are specific to the HTTP boundary. They may wrap or mirror memory service dataclasses, but routes should not pass anonymous dictionaries across the service boundary.

### D3. App Structure

Question: where should app, router, and dependencies live?

Recommendation:

- `src/smriti/api/app.py`
- `src/smriti/api/dependencies.py`
- `src/smriti/api/routes.py`

`app.py` should own `create_app()` and lifespan wiring. `dependencies.py` should own dependency providers for settings, pool, embedder, memory service, and current local user. `routes.py` should own API route handlers and error registration if that remains the simplest structure.

### D4. Error Mapping

Question: how should `MemoryService` errors map to HTTP status codes?

Recommendation:

- `ScopeNotFoundError` -> `404 Not Found`
- `ConversationNotFoundError` -> `404 Not Found`
- `InvalidRetrievalRequestError` -> `400 Bad Request`
- `EmbeddingModelNotFoundError` -> `500 Internal Server Error`
- `VectorDimensionError` -> `500 Internal Server Error`
- unknown `MemoryServiceError` -> `500 Internal Server Error`
- local embedder connection, timeout, or invalid/failed response errors -> `503 Service Unavailable`
- embedder configuration, model registry, vector dimension, or schema errors -> `500 Internal Server Error`

Validation errors raised by Pydantic should use FastAPI's normal `422 Unprocessable Entity` response. Not-found responses must be generic enough that they do not reveal whether a resource exists under a different user or scope.

### D5. CORS

Question: which origins are allowed?

Recommendation:

- `http://127.0.0.1:5173`
- `http://localhost:5173`

CORS should be configured explicitly for the local Vite frontend only. Do not use `*`, broad localhost port ranges, or external origins.

### D6. Retrieval Side Effects

Question: should API retrieval preserve memory service retrieval side effects?

Recommendation: yes. The retrieval endpoint may expose `MemoryService.retrieve_scoped_episodes` and therefore preserve access metadata updates for returned rows, including `access_count` and `last_accessed_at`.

The API must not implement a read-only retrieval shortcut, direct vector SQL path, or global search followed by post-filtering. Stage 6 retrieval must not write `message_retrievals`.

Used-memory provenance remains explicit but is not exposed through HTTP in Stage 6. Provenance API wiring is deferred until the chat/SSE stage, where the server can pass trustworthy retrieval result snapshots into `MemoryService.record_used_memories`.

## 6. Testing Requirements

Stage 6 implementation must include tests that prove:

- the app creates without network exposure
- the app run command and settings default to `127.0.0.1`, not `0.0.0.0`
- the health endpoint works
- scope creation works
- scope listing works
- conversation creation works
- message append plus episode embedding flow works
- the retrieval endpoint respects scope isolation
- the retrieval endpoint preserves `retrieve_scoped_episodes` access metadata updates
- the retrieval endpoint does not write `message_retrievals`
- error mapping works for invalid scope, user, message, and retrieval cases
- local embedder connection and timeout errors map to `503 Service Unavailable`
- embedder configuration and invalid response errors map to `500 Internal Server Error`
- CORS allows only the approved localhost origins
- API route modules do not issue raw SQL or import `asyncpg` directly, except dependency and lifespan wiring modules
- no MCP implementation is added
- no SSE implementation is added
- no frontend implementation is added

Tests should use deterministic embedding behavior through `FakeEmbedder` where possible. Real Ollama calls remain opt-in and should not be required for default API tests.

## 7. Implementation Order

### Step 6.1: Create API App and Dependency Structure Only

- add `src/smriti/api/`
- add `create_app()` without feature routes beyond wiring
- add lifespan setup and teardown for local dependencies
- add dependency providers for pool, embedder, memory service, and local user
- add no MCP, SSE, or frontend code

### Step 6.2: Add Health and Scope Endpoints

- add `GET /health`
- add scope creation
- add scope listing
- map service errors to HTTP responses
- add focused tests for these endpoints

### Step 6.3: Add Conversation, Message, and Episode Endpoints

- add conversation creation
- add message append
- add message episode creation and embedding
- preserve append-only message semantics
- add focused tests for the message-to-episode flow

### Step 6.4: Add Retrieval Endpoint

- add scoped retrieval
- call `MemoryService.retrieve_scoped_episodes`
- preserve scope isolation and retrieval side effects
- do not write `message_retrievals`
- do not expose `MemoryService.record_used_memories`
- add focused tests for isolation, access metadata updates, and no provenance writes

### Step 6.5: Add API Tests

- finish route-level tests
- finish error mapping tests
- finish localhost and CORS privacy tests
- confirm no MCP, SSE, or frontend code was introduced by Stage 6
