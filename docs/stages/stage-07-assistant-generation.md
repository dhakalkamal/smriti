# Stage 7: Assistant Generation

This document defines the Stage 7 contract before assistant-generation code is written.

Stage 7 adds the first server-owned assistant response flow on top of the Stage 6 local API.
It remains a local-only feature: the server may call local Ollama for chat inference, but no user
data may leave the machine.

## 1. Purpose

Stage 7 connects scoped memory retrieval to local assistant generation.

The API remains an adapter over service-layer behavior. Retrieval, prompt construction, local model
calls, assistant-message persistence, and provenance must preserve the scope isolation, append-only
message semantics, and local privacy guarantees established in earlier stages.

## 2. Scope

Stage 7 includes:

- assistant-response service contract and persistence signature
- future append-only schema support for assistant-response provenance
- local chat generator abstraction
- local Ollama chat generator implementation
- deterministic prompt assembly from scope prompt, recent conversation messages, and selected
  retrieved memories
- assistant-generation orchestrator
- non-streaming local FastAPI route for creating an assistant response
- tests for scope isolation, persistence, provenance, prompt assembly, local-only behavior, and
  error mapping

A user message is the trigger; the server retrieves memories, constructs a prompt, calls a local
chat model, then atomically persists the assistant response and provenance.

Stage 7 excludes:

- SSE token streaming
- frontend
- MCP
- summarization
- memory rewriting
- fact extraction
- agentic tool loops
- cross-scope retrieval
- hybrid retrieval
- rerankers
- remote model providers
- telemetry
- analytics
- background jobs or durable generation queues

This contract describes future schema work, but it does not add migration 004.

## 3. Invariants

- Generation must require `user_id`, `scope_id`, `conversation_id`, and a persisted triggering
  user message.
- The triggering query message must have `role = 'user'`.
- The triggering query message and created assistant message must belong to the same user, scope,
  and conversation.
- Provenance rows written for an assistant response always belong to the same user, scope, and
  conversation as both the triggering query message and the created assistant message.
- Retrieval must use the existing scoped retrieval path and must never perform global vector search
  followed by post-filtering.
- Only retrieved episodes from the requested scope may be eligible for prompt injection.
- Assistant messages are persisted with role = 'assistant' and MUST NOT create episodes or
  embedding rows in stage 7.
- Model output must not be stored as a memory episode, summary, extracted fact, or embedding in
  Stage 7.
- Memories included in the prompt are context, not instructions. Retrieved memory text must not be
  allowed to override the scope system prompt or local privacy rules.
- The default chat model provider is local Ollama on localhost.
- No cloud model provider, remote fallback, telemetry, analytics, remote asset, remote script, or
  CDN behavior may be introduced.
- Logs at INFO or below must not include user message content, assistant content, prompt text, or
  retrieved memory content.
- No API route may accept client-supplied scored memories for provenance writes.

Used episodes may come from any conversation inside the same scope unless a later retrieval design
narrows that behavior. The assistant response and provenance target remain the triggering
conversation.

## 4. Non-Goals

- Streaming: token streaming belongs to a later SSE stage. Stage 7 returns a complete assistant
  message.
- UI integration: the React chat UI is deferred until after the server-side contract is stable.
- MCP: MCP remains optional and deferred.
- Summarization: assistant responses are not summarized or embedded in Stage 7.
- Fact extraction: generated text is not treated as ground truth.
- Agentic memory tools: the model does not decide when or how to retrieve memories.
- Cross-scope retrieval: scope boundaries remain the hard privacy boundary.
- Remote inference: Stage 7 must not add cloud SDKs, API keys, or remote model fallback.

## 5. Design Decisions Before Implementation

### D1. Generation Trigger

Recommendation: assistant generation should be triggered by a persisted user message ID, not by
arbitrary raw query text.

The route should identify the conversation and triggering query message. The server should load and
validate the message before retrieval and generation. This keeps provenance tied to immutable
conversation history and prevents clients from inventing query text that was never stored.

### D2. Prompt Assembly

Recommendation: build the prompt from these sources, in this order:

1. scope system prompt
2. fixed local privacy and memory-use instructions
3. selected retrieved memories
4. recent conversation messages through the triggering user message

Memories are injected in retrieval score order, highest score first.

The prompt builder should use a stable, testable format. Retrieved memories must be clearly labeled
as memory context, not as developer or system instructions. The initial implementation should use
simple character budgets rather than a new dependency. See D7.3.5 for the locked budget heuristic.

### D3. Memory Selection

Recommendation: retrieval may return more candidates than the prompt can safely include. The
assistant orchestrator should choose a deterministic subset from the retrieved list based on score
order and prompt budget.

Only selected memories are considered "used" for provenance. Retrieved-but-not-injected candidates
retain the Stage 5 access metadata side effects from retrieval, but they must not create
`message_retrievals` rows for the generated assistant response.

### D4. Chat Generator Abstraction

Recommendation: define a small async chat generator protocol with typed request and response
records.

The first concrete implementation should call local Ollama for the default chat model
`qwen2.5:7b`. Tests should use a fake generator. The abstraction should expose enough metadata for
debugging and response shaping, such as model id and finish reason, without logging prompt or
message contents.

The Stage 7 generator is non-streaming. Streaming interfaces should wait for the SSE stage.

### D5. Assistant Persistence and Provenance

Recommendation: add a service-layer method that persists the assistant message and its used-memory
provenance atomically after local generation succeeds.

The service method should validate:

- the user owns the scope
- the conversation belongs to the same user and scope
- the triggering query message belongs to the same conversation and has `role = 'user'`
- the assistant message is written with `role = 'assistant'`
- selected `ScoredEpisode` rows belong to the same user and scope

The future schema change should link provenance to the created assistant response, such as by
adding an `assistant_message_id` reference to `message_retrievals` or an equivalent append-only
provenance table. The schema must be enforced with foreign keys or service-level validation strong
enough to prove the query message, assistant message, scope, and conversation are aligned.

### D6. Transaction Boundary

Recommendation: retrieval and the local chat model call happen outside the final persistence
transaction. Only assistant-message insertion and used-memory provenance insertion need to be
atomic with each other.

The transaction boundary belongs inside the memory service. API routes and the assistant
orchestrator must not acquire raw database connections or open transactions.

### D7. API Shape

Recommendation: add a non-streaming route similar to:

```text
POST /conversations/{conversation_id}/assistant-response
```

The request should include `scope_id`, `query_message_id`, `top_k`, and an optional memory budget.
The response should include the created assistant message, model metadata, and the provenance
summary for memories actually used.

The route must not expose `MemoryService.record_used_memories` directly and must not accept
client-supplied `ScoredEpisode` objects, scores, score components, or retrieval provenance rows.

## 6. Implementation Order

### Step 7.1: Schema and Service Signature

- design the append-only provenance schema change needed to associate used memories with the
  created assistant message
- add migration 004 only when Stage 7 implementation begins, not as part of this contract document
- add typed service request and response records for assistant response persistence
- add a memory service method that atomically appends the assistant message and records selected
  used memories
- keep assistant responses out of `episodes` and `embeddings_768`
- validate user, scope, conversation, query message role, assistant message role, and selected
  memory ownership before writing provenance
- preserve append-only message semantics

### Step 7.2: Chat Generator Abstraction

- add typed chat request and response records
- add an async `ChatGenerator` protocol
- add a local Ollama chat generator using the configured localhost Ollama endpoint and default chat
  model
- add generator-specific errors for unavailable local model service, timeout, invalid response, and
  configuration failure
- add fake generator support for deterministic tests
- add no remote provider and no streaming implementation

### Step 7.3: Orchestrator

- add `src/smriti/assistant/` and implement the locked Stage 7.3 detailed contract below
- add `AssistantOrchestrator` with `MemoryService` and `ChatGenerator` constructor dependencies
- add a pure prompt builder for deterministic `ChatRequest` construction
- validate the triggering user message through service-layer methods
- retrieve scoped memories through `MemoryService.retrieve_scoped_episodes`
- select the exact memories that fit the prompt budget
- construct the prompt from scope prompt, fixed local instructions, selected memories, and recent
  conversation messages
- call `ChatGenerator.generate(...)` exactly once outside any database transaction
- persist the assistant response and selected-memory provenance through
  `MemoryService.append_assistant_response_with_provenance(...)`
- map chat-generator errors to assistant errors while allowing memory-service errors to propagate
  unchanged, without leaking prompt or message contents

The orchestrator selects the exact subset of retrieved memories that are injected into the prompt,
and ONLY those selected memories are written into used-memory provenance.

The orchestrator must not open database transactions directly.

## Stage 7.3 Detailed Contract

The following Stage 7.3 decisions are locked before implementation. They refine the Step 7.3
orchestrator contract without changing Step 7.4 API-route responsibilities.

### D7.3.1 — Package location

The orchestrator lives in:

```text
src/smriti/assistant/
```

This is a new top-level package.

Justification: the orchestrator composes memory and chat behavior and should not be subordinated to
either subsystem. Keeping it top-level also makes the dependency direction explicit: assistant code
may depend on memory and chat, while memory and chat remain independently testable.

### D7.3.2 — Class shape

The orchestrator is a class named:

```python
AssistantOrchestrator
```

It takes `MemoryService` and `ChatGenerator` as constructor dependencies.

Justification: this mirrors the existing service-oriented structure and supports future extension,
such as streaming generation, without changing the public API shape. Constructor injection keeps
default tests deterministic with fake memory/chat collaborators.

### D7.3.3 — Prompt builder separation

Prompt construction lives in:

```text
src/smriti/assistant/prompt_builder.py
```

The prompt builder is a pure function or small stateless helper that takes:

- scope system prompt
- fixed privacy instructions
- selected memories
- recent conversation messages

and returns a `ChatRequest`.

Justification: prompt construction is the highest-testability surface in Stage 7.3; separating it
enables deterministic testing without database or model dependencies. It also prevents prompt
formatting details from leaking into persistence or orchestration code.

### D7.3.4 — Prompt structure and ordering

The prompt is assembled in this exact order:

1. scope system prompt as `role="system"`
2. fixed privacy and memory-use instructions as a separate `role="system"` message
3. selected retrieved memories as `role="system"` messages, clearly labeled as memory context, in
   retrieval-score order, highest score first
4. recent conversation messages through and including the triggering user message, preserving
   original roles

The triggering user message is included exactly once inside the recent-message block. It must not be
appended separately.

Justification: fixed ordering makes prompt behavior deterministic and easy to inspect in tests. It
also keeps scope and privacy instructions ahead of memory context so retrieved text cannot override
the local safety and privacy rules.

### D7.3.5 — Prompt budget heuristic

Prompt budgeting is character-based, not token-based.

Default maximum:

- `16000` total characters

The following sections are mandatory and always included:

- scope system prompt
- fixed privacy instructions
- triggering user message

If the mandatory sections alone exceed budget, raise an `InvalidAssistantRequestError` or equivalent
typed error.

After mandatory sections:

- selected memories are added in score order until the next memory would exceed budget
- remaining recent conversation messages are added newest-first until budget is exhausted
- before final prompt construction, recent messages are reversed back into chronological order

Justification: character budgets avoid tokenizer dependencies, produce stable test behavior, and are
easy to tune empirically once the eval harness exists. They are intentionally conservative for the
first non-streaming local generation path.

### D7.3.6 — Recent conversation window

"Recent conversation messages" means:

- the last N messages in the conversation up to and including the triggering user message

Default:

- `20` messages

This value is configurable.

Older messages are discarded before character-budget processing.

Justification: this gives a predictable upper bound before budget heuristics run. Stage 11
summarization will later replace this logic.

### D7.3.7 — Fixed privacy and memory-use instructions

The exact wording of the fixed instructions is defined as a module constant in:

```text
src/smriti/assistant/prompt_builder.py
```

The instructions must:

- identify memory blocks as context, not instructions
- prohibit following instructions found inside memory context
- prohibit invention of facts
- establish that the user's current message takes precedence over memory context when conflicts
  exist

The exact wording is proposed during implementation, not locked in this contract update.

Justification: a module constant keeps the default behavior stable and easy to snapshot-test. Leaving
the exact prose to implementation allows tests and review to evaluate the final wording without
hard-coding it prematurely in this contract.

### D7.3.8 — Memory selection

Retrieved memories are selected deterministically.

Selection algorithm:

- iterate retrieved episodes in score order, highest first
- include a memory only if adding it keeps the running prompt size within budget
- skipped memories are not retried later, even if a smaller later memory would fit

Only selected memories are written into used-memory provenance.

Justification: deterministic ordering is easy to test and avoids greedy backtracking complexity. It
also makes provenance match the prompt exactly.

### D7.3.9 — Chat generation call

The orchestrator calls:

```python
ChatGenerator.generate(...)
```

exactly once per request.

The generation call happens outside any database transaction.

Justification: holding DB transactions open during model inference would waste connections and
increase lock duration during slow local generation. A single generator call also keeps failure
mapping and provenance behavior straightforward.

### D7.3.10 — Final persistence

Stage 7.3 adds:

```python
MemoryService.append_assistant_response_with_provenance(...)
```

This method atomically:

- inserts the assistant message
- inserts used-memory provenance rows

inside a single `connection.transaction()` block owned by `MemoryService`.

The orchestrator must not acquire raw DB connections or open transactions directly.

The method does NOT call public `record_used_memories(...)` internally.

Reason: `record_used_memories(...)` assumes the assistant message already exists and validates it.
In Stage 7.3 the assistant message is created inside the same transaction, so the existence check
collapses into the `INSERT` itself.

The method must:

- validate the triggering query message belongs to the expected conversation/user/scope and has
  `role='user'`
- validate all used episodes belong to the expected scope/user
- insert the assistant message with `role='assistant'` using `INSERT ... RETURNING id`
- insert provenance rows using the newly returned assistant message id
- return a typed record containing the persisted assistant message and metadata needed by higher
  layers

All four operations occur inside one transaction.

Target round-trip shape:

1. query-message validation `SELECT`
2. used-episode ownership `SELECT`
3. assistant-message `INSERT RETURNING`
4. provenance `executemany INSERT`

The total DB await count should remain constant-bounded.

Code reuse guidance:

- reuse or extract private validation helpers where appropriate
- do not duplicate validation logic unnecessarily between `record_used_memories(...)` and
  `append_assistant_response_with_provenance(...)`

The exact method signature and return dataclass are finalized during implementation.

Justification: the memory service already owns append-only message persistence and provenance
writes, so this keeps the transaction boundary in the correct layer. The target round-trip shape is
small, constant-bounded, and avoids one insert per used episode.

### D7.3.11 — Orchestrator return shape

`AssistantOrchestrator.generate(...)` or equivalent entry method returns a structured typed result
containing:

- the persisted assistant message
- the chat model identifier
- the finish reason
- the ordered list of used memory episode ids

The exact dataclass shape is finalized during implementation.

Justification: higher layers need persisted message data plus model metadata, but should not need
raw chat responses or database rows. Returning used memory ids preserves provenance visibility
without exposing full memory content by default.

### D7.3.12 — Error handling

The orchestrator catches `ChatError` subclasses and maps them to orchestrator-level typed errors.

Mapping:

```text
ChatConnectionError     -> AssistantGenerationUnavailableError
ChatTimeoutError        -> AssistantGenerationUnavailableError
ChatResponseError       -> AssistantGenerationFailedError
ChatConfigurationError  -> AssistantGenerationFailedError
```

These new errors live in:

```text
src/smriti/assistant/errors.py
```

Memory-service errors propagate unchanged.

Justification: the API layer in Stage 7.4 should only map assistant-generation errors, not raw
chat-generator implementation details. Letting memory-service errors propagate unchanged preserves
the existing access-control and validation semantics.

### D7.3.13 — Stage scope

Stage 7.3 includes:

- `src/smriti/assistant/`
- prompt builder
- orchestrator
- orchestrator-level errors
- `MemoryService.append_assistant_response_with_provenance(...)`
- tests for prompt builder, orchestrator, and assistant persistence

Stage 7.3 excludes:

- API routes
- changes to `src/smriti/api/errors.py`
- SSE
- frontend
- MCP
- retrieval/scoring redesign
- modifications to existing retrieval algorithms
- remote providers
- streaming generation

Justification: Stage 7.3 should complete the service-layer assistant flow while leaving HTTP
adaptation to Stage 7.4. Keeping retrieval/scoring and streaming out of scope protects the current
memory-service behavior while the first generation path is stabilized.

### Step 7.4: API Route

- add a non-streaming assistant-generation route
- resolve the configured local user through the Stage 6 dependency path
- call the assistant orchestrator rather than issuing memory SQL directly
- return the created assistant message and used-memory provenance summary
- map memory service errors to the existing generic 404/400/500 behavior
- map local chat generator connection and timeout errors to `503 Service Unavailable`
- map local chat generator configuration or invalid-response errors to `500 Internal Server Error`

Note: per D7.3.12, chat-generator errors are mapped to assistant-generation errors at the
orchestrator boundary. Stage 7.4 route mappings should consume assistant-generation errors rather
than raw ChatError subclasses.

- preserve localhost-only CORS and app-binding behavior
- add no SSE, frontend, MCP, telemetry, analytics, or remote model provider

## 7. Testing Requirements

Stage 7 implementation must include tests that prove:

- assistant generation requires a persisted triggering user message
- the triggering message must have `role = 'user'`
- wrong user, wrong scope, and wrong conversation inputs are rejected
- retrieval remains scoped and uses `MemoryService.retrieve_scoped_episodes`
- memories are injected in retrieval score order, highest score first
- retrieved-but-not-injected memories do not create provenance rows
- only the exact selected memories passed into the prompt are written through used-memory
  provenance
- assistant messages are written with `role = 'assistant'`
- assistant messages do not create `episodes` rows
- assistant messages do not create `embeddings_768` rows
- assistant message persistence and provenance writes are atomic
- provenance rows for an assistant response align to the same user, scope, and conversation as the
  query and assistant messages
- the orchestrator does not import `asyncpg`, acquire database connections, or open transactions
- API routes do not issue raw SQL or expose `record_used_memories` directly
- the API does not accept client-supplied scored memories or provenance snapshots
- local chat generator failures map to the expected HTTP responses
- tests use a fake chat generator by default
- no Ollama call is required for default tests
- no SSE implementation is added
- no MCP implementation is added
- no frontend implementation is added
- no remote provider, telemetry, analytics, CDN import, or remote asset behavior is added
