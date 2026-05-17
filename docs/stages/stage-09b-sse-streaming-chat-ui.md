# Stage 9b: SSE Streaming Chat UI

## 1. Purpose

Add SSE token streaming to the chat UI built in Stage 9a, using the
existing `POST /conversations/{conversation_id}/assistant-response/stream`
endpoint locked in Stage 7.6.

Stage 9b proves the streaming chat loop:

1. user submits a message
2. `POST /conversations/{conversation_id}/messages` persists the user message
3. the messages query is refetched, making the persisted user message visible
4. `POST /conversations/{conversation_id}/assistant-response/stream` initiates an SSE stream
5. while the stream is active, an assistant draft is rendered transiently from streamed tokens
6. on `done`, the messages query is refetched, the persisted assistant message replaces the transient draft, and the transient draft is cleared
7. on `error`, the transient draft is cleared and a retry control is shown
8. the user may cancel mid-stream via a Stop control; no error UI appears and no partial assistant message is persisted

**Architectural spine of Stage 9b.** Persisted messages fetched from the
backend are the only source of truth for messages displayed in the
conversation. The transient assistant draft rendered during streaming
is in-memory UI state only and is never written into the TanStack Query
cache. Every other rule in this document follows from this invariant.

Stage 9b does not change any backend behavior. It does not modify
backend code, the SSE wire format, or the generated OpenAPI types.

## 2. Scope

Stage 9b delivers SSE streaming UX on top of Stage 9a's persisted chat
loop, with one deliberate evolution: the persisted user message becomes
visible immediately after `POST /messages` succeeds, rather than only
after the assistant response completes.

The transient assistant draft rendered during an active stream is the
only non-persisted UI element introduced in Stage 9b.

## 3. Non-Goals

Stage 9b does not deliver:

- Optimistic user messages (the persisted-only model still applies; the
  user message appears via refetch, not optimistically)
- Optimistic assistant messages outside the explicit transient-draft window
- Use of `done.assistant_message` as a cache write to skip refetch
- Surfacing of `chat_model`, `finish_reason`, `used_memory_episode_ids`,
  backend `code`, or backend `message` anywhere user- or developer-visible
- Differentiated handling of error `code` values
- A queue of pending sends, or any "send another message while streaming" affordance
- Scope or conversation management beyond what Stage 9a delivered (Stage 10)
- Memory episode or provenance visualization
- Routing or URL-based state
- Browser-storage persistence of any kind
- Background streaming (streams that continue after conversation switch)
- WebSockets, EventSource, or any non-`fetch` streaming transport
- MSW or any new test-mocking dependency
- Any new shadcn primitive, unless one is genuinely needed and copied locally
- Generalized streaming/event frameworks, stream registries, event buses,
  reusable transport managers, or protocol adapters. Stage 9b supports
  exactly one SSE endpoint.

## 4. Locked Decisions

### 4.1 Stage 9a invariants carried forward

All Stage 9a locked decisions remain in force except where explicitly
evolved by §4.2 below. In particular:

- The UI renders only persisted server state, except for the explicit
  transient assistant draft during an active stream (§4.2).
- `used_memory_episode_ids`, `chat_model`, and `finish_reason` are not
  surfaced anywhere in the UI, console, or developer tooling.
- Backend defaults for `top_k`, `max_prompt_chars`, and `recent_message_limit`
  are used. No UI controls are exposed.
- Conversations are filtered by `scope_id` client-side.
- Messages are sorted by `position` ascending.
- Assistant retry reuses the same `query_message_id`; retry does not
  re-create the user message.
- Selected scope and conversation live only in React state. No
  localStorage, sessionStorage, URL params, or query-string syncing.
- `ChatView` is keyed by conversation id in `ChatPage`. This keying is
  load-bearing for transient streaming state isolation and must not change.

### 4.2 Stage 9b evolution and new decisions

**Architectural spine (restated).**

Persisted messages fetched from the backend are the only source of
truth for messages displayed in the conversation. The transient
assistant draft is in-memory UI state only and is never written into
the TanStack Query cache.

The streaming lifecycle must not mutate persisted message entries
already present in the cache. The only cache interaction permitted by
the streaming hook is refetching `["messages", "list", conversationId]`
after Step 1 succeeds and again on `done`.

**Backend invariant relied upon.**

Stage 9b relies on the Stage 7.6 backend invariant that assistant
messages are persisted only after successful stream completion.
Disconnect, abort, or stream failure before `done` must not result in
partial assistant persistence. Stage 9b does not verify this invariant
itself; the backend's `request.is_disconnected()` polling and
orchestrator teardown are the mechanisms that enforce it.

**Streaming transport.**

- Streaming uses `fetch` with `ReadableStream`. `EventSource` is not used:
  the streaming endpoint is `POST`, which `EventSource` cannot issue.
- A new method is added to `src/api/client.ts`:

  ```ts
  postStream(path: string, body: unknown, signal: AbortSignal): Promise<Response>
  ```

  This method issues the `POST`, returns the raw `Response` without
  consuming the body, and binds the provided `AbortSignal`. It does not
  parse JSON, does not retry, and does not consume `response.body`.
- Components never call `fetch`. Hooks never call raw `fetch`. Hooks call
  `client.postStream`. `src/api/client.ts` is the only place `fetch` is
  used for streaming.

**SSE parsing.**

- SSE parsing should be extracted to `src/lib/sse.ts` if it exceeds
  ~30 lines, requires non-trivial chunk-boundary handling, or benefits
  from isolated test cases for malformed frames. Otherwise it may live
  inline in the streaming hook. The bias is toward extraction.
- The wire format is the format produced by `_stream_sse_events` in
  `src/smriti/api/routes/assistant.py`: events are framed by `\n\n`, with
  an `event:` line and a `data:` line. The frontend parser dispatches on
  the `event:` line value and `JSON.parse`s the `data:` line per event
  type.
- Recognized event types: `start`, `token`, `done`, `error`. Any other
  event type, or any malformed frame, is treated as a stream failure
  (see "Failure cases" below).

**Streaming hook scoping.**

- Streaming state must not be hoisted above `ChatView`. Each `ChatView`
  mount owns exactly one streaming lifecycle.
- The streaming hook (`useAssistantResponseStream` or equivalent) is
  called from within `ChatView` or a child component, never from
  `ChatPage` or higher. This guarantees that conversation-switch
  unmount cleans up the stream by destroying the hook instance.

**Two-refetch model.**

- After `POST /messages` succeeds, the messages query
  (`["messages", "list", conversationId]`) is refetched. Stage 9b must
  await the messages query reflecting the persisted user message before
  initiating `POST /assistant-response/stream`.
- The canonical implementation is
  `await queryClient.refetchQueries({ queryKey: ["messages", "list", conversationId] })`.
  Alternatives that preserve the invariant (`invalidateQueries` + await,
  `fetchQuery`, `ensureQueryData`) are permitted if the implementer has
  a justification, but the invariant — user message visible before
  stream initiation — is the requirement.
- The transient assistant draft must not begin rendering until the
  stream request is initiated.
- This sequential ordering guarantees:
  - the persisted user message appears before any assistant draft
  - no assistant draft is rendered below an empty message list
  - if the user switches conversation between `POST /messages` and the
    stream initiation, no stream is initiated at all
- On `done`, the messages query is refetched a second time. The
  transient assistant draft is cleared.

**`done` payload is not used as a cache write.**

- The `done` event carries a full `assistant_message` payload. Stage 9b
  parses it (the schema requires it) but does NOT write it to the
  TanStack Query cache directly. The refetch on `done` is the sole path
  by which the persisted assistant message reaches the UI.
- This preserves a single rendering path and prevents the entire class
  of bugs around cache divergence, phantom messages, partial
  persistence confusion, and retry reconciliation.

**Transient assistant draft.**

- The transient assistant draft is hook-local state owned by the
  streaming hook, called from `ChatView` (or a child of `ChatView`). It
  is NOT written to the TanStack Query cache.
- The draft buffer is the concatenation of `token.text` fragments from
  the active stream.
- Streaming correctness and lifecycle clarity are prioritized over
  rendering micro-optimizations. Simple `setState` per `token` event
  is acceptable. Localized batching/throttling (e.g., `requestAnimationFrame`,
  small token batching) is permitted if simple and localized, but not
  required.
- The draft is rendered with the same visual treatment as a persisted
  assistant message. A subtle in-progress indicator (e.g., a cursor
  or "..." after the streamed text) is permitted but not required.

**Failure cases.**

The following are all treated as a stream failure with the same retry
UI:

- non-2xx response from `POST /assistant-response/stream` before any
  event is received
- `event: error` arrives mid-stream
- connection closes without ever receiving `event: done`
- malformed SSE frame (missing `event:` or `data:` line, malformed
  framing)
- unrecognized event type
- JSON parse error on the `data:` payload
- `token` event received before any `start` event
- second `start` event during an active stream

On any failure:

- the transient assistant draft is cleared immediately
- a retry control is shown
- the persisted user message remains visible
- the stream state transitions to `error`

**Abort is not an error state transition.**

Abort completion transitions the stream state back to `idle`, not
`error`. Implementations must distinguish `AbortError` (or equivalent
signal-aborted rejection) from genuine stream failures. An `AbortError`
caught by the streaming hook must not route through the same code path
as `event: error` or any other failure case.

Concretely: on abort, the transient draft is cleared, the stream
state returns to `idle`, no retry control is shown, and no error UI
appears. This applies to both Stop-initiated and conversation-switch-
initiated aborts.

**Error display.**

- The UI shows a generic assistant-failure message (e.g., "Assistant
  failed to respond"). The backend's `code` and `message` from the
  `error` event are not displayed.
- `code` and `message` are parsed for protocol correctness but must not
  be logged or displayed anywhere user- or developer-visible.

**Privacy / logging.**

Stage 9b forbids `console` output (`console.log`, `console.info`,
`console.warn`, `console.error`, `console.debug`) of:

- stream payloads (any `data:` field content)
- message content (user or assistant, persisted or transient)
- memory or provenance identifiers (`used_memory_episode_ids`)
- `chat_model`
- `finish_reason`
- backend stream `code` and `message`
- `query_message_id`, message ids, episode ids, scope ids, conversation
  ids when they appear adjacent to message content

This applies in production and development builds. Temporary local
logging during debugging is acceptable and must be removed before
commit. `import.meta.env.DEV` gating does NOT exempt logging of the
items above.

**Retry semantics.**

- Retry re-issues `POST /assistant-response/stream` with the same
  `query_message_id`. It does NOT re-issue `POST /messages` and does
  NOT create a new user message.
- The transient assistant draft is cleared before the retry stream
  begins. Tokens from the failed attempt are not concatenated with
  retry tokens.
- The retry control is visible only when a stream has failed and no
  new stream is yet active. Initiating retry hides the retry control
  and shows the streaming UI (including the Stop control).

**Cancellation: Stop control.**

- During an active stream, the composer's Send control is replaced by
  a Stop control (or Send is hidden/disabled and Stop is visible alongside).
- An "active stream" exists from the moment `POST /assistant-response/stream`
  is initiated until a terminal event occurs: `done`, `error`, or
  abort completion.
- Stop calls `AbortController.abort()` on the active stream's signal.
- On abort completion:
  - the `fetch` rejects with `AbortError`; the hook catches this and
    transitions stream state to `idle`, not `error`
  - the transient assistant draft is cleared
  - no retry UI appears
  - the persisted user message remains visible
- After a terminal event (`done`, `error`, or abort completion), the
  Stop control disappears and the Send control returns.

**Cancellation: conversation switch.**

- Switching to a different conversation during an active stream unmounts
  `ChatView` for the prior conversation (via the existing conversationId
  keying). Unmount must trigger `AbortController.abort()` on any active
  stream.
- The effect of conversation-switch abort is identical to Stop abort:
  transient draft cleared, no retry UI, persisted user message remains.
- The new `ChatView` for the newly selected conversation mounts with
  empty transient state.

**Cancellation: page reload / navigation away.**

- A full page reload aborts the in-flight `fetch` at the browser level
  and clears all in-memory state. The persisted user message remains
  in the database and reappears on next load. No client-side enforcement
  is required for this case.

**Persistence semantics across abort.**

If Stop or conversation-switch abort occurs before the `done` event is
observed, no partial assistant message should appear after reload or
switch-back. If `done` had already been processed by the backend before
the abort propagated, the persisted assistant message may appear on a
later refetch; this is acceptable because persistence had already
completed and is consistent with the Stage 7.6 invariant. Stage 9b
performs no special reconciliation for this race.

**Composer state during streaming.**

- The text input is disabled while a stream is active.
- The Send control is replaced by or hidden in favor of the Stop control
  while a stream is active.
- The composer becomes enabled again on `done`, `error`, or abort
  completion.
- Input value is not preserved across a successful send (it was cleared
  when the message was sent), and is not specially manipulated on
  error or cancel.

**Concurrency guard.**

- Only one stream may be active per `ChatView` instance at a time.
- The composer being disabled during streaming structurally prevents a
  second submission via the UI.
- The hook should additionally guard against double-initiation
  (e.g., a defensive `if (active) return` at stream start), but this
  is an implementation detail and not a separately testable invariant.

**Generated OpenAPI types.**

- `pnpm generate:api` must produce no diff in
  `frontend/src/api/types.ts` for Stage 9b, since no backend changes
  are made.
- The generated types include `AssistantStreamStartData`,
  `AssistantStreamTokenData`, `AssistantStreamDoneData`, and
  `AssistantStreamErrorData` from Stage 7.6. The frontend uses these
  types for the parsed `data:` payloads.

## 5. Backend Flow

When the user submits a message in the composer:

**Step 1.** `POST /conversations/{conversation_id}/messages`

Request body:

```ts
{
  role: "user",
  content: string,
  token_count: number
}
```

Response: `CreatedMessageResponse` including the new message's `id`.

**Step 2.** Refetch `["messages", "list", conversationId]` and await
the messages query reflecting the persisted user message before
proceeding. Canonical: `await queryClient.refetchQueries(...)`. The
persisted user message becomes visible in the message list before
Step 3 begins. If the user has navigated away from the conversation
between Step 1 and the completion of Step 2, Step 3 is not initiated.

**Step 3.** `POST /conversations/{conversation_id}/assistant-response/stream`

Request body:

```ts
{
  scope_id: string,
  query_message_id: string  // the id from Step 1
}
```

Optional fields (`top_k`, `max_prompt_chars`, `recent_message_limit`)
are omitted; backend defaults apply.

The request is issued via `client.postStream(path, body, signal)`. The
returned `Response` body is read as a `ReadableStream` and parsed as
SSE.

**Step 4.** Consume SSE events until a terminal event.

- `start`: clear any prior retry UI, ensure the transient draft buffer
  is empty, begin rendering the transient assistant draft.
- `token`: append `text` to the transient draft buffer; re-render.
- `done`: terminal event. Proceed to Step 5.
- `error`: terminal event. Proceed to Step 6.

The `start` event's `used_memory_episode_ids` and `chat_model` are
parsed and discarded.

**Step 5.** On `done`:

- refetch `["messages", "list", conversationId]`
- await the refetch settling
- clear the transient assistant draft
- the persisted assistant message now appears via the refetched list
- the `done` event's `assistant_message`, `chat_model`, `finish_reason`,
  and `used_memory_episode_ids` are parsed and discarded (not used as
  a cache write)

**Step 6.** On `error` (or any other failure case from §4.2):

- clear the transient assistant draft immediately
- show the retry control
- stream state transitions to `error`
- the persisted user message remains visible
- the backend `code` and `message` are parsed and discarded (not logged
  or displayed)

**Step 7.** On Stop (abort) or conversation switch:

- the `fetch` is aborted via `AbortController.abort()`
- the hook catches the resulting `AbortError` and transitions stream
  state to `idle` (not `error`)
- the transient assistant draft is cleared
- no retry control is shown
- composer is re-enabled
- the persisted user message remains visible

**Failure of Step 1.** No further steps execute. An error is shown
near the composer. The composer remains enabled and the user's input
is preserved.

**Failure of Step 2 (refetch error).** Treated as a recoverable error
near the composer. Step 3 is not initiated. The user may resubmit
(which would create a duplicate user message — a known minor edge
case acceptable for Stage 9b, since refetch failure on a healthy
localhost setup is rare; explicit handling deferred).

## 6. UI Behavior

**Layout.** Unchanged from Stage 9a: sidebar + main pane.

**During an active stream (between Step 3 initiation and terminal
event):**

- the composer text input is disabled
- the Send control is replaced by, or hidden in favor of, the Stop
  control
- the transient assistant draft renders below the persisted user
  message
- the message list reflects the refetched server state from Step 2
  (user message visible, no assistant message yet)
- a subtle in-progress indicator on the draft is permitted but not
  required

**On `done`:**

- the transient draft is cleared
- the refetched message list (Step 5) renders the persisted assistant
  message
- the composer is re-enabled
- the Send control returns

**On `error` (any failure case):**

- the transient draft is cleared immediately
- a generic "Assistant failed to respond" message and a Retry control
  appear in the message area
- the composer is re-enabled
- the Send control returns
- the persisted user message remains visible

**On Stop / conversation switch / page navigation:**

- the transient draft is cleared
- no error UI appears
- no Retry control appears
- the composer is re-enabled (Stop case)
- the persisted user message remains visible

**Differentiated message rendering.** Unchanged from Stage 9a: user
and assistant messages are visually distinguishable by alignment,
background, or both. The transient assistant draft uses the same
visual treatment as a persisted assistant message.

**Empty states.** Unchanged from Stage 9a.

## 7. Query and Mutation Organization

Feature-local API hooks (Stage 9b additions to Stage 9a's structure):

```text
frontend/src/features/chat/api/
  useAssistantResponseStream.ts   (new)
```

`useAssistantResponse.ts` from Stage 9a may remain in place but is no
longer the chat composer's primary submission path. Stage 9b does not
require removing it; if it is unused after Stage 9b, removal is at the
implementer's discretion with a brief justification in the commit
message. Default: leave it in place.

Query keys (unchanged):

```ts
["scopes", "list"]
["conversations", "list"]
["messages", "list", conversationId]
```

The streaming hook is not a TanStack Query mutation. It manages its
own lifecycle: an `AbortController`, the transient draft buffer, and
terminal-state transitions. It uses `queryClient` to refetch
`["messages", "list", conversationId]` after Step 1 succeeds and again
on `done`.

Components do not call `fetch` directly. Hooks do not call raw
`fetch`. The streaming hook calls `client.postStream`. `client.ts` is
the only place `fetch` is used for streaming.

## 8. State Model

- Server state: TanStack Query (scopes, conversations, messages).
- Transient assistant draft state: hook-local (`useState` or `useReducer`)
  inside the streaming hook called from `ChatView` or a child of
  `ChatView`. Not written to the TanStack Query cache. Not hoisted
  above `ChatView`.
- Active stream state: hook-local. Includes the `AbortController` and
  a discriminated-union status (`idle` / `streaming` / `error`).
  Abort transitions return state to `idle`, not `error`.
- UI state for selected scope and conversation: unchanged from Stage 9a
  (React state in `ChatPage`).
- No browser storage of any state.

A suggested discriminated union for the streaming hook:

```ts
type StreamState =
  | { status: "idle" }
  | { status: "streaming"; draft: string; controller: AbortController }
  | { status: "error" }
```

Final shape is at the implementer's discretion provided it preserves
the locked invariants (no error UI on abort, draft cleared on terminal
events, abort returns to `idle`, etc.).

## 9. Component Organization

Stage 9b extends Stage 9a's structure:

```text
frontend/src/
├── features/
│   └── chat/
│       ├── api/
│       │   ├── useAssistantResponse.ts            (existing, 9a)
│       │   └── useAssistantResponseStream.ts      (new, 9b)
│       └── components/
│           ├── ChatView.tsx                       (modified, 9b)
│           ├── Composer.tsx                       (modified, 9b — Stop control)
│           └── ChatEmptyState.tsx                 (unchanged)
├── lib/
│   └── sse.ts                                     (new, biased toward extraction)
└── api/
    └── client.ts                                  (modified, 9b — postStream method)
```

`sse.ts` is biased toward extraction (see §4.2 "SSE parsing"). Inline
parsing in the streaming hook is acceptable only if the parser remains
trivially small (~30 lines or less) and does not require independent
test coverage for chunk-boundary or malformed-frame handling.

## 10. Token Count

Unchanged from Stage 9a. User message `token_count` is approximated
client-side:

```ts
Math.max(1, Math.ceil(content.length / 4))
```

## 11. Styling

Tailwind utility classes per `FRONTEND.md`. No new shadcn primitives
unless genuinely needed and copied locally. The aesthetic established
in Stage 8 and continued in Stage 9a should be preserved.

The Stop control should be visually distinct from the Send control
to avoid mis-clicks. Neither control should rely on color alone to
convey state (accessibility requirement from `FRONTEND.md`).

## 12. Testing

Per `FRONTEND.md`: Vitest + React Testing Library, tests alongside
code, no MSW.

Stage 9b builds a small test helper for constructing a fake
`ReadableStream` of SSE events. The helper produces a `Response`-like
object suitable for stubbing `client.postStream`. The helper supports:

- pre-programmed event sequences
- partial frames (incomplete `event:` or `data:` lines across reads,
  to exercise the parser's framing logic)
- mid-stream injection of malformed frames
- abort signaling

Required test coverage:

**Happy path.**

- Submitting a user message issues `POST /messages` first, then awaits
  the messages-query refetch, then issues `POST /assistant-response/stream`.
- The persisted user message appears in the DOM before
  `POST /assistant-response/stream` is initiated. Acceptable assertion
  forms: (a) snapshot DOM at the moment `client.postStream` is invoked,
  or (b) verify ordering — assert the user message is present when the
  first `token` event is dispatched. Either is acceptable.
- `start` event clears any prior retry UI and prepares the draft buffer.
- `token` events append to the transient draft, visible in the DOM.
- `done` event triggers a second refetch of the messages query.
- After the second refetch settles, the transient draft is cleared and
  the persisted assistant message is rendered from server state.
- During an active stream, the Stop control is visible and Send is
  hidden/disabled.
- After `done`, the Stop control is hidden and Send returns.

**Failure paths.**

- Non-2xx response from `POST /assistant-response/stream` shows the
  retry UI; no transient draft renders.
- `event: error` mid-stream clears the transient draft and shows the
  retry UI.
- Connection closes without `done` clears the draft and shows retry.
- Malformed SSE frame clears the draft and shows retry.
- `token` event before `start` clears the draft and shows retry.
- A second `start` event during an active stream clears the draft and
  shows retry.
- JSON parse error on `data:` payload clears the draft and shows retry.
- Retry click re-issues only `POST /assistant-response/stream` with the
  same `query_message_id`; `POST /messages` is NOT called again.
- Retry click clears any prior failed-attempt tokens before the new
  stream begins.

**Cancellation paths.**

- Stop click during an active stream calls `AbortController.abort()`,
  clears the transient draft, hides Stop, restores Send, shows no
  error UI, and does not show a Retry control.
- Stream state after Stop is `idle`, NOT `error`. The same code path
  that handles `event: error` must not fire on `AbortError`.
- Conversation switch during an active stream aborts the fetch (no
  request body or URL containing conversation A's `query_message_id`
  is sent to conversation B; no transient draft from A appears in B).
- After abort, the persisted user message in the prior conversation
  remains visible on switch-back. Tests the same-conversation-after-
  abort case: user message present, no assistant message (or persisted
  assistant message if backend race per §4.2 "Persistence semantics
  across abort"), composer enabled.

**Logging and privacy.**

- No `console.*` call is made with stream payload contents, message
  content, memory ids, `chat_model`, `finish_reason`, or backend
  `code`/`message`. Asserted via spies on `console.log`, `console.info`,
  `console.warn`, `console.error`, and `console.debug`.

**Persisted-only invariant.**

- The TanStack Query cache for `["messages", "list", conversationId]`
  is not written to directly by the streaming hook at any point.
  Assertion: no `setQueryData` call for that key during a stream
  lifecycle.
- More generally, the streaming lifecycle must not mutate persisted
  message entries already present in the cache. The `setQueryData`
  assertion is the specific check; the general invariant is the
  principle it enforces.
- The transient draft is not present in the cache snapshot after `done`
  or after abort.

**Contract scan: `stage9bContract.test.ts`.**

A new file enforces forbidden patterns in Stage 9b's application code.
The scan excludes:

- `frontend/src/api/openapi.json`
- `frontend/src/api/types.ts`
- `frontend/src/test/` (or wherever shared test helpers live)
- `*.test.ts` and `*.test.tsx`

Forbidden patterns in application code:

- `EventSource`
- `localStorage`
- `sessionStorage`
- `URLSearchParams`
- direct `fetch(` calls outside `src/api/client.ts`
- `console.log`, `console.info`, `console.warn`, `console.error`,
  `console.debug` in any chat/streaming code path
- `setQueryData` calls on `["messages", "list", ...]` keys
- string literal `"data:"` or `"event:"` outside `src/lib/sse.ts` and
  the streaming hook (defensive: prevents ad-hoc SSE parsing elsewhere)

Failures must include file path and matched line.

## 13. Validation

From `frontend/`:

```bash
pnpm generate:api
pnpm typecheck
pnpm lint
pnpm test
pnpm build
```

From repo root:

```bash
git diff --check
```

`pnpm generate:api` must produce no diff in `frontend/src/api/types.ts`.
If it does, that's a flag — investigate before proceeding.

## 14. Manual Verification

With the backend running on `127.0.0.1:8000`, Postgres up, Ollama
available, and the frontend running on `127.0.0.1:5173`:

1. Open `http://127.0.0.1:5173` in a browser.
2. Select an existing scope and conversation, or create them.
3. Send a message via the composer. Observe:
   - composer disables, Send is replaced by Stop
   - the user message appears in the list before any assistant text
   - the assistant response renders token-by-token below the user
     message
   - on completion, the streamed text is replaced (no visible flicker
     required) by the persisted assistant message
   - Stop disappears; Send returns
4. Send another message and click Stop mid-stream. Observe:
   - the streamed text disappears
   - no error message appears
   - no Retry control appears
   - the user message remains visible
   - composer re-enables; Send returns
5. Send another message; while streaming, switch to a different
   conversation in the same scope (creating one if needed). Observe:
   - the new conversation shows its own state, no transient text from
     the prior conversation
   - returning to the prior conversation shows the persisted user
     message; per §4.2 "Persistence semantics across abort", the
     assistant reply is absent if abort occurred before `done`, or
     present if `done` had already completed before abort propagated.
     Both are acceptable.
6. Force a backend failure (e.g., stop Ollama) and send a message.
   Observe:
   - user message appears
   - generic "Assistant failed to respond" message and Retry control
     appear
   - no `code`/`message` from the backend is visible in the UI or
     browser console
7. Restart Ollama and click Retry. Observe:
   - `POST /messages` is NOT issued again (verifiable in network tab)
   - `POST /assistant-response/stream` is issued with the same
     `query_message_id` as the failed attempt
   - the prior failed-attempt tokens (if any rendered before the
     failure) are gone before the retry stream begins
8. Browser console throughout: no logged message content, memory ids,
   `chat_model`, `finish_reason`, or backend error code/message.
9. Network tab throughout: only `127.0.0.1:8000` is contacted.
10. Reload mid-stream: UI state resets; the persisted user message
    remains intact in the database (verifiable by reselecting the
    conversation after reload). Per §4.2 "Persistence semantics across
    abort", no partial assistant message appears.

## 15. Exit Criteria

Stage 9b is complete when:

- The streaming chat flow works end-to-end per §5 and §14.
- The persisted user message appears via refetch before the stream is
  initiated.
- The transient assistant draft renders only during an active stream
  and is cleared on every terminal event (`done`, `error`, abort).
- The persisted assistant message reaches the UI only via refetch on
  `done`, never via direct cache writes from the stream.
- Stop, conversation switch, and page reload all abort the active
  stream cleanly. No error UI appears on user-initiated cancellation.
  Stream state after abort is `idle`, not `error`.
- Retry re-issues only the streaming request, never `POST /messages`.
- No `chat_model`, `finish_reason`, `used_memory_episode_ids`, or
  backend stream `code`/`message` is visible anywhere — UI, console,
  or developer tooling.
- `stage9bContract.test.ts` passes and enforces the forbidden-pattern
  list in §12.
- All frontend validation commands pass: `pnpm typecheck`, `pnpm lint`,
  `pnpm test`, `pnpm build`.
- `pnpm generate:api` produces no diff in committed types.
- `git diff --check` passes.
- Manual verification (§14) succeeds end-to-end.

## 16. Contract Discipline

Per `AGENTS.md` and `FRONTEND.md`, the implementer must not:

- Add dependencies not already in `package.json` without raising the
  question first. In particular: no MSW, no Zustand, no React Hook
  Form, no new shadcn primitives unless genuinely needed and copied
  locally.
- Modify backend code, routes, schemas, startup, or lifespan behavior.
- Modify the SSE wire format or the generated OpenAPI types.
- Add anything from the Non-Goals section.
- Resolve ambiguity silently. Stop and ask if the contract does not
  cover a needed decision.
- Introduce abstractions "for future flexibility." Stage 9b's surface
  is narrow; keep it narrow.
- Introduce generalized streaming/event frameworks, stream registries,
  event buses, reusable transport managers, or protocol adapters.
  Stage 9b supports exactly one SSE endpoint.
- Re-key or remove the conversation-id keying on `ChatView`. That
  keying is load-bearing for transient streaming state isolation across
  conversation switches.
- Hoist streaming state above `ChatView`. Each `ChatView` mount owns
  exactly one streaming lifecycle.
- Write `done.assistant_message` to the TanStack Query cache to skip
  the refetch.
- Route `AbortError` through the same code path as `event: error` or
  any other failure case. Abort returns the stream to `idle`, not
  `error`.
- Display, log, or otherwise surface the backend `error.code`,
  `error.message`, `chat_model`, `finish_reason`, or
  `used_memory_episode_ids`.
- Add `console.*` calls in any chat/streaming code path, even gated
  on `import.meta.env.DEV`.

If the contract is genuinely ambiguous on a needed point, the
implementer stops and asks. Locked decisions are not subject to
reinterpretation.
