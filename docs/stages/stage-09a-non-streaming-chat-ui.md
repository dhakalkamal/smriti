# Stage 9a: Non-Streaming Chat UI

## 1. Purpose

Implement the first usable chat UI for Smriti against the existing non-streaming assistant endpoint.

Stage 9a proves the full persisted chat loop:

1. select an existing scope, or create one if none exists
2. select a conversation in that scope, or create one
3. load persisted messages
4. submit a user message
5. generate a non-streaming assistant response
6. refetch and render persisted messages

No SSE streaming is implemented in this stage. SSE is Stage 9b.

## 2. Scope

Stage 9a delivers a working chat experience using only the non-streaming backend route. The UI renders only persisted server state. There is no optimistic rendering, no streaming, no client-side persistence.

## 3. Non-Goals

Stage 9a does not deliver:

- SSE token streaming
- Optimistic user messages
- Optimistic assistant messages
- Scope editing, deletion, or full management UX (Stage 10)
- Conversation editing or deletion (Stage 10)
- Cross-scope retrieval display
- Memory episode or provenance visualization
- Routing or URL-based state
- Browser-storage persistence of any kind

## 4. Locked Decisions

- Scope creation is included in Stage 9a, restricted to name + optional system prompt. Full scope management is Stage 10.
- Layout is sidebar + main pane. Left: scope selector and conversation list for selected scope. Right: active conversation (messages + composer) or an empty state.
- No optimistic updates. The UI renders only persisted server state.
- `used_memory_episode_ids`, `chat_model`, and `finish_reason` are not surfaced anywhere in the UI, console, or developer tooling. They remain available programmatically on the mutation result for future stages.
- Conversations are filtered by `scope_id` client-side after fetching `GET /conversations`.
- Messages are sorted by `position` ascending for display, not by `created_at`.
- Assistant retry reuses the same `query_message_id` from the original user-message creation. It does not re-create the user message.
- Backend defaults for `top_k`, `max_prompt_chars`, and `recent_message_limit` are used. No UI controls are exposed for these.
- Token count for user messages is approximated as `Math.max(1, Math.ceil(content.length / 4))`. Exact tokenization is deferred.
- Selected scope and conversation live only in React state. Reload resets UI state. No localStorage, sessionStorage, URL params, or query-string syncing.

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

**Step 2.** `POST /conversations/{conversation_id}/assistant-response`

Request body:

```ts
{
  scope_id: string,
  query_message_id: string  // the id from Step 1
}
```

Optional fields (`top_k`, `max_prompt_chars`, `recent_message_limit`) are omitted; backend defaults apply.

Response: `AssistantGenerationResponse`. The endpoint returns only after the assistant message has been persisted.

**Step 3.** Invalidate the messages query for the conversation. TanStack Query refetches `GET /conversations/{conversation_id}/messages`. The returned list includes both the user message and the new assistant message.

If Step 1 fails: no further calls are made; an error is shown; the composer retains the user's input.

If Step 2 fails: the user message is already persisted server-side and will appear after the messages query is refetched (or on next reload). An error indicator appears in place of the expected assistant response. A retry control is available that re-issues Step 2 with the same `query_message_id`. Retry does not re-create the user message.

## 6. UI Behavior

**Layout.** Persistent left sidebar showing scope selector and conversation list filtered by selected scope. Right pane shows the active conversation (message list + composer) or an empty state if no conversation is selected.

**Persisted-only rendering.** The message list reflects only the server response from `GET /conversations/{conversation_id}/messages`. No optimistic, transient, or simulated messages are rendered.

**During the message-submit-and-generate chain:**

- The composer is disabled.
- A loading indicator is visible (e.g., "Assistant is thinking...").
- The user message does NOT appear in the message list until the messages query is refetched after Step 2 completes.

**Empty states required:**

- No scopes exist: sidebar shows a prompt to create the first scope. Main pane shows an instructional empty state.
- Scope selected, no conversations in that scope: sidebar shows a "+ New conversation" affordance. Main pane shows an empty state.
- Conversation selected, no messages yet: composer is ready, message area is empty with subtle hint text.

**Error states required:**

- `POST /messages` fails: inline error near composer, composer remains enabled, user's input is preserved.
- `POST /assistant-response` fails: error indicator in the message area, retry control available, composer re-enabled.
- Scope/conversation create fails: inline error in the relevant form area.

**Differentiated message rendering.** User and assistant messages are visually distinguishable (alignment, background, or both). Role enum from `MessageResponse.role` drives the distinction.

## 7. Query and Mutation Organization

Feature-local API hooks:

```text
frontend/src/features/scopes/api/
frontend/src/features/conversations/api/
frontend/src/features/messages/api/
frontend/src/features/chat/api/
```

Query keys:

```ts
["scopes", "list"]
["conversations", "list"]
["messages", "list", conversationId]
```

Assistant generation is a mutation, not a query. It is not assigned a TanStack Query cache key.

After successful assistant generation, `["messages", "list", conversationId]` is invalidated to trigger refetch.

Components do not call `fetch` directly. All API access goes through `src/api/client.ts` and feature-local hooks.

## 8. State Model

- Server state: TanStack Query (scopes, conversations, messages).
- UI state: `useState` and `useReducer` only. No Zustand.
- Selected scope id and selected conversation id live in React state in the top-level chat component or via lifted state. No context required unless prop drilling becomes painful.
- No browser storage of UI state.

## 9. Component Organization

Suggested structure (Claude/codex may adjust modestly if justified):

```text
frontend/src/
├── features/
│   ├── scopes/
│   │   ├── api/
│   │   │   ├── useScopes.ts
│   │   │   └── useCreateScope.ts
│   │   └── components/
│   │       ├── ScopeSelector.tsx
│   │       └── CreateScopeForm.tsx
│   ├── conversations/
│   │   ├── api/
│   │   │   ├── useConversations.ts
│   │   │   └── useCreateConversation.ts
│   │   └── components/
│   │       ├── ConversationList.tsx
│   │       └── CreateConversationForm.tsx
│   ├── messages/
│   │   ├── api/
│   │   │   ├── useMessages.ts
│   │   │   └── useCreateMessage.ts
│   │   └── components/
│   │       ├── MessageList.tsx
│   │       └── MessageItem.tsx
│   └── chat/
│       ├── api/
│       │   └── useAssistantResponse.ts
│       └── components/
│           ├── ChatView.tsx
│           ├── Composer.tsx
│           └── ChatEmptyState.tsx
└── pages/
    └── ChatPage.tsx
```

## 10. Token Count

User message `token_count` is approximated client-side:

```ts
Math.max(1, Math.ceil(content.length / 4))
```

This satisfies the backend's `token_count: int = Field(ge=0)` requirement and avoids depending on a tokenization library. Exact token counting is deferred.

## 11. Styling

Tailwind utility classes per FRONTEND.md. No new shadcn components copied in Stage 9a unless one is genuinely needed and copied locally (not imported as a runtime library). The aesthetic established in Stage 8 (clean, minimal, accent bar) should be continued or modestly evolved.

No CSS-in-JS. No inline styles unless trivially justified.

## 12. Testing

Per FRONTEND.md: Vitest + React Testing Library. Tests live alongside the code they test (`*.test.ts` or `*.test.tsx`). API mocking via simple fetch stubs or mocked hooks. No MSW.

Required test coverage:

- Scope list renders given mocked scopes.
- Scope create flow: submitting the form calls `POST /scopes` and invalidates the scopes list.
- Conversation list renders given mocked conversations, filtered by selected scope.
- Conversation create flow: submitting calls `POST /conversations` with the selected `scope_id`.
- Message list renders given mocked messages, sorted by `position`.
- Submitting a user message triggers `POST /messages` first, then `POST /assistant-response` with the returned `id` as `query_message_id`.
- After assistant success, the messages query is invalidated.
- If `POST /messages` fails, `POST /assistant-response` is not called.
- If `POST /assistant-response` fails, the user message is not duplicated on retry; retry reuses the same `query_message_id`.
- No streaming UI behavior exists (no EventSource references, no SSE parsing).
- Empty states render appropriately for: no scopes, no conversations, no messages.

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

`pnpm generate:api` should produce no diff in `frontend/src/api/types.ts` since Stage 9a does not modify the backend. If it does produce a diff, that's a flag — investigate before proceeding.

## 14. Manual Verification

With the backend running on `127.0.0.1:8100`, Postgres up, Ollama available, and the frontend running on `127.0.0.1:5173`:

1. Open `http://127.0.0.1:5173` in a browser.
2. If no scopes exist, create one via the UI (name + optional system prompt).
3. Create a conversation in the selected scope.
4. Send a message via the composer.
5. Observe: composer disables, loading indicator appears, then both the user message and the assistant response appear in the message list.
6. Browser console: no errors, no logged message/memory data.
7. Network tab: only `127.0.0.1:8100` is contacted. No external hosts.
8. Reload: UI state resets (selected scope/conversation cleared). Persisted data (scopes, conversations, messages) remains intact server-side.

## 15. Exit Criteria

Stage 9a is complete when:

- The user can create or select a scope, create or select a conversation, and chat with the assistant via the non-streaming endpoint.
- The UI renders only persisted server state.
- No SSE, streaming, or optimistic-update code is introduced.
- No browser persistence is introduced.
- `used_memory_episode_ids`, `chat_model`, and `finish_reason` are not surfaced anywhere user- or developer-visible.
- All frontend validation commands pass: `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build`.
- `pnpm generate:api` produces no diff in committed types.
- `git diff --check` passes.
- Manual verification flow succeeds end-to-end.

## 16. Contract Discipline

Per AGENTS.md and FRONTEND.md, the implementer must not:

- Add dependencies not already in `package.json` without raising the question first.
- Modify backend code, routes, schema, startup, or lifespan behavior.
- Add anything from the Non-Goals section.
- Resolve ambiguity silently. Stop and ask if the contract does not cover a needed decision.
- Introduce abstractions for future flexibility. Stage 9a's surface is small; keep it small.

If the contract is genuinely ambiguous on a needed point, the implementer stops and asks. Locked decisions are not subject to reinterpretation.
