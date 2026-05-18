# Stage 10a: Conversation Delete / Cleanup UI

## 1. Purpose

Add explicit user-driven conversation deletion to Smriti.

Stage 10a implements the smallest safe management action for existing
conversations: deleting one conversation and cleaning up all dependent
memory records.

This is a contract-only stage before implementation.

## 2. Current Backend Inspection

Before drafting this contract, the current backend and schema were inspected.

Findings:

- `src/smriti/api/routes/conversations.py` currently has:
  - `GET /conversations`
  - `POST /conversations`
  - no existing `DELETE /conversations/{conversation_id}` route
- `src/smriti/memory/service.py` currently has:
  - `create_conversation`
  - `list_conversations`
  - message/retrieval/assistant-generation methods
  - no existing `delete_conversation` method
- Current migrations already define schema-level cascade behavior:
  - `messages.conversation_id -> conversations(id) ON DELETE CASCADE`
  - `episodes.conversation_id -> conversations(id) ON DELETE CASCADE`
  - `episodes.message_id -> messages(id) ON DELETE CASCADE`
  - `embeddings_768.episode_id -> episodes(id) ON DELETE CASCADE`
  - `message_retrievals.query_message_id/query_conversation_id -> messages(id, conversation_id) ON DELETE CASCADE`
  - `message_retrievals.query_conversation_id/scope_id -> conversations(id, scope_id) ON DELETE CASCADE`
  - `message_retrievals.episode_id/scope_id -> episodes(id, scope_id) ON DELETE CASCADE`
  - `message_retrievals.assistant_message_id -> messages(id) ON DELETE CASCADE`
- Conversation titles are user-provided and nullable:
  - `conversations.title TEXT` (nullable, no default)
  - `CreateConversationRequest.title: str | None = None`
  - `CreateConversationBody.title: str | None = None`
  - No code path in `src/smriti/` auto-generates a conversation title
    from message content.
  - The existing frontend renders `conversation.title ?? "Untitled conversation"`.

No new migration is required for Stage 10a. Existing cascade constraints are
considered sufficient and must instead be verified through tests.

## 3. Scope

Stage 10a adds:

- backend `MemoryService.delete_conversation`
- backend `DELETE /conversations/{conversation_id}`
- frontend delete affordance for conversations
- confirmation modal
- cache cleanup after successful delete
- tests for backend cascade correctness
- tests for frontend confirmation/cache behavior

## 4. Non-Goals

Stage 10a does not add:

- conversation rename/edit
- bulk deletion
- undo
- trash/archive
- soft delete
- scope deletion
- scope editing
- message-level deletion
- memory episode management UI
- provenance visualization
- browser persistence
- routing or URL state
- new frontend state libraries
- new backend dependencies

## 5. Locked Decision: Hard Delete

Stage 10a uses hard delete.

### 5.1 Argument for soft delete

Soft delete would preserve historical records, reduce accidental data-loss risk,
and make undo possible later.

However, soft delete would require new schema columns and deletion filters across:

- conversations queries
- message queries
- retrieval queries
- provenance queries
- embedding joins
- cache invalidation paths

It would also create new failure modes where "deleted" data accidentally reappears
through retrieval or UI filtering bugs.

This increases privacy risk and architectural complexity.

### 5.2 Argument for hard delete

Hard delete matches the user's explicit intent: delete this conversation.

AGENTS.md states that messages are append-only unless explicitly forgotten/exported.
Conversation deletion is exactly such an explicit forgetting action.

Smriti's privacy model is structural. If the user requests deletion, the system
should remove the conversation and all dependent memory artifacts rather than
hide them behind flags.

The safety mechanism is the confirmation modal, not soft-delete recovery.

### 5.3 Final decision

Stage 10a uses hard delete.

No `deleted_at`, `is_deleted`, `archived_at`, or soft-delete filtering is added.

## 6. Backend Contract

### 6.1 Request model

Add a typed request model, for example:

```python
@dataclass(frozen=True)
class DeleteConversationRequest:
    user_id: UUID
    conversation_id: UUID
```

Do not pass anonymous dictionaries across the service boundary.

### 6.2 New service method

Add:

```python
async def delete_conversation(
    self,
    request: DeleteConversationRequest,
) -> None:
    ...
```

The method must:

1. execute inside the existing MemoryService transaction conventions
2. delete by `conversation_id` and `user_id`
3. rely on schema-level cascades for dependent rows
4. not perform SQL outside MemoryService

Ownership and existence are checked by the delete statement itself:

```sql
DELETE FROM conversations
WHERE id = $1
  AND user_id = $2
RETURNING id;
```

If no row is returned:

- not-found and not-owned are treated identically
- the service raises `ConversationNotFoundError`. This error type already exists
  in `src/smriti/memory/service.py` (raised today as a safety check inside
  `create_conversation`) and Stage 10a reuses it for the missing/not-owned case
  on delete.

No separate `SELECT` verification query is performed.

### 6.3 New API route

Add:

```text
DELETE /conversations/{conversation_id}
```

Success response:

```text
204 No Content
```

Failure semantics:

- nonexistent conversation and non-owned conversation both return `404`
- the response must not distinguish ownership failure from true absence
- malformed UUIDs follow existing FastAPI validation behavior
- unexpected backend failures return generic `500` responses consistent with
  existing route conventions

The route must:

- live in `src/smriti/api/routes/conversations.py`
- obtain `local_user_id` from the existing dependency
- obtain `MemoryService` from the existing dependency
- call `MemoryService.delete_conversation`
- contain no direct SQL
- return no deleted-row payload
- return no message content or provenance data

## 7. Cascade Contract

Deleting a conversation must remove:

- the `conversations` row
- all messages in that conversation
- all episodes in that conversation, including:
  - `kind = 'message'`
  - `kind = 'summary'`
- all `embeddings_768` rows for deleted episodes
- all `message_retrievals` where:
  - the deleted conversation was the query conversation
  - the deleted conversation contained the query message
  - the deleted conversation contained the assistant message
  - the deleted conversation contained an episode referenced by provenance rows

The current schema should handle this through cascades. Stage 10a implementation
must verify this through tests.

## 8. Cross-Conversation Provenance Decision

Deleting a conversation must not delete unrelated conversations or unrelated
episodes that merely participated in retrieval.

Example:

- conversation B retrieved episodes from conversation A
- deleting conversation A deletes:
  - conversation A
  - its messages
  - its episodes
  - provenance rows referencing those deleted episodes
- conversation B itself remains intact

### Access metadata decision

`access_count` and `last_accessed_at` are aggregate retrieval signals, not
perfect provenance reconstruction data.

They are intentionally lossy retrieval-ranking inputs.

Recomputing them after deletion would require either:

- preserving deleted provenance rows, contradicting hard delete, or
- maintaining parallel durable counters outside the provenance graph

This complexity is not justified for low-weight ranking signals.

Minor metadata drift after deletion is accepted by contract.

## 9. Frontend Contract

### 9.1 Delete affordance

The sidebar conversation list exposes a delete affordance for **inactive**
conversation rows (see §9.5 for the active-conversation case).

The affordance must:

- be keyboard reachable
- not depend on hover-only visibility
- not interfere with normal conversation selection
- remain visually secondary to the primary conversation-select action

The exact visual treatment is implementation discretion:

- icon button
- kebab menu
- secondary action area

All are permitted provided accessibility requirements from FRONTEND.md are
satisfied.

### 9.2 Confirmation modal

Deletion requires confirmation.

Modal requirements:

- includes the conversation title in the body
- has one Cancel action
- has one Delete action
- Escape cancels
- Cancel closes the modal
- Delete triggers the mutation
- no message content is shown

If conversation titles are nullable or empty, use the same fallback the
conversation list already uses:

```text
Untitled conversation
```

Conversation titles are user-provided and never derived from message content
(see §2). The modal therefore never displays message content by construction.
Stage 10a must not introduce any code path that derives modal text from
messages.

### 9.3 Active conversation deletion

When the currently selected conversation is deleted successfully:

- clear selected conversation state
- invalidate/refetch `["conversations", "list"]`
- remove the deleted conversation's messages cache entry:
  `["messages", "list", conversationId]`
- render the Stage 9a empty-state view
- do not auto-select another conversation

Reason:

Many chat and email UIs auto-advance to the next item after a destructive
action, optimizing for fewer clicks during repeated cleanup. Stage 10a
deliberately diverges: conversation deletion is expected to be infrequent
and deliberate rather than a routine cleanup loop. Automatically transitioning
the user into another conversation immediately afterward can feel abrupt and
elides the moment of "the thing I asked to delete is gone." Stage 10a prefers
an explicit next selection by the user.

If usage patterns later show this is wrong (frequent deletion as a regular
flow), revisit in a later stage.

### 9.4 Non-active conversation deletion

When a non-active conversation is deleted successfully:

- preserve the currently selected conversation
- invalidate/refetch `["conversations", "list"]`
- remove the deleted conversation's messages cache entry

### 9.5 Streaming interaction

Stage 10a must preserve Stage 9b's streaming architecture invariants.

Stage 9b explicitly forbids hoisting streaming state above `ChatView`.
Stage 10a must not introduce:

- shared streaming context
- global streaming state
- lifted stream lifecycle state
- sidebar-owned stream state

Because Stage 9b streaming state is local to `ChatView`:

- the sidebar conversation list does not track streaming state
- the sidebar shows delete affordances for inactive conversations only
- the sidebar must not become aware of active streaming state
- non-active conversations may still be deleted from the sidebar
- the active conversation may only be deleted when no stream is active
- any active-conversation delete affordance must live within `ChatView`
  or within components already local to the active conversation tree

**Sidebar visual asymmetry.** Because the active conversation's sidebar row
does not display the inactive-row delete affordance, sidebar rows are
intentionally non-uniform: the currently selected row will lack the delete
control that inactive rows show. This asymmetry is accepted: it is the
architectural consequence of keeping the sidebar unaware of stream state.
The active conversation's delete affordance is reached from within `ChatView`.

**Active-conversation delete affordance constraints.** The active-conversation
delete affordance must:

- be a button (a single activation reaches the confirmation modal; no
  intermediate menu indirection is required)
- be keyboard reachable
- be visually subordinate to the composer
- not share visual region with the composer's Send/Stop control
- not be positioned such that an accidental click during composition is
  plausible

The specific visual placement within `ChatView` (header area, secondary
toolbar, footer chrome, conversation-title region) is implementation
discretion provided the above constraints are met. Stage 10a does not mandate
introducing a new structural region (e.g. a ChatView header) if one does
not already exist; the implementer may add a small dedicated affordance
area within the active-conversation tree.

The single-activation requirement is intentionally tighter than the
inactive-row affordance in §9.1, which permits kebab menus. The asymmetry
reflects context: inactive-row affordances live in a list of many rows where
kebab compactness is valuable, while the active-conversation affordance lives
in a single slot adjacent to active composition. An overflow/menu-based
delete affordance near the composer increases interaction complexity during
active composition and visually competes with Send/Stop. A single, clearly
delete-shaped button is the safer default. Future compact-layout iterations
may revisit this in a later stage.

Stage 10a does not implement abort-and-delete behavior.

If a stream is active for the selected conversation:

- the active-conversation delete affordance is disabled
- explanatory text must be available near the disabled control
- if keyboard focus on the disabled action is required, use
  `aria-disabled="true"` and prevent activation in the handler instead of
  using a native disabled button
- Stage 10a does not abort the stream automatically

**Re-enabling after stream termination.** Per Stage 9b §4.2, the stream
returns to `idle` on every terminal event (`done`, `error`, user Stop /
abort). The active-conversation delete affordance must re-enable as soon
as the stream returns to `idle`, regardless of which terminal event caused
the return. There is no separate "post-error" or "post-abort" disabled
state.

## 10. Failure Semantics

Backend deletion is all-or-nothing.

If deletion fails:

- the transaction rolls back
- the conversation remains visible
- selected conversation state is unchanged
- messages cache is not removed

Frontend behavior on failure:

- the confirmation modal remains open
- a generic inline error is shown in the modal
- the Delete button returns to enabled state
- the user may retry or cancel

During an active delete mutation:

- Delete and Cancel controls are disabled
- duplicate delete submissions are prevented

Reason for disabling Cancel during in-flight mutation: Stage 10a does not
implement request abort/cancellation for deletes. A visible Cancel that
dismisses the modal while the DELETE request continues to completion would
misrepresent what the user just did — the conversation would still disappear
after Cancel was clicked. Disabling Cancel during the mutation keeps the
control honest. Local DELETE latency is expected to be low; the disabled
window is short.

Raw backend exception details must not be displayed.

**No optimistic removal.** Stage 10a does not implement deletion as an
optimistic mutation. The deleted conversation must remain visible in the
sidebar until the backend `DELETE` returns `204`, at which point the
conversations list is invalidated/refetched and the messages cache entry
is removed. TanStack Query's `onMutate`-based optimistic patterns
(pre-removing from the cache and rolling back on error) must not be used.
This matches the persisted-only posture of Stage 9a and the strict
no-cache-write rule of Stage 9b: the UI reflects server state, not
predicted server state.

## 11. Query and Mutation Organization

Add feature-local API hook:

```text
frontend/src/features/conversations/api/useDeleteConversation.ts
```

The hook uses `useMutation`.

Components do not call `fetch` directly.

All API access goes through `src/api/client.ts` and feature-local hooks.

TanStack Query owns invalidation and cache removal.

## 12. Suggested Frontend Files

Likely modified/added files:

```text
frontend/src/features/conversations/api/useDeleteConversation.ts
frontend/src/features/conversations/components/ConversationList.tsx
frontend/src/features/conversations/components/DeleteConversationDialog.tsx
frontend/src/pages/ChatPage.tsx
```

The active-conversation delete trigger lives within `ChatView` or a
component local to the active-conversation tree, per §9.5. Exact file
boundaries for that trigger remain implementation discretion.

`DeleteConversationDialog` accepts only the props necessary to satisfy §9.2
and §15:

```ts
type DeleteConversationDialogProps = {
  conversationId: string;
  title: string | null;
  open: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  // mutation status props (e.g. isPending, errorMessage) may be added
  // as needed; no message-content-derived props are permitted.
};
```

The dialog must not receive the full conversation record, the messages list,
or any `MessageResponse` value. See §15 for the contract-scan enforcement.

No new frontend state library.

No browser persistence.

No React Router.

No new shadcn primitive unless genuinely needed and copied locally.

## 13. Backend Tests

Required backend tests:

1. deleting a conversation removes the conversation row
2. deleting a conversation removes its messages
3. deleting a conversation removes its `kind = 'message'` episodes
4. deleting a conversation removes its `kind = 'summary'` episodes if
   summaries exist
5. deleting a conversation removes embeddings for deleted episodes
6. deleting a conversation removes provenance rows where the deleted
   conversation is the query conversation
7. deleting a conversation removes provenance rows in other conversations
   that referenced deleted episodes
8. deleting a conversation does not delete unrelated conversations in the
   same scope
9. deleting a conversation does not delete conversations in other scopes
10. deleting a conversation does not delete external episodes that were
    merely retrieved by the deleted conversation
11. deleting nonexistent/non-owned conversations raises
    `ConversationNotFoundError` and `DELETE /conversations/{id}` returns
    `404` for both cases without distinguishing them
12. route test proves `DELETE /conversations/{conversation_id}` returns
    `204` on success

Cross-conversation provenance tests require constructing actual retrieval
relationships. If existing helper patterns already exist in the test suite,
they should be reused.

Backend tests use real Postgres.

## 14. Frontend Tests

Required frontend tests:

1. delete affordance opens confirmation modal
2. modal displays conversation title
3. modal displays the `"Untitled conversation"` fallback when title is null
4. Cancel closes modal without deletion
5. Escape closes modal without deletion
6. Confirm triggers delete mutation
7. successful delete invalidates/refetches conversations list
8. successful delete removes deleted conversation messages cache
9. deleting active conversation clears selected conversation state
10. deleting inactive conversation preserves current active conversation
11. delete failure keeps conversation visible and shows generic inline error
12. Delete and Cancel are disabled while the delete mutation is in flight
13. active-conversation delete affordance is disabled during an active stream
14. active-conversation delete affordance re-enables on every Stage 9b
    terminal event (`done`, `error`, abort)
15. contract scan verifies no direct fetch calls in components
16. contract scan verifies no browser persistence is introduced

## 15. Contract Scan

Add or extend:

```text
frontend/src/stage10aContract.test.ts
```

The scan excludes:

- `frontend/src/api/openapi.json`
- `frontend/src/api/types.ts`
- `frontend/src/test/`
- `*.test.ts`
- `*.test.tsx`

### 15.1 Forbidden patterns, all application code

Applied to every file under `frontend/src/` that is not in the exclusion list:

- `localStorage`
- `sessionStorage`
- `URLSearchParams`
- direct `fetch(` outside `src/api/client.ts`
- frontend references to:
  - `deleted_at`
  - `is_deleted`
  - `archived_at`

### 15.2 Forbidden patterns, deletion and chat code paths only

Applied to files in:

- `frontend/src/features/conversations/`
- `frontend/src/features/chat/`
- `frontend/src/features/messages/`
- `frontend/src/pages/ChatPage.tsx`

Forbidden:

- `console.log`
- `console.info`
- `console.warn`
- `console.error`
- `console.debug`

This scope matches Stage 9b §12's "in any chat/streaming code path" precedent.
Broadening the `console.*` rule project-wide is out of scope for Stage 10a and
would require an explicit FRONTEND.md update.

### 15.3 Delete-dialog prop guard

Any file matching the pattern `*DeleteConversation*.tsx` under
`frontend/src/features/conversations/` (and any file it directly imports
from within `frontend/src/features/conversations/components/`) must not
import or reference, in any form:

- `MessageResponse` (from `src/api/types.ts`)
- the `content` field of any message-like type
- any messages query key, e.g. `["messages", "list", ...]`

The pattern-based match means renaming the component (e.g. to
`DeleteConversationModal.tsx`) or splitting it into sibling files
(e.g. `DeleteConversationDialogBody.tsx`) keeps the guard in force as long
as the `DeleteConversation` prefix is retained — which itself becomes a
lightweight naming convention. Implementers introducing such a split must
not rename out of the pattern; doing so would silently disable the scan.

This positively encodes the rule that the delete confirmation modal never
displays text derived from message content, and remains enforceable
mechanically even if a later stage introduces auto-titling from messages.

### 15.4 Failure output

Contract-scan failures must include:

- file path
- matched line
- the rule subsection (§15.1, §15.2, or §15.3) that fired

## 16. Validation

Backend validation from repo root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/smriti/db src/smriti/config.py
uv run pytest -q
```

Frontend validation from `frontend/`:

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

Because Stage 10a adds a backend route, `pnpm generate:api` is expected
to update generated API types.

The generated diff must be reviewed and committed.

## 17. Manual Verification

With backend, Postgres, Ollama, and frontend running locally:

1. create/select a scope
2. create two conversations
3. send messages in one conversation
4. delete the inactive conversation
5. confirm:
   - it disappears from the list
   - the active conversation remains selected
6. delete the active conversation
7. confirm:
   - the selected conversation clears
   - the Stage 9a empty state appears
8. reload the page
9. confirm deleted conversations do not reappear
10. verify browser console contains no logged message content
11. verify network tab contacts only `127.0.0.1:8000`
12. create a new conversation; begin a Stage 9b stream in it
13. confirm the active-conversation delete affordance is disabled during
    streaming
14. let the stream complete on `done`; confirm the affordance re-enables
15. begin another stream; click Stop mid-stream; confirm the affordance
    re-enables once the stream returns to `idle`
16. confirm no abort-and-delete behavior exists (i.e., no code path attempts
    deletion while a stream is active)

## 18. Exit Criteria

Stage 10a is complete when:

- `DELETE /conversations/{conversation_id}` exists
- deletion flows through MemoryService
- no route or component performs direct SQL
- hard delete is implemented
- no migration is added
- schema cascades are verified by tests
- no orphan messages, episodes, embeddings, or provenance rows remain
- unrelated conversations and unrelated episodes remain intact
- selected conversation clears when the active conversation is deleted
- deleted conversation messages cache is removed
- active-conversation deletion is disabled during streaming and re-enables
  on every Stage 9b terminal event (`done`, `error`, abort)
- no abort-and-delete behavior exists
- confirmation modal is required
- Delete and Cancel are disabled during the in-flight delete mutation
- no message content appears in delete UI
- no file matching `*DeleteConversation*.tsx` imports `MessageResponse`
  or references message content in any form (§15.3)
- no `console.*` calls exist in conversation/chat/message feature code paths
  or in `ChatPage.tsx`
- frontend and backend tests pass
- generated API types are updated
- no browser persistence, telemetry, external network calls, or direct
  component fetches are introduced

## 19. Contract Discipline

Per AGENTS.md and FRONTEND.md, the implementer must not:

- add dependencies without explicit approval
- add migrations
- introduce soft delete
- hoist Stage 9b streaming state above `ChatView`
- introduce shared/global streaming state
- implement abort-and-delete
- expose ownership distinctions between nonexistent and non-owned
  conversations
- display raw backend exceptions
- pass `MessageResponse`, message content, or messages query keys into
  any file matching `*DeleteConversation*.tsx` (the delete-dialog
  component family)
- derive delete-modal text from message content
- broaden the `console.*` ban beyond the §15.2 code paths (project-wide
  rules go through FRONTEND.md, not a stage contract)
- add browser persistence
- add direct component fetches
- introduce broad refactors

If the contract becomes ambiguous during implementation, stop and ask rather
than silently reinterpret locked decisions.
