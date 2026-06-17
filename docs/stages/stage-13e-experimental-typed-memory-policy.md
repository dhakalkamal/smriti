# Stage 13e - Experimental Retrieval Policy Contract

## Status

- Branch inspected: `kamal/retrieval`.
- Date revised: 2026-06-17.
- Mode: planning and contract only.
- Implementation status: not implemented by this document.
- Allowed edit for this revision: this contract document only.

Stage 13e defines staged experimental retrieval-policy work for assistant
generation. The new policy is implemented as an experimental runtime policy and
used for local dogfooding, while legacy remains available as a rollback and
comparison baseline.

Legacy remains the default only as a reversible rollback path and control
baseline. It is not being preserved because current retrieval is good enough.
The goal is to replace legacy if the experimental policy improves
assembled-context quality and does not regress source recall.

Legacy must remain reversible by environment flag until the experimental policy
is proven on a larger Smriti-specific golden set.

## Motivation

The current retrieval path treats every embedded episode in a scope as one
ranking pool. Raw user source messages, summary episodes, assistant-authored
or imported echoes, recap questions, scaffold/warmup records, and distractors
can all compete for the same top-k slots. Stage 12 and Stage 13 diagnostics
showed that expected evidence is often visible at wider diagnostic depth, while
weight sweeps, role replay, and lexical replay were not enough to justify
production reranking.

The known problem is broader than scoring weights:

- recent conversation is working context,
- raw user/source messages are source evidence,
- summary episodes are compressed context,
- assistant-authored memories are derived evidence,
- the active query must never be evidence for itself,
- prompt assembly determines what the model actually sees.

The current retrieval eval harness calls `retrieve_scoped_episodes` directly.
It does not exercise active-query exclusion through assistant-generation
assembly, recent-context selection, recent-context message ID exclusion, prompt
assembly, final memory inclusion/exclusion, or lane budgeting. Retrieval-only
replay is therefore useful mechanism validation, but it is not enough to
compare legacy with a new assistant-memory policy.

Stage 13e splits the work into three sub-stages so the first real retrieval
upgrade can ship and be evaluated without bundling recent-context exclusion,
typed lanes, assistant-derived lanes, token budgeting, prompt metadata changes,
access-metadata behavior, and eval changes into one implementation.

## Non-Goals

This contract revision does not include:

- implementation code,
- backend retrieval changes,
- frontend changes,
- schema changes,
- migrations,
- test changes,
- scoring weight tuning,
- production lexical reranking,
- hybrid SQL/full-text/BM25 retrieval,
- remote or public benchmark integration,
- BEIR or ARES integration,
- summary-first retrieval as a production default,
- cross-scope retrieval,
- production default flip,
- persisted lane provenance,
- removal of legacy retrieval,
- user-visible product changes,
- committing changes.

Later implementations must not add remote services, telemetry, cloud providers,
external assets, or public network exposure.

## Current Behavior From Repo Inspection

Current runtime configuration lives in `src/smriti/config.py` under the
`SMRITI_` environment prefix. There is no memory policy setting today. Existing
summary memory settings are `summary_episode_memory_enabled` and
`summary_episode_window_messages`, exposed as
`SMRITI_SUMMARY_EPISODE_MEMORY_ENABLED` and
`SMRITI_SUMMARY_EPISODE_WINDOW_MESSAGES`.

The schema already contains the type signals needed for later policy work:

- `episodes.kind` is constrained to `message` or `summary`.
- Message episodes require `message_id` and no summary range.
- Summary episodes require `message_id IS NULL` plus `range_start` and
  `range_end`.
- `messages.role` is constrained to `system`, `user`, or `assistant`.
- `episodes.scope_id` and conversation ownership enforce scoped retrieval.

Summary episode memory is implemented behind its existing summary feature flag.
Summary episodes are written as `kind = 'summary'`, have no `message_id`, store
the summarized message-position range, and are embedded into `embeddings_768`.

Assistant generation currently does this:

1. `MemoryService.load_assistant_generation_context` loads the scope prompt, the
   query message, and recent messages in the same conversation up to the query
   position.
2. `AssistantOrchestrator._prepare_generation` calls
   `MemoryService.retrieve_scoped_episodes` with `exclude_message_id` set to the
   active query message ID.
3. `retrieve_scoped_episodes` embeds the query, fetches one bounded
   similarity-first candidate pool, reranks candidates with the Stage 5.2
   weighted score, and returns one score-ordered list.
4. The retrieval query filters by `episodes.scope_id` and conversation user, but
   does not filter or partition by `episodes.kind` or `messages.role`.
5. The optional active-query exclusion excludes message episodes whose
   `message_id` matches the active query. Summary episodes are preserved because
   the predicate uses `IS DISTINCT FROM`.
6. Retrieval updates `access_count` and `last_accessed_at` for the returned
   top-k episodes.
7. `build_chat_request` emits one mixed set of `Memory context` system messages
   in score order, then recent messages in their original roles.
8. Prompt budgeting is character-based through `max_prompt_chars`, defaulting to 16000. There is no reliable token budget model today.
9. Used-memory provenance records selected episode IDs, ranks, scores, scoring
   version, and query/assistant linkage. It does not record a lane.

Generated assistant responses are persisted as assistant messages without
creating retrieval episodes or embeddings. However, message episodes can still
have `messages.role = 'assistant'` if they are created through other supported
message-writing paths, imported data, direct fixtures, or future tools. Such
episodes must be treated as derived evidence, not source evidence.

`messages.token_count` is client-supplied and is not authoritative. It can be
zero, missing, stale, or arbitrary. Episodes do not have a reliable token field.
Any future token budgeting must therefore use a deterministic content-derived
estimator that works for both message content and episode content.

The eval harness already has diagnostic vocabulary for this policy:
`raw_source`, `summary_source`, `assistant_answer_echo`, `recap_question`,
`scaffold`, `distractor`, `current_query`, source-vs-derived metrics,
diagnostic top-k, and role/lexical replay. Those diagnostics are eval-only
today, and the current harness does not run the assistant memory assembly path.

## Runtime Policy Contract

A later implementation may add a memory policy setting such as:

```text
SMRITI_MEMORY_POLICY=legacy
SMRITI_MEMORY_POLICY=typed_v1
```

Contract:

- Default: `legacy`.
- Valid values must be explicit and configuration-validated before serving
  requests.
- Absence of the setting means `legacy`.
- Changing the setting and restarting the local runtime must fully reverse the
  policy.
- `legacy` must preserve current behavior, including the single mixed retrieval
  pool, current score formula, active-query exclusion, access-metadata behavior,
  character budget behavior, and existing prompt ordering.
- Experimental policies must not mutate production defaults when the flag is
  absent.
- Experimental policies are for local dogfooding and eval comparison until
  golden-set evidence supports a default flip.
- The policy flag must not require schema changes, migrations, frontend changes,
  or database backfill.
- The policy must remain local-only and must not introduce outbound network
  behavior beyond the existing local Ollama calls.

If a single `typed_v1` value is introduced before all sub-stages are complete,
its documented behavior must match only the implemented sub-stage. It must not
claim typed lane selection or token-aware budgeting until Stage 13e-2 or Stage
13e-3 has actually been implemented and validated.

Runtime assistant generation may use `SMRITI_MEMORY_POLICY`. Eval tooling must
also accept an explicit policy argument so legacy and experimental policies can
be compared side-by-side without relying only on environment changes.

## Stage 13e-1 - Recent-Context Exclusion + Assembly-Aware Eval

Stage 13e-1 is the first real retrieval upgrade. It is production-eligible only
after eval validation, and legacy remains available as rollback/control.

Scope:

- preserve the existing single mixed long-term retrieval pool,
- preserve existing scoring,
- preserve the existing mixed memory block for now,
- preserve existing character-budget behavior as much as possible,
- build or select recent conversation context explicitly,
- ensure the active query appears exactly once in the final chat request,
- exclude the active query from long-term message retrieval,
- exclude final selected recent-context message IDs from long-term message
  retrieval,
- add an assembly-aware eval path that exercises recent-context selection,
  exclusions, final prompt inclusion, and prompt assembly.

Stage 13e-1 must not add typed lane quotas, assistant-derived lane behavior, or
token-aware budgeting. Those belong to later sub-stages.

### Recent-Context Exclusion

The exclusion set for long-term message retrieval is computed from the final
selected recent context, not merely from the larger loaded recent-message
window.

Mandatory exclusions:

- Active-query exclusion: the active query message ID is excluded from message
  episode retrieval.
- Recent-context exclusion: every message ID selected for final recent context
  is excluded from long-term message episode retrieval.

The exclusion set applies to message episodes by `episodes.message_id`.
Summary episodes require range-aware handling because they do not have
`message_id`.

Summary overlap handling for 13e-1:

- fully covered summary range exclusion is preferred when it is cheap and
  reliable to determine,
- partial-overlap summary exclusion is out of scope,
- summary overlap logic must not block the first implementation.

### Active Query Behavior

The active query must appear exactly once in the final chat request.

It must never be:

- silently dropped,
- treated as optional memory,
- included in long-term memory as evidence for itself,
- duplicated as both the user turn and a retrieved memory item.

If mandatory sections exceed budget, the implementation must fail loudly or use
existing legacy error behavior. Optional recent context and long-term memory
must shrink before the active query is affected.

### Assembly-Aware Eval Requirement

Stage 13e-1 must introduce or specify an eval path that runs the assistant
memory assembly pipeline, not just `retrieve_scoped_episodes`.

The eval path must exercise:

- policy selection,
- recent-context selection,
- active-query exclusion,
- recent-context message ID exclusion,
- candidate retrieval,
- final prompt inclusion/exclusion,
- prompt assembly order,
- content-free diagnostics.

It must emit enough diagnostics to compare:

- legacy vs Stage 13e-1,
- self-query hit rate,
- recent-context duplication rate,
- source evidence inclusion,
- final prompt memory composition,
- skipped candidates and exclusion reasons.

Diagnostics must be content-free by default: IDs, roles, kinds, ranks, scores,
counts, labels, and reasons are acceptable; prompt text, message text, and
memory text are not acceptable at INFO or below.

## Stage 13e-2 - Typed Lane / Quota Selection, Eval-First

Stage 13e-2 introduces typed or quota-based selection after Stage 13e-1 proves
the assembly-aware path and recent-context exclusion.

Scope:

- raw source messages, summaries, and assistant-derived memories are selected
  using typed lanes or composition quotas,
- selection separates `episodes.kind` and `messages.role`,
- raw/source, summary, and assistant-derived categories are separated in eval,
- assistant-derived memories remain default top 0 unless explicitly enabled,
- assistant-derived memories must not count as source evidence,
- the stage remains eval-first until golden-set evidence supports making it the
  default.

Preferred implementation shape:

- perform one candidate fetch and partition in Python by `episodes.kind` and
  `messages.role`,
- avoid per-lane retrieval calls that multiply access-metadata side effects,
- apply quotas after exclusions and before final prompt inclusion,
- preserve scope/user filters for every candidate.

The lane names below describe policy roles, not required schema fields.

### `recent_context`

Source: persisted `messages` in the active conversation, ordered by position and
ending at the active query message.

Purpose: working conversational context, not long-term memory retrieval.

Requirements:

- always include the active query unless mandatory sections exceed budget and
  the implementation fails loudly or uses existing legacy error behavior,
- preserve message roles exactly as stored,
- compute long-term message exclusions from final selected recent-context
  message IDs.

### `raw_source_messages`

Source: `episodes.kind = 'message'` joined to `messages`, with
`messages.role = 'user'` by default.

Purpose: primary source evidence.

Requirements:

- exclude the active query message ID,
- exclude all final selected recent-context message IDs,
- preserve existing scope/user filters and scoring components,
- exclude system-authored message episodes unless a future contract gives them
  a source-evidence role.

Recap questions are often user-role messages. Older recap questions beyond the
selected recent context can still enter this category. Typed/quota selection
does not fully solve durable recap/question labeling; durable recap labeling or
weak-evidence classification remains future work.

### `rolling_summaries`

Source: `episodes.kind = 'summary'`.

Purpose: compressed context for broad or recap-style questions.

Requirements:

- keep summaries distinct from raw source messages in prompt labels and
  diagnostics,
- exclude fully covered summary ranges when it is cheap and reliable,
- allow partial overlaps initially,
- do not treat summaries as a replacement for raw-source-required evidence.

The lane name is `rolling_summaries` for policy clarity even though the current
Stage 11 implementation creates fixed complete-window summary episodes.

### `assistant_derived`

Source: `episodes.kind = 'message'` joined to `messages`, with
`messages.role = 'assistant'`.

Purpose: optional derived evidence, useful as a clue but not as source evidence.

Requirements:

- default top 0 unless explicitly enabled for the experiment,
- exclude selected recent-context message IDs,
- label every included item as derived assistant memory,
- never satisfy source-only retrieval metrics.

If assistant-derived memories are included in prompts, the prompt must tell the
model they are derived and may be wrong, stale, or merely a prior assistant
restatement.

## Stage 13e-3 - Token-Budget Model, Eval-First

Stage 13e-3 introduces token-aware budgeting after the assembly-aware eval path
and any typed/quota selection have been validated separately.

Scope:

- introduce deterministic content-derived token estimation,
- centralize the estimator and test it directly,
- make the estimator work for both messages and episode content,
- convert prompt and lane budgeting from character-based to token-aware,
- keep character limits only as a defensive secondary guard if needed,
- keep this separate from Stage 13e-1 and Stage 13e-2.

`messages.token_count` must not be used as the authoritative budgeting basis.
It may be carried for compatibility or diagnostics, but it cannot decide prompt
admission. Episodes do not have a reliable token field.

Stage 13e-1 should keep existing character budgeting unless a later
implementation explicitly adds the Stage 13e-3 estimator.

## Prompt Metadata Contract

New policies should reduce prompt noise, not increase it.

Prompt-visible labels should be minimal:

- lane or evidence type,
- source-vs-summary-vs-derived distinction,
- concise context labels needed for model behavior.

Normal prompt text should not include internal IDs, ranks, scores, score
components, token estimates, or database metadata. Those fields may be included
only when an explicit debug mode is enabled for a controlled eval/debug surface.

Debug and eval metadata should include:

- `policy`,
- `lane`,
- `episode_id` when applicable,
- `message_id` when applicable,
- `message_role` for message episodes,
- `episode_kind`,
- rank,
- score,
- score components,
- token estimate,
- lane budget status,
- final prompt inclusion status,
- exclusion or skip reason.

Full IDs, ranks, scores, and score components belong in debug/eval output, not
normal model prompt text.

## Access-Metadata Side-Effect Rule

`retrieve_scoped_episodes` currently updates `access_count` and
`last_accessed_at` for returned top-k episodes.

New policies must not reinforce fetched-but-skipped candidates. Access metadata
should update only for episodes admitted to the final prompt.

If this cannot be guaranteed in a given sub-stage, the implementation must:

- document the side effect,
- preserve legacy behavior for `legacy`,
- avoid adding additional reinforcement beyond the legacy path.

Stage 13e-2 should prefer a single candidate fetch followed by Python
partition/quota selection to avoid multiplying access updates through separate
per-lane retrieval calls.

Eval/debug candidate inspection must not quietly update production access
metadata unless the run is explicitly intended to exercise production side
effects.

## Budget Overflow Contract

Stage 13e-1 should preserve existing character-budget behavior as much as
possible.

For later typed/token stages:

- mandatory sections fail loudly rather than being truncated silently,
- optional recent context and optional long-term memory shrink before mandatory
  sections are affected,
- optional memory overflow must be deterministic and tested,
- if a candidate would exceed its lane or total memory cap, the implementation
  must define whether it skips that candidate or stops the lane,
- the chosen skip/stop behavior must be stated clearly and tested,
- prompt assembly must reserve answer room before admitting optional memory.

Logs at INFO or below must not include prompt text, message contents, or memory
contents.

## Golden-Set Gate

The existing Terrafold 9-case corpus is useful for mechanism validation. It is
not enough to justify flipping production defaults.

Before making a new retrieval policy the default, build a larger
Smriti-specific golden set, roughly 100-200 questions. A 100-150 question set
may be enough for an initial default-flip discussion if it is diverse and the
results are strong, but the target should leave room for more coverage.

The golden set should be:

- multi-persona,
- multi-scope,
- scoped to realistic local-memory use,
- explicit about source evidence requirements.

It should include:

- specific fact recall,
- broad continuation,
- summary-answerable cases,
- raw-source-required cases,
- recent-context-only cases,
- assistant-echo traps,
- distractors,
- update/correction cases,
- negative controls.

TODO for the next golden-set revision:

- include cases where `acceptable != source`;
- include summary-only evidence that is acceptable;
- include derived assistant echoes that look answer-correct but must still score
  as source misses;
- include no-answer / abstention behavior;
- include temporal reasoning;
- include recent-context-only answers;
- include older long-term-memory answers.

Current Terrafold caveat: `source_hit_at_k` and `acceptable_hit_at_k` can be
numerically equivalent because acceptable evidence currently collapses mostly to
raw plus summary source evidence. It is still useful for mechanism validation,
but it under-stresses source-vs-derived separation.

Later statistical decision rule:

- use paired comparisons;
- use McNemar for binary outcomes;
- use Wilcoxon signed-rank for MRR deltas;
- report Wilson confidence intervals and per-category breakdowns;
- require primary `MRR(source)` improvement with a paired confidence interval
  excluding zero;
- require per-type recall to be non-inferior;
- require `self_query_hit_rate == 0`;
- require no over-retrieval increase on no-answer / abstention cases.

Later answer-quality protocol:

- use deterministic normalized exact match or token-F1 for factual answers;
- use a local LLM judge only for open-ended answers;
- validate local-judge output against a human subset and report agreement.

## Retrieval Metric Caveat

Retrieval metric gains are necessary but not sufficient for product quality.
Answer quality is not fully measured by the current retrieval eval.

Source recall, self-query avoidance, duplication rate, and prompt composition
must improve or hold steady before a policy can become default. Answer-level
evaluation can be a later stage.

## Backward Compatibility

Backward compatibility is mandatory:

- `legacy` remains the default rollback/control baseline.
- Absence of `SMRITI_MEMORY_POLICY` means `legacy`.
- Existing tests for retrieval, assistant generation, SSE, summary memory,
  provenance, and config must continue to pass under `legacy`.
- The active-query exclusion remains in assistant generation for both policies.
- Existing `/retrieval/search` behavior remains unchanged unless a later
  contract explicitly adds policy-aware debug search.
- Existing provenance rows remain readable.
- No schema changes or migrations are required.
- No frontend changes are required.
- The policy must be reversible by changing one env flag and restarting.

## Acceptance Criteria

### Stage 13e-1

- Legacy remains available as rollback/control.
- The experimental policy can be enabled for local dogfooding and eval.
- The active query appears exactly once in the final chat request.
- The active query is excluded from long-term message retrieval.
- Final selected recent-context message IDs are excluded from long-term message
  retrieval.
- Existing single-pool scoring remains.
- Existing mixed memory block remains.
- Existing character-budget behavior is preserved as much as possible.
- Assembly-aware eval path exists.
- Eval accepts an explicit policy argument.
- Diagnostics show final included memory vs skipped/excluded candidates.
- Diagnostics include self-query hit rate, recent-context duplication rate,
  source evidence inclusion, and final prompt memory composition.
- No schema changes are required.
- No migrations are required.
- No frontend changes are required.
- Legacy tests remain green.

### Stage 13e-2

- Typed/quota selection uses one candidate fetch or otherwise avoids
  access-metadata inflation.
- Raw/source, summary, and assistant-derived categories are separated in
  selection and eval.
- Assistant-derived memories default to top 0 unless explicitly enabled.
- Assistant-derived memories do not count as source evidence.
- Access metadata updates only for final included episodes, or any unavoidable
  legacy-compatible side effect is documented.
- The stage remains eval-first until golden-set evidence supports a default
  flip.

### Stage 13e-3

- Deterministic content-derived token estimator exists.
- `messages.token_count` is not authoritative for budgeting.
- The estimator works for both messages and episode content.
- The estimator is centralized and tested.
- Prompt/lane budgeting is token-aware.
- Budget overflow behavior is deterministic and tested.
- Mandatory sections fail loudly rather than being silently truncated.

## Risks and Open Questions

- The current prompt builder is character-budgeted and mixed-memory oriented.
  Stage 13e-1 should work with that constraint rather than force a token-budget
  rewrite.
- Existing `retrieve_scoped_episodes` updates access metadata for returned
  results. A new policy may need a candidate-read mode or a stricter admission
  boundary before access updates can perfectly match final prompt inclusion.
  Stage 13e-1 still uses one retrieval call and avoids multi-lane reinforcement,
  but access writes can include retrieved candidates later skipped by prompt
  character budgeting. Stage 13e-2 should add single-fetch-then-partition or a
  read-only candidate mode so access writes become a subset of admitted memory.
- Summary overlap exclusion requires mapping selected recent-context positions
  to summary `range_start..range_end`. Fully covered summaries are cheap enough
  to prefer; partial overlap should wait for evidence.
- Assistant-derived memories can help phrase an answer but can also reinforce
  prior assistant mistakes. Default top 0 is safest unless product behavior
  demands otherwise.
- Recap questions are not durably labeled in production today. Without
  metadata, they cannot be cleanly separated from ordinary user source messages.
- The manual retrieval inspection endpoint may confuse users if assistant
  generation uses an experimental policy while `/retrieval/search` shows legacy
  order. A later UX/debug contract may be needed.
- Provenance currently has no lane field. Eval-only lane output is enough for
  this staged experiment; persisted lane provenance remains out of scope.

## Suggested Later Implementation Sequence

1. Stage 13e-1: add a `memory_policy` setting with `legacy` default and explicit
   validation.
2. Stage 13e-1: preserve the current assistant path as the `legacy`
   implementation.
3. Stage 13e-1: build/select recent context explicitly and compute final recent
   message IDs.
4. Stage 13e-1: apply active-query and recent-context message exclusions to the
   existing single long-term retrieval pool.
5. Stage 13e-1: add assembly-aware eval with content-free diagnostics and an
   explicit policy argument.
6. Stage 13e-1: compare legacy and experimental assembly output on the existing
   corpus, treating the corpus as mechanism validation only.
7. Stage 13e-2: add typed/quota selection using one candidate fetch plus Python
   partitioning where possible.
8. Stage 13e-2: keep assistant-derived memories disabled by default and report
   source-only vs source-plus-derived metrics separately.
9. Stage 13e-2: expand the Smriti-specific golden set before considering a
   default flip.
10. Stage 13e-3: add the deterministic content-derived token estimator and
    tests.
11. Stage 13e-3: convert prompt/lane budgets from character-based to
    token-aware and test deterministic overflow behavior.

The later implementation should remain small, staged, and reversible. If the
experimental policy does not improve assembled-context quality without
regressing source recall, keep it experimental and leave `legacy` as the
default rollback/control baseline.
