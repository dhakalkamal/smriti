# Stage 14 - Hybrid Retrieval Candidate Quality Contract

## Status

- Branch inspected: `kamal/retrieval`.
- Date drafted: 2026-06-18.
- Mode: contract and design only.
- Implementation status: not implemented by this document.
- Allowed edit for this drafting task: this contract document only.

Stage 14 defines the next retrieval experiment: improve long-term memory
candidate discovery with a local Postgres-native hybrid lexical plus semantic
candidate set, while preserving the Stage 13e prompt-assembly and typed
admission guarantees.

This document does not implement code, add migrations, change runtime defaults,
or commit changes.

## Problem Statement

Stage 13e improved prompt hygiene and introduced typed memory admission, but the
candidate pool is still semantic-first. Vector retrieval can miss or under-rank
memories whose distinguishing evidence is lexical:

- email addresses such as `hello@terrafold.studio`,
- exact money amounts such as `GBP 48` or `GBP 200`,
- dates such as `September 13`,
- multi-token names such as `Wedgwood & Vyne`,
- numeric constraints such as `17 working days`,
- rare names and relationship terms such as `Dele` and `bookkeeping`,
- material constraints such as `latex` and `nitrile`.

The next likely bottleneck is therefore candidate discovery and ranking before
typed admission. Stage 14 should test whether a hybrid lexical plus semantic
candidate set improves assembled-prompt evidence quality without weakening the
Stage 13e invariants.

## Current-State Evidence From Inspection

The current retrieval implementation has these relevant properties:

- `MemoryService.retrieve_scoped_episodes(...)` embeds the query, fetches a
  bounded vector-similarity candidate pool from `embeddings_768`, then reranks
  candidates in Python with the Stage 5.2 weighted score.
- The SQL candidate path orders by pgvector cosine similarity using
  `embeddings_768.embedding <=> $query_vector`.
- The final Python score combines similarity, recency, access reinforcement,
  importance, and frequency.
- The SQL filters by `episodes.scope_id` and conversation `user_id`.
- Active/recent message exclusion is applied to message episodes by
  `episodes.message_id`, while summaries remain eligible because their
  `message_id` is `NULL`.
- The schema has HNSW vector indexing and ordinary relational indexes. It has no
  full-text index, no `tsvector` column, no `pg_trgm` extension, and no trigram
  index.
- `typed_v1` currently fetches candidates read-only, partitions them in Python,
  admits raw/source, summary, and assistant-derived lanes, then updates access
  metadata only for prompt-selected memories.
- `typed_v1` defaults to a total limit of 6, raw source limit 4, summary source
  limit 2, and assistant-derived limit 0.
- Legacy remains the default runtime policy and the rollback/control baseline.

Stage 13e-1 and Stage 13e-2 verified the prompt-assembly boundary on the
nine-case Terrafold mechanism set:

- active query exactly once: `9/9 = 1.0000`,
- assembled self-query hit rate: `0/9 = 0.0000`,
- recent-context duplication rate: `0.0000`,
- summaries with `message_id IS NULL` remain eligible.

Stage 13e-2 comparison on the same nine cases:

| Policy | Source Hits | Raw Hits | Summary Hits | Mean Source nDCG |
| --- | ---: | ---: | ---: | ---: |
| legacy | 6/9 | 3/9 | 6/9 | 0.3248 |
| typed_v1 | 7/9 | 3/9 | 7/9 | 0.3528 |

Interpretation:

- `typed_v1` recovered one source-summary case:
  `terrafold_f5_dele_bookkeeping`.
- Raw source recall did not improve.
- Prompt hygiene remained correct.
- False-positive and over-retrieval behavior did not justify a production
  default flip.
- Because the corpus has only nine cases, one changed case is about 11.1
  percentage points. Stage 14 reports both rates and changed-case counts.

Stage 13c lexical replay showed useful mechanism evidence but not a production
policy:

- best acceptable hit rate moved from `5/9` to `7/9`,
- raw hit rate stayed `5/9`,
- summary hit rate moved from `5/9` to `7/9`,
- Dele/bookkeeping improved from first acceptable rank 12 to rank 7 under the
  strongest lexical profiles, still outside official `top_k = 5`.

That evidence argues for hybrid candidate discovery, not for replacing typed
admission or making lexical ranking the production default.

## Design Decision

Stage 14 should add an experimental candidate mode, not a new memory admission
policy.

Recommended configuration shape for a later implementation:

```text
SMRITI_MEMORY_POLICY=typed_v1
SMRITI_RETRIEVAL_CANDIDATE_MODE=semantic
SMRITI_RETRIEVAL_CANDIDATE_MODE=hybrid_v1
```

Contract:

- `semantic` preserves the existing vector-first candidate path.
- `hybrid_v1` generates semantic and lexical candidates, fuses them into one
  ranked candidate list, and then passes that one list into unchanged
  `typed_v1` admission.
- Absence of the candidate-mode setting means existing behavior.
- Legacy remains available as rollback/control.
- Hybrid is not a production default from the nine-case Terrafold set alone.

The implementation may choose a different setting name, but it must keep
candidate generation separable from typed admission so Stage 14 can compare:

- `typed_v1` semantic-only,
- `typed_v1` hybrid lexical plus semantic.

## Lexical Candidate Path

Stage 14 should use PostgreSQL full-text search as the primary lexical path,
with a small exact-anchor companion for identifiers and punctuation-heavy facts.

Recommended lexical shape:

1. Build a deterministic query analysis from the current user query. This may
   reuse the Stage 13 lexical token ideas, but it must not depend on eval-only
   provenance labels.
2. Use PostgreSQL full-text search over `episodes.content` with the `simple`
   configuration. `simple` is preferred because names, rare terms, and local
   memory tokens should not be stemmed away or treated as English prose only.
3. Rank lexical candidates with `ts_rank` or `ts_rank_cd`, plus deterministic
   exact-anchor bonuses for:
   - email-like strings,
   - numbers and currency-like amounts,
   - simple date strings,
   - quoted phrases,
   - rare tokens and multi-token proper-name phrases.
4. Use lexical rank for fusion. Do not treat the absolute lexical score as
   calibrated against vector similarity.

This design is local-first and uses the existing Postgres service. It does not
introduce a cloud search service, external search daemon, neural reranker, or
frontend feature.

### Why Not BM25 First

Postgres core full-text ranking is not BM25. A true BM25 implementation would
require either an additional extension or a custom ranking layer. That is too
large for the next smallest local-first retrieval step, and it would add
deployment and migration risk before proving that lexical candidate discovery
helps at the assembled-prompt layer.

Stage 14 may describe `ts_rank` as a lexical rank signal, not as BM25.

### Why Not Trigram First

`pg_trgm` is useful for fuzzy typo tolerance, but the current evidence points
first at exact anchors: names, emails, numbers, dates, and rare terms. Trigram
search also requires an extension and index migration, can widen false positives,
and does not by itself solve candidate fusion.

Trigram support should wait until the larger golden set contains typo, spelling
variant, or fuzzy-name cases that full-text plus exact anchors cannot handle.

### Why Not Phrase Matching Alone

Phrase matching is useful as an exact-anchor booster, especially for quoted
strings and multi-token names. It should not be the primary lexical retrieval
path because many memory queries paraphrase the relationship while preserving
only a few anchors.

## Candidate Generation

Hybrid candidate generation must produce one logical candidate boundary before
typed admission.

Inputs:

- `user_id`,
- `scope_id`,
- query text,
- active/recent `exclude_message_ids`,
- candidate depth,
- current embedding model.

Semantic path:

- Use the existing vector embedding path.
- Preserve the existing scope/user filters.
- Preserve current scoring components for semantic rank.
- Run read-only when used by `typed_v1` or eval, so fetched-but-skipped
  candidates are not reinforced.

Lexical path:

- Query the same `episodes` and `conversations` ownership boundary.
- Apply the same active/recent message exclusion predicate as the semantic
  path.
- Preserve summary eligibility.
- Join `messages` only for durable role metadata needed by typed admission and
  debug output.
- Return the same durable episode fields required to construct the candidate
  records.

Both paths must be content-local and Postgres-local. They must not call remote
services or log message content at INFO or below.

## Candidate Depth

The fused candidate list must be deeper than final prompt admission.

Default Stage 14 depth:

```text
hybrid_candidate_depth = max(
    25,
    request.top_k * 5,
    memory_typed_v1_total_limit * 4,
)
```

Initial implementation should cap this at a conservative local value, such as
100, until performance is measured. The exact cap may change with evidence, but
the contract requires the cap and depth to be explicit in config or metadata.

For the default `typed_v1` limit of 6 and normal `top_k = 5`, this yields 25
fused candidates before admission. That matches the diagnostic depth already
used in Stage 13 runs and gives lexical retrieval room to recover exact-anchor
evidence without changing the prompt memory limit.

Eval runs may use a wider diagnostic depth, but official assembled-prompt
metrics must still score the final prompt-admitted memories, not raw retrieval
output alone.

## Fusion Behavior

Stage 14 should merge semantic and lexical candidates with reciprocal rank
fusion.

Rationale:

- vector similarity, Stage 5.2 weighted score, `ts_rank`, and exact-anchor
  bonuses are not calibrated on the same scale;
- rank fusion is transparent, deterministic, and easy to debug per candidate;
- it avoids pretending a lexical score of `0.2` means the same thing as a vector
  similarity or weighted retrieval score;
- it can reward candidates that are strong in either channel while still
  favoring candidates that appear in both.

Recommended fusion:

```text
semantic_contribution = semantic_weight / (rrf_k + semantic_rank)
lexical_contribution = lexical_weight / (rrf_k + lexical_rank)
hybrid_score = semantic_contribution + lexical_contribution
```

Defaults for the first experiment:

- `semantic_weight = 1.0`,
- `lexical_weight = 1.0`,
- `rrf_k = 60`.

Tie-breakers:

1. higher hybrid score,
2. candidate present in both channels,
3. better semantic rank when present,
4. better lexical rank when present,
5. newer `episodes.created_at`,
6. stable `episodes.id`.

The exact numeric defaults may be tuned only through assembly-aware eval. The
first implementation should keep them simple and visible in run metadata.

## Duplicate Handling

Hybrid candidate generation must dedupe only identical episodes.

Rules:

- If the same `episode_id` appears in semantic and lexical results, keep one
  candidate and attach both channel ranks/scores in debug metadata.
- If a raw message episode and a summary episode support the same fact, they
  remain separate candidates. Runtime cannot use eval-only provenance labels to
  collapse them.
- If a summary overlaps recent context, preserve the Stage 13e behavior:
  message exclusions must not accidentally exclude summaries. Fully covered
  summary-range exclusion may be handled by a separate contract if needed.
- Prompt redundancy between raw and summary evidence is a later reranker or
  prompt-budget problem, not a Stage 14 candidate-generation shortcut.

## Typed Admission Boundary

Stage 14 must preserve `typed_v1` admission unless a small interface change is
needed to pass a fused candidate list.

Required behavior:

- one fused candidate list enters `apply_typed_memory_admission`;
- raw source limit remains 4 by default;
- summary source limit remains 2 by default;
- assistant-derived limit remains 0 by default;
- source-only backfill remains available up to total 6;
- lane classification still uses durable runtime metadata:
  `episodes.kind` and `messages.role`;
- eval-only labels such as `raw_source`, `summary_source`, `recap_question`, or
  `assistant_answer_echo` must not drive runtime filtering.

Access metadata:

- Hybrid candidate fetch for `typed_v1` must be read-only.
- Only final prompt-selected memories may update `access_count` and
  `last_accessed_at`.
- Fetched lexical-only candidates that are skipped by typed admission or prompt
  character budget must not be reinforced.

## Preserved Stage 13e Invariants

Hybrid must preserve:

- active query exactly once in final prompt assembly;
- active query excluded from long-term message retrieval;
- recent-context message IDs excluded from long-term message retrieval;
- summaries with `message_id IS NULL` eligible for retrieval;
- one logical candidate-fetch/admission boundary before typed admission;
- read-only candidate fetch for `typed_v1`;
- admitted-only access reinforcement for `typed_v1`;
- assistant-derived memories disabled by default;
- legacy as rollback/control;
- local-only operation and no hidden outbound network behavior.

Both semantic and lexical SQL paths must apply the same scope, user, and
message-exclusion predicates. Any implementation that cannot prove this with
tests should not ship.

## Data and Index Implications

No destructive schema change is needed.

The first Stage 14 evaluation can be implemented without a migration by using a
Postgres expression in the lexical query:

```sql
to_tsvector('simple', episodes.content)
```

However, a normal runtime dogfood path should likely add one append-only
expression index before broad use:

```sql
CREATE INDEX IF NOT EXISTS idx_episodes_content_fts_simple
    ON episodes USING gin (to_tsvector('simple', content));
```

This avoids a stored `tsvector` column and avoids content backfill. The existing
`idx_episodes_scope_id` index can still participate in scoped filtering.

Stage 14 should not add:

- a persisted `tsvector` column,
- a destructive migration,
- external search services,
- `pg_trgm`,
- BM25/search extensions,
- new frontend state.

If eval proves hybrid useful and performance is acceptable only with the GIN
expression index, that append-only index migration is justified for Stage 14
implementation. If eval does not prove utility, no index should be added.

## Debug and Eval Observability

Hybrid debug output must remain content-free by default.

Per candidate, debug/eval metadata should include:

- `episode_id`,
- `message_id` when present,
- `episode_kind`,
- `message_role` when present,
- fused rank,
- hybrid score,
- semantic rank and semantic score when present,
- lexical rank and lexical score when present,
- lexical match family flags such as `fts`, `email`, `number`, `date`,
  `phrase`, or `rare_token`,
- typed lane,
- admission decision,
- skip reason,
- final prompt inclusion status.

Per run, metadata should include:

- memory policy,
- retrieval candidate mode,
- semantic depth,
- lexical depth,
- fused candidate depth,
- RRF parameters,
- lexical query strategy,
- whether an FTS index is present,
- embedder mode and embedding model,
- max prompt chars,
- recent message limit.

Logs at INFO or below must not include prompt text, message text, or memory
content.

## Eval Comparison Plan

Stage 14 must use the existing assembly-aware runner path, not only raw
retrieval output.

Minimum comparison:

- `typed_v1` semantic-only,
- `typed_v1` hybrid lexical plus semantic.

Recommended later CLI shape:

```text
uv run python scripts/run_retrieval_assembly_eval.py \
  --memory-policy typed_v1 \
  --retrieval-candidate-mode semantic \
  --diagnostic-top-k 25

uv run python scripts/run_retrieval_assembly_eval.py \
  --memory-policy typed_v1 \
  --retrieval-candidate-mode hybrid_v1 \
  --diagnostic-top-k 25
```

The exact flag name may differ, but the run metadata must make the candidate
mode explicit.

Required aggregate metrics:

- `assembled_context_metrics.source_raw_hit_rate_at_k`,
- `assembled_context_metrics.source_summary_hit_rate_at_k`,
- `assembled_context_metrics.source_hit_rate_at_k`,
- `assembled_context_metrics.mean_source_ndcg_at_k`,
- `false_positive_retrieval_rate`,
- `over_retrieval_rate`,
- `active_query_exactly_once_rate`,
- `recent_context_duplication_rate`,
- assistant-derived admitted count,
- candidate and assembled self-query hit rate.

Required per-case reporting:

- semantic-only versus hybrid source hit change,
- raw hit change,
- summary hit change,
- first source/raw/summary rank movement at the assembled layer,
- first source/raw/summary rank movement in the fused candidate list,
- lexical-only recovered candidates,
- semantic-only candidates lost from the fused candidate depth,
- assistant-derived admitted count,
- negative-control admitted count.

Interpretation rule:

- On the nine-case Terrafold set, report both counts and rates.
- Do not describe one changed case as a broad percentage improvement.
- If hybrid improves only summary hits while raw source hits remain unchanged,
  say that plainly.
- If mean nDCG improves while hit counts do not, report rank-quality improvement
  rather than recall improvement.

## Acceptance Criteria

Stage 14 implementation is acceptable for experimental local dogfooding only if
all of the following hold.

Functional gates:

- Legacy remains available and unchanged.
- Semantic-only `typed_v1` remains available as a control.
- Hybrid mode is opt-in.
- Hybrid candidate fetch is read-only before typed admission.
- Access metadata updates only for final prompt-selected memories.
- No frontend feature is added.
- No remote service or outbound network behavior is added.
- No destructive schema change is added.

Invariant gates:

- `active_query_exactly_once_rate = 1.0000`.
- Candidate and assembled self-query hit rates remain `0.0000`.
- Recent-context duplication does not increase from semantic-only `typed_v1`.
- Assistant-derived admitted count remains 0 with default settings.
- Summary episodes with `message_id IS NULL` remain eligible.

Quality gates on the nine-case Terrafold mechanism set:

- Assembled source hit count must not decrease from semantic-only `typed_v1`.
- Assembled raw source hit count must not decrease from semantic-only
  `typed_v1`.
- `mean_source_ndcg_at_k` must not decrease from semantic-only `typed_v1`.
- `false_positive_retrieval_rate` and `over_retrieval_rate` must not increase.
- Per-case regressions must be listed even if aggregates improve.

Evidence gate for calling Stage 14 useful:

- At least one source case improves at the assembled-prompt layer, or
  `mean_source_ndcg_at_k` improves with no hit-count regression.
- The report names the changed case count, for example `1/9`, and does not rely
  only on percentage deltas.

Default-flip gate:

- The nine-case Terrafold set is not sufficient for a production default flip.
- Hybrid can become a default only after the larger 100-200 question golden set
  shows paired improvement without source-recall regressions.

## Failure Modes and Rollback

Failure modes:

- lexical path retrieves exact-token distractors and increases false positives;
- FTS tokenization misses punctuation-heavy anchors such as emails or currency;
- lexical candidates crowd out semantically relevant summaries;
- summary hits improve while raw source evidence remains weak;
- recap questions enter as user-role raw-source candidates because recap labels
  are eval-only;
- larger candidate depth increases local query latency;
- candidate fusion changes rank order enough to reduce typed lane quality;
- debug output accidentally exposes content.

Rollback:

- Set retrieval candidate mode back to `semantic`, or remove the opt-in setting.
- Set memory policy back to `legacy` for the broader rollback/control path.
- Because candidate fetch is read-only before admission, failed hybrid eval runs
  should not reinforce skipped candidates.
- If an optional FTS index migration has been added, it is non-destructive and
  can remain unused while the runtime mode is disabled.

## Deferred to a Later Reranker Stage

Stage 14 must not add a neural reranker.

Defer:

- local cross-encoder or LLM reranking,
- learned rank fusion or score calibration,
- BM25/search-extension adoption,
- trigram fuzzy matching,
- typo-tolerant retrieval,
- durable recap-question classification,
- assistant-echo demotion beyond the existing assistant-derived lane default,
- raw/summary redundancy suppression,
- parent-child summary expansion,
- token-aware prompt budgeting,
- answer-quality judging.

A later reranker stage should consume a stronger hybrid candidate set and decide
which evidence is most useful, not solve candidate discovery and reranking in
one change.

## Deferred to the Larger Golden Set

The 100-200 question Smriti golden set must wait for a later stage or a separate
contract. It should be used before any default flip and should include:

- multi-scope cases,
- direct exact-anchor recall,
- broad recap and summary-answerable cases,
- raw-source-required cases,
- recent-context-only cases,
- assistant-echo traps,
- recap-question traps,
- negative controls,
- corrections and temporal updates,
- typos or spelling variants if trigram/fuzzy retrieval is being considered.

Stage 14 may prepare eval output shapes that make this larger set easier to
compare, but it must not claim production readiness from the nine Terrafold
cases alone.

## Suggested Implementation Sequence

1. Add an opt-in candidate mode with existing semantic behavior as the default.
2. Extract or add a read-only candidate fetch path that can return enough
   durable metadata for fusion and typed admission.
3. Add a Postgres full-text lexical candidate query with exact-anchor matching.
4. Fuse semantic and lexical candidates with reciprocal rank fusion.
5. Feed the fused list into unchanged `typed_v1` admission.
6. Preserve admitted-only access metadata updates.
7. Extend assembly-aware eval metadata and reports for candidate mode, channel
   ranks, fusion metadata, and assistant-derived admitted counts.
8. Compare semantic-only `typed_v1` against hybrid `typed_v1` on the Terrafold
   corpus with per-case diffs.
9. Add the GIN expression index only if the hybrid implementation is useful
   enough to dogfood and query latency needs it.

The implementation should stay reversible, local-first, and tightly scoped to
candidate quality.
