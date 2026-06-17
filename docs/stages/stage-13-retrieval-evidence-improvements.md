# Stage 13 - Retrieval Evidence Improvements Closeout

## Status

- Branch: `stage-13-current-query-self-retrieval-fix`.
- Date closed: 2026-06-13.
- Mode: closeout and decision record.
- Implementation status: completed in this branch across Stage 13a, 13b, 13c,
  and 13d.

## Executive Summary

Stage 13 fixed one real production artifact and made the retrieval eval harness
more trustworthy. Assistant generation no longer retrieves the active user query
as evidence for its own response. The eval harness now has role-aware diagnostics,
lexical replay diagnostics, hardened lexical replay recommendations, and explicit
evidence provenance for source evidence versus derived assistant answers.

Stage 13 did not adopt production lexical reranking, summary-aware reranking,
role-aware reranking, summary-first retrieval, SQL hybrid retrieval, or
production retrieval weight changes. Production retrieval remains conservative.

## Stage 13a - Current-Query Self-Retrieval Fix

Assistant-generation retrieval now passes `exclude_message_id` with the active
query message ID when calling `MemoryService.retrieve_scoped_episodes`.

The active user message is still persisted normally. The exclusion only prevents
that active message episode from being returned as retrieval evidence for the
assistant response being generated from it. It remains available as memory for
future turns.

The retrieval SQL exclusion is message-specific:

- Message episodes whose `message_id` equals the active query message are
  excluded.
- Summary episodes are not accidentally excluded because their `message_id` is
  `NULL` and the predicate uses `IS DISTINCT FROM`.
- Scope and user filters remain intact.

`/retrieval/search` remains unchanged. The change applies to assistant
generation, not to the manual retrieval inspection endpoint.

Assistant-response provenance no longer records the active query as evidence for
its own assistant response. Tests cover the orchestrator call path, the memory
service exclusion behavior, and API-level provenance persistence.

## Stage 13b - Role-Aware Eval Diagnostics

Stage 13b added eval-only role-policy replay. It does not change production
retrieval behavior.

The replay can separately measure:

- assistant answer echoes,
- recap-question pollution,
- weak evidence before first acceptable evidence,
- top-k and diagnostic-depth appearances by role.

The role replay showed that assistant echo handling is a possible future
candidate, especially for reducing weak-evidence pollution without changing
source labels. It did not justify a production policy in Stage 13.

Recap-question policies remain blocked for production because recap-question
roles are eval metadata today, not durable production metadata. Stage 13 kept
those policies diagnostic only.

## Stage 13c - Lexical Diagnostics and Replay

Stage 13c added eval-only lexical features and replay profiles. The features
include token overlap, query token coverage, rare-anchor overlap, proper-name
overlap, exact number/currency/date overlap, and relationship-anchor overlap.

Fresh clean-memory Ollama lexical replay on the nine Terrafold cases showed
aggregate acceptable and summary improvement, but no raw hit improvement:

- baseline acceptable@k: `0.5556`
- best acceptable@k profiles: `0.7778`
- baseline raw@k: `0.5556`
- best raw@k: `0.5556`
- baseline summary@k: `0.5556`
- best summary@k: `0.7778`

The initial lexical replay recommendation identified a candidate lexical profile.
That recommendation was then hardened because the gains were summary-driven and
did not improve raw source evidence. The hardened recommendation became
`summary_only_gain_more_data_needed`.

The Dele/bookkeeping case improved from first acceptable rank 12 to rank 7 under
the strongest lexical profiles, but it still missed the official `top_k = 5`.
That is not enough evidence to justify production lexical reranking.

No SQL hybrid retrieval, full-text search, trigram search, BM25, candidate
generation changes, or production lexical reranking was adopted.

## Stage 13d - Evidence Provenance and Derived-Answer Metrics

The all-nine-case audit found no uncredited user-source or summary records in
official top-k. Source-only scoring therefore remains unchanged:

- `SOURCE_ONLY`: `5/9 = 0.5556`
- `SOURCE_PLUS_DERIVED`: `6/9 = 0.6667`

Only `terrafold_f5_dele_bookkeeping` flips when explicitly linked
assistant-derived answers are credited. That case is no longer a simple miss: it
is a source miss plus a derived-answer hit.

Assistant echoes are now represented as derived, non-source evidence. Default
official scoring remains source/summary only. The eval harness now distinguishes
source evidence from derived assistant answers with explicit metrics such as
`source_hit_at_k`, `derived_answer_hit_at_k`,
`source_miss_but_derived_hit_at_k`, `first_source_rank`, and
`first_derived_answer_rank`.

| Case | SOURCE_ONLY | SOURCE_PLUS_DERIVED | First Source | First Derived | Derived Ref | Flips |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| terrafold_f1_studio_name | yes | yes | 1 | 8 | tunde_echo_01_studio_name_answer | no |
| terrafold_f2_landlord_lease | yes | yes | 1 | none | none | no |
| terrafold_f3_kiln_budget | yes | yes | 1 | 3 | tunde_echo_02_kiln_budget_answer | no |
| terrafold_f4_latex_allergy | yes | yes | 2 | 1 | tunde_echo_03_latex_answer | no |
| terrafold_f5_dele_bookkeeping | no | yes | 12 | 1 | tunde_echo_05_dele_bookkeeping_answer | yes |
| terrafold_f6_class_size_wheels | yes | yes | 1 | 2 | tunde_echo_04_class_answer | no |
| terrafold_broad_operational_constraints | no | no | 9 | none | none | no |
| terrafold_either_opening_classes | no | no | 19 | none | none | no |
| terrafold_negative_clay_supplier | no | no | none | none | none | no |

Aggregates:

- `SOURCE_ONLY`: `5/9 = 0.5556`
- `SOURCE_PLUS_DERIVED`: `6/9 = 0.6667`

## What Stage 13 Did Not Do

Stage 13 deliberately did not add:

- production lexical reranking,
- production role-aware demotion or exclusion, except Stage 13a active-query
  exclusion,
- summary-first retrieval,
- hierarchical retrieval,
- parent-child summary expansion,
- SQL candidate-generation changes,
- full-text search, trigram search, or BM25,
- schema or migration changes,
- frontend changes,
- production weight tuning,
- answer-quality grading.

## Final Interpretation

Stage 13 makes the eval more trustworthy, not inflated. `acceptable_hit@k`
remains useful as a measure of source/summary retrieval quality. Derived
assistant answers are now visible separately rather than silently counted as
normal acceptable evidence.

The broad operational constraints and opening classes cases remain genuine
source/summary retrieval misses under the current production retrieval shape.
F5 is now better understood: the system can retrieve a derived assistant answer
near the top, but the original source evidence remains below official top-k.

Stage 13 supports keeping production retrieval conservative while improving the
measurement surface around future retrieval work.

## Recommended Next Stage

Do not merge further retrieval architecture into Stage 13. Close the branch with
the production active-query fix and the eval diagnostics.

Future retrieval work should happen on a new branch or stage. Good candidates
are:

- summary-aware retrieval planning for broad/opening cases,
- production assistant-echo exclusion or demotion if product behavior calls for
  it,
- an expanded eval corpus with more realistic multi-entity relationship cases.

Production lexical reranking should not be next unless additional corpora show
raw-evidence improvement, not just summary-driven aggregate gains.

## Acceptance Checklist

- [x] Active-query exclusion implemented and tested.
- [x] Role replay diagnostics implemented and tested.
- [x] Lexical replay implemented and hardened.
- [x] Derived-answer metrics implemented and tested.
- [x] Official source/summary baseline preserved.
- [x] No production ranking architecture change adopted.

## Validation Summary

Latest known validation for the Stage 13d state:

- Stage 13d targeted eval tests: `46 passed`.
- Full test suite: `213 passed`.
- `uv run mypy src/smriti/memory/eval.py`: passed.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `git diff --check`: passed.

