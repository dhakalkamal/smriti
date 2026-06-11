# Stage 12 - Retrieval Eval Harness Contract

## Status

- Branch: `stage-12-retrieval-eval-harness`.
- Date drafted: 2026-06-11.
- Mode: contract and design only.
- Implementation status: not implemented by this document.

Stage 12 defines how Smriti will measure retrieval quality before changing any
retrieval behavior. The contract is intentionally split into Stage 12a and
Stage 12b so the measuring instrument exists before any weight tuning begins.

## Problem Statement

Stage 11 added summary episode memory:

- Summary windows contain 12 persisted messages.
- Summary episodes use `episodes.kind = 'summary'`.
- Summary episodes are embedded into `embeddings_768`.
- Retrieval SQL and Python scoring were intentionally left unchanged.
- Raw message episodes and summary episodes compete in the same retrieval pool.

Manual Stage 11 runtime verification succeeded for summary episode creation:

- Conversation `e778fc27-54dd-42a3-8878-7cfacfb8ab55`.
- Summary window `13..24` existed with `embedding_rows = 1`.
- Summary window `25..36` existed with `embedding_rows = 1`.
- UI-triggered automatic summary creation worked for `25..36`.
- A previous timeout was caused by `qwen3:14b` exceeding 60 seconds; the local
  mitigation was `SMRITI_OLLAMA_CHAT_TIMEOUT_SECONDS=180`.

Manual retrieval observation with a planted Tunde/Terrafold conversation showed
that answer quality is not a reliable proxy for retrieval quality. Answers were
mostly correct, but the retrieval panel often ranked the current user query at
rank 1 with similarity `1.000`, recent recap questions crowded out original
source memories, and embedded summary episodes were not obviously visible in the
top 5 results. Stage 12 must therefore measure retrieved episodes directly, not
grade assistant answers.

## Source Context From Inspection

These findings describe the current repository behavior that the future harness
must account for. They are source context, not Stage 12 completion claims.

- The existing helper is `src/smriti/memory/eval.py`.
- The current input model is `RetrievalEvalCase` with `name`, `user_id`,
  `scope_id`, `query`, `expected_episode_ids`, and `top_k`.
- The current helper calls `MemoryService.retrieve_scoped_episodes(...)` and
  returns in-memory results and summary metrics.
- Current metrics include hit@k, precision@k, recall@k, reciprocal rank, and
  aggregate mean reciprocal rank.
- Current tests use synthetic fixture data, `FakeEmbedder(dimensions=768)`, and
  direct embedding/scoring-field manipulation to force deterministic ranking.
- The live app wires `OllamaEmbedder` by default, using the same `Embedder`
  protocol accepted by `MemoryService`.
- `MemoryService` defaults the embedding model registry key to
  `nomic-embed-text`.
- The default migration registers active 768-dimensional `nomic-embed-text`
  embeddings and stores vectors in `embeddings_768`.
- Message episode creation, stored-message episode creation, summary episode
  creation, and retrieval all use the injected embedder.
- Retrieval first embeds the query, fetches a bounded similarity-first candidate
  pool from SQL, and reranks that pool in Python.
- Current Python-side scoring weights are:
  - similarity: `0.55`
  - recency: `0.20`
  - access reinforcement: `0.10`
  - importance: `0.10`
  - frequency: `0.05`
- Retrieval SQL filters by `scope_id` and `user_id`, joins `embeddings_768`, and
  does not filter by episode kind.
- Returned `ScoredEpisode` records include IDs, kind, optional message position,
  optional summary range, score, similarity, and score components.
- Retrieval currently mutates returned rows by updating `access_count` and
  `last_accessed_at`.
- The schema includes `eval_scenarios`, `eval_runs`, and `eval_run_results`, but
  the current Python helper does not write to those tables.
- No final hand-labeled corpus schema, runner CLI, Makefile eval target, or
  committed baseline output pattern is established yet.

## Non-Scope

Stage 12 must not introduce:

- Hybrid retrieval.
- Hierarchical retrieval.
- Summary-first retrieval.
- Parent-child summary expansion.
- Kind-aware SQL retrieval filtering.
- SQL candidate structure changes.
- Frontend changes.
- Answer-quality grading.
- Production auto-tuning.
- Schema or migration changes.
- Application-code implementation as part of this documentation task.

## Stage 12a - Retrieval Eval Harness Contract + Real Baseline Design

### Purpose

Stage 12a defines the measuring instrument and the real baseline format. It must
evaluate current retrieval behavior against labeled cases without tuning weights
or changing retrieval behavior.

Stage 12a must answer:

- Which expected memory episodes were retrieved?
- Whether the retrieved evidence came from raw source messages or summary
  episodes.
- Whether current-query self-retrieval distorted official metrics.
- Whether recap questions or assistant answer echoes polluted the top results.
- Which failures are retrieval failures rather than answer-generation failures.

### Labeled Eval Corpus Format

The Stage 12a corpus should be machine-readable and hand-reviewable. JSONL is the
preferred initial format because one labeled example can be added, reviewed, and
diffed independently. A single JSON document is also acceptable if it preserves
the same fields and includes corpus metadata.

Each corpus file should include:

- `corpus_id`: stable corpus name, for example `terrafold-planted-facts-v1`.
- `corpus_version`: monotonically increasing version string.
- `fixture_strategy`: how the eval data is loaded or rebuilt.
- `embedding_model`: expected embedding model, initially `nomic-embed-text`.
- `notes`: optional reviewer notes that do not affect scoring.
- `examples`: the labeled eval examples, or one example per JSONL line.

The corpus must be scoped. Every example must identify the intended `user_id` and
`scope_id`, or identify a named fixture that deterministically resolves them.
Cross-scope retrieval must not be part of the default Stage 12 corpus.

### Labeled Eval Example Contract

Each labeled example should contain:

- `example_id`: stable ID unique within the corpus.
- `scenario_id`: stable ID for the planted conversation or fixture scenario.
- `query`: the retrieval query text.
- `top_k`: requested retrieval depth.
- `question_type`: a controlled label such as `direct_fact`, `constraint`,
  `relationship`, `broad_recap`, `summary_seeking`, or `negative_control`.
- `fact_ids`: planted fact IDs tested by the query.
- `preferred_layer`: `raw`, `summary`, or `either`.
- `raw_expected_episode_ids`: raw message source episodes that should satisfy
  the query.
- `summary_expected_episode_ids`: summary episodes that should satisfy the query.
- `acceptable_episode_ids`: any episode IDs that should count as acceptable
  supporting evidence even if they are not preferred.
- `current_query_episode_ids`: persisted current-query episodes to exclude from
  official scoring when present.
- `episode_labels`: labels for source, summary, recap, echo, and distractor
  episodes that may appear in retrieval results.
- `notes`: optional human explanation of why the expected IDs are valid.

Expected IDs should be episode IDs, not message IDs, because retrieval returns
episodes. When a label originates from a message, the label may also record the
message ID and message position for review.

### Episode Label Roles

Stage 12a must distinguish the following roles:

- `raw_source`: Layer 1 raw message source episodes. These are
  `episodes.kind = 'message'` records that contain the original planted fact or
  the original user/assistant turn that established it.
- `summary_source`: Layer 2 summary episodes. These are
  `episodes.kind = 'summary'` records whose `range_start` and `range_end`
  include the relevant planted fact.
- `recap_question`: user question episodes that ask about a previously planted
  fact but are not the original source of the fact.
- `assistant_answer_echo`: assistant answer episodes that repeat or restate the
  planted fact after retrieval.
- `distractor`: in-scope episodes that are irrelevant, partially overlapping, or
  semantically tempting but should not count as supporting evidence.
- `current_query`: a persisted copy of the active user query when it appears as
  a retrievable episode.

The label should also record:

- `episode_kind`: `message` or `summary`.
- `layer`: `raw`, `summary`, or `diagnostic`.
- `fact_ids`: planted facts supported or mentioned by the episode.
- `message_position`: for message episodes when available.
- `range_start` and `range_end`: for summary episodes when available.
- `is_expected`: whether the episode belongs to any expected set.
- `is_acceptable`: whether the episode may count for acceptable-hit metrics.

### Expected Raw vs Summary Retrieval

The corpus must allow raw and summary expectations to be represented separately.

Direct fact questions should usually prefer `raw` because the ideal evidence is
the original memory that established the fact. Broad recap or summary-seeking
questions should usually prefer `summary` because the ideal evidence may be a
compressed episode covering several facts. Some examples may use `either` when
both raw and summary evidence are equally valid.

The scoring contract must treat these sets distinctly:

- `raw_expected_episode_ids` drive `raw_hit@k`.
- `summary_expected_episode_ids` drive `summary_hit@k`.
- `acceptable_episode_ids` drive `acceptable_hit@k`.
- `preferred_layer` determines whether layer-specific misses should be reported
  as failures even when an acceptable episode was retrieved.

### Current-Query Self-Retrieval

Current-query self-retrieval must be excluded from official scoring and recorded
as a diagnostic.

If a retrieved episode is labeled `current_query`, Stage 12a must:

- Preserve its original rank, score, similarity, and score components in output.
- Mark `is_current_query = true`.
- Exclude it from hit@k, MRR, precision@k, recall@k, raw_hit@k,
  summary_hit@k, and acceptable_hit@k.
- Recompute official ranks after exclusion so official metrics are based on the
  evidence-bearing result order.
- Record `self_query_hit = true`, `self_query_rank`, and
  `self_query_similarity` as diagnostics.

The harness must not change production retrieval behavior to solve this artifact
in Stage 12a.

### Recap-Question Pollution

Recap-question pollution must be explicitly labeled and measured. A recap
question can be semantically similar to the query while still being poor evidence
for the original fact.

Stage 12a must:

- Label recap question episodes as `recap_question`.
- Label answer echo episodes separately as `assistant_answer_echo`.
- Exclude only `current_query` episodes from official scoring by default.
- Count recap-question appearances in the official top-k after current-query
  exclusion.
- Report `recap_pollution@k` for each case and in aggregate.

Assistant answer echoes should be preserved in output with their own label role.
They may count as acceptable only when the corpus author explicitly includes
them in `acceptable_episode_ids`; otherwise they are diagnostics or distractors.

### Metrics

Stage 12a must report these metrics per case and in aggregate:

- `hit@k`: whether any preferred expected episode appears in the official top-k.
- `mrr` / `reciprocal_rank`: reciprocal rank of the first preferred expected
  episode in the official result order.
- `precision@k`: preferred expected hits divided by official retrieved count up
  to k.
- `recall@k`: preferred expected hits divided by preferred expected count.
- `raw_hit@k`: whether any `raw_expected_episode_ids` appear in the official
  top-k.
- `summary_hit@k`: whether any `summary_expected_episode_ids` appear in the
  official top-k.
- `acceptable_hit@k`: whether any `acceptable_episode_ids` appear in the
  official top-k.
- `kind_mix@k`: counts and ratios of `message` versus `summary` episodes in the
  official top-k.
- `self_query_hit_rate`: aggregate rate at which current-query episodes appeared
  in retrieved results before exclusion.
- `recap_pollution@k`: count and ratio of `recap_question` episodes in the
  official top-k.

For examples with `preferred_layer = raw`, preferred expected episodes are the
raw expected IDs. For `preferred_layer = summary`, preferred expected episodes
are the summary expected IDs. For `preferred_layer = either`, preferred expected
episodes are the union of raw and summary expected IDs.

### Baseline Output Proposal

Stage 12a must produce both machine-readable output and a human-readable report.

Machine-readable output should be JSON or JSONL and include run metadata:

- `run_id`.
- `corpus_id` and `corpus_version`.
- Git commit SHA and branch when available.
- Runtime timestamp.
- Database or fixture identifier with credentials redacted.
- Embedding model and dimensions.
- Retrieval `top_k`.
- Scoring version and Python-side scoring weights.
- Candidate-pool constants such as candidate multiplier and minimum candidates.
- Isolation strategy used for the run.

Each per-example result should include:

- Example identity and labels: `example_id`, `scenario_id`, `question_type`,
  `fact_ids`, and `preferred_layer`.
- Expected sets: raw expected IDs, summary expected IDs, acceptable IDs, and
  current-query IDs.
- Official metrics and diagnostics.
- Retrieved episodes in original order and official order.
- For each retrieved episode: episode ID, kind, original rank, official rank,
  conversation ID, scope ID, message ID, message position, range start, range
  end, score, similarity, recency score, access score, importance score,
  frequency score, expected flags, acceptable flag, current-query flag, and
  label roles.

The human-readable report should include:

- Aggregate metric table.
- Metric table grouped by `question_type`.
- Metric table grouped by `preferred_layer`.
- Top failure cases with expected IDs and retrieved IDs.
- Cases with current-query self-retrieval.
- Cases with high recap pollution.
- Kind mix summary showing how often summary episodes appear in top-k.
- Notes about any fixture or runtime anomaly.

### Avoiding Misleading Mutation

Current retrieval mutates `access_count` and `last_accessed_at` for returned
episodes. Stage 12a must avoid baselines that change as a side effect of earlier
eval cases.

The official baseline must declare and use one explicit isolation strategy:

- Preferred: disposable eval database rebuilt from a fixture before every run.
- Acceptable: disposable eval scope and conversations rebuilt before every run.
- Acceptable for narrow debugging: snapshot and restore access metadata for all
  fixture episodes before each case or before each run.

The isolation strategy must not require retrieval SQL changes. If the harness
uses the production retrieval service, it may accept the mutation only inside
disposable or resettable fixture state.

### Stage 12a Acceptance Criteria

Stage 12a is acceptable when:

- A labeled corpus contract exists and can represent raw source episodes,
  summary source episodes, recap questions, assistant answer echoes, distractors,
  and current-query episodes.
- Expected raw IDs, expected summary IDs, acceptable IDs, preferred layer,
  question type, and fact IDs are represented per example.
- Official metrics exclude current-query self-retrieval and diagnostics record
  the artifact.
- Recap-question pollution is labeled and reported.
- Baseline output includes retrieved episode IDs, kind, rank, score, similarity,
  score components, expected flags, and label roles.
- The baseline run uses an explicit disposable or resettable isolation strategy.
- The baseline measures retrieval quality only, not assistant answer quality.
- Retrieval behavior, retrieval SQL, schema, config, frontend, and scoring
  weights remain unchanged during Stage 12a.

## Stage 12b - Baseline Analysis + Controlled Weight Experiment Design

### Purpose

Stage 12b analyzes Stage 12a baseline failures and defines controlled
experiments for existing Python-side weights only. Stage 12b must not begin until
the Stage 12a baseline exists.

Stage 12b is still retrieval-eval work. It must not introduce hybrid retrieval,
hierarchical retrieval, summary-first retrieval, kind-aware SQL filtering, or SQL
candidate structure changes.

### Failure Classes

Stage 12b analysis must assign failed or suspicious cases to one or more failure
classes:

- `source_fact_not_in_candidate_pool`: the labeled source episode does not enter
  the bounded similarity-first candidate pool.
- `source_fact_reranked_too_low`: the labeled source episode is available to
  Python reranking but falls below the requested top-k.
- `summary_exists_but_does_not_rank`: a relevant summary episode exists and is
  embedded but does not appear in top-k.
- `recap_questions_outrank_sources`: recap question episodes rank above raw or
  summary source evidence.
- `current_query_self_retrieval_artifact`: the active query episode appears in
  retrieved results and would distort official metrics without exclusion.
- `direct_fact_prefers_non_raw`: a direct fact question retrieves acceptable
  non-raw evidence while missing the preferred raw source.
- `broad_question_misses_summary`: a broad or summary-seeking question misses
  relevant summary episodes.
- `distractor_outranks_expected`: labeled distractors outrank preferred expected
  evidence.

If candidate-pool visibility is needed for classification, the diagnostic must
mirror the current SQL candidate behavior without changing production retrieval
SQL. Any diagnostic candidate-pool query must be clearly marked as eval-only.

### Controlled Weight Experiments

Stage 12b experiments may vary only the existing Python-side scoring weights:

- similarity
- recency
- access reinforcement
- importance
- frequency

The baseline weight profile is:

- similarity: `0.55`
- recency: `0.20`
- access reinforcement: `0.10`
- importance: `0.10`
- frequency: `0.05`

Each experiment must declare:

- `weight_profile_id`.
- Full numeric weight values.
- Whether weights are normalized and how.
- Expected improvement hypothesis.
- Failure classes the profile is intended to address.
- Baseline run ID used for comparison.
- Corpus version and fixture isolation strategy.

Experiments must not:

- Change retrieval SQL.
- Change SQL candidate ordering or limit.
- Add hybrid retrieval.
- Add hierarchical retrieval.
- Add summary-first retrieval.
- Add kind-aware SQL filtering.
- Change schema or migrations.
- Change frontend behavior.
- Tune weights automatically in production.

### Before/After Comparison Format

Every Stage 12b experiment must compare against the Stage 12a baseline using the
same corpus version and equivalent fixture state.

The comparison output should include:

- Baseline run ID and experiment run ID.
- Baseline and experiment weight profiles.
- Aggregate metric deltas.
- Metric deltas by `question_type`.
- Metric deltas by `preferred_layer`.
- Raw-hit and summary-hit deltas.
- Acceptable-hit deltas.
- Kind-mix deltas.
- Self-query diagnostics, expected to remain diagnostic rather than optimized.
- Recap-pollution deltas.
- Per-case rank movement for expected raw and summary episodes.
- New regressions where a previously passing case fails.
- Newly passing cases with their assigned failure classes.

The human-readable comparison should end with a recommendation to adopt, reject,
or continue investigating the tested weight profile. Adoption must be justified
by retrieval metrics, not answer quality.

### Stage 12b Acceptance Criteria

Stage 12b is acceptable when:

- A Stage 12a baseline exists for the same corpus version.
- Failed and suspicious cases are assigned to documented failure classes.
- Controlled experiments vary only existing Python-side scoring weights.
- Before/after output shows aggregate and per-case deltas.
- Regressions are visible and reviewed.
- Direct fact cases can be evaluated separately from broad summary-seeking cases.
- Raw and summary retrieval quality are reported separately.
- No retrieval SQL, schema, migration, frontend, hybrid retrieval, hierarchical
  retrieval, or production auto-tuning changes are introduced.

## Eval Corpus Proposal

The first real corpus should be based on the Tunde/Terrafold planted-fact
conversation because it already exposed the retrieval-quality questions Stage 12
needs to answer.

Planted facts:

- F1: studio name is Terrafold, rejected alternative Kilnhouse.
- F2: landlord is Mr. Obafemi; lease signing is Friday, July 31, in person.
- F3: kiln budget hard cap is `$3,650`.
- F4: severe latex allergy; all gloves must be nitrile.
- F5: cousin Dele is silent partner handling bookkeeping.
- F6: class size is capped at 9 because there are exactly 9 pottery wheels.

The corpus should include at least:

- Direct fact questions for each planted fact.
- Broad recap questions that should make summary episodes competitive.
- Distractor questions that are semantically close but unsupported.
- Cases where raw source evidence is preferred.
- Cases where summary source evidence is preferred.
- Cases where raw or summary evidence is acceptable.
- Cases where recap questions and assistant answer echoes are labeled but should
  not silently inflate quality metrics.

The corpus should preserve enough fixture metadata to map planted facts to
episode IDs after rebuilding a disposable eval database or scope.

## Metrics Proposal Summary

The official Stage 12 metrics are:

- `hit@k`
- `mrr` / `reciprocal_rank`
- `precision@k`
- `recall@k`
- `raw_hit@k`
- `summary_hit@k`
- `acceptable_hit@k`
- `kind_mix@k`
- `self_query_hit_rate`
- `recap_pollution@k`

The reporting layer may add supporting diagnostics, but these metrics are the
minimum contract for Stage 12a and the required comparison set for Stage 12b.

## Implementation Sequencing

No implementation is performed by this document. Future implementation should
proceed in this order:

1. Finalize the labeled corpus schema and fixture isolation strategy.
2. Add or adapt an on-demand eval runner that uses the existing memory service
   retrieval path.
3. Add corpus loading and label validation.
4. Add current-query exclusion for official scoring while preserving diagnostics.
5. Add recap-pollution and kind-mix reporting.
6. Add JSON/JSONL baseline output and a human-readable summary report.
7. Run the real Stage 12a baseline on disposable or resettable eval state.
8. Analyze Stage 12a failures into the Stage 12b failure classes.
9. Run controlled Python-side weight experiments only after the baseline exists.
10. Compare before/after results and decide whether any weight profile is worth
    adopting in a later implementation stage.

## Completion Boundary For This Draft

This document is complete when it functions as the Stage 12 contract: it defines
what the harness must measure, how labeled retrieval cases should be represented,
how baseline output should look, how mutation must be isolated, what is out of
scope, and how Stage 12b may analyze and compare controlled weight experiments.
