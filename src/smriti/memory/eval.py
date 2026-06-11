from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from smriti.memory.errors import InvalidRetrievalRequestError
from smriti.memory.models import EpisodeKind, ScoredEpisode
from smriti.memory.service import (
    ACCESS_WEIGHT,
    FREQUENCY_WEIGHT,
    IMPORTANCE_WEIGHT,
    MIN_RETRIEVAL_CANDIDATES,
    RECENCY_WEIGHT,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    SCORING_VERSION,
    SIMILARITY_WEIGHT,
    MemoryService,
)

EpisodeLabelRole = Literal[
    "raw_source",
    "summary_source",
    "recap_question",
    "assistant_answer_echo",
    "distractor",
    "current_query",
]
EpisodeLabelLayer = Literal["raw", "summary", "diagnostic"]
PreferredLayer = Literal["raw", "summary", "either"]
QuestionType = Literal[
    "direct_fact",
    "constraint",
    "relationship",
    "broad_recap",
    "summary_seeking",
    "negative_control",
]
TimingMode = Literal["app_realistic", "clean_memory"]

_VALID_EPISODE_LABEL_ROLES = frozenset(
    {
        "raw_source",
        "summary_source",
        "recap_question",
        "assistant_answer_echo",
        "distractor",
        "current_query",
    }
)
_VALID_EPISODE_LABEL_LAYERS = frozenset({"raw", "summary", "diagnostic"})
_VALID_PREFERRED_LAYERS = frozenset({"raw", "summary", "either"})
_VALID_QUESTION_TYPES = frozenset(
    {
        "direct_fact",
        "constraint",
        "relationship",
        "broad_recap",
        "summary_seeking",
        "negative_control",
    }
)
_VALID_TIMING_MODES = frozenset({"app_realistic", "clean_memory"})
_VALID_EPISODE_KINDS = frozenset({"message", "summary"})


@dataclass(frozen=True)
class RetrievalEvalCase:
    name: str
    user_id: UUID
    scope_id: UUID
    query: str
    expected_episode_ids: tuple[UUID, ...]
    top_k: int


@dataclass(frozen=True)
class RetrievalEvalResult:
    case_name: str
    retrieved_episode_ids: tuple[UUID, ...]
    expected_episode_ids: tuple[UUID, ...]
    hit_at_k: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalEvalSummary:
    total_cases: int
    hit_rate_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class Stage12ExpectedRefs:
    raw: tuple[str, ...]
    summary: tuple[str, ...]
    acceptable: tuple[str, ...]
    current_query: tuple[str, ...]


@dataclass(frozen=True)
class Stage12ExpectedIds:
    raw: tuple[UUID, ...]
    summary: tuple[UUID, ...]
    acceptable: tuple[UUID, ...]
    current_query: tuple[UUID, ...]


@dataclass(frozen=True)
class Stage12CorpusEpisodeLabel:
    semantic_ref: str
    roles: tuple[EpisodeLabelRole, ...]
    episode_kind: EpisodeKind
    layer: EpisodeLabelLayer
    fact_ids: tuple[str, ...]
    message_position: int | None = None
    range_start: int | None = None
    range_end: int | None = None
    is_expected: bool = False
    is_acceptable: bool = False


@dataclass(frozen=True)
class Stage12ResolvedEpisodeLabel:
    semantic_ref: str
    episode_id: UUID
    roles: tuple[EpisodeLabelRole, ...]
    episode_kind: EpisodeKind
    layer: EpisodeLabelLayer
    fact_ids: tuple[str, ...]
    message_position: int | None = None
    range_start: int | None = None
    range_end: int | None = None
    is_expected: bool = False
    is_acceptable: bool = False


@dataclass(frozen=True)
class Stage12CorpusCase:
    example_id: str
    scenario_id: str
    query: str
    top_k: int
    question_type: QuestionType
    fact_ids: tuple[str, ...]
    preferred_layer: PreferredLayer
    expected_refs: Stage12ExpectedRefs
    episode_labels: tuple[Stage12CorpusEpisodeLabel, ...]
    notes: str | None = None


@dataclass(frozen=True)
class Stage12Corpus:
    corpus_id: str
    corpus_version: str
    fixture_strategy: str
    embedding_model: str
    cases: tuple[Stage12CorpusCase, ...]
    notes: str | None = None


@dataclass(frozen=True)
class Stage12ResolvedEvalCase:
    example_id: str
    scenario_id: str
    user_id: UUID
    scope_id: UUID
    query: str
    top_k: int
    question_type: QuestionType
    fact_ids: tuple[str, ...]
    preferred_layer: PreferredLayer
    expected_ids: Stage12ExpectedIds
    episode_labels: tuple[Stage12ResolvedEpisodeLabel, ...]
    notes: str | None = None


@dataclass(frozen=True)
class Stage12KindMixMetrics:
    message_count: int
    summary_count: int
    total_count: int
    message_ratio: float
    summary_ratio: float


@dataclass(frozen=True)
class Stage12RecapPollutionMetrics:
    count: int
    ratio: float


@dataclass(frozen=True)
class Stage12RetrievedEpisodeRecord:
    episode_id: UUID
    kind: EpisodeKind
    rank: int
    official_rank: int | None
    conversation_id: UUID
    scope_id: UUID
    message_id: UUID | None
    message_position: int | None
    range_start: int | None
    range_end: int | None
    score: float
    similarity: float
    recency_score: float
    access_score: float
    importance_score: float
    frequency_score: float
    label_roles: tuple[EpisodeLabelRole, ...]
    fact_ids: tuple[str, ...]
    is_raw_expected: bool
    is_summary_expected: bool
    is_preferred_expected: bool
    is_acceptable: bool
    is_current_query: bool


@dataclass(frozen=True)
class Stage12CaseMetrics:
    hit_at_k: bool
    reciprocal_rank: float
    precision_at_k: float
    recall_at_k: float
    raw_hit_at_k: bool
    summary_hit_at_k: bool
    acceptable_hit_at_k: bool
    kind_mix_at_k: Stage12KindMixMetrics
    self_query_hit: bool
    self_query_rank: int | None
    self_query_similarity: float | None
    recap_pollution_at_k: Stage12RecapPollutionMetrics


@dataclass(frozen=True)
class Stage12CaseResult:
    example_id: str
    scenario_id: str
    question_type: QuestionType
    preferred_layer: PreferredLayer
    fact_ids: tuple[str, ...]
    top_k: int
    expected_ids: Stage12ExpectedIds
    retrieved: tuple[Stage12RetrievedEpisodeRecord, ...]
    metrics: Stage12CaseMetrics
    notes: str | None = None


@dataclass(frozen=True)
class Stage12AggregateMetrics:
    total_cases: int
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    raw_hit_rate_at_k: float
    summary_hit_rate_at_k: float
    acceptable_hit_rate_at_k: float
    self_query_hit_rate: float
    kind_mix_at_k: Stage12KindMixMetrics
    mean_recap_pollution_count_at_k: float
    mean_recap_pollution_ratio_at_k: float


@dataclass(frozen=True)
class Stage12BaselineRunMetadata:
    run_id: str
    created_at: datetime
    corpus_id: str
    corpus_version: str
    timing_mode: TimingMode
    embedder_mode: str
    embedding_model: str
    isolation_strategy: str
    database_identifier: str | None = None
    git_branch: str | None = None
    git_sha: str | None = None
    scoring_version: str = SCORING_VERSION
    scoring_weights: Mapping[str, float] = field(default_factory=lambda: stage12_scoring_weights())
    retrieval_candidate_multiplier: int = RETRIEVAL_CANDIDATE_MULTIPLIER
    min_retrieval_candidates: int = MIN_RETRIEVAL_CANDIDATES


async def run_retrieval_eval(
    service: MemoryService,
    cases: Sequence[RetrievalEvalCase],
    now: datetime | None = None,
) -> tuple[list[RetrievalEvalResult], RetrievalEvalSummary]:
    """Evaluate scoped retrieval quality through the real memory service path."""

    for case in cases:
        _validate_eval_case(case)

    results: list[RetrievalEvalResult] = []
    for case in cases:
        retrieved = await service.retrieve_scoped_episodes(
            user_id=case.user_id,
            scope_id=case.scope_id,
            query=case.query,
            top_k=case.top_k,
            now=now,
        )
        retrieved_episode_ids = tuple(episode.id for episode in retrieved)
        results.append(
            _result_for_case(
                case=case,
                retrieved_episode_ids=retrieved_episode_ids,
            )
        )

    return results, _summarize_results(results)


async def run_stage12_retrieval_eval(
    service: MemoryService,
    cases: Sequence[Stage12ResolvedEvalCase],
    timing_mode: TimingMode = "app_realistic",
    now: datetime | None = None,
) -> tuple[list[Stage12CaseResult], Stage12AggregateMetrics]:
    """Evaluate Stage 12a retrieval cases through the current memory service path."""

    _validate_timing_mode(timing_mode)
    for case in cases:
        validate_stage12_eval_case(case)

    results: list[Stage12CaseResult] = []
    for case in cases:
        retrieved = await service.retrieve_scoped_episodes(
            user_id=case.user_id,
            scope_id=case.scope_id,
            query=case.query,
            top_k=case.top_k,
            now=now,
        )
        results.append(score_stage12_retrieval_case(case, retrieved, timing_mode))

    return results, summarize_stage12_results(results)


def score_stage12_retrieval_case(
    case: Stage12ResolvedEvalCase,
    retrieved: Sequence[ScoredEpisode],
    timing_mode: TimingMode = "app_realistic",
) -> Stage12CaseResult:
    """Score one Stage 12a case without changing retrieval behavior."""

    validate_stage12_eval_case(case)
    _validate_timing_mode(timing_mode)

    limited_retrieved = tuple(retrieved[: case.top_k])
    label_by_episode_id = {label.episode_id: label for label in case.episode_labels}
    raw_expected_ids = set(case.expected_ids.raw)
    summary_expected_ids = set(case.expected_ids.summary)
    acceptable_ids = set(case.expected_ids.acceptable)
    current_query_ids = set(case.expected_ids.current_query)
    preferred_ids = _preferred_expected_id_set(case.expected_ids, case.preferred_layer)

    official_episodes = tuple(
        episode
        for episode in limited_retrieved
        if not _is_current_query_episode(
            episode=episode,
            label=label_by_episode_id.get(episode.id),
            current_query_ids=current_query_ids,
            timing_mode=timing_mode,
        )
    )
    official_rank_by_episode_id = {
        episode.id: official_rank
        for official_rank, episode in enumerate(official_episodes, start=1)
    }
    retrieved_records = tuple(
        _stage12_retrieved_record(
            episode=episode,
            label=label_by_episode_id.get(episode.id),
            official_rank=official_rank_by_episode_id.get(episode.id),
            raw_expected_ids=raw_expected_ids,
            summary_expected_ids=summary_expected_ids,
            acceptable_ids=acceptable_ids,
            current_query_ids=current_query_ids,
            preferred_ids=preferred_ids,
            timing_mode=timing_mode,
        )
        for episode in limited_retrieved
    )

    official_records = tuple(
        record for record in retrieved_records if record.official_rank is not None
    )
    official_episode_ids = tuple(record.episode_id for record in official_records)
    preferred_hits = tuple(
        episode_id for episode_id in official_episode_ids if episode_id in preferred_ids
    )
    first_preferred_rank = next(
        (
            record.official_rank
            for record in official_records
            if record.episode_id in preferred_ids and record.official_rank is not None
        ),
        None,
    )
    self_query_record = next(
        (record for record in retrieved_records if record.is_current_query),
        None,
    )
    recap_pollution = _recap_pollution_metrics(official_records)

    return Stage12CaseResult(
        example_id=case.example_id,
        scenario_id=case.scenario_id,
        question_type=case.question_type,
        preferred_layer=case.preferred_layer,
        fact_ids=case.fact_ids,
        top_k=case.top_k,
        expected_ids=case.expected_ids,
        retrieved=retrieved_records,
        metrics=Stage12CaseMetrics(
            hit_at_k=bool(preferred_hits),
            reciprocal_rank=0.0 if first_preferred_rank is None else 1.0 / first_preferred_rank,
            precision_at_k=0.0
            if not official_records
            else len(set(preferred_hits)) / len(official_records),
            recall_at_k=0.0 if not preferred_ids else len(set(preferred_hits)) / len(preferred_ids),
            raw_hit_at_k=any(episode_id in raw_expected_ids for episode_id in official_episode_ids),
            summary_hit_at_k=any(
                episode_id in summary_expected_ids for episode_id in official_episode_ids
            ),
            acceptable_hit_at_k=any(
                episode_id in acceptable_ids for episode_id in official_episode_ids
            ),
            kind_mix_at_k=_kind_mix_metrics(official_records),
            self_query_hit=self_query_record is not None,
            self_query_rank=None if self_query_record is None else self_query_record.rank,
            self_query_similarity=None
            if self_query_record is None
            else self_query_record.similarity,
            recap_pollution_at_k=recap_pollution,
        ),
        notes=case.notes,
    )


def summarize_stage12_results(results: Sequence[Stage12CaseResult]) -> Stage12AggregateMetrics:
    """Aggregate the required Stage 12a metrics across cases."""

    total_cases = len(results)
    if total_cases == 0:
        return Stage12AggregateMetrics(
            total_cases=0,
            hit_rate_at_k=0.0,
            mean_reciprocal_rank=0.0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            raw_hit_rate_at_k=0.0,
            summary_hit_rate_at_k=0.0,
            acceptable_hit_rate_at_k=0.0,
            self_query_hit_rate=0.0,
            kind_mix_at_k=Stage12KindMixMetrics(
                message_count=0,
                summary_count=0,
                total_count=0,
                message_ratio=0.0,
                summary_ratio=0.0,
            ),
            mean_recap_pollution_count_at_k=0.0,
            mean_recap_pollution_ratio_at_k=0.0,
        )

    total_message_count = sum(result.metrics.kind_mix_at_k.message_count for result in results)
    total_summary_count = sum(result.metrics.kind_mix_at_k.summary_count for result in results)
    total_kind_count = total_message_count + total_summary_count

    return Stage12AggregateMetrics(
        total_cases=total_cases,
        hit_rate_at_k=sum(1 for result in results if result.metrics.hit_at_k) / total_cases,
        mean_reciprocal_rank=sum(result.metrics.reciprocal_rank for result in results)
        / total_cases,
        mean_precision_at_k=sum(result.metrics.precision_at_k for result in results) / total_cases,
        mean_recall_at_k=sum(result.metrics.recall_at_k for result in results) / total_cases,
        raw_hit_rate_at_k=sum(1 for result in results if result.metrics.raw_hit_at_k) / total_cases,
        summary_hit_rate_at_k=sum(1 for result in results if result.metrics.summary_hit_at_k)
        / total_cases,
        acceptable_hit_rate_at_k=sum(1 for result in results if result.metrics.acceptable_hit_at_k)
        / total_cases,
        self_query_hit_rate=sum(1 for result in results if result.metrics.self_query_hit)
        / total_cases,
        kind_mix_at_k=Stage12KindMixMetrics(
            message_count=total_message_count,
            summary_count=total_summary_count,
            total_count=total_kind_count,
            message_ratio=0.0 if total_kind_count == 0 else total_message_count / total_kind_count,
            summary_ratio=0.0 if total_kind_count == 0 else total_summary_count / total_kind_count,
        ),
        mean_recap_pollution_count_at_k=sum(
            result.metrics.recap_pollution_at_k.count for result in results
        )
        / total_cases,
        mean_recap_pollution_ratio_at_k=sum(
            result.metrics.recap_pollution_at_k.ratio for result in results
        )
        / total_cases,
    )


def stage12_scoring_weights() -> dict[str, float]:
    """Return the active Python-side scoring weights for baseline metadata."""

    return {
        "similarity": SIMILARITY_WEIGHT,
        "recency": RECENCY_WEIGHT,
        "access": ACCESS_WEIGHT,
        "importance": IMPORTANCE_WEIGHT,
        "frequency": FREQUENCY_WEIGHT,
    }


def load_stage12_corpus(path: Path) -> Stage12Corpus:
    """Load a Stage 12a corpus from JSON or JSONL without reading local notes."""

    raw_text = path.read_text(encoding="utf-8")
    records = [line for line in raw_text.splitlines() if line.strip()]
    if not records:
        raise InvalidRetrievalRequestError("Stage 12 corpus must not be empty")

    first_value = cast(dict[str, object], json.loads(records[0]))
    if len(records) == 1 and "examples" in first_value:
        corpus = _parse_stage12_corpus_document(first_value)
    else:
        corpus = _parse_stage12_jsonl_records(records)

    validate_stage12_corpus(corpus)
    return corpus


def validate_stage12_corpus(corpus: Stage12Corpus) -> None:
    """Validate semantic refs and controlled labels in a Stage 12a corpus."""

    if not corpus.corpus_id.strip():
        raise InvalidRetrievalRequestError("Stage 12 corpus_id must not be empty")
    if not corpus.corpus_version.strip():
        raise InvalidRetrievalRequestError("Stage 12 corpus_version must not be empty")
    if not corpus.cases:
        raise InvalidRetrievalRequestError("Stage 12 corpus must include at least one case")

    example_ids: set[str] = set()
    for case in corpus.cases:
        if case.example_id in example_ids:
            raise InvalidRetrievalRequestError("Stage 12 corpus example_id values must be unique")
        example_ids.add(case.example_id)
        validate_stage12_corpus_case(case)


def validate_stage12_corpus_case(case: Stage12CorpusCase) -> None:
    """Validate one corpus case before semantic refs are resolved to UUIDs."""

    if not case.example_id.strip():
        raise InvalidRetrievalRequestError("Stage 12 example_id must not be empty")
    if not case.scenario_id.strip():
        raise InvalidRetrievalRequestError("Stage 12 scenario_id must not be empty")
    if not case.query.strip():
        raise InvalidRetrievalRequestError("Stage 12 query must not be empty")
    if case.top_k <= 0:
        raise InvalidRetrievalRequestError("Stage 12 top_k must be greater than zero")
    _validate_question_type(case.question_type)
    _validate_preferred_layer(case.preferred_layer)

    label_refs: set[str] = set()
    for label in case.episode_labels:
        _validate_corpus_label(label)
        if label.semantic_ref in label_refs:
            raise InvalidRetrievalRequestError(
                f"Stage 12 duplicate episode label ref: {label.semantic_ref}"
            )
        label_refs.add(label.semantic_ref)

    for expected_ref in _all_expected_refs(case.expected_refs):
        if expected_ref not in label_refs:
            raise InvalidRetrievalRequestError(
                f"Stage 12 expected ref has no episode label: {expected_ref}"
            )


def validate_stage12_eval_case(case: Stage12ResolvedEvalCase) -> None:
    """Validate one resolved Stage 12a case before scoring."""

    if case.top_k <= 0:
        raise InvalidRetrievalRequestError("Stage 12 top_k must be greater than zero")
    if not case.query.strip():
        raise InvalidRetrievalRequestError("Stage 12 query must not be empty")
    _validate_question_type(case.question_type)
    _validate_preferred_layer(case.preferred_layer)

    label_ids: set[UUID] = set()
    for label in case.episode_labels:
        if label.episode_id in label_ids:
            raise InvalidRetrievalRequestError(
                f"Stage 12 duplicate resolved episode label: {label.semantic_ref}"
            )
        label_ids.add(label.episode_id)
        _validate_roles(label.roles)
        _validate_episode_kind(label.episode_kind)
        _validate_label_layer(label.layer)


def resolve_stage12_case(
    corpus_case: Stage12CorpusCase,
    user_id: UUID,
    scope_id: UUID,
    ref_to_episode_id: Mapping[str, UUID],
) -> Stage12ResolvedEvalCase:
    """Resolve one semantic-ref corpus case to runtime episode UUIDs."""

    validate_stage12_corpus_case(corpus_case)
    resolved_labels = tuple(
        Stage12ResolvedEpisodeLabel(
            semantic_ref=label.semantic_ref,
            episode_id=_resolve_semantic_ref(label.semantic_ref, ref_to_episode_id),
            roles=label.roles,
            episode_kind=label.episode_kind,
            layer=label.layer,
            fact_ids=label.fact_ids,
            message_position=label.message_position,
            range_start=label.range_start,
            range_end=label.range_end,
            is_expected=label.is_expected,
            is_acceptable=label.is_acceptable,
        )
        for label in corpus_case.episode_labels
    )

    return Stage12ResolvedEvalCase(
        example_id=corpus_case.example_id,
        scenario_id=corpus_case.scenario_id,
        user_id=user_id,
        scope_id=scope_id,
        query=corpus_case.query,
        top_k=corpus_case.top_k,
        question_type=corpus_case.question_type,
        fact_ids=corpus_case.fact_ids,
        preferred_layer=corpus_case.preferred_layer,
        expected_ids=Stage12ExpectedIds(
            raw=tuple(
                _resolve_semantic_ref(ref, ref_to_episode_id)
                for ref in corpus_case.expected_refs.raw
            ),
            summary=tuple(
                _resolve_semantic_ref(ref, ref_to_episode_id)
                for ref in corpus_case.expected_refs.summary
            ),
            acceptable=tuple(
                _resolve_semantic_ref(ref, ref_to_episode_id)
                for ref in corpus_case.expected_refs.acceptable
            ),
            current_query=tuple(
                _resolve_semantic_ref(ref, ref_to_episode_id)
                for ref in corpus_case.expected_refs.current_query
            ),
        ),
        episode_labels=resolved_labels,
        notes=corpus_case.notes,
    )


def resolve_stage12_corpus_cases(
    corpus: Stage12Corpus,
    user_id: UUID,
    scope_id: UUID,
    ref_to_episode_id: Mapping[str, UUID],
) -> list[Stage12ResolvedEvalCase]:
    """Resolve all corpus cases to runtime episode UUIDs."""

    validate_stage12_corpus(corpus)
    return [
        resolve_stage12_case(
            corpus_case=case,
            user_id=user_id,
            scope_id=scope_id,
            ref_to_episode_id=ref_to_episode_id,
        )
        for case in corpus.cases
    ]


def stage12_baseline_metadata_to_dict(metadata: Stage12BaselineRunMetadata) -> dict[str, object]:
    """Convert Stage 12a run metadata to JSON-safe values."""

    return {
        "run_id": metadata.run_id,
        "created_at": metadata.created_at.astimezone(UTC).isoformat(),
        "corpus_id": metadata.corpus_id,
        "corpus_version": metadata.corpus_version,
        "timing_mode": metadata.timing_mode,
        "embedder_mode": metadata.embedder_mode,
        "embedding_model": metadata.embedding_model,
        "isolation_strategy": metadata.isolation_strategy,
        "database_identifier": metadata.database_identifier,
        "git_branch": metadata.git_branch,
        "git_sha": metadata.git_sha,
        "scoring_version": metadata.scoring_version,
        "scoring_weights": dict(metadata.scoring_weights),
        "retrieval_candidate_multiplier": metadata.retrieval_candidate_multiplier,
        "min_retrieval_candidates": metadata.min_retrieval_candidates,
    }


def stage12_case_result_to_dict(result: Stage12CaseResult) -> dict[str, object]:
    """Convert one Stage 12a case result to JSON-safe values."""

    return {
        "example_id": result.example_id,
        "scenario_id": result.scenario_id,
        "question_type": result.question_type,
        "preferred_layer": result.preferred_layer,
        "fact_ids": list(result.fact_ids),
        "top_k": result.top_k,
        "expected_ids": _expected_ids_to_dict(result.expected_ids),
        "retrieved": [_retrieved_record_to_dict(record) for record in result.retrieved],
        "metrics": _case_metrics_to_dict(result.metrics),
        "notes": result.notes,
    }


def stage12_aggregate_metrics_to_dict(metrics: Stage12AggregateMetrics) -> dict[str, object]:
    """Convert Stage 12a aggregate metrics to JSON-safe values."""

    return {
        "total_cases": metrics.total_cases,
        "hit_rate_at_k": metrics.hit_rate_at_k,
        "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
        "mean_precision_at_k": metrics.mean_precision_at_k,
        "mean_recall_at_k": metrics.mean_recall_at_k,
        "raw_hit_rate_at_k": metrics.raw_hit_rate_at_k,
        "summary_hit_rate_at_k": metrics.summary_hit_rate_at_k,
        "acceptable_hit_rate_at_k": metrics.acceptable_hit_rate_at_k,
        "self_query_hit_rate": metrics.self_query_hit_rate,
        "kind_mix_at_k": _kind_mix_to_dict(metrics.kind_mix_at_k),
        "mean_recap_pollution_count_at_k": metrics.mean_recap_pollution_count_at_k,
        "mean_recap_pollution_ratio_at_k": metrics.mean_recap_pollution_ratio_at_k,
    }


def _validate_eval_case(case: RetrievalEvalCase) -> None:
    if case.top_k <= 0:
        raise InvalidRetrievalRequestError("eval case top_k must be greater than zero")
    if not case.expected_episode_ids:
        raise InvalidRetrievalRequestError("eval case expected_episode_ids must not be empty")


def _result_for_case(
    case: RetrievalEvalCase,
    retrieved_episode_ids: tuple[UUID, ...],
) -> RetrievalEvalResult:
    expected_episode_ids = case.expected_episode_ids
    expected_episode_id_set = set(expected_episode_ids)
    retrieved_expected_count = sum(
        1 for episode_id in retrieved_episode_ids if episode_id in expected_episode_id_set
    )
    retrieved_count = len(retrieved_episode_ids)
    first_expected_rank = next(
        (
            rank
            for rank, episode_id in enumerate(retrieved_episode_ids, start=1)
            if episode_id in expected_episode_id_set
        ),
        None,
    )

    return RetrievalEvalResult(
        case_name=case.name,
        retrieved_episode_ids=retrieved_episode_ids,
        expected_episode_ids=expected_episode_ids,
        hit_at_k=retrieved_expected_count > 0,
        precision_at_k=0.0 if retrieved_count == 0 else retrieved_expected_count / retrieved_count,
        recall_at_k=retrieved_expected_count / len(expected_episode_ids),
        reciprocal_rank=0.0 if first_expected_rank is None else 1.0 / first_expected_rank,
    )


def _summarize_results(results: Sequence[RetrievalEvalResult]) -> RetrievalEvalSummary:
    total_cases = len(results)
    if total_cases == 0:
        return RetrievalEvalSummary(
            total_cases=0,
            hit_rate_at_k=0.0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
        )

    return RetrievalEvalSummary(
        total_cases=total_cases,
        hit_rate_at_k=sum(1 for result in results if result.hit_at_k) / total_cases,
        mean_precision_at_k=sum(result.precision_at_k for result in results) / total_cases,
        mean_recall_at_k=sum(result.recall_at_k for result in results) / total_cases,
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in results) / total_cases,
    )


def _stage12_retrieved_record(
    episode: ScoredEpisode,
    label: Stage12ResolvedEpisodeLabel | None,
    official_rank: int | None,
    raw_expected_ids: set[UUID],
    summary_expected_ids: set[UUID],
    acceptable_ids: set[UUID],
    current_query_ids: set[UUID],
    preferred_ids: set[UUID],
    timing_mode: TimingMode,
) -> Stage12RetrievedEpisodeRecord:
    roles = () if label is None else label.roles
    fact_ids = () if label is None else label.fact_ids
    is_current_query = _is_current_query_episode(
        episode=episode,
        label=label,
        current_query_ids=current_query_ids,
        timing_mode=timing_mode,
    )

    return Stage12RetrievedEpisodeRecord(
        episode_id=episode.id,
        kind=episode.kind,
        rank=episode.result_rank,
        official_rank=official_rank,
        conversation_id=episode.conversation_id,
        scope_id=episode.scope_id,
        message_id=episode.message_id,
        message_position=episode.message_position,
        range_start=episode.range_start,
        range_end=episode.range_end,
        score=episode.score,
        similarity=episode.similarity,
        recency_score=episode.recency_score,
        access_score=episode.access_score,
        importance_score=episode.importance_score,
        frequency_score=episode.frequency_score,
        label_roles=roles,
        fact_ids=fact_ids,
        is_raw_expected=episode.id in raw_expected_ids,
        is_summary_expected=episode.id in summary_expected_ids,
        is_preferred_expected=episode.id in preferred_ids,
        is_acceptable=episode.id in acceptable_ids,
        is_current_query=is_current_query,
    )


def _is_current_query_episode(
    episode: ScoredEpisode,
    label: Stage12ResolvedEpisodeLabel | None,
    current_query_ids: set[UUID],
    timing_mode: TimingMode,
) -> bool:
    if timing_mode == "clean_memory":
        return False
    if episode.id in current_query_ids:
        return True
    return label is not None and "current_query" in label.roles


def _preferred_expected_id_set(
    expected_ids: Stage12ExpectedIds,
    preferred_layer: PreferredLayer,
) -> set[UUID]:
    if preferred_layer == "raw":
        return set(expected_ids.raw)
    if preferred_layer == "summary":
        return set(expected_ids.summary)
    return {*expected_ids.raw, *expected_ids.summary}


def _kind_mix_metrics(records: Sequence[Stage12RetrievedEpisodeRecord]) -> Stage12KindMixMetrics:
    message_count = sum(1 for record in records if record.kind == "message")
    summary_count = sum(1 for record in records if record.kind == "summary")
    total_count = message_count + summary_count
    return Stage12KindMixMetrics(
        message_count=message_count,
        summary_count=summary_count,
        total_count=total_count,
        message_ratio=0.0 if total_count == 0 else message_count / total_count,
        summary_ratio=0.0 if total_count == 0 else summary_count / total_count,
    )


def _recap_pollution_metrics(
    records: Sequence[Stage12RetrievedEpisodeRecord],
) -> Stage12RecapPollutionMetrics:
    count = sum(1 for record in records if "recap_question" in record.label_roles)
    total_count = len(records)
    return Stage12RecapPollutionMetrics(
        count=count,
        ratio=0.0 if total_count == 0 else count / total_count,
    )


def _parse_stage12_jsonl_records(records: Sequence[str]) -> Stage12Corpus:
    metadata: dict[str, object] | None = None
    cases: list[Stage12CorpusCase] = []

    for line_number, line in enumerate(records, start=1):
        record = cast(dict[str, object], json.loads(line))
        record_type = str(record.get("record_type", "case"))
        if record_type == "metadata":
            if metadata is not None:
                raise InvalidRetrievalRequestError("Stage 12 corpus has duplicate metadata")
            metadata = record
        elif record_type == "case":
            cases.append(_parse_stage12_case_record(record))
        else:
            raise InvalidRetrievalRequestError(
                f"Stage 12 corpus line {line_number} has unknown record_type: {record_type}"
            )

    if metadata is None:
        raise InvalidRetrievalRequestError("Stage 12 JSONL corpus requires a metadata record")

    return Stage12Corpus(
        corpus_id=_required_str(metadata, "corpus_id"),
        corpus_version=_required_str(metadata, "corpus_version"),
        fixture_strategy=_required_str(metadata, "fixture_strategy"),
        embedding_model=_required_str(metadata, "embedding_model"),
        notes=_optional_str(metadata, "notes"),
        cases=tuple(cases),
    )


def _parse_stage12_corpus_document(record: Mapping[str, object]) -> Stage12Corpus:
    examples = _required_list(record, "examples")
    return Stage12Corpus(
        corpus_id=_required_str(record, "corpus_id"),
        corpus_version=_required_str(record, "corpus_version"),
        fixture_strategy=_required_str(record, "fixture_strategy"),
        embedding_model=_required_str(record, "embedding_model"),
        notes=_optional_str(record, "notes"),
        cases=tuple(_parse_stage12_case_record(_as_mapping(example)) for example in examples),
    )


def _parse_stage12_case_record(record: Mapping[str, object]) -> Stage12CorpusCase:
    preferred_layer = _preferred_layer(_required_str(record, "preferred_layer"))
    question_type = _question_type(_required_str(record, "question_type"))
    return Stage12CorpusCase(
        example_id=_required_str(record, "example_id"),
        scenario_id=_required_str(record, "scenario_id"),
        query=_required_str(record, "query"),
        top_k=_required_int(record, "top_k"),
        question_type=question_type,
        fact_ids=_str_tuple(record.get("fact_ids", ())),
        preferred_layer=preferred_layer,
        expected_refs=Stage12ExpectedRefs(
            raw=_str_tuple(record.get("raw_expected_refs", ())),
            summary=_str_tuple(record.get("summary_expected_refs", ())),
            acceptable=_str_tuple(record.get("acceptable_refs", ())),
            current_query=_str_tuple(record.get("current_query_refs", ())),
        ),
        episode_labels=tuple(
            _parse_stage12_label_record(_as_mapping(label))
            for label in _required_list(record, "episode_labels")
        ),
        notes=_optional_str(record, "notes"),
    )


def _parse_stage12_label_record(record: Mapping[str, object]) -> Stage12CorpusEpisodeLabel:
    roles = tuple(_episode_label_role(role) for role in _str_tuple(record.get("roles", ())))
    episode_kind = _episode_kind(_required_str(record, "episode_kind"))
    layer = _episode_label_layer(_required_str(record, "layer"))
    return Stage12CorpusEpisodeLabel(
        semantic_ref=_required_str(record, "semantic_ref"),
        roles=roles,
        episode_kind=episode_kind,
        layer=layer,
        fact_ids=_str_tuple(record.get("fact_ids", ())),
        message_position=_optional_int(record, "message_position"),
        range_start=_optional_int(record, "range_start"),
        range_end=_optional_int(record, "range_end"),
        is_expected=_optional_bool(record, "is_expected", default=False),
        is_acceptable=_optional_bool(record, "is_acceptable", default=False),
    )


def _validate_corpus_label(label: Stage12CorpusEpisodeLabel) -> None:
    if not label.semantic_ref.strip():
        raise InvalidRetrievalRequestError("Stage 12 label semantic_ref must not be empty")
    if not label.roles:
        raise InvalidRetrievalRequestError(
            f"Stage 12 label must include at least one role: {label.semantic_ref}"
        )
    _validate_roles(label.roles)
    _validate_episode_kind(label.episode_kind)
    _validate_label_layer(label.layer)
    if label.episode_kind == "summary" and (label.range_start is None or label.range_end is None):
        raise InvalidRetrievalRequestError(
            f"Stage 12 summary label requires range_start/range_end: {label.semantic_ref}"
        )


def _all_expected_refs(expected_refs: Stage12ExpectedRefs) -> tuple[str, ...]:
    return (
        *expected_refs.raw,
        *expected_refs.summary,
        *expected_refs.acceptable,
        *expected_refs.current_query,
    )


def _resolve_semantic_ref(semantic_ref: str, ref_to_episode_id: Mapping[str, UUID]) -> UUID:
    try:
        return ref_to_episode_id[semantic_ref]
    except KeyError as exc:
        raise InvalidRetrievalRequestError(
            f"Stage 12 semantic ref was not resolved by fixture: {semantic_ref}"
        ) from exc


def _validate_roles(roles: Sequence[str]) -> None:
    for role in roles:
        if role not in _VALID_EPISODE_LABEL_ROLES:
            raise InvalidRetrievalRequestError(f"Stage 12 unknown episode label role: {role}")


def _validate_episode_kind(episode_kind: str) -> None:
    if episode_kind not in _VALID_EPISODE_KINDS:
        raise InvalidRetrievalRequestError(f"Stage 12 unknown episode kind: {episode_kind}")


def _validate_label_layer(layer: str) -> None:
    if layer not in _VALID_EPISODE_LABEL_LAYERS:
        raise InvalidRetrievalRequestError(f"Stage 12 unknown episode label layer: {layer}")


def _validate_preferred_layer(preferred_layer: str) -> None:
    if preferred_layer not in _VALID_PREFERRED_LAYERS:
        raise InvalidRetrievalRequestError(f"Stage 12 unknown preferred layer: {preferred_layer}")


def _validate_question_type(question_type: str) -> None:
    if question_type not in _VALID_QUESTION_TYPES:
        raise InvalidRetrievalRequestError(f"Stage 12 unknown question type: {question_type}")


def _validate_timing_mode(timing_mode: str) -> None:
    if timing_mode not in _VALID_TIMING_MODES:
        raise InvalidRetrievalRequestError(f"Stage 12 unknown timing mode: {timing_mode}")


def _episode_label_role(value: str) -> EpisodeLabelRole:
    _validate_roles((value,))
    return cast(EpisodeLabelRole, value)


def _episode_label_layer(value: str) -> EpisodeLabelLayer:
    _validate_label_layer(value)
    return cast(EpisodeLabelLayer, value)


def _preferred_layer(value: str) -> PreferredLayer:
    _validate_preferred_layer(value)
    return cast(PreferredLayer, value)


def _question_type(value: str) -> QuestionType:
    _validate_question_type(value)
    return cast(QuestionType, value)


def _episode_kind(value: str) -> EpisodeKind:
    _validate_episode_kind(value)
    return cast(EpisodeKind, value)


def _required_str(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a string: {key}")
    return value


def _optional_str(record: Mapping[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a string: {key}")
    return value


def _required_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be an integer: {key}")
    return value


def _optional_int(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be an integer: {key}")
    return value


def _optional_bool(record: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = record.get(key, default)
    if not isinstance(value, bool):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a boolean: {key}")
    return value


def _required_list(record: Mapping[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a list: {key}")
    return value


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise InvalidRetrievalRequestError("Stage 12 list item must be an object")
    return cast(Mapping[str, object], value)


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise InvalidRetrievalRequestError("Stage 12 field must be a string list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise InvalidRetrievalRequestError("Stage 12 field must be a string list")
        result.append(item)
    return tuple(result)


def _expected_ids_to_dict(expected_ids: Stage12ExpectedIds) -> dict[str, object]:
    return {
        "raw": [str(episode_id) for episode_id in expected_ids.raw],
        "summary": [str(episode_id) for episode_id in expected_ids.summary],
        "acceptable": [str(episode_id) for episode_id in expected_ids.acceptable],
        "current_query": [str(episode_id) for episode_id in expected_ids.current_query],
    }


def _retrieved_record_to_dict(record: Stage12RetrievedEpisodeRecord) -> dict[str, object]:
    return {
        "episode_id": str(record.episode_id),
        "kind": record.kind,
        "rank": record.rank,
        "official_rank": record.official_rank,
        "conversation_id": str(record.conversation_id),
        "scope_id": str(record.scope_id),
        "message_id": None if record.message_id is None else str(record.message_id),
        "message_position": record.message_position,
        "range_start": record.range_start,
        "range_end": record.range_end,
        "score": record.score,
        "similarity": record.similarity,
        "score_components": {
            "recency": record.recency_score,
            "access": record.access_score,
            "importance": record.importance_score,
            "frequency": record.frequency_score,
        },
        "expected_flags": {
            "raw": record.is_raw_expected,
            "summary": record.is_summary_expected,
            "preferred": record.is_preferred_expected,
            "acceptable": record.is_acceptable,
            "current_query": record.is_current_query,
        },
        "label_roles": list(record.label_roles),
        "fact_ids": list(record.fact_ids),
    }


def _case_metrics_to_dict(metrics: Stage12CaseMetrics) -> dict[str, object]:
    return {
        "hit_at_k": metrics.hit_at_k,
        "reciprocal_rank": metrics.reciprocal_rank,
        "precision_at_k": metrics.precision_at_k,
        "recall_at_k": metrics.recall_at_k,
        "raw_hit_at_k": metrics.raw_hit_at_k,
        "summary_hit_at_k": metrics.summary_hit_at_k,
        "acceptable_hit_at_k": metrics.acceptable_hit_at_k,
        "kind_mix_at_k": _kind_mix_to_dict(metrics.kind_mix_at_k),
        "self_query_hit": metrics.self_query_hit,
        "self_query_rank": metrics.self_query_rank,
        "self_query_similarity": metrics.self_query_similarity,
        "recap_pollution_at_k": _recap_pollution_to_dict(metrics.recap_pollution_at_k),
    }


def _kind_mix_to_dict(metrics: Stage12KindMixMetrics) -> dict[str, object]:
    return {
        "message_count": metrics.message_count,
        "summary_count": metrics.summary_count,
        "total_count": metrics.total_count,
        "message_ratio": metrics.message_ratio,
        "summary_ratio": metrics.summary_ratio,
    }


def _recap_pollution_to_dict(metrics: Stage12RecapPollutionMetrics) -> dict[str, object]:
    return {
        "count": metrics.count,
        "ratio": metrics.ratio,
    }
