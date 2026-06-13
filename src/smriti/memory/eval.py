from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
Stage12FailureClass = Literal[
    "self_query_artifact",
    "hit_at_official_k",
    "rerank_below_official_k",
    "absent_from_diagnostic_top_k",
    "recap_or_echo_pollution",
    "summary_visible_but_low",
    "raw_visible_but_low",
    "negative_control",
    "unclassified",
]
Stage12WeightSweepRecommendation = Literal[
    "no_change_recommended",
    "candidate_profile_for_future_stage",
    "weight_sweep_insufficient",
]
Stage13RolePolicyDemotionStrategy = Literal["move_behind_non_weak"]
Stage13FirstBlockingWeakEvidenceRole = Literal["recap", "assistant_echo", "both", "none"]
Stage13RolePolicyRecommendation = Literal[
    "no_role_policy_change_recommended",
    "assistant_echo_policy_candidate",
    "recap_policy_requires_metadata",
    "role_policy_insufficient",
]

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
_VALID_STAGE12_FAILURE_CLASSES = frozenset(
    {
        "self_query_artifact",
        "hit_at_official_k",
        "rerank_below_official_k",
        "absent_from_diagnostic_top_k",
        "recap_or_echo_pollution",
        "summary_visible_but_low",
        "raw_visible_but_low",
        "negative_control",
        "unclassified",
    }
)
STAGE12_WEIGHT_SWEEP_TARGET_CASES = (
    "terrafold_f5_dele_bookkeeping",
    "terrafold_broad_operational_constraints",
    "terrafold_either_opening_classes",
    "terrafold_f4_latex_allergy",
)
STAGE13_ROLE_POLICY_TARGET_CASES = STAGE12_WEIGHT_SWEEP_TARGET_CASES
_MATERIAL_MRR_DROP = 0.01
_METRIC_EPSILON = 1e-9


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
class Stage12DiagnosticMetrics:
    diagnostic_top_k: int
    expected_in_diagnostic_top_k: bool
    raw_expected_in_diagnostic_top_k: bool
    summary_expected_in_diagnostic_top_k: bool
    acceptable_in_diagnostic_top_k: bool
    first_expected_diagnostic_rank: int | None
    first_raw_diagnostic_rank: int | None
    first_summary_diagnostic_rank: int | None
    first_acceptable_diagnostic_rank: int | None
    diagnostic_kind_mix: Stage12KindMixMetrics
    diagnostic_recap_pollution: Stage12RecapPollutionMetrics
    failure_class: tuple[Stage12FailureClass, ...]


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
    diagnostics: Stage12DiagnosticMetrics
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
    diagnostic_top_k: int | None = None


@dataclass(frozen=True)
class Stage12WeightProfile:
    name: str
    similarity: float
    recency: float
    access: float
    importance: float
    frequency: float


@dataclass(frozen=True)
class Stage13RolePolicy:
    name: str
    description: str
    excluded_roles: tuple[EpisodeLabelRole, ...] = ()
    demoted_roles: tuple[EpisodeLabelRole, ...] = ()
    demotion_strategy: Stage13RolePolicyDemotionStrategy | None = None
    depends_on_recap_metadata: bool = False


@dataclass(frozen=True)
class Stage13WeakEvidenceMetrics:
    recap_above_first_acceptable_count: int
    assistant_echo_above_first_acceptable_count: int
    total_weak_evidence_above_first_acceptable_count: int
    recap_official_top_k_count: int
    assistant_echo_official_top_k_count: int
    recap_diagnostic_top_k_count: int
    assistant_echo_diagnostic_top_k_count: int
    first_blocking_weak_evidence_role: Stage13FirstBlockingWeakEvidenceRole


@dataclass(frozen=True)
class Stage13RolePolicyCaseResult:
    policy_name: str
    case_result: Stage12CaseResult
    weak_evidence: Stage13WeakEvidenceMetrics
    removed_legitimate_raw_source_count: int


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
    diagnostic_top_k: int | None = None,
) -> tuple[list[Stage12CaseResult], Stage12AggregateMetrics]:
    """Evaluate Stage 12a retrieval cases through the current memory service path."""

    _validate_timing_mode(timing_mode)
    for case in cases:
        validate_stage12_eval_case(case)

    results: list[Stage12CaseResult] = []
    for case in cases:
        case_diagnostic_top_k = _resolve_diagnostic_top_k(
            top_k=case.top_k,
            diagnostic_top_k=diagnostic_top_k,
        )
        retrieved = await service.retrieve_scoped_episodes(
            user_id=case.user_id,
            scope_id=case.scope_id,
            query=case.query,
            top_k=case_diagnostic_top_k,
            now=now,
        )
        results.append(
            score_stage12_retrieval_case(
                case,
                retrieved,
                timing_mode,
                diagnostic_top_k=case_diagnostic_top_k,
            )
        )

    return results, summarize_stage12_results(results)


def score_stage12_retrieval_case(
    case: Stage12ResolvedEvalCase,
    retrieved: Sequence[ScoredEpisode],
    timing_mode: TimingMode = "app_realistic",
    *,
    diagnostic_top_k: int | None = None,
) -> Stage12CaseResult:
    """Score one Stage 12a case without changing retrieval behavior."""

    validate_stage12_eval_case(case)
    _validate_timing_mode(timing_mode)

    case_diagnostic_top_k = _resolve_diagnostic_top_k(
        top_k=case.top_k,
        diagnostic_top_k=diagnostic_top_k,
    )
    diagnostic_retrieved = tuple(retrieved[:case_diagnostic_top_k])
    label_by_episode_id = {label.episode_id: label for label in case.episode_labels}
    raw_expected_ids = set(case.expected_ids.raw)
    summary_expected_ids = set(case.expected_ids.summary)
    acceptable_ids = set(case.expected_ids.acceptable)
    current_query_ids = set(case.expected_ids.current_query)
    preferred_ids = _preferred_expected_id_set(case.expected_ids, case.preferred_layer)

    diagnostic_official_episodes = tuple(
        episode
        for episode in diagnostic_retrieved
        if not _is_current_query_episode(
            episode=episode,
            label=label_by_episode_id.get(episode.id),
            current_query_ids=current_query_ids,
            timing_mode=timing_mode,
        )
    )
    official_rank_by_episode_id = {
        episode.id: official_rank
        for official_rank, episode in enumerate(diagnostic_official_episodes, start=1)
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
        for episode in diagnostic_retrieved
    )

    official_records = tuple(
        record
        for record in retrieved_records
        if record.rank <= case.top_k and record.official_rank is not None
    )
    diagnostic_official_records = tuple(
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
        (
            record
            for record in retrieved_records
            if record.rank <= case.top_k and record.is_current_query
        ),
        None,
    )
    recap_pollution = _recap_pollution_metrics(official_records)
    metrics = Stage12CaseMetrics(
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
        self_query_similarity=None if self_query_record is None else self_query_record.similarity,
        recap_pollution_at_k=recap_pollution,
    )
    diagnostics = _stage12_diagnostic_metrics(
        top_k=case.top_k,
        diagnostic_top_k=case_diagnostic_top_k,
        timing_mode=timing_mode,
        expected_ids=case.expected_ids,
        preferred_ids=preferred_ids,
        records=retrieved_records,
        diagnostic_official_records=diagnostic_official_records,
    )

    return Stage12CaseResult(
        example_id=case.example_id,
        scenario_id=case.scenario_id,
        question_type=case.question_type,
        preferred_layer=case.preferred_layer,
        fact_ids=case.fact_ids,
        top_k=case.top_k,
        expected_ids=case.expected_ids,
        retrieved=retrieved_records,
        metrics=metrics,
        diagnostics=diagnostics,
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


def stage12_weight_sweep_profiles() -> tuple[Stage12WeightProfile, ...]:
    """Return the fixed Stage 12b-2 diagnostic profile grid."""

    return (
        Stage12WeightProfile(
            name="baseline",
            similarity=0.55,
            recency=0.20,
            access=0.10,
            importance=0.10,
            frequency=0.05,
        ),
        Stage12WeightProfile(
            name="similarity_only",
            similarity=1.00,
            recency=0.00,
            access=0.00,
            importance=0.00,
            frequency=0.00,
        ),
        Stage12WeightProfile(
            name="similarity_heavy",
            similarity=0.70,
            recency=0.10,
            access=0.05,
            importance=0.10,
            frequency=0.05,
        ),
        Stage12WeightProfile(
            name="recency_light",
            similarity=0.65,
            recency=0.05,
            access=0.10,
            importance=0.15,
            frequency=0.05,
        ),
        Stage12WeightProfile(
            name="recency_heavy_control",
            similarity=0.45,
            recency=0.35,
            access=0.05,
            importance=0.10,
            frequency=0.05,
        ),
        Stage12WeightProfile(
            name="no_access_frequency",
            similarity=0.65,
            recency=0.25,
            access=0.00,
            importance=0.10,
            frequency=0.00,
        ),
    )


def stage12_weight_profile_to_dict(profile: Stage12WeightProfile) -> dict[str, object]:
    """Convert a Stage 12b-2 weight profile to JSON-safe values."""

    return {
        "name": profile.name,
        "weights": {
            "similarity": profile.similarity,
            "recency": profile.recency,
            "access": profile.access,
            "importance": profile.importance,
            "frequency": profile.frequency,
        },
    }


def score_stage12_weight_profile_record(
    record: Stage12RetrievedEpisodeRecord,
    profile: Stage12WeightProfile,
) -> float:
    """Replay the Python-side weighted score for one emitted diagnostic record."""

    return (
        profile.similarity * record.similarity
        + profile.recency * record.recency_score
        + profile.access * record.access_score
        + profile.importance * record.importance_score
        + profile.frequency * record.frequency_score
    )


def replay_stage12_weight_profile_case(
    case: Stage12CaseResult,
    profile: Stage12WeightProfile,
    *,
    official_top_k: int | None = None,
    diagnostic_top_k: int | None = None,
) -> Stage12CaseResult:
    """Replay one case against a diagnostic profile without calling retrieval."""

    top_k = _resolve_stage12_weight_sweep_top_k(case.top_k, official_top_k)
    case_diagnostic_top_k = _resolve_diagnostic_top_k(
        top_k=top_k,
        diagnostic_top_k=diagnostic_top_k
        if diagnostic_top_k is not None
        else case.diagnostics.diagnostic_top_k,
    )
    candidates = tuple(sorted(case.retrieved, key=lambda record: record.rank))[
        :case_diagnostic_top_k
    ]
    scored_candidates = tuple(
        (
            score_stage12_weight_profile_record(record, profile),
            record.rank,
            record,
        )
        for record in candidates
    )
    reranked_candidates = tuple(
        item[2] for item in sorted(scored_candidates, key=lambda item: (-item[0], item[1]))
    )

    official_rank = 0
    replayed_records: list[Stage12RetrievedEpisodeRecord] = []
    for rank, record in enumerate(reranked_candidates, start=1):
        profile_score = score_stage12_weight_profile_record(record, profile)
        if record.is_current_query:
            new_official_rank: int | None = None
        else:
            official_rank += 1
            new_official_rank = official_rank
        replayed_records.append(
            replace(
                record,
                rank=rank,
                official_rank=new_official_rank,
                score=profile_score,
            )
        )

    return _stage12_case_result_from_records(
        source=case,
        records=tuple(replayed_records),
        top_k=top_k,
        diagnostic_top_k=case_diagnostic_top_k,
    )


def run_stage12_weight_sweep(
    *,
    input_metadata: Mapping[str, object],
    cases: Sequence[Stage12CaseResult],
    input_run_path: str | None = None,
    official_top_k: int | None = None,
    diagnostic_top_k: int | None = None,
    profiles: Sequence[Stage12WeightProfile] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Compare fixed Stage 12b-2 profiles against emitted diagnostic records."""

    profile_grid = tuple(profiles) if profiles is not None else stage12_weight_sweep_profiles()
    if not profile_grid:
        raise InvalidRetrievalRequestError("Stage 12 weight sweep requires at least one profile")
    profile_names = [profile.name for profile in profile_grid]
    if len(set(profile_names)) != len(profile_names):
        raise InvalidRetrievalRequestError("Stage 12 weight sweep profile names must be unique")
    baseline_profile = next(
        (profile for profile in profile_grid if profile.name == "baseline"),
        None,
    )
    if baseline_profile is None:
        raise InvalidRetrievalRequestError("Stage 12 weight sweep requires a baseline profile")

    input_run_id = _required_str(input_metadata, "run_id")
    created = datetime.now(UTC) if created_at is None else created_at
    results_by_profile = {
        profile.name: tuple(
            replay_stage12_weight_profile_case(
                case,
                profile,
                official_top_k=official_top_k,
                diagnostic_top_k=diagnostic_top_k,
            )
            for case in cases
        )
        for profile in profile_grid
    }
    aggregate_by_profile = {
        profile_name: _weight_sweep_aggregate_to_dict(results)
        for profile_name, results in results_by_profile.items()
    }
    baseline_results = results_by_profile["baseline"]
    baseline_aggregate = summarize_stage12_results(baseline_results)
    regression_gates = {
        profile.name: _stage12_weight_sweep_regression_gates(
            input_metadata=input_metadata,
            baseline_results=baseline_results,
            baseline_aggregate=baseline_aggregate,
            profile_results=results_by_profile[profile.name],
            profile_aggregate=summarize_stage12_results(results_by_profile[profile.name]),
            is_baseline=profile.name == "baseline",
        )
        for profile in profile_grid
    }
    rank_movement = _stage12_weight_sweep_rank_movement(results_by_profile)
    recommendation = _stage12_weight_sweep_recommendation(
        baseline_results=baseline_results,
        baseline_aggregate=baseline_aggregate,
        aggregate_by_profile={
            profile_name: summarize_stage12_results(results)
            for profile_name, results in results_by_profile.items()
        },
        regression_gates=regression_gates,
    )

    return {
        "metadata": {
            "run_id": f"{input_run_id}-weight-sweep",
            "created_at": created.astimezone(UTC).isoformat(),
            "input_run_id": input_run_id,
            "input_diagnostic_run": input_run_path,
            "input_timing_mode": input_metadata.get("timing_mode"),
            "input_embedder_mode": input_metadata.get("embedder_mode"),
            "input_embedding_model": input_metadata.get("embedding_model"),
            "input_git_branch": input_metadata.get("git_branch"),
            "input_git_sha": input_metadata.get("git_sha"),
            "official_top_k_override": official_top_k,
            "diagnostic_top_k_override": diagnostic_top_k,
            "profile_definitions": [
                stage12_weight_profile_to_dict(profile) for profile in profile_grid
            ],
            "target_cases": list(STAGE12_WEIGHT_SWEEP_TARGET_CASES),
        },
        "input_run_metadata": dict(input_metadata),
        "aggregate_metrics_by_profile": aggregate_by_profile,
        "per_case_rank_movement": rank_movement,
        "target_case_focus": [
            item
            for item in rank_movement
            if item["example_id"] in STAGE12_WEIGHT_SWEEP_TARGET_CASES
        ],
        "regression_gates": regression_gates,
        "recommendation": recommendation,
    }


def stage12_weight_sweep_report_to_markdown(payload: Mapping[str, object]) -> str:
    """Render a concise Markdown report for a Stage 12b-2 weight sweep."""

    metadata = _required_mapping(payload, "metadata")
    aggregates = _required_mapping(payload, "aggregate_metrics_by_profile")
    gates = _required_mapping(payload, "regression_gates")
    rank_movement = _required_list(payload, "per_case_rank_movement")
    target_focus = _required_list(payload, "target_case_focus")
    recommendation = _required_mapping(payload, "recommendation")
    profile_definitions = _required_list(metadata, "profile_definitions")

    lines = [
        f"# Stage 12b-2 Weight Sweep: {_required_str(metadata, 'input_run_id')}",
        "",
        "## Metadata",
        "",
        f"- Input diagnostic run: {metadata.get('input_diagnostic_run')}",
        f"- Timing mode: {metadata.get('input_timing_mode')}",
        (
            f"- Embedder: {metadata.get('input_embedder_mode')} "
            f"({metadata.get('input_embedding_model')})"
        ),
        f"- Official top-k override: {metadata.get('official_top_k_override')}",
        f"- Diagnostic top-k override: {metadata.get('diagnostic_top_k_override')}",
        f"- Git SHA: {metadata.get('input_git_sha')}",
        "",
        "## Profile Definitions",
        "",
        "| Profile | Similarity | Recency | Access | Importance | Frequency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in profile_definitions:
        profile = _as_mapping(item)
        weights = _required_mapping(profile, "weights")
        lines.append(
            f"| {profile['name']} | {_format_metric(weights['similarity'])} | "
            f"{_format_metric(weights['recency'])} | {_format_metric(weights['access'])} | "
            f"{_format_metric(weights['importance'])} | "
            f"{_format_metric(weights['frequency'])} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics By Profile",
            "",
            (
                "| Profile | Hit@K | Acceptable@K | Raw@K | Summary@K | MRR | "
                "Precision | Recall | Recap/Echo Before Evidence | Kind Mix |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for profile_name, aggregate_value in aggregates.items():
        aggregate = _as_mapping(aggregate_value)
        kind_mix = _required_mapping(aggregate, "kind_mix_at_k")
        lines.append(
            f"| {profile_name} | {_format_metric(aggregate['hit_rate_at_k'])} | "
            f"{_format_metric(aggregate['acceptable_hit_rate_at_k'])} | "
            f"{_format_metric(aggregate['raw_hit_rate_at_k'])} | "
            f"{_format_metric(aggregate['summary_hit_rate_at_k'])} | "
            f"{_format_metric(aggregate['mean_reciprocal_rank'])} | "
            f"{_format_metric(aggregate['mean_precision_at_k'])} | "
            f"{_format_metric(aggregate['mean_recall_at_k'])} | "
            f"{aggregate['recap_echo_above_first_acceptable_count']} | "
            f"{kind_mix['message_count']} message / {kind_mix['summary_count']} summary |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Rank Movement",
            "",
            (
                "| Example | Top K | Profile | First Acceptable | First Raw | "
                "First Summary | Recap/Echo Before Evidence |"
            ),
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in rank_movement:
        movement = _as_mapping(item)
        profile_ranks = _required_mapping(movement, "profiles")
        for profile_name, rank_value in profile_ranks.items():
            ranks = _as_mapping(rank_value)
            lines.append(
                f"| {movement['example_id']} | {movement['top_k']} | {profile_name} | "
                f"{_format_metric(ranks['first_acceptable_rank'])} | "
                f"{_format_metric(ranks['first_raw_rank'])} | "
                f"{_format_metric(ranks['first_summary_rank'])} | "
                f"{ranks['recap_echo_above_first_acceptable']} |"
            )

    lines.extend(
        [
            "",
            "## Target-Case Focus",
            "",
            "| Example | Baseline First Acceptable | Best First Acceptable | Best Profile |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for item in target_focus:
        movement = _as_mapping(item)
        lines.append(
            f"| {movement['example_id']} | "
            f"{_format_metric(movement['baseline_first_acceptable_rank'])} | "
            f"{_format_metric(movement['best_first_acceptable_rank'])} | "
            f"{movement['best_first_acceptable_profile']} |"
        )

    lines.extend(
        [
            "",
            "## Regression Gates",
            "",
            "| Profile | Recommended | Failed Gates |",
            "| --- | ---: | --- |",
        ]
    )
    for profile_name, gate_value in gates.items():
        gate_record = _as_mapping(gate_value)
        failed = gate_record["failed_gates"]
        failed_text = ", ".join(str(item) for item in failed) if isinstance(failed, list) else ""
        lines.append(f"| {profile_name} | {gate_record['recommended']} | {failed_text or 'none'} |")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Decision: {recommendation['decision']}",
            f"- Profile: {recommendation.get('profile')}",
            f"- Explanation: {recommendation['explanation']}",
            "",
        ]
    )
    return "\n".join(lines)


def stage13_role_policies() -> tuple[Stage13RolePolicy, ...]:
    """Return eval-only Stage 13b role-aware replay policies."""

    return (
        Stage13RolePolicy(
            name="baseline",
            description="Keep the emitted diagnostic ordering and current-query exclusion.",
        ),
        Stage13RolePolicy(
            name="exclude_assistant_echo",
            description="Remove assistant-answer echo labels from official replay ranking.",
            excluded_roles=("assistant_answer_echo",),
        ),
        Stage13RolePolicy(
            name="exclude_recap_question",
            description="Remove recap-question labels from official replay ranking.",
            excluded_roles=("recap_question",),
            depends_on_recap_metadata=True,
        ),
        Stage13RolePolicy(
            name="exclude_recap_and_echo",
            description="Remove recap-question and assistant-answer echo labels from ranking.",
            excluded_roles=("recap_question", "assistant_answer_echo"),
            depends_on_recap_metadata=True,
        ),
        Stage13RolePolicy(
            name="demote_assistant_echo",
            description="Move assistant-answer echo labels behind non-weak evidence.",
            demoted_roles=("assistant_answer_echo",),
            demotion_strategy="move_behind_non_weak",
        ),
        Stage13RolePolicy(
            name="demote_recap_question",
            description="Move recap-question labels behind non-weak evidence.",
            demoted_roles=("recap_question",),
            demotion_strategy="move_behind_non_weak",
            depends_on_recap_metadata=True,
        ),
    )


def stage13_role_policy_to_dict(policy: Stage13RolePolicy) -> dict[str, object]:
    """Convert one eval-only role policy to JSON-safe values."""

    return {
        "name": policy.name,
        "description": policy.description,
        "excluded_roles": list(policy.excluded_roles),
        "demoted_roles": list(policy.demoted_roles),
        "demotion_strategy": policy.demotion_strategy,
        "depends_on_recap_metadata": policy.depends_on_recap_metadata,
    }


def replay_stage13_role_policy_case(
    case: Stage12CaseResult,
    policy: Stage13RolePolicy,
    *,
    official_top_k: int | None = None,
    diagnostic_top_k: int | None = None,
) -> Stage13RolePolicyCaseResult:
    """Replay one emitted diagnostic case under an eval-only role policy."""

    top_k = _resolve_stage12_weight_sweep_top_k(case.top_k, official_top_k)
    case_diagnostic_top_k = _resolve_diagnostic_top_k(
        top_k=top_k,
        diagnostic_top_k=diagnostic_top_k
        if diagnostic_top_k is not None
        else case.diagnostics.diagnostic_top_k,
    )
    candidates = tuple(sorted(case.retrieved, key=lambda record: record.rank))[
        :case_diagnostic_top_k
    ]
    ordered_candidates = _stage13_role_policy_ordered_candidates(candidates, policy)

    official_rank = 0
    replayed_records: list[Stage12RetrievedEpisodeRecord] = []
    for replay_rank, record in enumerate(ordered_candidates, start=1):
        if record.is_current_query or _stage13_record_has_any_role(
            record,
            policy.excluded_roles,
        ):
            new_official_rank: int | None = None
        else:
            official_rank += 1
            new_official_rank = official_rank
        replayed_records.append(
            replace(
                record,
                rank=replay_rank,
                official_rank=new_official_rank,
            )
        )

    replayed = _stage12_case_result_from_records(
        source=case,
        records=tuple(replayed_records),
        top_k=top_k,
        diagnostic_top_k=case_diagnostic_top_k,
    )
    return Stage13RolePolicyCaseResult(
        policy_name=policy.name,
        case_result=replayed,
        weak_evidence=stage13_weak_evidence_metrics(replayed),
        removed_legitimate_raw_source_count=_stage13_removed_legitimate_raw_source_count(
            candidates,
            policy,
        ),
    )


def stage13_weak_evidence_metrics(result: Stage12CaseResult) -> Stage13WeakEvidenceMetrics:
    """Measure recap-question and assistant-echo pollution separately."""

    diagnostic_records = tuple(
        record for record in result.retrieved if record.official_rank is not None
    )
    official_records = tuple(
        record
        for record in diagnostic_records
        if record.official_rank is not None and record.official_rank <= result.top_k
    )
    first_acceptable_rank = result.diagnostics.first_acceptable_diagnostic_rank
    weak_before_acceptable = tuple(
        record
        for record in diagnostic_records
        if _stage13_record_is_weak_evidence(record)
        and (
            first_acceptable_rank is None
            or (record.official_rank is not None and record.official_rank < first_acceptable_rank)
        )
    )

    return Stage13WeakEvidenceMetrics(
        recap_above_first_acceptable_count=_stage13_count_role(
            weak_before_acceptable,
            "recap_question",
        ),
        assistant_echo_above_first_acceptable_count=_stage13_count_role(
            weak_before_acceptable,
            "assistant_answer_echo",
        ),
        total_weak_evidence_above_first_acceptable_count=len(weak_before_acceptable),
        recap_official_top_k_count=_stage13_count_role(official_records, "recap_question"),
        assistant_echo_official_top_k_count=_stage13_count_role(
            official_records,
            "assistant_answer_echo",
        ),
        recap_diagnostic_top_k_count=_stage13_count_role(
            diagnostic_records,
            "recap_question",
        ),
        assistant_echo_diagnostic_top_k_count=_stage13_count_role(
            diagnostic_records,
            "assistant_answer_echo",
        ),
        first_blocking_weak_evidence_role=_stage13_first_blocking_weak_evidence_role(
            weak_before_acceptable
        ),
    )


def run_stage13_role_policy_replay(
    *,
    input_metadata: Mapping[str, object],
    cases: Sequence[Stage12CaseResult],
    input_run_path: str | None = None,
    official_top_k: int | None = None,
    diagnostic_top_k: int | None = None,
    policies: Sequence[Stage13RolePolicy] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Replay Stage 12 diagnostic records under eval-only role-aware policies."""

    policy_grid = tuple(policies) if policies is not None else stage13_role_policies()
    if not policy_grid:
        raise InvalidRetrievalRequestError("Stage 13 role replay requires at least one policy")
    policy_names = [policy.name for policy in policy_grid]
    if len(set(policy_names)) != len(policy_names):
        raise InvalidRetrievalRequestError("Stage 13 role replay policy names must be unique")
    if "baseline" not in set(policy_names):
        raise InvalidRetrievalRequestError("Stage 13 role replay requires a baseline policy")

    input_run_id = _required_str(input_metadata, "run_id")
    created = datetime.now(UTC) if created_at is None else created_at
    results_by_policy = {
        policy.name: tuple(
            replay_stage13_role_policy_case(
                case,
                policy,
                official_top_k=official_top_k,
                diagnostic_top_k=diagnostic_top_k,
            )
            for case in cases
        )
        for policy in policy_grid
    }
    baseline_results = results_by_policy["baseline"]
    aggregate_by_policy = {
        policy.name: _stage13_role_policy_aggregate_to_dict(
            baseline_results=baseline_results,
            policy_results=results_by_policy[policy.name],
        )
        for policy in policy_grid
    }
    regression_gates = {
        policy.name: _stage13_role_policy_regression_gates(
            policy=policy,
            baseline_results=baseline_results,
            policy_results=results_by_policy[policy.name],
        )
        for policy in policy_grid
    }
    per_case_comparison = _stage13_role_policy_case_comparison(results_by_policy)

    return {
        "metadata": {
            "run_id": f"{input_run_id}-role-policy-replay",
            "created_at": created.astimezone(UTC).isoformat(),
            "input_run_id": input_run_id,
            "input_diagnostic_run": input_run_path,
            "input_timing_mode": input_metadata.get("timing_mode"),
            "input_embedder_mode": input_metadata.get("embedder_mode"),
            "input_embedding_model": input_metadata.get("embedding_model"),
            "input_git_branch": input_metadata.get("git_branch"),
            "input_git_sha": input_metadata.get("git_sha"),
            "official_top_k_override": official_top_k,
            "diagnostic_top_k_override": diagnostic_top_k,
            "policy_definitions": [stage13_role_policy_to_dict(policy) for policy in policy_grid],
            "target_cases": list(STAGE13_ROLE_POLICY_TARGET_CASES),
        },
        "input_run_metadata": dict(input_metadata),
        "aggregate_metrics_by_policy": aggregate_by_policy,
        "per_case_policy_comparison": per_case_comparison,
        "weak_evidence_pollution": _stage13_role_policy_weak_evidence_table(results_by_policy),
        "target_case_focus": [
            item
            for item in per_case_comparison
            if item["example_id"] in STAGE13_ROLE_POLICY_TARGET_CASES
        ],
        "regression_gates": regression_gates,
        "recommendation": _stage13_role_policy_recommendation(
            policies=policy_grid,
            aggregate_by_policy=aggregate_by_policy,
            regression_gates=regression_gates,
        ),
    }


def stage13_role_policy_replay_report_to_markdown(payload: Mapping[str, object]) -> str:
    """Render a Markdown report for Stage 13b role-policy replay output."""

    metadata = _required_mapping(payload, "metadata")
    aggregates = _required_mapping(payload, "aggregate_metrics_by_policy")
    policy_definitions = _required_list(metadata, "policy_definitions")
    per_case = _required_list(payload, "per_case_policy_comparison")
    weak_table = _required_list(payload, "weak_evidence_pollution")
    target_focus = _required_list(payload, "target_case_focus")
    gates = _required_mapping(payload, "regression_gates")
    recommendation = _required_mapping(payload, "recommendation")

    lines = [
        f"# Stage 13b Role Policy Replay: {_required_str(metadata, 'input_run_id')}",
        "",
        "## Metadata",
        "",
        f"- Input diagnostic run: {metadata.get('input_diagnostic_run')}",
        f"- Timing mode: {metadata.get('input_timing_mode')}",
        (
            f"- Embedder: {metadata.get('input_embedder_mode')} "
            f"({metadata.get('input_embedding_model')})"
        ),
        f"- Official top-k override: {metadata.get('official_top_k_override')}",
        f"- Diagnostic top-k override: {metadata.get('diagnostic_top_k_override')}",
        f"- Git SHA: {metadata.get('input_git_sha')}",
        "",
        "## Policy Definitions",
        "",
        "| Policy | Excluded Roles | Demoted Roles | Recap Metadata |",
        "| --- | --- | --- | ---: |",
    ]
    for item in policy_definitions:
        policy = _as_mapping(item)
        lines.append(
            f"| {policy['name']} | {_format_role_list(policy.get('excluded_roles'))} | "
            f"{_format_role_list(policy.get('demoted_roles'))} | "
            f"{policy.get('depends_on_recap_metadata')} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics By Policy",
            "",
            (
                "| Policy | Acceptable@K | Raw@K | Summary@K | MRR | "
                "Weak Before Evidence | Recap Before | Echo Before | Direct Regressions | "
                "Negative Control Retained |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy_name, aggregate_value in aggregates.items():
        aggregate = _as_mapping(aggregate_value)
        lines.append(
            f"| {policy_name} | {_format_metric(aggregate['acceptable_hit_rate_at_k'])} | "
            f"{_format_metric(aggregate['raw_hit_rate_at_k'])} | "
            f"{_format_metric(aggregate['summary_hit_rate_at_k'])} | "
            f"{_format_metric(aggregate['mean_reciprocal_rank'])} | "
            f"{aggregate['total_weak_evidence_above_first_acceptable_count']} | "
            f"{aggregate['recap_above_first_acceptable_count']} | "
            f"{aggregate['assistant_echo_above_first_acceptable_count']} | "
            f"{aggregate['direct_fact_regression_count']} | "
            f"{aggregate['negative_control_no_hit_retained']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Policy Comparison",
            "",
            (
                "| Example | Policy | Acceptable@K | First Acceptable | First Raw | "
                "First Summary | Weak Before Evidence | First Blocking Role |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in per_case:
        comparison = _as_mapping(item)
        policies = _required_mapping(comparison, "policies")
        for policy_name, policy_value in policies.items():
            policy = _as_mapping(policy_value)
            weak = _required_mapping(policy, "weak_evidence")
            lines.append(
                f"| {comparison['example_id']} | {policy_name} | "
                f"{policy['acceptable_hit_at_k']} | "
                f"{_format_metric(policy['first_acceptable_rank'])} | "
                f"{_format_metric(policy['first_raw_rank'])} | "
                f"{_format_metric(policy['first_summary_rank'])} | "
                f"{weak['total_weak_evidence_above_first_acceptable_count']} | "
                f"{weak['first_blocking_weak_evidence_role']} |"
            )

    lines.extend(
        [
            "",
            "## Weak-Evidence Pollution",
            "",
            (
                "| Example | Policy | Recap Top K | Echo Top K | Recap Diagnostic | "
                "Echo Diagnostic | Recap Before | Echo Before |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in weak_table:
        row = _as_mapping(item)
        lines.append(
            f"| {row['example_id']} | {row['policy']} | "
            f"{row['recap_official_top_k_count']} | "
            f"{row['assistant_echo_official_top_k_count']} | "
            f"{row['recap_diagnostic_top_k_count']} | "
            f"{row['assistant_echo_diagnostic_top_k_count']} | "
            f"{row['recap_above_first_acceptable_count']} | "
            f"{row['assistant_echo_above_first_acceptable_count']} |"
        )

    lines.extend(
        [
            "",
            "## Target-Case Focus",
            "",
            "| Example | Policy | First Acceptable | Weak Before Evidence |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in target_focus:
        comparison = _as_mapping(item)
        policies = _required_mapping(comparison, "policies")
        for policy_name, policy_value in policies.items():
            policy = _as_mapping(policy_value)
            weak = _required_mapping(policy, "weak_evidence")
            lines.append(
                f"| {comparison['example_id']} | {policy_name} | "
                f"{_format_metric(policy['first_acceptable_rank'])} | "
                f"{weak['total_weak_evidence_above_first_acceptable_count']} |"
            )

    lines.extend(
        [
            "",
            "## Regression Gates",
            "",
            "| Policy | Recommended | Failed Gates |",
            "| --- | ---: | --- |",
        ]
    )
    for policy_name, gate_value in gates.items():
        gate_record = _as_mapping(gate_value)
        failed = gate_record["failed_gates"]
        failed_text = ", ".join(str(item) for item in failed) if isinstance(failed, list) else ""
        lines.append(f"| {policy_name} | {gate_record['recommended']} | {failed_text or 'none'} |")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Decision: {recommendation['decision']}",
            f"- Policy: {recommendation.get('policy')}",
            f"- Explanation: {recommendation['explanation']}",
            "",
        ]
    )
    return "\n".join(lines)


def stage12_case_result_from_dict(record: Mapping[str, object]) -> Stage12CaseResult:
    """Parse one emitted Stage 12 diagnostic case record."""

    expected_ids = _expected_ids_from_dict(_required_mapping(record, "expected_ids"))
    retrieved = tuple(
        _retrieved_record_from_dict(_as_mapping(item))
        for item in _required_list(record, "retrieved")
    )
    metrics = _case_metrics_from_dict(_required_mapping(record, "metrics"))
    diagnostics = Stage12DiagnosticMetrics(
        diagnostic_top_k=_required_int(record, "diagnostic_top_k"),
        expected_in_diagnostic_top_k=_required_bool(record, "expected_in_diagnostic_top_k"),
        raw_expected_in_diagnostic_top_k=_required_bool(
            record,
            "raw_expected_in_diagnostic_top_k",
        ),
        summary_expected_in_diagnostic_top_k=_required_bool(
            record,
            "summary_expected_in_diagnostic_top_k",
        ),
        acceptable_in_diagnostic_top_k=_required_bool(
            record,
            "acceptable_in_diagnostic_top_k",
        ),
        first_expected_diagnostic_rank=_optional_int(record, "first_expected_diagnostic_rank"),
        first_raw_diagnostic_rank=_optional_int(record, "first_raw_diagnostic_rank"),
        first_summary_diagnostic_rank=_optional_int(record, "first_summary_diagnostic_rank"),
        first_acceptable_diagnostic_rank=_optional_int(
            record,
            "first_acceptable_diagnostic_rank",
        ),
        diagnostic_kind_mix=_kind_mix_from_dict(_required_mapping(record, "diagnostic_kind_mix")),
        diagnostic_recap_pollution=_recap_pollution_from_dict(
            _required_mapping(record, "diagnostic_recap_pollution")
        ),
        failure_class=tuple(
            _stage12_failure_class(item) for item in _str_tuple(record.get("failure_class", ()))
        ),
    )

    return Stage12CaseResult(
        example_id=_required_str(record, "example_id"),
        scenario_id=_required_str(record, "scenario_id"),
        question_type=_question_type(_required_str(record, "question_type")),
        preferred_layer=_preferred_layer(_required_str(record, "preferred_layer")),
        fact_ids=_str_tuple(record.get("fact_ids", ())),
        top_k=_required_int(record, "top_k"),
        expected_ids=expected_ids,
        retrieved=retrieved,
        metrics=metrics,
        diagnostics=diagnostics,
        notes=_optional_str(record, "notes"),
    )


def _resolve_stage12_weight_sweep_top_k(case_top_k: int, official_top_k: int | None) -> int:
    top_k = case_top_k if official_top_k is None else official_top_k
    if top_k <= 0:
        raise InvalidRetrievalRequestError("Stage 12 official_top_k must be greater than zero")
    return top_k


def _stage12_case_result_from_records(
    *,
    source: Stage12CaseResult,
    records: Sequence[Stage12RetrievedEpisodeRecord],
    top_k: int,
    diagnostic_top_k: int,
) -> Stage12CaseResult:
    raw_expected_ids = set(source.expected_ids.raw)
    summary_expected_ids = set(source.expected_ids.summary)
    acceptable_ids = set(source.expected_ids.acceptable)
    preferred_ids = _preferred_expected_id_set(source.expected_ids, source.preferred_layer)
    timing_mode: TimingMode = (
        "app_realistic" if any(record.is_current_query for record in records) else "clean_memory"
    )

    official_records = tuple(
        record
        for record in records
        if record.official_rank is not None and record.official_rank <= top_k
    )
    diagnostic_official_records = tuple(
        record for record in records if record.official_rank is not None
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
        (record for record in records if record.rank <= top_k and record.is_current_query),
        None,
    )
    metrics = Stage12CaseMetrics(
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
        self_query_similarity=None if self_query_record is None else self_query_record.similarity,
        recap_pollution_at_k=_recap_pollution_metrics(official_records),
    )
    diagnostics = _stage12_diagnostic_metrics(
        top_k=top_k,
        diagnostic_top_k=diagnostic_top_k,
        timing_mode=timing_mode,
        expected_ids=source.expected_ids,
        preferred_ids=preferred_ids,
        records=records,
        diagnostic_official_records=diagnostic_official_records,
    )

    return Stage12CaseResult(
        example_id=source.example_id,
        scenario_id=source.scenario_id,
        question_type=source.question_type,
        preferred_layer=source.preferred_layer,
        fact_ids=source.fact_ids,
        top_k=top_k,
        expected_ids=source.expected_ids,
        retrieved=tuple(records),
        metrics=metrics,
        diagnostics=diagnostics,
        notes=source.notes,
    )


def _weight_sweep_aggregate_to_dict(
    results: Sequence[Stage12CaseResult],
) -> dict[str, object]:
    aggregate = stage12_aggregate_metrics_to_dict(summarize_stage12_results(results))
    recap_echo_count = sum(
        1 for result in results if _has_recap_echo_above_first_acceptable_result(result)
    )
    aggregate["recap_echo_above_first_acceptable_count"] = recap_echo_count
    aggregate["recap_echo_above_first_acceptable_rate"] = (
        0.0 if not results else recap_echo_count / len(results)
    )
    aggregate["negative_control_no_hit_retained"] = _negative_control_no_hit_retained(results)
    return aggregate


def _stage12_weight_sweep_rank_movement(
    results_by_profile: Mapping[str, Sequence[Stage12CaseResult]],
) -> list[dict[str, object]]:
    baseline_results = results_by_profile["baseline"]
    movement: list[dict[str, object]] = []
    for baseline_case in baseline_results:
        profiles: dict[str, object] = {}
        best_rank = baseline_case.diagnostics.first_acceptable_diagnostic_rank
        best_profile = "baseline"
        for profile_name, profile_results in results_by_profile.items():
            profile_case = _find_profile_case(profile_results, baseline_case.example_id)
            first_acceptable = profile_case.diagnostics.first_acceptable_diagnostic_rank
            profiles[profile_name] = {
                "hit_at_k": profile_case.metrics.hit_at_k,
                "acceptable_hit_at_k": profile_case.metrics.acceptable_hit_at_k,
                "raw_hit_at_k": profile_case.metrics.raw_hit_at_k,
                "summary_hit_at_k": profile_case.metrics.summary_hit_at_k,
                "reciprocal_rank": profile_case.metrics.reciprocal_rank,
                "first_acceptable_rank": first_acceptable,
                "first_raw_rank": profile_case.diagnostics.first_raw_diagnostic_rank,
                "first_summary_rank": profile_case.diagnostics.first_summary_diagnostic_rank,
                "acceptable_rank_delta": _rank_delta(
                    first_acceptable,
                    baseline_case.diagnostics.first_acceptable_diagnostic_rank,
                ),
                "raw_rank_delta": _rank_delta(
                    profile_case.diagnostics.first_raw_diagnostic_rank,
                    baseline_case.diagnostics.first_raw_diagnostic_rank,
                ),
                "summary_rank_delta": _rank_delta(
                    profile_case.diagnostics.first_summary_diagnostic_rank,
                    baseline_case.diagnostics.first_summary_diagnostic_rank,
                ),
                "recap_echo_above_first_acceptable": (
                    _has_recap_echo_above_first_acceptable_result(profile_case)
                ),
            }
            if _rank_is_better(first_acceptable, best_rank):
                best_rank = first_acceptable
                best_profile = profile_name

        movement.append(
            {
                "example_id": baseline_case.example_id,
                "question_type": baseline_case.question_type,
                "preferred_layer": baseline_case.preferred_layer,
                "top_k": baseline_case.top_k,
                "baseline_first_acceptable_rank": (
                    baseline_case.diagnostics.first_acceptable_diagnostic_rank
                ),
                "best_first_acceptable_rank": best_rank,
                "best_first_acceptable_profile": best_profile,
                "profiles": profiles,
            }
        )
    return movement


def _stage12_weight_sweep_regression_gates(
    *,
    input_metadata: Mapping[str, object],
    baseline_results: Sequence[Stage12CaseResult],
    baseline_aggregate: Stage12AggregateMetrics,
    profile_results: Sequence[Stage12CaseResult],
    profile_aggregate: Stage12AggregateMetrics,
    is_baseline: bool,
) -> dict[str, object]:
    direct_fact_regressions = sum(
        1
        for baseline, profile in zip(baseline_results, profile_results, strict=True)
        if baseline.question_type == "direct_fact"
        and baseline.metrics.hit_at_k
        and not profile.metrics.hit_at_k
    )
    recap_echo_count = sum(
        1 for result in profile_results if _has_recap_echo_above_first_acceptable_result(result)
    )
    baseline_recap_echo_count = sum(
        1 for result in baseline_results if _has_recap_echo_above_first_acceptable_result(result)
    )
    embedder_mode = input_metadata.get("embedder_mode")
    profile_improves = _profile_improves(profile_aggregate, baseline_aggregate)
    gates = [
        _gate_record(
            "direct_fact_regression",
            failed=direct_fact_regressions > 0,
            value=direct_fact_regressions,
            detail="previously passing direct facts must keep preferred hits",
        ),
        _gate_record(
            "raw_hit_not_dropped",
            failed=profile_aggregate.raw_hit_rate_at_k
            < baseline_aggregate.raw_hit_rate_at_k - _METRIC_EPSILON,
            value=profile_aggregate.raw_hit_rate_at_k,
            baseline=baseline_aggregate.raw_hit_rate_at_k,
        ),
        _gate_record(
            "acceptable_hit_not_dropped",
            failed=profile_aggregate.acceptable_hit_rate_at_k
            < baseline_aggregate.acceptable_hit_rate_at_k - _METRIC_EPSILON,
            value=profile_aggregate.acceptable_hit_rate_at_k,
            baseline=baseline_aggregate.acceptable_hit_rate_at_k,
        ),
        _gate_record(
            "mrr_not_materially_dropped",
            failed=profile_aggregate.mean_reciprocal_rank
            < baseline_aggregate.mean_reciprocal_rank - _MATERIAL_MRR_DROP,
            value=profile_aggregate.mean_reciprocal_rank,
            baseline=baseline_aggregate.mean_reciprocal_rank,
            threshold=_MATERIAL_MRR_DROP,
        ),
        _gate_record(
            "recap_echo_not_increased",
            failed=recap_echo_count > baseline_recap_echo_count,
            value=recap_echo_count,
            baseline=baseline_recap_echo_count,
        ),
        _gate_record(
            "negative_control_no_hit_retained",
            failed=not _negative_control_no_hit_retained(profile_results),
            value=_negative_control_no_hit_retained(profile_results),
        ),
        _gate_record(
            "summary_gain_not_at_raw_expense",
            failed=profile_aggregate.summary_hit_rate_at_k
            > baseline_aggregate.summary_hit_rate_at_k + _METRIC_EPSILON
            and profile_aggregate.raw_hit_rate_at_k
            < baseline_aggregate.raw_hit_rate_at_k - _METRIC_EPSILON,
            value=profile_aggregate.summary_hit_rate_at_k,
            baseline=baseline_aggregate.summary_hit_rate_at_k,
        ),
        _gate_record(
            "ollama_evidence_required_for_recommendation",
            failed=not is_baseline and embedder_mode == "fake" and profile_improves,
            value=embedder_mode,
            detail="fake-only improvements are diagnostic, not recommended",
        ),
    ]
    failed_gates = [str(gate["name"]) for gate in gates if gate.get("status") == "fail"]
    return {
        "recommended": not failed_gates,
        "failed_gates": failed_gates,
        "gates": gates,
    }


def _stage12_weight_sweep_recommendation(
    *,
    baseline_results: Sequence[Stage12CaseResult],
    baseline_aggregate: Stage12AggregateMetrics,
    aggregate_by_profile: Mapping[str, Stage12AggregateMetrics],
    regression_gates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for profile_name, aggregate in aggregate_by_profile.items():
        if profile_name == "baseline":
            continue
        gates = regression_gates[profile_name]
        if not gates.get("recommended"):
            continue
        if not _profile_improves(aggregate, baseline_aggregate):
            continue
        candidates.append(
            (
                (
                    aggregate.acceptable_hit_rate_at_k,
                    aggregate.hit_rate_at_k,
                    aggregate.mean_reciprocal_rank,
                ),
                profile_name,
            )
        )

    if candidates:
        _, profile_name = max(candidates)
        return {
            "decision": "candidate_profile_for_future_stage",
            "profile": profile_name,
            "explanation": (
                "The profile improved aggregate retrieval metrics without tripping "
                "the Stage 12b-2 regression gates."
            ),
        }

    has_visible_rerank_failures = any(
        result.diagnostics.first_acceptable_diagnostic_rank is not None
        and result.diagnostics.first_acceptable_diagnostic_rank > result.top_k
        for result in baseline_results
    )
    if has_visible_rerank_failures:
        return {
            "decision": "weight_sweep_insufficient",
            "profile": None,
            "explanation": (
                "Expected evidence is visible at diagnostic depth, but no safe profile "
                "improved the official top-k metrics."
            ),
        }

    return {
        "decision": "no_change_recommended",
        "profile": "baseline",
        "explanation": "The baseline profile remains the safest choice in this diagnostic run.",
    }


def _find_profile_case(
    results: Sequence[Stage12CaseResult],
    example_id: str,
) -> Stage12CaseResult:
    return next(result for result in results if result.example_id == example_id)


def _rank_delta(rank: int | None, baseline_rank: int | None) -> int | None:
    if rank is None or baseline_rank is None:
        return None
    return rank - baseline_rank


def _rank_is_better(rank: int | None, best_rank: int | None) -> bool:
    if rank is None:
        return False
    if best_rank is None:
        return True
    return rank < best_rank


def _profile_improves(
    profile_aggregate: Stage12AggregateMetrics,
    baseline_aggregate: Stage12AggregateMetrics,
) -> bool:
    return (
        profile_aggregate.acceptable_hit_rate_at_k
        > baseline_aggregate.acceptable_hit_rate_at_k + _METRIC_EPSILON
        or profile_aggregate.hit_rate_at_k > baseline_aggregate.hit_rate_at_k + _METRIC_EPSILON
        or profile_aggregate.mean_reciprocal_rank
        > baseline_aggregate.mean_reciprocal_rank + _MATERIAL_MRR_DROP
    )


def _has_recap_echo_above_first_acceptable_result(result: Stage12CaseResult) -> bool:
    first_acceptable = result.diagnostics.first_acceptable_diagnostic_rank
    if first_acceptable is None:
        return False
    return any(
        record.official_rank is not None
        and record.official_rank < first_acceptable
        and {"recap_question", "assistant_answer_echo"}.intersection(record.label_roles)
        for record in result.retrieved
    )


def _negative_control_no_hit_retained(results: Sequence[Stage12CaseResult]) -> bool:
    negative_controls = [
        result
        for result in results
        if not (
            result.expected_ids.raw or result.expected_ids.summary or result.expected_ids.acceptable
        )
    ]
    return all(
        not (
            result.metrics.hit_at_k
            or result.metrics.raw_hit_at_k
            or result.metrics.summary_hit_at_k
            or result.metrics.acceptable_hit_at_k
        )
        for result in negative_controls
    )


def _stage13_role_policy_ordered_candidates(
    candidates: Sequence[Stage12RetrievedEpisodeRecord],
    policy: Stage13RolePolicy,
) -> tuple[Stage12RetrievedEpisodeRecord, ...]:
    if not policy.demoted_roles:
        return tuple(candidates)
    if policy.demotion_strategy != "move_behind_non_weak":
        raise InvalidRetrievalRequestError(
            f"Stage 13 unknown role-policy demotion strategy: {policy.demotion_strategy}"
        )
    return tuple(
        sorted(
            candidates,
            key=lambda record: (
                1 if _stage13_record_has_any_role(record, policy.demoted_roles) else 0,
                record.rank,
            ),
        )
    )


def _stage13_record_has_any_role(
    record: Stage12RetrievedEpisodeRecord,
    roles: Sequence[EpisodeLabelRole],
) -> bool:
    return bool(set(record.label_roles).intersection(roles))


def _stage13_record_is_weak_evidence(record: Stage12RetrievedEpisodeRecord) -> bool:
    return _stage13_record_has_any_role(
        record,
        ("recap_question", "assistant_answer_echo"),
    )


def _stage13_count_role(
    records: Sequence[Stage12RetrievedEpisodeRecord],
    role: EpisodeLabelRole,
) -> int:
    return sum(1 for record in records if role in record.label_roles)


def _stage13_first_blocking_weak_evidence_role(
    weak_before_acceptable: Sequence[Stage12RetrievedEpisodeRecord],
) -> Stage13FirstBlockingWeakEvidenceRole:
    first = next(
        iter(sorted(weak_before_acceptable, key=lambda record: record.official_rank or 0)),
        None,
    )
    if first is None:
        return "none"
    has_recap = "recap_question" in first.label_roles
    has_echo = "assistant_answer_echo" in first.label_roles
    if has_recap and has_echo:
        return "both"
    if has_recap:
        return "recap"
    if has_echo:
        return "assistant_echo"
    return "none"


def _stage13_removed_legitimate_raw_source_count(
    candidates: Sequence[Stage12RetrievedEpisodeRecord],
    policy: Stage13RolePolicy,
) -> int:
    if not policy.excluded_roles:
        return 0
    return sum(
        1
        for record in candidates
        if _stage13_record_has_any_role(record, policy.excluded_roles)
        and (
            record.is_raw_expected or ("raw_source" in record.label_roles and record.is_acceptable)
        )
    )


def _stage13_role_policy_aggregate_to_dict(
    *,
    baseline_results: Sequence[Stage13RolePolicyCaseResult],
    policy_results: Sequence[Stage13RolePolicyCaseResult],
) -> dict[str, object]:
    baseline_cases = tuple(result.case_result for result in baseline_results)
    policy_cases = tuple(result.case_result for result in policy_results)
    aggregate = stage12_aggregate_metrics_to_dict(summarize_stage12_results(policy_cases))
    weak_metrics = tuple(result.weak_evidence for result in policy_results)
    first_blocking_counts = {
        role: sum(
            1 for metrics in weak_metrics if metrics.first_blocking_weak_evidence_role == role
        )
        for role in ("recap", "assistant_echo", "both", "none")
    }
    aggregate.update(
        {
            "recap_above_first_acceptable_count": sum(
                metrics.recap_above_first_acceptable_count for metrics in weak_metrics
            ),
            "assistant_echo_above_first_acceptable_count": sum(
                metrics.assistant_echo_above_first_acceptable_count for metrics in weak_metrics
            ),
            "total_weak_evidence_above_first_acceptable_count": sum(
                metrics.total_weak_evidence_above_first_acceptable_count for metrics in weak_metrics
            ),
            "recap_official_top_k_count": sum(
                metrics.recap_official_top_k_count for metrics in weak_metrics
            ),
            "assistant_echo_official_top_k_count": sum(
                metrics.assistant_echo_official_top_k_count for metrics in weak_metrics
            ),
            "recap_diagnostic_top_k_count": sum(
                metrics.recap_diagnostic_top_k_count for metrics in weak_metrics
            ),
            "assistant_echo_diagnostic_top_k_count": sum(
                metrics.assistant_echo_diagnostic_top_k_count for metrics in weak_metrics
            ),
            "first_blocking_weak_evidence_role_counts": first_blocking_counts,
            "direct_fact_regression_count": _stage13_direct_fact_regression_count(
                baseline_cases=baseline_cases,
                policy_cases=policy_cases,
            ),
            "negative_control_no_hit_retained": _negative_control_no_hit_retained(policy_cases),
            "legitimate_raw_source_removed_count": sum(
                result.removed_legitimate_raw_source_count for result in policy_results
            ),
        }
    )
    return aggregate


def _stage13_direct_fact_regression_count(
    *,
    baseline_cases: Sequence[Stage12CaseResult],
    policy_cases: Sequence[Stage12CaseResult],
) -> int:
    return sum(
        1
        for baseline, policy in zip(baseline_cases, policy_cases, strict=True)
        if baseline.question_type == "direct_fact"
        and baseline.metrics.hit_at_k
        and not policy.metrics.hit_at_k
    )


def _stage13_role_policy_regression_gates(
    *,
    policy: Stage13RolePolicy,
    baseline_results: Sequence[Stage13RolePolicyCaseResult],
    policy_results: Sequence[Stage13RolePolicyCaseResult],
) -> dict[str, object]:
    baseline_cases = tuple(result.case_result for result in baseline_results)
    policy_cases = tuple(result.case_result for result in policy_results)
    baseline_aggregate = summarize_stage12_results(baseline_cases)
    policy_aggregate = summarize_stage12_results(policy_cases)
    direct_fact_regressions = _stage13_direct_fact_regression_count(
        baseline_cases=baseline_cases,
        policy_cases=policy_cases,
    )
    negative_control_no_hit_retained = _negative_control_no_hit_retained(policy_cases)
    removed_raw_sources = sum(
        result.removed_legitimate_raw_source_count for result in policy_results
    )
    gates = [
        _gate_record(
            "acceptable_hit_not_dropped",
            failed=policy_aggregate.acceptable_hit_rate_at_k
            < baseline_aggregate.acceptable_hit_rate_at_k - _METRIC_EPSILON,
            value=policy_aggregate.acceptable_hit_rate_at_k,
            baseline=baseline_aggregate.acceptable_hit_rate_at_k,
        ),
        _gate_record(
            "raw_hit_not_dropped",
            failed=policy_aggregate.raw_hit_rate_at_k
            < baseline_aggregate.raw_hit_rate_at_k - _METRIC_EPSILON,
            value=policy_aggregate.raw_hit_rate_at_k,
            baseline=baseline_aggregate.raw_hit_rate_at_k,
        ),
        _gate_record(
            "mrr_not_materially_dropped",
            failed=policy_aggregate.mean_reciprocal_rank
            < baseline_aggregate.mean_reciprocal_rank - _MATERIAL_MRR_DROP,
            value=policy_aggregate.mean_reciprocal_rank,
            baseline=baseline_aggregate.mean_reciprocal_rank,
            threshold=_MATERIAL_MRR_DROP,
        ),
        _gate_record(
            "direct_fact_regression",
            failed=direct_fact_regressions > 0,
            value=direct_fact_regressions,
            detail="previously passing direct facts must keep preferred hits",
        ),
        _gate_record(
            "negative_control_no_hit_retained",
            failed=not negative_control_no_hit_retained,
            value=negative_control_no_hit_retained,
        ),
        _gate_record(
            "legitimate_raw_source_not_removed",
            failed=removed_raw_sources > 0,
            value=removed_raw_sources,
        ),
        _gate_record(
            "recap_metadata_available",
            failed=policy.depends_on_recap_metadata,
            value=not policy.depends_on_recap_metadata,
            detail="recap-question labels are eval-only today",
        ),
    ]
    failed_gates = [str(gate["name"]) for gate in gates if gate.get("status") == "fail"]
    return {
        "recommended": not failed_gates,
        "failed_gates": failed_gates,
        "gates": gates,
    }


def _stage13_role_policy_case_comparison(
    results_by_policy: Mapping[str, Sequence[Stage13RolePolicyCaseResult]],
) -> list[dict[str, object]]:
    baseline_results = results_by_policy["baseline"]
    comparison: list[dict[str, object]] = []
    for baseline in baseline_results:
        policy_records: dict[str, object] = {}
        for policy_name, policy_results in results_by_policy.items():
            policy_result = _find_stage13_policy_case(
                policy_results,
                baseline.case_result.example_id,
            )
            policy_case = policy_result.case_result
            policy_records[policy_name] = {
                "hit_at_k": policy_case.metrics.hit_at_k,
                "acceptable_hit_at_k": policy_case.metrics.acceptable_hit_at_k,
                "raw_hit_at_k": policy_case.metrics.raw_hit_at_k,
                "summary_hit_at_k": policy_case.metrics.summary_hit_at_k,
                "reciprocal_rank": policy_case.metrics.reciprocal_rank,
                "first_acceptable_rank": (policy_case.diagnostics.first_acceptable_diagnostic_rank),
                "first_raw_rank": policy_case.diagnostics.first_raw_diagnostic_rank,
                "first_summary_rank": policy_case.diagnostics.first_summary_diagnostic_rank,
                "weak_evidence": _stage13_weak_evidence_metrics_to_dict(
                    policy_result.weak_evidence
                ),
                "removed_legitimate_raw_source_count": (
                    policy_result.removed_legitimate_raw_source_count
                ),
            }
        comparison.append(
            {
                "example_id": baseline.case_result.example_id,
                "question_type": baseline.case_result.question_type,
                "preferred_layer": baseline.case_result.preferred_layer,
                "top_k": baseline.case_result.top_k,
                "policies": policy_records,
            }
        )
    return comparison


def _find_stage13_policy_case(
    results: Sequence[Stage13RolePolicyCaseResult],
    example_id: str,
) -> Stage13RolePolicyCaseResult:
    return next(result for result in results if result.case_result.example_id == example_id)


def _stage13_role_policy_weak_evidence_table(
    results_by_policy: Mapping[str, Sequence[Stage13RolePolicyCaseResult]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy_name, policy_results in results_by_policy.items():
        for result in policy_results:
            rows.append(
                {
                    "example_id": result.case_result.example_id,
                    "policy": policy_name,
                    **_stage13_weak_evidence_metrics_to_dict(result.weak_evidence),
                }
            )
    return rows


def _stage13_role_policy_recommendation(
    *,
    policies: Sequence[Stage13RolePolicy],
    aggregate_by_policy: Mapping[str, Mapping[str, object]],
    regression_gates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    baseline = aggregate_by_policy["baseline"]
    unsafe_improvements: list[str] = []
    recap_improvements: list[str] = []
    assistant_candidates: list[str] = []
    for policy in policies:
        if policy.name == "baseline":
            continue
        aggregate = aggregate_by_policy[policy.name]
        if not _stage13_role_policy_improves(policy=aggregate, baseline=baseline):
            continue
        gates = regression_gates[policy.name]
        if policy.depends_on_recap_metadata:
            recap_improvements.append(policy.name)
            continue
        if gates.get("recommended"):
            assistant_candidates.append(policy.name)
        else:
            unsafe_improvements.append(policy.name)

    if assistant_candidates:
        return {
            "decision": "assistant_echo_policy_candidate",
            "policy": assistant_candidates[0],
            "explanation": (
                "An assistant-echo policy improved or preserved retrieval metrics while "
                "reducing weak-evidence pollution without tripping regression gates."
            ),
        }
    if recap_improvements:
        return {
            "decision": "recap_policy_requires_metadata",
            "policy": recap_improvements[0],
            "explanation": (
                "A recap-question policy improved the replay, but recap labels are "
                "eval-only today and require durable metadata before production use."
            ),
        }
    if unsafe_improvements:
        return {
            "decision": "role_policy_insufficient",
            "policy": unsafe_improvements[0],
            "explanation": (
                "At least one role policy improved replay metrics but failed a regression "
                "or safety gate."
            ),
        }
    return {
        "decision": "no_role_policy_change_recommended",
        "policy": "baseline",
        "explanation": (
            "No eval-only role policy produced a safe improvement over the diagnostic baseline."
        ),
    }


def _stage13_role_policy_improves(
    *,
    policy: Mapping[str, object],
    baseline: Mapping[str, object],
) -> bool:
    return (
        _required_float(policy, "acceptable_hit_rate_at_k")
        > _required_float(baseline, "acceptable_hit_rate_at_k") + _METRIC_EPSILON
        or _required_float(policy, "hit_rate_at_k")
        > _required_float(baseline, "hit_rate_at_k") + _METRIC_EPSILON
        or _required_float(policy, "mean_reciprocal_rank")
        > _required_float(baseline, "mean_reciprocal_rank") + _MATERIAL_MRR_DROP
        or _required_int(policy, "total_weak_evidence_above_first_acceptable_count")
        < _required_int(baseline, "total_weak_evidence_above_first_acceptable_count")
    )


def _stage13_weak_evidence_metrics_to_dict(
    metrics: Stage13WeakEvidenceMetrics,
) -> dict[str, object]:
    return {
        "recap_above_first_acceptable_count": metrics.recap_above_first_acceptable_count,
        "assistant_echo_above_first_acceptable_count": (
            metrics.assistant_echo_above_first_acceptable_count
        ),
        "total_weak_evidence_above_first_acceptable_count": (
            metrics.total_weak_evidence_above_first_acceptable_count
        ),
        "recap_official_top_k_count": metrics.recap_official_top_k_count,
        "assistant_echo_official_top_k_count": metrics.assistant_echo_official_top_k_count,
        "recap_diagnostic_top_k_count": metrics.recap_diagnostic_top_k_count,
        "assistant_echo_diagnostic_top_k_count": metrics.assistant_echo_diagnostic_top_k_count,
        "first_blocking_weak_evidence_role": metrics.first_blocking_weak_evidence_role,
    }


def _gate_record(
    name: str,
    *,
    failed: bool,
    value: object,
    baseline: object | None = None,
    threshold: object | None = None,
    detail: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "status": "fail" if failed else "pass",
        "value": value,
        "baseline": baseline,
        "threshold": threshold,
        "detail": detail,
    }


def _format_metric(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_role_list(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _expected_ids_from_dict(record: Mapping[str, object]) -> Stage12ExpectedIds:
    return Stage12ExpectedIds(
        raw=_uuid_tuple(record.get("raw", ())),
        summary=_uuid_tuple(record.get("summary", ())),
        acceptable=_uuid_tuple(record.get("acceptable", ())),
        current_query=_uuid_tuple(record.get("current_query", ())),
    )


def _retrieved_record_from_dict(record: Mapping[str, object]) -> Stage12RetrievedEpisodeRecord:
    score_components = _required_mapping(record, "score_components")
    expected_flags = _required_mapping(record, "expected_flags")
    return Stage12RetrievedEpisodeRecord(
        episode_id=_required_uuid(record, "episode_id"),
        kind=_episode_kind(_required_str(record, "kind")),
        rank=_required_int(record, "rank"),
        official_rank=_optional_int(record, "official_rank"),
        conversation_id=_required_uuid(record, "conversation_id"),
        scope_id=_required_uuid(record, "scope_id"),
        message_id=_optional_uuid(record, "message_id"),
        message_position=_optional_int(record, "message_position"),
        range_start=_optional_int(record, "range_start"),
        range_end=_optional_int(record, "range_end"),
        score=_required_float(record, "score"),
        similarity=_required_float(record, "similarity"),
        recency_score=_required_score_component(score_components, "recency"),
        access_score=_required_score_component(score_components, "access"),
        importance_score=_required_score_component(score_components, "importance"),
        frequency_score=_required_score_component(score_components, "frequency"),
        label_roles=tuple(
            _episode_label_role(role) for role in _str_tuple(record.get("label_roles", ()))
        ),
        fact_ids=_str_tuple(record.get("fact_ids", ())),
        is_raw_expected=_required_bool(expected_flags, "raw"),
        is_summary_expected=_required_bool(expected_flags, "summary"),
        is_preferred_expected=_required_bool(expected_flags, "preferred"),
        is_acceptable=_required_bool(expected_flags, "acceptable"),
        is_current_query=_required_bool(expected_flags, "current_query"),
    )


def _case_metrics_from_dict(record: Mapping[str, object]) -> Stage12CaseMetrics:
    return Stage12CaseMetrics(
        hit_at_k=_required_bool(record, "hit_at_k"),
        reciprocal_rank=_required_float(record, "reciprocal_rank"),
        precision_at_k=_required_float(record, "precision_at_k"),
        recall_at_k=_required_float(record, "recall_at_k"),
        raw_hit_at_k=_required_bool(record, "raw_hit_at_k"),
        summary_hit_at_k=_required_bool(record, "summary_hit_at_k"),
        acceptable_hit_at_k=_required_bool(record, "acceptable_hit_at_k"),
        kind_mix_at_k=_kind_mix_from_dict(_required_mapping(record, "kind_mix_at_k")),
        self_query_hit=_required_bool(record, "self_query_hit"),
        self_query_rank=_optional_int(record, "self_query_rank"),
        self_query_similarity=_optional_float(record, "self_query_similarity"),
        recap_pollution_at_k=_recap_pollution_from_dict(
            _required_mapping(record, "recap_pollution_at_k")
        ),
    )


def _kind_mix_from_dict(record: Mapping[str, object]) -> Stage12KindMixMetrics:
    return Stage12KindMixMetrics(
        message_count=_required_int(record, "message_count"),
        summary_count=_required_int(record, "summary_count"),
        total_count=_required_int(record, "total_count"),
        message_ratio=_required_float(record, "message_ratio"),
        summary_ratio=_required_float(record, "summary_ratio"),
    )


def _recap_pollution_from_dict(record: Mapping[str, object]) -> Stage12RecapPollutionMetrics:
    return Stage12RecapPollutionMetrics(
        count=_required_int(record, "count"),
        ratio=_required_float(record, "ratio"),
    )


def _stage12_failure_class(value: str) -> Stage12FailureClass:
    if value not in _VALID_STAGE12_FAILURE_CLASSES:
        raise InvalidRetrievalRequestError(f"Stage 12 unknown failure class: {value}")
    return cast(Stage12FailureClass, value)


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
        "diagnostic_top_k": metadata.diagnostic_top_k,
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
        "diagnostic_top_k": result.diagnostics.diagnostic_top_k,
        "expected_in_diagnostic_top_k": result.diagnostics.expected_in_diagnostic_top_k,
        "raw_expected_in_diagnostic_top_k": (result.diagnostics.raw_expected_in_diagnostic_top_k),
        "summary_expected_in_diagnostic_top_k": (
            result.diagnostics.summary_expected_in_diagnostic_top_k
        ),
        "acceptable_in_diagnostic_top_k": (result.diagnostics.acceptable_in_diagnostic_top_k),
        "first_expected_diagnostic_rank": result.diagnostics.first_expected_diagnostic_rank,
        "first_raw_diagnostic_rank": result.diagnostics.first_raw_diagnostic_rank,
        "first_summary_diagnostic_rank": result.diagnostics.first_summary_diagnostic_rank,
        "first_acceptable_diagnostic_rank": (result.diagnostics.first_acceptable_diagnostic_rank),
        "diagnostic_kind_mix": _kind_mix_to_dict(result.diagnostics.diagnostic_kind_mix),
        "diagnostic_recap_pollution": _recap_pollution_to_dict(
            result.diagnostics.diagnostic_recap_pollution
        ),
        "failure_class": list(result.diagnostics.failure_class),
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


def _resolve_diagnostic_top_k(top_k: int, diagnostic_top_k: int | None) -> int:
    if diagnostic_top_k is None:
        return top_k
    if diagnostic_top_k <= 0:
        raise InvalidRetrievalRequestError("Stage 12 diagnostic_top_k must be greater than zero")
    return max(top_k, diagnostic_top_k)


def _stage12_diagnostic_metrics(
    *,
    top_k: int,
    diagnostic_top_k: int,
    timing_mode: TimingMode,
    expected_ids: Stage12ExpectedIds,
    preferred_ids: set[UUID],
    records: Sequence[Stage12RetrievedEpisodeRecord],
    diagnostic_official_records: Sequence[Stage12RetrievedEpisodeRecord],
) -> Stage12DiagnosticMetrics:
    raw_expected_ids = set(expected_ids.raw)
    summary_expected_ids = set(expected_ids.summary)
    acceptable_ids = set(expected_ids.acceptable)

    first_expected_rank = _first_diagnostic_rank(diagnostic_official_records, preferred_ids)
    first_raw_rank = _first_diagnostic_rank(diagnostic_official_records, raw_expected_ids)
    first_summary_rank = _first_diagnostic_rank(diagnostic_official_records, summary_expected_ids)
    first_acceptable_rank = _first_diagnostic_rank(diagnostic_official_records, acceptable_ids)
    failure_class = _stage12_failure_classes(
        top_k=top_k,
        timing_mode=timing_mode,
        expected_ids=expected_ids,
        records=records,
        diagnostic_official_records=diagnostic_official_records,
        first_acceptable_diagnostic_rank=first_acceptable_rank,
        first_raw_diagnostic_rank=first_raw_rank,
        first_summary_diagnostic_rank=first_summary_rank,
    )

    return Stage12DiagnosticMetrics(
        diagnostic_top_k=diagnostic_top_k,
        expected_in_diagnostic_top_k=first_expected_rank is not None,
        raw_expected_in_diagnostic_top_k=first_raw_rank is not None,
        summary_expected_in_diagnostic_top_k=first_summary_rank is not None,
        acceptable_in_diagnostic_top_k=first_acceptable_rank is not None,
        first_expected_diagnostic_rank=first_expected_rank,
        first_raw_diagnostic_rank=first_raw_rank,
        first_summary_diagnostic_rank=first_summary_rank,
        first_acceptable_diagnostic_rank=first_acceptable_rank,
        diagnostic_kind_mix=_kind_mix_metrics(diagnostic_official_records),
        diagnostic_recap_pollution=_recap_pollution_metrics(diagnostic_official_records),
        failure_class=failure_class,
    )


def _first_diagnostic_rank(
    records: Sequence[Stage12RetrievedEpisodeRecord],
    episode_ids: set[UUID],
) -> int | None:
    if not episode_ids:
        return None
    return next(
        (
            record.official_rank
            for record in records
            if record.episode_id in episode_ids and record.official_rank is not None
        ),
        None,
    )


def _stage12_failure_classes(
    *,
    top_k: int,
    timing_mode: TimingMode,
    expected_ids: Stage12ExpectedIds,
    records: Sequence[Stage12RetrievedEpisodeRecord],
    diagnostic_official_records: Sequence[Stage12RetrievedEpisodeRecord],
    first_acceptable_diagnostic_rank: int | None,
    first_raw_diagnostic_rank: int | None,
    first_summary_diagnostic_rank: int | None,
) -> tuple[Stage12FailureClass, ...]:
    classes: list[Stage12FailureClass] = []
    has_expected_ids = bool(expected_ids.raw or expected_ids.summary or expected_ids.acceptable)

    if timing_mode == "app_realistic" and any(record.is_current_query for record in records):
        classes.append("self_query_artifact")

    if not has_expected_ids:
        classes.append("negative_control")
        return tuple(classes) if classes else ("unclassified",)

    if _is_visible_within_official_window(first_acceptable_diagnostic_rank, top_k=top_k):
        classes.append("hit_at_official_k")
    elif first_acceptable_diagnostic_rank is None:
        classes.append("absent_from_diagnostic_top_k")
    else:
        classes.append("rerank_below_official_k")

    if _has_recap_or_echo_before_acceptable(
        diagnostic_official_records=diagnostic_official_records,
        first_acceptable_diagnostic_rank=first_acceptable_diagnostic_rank,
    ):
        classes.append("recap_or_echo_pollution")

    if _is_visible_below_official_window(
        first_raw_diagnostic_rank,
        top_k=top_k,
    ):
        classes.append("raw_visible_but_low")

    if _is_visible_below_official_window(
        first_summary_diagnostic_rank,
        top_k=top_k,
    ):
        classes.append("summary_visible_but_low")

    return tuple(classes) if classes else ("unclassified",)


def _has_recap_or_echo_before_acceptable(
    *,
    diagnostic_official_records: Sequence[Stage12RetrievedEpisodeRecord],
    first_acceptable_diagnostic_rank: int | None,
) -> bool:
    for record in diagnostic_official_records:
        if not {"recap_question", "assistant_answer_echo"}.intersection(record.label_roles):
            continue
        if record.official_rank is None:
            continue
        if first_acceptable_diagnostic_rank is None:
            return True
        if record.official_rank < first_acceptable_diagnostic_rank:
            return True
    return False


def _is_visible_within_official_window(rank: int | None, *, top_k: int) -> bool:
    return rank is not None and rank <= top_k


def _is_visible_below_official_window(rank: int | None, *, top_k: int) -> bool:
    return rank is not None and rank > top_k


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


def _required_bool(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a boolean: {key}")
    return value


def _required_float(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a number: {key}")
    return float(value)


def _optional_float(record: Mapping[str, object], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a number: {key}")
    return float(value)


def _required_uuid(record: Mapping[str, object], key: str) -> UUID:
    value = _required_str(record, key)
    try:
        return UUID(value)
    except ValueError as exc:
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a UUID: {key}") from exc


def _optional_uuid(record: Mapping[str, object], key: str) -> UUID | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a UUID: {key}")
    try:
        return UUID(value)
    except ValueError as exc:
        raise InvalidRetrievalRequestError(f"Stage 12 field must be a UUID: {key}") from exc


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


def _required_mapping(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise InvalidRetrievalRequestError(f"Stage 12 field must be an object: {key}")
    return cast(Mapping[str, object], value)


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


def _uuid_tuple(value: object) -> tuple[UUID, ...]:
    if not isinstance(value, (list, tuple)):
        raise InvalidRetrievalRequestError("Stage 12 field must be a UUID list")
    result: list[UUID] = []
    for item in value:
        if not isinstance(item, str):
            raise InvalidRetrievalRequestError("Stage 12 field must be a UUID list")
        try:
            result.append(UUID(item))
        except ValueError as exc:
            raise InvalidRetrievalRequestError("Stage 12 field must be a UUID list") from exc
    return tuple(result)


def _required_score_component(record: Mapping[str, object], key: str) -> float:
    try:
        return _required_float(record, key)
    except InvalidRetrievalRequestError as exc:
        raise InvalidRetrievalRequestError(
            f"Stage 12 diagnostic record is missing score component: {key}"
        ) from exc


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
