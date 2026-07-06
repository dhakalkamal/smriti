from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
from uuid import UUID

from smriti.assistant.models import MemoryAdmissionDecision
from smriti.memory import ScoredEpisode

MemoryPolicyName = Literal["legacy", "typed_v1"]

RAW_SOURCE_LANE = "raw_source"
SUMMARY_SOURCE_LANE = "summary_source"
ASSISTANT_DERIVED_LANE = "assistant_derived"
UNSUPPORTED_NOISY_LANE = "unsupported_noisy"
LEGACY_MIXED_LANE = "legacy_mixed"

SUMMARY_RELEVANCE_RANK_SKIP = "summary_relevance_rank"
SUMMARY_RELEVANCE_MARGIN_SKIP = "summary_relevance_margin"
SUMMARY_WINDOW_OVERLAP_SKIP = "summary_window_overlap"
SUMMARY_NEAR_DUPLICATE_SKIP = "summary_near_duplicate"
BACKFILL_RELEVANCE_MARGIN_SKIP = "backfill_relevance_margin"
PROMPT_OVERFLOW_BACKFILL_REASON = "prompt_overflow_backfill"

SUMMARY_NEAR_DUPLICATE_JACCARD = 0.9
# Below this pool size the observed score spread is degenerate (with two
# candidates the lower one always falls outside any fractional margin), so
# the relative margin gate stays open.
MARGIN_MIN_SOURCE_CANDIDATES = 3


@dataclass(frozen=True)
class TypedMemoryAdmissionConfig:
    total_limit: int = 6
    raw_source_limit: int = 4
    summary_source_limit: int = 2
    assistant_derived_limit: int = 0
    # Relative relevance gates: local embedding scores can be low or negative,
    # so eligibility is judged against the candidate pool itself, never
    # against a global absolute similarity threshold.
    #
    # Phase 1.6: the default spread fraction of 1.0 places the score cut at the
    # pool minimum, which no candidate is strictly below, so the margin gates
    # (summary, backfill, overflow) are DISABLED by default, not merely
    # relaxed. Real-embedding evals showed the pool-top anchor is usually a
    # recap-pollution raw message, so the gate rejected summaries exactly on
    # the broad recap queries that need them, while rejecting nothing on a
    # genuine negative control. The rank window of 25 matches the typed
    # candidate depth floor, so it is inert at the default depth and only
    # gates summaries when a wider pool (top_k > 5) is requested.
    summary_rank_window: int = 25
    relevance_spread_fraction: float = 1.0
    overflow_limit: int = 6

    def __post_init__(self) -> None:
        if self.total_limit < 0:
            raise ValueError("total_limit must be non-negative")
        if self.raw_source_limit < 0:
            raise ValueError("raw_source_limit must be non-negative")
        if self.summary_source_limit < 0:
            raise ValueError("summary_source_limit must be non-negative")
        if self.assistant_derived_limit < 0:
            raise ValueError("assistant_derived_limit must be non-negative")
        if self.summary_rank_window < 1:
            raise ValueError("summary_rank_window must be at least one")
        if not 0.0 <= self.relevance_spread_fraction <= 1.0:
            raise ValueError("relevance_spread_fraction must be between zero and one")
        if self.overflow_limit < 0:
            raise ValueError("overflow_limit must be non-negative")


@dataclass(frozen=True)
class MemoryAdmissionResult:
    admitted_memories: tuple[ScoredEpisode, ...]
    skipped_memories: tuple[ScoredEpisode, ...]
    decisions: tuple[MemoryAdmissionDecision, ...]
    overflow_memories: tuple[ScoredEpisode, ...] = ()


def apply_typed_memory_admission(
    candidates: tuple[ScoredEpisode, ...],
    *,
    config: TypedMemoryAdmissionConfig,
    excluded_message_ids: tuple[UUID, ...],
) -> MemoryAdmissionResult:
    """Partition one retrieval result set and admit typed memory lanes.

    Lane limits are ceilings, not force-fill targets: summary candidates must
    pass rank-window and diversity gates before they can occupy the summary
    quota. Score-margin gating (summary, backfill, overflow) only binds when
    relevance_spread_fraction is configured below 1.0; at the default of 1.0
    those margin gates are disabled and only retrieval depth, the rank window,
    exclusions, and diversity checks filter candidates.
    """

    excluded_ids = set(excluded_message_ids)
    by_lane: dict[str, list[ScoredEpisode]] = {
        RAW_SOURCE_LANE: [],
        SUMMARY_SOURCE_LANE: [],
        ASSISTANT_DERIVED_LANE: [],
        UNSUPPORTED_NOISY_LANE: [],
    }
    initial_skip_reasons: dict[UUID, str] = {}

    for candidate in _rank_order(candidates):
        lane, skip_reason = classify_memory_candidate(candidate, excluded_message_ids=excluded_ids)
        by_lane[lane].append(candidate)
        if skip_reason is not None:
            initial_skip_reasons[candidate.id] = skip_reason

    source_candidates = _rank_order(
        tuple((*by_lane[RAW_SOURCE_LANE], *by_lane[SUMMARY_SOURCE_LANE]))
    )
    relevance_cut = _relevance_score_cut(source_candidates, config.relevance_spread_fraction)
    relevance_skip_reasons = _summary_relevance_skip_reasons(
        source_candidates,
        summary_rank_window=config.summary_rank_window,
        relevance_cut=relevance_cut,
    )
    diversity_skip_reasons: dict[UUID, str] = {}

    admitted_ids: set[UUID] = set()
    admitted_order: list[UUID] = []
    decisions_by_id: dict[UUID, MemoryAdmissionDecision] = {}
    admitted = _select_with_limit(
        by_lane[RAW_SOURCE_LANE],
        remaining_total=config.total_limit - len(admitted_ids),
        lane_limit=config.raw_source_limit,
    )
    _mark_admitted(
        admitted,
        lane=RAW_SOURCE_LANE,
        reason="raw_source_quota",
        admitted_ids=admitted_ids,
        admitted_order=admitted_order,
        decisions_by_id=decisions_by_id,
    )

    admitted_summaries: list[ScoredEpisode] = []
    admitted = _select_relevant_summaries(
        by_lane[SUMMARY_SOURCE_LANE],
        remaining_total=config.total_limit - len(admitted_ids),
        lane_limit=config.summary_source_limit,
        relevance_skip_reasons=relevance_skip_reasons,
        diversity_skip_reasons=diversity_skip_reasons,
        admitted_summaries=admitted_summaries,
    )
    _mark_admitted(
        admitted,
        lane=SUMMARY_SOURCE_LANE,
        reason="summary_source_quota",
        admitted_ids=admitted_ids,
        admitted_order=admitted_order,
        decisions_by_id=decisions_by_id,
    )

    backfill_skip_reasons: dict[UUID, str] = {}
    admitted = _select_backfill_sources(
        source_candidates,
        remaining_total=config.total_limit - len(admitted_ids),
        admitted_ids=admitted_ids,
        relevance_cut=relevance_cut,
        relevance_skip_reasons=relevance_skip_reasons,
        diversity_skip_reasons=diversity_skip_reasons,
        backfill_skip_reasons=backfill_skip_reasons,
        admitted_summaries=admitted_summaries,
    )
    _mark_admitted(
        admitted,
        lane=None,
        reason="source_backfill",
        admitted_ids=admitted_ids,
        admitted_order=admitted_order,
        decisions_by_id=decisions_by_id,
    )

    admitted = _select_with_limit(
        by_lane[ASSISTANT_DERIVED_LANE],
        remaining_total=config.total_limit - len(admitted_ids),
        lane_limit=config.assistant_derived_limit,
    )
    _mark_admitted(
        admitted,
        lane=ASSISTANT_DERIVED_LANE,
        reason="assistant_derived_quota",
        admitted_ids=admitted_ids,
        admitted_order=admitted_order,
        decisions_by_id=decisions_by_id,
    )

    overflow_memories = _select_overflow_memories(
        source_candidates,
        admitted_ids=admitted_ids,
        relevance_cut=relevance_cut,
        relevance_skip_reasons=relevance_skip_reasons,
        diversity_skip_reasons=diversity_skip_reasons,
        admitted_summaries=admitted_summaries,
        overflow_limit=config.overflow_limit,
    )

    for lane, lane_candidates in by_lane.items():
        for candidate in lane_candidates:
            if candidate.id in decisions_by_id:
                continue
            decisions_by_id[candidate.id] = MemoryAdmissionDecision(
                memory=candidate,
                lane=lane,
                admitted=False,
                admission_reason=None,
                skip_reason=_skip_reason(
                    candidate,
                    lane=lane,
                    config=config,
                    initial_skip_reasons=initial_skip_reasons,
                    relevance_skip_reasons=relevance_skip_reasons,
                    diversity_skip_reasons=diversity_skip_reasons,
                    backfill_skip_reasons=backfill_skip_reasons,
                    total_is_full=len(admitted_ids) >= config.total_limit,
                ),
            )

    decisions = tuple(decisions_by_id[candidate.id] for candidate in _rank_order(candidates))
    admitted_memories = tuple(decisions_by_id[episode_id].memory for episode_id in admitted_order)
    skipped_memories = tuple(decision.memory for decision in decisions if not decision.admitted)
    return MemoryAdmissionResult(
        admitted_memories=admitted_memories,
        skipped_memories=skipped_memories,
        decisions=decisions,
        overflow_memories=overflow_memories,
    )


def classify_memory_candidate(
    candidate: ScoredEpisode,
    *,
    excluded_message_ids: set[UUID],
) -> tuple[str, str | None]:
    """Classify one retrieved candidate using only durable runtime metadata."""

    if candidate.kind == "summary":
        if candidate.message_id is None:
            return SUMMARY_SOURCE_LANE, None
        return UNSUPPORTED_NOISY_LANE, "invalid_summary_shape"

    if candidate.kind != "message":
        return UNSUPPORTED_NOISY_LANE, "unsupported_episode_kind"
    if candidate.message_id is None:
        return UNSUPPORTED_NOISY_LANE, "missing_message_id"
    if candidate.message_id in excluded_message_ids:
        return UNSUPPORTED_NOISY_LANE, "excluded_recent_or_active_message"
    if candidate.message_role == "user":
        return RAW_SOURCE_LANE, None
    if candidate.message_role == "assistant":
        return ASSISTANT_DERIVED_LANE, None
    if candidate.message_role == "system":
        return UNSUPPORTED_NOISY_LANE, "system_message"
    return UNSUPPORTED_NOISY_LANE, "unknown_message_role"


def finalize_memory_admission_decisions(
    decisions: tuple[MemoryAdmissionDecision, ...],
    *,
    selected_memories: tuple[ScoredEpisode, ...],
    prompt_skipped_memories: tuple[ScoredEpisode, ...],
    overflow_selected_memories: tuple[ScoredEpisode, ...] = (),
) -> tuple[MemoryAdmissionDecision, ...]:
    """Update policy decisions so admitted means final prompt admission."""

    selected_ids = {memory.id for memory in selected_memories}
    prompt_skipped_ids = {memory.id for memory in prompt_skipped_memories}
    overflow_selected_ids = {memory.id for memory in overflow_selected_memories}
    finalized: list[MemoryAdmissionDecision] = []
    for decision in decisions:
        if decision.memory.id in selected_ids:
            admission_reason = (
                PROMPT_OVERFLOW_BACKFILL_REASON
                if decision.memory.id in overflow_selected_ids
                else decision.admission_reason
            )
            finalized.append(
                replace(
                    decision,
                    admitted=True,
                    admission_reason=admission_reason,
                    skip_reason=None,
                )
            )
        elif decision.admitted and decision.memory.id in prompt_skipped_ids:
            finalized.append(
                replace(
                    decision,
                    admitted=False,
                    skip_reason="prompt_character_budget",
                )
            )
        else:
            finalized.append(decision)
    return tuple(finalized)


def legacy_memory_admission_decisions(
    *,
    retrieved_memories: tuple[ScoredEpisode, ...],
    selected_memories: tuple[ScoredEpisode, ...],
    skipped_memories: tuple[ScoredEpisode, ...],
) -> tuple[MemoryAdmissionDecision, ...]:
    """Represent legacy mixed-pool prompt admission in the common debug shape."""

    selected_ids = {memory.id for memory in selected_memories}
    skipped_ids = {memory.id for memory in skipped_memories}
    decisions: list[MemoryAdmissionDecision] = []
    for memory in _rank_order(retrieved_memories):
        admitted = memory.id in selected_ids
        decisions.append(
            MemoryAdmissionDecision(
                memory=memory,
                lane=LEGACY_MIXED_LANE,
                admitted=admitted,
                admission_reason="legacy_mixed_prompt_selection" if admitted else None,
                skip_reason="prompt_character_budget"
                if memory.id in skipped_ids and not admitted
                else None,
            )
        )
    return tuple(decisions)


def _rank_order(candidates: tuple[ScoredEpisode, ...]) -> tuple[ScoredEpisode, ...]:
    return tuple(
        sorted(candidates, key=lambda candidate: (candidate.result_rank, candidate.id.int))
    )


def _relevance_score_cut(
    source_candidates: tuple[ScoredEpisode, ...],
    spread_fraction: float,
) -> float:
    """Return the pool-relative minimum score for gated admissions.

    The cut is anchored to the observed score spread of this pool, so it keeps
    working when a local embedder produces low or negative absolute scores.
    """

    if len(source_candidates) < MARGIN_MIN_SOURCE_CANDIDATES:
        return float("-inf")
    best = max(candidate.score for candidate in source_candidates)
    worst = min(candidate.score for candidate in source_candidates)
    return best - spread_fraction * (best - worst)


def _summary_relevance_skip_reasons(
    source_candidates: tuple[ScoredEpisode, ...],
    *,
    summary_rank_window: int,
    relevance_cut: float,
) -> dict[UUID, str]:
    reasons: dict[UUID, str] = {}
    for source_rank, candidate in enumerate(source_candidates, start=1):
        if candidate.kind != "summary":
            continue
        if source_rank > summary_rank_window:
            reasons[candidate.id] = SUMMARY_RELEVANCE_RANK_SKIP
        elif candidate.score < relevance_cut:
            reasons[candidate.id] = SUMMARY_RELEVANCE_MARGIN_SKIP
    return reasons


def _summary_diversity_conflict(
    candidate: ScoredEpisode,
    admitted_summaries: list[ScoredEpisode],
) -> str | None:
    for admitted in admitted_summaries:
        if (
            candidate.conversation_id == admitted.conversation_id
            and candidate.range_start is not None
            and candidate.range_end is not None
            and admitted.range_start is not None
            and admitted.range_end is not None
            and candidate.range_start <= admitted.range_end
            and candidate.range_end >= admitted.range_start
        ):
            return SUMMARY_WINDOW_OVERLAP_SKIP
        if _content_jaccard(candidate.content, admitted.content) >= SUMMARY_NEAR_DUPLICATE_JACCARD:
            return SUMMARY_NEAR_DUPLICATE_SKIP
    return None


def _content_jaccard(first: str, second: str) -> float:
    first_tokens = set(first.lower().split())
    second_tokens = set(second.lower().split())
    if not first_tokens or not second_tokens:
        return 0.0
    intersection = len(first_tokens & second_tokens)
    union = len(first_tokens | second_tokens)
    return intersection / union


def _select_relevant_summaries(
    candidates: list[ScoredEpisode],
    *,
    remaining_total: int,
    lane_limit: int,
    relevance_skip_reasons: dict[UUID, str],
    diversity_skip_reasons: dict[UUID, str],
    admitted_summaries: list[ScoredEpisode],
) -> tuple[ScoredEpisode, ...]:
    if remaining_total <= 0 or lane_limit <= 0:
        return ()
    selected: list[ScoredEpisode] = []
    limit = min(remaining_total, lane_limit)
    for candidate in _rank_order(tuple(candidates)):
        if len(selected) >= limit:
            break
        if candidate.id in relevance_skip_reasons:
            continue
        conflict = _summary_diversity_conflict(candidate, admitted_summaries)
        if conflict is not None:
            diversity_skip_reasons[candidate.id] = conflict
            continue
        selected.append(candidate)
        admitted_summaries.append(candidate)
    return tuple(selected)


def _select_backfill_sources(
    source_candidates: tuple[ScoredEpisode, ...],
    *,
    remaining_total: int,
    admitted_ids: set[UUID],
    relevance_cut: float,
    relevance_skip_reasons: dict[UUID, str],
    diversity_skip_reasons: dict[UUID, str],
    backfill_skip_reasons: dict[UUID, str],
    admitted_summaries: list[ScoredEpisode],
) -> tuple[ScoredEpisode, ...]:
    selected: list[ScoredEpisode] = []
    for candidate in source_candidates:
        if candidate.id in admitted_ids or candidate.id in relevance_skip_reasons:
            continue
        if candidate.id in diversity_skip_reasons:
            continue
        # Margin gate for spare capacity; disabled at the default spread
        # fraction of 1.0, where the cut sits at the pool minimum.
        if candidate.score < relevance_cut:
            backfill_skip_reasons[candidate.id] = BACKFILL_RELEVANCE_MARGIN_SKIP
            continue
        if len(selected) >= remaining_total:
            continue
        if candidate.kind == "summary":
            conflict = _summary_diversity_conflict(candidate, admitted_summaries)
            if conflict is not None:
                diversity_skip_reasons[candidate.id] = conflict
                continue
            admitted_summaries.append(candidate)
        selected.append(candidate)
    return tuple(selected)


def _select_overflow_memories(
    source_candidates: tuple[ScoredEpisode, ...],
    *,
    admitted_ids: set[UUID],
    relevance_cut: float,
    relevance_skip_reasons: dict[UUID, str],
    diversity_skip_reasons: dict[UUID, str],
    admitted_summaries: list[ScoredEpisode],
    overflow_limit: int,
) -> tuple[ScoredEpisode, ...]:
    """Return eligible spare candidates for prompt-size backfill, in rank order."""

    if overflow_limit <= 0:
        return ()
    overflow: list[ScoredEpisode] = []
    overflow_summaries = list(admitted_summaries)
    for candidate in source_candidates:
        if len(overflow) >= overflow_limit:
            break
        if candidate.id in admitted_ids:
            continue
        if candidate.id in relevance_skip_reasons or candidate.id in diversity_skip_reasons:
            continue
        if candidate.score < relevance_cut:
            continue
        if candidate.kind == "summary":
            if _summary_diversity_conflict(candidate, overflow_summaries) is not None:
                continue
            overflow_summaries.append(candidate)
        overflow.append(candidate)
    return tuple(overflow)


def _select_with_limit(
    candidates: tuple[ScoredEpisode, ...] | list[ScoredEpisode],
    *,
    remaining_total: int,
    lane_limit: int,
) -> tuple[ScoredEpisode, ...]:
    if remaining_total <= 0 or lane_limit <= 0:
        return ()
    return tuple(_rank_order(tuple(candidates))[: min(remaining_total, lane_limit)])


def _mark_admitted(
    candidates: tuple[ScoredEpisode, ...],
    *,
    lane: str | None,
    reason: str,
    admitted_ids: set[UUID],
    admitted_order: list[UUID],
    decisions_by_id: dict[UUID, MemoryAdmissionDecision],
) -> None:
    for candidate in candidates:
        admitted_ids.add(candidate.id)
        admitted_order.append(candidate.id)
        candidate_lane = lane
        if candidate_lane is None:
            candidate_lane, _ = classify_memory_candidate(
                candidate,
                excluded_message_ids=set(),
            )
        decisions_by_id[candidate.id] = MemoryAdmissionDecision(
            memory=candidate,
            lane=candidate_lane,
            admitted=True,
            admission_reason=reason,
            skip_reason=None,
        )


def _skip_reason(
    candidate: ScoredEpisode,
    *,
    lane: str,
    config: TypedMemoryAdmissionConfig,
    initial_skip_reasons: dict[UUID, str],
    relevance_skip_reasons: dict[UUID, str],
    diversity_skip_reasons: dict[UUID, str],
    backfill_skip_reasons: dict[UUID, str],
    total_is_full: bool,
) -> str:
    if candidate.id in initial_skip_reasons:
        return initial_skip_reasons[candidate.id]
    if candidate.id in relevance_skip_reasons:
        return relevance_skip_reasons[candidate.id]
    if candidate.id in diversity_skip_reasons:
        return diversity_skip_reasons[candidate.id]
    if candidate.id in backfill_skip_reasons:
        return backfill_skip_reasons[candidate.id]
    if lane == ASSISTANT_DERIVED_LANE and config.assistant_derived_limit == 0:
        return "assistant_derived_disabled"
    if lane == ASSISTANT_DERIVED_LANE:
        if total_is_full:
            return "total_memory_limit_reached"
        return "assistant_derived_quota_exhausted"
    if lane in {RAW_SOURCE_LANE, SUMMARY_SOURCE_LANE}:
        return "total_memory_limit_reached" if total_is_full else "source_quota_exhausted"
    return "unsupported_noisy"
