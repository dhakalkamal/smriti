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


@dataclass(frozen=True)
class TypedMemoryAdmissionConfig:
    total_limit: int = 6
    raw_source_limit: int = 4
    summary_source_limit: int = 2
    assistant_derived_limit: int = 0

    def __post_init__(self) -> None:
        if self.total_limit < 0:
            raise ValueError("total_limit must be non-negative")
        if self.raw_source_limit < 0:
            raise ValueError("raw_source_limit must be non-negative")
        if self.summary_source_limit < 0:
            raise ValueError("summary_source_limit must be non-negative")
        if self.assistant_derived_limit < 0:
            raise ValueError("assistant_derived_limit must be non-negative")


@dataclass(frozen=True)
class MemoryAdmissionResult:
    admitted_memories: tuple[ScoredEpisode, ...]
    skipped_memories: tuple[ScoredEpisode, ...]
    decisions: tuple[MemoryAdmissionDecision, ...]


def apply_typed_memory_admission(
    candidates: tuple[ScoredEpisode, ...],
    *,
    config: TypedMemoryAdmissionConfig,
    excluded_message_ids: tuple[UUID, ...],
) -> MemoryAdmissionResult:
    """Partition one retrieval result set and admit typed memory lanes."""

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

    admitted = _select_with_limit(
        by_lane[SUMMARY_SOURCE_LANE],
        remaining_total=config.total_limit - len(admitted_ids),
        lane_limit=config.summary_source_limit,
    )
    _mark_admitted(
        admitted,
        lane=SUMMARY_SOURCE_LANE,
        reason="summary_source_quota",
        admitted_ids=admitted_ids,
        admitted_order=admitted_order,
        decisions_by_id=decisions_by_id,
    )

    remaining_source_candidates = _rank_order(
        tuple(
            candidate
            for candidate in (*by_lane[RAW_SOURCE_LANE], *by_lane[SUMMARY_SOURCE_LANE])
            if candidate.id not in admitted_ids
        )
    )
    admitted = _select_with_limit(
        remaining_source_candidates,
        remaining_total=config.total_limit - len(admitted_ids),
        lane_limit=config.total_limit,
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
) -> tuple[MemoryAdmissionDecision, ...]:
    """Update policy decisions so admitted means final prompt admission."""

    selected_ids = {memory.id for memory in selected_memories}
    prompt_skipped_ids = {memory.id for memory in prompt_skipped_memories}
    finalized: list[MemoryAdmissionDecision] = []
    for decision in decisions:
        if decision.memory.id in selected_ids:
            finalized.append(replace(decision, admitted=True, skip_reason=None))
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
    total_is_full: bool,
) -> str:
    if candidate.id in initial_skip_reasons:
        return initial_skip_reasons[candidate.id]
    if lane == ASSISTANT_DERIVED_LANE and config.assistant_derived_limit == 0:
        return "assistant_derived_disabled"
    if lane == ASSISTANT_DERIVED_LANE:
        if total_is_full:
            return "total_memory_limit_reached"
        return "assistant_derived_quota_exhausted"
    if lane in {RAW_SOURCE_LANE, SUMMARY_SOURCE_LANE}:
        return "total_memory_limit_reached" if total_is_full else "source_quota_exhausted"
    return "unsupported_noisy"
