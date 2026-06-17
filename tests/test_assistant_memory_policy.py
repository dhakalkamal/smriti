from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from smriti.assistant.memory_policy import (
    TypedMemoryAdmissionConfig,
    apply_typed_memory_admission,
    classify_memory_candidate,
)
from smriti.memory import ScoredEpisode

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017


def test_typed_memory_policy_classifies_runtime_lanes() -> None:
    excluded_message_id = UUID(int=99)

    assert _lane(_message(rank=1, role="user")) == "raw_source"
    assert _lane(_message(rank=2, role="assistant")) == "assistant_derived"
    assert _lane(_summary(rank=3)) == "summary_source"
    assert _lane(_message(rank=4, role="system")) == "unsupported_noisy"
    assert _lane(_message(rank=5, role=None)) == "unsupported_noisy"
    assert classify_memory_candidate(
        _message(rank=6, role="user", message_id=excluded_message_id),
        excluded_message_ids={excluded_message_id},
    ) == ("unsupported_noisy", "excluded_recent_or_active_message")


def test_typed_memory_policy_respects_raw_and_summary_quotas_then_source_backfill() -> None:
    raw_1 = _message(rank=1, role="user")
    raw_2 = _message(rank=2, role="user")
    raw_3 = _message(rank=3, role="user")
    summary_4 = _summary(rank=4)
    summary_5 = _summary(rank=5)

    result = apply_typed_memory_admission(
        (raw_1, raw_2, raw_3, summary_4, summary_5),
        config=TypedMemoryAdmissionConfig(
            total_limit=4,
            raw_source_limit=2,
            summary_source_limit=1,
            assistant_derived_limit=0,
        ),
        excluded_message_ids=(),
    )

    assert result.admitted_memories == (raw_1, raw_2, summary_4, raw_3)
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[raw_1.id].admission_reason == "raw_source_quota"
    assert decisions_by_id[raw_2.id].admission_reason == "raw_source_quota"
    assert decisions_by_id[summary_4.id].admission_reason == "summary_source_quota"
    assert decisions_by_id[raw_3.id].admission_reason == "source_backfill"
    assert decisions_by_id[summary_5.id].skip_reason == "total_memory_limit_reached"


def test_typed_memory_policy_skips_derived_and_unsupported_by_default() -> None:
    raw_1 = _message(rank=1, role="user")
    assistant_2 = _message(rank=2, role="assistant")
    unsupported_3 = _message(rank=3, role="system")
    summary_4 = _summary(rank=4)

    result = apply_typed_memory_admission(
        (raw_1, assistant_2, unsupported_3, summary_4),
        config=TypedMemoryAdmissionConfig(
            total_limit=4,
            raw_source_limit=1,
            summary_source_limit=1,
            assistant_derived_limit=0,
        ),
        excluded_message_ids=(),
    )

    assert result.admitted_memories == (raw_1, summary_4)
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[assistant_2.id].skip_reason == "assistant_derived_disabled"
    assert decisions_by_id[unsupported_3.id].skip_reason == "system_message"


def test_typed_memory_policy_can_admit_assistant_derived_after_source_lanes() -> None:
    raw_1 = _message(rank=1, role="user")
    assistant_2 = _message(rank=2, role="assistant")
    summary_3 = _summary(rank=3)
    assistant_4 = _message(rank=4, role="assistant")

    result = apply_typed_memory_admission(
        (raw_1, assistant_2, summary_3, assistant_4),
        config=TypedMemoryAdmissionConfig(
            total_limit=3,
            raw_source_limit=1,
            summary_source_limit=1,
            assistant_derived_limit=2,
        ),
        excluded_message_ids=(),
    )

    assert result.admitted_memories == (raw_1, summary_3, assistant_2)
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[assistant_2.id].admission_reason == "assistant_derived_quota"
    assert decisions_by_id[assistant_4.id].skip_reason == "total_memory_limit_reached"


def _lane(memory: ScoredEpisode) -> str:
    lane, _ = classify_memory_candidate(memory, excluded_message_ids=set())
    return lane


def _message(rank: int, role: str | None, message_id: UUID | None = None) -> ScoredEpisode:
    return _episode(
        rank=rank,
        kind="message",
        message_id=message_id or UUID(int=1000 + rank),
        message_role=role,
        range_start=None,
        range_end=None,
    )


def _summary(rank: int) -> ScoredEpisode:
    return _episode(
        rank=rank,
        kind="summary",
        message_id=None,
        message_role=None,
        range_start=1,
        range_end=4,
    )


def _episode(
    *,
    rank: int,
    kind: str,
    message_id: UUID | None,
    message_role: str | None,
    range_start: int | None,
    range_end: int | None,
) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=rank,
        id=UUID(int=rank),
        user_id=UUID(int=1),
        scope_id=UUID(int=2),
        conversation_id=UUID(int=3),
        kind=kind,  # type: ignore[arg-type]
        message_id=message_id,
        message_position=rank if message_id is not None else None,
        range_start=range_start,
        range_end=range_end,
        content=f"memory {rank}",
        created_at=FIXED_NOW,
        importance=0.0,
        access_count=0,
        last_accessed_at=None,
        embedding_model_id=1,
        similarity=1.0,
        recency_score=0.0,
        access_score=0.0,
        importance_score=0.0,
        frequency_score=0.0,
        score=1.0,
        message_role=message_role,  # type: ignore[arg-type]
    )
