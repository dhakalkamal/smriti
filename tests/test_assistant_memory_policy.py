from __future__ import annotations

from dataclasses import replace
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


def test_typed_memory_policy_keeps_hybrid_candidates_on_runtime_lanes() -> None:
    lexical_assistant = replace(
        _message(rank=1, role="assistant"),
        candidate_mode="hybrid_v1",
        lexical_rank=1,
        lexical_score=2.0,
        lexical_match_types=("email",),
        fused_rank=1,
        fused_score=0.03,
    )
    lexical_raw = replace(
        _message(rank=2, role="user"),
        candidate_mode="hybrid_v1",
        lexical_rank=2,
        lexical_score=1.0,
        lexical_match_types=("rare_token",),
        fused_rank=2,
        fused_score=0.02,
    )

    result = apply_typed_memory_admission(
        (lexical_assistant, lexical_raw),
        config=TypedMemoryAdmissionConfig(
            total_limit=2,
            raw_source_limit=1,
            summary_source_limit=0,
            assistant_derived_limit=0,
        ),
        excluded_message_ids=(),
    )

    assert result.admitted_memories == (lexical_raw,)
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[lexical_assistant.id].lane == "assistant_derived"
    assert decisions_by_id[lexical_assistant.id].skip_reason == "assistant_derived_disabled"
    assert decisions_by_id[lexical_raw.id].lane == "raw_source"
    assert decisions_by_id[lexical_raw.id].admission_reason == "raw_source_quota"


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


def test_typed_memory_policy_margin_and_rank_gates_still_work_when_configured() -> None:
    # The margin gate is disabled by default (spread fraction 1.0); this test
    # pins tighter values to prove the opt-in mechanism still functions.
    raw_1 = _message(rank=1, role="user", score=1.0)
    raw_2 = _message(rank=2, role="user", score=0.98)
    weak_summary = _summary(rank=3, range_start=1, range_end=12, score=0.2)
    raw_4 = _message(rank=4, role="user", score=0.96)
    deep_summary = _summary(rank=5, range_start=13, range_end=24, score=0.95)

    result = apply_typed_memory_admission(
        (raw_1, raw_2, weak_summary, raw_4, deep_summary),
        config=TypedMemoryAdmissionConfig(
            total_limit=6,
            raw_source_limit=4,
            summary_source_limit=2,
            assistant_derived_limit=0,
            summary_rank_window=4,
            relevance_spread_fraction=0.5,
        ),
        excluded_message_ids=(),
    )

    # Summary slots stay empty instead of force-filling irrelevant summaries,
    # and the total admitted count is allowed to fall short of total_limit.
    assert result.admitted_memories == (raw_1, raw_2, raw_4)
    assert result.overflow_memories == ()
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[weak_summary.id].skip_reason == "summary_relevance_margin"
    assert decisions_by_id[deep_summary.id].skip_reason == "summary_relevance_rank"


def test_typed_memory_policy_rejects_overlapping_and_near_duplicate_summaries() -> None:
    first_summary = _summary(
        rank=1,
        range_start=1,
        range_end=12,
        content="user chose terrafold and obafemi lease terms",
    )
    overlapping_summary = _summary(
        rank=2,
        range_start=6,
        range_end=18,
        content="completely different follow up planning notes",
    )
    duplicate_summary = _summary(
        rank=3,
        range_start=20,
        range_end=24,
        content="user chose terrafold and obafemi lease terms",
    )
    distinct_summary = _summary(
        rank=4,
        range_start=30,
        range_end=34,
        content="kiln budget capped with nitrile glove requirement",
    )

    result = apply_typed_memory_admission(
        (first_summary, overlapping_summary, duplicate_summary, distinct_summary),
        config=TypedMemoryAdmissionConfig(
            total_limit=6,
            raw_source_limit=0,
            summary_source_limit=2,
            assistant_derived_limit=0,
        ),
        excluded_message_ids=(),
    )

    assert result.admitted_memories == (first_summary, distinct_summary)
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[overlapping_summary.id].skip_reason == "summary_window_overlap"
    assert decisions_by_id[duplicate_summary.id].skip_reason == "summary_near_duplicate"


def test_typed_memory_policy_backfill_respects_relevance_margin_when_configured() -> None:
    # Backfill margin gating shares the summary gate's spread fraction, so it
    # is also disabled by default; pin 0.5 to test the opt-in mechanism.
    strong_raws = tuple(
        _message(rank=rank, role="user", score=1.0 - rank / 100) for rank in range(1, 5)
    )
    weak_raw_5 = _message(rank=5, role="user", score=0.2)
    weak_raw_6 = _message(rank=6, role="user", score=0.19)

    result = apply_typed_memory_admission(
        (*strong_raws, weak_raw_5, weak_raw_6),
        config=TypedMemoryAdmissionConfig(
            total_limit=6,
            raw_source_limit=4,
            summary_source_limit=2,
            assistant_derived_limit=0,
            relevance_spread_fraction=0.5,
        ),
        excluded_message_ids=(),
    )

    # Backfill does not force-fill spare capacity with relatively weak
    # candidates, so admission falls short of total_limit.
    assert result.admitted_memories == strong_raws
    assert result.overflow_memories == ()
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[weak_raw_5.id].skip_reason == "backfill_relevance_margin"
    assert decisions_by_id[weak_raw_6.id].skip_reason == "backfill_relevance_margin"


def test_typed_memory_policy_default_admits_recap_summary_despite_raw_echo_outlier() -> None:
    # Phase 1.6 regression: a recap-question echo raw message with an outlier
    # score must not raise a pool-wide cutoff that rejects relevant summaries.
    # These scores mirror the failing live broad-recap shape: under the old
    # spread fraction of 0.5 the cut would be 0.675 and both summaries lost.
    echo_outlier = _message(rank=1, role="user", score=0.95)
    raw_2 = _message(rank=2, role="user", score=0.62)
    recap_summary = _summary(rank=3, range_start=1, range_end=12, score=0.60)
    raw_4 = _message(rank=4, role="user", score=0.58)
    second_summary = _summary(rank=5, range_start=13, range_end=24, score=0.55)
    raw_6 = _message(rank=6, role="user", score=0.40)

    result = apply_typed_memory_admission(
        (echo_outlier, raw_2, recap_summary, raw_4, second_summary, raw_6),
        config=TypedMemoryAdmissionConfig(),
        excluded_message_ids=(),
    )

    assert recap_summary in result.admitted_memories
    assert second_summary in result.admitted_memories
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[recap_summary.id].admission_reason == "summary_source_quota"
    assert decisions_by_id[second_summary.id].admission_reason == "summary_source_quota"


def test_typed_memory_policy_default_excludes_recent_and_active_messages() -> None:
    excluded_recent_id = UUID(int=501)
    active_query_id = UUID(int=502)
    excluded_recent = _message(rank=1, role="user", message_id=excluded_recent_id)
    active_query = _message(rank=2, role="user", message_id=active_query_id)
    raw_3 = _message(rank=3, role="user")

    result = apply_typed_memory_admission(
        (excluded_recent, active_query, raw_3),
        config=TypedMemoryAdmissionConfig(),
        excluded_message_ids=(excluded_recent_id, active_query_id),
    )

    assert result.admitted_memories == (raw_3,)
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[excluded_recent.id].skip_reason == "excluded_recent_or_active_message"
    assert decisions_by_id[active_query.id].skip_reason == "excluded_recent_or_active_message"


def test_typed_memory_policy_rank_window_still_gates_pools_deeper_than_default() -> None:
    # The default rank window (25) matches the typed candidate depth floor, so
    # it is inert at default depth; it must still gate summaries that land
    # beyond source rank 25 when a wider pool (top_k > 5) is retrieved.
    in_window_summary = _summary(rank=10, range_start=1, range_end=12)
    beyond_window_summary = _summary(rank=30, range_start=13, range_end=24)
    raws = tuple(_message(rank=rank, role="user") for rank in range(1, 30) if rank not in (10, 30))

    result = apply_typed_memory_admission(
        (*raws, in_window_summary, beyond_window_summary),
        config=TypedMemoryAdmissionConfig(),
        excluded_message_ids=(),
    )

    assert in_window_summary in result.admitted_memories
    assert beyond_window_summary not in result.admitted_memories
    decisions_by_id = {decision.memory.id: decision for decision in result.decisions}
    assert decisions_by_id[beyond_window_summary.id].skip_reason == "summary_relevance_rank"


def test_typed_memory_policy_returns_bounded_rank_ordered_overflow() -> None:
    raws = tuple(_message(rank=rank, role="user") for rank in range(1, 16))

    result = apply_typed_memory_admission(
        raws,
        config=TypedMemoryAdmissionConfig(
            total_limit=6,
            raw_source_limit=4,
            summary_source_limit=2,
            assistant_derived_limit=0,
            overflow_limit=6,
        ),
        excluded_message_ids=(),
    )

    assert result.admitted_memories == raws[:6]
    assert result.overflow_memories == raws[6:12]


def _lane(memory: ScoredEpisode) -> str:
    lane, _ = classify_memory_candidate(memory, excluded_message_ids=set())
    return lane


def _message(
    rank: int,
    role: str | None,
    message_id: UUID | None = None,
    score: float = 1.0,
) -> ScoredEpisode:
    return _episode(
        rank=rank,
        kind="message",
        message_id=message_id or UUID(int=1000 + rank),
        message_role=role,
        range_start=None,
        range_end=None,
        score=score,
    )


def _summary(
    rank: int,
    range_start: int = 1,
    range_end: int = 4,
    score: float = 1.0,
    content: str | None = None,
) -> ScoredEpisode:
    return _episode(
        rank=rank,
        kind="summary",
        message_id=None,
        message_role=None,
        range_start=range_start,
        range_end=range_end,
        score=score,
        content=content,
    )


def _episode(
    *,
    rank: int,
    kind: str,
    message_id: UUID | None,
    message_role: str | None,
    range_start: int | None,
    range_end: int | None,
    score: float = 1.0,
    content: str | None = None,
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
        content=content if content is not None else f"memory {rank}",
        created_at=FIXED_NOW,
        importance=0.0,
        access_count=0,
        last_accessed_at=None,
        embedding_model_id=1,
        similarity=score,
        recency_score=0.0,
        access_score=0.0,
        importance_score=0.0,
        frequency_score=0.0,
        score=score,
        message_role=message_role,  # type: ignore[arg-type]
    )
