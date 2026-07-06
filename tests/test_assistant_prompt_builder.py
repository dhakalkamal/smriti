from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from smriti.assistant import (
    FIXED_PRIVACY_INSTRUCTIONS,
    InvalidAssistantRequestError,
    PromptBuildRequest,
    build_chat_request,
    reserved_memory_chars,
)
from smriti.memory import MessageRecord, ScoredEpisode

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017


def test_prompt_builder_orders_sections_and_preserves_roles() -> None:
    query_message = _message(3, "user", "current question")
    assistant_message = _message(2, "assistant", "previous answer")
    selected_memory = _episode(rank=1, score=0.9, content="remembered detail")

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope prompt",
            retrieved_memories=(selected_memory,),
            recent_messages=(
                _message(1, "user", "previous question"),
                assistant_message,
                query_message,
            ),
            query_message_id=query_message.id,
            max_prompt_chars=1000,
        )
    )

    messages = result.chat_request.messages
    assert [message.role for message in messages] == [
        "system",
        "system",
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0].content == "scope prompt"
    assert messages[1].content == FIXED_PRIVACY_INSTRUCTIONS
    assert messages[2].content.startswith("Memory context ")
    assert "remembered detail" in messages[2].content
    assert messages[-1].content == "current question"
    assert [message.content for message in messages].count("current question") == 1
    assert result.selected_memories == (selected_memory,)
    assert result.selected_recent_message_ids == (
        _message(1, "user", "previous question").id,
        assistant_message.id,
        query_message.id,
    )


def test_prompt_builder_skips_oversized_memory_without_blocking_smaller_ones() -> None:
    query_message = _message(1, "user", "q")
    high_score_memory = _episode(rank=2, score=0.95, content="small high")
    oversized_memory = _episode(rank=1, score=0.9, content="x" * 500)
    later_small_memory = _episode(rank=3, score=0.1, content="small low")
    base_budget = (
        len("scope")
        + len(FIXED_PRIVACY_INSTRUCTIONS)
        + len(query_message.content)
        + len(_memory_context_content_for_test(high_score_memory))
        + len(_memory_context_content_for_test(later_small_memory))
    )

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(oversized_memory, later_small_memory, high_score_memory),
            recent_messages=(query_message,),
            query_message_id=query_message.id,
            max_prompt_chars=base_budget,
        )
    )

    assert result.selected_memories == (high_score_memory, later_small_memory)
    assert result.skipped_memories == (oversized_memory,)
    prompt_contents = [message.content for message in result.chat_request.messages]
    assert any("small high" in content for content in prompt_contents)
    assert any("small low" in content for content in prompt_contents)
    assert not any("x" * 500 in content for content in prompt_contents)


def test_prompt_builder_typed_v1_oversized_raw_does_not_block_smaller_summary() -> None:
    query_message = _message(1, "user", "q")
    oversized_raw_memory = _episode(rank=1, score=0.9, content="x" * 2000, role="user")
    small_summary_memory = _summary_episode(rank=2, score=0.8, content="summary source")

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(oversized_raw_memory, small_summary_memory),
            recent_messages=(query_message,),
            query_message_id=query_message.id,
            max_prompt_chars=1000,
            memory_prompt_style="typed_v1",
        )
    )

    assert result.selected_memories == (small_summary_memory,)
    assert result.skipped_memories == (oversized_raw_memory,)
    prompt_text = "\n".join(message.content for message in result.chat_request.messages)
    assert "Long-term summary memories" in prompt_text
    assert "summary source" in prompt_text
    assert "x" * 2000 not in prompt_text


def test_prompt_builder_typed_v1_uses_clean_memory_sections() -> None:
    query_message = _message(1, "user", "q")
    raw_memory = _episode(rank=1, score=0.9, content="raw source", role="user")
    summary_memory = _summary_episode(rank=2, score=0.8, content="summary source")

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(raw_memory, summary_memory),
            recent_messages=(query_message,),
            query_message_id=query_message.id,
            max_prompt_chars=1000,
            memory_prompt_style="typed_v1",
        )
    )

    prompt_text = "\n".join(message.content for message in result.chat_request.messages)
    assert result.selected_memories == (raw_memory, summary_memory)
    assert "Long-term source memories" in prompt_text
    assert "Long-term summary memories" in prompt_text
    assert "raw source" in prompt_text
    assert "summary source" in prompt_text
    assert "episode_id=" not in prompt_text
    assert "rank=" not in prompt_text
    assert "score=" not in prompt_text


def test_prompt_builder_typed_v1_overflow_backfills_after_size_skip() -> None:
    query_message = _message(1, "user", "q")
    small_raw = _episode(rank=1, score=0.9, content="raw source", role="user")
    oversized_summary = _summary_episode(rank=2, score=0.8, content="s" * 2000)
    overflow_summary = _summary_episode(rank=3, score=0.7, content="small overflow summary")

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(small_raw, oversized_summary),
            recent_messages=(query_message,),
            query_message_id=query_message.id,
            max_prompt_chars=1000,
            memory_prompt_style="typed_v1",
            overflow_memories=(overflow_summary,),
        )
    )

    assert result.selected_memories == (small_raw, overflow_summary)
    assert result.overflow_selected_memories == (overflow_summary,)
    assert oversized_summary in result.skipped_memories
    prompt_text = "\n".join(message.content for message in result.chat_request.messages)
    assert "small overflow summary" in prompt_text
    assert "s" * 2000 not in prompt_text


def test_prompt_builder_typed_v1_ignores_overflow_without_size_skip() -> None:
    query_message = _message(1, "user", "q")
    small_raw = _episode(rank=1, score=0.9, content="raw source", role="user")
    overflow_summary = _summary_episode(rank=2, score=0.7, content="unused overflow summary")

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(small_raw,),
            recent_messages=(query_message,),
            query_message_id=query_message.id,
            max_prompt_chars=1000,
            memory_prompt_style="typed_v1",
            overflow_memories=(overflow_summary,),
        )
    )

    assert result.selected_memories == (small_raw,)
    assert result.overflow_selected_memories == ()
    prompt_text = "\n".join(message.content for message in result.chat_request.messages)
    assert "unused overflow summary" not in prompt_text


def test_prompt_builder_reserves_memory_budget_before_recent_context() -> None:
    max_prompt_chars = 2000
    recent_budget = max_prompt_chars - reserved_memory_chars(max_prompt_chars)
    older_message = _message(1, "user", "older")
    big_recent_message = _message(2, "assistant", "r" * 1100)
    query_message = _message(3, "user", "query")
    memory = _episode(rank=1, score=0.9, content="memory " * 70)
    mandatory_chars = len("scope") + len(FIXED_PRIVACY_INSTRUCTIONS) + len(query_message.content)
    # The big recent message would fit the total budget but not the
    # recent-context share left after the long-term memory reservation.
    assert mandatory_chars + len(big_recent_message.content) <= max_prompt_chars
    assert mandatory_chars + len(big_recent_message.content) > recent_budget

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(memory,),
            recent_messages=(older_message, big_recent_message, query_message),
            query_message_id=query_message.id,
            max_prompt_chars=max_prompt_chars,
        )
    )

    assert result.selected_recent_message_ids == (query_message.id,)
    assert result.selected_memories == (memory,)
    assert result.skipped_memories == ()


def test_prompt_builder_keeps_reserved_memory_budget_when_recents_fill_their_share() -> None:
    max_prompt_chars = 2000
    memory_reserve = reserved_memory_chars(max_prompt_chars)
    query_message = _message(3, "user", "query")
    mandatory_chars = len("scope") + len(FIXED_PRIVACY_INSTRUCTIONS) + len(query_message.content)
    recent_budget = max_prompt_chars - memory_reserve
    filler_chars = recent_budget - mandatory_chars
    recent_message = _message(2, "assistant", "r" * filler_chars)
    memories = tuple(
        _padded_memory(rank=rank, score=1.0 - rank / 10, context_chars=memory_reserve // 3)
        for rank in range(1, 5)
    )

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=memories,
            recent_messages=(recent_message, query_message),
            query_message_id=query_message.id,
            max_prompt_chars=max_prompt_chars,
        )
    )

    # Recents consumed their entire share, so exactly the reserved budget is
    # left and must be filled with eligible memories.
    assert result.selected_recent_message_ids == (recent_message.id, query_message.id)
    assert result.selected_memories == memories[:3]
    selected_memory_chars = sum(
        len(message.content)
        for message in result.chat_request.messages
        if message.content.startswith("Memory context ")
    )
    assert selected_memory_chars == 3 * (memory_reserve // 3)
    assert selected_memory_chars >= memory_reserve - (memory_reserve // 3)


def test_prompt_builder_adds_recent_messages_newest_first_for_budget_then_chronological() -> None:
    older_message = _message(1, "user", "older")
    newer_message = _message(2, "assistant", "newer")
    query_message = _message(3, "user", "query")
    max_prompt_chars = 715
    recent_budget = max_prompt_chars - reserved_memory_chars(max_prompt_chars)
    mandatory_chars = len("scope") + len(FIXED_PRIVACY_INSTRUCTIONS) + len(query_message.content)
    # Room for the newer message but not both within the recent-context share.
    assert mandatory_chars + len(newer_message.content) <= recent_budget
    assert mandatory_chars + len(newer_message.content) + len(older_message.content) > recent_budget

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(),
            recent_messages=(older_message, newer_message, query_message),
            query_message_id=query_message.id,
            max_prompt_chars=max_prompt_chars,
        )
    )

    assert [message.content for message in result.chat_request.messages[-2:]] == [
        "newer",
        "query",
    ]
    assert result.selected_recent_message_ids == (newer_message.id, query_message.id)


def test_prompt_builder_rejects_mandatory_sections_over_budget() -> None:
    query_message = _message(1, "user", "query")

    with pytest.raises(InvalidAssistantRequestError):
        build_chat_request(
            PromptBuildRequest(
                scope_system_prompt="scope",
                retrieved_memories=(),
                recent_messages=(query_message,),
                query_message_id=query_message.id,
                max_prompt_chars=1,
            )
        )


def test_prompt_builder_requires_query_message_exactly_once() -> None:
    query_message = _message(1, "user", "query")

    with pytest.raises(InvalidAssistantRequestError):
        build_chat_request(
            PromptBuildRequest(
                scope_system_prompt="scope",
                retrieved_memories=(),
                recent_messages=(),
                query_message_id=query_message.id,
            )
        )

    with pytest.raises(InvalidAssistantRequestError):
        build_chat_request(
            PromptBuildRequest(
                scope_system_prompt="scope",
                retrieved_memories=(),
                recent_messages=(query_message, query_message),
                query_message_id=query_message.id,
            )
        )


def test_prompt_builder_requires_user_query_message() -> None:
    query_message = _message(1, "assistant", "not a query")

    with pytest.raises(InvalidAssistantRequestError):
        build_chat_request(
            PromptBuildRequest(
                scope_system_prompt="scope",
                retrieved_memories=(),
                recent_messages=(query_message,),
                query_message_id=query_message.id,
            )
        )


def _message(position: int, role: str, content: str) -> MessageRecord:
    return MessageRecord(
        id=UUID(int=position),
        conversation_id=UUID(int=999),
        position=position,
        role=role,  # type: ignore[arg-type]
        content=content,
        token_count=max(1, len(content.split())),
        created_at=FIXED_NOW,
    )


def _episode(rank: int, score: float, content: str, role: str = "user") -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=rank,
        id=UUID(int=1000 + rank),
        user_id=UUID(int=1),
        scope_id=UUID(int=2),
        conversation_id=UUID(int=999),
        kind="message",
        message_id=UUID(int=2000 + rank),
        message_position=rank,
        range_start=None,
        range_end=None,
        content=content,
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
        message_role=role,  # type: ignore[arg-type]
    )


def _summary_episode(rank: int, score: float, content: str) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=rank,
        id=UUID(int=2000 + rank),
        user_id=UUID(int=1),
        scope_id=UUID(int=2),
        conversation_id=UUID(int=999),
        kind="summary",
        message_id=None,
        message_position=None,
        range_start=1,
        range_end=4,
        content=content,
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
    )


def _memory_context_content_for_test(memory: ScoredEpisode) -> str:
    from smriti.assistant.prompt_builder import _memory_context_content

    return _memory_context_content(memory)


def _padded_memory(rank: int, score: float, context_chars: int) -> ScoredEpisode:
    probe = _episode(rank=rank, score=score, content="")
    header_chars = len(_memory_context_content_for_test(probe))
    if context_chars <= header_chars:
        raise ValueError("context_chars must exceed the memory context header size")
    return _episode(rank=rank, score=score, content="m" * (context_chars - header_chars))
