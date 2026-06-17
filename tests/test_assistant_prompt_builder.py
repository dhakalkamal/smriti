from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from smriti.assistant import (
    FIXED_PRIVACY_INSTRUCTIONS,
    InvalidAssistantRequestError,
    PromptBuildRequest,
    build_chat_request,
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


def test_prompt_builder_selects_memories_by_score_and_stops_at_first_over_budget() -> None:
    query_message = _message(1, "user", "q")
    high_score_memory = _episode(rank=2, score=0.95, content="small high")
    oversized_memory = _episode(rank=1, score=0.9, content="x" * 500)
    later_small_memory = _episode(rank=3, score=0.1, content="small low")
    base_budget = (
        len("scope")
        + len(FIXED_PRIVACY_INSTRUCTIONS)
        + len(query_message.content)
        + len(_memory_context_content_for_test(high_score_memory))
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

    assert result.selected_memories == (high_score_memory,)
    assert result.skipped_memories == (oversized_memory, later_small_memory)
    prompt_contents = [message.content for message in result.chat_request.messages]
    assert any("small high" in content for content in prompt_contents)
    assert not any("small low" in content for content in prompt_contents)


def test_prompt_builder_reserves_selected_recent_context_before_memory() -> None:
    older_message = _message(1, "user", "older")
    recent_message = _message(2, "assistant", "recent")
    query_message = _message(3, "user", "query")
    memory = _episode(rank=1, score=0.9, content="memory" * 100)
    max_prompt_chars = (
        len("scope")
        + len(FIXED_PRIVACY_INSTRUCTIONS)
        + len(older_message.content)
        + len(recent_message.content)
        + len(query_message.content)
    )

    result = build_chat_request(
        PromptBuildRequest(
            scope_system_prompt="scope",
            retrieved_memories=(memory,),
            recent_messages=(older_message, recent_message, query_message),
            query_message_id=query_message.id,
            max_prompt_chars=max_prompt_chars,
        )
    )

    assert result.selected_memories == ()
    assert result.skipped_memories == (memory,)
    assert result.selected_recent_message_ids == (
        older_message.id,
        recent_message.id,
        query_message.id,
    )
    assert [message.content for message in result.chat_request.messages[-3:]] == [
        "older",
        "recent",
        "query",
    ]


def test_prompt_builder_adds_recent_messages_newest_first_for_budget_then_chronological() -> None:
    older_message = _message(1, "user", "older")
    newer_message = _message(2, "assistant", "newer")
    query_message = _message(3, "user", "query")
    max_prompt_chars = (
        len("scope")
        + len(FIXED_PRIVACY_INSTRUCTIONS)
        + len(query_message.content)
        + len(newer_message.content)
    )

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


def _episode(rank: int, score: float, content: str) -> ScoredEpisode:
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
    )


def _memory_context_content_for_test(memory: ScoredEpisode) -> str:
    from smriti.assistant.prompt_builder import _memory_context_content

    return _memory_context_content(memory)
