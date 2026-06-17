from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from smriti.assistant import (
    AssistantGenerationFailedError,
    AssistantGenerationRequest,
    AssistantGenerationUnavailableError,
    AssistantOrchestrator,
    AssistantStreamDone,
    AssistantStreamError,
    AssistantStreamStart,
    AssistantStreamToken,
)
from smriti.chat import (
    ChatConfigurationError,
    ChatConnectionError,
    ChatRequest,
    ChatResponse,
    ChatResponseError,
    ChatStreamFinal,
    ChatTimeoutError,
    ChatUsage,
    FakeStreamingChatGenerator,
)
from smriti.memory import (
    AppendAssistantResponseWithProvenanceRequest,
    AssistantGenerationContextRecord,
    AssistantResponseRecord,
    ConversationNotFoundError,
    ConversationRecord,
    LoadAssistantGenerationContextRequest,
    MessageRecord,
    ScopeRecord,
    ScoredEpisode,
)

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017


@pytest.mark.asyncio
async def test_assistant_orchestrator_generates_once_and_persists_selected_memories() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 2, "user", "find memory")
    high_memory = _episode(UUID(int=10), rank=1, score=0.9, content="high memory")
    low_memory = _episode(UUID(int=11), rank=2, score=0.1, content="low memory" * 200)
    persisted_message = _message(UUID(int=20), conversation_id, 3, "assistant", "answer")
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[high_memory, low_memory],
        persisted=AssistantResponseRecord(
            message=persisted_message,
            used_episode_ids=(high_memory.id,),
            scoring_version="stage-5.2-weighted-v1",
            retrieved_at=FIXED_NOW,
        ),
    )
    chat_generator = _RecordingChatGenerator(
        ChatResponse(
            content="answer",
            model="fake-chat",
            finish_reason="stop",
            usage=ChatUsage(completion_tokens=7),
        )
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=chat_generator,
    )

    result = await orchestrator.generate(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message.id,
            top_k=2,
            max_prompt_chars=700,
        )
    )

    assert len(chat_generator.requests) == 1
    assert memory_service.retrieve_queries == ["find memory"]
    assert memory_service.retrieve_exclude_message_id_sets == [(query_message.id,)]
    assert memory_service.persist_requests[0].used == (high_memory,)
    assert memory_service.persist_requests[0].token_count == 7
    assert result.assistant_message == persisted_message
    assert result.chat_model == "fake-chat"
    assert result.finish_reason == "stop"
    assert result.used_memory_episode_ids == (high_memory.id,)


@pytest.mark.asyncio
async def test_assistant_orchestrator_streams_tokens_then_persists_before_done() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 2, "user", "find memory")
    memory = _episode(UUID(int=10), rank=1, score=0.9, content="high memory")
    persisted_message = _message(UUID(int=20), conversation_id, 3, "assistant", "hello world")
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[memory],
        persisted=AssistantResponseRecord(
            message=persisted_message,
            used_episode_ids=(memory.id,),
            scoring_version="stage-5.2-weighted-v1",
            retrieved_at=FIXED_NOW,
        ),
    )
    chat_generator = FakeStreamingChatGenerator(
        tokens=["hello", " world"],
        final=ChatStreamFinal(model="fake-stream", finish_reason="stop", usage=ChatUsage(2, 2)),
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=chat_generator,
    )
    request = AssistantGenerationRequest(
        user_id=user_id,
        scope_id=scope_id,
        conversation_id=conversation_id,
        query_message_id=query_message.id,
        top_k=1,
    )

    prepared = await orchestrator.prepare_stream(request)
    stream = orchestrator.stream_prepared(prepared)

    assert await anext(stream) == AssistantStreamStart(
        used_memory_episode_ids=(memory.id,),
        chat_model="fake-stream",
    )
    assert memory_service.persist_requests == []
    assert await anext(stream) == AssistantStreamToken(text="hello")
    assert memory_service.persist_requests == []
    assert await anext(stream) == AssistantStreamToken(text=" world")
    assert memory_service.persist_requests == []
    done = await anext(stream)

    assert isinstance(done, AssistantStreamDone)
    assert done.assistant_message == persisted_message
    assert done.chat_model == "fake-stream"
    assert done.finish_reason == "stop"
    assert done.used_memory_episode_ids == (memory.id,)
    assert memory_service.retrieve_exclude_message_id_sets == [(query_message.id,)]
    assert memory_service.persist_requests[0].content == "hello world"
    assert memory_service.persist_requests[0].used == (memory,)
    assert memory_service.persist_requests[0].token_count == 2


@pytest.mark.asyncio
async def test_assistant_orchestrator_excludes_selected_recent_context_from_memory() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    older_message = _message(UUID(int=30), conversation_id, 1, "user", "older context")
    recent_answer = _message(UUID(int=31), conversation_id, 2, "assistant", "recent answer")
    query_message = _message(UUID(int=4), conversation_id, 3, "user", "find memory")
    persisted_message = _message(UUID(int=20), conversation_id, 4, "assistant", "answer")
    memory_service = _FakeMemoryService(
        context=_context(
            user_id,
            scope_id,
            conversation_id,
            query_message,
            recent_messages=(older_message, recent_answer, query_message),
        ),
        retrieved=[],
        persisted=AssistantResponseRecord(
            message=persisted_message,
            used_episode_ids=(),
            scoring_version="stage-5.2-weighted-v1",
            retrieved_at=FIXED_NOW,
        ),
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=_RecordingChatGenerator(ChatResponse(content="answer", model="fake-chat")),
    )

    await orchestrator.generate(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message.id,
            top_k=2,
            max_prompt_chars=1000,
        )
    )

    assert memory_service.retrieve_exclude_message_id_sets == [
        (older_message.id, recent_answer.id, query_message.id)
    ]


@pytest.mark.asyncio
async def test_assistant_orchestrator_debug_preparation_exposes_assembly_boundaries() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    older_message = _message(UUID(int=30), conversation_id, 1, "user", "older context")
    query_message = _message(UUID(int=4), conversation_id, 2, "user", "find memory")
    memory = _episode(UUID(int=10), rank=1, score=0.9, content="high memory")
    memory_service = _FakeMemoryService(
        context=_context(
            user_id,
            scope_id,
            conversation_id,
            query_message,
            recent_messages=(older_message, query_message),
        ),
        retrieved=[memory],
        persisted=None,
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=_RecordingChatGenerator(ChatResponse(content="unused", model="fake-chat")),
    )

    assembly = await orchestrator.prepare_generation_debug(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message.id,
            top_k=1,
            max_prompt_chars=1000,
        )
    )

    assert assembly.active_query_message_id == query_message.id
    assert assembly.recent_context.selected_recent_message_ids == (
        older_message.id,
        query_message.id,
    )
    assert assembly.excluded_message_ids == (older_message.id, query_message.id)
    assert assembly.retrieved_memories == (memory,)
    assert assembly.prompt.selected_memories == (memory,)
    assert assembly.prompt_message_order == (
        "system:scope",
        "system:privacy",
        f"memory:{memory.id}",
        f"recent:user:{older_message.id}",
        f"recent:user:{query_message.id}",
    )
    assert assembly.active_query_occurrences == 1


@pytest.mark.asyncio
async def test_assistant_orchestrator_stream_failure_after_token_persists_nothing() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 1, "user", "question")
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[],
        persisted=None,
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=FakeStreamingChatGenerator(
            tokens=["partial"],
            error=ChatResponseError("PRIVATE_STREAM_FAILURE"),
            fail_after_tokens=1,
        ),
    )
    prepared = await orchestrator.prepare_stream(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message.id,
            top_k=1,
        )
    )

    events = [event async for event in orchestrator.stream_prepared(prepared)]

    assert events == [
        AssistantStreamStart(
            used_memory_episode_ids=(),
            chat_model="fake-streaming-chat-generator",
        ),
        AssistantStreamToken(text="partial"),
        AssistantStreamError(
            code="assistant_generation_failed",
            message="Local assistant generation failed",
        ),
    ]
    assert memory_service.persist_requests == []


@pytest.mark.asyncio
async def test_assistant_orchestrator_stream_generator_failure_persists_nothing() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 1, "user", "question")
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[],
        persisted=None,
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=FakeStreamingChatGenerator(
            tokens=[],
            error=ChatConnectionError("PRIVATE_CONNECTION_FAILURE"),
        ),
    )
    prepared = await orchestrator.prepare_stream(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message.id,
            top_k=1,
        )
    )

    events = [event async for event in orchestrator.stream_prepared(prepared)]

    assert events == [
        AssistantStreamStart(
            used_memory_episode_ids=(),
            chat_model="fake-streaming-chat-generator",
        ),
        AssistantStreamError(
            code="assistant_generation_unavailable",
            message="Local assistant generation unavailable",
        ),
    ]
    assert memory_service.persist_requests == []


@pytest.mark.asyncio
async def test_assistant_orchestrator_stream_cancellation_persists_nothing() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 1, "user", "question")
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[],
        persisted=None,
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=FakeStreamingChatGenerator(tokens=["partial", " ignored"]),
    )
    prepared = await orchestrator.prepare_stream(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message.id,
            top_k=1,
        )
    )
    stream = orchestrator.stream_prepared(prepared)

    assert isinstance(await anext(stream), AssistantStreamStart)
    assert await anext(stream) == AssistantStreamToken(text="partial")
    await stream.aclose()

    assert memory_service.persist_requests == []


@pytest.mark.asyncio
async def test_assistant_orchestrator_uses_character_token_fallback_for_missing_usage() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 1, "user", "question")
    persisted_message = _message(UUID(int=20), conversation_id, 2, "assistant", "abcd")
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[],
        persisted=AssistantResponseRecord(
            message=persisted_message,
            used_episode_ids=(),
            scoring_version="stage-5.2-weighted-v1",
            retrieved_at=FIXED_NOW,
        ),
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=_RecordingChatGenerator(ChatResponse(content="abcd", model="fake-chat")),
    )

    await orchestrator.generate(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message.id,
            top_k=1,
        )
    )

    assert memory_service.persist_requests[0].token_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_error", "assistant_error"),
    [
        (ChatConnectionError("down"), AssistantGenerationUnavailableError),
        (ChatTimeoutError("slow"), AssistantGenerationUnavailableError),
        (ChatResponseError("bad"), AssistantGenerationFailedError),
        (ChatConfigurationError("bad config"), AssistantGenerationFailedError),
    ],
)
async def test_assistant_orchestrator_maps_chat_errors(
    chat_error: Exception,
    assistant_error: type[Exception],
) -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 1, "user", "question")
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[],
        persisted=None,
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=_FailingChatGenerator(chat_error),
    )

    with pytest.raises(assistant_error):
        await orchestrator.generate(
            AssistantGenerationRequest(
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                query_message_id=query_message.id,
                top_k=1,
            )
        )

    assert memory_service.persist_requests == []


@pytest.mark.asyncio
async def test_assistant_orchestrator_propagates_memory_errors_unchanged() -> None:
    expected = ConversationNotFoundError("missing")
    orchestrator = AssistantOrchestrator(
        memory_service=_FailingMemoryService(expected),  # type: ignore[arg-type]
        chat_generator=_RecordingChatGenerator(ChatResponse(content="unused", model="fake")),
    )

    with pytest.raises(ConversationNotFoundError) as exc_info:
        await orchestrator.generate(
            AssistantGenerationRequest(
                user_id=UUID(int=1),
                scope_id=UUID(int=2),
                conversation_id=UUID(int=3),
                query_message_id=UUID(int=4),
                top_k=1,
            )
        )

    assert exc_info.value is expected


@pytest.mark.asyncio
async def test_assistant_orchestrator_does_not_log_content_at_info_or_below(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "ASSISTANT_STAGE_7_3_SENTINEL"
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 1, "user", sentinel)
    persisted_message = _message(UUID(int=20), conversation_id, 2, "assistant", sentinel)
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[_episode(UUID(int=10), rank=1, score=0.9, content=sentinel)],
        persisted=AssistantResponseRecord(
            message=persisted_message,
            used_episode_ids=(UUID(int=10),),
            scoring_version="stage-5.2-weighted-v1",
            retrieved_at=FIXED_NOW,
        ),
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=_RecordingChatGenerator(ChatResponse(content=sentinel, model="fake")),
    )

    with caplog.at_level(logging.DEBUG):
        await orchestrator.generate(
            AssistantGenerationRequest(
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                query_message_id=query_message.id,
                top_k=1,
            )
        )

    assistant_records = [
        record
        for record in caplog.records
        if (record.name == "smriti.assistant" or record.name.startswith("smriti.assistant."))
        and record.levelno <= logging.INFO
    ]
    assert all(sentinel not in record.getMessage() for record in assistant_records)


@pytest.mark.asyncio
async def test_assistant_orchestrator_stream_does_not_log_content_at_info_or_below(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "ASSISTANT_STAGE_7_6_SENTINEL"
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message = _message(UUID(int=4), conversation_id, 1, "user", sentinel)
    persisted_message = _message(UUID(int=20), conversation_id, 2, "assistant", sentinel)
    memory_service = _FakeMemoryService(
        context=_context(user_id, scope_id, conversation_id, query_message),
        retrieved=[_episode(UUID(int=10), rank=1, score=0.9, content=sentinel)],
        persisted=AssistantResponseRecord(
            message=persisted_message,
            used_episode_ids=(UUID(int=10),),
            scoring_version="stage-5.2-weighted-v1",
            retrieved_at=FIXED_NOW,
        ),
    )
    orchestrator = AssistantOrchestrator(
        memory_service=memory_service,  # type: ignore[arg-type]
        chat_generator=FakeStreamingChatGenerator(tokens=[sentinel]),
    )

    with caplog.at_level(logging.DEBUG):
        prepared = await orchestrator.prepare_stream(
            AssistantGenerationRequest(
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                query_message_id=query_message.id,
                top_k=1,
            )
        )
        _ = [event async for event in orchestrator.stream_prepared(prepared)]

    assistant_records = [
        record
        for record in caplog.records
        if (record.name == "smriti.assistant" or record.name.startswith("smriti.assistant."))
        and record.levelno <= logging.INFO
    ]
    assert all(sentinel not in record.getMessage() for record in assistant_records)


def test_assistant_package_does_not_access_database_directly() -> None:
    assistant_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "assistant"
    forbidden_patterns = (
        "import asyncpg",
        "from asyncpg",
        "smriti.db",
        "connection.transaction",
        "conn.transaction",
        "self._pool",
        "._pool",
    )

    for path in assistant_dir.rglob("*.py"):
        source = path.read_text()
        for pattern in forbidden_patterns:
            assert pattern not in source, f"{path}: forbidden database access pattern {pattern!r}"


@dataclass
class _FakeMemoryService:
    context: AssistantGenerationContextRecord
    retrieved: list[ScoredEpisode]
    persisted: AssistantResponseRecord | None
    retrieve_queries: list[str] = field(default_factory=list)
    retrieve_exclude_message_id_sets: list[tuple[UUID, ...]] = field(default_factory=list)
    persist_requests: list[AppendAssistantResponseWithProvenanceRequest] = field(
        default_factory=list
    )

    async def load_assistant_generation_context(
        self,
        request: LoadAssistantGenerationContextRequest,
    ) -> AssistantGenerationContextRecord:
        _ = request
        return self.context

    async def retrieve_scoped_episodes(
        self,
        user_id: UUID,
        scope_id: UUID,
        query: str,
        top_k: int,
        *,
        exclude_message_id: UUID | None = None,
        exclude_message_ids: tuple[UUID, ...] = (),
    ) -> list[ScoredEpisode]:
        _ = (user_id, scope_id, top_k)
        self.retrieve_queries.append(query)
        ids = tuple(exclude_message_ids)
        if exclude_message_id is not None and exclude_message_id not in ids:
            ids = (*ids, exclude_message_id)
        self.retrieve_exclude_message_id_sets.append(ids)
        return self.retrieved

    async def append_assistant_response_with_provenance(
        self,
        request: AppendAssistantResponseWithProvenanceRequest,
    ) -> AssistantResponseRecord:
        self.persist_requests.append(request)
        assert self.persisted is not None
        return self.persisted


@dataclass
class _FailingMemoryService:
    error: Exception

    async def load_assistant_generation_context(
        self,
        request: LoadAssistantGenerationContextRequest,
    ) -> AssistantGenerationContextRecord:
        _ = request
        raise self.error


@dataclass
class _RecordingChatGenerator:
    response: ChatResponse
    requests: list[ChatRequest] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.response.model

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self.response


@dataclass
class _FailingChatGenerator:
    error: Exception

    @property
    def model(self) -> str:
        return "failing-chat"

    async def generate(self, request: ChatRequest) -> ChatResponse:
        _ = request
        raise self.error


def _context(
    user_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
    query_message: MessageRecord,
    recent_messages: tuple[MessageRecord, ...] | None = None,
) -> AssistantGenerationContextRecord:
    return AssistantGenerationContextRecord(
        scope=ScopeRecord(
            id=scope_id,
            user_id=user_id,
            name="Scope",
            system_prompt="scope prompt",
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
        ),
        conversation=ConversationRecord(
            id=conversation_id,
            user_id=user_id,
            scope_id=scope_id,
            title="Conversation",
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
        ),
        query_message=query_message,
        recent_messages=(query_message,) if recent_messages is None else recent_messages,
    )


def _message(
    message_id: UUID,
    conversation_id: UUID,
    position: int,
    role: str,
    content: str,
) -> MessageRecord:
    return MessageRecord(
        id=message_id,
        conversation_id=conversation_id,
        position=position,
        role=role,  # type: ignore[arg-type]
        content=content,
        token_count=max(1, len(content.split())),
        created_at=FIXED_NOW,
    )


def _episode(episode_id: UUID, rank: int, score: float, content: str) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=rank,
        id=episode_id,
        user_id=UUID(int=1),
        scope_id=UUID(int=2),
        conversation_id=UUID(int=3),
        kind="message",
        message_id=UUID(int=100 + rank),
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
