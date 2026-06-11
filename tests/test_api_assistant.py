from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from smriti.api import create_app
from smriti.api.dependencies import (
    get_assistant_orchestrator,
    get_current_local_user_id,
    get_summary_episode_memory_scheduler,
)
from smriti.api.routes import assistant as assistant_routes
from smriti.assistant import (
    AssistantGenerationFailedError,
    AssistantGenerationRequest,
    AssistantGenerationResult,
    AssistantGenerationUnavailableError,
    AssistantOrchestrator,
    AssistantStreamDone,
    AssistantStreamError,
    AssistantStreamEvent,
    AssistantStreamPreparation,
    AssistantStreamStart,
    AssistantStreamToken,
    InvalidAssistantRequestError,
)
from smriti.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    ChatStreamFinal,
    ChatStreamToken,
    FakeStreamingChatGenerator,
)
from smriti.config import Settings
from smriti.db.client import close_pool, get_pool
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    AppendMessageRequest,
    AppendMessageWithEpisodeRequest,
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    CreateConversationRequest,
    CreateScopeRequest,
    InvalidProvenanceTargetError,
    MemoryService,
    MemoryServiceError,
    MessageRecord,
    ScoredEpisode,
)

LOCAL_USER_ID = UUID("66666666-6666-4666-8666-666666666666")
FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017
VALIDATION_STATUS = 400


def test_assistant_route_generates_response_through_orchestrator() -> None:
    conversation_id = uuid4()
    scope_id = uuid4()
    query_message_id = uuid4()
    first_episode_id = uuid4()
    second_episode_id = uuid4()
    assistant_message = _message(
        message_id=uuid4(),
        conversation_id=conversation_id,
        content="Generated assistant answer.",
    )
    orchestrator = RecordingAssistantOrchestrator(
        result=AssistantGenerationResult(
            assistant_message=assistant_message,
            chat_model="fake-chat",
            finish_reason="stop",
            used_memory_episode_ids=(first_episode_id, second_episode_id),
        )
    )
    client = _client(orchestrator)

    response = client.post(
        f"/conversations/{conversation_id}/assistant-response",
        json={"scope_id": str(scope_id), "query_message_id": str(query_message_id)},
    )

    assert response.status_code == 201
    assert response.json() == {
        "assistant_message": {
            "id": str(assistant_message.id),
            "conversation_id": str(conversation_id),
            "position": 2,
            "role": "assistant",
            "content": "Generated assistant answer.",
            "token_count": assistant_message.token_count,
            "created_at": "2026-01-01T12:00:00Z",
        },
        "chat_model": "fake-chat",
        "finish_reason": "stop",
        "used_memory_episode_ids": [str(first_episode_id), str(second_episode_id)],
    }
    assert orchestrator.requests == [
        AssistantGenerationRequest(
            user_id=LOCAL_USER_ID,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message_id,
            top_k=5,
            max_prompt_chars=16000,
            recent_message_limit=20,
        )
    ]


def test_assistant_route_uses_body_limits_and_path_conversation_id_only() -> None:
    path_conversation_id = uuid4()
    body_conversation_id = uuid4()
    scope_id = uuid4()
    query_message_id = uuid4()
    orchestrator = RecordingAssistantOrchestrator(
        result=AssistantGenerationResult(
            assistant_message=_message(uuid4(), path_conversation_id, "answer"),
            chat_model="fake-chat",
            finish_reason=None,
            used_memory_episode_ids=(),
        )
    )
    client = _client(orchestrator)

    response = client.post(
        f"/conversations/{path_conversation_id}/assistant-response",
        json={
            "scope_id": str(scope_id),
            "query_message_id": str(query_message_id),
            "top_k": 50,
            "max_prompt_chars": 100000,
            "recent_message_limit": 200,
            "conversation_id": str(body_conversation_id),
        },
    )

    assert response.status_code == VALIDATION_STATUS
    assert orchestrator.requests == []

    accepted_response = client.post(
        f"/conversations/{path_conversation_id}/assistant-response",
        json={
            "scope_id": str(scope_id),
            "query_message_id": str(query_message_id),
            "top_k": 50,
            "max_prompt_chars": 100000,
            "recent_message_limit": 200,
        },
    )

    assert accepted_response.status_code == 201
    assert orchestrator.requests == [
        AssistantGenerationRequest(
            user_id=LOCAL_USER_ID,
            scope_id=scope_id,
            conversation_id=path_conversation_id,
            query_message_id=query_message_id,
            top_k=50,
            max_prompt_chars=100000,
            recent_message_limit=200,
        )
    ]


@pytest.mark.parametrize(
    "body",
    [
        {"scope_id": str(uuid4()), "query_message_id": str(uuid4()), "top_k": 0},
        {"scope_id": str(uuid4()), "query_message_id": str(uuid4()), "top_k": 51},
        {"scope_id": str(uuid4()), "query_message_id": str(uuid4()), "max_prompt_chars": 0},
        {"scope_id": str(uuid4()), "query_message_id": str(uuid4()), "max_prompt_chars": 100001},
        {"scope_id": str(uuid4()), "query_message_id": str(uuid4()), "recent_message_limit": 0},
        {"scope_id": str(uuid4()), "query_message_id": str(uuid4()), "recent_message_limit": 201},
        {"scope_id": "not-a-uuid", "query_message_id": str(uuid4())},
        {"scope_id": str(uuid4()), "query_message_id": "not-a-uuid"},
    ],
)
def test_assistant_route_validates_request_body(body: dict[str, object]) -> None:
    orchestrator = RecordingAssistantOrchestrator(
        result=AssistantGenerationResult(
            assistant_message=_message(uuid4(), uuid4(), "unused"),
            chat_model="fake-chat",
            finish_reason=None,
            used_memory_episode_ids=(),
        )
    )
    client = _client(orchestrator)

    response = client.post(f"/conversations/{uuid4()}/assistant-response", json=body)

    assert response.status_code == VALIDATION_STATUS
    assert orchestrator.requests == []


def test_assistant_route_rejects_client_supplied_scored_memories_and_provenance() -> None:
    orchestrator = RecordingAssistantOrchestrator(
        result=AssistantGenerationResult(
            assistant_message=_message(uuid4(), uuid4(), "unused"),
            chat_model="fake-chat",
            finish_reason=None,
            used_memory_episode_ids=(),
        )
    )
    client = _client(orchestrator)

    response = client.post(
        f"/conversations/{uuid4()}/assistant-response",
        json={
            "scope_id": str(uuid4()),
            "query_message_id": str(uuid4()),
            "scored_memories": [],
            "provenance_rows": [],
        },
    )

    assert response.status_code == VALIDATION_STATUS
    assert orchestrator.requests == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (InvalidAssistantRequestError("invalid"), 400),
        (AssistantGenerationUnavailableError("unavailable"), 503),
        (AssistantGenerationFailedError("failed"), 500),
    ],
)
def test_assistant_route_maps_assistant_errors(
    error: Exception,
    expected_status: int,
) -> None:
    response = _client(RaisingAssistantOrchestrator(error)).post(
        f"/conversations/{uuid4()}/assistant-response",
        json={"scope_id": str(uuid4()), "query_message_id": str(uuid4())},
    )

    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ConversationAccessDeniedError("wrong user"), 403),
        (ConversationNotFoundError("missing conversation"), 404),
        (InvalidProvenanceTargetError("bad provenance target"), 400),
        (MemoryServiceError("storage failed"), 500),
    ],
)
def test_assistant_route_preserves_memory_error_mappings(
    error: Exception,
    expected_status: int,
) -> None:
    response = _client(RaisingAssistantOrchestrator(error)).post(
        f"/conversations/{uuid4()}/assistant-response",
        json={"scope_id": str(uuid4()), "query_message_id": str(uuid4())},
    )

    assert response.status_code == expected_status


def test_assistant_stream_route_emits_start_tokens_and_done() -> None:
    conversation_id = uuid4()
    scope_id = uuid4()
    query_message_id = uuid4()
    first_episode_id = uuid4()
    assistant_message = _message(
        message_id=uuid4(),
        conversation_id=conversation_id,
        content="Hello streamed answer.",
    )
    orchestrator = StreamingAssistantOrchestrator(
        events=[
            AssistantStreamStart(
                used_memory_episode_ids=(first_episode_id,),
                chat_model="fake-stream",
            ),
            AssistantStreamToken(text="Hello "),
            AssistantStreamToken(text="streamed answer."),
            AssistantStreamDone(
                assistant_message=assistant_message,
                chat_model="fake-stream",
                finish_reason="stop",
                used_memory_episode_ids=(first_episode_id,),
            ),
        ]
    )
    client = _client(orchestrator)

    with client.stream(
        "POST",
        f"/conversations/{conversation_id}/assistant-response/stream",
        json={"scope_id": str(scope_id), "query_message_id": str(query_message_id)},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse(body) == [
        {
            "event": "start",
            "data": {
                "used_memory_episode_ids": [str(first_episode_id)],
                "chat_model": "fake-stream",
            },
        },
        {"event": "token", "data": {"text": "Hello "}},
        {"event": "token", "data": {"text": "streamed answer."}},
        {
            "event": "done",
            "data": {
                "assistant_message": {
                    "id": str(assistant_message.id),
                    "conversation_id": str(conversation_id),
                    "position": 2,
                    "role": "assistant",
                    "content": "Hello streamed answer.",
                    "token_count": assistant_message.token_count,
                    "created_at": "2026-01-01T12:00:00Z",
                },
                "chat_model": "fake-stream",
                "finish_reason": "stop",
                "used_memory_episode_ids": [str(first_episode_id)],
            },
        },
    ]
    assert orchestrator.prepare_requests == [
        AssistantGenerationRequest(
            user_id=LOCAL_USER_ID,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message_id,
            top_k=5,
            max_prompt_chars=16000,
            recent_message_limit=20,
        )
    ]


def test_assistant_stream_route_maps_pre_stream_errors_to_http_responses() -> None:
    response = _client(
        StreamingAssistantOrchestrator(prepare_error=InvalidAssistantRequestError())
    ).post(
        f"/conversations/{uuid4()}/assistant-response/stream",
        json={"scope_id": str(uuid4()), "query_message_id": str(uuid4())},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid assistant request"}


def test_assistant_stream_route_emits_post_token_error_event() -> None:
    orchestrator = StreamingAssistantOrchestrator(
        events=[
            AssistantStreamStart(used_memory_episode_ids=(), chat_model="fake-stream"),
            AssistantStreamToken(text="partial"),
            AssistantStreamError(
                code="assistant_generation_failed",
                message="Local assistant generation failed",
            ),
        ]
    )

    response = _client(orchestrator).post(
        f"/conversations/{uuid4()}/assistant-response/stream",
        json={"scope_id": str(uuid4()), "query_message_id": str(uuid4())},
    )

    assert response.status_code == 200
    assert _parse_sse(response.text) == [
        {
            "event": "start",
            "data": {"used_memory_episode_ids": [], "chat_model": "fake-stream"},
        },
        {"event": "token", "data": {"text": "partial"}},
        {
            "event": "error",
            "data": {
                "code": "assistant_generation_failed",
                "message": "Local assistant generation failed",
            },
        },
    ]


def test_assistant_stream_route_payloads_do_not_include_memory_content() -> None:
    memory_sentinel = "RETRIEVED_MEMORY_CONTENT_SENTINEL"
    orchestrator = StreamingAssistantOrchestrator(
        events=[
            AssistantStreamStart(used_memory_episode_ids=(uuid4(),), chat_model="fake-stream"),
            AssistantStreamToken(text="generated token"),
            AssistantStreamDone(
                assistant_message=_message(uuid4(), uuid4(), "generated token"),
                chat_model="fake-stream",
                finish_reason="stop",
                used_memory_episode_ids=(),
            ),
        ]
    )

    response = _client(orchestrator).post(
        f"/conversations/{uuid4()}/assistant-response/stream",
        json={"scope_id": str(uuid4()), "query_message_id": str(uuid4())},
    )

    assert response.status_code == 200
    assert memory_sentinel not in response.text


@pytest.mark.asyncio
async def test_assistant_stream_route_disconnect_stops_before_persistence() -> None:
    user_id = uuid4()
    scope_id = uuid4()
    conversation_id = uuid4()
    query_message_id = uuid4()
    orchestrator = StreamingAssistantOrchestrator(
        events=[
            AssistantStreamStart(used_memory_episode_ids=(), chat_model="fake-stream"),
            AssistantStreamToken(text="partial"),
            AssistantStreamDone(
                assistant_message=_message(uuid4(), conversation_id, "should not be sent"),
                chat_model="fake-stream",
                finish_reason="stop",
                used_memory_episode_ids=(),
            ),
        ]
    )
    prepared = await orchestrator.prepare_stream(
        AssistantGenerationRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message_id,
            top_k=1,
        )
    )
    request = _DisconnectingRequest(disconnect_after_checks=2)

    chunks = [
        chunk
        async for chunk in assistant_routes._stream_sse_events(
            request=cast(Request, request),
            events=orchestrator.stream_prepared(prepared),
        )
    ]

    assert [event["event"] for event in _parse_sse("".join(chunks))] == ["start", "token"]
    assert orchestrator.events_sent == 2


@pytest.mark.asyncio
async def test_assistant_stream_route_disconnect_persists_no_rows_in_postgres() -> None:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(
            database_url=_to_asyncpg_dsn(postgres.get_connection_url()),
            local_user_id=LOCAL_USER_ID,
        )
        await apply_migrations(settings=settings, migrations_dir=settings.migrations_dir)
        chat_generator = _BlockingStreamingChatGenerator()
        app = create_app(
            settings=settings,
            embedder=FakeEmbedder(dimensions=768),
            chat_generator=chat_generator,
        )

        async with app.router.lifespan_context(app):
            transport = _StreamingASGITransport(app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                scope = await _create_scope(client)
                conversation = await _create_conversation(client, scope_id=UUID(scope["id"]))
                query_message = await _create_user_message(
                    client,
                    conversation_id=UUID(conversation["id"]),
                )

                chunks: list[str] = []
                async with client.stream(
                    "POST",
                    f"/conversations/{conversation['id']}/assistant-response/stream",
                    json={
                        "scope_id": scope["id"],
                        "query_message_id": query_message["id"],
                    },
                ) as response:
                    assert response.status_code == 200
                    async for chunk in response.aiter_text():
                        chunks.append(chunk)
                        if "event: token" in "".join(chunks):
                            break

                streamed = _parse_sse("".join(chunks))
                assert [event["event"] for event in streamed] == ["start", "token"]
                assert chat_generator.first_token_sent.is_set()

        counts = await _assistant_stream_storage_counts(
            settings=settings,
            conversation_id=UUID(conversation["id"]),
        )
        assert counts == {"assistant_messages": 0, "message_retrievals": 0}


@pytest.mark.asyncio
async def test_assistant_stream_route_schedules_summary_and_creates_summary_row() -> None:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(
            database_url=_to_asyncpg_dsn(postgres.get_connection_url()),
            local_user_id=LOCAL_USER_ID,
            summary_episode_memory_enabled=True,
        )
        await apply_migrations(settings=settings, migrations_dir=settings.migrations_dir)
        chat_generator = FakeStreamingChatGenerator(
            tokens=["runtime", " summary"],
            final=ChatStreamFinal(model="fake-stream", finish_reason="stop"),
        )
        app = create_app(
            settings=settings,
            embedder=FakeEmbedder(dimensions=768),
            chat_generator=chat_generator,
        )

        async with app.router.lifespan_context(app):
            state = app.state.smriti
            memory_service = cast(MemoryService, state.memory_service)
            scope = await memory_service.create_scope(
                CreateScopeRequest(
                    user_id=LOCAL_USER_ID,
                    name="Streaming summary scope",
                    system_prompt="Keep streaming summary memory scoped.",
                )
            )
            conversation = await memory_service.create_conversation(
                CreateConversationRequest(
                    user_id=LOCAL_USER_ID,
                    scope_id=scope.id,
                    title="Streaming summary conversation",
                )
            )
            for index in range(11):
                await memory_service.append_message_with_episode(
                    AppendMessageWithEpisodeRequest(
                        user_id=LOCAL_USER_ID,
                        conversation_id=conversation.id,
                        role="user",
                        content=f"seed user memory {index}",
                        token_count=4,
                    )
                )
                await memory_service.append_message(
                    AppendMessageRequest(
                        user_id=LOCAL_USER_ID,
                        scope_id=scope.id,
                        conversation_id=conversation.id,
                        role="assistant",
                        content=f"seed assistant response {index}",
                        token_count=4,
                    )
                )
            query_message = await memory_service.append_message_with_episode(
                AppendMessageWithEpisodeRequest(
                    user_id=LOCAL_USER_ID,
                    conversation_id=conversation.id,
                    role="user",
                    content="Please answer and trigger summary memory.",
                    token_count=6,
                )
            )

            transport = _StreamingASGITransport(app)
            async with (
                httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client,
                client.stream(
                    "POST",
                    f"/conversations/{conversation.id}/assistant-response/stream",
                    json={
                        "scope_id": str(scope.id),
                        "query_message_id": str(query_message.message.id),
                    },
                ) as response,
            ):
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])

            streamed = _parse_sse(body)
            assert [event["event"] for event in streamed] == ["start", "token", "token", "done"]
            await state.summary_episode_memory_scheduler.drain(timeout_seconds=5.0)

            assert await _summary_episode_rows(settings, conversation.id) == [
                {
                    "range_start": 13,
                    "range_end": 24,
                    "embedding_count": 1,
                }
            ]
            assert len(chat_generator.requests) == 2


@pytest.mark.asyncio
async def test_assistant_stream_persistence_failure_rolls_back_postgres_rows() -> None:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(
            database_url=_to_asyncpg_dsn(postgres.get_connection_url()),
            local_user_id=LOCAL_USER_ID,
        )
        await apply_migrations(settings=settings, migrations_dir=settings.migrations_dir)
        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))

        try:
            async with pool.acquire() as connection:
                await connection.execute("INSERT INTO users (id) VALUES ($1);", LOCAL_USER_ID)
            scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=LOCAL_USER_ID,
                    name="Persistence failure scope",
                    system_prompt="Keep memory scoped.",
                )
            )
            conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=LOCAL_USER_ID,
                    scope_id=scope.id,
                    title="Streaming failure",
                )
            )
            query_message = await service.append_message_with_episode(
                AppendMessageWithEpisodeRequest(
                    user_id=LOCAL_USER_ID,
                    conversation_id=conversation.id,
                    role="user",
                    content="Remember this before failing.",
                    token_count=5,
                )
            )
            retrieved = await service.retrieve_scoped_episodes(
                user_id=LOCAL_USER_ID,
                scope_id=scope.id,
                query=query_message.message.content,
                top_k=1,
            )
            poisoned_memories = tuple(
                replace(memory, embedding_model_id=999999) for memory in retrieved
            )
            orchestrator = AssistantOrchestrator(
                memory_service=_PoisonedRetrievalMemoryService(
                    delegate=service,
                    poisoned_memories=poisoned_memories,
                ),  # type: ignore[arg-type]
                chat_generator=FakeStreamingChatGenerator(
                    tokens=["partial"],
                    final=ChatStreamFinal(model="fake-stream", finish_reason="stop"),
                ),
            )

            prepared = await orchestrator.prepare_stream(
                AssistantGenerationRequest(
                    user_id=LOCAL_USER_ID,
                    scope_id=scope.id,
                    conversation_id=conversation.id,
                    query_message_id=query_message.message.id,
                    top_k=1,
                )
            )

            events = [event async for event in orchestrator.stream_prepared(prepared)]
            counts = await _assistant_stream_storage_counts(
                settings=settings,
                conversation_id=conversation.id,
            )

            assert events == [
                AssistantStreamStart(
                    used_memory_episode_ids=(poisoned_memories[0].id,),
                    chat_model="fake-stream",
                ),
                AssistantStreamToken(text="partial"),
                AssistantStreamError(
                    code="assistant_persistence_failed",
                    message="Assistant response persistence failed",
                ),
            ]
            assert all(not isinstance(event, AssistantStreamDone) for event in events)
            assert counts == {"assistant_messages": 0, "message_retrievals": 0}
        finally:
            await close_pool()


def test_assistant_stream_route_does_not_log_private_content_at_info_or_below(
    caplog: pytest.LogCaptureFixture,
) -> None:
    user_content_sentinel = "USER_QUERY_CONTENT_SENTINEL"
    memory_content_sentinel = "RETRIEVED_MEMORY_CONTENT_SENTINEL"
    assistant_content_sentinel = "ASSISTANT_RESPONSE_CONTENT_SENTINEL"
    conversation_id = uuid4()
    orchestrator = StreamingAssistantOrchestrator(
        events=[
            AssistantStreamStart(used_memory_episode_ids=(uuid4(),), chat_model="fake-stream"),
            AssistantStreamToken(text=assistant_content_sentinel),
            AssistantStreamDone(
                assistant_message=_message(uuid4(), conversation_id, assistant_content_sentinel),
                chat_model="fake-stream",
                finish_reason="stop",
                used_memory_episode_ids=(),
            ),
        ]
    )
    client = _client(orchestrator)

    with caplog.at_level(logging.DEBUG):
        response = client.post(
            f"/conversations/{conversation_id}/assistant-response/stream",
            json={
                "scope_id": str(uuid4()),
                "query_message_id": str(uuid4()),
            },
        )

    assert response.status_code == 200
    api_records = [
        record
        for record in caplog.records
        if (record.name == "smriti.api" or record.name.startswith("smriti.api."))
        and record.levelno <= logging.INFO
    ]
    for record in api_records:
        message = record.getMessage()
        assert user_content_sentinel not in message
        assert memory_content_sentinel not in message
        assert assistant_content_sentinel not in message


@pytest.mark.parametrize(
    ("error", "expected_detail"),
    [
        (InvalidAssistantRequestError, "Invalid assistant request"),
        (AssistantGenerationUnavailableError, "Local assistant generation unavailable"),
        (AssistantGenerationFailedError, "Assistant generation failed"),
    ],
)
def test_assistant_error_responses_do_not_echo_private_content(
    error: type[Exception],
    expected_detail: str,
) -> None:
    user_content_sentinel = "USER_QUERY_CONTENT_SENTINEL"
    memory_content_sentinel = "RETRIEVED_MEMORY_CONTENT_SENTINEL"
    assistant_content_sentinel = "ASSISTANT_RESPONSE_CONTENT_SENTINEL"
    private_content = (
        f"{user_content_sentinel} {memory_content_sentinel} {assistant_content_sentinel}"
    )
    response = _client(RaisingAssistantOrchestrator(error(private_content))).post(
        f"/conversations/{uuid4()}/assistant-response",
        json={"scope_id": str(uuid4()), "query_message_id": str(uuid4())},
    )

    assert response.json() == {"detail": expected_detail}
    for sentinel in [
        user_content_sentinel,
        memory_content_sentinel,
        assistant_content_sentinel,
    ]:
        assert sentinel not in response.text


def test_assistant_route_does_not_log_private_content_at_info_or_below(
    caplog: pytest.LogCaptureFixture,
) -> None:
    user_content_sentinel = "USER_QUERY_CONTENT_SENTINEL"
    memory_content_sentinel = "RETRIEVED_MEMORY_CONTENT_SENTINEL"
    assistant_content_sentinel = "ASSISTANT_RESPONSE_CONTENT_SENTINEL"
    response_content = (
        f"{user_content_sentinel} {memory_content_sentinel} {assistant_content_sentinel}"
    )
    conversation_id = uuid4()
    orchestrator = RecordingAssistantOrchestrator(
        result=AssistantGenerationResult(
            assistant_message=_message(uuid4(), conversation_id, response_content),
            chat_model="fake-chat",
            finish_reason="stop",
            used_memory_episode_ids=(),
        )
    )
    client = _client(orchestrator)

    with caplog.at_level(logging.DEBUG):
        response = client.post(
            f"/conversations/{conversation_id}/assistant-response",
            json={"scope_id": str(uuid4()), "query_message_id": str(uuid4())},
        )

    assert response.status_code == 201
    api_records = [
        record
        for record in caplog.records
        if (record.name == "smriti.api" or record.name.startswith("smriti.api."))
        and record.levelno <= logging.INFO
    ]
    for record in api_records:
        message = record.getMessage()
        assert user_content_sentinel not in message
        assert memory_content_sentinel not in message
        assert assistant_content_sentinel not in message


def test_assistant_route_module_stays_a_thin_adapter() -> None:
    forbidden_patterns = [
        "SELECT ",
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "asyncpg",
        "smriti.db",
        "MemoryService",
        "ChatGenerator",
        "OllamaChatGenerator",
        "FakeChatGenerator",
        "prompt_builder",
        "retrieve_scoped_episodes",
        "load_assistant_generation_context",
        "append_assistant_response_with_provenance",
        "record_used_memories",
        "chat_generator.generate",
        "generate_stream",
        "WebSocket",
        "websocket",
        "socket.io",
    ]
    source = inspect.getsource(assistant_routes)

    assert "AssistantOrchestrator" in source
    assert "AssistantGenerationRequest" in source
    for pattern in forbidden_patterns:
        assert pattern not in source, f"Forbidden route-layer pattern {pattern!r}"


@dataclass
class RecordingAssistantOrchestrator:
    result: AssistantGenerationResult
    requests: list[AssistantGenerationRequest] = field(default_factory=list)

    async def generate(self, request: AssistantGenerationRequest) -> AssistantGenerationResult:
        self.requests.append(request)
        return self.result


@dataclass
class RaisingAssistantOrchestrator:
    error: Exception

    async def generate(self, request: AssistantGenerationRequest) -> AssistantGenerationResult:
        _ = request
        raise self.error


@dataclass
class StreamingAssistantOrchestrator:
    events: list[AssistantStreamEvent] = field(default_factory=list)
    prepare_error: Exception | None = None
    prepare_requests: list[AssistantGenerationRequest] = field(default_factory=list)
    events_sent: int = 0

    async def generate(self, request: AssistantGenerationRequest) -> AssistantGenerationResult:
        _ = request
        raise AssertionError("non-streaming generate should not be called")

    async def prepare_stream(
        self,
        request: AssistantGenerationRequest,
    ) -> AssistantStreamPreparation:
        self.prepare_requests.append(request)
        if self.prepare_error is not None:
            raise self.prepare_error
        return AssistantStreamPreparation(
            request=request,
            chat_request=ChatRequest(messages=()),
            selected_memories=(),
            used_memory_episode_ids=(),
            chat_model="fake-stream",
        )

    async def stream_prepared(
        self,
        prepared: AssistantStreamPreparation,
    ) -> AsyncIterator[AssistantStreamEvent]:
        _ = prepared
        for event in self.events:
            self.events_sent += 1
            yield event


class _NoopSummaryEpisodeMemoryScheduler:
    def schedule(self, request) -> None:
        _ = request


@dataclass
class _DisconnectingRequest:
    disconnect_after_checks: int
    checks: int = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > self.disconnect_after_checks


@dataclass
class _BlockingStreamingChatGenerator:
    first_token_sent: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def model(self) -> str:
        return "blocking-stream"

    async def generate(self, request: ChatRequest) -> ChatResponse:
        _ = request
        return ChatResponse(content="partial after release", model=self.model, finish_reason="stop")

    async def generate_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        _ = request
        self.first_token_sent.set()
        yield ChatStreamToken(text="partial")
        await self.release.wait()
        yield ChatStreamFinal(model=self.model, finish_reason="stop")


@dataclass
class _PoisonedRetrievalMemoryService:
    delegate: MemoryService
    poisoned_memories: tuple[ScoredEpisode, ...]

    async def load_assistant_generation_context(self, request):
        return await self.delegate.load_assistant_generation_context(request)

    async def retrieve_scoped_episodes(
        self,
        user_id: UUID,
        scope_id: UUID,
        query: str,
        top_k: int,
    ) -> list[ScoredEpisode]:
        _ = (user_id, scope_id, query, top_k)
        return list(self.poisoned_memories)

    async def append_assistant_response_with_provenance(self, request):
        return await self.delegate.append_assistant_response_with_provenance(request)


class _StreamingASGITransport(httpx.AsyncBaseTransport):
    def __init__(self, app) -> None:
        self._app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request_body = await request.aread()
        request_complete = False
        response_started = asyncio.Event()
        response_complete = asyncio.Event()
        disconnect = asyncio.Event()
        body_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        status_code: int | None = None
        response_headers: list[tuple[bytes, bytes]] = []

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(key.lower(), value) for key, value in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }

        async def receive() -> dict[str, object]:
            nonlocal request_complete
            if not request_complete:
                request_complete = True
                return {"type": "http.request", "body": request_body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = cast(list[tuple[bytes, bytes]], message.get("headers", []))
                response_started.set()
                return

            if message["type"] == "http.response.body":
                body = cast(bytes, message.get("body", b""))
                more_body = bool(message.get("more_body", False))
                if body and request.method != "HEAD":
                    await body_queue.put(body)
                if not more_body:
                    response_complete.set()
                    await body_queue.put(None)

        async def run_app() -> None:
            try:
                await self._app(scope, receive, send)
            finally:
                if response_started.is_set() and not response_complete.is_set():
                    response_complete.set()
                    await body_queue.put(None)

        app_task = asyncio.create_task(run_app())
        await response_started.wait()
        assert status_code is not None
        return httpx.Response(
            status_code,
            headers=response_headers,
            stream=_StreamingASGIByteStream(
                body_queue=body_queue,
                disconnect=disconnect,
                response_complete=response_complete,
                app_task=app_task,
            ),
            request=request,
        )


@dataclass
class _StreamingASGIByteStream(httpx.AsyncByteStream):
    body_queue: asyncio.Queue[bytes | None]
    disconnect: asyncio.Event
    response_complete: asyncio.Event
    app_task: asyncio.Task[None]

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self.body_queue.get()
            if chunk is None:
                break
            yield chunk

    async def aclose(self) -> None:
        self.disconnect.set()
        if not self.response_complete.is_set():
            self.app_task.cancel()
        with suppress(asyncio.CancelledError):
            await self.app_task


def _client(
    orchestrator: object,
) -> TestClient:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    app.dependency_overrides[get_assistant_orchestrator] = lambda: cast(
        AssistantOrchestrator,
        orchestrator,
    )
    app.dependency_overrides[get_current_local_user_id] = lambda: LOCAL_USER_ID
    app.dependency_overrides[get_summary_episode_memory_scheduler] = lambda: (
        _NoopSummaryEpisodeMemoryScheduler()
    )
    return TestClient(app, raise_server_exceptions=False)


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


async def _create_scope(client: httpx.AsyncClient) -> dict[str, str]:
    response = await client.post(
        "/scopes",
        json={"name": "Streaming Scope", "system_prompt": "Keep memory scoped."},
    )
    assert response.status_code == 201
    return response.json()


async def _create_conversation(client: httpx.AsyncClient, scope_id: UUID) -> dict[str, str]:
    response = await client.post(
        "/conversations",
        json={"scope_id": str(scope_id), "title": "Streaming Conversation"},
    )
    assert response.status_code == 201
    return response.json()


async def _create_user_message(
    client: httpx.AsyncClient,
    conversation_id: UUID,
) -> dict[str, str]:
    response = await client.post(
        f"/conversations/{conversation_id}/messages",
        json={"role": "user", "content": "Please answer from memory.", "token_count": 5},
    )
    assert response.status_code == 201
    return response.json()


async def _assistant_stream_storage_counts(
    settings: Settings,
    conversation_id: UUID,
) -> dict[str, int]:
    connection = await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )
    try:
        row = await connection.fetchrow(
            """
            SELECT
                (
                    SELECT COUNT(*)
                    FROM messages
                    WHERE conversation_id = $1
                      AND role = 'assistant'
                ) AS assistant_messages,
                (
                    SELECT COUNT(*)
                    FROM message_retrievals
                    WHERE query_conversation_id = $1
                ) AS message_retrievals;
            """,
            conversation_id,
        )
    finally:
        await connection.close()

    assert row is not None
    return {
        "assistant_messages": int(row["assistant_messages"]),
        "message_retrievals": int(row["message_retrievals"]),
    }


async def _summary_episode_rows(
    settings: Settings,
    conversation_id: UUID,
) -> list[dict[str, int]]:
    connection = await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )
    try:
        rows = await connection.fetch(
            """
            SELECT
                episodes.range_start,
                episodes.range_end,
                COUNT(embeddings_768.episode_id) AS embedding_count
            FROM episodes
            LEFT JOIN embeddings_768
                ON embeddings_768.episode_id = episodes.id
            WHERE episodes.conversation_id = $1
              AND episodes.kind = 'summary'
            GROUP BY episodes.id
            ORDER BY episodes.range_start ASC;
            """,
            conversation_id,
        )
    finally:
        await connection.close()

    return [
        {
            "range_start": int(row["range_start"]),
            "range_end": int(row["range_end"]),
            "embedding_count": int(row["embedding_count"]),
        }
        for row in rows
    ]


def _parse_sse(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for block in body.strip().split("\n\n"):
        if not block:
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_lines.append(line.removeprefix("data: "))
        assert event_name is not None
        events.append({"event": event_name, "data": json.loads("".join(data_lines))})
    return events


def _message(message_id: UUID, conversation_id: UUID, content: str) -> MessageRecord:
    return MessageRecord(
        id=message_id,
        conversation_id=conversation_id,
        position=2,
        role="assistant",
        content=content,
        token_count=max(1, len(content.split())),
        created_at=FIXED_NOW,
    )
