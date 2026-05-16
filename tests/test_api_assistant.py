from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from smriti.api import create_app
from smriti.api.dependencies import get_assistant_orchestrator, get_current_local_user_id
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
from smriti.chat import ChatRequest
from smriti.config import Settings
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    InvalidProvenanceTargetError,
    MemoryServiceError,
    MessageRecord,
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


@dataclass
class _DisconnectingRequest:
    disconnect_after_checks: int
    checks: int = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > self.disconnect_after_checks


def _client(
    orchestrator: object,
) -> TestClient:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    app.dependency_overrides[get_assistant_orchestrator] = lambda: cast(
        AssistantOrchestrator,
        orchestrator,
    )
    app.dependency_overrides[get_current_local_user_id] = lambda: LOCAL_USER_ID
    return TestClient(app, raise_server_exceptions=False)


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
