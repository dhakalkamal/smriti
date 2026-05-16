from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest
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
    InvalidAssistantRequestError,
)
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


def _client(
    orchestrator: RecordingAssistantOrchestrator | RaisingAssistantOrchestrator,
) -> TestClient:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    app.dependency_overrides[get_assistant_orchestrator] = lambda: cast(
        AssistantOrchestrator,
        orchestrator,
    )
    app.dependency_overrides[get_current_local_user_id] = lambda: LOCAL_USER_ID
    return TestClient(app, raise_server_exceptions=False)


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
