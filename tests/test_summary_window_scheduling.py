"""DB-free tests for rolling summary window scheduling.

These tests use fakes only: they cover window math, both-role transcripts,
catch-up across multiple windows, duplicate protection, failure observability,
and survival of summary work across the request/stream lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from smriti.api.app import create_app
from smriti.api.dependencies import (
    get_assistant_orchestrator,
    get_current_local_user_id,
    get_summary_episode_memory_scheduler,
)
from smriti.api.routes import assistant as assistant_routes
from smriti.assistant import (
    AssistantGenerationRequest,
    AssistantGenerationResult,
    AssistantOrchestrator,
    AssistantStreamDone,
    AssistantStreamEvent,
    AssistantStreamPreparation,
    AssistantStreamStart,
)
from smriti.chat import ChatRequest, FakeChatGenerator
from smriti.config import Settings
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    CreateSummaryEpisodeRequest,
    MemoryService,
    MessageRecord,
    MessageRole,
    SummaryEpisodeMemoryScheduler,
    SummaryEpisodeMemoryScheduleRequest,
)
from smriti.memory.errors import SummaryEpisodeMemoryError
from smriti.memory.models import SummaryEpisodeRecord
from smriti.memory.service import (
    _next_summary_window,
    _summary_chat_request,
    _SummaryWindowCandidate,
)
from smriti.memory.summary_tasks import (
    SUMMARY_EPISODE_MAX_WINDOWS_PER_TASK,
)
from smriti.memory.summary_tasks import (
    logger as summary_task_logger,
)

LOCAL_USER_ID = UUID("00000000-0000-4000-8000-000000000001")


# ---------------------------------------------------------------------------
# Window math: fixed non-overlapping windows with catch-up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message_count", "covered_summary_end", "expected"),
    [
        # An eligible completed first window is detected.
        (12, 0, (1, 12)),
        # Incomplete first window is not summarized.
        (11, 0, None),
        (0, 0, None),
        # A second window is detected after 24 messages.
        (24, 12, (13, 24)),
        # An in-progress second window is not summarized early.
        (23, 12, None),
        # Duplicate windows are not recreated once covered.
        (12, 12, None),
        (24, 24, None),
        # Missed windows are caught up even when the count is not an exact
        # multiple (feature enabled late, transient failure, or parity shift
        # after a failed assistant turn).
        (13, 0, (1, 12)),
        (24, 0, (1, 12)),
        (26, 12, (13, 24)),
    ],
)
def test_next_summary_window_fixed_windows_and_catch_up(
    message_count: int,
    covered_summary_end: int,
    expected: tuple[int, int] | None,
) -> None:
    assert (
        _next_summary_window(
            message_count=message_count,
            covered_summary_end=covered_summary_end,
            window_messages=12,
        )
        == expected
    )


def test_summary_window_transcript_includes_both_roles() -> None:
    conversation_id = uuid4()
    messages = tuple(
        _message(conversation_id, position=position, role="user" if position % 2 else "assistant")
        for position in range(1, 13)
    )
    candidate = _SummaryWindowCandidate(
        message_count=12,
        range_start=1,
        range_end=12,
        messages=messages,
    )

    chat_request = _summary_chat_request(candidate)
    transcript = chat_request.messages[1].content

    assert "position=1 role=user" in transcript
    assert "position=2 role=assistant" in transcript
    assert "position=12 role=assistant" in transcript
    assert transcript.count("role=user") == 6
    assert transcript.count("role=assistant") == 6


# ---------------------------------------------------------------------------
# Scheduler behavior with fake memory services
# ---------------------------------------------------------------------------


@dataclass
class _CatchUpSummaryMemoryService:
    """Fake service that reports two uncovered windows, then none."""

    records: list[SummaryEpisodeRecord]
    embedding_model_id: str = "fake-embedder"
    requests: list[CreateSummaryEpisodeRequest] = field(default_factory=list)

    async def create_summary_episode_for_next_uncovered_window(
        self,
        request: CreateSummaryEpisodeRequest,
        chat_generator: FakeChatGenerator,
    ) -> SummaryEpisodeRecord | None:
        _ = chat_generator
        self.requests.append(request)
        if self.records:
            return self.records.pop(0)
        return None


@dataclass
class _FailingSummaryMemoryService:
    error: Exception
    embedding_model_id: str = "fake-embedder"
    calls: int = 0

    async def create_summary_episode_for_next_uncovered_window(
        self,
        request: CreateSummaryEpisodeRequest,
        chat_generator: FakeChatGenerator,
    ) -> SummaryEpisodeRecord | None:
        _ = (request, chat_generator)
        self.calls += 1
        raise self.error


async def test_scheduler_creates_all_uncovered_windows_and_logs_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conversation_id = uuid4()
    memory_service = _CatchUpSummaryMemoryService(
        records=[
            _summary_record(conversation_id, 1, 12),
            _summary_record(conversation_id, 13, 24),
        ]
    )
    scheduler = _scheduler(memory_service)

    with caplog.at_level(logging.INFO, logger=summary_task_logger.name):
        task = scheduler.schedule(_schedule_request(conversation_id))
        assert task is not None
        await scheduler.drain()

    # Two windows created, then one final call that found nothing uncovered.
    assert len(memory_service.requests) == 3
    assert all(request.window_messages == 12 for request in memory_service.requests)
    created = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "summary_episode_memory_created"
    ]
    assert [(record.range_start, record.range_end) for record in created] == [(1, 12), (13, 24)]
    for record in created:
        assert record.conversation_id == conversation_id
        assert record.summary_model == "fake-chat-generator"
        assert record.embedding_model == "fake-embedder"
        assert record.status == "created"
        assert isinstance(record.elapsed_ms, int)
        rendered = record.getMessage()
        assert rendered.startswith("summary_episode_memory_created ")
        assert f"conversation_id={conversation_id}" in rendered


async def test_scheduler_stops_when_no_uncovered_window_exists() -> None:
    memory_service = _CatchUpSummaryMemoryService(records=[])
    scheduler = _scheduler(memory_service)

    task = scheduler.schedule(_schedule_request(uuid4()))
    assert task is not None
    await scheduler.drain()

    assert len(memory_service.requests) == 1


async def test_scheduler_caps_windows_per_task() -> None:
    conversation_id = uuid4()
    memory_service = _CatchUpSummaryMemoryService(
        records=[
            _summary_record(conversation_id, start, start + 11)
            for start in range(1, 12 * (SUMMARY_EPISODE_MAX_WINDOWS_PER_TASK + 3), 12)
        ]
    )
    scheduler = _scheduler(memory_service)

    task = scheduler.schedule(_schedule_request(conversation_id))
    assert task is not None
    await scheduler.drain()

    assert len(memory_service.requests) == SUMMARY_EPISODE_MAX_WINDOWS_PER_TASK


async def test_scheduler_failure_is_logged_not_swallowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conversation_id = uuid4()
    error = SummaryEpisodeMemoryError(
        "Summary episode memory failed",
        failure_step="summary_generation",
        user_id=LOCAL_USER_ID,
        scope_id=uuid4(),
        conversation_id=conversation_id,
        range_start=1,
        range_end=12,
        message_count=12,
        summary_model="fake-chat-generator",
        embedding_model="fake-embedder",
        exception_type="ChatResponseError",
    )
    memory_service = _FailingSummaryMemoryService(error=error)
    scheduler = _scheduler(memory_service)

    with caplog.at_level(logging.ERROR, logger=summary_task_logger.name):
        task = scheduler.schedule(_schedule_request(conversation_id))
        assert task is not None
        await scheduler.drain()

    failures = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "summary_episode_memory_failed"
    ]
    assert len(failures) == 1
    assert failures[0].failure_step == "summary_generation"
    assert failures[0].conversation_id == conversation_id
    assert failures[0].exception_type == "ChatResponseError"


async def test_scheduler_unexpected_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conversation_id = uuid4()
    memory_service = _FailingSummaryMemoryService(error=RuntimeError("boom"))
    scheduler = _scheduler(memory_service)

    with caplog.at_level(logging.ERROR, logger=summary_task_logger.name):
        task = scheduler.schedule(_schedule_request(conversation_id))
        assert task is not None
        await scheduler.drain()

    failures = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "summary_episode_memory_failed"
    ]
    assert len(failures) == 1
    assert failures[0].failure_step == "unexpected"
    assert failures[0].exception_type == "RuntimeError"


def test_scheduler_disabled_skips_scheduling() -> None:
    memory_service = _CatchUpSummaryMemoryService(records=[])
    scheduler = _scheduler(memory_service, enabled=False)

    assert scheduler.schedule(_schedule_request(uuid4())) is None
    assert memory_service.requests == []


# ---------------------------------------------------------------------------
# Request / stream lifecycle
# ---------------------------------------------------------------------------


class _NeverDisconnectingRequest:
    async def is_disconnected(self) -> bool:
        return False


@dataclass
class _RecordingSummaryScheduler:
    requests: list[SummaryEpisodeMemoryScheduleRequest] = field(default_factory=list)

    def schedule(self, request: SummaryEpisodeMemoryScheduleRequest) -> None:
        self.requests.append(request)


async def test_sse_summary_scheduling_survives_client_disconnect_after_done() -> None:
    """Closing the SSE generator right after the done chunk must not lose scheduling."""

    conversation_id = uuid4()
    schedule_request = _schedule_request(conversation_id)
    scheduler = _RecordingSummaryScheduler()

    events = _stream_events(conversation_id)
    sse = assistant_routes._stream_sse_events(
        request=cast(Request, _NeverDisconnectingRequest()),
        events=events,
        summary_episode_memory_scheduler=cast(SummaryEpisodeMemoryScheduler, scheduler),
        summary_episode_memory_request=schedule_request,
    )

    async for chunk in sse:
        if "event: done" in chunk:
            # Simulate the client tearing the stream down immediately after
            # receiving the terminal event, before the generator resumes.
            await sse.aclose()
            break

    assert scheduler.requests == [schedule_request]


async def test_scheduled_summary_task_outlives_stream_generator() -> None:
    """The background task must keep running after the SSE generator is closed."""

    conversation_id = uuid4()
    blocking_service = _BlockingSummaryMemoryService()
    scheduler = _scheduler(blocking_service)

    sse = assistant_routes._stream_sse_events(
        request=cast(Request, _NeverDisconnectingRequest()),
        events=_stream_events(conversation_id),
        summary_episode_memory_scheduler=scheduler,
        summary_episode_memory_request=_schedule_request(conversation_id),
    )
    async for chunk in sse:
        if "event: done" in chunk:
            await sse.aclose()
            break

    await asyncio.wait_for(blocking_service.started.wait(), timeout=1.0)
    assert scheduler.pending_count == 1

    blocking_service.release.set()
    await scheduler.drain()
    assert scheduler.pending_count == 0
    assert blocking_service.completed


@dataclass
class _BlockingSummaryMemoryService:
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    completed: bool = False
    embedding_model_id: str = "fake-embedder"

    async def create_summary_episode_for_next_uncovered_window(
        self,
        request: CreateSummaryEpisodeRequest,
        chat_generator: FakeChatGenerator,
    ) -> SummaryEpisodeRecord | None:
        _ = (request, chat_generator)
        self.started.set()
        await self.release.wait()
        self.completed = True
        return None


# ---------------------------------------------------------------------------
# API routes schedule summary work after a persisted assistant response
# ---------------------------------------------------------------------------


@dataclass
class _FakeAssistantOrchestrator:
    conversation_id: UUID

    async def generate(self, request: AssistantGenerationRequest) -> AssistantGenerationResult:
        return AssistantGenerationResult(
            assistant_message=_message(self.conversation_id, position=12, role="assistant"),
            chat_model="fake-stream",
            finish_reason="stop",
            used_memory_episode_ids=(),
        )

    async def prepare_stream(
        self,
        request: AssistantGenerationRequest,
    ) -> AssistantStreamPreparation:
        return AssistantStreamPreparation(
            request=request,
            chat_request=ChatRequest(messages=()),
            selected_memories=(),
            used_memory_episode_ids=(),
            chat_model="fake-stream",
        )

    async def stream_prepared(self, prepared: AssistantStreamPreparation):
        _ = prepared
        for event in [
            AssistantStreamStart(used_memory_episode_ids=(), chat_model="fake-stream"),
            AssistantStreamDone(
                assistant_message=_message(self.conversation_id, position=12, role="assistant"),
                chat_model="fake-stream",
                finish_reason="stop",
                used_memory_episode_ids=(),
            ),
        ]:
            yield event


def _route_client(
    conversation_id: UUID,
    scheduler: _RecordingSummaryScheduler,
) -> TestClient:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    app.dependency_overrides[get_assistant_orchestrator] = lambda: cast(
        AssistantOrchestrator,
        _FakeAssistantOrchestrator(conversation_id=conversation_id),
    )
    app.dependency_overrides[get_current_local_user_id] = lambda: LOCAL_USER_ID
    app.dependency_overrides[get_summary_episode_memory_scheduler] = lambda: cast(
        SummaryEpisodeMemoryScheduler,
        scheduler,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_streaming_route_schedules_summary_after_done() -> None:
    conversation_id = uuid4()
    scope_id = uuid4()
    scheduler = _RecordingSummaryScheduler()
    client = _route_client(conversation_id, scheduler)

    response = client.post(
        f"/conversations/{conversation_id}/assistant-response/stream",
        json={"scope_id": str(scope_id), "query_message_id": str(uuid4())},
    )

    assert response.status_code == 200
    assert "event: done" in response.text
    assert scheduler.requests == [
        SummaryEpisodeMemoryScheduleRequest(
            user_id=LOCAL_USER_ID,
            scope_id=scope_id,
            conversation_id=conversation_id,
        )
    ]


def test_non_streaming_route_schedules_summary_after_completion() -> None:
    conversation_id = uuid4()
    scope_id = uuid4()
    scheduler = _RecordingSummaryScheduler()
    client = _route_client(conversation_id, scheduler)

    response = client.post(
        f"/conversations/{conversation_id}/assistant-response",
        json={"scope_id": str(scope_id), "query_message_id": str(uuid4())},
    )

    assert response.status_code == 201
    assert scheduler.requests == [
        SummaryEpisodeMemoryScheduleRequest(
            user_id=LOCAL_USER_ID,
            scope_id=scope_id,
            conversation_id=conversation_id,
        )
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scheduler(memory_service: object, *, enabled: bool = True) -> SummaryEpisodeMemoryScheduler:
    return SummaryEpisodeMemoryScheduler(
        memory_service=cast(MemoryService, memory_service),
        chat_generator=FakeChatGenerator(),
        enabled=enabled,
        window_messages=12,
    )


def _schedule_request(conversation_id: UUID) -> SummaryEpisodeMemoryScheduleRequest:
    return SummaryEpisodeMemoryScheduleRequest(
        user_id=LOCAL_USER_ID,
        scope_id=uuid4(),
        conversation_id=conversation_id,
    )


def _message(conversation_id: UUID, position: int, role: str) -> MessageRecord:
    return MessageRecord(
        id=uuid4(),
        conversation_id=conversation_id,
        position=position,
        role=cast(MessageRole, role),
        content=f"window message {position}",
        token_count=4,
        created_at=datetime(2026, 1, 1, 12, 0, position, tzinfo=timezone.utc),  # noqa: UP017
    )


def _summary_record(
    conversation_id: UUID, range_start: int, range_end: int
) -> SummaryEpisodeRecord:
    return SummaryEpisodeRecord(
        id=uuid4(),
        conversation_id=conversation_id,
        scope_id=uuid4(),
        range_start=range_start,
        range_end=range_end,
        content="summary content",
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),  # noqa: UP017
        embedding_model_id=1,
    )


async def _stream_events(conversation_id: UUID):
    yield AssistantStreamStart(used_memory_episode_ids=(), chat_model="fake-stream")
    yield AssistantStreamDone(
        assistant_message=_message(conversation_id, position=12, role="assistant"),
        chat_model="fake-stream",
        finish_reason="stop",
        used_memory_episode_ids=(),
    )


_ = AssistantStreamEvent  # re-exported type used implicitly by generators above
