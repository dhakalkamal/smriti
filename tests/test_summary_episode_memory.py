from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest
from fastapi import Request
from testcontainers.postgres import PostgresContainer

from smriti.api.routes import assistant as assistant_routes
from smriti.assistant import (
    AssistantStreamDone,
    AssistantStreamEvent,
    AssistantStreamStart,
)
from smriti.chat import (
    ChatResponse,
    ChatResponseError,
    FakeChatGenerator,
)
from smriti.config import Settings
from smriti.db.client import close_pool, get_pool
from smriti.db.migrate import apply_migrations
from smriti.embeddings import EmbeddingConfigurationError, EmbeddingVector, FakeEmbedder
from smriti.memory import (
    AppendMessageRequest,
    CreateConversationRequest,
    CreateScopeRequest,
    CreateSummaryEpisodeRequest,
    DeleteConversationRequest,
    MemoryService,
    SummaryEpisodeMemoryScheduler,
    SummaryEpisodeMemoryScheduleRequest,
)
from smriti.memory.summary_tasks import logger as summary_task_logger

FIXED_USER_ID = UUID("77777777-7777-4777-8777-777777777777")
MESSAGE_CONTENT_SENTINEL = "SUMMARY_MESSAGE_CONTENT_SENTINEL"
SUMMARY_CONTENT_SENTINEL = "SUMMARY_OUTPUT_CONTENT_SENTINEL"
FAILURE_CONTENT_SENTINEL = "SUMMARY_FAILURE_CONTENT_SENTINEL"


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


@pytest.mark.asyncio
async def test_summary_episode_memory_creates_only_complete_fixed_windows() -> None:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(database_url=_to_asyncpg_dsn(postgres.get_connection_url()))
        await apply_migrations(settings=settings, migrations_dir=_migrations_dir())
        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))
        chat_generator = FakeChatGenerator(
            responses=[
                ChatResponse(
                    content="alpha complete summary memory",
                    model="fake-summary-model",
                    finish_reason="stop",
                ),
                ChatResponse(
                    content="beta complete summary memory",
                    model="fake-summary-model",
                    finish_reason="stop",
                ),
            ]
        )

        try:
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service=service,
                pool=pool,
            )

            await _append_messages(service, user_id, scope_id, conversation_id, start=1, end=8)
            assert (
                await _create_summary(
                    service=service,
                    chat_generator=chat_generator,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                )
                is None
            )
            assert await _summary_ranges(pool, conversation_id) == []

            await _append_messages(service, user_id, scope_id, conversation_id, start=9, end=12)
            first_summary = await _create_summary(
                service=service,
                chat_generator=chat_generator,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
            )
            assert first_summary is not None
            assert (first_summary.range_start, first_summary.range_end) == (1, 12)
            assert await _summary_ranges(pool, conversation_id) == [(1, 12)]

            assert (
                await _create_summary(
                    service=service,
                    chat_generator=chat_generator,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                )
                is None
            )
            assert len(chat_generator.requests) == 1

            await _append_messages(service, user_id, scope_id, conversation_id, start=13, end=21)
            assert (
                await _create_summary(
                    service=service,
                    chat_generator=chat_generator,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                )
                is None
            )
            assert await _summary_ranges(pool, conversation_id) == [(1, 12)]

            await _append_messages(service, user_id, scope_id, conversation_id, start=22, end=24)
            second_summary = await _create_summary(
                service=service,
                chat_generator=chat_generator,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
            )
            assert second_summary is not None
            assert (second_summary.range_start, second_summary.range_end) == (13, 24)

            await _append_messages(service, user_id, scope_id, conversation_id, start=25, end=32)
            assert (
                await _create_summary(
                    service=service,
                    chat_generator=chat_generator,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                )
                is None
            )

            rows = await _summary_rows(pool, conversation_id)
            assert [(row["range_start"], row["range_end"]) for row in rows] == [
                (1, 12),
                (13, 24),
            ]
            assert rows[0]["range_end"] < rows[1]["range_start"]
            assert all(row["kind"] == "summary" for row in rows)
            assert all(row["message_id"] is None for row in rows)
            assert all(row["conversation_id"] == conversation_id for row in rows)
            assert all(row["scope_id"] == scope_id for row in rows)
            assert all(row["embedding_count"] == 1 for row in rows)
            assert all(row["embedding_dimensions"] == 768 for row in rows)

            transcript = chat_generator.requests[0].messages[1].content
            assert "position=1 role=user" in transcript
            assert "position=12 role=assistant" in transcript
            assert "position=13" not in transcript

            retrieved = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query="alpha complete summary memory",
                top_k=3,
            )
            assert any(
                episode.kind == "summary"
                and episode.message_id is None
                and episode.range_start == 1
                and episode.range_end == 12
                for episode in retrieved
            )

            await service.delete_conversation(
                DeleteConversationRequest(user_id=user_id, conversation_id=conversation_id)
            )
            assert await _summary_storage_count(pool, conversation_id) == {
                "summary_episodes": 0,
                "summary_embeddings": 0,
            }
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_summary_episode_memory_refuses_overlapping_summary_ranges() -> None:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(database_url=_to_asyncpg_dsn(postgres.get_connection_url()))
        await apply_migrations(settings=settings, migrations_dir=_migrations_dir())
        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))
        chat_generator = FakeChatGenerator(
            response=ChatResponse(
                content="should not be generated",
                model="fake-summary-model",
                finish_reason="stop",
            )
        )

        try:
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service=service,
                pool=pool,
            )
            await _append_messages(service, user_id, scope_id, conversation_id, start=1, end=12)
            await _seed_summary_episode(
                pool=pool,
                service=service,
                conversation_id=conversation_id,
                scope_id=scope_id,
                range_start=6,
                range_end=17,
                content="pre-existing overlapping summary",
            )

            result = await _create_summary(
                service=service,
                chat_generator=chat_generator,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
            )

            assert result is None
            assert chat_generator.requests == []
            assert await _summary_ranges(pool, conversation_id) == [(6, 17)]
        finally:
            await close_pool()


@pytest.mark.parametrize(
    (
        "failure_mode",
        "expected_step",
        "expected_exception_type",
        "expected_embedding_model",
        "expected_exception_message",
    ),
    [
        (
            "summary_generation",
            "summary_generation",
            "ChatResponseError",
            "nomic-embed-text",
            "fake summary generation failure",
        ),
        (
            "embedding_generation",
            "embedding_generation",
            "EmbeddingConfigurationError",
            "nomic-embed-text",
            "fake embedding generation failure",
        ),
        (
            "summary_episode_write",
            "summary_episode_write",
            "EmbeddingModelNotFoundError",
            "missing-summary-model",
            "Embedding model is not registered for 768d storage",
        ),
    ],
)
@pytest.mark.asyncio
async def test_summary_episode_background_failures_log_without_content_and_leave_no_rows(
    failure_mode: str,
    expected_step: str,
    expected_exception_type: str,
    expected_embedding_model: str,
    expected_exception_message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(database_url=_to_asyncpg_dsn(postgres.get_connection_url()))
        await apply_migrations(settings=settings, migrations_dir=_migrations_dir())
        pool = await get_pool(settings)
        service = _failure_service(pool, failure_mode)
        chat_generator = _failure_chat_generator(failure_mode)
        scheduler = SummaryEpisodeMemoryScheduler(
            memory_service=service,
            chat_generator=chat_generator,
            enabled=True,
            window_messages=12,
        )

        try:
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service=MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768)),
                pool=pool,
            )
            await _append_messages(
                service=MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768)),
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                start=1,
                end=12,
                content_prefix=MESSAGE_CONTENT_SENTINEL,
            )

            with caplog.at_level(logging.ERROR, logger=summary_task_logger.name):
                task = scheduler.schedule(
                    SummaryEpisodeMemoryScheduleRequest(
                        user_id=user_id,
                        scope_id=scope_id,
                        conversation_id=conversation_id,
                    )
                )
                assert task is not None
                await scheduler.drain()

            assert scheduler.pending_count == 0
            assert await _summary_storage_count(pool, conversation_id) == {
                "summary_episodes": 0,
                "summary_embeddings": 0,
            }
            records = [
                record
                for record in caplog.records
                if getattr(record, "event", None) == "summary_episode_memory_failed"
            ]
            assert len(records) == 1
            record = records[0]
            assert record.failure_step == expected_step
            assert record.user_id == user_id
            assert record.scope_id == scope_id
            assert record.conversation_id == conversation_id
            assert record.range_start == 1
            assert record.range_end == 12
            assert record.message_count == 12
            assert record.embedding_model == expected_embedding_model
            assert not hasattr(record, "embedding_model_id")
            assert record.exception_type == expected_exception_type
            assert record.exception_message == expected_exception_message
            assert record.exc_info is None
            rendered_log = record.getMessage()
            assert rendered_log.startswith("summary_episode_memory_failed ")
            assert f"failure_step={expected_step}" in rendered_log
            assert f"conversation_id={conversation_id}" in rendered_log
            assert "range_start=1" in rendered_log
            assert "range_end=12" in rendered_log
            assert "message_count=12" in rendered_log
            assert "summary_model=fake-summary-model" in rendered_log
            assert f"embedding_model={expected_embedding_model}" in rendered_log
            assert f"exception_type={expected_exception_type}" in rendered_log
            assert f"exception_message={expected_exception_message}" in rendered_log
            assert MESSAGE_CONTENT_SENTINEL not in rendered_log
            assert SUMMARY_CONTENT_SENTINEL not in rendered_log
            assert FAILURE_CONTENT_SENTINEL not in rendered_log
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_summary_scheduler_retains_and_removes_background_tasks() -> None:
    memory_service = _BlockingSummaryMemoryService()
    scheduler = SummaryEpisodeMemoryScheduler(
        memory_service=cast(MemoryService, memory_service),
        chat_generator=FakeChatGenerator(),
        enabled=True,
        window_messages=12,
    )

    task = scheduler.schedule(
        SummaryEpisodeMemoryScheduleRequest(
            user_id=uuid4(),
            scope_id=uuid4(),
            conversation_id=uuid4(),
        )
    )

    assert task is not None
    await memory_service.started.wait()
    assert scheduler.pending_count == 1

    memory_service.release.set()
    await scheduler.drain()

    assert scheduler.pending_count == 0


@pytest.mark.asyncio
async def test_summary_scheduler_drain_timeout_cancels_and_logs_pending_tasks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = _BlockingSummaryMemoryService()
    scheduler = SummaryEpisodeMemoryScheduler(
        memory_service=cast(MemoryService, memory_service),
        chat_generator=FakeChatGenerator(),
        enabled=True,
        window_messages=12,
    )
    request = SummaryEpisodeMemoryScheduleRequest(
        user_id=uuid4(),
        scope_id=uuid4(),
        conversation_id=uuid4(),
    )

    task = scheduler.schedule(request)

    assert task is not None
    await memory_service.started.wait()

    with caplog.at_level(logging.ERROR, logger=summary_task_logger.name):
        await scheduler.drain(timeout_seconds=0.01)

    assert scheduler.pending_count == 0
    assert task.cancelled()
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "summary_episode_memory_failed"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.failure_step == "shutdown_drain_timeout"
    assert record.user_id == request.user_id
    assert record.scope_id == request.scope_id
    assert record.conversation_id == request.conversation_id
    assert record.range_start is None
    assert record.range_end is None
    assert record.message_count is None
    assert record.embedding_model == "fake-embedder"
    assert record.exception_type == "CancelledError"
    assert record.exception_message == "summary task cancelled during shutdown drain"
    assert record.exc_info is None
    rendered_log = record.getMessage()
    assert rendered_log.startswith("summary_episode_memory_failed ")
    assert "failure_step=shutdown_drain_timeout" in rendered_log
    assert f"conversation_id={request.conversation_id}" in rendered_log
    assert "range_start=null" in rendered_log
    assert "range_end=null" in rendered_log
    assert "message_count=null" in rendered_log
    assert "summary_model=fake-chat-generator" in rendered_log
    assert "embedding_model=fake-embedder" in rendered_log
    assert "exception_type=CancelledError" in rendered_log
    assert "exception_message=summary task cancelled during shutdown drain" in rendered_log


@pytest.mark.asyncio
async def test_sse_schedules_summary_work_before_done_event_is_yielded() -> None:
    conversation_id = uuid4()
    schedule_request = SummaryEpisodeMemoryScheduleRequest(
        user_id=uuid4(),
        scope_id=uuid4(),
        conversation_id=conversation_id,
    )
    scheduler = _RecordingSummaryScheduler()

    chunks: list[str] = []
    async for chunk in assistant_routes._stream_sse_events(
        request=cast(Request, _NeverDisconnectingRequest()),
        events=_assistant_done_events(conversation_id),
        summary_episode_memory_scheduler=cast(SummaryEpisodeMemoryScheduler, scheduler),
        summary_episode_memory_request=schedule_request,
    ):
        chunks.append(chunk)
        if "event: done" in chunk:
            assert scheduler.requests == [schedule_request]

    assert [event["event"] for event in _parse_sse("".join(chunks))] == ["start", "done"]
    assert scheduler.requests == [schedule_request]


def _migrations_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"


async def _create_user_scope_conversation(
    service: MemoryService,
    pool: asyncpg.Pool,
) -> tuple[UUID, UUID, UUID]:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO users (id)
            VALUES ($1)
            ON CONFLICT (id) DO NOTHING;
            """,
            FIXED_USER_ID,
        )
    scope = await service.create_scope(
        CreateScopeRequest(
            user_id=FIXED_USER_ID,
            name=f"Summary Scope {uuid4()}",
            system_prompt="Keep summary memory scoped.",
        )
    )
    conversation = await service.create_conversation(
        CreateConversationRequest(
            user_id=FIXED_USER_ID,
            scope_id=scope.id,
            title="Summary conversation",
        )
    )
    return FIXED_USER_ID, scope.id, conversation.id


async def _append_messages(
    service: MemoryService,
    user_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
    start: int,
    end: int,
    content_prefix: str = "window message",
) -> None:
    for position in range(start, end + 1):
        role = "user" if position % 2 == 1 else "assistant"
        await service.append_message(
            AppendMessageRequest(
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role=role,
                content=f"{content_prefix} {position}",
                token_count=4,
            )
        )


async def _create_summary(
    service: MemoryService,
    chat_generator: FakeChatGenerator,
    user_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
):
    return await service.create_summary_episode_for_next_uncovered_window(
        CreateSummaryEpisodeRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            window_messages=12,
        ),
        chat_generator,
    )


async def _summary_rows(
    pool: asyncpg.Pool,
    conversation_id: UUID,
) -> list[asyncpg.Record]:
    async with pool.acquire() as connection:
        return await connection.fetch(
            """
            SELECT
                episodes.id,
                episodes.conversation_id,
                episodes.scope_id,
                episodes.kind,
                episodes.message_id,
                episodes.range_start,
                episodes.range_end,
                COUNT(embeddings_768.episode_id) AS embedding_count,
                MAX(vector_dims(embeddings_768.embedding)) AS embedding_dimensions
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


async def _summary_ranges(pool: asyncpg.Pool, conversation_id: UUID) -> list[tuple[int, int]]:
    rows = await _summary_rows(pool, conversation_id)
    return [(row["range_start"], row["range_end"]) for row in rows]


async def _summary_storage_count(
    pool: asyncpg.Pool,
    conversation_id: UUID,
) -> dict[str, int]:
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT
                (
                    SELECT COUNT(*)
                    FROM episodes
                    WHERE conversation_id = $1
                      AND kind = 'summary'
                ) AS summary_episodes,
                (
                    SELECT COUNT(*)
                    FROM embeddings_768
                    INNER JOIN episodes
                        ON episodes.id = embeddings_768.episode_id
                    WHERE episodes.conversation_id = $1
                      AND episodes.kind = 'summary'
                ) AS summary_embeddings;
            """,
            conversation_id,
        )
    assert row is not None
    return {
        "summary_episodes": row["summary_episodes"],
        "summary_embeddings": row["summary_embeddings"],
    }


async def _seed_summary_episode(
    pool: asyncpg.Pool,
    service: MemoryService,
    conversation_id: UUID,
    scope_id: UUID,
    range_start: int,
    range_end: int,
    content: str,
) -> None:
    vector = await service.embedder.embed_text(content)
    async with pool.acquire() as connection:
        embedding_model_pk = await connection.fetchval(
            """
            SELECT id
            FROM embedding_models
            WHERE model_id = $1
              AND dimensions = 768
              AND is_active = TRUE;
            """,
            service.embedding_model_id,
        )
        episode_id = await connection.fetchval(
            """
            INSERT INTO episodes (conversation_id, scope_id, kind, range_start, range_end, content)
            VALUES ($1, $2, 'summary', $3, $4, $5)
            RETURNING id;
            """,
            conversation_id,
            scope_id,
            range_start,
            range_end,
            content,
        )
        await connection.execute(
            """
            INSERT INTO embeddings_768 (episode_id, model_id, embedding)
            VALUES ($1, $2, $3);
            """,
            episode_id,
            embedding_model_pk,
            list(vector),
        )


def _failure_service(pool: asyncpg.Pool, failure_mode: str) -> MemoryService:
    if failure_mode == "embedding_generation":
        return MemoryService(
            pool=pool,
            embedder=_FailingEmbedder(),
        )
    if failure_mode == "summary_episode_write":
        return MemoryService(
            pool=pool,
            embedder=FakeEmbedder(dimensions=768),
            embedding_model_id="missing-summary-model",
        )
    return MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))


def _failure_chat_generator(failure_mode: str) -> FakeChatGenerator:
    if failure_mode == "summary_generation":
        return FakeChatGenerator(
            response=ChatResponse(
                content="unused summary",
                model="fake-summary-model",
                finish_reason="stop",
            ),
            error=ChatResponseError("fake summary generation failure"),
        )
    return FakeChatGenerator(
        response=ChatResponse(
            content=SUMMARY_CONTENT_SENTINEL,
            model="fake-summary-model",
            finish_reason="stop",
        )
    )


@dataclass(frozen=True)
class _FailingEmbedder:
    async def embed_text(self, text: str) -> EmbeddingVector:
        _ = text
        raise EmbeddingConfigurationError("fake embedding generation failure")

    async def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        _ = texts
        raise EmbeddingConfigurationError("fake embedding generation failure")


@dataclass
class _BlockingSummaryMemoryService:
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    embedding_model_id: str = "fake-embedder"

    async def create_summary_episode_for_next_uncovered_window(
        self,
        request: CreateSummaryEpisodeRequest,
        chat_generator: FakeChatGenerator,
    ) -> None:
        _ = (request, chat_generator)
        self.started.set()
        await self.release.wait()


@dataclass
class _RecordingSummaryScheduler:
    requests: list[SummaryEpisodeMemoryScheduleRequest] = field(default_factory=list)

    def schedule(self, request: SummaryEpisodeMemoryScheduleRequest) -> None:
        self.requests.append(request)


class _NeverDisconnectingRequest:
    async def is_disconnected(self) -> bool:
        return False


async def _assistant_done_events(conversation_id: UUID) -> AsyncIterator[AssistantStreamEvent]:
    yield AssistantStreamStart(used_memory_episode_ids=(), chat_model="fake-stream")
    yield AssistantStreamDone(
        assistant_message=_message(conversation_id=conversation_id),
        chat_model="fake-stream",
        finish_reason="stop",
        used_memory_episode_ids=(),
    )


def _message(conversation_id: UUID):
    from smriti.memory import MessageRecord

    return MessageRecord(
        id=uuid4(),
        conversation_id=conversation_id,
        position=12,
        role="assistant",
        content="done",
        token_count=1,
        created_at=_aware_datetime(),
    )


def _aware_datetime():
    from datetime import datetime, timezone

    return datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017


def _parse_sse(body: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for raw_event in body.strip().split("\n\n"):
        event_name = ""
        data = "{}"
        for line in raw_event.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            if line.startswith("data: "):
                data = line.removeprefix("data: ")
        parsed.append({"event": event_name, "data": data})
    return parsed
