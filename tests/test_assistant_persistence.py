from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from uuid import UUID, uuid4

import asyncpg
import pytest
from testcontainers.postgres import PostgresContainer

from smriti.config import Settings
from smriti.db.client import close_pool, get_pool
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    AppendAssistantResponseWithProvenanceRequest,
    AppendMessageRequest,
    AssistantResponseRecord,
    CreateConversationRequest,
    CreateMessageEpisodeRequest,
    CreateScopeRequest,
    InvalidMemoryRequestError,
    LoadAssistantGenerationContextRequest,
    MemoryService,
    ScoredEpisode,
)

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


@pytest.mark.asyncio
async def test_load_assistant_generation_context_returns_recent_messages_through_query() -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(database_url=_to_asyncpg_dsn(postgres.get_connection_url()))
        await apply_migrations(settings=settings, migrations_dir=migrations_dir)
        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))

        try:
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            first = await _append_message(
                service,
                user_id,
                scope_id,
                conversation_id,
                "user",
                "one",
            )
            second = await _append_message(
                service, user_id, scope_id, conversation_id, "assistant", "two"
            )
            query = await _append_message(
                service, user_id, scope_id, conversation_id, "user", "three"
            )

            context = await service.load_assistant_generation_context(
                LoadAssistantGenerationContextRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    query_message_id=query.id,
                    recent_message_limit=20,
                )
            )
            limited_context = await service.load_assistant_generation_context(
                LoadAssistantGenerationContextRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    query_message_id=query.id,
                    recent_message_limit=2,
                )
            )

            assert context.scope.id == scope_id
            assert context.scope.system_prompt == "Keep memory scoped."
            assert context.query_message == query
            assert [message.id for message in context.recent_messages] == [
                first.id,
                second.id,
                query.id,
            ]
            assert [message.id for message in limited_context.recent_messages] == [
                second.id,
                query.id,
            ]
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_load_assistant_generation_context_rejects_bad_recent_limit_before_db() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    with pytest.raises(InvalidMemoryRequestError):
        await service.load_assistant_generation_context(
            LoadAssistantGenerationContextRequest(
                user_id=uuid4(),
                scope_id=uuid4(),
                conversation_id=uuid4(),
                query_message_id=uuid4(),
                recent_message_limit=0,
            )
        )


@pytest.mark.asyncio
async def test_append_assistant_response_with_provenance_persists_without_episode() -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(database_url=_to_asyncpg_dsn(postgres.get_connection_url()))
        await apply_migrations(settings=settings, migrations_dir=migrations_dir)
        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))

        try:
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query_message = await _append_message(
                service,
                user_id,
                scope_id,
                conversation_id,
                "user",
                "What should you remember?",
            )
            memory_episode = await _append_embedded_message(
                service,
                user_id,
                scope_id,
                conversation_id,
                "A useful memory",
            )
            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=memory_episode.content,
                top_k=1,
                now=FIXED_NOW,
            )

            response = await service.append_assistant_response_with_provenance(
                AppendAssistantResponseWithProvenanceRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    query_message_id=query_message.id,
                    content="Here is the remembered answer.",
                    token_count=7,
                    used=tuple(results),
                    retrieved_at=FIXED_NOW,
                )
            )

            async with pool.acquire() as connection:
                counts = await connection.fetchrow(
                    """
                    SELECT
                        (
                            SELECT COUNT(*)
                            FROM episodes
                            WHERE message_id = $1
                        ) AS episodes,
                        (
                            SELECT COUNT(*)
                            FROM embeddings_768
                            INNER JOIN episodes
                                ON episodes.id = embeddings_768.episode_id
                            WHERE episodes.message_id = $1
                        ) AS embeddings,
                        (
                            SELECT COUNT(*)
                            FROM message_retrievals
                            WHERE assistant_message_id = $1
                        ) AS retrievals;
                    """,
                    response.message.id,
                )
                row = await connection.fetchrow(
                    """
                    SELECT query_message_id, assistant_message_id, episode_id
                    FROM message_retrievals
                    WHERE assistant_message_id = $1;
                    """,
                    response.message.id,
                )

            assert isinstance(response, AssistantResponseRecord)
            assert response.message.role == "assistant"
            assert response.message.token_count == 7
            assert response.used_episode_ids == (results[0].id,)
            assert counts is not None
            assert counts["episodes"] == 0
            assert counts["embeddings"] == 0
            assert counts["retrievals"] == 1
            assert row is not None
            assert row["query_message_id"] == query_message.id
            assert row["assistant_message_id"] == response.message.id
            assert row["episode_id"] == results[0].id
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_append_assistant_response_with_provenance_rolls_back_on_provenance_failure() -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(database_url=_to_asyncpg_dsn(postgres.get_connection_url()))
        await apply_migrations(settings=settings, migrations_dir=migrations_dir)
        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))

        try:
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query_message = await _append_message(
                service,
                user_id,
                scope_id,
                conversation_id,
                "user",
                "Trigger rollback.",
            )
            memory_episode = await _append_embedded_message(
                service,
                user_id,
                scope_id,
                conversation_id,
                "rollback memory",
            )
            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=memory_episode.content,
                top_k=1,
                now=FIXED_NOW,
            )
            invalid_result = replace(results[0], embedding_model_id=999999)

            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await service.append_assistant_response_with_provenance(
                    AppendAssistantResponseWithProvenanceRequest(
                        user_id=user_id,
                        scope_id=scope_id,
                        conversation_id=conversation_id,
                        query_message_id=query_message.id,
                        content="This should roll back.",
                        token_count=5,
                        used=(invalid_result,),
                        retrieved_at=FIXED_NOW,
                    )
                )

            async with pool.acquire() as connection:
                message_count = await connection.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM messages
                    WHERE conversation_id = $1
                      AND role = 'assistant'
                      AND content = $2;
                    """,
                    conversation_id,
                    "This should roll back.",
                )
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert message_count == 0
            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_append_assistant_response_with_provenance_uses_constant_db_await_shape() -> None:
    user_id = UUID(int=1)
    scope_id = UUID(int=2)
    conversation_id = UUID(int=3)
    query_message_id = UUID(int=4)
    assistant_message_id = UUID(int=5)
    episode = _scored_episode(UUID(int=6))
    connection = _AppendSpyConnection(
        user_id=user_id,
        scope_id=scope_id,
        conversation_id=conversation_id,
        query_message_id=query_message_id,
        assistant_message_id=assistant_message_id,
        episode_id=episode.id,
    )
    service = MemoryService(
        pool=_AppendSpyPool(connection),  # type: ignore[arg-type]
        embedder=FakeEmbedder(dimensions=768),
    )

    response = await service.append_assistant_response_with_provenance(
        AppendAssistantResponseWithProvenanceRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            query_message_id=query_message_id,
            content="assistant",
            token_count=3,
            used=(episode,),
            retrieved_at=FIXED_NOW,
        )
    )

    assert response.message.id == assistant_message_id
    assert connection.operations == ["fetchrow", "fetch", "fetchrow", "executemany"]
    assert "FROM messages AS query_messages" in connection.fetchrow_queries[0]
    assert "FROM episodes" in connection.fetch_queries[0]
    assert "INSERT INTO messages" in connection.fetchrow_queries[1]
    assert "INSERT INTO message_retrievals" in connection.executemany_queries[0]


async def _create_user_scope_conversation(
    service: MemoryService,
    pool: asyncpg.Pool,
) -> tuple[UUID, UUID, UUID]:
    async with pool.acquire() as connection:
        user_id = await connection.fetchval("INSERT INTO users DEFAULT VALUES RETURNING id;")

    scope = await service.create_scope(
        CreateScopeRequest(
            user_id=user_id,
            name=f"Scope {uuid4()}",
            system_prompt="Keep memory scoped.",
        )
    )
    conversation = await service.create_conversation(
        CreateConversationRequest(
            user_id=user_id,
            scope_id=scope.id,
            title="Assistant test",
        )
    )
    return user_id, scope.id, conversation.id


async def _append_message(
    service: MemoryService,
    user_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
    role: str,
    content: str,
):
    return await service.append_message(
        AppendMessageRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            token_count=max(1, len(content.split())),
        )
    )


async def _append_embedded_message(
    service: MemoryService,
    user_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
    content: str,
):
    message = await _append_message(service, user_id, scope_id, conversation_id, "user", content)
    return await service.create_message_episode(
        CreateMessageEpisodeRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            message_id=message.id,
        )
    )


def _scored_episode(episode_id: UUID) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=1,
        id=episode_id,
        user_id=UUID(int=1),
        scope_id=UUID(int=2),
        conversation_id=UUID(int=3),
        kind="message",
        message_id=UUID(int=10),
        message_position=1,
        range_start=None,
        range_end=None,
        content="spy memory",
        created_at=FIXED_NOW,
        importance=0.0,
        access_count=0,
        last_accessed_at=None,
        embedding_model_id=1,
        similarity=1.0,
        recency_score=1.0,
        access_score=0.0,
        importance_score=0.0,
        frequency_score=0.0,
        score=1.0,
    )


class _AppendSpyPool:
    def __init__(self, connection: _AppendSpyConnection) -> None:
        self._connection = connection

    def acquire(self) -> _AppendSpyConnectionContext:
        return _AppendSpyConnectionContext(self._connection)


class _AppendSpyConnection:
    def __init__(
        self,
        user_id: UUID,
        scope_id: UUID,
        conversation_id: UUID,
        query_message_id: UUID,
        assistant_message_id: UUID,
        episode_id: UUID,
    ) -> None:
        self.user_id = user_id
        self.scope_id = scope_id
        self.conversation_id = conversation_id
        self.query_message_id = query_message_id
        self.assistant_message_id = assistant_message_id
        self.episode_id = episode_id
        self.operations: list[str] = []
        self.fetchrow_queries: list[str] = []
        self.fetch_queries: list[str] = []
        self.executemany_queries: list[str] = []

    def transaction(self) -> _AsyncNullContext:
        return _AsyncNullContext()

    async def fetchrow(self, query: str, *args: object) -> dict[str, object]:
        self.operations.append("fetchrow")
        self.fetchrow_queries.append(query)
        if "FROM messages AS query_messages" in query:
            assert args == (self.query_message_id,)
            return {
                "query_conversation_id": self.conversation_id,
                "query_position": 1,
                "query_role": "user",
                "query_user_id": self.user_id,
                "query_scope_id": self.scope_id,
            }
        if "INSERT INTO messages" in query:
            assert args == (self.conversation_id, "assistant", 3)
            return {
                "id": self.assistant_message_id,
                "conversation_id": self.conversation_id,
                "position": 2,
                "role": "assistant",
                "content": "assistant",
                "token_count": 3,
                "created_at": FIXED_NOW,
            }
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.operations.append("fetch")
        self.fetch_queries.append(query)
        assert args == ([self.episode_id], self.scope_id, self.user_id)
        return [{"id": self.episode_id}]

    async def executemany(self, query: str, args: list[tuple[object, ...]]) -> None:
        self.operations.append("executemany")
        self.executemany_queries.append(query)
        assert len(args) == 1


class _AppendSpyConnectionContext:
    def __init__(self, connection: _AppendSpyConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _AppendSpyConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = (exc_type, exc, traceback)


class _AsyncNullContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = (exc_type, exc, traceback)
