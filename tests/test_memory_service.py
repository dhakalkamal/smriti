from __future__ import annotations

import re
from pathlib import Path

import pytest
from testcontainers.postgres import PostgresContainer

from smriti.config import Settings
from smriti.db.client import close_pool, get_pool
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    AppendMessageRequest,
    ConversationNotFoundError,
    CreateConversationRequest,
    CreateMessageEpisodeRequest,
    CreateScopeRequest,
    ListScopesRequest,
    MemoryService,
    ScopeNotFoundError,
    VectorDimensionError,
)


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


@pytest.mark.asyncio
async def test_memory_service_appends_message_episode_and_embedding() -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        database_url = _to_asyncpg_dsn(postgres.get_connection_url())
        settings = Settings(database_url=database_url)
        await apply_migrations(settings=settings, migrations_dir=migrations_dir)

        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))

        try:
            async with pool.acquire() as connection:
                user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )
                other_user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )

            scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Research Notes",
                    system_prompt="Keep research memory scoped.",
                )
            )
            other_user_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Research Notes",
                    system_prompt="This belongs to another user.",
                )
            )

            scopes = await service.list_scopes(ListScopesRequest(user_id=user_id))
            assert [listed_scope.id for listed_scope in scopes] == [scope.id]

            with pytest.raises(ScopeNotFoundError):
                await service.create_conversation(
                    CreateConversationRequest(
                        user_id=user_id,
                        scope_id=other_user_scope.id,
                        title="Wrong owner",
                    )
                )

            conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    title="Paper trail",
                )
            )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Coding Helper",
                    system_prompt="Keep coding memory separate.",
                )
            )

            with pytest.raises(ConversationNotFoundError):
                await service.append_message(
                    AppendMessageRequest(
                        user_id=user_id,
                        scope_id=other_scope.id,
                        conversation_id=conversation.id,
                        role="user",
                        content="This should not cross scopes.",
                        token_count=6,
                    )
                )

            message = await service.append_message(
                AppendMessageRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    conversation_id=conversation.id,
                    role="user",
                    content="Remember the family research notes.",
                    token_count=6,
                )
            )

            with pytest.raises(ConversationNotFoundError):
                await service.append_message(
                    AppendMessageRequest(
                        user_id=other_user_id,
                        scope_id=scope.id,
                        conversation_id=conversation.id,
                        role="user",
                        content="This should not cross users.",
                        token_count=6,
                    )
                )

            with pytest.raises(ConversationNotFoundError):
                await service.create_message_episode(
                    CreateMessageEpisodeRequest(
                        user_id=other_user_id,
                        scope_id=scope.id,
                        conversation_id=conversation.id,
                        message_id=message.id,
                    )
                )

            with pytest.raises(ConversationNotFoundError):
                await service.create_message_episode(
                    CreateMessageEpisodeRequest(
                        user_id=user_id,
                        scope_id=other_scope.id,
                        conversation_id=conversation.id,
                        message_id=message.id,
                    )
                )

            episode = await service.create_message_episode(
                CreateMessageEpisodeRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    conversation_id=conversation.id,
                    message_id=message.id,
                )
            )

            assert message.position == 1
            assert episode.message_id == message.id
            assert episode.scope_id == scope.id
            assert episode.content == message.content

            async with pool.acquire() as connection:
                counts = await connection.fetchrow(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM messages WHERE conversation_id = $1) AS messages,
                        (SELECT COUNT(*) FROM episodes WHERE conversation_id = $1) AS episodes,
                        (
                            SELECT COUNT(*)
                            FROM embeddings_768
                            WHERE episode_id = $2
                        ) AS embeddings,
                        (
                            SELECT vector_dims(embedding)
                            FROM embeddings_768
                            WHERE episode_id = $2
                        ) AS dimensions,
                        (
                            SELECT scope_id
                            FROM episodes
                            WHERE id = $2
                        ) AS episode_scope_id;
                    """,
                    conversation.id,
                    episode.id,
                )

            assert counts is not None
            assert counts["messages"] == 1
            assert counts["episodes"] == 1
            assert counts["embeddings"] == 1
            assert counts["dimensions"] == 768
            assert counts["episode_scope_id"] == scope.id
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_memory_service_rejects_non_768_embedding_vectors() -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        database_url = _to_asyncpg_dsn(postgres.get_connection_url())
        settings = Settings(database_url=database_url)
        await apply_migrations(settings=settings, migrations_dir=migrations_dir)

        pool = await get_pool(settings)
        service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=16))

        try:
            async with pool.acquire() as connection:
                user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )

            scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Research Notes",
                    system_prompt="Keep research memory scoped.",
                )
            )
            conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    title="Paper trail",
                )
            )
            message = await service.append_message(
                AppendMessageRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    conversation_id=conversation.id,
                    role="user",
                    content="This vector should be rejected.",
                    token_count=5,
                )
            )

            with pytest.raises(VectorDimensionError):
                await service.create_message_episode(
                    CreateMessageEpisodeRequest(
                        user_id=user_id,
                        scope_id=scope.id,
                        conversation_id=conversation.id,
                        message_id=message.id,
                    )
                )

            async with pool.acquire() as connection:
                episode_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM episodes WHERE conversation_id = $1;",
                    conversation.id,
                )

            assert episode_count == 0
        finally:
            await close_pool()
