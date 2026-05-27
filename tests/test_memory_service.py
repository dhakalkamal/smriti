from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from uuid import UUID, uuid4

import pytest
from testcontainers.postgres import PostgresContainer

from smriti.config import Settings
from smriti.db.client import close_pool, get_pool
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    AppendAssistantResponseWithProvenanceRequest,
    AppendMessageRequest,
    AssistantMessageNotFoundError,
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    CreateConversationRequest,
    CreateMessageEpisodeRequest,
    CreateScopeRequest,
    DeleteConversationRequest,
    EpisodeRecord,
    InvalidMemoryRequestError,
    InvalidProvenanceTargetError,
    InvalidRetrievalRequestError,
    ListConversationsRequest,
    ListScopesRequest,
    MemoryService,
    ScopeAccessDeniedError,
    ScopeNotFoundError,
    ScoredEpisode,
    VectorDimensionError,
)

FIXED_RETRIEVAL_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017


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

            with pytest.raises(ScopeAccessDeniedError):
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
            other_user_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=other_user_id,
                    scope_id=other_user_scope.id,
                    title="Private other-user conversation",
                )
            )
            conversations = await service.list_conversations(
                ListConversationsRequest(user_id=user_id)
            )
            assert [listed_conversation.id for listed_conversation in conversations] == [
                conversation.id
            ]
            assert other_user_conversation.id not in {
                listed_conversation.id for listed_conversation in conversations
            }
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

            with pytest.raises(ConversationNotFoundError):
                await service.append_message(
                    AppendMessageRequest(
                        user_id=user_id,
                        scope_id=scope.id,
                        conversation_id=uuid4(),
                        role="user",
                        content="This conversation does not exist.",
                        token_count=5,
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

            with pytest.raises(ConversationAccessDeniedError):
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

            with pytest.raises(ConversationAccessDeniedError):
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
async def test_delete_conversation_cascades_memory_rows_and_preserves_other_conversations() -> None:
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
            user_id, scope_id, deleted_conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            cross_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    title="Cross-conversation provenance",
                )
            )
            same_scope_unrelated_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    title="Same scope survivor",
                )
            )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Other Scope",
                    system_prompt="Keep this partition separate.",
                )
            )
            other_scope_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=other_scope.id,
                    title="Other scope survivor",
                )
            )

            deleted_message_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=deleted_conversation_id,
                content="deleted conversation memory",
            )
            deleted_query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=deleted_conversation_id,
                role="user",
                content="Use memory from another conversation.",
            )
            deleted_assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=deleted_conversation_id,
                role="assistant",
                content="I used that external memory.",
            )

            cross_message_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=cross_conversation.id,
                content="external memory that must survive deletion",
            )
            cross_query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=cross_conversation.id,
                role="user",
                content="Use memory from the deleted conversation.",
            )
            cross_assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=cross_conversation.id,
                role="assistant",
                content="I used the deleted conversation memory.",
            )
            same_scope_unrelated_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=same_scope_unrelated_conversation.id,
                content="same scope unrelated memory",
            )
            other_scope_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=other_scope.id,
                conversation_id=other_scope_conversation.id,
                content="other scope memory",
            )

            summary_vector = await service.embedder.embed_text("deleted summary memory")
            async with pool.acquire() as connection:
                deleted_summary_episode_id = await connection.fetchval(
                    """
                    INSERT INTO episodes (
                        conversation_id,
                        scope_id,
                        kind,
                        range_start,
                        range_end,
                        content
                    )
                    VALUES ($1, $2, 'summary', 1, 3, $3)
                    RETURNING id;
                    """,
                    deleted_conversation_id,
                    scope_id,
                    "summary for the deleted conversation",
                )
                await connection.execute(
                    """
                    INSERT INTO embeddings_768 (episode_id, model_id, embedding)
                    VALUES ($1, $2, $3);
                    """,
                    deleted_summary_episode_id,
                    deleted_message_episode.embedding_model_id,
                    list(summary_vector),
                )

            await service.record_used_memories(
                user_id=user_id,
                scope_id=scope_id,
                query_message_id=cross_query_message.id,
                assistant_message_id=cross_assistant_message.id,
                used=[_scored_episode_stub(episode_id=deleted_message_episode.id)],
                retrieved_at=FIXED_RETRIEVAL_NOW,
            )
            await service.record_used_memories(
                user_id=user_id,
                scope_id=scope_id,
                query_message_id=deleted_query_message.id,
                assistant_message_id=deleted_assistant_message.id,
                used=[_scored_episode_stub(episode_id=cross_message_episode.id)],
                retrieved_at=FIXED_RETRIEVAL_NOW + timedelta(minutes=1),
            )

            async with pool.acquire() as connection:
                cross_references_deleted_retrieval_id = await connection.fetchval(
                    """
                    SELECT id
                    FROM message_retrievals
                    WHERE query_conversation_id = $1
                      AND episode_id = $2;
                    """,
                    cross_conversation.id,
                    deleted_message_episode.id,
                )
                deleted_references_cross_retrieval_id = await connection.fetchval(
                    """
                    SELECT id
                    FROM message_retrievals
                    WHERE query_conversation_id = $1
                      AND episode_id = $2;
                    """,
                    deleted_conversation_id,
                    cross_message_episode.id,
                )
                pre_delete_summary_embedding_exists = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM embeddings_768
                        WHERE episode_id = $1
                    );
                    """,
                    deleted_summary_episode_id,
                )

            assert cross_references_deleted_retrieval_id is not None
            assert deleted_references_cross_retrieval_id is not None
            assert pre_delete_summary_embedding_exists is True

            await service.delete_conversation(
                DeleteConversationRequest(
                    user_id=user_id,
                    conversation_id=deleted_conversation_id,
                )
            )

            async with pool.acquire() as connection:
                conversation_rows = await connection.fetch(
                    """
                    SELECT id
                    FROM conversations
                    WHERE id = ANY($1::uuid[]);
                    """,
                    [
                        deleted_conversation_id,
                        cross_conversation.id,
                        same_scope_unrelated_conversation.id,
                        other_scope_conversation.id,
                    ],
                )
                message_rows = await connection.fetch(
                    """
                    SELECT id
                    FROM messages
                    WHERE id = ANY($1::uuid[]);
                    """,
                    [
                        deleted_message_episode.message_id,
                        deleted_query_message.id,
                        deleted_assistant_message.id,
                        cross_message_episode.message_id,
                        cross_query_message.id,
                        cross_assistant_message.id,
                        same_scope_unrelated_episode.message_id,
                        other_scope_episode.message_id,
                    ],
                )
                episode_rows = await connection.fetch(
                    """
                    SELECT id
                    FROM episodes
                    WHERE id = ANY($1::uuid[]);
                    """,
                    [
                        deleted_message_episode.id,
                        deleted_summary_episode_id,
                        cross_message_episode.id,
                        same_scope_unrelated_episode.id,
                        other_scope_episode.id,
                    ],
                )
                embedding_rows = await connection.fetch(
                    """
                    SELECT episode_id
                    FROM embeddings_768
                    WHERE episode_id = ANY($1::uuid[]);
                    """,
                    [
                        deleted_message_episode.id,
                        deleted_summary_episode_id,
                        cross_message_episode.id,
                        same_scope_unrelated_episode.id,
                        other_scope_episode.id,
                    ],
                )
                retrieval_rows = await connection.fetch(
                    """
                    SELECT id
                    FROM message_retrievals
                    WHERE id = ANY($1::uuid[]);
                    """,
                    [
                        cross_references_deleted_retrieval_id,
                        deleted_references_cross_retrieval_id,
                    ],
                )

            conversation_ids = {row["id"] for row in conversation_rows}
            assert deleted_conversation_id not in conversation_ids
            assert cross_conversation.id in conversation_ids
            assert same_scope_unrelated_conversation.id in conversation_ids
            assert other_scope_conversation.id in conversation_ids

            message_ids = {row["id"] for row in message_rows}
            assert deleted_message_episode.message_id not in message_ids
            assert deleted_query_message.id not in message_ids
            assert deleted_assistant_message.id not in message_ids
            assert cross_message_episode.message_id in message_ids
            assert cross_query_message.id in message_ids
            assert cross_assistant_message.id in message_ids
            assert same_scope_unrelated_episode.message_id in message_ids
            assert other_scope_episode.message_id in message_ids

            episode_ids = {row["id"] for row in episode_rows}
            assert deleted_message_episode.id not in episode_ids
            assert deleted_summary_episode_id not in episode_ids
            assert cross_message_episode.id in episode_ids
            assert same_scope_unrelated_episode.id in episode_ids
            assert other_scope_episode.id in episode_ids

            embedding_episode_ids = {row["episode_id"] for row in embedding_rows}
            assert deleted_message_episode.id not in embedding_episode_ids
            assert deleted_summary_episode_id not in embedding_episode_ids
            assert cross_message_episode.id in embedding_episode_ids
            assert same_scope_unrelated_episode.id in embedding_episode_ids
            assert other_scope_episode.id in embedding_episode_ids

            retrieval_ids = {row["id"] for row in retrieval_rows}
            assert cross_references_deleted_retrieval_id not in retrieval_ids
            assert deleted_references_cross_retrieval_id not in retrieval_ids
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_delete_conversation_raises_not_found_for_missing_and_non_owned() -> None:
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
            user_id, _, _ = await _create_user_scope_conversation(service, pool)
            async with pool.acquire() as connection:
                other_user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Other User Scope",
                    system_prompt="Private to another user.",
                )
            )
            other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=other_user_id,
                    scope_id=other_scope.id,
                    title="Other user's conversation",
                )
            )

            for conversation_id in [uuid4(), other_conversation.id]:
                with pytest.raises(
                    ConversationNotFoundError,
                    match="Conversation does not exist",
                ):
                    await service.delete_conversation(
                        DeleteConversationRequest(
                            user_id=user_id,
                            conversation_id=conversation_id,
                        )
                    )

            async with pool.acquire() as connection:
                other_conversation_exists = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM conversations
                        WHERE id = $1
                    );
                    """,
                    other_conversation.id,
                )

            assert other_conversation_exists is True
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_append_message_rejects_negative_token_count_before_db_work() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    with pytest.raises(InvalidMemoryRequestError):
        await service.append_message(
            AppendMessageRequest(
                user_id=uuid4(),
                scope_id=uuid4(),
                conversation_id=uuid4(),
                role="user",
                content="Invalid token count.",
                token_count=-1,
            )
        )


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


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_returns_only_scope_and_embedded_rows() -> None:
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

            scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Research Notes",
                    system_prompt="Keep research memory scoped.",
                )
            )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Coding Helper",
                    system_prompt="Keep coding memory separate.",
                )
            )
            conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    title="Research",
                )
            )
            other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=other_scope.id,
                    title="Coding",
                )
            )

            scoped_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope.id,
                conversation_id=conversation.id,
                content="family research notes",
            )
            other_scope_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=other_scope.id,
                conversation_id=other_conversation.id,
                content="family research notes",
            )
            unembedded_message = await service.append_message(
                AppendMessageRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    conversation_id=conversation.id,
                    role="user",
                    content="family research notes without embedding",
                    token_count=5,
                )
            )
            async with pool.acquire() as connection:
                unembedded_episode_id = await connection.fetchval(
                    """
                    INSERT INTO episodes (conversation_id, scope_id, kind, message_id, content)
                    VALUES ($1, $2, 'message', $3, $4)
                    RETURNING id;
                    """,
                    conversation.id,
                    scope.id,
                    unembedded_message.id,
                    unembedded_message.content,
                )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope.id,
                query="family research notes",
                top_k=10,
                now=FIXED_RETRIEVAL_NOW,
            )

            result_ids = {result.id for result in results}
            assert result_ids == {scoped_episode.id}
            assert other_scope_episode.id not in result_ids
            assert unembedded_episode_id not in result_ids
            assert all(result.user_id == user_id for result in results)
            assert all(result.scope_id == scope.id for result in results)
            assert results[0].score != results[0].similarity
            assert 0.0 <= results[0].recency_score <= 1.0
            assert results[0].access_score == 0.0
            assert 0.0 <= results[0].importance_score <= 1.0
            assert 0.0 <= results[0].frequency_score <= 1.0
            assert results[0].embedding_model_id == scoped_episode.embedding_model_id
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_rejects_wrong_user_for_scope() -> None:
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

            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Other User Scope",
                    system_prompt="Private to the other user.",
                )
            )

            with pytest.raises(ScopeAccessDeniedError):
                await service.retrieve_scoped_episodes(
                    user_id=user_id,
                    scope_id=other_scope.id,
                    query="private notes",
                    top_k=5,
                    now=FIXED_RETRIEVAL_NOW,
                )

            with pytest.raises(ScopeNotFoundError):
                await service.retrieve_scoped_episodes(
                    user_id=user_id,
                    scope_id=uuid4(),
                    query="missing scope",
                    top_k=5,
                    now=FIXED_RETRIEVAL_NOW,
                )
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_respects_top_k() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )

            for content in [
                "alpha project memory one",
                "alpha project memory two",
                "alpha project memory three",
            ]:
                await _append_embedded_message(
                    service=service,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    content=content,
                )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query="alpha project memory",
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )

            assert len(results) == 2
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_populates_one_indexed_result_rank() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            episodes = [
                await _append_embedded_message(
                    service=service,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    content=f"ranked memory {index}",
                )
                for index in range(3)
            ]
            query_vector = await service.embedder.embed_text("ranked memory")
            for index, episode in enumerate(episodes):
                await _set_episode_embedding(pool, episode.id, query_vector)
                await _set_episode_scoring_fields(
                    pool=pool,
                    episode_id=episode.id,
                    created_at=FIXED_RETRIEVAL_NOW - timedelta(minutes=index),
                    last_accessed_at=None,
                    importance=0.0,
                    access_count=0,
                )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query="ranked memory",
                top_k=3,
                now=FIXED_RETRIEVAL_NOW,
            )

            assert [result.id for result in results] == [episode.id for episode in episodes]
            assert [result.result_rank for result in results] == [1, 2, 3]
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_uses_weighted_score_components() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query = "weighted scoring"
            episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content=query,
            )
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=episode.id,
                created_at=FIXED_RETRIEVAL_NOW,
                last_accessed_at=FIXED_RETRIEVAL_NOW,
                importance=1.0,
                access_count=10,
            )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=query,
                top_k=1,
                now=FIXED_RETRIEVAL_NOW,
            )

            result = results[0]
            assert result.similarity == pytest.approx(1.0)
            assert result.recency_score == pytest.approx(1.0)
            assert result.access_score == pytest.approx(1.0)
            assert result.importance_score == pytest.approx(1.0)
            assert result.frequency_score == pytest.approx(0.5)

            expected_score = (
                0.55 * result.similarity
                + 0.20 * result.recency_score
                + 0.10 * result.access_score
                + 0.10 * result.importance_score
                + 0.05 * result.frequency_score
            )
            assert result.score == pytest.approx(expected_score)
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_recency_can_rerank_close_similarity() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query = "recency rerank"
            old_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="old high similarity",
            )
            recent_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="recent close similarity",
            )
            query_vector = await service.embedder.embed_text(query)
            await _set_episode_embedding(pool, old_episode.id, query_vector)
            await _set_episode_embedding(pool, recent_episode.id, _near_vector(query_vector, 0.98))
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=old_episode.id,
                created_at=FIXED_RETRIEVAL_NOW - timedelta(days=365),
                last_accessed_at=None,
                importance=0.0,
                access_count=0,
            )
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=recent_episode.id,
                created_at=FIXED_RETRIEVAL_NOW,
                last_accessed_at=None,
                importance=0.0,
                access_count=0,
            )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=query,
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )

            assert results[0].id == recent_episode.id
            assert results[1].id == old_episode.id
            assert results[1].similarity > results[0].similarity
            assert results[0].score > results[1].score
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_access_importance_and_frequency_scores() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query = "component scoring"
            recent_access_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="recent access",
            )
            old_access_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="old access",
            )
            query_vector = await service.embedder.embed_text(query)
            for episode_id in [recent_access_episode.id, old_access_episode.id]:
                await _set_episode_embedding(pool, episode_id, query_vector)

            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=recent_access_episode.id,
                created_at=FIXED_RETRIEVAL_NOW,
                last_accessed_at=FIXED_RETRIEVAL_NOW,
                importance=1.0,
                access_count=30,
            )
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=old_access_episode.id,
                created_at=FIXED_RETRIEVAL_NOW,
                last_accessed_at=FIXED_RETRIEVAL_NOW - timedelta(days=365),
                importance=0.0,
                access_count=0,
            )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=query,
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )
            by_id = {result.id: result for result in results}
            recent_access = by_id[recent_access_episode.id]
            old_access = by_id[old_access_episode.id]

            assert old_access.access_score < recent_access.access_score
            assert recent_access.importance_score == pytest.approx(1.0)
            assert old_access.importance_score == pytest.approx(0.0)
            assert recent_access.frequency_score == pytest.approx(30 / (30 + 10))
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_clamps_importance_score() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            high_importance_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="high importance",
            )
            low_importance_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="low importance",
            )
            async with pool.acquire() as connection:
                await connection.execute(
                    "ALTER TABLE episodes DROP CONSTRAINT IF EXISTS episodes_importance_check;"
                )

            for episode_id, importance in [
                (high_importance_episode.id, 1.5),
                (low_importance_episode.id, -0.5),
            ]:
                await _set_episode_scoring_fields(
                    pool=pool,
                    episode_id=episode_id,
                    created_at=FIXED_RETRIEVAL_NOW,
                    last_accessed_at=None,
                    importance=importance,
                    access_count=0,
                )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query="importance",
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )
            by_id = {result.id: result for result in results}

            assert by_id[high_importance_episode.id].importance_score == pytest.approx(1.0)
            assert by_id[low_importance_episode.id].importance_score == pytest.approx(0.0)
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_updates_access_metadata_only_for_returned_rows() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query = "access metadata"
            episodes = [
                await _append_embedded_message(
                    service=service,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    content=f"{query} {index}",
                )
                for index in range(3)
            ]
            query_vector = await service.embedder.embed_text(query)
            for index, episode in enumerate(episodes):
                await _set_episode_embedding(pool, episode.id, query_vector)
                await _set_episode_scoring_fields(
                    pool=pool,
                    episode_id=episode.id,
                    created_at=FIXED_RETRIEVAL_NOW - timedelta(minutes=index),
                    last_accessed_at=None,
                    importance=0.0,
                    access_count=0,
                )

            first_results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=query,
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )

            assert [result.id for result in first_results] == [
                episodes[0].id,
                episodes[1].id,
            ]
            assert all(result.access_count == 0 for result in first_results)
            assert all(result.last_accessed_at is None for result in first_results)

            async with pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT id, access_count, last_accessed_at
                    FROM episodes
                    WHERE id = ANY($1::uuid[]);
                    """,
                    [episode.id for episode in episodes],
                )
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            by_id = {row["id"]: row for row in rows}
            assert by_id[episodes[0].id]["access_count"] == 1
            assert by_id[episodes[0].id]["last_accessed_at"] == FIXED_RETRIEVAL_NOW
            assert by_id[episodes[1].id]["access_count"] == 1
            assert by_id[episodes[1].id]["last_accessed_at"] == FIXED_RETRIEVAL_NOW
            assert by_id[episodes[2].id]["access_count"] == 0
            assert by_id[episodes[2].id]["last_accessed_at"] is None
            assert retrieval_count == 0

            second_results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=query,
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )

            assert [result.id for result in second_results] == [
                episodes[0].id,
                episodes[1].id,
            ]

            async with pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT id, access_count, last_accessed_at
                    FROM episodes
                    WHERE id = ANY($1::uuid[]);
                    """,
                    [episode.id for episode in episodes],
                )
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            by_id = {row["id"]: row for row in rows}
            assert by_id[episodes[0].id]["access_count"] == 2
            assert by_id[episodes[0].id]["last_accessed_at"] == FIXED_RETRIEVAL_NOW
            assert by_id[episodes[1].id]["access_count"] == 2
            assert by_id[episodes[1].id]["last_accessed_at"] == FIXED_RETRIEVAL_NOW
            assert by_id[episodes[2].id]["access_count"] == 0
            assert by_id[episodes[2].id]["last_accessed_at"] is None
            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_with_zero_results_performs_no_access_update() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            message = await service.append_message(
                AppendMessageRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    role="user",
                    content="unembedded retrieval candidate",
                    token_count=3,
                )
            )
            async with pool.acquire() as connection:
                unembedded_episode_id = await connection.fetchval(
                    """
                    INSERT INTO episodes (conversation_id, scope_id, kind, message_id, content)
                    VALUES ($1, $2, 'message', $3, $4)
                    RETURNING id;
                    """,
                    conversation_id,
                    scope_id,
                    message.id,
                    message.content,
                )
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=unembedded_episode_id,
                created_at=FIXED_RETRIEVAL_NOW,
                last_accessed_at=None,
                importance=0.0,
                access_count=4,
            )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query="unembedded retrieval candidate",
                top_k=5,
                now=FIXED_RETRIEVAL_NOW,
            )

            async with pool.acquire() as connection:
                row = await connection.fetchrow(
                    """
                    SELECT access_count, last_accessed_at
                    FROM episodes
                    WHERE id = $1;
                    """,
                    unembedded_episode_id,
                )
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert results == []
            assert row is not None
            assert row["access_count"] == 4
            assert row["last_accessed_at"] is None
            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_wrong_user_does_not_update_access_metadata() -> None:
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
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Other User Scope",
                    system_prompt="Private to the other user.",
                )
            )
            other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=other_user_id,
                    scope_id=other_scope.id,
                    title="Private",
                )
            )
            episode = await _append_embedded_message(
                service=service,
                user_id=other_user_id,
                scope_id=other_scope.id,
                conversation_id=other_conversation.id,
                content="private access metadata",
            )
            last_accessed_at = FIXED_RETRIEVAL_NOW - timedelta(days=2)
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=episode.id,
                created_at=FIXED_RETRIEVAL_NOW,
                last_accessed_at=last_accessed_at,
                importance=0.0,
                access_count=3,
            )

            with pytest.raises(ScopeAccessDeniedError):
                await service.retrieve_scoped_episodes(
                    user_id=user_id,
                    scope_id=other_scope.id,
                    query="private access metadata",
                    top_k=1,
                    now=FIXED_RETRIEVAL_NOW,
                )

            async with pool.acquire() as connection:
                row = await connection.fetchrow(
                    """
                    SELECT access_count, last_accessed_at
                    FROM episodes
                    WHERE id = $1;
                    """,
                    episode.id,
                )
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert row is not None
            assert row["access_count"] == 3
            assert row["last_accessed_at"] == last_accessed_at
            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_sql_filters_scope_before_similarity_ordering() -> None:
    user_id = uuid4()
    scope_id = uuid4()
    connection = _RetrievalSqlSpyConnection(scope_owner_id=user_id)
    pool = _RetrievalSqlSpyPool(connection)
    service = MemoryService(pool=pool, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    results = await service.retrieve_scoped_episodes(
        user_id=user_id,
        scope_id=scope_id,
        query="shape check",
        top_k=3,
        now=FIXED_RETRIEVAL_NOW,
    )

    assert results == []
    assert len(connection.fetch_queries) == 1
    retrieval_sql = " ".join(connection.fetch_queries[0].split())
    scope_predicate = "WHERE episodes.scope_id = $1"
    user_predicate = "AND conversations.user_id = $2"
    order_clause = "ORDER BY similarity DESC"
    limit_clause = "LIMIT $5"
    assert scope_predicate in retrieval_sql
    assert user_predicate in retrieval_sql
    assert order_clause in retrieval_sql
    assert limit_clause in retrieval_sql
    assert retrieval_sql.index(scope_predicate) < retrieval_sql.index(order_clause)
    assert retrieval_sql.index(user_predicate) < retrieval_sql.index(order_clause)
    assert retrieval_sql.index(scope_predicate) < retrieval_sql.index(limit_clause)
    assert retrieval_sql.index(user_predicate) < retrieval_sql.index(limit_clause)


@pytest.mark.asyncio
async def test_record_used_memories_writes_immutable_score_snapshots() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="user",
                content="Which memories were used?",
            )
            episodes = [
                await _append_embedded_message(
                    service=service,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    content=f"snapshot memory {index}",
                )
                for index in range(3)
            ]
            query_vector = await service.embedder.embed_text("snapshot memory")
            for index, episode in enumerate(episodes):
                await _set_episode_embedding(pool, episode.id, query_vector)
                await _set_episode_scoring_fields(
                    pool=pool,
                    episode_id=episode.id,
                    created_at=FIXED_RETRIEVAL_NOW - timedelta(minutes=index),
                    last_accessed_at=None,
                    importance=0.1 * index,
                    access_count=index,
                )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query="snapshot memory",
                top_k=3,
                now=FIXED_RETRIEVAL_NOW,
            )
            assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="assistant",
                content="Here are the memories I used.",
            )
            async with pool.acquire() as connection:
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )
            assert retrieval_count == 0

            used = [results[1], results[2]]
            retrieved_at = FIXED_RETRIEVAL_NOW + timedelta(minutes=5)
            await service.record_used_memories(
                user_id=user_id,
                scope_id=scope_id,
                query_message_id=query_message.id,
                assistant_message_id=assistant_message.id,
                used=used,
                scoring_version="stage-5.2-weighted-v1",
                retrieved_at=retrieved_at,
            )

            async with pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT
                        query_message_id,
                        assistant_message_id,
                        query_conversation_id,
                        scope_id,
                        episode_id,
                        embedding_model_id,
                        result_rank,
                        similarity,
                        recency_score,
                        access_score,
                        importance_score,
                        frequency_score,
                        score,
                        scoring_version,
                        retrieved_at
                    FROM message_retrievals
                    WHERE query_message_id = $1
                    ORDER BY result_rank ASC, episode_id ASC;
                    """,
                    query_message.id,
                )

            assert len(rows) == 2
            assert [row["episode_id"] for row in rows] == [episode.id for episode in used]
            assert [row["result_rank"] for row in rows] == [2, 3]
            for row, episode in zip(rows, used, strict=True):
                assert row["query_message_id"] == query_message.id
                assert row["assistant_message_id"] == assistant_message.id
                assert row["query_conversation_id"] == query_message.conversation_id
                assert row["scope_id"] == scope_id
                assert row["embedding_model_id"] == episode.embedding_model_id
                assert row["similarity"] == pytest.approx(episode.similarity, abs=1e-12)
                assert row["recency_score"] == pytest.approx(episode.recency_score, abs=1e-12)
                assert row["access_score"] == pytest.approx(episode.access_score, abs=1e-12)
                assert row["importance_score"] == pytest.approx(
                    episode.importance_score,
                    abs=1e-12,
                )
                assert row["frequency_score"] == pytest.approx(
                    episode.frequency_score,
                    abs=1e-12,
                )
                assert row["score"] == pytest.approx(episode.score, abs=1e-12)
                assert row["scoring_version"] == "stage-5.2-weighted-v1"
                assert row["retrieved_at"] == retrieved_at

            await service.record_used_memories(
                user_id=user_id,
                scope_id=scope_id,
                query_message_id=query_message.id,
                assistant_message_id=assistant_message.id,
                used=[used[0]],
                scoring_version="stage-5.2-weighted-v1",
                retrieved_at=retrieved_at + timedelta(minutes=1),
            )
            before_default_retrieved_at = datetime.now(timezone.utc)  # noqa: UP017
            await service.record_used_memories(
                user_id=user_id,
                scope_id=scope_id,
                query_message_id=query_message.id,
                assistant_message_id=assistant_message.id,
                used=[results[0]],
            )
            after_default_retrieved_at = datetime.now(timezone.utc)  # noqa: UP017

            async with pool.acquire() as connection:
                duplicate_count = await connection.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM message_retrievals
                    WHERE query_message_id = $1
                      AND episode_id = $2;
                    """,
                    query_message.id,
                    used[0].id,
                )
                default_row = await connection.fetchrow(
                    """
                    SELECT retrieved_at
                    FROM message_retrievals
                    WHERE query_message_id = $1
                      AND episode_id = $2
                    ORDER BY retrieved_at DESC
                    LIMIT 1;
                    """,
                    query_message.id,
                    results[0].id,
                )

            assert duplicate_count == 2
            assert default_row is not None
            assert default_row["retrieved_at"].tzinfo is not None
            assert before_default_retrieved_at <= default_row["retrieved_at"]
            assert default_row["retrieved_at"] <= after_default_retrieved_at
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_list_message_retrievals_returns_ordered_records_and_is_read_only() -> None:
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

            scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Inspection Scope",
                    system_prompt="Keep inspection scoped.",
                )
            )
            query_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    title="Question thread",
                )
            )
            source_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    title="Source Conversation",
                )
            )
            query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope.id,
                conversation_id=query_conversation.id,
                role="user",
                content="Which memories were used?",
            )
            rank_two_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope.id,
                conversation_id=source_conversation.id,
                content="The slower memory came second.",
            )
            rank_one_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope.id,
                conversation_id=source_conversation.id,
                content="The best memory came first.",
            )
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=rank_two_episode.id,
                created_at=FIXED_RETRIEVAL_NOW - timedelta(days=2),
                last_accessed_at=FIXED_RETRIEVAL_NOW - timedelta(hours=3),
                importance=0.25,
                access_count=4,
            )
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=rank_one_episode.id,
                created_at=FIXED_RETRIEVAL_NOW - timedelta(days=1),
                last_accessed_at=FIXED_RETRIEVAL_NOW - timedelta(hours=2),
                importance=0.5,
                access_count=7,
            )
            retrieved_at = FIXED_RETRIEVAL_NOW + timedelta(minutes=10)
            assistant_response = await service.append_assistant_response_with_provenance(
                AppendAssistantResponseWithProvenanceRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    conversation_id=query_conversation.id,
                    query_message_id=query_message.id,
                    content="I found two memories.",
                    token_count=5,
                    used=(
                        _scored_episode_for_retrieval(
                            episode=rank_two_episode,
                            user_id=user_id,
                            result_rank=2,
                            similarity=0.82,
                            recency_score=0.31,
                            access_score=0.42,
                            frequency_score=0.53,
                            importance_score=0.64,
                            score=0.75,
                        ),
                        _scored_episode_for_retrieval(
                            episode=rank_one_episode,
                            user_id=user_id,
                            result_rank=1,
                            similarity=0.93,
                            recency_score=0.34,
                            access_score=0.45,
                            frequency_score=0.56,
                            importance_score=0.67,
                            score=0.86,
                        ),
                    ),
                    scoring_version="inspection-test-v1",
                    retrieved_at=retrieved_at,
                )
            )

            episode_ids = [rank_one_episode.id, rank_two_episode.id]
            before_access_metadata = await _episode_access_metadata(pool, episode_ids)

            records = await service.list_message_retrievals(
                user_id=user_id,
                conversation_id=query_conversation.id,
                assistant_message_id=assistant_response.message.id,
            )

            after_access_metadata = await _episode_access_metadata(pool, episode_ids)

            assert before_access_metadata == after_access_metadata
            assert [record.rank for record in records] == [1, 2]

            first_record, second_record = records
            assert first_record.similarity == pytest.approx(0.93)
            assert first_record.score == pytest.approx(0.86)
            assert first_record.recency_score == pytest.approx(0.34)
            assert first_record.access_score == pytest.approx(0.45)
            assert first_record.frequency_score == pytest.approx(0.56)
            assert first_record.importance_score == pytest.approx(0.67)
            assert first_record.scoring_version == "inspection-test-v1"
            assert first_record.retrieved_at == retrieved_at
            assert first_record.query.message_id == query_message.id
            assert first_record.query.content == "Which memories were used?"
            assert first_record.episode.id == rank_one_episode.id
            assert first_record.episode.kind == "message"
            assert first_record.episode.content == "The best memory came first."
            assert first_record.episode.source_conversation_id == source_conversation.id
            assert first_record.episode.source_conversation_title == "Source Conversation"
            assert first_record.episode.source_scope_id == scope.id
            assert first_record.episode.source_scope_name == "Inspection Scope"

            assert second_record.episode.id == rank_two_episode.id
            assert second_record.rank == 2
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_list_message_retrievals_rejects_invalid_assistant_message_ids() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            user_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="user",
                content="I am not an assistant message.",
            )
            system_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="system",
                content="System messages cannot own retrieval provenance.",
            )
            owned_other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    title="Other owned conversation",
                )
            )
            other_conversation_assistant = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=owned_other_conversation.id,
                role="assistant",
                content="Assistant in the wrong owned conversation.",
            )

            async with pool.acquire() as connection:
                other_user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Other user's scope",
                    system_prompt="Private to another user.",
                )
            )
            other_user_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=other_user_id,
                    scope_id=other_scope.id,
                    title="Other user's conversation",
                )
            )
            other_user_assistant = await _append_message(
                service=service,
                user_id=other_user_id,
                scope_id=other_scope.id,
                conversation_id=other_user_conversation.id,
                role="assistant",
                content="Assistant in another user's conversation.",
            )

            invalid_assistant_message_ids = [
                uuid4(),
                user_message.id,
                system_message.id,
                other_conversation_assistant.id,
                other_user_assistant.id,
            ]
            for assistant_message_id in invalid_assistant_message_ids:
                with pytest.raises(AssistantMessageNotFoundError):
                    await service.list_message_retrievals(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        assistant_message_id=assistant_message_id,
                    )
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_list_message_retrievals_uses_conversation_authorization_first() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="assistant",
                content="Valid assistant message.",
            )

            async with pool.acquire() as connection:
                other_user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Conversation authorization scope",
                    system_prompt="Private to another user.",
                )
            )
            other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=other_user_id,
                    scope_id=other_scope.id,
                    title="Other user's conversation",
                )
            )

            with pytest.raises(ConversationNotFoundError):
                await service.list_message_retrievals(
                    user_id=user_id,
                    conversation_id=uuid4(),
                    assistant_message_id=assistant_message.id,
                )

            with pytest.raises(ConversationAccessDeniedError):
                await service.list_message_retrievals(
                    user_id=user_id,
                    conversation_id=other_conversation.id,
                    assistant_message_id=assistant_message.id,
                )
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_record_used_memories_empty_used_is_noop_without_pool() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    await service.record_used_memories(
        user_id=uuid4(),
        scope_id=uuid4(),
        query_message_id=uuid4(),
        assistant_message_id=uuid4(),
        used=[],
    )


@pytest.mark.asyncio
async def test_record_used_memories_rejects_duplicate_episode_ids_before_db_work() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]
    episode = _scored_episode_stub()

    with pytest.raises(InvalidRetrievalRequestError):
        await service.record_used_memories(
            user_id=uuid4(),
            scope_id=uuid4(),
            query_message_id=uuid4(),
            assistant_message_id=uuid4(),
            used=[episode, episode],
        )


@pytest.mark.asyncio
async def test_record_used_memories_rejects_empty_scoring_version_before_db_work() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    with pytest.raises(InvalidRetrievalRequestError):
        await service.record_used_memories(
            user_id=uuid4(),
            scope_id=uuid4(),
            query_message_id=uuid4(),
            assistant_message_id=uuid4(),
            used=[_scored_episode_stub()],
            scoring_version=" \t ",
        )


@pytest.mark.asyncio
async def test_record_used_memories_rejects_naive_retrieved_at_before_db_work() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    with pytest.raises(InvalidRetrievalRequestError):
        await service.record_used_memories(
            user_id=uuid4(),
            scope_id=uuid4(),
            query_message_id=uuid4(),
            assistant_message_id=uuid4(),
            used=[_scored_episode_stub()],
            retrieved_at=datetime(2026, 1, 1, 12, 0),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["assistant", "system"])
async def test_record_used_memories_requires_user_query_message(role: str) -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role=role,
                content="This message must not own provenance.",
            )
            episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="valid used memory",
            )
            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=episode.content,
                top_k=1,
                now=FIXED_RETRIEVAL_NOW,
            )
            assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="assistant",
                content="This assistant response may own provenance.",
            )

            with pytest.raises(InvalidProvenanceTargetError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=query_message.id,
                    assistant_message_id=assistant_message.id,
                    used=results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            async with pool.acquire() as connection:
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "system"])
async def test_record_used_memories_requires_assistant_response_message(role: str) -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="user",
                content="Valid user query.",
            )
            invalid_assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role=role,
                content="This message must not own assistant provenance.",
            )
            episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="valid used memory",
            )
            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=episode.content,
                top_k=1,
                now=FIXED_RETRIEVAL_NOW,
            )

            with pytest.raises(InvalidProvenanceTargetError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=query_message.id,
                    assistant_message_id=invalid_assistant_message.id,
                    used=results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            async with pool.acquire() as connection:
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_record_used_memories_rejects_misaligned_assistant_message() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="user",
                content="Valid user query.",
            )
            episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="valid used memory",
            )
            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=episode.content,
                top_k=1,
                now=FIXED_RETRIEVAL_NOW,
            )

            same_scope_other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope_id,
                    title="Wrong conversation",
                )
            )
            wrong_conversation_assistant = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=same_scope_other_conversation.id,
                role="assistant",
                content="Right scope, wrong conversation.",
            )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Assistant Other Scope",
                    system_prompt="Different memory partition.",
                )
            )
            other_scope_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=other_scope.id,
                    title="Wrong assistant scope",
                )
            )
            wrong_scope_assistant = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=other_scope.id,
                conversation_id=other_scope_conversation.id,
                role="assistant",
                content="Wrong scope assistant response.",
            )
            async with pool.acquire() as connection:
                other_user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )
            other_user_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Other User Assistant Scope",
                    system_prompt="Private to another user.",
                )
            )
            other_user_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=other_user_id,
                    scope_id=other_user_scope.id,
                    title="Wrong assistant user",
                )
            )
            wrong_user_assistant = await _append_message(
                service=service,
                user_id=other_user_id,
                scope_id=other_user_scope.id,
                conversation_id=other_user_conversation.id,
                role="assistant",
                content="Wrong user assistant response.",
            )

            with pytest.raises(ConversationNotFoundError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=query_message.id,
                    assistant_message_id=wrong_conversation_assistant.id,
                    used=results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            with pytest.raises(ConversationNotFoundError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=query_message.id,
                    assistant_message_id=wrong_scope_assistant.id,
                    used=results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            with pytest.raises(ConversationAccessDeniedError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=query_message.id,
                    assistant_message_id=wrong_user_assistant.id,
                    used=results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            async with pool.acquire() as connection:
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_record_used_memories_rejects_wrong_user_query_message() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            async with pool.acquire() as connection:
                other_user_id = await connection.fetchval(
                    "INSERT INTO users DEFAULT VALUES RETURNING id;"
                )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=other_user_id,
                    name="Other User Scope",
                    system_prompt="Private to the other user.",
                )
            )
            other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=other_user_id,
                    scope_id=other_scope.id,
                    title="Private query",
                )
            )
            other_query_message = await _append_message(
                service=service,
                user_id=other_user_id,
                scope_id=other_scope.id,
                conversation_id=other_conversation.id,
                role="user",
                content="Other user's query",
            )
            episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="valid local memory",
            )
            assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="assistant",
                content="Local assistant response.",
            )
            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=episode.content,
                top_k=1,
                now=FIXED_RETRIEVAL_NOW,
            )

            with pytest.raises(ConversationNotFoundError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=uuid4(),
                    assistant_message_id=assistant_message.id,
                    used=results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            with pytest.raises(ConversationAccessDeniedError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=other_query_message.id,
                    assistant_message_id=assistant_message.id,
                    used=results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            async with pool.acquire() as connection:
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_record_used_memories_rejects_wrong_scope_episode() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            other_scope = await service.create_scope(
                CreateScopeRequest(
                    user_id=user_id,
                    name="Other Scope",
                    system_prompt="Different memory partition.",
                )
            )
            other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=other_scope.id,
                    title="Other scope",
                )
            )
            query_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="user",
                content="Valid query",
            )
            assistant_message = await _append_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                role="assistant",
                content="Valid assistant response.",
            )
            other_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=other_scope.id,
                conversation_id=other_conversation.id,
                content="wrong scope memory",
            )
            other_results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=other_scope.id,
                query=other_episode.content,
                top_k=1,
                now=FIXED_RETRIEVAL_NOW,
            )

            with pytest.raises(InvalidRetrievalRequestError):
                await service.record_used_memories(
                    user_id=user_id,
                    scope_id=scope_id,
                    query_message_id=query_message.id,
                    assistant_message_id=assistant_message.id,
                    used=other_results,
                    retrieved_at=FIXED_RETRIEVAL_NOW,
                )

            async with pool.acquire() as connection:
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )

            assert retrieval_count == 0
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_orders_different_scores_by_score_desc() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query = "score ordering"
            low_score_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="low score",
            )
            high_score_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="high score",
            )
            query_vector = await service.embedder.embed_text(query)
            for episode_id, importance in [
                (low_score_episode.id, 0.0),
                (high_score_episode.id, 1.0),
            ]:
                await _set_episode_embedding(pool, episode_id, query_vector)
                await _set_episode_scoring_fields(
                    pool=pool,
                    episode_id=episode_id,
                    created_at=FIXED_RETRIEVAL_NOW,
                    last_accessed_at=None,
                    importance=importance,
                    access_count=0,
                )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=query,
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )

            assert [result.id for result in results] == [
                high_score_episode.id,
                low_score_episode.id,
            ]
            assert results[0].score > results[1].score
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_orders_equal_scores_by_created_at_desc() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )
            query = "created_at ordering"
            older_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="older equal score",
            )
            newer_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=scope_id,
                conversation_id=conversation_id,
                content="newer equal score",
            )
            query_vector = await service.embedder.embed_text(query)
            for episode_id, created_at in [
                (older_episode.id, FIXED_RETRIEVAL_NOW),
                (newer_episode.id, FIXED_RETRIEVAL_NOW + timedelta(hours=1)),
            ]:
                await _set_episode_embedding(pool, episode_id, query_vector)
                await _set_episode_scoring_fields(
                    pool=pool,
                    episode_id=episode_id,
                    created_at=created_at,
                    last_accessed_at=None,
                    importance=0.0,
                    access_count=0,
                )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query=query,
                top_k=2,
                now=FIXED_RETRIEVAL_NOW,
            )

            assert [result.id for result in results] == [
                newer_episode.id,
                older_episode.id,
            ]
            assert results[0].score == pytest.approx(results[1].score)
            assert results[0].created_at > results[1].created_at
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_orders_equal_scores_deterministically() -> None:
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
            user_id, scope_id, conversation_id = await _create_user_scope_conversation(
                service,
                pool,
            )

            episodes = [
                await _append_embedded_message(
                    service=service,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    content="identical retrieval content",
                )
                for _ in range(3)
            ]
            created_at = FIXED_RETRIEVAL_NOW - timedelta(hours=1)
            async with pool.acquire() as connection:
                for episode in episodes:
                    await connection.execute(
                        "UPDATE episodes SET created_at = $1 WHERE id = $2;",
                        created_at,
                        episode.id,
                    )

            results = await service.retrieve_scoped_episodes(
                user_id=user_id,
                scope_id=scope_id,
                query="identical retrieval content",
                top_k=3,
                now=FIXED_RETRIEVAL_NOW,
            )

            result_ids = [result.id for result in results]
            assert result_ids == sorted(result_ids)
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_rejects_non_768_query_embedding() -> None:
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
            user_id, scope_id, _ = await _create_user_scope_conversation(service, pool)

            with pytest.raises(VectorDimensionError):
                await service.retrieve_scoped_episodes(
                    user_id=user_id,
                    scope_id=scope_id,
                    query="wrong dimension",
                    top_k=5,
                    now=FIXED_RETRIEVAL_NOW,
                )
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_rejects_non_positive_top_k() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    with pytest.raises(InvalidRetrievalRequestError):
        await service.retrieve_scoped_episodes(
            user_id=uuid4(),
            scope_id=uuid4(),
            query="anything",
            top_k=0,
            now=FIXED_RETRIEVAL_NOW,
        )


@pytest.mark.asyncio
async def test_retrieve_scoped_episodes_rejects_naive_now_before_work() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=16))  # type: ignore[arg-type]

    with pytest.raises(InvalidRetrievalRequestError):
        await service.retrieve_scoped_episodes(
            user_id=uuid4(),
            scope_id=uuid4(),
            query="anything",
            top_k=1,
            now=datetime(2026, 1, 1, 12, 0),
        )


async def _create_user_scope_conversation(
    service: MemoryService,
    pool,
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
            title="Retrieval test",
        )
    )
    return user_id, scope.id, conversation.id


async def _append_embedded_message(
    service: MemoryService,
    user_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
    content: str,
):
    message = await service.append_message(
        AppendMessageRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            role="user",
            content=content,
            token_count=len(content.split()),
        )
    )
    return await service.create_message_episode(
        CreateMessageEpisodeRequest(
            user_id=user_id,
            scope_id=scope_id,
            conversation_id=conversation_id,
            message_id=message.id,
        )
    )


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
            role=role,
            content=content,
            token_count=len(content.split()),
        )
    )


class _RetrievalSqlSpyPool:
    def __init__(self, connection: _RetrievalSqlSpyConnection) -> None:
        self._connection = connection

    def acquire(self) -> _RetrievalSqlSpyConnectionContext:
        return _RetrievalSqlSpyConnectionContext(self._connection)


class _RetrievalSqlSpyConnection:
    def __init__(self, scope_owner_id: UUID) -> None:
        self._scope_owner_id = scope_owner_id
        self.fetch_queries: list[str] = []

    def transaction(self) -> _AsyncNullContext:
        return _AsyncNullContext()

    async def fetchval(self, query: str, *args: object) -> UUID | int:
        _ = args
        if "FROM scopes" in query:
            return self._scope_owner_id
        if "FROM embedding_models" in query:
            return 1
        raise AssertionError(f"Unexpected fetchval query: {query}")

    async def fetch(self, query: str, *args: object) -> list[object]:
        _ = args
        self.fetch_queries.append(query)
        return []


class _RetrievalSqlSpyConnectionContext:
    def __init__(self, connection: _RetrievalSqlSpyConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _RetrievalSqlSpyConnection:
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


def _scored_episode_stub(
    episode_id: UUID | None = None,
    result_rank: int = 1,
) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=result_rank,
        id=episode_id or uuid4(),
        user_id=uuid4(),
        scope_id=uuid4(),
        conversation_id=uuid4(),
        kind="message",
        message_id=uuid4(),
        message_position=1,
        range_start=None,
        range_end=None,
        content="stub memory",
        created_at=FIXED_RETRIEVAL_NOW,
        importance=0.0,
        access_count=0,
        last_accessed_at=None,
        embedding_model_id=1,
        similarity=1.0,
        recency_score=1.0,
        access_score=0.0,
        importance_score=0.0,
        frequency_score=0.0,
        score=0.75,
    )


def _scored_episode_for_retrieval(
    episode: EpisodeRecord,
    user_id: UUID,
    result_rank: int,
    similarity: float,
    recency_score: float,
    access_score: float,
    frequency_score: float,
    importance_score: float,
    score: float,
) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=result_rank,
        id=episode.id,
        user_id=user_id,
        scope_id=episode.scope_id,
        conversation_id=episode.conversation_id,
        kind="message",
        message_id=episode.message_id,
        message_position=None,
        range_start=None,
        range_end=None,
        content=episode.content,
        created_at=episode.created_at,
        importance=0.0,
        access_count=0,
        last_accessed_at=None,
        embedding_model_id=episode.embedding_model_id,
        similarity=similarity,
        recency_score=recency_score,
        access_score=access_score,
        importance_score=importance_score,
        frequency_score=frequency_score,
        score=score,
    )


async def _episode_access_metadata(
    pool,
    episode_ids: list[UUID],
) -> dict[UUID, tuple[int, datetime | None]]:
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT id, access_count, last_accessed_at
            FROM episodes
            WHERE id = ANY($1::uuid[])
            ORDER BY id ASC;
            """,
            episode_ids,
        )
    return {
        row["id"]: (
            row["access_count"],
            row["last_accessed_at"],
        )
        for row in rows
    }


async def _set_episode_scoring_fields(
    pool,
    episode_id: UUID,
    created_at: datetime,
    last_accessed_at: datetime | None,
    importance: float,
    access_count: int,
) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE episodes
            SET created_at = $1,
                last_accessed_at = $2,
                importance = $3,
                access_count = $4
            WHERE id = $5;
            """,
            created_at,
            last_accessed_at,
            importance,
            access_count,
            episode_id,
        )


async def _set_episode_embedding(
    pool,
    episode_id: UUID,
    vector: tuple[float, ...],
) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE embeddings_768
            SET embedding = $1
            WHERE episode_id = $2;
            """,
            list(vector),
            episode_id,
        )


def _near_vector(vector: tuple[float, ...], similarity: float) -> tuple[float, ...]:
    basis_index = next(index for index, value in enumerate(vector) if abs(value) < 0.9)
    basis = [0.0] * len(vector)
    basis[basis_index] = 1.0
    projection = vector[basis_index]
    orthogonal = [
        basis_value - projection * vector_value
        for basis_value, vector_value in zip(basis, vector, strict=True)
    ]
    orthogonal_norm = math.sqrt(sum(value * value for value in orthogonal))
    unit_orthogonal = tuple(value / orthogonal_norm for value in orthogonal)
    orthogonal_weight = math.sqrt(1.0 - similarity * similarity)
    return tuple(
        similarity * vector_value + orthogonal_weight * orthogonal_value
        for vector_value, orthogonal_value in zip(vector, unit_orthogonal, strict=True)
    )
