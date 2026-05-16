from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest
from testcontainers.postgres import PostgresContainer

from smriti.config import Settings
from smriti.db.client import close_pool, get_pool
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    AppendMessageRequest,
    CreateConversationRequest,
    CreateMessageEpisodeRequest,
    CreateScopeRequest,
    EpisodeRecord,
    InvalidRetrievalRequestError,
    MemoryService,
    MessageRole,
    RetrievalEvalCase,
    run_retrieval_eval,
)

FIXED_EVAL_NOW = datetime(2026, 2, 1, 9, 30, tzinfo=timezone.utc)  # noqa: UP017


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


@pytest.mark.asyncio
async def test_retrieval_eval_computes_metrics_and_preserves_boundaries() -> None:
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
                    system_prompt="Keep this out of the eval scope.",
                )
            )
            other_conversation = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=other_scope.id,
                    title="Other scope",
                )
            )
            episodes = [
                await _append_embedded_message(
                    service=service,
                    user_id=user_id,
                    scope_id=scope_id,
                    conversation_id=conversation_id,
                    content=f"eval memory {index}",
                )
                for index in range(3)
            ]
            other_scope_episode = await _append_embedded_message(
                service=service,
                user_id=user_id,
                scope_id=other_scope.id,
                conversation_id=other_conversation.id,
                content="eval memory from the wrong scope",
            )

            query_vector = await service.embedder.embed_text("eval query")
            for index, episode in enumerate(episodes):
                await _set_episode_embedding(pool, episode.id, query_vector)
                await _set_episode_scoring_fields(
                    pool=pool,
                    episode_id=episode.id,
                    created_at=FIXED_EVAL_NOW - timedelta(minutes=index),
                    last_accessed_at=None,
                    importance=0.0,
                    access_count=0,
                )
            await _set_episode_embedding(pool, other_scope_episode.id, query_vector)
            await _set_episode_scoring_fields(
                pool=pool,
                episode_id=other_scope_episode.id,
                created_at=FIXED_EVAL_NOW + timedelta(minutes=1),
                last_accessed_at=None,
                importance=1.0,
                access_count=0,
            )

            results, summary = await run_retrieval_eval(
                service=service,
                cases=[
                    RetrievalEvalCase(
                        name="partial precision full recall",
                        user_id=user_id,
                        scope_id=scope_id,
                        query="eval query",
                        expected_episode_ids=(episodes[0].id, episodes[2].id),
                        top_k=3,
                    ),
                    RetrievalEvalCase(
                        name="miss at k",
                        user_id=user_id,
                        scope_id=scope_id,
                        query="eval query",
                        expected_episode_ids=(episodes[2].id,),
                        top_k=2,
                    ),
                ],
                now=FIXED_EVAL_NOW,
            )

            assert len(results) == 2
            assert results[0].retrieved_episode_ids == tuple(episode.id for episode in episodes)
            assert results[0].expected_episode_ids == (episodes[0].id, episodes[2].id)
            assert results[0].hit_at_k is True
            assert results[0].precision_at_k == pytest.approx(2 / 3)
            assert results[0].recall_at_k == pytest.approx(1.0)
            assert results[0].reciprocal_rank == pytest.approx(1.0)

            assert results[1].retrieved_episode_ids == (episodes[0].id, episodes[1].id)
            assert results[1].expected_episode_ids == (episodes[2].id,)
            assert results[1].hit_at_k is False
            assert results[1].precision_at_k == pytest.approx(0.0)
            assert results[1].recall_at_k == pytest.approx(0.0)
            assert results[1].reciprocal_rank == pytest.approx(0.0)

            retrieved_ids = {
                episode_id for result in results for episode_id in result.retrieved_episode_ids
            }
            assert other_scope_episode.id not in retrieved_ids
            assert summary.total_cases == 2
            assert summary.hit_rate_at_k == pytest.approx(0.5)
            assert summary.mean_precision_at_k == pytest.approx(1 / 3)
            assert summary.mean_recall_at_k == pytest.approx(0.5)
            assert summary.mean_reciprocal_rank == pytest.approx(0.5)

            async with pool.acquire() as connection:
                retrieval_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM message_retrievals;"
                )
                episode_rows = await connection.fetch(
                    """
                    SELECT id, access_count, last_accessed_at
                    FROM episodes
                    WHERE id = ANY($1::uuid[])
                    ORDER BY id ASC;
                    """,
                    [episode.id for episode in [*episodes, other_scope_episode]],
                )

            by_id = {row["id"]: row for row in episode_rows}
            assert retrieval_count == 0
            assert by_id[episodes[0].id]["access_count"] == 2
            assert by_id[episodes[1].id]["access_count"] == 2
            assert by_id[episodes[2].id]["access_count"] == 1
            assert by_id[episodes[0].id]["last_accessed_at"] == FIXED_EVAL_NOW
            assert by_id[episodes[1].id]["last_accessed_at"] == FIXED_EVAL_NOW
            assert by_id[episodes[2].id]["last_accessed_at"] == FIXED_EVAL_NOW
            assert by_id[other_scope_episode.id]["access_count"] == 0
            assert by_id[other_scope_episode.id]["last_accessed_at"] is None
        finally:
            await close_pool()


@pytest.mark.asyncio
async def test_retrieval_eval_rejects_empty_expected_episode_ids() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    with pytest.raises(InvalidRetrievalRequestError):
        await run_retrieval_eval(
            service=service,
            cases=[
                RetrievalEvalCase(
                    name="invalid",
                    user_id=UUID("00000000-0000-0000-0000-000000000001"),
                    scope_id=UUID("00000000-0000-0000-0000-000000000002"),
                    query="anything",
                    expected_episode_ids=(),
                    top_k=1,
                )
            ],
        )


@pytest.mark.asyncio
async def test_retrieval_eval_rejects_non_positive_top_k() -> None:
    service = MemoryService(pool=None, embedder=FakeEmbedder(dimensions=768))  # type: ignore[arg-type]

    with pytest.raises(InvalidRetrievalRequestError):
        await run_retrieval_eval(
            service=service,
            cases=[
                RetrievalEvalCase(
                    name="invalid",
                    user_id=UUID("00000000-0000-0000-0000-000000000001"),
                    scope_id=UUID("00000000-0000-0000-0000-000000000002"),
                    query="anything",
                    expected_episode_ids=(UUID("00000000-0000-0000-0000-000000000003"),),
                    top_k=0,
                )
            ],
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
            name="Eval Scope",
            system_prompt="Keep eval memory scoped.",
        )
    )
    conversation = await service.create_conversation(
        CreateConversationRequest(
            user_id=user_id,
            scope_id=scope.id,
            title="Eval",
        )
    )
    return user_id, scope.id, conversation.id


async def _append_embedded_message(
    service: MemoryService,
    user_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
    content: str,
) -> EpisodeRecord:
    message = await _append_message(
        service=service,
        user_id=user_id,
        scope_id=scope_id,
        conversation_id=conversation_id,
        role="user",
        content=content,
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
    role: MessageRole,
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
