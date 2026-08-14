from __future__ import annotations

import math
import re
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from testcontainers.postgres import PostgresContainer

from smriti.assistant import (
    AssistantGenerationRequest,
    AssistantOrchestrator,
    TypedMemoryAdmissionConfig,
)
from smriti.chat import ChatResponse, FakeChatGenerator
from smriti.config import Settings
from smriti.db.client import close_pool, get_pool
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    AppendMessageWithEpisodeRequest,
    CreateConversationRequest,
    CreateScopeRequest,
    CreateSummaryEpisodeRequest,
    MemoryService,
)

SUMMARY_TEXT = "The user chose Terrafold as the studio name and agreed lease terms with Obafemi."
QUERY_TEXT = "Which studio name did the user choose?"


@pytest.mark.asyncio
async def test_summary_at_rank_nine_reaches_assembled_prompt_through_real_retrieval() -> None:
    """End-to-end: retrieve_scoped_episodes -> typed admission -> assembled prompt."""

    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        database_url = re.sub(
            r"^postgresql\+[^:]+://", "postgresql://", postgres.get_connection_url()
        )
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
                    name=f"Scope {uuid4()}",
                    system_prompt="Keep memory scoped.",
                )
            )
            history = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    title="History",
                )
            )

            history_episode_ids: list[UUID] = []
            for position in range(1, 13):
                record = await service.append_message_with_episode(
                    AppendMessageWithEpisodeRequest(
                        user_id=user_id,
                        conversation_id=history.id,
                        role="user",
                        content=f"history message {position} about studio planning",
                        token_count=6,
                    )
                )
                history_episode_ids.append(record.episode.id)

            summary = await service.create_summary_episode_for_next_uncovered_window(
                CreateSummaryEpisodeRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    conversation_id=history.id,
                    window_messages=12,
                ),
                FakeChatGenerator(
                    response=ChatResponse(
                        content=SUMMARY_TEXT,
                        model="fake-summary",
                        finish_reason="stop",
                    )
                ),
            )
            assert summary is not None

            active = await service.create_conversation(
                CreateConversationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    title="Active",
                )
            )
            query_record = await service.append_message_with_episode(
                AppendMessageWithEpisodeRequest(
                    user_id=user_id,
                    conversation_id=active.id,
                    role="user",
                    content=QUERY_TEXT,
                    token_count=7,
                )
            )

            # Plant similarities so exactly eight raw messages outrank the
            # summary (summary lands at retrieval rank 9) and four raw
            # messages sit far below it in the score spread.
            query_vector = await service.embedder.embed_text(QUERY_TEXT)
            for index, episode_id in enumerate(history_episode_ids[:8]):
                await _set_episode_embedding(
                    pool, episode_id, _near_vector(query_vector, 0.99 - index * 0.01)
                )
            for index, episode_id in enumerate(history_episode_ids[8:]):
                await _set_episode_embedding(
                    pool, episode_id, _near_vector(query_vector, 0.30 - index * 0.01)
                )
            await _set_episode_embedding(pool, summary.id, _near_vector(query_vector, 0.90))

            orchestrator = AssistantOrchestrator(
                memory_service=service,
                chat_generator=FakeChatGenerator(
                    response=ChatResponse(content="unused", model="fake-chat")
                ),
                memory_policy="typed_v1",
                typed_v1_memory_config=TypedMemoryAdmissionConfig(
                    total_limit=6,
                    raw_source_limit=4,
                    summary_source_limit=2,
                    assistant_derived_limit=0,
                ),
            )
            assembly = await orchestrator.prepare_generation_debug(
                AssistantGenerationRequest(
                    user_id=user_id,
                    scope_id=scope.id,
                    conversation_id=active.id,
                    query_message_id=query_record.message.id,
                    top_k=5,
                    max_prompt_chars=16000,
                    recent_message_limit=20,
                )
            )

            summary_decisions = [
                decision
                for decision in assembly.memory_admission_decisions
                if decision.memory.kind == "summary"
            ]
            assert len(summary_decisions) == 1
            summary_decision = summary_decisions[0]
            assert summary_decision.memory.id == summary.id
            assert summary_decision.memory.result_rank == 9
            assert summary_decision.lane == "summary_source"
            assert summary_decision.admitted is True
            assert summary_decision.admission_reason == "summary_source_quota"

            selected_ids = {memory.id for memory in assembly.prompt.selected_memories}
            assert summary.id in selected_ids
            prompt_text = "\n".join(
                message.content for message in assembly.prompt.chat_request.messages
            )
            assert SUMMARY_TEXT in prompt_text

            # The active query stays excluded from retrieval and appears in
            # the prompt exactly once, as the recent-context query message.
            assert assembly.excluded_message_ids == (query_record.message.id,)
            retrieved_ids = {memory.id for memory in assembly.retrieved_memories}
            assert query_record.episode.id not in retrieved_ids
            assert assembly.active_query_occurrences == 1
        finally:
            await close_pool()


async def _set_episode_embedding(pool, episode_id: UUID, vector: tuple[float, ...]) -> None:
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
