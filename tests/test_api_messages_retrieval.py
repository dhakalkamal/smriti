from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from smriti.api import create_app
from smriti.api.dependencies import get_current_local_user_id, get_memory_service
from smriti.api.routes import messages as messages_routes
from smriti.api.routes import retrieval as retrieval_routes
from smriti.config import Settings
from smriti.db.migrate import apply_migrations
from smriti.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    EmbeddingVector,
    FakeEmbedder,
)
from smriti.memory import (
    AppendMessageWithEpisodeRequest,
    EmbeddingModelNotFoundError,
    EpisodeRecord,
    ListMessagesRequest,
    MemoryService,
    MessageEpisodeRecord,
    MessageRecord,
    ScoredEpisode,
    VectorDimensionError,
)

LOCAL_USER_ID = UUID("44444444-4444-4444-8444-444444444444")
OTHER_USER_ID = UUID("55555555-5555-4555-8555-555555555555")


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


def test_message_and_retrieval_routes_create_list_search_and_skip_provenance() -> None:
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
        asyncio.run(apply_migrations(settings=settings, migrations_dir=settings.migrations_dir))
        app = create_app(settings=settings, embedder=FakeEmbedder(dimensions=768))

        with TestClient(app) as client:
            scope = client.post(
                "/scopes",
                json={"name": "Research Notes", "system_prompt": "Keep retrieval scoped."},
            ).json()
            conversation = client.post(
                "/conversations",
                json={"scope_id": scope["id"], "title": "Paper trail"},
            ).json()

            empty_messages_response = client.get(f"/conversations/{conversation['id']}/messages")
            assert empty_messages_response.status_code == 200
            assert empty_messages_response.json() == []

            message_response = client.post(
                f"/conversations/{conversation['id']}/messages",
                json={
                    "role": "user",
                    "content": "Remember the family research notes.",
                    "token_count": 6,
                },
            )
            assert message_response.status_code == 201
            message = message_response.json()
            assert message["conversation_id"] == conversation["id"]
            assert message["role"] == "user"
            assert message["content"] == "Remember the family research notes."
            assert message["position"] == 1
            assert "episode_id" in message

            messages_response = client.get(f"/conversations/{conversation['id']}/messages")
            assert messages_response.status_code == 200
            assert [listed_message["id"] for listed_message in messages_response.json()] == [
                message["id"]
            ]

            retrieval_response = client.post(
                "/retrieval/search",
                json={
                    "scope_id": scope["id"],
                    "query": "family research notes",
                    "top_k": 5,
                },
            )
            assert retrieval_response.status_code == 200
            retrieval_results = retrieval_response.json()
            assert [result["id"] for result in retrieval_results] == [message["episode_id"]]
            assert retrieval_results[0]["content"] == message["content"]
            assert "embedding" not in retrieval_results[0]

        counts = asyncio.run(_conversation_storage_counts(settings, UUID(conversation["id"])))
        assert counts == {
            "messages": 1,
            "episodes": 1,
            "embeddings": 1,
            "message_retrievals": 0,
        }


def test_message_post_embedding_failure_does_not_partially_write() -> None:
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
        asyncio.run(apply_migrations(settings=settings, migrations_dir=settings.migrations_dir))
        app = create_app(
            settings=settings,
            embedder=FailingEmbedder(EmbeddingConnectionError("local embedder unavailable")),
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            scope = client.post(
                "/scopes",
                json={"name": "Research Notes", "system_prompt": ""},
            ).json()
            conversation = client.post(
                "/conversations",
                json={"scope_id": scope["id"], "title": "Paper trail"},
            ).json()

            response = client.post(
                f"/conversations/{conversation['id']}/messages",
                json={
                    "role": "user",
                    "content": "This should not partially write.",
                    "token_count": 6,
                },
            )

            assert response.status_code == 503

        counts = asyncio.run(_conversation_storage_counts(settings, UUID(conversation["id"])))
        assert counts["messages"] == 0
        assert counts["episodes"] == 0


def test_message_and_retrieval_routes_map_forbidden_missing_invalid_and_embedding_errors() -> None:
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
        asyncio.run(apply_migrations(settings=settings, migrations_dir=settings.migrations_dir))
        app = create_app(settings=settings, embedder=FakeEmbedder(dimensions=768))

        with TestClient(app) as client:
            local_scope = client.post(
                "/scopes",
                json={"name": "Research Notes", "system_prompt": ""},
            ).json()
            missing_conversation_id = uuid4()

            assert (
                client.get(f"/conversations/{missing_conversation_id}/messages").status_code == 404
            )
            assert (
                client.post(
                    f"/conversations/{missing_conversation_id}/messages",
                    json={"role": "user", "content": "Missing.", "token_count": 1},
                ).status_code
                == 404
            )

            other_scope_id, other_conversation_id = asyncio.run(
                _create_other_user_scope_conversation(settings)
            )
            assert client.get(f"/conversations/{other_conversation_id}/messages").status_code == 403
            assert (
                client.post(
                    f"/conversations/{other_conversation_id}/messages",
                    json={"role": "user", "content": "Forbidden.", "token_count": 1},
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/retrieval/search",
                    json={
                        "scope_id": str(other_scope_id),
                        "query": "forbidden",
                        "top_k": 1,
                    },
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/retrieval/search",
                    json={"scope_id": str(uuid4()), "query": "missing", "top_k": 1},
                ).status_code
                == 404
            )

            assert (
                client.post(
                    f"/conversations/{uuid4()}/messages",
                    json={"role": "bogus", "content": "Invalid.", "token_count": 1},
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/retrieval/search",
                    json={"scope_id": local_scope["id"], "query": "invalid", "top_k": 0},
                ).status_code
                == 400
            )


@pytest.mark.parametrize(
    "exc",
    [
        EmbeddingConnectionError("local embedder unavailable"),
        EmbeddingTimeoutError("local embedder timed out"),
        EmbeddingResponseError("local embedder returned an invalid response"),
    ],
)
def test_retrieval_route_maps_transient_embedding_failures_to_503(exc: Exception) -> None:
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
        asyncio.run(apply_migrations(settings=settings, migrations_dir=settings.migrations_dir))
        app = create_app(settings=settings, embedder=FakeEmbedder(dimensions=768))

        with TestClient(app) as client:
            scope = client.post(
                "/scopes",
                json={"name": "Research Notes", "system_prompt": ""},
            ).json()

        failing_app = create_app(settings=settings, embedder=FailingEmbedder(exc))
        with TestClient(failing_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/retrieval/search",
                json={"scope_id": scope["id"], "query": "embed failure", "top_k": 1},
            )

        assert response.status_code == 503


@pytest.mark.parametrize(
    "exc",
    [
        EmbeddingConfigurationError("invalid local embedder configuration"),
        EmbeddingModelNotFoundError("missing embedding model"),
        VectorDimensionError("embedding dimension mismatch"),
    ],
)
def test_retrieval_route_maps_internal_embedding_failures_to_500(exc: Exception) -> None:
    scope_id = uuid4()
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    app.dependency_overrides[get_memory_service] = lambda: cast(
        MemoryService,
        RaisingRetrievalService(exc),
    )
    app.dependency_overrides[get_current_local_user_id] = lambda: LOCAL_USER_ID
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/retrieval/search",
        json={"scope_id": str(scope_id), "query": "model failure", "top_k": 1},
    )

    assert response.status_code == 500


def test_message_and_retrieval_routes_delegate_to_memory_service_dependencies() -> None:
    local_user_id = uuid4()
    conversation_id = uuid4()
    scope_id = uuid4()
    message_id = uuid4()
    episode_id = uuid4()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017
    message = MessageRecord(
        id=message_id,
        conversation_id=conversation_id,
        position=1,
        role="user",
        content="Delegated memory.",
        token_count=2,
        created_at=now,
    )
    episode = EpisodeRecord(
        id=episode_id,
        conversation_id=conversation_id,
        scope_id=scope_id,
        message_id=message_id,
        content=message.content,
        created_at=now,
        embedding_model_id=1,
    )
    scored_episode = _scored_episode(
        episode_id=episode_id,
        scope_id=scope_id,
        conversation_id=conversation_id,
        message_id=message_id,
        content=message.content,
        created_at=now,
    )
    service = RecordingMemoryService(
        message=message,
        message_episode=MessageEpisodeRecord(message=message, episode=episode),
        retrieval_results=[scored_episode],
    )
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    app.dependency_overrides[get_memory_service] = lambda: cast(MemoryService, service)
    app.dependency_overrides[get_current_local_user_id] = lambda: local_user_id
    client = TestClient(app)

    assert client.get(f"/conversations/{conversation_id}/messages").status_code == 200
    assert (
        client.post(
            f"/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Delegated memory.", "token_count": 2},
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/retrieval/search",
            json={"scope_id": str(scope_id), "query": "Delegated memory.", "top_k": 3},
        ).status_code
        == 200
    )

    assert service.calls == [
        (
            "list_messages",
            ListMessagesRequest(user_id=local_user_id, conversation_id=conversation_id),
        ),
        (
            "append_message_with_episode",
            AppendMessageWithEpisodeRequest(
                user_id=local_user_id,
                conversation_id=conversation_id,
                role="user",
                content="Delegated memory.",
                token_count=2,
            ),
        ),
        ("retrieve_scoped_episodes", local_user_id, scope_id, "Delegated memory.", 3),
    ]


def test_message_and_retrieval_route_modules_do_not_import_asyncpg_or_sql() -> None:
    for module in [messages_routes, retrieval_routes]:
        source = inspect.getsource(module)
        assert "Depends" in source
        assert "asyncpg" not in source
        assert "SELECT " not in source
        assert "INSERT " not in source
        assert "UPDATE " not in source
        assert "DELETE " not in source


class FailingEmbedder:
    dimensions = 768

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def embed_text(self, text: str) -> EmbeddingVector:
        _ = text
        raise self.exc

    async def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        _ = texts
        raise self.exc


class RaisingRetrievalService:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def retrieve_scoped_episodes(
        self,
        user_id: UUID,
        scope_id: UUID,
        query: str,
        top_k: int,
    ) -> list[ScoredEpisode]:
        _ = (user_id, scope_id, query, top_k)
        raise self.exc


class RecordingMemoryService:
    def __init__(
        self,
        message: MessageRecord,
        message_episode: MessageEpisodeRecord,
        retrieval_results: list[ScoredEpisode],
    ) -> None:
        self.message = message
        self.message_episode = message_episode
        self.retrieval_results = retrieval_results
        self.calls: list[object] = []

    async def list_messages(self, request: ListMessagesRequest) -> list[MessageRecord]:
        self.calls.append(("list_messages", request))
        return [self.message]

    async def append_message_with_episode(
        self,
        request: AppendMessageWithEpisodeRequest,
    ) -> MessageEpisodeRecord:
        self.calls.append(("append_message_with_episode", request))
        return self.message_episode

    async def retrieve_scoped_episodes(
        self,
        user_id: UUID,
        scope_id: UUID,
        query: str,
        top_k: int,
    ) -> list[ScoredEpisode]:
        self.calls.append(("retrieve_scoped_episodes", user_id, scope_id, query, top_k))
        return self.retrieval_results


async def _conversation_storage_counts(
    settings: Settings,
    conversation_id: UUID,
) -> dict[str, int]:
    connection = await _connect(settings)
    try:
        row = await connection.fetchrow(
            """
            SELECT
                (
                    SELECT COUNT(*)
                    FROM messages
                    WHERE conversation_id = $1
                ) AS messages,
                (
                    SELECT COUNT(*)
                    FROM episodes
                    WHERE conversation_id = $1
                ) AS episodes,
                (
                    SELECT COUNT(*)
                    FROM embeddings_768
                    WHERE episode_id IN (
                        SELECT id
                        FROM episodes
                        WHERE conversation_id = $1
                    )
                ) AS embeddings,
                (
                    SELECT COUNT(*)
                    FROM message_retrievals
                ) AS message_retrievals;
            """,
            conversation_id,
        )
    finally:
        await connection.close()

    assert row is not None
    return {
        "messages": int(row["messages"]),
        "episodes": int(row["episodes"]),
        "embeddings": int(row["embeddings"]),
        "message_retrievals": int(row["message_retrievals"]),
    }


async def _create_other_user_scope_conversation(settings: Settings) -> tuple[UUID, UUID]:
    connection = await _connect(settings)
    try:
        await connection.execute(
            """
            INSERT INTO users (id)
            VALUES ($1)
            ON CONFLICT (id) DO NOTHING;
            """,
            OTHER_USER_ID,
        )
        scope_id = await connection.fetchval(
            """
            INSERT INTO scopes (user_id, name, system_prompt)
            VALUES ($1, $2, $3)
            RETURNING id;
            """,
            OTHER_USER_ID,
            "Other User Scope",
            "Private.",
        )
        conversation_id = await connection.fetchval(
            """
            INSERT INTO conversations (user_id, scope_id, title)
            VALUES ($1, $2, $3)
            RETURNING id;
            """,
            OTHER_USER_ID,
            scope_id,
            "Private conversation",
        )
    finally:
        await connection.close()

    return cast(UUID, scope_id), cast(UUID, conversation_id)


async def _connect(settings: Settings) -> asyncpg.Connection:
    return await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )


def _scored_episode(
    episode_id: UUID,
    scope_id: UUID,
    conversation_id: UUID,
    message_id: UUID,
    content: str,
    created_at: datetime,
) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=1,
        id=episode_id,
        user_id=uuid4(),
        scope_id=scope_id,
        conversation_id=conversation_id,
        kind="message",
        message_id=message_id,
        message_position=1,
        range_start=None,
        range_end=None,
        content=content,
        created_at=created_at,
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
