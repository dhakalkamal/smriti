from __future__ import annotations

import asyncio
import inspect
import re
from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

import asyncpg
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from smriti.api import create_app
from smriti.api.dependencies import get_current_local_user_id, get_memory_service
from smriti.api.routes import conversations as conversations_routes
from smriti.api.routes import scopes as scopes_routes
from smriti.config import Settings
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    ConversationRecord,
    CreateConversationRequest,
    CreateScopeRequest,
    DeleteConversationRequest,
    ListConversationsRequest,
    ListScopesRequest,
    MemoryService,
    ScopeRecord,
)

LOCAL_USER_ID = UUID("22222222-2222-4222-8222-222222222222")
OTHER_USER_ID = UUID("33333333-3333-4333-8333-333333333333")


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


def test_scope_and_conversation_routes_create_and_list_local_records() -> None:
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
            assert client.get("/scopes").json() == []

            scope_response = client.post(
                "/scopes",
                json={
                    "name": "Research Notes",
                    "system_prompt": "Keep research memory scoped.",
                },
            )
            assert scope_response.status_code == 201
            scope = scope_response.json()
            assert scope["name"] == "Research Notes"
            assert scope["system_prompt"] == "Keep research memory scoped."
            assert "user_id" not in scope

            scopes_response = client.get("/scopes")
            assert scopes_response.status_code == 200
            assert [listed_scope["id"] for listed_scope in scopes_response.json()] == [scope["id"]]

            assert client.get("/conversations").json() == []

            conversation_response = client.post(
                "/conversations",
                json={"scope_id": scope["id"], "title": "Paper trail"},
            )
            assert conversation_response.status_code == 201
            conversation = conversation_response.json()
            assert conversation["scope_id"] == scope["id"]
            assert conversation["title"] == "Paper trail"
            assert "user_id" not in conversation

            conversations_response = client.get("/conversations")
            assert conversations_response.status_code == 200
            assert [
                listed_conversation["id"] for listed_conversation in conversations_response.json()
            ] == [conversation["id"]]

            delete_response = client.delete(f"/conversations/{conversation['id']}")
            assert delete_response.status_code == 204
            assert delete_response.content == b""
            assert client.get("/conversations").json() == []

        assert asyncio.run(_scope_owner_id(settings, UUID(scope["id"]))) == LOCAL_USER_ID
        assert asyncio.run(_conversation_exists(settings, UUID(conversation["id"]))) is False


def test_scope_and_conversation_routes_map_invalid_forbidden_and_missing_cases() -> None:
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
            invalid_scope_response = client.post(
                "/scopes",
                json={"name": "", "system_prompt": "No empty scope names."},
            )
            assert invalid_scope_response.status_code == 400

            first_scope_response = client.post(
                "/scopes",
                json={"name": "Research Notes", "system_prompt": ""},
            )
            assert first_scope_response.status_code == 201

            duplicate_scope_response = client.post(
                "/scopes",
                json={"name": "Research Notes", "system_prompt": "Duplicate."},
            )
            assert duplicate_scope_response.status_code == 400

            missing_scope_response = client.post(
                "/conversations",
                json={"scope_id": str(uuid4()), "title": "Missing scope"},
            )
            assert missing_scope_response.status_code == 404

            other_scope_id = asyncio.run(_create_other_user_scope(settings))
            forbidden_scope_response = client.post(
                "/conversations",
                json={"scope_id": str(other_scope_id), "title": "Wrong owner"},
            )
            assert forbidden_scope_response.status_code == 403

            missing_conversation_response = client.delete(f"/conversations/{uuid4()}")

            other_conversation_id = asyncio.run(_create_other_user_conversation(settings))
            forbidden_conversation_response = client.delete(
                f"/conversations/{other_conversation_id}"
            )

            assert missing_conversation_response.status_code == 404
            assert forbidden_conversation_response.status_code == 404
            assert missing_conversation_response.json() == forbidden_conversation_response.json()
            assert missing_conversation_response.json() == {"detail": "Conversation not found"}
            assert asyncio.run(_conversation_exists(settings, other_conversation_id)) is True


def test_scope_and_conversation_routes_delegate_to_memory_service_dependencies() -> None:
    local_user_id = uuid4()
    scope_id = uuid4()
    conversation_id = uuid4()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017
    service = RecordingMemoryService(
        scope=ScopeRecord(
            id=scope_id,
            user_id=local_user_id,
            name="Research Notes",
            system_prompt="Scoped.",
            created_at=now,
            updated_at=now,
        ),
        conversation=ConversationRecord(
            id=conversation_id,
            user_id=local_user_id,
            scope_id=scope_id,
            title="Paper trail",
            created_at=now,
            updated_at=now,
        ),
    )
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    app.dependency_overrides[get_memory_service] = lambda: cast(MemoryService, service)
    app.dependency_overrides[get_current_local_user_id] = lambda: local_user_id

    client = TestClient(app)

    assert client.get("/scopes").status_code == 200
    assert (
        client.post(
            "/scopes",
            json={"name": "Research Notes", "system_prompt": "Scoped."},
        ).status_code
        == 201
    )
    assert client.get("/conversations").status_code == 200
    assert (
        client.post(
            "/conversations",
            json={"scope_id": str(scope_id), "title": "Paper trail"},
        ).status_code
        == 201
    )
    assert client.delete(f"/conversations/{conversation_id}").status_code == 204

    assert service.calls == [
        ("list_scopes", ListScopesRequest(user_id=local_user_id)),
        (
            "create_scope",
            CreateScopeRequest(
                user_id=local_user_id,
                name="Research Notes",
                system_prompt="Scoped.",
            ),
        ),
        ("list_conversations", ListConversationsRequest(user_id=local_user_id)),
        (
            "create_conversation",
            CreateConversationRequest(
                user_id=local_user_id,
                scope_id=scope_id,
                title="Paper trail",
            ),
        ),
        (
            "delete_conversation",
            DeleteConversationRequest(
                user_id=local_user_id,
                conversation_id=conversation_id,
            ),
        ),
    ]


def test_scope_and_conversation_route_modules_do_not_import_asyncpg_or_sql() -> None:
    for module in [scopes_routes, conversations_routes]:
        source = inspect.getsource(module)
        assert "Depends" in source
        assert "asyncpg" not in source
        assert "SELECT " not in source
        assert "INSERT " not in source
        assert "UPDATE " not in source
        assert "DELETE " not in source


class RecordingMemoryService:
    def __init__(self, scope: ScopeRecord, conversation: ConversationRecord) -> None:
        self.scope = scope
        self.conversation = conversation
        self.calls: list[
            tuple[
                str,
                ListScopesRequest
                | CreateScopeRequest
                | ListConversationsRequest
                | CreateConversationRequest
                | DeleteConversationRequest,
            ]
        ] = []

    async def list_scopes(self, request: ListScopesRequest) -> list[ScopeRecord]:
        self.calls.append(("list_scopes", request))
        return [self.scope]

    async def create_scope(self, request: CreateScopeRequest) -> ScopeRecord:
        self.calls.append(("create_scope", request))
        return self.scope

    async def list_conversations(
        self,
        request: ListConversationsRequest,
    ) -> list[ConversationRecord]:
        self.calls.append(("list_conversations", request))
        return [self.conversation]

    async def create_conversation(
        self,
        request: CreateConversationRequest,
    ) -> ConversationRecord:
        self.calls.append(("create_conversation", request))
        return self.conversation

    async def delete_conversation(
        self,
        request: DeleteConversationRequest,
    ) -> None:
        self.calls.append(("delete_conversation", request))


async def _scope_owner_id(settings: Settings, scope_id: UUID) -> UUID:
    connection = await _connect(settings)
    try:
        value = await connection.fetchval(
            "SELECT user_id FROM scopes WHERE id = $1;",
            scope_id,
        )
    finally:
        await connection.close()

    return cast(UUID, value)


async def _conversation_owner_id(settings: Settings, conversation_id: UUID) -> UUID:
    connection = await _connect(settings)
    try:
        value = await connection.fetchval(
            "SELECT user_id FROM conversations WHERE id = $1;",
            conversation_id,
        )
    finally:
        await connection.close()

    return cast(UUID, value)


async def _conversation_exists(settings: Settings, conversation_id: UUID) -> bool:
    connection = await _connect(settings)
    try:
        value = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM conversations
                WHERE id = $1
            );
            """,
            conversation_id,
        )
    finally:
        await connection.close()

    return cast(bool, value)


async def _create_other_user_scope(settings: Settings) -> UUID:
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
    finally:
        await connection.close()

    return cast(UUID, scope_id)


async def _create_other_user_conversation(settings: Settings) -> UUID:
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
            f"Other User Scope {uuid4()}",
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
            "Other user conversation",
        )
    finally:
        await connection.close()

    return cast(UUID, conversation_id)


async def _connect(settings: Settings) -> asyncpg.Connection:
    return await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )
