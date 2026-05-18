from __future__ import annotations

import asyncio
import re
from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from smriti.api import create_app
from smriti.api.dependencies import (
    ApiAppState,
    get_assistant_orchestrator,
    get_chat_generator,
    get_current_local_user_id,
    get_memory_service,
)
from smriti.assistant import AssistantOrchestrator
from smriti.chat import ChatGenerator, FakeChatGenerator
from smriti.config import Settings
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    InvalidProvenanceTargetError,
    MemoryService,
)

LOCAL_USER_ID = UUID("11111111-1111-4111-8111-111111111111")
EXPECTED_STAGE_7_6_ROUTES = {
    ("GET", "/health"),
    ("GET", "/scopes"),
    ("POST", "/scopes"),
    ("GET", "/conversations"),
    ("POST", "/conversations"),
    ("DELETE", "/conversations/{conversation_id}"),
    ("GET", "/conversations/{conversation_id}/messages"),
    ("POST", "/conversations/{conversation_id}/messages"),
    ("POST", "/retrieval/search"),
    ("POST", "/conversations/{conversation_id}/assistant-response"),
    ("POST", "/conversations/{conversation_id}/assistant-response/stream"),
}
DEFAULT_DOCUMENTATION_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)


def test_create_app_registers_only_stage_7_6_http_routes_without_binding_socket() -> None:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))

    route_paths = {getattr(route, "path", None) for route in app.routes}

    assert app.title == "Smriti Local API"
    assert _external_http_api_routes(app) == EXPECTED_STAGE_7_6_ROUTES
    assert DEFAULT_DOCUMENTATION_PATHS.isdisjoint(route_paths)


def test_default_fastapi_documentation_routes_are_disabled() -> None:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    client = TestClient(app, raise_server_exceptions=False)

    for path in DEFAULT_DOCUMENTATION_PATHS:
        response = client.get(path)

        assert response.status_code == 404


def test_cors_allows_only_documented_localhost_origins() -> None:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    client = TestClient(app, raise_server_exceptions=False)

    for origin in ["http://127.0.0.1:5173", "http://localhost:5173"]:
        response = client.options(
            "/scopes",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert response.headers["access-control-allow-origin"] != "*"
        assert _header_values(response.headers["access-control-allow-methods"]) == {
            "GET",
            "POST",
            "DELETE",
        }

    denied_response = client.options(
        "/scopes",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert denied_response.status_code == 400
    assert "access-control-allow-origin" not in denied_response.headers


def test_health_endpoint_returns_local_status() -> None:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "local"}


def test_error_handlers_map_access_denied_missing_and_invalid_provenance() -> None:
    app = create_app(settings=Settings(), embedder=FakeEmbedder(dimensions=768))

    @app.get("/_test/conversation-access-denied")
    async def conversation_access_denied_probe() -> None:
        raise ConversationAccessDeniedError("Conversation belongs to a different user")

    @app.get("/_test/conversation-not-found")
    async def conversation_not_found_probe() -> None:
        raise ConversationNotFoundError("Conversation does not exist")

    @app.get("/_test/invalid-provenance-target")
    async def invalid_provenance_target_probe() -> None:
        raise InvalidProvenanceTargetError("Retrieval provenance requires a user query message")

    client = TestClient(app, raise_server_exceptions=False)

    access_denied_response = client.get("/_test/conversation-access-denied")
    not_found_response = client.get("/_test/conversation-not-found")
    invalid_provenance_response = client.get("/_test/invalid-provenance-target")

    assert access_denied_response.status_code == 403
    assert not_found_response.status_code == 404
    assert invalid_provenance_response.status_code == 400


def test_lifespan_wires_dependencies_and_bootstraps_configured_user() -> None:
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
        embedder = FakeEmbedder(dimensions=768)
        chat_generator = FakeChatGenerator()
        app = create_app(settings=settings, embedder=embedder, chat_generator=chat_generator)

        @app.get("/_test/dependencies")
        async def dependency_probe(
            memory_service: Annotated[MemoryService, Depends(get_memory_service)],
            chat_generator: Annotated[ChatGenerator, Depends(get_chat_generator)],
            assistant_orchestrator: Annotated[
                AssistantOrchestrator,
                Depends(get_assistant_orchestrator),
            ],
            local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
        ) -> dict[str, str]:
            return {
                "local_user_id": str(local_user_id),
                "memory_service": type(memory_service).__name__,
                "chat_generator": type(chat_generator).__name__,
                "assistant_orchestrator": type(assistant_orchestrator).__name__,
            }

        with TestClient(app) as client:
            state = app.state.smriti
            assert isinstance(state, ApiAppState)
            assert state.settings is settings
            assert state.embedder is embedder
            assert state.memory_service.pool is state.pool
            assert state.memory_service.embedder is embedder
            assert state.chat_generator is chat_generator
            assert state.assistant_orchestrator.memory_service is state.memory_service
            assert state.assistant_orchestrator.chat_generator is chat_generator
            assert state.local_user_id == LOCAL_USER_ID

            response = client.get("/_test/dependencies")

        assert response.status_code == 200
        assert response.json() == {
            "local_user_id": str(LOCAL_USER_ID),
            "memory_service": "MemoryService",
            "chat_generator": "FakeChatGenerator",
            "assistant_orchestrator": "AssistantOrchestrator",
        }
        assert asyncio.run(_local_user_count(settings, LOCAL_USER_ID)) == 1

        with TestClient(app):
            pass

        assert asyncio.run(_local_user_count(settings, LOCAL_USER_ID)) == 1


async def _local_user_count(settings: Settings, local_user_id: UUID) -> int:
    connection = await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )
    try:
        count = await connection.fetchval(
            "SELECT COUNT(*) FROM users WHERE id = $1;",
            local_user_id,
        )
    finally:
        await connection.close()

    return int(count)


def _external_http_api_routes(app: FastAPI) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or set():
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.add((method, route.path))
    return routes


def _header_values(value: str) -> set[str]:
    return {part.strip() for part in value.split(",")}
