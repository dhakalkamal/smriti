from __future__ import annotations

import asyncio
import re
import threading
from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

import smriti.api.app as api_app_module
from smriti.api import create_app
from smriti.api.dependencies import (
    ApiAppState,
    get_assistant_orchestrator,
    get_chat_generator,
    get_current_local_user_id,
    get_memory_service,
    get_summary_episode_memory_scheduler,
)
from smriti.assistant import AssistantOrchestrator
from smriti.chat import ChatGenerator, ChatRequest, ChatResponse, FakeChatGenerator
from smriti.config import Settings
from smriti.db.migrate import apply_migrations
from smriti.embeddings import FakeEmbedder
from smriti.memory import (
    AppendMessageRequest,
    AssistantMessageNotFoundError,
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    CreateConversationRequest,
    CreateScopeRequest,
    InvalidProvenanceTargetError,
    MemoryService,
    SummaryEpisodeMemoryScheduler,
    SummaryEpisodeMemoryScheduleRequest,
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
    ("GET", "/conversations/{conversation_id}/messages/{message_id}/retrievals"),
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

    @app.get("/_test/assistant-message-not-found")
    async def assistant_message_not_found_probe() -> None:
        raise AssistantMessageNotFoundError("Assistant message does not exist")

    @app.get("/_test/invalid-provenance-target")
    async def invalid_provenance_target_probe() -> None:
        raise InvalidProvenanceTargetError("Retrieval provenance requires a user query message")

    client = TestClient(app, raise_server_exceptions=False)

    access_denied_response = client.get("/_test/conversation-access-denied")
    not_found_response = client.get("/_test/conversation-not-found")
    assistant_not_found_response = client.get("/_test/assistant-message-not-found")
    invalid_provenance_response = client.get("/_test/invalid-provenance-target")

    assert access_denied_response.status_code == 403
    assert not_found_response.status_code == 404
    assert assistant_not_found_response.status_code == 404
    assert assistant_not_found_response.json() == {"detail": "Assistant message not found"}
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
            summary_episode_memory_scheduler: Annotated[
                SummaryEpisodeMemoryScheduler,
                Depends(get_summary_episode_memory_scheduler),
            ],
            local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
        ) -> dict[str, str]:
            return {
                "local_user_id": str(local_user_id),
                "memory_service": type(memory_service).__name__,
                "chat_generator": type(chat_generator).__name__,
                "assistant_orchestrator": type(assistant_orchestrator).__name__,
                "summary_episode_memory_scheduler": type(summary_episode_memory_scheduler).__name__,
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
            assert state.summary_episode_memory_scheduler.memory_service is state.memory_service
            assert state.summary_episode_memory_scheduler.chat_generator is chat_generator
            assert state.summary_episode_memory_scheduler.enabled is False
            assert state.summary_episode_memory_scheduler.window_messages == 12
            assert state.local_user_id == LOCAL_USER_ID

            response = client.get("/_test/dependencies")

        assert response.status_code == 200
        assert response.json() == {
            "local_user_id": str(LOCAL_USER_ID),
            "memory_service": "MemoryService",
            "chat_generator": "FakeChatGenerator",
            "assistant_orchestrator": "AssistantOrchestrator",
            "summary_episode_memory_scheduler": "SummaryEpisodeMemoryScheduler",
        }
        assert asyncio.run(_local_user_count(settings, LOCAL_USER_ID)) == 1

        with TestClient(app):
            pass

        assert asyncio.run(_local_user_count(settings, LOCAL_USER_ID)) == 1


def test_lifespan_drains_in_flight_summary_tasks_before_closing_pool(monkeypatch) -> None:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        settings = Settings(
            database_url=_to_asyncpg_dsn(postgres.get_connection_url()),
            local_user_id=LOCAL_USER_ID,
            summary_episode_memory_enabled=True,
        )
        asyncio.run(apply_migrations(settings=settings, migrations_dir=settings.migrations_dir))
        chat_generator = _ReleasableSummaryChatGenerator()
        app = create_app(
            settings=settings,
            embedder=FakeEmbedder(dimensions=768),
            chat_generator=chat_generator,
        )
        scheduled_conversation_id: UUID | None = None
        close_pool_summary_count: int | None = None
        close_pool_observed_finished = False
        shutdown_events: list[str] = []
        original_close_pool = api_app_module.close_pool

        async def recording_close_pool() -> None:
            nonlocal close_pool_observed_finished
            nonlocal close_pool_summary_count

            shutdown_events.append("close_pool")
            assert scheduled_conversation_id is not None
            close_pool_summary_count = await _summary_episode_count(
                settings,
                scheduled_conversation_id,
            )
            close_pool_observed_finished = chat_generator.finished.is_set()
            await original_close_pool()

        monkeypatch.setattr(api_app_module, "close_pool", recording_close_pool)

        @app.post("/_test/schedule-summary")
        async def schedule_summary(
            memory_service: Annotated[MemoryService, Depends(get_memory_service)],
            summary_episode_memory_scheduler: Annotated[
                SummaryEpisodeMemoryScheduler,
                Depends(get_summary_episode_memory_scheduler),
            ],
            local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
        ) -> dict[str, str]:
            scope = await memory_service.create_scope(
                CreateScopeRequest(
                    user_id=local_user_id,
                    name="Shutdown drain summary scope",
                    system_prompt="Keep shutdown summary memory scoped.",
                )
            )
            conversation = await memory_service.create_conversation(
                CreateConversationRequest(
                    user_id=local_user_id,
                    scope_id=scope.id,
                    title="Shutdown drain summary conversation",
                )
            )
            for position in range(1, 13):
                await memory_service.append_message(
                    AppendMessageRequest(
                        user_id=local_user_id,
                        scope_id=scope.id,
                        conversation_id=conversation.id,
                        role="user" if position % 2 else "assistant",
                        content=f"shutdown summary message {position}",
                        token_count=4,
                    )
                )

            task = summary_episode_memory_scheduler.schedule(
                SummaryEpisodeMemoryScheduleRequest(
                    user_id=local_user_id,
                    scope_id=scope.id,
                    conversation_id=conversation.id,
                )
            )
            if task is None:
                raise RuntimeError("summary task was not scheduled")

            for _ in range(100):
                if chat_generator.started.is_set():
                    break
                await asyncio.sleep(0.01)
            if not chat_generator.started.is_set():
                raise RuntimeError("summary task did not start")

            return {"conversation_id": str(conversation.id)}

        with TestClient(app) as client:
            state = app.state.smriti
            assert isinstance(state, ApiAppState)
            original_drain = state.summary_episode_memory_scheduler.drain

            async def release_then_drain() -> None:
                shutdown_events.append("drain")
                assert state.summary_episode_memory_scheduler.enabled is False
                chat_generator.release()
                await original_drain(timeout_seconds=5.0)

            state.summary_episode_memory_scheduler.drain = release_then_drain

            response = client.post("/_test/schedule-summary")

            assert response.status_code == 200
            scheduled_conversation_id = UUID(response.json()["conversation_id"])
            assert chat_generator.started.is_set()
            assert not chat_generator.finished.is_set()
            assert state.summary_episode_memory_scheduler.pending_count == 1

        assert shutdown_events == ["drain", "close_pool"]
        assert chat_generator.finished.is_set()
        assert close_pool_observed_finished is True
        assert close_pool_summary_count == 1
        assert asyncio.run(_summary_episode_count(settings, scheduled_conversation_id)) == 1


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


async def _summary_episode_count(settings: Settings, conversation_id: UUID) -> int:
    connection = await asyncpg.connect(
        dsn=settings.database_url,
        timeout=settings.database_connect_timeout,
        command_timeout=settings.database_command_timeout,
    )
    try:
        count = await connection.fetchval(
            """
            SELECT COUNT(*)
            FROM episodes
            WHERE conversation_id = $1
              AND kind = 'summary';
            """,
            conversation_id,
        )
    finally:
        await connection.close()

    return int(count)


class _ReleasableSummaryChatGenerator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.finished = threading.Event()
        self._release = threading.Event()

    @property
    def model(self) -> str:
        return "releasable-summary-chat-generator"

    async def generate(self, request: ChatRequest) -> ChatResponse:
        _ = request
        self.started.set()
        released = await asyncio.to_thread(self._release.wait, 5.0)
        if not released:
            raise RuntimeError("summary generator was not released")
        self.finished.set()
        return ChatResponse(
            content="summary written during lifespan shutdown drain",
            model=self.model,
            finish_reason="stop",
        )

    def release(self) -> None:
        self._release.set()


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
