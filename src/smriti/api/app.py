from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smriti.api.dependencies import ApiAppState, set_api_state
from smriti.api.errors import register_error_handlers
from smriti.api.routes import (
    assistant_router,
    conversations_router,
    health_router,
    messages_router,
    retrieval_router,
    scopes_router,
)
from smriti.assistant import AssistantOrchestrator
from smriti.chat import ChatGenerator, OllamaChatGenerator
from smriti.config import Settings, get_settings
from smriti.db.client import close_pool, get_pool
from smriti.embeddings import Embedder, OllamaEmbedder
from smriti.memory import MemoryService, SummaryEpisodeMemoryScheduler

logger = logging.getLogger(__name__)

LOCAL_CORS_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")


def create_app(
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    chat_generator: ChatGenerator | None = None,
) -> FastAPI:
    """Create the local FastAPI app without binding a network socket."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings or get_settings()
        logger.info(
            "Smriti Ollama config resolved: chat_model=%s chat_num_ctx=%s embed_num_ctx=%s",
            resolved_settings.ollama_chat_model,
            resolved_settings.ollama_chat_num_ctx,
            resolved_settings.ollama_embed_num_ctx,
        )
        pool = await get_pool(resolved_settings)
        resolved_embedder = embedder or OllamaEmbedder(
            num_ctx=resolved_settings.ollama_embed_num_ctx,
        )
        memory_service = MemoryService(pool=pool, embedder=resolved_embedder)
        resolved_chat_generator = chat_generator or OllamaChatGenerator(
            model=resolved_settings.ollama_chat_model,
            base_url=resolved_settings.ollama_base_url,
            timeout_seconds=resolved_settings.ollama_chat_timeout_seconds,
            num_ctx=resolved_settings.ollama_chat_num_ctx,
        )
        assistant_orchestrator = AssistantOrchestrator(
            memory_service=memory_service,
            chat_generator=resolved_chat_generator,
        )
        summary_episode_memory_scheduler = SummaryEpisodeMemoryScheduler(
            memory_service=memory_service,
            chat_generator=resolved_chat_generator,
            enabled=resolved_settings.summary_episode_memory_enabled,
            window_messages=resolved_settings.summary_episode_window_messages,
        )
        local_user_id = await _ensure_local_user(pool, resolved_settings.local_user_id)

        set_api_state(
            app=app,
            state=ApiAppState(
                settings=resolved_settings,
                pool=pool,
                embedder=resolved_embedder,
                memory_service=memory_service,
                chat_generator=resolved_chat_generator,
                assistant_orchestrator=assistant_orchestrator,
                summary_episode_memory_scheduler=summary_episode_memory_scheduler,
                local_user_id=local_user_id,
            ),
        )

        try:
            yield
        finally:
            set_api_state(app=app, state=None)
            await close_pool()

    app = FastAPI(
        title="Smriti Local API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(LOCAL_CORS_ORIGINS),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(scopes_router)
    app.include_router(conversations_router)
    app.include_router(messages_router)
    app.include_router(retrieval_router)
    app.include_router(assistant_router)
    return app


async def _ensure_local_user(pool: asyncpg.Pool, local_user_id: UUID) -> UUID:
    """Create the configured local user row if it is not already present."""

    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO users (id)
            VALUES ($1)
            ON CONFLICT (id) DO NOTHING;
            """,
            local_user_id,
        )

    return local_user_id
