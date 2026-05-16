from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import asyncpg
from fastapi import FastAPI, Request

from smriti.config import Settings
from smriti.embeddings import Embedder
from smriti.memory import MemoryService


@dataclass(frozen=True)
class ApiAppState:
    """Typed runtime dependencies built once during FastAPI lifespan."""

    settings: Settings
    pool: asyncpg.Pool
    embedder: Embedder
    memory_service: MemoryService
    local_user_id: UUID


def set_api_state(app: FastAPI, state: ApiAppState | None) -> None:
    """Store or clear the typed Smriti state on FastAPI's dynamic state bag."""

    app.state.smriti = state


def get_api_state(request: Request) -> ApiAppState:
    """Return the initialized app state for request-scoped dependencies."""

    state = getattr(request.app.state, "smriti", None)
    if not isinstance(state, ApiAppState):
        raise RuntimeError("Smriti API state has not been initialized")
    return state


def get_app_settings(request: Request) -> Settings:
    """Return process settings for API handlers."""

    return get_api_state(request).settings


def get_database_pool(request: Request) -> asyncpg.Pool:
    """Return the initialized asyncpg pool for API handlers."""

    return get_api_state(request).pool


def get_embedder(request: Request) -> Embedder:
    """Return the configured local embedder for API handlers."""

    return get_api_state(request).embedder


def get_memory_service(request: Request) -> MemoryService:
    """Return the memory service singleton used by thin API routes."""

    return get_api_state(request).memory_service


def get_current_local_user_id(request: Request) -> UUID:
    """Return the configured local user identity for the single-user API."""

    return get_api_state(request).local_user_id
