from __future__ import annotations

from smriti.api.routes.conversations import router as conversations_router
from smriti.api.routes.health import router as health_router
from smriti.api.routes.messages import router as messages_router
from smriti.api.routes.retrieval import router as retrieval_router
from smriti.api.routes.scopes import router as scopes_router

__all__ = [
    "conversations_router",
    "health_router",
    "messages_router",
    "retrieval_router",
    "scopes_router",
]
