from __future__ import annotations

from smriti.assistant.errors import (
    AssistantError,
    AssistantGenerationFailedError,
    AssistantGenerationUnavailableError,
    InvalidAssistantRequestError,
)
from smriti.assistant.models import (
    AssistantGenerationRequest,
    AssistantGenerationResult,
    PromptBuildRequest,
    PromptBuildResult,
)
from smriti.assistant.orchestrator import AssistantOrchestrator
from smriti.assistant.prompt_builder import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_RECENT_MESSAGE_LIMIT,
    FIXED_PRIVACY_INSTRUCTIONS,
    build_chat_request,
)

__all__ = [
    "AssistantError",
    "AssistantGenerationFailedError",
    "AssistantGenerationRequest",
    "AssistantGenerationResult",
    "AssistantGenerationUnavailableError",
    "AssistantOrchestrator",
    "DEFAULT_MAX_PROMPT_CHARS",
    "DEFAULT_RECENT_MESSAGE_LIMIT",
    "FIXED_PRIVACY_INSTRUCTIONS",
    "InvalidAssistantRequestError",
    "PromptBuildRequest",
    "PromptBuildResult",
    "build_chat_request",
]
