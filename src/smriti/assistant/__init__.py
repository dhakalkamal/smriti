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
    AssistantPromptAssembly,
    AssistantStreamDone,
    AssistantStreamError,
    AssistantStreamEvent,
    AssistantStreamPreparation,
    AssistantStreamStart,
    AssistantStreamToken,
    PromptBuildRequest,
    PromptBuildResult,
    RecentContextSelectionRequest,
    RecentContextSelectionResult,
)
from smriti.assistant.orchestrator import AssistantOrchestrator
from smriti.assistant.prompt_builder import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_RECENT_MESSAGE_LIMIT,
    FIXED_PRIVACY_INSTRUCTIONS,
    build_chat_request,
    select_recent_context,
)

__all__ = [
    "AssistantError",
    "AssistantGenerationFailedError",
    "AssistantGenerationRequest",
    "AssistantGenerationResult",
    "AssistantGenerationUnavailableError",
    "AssistantOrchestrator",
    "AssistantPromptAssembly",
    "AssistantStreamDone",
    "AssistantStreamError",
    "AssistantStreamEvent",
    "AssistantStreamPreparation",
    "AssistantStreamStart",
    "AssistantStreamToken",
    "DEFAULT_MAX_PROMPT_CHARS",
    "DEFAULT_RECENT_MESSAGE_LIMIT",
    "FIXED_PRIVACY_INSTRUCTIONS",
    "InvalidAssistantRequestError",
    "PromptBuildRequest",
    "PromptBuildResult",
    "RecentContextSelectionRequest",
    "RecentContextSelectionResult",
    "build_chat_request",
    "select_recent_context",
]
