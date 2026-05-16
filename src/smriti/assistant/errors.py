from __future__ import annotations


class AssistantError(Exception):
    """Base class for assistant-generation failures."""


class InvalidAssistantRequestError(AssistantError):
    """Raised when an assistant-generation request cannot be accepted."""


class AssistantGenerationUnavailableError(AssistantError):
    """Raised when local assistant generation is temporarily unavailable."""


class AssistantGenerationFailedError(AssistantError):
    """Raised when local assistant generation fails unexpectedly."""
