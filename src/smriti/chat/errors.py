from __future__ import annotations


class ChatError(Exception):
    """Base class for chat generation failures."""


class ChatConfigurationError(ChatError):
    """Raised when a chat generator is configured with unsupported settings."""


class ChatConnectionError(ChatError):
    """Raised when a chat backend cannot be reached."""


class ChatTimeoutError(ChatError):
    """Raised when a chat backend exceeds the configured timeout."""


class ChatResponseError(ChatError):
    """Raised when a chat backend returns an invalid or failed response."""
