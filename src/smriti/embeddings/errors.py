from __future__ import annotations


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedder is configured with unsupported settings."""


class EmbeddingConnectionError(EmbeddingError):
    """Raised when an embedding backend cannot be reached."""


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding backend exceeds the configured timeout."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when an embedding backend returns an invalid or failed response."""
