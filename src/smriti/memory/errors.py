from __future__ import annotations


class MemoryServiceError(Exception):
    """Base class for memory service failures."""


class ScopeNotFoundError(MemoryServiceError):
    """Raised when a scope is missing or does not belong to the expected user."""


class ConversationNotFoundError(MemoryServiceError):
    """Raised when a conversation is missing from the expected user and scope."""


class EmbeddingModelNotFoundError(MemoryServiceError):
    """Raised when the configured embedding model is not registered in the database."""


class VectorDimensionError(MemoryServiceError):
    """Raised when an embedder returns a vector incompatible with the target table."""
