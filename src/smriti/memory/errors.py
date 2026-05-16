from __future__ import annotations


class MemoryServiceError(Exception):
    """Base class for memory service failures."""


class InvalidMemoryRequestError(MemoryServiceError):
    """Raised when a memory request is syntactically valid but cannot be accepted."""


class MemoryAccessDeniedError(MemoryServiceError):
    """Raised when a resource exists but belongs to a different user boundary."""


class ScopeAccessDeniedError(MemoryAccessDeniedError):
    """Raised when a scope belongs to a different user."""


class ConversationAccessDeniedError(MemoryAccessDeniedError):
    """Raised when a conversation belongs to a different user."""


class ScopeNotFoundError(MemoryServiceError):
    """Raised when a scope is missing."""


class ConversationNotFoundError(MemoryServiceError):
    """Raised when a conversation is missing from the expected user and scope."""


class InvalidRetrievalRequestError(MemoryServiceError):
    """Raised when a retrieval request cannot be executed safely."""


class InvalidProvenanceTargetError(MemoryServiceError):
    """Raised when retrieval provenance targets a message that cannot own it."""


class EmbeddingModelNotFoundError(MemoryServiceError):
    """Raised when the configured embedding model is not registered in the database."""


class VectorDimensionError(MemoryServiceError):
    """Raised when an embedder returns a vector incompatible with the target table."""
