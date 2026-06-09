from __future__ import annotations

from uuid import UUID


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


class AssistantMessageNotFoundError(MemoryServiceError):
    """Raised when an assistant message is missing from the expected conversation."""


class InvalidRetrievalRequestError(MemoryServiceError):
    """Raised when a retrieval request cannot be executed safely."""


class InvalidProvenanceTargetError(MemoryServiceError):
    """Raised when retrieval provenance targets a message that cannot own it."""


class EmbeddingModelNotFoundError(MemoryServiceError):
    """Raised when the configured embedding model is not registered in the database."""


class VectorDimensionError(MemoryServiceError):
    """Raised when an embedder returns a vector incompatible with the target table."""


class SummaryEpisodeMemoryError(MemoryServiceError):
    """Raised with content-free context when summary episode work fails."""

    def __init__(
        self,
        message: str,
        *,
        failure_step: str,
        user_id: UUID,
        scope_id: UUID,
        conversation_id: UUID,
        exception_type: str,
        range_start: int | None = None,
        range_end: int | None = None,
        message_count: int | None = None,
        summary_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_step = failure_step
        self.user_id = user_id
        self.scope_id = scope_id
        self.conversation_id = conversation_id
        self.range_start = range_start
        self.range_end = range_end
        self.message_count = message_count
        self.summary_model = summary_model
        self.embedding_model = embedding_model
        self.exception_type = exception_type
