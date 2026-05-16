from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from smriti.memory import (
    ConversationRecord,
    MessageEpisodeRecord,
    MessageRecord,
    MessageRole,
    ScopeRecord,
    ScoredEpisode,
)


class HealthResponse(BaseModel):
    """Local API health status."""

    status: Literal["ok"]
    mode: Literal["local"]

    model_config = ConfigDict(extra="forbid")


class CreateScopeBody(BaseModel):
    """Request body for creating a local memory scope."""

    name: str = Field(min_length=1)
    system_prompt: str = ""

    model_config = ConfigDict(extra="forbid")


class ScopeResponse(BaseModel):
    """HTTP representation of a memory scope."""

    id: UUID
    name: str
    system_prompt: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: ScopeRecord) -> ScopeResponse:
        return cls(
            id=record.id,
            name=record.name,
            system_prompt=record.system_prompt,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class CreateConversationBody(BaseModel):
    """Request body for creating a conversation in an existing scope."""

    scope_id: UUID
    title: str | None = None

    model_config = ConfigDict(extra="forbid")


class ConversationResponse(BaseModel):
    """HTTP representation of a conversation."""

    id: UUID
    scope_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: ConversationRecord) -> ConversationResponse:
        return cls(
            id=record.id,
            scope_id=record.scope_id,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class CreateMessageBody(BaseModel):
    """Request body for appending a message and retrieval episode."""

    role: MessageRole
    content: str = Field(min_length=1)
    token_count: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


class MessageResponse(BaseModel):
    """HTTP representation of an immutable conversation message."""

    id: UUID
    conversation_id: UUID
    position: int
    role: MessageRole
    content: str
    token_count: int
    created_at: datetime

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: MessageRecord) -> MessageResponse:
        return cls(
            id=record.id,
            conversation_id=record.conversation_id,
            position=record.position,
            role=record.role,
            content=record.content,
            token_count=record.token_count,
            created_at=record.created_at,
        )


class CreatedMessageResponse(BaseModel):
    """HTTP representation of a message created with its retrieval episode."""

    id: UUID
    conversation_id: UUID
    position: int
    role: MessageRole
    content: str
    token_count: int
    created_at: datetime
    episode_id: UUID

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: MessageEpisodeRecord) -> CreatedMessageResponse:
        return cls(
            id=record.message.id,
            conversation_id=record.message.conversation_id,
            position=record.message.position,
            role=record.message.role,
            content=record.message.content,
            token_count=record.message.token_count,
            created_at=record.message.created_at,
            episode_id=record.episode.id,
        )


class RetrievalSearchBody(BaseModel):
    """Request body for scoped retrieval search."""

    scope_id: UUID
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)

    model_config = ConfigDict(extra="forbid")


class ScoredEpisodeResponse(BaseModel):
    """HTTP representation of one scored retrieval result."""

    result_rank: int
    id: UUID
    scope_id: UUID
    conversation_id: UUID
    kind: str
    message_id: UUID | None
    message_position: int | None
    range_start: int | None
    range_end: int | None
    content: str
    created_at: datetime
    importance: float
    access_count: int
    last_accessed_at: datetime | None
    embedding_model_id: int
    similarity: float
    recency_score: float
    access_score: float
    importance_score: float
    frequency_score: float
    score: float

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: ScoredEpisode) -> ScoredEpisodeResponse:
        return cls(
            result_rank=record.result_rank,
            id=record.id,
            scope_id=record.scope_id,
            conversation_id=record.conversation_id,
            kind=record.kind,
            message_id=record.message_id,
            message_position=record.message_position,
            range_start=record.range_start,
            range_end=record.range_end,
            content=record.content,
            created_at=record.created_at,
            importance=record.importance,
            access_count=record.access_count,
            last_accessed_at=record.last_accessed_at,
            embedding_model_id=record.embedding_model_id,
            similarity=record.similarity,
            recency_score=record.recency_score,
            access_score=record.access_score,
            importance_score=record.importance_score,
            frequency_score=record.frequency_score,
            score=record.score,
        )
