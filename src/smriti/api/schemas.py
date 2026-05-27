from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from smriti.memory import (
    ConversationRecord,
    EpisodeKind,
    MessageEpisodeRecord,
    MessageRecord,
    MessageRole,
    RetrievalEpisodeSource,
    RetrievalQueryMessage,
    RetrievalRecord,
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


class CreateAssistantResponseBody(BaseModel):
    """Request body for creating a local assistant response."""

    scope_id: UUID
    query_message_id: UUID
    top_k: int = Field(default=5, ge=1, le=50)
    max_prompt_chars: int = Field(default=16000, ge=1, le=100000)
    recent_message_limit: int = Field(default=20, ge=1, le=200)

    model_config = ConfigDict(extra="forbid")


class AssistantGenerationResponse(BaseModel):
    """HTTP representation of a generated assistant response."""

    assistant_message: MessageResponse
    chat_model: str
    finish_reason: str | None
    used_memory_episode_ids: list[UUID]

    model_config = ConfigDict(extra="forbid")


class AssistantStreamStartData(BaseModel):
    """SSE payload for the start of assistant streaming."""

    used_memory_episode_ids: list[UUID]
    chat_model: str | None = None

    model_config = ConfigDict(extra="forbid")


class AssistantStreamTokenData(BaseModel):
    """SSE payload for one assistant token fragment."""

    text: str

    model_config = ConfigDict(extra="forbid")


class AssistantStreamDoneData(BaseModel):
    """SSE payload for a successfully persisted streaming assistant response."""

    assistant_message: MessageResponse
    chat_model: str
    finish_reason: str | None
    used_memory_episode_ids: list[UUID]

    model_config = ConfigDict(extra="forbid")


class AssistantStreamErrorData(BaseModel):
    """SSE payload for a post-stream-start assistant error."""

    code: str
    message: str

    model_config = ConfigDict(extra="forbid")


class RetrievalQuery(BaseModel):
    """HTTP representation of the query message that produced retrieval provenance."""

    message_id: UUID
    content: str

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: RetrievalQueryMessage) -> RetrievalQuery:
        return cls(message_id=record.message_id, content=record.content)


class RetrievalEpisode(BaseModel):
    """HTTP representation of the episode source used by a retrieval result."""

    id: UUID
    kind: EpisodeKind
    content: str
    source_conversation_id: UUID
    source_conversation_title: str | None
    source_scope_id: UUID
    source_scope_name: str

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: RetrievalEpisodeSource) -> RetrievalEpisode:
        return cls(
            id=record.id,
            kind=record.kind,
            content=record.content,
            source_conversation_id=record.source_conversation_id,
            source_conversation_title=record.source_conversation_title,
            source_scope_id=record.source_scope_id,
            source_scope_name=record.source_scope_name,
        )


class RetrievalEntry(BaseModel):
    """HTTP representation of one recorded retrieval provenance row."""

    rank: int
    similarity: float
    score: float
    recency_score: float
    access_score: float
    frequency_score: float
    importance_score: float
    scoring_version: str
    retrieved_at: datetime
    query: RetrievalQuery
    episode: RetrievalEpisode

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_record(cls, record: RetrievalRecord) -> RetrievalEntry:
        return cls(
            rank=record.rank,
            similarity=record.similarity,
            score=record.score,
            recency_score=record.recency_score,
            access_score=record.access_score,
            frequency_score=record.frequency_score,
            importance_score=record.importance_score,
            scoring_version=record.scoring_version,
            retrieved_at=record.retrieved_at,
            query=RetrievalQuery.from_record(record.query),
            episode=RetrievalEpisode.from_record(record.episode),
        )


class MessageRetrievalsResponse(BaseModel):
    """HTTP representation of recorded retrieval provenance for one assistant message."""

    assistant_message_id: UUID
    total: int
    retrievals: list[RetrievalEntry]

    model_config = ConfigDict(extra="forbid")


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
