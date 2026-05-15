from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

MessageRole = Literal["system", "user", "assistant"]
EpisodeKind = Literal["message", "summary"]


@dataclass(frozen=True)
class CreateScopeRequest:
    user_id: UUID
    name: str
    system_prompt: str


@dataclass(frozen=True)
class ListScopesRequest:
    user_id: UUID


@dataclass(frozen=True)
class CreateConversationRequest:
    user_id: UUID
    scope_id: UUID
    title: str | None = None


@dataclass(frozen=True)
class AppendMessageRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    token_count: int


@dataclass(frozen=True)
class CreateMessageEpisodeRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    message_id: UUID


@dataclass(frozen=True)
class ScopeRecord:
    id: UUID
    user_id: UUID
    name: str
    system_prompt: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationRecord:
    id: UUID
    user_id: UUID
    scope_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MessageRecord:
    id: UUID
    conversation_id: UUID
    position: int
    role: MessageRole
    content: str
    token_count: int
    created_at: datetime


@dataclass(frozen=True)
class EpisodeRecord:
    id: UUID
    conversation_id: UUID
    scope_id: UUID
    message_id: UUID
    content: str
    created_at: datetime
    embedding_model_id: int


@dataclass(frozen=True)
class ScoredEpisode:
    id: UUID
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    kind: EpisodeKind
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
