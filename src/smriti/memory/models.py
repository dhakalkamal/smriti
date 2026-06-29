from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

MessageRole = Literal["system", "user", "assistant"]
EpisodeKind = Literal["message", "summary"]
RetrievalCandidateMode = Literal["semantic", "hybrid_v1"]


@dataclass(frozen=True)
class CreateScopeRequest:
    user_id: UUID
    name: str
    system_prompt: str


@dataclass(frozen=True)
class ListScopesRequest:
    user_id: UUID


@dataclass(frozen=True)
class ListConversationsRequest:
    user_id: UUID


@dataclass(frozen=True)
class ListMessagesRequest:
    user_id: UUID
    conversation_id: UUID
    limit: int = 100


@dataclass(frozen=True)
class LoadAssistantGenerationContextRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    query_message_id: UUID
    recent_message_limit: int = 20


@dataclass(frozen=True)
class CreateConversationRequest:
    user_id: UUID
    scope_id: UUID
    title: str | None = None


@dataclass(frozen=True)
class DeleteConversationRequest:
    user_id: UUID
    conversation_id: UUID


@dataclass(frozen=True)
class AppendMessageRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    token_count: int


@dataclass(frozen=True)
class AppendMessageWithEpisodeRequest:
    user_id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    token_count: int


@dataclass(frozen=True)
class AppendAssistantResponseWithProvenanceRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    query_message_id: UUID
    content: str
    token_count: int
    used: tuple[ScoredEpisode, ...]
    scoring_version: str | None = None
    retrieved_at: datetime | None = None


@dataclass(frozen=True)
class CreateMessageEpisodeRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    message_id: UUID


@dataclass(frozen=True)
class CreateSummaryEpisodeRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    window_messages: int


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
class MessageEpisodeRecord:
    message: MessageRecord
    episode: EpisodeRecord


@dataclass(frozen=True)
class SummaryEpisodeRecord:
    id: UUID
    conversation_id: UUID
    scope_id: UUID
    range_start: int
    range_end: int
    content: str
    created_at: datetime
    embedding_model_id: int


@dataclass(frozen=True)
class AssistantGenerationContextRecord:
    scope: ScopeRecord
    conversation: ConversationRecord
    query_message: MessageRecord
    recent_messages: tuple[MessageRecord, ...]


@dataclass(frozen=True)
class ScoredEpisode:
    result_rank: int
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
    message_role: MessageRole | None = None
    candidate_mode: RetrievalCandidateMode = "semantic"
    semantic_rank: int | None = None
    semantic_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    lexical_match_types: tuple[str, ...] = ()
    fused_rank: int | None = None
    fused_score: float | None = None


@dataclass(frozen=True)
class RetrievalQueryMessage:
    message_id: UUID
    content: str


@dataclass(frozen=True)
class RetrievalEpisodeSource:
    id: UUID
    kind: EpisodeKind
    content: str
    source_conversation_id: UUID
    source_conversation_title: str | None
    source_scope_id: UUID
    source_scope_name: str


@dataclass(frozen=True)
class RetrievalRecord:
    rank: int
    similarity: float
    score: float
    recency_score: float
    access_score: float
    frequency_score: float
    importance_score: float
    scoring_version: str
    retrieved_at: datetime
    query: RetrievalQueryMessage
    episode: RetrievalEpisodeSource


@dataclass(frozen=True)
class AssistantResponseRecord:
    message: MessageRecord
    used_episode_ids: tuple[UUID, ...]
    scoring_version: str
    retrieved_at: datetime
