from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from smriti.chat import ChatRequest
from smriti.memory import MessageRecord, RetrievalCandidateMode, ScoredEpisode

MemoryPromptStyle = Literal["legacy", "typed_v1"]


@dataclass(frozen=True)
class AssistantGenerationRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID
    query_message_id: UUID
    top_k: int
    max_prompt_chars: int = 16000
    recent_message_limit: int = 20


@dataclass(frozen=True)
class PromptBuildRequest:
    scope_system_prompt: str
    retrieved_memories: tuple[ScoredEpisode, ...]
    recent_messages: tuple[MessageRecord, ...]
    query_message_id: UUID
    max_prompt_chars: int = 16000
    memory_prompt_style: MemoryPromptStyle = "legacy"
    overflow_memories: tuple[ScoredEpisode, ...] = ()


@dataclass(frozen=True)
class RecentContextSelectionRequest:
    scope_system_prompt: str
    recent_messages: tuple[MessageRecord, ...]
    query_message_id: UUID
    max_prompt_chars: int = 16000


@dataclass(frozen=True)
class RecentContextSelectionResult:
    active_query_message_id: UUID
    selected_recent_messages: tuple[MessageRecord, ...]
    selected_recent_message_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class PromptBuildResult:
    chat_request: ChatRequest
    selected_memories: tuple[ScoredEpisode, ...]
    selected_recent_messages: tuple[MessageRecord, ...]
    selected_recent_message_ids: tuple[UUID, ...]
    skipped_memories: tuple[ScoredEpisode, ...]
    overflow_selected_memories: tuple[ScoredEpisode, ...] = ()


@dataclass(frozen=True)
class MemoryAdmissionDecision:
    memory: ScoredEpisode
    lane: str
    admitted: bool
    admission_reason: str | None
    skip_reason: str | None


@dataclass(frozen=True)
class AssistantPromptAssembly:
    prompt: PromptBuildResult
    recent_context: RecentContextSelectionResult
    active_query_message_id: UUID
    excluded_message_ids: tuple[UUID, ...]
    retrieved_memories: tuple[ScoredEpisode, ...]
    memory_admission_decisions: tuple[MemoryAdmissionDecision, ...]
    memory_policy: str
    retrieval_candidate_mode: RetrievalCandidateMode
    prompt_message_order: tuple[str, ...]
    active_query_occurrences: int
    overflow_memories: tuple[ScoredEpisode, ...] = ()


@dataclass(frozen=True)
class AssistantGenerationResult:
    assistant_message: MessageRecord
    chat_model: str
    finish_reason: str | None
    used_memory_episode_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class AssistantStreamPreparation:
    request: AssistantGenerationRequest
    chat_request: ChatRequest
    selected_memories: tuple[ScoredEpisode, ...]
    used_memory_episode_ids: tuple[UUID, ...]
    chat_model: str


@dataclass(frozen=True)
class AssistantStreamStart:
    used_memory_episode_ids: tuple[UUID, ...]
    chat_model: str | None = None


@dataclass(frozen=True)
class AssistantStreamToken:
    text: str


@dataclass(frozen=True)
class AssistantStreamDone:
    assistant_message: MessageRecord
    chat_model: str
    finish_reason: str | None
    used_memory_episode_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class AssistantStreamError:
    code: str
    message: str


AssistantStreamEvent = (
    AssistantStreamStart | AssistantStreamToken | AssistantStreamDone | AssistantStreamError
)
