from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from smriti.chat import ChatRequest
from smriti.memory import MessageRecord, ScoredEpisode


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


@dataclass(frozen=True)
class PromptBuildResult:
    chat_request: ChatRequest
    selected_memories: tuple[ScoredEpisode, ...]


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
