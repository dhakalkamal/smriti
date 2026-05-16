from __future__ import annotations

from dataclasses import dataclass

from smriti.assistant.errors import (
    AssistantGenerationFailedError,
    AssistantGenerationUnavailableError,
)
from smriti.assistant.models import (
    AssistantGenerationRequest,
    AssistantGenerationResult,
    PromptBuildRequest,
)
from smriti.assistant.prompt_builder import build_chat_request
from smriti.chat import (
    ChatConfigurationError,
    ChatConnectionError,
    ChatGenerator,
    ChatResponseError,
    ChatTimeoutError,
)
from smriti.memory import (
    AppendAssistantResponseWithProvenanceRequest,
    LoadAssistantGenerationContextRequest,
    MemoryService,
)


@dataclass(frozen=True)
class AssistantOrchestrator:
    """Coordinate scoped retrieval, prompt construction, local chat, and persistence."""

    memory_service: MemoryService
    chat_generator: ChatGenerator

    async def generate(self, request: AssistantGenerationRequest) -> AssistantGenerationResult:
        """Generate and persist one assistant response for a stored user message."""

        context = await self.memory_service.load_assistant_generation_context(
            LoadAssistantGenerationContextRequest(
                user_id=request.user_id,
                scope_id=request.scope_id,
                conversation_id=request.conversation_id,
                query_message_id=request.query_message_id,
                recent_message_limit=request.recent_message_limit,
            )
        )
        retrieved_memories = await self.memory_service.retrieve_scoped_episodes(
            user_id=request.user_id,
            scope_id=request.scope_id,
            query=context.query_message.content,
            top_k=request.top_k,
        )
        prompt = build_chat_request(
            request=PromptBuildRequest(
                scope_system_prompt=context.scope.system_prompt,
                retrieved_memories=tuple(retrieved_memories),
                recent_messages=context.recent_messages,
                query_message_id=request.query_message_id,
                max_prompt_chars=request.max_prompt_chars,
            )
        )

        try:
            chat_response = await self.chat_generator.generate(prompt.chat_request)
        except (ChatConnectionError, ChatTimeoutError) as exc:
            message = "Local assistant generation unavailable"
            raise AssistantGenerationUnavailableError(message) from exc
        except (ChatConfigurationError, ChatResponseError) as exc:
            raise AssistantGenerationFailedError("Local assistant generation failed") from exc

        persisted = await self.memory_service.append_assistant_response_with_provenance(
            AppendAssistantResponseWithProvenanceRequest(
                user_id=request.user_id,
                scope_id=request.scope_id,
                conversation_id=request.conversation_id,
                query_message_id=request.query_message_id,
                content=chat_response.content,
                token_count=_assistant_token_count(
                    chat_response.content,
                    chat_response.usage.completion_tokens,
                ),
                used=prompt.selected_memories,
            )
        )

        return AssistantGenerationResult(
            assistant_message=persisted.message,
            chat_model=chat_response.model,
            finish_reason=chat_response.finish_reason,
            used_memory_episode_ids=persisted.used_episode_ids,
        )


def _assistant_token_count(content: str, completion_tokens: int | None) -> int:
    if completion_tokens is not None and completion_tokens > 0:
        return completion_tokens
    if content == "":
        return 0
    return max(1, len(content) // 4)
