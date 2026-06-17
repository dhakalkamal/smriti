from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from smriti.assistant.errors import (
    AssistantGenerationFailedError,
    AssistantGenerationUnavailableError,
)
from smriti.assistant.models import (
    AssistantGenerationRequest,
    AssistantGenerationResult,
    AssistantPromptAssembly,
    AssistantStreamDone,
    AssistantStreamError,
    AssistantStreamEvent,
    AssistantStreamPreparation,
    AssistantStreamStart,
    AssistantStreamToken,
    PromptBuildRequest,
    PromptBuildResult,
    RecentContextSelectionRequest,
)
from smriti.assistant.prompt_builder import build_chat_request, select_recent_context
from smriti.chat import (
    ChatConfigurationError,
    ChatConnectionError,
    ChatGenerator,
    ChatResponseError,
    ChatStreamFinal,
    ChatStreamToken,
    ChatTimeoutError,
    StreamingChatGenerator,
)
from smriti.memory import (
    AppendAssistantResponseWithProvenanceRequest,
    AssistantResponseRecord,
    LoadAssistantGenerationContextRequest,
    MemoryService,
    ScoredEpisode,
)

ASSISTANT_GENERATION_UNAVAILABLE_CODE = "assistant_generation_unavailable"
ASSISTANT_GENERATION_FAILED_CODE = "assistant_generation_failed"
ASSISTANT_PERSISTENCE_FAILED_CODE = "assistant_persistence_failed"


@dataclass(frozen=True)
class AssistantOrchestrator:
    """Coordinate scoped retrieval, prompt construction, local chat, and persistence."""

    memory_service: MemoryService
    chat_generator: ChatGenerator

    async def generate(self, request: AssistantGenerationRequest) -> AssistantGenerationResult:
        """Generate and persist one assistant response for a stored user message."""

        assembly = await self._prepare_generation(request)
        prompt = assembly.prompt

        try:
            chat_response = await self.chat_generator.generate(prompt.chat_request)
        except (ChatConnectionError, ChatTimeoutError) as exc:
            message = "Local assistant generation unavailable"
            raise AssistantGenerationUnavailableError(message) from exc
        except (ChatConfigurationError, ChatResponseError) as exc:
            raise AssistantGenerationFailedError("Local assistant generation failed") from exc

        persisted = await self._persist_assistant_response(
            request=request,
            content=chat_response.content,
            completion_tokens=chat_response.usage.completion_tokens,
            selected_memories=prompt.selected_memories,
        )

        return AssistantGenerationResult(
            assistant_message=persisted.message,
            chat_model=chat_response.model,
            finish_reason=chat_response.finish_reason,
            used_memory_episode_ids=persisted.used_episode_ids,
        )

    async def prepare_stream(
        self,
        request: AssistantGenerationRequest,
    ) -> AssistantStreamPreparation:
        """Prepare a streaming response before the HTTP stream is opened."""

        assembly = await self._prepare_generation(request)
        prompt = assembly.prompt
        streaming_generator = self._streaming_chat_generator()
        return AssistantStreamPreparation(
            request=request,
            chat_request=prompt.chat_request,
            selected_memories=prompt.selected_memories,
            used_memory_episode_ids=tuple(memory.id for memory in prompt.selected_memories),
            chat_model=streaming_generator.model,
        )

    async def prepare_generation_debug(
        self,
        request: AssistantGenerationRequest,
    ) -> AssistantPromptAssembly:
        """Prepare assistant generation and expose content-free assembly boundaries."""

        return await self._prepare_generation(request)

    async def stream_prepared(
        self,
        prepared: AssistantStreamPreparation,
    ) -> AsyncIterator[AssistantStreamEvent]:
        """Stream a prepared assistant response and persist only after successful completion."""

        yield AssistantStreamStart(
            used_memory_episode_ids=prepared.used_memory_episode_ids,
            chat_model=prepared.chat_model,
        )

        content_parts: list[str] = []
        final_event: ChatStreamFinal | None = None
        streaming_generator = self._streaming_chat_generator()

        try:
            async for chat_event in streaming_generator.generate_stream(prepared.chat_request):
                if isinstance(chat_event, ChatStreamToken):
                    content_parts.append(chat_event.text)
                    yield AssistantStreamToken(text=chat_event.text)
                else:
                    final_event = chat_event
                    break
        except (ChatConnectionError, ChatTimeoutError):
            yield AssistantStreamError(
                code=ASSISTANT_GENERATION_UNAVAILABLE_CODE,
                message="Local assistant generation unavailable",
            )
            return
        except (ChatConfigurationError, ChatResponseError):
            yield AssistantStreamError(
                code=ASSISTANT_GENERATION_FAILED_CODE,
                message="Local assistant generation failed",
            )
            return

        if final_event is None:
            yield AssistantStreamError(
                code=ASSISTANT_GENERATION_FAILED_CODE,
                message="Local assistant generation failed",
            )
            return

        content = "".join(content_parts)
        try:
            persisted = await self._persist_assistant_response(
                request=prepared.request,
                content=content,
                completion_tokens=final_event.usage.completion_tokens,
                selected_memories=prepared.selected_memories,
            )
        except Exception:
            yield AssistantStreamError(
                code=ASSISTANT_PERSISTENCE_FAILED_CODE,
                message="Assistant response persistence failed",
            )
            return

        yield AssistantStreamDone(
            assistant_message=persisted.message,
            chat_model=final_event.model,
            finish_reason=final_event.finish_reason,
            used_memory_episode_ids=persisted.used_episode_ids,
        )

    async def _prepare_generation(
        self,
        request: AssistantGenerationRequest,
    ) -> AssistantPromptAssembly:
        context = await self.memory_service.load_assistant_generation_context(
            LoadAssistantGenerationContextRequest(
                user_id=request.user_id,
                scope_id=request.scope_id,
                conversation_id=request.conversation_id,
                query_message_id=request.query_message_id,
                recent_message_limit=request.recent_message_limit,
            )
        )
        recent_context = select_recent_context(
            RecentContextSelectionRequest(
                scope_system_prompt=context.scope.system_prompt,
                recent_messages=context.recent_messages,
                query_message_id=request.query_message_id,
                max_prompt_chars=request.max_prompt_chars,
            )
        )
        retrieved_memories = await self.memory_service.retrieve_scoped_episodes(
            user_id=request.user_id,
            scope_id=request.scope_id,
            query=context.query_message.content,
            top_k=request.top_k,
            exclude_message_ids=recent_context.selected_recent_message_ids,
        )
        prompt = build_chat_request(
            request=PromptBuildRequest(
                scope_system_prompt=context.scope.system_prompt,
                retrieved_memories=tuple(retrieved_memories),
                recent_messages=recent_context.selected_recent_messages,
                query_message_id=request.query_message_id,
                max_prompt_chars=request.max_prompt_chars,
            )
        )
        return AssistantPromptAssembly(
            prompt=prompt,
            recent_context=recent_context,
            active_query_message_id=recent_context.active_query_message_id,
            excluded_message_ids=recent_context.selected_recent_message_ids,
            retrieved_memories=tuple(retrieved_memories),
            prompt_message_order=_prompt_message_order(prompt),
            active_query_occurrences=sum(
                1
                for message in prompt.selected_recent_messages
                if message.id == recent_context.active_query_message_id
            ),
        )

    async def _persist_assistant_response(
        self,
        request: AssistantGenerationRequest,
        content: str,
        completion_tokens: int | None,
        selected_memories: tuple[ScoredEpisode, ...],
    ) -> AssistantResponseRecord:
        return await self.memory_service.append_assistant_response_with_provenance(
            AppendAssistantResponseWithProvenanceRequest(
                user_id=request.user_id,
                scope_id=request.scope_id,
                conversation_id=request.conversation_id,
                query_message_id=request.query_message_id,
                content=content,
                token_count=_assistant_token_count(content, completion_tokens),
                used=selected_memories,
            )
        )

    def _streaming_chat_generator(self) -> StreamingChatGenerator:
        if not isinstance(self.chat_generator, StreamingChatGenerator):
            raise AssistantGenerationFailedError("Local assistant generation failed")
        return self.chat_generator


def _assistant_token_count(content: str, completion_tokens: int | None) -> int:
    if completion_tokens is not None and completion_tokens > 0:
        return completion_tokens
    if content == "":
        return 0
    return max(1, len(content) // 4)


def _prompt_message_order(prompt: PromptBuildResult) -> tuple[str, ...]:
    return (
        "system:scope",
        "system:privacy",
        *(f"memory:{memory.id}" for memory in prompt.selected_memories),
        *(f"recent:{message.role}:{message.id}" for message in prompt.selected_recent_messages),
    )
