from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from smriti.api.dependencies import get_assistant_orchestrator, get_current_local_user_id
from smriti.api.schemas import (
    AssistantGenerationResponse,
    AssistantStreamDoneData,
    AssistantStreamErrorData,
    AssistantStreamStartData,
    AssistantStreamTokenData,
    CreateAssistantResponseBody,
    MessageResponse,
)
from smriti.assistant import (
    AssistantGenerationRequest,
    AssistantOrchestrator,
    AssistantStreamDone,
    AssistantStreamError,
    AssistantStreamEvent,
    AssistantStreamStart,
    AssistantStreamToken,
)

router = APIRouter(tags=["assistant"])


@router.post(
    "/conversations/{conversation_id}/assistant-response",
    response_model=AssistantGenerationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_assistant_response(
    conversation_id: UUID,
    body: CreateAssistantResponseBody,
    assistant_orchestrator: Annotated[
        AssistantOrchestrator,
        Depends(get_assistant_orchestrator),
    ],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> AssistantGenerationResponse:
    """Generate and persist one assistant response for a local conversation."""

    result = await assistant_orchestrator.generate(
        AssistantGenerationRequest(
            user_id=local_user_id,
            scope_id=body.scope_id,
            conversation_id=conversation_id,
            query_message_id=body.query_message_id,
            top_k=body.top_k,
            max_prompt_chars=body.max_prompt_chars,
            recent_message_limit=body.recent_message_limit,
        )
    )
    return AssistantGenerationResponse(
        assistant_message=MessageResponse.from_record(result.assistant_message),
        chat_model=result.chat_model,
        finish_reason=result.finish_reason,
        used_memory_episode_ids=list(result.used_memory_episode_ids),
    )


@router.post("/conversations/{conversation_id}/assistant-response/stream")
async def stream_assistant_response(
    request: Request,
    conversation_id: UUID,
    body: CreateAssistantResponseBody,
    assistant_orchestrator: Annotated[
        AssistantOrchestrator,
        Depends(get_assistant_orchestrator),
    ],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> StreamingResponse:
    """Stream and persist one assistant response for a local conversation."""

    prepared = await assistant_orchestrator.prepare_stream(
        AssistantGenerationRequest(
            user_id=local_user_id,
            scope_id=body.scope_id,
            conversation_id=conversation_id,
            query_message_id=body.query_message_id,
            top_k=body.top_k,
            max_prompt_chars=body.max_prompt_chars,
            recent_message_limit=body.recent_message_limit,
        )
    )
    return StreamingResponse(
        _stream_sse_events(
            request=request,
            events=assistant_orchestrator.stream_prepared(prepared),
        ),
        media_type="text/event-stream",
    )


async def _stream_sse_events(
    request: Request,
    events: AsyncIterator[AssistantStreamEvent],
) -> AsyncIterator[str]:
    try:
        while not await request.is_disconnected():
            try:
                event = await anext(events)
            except StopAsyncIteration:
                break
            yield _encode_sse_event(event)
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()


def _encode_sse_event(event: AssistantStreamEvent) -> str:
    if isinstance(event, AssistantStreamStart):
        return _format_sse_event(
            "start",
            AssistantStreamStartData(
                used_memory_episode_ids=list(event.used_memory_episode_ids),
                chat_model=event.chat_model,
            ).model_dump_json(),
        )
    if isinstance(event, AssistantStreamToken):
        return _format_sse_event(
            "token",
            AssistantStreamTokenData(text=event.text).model_dump_json(),
        )
    if isinstance(event, AssistantStreamDone):
        return _format_sse_event(
            "done",
            AssistantStreamDoneData(
                assistant_message=MessageResponse.from_record(event.assistant_message),
                chat_model=event.chat_model,
                finish_reason=event.finish_reason,
                used_memory_episode_ids=list(event.used_memory_episode_ids),
            ).model_dump_json(),
        )
    if isinstance(event, AssistantStreamError):
        return _format_sse_event(
            "error",
            AssistantStreamErrorData(code=event.code, message=event.message).model_dump_json(),
        )
    raise TypeError(f"Unsupported assistant stream event: {type(event)!r}")


def _format_sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
