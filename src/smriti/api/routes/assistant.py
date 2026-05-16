from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from smriti.api.dependencies import get_assistant_orchestrator, get_current_local_user_id
from smriti.api.schemas import (
    AssistantGenerationResponse,
    CreateAssistantResponseBody,
    MessageResponse,
)
from smriti.assistant import AssistantGenerationRequest, AssistantOrchestrator

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
