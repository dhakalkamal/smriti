from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from smriti.api.dependencies import get_current_local_user_id, get_memory_service
from smriti.api.schemas import CreatedMessageResponse, CreateMessageBody, MessageResponse
from smriti.memory import (
    AppendMessageWithEpisodeRequest,
    ListMessagesRequest,
    MemoryService,
)

router = APIRouter(tags=["messages"])


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(
    conversation_id: UUID,
    memory_service: Annotated[MemoryService, Depends(get_memory_service)],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> list[MessageResponse]:
    """List messages for a local user's conversation."""

    messages = await memory_service.list_messages(
        ListMessagesRequest(user_id=local_user_id, conversation_id=conversation_id)
    )
    return [MessageResponse.from_record(message) for message in messages]


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=CreatedMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_message(
    conversation_id: UUID,
    body: CreateMessageBody,
    memory_service: Annotated[MemoryService, Depends(get_memory_service)],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> CreatedMessageResponse:
    """Append a message and retrieval episode for a local user's conversation."""

    record = await memory_service.append_message_with_episode(
        AppendMessageWithEpisodeRequest(
            user_id=local_user_id,
            conversation_id=conversation_id,
            role=body.role,
            content=body.content,
            token_count=body.token_count,
        )
    )
    return CreatedMessageResponse.from_record(record)
