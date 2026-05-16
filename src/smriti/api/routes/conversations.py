from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from smriti.api.dependencies import get_current_local_user_id, get_memory_service
from smriti.api.schemas import ConversationResponse, CreateConversationBody
from smriti.memory import (
    CreateConversationRequest,
    ListConversationsRequest,
    MemoryService,
)

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    memory_service: Annotated[MemoryService, Depends(get_memory_service)],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> list[ConversationResponse]:
    """List conversations for the configured local user."""

    conversations = await memory_service.list_conversations(
        ListConversationsRequest(user_id=local_user_id)
    )
    return [ConversationResponse.from_record(conversation) for conversation in conversations]


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    body: CreateConversationBody,
    memory_service: Annotated[MemoryService, Depends(get_memory_service)],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> ConversationResponse:
    """Create a conversation in a local user's scope."""

    conversation = await memory_service.create_conversation(
        CreateConversationRequest(
            user_id=local_user_id,
            scope_id=body.scope_id,
            title=body.title,
        )
    )
    return ConversationResponse.from_record(conversation)
