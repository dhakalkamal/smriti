from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from smriti.api.dependencies import get_current_local_user_id, get_memory_service
from smriti.api.schemas import CreateScopeBody, ScopeResponse
from smriti.memory import CreateScopeRequest, ListScopesRequest, MemoryService

router = APIRouter(tags=["scopes"])


@router.get("/scopes", response_model=list[ScopeResponse])
async def list_scopes(
    memory_service: Annotated[MemoryService, Depends(get_memory_service)],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> list[ScopeResponse]:
    """List scopes for the configured local user."""

    scopes = await memory_service.list_scopes(ListScopesRequest(user_id=local_user_id))
    return [ScopeResponse.from_record(scope) for scope in scopes]


@router.post("/scopes", response_model=ScopeResponse, status_code=status.HTTP_201_CREATED)
async def create_scope(
    body: CreateScopeBody,
    memory_service: Annotated[MemoryService, Depends(get_memory_service)],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> ScopeResponse:
    """Create a scope for the configured local user."""

    scope = await memory_service.create_scope(
        CreateScopeRequest(
            user_id=local_user_id,
            name=body.name,
            system_prompt=body.system_prompt,
        )
    )
    return ScopeResponse.from_record(scope)
