from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from smriti.api.dependencies import get_current_local_user_id, get_memory_service
from smriti.api.schemas import RetrievalSearchBody, ScoredEpisodeResponse
from smriti.memory import MemoryService

router = APIRouter(tags=["retrieval"])


@router.post("/retrieval/search", response_model=list[ScoredEpisodeResponse])
async def search_retrieval(
    body: RetrievalSearchBody,
    memory_service: Annotated[MemoryService, Depends(get_memory_service)],
    local_user_id: Annotated[UUID, Depends(get_current_local_user_id)],
) -> list[ScoredEpisodeResponse]:
    """Search memories inside one local scope."""

    results = await memory_service.retrieve_scoped_episodes(
        user_id=local_user_id,
        scope_id=body.scope_id,
        query=body.query,
        top_k=body.top_k,
    )
    return [ScoredEpisodeResponse.from_record(result) for result in results]
