from __future__ import annotations

from fastapi import APIRouter

from smriti.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a minimal local readiness signal."""

    return HealthResponse(status="ok", mode="local")
