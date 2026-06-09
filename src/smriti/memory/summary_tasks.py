from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import UUID

from smriti.chat import ChatGenerator
from smriti.memory.errors import SummaryEpisodeMemoryError
from smriti.memory.models import CreateSummaryEpisodeRequest
from smriti.memory.service import MemoryService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummaryEpisodeMemoryScheduleRequest:
    user_id: UUID
    scope_id: UUID
    conversation_id: UUID


@dataclass
class SummaryEpisodeMemoryScheduler:
    """Retain fire-and-forget summary tasks until their background work finishes."""

    memory_service: MemoryService
    chat_generator: ChatGenerator
    enabled: bool
    window_messages: int
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)

    def schedule(self, request: SummaryEpisodeMemoryScheduleRequest) -> asyncio.Task[None] | None:
        """Schedule summary memory work without blocking the user stream."""

        if not self.enabled:
            return None

        task = asyncio.create_task(self._run(request))
        self._tasks.add(task)
        task.add_done_callback(self._discard_task)
        return task

    @property
    def pending_count(self) -> int:
        """Return the number of retained background tasks."""

        return len(self._tasks)

    async def drain(self) -> None:
        """Wait for currently retained tasks; useful for deterministic tests."""

        if not self._tasks:
            return
        await asyncio.gather(*tuple(self._tasks))

    async def _run(self, request: SummaryEpisodeMemoryScheduleRequest) -> None:
        try:
            await self.memory_service.create_summary_episode_for_latest_complete_window(
                CreateSummaryEpisodeRequest(
                    user_id=request.user_id,
                    scope_id=request.scope_id,
                    conversation_id=request.conversation_id,
                    window_messages=self.window_messages,
                ),
                self.chat_generator,
            )
        except SummaryEpisodeMemoryError as exc:
            _log_summary_failure(exc)
        except Exception as exc:
            logger.error(
                "summary_episode_memory_failed",
                extra={
                    "event": "summary_episode_memory_failed",
                    "failure_step": "unexpected",
                    "user_id": request.user_id,
                    "scope_id": request.scope_id,
                    "conversation_id": request.conversation_id,
                    "range_start": None,
                    "range_end": None,
                    "message_count": None,
                    "summary_model": _summary_model(self.chat_generator),
                    "embedding_model": self.memory_service.embedding_model_id,
                    "exception_type": type(exc).__name__,
                },
            )

    def _discard_task(self, task: asyncio.Future[None]) -> None:
        if isinstance(task, asyncio.Task):
            self._tasks.discard(task)


def _log_summary_failure(exc: SummaryEpisodeMemoryError) -> None:
    logger.error(
        "summary_episode_memory_failed",
        extra={
            "event": "summary_episode_memory_failed",
            "failure_step": exc.failure_step,
            "user_id": exc.user_id,
            "scope_id": exc.scope_id,
            "conversation_id": exc.conversation_id,
            "range_start": exc.range_start,
            "range_end": exc.range_end,
            "message_count": exc.message_count,
            "summary_model": exc.summary_model,
            "embedding_model": exc.embedding_model,
            "exception_type": exc.exception_type,
        },
    )


def _summary_model(chat_generator: ChatGenerator) -> str | None:
    try:
        return chat_generator.model
    except Exception:
        return None
