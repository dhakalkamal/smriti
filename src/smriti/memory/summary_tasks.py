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
SUMMARY_EPISODE_DRAIN_TIMEOUT_SECONDS = 120.0


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
    _task_requests: dict[asyncio.Task[None], SummaryEpisodeMemoryScheduleRequest] = field(
        default_factory=dict,
        init=False,
    )

    def stop_accepting_tasks(self) -> None:
        """Prevent any new summary work from being scheduled."""

        self.enabled = False

    def schedule(self, request: SummaryEpisodeMemoryScheduleRequest) -> asyncio.Task[None] | None:
        """Schedule summary memory work without blocking the user stream."""

        if not self.enabled:
            return None

        task = asyncio.create_task(self._run(request))
        self._tasks.add(task)
        self._task_requests[task] = request
        task.add_done_callback(self._discard_task)
        return task

    @property
    def pending_count(self) -> int:
        """Return the number of retained background tasks."""

        return len(self._tasks)

    async def drain(self, timeout_seconds: float = SUMMARY_EPISODE_DRAIN_TIMEOUT_SECONDS) -> None:
        """Wait for retained tasks to finish, then cancel any that exceed the timeout."""

        if not self._tasks:
            return

        done, pending = await asyncio.wait(tuple(self._tasks), timeout=timeout_seconds)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        if not pending:
            return

        for task in pending:
            request = self._task_requests.get(task)
            if request is not None:
                _log_summary_task_failure(
                    request=request,
                    failure_step="shutdown_drain_timeout",
                    summary_model=_summary_model(self.chat_generator),
                    embedding_model=self.memory_service.embedding_model_id,
                    exception_type="CancelledError",
                )
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

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
            exception_message = str(exc)
            logger.error(
                _summary_failure_log_message(
                    failure_step="unexpected",
                    conversation_id=request.conversation_id,
                    range_start=None,
                    range_end=None,
                    message_count=None,
                    summary_model=_summary_model(self.chat_generator),
                    embedding_model=self.memory_service.embedding_model_id,
                    exception_type=type(exc).__name__,
                    exception_message=exception_message,
                ),
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
                    "exception_message": exception_message,
                },
            )

    def _discard_task(self, task: asyncio.Future[None]) -> None:
        if isinstance(task, asyncio.Task):
            self._tasks.discard(task)
            self._task_requests.pop(task, None)


def _log_summary_failure(exc: SummaryEpisodeMemoryError) -> None:
    exception_message = _exception_message(exc)
    logger.error(
        _summary_failure_log_message(
            failure_step=exc.failure_step,
            conversation_id=exc.conversation_id,
            range_start=exc.range_start,
            range_end=exc.range_end,
            message_count=exc.message_count,
            summary_model=exc.summary_model,
            embedding_model=exc.embedding_model,
            exception_type=exc.exception_type,
            exception_message=exception_message,
        ),
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
            "exception_message": exception_message,
        },
    )


def _log_summary_task_failure(
    *,
    request: SummaryEpisodeMemoryScheduleRequest,
    failure_step: str,
    summary_model: str | None,
    embedding_model: str | None,
    exception_type: str,
) -> None:
    exception_message = "summary task cancelled during shutdown drain"
    logger.error(
        _summary_failure_log_message(
            failure_step=failure_step,
            conversation_id=request.conversation_id,
            range_start=None,
            range_end=None,
            message_count=None,
            summary_model=summary_model,
            embedding_model=embedding_model,
            exception_type=exception_type,
            exception_message=exception_message,
        ),
        extra={
            "event": "summary_episode_memory_failed",
            "failure_step": failure_step,
            "user_id": request.user_id,
            "scope_id": request.scope_id,
            "conversation_id": request.conversation_id,
            "range_start": None,
            "range_end": None,
            "message_count": None,
            "summary_model": summary_model,
            "embedding_model": embedding_model,
            "exception_type": exception_type,
            "exception_message": exception_message,
        },
    )


def _summary_model(chat_generator: ChatGenerator) -> str | None:
    try:
        return chat_generator.model
    except Exception:
        return None


def _exception_message(exc: SummaryEpisodeMemoryError) -> str:
    if exc.__cause__ is not None:
        return str(exc.__cause__)
    return str(exc)


def _summary_failure_log_message(
    *,
    failure_step: str,
    conversation_id: UUID,
    range_start: int | None,
    range_end: int | None,
    message_count: int | None,
    summary_model: str | None,
    embedding_model: str | None,
    exception_type: str,
    exception_message: str,
) -> str:
    return (
        "summary_episode_memory_failed "
        f"failure_step={failure_step} "
        f"conversation_id={conversation_id} "
        f"range_start={_log_value(range_start)} "
        f"range_end={_log_value(range_end)} "
        f"message_count={_log_value(message_count)} "
        f"summary_model={_log_value(summary_model)} "
        f"embedding_model={_log_value(embedding_model)} "
        f"exception_type={exception_type} "
        f"exception_message={_log_value(exception_message)}"
    )


def _log_value(value: object | None) -> str:
    if value is None:
        return "null"
    return str(value)
