from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from smriti.memory.errors import InvalidRetrievalRequestError
from smriti.memory.service import MemoryService


@dataclass(frozen=True)
class RetrievalEvalCase:
    name: str
    user_id: UUID
    scope_id: UUID
    query: str
    expected_episode_ids: tuple[UUID, ...]
    top_k: int


@dataclass(frozen=True)
class RetrievalEvalResult:
    case_name: str
    retrieved_episode_ids: tuple[UUID, ...]
    expected_episode_ids: tuple[UUID, ...]
    hit_at_k: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalEvalSummary:
    total_cases: int
    hit_rate_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float


async def run_retrieval_eval(
    service: MemoryService,
    cases: Sequence[RetrievalEvalCase],
    now: datetime | None = None,
) -> tuple[list[RetrievalEvalResult], RetrievalEvalSummary]:
    """Evaluate scoped retrieval quality through the real memory service path."""

    for case in cases:
        _validate_eval_case(case)

    results: list[RetrievalEvalResult] = []
    for case in cases:
        retrieved = await service.retrieve_scoped_episodes(
            user_id=case.user_id,
            scope_id=case.scope_id,
            query=case.query,
            top_k=case.top_k,
            now=now,
        )
        retrieved_episode_ids = tuple(episode.id for episode in retrieved)
        results.append(
            _result_for_case(
                case=case,
                retrieved_episode_ids=retrieved_episode_ids,
            )
        )

    return results, _summarize_results(results)


def _validate_eval_case(case: RetrievalEvalCase) -> None:
    if case.top_k <= 0:
        raise InvalidRetrievalRequestError("eval case top_k must be greater than zero")
    if not case.expected_episode_ids:
        raise InvalidRetrievalRequestError("eval case expected_episode_ids must not be empty")


def _result_for_case(
    case: RetrievalEvalCase,
    retrieved_episode_ids: tuple[UUID, ...],
) -> RetrievalEvalResult:
    expected_episode_ids = case.expected_episode_ids
    expected_episode_id_set = set(expected_episode_ids)
    retrieved_expected_count = sum(
        1 for episode_id in retrieved_episode_ids if episode_id in expected_episode_id_set
    )
    retrieved_count = len(retrieved_episode_ids)
    first_expected_rank = next(
        (
            rank
            for rank, episode_id in enumerate(retrieved_episode_ids, start=1)
            if episode_id in expected_episode_id_set
        ),
        None,
    )

    return RetrievalEvalResult(
        case_name=case.name,
        retrieved_episode_ids=retrieved_episode_ids,
        expected_episode_ids=expected_episode_ids,
        hit_at_k=retrieved_expected_count > 0,
        precision_at_k=0.0 if retrieved_count == 0 else retrieved_expected_count / retrieved_count,
        recall_at_k=retrieved_expected_count / len(expected_episode_ids),
        reciprocal_rank=0.0 if first_expected_rank is None else 1.0 / first_expected_rank,
    )


def _summarize_results(results: Sequence[RetrievalEvalResult]) -> RetrievalEvalSummary:
    total_cases = len(results)
    if total_cases == 0:
        return RetrievalEvalSummary(
            total_cases=0,
            hit_rate_at_k=0.0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
        )

    return RetrievalEvalSummary(
        total_cases=total_cases,
        hit_rate_at_k=sum(1 for result in results if result.hit_at_k) / total_cases,
        mean_precision_at_k=sum(result.precision_at_k for result in results) / total_cases,
        mean_recall_at_k=sum(result.recall_at_k for result in results) / total_cases,
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in results) / total_cases,
    )
