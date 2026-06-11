from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from testcontainers.postgres import PostgresContainer

from scripts.run_retrieval_baseline import main as run_retrieval_baseline
from smriti.config import Settings
from smriti.db.migrate import apply_migrations
from smriti.memory import EpisodeKind, InvalidRetrievalRequestError, ScoredEpisode
from smriti.memory.eval import (
    EpisodeLabelRole,
    PreferredLayer,
    Stage12Corpus,
    Stage12CorpusCase,
    Stage12CorpusEpisodeLabel,
    Stage12ExpectedIds,
    Stage12ExpectedRefs,
    Stage12ResolvedEpisodeLabel,
    Stage12ResolvedEvalCase,
    score_stage12_retrieval_case,
    validate_stage12_corpus,
)

FIXED_NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)  # noqa: UP017
USER_ID = UUID("00000000-0000-4000-8000-000000000101")
SCOPE_ID = UUID("00000000-0000-4000-8000-000000000102")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000103")


def test_stage12_current_query_exclusion_recomputes_official_ranks() -> None:
    current_id = UUID("00000000-0000-4000-8000-000000000201")
    expected_id = UUID("00000000-0000-4000-8000-000000000202")
    other_id = UUID("00000000-0000-4000-8000-000000000203")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        current_query=(current_id,),
        labels=(
            _label(current_id, "current_query"),
            _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
        ),
    )

    result = score_stage12_retrieval_case(
        case,
        (
            _episode(current_id, rank=1, similarity=1.0),
            _episode(expected_id, rank=2, similarity=0.8),
            _episode(other_id, rank=3, similarity=0.7),
        ),
        timing_mode="app_realistic",
    )

    assert [record.official_rank for record in result.retrieved] == [None, 1, 2]
    assert result.metrics.hit_at_k is True
    assert result.metrics.reciprocal_rank == pytest.approx(1.0)
    assert result.metrics.self_query_hit is True
    assert result.metrics.self_query_rank == 1
    assert result.metrics.self_query_similarity == pytest.approx(1.0)


def test_stage12_clean_memory_keeps_self_query_diagnostics_false() -> None:
    current_id = UUID("00000000-0000-4000-8000-000000000211")
    expected_id = UUID("00000000-0000-4000-8000-000000000212")
    case = _case(
        raw=(expected_id,),
        current_query=(current_id,),
        labels=(
            _label(current_id, "current_query"),
            _label(expected_id, "raw_source", is_expected=True),
        ),
    )

    result = score_stage12_retrieval_case(
        case,
        (
            _episode(current_id, rank=1, similarity=1.0),
            _episode(expected_id, rank=2, similarity=0.8),
        ),
        timing_mode="clean_memory",
    )

    assert [record.official_rank for record in result.retrieved] == [1, 2]
    assert result.retrieved[0].is_current_query is False
    assert result.metrics.self_query_hit is False
    assert result.metrics.self_query_rank is None
    assert result.metrics.self_query_similarity is None


@pytest.mark.parametrize(
    ("preferred_layer", "expected_hit"),
    [
        ("raw", False),
        ("summary", True),
        ("either", True),
    ],
)
def test_stage12_preferred_layer_drives_hit_metric(
    preferred_layer: str,
    expected_hit: bool,
) -> None:
    raw_id = UUID("00000000-0000-4000-8000-000000000221")
    summary_id = UUID("00000000-0000-4000-8000-000000000222")
    case = _case(
        raw=(raw_id,),
        summary=(summary_id,),
        acceptable=(raw_id, summary_id),
        preferred_layer=cast(PreferredLayer, preferred_layer),
        labels=(
            _label(raw_id, "raw_source", is_expected=True, is_acceptable=True),
            _label(
                summary_id,
                "summary_source",
                kind="summary",
                is_expected=True,
                is_acceptable=True,
            ),
        ),
    )

    result = score_stage12_retrieval_case(
        case,
        (_episode(summary_id, rank=1, kind="summary"),),
    )

    assert result.metrics.hit_at_k is expected_hit
    assert result.metrics.raw_hit_at_k is False
    assert result.metrics.summary_hit_at_k is True
    assert result.metrics.acceptable_hit_at_k is True


def test_stage12_acceptable_hit_is_separate_from_preferred_hit() -> None:
    raw_id = UUID("00000000-0000-4000-8000-000000000231")
    summary_id = UUID("00000000-0000-4000-8000-000000000232")
    case = _case(
        raw=(raw_id,),
        summary=(summary_id,),
        acceptable=(summary_id,),
        preferred_layer="raw",
        labels=(
            _label(raw_id, "raw_source", is_expected=True),
            _label(summary_id, "summary_source", kind="summary", is_acceptable=True),
        ),
    )

    result = score_stage12_retrieval_case(
        case,
        (_episode(summary_id, rank=1, kind="summary"),),
    )

    assert result.metrics.hit_at_k is False
    assert result.metrics.acceptable_hit_at_k is True


def test_stage12_recap_pollution_counts_labeled_recap_questions() -> None:
    recap_id = UUID("00000000-0000-4000-8000-000000000241")
    raw_id = UUID("00000000-0000-4000-8000-000000000242")
    case = _case(
        raw=(raw_id,),
        labels=(
            _label(recap_id, "recap_question"),
            _label(raw_id, "raw_source", is_expected=True),
        ),
    )

    result = score_stage12_retrieval_case(
        case,
        (_episode(recap_id, rank=1), _episode(raw_id, rank=2)),
    )

    assert result.metrics.recap_pollution_at_k.count == 1
    assert result.metrics.recap_pollution_at_k.ratio == pytest.approx(0.5)


def test_stage12_kind_mix_counts_official_message_and_summary_results() -> None:
    raw_id = UUID("00000000-0000-4000-8000-000000000251")
    summary_id = UUID("00000000-0000-4000-8000-000000000252")
    case = _case(
        raw=(raw_id,),
        summary=(summary_id,),
        labels=(
            _label(raw_id, "raw_source", is_expected=True),
            _label(summary_id, "summary_source", kind="summary", is_expected=True),
        ),
    )

    result = score_stage12_retrieval_case(
        case,
        (_episode(raw_id, rank=1), _episode(summary_id, rank=2, kind="summary")),
    )

    assert result.metrics.kind_mix_at_k.message_count == 1
    assert result.metrics.kind_mix_at_k.summary_count == 1
    assert result.metrics.kind_mix_at_k.total_count == 2
    assert result.metrics.kind_mix_at_k.summary_ratio == pytest.approx(0.5)


def test_stage12_corpus_validation_rejects_unknown_refs_and_roles() -> None:
    valid_label = Stage12CorpusEpisodeLabel(
        semantic_ref="known",
        roles=("raw_source",),
        episode_kind="message",
        layer="raw",
        fact_ids=("F1",),
    )
    unknown_ref_corpus = _corpus(
        Stage12CorpusCase(
            example_id="unknown-ref",
            scenario_id="scenario",
            query="What is known?",
            top_k=1,
            question_type="direct_fact",
            fact_ids=("F1",),
            preferred_layer="raw",
            expected_refs=Stage12ExpectedRefs(
                raw=("missing",),
                summary=(),
                acceptable=(),
                current_query=(),
            ),
            episode_labels=(valid_label,),
        )
    )

    with pytest.raises(InvalidRetrievalRequestError):
        validate_stage12_corpus(unknown_ref_corpus)

    invalid_role_label = Stage12CorpusEpisodeLabel(
        semantic_ref="known",
        roles=("not_a_role",),  # type: ignore[arg-type]
        episode_kind="message",
        layer="raw",
        fact_ids=("F1",),
    )
    invalid_role_corpus = _corpus(
        Stage12CorpusCase(
            example_id="unknown-role",
            scenario_id="scenario",
            query="What is known?",
            top_k=1,
            question_type="direct_fact",
            fact_ids=("F1",),
            preferred_layer="raw",
            expected_refs=Stage12ExpectedRefs(
                raw=("known",),
                summary=(),
                acceptable=(),
                current_query=(),
            ),
            episode_labels=(invalid_role_label,),
        )
    )

    with pytest.raises(InvalidRetrievalRequestError):
        validate_stage12_corpus(invalid_role_corpus)


def test_stage12_runner_smoke_writes_json_jsonl_and_markdown(tmp_path: Path) -> None:
    migrations_dir = Path(__file__).resolve().parents[1] / "src" / "smriti" / "db" / "migrations"

    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="smriti",
        password="smriti",
        dbname="smriti",
    ) as postgres:
        database_url = _to_asyncpg_dsn(postgres.get_connection_url())
        settings = Settings(database_url=database_url)
        asyncio.run(apply_migrations(settings=settings, migrations_dir=migrations_dir))

        exit_code = run_retrieval_baseline(
            [
                "--database-url",
                database_url,
                "--output-root",
                str(tmp_path),
                "--run-id",
                "stage12-smoke",
            ]
        )

    assert exit_code == 0
    run_dir = tmp_path / "runs" / "stage12-smoke"
    report_path = tmp_path / "results" / "stage12-smoke.md"
    assert (run_dir / "run.json").exists()
    assert (run_dir / "cases.jsonl").exists()
    assert report_path.exists()

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["timing_mode"] == "app_realistic"
    assert payload["metadata"]["embedder_mode"] == "fake"
    assert len(payload["cases"]) == 9
    assert "Stage 12a Retrieval Baseline" in report_path.read_text(encoding="utf-8")


def _case(
    raw: tuple[UUID, ...] = (),
    summary: tuple[UUID, ...] = (),
    acceptable: tuple[UUID, ...] = (),
    current_query: tuple[UUID, ...] = (),
    preferred_layer: str = "raw",
    labels: tuple[Stage12ResolvedEpisodeLabel, ...] = (),
) -> Stage12ResolvedEvalCase:
    return Stage12ResolvedEvalCase(
        example_id="case",
        scenario_id="scenario",
        user_id=USER_ID,
        scope_id=SCOPE_ID,
        query="query",
        top_k=5,
        question_type="direct_fact",
        fact_ids=("F1",),
        preferred_layer=cast(PreferredLayer, preferred_layer),
        expected_ids=Stage12ExpectedIds(
            raw=raw,
            summary=summary,
            acceptable=acceptable,
            current_query=current_query,
        ),
        episode_labels=labels,
    )


def _label(
    episode_id: UUID,
    role: str,
    *,
    kind: str = "message",
    is_expected: bool = False,
    is_acceptable: bool = False,
) -> Stage12ResolvedEpisodeLabel:
    return Stage12ResolvedEpisodeLabel(
        semantic_ref=str(episode_id),
        episode_id=episode_id,
        roles=(cast(EpisodeLabelRole, role),),
        episode_kind=cast(EpisodeKind, kind),
        layer="summary" if kind == "summary" else "raw",
        fact_ids=("F1",),
        is_expected=is_expected,
        is_acceptable=is_acceptable,
    )


def _episode(
    episode_id: UUID,
    *,
    rank: int,
    kind: str = "message",
    similarity: float = 0.75,
) -> ScoredEpisode:
    return ScoredEpisode(
        result_rank=rank,
        id=episode_id,
        user_id=USER_ID,
        scope_id=SCOPE_ID,
        conversation_id=CONVERSATION_ID,
        kind=cast(EpisodeKind, kind),
        message_id=UUID(f"00000000-0000-4000-8000-{rank:012d}") if kind == "message" else None,
        message_position=rank if kind == "message" else None,
        range_start=13 if kind == "summary" else None,
        range_end=24 if kind == "summary" else None,
        content="not serialized by Stage 12 result helpers",
        created_at=FIXED_NOW,
        importance=0.0,
        access_count=0,
        last_accessed_at=None,
        embedding_model_id=1,
        similarity=similarity,
        recency_score=0.0,
        access_score=0.0,
        importance_score=0.0,
        frequency_score=0.0,
        score=similarity,
    )


def _corpus(case: Stage12CorpusCase) -> Stage12Corpus:
    return Stage12Corpus(
        corpus_id="test-corpus",
        corpus_version="1.0.0",
        fixture_strategy="test",
        embedding_model="nomic-embed-text",
        cases=(case,),
    )


def _to_asyncpg_dsn(container_url: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", container_url)
