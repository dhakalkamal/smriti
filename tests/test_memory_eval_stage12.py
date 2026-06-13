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
from scripts.run_retrieval_lexical_replay import main as run_retrieval_lexical_replay
from scripts.run_retrieval_role_policy_replay import main as run_retrieval_role_policy_replay
from scripts.run_retrieval_weight_sweep import main as run_retrieval_weight_sweep
from smriti.config import Settings
from smriti.db.migrate import apply_migrations
from smriti.memory import EpisodeKind, InvalidRetrievalRequestError, ScoredEpisode
from smriti.memory.eval import (
    EpisodeLabelRole,
    PreferredLayer,
    QuestionType,
    Stage12Corpus,
    Stage12CorpusCase,
    Stage12CorpusEpisodeLabel,
    Stage12ExpectedIds,
    Stage12ExpectedRefs,
    Stage12ResolvedEpisodeLabel,
    Stage12ResolvedEvalCase,
    Stage12WeightProfile,
    Stage13LexicalProfile,
    Stage13RolePolicy,
    replay_stage12_weight_profile_case,
    replay_stage13_lexical_profile_case,
    replay_stage13_role_policy_case,
    run_stage12_weight_sweep,
    run_stage13_lexical_replay,
    run_stage13_role_policy_replay,
    score_stage12_retrieval_case,
    score_stage12_weight_profile_record,
    stage12_case_result_from_dict,
    stage12_case_result_to_dict,
    stage13_lexical_features,
    stage13_lexical_replay_report_to_markdown,
    stage13_lexical_tokens,
    stage13_role_policy_replay_report_to_markdown,
    stage13_weak_evidence_metrics,
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


def test_stage12_diagnostic_top_k_finds_expected_below_official_window() -> None:
    expected_id = UUID("00000000-0000-4000-8000-000000000261")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        labels=(_label(expected_id, "raw_source", is_expected=True, is_acceptable=True),),
    )
    retrieved = tuple(
        _episode(UUID(f"00000000-0000-4000-8000-00000000027{rank}"), rank=rank)
        for rank in range(1, 6)
    ) + (_episode(expected_id, rank=6),)

    result = score_stage12_retrieval_case(case, retrieved, diagnostic_top_k=6)

    assert result.metrics.acceptable_hit_at_k is False
    assert result.diagnostics.diagnostic_top_k == 6
    assert result.diagnostics.acceptable_in_diagnostic_top_k is True
    assert result.diagnostics.first_acceptable_diagnostic_rank == 6
    assert "rerank_below_official_k" in result.diagnostics.failure_class
    assert "raw_visible_but_low" in result.diagnostics.failure_class


def test_stage12_diagnostic_top_k_defaults_to_case_top_k() -> None:
    expected_id = UUID("00000000-0000-4000-8000-000000000281")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        labels=(_label(expected_id, "raw_source", is_expected=True, is_acceptable=True),),
    )
    retrieved = tuple(
        _episode(UUID(f"00000000-0000-4000-8000-00000000029{rank}"), rank=rank)
        for rank in range(1, 6)
    ) + (_episode(expected_id, rank=6),)

    result = score_stage12_retrieval_case(case, retrieved)

    assert len(result.retrieved) == case.top_k
    assert result.diagnostics.diagnostic_top_k == case.top_k
    assert result.diagnostics.acceptable_in_diagnostic_top_k is False
    assert result.diagnostics.failure_class == ("absent_from_diagnostic_top_k",)


def test_stage12_diagnostic_ranks_exclude_current_query_in_app_realistic() -> None:
    current_id = UUID("00000000-0000-4000-8000-000000000301")
    expected_id = UUID("00000000-0000-4000-8000-000000000302")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        current_query=(current_id,),
        labels=(
            _label(current_id, "current_query"),
            _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
        ),
    )
    retrieved = (
        _episode(current_id, rank=1, similarity=1.0),
        _episode(UUID("00000000-0000-4000-8000-000000000303"), rank=2),
        _episode(UUID("00000000-0000-4000-8000-000000000304"), rank=3),
        _episode(UUID("00000000-0000-4000-8000-000000000305"), rank=4),
        _episode(UUID("00000000-0000-4000-8000-000000000306"), rank=5),
        _episode(expected_id, rank=6),
    )

    result = score_stage12_retrieval_case(
        case,
        retrieved,
        timing_mode="app_realistic",
        diagnostic_top_k=6,
    )

    assert result.metrics.acceptable_hit_at_k is False
    assert result.metrics.self_query_hit is True
    assert result.diagnostics.first_acceptable_diagnostic_rank == 5
    assert "self_query_artifact" in result.diagnostics.failure_class
    assert "hit_at_official_k" in result.diagnostics.failure_class


def test_stage12_failure_class_absent_from_diagnostic_top_k() -> None:
    expected_id = UUID("00000000-0000-4000-8000-000000000311")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        labels=(_label(expected_id, "raw_source", is_expected=True, is_acceptable=True),),
    )

    result = score_stage12_retrieval_case(
        case,
        (_episode(UUID("00000000-0000-4000-8000-000000000312"), rank=1),),
        diagnostic_top_k=6,
    )

    assert result.diagnostics.acceptable_in_diagnostic_top_k is False
    assert result.diagnostics.failure_class == ("absent_from_diagnostic_top_k",)


def test_stage12_failure_class_recap_or_echo_pollution() -> None:
    echo_id = UUID("00000000-0000-4000-8000-000000000321")
    expected_id = UUID("00000000-0000-4000-8000-000000000322")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        labels=(
            _label(echo_id, "assistant_answer_echo"),
            _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
        ),
    )

    result = score_stage12_retrieval_case(
        case,
        (_episode(echo_id, rank=1), _episode(expected_id, rank=2)),
    )

    assert result.metrics.acceptable_hit_at_k is True
    assert "hit_at_official_k" in result.diagnostics.failure_class
    assert "recap_or_echo_pollution" in result.diagnostics.failure_class


def test_stage12_failure_class_negative_control() -> None:
    result = score_stage12_retrieval_case(
        _case(),
        (_episode(UUID("00000000-0000-4000-8000-000000000331"), rank=1),),
    )

    assert result.diagnostics.acceptable_in_diagnostic_top_k is False
    assert result.diagnostics.failure_class == ("negative_control",)


def test_stage13_weak_evidence_metrics_split_recap_and_assistant_echo() -> None:
    recap_id = UUID("00000000-0000-4000-8000-0000000003a1")
    echo_id = UUID("00000000-0000-4000-8000-0000000003a2")
    expected_id = UUID("00000000-0000-4000-8000-0000000003a3")
    result = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            labels=(
                _label(recap_id, "recap_question"),
                _label(echo_id, "assistant_answer_echo"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (
            _episode(recap_id, rank=1),
            _episode(echo_id, rank=2),
            _episode(expected_id, rank=3),
        ),
    )

    metrics = stage13_weak_evidence_metrics(result)

    assert metrics.recap_above_first_acceptable_count == 1
    assert metrics.assistant_echo_above_first_acceptable_count == 1
    assert metrics.total_weak_evidence_above_first_acceptable_count == 2
    assert metrics.recap_official_top_k_count == 1
    assert metrics.assistant_echo_official_top_k_count == 1
    assert metrics.recap_diagnostic_top_k_count == 1
    assert metrics.assistant_echo_diagnostic_top_k_count == 1
    assert metrics.first_blocking_weak_evidence_role == "recap"


def test_stage13_weak_evidence_metrics_reports_both_for_dual_label() -> None:
    weak_id = UUID("00000000-0000-4000-8000-0000000003b1")
    expected_id = UUID("00000000-0000-4000-8000-0000000003b2")
    dual_label = Stage12ResolvedEpisodeLabel(
        semantic_ref="dual",
        episode_id=weak_id,
        roles=("recap_question", "assistant_answer_echo"),
        episode_kind="message",
        layer="diagnostic",
        fact_ids=("F1",),
    )
    result = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            labels=(
                dual_label,
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (_episode(weak_id, rank=1), _episode(expected_id, rank=2)),
    )

    metrics = stage13_weak_evidence_metrics(result)

    assert metrics.recap_above_first_acceptable_count == 1
    assert metrics.assistant_echo_above_first_acceptable_count == 1
    assert metrics.total_weak_evidence_above_first_acceptable_count == 1
    assert metrics.first_blocking_weak_evidence_role == "both"


def test_stage13_exclude_assistant_echo_policy_recomputes_official_ranks() -> None:
    echo_id = UUID("00000000-0000-4000-8000-0000000003c1")
    expected_id = UUID("00000000-0000-4000-8000-0000000003c2")
    baseline = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(
                _label(echo_id, "assistant_answer_echo"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (_episode(echo_id, rank=1), _episode(expected_id, rank=2)),
        diagnostic_top_k=2,
    )
    policy = Stage13RolePolicy(
        name="exclude_assistant_echo",
        description="test",
        excluded_roles=("assistant_answer_echo",),
    )

    replayed = replay_stage13_role_policy_case(baseline, policy)
    rank_by_id = {
        record.episode_id: record.official_rank for record in replayed.case_result.retrieved
    }

    assert baseline.metrics.acceptable_hit_at_k is False
    assert rank_by_id[echo_id] is None
    assert rank_by_id[expected_id] == 1
    assert replayed.case_result.metrics.acceptable_hit_at_k is True
    assert replayed.weak_evidence.assistant_echo_diagnostic_top_k_count == 0


def test_stage13_exclude_recap_policy_recomputes_official_ranks() -> None:
    recap_id = UUID("00000000-0000-4000-8000-0000000003d1")
    expected_id = UUID("00000000-0000-4000-8000-0000000003d2")
    baseline = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(
                _label(recap_id, "recap_question"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (_episode(recap_id, rank=1), _episode(expected_id, rank=2)),
        diagnostic_top_k=2,
    )
    policy = Stage13RolePolicy(
        name="exclude_recap_question",
        description="test",
        excluded_roles=("recap_question",),
        depends_on_recap_metadata=True,
    )

    replayed = replay_stage13_role_policy_case(baseline, policy)

    assert replayed.case_result.metrics.acceptable_hit_at_k is True
    assert replayed.weak_evidence.recap_diagnostic_top_k_count == 0


def test_stage13_exclude_recap_and_echo_policy_removes_both_roles() -> None:
    recap_id = UUID("00000000-0000-4000-8000-0000000003e1")
    echo_id = UUID("00000000-0000-4000-8000-0000000003e2")
    expected_id = UUID("00000000-0000-4000-8000-0000000003e3")
    baseline = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(
                _label(recap_id, "recap_question"),
                _label(echo_id, "assistant_answer_echo"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (
            _episode(recap_id, rank=1),
            _episode(echo_id, rank=2),
            _episode(expected_id, rank=3),
        ),
        diagnostic_top_k=3,
    )
    policy = Stage13RolePolicy(
        name="exclude_recap_and_echo",
        description="test",
        excluded_roles=("recap_question", "assistant_answer_echo"),
        depends_on_recap_metadata=True,
    )

    replayed = replay_stage13_role_policy_case(baseline, policy)

    assert replayed.case_result.metrics.acceptable_hit_at_k is True
    assert replayed.weak_evidence.recap_diagnostic_top_k_count == 0
    assert replayed.weak_evidence.assistant_echo_diagnostic_top_k_count == 0


def test_stage13_demote_assistant_echo_policy_moves_echo_behind_source() -> None:
    echo_id = UUID("00000000-0000-4000-8000-0000000003f1")
    expected_id = UUID("00000000-0000-4000-8000-0000000003f2")
    baseline = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(
                _label(echo_id, "assistant_answer_echo"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (_episode(echo_id, rank=1), _episode(expected_id, rank=2)),
        diagnostic_top_k=2,
    )
    policy = Stage13RolePolicy(
        name="demote_assistant_echo",
        description="test",
        demoted_roles=("assistant_answer_echo",),
        demotion_strategy="move_behind_non_weak",
    )

    replayed = replay_stage13_role_policy_case(baseline, policy)

    assert [record.episode_id for record in replayed.case_result.retrieved] == [
        expected_id,
        echo_id,
    ]
    assert replayed.case_result.metrics.acceptable_hit_at_k is True
    assert replayed.weak_evidence.assistant_echo_above_first_acceptable_count == 0


def test_stage13_role_policy_replay_preserves_current_query_exclusion() -> None:
    current_id = UUID("00000000-0000-4000-8000-000000000401")
    echo_id = UUID("00000000-0000-4000-8000-000000000402")
    expected_id = UUID("00000000-0000-4000-8000-000000000403")
    baseline = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            current_query=(current_id,),
            top_k=1,
            labels=(
                _label(current_id, "current_query"),
                _label(echo_id, "assistant_answer_echo"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (
            _episode(current_id, rank=1, similarity=1.0),
            _episode(echo_id, rank=2),
            _episode(expected_id, rank=3),
        ),
        timing_mode="app_realistic",
        diagnostic_top_k=3,
    )
    policy = Stage13RolePolicy(
        name="exclude_assistant_echo",
        description="test",
        excluded_roles=("assistant_answer_echo",),
    )

    replayed = replay_stage13_role_policy_case(baseline, policy)
    official_ranks = {
        record.episode_id: record.official_rank for record in replayed.case_result.retrieved
    }

    assert official_ranks[current_id] is None
    assert official_ranks[echo_id] is None
    assert official_ranks[expected_id] == 1
    assert replayed.case_result.metrics.self_query_hit is True
    assert replayed.case_result.metrics.acceptable_hit_at_k is True


def test_stage13_role_policy_replay_recommends_assistant_echo_candidate() -> None:
    echo_id = UUID("00000000-0000-4000-8000-000000000411")
    expected_id = UUID("00000000-0000-4000-8000-000000000412")
    negative_id = UUID("00000000-0000-4000-8000-000000000413")
    blocked = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(
                _label(echo_id, "assistant_answer_echo"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (_episode(echo_id, rank=1), _episode(expected_id, rank=2)),
        timing_mode="clean_memory",
        diagnostic_top_k=2,
    )
    negative_control = score_stage12_retrieval_case(
        _case(top_k=1, question_type="negative_control"),
        (_episode(negative_id, rank=1),),
        timing_mode="clean_memory",
    )

    payload = run_stage13_role_policy_replay(
        input_metadata={
            "run_id": "stage13-role-synthetic",
            "timing_mode": "clean_memory",
            "embedder_mode": "ollama",
            "embedding_model": "nomic-embed-text",
        },
        cases=(blocked, negative_control),
        policies=(
            Stage13RolePolicy(name="baseline", description="baseline"),
            Stage13RolePolicy(
                name="exclude_assistant_echo",
                description="test",
                excluded_roles=("assistant_answer_echo",),
            ),
        ),
        created_at=FIXED_NOW,
    )

    recommendation = cast(dict[str, object], payload["recommendation"])
    aggregates = cast(dict[str, object], payload["aggregate_metrics_by_policy"])
    policy_aggregate = cast(dict[str, object], aggregates["exclude_assistant_echo"])

    assert recommendation["decision"] == "assistant_echo_policy_candidate"
    assert recommendation["policy"] == "exclude_assistant_echo"
    assert policy_aggregate["negative_control_no_hit_retained"] is True


def test_stage13_role_policy_replay_keeps_recap_eval_only() -> None:
    recap_id = UUID("00000000-0000-4000-8000-000000000421")
    expected_id = UUID("00000000-0000-4000-8000-000000000422")
    blocked = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(
                _label(recap_id, "recap_question"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (_episode(recap_id, rank=1), _episode(expected_id, rank=2)),
        timing_mode="clean_memory",
        diagnostic_top_k=2,
    )

    payload = run_stage13_role_policy_replay(
        input_metadata={
            "run_id": "stage13-recap-synthetic",
            "timing_mode": "clean_memory",
            "embedder_mode": "ollama",
            "embedding_model": "nomic-embed-text",
        },
        cases=(blocked,),
        policies=(
            Stage13RolePolicy(name="baseline", description="baseline"),
            Stage13RolePolicy(
                name="exclude_recap_question",
                description="test",
                excluded_roles=("recap_question",),
                depends_on_recap_metadata=True,
            ),
        ),
        created_at=FIXED_NOW,
    )

    recommendation = cast(dict[str, object], payload["recommendation"])
    gates = cast(dict[str, object], payload["regression_gates"])
    recap_gates = cast(dict[str, object], gates["exclude_recap_question"])

    assert recommendation["decision"] == "recap_policy_requires_metadata"
    assert "recap_metadata_available" in recap_gates["failed_gates"]


def test_stage12_weight_profile_score_recomputation() -> None:
    episode_id = UUID("00000000-0000-4000-8000-000000000341")
    case = _case(
        raw=(episode_id,),
        labels=(_label(episode_id, "raw_source", is_expected=True),),
    )
    result = score_stage12_retrieval_case(
        case,
        (
            _episode(
                episode_id,
                rank=1,
                similarity=0.40,
                recency_score=0.20,
                access_score=0.10,
                importance_score=0.30,
                frequency_score=0.50,
            ),
        ),
    )
    profile = Stage12WeightProfile(
        name="custom",
        similarity=0.50,
        recency=0.20,
        access=0.10,
        importance=0.15,
        frequency=0.05,
    )

    assert score_stage12_weight_profile_record(result.retrieved[0], profile) == pytest.approx(
        0.50 * 0.40 + 0.20 * 0.20 + 0.10 * 0.10 + 0.15 * 0.30 + 0.05 * 0.50
    )


def test_stage12_weight_profile_replay_sorts_and_preserves_current_query_exclusion() -> None:
    current_id = UUID("00000000-0000-4000-8000-000000000351")
    expected_id = UUID("00000000-0000-4000-8000-000000000352")
    other_id = UUID("00000000-0000-4000-8000-000000000353")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        current_query=(current_id,),
        top_k=1,
        labels=(
            _label(current_id, "current_query"),
            _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
        ),
    )
    baseline = score_stage12_retrieval_case(
        case,
        (
            _episode(current_id, rank=1, similarity=1.0, importance_score=0.90),
            _episode(other_id, rank=2, similarity=0.9, importance_score=0.10),
            _episode(expected_id, rank=3, similarity=0.8, importance_score=0.80),
        ),
        timing_mode="app_realistic",
        diagnostic_top_k=3,
    )
    profile = Stage12WeightProfile(
        name="importance_only",
        similarity=0.0,
        recency=0.0,
        access=0.0,
        importance=1.0,
        frequency=0.0,
    )

    replayed = replay_stage12_weight_profile_case(baseline, profile)

    replayed_rank_tuples = [
        (record.episode_id, record.rank, record.official_rank) for record in replayed.retrieved
    ]
    assert replayed_rank_tuples == [
        (current_id, 1, None),
        (expected_id, 2, 1),
        (other_id, 3, 2),
    ]
    assert replayed.metrics.acceptable_hit_at_k is True
    assert replayed.metrics.reciprocal_rank == pytest.approx(1.0)
    assert replayed.metrics.self_query_hit is True


def test_stage12_weight_sweep_detects_regressions_and_keeps_negative_control() -> None:
    expected_id = UUID("00000000-0000-4000-8000-000000000361")
    distractor_id = UUID("00000000-0000-4000-8000-000000000362")
    direct_fact = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(_label(expected_id, "raw_source", is_expected=True, is_acceptable=True),),
        ),
        (
            _episode(expected_id, rank=1, similarity=1.0, recency_score=0.0),
            _episode(distractor_id, rank=2, similarity=0.1, recency_score=1.0),
        ),
        timing_mode="clean_memory",
        diagnostic_top_k=2,
    )
    negative_control = score_stage12_retrieval_case(
        _case(top_k=1, question_type="negative_control"),
        (_episode(distractor_id, rank=1, similarity=0.1, recency_score=1.0),),
        timing_mode="clean_memory",
        diagnostic_top_k=1,
    )
    payload = run_stage12_weight_sweep(
        input_metadata={
            "run_id": "stage12-synthetic",
            "timing_mode": "clean_memory",
            "embedder_mode": "ollama",
            "embedding_model": "nomic-embed-text",
        },
        cases=(direct_fact, negative_control),
        profiles=(
            Stage12WeightProfile(
                name="baseline",
                similarity=0.55,
                recency=0.20,
                access=0.10,
                importance=0.10,
                frequency=0.05,
            ),
            Stage12WeightProfile(
                name="recency_bad",
                similarity=0.0,
                recency=1.0,
                access=0.0,
                importance=0.0,
                frequency=0.0,
            ),
        ),
        created_at=FIXED_NOW,
    )

    regression_gates = cast(dict[str, object], payload["regression_gates"])
    bad_gates = cast(dict[str, object], regression_gates["recency_bad"])
    aggregate = cast(dict[str, object], payload["aggregate_metrics_by_profile"])
    bad_aggregate = cast(dict[str, object], aggregate["recency_bad"])

    assert "direct_fact_regression" in bad_gates["failed_gates"]
    assert "raw_hit_not_dropped" in bad_gates["failed_gates"]
    assert bad_aggregate["negative_control_no_hit_retained"] is True
    assert cast(dict[str, object], payload["recommendation"])["decision"] == "no_change_recommended"


def test_stage12_weight_sweep_rejects_missing_score_components() -> None:
    episode_id = UUID("00000000-0000-4000-8000-000000000371")
    result = score_stage12_retrieval_case(
        _case(
            raw=(episode_id,),
            labels=(_label(episode_id, "raw_source", is_expected=True),),
        ),
        (_episode(episode_id, rank=1),),
    )
    record = stage12_case_result_to_dict(result)
    retrieved = cast(list[dict[str, object]], record["retrieved"])
    score_components = cast(dict[str, object], retrieved[0]["score_components"])
    del score_components["recency"]

    with pytest.raises(InvalidRetrievalRequestError, match="missing score component: recency"):
        stage12_case_result_from_dict(record)


def test_stage12_weight_sweep_runner_smoke_writes_json_and_markdown(tmp_path: Path) -> None:
    episode_id = UUID("00000000-0000-4000-8000-000000000381")
    result = score_stage12_retrieval_case(
        _case(
            raw=(episode_id,),
            acceptable=(episode_id,),
            labels=(_label(episode_id, "raw_source", is_expected=True, is_acceptable=True),),
        ),
        (_episode(episode_id, rank=1),),
        timing_mode="clean_memory",
    )
    input_run = tmp_path / "input-run"
    input_run.mkdir()
    (input_run / "run.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "run_id": "stage12-synthetic",
                    "timing_mode": "clean_memory",
                    "embedder_mode": "fake",
                    "embedding_model": "nomic-embed-text",
                    "git_branch": "stage-12-retrieval-eval-harness",
                    "git_sha": "synthetic",
                },
                "aggregate_metrics": {},
                "cases": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (input_run / "cases.jsonl").write_text(
        json.dumps(stage12_case_result_to_dict(result), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = run_retrieval_weight_sweep(
        [
            "--input-run",
            str(input_run),
            "--output-dir",
            str(tmp_path / "runs"),
            "--report-dir",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 0
    weight_sweep_path = tmp_path / "runs" / "stage12-synthetic" / "weight_sweep.json"
    report_path = tmp_path / "results" / "stage12-synthetic-weight-sweep.md"
    assert weight_sweep_path.exists()
    assert report_path.exists()
    payload = json.loads(weight_sweep_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["input_run_id"] == "stage12-synthetic"
    assert payload["metadata"]["profile_definitions"][0]["name"] == "baseline"
    assert "Stage 12b-2 Weight Sweep" in report_path.read_text(encoding="utf-8")


def test_stage13_lexical_token_normalization() -> None:
    assert stage13_lexical_tokens("What did Tunde's $3,650 lease do?") == (
        "tunde",
        "3650",
        "lease",
    )


def test_stage13_lexical_feature_families_find_names_and_relationships() -> None:
    features = stage13_lexical_features(
        "What role does cousin Dele have in Tunde's studio plan?",
        "Cousin Dele is the silent partner handling bookkeeping.",
    )

    assert features.token_overlap > 0.0
    assert features.query_token_coverage == pytest.approx(2 / 6)
    assert features.rare_token_overlap == pytest.approx(2 / 5)
    assert features.proper_name_overlap == pytest.approx(1 / 2)
    assert features.relationship_anchor_overlap == pytest.approx(1 / 3)
    assert "Dele" in features.diagnostic_anchor_hits
    assert "bookkeeping" in features.diagnostic_anchor_hits


def test_stage13_lexical_number_currency_and_date_overlap() -> None:
    features = stage13_lexical_features(
        "What is the $3,650 cap for Friday July 31?",
        "The kiln budget has a hard cap of $3,650 and the signing is Friday, July 31.",
    )

    assert features.number_currency_overlap == pytest.approx(1.0)
    assert "$3,650" in features.diagnostic_anchor_hits


def test_stage13_lexical_profile_recomputes_ranks() -> None:
    expected_id = UUID("00000000-0000-4000-8000-0000000003b1")
    distractor_id = UUID("00000000-0000-4000-8000-0000000003b2")
    case = _case(
        raw=(expected_id,),
        acceptable=(expected_id,),
        top_k=1,
        query="What role does cousin Dele have in Tunde's studio plan?",
        labels=(_label(expected_id, "raw_source", is_expected=True, is_acceptable=True),),
    )
    baseline = score_stage12_retrieval_case(
        case,
        (
            _episode(
                distractor_id,
                rank=1,
                similarity=0.9,
                content="Lease timing and pottery wheels are still being planned.",
            ),
            _episode(
                expected_id,
                rank=2,
                similarity=0.8,
                content="Cousin Dele is the silent partner handling bookkeeping.",
            ),
        ),
        timing_mode="clean_memory",
        diagnostic_top_k=2,
    )
    profile = Stage13LexicalProfile(
        name="proper_name_only",
        description="Synthetic test profile.",
        original_score=0.0,
        proper_name_overlap=1.0,
    )

    replayed = replay_stage13_lexical_profile_case(baseline, profile)

    assert replayed.retrieved[0].episode_id == expected_id
    assert replayed.metrics.acceptable_hit_at_k is True
    assert replayed.diagnostics.first_acceptable_diagnostic_rank == 1


def test_stage13_lexical_replay_recommends_safe_candidate_and_keeps_negative_control() -> None:
    expected_id = UUID("00000000-0000-4000-8000-0000000003c1")
    distractor_id = UUID("00000000-0000-4000-8000-0000000003c2")
    direct_case = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            query="What role does cousin Dele have in Tunde's studio plan?",
            labels=(_label(expected_id, "raw_source", is_expected=True, is_acceptable=True),),
        ),
        (
            _episode(
                distractor_id,
                rank=1,
                similarity=0.9,
                content="Lease timing and pottery wheels are still being planned.",
            ),
            _episode(
                expected_id,
                rank=2,
                similarity=0.8,
                content="Cousin Dele is the silent partner handling bookkeeping.",
            ),
        ),
        timing_mode="clean_memory",
        diagnostic_top_k=2,
    )
    negative_control = score_stage12_retrieval_case(
        _case(top_k=1, question_type="negative_control", query="Which supplier was chosen?"),
        (
            _episode(
                distractor_id,
                rank=1,
                similarity=0.6,
                content="The clay supplier is undecided.",
            ),
        ),
        timing_mode="clean_memory",
        diagnostic_top_k=1,
    )
    payload = run_stage13_lexical_replay(
        input_metadata={
            "run_id": "stage13-lexical-synthetic",
            "timing_mode": "clean_memory",
            "embedder_mode": "ollama",
            "embedding_model": "nomic-embed-text",
        },
        cases=(direct_case, negative_control),
        profiles=(
            Stage13LexicalProfile(
                name="baseline",
                description="Preserve baseline.",
                original_score=1.0,
            ),
            Stage13LexicalProfile(
                name="proper_name_only",
                description="Synthetic name profile.",
                original_score=0.0,
                proper_name_overlap=1.0,
            ),
        ),
        created_at=FIXED_NOW,
    )

    recommendation = cast(dict[str, object], payload["recommendation"])
    aggregates = cast(dict[str, object], payload["aggregate_metrics_by_profile"])
    profile_aggregate = cast(dict[str, object], aggregates["proper_name_only"])

    assert recommendation["decision"] == "lexical_rerank_candidate"
    assert recommendation["profile"] == "proper_name_only"
    assert profile_aggregate["negative_control_no_hit_retained"] is True
    assert profile_aggregate["missing_lexical_feature_count"] == 0


def test_stage13_lexical_replay_rejects_missing_features() -> None:
    expected_id = UUID("00000000-0000-4000-8000-0000000003d1")
    result = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            labels=(_label(expected_id, "raw_source", is_expected=True),),
        ),
        (_episode(expected_id, rank=1),),
        timing_mode="clean_memory",
    )
    record = stage12_case_result_to_dict(result)
    retrieved = cast(list[dict[str, object]], record["retrieved"])
    del retrieved[0]["lexical_features"]
    parsed = stage12_case_result_from_dict(record)

    with pytest.raises(InvalidRetrievalRequestError, match="requires lexical_features"):
        run_stage13_lexical_replay(
            input_metadata={"run_id": "stage13-missing-features"},
            cases=(parsed,),
            created_at=FIXED_NOW,
        )


def test_stage13_lexical_replay_runner_smoke_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    expected_id = UUID("00000000-0000-4000-8000-0000000003e1")
    result = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            query="What role does cousin Dele have?",
            labels=(_label(expected_id, "raw_source", is_expected=True, is_acceptable=True),),
        ),
        (
            _episode(
                expected_id,
                rank=1,
                content="Cousin Dele is the silent partner handling bookkeeping.",
            ),
        ),
        timing_mode="clean_memory",
    )
    input_run = tmp_path / "input-run"
    input_run.mkdir()
    (input_run / "run.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "run_id": "stage13-lexical-smoke",
                    "timing_mode": "clean_memory",
                    "embedder_mode": "fake",
                    "embedding_model": "nomic-embed-text",
                    "git_branch": "stage-13-retrieval-evidence-improvements",
                    "git_sha": "synthetic",
                },
                "aggregate_metrics": {},
                "cases": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (input_run / "cases.jsonl").write_text(
        json.dumps(stage12_case_result_to_dict(result), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = run_retrieval_lexical_replay(
        [
            "--input-run",
            str(input_run),
            "--output-dir",
            str(tmp_path / "runs"),
            "--report-dir",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 0
    replay_path = tmp_path / "runs" / "stage13-lexical-smoke" / "lexical_replay.json"
    report_path = tmp_path / "results" / "stage13-lexical-smoke-lexical-replay.md"
    assert replay_path.exists()
    assert report_path.exists()
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["input_run_id"] == "stage13-lexical-smoke"
    assert payload["metadata"]["profile_definitions"][0]["name"] == "baseline"
    assert "lexical_combined" in payload["aggregate_metrics_by_profile"]
    assert "Stage 13c-1 Lexical Replay" in report_path.read_text(encoding="utf-8")
    assert "Exact-Name And Anchor Diagnostics" in stage13_lexical_replay_report_to_markdown(payload)


def test_stage13_role_policy_replay_runner_smoke_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    echo_id = UUID("00000000-0000-4000-8000-000000000391")
    expected_id = UUID("00000000-0000-4000-8000-000000000392")
    result = score_stage12_retrieval_case(
        _case(
            raw=(expected_id,),
            acceptable=(expected_id,),
            top_k=1,
            labels=(
                _label(echo_id, "assistant_answer_echo"),
                _label(expected_id, "raw_source", is_expected=True, is_acceptable=True),
            ),
        ),
        (_episode(echo_id, rank=1), _episode(expected_id, rank=2)),
        timing_mode="clean_memory",
        diagnostic_top_k=2,
    )
    input_run = tmp_path / "input-run"
    input_run.mkdir()
    (input_run / "run.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "run_id": "stage13-role-synthetic",
                    "timing_mode": "clean_memory",
                    "embedder_mode": "fake",
                    "embedding_model": "nomic-embed-text",
                    "git_branch": "stage-13b-role-aware-evidence-diagnostics",
                    "git_sha": "synthetic",
                },
                "aggregate_metrics": {},
                "cases": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (input_run / "cases.jsonl").write_text(
        json.dumps(stage12_case_result_to_dict(result), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = run_retrieval_role_policy_replay(
        [
            "--input-run",
            str(input_run),
            "--output-dir",
            str(tmp_path / "runs"),
            "--report-dir",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 0
    replay_path = tmp_path / "runs" / "stage13-role-synthetic" / "role_policy_replay.json"
    report_path = tmp_path / "results" / "stage13-role-synthetic-role-policy-replay.md"
    assert replay_path.exists()
    assert report_path.exists()
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["input_run_id"] == "stage13-role-synthetic"
    assert payload["metadata"]["policy_definitions"][0]["name"] == "baseline"
    assert "exclude_assistant_echo" in payload["aggregate_metrics_by_policy"]
    assert "Stage 13b Role Policy Replay" in report_path.read_text(encoding="utf-8")
    assert "Regression Gates" in stage13_role_policy_replay_report_to_markdown(payload)


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
                "--diagnostic-top-k",
                "25",
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
    assert payload["metadata"]["diagnostic_top_k"] == 25
    assert len(payload["cases"]) == 9
    assert payload["cases"][0]["diagnostic_top_k"] == 25
    assert "failure_class" in payload["cases"][0]
    assert "Stage 12a Retrieval Baseline" in report_path.read_text(encoding="utf-8")
    assert "Diagnostic Classification" in report_path.read_text(encoding="utf-8")


def _case(
    raw: tuple[UUID, ...] = (),
    summary: tuple[UUID, ...] = (),
    acceptable: tuple[UUID, ...] = (),
    current_query: tuple[UUID, ...] = (),
    preferred_layer: str = "raw",
    question_type: str = "direct_fact",
    top_k: int = 5,
    labels: tuple[Stage12ResolvedEpisodeLabel, ...] = (),
    query: str = "query",
) -> Stage12ResolvedEvalCase:
    return Stage12ResolvedEvalCase(
        example_id="case",
        scenario_id="scenario",
        user_id=USER_ID,
        scope_id=SCOPE_ID,
        query=query,
        top_k=top_k,
        question_type=cast(QuestionType, question_type),
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
    recency_score: float = 0.0,
    access_score: float = 0.0,
    importance_score: float = 0.0,
    frequency_score: float = 0.0,
    content: str = "not serialized by Stage 12 result helpers",
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
        content=content,
        created_at=FIXED_NOW,
        importance=0.0,
        access_count=0,
        last_accessed_at=None,
        embedding_model_id=1,
        similarity=similarity,
        recency_score=recency_score,
        access_score=access_score,
        importance_score=importance_score,
        frequency_score=frequency_score,
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
