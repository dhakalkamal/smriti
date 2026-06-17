from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CORPUS_PATH = REPO_ROOT / "docs" / "evals" / "corpora" / "terrafold_planted_facts_v1.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "evals"
DEFAULT_MAX_PROMPT_CHARS = 16000
DEFAULT_RECENT_MESSAGE_LIMIT = 20

if TYPE_CHECKING:
    from smriti.memory.eval import Stage12ResolvedEvalCase


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage 13e-1 assembly-aware retrieval eval."""

    parser = argparse.ArgumentParser(
        description="Run the Stage 13e-1 assembly-aware retrieval eval.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Corpus JSONL path. Defaults to {DEFAULT_CORPUS_PATH}.",
    )
    parser.add_argument(
        "--mode",
        choices=("app_realistic",),
        default="app_realistic",
        help="Assembly eval mode. Stage 13e-1 currently requires app_realistic.",
    )
    parser.add_argument(
        "--embedder",
        choices=("fake", "ollama"),
        default="fake",
        help="Embedding backend for the eval. Ollama is manual/on-demand only.",
    )
    parser.add_argument(
        "--ollama-model",
        default="nomic-embed-text",
        help="Ollama embedding model when --embedder=ollama.",
    )
    parser.add_argument("--database-url", default=None, help="Override SMRITI_DATABASE_URL.")
    parser.add_argument(
        "--apply-migrations",
        action="store_true",
        help="Apply pending migrations before building the eval fixture.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root containing runs/ and results/. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument("--run-id", default=None, help="Stable run ID for reproducible tests.")
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=None,
        help="Optional wider retrieval depth for diagnostic visibility.",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CHARS,
        help=f"Prompt character budget. Defaults to {DEFAULT_MAX_PROMPT_CHARS}.",
    )
    parser.add_argument(
        "--recent-message-limit",
        type=int,
        default=DEFAULT_RECENT_MESSAGE_LIMIT,
        help=f"Recent message load limit. Defaults to {DEFAULT_RECENT_MESSAGE_LIMIT}.",
    )
    args = parser.parse_args(argv)
    if args.diagnostic_top_k is not None and args.diagnostic_top_k <= 0:
        parser.error("--diagnostic-top-k must be greater than zero")
    if args.max_prompt_chars <= 0:
        parser.error("--max-prompt-chars must be greater than zero")
    if args.recent_message_limit <= 0:
        parser.error("--recent-message-limit must be greater than zero")
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    from smriti.assistant import AssistantGenerationRequest, AssistantOrchestrator
    from smriti.chat import ChatResponse, FakeChatGenerator
    from smriti.config import Settings
    from smriti.db.client import close_pool, get_pool
    from smriti.db.migrate import apply_migrations
    from smriti.embeddings import FakeEmbedder, OllamaEmbedder
    from smriti.memory import MemoryService
    from smriti.memory.eval import (
        Stage12BaselineRunMetadata,
        load_stage12_corpus,
        score_stage13_assembly_case,
        stage12_baseline_metadata_to_dict,
        stage13_assembly_aggregate_metrics_to_dict,
        stage13_assembly_case_result_to_dict,
        summarize_stage13_assembly_results,
    )
    from tests.eval.fixtures import build_terrafold_fixture, reset_fixture_access_metadata

    corpus = load_stage12_corpus(args.corpus)
    settings = Settings()
    if args.database_url is not None:
        settings = settings.model_copy(update={"database_url": args.database_url})

    if args.apply_migrations:
        await apply_migrations(settings=settings, migrations_dir=settings.migrations_dir)

    embedder = (
        FakeEmbedder(dimensions=768)
        if args.embedder == "fake"
        else OllamaEmbedder(
            model=args.ollama_model,
            base_url=settings.ollama_base_url,
            dimensions=768,
            num_ctx=settings.ollama_embed_num_ctx,
        )
    )

    pool = await get_pool(settings)
    service = MemoryService(pool=pool, embedder=embedder)
    orchestrator = AssistantOrchestrator(
        memory_service=service,
        chat_generator=FakeChatGenerator(
            response=ChatResponse(
                content="unused assembly eval response",
                model="stage-13e-assembly-eval",
                finish_reason="stop",
            )
        ),
    )
    created_at = datetime.now(UTC)
    run_id = args.run_id or _default_run_id(created_at, args.mode, args.embedder)

    try:
        fixture, cases = await build_terrafold_fixture(
            service=service,
            corpus=corpus,
            timing_mode=args.mode,
        )
        results = []
        for case in cases:
            await reset_fixture_access_metadata(pool, fixture.episode_ids)
            query_message_id, query_conversation_id = await _current_query_message_context(
                pool,
                case,
            )
            retrieval_top_k = max(case.top_k, args.diagnostic_top_k or case.top_k)
            assembly = await orchestrator.prepare_generation_debug(
                AssistantGenerationRequest(
                    user_id=case.user_id,
                    scope_id=case.scope_id,
                    conversation_id=query_conversation_id,
                    query_message_id=query_message_id,
                    top_k=retrieval_top_k,
                    max_prompt_chars=args.max_prompt_chars,
                    recent_message_limit=args.recent_message_limit,
                )
            )
            results.append(
                score_stage13_assembly_case(
                    case,
                    retrieved_candidates=assembly.retrieved_memories,
                    admitted_memories=assembly.prompt.selected_memories,
                    skipped_memories=assembly.prompt.skipped_memories,
                    recent_context_ids=assembly.recent_context.selected_recent_message_ids,
                    active_query_message_id=assembly.active_query_message_id,
                    excluded_message_ids=assembly.excluded_message_ids,
                    prompt_message_order=assembly.prompt_message_order,
                    active_query_occurrences=assembly.active_query_occurrences,
                    timing_mode=args.mode,
                    diagnostic_top_k=retrieval_top_k,
                )
            )

        await reset_fixture_access_metadata(pool, fixture.episode_ids)
        aggregate_metrics = summarize_stage13_assembly_results(results)
        metadata = Stage12BaselineRunMetadata(
            run_id=run_id,
            created_at=created_at,
            corpus_id=corpus.corpus_id,
            corpus_version=corpus.corpus_version,
            timing_mode=args.mode,
            embedder_mode=args.embedder,
            embedding_model=service.embedding_model_id
            if args.embedder == "fake"
            else args.ollama_model,
            isolation_strategy=(
                "disposable eval scope with per-case access metadata reset and "
                "assistant assembly debug preparation"
            ),
            database_identifier=_redact_database_url(settings.database_url),
            git_branch=_git_output(("rev-parse", "--abbrev-ref", "HEAD")),
            git_sha=_git_output(("rev-parse", "HEAD")),
            diagnostic_top_k=args.diagnostic_top_k,
        )
        metadata_payload = stage12_baseline_metadata_to_dict(metadata)
        metadata_payload.update(
            {
                "eval_stage": "stage13e-1",
                "eval_layer": "assembly",
                "memory_policy": "stage13e1_recent_context_exclusion",
                "max_prompt_chars": args.max_prompt_chars,
                "recent_message_limit": args.recent_message_limit,
            }
        )
        aggregate_payload = stage13_assembly_aggregate_metrics_to_dict(aggregate_metrics)
        case_records = [stage13_assembly_case_result_to_dict(result) for result in results]

        _write_outputs(
            output_root=args.output_root,
            run_id=run_id,
            metadata=metadata_payload,
            aggregate_metrics=aggregate_payload,
            case_records=case_records,
        )
    finally:
        await close_pool()

    print(f"Wrote Stage 13e-1 assembly eval output to {args.output_root / 'runs' / run_id}")
    print(
        "Wrote Stage 13e-1 assembly eval Markdown report to "
        f"{args.output_root / 'results' / f'{run_id}.md'}"
    )
    return 0


async def _current_query_message_context(
    pool,
    case: Stage12ResolvedEvalCase,
) -> tuple[UUID, UUID]:
    current_query_ids = case.expected_ids.current_query
    if len(current_query_ids) != 1:
        raise ValueError(
            f"Stage 13e-1 assembly eval requires exactly one current query episode: "
            f"{case.example_id}"
        )

    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT message_id, conversation_id
            FROM episodes
            WHERE id = $1
              AND scope_id = $2;
            """,
            current_query_ids[0],
            case.scope_id,
        )
    if row is None or row["message_id"] is None:
        raise ValueError(
            "Stage 13e-1 assembly eval current query episode must resolve to a message_id: "
            f"{case.example_id}"
        )
    return row["message_id"], row["conversation_id"]


def _write_outputs(
    output_root: Path,
    run_id: str,
    metadata: dict[str, object],
    aggregate_metrics: dict[str, object],
    case_records: list[dict[str, object]],
) -> None:
    run_dir = output_root / "runs" / run_id
    results_dir = output_root / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "metadata": metadata,
        "aggregate_metrics": aggregate_metrics,
        "cases": case_records,
    }
    (run_dir / "run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "cases.jsonl").write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in case_records),
        encoding="utf-8",
    )
    (results_dir / f"{run_id}.md").write_text(
        _markdown_report(metadata, aggregate_metrics, case_records),
        encoding="utf-8",
    )


def _markdown_report(
    metadata: dict[str, object],
    aggregate_metrics: dict[str, object],
    case_records: list[dict[str, object]],
) -> str:
    retrieval_metrics = _mapping_value(aggregate_metrics, "retrieval_candidate_metrics")
    assembled_metrics = _mapping_value(aggregate_metrics, "assembled_context_metrics")
    lines = [
        f"# Stage 13e-1 Assembly Eval: {metadata['run_id']}",
        "",
        "## Metadata",
        "",
        f"- Corpus: {metadata['corpus_id']} {metadata['corpus_version']}",
        f"- Timing mode: {metadata['timing_mode']}",
        f"- Embedder: {metadata['embedder_mode']} ({metadata['embedding_model']})",
        f"- Diagnostic top-k: {metadata['diagnostic_top_k']}",
        f"- Max prompt chars: {metadata['max_prompt_chars']}",
        f"- Recent message limit: {metadata['recent_message_limit']}",
        f"- Git branch: {metadata['git_branch']}",
        f"- Git SHA: {metadata['git_sha']}",
        "",
        "## Aggregate Assembly Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            "| retrieval_candidate_metrics.self_query_hit_rate | "
            f"{_format_metric(retrieval_metrics['self_query_hit_rate'])} |"
        ),
        (
            "| assembled_context_metrics.self_query_hit_rate | "
            f"{_format_metric(assembled_metrics['self_query_hit_rate'])} |"
        ),
        (
            "| active_query_exactly_once_rate | "
            f"{_format_metric(aggregate_metrics['active_query_exactly_once_rate'])} |"
        ),
        (
            "| recent_context_duplication_rate | "
            f"{_format_metric(aggregate_metrics['recent_context_duplication_rate'])} |"
        ),
        (
            "| assembled_context_metrics.source_raw_hit_rate_at_k | "
            f"{_format_metric(assembled_metrics['source_raw_hit_rate_at_k'])} |"
        ),
        (
            "| assembled_context_metrics.source_summary_hit_rate_at_k | "
            f"{_format_metric(assembled_metrics['source_summary_hit_rate_at_k'])} |"
        ),
        (
            "| assembled_context_metrics.mean_source_ndcg_at_k | "
            f"{_format_metric(assembled_metrics['mean_source_ndcg_at_k'])} |"
        ),
        "",
        "## Source Rank Metrics",
        "",
        "| Layer | Source MRR | Raw MRR | Summary MRR | Source nDCG |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            "| retrieval candidates | "
            f"{_format_metric(retrieval_metrics['mean_source_reciprocal_rank'])} | "
            f"{_format_metric(retrieval_metrics['mean_source_raw_reciprocal_rank'])} | "
            f"{_format_metric(retrieval_metrics['mean_source_summary_reciprocal_rank'])} | "
            f"{_format_metric(retrieval_metrics['mean_source_ndcg_at_k'])} |"
        ),
        (
            "| assembled context | "
            f"{_format_metric(assembled_metrics['mean_source_reciprocal_rank'])} | "
            f"{_format_metric(assembled_metrics['mean_source_raw_reciprocal_rank'])} | "
            f"{_format_metric(assembled_metrics['mean_source_summary_reciprocal_rank'])} | "
            f"{_format_metric(assembled_metrics['mean_source_ndcg_at_k'])} |"
        ),
        "",
        "## Per-Case Metrics",
        "",
        (
            "| Example | Active Query Count | Duplication | Candidate Self Query | "
            "Assembled Self Query | Raw | Summary | Source nDCG |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in case_records:
        assembly = _mapping_value(case, "assembly_metrics")
        candidate = _mapping_value(case, "retrieval_candidate_metrics")
        assembled = _mapping_value(case, "assembled_context_metrics")
        lines.append(
            f"| {case['example_id']} | "
            f"{assembly['active_query_occurrences']} | "
            f"{_format_metric(assembly['recent_context_duplication_rate'])} | "
            f"{candidate['self_query_hit']} | "
            f"{assembled['self_query_hit']} | "
            f"{assembled['source_raw_hit_at_k']} | "
            f"{assembled['source_summary_hit_at_k']} | "
            f"{_format_metric(assembled['source_ndcg_at_k'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def _mapping_value(record: dict[str, object], key: str) -> dict[str, object]:
    value = record[key]
    if not isinstance(value, dict):
        raise ValueError(f"Expected object for {key}")
    return value


def _format_metric(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _default_run_id(created_at: datetime, mode: str, embedder: str) -> str:
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"stage13e1-assembly-{mode}-{embedder}-{timestamp}"


def _git_output(args: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _redact_database_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    hostname = parsed.hostname or ""
    port = "" if parsed.port is None else f":{parsed.port}"
    username = "" if parsed.username is None else f"{parsed.username}:***@"
    netloc = f"{username}{hostname}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


if __name__ == "__main__":
    raise SystemExit(main())
