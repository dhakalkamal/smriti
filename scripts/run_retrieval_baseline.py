from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CORPUS_PATH = REPO_ROOT / "docs" / "evals" / "corpora" / "terrafold_planted_facts_v1.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "evals"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage 12a retrieval baseline against an isolated fixture scope."""

    parser = argparse.ArgumentParser(description="Run the Stage 12a retrieval baseline.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Corpus JSONL path. Defaults to {DEFAULT_CORPUS_PATH}.",
    )
    parser.add_argument(
        "--mode",
        choices=("app_realistic", "clean_memory"),
        default="app_realistic",
        help="Retrieval timing mode. Defaults to app_realistic.",
    )
    parser.add_argument(
        "--embedder",
        choices=("fake", "ollama"),
        default="fake",
        help="Embedding backend for the baseline. Ollama is manual/on-demand only.",
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
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    from smriti.config import Settings
    from smriti.db.client import close_pool, get_pool
    from smriti.db.migrate import apply_migrations
    from smriti.embeddings import FakeEmbedder, OllamaEmbedder
    from smriti.memory import MemoryService
    from smriti.memory.eval import (
        Stage12BaselineRunMetadata,
        load_stage12_corpus,
        score_stage12_retrieval_case,
        stage12_aggregate_metrics_to_dict,
        stage12_baseline_metadata_to_dict,
        stage12_case_result_to_dict,
        summarize_stage12_results,
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
            retrieved = await service.retrieve_scoped_episodes(
                user_id=case.user_id,
                scope_id=case.scope_id,
                query=case.query,
                top_k=case.top_k,
                now=created_at,
            )
            results.append(score_stage12_retrieval_case(case, retrieved, args.mode))

        await reset_fixture_access_metadata(pool, fixture.episode_ids)
        aggregate_metrics = summarize_stage12_results(results)
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
            isolation_strategy="disposable eval scope with per-case access metadata reset",
            database_identifier=_redact_database_url(settings.database_url),
            git_branch=_git_output(("rev-parse", "--abbrev-ref", "HEAD")),
            git_sha=_git_output(("rev-parse", "HEAD")),
        )

        _write_outputs(
            output_root=args.output_root,
            run_id=run_id,
            metadata=stage12_baseline_metadata_to_dict(metadata),
            aggregate_metrics=stage12_aggregate_metrics_to_dict(aggregate_metrics),
            case_records=[stage12_case_result_to_dict(result) for result in results],
        )
    finally:
        await close_pool()

    print(f"Wrote Stage 12a run output to {args.output_root / 'runs' / run_id}")
    print(f"Wrote Stage 12a Markdown report to {args.output_root / 'results' / f'{run_id}.md'}")
    return 0


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
    lines = [
        f"# Stage 12a Retrieval Baseline: {metadata['run_id']}",
        "",
        "## Metadata",
        "",
        f"- Corpus: {metadata['corpus_id']} {metadata['corpus_version']}",
        f"- Timing mode: {metadata['timing_mode']}",
        f"- Embedder: {metadata['embedder_mode']} ({metadata['embedding_model']})",
        f"- Isolation: {metadata['isolation_strategy']}",
        f"- Git branch: {metadata['git_branch']}",
        f"- Git SHA: {metadata['git_sha']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in aggregate_metrics.items():
        if isinstance(value, dict):
            continue
        lines.append(f"| {key} | {_format_metric(value)} |")

    kind_mix = aggregate_metrics["kind_mix_at_k"]
    if isinstance(kind_mix, dict):
        lines.extend(
            [
                "",
                "## Kind Mix",
                "",
                "| Kind | Count | Ratio |",
                "| --- | ---: | ---: |",
                (
                    f"| message | {kind_mix['message_count']} | "
                    f"{_format_metric(kind_mix['message_ratio'])} |"
                ),
                (
                    f"| summary | {kind_mix['summary_count']} | "
                    f"{_format_metric(kind_mix['summary_ratio'])} |"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Per-Case Metrics",
            "",
            (
                "| Example | Type | Preferred | Hit | RR | Precision | Recall | "
                "Raw | Summary | Acceptable | Self Query | Recap Ratio |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in case_records:
        metrics = case["metrics"]
        if not isinstance(metrics, dict):
            continue
        recap = metrics["recap_pollution_at_k"]
        if not isinstance(recap, dict):
            continue
        lines.append(
            f"| {case['example_id']} | {case['question_type']} | {case['preferred_layer']} | "
            f"{metrics['hit_at_k']} | {_format_metric(metrics['reciprocal_rank'])} | "
            f"{_format_metric(metrics['precision_at_k'])} | "
            f"{_format_metric(metrics['recall_at_k'])} | {metrics['raw_hit_at_k']} | "
            f"{metrics['summary_hit_at_k']} | {metrics['acceptable_hit_at_k']} | "
            f"{metrics['self_query_hit']} | {_format_metric(recap['ratio'])} |"
        )

    failures = [
        case
        for case in case_records
        if isinstance(case["metrics"], dict) and not case["metrics"]["hit_at_k"]
    ]
    lines.extend(["", "## Failure Cases", ""])
    if not failures:
        lines.append("No preferred-hit failures.")
    else:
        for case in failures:
            retrieved = case["retrieved"]
            retrieved_ids = []
            if isinstance(retrieved, list):
                for item in retrieved:
                    if isinstance(item, dict):
                        retrieved_ids.append(str(item["episode_id"]))
            lines.append(f"- {case['example_id']}: retrieved {', '.join(retrieved_ids)}")

    self_query_cases = [
        case
        for case in case_records
        if isinstance(case["metrics"], dict) and case["metrics"]["self_query_hit"]
    ]
    lines.extend(["", "## Current-Query Self Retrieval", ""])
    if not self_query_cases:
        lines.append("No current-query self-retrieval detected.")
    else:
        for case in self_query_cases:
            metrics = case["metrics"]
            if isinstance(metrics, dict):
                lines.append(
                    f"- {case['example_id']}: rank {metrics['self_query_rank']}, "
                    f"similarity {_format_metric(metrics['self_query_similarity'])}"
                )

    recap_cases = [
        case
        for case in case_records
        if isinstance(case["metrics"], dict)
        and isinstance(case["metrics"]["recap_pollution_at_k"], dict)
        and case["metrics"]["recap_pollution_at_k"]["count"] > 0
    ]
    lines.extend(["", "## Recap Pollution", ""])
    if not recap_cases:
        lines.append("No recap-question pollution detected.")
    else:
        for case in recap_cases:
            metrics = case["metrics"]
            if not isinstance(metrics, dict):
                continue
            recap = metrics["recap_pollution_at_k"]
            if isinstance(recap, dict):
                lines.append(
                    f"- {case['example_id']}: {recap['count']} recap result(s), "
                    f"ratio {_format_metric(recap['ratio'])}"
                )

    lines.append("")
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _default_run_id(created_at: datetime, mode: str, embedder: str) -> str:
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"stage12a-{mode}-{embedder}-{timestamp}"


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
