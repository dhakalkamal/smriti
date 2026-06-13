from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_EVAL_ROOT = REPO_ROOT / "docs" / "evals"


def main(argv: Sequence[str] | None = None) -> int:
    """Run an offline Stage 13b role-policy replay from diagnostic output."""

    parser = argparse.ArgumentParser(
        description="Replay Stage 12 diagnostic candidates against eval-only role policies.",
    )
    parser.add_argument(
        "--input-run",
        type=Path,
        required=True,
        help="Path to a Stage 12 diagnostic run directory containing run.json/cases.jsonl.",
    )
    parser.add_argument(
        "--official-top-k",
        type=int,
        default=None,
        help="Optional official top-k override. Defaults to each case's emitted top_k.",
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=None,
        help="Optional diagnostic top-k replay depth. Defaults to emitted diagnostic_top_k.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVAL_ROOT / "runs",
        help=f"Directory containing run output folders. Defaults to {DEFAULT_EVAL_ROOT / 'runs'}.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_EVAL_ROOT / "results",
        help=f"Directory for Markdown reports. Defaults to {DEFAULT_EVAL_ROOT / 'results'}.",
    )
    args = parser.parse_args(argv)

    if args.official_top_k is not None and args.official_top_k <= 0:
        parser.error("--official-top-k must be greater than zero")
    if args.diagnostic_top_k is not None and args.diagnostic_top_k <= 0:
        parser.error("--diagnostic-top-k must be greater than zero")
    if not args.input_run.is_dir():
        parser.error(f"--input-run must be a directory: {args.input_run}")

    from smriti.memory.errors import InvalidRetrievalRequestError
    from smriti.memory.eval import (
        run_stage13_role_policy_replay,
        stage12_case_result_from_dict,
        stage13_role_policy_replay_report_to_markdown,
    )

    run_payload = _load_json_object(args.input_run / "run.json")
    metadata = _metadata_from_run_payload(run_payload)
    try:
        cases = tuple(
            stage12_case_result_from_dict(record)
            for record in _load_jsonl_objects(args.input_run / "cases.jsonl")
        )
    except InvalidRetrievalRequestError as exc:
        parser.error(str(exc))

    replay_payload = run_stage13_role_policy_replay(
        input_metadata=metadata,
        cases=cases,
        input_run_path=str(args.input_run),
        official_top_k=args.official_top_k,
        diagnostic_top_k=args.diagnostic_top_k,
    )
    replay_metadata = replay_payload["metadata"]
    if not isinstance(replay_metadata, dict):
        raise ValueError("role-policy replay payload metadata must be an object")
    input_run_id = str(replay_metadata["input_run_id"])
    run_dir = args.output_dir / input_run_id
    report_path = args.report_dir / f"{input_run_id}-role-policy-replay.md"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "role_policy_replay.json").write_text(
        json.dumps(replay_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        stage13_role_policy_replay_report_to_markdown(replay_payload),
        encoding="utf-8",
    )

    print(f"Wrote Stage 13b role policy replay output to {run_dir / 'role_policy_replay.json'}")
    print(f"Wrote Stage 13b Markdown report to {report_path}")
    return 0


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, object], value)


def _load_jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_number}")
        records.append(cast(dict[str, object], value))
    return tuple(records)


def _metadata_from_run_payload(payload: dict[str, object]) -> dict[str, object]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("run.json metadata must be an object")
    return cast(dict[str, object], metadata)


if __name__ == "__main__":
    raise SystemExit(main())
