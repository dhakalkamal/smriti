from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "frontend" / "src" / "api" / "openapi.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Export the FastAPI OpenAPI schema without starting live services."""

    parser = argparse.ArgumentParser(description="Export Smriti's FastAPI OpenAPI schema.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Schema output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    args = parser.parse_args(argv)

    from smriti.api import create_app

    output_path = args.output.resolve()
    schema = create_app().openapi()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
