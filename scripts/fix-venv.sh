#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  printf '[fix-venv] .venv does not exist; nothing to repair.\n'
  exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  exit 0
fi

if ! command -v chflags >/dev/null 2>&1; then
  printf '[fix-venv] chflags is unavailable; skipping macOS flag repair.\n'
  exit 0
fi

hidden_path="$(find .venv -flags hidden -print -quit 2>/dev/null || true)"
if [[ -z "$hidden_path" ]]; then
  exit 0
fi

printf '[fix-venv] Clearing macOS hidden flags from .venv.\n'
chflags -R nouchg,nohidden .venv
