#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OLLAMA_URL="http://127.0.0.1:11434"
EMBEDDING_MODEL="nomic-embed-text"
CURL_AVAILABLE=1

cd "$ROOT_DIR"

info() {
  printf '[smriti-setup] %s\n' "$*"
}

warn() {
  printf '[smriti-setup] Warning: %s\n' "$*" >&2
}

die() {
  printf '[smriti-setup] Error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Required setup command '$1' was not found. Install it and run make setup again."
  fi
}

check_runtime_prerequisites() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker was not found. Install Docker with Compose v2 before running make start."
  elif ! docker compose version >/dev/null 2>&1; then
    warn "Docker Compose v2 is unavailable. Install it before running make start."
  fi

  if ! command -v curl >/dev/null 2>&1; then
    warn "curl was not found. Install it before running make start."
    CURL_AVAILABLE=0
  fi
}

prepare_env_file() {
  if [[ -e .env ]]; then
    info "Preserving existing .env."
    return
  fi

  if [[ ! -f .env.example ]]; then
    die ".env.example was not found."
  fi

  cp .env.example .env
  info "Created .env from .env.example."
}

model_is_installed() {
  local required_model="$1"

  uv run python -c '
import json
import sys

required_model = sys.argv[1]
models = {model.get("name", "") for model in json.load(sys.stdin).get("models", [])}
is_installed = required_model in models or (
    ":" not in required_model and f"{required_model}:latest" in models
)
raise SystemExit(0 if is_installed else 1)
' "$required_model" <<<"$OLLAMA_TAGS_JSON"
}

print_pull_commands() {
  local model

  printf '[smriti-setup] Install missing Ollama models manually when needed:\n' >&2
  for model in "$@"; do
    printf '  ollama pull %s\n' "$model" >&2
  done
}

check_ollama_models() {
  local configured_chat_model="$1"
  local missing_models=()
  local model

  if [[ "$CURL_AVAILABLE" -eq 0 ]]; then
    warn "Skipping the Ollama check because curl is unavailable."
    print_pull_commands "$EMBEDDING_MODEL" "$configured_chat_model"
    return
  fi

  info "Checking Ollama at $OLLAMA_URL."
  if ! OLLAMA_TAGS_JSON="$(
    curl --fail --silent --show-error --connect-timeout 2 --max-time 5 "$OLLAMA_URL/api/tags"
  )"; then
    warn "Ollama is not reachable at $OLLAMA_URL. Start Ollama manually before running make start."
    print_pull_commands "$EMBEDDING_MODEL" "$configured_chat_model"
    return
  fi

  for model in "$EMBEDDING_MODEL" "$configured_chat_model"; do
    if ! model_is_installed "$model"; then
      missing_models+=("$model")
    fi
  done

  if [[ "${#missing_models[@]}" -gt 0 ]]; then
    warn "Missing required Ollama model(s): ${missing_models[*]}"
    print_pull_commands "${missing_models[@]}"
    return
  fi

  info "Required Ollama models are installed: $EMBEDDING_MODEL, $configured_chat_model."
}

main() {
  local configured_chat_model

  require_command uv
  require_command node
  require_command pnpm
  check_runtime_prerequisites
  prepare_env_file

  info "Installing locked Python dependencies."
  uv sync --python 3.13 --extra dev --locked

  "$ROOT_DIR/scripts/fix-venv.sh"

  info "Verifying the editable Smriti import."
  uv run python -c "import smriti; print(smriti.__file__)"

  info "Installing locked frontend dependencies."
  (
    cd "$ROOT_DIR/frontend"
    pnpm install --frozen-lockfile
  )

  configured_chat_model="$(
    uv run python -c 'from smriti.config import Settings; print(Settings().ollama_chat_model)'
  )"
  check_ollama_models "$configured_chat_model"

  printf '\nSetup complete. Start Smriti with: make start\n'
}

main "$@"
