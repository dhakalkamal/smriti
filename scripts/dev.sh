#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_DIR="$ROOT_DIR/.smriti-dev"

BACKEND_PID_FILE="$DEV_DIR/backend.pid"
FRONTEND_PID_FILE="$DEV_DIR/frontend.pid"
BACKEND_LOG="$DEV_DIR/backend.log"
FRONTEND_LOG="$DEV_DIR/frontend.log"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8100"
BACKEND_URL="http://$BACKEND_HOST:$BACKEND_PORT"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="5173"
FRONTEND_URL="http://$FRONTEND_HOST:$FRONTEND_PORT"
POSTGRES_PORT="5432"
OLLAMA_HOST="127.0.0.1"
OLLAMA_PORT="11434"
OLLAMA_URL="http://$OLLAMA_HOST:$OLLAMA_PORT"
EMBEDDING_MODEL="nomic-embed-text"

cd "$ROOT_DIR"

info() {
  printf '[smriti-dev] %s\n' "$*"
}

warn() {
  printf '[smriti-dev] Warning: %s\n' "$*" >&2
}

die() {
  printf '[smriti-dev] Error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Required command '$1' was not found."
  fi
}

pid_file_for() {
  case "$1" in
    backend) printf '%s\n' "$BACKEND_PID_FILE" ;;
    frontend) printf '%s\n' "$FRONTEND_PID_FILE" ;;
    *) die "Unknown service '$1'." ;;
  esac
}

process_matches_service() {
  local service="$1"
  local pid="$2"
  local command_line

  command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  case "$service" in
    backend)
      [[ "$command_line" == *"uvicorn smriti.api:create_app"* && "$command_line" == *"--port 8100"* ]]
      ;;
    frontend)
      [[ "$command_line" == *"pnpm"* && "$command_line" == *"dev"* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

tracked_pid() {
  local service="$1"
  local pid_file
  local pid

  pid_file="$(pid_file_for "$service")"
  [[ -f "$pid_file" ]] || return 1

  pid="$(cat "$pid_file")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    warn "Removing malformed $service PID file."
    rm -f "$pid_file"
    return 1
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    warn "Removing stale $service PID file for PID $pid."
    rm -f "$pid_file"
    return 1
  fi

  if ! process_matches_service "$service" "$pid"; then
    warn "Removing stale $service PID file because PID $pid is not a tracked Smriti process."
    rm -f "$pid_file"
    return 1
  fi

  printf '%s\n' "$pid"
}

port_is_listening() {
  local port="$1"

  if command -v nc >/dev/null 2>&1; then
    nc -z "$BACKEND_HOST" "$port" >/dev/null 2>&1
    return
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return
  fi

  return 2
}

ensure_port_available() {
  local service="$1"
  local port="$2"
  local result

  if port_is_listening "$port"; then
    die "Port $port is already occupied. Refusing to start $service or stop an unrelated process."
  else
    result=$?
  fi

  if [[ "$result" -eq 2 ]]; then
    warn "Could not preflight port $port because neither nc nor lsof is available."
  fi
}

wait_for_http() {
  local service="$1"
  local url="$2"
  local attempt

  for attempt in $(seq 1 30); do
    if curl --fail --silent --show-error --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi

    if ! tracked_pid "$service" >/dev/null; then
      return 1
    fi

    sleep 1
  done

  return 1
}

collect_process_tree() {
  local pid="$1"
  local child

  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r child; do
      [[ -n "$child" ]] && collect_process_tree "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
  fi

  printf '%s\n' "$pid"
}

stop_tracked_process() {
  local service="$1"
  local pid_file
  local pid
  local process_tree
  local process
  local attempt
  local remaining=()

  pid_file="$(pid_file_for "$service")"
  if ! pid="$(tracked_pid "$service")"; then
    info "$service is not running from a tracked PID."
    return 0
  fi

  process_tree="$(collect_process_tree "$pid")"
  info "Stopping tracked $service process tree rooted at PID $pid."
  # The PID list comes only from the validated tracked process and its descendants.
  # shellcheck disable=SC2086
  kill $process_tree 2>/dev/null || true

  for attempt in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      return 0
    fi
    sleep 1
  done

  warn "$service did not stop after 10 seconds; sending SIGKILL to its tracked process tree."
  while IFS= read -r process; do
    if [[ -n "$process" ]] && kill -0 "$process" 2>/dev/null; then
      remaining+=("$process")
    fi
  done <<<"$process_tree"

  if [[ "${#remaining[@]}" -gt 0 ]]; then
    kill -KILL "${remaining[@]}" 2>/dev/null || true
  fi
  rm -f "$pid_file"
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

verify_ollama() {
  local configured_chat_model
  local missing_models=()
  local model

  info "Verifying Ollama at $OLLAMA_URL."
  if ! OLLAMA_TAGS_JSON="$(
    curl --fail --silent --show-error --connect-timeout 2 --max-time 5 "$OLLAMA_URL/api/tags"
  )"; then
    die "Ollama is not reachable at $OLLAMA_URL. Start Ollama and try again."
  fi

  configured_chat_model="$(
    uv run python -c 'from smriti.config import Settings; print(Settings().ollama_chat_model)'
  )"

  for model in "$EMBEDDING_MODEL" "$configured_chat_model"; do
    if ! model_is_installed "$model"; then
      missing_models+=("$model")
    fi
  done

  if [[ "${#missing_models[@]}" -gt 0 ]]; then
    warn "Missing required Ollama model(s): ${missing_models[*]}"
    for model in "${missing_models[@]}"; do
      printf '  ollama pull %s\n' "$model" >&2
    done
    die "Install the missing Ollama model(s) manually and run make start again."
  fi

  info "Required Ollama models are installed: $EMBEDDING_MODEL, $configured_chat_model."
}

start_backend() {
  local pid

  if pid="$(tracked_pid backend)"; then
    info "Backend is already running from tracked PID $pid."
  else
    ensure_port_available backend "$BACKEND_PORT"
    mkdir -p "$DEV_DIR"
    printf '\n[%s] Starting backend.\n' "$(date)" >>"$BACKEND_LOG"
    nohup uv run python -m uvicorn smriti.api:create_app --factory \
      --host "$BACKEND_HOST" --port "$BACKEND_PORT" >>"$BACKEND_LOG" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$BACKEND_PID_FILE"
    info "Started backend with PID $pid. Log: $BACKEND_LOG"
  fi

  if ! wait_for_http backend "$BACKEND_URL/health"; then
    warn "Backend did not become ready. Check $BACKEND_LOG."
    stop_tracked_process backend
    die "Backend readiness check failed."
  fi

  info "Backend is ready at $BACKEND_URL."
}

start_frontend() {
  local pid

  if pid="$(tracked_pid frontend)"; then
    info "Frontend is already running from tracked PID $pid."
  else
    ensure_port_available frontend "$FRONTEND_PORT"
    mkdir -p "$DEV_DIR"
    printf '\n[%s] Starting frontend.\n' "$(date)" >>"$FRONTEND_LOG"
    (
      cd "$ROOT_DIR/frontend"
      exec nohup pnpm dev
    ) >>"$FRONTEND_LOG" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$FRONTEND_PID_FILE"
    info "Started frontend with PID $pid. Log: $FRONTEND_LOG"
  fi

  if ! wait_for_http frontend "$FRONTEND_URL/"; then
    warn "Frontend did not become ready. Check $FRONTEND_LOG."
    stop_tracked_process frontend
    die "Frontend readiness check failed."
  fi

  info "Frontend is ready at $FRONTEND_URL/."
}

open_browser() {
  if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
    if ! open "$FRONTEND_URL/" >/dev/null 2>&1; then
      warn "Could not open a browser automatically. Open $FRONTEND_URL/ manually."
    fi
    return
  fi

  info "Open $FRONTEND_URL/ in a browser."
}

start() {
  require_command uv
  require_command docker
  require_command curl
  require_command pnpm

  "$ROOT_DIR/scripts/fix-venv.sh"

  info "Verifying the editable Smriti import."
  uv run python -c "import smriti"

  info "Starting Postgres and waiting for its healthcheck."
  docker compose up -d --wait postgres

  info "Applying database migrations."
  uv run migrate up

  verify_ollama
  start_backend
  start_frontend
  open_browser

  info "Smriti local development services are ready."
}

stop() {
  local compose_failed=0

  stop_tracked_process frontend
  stop_tracked_process backend

  if command -v docker >/dev/null 2>&1; then
    info "Stopping the Smriti Postgres container without deleting its volume."
    if ! docker compose stop postgres; then
      warn "Docker Compose could not stop the Smriti Postgres container."
      compose_failed=1
    fi
  else
    warn "Docker was not found; skipping the Postgres stop."
    compose_failed=1
  fi

  if [[ "$compose_failed" -ne 0 ]]; then
    return 1
  fi

  info "Smriti local development services are stopped. Ollama was left running."
}

restart() {
  stop
  start
}

logs() {
  local docker_logs_pid
  local app_logs_pid

  require_command docker
  mkdir -p "$DEV_DIR"
  touch "$BACKEND_LOG" "$FRONTEND_LOG"

  info "Following backend, frontend, and Postgres logs. Press Ctrl-C to stop following."
  docker compose logs --tail=100 --follow postgres &
  docker_logs_pid=$!
  tail -n 100 -F "$BACKEND_LOG" "$FRONTEND_LOG" &
  app_logs_pid=$!

  trap "kill $docker_logs_pid $app_logs_pid 2>/dev/null || true" EXIT INT TERM
  wait "$docker_logs_pid" "$app_logs_pid"
}

report_process_status() {
  local service="$1"
  local pid

  if pid="$(tracked_pid "$service")"; then
    printf '  %-10s running (tracked PID %s)\n' "$service:" "$pid"
  else
    printf '  %-10s not running from a tracked PID\n' "$service:"
  fi
}

report_port_status() {
  local label="$1"
  local port="$2"
  local result

  if port_is_listening "$port"; then
    printf '  %-10s listening on 127.0.0.1:%s\n' "$label:" "$port"
    return
  else
    result=$?
  fi

  if [[ "$result" -eq 2 ]]; then
    printf '  %-10s unknown (install nc or lsof to inspect port %s)\n' "$label:" "$port"
  else
    printf '  %-10s not listening on 127.0.0.1:%s\n' "$label:" "$port"
  fi
}

status() {
  info "Tracked application processes:"
  report_process_status backend
  report_process_status frontend

  printf '\n'
  info "Docker Compose Postgres status:"
  if command -v docker >/dev/null 2>&1; then
    docker compose ps postgres || warn "Docker Compose status is unavailable."
  else
    warn "Docker was not found; Docker Compose status is unavailable."
  fi

  printf '\n'
  info "Local port status:"
  report_port_status backend "$BACKEND_PORT"
  report_port_status frontend "$FRONTEND_PORT"
  report_port_status postgres "$POSTGRES_PORT"
  report_port_status ollama "$OLLAMA_PORT"
}

usage() {
  printf 'Usage: %s {start|stop|restart|logs|status}\n' "$0" >&2
  exit 1
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  logs) logs ;;
  status) status ;;
  *) usage ;;
esac
