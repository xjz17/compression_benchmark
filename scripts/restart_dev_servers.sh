#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
LOG_DIR="$ROOT_DIR/.dev-logs"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
BACKEND_SCREEN="compression_benchmark_backend"
FRONTEND_SCREEN="compression_benchmark_frontend"

mkdir -p "$LOG_DIR"

pid_cwd() {
  local pid="$1"
  lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1
}

pid_args() {
  local pid="$1"
  ps -p "$pid" -o args= 2>/dev/null || true
}

kill_if_project_listener() {
  local port="$1"
  local role="$2"
  local pids
  pids=("${(@f)$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)}")

  if (( ${#pids[@]} == 0 )); then
    echo "No $role listener on :$port"
    return 0
  fi

  for pid in "${pids[@]}"; do
    [[ -z "$pid" ]] && continue
    local cwd args
    cwd="$(pid_cwd "$pid")"
    args="$(pid_args "$pid")"

    case "$role" in
      backend)
        if [[ "$cwd" == "$BACKEND_DIR"* && "$args" == *"manage.py runserver"* ]]; then
          echo "Stopping backend pid=$pid cwd=$cwd"
          kill "$pid" 2>/dev/null || true
        else
          echo "Refusing to kill non-project backend listener pid=$pid cwd=$cwd args=$args" >&2
        fi
        ;;
      frontend)
        if [[ "$cwd" == "$FRONTEND_DIR"* && "$args" == *"vite"* ]]; then
          echo "Stopping frontend pid=$pid cwd=$cwd"
          kill "$pid" 2>/dev/null || true
        else
          echo "Refusing to kill non-project frontend listener pid=$pid cwd=$cwd args=$args" >&2
        fi
        ;;
      *)
        echo "Unknown role: $role" >&2
        return 2
        ;;
    esac
  done
}

wait_port_free() {
  local port="$1"
  local label="$2"
  for _ in {1..30}; do
    if ! lsof -nP -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  echo "$label port :$port is still occupied by a listener:" >&2
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
  return 1
}

kill_if_project_listener "$BACKEND_PORT" backend
kill_if_project_listener "$FRONTEND_PORT" frontend
screen -S "$BACKEND_SCREEN" -X quit >/dev/null 2>&1 || true
screen -S "$FRONTEND_SCREEN" -X quit >/dev/null 2>&1 || true
wait_port_free "$BACKEND_PORT" backend
wait_port_free "$FRONTEND_PORT" frontend

screen -dmS "$BACKEND_SCREEN" zsh -lc "cd ${(q)BACKEND_DIR} && exec python3 manage.py runserver 0.0.0.0:${BACKEND_PORT} > ${(q)LOG_DIR}/backend.log 2>&1"
screen -dmS "$FRONTEND_SCREEN" zsh -lc "cd ${(q)FRONTEND_DIR} && exec npm run dev -- --host 127.0.0.1 --port ${FRONTEND_PORT} > ${(q)LOG_DIR}/frontend.log 2>&1"

echo "Backend starting on http://127.0.0.1:$BACKEND_PORT/ (screen $BACKEND_SCREEN)"
echo "Frontend starting on http://127.0.0.1:$FRONTEND_PORT/benchmark/ (screen $FRONTEND_SCREEN)"
