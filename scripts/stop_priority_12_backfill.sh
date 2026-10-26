#!/usr/bin/env bash
set -euo pipefail

SESSION="backfill_priority_12_remaining"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$ROOT/analysis_outputs/backfill_priority_12_remaining.nohup.log"

existing_sessions="$(
  (screen -ls || true) | awk -v session="$SESSION" '$1 ~ ("^[0-9]+[.]" session "$") { print $1 }'
)"

if [[ -z "$existing_sessions" ]]; then
  echo "No running screen session found: ${SESSION}"
  exit 0
fi

{
  echo
  echo "==== stop requested $(date '+%Y-%m-%d %H:%M:%S') ===="
} >> "$LOG_FILE"

while IFS= read -r screen_session; do
  [[ -z "$screen_session" ]] && continue
  screen -S "$screen_session" -p 0 -X stuff $'\003' || true
done <<< "$existing_sessions"

ps -axo pid=,command= \
  | awk '/scripts\\/backfill_missing_staticdata[.]py/ && /backfill_priority_12_remaining/ { print $1 }' \
  | while read -r pid; do
      [[ -z "$pid" ]] && continue
      kill -INT "$pid" 2>/dev/null || true
    done

for _ in $(seq 1 90); do
  remaining_sessions="$(
    (screen -ls || true) | awk -v session="$SESSION" '$1 ~ ("^[0-9]+[.]" session "$") { print $1 }'
  )"
  if [[ -z "$remaining_sessions" ]]; then
    echo "Stopped screen session: ${SESSION}"
    exit 0
  fi
  sleep 1
done

echo "Stop signal sent, but session is still present after 90 seconds: ${SESSION}" >&2
echo "Check recovery log before forcing anything: ${LOG_FILE}" >&2
exit 1
