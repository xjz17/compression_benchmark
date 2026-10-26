#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="backfill_priority_12_remaining"
LOG_FILE="$ROOT/analysis_outputs/backfill_priority_12_remaining.nohup.log"
LOG_DIR="$ROOT/analysis_outputs/backfill_priority_12_remaining"
ALGORITHMS="kanzi_9,bzip3_10,ppmd8_9,bzip2_9,DELTA_BINARY_PACKED,lzlib_9,fastlzma2_10,bsc6,brotli,reger,lzf_1,fastlz_2"

existing_sessions="$(
  (screen -ls || true) | awk -v session="$SESSION" '$1 ~ ("^[0-9]+[.]" session "$") { print $1 }'
)"

if [[ -n "$existing_sessions" ]]; then
  echo "Backfill is already running in screen session(s):"
  echo "$existing_sessions"
  exit 0
fi

mkdir -p "$LOG_DIR" "$(dirname "$LOG_FILE")"
{
  echo
  echo "==== resume $(date '+%Y-%m-%d %H:%M:%S') ===="
} >> "$LOG_FILE"

screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT' && python3 scripts/backfill_missing_staticdata.py \
    --languages java,python,cpp \
    --dtypes int,float \
    --algorithms '$ALGORITHMS' \
    --batch-size 1 \
    --timeout-sec 1800 \
    --log-dir analysis_outputs/backfill_priority_12_remaining \
    >> analysis_outputs/backfill_priority_12_remaining.nohup.log 2>&1"

echo "Started screen session: ${SESSION}"
echo "Log: ${LOG_FILE}"
