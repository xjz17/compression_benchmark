#!/usr/bin/env python3
"""Hand off not-yet-started Python missing-benchmark chunks to the cached runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _state_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _flatten_chunks(rows: list[dict], batch_size: int) -> list[dict]:
    out: list[dict] = []
    batch_size = max(1, int(batch_size))
    for row in rows:
        missing = list(row["missing"])
        for i in range(0, len(missing), batch_size):
            chunk = dict(row)
            chunk["missing"] = missing[i : i + batch_size]
            out.append(chunk)
    return out


def _tmux_has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _tmux_capture(name: str) -> str:
    proc = subprocess.run(
        ["tmux", "capture-pane", "-pt", name, "-S", "-80"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--old-session", required=True)
    parser.add_argument("--old-log-dir", type=Path, required=True)
    parser.add_argument("--new-session", required=True)
    parser.add_argument("--new-log-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-batch-size", type=int, default=20)
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--wait-current", action="store_true")
    args = parser.parse_args()

    state_path = args.old_log_dir / "state.jsonl"
    initial_done = _state_count(state_path)
    target_done = initial_done + 1 if args.wait_current and _tmux_has_session(args.old_session) else initial_done
    print(
        f"handoff_start done={initial_done} target_done={target_done} "
        f"old_session={args.old_session}",
        flush=True,
    )
    while _tmux_has_session(args.old_session) and _state_count(state_path) < target_done:
        time.sleep(max(5, args.poll_sec))

    while _tmux_has_session(args.old_session):
        done_now = _state_count(state_path)
        if done_now > target_done:
            break
        pane = _tmux_capture(args.old_session)
        if f"RUN {target_done + 1}/" in pane or "SPLIT " not in pane:
            break
        time.sleep(max(5, args.poll_sec))

    done = _state_count(state_path)
    if _tmux_has_session(args.old_session):
        subprocess.run(["tmux", "kill-session", "-t", args.old_session], check=False)
        print(f"killed_old_session={args.old_session} done={done}", flush=True)

    plan = json.loads(args.source_plan.read_text(encoding="utf-8"))
    chunks = _flatten_chunks(plan["rows"], args.source_batch_size)
    remaining = chunks[done:]
    args.new_log_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    new_plan = args.new_log_dir / "remaining_cached_plan.json"
    new_plan.write_text(
        json.dumps(
            {
                **{k: v for k, v in plan.items() if k != "rows"},
                "handoff_done_chunks": done,
                "source_plan": str(args.source_plan),
                "source_batch_size": args.source_batch_size,
                "cache_dir": str(args.cache_dir),
                "rows": remaining,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"remaining_chunks={len(remaining)} plan={new_plan}", flush=True)
    if not remaining:
        return 0

    command = (
        f"cd {ROOT} && "
        f"export COMPRESSION_BENCHMARK_COLUMN_CSV_CACHE_DIR={args.cache_dir} && "
        f"export COMPRESSION_BENCHMARK_CSV_PATCH_ROWS=10000 && "
        f"{sys.executable} run_logs/run_missing_plan_evaluation.py "
        f"--plan {new_plan} --batch-size 999 --log-dir {args.new_log_dir} "
        f"2>&1 | tee {args.new_log_dir / 'tmux_stdout.log'}"
    )
    if _tmux_has_session(args.new_session):
        raise SystemExit(f"new tmux session already exists: {args.new_session}")
    subprocess.run(["tmux", "new-session", "-d", "-s", args.new_session, "bash", "-lc", command], check=True)
    print(f"started_new_session={args.new_session}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
