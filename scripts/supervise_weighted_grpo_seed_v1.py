#!/usr/bin/env python3
"""Hard wall-clock supervision of one explicitly launched seed process group.

Run initialization before this command and frozen validation afterwards. This
tool does not waive data/native gates, select models, or modify checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any, Sequence

MAX_SEED_SECONDS = 16 * 60 * 60


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _linux_process_identity(pid: int) -> dict[str, str] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": fields[19]}
    except (OSError, IndexError):
        return None


def _kill_owned_group(pid: int, sig: int) -> None:
    # pid is obtained solely from Popen(start_new_session=True), never from a
    # user-supplied arbitrary PID or a broad process-name/pkill search.
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _leader_exited_without_reaping(pid: int) -> bool:
    # Keep the owned leader as an unreaped child until its group is cleaned.
    # This reserves its PID, preventing it being reused as an unrelated PGID
    # between exit detection and killpg. Popen.poll() would reap it too early.
    result = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    return result is not None and result.si_pid == pid


@contextmanager
def _lock(path: Path):
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this run/seed already has an active supervisor") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def supervise(*, run_root: Path, seed: int, command: Sequence[str],
              cwd: Path, budget_seconds: float = MAX_SEED_SECONDS,
              grace_seconds: float = 1., recover_only: bool = False) -> dict[str, Any]:
    if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        raise RuntimeError("safe process-group supervision requires POSIX waitid with WNOWAIT")
    if seed not in (0, 1, 2):
        raise ValueError("seed must be one of the three formal seeds 0, 1, 2")
    if not math.isfinite(budget_seconds) or not 0 < budget_seconds <= MAX_SEED_SECONDS:
        raise ValueError("seed budget must be positive and at most 16 hours")
    if not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("grace must be finite and non-negative")
    if not command or any(not isinstance(v, str) or not v for v in command):
        raise ValueError("an explicit argv command is required")
    root = run_root.expanduser().resolve()
    working = cwd.expanduser().resolve(strict=True)
    state_dir = root / "seed_supervision"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"seed_{seed}.json"
    budget_ns = int(budget_seconds * 1e9)
    identity = {"run_root": str(root), "seed": seed, "command": list(command),
                "cwd": str(working), "budget_ns": budget_ns,
                "accounting": "training worker process wall-clock; initialization and final frozen validation separate"}
    with _lock(state_dir / f"seed_{seed}.lock"):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("identity") != identity:
                raise RuntimeError("supervisor immutable run/seed/command/budget identity mismatch")
        else:
            state = {"schema": "weighted-grpo-seed-supervision-v1", "identity": identity,
                     "consumed_wall_ns": 0, "attempts": [], "status": "ready"}
        if state["status"] == "running":
            # A killed supervisor must not grant another full budget. On Linux,
            # identify the exact still-running owned leader before stopping it.
            active = state["active_attempt"]
            saved_identity = active.get("process_identity")
            pid = active.get("pid")
            exact_owned_group = False
            if (type(pid) is int and pid > 0 and active.get("pgid") == pid
                    and saved_identity and _linux_process_identity(pid) == saved_identity):
                try:
                    exact_owned_group = os.getpgid(pid) == pid
                except ProcessLookupError:
                    exact_owned_group = False
            if exact_owned_group:
                _kill_owned_group(pid, signal.SIGKILL)
            state["consumed_wall_ns"] = budget_ns
            state["status"] = "abandoned_attempt_budget_exhausted"
            state["recovery_note"] = "uncertain unrecorded wall time conservatively charged in full"
            state["orphan_recovery"] = {
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "saved_pid": pid,
                "saved_pgid": active.get("pgid"),
                "saved_process_identity": saved_identity,
                "exact_owned_group_verified": exact_owned_group,
                "sigkill_sent_to_owned_group": exact_owned_group,
                "mode": "recover_only" if recover_only else "pre_attempt_recovery",
            }
            _atomic_json(state_path, state)
            return state
        if recover_only:
            # The caller observed a stale running ledger before acquiring this
            # lock. If the real supervisor completed in the meantime, do not
            # turn this recovery probe into a fresh training attempt.
            return state
        remaining_ns = budget_ns - int(state["consumed_wall_ns"])
        if remaining_ns <= 0:
            state["status"] = "budget_exhausted"
            _atomic_json(state_path, state)
            return state
        attempt = len(state["attempts"]) + 1
        log_path = state_dir / f"seed_{seed}_attempt_{attempt:03d}.log"
        started = time.monotonic_ns()
        hard_deadline = started + remaining_ns
        grace_ns = min(int(grace_seconds * 1e9), remaining_ns // 4)
        soft_deadline = hard_deadline - grace_ns
        process = None
        group_cleaned = False
        status = "failed"
        old_handlers = {}
        def interrupted(signum, frame):
            raise InterruptedError(f"supervisor received signal {signum}")
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, interrupted)
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(list(command), cwd=working,
                                           stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                if os.getpgid(process.pid) != process.pid:
                    raise RuntimeError("worker did not enter its isolated process group")
                state["status"] = "running"
                state["active_attempt"] = {
                    "pid": process.pid, "pgid": process.pid,
                    "process_identity": _linux_process_identity(process.pid),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "remaining_budget_ns_at_launch": remaining_ns, "log": str(log_path)}
                _atomic_json(state_path, state)
                while not _leader_exited_without_reaping(process.pid) and time.monotonic_ns() < soft_deadline:
                    time.sleep(min(.02, max(0., (soft_deadline-time.monotonic_ns()) / 1e9)))
                if not _leader_exited_without_reaping(process.pid):
                    status = "hard_deadline_reached"
                    _kill_owned_group(process.pid, signal.SIGTERM)
                    while not _leader_exited_without_reaping(process.pid) and time.monotonic_ns() < hard_deadline:
                        time.sleep(min(.01, max(0., (hard_deadline-time.monotonic_ns()) / 1e9)))
                    # Also clean descendants if the leader exited after SIGTERM.
                    _kill_owned_group(process.pid, signal.SIGKILL)
                    group_cleaned = True
                else:
                    # A leader can exit while its native descendants keep running.
                    # Reap only after cleaning its still-reserved process group.
                    _kill_owned_group(process.pid, signal.SIGKILL)
                    group_cleaned = True
                    status = "worker_exited"
                process.wait()
                if status == "worker_exited":
                    status = "complete" if process.returncode == 0 else "failed"
        except BaseException:
            status = "supervisor_interrupted_or_failed"
            raise
        finally:
            if process is not None:
                if not group_cleaned:
                    _kill_owned_group(process.pid, signal.SIGKILL)
                process.wait()
                elapsed = time.monotonic_ns() - started
                state["consumed_wall_ns"] += elapsed
                state["status"] = status
                state["attempts"].append({
                    "attempt": attempt, "pid": process.pid, "pgid": process.pid,
                    "status": status, "returncode": process.returncode,
                    "elapsed_wall_ns": elapsed,
                    "remaining_budget_ns_at_launch": remaining_ns,
                    "log": str(log_path),
                    "checkpoint_policy": "worker atomic checkpoint only; never reconstruct incomplete update",
                    "final_frozen_validation_included": False})
                state.pop("active_attempt", None)
                _atomic_json(state_path, state)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=float, default=MAX_SEED_SECONDS)
    parser.add_argument("--grace-seconds", type=float, default=1.)
    parser.add_argument("--recover-only", action="store_true",
                        help="recover one ledger-marked orphan, but never launch a new attempt")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    state = supervise(run_root=args.run_root, seed=args.seed, cwd=args.cwd,
                      budget_seconds=args.budget_seconds, grace_seconds=args.grace_seconds,
                      command=command, recover_only=args.recover_only)
    print(json.dumps(state, sort_keys=True))
    return 0 if args.recover_only or state["status"] == "complete" else 124


if __name__ == "__main__":
    raise SystemExit(main())
