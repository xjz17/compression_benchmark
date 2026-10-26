#!/usr/bin/env python3
"""Attempt every frozen THULab160 CSV dataset once, isolating failures.

This coordinator is deliberately different from the formal fail-fast runner.
It launches one fresh exporter process for one dataset and, only after that
export succeeds, one fresh independent validator process.  A dataset failure
is recorded and the remaining frozen datasets continue.  Global contract,
runtime, payload-pin, and launcher-lock failures remain fatal.

The parent launcher owns the single-writer and raw-mirror locks.  This script
never acquires, clears, or bypasses those locks and never reuses a failed
collection as a checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Sequence
import uuid

import export_project179_thulab_csv_v1 as exporter


ATTEMPT_RESULTS_SCHEMA = "project179-thulab-csv-best-effort-attempt-results-v1"
ATTEMPT_SCHEMA = "project179-thulab-csv-best-effort-dataset-attempt-v1"
PROGRESS_SCHEMA = "project179-thulab-csv-best-effort-progress-v1"
EXPECTED_DATASET_COUNT = 160
EXPECTED_TSFILE_SOURCE_COUNT = 1_524
DEFAULT_WORKERS = 10
MAX_WORKERS = 10
DEFAULT_DATASET_BYTE_CAP = 1_000_000_000
DEFAULT_MIN_FREE_TEMP_BYTES = 100_000_000_000
DEFAULT_TEMP_SPACE_FACTOR = 64.0
RAW_MANIFEST_NAME = "DOWNLOAD_MANIFEST.json"
DATASET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
FULL_VALUE_RESCAN_DATASETS = frozenset(
    {
        "ETDataset-ett",
        "trending_stocks_yahoo_finance",
        "africa_senegal_mauritania_crops_spi",
        "bitcoin",
        "dmi-aarhus-weather-data",
        "forbush_decreases",
    }
)
MACROECON_EXCEPTION_AUDIT_DATASET = "macroecon"
PILOT_RESCAN_DATASETS = FULL_VALUE_RESCAN_DATASETS | {
    MACROECON_EXCEPTION_AUDIT_DATASET
}


class BestEffortError(RuntimeError):
    """Fatal coordinator/global-contract error."""


class _TerminationRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"termination requested by signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    repo_id: str
    revision: str
    binding_sha256: str
    source_file_count: int
    source_bytes: int
    source_files: tuple[tuple[str, str], ...]
    attempt_index: int


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    scope: Path
    mirror: Path
    timestamp_policies: Path
    row_exceptions: Path
    output_root: Path
    receipts: Path
    lock_proof: Path
    python: Path
    exporter_path: Path
    validator_path: Path
    expected_exporter_sha256: str
    expected_validator_sha256: str
    dataset_byte_cap: int
    min_free_temp_bytes: int
    temp_space_factor: float
    batch_rows: int
    insert_batch_rows: int


class _ProcessRegistry:
    """Let the main signal handler settle every active child process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[tuple[str, str], subprocess.Popen[bytes]] = {}

    def add(self, dataset_id: str, stage: str, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes[(dataset_id, stage)] = process

    def remove(self, dataset_id: str, stage: str) -> None:
        with self._lock:
            self._processes.pop((dataset_id, stage), None)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_dataset_id(value: str) -> str:
    if not DATASET_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise BestEffortError(f"unsafe dataset ID in frozen scope: {value!r}")
    return value


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BestEffortError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _require_executable(path: Path, label: str) -> Path:
    """Validate an executable while preserving a venv's symlink invocation path."""

    if not path.is_absolute() or not path.exists() or not path.is_file():
        raise BestEffortError(f"{label} must be an existing absolute file: {path}")
    if not os.access(path, os.X_OK):
        raise BestEffortError(f"{label} is not executable: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise BestEffortError(f"{label} target is not a regular file: {resolved}")
    return path


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise BestEffortError(f"evidence path escapes output root: {path}") from error


def _file_reference(root: Path, path: Path) -> Mapping[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    return {
        "path": _relative(root, path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _returncode_signal(returncode: int | None) -> int | None:
    return -returncode if returncode is not None and returncode < 0 else None


def _last_nonempty_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last = stripped
    return last


def _last_json_object(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    result: Mapping[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                result = value
    return result


def _last_source_event(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    result: Mapping[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(value, Mapping)
                and value.get("event") in {"source_start", "source_complete"}
            ):
                result = value
    return result


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BestEffortError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise BestEffortError(f"{label} is not a JSON object: {path}")
    return value


def _verify_lock_proof(
    config: RunConfig,
    *,
    proc_root: Path = Path("/proc"),
) -> Mapping[str, Any]:
    """Bind the coordinator to the launcher's live raw/writer lock proof."""

    proof_path = _require_regular_file(config.lock_proof, "launcher lock proof")
    if proof_path.parent != config.receipts.resolve():
        raise BestEffortError("launcher lock proof must be directly under receipts")
    proof = _read_json(proof_path, "launcher lock proof")
    if proof.get("schema") != "project179-thulab-csv-best-effort-lock-proof-v1":
        raise BestEffortError("launcher lock proof schema mismatch")
    if proof.get("run_id") != config.run_id:
        raise BestEffortError("launcher lock proof run_id mismatch")
    parent_pid = proof.get("parent_pid")
    if isinstance(parent_pid, bool) or not isinstance(parent_pid, int) or parent_pid <= 1:
        raise BestEffortError("launcher lock proof parent_pid is invalid")
    if parent_pid != os.getppid():
        raise BestEffortError("launcher lock proof parent_pid is not this process's parent")
    try:
        os.kill(parent_pid, 0)
    except OSError as error:
        raise BestEffortError("launcher lock proof parent process is not alive") from error

    writer_text = proof.get("writer_lock")
    mirror_text = proof.get("mirror_lock")
    target_text = proof.get("fd9_target")
    fdinfo_recorded = proof.get("fd9_fdinfo")
    if not all(isinstance(item, str) and item for item in (
        writer_text, mirror_text, target_text, fdinfo_recorded
    )):
        raise BestEffortError("launcher lock proof path/fdinfo fields are invalid")
    if proof.get("fd_number") != 9:
        raise BestEffortError("launcher lock proof must bind file descriptor 9")
    if not isinstance(proof.get("created_at"), str) or not proof["created_at"]:
        raise BestEffortError("launcher lock proof created_at is missing")

    writer_lock = Path(writer_text)
    mirror_lock = Path(mirror_text)
    recorded_target = Path(target_text)
    if (
        not writer_lock.is_absolute()
        or writer_lock.is_symlink()
        or not writer_lock.is_dir()
    ):
        raise BestEffortError("writer lock must be a real existing directory")
    expected_mirror_lock = config.mirror / ".sync.lock"
    if mirror_lock != expected_mirror_lock:
        raise BestEffortError("launcher mirror lock path does not match frozen mirror")
    if mirror_lock.is_symlink() or not mirror_lock.is_file():
        raise BestEffortError("launcher mirror lock must be a regular non-symlink file")
    if recorded_target != mirror_lock:
        raise BestEffortError("launcher recorded fd9 target does not match mirror lock")

    proc_fd = proc_root / str(parent_pid) / "fd" / "9"
    proc_fdinfo = proc_root / str(parent_pid) / "fdinfo" / "9"
    try:
        live_target = Path(os.readlink(proc_fd))
        live_fdinfo = proc_fdinfo.read_text(encoding="utf-8")
    except OSError as error:
        raise BestEffortError("cannot inspect launcher's live fd9 lock") from error
    if live_target != mirror_lock:
        raise BestEffortError("launcher's live fd9 target does not match mirror lock")
    try:
        if not os.path.samefile(proc_fd, mirror_lock):
            raise BestEffortError("launcher's live fd9 inode does not match mirror lock")
    except OSError as error:
        raise BestEffortError("cannot compare launcher's live fd9 inode") from error
    for fdinfo, label in ((fdinfo_recorded, "recorded"), (live_fdinfo, "live")):
        normalized = fdinfo.upper()
        if "FLOCK" not in normalized or "WRITE" not in normalized:
            raise BestEffortError(f"launcher {label} fd9 lacks a WRITE FLOCK")
    return {
        "path": str(proof_path),
        "bytes": proof_path.stat().st_size,
        "sha256": _sha256(proof_path),
        "parent_pid": parent_pid,
        "writer_lock": str(writer_lock),
        "mirror_lock": str(mirror_lock),
    }


def _reservation_bytes(spec: DatasetSpec, config: RunConfig) -> int:
    return max(
        config.dataset_byte_cap * 2,
        int(math.ceil(spec.source_bytes * config.temp_space_factor)),
    )


def _validate_config(config: RunConfig, workers: int) -> None:
    if not 1 <= workers <= MAX_WORKERS:
        raise BestEffortError(f"workers must be in 1..{MAX_WORKERS}")
    if not DATASET_ID_RE.fullmatch(config.run_id):
        raise BestEffortError("run_id must use only letters, digits, dot, underscore, hyphen")
    if not 0 < config.dataset_byte_cap <= DEFAULT_DATASET_BYTE_CAP:
        raise BestEffortError("dataset byte cap must be in 1..1,000,000,000")
    if config.min_free_temp_bytes < DEFAULT_MIN_FREE_TEMP_BYTES:
        raise BestEffortError("minimum free temporary reserve must be at least 100 GB")
    if config.temp_space_factor < DEFAULT_TEMP_SPACE_FACTOR:
        raise BestEffortError("temporary-space factor must be at least 64")
    if config.batch_rows <= 0 or config.insert_batch_rows <= 0:
        raise BestEffortError("batch sizes must be positive")


def _verify_payload_pins(config: RunConfig) -> None:
    actual_exporter = _sha256(config.exporter_path)
    actual_validator = _sha256(config.validator_path)
    if actual_exporter != config.expected_exporter_sha256:
        raise BestEffortError(
            f"exporter SHA-256 mismatch: {actual_exporter} != "
            f"{config.expected_exporter_sha256}"
        )
    if actual_validator != config.expected_validator_sha256:
        raise BestEffortError(
            f"validator SHA-256 mismatch: {actual_validator} != "
            f"{config.expected_validator_sha256}"
        )


def _verify_global_contract_unchanged(
    config: RunConfig,
    contract: Mapping[str, Any],
) -> None:
    checks = (
        (config.scope, "scope_sha256"),
        (config.timestamp_policies, "timestamp_policy_file_sha256"),
        (config.row_exceptions, "row_exception_policy_file_sha256"),
        (config.mirror / RAW_MANIFEST_NAME, "raw_manifest_sha256"),
    )
    for path, key in checks:
        if _sha256(path) != contract.get(key):
            raise BestEffortError(f"global frozen input changed during run: {key}")


def _load_frozen_specs(
    config: RunConfig,
) -> tuple[
    list[DatasetSpec],
    Mapping[str, Any],
    Mapping[str, Mapping[str, Any]],
]:
    """Freeze exact all-160 identities without loading the native TsFile SDK.

    Native schema planning deliberately occurs only inside each one-dataset
    exporter subprocess.  A native crash or incompatible schema therefore
    becomes that dataset's export failure and cannot abort the coordinator.
    """

    _verify_payload_pins(config)
    scope, rows, scope_sha = exporter.policy_core.load_scope(config.scope)
    policies, policy_sha = exporter.policy_core.load_policies(
        config.timestamp_policies,
        scope_id=str(scope["scope_id"]),
        scope_sha256=scope_sha,
        scope_rows=rows,
    )
    exporter._check_frozen_policy_closure(config.timestamp_policies, policies)
    row_exception_policy = exporter._load_row_exception_policy(
        config.row_exceptions,
        scope_sha256=scope_sha,
        timestamp_policy_file_sha256=policy_sha,
    )
    producer_sha = _sha256(config.exporter_path)
    bindings = exporter.policy_core.bind_mirror(
        config.mirror,
        scope=scope,
        rows=rows,
        scope_sha256=scope_sha,
        policies=policies,
        policy_file_sha256=policy_sha,
        dataset_byte_cap=config.dataset_byte_cap,
        chunk_byte_cap=1,
        metadata_compressed_byte_cap=1,
        storage_layout="legacy_v1",
        producer_sha256=producer_sha,
        helper_timeseries_sha256=_sha256(exporter.policy_core.HELPER_TIMESERIES_PATH),
        helper_tsfile_sha256=_sha256(exporter.policy_core.HELPER_TSFILE_PATH),
    )
    exporter._bind_row_exception_to_bindings(row_exception_policy, bindings)
    if len(rows) != EXPECTED_DATASET_COUNT or len(bindings) != EXPECTED_DATASET_COUNT:
        raise BestEffortError("frozen scope/binding count is not exactly 160")
    source_count = sum(len(binding.source_files) for binding in bindings)
    if source_count != EXPECTED_TSFILE_SOURCE_COUNT:
        raise BestEffortError(
            f"frozen TsFile source count is {source_count}, expected 1524"
        )
    row_ids = [row.dataset_id for row in rows]
    binding_ids = [binding.row.dataset_id for binding in bindings]
    if binding_ids != row_ids or len(set(binding_ids)) != EXPECTED_DATASET_COUNT:
        raise BestEffortError("binding order/identity differs from frozen scope")
    specs = [
        DatasetSpec(
            dataset_id=_safe_dataset_id(binding.row.dataset_id),
            repo_id=binding.row.repo_id,
            revision=binding.row.revision,
            binding_sha256=binding.binding_sha256,
            source_file_count=len(binding.source_files),
            source_bytes=sum(size for _relative, _path, size, _digest in binding.source_files),
            source_files=tuple(
                (relative, digest)
                for relative, _path, _size, digest in binding.source_files
            ),
            attempt_index=index,
        )
        for index, binding in enumerate(bindings)
    ]
    raw_manifest = _require_regular_file(
        config.mirror / RAW_MANIFEST_NAME, "raw download manifest"
    )
    return specs, {
        "scope_id": str(scope["scope_id"]),
        "scope_sha256": scope_sha,
        "timestamp_policy_file_sha256": policy_sha,
        "row_exception_policy_file_sha256": row_exception_policy.file_sha256,
        "raw_manifest_sha256": _sha256(raw_manifest),
        "source_count": source_count,
        "global_schema_preflight": (
            "performed_in_isolated_dataset_export_subprocesses"
        ),
    }, {}


def _export_command(config: RunConfig, spec: DatasetSpec) -> list[str]:
    return [
        str(config.python),
        str(config.exporter_path),
        "--scope", str(config.scope),
        "--mirror", str(config.mirror),
        "--timestamp-policies", str(config.timestamp_policies),
        "--row-exceptions", str(config.row_exceptions),
        "--output", str(config.output_root / "collections" / spec.dataset_id),
        "--dataset", spec.dataset_id,
        "--workers", "1",
        "--dataset-byte-cap", str(config.dataset_byte_cap),
        "--batch-rows", str(config.batch_rows),
        "--insert-batch-rows", str(config.insert_batch_rows),
        "--min-free-temp-bytes", str(config.min_free_temp_bytes),
        "--temp-space-factor", format(config.temp_space_factor, "g"),
    ]


def _validator_command(config: RunConfig, spec: DatasetSpec) -> list[str]:
    command = [
        str(config.python),
        str(config.validator_path),
        "--scope", str(config.scope),
        "--mirror", str(config.mirror),
        "--timestamp-policies", str(config.timestamp_policies),
        "--row-exceptions", str(config.row_exceptions),
        "--output", str(config.output_root / "collections" / spec.dataset_id),
        "--expected-dataset-byte-cap", str(config.dataset_byte_cap),
        "--schema-metadata-pass",
        "--workers", "1",
    ]
    if spec.dataset_id in PILOT_RESCAN_DATASETS:
        command.extend(("--pilot-source-rescan", spec.dataset_id))
    return command


def _run_logged_command(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    dataset_id: str,
    stage: str,
    registry: _ProcessRegistry,
) -> int:
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
            )
            registry.add(dataset_id, stage, process)
            try:
                returncode = process.wait()
            finally:
                registry.remove(dataset_id, stage)
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
        return returncode
    except OSError as error:
        raise BestEffortError(
            f"cannot launch {stage} subprocess for {dataset_id}: {error}"
        ) from error


def _failure_stage_evidence(config: RunConfig, spec: DatasetSpec) -> list[Mapping[str, Any]]:
    collections = config.output_root / "collections"
    matches = sorted(collections.glob(f".{spec.dataset_id}.failed-*"))
    evidence: list[Mapping[str, Any]] = []
    for stage in matches:
        if stage.is_symlink() or not stage.is_dir():
            continue
        receipt = stage / "CSV_EXPORT_FAILURE.json"
        diagnostics = stage / exporter.WORKER_DIAGNOSTIC_DIRECTORY
        item: dict[str, Any] = {"stage_path": _relative(config.output_root, stage)}
        reference = _file_reference(config.output_root, receipt)
        if reference is not None:
            item["failure_receipt"] = reference
            try:
                item["failure_receipt_value"] = _read_json(
                    receipt, "export failure receipt"
                )
            except BestEffortError as error:
                item["failure_receipt_parse_error"] = str(error)
        diagnostic_files: list[Mapping[str, Any]] = []
        if diagnostics.is_dir() and not diagnostics.is_symlink():
            for path in sorted(diagnostics.iterdir()):
                reference = _file_reference(config.output_root, path)
                if reference is not None:
                    diagnostic_files.append(reference)
        item["worker_diagnostics"] = diagnostic_files
        evidence.append(item)
    return evidence


def _base_result(config: RunConfig, spec: DatasetSpec, reservation: int) -> dict[str, Any]:
    attempt_dir = config.output_root / "attempts" / spec.dataset_id
    return {
        "schema": ATTEMPT_SCHEMA,
        "run_id": config.run_id,
        "dataset_id": spec.dataset_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "binding_sha256": spec.binding_sha256,
        "source_file_count": spec.source_file_count,
        "source_bytes": spec.source_bytes,
        "source_files": [
            {"path": path, "sha256": digest}
            for path, digest in spec.source_files
        ],
        "reservation_bytes": reservation,
        "exactly_once_attempt_index": spec.attempt_index,
        "collection": None,
        "validation": None,
        "export_process": {"returncode": None, "signal": None},
        "validation_process": {"returncode": None, "signal": None},
        "logs": {
            "export_stdout": _relative(config.output_root, attempt_dir / "export.stdout.log"),
            "export_stderr": _relative(config.output_root, attempt_dir / "export.stderr.log"),
            "validation_stdout": _relative(config.output_root, attempt_dir / "validation.stdout.log"),
            "validation_stderr": _relative(config.output_root, attempt_dir / "validation.stderr.log"),
        },
        "failure": None,
    }


def _complete_log_references(config: RunConfig, result: dict[str, Any]) -> None:
    references: dict[str, Any] = {}
    for name, relative in result["logs"].items():
        path = config.output_root / relative
        if not path.exists():
            with path.open("xb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(path.parent)
        reference = _file_reference(config.output_root, path)
        if reference is None:
            raise BestEffortError(f"attempt log is not a regular file: {path}")
        references[name] = reference
    result["log_evidence"] = references


def _finalize_attempt_record(config: RunConfig, result: dict[str, Any]) -> None:
    """Persist the immutable attempt, then attach five refs to the final summary."""

    _complete_log_references(config, result)
    attempt_path = (
        config.output_root / "attempts" / str(result["dataset_id"]) / "ATTEMPT.json"
    )
    _write_atomic_json(attempt_path, result)
    attempt_reference = _file_reference(config.output_root, attempt_path)
    if attempt_reference is None:
        raise BestEffortError("attempt record disappeared after atomic publication")
    result["evidence"] = {
        "attempt": attempt_reference,
        **result["log_evidence"],
    }


def _write_failure_evidence(
    config: RunConfig,
    spec: DatasetSpec,
    failure: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = config.output_root / "attempts" / spec.dataset_id / "FAILURE.json"
    _write_atomic_json(
        path,
        {
            "schema": "project179-thulab-csv-best-effort-failure-v1",
            "dataset_id": spec.dataset_id,
            "recorded_at_utc": _utc_now(),
            "failure": failure,
        },
    )
    reference = _file_reference(config.output_root, path)
    if reference is None:
        raise BestEffortError("failure evidence disappeared after atomic publication")
    return reference


def _failed_result(
    config: RunConfig,
    spec: DatasetSpec,
    result: dict[str, Any],
    *,
    stage: str,
    returncode: int | None,
    message: str | None,
) -> Mapping[str, Any]:
    stdout_path = config.output_root / result["logs"].get(f"{stage}_stdout", "")
    stderr_path = config.output_root / result["logs"].get(f"{stage}_stderr", "")
    last_event = _last_json_object(stdout_path)
    last_source_event = _last_source_event(stdout_path)
    signal_number = _returncode_signal(returncode)
    native_crash_source = None
    if signal_number is not None:
        native_crash_source = (
            str(last_source_event["source_path"])
            if last_source_event is not None
            and last_source_event.get("event") == "source_start"
            and isinstance(last_source_event.get("source_path"), str)
            else "not_identified"
        )
    failure = {
        "stage": stage,
        "returncode": returncode,
        "signal": signal_number,
        "error_type": (
            "subprocess_signal"
            if signal_number is not None
            else "subprocess_nonzero"
            if returncode not in {None, 0}
            else "postcondition_failed"
        ),
        "error_message": (
            message
            or _last_nonempty_line(stderr_path)
            or f"{stage} subprocess exited with return code {returncode}"
        ),
        "last_stdout_event": last_event,
        "last_source_event": last_source_event,
        "native_crash_source": native_crash_source,
        "known_source_files": [
            {"path": path, "sha256": digest}
            for path, digest in spec.source_files
        ],
        "failed_export_stages": _failure_stage_evidence(config, spec),
    }
    failure["evidence"] = [_write_failure_evidence(config, spec, failure)]
    result.update(
        status="failed",
        stage=stage,
        finished_at_utc=_utc_now(),
        failure=failure,
    )
    _finalize_attempt_record(config, result)
    return result


def _attempt_dataset(
    config: RunConfig,
    spec: DatasetSpec,
    reservation: int,
    registry: _ProcessRegistry,
) -> Mapping[str, Any]:
    attempt_dir = config.output_root / "attempts" / spec.dataset_id
    attempt_dir.mkdir()
    result = _base_result(config, spec, reservation)
    result.update(status="running", stage="export", started_at_utc=_utc_now())
    _write_atomic_json(attempt_dir / "ATTEMPT.json", result)
    export_stdout = attempt_dir / "export.stdout.log"
    export_stderr = attempt_dir / "export.stderr.log"
    export_rc = _run_logged_command(
        _export_command(config, spec),
        stdout_path=export_stdout,
        stderr_path=export_stderr,
        dataset_id=spec.dataset_id,
        stage="export",
        registry=registry,
    )
    result["export_process"] = {
        "returncode": export_rc,
        "signal": _returncode_signal(export_rc),
    }
    if export_rc != 0:
        return _failed_result(
            config,
            spec,
            result,
            stage="export",
            returncode=export_rc,
            message=None,
        )

    collection = config.output_root / "collections" / spec.dataset_id
    manifest_path = collection / "CSV_EXPORT_MANIFEST.json"
    receipt_path = collection / spec.dataset_id / "dataset_manifest.json"
    try:
        manifest = _read_json(manifest_path, "single-dataset collection manifest")
        if (
            manifest.get("status") != "complete"
            or manifest.get("dataset_count") != 1
            or [item.get("dataset_id") for item in manifest.get("datasets", [])]
            != [spec.dataset_id]
        ):
            raise BestEffortError("exporter returned zero but collection manifest is not exact")
        _read_json(receipt_path, "single-dataset receipt")
    except BestEffortError as error:
        return _failed_result(
            config,
            spec,
            result,
            stage="export",
            returncode=export_rc,
            message=str(error),
        )
    result["collection"] = {
        "path": _relative(config.output_root, collection),
        "manifest_path": _relative(config.output_root, manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "dataset_receipt_path": _relative(config.output_root, receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
    }
    result["stage"] = "validation"
    _write_atomic_json(attempt_dir / "ATTEMPT.json", result)

    validation_stdout = attempt_dir / "validation.stdout.log"
    validation_stderr = attempt_dir / "validation.stderr.log"
    validation_rc = _run_logged_command(
        _validator_command(config, spec),
        stdout_path=validation_stdout,
        stderr_path=validation_stderr,
        dataset_id=spec.dataset_id,
        stage="validation",
        registry=registry,
    )
    result["validation_process"] = {
        "returncode": validation_rc,
        "signal": _returncode_signal(validation_rc),
    }
    if validation_rc != 0:
        return _failed_result(
            config,
            spec,
            result,
            stage="validation",
            returncode=validation_rc,
            message=None,
        )
    validation = _last_json_object(validation_stdout)
    if (
        validation is None
        or validation.get("status") != "passed"
        or validation.get("dataset_count") != 1
        or [item.get("dataset_id") for item in validation.get("datasets", [])]
        != [spec.dataset_id]
    ):
        return _failed_result(
            config,
            spec,
            result,
            stage="validation",
            returncode=validation_rc,
            message="validator returned zero without an exact passed one-dataset report",
        )
    if spec.dataset_id in FULL_VALUE_RESCAN_DATASETS:
        rescans = validation.get("pilot_source_rescans")
        if (
            not isinstance(rescans, list)
            or [item.get("dataset_id") for item in rescans] != [spec.dataset_id]
        ):
            return _failed_result(
                config,
                spec,
                result,
                stage="validation",
                returncode=validation_rc,
                message="required full-value pilot source rescan is absent",
            )
    if spec.dataset_id == MACROECON_EXCEPTION_AUDIT_DATASET:
        audits = validation.get("row_exception_audits")
        if not isinstance(audits, list) or len(audits) != 1:
            return _failed_result(
                config,
                spec,
                result,
                stage="validation",
                returncode=validation_rc,
                message="required macroecon row-exception audit is absent",
            )
    validation_path = config.output_root / "validations" / f"{spec.dataset_id}.json"
    _write_atomic_json(validation_path, validation)
    result.update(
        status="success",
        stage="complete",
        finished_at_utc=_utc_now(),
        validation={
            "path": _relative(config.output_root, validation_path),
            "sha256": _sha256(validation_path),
        },
    )
    _finalize_attempt_record(config, result)
    return result


def _admission_failure(
    config: RunConfig,
    spec: DatasetSpec,
    reservation: int,
    *,
    free_bytes: int,
    required_bytes: int,
) -> Mapping[str, Any]:
    attempt_dir = config.output_root / "attempts" / spec.dataset_id
    attempt_dir.mkdir()
    result = _base_result(config, spec, reservation)
    now = _utc_now()
    failure = {
        "stage": "admission",
        "returncode": None,
        "signal": None,
        "error_type": "insufficient_free_temporary_space",
        "error_message": (
            f"idle scheduler cannot admit dataset safely: free={free_bytes}, "
            f"required={required_bytes}, reservation={reservation}"
        ),
        "last_stdout_event": None,
        "last_source_event": None,
        "failed_export_stages": [],
    }
    failure["evidence"] = [_write_failure_evidence(config, spec, failure)]
    result.update(
        status="failed",
        stage="admission",
        started_at_utc=now,
        finished_at_utc=now,
        failure=failure,
    )
    _finalize_attempt_record(config, result)
    return result


def _schema_preflight_failure(
    config: RunConfig,
    spec: DatasetSpec,
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Record one schema-incompatible dataset without blocking the other 159."""

    attempt_dir = config.output_root / "attempts" / spec.dataset_id
    attempt_dir.mkdir()
    result = _base_result(config, spec, _reservation_bytes(spec, config))
    now = _utc_now()
    failure = {
        "stage": "export",
        "returncode": None,
        "signal": None,
        "error_type": "schema_preflight_failed",
        "error_message": str(record.get("error_message") or "schema preflight failed"),
        "last_stdout_event": None,
        "last_source_event": None,
        "failed_export_stages": [],
        "schema_preflight_record": dict(record),
    }
    failure["evidence"] = [_write_failure_evidence(config, spec, failure)]
    result.update(
        status="failed",
        stage="export",
        started_at_utc=now,
        finished_at_utc=now,
        failure=failure,
    )
    _finalize_attempt_record(config, result)
    return result


def _progress_document(
    specs: Sequence[DatasetSpec],
    results: Mapping[str, Mapping[str, Any]],
    running_ids: Sequence[str],
) -> Mapping[str, Any]:
    success = sum(item.get("status") == "success" for item in results.values())
    failed = sum(item.get("status") == "failed" for item in results.values())
    completed = success + failed
    return {
        "schema": PROGRESS_SCHEMA,
        "status": "running" if completed < len(specs) else "attempts_complete",
        "updated_at_utc": _utc_now(),
        "dataset_count": len(specs),
        "completed_count": completed,
        "success_count": success,
        "failure_count": failed,
        "running_count": len(running_ids),
        "pending_count": len(specs) - completed - len(running_ids),
        "running_dataset_ids": list(running_ids),
        "completed_dataset_ids": [
            spec.dataset_id for spec in specs if spec.dataset_id in results
        ],
    }


def _schedule_attempts(
    config: RunConfig,
    specs: Sequence[DatasetSpec],
    *,
    workers: int,
    registry: _ProcessRegistry,
    preflight_failures: Mapping[str, Mapping[str, Any]] | None = None,
    disk_free: Callable[[Path], int] | None = None,
    attempt_worker: Callable[
        [RunConfig, DatasetSpec, int, _ProcessRegistry], Mapping[str, Any]
    ] = _attempt_dataset,
) -> list[Mapping[str, Any]]:
    if disk_free is None:
        disk_free = lambda path: shutil.disk_usage(path).free
    pending_index = 0
    active_reserved = 0
    futures: dict[Future[Mapping[str, Any]], tuple[DatasetSpec, int]] = {}
    results: dict[str, Mapping[str, Any]] = {}
    collection_reserve = max(config.dataset_byte_cap, config.min_free_temp_bytes)
    if preflight_failures is None:
        preflight_failures = {}

    def publish_progress() -> None:
        running = sorted(spec.dataset_id for spec, _reservation in futures.values())
        _write_atomic_json(
            config.output_root / "LIVE_PROGRESS.json",
            _progress_document(specs, results, running),
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        publish_progress()
        while pending_index < len(specs) or futures:
            admitted = False
            while pending_index < len(specs) and len(futures) < workers:
                spec = specs[pending_index]
                preflight_failure = preflight_failures.get(spec.dataset_id)
                if preflight_failure is not None:
                    results[spec.dataset_id] = _schema_preflight_failure(
                        config, spec, preflight_failure
                    )
                    pending_index += 1
                    publish_progress()
                    continue
                reservation = _reservation_bytes(spec, config)
                free = disk_free(config.output_root)
                required = collection_reserve + active_reserved + reservation
                if free < required:
                    if futures:
                        break
                    result = _admission_failure(
                        config,
                        spec,
                        reservation,
                        free_bytes=free,
                        required_bytes=required,
                    )
                    results[spec.dataset_id] = result
                    pending_index += 1
                    publish_progress()
                    continue
                future = executor.submit(
                    attempt_worker, config, spec, reservation, registry
                )
                futures[future] = (spec, reservation)
                active_reserved += reservation
                pending_index += 1
                admitted = True
                publish_progress()
            if futures and (not admitted or pending_index >= len(specs) or len(futures) >= workers):
                completed, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    spec, reservation = futures.pop(future)
                    active_reserved -= reservation
                    result = future.result()
                    if result.get("dataset_id") != spec.dataset_id:
                        raise BestEffortError("attempt worker returned the wrong dataset ID")
                    if spec.dataset_id in results:
                        raise BestEffortError("dataset received more than one attempt result")
                    results[spec.dataset_id] = result
                publish_progress()
    if active_reserved != 0 or futures:
        raise BestEffortError("scheduler ended with active resource reservations")
    ordered = [results[spec.dataset_id] for spec in specs]
    if len(ordered) != len(specs) or len(results) != len(specs):
        raise BestEffortError("scheduler did not produce exactly one result per dataset")
    return ordered


def _build_final_document(
    config: RunConfig,
    specs: Sequence[DatasetSpec],
    contract: Mapping[str, Any],
    lock_proof: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    workers: int,
) -> Mapping[str, Any]:
    expected_ids = [spec.dataset_id for spec in specs]
    result_ids = [str(item.get("dataset_id")) for item in results]
    if len(specs) != EXPECTED_DATASET_COUNT or len(results) != EXPECTED_DATASET_COUNT:
        raise BestEffortError("final attempt summary must contain exactly 160 datasets")
    if result_ids != expected_ids or len(set(result_ids)) != EXPECTED_DATASET_COUNT:
        raise BestEffortError("final attempt summary order/identity is not frozen exact160")
    if any(item.get("status") not in {"success", "failed"} for item in results):
        raise BestEffortError("final attempt summary contains a non-terminal result")
    success_count = sum(item["status"] == "success" for item in results)
    failure_count = len(results) - success_count
    failures = [
        {
            "dataset_id": item["dataset_id"],
            "stage": item["stage"],
            "error_type": item["failure"]["error_type"],
            "error_message": item["failure"]["error_message"],
            "native_crash_source": item["failure"].get("native_crash_source"),
            "known_source_files": item["failure"].get("known_source_files", []),
            "evidence": item["failure"]["evidence"],
        }
        for item in results
        if item["status"] == "failed"
    ]
    return {
        "schema": ATTEMPT_RESULTS_SCHEMA,
        "status": (
            "attempts_complete" if failure_count == 0
            else "attempts_complete_with_failures"
        ),
        "completion_claim": "exactly_once_attempts_not_formal_160_success",
        "run_id": config.run_id,
        "completed_at_utc": _utc_now(),
        **contract,
        "exporter_sha256": config.expected_exporter_sha256,
        "validator_sha256": config.expected_validator_sha256,
        "coordinator_sha256": _sha256(Path(__file__).resolve()),
        "locks_managed_by_parent_launcher": True,
        "lock_proof": lock_proof,
        "workers": workers,
        "dataset_byte_cap": config.dataset_byte_cap,
        "min_free_temp_bytes": config.min_free_temp_bytes,
        "temp_space_factor": config.temp_space_factor,
        "dataset_count": len(specs),
        "attempted_dataset_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "dataset_ids": expected_ids,
        "success_dataset_ids": [
            item["dataset_id"] for item in results if item["status"] == "success"
        ],
        "failure_dataset_ids": [item["dataset_id"] for item in failures],
        "failures": failures,
        "results": list(results),
    }


def run_best_effort(
    config: RunConfig,
    *,
    workers: int = DEFAULT_WORKERS,
    registry: _ProcessRegistry | None = None,
) -> Mapping[str, Any]:
    _validate_config(config, workers)
    if config.output_root.exists() or config.output_root.is_symlink():
        raise BestEffortError(f"refusing to overwrite output root: {config.output_root}")
    parent = config.output_root.resolve().parent
    if not parent.is_dir() or parent.is_symlink():
        raise BestEffortError("output parent must be an existing trusted directory")
    for path, label in (
        (config.scope, "scope"),
        (config.timestamp_policies, "timestamp policy"),
        (config.row_exceptions, "row exceptions"),
        (config.exporter_path, "exporter"),
        (config.validator_path, "validator"),
    ):
        _require_regular_file(path, label)
    _require_executable(config.python, "Python runtime")
    if config.mirror.is_symlink() or not config.mirror.is_dir():
        raise BestEffortError("mirror must be a non-symlink directory")
    if config.receipts.is_symlink() or not config.receipts.is_dir():
        raise BestEffortError("receipts must be a real existing directory")
    lock_proof = _verify_lock_proof(config)
    specs, contract, preflight_failures = _load_frozen_specs(config)
    config.output_root.mkdir()
    for name in ("collections", "validations", "attempts"):
        (config.output_root / name).mkdir()
    _fsync_directory(config.output_root)

    if registry is None:
        registry = _ProcessRegistry()
    results = _schedule_attempts(
        config,
        specs,
        workers=workers,
        registry=registry,
        preflight_failures=preflight_failures,
    )
    _verify_payload_pins(config)
    _verify_global_contract_unchanged(config, contract)
    final_lock_proof = _verify_lock_proof(config)
    if final_lock_proof != lock_proof:
        raise BestEffortError("launcher lock proof changed during the run")
    final = _build_final_document(
        config,
        specs,
        contract,
        lock_proof,
        results,
        workers=workers,
    )
    _write_atomic_json(config.output_root / "ATTEMPT_RESULTS.json", final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--timestamp-policies", type=Path, required=True)
    parser.add_argument("--row-exceptions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--lock-proof", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--exporter",
        type=Path,
        default=Path(__file__).with_name("export_project179_thulab_csv_v1.py"),
    )
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path(__file__).with_name("validate_project179_thulab_csv_v1.py"),
    )
    parser.add_argument(
        "--exporter-sha256", "--expected-exporter-sha256",
        dest="expected_exporter_sha256", required=True,
    )
    parser.add_argument(
        "--validator-sha256", "--expected-validator-sha256",
        dest="expected_validator_sha256", required=True,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dataset-byte-cap", type=int, default=DEFAULT_DATASET_BYTE_CAP)
    parser.add_argument("--min-free-temp-bytes", type=int, default=DEFAULT_MIN_FREE_TEMP_BYTES)
    parser.add_argument("--temp-space-factor", type=float, default=DEFAULT_TEMP_SPACE_FACTOR)
    parser.add_argument("--batch-rows", type=int, default=exporter.DEFAULT_BATCH_ROWS)
    parser.add_argument(
        "--insert-batch-rows", type=int, default=exporter.DEFAULT_INSERT_BATCH_ROWS
    )
    args = parser.parse_args(argv)
    python_path = Path(os.path.abspath(os.fspath(args.python)))
    receipts = args.receipts.resolve()
    lock_proof = (
        args.lock_proof.resolve()
        if args.lock_proof is not None
        else receipts / "LOCK_PROOF.json"
    )
    config = RunConfig(
        run_id=args.run_id,
        scope=args.scope.resolve(),
        mirror=args.mirror.resolve(),
        timestamp_policies=args.timestamp_policies.resolve(),
        row_exceptions=args.row_exceptions.resolve(),
        output_root=args.output_root.resolve(),
        receipts=receipts,
        lock_proof=lock_proof,
        python=python_path,
        exporter_path=args.exporter.resolve(),
        validator_path=args.validator.resolve(),
        expected_exporter_sha256=args.expected_exporter_sha256,
        expected_validator_sha256=args.expected_validator_sha256,
        dataset_byte_cap=args.dataset_byte_cap,
        min_free_temp_bytes=args.min_free_temp_bytes,
        temp_space_factor=args.temp_space_factor,
        batch_rows=args.batch_rows,
        insert_batch_rows=args.insert_batch_rows,
    )
    registry = _ProcessRegistry()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)

    def terminate(signum: int, _frame: Any) -> None:
        registry.terminate_all()
        raise _TerminationRequested(signum)

    # The same registry is shared with the scheduler so signal handling first
    # asks active children to settle.  The launcher independently owns the
    # process group and keeps both locks until this coordinator exits.
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        try:
            result = run_best_effort(
                config,
                workers=args.workers,
                registry=registry,
            )
        except _TerminationRequested as error:
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            return 128 + error.signum
        except (
            BestEffortError,
            exporter.CsvExportError,
            exporter.PreparationError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
