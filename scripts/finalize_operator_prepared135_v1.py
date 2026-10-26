#!/usr/bin/env python3
"""Finalize the frozen prepared-selector campaign after all 3,519 columns.

This coordinator is deliberately post-selection.  It never fits or changes a
model and it never reads the per-column oracle until the reporting stage.  Run
it under the same externally enforced idle profile as the measurement job.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable


EXPECTED_COLUMNS = 3519
EXPECTED_DATASETS = 135
EXPECTED_CATALOG_SHA256 = (
    "568f2422b415d2869712a4a0e5f8edb37205555d92ecfe3fc0b36946c1d56326"
)
EXPECTED_ORACLE_SHA256 = (
    "6adef4003a23ee14253d2c04b844978ec340c056863ebabf61c11ecd632b35ba"
)
EXPECTED_SPLIT_SHA256 = (
    "d0545713f3d56aed5d9934814ca4144ab9cc8e968af032a3cffb4741cb4676a4"
)
EXPECTED_COLUMNS_SHA256 = (
    "6184f8bc0cba374c2e0e05725681d85ddee20e7f5335e9bc751c55009aaabdc8"
)
EXPECTED_TRAINING_ROWS_SHA256 = (
    "4847e00c1edfb4f13f41a8594b537722f84829767c5e2981ee94c4efe711113e"
)
EXPECTED_OLD_CLI_SHA256 = (
    "cb70075191916bd5c75a2b792dc626d5661d545164b669a6b66396ba4bf68b58"
)
EXPECTED_EXCLUDED_ACTION_IDS = [106, 107, 108, 109, 110]
RECOVERY_ACCEPTANCE_SCHEMA = (
    "operator-prepared135-protection-recovery-acceptance-v1"
)
RECOVERY_VALIDATION_SCHEMA = (
    "operator-prepared135-protection-recovery-validation-v1"
)
RECOVERY_HELPER_NAME = "reconcile_operator_prepared135_protection_v1.py"
OBJECTIVES = ("space", "encode", "decode", "balanced")
SMOKE_DTYPES = ("float64", "float32", "int64")
TWO_GIB = 2 * 1024 * 1024 * 1024


class FinalizeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizeError(message)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizeError(f"cannot read JSON {path}: {error}") from error


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def regular_file(path: Path, label: str) -> Path:
    require(path.is_file(), f"missing {label}: {path}")
    return path


def resource_profile() -> dict[str, Any]:
    profile = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "scheduler": os.sched_getscheduler(0),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "ionice": subprocess.check_output(
            ["ionice", "-p", str(os.getpid())], text=True
        ).strip(),
        "address_space_limit_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
        "observed_at": now(),
    }
    require(profile["hostname"] == "timecho-20", "finalizer must run on server 192")
    require(profile["affinity"] == [38], f"CPU affinity mismatch: {profile}")
    require(
        profile["scheduler"] == os.SCHED_IDLE,
        f"scheduler is not SCHED_IDLE: {profile}",
    )
    require(profile["nice"] == 19, f"nice level mismatch: {profile}")
    require(profile["ionice"] == "idle", f"I/O priority mismatch: {profile}")
    require(
        profile["address_space_limit_bytes"] == [TWO_GIB, TWO_GIB],
        f"address-space limit mismatch: {profile}",
    )
    return profile


class StageBook:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.directory = root / "finalize"
        self.stage_directory = self.directory / "stages"
        self.log_directory = self.directory / "logs"
        self.stage_directory.mkdir(parents=True, exist_ok=False)
        self.log_directory.mkdir(parents=True, exist_ok=False)
        self.records: list[dict[str, Any]] = []

    def _save(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        number = len(self.records)
        name = record["stage"].lower()
        atomic_json(self.stage_directory / f"{number:02d}_{name}.json", record)
        atomic_json(self.directory / "STAGES.json", self.records)

    def command(
        self,
        name: str,
        command: list[str],
        artifacts: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        number = len(self.records) + 1
        log_path = self.log_directory / f"{number:02d}_{name.lower()}.log"
        started_at = now()
        started = time.monotonic()
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            pid = process.pid
            returncode = process.wait()
        record: dict[str, Any] = {
            "stage": name,
            "status": "complete" if returncode == 0 else "failed",
            "started_at": started_at,
            "finished_at": now(),
            "wall_seconds": time.monotonic() - started,
            "command": command,
            "pid": pid,
            "parent_pid": os.getpid(),
            "returncode": returncode,
            "log_path": str(log_path),
            "log_sha256": sha256(log_path),
        }
        if returncode == 0 and artifacts is not None:
            record["artifacts"] = artifacts()
        self._save(record)
        require(returncode == 0, f"{name} failed with return code {returncode}")
        return record

    def internal(
        self,
        name: str,
        operation: Callable[[Any], dict[str, Any]],
        command: list[str],
    ) -> dict[str, Any]:
        number = len(self.records) + 1
        log_path = self.log_directory / f"{number:02d}_{name.lower()}.log"
        started_at = now()
        started = time.monotonic()
        result: dict[str, Any] | None = None
        error: Exception | None = None
        with log_path.open("xt", encoding="utf-8") as log:
            try:
                result = operation(log)
                print(json.dumps({"status": "complete", "result": result}), file=log)
            except Exception as caught:  # preserve the stage failure receipt
                error = caught
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "error_type": type(caught).__name__,
                            "error": str(caught),
                        }
                    ),
                    file=log,
                )
        record = {
            "stage": name,
            "status": "complete" if error is None else "failed",
            "started_at": started_at,
            "finished_at": now(),
            "wall_seconds": time.monotonic() - started,
            "command": command,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "returncode": 0 if error is None else 1,
            "log_path": str(log_path),
            "log_sha256": sha256(log_path),
            "result": result,
        }
        self._save(record)
        if error is not None:
            raise error
        assert result is not None
        return record


def recovery_namespace() -> dict[str, Any]:
    helper = Path(__file__).resolve().with_name(RECOVERY_HELPER_NAME)
    regular_file(helper, "kernel-aware recovery helper")
    namespace = runpy.run_path(str(helper))
    require(
        callable(namespace.get("validate_acceptance"))
        and callable(namespace.get("reconcile_protection_reports")),
        "recovery helper lacks its public validation API",
    )
    return namespace


def validate_recovery_acceptance(
    root: Path,
    receipt_path: Path,
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.resolve(strict=True)
    require(
        receipt_path == root / "RECOVERY_ACCEPTANCE.json",
        "recovery acceptance must be ROOT/RECOVERY_ACCEPTANCE.json",
    )
    require(not (root / "DONE.json").exists(), "recovery mode cannot replace DONE.json")
    regular_file(root / "FAILED.json", "original coordinator failure receipt")
    namespace = recovery_namespace() if namespace is None else namespace
    validation = namespace["validate_acceptance"](root, receipt_path)
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema") == RECOVERY_ACCEPTANCE_SCHEMA
        and receipt.get("status") == "pass",
        "recovery acceptance schema/status mismatch",
    )
    require(
        validation.get("schema") == RECOVERY_VALIDATION_SCHEMA
        and validation.get("status") == "pass"
        and validation.get("acceptance_path") == str(receipt_path)
        and validation.get("acceptance_sha256") == sha256(receipt_path),
        "recovery helper validation binding mismatch",
    )
    return {"receipt": receipt, "validation": validation, "namespace": namespace}


def wait_for_campaign(
    root: Path,
    log: Any,
    recovery_acceptance: Path | None = None,
) -> dict[str, Any]:
    checks = 0
    started = time.monotonic()
    print(json.dumps({"event": "wait_started", "at": now()}), file=log, flush=True)
    if recovery_acceptance is not None:
        recovered = validate_recovery_acceptance(root, recovery_acceptance)
        failed = regular_file(root / "FAILED.json", "original coordinator failure receipt")
        print(
            json.dumps(
                {
                    "event": "recovery_acceptance_validated",
                    "checks": 1,
                    "at": now(),
                    "acceptance_sha256": sha256(recovery_acceptance),
                }
            ),
            file=log,
            flush=True,
        )
        return {
            "poll_interval_seconds": 5,
            "checks": 1,
            "wait_seconds": time.monotonic() - started,
            "completion_evidence": "RECOVERY_ACCEPTANCE",
            "done_absent": True,
            "failed_path": str(failed),
            "failed_sha256": sha256(failed),
            "recovery_acceptance_path": str(recovery_acceptance),
            "recovery_acceptance_sha256": sha256(recovery_acceptance),
            "recovery_validation": recovered["validation"],
        }
    while True:
        checks += 1
        failed = root / "FAILED.json"
        done = root / "DONE.json"
        if failed.exists():
            failure = read_json(failed)
            raise FinalizeError(f"measurement coordinator failed: {failure}")
        if done.exists():
            value = read_json(done)
            require(value.get("status") == "complete", "DONE.json is not complete")
            print(
                json.dumps({"event": "done_observed", "checks": checks, "at": now()}),
                file=log,
                flush=True,
            )
            return {
                "poll_interval_seconds": 5,
                "checks": checks,
                "wait_seconds": time.monotonic() - started,
                "done_path": str(done),
                "done_sha256": sha256(done),
            }
        time.sleep(5)


def validate_full_campaign(root: Path, old_root: Path) -> dict[str, Any]:
    columns_path = regular_file(root / "COLUMNS.json", "frozen columns")
    split_path = regular_file(root / "split_config.json", "frozen split")
    oracle_path = regular_file(
        root / "reference/current135_column_oracle_ground_truth.jsonl",
        "frozen per-column oracle",
    )
    training_rows = regular_file(old_root / "campaign/ROWS.json", "training rows")
    require(sha256(columns_path) == EXPECTED_COLUMNS_SHA256, "COLUMNS SHA mismatch")
    require(sha256(split_path) == EXPECTED_SPLIT_SHA256, "split SHA mismatch")
    require(sha256(oracle_path) == EXPECTED_ORACLE_SHA256, "oracle SHA mismatch")
    require(
        sha256(training_rows) == EXPECTED_TRAINING_ROWS_SHA256,
        "historical training rows SHA mismatch",
    )
    source_freeze_path = regular_file(root / "SOURCE_FREEZE.json", "core source freeze")
    source_freeze = read_json(source_freeze_path)
    frozen_sources = source_freeze.get("files")
    require(isinstance(frozen_sources, dict) and frozen_sources, "SOURCE_FREEZE is empty")
    changed_sources = [
        path
        for path, expected in frozen_sources.items()
        if not Path(path).is_file() or sha256(Path(path)) != expected
    ]
    require(not changed_sources, f"core source differs from SOURCE_FREEZE: {changed_sources}")

    complete_path = regular_file(
        root / "columns3519/COMPLETE.json", "3,519-column completion receipt"
    )
    index_path = regular_file(root / "columns3519/INDEX.json", "column index")
    bindings_path = regular_file(root / "columns3519/BINDINGS.json", "column bindings")
    complete = read_json(complete_path)
    bindings = read_json(bindings_path)
    require(
        complete.get("schema") == "operator-prepared135-columns-complete-v1"
        and complete.get("status") == "complete"
        and complete.get("completed") == EXPECTED_COLUMNS
        and complete.get("failed") == 0
        and complete.get("expected") == EXPECTED_COLUMNS
        and complete.get("limited") is False,
        "COLUMNS3519 is not a complete, failure-free formal campaign",
    )
    require(complete.get("bindings") == bindings, "COMPLETE/BINDINGS mismatch")
    require(complete.get("index_sha256") == sha256(index_path), "INDEX SHA mismatch")
    require(
        bindings.get("catalog_sha256") == EXPECTED_CATALOG_SHA256,
        "formal run catalog SHA mismatch",
    )
    require(
        bindings.get("NNCP") == "OFF"
        and bindings.get("excluded_action_ids") == EXPECTED_EXCLUDED_ACTION_IDS,
        "NNCP/excluded-action policy mismatch",
    )
    index = read_json(index_path)
    require(isinstance(index, list) and len(index) == EXPECTED_COLUMNS, "INDEX coverage mismatch")
    keys: set[tuple[str, str]] = set()
    datasets: set[str] = set()
    for position, row in enumerate(index):
        require(isinstance(row, dict), f"INDEX[{position}] is not an object")
        key = (row.get("dataset_id"), row.get("column_id"))
        require(all(isinstance(value, str) and value for value in key), f"bad INDEX key {position}")
        require(key not in keys, f"duplicate INDEX key: {key}")
        require(row.get("status") == "strict_verified", f"INDEX row not strict: {key}")
        keys.add(key)
        datasets.add(key[0])
    require(len(datasets) == EXPECTED_DATASETS, "INDEX dataset coverage mismatch")

    models_path = regular_file(root / "models/MODELS.json", "formal model index")
    model_path = regular_file(root / "models/MODEL_confirm.json", "confirm model")
    models = read_json(models_path)
    model = read_json(model_path)
    require(models.get("confirm_labels_in_fit") is False, "confirm labels entered model fitting")
    require(models.get("hyperparameters_changed") is False, "formal hyperparameters changed")
    require(models.get("catalog_sha256") == EXPECTED_CATALOG_SHA256, "model catalog mismatch")
    require(model.get("candidate_catalog_sha256") == EXPECTED_CATALOG_SHA256, "confirm model catalog mismatch")
    formal = model.get("formal_protocol", {})
    require(
        formal.get("confirm_labels_in_fit") is False
        and formal.get("model_key") == "confirm"
        and formal.get("split_sha256") == EXPECTED_SPLIT_SHA256,
        "MODEL_confirm is not the frozen development-only deployment model",
    )
    model_index_row = models.get("models", {}).get("confirm", {})
    require(model_index_row.get("sha256") == sha256(model_path), "MODELS/model SHA mismatch")
    return {
        "columns": len(index),
        "datasets": len(datasets),
        "failed_columns": 0,
        "strict_verified_columns": len(index),
        "columns_sha256": sha256(columns_path),
        "split_sha256": sha256(split_path),
        "oracle_sha256": sha256(oracle_path),
        "training_rows_sha256": sha256(training_rows),
        "complete_sha256": sha256(complete_path),
        "index_sha256": sha256(index_path),
        "bindings_sha256": sha256(bindings_path),
        "models_sha256": sha256(models_path),
        "model_confirm_sha256": sha256(model_path),
        "source_freeze_sha256": sha256(source_freeze_path),
        "source_freeze_files": len(frozen_sources),
        "candidate_scope": {
            "catalog_entries": 24,
            "ranked_compression_candidates": 23,
            "raw_excluded_from_resident_ranking": True,
            "NNCP": "OFF",
            "excluded_action_ids": EXPECTED_EXCLUDED_ACTION_IDS,
        },
        "evaluation_scope": {
            "split": "frozen source-group split",
            "confirm_labels_in_fit_threshold_or_model_selection": False,
            "confirm_model": "development-only fit after frozen source-group purge",
            "generalization_claim_from_smoke": False,
        },
    }


def hash_artifacts(paths: list[Path]) -> dict[str, Any]:
    return {
        str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in paths
    }


def run_captured(
    command: list[str], log_path: Path, input_text: str | None = None
) -> dict[str, Any]:
    require(not log_path.exists(), f"refusing to overwrite process log: {log_path}")
    started_at = now()
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(input_text)
    atomic_text(
        log_path,
        json.dumps({"argv": command, "pid": process.pid})
        + "\n--- stdout ---\n"
        + stdout
        + "\n--- stderr ---\n"
        + stderr,
    )
    return {
        "argv": command,
        "pid": process.pid,
        "parent_pid": os.getpid(),
        "returncode": process.returncode,
        "started_at": started_at,
        "finished_at": now(),
        "wall_seconds": time.monotonic() - started,
        "stdout": stdout,
        "stderr": stderr,
        "log_path": str(log_path),
        "log_sha256": sha256(log_path),
    }


def select_smoke_samples(old_root: Path, split_path: Path) -> list[dict[str, Any]]:
    inputs_path = regular_file(old_root / "inputs/INPUTS.json", "historical inputs")
    inputs = read_json(inputs_path)
    split = read_json(split_path)
    samples = inputs.get("samples")
    assignments = split.get("dataset_assignments")
    require(isinstance(samples, list), "INPUTS.samples is not an array")
    require(isinstance(assignments, dict), "split lacks dataset_assignments")
    selected = []
    for dtype in SMOKE_DTYPES:
        eligible = [
            sample
            for sample in samples
            if sample.get("dtype") == dtype
            and sample.get("scale") == "u64"
            and assignments.get(sample.get("dataset_id"), {}).get("split")
            == "development"
        ]
        eligible.sort(key=lambda sample: sample.get("sample_id", ""))
        require(eligible, f"no development u64 smoke sample for {dtype}")
        selected.append(eligible[0])
    return selected


def copy_head_sample(sample: dict[str, Any], destination: Path) -> tuple[Path, dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    typed = copy.deepcopy(sample)
    typed.pop("tail", None)
    head = typed.get("head")
    require(isinstance(head, dict), "selected sample lacks head")
    values_source = regular_file(Path(head["values_path"]), "smoke head values")
    missing_source = regular_file(Path(head["missing_path"]), "smoke head missing bitmap")
    require(sha256(values_source) == head.get("values_sha256"), "head values SHA mismatch")
    require(sha256(missing_source) == head.get("missing_sha256"), "head missing SHA mismatch")
    values_destination = destination / "head.values.bin"
    missing_destination = destination / "head.missing.bitmap"
    shutil.copyfile(values_source, values_destination)
    shutil.copyfile(missing_source, missing_destination)
    head["values_path"] = str(values_destination)
    head["missing_path"] = str(missing_destination)
    require(sha256(values_destination) == head["values_sha256"], "copied values changed")
    require(sha256(missing_destination) == head["missing_sha256"], "copied bitmap changed")
    sample_path = destination / "SAMPLE.json"
    atomic_json(sample_path, typed)
    return sample_path, typed


def filtered_reference_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    expected = {
        key: copy.deepcopy(prediction[key])
        for key in (
            "schema",
            "candidate_catalog_sha256",
            "eligible_candidates",
            "unscored_candidates",
            "targets",
            "serving",
            "top3_top5_semantics",
        )
    }
    expected["eligible_candidates"] = [
        value for value in expected["eligible_candidates"] if value != "raw"
    ]
    expected["unscored_candidates"] = [
        value for value in expected["unscored_candidates"] if value != "raw"
    ]
    for objective in OBJECTIVES:
        target = expected["targets"][objective]
        ranking = [
            row for row in target["ranking"] if row.get("candidate_id") != "raw"
        ]
        target["ranking"] = ranking
        ids = [row["candidate_id"] for row in ranking]
        target["top1"] = ids[:1]
        target["top3"] = ids[:3]
        target["top5"] = ids[:5]
        expected["serving"][objective] = {
            "selected_candidate_id": ids[0] if ids else None,
            "selection": "top1_only",
            "probe_used": False,
            "probe_cost_cpu_ns": 0,
            "probe_cost_wall_ns": 0,
        }
    return expected


def resident_prediction_core(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(prediction[key])
        for key in (
            "schema",
            "candidate_catalog_sha256",
            "eligible_candidates",
            "unscored_candidates",
            "targets",
            "serving",
            "top3_top5_semantics",
        )
    }


def validate_decode(
    process: dict[str, Any],
    decode_root: Path,
    response: dict[str, Any],
    sample: dict[str, Any],
    service_pid: int,
) -> dict[str, Any]:
    require(process["returncode"] == 0, "fresh decode process failed")
    receipt_path = regular_file(decode_root / "DECODE.json", "decode receipt")
    values_path = regular_file(decode_root / "values.bin", "decoded values")
    missing_path = regular_file(decode_root / "missing.bin", "decoded missing bitmap")
    receipt = read_json(receipt_path)
    head = sample["head"]
    encoded = response["encoded"]
    checks = {
        "decode_pid_bound": receipt.get("decode_pid") == process["pid"],
        "fresh_process": process["pid"] != service_pid,
        "frame_sha_bound": receipt.get("frame_sha256") == encoded.get("frame_sha256"),
        "dtype_exact": receipt.get("dtype") == sample.get("dtype"),
        "row_count_exact": receipt.get("row_count") == head.get("row_count"),
        "values_file_exact": sha256(values_path) == head.get("values_sha256"),
        "missing_file_exact": sha256(missing_path) == head.get("missing_sha256"),
        "values_receipt_bound": receipt.get("values_sha256") == sha256(values_path),
        "missing_receipt_bound": receipt.get("missing_sha256") == sha256(missing_path),
    }
    require(all(checks.values()), f"fresh decode verification failed: {checks}")
    return {
        "process": {key: value for key, value in process.items() if key not in ("stdout", "stderr")},
        "checks": checks,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "values_sha256": sha256(values_path),
        "missing_sha256": sha256(missing_path),
    }


def run_resident_smoke(root: Path, old_root: Path, log: Any) -> dict[str, Any]:
    smoke_root = root / "serve_smoke"
    smoke_root.mkdir(parents=True, exist_ok=False)
    process_logs = smoke_root / "logs"
    process_logs.mkdir()
    model_path = regular_file(root / "models/MODEL_confirm.json", "confirm model")
    serve = regular_file(root / "serve_build/prepared_serve", "resident service")
    old_cli = regular_file(
        old_root / "build/operator_recommend135_cli", "historical frozen recommend135 CLI"
    )
    require(
        sha256(old_cli) == EXPECTED_OLD_CLI_SHA256,
        "historical frozen recommend135 CLI SHA mismatch",
    )
    delivery_old_cli = root / "delivery_bins/operator_recommend135_cli"
    delivery_old_cli.parent.mkdir(exist_ok=False)
    shutil.copy2(old_cli, delivery_old_cli)
    require(
        sha256(delivery_old_cli) == EXPECTED_OLD_CLI_SHA256,
        "copied historical recommend135 CLI SHA mismatch",
    )
    split_path = root / "split_config.json"
    originals = select_smoke_samples(old_root, split_path)
    typed_samples: list[tuple[Path, dict[str, Any]]] = []
    for original in originals:
        sample_path, sample = copy_head_sample(
            original, smoke_root / "samples" / original["dtype"]
        )
        typed_samples.append((sample_path, sample))

    frames = [smoke_root / "frames" / f"{sample['dtype']}.ngf" for _, sample in typed_samples]
    (smoke_root / "frames").mkdir()
    requests = [
        {
            "sample_path": str(sample_path),
            "objective": "balanced",
            "frame_path": str(frame),
        }
        for (sample_path, _), frame in zip(typed_samples, frames)
    ]
    stream_input = "".join(json.dumps(request, separators=(",", ":")) + "\n" for request in requests)
    service_process = run_captured(
        [str(serve), "--stream", str(model_path)],
        process_logs / "resident_stream.log",
        stream_input,
    )
    require(service_process["returncode"] == 0, "resident stream process failed")
    response_lines = [line for line in service_process["stdout"].splitlines() if line.strip()]
    require(len(response_lines) == len(requests), "resident stream response count mismatch")
    responses = [json.loads(line) for line in response_lines]
    require(
        all(response.get("status") != "request_failed" for response in responses),
        "resident stream rejected a valid smoke request",
    )
    setups = [response.get("service_setup") for response in responses]
    require(all(setup == setups[0] for setup in setups), "service setup was not reused exactly")
    require(
        [response.get("request_index") for response in responses] == [1, 2, 3]
        and [response.get("setup_reused") for response in responses]
        == [False, True, True],
        "resident request index/setup reuse contract failed",
    )

    cases = []
    for position, ((sample_path, sample), frame, response) in enumerate(
        zip(typed_samples, frames, responses), start=1
    ):
        dtype = sample["dtype"]
        require(frame.is_file(), f"resident frame missing for {dtype}")
        encoded = response.get("encoded")
        require(isinstance(encoded, dict), f"resident did not encode {dtype}")
        require(
            encoded.get("frame_sha256") == sha256(frame)
            and encoded.get("frame_bytes") == frame.stat().st_size
            and Path(encoded.get("frame_path", "")) == frame,
            f"resident frame receipt mismatch for {dtype}",
        )
        require(
            encoded.get("candidate_id") == response["selected_ids"]["balanced"],
            f"resident selected/encoded mismatch for {dtype}",
        )
        head = sample["head"]
        require(
            response.get("request_index") == position
            and response.get("sample_id") == sample.get("sample_id")
            and response.get("dtype") == dtype
            and response.get("row_count") == head.get("row_count")
            and response.get("source_values_sha256") == head.get("values_sha256")
            and response.get("source_missing_sha256") == head.get("missing_sha256")
            and response.get("model_sha256") == sha256(model_path),
            f"resident source/model binding mismatch for {dtype}",
        )

        reference_path = smoke_root / "reference" / f"{dtype}.json"
        reference_path.parent.mkdir(exist_ok=True)
        reference_process = run_captured(
            [str(old_cli), "recommend", str(model_path), str(sample_path), str(reference_path)],
            process_logs / f"reference_{dtype}.log",
        )
        require(reference_process["returncode"] == 0, f"reference recommend failed for {dtype}")
        reference = read_json(reference_path)
        expected = filtered_reference_prediction(reference)
        observed = resident_prediction_core(response)
        require(expected == observed, f"complete reference/resident ranking parity failed for {dtype}")
        require(reference.get("features") == response.get("features"), f"feature parity failed for {dtype}")

        decode_root = smoke_root / "decoded" / dtype
        decode_root.mkdir(parents=True)
        decode_process = run_captured(
            [str(old_cli), "decode", str(frame), str(decode_root)],
            process_logs / f"decode_{dtype}.log",
        )
        decode = validate_decode(
            decode_process, decode_root, response, sample, service_process["pid"]
        )
        cases.append(
            {
                "dtype": dtype,
                "dataset_id": sample["dataset_id"],
                "sample_id": sample["sample_id"],
                "scale": sample["scale"],
                "split": "development",
                "sample_path": str(sample_path),
                "sample_sha256": sha256(sample_path),
                "source_head": {
                    "row_count": head["row_count"],
                    "values_sha256": head["values_sha256"],
                    "missing_sha256": head["missing_sha256"],
                },
                "reference_process": {
                    key: value
                    for key, value in reference_process.items()
                    if key not in ("stdout", "stderr")
                },
                "reference_prediction_path": str(reference_path),
                "reference_prediction_sha256": sha256(reference_path),
                "filtered_reference_core_sha256": json_sha256(expected),
                "resident_core_sha256": json_sha256(observed),
                "complete_score_rank_parity": True,
                "feature_parity": True,
                "selected_candidate_id": encoded["candidate_id"],
                "frame": {
                    "path": str(frame),
                    "bytes": frame.stat().st_size,
                    "sha256": sha256(frame),
                },
                "fresh_decode": decode,
            }
        )

    decode_pids = [case["fresh_decode"]["process"]["pid"] for case in cases]
    require(
        len(set(decode_pids)) == len(decode_pids)
        and service_process["pid"] not in decode_pids,
        "fresh decodes did not use three distinct processes separate from the service",
    )

    wrong_model = copy.deepcopy(read_json(model_path))
    wrong_model["candidate_catalog_sha256"] = "0" * 64
    wrong_model_path = smoke_root / "negative_wrong_catalog_model.json"
    atomic_json(wrong_model_path, wrong_model)
    wrong_plan = smoke_root / "negative_wrong_catalog_plan.json"
    wrong_process = run_captured(
        [str(serve), str(wrong_model_path), str(typed_samples[0][0]), str(wrong_plan)],
        process_logs / "negative_wrong_catalog.log",
    )
    require(wrong_process["returncode"] != 0, "wrong-catalog model was accepted")
    require(not wrong_plan.exists(), "wrong-catalog invocation emitted a plan")

    bad_frame = smoke_root / "negative_bad_sample.ngf"
    bad_request = {
        "sample_path": str(smoke_root / "does-not-exist.json"),
        "objective": "balanced",
        "frame_path": str(bad_frame),
    }
    bad_process = run_captured(
        [str(serve), "--stream", str(model_path)],
        process_logs / "negative_bad_sample.log",
        json.dumps(bad_request, separators=(",", ":")) + "\n",
    )
    bad_lines = [line for line in bad_process["stdout"].splitlines() if line.strip()]
    require(bad_process["returncode"] == 0, "resident stream did not preserve service availability")
    require(len(bad_lines) == 1, "bad sample response count mismatch")
    bad_response = json.loads(bad_lines[0])
    require(bad_response.get("status") == "request_failed", "bad sample did not fail closed")
    require(not bad_frame.exists(), "bad sample emitted a frame")

    receipt = {
        "schema": "operator-prepared135-resident-smoke-v1",
        "status": "strict_verified",
        "scope": {
            "requests_in_one_resident_process": 3,
            "objective": "balanced",
            "dtypes": list(SMOKE_DTYPES),
            "scale": "u64",
            "split": "development only",
            "candidate_catalog_sha256": EXPECTED_CATALOG_SHA256,
            "ranked_candidates": 23,
            "raw_statically_filtered_from_reference": True,
            "model_fit": "development-only MODEL_confirm; no refit",
            "generalization_evidence": False,
        },
        "service": {
            "binary_path": str(serve),
            "binary_sha256": sha256(serve),
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "process": {
                key: value
                for key, value in service_process.items()
                if key not in ("stdout", "stderr")
            },
            "service_setup": setups[0],
            "request_indexes": [1, 2, 3],
            "setup_reused": [False, True, True],
        },
        "reference_cli": {
            "path": str(old_cli),
            "sha256": sha256(old_cli),
            "expected_sha256": EXPECTED_OLD_CLI_SHA256,
            "delivery_copy_path": str(delivery_old_cli),
            "delivery_copy_sha256": sha256(delivery_old_cli),
            "sample_part": "head (typed smoke SAMPLE contains head and no tail)",
        },
        "cases": cases,
        "negative_wrong_catalog": {
            "passed": True,
            "process": {
                key: value for key, value in wrong_process.items() if key not in ("stdout", "stderr")
            },
            "plan_absent": not wrong_plan.exists(),
        },
        "negative_bad_stream_sample": {
            "passed": True,
            "process": {
                key: value for key, value in bad_process.items() if key not in ("stdout", "stderr")
            },
            "response": bad_response,
            "frame_absent": not bad_frame.exists(),
            "request_level_fail_closed_process_remains_available": True,
        },
    }
    receipt_path = smoke_root / "SMOKE.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "sha256": sha256(receipt_path)}), file=log)
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "resident_pid": service_process["pid"],
        "fresh_decode_pids": decode_pids,
        "complete_score_rank_parity_cases": len(cases),
        "bit_exact_fresh_decode_cases": len(cases),
        "negative_cases": 2,
    }


def preservation_postcheck(
    root: Path,
    log: Any,
    recovery_acceptance: Path | None = None,
) -> dict[str, Any]:
    pre_path = regular_file(root / "PROTECTION_PRE.json", "pre-run protection receipt")
    coordinator_path = regular_file(
        root / "source/scripts/run_operator_prepared135_v1.py", "frozen coordinator source"
    )
    before = read_json(pre_path)
    measurement_post_path = regular_file(
        root / "PROTECTION_POST.json", "measurement-end protection receipt"
    )
    measurement_post = read_json(measurement_post_path)
    if recovery_acceptance is None:
        require(measurement_post.get("status") == "pass", "measurement protection did not pass")
    else:
        require(
            measurement_post.get("status") == "fail",
            "recovery mode requires the bound raw measurement protection failure",
        )
    measurement_post_sha = sha256(measurement_post_path)
    preserved_measurement_post = root / "finalize/MEASUREMENT_PROTECTION_POST.json"
    require(
        not preserved_measurement_post.exists(),
        "preserved measurement protection receipt already exists",
    )
    shutil.copyfile(measurement_post_path, preserved_measurement_post)
    require(
        sha256(preserved_measurement_post) == measurement_post_sha,
        "measurement protection receipt changed while preserving it",
    )
    namespace = runpy.run_path(str(coordinator_path))
    protection_post = namespace.get("protection_post")
    require(callable(protection_post), "coordinator lacks protection_post")
    if recovery_acceptance is None:
        receipt_root = root
        receipt_path = root / "PROTECTION_POST.json"
    else:
        receipt_root = root / "finalize/protection_recheck/raw"
        require(not receipt_root.exists(), "raw protection recheck already exists")
        receipt_root.mkdir(parents=True)
        receipt_path = receipt_root / "PROTECTION_POST.json"
    result = protection_post(receipt_root, before)
    regular_file(receipt_path, "post protection receipt")
    reread = read_json(receipt_path)
    require(result == reread, "protection_post return/file mismatch")
    expected_count = len(before.get("immutable_files", {})) + len(
        before.get("source_manifests", {})
    )
    require(
        reread.get("checked_immutable_files") == expected_count,
        "preservation check did not cover all pre-bound files",
    )
    recovery_record: dict[str, Any] | None = None
    final_receipt_path = receipt_path
    if recovery_acceptance is None:
        require(
            reread.get("status") == "pass"
            and reread.get("mismatch") == []
            and reread.get("settings_changed") == [],
            f"preservation postcheck failed: {reread}",
        )
    else:
        recovered = validate_recovery_acceptance(root, recovery_acceptance)
        if (
            reread.get("status") == "pass"
            and reread.get("mismatch") == []
            and reread.get("settings_changed") == []
        ):
            reconciliation = {
                "status": "pass",
                "raw_status": "pass",
                "raw_file_mismatches": 0,
                "accepted_kernel_affinity_changes": 0,
                "live_kernel_evidence": [],
            }
        else:
            reconciliation = recovered["namespace"][
                "reconcile_protection_reports"
            ](before, reread, recovered["receipt"])
        require(
            reconciliation.get("status") == "pass",
            "kernel-aware protection reconciliation failed",
        )
        require(
            sha256(measurement_post_path) == measurement_post_sha,
            "original measurement protection receipt was overwritten",
        )
        final_receipt_path = root / "PROTECTION_POST_FINAL.json"
        require(not final_receipt_path.exists(), "final protection receipt already exists")
        final_receipt = {
            "schema": "operator-prepared135-protection-post-final-v1",
            "status": "pass",
            "mode": "kernel-affinity-recovery",
            "checked_immutable_files": expected_count,
            "mismatch": [],
            "settings_changed": [],
            "accepted_kernel_affinity_changes": reconciliation.get(
                "accepted_kernel_affinity_changes"
            ),
            "raw_report_path": str(receipt_path),
            "raw_report_sha256": sha256(receipt_path),
            "raw_report_status": reread.get("status"),
            "measurement_raw": {
                "path": str(measurement_post_path),
                "sha256": measurement_post_sha,
                "status": "fail",
                "preserved_path": str(preserved_measurement_post),
                "preserved_sha256": sha256(preserved_measurement_post),
            },
            "recheck_raw": {
                "path": str(receipt_path),
                "sha256": sha256(receipt_path),
                "status": reread.get("status"),
            },
            "recovery_acceptance": {
                "path": str(recovery_acceptance),
                "sha256": sha256(recovery_acceptance),
                "schema": RECOVERY_ACCEPTANCE_SCHEMA,
                "status": "pass",
            },
            "reconciliation": reconciliation,
            "scope": {
                "original_failed_receipt_preserved": True,
                "original_measurement_protection_preserved": True,
                "done_written": False,
                "measurements_rerun": False,
                "models_modified": False,
                "results_modified": False,
            },
        }
        atomic_json(final_receipt_path, final_receipt)
        recovery_record = {
            "acceptance_path": str(recovery_acceptance),
            "acceptance_sha256": sha256(recovery_acceptance),
            "validation": recovered["validation"],
            "reconciliation": reconciliation,
            "original_failed_sha256": sha256(root / "FAILED.json"),
        }
    print(json.dumps(reread, sort_keys=True), file=log)
    return {
        "status": "pass",
        "pre_path": str(pre_path),
        "pre_sha256": sha256(pre_path),
        "measurement_post_preserved_path": str(preserved_measurement_post),
        "measurement_post_preserved_sha256": sha256(preserved_measurement_post),
        "post_path": str(final_receipt_path),
        "post_sha256": sha256(final_receipt_path),
        "raw_recheck_path": str(receipt_path),
        "raw_recheck_sha256": sha256(receipt_path),
        "checked_immutable_files": expected_count,
        "settings_changed": 0,
        "hash_mismatches": 0,
        "runpy_no_import_pyc": True,
        "recovery": recovery_record,
    }


def package_role(relative: Path) -> str:
    parts = relative.parts
    if relative.name == "DELIVERY_FILES_SHA256.json":
        return "payload_manifest"
    if "columns3519" in parts and relative.name == "RESULT.json":
        return "strict_column_result"
    if "models" in parts:
        return "frozen_model"
    if "report" in parts:
        return "post_selection_report"
    if "baseline_analysis" in parts:
        return "fixed_baseline_analysis"
    if parts and parts[0] in ("source", "serve_source", "postprocess"):
        return "source"
    if parts and parts[0] in ("build", "serve_build", "delivery_bins"):
        return "linux_binary_or_build_receipt"
    return "metadata"


def payload_files(root: Path, archive: Path, manifest: Path) -> list[Path]:
    metadata_suffixes = {".json", ".jsonl", ".md", ".log", ".sha256", ".txt"}
    source_roots = (root / "source", root / "serve_source", root / "postprocess")
    explicit_binaries = (
        root / "build/operator_recommend135_cli",
        root / "build/prepared_replay",
        root / "build/prepared_columns",
        root / "serve_build/prepared_serve",
        root / "delivery_bins/operator_recommend135_cli",
    )
    result: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path == archive or path == manifest:
            continue
        relative = path.relative_to(root)
        lower_name = path.name.lower()
        if relative == Path("finalize/STAGES.json"):
            continue
        if lower_name == "finalize.log" or lower_name.endswith("_package_delivery.log"):
            continue
        if lower_name.endswith((".tar", ".tar.gz", ".tgz")):
            continue
        if path.suffix in (".o", ".map", ".ngf", ".bin", ".bitmap"):
            continue
        if any(part.endswith(".tmp") for part in relative.parts):
            continue
        if path.suffix.lower() in metadata_suffixes:
            result.add(path)
            continue
        if any(path.is_relative_to(source_root) for source_root in source_roots if source_root.exists()):
            result.add(path)
    for binary in explicit_binaries:
        if binary.is_file():
            result.add(binary)
    required = {
        root / "build/operator_recommend135_cli",
        root / "build/prepared_replay",
        root / "build/prepared_columns",
        root / "serve_build/prepared_serve",
        root / "delivery_bins/operator_recommend135_cli",
    }
    missing = [str(path) for path in required if path not in result]
    require(not missing, f"delivery lacks required Linux binaries: {missing}")
    require(
        any(path.name == "RESULT.json" for path in result),
        "delivery lacks column RESULT receipts",
    )
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def build_archive(root: Path, log: Any) -> dict[str, Any]:
    archive = root / "operator_prepared135_v1_delivery.tar.gz"
    manifest = root / "DELIVERY_FILES_SHA256.json"
    require(not archive.exists() and not manifest.exists(), "delivery output already exists")
    files = payload_files(root, archive, manifest)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": package_role(path.relative_to(root)),
        }
        for path in files
    ]
    result_count = sum(entry["role"] == "strict_column_result" for entry in entries)
    require(result_count == EXPECTED_COLUMNS, f"delivery RESULT coverage is {result_count}")
    manifest_value = {
        "schema": "operator-prepared135-delivery-files-sha256-v1",
        "status": "complete",
        "generated_at": now(),
        "root": str(root),
        "files": entries,
        "listed_file_count": len(entries),
        "payload_file_count_including_manifest": len(entries) + 1,
        "strict_column_result_count": result_count,
        "manifest_self_hash_listed": False,
        "exclusions": [
            "all frame files",
            "all values/missing binary payloads including replay and smoke payloads",
            "intermediate object and link-map files",
            "all tar archives, including prior source archives",
            "live finalizer STAGES.json, current packaging log, and launcher FINALIZE.log; final copies remain beside the archive",
        ],
        "linux_binaries": [
            entry
            for entry in entries
            if entry["path"]
            in {
                "build/operator_recommend135_cli",
                "build/prepared_replay",
                "build/prepared_columns",
                "serve_build/prepared_serve",
                "delivery_bins/operator_recommend135_cli",
            }
        ],
    }
    atomic_json(manifest, manifest_value)
    all_files = [*files, manifest]
    with tarfile.open(archive, "x:gz", format=tarfile.PAX_FORMAT) as output:
        for path in all_files:
            output.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    archive_sha = sha256(archive)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": archive_sha,
                "payload_files": len(all_files),
            }
        ),
        file=log,
    )
    return {
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_bytes": archive.stat().st_size,
        "manifest_path": str(manifest),
        "manifest_sha256": sha256(manifest),
        "payload_file_count": len(all_files),
        "strict_column_result_count": result_count,
    }


def write_finalization_note(root: Path, coverage: dict[str, Any]) -> Path:
    path = root / "finalize/FINALIZATION.md"
    atomic_text(
        path,
        "# Prepared selector finalization\n\n"
        f"The formal run contains {coverage['columns']:,} strict verified columns "
        f"from {coverage['datasets']} datasets with zero failed columns.\n\n"
        "The primary report joins the frozen oracle only after native predictions and "
        "measurements are complete. The fixed-baseline analysis is descriptive and does "
        "not tune a selector. MODEL_confirm was fit only from the frozen development "
        "rows after the source-group purge; confirm labels did not enter fitting, "
        "threshold selection, or model selection.\n\n"
        "The resident smoke uses one development u64 sample for each of float64, "
        "float32, and int64. It verifies resident setup reuse, complete score/ranking "
        "parity after the documented raw filter, frame binding, and fresh-process "
        "bit-exact recovery. This smoke is an interface and preservation witness, not "
        "generalization evidence.\n\n"
        "The core measurement source remains bound by SOURCE_FREEZE.json. The old "
        "360-row compatibility replay used the available frozen complete MODEL.json; "
        "the original OOF fold-model artifacts are absent, so that replay does not "
        "reconstruct the original OOF evaluation and is not generalization evidence.\n",
    )
    return path


def finalize(
    root: Path,
    old_root: Path,
    recovery_acceptance: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    old_root = old_root.resolve(strict=True)
    if recovery_acceptance is not None:
        recovery_acceptance = recovery_acceptance.resolve(strict=True)
    require(root.is_dir() and old_root.is_dir(), "ROOT and OLDROOT must be directories")
    require(not (root / "FINALIZED").exists(), "ROOT already has FINALIZED")
    require(not (root / "FINALIZED.json").exists(), "ROOT already has FINALIZED.json")
    require(not (root / "finalize").exists(), "finalization evidence already exists; use a new root")
    for output in ("report", "baseline_analysis", "serve_build", "serve_smoke"):
        require(not (root / output).exists(), f"refusing to overwrite ROOT/{output}")

    book = StageBook(root)
    profile: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    try:
        profile = resource_profile()
        atomic_json(
            book.directory / "START.json",
            {
                "schema": "operator-prepared135-finalizer-start-v1",
                "status": "started",
                "root": str(root),
                "old_root": str(old_root),
                **(
                    {"recovery_acceptance": {
                        "path": str(recovery_acceptance),
                        "sha256": sha256(recovery_acceptance),
                    }}
                    if recovery_acceptance is not None
                    else {}
                ),
                "profile": profile,
                "started_at": now(),
            },
        )
        book.internal(
            "WAIT_FOR_COLUMNS3519",
            lambda log: wait_for_campaign(root, log, recovery_acceptance),
            (
                ["internal", "poll ROOT/{DONE.json,FAILED.json}", "--interval", "5s"]
                if recovery_acceptance is None
                else ["internal", "validate explicit kernel-affinity recovery acceptance"]
            ),
        )
        coverage_record = book.internal(
            "VALIDATE_COLUMNS3519",
            lambda log: (
                (lambda value: (print(json.dumps(value), file=log), value)[1])(
                    validate_full_campaign(root, old_root)
                )
            ),
            ["internal", "validate full 3519/0-failure frozen bindings"],
        )
        coverage = coverage_record["result"]

        report_command = [
            "python3",
            str(root / "postprocess/report_operator_prepared135_v1.py"),
            "--root",
            str(root / "columns3519"),
            "--oracle",
            str(root / "reference/current135_column_oracle_ground_truth.jsonl"),
            "--output",
            str(root / "report"),
            "--split",
            str(root / "split_config.json"),
            "--columns",
            str(root / "COLUMNS.json"),
            "--protocol",
            str(root / "PROTOCOL.json"),
            "--mapping-audit",
            str(root / "ORACLE_MAPPING_AUDIT.json"),
            "--system-reference-contract",
            str(root / "SYSTEM_REFERENCE_CONTRACT.json"),
            "--models",
            str(root / "models/MODELS.json"),
            "--training-rows",
            str(old_root / "campaign/ROWS.json"),
        ]
        book.command(
            "REPORT",
            report_command,
            lambda: hash_artifacts(
                [
                    root / "report/REPORT.json",
                    root / "report/REPORT.md",
                    root / "report/COLUMNS.jsonl",
                    root / "report/COMPLETE.json",
                ]
            ),
        )
        report_complete = read_json(root / "report/COMPLETE.json")
        require(
            report_complete.get("status") == "complete_strict_verified"
            and report_complete.get("verified_columns") == EXPECTED_COLUMNS
            and report_complete.get("verified_datasets") == EXPECTED_DATASETS,
            "primary report completion contract failed",
        )

        baseline_command = [
            "python3",
            str(root / "postprocess/analyze_operator_prepared135_baselines_v1.py"),
            "--root",
            str(root / "columns3519"),
            "--report",
            str(root / "report"),
            "--output",
            str(root / "baseline_analysis"),
        ]
        book.command(
            "BASELINE_ANALYSIS",
            baseline_command,
            lambda: hash_artifacts(
                [
                    root / "baseline_analysis/BASELINE_ANALYSIS.json",
                    root / "baseline_analysis/BASELINE_ANALYSIS.md",
                    root / "baseline_analysis/COLUMNS.jsonl",
                    root / "baseline_analysis/COMPLETE.json",
                ]
            ),
        )
        baseline_complete = read_json(root / "baseline_analysis/COMPLETE.json")
        require(
            baseline_complete.get("status") == "complete_descriptive_3519"
            and baseline_complete.get("columns") == EXPECTED_COLUMNS
            and baseline_complete.get("datasets") == EXPECTED_DATASETS,
            "fixed-baseline completion contract failed",
        )

        build_command = [
            "python3",
            str(root / "postprocess/build_operator_prepared_serve_v1.py"),
            "--root",
            str(root),
            "--source",
            str(root / "serve_source/prepared_serve.cpp"),
            "--output",
            str(root / "serve_build"),
        ]
        book.command(
            "BUILD_RESIDENT_SERVICE",
            build_command,
            lambda: hash_artifacts(
                [
                    root / "serve_build/prepared_serve",
                    root / "serve_build/BUILD_RECEIPT.json",
                ]
            ),
        )
        serve_build = read_json(root / "serve_build/BUILD_RECEIPT.json")
        require(
            serve_build.get("status") == "built"
            and serve_build.get("NNCP") == "OFF"
            and serve_build.get("excluded_action_ids") == EXPECTED_EXCLUDED_ACTION_IDS
            and serve_build.get("binary_sha256")
            == sha256(root / "serve_build/prepared_serve"),
            "resident build receipt failed",
        )

        book.internal(
            "RESIDENT_SMOKE",
            lambda log: run_resident_smoke(root, old_root, log),
            [str(root / "serve_build/prepared_serve"), "--stream", str(root / "models/MODEL_confirm.json")],
        )
        note_path = write_finalization_note(root, coverage)
        book.internal(
            "PRESERVATION_POSTCHECK",
            lambda log: preservation_postcheck(root, log, recovery_acceptance),
            ["runpy.run_path", str(root / "source/scripts/run_operator_prepared135_v1.py"), "protection_post"],
        )
        require(note_path.is_file(), "finalization note disappeared")
        archive_record = book.internal(
            "PACKAGE_DELIVERY",
            lambda log: build_archive(root, log),
            ["internal", "manifest-bound tar.gz delivery"],
        )
        archive = archive_record["result"]
        if recovery_acceptance is None:
            final_protection_path = root / "PROTECTION_POST.json"
            measurement_protection_path = book.directory / "MEASUREMENT_PROTECTION_POST.json"
            recovery_binding = None
        else:
            final_protection_path = root / "PROTECTION_POST_FINAL.json"
            measurement_protection_path = root / "PROTECTION_POST.json"
            acceptance = read_json(recovery_acceptance)
            recovery_binding = {
                "path": str(recovery_acceptance),
                "sha256": sha256(recovery_acceptance),
                "schema": acceptance.get("schema"),
                "status": acceptance.get("status"),
                "scope": acceptance.get("scope"),
                "validation_helper_path": str(
                    Path(__file__).resolve().with_name(RECOVERY_HELPER_NAME)
                ),
                "validation_helper_sha256": sha256(
                    Path(__file__).resolve().with_name(RECOVERY_HELPER_NAME)
                ),
                "original_failed_path": str(root / "FAILED.json"),
                "original_failed_sha256": sha256(root / "FAILED.json"),
                "done_absent": not (root / "DONE.json").exists(),
            }
        finalized = {
            "schema": "operator-prepared135-finalized-v1",
            "status": "FINALIZED",
            "finished_at": now(),
            "root": str(root),
            "old_root": str(old_root),
            "profile": profile,
            "coverage": coverage,
            "report": {
                "path": str(root / "report/REPORT.json"),
                "sha256": sha256(root / "report/REPORT.json"),
            },
            "baseline_analysis": {
                "path": str(root / "baseline_analysis/BASELINE_ANALYSIS.json"),
                "sha256": sha256(root / "baseline_analysis/BASELINE_ANALYSIS.json"),
            },
            "resident_smoke": {
                "path": str(root / "serve_smoke/SMOKE.json"),
                "sha256": sha256(root / "serve_smoke/SMOKE.json"),
                "generalization_evidence": False,
            },
            "protection_post": {
                "path": str(final_protection_path),
                "sha256": sha256(final_protection_path),
                "status": "pass",
            },
            "measurement_protection_post": {
                "path": str(measurement_protection_path),
                "sha256": sha256(measurement_protection_path),
                "status": (
                    "preserved_before_finalizer_recheck"
                    if recovery_acceptance is None
                    else "fail"
                ),
                **(
                    {
                        "accepted_by_recovery": True,
                        "classification": "bound_kernel_affinity_only",
                    }
                    if recovery_acceptance is not None
                    else {}
                ),
            },
            **(
                {"recovery_acceptance": recovery_binding}
                if recovery_binding is not None
                else {}
            ),
            "source_freeze": {
                "path": str(root / "SOURCE_FREEZE.json"),
                "sha256": sha256(root / "SOURCE_FREEZE.json"),
                "verified_files": coverage["source_freeze_files"],
            },
            "delivery": archive,
            "stages_path": str(book.directory / "STAGES.json"),
            "stages_sha256": sha256(book.directory / "STAGES.json"),
            "stage_count": len(book.records),
            "confirm_labels_used_for_fit_threshold_or_model_selection": False,
            "oracle_join_phase": "post-selection only",
            "historical_oof_gap": (
                "the old complete MODEL.json is available for same-model compatibility replay, "
                "but the original OOF fold-model artifacts are absent; no original OOF or "
                "generalization parity is claimed"
            ),
            "NNCP": "OFF",
            "excluded_action_ids": EXPECTED_EXCLUDED_ACTION_IDS,
        }
        atomic_json(root / "FINALIZED.json", finalized)
        atomic_text(root / "FINALIZED", sha256(root / "FINALIZED.json") + "\n")
        return finalized
    except Exception as error:
        failure = {
            "schema": "operator-prepared135-finalizer-failed-v1",
            "status": "failed",
            "failed_at": now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "root": str(root),
            "old_root": str(old_root),
            "profile": profile,
            "coverage": coverage,
            **(
                {"recovery_acceptance": {
                    "path": str(recovery_acceptance),
                    "sha256": sha256(recovery_acceptance),
                }}
                if recovery_acceptance is not None
                else {}
            ),
            "stages": book.records,
        }
        atomic_json(book.directory / "FAILED.json", failure)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="new prepared135 formal run root")
    parser.add_argument("--old-root", type=Path, required=True, help="frozen historical recommend135 root")
    parser.add_argument(
        "--recovery-acceptance",
        type=Path,
        help="explicit ROOT/RECOVERY_ACCEPTANCE.json for a kernel-affinity-only coordinator failure",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = finalize(args.root, args.old_root, args.recovery_acceptance)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "columns": result["coverage"]["columns"],
                    "archive_sha256": result["delivery"]["archive_sha256"],
                    "payload_file_count": result["delivery"]["payload_file_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (FinalizeError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
