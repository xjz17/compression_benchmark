#!/usr/bin/env python3
"""Independently gate one best-effort attempt of every frozen THULab dataset.

This gate is deliberately pure Python.  It never opens a TsFile and never
turns a failed export or validation into a successful dataset.  A formal
completion receipt is published only when all 160 per-dataset exports have a
complete, hash-bound collection and an independent validation report.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence


ATTEMPT_SCHEMA = "project179-thulab-csv-best-effort-attempt-results-v1"
COLLECTION_SCHEMA = "project179-thulab-numeric-wide-csv-collection-v1"
VALIDATION_SCHEMA = "project179-thulab-numeric-wide-csv-validation-v1"
FAILURE_SCHEMA = "project179-thulab-csv-failure-datasets-v1"
PASS_GATE_SCHEMA = "project179-thulab-csv-best-effort-completion-gate-v1"
FORMAL_GATE_SCHEMA = "project179-thulab-csv-formal-completion-gate-v1"
DATASET_CAP = 1_000_000_000
OUTPUT_FILES = ["data.csv", "series_map.csv", "columns.csv", "dataset_manifest.json"]
FULL_VALUE_DATASETS = {
    "ETDataset-ett",
    "trending_stocks_yahoo_finance",
    "africa_senegal_mauritania_crops_spi",
    "bitcoin",
    "dmi-aarhus-weather-data",
    "forbush_decreases",
}
MACRO = "macroecon"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class AttemptGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttemptGateError(f"attempt gate failed: {message}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
            f"{label} is not lowercase SHA-256")
    return value


def digest(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            output.update(block)
    return output.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    digest(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AttemptGateError(f"attempt gate failed: invalid JSON: {path}") from error
    require(isinstance(value, Mapping), f"JSON is not an object: {path}")
    return value


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    require(isinstance(value, str) and value != "", f"{label} is not a path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts),
            f"{label} is not a safe relative path")
    return path


def _under(root: Path, value: Any, label: str) -> Path:
    relative = _safe_relative(value, label)
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise AttemptGateError(f"attempt gate failed: {label} escapes output root") from error
    return candidate


def _file_ref(root: Path, value: Any, label: str, prefix: str | None = None) -> tuple[Path, Mapping[str, Any]]:
    require(isinstance(value, Mapping), f"{label} is not an object")
    path_text = value.get("path")
    path = _under(root, path_text, f"{label}.path")
    if prefix is not None:
        relative = str(_safe_relative(path_text, f"{label}.path"))
        require(relative == prefix or relative.startswith(prefix + "/"),
                f"{label}.path is outside its dataset attempt directory")
    expected_sha = _sha(value.get("sha256"), f"{label}.sha256")
    expected_bytes = value.get("bytes")
    require(_is_int(expected_bytes) and expected_bytes >= 0, f"{label}.bytes is invalid")
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    require(path.stat().st_size == expected_bytes, f"{label} byte size differs")
    require(digest(path) == expected_sha, f"{label} SHA-256 differs")
    return path, value


def _exact_directory(path: Path, names: Sequence[str], label: str) -> None:
    require(path.is_dir() and not path.is_symlink(), f"{label} is not a real directory")
    children = list(path.iterdir())
    require(all(not child.is_symlink() for child in children), f"{label} contains a symlink")
    require(sorted(child.name for child in children) == sorted(names),
            f"{label} has missing or extra entries")
    require(all(child.is_file() for child in children), f"{label} contains a non-file entry")


def _raw_bindings(raw_manifest_path: Path, raw_manifest: Mapping[str, Any], dataset_ids: list[str]) -> dict[str, Mapping[str, Any]]:
    require(raw_manifest.get("schema") == "thulab-timestamp-scope-download-v1"
            and raw_manifest.get("status") == "complete"
            and raw_manifest.get("dataset_count") == 160
            and raw_manifest.get("expected_dataset_count") == 160,
            "raw manifest schema/status differs")
    rows = raw_manifest.get("datasets")
    require(isinstance(rows, list) and len(rows) == 160, "raw manifest does not contain 160 datasets")
    require([row.get("dataset_id") for row in rows if isinstance(row, Mapping)] == dataset_ids,
            "raw manifest dataset order/scope differs")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        require(isinstance(row, Mapping), "raw manifest dataset row is not an object")
        dataset_id = str(row["dataset_id"])
        legacy_locator = row.get("receipt")
        require(isinstance(legacy_locator, str)
                and PurePosixPath(legacy_locator).name == f"{dataset_id}.json",
                f"raw manifest receipt locator differs for {dataset_id}")
        # DOWNLOAD_MANIFEST.json was relocated after its original publication;
        # its absolute receipt locators intentionally retain the legacy server
        # path.  The canonical mirror-local receipt is bound by dataset ID.
        receipt_path = raw_manifest_path.parent / "_receipts" / f"{dataset_id}.json"
        receipt = load_json(receipt_path)
        require(receipt.get("schema") == "thulab-timestamp-dataset-download-receipt-v1"
                and receipt.get("status") == "complete"
                and receipt.get("dataset_id") == dataset_id
                and receipt.get("repo_id") == row.get("repo_id")
                and receipt.get("revision") == row.get("revision"),
                f"raw receipt identity differs for {dataset_id}")
        require(_is_int(receipt.get("tsfile_count")) and receipt["tsfile_count"] >= 1,
                f"raw TsFile count is invalid for {dataset_id}")
        require(_is_int(receipt.get("tsfile_bytes")) and receipt["tsfile_bytes"] >= 0,
                f"raw TsFile bytes are invalid for {dataset_id}")
        receipt_files = receipt.get("files")
        require(isinstance(receipt_files, list), f"raw receipt files missing for {dataset_id}")
        tsfile_rows = [
            item for item in receipt_files
            if isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and item["path"].casefold().endswith(".tsfile")
        ]
        require(len(tsfile_rows) == receipt["tsfile_count"]
                and all(_is_int(item.get("bytes")) and item["bytes"] >= 0
                        and isinstance(item.get("sha256"), str)
                        and SHA_RE.fullmatch(item["sha256"]) is not None
                        for item in tsfile_rows)
                and sum(item["bytes"] for item in tsfile_rows) == receipt["tsfile_bytes"],
                f"raw receipt TsFile list/totals differ for {dataset_id}")
        output[dataset_id] = receipt
    return output


def _validate_report(report: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    require(report.get("schema") == VALIDATION_SCHEMA and report.get("status") == "passed",
            f"validation did not pass for {dataset_id}")
    require(report.get("dataset_count") == 1 and report.get("all_160_required") is False,
            f"validation selection differs for {dataset_id}")
    require(report.get("validation_workers") == 1,
            f"per-dataset validation was not isolated for {dataset_id}")
    rows = report.get("datasets")
    require(isinstance(rows, list) and len(rows) == 1
            and isinstance(rows[0], Mapping) and rows[0].get("dataset_id") == dataset_id,
            f"validation dataset differs for {dataset_id}")
    schema_rows = report.get("schema_metadata_pass")
    require(isinstance(schema_rows, list) and len(schema_rows) == 1
            and isinstance(schema_rows[0], Mapping)
            and schema_rows[0].get("dataset_id") == dataset_id
            and schema_rows[0].get("status") == "passed",
            f"schema metadata pass missing for {dataset_id}")
    nan_rows = report.get("nan_source_audits")
    require(isinstance(nan_rows, list) and len(nan_rows) == 1
            and isinstance(nan_rows[0], Mapping)
            and nan_rows[0].get("dataset_id") == dataset_id
            and nan_rows[0].get("status") == "passed",
            f"NaN source audit missing for {dataset_id}")
    rescans = report.get("pilot_source_rescans")
    row_audits = report.get("row_exception_audits")
    require(isinstance(rescans, list) and isinstance(row_audits, list),
            f"targeted validation lists missing for {dataset_id}")
    if dataset_id in FULL_VALUE_DATASETS:
        require(len(rescans) == 1 and isinstance(rescans[0], Mapping)
                and rescans[0].get("dataset_id") == dataset_id
                and rescans[0].get("status") == "passed",
                f"full source/output comparison missing for {dataset_id}")
    else:
        require(rescans == [], f"unexpected full-value rescan for {dataset_id}")
    if dataset_id == MACRO:
        require(len(row_audits) == 1
                and isinstance(row_audits[0], Mapping)
                and row_audits[0].get("dataset_id") == MACRO
                and row_audits[0].get("status") == "passed"
                and row_audits[0].get("source_files") == 64
                and row_audits[0].get("source_rows") == 1_284_546
                and row_audits[0].get("eligible_rows") == 1_284_538
                and row_audits[0].get("dropped_rows") == 8
                and isinstance(row_audits[0].get("affected_sources"), list)
                and len(row_audits[0]["affected_sources"]) == 8,
                "macroecon independent 64-source/eight-row audit missing")
    else:
        require(row_audits == [], f"unexpected row-exception audit for {dataset_id}")
    return rows[0]


def _validate_collection(
    *, root: Path, result: Mapping[str, Any], dataset_id: str,
    repo_id: Any, revision: Any, pins: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    collection = result.get("collection")
    require(isinstance(collection, Mapping), f"collection reference missing for {dataset_id}")
    expected_root = f"collections/{dataset_id}"
    require(collection.get("path") == expected_root, f"collection path differs for {dataset_id}")
    collection_root = _under(root, expected_root, f"collection root {dataset_id}")
    require(collection_root.is_dir() and not collection_root.is_symlink(),
            f"collection root is not a real directory for {dataset_id}")
    manifest_path = _under(root, collection.get("manifest_path"),
                           f"collection manifest {dataset_id}")
    require(manifest_path.is_file() and not manifest_path.is_symlink(),
            f"collection manifest is not a regular file for {dataset_id}")
    require(digest(manifest_path) == _sha(collection.get("manifest_sha256"),
                                          f"collection manifest SHA {dataset_id}"),
            f"collection manifest SHA differs for {dataset_id}")
    require(manifest_path == collection_root / "CSV_EXPORT_MANIFEST.json",
            f"collection manifest path differs for {dataset_id}")
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == COLLECTION_SCHEMA and manifest.get("status") == "complete",
            f"collection manifest is not complete for {dataset_id}")
    require(manifest.get("scope_id") == pins["scope_id"]
            and manifest.get("scope_sha256") == pins["scope_sha256"]
            and manifest.get("timestamp_policy_file_sha256") == pins["timestamp_policy_sha256"]
            and manifest.get("row_exception_policy_file_sha256") == pins["row_exception_policy_sha256"]
            and manifest.get("dataset_byte_cap") == DATASET_CAP
            and manifest.get("dataset_count") == 1,
            f"collection policy/scope/cap binding differs for {dataset_id}")
    manifest_rows = manifest.get("datasets")
    require(isinstance(manifest_rows, list) and len(manifest_rows) == 1
            and isinstance(manifest_rows[0], Mapping)
            and manifest_rows[0].get("dataset_id") == dataset_id,
            f"collection dataset differs for {dataset_id}")
    receipt_relative = _safe_relative(manifest_rows[0].get("receipt"), f"receipt in collection {dataset_id}")
    require(str(receipt_relative) == f"{dataset_id}/dataset_manifest.json",
            f"dataset receipt location differs for {dataset_id}")
    receipt_path = collection_root.joinpath(*receipt_relative.parts)
    require(str(receipt_path.relative_to(root)) == collection.get("dataset_receipt_path"),
            f"dataset receipt path differs for {dataset_id}")
    require(digest(receipt_path) == _sha(collection.get("dataset_receipt_sha256"),
                                         f"dataset receipt SHA {dataset_id}"),
            f"dataset receipt SHA differs for {dataset_id}")
    receipt = load_json(receipt_path)
    require(receipt.get("status") == "complete" and receipt.get("dataset_id") == dataset_id
            and receipt.get("repo_id") == repo_id and receipt.get("revision") == revision,
            f"dataset receipt identity differs for {dataset_id}")
    require(receipt.get("scope_id") == pins["scope_id"]
            and receipt.get("scope_sha256") == pins["scope_sha256"]
            and receipt.get("timestamp_policy_file_sha256") == pins["timestamp_policy_sha256"]
            and receipt.get("row_exception_policy_file_sha256") == pins["row_exception_policy_sha256"]
            and receipt.get("exporter_sha256") == pins["exporter_sha256"],
            f"dataset receipt payload binding differs for {dataset_id}")
    dataset_dir = receipt_path.parent
    root_children = list(collection_root.iterdir())
    require(all(not child.is_symlink() for child in root_children)
            and sorted(child.name for child in root_children)
            == sorted(["CSV_EXPORT_MANIFEST.json", dataset_id])
            and (collection_root / dataset_id).is_dir(),
            f"collection root has missing, extra, or linked entries for {dataset_id}")
    _exact_directory(dataset_dir, OUTPUT_FILES, f"dataset output {dataset_id}")
    files = receipt.get("files")
    require(isinstance(files, list) and [row.get("path") for row in files] == OUTPUT_FILES,
            f"dataset receipt file set differs for {dataset_id}")
    actual_total = 0
    for file_row in files:
        require(isinstance(file_row, Mapping), f"dataset file row is invalid for {dataset_id}")
        file_path = dataset_dir / str(file_row["path"])
        expected_bytes = file_row.get("bytes_decimal_20")
        require(isinstance(expected_bytes, str) and expected_bytes.isdigit(),
                f"dataset file bytes are invalid for {dataset_id}")
        require(file_path.stat().st_size == int(expected_bytes),
                f"dataset file size differs for {dataset_id}/{file_path.name}")
        if file_path.name != "dataset_manifest.json":
            require(digest(file_path) == _sha(file_row.get("sha256"),
                                              f"dataset file SHA {dataset_id}/{file_path.name}"),
                    f"dataset file SHA differs for {dataset_id}/{file_path.name}")
        else:
            require(file_row.get("sha256") is None,
                    f"self-referential receipt SHA must be null for {dataset_id}")
        actual_total += file_path.stat().st_size
    total_text = receipt.get("actual_total_bytes_decimal_20")
    cap_text = receipt.get("cap_bytes_decimal_20")
    require(isinstance(total_text, str) and total_text.isdigit()
            and isinstance(cap_text, str) and cap_text.isdigit()
            and int(total_text) == actual_total
            and int(cap_text) == DATASET_CAP
            and actual_total <= DATASET_CAP
            and manifest_rows[0].get("actual_capped_bytes") == actual_total,
            f"dataset cap/total differs for {dataset_id}")
    return manifest, receipt, dataset_dir


def _require_result_identity(
    result: Mapping[str, Any], scope_row: Mapping[str, Any], raw: Mapping[str, Any],
    index: int, pins: Mapping[str, str],
) -> None:
    dataset_id = str(scope_row["id"])
    require(result.get("dataset_id") == dataset_id
            and result.get("repo_id") == scope_row.get("repo_id")
            and result.get("revision") == scope_row.get("revision"),
            f"attempt result identity differs at frozen ordinal {index}")
    require(result.get("exactly_once_attempt_index") == index,
            f"attempt ordinal differs for {dataset_id}")
    _sha(result.get("binding_sha256"), f"binding SHA {dataset_id}")
    require(result.get("source_file_count") == raw["tsfile_count"]
            and result.get("source_bytes") == raw["tsfile_bytes"],
            f"attempt raw-source totals differ for {dataset_id}")
    raw_files = raw.get("files")
    require(isinstance(raw_files, list), f"raw receipt file list is missing for {dataset_id}")
    expected_sources = [
        {"path": item.get("path"), "sha256": item.get("sha256")}
        for item in raw_files
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and item["path"].casefold().endswith(".tsfile")
    ]
    require(len(expected_sources) == raw["tsfile_count"]
            and all(isinstance(item["path"], str)
                    and SHA_RE.fullmatch(str(item["sha256"])) is not None
                    for item in expected_sources)
            and result.get("source_files") == expected_sources,
            f"attempt frozen source path/SHA list differs for {dataset_id}")
    require(_is_int(result.get("reservation_bytes")) and result["reservation_bytes"] >= 0,
            f"attempt reservation bytes are invalid for {dataset_id}")
    for key, expected in (
        ("scope_sha256", pins["scope_sha256"]),
        ("raw_manifest_sha256", pins["raw_manifest_sha256"]),
        ("timestamp_policy_file_sha256", pins["timestamp_policy_sha256"]),
        ("row_exception_policy_file_sha256", pins["row_exception_policy_sha256"]),
        ("exporter_sha256", pins["exporter_sha256"]),
        ("validator_sha256", pins["validator_sha256"]),
    ):
        if key in result:
            require(result[key] == expected, f"attempt result {key} differs for {dataset_id}")


def _check_process(value: Any, label: str, *, must_succeed: bool) -> tuple[int | None, int | None]:
    require(isinstance(value, Mapping), f"{label} process result is missing")
    returncode, signal = value.get("returncode"), value.get("signal")
    require(returncode is None or _is_int(returncode), f"{label} returncode is invalid")
    require(signal is None or (_is_int(signal) and signal > 0), f"{label} signal is invalid")
    if returncode is None or returncode >= 0:
        require(signal is None, f"{label} has a signal without a negative returncode")
    if must_succeed:
        require(returncode == 0 and signal is None, f"{label} subprocess did not succeed")
    elif returncode is not None and returncode < 0:
        require(signal == -returncode, f"{label} signal/returncode differ")
    return returncode, signal


def audit(args: argparse.Namespace) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str | None]:
    root = args.output_root
    require(root.is_dir() and not root.is_symlink(), "output root is not a real directory")
    attempt_path = args.attempt_results
    require(attempt_path == root / "ATTEMPT_RESULTS.json", "attempt results path is not canonical")
    attempt_sha = digest(attempt_path)
    scope_sha = digest(args.scope)
    raw_sha = digest(args.raw_manifest)
    require(scope_sha == _sha(args.expected_scope_sha256, "expected scope SHA"), "scope SHA differs")
    require(raw_sha == _sha(args.expected_raw_manifest_sha256, "expected raw manifest SHA"),
            "raw manifest SHA differs")
    pins = {
        "scope_id": args.expected_scope_id,
        "scope_sha256": scope_sha,
        "raw_manifest_sha256": raw_sha,
        "timestamp_policy_sha256": _sha(args.expected_timestamp_policy_sha256, "timestamp policy SHA"),
        "row_exception_policy_sha256": _sha(args.expected_row_exception_policy_sha256, "row-exception policy SHA"),
        "exporter_sha256": _sha(args.expected_exporter_sha256, "exporter SHA"),
        "validator_sha256": _sha(args.expected_validator_sha256, "validator SHA"),
        "coordinator_sha256": _sha(args.expected_coordinator_sha256, "coordinator SHA"),
    }
    scope = load_json(args.scope)
    require(scope.get("scope_id") == pins["scope_id"], "scope ID differs")
    scope_rows = scope.get("timestamp_datasets")
    require(isinstance(scope_rows, list) and len(scope_rows) == 160,
            "scope does not contain exactly 160 timestamp datasets")
    dataset_ids = [row.get("id") for row in scope_rows if isinstance(row, Mapping)]
    require(len(dataset_ids) == 160 and len(set(dataset_ids)) == 160
            and all(isinstance(item, str) and item for item in dataset_ids),
            "scope dataset IDs are not 160 unique non-empty strings")
    raw_manifest = load_json(args.raw_manifest)
    require(raw_manifest.get("scope_id") == pins["scope_id"]
            and raw_manifest.get("scope_sha256") == scope_sha,
            "raw manifest scope binding differs")
    raw_by_id = _raw_bindings(args.raw_manifest, raw_manifest, dataset_ids)
    attempt = load_json(attempt_path)
    require(attempt.get("schema") == ATTEMPT_SCHEMA, "attempt-results schema differs")
    require(attempt.get("run_id") == args.run_id
            and attempt.get("scope_id") == pins["scope_id"]
            and attempt.get("scope_sha256") == scope_sha
            and attempt.get("raw_manifest_sha256") == raw_sha
            and attempt.get("timestamp_policy_file_sha256") == pins["timestamp_policy_sha256"]
            and attempt.get("row_exception_policy_file_sha256") == pins["row_exception_policy_sha256"]
            and attempt.get("exporter_sha256") == pins["exporter_sha256"]
            and attempt.get("validator_sha256") == pins["validator_sha256"]
            and attempt.get("coordinator_sha256") == pins["coordinator_sha256"],
            "attempt-results run/scope/payload binding differs")
    require(attempt.get("dataset_count") == 160
            and attempt.get("attempted_dataset_count") == 160
            and attempt.get("dataset_ids") == dataset_ids,
            "attempt-results frozen coverage differs")
    results = attempt.get("results")
    require(isinstance(results, list) and len(results) == 160,
            "attempt-results does not contain exactly 160 results")
    require([row.get("dataset_id") for row in results if isinstance(row, Mapping)] == dataset_ids,
            "attempt-results has a coverage gap, duplicate, or order mismatch")
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (scope_row, result) in enumerate(zip(scope_rows, results)):
        require(isinstance(scope_row, Mapping) and isinstance(result, Mapping),
                f"attempt row {index} is not an object")
        dataset_id = dataset_ids[index]
        raw = raw_by_id[dataset_id]
        require(result.get("schema")
                == "project179-thulab-csv-best-effort-dataset-attempt-v1"
                and result.get("run_id") == args.run_id,
                f"attempt result schema/run differs for {dataset_id}")
        _require_result_identity(result, scope_row, raw, index, pins)
        attempt_prefix = f"attempts/{dataset_id}"
        evidence = result.get("evidence")
        required_evidence_keys = {
            "attempt", "export_stdout", "export_stderr",
            "validation_stdout", "validation_stderr",
        }
        require(isinstance(evidence, Mapping)
                and set(evidence) == required_evidence_keys,
                f"attempt evidence missing for {dataset_id}")
        evidence_paths: list[str] = []
        for evidence_name in sorted(required_evidence_keys):
            ref = evidence[evidence_name]
            _path, checked_ref = _file_ref(
                root, ref, f"evidence {dataset_id}/{evidence_name}", attempt_prefix
            )
            evidence_paths.append(str(checked_ref["path"]))
        require(len(evidence_paths) == len(set(evidence_paths)),
                f"duplicate evidence path for {dataset_id}")
        required_evidence_paths = {
            f"{attempt_prefix}/ATTEMPT.json",
            f"{attempt_prefix}/export.stdout.log",
            f"{attempt_prefix}/export.stderr.log",
            f"{attempt_prefix}/validation.stdout.log",
            f"{attempt_prefix}/validation.stderr.log",
        }
        require(required_evidence_paths == set(evidence_paths),
                f"attempt receipt or subprocess logs missing for {dataset_id}")
        log_evidence = {key: evidence[key] for key in required_evidence_keys if key != "attempt"}
        require(result.get("log_evidence") == log_evidence
                and result.get("logs")
                == {key: value["path"] for key, value in log_evidence.items()},
                f"attempt log paths/hashes differ from evidence for {dataset_id}")
        status, stage = result.get("status"), result.get("stage")
        attempt_receipt = load_json(root / attempt_prefix / "ATTEMPT.json")
        expected_attempt_receipt = {
            key: value for key, value in result.items() if key != "evidence"
        }
        require(attempt_receipt == expected_attempt_receipt,
                f"attempt receipt does not exactly bind terminal result for {dataset_id}")
        for key, expected in (
            ("scope_sha256", scope_sha),
            ("raw_manifest_sha256", raw_sha),
            ("timestamp_policy_file_sha256", pins["timestamp_policy_sha256"]),
            ("row_exception_policy_file_sha256", pins["row_exception_policy_sha256"]),
            ("exporter_sha256", pins["exporter_sha256"]),
            ("validator_sha256", pins["validator_sha256"]),
            ("coordinator_sha256", pins["coordinator_sha256"]),
        ):
            if key in attempt_receipt:
                require(attempt_receipt[key] == expected,
                        f"attempt receipt {key} differs for {dataset_id}")
        if status == "success":
            require(stage == "complete" and result.get("failure") is None,
                    f"successful result has failure metadata for {dataset_id}")
            _check_process(result.get("export_process"), f"export {dataset_id}", must_succeed=True)
            _check_process(result.get("validation_process"), f"validation {dataset_id}", must_succeed=True)
            manifest, receipt, _dataset_dir = _validate_collection(
                root=root, result=result, dataset_id=dataset_id,
                repo_id=scope_row["repo_id"], revision=scope_row["revision"], pins=pins,
            )
            validation = result.get("validation")
            require(isinstance(validation, Mapping), f"validation reference missing for {dataset_id}")
            expected_validation_path = f"validations/{dataset_id}.json"
            require(validation.get("path") == expected_validation_path,
                    f"validation path differs for {dataset_id}")
            validation_path = _under(root, expected_validation_path, f"validation {dataset_id}")
            require(validation_path.is_file() and not validation_path.is_symlink(),
                    f"validation is not a regular file for {dataset_id}")
            require(digest(validation_path) == _sha(validation.get("sha256"),
                                                    f"validation SHA {dataset_id}"),
                    f"validation SHA differs for {dataset_id}")
            validation_row = _validate_report(load_json(validation_path), dataset_id)
            actual_total = int(receipt["actual_total_bytes_decimal_20"])
            require(validation_row.get("actual_bytes") == actual_total,
                    f"validation byte total differs for {dataset_id}")
            emitted_text = receipt.get("emitted_rows_decimal_20")
            retained_text = receipt.get("retained_value_column_count_decimal_20")
            if emitted_text is not None:
                require(isinstance(emitted_text, str) and emitted_text.isdigit()
                        and validation_row.get("emitted_rows") == int(emitted_text),
                        f"validation row total differs for {dataset_id}")
            if retained_text is not None:
                require(isinstance(retained_text, str) and retained_text.isdigit()
                        and isinstance(validation_row.get("value_columns"), list)
                        and len(validation_row["value_columns"]) == int(retained_text),
                        f"validation value-column total differs for {dataset_id}")
            successes.append({
                "dataset_id": dataset_id,
                "collection_manifest_sha256": digest(_under(root, result["collection"]["manifest_path"], "manifest")),
                "dataset_receipt_sha256": digest(_under(root, result["collection"]["dataset_receipt_path"], "receipt")),
                "validation_sha256": digest(validation_path),
                "actual_total_bytes": actual_total,
            })
        elif status == "failed":
            require(stage in {"admission", "export", "validation"},
                    f"failed stage is invalid for {dataset_id}")
            failure = result.get("failure")
            require(isinstance(failure, Mapping)
                    and failure.get("stage") == stage
                    and isinstance(failure.get("error_type"), str)
                    and failure.get("error_type")
                    and isinstance(failure.get("error_message"), str)
                    and failure.get("error_message"),
                    f"failure reason is incomplete for {dataset_id}")
            failure_evidence = failure.get("evidence")
            require(isinstance(failure_evidence, list) and len(failure_evidence) == 1,
                    f"failure evidence is missing for {dataset_id}")
            failure_path, _ = _file_ref(
                root, failure_evidence[0], f"failure evidence {dataset_id}/0",
                attempt_prefix,
            )
            require(failure_path == root / attempt_prefix / "FAILURE.json",
                    f"failure evidence path differs for {dataset_id}")
            failure_receipt = load_json(failure_path)
            expected_failure = {
                key: value for key, value in failure.items() if key != "evidence"
            }
            require(failure_receipt.get("schema")
                    == "project179-thulab-csv-best-effort-failure-v1"
                    and failure_receipt.get("dataset_id") == dataset_id
                    and failure_receipt.get("failure") == expected_failure,
                    f"failure receipt does not exactly bind reason for {dataset_id}")
            failed_stages = failure.get("failed_export_stages")
            require(isinstance(failed_stages, list),
                    f"failed export-stage evidence list missing for {dataset_id}")
            for stage_index, stage_evidence in enumerate(failed_stages):
                require(isinstance(stage_evidence, Mapping),
                        f"failed export-stage evidence is invalid for {dataset_id}")
                stage_relative = _safe_relative(
                    stage_evidence.get("stage_path"),
                    f"failed stage path {dataset_id}/{stage_index}",
                )
                stage_text = str(stage_relative)
                require(stage_text.startswith(f"collections/.{dataset_id}.failed-"),
                        f"failed stage path differs for {dataset_id}")
                stage_path = root.joinpath(*stage_relative.parts)
                require(stage_path.is_dir() and not stage_path.is_symlink(),
                        f"failed stage is not a real directory for {dataset_id}")
                failure_receipt_ref = stage_evidence.get("failure_receipt")
                if failure_receipt_ref is not None:
                    _file_ref(
                        root, failure_receipt_ref,
                        f"export failure receipt {dataset_id}/{stage_index}",
                        stage_text,
                    )
                diagnostics = stage_evidence.get("worker_diagnostics")
                require(isinstance(diagnostics, list),
                        f"worker diagnostic list missing for {dataset_id}")
                for diagnostic_index, diagnostic_ref in enumerate(diagnostics):
                    _file_ref(
                        root, diagnostic_ref,
                        f"worker diagnostic {dataset_id}/{stage_index}/{diagnostic_index}",
                        stage_text,
                    )
            process_name = "validation_process" if stage == "validation" else "export_process"
            returncode, signal = _check_process(
                result.get(process_name), f"failed {stage} {dataset_id}",
                must_succeed=False,
            )
            error_type = failure.get("error_type")
            if stage == "admission" or error_type == "precondition_failed":
                require(returncode is None and signal is None,
                        f"non-launched failure has subprocess status for {dataset_id}")
            elif error_type == "schema_preflight_failed":
                require(
                    (returncode is None and signal is None)
                    or (_is_int(returncode) and returncode != 0),
                    f"schema-preflight failure has a successful subprocess for {dataset_id}",
                )
            elif error_type == "postcondition_failed":
                require(returncode == 0 and signal is None,
                        f"postcondition failure lacks a successful subprocess for {dataset_id}")
            else:
                require(_is_int(returncode) and returncode != 0,
                        f"subprocess failure lacks nonzero returncode for {dataset_id}")
            require(failure.get("returncode") == returncode
                    and failure.get("signal") == signal,
                    f"failure process identity differs for {dataset_id}")
            validation = result.get("validation")
            if validation is not None:
                require(isinstance(validation, Mapping), f"failed validation reference invalid for {dataset_id}")
                validation_path = _under(root, validation.get("path"), f"failed validation {dataset_id}")
                if validation_path.exists():
                    require(digest(validation_path) == _sha(validation.get("sha256"),
                                                            f"failed validation SHA {dataset_id}"),
                            f"failed validation SHA differs for {dataset_id}")
                    failed_report = load_json(validation_path)
                    require(failed_report.get("status") != "passed",
                            f"failed result contains a passed validation for {dataset_id}")
            failures.append({
                "attempt_index": index,
                "dataset_id": dataset_id,
                "repo_id": scope_row["repo_id"],
                "revision": scope_row["revision"],
                "stage": stage,
                "returncode": returncode,
                "signal": signal,
                "error_type": failure["error_type"],
                "error_message": failure["error_message"],
                "evidence": evidence,
                "failure": failure,
            })
        else:
            raise AttemptGateError(f"attempt gate failed: invalid result status for {dataset_id}")
    require(attempt.get("success_count") == len(successes)
            and attempt.get("failure_count") == len(failures),
            "attempt-results success/failure counts differ")
    expected_attempt_status = "attempts_complete_with_failures" if failures else "attempts_complete"
    require(attempt.get("status") == expected_attempt_status,
            "attempt-results terminal status differs")
    checked_at = datetime.now(timezone.utc).isoformat()
    common = {
        "run_id": args.run_id,
        "scope_id": pins["scope_id"],
        "scope_sha256": scope_sha,
        "raw_manifest_sha256": raw_sha,
        "attempt_results_sha256": attempt_sha,
        "timestamp_policy_file_sha256": pins["timestamp_policy_sha256"],
        "row_exception_policy_file_sha256": pins["row_exception_policy_sha256"],
        "exporter_sha256": pins["exporter_sha256"],
        "validator_sha256": pins["validator_sha256"],
        "coordinator_sha256": pins["coordinator_sha256"],
        "attempted_dataset_count": 160,
        "validated_success_count": len(successes),
        "failed_dataset_count": len(failures),
    }
    failure_report = {
        "schema": FAILURE_SCHEMA,
        "status": "complete",
        "checked_at": checked_at,
        **common,
        "failed_datasets": failures,
    }
    pass_gate = {
        "schema": PASS_GATE_SCHEMA,
        "status": "complete_with_failures" if failures else "complete",
        "gate": "all_frozen_datasets_attempted_once_and_all_successes_validated",
        "checked_at": checked_at,
        **common,
        "successful_dataset_ids": [row["dataset_id"] for row in successes],
        "failed_dataset_ids": [row["dataset_id"] for row in failures],
        "successful_datasets": successes,
    }
    formal = None
    if not failures:
        formal = {
            "schema": FORMAL_GATE_SCHEMA,
            "status": "complete",
            "gate": "formal_160_dataset_csv_materialization_verified",
            "checked_at": checked_at,
            **common,
            "dataset_count": 160,
            "validated_dataset_count": 160,
            "dataset_ids": dataset_ids,
            "successful_datasets": successes,
        }
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "attempt_index", "dataset_id", "repo_id", "revision", "stage",
        "returncode", "signal", "error_type", "error_message", "evidence_json",
        "failure_json",
    ], lineterminator="\n")
    writer.writeheader()
    for row in failures:
        writer.writerow({
            **{key: row[key] for key in writer.fieldnames if key in row},
            "signal": "" if row["signal"] is None else row["signal"],
            "evidence_json": json.dumps(row["evidence"], sort_keys=True, separators=(",", ":")),
            "failure_json": json.dumps(row["failure"], sort_keys=True, separators=(",", ":")),
        })
    return failure_report, pass_gate, csv_buffer.getvalue(), (
        json.dumps(formal, sort_keys=True, separators=(",", ":")) + "\n"
        if formal is not None else None
    )


def _write_new(path: Path, payload: bytes) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), f"output parent is invalid: {path.parent}")
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-results", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-scope-id", required=True)
    for name in (
        "scope-sha256", "raw-manifest-sha256", "timestamp-policy-sha256",
        "row-exception-policy-sha256", "exporter-sha256", "validator-sha256",
        "coordinator-sha256",
    ):
        parser.add_argument("--expected-" + name, required=True)
    parser.add_argument("--failure-json", type=Path, required=True)
    parser.add_argument("--failure-csv", type=Path, required=True)
    parser.add_argument("--pass-gate", type=Path, required=True)
    parser.add_argument("--formal-completion-gate", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = [args.failure_json, args.failure_csv, args.pass_gate]
    require(len({path.resolve(strict=False) for path in outputs + [args.formal_completion_gate]}) == 4,
            "gate output paths are not distinct")
    for path in outputs + [args.formal_completion_gate]:
        require(not path.exists() and not path.is_symlink(), f"gate output already exists: {path}")
    failure, passed, failure_csv, formal = audit(args)
    _write_new(args.failure_json,
               (json.dumps(failure, sort_keys=True, separators=(",", ":")) + "\n").encode())
    _write_new(args.failure_csv, failure_csv.encode("utf-8"))
    _write_new(args.pass_gate,
               (json.dumps(passed, sort_keys=True, separators=(",", ":")) + "\n").encode())
    if formal is not None:
        _write_new(args.formal_completion_gate, formal.encode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AttemptGateError as error:
        print(f"ERROR: {error}", file=os.sys.stderr, flush=True)
        raise SystemExit(1)
