#!/usr/bin/env python3
"""Fail-closed launch/completion gates for the project179 THULab run.

This validator is deliberately independent from the materializer.  It binds
the frozen inputs, validates count-only planning receipts, verifies a sharded
materialization tree (including every declared object hash), and publishes a
new gate receipt atomically without overwriting an existing path.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


VERSION = "project179-thulab-launch-gate-validator-v1"
PLAN_SCHEMA = "project179-thulab-sharded-planning-receipt-v1"
COLLECTION_SCHEMA = "project179-thulab-numeric-collection-v1"
DATASET_SCHEMA = "project179-thulab-numeric-materialization-v1"
DATASET_RECEIPT_SCHEMA = "project179-thulab-numeric-receipt-v1"
STREAM_INDEX_SCHEMA = "project179-thulab-stream-index-v1"
SCOPE_SCHEMA = "web-compression-project-dataset-scope-v1"
LAYOUT = "sharded_v1"
FULL_DATASET_COUNT = 160
PILOT_DATASET_IDS = (
    "finance1k",
    "freshretailnet_50k",
    "kaggle_web_traffic",
    "kaggle_web_traffic_weekly",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GateError(RuntimeError):
    """A fail-closed validation error."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe_regular_file(path: Path, label: str) -> Path:
    absolute = Path(path).absolute()
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as error:
        raise GateError(f"{label} is missing or unreadable: {absolute}") from error
    if not stat.S_ISREG(mode):
        raise GateError(f"{label} must be a regular non-symlink file: {absolute}")
    return absolute.resolve(strict=True)


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], Path, str]:
    regular = _safe_regular_file(path, label)
    digest = sha256_file(regular)
    try:
        value = json.loads(
            regular.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise GateError(f"{label} top level must be an object")
    return value, regular, digest


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GateError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"{label} must be an integer >= {minimum}")
    return value


def _require_safe_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise GateError(f"{label} must be a non-empty path-safe name")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise GateError(f"{label} must be a single path component")
    return value


def _require_safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != value
    ):
        raise GateError(f"{label} is not a canonical safe relative path")
    return value


def _assert_hash(path: Path, expected: str, label: str) -> tuple[Path, str]:
    expected = _require_sha(expected, f"expected {label} SHA-256")
    regular = _safe_regular_file(path, label)
    observed = sha256_file(regular)
    if observed != expected:
        raise GateError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return regular, observed


def _tree_inventory(root: Path, label: str) -> tuple[Path, set[str], set[str]]:
    requested = root.absolute()
    try:
        mode = os.lstat(requested).st_mode
    except OSError as error:
        raise GateError(f"{label} directory is missing: {requested}") from error
    if not stat.S_ISDIR(mode):
        raise GateError(f"{label} must be a real non-symlink directory")
    resolved = requested.resolve(strict=True)
    files: set[str] = set()
    directories: set[str] = set()
    stack: list[tuple[Path, str]] = [(resolved, "")]
    while stack:
        directory, prefix = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise GateError(f"cannot scan {label}: {directory}") from error
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                entry_mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise GateError(f"cannot stat {label} path: {relative}") from error
            if stat.S_ISLNK(entry_mode):
                raise GateError(f"{label} contains a symlink: {relative}")
            if stat.S_ISDIR(entry_mode):
                directories.add(relative)
                stack.append((Path(entry.path), relative))
            elif stat.S_ISREG(entry_mode):
                files.add(relative)
            else:
                raise GateError(f"{label} contains a non-regular object: {relative}")
    return resolved, files, directories


def _scope_rows(scope: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if scope.get("schema") != SCOPE_SCHEMA:
        raise GateError("unsupported frozen scope schema")
    scope_id = scope.get("scope_id")
    if not isinstance(scope_id, str) or not scope_id:
        raise GateError("scope_id must be non-empty")
    rows = scope.get("timestamp_datasets")
    if not isinstance(rows, list) or len(rows) != FULL_DATASET_COUNT:
        raise GateError(f"frozen timestamp scope must contain exactly {FULL_DATASET_COUNT} rows")
    if scope.get("timestamp_dataset_count") != FULL_DATASET_COUNT:
        raise GateError("timestamp_dataset_count does not match the frozen scope")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GateError(f"scope row {ordinal} must be an object")
        dataset_id = _require_safe_name(row.get("id"), f"scope row {ordinal} id")
        if dataset_id in seen:
            raise GateError(f"duplicate dataset id in scope: {dataset_id}")
        seen.add(dataset_id)
        repo_id = row.get("repo_id")
        revision = row.get("revision")
        if not isinstance(repo_id, str) or not repo_id:
            raise GateError(f"scope row {dataset_id} has no repo_id")
        if not isinstance(revision, str) or not revision:
            raise GateError(f"scope row {dataset_id} has no revision")
        result.append({"id": dataset_id, "repo_id": repo_id, "revision": revision})
    missing_pilot = set(PILOT_DATASET_IDS) - seen
    if missing_pilot:
        raise GateError(f"frozen scope is missing pilot datasets: {sorted(missing_pilot)}")
    return scope_id, result


def bind_fixed_inputs(
    *,
    scope_path: Path,
    scope_sha256: str,
    policy_path: Path,
    policy_sha256: str,
    producer_path: Path,
    producer_sha256: str,
    helper_timeseries_path: Path,
    helper_timeseries_sha256: str,
    helper_tsfile_path: Path,
    helper_tsfile_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    bindings: dict[str, Any] = {}
    inputs = (
        ("scope", scope_path, scope_sha256),
        ("policy", policy_path, policy_sha256),
        ("producer", producer_path, producer_sha256),
        ("helper_timeseries", helper_timeseries_path, helper_timeseries_sha256),
        ("helper_tsfile", helper_tsfile_path, helper_tsfile_sha256),
    )
    for name, path, expected in inputs:
        resolved, observed = _assert_hash(path, expected, name)
        bindings[name] = {"path": str(resolved), "sha256": observed}
    scope, _resolved, _digest = _load_json(bindings["scope"]["path"], "scope")
    scope_id, rows = _scope_rows(scope)
    bindings["scope"]["scope_id"] = scope_id
    return bindings, scope, rows


def _expected_ids(selection: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if selection == "pilot":
        return list(PILOT_DATASET_IDS)
    if selection == "full":
        return [str(row["id"]) for row in rows]
    raise GateError(f"unsupported selection: {selection}")


def validate_plan(
    planning_receipt_path: Path,
    *,
    selection: str,
    fixed_inputs: Mapping[str, Any],
    scope_rows: Sequence[Mapping[str, Any]],
    dataset_byte_cap: int,
    chunk_byte_cap: int,
    metadata_compressed_byte_cap: int,
) -> dict[str, Any]:
    receipt, path, receipt_sha256 = _load_json(
        planning_receipt_path, f"{selection} planning receipt"
    )
    expected_ids = _expected_ids(selection, scope_rows)
    scope_by_id = {str(row["id"]): row for row in scope_rows}
    expected_runtime = {
        "producer_sha256": fixed_inputs["producer"]["sha256"],
        "helper_timeseries_sha256": fixed_inputs["helper_timeseries"]["sha256"],
        "helper_tsfile_sha256": fixed_inputs["helper_tsfile"]["sha256"],
    }
    expected_caps = {
        "dataset_logical_byte_cap": dataset_byte_cap,
        "payload_shard_byte_cap": chunk_byte_cap,
        "metadata_compressed_byte_cap": metadata_compressed_byte_cap,
    }
    if receipt.get("schema") != PLAN_SCHEMA:
        raise GateError(f"{selection} planning receipt schema mismatch")
    if receipt.get("status") != "complete":
        raise GateError(f"{selection} planning status is not complete")
    if receipt.get("mode") != "count_only_no_payload_planning":
        raise GateError(f"{selection} planning mode is not count-only")
    if receipt.get("scope_id") != fixed_inputs["scope"]["scope_id"]:
        raise GateError(f"{selection} planning scope_id mismatch")
    if receipt.get("scope_sha256") != fixed_inputs["scope"]["sha256"]:
        raise GateError(f"{selection} planning scope hash mismatch")
    if receipt.get("policy_file_sha256") != fixed_inputs["policy"]["sha256"]:
        raise GateError(f"{selection} planning policy hash mismatch")
    _require_sha(
        receipt.get("mirror_manifest_sha256"),
        f"{selection} planning mirror manifest",
    )
    if receipt.get("runtime_code_bindings") != expected_runtime:
        raise GateError(f"{selection} planning runtime bindings mismatch")
    if receipt.get("caps") != expected_caps:
        raise GateError(f"{selection} planning caps mismatch")
    if receipt.get("storage_layout") != LAYOUT:
        raise GateError(f"{selection} planning layout must be {LAYOUT}")
    if receipt.get("full_scope_dataset_count") != FULL_DATASET_COUNT:
        raise GateError(f"{selection} planning full-scope count mismatch")
    if receipt.get("selected_dataset_count") != len(expected_ids):
        raise GateError(f"{selection} planning selected count mismatch")
    if receipt.get("selected_dataset_ids") != expected_ids:
        raise GateError(f"{selection} planning dataset order/content mismatch")
    plans = receipt.get("datasets")
    if not isinstance(plans, list) or len(plans) != len(expected_ids):
        raise GateError(f"{selection} planning datasets list mismatch")
    if [plan.get("dataset_id") if isinstance(plan, dict) else None for plan in plans] != expected_ids:
        raise GateError(f"{selection} per-dataset plan order/content mismatch")

    candidate_total = selected_total = logical_total = shard_total = 0
    index_uncompressed_total = index_skeleton_total = index_worst_total = 0
    regular_total = directory_total = 0
    peak_temporary_max = 0
    normalized_plans: list[dict[str, Any]] = []
    for plan in plans:
        if not isinstance(plan, dict):
            raise GateError("dataset plan must be an object")
        dataset_id = str(plan["dataset_id"])
        scope_row = scope_by_id[dataset_id]
        if (
            plan.get("repo_id") != scope_row["repo_id"]
            or plan.get("revision") != scope_row["revision"]
        ):
            raise GateError(f"plan scope identity mismatch for {dataset_id}")
        binding = _require_sha(plan.get("binding_sha256"), f"{dataset_id} binding")
        policy = _require_sha(plan.get("policy_sha256"), f"{dataset_id} policy")
        source_receipt = _require_sha(
            plan.get("source_receipt_sha256"), f"{dataset_id} source receipt"
        )
        sources = plan.get("source_files")
        if not isinstance(sources, list) or not sources:
            raise GateError(f"{dataset_id} source_files must be non-empty")
        source_paths: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                raise GateError(f"{dataset_id} source row must be an object")
            relative = _require_safe_relative(source.get("path"), f"{dataset_id} source path")
            if relative in source_paths:
                raise GateError(f"duplicate source path for {dataset_id}: {relative}")
            source_paths.add(relative)
            _require_int(source.get("bytes"), f"{dataset_id} source bytes")
            _require_sha(source.get("sha256"), f"{dataset_id} source hash")
        source_validation = plan.get("source_validation")
        if (
            not isinstance(source_validation, dict)
            or source_validation.get("status") != "validated_count_only"
            or source_validation.get("payload_values_packed") is not False
        ):
            raise GateError(f"count-only source validation missing for {dataset_id}")
        candidate = _require_int(
            plan.get("candidate_stream_count"), f"{dataset_id} candidate streams", minimum=1
        )
        selected = _require_int(
            plan.get("selected_stream_count"), f"{dataset_id} selected streams", minimum=1
        )
        unselected = _require_int(
            plan.get("unselected_stream_count"), f"{dataset_id} unselected streams"
        )
        logical = _require_int(
            plan.get("selected_logical_bytes"), f"{dataset_id} selected logical bytes", minimum=1
        )
        if selected > candidate or unselected != candidate - selected:
            raise GateError(f"invalid stream selection counts for {dataset_id}")
        if logical > dataset_byte_cap:
            raise GateError(f"selected logical bytes exceed dataset cap for {dataset_id}")
        witness = plan.get("longest_prefix_witness")
        if not isinstance(witness, dict):
            raise GateError(f"missing longest-prefix witness for {dataset_id}")
        all_selected = selected == candidate
        first_unselected = witness.get("first_unselected_logical_bytes")
        if witness.get("all_candidates_selected") is not all_selected:
            raise GateError(f"longest-prefix all-selected flag mismatch for {dataset_id}")
        if all_selected:
            if first_unselected is not None or witness.get("proof_boundary") != "all_candidates_selected":
                raise GateError(f"invalid all-selected witness for {dataset_id}")
        else:
            first = _require_int(first_unselected, f"{dataset_id} first unselected bytes", minimum=1)
            if logical + first <= dataset_byte_cap or witness.get("proof_boundary") != "selected_bytes_plus_first_unselected_exceeds_cap":
                raise GateError(f"invalid cap-boundary witness for {dataset_id}")

        payload = plan.get("expected_payload")
        index = plan.get("expected_stream_index")
        objects = plan.get("expected_filesystem_objects")
        layout = plan.get("layout_assertions")
        if not all(isinstance(item, dict) for item in (payload, index, objects, layout)):
            raise GateError(f"planning bounds missing for {dataset_id}")
        payload_bytes = _require_int(payload.get("byte_upper_bound"), f"{dataset_id} payload upper", minimum=logical)
        shard_upper = _require_int(payload.get("shard_count_upper_bound"), f"{dataset_id} shard upper", minimum=1)
        if payload.get("shard_byte_cap") != chunk_byte_cap:
            raise GateError(f"payload shard cap mismatch for {dataset_id}")
        index_records = _require_int(index.get("record_count"), f"{dataset_id} index records", minimum=3)
        index_streams = _require_int(index.get("stream_record_count"), f"{dataset_id} index streams", minimum=1)
        index_entities = _require_int(index.get("entity_record_count"), f"{dataset_id} index entities", minimum=1)
        index_uncompressed = _require_int(index.get("uncompressed_bytes_upper_bound"), f"{dataset_id} index uncompressed upper", minimum=1)
        index_skeleton = _require_int(index.get("compressed_bytes_conservative_skeleton"), f"{dataset_id} index skeleton", minimum=1)
        index_worst = _require_int(index.get("compressed_bytes_worst_case_upper_bound"), f"{dataset_id} index worst upper", minimum=index_uncompressed)
        if (
            index_records != 2 + index_streams + index_entities
            or index_streams != selected
            or index_skeleton > index_worst
            or index_skeleton > metadata_compressed_byte_cap
            or index_worst > metadata_compressed_byte_cap
            or index.get("compressed_byte_cap") != metadata_compressed_byte_cap
        ):
            raise GateError(f"invalid stream-index plan for {dataset_id}")
        expected_regular = _require_int(objects.get("regular_files"), f"{dataset_id} regular objects")
        expected_directories = _require_int(objects.get("directories"), f"{dataset_id} directory objects")
        expected_objects = _require_int(objects.get("total"), f"{dataset_id} total objects")
        if (
            expected_regular != shard_upper + 2
            or expected_directories != 2
            or expected_objects != expected_regular + expected_directories
        ):
            raise GateError(f"invalid filesystem-object bound for {dataset_id}")
        if layout != {
            "per_stream_payload_files": 0,
            "per_stream_payload_directories": 0,
            "payload_files_are_dataset_level_shards": True,
            "stream_metadata_is_in_single_compressed_index": True,
        }:
            raise GateError(f"invalid sharded layout assertions for {dataset_id}")
        peak = _require_int(plan.get("peak_temporary_bytes_upper_bound"), f"{dataset_id} temporary upper", minimum=1)
        candidate_total += candidate
        selected_total += selected
        logical_total += logical
        shard_total += shard_upper
        index_uncompressed_total += index_uncompressed
        index_skeleton_total += index_skeleton
        index_worst_total += index_worst
        regular_total += expected_regular
        directory_total += expected_directories
        peak_temporary_max = max(peak_temporary_max, peak)
        normalized_plans.append({
            "dataset_id": dataset_id,
            # Bind every raw per-dataset planning field, including validation
            # counts, the complete longest-prefix witness, all index estimates,
            # and the temporary-space bound.  The named fields below remain
            # convenient for materialization checks; this digest makes the
            # pilot/full equality check exact rather than lossy.
            "plan_row_sha256": _sha256_bytes(_canonical_json(plan)),
            "repo_id": scope_row["repo_id"],
            "revision": scope_row["revision"],
            "binding_sha256": binding,
            "policy_sha256": policy,
            "source_receipt_sha256": source_receipt,
            "candidate_stream_count": candidate,
            "selected_stream_count": selected,
            "selected_logical_bytes": logical,
            "payload_byte_upper_bound": payload_bytes,
            "payload_shard_count_upper_bound": shard_upper,
            "index_uncompressed_bytes_upper_bound": index_uncompressed,
            "index_compressed_bytes_upper_bound": index_worst,
            "index_record_count": index_records,
            "index_entity_record_count": index_entities,
            "filesystem_regular_files_upper_bound": expected_regular,
            "filesystem_directories_upper_bound": expected_directories,
            "source_files": sources,
        })

    totals = receipt.get("totals")
    if not isinstance(totals, dict):
        raise GateError(f"{selection} planning totals are missing")
    expected_collection_regular = regular_total + len(plans) + 1
    expected_collection_directories = directory_total + 1
    expected_collection_objects = {
        "regular_files": expected_collection_regular,
        "directories": expected_collection_directories,
        "total": expected_collection_regular + expected_collection_directories,
    }
    observed_collection = totals.get("expected_collection_filesystem_objects")
    if not isinstance(observed_collection, dict) or any(
        observed_collection.get(key) != value
        for key, value in expected_collection_objects.items()
    ):
        raise GateError(f"{selection} collection object bounds mismatch")
    expected_totals = {
        "candidate_stream_count": candidate_total,
        "selected_stream_count": selected_total,
        "selected_logical_bytes": logical_total,
        "expected_payload_shard_count_upper_bound": shard_total,
        "expected_index_uncompressed_bytes_upper_bound": index_uncompressed_total,
        "expected_index_compressed_bytes_conservative_skeleton": index_skeleton_total,
        "expected_index_compressed_bytes_worst_case_upper_bound": index_worst_total,
        "peak_temporary_bytes_upper_bound": peak_temporary_max,
        "per_stream_payload_files": 0,
        "per_stream_payload_directories": 0,
    }
    for key, expected in expected_totals.items():
        if totals.get(key) != expected:
            raise GateError(f"{selection} planning total mismatch: {key}")
    if receipt.get("determinism", {}).get("payload_materialized") is not False:
        raise GateError(f"{selection} planning receipt claims payload materialization")
    return {
        "selection": selection,
        "path": str(path),
        "sha256": receipt_sha256,
        "mirror_manifest_sha256": receipt["mirror_manifest_sha256"],
        "dataset_ids": expected_ids,
        "datasets": normalized_plans,
        "totals": expected_totals | {
            "expected_collection_filesystem_objects": expected_collection_objects
        },
    }


def _read_gzip_binding(path: Path) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as handle:
            while True:
                block = handle.read(8 << 20)
                if not block:
                    return count, digest.hexdigest()
                count += len(block)
                digest.update(block)
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        raise GateError(f"invalid gzip stream index: {path}") from error


def _file_binding(root: Path, relative: str, label: str) -> tuple[Path, int, str]:
    safe = _require_safe_relative(relative, label)
    path = root / safe
    regular = _safe_regular_file(path, label)
    if root.resolve(strict=True) not in regular.parents:
        raise GateError(f"{label} escapes its materialization root")
    return regular, regular.stat().st_size, sha256_file(regular)


def validate_materialization(
    output: Path,
    *,
    plan: Mapping[str, Any],
    fixed_inputs: Mapping[str, Any],
    scope_rows: Sequence[Mapping[str, Any]],
    dataset_byte_cap: int,
    chunk_byte_cap: int,
    metadata_compressed_byte_cap: int,
) -> dict[str, Any]:
    selection = str(plan["selection"])
    expected_ids = _expected_ids(selection, scope_rows)
    plans = plan["datasets"]
    root, actual_files, actual_directories = _tree_inventory(
        output, f"{selection} materialization"
    )
    top, top_path, top_sha256 = _load_json(
        root / "MATERIALIZATION_MANIFEST.json", "materialization manifest"
    )
    expected_runtime = {
        "producer_sha256": fixed_inputs["producer"]["sha256"],
        "helper_timeseries_sha256": fixed_inputs["helper_timeseries"]["sha256"],
        "helper_tsfile_sha256": fixed_inputs["helper_tsfile"]["sha256"],
    }
    if (
        top.get("schema") != COLLECTION_SCHEMA
        or top.get("status") != "complete"
        or top.get("mode") != "prepare_or_resume"
        or top.get("storage_layout") != LAYOUT
        or top.get("scope_id") != fixed_inputs["scope"]["scope_id"]
        or top.get("scope_sha256") != fixed_inputs["scope"]["sha256"]
        or top.get("policy_file_sha256") != fixed_inputs["policy"]["sha256"]
        or top.get("runtime_code_bindings") != expected_runtime
    ):
        raise GateError(f"{selection} top-level materialization binding/status mismatch")
    if (
        top.get("selected_dataset_count") != len(expected_ids)
        or top.get("full_scope_dataset_count") != FULL_DATASET_COUNT
        or top.get("status_counts") != {"materialized": len(expected_ids), "blocked": 0}
        or top.get("dataset_byte_cap") != dataset_byte_cap
        or top.get("chunk_byte_cap") != chunk_byte_cap
        or top.get("metadata_compressed_byte_cap") != metadata_compressed_byte_cap
    ):
        raise GateError(f"{selection} collection counts/caps are not launch-safe")
    top_rows = top.get("datasets")
    if not isinstance(top_rows, list) or [
        row.get("dataset_id") if isinstance(row, dict) else None for row in top_rows
    ] != expected_ids:
        raise GateError(f"{selection} collection dataset order/content mismatch")

    expected_files = {"MATERIALIZATION_MANIFEST.json"}
    expected_directories = {"_receipts"}
    inventory_rows: list[dict[str, Any]] = [{
        "path": "MATERIALIZATION_MANIFEST.json",
        "bytes": top_path.stat().st_size,
        "sha256": top_sha256,
    }]
    total_payload_bytes = total_index_compressed = total_index_uncompressed = 0
    total_shards = 0
    plan_by_id = {row["dataset_id"]: row for row in plans}
    for top_row, dataset_id in zip(top_rows, expected_ids):
        if top_row.get("status") != "materialized":
            raise GateError(f"materialization top row is not materialized: {dataset_id}")
        receipt_relative = f"_receipts/{dataset_id}.json"
        if top_row.get("receipt") != receipt_relative:
            raise GateError(f"receipt path mismatch for {dataset_id}")
        receipt, receipt_path, receipt_sha256 = _load_json(
            root / receipt_relative, f"{dataset_id} materialization receipt"
        )
        if top_row.get("receipt_sha256") != receipt_sha256:
            raise GateError(f"top-level receipt hash mismatch for {dataset_id}")
        expected_files.add(receipt_relative)
        inventory_rows.append({
            "path": receipt_relative,
            "bytes": receipt_path.stat().st_size,
            "sha256": receipt_sha256,
        })
        dataset_plan = plan_by_id[dataset_id]
        if (
            receipt.get("schema") != DATASET_RECEIPT_SCHEMA
            or receipt.get("status") != "materialized"
            or receipt.get("dataset_id") != dataset_id
            or receipt.get("repo_id") != dataset_plan["repo_id"]
            or receipt.get("revision") != dataset_plan["revision"]
            or receipt.get("output_directory") != dataset_id
            or receipt.get("storage_layout") != LAYOUT
            or receipt.get("scope_id") != fixed_inputs["scope"]["scope_id"]
            or receipt.get("scope_sha256") != fixed_inputs["scope"]["sha256"]
            or receipt.get("policy_file_sha256") != fixed_inputs["policy"]["sha256"]
            or receipt.get("runtime_code_bindings") != expected_runtime
        ):
            raise GateError(f"materialization receipt binding/status mismatch for {dataset_id}")
        for key in (
            "binding_sha256",
            "policy_sha256",
            "source_receipt_sha256",
            "candidate_stream_count",
            "selected_stream_count",
            "selected_logical_bytes",
        ):
            if receipt.get(key) != dataset_plan[key]:
                raise GateError(f"plan/receipt mismatch for {dataset_id}: {key}")
        receipt_sources = [
            {"path": row.get("path"), "bytes": row.get("bytes"), "sha256": row.get("sha256")}
            for row in receipt.get("files", [])
            if isinstance(row, dict)
        ]
        if receipt_sources != dataset_plan["source_files"]:
            raise GateError(f"source binding mismatch in receipt for {dataset_id}")

        dataset_relative = dataset_id
        expected_directories.update({dataset_relative, f"{dataset_relative}/payload"})
        manifest_relative = f"{dataset_relative}/dataset_manifest.json"
        manifest, manifest_path, manifest_sha256 = _load_json(
            root / manifest_relative, f"{dataset_id} dataset manifest"
        )
        if receipt.get("manifest_sha256") != manifest_sha256:
            raise GateError(f"dataset manifest hash mismatch for {dataset_id}")
        expected_files.add(manifest_relative)
        inventory_rows.append({
            "path": manifest_relative,
            "bytes": manifest_path.stat().st_size,
            "sha256": manifest_sha256,
        })
        provenance = manifest.get("provenance_sha256")
        without_provenance = dict(manifest)
        without_provenance.pop("provenance_sha256", None)
        if provenance != _sha256_bytes(_canonical_json(without_provenance)):
            raise GateError(f"dataset provenance hash mismatch for {dataset_id}")
        timestamp_policy = manifest.get("timestamp_policy")
        selection_receipt = manifest.get("selection")
        totals = manifest.get("totals")
        if not all(isinstance(item, dict) for item in (timestamp_policy, selection_receipt, totals)):
            raise GateError(f"dataset manifest sections missing for {dataset_id}")
        if (
            manifest.get("schema") != DATASET_SCHEMA
            or manifest.get("status") != "materialized"
            or manifest.get("storage_layout") != LAYOUT
            or manifest.get("dataset_id") != dataset_id
            or manifest.get("repo_id") != dataset_plan["repo_id"]
            or manifest.get("revision") != dataset_plan["revision"]
            or manifest.get("binding_sha256") != dataset_plan["binding_sha256"]
            or manifest.get("scope_id") != fixed_inputs["scope"]["scope_id"]
            or manifest.get("scope_sha256") != fixed_inputs["scope"]["sha256"]
            or manifest.get("source_receipt_sha256") != dataset_plan["source_receipt_sha256"]
            or manifest.get("runtime_code_bindings") != expected_runtime
            or manifest.get("source_tsfiles") != dataset_plan["source_files"]
            or timestamp_policy.get("policy_file_sha256") != fixed_inputs["policy"]["sha256"]
            or timestamp_policy.get("policy_sha256") != dataset_plan["policy_sha256"]
        ):
            raise GateError(f"dataset manifest binding mismatch for {dataset_id}")
        expected_selection = {
            "candidate_stream_count": dataset_plan["candidate_stream_count"],
            "selected_stream_count": dataset_plan["selected_stream_count"],
            "selected_logical_bytes": dataset_plan["selected_logical_bytes"],
            "dataset_byte_cap": dataset_byte_cap,
            "chunk_byte_cap": chunk_byte_cap,
            "metadata_compressed_byte_cap": metadata_compressed_byte_cap,
        }
        for key, expected in expected_selection.items():
            if selection_receipt.get(key) != expected:
                raise GateError(f"plan/manifest selection mismatch for {dataset_id}: {key}")
        if (
            totals.get("stream_count") != dataset_plan["selected_stream_count"]
            or totals.get("logical_bytes") != dataset_plan["selected_logical_bytes"]
        ):
            raise GateError(f"plan/manifest totals mismatch for {dataset_id}")

        dataset_root = root / dataset_id
        index = manifest.get("stream_index")
        shards = manifest.get("payload_shards")
        if not isinstance(index, dict) or not isinstance(shards, list) or not shards:
            raise GateError(f"missing sharded objects for {dataset_id}")
        if "streams" in manifest:
            raise GateError(f"sharded manifest embeds streams for {dataset_id}")
        index_relative_inner = _require_safe_relative(index.get("path"), f"{dataset_id} index path")
        if Path(index_relative_inner).parent != Path("."):
            raise GateError(f"stream index must be directly in dataset directory: {dataset_id}")
        index_path, index_bytes, index_sha256 = _file_binding(
            dataset_root, index_relative_inner, f"{dataset_id} stream index"
        )
        if (
            index.get("schema") != STREAM_INDEX_SCHEMA
            or index.get("compression") != "gzip"
            or index.get("compressed_bytes") != index_bytes
            or index.get("compressed_sha256") != index_sha256
            or index.get("compressed_byte_cap") != metadata_compressed_byte_cap
            or index_bytes > metadata_compressed_byte_cap
            or index_bytes > dataset_plan["index_compressed_bytes_upper_bound"]
            or index.get("record_count") != dataset_plan["index_record_count"]
            or index.get("entity_record_count")
            != dataset_plan["index_entity_record_count"]
        ):
            raise GateError(f"stream-index compressed binding/bound mismatch for {dataset_id}")
        uncompressed_bytes, uncompressed_sha256 = _read_gzip_binding(index_path)
        if (
            index.get("uncompressed_bytes") != uncompressed_bytes
            or index.get("uncompressed_sha256") != uncompressed_sha256
            or uncompressed_bytes > dataset_plan["index_uncompressed_bytes_upper_bound"]
            or index.get("stream_record_count") != dataset_plan["selected_stream_count"]
        ):
            raise GateError(f"stream-index uncompressed binding/bound mismatch for {dataset_id}")
        index_relative = f"{dataset_id}/{index_relative_inner}"
        expected_files.add(index_relative)
        inventory_rows.append({"path": index_relative, "bytes": index_bytes, "sha256": index_sha256})
        total_index_compressed += index_bytes
        total_index_uncompressed += uncompressed_bytes

        if len(shards) > dataset_plan["payload_shard_count_upper_bound"]:
            raise GateError(f"payload shard count exceeds plan for {dataset_id}")
        shard_paths: set[str] = set()
        dataset_payload_bytes = 0
        for shard_ordinal, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise GateError(f"payload shard row is not an object for {dataset_id}")
            shard_relative_inner = _require_safe_relative(shard.get("path"), f"{dataset_id} shard path")
            if Path(shard_relative_inner).parent != Path("payload"):
                raise GateError(f"payload shard is outside dataset payload directory: {dataset_id}")
            if shard_relative_inner in shard_paths:
                raise GateError(f"duplicate payload shard for {dataset_id}: {shard_relative_inner}")
            shard_paths.add(shard_relative_inner)
            _path, shard_bytes, shard_sha256 = _file_binding(
                dataset_root, shard_relative_inner, f"{dataset_id} payload shard"
            )
            if (
                shard.get("shard_index") != shard_ordinal
                or
                shard.get("bytes") != shard_bytes
                or shard.get("sha256") != shard_sha256
                or shard_bytes > chunk_byte_cap
            ):
                raise GateError(f"payload shard binding/cap mismatch for {dataset_id}")
            relative = f"{dataset_id}/{shard_relative_inner}"
            expected_files.add(relative)
            inventory_rows.append({"path": relative, "bytes": shard_bytes, "sha256": shard_sha256})
            dataset_payload_bytes += shard_bytes
        if dataset_payload_bytes > dataset_plan["payload_byte_upper_bound"]:
            raise GateError(f"payload bytes exceed plan for {dataset_id}")
        dataset_actual_regular = 2 + len(shards)
        if dataset_actual_regular > dataset_plan["filesystem_regular_files_upper_bound"]:
            raise GateError(f"filesystem objects exceed plan for {dataset_id}")
        total_payload_bytes += dataset_payload_bytes
        total_shards += len(shards)

    if actual_files != expected_files or actual_directories != expected_directories:
        extra_files = sorted(actual_files - expected_files)
        missing_files = sorted(expected_files - actual_files)
        extra_dirs = sorted(actual_directories - expected_directories)
        missing_dirs = sorted(expected_directories - actual_directories)
        raise GateError(
            f"{selection} materialization is not an exact tree: "
            f"extra_files={extra_files[:5]}, missing_files={missing_files[:5]}, "
            f"extra_dirs={extra_dirs[:5]}, missing_dirs={missing_dirs[:5]}"
        )
    object_upper = plan["totals"]["expected_collection_filesystem_objects"]
    if (
        len(actual_files) > object_upper["regular_files"]
        or len(actual_directories) > object_upper["directories"]
        or len(actual_files) + len(actual_directories) > object_upper["total"]
    ):
        raise GateError(f"{selection} collection filesystem objects exceed plan")
    inventory_rows.sort(key=lambda row: row["path"])
    return {
        "selection": selection,
        "root": str(root),
        "top_manifest_sha256": top_sha256,
        "dataset_ids": expected_ids,
        "dataset_count": len(expected_ids),
        "blocked_dataset_count": 0,
        "payload_shard_count": total_shards,
        "payload_bytes": total_payload_bytes,
        "stream_index_compressed_bytes": total_index_compressed,
        "stream_index_uncompressed_bytes": total_index_uncompressed,
        "filesystem_objects": {
            "regular_files": len(actual_files),
            "directories": len(actual_directories),
            "total": len(actual_files) + len(actual_directories),
        },
        "tree_sha256": _sha256_bytes(_canonical_json(inventory_rows)),
        "validation": {
            "exact_tree": True,
            "no_symlinks": True,
            "all_declared_file_hashes_verified": True,
            "planning_bounds_verified": True,
            "plan_binding_selection_logical_bytes_verified": True,
        },
    }


def _atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> Path:
    requested = path.absolute()
    parent = requested.parent
    try:
        parent_mode = os.lstat(parent).st_mode
    except OSError as error:
        raise GateError("gate receipt parent must already exist") from error
    if not stat.S_ISDIR(parent_mode):
        raise GateError("gate receipt parent must be a real non-symlink directory")
    resolved_parent = parent.resolve(strict=True)
    target = resolved_parent / requested.name
    if target.exists() or target.is_symlink():
        raise GateError(f"gate receipt already exists: {target}")
    payload = _canonical_json(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=resolved_parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise GateError(f"gate receipt already exists: {target}") from error
        directory_fd = os.open(resolved_parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _ensure_receipt_outside_outputs(path: Path, outputs: Sequence[Path]) -> None:
    requested_parent = path.absolute().parent.resolve(strict=True)
    candidate = requested_parent / path.name
    for output in outputs:
        root = output.absolute().resolve(strict=True)
        if candidate == root or root in candidate.parents:
            raise GateError("gate receipt must be outside every materialization tree")


def build_launch_gate(
    *,
    pilot_planning_receipt: Path,
    full_planning_receipt: Path,
    pilot_output: Path,
    output_receipt: Path,
    fixed_inputs: Mapping[str, Any],
    scope_rows: Sequence[Mapping[str, Any]],
    dataset_byte_cap: int,
    chunk_byte_cap: int,
    metadata_compressed_byte_cap: int,
) -> tuple[dict[str, Any], Path]:
    validator_path = _safe_regular_file(Path(__file__), "launch-gate validator")
    validator_sha256_start = sha256_file(validator_path)
    pilot_plan = validate_plan(
        pilot_planning_receipt,
        selection="pilot",
        fixed_inputs=fixed_inputs,
        scope_rows=scope_rows,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=chunk_byte_cap,
        metadata_compressed_byte_cap=metadata_compressed_byte_cap,
    )
    full_plan = validate_plan(
        full_planning_receipt,
        selection="full",
        fixed_inputs=fixed_inputs,
        scope_rows=scope_rows,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=chunk_byte_cap,
        metadata_compressed_byte_cap=metadata_compressed_byte_cap,
    )
    if (
        pilot_plan["mirror_manifest_sha256"]
        != full_plan["mirror_manifest_sha256"]
    ):
        raise GateError("pilot/full planning mirror-manifest bindings differ")
    full_plans_by_id = {
        row["dataset_id"]: row for row in full_plan["datasets"]
    }
    for pilot_row in pilot_plan["datasets"]:
        dataset_id = pilot_row["dataset_id"]
        if full_plans_by_id.get(dataset_id) != pilot_row:
            raise GateError(
                f"pilot/full planning dataset bindings differ: {dataset_id}"
            )
    pilot = validate_materialization(
        pilot_output,
        plan=pilot_plan,
        fixed_inputs=fixed_inputs,
        scope_rows=scope_rows,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=chunk_byte_cap,
        metadata_compressed_byte_cap=metadata_compressed_byte_cap,
    )
    _ensure_receipt_outside_outputs(output_receipt, [pilot_output])
    inputs = {
        "fixed_inputs": fixed_inputs,
        "pilot_planning_receipt": {
            "path": pilot_plan["path"], "sha256": pilot_plan["sha256"]
        },
        "full_planning_receipt": {
            "path": full_plan["path"], "sha256": full_plan["sha256"]
        },
        "pilot_materialization": {
            "root": pilot["root"],
            "top_manifest_sha256": pilot["top_manifest_sha256"],
            "tree_sha256": pilot["tree_sha256"],
        },
    }
    if sha256_file(validator_path) != validator_sha256_start:
        raise GateError("launch-gate validator changed during validation")
    receipt = {
        "schema": "project179-thulab-formal-launch-gate-v1",
        "status": "complete",
        "gate": "formal_160_dataset_run_authorized",
        "validator": {
            "version": VERSION,
            "path": str(validator_path),
            "sha256": validator_sha256_start,
        },
        "input_bindings": inputs,
        "input_bindings_sha256": _sha256_bytes(_canonical_json(inputs)),
        "caps": {
            "dataset_logical_byte_cap": dataset_byte_cap,
            "payload_shard_byte_cap": chunk_byte_cap,
            "metadata_compressed_byte_cap": metadata_compressed_byte_cap,
        },
        "storage_layout": LAYOUT,
        "pilot": pilot,
        "full_plan": {
            "dataset_count": len(full_plan["dataset_ids"]),
            "dataset_ids_sha256": _sha256_bytes(_canonical_json(full_plan["dataset_ids"])),
            "totals": full_plan["totals"],
        },
        "requirements": {
            "pilot_exact_four_in_frozen_order": True,
            "pilot_tree_hashes_and_planning_bounds_independently_verified": True,
            "pilot_exact_tree_and_no_symlinks": True,
            "full_plan_exact_160_in_frozen_order": True,
            "per_stream_payload_files": 0,
            "planning_and_actual_bounds_verified": True,
        },
    }
    target = _atomic_json_no_clobber(output_receipt, receipt)
    return receipt, target


def build_completion_gate(
    *,
    full_planning_receipt: Path,
    formal_output: Path,
    output_receipt: Path,
    fixed_inputs: Mapping[str, Any],
    scope_rows: Sequence[Mapping[str, Any]],
    dataset_byte_cap: int,
    chunk_byte_cap: int,
    metadata_compressed_byte_cap: int,
) -> tuple[dict[str, Any], Path]:
    validator_path = _safe_regular_file(Path(__file__), "completion-gate validator")
    validator_sha256_start = sha256_file(validator_path)
    full_plan = validate_plan(
        full_planning_receipt,
        selection="full",
        fixed_inputs=fixed_inputs,
        scope_rows=scope_rows,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=chunk_byte_cap,
        metadata_compressed_byte_cap=metadata_compressed_byte_cap,
    )
    formal = validate_materialization(
        formal_output,
        plan=full_plan,
        fixed_inputs=fixed_inputs,
        scope_rows=scope_rows,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=chunk_byte_cap,
        metadata_compressed_byte_cap=metadata_compressed_byte_cap,
    )
    _ensure_receipt_outside_outputs(output_receipt, [formal_output])
    inputs = {
        "fixed_inputs": fixed_inputs,
        "full_planning_receipt": {
            "path": full_plan["path"], "sha256": full_plan["sha256"]
        },
        "formal_materialization": {
            "root": formal["root"],
            "top_manifest_sha256": formal["top_manifest_sha256"],
            "tree_sha256": formal["tree_sha256"],
        },
    }
    if sha256_file(validator_path) != validator_sha256_start:
        raise GateError("completion-gate validator changed during validation")
    receipt = {
        "schema": "project179-thulab-formal-completion-gate-v1",
        "status": "complete",
        "gate": "formal_160_dataset_materialization_verified",
        "validator": {
            "version": VERSION,
            "path": str(validator_path),
            "sha256": validator_sha256_start,
        },
        "input_bindings": inputs,
        "input_bindings_sha256": _sha256_bytes(_canonical_json(inputs)),
        "caps": {
            "dataset_logical_byte_cap": dataset_byte_cap,
            "payload_shard_byte_cap": chunk_byte_cap,
            "metadata_compressed_byte_cap": metadata_compressed_byte_cap,
        },
        "storage_layout": LAYOUT,
        "formal": formal,
        "requirements": {
            "formal_exact_160_in_frozen_order": True,
            "formal_status_complete_and_blocked_zero": True,
            "formal_exact_tree_and_no_symlinks": True,
            "all_declared_file_hashes_verified": True,
            "planning_and_actual_bounds_verified": True,
        },
    }
    target = _atomic_json_no_clobber(output_receipt, receipt)
    return receipt, target


def _add_fixed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--helper-timeseries", type=Path, required=True)
    parser.add_argument("--helper-timeseries-sha256", required=True)
    parser.add_argument("--helper-tsfile", type=Path, required=True)
    parser.add_argument("--helper-tsfile-sha256", required=True)
    parser.add_argument("--dataset-byte-cap", type=int, default=1 << 30)
    parser.add_argument("--chunk-byte-cap", type=int, default=64 << 20)
    parser.add_argument("--metadata-compressed-byte-cap", type=int, default=1 << 30)


def _fixed_from_args(args: argparse.Namespace):
    for name in (
        "dataset_byte_cap", "chunk_byte_cap", "metadata_compressed_byte_cap"
    ):
        if getattr(args, name) <= 0:
            raise GateError(f"{name} must be positive")
    if args.dataset_byte_cap > 1 << 30:
        raise GateError("dataset byte cap must not exceed 1 GiB")
    if args.chunk_byte_cap > min(args.dataset_byte_cap, 64 << 20):
        raise GateError("chunk byte cap must not exceed dataset cap or 64 MiB")
    if args.metadata_compressed_byte_cap > 1 << 30:
        raise GateError("metadata compressed byte cap must not exceed 1 GiB")
    return bind_fixed_inputs(
        scope_path=args.scope,
        scope_sha256=args.scope_sha256,
        policy_path=args.policy,
        policy_sha256=args.policy_sha256,
        producer_path=args.producer,
        producer_sha256=args.producer_sha256,
        helper_timeseries_path=args.helper_timeseries,
        helper_timeseries_sha256=args.helper_timeseries_sha256,
        helper_tsfile_path=args.helper_tsfile,
        helper_tsfile_sha256=args.helper_tsfile_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("validate-plan")
    _add_fixed_arguments(plan_parser)
    plan_parser.add_argument("--planning-receipt", type=Path, required=True)
    plan_parser.add_argument("--selection", choices=("pilot", "full"), required=True)
    launch_parser = subparsers.add_parser("build-launch-gate")
    _add_fixed_arguments(launch_parser)
    launch_parser.add_argument("--pilot-planning-receipt", type=Path, required=True)
    launch_parser.add_argument("--full-planning-receipt", type=Path, required=True)
    launch_parser.add_argument("--pilot-output", type=Path, required=True)
    launch_parser.add_argument("--output-receipt", type=Path, required=True)
    completion_parser = subparsers.add_parser("build-completion-gate")
    _add_fixed_arguments(completion_parser)
    completion_parser.add_argument("--full-planning-receipt", type=Path, required=True)
    completion_parser.add_argument("--formal-output", type=Path, required=True)
    completion_parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        fixed, _scope, rows = _fixed_from_args(args)
        common = {
            "fixed_inputs": fixed,
            "scope_rows": rows,
            "dataset_byte_cap": args.dataset_byte_cap,
            "chunk_byte_cap": args.chunk_byte_cap,
            "metadata_compressed_byte_cap": args.metadata_compressed_byte_cap,
        }
        if args.command == "validate-plan":
            result = validate_plan(
                args.planning_receipt,
                selection=args.selection,
                **common,
            )
            output = {
                "status": "complete",
                "validator": VERSION,
                "selection": result["selection"],
                "planning_receipt_sha256": result["sha256"],
                "dataset_count": len(result["dataset_ids"]),
            }
        elif args.command == "build-launch-gate":
            _receipt, target = build_launch_gate(
                pilot_planning_receipt=args.pilot_planning_receipt,
                full_planning_receipt=args.full_planning_receipt,
                pilot_output=args.pilot_output,
                output_receipt=args.output_receipt,
                **common,
            )
            output = {
                "status": "complete",
                "gate": "formal_160_dataset_run_authorized",
                "receipt": str(target),
                "receipt_sha256": sha256_file(target),
            }
        else:
            _receipt, target = build_completion_gate(
                full_planning_receipt=args.full_planning_receipt,
                formal_output=args.formal_output,
                output_receipt=args.output_receipt,
                **common,
            )
            output = {
                "status": "complete",
                "gate": "formal_160_dataset_materialization_verified",
                "receipt": str(target),
                "receipt_sha256": sha256_file(target),
            }
        print(_canonical_json(output).decode("ascii"))
        return 0
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
