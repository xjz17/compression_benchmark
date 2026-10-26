#!/usr/bin/env python3
"""Audit and report the frozen harness-combination pilot.

This reporter deliberately derives every number from committed receipts.  It
does not load source column manifests and it never uses historical oracle
labels to choose a pilot candidate or policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePath
from typing import Any, Iterable


STAGES = ("discovery", "tuning", "locked_validation")
DTYPES = ("float32", "float64", "int64")
BASELINES = ("fixed_zstd3", "fixed_lz4", "prior_top1")
KINDS = ("algorithm", "operator_tree", "algorithm_combination")
MIN_BYTES = 16 * 1024
MAX_BYTES = 8 * 1024 * 1024


class ReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def integer(value: Any, context: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{context}: expected integer")
    require(value >= minimum, f"{context}: expected >= {minimum}")
    return value


def finite_number(value: Any, context: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{context}: expected number")
    result = float(value)
    require(math.isfinite(result), f"{context}: expected finite number")
    return result


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing receipt: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read JSON receipt {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: top level must be an object")
    return value


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file for SHA-256: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: Any, context: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"{context}: invalid lowercase SHA-256")
    return value


def column_key(column: dict[str, Any]) -> str:
    dataset = column.get("dataset_id")
    column_id = column.get("column_id")
    require(isinstance(dataset, str) and dataset, "empty dataset_id")
    require(isinstance(column_id, str) and len(column_id) == 64,
            f"{dataset}: invalid column_id")
    return f"{dataset}__{column_id[:16]}"


def receipt_suffix_matches(declared: Any, expected: Path, context: str) -> None:
    require(isinstance(declared, str) and declared, f"{context}: empty path")
    declared_parts = PurePath(declared).parts
    expected_parts = expected.parts
    suffix_length = min(5, len(expected_parts))
    require(tuple(declared_parts[-suffix_length:]) ==
            tuple(expected_parts[-suffix_length:]),
            f"{context}: declared path does not identify expected receipt")


def compare_json(left: Any, right: Any, context: str) -> None:
    require(left == right, f"{context}: embedded and committed JSON differ")


def category(entry: dict[str, Any]) -> str:
    if entry["new_combination"]:
        return "new_combination"
    return entry["kind"]


def validate_plan(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = read_json(root / "PLAN.json")
    require(plan.get("schema") == "harness-combo-pilot-plan-v1",
            "unexpected PLAN schema")
    selections = plan.get("selections")
    require(isinstance(selections, list), "PLAN.selections must be an array")
    require(integer(plan.get("selected_count"), "PLAN.selected_count") ==
            len(selections), "PLAN selection count mismatch")
    require(len(selections) == 41,
            f"pilot contract requires 41 columns, found {len(selections)}")

    keys: set[str] = set()
    identities: set[tuple[str, str]] = set()
    groups: dict[str, set[str]] = defaultdict(set)
    stage_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    expected_stage = {0: "discovery", 1: "discovery", 2: "discovery",
                      3: "tuning", 4: "locked_validation"}
    for index, column in enumerate(selections):
        require(isinstance(column, dict), f"PLAN selection {index} is not object")
        require(column.get("selection_index") == index,
                f"PLAN selection_index drift at {index}")
        require(column.get("split") == "development",
                f"{column_key(column)}: confirm row is forbidden")
        fold = integer(column.get("dev_fold"), f"selection {index}.dev_fold")
        require(fold in expected_stage, f"selection {index}: invalid dev_fold")
        stage = column.get("stage")
        require(stage == expected_stage[fold],
                f"selection {index}: stage/fold mismatch")
        dtype = column.get("dtype")
        require(dtype in DTYPES, f"selection {index}: unsupported dtype {dtype}")
        original = integer(column.get("original_numeric_bytes"),
                           f"selection {index}.original_numeric_bytes")
        valid = integer(column.get("valid_value_count"),
                        f"selection {index}.valid_value_count")
        require(MIN_BYTES <= original <= MAX_BYTES,
                f"selection {index}: column outside 16KiB..8MiB")
        require(valid >= 1024, f"selection {index}: fewer than 1024 valid values")
        width = 4 if dtype == "float32" else 8
        require(original == valid * width,
                f"selection {index}: typed payload size mismatch")
        require(column.get("size_bucket") in {"short", "medium", "long"},
                f"selection {index}: invalid size bucket")
        group = column.get("source_group")
        require(isinstance(group, str) and group, f"selection {index}: source group")
        key = column_key(column)
        identity = (column["dataset_id"], column["column_id"])
        require(key not in keys and identity not in identities,
                f"duplicate PLAN selection: {key}")
        keys.add(key)
        identities.add(identity)
        groups[stage].add(group)
        stage_counts[stage] += 1
        dtype_counts[dtype] += 1

    require(not (groups["discovery"] & groups["tuning"]),
            "source-group leakage: discovery/tuning")
    require(not (groups["discovery"] & groups["locked_validation"]),
            "source-group leakage: discovery/locked_validation")
    require(not (groups["tuning"] & groups["locked_validation"]),
            "source-group leakage: tuning/locked_validation")
    require(stage_counts == Counter({"discovery": 23, "tuning": 9,
                                     "locked_validation": 9}),
            f"unexpected stage counts: {dict(stage_counts)}")
    require(dtype_counts == Counter({"float32": 15, "float64": 14,
                                     "int64": 12}),
            f"unexpected dtype counts: {dict(dtype_counts)}")
    return plan, selections


def validate_run_envelope(run_root: Path, plan_sha: str) -> tuple[Path, dict[str, Any]]:
    """Bind the immutable run wrapper to the result tree it authorizes."""
    build_path = run_root / "BUILD.json"
    complete_path = run_root / "RUN_COMPLETE.json"
    binary_path = run_root / "build" / "harness_combo_pilot"
    data_root = run_root / "results"
    build = read_json(build_path)
    complete = read_json(complete_path)
    require(data_root.is_dir(), f"missing result tree: {data_root}")
    require(binary_path.is_file(), f"missing experiment binary: {binary_path}")
    binary_sha = sha256_file(binary_path)
    require(build.get("status") == "complete",
            "BUILD.json does not record a completed build")
    require(complete.get("status") == "complete",
            "RUN_COMPLETE.json does not record a completed run")
    require(require_sha(build.get("binary_sha256"),
                        "BUILD.binary_sha256") == binary_sha,
            "BUILD binary SHA does not match the retained executable")
    require(require_sha(complete.get("binary_sha256"),
                        "RUN_COMPLETE.binary_sha256") == binary_sha,
            "RUN_COMPLETE binary SHA does not match BUILD/executable")
    require(require_sha(complete.get("plan_sha256"),
                        "RUN_COMPLETE.plan_sha256") == plan_sha,
            "RUN_COMPLETE plan SHA does not match runroot/PLAN.json")
    require(build.get("nncp") == "OFF",
            "BUILD.json must record NNCP=OFF")
    require(build.get("dependencies_rebuilt") is False,
            "BUILD.json must record that shared dependencies were not rebuilt")
    finite_number(complete.get("timestamp"), "RUN_COMPLETE.timestamp")
    return data_root, {
        "build_receipt_sha256": sha256_file(build_path),
        "run_complete_receipt_sha256": sha256_file(complete_path),
        "binary_path": str(binary_path),
        "binary_sha256": binary_sha,
        "plan_sha256": plan_sha,
        "result_root": str(data_root),
        "nncp": "OFF",
        "dependencies_rebuilt": False,
    }


def validate_catalog(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    catalog = read_json(root / "CATALOG.json")
    require(catalog.get("schema") == "harness-combo-pilot-catalog-v1",
            "unexpected CATALOG schema")
    rows = catalog.get("candidates")
    require(isinstance(rows, list) and len(rows) == 30,
            "CATALOG must contain 30 pilot candidates")
    result: dict[str, dict[str, Any]] = {}
    ordinals: set[int] = set()
    for row in rows:
        require(isinstance(row, dict), "CATALOG candidate is not object")
        candidate_id = row.get("id")
        require(isinstance(candidate_id, str) and candidate_id,
                "CATALOG candidate id")
        require(candidate_id == row.get("candidate_id"),
                f"catalog candidate_id mismatch: {candidate_id}")
        require(candidate_id not in result, f"duplicate candidate: {candidate_id}")
        ordinal = integer(row.get("candidate_ordinal"),
                          f"{candidate_id}.candidate_ordinal")
        require(ordinal not in ordinals, f"duplicate candidate ordinal {ordinal}")
        ordinals.add(ordinal)
        kind = row.get("kind")
        require(kind in KINDS, f"{candidate_id}: invalid kind {kind}")
        require(isinstance(row.get("new_combination"), bool),
                f"{candidate_id}: new_combination is not bool")
        require(row.get("strict_only") is True and
                row.get("allows_harness_raw_fallback") is False,
                f"{candidate_id}: catalog is not strict-only")
        dtypes = row.get("supported_dtypes")
        require(isinstance(dtypes, list) and dtypes and
                set(dtypes) <= set(DTYPES) | {"int32"},
                f"{candidate_id}: invalid supported dtypes")
        result[candidate_id] = row
    require(ordinals == set(range(30)), "CATALOG ordinals must be 0..29")
    require(sum(not row["new_combination"] for row in rows) == 18,
            "old-control pool must contain 18 candidates")
    require(sum(bool(row["new_combination"]) for row in rows) == 12,
            "new-combination pool must contain 12 candidates")
    counts = Counter(category(row) for row in rows)
    require(counts == Counter({"algorithm": 2, "operator_tree": 10,
                               "algorithm_combination": 6,
                               "new_combination": 12}),
            f"unexpected candidate classification: {dict(counts)}")
    excluded = catalog.get("scope", {}).get("excluded_families")
    require(isinstance(excluded, list) and "NNCP" in excluded,
            "CATALOG must explicitly exclude NNCP")
    return catalog, result


def timing_value(receipt: dict[str, Any], name: str, context: str,
                 expected_samples: int) -> int:
    timing = receipt.get(name)
    require(isinstance(timing, dict), f"{context}.{name}: missing timing")
    value = integer(timing.get("cpu_ns"), f"{context}.{name}.cpu_ns")
    integer(timing.get("wall_ns"), f"{context}.{name}.wall_ns")
    for field in ("cpu_samples_ns", "wall_samples_ns"):
        samples = timing.get(field)
        require(isinstance(samples, list) and len(samples) == expected_samples,
                f"{context}.{name}.{field}: sample count mismatch")
        for index, sample in enumerate(samples):
            integer(sample, f"{context}.{name}.{field}[{index}]")
    return value


def validate_complete_frame(
    result: dict[str, Any], directory: Path, column: dict[str, Any],
    source_values_sha: str, source_missing_sha: str, source_row_count: int, *,
    warmups: int, repeats: int, expected_strict: bool, context: str,
) -> dict[str, Any]:
    require(result.get("status") == "ok", f"{context}: result is not ok")
    require(result.get("strict") is expected_strict,
            f"{context}: strict/bypass classification mismatch")
    require(result.get("bit_exact") is True, f"{context}: bit_exact is not true")
    require(result.get("warmups") == warmups and result.get("repeats") == repeats,
            f"{context}: W/R timing contract mismatch")
    compressed = integer(result.get("compressed_bytes"),
                         f"{context}.compressed_bytes", minimum=1)
    require(result.get("original_numeric_bytes") ==
            column.get("original_numeric_bytes"),
            f"{context}: original byte count mismatch")
    timing_value(result, "encode", context, repeats)
    timing_value(result, "decode", context, repeats)
    all_iterations = result.get("measurement_all_iterations")
    require(isinstance(all_iterations, dict),
            f"{context}: missing all-iteration timing")
    integer(all_iterations.get("cpu_ns"), f"{context}.measurement cpu")
    integer(all_iterations.get("wall_ns"), f"{context}.measurement wall")

    frame = directory / "frame.ngf"
    frame_sha = require_sha(result.get("frame_sha256"), f"{context}.frame_sha256")
    require(sha256_file(frame) == frame_sha, f"{context}: frame SHA-256 mismatch")
    require(frame.stat().st_size == compressed,
            f"{context}: compressed byte count differs from frame length")
    receipt_suffix_matches(result.get("frame_path"), frame,
                           f"{context}.frame_path")

    decode_path = directory / "DECODE.json"
    decode = read_json(decode_path)
    embedded = result.get("independent_decode")
    require(isinstance(embedded, dict), f"{context}: missing independent decode")
    compare_json(embedded, decode, f"{context}.DECODE")
    require(decode.get("verified") is True and
            decode.get("frame_only_decode") is True,
            f"{context}: independent decode was not verified from frame only")
    require(require_sha(decode.get("frame_sha256"),
                        f"{context}.DECODE.frame_sha256") == frame_sha,
            f"{context}: DECODE frame SHA mismatch")
    require(decode.get("dtype") == column.get("dtype"),
            f"{context}: DECODE dtype mismatch")
    require(integer(decode.get("row_count"), f"{context}.DECODE.row_count") ==
            source_row_count, f"{context}: DECODE row count mismatch")
    require(require_sha(decode.get("values_sha256"),
                        f"{context}.DECODE.values_sha256") == source_values_sha,
            f"{context}: restored values SHA mismatch")
    require(require_sha(decode.get("missing_sha256"),
                        f"{context}.DECODE.missing_sha256") == source_missing_sha,
            f"{context}: restored missing bitmap SHA mismatch")
    return {"frame": frame, "decode": decode_path}


def result_metrics(result: dict[str, Any]) -> tuple[int, int, int]:
    return (integer(result.get("compressed_bytes"), "result bytes", minimum=1),
            integer(result.get("encode", {}).get("cpu_ns"), "encode CPU"),
            integer(result.get("decode", {}).get("cpu_ns"), "decode CPU"))


def best_result(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(rows)
    if not values:
        return None
    return min(values, key=lambda row: (integer(row["compressed_bytes"], "bytes"),
                                        row["candidate_id"]))


def oracle_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(rows)
    winner = best_result(values)
    if winner is None:
        return None
    best_bytes = winner["compressed_bytes"]
    ties = sorted(row["candidate_id"] for row in values
                  if row["compressed_bytes"] == best_bytes)
    return {"candidate_id": winner["candidate_id"],
            "tie_ids": ties,
            "compressed_bytes": best_bytes,
            "encode_cpu_ns": winner["encode"]["cpu_ns"],
            "decode_cpu_ns": winner["decode"]["cpu_ns"]}


def pareto_frontier(rows: Iterable[dict[str, Any]],
                    catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(rows)
    frontier: list[dict[str, Any]] = []
    for row in values:
        metrics = result_metrics(row)
        dominated = False
        for other in values:
            if other is row:
                continue
            other_metrics = result_metrics(other)
            if (all(left <= right for left, right in zip(other_metrics, metrics)) and
                    any(left < right for left, right in zip(other_metrics, metrics))):
                dominated = True
                break
        if not dominated:
            entry = catalog[row["candidate_id"]]
            frontier.append({
                "candidate_id": row["candidate_id"],
                "classification": category(entry),
                "kind": entry["kind"],
                "new_combination": entry["new_combination"],
                "compressed_bytes": metrics[0],
                "encode_cpu_ns": metrics[1],
                "decode_cpu_ns": metrics[2],
            })
    return sorted(frontier, key=lambda row: (
        row["compressed_bytes"], row["encode_cpu_ns"],
        row["decode_cpu_ns"], row["candidate_id"]))


def zero_counts() -> dict[str, int]:
    return {name: 0 for name in
            ("eligible", "strict", "bypass", "errors", "ineligible")}


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for name in target:
        target[name] += source[name]


def audit_matrix(
    root: Path, selections: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], set[Path], set[Path]]:
    catalog_ids = set(catalog)
    old_ids = {candidate_id for candidate_id, row in catalog.items()
               if not row["new_combination"]}
    new_ids = catalog_ids - old_ids
    require(len(old_ids) == 18 and len(new_ids) == 12,
            "old/new candidate pool size drift")

    columns: dict[str, dict[str, Any]] = {}
    expected_frames: set[Path] = set()
    expected_decodes: set[Path] = set()
    stage_counts = {stage: zero_counts() for stage in STAGES}
    stage_classes = {stage: {name: zero_counts() for name in
                             (*KINDS, "new_combination")}
                     for stage in STAGES}
    stage_oracle_accumulator: dict[str, dict[str, int]] = {
        stage: defaultdict(int) for stage in STAGES}
    stage_class_oracles: dict[str, dict[str, dict[str, int]]] = {
        stage: {name: {"columns_with_feasible": 0, "compressed_bytes": 0}
                for name in (*KINDS, "new_combination")}
        for stage in STAGES
    }

    for column in selections:
        key = column_key(column)
        directory = root / "matrix" / key
        column_path = directory / "COLUMN.json"
        receipt = read_json(column_path)
        compare_json(receipt.get("column"), column, f"{key}.column")
        values_sha = require_sha(receipt.get("source_values_sha256"),
                                 f"{key}.source_values_sha256")
        missing_sha = require_sha(receipt.get("source_missing_sha256"),
                                  f"{key}.source_missing_sha256")
        row_count = integer(receipt.get("row_count"), f"{key}.row_count")
        load = receipt.get("load")
        require(isinstance(load, dict), f"{key}: missing source load timing")
        integer(load.get("cpu_ns"), f"{key}.load.cpu_ns")
        integer(load.get("wall_ns"), f"{key}.load.wall_ns")
        results = receipt.get("results")
        ineligible = receipt.get("ineligible")
        require(isinstance(results, list) and isinstance(ineligible, list),
                f"{key}: malformed matrix result arrays")
        require(all(isinstance(item, str) for item in ineligible),
                f"{key}: malformed ineligible candidate")
        result_ids: list[str] = []
        strict_rows: list[dict[str, Any]] = []
        counts = zero_counts()
        class_counts = {name: zero_counts() for name in
                        (*KINDS, "new_combination")}

        for result in results:
            require(isinstance(result, dict), f"{key}: matrix result is not object")
            candidate_id = result.get("candidate_id")
            require(candidate_id in catalog, f"{key}: unknown candidate {candidate_id}")
            result_ids.append(candidate_id)
            classification = category(catalog[candidate_id])
            counts["eligible"] += 1
            class_counts[classification]["eligible"] += 1
            status = result.get("status")
            candidate_directory = directory / candidate_id
            if status == "error":
                require(result.get("rank_eligible") is False and
                        isinstance(result.get("error"), str),
                        f"{key}/{candidate_id}: malformed error receipt")
                committed_error = read_json(candidate_directory / "FAILED.json")
                compare_json(committed_error, result,
                             f"{key}/{candidate_id}.FAILED")
                counts["errors"] += 1
                class_counts[classification]["errors"] += 1
                continue
            require(status == "ok" and isinstance(result.get("strict"), bool),
                    f"{key}/{candidate_id}: invalid matrix status")
            committed = read_json(candidate_directory / "RESULT.json")
            compare_json(committed, result, f"{key}/{candidate_id}.RESULT")
            if result["strict"]:
                paths = validate_complete_frame(
                    result, candidate_directory, column, values_sha, missing_sha,
                    row_count, warmups=1, repeats=3, expected_strict=True,
                    context=f"matrix/{key}/{candidate_id}")
                expected_frames.add(paths["frame"])
                expected_decodes.add(paths["decode"])
                strict_rows.append(result)
                counts["strict"] += 1
                class_counts[classification]["strict"] += 1
            else:
                # A bypass is retained for accounting but cannot enter any oracle.
                paths = validate_complete_frame(
                    result, candidate_directory, column, values_sha, missing_sha,
                    row_count, warmups=1, repeats=3, expected_strict=False,
                    context=f"matrix/{key}/{candidate_id}")
                expected_frames.add(paths["frame"])
                expected_decodes.add(paths["decode"])
                counts["bypass"] += 1
                class_counts[classification]["bypass"] += 1

        require(len(result_ids) == len(set(result_ids)),
                f"{key}: duplicate matrix candidate result")
        require(len(ineligible) == len(set(ineligible)),
                f"{key}: duplicate ineligible candidate")
        require(not (set(result_ids) & set(ineligible)),
                f"{key}: candidate both attempted and ineligible")
        require(set(result_ids) | set(ineligible) == catalog_ids,
                f"{key}: matrix does not partition the entire catalog")
        counts["ineligible"] = len(ineligible)
        for candidate_id in ineligible:
            require(candidate_id in catalog, f"{key}: unknown ineligible candidate")
            class_counts[category(catalog[candidate_id])]["ineligible"] += 1
        require(counts["eligible"] + counts["ineligible"] == len(catalog),
                f"{key}: eligibility count mismatch")
        require(strict_rows, f"{key}: no strict candidate for oracle")

        full = oracle_summary(strict_rows)
        old = oracle_summary(row for row in strict_rows
                             if row["candidate_id"] in old_ids)
        new = oracle_summary(row for row in strict_rows
                             if row["candidate_id"] in new_ids)
        require(full is not None and old is not None,
                f"{key}: missing full or old-control oracle")
        incremental = 0 if new is None else max(
            0, old["compressed_bytes"] - new["compressed_bytes"])
        class_oracles: dict[str, Any] = {}
        for name in (*KINDS, "new_combination"):
            value = oracle_summary(
                row for row in strict_rows
                if category(catalog[row["candidate_id"]]) == name)
            class_oracles[name] = value
            if value is not None:
                stage_class_oracles[column["stage"]][name]["columns_with_feasible"] += 1
                stage_class_oracles[column["stage"]][name]["compressed_bytes"] += \
                    value["compressed_bytes"]

        columns[key] = {
            "dataset_id": column["dataset_id"],
            "column_id": column["column_id"],
            "column": column["column"],
            "dtype": column["dtype"],
            "source_group": column["source_group"],
            "dev_fold": column["dev_fold"],
            "stage": column["stage"],
            "size_bucket": column["size_bucket"],
            "original_numeric_bytes": column["original_numeric_bytes"],
            "matrix_receipt_sha256": sha256_file(column_path),
            "counts": counts,
            "classification_counts": class_counts,
            "full_pool_oracle": full,
            "old_control_18_oracle": old,
            "best_new_12": new,
            "new_12_incremental_bytes": incremental,
            "class_oracles": class_oracles,
            "pareto_frontier": pareto_frontier(strict_rows, catalog),
        }
        stage = column["stage"]
        add_counts(stage_counts[stage], counts)
        for name in class_counts:
            add_counts(stage_classes[stage][name], class_counts[name])
        accumulator = stage_oracle_accumulator[stage]
        accumulator["columns"] += 1
        accumulator["original_numeric_bytes"] += column["original_numeric_bytes"]
        accumulator["full_pool_bytes"] += full["compressed_bytes"]
        accumulator["old_control_18_bytes"] += old["compressed_bytes"]
        if new is not None:
            accumulator["columns_with_eligible_new"] += 1
            accumulator["best_new_12_bytes"] += new["compressed_bytes"]
        accumulator["new_12_incremental_bytes"] += incremental
        if incremental > 0:
            accumulator["columns_improved_by_new_12"] += 1

    for stage in STAGES:
        stage_path = root / f"MATRIX_{stage}.json"
        committed = read_json(stage_path)
        require(committed.get("stage") == stage and committed.get("status") in
                {"complete", "complete_with_rejections"},
                f"MATRIX_{stage}: invalid completion receipt")
        expected = stage_counts[stage]
        require(committed.get("cases") == expected["eligible"],
                f"MATRIX_{stage}: cases mismatch")
        require(committed.get("failures") == expected["errors"],
                f"MATRIX_{stage}: failures mismatch")
        require(committed.get("bypass_cases") == expected["bypass"],
                f"MATRIX_{stage}: bypass count mismatch")

    summary = {
        "columns": len(columns),
        "counts": zero_counts(),
        "by_stage": {},
        "classification_definition": {
            "algorithm": "catalog kind=algorithm and not new",
            "operator_tree": "catalog kind=operator_tree and not new",
            "algorithm_combination":
                "catalog kind=algorithm_combination and not new",
            "new_combination": "catalog new_combination=true",
        },
    }
    for stage in STAGES:
        add_counts(summary["counts"], stage_counts[stage])
        summary["by_stage"][stage] = {
            "counts": stage_counts[stage],
            "classification_counts": stage_classes[stage],
            "oracle_totals": dict(stage_oracle_accumulator[stage]),
            "class_oracle_totals": stage_class_oracles[stage],
        }
    return summary, columns, expected_frames, expected_decodes


def validate_prior(
    root: Path, selections: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    path = root / "PRIOR.json"
    prior = read_json(path)
    require(prior.get("schema") == "harness-combo-discovery-prior-v1",
            "unexpected PRIOR schema")
    require(prior.get("training_stage") == "discovery" and
            prior.get("confirm_labels_in_fit") is False,
            "PRIOR violates discovery-only/no-confirm contract")
    expected = {column_key(column): column for column in selections
                if column["stage"] == "discovery"}
    sources = prior.get("sources")
    require(isinstance(sources, list) and len(sources) == len(expected),
            "PRIOR discovery source count mismatch")
    seen: set[str] = set()
    for source in sources:
        require(isinstance(source, dict), "PRIOR source is not object")
        key = source.get("key")
        require(key in expected and key not in seen, f"PRIOR invalid source: {key}")
        seen.add(key)
        column = expected[key]
        require(source.get("dev_fold") == column["dev_fold"] and
                source.get("source_group") == column["source_group"],
                f"PRIOR source metadata mismatch: {key}")
        expected_sha = sha256_file(root / "matrix" / key / "COLUMN.json")
        require(source.get("matrix_sha256") == expected_sha,
                f"PRIOR matrix SHA mismatch: {key}")
    require(seen == set(expected), "PRIOR source set mismatch")
    dtypes = prior.get("dtypes")
    require(isinstance(dtypes, dict), "PRIOR.dtypes is not object")
    for dtype in DTYPES:
        ranking = dtypes.get(dtype)
        require(isinstance(ranking, list) and ranking,
                f"PRIOR missing nonempty ranking for {dtype}")
        ids: set[str] = set()
        for row in ranking:
            candidate_id = row.get("id") if isinstance(row, dict) else None
            require(candidate_id in catalog and candidate_id not in ids,
                    f"PRIOR invalid/duplicate {dtype} candidate: {candidate_id}")
            require(dtype in catalog[candidate_id]["supported_dtypes"],
                    f"PRIOR dtype-ineligible candidate: {dtype}/{candidate_id}")
            finite_number(row.get("score"), f"PRIOR {dtype}/{candidate_id}.score")
            integer(row.get("training_columns"),
                    f"PRIOR {dtype}/{candidate_id}.training_columns", minimum=1)
            integer(row.get("training_source_groups"),
                    f"PRIOR {dtype}/{candidate_id}.training_source_groups", minimum=1)
            ids.add(candidate_id)
    return prior, sha256_file(path)


def policy_definitions(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = read_json(root / "POLICIES.json")
    require(receipt.get("no_confirm") is True,
            "POLICIES does not record the no-confirm contract")
    require(receipt.get("fit_scope") ==
            "discovery folds 0-2; policy selected on fold3 only",
            "POLICIES fit scope drift")
    rows = receipt.get("policies")
    require(isinstance(rows, list), "POLICIES.policies is not an array")
    definitions: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "policy definition is not an object")
        policy_id = row["id"]
        require(isinstance(policy_id, str) and policy_id and
                policy_id not in BASELINES,
                f"invalid or reserved online policy id: {policy_id}")
        require(policy_id not in definitions, f"duplicate policy {policy_id}")
        integer(row.get("windows"), f"{policy_id}.windows", minimum=1)
        integer(row.get("valid_values_per_window"),
                f"{policy_id}.valid_values_per_window", minimum=1)
        integer(row.get("max_candidates"),
                f"{policy_id}.max_candidates", minimum=2)
        definitions[policy_id] = row
    require(definitions, "POLICIES must define at least one online policy")
    return definitions, receipt


def online_metrics(result: dict[str, Any], policy_id: str,
                   oracle_bytes: int, source_group: str,
                   *, probe_policy: bool) -> dict[str, Any]:
    selected = result["selected"]
    selected_bytes = integer(selected.get("compressed_bytes"),
                             f"{policy_id}.selected bytes", minimum=1)
    online_cpu = integer(result.get("online_total_cpu_ns"),
                         f"{policy_id}.online_total_cpu_ns")
    online_wall = integer(result.get("online_total_wall_ns"),
                          f"{policy_id}.online_total_wall_ns")
    value: dict[str, Any] = {
        "source_group": source_group,
        "selected_id": result.get("selected_id"),
        "compressed_bytes": selected_bytes,
        "oracle_bytes": oracle_bytes,
        "excess_bytes": selected_bytes - oracle_bytes,
        "byte_ratio_to_oracle": selected_bytes / oracle_bytes,
        "online_total_cpu_ns": online_cpu,
        "online_total_wall_ns": online_wall,
        "probe_cpu_ns": 0,
        "probe_wall_ns": 0,
        "probe_encode_cpu_ns": 0,
        "probe_decode_verification_cpu_ns": 0,
        "proposed_candidate_count": 0,
        "probe_window_attempt_count": 0,
        "candidate_budget": 0,
        "candidate_budget_overshoot": 0,
        "fallback": False,
        "no_feasible_probe_candidate": False,
    }
    if probe_policy:
        probe = result.get("probe_all")
        require(isinstance(probe, dict), f"{policy_id}: missing probe timing")
        value["probe_cpu_ns"] = integer(probe.get("cpu_ns"),
                                         f"{policy_id}.probe CPU")
        value["probe_wall_ns"] = integer(probe.get("wall_ns"),
                                          f"{policy_id}.probe wall")
        value["probe_encode_cpu_ns"] = integer(
            result.get("probe_encode_cpu_ns"), f"{policy_id}.probe encode CPU")
        value["probe_decode_verification_cpu_ns"] = integer(
            result.get("probe_decode_verification_cpu_ns"),
            f"{policy_id}.probe decode CPU")
        proposed = integer(result.get("proposed_candidate_count"),
                           f"{policy_id}.proposed count")
        budget = integer(result.get("policy", {}).get("max_candidates"),
                         f"{policy_id}.candidate budget", minimum=2)
        trials = result.get("trials")
        require(isinstance(trials, list) and len(trials) == proposed,
                f"{policy_id}: trial/candidate count mismatch")
        attempts = 0
        for trial in trials:
            windows = trial.get("windows") if isinstance(trial, dict) else None
            require(isinstance(windows, list), f"{policy_id}: malformed trial")
            attempts += len(windows)
        value.update({
            "proposed_candidate_count": proposed,
            "probe_window_attempt_count": attempts,
            "candidate_budget": budget,
            "candidate_budget_overshoot": max(0, proposed - budget),
            "fallback": result.get("fallback") is True,
            "no_feasible_probe_candidate":
                result.get("no_feasible_probe_candidate") is True,
        })
    return value


def summarize_online_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(rows, "cannot summarize empty online policy")
    summed_fields = (
        "compressed_bytes", "oracle_bytes", "excess_bytes",
        "online_total_cpu_ns", "online_total_wall_ns", "probe_cpu_ns",
        "probe_wall_ns", "probe_encode_cpu_ns",
        "probe_decode_verification_cpu_ns", "proposed_candidate_count",
        "probe_window_attempt_count", "candidate_budget",
        "candidate_budget_overshoot",
    )
    summary: dict[str, Any] = {"columns": len(rows)}
    for field in summed_fields:
        summary[field] = sum(row[field] for row in rows)
    summary["byte_ratio_to_oracle"] = (
        summary["compressed_bytes"] / summary["oracle_bytes"])
    summary["fallback_columns"] = sum(bool(row["fallback"]) for row in rows)
    summary["no_feasible_probe_columns"] = sum(
        bool(row["no_feasible_probe_candidate"]) for row in rows)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {
        "columns": 0, "compressed_bytes": 0, "oracle_bytes": 0,
        "excess_bytes": 0})
    for row in rows:
        group = groups[row["source_group"]]
        group["columns"] += 1
        group["compressed_bytes"] += row["compressed_bytes"]
        group["oracle_bytes"] += row["oracle_bytes"]
        group["excess_bytes"] += row["excess_bytes"]
    group_rows = []
    for group_id, value in groups.items():
        group_rows.append({"source_group": group_id, **value,
                           "byte_ratio_to_oracle":
                               value["compressed_bytes"] / value["oracle_bytes"]})
    group_rows.sort(key=lambda row: (-row["byte_ratio_to_oracle"],
                                     -row["excess_bytes"], row["source_group"]))
    summary["worst_source_group"] = group_rows[0]
    summary["by_source_group"] = group_rows
    return summary


def validate_policy_result(
    result: dict[str, Any], policy_id: str, definition: dict[str, Any],
    catalog: dict[str, dict[str, Any]], context: str,
) -> None:
    compare_json(result.get("policy"), definition, f"{context}.policy")
    proposed = result.get("proposed_ids")
    trials = result.get("trials")
    require(isinstance(proposed, list) and isinstance(trials, list),
            f"{context}: malformed probe arrays")
    require(len(proposed) == len(set(proposed)) and
            all(candidate_id in catalog for candidate_id in proposed),
            f"{context}: invalid proposed candidate set")
    require(result.get("proposed_candidate_count") == len(proposed) and
            len(trials) == len(proposed),
            f"{context}: proposed count mismatch")
    require(len(proposed) <= definition["max_candidates"],
            f"{context}: hard candidate budget exceeded")
    starts = result.get("window_start_rows")
    valid_starts = result.get("window_start_valid_values")
    require(isinstance(starts, list) and isinstance(valid_starts, list) and
            len(starts) == definition["windows"] and
            len(valid_starts) == definition["windows"],
            f"{context}: probe window count mismatch")
    require(len(valid_starts) == len(set(valid_starts)),
            f"{context}: probe valid-value windows overlap at start")
    n = integer(result.get("valid_values_per_window"),
                f"{context}.valid_values_per_window", minimum=1)
    for left, right in zip(valid_starts, valid_starts[1:]):
        require(integer(right, f"{context}.valid start") >=
                integer(left, f"{context}.valid start") + n,
                f"{context}: valid-value probe windows overlap")
    trial_ids: list[str] = []
    strict_trials = 0
    feasible_ids: set[str] = set()
    recomputed_probe_encode_cpu = 0
    recomputed_probe_decode_cpu = 0
    for trial in trials:
        require(isinstance(trial, dict), f"{context}: malformed trial")
        candidate_id = trial.get("candidate_id")
        trial_ids.append(candidate_id)
        windows = trial.get("windows")
        require(isinstance(windows, list) and len(windows) == definition["windows"],
                f"{context}/{candidate_id}: window result count mismatch")
        all_strict = True
        score_bytes = 0
        score_input_bytes = 0
        for window_index, window in enumerate(windows):
            require(isinstance(window, dict), f"{context}: malformed window")
            if window.get("status") == "ok":
                require(isinstance(window.get("strict"), bool),
                        f"{context}: probe strict flag missing")
                require(window.get("candidate_id") == candidate_id and
                        window.get("bit_exact") is True and
                        window.get("warmups") == 0 and
                        window.get("repeats") == 1 and
                        window.get("window") == window_index and
                        window.get("start_row") == starts[window_index],
                        f"{context}/{candidate_id}: malformed probe measurement")
                require_sha(window.get("frame_sha256"),
                            f"{context}/{candidate_id}/window frame SHA")
                encode_cpu = timing_value(
                    window, "encode", f"{context}/{candidate_id}/window", 1)
                decode_cpu = timing_value(
                    window, "decode", f"{context}/{candidate_id}/window", 1)
                recomputed_probe_encode_cpu += encode_cpu
                recomputed_probe_decode_cpu += decode_cpu
                score_bytes += integer(
                    window.get("compressed_bytes"),
                    f"{context}/{candidate_id}/window bytes", minimum=1)
                score_input_bytes += integer(
                    window.get("original_numeric_bytes"),
                    f"{context}/{candidate_id}/window input bytes", minimum=1)
                if window["strict"]:
                    strict_trials += 1
                else:
                    all_strict = False
            else:
                require(window.get("status") in {"error", "empty_sample"},
                        f"{context}: invalid probe status")
                all_strict = False
        feasible = trial.get("feasible")
        require(isinstance(feasible, bool) and feasible == all_strict,
                f"{context}/{candidate_id}: feasible flag mismatch")
        if feasible:
            score = finite_number(trial.get("score"),
                                  f"{context}/{candidate_id}.score")
            require(score_input_bytes > 0 and
                    math.isclose(score, score_bytes / score_input_bytes,
                                 rel_tol=1e-12, abs_tol=1e-15),
                    f"{context}/{candidate_id}: probe score mismatch")
            feasible_ids.add(candidate_id)
        else:
            require(trial.get("score") is None,
                    f"{context}/{candidate_id}: infeasible score must be null")
    require(trial_ids == proposed, f"{context}: trial order differs from proposal")
    require(result.get("strict_trial_count") == strict_trials,
            f"{context}: strict trial count mismatch")
    require(result.get("probe_encode_cpu_ns") == recomputed_probe_encode_cpu and
            result.get("probe_decode_verification_cpu_ns") ==
            recomputed_probe_decode_cpu,
            f"{context}: probe CPU component sum mismatch")
    no_feasible = not feasible_ids
    require(result.get("no_feasible_probe_candidate") is no_feasible,
            f"{context}: no-feasible flag mismatch")
    chosen = result.get("chosen_from_probes")
    if feasible_ids:
        scored = {trial["candidate_id"]: trial["score"] for trial in trials
                  if trial["feasible"]}
        expected = min(scored, key=lambda candidate_id:
                       (scored[candidate_id], candidate_id))
        require(chosen == expected,
                f"{context}: chosen probe candidate is not score minimum")
    require(result.get("selected_id") in catalog,
            f"{context}: selected candidate is not in catalog")
    for flag in ("source_file_io_excluded", "parent_startup_excluded",
                 "verifier_startup_included"):
        require(result.get(flag) is True, f"{context}: timing flag {flag} drift")


def audit_online(
    root: Path, selections: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]], matrix_columns: dict[str, dict[str, Any]],
    prior: dict[str, Any], prior_sha: str,
    policy_defs: dict[str, dict[str, Any]],
    expected_frames: set[Path], expected_decodes: set[Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    lock_path = root / "POLICY_LOCK.json"
    lock = read_json(lock_path)
    online_policy_ids = tuple(policy_defs)
    tuning_methods = online_policy_ids + BASELINES
    selected_policy = lock.get("selected_policy")
    require(selected_policy in tuning_methods,
            f"POLICY_LOCK selected unknown method: {selected_policy}")
    require(lock.get("selection_stage") == "tuning fold3 only" and
            lock.get("confirm_used") is False and
            lock.get("prior_sha256") == prior_sha,
            "POLICY_LOCK violates tuning-only/no-confirm/prior binding")

    online_rows: dict[str, dict[str, dict[str, Any]]] = {
        "tuning": defaultdict(dict), "locked_validation": defaultdict(dict)}
    source_hashes: dict[str, tuple[str, str, int]] = {}
    matrix_strict_results: dict[str, dict[str, dict[str, Any]]] = {}
    for key in matrix_columns:
        receipt = read_json(root / "matrix" / key / "COLUMN.json")
        source_hashes[key] = (
            receipt["source_values_sha256"], receipt["source_missing_sha256"],
            integer(receipt.get("row_count"), f"matrix/{key}.row_count"))
        matrix_strict_results[key] = {
            row["candidate_id"]: row for row in receipt["results"]
            if row.get("status") == "ok" and row.get("strict") is True
        }
    index_sha: dict[str, str] = {}
    for column in selections:
        stage = column["stage"]
        if stage == "discovery":
            continue
        key = column_key(column)
        directory = root / "online" / key
        index_path = directory / "COLUMN.json"
        index = read_json(index_path)
        compare_json(index.get("column"), column, f"online/{key}.column")
        require(index.get("status") == "complete" and
                index.get("prior_sha256") == prior_sha,
                f"online/{key}: incomplete or stale prior")
        records = index.get("records")
        require(isinstance(records, list), f"online/{key}: records not array")
        expected_methods = set(tuning_methods if stage == "tuning" else
                               BASELINES + ((selected_policy,)
                                            if selected_policy in online_policy_ids
                                            else ()))
        record_ids = [record.get("policy_id") for record in records
                      if isinstance(record, dict)]
        require(len(record_ids) == len(records) and
                len(record_ids) == len(set(record_ids)) and
                set(record_ids) == expected_methods,
                f"online/{key}: policy record set mismatch")
        baseline_first = column["selection_index"] % 2 == 0
        active_policies = list(online_policy_ids if stage == "tuning" else
                               ((selected_policy,)
                                if selected_policy in online_policy_ids else ()))
        expected_order = (list(BASELINES) + active_policies if baseline_first
                          else active_policies + list(BASELINES))
        require(record_ids == expected_order,
                f"online/{key}: baseline-order rotation mismatch")
        index_sha[key] = sha256_file(index_path)
        for record in records:
            policy_id = record["policy_id"]
            result_path = directory / policy_id / "RESULT.json"
            receipt_suffix_matches(record.get("path"), result_path,
                                   f"online/{key}/{policy_id}.path")
            require(record.get("sha256") == sha256_file(result_path),
                    f"online/{key}/{policy_id}: index SHA mismatch")
            result = read_json(result_path)
            compare_json(result.get("column"), column,
                         f"online/{key}/{policy_id}.column")
            require(result.get("prior_sha256") == prior_sha,
                    f"online/{key}/{policy_id}: prior SHA mismatch")
            if policy_id in online_policy_ids:
                validate_policy_result(result, policy_id, policy_defs[policy_id],
                                       catalog, f"online/{key}/{policy_id}")
            else:
                require(result.get("probe_count") == 0,
                        f"online/{key}/{policy_id}: baseline probe count is not zero")
                expected_id = ("zstd3" if policy_id == "fixed_zstd3" else
                               "lz4" if policy_id == "fixed_lz4" else None)
                if expected_id is not None:
                    require(result.get("selected_id") == expected_id,
                            f"online/{key}/{policy_id}: fixed baseline ID mismatch")
                else:
                    ranking = prior["dtypes"][column["dtype"]]
                    require(result.get("selected_id") == ranking[0]["id"],
                            f"online/{key}/prior_top1: discovery prior mismatch")
            selected = result.get("selected")
            require(isinstance(selected, dict) and
                    selected.get("candidate_id") == result.get("selected_id"),
                    f"online/{key}/{policy_id}: selected result mismatch")
            matrix_selected = matrix_strict_results[key].get(result["selected_id"])
            require(matrix_selected is not None,
                    f"online/{key}/{policy_id}: selection lacks strict matrix result")
            require(selected.get("compressed_bytes") ==
                    matrix_selected.get("compressed_bytes") and
                    selected.get("frame_sha256") ==
                    matrix_selected.get("frame_sha256"),
                    f"online/{key}/{policy_id}: one-shot frame differs from matrix frame")
            require(selected["compressed_bytes"] >=
                    matrix_columns[key]["full_pool_oracle"]["compressed_bytes"],
                    f"online/{key}/{policy_id}: negative same-pool oracle excess")
            if policy_id in online_policy_ids:
                require(result.get("baseline_first_for_column") is baseline_first,
                        f"online/{key}/{policy_id}: baseline-order flag mismatch")
            values_sha, missing_sha, row_count = source_hashes[key]
            selected_directory = directory / policy_id / "selected"
            paths = validate_complete_frame(
                selected, selected_directory, column, values_sha, missing_sha,
                row_count, warmups=0, repeats=1, expected_strict=True,
                context=f"online/{key}/{policy_id}/selected")
            expected_frames.add(paths["frame"])
            expected_decodes.add(paths["decode"])
            online_rows[stage][policy_id][key] = online_metrics(
                result, policy_id,
                matrix_columns[key]["full_pool_oracle"]["compressed_bytes"],
                column["source_group"],
                probe_policy=policy_id in online_policy_ids)

    for stage in ("tuning", "locked_validation"):
        receipt = read_json(root / f"ONLINE_{stage}.json")
        require(receipt == {"status": "complete", "stage": stage,
                            "prior_sha256": prior_sha},
                f"ONLINE_{stage}: completion receipt mismatch")

    # Every policy-lock input must be a unique tuning receipt and every tuning
    # method/column pair must be bound exactly once.
    expected_inputs: dict[tuple[str, str], tuple[Path, str]] = {}
    for column in selections:
        if column["stage"] != "tuning":
            continue
        key = column_key(column)
        for method in tuning_methods:
            path = root / "online" / key / method / "RESULT.json"
            expected_inputs[(key, method)] = (path, sha256_file(path))
    inputs = lock.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == len(expected_inputs),
            "POLICY_LOCK input count mismatch")
    observed_inputs: set[tuple[str, str]] = set()
    for item in inputs:
        require(isinstance(item, dict), "POLICY_LOCK input is not object")
        declared = item.get("path")
        require(isinstance(declared, str), "POLICY_LOCK input path missing")
        parts = PurePath(declared).parts
        require(len(parts) >= 3 and parts[-1] == "RESULT.json",
                "POLICY_LOCK input path malformed")
        key, method = parts[-3], parts[-2]
        pair = (key, method)
        require(pair in expected_inputs and pair not in observed_inputs,
                f"POLICY_LOCK non-tuning/duplicate input: {declared}")
        expected_path, expected_sha = expected_inputs[pair]
        receipt_suffix_matches(declared, expected_path, "POLICY_LOCK input")
        require(item.get("sha256") == expected_sha,
                f"POLICY_LOCK input SHA mismatch: {declared}")
        observed_inputs.add(pair)
    require(observed_inputs == set(expected_inputs),
            "POLICY_LOCK tuning input coverage mismatch")

    # Recompute the exact lock statistics and frozen selection rule.
    recomputed_stats: dict[str, dict[str, Any]] = {}
    for method in tuning_methods:
        rows = online_rows["tuning"][method]
        require(len(rows) == 9, f"tuning/{method}: expected nine columns")
        zstd_rows = online_rows["tuning"]["fixed_zstd3"]
        grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        total_bytes = total_cpu = 0
        for key, row in rows.items():
            total_bytes += row["compressed_bytes"]
            total_cpu += row["online_total_cpu_ns"]
            group = row["source_group"]
            grouped[group][0] += row["compressed_bytes"]
            grouped[group][1] += zstd_rows[key]["compressed_bytes"]
        worst = max(value[0] / value[1] for value in grouped.values())
        recomputed_stats[method] = {
            "bytes": total_bytes,
            "online_total_cpu_ns": total_cpu,
            "worst_source_byte_ratio_to_zstd3": worst,
        }
    cpu_cap = 3.0 * recomputed_stats["fixed_zstd3"]["online_total_cpu_ns"]
    risk_cap = recomputed_stats["prior_top1"]["worst_source_byte_ratio_to_zstd3"]
    feasible_methods: list[str] = []
    for method in tuning_methods:
        row = recomputed_stats[method]
        feasible = (row["online_total_cpu_ns"] <= cpu_cap and
                    row["worst_source_byte_ratio_to_zstd3"] <= risk_cap)
        row["feasible"] = feasible
        if feasible:
            feasible_methods.append(method)
    chosen = (min(feasible_methods,
                  key=lambda method: (
                      recomputed_stats[method]["bytes"],
                      recomputed_stats[method]["online_total_cpu_ns"],
                      tuning_methods.index(method)))
              if feasible_methods else "prior_top1")
    require(chosen == selected_policy,
            f"POLICY_LOCK selected {selected_policy}, recomputed {chosen}")
    require(math.isclose(finite_number(lock.get("worst_source_cap_relative_to_prior_top1"),
                                       "POLICY_LOCK risk cap"), risk_cap,
                         rel_tol=1e-12, abs_tol=1e-15),
            "POLICY_LOCK risk cap mismatch")
    require(finite_number(lock.get("cpu_cap_relative_to_fixed_zstd3"),
                          "POLICY_LOCK CPU cap") == 3.0,
            "POLICY_LOCK CPU cap mismatch")
    committed_stats = lock.get("stats")
    require(isinstance(committed_stats, dict) and
            set(committed_stats) == set(recomputed_stats),
            "POLICY_LOCK stats method set mismatch")
    for method, expected in recomputed_stats.items():
        actual = committed_stats[method]
        require(actual.get("bytes") == expected["bytes"] and
                actual.get("online_total_cpu_ns") == expected["online_total_cpu_ns"] and
                actual.get("feasible") is expected["feasible"] and
                math.isclose(finite_number(actual.get("worst_source_byte_ratio_to_zstd3"),
                                           f"POLICY_LOCK {method} worst source"),
                             expected["worst_source_byte_ratio_to_zstd3"],
                             rel_tol=1e-12, abs_tol=1e-15),
                f"POLICY_LOCK stats mismatch: {method}")

    summaries: dict[str, Any] = {}
    for stage in ("tuning", "locked_validation"):
        methods = list(tuning_methods if stage == "tuning" else
                       BASELINES + ((selected_policy,)
                                    if selected_policy in online_policy_ids else ()))
        summaries[stage] = {
            "reported_methods": methods,
            "methods": {
                method: summarize_online_rows(list(online_rows[stage][method].values()))
                for method in methods
            },
        }
        if selected_policy in BASELINES and stage == "locked_validation":
            summaries[stage]["locked_selected_alias"] = selected_policy
        else:
            summaries[stage]["locked_selected_alias"] = (
                selected_policy if stage == "locked_validation" else None)

    lock_report = {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path),
        "selected_policy": selected_policy,
        "online_policy_ids": list(online_policy_ids),
        "selection_stage": lock["selection_stage"],
        "confirm_used": False,
        "recomputed_stats": recomputed_stats,
        "tuning_input_receipts_verified": len(expected_inputs),
    }
    validation = {
        "online_column_index_sha256": index_sha,
        "online_column_indexes_verified": len(index_sha),
        "prior_sha256_bound": prior_sha,
        "policy_lock_input_sha256_verified": len(expected_inputs),
    }
    return summaries, lock_report, validation


def audit_frame_inventory(root: Path, expected_frames: set[Path],
                          expected_decodes: set[Path]) -> dict[str, Any]:
    actual_frames = set((root / "matrix").rglob("frame.ngf")) | \
        set((root / "online").rglob("frame.ngf"))
    actual_decodes = set((root / "matrix").rglob("DECODE.json")) | \
        set((root / "online").rglob("DECODE.json"))
    require(actual_frames == expected_frames,
            f"frame inventory mismatch: expected {len(expected_frames)}, "
            f"found {len(actual_frames)}")
    require(actual_decodes == expected_decodes,
            f"DECODE inventory mismatch: expected {len(expected_decodes)}, "
            f"found {len(actual_decodes)}")
    return {"complete_frames_sha256_verified": len(expected_frames),
            "decode_receipts_verified": len(expected_decodes),
            "unindexed_frames": 0, "unindexed_decode_receipts": 0}


def read_reference(reference_path: Path,
                   selections: list[dict[str, Any]]) -> dict[str, Any]:
    # The whitelist is fixed from PLAN before opening the optional reference.
    selected = {(column["dataset_id"], column["column_id"]): column
                for column in selections if column["split"] == "development"}
    datasets = {identity[0] for identity in selected}
    found: dict[tuple[str, str], dict[str, Any]] = {}
    dataset_pattern = re.compile(
        r'"dataset_id"\s*:\s*("(?:\\.|[^"\\])*")')
    require(reference_path.is_file(), f"reference JSONL not found: {reference_path}")
    with reference_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            match = dataset_pattern.search(line)
            require(match is not None,
                    f"reference line {line_number}: dataset_id metadata missing")
            try:
                dataset_metadata = json.loads(match.group(1))
            except json.JSONDecodeError as error:
                raise ReportError(
                    f"reference line {line_number}: invalid dataset_id: {error}") from error
            # Filter on raw line metadata before parsing any row that could
            # carry a label outside the selected development dataset whitelist.
            if dataset_metadata not in datasets:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReportError(
                    f"reference line {line_number}: invalid JSON: {error}") from error
            require(isinstance(row, dict),
                    f"reference line {line_number}: row is not object")
            # Do not touch any label field before the development dataset and
            # exact selected-column whitelist checks pass.
            dataset_id = row.get("dataset_id")
            require(dataset_id == dataset_metadata,
                    f"reference line {line_number}: dataset_id metadata mismatch")
            column_id = row.get("column_id")
            identity = (dataset_id, column_id)
            if identity not in selected:
                continue
            require(identity not in found,
                    f"duplicate selected reference row: {identity}")
            column = selected[identity]
            require(row.get("schema") == "current135-column-oracle-ground-truth-v1",
                    f"reference schema mismatch: {identity}")
            require(row.get("dtype") == column["dtype"] and
                    row.get("original_numeric_bytes") ==
                    column["original_numeric_bytes"] and
                    row.get("manifest_sha256") == column["manifest_sha256"],
                    f"reference identity/payload mismatch: {identity}")
            require(row.get("algorithm_candidate_count") == 85,
                    f"historical algorithm scope drift: {identity}")
            require(row.get("joint_operator_algorithm_measured") is False and
                    row.get("joint_global_optimal") is False,
                    f"historical row makes an invalid joint-optimum claim: {identity}")
            found[identity] = {
                "dataset_id": dataset_id,
                "column_id": column_id,
                "dtype": row["dtype"],
                "stage": column["stage"],
                "historical_85_algorithm": {
                    "best_algorithm": row.get("best_algorithm"),
                    "compressed_bytes": row.get("best_algorithm_compressed_bytes"),
                    "byte_boundary": row.get("best_algorithm_byte_boundary"),
                    "timing_contract": row.get("best_algorithm_timing_contract"),
                },
                "historical_97_operator_common_zstd3": {
                    "best_operator_id": row.get("best_operator_id"),
                    "compressed_bytes": row.get("best_operator_compressed_bytes"),
                    "backend": row.get("best_operator_backend"),
                    "timing_status": row.get("best_operator_timing_status"),
                },
                "historical_182_observed_union": {
                    "best_observed_pipeline_kind":
                        row.get("best_observed_pipeline_kind"),
                    "best_observed_pipeline_id": row.get("best_observed_pipeline_id"),
                    "compressed_bytes":
                        row.get("best_observed_pipeline_compressed_bytes"),
                    "joint_cross_product_measured": False,
                },
            }
    require(set(found) == set(selected),
            f"reference missing {len(set(selected) - set(found))} selected development rows")
    return {
        "path": str(reference_path),
        "sha256": sha256_file(reference_path),
        "selected_development_rows": len(found),
        "confirm_rows_read_for_labels": 0,
        "historical_context": {
            "algorithm_candidates": 85,
            "operator_candidates_common_zstd3": 97,
            "observed_union_candidates": 182,
            "full_97x85_cartesian_product_measured": False,
        },
        "comparability": (
            "Context only. The pilot uses HCB1 complete whole-column frames; "
            "the historical 85/97/182 receipts use their frozen block/frame "
            "and timing contracts. Bytes and times are not equal-contract gains."
        ),
        "rows": [found[identity] for identity in sorted(found)],
    }


def markdown_report(report: dict[str, Any]) -> str:
    matrix = report["matrix"]
    lines = [
        "# Harness 组合选型 pilot 结果",
        "",
        f"状态：**{report['status']}**。本报告由已提交收据独立重算，"
        f"覆盖 {report['scope']['columns']} 列，全部来自 development，"
        "列大小为 16 KiB–8 MiB；confirm 标签读取数为 0。",
        "",
        "## 候选与矩阵",
        "",
        "候选池为 30 项：2 个算法、10 个已有算子树、6 个已有算法组合、"
        "12 个新组合。旧对照池固定为 18 项，新组合池固定为 12 项；"
        "未测量任何自由笛卡尔积。NNCP、A106–A110、PPMd、DMC、PAQ8、"
        "CMIX、LSTM 均不在候选池。",
        "",
        "| 阶段 | 列 | eligible | strict | bypass | errors | ineligible | "
        "old18 bytes | full bytes | new12 减少字节 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        row = matrix["by_stage"][stage]
        counts = row["counts"]
        oracle = row["oracle_totals"]
        lines.append(
            f"| {stage} | {oracle['columns']} | {counts['eligible']} | "
            f"{counts['strict']} | {counts['bypass']} | {counts['errors']} | "
            f"{counts['ineligible']} | {oracle['old_control_18_bytes']} | "
            f"{oracle['full_pool_bytes']} | {oracle['new_12_incremental_bytes']} |")

    lines.extend(["", "每列报告同时保存同池 full oracle、old-control-18 oracle、"
                  "best-new-12、四类候选 oracle，以及 bytes/encode CPU/decode CPU "
                  "三目标 Pareto 前沿。bypass、error 和 ineligible 均不参与 oracle。",
                  "", "## 在线选择", ""])
    locked = report["policy_lock"]["selected_policy"]
    lines.append(f"tuning（fold 3）冻结的策略为 **{locked}**；locked-validation "
                 "只报告该冻结策略和三个固定对照。")
    lines.append("`prior_top1` 是 discovery 阶段按 dtype 冻结的候选首位，"
                 "不代表历史 PreparedModel。")
    for stage in ("tuning", "locked_validation"):
        lines.extend([
            "",
            f"### {stage}",
            "",
            "| 方法 | bytes | excess vs full oracle | CPU ms | wall ms | "
            "probe CPU ms | 候选次数 | 窗口试压次数 | budget overshoot | "
            "最差来源比值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        stage_result = report["online"][stage]
        for method in stage_result["reported_methods"]:
            value = stage_result["methods"][method]
            worst = value["worst_source_group"]["byte_ratio_to_oracle"]
            lines.append(
                f"| {method} | {value['compressed_bytes']} | "
                f"{value['excess_bytes']} | "
                f"{value['online_total_cpu_ns']/1e6:.3f} | "
                f"{value['online_total_wall_ns']/1e6:.3f} | "
                f"{value['probe_cpu_ns']/1e6:.3f} | "
                f"{value['proposed_candidate_count']} | "
                f"{value['probe_window_attempt_count']} | "
                f"{value['candidate_budget_overshoot']} | {worst:.6f} |")

    validation = report["validation"]
    lines.extend([
        "", "## 证据与口径", "",
        f"已复核 {validation['complete_frames_sha256_verified']} 个完整帧 SHA-256 "
        f"和 {validation['decode_receipts_verified']} 个独立 DECODE 收据；"
        f"{validation['online_column_indexes_verified']} 个 ONLINE 列索引及 "
        f"{validation['policy_lock_input_sha256_verified']} 个策略锁输入也已复核。",
        "",
        "RUN_COMPLETE 中的 PLAN 与 binary SHA-256 已分别和 runroot/PLAN.json、"
        "runroot/build/harness_combo_pilot 重新比对，BUILD 同一 binary 绑定也已验证。",
        "",
        "在线结果是单次完整列编码/解码与独立 verifier 的端到端收据；"
        "矩阵结果是 W1/R3 的热测量中位数。两者不是同一种计时。在线时间排除"
        "源文件 I/O 与父进程启动，包含独立 verifier 的启动和子进程 CPU。",
        "",
        "这是 41 列、单列不超过 8 MiB 的小规模研究，不能外推为 135 数据集全量收益。",
    ])
    if "historical_reference" in report:
        reference = report["historical_reference"]
        lines.extend([
            "", "## 历史 85/97/182 上下文", "",
            f"仅对白名单中的 {reference['selected_development_rows']} 个 development "
            "列读取历史字段；confirm 标签读取数为 0。历史 85 算法、97 算子（统一 "
            "Zstd-3 后端）和两者 182 项已观测并集保持分开，未声称测量 97×85 "
            "笛卡尔积。pilot 使用 HCB1 整列完整帧，而历史记录采用冻结的 block/frame "
            "和 timing contract，因此不把二者差值表述为等口径收益。",
        ])
    return "\n".join(lines) + "\n"


def atomic_write_pair(output: Path, report: dict[str, Any], markdown: str) -> None:
    report_path = output / "REPORT.json"
    markdown_path = output / "RESULTS.zh.md"
    require(not report_path.exists() and not markdown_path.exists(),
            f"refusing to overwrite report in {output}")
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        report_path: (json.dumps(report, ensure_ascii=False, indent=2,
                                 sort_keys=True) + "\n").encode("utf-8"),
        markdown_path: markdown.encode("utf-8"),
    }
    temporaries: dict[Path, Path] = {}
    try:
        for target, payload in payloads.items():
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o644)
            try:
                position = 0
                while position < len(payload):
                    position += os.write(descriptor, payload[position:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            temporaries[target] = temporary
        for target, temporary in temporaries.items():
            os.replace(temporary, target)
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        for temporary in temporaries.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def build_report(run_root: Path, reference: Path | None) -> dict[str, Any]:
    plan, selections = validate_plan(run_root)
    plan_sha = sha256_file(run_root / "PLAN.json")
    data_root, run_binding = validate_run_envelope(run_root, plan_sha)
    catalog_receipt, catalog = validate_catalog(data_root)
    matrix, matrix_columns, frames, decodes = audit_matrix(
        data_root, selections, catalog)
    prior, prior_sha = validate_prior(data_root, selections, catalog)
    definitions, policies_receipt = policy_definitions(data_root)
    online, policy_lock, online_validation = audit_online(
        data_root, selections, catalog, matrix_columns, prior, prior_sha,
        definitions,
        frames, decodes)
    inventory = audit_frame_inventory(data_root, frames, decodes)

    groups = {stage: sorted({column["source_group"] for column in selections
                             if column["stage"] == stage}) for stage in STAGES}
    report: dict[str, Any] = {
        "schema": "harness-combo-pilot-report-v1",
        "status": "complete_verified",
        "run_root": str(run_root),
        "result_root": str(data_root),
        "scope": {
            "study": "small development-only harness combination pilot",
            "columns": len(selections),
            "datasets": len({column["dataset_id"] for column in selections}),
            "dtypes": dict(Counter(column["dtype"] for column in selections)),
            "stages": dict(Counter(column["stage"] for column in selections)),
            "original_numeric_bytes_min":
                min(column["original_numeric_bytes"] for column in selections),
            "original_numeric_bytes_max":
                max(column["original_numeric_bytes"] for column in selections),
            "max_column_bytes_contract": MAX_BYTES,
            "confirm_rows": 0,
            "generalization_limit":
                "41 development columns <=8MiB; not a 135-dataset full-scale result",
        },
        "candidate_scope": {
            "total": len(catalog),
            "old_control": sum(not row["new_combination"]
                               for row in catalog.values()),
            "new_combination": sum(bool(row["new_combination"])
                                   for row in catalog.values()),
            "classification_counts":
                dict(Counter(category(row) for row in catalog.values())),
            "candidate_ids": list(catalog),
            "excluded_families": catalog_receipt["scope"]["excluded_families"],
            "full_operator_algorithm_cartesian_product_measured": False,
        },
        "baseline_semantics": {
            "fixed_zstd3": "pilot candidate zstd3",
            "fixed_lz4": "pilot candidate lz4",
            "prior_top1": (
                "discovery-only dtype-specific frozen candidate; not the "
                "historical PreparedModel"
            ),
        },
        "matrix": matrix,
        "columns": matrix_columns,
        "online": online,
        "policy_lock": policy_lock,
        "split_isolation": {
            "source_groups_by_stage": groups,
            "pairwise_disjoint": True,
            "prior_training_stage": "discovery",
            "policy_selection_stage": "tuning",
            "locked_validation_used_for_selection": False,
            "confirm_labels_used": False,
        },
        "timing_contract": {
            "matrix": "W1/R3; medians of three warmed complete-column measurements",
            "online": "W0/R1; one-shot complete-column selection and execution",
            "source_file_io_excluded": True,
            "parent_startup_excluded": True,
            "independent_verifier_startup_included": True,
            "warning": "matrix and online timings are not interchangeable",
        },
        "policy_design_provenance": {
            "fit_scope": policies_receipt.get("fit_scope"),
            "selection": policies_receipt.get("selection"),
            "budget": policies_receipt.get("budget"),
            "timing": policies_receipt.get("timing"),
            "design_history": policies_receipt.get("design_history"),
            "policy_count": len(definitions),
            "policy_ids": list(definitions),
        },
        "validation": {**inventory, **online_validation,
                       **run_binding,
                       "catalog_sha256": sha256_file(data_root / "CATALOG.json"),
                       "prior_sha256": prior_sha,
                       "source_group_stage_isolation_verified": True},
        "plan_metadata": {
            "schema": plan["schema"], "status": plan.get("status"),
            "inputs": plan.get("inputs"), "gaps": plan.get("gaps"),
        },
    }
    if reference is not None:
        report["historical_reference"] = read_reference(reference, selections)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit and report the HCB1 combination pilot")
    parser.add_argument("--root", required=True, type=Path,
                        help="completed pilot run root")
    parser.add_argument("--output", required=True, type=Path,
                        help="new/existing empty report directory")
    parser.add_argument("--reference", type=Path,
                        help="optional frozen current135 oracle JSONL; development "
                             "whitelist only, contextual fields only")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    root = arguments.root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    reference = (arguments.reference.expanduser().resolve()
                 if arguments.reference is not None else None)
    try:
        require(root.is_dir(), f"run root is not a directory: {root}")
        # Nothing is written until the complete receipt graph has validated.
        report = build_report(root, reference)
        markdown = markdown_report(report)
        atomic_write_pair(output, report, markdown)
    except ReportError as error:
        print(f"report_harness_combo_pilot_v1: ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "complete_verified",
        "report": str(output / "REPORT.json"),
        "markdown": str(output / "RESULTS.zh.md"),
        "columns": report["scope"]["columns"],
        "frames_verified": report["validation"]["complete_frames_sha256_verified"],
        "confirm_rows": 0,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
