#!/usr/bin/env python3
"""Prepare a deterministic, development-only harness combination pilot plan.

This program reads only frozen column metadata and the source-group split.  It
does not open source manifests, source payloads, feature files, cost matrices,
or oracle-label files, and it performs no numerical encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "harness-combo-pilot-plan-v1"
DTYPES = ("float32", "float64", "int64")
BUCKETS = (
    ("short", 0.10),
    ("medium", 0.50),
    ("long", 0.90),
)
MIN_BYTES = 16 * 1024
MAX_BYTES = 8 * 1024 * 1024
MIN_VALID_VALUES = 1024
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SELECTIONS_PER_CELL = 3
EXPECTED_SELECTION_COUNT = len(EXPECTED_FOLDS) * len(DTYPES) * EXPECTED_SELECTIONS_PER_CELL


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def integer(value: Any, field: str, context: str) -> int:
    require(type(value) is int and value >= 0, f"{context}: invalid {field}")
    return value


def text(value: Any, field: str, context: str) -> str:
    require(isinstance(value, str) and value and "\x00" not in value,
            f"{context}: invalid {field}")
    return value


def quantile_index(count: int, quantile: float) -> int:
    require(count > 0, "cannot take a quantile of an empty sequence")
    # Deterministic nearest-rank-on-index with half values rounded upward.
    return min(count - 1, int(math.floor(quantile * (count - 1) + 0.5)))


def column_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["dataset_id"], row["column_id"]


def stable_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["original_numeric_bytes"],
        row["source_group"],
        row["dataset_id"],
        row["column_id"],
    )


def source_group_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["source_group"], []).append(row)
    result = []
    for group in sorted(grouped):
        members = sorted(grouped[group], key=stable_row_key)
        target = members[quantile_index(len(members), 0.50)]["original_numeric_bytes"]
        result.append(min(
            members,
            key=lambda row: (
                abs(row["original_numeric_bytes"] - target),
                stable_row_key(row),
            ),
        ))
    return sorted(result, key=stable_row_key)


def candidate_list(
    rows: list[dict[str, Any]], target_bytes: int, limit: int = 32
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            abs(row["original_numeric_bytes"] - target_bytes),
            stable_row_key(row),
        )

    ordered = sorted(rows, key=key)
    chosen: dict[tuple[str, str], dict[str, Any]] = {
        column_key(row): row for row in ordered[:limit]
    }
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["source_group"], []).append(row)
    for group in sorted(by_group):
        row = min(by_group[group], key=key)
        chosen[column_key(row)] = row
    return sorted(chosen.values(), key=key)


def bucket_subset(column_count: int) -> tuple[tuple[str, float], ...]:
    if column_count >= 3:
        return BUCKETS
    if column_count == 2:
        return (BUCKETS[0], BUCKETS[2])
    if column_count == 1:
        return (BUCKETS[1],)
    return ()


def select_cell(
    rows: list[dict[str, Any]], fold_sources: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    representatives = source_group_representatives(rows) if rows else []
    targets = {
        name: representatives[quantile_index(len(representatives), quantile)][
            "original_numeric_bytes"
        ]
        for name, quantile in BUCKETS
    } if representatives else {}
    selected_buckets = bucket_subset(len(rows))
    if not selected_buckets:
        return [], {
            "eligible_columns": 0,
            "eligible_source_groups": 0,
            "source_group_representatives": 0,
            "quantile_targets_bytes": {},
            "selected_buckets": [],
            "missing_buckets": [name for name, _ in BUCKETS],
            "source_group_reuse_within_cell": 0,
            "source_group_reuse_across_fold_dtypes": 0,
        }

    pools = [candidate_list(rows, targets[name]) for name, _ in selected_buckets]
    best: tuple[tuple[Any, ...], tuple[dict[str, Any], ...]] | None = None
    for assignment in itertools.product(*pools):
        keys = [column_key(row) for row in assignment]
        if len(set(keys)) != len(keys):
            continue
        groups = [row["source_group"] for row in assignment]
        within_reuse = len(groups) - len(set(groups))
        cross_reuse = sum(group in fold_sources for group in groups)
        distances = [
            abs(row["original_numeric_bytes"] - targets[bucket[0]]) /
            max(1, targets[bucket[0]])
            for bucket, row in zip(selected_buckets, assignment)
        ]
        score = (
            within_reuse,
            cross_reuse,
            sum(distances),
            max(distances, default=0.0),
            tuple(stable_row_key(row) for row in assignment),
        )
        if best is None or score < best[0]:
            best = score, assignment
    require(best is not None, "no distinct-column bucket assignment exists")

    output = []
    for (bucket, quantile), row in zip(selected_buckets, best[1]):
        target = targets[bucket]
        output.append({
            **row,
            "size_bucket": bucket,
            "target_quantile": quantile,
            "target_original_numeric_bytes": target,
            "absolute_target_distance_bytes": abs(
                row["original_numeric_bytes"] - target
            ),
        })
    groups = [row["source_group"] for row in output]
    missing = [name for name, _ in BUCKETS if name not in {row["size_bucket"] for row in output}]
    diagnostics = {
        "eligible_columns": len(rows),
        "eligible_source_groups": len({row["source_group"] for row in rows}),
        "source_group_representatives": len(representatives),
        "quantile_targets_bytes": targets,
        "selected_buckets": [row["size_bucket"] for row in output],
        "missing_buckets": missing,
        "source_group_reuse_within_cell": len(groups) - len(set(groups)),
        "source_group_reuse_across_fold_dtypes": sum(
            group in fold_sources for group in groups
        ),
    }
    return output, diagnostics


def stage_for_fold(fold: int) -> str:
    if fold <= 2:
        return "discovery"
    if fold == 3:
        return "tuning"
    return "locked_validation"


def atomic_write_new(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        position = 0
        while position < len(encoded):
            position += os.write(descriptor, encoded[position:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_plan(columns_path: Path, split_path: Path) -> dict[str, Any]:
    columns = read_object(columns_path)
    split = read_object(split_path)
    require(columns.get("schema") == "operator-prepared135-source-columns-v1",
            "unexpected COLUMNS schema")
    require(split.get("schema") == "operator-selection-source-group-split-v1",
            "unexpected split schema")
    require(isinstance(columns.get("columns"), list), "COLUMNS.columns must be an array")
    assignments = split.get("dataset_assignments")
    require(isinstance(assignments, dict), "split.dataset_assignments must be an object")

    eligible: dict[tuple[int, str], list[dict[str, Any]]] = {
        (fold, dtype): [] for fold in EXPECTED_FOLDS for dtype in DTYPES
    }
    rejected = {
        "confirm_split": 0,
        "unsupported_dtype": 0,
        "below_minimum_bytes": 0,
        "above_maximum_bytes": 0,
        "below_minimum_valid_values": 0,
    }
    development_rows_seen = 0
    confirm_rows_seen = 0
    seen: set[tuple[str, str]] = set()
    for index, source in enumerate(columns["columns"]):
        context = f"COLUMNS row {index}"
        require(isinstance(source, dict), f"{context}: expected object")
        dataset_id = text(source.get("dataset_id"), "dataset_id", context)
        column_id = text(source.get("column_id"), "column_id", context)
        key = dataset_id, column_id
        require(key not in seen, f"{context}: duplicate dataset/column key")
        seen.add(key)
        require(dataset_id in assignments, f"{context}: dataset absent from split")
        assignment = assignments[dataset_id]
        require(isinstance(assignment, dict), f"{context}: invalid split assignment")
        split_name = text(assignment.get("split"), "split", context)
        require(split_name in {"development", "confirm"}, f"{context}: unknown split")
        if split_name == "confirm":
            confirm_rows_seen += 1
            rejected["confirm_split"] += 1
            continue
        development_rows_seen += 1
        fold = integer(assignment.get("dev_fold"), "dev_fold", context)
        require(fold in EXPECTED_FOLDS, f"{context}: dev_fold outside [0,4]")
        dtype = text(source.get("dtype"), "dtype", context)
        original_bytes = integer(
            source.get("original_numeric_bytes"), "original_numeric_bytes", context
        )
        valid_values = integer(
            source.get("valid_value_count"), "valid_value_count", context
        )
        manifest_path = text(source.get("manifest_path"), "manifest_path", context)
        manifest_sha = text(source.get("manifest_sha256"), "manifest_sha256", context)
        require(len(manifest_sha) == 64 and all(c in "0123456789abcdef" for c in manifest_sha),
                f"{context}: invalid manifest SHA256")
        if dtype not in DTYPES:
            rejected["unsupported_dtype"] += 1
            continue
        if original_bytes < MIN_BYTES:
            rejected["below_minimum_bytes"] += 1
            continue
        if original_bytes > MAX_BYTES:
            rejected["above_maximum_bytes"] += 1
            continue
        if valid_values < MIN_VALID_VALUES:
            rejected["below_minimum_valid_values"] += 1
            continue
        source_group = text(
            assignment.get("source_identity_sha256"), "source_identity_sha256", context
        )
        require(len(source_group) == 64 and all(c in "0123456789abcdef" for c in source_group),
                f"{context}: invalid source-group SHA256")
        eligible[(fold, dtype)].append({
            "dataset_id": dataset_id,
            "column_id": column_id,
            "column": text(source.get("column"), "column", context),
            "dtype": dtype,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha,
            "original_numeric_bytes": original_bytes,
            "valid_value_count": valid_values,
            "source_group": source_group,
            "dev_fold": fold,
            "split": "development",
            "stage": stage_for_fold(fold),
        })

    selections: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_sources: set[str] = set()
        # Protect scarce dtype/source cells first; ties use the canonical dtype order.
        dtype_order = sorted(
            DTYPES,
            key=lambda dtype: (
                len({row["source_group"] for row in eligible[(fold, dtype)]}),
                len(eligible[(fold, dtype)]),
                DTYPES.index(dtype),
            ),
        )
        fold_results: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        for dtype in dtype_order:
            rows = sorted(eligible[(fold, dtype)], key=stable_row_key)
            chosen, diagnostics = select_cell(rows, fold_sources)
            fold_results[dtype] = chosen, diagnostics
            fold_sources.update(row["source_group"] for row in chosen)
        for dtype in DTYPES:
            chosen, diagnostics = fold_results[dtype]
            for row in chosen:
                row["selection_index"] = len(selections)
                selections.append(row)
            cell = {
                "dev_fold": fold,
                "stage": stage_for_fold(fold),
                "dtype": dtype,
                **diagnostics,
                "selected_count": len(chosen),
                "expected_selected_count": EXPECTED_SELECTIONS_PER_CELL,
            }
            cells.append(cell)
            if diagnostics["source_group_reuse_within_cell"]:
                gaps.append({
                    "kind": "source_group_reuse_within_cell",
                    "dev_fold": fold,
                    "dtype": dtype,
                    "count": diagnostics["source_group_reuse_within_cell"],
                    "reason": "fewer than three eligible source groups; distinct columns retained",
                })
            if diagnostics["source_group_reuse_across_fold_dtypes"]:
                gaps.append({
                    "kind": "source_group_reuse_across_fold_dtypes",
                    "dev_fold": fold,
                    "dtype": dtype,
                    "count": diagnostics["source_group_reuse_across_fold_dtypes"],
                    "reason": "no equally sized assignment with a new eligible source group",
                })
            if diagnostics["missing_buckets"]:
                gaps.append({
                    "kind": "missing_size_buckets",
                    "dev_fold": fold,
                    "dtype": dtype,
                    "missing_buckets": diagnostics["missing_buckets"],
                    "eligible_columns": diagnostics["eligible_columns"],
                    "reason": "cell has fewer than three eligible distinct columns; confirm was not used",
                })

    require(len({column_key(row) for row in selections}) == len(selections),
            "a column was selected more than once")
    require(all(row["split"] == "development" for row in selections),
            "non-development row reached output")
    selected_by_stage = {
        stage: sum(row["stage"] == stage for row in selections)
        for stage in ("discovery", "tuning", "locked_validation")
    }
    selected_by_dtype = {
        dtype: sum(row["dtype"] == dtype for row in selections) for dtype in DTYPES
    }
    selected_by_fold = {
        str(fold): sum(row["dev_fold"] == fold for row in selections)
        for fold in EXPECTED_FOLDS
    }
    return {
        "schema": SCHEMA,
        "status": "ready" if len(selections) == EXPECTED_SELECTION_COUNT else "ready_with_gaps",
        "scope": "frozen development metadata only",
        "inputs": {
            "columns_path": str(columns_path),
            "columns_sha256": sha256(columns_path),
            "columns_schema": columns["schema"],
            "columns_count": len(columns["columns"]),
            "columns_oracle_reference_sha256_binding": columns.get(
                "oracle_reference_jsonl_sha256"
            ),
            "split_path": str(split_path),
            "split_sha256": sha256(split_path),
            "split_schema": split["schema"],
        },
        "io_audit": {
            "oracle_labels_read": False,
            "cost_or_feature_files_read": False,
            "source_manifests_opened": 0,
            "source_payloads_opened": 0,
            "numerical_encoding_performed": False,
            "selected_confirm_rows": 0,
            "confirm_rows": 0,
            "input_confirm_metadata_rows_seen": confirm_rows_seen,
            "input_development_metadata_rows_seen": development_rows_seen,
        },
        "selection_rule": {
            "allowed_split": "development",
            "dev_folds": list(EXPECTED_FOLDS),
            "dtype_order_in_output": list(DTYPES),
            "bytes_inclusive": [MIN_BYTES, MAX_BYTES],
            "minimum_valid_value_count": MIN_VALID_VALUES,
            "size_buckets": [
                {"name": name, "quantile": quantile} for name, quantile in BUCKETS
            ],
            "quantile_rule": "nearest index floor(q*(n-1)+0.5) over one deterministic median-size representative per source group",
            "assignment_rule": "distinct columns; minimize within-cell source reuse, then same-fold cross-dtype source reuse, then normalized quantile distance; scarce source-group cells processed first",
            "shortage_rule": "two eligible columns map to short/long; one maps to medium; confirm never fills a gap",
            "tie_break": "original_numeric_bytes, source_group, dataset_id, column_id",
            "compression_results_used": False,
        },
        "expected_selection_count": EXPECTED_SELECTION_COUNT,
        "selected_count": len(selections),
        "selected_by_stage": selected_by_stage,
        "selected_by_fold": selected_by_fold,
        "selected_by_dtype": selected_by_dtype,
        "eligible_filter_rejections": rejected,
        "cells": cells,
        "gaps": gaps,
        "selections": selections,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    columns = args.columns.resolve(strict=True)
    split = args.split.resolve(strict=True)
    output = args.output.resolve()
    plan = build_plan(columns, split)
    atomic_write_new(output, plan)
    print(json.dumps({
        "status": plan["status"],
        "output": str(output),
        "selected_count": plan["selected_count"],
        "expected_selection_count": plan["expected_selection_count"],
        "gap_count": len(plan["gaps"]),
        "confirm_rows": plan["io_audit"]["confirm_rows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
