#!/usr/bin/env python3
"""Audit whether primitive action pipelines reproduce benchmark-86 byte sizes.

The acceptance rule is intentionally strict: for every one of the 20 numeric
benchmark columns, the composed pipeline must have exactly the same compressed
byte count as the selected original codec.  For newly split native-LZ stages
we additionally require byte-for-byte payload identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPT_DIR = ROOT / "scripts"
for path in (str(BACKEND), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import primitive_operator_action_runtime as primitive  # noqa: E402
from myapp.experiment_data_policy import prepare_numeric_column  # noqa: E402


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_86_combinations_for_exact_audit",
        SCRIPT_DIR / "benchmark_86_combinations.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark_86_combinations.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()

METRO_NUMERIC_NAMES = (
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "date_time",
    "traffic_volume",
)
AIR_NUMERIC_NAMES = (
    "timestamp",
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
)

HISTORICAL_RESULTS = ROOT / "analysis_outputs/compression_86_combinations_20260826_v1"


def load_numeric_columns() -> list[Any]:
    columns: list[Any] = []
    for dataset, names in (
        ("Metro-Traffic", METRO_NUMERIC_NAMES),
        ("Air-Quality", AIR_NUMERIC_NAMES),
    ):
        frame = pd.read_csv(ROOT / "orignial_datasets" / f"{dataset}.csv", dtype=str)
        metadata = {
            row["column_name"]: row
            for row in csv.DictReader(
                (ROOT / "backend/dataset" / dataset / f"{dataset}_column_info.csv").open()
            )
        }
        for index, name in enumerate(names):
            # The normalized benchmark CSV materialized timestamp/date_time as
            # int64 even though the original source metadata calls date_time a
            # string column.
            if name in {"timestamp", "date_time"}:
                logical_type, max_point = "integer", 0
            else:
                logical_type = metadata[name]["logical_type"]
                max_point = int(metadata[name]["max_point_number"])
            values = prepare_numeric_column(frame[name].tolist(), logical_type).values
            columns.append(
                benchmark.Column(dataset, index, name, logical_type, max_point, values)
            )
    return columns


def historical_specialist_rows(column: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    path = (
        HISTORICAL_RESULTS
        / f"{column.dataset}_{column.index}"
        / f"{column.dataset}__{column.index:03d}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["raw_sha256"] != hashlib.sha256(column.raw).hexdigest():
        raise AssertionError(f"historical raw hash mismatch: {path}")
    materialized = {row["method"]: row for row in payload["specialists_materialized"]}
    excluded = {row["method"]: row["reason"] for row in payload["specialist_exclusions"]}
    return materialized, excluded


def select_global_best_levels(columns: list[Any]) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Choose one level per method by minimum total bytes over all 20 columns."""

    totals: dict[str, dict[str, list[int]]] = {
        method: defaultdict(lambda: [0, 0]) for method in benchmark.GENERAL_METHODS
    }
    errors: dict[str, dict[str, list[str]]] = {
        method: defaultdict(list) for method in benchmark.GENERAL_METHODS
    }
    for column_index, column in enumerate(columns, 1):
        started = time.time()
        for method in benchmark.GENERAL_METHODS:
            _best, attempts = benchmark.best_generic(column.raw, method)
            for attempt in attempts:
                key = attempt["key"]
                if attempt.get("ok"):
                    totals[method][key][0] += int(attempt["bytes"])
                    totals[method][key][1] += 1
                else:
                    errors[method][key].append(
                        f"{column.dataset}:{column.index}:{attempt.get('error', 'roundtrip failed')}"
                    )
        print(
            f"LEVEL_SELECTION {column_index}/{len(columns)} "
            f"{column.dataset}:{column.index} sec={time.time() - started:.2f}",
            flush=True,
        )

    selected: dict[str, str | None] = {}
    evidence: dict[str, Any] = {}
    for method in benchmark.GENERAL_METHODS:
        candidates = [
            (values[0], key)
            for key, values in totals[method].items()
            if values[1] == len(columns)
        ]
        if candidates:
            total_bytes, key = min(candidates)
            selected[method] = key
            evidence[method] = {
                "selected_key": key,
                "total_compressed_bytes_20cols": total_bytes,
                "verified_columns": len(columns),
                "selection_rule": "minimum sum of compressed bytes; key breaks ties",
            }
            continue
        fallback = [
            (-values[1], values[0], key) for key, values in totals[method].items()
        ]
        selected[method] = None
        evidence[method] = {
            "selected_key": None,
            "reason": "no single level round-tripped on all 20 columns",
            "best_partial": min(fallback)[2] if fallback else None,
            "errors": {key: rows[:3] for key, rows in errors[method].items() if rows},
        }
    return selected, evidence


def _logical_variant(column: Any) -> str:
    return "int" if column.logical_type == "integer" else "float"


def _primitive_size(column: Any, method: str, cache: dict[tuple[Any, ...], int]) -> int:
    action_names = primitive.concrete_action_pipeline(
        method, logical_type=_logical_variant(column)
    )
    key = (column.dataset, column.index, action_names)
    if key not in cache:
        values = np.frombuffer(column.raw, dtype="<i8")
        cache[key] = len(primitive.encode_operator_pipeline(values, action_names))
    return cache[key]


def _exact_native_action_names(algorithm_key: str) -> tuple[str, ...]:
    return tuple(
        action.name for action in primitive.exact_native_lz_action_pipeline(algorithm_key)
    )


def _exact_atomic_payload(column: Any, method: str) -> bytes:
    logical_type = _logical_variant(column) if method in {"RLE", "Dict."} else "bytes"
    max_point = column.max_point if method == "RLE" and logical_type == "float" else None
    action_names = primitive.exact_atomic_action_pipeline(
        method, logical_type, max_point=max_point
    )
    return primitive.encode_registered_exact_action_pipeline(
        column.raw,
        action_names,
        logical_type=logical_type,
        max_point=max_point,
    )


def run_audit(columns: list[Any], selected: dict[str, str | None]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    primitive_size_cache: dict[tuple[Any, ...], int] = {}

    for method in benchmark.SPECIALIST_METHODS:
        comparisons: list[dict[str, Any]] = []
        for column in columns:
            materialized, excluded = historical_specialist_rows(column)
            original = materialized.get(method)
            if original is None:
                comparisons.append(
                    {
                        "column": f"{column.dataset}:{column.index}",
                        "status": "not_materialized",
                        "reason": excluded.get(method, "missing historical row"),
                    }
                )
                continue
            comparison: dict[str, Any] = {
                "column": f"{column.dataset}:{column.index}",
                "status": "compared",
                "original_bytes": int(original["bytes"]),
            }
            try:
                primitive_bytes = _primitive_size(column, method, primitive_size_cache)
                comparison.update(
                    {
                        "primitive_29_bytes": primitive_bytes,
                        "primitive_ratio_equal": primitive_bytes == int(original["bytes"]),
                    }
                )
            except Exception as exc:
                comparison.update(
                    {
                        "primitive_29_bytes": None,
                        "primitive_ratio_equal": False,
                        "primitive_29_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if method in {"FPCmp-R", "FPCmp-S"}:
                key = "fpcompress_dpratio" if method == "FPCmp-R" else "fpcompress_dpspeed"
                original_payload = benchmark.compress_native_lz_family(column.raw, key)
                split_payload = primitive.encode_registered_exact_action_pipeline(
                    column.raw, _exact_native_action_names(key)
                )
                comparison.update(
                    {
                        "split_bytes": len(split_payload),
                        "split_ratio_equal": len(split_payload) == int(original["bytes"]),
                        "split_payload_identical": split_payload == original_payload,
                    }
                )
            elif method in {"RLE", "Dict."}:
                original_artifact, reason = benchmark.specialist_artifact(column, method)
                if original_artifact is None:
                    raise AssertionError(f"fresh {method} materialization failed: {reason}")
                split_payload = _exact_atomic_payload(column, method)
                comparison.update(
                    {
                        "split_bytes": len(split_payload),
                        "split_ratio_equal": len(split_payload) == int(original["bytes"]),
                        "split_payload_identical": split_payload == original_artifact.payload,
                    }
                )
            comparisons.append(comparison)

        compared = [row for row in comparisons if row["status"] == "compared"]
        split_rows = [row for row in compared if "split_payload_identical" in row]
        rows.append(
            {
                "method": method,
                "kind": "specialist",
                "selected_key": None,
                "original_columns": len(compared),
                "primitive_29_equal_columns": sum(row["primitive_ratio_equal"] for row in compared),
                "primitive_29_complete": len(compared) == len(columns)
                and all(row["primitive_ratio_equal"] for row in compared),
                "split_stage_complete": len(split_rows) == len(columns)
                and all(row["split_ratio_equal"] for row in split_rows),
                "split_payload_identity_complete": len(split_rows) == len(columns)
                and all(row["split_payload_identical"] for row in split_rows),
                "comparisons": comparisons,
            }
        )

    for method in benchmark.GENERAL_METHODS:
        key = selected[method]
        comparisons = []
        for column in columns:
            if key is None:
                comparisons.append(
                    {
                        "column": f"{column.dataset}:{column.index}",
                        "status": "no_global_level",
                    }
                )
                continue
            original_payload, decode, implementation = benchmark._compress_generic(column.raw, key)
            if decode(original_payload) != column.raw:
                raise AssertionError(f"selected original failed round trip: {method}:{key}")
            comparison = {
                "column": f"{column.dataset}:{column.index}",
                "status": "compared",
                "original_bytes": len(original_payload),
                "original_implementation": implementation,
            }
            try:
                primitive_bytes = _primitive_size(column, method, primitive_size_cache)
                comparison.update(
                    {
                        "primitive_29_bytes": primitive_bytes,
                        "primitive_ratio_equal": primitive_bytes == len(original_payload),
                    }
                )
            except Exception as exc:
                comparison.update(
                    {
                        "primitive_29_bytes": None,
                        "primitive_ratio_equal": False,
                        "primitive_29_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if implementation.startswith("project_native_reference:"):
                split_payload = primitive.encode_registered_exact_action_pipeline(
                    column.raw, _exact_native_action_names(key)
                )
                comparison.update(
                    {
                        "split_bytes": len(split_payload),
                        "split_ratio_equal": len(split_payload) == len(original_payload),
                        "split_payload_identical": split_payload == original_payload,
                    }
                )
            elif method in {"Huffman", "Arithmetic", "ANS"}:
                split_payload = _exact_atomic_payload(column, method)
                comparison.update(
                    {
                        "split_bytes": len(split_payload),
                        "split_ratio_equal": len(split_payload) == len(original_payload),
                        "split_payload_identical": split_payload == original_payload,
                    }
                )
            comparisons.append(comparison)
        compared = [row for row in comparisons if row["status"] == "compared"]
        split_rows = [row for row in compared if "split_payload_identical" in row]
        rows.append(
            {
                "method": method,
                "kind": "general",
                "selected_key": key,
                "original_columns": len(compared),
                "primitive_29_equal_columns": sum(row["primitive_ratio_equal"] for row in compared),
                "primitive_29_complete": len(compared) == len(columns)
                and all(row["primitive_ratio_equal"] for row in compared),
                "split_stage_complete": len(split_rows) == len(columns)
                and all(row["split_ratio_equal"] for row in split_rows),
                "split_payload_identity_complete": len(split_rows) == len(columns)
                and all(row["split_payload_identical"] for row in split_rows),
                "comparisons": comparisons,
            }
        )
        print(f"AUDIT_METHOD {len(rows)}/86 {method}", flush=True)

    selected_native_keys = [
        row["selected_key"]
        for row in rows
        if row["kind"] == "general"
        and row["method"] in primitive.EXACT_NATIVE_GENERAL_METHODS
        and row["split_payload_identity_complete"]
    ] + ["fpcompress_dpratio", "fpcompress_dpspeed"]
    selected_native_action_names = {
        action.name
        for key in selected_native_keys
        for action in primitive.exact_native_lz_action_pipeline(key)
    }
    exact_native_actions = tuple(
        action
        for action in primitive.EXACT_ACTIONS
        if action.name in selected_native_action_names
    )
    exact_atomic_actions = tuple(
        action
        for action in primitive.EXACT_ACTIONS
        if action.name not in selected_native_action_names
    )
    exact_actions = exact_native_actions + exact_atomic_actions
    current_complete = [row["method"] for row in rows if row["primitive_29_complete"]]
    split_complete = [row["method"] for row in rows if row["split_payload_identity_complete"]]
    return {
        "schema": "benchmark-86-exact-action-coverage-v1",
        "acceptance_rule": (
            "same compressed byte count on all 20 columns; newly split native-LZ "
            "pipelines must also be byte-for-byte identical"
        ),
        "columns": len(columns),
        "methods": len(rows),
        "current_29_complete_methods": current_complete,
        "current_29_complete_count": len(current_complete),
        "new_split_complete_methods": split_complete,
        "new_split_complete_count": len(split_complete),
        "new_exact_native_lz_actions": [action.__dict__ for action in exact_native_actions],
        "new_exact_native_lz_action_count": len(exact_native_actions),
        "new_exact_atomic_actions": [action.__dict__ for action in exact_atomic_actions],
        "new_exact_atomic_action_count": len(exact_atomic_actions),
        "new_exact_action_count": len(exact_actions),
        "final_action_count_reference_plus_exact_variants": len(primitive.POLICY_ACTIONS),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=Path)
    parser.add_argument("--reuse-levels", action="store_true")
    args = parser.parse_args()
    columns = load_numeric_columns()
    levels_path = args.levels or args.output.with_name("selected_levels.json")
    if args.reuse_levels:
        levels_payload = json.loads(levels_path.read_text(encoding="utf-8"))
        selected = levels_payload["selected"]
    else:
        selected, evidence = select_global_best_levels(columns)
        levels_payload = {
            "schema": "benchmark-86-global-level-selection-v1",
            "columns": len(columns),
            "selection_rule": "minimum total compressed bytes over the fixed 20 columns",
            "selected": selected,
            "evidence": evidence,
        }
        levels_path.parent.mkdir(parents=True, exist_ok=True)
        levels_path.write_text(
            json.dumps(levels_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    result = run_audit(columns, selected)
    result["level_selection"] = levels_payload
    provenance_paths = {
        "audit_script": Path(__file__).resolve(),
        "action_runtime": SCRIPT_DIR / "primitive_operator_action_runtime.py",
        "ans_runtime": BACKEND / "myapp/entropy_cpp.py",
        "ans_cpp_kernel": BACKEND.parent / "cpp/src/ans_rans.cpp",
        "native_lz_runtime": BACKEND / "myapp/native_lz_family.py",
        "benchmark_driver": SCRIPT_DIR / "benchmark_86_combinations.py",
        "historical_aggregate": HISTORICAL_RESULTS / "aggregate_manifest.json",
        "selected_levels": levels_path.resolve(),
    }
    result["provenance"] = {
        "code_and_input_sha256": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in provenance_paths.items()
        },
        "primitive_registry_sha256": primitive.registry()["registry_sha256"],
        "policy_action_ids": [action.action_id for action in primitive.POLICY_ACTIONS],
        "stop_action_id": primitive.STOP_ACTION_ID,
        "raw_columns": [
            {
                "dataset": column.dataset,
                "column_index": column.index,
                "column_name": column.name,
                "logical_type": column.logical_type,
                "rows": len(column.values),
                "raw_bytes": len(column.raw),
                "raw_sha256": hashlib.sha256(column.raw).hexdigest(),
            }
            for column in columns
        ],
        "executor": "encode_registered_exact_action_pipeline(action_names)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "methods",
                    "current_29_complete_count",
                    "new_split_complete_count",
                    "new_exact_native_lz_action_count",
                    "new_exact_atomic_action_count",
                    "new_exact_action_count",
                    "final_action_count_reference_plus_exact_variants",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
