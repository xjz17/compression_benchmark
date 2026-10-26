#!/usr/bin/env python3
"""Audit current policy-action chains against the versioned benchmark-86 adapters.

Every stage is called and inverted through run_action; canonical C++ encoders
are not substituted silently for the selected benchmark contract in this audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_86_composability_realdata as external  # noqa: E402
import benchmark_86_combinations as benchmark  # noqa: E402
import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402


def select_actions(method: str, column: Any, pipeline_map: str = "legacy") -> tuple[tuple[str, ...], str]:
    logical = "int" if column.logical_type == "integer" else "float"
    pipelines = (primitive.benchmark_86_canonical_action_pipelines()
                 if pipeline_map == "cpp-deduplicated"
                 else primitive.benchmark_86_exact_action_pipelines())
    mapping_source = "current_public_method_mapping"
    legacy_mapper = getattr(primitive, "benchmark_86_legacy_action_pipelines", None)
    if pipeline_map == "legacy" and legacy_mapper is not None:
        legacy_pipelines = legacy_mapper()
        if method in legacy_pipelines:
            pipelines[method] = legacy_pipelines[method]
            mapping_source = "benchmark_86_legacy_action_pipelines"
    # The executable PFor/VariableByte stages exist even when the older public
    # method map still contains the generic reference/proxy path.
    if method == "FastPFor" and pipeline_map != "cpp-deduplicated" and mapping_source != "benchmark_86_legacy_action_pipelines":
        return (
            primitive.FASTPFOR_INT_ACTIONS if logical == "int"
            else primitive.FASTPFOR_FLOAT_ACTIONS
        ), "explicit_existing_fastpfor_stages"
    pipeline = pipelines[method]
    if isinstance(pipeline, dict):
        if method in {"Blosc", "Blosc2"}:
            key = "bytes"  # Unchanged benchmark adapters use typesize=1.
        elif logical == "float" and "float_max_point_0_to_18" in pipeline:
            key = "float_max_point_0_to_18"
        else:
            key = logical
        if key not in pipeline:
            return (), f"no_current_action_pipeline_for_{logical}"
        pipeline = pipeline[key]
    return tuple(pipeline), mapping_source


def diagnose_baseline(column: Any, method: str) -> dict[str, Any]:
    """Inspect rejected legacy payloads without changing their acceptance rule."""
    if method not in {"Chimp", "Chimp128", "Gorilla", "2DIFF", "RLE", "Camel", "Elf", "Elf+"}:
        return {}
    try:
        artifact = (
            benchmark._camel_artifact(column) if method == "Camel"
            else benchmark._elf_artifact(column, method) if method in {"Elf", "Elf+"}
            else benchmark._direct_ts_artifact(column, method)
        )
        recovered = artifact.decode(artifact.payload)
        mismatches = []
        for index, (expected, actual) in enumerate(zip(column.values, recovered)):
            equal = (
                int(expected) == int(actual) if column.logical_type == "integer"
                else struct.pack("<d", float(expected)) == struct.pack("<d", float(actual))
            )
            if not equal:
                mismatches.append({"index": index, "expected": repr(expected), "decoded": repr(actual)})
        return {
            "expected_count": len(column.values), "decoded_count": len(recovered),
            "nonfinite_indexes": [i for i, value in enumerate(column.values) if not math.isfinite(value)],
            "mismatching_values": len(mismatches), "first_mismatches": mismatches[:3],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def original(column: Any, method: str) -> dict[str, Any]:
    if method in benchmark.IGNORED_SPECIALIST_METHODS:
        return {"status": "EXCLUDED_BY_USER", "error": benchmark.IGNORED_SPECIALIST_METHODS[method]}
    if method in benchmark.SPECIALIST_METHODS:
        artifact, reason = benchmark.specialist_artifact(column, method)
        if artifact is not None:
            return {"status": "OK", "artifact": artifact, "payload": artifact.payload,
                    "decode": artifact.decode, "implementation": artifact.implementation}
        if "float32-only" in reason or "accepts floating-point" in reason:
            status = "UNSUPPORTED_INPUT_TYPE"
        elif "exact decimal scaling unavailable" in reason or "cannot convert float NaN to integer" in reason:
            status = "UNSUPPORTED_INPUT_VALUES"
        elif "no decoder" in reason or "no exact float64 reconstruction" in reason:
            status = "NO_LOSSLESS_BASELINE"
        elif "roundtrip mismatch" in reason:
            status = "BASELINE_ROUNDTRIP_FAILED"
        else:
            status = "BASELINE_UNAVAILABLE"
        return {"status": status, "error": reason,
                "diagnostic": diagnose_baseline(column, method)
                if status in {"BASELINE_ROUNDTRIP_FAILED", "BASELINE_UNAVAILABLE"} else {}}
    key = primitive.BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS[method]
    # No level was frozen for these two because the package was absent. Their
    # benchmark adapter nevertheless has a single explicitly specified variant.
    if key is None and method in {"Blosc", "Blosc2"}:
        key = "blosc_blosclz" if method == "Blosc" else "blosc2_blosclz"
    try:
        if key is None:
            raise ValueError("no frozen benchmark key")
        payload, decode, implementation = benchmark._compress_generic(column.raw, key)
        return {"status": "OK" if decode(payload) == column.raw else "BASELINE_ROUNDTRIP_FAILED",
                "payload": payload, "decode": decode, "implementation": implementation,
                "key": key}
    except Exception as exc:
        return {"status": "BASELINE_UNAVAILABLE", "key": key,
                "error": f"{type(exc).__name__}: {exc}"}


def compare(column: Any, method: str, policy_names: set[str], pipeline_map: str = "legacy") -> dict[str, Any]:
    names, mapping = select_actions(method, column, pipeline_map)
    baseline = original(column, method)
    row: dict[str, Any] = {
        "method": method, "dataset": column.dataset, "column": column.name,
        "logical_type": column.logical_type, "rows": len(column.values),
        "raw_bytes": len(column.raw), "raw_sha256": hashlib.sha256(column.raw).hexdigest(),
        "mapping_source": mapping, "actions": "|".join(names), "action_count": len(names),
        "all_actions_in_policy": bool(names) and all(name in policy_names for name in names),
        "non_policy_actions": "|".join(name for name in names if name not in policy_names),
        "original_status": baseline["status"], "original_key": baseline.get("key", ""),
        "original_implementation": baseline.get("implementation", ""),
        "original_error": baseline.get("error", ""),
        "original_diagnostic": json.dumps(baseline.get("diagnostic", {}), ensure_ascii=False),
        "original_bytes": len(baseline["payload"]) if "payload" in baseline else None,
        "action_status": "NOT_RUN", "action_error": "", "failed_action": "",
        "completed_forward_steps": 0, "completed_inverse_steps": 0,
        "per_stage_inverse_exact": None, "full_inverse_exact": None,
        "compressed_bytes": None, "size_delta": None, "ratio_equal": None,
        "payload_identical": None, "original_decoder_accepts_action_wire": None,
        "cross_decode_error": "", "status": "NOT_CHECKED",
    }
    if baseline["status"] == "EXCLUDED_BY_USER":
        row["status"] = "EXCLUDED_BY_USER"
        row["action_status"] = "SKIPPED"
        return row
    if not names:
        row["action_status"] = "UNSUPPORTED_INPUT_TYPE"
    elif not row["all_actions_in_policy"]:
        row["action_status"] = "NOT_A_POLICY_CHAIN"
    else:
        kind = "int64" if column.logical_type == "integer" else "float64"
        accepted = unified.describe_action(names[0]).accepted_input_kinds
        if kind not in accepted:
            kind = "bytes" if "bytes" in accepted else kind
        source = unified.ActionData(kind, column.raw, len(column.raw) if kind == "bytes" else len(column.values))
        max_point = column.max_point if column.logical_type == "float" else None
        current = source
        action = ""
        try:
            single_inverse_ok = True
            for action in names:
                previous = current
                current = unified.run_action(action, current, max_point=max_point)
                row["completed_forward_steps"] += 1
                inverse = unified.run_action(action, current, reverse=True, max_point=max_point)
                single_inverse_ok = single_inverse_ok and inverse == previous
            wire = unified._unpack_result(current)[1]
            row["compressed_bytes"] = len(wire)
            row["per_stage_inverse_exact"] = single_inverse_ok
            for action in reversed(names):
                current = unified.run_action(action, current, reverse=True, max_point=max_point)
                row["completed_inverse_steps"] += 1
            row["full_inverse_exact"] = current == source
            row["action_status"] = "OK" if single_inverse_ok and current == source else "ROUNDTRIP_FAILED"
            if baseline["status"] == "OK":
                row["size_delta"] = len(wire) - len(baseline["payload"])
                row["ratio_equal"] = row["size_delta"] == 0
                row["payload_identical"] = wire == baseline["payload"]
                try:
                    recovered = baseline["decode"](wire)
                    row["original_decoder_accepts_action_wire"] = (
                        benchmark._values_equal(column, recovered)
                        if method in benchmark.SPECIALIST_METHODS else recovered == column.raw
                    )
                except Exception as exc:
                    row["original_decoder_accepts_action_wire"] = False
                    row["cross_decode_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            row["action_status"] = "ERROR"
            row["failed_action"] = action
            row["action_error"] = f"{type(exc).__name__}: {exc}"
    if baseline["status"] != "OK":
        row["status"] = baseline["status"]
    elif row["action_status"] != "OK":
        row["status"] = "ACTION_CHAIN_FAILED"
    elif row["payload_identical"]:
        row["status"] = "EXACT_BYTE_AND_RATIO_PARITY"
    elif row["ratio_equal"]:
        row["status"] = "RATIO_ONLY_PARITY"
    else:
        row["status"] = "COMPRESSION_RATIO_MISMATCH"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="*", help="Optional subset; omitted audits all 86")
    parser.add_argument("--pipeline-map", choices=("legacy", "canonical", "cpp-deduplicated"), default="legacy",
                        help="Prefer registered legacy-compatible chains when available")
    args = parser.parse_args()
    methods = (*benchmark.SPECIALIST_METHODS, *benchmark.GENERAL_METHODS)
    assert len(methods) == len(set(methods)) == 86
    if args.methods:
        unknown = set(args.methods) - set(methods)
        if unknown:
            parser.error(f"unknown methods: {sorted(unknown)}")
        methods = tuple(method for method in methods if method in args.methods)
    columns, manifest = external.load_external_columns(args.rows)
    policy_names = {action.name for action in primitive.POLICY_ACTIONS}
    rows = []
    for method in methods:
        comparisons = [compare(column, method, policy_names, args.pipeline_map) for column in columns]
        rows.extend(comparisons)
        print(method, dict(Counter(row["status"] for row in comparisons)), flush=True)
    assert len(rows) == len(methods) * len(columns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "method_column_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_method = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        available = [row for row in selected if row["original_status"] == "OK"]
        failures = [row for row in available if row["status"] not in {"EXACT_BYTE_AND_RATIO_PARITY", "RATIO_ONLY_PARITY"}]
        unresolved = [row for row in selected if row["original_status"] in {"NO_LOSSLESS_BASELINE", "BASELINE_UNAVAILABLE", "BASELINE_ROUNDTRIP_FAILED"}]
        by_method[method] = {
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "action_status_counts": dict(Counter(row["action_status"] for row in selected)),
            "available_reference_columns": len(available), "failed_replacement_columns": len(failures),
            "unresolved_reference_columns": len(unresolved),
            "replacement_proved_on_available_columns": bool(available) and not failures,
            "verdict": ("EXCLUDED_BY_USER" if method in benchmark.IGNORED_SPECIALIST_METHODS else
                        "REGISTERED_CHAIN_GAP" if failures else "UNVERIFIED_REFERENCE" if unresolved else
                        "VERIFIED_ON_TESTED_APPLICABLE_INPUTS" if available else "NO_APPLICABLE_TEST_INPUT"),
        }
    summary = {
        "schema": "policy-action-replacement-86-v2", "reference": "versioned benchmark_86_combinations.py adapters",
        "benchmark_contract": benchmark.BENCHMARK_CONTRACT,
        "input_policy": benchmark.NAN_INPUT_POLICY,
        "ignored_methods": benchmark.IGNORED_SPECIALIST_METHODS,
        "policy_action_count": len(policy_names), "method_count": len(methods), "pipeline_map": args.pipeline_map,
        "unified_registry_count": len(unified.ACTIONS),
        "callable_concrete_count_including_experimental": len(primitive.ALL_CONCRETE_ACTION_IMPLEMENTATIONS),
        "policy_ids_dense": [action.action_id for action in primitive.POLICY_ACTIONS] == list(range(len(policy_names))),
        "stop_id_does_not_collide": primitive.STOP_ACTION_ID not in {action.action_id for action in primitive.POLICY_ACTIONS},
        "rows_per_column": args.rows, "checks": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "method_verdict_counts": dict(Counter(row["verdict"] for row in by_method.values())),
        "by_method": by_method, "dataset_manifest": manifest,
        "python_executable": sys.executable,
        "package_versions": {},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), ROOT / "scripts/benchmark_86_combinations.py",
            ROOT / "scripts/primitive_operator_action_runtime.py", ROOT / "scripts/unified_action_runtime.py",
            ROOT / "backend/myapp/legacy_numeric_stages.py", ROOT / "backend/myapp/snappy_legacy_stages.py",
            ROOT / "backend/myapp/reger_legacy_stages.py", ROOT / "backend/myapp/blosc2_legacy_stages.py",
            ROOT / "cpp/src/buff_native_exact_stages.cpp",
            ROOT / "backend/myapp/experiment_data_policy.py",
            ROOT / "backend/myapp/elf64_exact_stages.py", ROOT / "cpp/src/elf_double_codec.cpp",
            ROOT / "backend/myapp/corad_lossless_stages.py",
        )},
        "scope_limits": [
            "Equality is empirical on the supplied columns and prefixes, not a proof over all possible inputs.",
            "Only registered codec chains are tested; alternative action sequences are not exhaustively searched.",
            "Benchmark methods backed by project_native_reference are compared to that reference, not the similarly named upstream libraries.",
            "UAR1 runtime envelopes are excluded from both compression byte counts.",
            "Blosc and Blosc2 use the adapter's sole typesize-1 variant; no frozen successful level exists in the prior selection.",
            "The v2 contract removes NaN before byte measurement, excludes Camel, replaces Elf/Elf+ with binary64, and uses revised lossless CORAD and native BUFF.",
        ],
    }
    for package in ("blosc", "blosc2", "numpy", "pandas", "python-snappy", "cramjam", "brotli", "zstandard", "fpzip"):
        try:
            summary["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            summary["package_versions"][package] = "unavailable"
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "action_registry.json").write_text(
        json.dumps(primitive.registry(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"method_verdict_counts": summary["method_verdict_counts"], "status_counts": summary["status_counts"]}, ensure_ascii=False, indent=2))
    return 1 if any(row["verdict"] == "REGISTERED_CHAIN_GAP" for row in by_method.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
