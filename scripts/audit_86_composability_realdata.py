#!/usr/bin/env python3
"""Falsify exact benchmark-86 composition claims on unseen real columns.

The historical 194-record inventory mixes executable registered actions with
method-scoped C++ ablation candidates. This audit applies the semantic identity
corrections and newly extracted stages before asking
two concrete questions:

1. Does every individually executable action round-trip on the selected data?
2. For each benchmark method, does the strongest registered action sequence
   reproduce the original codec's exact byte count and (for a shared wire)
   byte-for-byte payload?

Codec levels are frozen from the earlier fixed-20-column selection.  They are
never re-selected on the external columns below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
BACKEND = ROOT / "backend"
for path in (str(SCRIPT_DIR), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402
from myapp.experiment_data_policy import prepare_numeric_column  # noqa: E402


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_86_combinations_for_realdata_audit",
        SCRIPT_DIR / "benchmark_86_combinations.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark_86_combinations.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()

INVENTORY = (
    ROOT
    / "analysis_outputs"
    / "operator_implementation_dedup_20260903_v1"
    / "implementations_194.csv"
)
INVENTORY_AUDIT = INVENTORY.with_name("audit.json")
FROZEN_LEVELS = (
    ROOT
    / "analysis_outputs"
    / "cpp_codec_action_split_20260902_v1"
    / "selected_levels.json"
)
PIPELINE_CATALOG = ROOT / "benchmark/reference/methods/method_operator_pipelines.json"

# These are external to the Metro-Traffic + Air-Quality fixed-20 development
# columns used to identify and deduplicate the actions.  We deliberately include
# smooth time series, high-cardinality integers, low-cardinality integers, high
# decimal precision, and one NaN-containing float column.
EXTERNAL_COLUMN_SPECS: tuple[dict[str, Any], ...] = (
    {
        "dataset": "Bird-migration",
        "column_index": 0,
        "column": "event-id",
        "logical_type": "integer",
        "max_point": 0,
        "header": True,
    },
    {
        "dataset": "Bird-migration",
        "column_index": 3,
        "column": "location-long",
        "logical_type": "float",
        "max_point": 5,
        "header": True,
    },
    {
        "dataset": "meteoblue",
        "column_index": 1,
        "column": "Basel Temperature [2 m elevation corrected]",
        "logical_type": "float",
        "max_point": 11,
        "header": True,
    },
    {
        "dataset": "meteoblue",
        "column_index": 3,
        "column": "Basel Temperature [900 mb]",
        "logical_type": "float",
        "max_point": 2,
        "header": True,
    },
    {
        "dataset": "Wins",
        "column_index": 8,
        "column": "column_8",
        "logical_type": "integer",
        "max_point": 0,
        "header": True,
    },
    {
        "dataset": "Wins",
        "column_index": 39,
        "column": "column_39",
        "logical_type": "integer",
        "max_point": 0,
        "header": True,
    },
    {
        "dataset": "Romance",
        "column_index": 5,
        "column": "column_5",
        "logical_type": "float",
        "max_point": 15,
        "header": True,
    },
)

EXACT_ATOMIC_METHODS = {"Huffman", "Arithmetic", "ANS", "Dict."}
EXACT_STAGEWISE_METHODS = {
    "Chimp", "Chimp128", "Pongo", "Gorilla", "2DIFF", "SVB", "Simple8b", "RLE",
    "RLBE", "Sprintz", "DBP", "BOS", "SPDP", "FPC", "PFPC", "Elf",
    "Elf+", "Camel", "FPZIP",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_stage_artifact_hashes() -> dict[str, str]:
    """Hash the task-specific sources and executables that establish parity."""

    paths = [
        ROOT / "cpp/CMakeLists.txt",
        SCRIPT_DIR / "primitive_operator_action_runtime.py",
        SCRIPT_DIR / "unified_action_runtime.py",
        SCRIPT_DIR / "test_unified_action_runtime.py",
        SCRIPT_DIR / "audit_new_exact_stage_actions_realdata.py",
        SCRIPT_DIR / "audit_exact_action_split_35.py",
        SCRIPT_DIR / "audit_requested_codec_splits_realdata.py",
        SCRIPT_DIR / "audit_general_codec_splits_realdata.py",
        SCRIPT_DIR / "verify_exact_action_split_determinism.py",
        SCRIPT_DIR / "measure_selected_codec_cost.py",
        ROOT / "cpp/src/remaining_codec_action_catalog.cpp",
        ROOT / "cpp/src/remaining_codec_action_catalog.hpp",
        ROOT / "cpp/src/elf_float_codec.cpp",
        ROOT / "cpp/src/elf_float_codec.hpp",
        ROOT / "cpp/src/deflate_container_stages.cpp",
        ROOT / "cpp/src/deflate_container_stages.hpp",
        ROOT / "cpp/src/xz_container_stage.cpp",
        ROOT / "cpp/src/xz_container_stage.hpp",
        ROOT / "cpp/src/pongo_codec.cpp",
        ROOT / "cpp/src/pongo_exact_stages.hpp",
    ]
    paths.extend(sorted((BACKEND / "myapp").rglob("*.py")))
    paths.extend(sorted((ROOT / "cpp/src").glob("*exact_stages*.cpp")))
    paths.extend(sorted((ROOT / "cpp/src").glob("*exact_stages*.hpp")))
    paths.append(ROOT / "cpp/src/rle_int64_action_cli.cpp")
    for stem in (
        "rle", "ts2diff", "svb", "simple8b", "p04", "dbp", "gorilla",
        "chimp128", "pongo", "spdp", "fpc", "pfpc", "bos", "fastpfor", "elf",
        "camel", "fpzip", "lz4", "snappy", "zstd",
    ):
        paths.extend(sorted((ROOT / "cpp/build").glob(f"{stem}_exact_stages_*")))
    paths.extend(
        [
            ROOT / "cpp/build/rle_int64_action_cli",
            ROOT / "cpp/build/remaining_codec_action_catalog_cli",
            ROOT / "cpp/build/deflate_container_stages_cli",
            ROOT / "cpp/build/xz_container_stage_cli",
        ]
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
        if path.is_file()
    }


def _json_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _logical_variant(column: Any) -> str:
    return "int" if column.logical_type == "integer" else "float"


def load_external_columns(rows: int) -> tuple[list[Any], list[dict[str, Any]]]:
    columns: list[Any] = []
    manifest: list[dict[str, Any]] = []
    for spec in EXTERNAL_COLUMN_SPECS:
        source = ROOT / "orignial_datasets" / f"{spec['dataset']}.csv"
        metadata_path = (
            ROOT
            / "backend"
            / "dataset"
            / spec["dataset"]
            / f"{spec['dataset']}_column_info.csv"
        )
        metadata_rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
        metadata_matches = [
            row
            for row in metadata_rows
            if int(row["column_index"]) == int(spec["column_index"])
        ]
        if len(metadata_matches) != 1:
            raise ValueError(
                f"metadata index mismatch: {spec['dataset']}:{spec['column_index']}"
            )
        metadata = metadata_matches[0]
        if metadata["logical_type"] != spec["logical_type"] or int(
            metadata["max_point_number"]
        ) != int(spec["max_point"]):
            raise ValueError(f"metadata type/max_point mismatch: {spec}")
        if spec["header"] and metadata["column_name"] != spec["column"]:
            raise ValueError(f"metadata column-name mismatch: {spec}")
        if spec["header"]:
            frame = pd.read_csv(
                source,
                usecols=[spec["column"]],
                nrows=rows,
                low_memory=False,
                dtype=str if spec["logical_type"] == "integer" else None,
            )
        else:
            frame = pd.read_csv(
                source,
                header=None,
                names=[spec["column"]],
                nrows=rows,
                low_memory=False,
                dtype=str if spec["logical_type"] == "integer" else None,
            )
        series = frame[spec["column"]]
        if len(series) != rows:
            raise ValueError(
                f"{spec['dataset']}:{spec['column']} has {len(series)} rows, expected {rows}"
            )
        prepared = prepare_numeric_column(series.tolist(), spec["logical_type"])
        values = prepared.values
        column = benchmark.Column(
            spec["dataset"],
            int(spec["column_index"]),
            spec["column"],
            spec["logical_type"],
            int(spec["max_point"]),
            values,
        )
        columns.append(column)
        manifest.append(
            {
                "dataset": column.dataset,
                "column_index": column.index,
                "column": column.name,
                "logical_type": column.logical_type,
                "max_point": column.max_point,
                **prepared.manifest(),
                "nulls": 0,
                "unique_values_in_prefix": len(set(values)),
                "source": str(source.relative_to(ROOT)),
                "source_file_sha256": _sha256(source),
                "metadata": str(metadata_path.relative_to(ROOT)),
                "metadata_file_sha256": _sha256(metadata_path),
            }
        )
    return columns, manifest


def load_and_validate_frozen_levels() -> dict[str, str | None]:
    payload = json.loads(FROZEN_LEVELS.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"invalid frozen level file: {FROZEN_LEVELS}")
    expected = {
        method: primitive.BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS[method]
        for method in benchmark.GENERAL_METHODS
    }
    actual = {method: selected.get(method) for method in benchmark.GENERAL_METHODS}
    if actual != expected:
        mismatches = {
            method: {"frozen": actual[method], "runtime": expected[method]}
            for method in benchmark.GENERAL_METHODS
            if actual[method] != expected[method]
        }
        raise AssertionError(f"runtime/frozen level mismatch: {mismatches}")
    return actual


def original_artifact(column: Any, method: str) -> dict[str, Any]:
    if method in benchmark.SPECIALIST_METHODS:
        artifact, reason = benchmark.specialist_artifact(column, method)
        if artifact is None:
            return {"status": "not_materialized", "reason": reason}
        return {
            "status": "ok",
            "payload": artifact.payload,
            "decode": artifact.decode,
            "implementation": artifact.implementation,
            "kind": "specialist",
        }

    key = primitive.BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS[method]
    if key is None:
        return {
            "status": "not_materialized",
            "reason": "no frozen level round-tripped on every fixed-20 development column",
            "selected_key": None,
        }
    try:
        payload, decode, implementation = benchmark._compress_generic(column.raw, key)
        if decode(payload) != column.raw:
            return {
                "status": "roundtrip_failed",
                "reason": "original codec decoder did not recover the input bytes",
                "selected_key": key,
            }
        return {
            "status": "ok",
            "payload": payload,
            "decode": decode,
            "implementation": implementation,
            "kind": "general",
            "selected_key": key,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": _error(exc),
            "selected_key": key,
        }


def exact_action_names(column: Any, method: str) -> tuple[str, ...]:
    if method in primitive.EXACT_NATIVE_METHOD_KEYS:
        return tuple(
            action.name
            for action in primitive.exact_native_lz_action_pipeline(
                primitive.EXACT_NATIVE_METHOD_KEYS[method]
            )
        )
    if method == "2DIFF":
        return (
            primitive.TS2DIFF_INT_ACTIONS
            if _logical_variant(column) == "int"
            else primitive.TS2DIFF_FLOAT_ACTIONS
        )
    if method == "Gorilla":
        return primitive.GORILLA_ACTIONS
    if method in {"Chimp", "Chimp128"}:
        return primitive.CHIMP128_ACTIONS
    if method == "Pongo":
        return primitive.PONGO_ACTIONS
    if method == "BOS":
        return (
            primitive.BOS64_INT_ACTIONS
            if _logical_variant(column) == "int"
            else primitive.BOS64_FLOAT_ACTIONS
        )
    if method == "Camel":
        return primitive.CAMEL_ACTIONS
    if method == "SPDP":
        return primitive.SPDP_ACTIONS
    if method == "FPC":
        return primitive.FPC_ACTIONS
    if method == "PFPC":
        return primitive.PFPC_ACTIONS
    if method == "Elf":
        return primitive.ELF_ACTIONS
    if method == "Elf+":
        return primitive.ELF_PLUS_ACTIONS
    if method == "FPZIP":
        return primitive.FPZIP_ACTIONS
    if method == "SVB":
        return (
            primitive.SVB_INT_ACTIONS
            if _logical_variant(column) == "int"
            else primitive.SVB_FLOAT_ACTIONS
        )
    if method == "Simple8b":
        return (
            primitive.SIMPLE8B_INT_ACTIONS
            if _logical_variant(column) == "int"
            else primitive.SIMPLE8B_FLOAT_ACTIONS
        )
    if method == "RLBE":
        return primitive.RLBE_ACTIONS
    if method == "Sprintz":
        return primitive.SPRINTZ_ACTIONS
    if method == "DBP":
        return (
            primitive.DBP_INT_ACTIONS
            if _logical_variant(column) == "int"
            else primitive.DBP_FLOAT_ACTIONS
        )
    logical_type = _logical_variant(column) if method in {"RLE", "Dict."} else "bytes"
    max_point = column.max_point if method == "RLE" and logical_type == "float" else None
    return primitive.exact_atomic_action_pipeline(
        method, logical_type, max_point=max_point
    )


def run_exact_composition(
    column: Any, method: str, original: dict[str, Any]
) -> dict[str, Any]:
    logical_type = (
        _logical_variant(column)
        if method in {
            "RLE", "Dict.", "Chimp", "Chimp128", "Pongo", "Gorilla", "2DIFF", "SVB",
            "Simple8b", "RLBE", "Sprintz", "DBP", "BOS", "Camel", "Elf", "Elf+",
            "FPZIP",
        }
        else "bytes"
    )
    max_point = (
        column.max_point
        if method in {"RLE", "2DIFF", "SVB", "Simple8b", "DBP", "BOS"} and logical_type == "float"
        else None
    )
    try:
        action_names = exact_action_names(column, method)
    except Exception as exc:
        return {
            "mode": "exact_registered",
            "status": "no_legal_action_sequence",
            "action_names": [],
            "reason": _error(exc),
        }
    try:
        if logical_type in {"int", "integer"}:
            source = unified.ActionData("int64", column.raw, len(column.raw) // 8)
        elif logical_type == "float":
            source = unified.ActionData("float64", column.raw, len(column.raw) // 8)
        else:
            source = unified.ActionData.bytes(column.raw)
        encoded = unified.run_actions(
            action_names, source, max_point=max_point
        )
        _, payload = unified._unpack_result(encoded)
        restored = unified.run_actions(
            action_names, encoded, reverse=True, max_point=max_point
        )
        stage_roundtrip = restored == source
        recovered = original["decode"](payload)
        if original["kind"] == "specialist":
            original_decoder_roundtrip = benchmark._values_equal(column, recovered)
        else:
            original_decoder_roundtrip = bytes(recovered) == column.raw
        roundtrip = stage_roundtrip and original_decoder_roundtrip
        return {
            "mode": "exact_registered",
            "status": "ok" if roundtrip else "roundtrip_failed",
            "action_names": list(action_names),
            "bytes": len(payload),
            "roundtrip": bool(roundtrip),
            "stagewise_inverse_roundtrip": bool(stage_roundtrip),
            "original_decoder_roundtrip": bool(original_decoder_roundtrip),
            "ratio_equal": len(payload) == len(original["payload"]),
            "payload_identical": payload == original["payload"],
        }
    except Exception as exc:
        return {
            "mode": "exact_registered",
            "status": "error",
            "action_names": list(action_names),
            "reason": _error(exc),
        }


def run_reference_proxy(column: Any, method: str) -> dict[str, Any]:
    try:
        action_names = primitive.concrete_action_pipeline(
            method, logical_type=_logical_variant(column)
        )
        input_values = np.frombuffer(column.raw, dtype="<i8").copy()
        payload = primitive.encode_operator_pipeline(input_values, action_names)
        recovered, decoded_names = primitive.decode_operator_pipeline(payload)
        roundtrip = np.array_equal(recovered, input_values)
        return {
            "mode": "reference_proxy",
            "status": "ok" if roundtrip else "roundtrip_failed",
            "action_names": list(action_names),
            "decoded_action_names_match": tuple(decoded_names) == tuple(action_names),
            "bytes": len(payload),
            "roundtrip": bool(roundtrip),
        }
    except Exception as exc:
        return {
            "mode": "reference_proxy",
            "status": "error",
            "action_names": [],
            "reason": _error(exc),
        }


def run_method_comparisons(columns: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    exact_declared = (
        set(primitive.EXACT_NATIVE_METHOD_KEYS)
        | EXACT_ATOMIC_METHODS
        | EXACT_STAGEWISE_METHODS
    )
    specialist_set = set(benchmark.SPECIALIST_METHODS)

    for method_index, method in enumerate(primitive.BENCHMARK_86_METHODS, 1):
        print(f"METHOD {method_index:02d}/86 {method}", flush=True)
        for column in columns:
            row: dict[str, Any] = {
                "method": method,
                "method_kind": "specialist" if method in specialist_set else "general",
                "dataset": column.dataset,
                "column_index": column.index,
                "column": column.name,
                "logical_type": column.logical_type,
                "max_point": column.max_point,
                "raw_bytes": len(column.raw),
                "declared_exact_method": method in exact_declared,
            }
            original = original_artifact(column, method)
            row["original_status"] = original["status"]
            row["selected_key"] = original.get("selected_key")
            if original["status"] != "ok":
                row.update(
                    {
                        "original_bytes": None,
                        "original_implementation": None,
                        "composition_mode": "not_run",
                        "composition_status": "original_unavailable",
                        "composition_bytes": None,
                        "composition_roundtrip": False,
                        "ratio_equal": False,
                        "payload_identical": None,
                        "action_names": [],
                        "reason": original.get("reason", "original unavailable"),
                    }
                )
                comparisons.append(row)
                continue

            row["original_bytes"] = len(original["payload"])
            row["original_implementation"] = original["implementation"]
            if method in exact_declared:
                composition = run_exact_composition(column, method, original)
                composition["semantics"] = (
                    "closed_sequence_local_native_surrogate_parity"
                    if method in primitive.EXACT_NATIVE_METHOD_KEYS
                    else (
                        "independent_stagewise_project_codec_parity"
                        if method in EXACT_STAGEWISE_METHODS
                        else "single_atomic_project_codec_parity"
                    )
                )
            else:
                composition = run_reference_proxy(column, method)
                composition["semantics"] = "normalized_int64_reference_proxy"
                if composition.get("bytes") is not None:
                    composition["ratio_equal"] = (
                        int(composition["bytes"]) == len(original["payload"])
                    )
                composition["payload_identical"] = None

            row.update(
                {
                    "composition_mode": composition["mode"],
                    "composition_status": composition["status"],
                    "composition_semantics": composition["semantics"],
                    "composition_bytes": composition.get("bytes"),
                    "composition_roundtrip": bool(composition.get("roundtrip", False)),
                    "stagewise_inverse_roundtrip": composition.get(
                        "stagewise_inverse_roundtrip"
                    ),
                    "original_decoder_roundtrip": composition.get(
                        "original_decoder_roundtrip"
                    ),
                    "ratio_equal": bool(composition.get("ratio_equal", False)),
                    "payload_identical": composition.get("payload_identical"),
                    "action_names": composition.get("action_names", []),
                    "reason": composition.get("reason", ""),
                }
            )
            comparisons.append(row)

    methods: list[dict[str, Any]] = []
    for method in primitive.BENCHMARK_86_METHODS:
        rows = [row for row in comparisons if row["method"] == method]
        original_rows = [row for row in rows if row["original_status"] == "ok"]
        exact_rows = [
            row
            for row in original_rows
            if row["composition_mode"] == "exact_registered"
            and row["composition_status"] == "ok"
        ]
        exact_pass_rows = [
            row
            for row in exact_rows
            if row["composition_roundtrip"]
            and row["ratio_equal"]
            and row["payload_identical"] is True
        ]
        proxy_rows = [
            row for row in original_rows if row["composition_mode"] == "reference_proxy"
        ]
        declared = method in exact_declared
        exact_complete = (
            declared
            and bool(original_rows)
            and len(exact_pass_rows) == len(original_rows)
        )
        exact_all_external = exact_complete and len(original_rows) == len(rows)
        if exact_complete:
            status = (
                "pass_closed_sequence_local_surrogate_parity"
                if method in primitive.EXACT_NATIVE_METHOD_KEYS
                else (
                    "pass_independent_stagewise_project_codec_parity"
                    if method in EXACT_STAGEWISE_METHODS
                    else "pass_single_atomic_project_codec_parity"
                )
            )
        elif declared and exact_pass_rows:
            status = "partial_exact_external_failure"
        elif declared:
            status = "local_parity_path_failed_or_not_materialized"
        else:
            status = "no_exact_registered_composition"
        methods.append(
            {
                "method": method,
                "method_kind": rows[0]["method_kind"],
                "declared_exact_method": declared,
                "selected_key": rows[0].get("selected_key"),
                "test_columns": len(rows),
                "original_materialized_columns": len(original_rows),
                "original_unavailable_columns": len(rows) - len(original_rows),
                "exact_pass_columns": len(exact_pass_rows),
                "proxy_roundtrip_columns": sum(
                    bool(row["composition_roundtrip"]) for row in proxy_rows
                ),
                "proxy_ratio_equal_columns": sum(bool(row["ratio_equal"]) for row in proxy_rows),
                "exact_complete_on_materialized_columns": exact_complete,
                "local_parity_complete_on_all_external_columns": exact_all_external,
                "status": status,
                "failure_reasons": sorted(
                    {
                        row["reason"]
                        for row in rows
                        if row.get("reason") and row["composition_status"] != "ok"
                    }
                ),
            }
        )
    return comparisons, methods


def _test_atomic_action(action_name: str, column: Any) -> tuple[bool, bool, str]:
    logical_type = _logical_variant(column)
    method_and_type: dict[str, tuple[str, str]] = {
        "op.entropy_huffman.project_huffman_bytes.v1": ("Huffman", "bytes"),
        "op.entropy_range.project_arithmetic_bytes.v1": ("Arithmetic", "bytes"),
        "op.entropy_fse.project_rans_bytes.v1": ("ANS", "bytes"),
        "op.dictionary.project_int64.v1": ("Dict.", "int"),
        "op.dictionary.project_float64.v1": ("Dict.", "float"),
        "op.run_length.tsfile_int64.v1": ("RLE", "int"),
    }
    method, required = method_and_type[action_name]
    if required in {"int", "float"} and logical_type != required:
        return False, False, "not_applicable_to_logical_type"
    payload = primitive.encode_registered_exact_action_pipeline(
        column.raw, (action_name,), logical_type=required
    )
    if method == "Huffman":
        recovered = benchmark.huffman_decode(payload)
        ok = recovered == column.raw
    elif method == "Arithmetic":
        recovered = benchmark.arithmetic_decode(payload)
        ok = recovered == column.raw
    elif method == "ANS":
        recovered = benchmark.ans_decode(payload)
        ok = recovered == column.raw
    elif method == "Dict." and required == "int":
        recovered = benchmark.decode_dictionary_int(payload)
        ok = benchmark._values_equal(column, recovered)
    elif method == "Dict." and required == "float":
        recovered = benchmark.decode_dictionary_float(payload)
        ok = benchmark._values_equal(column, recovered)
    elif method == "RLE":
        recovered = benchmark.decode_rle_int64(payload)
        ok = benchmark._values_equal(column, recovered)
    else:
        raise AssertionError(action_name)
    return True, bool(ok), ""


def audit_inventory_actions(
    columns: list[Any], comparisons: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_rows = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    if len(inventory_rows) != 194:
        raise AssertionError(f"expected 194 source inventory rows, found {len(inventory_rows)}")
    # Ratio/name-only dedup had merged Gorilla's state machine with Chimp128.
    # Use the corrected callable identity surface from the unified registry.
    by_name = {row["canonical_name"]: row for row in inventory_rows}
    expanded: list[dict[str, Any]] = []
    for descriptor in unified.ACTIONS:
        source = by_name.get(descriptor.name)
        if source is None and descriptor.name in primitive.GORILLA_ACTIONS:
            source = next(
                row for row in inventory_rows
                if descriptor.name in row["member_names"].split(" | ")
            )
        if source is None:
            source = {
                "implementation_index": "-1",
                "operator_id": descriptor.operator_id,
                "operator_label_zh": descriptor.operator_id,
                "within_operator_index": "",
                "canonical_name": descriptor.name,
                "member_names": descriptor.name,
                "origin": "unified_registry_semantic_extension",
                "registered_action_id": (
                    "" if descriptor.legacy_registered_action_id is None
                    else str(descriptor.legacy_registered_action_id)
                ),
                "execution_domain": descriptor.execution_mode,
                "source_methods": " | ".join(descriptor.source_methods),
                "selected_levels": "",
                "evidence": descriptor.evidence,
                "standalone_composable": str(
                    descriptor.execution_mode in {"standalone", "stagewise"}
                ),
                "implementation_ids": descriptor.name,
                "implementation_sources": descriptor.implementation_source,
                "implementation_scope": descriptor.unavailable_reason,
            }
        row = dict(source)
        concrete = primitive.ALL_CONCRETE_ACTION_BY_NAME.get(descriptor.name)
        row.update(
            {
                "operator_id": descriptor.operator_id,
                "canonical_name": descriptor.name,
                "member_names": descriptor.name,
                "origin": "unified_action_registry",
                "registered_action_id": (
                    ""
                    if descriptor.legacy_registered_action_id is None
                    else str(descriptor.legacy_registered_action_id)
                ),
                "execution_domain": descriptor.execution_mode,
                "source_methods": " | ".join(descriptor.source_methods),
                "evidence": descriptor.evidence,
                "standalone_composable": str(
                    descriptor.execution_mode in {"standalone", "stagewise"}
                ),
                "implementation_ids": getattr(
                    concrete, "implementation_id", descriptor.name
                ),
                "implementation_sources": descriptor.implementation_source,
                "implementation_scope": getattr(
                    concrete, "implementation_scope", descriptor.unavailable_reason
                ),
            }
        )
        expanded.append(row)
    inventory_rows = expanded
    if len(inventory_rows) != len(unified.ACTIONS):
        raise AssertionError(
            "corrected inventory must match the unified registry: "
            f"{len(inventory_rows)} != {len(unified.ACTIONS)}"
        )
    atomic_names = {action.name for action in primitive.EXACT_ATOMIC_ACTIONS}
    exact_pass_sequences = [
        row
        for row in comparisons
        if row["composition_mode"] == "exact_registered"
        and row["composition_status"] == "ok"
        and row["composition_roundtrip"]
        and row["ratio_equal"]
        and row["payload_identical"] is True
    ]
    details: list[dict[str, Any]] = []
    per_action_tests: list[dict[str, Any]] = []

    for inventory_row in inventory_rows:
        name = inventory_row["canonical_name"]
        applicable = 0
        passed = 0
        failures: list[str] = []
        if name in primitive.ACTION_BY_NAME:
            testability = "standalone_registered"
            for column in columns:
                applicable += 1
                try:
                    values = np.frombuffer(column.raw, dtype="<i8").copy()
                    payload = primitive.encode_operator_pipeline(values, (name,))
                    recovered, decoded_names = primitive.decode_operator_pipeline(payload)
                    ok = np.array_equal(recovered, values) and decoded_names == (name,)
                    passed += int(ok)
                    if not ok:
                        failures.append(
                            f"single-action roundtrip mismatch: {column.dataset}:{column.name}"
                        )
                    per_action_tests.append(
                        {
                            "action": name,
                            "dataset": column.dataset,
                            "column": column.name,
                            "applicable": True,
                            "roundtrip": bool(ok),
                            "encoded_bytes": len(payload),
                            "reason": "" if ok else "single-action roundtrip mismatch",
                        }
                    )
                except Exception as exc:
                    failures.append(_error(exc))
                    per_action_tests.append(
                        {
                            "action": name,
                            "dataset": column.dataset,
                            "column": column.name,
                            "applicable": True,
                            "roundtrip": False,
                            "encoded_bytes": None,
                            "reason": _error(exc),
                        }
                    )
            status = "pass" if passed == applicable else "fail"
        elif name in atomic_names:
            testability = "standalone_registered"
            for column in columns:
                try:
                    is_applicable, ok, reason = _test_atomic_action(name, column)
                    if is_applicable:
                        applicable += 1
                        passed += int(ok)
                    per_action_tests.append(
                        {
                            "action": name,
                            "dataset": column.dataset,
                            "column": column.name,
                            "applicable": is_applicable,
                            "roundtrip": bool(ok) if is_applicable else None,
                            "encoded_bytes": None,
                            "reason": reason,
                        }
                    )
                except Exception as exc:
                    applicable += 1
                    failures.append(_error(exc))
                    per_action_tests.append(
                        {
                            "action": name,
                            "dataset": column.dataset,
                            "column": column.name,
                            "applicable": True,
                            "roundtrip": False,
                            "encoded_bytes": None,
                            "reason": _error(exc),
                        }
                    )
            status = "pass" if applicable > 0 and passed == applicable else "fail"
        elif unified.ACTION_BY_NAME[name].execution_mode == "stagewise":
            testability = "stagewise_registered"
            exercised = [
                row for row in exact_pass_sequences if name in row["action_names"]
            ]
            applicable = len(exercised)
            passed = len(exercised)
            status = (
                "observed_in_successful_independent_sequence"
                if exercised
                else "not_exercised"
            )
        elif unified.ACTION_BY_NAME[name].execution_mode == "closed_sequence_only":
            testability = "sequence_only_registered_stage"
            exercised = [
                row for row in exact_pass_sequences if name in row["action_names"]
            ]
            applicable = len(exercised)
            passed = len(exercised)
            status = (
                "observed_in_successful_closed_sequence"
                if exercised
                else "not_exercised"
            )
        elif unified.ACTION_BY_NAME[name].execution_mode == "experimental_non_atomic":
            testability = "experimental_non_atomic_helper"
            status = "excluded_from_policy"
        else:
            testability = "method_scoped_non_standalone_candidate"
            status = "not_executable"

        details.append(
            {
                **inventory_row,
                "runtime_testability": testability,
                "realdata_status": status,
                "applicable_tests": applicable,
                "passed_tests": passed,
                "failure_reasons": sorted(set(failures)),
            }
        )
    return details, per_action_tests


def core_result(
    columns_manifest: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    action_tests: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_pass_methods = [
        row["method"]
        for row in methods
        if row["exact_complete_on_materialized_columns"]
    ]
    exact_declared_methods = [row["method"] for row in methods if row["declared_exact_method"]]
    atomic_parity_methods = [
        row["method"]
        for row in methods
        if row["status"] == "pass_single_atomic_project_codec_parity"
    ]
    surrogate_parity_methods = [
        row["method"]
        for row in methods
        if row["status"] == "pass_closed_sequence_local_surrogate_parity"
    ]
    stagewise_parity_methods = [
        row["method"]
        for row in methods
        if row["status"] == "pass_independent_stagewise_project_codec_parity"
    ]
    action_status_counts = Counter(row["realdata_status"] for row in actions)
    testability_counts = Counter(row["runtime_testability"] for row in actions)
    return {
        "schema": "benchmark-86-realdata-composability-audit-v1",
        "claim_rule": (
            "An algorithm is exactly composable only when a registered executable action "
            "sequence losslessly decodes and matches the frozen original codec byte count "
            "on every materialized external column; shared-wire paths must be byte-identical."
        ),
        "candidate_boundary": (
            "Method-scoped C++ ablation records are not standalone encoders and cannot count "
            "as executable action composition evidence."
        ),
        "positive_evidence_boundary": (
            "The passing count is local project implementation/reference parity. Native-LZ "
            "rows compare a project surrogate direct path with its closed staged path, not "
            "the same-named upstream codec. Independently callable stagewise paths are reported "
            "separately from those closed native-LZ sequences."
        ),
        "frozen_level_config": {
            "path": str(FROZEN_LEVELS.relative_to(ROOT)),
            "sha256": _sha256(FROZEN_LEVELS),
            "selected": load_and_validate_frozen_levels(),
        },
        "development_columns_used_for_action_identity": 20,
        "external_columns": columns_manifest,
        "inventory": {
            "records": len(actions),
            "runtime_testability_counts": dict(sorted(testability_counts.items())),
            "realdata_status_counts": dict(sorted(action_status_counts.items())),
            "standalone_test_rows": len(action_tests),
        },
        "methods": {
            "total": len(methods),
            "declared_exact_before_external_test": len(exact_declared_methods),
            "declared_exact_method_names": exact_declared_methods,
            "local_reference_parity_pass_on_all_external_columns": len(exact_pass_methods),
            "exact_pass_on_all_materialized_external_columns": len(exact_pass_methods),
            "exact_pass_method_names": exact_pass_methods,
            "single_atomic_project_codec_parity": len(atomic_parity_methods),
            "single_atomic_project_codec_parity_names": atomic_parity_methods,
            "closed_sequence_local_surrogate_parity": len(surrogate_parity_methods),
            "closed_sequence_local_surrogate_parity_names": surrogate_parity_methods,
            "independent_stagewise_project_codec_parity": len(stagewise_parity_methods),
            "independent_stagewise_project_codec_parity_names": stagewise_parity_methods,
            "independent_stagewise_named_upstream_codec_compositions_proven": 0,
            "not_exactly_composable": len(methods) - len(exact_pass_methods),
            "all_86_exactly_composable": len(exact_pass_methods) == 86,
            "status_counts": dict(sorted(Counter(row["status"] for row in methods).items())),
        },
        "method_rows": methods,
        "comparison_rows": comparisons,
        "action_rows": actions,
        "standalone_action_test_rows": action_tests,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def report_markdown(audit: dict[str, Any]) -> str:
    method_rows = audit["method_rows"]
    passed = [row["method"] for row in method_rows if row["exact_complete_on_materialized_columns"]]
    partial = [row for row in method_rows if row["status"] == "partial_exact_external_failure"]
    missing = [row["method"] for row in method_rows if row["status"] == "no_exact_registered_composition"]
    exact_failed = [
        row
        for row in method_rows
        if row["status"] == "local_parity_path_failed_or_not_materialized"
    ]
    failed_actions = [
        row["canonical_name"]
        for row in audit["action_rows"]
        if row["runtime_testability"] == "standalone_registered"
        and row["realdata_status"] == "fail"
    ]
    inv = audit["inventory"]
    lines = [
        f"# {len(audit['action_rows'])} 个语义算子记录能否精确组成 86 个压缩算法：真实数据审计",
        "",
        "- 结论：**仍不能组成全部 86 个**。报告中标为 independent stagewise 的 project path 均是真实逐-stage 组合；其余 method-scoped omission candidate 仍不能据此算作可组合实现。",
        f"- 本地 parity：**{len(passed)}/86**；其中 "
        f"{audit['methods']['single_atomic_project_codec_parity']} 个是单个完整 project codec action，"
        f"{audit['methods']['closed_sequence_local_surrogate_parity']} 个是 project native-LZ surrogate 的 direct-vs-closed-staged parity，"
        f"{audit['methods']['independent_stagewise_project_codec_parity']} 个是独立逐 stage 的 project codec parity。",
        f"- 测试：7 个真实世界数值列，每列前 "
        f"{audit['external_columns'][0]['rows']:,} 行；数据集未参与此前固定 20 列 action 拆分。",
        "- 判据：原 codec 与注册 action 序列均无损，压缩字节数相同；同线格式还须 payload 逐字节相同。",
        "- 参数：沿用固定 20 列选出的全局 level；没有在本测试集上重选。",
        "",
        f"## {len(audit['action_rows'])} 条记录的可执行边界",
        "",
    ]
    for key, value in inv["runtime_testability_counts"].items():
        lines.append(f"- `{key}`：{value}")
    lines.extend(
        [
            f"- standalone 实测失败（{len(failed_actions)}）："
            + (", ".join(failed_actions) if failed_actions else "无"),
            "",
            "其中 method-scoped C++ candidate 只提供‘删掉某阶段后的整 codec 大小’，不是可串联编码器；因此不能用来证明 86 个算法已由 action 组成。",
            "",
            "## 方法级结论",
            "",
            f"- 本地实现/reference parity 在全部可物化外部列上通过（{len(passed)}）：{', '.join(passed)}",
            f"- Project codec 的独立逐-stage 组合已证明（{audit['methods']['independent_stagewise_project_codec_parity']}）："
            + ", ".join(audit["methods"]["independent_stagewise_project_codec_parity_names"]),
            "- 同名 upstream codec 的独立逐-stage 组合已证明：0/86。",
            f"- 有 exact 声明但外部测试不完整（{len(partial) + len(exact_failed)}）："
            + ", ".join(row["method"] for row in partial + exact_failed),
            f"- 尚无 exact 注册组合（{len(missing)}）：{', '.join(missing)}",
            "",
            "## 86 个方法逐项",
            "",
            "| 方法 | 类型 | 原 codec 可测列 | exact 通过列 | proxy 同字节列 | 结论 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['method_kind']} | "
            f"{row['original_materialized_columns']}/{row['test_columns']} | "
            f"{row['exact_pass_columns']} | {row['proxy_ratio_equal_columns']} | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 适用范围",
            "",
            f"这是一组确定性外部前缀测试，不是所有输入的形式化证明。{len(passed)}/86 只表示本地实现/reference parity；"
            "其中仅显式标为 independent stagewise 的项目路径证明了真实逐-stage 数据流，不能外推为同名 upstream codec 兼容。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_outputs" / "benchmark_86_realdata_composability_20260903_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.repetitions <= 0:
        raise ValueError("rows and repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns, columns_manifest = load_external_columns(args.rows)
    load_and_validate_frozen_levels()
    cores: list[dict[str, Any]] = []
    action_tests_first: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        print(f"REPETITION {repetition}/{args.repetitions}", flush=True)
        comparisons, methods = run_method_comparisons(columns)
        actions, action_tests = audit_inventory_actions(columns, comparisons)
        core = core_result(columns_manifest, comparisons, methods, actions, action_tests)
        cores.append(core)
        if repetition == 1:
            action_tests_first = action_tests
        (args.output_dir / f"audit_repeat{repetition}.json").write_text(
            json.dumps(core, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    hashes = [_json_hash(core) for core in cores]
    deterministic = len(set(hashes)) == 1
    canonical = cores[0]
    canonical["execution"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows_per_column": args.rows,
        "repetitions": args.repetitions,
        "repeat_core_sha256": hashes,
        "deterministic_repeats_identical": deterministic,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "codec_python_packages": {
            name: _package_version(name)
            for name in ("fpzip", "lz4", "python-snappy", "zstandard", "brotli")
        },
        "vendored_fpzip_git_head": _git_value(
            "-C", str(ROOT / "external/compression/fpzip"), "rev-parse", "HEAD"
        ),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain_sha256": hashlib.sha256(
            _git_value("status", "--porcelain=v1").encode("utf-8")
        ).hexdigest(),
        "input_file_hashes": {
            str(INVENTORY.relative_to(ROOT)): _sha256(INVENTORY),
            str(INVENTORY_AUDIT.relative_to(ROOT)): _sha256(INVENTORY_AUDIT),
            str(FROZEN_LEVELS.relative_to(ROOT)): _sha256(FROZEN_LEVELS),
            str(PIPELINE_CATALOG.relative_to(ROOT)): _sha256(PIPELINE_CATALOG),
            "scripts/primitive_operator_action_runtime.py": _sha256(
                SCRIPT_DIR / "primitive_operator_action_runtime.py"
            ),
            "scripts/benchmark_86_combinations.py": _sha256(
                SCRIPT_DIR / "benchmark_86_combinations.py"
            ),
            "scripts/audit_86_composability_realdata.py": _sha256(Path(__file__)),
            "backend/myapp/native_lz_family.py": _sha256(BACKEND / "myapp/native_lz_family.py"),
            "backend/myapp/entropy_cpp.py": _sha256(BACKEND / "myapp/entropy_cpp.py"),
            "backend/myapp/cpp_action_bridge.py": _sha256(BACKEND / "myapp/cpp_action_bridge.py"),
            "cpp/src/ans_rans.cpp": _sha256(ROOT / "cpp/src/ans_rans.cpp"),
            "cpp/src/arithmetic_bytes.cpp": _sha256(ROOT / "cpp/src/arithmetic_bytes.cpp"),
            "cpp/src/huffman_bytes.cpp": _sha256(ROOT / "cpp/src/huffman_bytes.cpp"),
            "backend/myapp/column_encodings.py": _sha256(BACKEND / "myapp/column_encodings.py"),
            "backend/myapp/tsfile_encoding/p02.py": _sha256(BACKEND / "myapp/tsfile_encoding/p02.py"),
        },
        "exact_stage_implementation_artifact_hashes": _exact_stage_artifact_hashes(),
    }
    if not deterministic:
        canonical["execution"]["integrity_warning"] = (
            "deterministic repetitions produced different core result hashes"
        )

    (args.output_dir / "audit.json").write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        report_markdown(canonical), encoding="utf-8"
    )
    write_csv(args.output_dir / "dataset_manifest.csv", columns_manifest)
    write_csv(args.output_dir / "method_summary.csv", canonical["method_rows"])
    write_csv(args.output_dir / "method_column_comparisons.csv", canonical["comparison_rows"])
    write_csv(
        args.output_dir / f"action_{len(canonical['action_rows'])}_testability.csv",
        canonical["action_rows"],
    )
    write_csv(args.output_dir / "standalone_action_roundtrips.csv", action_tests_first)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "exact_pass": canonical["methods"]["exact_pass_on_all_materialized_external_columns"],
                "all_86": canonical["methods"]["all_86_exactly_composable"],
                "deterministic": deterministic,
                "core_sha256": hashes[0],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    failed_policy_actions = [
        row["canonical_name"]
        for row in canonical["action_rows"]
        if row["runtime_testability"] == "standalone_registered"
        and row["realdata_status"] == "fail"
    ]
    if not deterministic or failed_policy_actions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
