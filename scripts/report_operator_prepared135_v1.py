#!/usr/bin/env python3
"""Audit and report the prepared-selector whole-column campaign.

This is deliberately a post-selection join.  It never fits a model, changes a
threshold, or selects a candidate from oracle outcomes.  The native harness and
the frozen 85-algorithm/97-operator oracle use different stored-byte contexts,
so their results are kept side by side and native-versus-oracle regret is null.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


SCHEMA = "operator-prepared135-post-selection-report-v1"
EXPECTED_COLUMNS = 3519
EXPECTED_DATASETS = 135
EXPECTED_ALGORITHMS = 85
EXPECTED_OPERATORS = 97
EXPECTED_ORACLE_SHA256 = (
    "6adef4003a23ee14253d2c04b844978ec340c056863ebabf61c11ecd632b35ba"
)
EXPECTED_SPLIT_SHA256 = (
    "d0545713f3d56aed5d9934814ca4144ab9cc8e968af032a3cffb4741cb4676a4"
)
EXPECTED_CATALOG_SHA256 = (
    "568f2422b415d2869712a4a0e5f8edb37205555d92ecfe3fc0b36946c1d56326"
)
EXPECTED_TRAINING_ROWS_SHA256 = (
    "4847e00c1edfb4f13f41a8594b537722f84829767c5e2981ee94c4efe711113e"
)
TARGETS = ("space", "encode", "decode", "balanced")
METHODS = ("per_block", "reuse")
CANDIDATES = (
    "raw",
    "zstd1", "zstd3", "zstd9", "lz4frame", "zlib6",
    "huffman_a077", "rans_a080",
    "columnize_zstd3", "columnize_lz4", "columnize_zlib6",
    "xor_zstd3", "xor_lz4", "xor_zlib6",
    "float_deconstruct_zstd3", "float_deconstruct_lz4",
    "float_deconstruct_zstd3_lz4",
    "for_bitpack_zstd3", "for_bitpack_lz4",
    "a048_planes_zstd3", "a048_planes_lz4",
    "a048_planes_zstd3_lz4",
    "split_zstd3_lz4", "split_lz4_zstd3",
)
COMPRESSING_CANDIDATES = frozenset(CANDIDATES) - {"raw"}
NEW_BYTE_CONTEXT = (
    "native_graph_v1:r1b1:three_root:operator_recommend135_candidates_v1"
)
ALGORITHM_BYTE_CONTEXT = (
    "native_graph_v1:baseline86_fixed_profiles_20260919_v3:two_root"
)
OPERATOR_BYTE_CONTEXT = "operator97:o2r1:zstd3_common_sink"
IDENTITY_FIELDS = (
    "dataset_id", "column_id", "column", "dtype", "manifest_path",
    "manifest_sha256", "original_numeric_bytes", "valid_value_count",
)


class ReportError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read JSON {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{number} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read JSONL {path}: {error}") from error
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def nlohmann_compact_bytes(value: Any) -> bytes:
    """Match default nlohmann::json object ordering and compact UTF-8 dump."""
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def integer(value: Any, label: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{label} must be an integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def timing(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == {"cpu_ns", "wall_ns"},
            f"{label} must contain exactly cpu_ns and wall_ns")
    return {
        "cpu_ns": integer(value["cpu_ns"], f"{label}.cpu_ns"),
        "wall_ns": integer(value["wall_ns"], f"{label}.wall_ns"),
    }


def target_ids(value: Any, label: str) -> dict[str, str]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == set(TARGETS),
            f"{label} must contain exactly {list(TARGETS)}")
    result: dict[str, str] = {}
    for target in TARGETS:
        candidate = value[target]
        require(isinstance(candidate, str) and candidate in COMPRESSING_CANDIDATES,
                f"{label}.{target} has an unknown or raw candidate: {candidate!r}")
        result[target] = candidate
    return result


def key(row: dict[str, Any], label: str) -> tuple[str, str]:
    dataset = row.get("dataset_id")
    column = row.get("column_id")
    require(isinstance(dataset, str) and dataset,
            f"{label}.dataset_id must be a nonempty string")
    require(isinstance(column, str) and column,
            f"{label}.column_id must be a nonempty string")
    return dataset, column


def index_unique(rows: Iterable[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), f"{label} contains a non-object")
        item_key = key(row, label)
        require(item_key not in result, f"duplicate {label} key: {item_key}")
        result[item_key] = row
    return result


def sum_timing(destination: dict[str, int], source: dict[str, int]) -> None:
    destination["cpu_ns"] += source["cpu_ns"]
    destination["wall_ns"] += source["wall_ns"]


def empty_timing() -> dict[str, int]:
    return {"cpu_ns": 0, "wall_ns": 0}


def empty_scope() -> dict[str, Any]:
    return {
        "columns": 0,
        "datasets": set(),
        "blocks": 0,
        "original_numeric_bytes": 0,
        "valid_value_count": 0,
        "oracle_best_observed_pipeline_compressed_bytes": 0,
        "model_storage_bytes_charged_per_column_sum": 0,
        "fresh_process_frame_decodes": 0,
        "source_receipt_modes": Counter(),
        "methods": {
            method: {
                "plan_bytes": 0,
                "decision": empty_timing(),
                "four_goal_batch_e2e": empty_timing(),
                "objectives": {
                    target: {
                        "frame_bytes": 0,
                        "column_container_bytes": 0,
                        "column_container_construction": empty_timing(),
                        "decision_plus_encode_component_sum": empty_timing(),
                        "decode": empty_timing(),
                        "selected_candidate_counts": Counter(),
                    }
                    for target in TARGETS
                },
            }
            for method in METHODS
        },
        "candidate_measurements": {},
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def serialise_scope(scope: dict[str, Any]) -> dict[str, Any]:
    raw = scope["original_numeric_bytes"]
    oracle_bytes = scope["oracle_best_observed_pipeline_compressed_bytes"]
    models = scope["model_storage_bytes_charged_per_column_sum"]
    output = {
        name: value for name, value in scope.items()
        if name not in {"datasets", "source_receipt_modes", "methods",
                        "candidate_measurements"}
    }
    output["dataset_count"] = len(scope["datasets"])
    output["source_receipt_modes"] = dict(sorted(scope["source_receipt_modes"].items()))
    output["methods"] = {}
    for method, values in scope["methods"].items():
        method_out = {
            "plan_bytes": values["plan_bytes"],
            "model_storage_bytes_charged_per_column_sum": models,
            "decision": values["decision"],
            "four_goal_batch_e2e": values["four_goal_batch_e2e"],
            "objectives": {},
        }
        for target, objective in values["objectives"].items():
            frames = objective["frame_bytes"]
            containers = objective["column_container_bytes"]
            with_container = frames + containers
            system = with_container + values["plan_bytes"] + models
            method_out["objectives"][target] = {
                "byte_context_id": NEW_BYTE_CONTEXT,
                "complete_frame_bytes": frames,
                "column_container_bytes": containers,
                "bytes_with_column_container": with_container,
                "column_container_construction": objective[
                    "column_container_construction"
                ],
                "plan_bytes_reported_separately": values["plan_bytes"],
                "model_storage_bytes_reported_separately": models,
                "bytes_with_conservative_model_plan_charge": system,
                "frame_compression_ratio": ratio(frames, raw),
                "complete_system_space_ratio": ratio(system, raw),
                "decision_plus_encode_component_sum": objective[
                    "decision_plus_encode_component_sum"
                ],
                "decode": objective["decode"],
                "selected_candidate_counts": dict(sorted(
                    objective["selected_candidate_counts"].items()
                )),
                "native_vs_oracle_regret": None,
                "same_source_complete_frame_reference_multiple": ratio(
                    frames, oracle_bytes
                ),
                "conservative_selector_system_to_oracle_frame_reference_multiple": ratio(
                    system, oracle_bytes
                ),
            }
        output["methods"][method] = method_out
    output["candidate_measurements"] = dict(sorted(
        scope["candidate_measurements"].items()
    ))
    return output


def validate_oracle(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    require(sha256(path) == EXPECTED_ORACLE_SHA256,
            "oracle SHA-256 is not the frozen current135 oracle")
    rows = load_jsonl(path)
    require(len(rows) == EXPECTED_COLUMNS,
            f"oracle must have {EXPECTED_COLUMNS} rows, found {len(rows)}")
    indexed = index_unique(rows, "oracle")
    require(len({row["dataset_id"] for row in rows}) == EXPECTED_DATASETS,
            f"oracle must cover {EXPECTED_DATASETS} datasets")
    for row in rows:
        item_key = key(row, "oracle")
        require(row.get("schema") == "current135-column-oracle-ground-truth-v1",
                f"oracle {item_key} schema mismatch")
        require(row.get("algorithm_candidate_count") == EXPECTED_ALGORITHMS,
                f"oracle {item_key} algorithm candidate count mismatch")
        require(row.get("joint_operator_algorithm_measured") is False and
                row.get("joint_global_optimal") is False,
                f"oracle {item_key} incorrectly claims a joint optimum")
        require(row.get("best_operator_compression_ns") is None and
                row.get("best_operator_decompression_ns") is None,
                f"oracle {item_key} operator timing must be null")
        require(row.get("best_observed_pipeline_kind") in
                {"algorithm", "operator_zstd3"},
                f"oracle {item_key} best observed kind is unknown")
        for field in IDENTITY_FIELDS:
            require(field in row, f"oracle {item_key} lacks {field}")
    return rows, indexed


def validate_split(path: Path, bindings: dict[str, Any]) -> dict[str, Any]:
    require(sha256(path) == EXPECTED_SPLIT_SHA256,
            "split SHA-256 is not the frozen source-group split")
    require(bindings.get("split_sha256") == EXPECTED_SPLIT_SHA256,
            "run is not bound to the frozen source-group split")
    split = load_json(path)
    require(split.get("status") == "frozen_before_selector_outcomes",
            "split was not frozen before selector outcomes")
    assignments = split.get("dataset_assignments")
    require(isinstance(assignments, dict) and len(assignments) == EXPECTED_DATASETS,
            "split must assign exactly 135 datasets")
    return split


def validate_columns(path: Path, protocol_path: Path, bindings: dict[str, Any],
                     oracle_sha256: str,
                     oracle: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    columns_sha256 = sha256(path)
    require(bindings.get("columns_sha256") == columns_sha256,
            "run is not bound to the supplied frozen COLUMNS.json")
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == "operator-prepared135-protocol-v1",
            "PROTOCOL.json schema mismatch")
    require(protocol.get("columns_sha256") == columns_sha256,
            "PROTOCOL.json does not bind the supplied COLUMNS.json")
    require(protocol.get("source_bindings", {}).get(
        "analysis_outputs/column_oracle_ground_truth_20261020_v1/current135_column_oracle_ground_truth.jsonl"
    ) == oracle_sha256,
            "PROTOCOL.json does not bind the supplied frozen oracle")
    require(protocol.get("model", {}).get("confirm_labels_in_fit") is False and
            protocol.get("model", {}).get("no_threshold_or_hyperparameter_changes") is True,
            "PROTOCOL.json model isolation contract mismatch")
    value = load_json(path)
    require(value.get("schema") == "operator-prepared135-source-columns-v1",
            "COLUMNS.json schema mismatch")
    require(value.get("scope") != "witness",
            "COLUMNS.json is a witness scope, not the formal 3,519-column scope")
    require(value.get("oracle_reference_jsonl_sha256") == oracle_sha256,
            "COLUMNS.json does not bind the frozen oracle SHA-256")
    rows = value.get("columns")
    require(isinstance(rows, list) and len(rows) == EXPECTED_COLUMNS,
            "COLUMNS.json must contain exactly 3,519 columns")
    indexed = index_unique(rows, "COLUMNS.json")
    require(set(indexed) == set(oracle),
            "COLUMNS.json and oracle have different dataset/column keys")
    for item_key, column in indexed.items():
        reference = oracle[item_key]
        for field in IDENTITY_FIELDS:
            require(column.get(field) == reference.get(field),
                    f"COLUMNS/oracle mismatch for {item_key} field {field}")
    return indexed


def find_models(root: Path, explicit: Path | None, expected_sha: str) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend((
        root / "MODELS.json",
        root / "models" / "MODELS.json",
        root.parent / "models" / "MODELS.json",
        root.parent / "MODELS.json",
    ))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and sha256(candidate) == expected_sha:
            return candidate
    raise ReportError(
        "cannot locate the MODELS.json bound by BINDINGS.json; pass --models"
    )


def validate_models(path: Path, bindings: dict[str, Any], split: dict[str, Any],
                    training_rows_path: Path) -> dict[str, Any]:
    require(sha256(path) == bindings.get("models_sha256"),
            "MODELS.json does not match the run binding")
    value = load_json(path)
    require(value.get("schema") == "operator-prepared135-models-v1",
            "MODELS.json schema mismatch")
    require(value.get("split_sha256") == EXPECTED_SPLIT_SHA256,
            "models use a different split")
    require(value.get("source_rows_sha256") == EXPECTED_TRAINING_ROWS_SHA256,
            "models do not use the frozen historical 360-row source")
    require(value.get("catalog_sha256") == EXPECTED_CATALOG_SHA256,
            "model candidate catalog mismatch")
    require(value.get("confirm_labels_in_fit") is False,
            "MODELS.json does not prohibit confirm labels in fitting")
    require(value.get("hyperparameters_changed") is False,
            "MODELS.json says hyperparameters changed")
    require(sha256(training_rows_path) == EXPECTED_TRAINING_ROWS_SHA256,
            "historical training ROWS.json is not the frozen 360-row source")
    training_rows = load_json(training_rows_path)
    require(isinstance(training_rows, list) and len(training_rows) == 360,
            "historical training source must contain exactly 360 rows")
    require(all(isinstance(row, dict) for row in training_rows),
            "historical training rows must be objects")
    training_sample_ids = [row.get("sample_id") for row in training_rows]
    require(all(isinstance(sample, str) and sample for sample in training_sample_ids)
            and len(set(training_sample_ids)) == 360,
            "historical training sample IDs must be unique")
    models = value.get("models")
    expected_keys = {"dev0", "dev1", "dev2", "dev3", "dev4", "confirm"}
    require(isinstance(models, dict) and set(models) == expected_keys,
            "MODELS.json must contain dev0..dev4 and confirm")
    assignments = split["dataset_assignments"]
    confirm_legacy_groups = {
        row["group_id"] for row in training_rows
        if assignments[row["dataset_id"]]["split"] == "confirm"
    }
    all_samples: set[str] = set()
    for model_key, model in models.items():
        require(isinstance(model, dict), f"model {model_key} binding is not an object")
        rows = integer(model.get("training_row_count"),
                       f"model {model_key}.training_row_count", 1)
        samples = model.get("training_sample_ids")
        datasets = model.get("training_dataset_ids")
        groups = model.get("training_source_groups")
        held_out = None if model_key == "confirm" else int(model_key[3:])
        excluded_legacy_groups = set(confirm_legacy_groups)
        if held_out is not None:
            excluded_legacy_groups.update(
                row["group_id"] for row in training_rows
                if assignments[row["dataset_id"]]["split"] == "development"
                and assignments[row["dataset_id"]]["dev_fold"] == held_out
            )
        expected_rows = [
            row for row in training_rows
            if assignments[row["dataset_id"]]["split"] == "development"
            and (held_out is None or
                 assignments[row["dataset_id"]]["dev_fold"] != held_out)
            and row["group_id"] not in excluded_legacy_groups
        ]
        expected_samples = [row["sample_id"] for row in expected_rows]
        expected_datasets = sorted({row["dataset_id"] for row in expected_rows})
        expected_groups = sorted({
            assignments[row["dataset_id"]]["source_identity_sha256"]
            for row in expected_rows
        })
        require(isinstance(samples, list) and len(samples) == rows and
                len(set(samples)) == rows,
                f"model {model_key} training sample receipt mismatch")
        require(isinstance(datasets, list) and len(set(datasets)) == len(datasets),
                f"model {model_key} training datasets receipt mismatch")
        require(isinstance(groups, list) and len(set(groups)) == len(groups),
                f"model {model_key} training groups receipt mismatch")
        require(samples == expected_samples and rows == len(expected_rows),
                f"model {model_key} does not use the exact source-group-purged development rows")
        require(sorted(datasets) == expected_datasets and sorted(groups) == expected_groups,
                f"model {model_key} dataset/source-group receipt differs from exact recomputation")
        require(sorted(model.get("excluded_legacy_groups", [])) ==
                sorted(excluded_legacy_groups),
                f"model {model_key} excluded legacy-group receipt mismatch")
        for dataset in datasets:
            assignment = assignments.get(dataset)
            require(assignment is not None and assignment.get("split") == "development",
                    f"model {model_key} contains non-development dataset {dataset}")
            if model_key.startswith("dev"):
                require(assignment.get("dev_fold") != held_out,
                        f"model {model_key} contains its held-out fold dataset {dataset}")
        require(all(group in {
            assignments[dataset]["source_identity_sha256"] for dataset in datasets
        } for group in groups),
                f"model {model_key} has a source group outside its dataset receipt")
        all_samples.update(samples)
    require(len(all_samples) <= 336,
            "model receipts exceed the 336 development historical rows")
    require(models["confirm"].get("training_row_count") == 328,
            "confirm model must contain 328 rows after purging four RealEstate2 and four spx rows")
    return value


def validate_manifest_if_available(result: dict[str, Any],
                                   result_path: Path) -> str:
    column = result["column"]
    manifest_path = Path(column["manifest_path"])
    if not manifest_path.is_file():
        return "runtime_receipt_bound_manifest_not_locally_readable"
    require(sha256(manifest_path) == column["manifest_sha256"],
            f"canonical source manifest SHA mismatch for {result_path}")
    manifest = load_json(manifest_path)
    require(manifest.get("schema") ==
            "native-baseline86-scaled-column-segment-manifest-v1",
            f"canonical manifest schema mismatch for {result_path}")
    require(manifest_path.stem == column["column_id"],
            f"canonical manifest filename/column ID mismatch for {result_path}")
    require(manifest.get("dataset_id") == column["dataset_id"],
            f"canonical manifest dataset mismatch for {result_path}")
    identity = manifest.get("column_identity")
    require(isinstance(identity, dict) and
            identity.get("dtype") == column["dtype"] and
            identity.get("column") == column["column"],
            f"canonical manifest column identity mismatch for {result_path}")
    source_blocks = manifest.get("blocks")
    blocks = result["blocks"]
    require(isinstance(source_blocks, list) and len(source_blocks) == len(blocks),
            f"canonical manifest block coverage mismatch for {result_path}")
    comparisons = (
        ("block_id", "block_id"),
        ("entity_id", "entity_id"),
        ("values_sha256", "source_values_sha256"),
        ("missing_bitmap_sha256", "source_missing_sha256"),
        ("row_count", "row_count"),
        ("valid_count", "valid_count"),
        ("original_numeric_bytes", "original_numeric_bytes"),
    )
    for position, (source, measured) in enumerate(zip(source_blocks, blocks)):
        require(source.get("dataset_id") == column["dataset_id"] and
                source.get("dtype") == column["dtype"] and
                source.get("column") == column["column"],
                f"canonical manifest block {position} identity mismatch for {result_path}")
        require(measured["block_index"] == position,
                f"block order mismatch for {result_path}")
        for source_field, measured_field in comparisons:
            require(source.get(source_field) == measured.get(measured_field),
                    f"manifest block {position} {source_field} mismatch for {result_path}")
    return "canonical_manifest_reopened_and_matched"


def validate_candidate(value: Any, candidate_id: str, block_label: str,
                       encoder_pid: int) -> dict[str, Any]:
    require(isinstance(value, dict), f"{block_label} candidate {candidate_id} is not an object")
    frame_bytes = integer(value.get("frame_bytes"),
                          f"{block_label}.{candidate_id}.frame_bytes", 1)
    frame_sha = value.get("frame_sha256")
    require(isinstance(frame_sha, str) and len(frame_sha) == 64,
            f"{block_label}.{candidate_id} frame SHA is invalid")
    decoded = timing(value.get("decode"), f"{block_label}.{candidate_id}.decode")
    encoded = None
    if "encode" in value:
        encoded = timing(value["encode"], f"{block_label}.{candidate_id}.encode")
    fixed = None
    if "fixed_encode" in value:
        fixed = timing(value["fixed_encode"],
                       f"{block_label}.{candidate_id}.fixed_encode")
    witness = value.get("fresh_decode")
    require(isinstance(witness, dict) and witness.get("verified") is True and
            witness.get("frame_only_decode") is True and
            witness.get("frame_sha256") == frame_sha,
            f"{block_label}.{candidate_id} fresh-process decode receipt mismatch")
    decode_pid = integer(witness.get("decode_pid"),
                         f"{block_label}.{candidate_id}.decode_pid", 1)
    require(decode_pid != encoder_pid,
            f"{block_label}.{candidate_id} fresh decode reused encoder process")
    return {
        "frame_bytes": frame_bytes,
        "frame_sha256": frame_sha,
        "encode": encoded,
        "fixed_encode": fixed,
        "decode": decoded,
        "fresh_decode": witness,
    }


def oracle_projection(oracle: dict[str, Any]) -> dict[str, Any]:
    observed_kind = oracle["best_observed_pipeline_kind"]
    observed_context = (
        ALGORITHM_BYTE_CONTEXT if observed_kind == "algorithm"
        else OPERATOR_BYTE_CONTEXT
    )
    observed_encode = (
        oracle["best_algorithm_compression_ns"]
        if observed_kind == "algorithm" else None
    )
    observed_decode = (
        oracle["best_algorithm_decompression_ns"]
        if observed_kind == "algorithm" else None
    )
    return {
        "post_selection_read_only": True,
        "algorithm_pool": {
            "candidate_count": EXPECTED_ALGORITHMS,
            "id": oracle["best_algorithm"],
            "compressed_bytes": oracle["best_algorithm_compressed_bytes"],
            "compression_ratio": oracle["best_algorithm_compression_ratio"],
            "compression_ns": oracle["best_algorithm_compression_ns"],
            "decompression_ns": oracle["best_algorithm_decompression_ns"],
            "timing_contract": oracle["best_algorithm_timing_contract"],
            "byte_context_id": ALGORITHM_BYTE_CONTEXT,
        },
        "operator_pool": {
            "candidate_count": EXPECTED_OPERATORS,
            "id": oracle["best_operator_id"],
            "name": oracle["best_operator_name"],
            "backend": oracle["best_operator_backend"],
            "compressed_bytes": oracle["best_operator_compressed_bytes"],
            "compression_ratio": oracle["best_operator_compression_ratio"],
            "compression_ns": None,
            "decompression_ns": None,
            "timing_status": oracle["best_operator_timing_status"],
            "byte_context_id": OPERATOR_BYTE_CONTEXT,
        },
        "best_observed_pipeline": {
            "kind": observed_kind,
            "id": oracle["best_observed_pipeline_id"],
            "name": oracle["best_observed_pipeline_name"],
            "compressed_bytes": oracle["best_observed_pipeline_compressed_bytes"],
            "compression_ratio": oracle["best_observed_pipeline_compression_ratio"],
            "compression_ns": observed_encode,
            "decompression_ns": observed_decode,
            "byte_context_id": observed_context,
        },
        "joint_operator_algorithm_measured": False,
        "joint_global_optimal": False,
    }


def validate_result(result: dict[str, Any], result_path: Path,
                    expected_column: dict[str, Any], oracle: dict[str, Any],
                    assignment: dict[str, Any], bindings: dict[str, Any],
                    models: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    item_key = key(expected_column, "expected column")
    label = f"result {item_key}"
    require(result.get("schema") == "operator-prepared135-column-v1" and
            result.get("status") == "strict_verified",
            f"{label} is not strict_verified")
    require(result.get("bindings") == bindings, f"{label} bindings mismatch")
    require(result.get("column") == expected_column,
            f"{label} does not exactly match frozen COLUMNS.json")
    for field in IDENTITY_FIELDS:
        require(expected_column.get(field) == oracle.get(field),
                f"{label} oracle identity mismatch on {field}")
    require(result.get("split") == assignment.get("split"),
            f"{label} split mismatch")
    require(result.get("source_identity_sha256") ==
            assignment.get("source_identity_sha256"),
            f"{label} source group mismatch")
    require(result.get("dev_fold") == assignment.get("dev_fold"),
            f"{label} dev fold mismatch")
    expected_model_key = (
        "confirm" if assignment["split"] == "confirm"
        else f"dev{assignment['dev_fold']}"
    )
    require(result.get("model_key") == expected_model_key,
            f"{label} model key mismatch")
    model_binding = models["models"][expected_model_key]
    require(result.get("model_sha256") == model_binding.get("sha256"),
            f"{label} model SHA mismatch")
    model_bytes = integer(result.get("model_storage_bytes_charged_per_column"),
                          f"{label}.model_storage_bytes_charged_per_column", 1)
    require(model_bytes == model_binding.get("model_json_bytes"),
            f"{label} model byte charge mismatch")
    require(result.get("reference_prepared_full_ranking_equal") is True,
            f"{label} PreparedModel/reference ranking mismatch")
    initial = result.get("initial_selection")
    require(isinstance(initial, dict) and initial.get("validation_passes") == 1,
            f"{label} did not perform exactly one selector input validation")
    for name in ("feature", "eligibility", "model_and_choices", "total"):
        timing(initial.get(name), f"{label}.initial_selection.{name}")
    initial_total = timing(initial.get("total"), f"{label}.initial_selection.total")
    initial_plan_time = timing(result.get("initial_plan_construction"),
                               f"{label}.initial_plan_construction")
    initial_ids = target_ids(initial.get("selected_ids"),
                             f"{label}.initial_selection.selected_ids")
    plan = result.get("plan")
    require(isinstance(plan, dict) and
            plan.get("schema") == "operator-prepared135-column-plan-v1",
            f"{label} plan schema mismatch")
    require(plan.get("selected_ids") == initial_ids and
            plan.get("source_manifest_sha256") == expected_column["manifest_sha256"] and
            plan.get("dtype") == expected_column["dtype"] and
            plan.get("catalog_sha256") == EXPECTED_CATALOG_SHA256 and
            plan.get("model_sha256") == result["model_sha256"],
            f"{label} frozen reuse plan binding mismatch")
    reuse_plan_bytes = integer(result.get("reuse_plan_bytes"),
                               f"{label}.reuse_plan_bytes", 1)
    require(reuse_plan_bytes == len(nlohmann_compact_bytes(plan)),
            f"{label} reuse plan byte count mismatch")
    encoder_pid = integer(result.get("encoder_pid"), f"{label}.encoder_pid", 1)
    blocks = result.get("blocks")
    require(isinstance(blocks, list) and blocks, f"{label} has no blocks")
    representative = integer(plan.get("representative_block_index"),
                             f"{label}.representative_block_index")
    require(representative < len(blocks), f"{label} representative block out of range")
    source_mode = validate_manifest_if_available(result, result_path)
    source_bytes = 0
    valid_values = 0
    fresh_decodes = 0
    column_methods = {
        method: {
            "plan_bytes": reuse_plan_bytes if method == "reuse" else 0,
            "decision": empty_timing(),
            "four_goal_batch_e2e": empty_timing(),
            "objectives": {
                target: {
                    "frame_bytes": 0,
                    "column_container_bytes": 0,
                    "column_container_construction": empty_timing(),
                    "decision_plus_encode_component_sum": empty_timing(),
                    "decode": empty_timing(),
                    "selected_candidate_counts": Counter(),
                }
                for target in TARGETS
            },
        }
        for method in METHODS
    }
    candidate_totals: dict[str, Any] = {}
    for position, block in enumerate(blocks):
        block_label = f"{label}.blocks[{position}]"
        require(isinstance(block, dict) and block.get("block_index") == position,
                f"{block_label} order/index mismatch")
        source_bytes += integer(block.get("original_numeric_bytes"),
                                f"{block_label}.original_numeric_bytes")
        block_valid = integer(block.get("valid_count"), f"{block_label}.valid_count")
        valid_values += block_valid
        for source_hash in ("source_values_sha256", "source_missing_sha256"):
            value = block.get(source_hash)
            require(isinstance(value, str) and len(value) == 64,
                    f"{block_label}.{source_hash} is invalid")
        per_ids = target_ids(block.get("per_block_selected_ids"),
                             f"{block_label}.per_block_selected_ids")
        reuse_ids = target_ids(block.get("reuse_selected_ids"),
                               f"{block_label}.reuse_selected_ids")
        adaptations = block.get("empty_domain_adaptations")
        require(isinstance(adaptations, list) and
                len(set(adaptations)) == len(adaptations) and
                all(target in TARGETS for target in adaptations),
                f"{block_label} empty-domain adaptations are invalid")
        for target in TARGETS:
            if target in adaptations:
                require(block_valid == 0 and reuse_ids[target] == "zstd3",
                        f"{block_label} invalid empty-domain adaptation for {target}")
            else:
                require(reuse_ids[target] == initial_ids[target],
                        f"{block_label} changed the frozen reuse plan for {target}")
        per_plan_bytes = integer(block.get("per_block_plan_bytes"),
                                 f"{block_label}.per_block_plan_bytes", 1)
        column_methods["per_block"]["plan_bytes"] += per_plan_bytes
        per_selection = timing(block.get("selection"), f"{block_label}.selection")
        reuse_guard = timing(block.get("reuse_guard"), f"{block_label}.reuse_guard")
        sum_timing(column_methods["per_block"]["decision"], per_selection)
        sum_timing(column_methods["reuse"]["decision"], reuse_guard)
        per_batch = timing(block.get("four_goal_batch_per_block_e2e"),
                           f"{block_label}.four_goal_batch_per_block_e2e")
        reuse_batch = timing(block.get("four_goal_batch_reuse_e2e"),
                             f"{block_label}.four_goal_batch_reuse_e2e")
        sum_timing(column_methods["per_block"]["four_goal_batch_e2e"], per_batch)
        sum_timing(column_methods["reuse"]["four_goal_batch_e2e"], reuse_batch)
        per_components = block.get("per_goal_per_block_component_sum")
        reuse_components = block.get("per_goal_reuse_component_sum")
        require(isinstance(per_components, dict) and set(per_components) == set(TARGETS),
                f"{block_label} per-block per-goal timing mismatch")
        require(isinstance(reuse_components, dict) and set(reuse_components) == set(TARGETS),
                f"{block_label} reuse per-goal timing mismatch")
        candidates = block.get("candidates")
        require(isinstance(candidates, dict), f"{block_label}.candidates must be an object")
        require(set(candidates).issubset(COMPRESSING_CANDIDATES) and
                {"zstd3", "lz4frame"}.issubset(candidates),
                f"{block_label} candidate measurement scope is invalid")
        measured: dict[str, dict[str, Any]] = {}
        for candidate_id, candidate in candidates.items():
            measured[candidate_id] = validate_candidate(
                candidate, candidate_id, block_label, encoder_pid
            )
            fresh_decodes += 1
            aggregate = candidate_totals.setdefault(candidate_id, {
                "measurements": 0,
                "frame_bytes_sum": 0,
                "encode_samples": 0,
                "encode": empty_timing(),
                "fixed_encode_samples": 0,
                "fixed_encode": empty_timing(),
                "decode": empty_timing(),
                "fresh_process_frame_decodes": 0,
            })
            aggregate["measurements"] += 1
            aggregate["frame_bytes_sum"] += measured[candidate_id]["frame_bytes"]
            sum_timing(aggregate["decode"], measured[candidate_id]["decode"])
            if measured[candidate_id]["encode"] is not None:
                aggregate["encode_samples"] += 1
                sum_timing(aggregate["encode"], measured[candidate_id]["encode"])
            if measured[candidate_id]["fixed_encode"] is not None:
                aggregate["fixed_encode_samples"] += 1
                sum_timing(aggregate["fixed_encode"], measured[candidate_id]["fixed_encode"])
            aggregate["fresh_process_frame_decodes"] += 1
        for method, selected, components in (
            ("per_block", per_ids, per_components),
            ("reuse", reuse_ids, reuse_components),
        ):
            for target in TARGETS:
                candidate_id = selected[target]
                require(candidate_id in measured,
                        f"{block_label} lacks selected {method}/{target} frame")
                candidate = measured[candidate_id]
                require(candidate.get("encode") is not None,
                        f"{block_label} lacks selected {method}/{target} encode timing")
                objective = column_methods[method]["objectives"][target]
                objective["frame_bytes"] += candidate["frame_bytes"]
                objective["selected_candidate_counts"][candidate_id] += 1
                sum_timing(objective["decode"], candidate["decode"])
                sum_timing(
                    objective["decision_plus_encode_component_sum"],
                    timing(components[target],
                           f"{block_label}.{method}.{target}.component_sum"),
                )
    require(source_bytes == expected_column["original_numeric_bytes"] ==
            result.get("source_bytes_verified"),
            f"{label} original byte coverage mismatch")
    require(valid_values == expected_column["valid_value_count"] ==
            result.get("source_valid_values_verified"),
            f"{label} valid-value coverage mismatch")
    require(fresh_decodes == result.get("independent_frame_decodes"),
            f"{label} fresh decode count mismatch")
    setup = {
        "cpu_ns": initial_total["cpu_ns"] + initial_plan_time["cpu_ns"],
        "wall_ns": initial_total["wall_ns"] + initial_plan_time["wall_ns"],
    }
    sum_timing(column_methods["reuse"]["decision"], setup)
    sum_timing(column_methods["reuse"]["four_goal_batch_e2e"], setup)
    for target in TARGETS:
        sum_timing(
            column_methods["reuse"]["objectives"][target][
                "decision_plus_encode_component_sum"
            ],
            setup,
        )
    whole_column = result.get("whole_column")
    require(isinstance(whole_column, dict) and set(whole_column) == set(METHODS),
            f"{label}.whole_column must contain per_block and reuse")
    for method in METHODS:
        measured = whole_column[method]
        computed = column_methods[method]
        require(isinstance(measured, dict), f"{label}.whole_column.{method} is invalid")
        expected_setup = setup if method == "reuse" else empty_timing()
        require(measured.get("selection_cpu_ns") == computed["decision"]["cpu_ns"] and
                measured.get("selection_wall_ns") == computed["decision"]["wall_ns"],
                f"{label}.whole_column.{method} selection total mismatch")
        require(measured.get("initial_selection_and_plan_cpu_ns") ==
                expected_setup["cpu_ns"] and
                measured.get("initial_selection_and_plan_wall_ns") ==
                expected_setup["wall_ns"],
                f"{label}.whole_column.{method} setup charge mismatch")
        require(measured.get("plan_storage_bytes") == computed["plan_bytes"] and
                measured.get("model_storage_bytes") == model_bytes,
                f"{label}.whole_column.{method} model/plan byte charge mismatch")
        goals = measured.get("goals")
        require(isinstance(goals, dict) and set(goals) == set(TARGETS),
                f"{label}.whole_column.{method}.goals mismatch")
        for target in TARGETS:
            objective = computed["objectives"][target]
            goal = goals[target]
            ordered_frames = []
            for block in blocks:
                candidate_id = block[
                    "reuse_selected_ids" if method == "reuse"
                    else "per_block_selected_ids"
                ][target]
                frame = block["candidates"][candidate_id]
                ordered_frames.append({
                    "block_index": block["block_index"],
                    "block_id": block["block_id"],
                    "entity_id": block["entity_id"],
                    "row_count": block["row_count"],
                    "valid_count": block["valid_count"],
                    "candidate_id": candidate_id,
                    "frame_bytes": frame["frame_bytes"],
                    "frame_sha256": frame["frame_sha256"],
                })
            container = {
                "schema": "operator-prepared135-column-container-v1",
                "dataset_id": expected_column["dataset_id"],
                "column_id": expected_column["column_id"],
                "column": expected_column["column"],
                "dtype": expected_column["dtype"],
                "source_manifest_sha256": expected_column["manifest_sha256"],
                "policy": method,
                "objective": target,
                "ordered_frames": ordered_frames,
            }
            serialized = nlohmann_compact_bytes(container)
            container_sha256 = hashlib.sha256(serialized).hexdigest()
            packaging = timing(
                goal.get("column_container_construction"),
                f"{label}.whole_column.{method}.{target}.column_container_construction",
            )
            objective["column_container_bytes"] = len(serialized)
            sum_timing(objective["column_container_construction"], packaging)
            sum_timing(objective["decision_plus_encode_component_sum"], packaging)
            sum_timing(computed["four_goal_batch_e2e"], packaging)
            expected_goal = {
                "complete_frame_bytes": objective["frame_bytes"],
                "encode_component_sum_cpu_ns": objective[
                    "decision_plus_encode_component_sum"
                ]["cpu_ns"],
                "encode_component_sum_wall_ns": objective[
                    "decision_plus_encode_component_sum"
                ]["wall_ns"],
                "decode_cpu_ns": objective["decode"]["cpu_ns"],
                "decode_wall_ns": objective["decode"]["wall_ns"],
                "column_container_bytes": len(serialized),
                "column_container_sha256": container_sha256,
                "column_container": container,
                "column_container_construction": packaging,
                "bytes_with_column_container": (
                    objective["frame_bytes"] + len(serialized)
                ),
                "bytes_with_conservative_model_plan_charge": (
                    objective["frame_bytes"] + len(serialized) +
                    computed["plan_bytes"] + model_bytes
                ),
            }
            require(goal == expected_goal,
                    f"{label}.whole_column.{method}.{target} aggregate mismatch")
        require(measured.get("four_goal_batch_e2e_cpu_ns") ==
                computed["four_goal_batch_e2e"]["cpu_ns"] and
                measured.get("four_goal_batch_e2e_wall_ns") ==
                computed["four_goal_batch_e2e"]["wall_ns"],
                f"{label}.whole_column.{method} four-goal batch mismatch")
    contract = result.get("timing_contract")
    require(isinstance(contract, dict) and contract.get("warmups") == 1 and
            contract.get("repeats") == 3 and
            contract.get("all_four_targets_computed") is True and
            contract.get("reuse_initial_selection_charged_once_per_column") ==
            "whole_column.reuse includes initial_selection + initial_plan_construction once" and
            contract.get("whole_column_packaging") ==
            "goal-specific complete column/entity/block/frame index serialized and charged once; its measured construction is added to component sums and whole-column batch totals",
            f"{label} timing contract mismatch")
    oracle_bytes = integer(
        oracle.get("best_observed_pipeline_compressed_bytes"),
        f"oracle {item_key}.best_observed_pipeline_compressed_bytes", 1,
    )
    detailed = {
        "dataset_id": item_key[0],
        "column_id": item_key[1],
        "column": expected_column["column"],
        "dtype": expected_column["dtype"],
        "manifest_sha256": expected_column["manifest_sha256"],
        "original_numeric_bytes": source_bytes,
        "valid_value_count": valid_values,
        "split": assignment["split"],
        "dev_fold": assignment.get("dev_fold"),
        "source_identity_sha256": assignment["source_identity_sha256"],
        "model_key": expected_model_key,
        "model_sha256": result["model_sha256"],
        "model_storage_bytes_charged_per_column": model_bytes,
        "source_receipt_mode": source_mode,
        "block_count": len(blocks),
        "fresh_process_frame_decodes": fresh_decodes,
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "native_byte_context_id": NEW_BYTE_CONTEXT,
        "native": {},
        "oracle": oracle_projection(oracle),
        "comparison": {
            "exact_candidate_retrieval": None,
            "exact_candidate_retrieval_reason": "0/24 complete stored pipelines map to a frozen oracle candidate",
            "native_vs_oracle_regret": None,
            "native_vs_oracle_regret_reason": "candidate scope and byte_context_id differ",
            "reference_contract": "operator-prepared135-system-reference-contract-v1",
            "oracle_best_observed_pipeline_compressed_bytes": oracle_bytes,
        },
    }
    for method in METHODS:
        method_value = column_methods[method]
        detailed["native"][method] = {
            "plan_bytes": method_value["plan_bytes"],
            "model_storage_bytes_reported_separately": model_bytes,
            "decision": method_value["decision"],
            "four_goal_batch_e2e": method_value["four_goal_batch_e2e"],
            "objectives": {},
        }
        for target, objective in method_value["objectives"].items():
            bytes_with_container = (
                objective["frame_bytes"] + objective["column_container_bytes"]
            )
            system_bytes = (
                bytes_with_container + method_value["plan_bytes"] + model_bytes
            )
            detailed["native"][method]["objectives"][target] = {
                "selected_candidate_counts": dict(sorted(
                    objective["selected_candidate_counts"].items()
                )),
                "complete_frame_bytes": objective["frame_bytes"],
                "frame_compression_ratio": ratio(objective["frame_bytes"], source_bytes),
                "column_container_bytes": objective["column_container_bytes"],
                "bytes_with_column_container": bytes_with_container,
                "column_container_construction": objective[
                    "column_container_construction"
                ],
                "plan_bytes_reported_separately": method_value["plan_bytes"],
                "model_storage_bytes_reported_separately": model_bytes,
                "bytes_with_conservative_model_plan_charge": system_bytes,
                "complete_system_space_ratio": ratio(system_bytes, source_bytes),
                "decision_plus_encode_component_sum": objective[
                    "decision_plus_encode_component_sum"
                ],
                "decode": objective["decode"],
                "native_vs_oracle_regret": None,
                "same_source_complete_frame_reference_multiple": ratio(
                    objective["frame_bytes"], oracle_bytes
                ),
                "conservative_selector_system_to_oracle_frame_reference_multiple": ratio(
                    system_bytes, oracle_bytes
                ),
            }
    column_summary = {
        "split": assignment["split"],
        "dataset_id": item_key[0],
        "blocks": len(blocks),
        "source_bytes": source_bytes,
        "valid_values": valid_values,
        "oracle_bytes": oracle_bytes,
        "model_bytes": model_bytes,
        "fresh_decodes": fresh_decodes,
        "source_mode": source_mode,
        "methods": column_methods,
        "candidate_totals": candidate_totals,
    }
    return detailed, column_summary


def add_column_to_scope(scope: dict[str, Any], column: dict[str, Any]) -> None:
    scope["columns"] += 1
    scope["datasets"].add(column["dataset_id"])
    scope["blocks"] += column["blocks"]
    scope["original_numeric_bytes"] += column["source_bytes"]
    scope["valid_value_count"] += column["valid_values"]
    scope["oracle_best_observed_pipeline_compressed_bytes"] += column["oracle_bytes"]
    scope["model_storage_bytes_charged_per_column_sum"] += column["model_bytes"]
    scope["fresh_process_frame_decodes"] += column["fresh_decodes"]
    scope["source_receipt_modes"][column["source_mode"]] += 1
    for method in METHODS:
        source_method = column["methods"][method]
        target_method = scope["methods"][method]
        target_method["plan_bytes"] += source_method["plan_bytes"]
        sum_timing(target_method["decision"], source_method["decision"])
        sum_timing(target_method["four_goal_batch_e2e"],
                   source_method["four_goal_batch_e2e"])
        for target in TARGETS:
            source_objective = source_method["objectives"][target]
            target_objective = target_method["objectives"][target]
            target_objective["frame_bytes"] += source_objective["frame_bytes"]
            target_objective["column_container_bytes"] += source_objective[
                "column_container_bytes"
            ]
            sum_timing(
                target_objective["column_container_construction"],
                source_objective["column_container_construction"],
            )
            sum_timing(target_objective["decision_plus_encode_component_sum"],
                       source_objective["decision_plus_encode_component_sum"])
            sum_timing(target_objective["decode"], source_objective["decode"])
            target_objective["selected_candidate_counts"].update(
                source_objective["selected_candidate_counts"]
            )
    for candidate_id, source in column["candidate_totals"].items():
        target = scope["candidate_measurements"].setdefault(candidate_id, {
            "measurements": 0,
            "frame_bytes_sum": 0,
            "encode_samples": 0,
            "encode": empty_timing(),
            "fixed_encode_samples": 0,
            "fixed_encode": empty_timing(),
            "decode": empty_timing(),
            "fresh_process_frame_decodes": 0,
        })
        for field in ("measurements", "frame_bytes_sum", "encode_samples",
                      "fixed_encode_samples", "fresh_process_frame_decodes"):
            target[field] += source[field]
        for field in ("encode", "fixed_encode", "decode"):
            sum_timing(target[field], source[field])


def validate_mapping_audit(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("status") == "complete_no_exact_pipeline_mapping" and
            value.get("candidate_count") == 24 and
            value.get("exact_retrievable_candidate_count") == 0,
            "oracle mapping audit does not establish 0/24 exact mappings")
    require({row.get("candidate_id") for row in value.get("candidates", [])} ==
            set(CANDIDATES),
            "oracle mapping audit candidate set mismatch")
    return value


def validate_system_reference_contract(path: Path, columns_sha256: str,
                                       oracle_sha256: str) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") ==
            "operator-prepared135-system-reference-contract-v1" and
            value.get("status") == "frozen_before_formal_whole_column_outcomes",
            "system reference contract was not frozen before formal outcomes")
    scope = value.get("scope")
    require(isinstance(scope, dict) and
            scope.get("datasets") == EXPECTED_DATASETS and
            scope.get("columns") == EXPECTED_COLUMNS and
            scope.get("columns_sha256") == columns_sha256 and
            scope.get("oracle_jsonl_sha256") == oracle_sha256 and
            scope.get("split_sha256") == EXPECTED_SPLIT_SHA256 and
            scope.get("candidate_catalog_sha256") == EXPECTED_CATALOG_SHA256,
            "system reference contract scope binding mismatch")
    phase = value.get("phase_rule")
    require(isinstance(phase, dict) and
            phase.get("oracle_join") == "post-selection only" and
            phase.get("confirm_labels_in_fit_threshold_or_model_selection") is False and
            phase.get("contract_fields_may_change_after_formal_outcomes") is False,
            "system reference contract phase rule mismatch")
    multiples = value.get("reference_multiples")
    require(isinstance(multiples, dict) and set(multiples) == {
        "same_source_complete_frame_reference_multiple",
        "conservative_selector_system_to_oracle_frame_reference_multiple",
    }, "system reference contract multiple definitions mismatch")
    denominator = value.get("reference_denominator")
    require(isinstance(denominator, dict) and
            denominator.get("field") == "best_observed_pipeline_compressed_bytes" and
            denominator.get("selector_model_plan_bytes") == 0 and
            denominator.get("operator_ranking_timing") is None,
            "system reference denominator mismatch")
    interpretation = value.get("interpretation")
    require(isinstance(interpretation, dict) and
            interpretation.get("exact_candidate_retrieval") is False and
            interpretation.get("within_pool_oracle_regret") is False and
            interpretation.get("joint_operator_algorithm_optimum") is False,
            "system reference interpretation boundary mismatch")
    return value


def markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    all_scope = report["metrics"]["all"]
    development = report["metrics"]["development"]
    confirm = report["metrics"]["confirm"]
    lines = [
        "# Prepared selector current-135 report",
        "",
        f"Status: **{report['status']}**.",
        "",
        "## Verified scope",
        "",
        f"- {coverage['verified_columns']:,}/{coverage['expected_columns']:,} columns across "
        f"{coverage['verified_datasets']}/{coverage['expected_datasets']} datasets.",
        f"- Development: {development['columns']:,} columns; confirm: {confirm['columns']:,} columns.",
        f"- {all_scope['blocks']:,} original manifest blocks and "
        f"{all_scope['fresh_process_frame_decodes']:,} independent frame-only decodes.",
        "- Model labels come only from the frozen historical development subset; confirm labels were not fitted or used for tuning.",
        "- NNCP and excluded action IDs 106-110 remained disabled.",
        "",
        "## Candidate and oracle boundary",
        "",
        "The native selector ranks 23 compression candidates from the frozen 24-entry catalog (raw is excluded). "
        "It is not a selector over all 85 algorithms or all 97 operators.",
        "",
        "Exact complete-pipeline retrieval coverage against the frozen oracle is **0/24**. "
        "The report therefore keeps native NGF1/R1B1 measurements, frozen direct-algorithm measurements, "
        "and frozen O2R1 operator measurements in separate byte contexts. Native-versus-oracle regret is null. "
        "The oracle is read only after predictions and measurements are frozen.",
        "",
        "No 97 x 85 operator-algorithm Cartesian product was measured, and this report does not claim a joint optimum.",
        "",
        "## Native whole-column totals",
        "",
        "Column-container, plan, and model storage charges are shown separately from frame bytes. "
        "The frozen pre-outcome system-reference contract permits two explicitly named reference multiples.",
        "",
        "| policy | goal | frame ratio | container bytes | plan bytes | model bytes | conservative system ratio | frame/oracle ref | conservative/oracle ref | encode component wall ms | decode wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        method_values = all_scope["methods"][method]
        for target in TARGETS:
            objective = method_values["objectives"][target]
            lines.append(
                f"| {method} | {target} | {objective['frame_compression_ratio']:.6f} | "
                f"{objective['column_container_bytes']:,} | "
                f"{method_values['plan_bytes']:,} | "
                f"{method_values['model_storage_bytes_charged_per_column_sum']:,} | "
                f"{objective['complete_system_space_ratio']:.6f} | "
                f"{objective['same_source_complete_frame_reference_multiple']:.6f} | "
                f"{objective['conservative_selector_system_to_oracle_frame_reference_multiple']:.6f} | "
                f"{objective['decision_plus_encode_component_sum']['wall_ns'] / 1e6:.3f} | "
                f"{objective['decode']['wall_ns'] / 1e6:.3f} |"
            )
    lines.extend([
        "",
        "`four_goal_batch_e2e` is measured around one decision/guard plus four complete encode calls. "
        "Per-goal time is the recorded decision-plus-encode component sum, not a separately timed single-target call. "
        "Fresh-process decode entries are frame-only correctness receipts; the timed decode values are the native median decode measurements.",
        "",
        "The frame/oracle value is a same-source complete-frame reference multiple. The conservative/oracle value adds the "
        "goal-specific column container, plan bytes, and a full model-file charge per column. It is intentionally asymmetric "
        "because the frozen oracle is a zero-selection-cost hindsight frame-byte reference. Neither value is candidate retrieval, "
        "within-pool regret, or a joint 97 x 85 optimum.",
        "",
    ])
    return "\n".join(lines)


def locate_default(relative: str) -> Path:
    return Path(__file__).resolve().parents[1] / relative


def report(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    require(root.is_dir(), f"--root is not a directory: {root}")
    oracle_path = args.oracle.resolve(strict=True)
    split_path = args.split.resolve(strict=True)
    columns_path = args.columns.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    training_rows_path = args.training_rows.resolve(strict=True)
    mapping_path = args.mapping_audit.resolve(strict=True)
    system_reference_contract_path = args.system_reference_contract.resolve(strict=True)

    oracle_rows, oracle = validate_oracle(oracle_path)
    bindings_path = root / "BINDINGS.json"
    index_path = root / "INDEX.json"
    complete_path = root / "COMPLETE.json"
    bindings = load_json(bindings_path)
    require(isinstance(bindings, dict), "BINDINGS.json must be an object")
    require(bindings.get("catalog_sha256") == EXPECTED_CATALOG_SHA256,
            "run candidate catalog differs from frozen 24-entry catalog")
    require(bindings.get("NNCP") == "OFF" and
            bindings.get("excluded_action_ids") == [106, 107, 108, 109, 110],
            "NNCP/excluded-action binding mismatch")
    split = validate_split(split_path, bindings)
    columns = validate_columns(
        columns_path, protocol_path, bindings, sha256(oracle_path), oracle
    )
    models_path = find_models(root, args.models, bindings.get("models_sha256"))
    models = validate_models(models_path, bindings, split, training_rows_path)
    validate_mapping_audit(mapping_path)
    system_reference_contract = validate_system_reference_contract(
        system_reference_contract_path, sha256(columns_path), sha256(oracle_path)
    )

    index_value = load_json(index_path)
    require(isinstance(index_value, list), "INDEX.json must be an array")
    index = index_unique(index_value, "INDEX.json")
    complete = load_json(complete_path)
    require(complete.get("schema") == "operator-prepared135-columns-complete-v1" and
            complete.get("status") == "complete" and
            complete.get("completed") == EXPECTED_COLUMNS and
            complete.get("failed") == 0 and
            complete.get("expected") == EXPECTED_COLUMNS and
            complete.get("limited") is False,
            "COMPLETE.json is not a successful full 3,519-column campaign")
    require(complete.get("bindings") == bindings,
            "COMPLETE.json bindings mismatch")
    require(complete.get("index_sha256") == sha256(index_path),
            "COMPLETE.json INDEX SHA mismatch")
    require(set(index) == set(oracle) == set(columns),
            "INDEX/oracle/COLUMNS coverage differs")

    assignments = split["dataset_assignments"]
    scopes = {
        "all": empty_scope(),
        "development": empty_scope(),
        "confirm": empty_scope(),
    }
    details: list[dict[str, Any]] = []
    for item_key in sorted(oracle):
        dataset_id, column_id = item_key
        index_row = index[item_key]
        require(index_row.get("status") == "strict_verified",
                f"INDEX row {item_key} is not strict_verified")
        require(index_row.get("split") == assignments[dataset_id]["split"],
                f"INDEX row {item_key} split mismatch")
        relative = Path("columns") / dataset_id / column_id / "RESULT.json"
        result_path = root / relative
        require(result_path.is_file(), f"missing result receipt: {result_path}")
        indexed_path = Path(str(index_row.get("path", "")))
        require(tuple(indexed_path.parts[-4:]) == tuple(relative.parts[-4:]),
                f"INDEX row {item_key} path suffix mismatch")
        require(index_row.get("sha256") == sha256(result_path),
                f"INDEX row {item_key} result SHA mismatch")
        result = load_json(result_path)
        detail, summary = validate_result(
            result, result_path, columns[item_key], oracle[item_key],
            assignments[dataset_id], bindings, models,
        )
        require(index_row.get("independent_frame_decodes") ==
                detail["fresh_process_frame_decodes"],
                f"INDEX row {item_key} fresh decode count mismatch")
        details.append(detail)
        add_column_to_scope(scopes["all"], summary)
        add_column_to_scope(scopes[summary["split"]], summary)

    require(scopes["all"]["columns"] == EXPECTED_COLUMNS and
            len(scopes["all"]["datasets"]) == EXPECTED_DATASETS,
            "validated result coverage is incomplete")
    require(scopes["confirm"]["columns"] == 86 and
            scopes["development"]["columns"] == 3433,
            "development/confirm column coverage mismatch")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    detail_path = output_root / "COLUMNS.jsonl"
    detail_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in details
    )
    atomic_text(detail_path, detail_text)
    report_value = {
        "schema": SCHEMA,
        "status": "complete_strict_verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "post_selection_only": True,
        "confirm_labels_used_for_fit_threshold_or_model_selection": False,
        "candidate_scope": {
            "catalog_sha256": EXPECTED_CATALOG_SHA256,
            "catalog_entries": 24,
            "ranked_compression_candidates": 23,
            "candidate_ids": list(CANDIDATES),
            "raw_excluded_from_ranking": True,
            "full_85_algorithm_pool": False,
            "full_97_operator_pool": False,
            "NNCP": "OFF",
            "excluded_action_ids": [106, 107, 108, 109, 110],
        },
        "model_scope": {
            "source": "frozen historical 360 representative head/tail rows",
            "source_rows_sha256": EXPECTED_TRAINING_ROWS_SHA256,
            "fit_subset": "development rows only with held-out official dev fold and source-group guards",
            "oracle_3519_labels_used_in_fit": False,
            "confirm_labels_used_in_fit": False,
            "models_sha256": sha256(models_path),
            "model_count": 6,
            "confirm_training_row_count_after_source_group_purge": 328,
        },
        "coverage": {
            "expected_columns": EXPECTED_COLUMNS,
            "verified_columns": len(details),
            "expected_datasets": EXPECTED_DATASETS,
            "verified_datasets": len(scopes["all"]["datasets"]),
            "oracle_rows": len(oracle_rows),
            "missing_columns": 0,
            "extra_columns": 0,
            "failed_columns": 0,
            "development_columns": scopes["development"]["columns"],
            "confirm_columns": scopes["confirm"]["columns"],
        },
        "byte_contexts": {
            "new_harness": NEW_BYTE_CONTEXT,
            "frozen_algorithm": ALGORITHM_BYTE_CONTEXT,
            "frozen_operator": OPERATOR_BYTE_CONTEXT,
            "cross_context_regret": None,
        },
        "oracle_comparison_contract": {
            "phase": "read only after native plans and measurements were frozen",
            "exact_retrievable_candidates": 0,
            "exact_candidate_denominator": 24,
            "native_vs_oracle_regret": None,
            "operator_ranking_times": None,
            "joint_operator_algorithm_measured": False,
            "joint_global_optimum_claimed": False,
            "system_reference_contract_schema": system_reference_contract["schema"],
            "same_source_complete_frame_reference_multiple_enabled": True,
            "conservative_selector_system_to_oracle_frame_reference_multiple_enabled": True,
            "conservative_multiple_is_asymmetric": True,
            "oracle_selector_model_plan_bytes": 0,
            "oracle_interpretation": "zero-cost hindsight frame-byte reference",
        },
        "timing_contract": {
            "whole_column_repeats": 3,
            "warmups": 1,
            "per_goal": "decision/guard plus encode component sum, not a separately timed single-target call",
            "four_goal_batch": "outer timing around decision/guard plus four complete encode calls",
            "column_container": "goal-specific construction time is added to per-goal component and four-goal batch totals",
            "decode": "median native decode timing",
            "fresh_process_decode": "frame-only correctness receipt; no fresh-process timing is claimed",
        },
        "metrics": {name: serialise_scope(scope) for name, scope in scopes.items()},
        "artifacts": {
            "root": str(root),
            "bindings": {"path": str(bindings_path), "sha256": sha256(bindings_path)},
            "index": {"path": str(index_path), "sha256": sha256(index_path)},
            "complete": {"path": str(complete_path), "sha256": sha256(complete_path)},
            "models": {"path": str(models_path), "sha256": sha256(models_path)},
            "training_rows": {
                "path": str(training_rows_path),
                "sha256": sha256(training_rows_path),
            },
            "columns": {"path": str(columns_path), "sha256": sha256(columns_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "split": {"path": str(split_path), "sha256": sha256(split_path)},
            "oracle": {"path": str(oracle_path), "sha256": sha256(oracle_path)},
            "mapping_audit": {"path": str(mapping_path), "sha256": sha256(mapping_path)},
            "system_reference_contract": {
                "path": str(system_reference_contract_path),
                "sha256": sha256(system_reference_contract_path),
            },
            "column_report": {"path": str(detail_path), "sha256": sha256(detail_path)},
        },
    }
    report_path = output_root / "REPORT.json"
    atomic_json(report_path, report_value)
    markdown_path = output_root / "REPORT.md"
    atomic_text(markdown_path, markdown(report_value))
    completion = {
        "schema": "operator-prepared135-report-complete-v1",
        "status": "complete_strict_verified",
        "report_sha256": sha256(report_path),
        "markdown_sha256": sha256(markdown_path),
        "columns_jsonl_sha256": sha256(detail_path),
        "verified_columns": len(details),
        "verified_datasets": len(scopes["all"]["datasets"]),
    }
    atomic_json(output_root / "COMPLETE.json", completion)
    return report_value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="prepared_columns run output containing BINDINGS/INDEX/COMPLETE/columns")
    parser.add_argument("--oracle", type=Path, required=True,
                        help="frozen current135 per-column oracle JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory for REPORT.json, COLUMNS.jsonl and receipts")
    parser.add_argument("--split", type=Path, default=locate_default(
        "analysis_outputs/operator_selection_execution_20261020_v1/data_protocol/split_config.json"
    ))
    parser.add_argument("--columns", type=Path, default=locate_default(
        "analysis_outputs/operator_prepared135_evaluation_20261020_v1/COLUMNS.json"
    ))
    parser.add_argument("--protocol", type=Path, default=locate_default(
        "analysis_outputs/operator_prepared135_evaluation_20261020_v1/PROTOCOL.json"
    ))
    parser.add_argument("--mapping-audit", type=Path, default=locate_default(
        "analysis_outputs/operator_prepared135_evaluation_20261020_v1/ORACLE_MAPPING_AUDIT.json"
    ))
    parser.add_argument("--system-reference-contract", type=Path, default=locate_default(
        "analysis_outputs/operator_prepared135_evaluation_20261020_v1/SYSTEM_REFERENCE_CONTRACT.json"
    ))
    parser.add_argument("--models", type=Path,
                        help="MODELS.json; otherwise locate it beside --root")
    parser.add_argument("--training-rows", type=Path, default=locate_default(
        "analysis_outputs/operator_recommend135_evaluation_20261020_v1/remote_evidence/campaign/ROWS.json"
    ), help="frozen historical 360-row selector measurement source")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        value = report(parse_args(sys.argv[1:] if argv is None else argv))
        print(json.dumps({
            "status": value["status"],
            "verified_columns": value["coverage"]["verified_columns"],
            "verified_datasets": value["coverage"]["verified_datasets"],
            "exact_retrievable_candidates": 0,
            "native_vs_oracle_regret": None,
        }, sort_keys=True))
        return 0
    except (ReportError, OSError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
