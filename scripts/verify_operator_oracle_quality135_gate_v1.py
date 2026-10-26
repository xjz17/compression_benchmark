#!/usr/bin/env python3
"""Open the one-shot confirm evaluation gate without reading confirm costs.

The gate independently reconstructs development training identities and
candidate masks from the frozen split, legacy groups and development costs.
It deliberately never opens or hashes CONFIRM_COSTS.jsonl.  Its expected hash
must agree between LOCK.json and the completed preparation receipt; the final
evaluator is responsible for hashing that file after this gate is open.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


FROZEN_ORACLE_SHA256 = (
    "6adef4003a23ee14253d2c04b844978ec340c056863ebabf61c11ecd632b35ba"
)
FROZEN_SPLIT_SHA256 = (
    "d0545713f3d56aed5d9934814ca4144ab9cc8e968af032a3cffb4741cb4676a4"
)
FROZEN_GROUPS_SHA256 = (
    "4042d8efd80110b966e471befb3b672b350e8ba52335378005f45ad22732e3c2"
)
FROZEN_COLUMNS_SHA256 = (
    "6184f8bc0cba374c2e0e05725681d85ddee20e7f5335e9bc751c55009aaabdc8"
)
FROZEN_OPERATOR_ROWS_SHA256 = (
    "7dafd720614b9cc4c566edb6385c8b01f3e8be98e04f345922dcb2d9cbc52013"
)
TIME_CONTRACT = "server192-current135-valid-value-w1-r3-t1-v1"

EXPECTED_FEATURE_NAMES = [
    "log1p_row_count",
    "log1p_valid_count",
    "scanned_word_count",
    "scanned_value_bytes",
    "dtype_bit_width",
    "dtype_is_float",
    "dtype_is_signed",
    "missing_fraction",
    "byte_entropy",
    "distinct_fraction",
    "run_fraction",
    "value_zero_fraction",
    "finite_fraction",
    "nonfinite_fraction",
    "delta_zero_fraction",
    "delta_bw_1_8_fraction",
    "delta_bw_9_16_fraction",
    "delta_bw_17_32_fraction",
    "delta_bw_33_64_fraction",
    "xor_zero_fraction",
    "xor_bw_1_8_fraction",
    "xor_bw_9_16_fraction",
    "xor_bw_17_32_fraction",
    "xor_bw_33_64_fraction",
]

TASKS = {
    "algorithm_space": ("algorithm", "space"),
    "algorithm_encode": ("algorithm", "encode"),
    "algorithm_decode": ("algorithm", "decode"),
    "algorithm_balanced": ("algorithm", "balanced"),
    "operator_zstd3_space": ("operator_zstd3", "space"),
    "observed_pipeline_space": ("observed_pipeline", "space"),
}
TIME_TASKS = {"algorithm_encode", "algorithm_decode", "algorithm_balanced"}
DTYPES = {"float32", "float64", "int64"}
CORE_ARTIFACT_PATHS = {
    "development_costs": "costs/DEVELOPMENT_COSTS.jsonl",
    "confirm_costs": "costs/CONFIRM_COSTS.jsonl",
    "model_index": "models/MODEL_INDEX.json",
    "chosen_oof": "models/CHOSEN_OOF_FULL_RANKS.jsonl",
    "all_predictions": "predictions/ALL_PREDICTIONS.jsonl",
    "candidate_catalog": "costs/CANDIDATES.json",
    "features_receipt": "features/FEATURES_RECEIPT.json",
    "cv_summary": "models/CV_SUMMARY.json",
}
REQUIRED_ARTIFACTS = set(CORE_ARTIFACT_PATHS) | {
    "prediction_receipt",
    "features",
    "split",
    "groups",
    "input_metadata",
    "cost_preparation",
    "source_freeze",
    "cpp_source_freeze",
    "protocol",
    "build_receipt",
    "witness",
}
EXPECTED_PREPARATION_SOURCES = {
    "main26": "6c3d693556fb58f8277ac201dfe217ea85aec65489a707260af85981faa21e9b",
    "valid48": "6cb4139b808a895a082133a238bac7240ccf2f62e3c87d4610a9cfdf5cf0f728",
    "target5": "2b39109c3d71200f9c1f36eee802f53b685f39510a9212d463419c5511c541c8",
    "completed5": "ed3ce96c07235854ab08c393f8bea2745fec4e887e57bd34ee92cf03aeacb70b",
    "lzg135": "c974cd96a5b6f933258f630eb008a00ce08cda164eba3eb5ce37b7c8d51a81d4",
    "bos_reger_optimized": "f80a998cccb0c7fd6246b8440e977fadbdce47f1bf1eacef4a963e007608f877",
    "buff_corad_lossless": "56d7b59b3bc2305ba39f933418b35e69d6a30107c02f93e6be22bcfc83c4cb8a",
    "operator97": FROZEN_OPERATOR_ROWS_SHA256,
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as source:
        value = json.load(source)
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateError(f"{path}:{line_number}: {error}") from error
            require(isinstance(value, dict), f"{path}:{line_number} is not an object")
            yield line_number, value


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def strings(value: Any, context: str) -> list[str]:
    require(isinstance(value, list), f"{context} must be an array")
    require(
        all(isinstance(item, str) and item for item in value),
        f"{context} must contain nonempty strings",
    )
    require(len(value) == len(set(value)), f"{context} contains duplicates")
    return list(value)


def finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def sample_identity(dataset_id: str, sample_id: str) -> str:
    return f"{dataset_id}:{sample_id}"


def identity_receipt(identities: Iterable[str]) -> dict[str, Any]:
    canonical = sorted(identities)
    require(len(canonical) == len(set(canonical)), "identity receipt input has duplicates")
    return {
        "count": len(canonical),
        "canonical_json_sha256": canonical_json_sha256(canonical),
    }


def safe_relative(root: Path, raw: Any, context: str) -> Path:
    require(isinstance(raw, str) and raw, f"{context}.path must be a string")
    relative = Path(raw)
    require(not relative.is_absolute(), f"{context}.path must be root-relative")
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise GateError(f"{context}.path escapes experiment root") from error
    return result


def exact_path(left: Path, right: Path, context: str) -> None:
    require(left.resolve() == right.resolve(), f"{context} path mismatch")


def forbidden_candidate(candidate_id: str) -> bool:
    upper = candidate_id.upper()
    leaf = upper.split(":", 1)[-1]
    if leaf in {"NNCP", "PPMD8", "PPMD", "DMC", "PAQ8", "CMIX", "LSTM"}:
        return True
    return re.match(r"^A(?:106|107|108|109|110)(?:_|$)", leaf) is not None


def verify_no_forbidden(candidates: Iterable[str], context: str) -> None:
    forbidden = sorted(item for item in candidates if forbidden_candidate(item))
    require(not forbidden, f"{context} contains disabled candidates: {forbidden}")


def main(root: Path, output: Path | None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.is_dir(), "experiment root is not a directory")
    output = (output or root / "CONFIRM_GATE.json").resolve()
    exact_path(output, root / "CONFIRM_GATE.json", "gate output")
    require(not output.exists(), f"refusing to overwrite {output}")

    lock_path = root / "LOCK.json"
    lock = load_json(lock_path)
    require(lock.get("schema") == "operator-oracle-quality135-lock-v1", "unexpected LOCK schema")
    require(lock.get("status") == "locked", "LOCK is not locked")
    require(lock.get("confirm_used_for_fit_or_selection") is False, "LOCK does not exclude confirm from fitting")
    require(lock.get("confirm_rows_read_before_lock") == 0, "confirm rows were read before LOCK")
    artifacts = lock.get("artifacts")
    require(isinstance(artifacts, dict), "LOCK lacks artifacts")
    require(REQUIRED_ARTIFACTS <= set(artifacts), "LOCK lacks required artifacts")

    paths: dict[str, Path] = {}
    locked_hashes: dict[str, str] = {}
    for name, binding in artifacts.items():
        require(isinstance(binding, dict), f"LOCK artifact {name} is not an object")
        paths[name] = safe_relative(root, binding.get("path"), f"LOCK.artifacts.{name}")
        digest = binding.get("sha256")
        require(valid_sha256(digest), f"LOCK artifact {name} has invalid SHA256")
        locked_hashes[name] = digest
    for name, relative in CORE_ARTIFACT_PATHS.items():
        exact_path(paths[name], root / relative, f"LOCK core artifact {name}")
    for name, path in paths.items():
        require(path.is_file(), f"LOCK artifact is missing: {name}: {path}")
        if name != "confirm_costs":
            require(sha256(path) == locked_hashes[name], f"LOCK artifact SHA mismatch: {name}")

    require(locked_hashes["split"] == FROZEN_SPLIT_SHA256, "LOCK split is not frozen split")
    require(locked_hashes["groups"] == FROZEN_GROUPS_SHA256, "LOCK groups are not frozen groups")

    split = load_json(paths["split"])
    require(split.get("schema") == "operator-selection-source-group-split-v1", "unexpected split schema")
    require(split.get("status") == "frozen_before_selector_outcomes", "split is not frozen before outcomes")
    assignments = split.get("dataset_assignments")
    require(isinstance(assignments, dict) and len(assignments) == 135, "split must contain 135 datasets")
    require(set(split.get("confirm_dataset_ids", [])) == {k for k, v in assignments.items() if v.get("split") == "confirm"}, "confirm dataset list differs from assignments")
    require(len(split.get("confirm_dataset_ids", [])) == 8, "split must contain eight confirm datasets")

    groups = load_json(paths["groups"])
    require(groups.get("schema") == "operator-recommend135-source-groups-v1", "unexpected GROUPS schema")
    dataset_to_group = groups.get("dataset_to_group")
    require(isinstance(dataset_to_group, dict) and set(dataset_to_group) == set(assignments), "GROUPS dataset coverage mismatch")
    confirm_legacy_groups = {
        dataset_to_group[dataset_id] for dataset_id in split["confirm_dataset_ids"]
    }

    source_fold: dict[str, int] = {}
    for dataset_id, assignment in assignments.items():
        require(isinstance(assignment, dict), f"split assignment {dataset_id} is invalid")
        require(assignment.get("split") in {"development", "confirm"}, f"invalid split for {dataset_id}")
        source_group = assignment.get("source_identity_sha256")
        require(valid_sha256(source_group), f"invalid source identity for {dataset_id}")
        if assignment["split"] == "development":
            fold = assignment.get("dev_fold")
            require(isinstance(fold, int) and not isinstance(fold, bool) and 0 <= fold < 5, f"invalid dev fold for {dataset_id}")
            if source_group in source_fold:
                require(source_fold[source_group] == fold, "official source group crosses folds")
            source_fold[source_group] = fold
        else:
            require(assignment.get("dev_fold") is None, f"confirm dataset {dataset_id} has dev fold")

    protocol = load_json(paths["protocol"])
    require(protocol.get("schema") == "operator-oracle-quality135-protocol-v1", "unexpected protocol schema")
    require(protocol.get("dev_fold_count") == 5, "protocol must declare five folds")
    require(protocol.get("require_frozen_current135", True) is True, "protocol disabled frozen-current135 checks")
    require(protocol.get("algorithm_time_contract") == TIME_CONTRACT, "protocol timing contract mismatch")
    require(protocol.get("NNCP") == "OFF", "protocol does not disable NNCP")
    require(protocol.get("excluded_algorithm_methods") == ["PPMD8"], "protocol PPMD8 exclusion mismatch")
    require(protocol.get("excluded_action_ids") == [106, 107, 108, 109, 110], "protocol action exclusions mismatch")
    require(protocol.get("operator_ranking_times") is None, "protocol improperly enables operator timing")
    protocol_confirm_groups = set(strings(protocol.get("confirm_legacy_groups"), "protocol.confirm_legacy_groups"))
    require(protocol_confirm_groups == confirm_legacy_groups, "protocol confirm legacy purge is incomplete or overbroad")
    allowed_time = set(strings(protocol.get("allowed_time_candidate_ids", []), "protocol.allowed_time_candidate_ids"))
    protocol_bindings = protocol.get("source_bindings")
    require(isinstance(protocol_bindings, dict), "protocol lacks source bindings")
    for field, artifact in {
        "cost_preparation": "cost_preparation",
        "candidate_catalog": "candidate_catalog",
        "features": "features",
        "features_receipt": "features_receipt",
        "cpp_source_freeze": "cpp_source_freeze",
        "build_receipt": "build_receipt",
    }.items():
        require(protocol_bindings.get(field) == locked_hashes[artifact], f"protocol source binding mismatch: {field}")
    require(protocol_bindings.get("oracle") == FROZEN_ORACLE_SHA256, "protocol oracle binding mismatch")
    require(protocol_bindings.get("split") == FROZEN_SPLIT_SHA256, "protocol split binding mismatch")
    raw_feature_receipt = protocol.get("features_receipt_path")
    require(isinstance(raw_feature_receipt, str) and raw_feature_receipt, "protocol lacks features_receipt_path")
    protocol_feature_receipt = Path(raw_feature_receipt)
    if not protocol_feature_receipt.is_absolute():
        protocol_feature_receipt = paths["protocol"].parent / protocol_feature_receipt
    exact_path(protocol_feature_receipt, paths["features_receipt"], "protocol feature receipt")

    features_receipt = load_json(paths["features_receipt"])
    require(features_receipt.get("schema") == "operator-oracle-quality135-features-receipt-v1", "unexpected features receipt schema")
    require(features_receipt.get("status") == "complete", "features receipt is incomplete")
    require(features_receipt.get("columns_sha256") == FROZEN_COLUMNS_SHA256, "features are not bound to frozen COLUMNS")
    require(features_receipt.get("oracle_reference_jsonl_sha256") == FROZEN_ORACLE_SHA256, "features are not bound to frozen oracle identity")
    require(features_receipt.get("feature_names") == EXPECTED_FEATURE_NAMES, "features receipt names mismatch")
    expected_feature_names_sha = canonical_json_sha256(EXPECTED_FEATURE_NAMES)
    require(features_receipt.get("feature_names_sha256") == expected_feature_names_sha, "feature-name SHA mismatch")
    require(features_receipt.get("column_count") == 3519 and features_receipt.get("dataset_count") == 135, "feature scope differs from frozen current135")
    require(features_receipt.get("output_sha256") == locked_hashes["features"], "features receipt output SHA differs from LOCK")
    exact_path(Path(features_receipt.get("output_path", "")), paths["features"], "features receipt output")
    timing_contract = features_receipt.get("timing_contract", {})
    require(timing_contract.get("oracle_label_io_included") is False, "feature builder included oracle label I/O")

    catalog = load_json(paths["candidate_catalog"])
    require(catalog.get("schema") == "operator-oracle-quality135-catalog-v1", "unexpected candidate catalog schema")
    algorithms = strings(catalog.get("algorithm"), "catalog.algorithm")
    operators = strings(catalog.get("operator_zstd3"), "catalog.operator_zstd3")
    observed = strings(catalog.get("observed_pipeline"), "catalog.observed_pipeline")
    require(len(algorithms) == 84, "algorithm scope must contain 84 candidates after PPMD8 exclusion")
    require(len(operators) == 97, "operator scope must contain 97 candidates")
    require(len(observed) == 181, "observed-pipeline scope must contain 181 candidates")
    require(observed == [f"algorithm:{x}" for x in algorithms] + [f"operator_zstd3:{x}" for x in operators], "observed pipeline is not the measured namespaced union")
    require(catalog.get("excluded_algorithm_methods") == ["PPMD8"], "catalog must exclude PPMD8")
    require(catalog.get("excluded_action_ids") == [106, 107, 108, 109, 110], "catalog disabled action IDs mismatch")
    require(catalog.get("NNCP") == "OFF", "NNCP is not OFF")
    require(catalog.get("joint_cross_product_measured") is False, "catalog claims an unmeasured Cartesian product")
    require(catalog.get("operator_and_observed_objectives") == ["space"], "operator/observed objectives must be space only")
    require(catalog.get("algorithm_objectives") == ["space", "encode", "decode", "balanced"], "algorithm objectives mismatch")
    for name, values in (("algorithm", algorithms), ("operator", operators), ("observed", observed)):
        verify_no_forbidden(values, f"catalog {name}")
    require(allowed_time <= set(algorithms), "timing subset contains non-algorithm candidates")

    for artifact_name in ("source_freeze", "cpp_source_freeze"):
        freeze = load_json(paths[artifact_name])
        require(
            freeze.get("schema") == "operator-oracle-quality135-source-freeze-v1",
            f"unexpected {artifact_name} schema",
        )
        frozen_files = freeze.get("files")
        require(isinstance(frozen_files, dict) and frozen_files, f"{artifact_name} lacks files")
        for relative, expected in frozen_files.items():
            frozen_path = safe_relative(root, relative, f"{artifact_name}.files.{relative}")
            require(valid_sha256(expected), f"{artifact_name} has invalid SHA for {relative}")
            require(frozen_path.is_file() and sha256(frozen_path) == expected, f"{artifact_name} source changed: {relative}")

    build_receipt = load_json(paths["build_receipt"])
    require(build_receipt.get("schema") == "operator-oracle-quality135-build-v1", "unexpected build receipt schema")
    require(build_receipt.get("status") == "built-and-tested", "C++ build/tests are incomplete")
    require(build_receipt.get("NNCP") == "OFF", "build receipt does not disable NNCP")
    require(build_receipt.get("excluded_action_ids") == [106, 107, 108, 109, 110], "build action exclusions mismatch")
    require(build_receipt.get("test_assertions_enabled") is True, "build tests did not enable assertions")
    for test_name in ("quality_model_test", "quality_data_test"):
        require(build_receipt.get("tests", {}).get(test_name, {}).get("returncode") == 0, f"build unit failed: {test_name}")
    witness = load_json(paths["witness"])
    require(witness.get("status") == "pass", "independent README witness did not pass")

    cost_preparation = load_json(paths["cost_preparation"])
    require(cost_preparation.get("schema") == "operator-oracle-quality135-cost-preparation-v1", "unexpected preparation receipt schema")
    require(cost_preparation.get("status") == "complete", "cost preparation is incomplete")
    require(cost_preparation.get("columns") == {"confirm": 86, "development": 3433}, "prepared split row counts mismatch")
    require(cost_preparation.get("oracle_sha256") == FROZEN_ORACLE_SHA256, "preparation oracle SHA mismatch")
    require(cost_preparation.get("split_sha256") == FROZEN_SPLIT_SHA256, "preparation split SHA mismatch")
    require(cost_preparation.get("confirm_outcomes_summarized_or_used_for_tuning") is False, "preparation receipt reports confirm use")
    require(cost_preparation.get("operator_time_fields") == "null; diagnostic timing excluded", "operator time boundary mismatch")
    frozen_sources = cost_preparation.get("sources")
    require(isinstance(frozen_sources, list), "preparation receipt lacks sources")
    source_hashes = {item.get("name"): item.get("sha256") for item in frozen_sources if isinstance(item, dict)}
    require(source_hashes == EXPECTED_PREPARATION_SOURCES, "preparation source SHA set mismatch")
    prepared_outputs = cost_preparation.get("outputs")
    require(isinstance(prepared_outputs, dict), "preparation receipt lacks output hashes")
    expected_output_bindings = {
        "DEVELOPMENT_COSTS.jsonl": "development_costs",
        "CONFIRM_COSTS.jsonl": "confirm_costs",
        "CANDIDATES.json": "candidate_catalog",
        "INPUT_METADATA.json": "input_metadata",
    }
    for filename, artifact in expected_output_bindings.items():
        binding = prepared_outputs.get(filename)
        require(isinstance(binding, dict) and binding.get("sha256") == locked_hashes[artifact], f"preparation output SHA mismatch for {filename}")
    # Intentionally do not call sha256(), open(), read_text(), stat().st_size or
    # any JSON parser on paths["confirm_costs"] below this point.

    input_metadata = load_json(paths["input_metadata"])
    require(input_metadata.get("schema") == "operator-oracle-quality135-input-metadata-v1", "unexpected input metadata schema")
    require(input_metadata.get("oracle_sha256") == FROZEN_ORACLE_SHA256, "input metadata oracle SHA mismatch")
    require(input_metadata.get("split_sha256") == FROZEN_SPLIT_SHA256, "input metadata split SHA mismatch")
    require(input_metadata.get("legacy_groups_sha256") == FROZEN_GROUPS_SHA256, "input metadata groups SHA mismatch")
    metadata_rows = input_metadata.get("columns")
    require(isinstance(metadata_rows, list) and len(metadata_rows) == 3519, "input metadata must contain 3519 columns")
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    development_keys: set[tuple[str, str]] = set()
    confirm_keys: set[tuple[str, str]] = set()
    for row in metadata_rows:
        require(isinstance(row, dict), "input metadata column is not an object")
        dataset_id = row.get("dataset_id")
        column_id = row.get("column_id")
        require(isinstance(dataset_id, str) and isinstance(column_id, str), "metadata key is invalid")
        key = (dataset_id, column_id)
        require(key not in metadata, "duplicate input metadata key")
        require(dataset_id in assignments, f"metadata dataset absent from split: {dataset_id}")
        assignment = assignments[dataset_id]
        require(row.get("split") == assignment.get("split"), f"metadata split mismatch for {dataset_id}")
        require(row.get("source_group") == assignment.get("source_identity_sha256"), f"metadata source group mismatch for {dataset_id}")
        require(row.get("legacy_group") == dataset_to_group[dataset_id], f"metadata legacy group mismatch for {dataset_id}")
        require(row.get("dev_fold") == assignment.get("dev_fold"), f"metadata fold mismatch for {dataset_id}")
        require(row.get("dtype") in DTYPES, "metadata contains unsupported dtype")
        metadata[key] = row
        (development_keys if row["split"] == "development" else confirm_keys).add(key)
    require(len(development_keys) == 3433 and len(confirm_keys) == 86, "metadata split column counts mismatch")

    feature_rows: dict[tuple[str, str], str] = {}
    feature_datasets: set[str] = set()
    for line_number, row in jsonl(paths["features"]):
        context = f"features:{line_number}"
        require(row.get("schema") == "operator-oracle-quality135-feature-row-v1", f"{context} schema mismatch")
        require(row.get("oracle_labels_read") is False, f"{context} is not label-isolated")
        key = (row.get("dataset_id"), row.get("column_id"))
        require(key in metadata and key not in feature_rows, f"{context} key mismatch or duplicate")
        meta = metadata[key]
        dtype = row.get("input_dtype")
        require(dtype == row.get("dtype") == meta["dtype"], f"{context} dtype mismatch")
        require(row.get("columns_sha256") == FROZEN_COLUMNS_SHA256, f"{context} columns SHA mismatch")
        require(row.get("oracle_reference_jsonl_sha256") == FROZEN_ORACLE_SHA256, f"{context} oracle identity mismatch")
        require(row.get("feature_names_sha256") == expected_feature_names_sha, f"{context} feature names SHA mismatch")
        values = row.get("features")
        require(isinstance(values, list) and len(values) == 24 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values), f"{context} feature vector invalid")
        column_source = row.get("column_source")
        require(isinstance(column_source, dict), f"{context} lacks column source")
        for field in ("manifest_path", "manifest_sha256", "original_numeric_bytes", "valid_value_count"):
            require(column_source.get(field) == meta.get(field), f"{context} column source {field} mismatch")
        feature_rows[key] = dtype
        feature_datasets.add(key[0])
    require(set(feature_rows) == set(metadata), "feature keys differ from frozen metadata")
    require(len(feature_datasets) == 135, "features do not cover 135 datasets")

    heldout_legacy: dict[int, set[str]] = {fold: set() for fold in range(5)}
    heldout_sources: dict[int, set[str]] = {fold: set() for fold in range(5)}
    for key in development_keys:
        row = metadata[key]
        heldout_legacy[row["dev_fold"]].add(row["legacy_group"])
        heldout_sources[row["dev_fold"]].add(row["source_group"])

    def included(meta: dict[str, Any], view: str | int) -> bool:
        if meta["legacy_group"] in confirm_legacy_groups:
            return False
        if view == "final":
            return True
        return meta["dev_fold"] != view and meta["legacy_group"] not in heldout_legacy[view]

    views: list[str | int] = ["final", 0, 1, 2, 3, 4]
    expected_training: dict[str | int, set[str]] = {view: set() for view in views}
    for key in development_keys:
        meta = metadata[key]
        identity = sample_identity(*key)
        for view in views:
            if included(meta, view):
                expected_training[view].add(identity)

    declared_by_domain = {
        "algorithm": algorithms,
        "operator_zstd3": operators,
        "observed_pipeline": observed,
    }
    pool_sets: dict[str, dict[str | int, dict[str, set[str]]]] = {
        task: {view: {} for view in views} for task in TASKS
    }

    def usable(candidate: dict[str, Any], task_id: str, objective: str) -> bool:
        if candidate.get("eligible") is not True:
            return False
        if not finite_positive(candidate.get(f"{objective}_cost")):
            return False
        if task_id in TIME_TASKS:
            if candidate.get("timing_contract") != TIME_CONTRACT:
                return False
            if allowed_time and candidate.get("candidate_id") not in allowed_time:
                return False
        return True

    cost_keys: set[tuple[str, str]] = set()
    exact_time_observations = 0
    for line_number, row in jsonl(paths["development_costs"]):
        context = f"development costs:{line_number}"
        key = (row.get("dataset_id"), row.get("sample_id"))
        require(key in development_keys and key not in cost_keys, f"{context} key mismatch or duplicate")
        meta = metadata[key]
        for field in ("dtype", "source_group", "legacy_group", "dev_fold"):
            require(row.get(field) == meta[field], f"{context} {field} mismatch")
        require(row.get("split") == "development", f"{context} is not development")
        domains = row.get("domains")
        require(isinstance(domains, dict), f"{context} lacks domains")
        usable_by_task: dict[str, set[str]] = {}
        for task_id, (domain, objective) in TASKS.items():
            candidates = domains.get(domain)
            require(isinstance(candidates, list), f"{context} lacks {domain}")
            ids = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
            require(ids == declared_by_domain[domain], f"{context} {domain} candidate order/scope mismatch")
            usable_by_task[task_id] = {
                item["candidate_id"] for item in candidates if usable(item, task_id, objective)
            }
            if task_id == "algorithm_encode":
                exact_time_observations += sum(
                    1
                    for item in candidates
                    if (not allowed_time or item.get("candidate_id") in allowed_time)
                    and item.get("eligible") is True
                    and item.get("timing_contract") == TIME_CONTRACT
                    and finite_positive(item.get("encode_cost"))
                    and finite_positive(item.get("decode_cost"))
                )
        for task_id in TASKS:
            for view in views:
                if not included(meta, view):
                    continue
                dtype_pools = pool_sets[task_id][view]
                if meta["dtype"] not in dtype_pools:
                    dtype_pools[meta["dtype"]] = set(declared_by_domain[TASKS[task_id][0]])
                dtype_pools[meta["dtype"]].intersection_update(usable_by_task[task_id])
        cost_keys.add(key)
    require(cost_keys == development_keys, "development costs do not exactly cover development metadata")
    require(exact_time_observations > 0, "no exact-contract historical timing observations remain")

    pools: dict[str, dict[str | int, dict[str, list[str]]]] = {
        task: {view: {} for view in views} for task in TASKS
    }
    for task_id, (domain, _objective) in TASKS.items():
        for view in views:
            require(set(pool_sets[task_id][view]) == DTYPES, f"{task_id} {view} misses a dtype")
            for dtype, pool in pool_sets[task_id][view].items():
                ordered = [item for item in declared_by_domain[domain] if item in pool]
                require(ordered, f"empty pool for {task_id} {view} {dtype}")
                verify_no_forbidden(ordered, f"pool {task_id} {view} {dtype}")
                if task_id in TIME_TASKS and allowed_time:
                    require(set(ordered) <= allowed_time, f"time pool escapes frozen subset for {dtype}")
                pools[task_id][view][dtype] = ordered

    model_index = load_json(paths["model_index"])
    require(model_index.get("schema") == "operator-oracle-quality135-model-index-v1", "unexpected model index schema")
    require(model_index.get("status") == "locked_development_only", "model index is not development-locked")
    require(model_index.get("confirm_used_for_fit_or_selection") is False, "model index reports confirm use")
    require(model_index.get("feature_names") == EXPECTED_FEATURE_NAMES, "model index feature names mismatch")
    require(model_index.get("feature_names_sha256") == expected_feature_names_sha, "model index feature SHA mismatch")
    require(model_index.get("NNCP") == "OFF", "model index does not disable NNCP")
    require(model_index.get("excluded_algorithm_methods") == ["PPMD8"], "model index PPMD8 exclusion mismatch")
    require(model_index.get("excluded_action_ids") == [106, 107, 108, 109, 110], "model index action exclusions mismatch")
    require(model_index.get("joint_cross_product_measured") is False, "model index claims unmeasured joint search")
    require(model_index.get("algorithm_time_contract") == TIME_CONTRACT, "model index timing contract mismatch")

    index_inputs = model_index.get("inputs")
    require(isinstance(index_inputs, dict), "model index lacks input bindings")

    def verify_input_binding(binding: Any, artifact: str, context: str) -> None:
        require(isinstance(binding, dict), f"{context} is not an object")
        exact_path(Path(binding.get("path", "")), paths[artifact], context)
        require(binding.get("sha256") == locked_hashes[artifact], f"{context} SHA mismatch")

    verify_input_binding(index_inputs.get("development_costs"), "development_costs", "model development input")
    verify_input_binding(index_inputs.get("protocol"), "protocol", "model protocol input")
    feature_binding = index_inputs.get("features")
    verify_input_binding(feature_binding, "features", "model feature input")
    exact_path(Path(feature_binding.get("receipt_path", "")), paths["features_receipt"], "model feature receipt")
    require(feature_binding.get("receipt_sha256") == locked_hashes["features_receipt"], "model feature receipt SHA mismatch")
    preparation = index_inputs.get("preparation")
    require(isinstance(preparation, dict), "model index lacks preparation bindings")
    for filename, artifact in {
        "CANDIDATES.json": "candidate_catalog",
        "INPUT_METADATA.json": "input_metadata",
        "COMPLETE.json": "cost_preparation",
    }.items():
        verify_input_binding(preparation.get(filename), artifact, f"model preparation {filename}")

    training_audit_path = (paths["model_index"].parent / model_index.get("training_audit_path", "")).resolve()
    require(training_audit_path.is_relative_to(paths["model_index"].parent), "training audit escapes model directory")
    require(training_audit_path.is_file(), "training audit is missing")
    require(sha256(training_audit_path) == model_index.get("training_audit_sha256"), "training audit SHA mismatch")
    training_audit = load_json(training_audit_path)
    require(training_audit.get("schema") == "operator-oracle-quality135-training-audit-v1", "unexpected training audit schema")
    require(training_audit.get("status") == "complete", "training audit is incomplete")
    require(training_audit.get("fold_assignment") == "official_frozen_source_group_folds", "training audit fold contract mismatch")
    require(training_audit.get("legacy_alias_purge") is True, "training audit lacks alias purge")
    require(training_audit.get("confirm_labels_read") is False, "training audit reports confirm-label use")

    audit_tasks = training_audit.get("tasks")
    require(isinstance(audit_tasks, list), "training audit lacks tasks")
    audit_by_task = {item.get("task_id"): item for item in audit_tasks if isinstance(item, dict)}
    require(set(audit_by_task) == set(TASKS) and len(audit_tasks) == 6, "training audit task set mismatch")

    index_tasks = model_index.get("tasks")
    require(isinstance(index_tasks, list), "model index lacks tasks")
    index_by_task = {item.get("task_id"): item for item in index_tasks if isinstance(item, dict)}
    require(set(index_by_task) == set(TASKS) and len(index_tasks) == 6, "model index task set mismatch")
    selected_policies: dict[str, dict[str, Any]] = {}

    for task_id, (domain, objective) in TASKS.items():
        item = index_by_task[task_id]
        require((item.get("domain"), item.get("objective")) == (domain, objective), f"model index task mapping mismatch: {task_id}")
        require(item.get("declared_candidate_ids") == declared_by_domain[domain], f"declared candidates mismatch: {task_id}")
        verify_no_forbidden(item["declared_candidate_ids"], f"model declared candidates {task_id}")
        require(item.get("training_sample_ids") == sorted(expected_training["final"]), f"final training IDs mismatch: {task_id}")
        expected_train_rows = [metadata[key] for key in development_keys if included(metadata[key], "final")]
        expected_datasets = sorted({row["dataset_id"] for row in expected_train_rows})
        expected_sources = sorted({row["source_group"] for row in expected_train_rows})
        expected_legacy = sorted({row["legacy_group"] for row in expected_train_rows})
        require(item.get("training_dataset_ids") == expected_datasets, f"training datasets mismatch: {task_id}")
        require(item.get("training_source_groups") == expected_sources, f"training source groups mismatch: {task_id}")
        require(item.get("training_legacy_groups") == expected_legacy, f"training legacy groups mismatch: {task_id}")
        require(not (set(expected_legacy) & confirm_legacy_groups), f"confirm alias leaked into {task_id}")
        require(item.get("dtype_candidate_pools") == pools[task_id]["final"], f"final dtype pool mismatch: {task_id}")
        policy = item.get("selected_policy")
        require(isinstance(policy, dict) and policy.get("kind") in {"model", "constant", "fixed"}, f"invalid selected policy: {task_id}")
        selected_policies[task_id] = policy

        raw_model_path = item.get("model_path")
        require(isinstance(raw_model_path, str) and raw_model_path, f"model path missing: {task_id}")
        model_path = (paths["model_index"].parent / raw_model_path).resolve()
        require(model_path.is_relative_to(paths["model_index"].parent), f"model path escapes directory: {task_id}")
        require(model_path.is_file() and sha256(model_path) == item.get("model_sha256"), f"model SHA mismatch: {task_id}")
        envelope = load_json(model_path)
        require(envelope.get("schema") == "operator-oracle-quality135-model-envelope-v1" and envelope.get("task_id") == task_id, f"model envelope mismatch: {task_id}")
        core = envelope.get("core")
        require(isinstance(core, dict) and core.get("schema") == "operator-oracle-quality135-model-v1", f"model core mismatch: {task_id}")
        require((core.get("domain"), core.get("objective")) == (domain, objective), f"model core task mismatch: {task_id}")
        require(core.get("feature_names") == EXPECTED_FEATURE_NAMES, f"model feature names mismatch: {task_id}")
        require(core.get("candidate_catalog_sha256") == canonical_json_sha256(declared_by_domain[domain]), f"model candidate catalog SHA mismatch: {task_id}")
        core_candidates = core.get("candidates")
        require(isinstance(core_candidates, list) and [x.get("candidate_id") for x in core_candidates if isinstance(x, dict)] == declared_by_domain[domain], f"model candidates mismatch: {task_id}")
        core_training = core.get("training")
        require(isinstance(core_training, dict), f"model training metadata missing: {task_id}")
        require(core_training.get("row_count") == len(expected_training["final"]), f"model row count mismatch: {task_id}")
        require(core_training.get("dataset_ids") == expected_datasets and core_training.get("source_groups") == expected_sources, f"model training scope mismatch: {task_id}")
        points = core.get("points")
        require(isinstance(points, list) and len(points) == len(expected_training["final"]), f"model points count mismatch: {task_id}")
        point_ids = [x.get("sample_id") for x in points if isinstance(x, dict)]
        require(set(point_ids) == expected_training["final"] and len(point_ids) == len(set(point_ids)), f"model point IDs mismatch: {task_id}")
        driver = envelope.get("driver_metadata")
        require(isinstance(driver, dict), f"model driver metadata missing: {task_id}")
        require(driver.get("query_costs_used_for_selection") is False, f"model driver reports query cost use: {task_id}")
        require(driver.get("selected_policy") == policy, f"model selected policy mismatch: {task_id}")
        require(driver.get("training_sample_ids") == sorted(expected_training["final"]), f"model driver training IDs mismatch: {task_id}")
        require(driver.get("dtype_candidate_pools") == pools[task_id]["final"], f"model driver pools mismatch: {task_id}")
        require(set(driver.get("excluded_confirm_legacy_groups", [])) == confirm_legacy_groups, f"model confirm alias purge mismatch: {task_id}")
        if task_id in TIME_TASKS:
            require(driver.get("algorithm_time_contract") == TIME_CONTRACT, f"time model contract mismatch: {task_id}")
            require(set(driver.get("allowed_time_candidate_ids", [])) == allowed_time, f"time model candidate subset mismatch: {task_id}")
        else:
            require(driver.get("algorithm_time_contract") is None, f"non-time task carries a timing contract: {task_id}")

        audit = audit_by_task[task_id]
        require((audit.get("domain"), audit.get("objective")) == (domain, objective), f"training audit mapping mismatch: {task_id}")
        require(audit.get("final_training_sample_ids") == identity_receipt(expected_training["final"]), f"final training receipt mismatch: {task_id}")
        require(audit.get("final_dtype_candidate_pools") == pools[task_id]["final"], f"final audit pool mismatch: {task_id}")
        fold_items = audit.get("folds")
        require(isinstance(fold_items, list) and len(fold_items) == 5, f"training audit fold count mismatch: {task_id}")
        fold_by_id = {x.get("fold"): x for x in fold_items if isinstance(x, dict)}
        require(set(fold_by_id) == set(range(5)), f"training audit fold IDs mismatch: {task_id}")
        for fold in range(5):
            fold_item = fold_by_id[fold]
            validation = {
                sample_identity(*key)
                for key in development_keys
                if metadata[key]["dev_fold"] == fold
            }
            train_rows = [metadata[key] for key in development_keys if included(metadata[key], fold)]
            require(fold_item.get("training_sample_ids") == identity_receipt(expected_training[fold]), f"fold training receipt mismatch: {task_id}/{fold}")
            require(fold_item.get("validation_sample_ids") == identity_receipt(validation), f"fold validation receipt mismatch: {task_id}/{fold}")
            require(fold_item.get("training_source_groups") == sorted({x["source_group"] for x in train_rows}), f"fold source groups mismatch: {task_id}/{fold}")
            require(fold_item.get("training_legacy_groups") == sorted({x["legacy_group"] for x in train_rows}), f"fold legacy groups mismatch: {task_id}/{fold}")
            require(fold_item.get("excluded_heldout_source_groups") == sorted(heldout_sources[fold]), f"heldout source groups mismatch: {task_id}/{fold}")
            require(fold_item.get("excluded_heldout_legacy_groups") == sorted(heldout_legacy[fold]), f"heldout legacy groups mismatch: {task_id}/{fold}")
            require(set(fold_item.get("excluded_confirm_legacy_groups", [])) == confirm_legacy_groups, f"confirm legacy purge mismatch: {task_id}/{fold}")
            require(fold_item.get("dtype_candidate_pools") == pools[task_id][fold], f"training-derived dtype pool mismatch: {task_id}/{fold}")

    model_complete_path = paths["model_index"].parent / "COMPLETE.json"
    cv_summary_path = paths["cv_summary"]
    require(model_complete_path.is_file() and cv_summary_path.is_file(), "model completion artifacts are missing")
    model_complete = load_json(model_complete_path)
    require(model_complete.get("schema") == "operator-oracle-quality135-tune-complete-v1" and model_complete.get("status") == "complete", "model tuning is incomplete")
    require(model_complete.get("confirm_rows_read") == 0, "model tuning read confirm rows")
    require(model_complete.get("model_index_sha256") == locked_hashes["model_index"], "model COMPLETE index SHA mismatch")
    require(model_complete.get("chosen_oof_sha256") == locked_hashes["chosen_oof"], "model COMPLETE OOF SHA mismatch")
    require(model_complete.get("training_audit_sha256") == model_index.get("training_audit_sha256"), "model COMPLETE training-audit SHA mismatch")
    require(model_complete.get("cv_summary_sha256") == locked_hashes["cv_summary"], "model COMPLETE CV SHA mismatch")
    cv_summary = load_json(cv_summary_path)
    require(cv_summary.get("schema") == "operator-oracle-quality135-development-cv-v1" and cv_summary.get("status") == "complete", "development CV is incomplete")
    require(cv_summary.get("confirm_used_for_tuning") is False, "CV reports confirm use")

    oof_seen: set[tuple[str, str]] = set()
    oof_count = 0
    for line_number, row in jsonl(paths["chosen_oof"]):
        context = f"chosen OOF:{line_number}"
        require(row.get("schema") == "operator-oracle-quality135-oof-row-v1", f"{context} schema mismatch")
        task_id = row.get("task_id")
        require(task_id in TASKS, f"{context} task mismatch")
        key = (row.get("dataset_id"), row.get("sample_id"))
        require(key in development_keys, f"{context} is not a development key")
        unique = (task_id, sample_identity(*key))
        require(unique not in oof_seen, f"{context} duplicate")
        oof_seen.add(unique)
        meta = metadata[key]
        fold = meta["dev_fold"]
        for field in ("dtype", "source_group", "legacy_group"):
            require(row.get(field) == meta[field], f"{context} {field} mismatch")
        require(row.get("fold") == fold, f"{context} official fold mismatch")
        require(row.get("policy_id") == selected_policies[task_id].get("id"), f"{context} selected policy mismatch")
        require(row.get("query_costs_used_for_selection") is False, f"{context} reports query cost selection")
        candidate_pool = strings(row.get("candidate_pool"), f"{context}.candidate_pool")
        require(candidate_pool == pools[task_id][fold][meta["dtype"]], f"{context} candidate pool is not training-derived")
        ranking_ids = strings(row.get("ranking_candidate_ids"), f"{context}.ranking_candidate_ids")
        require(set(ranking_ids) == set(candidate_pool), f"{context} served ranking differs from pool")
        scores = row.get("ranking_predicted_log_costs")
        require(isinstance(scores, list) and len(scores) == len(ranking_ids), f"{context} ranking score shape mismatch")
        selected = row.get("selected_candidate_id")
        require(selected is None or selected in candidate_pool, f"{context} selected candidate leaves pool")
        if selected_policies[task_id]["kind"] == "model":
            require(selected == ranking_ids[0], f"{context} model selection is not rank one")
        verify_no_forbidden(ranking_ids, f"{context} served ranking")
        oof_count += 1
    require(oof_count == 3433 * 6, "chosen OOF row count mismatch")
    require(len(oof_seen) == 3433 * 6, "chosen OOF identity coverage mismatch")

    prediction_seen: set[tuple[str, str]] = set()
    prediction_count = 0
    confirm_prediction_count = 0
    for line_number, row in jsonl(paths["all_predictions"]):
        context = f"predictions:{line_number}"
        require(row.get("schema") == "operator-oracle-quality135-prediction-v1", f"{context} schema mismatch")
        task_id = row.get("task_id")
        require(task_id in TASKS, f"{context} task mismatch")
        domain, objective = TASKS[task_id]
        require((row.get("domain"), row.get("objective")) == (domain, objective), f"{context} task mapping mismatch")
        key = (row.get("dataset_id"), row.get("sample_id"))
        require(key in feature_rows and row.get("column_id") == key[1], f"{context} feature key mismatch")
        unique = (task_id, sample_identity(*key))
        require(unique not in prediction_seen, f"{context} duplicate")
        prediction_seen.add(unique)
        dtype = feature_rows[key]
        require(row.get("dtype") == dtype, f"{context} dtype mismatch")
        require(row.get("selected_policy") == selected_policies[task_id], f"{context} policy mismatch")
        require(row.get("query_costs_read") is False, f"{context} reports query cost input")
        require(row.get("query_features_sha256") == locked_hashes["features"], f"{context} feature SHA mismatch")
        require(row.get("model_index_sha256") == locked_hashes["model_index"], f"{context} model-index SHA mismatch")
        candidate_pool = strings(row.get("candidate_pool"), f"{context}.candidate_pool")
        require(candidate_pool == pools[task_id]["final"][dtype], f"{context} pool differs from development-only model")
        ranking_ids = strings(row.get("ranking_candidate_ids"), f"{context}.ranking_candidate_ids")
        require(set(ranking_ids) == set(candidate_pool), f"{context} served ranking differs from pool")
        scores = row.get("ranking_predicted_log_costs")
        require(isinstance(scores, list) and len(scores) == len(ranking_ids), f"{context} ranking score shape mismatch")
        selected = row.get("selected_candidate_id")
        require(selected is None or selected in candidate_pool, f"{context} selected candidate leaves pool")
        if selected_policies[task_id]["kind"] == "model":
            require(selected == ranking_ids[0], f"{context} model selection is not rank one")
        verify_no_forbidden(ranking_ids, f"{context} served ranking")
        forbidden_outcome_fields = {
            "space_cost",
            "encode_cost",
            "decode_cost",
            "balanced_cost",
            "compressed_bytes",
            "oracle_cost",
            "loss_log_regret",
            "selected_supported_on_validation_row",
        }
        require(not (forbidden_outcome_fields & set(row)), f"{context} contains outcome fields")
        prediction_count += 1
        if key in confirm_keys:
            confirm_prediction_count += 1
    require(prediction_count == 3519 * 6 and len(prediction_seen) == 3519 * 6, "prediction coverage mismatch")
    require(confirm_prediction_count == 86 * 6, "confirm feature prediction coverage mismatch")

    prediction_receipt = load_json(paths["prediction_receipt"])
    require(prediction_receipt.get("schema") == "operator-oracle-quality135-prediction-receipt-v1", "unexpected prediction receipt schema")
    require(prediction_receipt.get("status") == "complete", "prediction receipt is incomplete")
    prediction_output = prediction_receipt.get("output")
    require(isinstance(prediction_output, dict), "prediction receipt lacks output binding")
    exact_path(Path(prediction_output.get("path", "")), paths["all_predictions"], "prediction receipt output")
    require(prediction_output.get("sha256") == locked_hashes["all_predictions"], "prediction receipt output SHA mismatch")
    require(prediction_output.get("row_count") == 3519 * 6, "prediction receipt row count mismatch")
    prediction_inputs = prediction_receipt.get("inputs")
    require(isinstance(prediction_inputs, dict), "prediction receipt lacks input bindings")
    for field, artifact in (("model_index", "model_index"), ("features", "features")):
        verify_input_binding(prediction_inputs.get(field), artifact, f"prediction receipt {field}")
    require(prediction_receipt.get("task_count") == 6 and prediction_receipt.get("feature_row_count") == 3519, "prediction receipt scope mismatch")
    require(prediction_receipt.get("query_costs_read") is False, "prediction receipt reports query cost input")
    require(prediction_receipt.get("timing", {}).get("codec_runtime_included") is False, "prediction timing improperly claims codec runtime")

    served_time_candidates = sorted(
        {
            candidate
            for task_id in TIME_TASKS
            for values in pools[task_id]["final"].values()
            for candidate in values
        }
    )
    require(served_time_candidates, "time tasks have no served candidates")
    require(len(served_time_candidates) < len(algorithms), "historical time tasks unexpectedly cover all 84 algorithms")

    lock_sha = sha256(lock_path)
    gate = {
        "schema": "operator-oracle-quality135-confirm-gate-v1",
        "status": "open_for_single_final_evaluation",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lock_sha256": lock_sha,
        "confirm_costs_sha256": locked_hashes["confirm_costs"],
        "model_index_sha256": locked_hashes["model_index"],
        "all_predictions_sha256": locked_hashes["all_predictions"],
        "prediction_receipt_sha256": locked_hashes["prediction_receipt"],
        "confirm_used_for_fit_or_selection": False,
        "confirm_costs_opened_or_hashed_by_gate": False,
        "checks": {
            "frozen_scope": {"datasets": 135, "columns": 3519, "development_columns": 3433, "confirm_columns": 86},
            "tasks": sorted(TASKS),
            "development_oof_rows": oof_count,
            "all_prediction_rows": prediction_count,
            "confirm_feature_prediction_rows": confirm_prediction_count,
            "folds_recomputed": 5,
            "final_training_rows_after_confirm_alias_purge": len(expected_training["final"]),
            "confirm_legacy_groups": sorted(confirm_legacy_groups),
            "protocol_allowed_time_candidate_ids": sorted(allowed_time),
            "historical_timing_served_candidate_subset": served_time_candidates,
            "final_dtype_candidate_pool_sizes": {
                task: {dtype: len(values) for dtype, values in pools[task]["final"].items()}
                for task in TASKS
            },
            "source_sha_contract": EXPECTED_PREPARATION_SOURCES,
            "locked_artifact_sha256": {name: locked_hashes[name] for name in sorted(locked_hashes)},
        },
        "claim_boundaries": {
            "operator_and_observed_pipeline_objective": "space_only",
            "algorithm_timing_scope": "historical_exact_contract_subset_only",
            "algorithm_timing_contract": TIME_CONTRACT,
            "joint_operator_algorithm_cartesian_product_measured": False,
            "offline_recommendation_quality": True,
            "end_to_end_execution_of_all_historical_candidates_validated": False,
            "prepared_registry_serves_all_84_algorithms_and_97_operators": False,
        },
        "next_step": "the single final evaluator must first hash CONFIRM_COSTS.jsonl and require equality with confirm_costs_sha256 before parsing any confirm outcome",
    }
    output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": gate["status"],
                "gate": str(output),
                "lock_sha256": lock_sha,
                "confirm_costs_opened_or_hashed_by_gate": False,
            },
            sort_keys=True,
        )
    )
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the development-only lock before one confirm evaluation"
    )
    parser.add_argument("root", type=Path, help="locked experiment root containing LOCK.json")
    parser.add_argument("--output", type=Path, help="must resolve to ROOT/CONFIRM_GATE.json")
    arguments = parser.parse_args()
    try:
        main(arguments.root, arguments.output)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, sort_keys=True))
        raise SystemExit(1)
