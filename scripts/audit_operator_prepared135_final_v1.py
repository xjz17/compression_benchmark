#!/usr/bin/env python3
"""Independent read-only audit of the finalized prepared-selector campaign.

This script deliberately does not import the production report generator.  It
reads the frozen receipts and every formal RESULT.json directly, reconstructs
the requested aggregate fields, and compares them with the finalized report.
It performs no fitting, prediction, compression, or decompression.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
from typing import Any


EXPECTED_COLUMNS = 3519
EXPECTED_DATASETS = 135
EXPECTED_CONFIRM_COLUMNS = 86
EXPECTED_DEVELOPMENT_COLUMNS = 3433
METHODS = ("per_block", "reuse")
GOALS = ("space", "encode", "decode", "balanced")
MODEL_KEYS = {"dev0", "dev1", "dev2", "dev3", "dev4", "confirm"}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_bytes_json(path: Path) -> tuple[bytes, Any]:
    require(path.is_file(), f"missing file: {path}")
    raw = path.read_bytes()
    try:
        return raw, json.loads(raw)
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid JSON {path}: {error}") from error


def read_json(path: Path) -> Any:
    return read_bytes_json(path)[1]


def integer(value: Any, label: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{label} must be an integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def timing_cpu(value: Any, label: str) -> int:
    require(isinstance(value, dict) and set(value) == {"cpu_ns", "wall_ns"},
            f"{label} must contain exactly cpu_ns/wall_ns")
    integer(value["wall_ns"], f"{label}.wall_ns")
    return integer(value["cpu_ns"], f"{label}.cpu_ns")


def new_method() -> dict[str, Any]:
    return {
        "decision_cpu_ns": 0,
        "goals": {
            goal: {
                "complete_frame_bytes": 0,
                "encode_component_cpu_ns": 0,
                "decode_cpu_ns": 0,
            }
            for goal in GOALS
        },
    }


def new_scope() -> dict[str, Any]:
    return {
        "columns": 0,
        "datasets": set(),
        "blocks": 0,
        "fresh_frame_decodes": 0,
        "original_numeric_bytes": 0,
        "valid_value_count": 0,
        "methods": {method: new_method() for method in METHODS},
    }


def add_column(scope: dict[str, Any], column: dict[str, Any]) -> None:
    scope["columns"] += 1
    scope["datasets"].add(column["dataset_id"])
    for field in ("blocks", "fresh_frame_decodes", "original_numeric_bytes",
                  "valid_value_count"):
        scope[field] += column[field]
    for method in METHODS:
        target = scope["methods"][method]
        source = column["methods"][method]
        target["decision_cpu_ns"] += source["decision_cpu_ns"]
        for goal in GOALS:
            for field in ("complete_frame_bytes", "encode_component_cpu_ns",
                          "decode_cpu_ns"):
                target["goals"][goal][field] += source["goals"][goal][field]


def serialise_scope(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (len(value) if key == "datasets" else value)
        for key, value in scope.items()
    }


def expected_result_path(root: Path, dataset_id: str, column_id: str) -> Path:
    return root / "columns3519" / "columns" / dataset_id / column_id / "RESULT.json"


def validate_models(root: Path, bindings: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    models_path = root / "models" / "MODELS.json"
    models = read_json(models_path)
    require(sha256(models_path) == bindings["models_sha256"],
            "MODELS.json SHA differs from BINDINGS")
    require(models.get("schema") == "operator-prepared135-models-v1" and
            models.get("confirm_labels_in_fit") is False and
            models.get("hyperparameters_changed") is False,
            "model index protocol mismatch")
    require(models.get("split_sha256") == bindings["split_sha256"],
            "model split binding mismatch")
    entries = models.get("models")
    require(isinstance(entries, dict) and set(entries) == MODEL_KEYS,
            "model index does not contain the six formal models")

    old_root = Path(read_json(root / "FINALIZED.json")["old_root"])
    rows_path = old_root / "campaign" / "ROWS.json"
    rows = read_json(rows_path)
    require(isinstance(rows, list) and len(rows) == 360,
            "historical training source is not the frozen 360 rows")
    require(sha256(rows_path) == models["source_rows_sha256"],
            "historical training rows SHA mismatch")
    assignments = split["dataset_assignments"]
    confirm_legacy = {
        row["group_id"] for row in rows
        if assignments[row["dataset_id"]]["split"] == "confirm"
    }
    checked: dict[str, Any] = {}
    for model_key in sorted(entries):
        binding = entries[model_key]
        path = Path(binding["path"])
        require(path.is_file(), f"missing model file {model_key}")
        require(sha256(path) == binding["sha256"], f"model SHA mismatch {model_key}")
        require(path.stat().st_size == binding["model_json_bytes"],
                f"model byte count mismatch {model_key}")
        model = read_json(path)
        formal = model.get("formal_protocol")
        require(isinstance(formal, dict) and formal.get("model_key") == model_key and
                formal.get("split_sha256") == bindings["split_sha256"] and
                formal.get("confirm_labels_in_fit") is False,
                f"model formal protocol mismatch {model_key}")
        held_out = None if model_key == "confirm" else int(model_key[3:])
        excluded = set(confirm_legacy)
        if held_out is not None:
            excluded.update(
                row["group_id"] for row in rows
                if assignments[row["dataset_id"]]["split"] == "development"
                and assignments[row["dataset_id"]]["dev_fold"] == held_out
            )
        expected = [
            row for row in rows
            if assignments[row["dataset_id"]]["split"] == "development"
            and (held_out is None or assignments[row["dataset_id"]]["dev_fold"] != held_out)
            and row["group_id"] not in excluded
        ]
        sample_ids = [row["sample_id"] for row in expected]
        datasets = sorted({row["dataset_id"] for row in expected})
        source_groups = sorted({assignments[row["dataset_id"]]["source_identity_sha256"]
                                for row in expected})
        require(binding["training_row_count"] == len(expected) and
                binding["training_sample_ids"] == sample_ids and
                sorted(binding["training_dataset_ids"]) == datasets and
                sorted(binding["training_source_groups"]) == source_groups and
                sorted(binding["excluded_legacy_groups"]) == sorted(excluded),
                f"model training receipt mismatch {model_key}")
        require(formal["training_sample_ids"] == binding["training_sample_ids"] and
                sorted(formal["training_dataset_ids"]) == datasets and
                sorted(formal["training_source_groups"]) == source_groups and
                sorted(formal["excluded_legacy_groups"]) == sorted(excluded),
                f"model payload/index training mismatch {model_key}")
        checked[model_key] = {
            "sha256": binding["sha256"],
            "bytes": binding["model_json_bytes"],
            "training_rows": len(expected),
            "training_datasets": len(datasets),
            "training_source_groups": len(source_groups),
        }
    require(checked["confirm"]["training_rows"] == 328,
            "confirm model is not the exact 328-row purged fit")
    return {
        "status": "pass",
        "models_sha256": sha256(models_path),
        "historical_rows_sha256": sha256(rows_path),
        "confirm_outcomes_read": False,
        "models": checked,
    }


def validate_column(result: dict[str, Any], expected_column: dict[str, Any],
                    assignment: dict[str, Any], bindings: dict[str, Any],
                    models: dict[str, Any], label: str) -> dict[str, Any]:
    require(result.get("schema") == "operator-prepared135-column-v1" and
            result.get("status") == "strict_verified", f"{label} status/schema mismatch")
    require(result.get("bindings") == bindings, f"{label} BINDINGS mismatch")
    require(result.get("column") == expected_column, f"{label} COLUMNS mismatch")
    require(result.get("split") == assignment["split"] and
            result.get("source_identity_sha256") == assignment["source_identity_sha256"] and
            result.get("dev_fold") == assignment["dev_fold"],
            f"{label} split/source identity mismatch")
    model_key = "confirm" if assignment["split"] == "confirm" else f"dev{assignment['dev_fold']}"
    model_binding = models["models"][model_key]
    require(result.get("model_key") == model_key and
            result.get("model_sha256") == model_binding["sha256"] and
            result.get("model_storage_bytes_charged_per_column") == model_binding["model_json_bytes"],
            f"{label} selected model binding mismatch")
    initial = result.get("initial_selection")
    require(isinstance(initial, dict) and initial.get("validation_passes") == 1,
            f"{label} initial selection validation mismatch")
    initial_cpu = timing_cpu(initial.get("total"), f"{label}.initial_selection.total")
    plan_cpu = timing_cpu(result.get("initial_plan_construction"),
                          f"{label}.initial_plan_construction")
    blocks = result.get("blocks")
    require(isinstance(blocks, list) and blocks, f"{label} has no blocks")
    methods = {method: new_method() for method in METHODS}
    methods["reuse"]["decision_cpu_ns"] = initial_cpu + plan_cpu
    for goal in GOALS:
        methods["reuse"]["goals"][goal]["encode_component_cpu_ns"] = initial_cpu + plan_cpu
    original_bytes = 0
    valid_values = 0
    fresh_decodes = 0
    encoder_pid = integer(result.get("encoder_pid"), f"{label}.encoder_pid", 1)
    for position, block in enumerate(blocks):
        require(block.get("block_index") == position, f"{label} block order mismatch")
        original_bytes += integer(block.get("original_numeric_bytes"),
                                  f"{label}.block[{position}].original_numeric_bytes")
        valid_values += integer(block.get("valid_count"),
                                f"{label}.block[{position}].valid_count")
        methods["per_block"]["decision_cpu_ns"] += timing_cpu(
            block.get("selection"), f"{label}.block[{position}].selection")
        methods["reuse"]["decision_cpu_ns"] += timing_cpu(
            block.get("reuse_guard"), f"{label}.block[{position}].reuse_guard")
        candidates = block.get("candidates")
        require(isinstance(candidates, dict) and candidates,
                f"{label}.block[{position}] candidate map missing")
        for candidate_id, candidate in candidates.items():
            require(integer(candidate.get("frame_bytes"),
                            f"{label}.block[{position}].{candidate_id}.frame_bytes", 1) > 0,
                    "unreachable")
            frame_sha = candidate.get("frame_sha256")
            witness = candidate.get("fresh_decode")
            require(isinstance(frame_sha, str) and len(frame_sha) == 64 and
                    isinstance(witness, dict) and witness.get("verified") is True and
                    witness.get("frame_only_decode") is True and
                    witness.get("frame_sha256") == frame_sha and
                    integer(witness.get("decode_pid"), "fresh decode pid", 1) != encoder_pid,
                    f"{label}.block[{position}].{candidate_id} fresh decode mismatch")
            timing_cpu(candidate.get("decode"),
                       f"{label}.block[{position}].{candidate_id}.decode")
            fresh_decodes += 1
        for method in METHODS:
            selected_key = "per_block_selected_ids" if method == "per_block" else "reuse_selected_ids"
            component_key = ("per_goal_per_block_component_sum" if method == "per_block"
                             else "per_goal_reuse_component_sum")
            selected = block.get(selected_key)
            components = block.get(component_key)
            require(isinstance(selected, dict) and set(selected) == set(GOALS) and
                    isinstance(components, dict) and set(components) == set(GOALS),
                    f"{label}.block[{position}] {method} selected/component scope mismatch")
            for goal in GOALS:
                candidate_id = selected[goal]
                require(candidate_id in candidates,
                        f"{label}.block[{position}] missing selected candidate")
                candidate = candidates[candidate_id]
                aggregate = methods[method]["goals"][goal]
                aggregate["complete_frame_bytes"] += candidate["frame_bytes"]
                aggregate["encode_component_cpu_ns"] += timing_cpu(
                    components[goal], f"{label}.block[{position}].{method}.{goal}.component")
                aggregate["decode_cpu_ns"] += candidate["decode"]["cpu_ns"]
    require(original_bytes == expected_column["original_numeric_bytes"] ==
            result.get("source_bytes_verified"), f"{label} source byte denominator mismatch")
    require(valid_values == expected_column["valid_value_count"] ==
            result.get("source_valid_values_verified"), f"{label} valid count mismatch")
    require(fresh_decodes == result.get("independent_frame_decodes"),
            f"{label} fresh frame count mismatch")
    whole = result.get("whole_column")
    require(isinstance(whole, dict) and set(whole) == set(METHODS),
            f"{label} whole_column method mismatch")
    for method in METHODS:
        measured = whole[method]
        computed = methods[method]
        require(measured.get("selection_cpu_ns") == computed["decision_cpu_ns"],
                f"{label}.{method} decision CPU mismatch")
        goals = measured.get("goals")
        require(isinstance(goals, dict) and set(goals) == set(GOALS),
                f"{label}.{method} goal scope mismatch")
        for goal in GOALS:
            packaging_cpu = timing_cpu(
                goals[goal].get("column_container_construction"),
                f"{label}.{method}.{goal}.column_container_construction")
            computed_goal = computed["goals"][goal]
            computed_goal["encode_component_cpu_ns"] += packaging_cpu
            require(goals[goal].get("complete_frame_bytes") ==
                    computed_goal["complete_frame_bytes"] and
                    goals[goal].get("encode_component_sum_cpu_ns") ==
                    computed_goal["encode_component_cpu_ns"] and
                    goals[goal].get("decode_cpu_ns") == computed_goal["decode_cpu_ns"],
                    f"{label}.{method}.{goal} recomputed aggregate mismatch")
    return {
        "dataset_id": expected_column["dataset_id"],
        "split": assignment["split"],
        "blocks": len(blocks),
        "fresh_frame_decodes": fresh_decodes,
        "original_numeric_bytes": original_bytes,
        "valid_value_count": valid_values,
        "methods": methods,
    }


def compare_report(scopes: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for scope_name in ("all", "development", "confirm"):
        actual = serialise_scope(scopes[scope_name])
        expected = report["metrics"][scope_name]
        require(actual["columns"] == expected["columns"] and
                actual["datasets"] == expected["dataset_count"] and
                actual["blocks"] == expected["blocks"] and
                actual["fresh_frame_decodes"] == expected["fresh_process_frame_decodes"] and
                actual["original_numeric_bytes"] == expected["original_numeric_bytes"] and
                actual["valid_value_count"] == expected["valid_value_count"],
                f"report scope totals differ for {scope_name}")
        for method in METHODS:
            report_method = expected["methods"][method]
            computed_method = actual["methods"][method]
            require(computed_method["decision_cpu_ns"] == report_method["decision"]["cpu_ns"],
                    f"report decision CPU differs for {scope_name}/{method}")
            for goal in GOALS:
                report_goal = report_method["objectives"][goal]
                computed_goal = computed_method["goals"][goal]
                require(computed_goal["complete_frame_bytes"] ==
                        report_goal["complete_frame_bytes"] and
                        computed_goal["encode_component_cpu_ns"] ==
                        report_goal["decision_plus_encode_component_sum"]["cpu_ns"] and
                        computed_goal["decode_cpu_ns"] == report_goal["decode"]["cpu_ns"],
                        f"report goal CPU/bytes differ for {scope_name}/{method}/{goal}")
        checked[scope_name] = actual
    return checked


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    finalized_path = root / "FINALIZED.json"
    finalized = read_json(finalized_path)
    require(finalized.get("schema") == "operator-prepared135-finalized-v1" and
            finalized.get("status") == "FINALIZED" and finalized.get("stage_count") == 8,
            "FINALIZED status/stage mismatch")
    bindings_path = root / "columns3519" / "BINDINGS.json"
    bindings = read_json(bindings_path)
    complete_path = root / "columns3519" / "COMPLETE.json"
    complete = read_json(complete_path)
    index_path = root / "columns3519" / "INDEX.json"
    index = read_json(index_path)
    require(complete.get("schema") == "operator-prepared135-columns-complete-v1" and
            complete.get("status") == "complete" and
            complete.get("completed") == EXPECTED_COLUMNS and
            complete.get("expected") == EXPECTED_COLUMNS and
            complete.get("failed") == 0 and complete.get("limited") is False and
            complete.get("bindings") == bindings and
            complete.get("index_sha256") == sha256(index_path),
            "formal C++ COMPLETE contract mismatch")
    require(isinstance(index, list) and len(index) == EXPECTED_COLUMNS,
            "formal INDEX does not contain 3519 rows")
    index_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in index:
        key = (row.get("dataset_id"), row.get("column_id"))
        require(all(isinstance(item, str) and item for item in key) and key not in index_map and
                row.get("status") == "strict_verified",
                f"invalid/duplicate INDEX row {key}")
        index_map[key] = row

    columns_path = root / "COLUMNS.json"
    columns_value = read_json(columns_path)
    columns = columns_value.get("columns")
    require(isinstance(columns, list) and len(columns) == EXPECTED_COLUMNS,
            "COLUMNS.json does not contain 3519 rows")
    column_map = {(row["dataset_id"], row["column_id"]): row for row in columns}
    require(len(column_map) == EXPECTED_COLUMNS and set(column_map) == set(index_map),
            "COLUMNS/INDEX keys differ")
    split_path = root / "split_config.json"
    split = read_json(split_path)
    require(sha256(split_path) == bindings["split_sha256"], "split SHA differs from BINDINGS")
    assignments = split.get("dataset_assignments")
    require(isinstance(assignments, dict) and len(assignments) == EXPECTED_DATASETS,
            "split does not assign 135 datasets")
    source_groups: dict[str, set[tuple[str, int | None]]] = {}
    for assignment in assignments.values():
        group = assignment["source_identity_sha256"]
        identity = (assignment["split"], assignment["dev_fold"]
                    if assignment["split"] == "development" else None)
        source_groups.setdefault(group, set()).add(identity)
    require(all(len(values) == 1 for values in source_groups.values()),
            "source identity crosses split or development fold")

    build_path = root / "build" / "BUILD_RECEIPT.json"
    build = read_json(build_path)
    binary_path = root / "build" / "prepared_columns"
    binary_sha = sha256(binary_path)
    require(bindings.get("binary_sha256") == binary_sha ==
            build.get("binaries", {}).get("prepared_columns"),
            "BINDINGS/build/actual prepared_columns binary SHA mismatch")
    source_path = root / "source" / "cpp" / "operator_recommend135_v1" / "prepared_columns.cpp"
    source_sha = sha256(source_path)
    build_source = build.get("source_sha256", {})
    matching_source_hashes = {
        key: value for key, value in build_source.items()
        if key.endswith("/cpp/operator_recommend135_v1/prepared_columns.cpp")
    }
    require(len(matching_source_hashes) == 1 and
            next(iter(matching_source_hashes.values())) == source_sha,
            "build/source prepared_columns.cpp SHA mismatch")
    require(bindings.get("NNCP") == "OFF" and
            bindings.get("excluded_action_ids") == [106, 107, 108, 109, 110],
            "NNCP/excluded-action binding mismatch")

    model_audit = validate_models(root, bindings, split)
    models = read_json(root / "models" / "MODELS.json")
    report_path = root / "report" / "REPORT.json"
    report = read_json(report_path)
    require(sha256(report_path) == finalized["report"]["sha256"] and
            report.get("status") == "complete_strict_verified",
            "final report binding/status mismatch")

    scopes = {name: new_scope() for name in ("all", "development", "confirm")}
    result_hashes_checked = 0
    for key in sorted(index_map):
        dataset_id, column_id = key
        index_row = index_map[key]
        result_path = expected_result_path(root, dataset_id, column_id)
        raw, result = read_bytes_json(result_path)
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == index_row.get("sha256"), f"RESULT SHA mismatch {key}")
        indexed_path = Path(index_row.get("path", ""))
        require(tuple(indexed_path.parts[-4:]) ==
                ("columns", dataset_id, column_id, "RESULT.json"),
                f"INDEX path suffix mismatch {key}")
        assignment = assignments[dataset_id]
        column = validate_column(result, column_map[key], assignment, bindings,
                                 models, f"RESULT {dataset_id}/{column_id}")
        require(index_row.get("split") == assignment["split"] and
                index_row.get("independent_frame_decodes") == column["fresh_frame_decodes"],
                f"INDEX/result receipt mismatch {key}")
        add_column(scopes["all"], column)
        add_column(scopes[column["split"]], column)
        result_hashes_checked += 1
    require(scopes["all"]["columns"] == EXPECTED_COLUMNS and
            len(scopes["all"]["datasets"]) == EXPECTED_DATASETS and
            scopes["development"]["columns"] == EXPECTED_DEVELOPMENT_COLUMNS and
            scopes["confirm"]["columns"] == EXPECTED_CONFIRM_COLUMNS,
            "independently reconstructed coverage mismatch")
    report_comparison = compare_report(scopes, report)

    profile = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "scheduler": os.sched_getscheduler(0),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "ionice": subprocess.check_output(["ionice", "-p", str(os.getpid())],
                                           text=True).strip(),
        "address_space_limit_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
    }
    require(profile["affinity"] == [38] and profile["scheduler"] == os.SCHED_IDLE and
            profile["nice"] == 19 and profile["ionice"] == "idle" and
            profile["address_space_limit_bytes"] == [2147483648, 2147483648],
            "independent audit resource profile mismatch")
    return {
        "schema": "operator-prepared135-independent-final-review-v1",
        "status": "pass",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "profile": profile,
        "finalized": {
            "sha256": sha256(finalized_path),
            "status": finalized["status"],
            "stage_count": finalized["stage_count"],
            "archive_sha256": finalized["delivery"]["archive_sha256"],
            "archive_bytes": finalized["delivery"]["archive_bytes"],
            "manifest_sha256": finalized["delivery"]["manifest_sha256"],
            "payload_file_count": finalized["delivery"]["payload_file_count"],
            "protection_final_status": finalized["protection_post"]["status"],
            "measurement_protection_recovery_classification":
                finalized["measurement_protection_post"]["classification"],
        },
        "cross_bindings": {
            "bindings_sha256": sha256(bindings_path),
            "complete_sha256": sha256(complete_path),
            "index_sha256": sha256(index_path),
            "report_sha256": sha256(report_path),
            "prepared_columns_binary_sha256": binary_sha,
            "prepared_columns_source_sha256": source_sha,
            "binary_matches_bindings_build_and_actual": True,
            "source_matches_build_receipt": True,
            "NNCP": bindings["NNCP"],
            "excluded_action_ids": bindings["excluded_action_ids"],
        },
        "models": model_audit,
        "result_receipts": {
            "sha256_checked": result_hashes_checked,
            "fresh_decode_receipts_checked": report_comparison["all"]["fresh_frame_decodes"],
            "fresh_decode_scope": "receipt validation only; no decode rerun",
        },
        "recomputed": report_comparison,
        "report_exact_match": {
            "scopes": ["all", "development", "confirm"],
            "scope_totals": ["columns", "datasets", "blocks", "fresh_frame_decodes",
                             "original_numeric_bytes", "valid_value_count"],
            "per_method": ["decision_cpu_ns"],
            "per_goal": ["complete_frame_bytes", "encode_component_cpu_ns",
                         "decode_cpu_ns"],
            "all_fields_equal": True,
        },
        "limits": [
            "No compression, decompression, selection, fitting, or prediction was rerun.",
            "Fresh frame-only recovery was checked from the stored receipts, not re-executed.",
            "Confirm outcomes were aggregated only after fitting receipts were validated; they were not used for tuning.",
            "Model cold-start/load time and per-block plan construction time remain outside the frozen timing boundary.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.root), sort_keys=True))
        return 0
    except (AuditError, OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
