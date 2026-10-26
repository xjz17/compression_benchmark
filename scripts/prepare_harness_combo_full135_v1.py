#!/usr/bin/env python3
"""Build the immutable full-current135 harness-combination work manifest.

Only frozen column metadata, the frozen source-group split, PLAN_COLUMNS, and
the driver-emitted candidate catalog are read.  Oracle label rows, costs,
features, and compression results are deliberately outside this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "harness-combo-full135-scope-v1"
TASK_SCHEMA = "harness-combo-full135-column-task-v1"
EXPECTED = {
    "datasets": 135,
    "columns": 3519,
    "blocks": 558035,
    "valid_values": 1733672514,
    "original_numeric_bytes": 12406364520,
    "development_columns": 3433,
    "confirm_columns": 86,
    "development_source_groups": 106,
    "confirm_source_groups": 8,
}
EXPECTED_ORACLE_BINDING = (
    "6adef4003a23ee14253d2c04b844978ec340c056863ebabf61c11ecd632b35ba"
)
EXPECTED_COLUMNS_SHA256 = (
    "6184f8bc0cba374c2e0e05725681d85ddee20e7f5335e9bc751c55009aaabdc8"
)
EXPECTED_SPLIT_SHA256 = (
    "d0545713f3d56aed5d9934814ca4144ab9cc8e968af032a3cffb4741cb4676a4"
)
EXPECTED_PLAN_SHA256 = (
    "ff1404b37e9e9bee19550ea5b7c8ce54480b3c05d6ec73c9721655fdd2c58a47"
)
EXPECTED_DATASET_SET_SHA256 = (
    "89452e195ec47ced9c189d067cc4d4db3d097b47e94d212996e08194607f1f93"
)
EXCLUDED_FAMILIES = ["NNCP", "PPMd", "DMC", "PAQ8", "CMIX", "LSTM"]


class PreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def integer(value: Any, context: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{context}: expected nonnegative integer")
    return value


def nonempty_text(value: Any, context: str) -> str:
    require(isinstance(value, str) and value and "\x00" not in value,
            f"{context}: expected nonempty text")
    return value


def lowercase_sha(value: Any, context: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"{context}: invalid lowercase SHA-256")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) +
            "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def candidate_id(row: dict[str, Any], index: int) -> str:
    value = row.get("candidate_id", row.get("id"))
    result = nonempty_text(value, f"candidate {index}.candidate_id")
    if "id" in row:
        require(row["id"] == result, f"candidate {index}: id mismatch")
    return result


def parse_catalog(catalog: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(catalog.get("schema") == "harness-combo-full135-catalog-v1",
            "catalog was not emitted by the full135 C++ driver")
    require(catalog.get("phase") == 1 and catalog.get("candidate_count") == 30,
            "catalog phase/count mismatch")
    require(catalog.get("candidate_scope") == "harness_combo_pilot_v1_fixed30",
            "catalog candidate scope mismatch")
    require(catalog.get("no_cartesian_product_claim") is True,
            "catalog must explicitly disclaim Cartesian-product coverage")
    rows = catalog.get("candidates")
    require(isinstance(rows, list) and len(rows) == 30,
            "driver catalog must contain the fixed phase-1 pool of 30 candidates")
    bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        require(isinstance(raw, dict), f"candidate {index}: expected object")
        identifier = candidate_id(raw, index)
        require(identifier not in seen, f"duplicate candidate id: {identifier}")
        seen.add(identifier)
        ordinal = raw.get("candidate_ordinal", index)
        require(integer(ordinal, f"{identifier}.candidate_ordinal") == index,
                "candidate ordinals must be dense and match catalog order")
        bindings.append({
            "candidate_id": identifier,
            "candidate_ordinal": index,
            "result_directory": f"c{index:04d}_{sha256_text(identifier)[:16]}",
            "new_combination": bool(raw.get("new_combination", False)),
            "supported_dtypes": raw.get("supported_dtypes", raw.get("dtypes")),
        })
    require("zstd3" in seen and "lz4" in seen,
            "driver catalog must contain zstd3 and lz4")
    require(sum(not binding["new_combination"] for binding in bindings) == 18,
            "driver catalog must contain exactly 18 control candidates")
    searchable = "\n".join(binding["candidate_id"].lower()
                           for binding in bindings)
    require(not any(token.lower() in searchable for token in EXCLUDED_FAMILIES),
            "driver catalog contains an explicitly excluded family")
    return rows, bindings


def verify_manifest(path_text: str, expected_sha: str,
                    verify: bool) -> tuple[str, int | None]:
    if not verify:
        return "deferred_to_execution_host", None
    path = Path(path_text)
    require(path.is_file(), f"manifest unavailable on execution host: {path}")
    require(sha256_file(path) == expected_sha, f"manifest SHA mismatch: {path}")
    manifest = read_object(path)
    blocks = manifest.get("blocks")
    require(isinstance(blocks, list) and blocks,
            f"manifest has no nonempty blocks array: {path}")
    return "verified", len(blocks)


def build_scope(columns_path: Path, split_path: Path, plan_path: Path | None,
                catalog_path: Path, *, verify_manifests: bool,
                strict_full135: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    columns = read_object(columns_path)
    split = read_object(split_path)
    plan = read_object(plan_path) if plan_path is not None else None
    catalog = read_object(catalog_path)
    require(plan is not None or verify_manifests,
            "strict block-count binding requires --plan-columns or --verify-manifests")
    require(columns.get("schema") == "operator-prepared135-source-columns-v1",
            "unexpected COLUMNS schema")
    require(split.get("schema") == "operator-selection-source-group-split-v1",
            "unexpected split schema")
    if plan is not None:
        require(plan.get("schema") == "cpp135-valid48-column-plan-v1",
                "unexpected PLAN_COLUMNS schema")
    rows = columns.get("columns")
    assignments = split.get("dataset_assignments")
    manifest_bindings = plan.get("manifest_sha256") if plan is not None else None
    require(isinstance(rows, list), "COLUMNS.columns must be an array")
    require(isinstance(assignments, dict), "split.dataset_assignments must be an object")
    if plan is not None:
        require(isinstance(manifest_bindings, dict),
                "PLAN_COLUMNS.manifest_sha256 must be an object")
    _, candidate_bindings = parse_catalog(catalog)

    tasks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    datasets: set[str] = set()
    source_groups = {"development": set(), "confirm": set()}
    totals = {"valid_values": 0, "original_numeric_bytes": 0}
    verified_block_count = 0
    for source_index, row in enumerate(rows):
        context = f"COLUMNS row {source_index}"
        require(isinstance(row, dict), f"{context}: expected object")
        dataset = nonempty_text(row.get("dataset_id"), f"{context}.dataset_id")
        column_id = lowercase_sha(row.get("column_id"), f"{context}.column_id")
        key = dataset, column_id
        require(key not in seen, f"duplicate column: {dataset}/{column_id}")
        seen.add(key)
        datasets.add(dataset)
        require(dataset in assignments, f"{context}: dataset absent from split")
        assignment = assignments[dataset]
        require(isinstance(assignment, dict), f"{context}: invalid split assignment")
        split_name = nonempty_text(assignment.get("split"), f"{context}.split")
        require(split_name in {"development", "confirm"},
                f"{context}: unexpected split {split_name}")
        group = lowercase_sha(assignment.get("source_identity_sha256"),
                              f"{context}.source_group")
        source_groups[split_name].add(group)
        fold = assignment.get("dev_fold")
        if split_name == "development":
            fold = integer(fold, f"{context}.dev_fold")
            require(fold < 5, f"{context}: dev_fold must be in [0,4]")
        else:
            require(fold is None, f"{context}: confirm row has dev_fold")
        manifest_path = nonempty_text(row.get("manifest_path"),
                                      f"{context}.manifest_path")
        manifest_sha = lowercase_sha(row.get("manifest_sha256"),
                                     f"{context}.manifest_sha256")
        if manifest_bindings is not None:
            require(manifest_bindings.get(manifest_path) == manifest_sha,
                    f"{context}: COLUMNS/PLAN manifest binding mismatch")
        valid_values = integer(row.get("valid_value_count"),
                               f"{context}.valid_value_count")
        original_bytes = integer(row.get("original_numeric_bytes"),
                                 f"{context}.original_numeric_bytes")
        totals["valid_values"] += valid_values
        totals["original_numeric_bytes"] += original_bytes
        column_name = nonempty_text(row.get("column"), f"{context}.column")
        dtype = nonempty_text(row.get("dtype"), f"{context}.dtype")
        require(dtype in {"float32", "float64", "int64"},
                f"{context}: unsupported frozen dtype {dtype}")
        manifest_verification, manifest_blocks = verify_manifest(
            manifest_path, manifest_sha, verify_manifests)
        if manifest_blocks is not None:
            verified_block_count += manifest_blocks
        tasks.append({
            "schema": TASK_SCHEMA,
            "scope": "legacy_current135",
            "phase": 1,
            "task_id": "",  # assigned after development-first ordering
            "ordinal": -1,
            "source_row_index": source_index,
            "column": {
                "dataset_id": dataset,
                "column_id": column_id,
                "column": column_name,
                "dtype": dtype,
                "valid_value_count": valid_values,
                "original_numeric_bytes": original_bytes,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_sha,
                "manifest_verification": manifest_verification,
            },
            "split": {
                "name": split_name,
                "fold": fold,
                "source_group": group,
                "source_identity_kind": assignment.get("source_identity_kind"),
            },
            "candidate_scope": "harness_combo_pilot_v1_fixed30",
            "candidate_count": len(candidate_bindings),
            # Flat aliases retain compatibility with simple column drivers.
            "dataset_id": dataset,
            "column_id": column_id,
            "column_name": column_name,
            "dtype": dtype,
            "valid_value_count": valid_values,
            "original_numeric_bytes": original_bytes,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha,
        })

    tasks.sort(key=lambda row: (
        0 if row["split"]["name"] == "development" else 1,
        row["column"]["dataset_id"], row["column"]["column_id"],
    ))
    dataset_set_sha256 = sha256_text(
        "".join(f"{dataset}\n" for dataset in sorted(datasets)))
    columns_sha = sha256_file(columns_path)
    split_sha = sha256_file(split_path)
    catalog_sha = sha256_file(catalog_path)
    for index, task in enumerate(tasks):
        task["ordinal"] = index
        task["task_id"] = f"t{index:04d}_{task['column']['column_id'][:16]}"
        task["bindings"] = {
            "columns_path": str(columns_path),
            "columns_sha256": columns_sha,
            "split_path": str(split_path),
            "split_sha256": split_sha,
            "catalog_path": str(catalog_path),
            "catalog_sha256": catalog_sha,
            "oracle_reference_sha256": columns.get(
                "oracle_reference_jsonl_sha256"),
            "dataset_set_sha256": dataset_set_sha256,
        }

    plan_counts = plan.get("counts") if plan is not None else None
    if plan is not None:
        require(isinstance(plan_counts, dict), "PLAN_COLUMNS.counts must be an object")
    if verify_manifests:
        observed_blocks = verified_block_count
        if plan_counts is not None:
            require(integer(plan_counts.get("blocks"), "PLAN_COLUMNS.counts.blocks") ==
                    observed_blocks,
                    "verified manifest block count differs from PLAN_COLUMNS")
    else:
        observed_blocks = integer(plan_counts.get("blocks"),
                                  "PLAN_COLUMNS.counts.blocks")
    observed = {
        "datasets": len(datasets),
        "columns": len(tasks),
        "blocks": observed_blocks,
        **totals,
        "development_columns": sum(
            task["split"]["name"] == "development" for task in tasks),
        "confirm_columns": sum(
            task["split"]["name"] == "confirm" for task in tasks),
        "development_source_groups": len(source_groups["development"]),
        "confirm_source_groups": len(source_groups["confirm"]),
    }
    if strict_full135:
        require(observed == EXPECTED,
                f"frozen current135 totals changed: {observed} != {EXPECTED}")
        require(columns_sha == EXPECTED_COLUMNS_SHA256,
                "COLUMNS bytes differ from the frozen current135 source")
        require(split_sha == EXPECTED_SPLIT_SHA256,
                "source-group split bytes differ from the frozen split")
        require(dataset_set_sha256 == EXPECTED_DATASET_SET_SHA256,
                "dataset membership differs from frozen current135")
        require(columns.get("scope") == "full3519", "COLUMNS scope is not full3519")
        require(columns.get("oracle_reference_jsonl_sha256") == EXPECTED_ORACLE_BINDING,
                "COLUMNS oracle hash binding changed")
        require(set(assignments) == datasets,
                "split assignments do not exactly match frozen current135 datasets")
        if plan is not None:
            require(sha256_file(plan_path) == EXPECTED_PLAN_SHA256,
                    "PLAN_COLUMNS bytes differ from the frozen block plan")
            require(len(manifest_bindings) == EXPECTED["columns"],
                    "PLAN_COLUMNS manifest binding count changed")
            require(len(plan.get("jobs", [])) == EXPECTED["columns"],
                    "PLAN_COLUMNS job count changed")

    index_rows = [
        {**task, "task_path": f"tasks/{task['task_id']}.json"}
        for task in tasks
    ]
    jsonl = b"".join(canonical_bytes(task) for task in index_rows)
    scope = {
        "schema": SCHEMA,
        "status": "ready",
        "scope": "frozen legacy_current135 only",
        "input_contract": "frozen value-column manifests; NULL removed; historical NaN words retained",
        "counts": observed,
        "ordering": "all development columns first, then all confirm columns; each partition sorted by dataset_id,column_id",
        "inputs": {
            "columns": {"path": str(columns_path), "sha256": columns_sha},
            "split": {"path": str(split_path), "sha256": split_sha},
            "plan_columns": ({"path": str(plan_path), "sha256": sha256_file(plan_path)}
                             if plan_path is not None else None),
            "blocks_binding": ("PLAN_COLUMNS.counts.blocks" if plan_path is not None
                               else "verified sum of manifest blocks on this host"),
            "driver_catalog": {"path": str(catalog_path), "sha256": catalog_sha},
            "oracle_labels_opened": False,
            "oracle_binding_only": columns.get("oracle_reference_jsonl_sha256"),
        },
        "candidate_count": len(candidate_bindings),
        "candidate_scope": "harness_combo_pilot_v1_fixed30",
        "explicitly_excluded_families": EXCLUDED_FAMILIES,
        "candidate_bindings": candidate_bindings,
        "dataset_set_sha256": dataset_set_sha256,
        "task_schema": TASK_SCHEMA,
        "column_tasks_sha256": hashlib.sha256(jsonl).hexdigest(),
        "manifest_verification": (
            "verified_all_on_this_host" if verify_manifests
            else "deferred_to_execution_host"
        ),
        "split_policy": {
            "development_may_drive_search": True,
            "confirm_labels_may_drive_search_or_tuning": False,
            "confirm_execution": "after development; final evaluation only",
        },
        "claims_full_operator_algorithm_cartesian_product": False,
    }
    return scope, tasks, jsonl


def write_bundle(output_root: Path, scope: dict[str, Any],
                 tasks: list[dict[str, Any]], tasks_jsonl: bytes,
                 catalog_path: Path) -> None:
    require(not output_root.exists(), f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    atomic_write(output_root / "SCOPE.json", pretty_bytes(scope))
    atomic_write(output_root / "COLUMN_TASKS.jsonl", tasks_jsonl)
    # Preserve the exact driver-emitted catalog bytes.
    atomic_write(output_root / "CANDIDATES.json", catalog_path.read_bytes())
    for task in tasks:
        atomic_write(output_root / "tasks" / f"{task['task_id']}.json",
                     pretty_bytes(task))
    complete = {
        "schema": "harness-combo-full135-preparation-receipt-v1",
        "status": "complete",
        "scope_sha256": sha256_file(output_root / "SCOPE.json"),
        "column_tasks_sha256": sha256_file(output_root / "COLUMN_TASKS.jsonl"),
        "candidates_sha256": sha256_file(output_root / "CANDIDATES.json"),
        "task_files": len(tasks),
        "oracle_labels_opened": False,
    }
    atomic_write(output_root / "PREPARED.json", pretty_bytes(complete))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--plan-columns", type=Path)
    parser.add_argument("--catalog", "--driver-catalog", dest="driver_catalog",
                        type=Path, default=Path("CATALOG.json"))
    parser.add_argument("--output", "--output-root", dest="output_root",
                        type=Path, required=True)
    parser.add_argument("--verify-manifests", action="store_true",
                        help="open every declared manifest and verify its SHA-256")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    resolved_columns = args.columns.resolve(strict=True)
    resolved_split = args.split.resolve(strict=True)
    resolved_plan = (args.plan_columns.resolve(strict=True)
                     if args.plan_columns is not None else None)
    resolved_catalog = args.driver_catalog.resolve(strict=True)
    scope, tasks, tasks_jsonl = build_scope(
        resolved_columns, resolved_split, resolved_plan, resolved_catalog,
        verify_manifests=args.verify_manifests,
    )
    output_root = args.output_root.resolve()
    write_bundle(output_root, scope, tasks, tasks_jsonl, resolved_catalog)
    print(json.dumps({
        "status": "complete",
        "output_root": str(output_root),
        "datasets": scope["counts"]["datasets"],
        "columns": scope["counts"]["columns"],
        "blocks": scope["counts"]["blocks"],
        "candidates": scope["candidate_count"],
        "oracle_labels_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
