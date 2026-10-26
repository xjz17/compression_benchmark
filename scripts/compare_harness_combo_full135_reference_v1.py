#!/usr/bin/env python3
"""Add post-run byte context from the frozen 85/97/182 per-column oracle.

This program first proves that all frozen current135 column executions are
complete. Only after that gate passes does it open the oracle. The oracle is
used for final reporting, never fitting, tuning, candidate generation, or
selection during execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


ORACLE_SHA256 = "6adef4003a23ee14253d2c04b844978ec340c056863ebabf61c11ecd632b35ba"
EXCLUDED_FAMILIES = ["NNCP", "PPMd", "DMC", "PAQ8", "CMIX", "LSTM"]
RESULT_SCHEMA = "harness-combo-full135-result-v1"
TASK_SCHEMA = "harness-combo-full135-column-task-v1"
REPORT_SCHEMA = "harness-combo-full135-report-v1"
RESOLVED_CANDIDATE_STATUSES = {"success", "ineligible", "domain_rejected"}
REFERENCE_TARGETS = {
    "best_algorithm": {
        "id_field": "best_algorithm",
        "kind_field": None,
        "bytes_field": "best_algorithm_compressed_bytes",
        "candidate_scope": "85 frozen algorithms",
        "format": "complete NGF1 algorithm frames; row byte boundary retained",
    },
    "best_operator": {
        "id_field": "best_operator_id",
        "kind_field": None,
        "bytes_field": "best_operator_compressed_bytes",
        "candidate_scope": "97 frozen operators with common Zstd-3 backend",
        "format": "O2R1 per-block bundles with common Zstd-3 backend",
    },
    "best_observed_pipeline": {
        "id_field": "best_observed_pipeline_id",
        "kind_field": "best_observed_pipeline_kind",
        "bytes_field": "best_observed_pipeline_compressed_bytes",
        "candidate_scope": "182 observed algorithm or fixed-backend operator entries",
        "format": "winning observed entry native algorithm or operator format",
    },
}


class ComparisonError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ComparisonError(f"{path}:{line_number}: {error}") from error
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) +
            "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def candidate_id(row: dict[str, Any], context: str) -> str:
    identifier = row.get("candidate_id", row.get("id"))
    require(isinstance(identifier, str) and identifier,
            f"{context}: candidate id missing")
    if "candidate_id" in row and "id" in row:
        require(row["candidate_id"] == row["id"],
                f"{context}: candidate id aliases differ")
    return identifier


def success_is_strict(row: dict[str, Any]) -> bool:
    independent = row.get("independent_decode")
    independent_verified = (
        isinstance(independent, dict) and independent.get("verified") is True
    ) or row.get("bit_exact_fresh_exec") is True
    same_process_verified = (
        row.get("bit_exact") is True or
        row.get("bit_exact_same_process") is True
    )
    return (
        row.get("status") == "success" and
        row.get("strict") is True and
        row.get("rank_eligible") is True and
        row.get("complete_column", True) is True and
        same_process_verified and independent_verified
    )


def validate_completed_run(
    root: Path,
    *,
    expected_columns: int = 3519,
    expected_datasets: int = 135,
    expected_blocks: int = 558035,
    expected_candidate_attempts: int = 105570,
    expected_development_columns: int = 3433,
    expected_confirm_columns: int = 86,
) -> tuple[dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]],
           dict[str, Any]]:
    """Validate the complete run without opening or receiving oracle content."""
    require(root.is_dir(), f"experiment root missing: {root}")
    freeze_path = root / "FREEZE.json"
    finished = read_object(root / "RUN_FINISHED.json")
    require(finished.get("status") == "complete", "RUN_FINISHED is not complete")
    require(finished.get("completed_columns") == expected_columns and
            finished.get("total_columns") == expected_columns and
            finished.get("failed_columns") == 0,
            "RUN_FINISHED does not prove complete zero-failure column coverage")
    require(finished.get("freeze_sha256") == sha256_file(freeze_path),
            "RUN_FINISHED freeze binding mismatch")

    report_path = root / "report" / "REPORT.json"
    require(finished.get("report_sha256") == sha256_file(report_path),
            "RUN_FINISHED report binding mismatch")
    report = read_object(report_path)
    coverage = report.get("coverage")
    require(report.get("schema") == REPORT_SCHEMA and report.get("status") == "complete",
            "final harness report is not complete")
    require(isinstance(coverage, dict), "final harness report coverage missing")
    require(coverage.get("expected_columns") == expected_columns and
            coverage.get("bound_column_results") == expected_columns and
            coverage.get("source_blocks_in_bound_results") == expected_blocks and
            coverage.get("expected_candidate_attempts") == expected_candidate_attempts and
            coverage.get("observed_terminal_attempts") == expected_candidate_attempts and
            coverage.get("missing") == 0 and coverage.get("failures") == 0,
            "final harness report coverage differs from the frozen contract")
    require(report.get("all_expected_results_terminal") is True and
            report.get("all_retained_strict_frames_verified") is True,
            "final harness report lacks terminal/frame verification")

    catalog_path = root / "CANDIDATES.json"
    catalog = read_object(catalog_path)
    require(catalog.get("schema") == "harness-combo-full135-catalog-v1" and
            catalog.get("phase") == 1 and
            catalog.get("candidate_scope") == "harness_combo_pilot_v1_fixed30" and
            catalog.get("candidate_count") == 30 and
            catalog.get("no_cartesian_product_claim") is True,
            "fixed full135 C++ catalog contract mismatch")
    catalog_rows = catalog.get("candidates")
    require(isinstance(catalog_rows, list) and len(catalog_rows) == 30,
            "fixed HCB30 catalog missing")
    catalog_by_id: dict[str, dict[str, Any]] = {}
    ordinal_by_id: dict[str, int] = {}
    for ordinal, row in enumerate(catalog_rows):
        require(isinstance(row, dict), f"catalog candidate {ordinal}: expected object")
        identifier = candidate_id(row, f"catalog candidate {ordinal}")
        require(identifier not in catalog_by_id, f"duplicate catalog candidate: {identifier}")
        require(row.get("candidate_ordinal", ordinal) == ordinal,
                f"catalog candidate {identifier}: ordinal mismatch")
        catalog_by_id[identifier] = row
        ordinal_by_id[identifier] = ordinal
    catalog_sha = sha256_file(catalog_path)
    require(sum(not bool(row.get("new_combination", False))
                for row in catalog_rows) == 18,
            "HCB30 control18 membership changed")
    searchable = "\n".join(catalog_by_id).lower()
    require(not any(token.lower() in searchable for token in EXCLUDED_FAMILIES),
            "HCB30 contains an explicitly excluded family")

    tasks = read_jsonl(root / "plan" / "COLUMN_TASKS.jsonl")
    require(len(tasks) == expected_columns, "task index does not cover every column")
    split_sequence = [task.get("split", {}).get("name") for task in tasks]
    require(split_sequence == (["development"] * expected_development_columns +
                               ["confirm"] * expected_confirm_columns),
            "task index does not preserve development-then-confirm order")
    require(expected_development_columns + expected_confirm_columns == expected_columns,
            "invalid expected split counts")

    bound: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    datasets: set[str] = set()
    source_blocks = 0
    source_values = 0
    source_bytes = 0
    for ordinal, task in enumerate(tasks):
        require(task.get("schema") == TASK_SCHEMA and task.get("ordinal") == ordinal,
                f"task {ordinal}: schema/ordinal mismatch")
        task_id = task.get("task_id")
        require(isinstance(task_id, str) and task_id,
                f"task {ordinal}: task_id missing")
        task_path = root / "plan" / task.get("task_path", "")
        require(task_path == root / "plan" / "tasks" / f"{task_id}.json",
                f"{task_id}: noncanonical task path")
        task_body = read_object(task_path)
        require(task_body == {key: value for key, value in task.items()
                              if key != "task_path"},
                f"{task_id}: task body/index mismatch")
        column = task.get("column")
        require(isinstance(column, dict), f"{task_id}: column metadata missing")
        key = (column.get("dataset_id"), column.get("column_id"))
        require(all(isinstance(value, str) and value for value in key),
                f"{task_id}: invalid dataset/column identity")
        require(key not in bound, f"duplicate dataset/column identity: {key}")
        datasets.add(key[0])

        result_path = root / "results" / task_id / "RESULT.json"
        result = read_object(result_path)
        require(result.get("schema") == RESULT_SCHEMA and
                result.get("status") == "complete" and
                result.get("phase") == 1 and
                result.get("candidate_scope") == "harness_combo_pilot_v1_fixed30" and
                result.get("candidate_count") == 30,
                f"{task_id}: result scope/status mismatch")
        if "task_id" in result:
            require(result["task_id"] == task_id, f"{task_id}: result task_id mismatch")
        if "task_ordinal" in result:
            require(result["task_ordinal"] == ordinal,
                    f"{task_id}: result task ordinal mismatch")
        require(result.get("task_sha256") == sha256_file(task_path),
                f"{task_id}: result task SHA mismatch")
        require(result.get("catalog_json_sha256") == catalog_sha,
                f"{task_id}: result catalog SHA mismatch")
        require(result.get("column") == column and
                result.get("split") == task.get("split") and
                result.get("bindings") == task.get("bindings"),
                f"{task_id}: result task bindings differ")
        require(result.get("source_manifest_sha256") == column.get("manifest_sha256") and
                result.get("source_valid_value_count") == column.get("valid_value_count") and
                result.get("source_numeric_bytes") == column.get("original_numeric_bytes"),
                f"{task_id}: source manifest/value/byte binding mismatch")
        block_count = result.get("source_block_count")
        require(isinstance(block_count, int) and not isinstance(block_count, bool) and
                block_count > 0, f"{task_id}: invalid source block count")
        source_blocks += block_count
        source_values += column["valid_value_count"]
        source_bytes += column["original_numeric_bytes"]

        candidates = result.get("candidates")
        require(isinstance(candidates, list) and len(candidates) == 30,
                f"{task_id}: incomplete candidate inventory")
        seen_candidates: set[str] = set()
        rankable = []
        for candidate in candidates:
            require(isinstance(candidate, dict), f"{task_id}: invalid candidate receipt")
            identifier = candidate_id(candidate, task_id)
            require(identifier in catalog_by_id and identifier not in seen_candidates,
                    f"{task_id}: unknown or duplicate candidate {identifier}")
            seen_candidates.add(identifier)
            require(candidate.get("ordinal") == ordinal_by_id[identifier],
                    f"{task_id}/{identifier}: ordinal mismatch")
            catalog_row = catalog_by_id[identifier]
            require(candidate.get("kind") == catalog_row.get("kind") and
                    candidate.get("new_combination") ==
                    catalog_row.get("new_combination"),
                    f"{task_id}/{identifier}: catalog classification mismatch")
            require(candidate.get("status") in RESOLVED_CANDIDATE_STATUSES,
                    f"{task_id}/{identifier}: unresolved candidate status")
            if candidate["status"] == "success":
                require(success_is_strict(candidate),
                        f"{task_id}/{identifier}: success is not strict/independently verified")
                container_bytes = candidate.get("container_bytes")
                require(isinstance(container_bytes, int) and
                        not isinstance(container_bytes, bool) and container_bytes > 0,
                        f"{task_id}/{identifier}: invalid complete-container bytes")
                if "compressed_bytes" in candidate:
                    require(candidate["compressed_bytes"] == container_bytes,
                            f"{task_id}/{identifier}: byte aliases differ")
                require(candidate.get("original_numeric_bytes") ==
                        column["original_numeric_bytes"],
                        f"{task_id}/{identifier}: source-byte count differs")
                rankable.append(candidate)
            else:
                require(candidate.get("rank_eligible", False) is False and
                        candidate.get("strict", False) is False,
                        f"{task_id}/{identifier}: rejected candidate marked rankable")
        require(seen_candidates == set(catalog_by_id),
                f"{task_id}: candidate set differs from HCB30")
        require(rankable, f"{task_id}: no strict HCB30 candidate")
        bound[key] = (task, result)

    require(len(bound) == expected_columns and len(datasets) == expected_datasets,
            "bound results do not cover expected columns/datasets")
    require(source_blocks == expected_blocks,
            "bound result block coverage differs from frozen contract")
    return bound, {
        "finished_sha256": sha256_file(root / "RUN_FINISHED.json"),
        "freeze_sha256": sha256_file(freeze_path),
        "report_sha256": sha256_file(report_path),
        "catalog_sha256": catalog_sha,
        "columns": len(bound),
        "datasets": len(datasets),
        "source_blocks": source_blocks,
        "source_valid_values": source_values,
        "source_numeric_bytes": source_bytes,
    }


def relation(delta: int) -> str:
    if delta < 0:
        return "hcb30_container_has_fewer_bytes"
    if delta > 0:
        return "hcb30_container_has_more_bytes"
    return "equal_byte_counts_across_distinct_formats"


def compare(
    root: Path,
    oracle_path: Path,
    *,
    expected_oracle_sha256: str = ORACLE_SHA256,
    expected_columns: int = 3519,
    expected_datasets: int = 135,
    expected_blocks: int = 558035,
    expected_candidate_attempts: int = 105570,
    expected_development_columns: int = 3433,
    expected_confirm_columns: int = 86,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Mandatory no-oracle gate. Do not move oracle access above this call.
    bound, run_binding = validate_completed_run(
        root,
        expected_columns=expected_columns,
        expected_datasets=expected_datasets,
        expected_blocks=expected_blocks,
        expected_candidate_attempts=expected_candidate_attempts,
        expected_development_columns=expected_development_columns,
        expected_confirm_columns=expected_confirm_columns,
    )

    require(sha256_file(oracle_path) == expected_oracle_sha256,
            "wrong frozen per-column oracle")
    oracle_rows = read_jsonl(oracle_path)
    require(len(oracle_rows) == expected_columns,
            "frozen oracle does not contain every completed column")

    totals = {
        target: {
            "columns": 0,
            "hcb30_container_fewer_bytes_columns": 0,
            "equal_byte_counts_columns": 0,
            "hcb30_container_more_bytes_columns": 0,
            "hcb30_complete_container_bytes": 0,
            "reference_format_bytes": 0,
        }
        for target in REFERENCE_TARGETS
    }
    seen: set[tuple[str, str]] = set()
    output_rows: list[dict[str, Any]] = []
    for oracle in oracle_rows:
        require(oracle.get("schema") == "current135-column-oracle-ground-truth-v1",
                "unexpected oracle row schema")
        key = (oracle.get("dataset_id"), oracle.get("column_id"))
        require(key in bound and key not in seen,
                f"oracle has unknown or duplicate column: {key}")
        seen.add(key)
        task, result = bound[key]
        column = task["column"]
        require(oracle.get("manifest_sha256") == column["manifest_sha256"],
                f"{key}: oracle/result manifest SHA mismatch")
        require(oracle.get("original_numeric_bytes") ==
                column["original_numeric_bytes"],
                f"{key}: oracle/result source-byte count mismatch")
        require(oracle.get("valid_value_count") == column["valid_value_count"],
                f"{key}: oracle/result valid-value count mismatch")
        require(oracle.get("dtype") == column["dtype"] and
                oracle.get("column") == column["column"],
                f"{key}: oracle/result dtype or column name mismatch")
        require(oracle.get("joint_global_optimal") is False and
                oracle.get("joint_operator_algorithm_measured") is False,
                f"{key}: oracle unexpectedly claims a joint optimum")

        candidates = [candidate for candidate in result["candidates"]
                      if success_is_strict(candidate)]
        best = min(candidates,
                   key=lambda candidate: (candidate["container_bytes"],
                                          candidate["ordinal"],
                                          candidate_id(candidate, str(key))))
        best_id = candidate_id(best, str(key))
        hcb30_bytes = best["container_bytes"]
        row: dict[str, Any] = {
            "dataset_id": key[0],
            "column_id": key[1],
            "column": column["column"],
            "dtype": column["dtype"],
            "split": task["split"]["name"],
            "manifest_sha256": column["manifest_sha256"],
            "original_numeric_bytes": column["original_numeric_bytes"],
            "valid_value_count": column["valid_value_count"],
            "hcb30_best_candidate_id": best_id,
            "hcb30_best_candidate_kind": best.get("kind"),
            "hcb30_best_is_new_combination": best.get("new_combination"),
            "hcb30_complete_container_bytes": hcb30_bytes,
            "hcb30_container_sha256": best.get("container_sha256"),
            "hcb30_candidate_scope": "30 fixed typed recipes",
            "hcb30_byte_format": (
                "complete HCF1 column container with all blocks and inner HCB1 frames"),
            "byte_comparison_is_format_context_only": True,
            "timing_compared_to_historical_reference": False,
        }
        for target, spec in REFERENCE_TARGETS.items():
            reference_id = oracle.get(spec["id_field"])
            reference_bytes = oracle.get(spec["bytes_field"])
            require(isinstance(reference_id, str) and reference_id,
                    f"{key}/{target}: reference id missing")
            require(isinstance(reference_bytes, int) and
                    not isinstance(reference_bytes, bool) and reference_bytes > 0,
                    f"{key}/{target}: reference bytes invalid")
            reference_kind = (oracle.get(spec["kind_field"])
                              if spec["kind_field"] is not None else
                              ("algorithm" if target == "best_algorithm"
                               else "operator_zstd3"))
            delta = hcb30_bytes - reference_bytes
            row[f"{target}_reference_id"] = reference_id
            row[f"{target}_reference_kind"] = reference_kind
            row[f"{target}_reference_format_bytes"] = reference_bytes
            row[f"hcb30_minus_{target}_reference_format_bytes"] = delta
            row[f"hcb30_vs_{target}_byte_relation"] = relation(delta)
            row[f"{target}_reference_candidate_scope"] = spec["candidate_scope"]
            row[f"{target}_reference_byte_format"] = spec["format"]
            if target == "best_algorithm":
                row["best_algorithm_reference_byte_boundary"] = oracle.get(
                    "best_algorithm_byte_boundary")
            total = totals[target]
            total["columns"] += 1
            bucket = (
                "hcb30_container_fewer_bytes_columns" if delta < 0 else
                "hcb30_container_more_bytes_columns" if delta > 0 else
                "equal_byte_counts_columns"
            )
            total[bucket] += 1
            total["hcb30_complete_container_bytes"] += hcb30_bytes
            total["reference_format_bytes"] += reference_bytes
        output_rows.append(row)

    require(seen == set(bound), "oracle and completed result column sets differ")
    for total in totals.values():
        difference = (total["hcb30_complete_container_bytes"] -
                      total["reference_format_bytes"])
        total["hcb30_minus_reference_format_bytes"] = difference
        total["hcb30_to_reference_format_byte_ratio"] = (
            total["hcb30_complete_container_bytes"] /
            total["reference_format_bytes"])
        total["interpretation"] = (
            "descriptive byte context over same numeric inputs but distinct formats")

    summary = {
        "schema": "harness-combo-full135-frozen-reference-context-v1",
        "status": "complete",
        "columns": len(output_rows),
        "datasets": len({row["dataset_id"] for row in output_rows}),
        "run_binding": run_binding,
        "oracle_sha256": expected_oracle_sha256,
        "candidate_scope": {
            "hcb30": "30 fixed typed recipes measured in this completed run",
            "best_algorithm": "85 frozen algorithms",
            "best_operator": "97 frozen operators with common Zstd-3 backend",
            "best_observed_pipeline": (
                "182 previously observed algorithm or fixed-backend operator entries"),
        },
        "explicitly_excluded_from_hcb30": EXCLUDED_FAMILIES,
        "reference_targets_kept_distinct": [
            "best_algorithm", "best_operator", "best_observed_pipeline"
        ],
        "byte_contracts": {
            "hcb30": (
                "complete HCF1 column container with all blocks and inner HCB1 frames"),
            "best_algorithm": (
                "complete NGF1 algorithm frames under retained row byte boundary"),
            "best_operator": (
                "O2R1 per-block bundles with common single-thread Zstd level 3"),
            "best_observed_pipeline": (
                "native format of selected observed algorithm or operator entry"),
        },
        "same_manifest_source_bytes_and_valid_values_required": True,
        "format_normalized_byte_comparison": False,
        "timing_comparison_available": False,
        "timing_reason": (
            "historical operator times are null and historical timing contracts "
            "are not pooled with this run"),
        "full_operator_by_algorithm_cartesian_product_measured": False,
        "joint_global_optimum_claimed": False,
        "independent_optima_combined": False,
        "labels_used_for_fitting_tuning_or_candidate_selection": False,
        "confirm_labels_read_only_after_complete_3519_column_gate": True,
        "totals": totals,
        "interpretation": (
            "Post-run byte-count differences across distinct delivered formats on "
            "identically bound numeric inputs. They do not establish matched-format "
            "dominance or a joint operator-algorithm optimum."),
    }
    return summary, output_rows


def write_outputs(root: Path, summary: dict[str, Any],
                  rows: list[dict[str, Any]]) -> None:
    output = root / "report"
    atomic_write(output / "FROZEN_REFERENCE_CONTEXT.json", canonical_json(summary))
    atomic_write(
        output / "FROZEN_REFERENCE_CONTEXT.jsonl",
        b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")) + "\n").encode("utf-8")
                 for row in rows),
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    target = output / "FROZEN_REFERENCE_CONTEXT.csv"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    root = args.root.resolve(strict=True)
    # Build an absolute name without opening/statting oracle before the gate.
    oracle = args.oracle.absolute()
    summary, rows = compare(root, oracle)
    write_outputs(root, summary, rows)
    print(json.dumps({
        "status": summary["status"],
        "columns": summary["columns"],
        "datasets": summary["datasets"],
        "totals": summary["totals"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
