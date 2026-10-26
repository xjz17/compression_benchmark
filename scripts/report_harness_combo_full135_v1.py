#!/usr/bin/env python3
"""Audit and summarize per-column full-current135 harness results.

The reporter consumes only the prepared scope/tasks/catalog and newly measured
RESULT receipts.  It never opens the frozen oracle label rows and performs no
fitting, threshold search, or confirm-driven tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCOPE_SCHEMA = "harness-combo-full135-scope-v1"
TASK_SCHEMA = "harness-combo-full135-column-task-v1"
RESULT_SCHEMA = "harness-combo-full135-result-v1"
SUCCESS = {"success", "ok", "strict"}
INELIGIBLE = {"ineligible", "domain_rejected", "unsupported", "not_applicable"}
EXPECTED_COUNTS = {
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


class ReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


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
        raise ReportError(f"cannot read {path}: {error}") from error
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
                raise ReportError(f"{path}:{line_number}: {error}") from error
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def integer(value: Any, context: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{context}: expected integer")
    require(value >= minimum, f"{context}: expected >= {minimum}")
    return value


def lowercase_sha(value: Any, context: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"{context}: invalid lowercase SHA-256")
    return value


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


def pretty(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) +
            "\n").encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def parse_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    require(catalog.get("schema") == "harness-combo-full135-catalog-v1",
            "CANDIDATES was not emitted by the full135 C++ driver")
    require(catalog.get("phase") == 1 and catalog.get("candidate_count") == 30,
            "CANDIDATES phase/count mismatch")
    require(catalog.get("candidate_scope") == "harness_combo_pilot_v1_fixed30",
            "CANDIDATES candidate scope mismatch")
    require(catalog.get("no_cartesian_product_claim") is True,
            "CANDIDATES Cartesian-product disclaimer missing")
    rows = catalog.get("candidates")
    require(isinstance(rows, list) and len(rows) == 30,
            "CANDIDATES must contain fixed phase-1 pool of 30")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"candidate {index}: expected object")
        identifier = row.get("candidate_id", row.get("id"))
        require(isinstance(identifier, str) and identifier,
                f"candidate {index}: missing id")
        require(identifier not in seen, f"duplicate candidate {identifier}")
        seen.add(identifier)
        require(row.get("candidate_ordinal", index) == index,
                f"{identifier}: ordinal mismatch")
        parsed.append({**row, "candidate_id": identifier,
                       "candidate_ordinal": index})
    require("zstd3" in seen and "lz4" in seen, "missing fixed baseline candidate")
    require(sum(not bool(row.get("new_combination", False)) for row in parsed) == 18,
            "control18 membership changed")
    searchable = "\n".join(row["candidate_id"].lower() for row in parsed)
    require(not any(token.lower() in searchable for token in EXCLUDED_FAMILIES),
            "CANDIDATES contains an explicitly excluded family")
    return parsed


def validate_prepared(plan_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]],
                                                list[dict[str, Any]], str]:
    scope_path = plan_root / "SCOPE.json"
    tasks_path = plan_root / "COLUMN_TASKS.jsonl"
    catalog_path = plan_root / "CANDIDATES.json"
    scope = read_object(scope_path)
    require(scope.get("schema") == SCOPE_SCHEMA, "unexpected SCOPE schema")
    require(scope.get("status") == "ready", "SCOPE is not ready")
    require(scope.get("candidate_scope") == "harness_combo_pilot_v1_fixed30",
            "unexpected SCOPE candidate pool")
    tasks = read_jsonl(tasks_path)
    catalog = parse_catalog(read_object(catalog_path))
    require(scope.get("candidate_count") == len(catalog), "candidate count mismatch")
    require(scope.get("column_tasks_sha256") == sha256_file(tasks_path),
            "COLUMN_TASKS SHA mismatch")
    require(scope.get("inputs", {}).get("driver_catalog", {}).get("sha256") ==
            sha256_file(catalog_path), "CANDIDATES SHA mismatch")
    counts = scope.get("counts")
    require(isinstance(counts, dict), "SCOPE counts missing")
    require(all(counts.get(key) == value for key, value in EXPECTED_COUNTS.items()),
            "SCOPE is not the frozen legacy current135/3519/558035 scope")
    inputs = scope.get("inputs")
    require(isinstance(inputs, dict), "SCOPE inputs missing")
    require(inputs.get("columns", {}).get("sha256") == EXPECTED_COLUMNS_SHA256,
            "SCOPE COLUMNS binding is not frozen current135")
    require(inputs.get("split", {}).get("sha256") == EXPECTED_SPLIT_SHA256,
            "SCOPE split binding changed")
    plan_binding = inputs.get("plan_columns")
    if plan_binding is not None:
        require(plan_binding.get("sha256") == EXPECTED_PLAN_SHA256,
                "SCOPE PLAN_COLUMNS binding changed")
    require(scope.get("dataset_set_sha256") == EXPECTED_DATASET_SET_SHA256,
            "SCOPE dataset membership changed")
    require(integer(counts.get("columns"), "SCOPE.columns") == len(tasks),
            "task count mismatch")
    seen: set[str] = set()
    seen_columns: set[tuple[str, str]] = set()
    datasets: set[str] = set()
    previous_split = "development"
    for index, task in enumerate(tasks):
        require(task.get("schema") == TASK_SCHEMA, f"task {index}: schema")
        require(task.get("ordinal") == index, f"task {index}: ordinal drift")
        task_id = task.get("task_id")
        require(isinstance(task_id, str) and task_id and task_id not in seen,
                f"task {index}: invalid or duplicate task_id")
        seen.add(task_id)
        column = task.get("column")
        require(isinstance(column, dict), f"{task_id}: column binding missing")
        dataset_id = column.get("dataset_id")
        column_id = column.get("column_id")
        require(isinstance(dataset_id, str) and isinstance(column_id, str),
                f"{task_id}: dataset_id/column_id missing")
        column_key = (dataset_id, column_id)
        require(column_key not in seen_columns,
                f"duplicate dataset_id/column_id: {dataset_id}/{column_id}")
        seen_columns.add(column_key)
        datasets.add(dataset_id)
        split = task.get("split", {}).get("name")
        require(split in {"development", "confirm"}, f"{task_id}: split")
        require(not (previous_split == "confirm" and split == "development"),
                "development task appears after confirm task")
        previous_split = split
        expected_task_path = f"tasks/{task_id}.json"
        require(task.get("task_path") == expected_task_path,
                f"{task_id}: noncanonical task_path")
        task_path = plan_root / expected_task_path
        require(task_path.is_file(), f"{task_id}: task file missing")
        task_body = {key: value for key, value in task.items() if key != "task_path"}
        require(read_object(task_path) == task_body,
                f"{task_id}: task file differs from JSONL task body")
    require(len(seen_columns) == 3519 and len(datasets) == 135,
            "task index does not cover exactly 135 datasets/3519 unique columns")
    return scope, tasks, catalog, sha256_file(catalog_path)


def result_path(run_root: Path, task_id: str) -> Path | None:
    candidates = [
        run_root / "results" / task_id / "RESULT.json",
        run_root / task_id / "RESULT.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def candidate_receipt_paths(run_root: Path, task_id: str,
                            binding: dict[str, Any]) -> list[Path]:
    identifier = binding["candidate_id"]
    ordinal = binding["candidate_ordinal"]
    safe = f"c{ordinal:04d}_{hashlib.sha256(identifier.encode()).hexdigest()[:16]}"
    return [
        run_root / "results" / task_id / "candidates" / safe / "RESULT.json",
        run_root / "results" / task_id / safe / "RESULT.json",
        run_root / "results" / task_id / identifier / "RESULT.json",
    ]


def load_candidate_rows(run_root: Path, task: dict[str, Any],
                        catalog: list[dict[str, Any]], catalog_sha: str,
                        task_file: Path) -> tuple[list[dict[str, Any] | None],
                                                 str | None,
                                                 dict[str, Any] | None]:
    task_id = task["task_id"]
    aggregate_path = result_path(run_root, task_id)
    if aggregate_path is not None:
        aggregate = read_object(aggregate_path)
        require(aggregate.get("schema") == RESULT_SCHEMA,
                f"{task_id}: unexpected RESULT schema")
        require(aggregate.get("status") == "complete",
                f"{task_id}: RESULT is not terminal complete")
        require(aggregate.get("phase") == 1 and aggregate.get("candidate_count") == 30,
                f"{task_id}: phase/candidate scope mismatch")
        require(aggregate.get("candidate_scope") ==
                "harness_combo_pilot_v1_fixed30",
                f"{task_id}: candidate scope mismatch")
        if "task_id" in aggregate:
            require(aggregate["task_id"] == task_id,
                    f"{task_id}: RESULT task_id mismatch")
        require(aggregate.get("task_sha256") == sha256_file(task_file),
                f"{task_id}: task SHA mismatch")
        require(aggregate.get("source_manifest_sha256") ==
                task["column"]["manifest_sha256"],
                f"{task_id}: source manifest SHA mismatch")
        require(aggregate.get("catalog_json_sha256") == catalog_sha,
                f"{task_id}: catalog SHA mismatch")
        require(aggregate.get("source_valid_value_count") ==
                task["column"]["valid_value_count"],
                f"{task_id}: source valid-value count mismatch")
        require(aggregate.get("source_numeric_bytes") ==
                task["column"]["original_numeric_bytes"],
                f"{task_id}: source numeric-byte count mismatch")
        require(aggregate.get("column") == task["column"],
                f"{task_id}: result column binding mismatch")
        require(aggregate.get("split") == task["split"],
                f"{task_id}: result split binding mismatch")
        require(aggregate.get("bindings") == task["bindings"],
                f"{task_id}: result input bindings mismatch")
        integer(aggregate.get("source_block_count"),
                f"{task_id}.source_block_count", minimum=1)
        lowercase_sha(aggregate.get("ordered_source_blocks_sha256"),
                      f"{task_id}.ordered_source_blocks_sha256")
        rows = aggregate.get("candidates")
        require(isinstance(rows, list) and len(rows) == 30,
                f"{task_id}: RESULT must contain all 30 candidates")
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            require(isinstance(row, dict), f"{task_id}: candidate result not object")
            identifier = row.get("candidate_id", row.get("id"))
            require(isinstance(identifier, str) and identifier,
                    f"{task_id}: candidate id missing")
            require(identifier not in by_id, f"{task_id}: duplicate candidate {identifier}")
            by_id[identifier] = row
        expected = {row["candidate_id"] for row in catalog}
        require(set(by_id) <= expected, f"{task_id}: unknown candidate result")
        return ([by_id.get(row["candidate_id"]) for row in catalog],
                str(aggregate_path), aggregate)

    output: list[dict[str, Any] | None] = []
    found_path: str | None = None
    for binding in catalog:
        selected = None
        for path in candidate_receipt_paths(run_root, task_id, binding):
            if path.is_file():
                selected = path
                break
        if selected is None:
            output.append(None)
        else:
            found_path = str(selected.parent.parent)
            output.append(read_object(selected))
    return output, found_path, None


def timing_samples(row: dict[str, Any], operation: str) -> list[int] | None:
    nested = row.get(operation)
    samples: Any = None
    if isinstance(nested, dict):
        samples = nested.get("wall_samples_ns")
    if samples is None:
        samples = row.get(f"{operation}_wall_samples_ns")
    if not isinstance(samples, list) or len(samples) != 3:
        return None
    result: list[int] = []
    for sample in samples:
        if not isinstance(sample, int) or isinstance(sample, bool) or sample <= 0:
            # A zero observation is timer censoring/insufficient resolution,
            # never evidence of zero cost.
            return None
        result.append(sample)
    return result


def median_ns(samples: list[int] | None) -> int | None:
    return int(statistics.median(samples)) if samples is not None else None


def validate_frame(row: dict[str, Any], verify_frames: bool,
                   context: str) -> tuple[int, str, str]:
    compressed = row.get("container_bytes", row.get("compressed_bytes"))
    compressed = integer(compressed, f"{context}.container_bytes", minimum=1)
    path_text = row.get("container_path", row.get("frame_path"))
    sha = row.get("container_sha256", row.get("frame_sha256"))
    require(isinstance(path_text, str) and path_text, f"{context}: container path missing")
    sha = lowercase_sha(sha, f"{context}.container_sha256")
    if verify_frames:
        path = Path(path_text)
        require(path.is_file(), f"{context}: retained frame missing: {path}")
        require(path.stat().st_size == compressed, f"{context}: frame size mismatch")
        require(sha256_file(path) == sha, f"{context}: frame SHA mismatch")
    return compressed, path_text, sha


def validate_receipt(row: dict[str, Any], verify_files: bool,
                     context: str) -> tuple[str, str]:
    path_text = row.get("receipt_path")
    sha = row.get("receipt_sha256")
    require(isinstance(path_text, str) and path_text,
            f"{context}: verifier receipt path missing")
    sha = lowercase_sha(sha, f"{context}.receipt_sha256")
    if verify_files:
        path = Path(path_text)
        require(path.is_file(), f"{context}: verifier receipt missing: {path}")
        require(sha256_file(path) == sha, f"{context}: verifier receipt SHA mismatch")
    return path_text, sha


def normalize_candidate(row: dict[str, Any] | None, binding: dict[str, Any],
                        task: dict[str, Any], *, verify_frames: bool) -> dict[str, Any]:
    identifier = binding["candidate_id"]
    base = {
        "candidate_id": identifier,
        "candidate_ordinal": binding["candidate_ordinal"],
        "kind": binding.get("kind"),
        "new_combination": bool(binding.get("new_combination", False)),
    }
    if row is None:
        return {**base, "classification": "missing", "status": "missing"}
    declared = row.get("candidate_id", row.get("id", identifier))
    require(declared == identifier, f"{task['task_id']}/{identifier}: id mismatch")
    if "ordinal" in row:
        require(row["ordinal"] == binding["candidate_ordinal"],
                f"{task['task_id']}/{identifier}: ordinal mismatch")
    status = row.get("status")
    require(isinstance(status, str) and status,
            f"{task['task_id']}/{identifier}: missing status")
    context = f"{task['task_id']}/{identifier}"
    if status in INELIGIBLE:
        return {**base, "classification": "ineligible", "status": status,
                "failure_reason": row.get("failure_reason", row.get("reason"))}
    if status not in SUCCESS:
        return {**base, "classification": "failure", "status": status,
                "failure_reason": row.get("failure_reason", row.get("error"))}
    independent_decode = row.get("independent_decode")
    independent_verified = (
        isinstance(independent_decode, dict) and
        independent_decode.get("verified") is True
    ) or row.get("bit_exact_fresh_exec") is True
    same_process_verified = (
        row.get("bit_exact") is True or
        row.get("bit_exact_same_process") is True
    )
    strict_valid = (row.get("strict") is True and
                    row.get("rank_eligible", True) is True and
                    row.get("complete_column", True) is True and
                    same_process_verified and independent_verified)
    if not strict_valid:
        return {**base, "classification": "failure", "status": "verify_failed",
                "failure_reason": "success receipt lacks strict bit-exact independent verification"}
    compressed, frame_path, frame_sha = validate_frame(row, verify_frames, context)
    receipt_path, receipt_sha = validate_receipt(row, verify_frames, context)
    original = integer(row.get("original_numeric_bytes",
                               task["column"]["original_numeric_bytes"]),
                       f"{context}.original_numeric_bytes", minimum=1)
    require(original == task["column"]["original_numeric_bytes"],
            f"{context}: source bytes mismatch")
    encode_samples = timing_samples(row, "encode")
    decode_samples = timing_samples(row, "decode")
    return {
        **base,
        "classification": "strict",
        "status": status,
        "compressed_bytes": compressed,
        "compression_ratio": compressed / original,
        "encode_wall_samples_ns": encode_samples,
        "encode_median_wall_ns": median_ns(encode_samples),
        "decode_wall_samples_ns": decode_samples,
        "decode_median_wall_ns": median_ns(decode_samples),
        "container_path": frame_path,
        "container_sha256": frame_sha,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha,
        "cold_verify_wall_ns": row.get("cold_verify_wall_ns"),
        "zero_timing_treated_as_zero_cost": False,
    }


def metric_entry(row: dict[str, Any], metric: str) -> dict[str, Any]:
    output = {"candidate_id": row["candidate_id"],
              "candidate_ordinal": row["candidate_ordinal"],
              "kind": row.get("kind"),
              "new_combination": row["new_combination"]}
    if metric == "bytes":
        output.update(compressed_bytes=row["compressed_bytes"],
                      compression_ratio=row["compression_ratio"])
    else:
        output[f"{metric}_median_wall_ns"] = row[f"{metric}_median_wall_ns"]
        output[f"{metric}_wall_samples_ns"] = row[f"{metric}_wall_samples_ns"]
    return output


def best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    if metric == "bytes":
        eligible = rows
        key = lambda row: (row["compressed_bytes"], row["candidate_ordinal"])
    else:
        field = f"{metric}_median_wall_ns"
        eligible = [row for row in rows if row.get(field) is not None]
        key = lambda row: (row[field], row["candidate_ordinal"])
    if not eligible:
        return None
    return min(eligible, key=key)


def speed_comparison(candidate: dict[str, Any] | None,
                     baseline: dict[str, Any] | None, operation: str) -> dict[str, Any] | None:
    if candidate is None or baseline is None:
        return None
    candidate_samples = candidate.get(f"{operation}_wall_samples_ns")
    baseline_samples = baseline.get(f"{operation}_wall_samples_ns")
    if candidate_samples is None or baseline_samples is None:
        return {"available": False, "reason": "three positive wall samples required"}
    candidate_median = candidate[f"{operation}_median_wall_ns"]
    baseline_median = baseline[f"{operation}_median_wall_ns"]
    faster_samples = sum(left < right for left, right in
                         zip(candidate_samples, baseline_samples))
    return {
        "available": True,
        "candidate_median_wall_ns": candidate_median,
        "baseline_median_wall_ns": baseline_median,
        "median_ratio": candidate_median / baseline_median,
        "median_at_most_95_percent": candidate_median * 100 <= baseline_median * 95,
        "paired_samples_faster": faster_samples,
        "paired_samples_required": 2,
        "qualifies_as_faster": (
            candidate_median * 100 <= baseline_median * 95 and faster_samples >= 2),
        "threshold_frozen": "median<=95% of baseline and >=2/3 aligned samples faster",
    }


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows
                if row.get("encode_median_wall_ns") is not None and
                row.get("decode_median_wall_ns") is not None]
    output: list[dict[str, Any]] = []
    for row in eligible:
        point = (row["compressed_bytes"], row["encode_median_wall_ns"],
                 row["decode_median_wall_ns"])
        dominated = False
        for other in eligible:
            if other is row:
                continue
            candidate = (other["compressed_bytes"], other["encode_median_wall_ns"],
                         other["decode_median_wall_ns"])
            if all(left <= right for left, right in zip(candidate, point)) and \
                    any(left < right for left, right in zip(candidate, point)):
                dominated = True
                break
        if not dominated:
            output.append({
                "candidate_id": row["candidate_id"],
                "candidate_ordinal": row["candidate_ordinal"],
                "kind": row.get("kind"),
                "new_combination": row["new_combination"],
                "compressed_bytes": row["compressed_bytes"],
                "encode_median_wall_ns": row["encode_median_wall_ns"],
                "decode_median_wall_ns": row["decode_median_wall_ns"],
            })
    return sorted(output, key=lambda row: (
        row["compressed_bytes"], row["encode_median_wall_ns"],
        row["decode_median_wall_ns"], row["candidate_ordinal"]))


def baseline_summary(strict: list[dict[str, Any]], best_bytes: dict[str, Any] | None,
                     best_encode: dict[str, Any] | None,
                     best_decode: dict[str, Any] | None) -> dict[str, Any]:
    by_id = {row["candidate_id"]: row for row in strict}
    control = [row for row in strict if not row["new_combination"]]
    baselines = {
        "zstd3": by_id.get("zstd3"),
        "lz4": by_id.get("lz4"),
        "control18_bytes": best(control, "bytes"),
        "control18_encode": best(control, "encode"),
        "control18_decode": best(control, "decode"),
    }
    result: dict[str, Any] = {}
    for name in ("zstd3", "lz4", "control18_bytes"):
        baseline = baselines[name]
        if baseline is None or best_bytes is None:
            result[name] = {"available": False}
        else:
            result[name] = {
                "available": True,
                "baseline_candidate_id": baseline["candidate_id"],
                "baseline_bytes": baseline["compressed_bytes"],
                "best_bytes": best_bytes["compressed_bytes"],
                "saved_bytes": baseline["compressed_bytes"] - best_bytes["compressed_bytes"],
                "relative_byte_improvement": (
                    baseline["compressed_bytes"] - best_bytes["compressed_bytes"]
                ) / baseline["compressed_bytes"],
            }
    for operation, selected in (("encode", best_encode), ("decode", best_decode)):
        for name, baseline in (("zstd3", baselines["zstd3"]),
                               ("lz4", baselines["lz4"]),
                               ("control18", baselines[f"control18_{operation}"])):
            result[f"{name}_{operation}"] = speed_comparison(
                selected, baseline, operation)
    return result


def aggregate_comparisons(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    byte_metrics: dict[str, Any] = {}
    for name in ("zstd3", "lz4", "control18_bytes"):
        rows = [decision["comparisons"][name] for decision in decisions
                if decision["comparisons"][name].get("available") is True]
        baseline_bytes = sum(row["baseline_bytes"] for row in rows)
        selected_bytes = sum(row["best_bytes"] for row in rows)
        saved_bytes = baseline_bytes - selected_bytes
        byte_metrics[name] = {
            "available_columns": len(rows),
            "strictly_better_columns": sum(row["saved_bytes"] > 0 for row in rows),
            "tied_columns": sum(row["saved_bytes"] == 0 for row in rows),
            "baseline_bytes": baseline_bytes,
            "selected_best_bytes": selected_bytes,
            "saved_bytes": saved_bytes,
            "relative_byte_improvement": (
                saved_bytes / baseline_bytes if baseline_bytes else None),
        }
    speed_metrics: dict[str, Any] = {}
    for operation in ("encode", "decode"):
        for name in ("zstd3", "lz4", "control18"):
            key = f"{name}_{operation}"
            rows = [decision["comparisons"][key] for decision in decisions
                    if isinstance(decision["comparisons"].get(key), dict) and
                    decision["comparisons"][key].get("available") is True]
            speed_metrics[key] = {
                "available_columns": len(rows),
                "qualifying_faster_columns": sum(
                    row["qualifies_as_faster"] for row in rows),
                "median_at_most_95_percent_columns": sum(
                    row["median_at_most_95_percent"] for row in rows),
                "paired_2_of_3_faster_columns": sum(
                    row["paired_samples_faster"] >= 2 for row in rows),
            }
    winner_counts: dict[str, dict[str, int]] = {}
    for metric in ("best_bytes", "best_encode", "best_decode"):
        winner_counts[metric] = dict(sorted(Counter(
            decision[metric]["candidate_id"] for decision in decisions
            if decision[metric] is not None
        ).items()))
    return {
        "columns": len(decisions),
        "bytes": byte_metrics,
        "speed": speed_metrics,
        "winner_counts": winner_counts,
    }


def report(plan_root: Path, run_root: Path, *, verify_frames: bool) -> tuple[dict[str, Any],
                                                                             list[dict[str, Any]]]:
    scope, tasks, catalog, catalog_sha = validate_prepared(plan_root)
    bindings = [{
        "candidate_id": row["candidate_id"],
        "candidate_ordinal": row["candidate_ordinal"],
        "kind": row.get("kind"),
        "new_combination": bool(row.get("new_combination", False)),
    } for row in catalog]
    totals = Counter()
    source_blocks_observed = 0
    aggregate_results_observed = 0
    decisions: list[dict[str, Any]] = []
    for task in tasks:
        task_file = plan_root / task["task_path"]
        candidate_rows, source_path, aggregate = load_candidate_rows(
            run_root, task, catalog, catalog_sha, task_file)
        if aggregate is not None:
            aggregate_results_observed += 1
            source_blocks_observed += aggregate["source_block_count"]
        normalized = [normalize_candidate(row, binding, task,
                                          verify_frames=verify_frames)
                      for row, binding in zip(candidate_rows, bindings)]
        counts = Counter(row["classification"] for row in normalized)
        totals.update(counts)
        strict = [row for row in normalized if row["classification"] == "strict"]
        encode_timing_gaps = [row["candidate_id"] for row in strict
                              if row["encode_median_wall_ns"] is None]
        decode_timing_gaps = [row["candidate_id"] for row in strict
                              if row["decode_median_wall_ns"] is None]
        totals["strict_encode_timing_unavailable"] += len(encode_timing_gaps)
        totals["strict_decode_timing_unavailable"] += len(decode_timing_gaps)
        best_bytes = best(strict, "bytes")
        best_encode = best(strict, "encode")
        best_decode = best(strict, "decode")
        missing = counts["missing"]
        failures = counts["failure"]
        if aggregate is None and source_path is not None:
            column_status = "partial"
        elif missing:
            column_status = "running" if not strict and not failures and not counts["ineligible"] else "partial"
        elif failures:
            column_status = "complete_with_failures"
        else:
            column_status = "complete"
        decisions.append({
            "schema": "harness-combo-full135-column-decision-v1",
            "task_id": task["task_id"],
            "ordinal": task["ordinal"],
            "dataset_id": task["column"]["dataset_id"],
            "column_id": task["column"]["column_id"],
            "column": task["column"]["column"],
            "dtype": task["column"]["dtype"],
            "split": task["split"]["name"],
            "dev_fold": task["split"]["fold"],
            "source_group": task["split"]["source_group"],
            "original_numeric_bytes": task["column"]["original_numeric_bytes"],
            "status": column_status,
            "result_source": source_path,
            "bound_column_result": aggregate is not None,
            "counts": dict(counts),
            "best_bytes": metric_entry(best_bytes, "bytes") if best_bytes else None,
            "best_encode": metric_entry(best_encode, "encode") if best_encode else None,
            "best_decode": metric_entry(best_decode, "decode") if best_decode else None,
            "pareto": pareto(strict),
            "comparisons": baseline_summary(strict, best_bytes, best_encode, best_decode),
            "failure_candidates": [
                {"candidate_id": row["candidate_id"], "status": row["status"],
                 "reason": row.get("failure_reason")}
                for row in normalized if row["classification"] == "failure"
            ],
            "ineligible_candidates": [row["candidate_id"] for row in normalized
                                      if row["classification"] == "ineligible"],
            "missing_candidates": [row["candidate_id"] for row in normalized
                                   if row["classification"] == "missing"],
            "strict_encode_timing_unavailable": encode_timing_gaps,
            "strict_decode_timing_unavailable": decode_timing_gaps,
            "single_metric_optima_are_not_combined": True,
            "oracle_labels_read_or_used": False,
        })

    expected_attempts = len(tasks) * len(catalog)
    missing = totals["missing"]
    failures = totals["failure"]
    observed = expected_attempts - missing
    bound_all_columns = aggregate_results_observed == len(tasks)
    if bound_all_columns:
        require(source_blocks_observed == scope["counts"]["blocks"],
                "complete bound results do not cover all 558035 source blocks")
    if observed == 0:
        status = "running"
    elif missing or not bound_all_columns:
        status = "partial"
    elif failures:
        status = "complete_with_failures"
    else:
        status = "complete"
    summary = {
        "schema": "harness-combo-full135-report-v1",
        "status": status,
        "scope": "frozen legacy_current135 phase1 fixed30",
        "candidate_scope": {
            "name": "harness_combo_pilot_v1_fixed30",
            "candidates": 30,
            "control_candidates": 18,
            "new_combination_candidates": 12,
            "explicitly_excluded_families": EXCLUDED_FAMILIES,
            "full_cartesian_product": False,
        },
        "coverage": {
            "expected_datasets": scope["counts"]["datasets"],
            "expected_columns": len(tasks),
            "expected_candidate_attempts": expected_attempts,
            "observed_terminal_attempts": observed,
            "strict": totals["strict"],
            "ineligible": totals["ineligible"],
            "failures": failures,
            "missing": missing,
            "columns_complete": sum(row["status"] == "complete" for row in decisions),
            "columns_complete_with_failures": sum(
                row["status"] == "complete_with_failures" for row in decisions),
            "columns_partial_or_running": sum(
                row["status"] in {"partial", "running"} for row in decisions),
            "columns_with_strict_result": sum(
                row["counts"].get("strict", 0) > 0 for row in decisions),
            "bound_column_results": aggregate_results_observed,
            "source_blocks_in_bound_results": source_blocks_observed,
            "strict_encode_timing_unavailable": totals[
                "strict_encode_timing_unavailable"],
            "strict_decode_timing_unavailable": totals[
                "strict_decode_timing_unavailable"],
        },
        "bindings": {
            "scope_sha256": sha256_file(plan_root / "SCOPE.json"),
            "column_tasks_sha256": sha256_file(plan_root / "COLUMN_TASKS.jsonl"),
            "catalog_sha256": catalog_sha,
        },
        "timing_rule": {
            "source": "three positive wall-clock samples",
            "faster": "candidate median <=95% baseline median and candidate faster in >=2/3 aligned samples",
            "zero_sample_policy": "zero is timer censoring/insufficient resolution, not zero cost",
            "tuned": False,
        },
        "decision_contract": {
            "best_bytes_best_encode_best_decode_are_independent": True,
            "independent_optima_combined_into_joint_optimum": False,
            "pareto_dimensions": ["complete_frame_bytes", "encode_median_wall_ns",
                                  "decode_median_wall_ns"],
        },
        "comparative_outcomes": {
            "all_observed_columns": aggregate_comparisons(decisions),
            "development": aggregate_comparisons(
                [row for row in decisions if row["split"] == "development"]),
            "confirm_final_evaluation_only": aggregate_comparisons(
                [row for row in decisions if row["split"] == "confirm"]),
        },
        "all_expected_results_terminal": missing == 0 and bound_all_columns,
        "all_retained_strict_frames_verified": verify_frames,
        "oracle_labels_read_or_used": False,
        "confirm_used_for_tuning": False,
    }
    return summary, decisions


def write_outputs(output_root: Path, summary: dict[str, Any],
                  decisions: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write(output_root / "REPORT.json", pretty(summary))
    jsonl = b"".join(canonical_line(row) for row in decisions)
    atomic_write(output_root / "COLUMN_DECISIONS.jsonl", jsonl)
    csv_path = output_root / "COLUMN_DECISIONS.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        fields = [
            "ordinal", "dataset_id", "column_id", "column", "dtype", "split",
            "status", "strict", "ineligible", "failures", "missing",
            "best_bytes_id", "best_bytes", "best_encode_id",
            "best_encode_median_wall_ns", "best_decode_id",
            "best_decode_median_wall_ns", "pareto_count",
            "saved_bytes_vs_zstd3", "saved_bytes_vs_lz4",
            "saved_bytes_vs_control18", "faster_encode_vs_zstd3",
            "faster_encode_vs_lz4", "faster_encode_vs_control18",
            "faster_decode_vs_zstd3", "faster_decode_vs_lz4",
            "faster_decode_vs_control18",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in decisions:
            writer.writerow({
                "ordinal": row["ordinal"], "dataset_id": row["dataset_id"],
                "column_id": row["column_id"], "column": row["column"],
                "dtype": row["dtype"], "split": row["split"],
                "status": row["status"],
                "strict": row["counts"].get("strict", 0),
                "ineligible": row["counts"].get("ineligible", 0),
                "failures": row["counts"].get("failure", 0),
                "missing": row["counts"].get("missing", 0),
                "best_bytes_id": (row["best_bytes"] or {}).get("candidate_id"),
                "best_bytes": (row["best_bytes"] or {}).get("compressed_bytes"),
                "best_encode_id": (row["best_encode"] or {}).get("candidate_id"),
                "best_encode_median_wall_ns": (row["best_encode"] or {}).get(
                    "encode_median_wall_ns"),
                "best_decode_id": (row["best_decode"] or {}).get("candidate_id"),
                "best_decode_median_wall_ns": (row["best_decode"] or {}).get(
                    "decode_median_wall_ns"),
                "pareto_count": len(row["pareto"]),
                "saved_bytes_vs_zstd3": row["comparisons"]["zstd3"].get(
                    "saved_bytes"),
                "saved_bytes_vs_lz4": row["comparisons"]["lz4"].get(
                    "saved_bytes"),
                "saved_bytes_vs_control18": row["comparisons"][
                    "control18_bytes"].get("saved_bytes"),
                "faster_encode_vs_zstd3": (
                    row["comparisons"].get("zstd3_encode") or {}).get(
                        "qualifies_as_faster"),
                "faster_encode_vs_lz4": (
                    row["comparisons"].get("lz4_encode") or {}).get(
                        "qualifies_as_faster"),
                "faster_encode_vs_control18": (
                    row["comparisons"].get("control18_encode") or {}).get(
                        "qualifies_as_faster"),
                "faster_decode_vs_zstd3": (
                    row["comparisons"].get("zstd3_decode") or {}).get(
                        "qualifies_as_faster"),
                "faster_decode_vs_lz4": (
                    row["comparisons"].get("lz4_decode") or {}).get(
                        "qualifies_as_faster"),
                "faster_decode_vs_control18": (
                    row["comparisons"].get("control18_decode") or {}).get(
                        "qualifies_as_faster"),
            })
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, csv_path)
    coverage = summary["coverage"]
    comparison = summary["comparative_outcomes"]["all_observed_columns"]
    comparison_lines = []
    for name in ("zstd3", "lz4", "control18_bytes"):
        row = comparison["bytes"][name]
        comparison_lines.append(
            f"| {name} | {row['available_columns']} | "
            f"{row['strictly_better_columns']} | {row['saved_bytes']} | "
            f"{row['relative_byte_improvement']} |")
    markdown = "\n".join([
        "# Full current135 harness combination report",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "| coverage | count |",
        "|---|---:|",
        f"| expected columns | {coverage['expected_columns']} |",
        f"| expected candidate attempts | {coverage['expected_candidate_attempts']} |",
        f"| observed terminal attempts | {coverage['observed_terminal_attempts']} |",
        f"| bound column results | {coverage['bound_column_results']} |",
        f"| source blocks in bound results | {coverage['source_blocks_in_bound_results']} |",
        f"| strict | {coverage['strict']} |",
        f"| ineligible/domain rejected | {coverage['ineligible']} |",
        f"| failures | {coverage['failures']} |",
        f"| missing | {coverage['missing']} |",
        f"| strict encode timing unavailable | {coverage['strict_encode_timing_unavailable']} |",
        f"| strict decode timing unavailable | {coverage['strict_decode_timing_unavailable']} |",
        "",
        "| byte baseline | comparable columns | better columns | saved bytes | aggregate relative improvement |",
        "|---|---:|---:|---:|---:|",
        *comparison_lines,
        "",
        "Compression bytes, encode time, and decode time are selected independently. "
        "They are not combined into an unmeasured joint optimum. Timing improvements "
        "use the frozen median<=95% plus 2/3 paired-sample rule; zero timer samples "
        "are treated as unavailable rather than zero cost.",
        "",
        "Confirm metadata may appear in final results, but confirm labels were not "
        "read and confirm results were not used for tuning.",
        "",
    ]).encode("utf-8")
    atomic_write(output_root / "SUMMARY.md", markdown)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", "--output-root", dest="output_root",
                        type=Path, required=True)
    parser.add_argument("--no-verify-frames", action="store_true",
                        help="permit reporting copied receipts when frame files are unavailable")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plan_root = args.plan_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    summary, decisions = report(plan_root, run_root,
                                verify_frames=not args.no_verify_frames)
    write_outputs(output_root, summary, decisions)
    print(json.dumps({
        "status": summary["status"],
        "output_root": str(output_root),
        **summary["coverage"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
