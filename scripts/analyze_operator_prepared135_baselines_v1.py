#!/usr/bin/env python3
"""Descriptive comparison with the two pre-measured fixed codecs.

The script is post-selection only.  It reads a completed 3,519-column
prepared-selector run and its already completed primary report.  It never
chooses a baseline, changes a threshold, fits a model, or treats a partial run
as complete evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SCHEMA = "operator-prepared135-fixed-baseline-analysis-v1"
EXPECTED_COLUMNS = 3519
EXPECTED_DATASETS = 135
EXPECTED_DEVELOPMENT_COLUMNS = 3433
EXPECTED_CONFIRM_COLUMNS = 86
POLICIES = ("per_block", "reuse")
OBJECTIVES = ("space", "encode", "decode", "balanced")
BASELINES = ("lz4frame", "zstd3")
TIME_FIELDS = ("cpu_ns", "wall_ns")


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read JSON {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{number} is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read JSONL {path}: {error}") from error
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def integer(value: Any, label: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{label} must be an integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def positive(value: Any, label: str) -> int:
    return integer(value, label, 1)


def timing(value: Any, label: str, *, require_positive: bool = False) -> dict[str, int]:
    require(isinstance(value, dict) and set(value) == set(TIME_FIELDS),
            f"{label} must contain exactly cpu_ns and wall_ns")
    parse = positive if require_positive else integer
    return {field: parse(value[field], f"{label}.{field}") for field in TIME_FIELDS}


def ratio(numerator: int, denominator: int) -> float:
    require(numerator > 0 and denominator > 0,
            "descriptive ratios require positive numerators and denominators")
    return numerator / denominator


def geometric_mean(log_sum: float, count: int) -> float:
    require(count > 0, "cannot calculate an empty geometric mean")
    return math.exp(log_sum / count)


def key(row: dict[str, Any], label: str) -> tuple[str, str]:
    dataset = row.get("dataset_id")
    column = row.get("column_id")
    require(isinstance(dataset, str) and dataset and
            isinstance(column, str) and column,
            f"{label} lacks dataset_id/column_id")
    return dataset, column


def unique(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        item_key = key(row, label)
        require(item_key not in result, f"duplicate {label} key: {item_key}")
        result[item_key] = row
    return result


def add_time(destination: dict[str, int], source: dict[str, int]) -> None:
    for field in TIME_FIELDS:
        destination[field] += source[field]


def empty_time() -> dict[str, int]:
    return {field: 0 for field in TIME_FIELDS}


def empty_metric() -> dict[str, Any]:
    return {
        "columns": 0,
        "selected_complete_frame_bytes": 0,
        "fixed_complete_frame_bytes": 0,
        "selected_decision_encode_container": empty_time(),
        "fixed_encode": empty_time(),
        "selected_decode": empty_time(),
        "fixed_decode": empty_time(),
        "column_log_sums": {
            "complete_frame_size": 0.0,
            "encode_cpu": 0.0,
            "encode_wall": 0.0,
            "decode_cpu": 0.0,
            "decode_wall": 0.0,
        },
        "all_three_no_worse_cpu_count": 0,
        "all_three_no_worse_wall_count": 0,
        "datasets": {},
    }


def empty_scope() -> dict[str, Any]:
    return {
        "columns": 0,
        "datasets": set(),
        "metrics": {
            policy: {
                objective: {baseline: empty_metric() for baseline in BASELINES}
                for objective in OBJECTIVES
            }
            for policy in POLICIES
        },
    }


def metric_ratios(values: dict[str, Any]) -> dict[str, float]:
    return {
        "complete_frame_size": ratio(
            values["selected_complete_frame_bytes"],
            values["fixed_complete_frame_bytes"],
        ),
        "encode_cpu": ratio(
            values["selected_decision_encode_container"]["cpu_ns"],
            values["fixed_encode"]["cpu_ns"],
        ),
        "encode_wall": ratio(
            values["selected_decision_encode_container"]["wall_ns"],
            values["fixed_encode"]["wall_ns"],
        ),
        "decode_cpu": ratio(
            values["selected_decode"]["cpu_ns"],
            values["fixed_decode"]["cpu_ns"],
        ),
        "decode_wall": ratio(
            values["selected_decode"]["wall_ns"],
            values["fixed_decode"]["wall_ns"],
        ),
    }


def add_observation(metric: dict[str, Any], dataset_id: str,
                    observation: dict[str, Any]) -> None:
    metric["columns"] += 1
    for field in ("selected_complete_frame_bytes", "fixed_complete_frame_bytes"):
        metric[field] += observation[field]
    for field in ("selected_decision_encode_container", "fixed_encode",
                  "selected_decode", "fixed_decode"):
        add_time(metric[field], observation[field])
    ratios = observation["ratios"]
    for name, value in ratios.items():
        require(math.isfinite(value) and value > 0,
                f"invalid per-column ratio {name}={value}")
        metric["column_log_sums"][name] += math.log(value)
    for clock in ("cpu", "wall"):
        if observation[f"all_three_no_worse_{clock}"]:
            metric[f"all_three_no_worse_{clock}_count"] += 1
    dataset = metric["datasets"].setdefault(dataset_id, {
        "selected_complete_frame_bytes": 0,
        "fixed_complete_frame_bytes": 0,
        "selected_decision_encode_container": empty_time(),
        "fixed_encode": empty_time(),
        "selected_decode": empty_time(),
        "fixed_decode": empty_time(),
    })
    for field in ("selected_complete_frame_bytes", "fixed_complete_frame_bytes"):
        dataset[field] += observation[field]
    for field in ("selected_decision_encode_container", "fixed_encode",
                  "selected_decode", "fixed_decode"):
        add_time(dataset[field], observation[field])


def finalise_metric(metric: dict[str, Any]) -> dict[str, Any]:
    columns = metric["columns"]
    require(columns > 0, "cannot finalise an empty fixed-baseline metric")
    ratio_of_sums = metric_ratios(metric)
    column_geometric = {
        name: geometric_mean(value, columns)
        for name, value in metric["column_log_sums"].items()
    }
    dataset_log_sums = Counter()
    for values in metric["datasets"].values():
        for name, value in metric_ratios(values).items():
            dataset_log_sums[name] += math.log(value)
    dataset_count = len(metric["datasets"])
    dataset_geometric = {
        name: geometric_mean(dataset_log_sums[name], dataset_count)
        for name in ratio_of_sums
    }
    return {
        "columns": columns,
        "datasets": dataset_count,
        "raw_sums": {
            key: value for key, value in metric.items()
            if key in {
                "selected_complete_frame_bytes", "fixed_complete_frame_bytes",
                "selected_decision_encode_container", "fixed_encode",
                "selected_decode", "fixed_decode",
            }
        },
        "ratio_of_sums": ratio_of_sums,
        "column_equal_geometric_mean": column_geometric,
        "dataset_equal_geometric_mean": dataset_geometric,
        "joint_nonworse_primary": "cpu",
        "all_three_no_worse_cpu": {
            "definition": "complete-frame ratio <= 1 and encode-CPU ratio <= 1 and decode-CPU ratio <= 1",
            "count": metric["all_three_no_worse_cpu_count"],
            "fraction": metric["all_three_no_worse_cpu_count"] / columns,
            "threshold_fixed_at_one_not_tuned": True,
        },
        "all_three_no_worse_wall": {
            "definition": "complete-frame ratio <= 1 and encode-wall ratio <= 1 and decode-wall ratio <= 1",
            "count": metric["all_three_no_worse_wall_count"],
            "fraction": metric["all_three_no_worse_wall_count"] / columns,
            "threshold_fixed_at_one_not_tuned": True,
            "secondary_shared_host_measure": True,
        },
    }


def finalise_scope(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "columns": scope["columns"],
        "datasets": len(scope["datasets"]),
        "metrics": {
            policy: {
                objective: {
                    baseline: finalise_metric(scope["metrics"][policy][objective][baseline])
                    for baseline in BASELINES
                }
                for objective in OBJECTIVES
            }
            for policy in POLICIES
        },
    }


def locate_report(path: Path) -> tuple[Path, Path, Path]:
    resolved = path.resolve(strict=True)
    report_path = resolved / "REPORT.json" if resolved.is_dir() else resolved
    require(report_path.name == "REPORT.json", "--report must be REPORT.json or its directory")
    root = report_path.parent
    return report_path, root / "COLUMNS.jsonl", root / "COMPLETE.json"


def validate_primary_report(report_path: Path, columns_path: Path,
                            complete_path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    complete = read_json(complete_path)
    require(complete.get("schema") == "operator-prepared135-report-complete-v1" and
            complete.get("status") == "complete_strict_verified" and
            complete.get("verified_columns") == EXPECTED_COLUMNS and
            complete.get("verified_datasets") == EXPECTED_DATASETS,
            "primary report COMPLETE.json is not full current135 evidence")
    require(complete.get("report_sha256") == sha256(report_path) and
            complete.get("columns_jsonl_sha256") == sha256(columns_path),
            "primary report artifact SHA mismatch")
    report = read_json(report_path)
    coverage = report.get("coverage", {})
    require(report.get("status") == "complete_strict_verified" and
            coverage.get("verified_columns") == EXPECTED_COLUMNS and
            coverage.get("verified_datasets") == EXPECTED_DATASETS and
            coverage.get("development_columns") == EXPECTED_DEVELOPMENT_COLUMNS and
            coverage.get("confirm_columns") == EXPECTED_CONFIRM_COLUMNS and
            coverage.get("failed_columns") == 0,
            "primary report is not a zero-failure 3,519-column report")
    require(report.get("confirm_labels_used_for_fit_threshold_or_model_selection") is False,
            "primary report does not preserve confirm isolation")
    rows = read_jsonl(columns_path)
    require(len(rows) == EXPECTED_COLUMNS,
            "primary report COLUMNS.jsonl is not full current135")
    return report, unique(rows, "primary report columns")


def validate_run(root: Path, report: dict[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    index_path = root / "INDEX.json"
    complete_path = root / "COMPLETE.json"
    complete = read_json(complete_path)
    require(complete.get("schema") == "operator-prepared135-columns-complete-v1" and
            complete.get("status") == "complete" and
            complete.get("completed") == EXPECTED_COLUMNS and
            complete.get("expected") == EXPECTED_COLUMNS and
            complete.get("failed") == 0 and
            complete.get("limited") is False,
            "prepared run is partial, limited, or failed")
    require(complete.get("index_sha256") == sha256(index_path),
            "prepared run INDEX SHA mismatch")
    report_index = report.get("artifacts", {}).get("index", {})
    require(report_index.get("sha256") == sha256(index_path),
            "primary report does not bind this prepared run INDEX")
    index_rows = read_json(index_path)
    require(isinstance(index_rows, list) and len(index_rows) == EXPECTED_COLUMNS,
            "prepared INDEX must contain exactly 3,519 rows")
    index = unique(index_rows, "prepared INDEX")
    require(all(row.get("status") == "strict_verified" for row in index_rows),
            "prepared INDEX contains a non-verified row")
    return complete, index


def selected_totals(result: dict[str, Any], primary: dict[str, Any],
                    policy: str, objective: str, label: str) -> dict[str, Any]:
    whole = result.get("whole_column", {}).get(policy, {})
    goal = whole.get("goals", {}).get(objective)
    require(isinstance(goal, dict), f"{label} lacks {policy}/{objective} whole-column goal")
    selected = {
        "complete_frame_bytes": positive(
            goal.get("complete_frame_bytes"), f"{label}.{policy}.{objective}.complete_frame_bytes"
        ),
        "decision_encode_container": {
            "cpu_ns": positive(goal.get("encode_component_sum_cpu_ns"),
                               f"{label}.{policy}.{objective}.encode_cpu"),
            "wall_ns": positive(goal.get("encode_component_sum_wall_ns"),
                                f"{label}.{policy}.{objective}.encode_wall"),
        },
        "decode": {
            "cpu_ns": positive(goal.get("decode_cpu_ns"),
                               f"{label}.{policy}.{objective}.decode_cpu"),
            "wall_ns": positive(goal.get("decode_wall_ns"),
                                f"{label}.{policy}.{objective}.decode_wall"),
        },
    }
    primary_goal = primary.get("native", {}).get(policy, {}).get(
        "objectives", {}
    ).get(objective)
    require(isinstance(primary_goal, dict),
            f"primary report lacks {label} {policy}/{objective}")
    require(primary_goal.get("complete_frame_bytes") == selected["complete_frame_bytes"] and
            primary_goal.get("decision_plus_encode_component_sum") ==
            selected["decision_encode_container"] and
            primary_goal.get("decode") == selected["decode"],
            f"primary report/root whole-column mismatch for {label} {policy}/{objective}")
    return selected


def fixed_totals(result: dict[str, Any], baseline: str,
                 label: str) -> dict[str, Any]:
    frames = 0
    encoded = empty_time()
    decoded = empty_time()
    blocks = result.get("blocks")
    require(isinstance(blocks, list) and blocks,
            f"{label} has no measured blocks")
    for position, block in enumerate(blocks):
        candidate = block.get("candidates", {}).get(baseline)
        require(isinstance(candidate, dict),
                f"{label} block {position} lacks fixed {baseline}")
        frames += positive(candidate.get("frame_bytes"),
                           f"{label}.blocks[{position}].{baseline}.frame_bytes")
        add_time(encoded, timing(
            candidate.get("fixed_encode"),
            f"{label}.blocks[{position}].{baseline}.fixed_encode",
            require_positive=True,
        ))
        add_time(decoded, timing(
            candidate.get("decode"),
            f"{label}.blocks[{position}].{baseline}.decode",
            require_positive=True,
        ))
    return {
        "complete_frame_bytes": frames,
        "fixed_encode": encoded,
        "decode": decoded,
        "block_count": len(blocks),
    }


def observation(selected: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    value = {
        "selected_complete_frame_bytes": selected["complete_frame_bytes"],
        "fixed_complete_frame_bytes": fixed["complete_frame_bytes"],
        "selected_decision_encode_container": selected["decision_encode_container"],
        "fixed_encode": fixed["fixed_encode"],
        "selected_decode": selected["decode"],
        "fixed_decode": fixed["decode"],
    }
    ratios = metric_ratios(value)
    value["ratios"] = ratios
    for clock in ("cpu", "wall"):
        value[f"all_three_no_worse_{clock}"] = (
            ratios["complete_frame_size"] <= 1.0 and
            ratios[f"encode_{clock}"] <= 1.0 and
            ratios[f"decode_{clock}"] <= 1.0
        )
    return value


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Prepared selector versus fixed-codec baselines",
        "",
        f"Status: **{report['status']}**.",
        "",
        "Both `lz4frame` and `zstd3` are reported for every policy and objective. Confirm results do not choose a baseline, threshold, or model.",
        "",
        "The encode numerator contains selected decision/guard, selected encode, and goal-specific column-container construction. "
        "The fixed denominator is the sum of the existing per-block `fixed_encode` medians. It contains no selector, model decision, "
        "reuse guard, or fixed-codec column-container construction, so the encode comparison is deliberately asymmetric. "
        "`fixed_encode` is an in-process warmed measurement; it is not a fresh-process timing.",
        "",
        "Thread CPU is the primary timing and joint-nonworse boundary. Wall time is secondary because shared-host contention can affect it.",
        "",
        "| scope | policy | goal | fixed codec | frame ratio-of-sums | encode CPU ratio-of-sums | decode CPU ratio-of-sums | encode wall ratio-of-sums | decode wall ratio-of-sums | frame column-geo | encode CPU column-geo | decode CPU column-geo | frame dataset-geo | encode CPU dataset-geo | decode CPU dataset-geo | joint nonworse CPU (primary) | joint nonworse wall (secondary) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope_name in ("all", "development", "confirm"):
        scope = report["scopes"][scope_name]
        for policy in POLICIES:
            for objective in OBJECTIVES:
                for baseline in BASELINES:
                    metric = scope["metrics"][policy][objective][baseline]
                    ros = metric["ratio_of_sums"]
                    col = metric["column_equal_geometric_mean"]
                    dataset = metric["dataset_equal_geometric_mean"]
                    no_worse_cpu = metric["all_three_no_worse_cpu"]
                    no_worse_wall = metric["all_three_no_worse_wall"]
                    lines.append(
                        f"| {scope_name} | {policy} | {objective} | {baseline} | "
                        f"{ros['complete_frame_size']:.6f} | {ros['encode_cpu']:.6f} | {ros['decode_cpu']:.6f} | "
                        f"{ros['encode_wall']:.6f} | {ros['decode_wall']:.6f} | "
                        f"{col['complete_frame_size']:.6f} | {col['encode_cpu']:.6f} | {col['decode_cpu']:.6f} | "
                        f"{dataset['complete_frame_size']:.6f} | {dataset['encode_cpu']:.6f} | {dataset['decode_cpu']:.6f} | "
                        f"{no_worse_cpu['count']}/{metric['columns']} | "
                        f"{no_worse_wall['count']}/{metric['columns']} |"
                    )
    lines.extend([
        "",
        "All quantities are descriptive. Ratio-of-sums is byte/time weighted, column geometry gives every column equal log weight, "
        "and dataset geometry gives every dataset equal log weight after computing a within-dataset ratio-of-sums. "
        "Both simultaneous no-worse counts use the fixed boundary of 1.0. The primary count combines frame bytes with encode/decode Thread CPU; the secondary count uses wall time. "
        "No significance, superiority, or tuned-threshold claim is made.",
        "",
    ])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    require(root.is_dir(), f"--root is not a directory: {root}")
    report_path, primary_columns_path, primary_complete_path = locate_report(args.report)
    primary_report, primary_columns = validate_primary_report(
        report_path, primary_columns_path, primary_complete_path
    )
    run_complete, index = validate_run(root, primary_report)
    require(set(index) == set(primary_columns),
            "prepared INDEX and primary report column sets differ")

    scopes = {
        "all": empty_scope(),
        "development": empty_scope(),
        "confirm": empty_scope(),
    }
    detailed: list[dict[str, Any]] = []
    for item_key in sorted(index):
        dataset_id, column_id = item_key
        index_row = index[item_key]
        primary = primary_columns[item_key]
        split = primary.get("split")
        require(split in {"development", "confirm"},
                f"primary report {item_key} has invalid split")
        require(index_row.get("split") == split,
                f"INDEX/primary split mismatch for {item_key}")
        result_path = root / "columns" / dataset_id / column_id / "RESULT.json"
        require(result_path.is_file() and
                sha256(result_path) == index_row.get("sha256") == primary.get("result_sha256"),
                f"RESULT receipt SHA mismatch for {item_key}")
        result = read_json(result_path)
        require(result.get("status") == "strict_verified" and
                result.get("split") == split,
                f"RESULT {item_key} is not strict_verified")
        fixed_by_baseline = {
            baseline: fixed_totals(result, baseline, f"result {item_key}")
            for baseline in BASELINES
        }
        row = {
            "dataset_id": dataset_id,
            "column_id": column_id,
            "column": primary.get("column"),
            "dtype": primary.get("dtype"),
            "split": split,
            "result_sha256": sha256(result_path),
            "policies": {},
        }
        for policy in POLICIES:
            row["policies"][policy] = {"objectives": {}}
            for objective in OBJECTIVES:
                selected = selected_totals(
                    result, primary, policy, objective, f"result {item_key}"
                )
                goal = {"fixed_baselines": {}}
                for baseline in BASELINES:
                    observed = observation(selected, fixed_by_baseline[baseline])
                    goal["fixed_baselines"][baseline] = observed
                    for scope_name in ("all", split):
                        add_observation(
                            scopes[scope_name]["metrics"][policy][objective][baseline],
                            dataset_id,
                            observed,
                        )
                row["policies"][policy]["objectives"][objective] = goal
        detailed.append(row)
        for scope_name in ("all", split):
            scopes[scope_name]["columns"] += 1
            scopes[scope_name]["datasets"].add(dataset_id)

    require(scopes["all"]["columns"] == EXPECTED_COLUMNS and
            len(scopes["all"]["datasets"]) == EXPECTED_DATASETS and
            scopes["development"]["columns"] == EXPECTED_DEVELOPMENT_COLUMNS and
            scopes["confirm"]["columns"] == EXPECTED_CONFIRM_COLUMNS,
            "fixed-baseline analysis coverage is incomplete")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    details_path = output / "COLUMNS.jsonl"
    atomic_text(details_path, "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in detailed
    ))
    value = {
        "schema": SCHEMA,
        "status": "complete_descriptive_3519",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "columns": EXPECTED_COLUMNS,
            "datasets": EXPECTED_DATASETS,
            "development_columns": EXPECTED_DEVELOPMENT_COLUMNS,
            "confirm_columns": EXPECTED_CONFIRM_COLUMNS,
            "failed_columns": 0,
            "partial_results_accepted": False,
        },
        "fixed_baselines": list(BASELINES),
        "baseline_selection": "none; both fixed codecs are always reported",
        "confirm_used_for_baseline_threshold_or_model_selection": False,
        "comparison_contract": {
            "frame_size": "selected sum complete frame bytes / fixed codec sum complete frame bytes",
            "encode": "selected whole-column decision/guard + selected encode + selected column-container construction / sum per-block fixed_encode",
            "decode": "selected sum decode / fixed codec sum decode",
            "fixed_encode_source_field": "candidates.<codec>.fixed_encode",
            "fixed_encode_is_fresh_process": False,
            "fixed_has_selection_or_model_cost": False,
            "fixed_column_container_construction_included": False,
            "encode_comparison_is_asymmetric": True,
            "aggregation": {
                "ratio_of_sums": "sum numerator / sum denominator",
                "column_equal_geometric_mean": "geometric mean of per-column ratios",
                "dataset_equal_geometric_mean": "geometric mean across per-dataset ratio-of-sums"
            },
            "joint_nonworse_primary_clock": "Thread CPU",
            "joint_nonworse_secondary_clock": "wall time, sensitive to shared-host contention",
            "simultaneous_no_worse_cpu": "frame-size ratio <= 1, encode-CPU ratio <= 1, and decode-CPU ratio <= 1",
            "simultaneous_no_worse_wall": "frame-size ratio <= 1, encode-wall ratio <= 1, and decode-wall ratio <= 1",
            "new_tuned_thresholds": False,
            "statistical_significance_claim": False,
        },
        "scopes": {name: finalise_scope(scope) for name, scope in scopes.items()},
        "artifacts": {
            "prepared_root": str(root),
            "prepared_complete_sha256": sha256(root / "COMPLETE.json"),
            "prepared_index_sha256": sha256(root / "INDEX.json"),
            "primary_report": {"path": str(report_path), "sha256": sha256(report_path)},
            "primary_report_complete": {
                "path": str(primary_complete_path),
                "sha256": sha256(primary_complete_path),
            },
            "primary_report_columns": {
                "path": str(primary_columns_path),
                "sha256": sha256(primary_columns_path),
            },
            "column_analysis": {"path": str(details_path), "sha256": sha256(details_path)},
            "run_bindings": run_complete.get("bindings"),
        },
    }
    analysis_path = output / "BASELINE_ANALYSIS.json"
    atomic_json(analysis_path, value)
    markdown_path = output / "BASELINE_ANALYSIS.md"
    atomic_text(markdown_path, markdown(value))
    atomic_json(output / "COMPLETE.json", {
        "schema": "operator-prepared135-fixed-baseline-analysis-complete-v1",
        "status": "complete_descriptive_3519",
        "analysis_sha256": sha256(analysis_path),
        "markdown_sha256": sha256(markdown_path),
        "columns_jsonl_sha256": sha256(details_path),
        "columns": EXPECTED_COLUMNS,
        "datasets": EXPECTED_DATASETS,
    })
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="completed prepared_columns output root")
    parser.add_argument("--report", type=Path, required=True,
                        help="completed primary REPORT.json or its directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        value = analyze(parse_args(sys.argv[1:] if argv is None else argv))
        print(json.dumps({
            "status": value["status"],
            "columns": value["coverage"]["columns"],
            "datasets": value["coverage"]["datasets"],
            "fixed_baselines": value["fixed_baselines"],
        }, sort_keys=True))
        return 0
    except (AnalysisError, OSError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
