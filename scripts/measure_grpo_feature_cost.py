#!/usr/bin/env python3
"""Measure deployable online costs for the 106D GRPO selector state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def actor_latency_ms(base, input_dim: int, calls: int, repeats: int, seed: int) -> float:
    actor = base.MLP(input_dim, 64, 29, np.random.default_rng(seed))
    x = np.random.default_rng(seed + 1).normal(size=(1, input_dim))
    for _ in range(200):
        for _call in range(calls):
            actor.forward(x)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _call in range(calls):
            actor.forward(x)
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    return float(np.median(samples))


def raw_group_times(base, module, repo_root: Path, columns, repeats: int) -> dict[tuple[str, int], float]:
    by_dataset: dict[str, list] = defaultdict(list)
    for column in columns:
        by_dataset[column.dataset].append(column)
    result: dict[tuple[str, int], float] = {}
    for dataset, selected in sorted(by_dataset.items()):
        data_path, info_path = base.locate_dataset(repo_root, dataset)
        metadata = {int(row["column_index"]): row for row in read_csv(info_path)}
        frame = pd.read_csv(data_path, usecols=[column.name for column in selected], low_memory=False, dtype=str)
        for column in selected:
            row = metadata[column.index]
            values = module.column_values(frame[column.name], row)[: column.sample_values]
            decimal = int(row.get("max_point_number", 0) or 0)
            for _ in range(5):
                base.column_features(values, column.logical_type, decimal)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter_ns()
                base.column_features(values, column.logical_type, decimal)
                samples.append((time.perf_counter_ns() - started) / 1.0e6)
            result[column.key] = float(np.median(samples))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--full-evaluation-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=("Metro-Traffic", "Air-Quality"))
    parser.add_argument("--expected-columns", type=int, required=True)
    parser.add_argument("--full-algorithm", default="feat_full_std_fixed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-repeats", type=int, default=31)
    parser.add_argument("--actor-repeats", type=int, default=20000)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base = load_module(script_dir / "cross_column_rl_compare.py", "feature_cost_base")
    env, data_inputs, operator_source = base.load_environment(
        args.repo_root.resolve(), args.measurement_root.resolve(),
        args.summary_csv.resolve(), args.baseline_csv.resolve(),
        tuple(args.datasets),
    )
    if len(env.columns) != args.expected_columns:
        raise AssertionError(len(env.columns))
    module, _ = base.load_operator_module(args.repo_root.resolve())
    raw_ms = raw_group_times(base, module, args.repo_root.resolve(), env.columns, args.raw_repeats)
    full_rows = [
        row for path in args.full_evaluation_csvs for row in read_csv(path.resolve())
        if row["algorithm"] == args.full_algorithm
    ]
    if len(full_rows) != len(env.columns) * 3:
        raise AssertionError(len(full_rows))
    full_by_column: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in full_rows:
        full_by_column[(row["dataset"], int(row["column_index"]))].append(row)

    actor_ms = {dim: actor_latency_ms(base, dim, 3, args.actor_repeats, 20260826 + dim) for dim in (50, 78, 88, 103, 105, 106)}
    per_column: list[dict[str, object]] = []
    for column in env.columns:
        singles = np.asarray([env.eval_ms[column.key][(action,)] for action in range(28)], dtype=np.float64)
        operator_to_id = {name: i for i, name in enumerate(env.operators)}
        prefix_costs = []
        for row in full_by_column[column.key]:
            pipeline = tuple(operator_to_id[name.strip()] for name in row["pipeline"].split(" -> "))
            # The first-prefix rate is already cached by the 28 single-op probes.
            # A pair replay is causally needed before a third decision or STOP-at-2.
            prefix_costs.append(env.eval_ms[column.key][pipeline[:2]] if len(pipeline) >= 2 else 0.0)
        current_rate_ms = float(np.median(prefix_costs))
        serial_probes = float(np.sum(singles))
        parallel_probes = float(np.max(singles))
        full_serial = raw_ms[column.key] + serial_probes + current_rate_ms + actor_ms[106]
        full_parallel = raw_ms[column.key] + parallel_probes + current_rate_ms + actor_ms[106]
        common = {
            "dataset": column.dataset,
            "column_index": column.index,
            "column_name": column.name,
            "raw18_ms": raw_ms[column.key],
            "serial_probe28_ms": serial_probes,
            "ideal_parallel_probe28_ms": parallel_probes,
            "current_rate_pair_ms": current_rate_ms,
            "actor106_three_calls_ms": actor_ms[106],
            "selector_serial_ms": full_serial,
            "selector_ideal_parallel_ms": full_parallel,
        }
        for action, name in enumerate(env.operators):
            without_serial = full_serial - float(singles[action]) + (actor_ms[105] - actor_ms[106])
            remaining = np.delete(singles, action)
            without_parallel = raw_ms[column.key] + float(np.max(remaining)) + current_rate_ms + actor_ms[105]
            per_column.append({
                **common,
                "unit": f"dim_{18 + action:03d}",
                "dimension": 18 + action,
                "family": "probe",
                "feature_name": f"probe:{name}",
                "skippable": 1,
                "feature_cost_serial_ms": float(singles[action]) + actor_ms[106] - actor_ms[105],
                "cost_increase_serial_pct": 100.0 * (full_serial - without_serial) / without_serial,
                "cost_increase_ideal_parallel_pct": 100.0 * (full_parallel - without_parallel) / without_parallel,
            })
        without_rate = full_serial - current_rate_ms + (actor_ms[105] - actor_ms[106])
        per_column.append({
            **common,
            "unit": "dim_046",
            "dimension": 46,
            "family": "rate",
            "feature_name": "current_prefix_log_rate_ratio",
            "skippable": 1,
            "feature_cost_serial_ms": current_rate_ms + actor_ms[106] - actor_ms[105],
            "cost_increase_serial_pct": 100.0 * (full_serial - without_rate) / without_rate,
            "cost_increase_ideal_parallel_pct": 100.0 * (full_parallel - (full_parallel - current_rate_ms + actor_ms[105] - actor_ms[106])) / (full_parallel - current_rate_ms + actor_ms[105] - actor_ms[106]),
        })
        without_raw = full_serial - raw_ms[column.key] + (actor_ms[88] - actor_ms[106])
        per_column.append({
            **common,
            "unit": "group_raw18",
            "dimension": "0:17",
            "family": "raw_group",
            "feature_name": "raw18_shared_extractor",
            "skippable": 1,
            "feature_cost_serial_ms": raw_ms[column.key] + actor_ms[106] - actor_ms[88],
            "cost_increase_serial_pct": 100.0 * (full_serial - without_raw) / without_raw,
            "cost_increase_ideal_parallel_pct": 100.0 * (full_parallel - (full_parallel - raw_ms[column.key] + actor_ms[88] - actor_ms[106])) / (full_parallel - raw_ms[column.key] + actor_ms[88] - actor_ms[106]),
        })
        without_probes = full_serial - serial_probes + (actor_ms[78] - actor_ms[106])
        per_column.append({
            **common,
            "unit": "group_probe28",
            "dimension": "18:45",
            "family": "probe_group",
            "feature_name": "all_28_single_operator_probes",
            "skippable": 1,
            "feature_cost_serial_ms": serial_probes + actor_ms[106] - actor_ms[78],
            "cost_increase_serial_pct": 100.0 * (full_serial - without_probes) / without_probes,
            "cost_increase_ideal_parallel_pct": 100.0 * (full_parallel - (raw_ms[column.key] + current_rate_ms + actor_ms[78])) / (raw_ms[column.key] + current_rate_ms + actor_ms[78]),
        })

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_column:
        grouped[str(row["unit"])].append(row)
    summary = []
    for unit, rows in sorted(grouped.items()):
        serial = np.asarray([float(row["cost_increase_serial_pct"]) for row in rows])
        parallel = np.asarray([float(row["cost_increase_ideal_parallel_pct"]) for row in rows])
        feature_ms = np.asarray([float(row["feature_cost_serial_ms"]) for row in rows])
        summary.append({
            "unit": unit,
            "dimension": rows[0]["dimension"],
            "family": rows[0]["family"],
            "feature_name": rows[0]["feature_name"],
            "columns": len(rows),
            "median_feature_cost_serial_ms": float(np.median(feature_ms)),
            "dataset_macro_cost_increase_serial_pct": float(np.mean([
                np.mean([float(row["cost_increase_serial_pct"]) for row in rows if row["dataset"] == dataset])
                for dataset in args.datasets
            ])),
            "median_cost_increase_serial_pct": float(np.median(serial)),
            "p10_cost_increase_serial_pct": float(np.quantile(serial, 0.10)),
            "p90_cost_increase_serial_pct": float(np.quantile(serial, 0.90)),
            "median_cost_increase_ideal_parallel_pct": float(np.median(parallel)),
            "columns_serial_gate_ge_5pct": int(np.sum(serial >= 5.0)),
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_column_feature_cost.csv", per_column)
    write_csv(output / "feature_cost_summary.csv", summary)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "actor_ms": actor_ms,
        "protocol": {
            "primary": "serial probe execution matching current Python environment",
            "sensitivity": "ideal 28-way parallel probes; removal only saves the critical path",
            "raw": "raw18 measured as one shared extractor; no per-coordinate cost allocation",
            "current_rate": "pair-prefix replay when selected depth >=2; single-prefix rate reuses probe cache",
            "denominator": "online selector only; excludes exhaustive oracle and selected-codec execution",
        },
        "inputs": data_inputs,
        "operator_source": str(operator_source),
        "counts": {"columns": len(env.columns), "per_column_rows": len(per_column), "summary_rows": len(summary)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "counts": manifest["counts"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
