#!/usr/bin/env python3
"""Test whether shallow operator search leaves enough regret to justify RL.

This deliberately cheap experiment reuses the executable lossless transforms in
``generate_air_operator_pair_heatmap.py``.  It enumerates all no-repeat
pipelines up to a small depth, then evaluates greedy and beam search against the
exhaustive oracle.  It does not train an RL policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_OPERATORS = (
    "delta",
    "xor",
    "frame_of_reference",
    "zigzag",
    "trailing_zero_pack",
    "dictionary",
    "run_length",
    "outlier_separation",
)
SUMMARY_FIELDS = (
    "dataset",
    "column_index",
    "column_name",
    "logical_type",
    "sample_values",
    "sample_sha256",
    "search_space",
    "lossless_pipelines",
    "exhaustive_ms",
    "oracle_pipeline",
    "oracle_rate",
    "best_single_pipeline",
    "best_single_rate",
    "oracle_gain_vs_single_pct",
    "best_pair_pipeline",
    "best_pair_rate",
    "oracle_gain_vs_pair_pct",
    "greedy_pipeline",
    "greedy_rate",
    "greedy_regret_pct",
    "greedy_evaluations",
    "beam2_pipeline",
    "beam2_rate",
    "beam2_regret_pct",
    "beam2_evaluations",
    "beam4_pipeline",
    "beam4_rate",
    "beam4_regret_pct",
    "beam4_evaluations",
    "beam8_pipeline",
    "beam8_rate",
    "beam8_regret_pct",
    "beam8_evaluations",
    "random50_pipeline",
    "random50_rate",
    "random50_regret_pct",
    "strongest_pair_synergy_pct",
    "strongest_pair_synergy_pipeline",
)


@dataclass(frozen=True)
class Measurement:
    pipeline: tuple[str, ...]
    rate: float
    eval_ms: float
    lossless: bool
    status: str


def load_operator_module(repo_root: Path):
    path = repo_root / "scripts" / "generate_air_operator_pair_heatmap.py"
    spec = importlib.util.spec_from_file_location("operator_pair_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load operator implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def pipeline_text(pipeline: Iterable[str]) -> str:
    return " -> ".join(pipeline)


def pipeline_space(
    operators: tuple[str, ...], max_depth: int, allow_repeated_operators: bool
) -> list[tuple[str, ...]]:
    return [
        pipeline
        for depth in range(1, max_depth + 1)
        for pipeline in (
            itertools.product(operators, repeat=depth)
            if allow_repeated_operators
            else itertools.permutations(operators, depth)
        )
    ]


def measure_all(
    module,
    original: np.ndarray,
    operators: tuple[str, ...],
    max_depth: int,
    allow_repeated_operators: bool,
) -> tuple[dict[tuple[str, ...], Measurement], float]:
    original_size = len(module.i64_bytes(original))
    base = module.initial_state(original)
    out: dict[tuple[str, ...], Measurement] = {}
    started = time.perf_counter()

    def visit(prefix: tuple[str, ...], state, remaining: tuple[str, ...]) -> None:
        if len(prefix) >= max_depth:
            return
        for operator in remaining:
            pipeline = prefix + (operator,)
            t0 = time.perf_counter()
            try:
                next_state = module.apply_operator(state, operator)
                rate, lossless = module.checked_rate(next_state, original, original_size)
                status = "ok" if lossless and math.isfinite(rate) else "not_lossless"
            except Exception as exc:  # keep the failed action visible
                next_state = None
                rate = math.nan
                lossless = False
                status = f"failed:{type(exc).__name__}"
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out[pipeline] = Measurement(pipeline, rate, elapsed_ms, lossless, status)
            if next_state is not None and lossless:
                next_operators = (
                    operators
                    if allow_repeated_operators
                    else tuple(item for item in remaining if item != operator)
                )
                visit(pipeline, next_state, next_operators)

    visit((), base, operators)
    return out, (time.perf_counter() - started) * 1000.0


def best_measurement(measurements: Iterable[Measurement]) -> Measurement:
    usable = [m for m in measurements if m.lossless and math.isfinite(m.rate)]
    if not usable:
        raise RuntimeError("No finite lossless measurements")
    return min(usable, key=lambda item: (item.rate, len(item.pipeline), item.pipeline))


def beam_result(
    measurements: dict[tuple[str, ...], Measurement],
    operators: tuple[str, ...],
    max_depth: int,
    width: int,
    allow_repeated_operators: bool,
) -> tuple[Measurement, int]:
    frontier: list[tuple[str, ...]] = [()]
    evaluated: set[tuple[str, ...]] = set()
    candidates: list[Measurement] = []
    for _depth in range(1, max_depth + 1):
        expanded: list[Measurement] = []
        for prefix in frontier:
            for operator in operators:
                if not allow_repeated_operators and operator in prefix:
                    continue
                pipeline = prefix + (operator,)
                measurement = measurements.get(pipeline)
                if measurement is None or not measurement.lossless or not math.isfinite(measurement.rate):
                    continue
                evaluated.add(pipeline)
                expanded.append(measurement)
                candidates.append(measurement)
        expanded.sort(key=lambda item: (item.rate, item.pipeline))
        frontier = [item.pipeline for item in expanded[:width]]
        if not frontier:
            break
    return best_measurement(candidates), len(evaluated)


def regret_pct(rate: float, oracle_rate: float) -> float:
    return 100.0 * (rate - oracle_rate) / oracle_rate


def gain_pct(reference_rate: float, improved_rate: float) -> float:
    return 100.0 * (reference_rate - improved_rate) / reference_rate


def strongest_pair_synergy(measurements: dict[tuple[str, ...], Measurement], base_rate: float) -> tuple[float, tuple[str, ...]]:
    best_value = math.inf
    best_pipeline: tuple[str, ...] = ()
    for pipeline, pair in measurements.items():
        if len(pipeline) != 2 or not pair.lossless or not math.isfinite(pair.rate):
            continue
        first = measurements.get((pipeline[0],))
        second = measurements.get((pipeline[1],))
        if first is None or second is None or min(first.rate, second.rate, base_rate) <= 0:
            continue
        interaction = 100.0 * (pair.rate * base_rate / (first.rate * second.rate) - 1.0)
        if interaction < best_value:
            best_value = interaction
            best_pipeline = pipeline
    return best_value, best_pipeline


def locate_dataset(repo_root: Path, dataset: str) -> tuple[Path, Path]:
    data_candidates = (
        repo_root / "orignial_datasets" / f"{dataset}.csv",
        repo_root / "backend" / "dataset_csv" / f"{dataset}.csv",
    )
    info_candidates = (
        repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv",
        repo_root / "dataset" / dataset / f"{dataset}_column_info.csv",
    )
    data_path = next((path for path in data_candidates if path.is_file()), None)
    info_path = next((path for path in info_candidates if path.is_file()), None)
    if data_path is None or info_path is None:
        raise FileNotFoundError(f"Missing dataset or column metadata for {dataset}")
    return data_path, info_path


def load_columns(
    module,
    repo_root: Path,
    dataset: str,
    max_points: int,
    max_columns: int,
    column_indices: tuple[int, ...] | None,
):
    data_path, info_path = locate_dataset(repo_root, dataset)
    columns = module.numeric_columns(info_path)
    if column_indices is not None:
        selected = set(column_indices)
        columns = [row for row in columns if int(row["column_index"]) in selected]
    elif max_columns > 0:
        columns = columns[:max_columns]
    frame = pd.read_csv(
        data_path,
        usecols=[row["column_name"] for row in columns],
        nrows=max_points,
        low_memory=False,
        dtype=str,
    )
    for column in columns:
        values = module.column_values(frame[column["column_name"]], column)[:max_points]
        if values.size:
            yield data_path, info_path, column, values


def summarize_column(
    module,
    dataset: str,
    column: dict[str, str],
    original: np.ndarray,
    operators: tuple[str, ...],
    max_depth: int,
    seed: int,
    allow_repeated_operators: bool,
):
    measurements, exhaustive_ms = measure_all(
        module, original, operators, max_depth, allow_repeated_operators
    )
    usable = [m for m in measurements.values() if m.lossless and math.isfinite(m.rate)]
    oracle = best_measurement(usable)
    best_single = best_measurement(m for m in usable if len(m.pipeline) == 1)
    best_pair = best_measurement(m for m in usable if len(m.pipeline) == 2)
    greedy, greedy_evals = beam_result(
        measurements, operators, max_depth, 1, allow_repeated_operators
    )
    beam2, beam2_evals = beam_result(
        measurements, operators, max_depth, 2, allow_repeated_operators
    )
    beam4, beam4_evals = beam_result(
        measurements, operators, max_depth, 4, allow_repeated_operators
    )
    beam8, beam8_evals = beam_result(
        measurements, operators, max_depth, 8, allow_repeated_operators
    )
    base_rate, base_ok = module.checked_rate(module.initial_state(original), original, len(module.i64_bytes(original)))
    if not base_ok:
        raise RuntimeError("Initial state failed lossless check")
    synergy, synergy_pipeline = strongest_pair_synergy(measurements, base_rate)
    rng = random.Random(f"{seed}:{dataset}:{column['column_index']}:{column['column_name']}")
    sample = rng.sample(usable, min(50, len(usable)))
    random50 = best_measurement(sample)
    row: dict[str, Any] = {
        "dataset": dataset,
        "column_index": int(column["column_index"]),
        "column_name": column["column_name"],
        "logical_type": str(column.get("logical_type", "")).strip().lower(),
        "sample_values": int(original.size),
        "sample_sha256": hashlib.sha256(module.i64_bytes(original)).hexdigest(),
        "search_space": len(measurements),
        "lossless_pipelines": len(usable),
        "exhaustive_ms": exhaustive_ms,
        "oracle_pipeline": pipeline_text(oracle.pipeline),
        "oracle_rate": oracle.rate,
        "best_single_pipeline": pipeline_text(best_single.pipeline),
        "best_single_rate": best_single.rate,
        "oracle_gain_vs_single_pct": gain_pct(best_single.rate, oracle.rate),
        "best_pair_pipeline": pipeline_text(best_pair.pipeline),
        "best_pair_rate": best_pair.rate,
        "oracle_gain_vs_pair_pct": gain_pct(best_pair.rate, oracle.rate),
        "random50_pipeline": pipeline_text(random50.pipeline),
        "random50_rate": random50.rate,
        "random50_regret_pct": regret_pct(random50.rate, oracle.rate),
        "strongest_pair_synergy_pct": synergy,
        "strongest_pair_synergy_pipeline": pipeline_text(synergy_pipeline),
    }
    for label, result, count in (
        ("greedy", greedy, greedy_evals),
        ("beam2", beam2, beam2_evals),
        ("beam4", beam4, beam4_evals),
        ("beam8", beam8, beam8_evals),
    ):
        row[f"{label}_pipeline"] = pipeline_text(result.pipeline)
        row[f"{label}_rate"] = result.rate
        row[f"{label}_regret_pct"] = regret_pct(result.rate, oracle.rate)
        row[f"{label}_evaluations"] = count
    return row, measurements


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_decision(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = np.asarray([row["exhaustive_ms"] for row in summary_rows], dtype=float)
    beam4 = np.asarray([row["beam4_regret_pct"] for row in summary_rows], dtype=float)
    gains = np.asarray([row["oracle_gain_vs_single_pct"] for row in summary_rows], dtype=float)
    greedy = np.asarray([row["greedy_regret_pct"] for row in summary_rows], dtype=float)
    decision = "continue_to_larger_search"
    reason = "Shallow search leaves a repeatable gap; test a larger typed grammar before RL training."
    if float(np.max(beam4)) < 1.0:
        decision = "stop_rl_at_depth3"
        reason = "Beam-4 stays below 1% oracle regret on every sampled column."
    return {
        "decision": decision,
        "reason": reason,
        "columns": len(summary_rows),
        "median_exhaustive_ms": float(np.median(exact)),
        "max_exhaustive_ms": float(np.max(exact)),
        "median_beam4_regret_pct": float(np.median(beam4)),
        "max_beam4_regret_pct": float(np.max(beam4)),
        "median_greedy_regret_pct": float(np.median(greedy)),
        "max_greedy_regret_pct": float(np.max(greedy)),
        "median_oracle_gain_vs_single_pct": float(np.median(gains)),
        "columns_oracle_gain_over_3pct": int(np.sum(gains > 3.0)),
    }


def self_check(
    module,
    operators: tuple[str, ...],
    max_depth: int,
    allow_repeated_operators: bool,
) -> None:
    synthetic = np.asarray([100, 101, 102, 102, 102, 200, 201, 202] * 64, dtype=np.int64)
    measurements, _ = measure_all(
        module, synthetic, operators, max_depth, allow_repeated_operators
    )
    expected = sum(
        len(operators) ** depth if allow_repeated_operators else math.perm(len(operators), depth)
        for depth in range(1, max_depth + 1)
    )
    assert len(measurements) == expected, (len(measurements), expected)
    assert all(item.lossless for item in measurements.values())
    oracle = best_measurement(measurements.values())
    beam4, evaluations = beam_result(
        measurements, operators, max_depth, 4, allow_repeated_operators
    )
    assert beam4.rate >= oracle.rate
    assert 0 < evaluations <= expected
    if max_depth > 1:
        assert evaluations < expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["Air-Quality", "Wins", "Telco"])
    parser.add_argument("--operators", nargs="+", default=list(DEFAULT_OPERATORS))
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--max-columns", type=int, default=4)
    parser.add_argument("--column-indices", nargs="+", type=int)
    parser.add_argument("--allow-repeated-operators", action="store_true")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    operators = tuple(args.operators)
    if len(set(operators)) != len(operators):
        raise ValueError("Operators must be unique")
    if not 1 <= args.max_depth <= len(operators):
        raise ValueError("max-depth must be between 1 and the operator count")
    module, operator_source = load_operator_module(repo_root)
    unknown = sorted(set(operators) - {item.operator_id for item in module.OPERATORS})
    if unknown:
        raise ValueError(f"Unknown operators: {unknown}")
    self_check(module, operators, args.max_depth, args.allow_repeated_operators)
    expected_pipelines = sum(
        len(operators) ** depth
        if args.allow_repeated_operators
        else math.perm(len(operators), depth)
        for depth in range(1, args.max_depth + 1)
    )
    print(f"SELF_CHECK ok pipelines={expected_pipelines}")
    if args.self_check_only:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for data_path, info_path, column, original in load_columns(
            module,
            repo_root,
            dataset,
            args.max_points,
            args.max_columns,
            tuple(args.column_indices) if args.column_indices is not None else None,
        ):
            print(f"RUN dataset={dataset} column={column['column_index']} name={column['column_name']} n={original.size}", flush=True)
            summary, measurements = summarize_column(
                module,
                dataset,
                column,
                original,
                operators,
                args.max_depth,
                args.seed,
                args.allow_repeated_operators,
            )
            summary_rows.append(summary)
            inputs.append(
                {
                    "dataset": dataset,
                    "data_path": str(data_path),
                    "data_size": data_path.stat().st_size,
                    "data_mtime_ns": data_path.stat().st_mtime_ns,
                    "column_info_path": str(info_path),
                }
            )
            for measurement in measurements.values():
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "column_index": int(column["column_index"]),
                        "column_name": column["column_name"],
                        "pipeline": pipeline_text(measurement.pipeline),
                        "depth": len(measurement.pipeline),
                        "rate": measurement.rate,
                        "eval_ms": measurement.eval_ms,
                        "lossless": int(measurement.lossless),
                        "status": measurement.status,
                    }
                )
            # Persist completed columns so an interrupted remote lane can resume
            # without losing evidence from earlier columns.
            write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
            write_csv(
                output_dir / "pipeline_measurements.csv",
                detail_rows,
                ("dataset", "column_index", "column_name", "pipeline", "depth", "rate", "eval_ms", "lossless", "status"),
            )

    decision = aggregate_decision(summary_rows)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(
        output_dir / "pipeline_measurements.csv",
        detail_rows,
        ("dataset", "column_index", "column_name", "pipeline", "depth", "rate", "eval_ms", "lossless", "status"),
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "repo_root": str(repo_root),
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operator_source": str(operator_source),
        "operator_source_sha256": hashlib.sha256(operator_source.read_bytes()).hexdigest(),
        "config": {
            "datasets": args.datasets,
            "operators": operators,
            "max_depth": args.max_depth,
            "max_points": args.max_points,
            "max_columns": args.max_columns,
            "column_indices": args.column_indices,
            "allow_repeated_operators": args.allow_repeated_operators,
            "seed": args.seed,
            "grammar": (
                "all selected state-to-state operators, repeated operators allowed"
                if args.allow_repeated_operators
                else "all selected state-to-state operators, no repeated operator"
            ),
        },
        "inputs": inputs,
        "decision": decision,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DECISION", json.dumps(decision, ensure_ascii=False, sort_keys=True))
    print(f"OUTPUT {output_dir}")


if __name__ == "__main__":
    main()
