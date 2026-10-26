#!/usr/bin/env python3
"""Paired latency diagnosis for Full60 versus exactly folded Sparse19 PV2F actors."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import ablate_physical_policy_features as ablation
import search_physical_operator_variants_v2 as base
from benchmark_physical_operator_selected_timings_v2 import (
    load_actor,
    load_sparse19_actor,
    policy_select,
    representative,
    sparse19_policy_select,
)
from enumerate_physical_operator_variants_v2 import load_column


RAW16 = tuple(index for index in range(17) if index != 8)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def archived_features(path: Path, tables: list[base.ColumnTable]) -> np.ndarray:
    rows = read_csv(path)
    if len(rows) != len(tables):
        raise AssertionError((path, len(rows), len(tables)))
    for table, row in zip(tables, rows):
        if (
            row["dataset"] != table.dataset
            or int(row["column_index"]) != table.column_index
            or row["column_name"] != table.column_name
        ):
            raise AssertionError((table.key, row))
    return np.asarray(
        [[float(row[f"feature_{index}"]) for index in range(17)] for row in rows],
        dtype=np.float64,
    )


def timed_ns(function: Callable[[], Any]) -> tuple[int, Any]:
    started = time.perf_counter_ns()
    value = function()
    return time.perf_counter_ns() - started, value


def summarize_pair(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, list[int]]] = defaultdict(
        lambda: {"full60": [], "sparse19": []}
    )
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)][row["observation"]].append(
            int(row["elapsed_ns"])
        )
    full_medians: list[float] = []
    sparse_medians: list[float] = []
    speedups: list[float] = []
    for values in grouped.values():
        full = statistics.median(values["full60"])
        sparse = statistics.median(values["sparse19"])
        full_medians.append(full)
        sparse_medians.append(sparse)
        speedups.append(100.0 * (full - sparse) / full)
    return {
        "groups": len(grouped),
        "full60_group_median_ns": statistics.median(full_medians),
        "sparse19_group_median_ns": statistics.median(sparse_medians),
        "paired_speedup_pct_median": statistics.median(speedups),
        "paired_speedup_pct_min": min(speedups),
        "paired_speedup_pct_max": max(speedups),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--sparse19-policy-dir", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-repeats", type=int, default=15)
    parser.add_argument("--feature-repeats", type=int, default=31)
    parser.add_argument("--kernel-repeats", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo_root = args.repo_root.resolve()
    measurement_root = args.measurement_root.resolve()
    search_dir = args.search_dir.resolve()
    sparse_policy_dir = args.sparse19_policy_dir.resolve()
    tables = base.load_tables(measurement_root)
    raw = archived_features(args.feature_csv.resolve(), tables)
    table_index = {table.key: index for index, table in enumerate(tables)}

    full_builders: dict[int, base.FeatureBuilder] = {}
    sparse_builders: dict[int, base.Sparse19FeatureBuilder] = {}
    datasets = sorted({table.dataset for table in tables})
    for fold, test_dataset in enumerate(datasets):
        train = np.asarray(
            [index for index, table in enumerate(tables) if table.dataset != test_dataset],
            dtype=np.int32,
        )
        full_builders[fold] = base.FeatureBuilder(raw, train)
        sparse_builders[fold] = base.Sparse19FeatureBuilder(raw, train)

    rl_rows = read_csv(search_dir / "rl_evaluation.csv")
    proposal_budget = int(rl_rows[0]["proposal_rollouts"])
    grouped_rl: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rl_rows:
        grouped_rl[(row["test_dataset"], int(row["column_index"]), row["algorithm"])].append(row)

    actor_full: dict[tuple[str, int, int], base.MLP] = {}
    actor_sparse: dict[tuple[str, int, int], base.Sparse19MLP] = {}
    selector_specs: list[dict[str, Any]] = []
    for table in tables:
        for algorithm in ("ppo", "grpo"):
            selected = representative(grouped_rl[(table.dataset, table.column_index, algorithm)])
            key = (algorithm, int(selected["seed"]), int(selected["fold"]))
            fold = key[2]
            if key not in actor_full:
                actor_full[key] = load_actor(
                    search_dir / "policies" / f"{algorithm}_seed{key[1]}_fold{fold}.npz",
                    full_builders[fold].input_dim,
                )
                actor_sparse[key] = load_sparse19_actor(
                    sparse_policy_dir / f"{algorithm}_seed{key[1]}_fold{fold}.npz"
                )
            selector_specs.append({
                "algorithm": algorithm,
                "seed": key[1],
                "fold": fold,
                "table": table,
                "column": table_index[table.key],
                "expected_pipeline": base.parse_pipeline(selected["pipeline"]),
                "expected_proposals": int(selected["unique_proposals"]),
                "actor_key": key,
            })

    feature_rows: list[dict[str, Any]] = []
    values_by_column: dict[int, np.ndarray] = {}
    order_rng = np.random.default_rng(20260828)
    for column, table in enumerate(tables):
        values, _metadata, _data, _info = load_column(
            repo_root, table.dataset, table.column_index, 0
        )
        values_by_column[column] = values
        full_check = base.column_features(values)
        sparse_check = ablation.extract_selected(values, RAW16)
        if not np.allclose(full_check, raw[column], rtol=0.0, atol=1.0e-12):
            raise AssertionError((table.key, "full17 differs from archived features"))
        if not np.allclose(sparse_check, raw[column, RAW16], rtol=0.0, atol=1.0e-12):
            raise AssertionError((table.key, "sparse16 differs from archived features"))
        for repeat in range(args.feature_repeats):
            order = ["full60", "sparse19"]
            order_rng.shuffle(order)
            for observation in order:
                if observation == "full60":
                    function = lambda values=values: base.column_features(values)
                else:
                    function = lambda values=values: ablation.extract_selected(values, RAW16)
                elapsed, _result = timed_ns(function)
                feature_rows.append({
                    "dataset": table.dataset,
                    "column_index": table.column_index,
                    "column_name": table.column_name,
                    "repeat": repeat,
                    "observation": observation,
                    "elapsed_ns": elapsed,
                })

    kernel_rows: list[dict[str, Any]] = []
    kernel_key = ("ppo", 20260825, 0)
    if kernel_key not in actor_full:
        actor_full[kernel_key] = load_actor(
            search_dir / "policies/ppo_seed20260825_fold0.npz",
            full_builders[0].input_dim,
        )
        actor_sparse[kernel_key] = load_sparse19_actor(
            sparse_policy_dir / "ppo_seed20260825_fold0.npz"
        )
    prefixes = (
        (),
        (base.TRANSFORM_NAMES[0],),
        (base.TRANSFORM_NAMES[0], base.TRANSFORM_NAMES[1]),
        (base.PACKER_NAMES[0],),
        (base.TRANSFORM_NAMES[0], base.PACKER_NAMES[0]),
    )
    full_actor = actor_full[kernel_key]
    sparse_actor = actor_sparse[kernel_key]
    kernel_rng = np.random.default_rng(20260829)
    for prefix_index, prefix in enumerate(prefixes):
        full_feature = full_builders[0].one(0, prefix)
        sparse_feature = sparse_builders[0].one(0, prefix)
        full_logits = full_actor.forward(full_feature[None, :])[0][0]
        sparse_logits = sparse_actor.forward(sparse_feature)[0]
        if not np.allclose(full_logits, sparse_logits, rtol=0.0, atol=1.0e-12):
            raise AssertionError((prefix, "logit mismatch"))
        for operation in ("state_assembly", "model_forward"):
            for repeat in range(5):
                order = ["full60", "sparse19"]
                kernel_rng.shuffle(order)
                for observation in order:
                    if operation == "state_assembly":
                        if observation == "full60":
                            function = lambda: full_builders[0].one(0, prefix)
                        else:
                            function = lambda: sparse_builders[0].one(0, prefix)
                    else:
                        if observation == "full60":
                            function = lambda: full_actor.forward(full_feature[None, :])
                        else:
                            function = lambda: sparse_actor.forward(sparse_feature)
                    started = time.perf_counter_ns()
                    for _ in range(args.kernel_repeats):
                        function()
                    elapsed = time.perf_counter_ns() - started
                    kernel_rows.append({
                        "operation": operation,
                        "prefix_index": prefix_index,
                        "prefix": " -> ".join(prefix),
                        "repeat": repeat,
                        "observation": observation,
                        "calls": args.kernel_repeats,
                        "elapsed_ns": elapsed,
                        "elapsed_ns_per_call": elapsed / args.kernel_repeats,
                    })

    selector_rows: list[dict[str, Any]] = []
    selector_rng = np.random.default_rng(20260830)
    for spec_index, spec in enumerate(selector_specs):
        table = spec["table"]
        column = spec["column"]
        key = spec["actor_key"]
        full_result = policy_select(
            actor_full[key], full_builders[spec["fold"]], column, table,
            spec["seed"], spec["fold"], proposal_budget,
        )
        sparse_result = sparse19_policy_select(
            actor_sparse[key], sparse_builders[spec["fold"]], column, table,
            spec["seed"], spec["fold"], proposal_budget,
        )
        if full_result != sparse_result:
            raise AssertionError((table.key, spec["algorithm"], full_result, sparse_result))
        if full_result != (spec["expected_pipeline"], spec["expected_proposals"]):
            raise AssertionError((table.key, spec["algorithm"], full_result))
        for repeat in range(args.selection_repeats):
            order = ["full60", "sparse19"]
            selector_rng.shuffle(order)
            for observation in order:
                if observation == "full60":
                    function = lambda: policy_select(
                        actor_full[key], full_builders[spec["fold"]], column, table,
                        spec["seed"], spec["fold"], proposal_budget,
                    )
                else:
                    function = lambda: sparse19_policy_select(
                        actor_sparse[key], sparse_builders[spec["fold"]], column, table,
                        spec["seed"], spec["fold"], proposal_budget,
                    )
                elapsed, result = timed_ns(function)
                if result != full_result:
                    raise AssertionError((table.key, spec["algorithm"], observation, result))
                selector_rows.append({
                    "dataset": table.dataset,
                    "column_index": table.column_index,
                    "column_name": table.column_name,
                    "algorithm": spec["algorithm"],
                    "seed": spec["seed"],
                    "fold": spec["fold"],
                    "repeat": repeat,
                    "order_in_pair": order.index(observation),
                    "observation": observation,
                    "elapsed_ns": elapsed,
                    "unique_proposals": result[1],
                })
        print(json.dumps({"finished": spec_index + 1, "total": len(selector_specs)}), flush=True)

    write_csv(output / "feature_extraction_samples.csv", feature_rows)
    write_csv(output / "kernel_samples.csv", kernel_rows)
    write_csv(output / "selector_samples.csv", selector_rows)
    kernel_summary: dict[str, Any] = {}
    for operation in ("state_assembly", "model_forward"):
        selected = [
            {**row, "elapsed_ns": row["elapsed_ns_per_call"]}
            for row in kernel_rows if row["operation"] == operation
        ]
        kernel_summary[operation] = summarize_pair(selected, ("prefix_index",))
    selector_summary = summarize_pair(
        selector_rows, ("dataset", "column_index", "algorithm")
    )
    for algorithm in ("ppo", "grpo"):
        selector_summary[algorithm] = summarize_pair(
            [row for row in selector_rows if row["algorithm"] == algorithm],
            ("dataset", "column_index"),
        )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "comparison": "same process, same policy, same column, shuffled AB/BA order within each repeat",
        "full60_definition": "17 continuous column features plus 43 dense prefix one-hot dimensions",
        "sparse19_definition": "16 continuous column features plus three categorical lookup fields",
        "feature_extraction_in_original_selector_timing": False,
        "proposal_budget": proposal_budget,
        "feature_extraction": summarize_pair(
            feature_rows, ("dataset", "column_index")
        ),
        "kernels": kernel_summary,
        "selector": selector_summary,
        "claim_boundary": (
            "Python/NumPy prototype microbenchmark; selector result includes 512 rollout generation, "
            "RNG, masks, prefix parsing, set deduplication, and frozen byte-table minimum selection"
        ),
    }
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
