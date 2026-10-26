#!/usr/bin/env python3
"""Paired benchmark for cached scalar and unique-prefix batched Sparse19 inference."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import statistics
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import search_physical_operator_variants_v2 as base
from benchmark_physical_operator_selected_timings_v2 import (
    load_actor,
    load_sparse19_actor,
    representative,
)


VARIANTS = (
    "full60_scalar",
    "sparse19_cached_fields_scalar",
    "sparse19_cached_scalar",
    "sparse19_unique_prefix_exact",
    "full60_batch_prefix",
    "sparse19_batch_prefix",
)


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


def proposal_seed(seed: int, fold: int, column: int) -> int:
    return seed + 10_000_019 * (fold + 1) + 1009 * (column + 1)


def finish_selection(
    table: base.ColumnTable,
    proposals: set[tuple[str, ...]],
) -> tuple[tuple[str, ...], int]:
    valid = [pipeline for pipeline in proposals if pipeline in table.sizes]
    if not valid:
        raise AssertionError(f"no valid proposal for {table.key}")
    return min(valid, key=lambda pipeline: (table.sizes[pipeline], pipeline)), len(proposals)


def scalar_selection(
    proposal: Callable[[np.random.Generator | None], tuple[str, ...]],
    table: base.ColumnTable,
    seed: int,
    fold: int,
    column: int,
    budget: int,
) -> tuple[tuple[str, ...], int, set[tuple[str, ...]]]:
    rng = np.random.default_rng(proposal_seed(seed, fold, column))
    proposals = {proposal(None), ("raw.i64", "none")}
    for _ in range(budget):
        proposals.add(proposal(rng))
    selected, count = finish_selection(table, proposals)
    return selected, count, proposals


def batch_selection(
    greedy: Callable[[], tuple[str, ...]],
    batched: Callable[
        [np.ndarray], tuple[set[tuple[str, ...]], dict[str, int]]
    ],
    table: base.ColumnTable,
    seed: int,
    fold: int,
    column: int,
    budget: int,
) -> tuple[tuple[str, ...], int, set[tuple[str, ...]], dict[str, int]]:
    rng = np.random.default_rng(proposal_seed(seed, fold, column))
    uniforms = rng.random((budget, base.MAX_STEPS))
    proposals, diagnostics = batched(uniforms)
    proposals.update((greedy(), ("raw.i64", "none")))
    selected, count = finish_selection(table, proposals)
    return selected, count, proposals, diagnostics


def unique_prefix_exact_selection(
    cache: base.Sparse19InferenceCache,
    table: base.ColumnTable,
    seed: int,
    fold: int,
    column: int,
    budget: int,
) -> tuple[tuple[str, ...], int, set[tuple[str, ...]], dict[str, int]]:
    """Reuse prefix probabilities while preserving the old scalar RNG order."""

    probabilities: dict[int, np.ndarray] = {}
    cdfs: dict[int, list[float]] = {}

    def distribution(prefix_id: int) -> np.ndarray:
        result = probabilities.get(prefix_id)
        if result is None:
            result = base.sparse19_probabilities(
                cache.logits(column, cache.builder.prefixes[prefix_id]),
                cache.builder.prefix_legal_masks[prefix_id],
            )
            probabilities[prefix_id] = result
            cdfs[prefix_id] = np.cumsum(result).tolist()
        return result

    prefix_id = 0
    while True:
        action = int(np.argmax(distribution(prefix_id)))
        next_id = cache.builder.prefix_next_id_rows[prefix_id][action]
        if next_id == -1:
            greedy = cache.builder.prefixes[prefix_id] + (base.ACTION_NAMES[action],)
            break
        if next_id < 0:
            raise AssertionError((prefix_id, action, next_id))
        prefix_id = next_id

    rng = np.random.default_rng(proposal_seed(seed, fold, column))
    uniform_sequence = rng.random(budget * base.MAX_STEPS).tolist()
    uniform_cursor = 0
    proposals = {greedy, ("raw.i64", "none")}
    trajectory_prefix_evaluations = 0
    for _ in range(budget):
        prefix_id = 0
        for _step in range(base.MAX_STEPS):
            distribution(prefix_id)
            action = bisect_left(cdfs[prefix_id], uniform_sequence[uniform_cursor])
            uniform_cursor += 1
            trajectory_prefix_evaluations += 1
            next_id = cache.builder.prefix_next_id_rows[prefix_id][action]
            if next_id == -1:
                proposals.add(
                    cache.builder.prefixes[prefix_id] + (base.ACTION_NAMES[action],)
                )
                break
            if next_id < 0:
                raise AssertionError((prefix_id, action, next_id))
            prefix_id = next_id
        else:
            raise AssertionError("unique-prefix proposal exceeded typed horizon")
    selected, count = finish_selection(table, proposals)
    return selected, count, proposals, {
        "rollouts": budget,
        "trajectory_prefix_evaluations": trajectory_prefix_evaluations,
        "unique_prefix_evaluations": len(probabilities),
        "merged_prefix_evaluations": trajectory_prefix_evaluations - len(probabilities),
    }


def bootstrap_median_ci(values: list[float], seed: int = 20260828) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.median(
        array[rng.integers(0, array.size, size=(10_000, array.size))], axis=1
    )
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def paired_summary(
    rows: list[dict[str, Any]],
    baseline: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(row["dataset"], row["column_index"], row["algorithm"])][
            row["variant"]
        ].append(int(row["elapsed_ns"]))
    medians = {
        key: {variant: statistics.median(samples) for variant, samples in variants.items()}
        for key, variants in grouped.items()
    }
    result: dict[str, Any] = {"groups": len(medians), "baseline": baseline, "variants": {}}
    for variant in VARIANTS:
        group_ns = [variants[variant] for variants in medians.values()]
        speedups = [
            100.0 * (variants[baseline] - variants[variant]) / variants[baseline]
            for variants in medians.values()
        ]
        result["variants"][variant] = {
            "group_median_ms": statistics.median(group_ns) / 1_000_000.0,
            "paired_speedup_pct_median": statistics.median(speedups),
            "paired_speedup_pct_min": min(speedups),
            "paired_speedup_pct_max": max(speedups),
            "paired_speedup_pct_bootstrap_median_ci95": bootstrap_median_ci(speedups),
        }
    return result


def quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["batch_improvement_vs_scalar_pct"]) for row in rows]
    old_bytes = sum(int(row["scalar_encoded_bytes"]) for row in rows)
    batch_bytes = sum(int(row["batch_encoded_bytes"]) for row in rows)
    return {
        "rows": len(rows),
        "batch_better": sum(delta > 0.0 for delta in deltas),
        "batch_equal": sum(delta == 0.0 for delta in deltas),
        "batch_worse": sum(delta < 0.0 for delta in deltas),
        "median_improvement_pct": statistics.median(deltas),
        "total_scalar_bytes": old_bytes,
        "total_batch_bytes": batch_bytes,
        "total_improvement_pct": 100.0 * (old_bytes - batch_bytes) / old_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--sparse19-policy-dir", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-repeats", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selection_repeats <= 0:
        raise ValueError(args.selection_repeats)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    measurement_root = args.measurement_root.resolve()
    search_dir = args.search_dir.resolve()
    sparse_policy_dir = args.sparse19_policy_dir.resolve()
    tables = base.load_tables(measurement_root)
    raw = archived_features(args.feature_csv.resolve(), tables)
    table_position = {table.key: index for index, table in enumerate(tables)}
    datasets = sorted({table.dataset for table in tables})

    full_builders: dict[int, base.FeatureBuilder] = {}
    sparse_builders: dict[int, base.Sparse19FeatureBuilder] = {}
    for fold, test_dataset in enumerate(datasets):
        train = np.asarray(
            [index for index, table in enumerate(tables) if table.dataset != test_dataset],
            dtype=np.int32,
        )
        full_builders[fold] = base.FeatureBuilder(raw, train)
        sparse_builders[fold] = base.Sparse19FeatureBuilder(raw, train)
        if len(sparse_builders[fold].prefixes) != 2030:
            raise AssertionError(len(sparse_builders[fold].prefixes))

    rl_rows = read_csv(search_dir / "rl_evaluation.csv")
    proposal_budget = int(rl_rows[0]["proposal_rollouts"])
    full_actors: dict[tuple[str, int, int], base.MLP] = {}
    sparse_actors: dict[tuple[str, int, int], base.Sparse19MLP] = {}
    caches: dict[tuple[str, int, int], base.Sparse19InferenceCache] = {}
    cache_rows: list[dict[str, Any]] = []
    for row in rl_rows:
        key = (row["algorithm"], int(row["seed"]), int(row["fold"]))
        if key in full_actors:
            continue
        algorithm, seed, fold = key
        full_actors[key] = load_actor(
            search_dir / "policies" / f"{algorithm}_seed{seed}_fold{fold}.npz",
            full_builders[fold].input_dim,
        )
        sparse_actors[key] = load_sparse19_actor(
            sparse_policy_dir / f"{algorithm}_seed{seed}_fold{fold}.npz"
        )
        started = time.perf_counter_ns()
        caches[key] = base.Sparse19InferenceCache(sparse_actors[key], sparse_builders[fold])
        cache_rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "fold": fold,
            "elapsed_ns": time.perf_counter_ns() - started,
            "columns_cached": raw.shape[0],
            "prefixes_cached": len(sparse_builders[fold].prefixes),
        })

    correctness_rows: list[dict[str, Any]] = []
    correctness_results: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}
    for row_index, row in enumerate(rl_rows):
        table = tables[table_position[(row["test_dataset"], int(row["column_index"]))]]
        column = table_position[table.key]
        key = (row["algorithm"], int(row["seed"]), int(row["fold"]))
        algorithm, seed, fold = key
        full_actor = full_actors[key]
        sparse_actor = sparse_actors[key]
        full_builder = full_builders[fold]
        sparse_builder = sparse_builders[fold]
        cache = caches[key]

        full_scalar = scalar_selection(
            lambda rng: base.propose_pipeline(full_actor, full_builder, column, rng),
            table, seed, fold, column, proposal_budget,
        )
        cached_fields_scalar = scalar_selection(
            lambda rng: base.propose_pipeline_sparse19(sparse_actor, sparse_builder, column, rng),
            table, seed, fold, column, proposal_budget,
        )
        cached_scalar = scalar_selection(
            lambda rng: base.propose_pipeline_sparse19_cached(cache, column, rng),
            table, seed, fold, column, proposal_budget,
        )
        unique_exact = unique_prefix_exact_selection(
            cache, table, seed, fold, column, proposal_budget
        )
        full_batch = batch_selection(
            lambda: base.propose_pipeline(full_actor, full_builder, column, None),
            lambda uniforms: base.propose_pipelines_full60_batched(
                full_actor, full_builder, column, uniforms
            ),
            table, seed, fold, column, proposal_budget,
        )
        sparse_batch = batch_selection(
            lambda: base.propose_pipeline_sparse19_cached(cache, column, None),
            lambda uniforms: base.propose_pipelines_sparse19_batched(cache, column, uniforms),
            table, seed, fold, column, proposal_budget,
        )
        expected = (base.parse_pipeline(row["pipeline"]), int(row["unique_proposals"]))
        if full_scalar[:2] != expected:
            raise AssertionError((table.key, key, "archived", full_scalar[:2], expected))
        if (
            full_scalar != cached_fields_scalar
            or full_scalar != cached_scalar
            or full_scalar != unique_exact[:3]
        ):
            raise AssertionError((table.key, key, "scalar mismatch"))
        if full_batch[:3] != sparse_batch[:3] or full_batch[3] != sparse_batch[3]:
            raise AssertionError((table.key, key, "batch mismatch"))

        scalar_bytes = table.sizes[full_scalar[0]]
        batch_bytes = table.sizes[full_batch[0]]
        result_key = (table.dataset, table.column_index, algorithm, seed, fold)
        correctness_results[result_key] = {
            "full_scalar": full_scalar,
            "cached_fields_scalar": cached_fields_scalar,
            "cached_scalar": cached_scalar,
            "unique_exact": unique_exact,
            "full_batch": full_batch,
            "sparse_batch": sparse_batch,
        }
        correctness_rows.append({
            "dataset": table.dataset,
            "column_index": table.column_index,
            "column_name": table.column_name,
            "algorithm": algorithm,
            "seed": seed,
            "fold": fold,
            "scalar_pipeline": " -> ".join(full_scalar[0]),
            "scalar_unique_proposals": full_scalar[1],
            "scalar_encoded_bytes": scalar_bytes,
            "archived_selected_pipeline_and_count_exact": 1,
            "cached_scalar_exact": int(full_scalar == cached_scalar),
            "unique_prefix_exact": int(full_scalar == unique_exact[:3]),
            "exact_trajectory_prefix_evaluations": unique_exact[3][
                "trajectory_prefix_evaluations"
            ],
            "exact_unique_prefix_evaluations": unique_exact[3]["unique_prefix_evaluations"],
            "exact_merged_prefix_evaluations": unique_exact[3]["merged_prefix_evaluations"],
            "batch_pipeline": " -> ".join(full_batch[0]),
            "batch_unique_proposals": full_batch[1],
            "batch_encoded_bytes": batch_bytes,
            "batch_full60_sparse19_exact": int(full_batch[:3] == sparse_batch[:3]),
            "batch_matches_scalar_proposal_set": int(full_batch[2] == full_scalar[2]),
            "batch_improvement_vs_scalar_pct": 100.0 * (scalar_bytes - batch_bytes) / scalar_bytes,
            "batch_trajectory_prefix_evaluations": full_batch[3][
                "trajectory_prefix_evaluations"
            ],
            "batch_unique_prefix_evaluations": full_batch[3]["unique_prefix_evaluations"],
            "batch_merged_prefix_evaluations": full_batch[3]["merged_prefix_evaluations"],
        })
        if (row_index + 1) % 20 == 0:
            print(json.dumps({"correctness_finished": row_index + 1, "total": len(rl_rows)}), flush=True)

    grouped_rl: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rl_rows:
        grouped_rl[(row["test_dataset"], int(row["column_index"]), row["algorithm"])].append(row)
    timing_specs: list[dict[str, Any]] = []
    for table in tables:
        for algorithm in ("ppo", "grpo"):
            row = representative(grouped_rl[(table.dataset, table.column_index, algorithm)])
            timing_specs.append({"table": table, "row": row, "column": table_position[table.key]})

    timing_rows: list[dict[str, Any]] = []
    order_rng = np.random.default_rng(20260828)
    for spec_index, spec in enumerate(timing_specs):
        table = spec["table"]
        row = spec["row"]
        column = spec["column"]
        key = (row["algorithm"], int(row["seed"]), int(row["fold"]))
        algorithm, seed, fold = key
        full_actor = full_actors[key]
        sparse_actor = sparse_actors[key]
        full_builder = full_builders[fold]
        sparse_builder = sparse_builders[fold]
        cache = caches[key]
        expected = correctness_results[(table.dataset, table.column_index, algorithm, seed, fold)]

        functions: dict[str, Callable[[], Any]] = {
            "full60_scalar": lambda: scalar_selection(
                lambda rng: base.propose_pipeline(full_actor, full_builder, column, rng),
                table, seed, fold, column, proposal_budget,
            ),
            "sparse19_cached_fields_scalar": lambda: scalar_selection(
                lambda rng: base.propose_pipeline_sparse19(sparse_actor, sparse_builder, column, rng),
                table, seed, fold, column, proposal_budget,
            ),
            "sparse19_cached_scalar": lambda: scalar_selection(
                lambda rng: base.propose_pipeline_sparse19_cached(cache, column, rng),
                table, seed, fold, column, proposal_budget,
            ),
            "sparse19_unique_prefix_exact": lambda: unique_prefix_exact_selection(
                cache, table, seed, fold, column, proposal_budget
            ),
            "full60_batch_prefix": lambda: batch_selection(
                lambda: base.propose_pipeline(full_actor, full_builder, column, None),
                lambda uniforms: base.propose_pipelines_full60_batched(
                    full_actor, full_builder, column, uniforms
                ),
                table, seed, fold, column, proposal_budget,
            ),
            "sparse19_batch_prefix": lambda: batch_selection(
                lambda: base.propose_pipeline_sparse19_cached(cache, column, None),
                lambda uniforms: base.propose_pipelines_sparse19_batched(cache, column, uniforms),
                table, seed, fold, column, proposal_budget,
            ),
        }
        expected_by_variant = {
            "full60_scalar": expected["full_scalar"],
            "sparse19_cached_fields_scalar": expected["cached_fields_scalar"],
            "sparse19_cached_scalar": expected["cached_scalar"],
            "sparse19_unique_prefix_exact": expected["unique_exact"],
            "full60_batch_prefix": expected["full_batch"],
            "sparse19_batch_prefix": expected["sparse_batch"],
        }
        for repeat in range(args.selection_repeats):
            order = list(VARIANTS)
            order_rng.shuffle(order)
            for order_in_round, variant in enumerate(order):
                started = time.perf_counter_ns()
                result = functions[variant]()
                elapsed = time.perf_counter_ns() - started
                if result != expected_by_variant[variant]:
                    raise AssertionError((table.key, key, variant, "non-deterministic"))
                timing_rows.append({
                    "dataset": table.dataset,
                    "column_index": table.column_index,
                    "column_name": table.column_name,
                    "algorithm": algorithm,
                    "seed": seed,
                    "fold": fold,
                    "repeat": repeat,
                    "order_in_round": order_in_round,
                    "variant": variant,
                    "elapsed_ns": elapsed,
                    "selected_bytes": table.sizes[result[0]],
                    "unique_proposals": result[1],
                })
        print(json.dumps({"timing_finished": spec_index + 1, "total": len(timing_specs)}), flush=True)

    write_csv(output / "cache_initialization.csv", cache_rows)
    write_csv(output / "correctness_and_quality.csv", correctness_rows)
    write_csv(output / "selector_samples.csv", timing_rows)
    overall_quality = quality_summary(correctness_rows)
    quality_by_algorithm = {
        algorithm: quality_summary([row for row in correctness_rows if row["algorithm"] == algorithm])
        for algorithm in ("ppo", "grpo")
    }
    versus_full60 = paired_summary(timing_rows, "full60_scalar")
    versus_sparse_fields = paired_summary(timing_rows, "sparse19_cached_fields_scalar")
    batch_speedup = versus_full60["variants"]["sparse19_batch_prefix"]
    exact_speedup = versus_full60["variants"]["sparse19_unique_prefix_exact"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "protocol": (
            "same process and representative saved policy per column/algorithm; six variants shuffled "
            "within every repeat; fixed seed per selector; 512 by 4 i.i.d. uniform matrix for batched variants"
        ),
        "proposal_budget": proposal_budget,
        "legal_nonterminal_prefixes": 2030,
        "correctness_gates": {
            "archived_selected_pipeline_and_count_rows_exact": sum(
                int(row["archived_selected_pipeline_and_count_exact"])
                for row in correctness_rows
            ),
            "fresh_full60_vs_cached_scalar_proposal_sets_exact": sum(
                int(row["cached_scalar_exact"]) for row in correctness_rows
            ),
            "fresh_full60_vs_unique_prefix_proposal_sets_exact": sum(
                int(row["unique_prefix_exact"]) for row in correctness_rows
            ),
            "fixed_matrix_full60_vs_sparse19_batch_proposal_sets_exact": sum(
                int(row["batch_full60_sparse19_exact"]) for row in correctness_rows
            ),
            "step_batch_vs_scalar_proposal_sets_exact": sum(
                int(row["batch_matches_scalar_proposal_set"]) for row in correctness_rows
            ),
            "total_rows": len(correctness_rows),
        },
        "cache_initialization": {
            "actors": len(cache_rows),
            "median_ms": statistics.median(row["elapsed_ns"] for row in cache_rows) / 1_000_000.0,
            "max_ms": max(row["elapsed_ns"] for row in cache_rows) / 1_000_000.0,
            "timed_selector_excludes_one_time_cache_initialization": True,
        },
        "prefix_merging": {
            "pathwise_exact": {
                "trajectory_prefix_evaluations_median": statistics.median(
                    int(row["exact_trajectory_prefix_evaluations"])
                    for row in correctness_rows
                ),
                "unique_prefix_evaluations_median": statistics.median(
                    int(row["exact_unique_prefix_evaluations"]) for row in correctness_rows
                ),
                "merged_fraction_median": statistics.median(
                    int(row["exact_merged_prefix_evaluations"])
                    / int(row["exact_trajectory_prefix_evaluations"])
                    for row in correctness_rows
                ),
                "trajectory_count_excludes_greedy_decode": True,
            },
            "step_batch": {
                "trajectory_prefix_evaluations_median": statistics.median(
                    int(row["batch_trajectory_prefix_evaluations"])
                    for row in correctness_rows
                ),
                "unique_prefix_evaluations_median": statistics.median(
                    int(row["batch_unique_prefix_evaluations"])
                    for row in correctness_rows
                ),
                "merged_fraction_median": statistics.median(
                    int(row["batch_merged_prefix_evaluations"])
                    / int(row["batch_trajectory_prefix_evaluations"])
                    for row in correctness_rows
                ),
            },
        },
        "quality_under_iid_matrix_schedule": {
            "overall": overall_quality,
            **quality_by_algorithm,
            "interpretation": (
                "The batched RNG assignment is distribution-equivalent but not pathwise-identical to the "
                "trajectory-major scalar schedule, so selected bytes are reported rather than assumed equal."
            ),
        },
        "selector_timing": {
            "repeats": args.selection_repeats,
            "versus_full60_scalar": versus_full60,
            "versus_sparse19_cached_fields_scalar": versus_sparse_fields,
            "optimized_batch_faster_than_full60_in_median": batch_speedup[
                "paired_speedup_pct_median"
            ] > 0.0,
            "optimized_pathwise_exact_faster_than_full60_in_median": exact_speedup[
                "paired_speedup_pct_median"
            ] > 0.0,
        },
        "claim_boundary": (
            "Warm-cache Python/NumPy table-selection benchmark. It includes RNG, rollout sampling, prefix "
            "deduplication and frozen byte-table minimum selection, but excludes raw feature extraction, "
            "policy loading, real candidate compression and decoding. Cache initialization is reported separately."
        ),
    }
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
