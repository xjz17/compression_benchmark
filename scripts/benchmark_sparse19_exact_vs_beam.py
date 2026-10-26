#!/usr/bin/env python3
"""Same-process paired timing of Beam-29 versus pathwise-exact Sparse19 selection."""

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
from typing import Any

import numpy as np

import search_physical_operator_variants_v2 as base
from benchmark_physical_operator_selected_timings_v2 import load_sparse19_actor, representative
from benchmark_sparse19_prefix_batch import archived_features, unique_prefix_exact_selection


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--sparse19-policy-dir", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tables = base.load_tables(args.measurement_root.resolve())
    raw = archived_features(args.feature_csv.resolve(), tables)
    table_position = {table.key: index for index, table in enumerate(tables)}
    datasets = sorted({table.dataset for table in tables})
    builders: dict[int, base.Sparse19FeatureBuilder] = {}
    for fold, test_dataset in enumerate(datasets):
        train = np.asarray(
            [index for index, table in enumerate(tables) if table.dataset != test_dataset],
            dtype=np.int32,
        )
        builders[fold] = base.Sparse19FeatureBuilder(raw, train)

    rl_rows = read_csv(args.search_dir.resolve() / "rl_evaluation.csv")
    proposal_budget = int(rl_rows[0]["proposal_rollouts"])
    grouped_rl: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rl_rows:
        grouped_rl[(row["test_dataset"], int(row["column_index"]), row["algorithm"])].append(row)
    actors: dict[tuple[str, int, int], base.Sparse19MLP] = {}
    caches: dict[tuple[str, int, int], base.Sparse19InferenceCache] = {}
    samples: list[dict[str, Any]] = []
    order_rng = np.random.default_rng(20260829)
    for table in tables:
        column = table_position[table.key]
        for algorithm in ("ppo", "grpo"):
            row = representative(grouped_rl[(table.dataset, table.column_index, algorithm)])
            key = (algorithm, int(row["seed"]), int(row["fold"]))
            if key not in actors:
                actors[key] = load_sparse19_actor(
                    args.sparse19_policy_dir.resolve()
                    / f"{key[0]}_seed{key[1]}_fold{key[2]}.npz"
                )
                caches[key] = base.Sparse19InferenceCache(actors[key], builders[key[2]])
            cache = caches[key]
            beam_expected = base.beam29_select(table, 29)["pipeline"]
            exact_expected = unique_prefix_exact_selection(
                cache, table, key[1], key[2], column, proposal_budget
            )
            if exact_expected[:2] != (
                base.parse_pipeline(row["pipeline"]), int(row["unique_proposals"])
            ):
                raise AssertionError((table.key, key, exact_expected[:2]))
            functions = {
                "beam29_selector_only": lambda: base.beam29_select(table, 29)["pipeline"],
                "sparse19_unique_prefix_exact": lambda: unique_prefix_exact_selection(
                    cache, table, key[1], key[2], column, proposal_budget
                ),
            }
            for repeat in range(args.repeats):
                order = list(functions)
                order_rng.shuffle(order)
                for order_in_pair, variant in enumerate(order):
                    started = time.perf_counter_ns()
                    result = functions[variant]()
                    elapsed = time.perf_counter_ns() - started
                    if variant == "beam29_selector_only":
                        if result != beam_expected:
                            raise AssertionError((table.key, result, beam_expected))
                    elif result != exact_expected:
                        raise AssertionError((table.key, key, "exact changed"))
                    samples.append({
                        "dataset": table.dataset,
                        "column_index": table.column_index,
                        "column_name": table.column_name,
                        "algorithm": algorithm,
                        "seed": key[1],
                        "fold": key[2],
                        "repeat": repeat,
                        "order_in_pair": order_in_pair,
                        "variant": variant,
                        "elapsed_ns": elapsed,
                    })

    grouped: dict[tuple[str, int, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in samples:
        grouped[(row["dataset"], row["column_index"], row["algorithm"])][
            row["variant"]
        ].append(row["elapsed_ns"])
    group_medians = {
        key: {variant: statistics.median(values) for variant, values in variants.items()}
        for key, variants in grouped.items()
    }
    beam = [values["beam29_selector_only"] for values in group_medians.values()]
    exact = [values["sparse19_unique_prefix_exact"] for values in group_medians.values()]
    speedups = [100.0 * (old - new) / old for old, new in zip(beam, exact)]
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
            "same process, same column and representative policy, randomized AB/BA order; "
            "Beam timing includes selector frozen-table lookups only and excludes the offline full-table oracle scan"
        ),
        "groups": len(group_medians),
        "repeats": args.repeats,
        "beam29_group_median_ms": statistics.median(beam) / 1_000_000.0,
        "exact_group_median_ms": statistics.median(exact) / 1_000_000.0,
        "paired_exact_speedup_vs_beam_pct_median": statistics.median(speedups),
        "paired_exact_speedup_vs_beam_pct_min": min(speedups),
        "paired_exact_speedup_vs_beam_pct_max": max(speedups),
        "all_groups_exact_faster": all(value > 0.0 for value in speedups),
        "beam_timer_excludes_offline_oracle_scan": True,
        "claim_boundary": "Warm-cache frozen-byte-table lookup selection; excludes feature extraction and real candidate compression.",
    }
    write_csv(output / "samples.csv", samples)
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
