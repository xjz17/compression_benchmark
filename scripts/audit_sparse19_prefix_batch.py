#!/usr/bin/env python3
"""Deterministically recompute the Sparse19 prefix-cache benchmark claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1.0e-9:
        raise AssertionError((actual, expected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--sparse19-policy-dir", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    summary_path = result_dir / "SUMMARY.json"
    correctness_path = result_dir / "correctness_and_quality.csv"
    samples_path = result_dir / "selector_samples.csv"
    cache_path = result_dir / "cache_initialization.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    correctness = rows(correctness_path)
    samples = rows(samples_path)
    cache = rows(cache_path)
    repeats = int(summary["selector_timing"]["repeats"])
    variants = tuple(summary["selector_timing"]["versus_full60_scalar"]["variants"])

    if len(correctness) != 120 or len(cache) != 12:
        raise AssertionError((len(correctness), len(cache)))
    if len(samples) != 40 * repeats * len(variants):
        raise AssertionError((len(samples), 40 * repeats * len(variants)))
    if any(
        row["archived_selected_pipeline_and_count_exact"] != "1"
        or row["cached_scalar_exact"] != "1"
        or row["unique_prefix_exact"] != "1"
        or row["batch_full60_sparse19_exact"] != "1"
        for row in correctness
    ):
        raise AssertionError("correctness flag failed")

    scalar_bytes = sum(int(row["scalar_encoded_bytes"]) for row in correctness)
    batch_bytes = sum(int(row["batch_encoded_bytes"]) for row in correctness)
    quality = summary["quality_under_iid_matrix_schedule"]["overall"]
    if scalar_bytes != quality["total_scalar_bytes"] or batch_bytes != quality["total_batch_bytes"]:
        raise AssertionError((scalar_bytes, batch_bytes, quality))

    gates = summary["correctness_gates"]
    if gates["step_batch_vs_scalar_proposal_sets_exact"] != sum(
        int(row["batch_matches_scalar_proposal_set"]) for row in correctness
    ):
        raise AssertionError(gates)
    merging = summary["prefix_merging"]
    exact_merging = merging["pathwise_exact"]
    batch_merging = merging["step_batch"]
    close(
        statistics.median(int(row["exact_trajectory_prefix_evaluations"]) for row in correctness),
        exact_merging["trajectory_prefix_evaluations_median"],
    )
    close(
        statistics.median(int(row["exact_unique_prefix_evaluations"]) for row in correctness),
        exact_merging["unique_prefix_evaluations_median"],
    )
    close(
        statistics.median(
            int(row["exact_merged_prefix_evaluations"])
            / int(row["exact_trajectory_prefix_evaluations"])
            for row in correctness
        ),
        exact_merging["merged_fraction_median"],
    )
    close(
        statistics.median(int(row["batch_trajectory_prefix_evaluations"]) for row in correctness),
        batch_merging["trajectory_prefix_evaluations_median"],
    )
    close(
        statistics.median(int(row["batch_unique_prefix_evaluations"]) for row in correctness),
        batch_merging["unique_prefix_evaluations_median"],
    )
    close(
        statistics.median(
            int(row["batch_merged_prefix_evaluations"])
            / int(row["batch_trajectory_prefix_evaluations"])
            for row in correctness
        ),
        batch_merging["merged_fraction_median"],
    )

    grouped: dict[tuple[str, int, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in samples:
        grouped[(row["dataset"], int(row["column_index"]), row["algorithm"])][
            row["variant"]
        ].append(int(row["elapsed_ns"]))
    if len(grouped) != 40:
        raise AssertionError(len(grouped))
    medians: dict[tuple[str, int, str], dict[str, float]] = {
        key: {variant: statistics.median(values) for variant, values in by_variant.items()}
        for key, by_variant in grouped.items()
    }
    timing = summary["selector_timing"]["versus_full60_scalar"]["variants"]
    recomputed: dict[str, dict[str, float]] = {}
    for variant in variants:
        group_values = [by_variant[variant] for by_variant in medians.values()]
        speedups = [
            100.0 * (by_variant["full60_scalar"] - by_variant[variant])
            / by_variant["full60_scalar"]
            for by_variant in medians.values()
        ]
        recomputed[variant] = {
            "group_median_ms": statistics.median(group_values) / 1_000_000.0,
            "paired_speedup_pct_median": statistics.median(speedups),
        }
        close(recomputed[variant]["group_median_ms"], timing[variant]["group_median_ms"])
        close(
            recomputed[variant]["paired_speedup_pct_median"],
            timing[variant]["paired_speedup_pct_median"],
        )

    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "checks": {
            "correctness_rows": len(correctness),
            "all_scalar_and_unique_prefix_exact": True,
            "selector_sample_rows": len(samples),
            "paired_groups": len(grouped),
            "variants": len(variants),
            "timing_summary_recomputed": True,
            "quality_byte_totals_recomputed": True,
            "strict_and_step_batch_prefix_diagnostics_recomputed": True,
            "step_batch_scalar_proposal_set_matches": gates[
                "step_batch_vs_scalar_proposal_sets_exact"
            ],
        },
        "recomputed": recomputed,
        "source_sha256": {
            path.name: sha256(path)
            for path in (summary_path, correctness_path, samples_path, cache_path)
        },
        "input_sha256": {
            str(path.resolve()): sha256(path.resolve())
            for path in (
                sorted(args.measurement_root.resolve().glob("*/pipeline_measurements.csv"))
                + [args.search_dir.resolve() / "rl_evaluation.csv"]
                + sorted((args.search_dir.resolve() / "policies").glob("*.npz"))
                + sorted(args.sparse19_policy_dir.resolve().glob("*.npz"))
                + [args.feature_csv.resolve()]
            )
        },
        "claim_boundary": (
            "Deterministic table and arithmetic audit with input hashes; it does not independently rerun policies, "
            "inspect hardware counters or establish production latency."
        ),
    }
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
