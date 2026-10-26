#!/usr/bin/env python3
"""Paired analysis for the multi-seed PV2F feature-ablation confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


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


def row_key(row: dict[str, str]) -> tuple[int, str, int]:
    return int(row["seed"]), row["test_dataset"], int(row["column_index"])


def bootstrap_upper(candidate: np.ndarray, baseline: np.ndarray, repeats: int = 20000) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260827)
    estimates = np.empty(repeats)
    for index in range(repeats):
        take = rng.integers(0, candidate.size, candidate.size)
        estimates[index] = 100.0 * (candidate[take].sum() - baseline[take].sum()) / baseline[take].sum()
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 50.0, 95.0]))


def embedding_min_distance(result_dir: Path) -> float | None:
    distances: list[float] = []
    for policy in sorted((result_dir / "policies").glob("*.npz")):
        data = np.load(policy)
        if len(data.files) != 7:
            return None
        for key in ("arr_4", "arr_5", "arr_6"):
            values = np.asarray(data[key])
            for left in range(values.shape[0]):
                for right in range(left + 1, values.shape[0]):
                    distances.append(float(np.linalg.norm(values[left] - values[right])))
    return min(distances) if distances else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--baseline-name", default="full60")
    parser.add_argument("--variant-prefix", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    summaries = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/summary.json"))
        if not args.variant_prefix or path.parent.name.startswith(args.variant_prefix)
    }
    evaluations = {name: read_csv(root / name / "evaluation.csv") for name in summaries}
    result_dirs = {name: root / name for name in summaries}
    if args.baseline_result is not None:
        baseline_dir = args.baseline_result.resolve()
        summaries[args.baseline_name] = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
        evaluations[args.baseline_name] = read_csv(baseline_dir / "evaluation.csv")
        result_dirs[args.baseline_name] = baseline_dir
    if args.baseline_name not in summaries:
        raise AssertionError(f"missing baseline {args.baseline_name}")
    protocol = summaries[args.baseline_name]["protocol"]
    if any(summary["protocol"] != protocol for summary in summaries.values()):
        raise AssertionError("mixed protocols in one comparison")
    base_rows = {row_key(row): row for row in evaluations[args.baseline_name]}
    seeds = sorted({int(row["seed"]) for row in evaluations[args.baseline_name]})
    column_keys = sorted({(row["test_dataset"], int(row["column_index"])) for row in evaluations[args.baseline_name]})

    rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    for name, summary in summaries.items():
        current = {row_key(row): row for row in evaluations[name]}
        seed_totals = []
        seed_base_totals = []
        seed_acc = []
        seed_exact = []
        for seed in seeds:
            selected = [row for key, row in current.items() if key[0] == seed]
            baseline = [row for key, row in base_rows.items() if key[0] == seed]
            total = sum(int(row["encoded_bytes"]) for row in selected)
            base_total = sum(int(row["encoded_bytes"]) for row in baseline)
            seed_totals.append(total)
            seed_base_totals.append(base_total)
            seed_acc.append(sum(int(row["acc_at_1pct"]) for row in selected))
            seed_exact.append(sum(int(row["byte_equivalent_oracle_hit"]) for row in selected))
            per_seed_rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "total_bytes": total,
                    "full60_bytes": base_total,
                    "penalty_vs_full_pct": 100.0 * (total - base_total) / base_total,
                    "acc_at_1pct_hits": seed_acc[-1],
                    "byte_equivalent_hits": seed_exact[-1],
                }
            )

        candidate_medians = []
        baseline_medians = []
        column_penalties = []
        for dataset, column in column_keys:
            candidate_values = [int(current[(seed, dataset, column)]["encoded_bytes"]) for seed in seeds]
            baseline_values = [int(base_rows[(seed, dataset, column)]["encoded_bytes"]) for seed in seeds]
            candidate_median = float(statistics.median(candidate_values))
            baseline_median = float(statistics.median(baseline_values))
            candidate_medians.append(candidate_median)
            baseline_medians.append(baseline_median)
            column_penalties.append(100.0 * (candidate_median - baseline_median) / baseline_median)
        candidate_array = np.asarray(candidate_medians)
        baseline_array = np.asarray(baseline_medians)
        ci_low, ci_mid, ci_high = bootstrap_upper(candidate_array, baseline_array)
        median_total = statistics.median(seed_totals)
        median_base = statistics.median(seed_base_totals)
        aggregate_penalty = 100.0 * (median_total - median_base) / median_base
        base_acc = statistics.median(
            sum(int(row["acc_at_1pct"]) for key, row in base_rows.items() if key[0] == seed)
            for seed in seeds
        )
        acc_delta = 100.0 * (statistics.median(seed_acc) - base_acc) / len(column_keys)
        point_gate = aggregate_penalty <= 0.25 and max(column_penalties) <= 5.0 and acc_delta >= -5.0
        strict = point_gate and ci_high <= 0.25
        provisional = point_gate and ci_high <= 1.0
        rows.append(
            {
                "variant": name,
                "dense_model_input_dim": int(summary["dense_model_input_dim"]),
                "logical_observation_fields": int(summary["logical_observation_fields"]),
                "raw_dimensions": len(summary["raw_dimensions"]),
                "raw_blocks": "+".join(summary["enabled_raw_blocks"]) or "none",
                "prefix_mode": summary["prefix_mode"],
                "embedding_dim": int(summary["embedding_dim"]),
                "feature_ms": float(summary["feature_extraction_ms_20_columns_median"]),
                "median_total_bytes": median_total,
                "median_full60_bytes": median_base,
                "aggregate_penalty_vs_full_pct": aggregate_penalty,
                "median_column_penalty_vs_full_pct": statistics.median(column_penalties),
                "max_column_penalty_vs_full_pct": max(column_penalties),
                "median_acc_at_1pct_hits": statistics.median(seed_acc),
                "acc_at_1pct_delta_pp": acc_delta,
                "median_byte_equivalent_hits": statistics.median(seed_exact),
                "paired_bootstrap_low_pct": ci_low,
                "paired_bootstrap_mid_pct": ci_mid,
                "paired_bootstrap_high_pct": ci_high,
                "embedding_min_pair_distance": embedding_min_distance(result_dirs[name]),
                "point_gate": "PASS" if point_gate else "FAIL",
                "strict_gate": "PASS" if strict else "FAIL",
                "provisional_gate": "PASS" if provisional else "FAIL",
            }
        )
    rows.sort(key=lambda row: (row["dense_model_input_dim"], row["median_total_bytes"], row["variant"]))
    write_csv(output / "confirm_comparison.csv", rows)
    write_csv(output / "per_seed_comparison.csv", per_seed_rows)

    point = [row for row in rows if row["point_gate"] == "PASS"]
    final_candidates = point[:3]
    if not any(row["variant"] == args.baseline_name for row in final_candidates):
        final_candidates.append(next(row for row in rows if row["variant"] == args.baseline_name))
    (output / "final_candidates.json").write_text(json.dumps(final_candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    split_label = (
        f"Air leave-one-column-out：{len(column_keys)} 列"
        if protocol == "air_loco"
        else f"Air train -> Metro lockbox：{len(column_keys)} 列"
    )
    boundary = (
        "当前表只决定进入 10-seed 与 Metro lockbox 的候选。任何 embedding 候选还必须满足 token embedding 无精确碰撞。"
        if protocol == "air_loco"
        else "此表是冻结候选的 Metro lockbox 结果，不再用于调参或重新选择候选。"
    )
    report = [
        "# PV2F PPO 特征确认实验",
        "",
        f"- seeds：{len(seeds)}",
        f"- {split_label}",
        "- paired bootstrap 单位：列；每列先取跨 seed 中位字节",
        "",
        "| variant | dim | feature ms | median bytes | delta | Acc@1% | max column | CI95 upper | point | strict | provisional |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['variant']} | {row['dense_model_input_dim']} | {row['feature_ms']:.3f} | {int(row['median_total_bytes']):,} | {row['aggregate_penalty_vs_full_pct']:+.3f}% | {int(row['median_acc_at_1pct_hits'])}/{len(column_keys)} | {row['max_column_penalty_vs_full_pct']:+.3f}% | {row['paired_bootstrap_high_pct']:+.3f}% | {row['point_gate']} | {row['strict_gate']} | {row['provisional_gate']} |"
        )
    report.extend(
        [
            "",
            boundary,
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(rows), "final_candidates": [row["variant"] for row in final_candidates]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
