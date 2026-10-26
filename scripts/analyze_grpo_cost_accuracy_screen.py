#!/usr/bin/env python3
"""Recompute development-only accuracy and combine it with feature cost."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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


def exact(row: dict[str, str]) -> float:
    return float(abs(float(row["proxy_rate"]) - float(row["oracle_rate"])) <= 1.0e-15)


def within(row: dict[str, str]) -> float:
    return float(float(row["oracle_regret_pct"]) <= 1.0 + 1.0e-12)


def stratified_bootstrap(values: dict[str, np.ndarray], indices: dict[str, np.ndarray]) -> np.ndarray:
    return 0.5 * np.mean(values["Metro-Traffic"][indices["Metro-Traffic"]], axis=1) + 0.5 * np.mean(values["Air-Quality"][indices["Air-Quality"]], axis=1)


def paired_effect(full_rows, variant_rows, boot_indices):
    full = {(row["dataset"], int(row["column_index"]), int(row["seed"])): row for row in full_rows}
    variant = {(row["dataset"], int(row["column_index"]), int(row["seed"])): row for row in variant_rows}
    if set(full) != set(variant):
        raise AssertionError("paired coverage differs")
    per_column = []
    for dataset, column in sorted({(key[0], key[1]) for key in full}):
        keys = [key for key in full if key[0] == dataset and key[1] == column]
        per_column.append({
            "dataset": dataset,
            "column_index": column,
            "exact_gain": 100.0 * float(np.mean([exact(full[key]) - exact(variant[key]) for key in keys])),
            "within_gain": 100.0 * float(np.mean([within(full[key]) - within(variant[key]) for key in keys])),
            "regret_reduction_pp": float(np.mean([float(variant[key]["oracle_regret_pct"]) - float(full[key]["oracle_regret_pct"]) for key in keys])),
        })
    arrays = {}
    within_arrays = {}
    for dataset in ("Metro-Traffic", "Air-Quality"):
        selected = [row for row in per_column if row["dataset"] == dataset]
        arrays[dataset] = np.asarray([row["exact_gain"] for row in selected], dtype=np.float64)
        within_arrays[dataset] = np.asarray([row["within_gain"] for row in selected], dtype=np.float64)
    exact_boot = stratified_bootstrap(arrays, boot_indices)
    within_boot = stratified_bootstrap(within_arrays, boot_indices)
    return {
        "exact_accuracy_gain_pp": 0.5 * float(np.mean(arrays["Metro-Traffic"])) + 0.5 * float(np.mean(arrays["Air-Quality"])),
        "exact_gain_ci_low_pp": float(np.quantile(exact_boot, 0.025)),
        "exact_gain_ci_high_pp": float(np.quantile(exact_boot, 0.975)),
        "within_1pct_accuracy_gain_pp": 0.5 * float(np.mean(within_arrays["Metro-Traffic"])) + 0.5 * float(np.mean(within_arrays["Air-Quality"])),
        "within_gain_ci_low_pp": float(np.quantile(within_boot, 0.025)),
        "within_gain_ci_high_pp": float(np.quantile(within_boot, 0.975)),
        "metro_exact_gain_pp": float(np.mean(arrays["Metro-Traffic"])),
        "air_exact_gain_pp": float(np.mean(arrays["Air-Quality"])),
        "columns": len(per_column),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension-evaluation-csvs", nargs=2, type=Path, required=True)
    parser.add_argument("--full-evaluation-csvs", nargs=2, type=Path, required=True)
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument("--cost-summary-csv", type=Path, required=True)
    parser.add_argument("--cost-per-column-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200000)
    parser.add_argument("--random-seed", type=int, default=20260826)
    args = parser.parse_args()

    dimensions = [row for path in args.dimension_evaluation_csvs for row in read_csv(path.resolve())]
    all_group_rows = [row for path in args.full_evaluation_csvs for row in read_csv(path.resolve())]
    full_rows = [row for row in all_group_rows if row["algorithm"] == "feat_full_std_fixed"]
    if len(dimensions) != 106 * 60 or len(full_rows) != 60:
        raise AssertionError((len(dimensions), len(full_rows)))
    catalog = {int(row["dimension"]): row for row in read_csv(args.catalog_csv.resolve())}
    cost_summary = {row["unit"]: row for row in read_csv(args.cost_summary_csv.resolve())}
    cost_columns = read_csv(args.cost_per_column_csv.resolve())

    rng = np.random.default_rng(args.random_seed)
    boot_indices = {
        "Metro-Traffic": rng.integers(0, 6, size=(args.bootstrap_samples, 6), dtype=np.int16),
        "Air-Quality": rng.integers(0, 14, size=(args.bootstrap_samples, 14), dtype=np.int16),
    }

    continuous_rows = []
    locked_dims = []
    for dim in range(47):
        variant = [row for row in dimensions if int(row["disabled_dimension"]) == dim]
        effect = paired_effect(full_rows, variant, boot_indices)
        if dim < 18:
            cost = None
            point = False
            conservative = False
            verdict = "not_applicable_shared_raw_scan"
        else:
            cost_row = cost_summary[f"dim_{dim:03d}"]
            cost = float(cost_row["dataset_macro_cost_increase_serial_pct"])
            point = cost >= 5.0 and effect["exact_accuracy_gain_pp"] < 5.0 and effect["within_1pct_accuracy_gain_pp"] < 5.0
            selected_cost = [row for row in cost_columns if row["unit"] == f"dim_{dim:03d}"]
            by_dataset = {
                dataset: np.asarray([float(row["cost_increase_serial_pct"]) for row in selected_cost if row["dataset"] == dataset])
                for dataset in ("Metro-Traffic", "Air-Quality")
            }
            cost_boot = stratified_bootstrap(by_dataset, boot_indices)
            cost_ci_low, cost_ci_high = np.quantile(cost_boot, [0.025, 0.975])
            conservative = float(cost_ci_low) >= 5.0 and effect["exact_gain_ci_high_pp"] < 5.0 and effect["within_gain_ci_high_pp"] < 5.0
            verdict = "point_candidate_threshold_unstable" if point and not conservative else "supported_delete" if conservative else "cost_gate_not_triggered" if cost < 5.0 else "accuracy_gate_retains"
            if point:
                locked_dims.append(dim)
        continuous_rows.append({
            "dimension": dim,
            "family": catalog[dim]["family"],
            "feature_name": catalog[dim]["feature_name"],
            "deployable_individual_cost": int(dim >= 18),
            "cost_increase_serial_pct": "" if cost is None else cost,
            "cost_ci_low_pct": "" if cost is None else float(cost_ci_low),
            "cost_ci_high_pct": "" if cost is None else float(cost_ci_high),
            **effect,
            "point_rule_delete": int(point),
            "ci_supported_delete": int(conservative),
            "verdict": verdict,
        })

    group_specs = (
        ("group_raw18", "feat_no_raw"),
        ("group_probe28", "feat_no_probes"),
        ("dim_046", "feat_no_rate"),
    )
    group_rows = []
    locked_groups = []
    for unit, algorithm in group_specs:
        variant = [row for row in all_group_rows if row["algorithm"] == algorithm]
        effect = paired_effect(full_rows, variant, boot_indices)
        cost = float(cost_summary[unit]["dataset_macro_cost_increase_serial_pct"])
        point = cost >= 5.0 and effect["exact_accuracy_gain_pp"] < 5.0 and effect["within_1pct_accuracy_gain_pp"] < 5.0
        if point:
            locked_groups.append(unit)
        group_rows.append({
            "unit": unit,
            "algorithm": algorithm,
            "cost_increase_serial_pct": cost,
            **effect,
            "point_rule_delete": int(point),
            "verdict": "point_candidate_requires_joint_external_test" if point else "rule_retains_or_cost_gate_not_triggered",
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "continuous_feature_screen.csv", continuous_rows)
    write_csv(output / "deployable_group_screen.csv", group_rows)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_coordinate_point_candidates": locked_dims,
        "locked_coordinate_feature_names": [catalog[dim]["feature_name"] for dim in locked_dims],
        "locked_group_point_candidates": locked_groups,
        "ci_supported_coordinate_deletions": [int(row["dimension"]) for row in continuous_rows if int(row["ci_supported_delete"])],
        "external_variants_frozen_before_external_evaluation": ["full_106", "compact_coordinate_candidates", "compact_no_probes_group"],
        "rule": "serial selector cost >=5%; exact and within-1% gains both <5 pp",
        "claim_boundary": "point candidates are threshold-unstable unless CI-supported; external set only tests frozen candidates",
    }
    (output / "LOCKED_DECISION.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "# Development-only cost-accuracy screen",
        "",
        f"Point-rule coordinate candidates: {locked_dims or 'none'}.",
        f"Point-rule deployable-group candidates: {locked_groups or 'none'}.",
        f"CI-supported coordinate deletions: {decision['ci_supported_coordinate_deletions'] or 'none'}.",
        "",
        "Raw coordinates 0-17 share one extractor and are not assigned fictitious individual savings. Depth/prefix one-hot coordinates remain semantic-group questions. Bird-migration and meteoblue were not used for this locked decision.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": decision["created_utc"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {str(path): hashlib.sha256(path.resolve().read_bytes()).hexdigest() for path in args.dimension_evaluation_csvs + args.full_evaluation_csvs + [args.catalog_csv, args.cost_summary_csv, args.cost_per_column_csv]},
        "counts": {"continuous_dimensions": len(continuous_rows), "group_rows": len(group_rows), "full_rows": len(full_rows), "dimension_rows": len(dimensions)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
