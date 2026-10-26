#!/usr/bin/env python3
"""Analyze frozen Full106 versus compact candidates on external datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DATASETS = ("Bird-migration", "meteoblue")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, values: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def hit_exact(row):
    return float(abs(float(row["proxy_rate"]) - float(row["oracle_rate"])) <= 1.0e-15)


def hit_within(row):
    return float(float(row["oracle_regret_pct"]) <= 1.0 + 1.0e-12)


def macro(values: dict[str, np.ndarray]) -> float:
    return float(np.mean([np.mean(values[dataset]) for dataset in DATASETS]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-evaluation-csv", type=Path, required=True)
    parser.add_argument("--no-zigzag-evaluation-csv", type=Path, required=True)
    parser.add_argument("--no-probes-evaluation-csv", type=Path, required=True)
    parser.add_argument("--external-cost-per-column-csv", type=Path, required=True)
    parser.add_argument("--codec-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200000)
    parser.add_argument("--random-seed", type=int, default=20260826)
    args = parser.parse_args()

    full = [row for row in rows(args.full_evaluation_csv.resolve()) if row["algorithm"] == "C0"]
    variants = {
        "compact_no_zigzag": rows(args.no_zigzag_evaluation_csv.resolve()),
        "compact_no_probes": rows(args.no_probes_evaluation_csv.resolve()),
    }
    if len(full) != 54 or any(len(value) != 54 for value in variants.values()):
        raise AssertionError({"full": len(full), **{key: len(value) for key, value in variants.items()}})
    full_by_key = {(row["dataset"], int(row["column_index"]), int(row["seed"])): row for row in full}
    if len(full_by_key) != 54:
        raise AssertionError("full coverage")

    rng = np.random.default_rng(args.random_seed)
    sizes = {"Bird-migration": 5, "meteoblue": 13}
    boot_indices = {dataset: rng.integers(0, size, size=(args.bootstrap_samples, size), dtype=np.int16) for dataset, size in sizes.items()}
    cost_rows = rows(args.external_cost_per_column_csv.resolve())
    codec_rows = rows(args.codec_csv.resolve())
    if len(codec_rows) != 162 or any(int(row["lossless"]) != 1 for row in codec_rows):
        raise AssertionError("codec replay coverage or losslessness")
    codec_algorithm = {"compact_no_zigzag": "compact_no_zigzag", "compact_no_probes": "compact_no_probes"}
    unit_by_variant = {"compact_no_zigzag": "dim_045", "compact_no_probes": "group_probe28"}
    summary = []
    per_column = []
    for variant_name, variant_values in variants.items():
        variant_by_key = {(row["dataset"], int(row["column_index"]), int(row["seed"])): row for row in variant_values}
        if set(variant_by_key) != set(full_by_key):
            raise AssertionError(variant_name)
        exact_full: dict[str, list[float]] = defaultdict(list)
        exact_variant: dict[str, list[float]] = defaultdict(list)
        within_full: dict[str, list[float]] = defaultdict(list)
        within_variant: dict[str, list[float]] = defaultdict(list)
        regret_full: dict[str, list[float]] = defaultdict(list)
        regret_variant: dict[str, list[float]] = defaultdict(list)
        gains_exact: dict[str, list[float]] = defaultdict(list)
        gains_within: dict[str, list[float]] = defaultdict(list)
        for dataset, column_index in sorted({(key[0], key[1]) for key in full_by_key}):
            keys = [key for key in full_by_key if key[0] == dataset and key[1] == column_index]
            ef = float(np.mean([hit_exact(full_by_key[key]) for key in keys]))
            ev = float(np.mean([hit_exact(variant_by_key[key]) for key in keys]))
            wf = float(np.mean([hit_within(full_by_key[key]) for key in keys]))
            wv = float(np.mean([hit_within(variant_by_key[key]) for key in keys]))
            rf = float(np.median([float(full_by_key[key]["oracle_regret_pct"]) for key in keys]))
            rv = float(np.median([float(variant_by_key[key]["oracle_regret_pct"]) for key in keys]))
            exact_full[dataset].append(ef)
            exact_variant[dataset].append(ev)
            within_full[dataset].append(wf)
            within_variant[dataset].append(wv)
            regret_full[dataset].append(rf)
            regret_variant[dataset].append(rv)
            gains_exact[dataset].append(100.0 * (ef - ev))
            gains_within[dataset].append(100.0 * (wf - wv))
            per_column.append({
                "variant": variant_name,
                "dataset": dataset,
                "column_index": column_index,
                "full_exact_seed_fraction": ef,
                "variant_exact_seed_fraction": ev,
                "full_within_1pct_seed_fraction": wf,
                "variant_within_1pct_seed_fraction": wv,
                "full_median_regret_pct": rf,
                "variant_median_regret_pct": rv,
            })
        exact_arrays = {dataset: np.asarray(gains_exact[dataset]) for dataset in DATASETS}
        within_arrays = {dataset: np.asarray(gains_within[dataset]) for dataset in DATASETS}
        exact_boot = 0.5 * np.mean(exact_arrays[DATASETS[0]][boot_indices[DATASETS[0]]], axis=1) + 0.5 * np.mean(exact_arrays[DATASETS[1]][boot_indices[DATASETS[1]]], axis=1)
        within_boot = 0.5 * np.mean(within_arrays[DATASETS[0]][boot_indices[DATASETS[0]]], axis=1) + 0.5 * np.mean(within_arrays[DATASETS[1]][boot_indices[DATASETS[1]]], axis=1)
        selected_cost = [row for row in cost_rows if row["unit"] == unit_by_variant[variant_name]]
        if len(selected_cost) != 18:
            raise AssertionError((variant_name, len(selected_cost)))
        speedups = {dataset: np.asarray([100.0 * float(row["feature_cost_serial_ms"]) / float(row["selector_serial_ms"]) for row in selected_cost if row["dataset"] == dataset]) for dataset in DATASETS}
        parallel_speedups = {dataset: np.asarray([100.0 * (1.0 - 1.0 / (1.0 + float(row["cost_increase_ideal_parallel_pct"]) / 100.0)) for row in selected_cost if row["dataset"] == dataset]) for dataset in DATASETS}
        full_ms = np.asarray([float(row["selector_serial_ms"]) for row in selected_cost])
        compact_ms = np.asarray([float(row["selector_serial_ms"]) - float(row["feature_cost_serial_ms"]) for row in selected_cost])
        codec_by_algorithm: dict[str, dict[tuple[str, int], list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in codec_rows:
            codec_by_algorithm[row["algorithm"]][(row["dataset"], int(row["column_index"]))].append(float(row["codec_rate"]))
        codec_delta: dict[str, np.ndarray] = {}
        codec_comparisons = []
        for dataset in DATASETS:
            values = []
            for key in sorted(key for key in codec_by_algorithm["full_106"] if key[0] == dataset):
                full_rate = float(np.median(codec_by_algorithm["full_106"][key]))
                compact_rate = float(np.median(codec_by_algorithm[codec_algorithm[variant_name]][key]))
                values.append(100.0 * (compact_rate - full_rate))
                codec_comparisons.append(np.sign(compact_rate - full_rate))
            codec_delta[dataset] = np.asarray(values, dtype=np.float64)
        codec_boot = 0.5 * np.mean(codec_delta[DATASETS[0]][boot_indices[DATASETS[0]]], axis=1) + 0.5 * np.mean(codec_delta[DATASETS[1]][boot_indices[DATASETS[1]]], axis=1)
        exact_gain = macro(exact_arrays)
        within_gain = macro(within_arrays)
        summary.append({
            "variant": variant_name,
            "input_dimensions": 105 if variant_name == "compact_no_zigzag" else 78,
            "external_columns": 18,
            "full_exact_accuracy_pct": 100.0 * macro({dataset: np.asarray(exact_full[dataset]) for dataset in DATASETS}),
            "variant_exact_accuracy_pct": 100.0 * macro({dataset: np.asarray(exact_variant[dataset]) for dataset in DATASETS}),
            "full_minus_variant_exact_gain_pp": exact_gain,
            "exact_gain_ci_low_pp": float(np.quantile(exact_boot, 0.025)),
            "exact_gain_ci_high_pp": float(np.quantile(exact_boot, 0.975)),
            "full_within_1pct_accuracy_pct": 100.0 * macro({dataset: np.asarray(within_full[dataset]) for dataset in DATASETS}),
            "variant_within_1pct_accuracy_pct": 100.0 * macro({dataset: np.asarray(within_variant[dataset]) for dataset in DATASETS}),
            "full_minus_variant_within_gain_pp": within_gain,
            "within_gain_ci_low_pp": float(np.quantile(within_boot, 0.025)),
            "within_gain_ci_high_pp": float(np.quantile(within_boot, 0.975)),
            "full_dataset_macro_median_regret_pct": macro({dataset: np.asarray(regret_full[dataset]) for dataset in DATASETS}),
            "variant_dataset_macro_median_regret_pct": macro({dataset: np.asarray(regret_variant[dataset]) for dataset in DATASETS}),
            "median_full_selector_serial_ms": float(np.median(full_ms)),
            "median_variant_selector_serial_ms": float(np.median(compact_ms)),
            "dataset_macro_selector_speedup_pct": macro(speedups),
            "dataset_macro_ideal_parallel_speedup_pct": macro(parallel_speedups),
            "compact_minus_full_codec_rate_pp": macro(codec_delta),
            "codec_delta_ci_low_pp": float(np.quantile(codec_boot, 0.025)),
            "codec_delta_ci_high_pp": float(np.quantile(codec_boot, 0.975)),
            "codec_columns_compact_better": int(sum(value < 0 for value in codec_comparisons)),
            "codec_columns_tied": int(sum(value == 0 for value in codec_comparisons)),
            "codec_columns_compact_worse": int(sum(value > 0 for value in codec_comparisons)),
            "point_rule_external_pass": int(exact_gain < 5.0 and within_gain < 5.0),
            "ci_noninferiority_supported": int(float(np.quantile(exact_boot, 0.975)) < 5.0 and float(np.quantile(within_boot, 0.975)) < 5.0),
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "external_variant_summary.csv", summary)
    write_csv(output / "external_per_column.csv", per_column)
    report = ["# Frozen external Full106 versus compact models", ""]
    for row in summary:
        report.append(
            f"- {row['variant']}: selector speedup {row['dataset_macro_selector_speedup_pct']:.2f}%; "
            f"Full-minus-compact exact {row['full_minus_variant_exact_gain_pp']:+.2f} pp; "
            f"within-1% {row['full_minus_variant_within_gain_pp']:+.2f} pp; "
            f"real codec delta {row['compact_minus_full_codec_rate_pp']:+.3f} pp; "
            f"point rule {'PASS' if row['point_rule_external_pass'] else 'FAIL'}; "
            f"CI non-inferiority {'supported' if row['ci_noninferiority_supported'] else 'not supported'}."
        )
    report.extend(["", "Bird-migration/meteoblue were used only after the development decision was locked. Negative gains mean the compact model was more accurate. Accuracy is exhaustive-proxy pipeline accuracy, not a production-codec claim."])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {str(path): hashlib.sha256(path.resolve().read_bytes()).hexdigest() for path in (args.full_evaluation_csv, args.no_zigzag_evaluation_csv, args.no_probes_evaluation_csv, args.external_cost_per_column_csv, args.codec_csv)},
        "counts": {"summary_rows": len(summary), "per_column_rows": len(per_column), "full_rows": len(full)},
        "claim_boundary": {"probe_conditioned": True, "external_columns": 18, "production_claim_allowed": False},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
