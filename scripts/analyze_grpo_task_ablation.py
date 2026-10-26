#!/usr/bin/env python3
"""Audit real-codec feature and task-adapted GRPO ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REFERENCE = "feat_full_std_fixed"
ROPE_PP = 0.25


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_dataset_spec(text: str) -> tuple[str, Path, Path, Path, Path]:
    name, separator, rest = text.partition("=")
    paths = rest.split(",")
    if not separator or len(paths) != 4:
        raise argparse.ArgumentTypeError("DATASET=per_column,beam,training_curve,raw_manifest")
    return name, *(Path(path).resolve() for path in paths)


def bootstrap_ci(values: list[float], rng: np.random.Generator, repetitions: int = 20_000) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    samples = rng.choice(data, size=(repetitions, data.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_rows: list[dict[str, object]] = []
    column_rates: dict[tuple[str, str, int], float] = {}
    column_bytes: dict[tuple[str, str, int], int] = {}
    variant_catalog: dict[str, dict] = {}
    input_paths: list[Path] = []
    total_original_by_dataset: dict[str, int] = {}
    beam_bytes_by_dataset: dict[str, int] = {}
    baseline_bytes_by_dataset: dict[str, int] = {}

    for dataset, per_column_path, beam_path, training_path, manifest_path in args.dataset:
        input_paths.extend((per_column_path, beam_path, training_path, manifest_path))
        per_column = read_csv(per_column_path)
        beams = read_csv(beam_path)
        training = read_csv(training_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for variant in manifest["variants"]:
            existing = variant_catalog.setdefault(variant["name"], variant)
            if existing != variant:
                raise AssertionError(f"variant catalog mismatch: {variant['name']}")
        if {row["dataset"] for row in per_column + beams} != {dataset}:
            raise AssertionError(f"dataset mismatch: {dataset}")
        beam_by_column = {int(row["column_index"]): row for row in beams}
        total_original = sum(int(row["original_bytes"]) for row in beams)
        beam_bytes = sum(int(row["beam_encoded_bytes"]) for row in beams)
        baseline_bytes = sum(int(row["baseline_compressed_bytes"]) for row in beams)
        total_original_by_dataset[dataset] = total_original
        beam_bytes_by_dataset[dataset] = beam_bytes
        baseline_bytes_by_dataset[dataset] = baseline_bytes

        final_elapsed: dict[tuple[str, int, int], float] = {}
        final_diag: dict[tuple[str, int, int], dict[str, str]] = {}
        for row in training:
            key = row["algorithm"], int(row["seed"]), int(row["fold"])
            elapsed = float(row["elapsed_sec"])
            if elapsed >= final_elapsed.get(key, -1.0):
                final_elapsed[key] = elapsed
                final_diag[key] = row

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in per_column:
            grouped[row["algorithm"]].append(row)
            key = dataset, row["algorithm"], int(row["column_index"])
            column_rates[key] = float(row["median_codec_rate"])
            column_bytes[key] = int(row["median_encoded_bytes"])
        if set(grouped) != set(variant_catalog):
            raise AssertionError((dataset, sorted(grouped), sorted(variant_catalog)))

        for variant, rows in sorted(grouped.items()):
            if len(rows) != len(beams):
                raise AssertionError((dataset, variant, len(rows), len(beams)))
            selected_bytes = sum(int(row["median_encoded_bytes"]) for row in rows)
            wins = ties = losses = 0
            for row in rows:
                beam_encoded = int(beam_by_column[int(row["column_index"])]["beam_encoded_bytes"])
                candidate = int(row["median_encoded_bytes"])
                wins += candidate < beam_encoded
                ties += candidate == beam_encoded
                losses += candidate > beam_encoded
            run_keys = [key for key in final_elapsed if key[0] == variant]
            diags = [final_diag[key] for key in run_keys]
            dataset_rows.append({
                "dataset": dataset,
                "variant": variant,
                "family": variant_catalog[variant]["family"],
                "columns": len(rows),
                "selected_bytes": selected_bytes,
                "original_bytes": total_original,
                "codec_rate": selected_bytes / total_original,
                "beam29_rate": beam_bytes / total_original,
                "baseline_rate": baseline_bytes / total_original,
                "gain_vs_beam29_pct": 100.0 * (beam_bytes - selected_bytes) / beam_bytes,
                "gain_vs_baseline_pct": 100.0 * (baseline_bytes - selected_bytes) / baseline_bytes,
                "wins_vs_beam29": wins,
                "ties_vs_beam29": ties,
                "losses_vs_beam29": losses,
                "offline_training_sum_sec": sum(final_elapsed[key] for key in run_keys),
                "offline_training_median_sec_per_fold_seed": statistics.median(final_elapsed[key] for key in run_keys),
                "median_standalone_encode_ms": statistics.median(float(row["median_encode_ms"]) for row in rows),
                "median_standalone_decode_ms": statistics.median(float(row["median_decode_ms"]) for row in rows),
                "median_final_clip_fraction": statistics.median(float(row["clip_fraction"]) for row in diags),
                "median_final_approx_kl": statistics.median(float(row["approx_kl"]) for row in diags),
                "median_final_entropy": statistics.median(float(row["policy_entropy"]) for row in diags),
                "median_final_stop_probability": statistics.median(float(row["mean_stop_probability"]) for row in diags),
                "median_final_gradient_norm": statistics.median(float(row["mean_gradient_norm"]) for row in diags),
            })

    if REFERENCE not in variant_catalog:
        raise AssertionError("missing reference")
    dataset_by_key = {(row["dataset"], row["variant"]): row for row in dataset_rows}
    overall_rows: list[dict[str, object]] = []
    all_datasets = sorted(total_original_by_dataset)
    total_original = sum(total_original_by_dataset.values())
    beam_bytes = sum(beam_bytes_by_dataset.values())
    baseline_bytes = sum(baseline_bytes_by_dataset.values())
    reference_bytes = sum(int(dataset_by_key[(dataset, REFERENCE)]["selected_bytes"]) for dataset in all_datasets)
    for variant in sorted(variant_catalog):
        rows = [dataset_by_key[(dataset, variant)] for dataset in all_datasets]
        selected_bytes = sum(int(row["selected_bytes"]) for row in rows)
        overall_rows.append({
            "variant": variant,
            "family": variant_catalog[variant]["family"],
            "datasets": len(rows),
            "columns": sum(int(row["columns"]) for row in rows),
            "selected_bytes": selected_bytes,
            "original_bytes": total_original,
            "codec_rate": selected_bytes / total_original,
            "beam29_rate": beam_bytes / total_original,
            "baseline_rate": baseline_bytes / total_original,
            "gain_vs_beam29_pct": 100.0 * (beam_bytes - selected_bytes) / beam_bytes,
            "gain_vs_full_reference_pct": 100.0 * (reference_bytes - selected_bytes) / reference_bytes,
            "delta_bytes_vs_full_reference": selected_bytes - reference_bytes,
            "wins_vs_beam29": sum(int(row["wins_vs_beam29"]) for row in rows),
            "ties_vs_beam29": sum(int(row["ties_vs_beam29"]) for row in rows),
            "losses_vs_beam29": sum(int(row["losses_vs_beam29"]) for row in rows),
            "offline_training_sum_sec": sum(float(row["offline_training_sum_sec"]) for row in rows),
        })

    rng = np.random.default_rng(20260825)
    feature_rows: list[dict[str, object]] = []
    columns = sorted({(dataset, column) for dataset, _variant, column in column_rates})
    for variant in sorted(name for name, item in variant_catalog.items() if item["family"] == "feature" and name != REFERENCE):
        deltas_pp = [
            100.0 * (column_rates[(dataset, variant, column)] - column_rates[(dataset, REFERENCE, column)])
            for dataset, column in columns
        ]
        ci_low, ci_high = bootstrap_ci(deltas_pp, rng)
        if ci_low > ROPE_PP:
            verdict = "necessary_in_this_prototype"
        elif ci_high < -ROPE_PP:
            verdict = "harmful_in_this_prototype"
        elif ci_low >= -ROPE_PP and ci_high <= ROPE_PP:
            verdict = "not_necessary_within_rope"
        else:
            verdict = "inconclusive"
        overall = next(row for row in overall_rows if row["variant"] == variant)
        feature_rows.append({
            "variant": variant,
            "disabled_features": "+".join(variant_catalog[variant]["disabled_features"]),
            "mean_paired_delta_codec_rate_pp_vs_full": statistics.mean(deltas_pp),
            "median_paired_delta_codec_rate_pp_vs_full": statistics.median(deltas_pp),
            "bootstrap95_mean_delta_low_pp": ci_low,
            "bootstrap95_mean_delta_high_pp": ci_high,
            "rope_pp": ROPE_PP,
            "verdict": verdict,
            "delta_bytes_vs_full_reference": overall["delta_bytes_vs_full_reference"],
            "gain_vs_full_reference_pct": overall["gain_vs_full_reference_pct"],
        })

    method_rows = [dict(row) for row in overall_rows if row["family"] == "method"]
    method_rows.sort(key=lambda row: (int(row["selected_bytes"]), row["variant"]))
    for rank, row in enumerate(method_rows, start=1):
        row["exploratory_byte_rank"] = rank
        row["confirmatory_selection_allowed"] = 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "dataset_variant_summary.csv", dataset_rows)
    write_csv(output_dir / "overall_variant_summary.csv", overall_rows)
    write_csv(output_dir / "feature_ablation.csv", feature_rows)
    write_csv(output_dir / "method_ablation.csv", method_rows)

    best_method = method_rows[0]
    no_global = next(row for row in overall_rows if row["variant"] == "method_std_no_global")
    positive = next(row for row in overall_rows if row["variant"] == "method_robust_adaptive")
    reverse = next(row for row in overall_rows if row["variant"] == "method_robust_reverse_clip")
    reference_overall = next(row for row in overall_rows if row["variant"] == REFERENCE)
    report_lines = [
        "# GRPO task-ablation report",
        "",
        "## Status",
        "",
        "PASS: all rows are real standalone codec measurements and all selected streams passed lossless replay. The comparison is exploratory because these datasets informed method design.",
        "",
        "## Key findings",
        "",
        f"1. Full reference selected {reference_bytes:,} bytes across {len(columns)} columns. Removing the second global z-score changed this by {int(no_global['delta_bytes_vs_full_reference']):+,} bytes.",
        f"2. The smallest exploratory method total is `{best_method['variant']}` at {int(best_method['selected_bytes']):,} bytes ({float(best_method['gain_vs_beam29_pct']):+.4f}% vs Beam-29). This is not a confirmatory winner selection.",
        f"3. Positive-beta robust adaptive clip used {int(positive['selected_bytes']):,} bytes; the reverse-beta falsifier used {int(reverse['selected_bytes']):,}. The direction test determines whether dispersion acts more like signal separation or instability.",
        "4. Feature necessity is reported with a predeclared 0.25 percentage-point ROPE and paired column bootstrap; negative and inconclusive outcomes are retained.",
        "",
        "## Dataset totals",
        "",
        "| Dataset | Full GRPO rate | Beam-29 rate | Best baseline rate | Full offline train |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in all_datasets:
        row = dataset_by_key[(dataset, REFERENCE)]
        report_lines.append(
            f"| {dataset} | {100.0 * float(row['codec_rate']):.5f}% | "
            f"{100.0 * float(row['beam29_rate']):.5f}% | {100.0 * float(row['baseline_rate']):.5f}% | "
            f"{float(row['offline_training_sum_sec']):.2f}s |"
        )
    report_lines.extend([
        "",
        f"Across the pooled datasets the full reference is {100.0 * float(reference_overall['codec_rate']):.5f}% by bytes. Dataset rows are also retained because the pooled byte total is not a dataset-level generalization estimate.",
        "",
        "## Feature ablations",
        "",
        f"Positive delta means removing the block made codec rate worse. The bootstrap resamples the {len(columns)} held-out columns, not unseen datasets.",
        "",
        "| Removed block | Mean paired delta | 95% bootstrap CI | Verdict | Byte delta vs full |",
        "|---|---:|---:|---|---:|",
    ])
    for row in feature_rows:
        report_lines.append(
            f"| `{row['disabled_features']}` | {float(row['mean_paired_delta_codec_rate_pp_vs_full']):+.4f} pp | "
            f"[{float(row['bootstrap95_mean_delta_low_pp']):+.4f}, {float(row['bootstrap95_mean_delta_high_pp']):+.4f}] pp | "
            f"`{row['verdict']}` | {int(row['delta_bytes_vs_full_reference']):+,} |"
        )
    report_lines.extend([
        "",
        "## Method ablations",
        "",
        "| Exploratory rank | Variant | Codec rate | vs full reference | vs Beam-29 | W/T/L vs Beam | Offline train |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in method_rows:
        report_lines.append(
            f"| {int(row['exploratory_byte_rank'])} | `{row['variant']}` | {100.0 * float(row['codec_rate']):.5f}% | "
            f"{float(row['gain_vs_full_reference_pct']):+.4f}% | {float(row['gain_vs_beam29_pct']):+.4f}% | "
            f"{int(row['wins_vs_beam29'])}/{int(row['ties_vs_beam29'])}/{int(row['losses_vs_beam29'])} | "
            f"{float(row['offline_training_sum_sec']):.2f}s |"
        )
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "A lower exploratory byte total motivates a locked confirmatory run; it does not establish that the formula generalizes. The feature ablation changes one normalized block at a time while preserving actor shape and initialization.",
        "",
        "## Next experiment",
        "",
        "Lock one formula without consulting any new outcome, then test it once on fresh datasets. Add calibration-prefix/evaluation-suffix separation if the claim needs to move beyond full-column transductive probing.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "datasets": all_datasets,
        "variants": len(variant_catalog),
        "columns": len(columns),
        "reference": REFERENCE,
        "feature_rope_pp": ROPE_PP,
        "lossless_contract": "enforced by upstream analyze_rl29_codec.py on every codec row",
        "selection_contract": "all variants reported; exploratory rank only; no confirmatory method selection",
        "inputs": {str(path): sha256(path) for path in input_paths},
        "outputs": {},
    }
    for name in ("dataset_variant_summary.csv", "overall_variant_summary.csv", "feature_ablation.csv", "method_ablation.csv", "REPORT.md"):
        manifest["outputs"][name] = sha256(output_dir / name)
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
