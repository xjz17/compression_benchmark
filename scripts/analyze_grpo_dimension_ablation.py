#!/usr/bin/env python3
"""Analyze 106D LOFO-retrain ablations against the full GRPO reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SEEDS = (20260825, 20260826, 20260827)


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


def holm_adjust(pvalues: dict[int, float]) -> dict[int, float]:
    ordered = sorted(pvalues, key=lambda key: pvalues[key])
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (total - rank) * pvalues[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def signflip_p(values: np.ndarray, weights: np.ndarray, signs: np.ndarray) -> float:
    observed = float(weights @ values)
    if observed <= 0:
        return 1.0
    null = signs @ (weights * values)
    return float((1 + np.sum(null >= observed)) / (null.size + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension-codec-csvs", nargs=2, type=Path, required=True, metavar=("METRO", "AIR"))
    parser.add_argument("--full-codec-csvs", nargs=2, type=Path, required=True, metavar=("METRO", "AIR"))
    parser.add_argument("--support-csvs", nargs=2, type=Path, required=True, metavar=("METRO", "AIR"))
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rope-pp", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=200000)
    parser.add_argument("--randomization-samples", type=int, default=200000)
    parser.add_argument("--random-seed", type=int, default=20260825)
    args = parser.parse_args()

    catalog = {int(row["dimension"]): row for row in read_csv(args.catalog_csv.resolve())}
    if sorted(catalog) != list(range(106)):
        raise AssertionError("catalog is not 106D")
    dimension_rows = [row for path in args.dimension_codec_csvs for row in read_csv(path.resolve())]
    full_rows = [row for path in args.full_codec_csvs for row in read_csv(path.resolve()) if row["algorithm"] == "feat_full_std_fixed"]
    support_rows = [row for path in args.support_csvs for row in read_csv(path.resolve())]
    if len(dimension_rows) != 106 * 20 * len(SEEDS) or len(full_rows) != 20 * len(SEEDS) or len(support_rows) != 106 * 20:
        raise AssertionError({"dimension": len(dimension_rows), "full": len(full_rows), "support": len(support_rows)})
    if any(int(row["lossless"]) != 1 for row in dimension_rows + full_rows):
        raise AssertionError("non-lossless codec row")

    full_by_key = {(row["dataset"], int(row["seed"]), int(row["column_index"])): row for row in full_rows}
    if len(full_by_key) != 20 * len(SEEDS):
        raise AssertionError("duplicate full reference seed/column")
    per_seed: list[dict[str, object]] = []
    for row in dimension_rows:
        dim = int(row["algorithm"].split("_")[1])
        key = (row["dataset"], int(row["seed"]), int(row["column_index"]))
        full = full_by_key[key]
        if row["input_sha256"] != full["input_sha256"] or int(row["original_bytes"]) != int(full["original_bytes"]):
            raise AssertionError(f"input denominator mismatch {key}")
        item = catalog[dim]
        per_seed.append({
            "dimension": dim,
            "family": item["family"],
            "feature_name": item["feature_name"],
            "inference_class": item["inference_class"],
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "column_index": int(row["column_index"]),
            "column_name": row["column_name"],
            "full_codec_rate": float(full["codec_rate"]),
            "ablated_codec_rate": float(row["codec_rate"]),
            "delta_codec_rate_pp": 100.0 * (float(row["codec_rate"]) - float(full["codec_rate"])),
            "full_encoded_bytes": int(full["encoded_bytes"]),
            "ablated_encoded_bytes": int(row["encoded_bytes"]),
            "delta_encoded_bytes": int(row["encoded_bytes"]) - int(full["encoded_bytes"]),
        })
    if len({(row["dimension"], row["dataset"], row["seed"], row["column_index"]) for row in per_seed}) != 106 * 20 * len(SEEDS):
        raise AssertionError("duplicate dimension/seed/column row")

    grouped_seed: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped_seed[(int(row["dimension"]), str(row["dataset"]), int(row["column_index"]))].append(row)
    per_column: list[dict[str, object]] = []
    for (dim, dataset, column_index), rows in sorted(grouped_seed.items()):
        if len(rows) != len(SEEDS) or {int(row["seed"]) for row in rows} != set(SEEDS):
            raise AssertionError((dim, dataset, column_index, len(rows)))
        delta_pp = float(np.mean([float(row["delta_codec_rate_pp"]) for row in rows]))
        delta_bytes = float(np.mean([float(row["delta_encoded_bytes"]) for row in rows]))
        item = catalog[dim]
        per_column.append({
            "dimension": dim,
            "family": item["family"],
            "feature_name": item["feature_name"],
            "inference_class": item["inference_class"],
            "dataset": dataset,
            "column_index": column_index,
            "column_name": rows[0]["column_name"],
            "full_seed_mean_codec_rate": float(np.mean([float(row["full_codec_rate"]) for row in rows])),
            "ablated_seed_mean_codec_rate": float(np.mean([float(row["ablated_codec_rate"]) for row in rows])),
            "delta_codec_rate_pp": delta_pp,
            "full_seed_mean_encoded_bytes": float(np.mean([float(row["full_encoded_bytes"]) for row in rows])),
            "ablated_seed_mean_encoded_bytes": float(np.mean([float(row["ablated_encoded_bytes"]) for row in rows])),
            "delta_encoded_bytes_seed_mean": delta_bytes,
            "comparison": "worse" if delta_bytes > 0 else "better" if delta_bytes < 0 else "tie",
        })
    if len({(row["dimension"], row["dataset"], row["column_index"]) for row in per_column}) != 106 * 20:
        raise AssertionError("duplicate dimension/column row")

    support_by_dim: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in support_rows:
        support_by_dim[int(row["dimension"])].append(row)

    rng = np.random.default_rng(args.random_seed)
    metro_boot = rng.integers(0, 6, size=(args.bootstrap_samples, 6), dtype=np.int16)
    air_boot = rng.integers(0, 14, size=(args.bootstrap_samples, 14), dtype=np.int16)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(args.randomization_samples, 20))
    weights = np.concatenate((np.full(6, 1.0 / 12.0), np.full(14, 1.0 / 28.0)))
    continuous_dims = [dim for dim, row in catalog.items() if row["inference_class"] == "continuous"]
    if continuous_dims != list(range(47)):
        raise AssertionError("continuous family must be dimensions 0..46")
    p_necessary: dict[int, float] = {}
    p_harmful: dict[int, float] = {}
    intermediate: dict[int, dict[str, object]] = {}
    dataset_rows: list[dict[str, object]] = []

    for dim in range(106):
        rows = [row for row in per_column if row["dimension"] == dim]
        metro = np.asarray([row["delta_codec_rate_pp"] for row in rows if row["dataset"] == "Metro-Traffic"], dtype=np.float64)
        air = np.asarray([row["delta_codec_rate_pp"] for row in rows if row["dataset"] == "Air-Quality"], dtype=np.float64)
        if metro.size != 6 or air.size != 14:
            raise AssertionError((dim, metro.size, air.size))
        effect = 0.5 * float(np.mean(metro)) + 0.5 * float(np.mean(air))
        boot = 0.5 * np.mean(metro[metro_boot], axis=1) + 0.5 * np.mean(air[air_boot], axis=1)
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        family_alpha = 0.05 / len(continuous_dims)
        sim_low, sim_high = np.quantile(boot, [0.5 * family_alpha, 1.0 - 0.5 * family_alpha])
        vector = np.concatenate((metro, air))
        support = support_by_dim[dim]
        min_train_std = min(float(row["train_std"]) for row in support)
        min_train_nonzero = min(float(row["train_nonzero_fraction"]) for row in support)
        if dim in continuous_dims:
            if min_train_std < 1.0e-8:
                p_necessary[dim] = 1.0
                p_harmful[dim] = 1.0
            else:
                p_necessary[dim] = signflip_p(vector - args.rope_pp, weights, signs)
                p_harmful[dim] = signflip_p(-vector - args.rope_pp, weights, signs)
        byte_delta = sum(float(row["delta_encoded_bytes_seed_mean"]) for row in rows)
        intermediate[dim] = {
            "dimension": dim,
            "family": catalog[dim]["family"],
            "feature_name": catalog[dim]["feature_name"],
            "inference_class": catalog[dim]["inference_class"],
            "stratified_macro_delta_pp": effect,
            "pooled_20_column_mean_delta_pp": float(np.mean(vector)),
            "median_delta_pp": float(np.median(vector)),
            "bootstrap95_low_pp": float(ci_low),
            "bootstrap95_high_pp": float(ci_high),
            "bonferroni95_low_pp": float(sim_low),
            "bonferroni95_high_pp": float(sim_high),
            "rope_pp": args.rope_pp,
            "delta_encoded_bytes_seed_mean_vs_full": byte_delta,
            "columns_worse": sum(row["comparison"] == "worse" for row in rows),
            "columns_tied": sum(row["comparison"] == "tie" for row in rows),
            "columns_better": sum(row["comparison"] == "better" for row in rows),
            "min_fold_train_std": min_train_std,
            "min_fold_train_nonzero_fraction": min_train_nonzero,
        }
        for dataset, values in (("Metro-Traffic", metro), ("Air-Quality", air)):
            dataset_rows.append({
                "dimension": dim,
                "family": catalog[dim]["family"],
                "feature_name": catalog[dim]["feature_name"],
                "dataset": dataset,
                "columns": values.size,
                "mean_delta_pp": float(np.mean(values)),
                "median_delta_pp": float(np.median(values)),
                "min_delta_pp": float(np.min(values)),
                "max_delta_pp": float(np.max(values)),
            })

    necessary_adjusted = holm_adjust(p_necessary)
    harmful_adjusted = holm_adjust(p_harmful)
    per_dimension: list[dict[str, object]] = []
    for dim in range(106):
        row = intermediate[dim]
        if dim not in continuous_dims:
            row.update({"necessary_p": "", "necessary_holm_p": "", "harmful_p": "", "harmful_holm_p": "", "verdict": "non_identifiable_one_hot"})
        else:
            row.update({
                "necessary_p": p_necessary[dim],
                "necessary_holm_p": necessary_adjusted[dim],
                "harmful_p": p_harmful[dim],
                "harmful_holm_p": harmful_adjusted[dim],
            })
            if float(row["min_fold_train_std"]) < 1.0e-8:
                verdict = "non_identifiable_zero_variance"
            elif necessary_adjusted[dim] < 0.05 and float(row["bonferroni95_low_pp"]) > args.rope_pp:
                verdict = "supported_conditional_contribution"
            elif harmful_adjusted[dim] < 0.05 and float(row["bonferroni95_high_pp"]) < -args.rope_pp:
                verdict = "potentially_harmful"
            elif float(row["bonferroni95_low_pp"]) > -args.rope_pp and float(row["bonferroni95_high_pp"]) < args.rope_pp:
                verdict = "equivalent_within_rope"
            else:
                verdict = "inconclusive"
            row["verdict"] = verdict
        per_dimension.append(row)

    counts = defaultdict(int)
    for row in per_dimension:
        counts[str(row["verdict"])] += 1
    ranked_positive = sorted((row for row in per_dimension if row["inference_class"] == "continuous"), key=lambda row: float(row["stratified_macro_delta_pp"]), reverse=True)[:8]
    ranked_negative = sorted((row for row in per_dimension if row["inference_class"] == "continuous"), key=lambda row: float(row["stratified_macro_delta_pp"]))[:5]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "per_seed_delta.csv", per_seed)
    write_csv(output_dir / "per_column_delta.csv", per_column)
    write_csv(output_dir / "per_dimension_dataset.csv", dataset_rows)
    write_csv(output_dir / "per_dimension.csv", per_dimension)
    report = [
        "# GRPO 106D coordinate-ablation report",
        "",
        "## Status",
        "",
        "PASS: all 106 coordinates were set to their zero reference after state construction and retrained from the paired initialization. For raw/probe dimensions zero is the training mean; for current-rate it is the best-single reference; one-hot zero is a reference/padding code. Compression claims use standalone codec replay. This is exploratory coordinate sensitivity, not a proof that every feature is necessary.",
        "",
        "## Verdict counts",
        "",
        f"- Continuous dimensions: 47; supported conditional contribution: {counts['supported_conditional_contribution']}; equivalent within 0.25 pp ROPE: {counts['equivalent_within_rope']}; potentially harmful: {counts['potentially_harmful']}; inconclusive: {counts['inconclusive']}; zero-variance non-identifiable: {counts['non_identifiable_zero_variance']}.",
        f"- One-hot coordinates: {counts['non_identifiable_one_hot']}; all are labeled non-identifiable for individual necessity because reference coding and structural redundancy can absorb one removed bit.",
        "- The experimental unit is a held-out column. Three seeds are column-internal repeats, not 60 independent samples.",
        "",
        "## Largest positive continuous-coordinate effects",
        "",
        "| Dim | Feature | Stratified delta | Bonferroni 95% interval | Holm p | Verdict |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in ranked_positive:
        report.append(f"| {row['dimension']} | `{row['feature_name']}` | {float(row['stratified_macro_delta_pp']):+.4f} pp | [{float(row['bonferroni95_low_pp']):+.4f}, {float(row['bonferroni95_high_pp']):+.4f}] | {float(row['necessary_holm_p']):.6f} | `{row['verdict']}` |")
    report.extend(["", "## Largest negative continuous-coordinate effects", "", "| Dim | Feature | Stratified delta | Bonferroni 95% interval | Verdict |", "|---:|---|---:|---:|---|"])
    for row in ranked_negative:
        report.append(f"| {row['dimension']} | `{row['feature_name']}` | {float(row['stratified_macro_delta_pp']):+.4f} pp | [{float(row['bonferroni95_low_pp']):+.4f}, {float(row['bonferroni95_high_pp']):+.4f}] | `{row['verdict']}` |")
    report.extend([
        "",
        "## Interpretation boundary",
        "",
        "A positive effect means that removing one coordinate worsened codec rate after retraining. A null result means no additional independent contribution was detected given the other 105 coordinates; it does not establish that the underlying information is useless. Existing semantic block ablations remain the proper evidence for depth and prefix as concepts.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    inputs = [*args.dimension_codec_csvs, *args.full_codec_csvs, *args.support_csvs, args.catalog_csv]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "counts": {"dimensions": 106, "continuous": 47, "one_hot": 59, "columns": 20, **dict(counts)},
        "statistics": {
            "effect": "equal-weight mean of Metro column mean and Air column mean",
            "bootstrap": f"stratified paired column bootstrap, {args.bootstrap_samples} samples",
            "randomization": f"paired sign-flip, {args.randomization_samples} samples; Holm FWER over 47 continuous dimensions",
            "rope_pp": args.rope_pp,
            "seed": args.random_seed,
        },
        "claim_boundary": "one-hot coordinate results are non-identifiable; all results are exploratory on previously inspected datasets",
        "inputs": {str(path.resolve()): hashlib.sha256(path.resolve().read_bytes()).hexdigest() for path in inputs},
        "outputs": {},
    }
    for name in ("per_seed_delta.csv", "per_column_delta.csv", "per_dimension_dataset.csv", "per_dimension.csv", "REPORT.md"):
        path = output_dir / name
        manifest["outputs"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
