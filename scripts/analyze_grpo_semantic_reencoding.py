#!/usr/bin/env python3
"""Analyze fresh-dataset depth/prefix semantic re-encoding codec results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


E_FAMILY = ("E1", "E2", "E3")
D_FAMILY = ("D1", "D2", "D3", "D4")
VARIANTS = E_FAMILY + D_FAMILY


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


def macro_effect(values: dict[str, np.ndarray]) -> float:
    return float(np.mean([np.mean(group) for group in values.values()]))


def stratified_bootstrap(values: dict[str, np.ndarray], samples: int, rng: np.random.Generator) -> np.ndarray:
    datasets = sorted(values)
    output = np.empty(samples, dtype=np.float64)
    batch = 10_000
    for start in range(0, samples, batch):
        stop = min(samples, start + batch)
        pieces = []
        for dataset in datasets:
            group = values[dataset]
            indices = rng.integers(0, group.size, size=(stop - start, group.size))
            pieces.append(group[indices].mean(axis=1))
        output[start:stop] = np.mean(pieces, axis=0)
    return output


def signflip_greater(values: dict[str, np.ndarray], null: float, samples: int, rng: np.random.Generator) -> float:
    observed = macro_effect(values) - null
    shifted = {dataset: group - null for dataset, group in values.items()}
    exceed = 0
    batch = 10_000
    for start in range(0, samples, batch):
        count = min(batch, samples - start)
        pieces = []
        for dataset in sorted(shifted):
            group = shifted[dataset]
            signs = rng.choice((-1.0, 1.0), size=(count, group.size))
            pieces.append((signs * group).mean(axis=1))
        permuted = np.mean(pieces, axis=0)
        exceed += int(np.sum(permuted >= observed - 1.0e-15))
    return (exceed + 1.0) / (samples + 1.0)


def equivalence_p(values: dict[str, np.ndarray], rope: float, samples: int, seed: int) -> float:
    lower = signflip_greater(values, -rope, samples, np.random.default_rng(seed))
    reflected = {dataset: -group for dataset, group in values.items()}
    upper = signflip_greater(reflected, -rope, samples, np.random.default_rng(seed + 1))
    return max(lower, upper)


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (total - rank) * raw[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-csv", type=Path, required=True)
    parser.add_argument("--codec-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--rope-pp", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    rows = read_csv(args.codec_csv.resolve())
    if len(rows) != 450 or any(int(row["lossless"]) != 1 for row in rows):
        raise AssertionError("expected 450 lossless codec rows")
    if {int(row["original_values"]) for row in rows} != {32768}:
        raise AssertionError("evaluation suffix length mismatch")
    algorithms = {row["algorithm"] for row in rows}
    if algorithms != {"C0", *VARIANTS, "Beam-29"}:
        raise AssertionError(algorithms)

    rl_rows = [row for row in rows if row["algorithm"] != "Beam-29"]
    by_key = {
        (row["algorithm"], int(row["seed"]), row["dataset"], int(row["column_index"])): row
        for row in rl_rows
    }
    if len(by_key) != 432:
        raise AssertionError("duplicate RL rows")

    per_seed = []
    for row in rl_rows:
        if row["algorithm"] == "C0":
            continue
        reference = by_key[("C0", int(row["seed"]), row["dataset"], int(row["column_index"]))]
        per_seed.append({
            "variant": row["algorithm"],
            "seed": int(row["seed"]),
            "dataset": row["dataset"],
            "column_index": int(row["column_index"]),
            "column_name": row["column_name"],
            "variant_pipeline": row["safe_pipeline"],
            "reference_pipeline": reference["safe_pipeline"],
            "variant_rate": float(row["codec_rate"]),
            "reference_rate": float(reference["codec_rate"]),
            "delta_rate_pp": 100.0 * (float(row["codec_rate"]) - float(reference["codec_rate"])),
            "delta_bytes": int(row["encoded_bytes"]) - int(reference["encoded_bytes"]),
        })

    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped[(str(row["variant"]), str(row["dataset"]), int(row["column_index"]), str(row["column_name"]))].append(row)
    per_column = []
    for (variant, dataset, column_index, column_name), group in sorted(grouped.items()):
        deltas = np.asarray([float(row["delta_rate_pp"]) for row in group])
        per_column.append({
            "variant": variant,
            "dataset": dataset,
            "column_index": column_index,
            "column_name": column_name,
            "seed_mean_delta_rate_pp": float(np.mean(deltas)),
            "seed_std_delta_rate_pp": float(np.std(deltas, ddof=1)),
            "seed_mean_delta_bytes": float(np.mean([float(row["delta_bytes"]) for row in group])),
        })

    values_by_variant: dict[str, dict[str, np.ndarray]] = {}
    for variant in VARIANTS:
        values_by_variant[variant] = {
            dataset: np.asarray([
                float(row["seed_mean_delta_rate_pp"])
                for row in per_column if row["variant"] == variant and row["dataset"] == dataset
            ])
            for dataset in sorted({row["dataset"] for row in per_column})
        }
        if {group.size for group in values_by_variant[variant].values()} != {5, 13}:
            raise AssertionError((variant, values_by_variant[variant]))

    e_equiv_raw = {
        variant: equivalence_p(values_by_variant[variant], args.rope_pp, args.samples, args.seed + 100 * i)
        for i, variant in enumerate(E_FAMILY)
    }
    d_equiv_raw = {
        variant: equivalence_p(values_by_variant[variant], args.rope_pp, args.samples, args.seed + 1000 + 100 * i)
        for i, variant in enumerate(D_FAMILY)
    }
    d_necessary_raw = {
        variant: signflip_greater(
            values_by_variant[variant], args.rope_pp, args.samples,
            np.random.default_rng(args.seed + 2000 + i),
        )
        for i, variant in enumerate(D_FAMILY)
    }
    e_equiv = holm(e_equiv_raw)
    d_equiv = holm(d_equiv_raw)
    d_necessary = holm(d_necessary_raw)

    results = []
    for index, variant in enumerate(VARIANTS):
        values = values_by_variant[variant]
        family_size = len(E_FAMILY) if variant in E_FAMILY else len(D_FAMILY)
        alpha = 0.05 / family_size
        boot = stratified_bootstrap(values, args.samples, np.random.default_rng(args.seed + 3000 + index))
        low, high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        effect = macro_effect(values)
        flat = np.concatenate(list(values.values()))
        if variant in E_FAMILY:
            if e_equiv[variant] < 0.05 and low >= -args.rope_pp and high <= args.rope_pp:
                verdict = "equivalent_encoding"
            elif low > args.rope_pp or high < -args.rope_pp:
                verdict = "parameterization_sensitive"
            else:
                verdict = "inconclusive"
        else:
            if d_necessary[variant] < 0.05 and low > args.rope_pp:
                verdict = "supported_conditional_contribution"
            elif d_equiv[variant] < 0.05 and low >= -args.rope_pp and high <= args.rope_pp:
                verdict = "empirically_removable"
            elif high < -args.rope_pp:
                verdict = "potentially_harmful"
            else:
                verdict = "inconclusive"
        results.append({
            "variant": variant,
            "family": "equivalent" if variant in E_FAMILY else "deletion",
            "macro_delta_rate_pp": effect,
            "familywise_ci_low_pp": float(low),
            "familywise_ci_high_pp": float(high),
            "equivalence_p_raw": (e_equiv_raw if variant in E_FAMILY else d_equiv_raw)[variant],
            "equivalence_p_holm": (e_equiv if variant in E_FAMILY else d_equiv)[variant],
            "necessity_p_raw": "" if variant in E_FAMILY else d_necessary_raw[variant],
            "necessity_p_holm": "" if variant in E_FAMILY else d_necessary[variant],
            "wins_better_than_c0": int(np.sum(flat < -1.0e-12)),
            "ties": int(np.sum(np.abs(flat) <= 1.0e-12)),
            "losses_worse_than_c0": int(np.sum(flat > 1.0e-12)),
            "Bird-migration_delta_pp": float(np.mean(values["Bird-migration"])),
            "meteoblue_delta_pp": float(np.mean(values["meteoblue"])),
            "worst_column_delta_pp": float(np.max(flat)),
            "verdict": verdict,
        })

    beam = {(row["dataset"], int(row["column_index"])): row for row in rows if row["algorithm"] == "Beam-29"}
    beam_comparison = []
    for dataset, column_index in sorted(beam):
        reference_rows = [
            row for row in rl_rows
            if row["algorithm"] == "C0" and row["dataset"] == dataset and int(row["column_index"]) == column_index
        ]
        c0_rate = float(np.mean([float(row["codec_rate"]) for row in reference_rows]))
        beam_rate = float(beam[(dataset, column_index)]["codec_rate"])
        beam_comparison.append({
            "dataset": dataset,
            "column_index": column_index,
            "column_name": beam[(dataset, column_index)]["column_name"],
            "c0_seed_mean_rate": c0_rate,
            "beam29_rate": beam_rate,
            "c0_minus_beam29_pp": 100.0 * (c0_rate - beam_rate),
            "c0_strictly_better": int(c0_rate < beam_rate),
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "per_seed_delta.csv", per_seed)
    write_csv(output_dir / "per_column_delta.csv", per_column)
    write_csv(output_dir / "variant_results.csv", results)
    write_csv(output_dir / "c0_vs_beam29.csv", beam_comparison)

    lines = [
        "# Fresh-dataset GRPO depth/prefix 语义重编码结果",
        "",
        "## Raw result table",
        "",
        "| Variant | Family | Macro Δ rate | Familywise interval | Holm equivalence p | Holm necessity p | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        necessity = "-" if row["necessity_p_holm"] == "" else f"{float(row['necessity_p_holm']):.6f}"
        lines.append(
            f"| {row['variant']} | {row['family']} | {float(row['macro_delta_rate_pp']):+.4f} pp | "
            f"[{float(row['familywise_ci_low_pp']):+.4f}, {float(row['familywise_ci_high_pp']):+.4f}] | "
            f"{float(row['equivalence_p_holm']):.6f} | {necessity} | {row['verdict']} |"
        )
    lines.extend([
        "",
        "## Key findings",
        "",
        "1. E1-E3 只改变合法编码，不改变状态信息；只有 `equivalent_encoding` 才支持参数化稳定性。",
        "2. D1 只检验显式 depth shortcut，因为 prefix occupancy 本身可推出 depth；不能把它解释为删除了所有 depth 信息。",
        "3. D2 删除执行顺序但保留算子集合和重数；D3 删除 prefix identity/order；D4 删除所有显式 depth/prefix cue。",
        "4. 所有压缩结论来自与 calibration prefix 不重叠的 32,768-value evaluation suffix。",
        "",
        "## Claim boundary",
        "",
        "结果是 standalone normalized-int64 Python codec 上的 probe-conditioned external-test sensitivity，不是生产 codec、zero-shot 或 one-hot bit 必要性结论。",
    ])
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "input_sha256": {
            str(args.codec_csv.resolve()): sha256(args.codec_csv.resolve()),
            str(args.codec_manifest.resolve()): sha256(args.codec_manifest.resolve()),
        },
        "counts": {
            "codec_rows": len(rows),
            "rl_rows": len(rl_rows),
            "beam_rows": len(beam),
            "columns": len(beam),
            "variants": len(VARIANTS),
            "lossless_rows": sum(int(row["lossless"]) for row in rows),
        },
        "statistics": {
            "column_is_primary_unit": True,
            "seeds_are_column_internal_repeats": True,
            "dataset_macro_weighting": "equal",
            "bootstrap_samples": args.samples,
            "signflip_samples": args.samples,
            "rope_pp": args.rope_pp,
            "families": {"equivalent": list(E_FAMILY), "deletion": list(D_FAMILY)},
        },
        "verdicts": {row["variant"]: row["verdict"] for row in results},
        "outputs": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.name != "MANIFEST.json":
            manifest["outputs"][path.name] = sha256(path)
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
