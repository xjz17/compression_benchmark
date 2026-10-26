#!/usr/bin/env python3
"""Analyze depth saturation after disallowing repeated operators.

Depths 1-3 are exact: they are reconstructed by filtering the prior exhaustive
measurements to pipelines with distinct operators.  Depths 4-8 use the widest
no-repeat beam and retain the exact depth-3 incumbent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


MAX_DEPTH = 8
THRESHOLD_PCT = 0.1
EXPECTED_COLUMNS = {"Air-Quality": set(range(14)), "Metro-Traffic": set(range(6))}
EXPECTED_WIDTH = {"Air-Quality": 256, "Metro-Traffic": 64}


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


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-15)


def gain(before: float, after: float) -> float:
    return 100.0 * (before - after) / before if before else 0.0


def operator_tuple(pipeline: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in pipeline.split(" -> ") if part.strip())


def load_exact_norepeat(exact_root: Path):
    summary_paths = sorted(exact_root.glob("*/summary.csv"))
    measurement_paths = sorted(exact_root.glob("*/pipeline_measurements.csv"))
    if len(summary_paths) != 4 or len(measurement_paths) != 4:
        raise AssertionError(
            f"expected four exhaustive shards, got summaries={len(summary_paths)} "
            f"measurements={len(measurement_paths)}"
        )

    metadata: dict[tuple[str, int], dict[str, str]] = {}
    for path in summary_paths:
        for row in read_csv(path):
            key = row["dataset"], int(row["column_index"])
            if key in metadata:
                raise AssertionError(f"duplicate exact summary {key}")
            metadata[key] = row

    exact: dict[tuple[str, int, int], tuple[float, str]] = {}
    eligible_counts = {1: 0, 2: 0, 3: 0}
    for path in measurement_paths:
        for row in read_csv(path):
            if row["lossless"] != "1":
                continue
            depth = int(row["depth"])
            operators = operator_tuple(row["pipeline"])
            if len(operators) != len(set(operators)):
                continue
            eligible_counts[depth] += 1
            key2 = row["dataset"], int(row["column_index"])
            candidate = float(row["rate"]), row["pipeline"]
            key3 = key2 + (depth,)
            if key3 not in exact or candidate < exact[key3]:
                exact[key3] = candidate

    if set(metadata) != {
        (dataset, column) for dataset, columns in EXPECTED_COLUMNS.items() for column in columns
    }:
        raise AssertionError("exact metadata does not cover the expected 20 columns")
    for key2 in metadata:
        for depth in range(1, 4):
            if key2 + (depth,) not in exact:
                raise AssertionError(f"missing exact no-repeat result {key2 + (depth,)}")
    return metadata, exact, summary_paths + measurement_paths, eligible_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--repeat-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    exact_root = args.exact_root.resolve()
    raw_root = args.raw_root.resolve()
    repeat_path = args.repeat_analysis.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    csv_paths = sorted(raw_root.glob("*/beam_depth_results.csv"))
    manifest_paths = sorted(raw_root.glob("*/manifest.json"))
    if not csv_paths or len(csv_paths) != len(manifest_paths):
        raise AssertionError(
            f"expected one manifest per no-repeat lane, got csv={len(csv_paths)} manifests={len(manifest_paths)}"
        )
    for path in manifest_paths:
        config = json.loads(path.read_text(encoding="utf-8"))["config"]
        if config["allow_repeated_operators"] or config["max_depth"] != MAX_DEPTH:
            raise AssertionError(f"unexpected config {path}")

    rows = [row for path in csv_paths for row in read_csv(path)]
    if not rows or len(rows) % MAX_DEPTH:
        raise AssertionError(f"incomplete no-repeat rows: {len(rows)}")
    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = row["dataset"], int(row["column_index"]), int(row["depth"])
        if key in indexed:
            raise AssertionError(f"duplicate {key}")
        if int(row["beam_width"]) != EXPECTED_WIDTH[row["dataset"]]:
            raise AssertionError(f"wrong width {key}")
        if int(row["lossless_expansions"]) <= 0:
            raise AssertionError(f"no lossless expansion {key}")
        indexed[key] = row
    observed_columns = {
        dataset: {column for name, column, _ in indexed if name == dataset}
        for dataset in EXPECTED_COLUMNS
    }
    for dataset, columns in observed_columns.items():
        if not columns or not columns.issubset(EXPECTED_COLUMNS[dataset]):
            raise AssertionError(f"unexpected controlled columns: {dataset} {sorted(columns)}")
        for column in columns:
            for depth in range(1, MAX_DEPTH + 1):
                if (dataset, column, depth) not in indexed:
                    raise AssertionError(f"missing {(dataset, column, depth)}")

    metadata, exact, exact_paths, eligible_counts = load_exact_norepeat(exact_root)
    exact_depth3_optimal_distribution = {1: 0, 2: 0, 3: 0}
    for dataset, column in metadata:
        candidates = [exact[(dataset, column, depth)] for depth in range(1, 4)]
        best = min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))
        best_depth = next(depth for depth, candidate in enumerate(candidates, start=1) if candidate == best)
        exact_depth3_optimal_distribution[best_depth] += 1
    repeat_rows = read_csv(repeat_path)
    repeat = {(row["dataset"], int(row["column_index"])): row for row in repeat_rows}
    if len(repeat) != 20:
        raise AssertionError(f"expected 20 repeat rows, got {len(repeat)}")
    gain_positive = {
        key for key, row in repeat.items()
        if float(row["gain_beyond_depth3_pct"]) > 1.0e-12
    }
    controlled = {(dataset, column) for dataset, columns in observed_columns.items() for column in columns}
    missing_positive_controls = gain_positive - controlled
    if missing_positive_controls:
        raise AssertionError(f"missing no-repeat controls for gain-positive columns: {sorted(missing_positive_controls)}")

    per_depth: list[dict[str, object]] = []
    per_column: list[dict[str, object]] = []
    for dataset in ("Air-Quality", "Metro-Traffic"):
        for column in sorted(observed_columns[dataset]):
            beam_curve = [indexed[(dataset, column, depth)] for depth in range(1, MAX_DEPTH + 1)]
            repeat_row = repeat[(dataset, column)]
            if beam_curve[0]["sample_sha256"] != repeat_row["sample_sha256"]:
                raise AssertionError(f"sample hash mismatch {(dataset, column)}")
            exact_candidates = [exact[(dataset, column, depth)] for depth in range(1, 4)]
            exact_through = [min(exact_candidates[:depth], key=lambda item: (item[0], len(item[1]), item[1])) for depth in range(1, 4)]
            rates: list[float] = []
            pipelines: list[str] = []
            sources: list[str] = []
            for depth, row in enumerate(beam_curve, start=1):
                beam_candidate = float(row["best_rate_through_depth"]), row["best_pipeline_through_depth"]
                if depth <= 3:
                    chosen = exact_through[depth - 1]
                    source = "exhaustive_norepeat"
                else:
                    chosen = min(exact_through[2], beam_candidate, key=lambda item: (item[0], len(item[1]), item[1]))
                    source = "beam_norepeat" if chosen == beam_candidate else "depth3_exhaustive_norepeat_incumbent"
                rates.append(chosen[0])
                pipelines.append(chosen[1])
                sources.append(source)
            final = rates[-1]
            for depth, row in enumerate(beam_curve, start=1):
                operators = operator_tuple(pipelines[depth - 1])
                if len(operators) != len(set(operators)):
                    raise AssertionError(f"repeated operator survived {(dataset, column, depth)}")
                previous = rates[depth - 2] if depth > 1 else rates[0]
                per_depth.append(
                    {
                        "dataset": dataset,
                        "column_index": column,
                        "column_name": row["column_name"],
                        "sample_values": int(row["sample_values"]),
                        "sample_sha256": row["sample_sha256"],
                        "depth": depth,
                        "beam_width": int(row["beam_width"]),
                        "best_rate_through_depth": rates[depth - 1],
                        "best_pipeline_through_depth": pipelines[depth - 1],
                        "best_known_source": sources[depth - 1],
                        "wide_beam_rate_through_depth": float(row["best_rate_through_depth"]),
                        "wide_beam_pipeline_through_depth": row["best_pipeline_through_depth"],
                        "incremental_gain_vs_previous_depth_pct": gain(previous, rates[depth - 1]),
                        "future_gain_to_depth8_pct": gain(rates[depth - 1], final),
                        "cumulative_evaluations": int(row["cumulative_evaluations"]),
                        "cumulative_ms": float(row["cumulative_ms"]),
                    }
                )
            exact_sat = next(
                depth
                for depth in range(1, MAX_DEPTH + 1)
                if all(close(rates[depth - 1], rate) for rate in rates[depth - 1 :])
            )
            practical_sat = next(
                depth
                for depth in range(1, MAX_DEPTH + 1)
                if gain(rates[depth - 1], final) <= THRESHOLD_PCT + 1.0e-12
            )
            per_column.append(
                {
                    "dataset": dataset,
                    "column_index": column,
                    "column_name": beam_curve[0]["column_name"],
                    "sample_values": int(beam_curve[0]["sample_values"]),
                    "sample_sha256": beam_curve[0]["sample_sha256"],
                    "norepeat_exact_saturation_depth_through_8": exact_sat,
                    "norepeat_practical_saturation_depth_0p1pct": practical_sat,
                    "norepeat_depth3_rate": rates[2],
                    "norepeat_depth8_rate": final,
                    "norepeat_gain_beyond_depth3_pct": gain(rates[2], final),
                    "norepeat_depth8_pipeline": pipelines[-1],
                    "repeat_depth8_rate": float(repeat_row["depth8_best_known_rate"]),
                    "repeat_gain_beyond_depth3_pct": float(repeat_row["gain_beyond_depth3_pct"]),
                    "repeat_depth8_pipeline": repeat_row["depth8_best_known_pipeline"],
                    "repeat_advantage_vs_norepeat_depth8_pct": gain(final, float(repeat_row["depth8_best_known_rate"])),
                }
            )

    write_csv(output_dir / "norepeat_per_column_depth.csv", per_depth)
    write_csv(output_dir / "grammar_comparison.csv", per_column)
    meaningful = [row for row in per_column if float(row["norepeat_gain_beyond_depth3_pct"]) > THRESHOLD_PCT + 1.0e-12]
    large = [row for row in per_column if float(row["norepeat_gain_beyond_depth3_pct"]) > 1.0 + 1.0e-12]
    top = sorted(per_column, key=lambda row: float(row["norepeat_gain_beyond_depth3_pct"]), reverse=True)
    report = [
        "# 禁止重复算子的深度对照",
        "",
        f"- 控制列数：{len(per_column)}；覆盖主实验中全部 depth>3 出现可测收益的列。",
        f"- 禁止重复后，depth>3 仍带来超过 {THRESHOLD_PCT:.1f}% 收益：{len(meaningful)}/{len(per_column)} 列。",
        f"- 禁止重复后，depth>3 仍带来超过 1% 收益：{len(large)}/{len(per_column)} 列。",
        "- 禁重复的 exact depth≤3 最优深度分布："
        + "、".join(f"depth {depth}={count} 列" for depth, count in exact_depth3_optimal_distribution.items())
        + "。",
        "- 深度 1–3 来自禁重复管线的精确穷举；深度 4–8 使用宽 Beam，因此深层结论是 best-known、不是 exhaustive oracle。",
        "",
        "| Dataset | Column | No-repeat gain beyond 3 | No-repeat saturation | No-repeat depth-8 pipeline |",
        "|---|---|---:|---:|---|",
    ]
    for row in top[:10]:
        report.append(
            f"| {row['dataset']} | {row['column_name']} | {float(row['norepeat_gain_beyond_depth3_pct']):.6f}% | "
            f"{row['norepeat_practical_saturation_depth_0p1pct']} | `{row['norepeat_depth8_pipeline']}` |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "counts": {"columns": len(per_column), "depth_rows": len(per_depth)},
        "controlled_columns": {
            dataset: sorted(columns) for dataset, columns in observed_columns.items()
        },
        "gain_positive_columns_from_repeat_run": [list(key) for key in sorted(gain_positive)],
        "decision": {
            "columns_gain_over_0p1pct_beyond_depth3": len(meaningful),
            "columns_gain_over_1pct_beyond_depth3": len(large),
        },
        "claim_boundary": {
            "depths_1_to_3_exhaustive": True,
            "depths_4_to_8_exhaustive": False,
            "production_codec_claim_allowed": False,
            "repeated_operators_allowed": False,
        },
        "inputs": {str(path): sha256(path) for path in exact_paths + csv_paths + manifest_paths + [repeat_path]},
        "exact_norepeat_eligible_pipeline_counts": eligible_counts,
        "exact_norepeat_optimal_depth_distribution_through_3": exact_depth3_optimal_distribution,
        "outputs": {
            name: sha256(output_dir / name)
            for name in ("norepeat_per_column_depth.csv", "grammar_comparison.csv", "REPORT.md")
        },
    }
    (output_dir / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
