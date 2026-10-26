#!/usr/bin/env python3
"""Audit depth 1-8 operator search and locate compression-rate saturation.

Depths 1-3 are anchored to the prior exhaustive 28-operator measurements.
Depths 4-8 use the widest available beam as a stronger reference, not an
exhaustive oracle.  The analysis keeps the exact depth-3 incumbent even when a
wide beam fails to rediscover it, so a beam miss cannot be reported as a gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MAX_DEPTH = 8
PRACTICAL_GAIN_PCT = 0.1
REFERENCE_WIDTH = {"Air-Quality": 256, "Metro-Traffic": 64}
EXPECTED_COLUMNS = {"Air-Quality": set(range(14)), "Metro-Traffic": set(range(6))}


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


def gain_pct(before: float, after: float) -> float:
    return 100.0 * (before - after) / before if before else 0.0


def load_exact(exact_root: Path):
    summaries: dict[tuple[str, int], dict[str, str]] = {}
    paths = sorted(exact_root.glob("*/summary.csv"))
    if len(paths) != 4:
        raise AssertionError(f"expected four exhaustive summary shards, got {len(paths)}")
    for path in paths:
        for row in read_csv(path):
            key = row["dataset"], int(row["column_index"])
            if key in summaries:
                raise AssertionError(f"duplicate exhaustive summary {key}")
            if int(row["search_space"]) != 22_764 or int(row["lossless_pipelines"]) != 22_764:
                raise AssertionError(f"incomplete exhaustive result {key}")
            summaries[key] = row
    for dataset, columns in EXPECTED_COLUMNS.items():
        observed = {index for (name, index) in summaries if name == dataset}
        if observed != columns:
            raise AssertionError((dataset, sorted(observed), sorted(columns)))
    return summaries, paths


def load_deep(raw_root: Path):
    rows: list[dict[str, str]] = []
    paths = sorted(raw_root.glob("*/beam_depth_results.csv"))
    manifests = sorted(raw_root.glob("*/manifest.json"))
    if len(paths) != 4 or len(manifests) != 4:
        raise AssertionError(f"expected four complete lanes, got csv={len(paths)} manifest={len(manifests)}")
    for path in paths:
        rows.extend(read_csv(path))
    if len(rows) != 480:
        raise AssertionError(f"expected 480 beam-depth rows, got {len(rows)}")

    indexed: dict[tuple[str, int, int, int], dict[str, str]] = {}
    for row in rows:
        key = (
            row["dataset"],
            int(row["column_index"]),
            int(row["beam_width"]),
            int(row["depth"]),
        )
        if key in indexed:
            raise AssertionError(f"duplicate beam-depth row {key}")
        if int(row["lossless_expansions"]) <= 0:
            raise AssertionError(f"no lossless expansion {key}")
        indexed[key] = row

    for dataset, columns in EXPECTED_COLUMNS.items():
        widths = (4, 29, REFERENCE_WIDTH[dataset])
        for column in columns:
            for width in widths:
                for depth in range(1, MAX_DEPTH + 1):
                    if (dataset, column, width, depth) not in indexed:
                        raise AssertionError(f"missing {(dataset, column, width, depth)}")

    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        config = manifest["config"]
        if config["max_depth"] != MAX_DEPTH or not config["allow_repeated_operators"]:
            raise AssertionError(f"unexpected deep-search config in {path}")
        if config["stop_semantics"] != "best result through depth; unused suffix is STOP/no-op":
            raise AssertionError(f"unexpected STOP semantics in {path}")
    return indexed, paths, manifests


def exact_depth(summary: dict[str, str], depth: int) -> tuple[float, str]:
    single = (float(summary["best_single_rate"]), summary["best_single_pipeline"])
    if depth == 1:
        return single
    pair = (float(summary["best_pair_rate"]), summary["best_pair_pipeline"])
    if depth == 2:
        return min(single, pair, key=lambda item: (item[0], len(item[1]), item[1]))
    oracle = (float(summary["oracle_rate"]), summary["oracle_pipeline"])
    return min(single, pair, oracle, key=lambda item: (item[0], len(item[1]), item[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    exact_root = args.exact_root.resolve()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    summaries, exact_paths = load_exact(exact_root)
    deep, deep_paths, manifest_paths = load_deep(raw_root)

    per_depth: list[dict[str, object]] = []
    per_column: list[dict[str, object]] = []
    width_rows: list[dict[str, object]] = []
    curves: dict[tuple[str, int], list[dict[str, object]]] = {}
    exact_depth3_recovered = 0

    for key in sorted(summaries):
        dataset, column_index = key
        summary = summaries[key]
        reference_width = REFERENCE_WIDTH[dataset]
        sample_values = int(summary["sample_values"])
        sample_sha = summary["sample_sha256"]
        curve: list[dict[str, object]] = []
        exact3_rate, exact3_pipeline = exact_depth(summary, 3)

        for depth in range(1, MAX_DEPTH + 1):
            reference = deep[(dataset, column_index, reference_width, depth)]
            if reference["sample_sha256"] != sample_sha:
                raise AssertionError(f"sample hash mismatch {(dataset, column_index)}")
            beam_rate = float(reference["best_rate_through_depth"])
            beam_pipeline = reference["best_pipeline_through_depth"]
            if depth <= 3:
                best_rate, best_pipeline = exact_depth(summary, depth)
                source = "exhaustive"
                reference_regret = 100.0 * (beam_rate - best_rate) / best_rate
            elif beam_rate < exact3_rate:
                best_rate, best_pipeline = beam_rate, beam_pipeline
                source = f"beam{reference_width}"
                reference_regret = 0.0
            else:
                best_rate, best_pipeline = exact3_rate, exact3_pipeline
                source = "depth3_exhaustive_incumbent"
                reference_regret = 100.0 * (beam_rate - exact3_rate) / exact3_rate
            curve.append(
                {
                    "dataset": dataset,
                    "column_index": column_index,
                    "column_name": summary["column_name"],
                    "logical_type": summary["logical_type"],
                    "sample_values": sample_values,
                    "sample_sha256": sample_sha,
                    "depth": depth,
                    "reference_beam_width": reference_width,
                    "best_known_rate_through_depth": best_rate,
                    "best_known_pipeline_through_depth": best_pipeline,
                    "best_known_source": source,
                    "wide_beam_rate_through_depth": beam_rate,
                    "wide_beam_pipeline_through_depth": beam_pipeline,
                    "wide_beam_regret_vs_anchor_pct": reference_regret,
                    "cumulative_evaluations": int(reference["cumulative_evaluations"]),
                    "cumulative_ms": float(reference["cumulative_ms"]),
                }
            )

        if close(float(curve[2]["wide_beam_rate_through_depth"]), exact3_rate):
            exact_depth3_recovered += 1

        final_rate = float(curve[-1]["best_known_rate_through_depth"])
        for index, row in enumerate(curve):
            rate = float(row["best_known_rate_through_depth"])
            previous_rate = rate if index == 0 else float(curve[index - 1]["best_known_rate_through_depth"])
            row["incremental_gain_vs_previous_depth_pct"] = gain_pct(previous_rate, rate)
            row["future_gain_to_depth8_pct"] = gain_pct(rate, final_rate)
            row["improved_at_this_depth"] = int(index > 0 and rate < previous_rate and not close(rate, previous_rate))
            per_depth.append(row)
        curves[key] = curve

        exact_saturation = next(
            depth
            for depth in range(1, MAX_DEPTH + 1)
            if all(close(float(curve[depth - 1]["best_known_rate_through_depth"]), float(later["best_known_rate_through_depth"])) for later in curve[depth - 1 :])
        )
        practical_saturation = next(
            depth
            for depth in range(1, MAX_DEPTH + 1)
            if float(curve[depth - 1]["future_gain_to_depth8_pct"]) <= PRACTICAL_GAIN_PCT + 1.0e-12
        )
        depth3_rate = float(curve[2]["best_known_rate_through_depth"])
        final_pipeline = str(curve[-1]["best_known_pipeline_through_depth"])
        operators = [part.strip() for part in final_pipeline.split(" -> ") if part.strip()]
        per_column.append(
            {
                "dataset": dataset,
                "column_index": column_index,
                "column_name": summary["column_name"],
                "logical_type": summary["logical_type"],
                "sample_values": sample_values,
                "sample_sha256": sample_sha,
                "exact_saturation_depth_through_8": exact_saturation,
                "practical_saturation_depth_0p1pct": practical_saturation,
                "depth3_rate": depth3_rate,
                "depth8_best_known_rate": final_rate,
                "gain_beyond_depth3_pct": gain_pct(depth3_rate, final_rate),
                "depth8_best_known_pipeline": final_pipeline,
                "depth8_pipeline_length": len(operators),
                "depth8_pipeline_has_repeat": int(len(operators) != len(set(operators))),
                "wide_beam_recovered_exact_depth3": int(close(float(curve[2]["wide_beam_rate_through_depth"]), exact3_rate)),
                "wide_beam_depth3_regret_pct": float(curve[2]["wide_beam_regret_vs_anchor_pct"]),
            }
        )

        for width in (4, 29, reference_width):
            depth8 = deep[(dataset, column_index, width, MAX_DEPTH)]
            rate = float(depth8["best_rate_through_depth"])
            width_rows.append(
                {
                    "dataset": dataset,
                    "column_index": column_index,
                    "column_name": summary["column_name"],
                    "beam_width": width,
                    "depth8_rate": rate,
                    "depth8_pipeline": depth8["best_pipeline_through_depth"],
                    "regret_vs_best_known_depth8_pct": 100.0 * (rate - final_rate) / final_rate,
                    "cumulative_evaluations": int(depth8["cumulative_evaluations"]),
                    "cumulative_ms": float(depth8["cumulative_ms"]),
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for dataset in ("Air-Quality", "Metro-Traffic", "ALL"):
        selected = [key for key in curves if dataset == "ALL" or key[0] == dataset]
        total_original = sum(int(summaries[key]["sample_values"]) * 8 for key in selected)
        depth_rates = []
        for depth in range(1, MAX_DEPTH + 1):
            compressed = sum(
                float(curves[key][depth - 1]["best_known_rate_through_depth"])
                * int(summaries[key]["sample_values"])
                * 8
                for key in selected
            )
            depth_rates.append(compressed / total_original)
        final_rate = depth_rates[-1]
        for depth, rate in enumerate(depth_rates, start=1):
            previous = rate if depth == 1 else depth_rates[depth - 2]
            gains = [float(curves[key][depth - 1]["future_gain_to_depth8_pct"]) for key in selected]
            aggregate_rows.append(
                {
                    "dataset_scope": dataset,
                    "depth": depth,
                    "columns": len(selected),
                    "aggregate_best_known_rate": rate,
                    "incremental_gain_vs_previous_depth_pct": gain_pct(previous, rate),
                    "aggregate_future_gain_to_depth8_pct": gain_pct(rate, final_rate),
                    "columns_practically_saturated_by_depth": sum(value <= PRACTICAL_GAIN_PCT + 1.0e-12 for value in gains),
                    "columns_with_future_gain_over_1pct": sum(value > 1.0 + 1.0e-12 for value in gains),
                    "max_column_future_gain_pct": max(gains),
                }
            )

    global_rows = [row for row in aggregate_rows if row["dataset_scope"] == "ALL"]
    recommended = None
    for row in global_rows:
        if (
            float(row["aggregate_future_gain_to_depth8_pct"]) <= PRACTICAL_GAIN_PCT + 1.0e-12
            and int(row["columns_practically_saturated_by_depth"]) >= 18
            and int(row["columns_with_future_gain_over_1pct"]) == 0
        ):
            recommended = int(row["depth"])
            break

    improved_beyond3 = [row for row in per_column if float(row["gain_beyond_depth3_pct"]) > 1.0e-12]
    meaningful_beyond3 = [row for row in per_column if float(row["gain_beyond_depth3_pct"]) > PRACTICAL_GAIN_PCT + 1.0e-12]
    large_beyond3 = [row for row in per_column if float(row["gain_beyond_depth3_pct"]) > 1.0 + 1.0e-12]
    repeat_winners = [row for row in per_column if int(row["depth8_pipeline_has_repeat"]) == 1]

    write_csv(output_dir / "per_column_depth.csv", per_depth)
    write_csv(output_dir / "per_column_saturation.csv", per_column)
    write_csv(output_dir / "aggregate_depth.csv", aggregate_rows)
    write_csv(output_dir / "beam_width_sensitivity.csv", width_rows)

    top = sorted(per_column, key=lambda row: float(row["gain_beyond_depth3_pct"]), reverse=True)
    report = [
        "# 算子深度 1-8 饱和实验",
        "",
        "## 结论",
        "",
        f"- 20 列中，depth>3 继续降低最佳已知压缩率的列数：{len(improved_beyond3)}/20。",
        f"- depth>3 带来超过 {PRACTICAL_GAIN_PCT:.1f}% 相对字节收益的列数：{len(meaningful_beyond3)}/20。",
        f"- depth>3 带来超过 1% 收益的列数：{len(large_beyond3)}/20。",
        f"- 宽 Beam 在 depth=3 恢复 exact oracle：{exact_depth3_recovered}/20。",
        f"- 满足聚合未来收益不超过 0.1%、至少 18/20 列实用饱和且无列剩余收益超过 1% 的最小统一深度：{recommended if recommended is not None else '未在 depth<=8 内出现'}。",
        f"- depth=8 最佳 pipeline 含重复算子的列数：{len(repeat_winners)}/20。",
        "",
        "## Depth 3 以后收益最大的列",
        "",
        "| Dataset | Column | Gain beyond depth 3 | Practical saturation | Depth-8 best pipeline |",
        "|---|---|---:|---:|---|",
    ]
    for row in top[:10]:
        report.append(
            f"| {row['dataset']} | {row['column_name']} | {float(row['gain_beyond_depth3_pct']):.6f}% | "
            f"{row['practical_saturation_depth_0p1pct']} | `{row['depth8_best_known_pipeline']}` |"
        )
    report.extend(
        [
            "",
            "## 证据边界",
            "",
            "- Depth 1-3 来自 28 个算子、允许重复的 22,764 条 pipeline 完整枚举。",
            "- Depth 4-8 是宽 Beam 参考，不是 exhaustive oracle；分析始终保留 exact depth-3 incumbent。",
            "- 结果来自 normalized-int64、概念算子和 zlib proxy；不等价于生产 codec 的最优深度。",
            "- 重复算子允许存在。若收益主要来自重复 `zigzag` 等路径，首先应收紧 typed grammar，而不是据此扩大 RL horizon。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "counts": {
            "columns": len(per_column),
            "per_column_depth_rows": len(per_depth),
            "aggregate_rows": len(aggregate_rows),
            "beam_width_rows": len(width_rows),
            "depth3_exact_recovered_by_wide_beam": exact_depth3_recovered,
        },
        "decision": {
            "columns_improved_beyond_depth3": len(improved_beyond3),
            "columns_gain_over_0p1pct_beyond_depth3": len(meaningful_beyond3),
            "columns_gain_over_1pct_beyond_depth3": len(large_beyond3),
            "recommended_uniform_depth": recommended,
            "max_tested_depth": MAX_DEPTH,
        },
        "claim_boundary": {
            "depth1_to_3_exhaustive": True,
            "depth4_to_8_exhaustive": False,
            "production_codec_claim_allowed": False,
            "repeated_operators_allowed": True,
        },
        "inputs": {
            str(path): sha256(path)
            for path in exact_paths + deep_paths + manifest_paths
        },
        "outputs": {
            name: sha256(output_dir / name)
            for name in (
                "per_column_depth.csv",
                "per_column_saturation.csv",
                "aggregate_depth.csv",
                "beam_width_sensitivity.csv",
                "REPORT.md",
            )
        },
    }
    (output_dir / "AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
