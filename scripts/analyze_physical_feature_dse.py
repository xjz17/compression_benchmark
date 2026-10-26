#!/usr/bin/env python3
"""Analyze the development screen for PV2F policy feature minimization."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


BLOCK_NAMES = (
    "length",
    "moments_range_sign",
    "cardinality",
    "runs",
    "delta_order",
    "percentile",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["test_dataset"], int(row["column_index"]), int(row["seed"])


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    le = (
        left["dense_model_input_dim"] <= right["dense_model_input_dim"]
        and left["feature_ms"] <= right["feature_ms"]
        and left["total_bytes"] <= right["total_bytes"]
    )
    lt = (
        left["dense_model_input_dim"] < right["dense_model_input_dim"]
        or left["feature_ms"] < right["feature_ms"]
        or left["total_bytes"] < right["total_bytes"]
    )
    return le and lt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    screen_root = args.screen_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    variants: dict[str, dict[str, Any]] = {}
    evals: dict[str, list[dict[str, str]]] = {}
    for summary_path in sorted(screen_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = summary["variant"]
        variants[name] = summary
        evals[name] = read_csv(summary_path.with_name("evaluation.csv"))
    if "full60" not in variants:
        raise AssertionError("missing full60 baseline")
    baseline = variants["full60"]
    baseline_rows = {key(row): row for row in evals["full60"]}
    baseline_hits = float(baseline["median_acc_at_1pct_hits"])
    baseline_exact = float(baseline["median_byte_equivalent_oracle_hits"])
    columns = len(evals["full60"])

    comparison: list[dict[str, Any]] = []
    for name, summary in variants.items():
        rows = evals[name]
        penalties = []
        for row in rows:
            before = int(baseline_rows[key(row)]["encoded_bytes"])
            penalties.append(100.0 * (int(row["encoded_bytes"]) - before) / before)
        total_bytes = int(summary["median_total_bytes"])
        baseline_bytes = int(baseline["median_total_bytes"])
        acc_hits = float(summary["median_acc_at_1pct_hits"])
        exact_hits = float(summary["median_byte_equivalent_oracle_hits"])
        item = {
            "variant": name,
            "raw_group_mask": int(summary["raw_group_mask"]),
            "enabled_raw_blocks": "+".join(summary["enabled_raw_blocks"]) or "none",
            "raw_dimensions": len(summary["raw_dimensions"]),
            "prefix_mode": summary["prefix_mode"],
            "embedding_dim": int(summary["embedding_dim"]),
            "logical_observation_fields": int(summary["logical_observation_fields"]),
            "dense_model_input_dim": int(summary["dense_model_input_dim"]),
            "feature_ms": float(summary["feature_extraction_ms_20_columns_median"]),
            "feature_speedup_vs_full_pct": 100.0 * (float(baseline["feature_extraction_ms_20_columns_median"]) - float(summary["feature_extraction_ms_20_columns_median"])) / float(baseline["feature_extraction_ms_20_columns_median"]),
            "total_bytes": total_bytes,
            "aggregate_bytes_penalty_vs_full_pct": 100.0 * (total_bytes - baseline_bytes) / baseline_bytes,
            "median_column_penalty_vs_full_pct": statistics.median(penalties),
            "max_column_penalty_vs_full_pct": max(penalties),
            "acc_at_1pct_hits": acc_hits,
            "acc_at_1pct_delta_pp_vs_full": 100.0 * (acc_hits - baseline_hits) / columns,
            "byte_equivalent_hits": exact_hits,
            "byte_equivalent_delta_pp_vs_full": 100.0 * (exact_hits - baseline_exact) / columns,
            "oracle_regret_pct": float(summary["median_oracle_regret_pct"]),
            "screen_gate": "PASS" if (
                100.0 * (total_bytes - baseline_bytes) / baseline_bytes <= 2.0
                and max(penalties) <= 10.0
                and 100.0 * (acc_hits - baseline_hits) / columns >= -7.15
            ) else "FAIL",
        }
        comparison.append(item)
    comparison.sort(key=lambda row: (row["dense_model_input_dim"], row["total_bytes"], row["variant"]))
    write_csv(output_dir / "screen_comparison.csv", comparison)

    pareto = [
        row for row in comparison
        if not any(dominates(other, row) for other in comparison if other is not row)
    ]
    write_csv(output_dir / "pareto_frontier.csv", pareto)

    by_mask = {
        int(row["raw_group_mask"]): row
        for row in comparison
        if row["variant"].startswith("rawmask_") and row["prefix_mode"] == "embedding" and row["embedding_dim"] == 1
    }
    if set(by_mask) != set(range(64)):
        raise AssertionError(f"expected masks 0..63, got {sorted(by_mask)}")
    marginal_rows: list[dict[str, Any]] = []
    for mask in range(64):
        without = by_mask[mask]
        for bit, block in enumerate(BLOCK_NAMES):
            if mask & (1 << bit):
                continue
            with_block = by_mask[mask | (1 << bit)]
            time_increase = 100.0 * (with_block["feature_ms"] - without["feature_ms"]) / max(without["feature_ms"], 1.0e-9)
            acc_gain = with_block["acc_at_1pct_delta_pp_vs_full"] - without["acc_at_1pct_delta_pp_vs_full"]
            exact_gain = with_block["byte_equivalent_delta_pp_vs_full"] - without["byte_equivalent_delta_pp_vs_full"]
            bytes_improvement = 100.0 * (without["total_bytes"] - with_block["total_bytes"]) / without["total_bytes"]
            marginal_rows.append(
                {
                    "without_mask": mask,
                    "with_mask": mask | (1 << bit),
                    "added_block": block,
                    "added_raw_dimensions": with_block["raw_dimensions"] - without["raw_dimensions"],
                    "feature_time_increase_pct": time_increase,
                    "acc_at_1pct_gain_pp": acc_gain,
                    "byte_equivalent_gain_pp": exact_gain,
                    "total_bytes_improvement_pct": bytes_improvement,
                    "delete_by_user_5pct_rule": int(time_increase >= 5.0 and acc_gain < 5.0 and exact_gain < 5.0),
                }
            )
    write_csv(output_dir / "raw_block_marginals.csv", marginal_rows)

    shortlist = [row for row in comparison if row["screen_gate"] == "PASS"]
    shortlist.sort(key=lambda row: (row["dense_model_input_dim"], row["aggregate_bytes_penalty_vs_full_pct"], row["feature_ms"]))
    # Preserve both the smallest candidates and the strongest byte candidates.
    chosen: list[dict[str, Any]] = []
    for row in shortlist[:8] + sorted(shortlist, key=lambda item: item["total_bytes"])[:4]:
        if row["variant"] not in {item["variant"] for item in chosen}:
            chosen.append(row)
    if not any(row["variant"] == "full60" for row in chosen):
        chosen.append(next(row for row in comparison if row["variant"] == "full60"))
    (output_dir / "shortlist.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# PV2F PPO 特征联合筛选报告",
        "",
        f"- 完成设计点：{len(comparison)}",
        f"- Raw 计算块联合空间：{len(by_mask)}/64",
        f"- Full60：{int(baseline['median_total_bytes']):,} B，Acc@1% {int(baseline_hits)}/{columns}",
        f"- 单 seed 开发筛选通过宽松门槛：{sum(row['screen_gate'] == 'PASS' for row in comparison)}/{len(comparison)}",
        "- 本页只用于晋级，不是最终删除结论。最终门槛仍需 3/10 paired seeds 和 Metro lockbox。",
        "",
        "## 最小晋级候选",
        "",
        "| variant | dim | raw dim | feature ms | bytes | delta vs Full | Acc@1% | max column delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in chosen:
        report.append(
            f"| {row['variant']} | {row['dense_model_input_dim']} | {row['raw_dimensions']} | {row['feature_ms']:.3f} | {row['total_bytes']:,} | {row['aggregate_bytes_penalty_vs_full_pct']:+.3f}% | {int(row['acc_at_1pct_hits'])}/{columns} | {row['max_column_penalty_vs_full_pct']:+.3f}% |"
        )
    report.extend(
        [
            "",
            "## 边界",
            "",
            "筛选使用 Air-Quality leave-one-column-out 和冻结物理字节表。合法 mask、grammar、PV2F 字节均未改变。单 seed 结果不能用于最终主张。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(comparison), "pareto": len(pareto), "shortlist": [row["variant"] for row in chosen]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
