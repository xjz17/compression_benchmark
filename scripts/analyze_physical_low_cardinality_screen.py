#!/usr/bin/env python3
"""Select low-cardinality raw-feature candidates for multi-seed confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[int, str, int]:
    return int(row["seed"]), row["test_dataset"], int(row["column_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base_summary = json.loads((args.baseline_result / "summary.json").read_text(encoding="utf-8"))
    base_rows = {key(row): row for row in read_csv(args.baseline_result / "evaluation.csv")}
    base_total = int(base_summary["median_total_bytes"])
    base_hits = int(base_summary["median_acc_at_1pct_hits"])
    columns = len(base_rows)

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(args.results_root.resolve().glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation = read_csv(summary_path.with_name("evaluation.csv"))
        penalties = []
        for row in evaluation:
            before = int(base_rows[key(row)]["encoded_bytes"])
            penalties.append(100.0 * (int(row["encoded_bytes"]) - before) / before)
        total = int(summary["median_total_bytes"])
        hits = int(summary["median_acc_at_1pct_hits"])
        aggregate = 100.0 * (total - base_total) / base_total
        acc_delta = 100.0 * (hits - base_hits) / columns
        loose = aggregate <= 2.0 and max(penalties) <= 10.0 and acc_delta >= -7.15
        rows.append(
            {
                "variant": summary["variant"],
                "raw_dimensions": " ".join(str(dim) for dim in summary["raw_dimensions"]),
                "raw_dimension_count": len(summary["raw_dimensions"]),
                "dense_model_input_dim": int(summary["dense_model_input_dim"]),
                "feature_ms": float(summary["feature_extraction_ms_20_columns_median"]),
                "total_bytes": total,
                "aggregate_penalty_vs_full_pct": aggregate,
                "acc_at_1pct_hits": hits,
                "acc_at_1pct_delta_pp": acc_delta,
                "max_column_penalty_vs_full_pct": max(penalties),
                "loose_screen_gate": "PASS" if loose else "FAIL",
            }
        )
    rows.sort(key=lambda row: (row["dense_model_input_dim"], row["aggregate_penalty_vs_full_pct"], row["variant"]))
    write_csv(output / "comparison.csv", rows)
    shortlist = [row for row in rows if row["loose_screen_gate"] == "PASS"]
    (output / "shortlist.json").write_text(json.dumps(shortlist, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "# 低基数 raw 特征筛选",
        "",
        f"- 完成：{len(rows)}/136",
        f"- 宽松筛选通过：{len(shortlist)}",
        "- 仅用于晋级 3-seed，不能用于最终删除结论。",
        "",
        "| variant | raw dims | dim | bytes delta | Acc@1% | max column |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in shortlist:
        report.append(
            f"| {row['variant']} | {row['raw_dimensions']} | {row['dense_model_input_dim']} | {row['aggregate_penalty_vs_full_pct']:+.3f}% | {row['acc_at_1pct_hits']}/{columns} | {row['max_column_penalty_vs_full_pct']:+.3f}% |"
        )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(rows), "shortlist": len(shortlist)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
