#!/usr/bin/env python3
"""Audit exact depth-5 fixed-tail and full-search JSON summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct_gain(reference: int, candidate: int) -> float:
    return 100.0 * (reference - candidate) / reference


def load_fixed_tail(raw_root: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    inputs = []
    paths = sorted((raw_root / "fixed_tail").glob("*/*/summary.json"))
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        best = summary["best_by_depth"]
        row = {
            "dataset": summary["dataset"],
            "column_index": int(summary["column_index"]),
            "column_name": summary["column_name"],
            "sample_values": int(summary["sample_values"]),
            "original_bytes": int(summary["original_bytes"]),
            "input_sha256": summary["input_sha256"],
            "depth1_bytes": int(best["1"]["encoded_bytes"]),
            "depth1_pipeline": best["1"]["pipeline_text"],
            "depth2_bytes": int(best["2"]["encoded_bytes"]),
            "depth2_pipeline": best["2"]["pipeline_text"],
            "depth2_gain_vs_depth1_pct": pct_gain(
                int(best["1"]["encoded_bytes"]), int(best["2"]["encoded_bytes"])
            ),
            "depth3_bytes": int(summary["exact_best_through_depth3"]["encoded_bytes"]),
            "depth3_pipeline": summary["exact_best_through_depth3"]["pipeline_text"],
            "fixed_tail_bytes": int(summary["fixed_tail_best"]["encoded_bytes"]),
            "fixed_tail_pipeline": summary["fixed_tail_best"]["pipeline_text"],
            "fixed_tail_gain_vs_depth3_pct": float(
                summary["fixed_tail_gain_vs_exact_depth3_pct"]
            ),
            "fixed_tail_beats_depth3": int(summary["beats_exact_depth3"]),
            "preconditioner_tail_bytes": int(
                summary["preconditioner_prefix_best"]["encoded_bytes"]
            ),
            "preconditioner_tail_pipeline": summary["preconditioner_prefix_best"][
                "pipeline_text"
            ],
            "preconditioner_tail_beats_depth3": int(
                summary["preconditioner_prefix_beats_exact_depth3"]
            ),
            "depth3_evaluations": int(summary["depth3_evaluations"]),
            "fixed_tail_evaluations": int(summary["fixed_tail_evaluations"]),
            "failures": int(summary["failures"]),
            "verified_candidates_lossless": int(
                bool(summary.get("verified_candidates"))
                and all(int(item.get("lossless", 0)) == 1 for item in summary["verified_candidates"])
            ),
            "wall_seconds": float(summary["wall_seconds"]),
        }
        rows.append(row)
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    rows.sort(key=lambda row: (row["dataset"], row["column_index"]))
    return rows, inputs


def load_full(raw_root: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    inputs = []
    for path in sorted((raw_root / "full").glob("*/*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": summary["dataset"],
                "column_index": int(summary["column_index"]),
                "column_name": summary["column_name"],
                "complete_canonical_space": int(summary["complete_canonical_space"]),
                "observed_evaluations": int(summary["observed_evaluations"]),
                "failures": int(summary["failures"]),
                "depth3_bytes": int(summary["best_through_depth3"]["encoded_bytes"]),
                "depth3_pipeline": summary["best_through_depth3"]["pipeline_text"],
                "depth5_bytes": int(summary["best_through_depth5"]["encoded_bytes"]),
                "depth5_pipeline": summary["best_through_depth5"]["pipeline_text"],
                "gain_depth5_vs_depth3_pct": float(
                    summary["gain_exact_depth5_vs_exact_depth3_pct"]
                ),
                "wall_seconds": float(summary["wall_seconds_current_invocation"]),
            }
        )
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    rows.sort(key=lambda row: (row["dataset"], row["column_index"]))
    return rows, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-columns", type=int, default=20)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    fixed, fixed_inputs = load_fixed_tail(raw_root)
    if not fixed:
        raise ValueError("no fixed-tail summaries")
    full, full_inputs = load_full(raw_root)
    write_csv(output_dir / "fixed_tail_per_column.csv", fixed)
    if full:
        write_csv(output_dir / "full_depth5_per_column.csv", full)

    exact3_total = sum(row["depth3_bytes"] for row in fixed)
    fixed_total = sum(row["fixed_tail_bytes"] for row in fixed)
    depth1_total = sum(row["depth1_bytes"] for row in fixed)
    depth2_total = sum(row["depth2_bytes"] for row in fixed)
    all_counts_ok = all(
        row["depth3_evaluations"] == 22_764
        and row["fixed_tail_evaluations"] == 784
        and row["failures"] == 0
        and row["verified_candidates_lossless"] == 1
        for row in fixed
    )
    audit = {
        "created_local": datetime.now().astimezone().isoformat(),
        "raw_root": str(raw_root),
        "fixed_tail_columns": len(fixed),
        "expected_columns": args.expected_columns,
        "fixed_tail_complete": len(fixed) == args.expected_columns and all_counts_ok,
        "all_selected_candidates_lossless": all(
            row["verified_candidates_lossless"] == 1 for row in fixed
        ),
        "fixed_tail_wins_vs_exact_depth3": sum(row["fixed_tail_beats_depth3"] for row in fixed),
        "preconditioner_tail_wins_vs_exact_depth3": sum(
            row["preconditioner_tail_beats_depth3"] for row in fixed
        ),
        "depth2_wins_vs_depth1": sum(row["depth2_bytes"] < row["depth1_bytes"] for row in fixed),
        "aggregate_depth1_bytes": depth1_total,
        "aggregate_depth2_bytes": depth2_total,
        "aggregate_depth2_gain_vs_depth1_pct": pct_gain(depth1_total, depth2_total),
        "aggregate_exact_depth3_bytes": exact3_total,
        "aggregate_fixed_tail_bytes": fixed_total,
        "aggregate_fixed_tail_gain_vs_depth3_pct": pct_gain(exact3_total, fixed_total),
        "median_fixed_tail_gain_vs_depth3_pct": statistics.median(
            row["fixed_tail_gain_vs_depth3_pct"] for row in fixed
        ),
        "full_columns": len(full),
        "full_complete_columns": sum(row["complete_canonical_space"] for row in full),
        "full_depth5_wins": sum(row["depth5_bytes"] < row["depth3_bytes"] for row in full),
        "inputs": fixed_inputs + full_inputs,
        "claim_boundary": (
            "Standalone normalized-int64 Python proxy codec. zlib is already the terminal "
            "LZ77/DEFLATE/framing backend; explicit lz/huffman/container actions double-model "
            "that terminal stack and are not production implementations."
        ),
    }
    (output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    status = "完整" if audit["fixed_tail_complete"] else "部分"
    full_line = "完整 depth≤5 尚无完成列。"
    if full:
        full_line = (
            f"完整 depth≤5 已完成 {audit['full_complete_columns']}/{len(full)} 列，"
            f"其中 {audit['full_depth5_wins']} 列严格优于 exact depth≤3。"
        )
    report = f"""# Depth-5 精确枚举审计报告

状态：{status}，固定尾部完成 {len(fixed)}/{args.expected_columns} 列。

## 结论

- `a -> b -> lz_match -> entropy_huffman -> container_frame` 固定五段族为 {audit['fixed_tail_wins_vs_exact_depth3']}/{len(fixed)} 列严格优于 exact depth≤3。
- 只限定前两步为 preconditioner 时为 {audit['preconditioner_tail_wins_vs_exact_depth3']}/{len(fixed)} 列严格优于 exact depth≤3。
- 两个显式算子后直接进入既有 zlib terminal backend，则有 {audit['depth2_wins_vs_depth1']}/{len(fixed)} 列优于最佳单算子；聚合少 {audit['aggregate_depth2_gain_vs_depth1_pct']:.4f}% 字节。
- 固定五段相对 exact depth≤3 的聚合收益为 {audit['aggregate_fixed_tail_gain_vs_depth3_pct']:.4f}%（负数表示更多字节）。
- {full_line}

## 解释边界

当前候选最终都会进入 zlib，后者已经执行 LZ77/DEFLATE 和 framing。因此“两个显式编码算子 + zlib backend”在物理阶段上已经包含用户提出的匹配、熵编码和容器步骤；RL depth 只计算可选择的显式动作。

显式再串接当前 `lz_match`、`entropy_huffman`、`container_frame` 会重复模拟 terminal stack。并且这些算子只是概念代理：`lz_match` 只识别相邻相同整数，`entropy_huffman` 实际写入 frequency-ranked dictionary，并非真实 Huffman bitstream。固定五段结果不能外推到真实 LZ77+Huffman codec。

## 审计

- 每列 exact depth≤3：22,764 条。
- 每列固定五段：784 条 ordered prefix。
- 计数与零失败检查：{'通过' if all_counts_ok else '未通过'}。
- 机器审计与输入哈希：`AUDIT.json`。
"""
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    (output_dir / f"REPORT_{stamp}.md").write_text(report, encoding="utf-8")
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
