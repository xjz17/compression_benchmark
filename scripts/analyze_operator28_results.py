#!/usr/bin/env python3
"""Merge the four 28-operator search lanes and audit the depth-3 result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


LANES = ("air_a", "air_b", "metro_a", "metro_b")
EXPECTED_COLUMNS = {"Air-Quality": 14, "Metro-Traffic": 6}
EXPECTED_DEPTH_COUNTS = {1: 28, 2: 784, 3: 21_952}
TIMESTAMP_NAMES = {"timestamp", "date_time"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_columns(repo_root: Path, dataset: str) -> list[dict[str, str]]:
    path = repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv"
    rows = read_csv(path)
    return [
        row
        for row in rows
        if row["logical_type"].strip().lower() in {"integer", "int", "long", "float", "double"}
    ]


def report_dtype(row: dict[str, str]) -> str:
    logical = row["logical_type"].strip().lower()
    return "float" if logical in {"float", "double"} else "int"


def aggregate_baselines(
    repo_root: Path,
    dataset: str,
    expected_original: dict[str, int],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    root = repo_root / "backend" / "dataset" / dataset
    by_dtype: dict[str, dict[str, tuple[float, int, int, str]]] = {}
    best_rows: dict[str, dict[str, object]] = {}
    for dtype in ("int", "float"):
        by_algo: dict[str, tuple[float, int, int, str]] = {}
        for path in root.glob(f"{dataset}_{dtype}_*_compression_result.csv"):
            if "_tsfile_" in path.name:
                continue
            language = path.name[len(f"{dataset}_{dtype}_") : -len("_compression_result.csv")]
            for row in read_csv(path):
                try:
                    index = int(row.get("column_index", "-2"))
                    original = int(float(row["originalSize"]))
                    compressed = int(float(row["compressedSize"]))
                    lossy = int(float(row.get("isLossy", 0) or 0))
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    index != -1
                    or lossy
                    or original != expected_original[dtype]
                    or compressed <= 0
                ):
                    continue
                algorithm = row["algorithm"]
                candidate = (compressed / original, original, compressed, language)
                if algorithm not in by_algo or candidate[0] < by_algo[algorithm][0]:
                    by_algo[algorithm] = candidate
        if not by_algo:
            raise RuntimeError(f"No aligned aggregate baseline for {dataset}/{dtype}")
        by_dtype[dtype] = by_algo
        algorithm, value = min(by_algo.items(), key=lambda item: (item[1][0], item[0]))
        best_rows[dtype] = {
            "algorithm": algorithm,
            "rate": value[0],
            "original": value[1],
            "compressed": value[2],
            "language": value[3],
        }

    fixed: list[tuple[float, str, int, int]] = []
    for algorithm in set(by_dtype["int"]) & set(by_dtype["float"]):
        int_row = by_dtype["int"][algorithm]
        float_row = by_dtype["float"][algorithm]
        original = int_row[1] + float_row[1]
        compressed = int_row[2] + float_row[2]
        fixed.append((compressed / original, algorithm, original, compressed))
    if not fixed:
        raise RuntimeError(f"No fixed algorithm covers both dtypes for {dataset}")
    rate, algorithm, original, compressed = min(fixed)
    return best_rows, {
        "algorithm": algorithm,
        "rate": rate,
        "original": original,
        "compressed": compressed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    summaries: list[dict[str, str]] = []
    depth3_best: dict[tuple[str, int], dict[str, object]] = {}
    audit_columns: dict[tuple[str, int], dict[str, object]] = {}
    for lane in LANES:
        lane_dir = results_root / lane
        summaries.extend(read_csv(lane_dir / "summary.csv"))
        counts: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
        seen: dict[tuple[str, int], set[str]] = defaultdict(set)
        all_lossless: dict[tuple[str, int], bool] = defaultdict(lambda: True)
        for row in read_csv(lane_dir / "pipeline_measurements.csv"):
            key = (row["dataset"], int(row["column_index"]))
            depth = int(row["depth"])
            pipeline = row["pipeline"]
            counts[key][depth] += 1
            duplicate = pipeline in seen[key]
            seen[key].add(pipeline)
            all_lossless[key] &= row["lossless"] == "1" and row["status"] == "ok"
            if duplicate:
                all_lossless[key] = False
            if depth == 3:
                rate = float(row["rate"])
                current = depth3_best.get(key)
                if current is None or (rate, pipeline) < (float(current["rate"]), str(current["pipeline"])):
                    depth3_best[key] = {
                        "dataset": key[0],
                        "column_index": key[1],
                        "column_name": row["column_name"],
                        "pipeline": pipeline,
                        "rate": rate,
                    }
        for key in counts:
            audit_columns[key] = {
                "counts": dict(counts[key]),
                "unique": len(seen[key]),
                "all_lossless": all_lossless[key],
            }

    summaries.sort(key=lambda row: (row["dataset"], int(row["column_index"])))
    expected_total = sum(EXPECTED_COLUMNS.values())
    if len(summaries) != expected_total:
        raise RuntimeError(f"Expected {expected_total} summary rows, found {len(summaries)}")
    if len({(row["dataset"], int(row["column_index"])) for row in summaries}) != expected_total:
        raise RuntimeError("Duplicate or missing summary column keys")
    for key, evidence in audit_columns.items():
        if evidence["counts"] != EXPECTED_DEPTH_COUNTS:
            raise RuntimeError(f"Bad depth counts for {key}: {evidence['counts']}")
        if evidence["unique"] != sum(EXPECTED_DEPTH_COUNTS.values()):
            raise RuntimeError(f"Duplicate pipelines for {key}")
        if not evidence["all_lossless"]:
            raise RuntimeError(f"Non-lossless or failed pipeline for {key}")

    summary_by_key = {(row["dataset"], int(row["column_index"])): row for row in summaries}
    depth3_rows: list[dict[str, object]] = []
    operator_counts = Counter()
    position_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for key in sorted(depth3_best):
        row = depth3_best[key]
        summary = summary_by_key[key]
        rate = float(row["rate"])
        oracle_rate = float(summary["oracle_rate"])
        single_rate = float(summary["best_single_rate"])
        pipeline = str(row["pipeline"])
        operators = pipeline.split(" -> ")
        operator_counts.update(operators)
        for position, operator in enumerate(operators, start=1):
            position_counts[position][operator] += 1
        depth3_rows.append(
            {
                **row,
                "sample_values": int(summary["sample_values"]),
                "oracle_rate_depth_le_3": oracle_rate,
                "best_single_rate": single_rate,
                "depth3_gain_vs_single_pct": 100.0 * (single_rate - rate) / single_rate,
                "depth3_regret_vs_depth_le_3_pct": 100.0 * (rate - oracle_rate) / oracle_rate,
                "beam4_regret_pct": float(summary["beam4_regret_pct"]),
                "exhaustive_ms": float(summary["exhaustive_ms"]),
            }
        )

    comparisons: list[dict[str, object]] = []
    for dataset in EXPECTED_COLUMNS:
        columns = numeric_columns(repo_root, dataset)
        by_index = {int(row["column_index"]): row for row in columns}
        non_timestamp = {
            idx: row
            for idx, row in by_index.items()
            if row["column_name"].strip().lower() not in TIMESTAMP_NAMES
        }
        sample_values = {
            int(row["column_index"]): int(row["sample_values"])
            for row in summaries
            if row["dataset"] == dataset
        }
        expected_original = {
            dtype: sum(sample_values[idx] * 8 for idx, row in non_timestamp.items() if report_dtype(row) == dtype)
            for dtype in ("int", "float")
        }
        best_dtype, best_fixed = aggregate_baselines(repo_root, dataset, expected_original)
        candidate_compressed: dict[str, float] = defaultdict(float)
        candidate_original: dict[str, int] = defaultdict(int)
        for idx, column in non_timestamp.items():
            dtype = report_dtype(column)
            original = sample_values[idx] * 8
            candidate_original[dtype] += original
            candidate_compressed[dtype] += float(depth3_best[(dataset, idx)]["rate"]) * original
        candidate_total_original = sum(candidate_original.values())
        candidate_total_compressed = sum(candidate_compressed.values())
        mixed_baseline_original = sum(int(row["original"]) for row in best_dtype.values())
        mixed_baseline_compressed = sum(int(row["compressed"]) for row in best_dtype.values())
        if candidate_total_original != mixed_baseline_original:
            raise RuntimeError(
                f"Original-size mismatch for {dataset}: candidate={candidate_total_original}, baseline={mixed_baseline_original}"
            )
        for dtype in ("int", "float"):
            dtype_candidate_rate = candidate_compressed[dtype] / candidate_original[dtype]
            dtype_baseline_rate = float(best_dtype[dtype]["rate"])
            comparisons.append(
                {
                    "dataset": dataset,
                    "scope": f"{dtype}_non_timestamp",
                    "candidate_depth3_rate": dtype_candidate_rate,
                    "best_dtype_mix_rate": dtype_baseline_rate,
                    "best_dtype_mix_algorithms": str(best_dtype[dtype]["algorithm"]),
                    "gain_vs_best_dtype_mix_pct": 100.0
                    * (dtype_baseline_rate - dtype_candidate_rate)
                    / dtype_baseline_rate,
                    "best_fixed_algorithm": str(best_dtype[dtype]["algorithm"]),
                    "best_fixed_rate": dtype_baseline_rate,
                    "gain_vs_best_fixed_pct": 100.0
                    * (dtype_baseline_rate - dtype_candidate_rate)
                    / dtype_baseline_rate,
                    "original_bytes": candidate_original[dtype],
                    "candidate_compressed_bytes_est": candidate_compressed[dtype],
                }
            )
        candidate_rate = candidate_total_compressed / candidate_total_original
        mixed_rate = mixed_baseline_compressed / mixed_baseline_original
        comparisons.append(
            {
                "dataset": dataset,
                "scope": "non_timestamp_numeric",
                "candidate_depth3_rate": candidate_rate,
                "best_dtype_mix_rate": mixed_rate,
                "best_dtype_mix_algorithms": "; ".join(
                    f"{dtype}:{best_dtype[dtype]['algorithm']}" for dtype in ("int", "float")
                ),
                "gain_vs_best_dtype_mix_pct": 100.0 * (mixed_rate - candidate_rate) / mixed_rate,
                "best_fixed_algorithm": best_fixed["algorithm"],
                "best_fixed_rate": best_fixed["rate"],
                "gain_vs_best_fixed_pct": 100.0 * (float(best_fixed["rate"]) - candidate_rate) / float(best_fixed["rate"]),
                "original_bytes": candidate_total_original,
                "candidate_compressed_bytes_est": candidate_total_compressed,
            }
        )

    max_beam4 = max(float(row["beam4_regret_pct"]) for row in summaries)
    median_beam4 = statistics.median(float(row["beam4_regret_pct"]) for row in summaries)
    max_gain = max(float(row["depth3_gain_vs_single_pct"]) for row in depth3_rows)
    large_gain_columns = sum(float(row["depth3_gain_vs_single_pct"]) > 3.0 for row in depth3_rows)
    rl_decision = "no_rl_at_depth3" if max_beam4 < 1.0 else "larger_search_then_rl_candidate"

    write_csv(output_dir / "combined_summary.csv", summaries, list(summaries[0]))
    write_csv(output_dir / "depth3_best.csv", depth3_rows, list(depth3_rows[0]))
    write_csv(output_dir / "dataset_comparison.csv", comparisons, list(comparisons[0]))
    frequency_rows = [
        {
            "operator": operator,
            "total_best_depth3_columns": count,
            "position1": position_counts[1][operator],
            "position2": position_counts[2][operator],
            "position3": position_counts[3][operator],
        }
        for operator, count in operator_counts.most_common()
    ]
    write_csv(
        output_dir / "operator_frequency.csv",
        frequency_rows,
        ["operator", "total_best_depth3_columns", "position1", "position2", "position3"],
    )
    evidence_files = [
        results_root / lane / filename
        for lane in LANES
        for filename in ("summary.csv", "pipeline_measurements.csv", "manifest.json")
    ]
    evidence_files.extend(
        path
        for dataset in EXPECTED_COLUMNS
        for dtype in ("int", "float")
        for path in (repo_root / "backend" / "dataset" / dataset).glob(
            f"{dataset}_{dtype}_*_compression_result.csv"
        )
        if "_tsfile_" not in path.name
    )
    evidence_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(evidence_files)
    }
    audit = {
        "status": "pass",
        "evaluation_type": "real_data_proxy_codec",
        "columns": len(summaries),
        "measurements": len(summaries) * sum(EXPECTED_DEPTH_COUNTS.values()),
        "depth3_measurements": len(summaries) * EXPECTED_DEPTH_COUNTS[3],
        "expected_depth_counts_per_column": EXPECTED_DEPTH_COUNTS,
        "all_lossless": True,
        "unique_pipeline_keys": True,
        "baseline_scope": "aggregate non-timestamp numeric columns with exact originalSize match",
        "per_column_baseline_scope": "excluded because old and current column indices are mixed",
        "review_independence": "self-audit only",
        "acceptance_status": "provisional",
        "evidence_sha256": evidence_hashes,
        "rl_decision": rl_decision,
        "max_beam4_regret_pct": max_beam4,
        "median_beam4_regret_pct": median_beam4,
        "max_depth3_gain_vs_single_pct": max_gain,
        "columns_depth3_gain_over_3pct": large_gain_columns,
    }
    (output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    comparison_lines = "\n".join(
        f"| {row['dataset']} / {row['scope']} | {float(row['candidate_depth3_rate']):.6f} | {float(row['best_dtype_mix_rate']):.6f} | {float(row['gain_vs_best_dtype_mix_pct']):+.2f}% | {row['best_fixed_algorithm']} | {float(row['best_fixed_rate']):.6f} |"
        for row in comparisons
    )
    report = f"""# 28 算子三层穷举结果

## 审计结论

- 数值列：{len(summaries)}。
- 每列组合：深度 1 为 28，深度 2 为 784，深度 3 为 21,952。
- 总测量：{audit['measurements']:,}，其中三层组合 {audit['depth3_measurements']:,}。
- 可逆性：全部通过；组合键无重复。
- Baseline：只使用 originalSize 与当前非时间戳数值列严格匹配的聚合无损结果。旧报告的逐列索引混有清洗前后 schema，未用于直接比较。

## 与现有最佳完整算法比较

| Dataset / scope | depth-3 per-column oracle | production baseline | relative gain | algorithm | baseline rate |
|---|---:|---:|---:|---|---:|
{comparison_lines}

## RL 必要性

- Beam-4 最大 oracle regret：{max_beam4:.6f}%。
- Beam-4 中位 oracle regret：{median_beam4:.6f}%。
- 三层组合相对最佳单算子的最大改善：{max_gain:.2f}%。
- 改善超过 3% 的列：{large_gain_columns}/{len(summaries)}。
- 决策：{rl_decision}。

## 口径限制

候选流水线使用概念算子实现和 zlib level 6，现有算法是项目中的生产 codec 报告。两者原始分母已经对齐，但字节流格式、元数据和实现成熟度不同。浮点列还在搜索前按十进制精度转成整数，相当于免费使用 scaling。因此跨 codec 的数值是 falsification evidence，不是论文级最终 benchmark。独立 reviewer 未启用，语义结论为 provisional。
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
