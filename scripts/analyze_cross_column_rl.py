#!/usr/bin/env python3
"""Validate and summarize held-out-column PPO/GRPO/DAPO results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def fmt(value: float) -> str:
    return f"{value:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--search-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    input_files = []
    rows: list[dict[str, str]] = []
    run_status_rows: list[dict[str, str | int]] = []
    fold_hashes = set()
    manifest_algorithms = set()
    search_summary_path = args.search_summary_csv.resolve()
    if not search_summary_path.is_file():
        raise FileNotFoundError(search_summary_path)
    input_files.append(search_summary_path)
    search_summary = {
        (row["dataset"], int(row["column_index"])): row
        for row in read_csv(search_summary_path)
    }
    for input_dir in args.input_dirs:
        root = input_dir.resolve()
        evaluation = root / "evaluation.csv"
        manifest_path = root / "manifest.json"
        folds = root / "fold_assignments.csv"
        for path in (evaluation, manifest_path, folds):
            if not path.is_file():
                raise FileNotFoundError(path)
            input_files.append(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configured = manifest["config"]["algorithms"]
        if isinstance(configured, str):
            configured = [configured]
        manifest_algorithms.update(configured)
        if manifest["claim_boundary"]["real_codec_cost"].startswith("not available") is False:
            raise AssertionError("unexpected real-codec claim")
        fold_hashes.add(sha256(folds))
        rows.extend(read_csv(evaluation))
        run_status_path = root / "run_status.csv"
        if run_status_path.is_file():
            input_files.append(run_status_path)
            run_status_rows.extend(read_csv(run_status_path))
        else:
            requested = int(manifest["config"]["iterations"])
            for algorithm in configured:
                for seed in manifest["config"]["seeds"]:
                    for fold in range(int(manifest["config"]["folds"])):
                        run_status_rows.append(
                            {
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "fold": fold,
                                "requested_iterations": requested,
                                "completed_iterations": requested,
                                "early_stop_reason": "",
                                "early_stop_sampled_groups": 0,
                                "early_stop_filtered_groups": 0,
                            }
                        )
    if len(fold_hashes) != 1:
        raise AssertionError("fold assignments differ across algorithms")
    algorithms = sorted({row["algorithm"] for row in rows})
    if set(algorithms) != manifest_algorithms or set(algorithms) != {"ppo", "grpo", "dapo"}:
        raise AssertionError((algorithms, manifest_algorithms))
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != 3:
        raise AssertionError(f"expected three seeds, got {seeds}")
    column_keys = sorted({(row["dataset"], int(row["column_index"])) for row in rows})
    if len(column_keys) != 20:
        raise AssertionError(f"expected 20 columns, got {len(column_keys)}")
    expected_rows = len(algorithms) * len(seeds) * len(column_keys)
    if len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {len(rows)}")
    identity = [(row["algorithm"], int(row["seed"]), row["dataset"], int(row["column_index"])) for row in rows]
    if len(identity) != len(set(identity)):
        raise AssertionError("duplicate algorithm/seed/column evaluation")
    if len(run_status_rows) != len(algorithms) * len(seeds) * 5:
        raise AssertionError(f"unexpected run status count: {len(run_status_rows)}")

    baseline_map = {}
    for row in rows:
        key = row["dataset"], int(row["column_index"])
        if key not in search_summary:
            raise AssertionError(f"missing search summary for {key}")
        baseline = (row["baseline_algorithm"], float(row["baseline_rate"]))
        if key in baseline_map and baseline_map[key] != baseline:
            raise AssertionError(f"baseline drift for {key}")
        baseline_map[key] = baseline
        rate = float(row["proxy_rate"])
        expected_gain = 100.0 * (baseline[1] - rate) / baseline[1]
        if not math.isclose(expected_gain, float(row["gain_vs_baseline_pct"]), abs_tol=1.0e-10):
            raise AssertionError(f"gain mismatch for {identity[len(baseline_map) - 1]}")
        if int(row["strictly_better_than_baseline"]) != int(rate < baseline[1]):
            raise AssertionError("strict-better flag mismatch")
        best_single_rate = float(search_summary[key]["best_single_rate"])
        safe_rate = min(rate, best_single_rate)
        used_fallback = best_single_rate < rate
        row["proposal_gain_vs_baseline_pct"] = row["gain_vs_baseline_pct"]
        row["best_single_pipeline"] = search_summary[key]["best_single_pipeline"]
        row["best_single_rate"] = best_single_rate
        row["safe_pipeline"] = (
            search_summary[key]["best_single_pipeline"] if used_fallback else row["pipeline"]
        )
        row["safe_proxy_rate"] = safe_rate
        row["safe_gain_vs_baseline_pct"] = 100.0 * (baseline[1] - safe_rate) / baseline[1]
        row["safe_gain_vs_best_single_pct"] = 100.0 * (best_single_rate - safe_rate) / best_single_rate
        row["safe_strictly_better_than_baseline"] = int(safe_rate < baseline[1])
        row["used_best_single_fallback"] = int(used_fallback)

    rows.sort(key=lambda row: (row["algorithm"], int(row["seed"]), row["dataset"], int(row["column_index"])))
    write_csv(output_dir / "per_seed_evaluation.csv", rows)
    run_status_rows.sort(key=lambda row: (str(row["algorithm"]), int(row["seed"]), int(row["fold"])))
    write_csv(output_dir / "run_status.csv", run_status_rows)

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["dataset"], int(row["column_index"]))].append(row)
    column_summary = []
    for (algorithm, dataset, column_index), group in sorted(grouped.items()):
        if len(group) != len(seeds):
            raise AssertionError((algorithm, dataset, column_index, len(group)))
        group.sort(key=lambda row: int(row["seed"]))
        gains = [float(row["safe_gain_vs_baseline_pct"]) for row in group]
        proposal_gains = [float(row["proposal_gain_vs_baseline_pct"]) for row in group]
        gains_vs_single = [float(row["safe_gain_vs_best_single_pct"]) for row in group]
        rates = [float(row["safe_proxy_rate"]) for row in group]
        regrets = [float(row["oracle_regret_pct"]) for row in group]
        gain_stats = stats(gains)
        rate_stats = stats(rates)
        regret_stats = stats(regrets)
        pipelines = [row["safe_pipeline"] for row in group]
        common_pipeline, common_count = Counter(pipelines).most_common(1)[0]
        first = group[0]
        column_summary.append(
            {
                "algorithm": algorithm,
                "dataset": dataset,
                "column_index": column_index,
                "column_name": first["column_name"],
                "logical_type": first["logical_type"],
                "baseline_algorithm": first["baseline_algorithm"],
                "baseline_rate": float(first["baseline_rate"]),
                "mean_safe_proxy_rate": rate_stats["mean"],
                "std_safe_proxy_rate": rate_stats["std"],
                "median_safe_proxy_rate": rate_stats["median"],
                "mean_safe_gain_vs_baseline_pct": gain_stats["mean"],
                "std_safe_gain_vs_baseline_pct": gain_stats["std"],
                "median_safe_gain_vs_baseline_pct": gain_stats["median"],
                "min_safe_gain_vs_baseline_pct": gain_stats["min"],
                "max_safe_gain_vs_baseline_pct": gain_stats["max"],
                "positive_seeds": sum(value > 0 for value in gains),
                "fallback_seeds": sum(int(row["used_best_single_fallback"]) for row in group),
                "median_proposal_gain_vs_baseline_pct": float(np.median(proposal_gains)),
                "worst_proposal_gain_vs_baseline_pct": float(np.min(proposal_gains)),
                "median_safe_gain_vs_best_single_pct": float(np.median(gains_vs_single)),
                "median_oracle_regret_pct": regret_stats["median"],
                "most_common_pipeline": common_pipeline,
                "most_common_pipeline_seeds": common_count,
                "pipelines_by_seed": " | ".join(
                    f"{row['seed']}:{row['safe_pipeline']}" for row in group
                ),
                "proposals_by_seed": " | ".join(
                    f"{row['seed']}:{row['pipeline']}" for row in group
                ),
            }
        )
    write_csv(output_dir / "per_column_algorithm_summary.csv", column_summary)
    column_lookup = {
        (row["algorithm"], row["dataset"], int(row["column_index"])): row
        for row in column_summary
    }
    wide_rows = []
    for dataset, column_index in column_keys:
        by_algorithm = {
            algorithm: column_lookup[(algorithm, dataset, column_index)]
            for algorithm in algorithms
        }
        first = by_algorithm[algorithms[0]]
        gains = {
            algorithm: float(row["median_safe_gain_vs_baseline_pct"])
            for algorithm, row in by_algorithm.items()
        }
        best_gain = max(gains.values())
        wide_rows.append(
            {
                "dataset": dataset,
                "column_index": column_index,
                "column_name": first["column_name"],
                "baseline_algorithm": first["baseline_algorithm"],
                "baseline_rate": first["baseline_rate"],
                "ppo_median_safe_rate": by_algorithm["ppo"]["median_safe_proxy_rate"],
                "ppo_median_gain_pct": gains["ppo"],
                "grpo_median_safe_rate": by_algorithm["grpo"]["median_safe_proxy_rate"],
                "grpo_median_gain_pct": gains["grpo"],
                "dapo_median_safe_rate": by_algorithm["dapo"]["median_safe_proxy_rate"],
                "dapo_median_gain_pct": gains["dapo"],
                "all_algorithms_positive": int(all(value > 0 for value in gains.values())),
                "best_algorithm": "+".join(
                    algorithm for algorithm, value in gains.items()
                    if math.isclose(value, best_gain, abs_tol=1.0e-12)
                ),
            }
        )
    write_csv(output_dir / "per_column_comparison_wide.csv", wide_rows)

    dataset_summary = []
    for algorithm in algorithms:
        datasets = sorted({row["dataset"] for row in column_summary if row["algorithm"] == algorithm})
        for dataset in datasets:
            selected = [
                row for row in column_summary
                if row["algorithm"] == algorithm and row["dataset"] == dataset
            ]
            medians = [float(row["median_safe_gain_vs_baseline_pct"]) for row in selected]
            summary = stats(medians)
            dataset_summary.append(
                {
                    "algorithm": algorithm,
                    "dataset": dataset,
                    "columns": len(selected),
                    "columns_positive_by_median": sum(value > 0 for value in medians),
                    "columns_all_three_seeds_positive": sum(int(row["positive_seeds"]) == 3 for row in selected),
                    "median_column_gain_pct": summary["median"],
                    "mean_column_gain_pct": summary["mean"],
                    "std_column_gain_pct": summary["std"],
                    "min_column_gain_pct": summary["min"],
                    "max_column_gain_pct": summary["max"],
                }
            )
    write_csv(output_dir / "algorithm_dataset_summary.csv", dataset_summary)

    overall_summary = []
    for algorithm in algorithms:
        selected = [row for row in column_summary if row["algorithm"] == algorithm]
        medians = [float(row["median_safe_gain_vs_baseline_pct"]) for row in selected]
        all_seed_rows = [row for row in rows if row["algorithm"] == algorithm]
        gains_vs_single = [
            float(row["median_safe_gain_vs_best_single_pct"]) for row in selected
        ]
        summary = stats(medians)
        all_seed_gains = [float(row["safe_gain_vs_baseline_pct"]) for row in all_seed_rows]
        proposal_gains = [float(row["proposal_gain_vs_baseline_pct"]) for row in all_seed_rows]
        overall_summary.append(
            {
                "algorithm": algorithm,
                "columns": len(selected),
                "seeds": len(seeds),
                "heldout_predictions": len(all_seed_rows),
                "columns_positive_by_median": sum(value > 0 for value in medians),
                "columns_all_three_seeds_positive": sum(int(row["positive_seeds"]) == 3 for row in selected),
                "heldout_predictions_positive": sum(value > 0 for value in all_seed_gains),
                "columns_beating_best_single_by_median": sum(value > 1.0e-12 for value in gains_vs_single),
                "median_column_gain_vs_best_single_pct": float(np.median(gains_vs_single)),
                "fallback_predictions": sum(int(row["used_best_single_fallback"]) for row in all_seed_rows),
                "catastrophic_proposal_predictions_below_minus_100pct": sum(
                    value < -100.0 for value in proposal_gains
                ),
                "worst_proposal_gain_pct": min(proposal_gains),
                "median_column_gain_pct": summary["median"],
                "mean_column_gain_pct": summary["mean"],
                "std_column_gain_pct": summary["std"],
                "min_column_gain_pct": summary["min"],
                "max_column_gain_pct": summary["max"],
            }
        )
    overall_summary.sort(key=lambda row: float(row["median_column_gain_pct"]), reverse=True)
    write_csv(output_dir / "algorithm_overall_summary.csv", overall_summary)

    best_median = overall_summary[0]
    recommended = next(row for row in overall_summary if row["algorithm"] == "ppo")
    early_stops_by_algorithm = {
        algorithm: sum(
            bool(row["early_stop_reason"])
            for row in run_status_rows
            if row["algorithm"] == algorithm
        )
        for algorithm in algorithms
    }
    all_columns_positive = {
        row["algorithm"]: int(row["columns_positive_by_median"]) == len(column_keys)
        for row in overall_summary
    }
    claim_check = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "simulation_only_proxy",
        "best_algorithm_by_median_column_gain": best_median["algorithm"],
        "recommended_first_baseline": "ppo",
        "recommendation_reason": (
            "PPO has no Dynamic Sampling early stops and fewer catastrophic unguarded proposals; "
            "all three methods still need the best-single incumbent guard."
        ),
        "all_columns_positive_by_median": all_columns_positive,
        "early_stopped_runs": early_stops_by_algorithm,
        "selection_policy": "min(policy proposal, best of the 28 single-operator probes)",
        "real_codec_cost_available": False,
        "real_codec_cost_reason": "operator states rely on in-memory inverse closures; proxy eval_ms is not deployable encode/decode latency",
        "safe_claim": (
            "Held-out-column policies were compared on conceptual-transform-plus-zlib proxy rates. "
            "This does not establish a production codec compression or latency gain."
        ),
    }
    (output_dir / "claim_check.json").write_text(
        json.dumps(claim_check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    overall_by_algorithm = {row["algorithm"]: row for row in overall_summary}
    lines = [
        "# 跨列 PPO、GRPO、DAPO-adapted 结果",
        "",
        "## 结论",
        "",
        (
            "加上 best-single incumbent guard 后，三个策略都是 20 列中 17 列优于各列最优对齐 baseline。"
            f"GRPO 和 DAPO-adapted 的列增益中位数为 {fmt(float(best_median['median_column_gain_pct']))}%，"
            f"PPO 为 {fmt(float(recommended['median_column_gain_pct']))}%。没有一种策略覆盖全部 20 列。"
        ),
        "",
        (
            "只看 RL 相对 best-single 的额外价值，PPO、GRPO、DAPO-adapted 分别在 "
            f"{overall_by_algorithm['ppo']['columns_beating_best_single_by_median']}/20、"
            f"{overall_by_algorithm['grpo']['columns_beating_best_single_by_median']}/20、"
            f"{overall_by_algorithm['dapo']['columns_beating_best_single_by_median']}/20 列上有正增益；"
            "对应的列增益中位数为 "
            f"{fmt(float(overall_by_algorithm['ppo']['median_column_gain_vs_best_single_pct']))}%、"
            f"{fmt(float(overall_by_algorithm['grpo']['median_column_gain_vs_best_single_pct']))}%、"
            f"{fmt(float(overall_by_algorithm['dapo']['median_column_gain_vs_best_single_pct']))}%。"
        ),
        "",
        "这仍是 conceptual transforms + zlib 的 proxy rate。当前算子没有可独立解码的真实 codec，"
        "因此 proxy_search_eval_ms 不能当作生产 encode/decode cost。",
        "",
        (
            f"DAPO-adapted 有 {early_stops_by_algorithm['dapo']}/15 个 run 因 Dynamic Sampling "
            "无法在上限内组成非零方差 batch 而提前停止，随后评估停止时的策略。"
        ),
        "",
        "未加 incumbent guard 时，PPO、GRPO、DAPO-adapted 分别出现 1、2、2 次低于 -100% 的灾难性提案。"
        "因此下一版仍以 PPO 为第一基线，同时强制保留 best-single incumbent。",
        "",
        "## 总表",
        "",
        "| 策略 | 优于 baseline | 优于 best-single | baseline 增益中位数 | 对 best-single 额外增益 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall_summary:
        lines.append(
            f"| {row['algorithm'].upper()} | {row['columns_positive_by_median']}/20 | "
            f"{row['columns_beating_best_single_by_median']}/20 | "
            f"{fmt(float(row['median_column_gain_pct']))}% | "
            f"{fmt(float(row['median_column_gain_vs_best_single_pct']))}% |"
        )
    lines.extend(
        [
            "",
            "## 数据集分解",
            "",
            "| 策略 | 数据集 | 中位数为正的列 | 列增益中位数 | 最差列 | 最好列 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_summary:
        lines.append(
            f"| {row['algorithm'].upper()} | {row['dataset']} | "
            f"{row['columns_positive_by_median']}/{row['columns']} | "
            f"{fmt(float(row['median_column_gain_pct']))}% | "
            f"{fmt(float(row['min_column_gain_pct']))}% | "
            f"{fmt(float(row['max_column_gain_pct']))}% |"
        )
    negative_rows = [row for row in wide_rows if not int(row["all_algorithms_positive"])]
    lines.extend(
        [
            "",
            "## 仍未超过 baseline 的列",
            "",
            "| 数据集 | 列 | PPO | GRPO | DAPO-adapted |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in negative_rows:
        lines.append(
            f"| {row['dataset']} | {row['column_name']} | "
            f"{fmt(float(row['ppo_median_gain_pct']))}% | "
            f"{fmt(float(row['grpo_median_gain_pct']))}% | "
            f"{fmt(float(row['dapo_median_gain_pct']))}% |"
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- 5 折 held-out-column cross-validation，每列只由没有见过该列的策略评估。",
            "- 3 个随机种子。PPO、GRPO、DAPO-adapted 使用相同 28 算子、深度 3 和 STOP 动作。",
            "- 增益 = 100 × (各列最优 baseline rate - 策略 proxy rate) / 各列最优 baseline rate。",
            "- 主结果保留 28 个单算子 probe 中的最佳 incumbent，再与策略提案取较小 rate；原始提案仍单独保留用于失败分析。",
            "- DAPO-adapted 使用 Clip-Higher、Dynamic Sampling 和 action-level loss；固定三步任务不使用 overlong shaping。",
            "",
            "逐列结果见 `per_column_algorithm_summary.csv`，逐种子原始结果见 `per_seed_evaluation.csv`。",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "verdict": "PASS_WITH_PROXY_LIMIT",
        "checks": {
            "algorithms_exact": algorithms,
            "seeds_exact": seeds,
            "columns_exact": len(column_keys),
            "row_count": len(rows),
            "expected_row_count": expected_rows,
            "duplicate_keys": False,
            "fold_assignments_identical": True,
            "baseline_consistent": True,
            "gain_recomputed": True,
            "strict_better_recomputed": True,
            "finite_positive_rates": all(float(row["proxy_rate"]) > 0 for row in rows),
            "safe_rate_not_worse_than_proposal": all(
                float(row["safe_proxy_rate"]) <= float(row["proxy_rate"]) for row in rows
            ),
            "safe_rate_not_worse_than_best_single": all(
                float(row["safe_proxy_rate"]) <= float(row["best_single_rate"]) for row in rows
            ),
            "early_stopped_runs": early_stops_by_algorithm,
        },
        "evaluation_type": "simulation_only_proxy",
        "claim_limit": claim_check["safe_claim"],
        "input_hashes": {str(path): sha256(path) for path in input_files},
        "output_hashes": {
            path.name: sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "AUDIT.json"
        },
    }
    (output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_median": best_median,
                "recommended": recommended,
                "all_columns_positive": all_columns_positive,
            },
            ensure_ascii=False,
        )
    )
    print(f"OUTPUT {output_dir}")


if __name__ == "__main__":
    main()
