#!/usr/bin/env python3
"""Summarize corrected PV2F oracle, Beam-29, PPO, and GRPO-style results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def median_representative(rows: list[dict[str, str]]) -> dict[str, str]:
    median_bytes = statistics.median(int(row["encoded_bytes"]) for row in rows)
    return min(
        rows,
        key=lambda row: (
            abs(int(row["encoded_bytes"]) - median_bytes),
            int(row["encoded_bytes"]),
            int(row["seed"]),
            row["pipeline"],
        ),
    )


def is_single_codec(row: dict[str, str]) -> bool:
    """A complete PV2F pipeline with exactly one non-trivial codec action."""
    transforms = int(row["transform_count"])
    return transforms + int(row["packer"] != "raw.i64") + int(row["compressor"] != "none") == 1


def old_beam_lookup(repo_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    candidates = (
        ("Metro-Traffic", repo_root / "analysis_outputs/metro_beam29_20260825_v1/per_column.csv"),
        ("Air-Quality", repo_root / "analysis_outputs/air_beam29_20260825_v1/output/per_column.csv"),
    )
    result: dict[tuple[str, str], dict[str, str]] = {}
    for dataset, path in candidates:
        if not path.is_file():
            continue
        for row in read_csv(path):
            result[(dataset, row["column_name"])] = row
    return result


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    measurement_root = args.measurement_root.resolve()
    search_dir = args.search_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.audit_json.resolve().read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise AssertionError("deterministic audit must pass before analysis")
    fresh_audit_path = args.audit_json.resolve().parent / "EXPERIMENT_AUDIT.json"
    fresh_audit = (
        json.loads(fresh_audit_path.read_text(encoding="utf-8"))
        if fresh_audit_path.is_file()
        else {"status": "NOT_RUN", "claim_boundary": "fresh independent audit not found"}
    )
    beam_rows = read_csv(search_dir / "beam29_results.csv")
    rl_rows = read_csv(search_dir / "rl_evaluation.csv")
    summary = json.loads((search_dir / "summary.json").read_text(encoding="utf-8"))
    timing_rows = read_csv(args.timing_csv.resolve()) if args.timing_csv else []
    timing_by_key = {
        (row["dataset"], int(row["column_index"]), row["method"]): row
        for row in timing_rows
    }
    timing_manifest_path = args.timing_csv.resolve().parent / "TIMING_MANIFEST.json" if args.timing_csv else None
    timing_manifest = (
        json.loads(timing_manifest_path.read_text(encoding="utf-8"))
        if timing_manifest_path and timing_manifest_path.is_file()
        else None
    )
    old_lookup = old_beam_lookup(repo_root)
    catalog_paths = sorted(measurement_root.glob("*/action_catalog.json"))
    if not catalog_paths:
        raise FileNotFoundError(f"no action catalog under {measurement_root}")
    catalog = json.loads(catalog_paths[0].read_text(encoding="utf-8"))
    action_by_name = {action["name"]: action for action in catalog["actions"]}
    measurement_by_key: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for csv_path in sorted(measurement_root.glob("*/pipeline_measurements.csv")):
        rows = read_csv(csv_path)
        if not rows:
            continue
        key = (rows[0]["dataset"], int(rows[0]["column_index"]))
        measurement_by_key[key] = {row["pipeline"]: row for row in rows}

    grouped_rl: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rl_rows:
        grouped_rl[(row["test_dataset"], int(row["column_index"]), row["algorithm"])].append(row)
    per_column: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    expanded_rows: list[dict[str, Any]] = []
    depth_frontier_rows: list[dict[str, Any]] = []
    oracle_action_frequency: Counter[str] = Counter()
    corrected_old_changes = 0
    for beam in beam_rows:
        key = (beam["dataset"], int(beam["column_index"]))
        ppo = median_representative(grouped_rl[(key[0], key[1], "ppo")])
        grpo = median_representative(grouped_rl[(key[0], key[1], "grpo")])
        methods = {
            "Beam-29": (int(beam["encoded_bytes"]), beam["pipeline"]),
            "PPO": (int(ppo["encoded_bytes"]), ppo["pipeline"]),
            "GRPO-style": (int(grpo["encoded_bytes"]), grpo["pipeline"]),
        }
        best_bytes = min(result[0] for result in methods.values())
        best_methods = tuple(sorted(name for name, result in methods.items() if result[0] == best_bytes))
        best_method = "=".join(best_methods)
        best_pipelines = tuple((name, methods[name][1]) for name in best_methods)
        best_pipeline = best_pipelines[0][1]
        oracle_pipeline = beam["oracle_pipeline"]
        oracle_row = measurement_by_key[key][oracle_pipeline]
        oracle_timing = timing_by_key.get((key[0], key[1], "Bounded oracle"))
        if args.timing_csv and oracle_timing is None:
            raise AssertionError((key[0], key[1], "Bounded oracle", "missing timing"))
        if oracle_timing:
            if (
                oracle_timing["pipeline"] != oracle_pipeline
                or int(oracle_timing["encoded_bytes"]) != int(beam["oracle_bytes"])
            ):
                raise AssertionError((key[0], key[1], "Bounded oracle", "timing/result mismatch"))
            oracle_timing_fields: dict[str, Any] = {
                "oracle_candidate_count": int(oracle_timing["candidate_count"]),
                "oracle_encode_repeats": int(oracle_timing["encode_repeats"]),
                "oracle_encode_ms_median": float(oracle_timing["encode_ms_median"]),
                "oracle_encode_ms_p25": float(oracle_timing["encode_ms_p25"]),
                "oracle_encode_ms_p75": float(oracle_timing["encode_ms_p75"]),
            }
        else:
            oracle_timing_fields = {
                "oracle_candidate_count": "",
                "oracle_encode_repeats": "",
                "oracle_encode_ms_median": "",
                "oracle_encode_ms_p25": "",
                "oracle_encode_ms_p75": "",
            }
        ok_measurements = [row for row in measurement_by_key[key].values() if row["status"] == "ok"]
        single_candidates = [row for row in ok_measurements if is_single_codec(row)]
        if not single_candidates:
            raise AssertionError((key, "no single-codec candidate"))
        single = min(single_candidates, key=lambda row: (int(row["final_bytes"]), row["pipeline"]))
        single_timing = timing_by_key.get((key[0], key[1], "Best single codec"))
        if args.timing_csv and single_timing is None:
            raise AssertionError((key[0], key[1], "Best single codec", "missing timing"))
        if single_timing:
            if (
                single_timing["pipeline"] != single["pipeline"]
                or int(single_timing["encoded_bytes"]) != int(single["final_bytes"])
            ):
                raise AssertionError((key[0], key[1], "Best single codec", "timing/result mismatch"))
            single_timing_fields: dict[str, Any] = {
                "single_codec_candidates": int(single_timing["candidate_count"]),
                "single_codec_encode_repeats": int(single_timing["encode_repeats"]),
                "single_codec_encode_ms_median": float(single_timing["encode_ms_median"]),
                "single_codec_encode_ms_p25": float(single_timing["encode_ms_p25"]),
                "single_codec_encode_ms_p75": float(single_timing["encode_ms_p75"]),
            }
        else:
            single_timing_fields = {
                "single_codec_candidates": len(single_candidates),
                "single_codec_encode_repeats": "",
                "single_codec_encode_ms_median": "",
                "single_codec_encode_ms_p25": "",
                "single_codec_encode_ms_p75": "",
            }
        for transform_count in range(3):
            depth_rows = [
                row for row in ok_measurements if int(row["transform_count"]) == transform_count
            ]
            depth_best = min(
                depth_rows,
                key=lambda row: (int(row["final_bytes"]), row["pipeline"]),
            )
            depth_frontier_rows.append(
                {
                    "dataset": key[0],
                    "column_index": key[1],
                    "column_name": beam["column_name"],
                    "exact_transform_count": transform_count,
                    "pipeline": depth_best["pipeline"],
                    "final_bytes": int(depth_best["final_bytes"]),
                    "compression_rate": float(depth_best["compression_rate"]),
                    "gap_vs_bounded_oracle_pct": 100.0
                    * (int(depth_best["final_bytes"]) - int(beam["oracle_bytes"]))
                    / int(beam["oracle_bytes"]),
                }
            )
        for action in oracle_pipeline.split(" -> "):
            oracle_action_frequency[action] += 1
        for position, action_name in enumerate(oracle_pipeline.split(" -> "), start=1):
            action = action_by_name[action_name]
            expanded_rows.append(
                {
                    "dataset": key[0],
                    "column_index": key[1],
                    "column_name": beam["column_name"],
                    "oracle_pipeline": oracle_pipeline,
                    "position": position,
                    "action": action_name,
                    "phase": action["phase"],
                    "family": action["family"],
                    "implementation_source": action["implementation_source"],
                    "conceptual_stages": " -> ".join(action["conceptual_stages"]),
                    "block_size": action["block_size"],
                    "dictionary_size": action["dictionary_size"],
                    "window_size": action["window_size"],
                    "exception_stream": int(action["exception_stream"]),
                    "inner_framing": action["inner_framing"],
                    "inner_checksum": action["inner_checksum"],
                }
            )
        old = old_lookup.get((key[0], beam["column_name"]))
        old_pipeline = old.get("beam_pipeline", "") if old else ""
        changed = bool(old_pipeline and old_pipeline != oracle_pipeline)
        corrected_old_changes += int(changed)
        per_column.append(
            {
                "dataset": key[0],
                "column_index": key[1],
                "column_name": beam["column_name"],
                "original_bytes": int(beam["original_bytes"]),
                "oracle_pipeline": oracle_pipeline,
                "oracle_bytes": int(beam["oracle_bytes"]),
                "oracle_rate": float(beam["oracle_rate"]),
                **oracle_timing_fields,
                "single_codec_pipeline": single["pipeline"],
                "single_codec_bytes": int(single["final_bytes"]),
                "single_codec_rate": float(single["compression_rate"]),
                "single_codec_gap_vs_oracle_pct": 100.0
                * (int(single["final_bytes"]) - int(beam["oracle_bytes"]))
                / int(beam["oracle_bytes"]),
                **single_timing_fields,
                "beam29_pipeline": beam["pipeline"],
                "beam29_bytes": int(beam["encoded_bytes"]),
                "beam29_rate": float(beam["compression_rate"]),
                "beam29_oracle_regret_pct": float(beam["oracle_regret_pct"]),
                "beam29_evaluations": int(beam["candidate_evaluations"]),
                "ppo_median_pipeline": ppo["pipeline"],
                "ppo_median_bytes": int(ppo["encoded_bytes"]),
                "ppo_median_rate": float(ppo["compression_rate"]),
                "ppo_oracle_regret_pct": float(ppo["oracle_regret_pct"]),
                "grpo_median_pipeline": grpo["pipeline"],
                "grpo_median_bytes": int(grpo["encoded_bytes"]),
                "grpo_median_rate": float(grpo["compression_rate"]),
                "grpo_oracle_regret_pct": float(grpo["oracle_regret_pct"]),
                "best_search_method": best_method,
                "best_search_tie_count": len(best_methods),
                "best_search_bytes": best_bytes,
                "best_search_pipeline": best_pipeline,
                "best_search_pipelines": " || ".join(
                    f"{name}: {pipeline}" for name, pipeline in best_pipelines
                ),
                "old_abstract_beam_pipeline": old_pipeline,
                "old_conclusion_invalidated": int(bool(old_pipeline)),
                "winner_changed_from_old_text": int(changed),
            }
        )
        component_rows.append(
            {
                "dataset": key[0],
                "column_index": key[1],
                "column_name": beam["column_name"],
                "oracle_pipeline": oracle_pipeline,
                "outer_header_bytes": int(oracle_row["outer_header_bytes"]),
                "action_table_bytes": int(oracle_row["action_table_bytes"]),
                "metadata_length_prefix_bytes": int(oracle_row["metadata_length_prefix_bytes"]),
                "transform_metadata_bytes": int(oracle_row["transform_metadata_bytes"]),
                "packer_payload_bytes_before_generic": int(oracle_row["packer_payload_bytes"]),
                "compressed_payload_bytes": int(oracle_row["compressed_payload_bytes"]),
                "checksum_bytes": int(oracle_row["checksum_bytes"]),
                "outer_frame_bytes": int(oracle_row["outer_frame_bytes"]),
                "final_bytes": int(oracle_row["final_bytes"]),
            }
        )
    write_csv(output_dir / "per_column_comparison.csv", per_column)
    write_csv(
        output_dir / "single_codec_baseline.csv",
        [
            {
                "dataset": row["dataset"],
                "column_index": row["column_index"],
                "column_name": row["column_name"],
                "original_bytes": row["original_bytes"],
                "pipeline": row["single_codec_pipeline"],
                "encoded_bytes": row["single_codec_bytes"],
                "compression_rate": row["single_codec_rate"],
                "gap_vs_bounded_oracle_pct": row["single_codec_gap_vs_oracle_pct"],
                "candidate_count": row["single_codec_candidates"],
                "encode_repeats": row["single_codec_encode_repeats"],
                "encode_ms_median": row["single_codec_encode_ms_median"],
                "encode_ms_p25": row["single_codec_encode_ms_p25"],
                "encode_ms_p75": row["single_codec_encode_ms_p75"],
            }
            for row in per_column
        ],
    )
    per_method_rows: list[dict[str, Any]] = []
    for row in per_column:
        methods = (
            ("Beam-29", "", row["beam29_pipeline"], row["beam29_bytes"], row["beam29_rate"]),
            ("PPO", "ppo", row["ppo_median_pipeline"], row["ppo_median_bytes"], row["ppo_median_rate"]),
            ("GRPO-style", "grpo", row["grpo_median_pipeline"], row["grpo_median_bytes"], row["grpo_median_rate"]),
        )
        best_search_bytes = min(int(method[3]) for method in methods)
        for method, algorithm, pipeline, encoded_bytes, compression_rate in methods:
            timing = timing_by_key.get((row["dataset"], int(row["column_index"]), method))
            if args.timing_csv and timing is None:
                raise AssertionError((row["dataset"], row["column_index"], method, "missing timing"))
            if timing:
                if timing["pipeline"] != pipeline or int(timing["encoded_bytes"]) != int(encoded_bytes):
                    raise AssertionError((row["dataset"], row["column_index"], method, "timing/result mismatch"))
                timing_fields: dict[str, Any] = {
                    "representative_seed": timing["representative_seed"],
                    "fold": timing["fold"],
                    "candidate_count": int(timing["candidate_count"]),
                    "selection_repeats": int(timing["selection_repeats"]),
                    "selection_ms_median": float(timing["selection_ms_median"]),
                    "selection_ms_p25": float(timing["selection_ms_p25"]),
                    "selection_ms_p75": float(timing["selection_ms_p75"]),
                    "encode_repeats": int(timing["encode_repeats"]),
                    "encode_ms_median": float(timing["encode_ms_median"]),
                    "encode_ms_p25": float(timing["encode_ms_p25"]),
                    "encode_ms_p75": float(timing["encode_ms_p75"]),
                }
            else:
                timing_fields = {
                    "representative_seed": "",
                    "fold": "",
                    "candidate_count": "",
                    "selection_repeats": "",
                    "selection_ms_median": "",
                    "selection_ms_p25": "",
                    "selection_ms_p75": "",
                    "encode_repeats": "",
                    "encode_ms_median": "",
                    "encode_ms_p25": "",
                    "encode_ms_p75": "",
                }
            per_method_rows.append(
                {
                    "dataset": row["dataset"],
                    "column_index": row["column_index"],
                    "column_name": row["column_name"],
                    "original_bytes": row["original_bytes"],
                    "method": method,
                    "algorithm": algorithm,
                    "pipeline": pipeline,
                    "encoded_bytes": encoded_bytes,
                    "compression_rate": compression_rate,
                    "is_best_among_three": int(int(encoded_bytes) == best_search_bytes),
                    "bounded_oracle_pipeline": row["oracle_pipeline"],
                    "bounded_oracle_bytes": row["oracle_bytes"],
                    "bounded_oracle_rate": row["oracle_rate"],
                    **timing_fields,
                }
            )
    write_csv(output_dir / "per_column_method_comparison.csv", per_method_rows)
    write_csv(output_dir / "oracle_winner_components.csv", component_rows)
    write_csv(output_dir / "oracle_pipeline_expanded.csv", expanded_rows)
    write_csv(output_dir / "exact_transform_depth_frontier.csv", depth_frontier_rows)
    frequency_rows = [
        {"action": action, "oracle_columns": count}
        for action, count in oracle_action_frequency.most_common()
    ]
    write_csv(output_dir / "oracle_action_frequency.csv", frequency_rows)

    algorithm_by_name = {row["algorithm"]: row for row in summary["algorithms"]}
    beam = summary["beam29"]
    ppo = algorithm_by_name["ppo"]
    grpo = algorithm_by_name["grpo"]
    best_method_sets = Counter(row["best_search_method"] for row in per_column)
    strict_method_wins = Counter(
        row["best_search_method"]
        for row in per_column
        if int(row["best_search_tie_count"]) == 1
    )
    tied_columns = sum(int(row["best_search_tie_count"]) > 1 for row in per_column)
    beam_evaluations = [int(row["candidate_evaluations"]) for row in beam_rows]
    ppo_unique = [int(row["unique_proposals"]) for row in rl_rows if row["algorithm"] == "ppo"]
    grpo_unique = [int(row["unique_proposals"]) for row in rl_rows if row["algorithm"] == "grpo"]
    seed_totals: dict[tuple[str, int], int] = defaultdict(int)
    dataset_seed_totals: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in rl_rows:
        algorithm = row["algorithm"]
        seed = int(row["seed"])
        encoded_bytes = int(row["encoded_bytes"])
        seed_totals[(algorithm, seed)] += encoded_bytes
        dataset_seed_totals[(algorithm, row["test_dataset"], seed)] += encoded_bytes
    beam_dataset_totals: dict[str, int] = defaultdict(int)
    for row in beam_rows:
        beam_dataset_totals[row["dataset"]] += int(row["encoded_bytes"])
    rl_seed_diagnostics: dict[str, Any] = {}
    for algorithm in ("ppo", "grpo"):
        combined = sorted(
            total for (name, _seed), total in seed_totals.items() if name == algorithm
        )
        per_dataset: dict[str, Any] = {}
        for dataset in sorted(beam_dataset_totals):
            totals = sorted(
                total
                for (name, test_dataset, _seed), total in dataset_seed_totals.items()
                if name == algorithm and test_dataset == dataset
            )
            median_total = statistics.median(totals)
            beam_total = beam_dataset_totals[dataset]
            per_dataset[dataset] = {
                "seed_totals": totals,
                "min_bytes": min(totals),
                "median_bytes": median_total,
                "max_bytes": max(totals),
                "beam29_bytes": beam_total,
                "saving_vs_beam29_pct": 100.0 * (beam_total - median_total) / beam_total,
            }
        rl_seed_diagnostics[algorithm] = {
            "combined_seed_totals": combined,
            "combined_min_bytes": min(combined),
            "combined_median_bytes": statistics.median(combined),
            "combined_max_bytes": max(combined),
            "per_dataset": per_dataset,
        }
    budget_diagnostic = {
        "beam_candidate_evaluations_min": min(beam_evaluations),
        "beam_candidate_evaluations_median": statistics.median(beam_evaluations),
        "beam_candidate_evaluations_max": max(beam_evaluations),
        "rl_sampled_rollouts_per_column": int(rl_rows[0]["proposal_rollouts"]),
        "ppo_unique_proposals_min": min(ppo_unique),
        "ppo_unique_proposals_median": statistics.median(ppo_unique),
        "ppo_unique_proposals_max": max(ppo_unique),
        "grpo_unique_proposals_min": min(grpo_unique),
        "grpo_unique_proposals_median": statistics.median(grpo_unique),
        "grpo_unique_proposals_max": max(grpo_unique),
    }
    oracle_at_transform_cap = sum(
        1 for row in per_column if measurement_by_key[(row["dataset"], row["column_index"])][row["oracle_pipeline"]]["transform_count"] == "2"
    )
    timing_summary: dict[str, Any] = {}
    if timing_rows:
        for method in ("Beam-29", "PPO", "GRPO-style"):
            rows = [row for row in timing_rows if row["method"] == method]
            timing_summary[method] = {
                "columns": len(rows),
                "selection_ms_median_across_columns": statistics.median(
                    float(row["selection_ms_median"]) for row in rows
                ),
                "selection_ms_range_across_columns": [
                    min(float(row["selection_ms_median"]) for row in rows),
                    max(float(row["selection_ms_median"]) for row in rows),
                ],
                "encode_ms_median_across_columns": statistics.median(
                    float(row["encode_ms_median"]) for row in rows
                ),
                "encode_ms_range_across_columns": [
                    min(float(row["encode_ms_median"]) for row in rows),
                    max(float(row["encode_ms_median"]) for row in rows),
                ],
            }
        oracle_timing_rows = [row for row in timing_rows if row["method"] == "Bounded oracle"]
        if len(oracle_timing_rows) != len(per_column):
            raise AssertionError(("Bounded oracle", len(oracle_timing_rows), len(per_column)))
        timing_summary["Bounded oracle"] = {
            "columns": len(oracle_timing_rows),
            "candidate_count": int(oracle_timing_rows[0]["candidate_count"]),
            "selection_definition": "offline complete enumeration; not timed as an online selector",
            "encode_ms_median_across_columns": statistics.median(
                float(row["encode_ms_median"]) for row in oracle_timing_rows
            ),
            "encode_ms_range_across_columns": [
                min(float(row["encode_ms_median"]) for row in oracle_timing_rows),
                max(float(row["encode_ms_median"]) for row in oracle_timing_rows),
            ],
        }
        single_timing_rows = [row for row in timing_rows if row["method"] == "Best single codec"]
        if len(single_timing_rows) != len(per_column):
            raise AssertionError(("Best single codec", len(single_timing_rows), len(per_column)))
        timing_summary["Best single codec"] = {
            "columns": len(single_timing_rows),
            "candidate_count": int(single_timing_rows[0]["candidate_count"]),
            "selection_definition": "offline minimum over single-codec candidates; no online selection time",
            "encode_ms_median_across_columns": statistics.median(
                float(row["encode_ms_median"]) for row in single_timing_rows
            ),
            "encode_ms_range_across_columns": [
                min(float(row["encode_ms_median"]) for row in single_timing_rows),
                max(float(row["encode_ms_median"]) for row in single_timing_rows),
            ],
        }
    table_lines = []
    for row in per_column:
        table_lines.append(
            "| {dataset} | {column_name} | `{oracle_pipeline}` | {oracle_rate:.5%} | {beam29_oracle_regret_pct:.4f}% | {ppo_oracle_regret_pct:.4f}% | {grpo_oracle_regret_pct:.4f}% | {best_search_method} |".format(**row)
        )
    report = f"""# 变种级物理算子搜索结果

## 结论

旧 28 标签结果已经失效。那套脚本中的多个算子没有独立物理实现，并且统一套 zlib；本轮改为 12 个 transform 变种、13 个 packer 变种和 10 个通用 compressor 变种。每条候选都计入 transform metadata、block/dictionary 参数、exception stream、action table、inner compressor frame、PV2F outer frame 和 4-byte CRC32。

声明的 PV2F depth≤2 grammar 每列 {summary['complete_candidates_per_column']:,} 条，20 列合计 {summary['complete_candidates_per_column'] * summary['columns']:,} 条，已在该声明边界内完整枚举。每条候选都在枚举时完成 in-run encode/decode；确定性审计对 {audit['checked_component_rows']:,} 行完成 hash、字节算术与 oracle 检查，并对 {audit['replayed_unique_claimed_pipelines']} 条被 Beam/PPO/GRPO-style 或 oracle 引用的 pipeline 做 post-hoc 重新编码/解码。审计状态为 {audit['status']}。

独立 `experiment-audit` 的最终状态为 **{fresh_audit['status']}**：有界 benchmark 数值可用，但全 int64 codec contract、全 frame/action 完整性与细粒度 side-stream 计费主张未通过。因此本报告的 `PASS` 只指确定性表格/算术审计，不表示 codec 已通过生产级或任意 int64 正确性审计。

Beam-29 总字节为 {beam['total_bytes']:,}，oracle 为 {beam['oracle_total_bytes']:,}，聚合 oracle regret 为 {beam['aggregate_oracle_regret_pct']:.6f}%。PPO 的 3-seed 中位总字节为 {int(ppo['median_total_bytes']):,}，相对 Beam-29 为 {ppo['median_gain_vs_beam29_pct']:+.6f}%；GRPO-style 为 {int(grpo['median_total_bytes']):,}，相对 Beam-29 为 {grpo['median_gain_vs_beam29_pct']:+.6f}%。这里正值表示比 Beam-29 少字节。

PPO 三个 combined-byte seed 总量范围为 {rl_seed_diagnostics['ppo']['combined_min_bytes']:,}–{rl_seed_diagnostics['ppo']['combined_max_bytes']:,} B；GRPO-style 为 {rl_seed_diagnostics['grpo']['combined_min_bytes']:,}–{rl_seed_diagnostics['grpo']['combined_max_bytes']:,} B。分数据集看，PPO 在 Air-Quality 的中位数为 {rl_seed_diagnostics['ppo']['per_dataset']['Air-Quality']['median_bytes']:,.0f} B，相对 Beam 的 {rl_seed_diagnostics['ppo']['per_dataset']['Air-Quality']['beam29_bytes']:,} B 节省 {rl_seed_diagnostics['ppo']['per_dataset']['Air-Quality']['saving_vs_beam29_pct']:.3f}%；在 Metro-Traffic 为 {rl_seed_diagnostics['ppo']['per_dataset']['Metro-Traffic']['median_bytes']:,.0f} B，相对 Beam 的 {rl_seed_diagnostics['ppo']['per_dataset']['Metro-Traffic']['beam29_bytes']:,} B 多 {abs(rl_seed_diagnostics['ppo']['per_dataset']['Metro-Traffic']['saving_vs_beam29_pct']):.3f}%。GRPO-style 在 Air/Metro 分别为 {rl_seed_diagnostics['grpo']['per_dataset']['Air-Quality']['median_bytes']:,.0f}/{rl_seed_diagnostics['grpo']['per_dataset']['Metro-Traffic']['median_bytes']:,.0f} B，相对 Beam 多 {abs(rl_seed_diagnostics['grpo']['per_dataset']['Air-Quality']['saving_vs_beam29_pct']):.3f}%/{abs(rl_seed_diagnostics['grpo']['per_dataset']['Metro-Traffic']['saving_vs_beam29_pct']):.3f}%。只有两个 dataset unit，seed 只反映优化器随机性，不能据此估计跨数据集泛化置信区间。

补充计时对每列的有界 oracle、最优单算法、Beam、PPO 代表 seed 和 GRPO-style 代表 seed 重新复现了相同 pipeline/字节。有界 oracle与最优单算法的完整 prototype Python 编码时间跨列中位数分别为 {timing_summary.get('Bounded oracle', {}).get('encode_ms_median_across_columns', float('nan')):.3f}/{timing_summary.get('Best single codec', {}).get('encode_ms_median_across_columns', float('nan')):.3f} ms；Beam/PPO/GRPO-style 分别为 {timing_summary.get('Beam-29', {}).get('encode_ms_median_across_columns', float('nan')):.3f}/{timing_summary.get('PPO', {}).get('encode_ms_median_across_columns', float('nan')):.3f}/{timing_summary.get('GRPO-style', {}).get('encode_ms_median_across_columns', float('nan')):.3f} ms。三种搜索的跨列选择时间中位数为 Beam {timing_summary.get('Beam-29', {}).get('selection_ms_median_across_columns', float('nan')):.3f} ms、PPO {timing_summary.get('PPO', {}).get('selection_ms_median_across_columns', float('nan')):.3f} ms、GRPO-style {timing_summary.get('GRPO-style', {}).get('selection_ms_median_across_columns', float('nan')):.3f} ms。Oracle 和最优单算法都是离线基线，不把完整枚举或离线取最小的总时间伪装成在线选择时间；页面只报告已知 pipeline 的单次完整编码时间。三种搜索的选择时间只包含冻结字节表和已加载策略/特征后的查表搜索，不含候选真实压缩、数据读取、特征计算或模型加载；编码时间从 normalized values 开始，包含完整 transform/packer/compressor/PV2F framing，但不含解码验证。

逐列把 3-seed 中位 PPO/GRPO-style 与 Beam-29 比较，严格单胜计数为 {dict(strict_method_wins)}，另有 {tied_columns} 列并列；完整 best-set 分布为 {dict(best_method_sets)}。在当前两个数据集上，只有在 RL 跨数据集测试结果稳定少于 Beam-29 时，才能说 RL 对搜索有帮助；否则更合理的结论是先保留 Beam，或直接使用声明的 depth≤2 空间 oracle 做离线标注。

这不是“相同 unique candidate 数”的样本效率比较。Beam-29 每列实际查表 {budget_diagnostic['beam_candidate_evaluations_min']}–{budget_diagnostic['beam_candidate_evaluations_max']} 个候选，中位 {budget_diagnostic['beam_candidate_evaluations_median']:.0f}；PPO/GRPO-style 各做 512 次 sampled rollout，但策略重复导致 PPO 的 unique proposal 中位仅 {budget_diagnostic['ppo_unique_proposals_median']:.0f}、GRPO-style 仅 {budget_diagnostic['grpo_unique_proposals_median']:.0f}。held-out 结果是 frozen-policy proposal search 后再在 test byte table 中选最小值，不是纯 greedy zero-shot。训练 lookup、测试 rollout 与 Beam lookup 预算也不匹配，两个算法的 seed sampling stream 并非 paired rollout。因此当前负结果只说明“这组自定义 PPO/GRPO-style 配置发生 proposal collapse 且没有在 combined-byte 中位数上胜过 Beam”，不能上升为 PPO/GRPO 算法本身不适合该问题。

## 每列结果

| 数据集 | 列 | 声明空间 oracle pipeline | Oracle rate | Beam regret | PPO regret | GRPO-style regret | 三种搜索最佳 |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(table_lines)}

## 旧结论为什么不能继续使用

旧 HTML/CSV 中有 {sum(1 for row in per_column if row['old_abstract_beam_pipeline'])} 列能找到旧 abstract Beam winner；这些行全部标记为 invalidated，其中 {corrected_old_changes} 列的文字 pipeline 与新物理 oracle 不同。即使文字偶然相同，也不能把旧字节数移植过来，因为新的 frame 固定计入 block size、dictionary limit、exception stream、header、framing 和 checksum。

## 字节口径

`oracle_winner_components.csv` 保存每列 winner 的精确分项。最终字节恒等于：

`outer header + action table + metadata length prefixes + transform metadata + compressed payload + CRC32`

`packer_payload_bytes_before_generic` 单独记录通用压缩前的 payload；如果 compressor 不是 `none`，`compressed_payload_bytes` 已包含 zlib/zstd/XZ/Brotli/bzip2/LZ4 自己的 frame，以及仅在相应 codec 配置存在时的内部 checksum（Brotli 本轮没有 inner checksum）。所有候选另外固定带 PV2F CRC32；这个 CRC32 只校验还原后的 normalized values，并不认证 header、action table 或完整 frame，所以不能声称能检测任意 frame 字节篡改。

当前 PV2F 把 dictionary、outlier exception/escape 等旁路流用 varint 写入 transform metadata，完整计入最终字节，但 terminal compressor 只压缩主 packer payload，不会再次压缩这些 metadata/旁路流。现有 counter 只精确到聚合的 transform metadata / packer payload，不能把 block 参数、dictionary、exception/escape、inner frame 和 inner checksum 各自单独归因。真实 Parquet/BOS 容器可能对不同 page/stream 分别压缩，因此本轮 winner 不能直接当成这些生产格式的 byte-identical winner。

`oracle_pipeline_expanded.csv` 把每个 winner 的宏动作展开为内部 stage，并逐动作记录 block size、dictionary size、window size、异常值流、实现来源、inner framing 和 inner checksum。这里的搜索深度是“最多两个外层 transform + 一个 packer + 一个 compressor”；packer/compressor 的内部 stage 仍完整展示，例如 PFOR 会展开为 FOR、bitpack、exception stream，LZMA2/XZ 会展开为 match、range coder 和 container frame。

`exact_transform_depth_frontier.csv` 比较每列恰好 0、1、2 个外层 transform 的最优结果。共有 {oracle_at_transform_cap}/{len(per_column)} 列的 bounded oracle 落在 2-transform 上限。如果该数值较大，本轮只能证明“修正后的有界空间”结论，不能声称继续增加外层 transform 一定无效；这会直接触发下一轮 depth-3 验证。

## 主张边界

有效性只覆盖本轮 Air-Quality/Metro-Traffic 20 个有限 normalized-integer 列中实际观察到的取值范围，不覆盖任意 int64；极值反例已使全 int64 contract 的独立审计失败。这仍是 standalone prototype，不能替代生产 Parquet/BOS 格式兼容性、旁路流压缩策略、吞吐、随机访问、查询下推或原始缺失值语义实验。PPO/GRPO-style 只做了两个 dataset unit 的双向跨数据集测试，不能写成广泛泛化结论。旧 winner 的差异只是读取 repo 外部旧 CSV 后记录的比较，旧输入没有被归档/哈希，不能当成独立审计过的 legacy evidence。

> On the 14 Air-Quality and six Metro-Traffic finite normalized-integer columns, we completely enumerated the declared PV2F grammar of 0–2 non-repeated exact transforms, one packer and one compressor: 18,850 candidates per column. Every candidate passed an in-run round-trip under the recorded environment, and `final_bytes` is the actual serialized PV2F frame length. Beam-29 used 273,436 B versus a 264,323 B bounded oracle. The tested custom PPO and GRPO-style configurations had three initialization seeds and frozen cross-dataset proposal evaluation; their combined-byte seed medians were respectively 4.55% and 6.62% worse than Beam. This was not a matched-unique-candidate comparison, proposal collapse occurred, and the result does not establish that PPO/GRPO are intrinsically inferior.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    analysis = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BOUNDED_RESULT_USABLE_FRESH_AUDIT_FAIL"
        if fresh_audit.get("status") == "FAIL"
        else ("PASS" if audit["status"] == "PASS" else "BLOCKED"),
        "columns": len(per_column),
        "complete_candidates": summary["complete_candidates_per_column"] * summary["columns"],
        "beam29": beam,
        "ppo": ppo,
        "grpo": grpo,
        "strict_method_wins": dict(strict_method_wins),
        "tied_columns": tied_columns,
        "best_method_sets": dict(best_method_sets),
        "oracle_at_transform_cap": oracle_at_transform_cap,
        "budget_diagnostic": budget_diagnostic,
        "rl_seed_diagnostics": rl_seed_diagnostics,
        "timing_summary": timing_summary,
        "timing_manifest": timing_manifest,
        "old_rows_invalidated": sum(1 for row in per_column if row["old_abstract_beam_pipeline"]),
        "winner_text_changed": corrected_old_changes,
        "audit": audit,
        "fresh_audit": fresh_audit,
    }
    (output_dir / "ANALYSIS.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run(parse_args())
