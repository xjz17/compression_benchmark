#!/usr/bin/env python3
"""Render a standalone HTML dashboard for the corrected PV2F experiment."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pipeline_html(value: str) -> str:
    return '<span class="pipe">' + "<i>→</i>".join(
        f"<code>{esc(part.strip())}</code>" for part in value.split(" -> ")
    ) + "</span>"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--sparse19-audit", type=Path, required=True)
    parser.add_argument("--external-sparse19-audit", type=Path)
    parser.add_argument("--minimal6-summary", type=Path)
    parser.add_argument("--prefix-batch-summary", type=Path)
    parser.add_argument("--exact-beam-summary", type=Path)
    parser.add_argument("--previous-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    analysis_dir = args.analysis_dir.resolve()
    search_dir = args.search_dir.resolve()
    measurement_root = args.measurement_root.resolve()
    analysis = json.loads((analysis_dir / "ANALYSIS.json").read_text(encoding="utf-8"))
    summary = json.loads((search_dir / "summary.json").read_text(encoding="utf-8"))
    per_column = read_csv(analysis_dir / "per_column_comparison.csv")
    per_method = read_csv(analysis_dir / "per_column_method_comparison.csv")
    depths = read_csv(analysis_dir / "exact_transform_depth_frontier.csv")
    expanded = read_csv(analysis_dir / "oracle_pipeline_expanded.csv")
    catalog_path = next(iter(sorted(measurement_root.glob("*/action_catalog.json"))))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    algorithms = {row["algorithm"]: row for row in summary["algorithms"]}
    best_sets = Counter(row["best_search_method"] for row in per_column)
    seed_diagnostics = analysis["rl_seed_diagnostics"]
    timing_summary = analysis["timing_summary"]
    timing_manifest = analysis["timing_manifest"]
    fresh_audit = analysis.get("fresh_audit", {"status": "NOT_RUN"})
    sparse19 = json.loads(args.sparse19_audit.resolve().read_text(encoding="utf-8"))
    if sparse19.get("status") != "PASS" or sparse19.get("representation", {}).get("logical_observation_fields") != 19:
        raise AssertionError("Sparse19 audit must PASS and declare 19 logical fields")
    external_sparse19 = (
        json.loads(args.external_sparse19_audit.resolve().read_text(encoding="utf-8"))
        if args.external_sparse19_audit
        else None
    )
    minimal6 = (
        json.loads(args.minimal6_summary.resolve().read_text(encoding="utf-8"))
        if args.minimal6_summary
        else None
    )
    prefix_batch = (
        json.loads(args.prefix_batch_summary.resolve().read_text(encoding="utf-8"))
        if args.prefix_batch_summary
        else None
    )
    exact_beam = (
        json.loads(args.exact_beam_summary.resolve().read_text(encoding="utf-8"))
        if args.exact_beam_summary
        else None
    )
    exact_beam_detail: dict[str, object] | None = None
    beam_selector_timing: dict[tuple[str, int], dict[str, float]] = {}
    if exact_beam:
        paired_rows = read_csv(args.exact_beam_summary.resolve().with_name("samples.csv"))
        paired_groups: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in paired_rows:
            paired_groups[(row["dataset"], int(row["column_index"]), row["algorithm"])][
                row["variant"]
            ].append(int(row["elapsed_ns"]) / 1_000_000.0)
        paired_speedups = []
        slow_groups = []
        speedups_by_dataset: dict[str, list[float]] = defaultdict(list)
        for key, values in paired_groups.items():
            beam_ms = percentile(values["beam29_selector_only"], 0.5)
            exact_ms = percentile(values["sparse19_unique_prefix_exact"], 0.5)
            speedup = 100.0 * (beam_ms - exact_ms) / beam_ms
            paired_speedups.append(speedup)
            speedups_by_dataset[key[0]].append(speedup)
            if speedup < 0.0:
                slow_groups.append((key, beam_ms, exact_ms, speedup))
        exact_beam_detail = {
            "faster": sum(value > 0.0 for value in paired_speedups),
            "slower": sum(value < 0.0 for value in paired_speedups),
            "slow_groups": slow_groups,
            "slow_by_dataset": Counter(key[0] for key, *_rest in slow_groups),
            "groups_by_dataset": Counter(key[0] for key in paired_groups),
            "median_reduction_by_dataset": {
                dataset: percentile(values, 0.5)
                for dataset, values in speedups_by_dataset.items()
            },
        }
        beam_by_column: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in paired_rows:
            if row["variant"] == "beam29_selector_only":
                beam_by_column[(row["dataset"], int(row["column_index"]))].append(
                    int(row["elapsed_ns"]) / 1_000_000.0
                )
        beam_selector_timing = {
            key: {
                "median": percentile(values, 0.5),
                "p25": percentile(values, 0.25),
                "p75": percentile(values, 0.75),
            }
            for key, values in beam_by_column.items()
        }
        if len(beam_selector_timing) != 20:
            raise AssertionError(("Beam selector timing columns", len(beam_selector_timing)))
    optimized_timing: dict[tuple[str, int, str], dict[str, float]] = {}
    if prefix_batch:
        timing_rows = read_csv(args.prefix_batch_summary.resolve().with_name("selector_samples.csv"))
        grouped_timing: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        for row in timing_rows:
            if row["variant"] == "sparse19_unique_prefix_exact":
                grouped_timing[
                    (row["dataset"], int(row["column_index"]), row["algorithm"])
                ].append(int(row["elapsed_ns"]) / 1_000_000.0)
        optimized_timing = {
            key: {
                "median": percentile(values, 0.5),
                "p25": percentile(values, 0.25),
                "p75": percentile(values, 0.75),
            }
            for key, values in grouped_timing.items()
        }
        if len(optimized_timing) != 40:
            raise AssertionError(("optimized timing groups", len(optimized_timing)))
    previous_analysis = (
        json.loads(args.previous_analysis.resolve().read_text(encoding="utf-8"))
        if args.previous_analysis
        else None
    )
    total_original = sum(int(row["original_bytes"]) for row in per_column)
    single_codec_total = sum(int(row["single_codec_bytes"]) for row in per_column)

    cards = [
        ("有界候选", f"{analysis['complete_candidates']:,}", "声明的 depth≤2 grammar 完整枚举"),
        ("最优单算法 pooled rate", f"{single_codec_total / total_original:.4%}", f"{single_codec_total:,} B / {total_original:,} B"),
        ("Beam-29 regret", f"{summary['beam29']['aggregate_oracle_regret_pct']:.4f}%", "相对声明的 depth≤2 oracle"),
        ("PPO 19字段 vs Beam", f"{algorithms['ppo']['median_gain_vs_beam29_pct']:+.4f}%", "3 seeds 中位，正值更好"),
        ("GRPO 19字段 vs Beam", f"{algorithms['grpo']['median_gain_vs_beam29_pct']:+.4f}%", "3 seeds 中位，正值更好"),
        ("当前 RL 推理状态", "19 字段", "16 个连续统计 + 3 个分类 lookup"),
        ("独立新审计", fresh_audit["status"], "有界结果可用；完整 codec contract 未通过"),
    ]
    card_html = "".join(
        f'<article class="card"><small>{esc(title)}</small><strong>{esc(value)}</strong><span>{esc(note)}</span></article>'
        for title, value, note in cards
    )

    method_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in per_method:
        method_by_key[(row["dataset"], int(row["column_index"]))].append(row)
    method_order = {"Beam-29": 0, "PPO": 1, "GRPO-style": 2}
    method_display = {
        "Beam-29": "Beam-29",
        "PPO": "PPO · 19字段",
        "GRPO-style": "GRPO-style · 19字段",
    }
    dataset_sections: list[str] = []
    for dataset in ("Air-Quality", "Metro-Traffic"):
        column_cards: list[str] = []
        dataset_columns = [row for row in per_column if row["dataset"] == dataset]
        for row in dataset_columns:
            key = (dataset, int(row["column_index"]))
            methods_html: list[str] = []
            for method in sorted(method_by_key[key], key=lambda item: method_order[item["method"]]):
                is_best = method["is_best_among_three"] == "1"
                seed = (
                    f"代表 seed {method['representative_seed']}"
                    if method["representative_seed"]
                    else f"查表 {int(method['candidate_count']):,} 个候选"
                )
                candidate_note = (
                    f"{int(method['candidate_count'])} unique / 512 rollouts，严格唯一-prefix复测"
                    if method["representative_seed"]
                    else seed
                )
                algorithm = {"PPO": "ppo", "GRPO-style": "grpo"}.get(method["method"])
                measured = (
                    beam_selector_timing.get((dataset, int(row["column_index"])))
                    if method["method"] == "Beam-29"
                    else optimized_timing.get((dataset, int(row["column_index"]), algorithm))
                )
                selection_median = (
                    measured["median"] if measured else float(method["selection_ms_median"])
                )
                selection_p25 = measured["p25"] if measured else float(method["selection_ms_p25"])
                selection_p75 = measured["p75"] if measured else float(method["selection_ms_p75"])
                methods_html.append(
                    f'<div class="method-row{" method-best" if is_best else ""}">'
                    f'<div class="method-name"><b>{esc(method_display[method["method"]])}</b>'
                    f'<span>{esc(seed)}</span>{"<em>本列三者最佳</em>" if is_best else ""}</div>'
                    f'<div class="metric rate"><small>压缩率</small><strong>{float(method["compression_rate"]):.5%}</strong>'
                    f'<span>{int(method["encoded_bytes"]):,} B</span></div>'
                    f'<div class="metric"><small>选择算子时间</small><strong>{selection_median:.3f} ms</strong>'
                    f'<span>IQR {selection_p25:.3f} 至 {selection_p75:.3f}<br>{esc(candidate_note)}</span></div>'
                    f'<div class="metric"><small>实际压缩时间</small><strong>{float(method["encode_ms_median"]):.3f} ms</strong>'
                    f'<span>IQR {float(method["encode_ms_p25"]):.3f} 至 {float(method["encode_ms_p75"]):.3f}</span></div>'
                    f'<div class="method-pipeline"><small>选出的算子组合</small>{pipeline_html(method["pipeline"])}</div>'
                    '</div>'
                )
            column_cards.append(
                '<article class="column-card">'
                f'<header><div><small>{esc(dataset)} · column {int(row["column_index"])}</small>'
                f'<h3>{esc(row["column_name"])}</h3></div><div class="input-size">原始 <b>{int(row["original_bytes"]):,} B</b></div></header>'
                '<div class="oracle">'
                '<div class="oracle-title">声明空间最优压缩组合</div>'
                f'<div class="oracle-metric"><small>最优压缩率</small><b>{float(row["oracle_rate"]):.5%}</b>'
                f'<span>{int(row["oracle_bytes"]):,} B</span></div>'
                f'<div class="oracle-metric"><small>实际压缩时间</small><b>{float(row["oracle_encode_ms_median"]):.3f} ms</b>'
                f'<span>IQR {float(row["oracle_encode_ms_p25"]):.3f} 至 {float(row["oracle_encode_ms_p75"]):.3f}</span></div>'
                f'<div class="oracle-pipeline"><small>最优算子组合</small>{pipeline_html(row["oracle_pipeline"])}</div>'
                '</div>'
                '<div class="single-baseline">'
                '<div class="single-title">最优单个压缩算法</div>'
                f'<div class="single-metric"><small>压缩率</small><b>{float(row["single_codec_rate"]):.5%}</b>'
                f'<span>{int(row["single_codec_bytes"]):,} B</span></div>'
                f'<div class="single-metric"><small>实际压缩时间</small><b>{float(row["single_codec_encode_ms_median"]):.3f} ms</b>'
                f'<span>IQR {float(row["single_codec_encode_ms_p25"]):.3f} 至 {float(row["single_codec_encode_ms_p75"]):.3f}</span></div>'
                f'<div class="single-pipeline"><small>单算法 pipeline</small>{pipeline_html(row["single_codec_pipeline"])}</div>'
                '</div>'
                f'<div class="search-winner">三种搜索最佳：<b>{esc(row["best_search_method"])}</b></div>'
                f'{"".join(methods_html)}</article>'
            )
        dataset_sections.append(
            f'<section id="{dataset.lower().replace("-", "-")}" class="dataset-section">'
            f'<h2>{esc(dataset)}：每列单算法 / 有界最优 / 三种搜索</h2>'
            f'<p class="meta">共 {len(dataset_columns)} 列。压缩率越低越好，浅绿色行表示该列三种搜索中的最小字节结果。</p>'
            f'{"".join(column_cards)}</section>'
        )

    oracle_timing_card = (
        '<article><b>有界最优压缩组合</b><span>选择：离线完整枚举（不与查表直接对比）</span>'
        f'<span>实际压缩 {timing_summary["Bounded oracle"]["encode_ms_median_across_columns"]:.3f} ms</span></article>'
    )
    single_timing_card = (
        '<article><b>最优单个压缩算法</b>'
        f'<span>选择：离线比较 {timing_summary["Best single codec"]["candidate_count"]} 个单算法候选</span>'
        f'<span>实际压缩 {timing_summary["Best single codec"]["encode_ms_median_across_columns"]:.3f} ms</span></article>'
    )
    optimized_method_median = {
        method: percentile(
            [
                values["median"]
                for (dataset, column, algorithm), values in optimized_timing.items()
                if algorithm == {"PPO": "ppo", "GRPO-style": "grpo"}.get(method)
            ],
            0.5,
        )
        for method in ("PPO", "GRPO-style")
    } if optimized_timing else {}
    if beam_selector_timing:
        optimized_method_median["Beam-29"] = percentile(
            [values["median"] for values in beam_selector_timing.values()], 0.5
        )
    timing_cards = oracle_timing_card + single_timing_card + "".join(
        f'<article><b>{esc(method_display[method])}</b><span>选择算子 {optimized_method_median.get(method, timing_summary[method]["selection_ms_median_across_columns"]):.3f} ms</span>'
        f'<span>实际压缩 {timing_summary[method]["encode_ms_median_across_columns"]:.3f} ms</span></article>'
        for method in ("Beam-29", "PPO", "GRPO-style")
    )

    depth_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in depths:
        depth_by_key.setdefault((row["dataset"], row["column_name"]), []).append(row)
    depth_html = []
    for (dataset, column), rows in depth_by_key.items():
        ordered = sorted(rows, key=lambda row: int(row["exact_transform_count"]))
        cells = "".join(
            f"<td>{float(row['compression_rate']):.5%}<br><small>{esc(row['final_bytes'])} B</small></td>"
            for row in ordered
        )
        depth_html.append(f"<tr><td>{esc(dataset)}</td><td>{esc(column)}</td>{cells}</tr>")

    expanded_html = []
    for row in expanded:
        parameters = []
        for key, label in (
            ("block_size", "block"),
            ("dictionary_size", "dict"),
            ("window_size", "window"),
        ):
            if int(row[key]):
                parameters.append(f"{label}={int(row[key]):,}")
        if row["exception_stream"] == "1":
            parameters.append("exception stream")
        if row["inner_framing"] != "none":
            parameters.append(f"frame={row['inner_framing']}")
        if row["inner_checksum"] != "none":
            parameters.append(f"checksum={row['inner_checksum']}")
        expanded_html.append(
            "<tr>"
            f"<td>{esc(row['dataset'])}</td><td>{esc(row['column_name'])}</td>"
            f"<td>{esc(row['position'])}</td><td><code>{esc(row['action'])}</code></td>"
            f"<td>{esc(row['conceptual_stages'])}</td><td>{esc(', '.join(parameters) or 'none')}</td>"
            f"<td><code>{esc(row['implementation_source'])}</code></td></tr>"
        )

    action_counts = catalog["counts"]
    cap = int(analysis["oracle_at_transform_cap"])
    budget = analysis["budget_diagnostic"]
    winner_text = "，".join(f"{name} {count} 列" for name, count in sorted(best_sets.items()))
    dataset_rows = []
    for algorithm, label in (("ppo", "PPO · 19字段"), ("grpo", "GRPO-style · 19字段")):
        combined = seed_diagnostics[algorithm]
        for dataset, values in combined["per_dataset"].items():
            saving = float(values["saving_vs_beam29_pct"])
            comparison = f"节省 {saving:.3f}%" if saving >= 0 else f"多 {-saving:.3f}%"
            dataset_rows.append(
                "<tr>"
                f"<td>{label}</td><td>{esc(dataset)}</td>"
                f"<td>{int(values['min_bytes']):,} 至 {int(values['max_bytes']):,} B</td>"
                f"<td>{float(values['median_bytes']):,.0f} B</td>"
                f"<td>{int(values['beam29_bytes']):,} B</td><td>{comparison}</td></tr>"
            )
    representation = sparse19["representation"]
    previous_timing = previous_analysis.get("timing_summary", {}) if previous_analysis else {}
    timing_change_rows = []
    timing_change_pct: dict[str, float] = {}
    for method in ("PPO", "GRPO-style"):
        current_ms = float(timing_summary[method]["selection_ms_median_across_columns"])
        old_ms = float(previous_timing.get(method, {}).get("selection_ms_median_across_columns", current_ms))
        change_pct = 100.0 * (current_ms - old_ms) / old_ms if old_ms else 0.0
        timing_change_pct[method] = change_pct
        timing_change_rows.append(
            f"<tr><td>{esc(method_display[method])}</td><td>{old_ms:.3f} ms</td>"
            f"<td>{current_ms:.3f} ms</td><td>{change_pct:+.3f}%</td></tr>"
        )
    timing_change_note = "，".join(
        f"{method} {'变慢' if change > 0.0 else '变快'}约 {abs(change):.2f}%"
        for method, change in timing_change_pct.items()
    )
    external_note = (
        f"另一组 10 个 PPO policy 的扩展审计覆盖 38 列，含 Bird-migration 与 meteoblue："
        f"logit 误差 ≤ {float(external_sparse19['max_dense53_vs_sparse19_logit_abs_error']):.3g}，"
        f"外部 proposal {int(external_sparse19['external_exact_proposal_rows'])}/"
        f"{int(external_sparse19['external_proposal_rows'])} 一致。"
        if external_sparse19
        else "未提供外部列折叠审计。"
    )
    minimal6_note = (
        f"继续压到 6 个连续字段的外部实验未过门槛：18 个外部列的 column-median 聚合惩罚为 "
        f"{float(minimal6['aggregate_column_median_penalty_pct']):+.3f}%，最大列惩罚 "
        f"{float(minimal6['max_column_median_penalty_pct']):+.3f}%，95% bootstrap 上界 "
        f"{float(minimal6['paired_column_bootstrap_95pct']['high']):+.3f}%。因此当前不把模型再删到 6 字段。"
        if minimal6
        else "未提供 6 字段外部门槛结果。"
    )
    prefix_batch_html = ""
    if prefix_batch:
        timing = prefix_batch["selector_timing"]["versus_full60_scalar"]
        variants = timing["variants"]
        labels = {
            "full60_scalar": ("Full60 逐轨迹", "原始采样顺序与原始 proposal 集"),
            "sparse19_cached_fields_scalar": ("19字段缓存输入", "逐轨迹，proposal 集严格一致"),
            "sparse19_cached_scalar": ("19字段缓存 preactivation", "逐轨迹，proposal 集严格一致"),
            "sparse19_unique_prefix_exact": ("唯一-prefix 严格版", "复用 prefix 概率，旧 RNG 顺序与 proposal 集严格一致"),
            "full60_batch_prefix": ("Full60 按 step 批量", "固定随机数矩阵的语义参考"),
            "sparse19_batch_prefix": ("19字段按 step 批量", "同分布但不逐路径复现旧 RNG 顺序"),
        }
        rows = []
        for key, (label, guarantee) in labels.items():
            value = variants[key]
            low, high = value["paired_speedup_pct_bootstrap_median_ci95"]
            rows.append(
                f"<tr><td>{esc(label)}</td><td>{float(value['group_median_ms']):.3f} ms</td>"
                f"<td>{float(value['paired_speedup_pct_median']):+.2f}%</td>"
                f"<td>{float(low):+.2f}% 至 {float(high):+.2f}%</td><td>{esc(guarantee)}</td></tr>"
            )
        gates = prefix_batch["correctness_gates"]
        merging = prefix_batch["prefix_merging"]
        exact_merging = merging["pathwise_exact"]
        batch_merging = merging["step_batch"]
        quality = prefix_batch["quality_under_iid_matrix_schedule"]["overall"]
        prefix_batch_html = (
            '<h3>最终配对复测</h3>'
            '<div class="panel"><table><thead><tr><th>实现</th><th>40组跨列中位</th>'
            '<th>相对 Full60 配对延迟降低</th><th>bootstrap 95% 区间</th><th>语义保证</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
            f'<p class="meta">每组重复 {int(prefix_batch["selector_timing"]["repeats"])} 次并在同一进程内随机化顺序。'
            f'严格唯一-prefix版在 {int(gates["fresh_full60_vs_unique_prefix_proposal_sets_exact"])}/{int(gates["total_rows"])} 条评估上'
            '逐 proposal 集复现本次新鲜重建的 Full60 scalar；归档文件只用于核对选中 pipeline 与 proposal 数。'
            f'严格版每次 512 rollout 的轨迹-prefix 访问中位为 {float(exact_merging["trajectory_prefix_evaluations_median"]):,.1f} 次（不含 greedy decode），'
            f'实际只需计算 {float(exact_merging["unique_prefix_evaluations_median"]):.0f} 个唯一 prefix。'
            f'这两个数均来自严格版；step-batch 的对应中位为 {float(batch_merging["trajectory_prefix_evaluations_median"]):,.1f}/'
            f'{float(batch_merging["unique_prefix_evaluations_median"]):.0f}。'
            f'一次性 actor 缓存初始化中位 {float(prefix_batch["cache_initialization"]["median_ms"]):.3f} ms，未计入暖缓存逐列选择时间；'
            '该数只覆盖 actor-specific inference cache，不覆盖 fold 级标准化、prefix mask 与转移表构造。'
            'bootstrap 以 40 个列×算法组重采样，同一物理列在 PPO/GRPO 下相关，因此不是硬件或部署不确定性区间。</p>'
            f'<div class="notice"><b>最终公平速度结论采用“唯一-prefix 严格版”。</b> '
            f'按 step 批量版虽相对 Full60 更快，但固定随机数矩阵改变了旧 RNG 的逐路径分配：120 行中 '
            f'仅 {int(gates["step_batch_vs_scalar_proposal_sets_exact"])}/{int(gates["total_rows"])} 行 proposal 集与 scalar 相同；'
            f'{int(quality["batch_better"])} 行更好、{int(quality["batch_equal"])} 行相同、'
            f'{int(quality["batch_worse"])} 行更差，总字节变化 {float(quality["total_improvement_pct"]):+.3f}%。'
            '因此它只证明同分布批量推理的速度，不用于同压缩结果的最终提速主张。</div>'
        )
        if exact_beam and exact_beam_detail:
            slow_groups = exact_beam_detail["slow_groups"]
            worst_text = "无"
            if slow_groups:
                key, beam_ms, exact_ms, speedup = min(slow_groups, key=lambda item: item[3])
                worst_text = (
                    f"{key[0]} column {key[1]} {key[2].upper()}："
                    f"严格版 {exact_ms:.3f} ms，Beam {beam_ms:.3f} ms"
                )
            slow_distribution = "，".join(
                f"{dataset} {int(exact_beam_detail['slow_by_dataset'][dataset])}/"
                f"{int(exact_beam_detail['groups_by_dataset'][dataset])}"
                for dataset in sorted(exact_beam_detail["groups_by_dataset"])
            )
            prefix_batch_html += (
                '<h3>与 Beam-29 纯选择器的同进程配对</h3>'
                f'<p class="meta">{int(exact_beam["repeats"])} 次 AB/BA 配对中，Beam-29 纯选择器的 '
                f'{int(exact_beam["groups"])} 组跨列中位为 '
                f'{float(exact_beam["beam29_group_median_ms"]):.3f} ms，严格唯一-prefix版为 '
                f'{float(exact_beam["exact_group_median_ms"]):.3f} ms；配对中位 '
                f'{float(exact_beam["paired_exact_speedup_vs_beam_pct_median"]):.2f}% 的延迟降低。'
                f'{int(exact_beam_detail["faster"])}/{int(exact_beam["groups"])} 组更快，'
                f'{int(exact_beam_detail["slower"])}/{int(exact_beam["groups"])} 组更慢。'
                f'分数据集的配对中位延迟降低率为 Air-Quality '
                f'{float(exact_beam_detail["median_reduction_by_dataset"]["Air-Quality"]):+.2f}%，Metro-Traffic '
                f'{float(exact_beam_detail["median_reduction_by_dataset"]["Metro-Traffic"]):+.2f}%。'
                f'较慢组分布为 {esc(slow_distribution)}；最慢一组是 {esc(worst_text)}。'
                'Beam 计时明确排除了只用于事后报告 regret 的全表 oracle 扫描。</p>'
            )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>19字段 PPO/GRPO 与 Beam-29：逐列压缩率、时间与算子组合</title>
<style>
:root{{--ink:#172133;--muted:#627087;--line:#dce2eb;--paper:#fff;--bg:#f3f6fa;--blue:#155eef;--orange:#d45d0b;--green:#137a4b;--green-bg:#effaf4}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
main{{max-width:1440px;margin:auto;padding:40px 28px 72px}} h1{{font-size:36px;line-height:1.15;margin:0 0 12px}} h2{{margin:42px 0 14px;font-size:22px}} p{{max-width:980px}} p,.notice{{overflow-wrap:anywhere}} .lead{{font-size:18px;color:var(--muted)}}
.notice{{border-left:5px solid var(--orange);background:#fff4eb;padding:14px 18px;margin:24px 0;border-radius:8px}} .notice ul{{margin:8px 0 0;padding-left:20px}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 4px 18px #1721330a}} .card small,.card span{{display:block;color:var(--muted)}} .card strong{{display:block;font-size:27px;margin:8px 0}}
.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;overflow:auto}} table{{border-collapse:collapse;width:100%;min-width:980px}} th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#f8fafc;font-size:12px;text-transform:uppercase;color:var(--muted)}}
code{{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef3fb;padding:2px 5px;border-radius:5px;white-space:nowrap}} .pipe{{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-top:5px}} .pipe i{{font-style:normal;color:var(--blue)}} .meta{{color:var(--muted)}}
.jump-links{{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}} .jump-links a{{color:var(--blue);background:#fff;border:1px solid #b9cdfb;border-radius:999px;padding:7px 13px;text-decoration:none;font-weight:650}}
.timing-explain{{background:#eaf2ff;border:1px solid #bdd0f7;border-radius:12px;padding:15px 18px;margin:22px 0}} .timing-explain p{{margin:4px 0;max-width:none}}
.timing-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:15px 0 30px}} .timing-strip article{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}} .timing-strip b,.timing-strip span{{display:block}} .timing-strip span{{color:var(--muted);margin-top:4px}}
.model19{{background:linear-gradient(135deg,#f5f9ff,#f7fff9);border:1px solid #bcd1f4;border-radius:16px;padding:20px;margin:30px 0}} .model19 h2{{margin:0 0 10px}} .model19 h3{{margin:24px 0 9px}} .field-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:17px 0}} .field-grid article{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:15px}} .field-grid b,.field-grid span{{display:block}} .field-grid b{{font-size:18px}} .field-grid span{{color:var(--muted);margin-top:5px}}
.dataset-section{{scroll-margin-top:18px}} .column-card{{background:#fff;border:1px solid var(--line);border-radius:16px;margin:16px 0;overflow:hidden;box-shadow:0 5px 20px #1721330a}} .column-card>header{{display:flex;justify-content:space-between;gap:18px;align-items:end;padding:18px 20px 14px;border-bottom:1px solid var(--line)}} .column-card h3{{margin:2px 0 0;font-size:22px}} .column-card header small,.input-size{{color:var(--muted)}}
.oracle{{display:grid;grid-template-columns:170px 135px 210px minmax(280px,1fr);gap:16px;align-items:center;padding:14px 20px;background:#f7f2ff;border-bottom:1px solid #e5dcfa}} .oracle-title{{color:#6b46a5;font-weight:750}} .oracle-metric small,.oracle-metric span,.oracle-pipeline small{{display:block;color:var(--muted);font-size:12px}} .oracle-metric b{{display:block;font-size:18px;color:#5c35a0}} .oracle-pipeline .pipe{{margin-top:5px}}
.single-baseline{{display:grid;grid-template-columns:170px 135px 210px minmax(280px,1fr);gap:16px;align-items:center;padding:14px 20px;background:#fff7e8;border-bottom:1px solid #f1d9ad}} .single-title{{color:#9a5a00;font-weight:750}} .single-metric small,.single-metric span,.single-pipeline small{{display:block;color:var(--muted);font-size:12px}} .single-metric b{{display:block;font-size:18px;color:#9a5a00}} .single-pipeline .pipe{{margin-top:5px}}
.search-winner{{padding:9px 20px;background:#f8fafc;color:var(--muted);border-bottom:1px solid var(--line)}} .search-winner b{{color:var(--ink)}}
.method-row{{display:grid;grid-template-columns:145px 130px 190px 190px minmax(300px,1fr);gap:14px;align-items:start;padding:15px 20px;border-bottom:1px solid var(--line)}} .method-row:last-child{{border-bottom:0}} .method-best{{background:var(--green-bg);box-shadow:inset 4px 0 var(--green)}}
.method-name b,.metric strong{{display:block;font-size:17px}} .method-name span,.metric span,.method-pipeline small,.metric small{{display:block;color:var(--muted);font-size:12px}} .method-name em{{display:inline-block;margin-top:7px;background:var(--green);color:#fff;border-radius:999px;padding:2px 8px;font-size:11px;font-style:normal}} .rate strong{{color:var(--blue);font-size:19px}} .method-pipeline .pipe{{margin-top:6px}}
footer{{margin-top:36px;color:var(--muted);font-size:13px}} @media(max-width:1200px){{.timing-strip{{grid-template-columns:1fr 1fr}}.oracle,.single-baseline{{grid-template-columns:1fr 1fr}}.oracle-title,.oracle-pipeline,.single-title,.single-pipeline{{grid-column:1/-1}}.method-row{{grid-template-columns:120px 120px 1fr 1fr}}.method-pipeline{{grid-column:1/-1}}}} @media(max-width:900px){{main{{padding:26px 14px}}h1{{font-size:29px}}.grid{{grid-template-columns:1fr 1fr}}.field-grid{{grid-template-columns:1fr}}}} @media(max-width:620px){{.grid,.timing-strip{{grid-template-columns:1fr}}.column-card>header{{align-items:start;flex-direction:column;gap:5px}}.method-row{{grid-template-columns:1fr 1fr;padding:14px}}.method-name,.method-pipeline{{grid-column:1/-1}}.oracle,.single-baseline{{grid-template-columns:1fr;padding:13px 14px}}.oracle-title,.oracle-pipeline,.single-title,.single-pipeline{{grid-column:auto}}}}
</style></head><body><main>
<h1>19字段 PPO/GRPO 与 Beam-29 对比</h1>
<p class="lead">当前冻结 PPO/GRPO policy 已改为 19 个逻辑字段推理：16 个连续列统计，加 3 个保持顺序语义的分类 lookup。Beam-29 与两种 RL 方法继续共用可独立解码的真实字节表。动作空间含 {action_counts['transforms']} 个 transform、{action_counts['packers']} 个 packer、{action_counts['compressors']} 个 compressor 变种，共 {action_counts['total_actions']} 个物理动作。</p>
<div class="notice"><b>旧 28 标签结论已失效；新实验的独立审计为 {esc(fresh_audit['status'])}。</b> 有界 benchmark 数值可用，但不能据此声称 arbitrary-int64 codec contract、全 frame 防篡改或生产格式兼容性已经通过。新目标计入 block/dictionary 参数、异常值与 escape 流、transform metadata、action table、容器 framing、仅在配置存在时的内部 checksum，以及校验还原值的 PV2F CRC32。</div>
<section class="grid">{card_html}</section>
<section id="current-model" class="model19">
<h2>当前模型到底是哪 19 个字段</h2>
<p><b>部署与评估推理已经是 19 个逻辑字段，训练过程仍沿用原 Full60 policy，再做严格等价折叠。</b> 这样不会把上一轮 53D 重训练的 Metro 退化混入当前结果。折叠后的第一层保存为 53 个权重槽，但运行时输入不是 53 个独立字段。</p>
<div class="field-grid">
<article><b>{representation['continuous_fields']} 个连续字段</b><span>原 17 个列统计中删除 feature_8；它在当前审计列上与 feature_6 完全重复，权重合并进 feature_6。</span></article>
<article><b>3 个分类字段</b><span>有序 transform slot 1、有序 transform slot 2、packer。三者均为独立 lookup，类别 ID 不作为有大小关系的数字输入。</span></article>
<article><b>{representation['dense_equivalent_dimensions']} 个权重槽</b><span>phase 与 depth one-hot 可由三个 prefix 字段严格推导，直接折进 bias 和 lookup 权重；这是实现维度，不是额外观测字段。</span></article>
</div>
<p>当前页面使用的 PPO/GRPO 共 {int(sparse19['policies'])} 个冻结策略。对 {int(sparse19['legal_nonterminal_prefixes']):,} 个合法非终止 prefix 做全覆盖 logit 比较，最大绝对误差为 {float(sparse19['max_full60_vs_sparse19_logit_abs_error']):.3g}；原始跨数据集评估 {int(sparse19['development_exact_proposal_rows'])}/{int(sparse19['development_proposal_rows'])} 条的 proposal 集、选中 pipeline、字节数与归档记录完全一致。{esc(external_note)}</p>
<div class="notice"><b>所以本页压缩率不因 60→19 而变化。</b> 这是同一个已训练函数的语义保持重编码，不是重新训练后得到的新压缩收益。PPO 的 3-seed 中位总字节仍为 {seed_diagnostics['ppo']['combined_median_bytes']:,.0f} B，GRPO-style 仍为 {seed_diagnostics['grpo']['combined_median_bytes']:,.0f} B。</div>
<h3>上一版未优化 19 字段直译测时</h3>
<div class="panel"><table><thead><tr><th>方法</th><th>旧 Full60 测时</th><th>19 字段实测</th><th>变化，正值表示更慢</th></tr></thead><tbody>{''.join(timing_change_rows)}</tbody></table></div>
<p class="meta">该版使用 host <code>{esc(timing_manifest['host'])}</code>，状态为 <code>{esc(timing_manifest.get('observation', 'unknown'))}</code>，每列做 {timing_manifest['selection_repeats']} 次选择。PPO 的跨列中位为 {timing_summary['PPO']['selection_ms_median_across_columns']:.3f} ms，GRPO-style 为 {timing_summary['GRPO-style']['selection_ms_median_across_columns']:.3f} ms。相对表中的旧 Full60 结果，{esc(timing_change_note)}；原因是 512 次 Python rollout 仍重复做列标准化、prefix 解析和小数组构造。它是问题定位结果，不再作为最终实现的速度结论。计时不含特征抽取、模型加载与候选真实压缩。</p>
{prefix_batch_html}
<p>{esc(minimal6_note)}</p>
</section>
<p>三种搜索逐列最优集合（等号表示同字节并列）：{esc(winner_text)}。有界 oracle 有 <b>{cap}/{len(per_column)}</b> 列落在 2-transform 上限，见深度边界表；当前结果不能证明继续增加外层 transform 无效。</p>
<p class="meta">预算诊断：Beam-29 每列查表 {budget['beam_candidate_evaluations_min']} 至 {budget['beam_candidate_evaluations_max']} 个候选，中位 {budget['beam_candidate_evaluations_median']:.0f}。PPO/GRPO-style 各采样 512 次，但 unique proposal 中位仅 {budget['ppo_unique_proposals_median']:.0f}/{budget['grpo_unique_proposals_median']:.0f}。训练 lookup、测试 rollout 与 Beam lookup 预算并不匹配；held-out 也是 frozen-policy proposal search 后用测试字节表选优，不是纯 greedy zero-shot。因此这里只能说当前配置发生 proposal collapse 且 combined-byte 中位数未胜 Beam，不能声称 PPO/GRPO 本身无效。</p>

<nav class="jump-links"><a href="#current-model">看 19 字段模型</a><a href="#air-quality">直接看 Air-Quality</a><a href="#metro-traffic">直接看 Metro-Traffic</a><a href="#dataset-summary">看数据集汇总</a></nav>
<div class="timing-explain"><p><b>下面每一列按固定顺序列出：有界最优组合、最优单个压缩算法、Beam-29、PPO 19字段、GRPO-style 19字段。每项都给出压缩率、实际压缩时间和完整 pipeline；三种搜索另外给出选择算子时间。</b></p><p>单算法定义为完整 PV2F frame 中恰好一个非平凡 codec action，可以是一个 transform、一个 packer 或一个通用 compressor；<code>raw.i64</code> 与 <code>none</code> 只负责承载和终止。每列离线比较 {timing_summary['Best single codec']['candidate_count']} 个这种候选，不把取最小的离线过程当作在线选择时间。有界 oracle 同样是每列 18,850 个候选的离线完整枚举。三种搜索的选择算子时间是在基准环境中预先加载策略、19字段推理状态和候选字节表后的查表搜索时间，不包含真实压缩所有候选。实际压缩时间从 normalized values 开始，包含 transform、packer、compressor 和 PV2F framing，不包含解码验证。逐列 Beam 时间取纯选择器 AB/BA 文件中每列 2 个算法配对组各 {int(exact_beam['repeats']) if exact_beam else timing_manifest['selection_repeats']} 次的合并中位，PPO/GRPO 严格唯一-prefix时间来自本轮 {int(prefix_batch['selector_timing']['repeats']) if prefix_batch else timing_manifest['selection_repeats']} 次六实现随机顺序重复；五种 pipeline 编码均重复 {timing_manifest['encode_repeats']} 次。页面显示中位数和 IQR。</p></div>
<div class="timing-strip">{timing_cards}</div>

{''.join(dataset_sections)}

<h2 id="dataset-summary">跨数据集结果与 seed 波动</h2><div class="panel"><table><thead><tr><th>方法</th><th>测试数据集</th><th>3-seed 范围</th><th>中位数</th><th>Beam-29</th><th>中位差异</th></tr></thead><tbody>{''.join(dataset_rows)}</tbody></table></div>
<p class="meta">Combined-byte 总量：PPO 为 {seed_diagnostics['ppo']['combined_min_bytes']:,} 至 {seed_diagnostics['ppo']['combined_max_bytes']:,} B，中位 {seed_diagnostics['ppo']['combined_median_bytes']:,.0f} B；GRPO-style 为 {seed_diagnostics['grpo']['combined_min_bytes']:,} 至 {seed_diagnostics['grpo']['combined_max_bytes']:,} B，中位 {seed_diagnostics['grpo']['combined_median_bytes']:,.0f} B。只有两个 dataset unit，seed 反映优化器随机性，不构成数据集级置信区间。</p>

<h2>精确 transform 深度边界</h2><p class="meta">同一列分别限制为恰好 0、1、2 个外层 transform；每格均在全部 packer/compressor 变种中完整枚举。</p><div class="panel"><table><thead><tr><th>数据集</th><th>列</th><th>0 transform</th><th>1 transform</th><th>2 transforms</th></tr></thead><tbody>{''.join(depth_html)}</tbody></table></div>

<h2>Oracle 宏动作的内部算子与物理参数</h2><div class="panel"><table><thead><tr><th>数据集</th><th>列</th><th>位置</th><th>物理动作变种</th><th>内部 stages</th><th>参数 / 流 / framing</th><th>实现来源</th></tr></thead><tbody>{''.join(expanded_html)}</tbody></table></div>

<h2>口径与独立审计边界</h2><p>在这 14 个 Air-Quality 与 6 个 Metro-Traffic 有限 normalized-integer 列上，我们完整枚举了声明的 PV2F depth≤2 grammar：0 至 2 个不重复的精确 transform、一个 packer、一个 compressor，每列 18,850 个候选。所有候选在记录环境中完成 in-run round-trip，<code>final_bytes</code> 是真实序列化 PV2F frame 长度。Beam-29 为 273,436 B，有界 oracle 为 264,323 B；自定义 PPO 与 GRPO-style 的 3-seed combined-byte 中位数分别比 Beam 多 4.55% 与 6.62%。这不是 matched-unique-candidate 比较，且存在 proposal collapse，所以不能推出 PPO/GRPO 本质上更差。</p>
<p>最终字节 = outer header + action table + metadata length prefixes + transform metadata + generic-compressed packer payload + outer CRC32。通用压缩器的 frame 与仅在配置存在时的内部 checksum 已包含在 compressed payload 中。CRC32 校验的是还原值，不认证整个 frame/action table。exception/escape 字节已计入聚合 metadata/payload，但当前没有逐旁路流单独归因。</p>
<div class="notice"><b>独立审计未通过的 codec contract：</b><ul><li>有效性仅覆盖两个 benchmark 的实际取值范围，不覆盖任意 int64；INT64_MIN/MAX 可触发部分 project packer 失败。</li><li>全 377,000 候选都做过 in-run round-trip；表格/hash/算术全审计，但独立 post-hoc encode/decode 只覆盖 134 条 claim-bearing pipeline。</li><li>旧 winner 差异是记录性比较；旧 CSV 没有归档/哈希，不能当作独立审计的 legacy evidence。</li></ul></div>
<p>搜索深度最多为两个外层 transform，再接一个 packer 和一个 compressor。PFOR、LZMA2、zstd 等宏动作会在上表展开其内部 stages，因此宏动作数不等于实际编码阶段数。Dictionary/outlier exception 等旁路流以 varint metadata 计费，但 terminal compressor 只压缩主 packer payload；生产容器可能采用不同的分流压缩策略。</p>
<footer>主张边界：observed-range normalized-integer standalone research prototype；不包含生产 Parquet/BOS 兼容性、旁路流压缩策略、kernel 吞吐、随机访问、查询下推或原始缺失值语义。</footer>
</main></body></html>"""
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(document, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    run(parse_args())
