#!/usr/bin/env python3
"""Export a read-only JSON snapshot of a weight-grid experiment to report/.

No runner, codec, database, model or frame payload is imported or executed.
Only this script's REPORT.html, aggregate.csv, per_column.csv and REPORT_META.json
are written, within the requested experiment's report subdirectory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "integer-tenth-weight-grid-rl-v1"
FORMAL_COLUMNS = {"Air-Quality": 14, "Metro-Traffic": 6}
FORMAL_SEEDS = (20260906, 20260907, 20260908)
GRID = tuple((a, b, 10-a-b) for a in range(11) for b in range(11-a))
METHODS = {
    "ppo": "PPO", "grpo": "GRPO", "all38_J_best": "38 项固定目录：按本权重 J 选优",
    "fixed5_rho_best": "5 个单算法：按 ρ 选优", "fixed5_J_best": "5 个单算法：按本权重 J 选优",
    "complete30_rho_best": "30 项单算法及组合：按 ρ 选优", "complete30_J_best": "30 项单算法及组合：按本权重 J 选优",
    "old_b4_beam": "历史 B4 Beam 方案（仅重测）", "old_open_beam": "历史 Open Beam 方案（仅重测）",
}
MAIN = ("ppo", "grpo", "all38_J_best")
COLORS = {"ppo": "#146cb3", "grpo": "#bc4b12", "all38_J_best": "#656975"}
COST_FIELDS = ("train_seconds", "search_seconds", "optimizer_seconds", "refinement_seconds", "run_seconds")


def wid(weight):
    a, b, c = weight
    return f"a{a:02d}_b{b:02d}_c{c:02d}"


def weight_text(weight):
    return "/".join(f"{x/10:.1f}" for x in weight)


def escape(value):
    return html.escape(str(value), quote=True)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=3):
    return f"{value:,.{digits}f}" if finite(value) else "—"


class Snapshot:
    def __init__(self, root):
        self.root = root
        self.warnings = []
        self.sources = {}

    def read(self, path, required=False):
        if not path.exists():
            if required:
                self.warnings.append(f"缺少 {path.relative_to(self.root)}；按 partial 输出。")
            return None
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("expected JSON object")
            self.sources[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
            return value
        except (OSError, ValueError) as exc:
            self.warnings.append(f"无法读取 {path.relative_to(self.root)}：{exc}")
            return None


def summarize_seed_metrics(row):
    result = {"selection_seed_count": len(row.get("seed_metrics", []))}
    for source, label in (("rho", "rho"), ("E", "encode_seconds"), ("D", "decode_seconds"), ("cost", "J")):
        values = [m[source] for m in row.get("seed_metrics", []) if finite(m.get(source))]
        result[f"seed_selection_{label}_mean"] = statistics.mean(values) if values else None
        result[f"seed_selection_{label}_std"] = statistics.stdev(values) if len(values) > 1 else None
    return result


def collect(snapshot):
    root = snapshot.root
    manifest = snapshot.read(root / "MANIFEST.json", required=True) or {}
    summary = snapshot.read(root / "SUMMARY.json") or {}
    progress = snapshot.read(root / "PROGRESS.json") or snapshot.read(root / "STATUS.json") or {}
    failure = snapshot.read(root / "FAILURE.json") or {}
    identity = manifest.get("experiment_identity") or summary.get("experiment_identity")
    if summary and summary.get("experiment_identity") != identity:
        snapshot.warnings.append("SUMMARY identity 不匹配，未采用其完成状态或耗时。")
        summary = {}
    config = manifest.get("config", summary.get("config", {}))
    columns = {c["key"]: c for c in manifest.get("columns", []) if isinstance(c, dict) and "key" in c}
    weights = [tuple(w) for w in config.get("weights_tenths", [])]
    if any(w not in GRID for w in weights) or len(weights) != len(set(weights)):
        snapshot.warnings.append("配置权重不是合法的无重复整数十分位网格。")
        weights = sorted(set(w for w in weights if w in GRID))
    seeds = list(config.get("seeds", []))
    expected = set(itertools.product(columns, [wid(w) for w in weights], ("ppo", "grpo"), seeds))
    runs = {}
    for path in sorted((root / "runs").glob("*/*/*/RESULT.json")):
        row = snapshot.read(path)
        if not row:
            continue
        key = (row.get("key"), row.get("weight_id"), row.get("method"), row.get("seed"))
        valid = (row.get("schema") == SCHEMA and row.get("experiment_identity") == identity and key in expected
                 and row.get("weight_tenths") in [list(w) for w in weights]
                 and wid(row["weight_tenths"]) == row.get("weight_id"))
        col = columns.get(row.get("key"), {})
        budget = row.get("budget", {})
        valid = valid and row.get("input_sha256") == col.get("input_sha256")
        valid = valid and budget.get("initial_actor_sha256") and budget.get("final_actor_sha256")
        valid = valid and budget.get("initial_actor_sha256") != budget.get("final_actor_sha256")
        valid = valid and budget.get("rollouts") == config.get("iterations", 0)*config.get("episodes", 0)
        valid = valid and all(finite(row.get(k)) and row[k] >= 0 for k in COST_FIELDS)
        if not valid or key in runs:
            snapshot.warnings.append(f"未计入不完整、重复或身份不符的训练结果：{path.relative_to(root)}")
            continue
        runs[key] = row

    per_column, benchmark_columns = [], set()
    for path in sorted((root / "benchmark").glob("*/RESULT.json")):
        doc = snapshot.read(path)
        if not doc:
            continue
        col = doc.get("column", {})
        key = col.get("key")
        if (doc.get("experiment_identity") != identity or key not in columns
                or col.get("input_sha256") != columns[key].get("input_sha256")
                or doc.get("selection_not_changed_by_final_timing") is not True):
            snapshot.warnings.append(f"未采用身份/冻结选择证据不完整的 benchmark：{path.relative_to(root)}")
            continue
        expected_rows = {(wid(w), m) for w in weights for m in METHODS}
        accepted = set()
        for row in doc.get("rows", []):
            tag = (row.get("weight_id"), row.get("method"))
            raw, size = row.get("raw_bytes"), row.get("bytes")
            required = ("compression_rate", "encode_seconds_median", "decode_seconds_median",
                        "encode_time_normalized", "decode_time_normalized", "cost")
            valid = (tag in expected_rows and tag not in accepted and row.get("weight_tenths") in [list(w) for w in weights]
                     and wid(row["weight_tenths"]) == row.get("weight_id")
                     and all(finite(row.get(k)) and row[k] >= 0 for k in required)
                     and finite(raw) and raw > 0 and raw == columns[key].get("raw_bytes") and finite(size) and size >= 0
                     and row.get("lossless") is True and row.get("cold_decode_pass") is True
                     and row.get("deterministic") is True and row.get("repeats") == 11
                     and row.get("input_sha256") == col.get("input_sha256"))
            if valid:
                expected_cost = sum(w*v for w, v in zip(row["weight_tenths"],
                    (size/raw, row["encode_time_normalized"], row["decode_time_normalized"]))) / 10
                valid = math.isclose(size/raw, row["compression_rate"], rel_tol=1e-10, abs_tol=1e-12)
                valid = valid and math.isclose(expected_cost, row["cost"], rel_tol=1e-10, abs_tol=1e-12)
            if not valid:
                snapshot.warnings.append(f"跳过不完整/不一致 benchmark 行：{key} {tag}")
                continue
            accepted.add(tag)
            a, b, c = (v/10 for v in row["weight_tenths"])
            item = dict(dataset=col["dataset"], column_key=key, column_name=col["name"],
                        weight_id=row["weight_id"], weight_abc=weight_text(row["weight_tenths"]), a=a, b=b, c=c,
                        method=row["method"], method_label=METHODS[row["method"]], raw_bytes=raw, compressed_bytes=size,
                        compression_rate=row["compression_rate"], encode_seconds=row["encode_seconds_median"],
                        decode_seconds=row["decode_seconds_median"], normalized_encode=row["encode_time_normalized"],
                        normalized_decode=row["decode_time_normalized"], J=row["cost"], chain=row.get("chain", ""),
                        source_seed=row.get("source_seed"), selection=row.get("selection", ""),
                        selection_origin=row.get("source_selection_origin", ""),
                        before_benchmark_selection_cost=row.get("before_benchmark_selection_cost"),
                        lossless=True, cold_decode_pass=True, benchmark_repeats=11,
                        encode_repeat_std_seconds=statistics.stdev(row["encode_seconds_repeats"]) if len(row.get("encode_seconds_repeats", [])) > 1 else None,
                        decode_repeat_std_seconds=statistics.stdev(row["decode_seconds_repeats"]) if len(row.get("decode_seconds_repeats", [])) > 1 else None,
                        source_result=str(path.relative_to(root)), **summarize_seed_metrics(row))
            per_column.append(item)
        if accepted == expected_rows:
            benchmark_columns.add(key)
        else:
            snapshot.warnings.append(f"{key} 最终结果仅覆盖 {len(accepted)}/{len(expected_rows)} 个配置方法×权重。")

    weights_complete = [w for w in weights if columns and seeds and all((k, wid(w), m, s) in runs for k, m, s in itertools.product(columns, ("ppo", "grpo"), seeds))]
    formal_config = (len(columns) == 20 and Counter(c.get("dataset") for c in columns.values()) == FORMAL_COLUMNS
                     and set(weights) == set(GRID) and tuple(seeds) == FORMAL_SEEDS
                     and config.get("iterations") == 64 and config.get("episodes") == 32)
    full = (formal_config and set(runs) == expected and len(benchmark_columns) == len(columns)
            and summary.get("formal_66_weight_grid_complete") is True and not snapshot.warnings)
    coverage = dict(status="COMPLETE" if full else "PARTIAL", runner_status=summary.get("status", failure.get("status", progress.get("status", "UNKNOWN"))),
                    formal_grid_complete=full, configured_runs=len(expected), completed_runs=len(runs),
                    formal_target_runs=20*66*2*3, configured_weights=len(weights),
                    weights_with_any_completed_run=len({r[1] for r in runs}), weights_complete_for_configured_columns=len(weights_complete),
                    formal_target_weights=66, configured_columns=len(columns), benchmark_columns=len(benchmark_columns),
                    seeds=seeds, iterations=config.get("iterations"), episodes=config.get("episodes"),
                    live_progress_completed_runs=progress.get("completed_runs"))
    return dict(manifest=manifest, summary=summary, progress=progress, failure=failure, config=config,
                columns=columns, weights=weights, runs=runs, per_column=per_column, coverage=coverage)


def add_deltas(rows, group_keys):
    baselines = {tuple(r[k] for k in group_keys): r for r in rows if r["method"] == "all38_J_best"}
    for row in rows:
        baseline = baselines.get(tuple(row[k] for k in group_keys))
        for metric in ("compression_rate", "encode_seconds", "decode_seconds"):
            base = baseline.get(metric) if baseline else None
            row[f"{metric}_improvement_pct_vs_all38_J"] = 100*(base-row[metric])/base if base and base > 0 else None


def aggregate(data):
    grouped = defaultdict(list)
    for row in data["per_column"]:
        for ds in (row["dataset"], "ALL"):
            grouped[(ds, row["weight_id"], row["method"])].append(row)
    result = []
    for (ds, weight, method), rows in sorted(grouped.items()):
        raw = sum(r["raw_bytes"] for r in rows)
        keys = {r["column_key"] for r in rows}
        matched_runs = [r for k, r in data["runs"].items() if k[0] in keys and k[1] == weight and k[2] == method]
        configured = sum(1 for col in data["columns"].values() if ds == "ALL" or col["dataset"] == ds)
        result.append(dict(dataset=ds, weight_id=weight, weight_abc=rows[0]["weight_abc"],
                           a=rows[0]["a"], b=rows[0]["b"], c=rows[0]["c"], method=method, method_label=METHODS[method],
                           columns=len(keys), column_keys=";".join(sorted(keys)), configured_columns=configured,
                           formal_dataset_columns=20 if ds == "ALL" else FORMAL_COLUMNS.get(ds),
                           coverage="configured_columns_complete" if len(keys) == configured else "partial_columns",
                           raw_bytes=raw, compressed_bytes=sum(r["compressed_bytes"] for r in rows),
                           compression_rate=sum(r["compressed_bytes"] for r in rows)/raw,
                           encode_seconds=sum(r["encode_seconds"] for r in rows), decode_seconds=sum(r["decode_seconds"] for r in rows),
                           normalized_encode=statistics.mean(r["normalized_encode"] for r in rows),
                           normalized_decode=statistics.mean(r["normalized_decode"] for r in rows),
                           J=statistics.mean(r["J"] for r in rows), training_runs=len(matched_runs),
                           **{k: sum(r[k] for r in matched_runs) if matched_runs else None for k in COST_FIELDS}))
    add_deltas(data["per_column"], ("column_key", "weight_id"))
    add_deltas(result, ("dataset", "weight_id", "column_keys"))
    return result


def table(headers, records, attrs=None):
    body = []
    for i, cells in enumerate(records):
        attr = attrs[i] if attrs else ""
        body.append(f"<tr {attr}>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")
    return "<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def scatter(rows, dataset, metric):
    points = [r for r in rows if r["dataset"] == dataset and r["method"] in MAIN]
    if not points:
        return f"<div class='plot empty'>{escape(dataset)}：尚无完成的最终 benchmark 数据。</div>"
    width, height, left, top, pw, ph = 590, 350, 72, 36, 484, 252
    xs = [r["compression_rate"]*100 for r in points]
    ys = [r[metric]*1000 for r in points]
    xlo, xhi = min(xs), max(xs)
    xpad = max((xhi-xlo)*0.10, abs(xhi)*0.02, 0.1)
    xlo, xhi = max(0, xlo-xpad), xhi+xpad
    ymax = max(max(ys)*1.1, 0.001)
    parts = [f"<svg class='plot' viewBox='0 0 {width} {height}' role='img' aria-label='{escape(dataset)} 压缩率与{escape(metric)}散点图'>"]
    parts.append(f"<text x='{left}' y='20' class='plot-title'>{escape(dataset)} · {'压缩' if metric == 'encode_seconds' else '解压'}时间（ms）</text>")
    for i in range(6):
        x = left + pw*i/5; y = top+ph-ph*i/5
        parts.append(f"<path d='M {left},{y} H {left+pw}' class='grid'/><text x='{left-8}' y='{y+4}' text-anchor='end'>{fmt(ymax*i/5, 2)}</text>")
        parts.append(f"<text x='{x}' y='{top+ph+20}' text-anchor='middle'>{fmt(xlo+(xhi-xlo)*i/5, 1)}%</text>")
    parts.append(f"<path d='M {left},{top} V {top+ph} H {left+pw}' class='axis'/><text x='{left+pw/2}' y='{height-14}' text-anchor='middle'>压缩率 ρ（完整帧字节 / 原始列字节）；左下更好</text>")
    for row in sorted(points, key=lambda r: (r["method"] != "all38_J_best", r["weight_id"])):
        x = left+(row["compression_rate"]*100-xlo)/(xhi-xlo)*pw
        y = top+ph-row[metric]*1000/ymax*ph
        title = (f"{METHODS[row['method']]} | a/b/c={row['weight_abc']} | ρ={row['compression_rate']:.6%} | "
                 f"E={row['encode_seconds']*1000:.6f} ms | D={row['decode_seconds']*1000:.6f} ms | 覆盖{row['columns']}列")
        color = COLORS[row["method"]]
        common = f"fill='{color}' fill-opacity='.65' stroke='{color}' stroke-width='1.2'"
        shape = (f"<rect x='{x-4:.3f}' y='{y-4:.3f}' width='8' height='8' {common}>" if row["method"] == "all38_J_best" else
                 f"<path d='M {x:.3f},{y-5:.3f} L {x+5:.3f},{y:.3f} L {x:.3f},{y+5:.3f} L {x-5:.3f},{y:.3f} Z' {common}>" if row["method"] == "grpo" else
                 f"<circle cx='{x:.3f}' cy='{y:.3f}' r='3.2' {common}>")
        close = "</rect>" if row["method"] == "all38_J_best" else "</path>" if row["method"] == "grpo" else "</circle>"
        parts.append(f"<g class='point' data-weight='{escape(row['weight_id'])}'>{shape}<title>{escape(title)}</title>{close}</g>")
    parts.append("</svg>")
    return "".join(parts)


def render(data, aggregates, snapshot):
    cov, config, summary = data["coverage"], data["config"], data["summary"]
    ds_names = sorted(set(FORMAL_COLUMNS) | {c["dataset"] for c in data["columns"].values()})
    scope = "；".join(f"{d}: {sum(c['dataset']==d for c in data['columns'].values())}/{FORMAL_COLUMNS.get(d, '?')} 列" for d in ds_names)
    baseline = "38 项固定目录中，按当前权重 J 在独立最终计时前选出的方案；不是对所有可能组合的全局最优。"
    rows = data["per_column"]
    origins = Counter(r["selection_origin"] for r in rows if r["method"] in ("ppo", "grpo"))
    costs = {k: sum(r[k] for r in data["runs"].values()) for k in COST_FIELDS}
    invalid_episodes = sum(r["budget"].get("invalid_episodes", 0) for r in data["runs"].values())
    below_invalid_calls = sum(r["budget"].get("valid_reward_calls_at_or_below_invalid_penalty", 0) for r in data["runs"].values())
    oracle = summary.get("oracle", data["progress"].get("oracle", {}))
    warnings = "".join(f"<li>{escape(w)}</li>" for w in snapshot.warnings)
    options = "".join(f"<option value='{wid(w)}'>{weight_text(w)}</option>" for w in sorted(data["weights"]))
    datasets = "".join(f"<option value='{escape(d)}'>{escape(d)}</option>" for d in [*ds_names, "ALL"])
    agg_rows, agg_attrs = [], []
    for r in aggregates:
        agg_rows.append([r["dataset"], r["weight_abc"], r["method_label"], f"{r['columns']}/{r['configured_columns']}（全量 {r['formal_dataset_columns']}）",
                         fmt(r["compression_rate"]*100)+"%", fmt(r["encode_seconds"]*1000), fmt(r["decode_seconds"]*1000),
                         fmt(r["normalized_encode"], 4), fmt(r["normalized_decode"], 4), fmt(r["J"], 5),
                         fmt(r["compression_rate_improvement_pct_vs_all38_J"]), fmt(r["encode_seconds_improvement_pct_vs_all38_J"]), fmt(r["decode_seconds_improvement_pct_vs_all38_J"]),
                         fmt(r["train_seconds"]), fmt(r["optimizer_seconds"])])
        agg_attrs.append(f"class='filter-row' data-dataset='{escape(r['dataset'])}' data-weight='{r['weight_id']}' data-main='{str(r['method'] in MAIN).lower()}'")
    per_rows, per_attrs = [], []
    for r in rows:
        per_rows.append([r["dataset"], r["column_name"], r["weight_abc"], r["method_label"], r["chain"],
                         fmt(r["compression_rate"]*100)+"%", fmt(r["encode_seconds"]*1000), fmt(r["decode_seconds"]*1000), fmt(r["J"], 5),
                         r["source_seed"] if r["source_seed"] is not None else "—", r["selection_origin"] or r["selection"],
                         r["selection_seed_count"] or "—", fmt(r["seed_selection_rho_mean"]*100 if r["seed_selection_rho_mean"] is not None else None),
                         fmt(r["seed_selection_rho_std"]*100 if r["seed_selection_rho_std"] is not None else None)])
        per_attrs.append(f"class='filter-row' data-dataset='{escape(r['dataset'])}' data-weight='{r['weight_id']}' data-main='{str(r['method'] in MAIN).lower()}'")
    training = defaultdict(list)
    for (key, weight, method, seed), row in data["runs"].items():
        training[(row["dataset"], weight, method)].append(row)
    train_rows = [[d, weight_text(next(w for w in data["weights"] if wid(w) == w_id)), METHODS[m], len(rs),
                   *[fmt(sum(r[k] for r in rs)) for k in COST_FIELDS], sum(r["budget"].get("rollouts", 0) for r in rs),
                   sum(r["budget"].get("actual_new_training_encoder_calls", 0) for r in rs)]
                  for (d, w_id, m), rs in sorted(training.items())]
    matched = [r for r in aggregates if r["dataset"] != "ALL" and r["method"] in ("ppo", "grpo") and r["compression_rate_improvement_pct_vs_all38_J"] is not None]
    strict_rho = sum(r["compression_rate_improvement_pct_vs_all38_J"] > 1e-9 for r in matched)
    wins3 = sum(all(r[k+"_improvement_pct_vs_all38_J"] > 1e-9 for k in ("compression_rate", "encode_seconds", "decode_seconds")) for r in matched)
    observations = [f"当前导出 {len(rows)} 条逐列最终测量；配置内训练完成 {cov['completed_runs']}/{cov['configured_runs']}，最终 benchmark 完整列 {cov['benchmark_columns']}/{cov['configured_columns']}。",
                    f"{len(matched)} 个可匹配的“数据集覆盖子集×权重×RL 方法”比较中，ρ 严格更低 {strict_rho} 个，三指标均严格更低 {wins3} 个；相同方案和仅计时抖动不视为算法层面优势。",
                    "PPO/GRPO 最终选中来源：" + ("；".join(f"{k or '未记录'}={v}" for k, v in sorted(origins.items())) or "尚无最终数据") + "。common_safety_baseline 表示回退/选中公共基线，不能写成策略新发现的算子组合。"]
    plots = "".join(f"<section class='plot-section' data-dataset='{escape(d)}'><h3>{escape(d)}（只汇总已测列）</h3><div class='plots'>{scatter(aggregates,d,'encode_seconds')}{scatter(aggregates,d,'decode_seconds')}</div></section>" for d in ds_names)
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>66 权重 PPO / GRPO 压缩实验结果</title><style>
:root{{color-scheme:light;font-family:system-ui,-apple-system,'PingFang SC',sans-serif;color:#203044;background:#eef2f6}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1550px;margin:auto;padding:28px}}h1{{font-size:29px}}h2{{font-size:21px;margin-top:30px}}h3{{font-size:17px}}p,li{{line-height:1.65}}.panel,section{{background:white;padding:20px;border:1px solid #dce3ec;border-radius:10px;margin:16px 0}}.status{{display:inline-block;background:{'#d9f0df' if cov['formal_grid_complete'] else '#fff0cf'};padding:6px 12px;border-radius:5px;font-weight:650}}.muted,small{{color:#596779}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card{{background:#f5f8fc;padding:16px;border-radius:7px}}.card strong{{display:block;font-size:25px;margin-top:5px}}.formula{{font-size:22px;font-family:Georgia,'Times New Roman',serif;background:#f0f5fb;padding:16px}}.table-wrap{{overflow:auto;max-height:680px;border:1px solid #dbe2eb}}table{{border-collapse:collapse;font-size:12px;min-width:100%;background:#fff}}th,td{{padding:9px 10px;border-bottom:1px solid #e2e7ee;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#e8eef6;white-space:nowrap;z-index:1}}td{{min-width:85px}}tr:hover{{background:#f1f6fb}}.controls{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;padding:14px;background:#e7eef7;position:sticky;top:0;z-index:3}}select{{font:inherit;padding:6px}}.plots{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.plot{{width:100%;background:#fbfcfe;border:1px solid #e0e6ed;border-radius:8px}}svg text{{font:11px system-ui;fill:#46566b}}svg .plot-title{{font-size:14px;font-weight:600}}.grid{{stroke:#e4e9f0;stroke-width:1}}.axis{{stroke:#8a98a9;stroke-width:1}}.point:hover{{stroke-width:3;opacity:1}}.empty{{padding:40px;text-align:center}}.legend span{{display:inline-block;margin-right:18px}}.warning{{border-left:4px solid #bf671a;padding-left:14px}}[hidden]{{display:none!important}}a{{color:#146cb3}}code{{overflow-wrap:anywhere}}@media(max-width:800px){{main{{padding:12px}}.plots{{grid-template-columns:1fr}}h1{{font-size:23px}}}}@media print{{main{{padding:0}}.controls{{position:static}}.table-wrap{{max-height:none;overflow:visible}}th{{position:static}}section{{break-inside:avoid}}}}
</style><main><h1>66 权重 PPO / GRPO 压缩实验</h1><span class="status">{cov['status']} · {escape(cov['runner_status'])}</span>
<p class="muted">只读快照：{escape(datetime.now(timezone.utc).isoformat())} · 来源 <code>{escape(snapshot.root)}</code><br>实验 identity：<code>{escape(data['manifest'].get('experiment_identity','缺失'))}</code></p>
<div class="panel"><div class="cards"><div class="card">配置内真实训练完成<strong>{cov['completed_runs']} / {cov['configured_runs']}</strong></div><div class="card">已出现完成训练的权重<strong>{cov['weights_with_any_completed_run']} / 66</strong></div><div class="card">配置内全列全 seed 完成权重<strong>{cov['weights_complete_for_configured_columns']} / {cov['configured_weights']}</strong></div><div class="card">完整研究目标<strong>7,920 次训练</strong><small>20 列 × 66 权重 × 2 方法 × 3 seeds</small></div></div>
<p>{escape(scope)}；本次 seeds={escape(cov['seeds'])}；每次 {escape(cov['iterations'])} iterations × {escape(cov['episodes'])} episodes。完整目标为每次 64×32；小预算 smoke 不能当作等价完整训练。</p>
<p class="warning">{'完整 66 权重配置及最终结果已齐全。' if cov['formal_grid_complete'] else '这是 partial/pilot 报告，不是完整 66 权重、两数据集 20 列实验。表和图只代表已完成 benchmark 的列，不代表整个数据集。'} 当前配置最终 benchmark 完成 {cov['benchmark_columns']}/{cov['configured_columns']} 列；未完成项留空，不从旧报告补数。</p></div>
<h2>目标函数：只归一化时间，不归一化压缩率</h2><div class="panel"><div class="formula">J = a × ρ + b × E/(E+E₀) + c × D/(D+D₀)，reward = −J</div>
<ul><li>ρ = 完整压缩帧字节 / 原始类型列字节，是原值；例如 ρ=0.20 表示保留原始大小的 20%。所有可恢复元数据计入分子。这里不是原始 CSV 文件压缩率。</li><li>E、D 是完整执行路径的压缩、解压时间；E₀、D₀ 是每列本轮冻结 zlib6 参考时间。若与参考等快，归一化值为 0.5；更快趋近 0，更慢趋近 1。这不是除以候选中的最大时间。</li><li>a、b、c ≥ 0，步长 0.1，a+b+c=1，共 66 组。三项都越小越好；不同权重的绝对 J 没有共同标尺，不能按 J 最小选“最佳权重”。</li><li>数据集 ρ = Σ压缩字节/Σ原始字节；E、D 为逐列计时中位数之和，不是实际批量压缩的一次墙钟。归一化时间和 J 为逐列算术平均；不以字节数加权，故 J 不能由表内聚合 ρ 直接代回计算。</li></ul></div>
<h2>当前观察</h2><div class="panel"><ol>{''.join('<li>'+escape(x)+'</li>' for x in observations)}</ol><p>未做统计显著性检验。单 seed 不给 seed 标准差；多 seed 的选优前均值/标准差放在逐列 CSV，最终主表是先选 seed 后独立重测，不是 seed 平均。</p><p>建议下一步：先补齐预定覆盖，再按实际部署的存储/压缩/读取次数选择权重；若讨论泛化，需要另设训练样本与留出列/数据集。</p></div>
<h2>同权重三指标对照</h2><p>{escape(baseline)} 图中每点是一个权重下的一个方法；悬停可看数值。无随机抖动，重合方案可能互相遮挡。</p>
<div class="controls"><label>权重 a/b/c <select id="weight"><option value="">全部已测权重</option>{options}</select></label><label>数据集 <select id="dataset"><option value="">全部</option>{datasets}</select></label><label><input id="all-methods" type="checkbox">显示全部基线和历史 Beam 行</label></div>
<p class="legend"><span style="color:#146cb3">● PPO</span><span style="color:#bc4b12">◆ GRPO</span><span style="color:#656975">■ 38 项固定目录、本权重 J 选优</span></p>{plots}
<p>表中“相对基线改善%”=(基线−本方法)/基线×100%，正值表示本项更小。仅在同权重、同列覆盖下计算。历史 Beam 只重测既有方案，未按每组权重重新搜索。</p>
{table(['数据集','a/b/c','方法','覆盖列','ρ','E 合计 ms','D 合计 ms','均值 E/(E+E₀)','均值 D/(D+D₀)','均值 J（仅同权重）','ρ 改善%','E 改善%','D 改善%','训练 wall 合计 s','optimizer 子项 s'],agg_rows,agg_attrs)}
<h2>训练成本：墙钟与优化器分开</h2><div class="panel"><p>完成运行的训练 wall 合计 {fmt(costs['train_seconds'])} s；optimizer 更新合计 {fmt(costs['optimizer_seconds'])} s（已包含在训练 wall 内，不再相加）。搜索 wall {fmt(costs['search_seconds'])} s；精测/选优 {fmt(costs['refinement_seconds'])} s；runner wall {fmt(costs['run_seconds'])} s（不含后续选中帧保存/复核）。</p>
<p>SUMMARY 记录的所有 invocation wall：{fmt(summary.get('all_recorded_invocation_wall_seconds'))} s；本次/最近一次 invocation：{fmt(summary.get('invocation_wall_seconds'))} s。中断未写完整 receipt 时，记录可能不包含该段；恢复扫描的耗时不等于重新训练时间。原始向量缓存跨运行共享，首次测量成本不均等，不能直接以每个方法 wall 判定优化器谁更快。</p>
<p>独立最终 benchmark：先冻结方案，2 次预热 + 11 次随机交错完整编码/解码，取中位数；同方案别名共享相同测量。独立进程冷解码仅验证，不计入 E/D。报告读取既有验证证据，不重新编解码或重新认证帧。</p>
{table(['数据集','a/b/c','方法','完成运行','train wall s','search wall s','optimizer s（子项）','refinement s','run wall s','真实 trajectories','新增训练编码 calls'],train_rows)}
<details><summary>全局 oracle 成本快照（只读结果记录）</summary><pre>{escape(json.dumps(oracle,ensure_ascii=False,indent=2))}</pre></details></div>
<h2>逐列完整结果</h2><p>来源 common_safety_baseline 是公共基线；sampling/greedy 以来源原字段为准。seed ρ 的均值/标准差是选优阶段数据，不是最终 11 轮数据；全部原始字段与 E/D seed 统计可在 CSV 查看。</p>
{table(['数据集','列','a/b/c','方法','算子/算法链','ρ','E ms','D ms','J（仅同权重）','选中 seed','选择来源','seed 数','选优 seed ρ 均值%','选优 seed ρ std 百分点'],per_rows,per_attrs)}
<h2>证据边界与文件</h2><div class="panel"><ul><li>同一完整类型列在线训练与评估，没有 held-out；不能声称跨列/新模型泛化。</li><li>当前 legacy 策略：17 家族、38 个前缀选项 + 38 个尾端 + STOP = 77 动作；不是新 30 家族多流图策略，也不是所有算子组合穷举。30 项算法目录包含 5 个单算法 + 25 个有序算法对；38 项目录另有 8 个独立熵编码尾端。</li><li>每权重/列/method/seed 独立训练。共享本轮真实测量向量，但不共享候选池；奖励按当前权重重新计算。PPO/GRPO 最终允许公共基线保底，必须区分“找到新组合”和“选中基线”。</li><li>无效动作固定 reward=−5，并非所有合法 reward 的下界；例如纯 ρ 目标下，合法方案若 ρ≥5，也会获得 ≤−5。因此不能把无效惩罚当作严格最差保证。完成运行累计无效 episodes={invalid_episodes}；合法 reward≤−5 的训练调用={below_invalid_calls}（跨运行调用次数，不是全局唯一方案数）。</li><li>计时包含实际 Python/NumPy 调度及对应原生 codec/子进程路径；不是统一纯 C++ 内核耗时。各结果独立重复的方差不等于跨 seed 泛化方差。</li><li>实验磁盘守卫是剩余空间警戒线，不是输出目录硬上限；本脚本不访问测量数据库、不执行编解码、不修改实验结果。</li></ul>
<p><a href="aggregate.csv">aggregate.csv</a> · <a href="per_column.csv">per_column.csv</a> · <a href="REPORT_META.json">快照覆盖、警告与输入 SHA256</a></p>
<p>证据：<a href="../MANIFEST.json">MANIFEST</a> · <a href="../SUMMARY.json">SUMMARY（可能尚未生成）</a>。来源结果为相同 identity 的独立最终 benchmark；运行中读取得到的是非事务性快照，重新导出可刷新。</p>
{'<h3>读取警告</h3><ul>'+warnings+'</ul>' if warnings else '<p>未发现结果结构/identity/公式一致性警告；这不是源码或帧的重新审计。</p>'}</div></main>
<script>const w=document.getElementById('weight'),d=document.getElementById('dataset'),a=document.getElementById('all-methods');function filter(){{document.querySelectorAll('.filter-row').forEach(r=>r.hidden=!!((w.value&&r.dataset.weight!==w.value)||(d.value&&r.dataset.dataset!==d.value)||(!a.checked&&r.dataset.main!=='true')));document.querySelectorAll('.point').forEach(p=>p.style.opacity=w.value&&p.dataset.weight!==w.value?'0.08':'1');document.querySelectorAll('.plot-section').forEach(p=>p.hidden=!!(d.value&&d.value!=='ALL'&&p.dataset.dataset!==d.value))}}[w,d,a].forEach(x=>x.addEventListener('change',filter));filter();</script></html>"""


def write_csv(path, rows, default_fields):
    fields = list(dict.fromkeys([*default_fields, *(k for row in rows for k in row)]))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Prevent spreadsheet formula interpretation in source-supplied labels.
            writer.writerow({k: ("'"+v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v) for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Directory containing MANIFEST.json; outputs only report/.")
    args = parser.parse_args(argv)
    root = args.experiment.resolve()
    if not root.is_dir():
        parser.error("experiment directory must exist")
    output = root / "report"
    if output.resolve().parent != root or output.is_symlink():
        parser.error("report must be a real immediate child of the experiment directory")
    snapshot = Snapshot(root)
    data = collect(snapshot)
    aggregates = aggregate(data)
    document = render(data, aggregates, snapshot)
    output.mkdir(exist_ok=True)
    targets = [output / name for name in ("REPORT.html", "aggregate.csv", "per_column.csv", "REPORT_META.json")]
    if any(p.is_symlink() for p in targets):
        parser.error("refusing symlink output files")
    write_csv(targets[1], aggregates, ("dataset", "weight_id", "method", "compression_rate", "encode_seconds", "decode_seconds"))
    write_csv(targets[2], data["per_column"], ("dataset", "column_key", "weight_id", "method", "compression_rate", "encode_seconds", "decode_seconds"))
    targets[0].write_text(document, encoding="utf-8")
    meta = dict(report_schema="weight-grid-read-only-report-v1", generated_at=datetime.now(timezone.utc).isoformat(),
                experiment_identity=data["manifest"].get("experiment_identity"), coverage=data["coverage"],
                warnings=snapshot.warnings, aggregate_rows=len(aggregates), per_column_rows=len(data["per_column"]),
                source_sha256=snapshot.sources, report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                experiment_modified=False, actual_codec_calls=0, frame_validation="read recorded results only; not rerun",
                cross_weight_absolute_J_ranking=False)
    targets[3].write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(report=str(targets[0]), **data["coverage"], warnings=len(snapshot.warnings),
                          aggregate_rows=len(aggregates), per_column_rows=len(data["per_column"])), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
