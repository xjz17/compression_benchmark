#!/usr/bin/env python3
"""Report-only, provenance-preserving per-column tables for the C206 trial.

This module never invokes an encoder or changes the frozen experiment records.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METHODS = ("beam29", "grpo", "ppo")
METHOD_NAMES = {"beam29": "Beam29", "grpo": "GRPO（β=0）", "ppo": "PPO"}
METHOD_RANK = {name: i for i, name in enumerate(METHODS)}
FAMILIES = {
    "eraser": "Eraser", "scaling": "定标", "frame_of_reference": "FoR",
    "value_reorder": "重排", "regression_predictor": "回归预测",
    "outlier_separation": "离群分离", "bitpack_int": "Bitpack", "delta": "Delta",
    "xor": "XOR", "trailing_zero_pack": "尾零打包", "dictionary": "字典",
    "zigzag": "ZigZag", "block_transform": "块变换", "symbol_recode": "符号重编码",
    "run_length": "RLE", "lz_match": "LZ 匹配", "match_parse_policy": "匹配解析",
    "match_window_model": "匹配窗口", "match_token_format": "Token 格式",
    "integer_code": "整数码", "bitstream_pack": "位流打包",
    "entropy_huffman": "Huffman", "entropy_range": "Range", "entropy_fse": "FSE",
    "entropy_special": "特殊熵编码", "literal_entropy": "Literal 熵编码",
    "block_state_control": "块状态/核心", "container_frame": "容器",
}
IMPLEMENTATIONS = {
    "project_bytes": "project", "zstd_rans": "zstd-rANS", "tsfile_int64": "TsFile-i64",
    "tsfile_rle": "TsFile", "simple8b": "Simple8b", "delta_binary_packed": "DBP",
    "chimp128": "Chimp128", "fpc": "FPC", "bos64_adapter": "BOS64",
    "snappy_matchfinder_residual_core": "Snappy 残余核心",
    "lz4_matchfinder_residual_core": "LZ4 残余核心", "snappy": "Snappy",
    "fpzip": "FPZIP", "alp": "ALP", "alp_rd": "ALP-RD", "buff": "BUFF",
    "buff_column_split": "BUFF 分列", "brotli_static_small": "Brotli-small",
    "brotli_lazy65535": "Brotli-lazy65535", "bzip2_bwt": "BWT",
    "bzip2_mtf": "MTF", "bzip2_uint8": "bzip2-u8",
    "brotli_residual_core": "Brotli 残余核心", "snappy_snap_1_1_1": "Snap-1.1.1",
    "reger_legacy_block_core": "REGER 复合核心",
    "blosc2_3_3_3_binding_macro": "Blosc2-3.3.3 宏",
    "elf64": "Elf64", "elf64_on_chimp": "Elf64-on-Chimp", "elf64_plus": "Elf64+",
}


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _short_name(action_id, name):
    parts = name.split(".")
    assert len(parts) >= 3 and parts[0] == "op", (action_id, name)
    family = FAMILIES.get(parts[1], parts[1])
    suffix = ".".join(parts[2:])
    reference = "reference" in suffix
    for tail in (".cpp_reference_v1", ".cpp_v1", ".v1"):
        if suffix.endswith(tail):
            suffix = suffix[:-len(tail)]
            break
    if suffix == "reference_v1":
        detail = "ref"
    else:
        detail = IMPLEMENTATIONS.get(suffix, suffix)
        if reference:
            detail += "·ref"
    # Retain the implementation's identity without repeating "core" twice.
    if parts[1] == "block_state_control" and ("核心" in detail or "宏" in detail):
        label = detail
    else:
        label = f"{family}·{detail}"
    return f"{label} [{action_id}]"


def _stem(row):
    return (f"{row['method']}__seed{row['seed']}__{row['dataset']}__"
            f"{row['numeric_index']:02d}")


def _dump_csv(path, records):
    assert records
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_column_tables(rows, manifest, registry, out) -> list[str]:
    """Write new derived tables; return the section to insert into REPORT.md.

    ``rows`` are the unmodified decoded columns/*.json records. Selection is by
    size and then seed, never by the fastest timing sample or a mixed median.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    columns = {(c["dataset"], c["numeric_index"]): c for c in manifest["columns"]}
    assert len(columns) == len(manifest["columns"]), "duplicate manifest column"
    order = {key: i for i, key in enumerate(columns)}
    seeds = manifest["config"]["seeds"]
    expected = {(d, i, m, s) for d, i in columns for m in METHODS
                for s in ([0] if m == "beam29" else seeds)}
    observed = {(r["dataset"], r["numeric_index"], r["method"], r["seed"]) for r in rows}
    assert observed == expected and len(rows) == len(expected), "incomplete/duplicate rows"
    actions = {a["policy_action_id"]: a["name"] for a in registry["policy_actions"]}
    assert all(i in actions for i in range(206)), "missing canonical action ID"
    assert len(actions) == len(registry["policy_actions"]), "duplicate action ID"
    used = sorted({a for r in rows for a in r["selected_ids"]})
    aliases = {a: _short_name(a, actions[a]) for a in used}
    by_method = defaultdict(list)
    by_column = defaultdict(list)
    for r in rows:
        key = r["dataset"], r["numeric_index"]
        c = columns[key]
        assert r["selected_pipeline"] == [actions[i] for i in r["selected_ids"]], _stem(r)
        assert r["column"] == c["name"] and r["original_bytes"] == c["raw_bytes"], _stem(r)
        assert r["input_sha256"] == c["input_sha256"] and r["lossless"], _stem(r)
        assert math.isclose(r["compression_ratio"], r["original_bytes"] / r["compressed_bytes"],
                            rel_tol=1e-12), _stem(r)
        assert all(math.isfinite(r[k]) and r[k] >= 0 for k in ("compress_sec", "compose_sec"))
        by_method[key + (r["method"],)].append(r)
        by_column[key].append(r)

    def sort_key(r):
        return order[r["dataset"], r["numeric_index"]], METHOD_RANK[r["method"]], r["seed"]

    def chain(r):
        return " → ".join(aliases[a] for a in r["selected_ids"]) or "RAW（无算子）"

    def record(r, scope):
        c = columns[r["dataset"], r["numeric_index"]]
        result = {k: r[k] for k in (
            "dataset", "column", "numeric_index", "source_index", "kind", "method", "seed",
            "original_bytes", "compressed_bytes", "compression_ratio", "compress_sec", "compose_sec",
            "decompress_sec")}
        result.update(selected_ids=json.dumps(r["selected_ids"]),
                      selected_pipeline=json.dumps(r["selected_pipeline"], ensure_ascii=False),
                      chain_short=chain(r), source_row=f"columns/{_stem(r)}.json",
                      frame_path=r["frame_path"], input_sha256=r["input_sha256"],
                      frame_sha256=r["frame_sha256"], source_csv=c["source_path"],
                      source_csv_sha256=c["source_sha256"], selection_scope=scope)
        return result

    best = sorted((min(rr, key=lambda r: (r["compressed_bytes"], r["seed"]))
                   for rr in by_method.values()), key=sort_key)
    assert len(best) == len(columns) * len(METHODS)
    best_records = [record(r, "within_method_min_bytes_then_seed; best_of_saved_seeds") for r in best]
    _dump_csv(out / "per_column_best_by_method.csv", best_records)
    full = sorted(rows, key=sort_key)
    _dump_csv(out / "per_column_all_seed_details.csv",
              [record(r, "one_recorded_run; no_cross_seed_selection") for r in full])

    known = []
    known_records = []
    for key in columns:
        rr = by_column[key]
        min_size = min(r["compressed_bytes"] for r in rr)
        ties = sorted((r for r in rr if r["compressed_bytes"] == min_size),
                      key=lambda r: (METHOD_RANK[r["method"]], r["seed"]))
        representative = ties[0]
        winners = [m for m in METHODS if any(r["method"] == m for r in ties)]
        known.append((representative, winners))
        derived = record(representative, "best_observed_across_three_methods_and_saved_seeds; not_global_oracle")
        derived.update(tied_methods=json.dumps(winners),
                       tied_method_seeds=json.dumps([{k: r[k] for k in ("method", "seed")} for r in ties]),
                       representative_rule="minimum_bytes; stable_method_order_beam29_grpo_ppo; ascending_seed")
        known_records.append(derived)
    assert len(known) == len(columns)
    _dump_csv(out / "per_column_best_observed.csv", known_records)

    appendix = ["# 每个 seed 的逐列完整算子组合", "",
                f"共 {len(full)} 条原始实验记录的可读转录，不重跑、不修改原记录。",
                "所有名称逐项核对冻结注册表；箭头严格按实际执行顺序。压缩时间为该条记录的 5 次新编码中位数。",
                "", "[返回主报告](REPORT.md) · [下载完整 CSV](per_column_all_seed_details.csv)", ""]
    datasets = list(dict.fromkeys(c[0] for c in columns))
    for dataset in datasets:
        appendix.extend([f"## {dataset}", ""])
        for method in METHODS:
            appendix.extend([f"<details><summary>{METHOD_NAMES[method]}：全部列与 seed（点击展开）</summary>", "",
                             "| 列 | seed | 完整算子名与 ID（执行顺序） | 压缩字节 | 压缩比 | 压缩秒 | 组合秒 | 原始记录 |",
                             "|---|---:|---|---:|---:|---:|---:|---|"])
            for r in full:
                if r["dataset"] != dataset or r["method"] != method:
                    continue
                exact = " → ".join(f"`{_escape(actions[a])}` [{a}]" for a in r["selected_ids"]) or "RAW"
                appendix.append(f"| {_escape(r['column'])} | {r['seed']} | {exact} | {r['compressed_bytes']:,} | "
                                f"{r['compression_ratio']:.3f}× | {r['compress_sec']:.5f} | {r['compose_sec']:.3f} | "
                                f"[JSON](columns/{_stem(r)}.json) |")
            appendix.extend(["", "</details>", ""])
    appendix.extend(["## 三种搜索方法最终组合使用的 ID / 简称 / 完整名称", "",
                     f"以下 {len(used)} 个 ID 的简称仅帮助阅读，不把项目 reference/proxy 或残余核心冒充为上游生产库。",
                     "`ref` 表示注册名中含 reference；DBP = delta_binary_packed，FoR = frame_of_reference。",
                     "同一家族但不同实现保留不同 ID；简称相似不代表可合并。", "",
                     "| Policy ID | 表中简称 | 冻结注册表完整名称 |", "|---:|---|---|"])
    appendix.extend(f"| {a} | {_escape(aliases[a])} | `{_escape(actions[a])}` |" for a in used)
    (out / "PER_COLUMN_ALL_SEEDS.md").write_text("\n".join(appendix) + "\n", encoding="utf-8")

    section = ["## 三种搜索方法在已保存种子中的逐列最佳", "",
               "这里的“最佳”按**完整压缩文件字节最少**定义，不按速度排序。下表仅在本轮 Beam29、GRPO、PPO 的已保存结果中比较；不是 206 算子的全空间最优，也不是所有传统压缩算法的最优。",
               "出现大小并列时列出所有获胜方法；展示链和时间的代表记录固定按 Beam29 → GRPO → PPO、再按 seed 升序选取，**不挑最快计时**。所有指标均来自该条代表记录。", ""]
    for dataset in datasets:
        section.extend([f"### {dataset}：逐列已测最佳", "",
                        "| 列 | 最少字节的方法（并列全部列出） | 展示记录方法 / seed | 最佳已测组合 | 压缩字节 | 压缩比 | 压缩秒 |",
                        "|---|---|---|---|---:|---:|---:|"])
        for r, winners in known:
            if r["dataset"] != dataset:
                continue
            winner_names = "、".join(METHOD_NAMES[m] for m in winners)
            section.append(f"| {_escape(r['column'])} | {winner_names} | {METHOD_NAMES[r['method']]} / {r['seed']} | "
                           f"{_escape(chain(r))} | {r['compressed_bytes']:,} | {r['compression_ratio']:.3f}× | {r['compress_sec']:.5f} |")
        section.append("")
    section.extend(["[下载每列当前已测最佳 CSV](per_column_best_observed.csv)（含并列方法、全部并列 seed、原始 JSON 与 frame 路径）。", "",
                    "## Beam29、GRPO、PPO 各自为每列选中的最佳组合", "",
                    f"共 {len(best)} 条：{len(columns)} 列 × 3 种方法。Beam29 使用唯一的一次运行；GRPO/PPO 各自在 {len(seeds)} 个已保存 seed 中取压缩字节最小的一次，大小相同取 seed 较小者。",
                    "这是**逐列 best-of-3**，并非上方数据集级“三 seed 中位数”，也不代表另跑了一次可同时达到所有逐列最优值的统一策略。压缩比、压缩秒、组合秒和算子链始终来自同一条选中记录；组合秒仅该次运行，不含尝试另外两个 seed 的成本。", ""])
    for dataset in datasets:
        section.extend([f"### {dataset}：各方法逐列结果", "",
                        "| 列 | 方法 | 选中 seed | 最优已测算子组合（ID 顺序） | 压缩字节 | 压缩比 | 压缩秒 | 组合秒 |",
                        "|---|---|---:|---|---:|---:|---:|---:|"])
        for r in best:
            if r["dataset"] != dataset:
                continue
            section.append(f"| {_escape(r['column'])} | {METHOD_NAMES[r['method']]} | {r['seed']} | {_escape(chain(r))} | "
                           f"{r['compressed_bytes']:,} | {r['compression_ratio']:.3f}× | {r['compress_sec']:.5f} | {r['compose_sec']:.3f} |")
        section.append("")
    section.extend(["[下载各方法逐列最佳 CSV](per_column_best_by_method.csv) · "
                    "[查看全部 140 条完整名称、ID 和种子记录及简称对照表](PER_COLUMN_ALL_SEEDS.md) · "
                    "[下载全部种子明细 CSV](per_column_all_seed_details.csv)", "",
                    "这三个新增 CSV 仅是已完成实验的派生视图，保留 source_row、frame_path、输入和 frame SHA256；不覆盖原有 per_column_results.csv、逐列 JSON 或压缩文件。", ""])
    return section
