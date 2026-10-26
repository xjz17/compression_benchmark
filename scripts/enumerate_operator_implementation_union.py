#!/usr/bin/env python3
"""Expand the deduplicated 194-record union and group it by the 28 operators."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import primitive_operator_action_runtime as primitive  # noqa: E402


OPERATOR_LABELS = {
    "eraser": "Eraser 数值规整",
    "scaling": "小数定标",
    "frame_of_reference": "基准值变换（FOR）",
    "value_reorder": "值重排",
    "regression_predictor": "回归预测",
    "outlier_separation": "异常值分离",
    "bitpack_int": "整数位打包",
    "delta": "差分",
    "xor": "异或",
    "trailing_zero_pack": "尾零打包",
    "dictionary": "字典编码",
    "zigzag": "ZigZag 映射",
    "block_transform": "块变换",
    "symbol_recode": "符号重编码",
    "run_length": "行程编码",
    "lz_match": "LZ 匹配",
    "match_parse_policy": "匹配解析策略",
    "match_window_model": "匹配窗口模型",
    "match_token_format": "匹配令牌格式",
    "integer_code": "整数编码",
    "bitstream_pack": "比特流打包",
    "entropy_huffman": "Huffman 熵编码",
    "entropy_range": "Range/算术熵编码",
    "entropy_fse": "FSE/rANS 熵编码",
    "entropy_special": "专用熵编码",
    "literal_entropy": "字面量熵编码",
    "block_state_control": "块状态控制",
    "container_frame": "容器封装",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def joined(values: list[str]) -> str:
    return " | ".join(dict.fromkeys(value for value in values if value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "analysis_outputs/operator_implementation_dedup_20260903_v1/audit.json",
    )
    parser.add_argument(
        "--cpp-catalog",
        type=Path,
        default=ROOT / "analysis_outputs/cpp_codec_action_split_20260902_v1/cpp_action_candidates.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    cpp_rows = read_csv(args.cpp_catalog)
    registry_rows = primitive.registry()["actions"]

    alias_to_canonical: dict[str, str] = {}
    for group in audit["confirmed_cpp_candidate_duplicate_groups"]:
        for alias in group["aliases"]:
            alias_to_canonical[alias] = group["canonical"]

    cpp_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    cpp_order: dict[str, int] = {}
    for index, row in enumerate(cpp_rows):
        canonical = alias_to_canonical.get(row["action_name"], row["action_name"])
        cpp_groups[canonical].append(row)
        cpp_order.setdefault(canonical, index)

    cross_cpp_to_registered = {
        alias_to_canonical.get(group["cpp_candidate"], group["cpp_candidate"]):
        group["registered_action"]
        for group in audit["cross_catalog_duplicate_groups"]
    }
    cross_registered_to_cpp = {
        registered: cpp for cpp, registered in cross_cpp_to_registered.items()
    }

    records: list[dict[str, object]] = []
    for row in registry_rows:
        cpp_canonical = cross_registered_to_cpp.get(row["name"])
        merged_cpp = cpp_groups.get(cpp_canonical, []) if cpp_canonical else []
        member_names = [row["name"], *(item["action_name"] for item in merged_cpp)]
        records.append({
            "operator_id": row["operator_id"],
            "operator_label_zh": OPERATOR_LABELS[row["operator_id"]],
            "canonical_name": row["name"],
            "member_names": member_names,
            "origin": "registered+cpp_candidate" if merged_cpp else "registered",
            "registered_action_id": row["action_id"],
            "execution_domain": row["execution_domain"],
            "source_methods": list(row["source_methods"]),
            "selected_levels": [item["selected_level"] for item in merged_cpp],
            "evidence": ["registered", *(item["evidence"] for item in merged_cpp)],
            "standalone_composable": (
                row["execution_domain"] == "normalized_int64_reference"
                or row["name"] in primitive.EXACT_ATOMIC_ACTION_BY_NAME
            ),
            "implementation_ids": [
                row["implementation_id"],
                *(item["implementation_id"] for item in merged_cpp),
            ],
            "implementation_sources": [
                row["implementation_source"],
                *(item["implementation_source"] for item in merged_cpp),
            ],
            "implementation_scope": row["implementation_scope"],
            "sort_key": (0, int(row["action_id"])),
        })

    for canonical, group in cpp_groups.items():
        if canonical in cross_cpp_to_registered:
            continue
        operator_ids = {row["operator_id"] for row in group}
        if len(operator_ids) != 1:
            raise RuntimeError(f"merged group crosses operator families: {canonical}")
        records.append({
            "operator_id": group[0]["operator_id"],
            "operator_label_zh": OPERATOR_LABELS[group[0]["operator_id"]],
            "canonical_name": canonical,
            "member_names": [row["action_name"] for row in group],
            "origin": "cpp_candidate",
            "registered_action_id": "",
            "execution_domain": (
                "standalone_cpp_candidate"
                if all(row["standalone_composable"] == "1" for row in group)
                else "method_scoped_cpp_candidate"
            ),
            "source_methods": [row["method"] for row in group],
            "selected_levels": [row["selected_level"] for row in group],
            "evidence": [row["evidence"] for row in group],
            "standalone_composable": all(
                row["standalone_composable"] == "1" for row in group
            ),
            "implementation_ids": [row["implementation_id"] for row in group],
            "implementation_sources": [row["implementation_source"] for row in group],
            "implementation_scope": "",
            "sort_key": (1, cpp_order[canonical]),
        })

    operator_order = list(primitive.ACTIONS_BY_OPERATOR_ID)
    if set(operator_order) != set(OPERATOR_LABELS):
        raise RuntimeError("Chinese labels do not match the 28-operator taxonomy")
    grouped: dict[str, list[dict[str, object]]] = {operator_id: [] for operator_id in operator_order}
    for record in records:
        grouped[str(record["operator_id"])].append(record)
    for values in grouped.values():
        values.sort(key=lambda row: row["sort_key"])

    flattened: list[dict[str, object]] = []
    global_index = 1
    for operator_id in operator_order:
        for local_index, record in enumerate(grouped[operator_id], 1):
            record["implementation_index"] = global_index
            record["within_operator_index"] = local_index
            flattened.append(record)
            global_index += 1

    if len(registry_rows) != 95 or len(cpp_rows) != 106:
        raise RuntimeError("input counts drifted from the audited 95 + 106 contract")
    if len(cpp_groups) != 100 or len(flattened) != 194:
        raise RuntimeError(
            f"deduplicated count mismatch: cpp={len(cpp_groups)}, union={len(flattened)}"
        )
    if set(grouped) != set(operator_order) or any(not rows for rows in grouped.values()):
        raise RuntimeError("the union does not cover all 28 operator families")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "implementations_194.csv"
    fieldnames = [
        "implementation_index", "operator_id", "operator_label_zh",
        "within_operator_index", "canonical_name", "member_names", "origin",
        "registered_action_id", "execution_domain", "source_methods",
        "selected_levels", "evidence", "standalone_composable",
        "implementation_ids", "implementation_sources", "implementation_scope",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in flattened:
            writer.writerow({
                field: joined(record[field]) if isinstance(record[field], list) else record[field]
                for field in fieldnames
            })

    counts = {operator_id: len(grouped[operator_id]) for operator_id in operator_order}
    summary = {
        "schema": "operator-implementation-union-catalog-v1",
        "identity_rule": audit["identity_rule"],
        "total_implementations": len(flattened),
        "operator_family_count": len(operator_order),
        "origin_counts": dict(Counter(str(row["origin"]) for row in flattened)),
        "operator_counts": counts,
        "operator_count_sum": sum(counts.values()),
        "source_counts": {
            "registered_actions": len(registry_rows),
            "raw_cpp_candidates": len(cpp_rows),
            "deduplicated_cpp_candidates": len(cpp_groups),
            "cross_catalog_duplicates_removed": len(cross_cpp_to_registered),
        },
    }
    (args.output_dir / "operator_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# 194 个算子实现记录：28 类完整枚举",
        "",
        "身份规则：`operator_id` 相同，且固定 20 列压缩比向量完全相同，才合并。",
        "",
        "| 序号 | operator_id | 中文说明 | 实现数 |",
        "|---:|---|---|---:|",
    ]
    for index, operator_id in enumerate(operator_order, 1):
        lines.append(
            f"| {index} | `{operator_id}` | {OPERATOR_LABELS[operator_id]} | {counts[operator_id]} |"
        )
    lines.extend(["", f"合计：**{sum(counts.values())}**。", ""])
    for index, operator_id in enumerate(operator_order, 1):
        lines.extend([
            f"## {index}. `{operator_id}`：{OPERATOR_LABELS[operator_id]}（{counts[operator_id]} 个）",
            "",
        ])
        for record in grouped[operator_id]:
            aliases = list(record["member_names"])[1:]
            suffix = f"；合并别名：`{'`, `'.join(aliases)}`" if aliases else ""
            methods = joined(list(record["source_methods"])) or "无 method 标注"
            action_id = record["registered_action_id"]
            registered = f"；action_id={action_id}" if action_id != "" else ""
            lines.append(
                f"{record['within_operator_index']}. `{record['canonical_name']}`"
                f"；来源={record['origin']}{registered}；method={methods}{suffix}"
            )
        lines.append("")
    lines.extend([
        "## 口径说明",
        "",
        "- `registered`：已经进入策略 action registry。",
        "- `cpp_candidate`：35 个 codec 中识别出的 method-scoped C++ 候选；多数仍不能独立串联。",
        "- `registered+cpp_candidate`：ANS 的 C++ candidate 已与同压缩比、同算子名的注册 action 合并。",
        "- 完整源码、evidence、level、别名和执行域字段见 `implementations_194.csv`。",
    ])
    (args.output_dir / "CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
