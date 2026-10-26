#!/usr/bin/env python3
"""Export every policy action's primary membership in the existing 28 families.

The historical 218-name snapshot is kept distinct from newly registered actions.
Membership follows the registered operator_id, not a guess from a codec name.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified
from enumerate_operator_implementation_union import OPERATOR_LABELS
from canonical182_policy_paths import variants_for
import native_lz_parameters as lz

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "analysis_outputs/policy_action_replacement_86_20260905_v1/final_2048/action_registry.json"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(output: Path, baseline_path: Path = PREVIOUS) -> dict:
    old = json.loads(baseline_path.read_text(encoding="utf-8"))
    old_names = {action["name"] for action in old["policy_actions"]}
    assert len(old_names) == 218, "expected the user's frozen 218-action snapshot"
    old_ids = {action["name"]: action["policy_action_id"] for action in old["policy_actions"]}
    old_canonical_names = {primitive.ACTION_CANONICAL_NAMES.get(name, name) for name in old_names}
    registry = primitive.registry()
    concrete = {action["name"]: action for action in registry["actions"]}
    assert old_canonical_names <= set(primitive.POLICY_ACTION_BY_NAME), "historical action removed without migration"
    families = tuple(primitive.ACTIONS_BY_OPERATOR_ID)
    assert len(families) == 28 and set(families) == set(OPERATOR_LABELS)
    records = []
    for policy in primitive.POLICY_ACTIONS:
        spec = concrete[policy.name]
        descriptor = unified.describe_policy_action(policy.name)
        variants = variants_for(policy.action_id)
        variant_desc = [unified.describe_policy_action(policy.name, io_variant=v) for v in variants]
        legacy_names = [primitive.CANONICAL_TO_LEGACY_NAME[v] for v in variants]
        variant_specs = [primitive.ALL_CONCRETE_ACTION_BY_NAME[n] for n in legacy_names]
        assert spec["operator_id"] in families
        records.append({
            "policy_action_id": policy.action_id,
            "previous_218_policy_id": "|".join(str(old_ids[name]) for name in old_names
                if primitive.ACTION_CANONICAL_NAMES.get(name, name) == policy.name),
            "concrete_action_id": "|".join(str(v.action_id) for v in variant_specs),
            "policy_registry_version": primitive.ACTION_REGISTRY_VERSION,
            "io_variant_count": len(variants),
            "io_variants": "|".join(variants),
            "parameters": json.dumps(lz.parameter_schema() if policy.name == lz.ALGORITHM_NAME else {}, ensure_ascii=False),
            "variant_input_kinds": json.dumps({v: list(d.accepted_input_kinds) for v, d in zip(variants, variant_desc)}, ensure_ascii=False),
            "variant_output_kinds": json.dumps({v: list(d.output_kinds) for v, d in zip(variants, variant_desc)}, ensure_ascii=False),
            "unified_action_id": descriptor.action_id,
            "action_name": policy.name,
            "operator_id": spec["operator_id"],
            "operator_label_zh": OPERATOR_LABELS[spec["operator_id"]],
            "cohort": "original_218" if policy.name in old_canonical_names else "added_after_218",
            "execution_mode": descriptor.execution_mode,
            "input_kinds": "|".join(descriptor.accepted_input_kinds),
            "output_kinds": "|".join(descriptor.output_kinds),
            "source_methods": "|".join(spec.get("source_methods", [])),
            "implementation_scope": spec.get("implementation_scope", "reference implementation"),
            "implementation_source": "|".join(sorted({v.implementation_source for v in variant_specs})),
            "evidence": descriptor.evidence,
        })
    assert len(records) == len({row["action_name"] for row in records})
    previous = [row for row in records if row["cohort"] == "original_218"]
    new = [row for row in records if row["cohort"] == "added_after_218"]
    current_counts = Counter(row["operator_id"] for row in records)
    previous_counts = Counter(row["operator_id"] for row in previous)
    summary = [{"operator_id": family, "operator_label_zh": OPERATOR_LABELS[family],
                "original_218": previous_counts[family],
                "added": current_counts[family] - previous_counts[family],
                "current": current_counts[family]} for family in families]
    assert sum(row["original_218"] for row in summary) == len(old_canonical_names)
    assert sum(row["current"] for row in summary) == len(primitive.POLICY_ACTIONS)
    output.mkdir(parents=True, exist_ok=True)
    for label, rows in (("original_218_actions", previous), ("current_actions", records), ("added_actions", new)):
        if rows:
            write_csv(output / (label + ".csv"), rows)
        lines = [f"# {label}: {len(rows)} 个 action，按 28 类算子归类", "",
                 "每个 action 只按注册的 operator_id 计入一个主类。阶段接口或复合核心不等于原子实现；具体边界见 CSV 的 implementation_scope。", ""]
        for family in families:
            members = [row for row in rows if row["operator_id"] == family]
            lines += [f"## {OPERATOR_LABELS[family]} / `{family}`（{len(members)}）", "",
                      "| Policy ID | Action 完整名称 | 输入类型 |", "|---:|---|---|"]
            lines += [f"| {row['policy_action_id']} | `{row['action_name']}` | {row['input_kinds'].replace('|', ', ')} |" for row in members]
            lines += [""]
        (output / (label + ".md")).write_text("\n".join(lines), encoding="utf-8")
    write_csv(output / "operator_counts.csv", summary)
    payload = {"operator_families": 28, "original_records": len(old_names),
               "original_actions_after_canonicalization": len(previous),
               "added_actions": len(new), "current_policy_actions": len(records),
               "counts": summary, "baseline_snapshot": str(baseline_path),
               "policy_registry_version": primitive.ACTION_REGISTRY_VERSION,
               "io_variant_records": sum(row["io_variant_count"] for row in records),
               "identity_note": "algorithm-level policy IDs; each io_variant retains its concrete packet/width contract. Multiple concrete IDs are compatibility selectors, not extra policy logits. original_218 counts merged algorithms, not old records"}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-registry", type=Path, default=PREVIOUS)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir, args.baseline_registry), ensure_ascii=False, indent=2))
