#!/usr/bin/env python3
"""Audit and summarize depth-3 family-unique TypedBeam/exact PV2F results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from physical_operator_runtime_v2 import TRANSFORMS


TRANSFORM_FAMILY = {spec.name: spec.family for spec in TRANSFORMS}
WIDTHS = (29, 64, 256, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_pipeline(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(" -> ") if part.strip())


def legal_family_unique(pipeline: tuple[str, ...], max_transforms: int = 2) -> bool:
    transforms = pipeline[:-2]
    if len(transforms) > max_transforms or any(name not in TRANSFORM_FAMILY for name in transforms):
        return False
    families = tuple(TRANSFORM_FAMILY[name] for name in transforms)
    return len(families) == len(set(families))


def verify_manifest_outputs(directory: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "complete":
        raise AssertionError(f"incomplete manifest: {directory}")
    for name, expected in manifest.get("output_sha256", {}).items():
        actual = sha256_file(directory / name)
        if actual != expected:
            raise AssertionError(f"output hash mismatch {directory / name}: {actual} != {expected}")


def load_exact2(existing_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for manifest_path in sorted(existing_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = read_csv(manifest_path.with_name("pipeline_measurements.csv"))
        valid = [
            row
            for row in rows
            if row["status"] == "ok"
            and int(row["lossless"]) == 1
            and legal_family_unique(parse_pipeline(row["pipeline"]), 2)
        ]
        if not valid:
            raise AssertionError(f"no legal exact2 rows: {manifest_path}")
        best = min(valid, key=lambda row: (int(row["final_bytes"]), row["pipeline"]))
        key = manifest["dataset"], int(manifest["column_index"])
        result[key] = {
            "dataset": key[0],
            "column_index": key[1],
            "column_name": manifest["column_name"],
            "original_bytes": int(manifest["original_bytes"]),
            "input_sha256": manifest["input_sha256"],
            "exact2_bytes": int(best["final_bytes"]),
            "exact2_rate": float(best["compression_rate"]),
            "exact2_pipeline": best["pipeline"],
            "exact2_legal_candidates": len(valid),
        }
    if len(result) != 20:
        raise AssertionError(f"expected 20 exact2 columns, got {len(result)}")
    return result


def load_new(
    new_root: Path,
    allow_partial: bool,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    beams: dict[tuple[str, int], dict[str, Any]] = {}
    exacts: dict[tuple[str, int], dict[str, Any]] = {}
    source_pairs: set[tuple[str, str, str]] = set()
    for manifest_path in sorted((new_root / "beam").glob("*/manifest.json")):
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_manifest_outputs(directory, manifest)
        if manifest["mode"] != "beam" or manifest["widths"] != list(WIDTHS):
            raise AssertionError(f"unexpected beam mode/widths: {manifest_path}")
        key = manifest["dataset"], int(manifest["column_index"])
        if key in beams:
            raise AssertionError(f"duplicate beam key {key}")
        by_width = {int(row["width"]): row for row in manifest["result"]}
        if tuple(sorted(by_width)) != WIDTHS:
            raise AssertionError(f"missing widths for {key}: {sorted(by_width)}")
        if any(int(row["failed_candidates"]) != 0 for row in by_width.values()):
            raise AssertionError(f"failed Beam candidate at {key}")
        beams[key] = {"manifest": manifest, "by_width": by_width, "directory": str(directory)}
        src = manifest["source_sha256"]
        source_pairs.add((src["runner"], src["runtime"], src["loader"]))
    if not allow_partial and len(beams) != 20:
        raise AssertionError(f"expected 20 Beam columns, got {len(beams)}")
    if allow_partial and not beams:
        raise AssertionError("partial analysis still requires at least one complete Beam column")
    if len(source_pairs) != 1:
        raise AssertionError(f"mixed Beam source snapshots: {source_pairs}")
    exact_dir = new_root / "exact"
    if exact_dir.is_dir():
        for manifest_path in sorted(exact_dir.glob("*/manifest.json")):
            directory = manifest_path.parent
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_manifest_outputs(directory, manifest)
            if manifest["mode"] != "exact":
                raise AssertionError(f"unexpected exact manifest {manifest_path}")
            key = manifest["dataset"], int(manifest["column_index"])
            exacts[key] = manifest["result"]
    return beams, exacts


def percentile(values: Iterable[float], q: float) -> float:
    rows = list(values)
    return float(np.percentile(np.asarray(rows, dtype=np.float64), q)) if rows else math.nan


def summarize_group(rows: list[dict[str, Any]], width: int, dataset: str) -> dict[str, Any]:
    selected = [row for row in rows if dataset == "ALL" or row["dataset"] == dataset]
    bytes_total = sum(int(row[f"beam{width}_bytes"]) for row in selected)
    reference_total = sum(int(row["reference_bytes"]) for row in selected)
    regrets = [float(row[f"beam{width}_regret_pct"]) for row in selected]
    times = [float(row[f"beam{width}_wall_sec"]) for row in selected]
    return {
        "dataset": dataset,
        "width": width,
        "columns": len(selected),
        "total_bytes": bytes_total,
        "reference_total_bytes": reference_total,
        "aggregate_regret_pct": 100.0 * (bytes_total - reference_total) / reference_total,
        "median_column_regret_pct": statistics.median(regrets),
        "p90_column_regret_pct": percentile(regrets, 90),
        "max_column_regret_pct": max(regrets),
        "columns_over_1pct": sum(value > 1.0 for value in regrets),
        "median_search_wall_sec": statistics.median(times),
        "total_search_wall_sec": sum(times),
    }


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]], width_rows: list[dict[str, Any]]) -> str:
    teacher = summary["selected_teacher"]
    narrow = summary["beam29_stability"]
    header = "| 数据集 | 列 | exact<=2 rate | best-known rate | TB29 regret | TB64 | TB256 | TB1024 | observed depth-3 gain |"
    divider = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    data_lines = [header, divider]
    for row in rows:
        data_lines.append(
            "| {dataset} | {column_index}:{column_name} | {exact2_rate:.6f} | {reference_rate:.6f} | "
            "{beam29_regret_pct:.3f}% | {beam64_regret_pct:.3f}% | {beam256_regret_pct:.3f}% | "
            "{beam1024_regret_pct:.3f}% | {depth3_gain_vs_exact2_pct:.3f}% |".format(**row)
        )
    width_lines = [
        "| 范围 | Width | aggregate regret | median | p90 | max | >1% 列数 | median wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in width_rows:
        width_lines.append(
            f"| {row['dataset']} | {row['width']} | {row['aggregate_regret_pct']:.3f}% | "
            f"{row['median_column_regret_pct']:.3f}% | {row['p90_column_regret_pct']:.3f}% | "
            f"{row['max_column_regret_pct']:.3f}% | {row['columns_over_1pct']} | "
            f"{row['median_search_wall_sec']:.2f}s |"
        )
    return f"""# 更深、更宽合法 PV2F 搜索结果

## 一句话结论

{summary['conclusion_zh']}

这里的 **TypedBeam regret** 是同一列上 `(TypedBeam bytes / reference bytes - 1)`；0% 表示
TypedBeam 找到了当前参考最小值，越大表示它因剪枝漏掉了更好的 pipeline。非锚点参考是
exact depth<=2 与本轮所有宽度 TypedBeam 的最小值，因此只能称 best-known；两个 exact depth-3
锚点才提供全空间最优值。

## TypedBeam 教师校准

- 选定固定教师：TypedBeam-{teacher['width']}。每个 phase 各保留最多 width 个 prefix，
  与历史单一混合 frontier 的 Beam-N 不可直接数值比较。
- 教师 gate：**{teacher['gate']}**；aggregate {teacher['aggregate_regret_pct']:.3f}%，
  pooled p90 {teacher['p90_column_regret_pct']:.3f}%，worst {teacher['max_column_regret_pct']:.3f}%。
- exact depth-3 锚点数：{summary['exact_anchor_count']}；最大锚点 regret：
  {teacher['max_exact_anchor_regret_pct']:.3f}%。
- 完整 TypedBeam 列：{summary['columns']}/{summary['expected_columns']}；缺失：
  {', '.join(summary['missing_columns']) if summary['missing_columns'] else '无'}。
- 缺失列占 shallow exact reference bytes 的 {summary['missing_reference_byte_mass_pct']:.1f}%，
  因而 80% 的列数覆盖仅对应 {100.0 - summary['missing_reference_byte_mass_pct']:.1f}% 的 byte mass。
- TypedBeam-29 稳定大 gap gate：**{narrow['gate']}**；Air 中 >1% 为
  {narrow['air_columns_over_1pct']} 列，Metro 中 >1% 为 {narrow['metro_columns_over_1pct']} 列，
  pooled aggregate regret 为 {narrow['aggregate_regret_pct']:.3f}%。

## 宽度、质量和搜索时间

{chr(10).join(width_lines)}

时间是同一列按 29→64→256→1024 顺序共享不可变候选测量时，各宽度新增工作的边际
wall time；只能用于资源核算，不能当作宽度间公平 latency 比较。它不是生产 codec
latency，也不是未来 GRPO 的推理时间。

## 每列原始结果

{chr(10).join(data_lines)}

## 对 GRPO 的含义

本课题的合理主张不是让 GRPO 超过 TypedBeam 的压缩率，而是让它逼近冻结的教师：
测试列 aggregate regret <=1%，单列不超过 5%，同时端到端在线时间至少快 5 倍。
端到端时间必须包括 19 字段特征、策略推理和实际执行的 codec 候选；不能只报缓存表上的
selector latency。只有教师通过经验饱和校准后，才投入 robust GRPO 与真实 codec-cost 奖励。

## 审计边界

- 35 个动作均为可执行 PV2F 原型；0..3 个 transform 语义族不重复，随后 packer+compressor。
- 所有本轮候选均要求同一 runtime 的 decoder-only bitwise round trip；任一失败会使整列任务失败。
- 结果只适用于归一化 int64 研究 wire format，不代表生产内核吞吐、随机访问或查询性能。
- 机器可读结果见 `SUMMARY.json`、`per_column.csv` 和 `width_summary.csv`。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    exact2 = load_exact2(args.existing_root.resolve())
    beams, exacts = load_new(args.new_root.resolve(), args.allow_partial)
    if not args.allow_partial and set(exact2) != set(beams):
        raise AssertionError(f"column key mismatch: {set(exact2) ^ set(beams)}")
    if not set(beams).issubset(exact2):
        raise AssertionError(f"new results contain unknown columns: {set(beams) - set(exact2)}")
    per_column: list[dict[str, Any]] = []
    for key in sorted(beams):
        base = exact2[key]
        manifest = beams[key]["manifest"]
        if manifest["input_sha256"] != base["input_sha256"]:
            raise AssertionError(f"input hash mismatch {key}")
        by_width = beams[key]["by_width"]
        beam_bytes = {width: int(by_width[width]["final_bytes"]) for width in WIDTHS}
        best_known = min([base["exact2_bytes"], *beam_bytes.values()])
        reference_kind = "best_known"
        exact3 = exacts.get(key)
        if exact3 is not None:
            best_known = int(exact3["final_bytes"])
            reference_kind = "exact_depth3"
            if best_known > base["exact2_bytes"]:
                raise AssertionError(f"exact3 worse than exact2 at {key}")
        row: dict[str, Any] = {
            **base,
            "reference_kind": reference_kind,
            "reference_bytes": best_known,
            "reference_rate": best_known / base["original_bytes"],
            "reference_pipeline": exact3["pipeline"] if exact3 else min(
                [(base["exact2_bytes"], base["exact2_pipeline"])]
                + [(beam_bytes[width], by_width[width]["pipeline"]) for width in WIDTHS]
            )[1],
            "depth3_gain_vs_exact2_pct": 100.0 * (base["exact2_bytes"] - best_known) / base["exact2_bytes"],
        }
        for width in WIDTHS:
            result = by_width[width]
            row.update(
                {
                    f"beam{width}_bytes": beam_bytes[width],
                    f"beam{width}_rate": float(result["compression_rate"]),
                    f"beam{width}_regret_pct": 100.0 * (beam_bytes[width] - best_known) / best_known,
                    f"beam{width}_pipeline": result["pipeline"],
                    f"beam{width}_transform_count": int(result["transform_count"]),
                    f"beam{width}_evaluations": int(result["candidate_evaluations"]),
                    f"beam{width}_wall_sec": float(result["search_wall_sec"]),
                }
            )
        per_column.append(row)
    width_summary = [
        summarize_group(per_column, width, dataset)
        for dataset in ("Air-Quality", "Metro-Traffic", "ALL")
        for width in WIDTHS
    ]
    all_by_width = {row["width"]: row for row in width_summary if row["dataset"] == "ALL"}
    teacher_width = min((256, 1024), key=lambda width: (all_by_width[width]["aggregate_regret_pct"], width))
    teacher_row = all_by_width[teacher_width]
    anchor_regrets = [
        float(row[f"beam{teacher_width}_regret_pct"])
        for row in per_column
        if row["reference_kind"] == "exact_depth3"
    ]
    dataset_teacher = {
        row["dataset"]: row
        for row in width_summary
        if row["width"] == teacher_width and row["dataset"] != "ALL"
    }
    full_coverage = len(per_column) == len(exact2)
    missing_keys = set(exact2) - set(beams)
    total_reference_bytes = sum(int(row["exact2_bytes"]) for row in exact2.values())
    missing_reference_bytes = sum(int(exact2[key]["exact2_bytes"]) for key in missing_keys)
    missing_reference_byte_mass_pct = 100.0 * missing_reference_bytes / total_reference_bytes
    teacher_gate = full_coverage and (
        teacher_row["aggregate_regret_pct"] <= 0.5
        and all(row["median_column_regret_pct"] <= 0.5 for row in dataset_teacher.values())
        and teacher_row["p90_column_regret_pct"] <= 1.0
        and teacher_row["max_column_regret_pct"] <= 2.0
        and len(anchor_regrets) >= 2
        and max(anchor_regrets) <= 0.5
    )
    beam29_all = all_by_width[29]
    air_over = next(row for row in width_summary if row["dataset"] == "Air-Quality" and row["width"] == 29)["columns_over_1pct"]
    metro_over = next(row for row in width_summary if row["dataset"] == "Metro-Traffic" and row["width"] == 29)["columns_over_1pct"]
    narrow_gate = full_coverage and beam29_all["aggregate_regret_pct"] > 1.0 and air_over >= 2 and metro_over >= 2
    if not full_coverage:
        conclusion = (
            f"四小时预算仅完成 {len(per_column)}/{len(exact2)} 列，缺失列占 shallow-reference bytes "
            f"的 {missing_reference_byte_mass_pct:.1f}%；当前统计是预算截断描述，"
            "不能判定经验饱和教师 gate，也不能据此投入 robust GRPO。"
        )
    elif teacher_gate:
        conclusion = (
            f"TypedBeam-{teacher_width} 通过当前经验饱和教师 gate；可进入以压缩损失换速度的 GRPO，"
            "但必须冻结教师并用端到端真实 codec 时间评估。"
        )
    else:
        conclusion = (
            f"TypedBeam-{teacher_width} 尚未通过经验饱和教师 gate；暂不应继续投入 robust GRPO，"
            "应先修正/加宽 Beam 或补充精确锚点。"
        )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "columns": len(per_column),
        "expected_columns": len(exact2),
        "full_coverage": full_coverage,
        "missing_columns": [
            f"{dataset}:{column_index}"
            for dataset, column_index in sorted(missing_keys)
        ],
        "total_shallow_reference_bytes": total_reference_bytes,
        "missing_shallow_reference_bytes": missing_reference_bytes,
        "missing_reference_byte_mass_pct": missing_reference_byte_mass_pct,
        "datasets": sorted({row["dataset"] for row in per_column}),
        "widths": list(WIDTHS),
        "exact_anchor_count": len(anchor_regrets),
        "selected_teacher": {
            **teacher_row,
            "gate": "INCONCLUSIVE" if not full_coverage else ("PASS" if teacher_gate else "FAIL"),
            "max_exact_anchor_regret_pct": max(anchor_regrets) if anchor_regrets else math.nan,
        },
        "beam29_stability": {
            "gate": "INCONCLUSIVE" if not full_coverage else ("PASS" if narrow_gate else "FAIL"),
            "aggregate_regret_pct": beam29_all["aggregate_regret_pct"],
            "air_columns_over_1pct": air_over,
            "metro_columns_over_1pct": metro_over,
        },
        "teacher_gate_definition": "aggregate<=0.5%, each-dataset median<=0.5%, pooled p90<=1%, worst<=2%, >=2 exact anchors and max anchor regret<=0.5%",
        "beam29_stability_definition": "aggregate>1% and at least two >1% columns in each dataset",
        "future_grpo_gate": "aggregate regret vs frozen teacher<=1%, every column<=5%, end-to-end online speedup>=5x",
        "conclusion_zh": conclusion,
        "source_roots": {
            "new": str(args.new_root.resolve()),
            "exact2": str(args.existing_root.resolve()),
        },
        "claim_boundary": "non-anchor reference is best-known, not exact; standalone normalized-int64 PV2F research format",
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_column.csv", per_column)
    write_csv(output_dir / "width_summary.csv", width_summary)
    write_json(output_dir / "SUMMARY.json", summary)
    report = render_report(summary, per_column, width_summary)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    (output_dir / f"REPORT_{stamp}.md").write_text(report, encoding="utf-8")
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
