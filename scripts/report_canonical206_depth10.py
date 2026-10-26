#!/usr/bin/env python3
"""Audit and report the depth-10 probe without importing or executing codecs.

Final output requires the runner's COMPLETE.json PASS and every planned row.
--allow-partial labels incomplete coverage; --check-only never writes files.
All comparisons use integer complete-frame bytes, never rounded ratios.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis_outputs/depth10_probe_20260906_v1"
DEFAULT_PARENT = ROOT / "analysis_outputs/canonical206_beam29_ppo_grpo_smallest2_20260906_v1"
DEFAULT_AUDIT = ROOT / "analysis_outputs/reference_carrier_audit_20260906_v1/AUDIT.json"
METHODS = ("grpo_depth10", "frozen_depth10")
LABELS = {"grpo_depth10": "GRPO 搜索", "frozen_depth10": "不学习搜索"}
TIMES = ("compress_sec", "decompress_sec", "prior_verification_sec", "shared_probe_sec",
         "search_train_sec", "compose_sec")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取完整 JSON：{path}：{exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} 必须是 >= {minimum} 的整数，实际 {value!r}")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} 不是时间数值")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} 必须为有限非负数")
    return number


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["dataset"]), integer(row["numeric_index"], "numeric_index")


def comparison(left: int, right: int) -> str:
    return "win" if left < right else "tie" if left == right else "loss"


def ratio(raw: int, size: int) -> float:
    return raw / integer(size, "完整文件字节数", 1)


def check_ratio(actual: Any, raw: int, size: int, label: str) -> None:
    if not math.isclose(float(actual), ratio(raw, size), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} 不是原始字节数 / 完整文件字节数")


def chain_display(ids: list[int], names: list[str]) -> str:
    """Asterisk marks compound APIs without inventing extra registered IDs."""
    aliases = {1: "Identity-ref*", 7: "Delta-ref*", 11: "ZigZag-ref*"}
    return " → ".join(f"{aliases.get(i, name.removeprefix('op.'))}[{i}]"
                      for i, name in zip(ids, names)) or "（无算子）"


def histogram(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 缺少深度直方图")
    output = {}
    for depth, count in value.items():
        if str(int(depth)) != str(depth) or not 0 <= int(depth) <= 10:
            raise ValueError(f"{label} 出现非法/超过 10 的注册 ID 深度：{depth}")
        output[str(depth)] = integer(count, f"{label}[{depth}]")
    return output


def depth_summary(stats: dict[str, Any], online: bool) -> dict[str, Any]:
    if stats.get("max_atomic_depth") != 10:
        raise ValueError("search_stats.max_atomic_depth 必须为 10")
    if stats.get("automatic_backend_completion") is not False:
        raise ValueError("缺少关闭自动后端补全的审计记录")
    attempt = histogram(stats["candidate_depth_histogram_attempt"], "candidate attempt")
    success = histogram(stats["candidate_depth_histogram_ok"], "candidate ok")
    verified = histogram(stats["candidate_depth_histogram_verified_unique"], "verified unique")
    provenance = stats.get("candidate_depth_histograms_by_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("缺少 provenance 深度记录")
    normalized = {}
    for name, values in provenance.items():
        normalized[name] = {field: histogram(values[field], f"{name}.{field}")
                            for field in ("attempt", "ok", "unique")}
        for depth, count in normalized[name]["ok"].items():
            if count > normalized[name]["attempt"].get(depth, 0):
                raise ValueError(f"{name} 深度 {depth} 成功数超过尝试数")
    # Method clones can retain shared records; only exact 'search' provenance
    # represents fresh online calls. No subtraction of set histograms is valid.
    phase = normalized.get("search", {"attempt": {}, "ok": {}, "unique": {}}) if online else {
        field: dict(Counter({str(depth): sum(v[field].get(str(depth), 0)
                                             for name, v in normalized.items()
                                             if name.startswith("shared_"))
                             for depth in range(11)}))
        for field in ("attempt", "ok", "unique")
    }
    return {
        "definition": stats.get("depth_definition"),
        "phase_provenance": "search" if online else "shared_* provenance (excludes baseline/prior)",
        "phase": phase,
        "phase_9_10": {str(d): {f: phase[f].get(str(d), 0) for f in ("attempt", "ok", "unique")}
                       for d in (9, 10)},
        "all_recorded_attempt": attempt,
        "all_recorded_ok": success,
        "all_verified_unique": verified,
        "by_provenance": normalized,
        "depth_9_10_evidence": stats.get("depth_9_10_evidence", {}),
        "max_candidate_depth_attempted": stats.get("max_candidate_depth_attempted"),
        "max_verified_candidate_depth": stats.get("max_verified_candidate_depth"),
        "depth_guard_rejections": stats.get("depth_guard_rejections", 0),
        "grammar_guard_rejections": stats.get("grammar_guard_rejections", 0),
    }


def verify_phase_trace(depth: dict[str, Any], path: Path, online: bool) -> None:
    """Count public candidates once even when provenance labels overlap."""
    trace = read_json(path)
    rows = [row for row in trace["options"] if (row["provenance"] == "search" if online
                                                else row["provenance"].startswith("shared_"))]
    phase = {"attempt": Counter(), "ok": Counter(), "unique": Counter()}
    distinct = set()
    for row in rows:
        ids = row["ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 10:
            raise ValueError(f"trace 候选 ID 深度越界：{path}")
        phase["attempt"][str(len(ids))] += 1
        if row["status"] == "ok":
            phase["ok"][str(len(ids))] += 1
        distinct.add(tuple(ids))
    phase["unique"] = Counter(str(len(ids)) for ids in distinct)
    for field in ("attempt", "ok"):
        expected = {k: v for k, v in depth["phase"][field].items() if v}
        if dict(phase[field]) != expected:
            raise ValueError(f"trace 深度 {field} 与 search_stats 不一致：{path}")
    depth["phase"] = {field: dict(counts) for field, counts in phase.items()}
    depth["phase_9_10"] = {str(d): {field: phase[field].get(str(d), 0)
                                   for field in ("attempt", "ok", "unique")} for d in (9, 10)}
    depth["trace_source"] = str(path)
    depth["trace_source_sha256"] = sha256(path)


def verify_row(row: dict[str, Any], column: dict[str, Any], path: Path,
               run: Path, registry: dict[int, str]) -> dict[str, Any]:
    raw = integer(row["original_bytes"], "original_bytes", 1)
    if (row["column"] != column["name"] or raw != column["raw_bytes"]
            or row["input_sha256"] != column["input_sha256"]):
        raise ValueError(f"输入列不匹配：{path}")
    sizes = {name: integer(row[name], name, 1) for name in
             ("fixed_bytes", "prior_best_bytes", "shared_best_bytes", "compressed_bytes")}
    if not sizes["compressed_bytes"] <= sizes["shared_best_bytes"] <= sizes["prior_best_bytes"] <= sizes["fixed_bytes"]:
        raise ValueError(f"先验/公共/线上保底顺序不一致：{path}")
    check_ratio(row["compression_ratio"], raw, sizes["compressed_bytes"], str(path))
    check_ratio(row["fixed_ratio"], raw, sizes["fixed_bytes"], f"{path}:fixed")
    if row.get("lossless") is not True:
        raise ValueError(f"缺少上游完整无损重放 PASS：{path}")
    ids = row["selected_ids"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 10:
        raise ValueError(f"获胜链注册 ID 数不在 1..10：{path}")
    for action in ids:
        integer(action, "action ID")
        if action not in registry:
            raise ValueError(f"未知注册 ID {action}：{path}")
    if row["selected_pipeline"] != [registry[i] for i in ids]:
        raise ValueError(f"注册 ID 与算子名不一致：{path}")
    for field, stats_field in (("selected_ids", "best_ids"), ("compressed_bytes", "best_frame_bytes"),
                               ("selected_provenance", "best_first_provenance")):
        if row[field] != row["search_stats"][stats_field]:
            raise ValueError(f"所选链/字节/来源与搜索最佳记录不一致：{path}:{field}")
    for name in TIMES:
        finite(row[name], f"{path.name}:{name}")
    if not math.isclose(row["compose_sec"], sum(row[name] for name in
                        ("prior_verification_sec", "shared_probe_sec", "search_train_sec")),
                        rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"组合时间不是先验验证 + 公共探测 + 线上搜索：{path}")
    repeats = row["compress_sec_repeats"]
    if not isinstance(repeats, list) or not repeats:
        raise ValueError(f"缺少独立编码时间重放：{path}")
    for value in repeats:
        finite(value, "compress_sec_repeats")
    if not math.isclose(row["compress_sec"], statistics.median(repeats), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"编码时间不是独立重复测量的中位数：{path}")
    updates = integer(row["completed_update_iterations"], "completed_update_iterations")
    if not isinstance(row["weights_changed"], bool):
        raise ValueError(f"weights_changed 不是布尔值：{path}")
    if updates == 0 and row["weights_changed"]:
        raise ValueError(f"零次更新却报告权重改变：{path}")
    if row["method"] == "frozen_depth10" and (updates or row["weights_changed"]):
        raise ValueError(f"不学习对照发生权重更新：{path}")
    frame = (run / row["frame_path"]).resolve()
    if not frame.is_relative_to(run):
        raise ValueError(f"frame_path 越过实验目录：{path}")
    if not frame.is_file() or frame.stat().st_size != sizes["compressed_bytes"] or sha256(frame) != row["frame_sha256"]:
        raise ValueError(f"完整 frame 的大小 / SHA256 错误：{path}")
    result = dict(row)
    result.update(source=str(path), source_sha256=sha256(path), frame_path=str(frame),
                  compression_ratio=ratio(raw, sizes["compressed_bytes"]),
                  selected_depth=len(ids), chain_display=chain_display(ids, row["selected_pipeline"]),
                  depth=depth_summary(row["search_stats"], online=True),
                  bytes_saved_vs_fixed=sizes["fixed_bytes"] - sizes["compressed_bytes"],
                  new_bytes_vs_prior=sizes["prior_best_bytes"] - sizes["compressed_bytes"],
                  online_new_bytes_vs_shared=sizes["shared_best_bytes"] - sizes["compressed_bytes"],
                  learning_completed=bool(updates and row["weights_changed"]),
                  lossless_evidence="runner replay PASS; reporter checked size and SHA256 only")
    for name, field in (("online_option_depth_attempts", "attempt"), ("online_option_depth_ok", "ok")):
        if histogram(row[name], name) != result["depth"]["phase"][field]:
            raise ValueError(f"线上深度切片与 provenance=search 不一致：{path}:{name}")
    verify_phase_trace(result["depth"], run / "traces" / path.name, online=True)
    trace = read_json(run / "traces" / path.name)
    if len(trace["updates"]) != updates:
        raise ValueError(f"完成更新次数与实际 trace.updates 不符：{path}")
    if [item["iteration"] for item in trace["updates"]] != list(range(1, updates + 1)):
        raise ValueError(f"trace 更新迭代号不连续：{path}")
    if any(item["seed"] != row["seed"] or integer(item["optimizer_steps"], "optimizer_steps") == 0
           for item in trace["updates"]):
        raise ValueError(f"trace 更新 seed / optimizer 步数不正确：{path}")
    if row["weights_changed"] and not any(finite(item["parameter_change_l2"], "parameter_change_l2") > 0
                                          for item in trace["updates"]):
        raise ValueError(f"报告权重改变但所有已完成更新的 L2 改变量均为零：{path}")
    if trace["shared_snapshot_sha256"] != row["shared_initial_snapshot_sha256"]:
        raise ValueError(f"trace 的公共快照指纹与结果行不符：{path}")
    if len(trace["atomic"]) != integer(row["online_unique_atomic_edges"], "online_unique_atomic_edges"):
        raise ValueError(f"线上新增注册 ID 步骤记录数与 trace.atomic 不一致：{path}")
    return result


def shared_value(shared: dict[str, Any], field: str) -> Any:
    if field in shared:
        return shared[field]
    # Explicitly accommodate nested winner records, not guessed default bytes.
    nested = {"prior_best_bytes": ("prior_best", "compressed_bytes"),
              "shared_best_bytes": ("shared_best", "compressed_bytes")}
    if field in nested:
        obj, name = nested[field]
        return shared.get(obj, {}).get(name)
    return None


def validate_shared(shared: dict[str, Any], column: dict[str, Any], rows: list[dict[str, Any]],
                    path: Path, run: Path) -> dict[str, Any]:
    if "dataset" in shared and key(shared) != key(column):
        raise ValueError(f"公共记录列标识不一致：{path}")
    if "input_sha256" in shared and shared["input_sha256"] != column["input_sha256"]:
        raise ValueError(f"公共记录输入哈希不一致：{path}")
    prior = integer(shared_value(shared, "prior_best_bytes"), "shared.prior_best_bytes", 1)
    best = integer(shared_value(shared, "shared_best_bytes"), "shared.shared_best_bytes", 1)
    if best > prior:
        raise ValueError(f"公共探测丢失先验保底：{path}")
    for row in rows:
        if row["prior_best_bytes"] != prior or row["shared_best_bytes"] != best:
            raise ValueError(f"两方法没有使用相同先验/公共最佳值：{path}")
        for field in ("prior_best_source", "prior_verification_sec", "shared_probe_sec"):
            if row[field] != shared[field]:
                raise ValueError(f"方法行与公共阶段 {field} 不一致：{path}")
    fingerprint = shared.get("shared_initial_snapshot_sha256")
    fingerprints = [row.get("shared_initial_snapshot_sha256") for row in rows]
    if fingerprint is not None:
        snapshot = (run / shared["snapshot_path"]).resolve()
        if not snapshot.is_relative_to(run) or not snapshot.is_file() or sha256(snapshot) != fingerprint:
            raise ValueError(f"公共缓存快照文件 SHA256 不匹配：{path}")
        if any(item != fingerprint for item in fingerprints):
            raise ValueError(f"公共缓存快照 SHA256 与方法行不一致：{path}")
        cache_audit = "fingerprint_equal" if len(rows) == 2 else "one_method_observed"
    elif any(value is not None for value in fingerprints):
        raise ValueError(f"方法行有缓存指纹但公共记录没有：{path}")
    else:
        cache_audit = "not_recorded; only prior/shared best-byte equality checked"
    stats = shared.get("search_stats", shared.get("stats"))
    if stats is None:
        raise ValueError(f"公共阶段缺少 search_stats：{path}")
    prior_sources = shared.get("prior_sources")
    if not isinstance(prior_sources, list) or not prior_sources:
        raise ValueError(f"公共记录缺少先验来源清单：{path}")
    for source in prior_sources:
        source_path = Path(source["source"])
        if sha256(source_path) != source["source_sha256"]:
            raise ValueError(f"先验结果来源 SHA256 已改变：{source_path}")
        archived = read_json(source_path)
        if (archived["input_sha256"] != column["input_sha256"] or archived.get("lossless") is not True
                or any(archived[field] != source[field] for field in
                       ("compressed_bytes", "selected_ids", "frame_sha256"))):
            raise ValueError(f"先验来源字段/输入/无损证据不匹配：{source_path}")
    if min(source["compressed_bytes"] for source in prior_sources) != prior:
        raise ValueError(f"先验最佳不是列出的全部已知候选最小值：{path}")
    result = dict(shared)
    result.update(source=str(path), source_sha256=sha256(path), prior_best_bytes=prior,
                  shared_best_bytes=best, new_bytes_vs_prior=prior - best,
                  cache_equality_audit=cache_audit, depth=depth_summary(stats, online=False))
    verify_phase_trace(result["depth"], path.with_suffix(".trace.json"), online=False)
    return result


def carrier_evidence(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    if audit.get("status") != "PASS":
        raise ValueError("reference carrier AUDIT 不是 PASS")
    report = path.parent / "REPORT_compact_zlib6.md"
    report_hash = sha256(report)
    expected = audit.get("generated_artifact_hashes", {}).get(str(report))
    if expected and expected != report_hash:
        raise ValueError("reference carrier 文字报告 SHA256 已改变")
    ablation = audit["nmhc_identity_vs_zigzag"]
    for row in ablation["rows"]:
        check_ratio(row["compression_ratio"], row["original_bytes"], row["complete_frame_bytes"], "carrier ablation")
    residual_report = path.parent / "BZIP2_RESIDUAL_AUDIT.md"
    residual_source = ROOT / "cpp/src/general_codec_residual_stages.cpp"
    xz_report = path.parent / "XZ_FRAME_OMISSION_AUDIT.md"
    return {"audit": str(path), "audit_sha256": sha256(path), "status": "PASS",
            "report": str(report), "report_sha256": report_hash,
            "bzip2_residual_report": str(residual_report) if residual_report.exists() else None,
            "bzip2_residual_report_sha256": sha256(residual_report) if residual_report.exists() else None,
            "bzip2_residual_source": str(residual_source), "bzip2_residual_source_sha256": sha256(residual_source),
            "xz_frame_omission_report": str(xz_report) if xz_report.exists() else None,
            "xz_frame_omission_report_sha256": sha256(xz_report) if xz_report.exists() else None,
            "nmhc_identity_vs_zigzag": ablation,
            "interpretation": "Reference IDs 1/7/11 are compound APIs with integer-width compact and zlib6; ID1 is identity on integers."}


def xz_wrapper_attribution(columns: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    """Inspect existing transport bytes only; never decode or recompress XZ.

    The audited RH pair differs solely by the final registered container ID.
    Bind the claim to the current verified files instead of copying a 152-byte
    constant into results from another run or a different selected chain.
    """
    column = next((col for col in columns if col["dataset"] == "Air-Quality" and col["column"] == "RH"), None)
    if not column or any(not column.get(method) for method in METHODS):
        return {"status": "NOT_OBSERVED", "reason": "RH method pair is not complete"}
    grpo, frozen = (column[method] for method in METHODS)
    if grpo["selected_ids"] != frozen["selected_ids"] + [135]:
        return {"status": "NOT_APPLICABLE", "reason": "Current selected pair is not the audited XZ-container omission pattern"}
    if not evidence.get("xz_frame_omission_report"):
        raise ValueError("RH 出现 XZ container 省略模式，但缺少 XZ_FRAME_OMISSION_AUDIT.md 来源")

    def unpack(row: dict[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
        frame = Path(row["frame_path"]).read_bytes()
        if len(frame) != row["compressed_bytes"] or hashlib.sha256(frame).hexdigest() != row["frame_sha256"]:
            raise ValueError("XZ 封装归因期间 frame 大小 / SHA256 改变")
        if frame[:4] != b"C206" or frame[18:22] != b"UAR1":
            raise ValueError("XZ 封装归因需要既有 C206/UAR1 完整文件")
        header_size = int.from_bytes(frame[22:26], "little")
        if 26 + header_size > len(frame):
            raise ValueError("XZ 封装归因的 UAR1 metadata 长度越界")
        header = json.loads(frame[26:26 + header_size])
        payload = frame[26 + header_size:]
        accounting = {"frame_bytes": len(frame), "c206_header_bytes": 18,
                      "uar1_fixed_header_bytes": 8, "uar1_json_bytes": header_size,
                      "payload_bytes": len(payload), "frame_sha256": row["frame_sha256"],
                      "payload_sha256": hashlib.sha256(payload).hexdigest(),
                      "selected_ids": row["selected_ids"], "selected_provenance": row["selected_provenance"]}
        return header, payload, accounting

    gh, gp, ga = unpack(grpo)
    fh, fp, fa = unpack(frozen)
    if (not gp.startswith(b"\xfd7zXZ\x00") or fp[:4] != b"XZC1" or fp[8:] != gp
            or int.from_bytes(fp[4:8], "little") != len(gp) - 24
            or fh != gh.get("upstream_result_header")):
        raise ValueError("RH 末135省略模式未通过同一 native XZ 流/上游 metadata 逐字节校验，不能归因纯封装")
    metadata_saved = ga["uar1_json_bytes"] - fa["uar1_json_bytes"]
    packet_extra = fa["payload_bytes"] - ga["payload_bytes"]
    net_saved = grpo["compressed_bytes"] - frozen["compressed_bytes"]
    if net_saved != metadata_saved - packet_extra:
        raise ValueError("XZ 封装字节账不平衡")
    return {"status": "VERIFIED_EXISTING_BYTES", "dataset": column["dataset"], "column": column["column"],
            "seed": column["seed"], "audit": evidence["xz_frame_omission_report"],
            "audit_sha256": evidence["xz_frame_omission_report_sha256"],
            "grpo_depth10": ga, "frozen_depth10": fa,
            "same_native_xz_payload": True, "native_xz_bytes": len(gp),
            "native_xz_sha256": hashlib.sha256(gp).hexdigest(),
            "native_payload_bytes_saved": 0, "uar1_json_bytes_saved": metadata_saved,
            "xzc1_packet_extra_bytes": packet_extra, "net_frame_bytes_saved": net_saved,
            "grpo_new_bytes_vs_prior": grpo["new_bytes_vs_prior"],
            "frozen_new_bytes_vs_prior": frozen["new_bytes_vs_prior"],
            "shared_new_bytes_vs_prior": column["prior_best_bytes"] - column["shared_best_bytes"],
            "grpo_online_new_bytes_vs_shared": grpo["online_new_bytes_vs_shared"],
            "frozen_online_new_bytes_vs_shared": frozen["online_new_bytes_vs_shared"],
            "scope": "Only the paired GRPO/frozen difference is attributed to wrapping; common improvement versus prior is not decomposed here. No codec was executed."}


def paired_totals(columns: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [col for col in columns if all(col.get(m) for m in METHODS) and col.get("shared")]
    raw = sum(col["original_bytes"] for col in paired)
    sizes = {name: sum(col[name] for col in paired)
             for name in ("fixed_bytes", "prior_best_bytes", "shared_best_bytes")}
    counts = Counter(comparison(col[METHODS[0]]["compressed_bytes"], col[METHODS[1]]["compressed_bytes"])
                     for col in paired)
    methods = {}
    for method in METHODS:
        measured = [col[method] for col in paired]
        size = sum(row["compressed_bytes"] for row in measured)
        methods[method] = {
            "compressed_bytes": size, "compression_ratio": raw / size if size else None,
            "new_bytes_vs_prior": sizes["prior_best_bytes"] - size,
            "online_new_bytes_vs_shared": sizes["shared_best_bytes"] - size,
            "strict_new_columns": sum(row["new_bytes_vs_prior"] > 0 for row in measured),
            "strict_online_new_columns": sum(row["online_new_bytes_vs_shared"] > 0 for row in measured),
            "updated_columns": sum(row["learning_completed"] for row in measured),
            "zero_update_columns": sum(row["completed_update_iterations"] == 0 for row in measured),
            "zero_update_online_new_bytes": sum(row["online_new_bytes_vs_shared"] for row in measured
                                                if row["completed_update_iterations"] == 0),
            "compress_sec_sum": sum(row["compress_sec"] for row in measured),
            "search_train_sec_sum": sum(row["search_train_sec"] for row in measured),
            "compose_sec_sum": sum(row["compose_sec"] for row in measured),
        }
    common = {name: sum(col[METHODS[0]][name] for col in paired)
              for name in ("prior_verification_sec", "shared_probe_sec")}
    return {"paired_columns": len(paired), "expected_columns": len(columns), "raw_bytes": raw,
            **sizes, "ratios": {name: raw / size if size else None for name, size in sizes.items()},
            "shared_new_bytes_vs_prior": sizes["prior_best_bytes"] - sizes["shared_best_bytes"],
            "methods": methods,
            "grpo_vs_frozen": {"wins": counts["win"], "ties": counts["tie"], "losses": counts["loss"],
                               "bytes_saved": methods[METHODS[1]]["compressed_bytes"] - methods[METHODS[0]]["compressed_bytes"]},
            "common_phase_seconds_counted_once": common,
            "accounted_phase_sec": sum(common.values()) + sum(m["search_train_sec_sum"] for m in methods.values()),
            "time_caveat": "Phase sum is not end-to-end wall time; excludes historical prior discovery and final replay/report overhead. Do not sum both methods' compose times as shared cost occurs twice."}


def collect(run: Path, parent_override: Path | None, audit: Path, allow_partial: bool,
            require_twenty: bool = True) -> dict[str, Any]:
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    parent = (parent_override or Path(manifest.get("parent", DEFAULT_PARENT))).resolve()
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = read_json(parent_manifest_path)
    if manifest.get("parent_manifest_sha256") and sha256(parent_manifest_path) != manifest["parent_manifest_sha256"]:
        raise ValueError("旧实验 manifest SHA256 不匹配")
    registry_path = parent / "action_registry.json"
    if manifest.get("registry_sha256") and sha256(registry_path) != manifest["registry_sha256"]:
        raise ValueError("206 算子注册表 SHA256 不匹配")
    registry = {int(row["policy_action_id"]): row["name"] for row in read_json(registry_path)["policy_actions"]}
    plan = manifest.get("columns", parent_manifest["columns"])
    planned = {key(col): col for col in plan}
    if len(planned) != len(plan) or not plan or (require_twenty and len(plan) != 20):
        raise ValueError("本轮必须是唯一的 20 列计划；小样本调试不能标作此正式实验")
    for col in plan:
        original = next((c for c in parent_manifest["columns"] if key(c) == key(col)), None)
        if original is None or any(col[f] != original[f] for f in ("name", "raw_bytes", "input_sha256")):
            raise ValueError(f"计划列与旧实验输入不一致：{key(col)}")
    config = manifest["config"]
    if config.get("max_depth") != 10:
        raise ValueError("本报告只针对最多 10 注册 ID 的配置")
    if manifest.get("methods") != list(METHODS):
        raise ValueError("manifest methods 与本次配对方法不一致")
    if manifest.get("previous"):
        previous_manifest = Path(manifest["previous"]) / "manifest.json"
        if sha256(previous_manifest) != manifest["previous_manifest_sha256"]:
            raise ValueError("v3 先验 manifest SHA256 不匹配")
    for source, digest in manifest.get("source_hashes", {}).items():
        if sha256(run / "code" / Path(source).name) != digest:
            raise ValueError(f"保存的执行源码快照 SHA256 不匹配：{source}")
    seeds = config.get("seeds", [config["seed"]] if "seed" in config else [])
    if len(seeds) != 1:
        raise ValueError("本报告仅支持单 seed 深度 10 实验，不允许暗中 best-of 多 seed")
    seed = integer(seeds[0], "seed")
    expected_rows = len(plan) * len(METHODS)
    archived_by_column: dict[tuple[str, int], list[dict[str, Any]]] = {ident: [] for ident in planned}
    # Rebuild the entire eligible prior pool independently; an omitted smaller
    # archive row must not inflate the alleged new gain.
    excluded = {action for action, name in registry.items() if ".camel." in name}
    archives = [*sorted((parent / "columns").glob("*.json")),
                *sorted((Path(manifest["previous"]) / "columns").glob("*.json"))]
    archives += [parent / "fixed_algorithm_baseline_v1" / "columns" / f"{d}__{i:02d}.json" for d, i in planned]
    for path in archives:
        record = read_json(path)
        if key(record) not in planned:
            continue
        ids = record["selected_ids"]
        if not 0 < len(ids) <= 10 or len(set(ids)) != len(ids) or excluded.intersection(ids):
            continue
        record = dict(record, source=str(path.resolve()))
        archived_by_column[key(record)].append(record)
    found = {}
    for path in sorted((run / "columns").glob("*.json")):
        row = read_json(path)
        identifier = (str(row["method"]), integer(row["seed"], "row.seed"), *key(row))
        if identifier[0] not in METHODS or identifier[1] != seed or key(row) not in planned:
            raise ValueError(f"意外方法/seed/列：{path}")
        if identifier in found:
            raise ValueError(f"重复结果键：{identifier}")
        found[identifier] = verify_row(row, planned[key(row)], path, run, registry)
        if len(row["compress_sec_repeats"]) != config["repeats"]:
            raise ValueError(f"编码时间重复次数与 config 不符：{path}")
    columns = []
    missing = []
    for col in plan:
        col_key = key(col)
        rows = {method: found.get((method, seed, *col_key)) for method in METHODS}
        present = [row for row in rows.values() if row]
        common_path = run / "shared" / f"{col_key[0]}__{col_key[1]:02d}.json"
        common = validate_shared(read_json(common_path), col, present, common_path, run) if common_path.exists() else None
        if common:
            expected_sources = {row["source"] for row in archived_by_column[col_key]}
            observed_sources = [str(Path(row["source"]).resolve()) for row in common["prior_sources"]]
            if len(observed_sources) != len(set(observed_sources)) or set(observed_sources) != expected_sources:
                raise ValueError(f"公共先验清单漏掉/重复/多出历史候选：{col_key}")
            winner = min(archived_by_column[col_key], key=lambda row: (row["compressed_bytes"], row["selected_ids"], row["source"]))
            if common["prior_best_bytes"] != winner["compressed_bytes"] or common["prior_best_source"] != winner["source"]:
                raise ValueError(f"已知最佳不是完整历史池的确定性最小值：{col_key}")
        for method, row in rows.items():
            if row is None:
                missing.append(f"{method}/{col_key[0]}__{col_key[1]:02d}")
        if common is None:
            missing.append(f"shared/{col_key[0]}__{col_key[1]:02d}")
        fixed_path = parent / "fixed_algorithm_baseline_v1" / "columns" / f"{col_key[0]}__{col_key[1]:02d}.json"
        fixed = read_json(fixed_path)
        if fixed["input_sha256"] != col["input_sha256"]:
            raise ValueError(f"固定基准输入不匹配：{fixed_path}")
        fixed_bytes = integer(fixed["compressed_bytes"], "fixed bytes", 1)
        for row in present:
            if row["fixed_bytes"] != fixed_bytes or row["fixed_algorithm"] != fixed["algorithm"]:
                raise ValueError(f"固定基准改变：{row['source']}")
        if len(present) == 2:
            for field in ("prior_best_source", "prior_verification_sec", "shared_probe_sec"):
                if present[0][field] != present[1][field]:
                    raise ValueError(f"两方法共享阶段资料不一致：{col_key}:{field}")
        prior = common["prior_best_bytes"] if common else present[0]["prior_best_bytes"] if present else None
        shared = common["shared_best_bytes"] if common else present[0]["shared_best_bytes"] if present else None
        result = {"dataset": col_key[0], "numeric_index": col_key[1], "column": col["name"],
                  "seed": seed, "original_bytes": col["raw_bytes"], "input_sha256": col["input_sha256"],
                  "fixed_algorithm": fixed["algorithm"], "fixed_bytes": fixed_bytes,
                  "fixed_ratio": ratio(col["raw_bytes"], fixed_bytes), "fixed_source": str(fixed_path),
                  "prior_best_bytes": prior, "prior_best_ratio": ratio(col["raw_bytes"], prior) if prior else None,
                  "prior_best_source": present[0]["prior_best_source"] if present else None,
                  "shared_best_bytes": shared, "shared_best_ratio": ratio(col["raw_bytes"], shared) if shared else None,
                  "shared": common, **rows}
        result["grpo_vs_frozen_bytes_saved"] = (rows[METHODS[1]]["compressed_bytes"] - rows[METHODS[0]]["compressed_bytes"]
                                                 if len(present) == 2 else None)
        columns.append(result)
    complete_path = run / "COMPLETE.json"
    complete = read_json(complete_path) if complete_path.exists() else None
    marker_ok = bool(complete and complete.get("status") == "PASS"
                     and complete.get("expected_rows") == expected_rows
                     and complete.get("actual_rows") == expected_rows
                     and complete.get("frozen_after", {}).get("status") == "PASS"
                     and complete.get("new_sources_unchanged") is True)
    if complete and complete.get("status") == "PASS" and len(found) != complete.get("actual_rows"):
        raise ValueError("COMPLETE PASS 行数与实际已验证文件不一致")
    ready = not missing and len(found) == expected_rows and marker_ok
    if not ready and not allow_partial:
        raise ValueError(f"未完整结束：已校验 {len(found)}/{expected_rows} 行，COMPLETE PASS={marker_ok}；"
                         "需等待完成，或显式 --allow-partial（部分结果不能视为最终结论）")
    evidence = carrier_evidence(audit)
    wrapper = xz_wrapper_attribution(columns, evidence)
    datasets = {dataset: paired_totals([col for col in columns if col["dataset"] == dataset])
                for dataset in sorted({col["dataset"] for col in columns})}
    return {
        "status": ("COMPLETE" if len(plan) == 20 else "SMOKE_COMPLETE") if ready else "PARTIAL", "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "parent": str(parent), "seed": seed, "methods": list(METHODS),
        "expected_rows": expected_rows, "verified_rows": len(found), "missing": missing,
        "runner_complete": complete, "manifest": manifest, "config": config,
        "manifest_sha256": sha256(manifest_path), "registry_sha256": sha256(registry_path),
        "parent_manifest_sha256": sha256(parent_manifest_path), "reference_carrier_audit": evidence,
        "gain_attribution": {"xz_container_omission": wrapper},
        "integrity_scope": "Read-only frame size/SHA256 and exact-byte ratio validation; lossless relies on recorded upstream cold replay PASS. No codecs run by reporter.",
        "comparison_scope": f"{len(plan)} columns, one seed, two methods; no best-of-seeds, stability or held-out generalization claim.",
        "depth_scope": "Expanded canonical206 registered IDs <=10, not naked primitives; v3 already reached depth10 successfully.",
        "columns": columns, "datasets": datasets, "overall": paired_totals(columns),
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return html.escape(str(value)).replace("\n", "<br>")
    return "<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{cell(v)}</th>" for v in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def render(summary: dict[str, Any]) -> tuple[str, str]:
    partial = summary["status"] != "COMPLETE"
    title = "深度 10 组合搜索结果" + ("（部分结果，不是最终结论）" if partial else "（完整单 seed 探测）")
    text = [f"# {title}", "", f"状态：{summary['status']}；上游无损重放及本地文件审计：{summary['verified_rows']}/{summary['expected_rows']} 行；seed={summary['seed']}。", "",
            "深度是展开后的 canonical206 注册 ID 数，最多 10；不是 10 个裸压缩原语，也不是 10 次 recipe 决策。v3 已有成功的 10 层候选，本轮是在 10 层以内拓宽组合，不是首次到达 10 层。", "",
            "## 先分清三种收益", "",
            "已知最佳 = 固定算法、原 Beam29/PPO/GRPO、v3 GRPO/frozen 的已有最小文件，不是本轮新发现。公共探测先进行一次，两种方法从相同先验和公共缓存出发；GRPO 与不学习对照的差额，才是本轮学习引导相对冻结策略的探索性证据。", "",
            "本轮只有 1 个 seed × 20 列 × 2 方法，不能说明跨 seed 稳定性、held-out 泛化或每列严格必胜。等于先验/公共最佳是保底；零次权重更新的胜者来自初始策略采样，不是已完成的学习。发生过更新也不能把所有新收益自动归因于学习。", "",
            f"公共预算设置 {summary['config']['probe_seconds']:g} 秒 / {summary['config']['probe_edges']} 个新注册 ID 步骤；各线上设置 {summary['config']['max_seconds']:g} 秒 / {summary['config']['max_edges']} 个新注册 ID 步骤，另有迭代上限。步骤按注册 ID 计数，不能理解为裸压缩原语。时间是在注册 ID 步骤开始前检查的软截止，末次解码/更新可能略超时：两方法是相同上限设置，不是严格相同墙钟或相同实际评估数；实测时间如下。两方法共享公共阶段；没有单 Bzip2 自动补全。搜索调用超时 {summary['config']['search_timeout']:g} 秒会删失慢候选。新旧实验预算不同，不能直接据压缩比认定新算法更高效。", "",
            "## Reference 算子不是裸变换", "",
            "ZigZag-ref*[11]、Delta-ref*[7]、Identity-ref*[1] 保留原 reference compound API：变换后包含自适应整数宽度 compact 打包、元数据和 zlib level 6；ID1 虽注册名为 scaling，对整数实际是 Identity，不是浮点缩放。开头连续两个 reference ID 可能合并进一次 compact/zlib6 封装，不能机械理解为各自一个独立后端。表内 * 提醒这一语义。本轮没有将旧实现重写为 10 层裸原语，注册 ID 的执行语义保持不变。", "",
            "因此不能称“裸 ZigZag 带来 44.9 倍压缩”。已有 NMHC 消融中，Identity+compact+zlib6 完整文件 1,674 B，ZigZag+compact+zlib6 为 1,667 B，仅差 7 B；差额还含元数据/名字长度等影响，44.9×属于完整链。", "",
            f"<p>来源：<a href='{html.escape(Path(summary['reference_carrier_audit']['audit']).as_uri(), quote=True)}'>AUDIT.json</a>；<a href='{html.escape(Path(summary['reference_carrier_audit']['report']).as_uri(), quote=True)}'>compact/zlib6 解释</a>。</p>", "",
            "## Bzip2 分解链也不等于任意内部原语重组", "",
            "现有公开前缀 F = 项目 BWT → MTF → RLE → Huffman；末步 residual[181] 实际为 Bzip2_level4(F⁻¹(prefix))。所以完整链的载荷满足 Bzip2(F⁻¹(F(x))) = Bzip2(x)：末步先撤销公开前缀，再调用完整 Bzip2。完整文件仍保留外层 UAR1 元数据，并不等同裸 native Bzip2 字节；执行成本还包含 F 和逆 F。", "",
            "合法的外部前置变换 T 仍可保留效果，形成 Bzip2(T(x))；但若 T 是 reference standalone API，其 compact/zlib6 载荷及 UAR1 包装整体作为字节流送给 BWT，不是裸 ZigZag 数组直接接 Bzip2。不能把当前注册分解链理解为任意内部原语拼接、且全部中间效果都被 native Bzip2 沿用。本轮没有改变该语义，结果不能用来否定真正物理原语重组的潜力。", "",
            f"<p>来源：<a href='{html.escape(Path(summary['reference_carrier_audit']['bzip2_residual_report'] or summary['reference_carrier_audit']['bzip2_residual_source']).as_uri(), quote=True)}'>Bzip2 residual 语义与源码证据</a>。</p>", ""]
    wrapper = summary["gain_attribution"]["xz_container_omission"]
    if wrapper["status"] == "VERIFIED_EXISTING_BYTES":
        text += ["## RH：少存封装不等于 XZ 压缩流变小", "",
                 f"不学习搜索省略末尾 container[135]，将完整文件从 {wrapper['grpo_depth10']['frame_bytes']:,} B 减至 {wrapper['frozen_depth10']['frame_bytes']:,} B；内含 {wrapper['native_xz_bytes']:,} B native XZ 流逐字节完全相同。净省 {wrapper['net_frame_bytes_saved']} B = 少 {wrapper['uar1_json_bytes_saved']} B UAR1 JSON − 多 {wrapper['xzc1_packet_extra_bytes']} B XZC1 包装。这是真实完整文件收益，但属于封装层优化，不是 XZ/LZMA2 压缩能力或 GRPO 学习提升。", "",
                 f"RH 的 GRPO 相对共享阶段新增 {wrapper['grpo_online_new_bytes_vs_shared']} B，不学习搜索新增 {wrapper['frozen_online_new_bytes_vs_shared']} B。公共阶段相对历史最佳节省的 {wrapper['shared_new_bytes_vs_prior']} B 尚未在这份审计中拆解，不能把不学习搜索总共节省的 {wrapper['frozen_new_bytes_vs_prior']} B 全归为封装优化。下表保留实际完整文件字节与 CR，不扣除这项收益。", "",
                 f"<p>来源：<a href='{html.escape(Path(wrapper['audit']).as_uri(), quote=True)}'>XZ_FRAME_OMISSION_AUDIT.md（现有文件逐字节核验）</a>。</p>", ""]
    text += ["## 数据集整体（只在两方法均完成的同一列集上汇总）", ""]
    totals = []
    for dataset, data in summary["datasets"].items():
        totals.append([dataset, f"{data['paired_columns']}/{data['expected_columns']}", fmt(data["ratios"]["fixed_bytes"]),
                       fmt(data["ratios"]["prior_best_bytes"]), fmt(data["ratios"]["shared_best_bytes"]),
                       fmt(data["methods"][METHODS[0]]["compression_ratio"]), fmt(data["methods"][METHODS[1]]["compression_ratio"]),
                       data["shared_new_bytes_vs_prior"], data["methods"][METHODS[0]]["online_new_bytes_vs_shared"],
                       data["methods"][METHODS[1]]["online_new_bytes_vs_shared"], data["grpo_vs_frozen"]["bytes_saved"],
                       "/".join(str(data["grpo_vs_frozen"][v]) for v in ("wins", "ties", "losses"))])
    text += [table(["数据集", "同列覆盖", "固定 CR", "已知 CR", "公共后 CR", "GRPO CR", "不学习 CR", "公共新增 B", "GRPO 线上新增 B", "不学习线上新增 B", "GRPO较不学习省 B", "胜/平/负"], totals), "",
             "CR = Σ原始字节 / Σ完整文件字节，不是逐列 CR 的平均。正的字节收益表示更小；GRPO较不学习为负表示 GRPO 更差。所有胜负依据整数完整文件大小。", ""]
    for dataset in summary["datasets"]:
        cols = [col for col in summary["columns"] if col["dataset"] == dataset]
        text += [f"## {dataset} 逐列结果", ""]
        results, costs, depths = [], [], []
        for col in cols:
            grpo, frozen = (col[method] for method in METHODS)
            def result_cell(row: dict[str, Any] | None) -> str:
                return "未完成" if row is None else f"{row['chain_display']}\n{row['compressed_bytes']:,} B / {row['compression_ratio']:.6f}×"
            results.append([col["column"], f"{col['fixed_algorithm']}\n{col['fixed_bytes']:,} B / {col['fixed_ratio']:.6f}×",
                            f"{col['prior_best_bytes']:,} B / {col['prior_best_ratio']:.6f}×" if col["prior_best_bytes"] else "未完成",
                            result_cell(grpo), result_cell(frozen),
                            col["shared"]["new_bytes_vs_prior"] if col["shared"] else "—",
                            grpo["new_bytes_vs_prior"] if grpo else "—", frozen["new_bytes_vs_prior"] if frozen else "—",
                            col["grpo_vs_frozen_bytes_saved"] if grpo and frozen else "—"])
            for row in (grpo, frozen):
                if row:
                    costs.append([col["column"], LABELS[row["method"]], row["seed"], row["selected_depth"],
                                  fmt(row["compress_sec"], 6), fmt(row["compose_sec"], 6),
                                  fmt(row["search_train_sec"], 6), row["online_unique_atomic_edges"], row["online_new_bytes_vs_shared"],
                                  row["completed_update_iterations"], str(row["weights_changed"]), str(row["selected_provenance"])])
            phases = [("公共（一次）", col["shared"]["depth"] if col["shared"] else None)]
            phases += [(LABELS[method] + "线上", col[method]["depth"] if col[method] else None) for method in METHODS]
            for name, depth in phases:
                depths.append([col["column"], name] + ([depth["phase_9_10"][str(d)][f] for d in (9, 10) for f in ("attempt", "ok", "unique")] if depth else ["—"] * 6))
        text += [table(["列", "固定最佳 / CR", "已知最佳 / CR", "GRPO 链 / CR", "不学习链 / CR", "公共新增 B", "GRPO较已知新增 B", "不学习较已知新增 B", "GRPO较不学习省 B"], results), "",
                 "### 编码与组合时间、更新归因", "",
                 table(["列", "方法", "seed", "选中ID深度", "编码 s", "组合 s", "线上搜索/训练 s", "线上新增注册ID步骤记录", "线上较公共新增 B", "完成更新次数", "权重改变", "选中来源"], costs), "",
                 "编码时间来自 row 的完整链独立重放，原始重复测量值保存在 JSON。组合时间直接使用 row.compose_sec；公共阶段只实际执行一次，不可把两方法组合时间相加当总墙钟。历史先验的发现成本不在本轮计时里。", "",
                 "### 深度 9 / 10 的完整候选尝试与成功", "",
                 table(["列", "阶段", "9尝试", "9成功", "9唯一尝试", "10尝试", "10成功", "10唯一尝试"], depths), "",
                 "线上仅计 provenance=search；公共阶段单列一次，不将共享历史重复算为线上尝试。尝试/成功是完整候选记录次数（可能含缓存命中），不是新注册 ID 步骤数；唯一尝试按阶段去重。成功表示完整文件验证成功，不代表压缩更好；仅到达某个注册 ID 前缀不等于完整 10 层候选成功。", ""]
    total = summary["overall"]
    cache_states = Counter(col["shared"]["cache_equality_audit"] for col in summary["columns"] if col["shared"])
    text += ["## 审计与限制", "",
             f"GRPO 完成权重更新 {total['methods'][METHODS[0]]['updated_columns']}/{total['paired_columns']} 列；零更新 {total['methods'][METHODS[0]]['zero_update_columns']} 列，其线上新增 {total['methods'][METHODS[0]]['zero_update_online_new_bytes']} B 只能称采样收益。", "",
             f"共享缓存核验：{json.dumps(dict(cache_states), ensure_ascii=False)}。若没有快照指纹，报告只核对先验/公共最佳字节相同，不声称已独立证明所有缓存内容相同。", "",
             "本脚本没有执行 codec；它检查获胜完整文件的大小、SHA256、CR 算术、算子 ID/名称、深度、共享阶段及上游 lossless=True。完整解码无损沿用 runner 的独立重放 PASS。没有穷举证明、全局最优保证或每列严格改善保证。", "",
             "完整链、逐次编码时间、原始 search_stats/search_cli、来源、每列先验清单和深度证据见 final_summary.json。共享记录中的先验并非本轮创新；若候选没有超出 prior_best，其相对固定算法的收益仍可能完全来自历史保底。", "",
             "公共先验验证+探测（按列各一次）及两方法线上时间合计：" + fmt(total["accounted_phase_sec"], 3) + " s；它不是包含重放/输出等开销的端到端墙钟。", "",
             "### 本轮 config", "", "```json", json.dumps(summary["config"], ensure_ascii=False, indent=2), "```", "",
             f"实验目录：`{summary['run']}`；manifest SHA256：`{summary['manifest_sha256']}`。", ""]
    if partial:
        text += ["缺失项：", "", *[f"- {value}" for value in summary["missing"]], ""]
    md = "\n".join(text)
    # Minimal deterministic Markdown rendering; tables are already escaped HTML.
    rendered, code = [], False
    for line in text:
        if line == "```json":
            rendered.append("<pre>")
            code = True
        elif line == "```":
            rendered.append("</pre>")
            code = False
        elif code:
            rendered.append(html.escape(line))
        elif line.startswith(("<div class=", "<p>来源：")):
            rendered.append(line)
        elif line.startswith("### "):
            rendered.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            rendered.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            rendered.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line:
            rendered.append(f"<p>{html.escape(line)}</p>")
    css = "body{font:15px/1.65 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#183248;margin:32px;max-width:1800px}h1{font-size:30px}h2{margin-top:40px}.table-wrap{overflow-x:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ced9e2;padding:9px;vertical-align:top;min-width:75px;overflow-wrap:anywhere}th{background:#eaf0f6}tbody tr:nth-child(even){background:#f7f9fb}pre{white-space:pre-wrap;background:#eef2f6;padding:16px}"
    document = f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style><body>" + "\n".join(rendered) + "</body></html>\n"
    return md, document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--parent", type=Path, help="原固定/Beam29/PPO/GRPO 实验目录；默认从 manifest 读取")
    parser.add_argument("--carrier-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, help="默认 RUN/analysis_final，不覆盖 runner 的报告")
    parser.add_argument("--allow-partial", action="store_true", help="允许显式带 PARTIAL 标记的非最终报告；不忽略损坏数据")
    parser.add_argument("--check-only", action="store_true", help="仅审计并输出状态，绝不写文件")
    args = parser.parse_args(argv)
    try:
        run = args.run.resolve()
        summary = collect(run, args.parent, args.carrier_audit.resolve(), args.allow_partial,
                          require_twenty=not args.check_only)
        if args.check_only:
            print(json.dumps({key: summary[key] for key in ("status", "run", "seed", "verified_rows", "expected_rows", "missing")}, ensure_ascii=False, indent=2))
            return 0
        output = (args.output or run / "analysis_final").resolve()
        if output == run or output == Path(summary["parent"]):
            raise ValueError("输出必须使用独立报告子目录，不能覆盖实验根目录")
        markdown, document = render(summary)
        output.mkdir(parents=True, exist_ok=True)
        stem = "FINAL_REPORT" if summary["status"] == "COMPLETE" else "PARTIAL_REPORT"
        summary_name = "final_summary.json" if summary["status"] == "COMPLETE" else "partial_summary.json"
        (output / f"{stem}.md").write_text(markdown, encoding="utf-8")
        (output / f"{stem}.html").write_text(document, encoding="utf-8")
        (output / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": summary["status"], "verified_rows": summary["verified_rows"],
                          "output": str(output), "files": [f"{stem}.md", f"{stem}.html", summary_name]}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
