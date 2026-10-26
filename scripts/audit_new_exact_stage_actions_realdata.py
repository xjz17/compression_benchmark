#!/usr/bin/env python3
"""Real-data parity audit for newly promoted independent exact-stage actions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_86_composability_realdata as audit86  # noqa: E402
import benchmark_86_combinations as benchmark  # noqa: E402
import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_stepwise(
    raw: bytes,
    action_names: tuple[str, ...],
    logical_type: str,
    *,
    max_point: int | None = None,
) -> tuple[bytes, bool]:
    """Run every stage through the public one-action boundary in both directions."""

    if logical_type in {"int", "integer"}:
        source = unified.ActionData("int64", raw, len(raw) // 8)
    elif logical_type == "float":
        source = unified.ActionData("float64", raw, len(raw) // 8)
    elif logical_type == "bytes":
        source = unified.ActionData.bytes(raw)
    else:
        raise ValueError(f"unsupported logical type: {logical_type!r}")

    current = source
    for action_name in action_names:
        current = unified.run_action(
            action_name, current, max_point=max_point
        )
    _, public_wire = unified._unpack_result(current)
    for action_name in reversed(action_names):
        current = unified.run_action(
            action_name, current, reverse=True, max_point=max_point
        )
    return public_wire, current == source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    columns, manifest = audit86.load_external_columns(args.rows)
    rle_cli = ROOT / "cpp/build/rle_int64_action_cli"
    for executable in (
        rle_cli, primitive.RLE_STAGE_CLI,
        primitive.TS2DIFF_STAGE_CLI, primitive.SVB_STAGE_CLI,
        primitive.SIMPLE8B_STAGE_CLI, primitive.P04_STAGE_CLI,
        primitive.DBP_STAGE_CLI,
        primitive.GORILLA_STAGE_CLI,
        primitive.CHIMP128_STAGE_CLI,
        primitive.SPDP_STAGE_CLI,
        primitive.FPC_STAGE_CLI,
        primitive.PFPC_STAGE_CLI,
    ):
        if not executable.is_file():
            raise RuntimeError(f"missing executable: {executable}")

    rows: list[dict[str, Any]] = []
    for column in columns:
        original, reason = benchmark.specialist_artifact(column, "RLE")
        if original is not None:
            logical_type = "int" if column.logical_type == "integer" else "float"
            max_point = column.max_point if logical_type == "float" else None
            action_names = primitive.exact_atomic_action_pipeline(
                "RLE", logical_type, max_point=max_point
            )
            staged, roundtrip = run_stepwise(
                column.raw, action_names, logical_type, max_point=max_point
            )
            rows.append(
                {
                    "method": "RLE",
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": len(original.payload),
                    "staged_bytes": len(staged),
                    "ratio_equal": len(original.payload) == len(staged),
                    "payload_identical": original.payload == staged,
                    "roundtrip": roundtrip,
                    "scope": (
                        "stepwise_integer_rle"
                        if logical_type == "int"
                        else "stepwise_scaling_then_integer_rle"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "method": "RLE", "dataset": column.dataset,
                    "column": column.name, "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw), "reference_bytes": "",
                    "staged_bytes": "", "ratio_equal": "",
                    "payload_identical": "", "roundtrip": "",
                    "scope": f"not_applicable:{reason}",
                }
            )

        original, reason = benchmark.specialist_artifact(column, "2DIFF")
        if original is not None:
            logical_type = "int" if column.logical_type == "integer" else "float"
            action_names = (
                primitive.TS2DIFF_INT_ACTIONS
                if logical_type == "int"
                else primitive.TS2DIFF_FLOAT_ACTIONS
            )
            max_point = column.max_point if logical_type == "float" else None
            staged, roundtrip = run_stepwise(
                column.raw, action_names, logical_type, max_point=max_point
            )
            rows.append(
                {
                    "method": "2DIFF",
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": len(original.payload),
                    "staged_bytes": len(staged),
                    "ratio_equal": len(original.payload) == len(staged),
                    "payload_identical": original.payload == staged,
                    "roundtrip": roundtrip,
                    "scope": "stepwise_run_action_supported_domain",
                }
            )
        else:
            rows.append(
                {
                    "method": "2DIFF",
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": "",
                    "staged_bytes": "",
                    "ratio_equal": "",
                    "payload_identical": "",
                    "roundtrip": "",
                    "scope": f"not_applicable:{reason}",
                }
            )

        for method, action_names in (
            ("RLBE", primitive.RLBE_ACTIONS),
            ("Sprintz", primitive.SPRINTZ_ACTIONS),
        ):
            original, reason = benchmark.specialist_artifact(column, method)
            if original is not None:
                logical_type = "int" if column.logical_type == "integer" else "float"
                staged, roundtrip = run_stepwise(
                    column.raw, action_names, logical_type
                )
                rows.append(
                    {
                        "method": method,
                        "dataset": column.dataset,
                        "column": column.name,
                        "logical_type": column.logical_type,
                        "raw_bytes": len(column.raw),
                        "reference_bytes": len(original.payload),
                        "staged_bytes": len(staged),
                        "ratio_equal": len(original.payload) == len(staged),
                        "payload_identical": original.payload == staged,
                        "roundtrip": roundtrip,
                        "scope": "stepwise_run_action_p04_chain",
                    }
                )
            else:
                rows.append(
                    {
                        "method": method,
                        "dataset": column.dataset,
                        "column": column.name,
                        "logical_type": column.logical_type,
                        "raw_bytes": len(column.raw),
                        "reference_bytes": "",
                        "staged_bytes": "",
                        "ratio_equal": "",
                        "payload_identical": "",
                        "roundtrip": "",
                        "scope": f"not_applicable:{reason}",
                    }
                )

        for method, int_actions, float_actions, scope in (
            (
                "DBP", primitive.DBP_INT_ACTIONS, primitive.DBP_FLOAT_ACTIONS,
                "full_scaling_delta_zigzag_varint_chain",
            ),
            (
                "SVB", primitive.SVB_INT_ACTIONS, primitive.SVB_FLOAT_ACTIONS,
                "shared_scaling_then_delta_limbs_then_stream_vbyte",
            ),
            (
                "Gorilla", primitive.GORILLA_ACTIONS, primitive.GORILLA_ACTIONS,
                "previous-value XOR then Gorilla leading/trailing-zero control-bit pack",
            ),
            (
                "Chimp128", primitive.CHIMP128_ACTIONS, primitive.CHIMP128_ACTIONS,
                "history-128 reference XOR then Chimp four-case control-bit pack",
            ),
            (
                "Chimp", primitive.CHIMP128_ACTIONS, primitive.CHIMP128_ACTIONS,
                "same project TsFile Chimp wire path used by both public names",
            ),
        ):
            original, reason = benchmark.specialist_artifact(column, method)
            if original is not None:
                logical_type = "int" if column.logical_type == "integer" else "float"
                action_names = int_actions if logical_type == "int" else float_actions
                max_point = column.max_point if logical_type == "float" else None
                staged, roundtrip = run_stepwise(
                    column.raw, action_names, logical_type, max_point=max_point
                )
                rows.append(
                    {
                        "method": method,
                        "dataset": column.dataset,
                        "column": column.name,
                        "logical_type": column.logical_type,
                        "raw_bytes": len(column.raw),
                        "reference_bytes": len(original.payload),
                        "staged_bytes": len(staged),
                        "ratio_equal": len(original.payload) == len(staged),
                        "payload_identical": original.payload == staged,
                        "roundtrip": roundtrip,
                        "scope": f"stepwise_run_action:{scope}",
                    }
                )
            else:
                rows.append(
                    {
                        "method": method,
                        "dataset": column.dataset,
                        "column": column.name,
                        "logical_type": column.logical_type,
                        "raw_bytes": len(column.raw),
                        "reference_bytes": "",
                        "staged_bytes": "",
                        "ratio_equal": "",
                        "payload_identical": "",
                        "roundtrip": "",
                        "scope": f"not_applicable:{reason}",
                    }
                )

        original, reason = benchmark.specialist_artifact(column, "SPDP")
        if original is not None:
            staged, roundtrip = run_stepwise(
                column.raw, primitive.SPDP_ACTIONS, "bytes"
            )
            rows.append(
                {
                    "method": "SPDP",
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": len(original.payload),
                    "staged_bytes": len(staged),
                    "ratio_equal": len(staged) == len(original.payload),
                    "payload_identical": staged == original.payload,
                    "roundtrip": roundtrip,
                    "scope": "stepwise_run_action:second_order_delta_then_byte_transpose_delta_then_history_repeat",
                }
            )
        else:
            rows.append(
                {
                    "method": "SPDP", "dataset": column.dataset,
                    "column": column.name, "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw), "reference_bytes": "",
                    "staged_bytes": "", "ratio_equal": "",
                    "payload_identical": "", "roundtrip": "",
                    "scope": f"not_applicable:{reason}",
                }
            )

        original, reason = benchmark.specialist_artifact(column, "FPC")
        if original is not None:
            staged, roundtrip = run_stepwise(
                column.raw, primitive.FPC_ACTIONS, "bytes"
            )
            rows.append(
                {
                    "method": "FPC", "dataset": column.dataset,
                    "column": column.name, "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": len(original.payload),
                    "staged_bytes": len(staged),
                    "ratio_equal": len(staged) == len(original.payload),
                    "payload_identical": staged == original.payload,
                    "roundtrip": roundtrip,
                    "scope": "stepwise_run_action:fcm_dfcm_predictor_xor_then_nibble_byte_pack",
                }
            )
        else:
            rows.append(
                {
                    "method": "FPC", "dataset": column.dataset,
                    "column": column.name, "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw), "reference_bytes": "",
                    "staged_bytes": "", "ratio_equal": "",
                    "payload_identical": "", "roundtrip": "",
                    "scope": f"not_applicable:{reason}",
                }
            )

        original, reason = benchmark.specialist_artifact(column, "PFPC")
        if original is not None:
            staged, roundtrip = run_stepwise(
                column.raw, primitive.PFPC_ACTIONS, "bytes"
            )
            rows.append(
                {
                    "method": "PFPC", "dataset": column.dataset,
                    "column": column.name, "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": len(original.payload),
                    "staged_bytes": len(staged),
                    "ratio_equal": len(staged) == len(original.payload),
                    "payload_identical": staged == original.payload,
                    "roundtrip": roundtrip,
                    "scope": "stepwise_run_action:single_thread_block_predictor_xor_then_nibble_pack",
                }
            )
        else:
            rows.append(
                {
                    "method": "PFPC", "dataset": column.dataset,
                    "column": column.name, "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw), "reference_bytes": "",
                    "staged_bytes": "", "ratio_equal": "",
                    "payload_identical": "", "roundtrip": "",
                    "scope": f"not_applicable:{reason}",
                }
            )

        original, reason = benchmark.specialist_artifact(column, "Simple8b")
        if original is not None:
            logical_type = "int" if column.logical_type == "integer" else "float"
            action_names = (
                primitive.SIMPLE8B_INT_ACTIONS
                if logical_type == "int"
                else primitive.SIMPLE8B_FLOAT_ACTIONS
            )
            max_point = column.max_point if logical_type == "float" else None
            staged, roundtrip = run_stepwise(
                column.raw, action_names, logical_type, max_point=max_point
            )
            rows.append(
                {
                    "method": "Simple8b",
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": len(original.payload),
                    "staged_bytes": len(staged),
                    "ratio_equal": len(original.payload) == len(staged),
                    "payload_identical": original.payload == staged,
                    "roundtrip": roundtrip,
                    "scope": "stepwise_run_action_supported_domain",
                }
            )
        else:
            rows.append(
                {
                    "method": "Simple8b",
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "reference_bytes": "",
                    "staged_bytes": "",
                    "ratio_equal": "",
                    "payload_identical": "",
                    "roundtrip": "",
                    "scope": f"not_applicable:{reason}",
                }
            )
    applicable = [row for row in rows if row["roundtrip"] != ""]
    failures = [
        row for row in applicable
        if not (row["ratio_equal"] and row["payload_identical"] and row["roundtrip"])
    ]
    by_method = {}
    for method in ("Chimp", "Chimp128", "Gorilla", "2DIFF", "SVB", "Simple8b", "RLBE", "Sprintz", "DBP", "SPDP", "FPC", "PFPC", "RLE"):
        selected = [row for row in applicable if row["method"] == method]
        by_method[method] = {
            "applicable_columns": len(selected),
            "exact_pass_columns": sum(
                bool(row["ratio_equal"] and row["payload_identical"] and row["roundtrip"])
                for row in selected
            ),
        }
    payload = {
        "schema": "new-exact-stage-actions-realdata-v2",
        "rows_per_column": args.rows,
        "dataset_manifest": manifest,
        "comparisons": len(rows),
        "applicable_comparisons": len(applicable),
        "failures": len(failures),
        "status": "PASS" if not failures else "FAIL",
        "execution_path": (
            "every applicable comparison calls unified.run_action once per "
            "forward stage and once per inverse stage"
        ),
        "implementation_artifact_hashes": audit86._exact_stage_artifact_hashes(),
        "by_method": by_method,
        "boundaries": {
            "2DIFF": "full project codec parity on every materialized supported column",
            "SVB": "full typed adapter using the shared exact scaling action for float input",
            "RLE": (
                "the RLE action remains integer-only; float columns explicitly run "
                "the independent scaling action before integer RLE"
            ),
            "Simple8b": "full typed project adapter on every materialized supported column",
            "RLBE": "full delta then zero-run/Fibonacci bitstream project codec chain",
            "Sprintz": "full LongFIRE predictor then ZigZag then bitpack project codec chain",
            "DBP": "full optional scaling then delta, ZigZag, and LEB128 materializer chain",
            "Gorilla": "full previous-value XOR then leading/trailing-zero control-bit chain",
            "Chimp128": "full history-128 reference XOR then four-case control-bit chain",
            "Chimp": "same project TsFile Chimp wire path used by both public names",
            "SPDP": "full second-order delta, byte transpose/delta, and level-5 history-repeat chain",
            "FPC": "full FCM/dFCM predictor-XOR and nibble byte-pack chain",
            "PFPC": "full single-thread block-local FCM/dFCM and pFPC framing/pack chain",
        },
        "status_counts": dict(Counter("pass" if row not in failures else "fail" for row in applicable)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "new_stage_realdata.csv", rows)
    (args.output_dir / "new_stage_realdata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# 新增 exact action 的真实数据审计",
        "",
        f"- 状态：**{payload['status']}**；适用比较 {len(applicable)}，失败 {len(failures)}。",
        f"- 2DIFF：{by_method['2DIFF']['exact_pass_columns']}/{by_method['2DIFF']['applicable_columns']}，完整 stage chain 与原 project wire 相同。",
        f"- SVB：{by_method['SVB']['exact_pass_columns']}/{by_method['SVB']['applicable_columns']}，完整 typed adapter parity。",
        f"- RLE：{by_method['RLE']['exact_pass_columns']}/{by_method['RLE']['applicable_columns']}；integer 直接调用 RLE，float 逐步调用 scaling → integer RLE。",
        f"- Simple8b：{by_method['Simple8b']['exact_pass_columns']}/{by_method['Simple8b']['applicable_columns']}，完整 typed adapter parity。",
        f"- RLBE：{by_method['RLBE']['exact_pass_columns']}/{by_method['RLBE']['applicable_columns']}，完整 delta → run-length/Fibonacci bitstream parity。",
        f"- Sprintz：{by_method['Sprintz']['exact_pass_columns']}/{by_method['Sprintz']['applicable_columns']}，完整 LongFIRE predictor → ZigZag → bitpack parity。",
        f"- DBP：{by_method['DBP']['exact_pass_columns']}/{by_method['DBP']['applicable_columns']}，完整 scaling → delta → ZigZag → LEB128 parity。",
        f"- Gorilla：{by_method['Gorilla']['exact_pass_columns']}/{by_method['Gorilla']['applicable_columns']}，完整 XOR → leading/trailing-zero pack parity。",
        f"- Chimp128：{by_method['Chimp128']['exact_pass_columns']}/{by_method['Chimp128']['applicable_columns']}，完整 history-XOR → four-case pack parity。",
        f"- Chimp：{by_method['Chimp']['exact_pass_columns']}/{by_method['Chimp']['applicable_columns']}，Chimp 与 Chimp128 两个公开名称映射到同一 project TsFile Chimp wire path。",
        f"- SPDP：{by_method['SPDP']['exact_pass_columns']}/{by_method['SPDP']['applicable_columns']}，完整 second-order delta → byte transpose/delta → history-repeat parity。",
        f"- FPC：{by_method['FPC']['exact_pass_columns']}/{by_method['FPC']['applicable_columns']}，完整 FCM/dFCM predictor-XOR → nibble byte-pack parity。",
        f"- PFPC：{by_method['PFPC']['exact_pass_columns']}/{by_method['PFPC']['applicable_columns']}，完整 block-local predictor-XOR → pFPC framing/pack parity。",
        "",
    ]
    (args.output_dir / "NEW_STAGE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
