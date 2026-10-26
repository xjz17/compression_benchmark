#!/usr/bin/env python3
"""Audit the BOS/FastPFor/Elf/Camel exact stage splits on external columns.

The report distinguishes source-level C++ parity from benchmark-86 parity.
The C++ FastPFor path is split at CompositeCodec's real boundary into a
FastPFor<8> prefix and VariableByte tail.  The current Python benchmark adapter
with the same display name still uses Zstd on int64 limbs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_86_composability_realdata as external  # noqa: E402
import benchmark_86_combinations as benchmark  # noqa: E402
import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402


def _source(column: Any) -> unified.ActionData:
    kind = "int64" if column.logical_type == "integer" else "float64"
    return unified.ActionData(kind, column.raw, len(column.raw) // 8)


def _run_stepwise(
    source: unified.ActionData,
    names: tuple[str, ...],
    max_point: int | None = None,
) -> tuple[bytes, bool]:
    current = source
    for name in names:
        current = unified.run_action(name, current, max_point=max_point)
    _header, public_wire = unified._unpack_result(current)
    for name in reversed(names):
        current = unified.run_action(name, current, reverse=True, max_point=max_point)
    return public_wire, current == source


def _row(
    method: str,
    column: Any,
    names: tuple[str, ...],
    *,
    reference: bytes | None,
    reference_kind: str,
    max_point: int | None = None,
    success_status: str = "PASS",
) -> dict[str, Any]:
    base = {
        "method": method,
        "dataset": column.dataset,
        "column": column.name,
        "logical_type": column.logical_type,
        "actions": "|".join(names),
        "reference_kind": reference_kind,
    }
    try:
        wire, roundtrip = _run_stepwise(_source(column), names, max_point)
        base.update(
            {
                "status": success_status if roundtrip and (reference is None or wire == reference) else "FAIL",
                "roundtrip": roundtrip,
                "wire_bytes": len(wire),
                "reference_bytes": "" if reference is None else len(reference),
                "payload_identical": "" if reference is None else wire == reference,
                "reason": "",
            }
        )
    except Exception as exc:
        base.update(
            {
                "status": "ERROR",
                "roundtrip": "",
                "wire_bytes": "",
                "reference_bytes": "" if reference is None else len(reference),
                "payload_identical": "",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return base


def _reference_failure_status(reason: str) -> str:
    """Map only known semantic limitations to N/A; surface all other errors."""

    if "roundtrip mismatch" in reason:
        return "REFERENCE_NOT_LOSSLESS"
    if reason in {
        "not applicable or exact decimal scaling unavailable",
        "Elf project path is float32-only",
        "FPZIP accepts floating-point arrays",
    }:
        return "NOT_APPLICABLE_REFERENCE_DOMAIN"
    return "ERROR"


def _unavailable_row(
    method: str,
    column: Any,
    names: tuple[str, ...],
    *,
    reason: str,
    status: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "dataset": column.dataset,
        "column": column.name,
        "logical_type": column.logical_type,
        "actions": "|".join(names),
        "reference_kind": "benchmark specialist_artifact",
        "status": status,
        "roundtrip": "",
        "wire_bytes": "",
        "reference_bytes": "",
        "payload_identical": "",
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns, manifest = external.load_external_columns(args.rows)
    rows: list[dict[str, Any]] = []

    for column in columns:
        if column.logical_type == "integer":
            rows.append(
                _row(
                    "BOS-source-BOSA", column, primitive.BOS_BOSA_ACTIONS,
                    reference=primitive._run_bos_stage_cli("compose", column.raw),
                    reference_kind="independent BOSA C++ compose",
                    success_status="SOURCE_CHAIN_SELF_CONSISTENT",
                )
            )

        has_nonfinite = column.logical_type == "float" and any(
            not math.isfinite(float(value)) for value in column.values
        )
        if has_nonfinite:
            original, reason = None, "BOS decimal scaling requires finite float64 values"
        else:
            original, reason = benchmark.specialist_artifact(column, "BOS")
        if original is not None:
            names = (
                primitive.BOS64_INT_ACTIONS
                if column.logical_type == "integer"
                else primitive.BOS64_FLOAT_ACTIONS
            )
            rows.append(
                _row(
                    "BOS", column, names, reference=original.payload,
                    reference_kind="benchmark specialist_artifact",
                    max_point=None if column.logical_type == "integer" else column.max_point,
                    success_status="BENCHMARK_EXACT",
                )
            )
        else:
            rows.append(
                _unavailable_row(
                    "BOS", column, (), reason=reason,
                    status=(
                        "NOT_APPLICABLE_REFERENCE_DOMAIN"
                        if has_nonfinite
                        else _reference_failure_status(reason)
                    ),
                )
            )

        # Compare the two independent stages with the retained whole-composite
        # C++ helper. The benchmark specialist artifact is a different Zstd-limb
        # implementation and is deliberately not used as the source reference.
        fp_names = (
            primitive.FASTPFOR_INT_ACTIONS
            if column.logical_type == "integer"
            else primitive.FASTPFOR_FLOAT_ACTIONS
        )
        if column.logical_type == "float" and any(
            not math.isfinite(float(value)) for value in column.values
        ):
            row = _unavailable_row(
                "FastPFor-source-C++", column, fp_names,
                reason="explicit scaling contract requires finite float64 values",
                status="NOT_APPLICABLE_ACTION_DOMAIN",
            )
            row["reference_kind"] = "Lemire FastPFor+VariableByte source composite"
            rows.append(row)
        else:
            if column.logical_type == "integer":
                reference = primitive._run_fastpfor_stage_cli(
                    "adapter-forward-int", column.raw
                )
            else:
                scaling = primitive._run_dbp_stage_cli(
                    "scale-forward", column.raw, column.max_point
                )
                reference = primitive._run_fastpfor_stage_cli(
                    "adapter-forward-scaled", scaling
                )
            rows.append(
                _row(
                    "FastPFor-source-C++", column, fp_names, reference=reference,
                    reference_kind="Lemire FastPFor+VariableByte source composite",
                    max_point=None if column.logical_type == "integer" else column.max_point,
                    success_status="SOURCE_CHAIN_EXACT",
                )
            )

        if column.logical_type == "float":
            for method, names in (
                ("Elf", primitive.ELF_ACTIONS),
                ("Elf+", primitive.ELF_PLUS_ACTIONS),
                ("Camel", primitive.CAMEL_ACTIONS),
            ):
                if has_nonfinite:
                    original, reason = (
                        None,
                        f"{method} action contract requires finite float64 values",
                    )
                else:
                    original, reason = benchmark.specialist_artifact(column, method)
                if original is None:
                    rows.append(
                        _unavailable_row(
                            method, column, names, reason=reason,
                            status=(
                                "NOT_APPLICABLE_REFERENCE_DOMAIN"
                                if has_nonfinite
                                else _reference_failure_status(reason)
                            ),
                        )
                    )
                else:
                    rows.append(
                        _row(
                            method, column, names, reference=original.payload,
                            reference_kind="benchmark specialist_artifact",
                            success_status="BENCHMARK_EXACT",
                        )
                    )

            original, reason = benchmark.specialist_artifact(column, "FPZIP")
            if original is None:
                rows.append(
                    _unavailable_row(
                        "FPZIP", column, primitive.FPZIP_ACTIONS, reason=reason,
                        status=_reference_failure_status(reason),
                    )
                )
            else:
                rows.append(
                    _row(
                        "FPZIP", column, primitive.FPZIP_ACTIONS,
                        reference=original.payload,
                        reference_kind="benchmark specialist_artifact",
                        success_status="BENCHMARK_EXACT",
                    )
                )

    csv_path = args.output_dir / "requested_split_realdata_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(str(row["status"]) for row in rows)
    by_method: dict[str, dict[str, int]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        by_method[method] = dict(
            Counter(str(row["status"]) for row in rows if row["method"] == method)
        )
    summary = {
        "schema": "requested-codec-exact-split-audit-v1",
        "external_columns": len(columns),
        "real_world_columns": len(columns),
        "synthetic_columns": 0,
        "rows_per_column_limit": args.rows,
        "status_counts": dict(status_counts),
        "by_method": by_method,
        "strict_pass_definition": "stagewise roundtrip and exact payload equality when a benchmark/source reference exists",
        "fastpfor_boundary": "source C++ is independently split into FastPFor<8> prefix and VariableByte tail with exact composite-wire parity; the current benchmark FastPFor separately uses Zstd over low/high limbs",
        "manifest": manifest,
        "row_file": csv_path.name,
    }
    (args.output_dir / "requested_split_realdata_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status_counts.get("FAIL", 0) == 0 and status_counts.get("ERROR", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
