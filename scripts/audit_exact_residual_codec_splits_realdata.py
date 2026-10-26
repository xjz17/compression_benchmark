#!/usr/bin/env python3
"""Verify split action chains against vendored C/C++ codec encoders on real columns."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_86_composability_realdata as external  # noqa: E402
import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402


def _cli(path: Path, command: str, data: bytes) -> bytes:
    completed = subprocess.run(
        [str(path), command], input=data, capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{path.name} {command} failed: {detail}")
    return completed.stdout


def _stepwise(
    raw: bytes, names: tuple[str, ...], kind: unified.DataKind = "bytes"
) -> tuple[bytes, bool]:
    source = (
        unified.ActionData.bytes(raw)
        if kind == "bytes"
        else unified.ActionData(kind, raw, len(raw) // 8)
    )
    current = source
    for name in names:
        current = unified.run_action(name, current)
    _header, wire = unified._unpack_result(current)
    for name in reversed(names):
        current = unified.run_action(name, current, reverse=True)
    return wire, current == source


def _cases() -> list[tuple[str, tuple[str, ...], Path, str]]:
    residual_cli = ROOT / "cpp/build/general_codec_residual_stages_cli"
    lz4_cli = ROOT / "cpp/build/lz4_exact_stages_cli"
    snappy_cli = ROOT / "cpp/build/snappy_exact_stages_cli"
    cases = [
        ("Brotli", primitive.BROTLI_EXACT_ACTIONS, residual_cli, "original-brotli-encode"),
        ("Zstd", primitive.ZSTD_EXACT_ACTIONS, residual_cli, "original-zstd-encode"),
        ("Bzip2", primitive.BZIP2_EXACT_ACTIONS, residual_cli, "original-bzip2-encode"),
        ("gzip", primitive.GZIP_EXACT_ACTIONS, residual_cli, "original-gzip-encode"),
        ("Zlib", primitive.ZLIB_EXACT_ACTIONS, residual_cli, "original-zlib-encode"),
        ("XZ", primitive.XZ_EXACT_ACTIONS, residual_cli, "original-xz-encode"),
        ("LZ4", primitive.LZ4_ACTIONS, lz4_cli, "original-encode"),
        ("Snappy", primitive.SNAPPY_ACTIONS, snappy_cli, "original-encode"),
    ]
    for method, pipelines, prefix in (
        ("Blosc", primitive.BLOSC_EXACT_ACTIONS_BY_KIND, "blosc1"),
        ("Blosc2", primitive.BLOSC2_EXACT_ACTIONS_BY_KIND, "blosc2"),
    ):
        for kind, type_size in (
            ("bytes", 1),
            ("float32", 4),
            ("float64", 8),
            ("int64", 8),
        ):
            cases.append(
                (
                    f"{method}[{kind}]",
                    pipelines[kind],
                    residual_cli,
                    f"{prefix}-original-{type_size}",
                )
            )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns, manifest = external.load_external_columns(args.rows)
    results: list[dict[str, Any]] = []

    for column in columns:
        for method, actions, cli, command in _cases():
            try:
                wire, roundtrip = _stepwise(column.raw, actions)
                original = _cli(cli, command, column.raw)
                payload_equal = wire == original
                size_equal = len(wire) == len(original)
                status = (
                    "EXACT_BYTE_AND_RATIO_PARITY"
                    if roundtrip and payload_equal
                    else "RATIO_ONLY_PARITY"
                    if roundtrip and size_equal
                    else "FAIL"
                )
                error = ""
            except Exception as exc:
                wire = b""
                original = b""
                roundtrip = False
                payload_equal = False
                size_equal = False
                status = "ERROR"
                error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "method": method,
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "raw_bytes": len(column.raw),
                    "action_count": len(actions),
                    "actions": "|".join(actions),
                    "roundtrip": roundtrip,
                    "split_bytes": len(wire),
                    "original_cpp_bytes": len(original),
                    "compression_size_identical": size_equal,
                    "payload_identical": payload_equal,
                    "status": status,
                    "error": error,
                }
            )

        specialist = (
            (
                "ALP",
                primitive.ALP_MAIN_ACTIONS,
                "float64",
                "alp-original-encode",
            )
            if column.logical_type == "float"
            else (
                "REGER",
                primitive.REGER_ACTIONS,
                "int64",
                "reger-original-encode",
            )
        )
        method, actions, kind, command = specialist
        try:
            wire, roundtrip = _stepwise(column.raw, actions, kind)
            original = _cli(
                ROOT / "cpp/build/learning_codec_exact_stages_cli",
                command,
                column.raw,
            )
            payload_equal = wire == original
            size_equal = len(wire) == len(original)
            status = (
                "EXACT_BYTE_AND_RATIO_PARITY"
                if roundtrip and payload_equal
                else "RATIO_ONLY_PARITY"
                if roundtrip and size_equal
                else "FAIL"
            )
            error = ""
        except Exception as exc:
            wire = b""
            original = b""
            roundtrip = False
            payload_equal = False
            size_equal = False
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "method": method,
                "dataset": column.dataset,
                "column": column.name,
                "logical_type": column.logical_type,
                "raw_bytes": len(column.raw),
                "action_count": len(actions),
                "actions": "|".join(actions),
                "roundtrip": roundtrip,
                "split_bytes": len(wire),
                "original_cpp_bytes": len(original),
                "compression_size_identical": size_equal,
                "payload_identical": payload_equal,
                "status": status,
                "error": error,
            }
        )

    row_file = args.output_dir / "exact_residual_realdata_rows.csv"
    with row_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    by_method: dict[str, dict[str, int]] = defaultdict(dict)
    for method in sorted({str(row["method"]) for row in results}):
        by_method[method] = dict(
            Counter(str(row["status"]) for row in results if row["method"] == method)
        )
    summary = {
        "schema": "exact-residual-codec-split-realdata-v1",
        "contract": "split action payload must equal vendored C/C++ encoder payload byte-for-byte",
        "external_columns": len(columns),
        "real_world_columns": len(columns),
        "synthetic_columns": 0,
        "rows_per_column_limit": args.rows,
        "codec_variants": len(_cases()),
        "typed_specialist_variants": 2,
        "checks": len(results),
        "status_counts": dict(Counter(str(row["status"]) for row in results)),
        "by_method": by_method,
        "manifest": manifest,
        "row_file": row_file.name,
    }
    (args.output_dir / "exact_residual_realdata_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "EXACT_BYTE_AND_RATIO_PARITY" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
