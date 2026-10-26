#!/usr/bin/env python3
"""Compare canonical C++ exact-action wires with the legacy Python benchmark.

This is deliberately separate from the source-parity audit.  A mismatch here
can mean that the old benchmark used another library/version rather than that
the split C++ action chain differs from its canonical C/C++ encoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_86_composability_realdata as external  # noqa: E402
import benchmark_86_combinations as legacy  # noqa: E402
import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402


def _run(raw: bytes, actions: tuple[str, ...], kind: str = "bytes") -> bytes:
    current = (
        unified.ActionData.bytes(raw)
        if kind == "bytes"
        else unified.ActionData(kind, raw, len(raw) // 8)
    )
    for action in actions:
        current = unified.run_action(action, current)
    return unified._unpack_result(current)[1]


def _general_cases() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return (
        ("Brotli", "brotli", primitive.BROTLI_EXACT_ACTIONS),
        ("Zstd", "zstd_level_9", primitive.ZSTD_EXACT_ACTIONS),
        ("Blosc", "blosc_blosclz", primitive.BLOSC_EXACT_ACTIONS_BY_KIND["bytes"]),
        ("Blosc2", "blosc2_blosclz", primitive.BLOSC2_EXACT_ACTIONS_BY_KIND["bytes"]),
        ("Bzip2", "bzip2_4", primitive.BZIP2_EXACT_ACTIONS),
        ("gzip", "gz", primitive.GZIP_EXACT_ACTIONS),
        ("Zlib", "zlib_9", primitive.ZLIB_EXACT_ACTIONS),
        ("XZ", "xz_level_5", primitive.XZ_EXACT_ACTIONS),
        ("LZ4", "lz4-framed", primitive.LZ4_ACTIONS),
        ("Snappy", "snappy-framed", primitive.SNAPPY_ACTIONS),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns, _manifest = external.load_external_columns(args.rows)
    rows: list[dict[str, Any]] = []

    for column in columns:
        for method, key, actions in _general_cases():
            try:
                old, _decode, implementation = legacy._compress_generic(column.raw, key)
                current = _run(column.raw, actions)
                size_delta = len(current) - len(old)
                status = (
                    "EXACT_BYTE_AND_RATIO_PARITY"
                    if current == old
                    else "RATIO_ONLY_PARITY"
                    if size_delta == 0
                    else "VERSION_OR_BACKEND_DIFFERENCE"
                )
                error = ""
            except Exception as exc:
                old = current = b""
                implementation = "unavailable"
                size_delta = None
                status = "LEGACY_RUNTIME_UNAVAILABLE"
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "method": method,
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "legacy_implementation": implementation,
                    "legacy_bytes": len(old) if old else None,
                    "cpp_exact_action_bytes": len(current) if current else None,
                    "size_delta_cpp_minus_legacy": size_delta,
                    "status": status,
                    "error": error,
                }
            )

        specialist = (
            ("ALP", primitive.ALP_MAIN_ACTIONS, "float64")
            if column.logical_type == "float"
            else ("REGER", primitive.REGER_ACTIONS, "int64")
        )
        method, actions, kind = specialist
        old_artifact, reason = legacy.specialist_artifact(column, method)
        if old_artifact is None:
            rows.append(
                {
                    "method": method,
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "legacy_implementation": "unavailable",
                    "legacy_bytes": None,
                    "cpp_exact_action_bytes": None,
                    "size_delta_cpp_minus_legacy": None,
                    "status": "LEGACY_RUNTIME_UNAVAILABLE",
                    "error": reason,
                }
            )
        else:
            current = _run(column.raw, actions, kind)
            delta = len(current) - len(old_artifact.payload)
            rows.append(
                {
                    "method": method,
                    "dataset": column.dataset,
                    "column": column.name,
                    "logical_type": column.logical_type,
                    "legacy_implementation": old_artifact.implementation,
                    "legacy_bytes": len(old_artifact.payload),
                    "cpp_exact_action_bytes": len(current),
                    "size_delta_cpp_minus_legacy": delta,
                    "status": (
                        "EXACT_BYTE_AND_RATIO_PARITY"
                        if current == old_artifact.payload
                        else "RATIO_ONLY_PARITY"
                        if delta == 0
                        else "VERSION_OR_BACKEND_DIFFERENCE"
                    ),
                    "error": "",
                }
            )

    row_path = args.output_dir / "legacy_benchmark86_parity_rows.csv"
    with row_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_method[row["method"]][row["status"]] += 1
    summary = {
        "schema": "exact-actions-vs-legacy-benchmark86-v1",
        "scope": "compatibility check only; canonical C/C++ source parity is audited separately",
        "checks": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "by_method": {key: dict(value) for key, value in sorted(by_method.items())},
        "row_file": row_path.name,
    }
    summary_path = args.output_dir / "legacy_benchmark86_parity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
