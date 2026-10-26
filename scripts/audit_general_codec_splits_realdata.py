#!/usr/bin/env python3
"""Audit requested general-codec C++ stage boundaries on external columns.

The status vocabulary deliberately separates a complete benchmark-equivalent
pipeline from a source-level implementation and from a container-only split.
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
import benchmark_86_combinations as benchmark  # noqa: E402
import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402


def _stepwise(raw: bytes, names: tuple[str, ...]) -> tuple[bytes, bool]:
    source = unified.ActionData.bytes(raw)
    current = source
    for name in names:
        current = unified.run_action(name, current)
    _header, wire = unified._unpack_result(current)
    for name in reversed(names):
        current = unified.run_action(name, current, reverse=True)
    return wire, current == source


def _row(
    method: str,
    column: Any,
    scope: str,
    roundtrip: bool,
    candidate: bytes,
    reference: bytes,
    status: str,
    actions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "method": method,
        "dataset": column.dataset,
        "column": column.name,
        "scope": scope,
        "actions": "|".join(actions),
        "roundtrip": roundtrip,
        "candidate_bytes": len(candidate),
        "benchmark_bytes": len(reference),
        "compression_size_identical": len(candidate) == len(reference),
        "payload_identical": candidate == reference,
        "status": status,
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
        raw = column.raw

        lz4_reference, _decode, _impl = benchmark._compress_generic(raw, "lz4-framed")
        lz4_wire, lz4_roundtrip = _stepwise(raw, primitive.LZ4_ACTIONS)
        rows.append(
            _row(
                "LZ4", column, "wire_replay_non_atomic_match_extractor", lz4_roundtrip,
                lz4_wire, lz4_reference,
                (
                    "WIRE_REPLAY_EXACT_NON_ATOMIC_MATCH_EXTRACTOR"
                    if lz4_roundtrip and lz4_wire == lz4_reference
                    else "FAIL"
                ),
                primitive.LZ4_ACTIONS,
            )
        )

        snappy_reference, _decode, _impl = benchmark._compress_generic(raw, "snappy-framed")
        snappy_wire, snappy_roundtrip = _stepwise(raw, primitive.SNAPPY_ACTIONS)
        snappy_status = (
            "SOURCE_WIRE_REPLAY_EXACT_NON_ATOMIC_MATCH_EXTRACTOR"
            if snappy_roundtrip and snappy_wire == snappy_reference
            else "SOURCE_WIRE_REPLAY_BENCHMARK_MISMATCH_NON_ATOMIC_MATCH_EXTRACTOR"
            if snappy_roundtrip
            else "FAIL"
        )
        rows.append(
            _row(
                "Snappy", column, "google_snappy_wire_replay_non_atomic_match_extractor", snappy_roundtrip,
                snappy_wire, snappy_reference, snappy_status, primitive.SNAPPY_ACTIONS,
            )
        )

        zstd_reference, _decode, _impl = benchmark._compress_generic(raw, "zstd_level_9")
        zstd_packet = primitive._run_exact_stage_cli(
            ROOT / "cpp/build/zstd_exact_stages_cli", "Zstd", "sequence-forward", raw
        )
        zstd_raw = primitive._run_exact_stage_cli(
            ROOT / "cpp/build/zstd_exact_stages_cli", "Zstd", "sequence-inverse", zstd_packet
        )
        zstd_wire = primitive._run_exact_stage_cli(
            ROOT / "cpp/build/zstd_exact_stages_cli", "Zstd", "compose", raw
        )
        zstd_status = (
            "COMBINED_SEQUENCE_REPLAY_BENCHMARK_EXACT"
            if zstd_raw == raw and zstd_wire == zstd_reference
            else "COMBINED_SEQUENCE_REPLAY_BENCHMARK_MISMATCH"
            if zstd_raw == raw
            else "FAIL"
        )
        rows.append(
            _row(
                "Zstd", column, "combined_match_and_parse_sequence_api",
                zstd_raw == raw, zstd_wire, zstd_reference, zstd_status,
                ("experimental.zstd.sequence_forward", "experimental.zstd.entropy_frame"),
            )
        )

        for method, key, action, inverse_command in (
            ("gzip", "gz", primitive.GZIP_CONTAINER_ACTIONS[0], "gzip-inverse"),
            ("Zlib", "zlib_9", primitive.ZLIB_CONTAINER_ACTIONS[0], "zlib-inverse"),
        ):
            reference, _decode, _impl = benchmark._compress_generic(raw, key)
            packet = primitive._run_deflate_container_stage_cli(inverse_command, reference)
            rebuilt, roundtrip = _stepwise(packet, (action,))
            rows.append(
                _row(
                    method, column, "container_only", roundtrip, rebuilt, reference,
                    "CONTAINER_EXACT_CORE_UNSPLIT"
                    if roundtrip and rebuilt == reference
                    else "FAIL",
                    (action,),
                )
            )

        xz_reference, _decode, _impl = benchmark._compress_generic(raw, "xz_level_5")
        xz_packet = primitive._run_xz_container_stage_cli("inverse", xz_reference)
        xz_rebuilt, xz_roundtrip = _stepwise(xz_packet, primitive.XZ_CONTAINER_ACTIONS)
        rows.append(
            _row(
                "XZ", column, "container_only", xz_roundtrip, xz_rebuilt,
                xz_reference,
                "CONTAINER_EXACT_CORE_UNSPLIT"
                if xz_roundtrip and xz_rebuilt == xz_reference
                else "FAIL",
                primitive.XZ_CONTAINER_ACTIONS,
            )
        )

    csv_path = args.output_dir / "general_split_realdata_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, dict[str, int]] = defaultdict(dict)
    for method in sorted({str(row["method"]) for row in rows}):
        by_method[method] = dict(
            Counter(str(row["status"]) for row in rows if row["method"] == method)
        )
    summary = {
        "schema": "requested-general-codec-stage-audit-v1",
        "external_columns": len(columns),
        "real_world_columns": len(columns),
        "synthetic_columns": 0,
        "rows_per_column_limit": args.rows,
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "by_method": by_method,
        "benchmark_complete_methods": [],
        "wire_replay_non_atomic_match_extractors": ["LZ4", "Snappy"],
        "combined_non_atomic_experiments": ["Zstd"],
        "container_only_methods": ["gzip", "Zlib", "XZ"],
        "manifest": manifest,
        "row_file": csv_path.name,
    }
    (args.output_dir / "general_split_realdata_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not any(row["status"] in {"FAIL", "ERROR"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
