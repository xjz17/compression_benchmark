#!/usr/bin/env python3
"""Production Elf64 routing: no float32/native-LZ surrogate, no partial rows.

Production Java reports historical chunk-body bytes, not the benchmark-86
EL64 frame. Timing comes from JVM codec operations, never subprocess duration.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from myapp import elf_double_java as elf
from myapp import processor
from myapp import processor_patch_bench as patch_bench

NAMES = {"elf_on_chimp", "elf_plus"}


def fake_bench(columns):
    # Decimal precision and the sign of zero must survive production routing.
    assert struct.pack("<3d", *columns[0]) == struct.pack(
        "<3d", 1.2345678901234567, -0.0, 1.25,
    )
    return 24, 17, 11, 13


def main() -> None:
    columns = [[1.2345678901234567, float("nan"), -0.0, 1.25]]
    records = []
    record = lambda *row: records.append(row)
    selected = lambda name: name in NAMES
    with patch.object(elf, "_elf_jar", return_value="elf.jar"), \
         patch.object(elf, "_benchmark_jar", return_value="benchmark.jar"), \
         patch.object(elf, "bench_elf_on_chimp_double_columns", side_effect=fake_bench), \
         patch.object(elf, "bench_elf_plus_double_columns", side_effect=fake_bench):
        processor._run_elf64_production_benchmarks(columns, selected, record)
        assert {row[0] for row in records} == NAMES
        assert all(row[1:5] == (24, 17, 11, 13) for row in records)
        records.clear()
        # Only the explicit double source reaches Elf, not narrowed codec_cols.
        processor._run_shared_codec_benchmarks(
            [[1.0]], selected, record, double_native_cols=columns,
        )
        assert {row[0] for row in records} == NAMES
        records.clear()
        processor._run_shared_codec_benchmarks(
            [[float((1 << 63) - 1)]], selected, record,
            double_native_cols=[[float((1 << 63) - 1)]],
            int_native_cols=[[(1 << 63) - 1]],
        )
        assert not records, "int64 must not be cast through double for Elf"
        processor._run_shared_codec_benchmarks([[1.0]], selected, record)
        assert not records, "no unknown-precision fallback"

    for unsupported in ((16, 17, 11, 13), (0, 0, 0, 0)):
        with patch.object(elf, "_elf_jar", return_value="elf.jar"), \
             patch.object(elf, "_benchmark_jar", return_value="benchmark.jar"), \
             patch.object(elf, "bench_elf_on_chimp_double_columns", return_value=unsupported), \
             patch.object(elf, "bench_elf_plus_double_columns", return_value=unsupported):
            processor._run_elf64_production_benchmarks(columns, selected, record)
            assert not records, "partially verified columns cannot change denominator"
    with patch.object(elf, "_elf_jar", return_value=None), \
         patch.object(elf, "bench_elf_on_chimp_double_columns") as bench:
        processor._run_elf64_production_benchmarks(columns, selected, record)
        bench.assert_not_called()
        assert not records

    # Both the regular and literal-fast byte paths must exclude Elf and Camel.
    for fast in (False, True):
        with patch.object(patch_bench, "native_lz_literal_fast_enabled", return_value=fast), \
             patch.object(patch_bench, "bench_native_lz_family_bytes") as generic, \
             patch.object(patch_bench, "bench_native_lz_family_literal_bytes") as literal:
            patch_bench.accumulate_generic_byte_compression(
                [b"abcdefgh"], ["x"], record_fn=record,
                should_run=lambda name: name in NAMES | {"camel"},
            )
            generic.assert_not_called()
            literal.assert_not_called()
            assert not records
    for name in NAMES | {"camel"}:
        with patch.dict(os.environ, {"WEB_COMPRESSION_RUN_ALGORITHMS": name}):
            assert not patch_bench._byte_only_allowlist()

    # process_csv has an inline native loop; audit its actual loop constants.
    tree = ast.parse((ROOT / "backend/myapp/processor.py").read_text())
    full_csv = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_csv")
    loops = [node for node in ast.walk(full_csv) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "algorithm"]
    assert loops
    assert all(not (set(ast.literal_eval(loop.iter)) & (NAMES | {"camel"})) for loop in loops)

    # Bridge must forward exactly the measured codec times returned by Java.
    def subprocess_probe(argv, **kwargs):
        sent = kwargs["input"]
        assert struct.unpack_from("<ii", sent) == (1, 3)
        assert sent[8:] == struct.pack("<3d", 1.2345678901234567, -0.0, 1.25)
        assert argv[-1] == "elf_on_chimp"
        return SimpleNamespace(returncode=0, stdout=struct.pack("<qqqq", 24, 17, 11, 13))
    with patch.object(elf, "_elf_jar", return_value="elf.jar"), \
         patch.object(elf, "_benchmark_jar", return_value="benchmark.jar"), \
         patch.object(elf.subprocess, "run", side_effect=subprocess_probe):
        assert elf.bench_elf_on_chimp_double_columns(
            [[1.2345678901234567, -0.0, 1.25]],
        ) == (24, 17, 11, 13)

    # Genuine local-JAR smoke tests. If missing, explicitly report unsupported.
    real_results = []
    if elf._elf_jar() and elf._benchmark_jar():
        import audit_86_composability_realdata as realdata
        for rows in (128, 2048):
            real_columns, _ = realdata.load_external_columns(rows)
            for column in real_columns:
                if column.logical_type != "float":
                    continue
                values = [value for value in column.values if not math.isnan(value)]
                records.clear()
                processor._run_elf64_production_benchmarks([values], selected, record)
                by_name = {row[0]: row for row in records}
                for name in sorted(NAMES):
                    row = by_name.get(name)
                    if row:
                        assert row[1] == 8 * len(values)
                        assert row[2] > 0 and row[3] >= 0 and row[4] >= 0
                    real_results.append({
                        "dataset": column.dataset, "column": column.name,
                        "requested_rows": rows, "method": name,
                        "status": "PASS" if row else "UNSUPPORTED_FULL_ROUNDTRIP",
                        "original_bytes": 8 * len(values),
                        "chunk_body_bytes": row[2] if row else None,
                    })
    else:
        real_results.append({"status": "UNSUPPORTED_MISSING_JAR"})
    print(json.dumps({"routing_status": "PASS", "real_results": real_results,
                      "wire_note": "Java chunk bodies differ from benchmark86 EL64 framing"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
