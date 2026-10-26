#!/usr/bin/env python3
"""Verify Python BUFF raw64 wire and production benchmark timing boundaries."""
from __future__ import annotations

import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from myapp import buff_kernel_native as codec
from myapp import native_research_cli_python as benchmark


def original(command: str, raw: bytes) -> bytes:
    return subprocess.run(
        [str(ROOT / "cpp/build/buff_native_exact_stages_cli"), command],
        input=raw, capture_output=True, check=True,
    ).stdout


def check(raw: bytes) -> None:
    wire = codec.encode_raw64(raw)
    assert wire == original("original-encode", raw)
    assert codec.decode_raw64(wire) == raw
    assert original("original-decode", wire) == raw
    assert len(wire) == (len(raw) + 12 if raw else 0)


def main() -> None:
    rng = random.Random(20260905)
    cases = [[], [0], [0, 0x8000000000000000, 0x7ff0000000000000,
                      0xfff0000000000000, 0x7ff0000000000001,
                      0x7ff800000000002a, 1, (1 << 64) - 1]]
    for count in (39, 40, 41, 79, 80, 81, 128, 2048):
        cases.append([rng.getrandbits(64) for _ in range(count)])
    for words in cases:
        check(struct.pack(f"<{len(words)}Q", *words))

    integers = [-(1 << 63), (1 << 63) - 1, (1 << 53) + 1, 0, 1]
    expected_integer_raw = struct.pack("<5q", *integers)
    assert codec.pack_raw64_values(integers, "int64") == expected_integer_raw
    floats = [-0.0, float("inf"), float("nan"), 1.25]
    filtered_raw = struct.pack("<3d", -0.0, float("inf"), 1.25)
    assert codec.pack_raw64_values(floats, "float64") == filtered_raw
    with patch.dict(os.environ, {"COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE": "1"}):
        # Three calls bracket real encode/decode; scan/filter costs are outside.
        with patch.object(benchmark.time, "perf_counter_ns", side_effect=(100, 110, 140)):
            result = benchmark.bench_buff_float_kernel_native_columns([floats])
        assert result == (24, 36, 10, 30), result
        result = benchmark.bench_buff_int_kernel_native_columns([integers])
        assert result[:2] == (40, 52), result
        with patch.object(codec, "decode_raw64", return_value=b"bad"):
            try:
                benchmark.bench_buff_float_kernel_native_columns([[1.25]])
            except RuntimeError:
                pass
            else:
                raise AssertionError("BUFF benchmark counted a failed lossless decode")
        try:
            benchmark.bench_buff_int_kernel_native_columns([[float((1 << 53) + 1)]])
        except struct.error:
            pass
        else:
            raise AssertionError("BUFF int entrypoint silently accepted a float-cast column")
        assert benchmark.bench_buff_float_kernel_native_columns([[float("nan")]]) == (0, 0, 0, 0)

    import audit_86_composability_realdata as realdata
    real_cases = 0
    for count in (128, 2048):
        columns, _manifest = realdata.load_external_columns(count)
        for column in columns:
            kind = "int64" if column.logical_type == "integer" else "float64"
            raw = codec.pack_raw64_values(column.values, kind)
            check(raw)
            with patch.dict(os.environ, {"COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE": "1"}):
                function = (benchmark.bench_buff_int_kernel_native_columns if kind == "int64"
                            else benchmark.bench_buff_float_kernel_native_columns)
                result = function([column.values])
            assert result[:2] == (len(raw), len(codec.encode_raw64(raw)))
            real_cases += 1
    print(f"BUFF raw64 Python baseline PASS: {len(cases)} bit-pattern cases, "
          f"{real_cases} real-column cases, C++ original wire/decoder parity, "
          "NaN filtering, exact int64, true decode timing, and fail-closed checks.")


if __name__ == "__main__":
    main()
