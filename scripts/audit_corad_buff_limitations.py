#!/usr/bin/env python3
"""Reproduce CORAD information loss and BUFF's distinct benchmark formats.

Requires buff_native_exact_stages_cli and corad_lossless_counterexample_test.
Prints JSON; does not alter codec sources or benchmark baselines.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "backend"):
    sys.path.insert(0, str(directory))

import audit_86_composability_realdata as realdata
from myapp.buff_kernel import encode_doubles as bounded_buff_encode


def run(binary: str, command: str, data: bytes) -> bytes:
    result = subprocess.run(
        [str(ROOT / "cpp/build" / binary), command],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    args = parser.parse_args()
    columns, manifest = realdata.load_external_columns(args.rows)
    results = []
    collision = None
    for column in columns:
        if column.logical_type != "float":
            continue
        raw = column.raw
        words = run("buff_native_exact_stages_cli", "word-order-forward", raw)
        wire = run("buff_native_exact_stages_cli", "frame-forward", words)
        original = run("buff_native_exact_stages_cli", "original-encode", raw)
        previous = raw
        for command in ("buff-scaling-forward", "buff-column-split-forward", "buff-bitpack-forward"):
            previous = run("learning_codec_exact_stages_cli", command, previous)
        assert wire == original
        assert run("buff_native_exact_stages_cli", "frame-inverse", wire) == words
        assert run("buff_native_exact_stages_cli", "word-order-inverse", words) == raw
        assert run("buff_native_exact_stages_cli", "decompose", wire) == raw
        results.append({
            "dataset": column.dataset, "column": column.name,
            "count": len(column.values), "old_action_bytes": len(previous),
            "native_original_bytes": len(original), "new_stage_bytes": len(wire),
            "byte_parity": wire == original,
            "old_action_equal_original": previous == original,
            "bitwise_roundtrip": True,
        })
        if collision is None:
            for index, value in enumerate(column.values):
                value = float(value)
                if not math.isfinite(value):
                    continue
                adjacent = math.nextafter(value, math.inf)
                payload = bounded_buff_encode([value], 10000)
                if payload == bounded_buff_encode([adjacent], 10000):
                    collision = {
                        "dataset": column.dataset, "column": column.name, "row": index,
                        "real_value": value, "adjacent_float": adjacent,
                        "real_bits_le": struct.pack("<d", value).hex(),
                        "adjacent_bits_le": struct.pack("<d", adjacent).hex(),
                        "same_bounded_wire_hex": payload.hex(),
                        "same_bounded_wire_bytes": len(payload), "scale": 10000,
                    }
                    break
    assert collision is not None
    corad = subprocess.run(
        [str(ROOT / "cpp/build/corad_lossless_counterexample_test")],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(json.dumps({
        "rows_per_column": args.rows,
        "corad_original_cpp_collision": corad,
        "legacy_python_buff_collision": collision,
        "current_cpp_buff": results,
        "source_manifest": manifest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
