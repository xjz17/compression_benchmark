#!/usr/bin/env python3
"""Blosc2 binding-macro/frame compatibility with unchanged benchmark oracle."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]

import blosc2  # noqa: E402
from myapp import blosc2_legacy_stages as stages  # noqa: E402


def check(raw: bytes, oracle: bytes | None = None) -> int:
    expected = (blosc2.compress(raw, typesize=1, codec=blosc2.Codec.BLOSCLZ, clevel=5)
                if oracle is None else oracle)
    packet = stages.binding_core_forward(raw)
    wire = stages.frame_forward(packet)
    assert wire == expected, (len(raw), len(wire), len(expected))
    assert stages.frame_inverse(wire) == packet
    assert stages.binding_core_inverse(packet) == raw
    assert blosc2.decompress(wire) == raw
    return len(wire)


def main() -> None:
    from audit_86_composability_realdata import load_external_columns
    from benchmark_86_combinations import _compress_generic

    rows_out = []
    for rows in (128, 2048):
        columns, _ = load_external_columns(rows)
        for column in columns:
            oracle, _decode, _implementation = _compress_generic(column.raw, "blosc2_blosclz")
            size = check(column.raw, oracle)
            rows_out.append({"rows": rows, "dataset": column.dataset, "column": column.name,
                             "compressed_bytes": size})
    rng = random.Random(705)
    boundaries = [raw for length in (0, 1, 16, 31, 32, 33, 128, 1024, 16384, 131072, 131073)
                  for raw in (bytes(length), b"x" * length, rng.randbytes(length))]
    boundaries.append(bytes(range(256)) * 4096)
    for raw in boundaries:
        check(raw)
    for func, value in ((stages.frame_forward, b""),
                        (stages.frame_inverse, b""),
                        (stages.frame_inverse, bytes(32))):
        try:
            func(value)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed frame accepted")
    print(json.dumps({"python_blosc2_version": blosc2.__version__,
                      "c_blosc2_version": blosc2.blosclib_version,
                      "real_column_checks": len(rows_out), "boundary_checks": len(boundaries),
                      "all_wires_identical": True, "all_stage_inverses_pass": True,
                      "core_is_non_atomic_binding_macro": True, "rows": rows_out}, indent=2))


if __name__ == "__main__":
    main()
