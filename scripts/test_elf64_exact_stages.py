#!/usr/bin/env python3
"""Real-column original/staged binary64 Elf comparison, including special values."""
import json
import math
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from myapp import elf64_exact_stages as elf


def check(raw):
    erased = elf.eraser_forward(raw)
    assert elf.eraser_inverse(erased) == raw
    sizes = []
    for plus in (False, True):
        baseline = elf.encode(raw, plus)
        wire = elf.xor_forward(erased, plus)
        assert wire == baseline
        assert elf.xor_inverse(wire, plus) == erased
        assert elf.decode(wire, plus) == raw
        sizes.append(len(wire))
    return sizes


def main():
    from audit_86_composability_realdata import load_external_columns
    rows_out = []
    for n in (128, 2048):
        columns, _ = load_external_columns(n)
        for column in columns:
            if column.logical_type != "float":
                continue
            values = [v for v in column.values if not math.isnan(v)]
            sizes = check(struct.pack(f"<{len(values)}d", *values))
            rows_out.append({"rows": len(values), "column": column.name,
                             "elf_bytes": sizes[0], "elf_plus_bytes": sizes[1]})
    rng = random.Random(705)
    boundaries = [[], [0.0], [-0.0], [math.inf], [-math.inf],
                  [-0.0, 0.0, math.inf, -math.inf, 1.25, 1.2345678901234567],
                  [5e-324, -5e-324, 1e-300, 1e300, -1e300],
                  [1.25] * 1601,
                  [rng.uniform(-1000, 1000) for _ in range(3201)]]
    bit_pattern_values = [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
                          for _ in range(512)]
    boundaries.append([value for value in bit_pattern_values if not math.isnan(value)])
    for values in boundaries:
        check(struct.pack(f"<{len(values)}d", *values))
    for bits in (0x7ff8000000000000, 0x7ff8000000000001, 0xfff8000000000001):
        try:
            elf.eraser_forward(struct.pack("<Q", bits))
        except ValueError:
            pass
        else:
            raise AssertionError("NaN must be rejected because of EOS collision")
    print(json.dumps({"rows": rows_out, "real_column_checks": len(rows_out),
                      "boundary_checks": len(boundaries), "nan_rejections": 3,
                      "all_byte_parity_and_lossless": True}, indent=2))


if __name__ == "__main__":
    main()
