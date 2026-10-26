#!/usr/bin/env python3
"""Compare actual REGER block/framing stages to the unchanged legacy oracle."""

from __future__ import annotations

import json
from pathlib import Path
import random
import struct
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]

from myapp import reger_codec as original  # noqa: E402
from myapp import reger_legacy_stages as stages  # noqa: E402


def check(raw: bytes, oracle: bytes | None = None) -> int:
    values = [value for (value,) in struct.iter_unpack("<q", raw)]
    expected = original.encode_int_column(values) if oracle is None else oracle
    with patch.object(original, "_encode_i64_columns", side_effect=AssertionError("whole codec called")):
        packet = stages.block_core_forward(raw)
    wire = stages.frame_forward(packet)
    assert wire == expected, (len(raw), len(wire), len(expected))
    assert stages.frame_inverse(wire) == packet
    with patch.object(original, "_decode_i64_columns", side_effect=AssertionError("whole decoder called")):
        assert stages.block_core_inverse(packet) == raw
    decoded = original.decode_int_column(wire)
    assert b"".join(struct.pack("<q", value) for value in decoded) == raw
    return len(wire)


def main() -> None:
    from audit_86_composability_realdata import load_external_columns
    from benchmark_86_combinations import specialist_artifact

    checks = []
    for rows in (128, 2048):
        columns, _ = load_external_columns(rows)
        for column in columns:
            artifact, reason = specialist_artifact(column, "REGER")
            assert artifact is not None, reason
            size = check(column.raw, artifact.payload)
            checks.append({"rows": rows, "dataset": column.dataset,
                           "column": column.name, "type": column.logical_type,
                           "compressed_bytes": size})
            print(json.dumps(checks[-1]), flush=True)
    rng = random.Random(705)
    boundaries = [b"", struct.pack("<q", 42), struct.pack("<q", -(1 << 63)),
                  struct.pack("<Q", 0x8000000000000000),
                  struct.pack("<QQQ", 0x7FF8000000000001, 0x7FF0000000000000,
                              0xFFF0000000000000)]
    for size in (16, 17, 512, 513, 514):
        boundaries.extend((bytes(size * 8),
                           b"".join(struct.pack("<q", i) for i in range(size))))
    boundaries.append(b"".join(struct.pack("<q", rng.randrange(-10000, 10000))
                               for _ in range(40)))
    for raw in boundaries:
        check(raw)
    with patch.dict("os.environ", {"WEB_COMPRESSION_REGER_BLOCK_SIZE": "17",
                                  "WEB_COMPRESSION_REGER_SEGMENT_SIZE": "4"}):
        check(b"".join(struct.pack("<q", i * i) for i in range(40)))
    for function, data in ((stages.block_core_forward, b"x"),
                           (stages.block_core_inverse, b"RGP1"),
                           (stages.frame_forward, b""),
                           (stages.frame_inverse, b"REGER3")):
        try:
            function(data)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed input accepted")
    print(json.dumps({"real_column_checks": len(checks),
                      "boundary_checks": len(boundaries) + 1,
                      "all_wires_identical": True, "all_stage_inverses_pass": True}))


if __name__ == "__main__":
    main()
