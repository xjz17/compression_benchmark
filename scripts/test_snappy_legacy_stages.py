#!/usr/bin/env python3
"""Independent python-snappy wire oracle for the extracted snap match stage."""

from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import snappy  # noqa: E402
from myapp.snappy_legacy_stages import lz_match_forward, lz_match_inverse  # noqa: E402


def _cli(command: str, data: bytes) -> bytes:
    return subprocess.run(
        [str(ROOT / "cpp/build/snappy_exact_stages_cli"), command],
        input=data, capture_output=True, check=True,
    ).stdout


def check(raw: bytes) -> bool:
    original = snappy.compress(raw)
    # The extracted matcher must keep working with the whole encoder disabled.
    with patch("snappy.compress", side_effect=AssertionError("whole codec called")):
        match_packet = lz_match_forward(raw)
    assert lz_match_inverse(match_packet) == raw
    token_packet = _cli("token-forward", match_packet)
    assert _cli("token-inverse", token_packet) == match_packet
    wire = _cli("integer-forward", token_packet)
    assert _cli("integer-inverse", wire) == token_packet
    assert wire == original, (len(raw), len(wire), len(original))
    assert snappy.decompress(wire) == raw
    return _cli("original-encode", raw) != original


def main() -> None:
    from audit_86_composability_realdata import load_external_columns

    rng = random.Random(705)
    lengths = (0, 1, 4, 16, 17, 59, 60, 61, 64, 65, 67, 68, 255, 256,
               1024, 16384, 65535, 65536, 65537, 131073)
    boundaries = [
        raw
        for length in lengths
        for raw in (
            bytes(length),
            rng.randbytes(length),
            (b"ABCD1234" * ((length + 7) // 8))[:length],
        )
    ]
    # Increasing entropy and repeated runs stress hash skips and split copies.
    boundaries += [
        rng.randbytes(400) * 3 + b"x" * length + rng.randbytes(40)
        for length in (4, 11, 12, 60, 64, 65, 66, 67, 68, 128, 2048)
    ]
    for raw in boundaries:
        check(raw)
    columns, _ = load_external_columns(128)
    rows = []
    for column in columns:
        cpp_differs = check(column.raw)
        rows.append({"dataset": column.dataset, "column": column.name,
                     "raw_bytes": len(column.raw),
                     "legacy_bytes": len(snappy.compress(column.raw)),
                     "canonical_cpp_differs": cpp_differs})
    malformed = (b"", b"SNM1", b"NOPE" + bytes(8),
                 b"SNM1" + bytes(7) + b"\x01")
    for packet in malformed:
        try:
            lz_match_inverse(packet)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed packet accepted")
    print(json.dumps({"real_columns": len(rows), "rows_per_column": 128,
                      "boundary_cases": len(boundaries),
                      "all_wires_identical": True, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
