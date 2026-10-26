#!/usr/bin/env python3
"""CORAD v2 standalone packets and BUFF original baseline bitwise validation."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from myapp import corad_lossless_stages as corad


def check(raw: bytes, kind: str) -> int:
    current = raw
    packets = []
    for action in corad.ACTION_NAMES:
        previous = current
        current = corad.forward(action, current, kind if action == corad.NORMALIZE else "bytes")
        assert corad.inverse(action, current) == previous, (kind, action)
        packets.append(current)
    assert current == corad.encode(raw, kind)
    assert corad.decode(current, kind) == raw
    assert corad.decode(current) == raw
    assert current.startswith(b"CRD2")
    for action in reversed(corad.ACTION_NAMES):
        current = corad.inverse(action, current)
    assert current == raw
    # Required original data is entirely recoverable from final wire, with no
    # ActionData/UAR1 object or out-of-band source/header supplied to decode.
    assert len(packets[0]) == 29 + 2 * len(raw)
    return len(packets[-1])


def reject(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("malformed CORAD packet accepted")


def buff(command: str, raw: bytes) -> bytes:
    return subprocess.run(
        [str(ROOT / "cpp/build/buff_native_exact_stages_cli"), command],
        input=raw, capture_output=True, check=True,
    ).stdout


def check_benchmark_entrypoints() -> None:
    from myapp import research_extended_codecs as benchmark
    columns = [[], [0.0], [-0.0, float("inf"), float("-inf")], list(map(float, range(81)))]
    expected = sum(len(corad.encode(struct.pack(f"<{len(col)}d", *col), "float64"))
                   for col in columns if col)
    for function in (benchmark.bench_corad_tricklet_float_columns,
                     benchmark.bench_corad_tricklet_double_columns):
        result = function(columns)
        assert result[0] == 85 * 8 and result[1] == expected
        assert result[4:] == (0.0, 85.0)
        assert function([[]]) == (0, 0, 0, 0, 0.0, 0.0)
        assert function([[float("nan")]]) == (0, 0, 0, 0, 0.0, 0.0)
        reject(lambda: function([[1.0]], trick=41))
        reject(lambda: function([[1.0]], atoms=3))
    integers = [-(1 << 63), (1 << 63) - 1, (1 << 53) + 1, 0, 1]
    result = benchmark.bench_corad_tricklet_int64_columns([integers])
    assert result[0] == 40
    assert result[1] == len(corad.encode(struct.pack("<5q", *integers), "int64"))
    with patch.object(corad, "decode", return_value=b"corrupt"):
        try:
            benchmark.bench_corad_tricklet_double_columns([[1.0]])
        except RuntimeError:
            pass
        else:
            raise AssertionError("CORAD benchmark counted a failed original-value decode")


def main() -> None:
    check_benchmark_entrypoints()
    rng = random.Random(20260905)
    edge_words = [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                  0x7ff800000000002a, 0x7ff0000000000001, 1, 0x8000000000000001,
                  0x7fefffffffffffff, 0xffefffffffffffff, (1 << 53) + 1, (1 << 63) - 1]
    cases = [[], [0], [0x8000000000000000], edge_words]
    for count in (1, 39, 40, 41, 79, 80, 81, 128, 2048):
        cases.append([rng.getrandbits(64) for _ in range(count)])
        cases.append([0x3ff4000000000000] * count)
    total = 0
    for words in cases:
        raw = struct.pack(f"<{len(words)}Q", *words)
        for kind in ("int64", "float64"):
            check(raw, kind)
            total += 1
        native_wire = buff("original-encode", raw)
        assert buff("original-decode", native_wire) == raw
        assert buff("compose", raw) == native_wire

    x = struct.pack("<80d", *range(80))
    y = struct.pack("<80d", *(2 * i + 8 for i in range(80)))
    assert corad.encode(x, "float64") != corad.encode(y, "float64")
    for raw in (x, y):
        assert corad.decode(corad.encode(raw, "float64")) == raw

    good = corad.encode(x, "float64")
    reject(lambda: corad.normalize_forward(b"bad", "float64"))
    reject(lambda: corad.normalize_forward(x, "bytes"))
    reject(lambda: corad.decode(good, "int64"))
    reject(lambda: corad.decode(good[:-1]))
    reject(lambda: corad.decode(good + b"extra"))
    reject(lambda: corad.decode(b"CRD1" + good[4:]))
    reject(lambda: corad.entropy_inverse(b"CZL2" + struct.pack("<Q", (1 << 64) - 1)))
    normalized = corad.normalize_forward(x, "float64")
    reject(lambda: corad.normalize_inverse(normalized[:-1]))
    tricklet = corad.tricklet_forward(normalized)
    reject(lambda: corad.tricklet_inverse(tricklet[:-1]))
    bad = bytearray(tricklet)
    bad[32 + 2 * 40 * 8] = 255
    reject(lambda: corad.tricklet_inverse(bytes(bad)))

    import audit_86_composability_realdata as realdata
    real = []
    for rows in (128, 2048):
        columns, _manifest = realdata.load_external_columns(rows)
        for column in columns:
            kind = "int64" if column.logical_type == "integer" else "float64"
            if kind == "float64":
                words = corad._words(column.raw)
                words = [word for word in words if not math.isnan(corad._float(word))]
                raw = corad._pack_words(words)
            else:
                raw = column.raw
            size = check(raw, kind)
            row = {"dataset": column.dataset, "column": column.name, "requested_rows": rows,
                   "filtered_nan_rows": rows - len(raw) // 8,
                   "rows": len(raw) // 8, "kind": kind, "original_bytes": len(raw),
                   "corad_v2_bytes": size, "lossless": True}
            if kind == "float64":
                wire = buff("original-encode", raw)
                assert buff("original-decode", wire) == raw
                assert wire == buff("compose", raw)
                row["buff_native_bytes"] = len(wire)
            real.append(row)
    print(json.dumps({"status": "PASS", "corad_edge_cases": total,
                      "production_python_entrypoints": "PASS",
                      "buff_edge_cases": len(cases), "rejected_bad_packets": 10,
                      "old_collision_fixed": True, "real_cases": real}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
