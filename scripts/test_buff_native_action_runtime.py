#!/usr/bin/env python3
"""Check the two native BUFF actions against the original C++ encoder.

Uses exact bit-pattern comparisons, including NaN payloads and signed zeros.
Real-data probes are the four float columns from the external audit, at both
128 and 2,048 rows. No test compares the native wire to the older ablation wire.
"""

from __future__ import annotations

from pathlib import Path
import random
import struct
import subprocess

import audit_86_composability_realdata as realdata
import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified


ROOT = Path(__file__).resolve().parents[1]


def original_encode(raw: bytes) -> bytes:
    return subprocess.run(
        [str(ROOT / "cpp/build/buff_native_exact_stages_cli"), "original-encode"],
        input=raw, capture_output=True, check=True,
    ).stdout


def check_case(label: str, raw: bytes) -> None:
    source = unified.ActionData("float64", raw, len(raw) // 8)
    actions = primitive.BUFF_NATIVE_ACTIONS
    assert len(actions) == 2, actions
    current = source
    for action in actions:
        previous = current
        current = unified.run_action(action, previous)
        assert current.kind == "bytes", (label, action, current.kind)
        assert unified.run_action(action, current, reverse=True) == previous, (
            label, action, "single-stage reverse mismatch",
        )
    header, wire = unified._unpack_result(current)
    expected = original_encode(raw)
    assert wire == expected, (label, len(wire), len(expected), "source-wire mismatch")
    restored = current
    for action in reversed(actions):
        restored = unified.run_action(action, restored, reverse=True)
    assert restored == source, (label, "stepwise chain reverse mismatch")
    encoded_chain = unified.run_actions(actions, source)
    assert unified._unpack_result(encoded_chain)[1] == expected, (
        label, "run_actions source-wire mismatch",
    )
    assert unified.run_actions(actions, encoded_chain, reverse=True) == source, (
        label, "run_actions reverse mismatch",
    )


def main() -> None:
    edge_cases = [
        ("empty", []),
        ("single-zero", [0]),
        ("ieee-edge-bits", [
            0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
            0x7ff800000000002a, 0x7ff0000000000001, 1, 0x8000000000000001,
            0x7fefffffffffffff, 0xffefffffffffffff,
        ]),
    ]
    rng = random.Random(20260905)
    for count in (2, 7, 79, 80, 81, 128, 2048):
        edge_cases.append((f"random-bits-{count}", [rng.getrandbits(64) for _ in range(count)]))
    for label, words in edge_cases:
        check_case(label, struct.pack("<" + "Q" * len(words), *words))

    real_cases = 0
    for rows in (128, 2048):
        columns, _manifest = realdata.load_external_columns(rows)
        floats = [column for column in columns if column.logical_type == "float"]
        assert len(floats) == 4
        for column in floats:
            check_case(f"{column.dataset}/{column.name}/{rows}", column.raw)
            real_cases += 1
    print(
        f"BUFF native unified actions: {len(edge_cases)} edge cases and {real_cases} "
        "real-column cases passed; single-stage inverse, stepwise chain inverse, "
        "run_actions inverse, and byte parity with original C++ encoder all passed."
    )


if __name__ == "__main__":
    main()
