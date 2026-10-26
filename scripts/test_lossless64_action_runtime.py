#!/usr/bin/env python3
"""Audit the revised 64-bit codecs through the public unified action boundary.

Elf/Elf+ and BUFF are compared with their independent C++ original-encode
entry points. CORAD is explicitly the new project CRD2 format: its Python
baseline shares stage functions, so its parity is NOT independent evidence of
compatibility with upstream or the old lossy CORAD. Its final wire is also
parsed and reversed independently here to check parameter/correction/tail
accounting. Codec wire sizes and UAR1 transport sizes are kept separate.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import zlib

import audit_86_composability_realdata as realdata
import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified
from myapp import corad_lossless_stages as corad


ROOT = Path(__file__).resolve().parents[1]
CHAINS = {
    "Elf": primitive.ELF64_ACTIONS,
    "Elf+": primitive.ELF64_PLUS_ACTIONS,
    "CORAD": primitive.CORAD_LOSSLESS_ACTIONS,
    "BUFF": primitive.BUFF_NATIVE_ACTIONS,
}
HEADER_KEYS = {
    "actions", "source_kind", "source_count", "max_point",
    "upstream_result_header", "stage_input_payload_count",
}


def native(method: str, command: str, raw: bytes) -> bytes:
    if method == "BUFF":
        binary, prefix = "buff_native_exact_stages_cli", ""
    else:
        binary = "elf64_exact_stages_cli"
        prefix = "elf-plus-" if method == "Elf+" else "elf-"
    return subprocess.run(
        [str(ROOT / "cpp/build" / binary), prefix + command],
        input=raw, capture_output=True, check=True,
    ).stdout


def baseline_encode(method: str, source: unified.ActionData) -> bytes:
    if method == "CORAD":
        return corad.encode(source.payload, source.kind)
    return native(method, "original-encode", source.payload)


def baseline_decode(method: str, wire: bytes, kind: str) -> bytes:
    if method == "CORAD":
        return corad.decode(wire, kind)
    return native(method, "original-decode", wire)


def independent_corad_decode(wire: bytes, source: unified.ActionData) -> None:
    """Parse the complete stored CRD2 state without calling CORAD inverses."""
    assert wire[:4] == b"CRD2"
    assert struct.unpack_from("<Q", wire, 4)[0] == len(wire) - 12
    compressed = wire[12:]
    assert compressed[:4] == b"CZL2"
    size = struct.unpack_from("<Q", compressed, 4)[0]
    decoder = zlib.decompressobj()
    packet = decoder.decompress(compressed[12:]) + decoder.flush()
    assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    assert len(packet) == size
    magic, kind, count, mean, sigma = struct.unpack_from("<4sBQdd", packet)
    assert magic == b"CTK2" and count == source.count
    assert kind == {"float64": 0, "int64": 1}[source.kind]
    assert math.isfinite(mean) and math.isfinite(sigma) and sigma > 0
    trick, atoms = struct.unpack_from("<HB", packet, 29)
    blocks = (count + 39) // 40
    assert trick == 40 and atoms == min(4, blocks)
    # This exact extent includes every dictionary atom, block selector,
    # partial-tail residual, and original-word correction.
    assert len(packet) == 32 + atoms * 320 + blocks + count * 16
    atom_words = struct.unpack_from(f"<{atoms * 40}Q", packet, 32)
    pos = 32 + atoms * 320
    selectors = packet[pos:pos + blocks]
    pos += blocks
    residuals = struct.unpack_from(f"<{count}Q", packet, pos)
    corrections = struct.unpack_from(f"<{count}Q", packet, pos + count * 8)
    restored = []
    for index, (residual, correction) in enumerate(zip(residuals, corrections)):
        selector = selectors[index // 40]
        assert selector < atoms
        z_word = residual ^ atom_words[selector * 40 + index % 40]
        z = struct.unpack("<d", struct.pack("<Q", z_word))[0]
        assert math.isfinite(z)
        prediction = z * sigma + mean
        if kind == 0:
            predicted_word = struct.unpack("<Q", struct.pack("<d", prediction))[0]
        elif math.isfinite(prediction):
            predicted_word = min((1 << 63) - 1, max(-(1 << 63), round(prediction))) & ((1 << 64) - 1)
        else:
            predicted_word = 0
        restored.append(predicted_word ^ correction)
    assert struct.pack(f"<{count}Q", *restored) == source.payload


def check_header(header: dict, source: unified.ActionData, depth: int) -> None:
    current = header
    seen = 0
    while current is not None:
        # No opaque source bytes, corrections, or arrays may hide in transport
        # metadata and then escape the codec-wire denominator.
        assert set(current) <= HEADER_KEYS, current.keys()
        assert len(current["actions"]) == 1
        assert current["actions"][0] in unified.ACTION_BY_NAME
        assert current["max_point"] is None
        assert isinstance(current["source_count"], int) and current["source_count"] >= 0
        assert isinstance(current["stage_input_payload_count"], int)
        assert current["stage_input_payload_count"] >= 0
        seen += 1
        upstream = current.get("upstream_result_header")
        if upstream is None:
            assert current["source_kind"] == source.kind
            assert current["source_count"] == source.count
        else:
            assert current["source_kind"] == "bytes"
            assert isinstance(upstream, dict)
        current = upstream
    assert seen == depth


def check_case(label: str, method: str, source: unified.ActionData) -> dict:
    actions = CHAINS[method]
    current = source
    for depth, action in enumerate(actions, 1):
        previous = current
        current = unified.run_action(action, previous)
        assert current.kind == "bytes", (label, method, action)
        assert unified.run_action(action, current, reverse=True) == previous, (
            label, method, action, "single-stage reverse",
        )
        check_header(unified._unpack_result(current)[0], source, depth)
    _header, wire = unified._unpack_result(current)
    expected = baseline_encode(method, source)
    assert wire == expected, (label, method, "source encoder wire parity")
    assert baseline_decode(method, wire, source.kind) == source.payload, (
        label, method, "standalone wire decode without UAR1",
    )
    if method == "CORAD":
        independent_corad_decode(wire, source)
    restored = current
    for action in reversed(actions):
        restored = unified.run_action(action, restored, reverse=True)
    assert restored == source, (label, method, "stepwise reverse")
    full = unified.run_actions(actions, source)
    assert full == current, (label, method, "run_actions forward")
    assert unified.run_actions(actions, full, reverse=True) == source, (
        label, method, "run_actions reverse",
    )
    transport_header_bytes = 8 + struct.unpack_from("<I", current.payload, 4)[0]
    assert len(current.payload) == transport_header_bytes + len(wire)
    return {"case": label, "method": method, "kind": source.kind,
            "rows": source.count, "raw_bytes": len(source.payload),
            "codec_wire_bytes": len(wire), "uar1_total_bytes": len(current.payload)}


def check_policy() -> dict:
    policy = primitive.POLICY_ACTIONS
    assert len(policy) == 156, "Update audited count intentionally if registry changes"
    assert [item.action_id for item in policy] == list(range(156))
    assert primitive.STOP_ACTION_ID == 156
    assert primitive.STOP_ACTION_ID not in primitive.POLICY_ACTION_BY_ID
    assert len({item.name for item in policy}) == len(policy)
    concrete = primitive.ALL_CONCRETE_ACTION_IMPLEMENTATIONS
    assert len({item.action_id for item in concrete}) == len(concrete)
    assert len({item.name for item in concrete}) == len(concrete)
    for item in policy:
        target = primitive.ALL_CONCRETE_ACTION_BY_NAME[
            primitive.CANONICAL_TO_LEGACY_NAME[item.name]]
        assert primitive.POLICY_TO_CONCRETE_ACTION_ID[item.action_id] == target.action_id
        assert primitive.POLICY_ACTION_BY_ID[item.action_id].name == item.name
    for method, actions in CHAINS.items():
        for index, action in enumerate(actions):
            desc = unified.describe_action(action)
            kinds = ("float64",) if method in ("Elf", "Elf+") else ("int64", "float64")
            assert desc.accepted_input_kinds == (kinds if index == 0 else ("bytes",))
            assert desc.execution_mode == "stagewise"
            assert primitive.ACTION_CANONICAL_NAMES.get(action, action) in primitive.POLICY_ACTION_BY_NAME
    return {"actions": len(policy), "stop_id": primitive.STOP_ACTION_ID,
            "stable_concrete_ids": len(concrete)}


def main() -> None:
    policy = check_policy()
    results, counts = [], Counter()
    for rows in (128, 2048):
        columns, _manifest = realdata.load_external_columns(rows)
        assert len(columns) == 8
        assert sum(col.logical_type == "float" for col in columns) == 4
        for col in columns:
            kind = "int64" if col.logical_type == "integer" else "float64"
            assert kind != "float64" or not any(math.isnan(value) for value in col.values)
            source = unified.ActionData(kind, col.raw, len(col.values))
            label = f"{col.dataset}/{col.name}/requested-{rows}"
            for method in CHAINS:
                if kind == "int64" and method in ("Elf", "Elf+"):
                    continue
                result = check_case(label, method, source)
                # Independently check that benchmark dispatch selected the same
                # revised source codec and the already NaN-filtered denominator.
                art, reason = realdata.benchmark.specialist_artifact(col, method)
                assert art is not None, (label, method, reason)
                assert art.payload == baseline_encode(method, source)
                assert art.decode(art.payload) == col.raw
                results.append(result)
                counts[f"real_{method}"] += 1

    rng = random.Random(20260905)
    edge_words = [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                  1, 0x8000000000000001, 0x7fefffffffffffff, 0xffefffffffffffff,
                  0x3ff3c0ca428c59fb, 0x3ff4000000000000]
    cases = [("empty", []), ("one", [0]), ("ieee-finite-inf-zero", edge_words)]
    for count in (39, 40, 41, 79, 80, 81, 1600, 1601, 2048):
        words = [rng.getrandbits(64) for _ in range(count)]
        cases.append((f"random-words-{count}", words))
    cases.append(("large-int64", [word & ((1 << 64) - 1) for word in (
        -(1 << 63), (1 << 63) - 1, (1 << 53) + 1, -(1 << 53) - 1,
        (1 << 53) - 1, -(1 << 53) + 1, -1, 0, 1,
    )]))
    cases.append(("nan-payloads", [0x7ff800000000002a, 0x7ff0000000000001,
                                  0xfff8000000000042]))
    for label, words in cases:
        raw = struct.pack(f"<{len(words)}Q", *words)
        for kind in ("float64", "int64"):
            source = unified.ActionData(kind, raw, len(words))
            for method in CHAINS:
                if method in ("Elf", "Elf+"):
                    if kind != "float64" or any(
                        math.isnan(value[0]) for value in struct.iter_unpack("<d", raw)
                    ):
                        counts["elf_unsupported_type_or_nan_skips"] += 1
                        continue
                check_case(label, method, source)
                counts[f"edge_{method}"] += 1

    # NaN filtering retains the exact large integers and float64 signed zeros.
    bench = realdata.benchmark
    mixed = bench.Column("test", 0, "mixed", "float", 0,
                         (float("nan"), -0.0, float("inf"), 1.2345678901234567))
    assert mixed.source_count == 4 and mixed.removed_nan_count == 1
    assert mixed.raw == struct.pack("<3d", -0.0, float("inf"), 1.2345678901234567)
    large = bench.Column("test", 0, "large", "integer", 0, ((1 << 63) - 1, -(1 << 63)))
    assert large.raw == struct.pack("<2q", (1 << 63) - 1, -(1 << 63))
    print(json.dumps({"status": "PASS", "policy": policy, "checks": counts,
                      "byte_parity_and_bitwise_lossless": True, "real_results": results},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
