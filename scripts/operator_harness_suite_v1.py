"""Deterministic, bounded local examples for the native operator harness.

These fixtures qualify interfaces and timers, not dataset-wide performance.
Parameters and graph selection are prepared offline and are not timed.
"""
from __future__ import annotations

import random
import struct


def root(values, dtype="int64"):
    formats = {"int64": ("q", 8), "uint64": ("Q", 8), "uint32": ("I", 4)}
    if dtype == "bytes":
        data, width, kind = bytes(values), 1, "serial"
    else:
        fmt, width = formats[dtype]
        data = b"".join(struct.pack("<" + fmt, value) for value in values)
        kind = "numeric"
    return {"kind": kind, "width": width, "data_hex": data.hex(),
            "original_numeric_payload": True, "logical_view": dtype}


def stage(input_id, action, profile="bytes", **parameters):
    return {"kind": "stage", "inputs": [input_id],
            "stage": {"action": action, "profile": profile,
                      "parameters": {key: str(value) for key, value in parameters.items()}}}


def case(case_id, ids, input_root, steps, description):
    return {"case_id": case_id, "implementation_ids": ids, "block_id": "independent_block_0",
            "description": description, "roots": [input_root],
            "recipe": {"policy_registry": "operator-harness-v1",
                       "steps": [*steps, {"kind": "stop"}]}}


FOR = "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1"
PACK = "op.bitpack_int.max_value_width.cpp_v1"
PLANES = "harness.a048.byte_planes.v1"
EXCEPTIONS = "op.outlier_separation.high_bits.uint32.cpp_v1"
XOR = "op.xor.previous_word.semantic_ports.cpp_v1"
ZIGZAG = "op.zigzag.signed64.raw.cpp_v2"
ULEB = "op.integer_code.uleb128.reference_values.cpp_v2"
HUFFMAN = "op.entropy_huffman.project_bytes.cpp_v1"
RANS = "op.entropy_fse.zstd_rans.cpp_reference_v1"


def representative_suite():
    examples = [
        case("single_a013", ["A013"], root([-3, 0, 7, 7, 1]),
             [stage(0, FOR, "int64")], "Independent FOR result submission"),
        case("single_a020", ["A020"], root([0, 1, 7, 255, 256], "uint64"),
             [stage(0, PACK, "uint64")], "Independent Bit Packing result submission"),
        case("single_a023", ["A023"], root([0, 127, 128, (1 << 64) - 1], "uint64"),
             [stage(0, ULEB, "uint64")], "Independent ULEB128 result submission"),
        case("single_a031", ["A031"], root([0, 7, 7, 3, (1 << 64) - 1], "uint64"),
             [stage(0, XOR, "uint64")], "Independent ordered XOR result submission"),
        case("single_a046", ["A046"], root([-(1 << 63), -1, 0, 1, (1 << 63) - 1]),
             [stage(0, ZIGZAG, "int64")], "Independent signed ZigZag result submission"),
        case("single_a048", ["A048"], root([0, 1, 255, 256, 511], "uint64"),
             [stage(0, PLANES, "uint64", bit_width=9)], "Independent two-plane result submission"),
        case("single_a077", ["A077"], root(b"aaabcaaba" * 100, "bytes"),
             [stage(0, HUFFMAN)], "Independent Huffman result submission"),
        case("single_a080", ["A080"], root(b"aaabcaaba" * 100, "bytes"),
             [stage(0, RANS)], "Independent rANS result submission"),
        case("single_a111", ["A111"], root([0, 127, 128, (1 << 32) - 1], "uint32"),
             [stage(0, EXCEPTIONS, "uint32", width=7)], "Independent three-port exception result submission"),
    ]
    for name, values in [
        ("smooth", [1_700_000_000_000 + index % 4096 for index in range(4096)]),
        ("extremes", [-(1 << 63), (1 << 63) - 1, 0, -1, 1]),
        ("empty", []),
        ("constant", [17] * 1024),
    ]:
        examples.append(case("for_pack_" + name, ["A013", "A020"], root(values),
                             [stage(0, FOR, "int64"), stage(2, PACK, "uint64")],
                             "FOR minimum bypass plus packed unsigned offsets"))
    examples.append(case("for_planes_entropy", ["A013", "A048", "A077", "A080"],
                         root([5_000 + index % 4096 for index in range(4096)]),
                         [stage(0, FOR, "int64"), stage(2, PLANES, "uint64", bit_width=12),
                          stage(3, RANS), stage(4, HUFFMAN)],
                         "Two byte-plane leaves with independent entropy coders; minimum bypass"))
    for name, values in [("constant", [25] * 1024), ("empty", [])]:
        examples.append(case("for_zero_planes_" + name, ["A013", "A048"], root(values),
                             [stage(0, FOR, "int64"), stage(2, PLANES, "uint64", bit_width=0)],
                             "Zero semantic planes use an empty physical carrier for the existing graph ABI"))
    for width in (0, 7, 32):
        examples.append(case(f"exception_three_ports_b{width}", ["A111", "A077"],
                             root([0, 1, 127, 128, 255, 1 << 31, (1 << 32) - 1] * 50, "uint32"),
                             [stage(0, EXCEPTIONS, "uint32", width=width),
                              stage(1, HUFFMAN), stage(2, HUFFMAN), stage(3, HUFFMAN)],
                             "Low, positions and high ports are all retained and independently encoded"))
    for name, values in [("mixed", [0, -1, 1, -(1 << 63), (1 << 63) - 1] * 200),
                         ("empty", [])]:
        examples.append(case("zigzag_uleb_" + name, ["A046", "A023"], root(values),
                             [stage(0, ZIGZAG, "int64"), stage(1, ULEB, "uint64")],
                             "Signed extremes remain exact across ZigZag and unsigned ULEB128"))
    for name, values in [("repeated", [0, 1, 1, 3, 3, (1 << 64) - 1] * 400), ("empty", [])]:
        examples.append(case("xor_state_" + name, ["A031", "A020"], root(values, "uint64"),
                             [stage(0, XOR, "uint64"), stage(2, PACK, "uint64")],
                             "Ordered adjacent-XOR history; first word is a separate bypass port"))
    # Raw IEEE words avoid Python float canonicalization of NaN payloads.
    ieee_words = [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                  0x7ff8000000001234, 0x7ff0000000000001, 1, 0x000fffffffffffff] * 100
    ieee_root = root(ieee_words, "uint64")
    ieee_root["logical_view"] = "float64"
    examples.append(case("xor_ieee_payloads", ["A031", "A020"], ieee_root,
                         [stage(0, XOR, "float64"), stage(2, PACK, "uint64")],
                         "Preserve signed zero, infinity, NaN payloads and subnormal raw bits"))
    rng = random.Random(102)
    for identifier, action in [("A077", HUFFMAN), ("A080", RANS)]:
        for name, values in [("empty", b""), ("constant", b"q" * 4096),
                             ("random", bytes(rng.randrange(256) for _ in range(4096)))]:
            examples.append(case(identifier.lower() + "_" + name, [identifier], root(values, "bytes"),
                                 [stage(0, action)], "Static byte coder: empty, constant or high-entropy input"))
    return {"schema": "operator-harness-suite-v1", "suite_id": "representative-local-v1",
            "scope": "synthetic interface qualification and local fixed-recipe timing; no dataset ranking",
            "input_byte_order": "little", "state_scope": "reset at each independently encoded case",
            "nncp_policy": "disabled_by_policy", "cases": examples}
