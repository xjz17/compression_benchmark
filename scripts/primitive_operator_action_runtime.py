#!/usr/bin/env python3
"""RL-facing primitive operator action space.

Algorithm names such as ALP, LECO, FastPFor, Chimp, Zstd, and LZ4 are pipelines
or end-to-end codec oracles.  They are deliberately not actions in this
module.  An action is one concrete operator implementation from the
28-operator project taxonomy.

``LEGACY_PRIMITIVE_ACTIONS`` retains the original 29 reference/proxy records so
old IDs and packets remain readable. ``PRIMITIVE_ACTIONS`` is the active view
and excludes the retired identity Scaling proxy and the fused signed-integer
reference encoder. ``EXACT_ACTIONS`` is the callable implementation inventory;
helpers that invoke a whole codec or fuse several requested operators are kept
for forensic replay but excluded from ``POLICY_ACTIONS``.  The registry keeps
those boundaries explicit so wire reproduction cannot be mistaken for an
independent operator implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import generate_air_operator_pair_heatmap as operator_impl
import measure_selected_codec_cost as stream_codec
import action_cpp_migration as cpp_migration
import canonical127_registry as algorithm_io
import optimal_rle_actions as optimal_rle
import max_value_bitpack as bitpack
import minimum_for as minimum
import native_lz_parameters as lz_parameters
import current_action_parameters as action_parameters
import block_state_parameters as block_parameters
import native_window_parameters as window_parameters
import operator_implementation_catalog as implementation_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CATALOG = REPO_ROOT / "benchmark/reference/methods/method_operator_pipelines.json"
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from myapp.native_lz_family import (  # noqa: E402
    NativeLzPreset,
    NativeLzToken,
    _compress_native_lz_family_fast,
    _parse_native_lz_family_fast,
    _preset as native_lz_preset,
    compress_native_lz_family_staged_fast,
    decompress_native_lz_family,
)
from myapp import legacy_numeric_stages, snappy_legacy_stages, reger_legacy_stages, blosc2_legacy_stages  # noqa: E402
from myapp import elf64_exact_stages, corad_lossless_stages  # noqa: E402
from myapp.entropy_cpp import ans_decode, ans_encode, arithmetic_decode, arithmetic_encode  # noqa: E402
from myapp.column_encodings import (  # noqa: E402
    decode_dictionary_float,
    decode_dictionary_int,
    encode_dictionary_float,
    encode_dictionary_int,
)
from myapp.entropy_cpp import huffman_decode, huffman_encode  # noqa: E402
from myapp.cpp_action_bridge import run as _cpp_action_run  # noqa: E402


@dataclass(frozen=True)
class PrimitiveActionSpec:
    action_id: int
    name: str
    operator_id: str
    implementation_id: str
    implementation_source: str
    implementation_scope: str


@dataclass(frozen=True)
class ExactByteActionSpec:
    """A concrete byte-stage variant; registry evidence records exact/reference status."""

    name: str
    operator_id: str
    implementation_id: str
    implementation_source: str
    implementation_scope: str
    action_id: int = -1


IMPLEMENTATION_SCOPE: dict[str, str] = {
    "eraser": "lossless reference transform over normalized int64 values",
    "scaling": "identity after input normalization has already applied decimal scaling",
    "frame_of_reference": "column-wide minimum reference transform",
    "value_reorder": "stable value-sort reference transform with inverse permutation metadata",
    "regression_predictor": "first-order linear regression residual reference transform",
    "outlier_separation": "IQR clipping reference transform with exception side stream",
    "bitpack_int": "FOR-normalization reference stage; compact payload performs width selection",
    "delta": "previous-value delta reference transform",
    "xor": "previous-value XOR reference transform; Chimp history selection is a separate variant",
    "trailing_zero_pack": "shared trailing-zero reference transform",
    "dictionary": "unique-value dictionary reference transform",
    "zigzag": "signed-to-unsigned ZigZag reference transform",
    "block_transform": "block-local reversible ordering proxy, not a production BWT kernel",
    "symbol_recode": "frequency-ordered dictionary proxy for symbol recoding",
    "run_length": "value/count run-length reference transform",
    "lz_match": "repeated-value match reference transform, not a production LZ parser",
    "match_parse_policy": "reversible policy-effect proxy",
    "match_window_model": "reversible window/state-effect proxy",
    "match_token_format": "reversible nibble token-layout proxy",
    "integer_code": "unsigned integer-code materialization; signed values require an explicit ZigZag action",
    "bitstream_pack": "reversible nibble bitstream-layout proxy",
    "entropy_huffman": "frequency recoding proxy, not a production Huffman bitstream",
    "entropy_range": "frequency recoding proxy, not a production range coder",
    "entropy_fse": "frequency recoding proxy, not a production FSE/ANS coder",
    "entropy_special": "frequency recoding proxy for codec-specific entropy stages",
    "literal_entropy": "frequency recoding proxy for a separate literal stream",
    "block_state_control": "reversible state-boundary proxy",
    "container_frame": "reference header/trailer transform",
}


LEGACY_PRIMITIVE_ACTIONS: tuple[PrimitiveActionSpec, ...] = tuple(
    PrimitiveActionSpec(
        action_id=index,
        name=f"op.{operator_id}.reference_v1",
        operator_id=operator_id,
        implementation_id="reference_v1",
        implementation_source="cpp/src/reference_operator_actions.cpp",
        implementation_scope=IMPLEMENTATION_SCOPE[operator_id],
    )
    for index, operator_id in enumerate(stream_codec.OPERATOR_IDS)
) + (
    PrimitiveActionSpec(
        action_id=len(stream_codec.OPERATOR_IDS),
        name="op.xor.chimp_history128.reference_v1",
        operator_id="xor",
        implementation_id="chimp_history128_reference_v1",
        implementation_source="cpp/src/reference_operator_actions.cpp",
        implementation_scope="Chimp128 low-bit-indexed 128-value history selection followed by XOR",
    ),
)

RETIRED_IDENTITY_ACTION_NAMES = implementation_catalog.RETIRED_IDENTITY_ACTION_NAMES
RETIRED_REFERENCE_COMPOSITE_ACTION_NAMES = frozenset({
    "op.integer_code.reference_v1",
})
LEGACY_PRIMITIVE_ACTION_BY_NAME = {
    action.name: action for action in LEGACY_PRIMITIVE_ACTIONS
}
RETIRED_REFERENCE_OPERATOR_IDS = frozenset(
    action.operator_id for action in LEGACY_PRIMITIVE_ACTIONS
    if action.name in RETIRED_IDENTITY_ACTION_NAMES
)
PRIMITIVE_ACTIONS: tuple[PrimitiveActionSpec, ...] = tuple(
    action for action in LEGACY_PRIMITIVE_ACTIONS
    if action.name not in RETIRED_IDENTITY_ACTION_NAMES
    and action.name not in RETIRED_REFERENCE_COMPOSITE_ACTION_NAMES
)

ACTION_BY_NAME = {action.name: action for action in PRIMITIVE_ACTIONS}
ACTIONS_BY_OPERATOR_ID: dict[str, tuple[PrimitiveActionSpec, ...]] = {
    operator_id: tuple(action for action in PRIMITIVE_ACTIONS if action.operator_id == operator_id)
    for operator_id in stream_codec.OPERATOR_IDS
}
DEFAULT_ACTION_BY_OPERATOR_ID = {
    operator_id: actions[0] for operator_id, actions in ACTIONS_BY_OPERATOR_ID.items()
    if actions
}
if (len(LEGACY_PRIMITIVE_ACTIONS) != 29 or len(ACTION_BY_NAME) != 27
        or len(ACTIONS_BY_OPERATOR_ID) != 28 or ACTIONS_BY_OPERATOR_ID["scaling"]
        or ACTIONS_BY_OPERATOR_ID["integer_code"]):
    raise RuntimeError("reference-composite retirement changed the frozen or active primitive inventory")

CHIMP_XOR_ACTION = "op.xor.chimp_history128.reference_v1"
METHOD_ACTION_OVERRIDES: dict[str, dict[str, str]] = {
    "CHIMP": {"xor": CHIMP_XOR_ACTION},
    "Chimp128": {"xor": CHIMP_XOR_ACTION},
}
METHOD_ALIASES = {
    "Chimp": "CHIMP",
    "Gorilla": "GORILLA",
    "2DIFF": "TS_2DIFF",
    "SVB": "Stream VByte",
    "FastPFor": "fastpfor",
    "Simple8b": "simple8b",
    "Sprintz": "SPRINTZ",
    "Dict.": "col_dictionary",
    "DBP": "DELTA_BINARY_PACKED",
    "Camel": "camel",
    "Elf": "elf",
    "Elf+": "elf_plus",
    "FPC": "fpc",
    "PFPC": "pfpc",
    "SPDP": "spdp",
    "CORAD": "corad_tricklet",
    "BUFF": "buff",
    "FPCmp-R": "FPCompress DPratio",
    "FPCmp-S": "FPCompress DPspeed",
    "Brotli": "brotli",
    "Zstd": "zstd",
    "Snappy": "snappy",
    "NRV": "UCL NRV",
    "Libdeflate": "libdeflate",
    "Zlib": "zlib",
    "Zlib-ng": "zlib-ng",
}


def exact_native_lz_action_pipeline(algorithm_key: str) -> tuple[ExactByteActionSpec, ...]:
    """Return the real benchmark-86 native-LZ stage variants for one key.

    The benchmark fixes ``COMPRESSION_BENCHMARK_NATIVE_LZ_FAST_BYTES=0``.  In that
    path candidate-count and lazy flags are inactive; the byte stream is fully
    determined by the aligned-8 matcher, min/max match lengths, window, greedy
    parse, and ctrl8/offset16/length8 token format.  Parameters that can change
    compressed size are therefore part of the concrete action identity.
    """

    preset = native_lz_preset(algorithm_key)
    return (
        ExactByteActionSpec(
            name=f"op.match_window_model.native_lz.window{preset.window}",
            operator_id="match_window_model",
            implementation_id=f"native_lz_window_{preset.window}",
            implementation_source="cpp/src/native_lz_action_stages.cpp",
            implementation_scope=f"aligned-8 previous-match window of {preset.window} bytes",
        ),
        ExactByteActionSpec(
            name=f"op.lz_match.native_lz.aligned8.min{preset.min_match}.max{preset.max_match}",
            operator_id="lz_match",
            implementation_id=f"native_lz_aligned8_min{preset.min_match}_max{preset.max_match}",
            implementation_source="cpp/src/native_lz_action_stages.cpp",
            implementation_scope=(
                f"aligned 64-bit-key match search, min={preset.min_match}, "
                f"max={preset.max_match}"
            ),
        ),
        ExactByteActionSpec(
            name="op.match_parse_policy.native_lz.aligned8_greedy",
            operator_id="match_parse_policy",
            implementation_id="native_lz_aligned8_greedy",
            implementation_source="cpp/src/native_lz_action_stages.cpp",
            implementation_scope="greedy parse over aligned-8 match candidates",
        ),
        ExactByteActionSpec(
            name=f"op.match_token_format.native_lz.ctrl8_offset16_len8.bias{preset.min_match}",
            operator_id="match_token_format",
            implementation_id=f"native_lz_ctrl8_offset16_len8_bias{preset.min_match}",
            implementation_source="cpp/src/native_lz_action_stages.cpp",
            implementation_scope=(
                "eight-token control byte, little-endian uint16 offset, "
                f"uint8 length biased by {preset.min_match}"
            ),
        ),
    )


def exact_native_lz_action_registry(algorithm_keys: Iterable[str]) -> tuple[ExactByteActionSpec, ...]:
    """Deduplicate exact stage variants by operator name + implementation."""

    actions = {
        action.name: action
        for algorithm_key in algorithm_keys
        for action in exact_native_lz_action_pipeline(algorithm_key)
    }
    return tuple(actions[name] for name in sorted(actions))


_NATIVE_STAGE_PACKET_HEADER = struct.Struct("<4sBIIIQ")
_NATIVE_STAGE_TOKEN_COUNT = struct.Struct("<Q")
_NATIVE_STAGE_MAGIC = b"NLS1"
_NATIVE_STAGE_WINDOW = 1
_NATIVE_STAGE_MATCH = 2
_NATIVE_STAGE_PARSE = 3
_NATIVE_WINDOW_PREFIX = "op.match_window_model.native_lz.window"
_NATIVE_MATCH_PREFIX = "op.lz_match.native_lz.aligned8.min"
_NATIVE_PARSE_ACTION = "op.match_parse_policy.native_lz.aligned8_greedy"
_NATIVE_TOKEN_PREFIX = "op.match_token_format.native_lz.ctrl8_offset16_len8.bias"


def exact_native_lz_stage_role(action_name: str) -> str:
    """Return the normalized role of one benchmark-86 native-LZ action."""

    if action_name.startswith(_NATIVE_WINDOW_PREFIX):
        return "window"
    if action_name.startswith(_NATIVE_MATCH_PREFIX):
        return "match"
    if action_name == _NATIVE_PARSE_ACTION:
        return "parse"
    if action_name.startswith(_NATIVE_TOKEN_PREFIX):
        return "token_format"
    raise ValueError(f"not a native-LZ stage action: {action_name}")


def _native_window_from_action(action_name: str) -> int:
    try:
        return int(action_name.removeprefix(_NATIVE_WINDOW_PREFIX))
    except ValueError as exc:
        raise ValueError(f"invalid native-LZ window action: {action_name}") from exc


def _native_match_bounds_from_action(action_name: str) -> tuple[int, int]:
    try:
        suffix = action_name.removeprefix(_NATIVE_MATCH_PREFIX)
        minimum, maximum = suffix.split(".max", 1)
        return int(minimum), int(maximum)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid native-LZ match action: {action_name}") from exc


def _native_bias_from_action(action_name: str) -> int:
    try:
        return int(action_name.removeprefix(_NATIVE_TOKEN_PREFIX))
    except ValueError as exc:
        raise ValueError(f"invalid native-LZ token-format action: {action_name}") from exc


def _pack_native_stage_packet(
    role: int,
    *,
    window: int,
    min_match: int,
    max_match: int,
    raw_size: int,
    body: bytes,
) -> bytes:
    return _NATIVE_STAGE_PACKET_HEADER.pack(
        _NATIVE_STAGE_MAGIC,
        role,
        window,
        min_match,
        max_match,
        raw_size,
    ) + bytes(body)


def _unpack_native_stage_packet(data: bytes) -> tuple[int, int, int, int, int, bytes]:
    raw = bytes(data)
    if len(raw) < _NATIVE_STAGE_PACKET_HEADER.size:
        raise ValueError("native-LZ stage packet is truncated")
    magic, role, window, min_match, max_match, raw_size = (
        _NATIVE_STAGE_PACKET_HEADER.unpack_from(raw)
    )
    if magic != _NATIVE_STAGE_MAGIC or role not in {
        _NATIVE_STAGE_WINDOW,
        _NATIVE_STAGE_MATCH,
        _NATIVE_STAGE_PARSE,
    }:
        raise ValueError("invalid native-LZ stage packet")
    return (
        role,
        window,
        min_match,
        max_match,
        raw_size,
        raw[_NATIVE_STAGE_PACKET_HEADER.size :],
    )


def _serialize_native_tokens(tokens: Sequence[NativeLzToken]) -> bytes:
    out = bytearray(_NATIVE_STAGE_TOKEN_COUNT.pack(len(tokens)))
    for token in tokens:
        if token.is_match:
            if not (0 < token.offset <= 0xFFFF and 0 < token.length <= 0xFFFF):
                raise ValueError("native-LZ token is outside the normalized packet range")
            out.append(1)
            out.extend(struct.pack("<HH", token.offset, token.length))
        else:
            if token.literal is None or not 0 <= token.literal <= 0xFF:
                raise ValueError("native-LZ literal is outside the byte range")
            out.extend((0, token.literal))
    return bytes(out)


def _deserialize_native_tokens(data: bytes) -> tuple[NativeLzToken, ...]:
    raw = bytes(data)
    if len(raw) < _NATIVE_STAGE_TOKEN_COUNT.size:
        raise ValueError("native-LZ token packet is truncated")
    token_count = _NATIVE_STAGE_TOKEN_COUNT.unpack_from(raw)[0]
    pos = _NATIVE_STAGE_TOKEN_COUNT.size
    tokens: list[NativeLzToken] = []
    for _ in range(token_count):
        if pos >= len(raw):
            raise ValueError("native-LZ token packet is truncated")
        tag = raw[pos]
        pos += 1
        if tag == 0:
            if pos >= len(raw):
                raise ValueError("native-LZ literal token is truncated")
            tokens.append(NativeLzToken(literal=raw[pos]))
            pos += 1
        elif tag == 1:
            if pos + 4 > len(raw):
                raise ValueError("native-LZ match token is truncated")
            offset, length = struct.unpack_from("<HH", raw, pos)
            pos += 4
            if offset == 0 or length == 0:
                raise ValueError("native-LZ match token has a zero field")
            tokens.append(NativeLzToken(offset=offset, length=length))
        else:
            raise ValueError(f"unknown native-LZ token tag: {tag}")
    if pos != len(raw):
        raise ValueError("native-LZ token packet has trailing bytes")
    return tuple(tokens)


def _expand_native_tokens(
    tokens: Sequence[NativeLzToken], expected_size: int
) -> bytes:
    return _cpp_action_run(
        "native_lz_action_stages_cli", "expand", _serialize_native_tokens(tokens), expected_size
    )


def _pack_native_wire(tokens: Sequence[NativeLzToken], min_match: int) -> bytes:
    return _cpp_action_run(
        "native_lz_action_stages_cli", "pack", _serialize_native_tokens(tokens), min_match
    )


def _unpack_native_wire(
    data: bytes, min_match: int, expected_size: int
) -> tuple[NativeLzToken, ...]:
    return _deserialize_native_tokens(_cpp_action_run(
        "native_lz_action_stages_cli", "unpack", data, min_match, expected_size
    ))


def native_lz_stage_packet_context(data: bytes) -> dict[str, int]:
    """Read the self-describing parameters of an intermediate stage packet."""

    role, window, min_match, max_match, raw_size, _body = (
        _unpack_native_stage_packet(data)
    )
    return {
        "packet_role": role,
        "window": window,
        "min_match": min_match,
        "max_match": max_match,
        "raw_size": raw_size,
    }


def encode_exact_native_lz_stage(action_name: str, data: bytes) -> bytes:
    """Execute the C++ NLS1 stage; Python only supplies action identity."""
    exact_native_lz_stage_role(action_name)
    return _cpp_action_run("native_lz_action_stages_cli", "forward", data, action_name)


def decode_exact_native_lz_stage(
    action_name: str,
    data: bytes,
    *,
    context: dict[str, int] | None = None,
) -> bytes:
    """Invert a C++ stage with only its recorded scalar metadata context."""
    role = exact_native_lz_stage_role(action_name)
    if role == "token_format":
        if context is None:
            raise ValueError("token-format inverse requires recorded parse-packet context")
        return _cpp_action_run(
            "native_lz_action_stages_cli", "inverse", data, action_name,
            *(int(context[key]) for key in ("window", "min_match", "max_match", "raw_size"))
        )
    return _cpp_action_run("native_lz_action_stages_cli", "inverse", data, action_name)


def encode_exact_native_lz_pipeline(data: bytes, algorithm_key: str) -> bytes:
    """Execute the split native-LZ stages using the benchmark's fixed fast path."""

    return compress_native_lz_family_staged_fast(bytes(data), algorithm_key)


EXACT_ATOMIC_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.entropy_huffman.project_huffman_bytes.v1",
        operator_id="entropy_huffman",
        implementation_id="project_huffman_bytes_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp",
        implementation_scope="complete order-0 byte Huffman operator including its frequency-table wire",
    ),
    ExactByteActionSpec(
        name="op.entropy_range.project_arithmetic_bytes.v1",
        operator_id="entropy_range",
        implementation_id="project_arithmetic_bytes_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp",
        implementation_scope="complete project byte arithmetic/range-coding operator",
    ),
    ExactByteActionSpec(
        name="op.entropy_fse.project_rans_bytes.v1",
        operator_id="entropy_fse",
        implementation_id="project_rans_bytes_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp",
        implementation_scope="complete project order-0 rANS byte operator",
    ),
    ExactByteActionSpec(
        name="op.dictionary.project_int64.v1",
        operator_id="dictionary",
        implementation_id="project_dictionary_int64_v1",
        implementation_source="backend.myapp.column_encodings.encode_dictionary_int",
        implementation_scope="project int64 dictionary encoder and exact wire",
    ),
    ExactByteActionSpec(
        name="op.dictionary.project_float64.v1",
        operator_id="dictionary",
        implementation_id="project_dictionary_float64_v1",
        implementation_source="backend.myapp.column_encodings.encode_dictionary_float",
        implementation_scope="project float64 dictionary encoder and exact wire",
    ),
    ExactByteActionSpec(
        name="op.run_length.tsfile_int64.v1",
        operator_id="run_length",
        implementation_id="tsfile_rle_int64_v1",
        implementation_source="backend.myapp.tsfile_encoding.p02.encode_rle_int64",
        implementation_scope="TsFile int64 run-length operator and exact wire",
    ),
)
EXACT_ATOMIC_ACTION_BY_NAME = {action.name: action for action in EXACT_ATOMIC_ACTIONS}

RLE_SCALING_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.scaling.tsfile_rle.cpp_v1",
        operator_id="scaling",
        implementation_id="tsfile_rle_scaling_cpp_v1",
        implementation_source="cpp/src/rle_exact_stages.cpp:scaling_forward/scaling_inverse",
        implementation_scope=(
            "TsFile-RLE decimal float64-to-int64 scaling with effective scale "
            "metadata and a bitwise-reversible supported domain"
        ),
    ),
)
RLE_SCALING_ACTION = RLE_SCALING_EXACT_ACTIONS[0].name
RLE_MAX_POINT_LIMIT = 18

TS2DIFF_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.scaling.ts_2diff.cpp_v1",
        operator_id="scaling",
        implementation_id="ts_2diff_scaling_cpp_v1",
        implementation_source="cpp/src/ts2diff_exact_stages.cpp:scaling_forward/scaling_inverse",
        implementation_scope="bitwise-reversible finite float64 decimal scaling with explicit max_point metadata",
    ),
    ExactByteActionSpec(
        name="op.delta.ts_2diff.cpp_v1",
        operator_id="delta",
        implementation_id="ts_2diff_delta_cpp_v1",
        implementation_source="cpp/src/ts2diff_exact_stages.cpp:delta_forward_*/delta_inverse_*",
        implementation_scope="TS_2DIFF block-128 previous-value delta with reversible intermediate wire",
    ),
    ExactByteActionSpec(
        name="op.frame_of_reference.ts_2diff.cpp_v1",
        operator_id="frame_of_reference",
        implementation_id="ts_2diff_frame_of_reference_cpp_v1",
        implementation_source="cpp/src/ts2diff_exact_stages.cpp:frame_of_reference_forward/inverse",
        implementation_scope="per-block minimum-delta FOR normalization with reversible intermediate wire",
    ),
    ExactByteActionSpec(
        name="op.bitpack_int.ts_2diff.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="ts_2diff_bitpack_int_cpp_v1",
        implementation_source="cpp/src/ts2diff_exact_stages.cpp:bitpack_forward/bitpack_inverse",
        implementation_scope="MSB-first block bitpack that emits the exact project TS_2DIFF public wire",
    ),
)
TS2DIFF_INT_ACTIONS = tuple(action.name for action in TS2DIFF_EXACT_ACTIONS[1:])
TS2DIFF_FLOAT_ACTIONS = tuple(action.name for action in TS2DIFF_EXACT_ACTIONS)
TS2DIFF_STAGE_CLI = REPO_ROOT / "cpp/build/ts2diff_exact_stages_cli"

SVB_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.delta.stream_vbyte.cpp_v1",
        operator_id="delta",
        implementation_id="stream_vbyte_delta_cpp_v1",
        implementation_source="cpp/src/svb_exact_stages.cpp:delta_forward/delta_inverse",
        implementation_scope="modular delta-1 over the low/high uint32 limbs of an int64 column",
    ),
    ExactByteActionSpec(
        name="op.integer_code.stream_vbyte.cpp_v1",
        operator_id="integer_code",
        implementation_id="stream_vbyte_integer_code_cpp_v1",
        implementation_source="cpp/src/svb_exact_stages.cpp:integer_code_forward/integer_code_inverse",
        implementation_scope="Stream-VByte key/data emission over pre-delta-coded limbs",
    ),
)
SVB_INT_ACTIONS = tuple(action.name for action in SVB_EXACT_ACTIONS)
SVB_STAGE_CLI = REPO_ROOT / "cpp/build/svb_exact_stages_cli"

SIMPLE8B_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.scaling.simple8b.cpp_v1",
        operator_id="scaling",
        implementation_id="simple8b_scaling_cpp_v1",
        implementation_source="cpp/src/simple8b_exact_stages.cpp:scaling_forward/scaling_inverse",
        implementation_scope="bitwise-reversible finite float64 decimal scaling for the Simple8b adapter",
    ),
    ExactByteActionSpec(
        name="op.bitpack_int.simple8b.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="simple8b_bitpack_int_cpp_v1",
        implementation_source="cpp/src/simple8b_exact_stages.cpp:bitpack_forward_*/bitpack_inverse_*",
        implementation_scope="Simple8b selector/word pack over int64 limbs with exact typed adapter wire",
    ),
)
SIMPLE8B_INT_ACTIONS = (SIMPLE8B_EXACT_ACTIONS[1].name,)
SIMPLE8B_FLOAT_ACTIONS = tuple(action.name for action in SIMPLE8B_EXACT_ACTIONS)
SIMPLE8B_STAGE_CLI = REPO_ROOT / "cpp/build/simple8b_exact_stages_cli"

RLBE_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.delta.rlbe.cpp_v1",
        operator_id="delta",
        implementation_id="rlbe_delta_cpp_v1",
        implementation_source="cpp/src/p04_exact_stages.cpp:rlbe_delta_forward/rlbe_delta_inverse",
        implementation_scope="RLBE block-10000 modular previous-value delta over int64 or float64 bit patterns",
    ),
    ExactByteActionSpec(
        name="op.run_length.rlbe.cpp_v1",
        operator_id="run_length",
        implementation_id="rlbe_run_length_cpp_v1",
        implementation_source="cpp/src/p04_exact_stages.cpp:rlbe_run_length_forward/rlbe_run_length_inverse",
        implementation_scope="RLBE zero-run plus Fibonacci bitstream stage consuming an RLBE delta packet",
    ),
)
RLBE_ACTIONS = tuple(action.name for action in RLBE_EXACT_ACTIONS)

SPRINTZ_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.delta.sprintz.cpp_v1",
        operator_id="delta",
        implementation_id="sprintz_delta_cpp_v1",
        implementation_source="cpp/src/p04_exact_stages.cpp:sprintz_predictor_forward/sprintz_predictor_inverse",
        implementation_scope="Sprintz LongFIRE adaptive predictor residual stage over int64 or float64 bit patterns",
    ),
    ExactByteActionSpec(
        name="op.zigzag.sprintz.cpp_v1",
        operator_id="zigzag",
        implementation_id="sprintz_zigzag_cpp_v1",
        implementation_source="cpp/src/p04_exact_stages.cpp:sprintz_zigzag_forward/sprintz_zigzag_inverse",
        implementation_scope="Sprintz modular signed-residual ZigZag stage consuming a predictor packet",
    ),
    ExactByteActionSpec(
        name="op.bitpack_int.sprintz.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="sprintz_bitpack_int_cpp_v1",
        implementation_source="cpp/src/p04_exact_stages.cpp:sprintz_bitpack_forward/sprintz_bitpack_inverse",
        implementation_scope="Sprintz block-8 bit-width headers, packed residuals, and exact raw-tail wire",
    ),
)
SPRINTZ_ACTIONS = tuple(action.name for action in SPRINTZ_EXACT_ACTIONS)
P04_EXACT_ACTIONS = RLBE_EXACT_ACTIONS + SPRINTZ_EXACT_ACTIONS
P04_EXACT_ACTIONS_BY_NAME = {action.name: action for action in P04_EXACT_ACTIONS}
P04_STAGE_CLI = REPO_ROOT / "cpp/build/p04_exact_stages_cli"

DBP_LEGACY_VARINT_ACTION = ExactByteActionSpec(
    name="op.bitpack_int.delta_binary_packed.cpp_v1",
    operator_id="bitpack_int",
    implementation_id="delta_binary_packed_bitpack_int_cpp_v1",
    implementation_source="cpp/src/dbp_exact_stages.cpp:varint_forward/varint_inverse",
    implementation_scope="frozen DBP adapter that writes already-ZigZagged residual codes as ULEB128",
)

DBP_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.scaling.delta_binary_packed.cpp_v1",
        operator_id="scaling",
        implementation_id="delta_binary_packed_scaling_cpp_v1",
        implementation_source="cpp/src/dbp_exact_stages.cpp:scaling_forward/scaling_inverse",
        implementation_scope="bitwise-reversible benchmark decimal scaling with typed scale metadata",
    ),
    ExactByteActionSpec(
        name="op.delta.delta_binary_packed.cpp_v1",
        operator_id="delta",
        implementation_id="delta_binary_packed_delta_cpp_v1",
        implementation_source="cpp/src/dbp_exact_stages.cpp:delta_forward_*/delta_inverse_*",
        implementation_scope="modular int64 previous-value delta over raw integers or a DBP scaling packet",
    ),
    ExactByteActionSpec(
        name="op.zigzag.delta_binary_packed.cpp_v1",
        operator_id="zigzag",
        implementation_id="delta_binary_packed_zigzag_cpp_v1",
        implementation_source="cpp/src/dbp_exact_stages.cpp:zigzag_forward/zigzag_inverse",
        implementation_scope="DBP signed-delta ZigZag stage preserving the first value",
    ),
    DBP_LEGACY_VARINT_ACTION,
)
ULEB128_RAW_ACTION = ExactByteActionSpec(
    name="op.integer_code.uleb128.reference_values.cpp_v2",
    operator_id="integer_code",
    implementation_id="uleb128_unsigned_raw_uint64_cpp_v2",
    implementation_source="cpp/src/unsigned_leb128_cli.cpp:raw-forward/raw-inverse",
    implementation_scope="pure unsigned LEB128 over a uint64 little-endian value stream; no implicit ZigZag",
)
ZIGZAG_RAW_ACTION = ExactByteActionSpec(
    name="op.zigzag.signed64.raw.cpp_v2",
    operator_id="zigzag",
    implementation_id="zigzag_signed64_raw_cpp_v2",
    implementation_source="cpp/src/pure_numeric_operator_v1.cpp:zigzag-forward/zigzag-inverse",
    implementation_scope="pure A046 int64-to-uint64 mapping over raw 64-bit lanes without a carrier codec",
)
ULEB128_DBP_ACTION = replace(
    DBP_LEGACY_VARINT_ACTION,
    name="op.integer_code.uleb128.dbp_residuals.cpp_v2",
    operator_id="integer_code",
    implementation_id="uleb128_unsigned_dbp_residual_adapter_cpp_v2",
    implementation_scope="unsigned LEB128 over DBP residual codes with the first value and scale carried unchanged",
)
ULEB128_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ULEB128_RAW_ACTION,
    ULEB128_DBP_ACTION,
)
REFERENCE_INTEGER_ACTIONS = (ZIGZAG_RAW_ACTION.name, ULEB128_RAW_ACTION.name)
DBP_INT_ACTIONS = tuple(action.name for action in DBP_EXACT_ACTIONS[1:3]) + (ULEB128_DBP_ACTION.name,)
DBP_FLOAT_ACTIONS = (DBP_EXACT_ACTIONS[0].name,) + DBP_INT_ACTIONS
DBP_LEGACY_INT_ACTIONS = tuple(action.name for action in DBP_EXACT_ACTIONS[1:])
DBP_LEGACY_FLOAT_ACTIONS = tuple(action.name for action in DBP_EXACT_ACTIONS)
DBP_ACTIONS_BY_NAME = {
    action.name: action for action in DBP_EXACT_ACTIONS + (ULEB128_DBP_ACTION,)
}
DBP_STAGE_CLI = REPO_ROOT / "cpp/build/dbp_exact_stages_cli"
UNSIGNED_LEB128_STAGE_CLI = REPO_ROOT / "cpp/build/unsigned_leb128_cli"
PURE_NUMERIC_STAGE_CLI = REPO_ROOT / "cpp/build/pure_numeric_v1/pure_numeric_operator_v1"


def _run_simple_stage_cli(binary: Path, command: str, data: bytes) -> bytes:
    if not binary.is_file():
        raise RuntimeError(f"build {binary.name} first: {binary}")
    result = subprocess.run(
        [str(binary), command], input=bytes(data), capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _run_unsigned_leb128_stage_cli(command: str, data: bytes) -> bytes:
    return _run_simple_stage_cli(UNSIGNED_LEB128_STAGE_CLI, command, data)


def _run_pure_numeric_stage_cli(command: str, data: bytes) -> bytes:
    return _run_simple_stage_cli(PURE_NUMERIC_STAGE_CLI, command, data)
SVB_FLOAT_ACTIONS = (DBP_EXACT_ACTIONS[0].name,) + SVB_INT_ACTIONS

GORILLA_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.xor.gorilla.cpp_v1",
        operator_id="xor",
        implementation_id="gorilla_xor_cpp_v1",
        implementation_source="cpp/src/gorilla_exact_stages.cpp:xor_forward_*/xor_inverse",
        implementation_scope="Gorilla previous-value XOR residual stage with typed sentinel metadata",
    ),
    ExactByteActionSpec(
        name="op.trailing_zero_pack.gorilla.cpp_v1",
        operator_id="trailing_zero_pack",
        implementation_id="gorilla_trailing_zero_pack_cpp_v1",
        implementation_source="cpp/src/gorilla_exact_stages.cpp:trailing_pack_forward/trailing_pack_inverse_*",
        implementation_scope="Gorilla stateful leading/trailing-zero control-bit pack and exact public wire",
    ),
)
GORILLA_ACTIONS = tuple(action.name for action in GORILLA_EXACT_ACTIONS)
GORILLA_ACTIONS_BY_NAME = {action.name: action for action in GORILLA_EXACT_ACTIONS}
GORILLA_STAGE_CLI = REPO_ROOT / "cpp/build/gorilla_exact_stages_cli"

CHIMP128_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.xor.chimp128.cpp_v1",
        operator_id="xor",
        implementation_id="chimp128_xor_cpp_v1",
        implementation_source="cpp/src/chimp128_exact_stages.cpp:xor_forward_*/xor_inverse",
        implementation_scope="Chimp128 low-bit-indexed 128-value history reference selection and XOR records",
    ),
    ExactByteActionSpec(
        name="op.trailing_zero_pack.chimp128.cpp_v1",
        operator_id="trailing_zero_pack",
        implementation_id="chimp128_trailing_zero_pack_cpp_v1",
        implementation_source="cpp/src/chimp128_exact_stages.cpp:trailing_pack_forward/trailing_pack_inverse_*",
        implementation_scope="Chimp128 four-case control metadata and leading/trailing-zero bitstream",
    ),
)
CHIMP128_ACTIONS = tuple(action.name for action in CHIMP128_EXACT_ACTIONS)
CHIMP128_ACTIONS_BY_NAME = {action.name: action for action in CHIMP128_EXACT_ACTIONS}
CHIMP128_STAGE_CLI = REPO_ROOT / "cpp/build/chimp128_exact_stages_cli"

PONGO_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.eraser.pongo.cpp_v1",
        operator_id="eraser",
        implementation_id="pongo_decimal_eraser_cpp_v1",
        implementation_source="cpp/src/pongo_codec.cpp:exact_actions::pongo::eraser_forward/eraser_inverse",
        implementation_scope="Pongo shortest-decimal reversible mantissa re-encoding with decimal-zero metadata",
    ),
    ExactByteActionSpec(
        name="op.xor.pongo.cpp_v1",
        operator_id="xor",
        implementation_id="pongo_previous_value_xor_cpp_v1",
        implementation_source="cpp/src/pongo_codec.cpp:exact_actions::pongo::xor_forward/xor_inverse",
        implementation_scope="Pongo previous-target XOR residual records preserving eraser metadata",
    ),
    ExactByteActionSpec(
        name="op.trailing_zero_pack.pongo.cpp_v1",
        operator_id="trailing_zero_pack",
        implementation_id="pongo_stateful_zero_window_pack_cpp_v1",
        implementation_source="cpp/src/pongo_codec.cpp:exact_actions::pongo::trailing_pack_forward/trailing_pack_inverse",
        implementation_scope="Pongo stateful leading/trailing-zero window bit packing producing the exact public wire",
    ),
)
PONGO_ACTIONS = tuple(action.name for action in PONGO_EXACT_ACTIONS)
PONGO_ACTIONS_BY_NAME = {action.name: action for action in PONGO_EXACT_ACTIONS}
PONGO_STAGE_CLI = REPO_ROOT / "cpp/build/pongo_exact_stages_cli"

SPDP_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.delta.spdp.cpp_v1",
        operator_id="delta",
        implementation_id="spdp_delta_cpp_v1",
        implementation_source="cpp/src/spdp_exact_stages.cpp:delta_forward/delta_inverse",
        implementation_scope="SPDP second-order modular delta over little-endian 64-bit words",
    ),
    ExactByteActionSpec(
        name="op.block_transform.spdp.cpp_v1",
        operator_id="block_transform",
        implementation_id="spdp_block_transform_cpp_v1",
        implementation_source="cpp/src/spdp_exact_stages.cpp:block_transform_forward/block_transform_inverse",
        implementation_scope="SPDP eight-lane byte transpose followed by cross-lane byte delta",
    ),
    ExactByteActionSpec(
        name="op.entropy_special.spdp.cpp_v1",
        operator_id="entropy_special",
        implementation_id="spdp_entropy_special_cpp_v1",
        implementation_source="cpp/src/spdp_exact_stages.cpp:entropy_forward/entropy_inverse",
        implementation_scope="SPDP level-5 history-table repeat/literal byte coder producing the public wire",
    ),
)
SPDP_ACTIONS = tuple(action.name for action in SPDP_EXACT_ACTIONS)
SPDP_ACTIONS_BY_NAME = {action.name: action for action in SPDP_EXACT_ACTIONS}
SPDP_STAGE_CLI = REPO_ROOT / "cpp/build/spdp_exact_stages_cli"

FPC_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.xor.fpc.cpp_v1",
        operator_id="xor",
        implementation_id="fpc_xor_cpp_v1",
        implementation_source="cpp/src/fpc_exact_stages.cpp:predictor_xor_forward/predictor_xor_inverse",
        implementation_scope="FPC stateful FCM/dFCM predictor selection and XOR residual records",
    ),
    ExactByteActionSpec(
        name="op.trailing_zero_pack.fpc.cpp_v1",
        operator_id="trailing_zero_pack",
        implementation_id="fpc_trailing_zero_pack_cpp_v1",
        implementation_source="cpp/src/fpc_exact_stages.cpp:trailing_pack_forward/trailing_pack_inverse",
        implementation_scope="FPC nibble byte-code table and little-endian residual packing",
    ),
)
FPC_ACTIONS = tuple(action.name for action in FPC_EXACT_ACTIONS)
FPC_ACTIONS_BY_NAME = {action.name: action for action in FPC_EXACT_ACTIONS}
FPC_STAGE_CLI = REPO_ROOT / "cpp/build/fpc_exact_stages_cli"

PFPC_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.xor.pfpc.cpp_v1",
        operator_id="xor",
        implementation_id="pfpc_xor_cpp_v1",
        implementation_source="cpp/src/pfpc_exact_stages.cpp:predictor_xor_forward/predictor_xor_inverse",
        implementation_scope="single-thread benchmark pFPC block-local FCM/dFCM predictor and XOR residuals",
    ),
    ExactByteActionSpec(
        name="op.trailing_zero_pack.pfpc.cpp_v1",
        operator_id="trailing_zero_pack",
        implementation_id="pfpc_trailing_zero_pack_cpp_v1",
        implementation_source="cpp/src/pfpc_exact_stages.cpp:trailing_pack_forward/trailing_pack_inverse",
        implementation_scope="single-thread pFPC block/chunk framing, nibble byte codes, and residual packing",
    ),
)
PFPC_ACTIONS = tuple(action.name for action in PFPC_EXACT_ACTIONS)
PFPC_ACTIONS_BY_NAME = {action.name: action for action in PFPC_EXACT_ACTIONS}
PFPC_STAGE_CLI = REPO_ROOT / "cpp/build/pfpc_exact_stages_cli"

# BOS has two source-level implementations in this repository.  The BOSA
# operator-ablation codec exposes the full five-stage chain requested by the
# paper taxonomy.  The benchmark-86 BOS64 adapter is a different two-stage
# delta/ZigZag + bit-pack wire and must remain a separate implementation.
BOS_BOSA_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.delta.bos.cpp_v1",
        operator_id="delta",
        implementation_id="bos_bosa_delta_cpp_v1",
        implementation_source="cpp/src/bos_exact_stages.cpp:delta_forward/delta_inverse",
        implementation_scope="BOSA modular previous-value delta with a reversible stage packet",
    ),
    ExactByteActionSpec(
        name="op.frame_of_reference.bos.cpp_v1",
        operator_id="frame_of_reference",
        implementation_id="bos_bosa_first_for_cpp_v1",
        implementation_source="cpp/src/bos_exact_stages.cpp:first_for_forward/first_for_inverse",
        implementation_scope="BOSA first FOR transform using the minimum signed delta as source_base",
    ),
    ExactByteActionSpec(
        name="op.outlier_separation.bos.cpp_v1",
        operator_id="outlier_separation",
        implementation_id="bos_bosa_outlier_separation_cpp_v1",
        implementation_source="cpp/src/bos_exact_stages.cpp:outlier_forward/outlier_inverse",
        implementation_scope="BOSA normal-width selection, exception bitmap, and exception side stream",
    ),
    ExactByteActionSpec(
        name="op.frame_of_reference.bos.post_outlier.cpp_v1",
        operator_id="frame_of_reference",
        implementation_id="bos_bosa_second_for_cpp_v1",
        implementation_source="cpp/src/bos_exact_stages.cpp:second_for_forward/second_for_inverse",
        implementation_scope="BOSA second FOR transform subtracting the selected normal interval base",
    ),
    ExactByteActionSpec(
        name="op.bitpack_int.bos.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="bos_bosa_bitpack_cpp_v1",
        implementation_source="cpp/src/bos_exact_stages.cpp:bitpack_forward/bitpack_inverse",
        implementation_scope="BOSA bitmap, fixed-width normal stream, exceptions, and exact BOSA public wire",
    ),
)
BOS_BOSA_ACTIONS = tuple(action.name for action in BOS_BOSA_EXACT_ACTIONS)

BOS64_LEGACY_DELTA_ACTION = ExactByteActionSpec(
    name="op.delta.bos64_adapter.cpp_v1",
    operator_id="delta",
    implementation_id="bos64_delta_zigzag_cpp_v1",
    implementation_source="cpp/src/bos_exact_stages.cpp:adapter_delta_forward*/adapter_delta_inverse*",
    implementation_scope="frozen BOS64 adapter that combines modular previous-value delta and ZigZag",
)
BOS64_BITPACK_ACTION = ExactByteActionSpec(
    name="op.bitpack_int.bos64_adapter.cpp_v1",
    operator_id="bitpack_int",
    implementation_id="bos64_bitpack_cpp_v1",
    implementation_source="cpp/src/bos_exact_stages.cpp:adapter_bitpack_forward/adapter_bitpack_inverse",
    implementation_scope="benchmark BOS64 big-endian block header and MSB-first fixed-width bit-pack public wire",
)
BOS64_LEGACY_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    BOS64_LEGACY_DELTA_ACTION,
    BOS64_BITPACK_ACTION,
)
BOS64_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.delta.bos64_signed_residual.cpp_v2",
        operator_id="delta",
        implementation_id="bos64_modular_delta_cpp_v2",
        implementation_source="cpp/src/bos_exact_stages.cpp:adapter_plain_delta_forward*/adapter_plain_delta_inverse*",
        implementation_scope="BOS64 modular previous-value delta producing signed residuals without ZigZag",
    ),
    ExactByteActionSpec(
        name="op.zigzag.bos64_residual.cpp_v2",
        operator_id="zigzag",
        implementation_id="zigzag_signed64_bos64_residual_adapter_cpp_v2",
        implementation_source="cpp/src/bos_exact_stages.cpp:adapter_zigzag_forward*/adapter_zigzag_inverse*",
        implementation_scope="A046 ZigZag over BOS64 signed residual positions; first value and scale bypass",
    ),
    BOS64_BITPACK_ACTION,
)
BOS64_INT_ACTIONS = tuple(action.name for action in BOS64_EXACT_ACTIONS)
BOS64_FLOAT_ACTIONS = (DBP_EXACT_ACTIONS[0].name,) + BOS64_INT_ACTIONS
BOS64_LEGACY_INT_ACTIONS = tuple(action.name for action in BOS64_LEGACY_EXACT_ACTIONS)
BOS64_LEGACY_FLOAT_ACTIONS = (DBP_EXACT_ACTIONS[0].name,) + BOS64_LEGACY_INT_ACTIONS
BOS_ACTIONS_BY_NAME = {
    action.name: action
    for action in BOS_BOSA_EXACT_ACTIONS + BOS64_LEGACY_EXACT_ACTIONS + BOS64_EXACT_ACTIONS
}
BOS_STAGE_CLI = REPO_ROOT / "cpp/build/bos_exact_stages_cli"

FASTPFOR_COMPOSITE_HELPER_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.bitpack_int.fastpfor.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="lemire_fastpfor8_variablebyte_int64_limbs_cpp_v1",
        implementation_source="cpp/src/fastpfor_exact_stages.cpp:adapter_forward_*/adapter_inverse_*",
        implementation_scope="Lemire CompositeCodec<FastPFor<8>,VariableByte> over low/high int64 limbs",
    ),
)
FASTPFOR_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.bitpack_int.fastpfor_pfor.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="lemire_fastpfor8_prefix_int64_limbs_cpp_v1",
        implementation_source="cpp/src/fastpfor_exact_stages.cpp:pfor_stage_forward_*/pfor_stage_inverse_*",
        implementation_scope="Lemire FastPFor<8> over each complete 256-value low/high int64-limb prefix",
    ),
    ExactByteActionSpec(
        name="op.integer_code.fastpfor_variablebyte.cpp_v1",
        operator_id="integer_code",
        implementation_id="lemire_variablebyte_tail_int64_limbs_cpp_v1",
        implementation_source="cpp/src/fastpfor_exact_stages.cpp:variable_byte_forward/variable_byte_inverse",
        implementation_scope="Lemire VariableByte over each residual low/high int64-limb tail plus exact adapter wire",
    ),
)
FASTPFOR_INT_ACTIONS = tuple(action.name for action in FASTPFOR_EXACT_ACTIONS)
FASTPFOR_FLOAT_ACTIONS = (DBP_EXACT_ACTIONS[0].name,) + FASTPFOR_INT_ACTIONS
FASTPFOR_ACTIONS_BY_NAME = {
    action.name: action
    for action in FASTPFOR_COMPOSITE_HELPER_ACTIONS + FASTPFOR_EXACT_ACTIONS
}
FASTPFOR_STAGE_CLI = REPO_ROOT / "cpp/build/fastpfor_exact_stages_cli"

# Elf and Elf+ share one byte-identical eraser implementation but have
# different XOR state machines.  Keeping only one eraser action applies the
# operator-name + implementation identity rule instead of manufacturing an
# Elf+-scoped duplicate.
ELF_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.eraser.elf.cpp_v1",
        operator_id="eraser",
        implementation_id="elf32_shared_eraser_cpp_v1",
        implementation_source="cpp/src/elf_float_codec.cpp:eraser_forward_double_bits/eraser_inverse_double_bits",
        implementation_scope="shared Elf/Elf+ float32 decimal eraser records on the finite, bitwise-lossless float64 domain",
    ),
    ExactByteActionSpec(
        name="op.xor.elf.cpp_v1",
        operator_id="xor",
        implementation_id="elf32_chimp_xor_cpp_v1",
        implementation_source="cpp/src/elf_float_codec.cpp:elf_xor_forward/elf_xor_inverse",
        implementation_scope="Elf-on-Chimp32 XOR state machine and exact ELF1 chunk wire",
    ),
    ExactByteActionSpec(
        name="op.xor.elf_plus.cpp_v1",
        operator_id="xor",
        implementation_id="elf_plus32_xor_cpp_v1",
        implementation_source="cpp/src/elf_float_codec.cpp:elf_plus_xor_forward/elf_plus_xor_inverse",
        implementation_scope="Elf+ XOR state machine and exact ELF1 chunk wire",
    ),
)
ELF_ACTIONS = (ELF_EXACT_ACTIONS[0].name, ELF_EXACT_ACTIONS[1].name)
ELF_PLUS_ACTIONS = (ELF_EXACT_ACTIONS[0].name, ELF_EXACT_ACTIONS[2].name)
ELF_ACTIONS_BY_NAME = {action.name: action for action in ELF_EXACT_ACTIONS}
ELF_STAGE_CLI = REPO_ROOT / "cpp/build/elf_exact_stages_cli"

CAMEL_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.eraser.camel.cpp_v1",
        operator_id="eraser",
        implementation_id="camel_decimal_eraser_cpp_v1",
        implementation_source="cpp/src/camel_exact_stages.cpp:eraser_forward/eraser_inverse",
        implementation_scope="Camel decimal-path selection with exact literal fallback and reversible eraser records",
    ),
    ExactByteActionSpec(
        name="op.xor.camel.cpp_v1",
        operator_id="xor",
        implementation_id="camel_decimal_xor_cpp_v1",
        implementation_source="cpp/src/camel_exact_stages.cpp:xor_forward/xor_inverse",
        implementation_scope="Camel literal XOR and decimal integer-delta records before bitstream materialization",
    ),
    ExactByteActionSpec(
        name="op.bitstream_pack.camel.cpp_v1",
        operator_id="bitstream_pack",
        implementation_id="camel_control_bitstream_cpp_v1",
        implementation_source="cpp/src/camel_exact_stages.cpp:bitstream_pack_forward/bitstream_pack_inverse",
        implementation_scope="Camel decimal control coding, chunk framing, and exact CAM1 public wire",
    ),
)
CAMEL_ACTIONS = tuple(action.name for action in CAMEL_EXACT_ACTIONS)
CAMEL_ACTIONS_BY_NAME = {action.name: action for action in CAMEL_EXACT_ACTIONS}
CAMEL_STAGE_CLI = REPO_ROOT / "cpp/build/camel_exact_stages_cli"

SNAPPY_LEGACY_INTEGER_ACTION = ExactByteActionSpec(
    name="op.integer_code.snappy.cpp_v1",
    operator_id="integer_code",
    implementation_id="snappy_uncompressed_length_varint_cpp_v1",
    implementation_source="cpp/src/snappy_exact_stages.cpp:integer_code_forward/integer_code_inverse",
    implementation_scope="frozen Snappy adapter for its uint32 uncompressed-length ULEB128 field",
)
ULEB128_SNAPPY_ACTION = replace(
    SNAPPY_LEGACY_INTEGER_ACTION,
    name="op.integer_code.uleb128.snappy_raw_size.cpp_v2",
    implementation_id="uleb128_unsigned_snappy_raw_size_adapter_cpp_v2",
    implementation_scope="unsigned ULEB128 over the scalar Snappy uint32 raw length; token bytes bypass",
)
SNAPPY_LEGACY_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.block_state_control.snappy_matchfinder_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="snappy_matchfinder_remaining_core_cpp_v1",
        implementation_source="cpp/src/snappy_exact_stages.cpp:lz_match_forward/lz_match_inverse",
        implementation_scope="remaining upstream Snappy match finder and parse control exported as reversible semantic literal/copy records",
    ),
    ExactByteActionSpec(
        name="op.match_token_format.snappy.cpp_v1",
        operator_id="match_token_format",
        implementation_id="snappy_tag_format_cpp_v1",
        implementation_source="cpp/src/snappy_exact_stages.cpp:token_format_forward/token_format_inverse",
        implementation_scope="Snappy literal/COPY_1/COPY_2 tag materialization without the length prefix",
    ),
    SNAPPY_LEGACY_INTEGER_ACTION,
)
SNAPPY_EXACT_ACTIONS = SNAPPY_LEGACY_EXACT_ACTIONS[:2] + (ULEB128_SNAPPY_ACTION,)
SNAPPY_ACTIONS = tuple(action.name for action in SNAPPY_EXACT_ACTIONS)
SNAPPY_FUSED_ACTIONS = tuple(action.name for action in SNAPPY_LEGACY_EXACT_ACTIONS)
SNAPPY_ACTIONS_BY_NAME = {
    action.name: action for action in SNAPPY_LEGACY_EXACT_ACTIONS + SNAPPY_EXACT_ACTIONS
}
SNAPPY_STAGE_CLI = REPO_ROOT / "cpp/build/snappy_exact_stages_cli"

LZ4_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", operator_id="block_state_control",
        implementation_id="lz4_frame_matchfinder_remaining_core_cpp_v1",
        implementation_source="cpp/src/lz4_exact_stages.cpp:lz_match_forward/lz_match_inverse",
        implementation_scope="remaining upstream LZ4 frame matcher and block decisions exported as reversible per-block literal/copy records",
    ),
    ExactByteActionSpec(
        name="op.match_token_format.lz4.cpp_v1", operator_id="match_token_format",
        implementation_id="lz4_token_nibbles_cpp_v1",
        implementation_source="cpp/src/lz4_exact_stages.cpp:token_format_forward/token_format_inverse",
        implementation_scope="LZ4 token nibbles, offsets, and unencoded extended-length integers",
    ),
    ExactByteActionSpec(
        name="op.integer_code.lz4.cpp_v1", operator_id="integer_code",
        implementation_id="lz4_length_extension_cpp_v1",
        implementation_source="cpp/src/lz4_exact_stages.cpp:integer_code_forward/integer_code_inverse",
        implementation_scope="LZ4 repeated-255 extended-length materialization producing raw block bytes",
    ),
    ExactByteActionSpec(
        name="op.container_frame.lz4.cpp_v1", operator_id="container_frame",
        implementation_id="lz4_frame_v1_64k_cpp_v1",
        implementation_source="cpp/src/lz4_exact_stages.cpp:container_forward/container_inverse",
        implementation_scope="LZ4 frame descriptor, XXH32 header checksum, block headers, and end mark",
    ),
)
LZ4_ACTIONS = tuple(action.name for action in LZ4_EXACT_ACTIONS)
LZ4_ACTIONS_BY_NAME = {action.name: action for action in LZ4_EXACT_ACTIONS}
LZ4_STAGE_CLI = REPO_ROOT / "cpp/build/lz4_exact_stages_cli"

DEFLATE_CONTAINER_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.container_frame.gzip.cpp_v1",
        operator_id="container_frame",
        implementation_id="gzip_rfc1952_frame_cpp_v1",
        implementation_source="cpp/src/deflate_container_stages.cpp:gzip_forward/gzip_inverse",
        implementation_scope="Exact reversible RFC 1952 wrapper split; the enclosed raw DEFLATE stream is not relabelled as an LZ stage",
    ),
    ExactByteActionSpec(
        name="op.container_frame.zlib.cpp_v1",
        operator_id="container_frame",
        implementation_id="zlib_rfc1950_frame_cpp_v1",
        implementation_source="cpp/src/deflate_container_stages.cpp:zlib_forward/zlib_inverse",
        implementation_scope="Exact reversible RFC 1950 wrapper split; the enclosed raw DEFLATE stream is not relabelled as an LZ stage",
    ),
)
GZIP_CONTAINER_ACTIONS = (DEFLATE_CONTAINER_EXACT_ACTIONS[0].name,)
ZLIB_CONTAINER_ACTIONS = (DEFLATE_CONTAINER_EXACT_ACTIONS[1].name,)
DEFLATE_CONTAINER_ACTIONS_BY_NAME = {
    action.name: action for action in DEFLATE_CONTAINER_EXACT_ACTIONS
}
DEFLATE_CONTAINER_STAGE_CLI = REPO_ROOT / "cpp/build/deflate_container_stages_cli"

XZ_CONTAINER_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.container_frame.xz.cpp_v1",
        operator_id="container_frame",
        implementation_id="xz_stream_header_footer_cpp_v1",
        implementation_source="cpp/src/xz_container_stage.cpp:forward/inverse",
        implementation_scope="Exact reversible XZ stream header/footer split; block headers, LZMA2, range coding, and index stay in the core",
    ),
)
XZ_CONTAINER_ACTIONS = (XZ_CONTAINER_EXACT_ACTIONS[0].name,)
XZ_CONTAINER_ACTIONS_BY_NAME = {action.name: action for action in XZ_CONTAINER_EXACT_ACTIONS}
XZ_CONTAINER_STAGE_CLI = REPO_ROOT / "cpp/build/xz_container_stage_cli"

FPZIP_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.delta.fpzip.cpp_v1",
        operator_id="delta",
        implementation_id="fpzip_lorenzo1d_pcmap_residual_cpp_v1",
        implementation_source="cpp/src/fpzip_exact_stages.cpp:predictor_forward/predictor_inverse",
        implementation_scope="FPZIP lossless 1D double Lorenzo predictor and PCmap residual stream",
    ),
    ExactByteActionSpec(
        name="op.entropy_range.fpzip.cpp_v1",
        operator_id="entropy_range",
        implementation_id="fpzip_rcqsmodel_range_cpp_v1",
        implementation_source="cpp/src/fpzip_exact_stages.cpp:range_forward/range_inverse",
        implementation_scope="FPZIP adaptive RCqsmodel range coder plus exact lossless 1D double header",
    ),
)
FPZIP_ACTIONS = tuple(action.name for action in FPZIP_EXACT_ACTIONS)
FPZIP_ACTIONS_BY_NAME = {action.name: action for action in FPZIP_EXACT_ACTIONS}
FPZIP_STAGE_CLI = REPO_ROOT / "cpp/build/fpzip_exact_stages_cli"

# Source-level specialist stages added after the historical 194-record audit.
# LECO and BUFF end in the byte-identical public wire emitted by the project's
# current source implementation.  ALP and REGER keep their independently
# reversible explicit prefixes, then append one method-scoped residual core so
# the complete chain emits the canonical production wire byte-for-byte.
LECO_LEGACY_PREDICTOR_ACTION = ExactByteActionSpec(
    name="op.regression_predictor.leco.cpp_v1",
    operator_id="regression_predictor",
    implementation_id="leco_linear_model_residual_cpp_v1",
    implementation_source="cpp/src/learning_codec_exact_stages.cpp:leco_autoregression_forward/leco_autoregression_inverse",
    implementation_scope="frozen LECO adapter combining a fitted position model and ZigZag residuals",
)
LECO_BITPACK_ACTION = ExactByteActionSpec(
    name="op.bitpack_int.leco.cpp_v1",
    operator_id="bitpack_int",
    implementation_id="leco_residual_bitpack_cpp_v1",
    implementation_source="cpp/src/learning_codec_exact_stages.cpp:leco_bitpack_forward/leco_bitpack_inverse",
    implementation_scope="LECO maximum-value-width residual packing that emits the public wire",
)
LECO_LEGACY_EXACT_ACTIONS = (LECO_LEGACY_PREDICTOR_ACTION, LECO_BITPACK_ACTION)
LECO_SOURCE_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.regression_predictor.leco_linear.cpp_v2",
        operator_id="regression_predictor",
        implementation_id="leco_linear_model_signed_residual_cpp_v2",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:leco_regression_forward/leco_regression_inverse",
        implementation_scope="A016 LECO blockwise position regression producing signed residuals without ZigZag",
    ),
    ExactByteActionSpec(
        name="op.zigzag.leco_residual.cpp_v2",
        operator_id="zigzag",
        implementation_id="zigzag_signed64_leco_residual_adapter_cpp_v2",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:leco_zigzag_forward/leco_zigzag_inverse",
        implementation_scope="A046 ZigZag over LECO signed residuals; fitted float64 parameters bypass",
    ),
    LECO_BITPACK_ACTION,
)
LECO_ACTIONS = tuple(action.name for action in LECO_SOURCE_EXACT_ACTIONS)
LECO_LEGACY_ACTIONS = tuple(action.name for action in LECO_LEGACY_EXACT_ACTIONS)

ALP_REFERENCE_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.scaling.alp.cpp_v1",
        operator_id="scaling",
        implementation_id="alp_exact_decimal_scaling_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:alp_scaling_forward/alp_scaling_inverse",
        implementation_scope="ALP-style exact decimal scaling with raw-bit fallback",
    ),
    ExactByteActionSpec(
        name="op.frame_of_reference.alp.cpp_v1",
        operator_id="frame_of_reference",
        implementation_id="alp_for_normalization_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:alp_for_forward/alp_for_inverse",
        implementation_scope="ALP main-path frame-of-reference normalization",
    ),
    ExactByteActionSpec(
        name="op.bitpack_int.alp.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="alp_for_bitpack_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:alp_bitpack_forward/alp_bitpack_inverse",
        implementation_scope="ALP main-path FOR residual bit packing; source-reference wire",
    ),
    ExactByteActionSpec(
        name="op.dictionary.alp_rd.cpp_v1",
        operator_id="dictionary",
        implementation_id="alp_rd_dictionary_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:alp_dictionary_forward/alp_dictionary_inverse",
        implementation_scope="ALP-RD dictionary alternative; sibling to, not a serial member of, the ALP main path",
    ),
)
ALP_DICTIONARY_ACTIONS = (ALP_REFERENCE_ACTIONS[3].name,)

REGER_REORDER_ACTION = ExactByteActionSpec(
    name="op.value_reorder.reger.cpp_v1",
    operator_id="value_reorder",
    implementation_id="reger_stable_value_reorder_cpp_v1",
    implementation_source="cpp/src/learning_codec_exact_stages.cpp:reger_reorder_forward/reger_reorder_inverse",
    implementation_scope="reversible one-series REGER-style stable reorder with inverse permutation",
)
REGER_LEGACY_PREDICTOR_ACTION = ExactByteActionSpec(
    name="op.regression_predictor.reger.cpp_v1",
    operator_id="regression_predictor",
    implementation_id="reger_ar1_residual_cpp_v1",
    implementation_source="cpp/src/learning_codec_exact_stages.cpp:reger_autoregression_forward/reger_autoregression_inverse",
    implementation_scope="frozen REGER adapter combining modular previous-value delta and ZigZag",
)
REGER_DOWNSTREAM_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.frame_of_reference.reger.cpp_v1",
        operator_id="frame_of_reference",
        implementation_id="reger_residual_for_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:reger_for_forward/reger_for_inverse",
        implementation_scope="REGER residual frame-of-reference normalization",
    ),
    ExactByteActionSpec(
        name="op.bitpack_int.reger.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="reger_residual_bitpack_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:reger_bitpack_forward/reger_bitpack_inverse",
        implementation_scope="REGER maximum-value-width residual bit packing",
    ),
)
REGER_LEGACY_REFERENCE_ACTIONS = (
    REGER_REORDER_ACTION,
    REGER_LEGACY_PREDICTOR_ACTION,
    *REGER_DOWNSTREAM_ACTIONS,
)
REGER_REFERENCE_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    REGER_REORDER_ACTION,
    ExactByteActionSpec(
        name="op.delta.reger_signed_residual.cpp_v2",
        operator_id="delta",
        implementation_id="reger_modular_delta_cpp_v2",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:reger_delta_forward/reger_delta_inverse",
        implementation_scope="A028 modular previous-value delta over the reordered REGER values without ZigZag",
    ),
    ExactByteActionSpec(
        name="op.zigzag.reger_residual.cpp_v2",
        operator_id="zigzag",
        implementation_id="zigzag_signed64_reger_residual_adapter_cpp_v2",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:reger_zigzag_forward/reger_zigzag_inverse",
        implementation_scope="A046 ZigZag over REGER signed delta residuals; first value and permutation bypass",
    ),
    *REGER_DOWNSTREAM_ACTIONS,
)
LEARNING_CODEC_RESIDUAL_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.block_state_control.alp_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="alp_remaining_source_core_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:alp_residual_core_forward/alp_residual_core_inverse",
        implementation_scope="remaining ALP factor/exponent search, exception policy, FastPFor payload, and ALP2 block framing; appended prefix yields canonical ALP2 bytes",
    ),
    ExactByteActionSpec(
        name="op.block_state_control.reger_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="reger_remaining_source_core_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:reger_residual_core_forward/reger_residual_core_inverse",
        implementation_scope="remaining production REGER row/block/model control and public framing; appended prefix yields reger_codec bytes",
    ),
)
ALP_MAIN_ACTIONS = tuple(action.name for action in ALP_REFERENCE_ACTIONS[:3]) + (
    LEARNING_CODEC_RESIDUAL_ACTIONS[0].name,
)
REGER_ACTIONS = tuple(action.name for action in REGER_REFERENCE_ACTIONS) + (
    LEARNING_CODEC_RESIDUAL_ACTIONS[1].name,
)
REGER_FUSED_ACTIONS = tuple(action.name for action in REGER_LEGACY_REFERENCE_ACTIONS) + (
    LEARNING_CODEC_RESIDUAL_ACTIONS[1].name,
)

BUFF_SOURCE_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.scaling.buff.cpp_v1",
        operator_id="scaling",
        implementation_id="buff_exact_decimal_scaling_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:buff_scaling_forward/buff_scaling_inverse",
        implementation_scope="BUFF exact decimal scaling with raw-bit fallback",
    ),
    ExactByteActionSpec(
        name="op.block_transform.buff_column_split.cpp_v1",
        operator_id="block_transform",
        implementation_id="buff_byte_column_split_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:buff_column_split_forward/buff_column_split_inverse",
        implementation_scope="BUFF scaled-integer byte-column split with independently reversible packet",
    ),
    ExactByteActionSpec(
        name="op.bitpack_int.buff.cpp_v1",
        operator_id="bitpack_int",
        implementation_id="buff_fixed_width_bitpack_cpp_v1",
        implementation_source="cpp/src/learning_codec_exact_stages.cpp:buff_bitpack_forward/buff_bitpack_inverse",
        implementation_scope="BUFF fixed-width column bit packing that emits the current project public wire",
    ),
)
BUFF_ACTIONS = tuple(action.name for action in BUFF_SOURCE_EXACT_ACTIONS)  # frozen recipe
BUFF_FOR_ACTION = ExactByteActionSpec(
    name="op.frame_of_reference.buff.cpp_v2", operator_id="frame_of_reference",
    implementation_id="buff_minimum_for_profile_cpp_v2",
    implementation_source="cpp/src/learning_codec_exact_stages.cpp:buff_for_forward/buff_for_inverse",
    implementation_scope="A013 minimum FOR; base:int64 and offsets:uint64; scale fields bypass")
BUFF_PLANE_ACTION = ExactByteActionSpec(
    name="op.block_transform.buff_byte_plane_split.cpp_v2", operator_id="block_transform",
    implementation_id="buff_byte_plane_split_cpp_v2",
    implementation_source="cpp/src/learning_codec_exact_stages.cpp:buff_byte_plane_forward/buff_byte_plane_inverse",
    implementation_scope="A048 byte planes only, explicit bit_width; base and scale bypass unchanged")
BUFF_CURRENT_EXACT_ACTIONS = (BUFF_FOR_ACTION, BUFF_PLANE_ACTION)
BUFF_ATOMIC_ACTIONS = (BUFF_ACTIONS[0], BUFF_FOR_ACTION.name, BUFF_PLANE_ACTION.name, BUFF_ACTIONS[2])
LEARNING_CODEC_STAGE_ACTIONS_BY_NAME = {
    action.name: action
    for action in (
        LECO_SOURCE_EXACT_ACTIONS
        + ALP_REFERENCE_ACTIONS
        + REGER_REFERENCE_ACTIONS
        + LEARNING_CODEC_RESIDUAL_ACTIONS
        + BUFF_SOURCE_EXACT_ACTIONS
        + BUFF_CURRENT_EXACT_ACTIONS
    )
}
LEARNING_CODEC_STAGE_CLI = REPO_ROOT / "cpp/build/learning_codec_exact_stages_cli"

GENERAL_CODEC_REFERENCE_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.dictionary.brotli_static_small.cpp_reference_v1",
        operator_id="dictionary",
        implementation_id="brotli_static_small_dictionary_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:dictionary_forward/dictionary_inverse",
        implementation_scope="independent fixed-word dictionary reference; not the upstream Brotli dictionary wire",
    ),
    ExactByteActionSpec(
        name="op.lz_match.brotli_lazy65535.cpp_reference_v1",
        operator_id="lz_match",
        implementation_id="brotli_lazy_lz_window65535_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:lz_forward/lz_inverse",
        implementation_scope="independent lazy LZ semantic records using Brotli-shaped parameters; not upstream parity",
    ),
    ExactByteActionSpec(
        name="op.entropy_huffman.project_bytes.cpp_v1",
        operator_id="entropy_huffman",
        implementation_id="project_byte_huffman_cpp_v1",
        implementation_source="cpp/src/huffman_bytes.cpp:huffman_bytes::encode/decode",
        implementation_scope="project byte Huffman implementation shared by reference Brotli and Bzip2 chains",
    ),
    ExactByteActionSpec(
        name="op.lz_match.zstd_lazy65535.cpp_reference_v1",
        operator_id="lz_match",
        implementation_id="zstd_lazy_lz_window65535_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:lz_forward/lz_inverse",
        implementation_scope="independent lazy LZ semantic records; not ZSTD_generateSequences parity",
    ),
    ExactByteActionSpec(
        name="op.match_parse_policy.generic_packet.cpp_reference_v1",
        operator_id="match_parse_policy",
        implementation_id="zstd_policy_packet_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:parse_policy_forward/parse_policy_inverse",
        implementation_scope="reversible Zstd-reference policy boundary; upstream match generation remains fused",
    ),
    ExactByteActionSpec(
        name="op.literal_entropy.zstd_huffman.cpp_reference_v1",
        operator_id="literal_entropy",
        implementation_id="zstd_reference_literal_huffman_cpp_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:huffman_forward/huffman_inverse",
        implementation_scope="reference Huffman materialization over the policy packet; not upstream literal-section parity",
    ),
    ExactByteActionSpec(
        name="op.entropy_fse.zstd_rans.cpp_reference_v1",
        operator_id="entropy_fse",
        implementation_id="zstd_reference_rans_cpp_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:rans_forward/rans_inverse",
        implementation_scope="independent project rANS reference for the FSE-family slot; not upstream Zstd FSE wire",
    ),
    ExactByteActionSpec(
        name="op.block_transform.blosc_shuffle.typesize1.cpp_reference_v1",
        operator_id="block_transform",
        implementation_id="blosc_byte_shuffle_typesize1_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:shuffle_forward/shuffle_inverse",
        implementation_scope="byte shuffle with typesize=1 shared by Blosc and Blosc2 reference chains",
    ),
    ExactByteActionSpec(
        name="op.block_transform.blosc_shuffle.typesize4.cpp_reference_v1",
        operator_id="block_transform",
        implementation_id="blosc_byte_shuffle_typesize4_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:shuffle_forward/shuffle_inverse",
        implementation_scope="byte shuffle with typesize=4 shared by Blosc and Blosc2 reference chains",
    ),
    ExactByteActionSpec(
        name="op.block_transform.blosc_shuffle.typesize8.cpp_reference_v1",
        operator_id="block_transform",
        implementation_id="blosc_byte_shuffle_typesize8_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:shuffle_forward/shuffle_inverse",
        implementation_scope="byte shuffle with typesize=8 shared by Blosc and Blosc2 reference chains",
    ),
    ExactByteActionSpec(
        name="op.lz_match.blosclz.greedy.cpp_reference_v1",
        operator_id="lz_match",
        implementation_id="blosclz_greedy_lz_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:lz_forward/lz_inverse",
        implementation_scope="shared greedy semantic LZ reference for Blosc/Blosc2; not upstream blosclz wire",
    ),
    ExactByteActionSpec(
        name="op.match_token_format.blosclz.ctrl8.cpp_reference_v1",
        operator_id="match_token_format",
        implementation_id="blosclz_ctrl8_token_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:token_forward/token_inverse",
        implementation_scope="shared reversible ctrl8 reference token format for Blosc/Blosc2",
    ),
    ExactByteActionSpec(
        name="op.block_state_control.blosc1.cpp_reference_v1",
        operator_id="block_state_control",
        implementation_id="blosc1_reference_frame_cpp_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:frame_forward/frame_inverse",
        implementation_scope="reference Blosc1 block/state envelope; not upstream Blosc1 frame parity",
    ),
    ExactByteActionSpec(
        name="op.block_state_control.blosc2.cpp_reference_v1",
        operator_id="block_state_control",
        implementation_id="blosc2_reference_frame_cpp_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:frame_forward/frame_inverse",
        implementation_scope="reference Blosc2 block/state envelope; not upstream Blosc2 frame parity",
    ),
    ExactByteActionSpec(
        name="op.block_transform.bzip2_bwt.cpp_reference_v1",
        operator_id="block_transform",
        implementation_id="bzip2_cyclic_bwt_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:bwt_forward/bwt_inverse",
        implementation_scope="independent cyclic BWT reference stage; not upstream Bzip2 block wire",
    ),
    ExactByteActionSpec(
        name="op.symbol_recode.bzip2_mtf.cpp_reference_v1",
        operator_id="symbol_recode",
        implementation_id="bzip2_mtf_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:mtf_forward/mtf_inverse",
        implementation_scope="independent move-to-front symbol recoding stage",
    ),
    ExactByteActionSpec(
        name="op.run_length.bzip2_uint8.cpp_reference_v1",
        operator_id="run_length",
        implementation_id="bzip2_uint8_rle_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:rle_forward/rle_inverse",
        implementation_scope="integer-symbol-only uint8 run-length stage used after MTF; it never accepts float input",
    ),
    ExactByteActionSpec(
        name="op.lz_match.deflate_lazy32768.cpp_reference_v1",
        operator_id="lz_match",
        implementation_id="deflate_lazy_lz_window32768_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:lz_forward/lz_inverse",
        implementation_scope="independent DEFLATE-shaped lazy LZ reference; not zlib/gzip upstream wire parity",
    ),
    ExactByteActionSpec(
        name="op.lz_match.xz_lazy16777216.cpp_reference_v1",
        operator_id="lz_match",
        implementation_id="xz_lazy_lz_window16777216_cpp_reference_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:lz_forward/lz_inverse",
        implementation_scope="independent LZMA-shaped lazy LZ reference; not liblzma upstream wire parity",
    ),
    ExactByteActionSpec(
        name="op.entropy_range.project_bytes.cpp_v1",
        operator_id="entropy_range",
        implementation_id="project_static_range_bytes_cpp_v1",
        implementation_source="cpp/src/arithmetic_bytes.cpp:arithmetic_bytes::encode/decode",
        implementation_scope="independent project static range coder used by the XZ reference chain",
    ),
    ExactByteActionSpec(
        name="op.container_frame.gzip.cpp_reference_v1",
        operator_id="container_frame",
        implementation_id="gzip_reference_envelope_cpp_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:gzip_frame_forward/gzip_frame_inverse",
        implementation_scope="reference gzip-family envelope over the reference entropy stream; not RFC1952 parity",
    ),
    ExactByteActionSpec(
        name="op.container_frame.zlib.cpp_reference_v1",
        operator_id="container_frame",
        implementation_id="zlib_reference_envelope_cpp_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:zlib_frame_forward/zlib_frame_inverse",
        implementation_scope="reference zlib-family envelope over the reference entropy stream; not RFC1950 parity",
    ),
    ExactByteActionSpec(
        name="op.container_frame.xz.cpp_reference_v1",
        operator_id="container_frame",
        implementation_id="xz_reference_envelope_cpp_v1",
        implementation_source="cpp/src/general_codec_reference_stages.cpp:xz_frame_forward/xz_frame_inverse",
        implementation_scope="reference XZ-family envelope over the reference range stream; not liblzma XZ parity",
    ),
)

GENERAL_CODEC_RESIDUAL_ACTIONS: tuple[ExactByteActionSpec, ...] = (
    ExactByteActionSpec(
        name="op.block_state_control.brotli_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="brotli_quality6_remaining_core_cpp_v1",
        implementation_source="cpp/src/general_codec_residual_stages.cpp:brotli_forward/brotli_inverse",
        implementation_scope="remaining vendored Brotli quality-6 parsing, context modelling, entropy coding, and framing; appended prefix is byte-identical to BrotliEncoderCompress",
    ),
    ExactByteActionSpec(
        name="op.block_state_control.zstd_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="zstd_level9_remaining_core_cpp_v1",
        implementation_source="cpp/src/general_codec_residual_stages.cpp:zstd_forward/zstd_inverse",
        implementation_scope="remaining vendored Zstd level-9 sequence decisions, entropy tables, literals, and frame; appended prefix is byte-identical to ZSTD_compress",
    ),
    *tuple(
        ExactByteActionSpec(
            name=f"op.block_state_control.blosc1_residual_core_t{type_size}.cpp_v1",
            operator_id="block_state_control",
            implementation_id=f"blosc1_blosclz5_remaining_core_t{type_size}_cpp_v1",
            implementation_source="cpp/src/general_codec_residual_stages.cpp:blosc1_forward/blosc1_inverse",
            implementation_scope=f"remaining Blosc1 BLOSCLZ clevel-5 block/state control and exact buffer framing for typesize={type_size}",
        )
        for type_size in (1, 4, 8)
    ),
    *tuple(
        ExactByteActionSpec(
            name=f"op.block_state_control.blosc2_residual_core_t{type_size}.cpp_v1",
            operator_id="block_state_control",
            implementation_id=f"blosc2_blosclz5_remaining_core_t{type_size}_cpp_v1",
            implementation_source="cpp/src/general_codec_residual_stages.cpp:blosc2_forward/blosc2_inverse",
            implementation_scope=f"remaining Blosc2 BLOSCLZ clevel-5 context, block/state control, and exact chunk framing for typesize={type_size}",
        )
        for type_size in (1, 4, 8)
    ),
    ExactByteActionSpec(
        name="op.block_state_control.bzip2_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="bzip2_level4_remaining_core_cpp_v1",
        implementation_source="cpp/src/general_codec_residual_stages.cpp:bzip2_forward/bzip2_inverse",
        implementation_scope="remaining vendored Bzip2 level-4 block sorting, selectors, Huffman groups, CRC, and stream framing; appended prefix is byte-identical to BZ2_bzBuffToBuffCompress",
    ),
    ExactByteActionSpec(
        name="op.block_state_control.gzip_deflate_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="gzip_level6_deflate_remaining_core_cpp_v1",
        implementation_source="cpp/src/general_codec_residual_stages.cpp:gzip_forward/gzip_inverse",
        implementation_scope="remaining zlib level-6 DEFLATE decisions; emits the exact DFC1 packet consumed by the separate RFC1952 container action",
    ),
    ExactByteActionSpec(
        name="op.block_state_control.zlib_deflate_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="zlib_level9_deflate_remaining_core_cpp_v1",
        implementation_source="cpp/src/general_codec_residual_stages.cpp:zlib_forward/zlib_inverse",
        implementation_scope="remaining zlib level-9 DEFLATE decisions; emits the exact DFC1 packet consumed by the separate RFC1950 container action",
    ),
    ExactByteActionSpec(
        name="op.block_state_control.xz_lzma2_residual_core.cpp_v1",
        operator_id="block_state_control",
        implementation_id="xz_preset5_lzma2_remaining_core_cpp_v1",
        implementation_source="cpp/src/general_codec_residual_stages.cpp:xz_forward/xz_inverse",
        implementation_scope="remaining liblzma preset-5 LZMA2 state machine, range models, index, and checks; emits the exact XZC1 packet consumed by the separate container action",
    ),
)
GENERAL_CODEC_RESIDUAL_ACTIONS_BY_NAME = {
    action.name: action for action in GENERAL_CODEC_RESIDUAL_ACTIONS
}
GENERAL_CODEC_RESIDUAL_STAGE_CLI = REPO_ROOT / "cpp/build/general_codec_residual_stages_cli"

BROTLI_REFERENCE_ACTIONS = tuple(action.name for action in GENERAL_CODEC_REFERENCE_ACTIONS[:3])
ZSTD_REFERENCE_ACTIONS = tuple(action.name for action in GENERAL_CODEC_REFERENCE_ACTIONS[3:7])
BLOSC_SHUFFLE_ACTION_BY_KIND = {
    "bytes": GENERAL_CODEC_REFERENCE_ACTIONS[7].name,
    "float32": GENERAL_CODEC_REFERENCE_ACTIONS[8].name,
    "float64": GENERAL_CODEC_REFERENCE_ACTIONS[9].name,
    "int64": GENERAL_CODEC_REFERENCE_ACTIONS[9].name,
}
_BLOSCLZ_SHARED_ACTIONS = tuple(action.name for action in GENERAL_CODEC_REFERENCE_ACTIONS[10:12])
BLOSC_REFERENCE_ACTIONS_BY_KIND = {
    kind: (shuffle, *_BLOSCLZ_SHARED_ACTIONS, GENERAL_CODEC_REFERENCE_ACTIONS[12].name)
    for kind, shuffle in BLOSC_SHUFFLE_ACTION_BY_KIND.items()
}
BLOSC2_REFERENCE_ACTIONS_BY_KIND = {
    kind: (shuffle, *_BLOSCLZ_SHARED_ACTIONS, GENERAL_CODEC_REFERENCE_ACTIONS[13].name)
    for kind, shuffle in BLOSC_SHUFFLE_ACTION_BY_KIND.items()
}
BZIP2_REFERENCE_ACTIONS = (
    GENERAL_CODEC_REFERENCE_ACTIONS[14].name,
    GENERAL_CODEC_REFERENCE_ACTIONS[15].name,
    GENERAL_CODEC_REFERENCE_ACTIONS[16].name,
    GENERAL_CODEC_REFERENCE_ACTIONS[2].name,
)
DEFLATE_REFERENCE_LZ_ACTION = GENERAL_CODEC_REFERENCE_ACTIONS[17].name
XZ_REFERENCE_LZ_ACTION = GENERAL_CODEC_REFERENCE_ACTIONS[18].name
REFERENCE_RANGE_ACTION = GENERAL_CODEC_REFERENCE_ACTIONS[19].name
GZIP_REFERENCE_FRAME_ACTION = GENERAL_CODEC_REFERENCE_ACTIONS[20].name
ZLIB_REFERENCE_FRAME_ACTION = GENERAL_CODEC_REFERENCE_ACTIONS[21].name
XZ_REFERENCE_FRAME_ACTION = GENERAL_CODEC_REFERENCE_ACTIONS[22].name
GZIP_REFERENCE_ACTIONS = (
    DEFLATE_REFERENCE_LZ_ACTION,
    GENERAL_CODEC_REFERENCE_ACTIONS[2].name,
    GZIP_REFERENCE_FRAME_ACTION,
)
ZLIB_REFERENCE_ACTIONS = (
    DEFLATE_REFERENCE_LZ_ACTION,
    ZSTD_REFERENCE_ACTIONS[1],
    GENERAL_CODEC_REFERENCE_ACTIONS[2].name,
    ZLIB_REFERENCE_FRAME_ACTION,
)
XZ_REFERENCE_ACTIONS = (
    XZ_REFERENCE_LZ_ACTION,
    ZSTD_REFERENCE_ACTIONS[1],
    REFERENCE_RANGE_ACTION,
    XZ_REFERENCE_FRAME_ACTION,
)
# Backward-compatible names retained for reports/scripts generated during the
# first audit pass.  All stages in these three chains are now reference stages.
GZIP_MIXED_REFERENCE_ACTIONS = GZIP_REFERENCE_ACTIONS
ZLIB_MIXED_REFERENCE_ACTIONS = ZLIB_REFERENCE_ACTIONS
XZ_MIXED_REFERENCE_ACTIONS = XZ_REFERENCE_ACTIONS

BROTLI_EXACT_ACTIONS = BROTLI_REFERENCE_ACTIONS + (
    GENERAL_CODEC_RESIDUAL_ACTIONS[0].name,
)
ZSTD_EXACT_ACTIONS = ZSTD_REFERENCE_ACTIONS + (
    GENERAL_CODEC_RESIDUAL_ACTIONS[1].name,
)
_BLOSC1_RESIDUAL_BY_KIND = dict(zip(
    ("bytes", "float32", "float64", "int64"),
    (
        GENERAL_CODEC_RESIDUAL_ACTIONS[2].name,
        GENERAL_CODEC_RESIDUAL_ACTIONS[3].name,
        GENERAL_CODEC_RESIDUAL_ACTIONS[4].name,
        GENERAL_CODEC_RESIDUAL_ACTIONS[4].name,
    ),
))
_BLOSC2_RESIDUAL_BY_KIND = dict(zip(
    ("bytes", "float32", "float64", "int64"),
    (
        GENERAL_CODEC_RESIDUAL_ACTIONS[5].name,
        GENERAL_CODEC_RESIDUAL_ACTIONS[6].name,
        GENERAL_CODEC_RESIDUAL_ACTIONS[7].name,
        GENERAL_CODEC_RESIDUAL_ACTIONS[7].name,
    ),
))
BLOSC_EXACT_ACTIONS_BY_KIND = {
    kind: (shuffle, *_BLOSCLZ_SHARED_ACTIONS, _BLOSC1_RESIDUAL_BY_KIND[kind])
    for kind, shuffle in BLOSC_SHUFFLE_ACTION_BY_KIND.items()
}
BLOSC2_EXACT_ACTIONS_BY_KIND = {
    kind: (shuffle, *_BLOSCLZ_SHARED_ACTIONS, _BLOSC2_RESIDUAL_BY_KIND[kind])
    for kind, shuffle in BLOSC_SHUFFLE_ACTION_BY_KIND.items()
}
BZIP2_EXACT_ACTIONS = BZIP2_REFERENCE_ACTIONS + (
    GENERAL_CODEC_RESIDUAL_ACTIONS[8].name,
)
GZIP_EXACT_ACTIONS = (
    DEFLATE_REFERENCE_LZ_ACTION,
    GENERAL_CODEC_REFERENCE_ACTIONS[2].name,
    GENERAL_CODEC_RESIDUAL_ACTIONS[9].name,
    *GZIP_CONTAINER_ACTIONS,
)
ZLIB_EXACT_ACTIONS = (
    DEFLATE_REFERENCE_LZ_ACTION,
    ZSTD_REFERENCE_ACTIONS[1],
    GENERAL_CODEC_REFERENCE_ACTIONS[2].name,
    GENERAL_CODEC_RESIDUAL_ACTIONS[10].name,
    *ZLIB_CONTAINER_ACTIONS,
)
XZ_EXACT_ACTIONS = (
    XZ_REFERENCE_LZ_ACTION,
    ZSTD_REFERENCE_ACTIONS[1],
    REFERENCE_RANGE_ACTION,
    GENERAL_CODEC_RESIDUAL_ACTIONS[11].name,
    *XZ_CONTAINER_ACTIONS,
)
GENERAL_CODEC_REFERENCE_ACTIONS_BY_NAME = {
    action.name: action for action in GENERAL_CODEC_REFERENCE_ACTIONS
}
REFERENCE_STAGE_ACTION_NAMES = frozenset(
    action.name
    for action in (
        ALP_REFERENCE_ACTIONS + REGER_REFERENCE_ACTIONS + GENERAL_CODEC_REFERENCE_ACTIONS
    )
)
GENERAL_CODEC_REFERENCE_STAGE_CLI = (
    REPO_ROOT / "cpp/build/general_codec_reference_stages_cli"
)

RLE_INT64_ACTION_CLI = REPO_ROOT / "cpp/build/rle_int64_action_cli"
RLE_STAGE_CLI = REPO_ROOT / "cpp/build/rle_exact_stages_cli"


def _run_ts2diff_stage_cli(command: str, data: bytes, max_point: int | None = None) -> bytes:
    if not TS2DIFF_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {TS2DIFF_STAGE_CLI}; build target ts2diff_exact_stages_cli first"
        )
    argv = [str(TS2DIFF_STAGE_CLI), command]
    if max_point is not None:
        argv.append(str(int(max_point)))
    completed = subprocess.run(argv, input=bytes(data), capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"TS_2DIFF {command} failed: {detail}")
    return completed.stdout


def _run_svb_stage_cli(command: str, data: bytes) -> bytes:
    if not SVB_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {SVB_STAGE_CLI}; build target svb_exact_stages_cli first"
        )
    completed = subprocess.run(
        [str(SVB_STAGE_CLI), command],
        input=bytes(data),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"SVB {command} failed: {detail}")
    return completed.stdout


def _run_simple8b_stage_cli(
    command: str, data: bytes, max_point: int | None = None
) -> bytes:
    if not SIMPLE8B_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {SIMPLE8B_STAGE_CLI}; build target simple8b_exact_stages_cli first"
        )
    argv = [str(SIMPLE8B_STAGE_CLI), command]
    if max_point is not None:
        argv.append(str(int(max_point)))
    completed = subprocess.run(
        argv, input=bytes(data), capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Simple8b {command} failed: {detail}")
    return completed.stdout


def _run_p04_stage_cli(command: str, data: bytes) -> bytes:
    if not P04_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {P04_STAGE_CLI}; build target p04_exact_stages_cli first"
        )
    completed = subprocess.run(
        [str(P04_STAGE_CLI), command],
        input=bytes(data),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"p04 {command} failed: {detail}")
    return completed.stdout


def _run_dbp_stage_cli(
    command: str, data: bytes, max_point: int | None = None
) -> bytes:
    if not DBP_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {DBP_STAGE_CLI}; build target dbp_exact_stages_cli first"
        )
    argv = [str(DBP_STAGE_CLI), command]
    if max_point is not None:
        argv.append(str(int(max_point)))
    completed = subprocess.run(
        argv, input=bytes(data), capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"DBP {command} failed: {detail}")
    return completed.stdout


def _run_gorilla_stage_cli(command: str, data: bytes) -> bytes:
    if not GORILLA_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {GORILLA_STAGE_CLI}; build target gorilla_exact_stages_cli first"
        )
    completed = subprocess.run(
        [str(GORILLA_STAGE_CLI), command],
        input=bytes(data), capture_output=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Gorilla {command} failed: {detail}")
    return completed.stdout


def _run_chimp128_stage_cli(command: str, data: bytes) -> bytes:
    if not CHIMP128_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {CHIMP128_STAGE_CLI}; build target chimp128_exact_stages_cli first"
        )
    completed = subprocess.run(
        [str(CHIMP128_STAGE_CLI), command],
        input=bytes(data), capture_output=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Chimp128 {command} failed: {detail}")
    return completed.stdout


def _run_pongo_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(PONGO_STAGE_CLI, "Pongo", command, data)


def _run_spdp_stage_cli(
    command: str, data: bytes, original_size: int | None = None
) -> bytes:
    if not SPDP_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {SPDP_STAGE_CLI}; build target spdp_exact_stages_cli first"
        )
    argv = [str(SPDP_STAGE_CLI), command]
    if original_size is not None:
        argv.append(str(int(original_size)))
    completed = subprocess.run(
        argv, input=bytes(data), capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"SPDP {command} failed: {detail}")
    return completed.stdout


def _run_fpc_stage_cli(command: str, data: bytes) -> bytes:
    if not FPC_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {FPC_STAGE_CLI}; build target fpc_exact_stages_cli first"
        )
    completed = subprocess.run(
        [str(FPC_STAGE_CLI), command], input=bytes(data),
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FPC {command} failed: {detail}")
    return completed.stdout


def _run_pfpc_stage_cli(command: str, data: bytes) -> bytes:
    if not PFPC_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {PFPC_STAGE_CLI}; build target pfpc_exact_stages_cli first"
        )
    completed = subprocess.run(
        [str(PFPC_STAGE_CLI), command], input=bytes(data),
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"PFPC {command} failed: {detail}")
    return completed.stdout


def _run_exact_stage_cli(executable: Path, label: str, command: str, data: bytes) -> bytes:
    if not executable.is_file():
        raise RuntimeError(f"missing {executable}; build its exact-stage CLI target first")
    completed = subprocess.run(
        [str(executable), command], input=bytes(data), capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{label} {command} failed: {detail}")
    return completed.stdout


def _run_bos_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(BOS_STAGE_CLI, "BOS", command, data)


def _run_fastpfor_stage_cli(
    command: str, data: bytes, expected_count: int | None = None
) -> bytes:
    if expected_count is None:
        return _run_exact_stage_cli(FASTPFOR_STAGE_CLI, "FastPFor", command, data)
    if not FASTPFOR_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {FASTPFOR_STAGE_CLI}; build its exact-stage CLI target first"
        )
    completed = subprocess.run(
        [str(FASTPFOR_STAGE_CLI), command, str(int(expected_count))],
        input=bytes(data), capture_output=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FastPFor {command} failed: {detail}")
    return completed.stdout


def _run_elf_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(ELF_STAGE_CLI, "Elf", command, data)


def _run_camel_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(CAMEL_STAGE_CLI, "Camel", command, data)


def _run_snappy_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(SNAPPY_STAGE_CLI, "Snappy", command, data)


def _run_lz4_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(LZ4_STAGE_CLI, "LZ4", command, data)


def _run_deflate_container_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(
        DEFLATE_CONTAINER_STAGE_CLI, "DEFLATE container", command, data
    )


def _run_xz_container_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(XZ_CONTAINER_STAGE_CLI, "XZ container", command, data)


def _run_fpzip_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(FPZIP_STAGE_CLI, "FPZIP", command, data)


def normalize_buff_plane_parameters(parameters):
    if parameters is None: return {}
    if not isinstance(parameters, dict) or set(parameters) - {"bit_width"}:
        raise ValueError("BUFF byte-plane parameters accept only bit_width")
    if "bit_width" in parameters and (type(parameters["bit_width"]) is not int or not 0 <= parameters["bit_width"] <= 64):
        raise ValueError("BUFF bit_width must be an integer in [0,64]")
    return dict(parameters)


def run_buff_byte_plane(data: bytes, bit_width: int | None = None) -> tuple[bytes, int]:
    # Automatic width is an invocation helper; the C++ split kernel always takes b.
    if bit_width is None:
        selected = _run_learning_codec_stage_cli("buff-byte-plane-width", data)
        if len(selected) != 1: raise ValueError("invalid BUFF selected width")
        bit_width = selected[0]
    normalize_buff_plane_parameters({"bit_width":bit_width})
    result = subprocess.run([str(LEARNING_CODEC_STAGE_CLI), "buff-byte-plane-forward", str(bit_width)], input=data, capture_output=True)
    if result.returncode: raise ValueError(result.stderr.decode(errors="replace"))
    return result.stdout, bit_width


def _run_learning_codec_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(
        LEARNING_CODEC_STAGE_CLI, "LECO/ALP/REGER/BUFF", command, data
    )


def _run_general_codec_reference_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(
        GENERAL_CODEC_REFERENCE_STAGE_CLI, "general-codec reference", command, data
    )


def _run_general_codec_residual_stage_cli(command: str, data: bytes) -> bytes:
    return _run_exact_stage_cli(
        GENERAL_CODEC_RESIDUAL_STAGE_CLI, "general-codec residual", command, data
    )


def _run_rle_int64_action_cli(command: str, data: bytes) -> bytes:
    """Run the independently callable integer-only C++ RLE action."""

    if not RLE_INT64_ACTION_CLI.is_file():
        raise RuntimeError(
            f"missing {RLE_INT64_ACTION_CLI}; build target rle_int64_action_cli first"
        )
    completed = subprocess.run(
        [str(RLE_INT64_ACTION_CLI), command],
        input=bytes(data),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"RLE int64 C++ action {command} failed: {detail}")
    return completed.stdout


def _run_rle_stage_cli(
    command: str, data: bytes, max_point: int | None = None
) -> bytes:
    if not RLE_STAGE_CLI.is_file():
        raise RuntimeError(
            f"missing {RLE_STAGE_CLI}; build target rle_exact_stages_cli first"
        )
    argv = [str(RLE_STAGE_CLI), command]
    if max_point is not None:
        argv.append(str(int(max_point)))
    completed = subprocess.run(
        argv, input=bytes(data), capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"RLE exact stage {command} failed: {detail}")
    return completed.stdout


def encode_exact_simple8b_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_simple8b_stage_cli("bitpack-forward-int", data)
    if logical_type == "float":
        if max_point is None:
            raise ValueError("Simple8b float action pipeline requires max_point")
        return _run_simple8b_stage_cli("compose-float", data, max_point)
    raise ValueError(f"Simple8b requires int or float input, got {logical_type!r}")


def decode_exact_simple8b_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_simple8b_stage_cli("bitpack-inverse-int", data)
    if logical_type == "float":
        return _run_simple8b_stage_cli("decompose-float", data)
    raise ValueError(f"Simple8b requires int or float input, got {logical_type!r}")


def encode_exact_svb_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_svb_stage_cli("compose", data)
    if logical_type == "float":
        if max_point is None:
            raise ValueError("SVB float action pipeline requires max_point")
        scaling_packet = _run_dbp_stage_cli("scale-forward", data, max_point)
        return _run_svb_stage_cli("compose-scaled", scaling_packet)
    raise ValueError(f"SVB requires int or float input, got {logical_type!r}")


def decode_exact_svb_action_pipeline(data: bytes, logical_type: str) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_svb_stage_cli("decompose", data)
    if logical_type == "float":
        scaling_packet = _run_svb_stage_cli("decompose-scaled", data)
        return _run_dbp_stage_cli("scale-inverse", scaling_packet)
    raise ValueError(f"SVB requires int or float output, got {logical_type!r}")


def encode_exact_p04_action_pipeline(
    data: bytes, method: str, logical_type: str
) -> bytes:
    if logical_type not in {"int", "integer", "float"}:
        raise ValueError(f"{method} requires int or float input, got {logical_type!r}")
    command = {
        "RLBE": "rlbe-compose",
        "Sprintz": (
            "sprintz-compose-double" if logical_type == "float" else "sprintz-compose"
        ),
    }.get(method)
    if command is None:
        raise KeyError(f"unknown p04 method {method!r}")
    return _run_p04_stage_cli(command, data)


def decode_exact_p04_action_pipeline(
    data: bytes, method: str, logical_type: str
) -> bytes:
    if logical_type not in {"int", "integer", "float"}:
        raise ValueError(f"{method} requires int or float output, got {logical_type!r}")
    command = {
        "RLBE": "rlbe-decompose",
        "Sprintz": (
            "sprintz-decompose-double" if logical_type == "float" else "sprintz-decompose"
        ),
    }.get(method)
    if command is None:
        raise KeyError(f"unknown p04 method {method!r}")
    return _run_p04_stage_cli(command, data)


def encode_exact_dbp_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_dbp_stage_cli("compose-int", data)
    if logical_type == "float":
        if max_point is None:
            raise ValueError("DBP float action pipeline requires max_point")
        return _run_dbp_stage_cli("compose-float", data, max_point)
    raise ValueError(f"DBP requires int or float input, got {logical_type!r}")


def decode_exact_dbp_action_pipeline(data: bytes, logical_type: str) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_dbp_stage_cli("decompose-int", data)
    if logical_type == "float":
        return _run_dbp_stage_cli("decompose-float", data)
    raise ValueError(f"DBP requires int or float output, got {logical_type!r}")


def encode_exact_gorilla_action_pipeline(data: bytes, logical_type: str) -> bytes:
    command = {
        "int": "compose-int", "integer": "compose-int", "float": "compose-float"
    }.get(logical_type)
    if command is None:
        raise ValueError(f"Gorilla requires int or float input, got {logical_type!r}")
    return _run_gorilla_stage_cli(command, data)


def decode_exact_gorilla_action_pipeline(data: bytes, logical_type: str) -> bytes:
    command = {
        "int": "decompose-int", "integer": "decompose-int", "float": "decompose-float"
    }.get(logical_type)
    if command is None:
        raise ValueError(f"Gorilla requires int or float output, got {logical_type!r}")
    return _run_gorilla_stage_cli(command, data)


def encode_exact_chimp128_action_pipeline(data: bytes, logical_type: str) -> bytes:
    command = {
        "int": "compose-int", "integer": "compose-int", "float": "compose-float"
    }.get(logical_type)
    if command is None:
        raise ValueError(f"Chimp128 requires int or float input, got {logical_type!r}")
    return _run_chimp128_stage_cli(command, data)


def decode_exact_chimp128_action_pipeline(data: bytes, logical_type: str) -> bytes:
    command = {
        "int": "decompose-int", "integer": "decompose-int", "float": "decompose-float"
    }.get(logical_type)
    if command is None:
        raise ValueError(f"Chimp128 requires int or float output, got {logical_type!r}")
    return _run_chimp128_stage_cli(command, data)


def encode_exact_pongo_action_pipeline(data: bytes) -> bytes:
    return _run_pongo_stage_cli("compose", data)


def decode_exact_pongo_action_pipeline(data: bytes) -> bytes:
    return _run_pongo_stage_cli("decompose", data)


def encode_exact_spdp_action_pipeline(data: bytes) -> bytes:
    return _run_spdp_stage_cli("compose", data)


def decode_exact_spdp_action_pipeline(data: bytes, original_size: int) -> bytes:
    return _run_spdp_stage_cli("decompose", data, original_size)


def encode_exact_fpc_action_pipeline(data: bytes) -> bytes:
    return _run_fpc_stage_cli("compose", data)


def decode_exact_fpc_action_pipeline(data: bytes) -> bytes:
    return _run_fpc_stage_cli("decompose", data)


def encode_exact_pfpc_action_pipeline(data: bytes) -> bytes:
    return _run_pfpc_stage_cli("compose", data)


def decode_exact_pfpc_action_pipeline(data: bytes) -> bytes:
    return _run_pfpc_stage_cli("decompose", data)


def encode_exact_bos64_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        delta_packet = _run_bos_stage_cli("adapter-plain-delta-forward", data)
        zigzag_packet = _run_bos_stage_cli("adapter-zigzag-forward", delta_packet)
        return _run_bos_stage_cli("adapter-wire-forward-int", zigzag_packet)
    if logical_type == "float":
        if max_point is None:
            raise ValueError("BOS64 float action pipeline requires max_point")
        scaling_packet = _run_dbp_stage_cli("scale-forward", data, max_point)
        delta_packet = _run_bos_stage_cli("adapter-plain-delta-forward-scaled", scaling_packet)
        zigzag_packet = _run_bos_stage_cli("adapter-zigzag-forward-scaled", delta_packet)
        return _run_bos_stage_cli("adapter-wire-forward-scaled", zigzag_packet)
    raise ValueError(f"BOS64 requires int or float input, got {logical_type!r}")


def decode_exact_bos64_action_pipeline(data: bytes, logical_type: str) -> bytes:
    if logical_type in {"int", "integer"}:
        zigzag_packet = _run_bos_stage_cli("adapter-wire-inverse-int", data)
        delta_packet = _run_bos_stage_cli("adapter-zigzag-inverse", zigzag_packet)
        return _run_bos_stage_cli("adapter-plain-delta-inverse", delta_packet)
    if logical_type == "float":
        zigzag_packet = _run_bos_stage_cli("adapter-wire-inverse-scaled", data)
        delta_packet = _run_bos_stage_cli("adapter-zigzag-inverse-scaled", zigzag_packet)
        scaling_packet = _run_bos_stage_cli("adapter-plain-delta-inverse-scaled", delta_packet)
        return _run_dbp_stage_cli("scale-inverse", scaling_packet)
    raise ValueError(f"BOS64 requires int or float output, got {logical_type!r}")


def encode_exact_fastpfor_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        stage = _run_fastpfor_stage_cli("pfor-forward-int", data)
        return _run_fastpfor_stage_cli("variable-byte-forward", stage)
    if logical_type == "float":
        if max_point is None:
            raise ValueError("FastPFor float action pipeline requires max_point")
        scaling_packet = _run_dbp_stage_cli("scale-forward", data, max_point)
        stage = _run_fastpfor_stage_cli("pfor-forward-scaled", scaling_packet)
        return _run_fastpfor_stage_cli("variable-byte-forward", stage)
    raise ValueError(f"FastPFor requires int or float input, got {logical_type!r}")


def decode_exact_fastpfor_action_pipeline(
    data: bytes, logical_type: str, expected_count: int
) -> bytes:
    if logical_type in {"int", "integer"}:
        stage = _run_fastpfor_stage_cli(
            "variable-byte-inverse", data, expected_count
        )
        return _run_fastpfor_stage_cli("pfor-inverse-int", stage)
    if logical_type == "float":
        stage = _run_fastpfor_stage_cli(
            "variable-byte-inverse", data, expected_count
        )
        scaling_packet = _run_fastpfor_stage_cli("pfor-inverse-scaled", stage)
        return _run_dbp_stage_cli("scale-inverse", scaling_packet)
    raise ValueError(f"FastPFor requires int or float output, got {logical_type!r}")


def encode_exact_elf_action_pipeline(data: bytes, method: str) -> bytes:
    command = {"Elf": "elf-compose", "Elf+": "elf-plus-compose"}.get(method)
    if command is None:
        raise KeyError(f"unknown Elf-family method {method!r}")
    return _run_elf_stage_cli(command, data)


def decode_exact_elf_action_pipeline(data: bytes, method: str) -> bytes:
    command = {"Elf": "elf-decompose", "Elf+": "elf-plus-decompose"}.get(method)
    if command is None:
        raise KeyError(f"unknown Elf-family method {method!r}")
    return _run_elf_stage_cli(command, data)


def encode_exact_camel_action_pipeline(data: bytes, logical_type: str) -> bytes:
    command = "compose-int" if logical_type in {"int", "integer"} else "compose"
    return _run_camel_stage_cli(command, data)


def decode_exact_camel_action_pipeline(data: bytes, logical_type: str) -> bytes:
    command = "decompose-int" if logical_type in {"int", "integer"} else "decompose"
    return _run_camel_stage_cli(command, data)


def encode_exact_snappy_action_pipeline(data: bytes) -> bytes:
    return _run_snappy_stage_cli("compose", data)


def decode_exact_snappy_action_pipeline(data: bytes) -> bytes:
    return _run_snappy_stage_cli("decompose", data)


def encode_exact_lz4_action_pipeline(data: bytes) -> bytes:
    return _run_lz4_stage_cli("compose", data)


def decode_exact_lz4_action_pipeline(data: bytes) -> bytes:
    return _run_lz4_stage_cli("decompose", data)


def encode_exact_fpzip_action_pipeline(data: bytes) -> bytes:
    return _run_fpzip_stage_cli("compose", data)


def decode_exact_fpzip_action_pipeline(data: bytes) -> bytes:
    return _run_fpzip_stage_cli("decompose", data)


def _run_learning_chain(data: bytes, commands: Sequence[str]) -> bytes:
    current = bytes(data)
    for command in commands:
        current = _run_learning_codec_stage_cli(command, current)
    return current


def encode_exact_leco_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data, ("leco-regression-forward", "leco-zigzag-forward", "leco-bitpack-forward")
    )


def decode_exact_leco_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data, ("leco-bitpack-inverse", "leco-zigzag-inverse", "leco-regression-inverse")
    )


def encode_reference_alp_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        ("alp-scaling-forward", "alp-for-forward", "alp-bitpack-forward"),
    )


def decode_reference_alp_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        ("alp-bitpack-inverse", "alp-for-inverse", "alp-scaling-inverse"),
    )


def encode_exact_alp_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        (
            "alp-scaling-forward",
            "alp-for-forward",
            "alp-bitpack-forward",
            "alp-residual-core-forward",
        ),
    )


def decode_exact_alp_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        (
            "alp-residual-core-inverse",
            "alp-bitpack-inverse",
            "alp-for-inverse",
            "alp-scaling-inverse",
        ),
    )


def encode_reference_reger_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        (
            "reger-reorder-forward",
            "reger-delta-forward",
            "reger-zigzag-forward",
            "reger-for-forward",
            "reger-bitpack-forward",
        ),
    )


def decode_reference_reger_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        (
            "reger-bitpack-inverse",
            "reger-for-inverse",
            "reger-zigzag-inverse",
            "reger-delta-inverse",
            "reger-reorder-inverse",
        ),
    )


def encode_exact_reger_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        (
            "reger-reorder-forward",
            "reger-delta-forward",
            "reger-zigzag-forward",
            "reger-for-forward",
            "reger-bitpack-forward",
            "reger-residual-core-forward",
        ),
    )


def decode_exact_reger_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        (
            "reger-residual-core-inverse",
            "reger-bitpack-inverse",
            "reger-for-inverse",
            "reger-zigzag-inverse",
            "reger-delta-inverse",
            "reger-reorder-inverse",
        ),
    )


def _run_general_reference_chain(data: bytes, commands: Sequence[str]) -> bytes:
    current = bytes(data)
    for command in commands:
        current = _run_general_codec_reference_stage_cli(command, current)
    return current


def encode_exact_general_residual_action_pipeline(
    data: bytes, action_names: Sequence[str]
) -> bytes:
    names = tuple(action_names)
    if names == BROTLI_EXACT_ACTIONS:
        prefix = _run_general_reference_chain(
            data, ("dictionary-forward", "brotli-lz-forward", "huffman-forward")
        )
        return _run_general_codec_residual_stage_cli("brotli-forward", prefix)
    if names == ZSTD_EXACT_ACTIONS:
        prefix = _run_general_reference_chain(
            data,
            ("zstd-lz-forward", "parse-forward", "huffman-forward", "rans-forward"),
        )
        return _run_general_codec_residual_stage_cli("zstd-forward", prefix)
    if names == BZIP2_EXACT_ACTIONS:
        prefix = _run_general_reference_chain(
            data, ("bwt-forward", "mtf-forward", "rle-forward", "huffman-forward")
        )
        return _run_general_codec_residual_stage_cli("bzip2-forward", prefix)
    if names == GZIP_EXACT_ACTIONS:
        prefix = _run_general_reference_chain(
            data, ("deflate-lz-forward", "huffman-forward")
        )
        packet = _run_general_codec_residual_stage_cli("gzip-forward", prefix)
        return _run_deflate_container_stage_cli("gzip-forward", packet)
    if names == ZLIB_EXACT_ACTIONS:
        prefix = _run_general_reference_chain(
            data, ("deflate-lz-forward", "parse-forward", "huffman-forward")
        )
        packet = _run_general_codec_residual_stage_cli("zlib-forward", prefix)
        return _run_deflate_container_stage_cli("zlib-forward", packet)
    if names == XZ_EXACT_ACTIONS:
        prefix = _run_general_reference_chain(
            data, ("xz-lz-forward", "parse-forward", "range-forward")
        )
        packet = _run_general_codec_residual_stage_cli("xz-forward", prefix)
        return _run_xz_container_stage_cli("forward", packet)
    for kind, pipeline in BLOSC_EXACT_ACTIONS_BY_KIND.items():
        if names == pipeline:
            type_size = {"bytes": 1, "float32": 4, "float64": 8, "int64": 8}[kind]
            prefix = _run_general_reference_chain(
                data,
                (f"shuffle{type_size}-forward", "blosclz-forward", "token-forward"),
            )
            return _run_general_codec_residual_stage_cli(
                f"blosc1-forward-{type_size}", prefix
            )
    for kind, pipeline in BLOSC2_EXACT_ACTIONS_BY_KIND.items():
        if names == pipeline:
            type_size = {"bytes": 1, "float32": 4, "float64": 8, "int64": 8}[kind]
            prefix = _run_general_reference_chain(
                data,
                (f"shuffle{type_size}-forward", "blosclz-forward", "token-forward"),
            )
            return _run_general_codec_residual_stage_cli(
                f"blosc2-forward-{type_size}", prefix
            )
    raise ValueError("not a registered exact general-codec residual pipeline")


def encode_exact_buff_action_pipeline(data: bytes) -> bytes:
    scaled = _run_learning_codec_stage_cli("buff-scaling-forward", data)
    normalized = _run_learning_codec_stage_cli("buff-for-forward", scaled)
    planes, _ = run_buff_byte_plane(normalized)
    return _run_learning_codec_stage_cli("buff-bitpack-forward", planes)


def decode_exact_buff_action_pipeline(data: bytes) -> bytes:
    return _run_learning_chain(
        data,
        ("buff-bitpack-inverse", "buff-byte-plane-inverse", "buff-for-inverse", "buff-scaling-inverse"),
    )


def encode_exact_ts2diff_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_ts2diff_stage_cli("compose-int", data)
    if logical_type == "float":
        if max_point is None:
            raise ValueError("TS_2DIFF float action pipeline requires max_point")
        return _run_ts2diff_stage_cli("compose-float", data, max_point)
    raise ValueError(f"TS_2DIFF requires int or float input, got {logical_type!r}")


def decode_exact_ts2diff_action_pipeline(
    data: bytes, logical_type: str, *, max_point: int | None = None
) -> bytes:
    if logical_type in {"int", "integer"}:
        return _run_ts2diff_stage_cli("decompose-int", data)
    if logical_type == "float":
        if max_point is None:
            raise ValueError("TS_2DIFF float action pipeline requires max_point")
        return _run_ts2diff_stage_cli("decompose-float", data, max_point)
    raise ValueError(f"TS_2DIFF requires int or float input, got {logical_type!r}")

FLOAT_SCALING_THEN_INT_RLE_ACTIONS = (
    RLE_SCALING_ACTION,
    "op.run_length.tsfile_int64.v1",
)


def exact_atomic_action_pipeline(
    method: str,
    logical_type: str,
    *,
    max_point: int | None = None,
) -> tuple[str, ...]:
    """Resolve methods that are themselves one real primitive operator."""

    if method == "Huffman":
        return ("op.entropy_huffman.project_huffman_bytes.v1",)
    if method == "Arithmetic":
        return ("op.entropy_range.project_arithmetic_bytes.v1",)
    if method == "ANS":
        return ("op.entropy_fse.project_rans_bytes.v1",)
    if method == "Dict.":
        suffix = "int64" if logical_type in {"int", "integer"} else "float64"
        return (f"op.dictionary.project_{suffix}.v1",)
    if method == "RLE":
        if logical_type in {"int", "integer"}:
            return ("op.run_length.tsfile_int64.v1",)
        if (
            not isinstance(max_point, int)
            or max_point < 0
            or max_point > RLE_MAX_POINT_LIMIT
        ):
            raise ValueError("scaling -> integer RLE requires max_point in [0,18]")
        return FLOAT_SCALING_THEN_INT_RLE_ACTIONS
    raise KeyError(f"{method!r} is not an exact atomic operator method")


def encode_exact_atomic_method(
    data: bytes,
    method: str,
    logical_type: str,
    *,
    max_point: int | None = None,
) -> bytes:
    """Execute one exact atomic operator against the benchmark's 64-bit wire."""

    raw = bytes(data)
    if method == "Huffman":
        return huffman_encode(raw)
    if method == "Arithmetic":
        return arithmetic_encode(raw)
    if method == "ANS":
        return ans_encode(raw)
    if logical_type in {"int", "integer"}:
        values: Sequence[int] | Sequence[float] = np.frombuffer(raw, dtype="<i8").tolist()
        if method == "Dict.":
            return encode_dictionary_int([int(value) for value in values])
        if method == "RLE":
            return _run_rle_int64_action_cli("encode", raw)
    else:
        values = np.frombuffer(raw, dtype="<f8").tolist()
        if method == "Dict.":
            return encode_dictionary_float([float(value) for value in values])
        if method == "RLE":
            if max_point is None:
                raise ValueError("exact float RLE requires max_point")
            return _run_rle_stage_cli("compose", raw, max_point)
    raise KeyError(f"{method!r} is not an exact atomic operator method")


def decode_rle_int64(data: bytes) -> list[int]:
    """Compatibility API; the integer RLE decoder executes in C++."""
    return np.frombuffer(_run_rle_int64_action_cli("decode", data), dtype="<i8").tolist()


def decode_rle_double(data: bytes) -> list[float]:
    """Compatibility API; both RLE and decimal inverse execute in C++."""
    return np.frombuffer(_run_rle_stage_cli("decompose", data), dtype="<f8").tolist()

BENCHMARK_86_SPECIALISTS = (
    "Chimp", "Chimp128", "Pongo", "Gorilla", "2DIFF", "BOS", "REGER",
    "SVB", "FastPFor", "Simple8b", "RLE", "RLBE", "Sprintz", "Dict.",
    "DBP", "ALP", "LECO", "Camel", "Elf", "Elf+", "FPC", "PFPC",
    "SPDP", "CORAD", "BUFF", "FPZIP", "FPCmp-R", "FPCmp-S",
)

BENCHMARK_86_GENERALS = (
    "gzip", "Brotli", "Zstd", "LZ4", "XZ", "Density", "Gipfeli", "GLZA",
    "LZMAT", "LZO1", "LZO2A", "LZRW", "LZSSE", "LZVN", "Pithy",
    "Shrinker", "NRV", "WFLZ", "XPACK", "YALZ77", "Yappy", "Snappy",
    "ANS", "FSE", "Huffman", "Huff0", "Arithmetic", "Blosc", "Blosc2",
    "BriefLZ", "BSC", "Bzip2", "Bzip3", "Crush", "CSC", "FastLZ",
    "FastLZMA2", "Kanzi", "Libdeflate", "Lizard", "LZ4 Fast", "LZ4HC",
    "LZAV", "LZF", "LZFSE", "LZG", "LZHAM", "LZJB", "Lzlib", "PPMD8",
    "QuickLZ", "SLZ", "Tornado", "Zlib", "Zlib-ng", "Zling", "ZXC", "TAMP",
)

BENCHMARK_86_METHODS = BENCHMARK_86_SPECIALISTS + BENCHMARK_86_GENERALS
if len(BENCHMARK_86_SPECIALISTS) != 28 or len(BENCHMARK_86_GENERALS) != 58:
    raise RuntimeError("the benchmark catalog must contain exactly 28 specialist and 58 general methods")

# One level per general method, selected by minimum sum of compressed bytes over
# the fixed 20-column benchmark.  None means no single locally available level
# round-tripped on all 20 inputs and therefore cannot support an exact claim.
BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS: dict[str, str | None] = {
    "gzip": "gz",
    "Brotli": "brotli",
    "Zstd": "zstd_level_9",
    "LZ4": "lz4-framed",
    "XZ": "xz_level_5",
    "Density": "density",
    "Gipfeli": "gipfeli",
    "GLZA": "glza",
    "LZMAT": "lzmat",
    "LZO1": "lzo1x_334",
    "LZO2A": "lzo2a",
    "LZRW": "lzrw",
    "LZSSE": "lzsse8",
    "LZVN": "lzvn",
    "Pithy": "pithy",
    "Shrinker": "shrinker",
    "NRV": "ucl_nrv2b_9",
    "WFLZ": "wflz",
    "XPACK": "xpack",
    "YALZ77": "yalz77",
    "Yappy": "yappy",
    "Snappy": "snappy-framed",
    "ANS": "ans",
    "FSE": "fse",
    "Huffman": "huffman",
    "Huff0": "huff0",
    "Arithmetic": "arithmetic",
    "Blosc": None,
    "Blosc2": None,
    "BriefLZ": "brieflz_9",
    "BSC": "bsc5",
    "Bzip2": "bzip2_4",
    "Bzip3": "bzip3_10",
    "Crush": "crush_2",
    "CSC": "csc_5",
    "FastLZ": "fastlz_2",
    "FastLZMA2": "fastlzma2_9",
    "Kanzi": "kanzi_8",
    "Libdeflate": "libdeflate_12",
    "Lizard": "lizard_19",
    "LZ4 Fast": "lz4fast_17",
    "LZ4HC": "lz4hc_12",
    "LZAV": "lzav_2",
    "LZF": "lzf_1",
    "LZFSE": "lzfse",
    "LZG": "lzg_9",
    "LZHAM": "lzham_4",
    "LZJB": "lzjb",
    "Lzlib": "lzlib_8",
    "PPMD8": "ppmd8_9",
    "QuickLZ": "quicklz_3",
    "SLZ": "slz_zlib_3",
    "Tornado": "tornado_13",
    "Zlib": "zlib_9",
    "Zlib-ng": "zlib_ng_9",
    "Zling": "zling_4",
    "ZXC": "zxc_6",
    "TAMP": "tamp_15",
}
if tuple(BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS) != BENCHMARK_86_GENERALS:
    raise RuntimeError("global-best level map must follow the exact benchmark-86 general method order")

EXACT_NATIVE_GENERAL_METHODS = (
    "Density", "Gipfeli", "GLZA", "LZMAT", "LZO1", "LZO2A", "LZRW",
    "LZSSE", "LZVN", "Pithy", "Shrinker", "NRV", "WFLZ", "XPACK",
    "YALZ77", "Yappy", "FSE", "Huff0", "BriefLZ", "BSC", "Bzip3",
    "Crush", "CSC", "FastLZ", "FastLZMA2", "Kanzi", "Libdeflate",
    "Lizard", "LZ4 Fast", "LZ4HC", "LZAV", "LZF", "LZFSE", "LZG",
    "LZHAM", "LZJB", "Lzlib", "PPMD8", "QuickLZ", "SLZ", "Tornado",
    "Zlib-ng", "Zling", "ZXC", "TAMP",
)
EXACT_NATIVE_METHOD_KEYS: dict[str, str] = {
    method: str(BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS[method])
    for method in EXACT_NATIVE_GENERAL_METHODS
}
EXACT_NATIVE_METHOD_KEYS.update(
    {"FPCmp-R": "fpcompress_dpratio", "FPCmp-S": "fpcompress_dpspeed"}
)

_UNREGISTERED_EXACT_NATIVE_ACTIONS = exact_native_lz_action_registry(
    EXACT_NATIVE_METHOD_KEYS.values()
)
EXACT_NATIVE_ACTIONS = _UNREGISTERED_EXACT_NATIVE_ACTIONS
EXACT_NATIVE_ACTION_BY_NAME = {
    action.name: action for action in EXACT_NATIVE_ACTIONS
}
LEGACY_NUMERIC_ACTIONS = tuple(
    ExactByteActionSpec(
        name=name, operator_id=name.split('.')[1],
        implementation_id=name.split('op.', 1)[1],
        implementation_source="backend/myapp/legacy_numeric_stages.py:forward/inverse",
        implementation_scope=(
            "frozen Python benchmark numeric stage; transformed data flows forward; "
            "the legacy FastPFor/LECO backend is two native-LZ limbs, not C++ FastPFor"
        ),
    ) for name in legacy_numeric_stages.ACTION_NAMES
)
SNAPPY_LEGACY_MATCH_ACTION = "op.lz_match.snappy_snap_1_1_1.python_v1"
SNAPPY_LEGACY_ACTIONS = (SNAPPY_LEGACY_MATCH_ACTION, *SNAPPY_ACTIONS[1:])
REGER_LEGACY_ACTIONS = (
    "op.block_state_control.reger_legacy_block_core.python_v1",
    "op.container_frame.reger3_legacy.python_v1",
)
BUFF_NATIVE_ACTIONS = (
    "op.block_transform.buff_native_word_order.cpp_v1",
    "op.container_frame.buff_native_raw64.cpp_v1",
)
BUFF_NATIVE_STAGE_CLI = REPO_ROOT / "cpp/build/buff_native_exact_stages_cli"
BLOSC2_LEGACY_ACTIONS = (
    "op.block_state_control.blosc2_3_3_3_binding_macro.python_v1",
    "op.container_frame.blosc2_legacy_chunk.python_v1",
)
# Includes compatibility stages for both the frozen Python benchmark and the
# current C++ BUFF entry point. These two BUFF baselines are intentionally distinct.
LEGACY_COMPATIBILITY_ACTIONS = LEGACY_NUMERIC_ACTIONS + (
    ExactByteActionSpec(
        name=SNAPPY_LEGACY_MATCH_ACTION, operator_id="lz_match",
        implementation_id="snappy_snap_1_1_1_python_v1",
        implementation_source="backend/myapp/snappy_legacy_stages.py:lz_match_forward/lz_match_inverse",
        implementation_scope="snap 1.1.1 hash/skip/greedy matcher; no whole encoder call",
    ),
) + tuple(
    ExactByteActionSpec(
        name=name, operator_id=name.split('.')[1],
        implementation_id=name.split('op.', 1)[1],
        implementation_source="backend/myapp/reger_legacy_stages.py",
        implementation_scope=("original selected REGER block core; model search, reorder, prediction and bitpack remain coupled"
                              if index == 0 else "REGER3 stream/block framing only"),
    ) for index, name in enumerate(REGER_LEGACY_ACTIONS)
) + tuple(
    ExactByteActionSpec(
        name=name, operator_id=name.split('.')[1],
        implementation_id=name.split('op.', 1)[1],
        implementation_source="cpp/src/buff_native_exact_stages.cpp",
        implementation_scope=("current native BUFF raw float64 word ordering"
                              if index == 0 else "current native BUFF raw64 wire header; not legacy bounded BUFF"),
    ) for index, name in enumerate(BUFF_NATIVE_ACTIONS)
) + tuple(
    ExactByteActionSpec(
        name=name, operator_id=name.split('.')[1],
        implementation_id=name.split('op.', 1)[1],
        implementation_source="backend/myapp/blosc2_legacy_stages.py",
        implementation_scope=("non-atomic version-pinned binding macro; calls whole chunk encoder then extracts blocks; private BLOSCLZ matcher not split"
                              if index == 0 else "independent Blosc2 chunk header and block-offset framing"),
    ) for index, name in enumerate(BLOSC2_LEGACY_ACTIONS)
)
LEGACY_COMPATIBILITY_ACTION_BY_NAME = {a.name: a for a in LEGACY_COMPATIBILITY_ACTIONS}
ELF64_ERASER_ACTION = "op.eraser.elf64.cpp_v1"
ELF64_ACTIONS = (ELF64_ERASER_ACTION, "op.xor.elf64_on_chimp.cpp_v1")
ELF64_PLUS_ACTIONS = (ELF64_ERASER_ACTION, "op.xor.elf64_plus.cpp_v1")
CORAD_LOSSLESS_ACTIONS = tuple(corad_lossless_stages.ACTION_NAMES)
LOSSLESS64_ACTIONS = tuple(
    ExactByteActionSpec(
        name=name, operator_id=name.split('.')[1],
        implementation_id=name.split('op.', 1)[1],
        implementation_source="cpp/src/elf_double_codec.cpp",
        implementation_scope=("binary64 eraser with exact reconstruction check; NaN excluded by experiment policy"
                              if name == ELF64_ERASER_ACTION else
                              "binary64 XOR backend with EL64 framing; Elf uses Chimp64, Elf+ uses ElfXor64"),
    ) for name in dict.fromkeys(ELF64_ACTIONS + ELF64_PLUS_ACTIONS)
) + tuple(
    ExactByteActionSpec(
        name=name, operator_id=corad_lossless_stages.ACTION_OPERATOR_IDS[name],
        implementation_id=name.split('op.', 1)[1],
        implementation_source="backend/myapp/corad_lossless_stages.py",
        implementation_scope=scope,
    ) for name, scope in zip(CORAD_LOSSLESS_ACTIONS, (
        "normalization with mean/sigma and exact word corrections stored in stream",
        "40-lane tricklet dictionary prediction and normalized-word XOR residual; all tail lanes retained",
        "non-atomic zlib9 DEFLATE backend; remaining LZ and Huffman core stays coupled",
        "CRD2 lossless frame, explicit version and compressed payload length",
    ))
)
LOSSLESS64_ACTION_BY_NAME = {action.name: action for action in LOSSLESS64_ACTIONS}
_FROZEN_EXACT_ACTION_INPUTS = (
    _UNREGISTERED_EXACT_NATIVE_ACTIONS
    + EXACT_ATOMIC_ACTIONS
    + RLE_SCALING_EXACT_ACTIONS
    + TS2DIFF_EXACT_ACTIONS
    + SVB_EXACT_ACTIONS
    + SIMPLE8B_EXACT_ACTIONS
    + P04_EXACT_ACTIONS
    + DBP_EXACT_ACTIONS
    + GORILLA_EXACT_ACTIONS
    + CHIMP128_EXACT_ACTIONS
    + PONGO_EXACT_ACTIONS
    + SPDP_EXACT_ACTIONS
    + FPC_EXACT_ACTIONS
    + PFPC_EXACT_ACTIONS
    + BOS_BOSA_EXACT_ACTIONS
    + BOS64_LEGACY_EXACT_ACTIONS
    + FASTPFOR_COMPOSITE_HELPER_ACTIONS
    + FASTPFOR_EXACT_ACTIONS
    + ELF_EXACT_ACTIONS
    + CAMEL_EXACT_ACTIONS
    + SNAPPY_LEGACY_EXACT_ACTIONS
    + LZ4_EXACT_ACTIONS
    + DEFLATE_CONTAINER_EXACT_ACTIONS
    + XZ_CONTAINER_EXACT_ACTIONS
    + FPZIP_EXACT_ACTIONS
    + LECO_LEGACY_EXACT_ACTIONS
    + ALP_REFERENCE_ACTIONS
    + REGER_LEGACY_REFERENCE_ACTIONS
    + LEARNING_CODEC_RESIDUAL_ACTIONS
    + BUFF_SOURCE_EXACT_ACTIONS
    + GENERAL_CODEC_REFERENCE_ACTIONS
    + GENERAL_CODEC_RESIDUAL_ACTIONS
    + LEGACY_COMPATIBILITY_ACTIONS
    + LOSSLESS64_ACTIONS
)
CURRENT_ZIGZAG_SPLIT_EXACT_ACTIONS = (
    ZIGZAG_RAW_ACTION,
    *ULEB128_EXACT_ACTIONS,
    ULEB128_SNAPPY_ACTION,
    *BOS64_EXACT_ACTIONS[:2],
    *LECO_SOURCE_EXACT_ACTIONS[:2],
    *REGER_REFERENCE_ACTIONS[1:3],
)
FROZEN_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = tuple(
    # Preserve the historical concrete IDs even though one frozen primitive
    # slot is no longer selectable.
    replace(action, action_id=len(LEGACY_PRIMITIVE_ACTIONS) + index)
    for index, action in enumerate(_FROZEN_EXACT_ACTION_INPUTS)
)
CURRENT_EXTENSION_EXACT_ACTIONS: tuple[ExactByteActionSpec, ...] = tuple(
    replace(action, action_id=len(LEGACY_PRIMITIVE_ACTIONS) + len(FROZEN_EXACT_ACTIONS) + index)
    for index, action in enumerate(CURRENT_ZIGZAG_SPLIT_EXACT_ACTIONS + BUFF_CURRENT_EXACT_ACTIONS)
)
EXACT_ACTIONS = FROZEN_EXACT_ACTIONS + CURRENT_EXTENSION_EXACT_ACTIONS
ALL_CONCRETE_ACTION_IMPLEMENTATIONS = tuple(
    replace(action, implementation_source=cpp_migration.cpp_source(
        action.name, action.implementation_source))
    for action in LEGACY_PRIMITIVE_ACTIONS + EXACT_ACTIONS
)
ALL_CONCRETE_ACTION_BY_NAME = {action.name: action for action in ALL_CONCRETE_ACTION_IMPLEMENTATIONS}
# This whole FastPFor adapter is useful for source forensics but is not a legal
# RL action because its PFor and VariableByte parts are already registered.
# LZ4/Snappy whole-encoder-backed first stages are intentionally retained as
# method-scoped residual-core actions, per the exact-composition contract.
POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES = frozenset({
    FASTPFOR_COMPOSITE_HELPER_ACTIONS[0].name,
})
LEGACY_225_POLICY_ACTIONS = tuple(
    action for action in tuple(
        replace(action, implementation_source=cpp_migration.cpp_source(
            action.name, action.implementation_source))
        for action in LEGACY_PRIMITIVE_ACTIONS + FROZEN_EXACT_ACTIONS
    )
    if action.name not in POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES
)
# Old concrete records are compatibility selectors, not extra implementations.
# Existing numeric IDs must be migrated explicitly, never silently reinterpreted.
LEGACY_206_ACTION_CANONICAL_NAMES = cpp_migration.build_aliases(
    action.name for action in LEGACY_225_POLICY_ACTIONS)
LEGACY_206_CANONICAL_TO_LEGACY_NAME = {}
for _action in LEGACY_225_POLICY_ACTIONS:
    _canonical = LEGACY_206_ACTION_CANONICAL_NAMES[_action.name]
    if _canonical not in LEGACY_206_CANONICAL_TO_LEGACY_NAME or _action.name == _canonical:
        LEGACY_206_CANONICAL_TO_LEGACY_NAME[_canonical] = _action.name
_legacy_206_concrete = tuple(
    replace(ALL_CONCRETE_ACTION_BY_NAME[old], name=name,
            implementation_id=name.split("op.", 1)[1])
    for name, old in LEGACY_206_CANONICAL_TO_LEGACY_NAME.items()
)
LEGACY_206_POLICY_ACTIONS = tuple(replace(action, action_id=index)
                                for index, action in enumerate(_legacy_206_concrete))
algorithm_io.validate_previous_registry(LEGACY_206_POLICY_ACTIONS)
FROZEN_POLICY_IO_VARIANTS = {
    group["name"]: tuple(v["name"] for v in group["io_variants"])
    for group in algorithm_io.GROUPS
}
IO_VARIANT_TO_POLICY_NAME = {
    **algorithm_io.VARIANT_TO_ALGORITHM,
    **algorithm_io.LEGACY_VARIANT_TO_ALGORITHM,
}
ACTION_CANONICAL_NAMES = {
    old: IO_VARIANT_TO_POLICY_NAME[variant]
    for old, variant in LEGACY_206_ACTION_CANONICAL_NAMES.items()
}
ACTION_CANONICAL_NAMES.update(IO_VARIANT_TO_POLICY_NAME)
ACTION_CANONICAL_NAMES.update(algorithm_io.PREVIOUS_ALGORITHM_TO_CURRENT)
ACTION_CANONICAL_NAMES.update({name: name for name in FROZEN_POLICY_IO_VARIANTS})
_REPLACED_RLE_NAMES = optimal_rle.OLD_ALGORITHMS | optimal_rle.OLD_TO_CURRENT.keys() | {optimal_rle.OLD_SPRINTZ_VARIANT}
ACTION_ALIASES = {old: new for old, new in ACTION_CANONICAL_NAMES.items() if old != new and old not in _REPLACED_RLE_NAMES}
# Legacy names still mean their original I/O layout.  This map is used only by
# old exact-chain dispatch and source indexing; new merged execution selects a
# variant explicitly (or from unambiguous input context) in unified runtime.
CANONICAL_TO_LEGACY_NAME = dict(LEGACY_206_CANONICAL_TO_LEGACY_NAME)
CANONICAL_TO_LEGACY_NAME.update({
    name: LEGACY_206_CANONICAL_TO_LEGACY_NAME[
        algorithm_io.VARIANT_BY_NAME[variants[0]].get("legacy_names", [variants[0]])[0]]
    for name, variants in FROZEN_POLICY_IO_VARIANTS.items()
})
LEGACY_182_POLICY_ACTIONS = tuple(
    replace(ALL_CONCRETE_ACTION_BY_NAME[LEGACY_206_CANONICAL_TO_LEGACY_NAME[group["io_variants"][0]["name"]]],
            action_id=group["policy_action_id"], name=group["name"],
            implementation_id=group["name"].split("op.", 1)[1], implementation_scope=group["algorithm"])
    for group in algorithm_io.previous.GROUPS
)
POLICY_CONCRETE_ACTIONS = tuple(
    replace(ALL_CONCRETE_ACTION_BY_NAME[CANONICAL_TO_LEGACY_NAME[group["name"]]],
            name=group["name"], operator_id=group["operator_id"], implementation_id=group["name"].split("op.", 1)[1],
            implementation_scope=group["algorithm"],
            implementation_source=group.get("implementation_source", ALL_CONCRETE_ACTION_BY_NAME[CANONICAL_TO_LEGACY_NAME[group["name"]]].implementation_source))
    for group in algorithm_io.GROUPS
)
FROZEN_POLICY_ACTIONS = tuple(replace(action, action_id=index)
                             for index, action in enumerate(POLICY_CONCRETE_ACTIONS))
FROZEN_POLICY_ACTION_BY_ID = {action.action_id: action for action in FROZEN_POLICY_ACTIONS}
FROZEN_POLICY_ACTION_BY_NAME = {action.name: action for action in FROZEN_POLICY_ACTIONS}
LECO_POLICY_NAME = "op.regression_predictor.leco_linear.cpp_v2"
ULEB128_POLICY_NAME = "op.integer_code.uleb128_unsigned.cpp_v2"
RETIRED_COMPOSITE_POLICY_IDS = frozenset({18, 35, 87, 90})
RETIRED_POLICY_IDS = frozenset({5, *RETIRED_COMPOSITE_POLICY_IDS})

V3_POLICY_SNAPSHOT = json.loads((REPO_ROOT / "benchmark/reference/actions/canonical127_v3_frozen.json").read_text())
V3_POLICY_BY_ID = {a["action_id"]: a for a in V3_POLICY_SNAPSHOT["actions"]}
POLICY_IO_VARIANTS = dict(FROZEN_POLICY_IO_VARIANTS)
for _retired_name in (
    "op.regression_predictor.leco.cpp_v1",
    "op.regression_predictor.reger.cpp_v1",
    "op.delta.bos64_adapter.cpp_v1",
    "op.integer_code.reference_v1",
    "op.integer_code.snappy.cpp_v1",
):
    POLICY_IO_VARIANTS.pop(_retired_name)
POLICY_IO_VARIANTS.pop(BUFF_ACTIONS[1])
POLICY_IO_VARIANTS[BUFF_PLANE_ACTION.name] = (BUFF_PLANE_ACTION.name,)
POLICY_IO_VARIANTS[minimum.ALGORITHM_NAME] += (BUFF_FOR_ACTION.name,)
POLICY_IO_VARIANTS[LECO_POLICY_NAME] = (LECO_SOURCE_EXACT_ACTIONS[0].name,)
POLICY_IO_VARIANTS["op.delta.previous_modular.cpp_v2"] += (
    REGER_REFERENCE_ACTIONS[1].name,
    BOS64_EXACT_ACTIONS[0].name,
)
POLICY_IO_VARIANTS["op.zigzag.signed64.cpp_v2"] += (
    ZIGZAG_RAW_ACTION.name,
    LECO_SOURCE_EXACT_ACTIONS[1].name,
    REGER_REFERENCE_ACTIONS[2].name,
    BOS64_EXACT_ACTIONS[1].name,
)
POLICY_IO_VARIANTS[ULEB128_POLICY_NAME] = (
    ULEB128_RAW_ACTION.name,
    ULEB128_DBP_ACTION.name,
    ULEB128_SNAPPY_ACTION.name,
)
IO_VARIANT_TO_POLICY_NAME.update({
    variant: policy_name
    for policy_name, variants in POLICY_IO_VARIANTS.items()
    for variant in variants
})
ACTION_CANONICAL_NAMES.update(IO_VARIANT_TO_POLICY_NAME)
ACTION_CANONICAL_NAMES.update({name: name for name in POLICY_IO_VARIANTS})
CANONICAL_TO_LEGACY_NAME.update({
    BUFF_PLANE_ACTION.name: BUFF_PLANE_ACTION.name,
    LECO_POLICY_NAME: LECO_SOURCE_EXACT_ACTIONS[0].name,
    ULEB128_POLICY_NAME: ULEB128_RAW_ACTION.name,
})

_active_policy_by_id = {action.action_id: action for action in FROZEN_POLICY_ACTIONS}
_active_policy_by_id[17] = replace(
    ALL_CONCRETE_ACTION_BY_NAME[LECO_SOURCE_EXACT_ACTIONS[0].name],
    action_id=17,
    name=LECO_POLICY_NAME,
    implementation_id=LECO_POLICY_NAME.split("op.", 1)[1],
)
_active_policy_by_id[25] = replace(
    ALL_CONCRETE_ACTION_BY_NAME[ULEB128_RAW_ACTION.name],
    action_id=25,
    name=ULEB128_POLICY_NAME,
    implementation_id=ULEB128_POLICY_NAME.split("op.", 1)[1],
)
_active_policy_by_id[58] = replace(ALL_CONCRETE_ACTION_BY_NAME[BUFF_PLANE_ACTION.name], action_id=58)
# V4 preserves all 127 numeric slots. Retired composites are recipe migrations,
# never aliases that silently execute a different single stage.
POLICY_ACTIONS = tuple(
    _active_policy_by_id[action_id]
    for action_id in range(len(FROZEN_POLICY_ACTIONS))
    if action_id not in RETIRED_POLICY_IDS
)
POLICY_ACTION_BY_ID = {action.action_id: action for action in POLICY_ACTIONS}
POLICY_ACTION_BY_NAME = {action.name: action for action in POLICY_ACTIONS}
POLICY_TO_CONCRETE_ACTION_ID = {
    action.action_id: ALL_CONCRETE_ACTION_BY_NAME[
        CANONICAL_TO_LEGACY_NAME.get(action.name, action.name)
    ].action_id
    for action in POLICY_ACTIONS
}
STOP_ACTION_ID = len(FROZEN_POLICY_ACTIONS)
LEGACY_ACTION_REGISTRY_VERSION = algorithm_io.REGISTRY_VERSION
PREVIOUS_ACTION_REGISTRY_VERSION = "cpp-canonical-127-v3"
ACTION_REGISTRY_VERSION = "cpp-canonical-127-v4"

# A frozen selector may expand into multiple current atomic actions.  Numeric
# slots stay fixed; this table is recipe migration metadata, never a forward
# alias for a retired composite implementation.
COMPOSITE_POLICY_REPLACEMENTS = {
    BUFF_ACTIONS[1]: ((14, BUFF_FOR_ACTION.name), (58, BUFF_PLANE_ACTION.name)),
    "op.regression_predictor.leco.cpp_v1": (
        (17, LECO_SOURCE_EXACT_ACTIONS[0].name),
        (55, LECO_SOURCE_EXACT_ACTIONS[1].name),
    ),
    "op.regression_predictor.reger.cpp_v1": (
        (30, REGER_REFERENCE_ACTIONS[1].name),
        (55, REGER_REFERENCE_ACTIONS[2].name),
    ),
    "op.delta.bos64_adapter.cpp_v1": (
        (30, BOS64_EXACT_ACTIONS[0].name),
        (55, BOS64_EXACT_ACTIONS[1].name),
    ),
    "op.integer_code.reference_v1": (
        (55, ZIGZAG_RAW_ACTION.name),
        (25, ULEB128_RAW_ACTION.name),
    ),
    "op.integer_code.snappy.cpp_v1": (
        (25, ULEB128_SNAPPY_ACTION.name),
    ),
    "op.bitpack_int.delta_binary_packed.cpp_v1": (
        (25, ULEB128_DBP_ACTION.name),
    ),
}
RETIRED_COMPOSITE_ACTION_NAMES = frozenset(COMPOSITE_POLICY_REPLACEMENTS)
LEGACY_206_POLICY_ID_TO_CURRENT = {
    action.action_id: FROZEN_POLICY_ACTION_BY_NAME[IO_VARIANT_TO_POLICY_NAME[action.name]].action_id
    for action in LEGACY_206_POLICY_ACTIONS
}
LEGACY_206_POLICY_ID_TO_CURRENT[206] = STOP_ACTION_ID
LEGACY_225_POLICY_ID_TO_CURRENT = {
    index: FROZEN_POLICY_ACTION_BY_NAME[ACTION_CANONICAL_NAMES[action.name]].action_id
    for index, action in enumerate(LEGACY_225_POLICY_ACTIONS)
}
LEGACY_225_POLICY_ID_TO_CURRENT[225] = STOP_ACTION_ID
LEGACY_182_POLICY_ID_TO_CURRENT = {
    action.action_id: FROZEN_POLICY_ACTION_BY_NAME[ACTION_CANONICAL_NAMES[action.name]].action_id
    for action in LEGACY_182_POLICY_ACTIONS
}
LEGACY_182_POLICY_ID_TO_CURRENT[182] = STOP_ACTION_ID
LEGACY_156_POLICY_ID_TO_CURRENT = {
    group["policy_action_id"]: FROZEN_POLICY_ACTION_BY_NAME[ACTION_CANONICAL_NAMES[group["name"]]].action_id
    for group in algorithm_io.previous156.GROUPS
}
LEGACY_156_POLICY_ID_TO_CURRENT[156] = STOP_ACTION_ID
LEGACY_143_POLICY_ID_TO_CURRENT = {
    group["policy_action_id"]: FROZEN_POLICY_ACTION_BY_NAME[ACTION_CANONICAL_NAMES[group["name"]]].action_id
    for group in algorithm_io.previous143.GROUPS
}
LEGACY_143_POLICY_ID_TO_CURRENT[143] = STOP_ACTION_ID
LEGACY_140_POLICY_ID_TO_CURRENT = {
    group["policy_action_id"]: FROZEN_POLICY_ACTION_BY_NAME[ACTION_CANONICAL_NAMES[group["name"]]].action_id
    for group in algorithm_io.previous140.GROUPS
}
LEGACY_140_POLICY_ID_TO_CURRENT[140] = STOP_ACTION_ID
LEGACY_137_POLICY_ID_TO_CURRENT = {
    group["policy_action_id"]: FROZEN_POLICY_ACTION_BY_NAME[ACTION_CANONICAL_NAMES[group["name"]]].action_id
    for group in algorithm_io.previous137.GROUPS
}
LEGACY_137_POLICY_ID_TO_CURRENT[137] = STOP_ACTION_ID
LEGACY_129_POLICY_ID_TO_CURRENT = {
    group['policy_action_id']: FROZEN_POLICY_ACTION_BY_NAME[ACTION_CANONICAL_NAMES[group['name']]].action_id
    for group in algorithm_io.previous129.GROUPS
}
LEGACY_129_POLICY_ID_TO_CURRENT[129] = STOP_ACTION_ID
if (len(LEGACY_225_POLICY_ACTIONS) != 225 or len(FROZEN_POLICY_ACTIONS) != 127
        or len(POLICY_ACTIONS) != 122 or set(range(127)) - set(POLICY_ACTION_BY_ID) != RETIRED_POLICY_IDS
        or FROZEN_POLICY_ACTION_BY_ID[5].name not in RETIRED_IDENTITY_ACTION_NAMES):
    raise RuntimeError("unexpected migration inventory; re-audit aliases before changing it")
if len(ALL_CONCRETE_ACTION_BY_NAME) != len(ALL_CONCRETE_ACTION_IMPLEMENTATIONS):
    raise RuntimeError("concrete action names must be globally unique")


def migrate_policy_action(action_id: int, registry_version: str, *,
                          io_variant: str | None = None,
                          parameters: dict[str, int] | None = None) -> dict[str, Any]:
    """Migrate a historical ID together with its format, never just its logit.

    Old trained heads are not interchangeable with the new algorithm/parameter heads.
    This helper migrates saved steps/recipes; it does not rewrite weights.
    """
    if type(action_id) is not int:
        raise ValueError("policy action ID must be an integer")
    if io_variant is not None and not isinstance(io_variant, str):
        raise ValueError("io_variant must be a string")
    raw_name = None
    source_choices = None
    if registry_version == ACTION_REGISTRY_VERSION:
        source_stop = STOP_ACTION_ID
        if action_id != source_stop and action_id not in POLICY_ACTION_BY_ID:
            raise ValueError("unknown current policy ID")
        if action_id != source_stop:
            raw_name = POLICY_ACTION_BY_ID[action_id].name
    elif registry_version == PREVIOUS_ACTION_REGISTRY_VERSION:
        source_stop = STOP_ACTION_ID
        if action_id != source_stop:
            if action_id not in V3_POLICY_BY_ID:
                raise ValueError("unknown canonical127-v3 policy ID")
            raw_name = V3_POLICY_BY_ID[action_id]["name"]
            source_choices = tuple(V3_POLICY_SNAPSHOT["io_variants"][raw_name])
    elif registry_version == "cpp-canonical-127-v2":
        source_stop = STOP_ACTION_ID
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical127-v2 policy ID")
        if action_id == 5:
            raw_name = FROZEN_POLICY_ACTION_BY_ID[action_id].name
            source_choices = FROZEN_POLICY_IO_VARIANTS[raw_name]
        elif action_id != source_stop:
            raw_name = FROZEN_POLICY_ACTION_BY_ID[action_id].name
            source_choices = FROZEN_POLICY_IO_VARIANTS[raw_name]
    elif registry_version == LEGACY_ACTION_REGISTRY_VERSION:
        source_stop = STOP_ACTION_ID
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical127-v1 policy ID")
        if action_id != source_stop:
            group = algorithm_io.GROUPS[action_id]
            raw_name = group["name"]
            source_choices = tuple(v["name"] for v in group["io_variants"])
    elif registry_version == "cpp-canonical-129-v1":
        source_stop = 129
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical129 policy ID")
        if action_id != source_stop:
            group = algorithm_io.previous129.GROUPS[action_id]
            raw_name = group["name"]
            source_choices = tuple(v["name"] for v in group["io_variants"])
    elif registry_version == "cpp-canonical-137-v1":
        source_stop = 137
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical137 policy ID")
        if action_id != source_stop:
            group = algorithm_io.previous137.GROUPS[action_id]
            raw_name = group["name"]
            source_choices = tuple(v["name"] for v in group["io_variants"])
    elif registry_version == "cpp-canonical-140-v1":
        source_stop = 140
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical140 policy ID")
        if action_id != source_stop:
            group = algorithm_io.previous140.GROUPS[action_id]
            raw_name = group["name"]
            source_choices = tuple(v["name"] for v in group["io_variants"])
    elif registry_version == "cpp-canonical-143-v1":
        source_stop = 143
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical143 policy ID")
        if action_id != source_stop:
            group = algorithm_io.previous143.GROUPS[action_id]
            raw_name = group["name"]
            source_choices = tuple(v["name"] for v in group["io_variants"])
    elif registry_version == "cpp-canonical-156-v1":
        source_stop = 156
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical156 policy ID")
        if action_id != source_stop:
            group = algorithm_io.previous156.GROUPS[action_id]
            raw_name = group["name"]
            source_choices = tuple(v["name"] for v in group["io_variants"])
    elif registry_version == "cpp-canonical-182-v1":
        source_stop = 182
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical182 policy ID")
        if action_id != source_stop:
            group = algorithm_io.previous.GROUPS[action_id]
            raw_name = group["name"]
            source_choices = tuple(v["name"] for v in group["io_variants"])
    elif registry_version == "cpp-canonical-206-v1":
        source_stop = 206
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown canonical206 policy ID")
        if action_id != source_stop:
            raw_name = LEGACY_206_POLICY_ACTIONS[action_id].name
            source_choices = (raw_name,)
    elif registry_version in {"cpp-canonical-225-v1", "legacy-225-v1"}:
        source_stop = 225
        if not 0 <= action_id <= source_stop:
            raise ValueError("unknown legacy225 policy ID")
        if action_id != source_stop:
            raw_name = LEGACY_225_POLICY_ACTIONS[action_id].name
            source_choices = (LEGACY_206_ACTION_CANONICAL_NAMES[raw_name],)
    else:
        raise ValueError(f"unknown policy registry version: {registry_version!r}")
    if action_id == source_stop:
        if io_variant is not None or parameters not in (None, {}):
            raise ValueError("STOP has no I/O variant or parameters")
        return {"policy_action_id": STOP_ACTION_ID, "io_variant": None, "parameters": {}}
    canonical = ACTION_CANONICAL_NAMES.get(raw_name, raw_name)
    if canonical in RETIRED_IDENTITY_ACTION_NAMES:
        if io_variant not in (None, raw_name) or parameters not in (None, {}):
            raise ValueError("retired identity Scaling has no parameters or alternate I/O profile")
        return {
            "registry_version": ACTION_REGISTRY_VERSION,
            "migration_kind": "retired_identity_bypass",
            "replacement_policy_action_ids": [],
            "replacement_io_variants": [],
            "replacement_parameters": [],
            "source_registry_version": registry_version,
            "source_policy_action_id": action_id,
        }
    replacement = COMPOSITE_POLICY_REPLACEMENTS.get(canonical)
    if replacement is not None:
        if parameters not in (None, {}):
            raise ValueError("retired composite actions have no standalone parameters")
        if io_variant is not None:
            old_variant = LEGACY_206_ACTION_CANONICAL_NAMES.get(io_variant, io_variant)
            if source_choices is not None and old_variant not in source_choices:
                raise ValueError("I/O variant does not belong to the source policy ID")
        return {
            "registry_version": ACTION_REGISTRY_VERSION,
            "migration_kind": "replacement_recipe",
            "replacement_policy_action_ids": [row[0] for row in replacement],
            "replacement_io_variants": [row[1] for row in replacement],
            "replacement_parameters": [{} for _ in replacement],
            "source_registry_version": registry_version,
            "source_policy_action_id": action_id,
            "source_policy_action_name": canonical,
        }
    choices = POLICY_IO_VARIANTS[canonical]
    selected = io_variant
    if selected is not None:
        old_variant = LEGACY_206_ACTION_CANONICAL_NAMES.get(selected, selected)
        if source_choices is not None and old_variant not in source_choices:
            raise ValueError("I/O variant does not belong to the source policy ID")
        selected = algorithm_io.LEGACY_VARIANT_TO_CURRENT.get(old_variant, old_variant)
        if selected not in choices:
            raise ValueError("I/O variant does not belong to the target algorithm")
    elif source_choices is not None and len(source_choices) == 1:
        selected = algorithm_io.LEGACY_VARIANT_TO_CURRENT.get(source_choices[0], source_choices[0])
    elif len(choices) == 1:
        selected = choices[0]
    if canonical == BUFF_PLANE_ACTION.name:
        settings = normalize_buff_plane_parameters(parameters)
        return {"policy_action_id":58, "io_variant":selected, "parameters":settings}
    parameter_module = action_parameters.BY_NAME.get(canonical)
    if parameter_module in block_parameters.MODULES:
        if selected is None:
            selected = parameter_module.variant_for_name(raw_name)
        selected_parameters = ({} if selected is None and parameters is None and source_choices is not None and len(source_choices) > 1
                               else parameter_module.resolve_parameters(raw_name, io_variant or selected, parameters))
    elif parameter_module is not None:
        old_parameters = parameter_module.parameters_from_name(raw_name)
        alias_parameters = parameter_module.parameters_from_name(io_variant)
        if old_parameters is not None and alias_parameters is not None and old_parameters != alias_parameters:
            raise ValueError("legacy parameter-bearing name conflicts with the source ID")
        inferred = old_parameters or alias_parameters
        if parameters is not None and inferred is not None and parameter_module.normalize_parameters(parameters) != inferred:
            raise ValueError("explicit parameters conflict with a historical configuration")
        selected_parameters = parameter_module.normalize_parameters(parameters if parameters is not None else inferred)
    else:
        if parameters not in (None, {}):
            raise ValueError("this algorithm has no registered input parameters")
        selected_parameters = {}
    result = {"policy_action_id": POLICY_ACTION_BY_NAME[canonical].action_id,
              "io_variant": selected, "parameters": selected_parameters}
    if canonical in (optimal_rle.ALGORITHM_NAME, optimal_rle.OLD_SPRINTZ_VARIANT) and registry_version != ACTION_REGISTRY_VERSION:
        result.update(migration_kind="implementation_replacement",
                      source_registry_version=registry_version, source_action_id=action_id,
                      requires_remeasurement=True)
    return result

_EXACT_NATIVE_PIPELINE_KEY: dict[tuple[str, ...], str] = {}
for _method, _algorithm_key in EXACT_NATIVE_METHOD_KEYS.items():
    _pipeline_names = tuple(
        action.name for action in exact_native_lz_action_pipeline(_algorithm_key)
    )
    _EXACT_NATIVE_PIPELINE_KEY.setdefault(_pipeline_names, _algorithm_key)


def encode_registered_exact_action_pipeline(
    data: bytes,
    action_names: Sequence[str],
    *,
    logical_type: str = "bytes",
    max_point: int | None = None,
    io_variants: Sequence[str | None] | None = None,
    parameters: Sequence[dict[str, int] | None] | None = None,
) -> bytes:
    """Execute a legal exact-byte pipeline selected by action names."""

    if any(block_parameters.module_for_name(name) is not None or name in {BUFF_FOR_ACTION.name, BUFF_PLANE_ACTION.name} for name in action_names):
        import unified_action_runtime as uar
        kind = {"int": "int64", "integer": "int64", "float": "float64"}.get(logical_type, logical_type)
        source = uar.ActionData.bytes(data) if kind == "bytes" else uar.ActionData(kind, bytes(data), len(data)//8)
        result = uar.run_actions(action_names, source, io_variants=io_variants, parameters=parameters, max_point=max_point)
        return uar._unpack_result(result)[1]

    variants = tuple(io_variants) if io_variants is not None else (None,) * len(action_names)
    params = tuple(parameters) if parameters is not None else (None,) * len(action_names)
    if len(variants) != len(action_names) or len(params) != len(action_names):
        raise ValueError("one I/O variant and parameter record is required per action")
    selected = []
    for name, variant, settings in zip(action_names, variants, params):
        canonical = ACTION_CANONICAL_NAMES.get(name, name)
        parameter_module = action_parameters.module_for_name(canonical)
        if parameter_module is not None:
            named_parameters = parameter_module.parameters_from_name(name)
            variant_parameters = parameter_module.parameters_from_name(variant)
            if named_parameters is not None and variant_parameters is not None and named_parameters != variant_parameters:
                raise ValueError("historical action and I/O variant parameters conflict")
            historical = named_parameters or variant_parameters
            if variant is not None and variant != parameter_module.ALGORITHM_NAME and parameter_module.parameters_from_name(variant) is None:
                raise ValueError("invalid I/O variant for parameterized stage")
            if settings is not None and historical is not None and parameter_module.normalize_parameters(settings) != historical:
                raise ValueError("parameters conflict with the historical selector")
            selected.append(parameter_module.legacy_action_name(settings if settings is not None else historical))
            continue
        if settings not in (None, {}):
            raise ValueError(f"{name} has no registered input parameters")
        if variant is None and name in (bitpack.ALGORITHM_NAME, minimum.ALGORITHM_NAME):
            variant = bitpack.PLAIN_VARIANT if name == bitpack.ALGORITHM_NAME else minimum.PLAIN_VARIANT
        if variant is None:
            if name in algorithm_io.MERGED_NAMES:
                raise ValueError(f"specify an I/O variant for {name}")
            selected.append(name)
            continue
        variant = LEGACY_206_ACTION_CANONICAL_NAMES.get(variant, variant)
        variant = algorithm_io.LEGACY_VARIANT_TO_CURRENT.get(variant, variant)
        if variant not in POLICY_IO_VARIANTS.get(canonical, ()):
            raise ValueError(f"I/O variant does not belong to {name}")
        selected.append(variant)
    action_names = tuple(selected)
    if tuple(action_names) == (minimum.PLAIN_VARIANT,):
        if logical_type not in {"int", "integer", "int64", "uint64", "bytes"} or len(data) % 8:
            raise ValueError("minimum FOR requires signed int64 or 8-byte-aligned uint64 LE bytes")
        return _cpp_action_run("minimum_for_cli", "encode-int64" if logical_type in {"int", "integer", "int64"} else "encode-uint64", data)
    if tuple(action_names) == (bitpack.PLAIN_VARIANT,):
        if logical_type not in {"int", "integer", "uint64", "bytes"} or len(data) % 8:
            raise ValueError("max-value-width requires nonnegative int64 or uint64 LE bytes aligned to 8 bytes")
        if logical_type in {"int", "integer"} and any(value[0] < 0 for value in struct.iter_unpack("<q", data)):
            raise ValueError("max-value-width does not accept negative signed integers")
        return _cpp_action_run("max_value_bitpack_cli", "encode", data)
    if tuple(action_names) == (cpp_migration.DICTIONARY,):
        if logical_type not in {"int", "integer", "float"}:
            raise ValueError("words64 dictionary requires int64 or float64 input")
        return _cpp_action_run("dictionary64_action_cli", "encode", data)
    names = tuple(name if action_parameters.parameters_from_name(name) is not None else
                  CANONICAL_TO_LEGACY_NAME.get(name, name) for name in action_names)
    if (len(names) == 4 and names[0].startswith(_NATIVE_WINDOW_PREFIX)
            and lz_parameters.parameters_from_name(names[1]) is not None
            and names[2] == _NATIVE_PARSE_ACTION and names[3].startswith(_NATIVE_TOKEN_PREFIX)):
        for name in names:
            data = encode_exact_native_lz_stage(name, data)
        return data
    if names in (ELF64_ACTIONS, ELF64_PLUS_ACTIONS):
        if logical_type != "float":
            raise ValueError("Elf64 action sequence requires float64 input")
        return elf64_exact_stages.xor_forward(elf64_exact_stages.eraser_forward(data),
                                             "Elf+" if names == ELF64_PLUS_ACTIONS else "Elf")
    if names == CORAD_LOSSLESS_ACTIONS:
        if logical_type not in {"int", "integer", "float"}:
            raise ValueError("CORAD-lossless sequence requires int64 or float64 input")
        kind = "int64" if logical_type in {"int", "integer"} else "float64"
        for name in names:
            data = corad_lossless_stages.forward(name, data, kind)
            kind = "bytes"
        return data
    if names == FLOAT_SCALING_THEN_INT_RLE_ACTIONS:
        if logical_type != "float":
            raise ValueError("scaling -> int64 RLE requires logical_type=float")
        if (
            not isinstance(max_point, int)
            or max_point < 0
            or max_point > RLE_MAX_POINT_LIMIT
        ):
            raise ValueError("scaling -> int64 RLE requires max_point in [0,18]")
        return _run_rle_stage_cli("compose", data, max_point)
    if names == TS2DIFF_INT_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("integer TS_2DIFF action sequence requires logical_type=int")
        return encode_exact_ts2diff_action_pipeline(data, "int")
    if names == TS2DIFF_FLOAT_ACTIONS:
        if logical_type != "float":
            raise ValueError("floating TS_2DIFF action sequence requires logical_type=float")
        return encode_exact_ts2diff_action_pipeline(data, "float", max_point=max_point)
    if names == SVB_INT_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("SVB action sequence requires logical_type=int")
        return encode_exact_svb_action_pipeline(data, "int")
    if names == SVB_FLOAT_ACTIONS:
        if logical_type != "float":
            raise ValueError("floating SVB action sequence requires logical_type=float")
        return encode_exact_svb_action_pipeline(
            data, "float", max_point=max_point
        )
    if names == SIMPLE8B_INT_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("integer Simple8b action sequence requires logical_type=int")
        return encode_exact_simple8b_action_pipeline(data, "int")
    if names == SIMPLE8B_FLOAT_ACTIONS:
        if logical_type != "float":
            raise ValueError("floating Simple8b action sequence requires logical_type=float")
        return encode_exact_simple8b_action_pipeline(
            data, "float", max_point=max_point
        )
    if names == RLBE_ACTIONS:
        return encode_exact_p04_action_pipeline(data, "RLBE", logical_type)
    if names == SPRINTZ_ACTIONS:
        return encode_exact_p04_action_pipeline(data, "Sprintz", logical_type)
    if names == DBP_INT_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("integer DBP sequence requires logical_type=int")
        return encode_exact_dbp_action_pipeline(data, "int")
    if names == DBP_FLOAT_ACTIONS:
        if logical_type != "float":
            raise ValueError("floating DBP sequence requires logical_type=float")
        return encode_exact_dbp_action_pipeline(data, "float", max_point=max_point)
    if names == GORILLA_ACTIONS:
        return encode_exact_gorilla_action_pipeline(data, logical_type)
    if names == CHIMP128_ACTIONS:
        return encode_exact_chimp128_action_pipeline(data, logical_type)
    if names == PONGO_ACTIONS:
        if logical_type not in {"int", "integer", "float"}:
            raise ValueError("Pongo exact sequence requires int or float 64-bit lanes")
        return encode_exact_pongo_action_pipeline(data)
    if names == SPDP_ACTIONS:
        return encode_exact_spdp_action_pipeline(data)
    if names == FPC_ACTIONS:
        return encode_exact_fpc_action_pipeline(data)
    if names == PFPC_ACTIONS:
        return encode_exact_pfpc_action_pipeline(data)
    if names == BOS_BOSA_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("full BOSA action sequence requires logical_type=int")
        return _run_bos_stage_cli("compose", data)
    if names == BOS64_INT_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("integer BOS64 sequence requires logical_type=int")
        return encode_exact_bos64_action_pipeline(data, "int")
    if names == BOS64_FLOAT_ACTIONS:
        if logical_type != "float":
            raise ValueError("floating BOS64 sequence requires logical_type=float")
        return encode_exact_bos64_action_pipeline(data, "float", max_point=max_point)
    if names == FASTPFOR_INT_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("integer FastPFor action requires logical_type=int")
        return encode_exact_fastpfor_action_pipeline(data, "int")
    if names == FASTPFOR_FLOAT_ACTIONS:
        if logical_type != "float":
            raise ValueError("floating FastPFor sequence requires logical_type=float")
        return encode_exact_fastpfor_action_pipeline(data, "float", max_point=max_point)
    if names == ELF_ACTIONS:
        if logical_type != "float":
            raise ValueError("Elf sequence requires logical_type=float")
        return encode_exact_elf_action_pipeline(data, "Elf")
    if names == ELF_PLUS_ACTIONS:
        if logical_type != "float":
            raise ValueError("Elf+ sequence requires logical_type=float")
        return encode_exact_elf_action_pipeline(data, "Elf+")
    if names == CAMEL_ACTIONS:
        if logical_type not in {"int", "integer", "float"}:
            raise ValueError("Camel sequence requires int or float input")
        return encode_exact_camel_action_pipeline(data, logical_type)
    if names == SNAPPY_ACTIONS:
        if logical_type != "bytes":
            raise ValueError("Snappy exact sequence requires logical_type=bytes")
        return encode_exact_snappy_action_pipeline(data)
    if names == LZ4_ACTIONS:
        if logical_type != "bytes":
            raise ValueError("LZ4 exact sequence requires logical_type=bytes")
        return encode_exact_lz4_action_pipeline(data)
    if names == ALP_MAIN_ACTIONS:
        if logical_type != "float":
            raise ValueError("ALP exact sequence requires logical_type=float")
        return encode_exact_alp_action_pipeline(data)
    if names == REGER_ACTIONS:
        if logical_type not in {"int", "integer"}:
            raise ValueError("REGER exact sequence requires logical_type=int")
        return encode_exact_reger_action_pipeline(data)
    general_exact_pipelines = {
        BROTLI_EXACT_ACTIONS,
        ZSTD_EXACT_ACTIONS,
        BZIP2_EXACT_ACTIONS,
        GZIP_EXACT_ACTIONS,
        ZLIB_EXACT_ACTIONS,
        XZ_EXACT_ACTIONS,
        *BLOSC_EXACT_ACTIONS_BY_KIND.values(),
        *BLOSC2_EXACT_ACTIONS_BY_KIND.values(),
    }
    if names in general_exact_pipelines:
        return encode_exact_general_residual_action_pipeline(data, names)
    if names == FPZIP_ACTIONS:
        if logical_type != "float":
            raise ValueError("FPZIP exact sequence requires logical_type=float")
        return encode_exact_fpzip_action_pipeline(data)
    for name in names:
        action = ALL_CONCRETE_ACTION_BY_NAME.get(name)
        if not isinstance(action, ExactByteActionSpec):
            raise ValueError(f"not an exact-byte action: {name}")
    if (len(names) == 4 and all(name in EXACT_NATIVE_ACTION_BY_NAME for name in names)
            and tuple(exact_native_lz_stage_role(name) for name in names)
            == ("window", "match", "parse", "token_format")):
        for name in names:
            data = encode_exact_native_lz_stage(name, data)
        return data
    algorithm_key = _EXACT_NATIVE_PIPELINE_KEY.get(names)
    if algorithm_key is not None:
        return encode_exact_native_lz_pipeline(data, algorithm_key)
    atomic_method_by_name = {
        "op.entropy_huffman.project_huffman_bytes.v1": ("Huffman", "bytes", None),
        "op.entropy_range.project_arithmetic_bytes.v1": ("Arithmetic", "bytes", None),
        "op.entropy_fse.project_rans_bytes.v1": ("ANS", "bytes", None),
        "op.dictionary.project_int64.v1": ("Dict.", "int", None),
        "op.dictionary.project_float64.v1": ("Dict.", "float", None),
        "op.run_length.tsfile_int64.v1": ("RLE", "int", None),
    }
    if len(names) == 1 and names[0] in atomic_method_by_name:
        method, required_type, required_max_point = atomic_method_by_name[names[0]]
        if logical_type not in {required_type, "integer" if required_type == "int" else required_type}:
            raise ValueError(
                f"{names[0]} requires logical_type={required_type}, got {logical_type}"
            )
        if required_max_point is not None and max_point != required_max_point:
            raise ValueError(
                f"{names[0]} requires max_point={required_max_point}, got {max_point}"
            )
        return encode_exact_atomic_method(
            data,
            method,
            required_type,
            max_point=required_max_point,
        )
    raise ValueError(f"illegal or incomplete exact-byte action sequence: {names}")

ACTION_MAGIC = b"OPA1"
ACTION_VERSION = 1
CHIMP_HISTORY_SIZE = 128
CHIMP_HISTORY_LOG2 = 7
CHIMP_TRAILING_THRESHOLD = 6 + CHIMP_HISTORY_LOG2
CHIMP_KEY_MASK = (1 << (CHIMP_TRAILING_THRESHOLD + 1)) - 1
MASK64 = (1 << 64) - 1


def _load_catalog() -> dict[str, Any]:
    payload = json.loads(PIPELINE_CATALOG.read_text(encoding="utf-8"))
    pipelines = payload.get("pipelines")
    if not isinstance(pipelines, dict):
        raise ValueError(f"invalid method pipeline catalog: {PIPELINE_CATALOG}")
    return payload


def _pipeline_variants(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, list):
        return [("default", value)]
    if isinstance(value, dict):
        variants = []
        for logical_type, pipeline in value.items():
            if not isinstance(logical_type, str) or not isinstance(pipeline, list):
                raise ValueError(f"invalid typed pipeline: {value!r}")
            variants.append((logical_type, pipeline))
        return variants
    raise ValueError(f"invalid pipeline declaration: {value!r}")


def _operator_ids_in_node(node: Any) -> tuple[str, ...]:
    if isinstance(node, str):
        return (node,)
    if isinstance(node, dict) and set(node) == {"any_of"} and isinstance(node["any_of"], list):
        if not node["any_of"] or not all(isinstance(value, str) for value in node["any_of"]):
            raise ValueError(f"invalid any_of pipeline node: {node!r}")
        return tuple(node["any_of"])
    raise ValueError(f"invalid pipeline node: {node!r}")


def _action_name(method: str, operator_id: str) -> str:
    if operator_id in RETIRED_REFERENCE_OPERATOR_IDS:
        raise ValueError(f"{operator_id} has no active generic reference action")
    # The historical generic integer-code reference combined signed ZigZag and
    # ULEB128.  Current coarse reference recipes resolve this semantic node to
    # the unsigned A023 stage only; signed producers must list A046 separately.
    if operator_id == "integer_code":
        return ULEB128_RAW_ACTION.name
    return METHOD_ACTION_OVERRIDES.get(method, {}).get(
        operator_id, DEFAULT_ACTION_BY_OPERATOR_ID[operator_id].name
    )


def _action_node(method: str, node: Any) -> Any | None:
    operator_ids = _operator_ids_in_node(node)
    for operator_id in operator_ids:
        if operator_id not in ACTIONS_BY_OPERATOR_ID:
            raise ValueError(f"pipeline references unknown operator {operator_id!r}")
    if isinstance(node, str):
        if node in RETIRED_REFERENCE_OPERATOR_IDS:
            return None
        return _action_name(method, node)
    names = [_action_name(method, operator_id) for operator_id in operator_ids
             if operator_id not in RETIRED_REFERENCE_OPERATOR_IDS]
    return {"any_of": names} if names else None


def _active_reference_pipeline(method: str, pipeline: Sequence[Any]) -> list[Any]:
    """Translate semantic stages while omitting retired identity placeholders."""
    return [translated for node in pipeline
            if (translated := _action_node(method, node)) is not None]


def algorithm_action_pipelines() -> dict[str, Any]:
    """Return all catalog pipelines using primitive action names.

    Type-specific pipelines retain their type keys.  An unresolved source
    declaration such as LZSSE's ``any_of`` node remains explicit instead of
    being silently converted to an arbitrary executable pipeline.
    """

    catalog = _load_catalog()["pipelines"]
    output: dict[str, Any] = {}
    for method, declaration in catalog.items():
        if isinstance(declaration, list):
            output[method] = _active_reference_pipeline(method, declaration)
        else:
            output[method] = {
                logical_type: _active_reference_pipeline(method, pipeline)
                for logical_type, pipeline in declaration.items()
            }
    return output


def benchmark_86_action_pipelines() -> dict[str, Any]:
    """Return the exact 28 specialist + 58 general benchmark name surface."""

    all_pipelines = algorithm_action_pipelines()
    output: dict[str, Any] = {}
    for display_name in BENCHMARK_86_METHODS:
        catalog_name = METHOD_ALIASES.get(display_name, display_name)
        if catalog_name not in all_pipelines:
            raise KeyError(f"benchmark method {display_name!r} has no pipeline entry {catalog_name!r}")
        output[display_name] = all_pipelines[catalog_name]
    return output


def benchmark_86_exact_action_pipelines() -> dict[str, Any]:
    """Return the strongest currently executable pipeline for each of the 86.

    Methods backed by the project's native-LZ reference use its newly split
    real stages.  Atomic operator algorithms use their exact typed operator.
    Remaining methods retain the older reference/proxy pipeline and therefore
    carry no exact compression-ratio claim.
    """

    output = benchmark_86_action_pipelines()
    registered_by_name = {action.name: action for action in EXACT_ACTIONS}
    for method, key in EXACT_NATIVE_METHOD_KEYS.items():
        output[method] = [
            registered_by_name[action.name].name
            for action in exact_native_lz_action_pipeline(key)
        ]
    output["Huffman"] = list(exact_atomic_action_pipeline("Huffman", "bytes"))
    output["Arithmetic"] = list(exact_atomic_action_pipeline("Arithmetic", "bytes"))
    output["ANS"] = list(exact_atomic_action_pipeline("ANS", "bytes"))
    output["Dict."] = {
        "int": list(exact_atomic_action_pipeline("Dict.", "int")),
        "float": list(exact_atomic_action_pipeline("Dict.", "float")),
    }
    output["RLE"] = {
        "int": list(exact_atomic_action_pipeline("RLE", "int")),
        "float_max_point_0_to_18": list(
            exact_atomic_action_pipeline("RLE", "float", max_point=0)
        ),
    }
    output["2DIFF"] = {
        "int": list(TS2DIFF_INT_ACTIONS),
        "float": list(TS2DIFF_FLOAT_ACTIONS),
    }
    output["SVB"] = {
        "int": list(SVB_INT_ACTIONS),
        "float": list(SVB_FLOAT_ACTIONS),
    }
    output["Simple8b"] = {
        "int": list(SIMPLE8B_INT_ACTIONS),
        "float": list(SIMPLE8B_FLOAT_ACTIONS),
    }
    output["RLBE"] = {
        "int": list(RLBE_ACTIONS),
        "float": list(RLBE_ACTIONS),
    }
    output["Sprintz"] = {
        "int": list(SPRINTZ_ACTIONS),
        "float": list(SPRINTZ_ACTIONS),
    }
    output["DBP"] = {
        "int": list(DBP_INT_ACTIONS),
        "float": list(DBP_FLOAT_ACTIONS),
    }
    output["Gorilla"] = {
        "int": list(GORILLA_ACTIONS),
        "float": list(GORILLA_ACTIONS),
    }
    output["Chimp128"] = {
        "int": list(CHIMP128_ACTIONS),
        "float": list(CHIMP128_ACTIONS),
    }
    # benchmark_86_combinations.py intentionally routes both public names
    # through the same TSFile Chimp encoder/decoder.  Reuse the byte-identical
    # action implementation instead of manufacturing method-scoped aliases.
    output["Chimp"] = {
        "int": list(CHIMP128_ACTIONS),
        "float": list(CHIMP128_ACTIONS),
    }
    output["Pongo"] = {
        "int": list(PONGO_ACTIONS),
        "float": list(PONGO_ACTIONS),
    }
    output["BOS"] = {
        "int": list(BOS64_INT_ACTIONS),
        "float": list(BOS64_FLOAT_ACTIONS),
    }
    output["SPDP"] = list(SPDP_ACTIONS)
    output["FPC"] = list(FPC_ACTIONS)
    output["PFPC"] = list(PFPC_ACTIONS)
    output["Elf"] = {"float": list(ELF64_ACTIONS)}
    output["Elf+"] = {"float": list(ELF64_PLUS_ACTIONS)}
    output["Camel"] = {"int": list(CAMEL_ACTIONS), "float": list(CAMEL_ACTIONS)}
    output["FPZIP"] = {"float": list(FPZIP_ACTIONS)}
    output["LECO"] = {"int": list(LECO_ACTIONS)}
    output["ALP"] = {"float": list(ALP_MAIN_ACTIONS)}
    output["REGER"] = {"int": list(REGER_ACTIONS)}
    output["BUFF"] = {kind: list(BUFF_NATIVE_ACTIONS) for kind in ("int", "float")}
    output["CORAD"] = {kind: list(CORAD_LOSSLESS_ACTIONS) for kind in ("int", "float")}
    output["FastPFor"] = {"int": list(FASTPFOR_INT_ACTIONS), "float": list(FASTPFOR_FLOAT_ACTIONS)}
    output["Brotli"] = list(BROTLI_EXACT_ACTIONS)
    output["Zstd"] = list(ZSTD_EXACT_ACTIONS)
    output["Blosc"] = {
        kind: list(actions) for kind, actions in BLOSC_EXACT_ACTIONS_BY_KIND.items()
    }
    output["Blosc2"] = {
        kind: list(actions) for kind, actions in BLOSC2_EXACT_ACTIONS_BY_KIND.items()
    }
    output["Bzip2"] = list(BZIP2_EXACT_ACTIONS)
    output["gzip"] = list(GZIP_EXACT_ACTIONS)
    output["Zlib"] = list(ZLIB_EXACT_ACTIONS)
    output["XZ"] = list(XZ_EXACT_ACTIONS)
    output["LZ4"] = list(LZ4_ACTIONS)
    output["Snappy"] = list(SNAPPY_ACTIONS)
    return output


def benchmark_86_legacy_action_pipelines() -> dict[str, Any]:
    """Use frozen Python benchmark backends where they differ from C++ codecs."""
    output = benchmark_86_exact_action_pipelines()
    for method in ("FastPFor", "LECO"):
        output[method] = {kind: list(legacy_numeric_stages.SCALED_ACTIONS)
                          for kind in ("int", "float")}
    output["ALP"] = {"int": list(legacy_numeric_stages.SCALED_ACTIONS),
                     "float": list(legacy_numeric_stages.ALP_ACTIONS)}
    output["Snappy"] = list(SNAPPY_LEGACY_ACTIONS)
    output["REGER"] = {kind: list(REGER_LEGACY_ACTIONS) for kind in ("int", "float")}
    output["Blosc2"] = list(BLOSC2_LEGACY_ACTIONS)
    return output


def benchmark_86_canonical_action_pipelines() -> dict[str, Any]:
    """Algorithm names; pair with benchmark_86_io_variant_pipelines to execute."""
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(chain) for key, chain in value.items()}
        return [ACTION_CANONICAL_NAMES.get(name, name) for name in value]
    return {method: canonical(chain)
            for method, chain in benchmark_86_legacy_action_pipelines().items()}


def benchmark_86_io_variant_pipelines() -> dict[str, Any]:
    """I/O profiles parallel to algorithm names; LZ bounds are separate parameters."""
    def variants(value):
        if isinstance(value, dict):
            return {key: variants(chain) for key, chain in value.items()}
        result = []
        for index, name in enumerate(value):
            old = LEGACY_206_ACTION_CANONICAL_NAMES.get(name, name)
            previous = value[index - 1] if index else None
            upgraded = optimal_rle.upgrade_variant(old, previous)
            result.append(algorithm_io.LEGACY_VARIANT_TO_CURRENT.get(upgraded, upgraded))
        return result
    return {method: variants(chain)
            for method, chain in benchmark_86_legacy_action_pipelines().items()}


def benchmark_86_action_parameters() -> dict[str, Any]:
    """Exact historical bounds, including noncanonical maxima, for saved recipes."""
    def parameters(value):
        if isinstance(value, dict):
            return {key: parameters(chain) for key, chain in value.items()}
        return [action_parameters.parameters_from_name(name) or {} for name in value]
    return {method: parameters(chain)
            for method, chain in benchmark_86_legacy_action_pipelines().items()}


def encode_exact_benchmark_method(
    data: bytes,
    method: str,
    *,
    logical_type: str = "bytes",
    max_point: int | None = None,
) -> bytes:
    """Execute a benchmark method only when it has a real exact action path."""

    if method in EXACT_NATIVE_METHOD_KEYS:
        return encode_exact_native_lz_pipeline(data, EXACT_NATIVE_METHOD_KEYS[method])
    if method == "2DIFF":
        return encode_exact_ts2diff_action_pipeline(
            data, logical_type, max_point=max_point
        )
    if method == "SVB":
        return encode_exact_svb_action_pipeline(
            data, logical_type, max_point=max_point
        )
    if method == "Simple8b":
        return encode_exact_simple8b_action_pipeline(
            data, logical_type, max_point=max_point
        )
    if method == 'RLE':
        if logical_type in ('int', 'integer'):
            return optimal_rle.forward(data, 'int64', optimal_rle.WORDS_VARIANT)
        if logical_type == 'float':
            if max_point is None: raise ValueError('RLE scaling profile requires max_point')
            scaled = _run_rle_stage_cli('scale-forward', data, max_point)
            return optimal_rle.forward(scaled, 'bytes', optimal_rle.SCALED_VARIANT)
        raise ValueError('RLE method requires int64 or float64 input')
    if method == 'RLBE':
        if logical_type not in ('int', 'integer', 'float'):
            raise ValueError('RLBE method requires int64 or float64 raw bits')
        deltas = _run_p04_stage_cli('rlbe-delta-forward', data)
        return optimal_rle.forward(deltas, 'bytes', optimal_rle.RLBE_VARIANT)
    if method == 'Sprintz':
        if logical_type not in ('int', 'integer', 'float'):
            raise ValueError('Sprintz method requires int64 or float64 input')
        packet = _run_p04_stage_cli('sprintz-predictor-forward-double' if logical_type == 'float' else 'sprintz-predictor-forward', data)
        packet = _run_p04_stage_cli('sprintz-zigzag-forward', packet)
        return _run_p04_stage_cli('sprintz-bitpack-forward-current', packet)
    if method == "DBP":
        return encode_exact_dbp_action_pipeline(
            data, logical_type, max_point=max_point
        )
    if method == "Gorilla":
        return encode_exact_gorilla_action_pipeline(data, logical_type)
    if method in {"Chimp", "Chimp128"}:
        return encode_exact_chimp128_action_pipeline(data, logical_type)
    if method == "Pongo":
        return encode_exact_pongo_action_pipeline(data)
    if method == "Camel":
        return encode_exact_camel_action_pipeline(data, logical_type)
    if method == "FastPFor":
        return encode_exact_fastpfor_action_pipeline(data, logical_type, max_point=max_point)
    if method == "BOS":
        return encode_exact_bos64_action_pipeline(
            data, logical_type, max_point=max_point
        )
    if method == "SPDP":
        return encode_exact_spdp_action_pipeline(data)
    if method == "FPC":
        return encode_exact_fpc_action_pipeline(data)
    if method == "PFPC":
        return encode_exact_pfpc_action_pipeline(data)
    if method in {"Elf", "Elf+"}:
        if logical_type != "float":
            raise ValueError(f"{method} exact eraser/XOR path requires float input")
        return elf64_exact_stages.encode(data, method)
    if method == "CORAD":
        if logical_type not in {"float", "int", "integer"}:
            raise ValueError("CORAD-lossless requires int64 or float64 input")
        return corad_lossless_stages.encode(data, "float64" if logical_type == "float" else "int64")
    if method == "FPZIP":
        if logical_type != "float":
            raise ValueError("FPZIP exact predictor/range path requires float input")
        return encode_exact_fpzip_action_pipeline(data)
    if method == "LECO":
        if logical_type not in {"int", "integer"}:
            raise ValueError("LECO exact predictor/bitpack path requires int input")
        return encode_exact_leco_action_pipeline(data)
    if method == "ALP":
        if logical_type != "float":
            raise ValueError("ALP exact prefix/residual path requires float input")
        return encode_exact_alp_action_pipeline(data)
    if method == "REGER":
        if logical_type not in {"int", "integer"}:
            raise ValueError("REGER exact prefix/residual path requires int input")
        return encode_exact_reger_action_pipeline(data)
    if method == "BUFF":
        if logical_type not in {"float", "int", "integer"}:
            raise ValueError("BUFF native raw64 path requires 64-bit lane input")
        return _run_exact_stage_cli(BUFF_NATIVE_STAGE_CLI, "BUFF native", "compose", data)
    if method == "LZ4":
        return encode_exact_lz4_action_pipeline(data)
    if method == "Snappy":
        return encode_exact_snappy_action_pipeline(data)
    general_pipelines = {
        "Brotli": BROTLI_EXACT_ACTIONS,
        "Zstd": ZSTD_EXACT_ACTIONS,
        "Bzip2": BZIP2_EXACT_ACTIONS,
        "gzip": GZIP_EXACT_ACTIONS,
        "Zlib": ZLIB_EXACT_ACTIONS,
        "XZ": XZ_EXACT_ACTIONS,
    }
    if method in general_pipelines:
        if method == 'Bzip2':
            prefix = _run_general_reference_chain(data, ('bwt-forward', 'mtf-forward'))
            prefix = optimal_rle.forward(prefix, 'bytes', optimal_rle.BYTES_VARIANT)
            prefix = _run_general_reference_chain(prefix, ('huffman-forward',))
            return _run_general_codec_residual_stage_cli('bzip2-optimal-forward', prefix)
        return encode_exact_general_residual_action_pipeline(
            data, general_pipelines[method]
        )
    if method in {"Blosc", "Blosc2"}:
        kind = {"bytes": "bytes", "float": "float64", "int": "int64", "integer": "int64"}.get(logical_type)
        if kind is None:
            raise ValueError(f"{method} unsupported logical_type={logical_type!r}")
        pipelines = (
            BLOSC_EXACT_ACTIONS_BY_KIND
            if method == "Blosc"
            else BLOSC2_EXACT_ACTIONS_BY_KIND
        )
        return encode_exact_general_residual_action_pipeline(data, pipelines[kind])
    if method in {"Huffman", "Arithmetic", "ANS", "Dict.", "RLE"}:
        return encode_exact_atomic_method(
            data, method, logical_type, max_point=max_point
        )
    raise KeyError(f"{method!r} still has no exact action pipeline")


def concrete_operator_pipeline(
    method: str,
    *,
    logical_type: str | None = None,
) -> tuple[str, ...]:
    """Resolve one deterministic catalog method to operator IDs.

    Whole algorithm names are accepted only as a convenience for retrieving a
    declared pipeline.  They are never returned as actions.  Ambiguous
    ``any_of`` declarations must first be resolved by an experiment design.
    """

    method = METHOD_ALIASES.get(method, method)
    catalog = _load_catalog()["pipelines"]
    if method not in catalog:
        raise KeyError(f"unknown catalog method {method!r}")
    declaration = catalog[method]
    if isinstance(declaration, dict):
        if logical_type is None:
            if len(declaration) != 1:
                raise ValueError(f"{method} requires logical_type from {sorted(declaration)}")
            logical_type = next(iter(declaration))
        if logical_type not in declaration:
            raise ValueError(f"{method} has no {logical_type!r} pipeline; choose from {sorted(declaration)}")
        declaration = declaration[logical_type]
    output: list[str] = []
    for node in declaration:
        if not isinstance(node, str):
            raise ValueError(f"{method} contains unresolved pipeline alternative {node!r}")
        if node not in ACTIONS_BY_OPERATOR_ID:
            raise ValueError(f"{method} references unknown operator {node!r}")
        output.append(node)
    return tuple(output)


def concrete_action_pipeline(
    method: str,
    *,
    logical_type: str | None = None,
) -> tuple[str, ...]:
    canonical_method = METHOD_ALIASES.get(method, method)
    return tuple(
        _action_name(canonical_method, operator_id)
        for operator_id in concrete_operator_pipeline(canonical_method, logical_type=logical_type)
        if operator_id not in RETIRED_REFERENCE_OPERATOR_IDS
    )


def encode_operator_pipeline(values: Sequence[int] | np.ndarray, action_names: Sequence[str]) -> bytes:
    """Run the bit-identical OPA1 encoder in C++; Python only packs int64 input."""
    names = tuple(action_names)
    if any(name not in ACTION_BY_NAME for name in names):
        raise ValueError("not a primitive action")
    raw = np.asarray(values, dtype="<i8").tobytes()
    return operator_impl._run_reference("encode", raw, names)


def decode_operator_pipeline(encoded: bytes) -> tuple[np.ndarray, tuple[str, ...]]:
    """Decode both OPA1 transforms and their envelope entirely in C++."""
    raw = operator_impl._run_reference("decode", bytes(encoded))
    names = operator_impl._run_reference("actions", bytes(encoded)).decode().splitlines()
    return np.frombuffer(raw, dtype="<i8").copy(), tuple(names)


def encode_catalog_method(
    values: Sequence[int] | np.ndarray,
    method: str,
    *,
    logical_type: str | None = None,
) -> bytes:
    return encode_operator_pipeline(values, concrete_action_pipeline(method, logical_type=logical_type))


def _iter_catalog_variants() -> Iterable[tuple[str, str, list[Any]]]:
    for method, declaration in _load_catalog()["pipelines"].items():
        for logical_type, pipeline in _pipeline_variants(declaration):
            yield method, logical_type, pipeline


def _iter_benchmark_86_variants() -> Iterable[tuple[str, str, list[Any]]]:
    catalog = _load_catalog()["pipelines"]
    for display_name in BENCHMARK_86_METHODS:
        catalog_name = METHOD_ALIASES.get(display_name, display_name)
        if catalog_name not in catalog:
            raise KeyError(f"benchmark method {display_name!r} has no catalog pipeline")
        for logical_type, pipeline in _pipeline_variants(catalog[catalog_name]):
            yield display_name, logical_type, pipeline


def registry() -> dict[str, Any]:
    def reported_migration(action_id: int, version: str) -> dict[str, Any]:
        try:
            return migrate_policy_action(action_id, version)
        except ValueError as exc:
            if str(exc) != "identity Scaling is retired; omit the stage from new recipes":
                raise
            return {
                "migration_kind": "retired_identity_bypass",
                "source_registry_version": version,
                "source_action_id": action_id,
                "replacement_policy_action_ids": [],
            }

    def current_variant_records(policy_name: str) -> list[dict[str, Any]]:
        frozen = algorithm_io.GROUP_BY_NAME.get(policy_name)
        if (frozen is not None
                and tuple(row["name"] for row in frozen["io_variants"])
                == POLICY_IO_VARIANTS[policy_name]):
            return frozen["io_variants"]
        kinds = {
            BUFF_FOR_ACTION.name: (["int64"], ["int64", "uint64"]),
            BUFF_PLANE_ACTION.name: (["uint64"], ["uint8"]),
            LECO_SOURCE_EXACT_ACTIONS[0].name: (["int64"], ["float64", "int64"]),
            REGER_REFERENCE_ACTIONS[1].name: (["bytes"], ["int64", "uint32"]),
            BOS64_EXACT_ACTIONS[0].name: (["int64", "bytes"], ["int64"]),
            ZIGZAG_RAW_ACTION.name: (["int64"], ["uint64"]),
            LECO_SOURCE_EXACT_ACTIONS[1].name: (["bytes"], ["float64", "uint64"]),
            REGER_REFERENCE_ACTIONS[2].name: (["bytes"], ["uint64", "uint32"]),
            BOS64_EXACT_ACTIONS[1].name: (["bytes"], ["uint64"]),
            ULEB128_RAW_ACTION.name: (["uint64"], ["bytes"]),
            ULEB128_DBP_ACTION.name: (["bytes"], ["bytes"]),
            ULEB128_SNAPPY_ACTION.name: (["bytes"], ["bytes"]),
        }
        records = []
        for variant in POLICY_IO_VARIANTS[policy_name]:
            if variant in algorithm_io.VARIANT_BY_NAME:
                records.append(dict(algorithm_io.VARIANT_BY_NAME[variant]))
                continue
            input_kinds, output_kinds = kinds[variant]
            concrete = ALL_CONCRETE_ACTION_BY_NAME[variant]
            records.append({
                "name": variant,
                "input_kinds": input_kinds,
                "output_kinds": output_kinds,
                "description": concrete.implementation_scope,
                "implementation_source": concrete.implementation_source,
            })
        return records

    method_names_by_operator = {operator_id: set() for operator_id in ACTIONS_BY_OPERATOR_ID}
    method_names_by_action = {action.name: set() for action in PRIMITIVE_ACTIONS}
    method_names_by_action[ULEB128_RAW_ACTION.name] = set()
    variant_count = 0
    declared_stage_alternatives = 0
    unresolved_variants: list[str] = []
    max_pipeline_depth = 0
    for method, logical_type, pipeline in _iter_catalog_variants():
        variant_count += 1
        max_pipeline_depth = max(max_pipeline_depth, len(pipeline))
        unresolved = False
        for node in pipeline:
            operator_ids = _operator_ids_in_node(node)
            declared_stage_alternatives += len(operator_ids)
            unresolved = unresolved or not isinstance(node, str)
            for operator_id in operator_ids:
                if operator_id not in method_names_by_operator:
                    raise ValueError(f"{method} references unknown operator {operator_id!r}")
                method_names_by_operator[operator_id].add(method)
                if operator_id not in RETIRED_REFERENCE_OPERATOR_IDS:
                    method_names_by_action[_action_name(method, operator_id)].add(method)
        if unresolved:
            unresolved_variants.append(f"{method}:{logical_type}")
    methods = _load_catalog()["pipelines"]
    benchmark_86_pipelines = benchmark_86_action_pipelines()
    benchmark_86_exact_pipelines = benchmark_86_exact_action_pipelines()
    benchmark_86_variant_count = sum(1 for _ in _iter_benchmark_86_variants())
    exact_source_methods = {action.name: set() for action in EXACT_ACTIONS}
    exact_source_methods[ULEB128_RAW_ACTION.name].update(
        method_names_by_action[ULEB128_RAW_ACTION.name]
    )
    for method, key in EXACT_NATIVE_METHOD_KEYS.items():
        for action in exact_native_lz_action_pipeline(key):
            exact_source_methods[action.name].add(method)
    for method, logical_types in {
        "Huffman": ("bytes",),
        "Arithmetic": ("bytes",),
        "ANS": ("bytes",),
        "Dict.": ("int", "float"),
        "RLE": ("int",),
    }.items():
        for logical_type in logical_types:
            for action_name in exact_atomic_action_pipeline(method, logical_type):
                exact_source_methods[action_name].add(method)
    for action_name in exact_atomic_action_pipeline(
        "RLE", "float", max_point=0
    ):
        exact_source_methods[action_name].add("RLE")
    for action_name in TS2DIFF_FLOAT_ACTIONS:
        exact_source_methods[action_name].add("2DIFF")
    for action_name in SVB_INT_ACTIONS:
        exact_source_methods[action_name].add("SVB")
    exact_source_methods[DBP_EXACT_ACTIONS[0].name].add("SVB")
    for action_name in SIMPLE8B_FLOAT_ACTIONS:
        exact_source_methods[action_name].add("Simple8b")
    for action_name in RLBE_ACTIONS:
        exact_source_methods[action_name].add("RLBE")
    for action_name in SPRINTZ_ACTIONS:
        exact_source_methods[action_name].add("Sprintz")
    for action_name in DBP_FLOAT_ACTIONS:
        exact_source_methods[action_name].add("DBP")
    for action_name in GORILLA_ACTIONS:
        exact_source_methods[action_name].add("Gorilla")
    for action_name in CHIMP128_ACTIONS:
        exact_source_methods[action_name].update({"Chimp", "Chimp128"})
    for action_name in PONGO_ACTIONS:
        exact_source_methods[action_name].add("Pongo")
    for action_name in SPDP_ACTIONS:
        exact_source_methods[action_name].add("SPDP")
    for action_name in FPC_ACTIONS:
        exact_source_methods[action_name].add("FPC")
    for action_name in PFPC_ACTIONS:
        exact_source_methods[action_name].add("PFPC")
    for action_name in BOS_BOSA_ACTIONS:
        exact_source_methods[action_name].add("BOS-source-BOSA")
    for action_name in BOS64_FLOAT_ACTIONS:
        exact_source_methods[action_name].add("BOS")
    for action in FASTPFOR_COMPOSITE_HELPER_ACTIONS:
        exact_source_methods[action.name].add("FastPFor-source-C++-whole-helper")
    for action_name in FASTPFOR_FLOAT_ACTIONS:
        exact_source_methods[action_name].add("FastPFor-source-C++")
    for action_name in ELF_ACTIONS:
        exact_source_methods[action_name].add("Elf")
    for action_name in ELF_PLUS_ACTIONS:
        exact_source_methods[action_name].add("Elf+")
    for action_name in CAMEL_ACTIONS:
        exact_source_methods[action_name].add("Camel")
    for action_name in SNAPPY_ACTIONS:
        exact_source_methods[action_name].add("Snappy-source-C++")
    for action_name in LZ4_ACTIONS:
        exact_source_methods[action_name].add("LZ4")
    for action_name in GZIP_CONTAINER_ACTIONS:
        exact_source_methods[action_name].add("gzip-container-only")
    for action_name in ZLIB_CONTAINER_ACTIONS:
        exact_source_methods[action_name].add("Zlib-container-only")
    for action_name in XZ_CONTAINER_ACTIONS:
        exact_source_methods[action_name].add("XZ-container-only")
    for action_name in FPZIP_ACTIONS:
        exact_source_methods[action_name].add("FPZIP")
    for action_name in LECO_ACTIONS:
        exact_source_methods[action_name].add("LECO")
    for action_name in ALP_MAIN_ACTIONS:
        exact_source_methods[action_name].add("ALP")
    for action_name in ALP_DICTIONARY_ACTIONS:
        exact_source_methods[action_name].add("ALP-RD-reference")
    for action_name in REGER_ACTIONS:
        exact_source_methods[action_name].add("REGER")
    for action_name in BUFF_ATOMIC_ACTIONS:
        exact_source_methods[action_name].add("BUFF")
    for action_name in BROTLI_REFERENCE_ACTIONS:
        exact_source_methods[action_name].add("Brotli-reference")
    for action_name in ZSTD_REFERENCE_ACTIONS:
        exact_source_methods[action_name].add("Zstd-reference")
    for action_name in _BLOSCLZ_SHARED_ACTIONS:
        exact_source_methods[action_name].update({"Blosc-reference", "Blosc2-reference"})
    for action_name in BLOSC_SHUFFLE_ACTION_BY_KIND.values():
        exact_source_methods[action_name].update({"Blosc-reference", "Blosc2-reference"})
    exact_source_methods[GENERAL_CODEC_REFERENCE_ACTIONS[12].name].add("Blosc-reference")
    exact_source_methods[GENERAL_CODEC_REFERENCE_ACTIONS[13].name].add("Blosc2-reference")
    for action_name in BZIP2_REFERENCE_ACTIONS:
        exact_source_methods[action_name].add("Bzip2-reference")
    exact_source_methods[DEFLATE_REFERENCE_LZ_ACTION].update(
        {"gzip-reference", "Zlib-reference"}
    )
    exact_source_methods[ZSTD_REFERENCE_ACTIONS[1]].update(
        {"Zlib-reference", "XZ-reference"}
    )
    exact_source_methods[GENERAL_CODEC_REFERENCE_ACTIONS[2].name].update(
        {"gzip-reference", "Zlib-reference"}
    )
    exact_source_methods[XZ_REFERENCE_LZ_ACTION].add("XZ-reference")
    exact_source_methods[REFERENCE_RANGE_ACTION].add("XZ-reference")
    exact_source_methods[GZIP_REFERENCE_FRAME_ACTION].add("gzip-reference")
    exact_source_methods[ZLIB_REFERENCE_FRAME_ACTION].add("Zlib-reference")
    exact_source_methods[XZ_REFERENCE_FRAME_ACTION].add("XZ-reference")
    for action_name in BROTLI_EXACT_ACTIONS:
        exact_source_methods[action_name].add("Brotli")
    for action_name in ZSTD_EXACT_ACTIONS:
        exact_source_methods[action_name].add("Zstd")
    for actions in BLOSC_EXACT_ACTIONS_BY_KIND.values():
        for action_name in actions:
            exact_source_methods[action_name].add("Blosc")
    for actions in BLOSC2_EXACT_ACTIONS_BY_KIND.values():
        for action_name in actions:
            exact_source_methods[action_name].add("Blosc2")
    for action_name in BZIP2_EXACT_ACTIONS:
        exact_source_methods[action_name].add("Bzip2")
    for action_name in GZIP_EXACT_ACTIONS:
        exact_source_methods[action_name].add("gzip")
    for action_name in ZLIB_EXACT_ACTIONS:
        exact_source_methods[action_name].add("Zlib")
    for action_name in XZ_EXACT_ACTIONS:
        exact_source_methods[action_name].add("XZ")
    for method in ("FastPFor", "LECO", "ALP", "REGER", "Snappy", "Blosc2"):
        variants = benchmark_86_legacy_action_pipelines()[method]
        for chain in (variants.values() if isinstance(variants, dict) else (variants,)):
            for action_name in chain:
                if action_name in exact_source_methods:
                    exact_source_methods[action_name].add(method + "-legacy-Python")
    for action_name in BUFF_NATIVE_ACTIONS:
        exact_source_methods[action_name].add("BUFF-current-native")
    for actions, method in ((ELF64_ACTIONS, "Elf"), (ELF64_PLUS_ACTIONS, "Elf+"),
                            (CORAD_LOSSLESS_ACTIONS, "CORAD-lossless-v2")):
        for action_name in actions:
            exact_source_methods[action_name].add(method)
    payload = {
        "schema": "primitive-operator-action-space-v1",
        "implementation_directory": {
            "registry_version": implementation_catalog.REGISTRY_VERSION,
            "implementation_count": implementation_catalog.IMPLEMENTATION_COUNT,
            "manifest_sha256": implementation_catalog.MANIFEST_SHA256,
            "runtime_action_recipes": implementation_catalog.inventory()["runtime_action_recipes"],
            "retired_identity_action_names": sorted(RETIRED_IDENTITY_ACTION_NAMES),
            "note": "Equivalence directory count; not a runtime action or executability count.",
        },
        "action_identity_rule": "operator_id + implementation_id; whole algorithm names are pipeline provenance only",
        "production_codec_claim_allowed": (
            "only for benchmark_86_exact_action_pipelines entries that pass the external exact-byte audit"
        ),
        "actions": [
            {
                **asdict(action),
                "execution_domain": "normalized_int64_reference",
                "source_methods": sorted(method_names_by_action[action.name]),
                "operator_family_source_methods": sorted(method_names_by_operator[action.operator_id]),
            }
            for action in PRIMITIVE_ACTIONS
        ] + [
            {
                **asdict(action),
                "execution_domain": (
                    "reference_codec_byte_stage"
                    if action.name in REFERENCE_STAGE_ACTION_NAMES
                    else "exact_codec_byte_stage"
                ),
                "source_methods": sorted(exact_source_methods[action.name]),
                "operator_family_source_methods": sorted(exact_source_methods[action.name]),
            }
            for action in EXACT_ACTIONS
        ],
        "counts": {
            "compression_operator_actions": len(POLICY_ACTIONS),
            "callable_action_implementation_records": len(ALL_CONCRETE_ACTION_IMPLEMENTATIONS),
            "reference_or_proxy_actions": len(PRIMITIVE_ACTIONS),
            "exact_byte_stage_actions": (
                len(EXACT_ACTIONS)
                - len(POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES)
                - len(REFERENCE_STAGE_ACTION_NAMES)
            ),
            "reference_byte_stage_actions": len(REFERENCE_STAGE_ACTION_NAMES),
            "non_atomic_experimental_helpers": len(POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES),
            "exact_native_lz_stage_actions": len(_UNREGISTERED_EXACT_NATIVE_ACTIONS),
            "exact_atomic_operator_actions": len(EXACT_ATOMIC_ACTIONS),
            "policy_control_actions_including_stop": len(POLICY_ACTIONS) + 1,
            "frozen_policy_id_slots_including_stop": len(FROZEN_POLICY_ACTIONS) + 1,
            "retired_policy_id_slots": len(FROZEN_POLICY_ACTIONS) - len(POLICY_ACTIONS),
            "operator_families": len(ACTIONS_BY_OPERATOR_ID),
            "catalog_methods": len(methods),
            "typed_pipeline_variants": variant_count,
            "declared_stage_alternatives": declared_stage_alternatives,
            "max_declared_pipeline_depth": max_pipeline_depth,
            "unresolved_pipeline_variants": len(unresolved_variants),
            "benchmark_86_methods": len(benchmark_86_pipelines),
            "benchmark_86_typed_pipeline_variants": benchmark_86_variant_count,
            "benchmark_86_taxonomy_composable": len(benchmark_86_pipelines),
        },
        "unresolved_pipeline_variants": unresolved_variants,
        "policy_excluded_non_atomic_action_names": sorted(
            POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES
        ),
        "policy_actions": [
            {"policy_action_id": action.action_id,
             "concrete_action_id": POLICY_TO_CONCRETE_ACTION_ID[action.action_id],
             "name": action.name}
            for action in POLICY_ACTIONS
        ],
        "policy_stop_action_id": STOP_ACTION_ID,
        "algorithm_action_pipelines": algorithm_action_pipelines(),
        "benchmark_86_action_pipelines": benchmark_86_pipelines,
        "benchmark_86_exact_action_pipelines": benchmark_86_exact_pipelines,
        "benchmark_86_legacy_action_pipelines": benchmark_86_legacy_action_pipelines(),
        "benchmark_86_global_best_general_keys": BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS,
        "required_decompositions": {
            "fastpfor": ["scaling", "bitpack_int", "integer_code"],
            "ALP_main": ["scaling", "frame_of_reference", "bitpack_int"],
            "ALP_RD_alternative": ["dictionary"],
            "LECO": ["regression_predictor", "zigzag", "bitpack_int"],
            "REGER_reference": ["value_reorder", "delta", "zigzag", "frame_of_reference", "bitpack_int"],
            "BUFF": ["scaling", "block_transform", "bitpack_int"],
            "CHIMP": ["xor", "trailing_zero_pack"],
            "Chimp128": ["xor", "trailing_zero_pack"],
            "GORILLA": ["xor", "trailing_zero_pack"],
        },
        "legacy_macro_replacements": {
            "catalog.specialist.fastpfor": algorithm_action_pipelines()["fastpfor"],
            "catalog.specialist.alp": algorithm_action_pipelines()["ALP"],
            "catalog.specialist.leco": algorithm_action_pipelines()["LECO"],
            "catalog.specialist.chimp": algorithm_action_pipelines()["CHIMP"],
            "tsfile.chimp128": algorithm_action_pipelines()["Chimp128"],
            "tsfile.gorilla64": algorithm_action_pipelines()["GORILLA"],
        },
    }
    legacy_records = payload.pop("actions")
    by_name = {row["name"]: row for row in legacy_records}
    payload["legacy_replay_records"] = [
        {**row, "implementation_source": ALL_CONCRETE_ACTION_BY_NAME[row["name"]].implementation_source,
         "canonical_name": ACTION_CANONICAL_NAMES.get(row["name"]),
         "is_alias": row["name"] in ACTION_ALIASES}
        for row in legacy_records
    ]
    payload["actions"] = [
        {**by_name[CANONICAL_TO_LEGACY_NAME[spec.name]], **asdict(spec),
         "policy_action_id": spec.action_id,
         "io_variants": current_variant_records(spec.name),
         "parameter_schema": action_parameters.BY_NAME[spec.name].parameter_schema() if spec.name in action_parameters.BY_NAME else None,
         "aliases": sorted(old for old, new in ACTION_ALIASES.items() if new == spec.name)}
        for spec in POLICY_ACTIONS
    ]
    payload["schema"] = "primitive-operator-action-space-v6"
    payload["policy_registry_version"] = ACTION_REGISTRY_VERSION
    payload["action_aliases"] = ACTION_ALIASES
    payload["implementation_replacements"] = {
        name: {'current_algorithm': ACTION_CANONICAL_NAMES[name],
               'current_io_variant': None if name == 'op.run_length.value_count.cpp_v2' else optimal_rle.upgrade_variant(name),
               'historical_replay': 'original version/name/wire retained',
               'requires_remeasurement': True}
        for name in _REPLACED_RLE_NAMES
    }
    # These integer tables identify preserved numeric slots only. Invocation
    # migration below is authoritative because retired stages can map to an
    # empty recipe rather than to an active action ID.
    payload["legacy_225_policy_id_to_frozen_slot"] = LEGACY_225_POLICY_ID_TO_CURRENT
    payload["legacy_206_policy_id_to_frozen_slot"] = LEGACY_206_POLICY_ID_TO_CURRENT
    payload["legacy_182_policy_id_to_frozen_slot"] = LEGACY_182_POLICY_ID_TO_CURRENT
    payload["legacy_156_policy_id_to_frozen_slot"] = LEGACY_156_POLICY_ID_TO_CURRENT
    payload["legacy_137_policy_id_to_frozen_slot"] = LEGACY_137_POLICY_ID_TO_CURRENT
    payload["legacy_129_policy_id_to_frozen_slot"] = LEGACY_129_POLICY_ID_TO_CURRENT
    payload["legacy_127_v1_invocation_migration"] = {
        i: reported_migration(i, LEGACY_ACTION_REGISTRY_VERSION)
        for i in range(STOP_ACTION_ID + 1)
    }
    payload["legacy_129_invocation_migration"] = {i: reported_migration(i, "cpp-canonical-129-v1") for i in range(130)}
    payload["legacy_137_invocation_migration"] = {i: reported_migration(i, "cpp-canonical-137-v1") for i in range(138)}
    payload["legacy_140_policy_id_to_frozen_slot"] = LEGACY_140_POLICY_ID_TO_CURRENT
    payload["legacy_140_invocation_migration"] = {i: reported_migration(i, "cpp-canonical-140-v1") for i in range(141)}
    payload["legacy_143_policy_id_to_frozen_slot"] = LEGACY_143_POLICY_ID_TO_CURRENT
    payload["legacy_206_invocation_migration"] = {
        i: reported_migration(i, "cpp-canonical-206-v1") for i in range(207)}
    payload["legacy_143_invocation_migration"] = {
        i: reported_migration(i, "cpp-canonical-143-v1") for i in range(144)}
    payload["legacy_156_invocation_migration"] = {
        i: reported_migration(i, "cpp-canonical-156-v1") for i in range(157)}
    payload["legacy_182_invocation_migration"] = {
        i: reported_migration(i, "cpp-canonical-182-v1") for i in range(183)}
    payload["legacy_225_invocation_migration"] = {
        i: reported_migration(i, "legacy-225-v1") for i in range(226)}
    payload["benchmark_86_canonical_action_pipelines"] = benchmark_86_canonical_action_pipelines()
    payload["benchmark_86_io_variant_pipelines"] = benchmark_86_io_variant_pipelines()
    payload["benchmark_86_action_parameters"] = benchmark_86_action_parameters()
    payload["parameterized_actions"] = action_parameters.schemas()
    payload["parameter_catalog_fingerprint"] = action_parameters.parameter_fingerprint()
    payload["counts"]["legacy_callable_selector_records"] = payload["counts"].pop("callable_action_implementation_records")
    payload["counts"]["callable_action_implementation_records"] = len(POLICY_ACTIONS)
    payload["counts"]["input_output_variants"] = sum(map(len, POLICY_IO_VARIANTS.values()))
    payload["counts"]["parameterized_algorithms"] = len(action_parameters.BY_NAME)
    payload["counts"]["lz_learning_parameter_choices"] = len(lz_parameters.parameter_candidates())
    payload["counts"]["window_learning_parameter_choices"] = len(window_parameters.parameter_candidates())
    # These legacy category totals intentionally include replay selectors.
    for key in ("exact_byte_stage_actions", "reference_byte_stage_actions",
                "exact_native_lz_stage_actions", "exact_atomic_operator_actions"):
        payload["counts"]["legacy_" + key] = payload["counts"].pop(key)
    payload["registry_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def self_check() -> dict[str, Any]:
    samples = (
        np.asarray([], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([-7, -7, -3, 0, 1, 1, 1, 18, 18, 99], dtype=np.int64),
        np.arange(-35, 37, dtype=np.int64),
    )
    checks = 0
    for sample in samples:
        for action in PRIMITIVE_ACTIONS:
            encoded = encode_operator_pipeline(sample, (action.name,))
            restored, actions = decode_operator_pipeline(encoded)
            if actions != (action.name,) or not np.array_equal(restored, sample):
                raise AssertionError(f"primitive action round trip failed: {action.name}")
            checks += 1

    expected = {
        "fastpfor": ("scaling", "bitpack_int", "integer_code"),
        "ALP": ("scaling", "frame_of_reference", "bitpack_int"),
        "LECO": ("regression_predictor", "zigzag", "bitpack_int"),
        "REGER": ("value_reorder", "delta", "zigzag", "frame_of_reference",
                  "bitpack_int", "block_state_control"),
        "BOS": ("scaling", "delta", "zigzag", "bitpack_int"),
        "CHIMP": ("xor", "trailing_zero_pack"),
        "Chimp128": ("xor", "trailing_zero_pack"),
        "GORILLA": ("xor", "trailing_zero_pack"),
    }
    for method, pipeline in expected.items():
        actual = concrete_operator_pipeline(method)
        if actual != pipeline:
            raise AssertionError(f"decomposition mismatch for {method}: {actual} != {pipeline}")
        expected_names = concrete_action_pipeline(method)
        if all(name in ACTION_BY_NAME for name in expected_names):
            for sample in samples:
                encoded = encode_catalog_method(sample, method)
                restored, action_names = decode_operator_pipeline(encoded)
                if action_names != expected_names or not np.array_equal(restored, sample):
                    raise AssertionError(f"catalog pipeline round trip failed: {method}")
                checks += 1

    payload = registry()
    if payload["algorithm_action_pipelines"]["CHIMP"][0] != CHIMP_XOR_ACTION:
        raise AssertionError("CHIMP must use the history-128 XOR variant")
    if payload["algorithm_action_pipelines"]["Chimp128"][0] != CHIMP_XOR_ACTION:
        raise AssertionError("Chimp128 must use the history-128 XOR variant")
    if payload["algorithm_action_pipelines"]["GORILLA"][0] == CHIMP_XOR_ACTION:
        raise AssertionError("GORILLA must use previous-value XOR, not Chimp history selection")
    if any(
        isinstance(action_name, str) and action_name.startswith(("catalog.", "tsfile."))
        for pipeline in payload["legacy_macro_replacements"].values()
        for action_name in pipeline
    ):
        raise AssertionError("legacy macro replacement still contains a whole-codec action")

    history_probe = np.asarray(
        [0x10001, 0x20002, 0x30003, 0x40001, 0x50002, 0x60003, 0x70001, 0x80002],
        dtype=np.int64,
    )
    previous_xor = encode_operator_pipeline(history_probe, (DEFAULT_ACTION_BY_OPERATOR_ID["xor"].name,))
    chimp_xor = encode_operator_pipeline(history_probe, (CHIMP_XOR_ACTION,))
    if previous_xor == chimp_xor:
        raise AssertionError("previous-value and Chimp-history XOR variants unexpectedly match")

    catalog_probe = np.asarray([-9, -9, -3, 0, 1, 2, 2, 2, 17, 65], dtype=np.int64)
    unresolved_catalog_variants: list[str] = []
    non_reference_catalog_variants: list[str] = []
    catalog_variant_checks = 0
    for method, logical_type, _pipeline in _iter_catalog_variants():
        selected_type = None if logical_type == "default" else logical_type
        try:
            action_names = concrete_action_pipeline(method, logical_type=selected_type)
        except ValueError:
            unresolved_catalog_variants.append(f"{method}:{logical_type}")
            continue
        if not all(name in ACTION_BY_NAME for name in action_names):
            non_reference_catalog_variants.append(f"{method}:{logical_type}")
            continue
        encoded = encode_operator_pipeline(catalog_probe, action_names)
        restored, decoded_names = decode_operator_pipeline(encoded)
        if decoded_names != action_names or not np.array_equal(restored, catalog_probe):
            raise AssertionError(f"catalog variant round trip failed: {method}:{logical_type}")
        catalog_variant_checks += 1
        checks += 1
    if unresolved_catalog_variants:
        raise AssertionError(f"unexpected unresolved catalog variants: {unresolved_catalog_variants}")

    benchmark_86_variant_checks = 0
    for display_name, logical_type, _pipeline in _iter_benchmark_86_variants():
        selected_type = None if logical_type == "default" else logical_type
        action_names = concrete_action_pipeline(display_name, logical_type=selected_type)
        if not all(name in ACTION_BY_NAME for name in action_names):
            continue
        encoded = encode_operator_pipeline(catalog_probe, action_names)
        restored, decoded_names = decode_operator_pipeline(encoded)
        if decoded_names != action_names or not np.array_equal(restored, catalog_probe):
            raise AssertionError(f"benchmark-86 round trip failed: {display_name}:{logical_type}")
        benchmark_86_variant_checks += 1
        checks += 1
    if len(benchmark_86_action_pipelines()) != 86:
        raise AssertionError("benchmark-86 action mapping is incomplete")

    exact_byte_probe = bytes(range(64)) * 3 + b"native-lz-stage" * 9
    exact_native_checks = 0
    for algorithm_key in sorted(set(EXACT_NATIVE_METHOD_KEYS.values())):
        preset = native_lz_preset(algorithm_key)
        historical = _compress_native_lz_family_fast(exact_byte_probe, preset)
        staged = encode_exact_native_lz_pipeline(exact_byte_probe, algorithm_key)
        if staged != historical:
            raise AssertionError(f"native-LZ split changed bytes: {algorithm_key}")
        if decompress_native_lz_family(staged, algorithm_key, len(exact_byte_probe)) != exact_byte_probe:
            raise AssertionError(f"native-LZ split round trip failed: {algorithm_key}")
        action_selected = encode_registered_exact_action_pipeline(
            exact_byte_probe,
            tuple(
                action.name for action in exact_native_lz_action_pipeline(algorithm_key)
            ),
        )
        if action_selected != staged:
            raise AssertionError(f"native-LZ action-name execution failed: {algorithm_key}")
        exact_native_checks += 1
        checks += 1

    atomic_bytes = bytes(range(64)) * 4
    for method, decoder in (
        ("Huffman", huffman_decode),
        ("Arithmetic", arithmetic_decode),
        ("ANS", ans_decode),
    ):
        encoded = encode_exact_atomic_method(atomic_bytes, method, "bytes")
        action_encoded = encode_registered_exact_action_pipeline(
            atomic_bytes, exact_atomic_action_pipeline(method, "bytes")
        )
        if action_encoded != encoded:
            raise AssertionError(f"exact atomic action-name execution failed: {method}")
        if decoder(encoded) != atomic_bytes:
            raise AssertionError(f"exact atomic byte action failed: {method}")
        checks += 1
    atomic_ints = [-9, -9, -3, 0, 1, 1, 17, 17, 17]
    atomic_int_raw = np.asarray(atomic_ints, dtype="<i8").tobytes()
    if decode_dictionary_int(encode_exact_atomic_method(atomic_int_raw, "Dict.", "int")) != atomic_ints:
        raise AssertionError("exact int dictionary action failed")
    if decode_rle_int64(encode_exact_atomic_method(atomic_int_raw, "RLE", "int")) != atomic_ints:
        raise AssertionError("exact int RLE action failed")
    checks += 2
    atomic_floats = [-3.5, -3.5, 0.0, 1.25, 1.25, 9.0]
    atomic_float_raw = np.asarray(atomic_floats, dtype="<f8").tobytes()
    dict_float = decode_dictionary_float(
        encode_exact_atomic_method(atomic_float_raw, "Dict.", "float")
    )
    rle_float = decode_rle_double(encode_registered_exact_action_pipeline(
        atomic_float_raw,
        exact_atomic_action_pipeline("RLE", "float", max_point=2),
        logical_type="float",
        max_point=2,
    ))
    expected_float_bits = np.asarray(atomic_floats, dtype="<f8").view("<u8")
    if not np.array_equal(
        np.asarray(dict_float, dtype="<f8").view("<u8"), expected_float_bits
    ):
        raise AssertionError("exact float dictionary action failed")
    if not np.array_equal(
        np.asarray(rle_float, dtype="<f8").view("<u8"), expected_float_bits
    ):
        raise AssertionError("exact float RLE action failed")
    checks += 2

    if len(benchmark_86_exact_action_pipelines()) != 86:
        raise AssertionError("benchmark-86 exact/reference action mapping is incomplete")
    def iter_action_names(node: Any) -> Iterable[str]:
        if isinstance(node, str):
            yield node
        elif isinstance(node, list):
            for child in node:
                yield from iter_action_names(child)
        elif isinstance(node, dict):
            for child in node.values():
                yield from iter_action_names(child)

    unknown_exact_names = sorted(
        {
            action_name
            for pipeline in benchmark_86_exact_action_pipelines().values()
            for action_name in iter_action_names(pipeline)
            if action_name not in ALL_CONCRETE_ACTION_BY_NAME
        }
    )
    if unknown_exact_names:
        raise AssertionError(f"unregistered exact/reference actions: {unknown_exact_names}")
    if len(ALL_CONCRETE_ACTION_BY_NAME) != len(ALL_CONCRETE_ACTION_IMPLEMENTATIONS):
        raise AssertionError("concrete action registry contains duplicate names")
    return {
        "status": "PASS",
        "round_trips": checks,
        "compression_operator_actions": payload["counts"]["compression_operator_actions"],
        "callable_action_implementation_records": payload["counts"]["callable_action_implementation_records"],
        "policy_control_actions_including_stop": payload["counts"]["policy_control_actions_including_stop"],
        "catalog_methods": payload["counts"]["catalog_methods"],
        "typed_pipeline_variants": payload["counts"]["typed_pipeline_variants"],
        "deterministic_catalog_variant_round_trips": catalog_variant_checks,
        "unresolved_catalog_variants": unresolved_catalog_variants,
        "current_non_reference_catalog_variants": non_reference_catalog_variants,
        "benchmark_86_taxonomy_pipelines_resolved": len(benchmark_86_action_pipelines()),
        "benchmark_86_variant_round_trips": benchmark_86_variant_checks,
        "exact_native_lz_stage_round_trips": exact_native_checks,
        "legacy_exact_byte_stage_selectors": payload["counts"]["legacy_exact_byte_stage_actions"],
        "non_atomic_experimental_helpers": payload["counts"]["non_atomic_experimental_helpers"],
        "reference_or_proxy_actions": payload["counts"]["reference_or_proxy_actions"],
        "registry_sha256": payload["registry_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="store_true", help="print the full JSON action registry")
    parser.add_argument("--self-check", action="store_true", help="run primitive and named-pipeline round trips")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.registry:
        print(json.dumps(registry(), indent=2, ensure_ascii=False))
    if args.self_check or not args.registry:
        print(json.dumps(self_check(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
