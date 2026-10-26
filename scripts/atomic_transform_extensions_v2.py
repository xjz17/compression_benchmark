#!/usr/bin/env python3
"""Versioned lossless primitive extensions; the frozen v1 module is unmodified.

All side streams are explicit inverse metadata and MUST be included in any size
measurement. Float exceptions preserve raw bits only at <=5% of positions; this
is a sparse exception representation, never an unrestricted original-data copy.
Block FOR and block bit-packing are independent operations; neither implicitly
performs the other, entropy coding, a whole codec, or a container operation.
"""
from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np

import atomic_transform_runtime_v1 as v1

VERSION = "atomic-transform-extensions-v2"
State = v1.State
U64 = v1.U64
MAX_BYTES = v1.MAX_BYTES
EXTENSION_OPERATORS = ("float_scale_exceptions", "block_for", "block_bitpack")
SUPPORTED_OPERATORS = v1.SUPPORTED_OPERATORS + EXTENSION_OPERATORS
INVERSE_FIELDS = dict(v1.INVERSE_FIELDS, **{
    # max_fraction is an admission-integrity field rather than a mathematical
    # reconstruction parameter. Exception count is derived from its side stream.
    "float_scale_exceptions": ("factor", "max_fraction", "exceptions_positions_b64", "exceptions_values_b64"),
    "block_for": ("block_size", "minima_b64"),
    "block_bitpack": ("block_size", "widths_b64", "in_count"),
})
BLOCK_SIZES = (16, 32, 64, 128, 256, 512, 1024)
SCALE_FACTORS = (1, 10, 100, 1000, 10000, 100000, 1000000)


def _factor(value: Any) -> int:
    value = v1._integer(value, "factor", 1, 1000000)
    if value not in SCALE_FACTORS:
        raise ValueError("factor must be a power of ten <= 1e6")
    return value


def _block_size(value: Any) -> int:
    value = v1._integer(value, "block_size", 16, 1024)
    if value not in BLOCK_SIZES:
        raise ValueError("block_size must be a power of two in [16, 1024]")
    return value


def _fraction(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("max_fraction must be a finite number in [0, 0.05]")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 0.05:
        raise ValueError("max_fraction must be a finite number in [0, 0.05]")
    return value


def _encode(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype=U64).tobytes()).decode("ascii")


def _decode(encoded: Any, name: str, *, element_width: int) -> bytes:
    if not isinstance(encoded, str) or len(encoded) > (MAX_BYTES + 2) // 3 * 4:
        raise ValueError(f"invalid {name} side stream")
    try:
        result = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} base64") from exc
    if len(result) > MAX_BYTES or len(result) % element_width:
        raise ValueError(f"invalid {name} side stream length")
    return result


def _validate_extension(spec: dict, kind: str) -> str:
    op = spec["op"]
    if kind != "u64":
        raise ValueError(f"{op} requires u64 input, not byte state")
    allowed = {"op", "factor", "max_fraction"} if op == "float_scale_exceptions" else {"op", "block_size"}
    if set(spec) - allowed:
        raise ValueError(f"unexpected parameters for {op}")
    if op == "float_scale_exceptions":
        _factor(spec.get("factor", 1))
        _fraction(spec.get("max_fraction", 0.05))
    else:
        _block_size(spec.get("block_size", 128))
    return "bytes" if op == "block_bitpack" else "u64"


def validate_specs(specs: list[dict]) -> str:
    """Check the combined v1/v2 grammar without mutating either registry."""
    if not isinstance(specs, (list, tuple)):
        raise ValueError("specs must be a list/tuple")
    kind = "u64"
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("op") not in SUPPORTED_OPERATORS:
            raise ValueError(f"unknown operator spec {spec!r}")
        if spec["op"] in EXTENSION_OPERATORS:
            kind = _validate_extension(spec, kind)
        else:
            prefix = [{"op": "compact", "width": 8}] if kind == "bytes" else []
            kind = v1.validate_specs(prefix + [spec])
    return kind


def _metadata(state: State, op: str) -> dict:
    return {"op": op, "in_kind": state.kind, "in_count": state.count, "in_nbytes": len(state.payload)}


def apply_one(state: State, spec: dict) -> tuple[State, dict]:
    """Apply a v1 operator unchanged, or one explicit v2 primitive."""
    if not isinstance(spec, dict) or spec.get("op") not in SUPPORTED_OPERATORS:
        raise ValueError("unknown operator spec")
    op = spec["op"]
    if op not in EXTENSION_OPERATORS:
        return v1.apply_one(state, spec)
    _validate_extension(spec, state.kind)
    meta = _metadata(state, op)
    values = state.values()
    count = len(values)
    if op == "float_scale_exceptions":
        factor = _factor(spec.get("factor", 1))
        fraction = _fraction(spec.get("max_fraction", 0.05))
        floats = values.view("<f8")
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = np.rint(floats * factor)
        representable = np.isfinite(scaled) & (scaled >= -(2.0 ** 63)) & (scaled < 2.0 ** 63)
        integers = np.zeros(count, dtype="<i8")
        integers[representable] = scaled[representable].astype("<i8")
        restored = integers.astype("<f8") / factor
        exact = representable & (restored.view(U64) == values)
        positions = np.flatnonzero(~exact).astype(U64)
        if count and len(positions) / count > fraction:
            raise ValueError(f"float_scale_exceptions exceeds max_fraction: {len(positions)}/{count}")
        # A neutral placeholder, not an approximation disguised as losslessness.
        integers[~exact] = 0
        output = v1._numeric(integers.view(U64))
        meta.update(factor=factor, max_fraction=fraction,
                    exceptions_positions_b64=_encode(positions),
                    exceptions_values_b64=_encode(values[~exact]))
    elif op == "block_for":
        block_size = _block_size(spec.get("block_size", 128))
        starts = np.arange(0, count, block_size, dtype=np.intp)
        minima = np.minimum.reduceat(values, starts) if count else np.empty(0, dtype=U64)
        output = v1._numeric(values - np.repeat(minima, block_size)[:count])
        meta.update(block_size=block_size, minima_b64=_encode(minima))
    else:
        block_size = _block_size(spec.get("block_size", 128))
        widths = bytearray()
        blocks = []
        for offset in range(0, count, block_size):
            block = values[offset:offset + block_size]
            bits = int(block.max()).bit_length()
            widths.append(bits)
            blocks.append(v1._pack_bits(block, bits))
        output = State(b"".join(blocks), "bytes")
        meta.update(block_size=block_size,
                    widths_b64=base64.b64encode(bytes(widths)).decode("ascii"))
    meta.update(out_kind=output.kind, out_count=output.count, out_nbytes=len(output.payload))
    return output, meta


def _extension_record(state: State, meta: dict) -> int:
    v1._check_record_state(state, meta, "out")
    if meta.get("in_kind") != "u64":
        raise ValueError("extension input must have been u64")
    op = meta["op"]
    if state.kind != ("bytes" if op == "block_bitpack" else "u64"):
        raise ValueError("extension output kind mismatch")
    allowed = {"op", "in_kind", "in_count", "in_nbytes", "out_kind", "out_count", "out_nbytes"} | set(INVERSE_FIELDS[op])
    if set(meta) != allowed:
        raise ValueError("missing or unexpected extension inverse fields")
    count = v1._integer(meta.get("in_count"), "in_count", 0, MAX_BYTES // 8)
    if meta.get("in_nbytes") != count * 8:
        raise ValueError("extension input typed length mismatch")
    return count


def reverse_one(state: State, meta: dict) -> State:
    """Invert a mixed v1/v2 record with only its payload and charged metadata."""
    if not isinstance(meta, dict) or meta.get("op") not in SUPPORTED_OPERATORS:
        raise ValueError("unknown inverse operator record")
    op = meta["op"]
    if op not in EXTENSION_OPERATORS:
        return v1.reverse_one(state, meta)
    count = _extension_record(state, meta)
    if op == "float_scale_exceptions":
        factor = _factor(meta["factor"])
        fraction = _fraction(meta["max_fraction"])
        positions = np.frombuffer(_decode(meta["exceptions_positions_b64"], "exception positions", element_width=8), dtype=U64)
        exceptions = np.frombuffer(_decode(meta["exceptions_values_b64"], "exception values", element_width=8), dtype=U64)
        if len(positions) != len(exceptions) or len(positions) > count:
            raise ValueError("exception position/value count mismatch")
        if count and len(positions) / count > fraction:
            raise ValueError("exception side stream exceeds max_fraction")
        if len(positions) and (int(positions[-1]) >= count or np.any(positions[1:] <= positions[:-1])):
            raise ValueError("exception positions must be increasing and in bounds")
        integers = state.values()
        if len(integers) != count:
            raise ValueError("float-scale payload value count mismatch")
        indices = positions.astype(np.intp)
        if np.any(integers[indices] != 0):
            raise ValueError("exception positions must contain zero placeholders")
        restored = (integers.view("<i8").astype("<f8") / factor).view(U64).copy()
        restored[indices] = exceptions
        output = v1._numeric(restored)
    elif op == "block_for":
        block_size = _block_size(meta["block_size"])
        minima = np.frombuffer(_decode(meta["minima_b64"], "block minima", element_width=8), dtype=U64)
        if len(minima) != (count + block_size - 1) // block_size:
            raise ValueError("block minima count mismatch")
        values = state.values()
        if len(values) != count:
            raise ValueError("block FOR value count mismatch")
        expanded = np.repeat(minima, block_size)[:count]
        if np.any(values > np.uint64((1 << 64) - 1) - expanded):
            raise ValueError("block FOR inverse overflow")
        output = v1._numeric(values + expanded)
    else:
        block_size = _block_size(meta["block_size"])
        widths = _decode(meta["widths_b64"], "block widths", element_width=1)
        if len(widths) != (count + block_size - 1) // block_size or any(bits > 64 for bits in widths):
            raise ValueError("invalid block widths/count")
        cursor = 0
        blocks = []
        for index, bits in enumerate(widths):
            block_count = min(block_size, count - index * block_size)
            length = (block_count * bits + 7) // 8
            blocks.append(v1._unpack_bits(state.payload[cursor:cursor + length], bits, block_count))
            cursor += length
        if cursor != len(state.payload):
            raise ValueError("block-packed payload has trailing/missing bytes")
        output = v1._numeric(np.concatenate(blocks) if blocks else np.empty(0, dtype=U64))
    v1._check_record_state(output, meta, "in")
    return output


def apply_ops(raw: bytes, specs: list[dict]) -> tuple[bytes, list[dict]]:
    validate_specs(specs)
    state = State(raw, "u64")
    records = []
    for spec in specs:
        state, record = apply_one(state, spec)
        records.append(record)
    return state.payload, records


def reverse_ops(payload: bytes, metadata: list[dict]) -> bytes:
    if not isinstance(metadata, (list, tuple)) or any(not isinstance(record, dict) for record in metadata):
        raise ValueError("metadata must be a sequence of record dicts")
    state = State(payload, metadata[-1].get("out_kind") if metadata else "u64")
    for record in reversed(metadata):
        state = reverse_one(state, record)
    if state.kind != "u64":
        raise ValueError("inverse did not restore initial u64 state")
    return state.payload
