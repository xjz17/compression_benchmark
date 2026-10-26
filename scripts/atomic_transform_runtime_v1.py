#!/usr/bin/env python3
"""Lossless, composable numeric/byte primitives, with no compressor or frame.

Every input starts as little-endian uint64 bit patterns (including float64 bits).
Numeric operations are modulo 2**64 unless explicitly documented otherwise.
The caller must persist and charge *all* inverse metadata, including palettes.
No operation calls a codec, silently packs values, or saves its whole input as
an escape hatch. RLE and sparsezero expose both streams as concatenated uint64
values; packing those streams is a separately requested operation.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np

VERSION = "atomic-transform-v1"
U64 = np.dtype("<u8")
MAX_BYTES = 512 * 1024 * 1024
SUPPORTED_OPERATORS = (
    "zigzag", "delta", "xor", "for", "gcd", "dictionary", "compact",
    "bitpack", "shuffle", "bittranspose", "rle", "sparsezero", "float_scale",
)
# The generic typed lengths are audit fields, not all irreducible wire fields.
# A frame can reconstruct these from the original count and adjacent stages.
INVERSE_FIELDS = {
    "zigzag": (), "delta": ("lag", "order"), "xor": ("lag",),
    "for": ("minimum",), "gcd": ("factor",),
    "dictionary": ("palette_b64",), "compact": ("width", "in_count"),
    "bitpack": ("bits", "in_count"), "shuffle": ("width", "in_nbytes"),
    "bittranspose": ("width", "in_nbytes"), "rle": ("split", "in_count"),
    "sparsezero": ("split", "in_count"), "float_scale": ("factor",),
}
_SPEC_KEYS = {
    "zigzag": set(), "delta": {"lag", "order"}, "xor": {"lag"},
    "for": set(), "gcd": set(), "dictionary": {"max_entries"},
    "compact": {"width"}, "bitpack": {"bits"}, "shuffle": {"width"},
    "bittranspose": {"width"}, "rle": set(), "sparsezero": set(),
    "float_scale": {"factor"},
}


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside [{low}, {high}]")
    return value


@dataclass(frozen=True)
class State:
    payload: bytes
    kind: str = "u64"

    def __post_init__(self):
        if self.kind not in ("u64", "bytes"):
            raise ValueError("state kind must be u64 or bytes")
        if not isinstance(self.payload, bytes):
            raise ValueError("state payload must be bytes")
        if len(self.payload) > MAX_BYTES:
            raise ValueError("state exceeds safety size limit")
        if self.kind == "u64" and len(self.payload) % 8:
            raise ValueError("u64 payload must contain complete 8-byte lanes")

    @property
    def count(self) -> int:
        return len(self.payload) // (8 if self.kind == "u64" else 1)

    def values(self) -> np.ndarray:
        if self.kind != "u64":
            raise ValueError("this operator requires u64, not byte state")
        return np.frombuffer(self.payload, dtype=U64)


def _numeric(values: np.ndarray) -> State:
    return State(np.asarray(values, dtype=U64).tobytes(), "u64")


def validate_specs(specs: list[dict]) -> str:
    """Validate grammar/parameters without data; return the final state kind.

    Data-dependent checks (width overflow, exact float recovery) happen at apply.
    """
    if not isinstance(specs, (list, tuple)):
        raise ValueError("specs must be a list/tuple of explicit operator dicts")
    kind = "u64"
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("op") not in SUPPORTED_OPERATORS:
            raise ValueError(f"unknown operator spec {spec!r}")
        op = spec["op"]
        if set(spec) - (_SPEC_KEYS[op] | {"op"}):
            raise ValueError(f"unexpected parameters for {op}")
        if op not in ("shuffle", "bittranspose") and kind != "u64":
            raise ValueError(f"{op} requires u64 after a byte-producing operator")
        if op in ("delta", "xor"):
            _integer(spec.get("lag", 1), "lag", 1, 1024)
        if op == "delta":
            _integer(spec.get("order", 1), "order", 1, 2)
        if op == "dictionary":
            _integer(spec.get("max_entries", 4096), "max_entries", 1, MAX_BYTES // 8)
        if op == "compact":
            width = spec.get("width", "auto")
            if width != "auto" and _integer(width, "width", 1, 8) not in (1, 2, 4, 8):
                raise ValueError("compact width must be auto, 1, 2, 4, or 8")
        if op == "bitpack" and spec.get("bits", "auto") != "auto":
            _integer(spec["bits"], "bits", 0, 64)
        if op in ("shuffle", "bittranspose"):
            if _integer(spec.get("width", 8), "width", 1, 16) not in (1, 2, 4, 8, 16):
                raise ValueError("transpose width must be 1, 2, 4, 8, or 16")
        if op == "float_scale":
            factor = _integer(spec.get("factor", 1), "factor", 1, 1000000)
            if factor not in (1, 10, 100, 1000, 10000, 100000, 1000000):
                raise ValueError("float_scale factor must be a power of ten <= 1e6")
        if op in ("compact", "bitpack", "shuffle", "bittranspose"):
            kind = "bytes"
    return kind


def _difference(values: np.ndarray, lag: int, xor: bool = False) -> np.ndarray:
    result = values.copy()
    if len(values) > lag:
        if xor:
            result[lag:] = values[lag:] ^ values[:-lag]
        else:
            np.subtract(values[lag:], values[:-lag], out=result[lag:])
    return result


def _undifference(values: np.ndarray, lag: int, xor: bool = False) -> np.ndarray:
    result = values.copy()
    for offset in range(min(lag, len(result))):
        view = result[offset::lag]
        if xor:
            np.bitwise_xor.accumulate(view, out=view)
        else:
            np.cumsum(view, dtype=np.uint64, out=view)
    return result


def _transpose(payload: bytes, width: int, *, bits: bool, inverse: bool) -> bytes:
    """Transpose complete width-byte lanes; any tail bytes are copied verbatim."""
    complete = len(payload) // width * width
    if not complete:
        return payload
    lanes = complete // width
    data = np.frombuffer(payload[:complete], dtype=np.uint8)
    if bits:
        expanded = np.unpackbits(data, bitorder="little")
        if inverse:
            expanded = expanded.reshape(width * 8, lanes).T.reshape(-1)
        else:
            expanded = expanded.reshape(lanes, width * 8).T.reshape(-1)
        result = np.packbits(expanded, bitorder="little").tobytes()
    elif inverse:
        result = data.reshape(width, lanes).T.tobytes()
    else:
        result = data.reshape(lanes, width).T.tobytes()
    return result + payload[complete:]


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    if not len(values) or not bits:
        return b""
    if bits == 64:
        return values.astype(U64, copy=False).tobytes()
    # Lane-major, low-bit-first; final byte padding is always zero.
    bit_matrix = ((values[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def _unpack_bits(payload: bytes, bits: int, count: int) -> np.ndarray:
    if len(payload) != (count * bits + 7) // 8:
        raise ValueError("packed bit length mismatch")
    if not bits:
        return np.zeros(count, dtype=U64)
    if bits == 64:
        return np.frombuffer(payload, dtype=U64).copy()
    expanded = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if np.any(expanded[count * bits:]):
        raise ValueError("nonzero bitpack padding")
    bit_matrix = expanded[:count * bits].reshape(count, bits).astype(np.uint64)
    return np.bitwise_or.reduce(bit_matrix << np.arange(bits, dtype=np.uint64), axis=1)


def apply_one(state: State, spec: dict) -> tuple[State, dict]:
    """Apply one explicitly parameterized primitive and return its inverse record."""
    # Validate from the operator's actual incoming state, not a permissive guess.
    validate_specs(([{"op": "compact", "width": 8}] if state.kind == "bytes" else []) + [spec])
    op = spec["op"]
    meta: dict[str, Any] = {"op": op, "in_kind": state.kind, "in_count": state.count,
                           "in_nbytes": len(state.payload)}
    if op in ("shuffle", "bittranspose"):
        width = int(spec.get("width", 8))
        output = State(_transpose(state.payload, width, bits=op == "bittranspose", inverse=False), "bytes")
        meta["width"] = width
    else:
        values = state.values()
        count = len(values)
        maximum = int(values.max()) if count else 0
        if op == "zigzag":
            output = _numeric((values << np.uint64(1)) ^ (np.uint64(0) - (values >> np.uint64(63))))
        elif op == "delta":
            lag, order = int(spec.get("lag", 1)), int(spec.get("order", 1))
            result = values
            for _ in range(order):
                result = _difference(result, lag)
            output = _numeric(result)
            meta.update(lag=lag, order=order)
        elif op == "xor":
            lag = int(spec.get("lag", 1))
            output = _numeric(_difference(values, lag, xor=True))
            meta["lag"] = lag
        elif op == "for":
            minimum = int(values.min()) if count else 0
            output = _numeric(values - np.uint64(minimum))
            meta["minimum"] = minimum
        elif op == "gcd":
            factor = max(1, int(np.gcd.reduce(values)) if count else 1)
            output = _numeric(values // np.uint64(factor))
            meta["factor"] = factor
        elif op == "dictionary":
            palette, indices = np.unique(values, return_inverse=True)
            if len(palette) > int(spec.get("max_entries", 4096)):
                raise ValueError("dictionary cardinality exceeds explicit max_entries")
            output = _numeric(indices)
            meta["palette_b64"] = base64.b64encode(palette.astype(U64, copy=False).tobytes()).decode("ascii")
        elif op == "compact":
            needed = next(w for w in (1, 2, 4, 8) if maximum < (1 << (w * 8)))
            requested = spec.get("width", "auto")
            width = needed if requested == "auto" else int(requested)
            if width < needed:
                raise ValueError("compact width would truncate values")
            output = State(values.astype(f"<u{width}").tobytes(), "bytes")
            meta["width"] = width
        elif op == "bitpack":
            needed = maximum.bit_length()
            requested = spec.get("bits", "auto")
            bits = needed if requested == "auto" else int(requested)
            if bits < needed:
                raise ValueError("bitpack width would truncate values")
            output = State(_pack_bits(values, bits), "bytes")
            meta["bits"] = bits
        elif op == "rle":
            starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]]) if count else np.empty(0, dtype=np.int64)
            runs = np.diff(np.r_[starts, count]).astype(U64)
            output = _numeric(np.concatenate((values[starts], runs)))
            meta["split"] = len(starts)
        elif op == "sparsezero":
            positions = np.flatnonzero(values)
            output = _numeric(np.concatenate((values[positions], positions.astype(U64))))
            meta["split"] = len(positions)
        elif op == "float_scale":
            factor = int(spec.get("factor", 1))
            floats = values.view("<f8")
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = np.rint(floats * factor)
            if not np.all(np.isfinite(scaled)) or np.any(scaled < -(2.0 ** 63)) or np.any(scaled >= 2.0 ** 63):
                raise ValueError("float_scale cannot encode non-finite/out-of-int64 values")
            integers = scaled.astype("<i8")
            reconstructed = integers.astype("<f8") / factor
            if not np.array_equal(reconstructed.view(U64), values):
                raise ValueError("float_scale rejected: inverse is not bit-exact")
            output = _numeric(integers.view(U64))
            meta["factor"] = factor
        else:
            raise ValueError(f"unsupported operator {op}")
    meta.update(out_kind=output.kind, out_count=output.count, out_nbytes=len(output.payload))
    return output, meta


def _check_record_state(state: State, meta: dict, side: str) -> None:
    if state.kind != meta.get(f"{side}_kind"):
        raise ValueError("metadata state kind mismatch")
    if state.count != _integer(meta.get(f"{side}_count"), f"{side}_count", 0, MAX_BYTES):
        raise ValueError("metadata state count mismatch")
    if len(state.payload) != _integer(meta.get(f"{side}_nbytes"), f"{side}_nbytes", 0, MAX_BYTES):
        raise ValueError("metadata state byte length mismatch")


def reverse_one(state: State, meta: dict) -> State:
    """Invert a primitive from its recorded state, rejecting malformed metadata."""
    if not isinstance(meta, dict) or meta.get("op") not in SUPPORTED_OPERATORS:
        raise ValueError("invalid inverse operator record")
    _check_record_state(state, meta, "out")
    op = meta["op"]
    count = _integer(meta.get("in_count"), "in_count", 0, MAX_BYTES)
    in_kind = meta.get("in_kind")
    if in_kind not in ("u64", "bytes") or (op not in ("shuffle", "bittranspose") and in_kind != "u64"):
        raise ValueError("invalid inverse input kind")
    if count * (8 if in_kind == "u64" else 1) > MAX_BYTES:
        raise ValueError("inverse exceeds safety size limit")
    if op in ("shuffle", "bittranspose"):
        width = _integer(meta.get("width"), "width", 1, 16)
        if width not in (1, 2, 4, 8, 16):
            raise ValueError("invalid transpose width")
        result = State(_transpose(state.payload, width, bits=op == "bittranspose", inverse=True), in_kind)
    elif op == "compact":
        width = _integer(meta.get("width"), "width", 1, 8)
        if state.kind != "bytes" or width not in (1, 2, 4, 8) or len(state.payload) != count * width:
            raise ValueError("invalid compact record")
        result = _numeric(np.frombuffer(state.payload, dtype=f"<u{width}"))
    elif op == "bitpack":
        if state.kind != "bytes":
            raise ValueError("bitpack inverse requires byte state")
        bits = _integer(meta.get("bits"), "bits", 0, 64)
        result = _numeric(_unpack_bits(state.payload, bits, count))
    else:
        values = state.values()
        if op == "zigzag":
            result = _numeric((values >> np.uint64(1)) ^ (np.uint64(0) - (values & np.uint64(1))))
        elif op == "delta":
            lag = _integer(meta.get("lag"), "lag", 1, 1024)
            order = _integer(meta.get("order"), "order", 1, 2)
            restored = values
            for _ in range(order):
                restored = _undifference(restored, lag)
            result = _numeric(restored)
        elif op == "xor":
            result = _numeric(_undifference(values, _integer(meta.get("lag"), "lag", 1, 1024), xor=True))
        elif op == "for":
            result = _numeric(values + np.uint64(_integer(meta.get("minimum"), "minimum", 0, (1 << 64) - 1)))
        elif op == "gcd":
            factor = _integer(meta.get("factor"), "factor", 1, (1 << 64) - 1)
            if len(values) and int(values.max()) > ((1 << 64) - 1) // factor:
                raise ValueError("gcd inverse multiplication overflow")
            result = _numeric(values * np.uint64(factor))
        elif op == "dictionary":
            try:
                palette_raw = base64.b64decode(meta["palette_b64"], validate=True)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid dictionary palette") from exc
            if len(palette_raw) % 8 or len(palette_raw) > MAX_BYTES:
                raise ValueError("invalid dictionary palette length")
            palette = np.frombuffer(palette_raw, dtype=U64)
            if len(values) and int(values.max()) >= len(palette):
                raise ValueError("dictionary index outside palette")
            result = _numeric(palette[values.astype(np.intp)])
        elif op == "rle":
            split = _integer(meta.get("split"), "split", 0, MAX_BYTES // 16)
            if len(values) != split * 2:
                raise ValueError("RLE stream split mismatch")
            run_values, runs = values[:split], values[split:]
            if np.any(runs == 0) or np.any(runs > count) or int(runs.sum(dtype=np.uint64)) != count:
                raise ValueError("invalid RLE run lengths")
            result = _numeric(np.repeat(run_values, runs.astype(np.intp)))
        elif op == "sparsezero":
            split = _integer(meta.get("split"), "split", 0, MAX_BYTES // 16)
            if len(values) != split * 2:
                raise ValueError("sparsezero stream split mismatch")
            nonzero, positions = values[:split], values[split:]
            if np.any(nonzero == 0) or (len(positions) and int(positions[-1]) >= count) or np.any(positions[1:] <= positions[:-1]):
                raise ValueError("invalid sparsezero positions/values")
            restored = np.zeros(count, dtype=U64)
            restored[positions.astype(np.intp)] = nonzero
            result = _numeric(restored)
        elif op == "float_scale":
            factor = _integer(meta.get("factor"), "factor", 1, 1000000)
            if factor not in (1, 10, 100, 1000, 10000, 100000, 1000000):
                raise ValueError("invalid float_scale factor")
            result = _numeric((values.view("<i8").astype("<f8") / factor).view(U64))
        else:
            raise ValueError("unsupported inverse operator")
    _check_record_state(result, meta, "in")
    return result


def apply_ops(raw: bytes, specs: list[dict]) -> tuple[bytes, list[dict]]:
    """Apply only the listed primitives. The empty pipeline is exactly identity."""
    validate_specs(specs)
    state = State(raw, "u64")
    metadata = []
    for spec in specs:
        state, record = apply_one(state, spec)
        metadata.append(record)
    return state.payload, metadata


def reverse_ops(payload: bytes, metadata: list[dict]) -> bytes:
    """Reverse every recorded primitive. No external/original column is needed."""
    if not isinstance(metadata, (list, tuple)):
        raise ValueError("metadata must be a list/tuple")
    if any(not isinstance(record, dict) for record in metadata):
        raise ValueError("every inverse record must be a dict")
    kind = metadata[-1].get("out_kind") if metadata else "u64"
    state = State(payload, kind)
    for record in reversed(metadata):
        state = reverse_one(state, record)
    if state.kind != "u64":
        raise ValueError("inverse chain did not restore initial u64 state")
    return state.payload
