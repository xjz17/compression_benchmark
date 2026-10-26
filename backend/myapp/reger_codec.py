"""Typed/benchmark bridge to the native legacy-REGER3 implementation.

The former Python regression/reorder/bit-packing kernels have been removed.
Frozen pre-migration golden files, not this active bridge, are the independent
compatibility oracle. Integer and float lane conversion here is transport only.
"""
from __future__ import annotations

import os
import struct
import time
from typing import Iterable, Sequence

from .benchmark_loss import accumulate_double_list_loss, accumulate_long_list_loss
from .cpp_action_bridge import run as _cpp_run

U64_MASK = (1 << 64) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
MULTI_MAGIC = b"REGER3"
LEGACY_MAGIC = b"REGER1"
ORDER_PERMUTED_FLAG = 1
TIME_STREAM_FLAG = 2
MULTI_HEADER_STRUCT = struct.Struct("<6sIHHHI")
BLOCK_HEADER_STRUCT = struct.Struct("<HBI")
SERIES_HEADER_STRUCT = struct.Struct("<BI")
_TRANSPORT = struct.Struct("<4sQIB")


def _validate_i64(value: int) -> int:
    value = int(value)
    if not I64_MIN <= value <= I64_MAX:
        raise OverflowError("REGER int64 value out of range")
    return value


def _double_to_i64(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _i64_to_double(value: int) -> float:
    return struct.unpack("<d", struct.pack("<q", value))[0]


def _common_row_count(columns: Sequence[Sequence[int]]) -> int:
    return min((len(column) for column in columns), default=0)


def _normalize_times(times: Sequence[int] | None, row_count: int) -> list[int]:
    if times is None:
        return list(range(row_count))
    if len(times) < row_count:
        raise ValueError("REGER time column shorter than value columns")
    return [_validate_i64(times[index]) for index in range(row_count)]


def _reger_profile() -> str:
    return "fast" if os.environ.get("WEB_COMPRESSION_REGER_PROFILE", "").strip().lower() == "fast" else "balanced"


def _default_reorder_iterations(column_count: int = 1) -> int:
    try:
        return max(0, min(int(os.environ.get("WEB_COMPRESSION_REGER_REORDER_ITERS", "").strip()), 20))
    except ValueError:
        return 0 if _reger_profile() == "fast" else (4 if column_count == 1 else 0)


def _transport_columns(columns: Sequence[Sequence[int]], times: Sequence[int] | None = None) -> bytes:
    prepared = [[_validate_i64(value) for value in column] for column in columns]
    rows = _common_row_count(prepared)
    if rows > 0xFFFFFFFF or len(prepared) > 0xFFFF:
        raise ValueError("REGER column dimensions exceed wire format")
    out = bytearray(_TRANSPORT.pack(b"RGMI", rows, len(prepared), times is not None))
    if times is not None:
        out.extend(struct.pack(f"<{rows}q", *_normalize_times(times, rows)))
    for column in prepared:
        out.extend(struct.pack(f"<{rows}q", *column[:rows]))
    return bytes(out)


def _row_order_candidates(block_columns, block_times) -> list[list[int]]:
    """Diagnostic compatibility API; order selection is entirely native."""
    raw = _cpp_run("reger_legacy_cpp_cli", "orders-forward", _transport_columns(block_columns, block_times))
    if len(raw) < 8:
        raise ValueError("truncated native REGER order response")
    count, rows = struct.unpack_from("<II", raw)
    if len(raw) != 8 + count * rows * 4:
        raise ValueError("native REGER order response length mismatch")
    return [list(struct.unpack_from(f"<{rows}I", raw, 8 + index * rows * 4))
            for index in range(count)]


def _encode_i64_columns(columns: Sequence[Sequence[int]], times: Sequence[int] | None = None) -> bytes:
    return _cpp_run("reger_legacy_cpp_cli", "columns-forward", _transport_columns(columns, times))


def _decode_i64_columns(blob: bytes) -> list[list[int]]:
    raw = _cpp_run("reger_legacy_cpp_cli", "columns-inverse", blob)
    if len(raw) < _TRANSPORT.size:
        raise ValueError("truncated native REGER column response")
    magic, rows, count, has_times = _TRANSPORT.unpack_from(raw)
    if magic != b"RGMI" or has_times or len(raw) != _TRANSPORT.size + rows * count * 8:
        raise ValueError("invalid native REGER column response")
    return [list(struct.unpack_from(f"<{rows}q", raw, _TRANSPORT.size + index * rows * 8))
            for index in range(count)]


def encode_int_column(values: Sequence[int]) -> bytes:
    return _encode_i64_columns([values])


def decode_int_column(blob: bytes) -> list[int]:
    columns = _decode_i64_columns(blob)
    return columns[0] if columns else []


def encode_float_column(values: Sequence[float]) -> bytes:
    return _encode_i64_columns([[_double_to_i64(value) for value in values]])


def decode_float_column(blob: bytes) -> list[float]:
    columns = _decode_i64_columns(blob)
    return [_i64_to_double(value) for value in columns[0]] if columns else []


def encode_int_columns(columns: Sequence[Sequence[int]], times: Sequence[int] | None = None) -> bytes:
    return _encode_i64_columns(columns, times)


def decode_int_columns(blob: bytes) -> list[list[int]]:
    return _decode_i64_columns(blob)


def encode_float_columns(columns: Sequence[Sequence[float]], times: Sequence[int] | None = None) -> bytes:
    return _encode_i64_columns([[_double_to_i64(value) for value in column] for column in columns], times)


def decode_float_columns(blob: bytes) -> list[list[float]]:
    return [[_i64_to_double(value) for value in column] for column in _decode_i64_columns(blob)]


def _elapsed_ns(start: int) -> int:
    return max(0, int(time.perf_counter_ns() - start))


def bench_reger_int_columns(
    columns: Iterable[Sequence[int]],
    times: Sequence[int] | None = None,
) -> tuple[int, int, int, int, float, float]:
    source_columns = [[_validate_i64(v) for v in column] for column in columns]
    row_count = _common_row_count(source_columns)
    if row_count <= 0 or not source_columns:
        return (0, 0, 0, 0, 0.0, 0.0)
    values = [col[:row_count] for col in source_columns]
    row_times = _normalize_times(times, row_count) if times is not None else None
    loss = [0.0, 0.0]
    total_original = row_count * len(values) * 8
    start = time.perf_counter_ns()
    encoded = encode_int_columns(values, times=row_times)
    total_comp_ns = _elapsed_ns(start)
    start = time.perf_counter_ns()
    decoded = decode_int_columns(encoded)
    total_decomp_ns = _elapsed_ns(start)
    for expected, actual in zip(values, decoded):
        accumulate_long_list_loss(expected, actual, loss)
    return (
        int(total_original),
        int(len(encoded)),
        int(total_comp_ns),
        int(total_decomp_ns),
        float(loss[0]),
        float(loss[1]),
    )


def bench_reger_double_columns(
    columns: Iterable[Sequence[float]],
    times: Sequence[int] | None = None,
) -> tuple[int, int, int, int, float, float]:
    source_columns = [[float(v) for v in column] for column in columns]
    row_count = _common_row_count(source_columns)
    if row_count <= 0 or not source_columns:
        return (0, 0, 0, 0, 0.0, 0.0)
    values = [col[:row_count] for col in source_columns]
    row_times = _normalize_times(times, row_count) if times is not None else None
    loss = [0.0, 0.0]
    total_original = row_count * len(values) * 8
    start = time.perf_counter_ns()
    encoded = encode_float_columns(values, times=row_times)
    total_comp_ns = _elapsed_ns(start)
    start = time.perf_counter_ns()
    decoded = decode_float_columns(encoded)
    total_decomp_ns = _elapsed_ns(start)
    for expected, actual in zip(values, decoded):
        accumulate_double_list_loss(expected, actual, loss)
    return (
        int(total_original),
        int(len(encoded)),
        int(total_comp_ns),
        int(total_decomp_ns),
        float(loss[0]),
        float(loss[1]),
    )
