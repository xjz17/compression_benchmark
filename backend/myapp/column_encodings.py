"""
Column encodings inspired by Apache Parquet / Arrow and DuckDB-style analytics codecs.
Not wire-compatible with those engines; same *ideas* for benchmark comparison.

Apache IoTDB / TsFile time encodings (GORILLA, TS_2DIFF, etc.) are already benchmarked
via TsFile in the Java pipeline only; this module adds Parquet-like column transforms
for Python (and mirrors in Java/C++).

- DELTA_BINARY_PACKED: Parquet-style delta + zigzag + LEB128 (int64 columns)
- col_dictionary: dictionary + int64 indices (int or float64)
- col_rle_int: run-length on int64 values
- col_bss_lz4: BYTE_STREAM_SPLIT (float64) then LZ4 frame
"""

from __future__ import annotations

import struct
import time
from typing import List

from .benchmark_loss import accumulate_byte_loss, accumulate_double_list_loss, accumulate_long_list_loss
from .benchmark_wire import BENCHMARK_FLOAT_BYTES, BENCHMARK_INT_BYTES
from .cpp_action_bridge import run as _run_cpp_action

DELTA_BINARY_PACKED_ALGORITHM = 'DELTA_BINARY_PACKED'
# Legacy columnar RLE keys in old CSV reports; TsFile ``RLE`` is the canonical benchmark name.
LEGACY_COLUMN_RLE_ALGORITHMS = frozenset({'col_rle', 'col_rle_int'})


def is_delta_binary_packed_algorithm(name: str) -> bool:
    key = str(name or '').strip().removeprefix('\ufeff')
    return key == DELTA_BINARY_PACKED_ALGORITHM


def is_legacy_column_rle_algorithm(name: str) -> bool:
    key = str(name or '').strip().removeprefix('\ufeff')
    return key in LEGACY_COLUMN_RLE_ALGORITHMS

def _write_varint_u32(out: bytearray, v: int) -> None:
    v &= 0xFFFFFFFF
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)


def _read_varint_u32(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result & 0xFFFFFFFF, pos
        shift += 7
        if shift > 35:
            raise ValueError('bad varint')
    raise ValueError('truncated varint')


def _write_varint_u64(out: bytearray, v: int) -> None:
    x = v & ((1 << 64) - 1)
    while x >= 0x80:
        out.append((x & 0x7F) | 0x80)
        x >>= 7
    out.append(x)


def _read_varint_u64(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            if shift >= 63 and result >> 63:
                result -= 1 << 64
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError('bad varint')
    raise ValueError('truncated varint')


def _zigzag64(d: int) -> int:
    return ((d << 1) ^ (d >> 63)) & ((1 << 64) - 1)


def _unzigzag64(u: int) -> int:
    u &= (1 << 64) - 1
    return (u >> 1) ^ (-(u & 1))


def encode_delta_zigzag_varint_int(values: List[int]) -> bytes:
    if not values:
        return b''
    out = bytearray()
    out.extend(int(values[0]).to_bytes(8, 'little', signed=True))
    prev = int(values[0])
    for i in range(1, len(values)):
        d = int(values[i]) - prev
        prev = int(values[i])
        _write_varint_u64(out, _zigzag64(d))
    return bytes(out)


def decode_delta_zigzag_varint_int(blob: bytes) -> List[int]:
    if not blob:
        return []
    pos = 0
    v0 = int.from_bytes(blob[pos : pos + 8], 'little', signed=True)
    pos += 8
    out = [v0]
    cur = v0
    while pos < len(blob):
        u, pos = _read_varint_u64(blob, pos)
        d = _unzigzag64(u)
        cur = cur + d
        out.append(cur)
    return out


def encode_dictionary_int(values: List[int]) -> bytes:
    """Preserve int64 lanes; dictionary construction and index coding are C++."""
    raw = struct.pack(f'<{len(values)}q', *(int(value) for value in values))
    return _run_cpp_action('dictionary64_action_cli', 'encode', raw)


def decode_dictionary_int(blob: bytes) -> List[int]:
    if len(blob) < 8:
        return []
    raw = _run_cpp_action('dictionary64_action_cli', 'decode', blob)
    return list(struct.unpack(f'<{len(raw) // 8}q', raw))


def encode_dictionary_float(values: List[float]) -> bytes:
    """Pass raw float64 bits, including negative zero and NaN payloads."""
    raw = struct.pack(f'<{len(values)}d', *(float(value) for value in values))
    return _run_cpp_action('dictionary64_action_cli', 'encode', raw)


def decode_dictionary_float(blob: bytes) -> List[float]:
    if len(blob) < 8:
        return []
    raw = _run_cpp_action('dictionary64_action_cli', 'decode', blob)
    return list(struct.unpack(f'<{len(raw) // 8}d', raw))


def encode_rle_int(values: List[int]) -> bytes:
    out = bytearray()
    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and int(values[j]) == int(values[i]):
            j += 1
        run = j - i
        out.extend(int(values[i]).to_bytes(8, 'little', signed=True))
        _write_varint_u64(out, run)
        i = j
    return bytes(out)


def decode_rle_int(blob: bytes) -> List[int]:
    out: list[int] = []
    pos = 0
    while pos < len(blob):
        val = int.from_bytes(blob[pos : pos + 8], 'little', signed=True)
        pos += 8
        run, pos = _read_varint_u64(blob, pos)
        out.extend([val] * run)
    return out


def _byte_stream_split_float64(raw: bytes) -> bytes:
    n = len(raw) // 8
    lanes = [bytearray(n) for _ in range(8)]
    for i in range(n):
        base = i * 8
        for b in range(8):
            lanes[b][i] = raw[base + b]
    return b"".join(bytes(lane) for lane in lanes)


def _byte_stream_split_inverse_float64(bss: bytes, n: int) -> bytes:
    raw = bytearray(n * 8)
    for b in range(8):
        base_off = b * n
        for i in range(n):
            raw[i * 8 + b] = bss[base_off + i]
    return bytes(raw)


def encode_bss_lz4_float(values: List[float]) -> bytes:
    import lz4.frame

    raw = struct.pack('<%dd' % len(values), *(float(v) for v in values))
    bss = _byte_stream_split_float64(raw)
    return lz4.frame.compress(bss)


def decode_bss_lz4_float(blob: bytes, n: int) -> bytes:
    import lz4.frame

    bss = lz4.frame.decompress(blob)
    return _byte_stream_split_inverse_float64(bss, n)


def bench_delta_zigzag_varint_columns(columns: List[List[int]]):
    tot_o = tot_c = tot_enc = tot_dec = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        vals = [int(x) for x in col]
        tot_o += len(vals) * BENCHMARK_INT_BYTES
        t0 = time.perf_counter_ns()
        comp = encode_delta_zigzag_varint_int(vals)
        t1 = time.perf_counter_ns()
        dec = decode_delta_zigzag_varint_int(comp)
        t2 = time.perf_counter_ns()
        if dec != vals:
            continue
        accumulate_long_list_loss(vals, dec, loss_acc)
        tot_c += len(comp)
        tot_enc += t1 - t0
        tot_dec += t2 - t1
    return tot_o, tot_c, tot_enc, tot_dec, loss_acc[0], loss_acc[1]


def bench_dictionary_int_columns(columns: List[List[int]]):
    tot_o = tot_c = tot_enc = tot_dec = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        vals = [int(x) for x in col]
        t0 = time.perf_counter_ns()
        comp = encode_dictionary_int(vals)
        t1 = time.perf_counter_ns()
        dec = decode_dictionary_int(comp)
        t2 = time.perf_counter_ns()
        if dec != vals:
            continue
        accumulate_long_list_loss(vals, dec, loss_acc)
        tot_o += len(vals) * BENCHMARK_INT_BYTES
        tot_c += len(comp)
        tot_enc += t1 - t0
        tot_dec += t2 - t1
    return tot_o, tot_c, tot_enc, tot_dec, loss_acc[0], loss_acc[1]


def bench_dictionary_float_columns(columns: List[List[float]]):
    tot_o = tot_c = tot_enc = tot_dec = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        t0 = time.perf_counter_ns()
        comp = encode_dictionary_float(col)
        t1 = time.perf_counter_ns()
        dec = decode_dictionary_float(comp)
        t2 = time.perf_counter_ns()
        if len(dec) != len(col) or any(
            struct.pack('<d', float(a)) != struct.pack('<d', float(b))
            for a, b in zip(dec, col)
        ):
            continue
        accumulate_double_list_loss([float(x) for x in col], [float(x) for x in dec], loss_acc)
        tot_o += len(col) * BENCHMARK_FLOAT_BYTES
        tot_c += len(comp)
        tot_enc += t1 - t0
        tot_dec += t2 - t1
    return tot_o, tot_c, tot_enc, tot_dec, loss_acc[0], loss_acc[1]


def bench_rle_int_columns(columns: List[List[int]]):
    tot_o = tot_c = tot_enc = tot_dec = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        vals = [int(x) for x in col]
        tot_o += len(vals) * BENCHMARK_INT_BYTES
        t0 = time.perf_counter_ns()
        comp = encode_rle_int(vals)
        t1 = time.perf_counter_ns()
        dec = decode_rle_int(comp)
        t2 = time.perf_counter_ns()
        if dec != vals:
            continue
        accumulate_long_list_loss(vals, dec, loss_acc)
        tot_c += len(comp)
        tot_enc += t1 - t0
        tot_dec += t2 - t1
    return tot_o, tot_c, tot_enc, tot_dec, loss_acc[0], loss_acc[1]


def bench_bss_lz4_float_columns(columns: List[List[float]]):
    tot_o = tot_c = tot_enc = tot_dec = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        n = len(col)
        if n == 0:
            continue
        expected = struct.pack('<%dd' % n, *(float(v) for v in col))
        t0 = time.perf_counter_ns()
        comp = encode_bss_lz4_float(col)
        t1 = time.perf_counter_ns()
        raw = decode_bss_lz4_float(comp, n)
        t2 = time.perf_counter_ns()
        if raw != expected:
            continue
        accumulate_byte_loss(expected, raw, loss_acc)
        tot_o += n * BENCHMARK_FLOAT_BYTES
        tot_c += len(comp)
        tot_enc += t1 - t0
        tot_dec += t2 - t1
    return tot_o, tot_c, tot_enc, tot_dec, loss_acc[0], loss_acc[1]
