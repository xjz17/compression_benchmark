"""Sub-column integer codec with bitpack/RLE/dictionary subcolumn bodies.

The float benchmark path first applies decimal scaling, so the core codec stays integer-only:
``scaling -> subcolumn -> bitpacking/rle/dictionary``.
"""

from __future__ import annotations

import struct
import time
from typing import List, Sequence, Tuple

from .benchmark_wire import column_original_size


BLOCK_SIZE = 512


def _bit_width(v: int) -> int:
    return int(v).bit_length()


class _BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.cur = 0
        self.used = 0

    def write(self, value: int, bits: int) -> None:
        if bits <= 0:
            return
        v = int(value)
        for i in range(bits - 1, -1, -1):
            self.cur = (self.cur << 1) | ((v >> i) & 1)
            self.used += 1
            if self.used == 8:
                self.buf.append(self.cur)
                self.cur = 0
                self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.buf.append(self.cur << (8 - self.used))
            self.cur = 0
            self.used = 0
        return bytes(self.buf)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, bits: int) -> int:
        out = 0
        for _ in range(bits):
            byte = self.data[self.pos >> 3]
            bit = 7 - (self.pos & 7)
            out = (out << 1) | ((byte >> bit) & 1)
            self.pos += 1
        return out


def _pack(values: Sequence[int], width: int) -> bytes:
    if width <= 0 or not values:
        return b""
    total_bits = len(values) * width
    mask = (1 << width) - 1
    acc = 0
    for v in values:
        acc = (acc << width) | (int(v) & mask)
    acc <<= (-total_bits) & 7
    return acc.to_bytes((total_bits + 7) // 8, "big")


def _unpack(data: bytes, count: int, width: int) -> List[int]:
    if width <= 0:
        return [0] * count
    total_bits = count * width
    acc = int.from_bytes(data, "big") >> ((-total_bits) & 7)
    mask = (1 << width) - 1
    out = [0] * count
    for idx in range(count - 1, -1, -1):
        out[idx] = acc & mask
        acc >>= width
    return out


def _choose_beta(max_bits: int) -> int:
    if max_bits <= 1:
        return 1
    best_beta = 1
    best_cost = 1 << 62
    for beta in (1, 2, 3, 4):
        groups = (max_bits + beta - 1) // beta
        cost = groups * beta
        if cost < best_cost:
            best_cost = cost
            best_beta = beta
    return best_beta


def _encode_subcolumn(vals: List[int], beta: int, mode: int = 2) -> bytes:
    n = len(vals)
    max_v = max(vals) if vals else 0
    width = _bit_width(max_v)
    bitpack_body = _pack(vals, width)
    best_type = 0
    best_body = bytes([width]) + struct.pack("<I", len(bitpack_body)) + bitpack_body

    if mode >= 1:
        runs: List[tuple[int, int]] = []
        if vals:
            cur = vals[0]
            start = 0
            for i, v in enumerate(vals[1:], 1):
                if v != cur:
                    runs.append((i - start, cur))
                    cur = v
                    start = i
            runs.append((n - start, cur))
        run_width = _bit_width(max((r[0] for r in runs), default=0))
        rle_value_width = max(width, 1)
        rle_lengths = _pack([r[0] for r in runs], run_width)
        rle_values = _pack([r[1] for r in runs], rle_value_width)
        rle_body = (
            bytes([run_width, rle_value_width])
            + struct.pack("<HII", len(runs), len(rle_lengths), len(rle_values))
            + rle_lengths
            + rle_values
        )
        if len(rle_body) < len(best_body):
            best_type = 1
            best_body = rle_body

    if mode >= 2:
        keys = sorted(set(vals))
        if 0 < len(keys) <= min(256, 1 << beta):
            code_width = _bit_width(len(keys) - 1)
            key_width = max(width, 1)
            code_map = {v: i for i, v in enumerate(keys)}
            key_body = _pack(keys, key_width)
            code_body = _pack([code_map[v] for v in vals], code_width)
            dict_body = (
                bytes([key_width, code_width])
                + struct.pack("<HII", len(keys), len(key_body), len(code_body))
                + key_body
                + code_body
            )
            if len(dict_body) < len(best_body):
                best_type = 2
                best_body = dict_body

    return bytes([best_type]) + best_body


def _decode_subcolumn(data: bytes, pos: int, n: int) -> tuple[List[int], int]:
    typ = data[pos]
    pos += 1
    if typ == 0:
        width = data[pos]
        body_len = struct.unpack_from("<I", data, pos + 1)[0]
        pos += 5
        vals = _unpack(data[pos : pos + body_len], n, width)
        return vals, pos + body_len
    if typ == 1:
        run_width = data[pos]
        value_width = data[pos + 1]
        run_count, len_len, val_len = struct.unpack_from("<HII", data, pos + 2)
        pos += 12
        lengths = _unpack(data[pos : pos + len_len], run_count, run_width)
        pos += len_len
        values = _unpack(data[pos : pos + val_len], run_count, value_width)
        pos += val_len
        out: List[int] = []
        for ln, val in zip(lengths, values):
            out.extend([val] * ln)
        return out[:n], pos
    key_width = data[pos]
    code_width = data[pos + 1]
    key_count, key_len, code_len = struct.unpack_from("<HII", data, pos + 2)
    pos += 12
    keys = _unpack(data[pos : pos + key_len], key_count, key_width)
    pos += key_len
    codes = _unpack(data[pos : pos + code_len], n, code_width)
    pos += code_len
    return [keys[c] for c in codes], pos


def encode_int_column(
    values: Sequence[int],
    block_size: int = BLOCK_SIZE,
    mode: int = 2,
    beta_override: int | None = None,
) -> bytes:
    out = bytearray(struct.pack("<II", len(values), block_size))
    for base in range(0, len(values), block_size):
        block = [int(v) for v in values[base : base + block_size]]
        mn = min(block) if block else 0
        residuals = [v - mn for v in block]
        max_bits = _bit_width(max(residuals) if residuals else 0)
        beta = max(1, int(beta_override)) if beta_override is not None else _choose_beta(max_bits)
        groups = (max_bits + beta - 1) // beta if max_bits else 0
        body = bytearray(struct.pack("<qHBB", mn, len(block), max_bits, beta))
        mask = (1 << beta) - 1
        for g in range(groups):
            sub = [(v >> (g * beta)) & mask for v in residuals]
            enc = _encode_subcolumn(sub, beta, mode)
            body += struct.pack("<I", len(enc)) + enc
        out += body
    return bytes(out)


def decode_int_column(blob: bytes) -> List[int]:
    n, block_size = struct.unpack_from("<II", blob, 0)
    pos = 8
    out: List[int] = []
    while len(out) < n:
        mn, count, max_bits, beta = struct.unpack_from("<qHBB", blob, pos)
        pos += 12
        groups = (max_bits + beta - 1) // beta if max_bits else 0
        residuals = [0] * count
        for g in range(groups):
            body_len = struct.unpack_from("<I", blob, pos)[0]
            pos += 4
            sub, end = _decode_subcolumn(blob, pos, count)
            pos += body_len
            if end > pos:
                pos = end
            for i, v in enumerate(sub):
                residuals[i] |= v << (g * beta)
        out.extend([mn + v for v in residuals])
    if block_size <= 0:
        raise ValueError("subcolumn: invalid block size")
    return out[:n]


def bench_subcolumn_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    orig = comp = ct = dt = 0
    for col in columns:
        if not col:
            continue
        t0 = time.perf_counter_ns()
        enc = encode_int_column(col)
        t1 = time.perf_counter_ns()
        dec = decode_int_column(enc)
        t2 = time.perf_counter_ns()
        if dec != [int(v) for v in col]:
            continue
        orig += column_original_size(len(col), "int")
        comp += len(enc)
        ct += t1 - t0
        dt += t2 - t1
    return orig, comp, ct, dt


def bench_subcolumn_double_columns(
    columns: List[List[float]], max_points_per_column: Sequence[int] | None = None
) -> Tuple[int, int, int, int]:
    from .research_codec_python import (
        min_decimal_scale_exact_float,
        scale_float_column_to_ints,
        scale_float_column_to_ints_with_max_point,
    )

    orig = comp = ct = dt = 0
    for i, col in enumerate(columns):
        if not col:
            continue
        cap = (
            int(max_points_per_column[i])
            if max_points_per_column is not None and i < len(max_points_per_column)
            else None
        )
        ints = (
            scale_float_column_to_ints_with_max_point(col, cap)
            if cap is not None
            else scale_float_column_to_ints(col)
        )
        if ints is None:
            ints = scale_float_column_to_ints(col)
        if ints is None:
            continue
        scale = -1
        if cap is not None:
            for s in range(max(0, cap), -1, -1):
                pow10 = 10.0**s
                if [int(round(float(v) * pow10)) for v in col] == ints:
                    scale = s
                    break
        if scale < 0:
            scale = 0
            for v in col:
                s = min_decimal_scale_exact_float(float(v))
                if s < 0:
                    scale = -1
                    break
                scale = max(scale, s)
        if scale < 0:
            continue
        t0 = time.perf_counter_ns()
        body = encode_int_column(ints)
        enc = struct.pack("<i", scale) + body
        t1 = time.perf_counter_ns()
        sc = struct.unpack_from("<i", enc, 0)[0]
        dec = decode_int_column(enc[4:])
        t2 = time.perf_counter_ns()
        if sc != scale or dec != ints:
            continue
        orig += column_original_size(len(col), "float")
        comp += len(enc)
        ct += t1 - t0
        dt += t2 - t1
    return orig, comp, ct, dt
