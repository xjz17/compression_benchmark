"""
Simple8b (Vo Ngoc Anh & Moffat / Lemire FastPFOR layout) on uint32 streams.

CSV keys:

- ``simple8b_int``: int32 columns (values as unsigned ``uint32`` for the codec; same LE layout as ``<i``).
- ``simple8b_float``: same **decimal-scaled int** path as ``leco_float`` (per-column ``10^p``, lossless ``int32``
  quantization), then Simple8b on **int32 LE** words (unsigned view). Wire: LE ``i32 col_scale``, then the same
  Simple8b block as ``simple8b_int`` (LE ``u32 num_values``, LE ``u32 num_u64_words``, ``u64`` words).

- ``simple8b_int`` wire: LE ``u32 num_values``, LE ``u32 num_u64_words``, then ``num_u64_words`` × LE ``u64``.

Encoder follows the **tail** ``while (ValuesRemaining > 0)`` decision order from ``headers/simple8b.h``
(no separate ``ValuesRemaining >= 240`` fast path) so Python / Java / C++ stay bit-identical.
"""

from __future__ import annotations

from .benchmark_wire import column_original_size

import struct
import time
from typing import List, Optional, Sequence, Tuple

_LOGDESC = 4
_MASK64 = (1 << 64) - 1
# Same cap as ``ExternalCompressionBench`` / ``research_codec_python`` for float decimal scaling.
_SIMPLE8B_FLOAT_MAX_DECIMAL_SCALE = 15


def _float_raw_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _min_decimal_scale_exact_float(v: float) -> int:
    """Smallest ``s >= 0`` with bit-identical ``(float)(round(v*10^s)/10^s) == v``, or ``-1``."""
    if v != v or v in (float("inf"), float("-inf")):
        return -1
    if float(v) == 0.0:
        return 0
    bits = _float_raw_bits(v)
    for s in range(_SIMPLE8B_FLOAT_MAX_DECIMAL_SCALE + 1):
        pow10 = 10.0**s
        m = int(round(float(v) * pow10))
        if m < -(2**31) or m > 2**31 - 1:
            return -1
        r = float(m) / pow10
        if _float_raw_bits(r) == bits:
            return s
    return -1

# (max_count, log2_bits_per_value, selector_nibble) — selectors 0 and 1 use log 0 specials
_SEL_ROWS = (
    (240, 0, 0),
    (120, 0, 1),
    (60, 1, 2),
    (30, 2, 3),
    (20, 3, 4),
    (15, 4, 5),
    (12, 5, 6),
    (10, 6, 7),
    (8, 7, 8),
    (7, 8, 9),
    (6, 10, 10),
    (5, 12, 11),
    (4, 15, 12),
    (3, 20, 13),
    (2, 30, 14),
    (1, 60, 15),
)
_SEL_BY_SEL = {r[2]: r for r in _SEL_ROWS}


def _which(w: int) -> int:
    return (w >> 60) & 0xF


def _tryme(buf: Sequence[int], rem: int, num: int, log: int) -> bool:
    c = rem if rem < num else num
    if log >= 32:
        return True
    lim = 1 << log
    for i in range(c):
        if int(buf[i]) >= lim:
            return False
    return True


def _compose_word(sel: int, vals: Sequence[int], log: int) -> int:
    w = sel & 0xF
    if log == 0:
        if sel == 0:
            return 0
        if sel == 1:
            return (1 << 60) & _MASK64
        raise ValueError("simple8b: internal compose log=0")
    mask = (1 << log) - 1 if log < 32 else 0xFFFFFFFF
    for v in vals:
        w = (w << log) | (int(v) & mask)
        w &= _MASK64
    total_bits = _LOGDESC + log * len(vals)
    sh = 64 - total_bits
    if sh < 0 or sh > 64:
        raise ValueError("simple8b: compose shift")
    w = (w << sh) & _MASK64
    return w


def simple8b_encode_u32(values: Sequence[int]) -> List[int]:
    """Return list of uint64 words (as Python int, masked to 64 bits)."""
    vals = [int(v) & 0xFFFFFFFF for v in values]
    out: List[int] = []
    pos = 0
    n = len(vals)
    while pos < n:
        rem = n - pos
        chunk = vals[pos:]
        wrote = False
        for maxnum, log, sel in _SEL_ROWS:
            if not _tryme(chunk, rem, maxnum, log):
                continue
            take = rem if rem < maxnum else maxnum
            if sel <= 1:
                w = 0 if sel == 0 else ((1 << 60) & _MASK64)
            else:
                w = _compose_word(sel, chunk[:take], log)
            out.append(w)
            pos += take
            wrote = True
            break
        if not wrote:
            raise RuntimeError("simple8b: encode stuck")
    return out


def simple8b_decode_u32(words: Sequence[int], n_values: int) -> List[int]:
    out: List[int] = []
    wi = 0
    words = [int(w) & _MASK64 for w in words]
    while len(out) < n_values:
        if wi >= len(words):
            raise ValueError("simple8b: truncated compressed stream")
        w = words[wi]
        wi += 1
        sel = _which(w)
        rem = n_values - len(out)
        if sel == 0:
            take = rem if rem < 240 else 240
            out.extend([0] * take)
        elif sel == 1:
            take = rem if rem < 120 else 120
            out.extend([0] * take)
        else:
            if sel not in _SEL_BY_SEL or sel < 2:
                raise ValueError("simple8b: bad selector")
            maxnum, log, _ = _SEL_BY_SEL[sel]
            take = rem if rem < maxnum else maxnum
            if log == 0:
                raise ValueError("simple8b: decode log=0 for sel>=2")
            mask = (1 << log) - 1 if log < 32 else 0xFFFFFFFF
            for k in range(take):
                sh = 64 - _LOGDESC - log - k * log
                out.append((w >> sh) & mask)
    if len(out) != n_values:
        raise ValueError("simple8b: length mismatch")
    return out


def simple8b_encode_bytes_from_i32_column(raw_i32_le: bytes) -> bytes:
    """Encode any little-endian **uint32** stream (``n`` × 4 bytes), including int32 or float32 raw bits."""
    if len(raw_i32_le) % 4 != 0:
        raise ValueError("simple8b: column length must be multiple of 4")
    n = len(raw_i32_le) // 4
    u32 = list(struct.unpack("<%dI" % n, raw_i32_le))
    words = simple8b_encode_u32(u32)
    nw = len(words)
    b = bytearray(8 + nw * 8)
    struct.pack_into("<II", b, 0, n, nw)
    for i, w in enumerate(words):
        struct.pack_into("<Q", b, 8 + i * 8, int(w) & _MASK64)
    return bytes(b)


def simple8b_decode_bytes_to_i32_column(blob: bytes) -> bytes:
    """Decode to little-endian **uint32** stream (``n`` × 4 bytes)."""
    if len(blob) < 8:
        raise ValueError("simple8b: truncated header")
    n, nw = struct.unpack_from("<II", blob, 0)
    need = 8 + nw * 8
    if nw < 0 or len(blob) < need:
        raise ValueError("simple8b: truncated body")
    words = []
    for i in range(nw):
        words.append(struct.unpack_from("<Q", blob, 8 + i * 8)[0])
    u32 = simple8b_decode_u32(words, n)
    return struct.pack("<%dI" % n, *u32)


def bench_simple8b_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    tot_o = tot_c = te = td = 0
    for col in columns:
        if not col:
            continue
        raw = struct.pack("<%di" % len(col), *col)
        t0 = time.perf_counter_ns()
        comp = simple8b_encode_bytes_from_i32_column(raw)
        t1 = time.perf_counter_ns()
        dec = simple8b_decode_bytes_to_i32_column(comp)
        t2 = time.perf_counter_ns()
        if dec != raw:
            raise RuntimeError("simple8b_int roundtrip failed")
        tot_o += column_original_size(len(col), "int")
        tot_c += len(comp)
        te += t1 - t0
        td += t2 - t1
    return tot_o, tot_c, te, td


def _try_pack_scaled_float_column_i32_le(col: List[float]) -> Optional[Tuple[int, bytes]]:
    col_scale = 0
    for v in col:
        s = _min_decimal_scale_exact_float(float(v))
        if s < 0:
            return None
        if s > col_scale:
            col_scale = s
    pow10 = 10.0**col_scale
    ints: List[int] = []
    for v in col:
        m = int(round(float(v) * pow10))
        if m < -(2**31) or m > 2**31 - 1:
            return None
        r = float(m) / pow10
        if _float_raw_bits(r) != _float_raw_bits(float(v)):
            return None
        ints.append(m)
    return col_scale, struct.pack("<%di" % len(ints), *ints)


def bench_simple8b_double_columns(
    columns: List[List[float]], max_points_per_column: Optional[List[int]] = None
) -> Tuple[int, int, int, int]:
    from .int64_column_codec import simple8b_decode_int64, simple8b_encode_int64
    from .research_codec_python import (
        scale_float_column_to_ints,
        scale_float_column_to_ints_with_max_point,
        _double_bits,
        _double_bits_i64,
        min_decimal_scale_exact_float,
    )

    tot_o = tot_c = te = td = 0
    for i, col in enumerate(columns):
        if not col:
            continue
        cap = (
            max_points_per_column[i]
            if max_points_per_column is not None and i < len(max_points_per_column)
            else None
        )
        if cap is not None:
            data = scale_float_column_to_ints_with_max_point(col, cap)
            if data is None:
                data = scale_float_column_to_ints(col)
        else:
            data = scale_float_column_to_ints(col)
        if data is None:
            continue
        col_scale = -1
        if cap is not None:
            cap_scale = max(0, int(cap))
            for s in range(cap_scale, -1, -1):
                pow10 = 10.0**s
                candidate = [int(round(float(v) * pow10)) for v in col]
                if candidate == data:
                    col_scale = s
                    break
        if col_scale < 0:
            col_scale = 0
            for v in col:
                s = min_decimal_scale_exact_float(float(v))
                if s < 0:
                    col_scale = -1
                    break
                col_scale = max(col_scale, s)
        if col_scale < 0:
            continue
        pow10 = 10.0**col_scale
        t0 = time.perf_counter_ns()
        try:
            body = simple8b_encode_int64(data)
        except Exception:
            continue
        comp = struct.pack("<i", col_scale) + body
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        try:
            recovered = simple8b_decode_int64(body, len(data))
        except Exception:
            continue
        t3 = time.perf_counter_ns()
        if cap is not None:
            bits_ok = recovered == data
        else:
            bits_ok = True
            for i, im in enumerate(recovered):
                fv = float(col[i])
                if fv == 0.0:
                    if int(im) != _double_bits_i64(fv):
                        bits_ok = False
                        break
                    continue
                r = float(im) / pow10
                if _double_bits(r) != _double_bits(fv):
                    bits_ok = False
                    break
        if not bits_ok:
            continue
        tot_o += column_original_size(len(col), "float")
        tot_c += len(comp)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td


def bench_simple8b_int64_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    from .int64_column_codec import simple8b_decode_int64, simple8b_encode_int64

    tot_o = tot_c = te = td = 0
    for col in columns:
        if not col:
            continue
        data = [int(v) for v in col]
        t0 = time.perf_counter_ns()
        try:
            payload = simple8b_encode_int64(data)
        except Exception:
            continue
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        try:
            recovered = simple8b_decode_int64(payload, len(data))
        except Exception:
            continue
        t3 = time.perf_counter_ns()
        if recovered != data:
            continue
        tot_o += column_original_size(len(col), "int")
        tot_c += len(payload)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td


def bench_simple8b_float_columns(
    columns: List[List[float]], max_points_per_column: Optional[List[int]] = None
) -> Tuple[int, int, int, int]:
    """Legacy alias; float path now uses double/int64 internals."""
    return bench_simple8b_double_columns(columns, max_points_per_column)
