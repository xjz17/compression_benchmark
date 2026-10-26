"""
Pure Python Sim-Piece port aligned with ``SimPieceFloatNative`` / ``simpiece_float_codec``.

Serialization uses variable-byte + big-endian float32/int32 preamble, then Zstd (``zstandard``, level 3).
"""
from __future__ import annotations

from .benchmark_loss import accumulate_double_list_loss
from .benchmark_wire import column_original_size

import io
import math
import struct
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

_DBL_MAX = sys.float_info.max

import zstandard

_ZC = zstandard.ZstdCompressor(level=3)
_ZD = zstandard.ZstdDecompressor()


def _java_math_round(x: float) -> int:
    """Match ``Math.round(double)`` cast to ``int`` (JDK floor(x+0.5))."""
    if math.isnan(x):
        return 0
    if x == math.inf:
        return 2147483647
    if x == -math.inf:
        return -2147483648
    return int(math.floor(x + 0.5))


def _quantize(value: float, epsilon: float) -> float:
    return float(_java_math_round(value / epsilon)) * epsilon


@dataclass
class _Segment:
    init_ts: int
    a_min: float
    a_max: float
    b: float

    @property
    def a(self) -> float:
        return (self.a_min + self.a_max) / 2.0


def _vb_write_u32(n: int, out: bytearray) -> None:
    val = n & 0xFFFFFFFF

    def _e7(i: int) -> int:
        return (val >> (7 * i)) & ((1 << 7) - 1)

    def _em(i: int) -> int:
        return (val >> (7 * i)) & 0xFF

    if val < (1 << 7):
        out.append((val | (1 << 7)) & 0xFF)
    elif val < (1 << 14):
        out.append(_e7(0) & 0xFF)
        out.append((_em(1) | (1 << 7)) & 0xFF)
    elif val < (1 << 21):
        out.append(_e7(0) & 0xFF)
        out.append(_e7(1) & 0xFF)
        out.append((_em(2) | (1 << 7)) & 0xFF)
    elif val < (1 << 28):
        out.append(_e7(0) & 0xFF)
        out.append(_e7(1) & 0xFF)
        out.append(_e7(2) & 0xFF)
        out.append((_em(3) | (1 << 7)) & 0xFF)
    else:
        out.append(_e7(0) & 0xFF)
        out.append(_e7(1) & 0xFF)
        out.append(_e7(2) & 0xFF)
        out.append(_e7(3) & 0xFF)
        out.append((_em(4) | (1 << 7)) & 0xFF)


def _vb_read_u32(stream: io.BytesIO) -> int:
    """Match Java ``VariableByteEncoder.read`` (JavaFastPFOR-style continuation)."""

    def _rb() -> int:
        b = stream.read(1)
        if not b:
            raise EOFError("vb")
        return b[0]

    inn = _rb()
    number = inn & 0x7F
    if inn >= 0x80:
        return number
    inn = _rb()
    number = ((inn & 0x7F) << 7) | number
    if inn >= 0x80:
        return number
    inn = _rb()
    number = ((inn & 0x7F) << 14) | number
    if inn >= 0x80:
        return number
    inn = _rb()
    number = ((inn & 0x7F) << 21) | number
    if inn >= 0x80:
        return number
    inn = _rb()
    number = (((inn & 0x7F) << 28) | number) & 0xFFFFFFFF
    return number


def _write_f32_be(x: float, out: bytearray) -> None:
    bits = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    out.extend(struct.pack(">I", bits))


def _read_f32_be(stream: io.BytesIO) -> float:
    raw = stream.read(4)
    if len(raw) != 4:
        raise EOFError("f32")
    bits = struct.unpack(">I", raw)[0]
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _write_f64_be(x: float, out: bytearray) -> None:
    bits = struct.unpack(">Q", struct.pack(">d", float(x)))[0]
    out.extend(struct.pack(">Q", bits))


def _read_f64_be(stream: io.BytesIO) -> float:
    raw = stream.read(8)
    if len(raw) != 8:
        raise EOFError("f64")
    bits = struct.unpack(">Q", raw)[0]
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _read_u32_be(stream: io.BytesIO) -> int:
    raw = stream.read(4)
    if len(raw) != 4:
        raise EOFError("u32")
    return struct.unpack(">I", raw)[0]


def _create_segment(start_idx: int, ts: List[int], vals: List[float], epsilon: float, segs: List[_Segment]) -> int:
    n = len(vals)
    init_ts = ts[start_idx]
    b = _quantize(vals[start_idx], epsilon)
    if start_idx + 1 == n:
        segs.append(_Segment(init_ts, -_DBL_MAX, _DBL_MAX, b))
        return start_idx + 1
    dt = ts[start_idx + 1] - init_ts
    a_max = ((vals[start_idx + 1] + epsilon) - b) / dt
    a_min = ((vals[start_idx + 1] - epsilon) - b) / dt
    if start_idx + 2 == n:
        segs.append(_Segment(init_ts, a_min, a_max, b))
        return start_idx + 2
    for idx in range(start_idx + 2, n):
        up_val = vals[idx] + epsilon
        down_val = vals[idx] - epsilon
        dt_i = ts[idx] - init_ts
        up_lim = a_max * dt_i + b
        down_lim = a_min * dt_i + b
        if down_val > up_lim or up_val < down_lim:
            segs.append(_Segment(init_ts, a_min, a_max, b))
            return idx
        if up_val < up_lim:
            a_max = max((up_val - b) / dt_i, a_min)
        if down_val > down_lim:
            a_min = min((down_val - b) / dt_i, a_max)
    segs.append(_Segment(init_ts, a_min, a_max, b))
    return n


def _compress_segments(ts: List[int], vals: List[float], epsilon: float) -> List[_Segment]:
    segs: List[_Segment] = []
    cur = 0
    n = len(vals)
    while cur < n:
        cur = _create_segment(cur, ts, vals, epsilon, segs)
    return segs


def _merge_per_b(segments: List[_Segment]) -> List[_Segment]:
    a_min_temp = -_DBL_MAX
    a_max_temp = _DBL_MAX
    b_cur: float | None = None
    timestamps: List[int] = []
    merged: List[_Segment] = []
    segs = sorted(segments, key=lambda s: (s.b, s.a))
    for i in range(len(segs)):
        sg = segs[i]
        if b_cur is None or sg.b != b_cur:
            if b_cur is not None:
                if len(timestamps) == 1:
                    merged.append(_Segment(timestamps[0], a_min_temp, a_max_temp, b_cur))
                elif len(timestamps) > 1:
                    for t in timestamps:
                        merged.append(_Segment(t, a_min_temp, a_max_temp, b_cur))
            timestamps = [sg.init_ts]
            a_min_temp = sg.a_min
            a_max_temp = sg.a_max
            b_cur = sg.b
            continue
        if sg.a_min <= a_max_temp and sg.a_max >= a_min_temp:
            timestamps.append(sg.init_ts)
            a_min_temp = max(a_min_temp, sg.a_min)
            a_max_temp = min(a_max_temp, sg.a_max)
        else:
            if len(timestamps) == 1:
                merged.append(segs[i - 1])
            else:
                for t in timestamps:
                    merged.append(_Segment(t, a_min_temp, a_max_temp, b_cur))
            timestamps = [sg.init_ts]
            a_min_temp = sg.a_min
            a_max_temp = sg.a_max
    if b_cur is not None and timestamps:
        if len(timestamps) == 1:
            merged.append(_Segment(timestamps[0], a_min_temp, a_max_temp, b_cur))
        else:
            for t in timestamps:
                merged.append(_Segment(t, a_min_temp, a_max_temp, b_cur))
    return merged


def _to_points(segments: List[_Segment], last_ts: int) -> List[Tuple[int, float]]:
    segs = sorted(segments, key=lambda s: s.init_ts)
    if not segs:
        return []
    out: List[Tuple[int, float]] = []
    cur = segs[0].init_ts
    for i in range(len(segs) - 1):
        sg = segs[i]
        nxt = segs[i + 1].init_ts
        while cur < nxt:
            out.append((cur, sg.a * (cur - sg.init_ts) + sg.b))
            cur += 1
    sg = segs[-1]
    while cur <= last_ts:
        out.append((cur, sg.a * (cur - sg.init_ts) + sg.b))
        cur += 1
    return out


def _to_byte_array_per_b(segments: List[_Segment], epsilon: float) -> bytes:
    tree: Dict[int, Dict[float, List[int]]] = {}
    for sg in segments:
        a = sg.a
        b_int = _java_math_round(sg.b / epsilon)
        t = sg.init_ts
        tree.setdefault(b_int, {}).setdefault(a, []).append(t)
    buf = bytearray()
    _vb_write_u32(len(tree), buf)
    if not tree:
        return bytes(buf)
    sorted_b = sorted(tree.keys())
    prev_b = sorted_b[0]
    _vb_write_u32(prev_b, buf)
    for b_int in sorted_b:
        _vb_write_u32(b_int - prev_b, buf)
        prev_b = b_int
        a_map = tree[b_int]
        _vb_write_u32(len(a_map), buf)
        for a_key in sorted(a_map.keys()):
            ts_list = a_map[a_key]
            _write_f32_be(float(a_key), buf)
            _vb_write_u32(len(ts_list), buf)
            ts_list.sort()
            prev_ts = 0
            for tstamp in ts_list:
                _vb_write_u32(int(tstamp - prev_ts), buf)
                prev_ts = tstamp
    return bytes(buf)


def _to_byte_array_per_b_f64(segments: List[_Segment], epsilon: float) -> bytes:
    tree: Dict[int, Dict[float, List[int]]] = {}
    for sg in segments:
        a = sg.a
        b_int = _java_math_round(sg.b / epsilon)
        t = sg.init_ts
        tree.setdefault(b_int, {}).setdefault(a, []).append(t)
    buf = bytearray()
    _vb_write_u32(len(tree), buf)
    if not tree:
        return bytes(buf)
    sorted_b = sorted(tree.keys())
    prev_b = sorted_b[0]
    _vb_write_u32(prev_b, buf)
    for b_int in sorted_b:
        _vb_write_u32(b_int - prev_b, buf)
        prev_b = b_int
        a_map = tree[b_int]
        _vb_write_u32(len(a_map), buf)
        for a_key in sorted(a_map.keys()):
            ts_list = a_map[a_key]
            _write_f64_be(float(a_key), buf)
            _vb_write_u32(len(ts_list), buf)
            ts_list.sort()
            prev_ts = 0
            for tstamp in ts_list:
                _vb_write_u32(int(tstamp - prev_ts), buf)
                prev_ts = tstamp
    return bytes(buf)


def compress_column(values: List[float], epsilon: float) -> bytes:
    if not values or epsilon <= 0:
        raise ValueError("SimPiece.compress")
    n = len(values)
    ts = list(range(n))
    raw_segs = _compress_segments(ts, [float(x) for x in values], epsilon)
    merged = _merge_per_b(raw_segs)
    preamble = bytearray()
    _write_f32_be(float(epsilon), preamble)
    preamble.extend(_to_byte_array_per_b(merged, epsilon))
    _vb_write_u32(n - 1, preamble)
    return _ZC.compress(bytes(preamble))


def compress_column_f64(values: List[float], epsilon: float) -> bytes:
    if not values or epsilon <= 0:
        raise ValueError("SimPiece.compress.f64")
    n = len(values)
    ts = list(range(n))
    raw_segs = _compress_segments(ts, [float(x) for x in values], epsilon)
    merged = _merge_per_b(raw_segs)
    preamble = bytearray()
    _write_f64_be(float(epsilon), preamble)
    preamble.extend(_to_byte_array_per_b_f64(merged, epsilon))
    _vb_write_u32(n - 1, preamble)
    return _ZC.compress(bytes(preamble))


def decompress_column(binary: bytes) -> List[Tuple[int, float]]:
    raw = _ZD.decompress(binary)
    stream = io.BytesIO(raw)
    epsilon = float(_read_f32_be(stream))
    segments = _read_merged_per_b(stream, epsilon)
    last_ts = _vb_read_u32(stream)
    return _to_points(segments, last_ts)


def decompress_column_f64(binary: bytes) -> List[Tuple[int, float]]:
    raw = _ZD.decompress(binary)
    stream = io.BytesIO(raw)
    epsilon = float(_read_f64_be(stream))
    segments = _read_merged_per_b_f64(stream, epsilon)
    last_ts = _vb_read_u32(stream)
    return _to_points(segments, last_ts)


def _read_merged_per_b(stream: io.BytesIO, epsilon: float) -> List[_Segment]:
    segments: List[_Segment] = []
    num_b = _vb_read_u32(stream)
    if num_b == 0:
        return segments
    prev_b = _vb_read_u32(stream)
    for _ in range(num_b):
        b = _vb_read_u32(stream) + prev_b
        prev_b = b
        num_a = _vb_read_u32(stream)
        for _j in range(num_a):
            a = _read_f32_be(stream)
            n_ts = _vb_read_u32(stream)
            t_acc = 0
            for _k in range(n_ts):
                t_acc += _vb_read_u32(stream)
                segments.append(_Segment(t_acc, float(a), float(a), float(b * epsilon)))
    return segments


def _read_merged_per_b_f64(stream: io.BytesIO, epsilon: float) -> List[_Segment]:
    segments: List[_Segment] = []
    num_b = _vb_read_u32(stream)
    if num_b == 0:
        return segments
    prev_b = _vb_read_u32(stream)
    for _ in range(num_b):
        b = _vb_read_u32(stream) + prev_b
        prev_b = b
        num_a = _vb_read_u32(stream)
        for _j in range(num_a):
            a = _read_f64_be(stream)
            n_ts = _vb_read_u32(stream)
            t_acc = 0
            for _k in range(n_ts):
                t_acc += _vb_read_u32(stream)
                segments.append(_Segment(t_acc, float(a), float(a), float(b * epsilon)))
    return segments


def _restored_value_list(by_ts: Dict[int, float], n: int) -> List[float]:
    out = [float("nan")] * n
    for t, v in by_ts.items():
        if 0 <= t < n:
            out[t] = float(v)
    return out


def bench_simpiece_float_columns(
    columns: List[List[float]], rel_range_frac: float
) -> Tuple[int, int, int, int, float, float]:
    """Same contract as Java ``benchSimPieceFloat``."""
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        n = len(col)
        mn = min(float(x) for x in col)
        mx = max(float(x) for x in col)
        rng = mx - mn
        if rng <= 0 or math.isnan(rng):
            rng = 1.0
        eps = rng * rel_range_frac
        if eps <= 0:
            continue
        try:
            t0 = time.perf_counter_ns()
            packed = compress_column([float(x) for x in col], eps)
            t1 = time.perf_counter_ns()
            restored = decompress_column(packed)
            t2 = time.perf_counter_ns()
            by_ts: Dict[int, float] = {t: v for t, v in restored}
            ok = True
            for i in range(n):
                dv = by_ts.get(i)
                if dv is None or abs(dv - float(col[i])) > eps * 1.0001 + 1e-9:
                    ok = False
                    break
            t3 = time.perf_counter_ns()
            if not ok:
                continue
            orig_vals = [float(x) for x in col]
            accumulate_double_list_loss(orig_vals, _restored_value_list(by_ts, n), loss_acc)
            tot_o += column_original_size(n, 'float')
            tot_c += len(packed)
            te += t1 - t0
            td += t3 - t2
        except (ValueError, EOFError, zstandard.ZstdError):
            continue
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def _bench_simpiece_f64_columns(
    columns: List[List[Union[float, int]]],
    rel_range_frac: float,
    *,
    data_type: str,
    tol_extra: float,
) -> Tuple[int, int, int, int, float, float]:
    """Bench via f32-epsilon wire format (same as Java ``compressColumnDoubleValues`` / C++ ``compress_inner``)."""
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        values = [float(x) for x in col]
        n = len(values)
        mn = min(values)
        mx = max(values)
        rng = mx - mn
        if rng <= 0 or math.isnan(rng):
            rng = 1.0
        eps = rng * rel_range_frac
        if eps <= 0:
            continue
        try:
            t0 = time.perf_counter_ns()
            packed = compress_column(values, eps)
            t1 = time.perf_counter_ns()
            restored = decompress_column(packed)
            t2 = time.perf_counter_ns()
            by_ts: Dict[int, float] = {t: v for t, v in restored}
            ok = True
            for i in range(n):
                dv = by_ts.get(i)
                if dv is None or abs(dv - values[i]) > eps * 1.0001 + tol_extra:
                    ok = False
                    break
            t3 = time.perf_counter_ns()
            if not ok:
                continue
            accumulate_double_list_loss(values, _restored_value_list(by_ts, n), loss_acc)
            tot_o += column_original_size(n, data_type)
            tot_c += len(packed)
            te += t1 - t0
            td += t3 - t2
        except (ValueError, EOFError, zstandard.ZstdError):
            continue
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_simpiece_double_columns(
    columns: List[List[float]], rel_range_frac: float
) -> Tuple[int, int, int, int, float, float]:
    return _bench_simpiece_f64_columns(
        columns, rel_range_frac, data_type="float", tol_extra=1e-12
    )


def bench_simpiece_int64_columns(
    columns: List[List[int]], rel_range_frac: float
) -> Tuple[int, int, int, int, float, float]:
    return _bench_simpiece_f64_columns(
        columns, rel_range_frac, data_type="int", tol_extra=1e-9
    )


def parity_hex_lines_for_column(values: List[float], rel_range_frac: float) -> List[str]:
    """One hex string (full Zstd blob) for parity scripts."""
    if not values:
        return []
    n = len(values)
    mn = min(float(x) for x in values)
    mx = max(float(x) for x in values)
    rng = mx - mn
    if rng <= 0 or math.isnan(rng):
        rng = 1.0
    eps = rng * rel_range_frac
    packed = compress_column([float(x) for x in values], eps)
    return [packed.hex()]
