"""Column benchmarks for TsFile encodings (matches ``cpp/src/tsfile_bench_tsfile.cpp``)."""
from __future__ import annotations

import struct
import time
import math
from decimal import Decimal, InvalidOperation
from typing import Callable, List, Sequence, Tuple

from ..benchmark_loss import accumulate_double_list_loss, accumulate_long_list_loss
from . import p02, p03, p04

EncodeInt = Callable[[List[int]], bytes]
DecodeInt = Callable[[bytes], List[int]]
EncodeFloat = Callable[[List[float], int], bytes]
DecodeFloat = Callable[[bytes], List[float]]


def _max_decimal_scale_in_column(col: List[float]) -> int:
    m = 0
    for v in col:
        if not isinstance(v, (float, int)):
            continue
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):
            continue
        try:
            bd = Decimal(str(fv)).normalize()
        except (InvalidOperation, ValueError):
            continue
        exp = bd.as_tuple().exponent
        if exp < 0:
            m = max(m, min(24, -exp))
    return m


def _clamp_ts2diff_max_point_for_double(col: List[float], requested_max_point: int) -> int:
    normalized = max(0, int(requested_max_point))
    if not col:
        return normalized

    max_abs = 0.0
    for v in col:
        if not isinstance(v, (float, int)):
            continue
        fv = float(v)
        if not math.isfinite(fv):
            continue
        max_abs = max(max_abs, abs(fv))
    if not (max_abs > 0.0):
        return normalized

    ratio = float((2**63 - 1) / max_abs)
    if not math.isfinite(ratio):
        return normalized
    if ratio <= 0.0:
        return 0

    safe_upper = max(0, int(math.floor(math.log10(ratio))))
    return min(normalized, safe_upper)


def _bench_int_columns(
    columns: List[List[int]],
    encode_fn: EncodeInt,
    decode_fn: DecodeInt,
    *,
    exact_roundtrip: bool,
) -> Tuple[int, int, int, int, float, float]:
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        vals = [int(x) for x in col]
        t0 = time.perf_counter_ns()
        enc = encode_fn(vals)
        t1 = time.perf_counter_ns()
        if not enc:
            continue
        t2 = time.perf_counter_ns()
        try:
            dec = decode_fn(enc)
        except Exception:
            continue
        t3 = time.perf_counter_ns()
        if not dec:
            continue
        if exact_roundtrip and (len(dec) != len(vals) or dec != vals):
            continue
        accumulate_long_list_loss(vals, dec, loss_acc)
        tot_o += len(vals) * 8
        tot_c += len(enc)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def _bench_float_columns(
    columns: List[List[float]],
    encode_fn: EncodeFloat,
    decode_fn: DecodeFloat,
    max_point_per_column: Sequence[int] | None = None,
) -> Tuple[int, int, int, int, float, float]:
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for ci, col in enumerate(columns):
        if not col:
            continue
        if max_point_per_column is not None and ci < len(max_point_per_column):
            mp = int(max_point_per_column[ci])
        else:
            mp = _max_decimal_scale_in_column(col)
        t0 = time.perf_counter_ns()
        try:
            enc = encode_fn(col, mp)
        except (struct.error, OverflowError, ValueError):
            continue
        t1 = time.perf_counter_ns()
        if not enc:
            continue
        t2 = time.perf_counter_ns()
        try:
            dec = decode_fn(enc)
        except Exception:
            continue
        t3 = time.perf_counter_ns()
        if not dec:
            continue
        accumulate_double_list_loss([float(x) for x in col], [float(x) for x in dec], loss_acc)
        tot_o += len(col) * 8
        tot_c += len(enc)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_ts_2diff_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    return _bench_int_columns(
        columns,
        lambda c: p02.encode_ts_2diff_int64(c),
        p02.decode_ts_2diff_int64,
        exact_roundtrip=True,
    )


def bench_rle_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    return _bench_int_columns(
        columns,
        p02.encode_rle_int64,
        p02.decode_rle_int64,
        exact_roundtrip=False,
    )


def bench_gorilla_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    from . import p03_i64

    return _bench_int_columns(
        columns,
        p03_i64.encode_gorilla_int64,
        p03_i64.decode_gorilla_int64,
        exact_roundtrip=False,
    )


def bench_chimp_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    from . import p03_i64

    return _bench_int_columns(
        columns,
        p03_i64.encode_chimp_int64,
        p03_i64.decode_chimp_int64,
        exact_roundtrip=False,
    )


def bench_sprintz_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    from . import p04_i64

    return _bench_int_columns(
        columns,
        p04_i64.encode_sprintz_int64,
        p04_i64.decode_sprintz_int64,
        exact_roundtrip=False,
    )


def bench_rlbe_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    from . import p04_i64

    return _bench_int_columns(
        columns,
        p04_i64.encode_rlbe_int64,
        p04_i64.decode_rlbe_int64,
        exact_roundtrip=False,
    )


def bench_ts_2diff_float_columns(
    columns: List[List[float]],
    max_point_per_column: Sequence[int] | None = None,
) -> Tuple[int, int, int, int]:
    return _bench_float_columns(
        columns,
        lambda c, mp: p02.encode_ts_2diff_double(c, _clamp_ts2diff_max_point_for_double(c, mp)),
        p02.decode_ts_2diff_double,
        max_point_per_column,
    )


def bench_rle_float_columns(
    columns: List[List[float]],
    max_point_per_column: Sequence[int] | None = None,
) -> Tuple[int, int, int, int]:
    return _bench_float_columns(
        columns,
        lambda c, mp: p02.encode_rle_double(c, mp),
        p02.decode_rle_double,
        max_point_per_column,
    )


def bench_gorilla_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    from . import p03_i64

    return _bench_float_columns(
        columns,
        lambda c, _mp: p03_i64.encode_gorilla_double(c),
        lambda enc: p03_i64.decode_gorilla_double(enc),
        None,
    )


def bench_chimp_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    from . import p03_i64

    return _bench_float_columns(
        columns,
        lambda c, _mp: p03_i64.encode_chimp_double(c),
        lambda enc: p03_i64.decode_chimp_double(enc),
        None,
    )


def bench_sprintz_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    from . import p04_i64

    return _bench_float_columns(
        columns,
        lambda c, _mp: p04_i64.encode_sprintz_double(c),
        lambda enc: p04_i64.decode_sprintz_double(enc),
        None,
    )


def bench_rlbe_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    from . import p04_i64

    return _bench_float_columns(
        columns,
        lambda c, _mp: p04_i64.encode_rlbe_double(c),
        lambda enc: p04_i64.decode_rlbe_double(enc),
        None,
    )
