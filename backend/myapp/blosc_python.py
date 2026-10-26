"""Blosc1 (c-blosc) blosclz + shuffle column benchmarks (pip ``blosc`` / python-blosc)."""
from __future__ import annotations

import os
import time
from typing import List, Tuple

from .benchmark_loss import accumulate_double_list_loss


def _bench_enabled() -> bool:
    flag = os.environ.get("WEB_COMPRESSION_BLOSC_BENCH")
    return flag is None or flag.strip() == "" or flag.strip() == "1"


def _bench_blosc_columns(
    columns: List[List[float]],
    *,
    wire_dtype,
    typesize: int,
) -> Tuple[int, int, int, int, float, float]:
    if not _bench_enabled():
        return 0, 0, 0, 0, 0.0, 0.0
    try:
        import blosc
        import numpy as np
    except ImportError:
        return 0, 0, 0, 0, 0.0, 0.0

    orig_sz = comp_sz = 0
    comp_ns = decomp_ns = 0
    loss_acc = [0.0, 0.0]

    for col in columns:
        if not col:
            continue
        arr = np.ascontiguousarray(col, dtype=wire_dtype)
        o = int(arr.nbytes)
        try:
            t0 = time.perf_counter_ns()
            c = blosc.compress(
                arr.tobytes(),
                typesize=typesize,
                clevel=5,
                shuffle=blosc.SHUFFLE,
                cname="blosclz",
            )
            t1 = time.perf_counter_ns()
            raw = blosc.decompress(c)
            t2 = time.perf_counter_ns()
        except Exception:
            continue
        arr2 = np.frombuffer(raw, dtype=wire_dtype)
        if arr2.shape != arr.shape or not np.array_equal(arr, arr2):
            continue
        accumulate_double_list_loss(
            [float(x) for x in arr.reshape(-1)],
            [float(x) for x in arr2.reshape(-1)],
            loss_acc,
        )
        clen = len(c) if isinstance(c, (bytes, bytearray)) else int(c.nbytes)
        orig_sz += o
        comp_sz += clen
        comp_ns += t1 - t0
        decomp_ns += t2 - t1

    return orig_sz, comp_sz, comp_ns, decomp_ns, loss_acc[0], loss_acc[1]


def bench_blosc_blosclz_float_columns(
    columns: List[List[float]],
) -> Tuple[int, int, int, int, float, float]:
    """Float32 wire (typesize=4), aligned with C++ ``blosc_codec``."""
    import numpy as np

    return _bench_blosc_columns(columns, wire_dtype=np.float32, typesize=4)


def bench_blosc_blosclz_int64_columns(
    columns: List[List[int]],
) -> Tuple[int, int, int, int, float, float]:
    """Int64 wire (typesize=8)."""
    import numpy as np

    return _bench_blosc_columns(columns, wire_dtype=np.int64, typesize=8)


def bench_blosc_blosclz_byte_columns(
    column_byte_data: List[bytes],
) -> Tuple[int, int, int, int, float, float]:
    """UTF-8 concatenated string column bytes (typesize=1)."""
    if not _bench_enabled():
        return 0, 0, 0, 0, 0.0, 0.0
    try:
        import blosc
    except ImportError:
        return 0, 0, 0, 0, 0.0, 0.0

    orig_sz = comp_sz = 0
    comp_ns = decomp_ns = 0
    for raw in column_byte_data:
        if not raw:
            continue
        o = len(raw)
        try:
            t0 = time.perf_counter_ns()
            c = blosc.compress(
                raw,
                typesize=1,
                clevel=5,
                shuffle=blosc.SHUFFLE,
                cname="blosclz",
            )
            t1 = time.perf_counter_ns()
            back = blosc.decompress(c)
            t2 = time.perf_counter_ns()
        except Exception:
            continue
        if back != raw:
            continue
        clen = len(c) if isinstance(c, (bytes, bytearray)) else int(c.nbytes)
        orig_sz += o
        comp_sz += clen
        comp_ns += t1 - t0
        decomp_ns += t2 - t1
    return orig_sz, comp_sz, comp_ns, decomp_ns, 0.0, 0.0
