"""Optional wrappers for code aligned with external/compression (zfp, bitshuffle)."""
from __future__ import annotations

import os
import time
from typing import List, Tuple

from .benchmark_loss import accumulate_double_list_loss
from .benchmark_wire import column_original_size


def _zfp_bench_enabled() -> bool:
    flag = os.environ.get("COMPRESSION_BENCHMARK_ZFP_BENCH")
    return flag is None or flag.strip() == "" or flag.strip() == "1"


def bench_zfp_zfpy_columns(
    columns: List[List[float]], *, wire_data_type: str = "float"
) -> Tuple[int, int, int, int, float, float]:
    """Lossless ZFP (float64 wire) via pip ``zfpy``."""
    if not _zfp_bench_enabled():
        return 0, 0, 0, 0, 0.0, 0.0
    return _bench_zfp_zfpy_impl(columns, wire_data_type=wire_data_type)


def bench_zfp_lossy_columns(
    columns: List[List[float]], tolerance: float, *, wire_data_type: str = "float"
) -> Tuple[int, int, int, int, float, float]:
    """ZFP fixed-accuracy mode (lossy, float64 wire) via pip ``zfpy``."""
    if not _zfp_bench_enabled() or tolerance <= 0:
        return 0, 0, 0, 0, 0.0, 0.0

    try:
        import numpy as np
        import zfpy
    except ImportError:
        return 0, 0, 0, 0, 0.0, 0.0

    orig_sz = comp_sz = 0
    comp_ns = decomp_ns = 0
    loss_acc = [0.0, 0.0]
    verify_atol = max(tolerance * 16.0, 1e-6)

    for col in columns:
        if not col:
            continue
        a = np.ascontiguousarray(col, dtype=np.float64)
        o = column_original_size(len(a), wire_data_type)
        try:
            t0 = time.perf_counter_ns()
            c = zfpy.compress_numpy(a, tolerance=tolerance)
            t1 = time.perf_counter_ns()
            a2 = zfpy.decompress_numpy(c)
            t2 = time.perf_counter_ns()
        except Exception:
            continue
        if a2.shape != a.shape or a2.dtype != np.float64:
            continue
        if not np.allclose(a, a2, rtol=0.0, atol=verify_atol, equal_nan=True):
            continue
        accumulate_double_list_loss(
            [float(x) for x in a.reshape(-1)],
            [float(x) for x in a2.reshape(-1)],
            loss_acc,
        )
        if isinstance(c, (bytes, bytearray)):
            clen = len(c)
        else:
            clen = int(c.nbytes)
        orig_sz += o
        comp_sz += clen
        comp_ns += t1 - t0
        decomp_ns += t2 - t1

    return orig_sz, comp_sz, comp_ns, decomp_ns, loss_acc[0], loss_acc[1]


def _bench_zfp_zfpy_impl(
    columns: List[List[float]], *, wire_data_type: str
) -> Tuple[int, int, int, int, float, float]:
    try:
        import numpy as np
        import zfpy
    except ImportError:
        return 0, 0, 0, 0, 0.0, 0.0

    orig_sz = comp_sz = 0
    comp_ns = decomp_ns = 0
    loss_acc = [0.0, 0.0]

    for col in columns:
        if not col:
            continue
        a = np.ascontiguousarray(col, dtype=np.float64)
        o = column_original_size(len(a), wire_data_type)
        try:
            t0 = time.perf_counter_ns()
            c = zfpy.compress_numpy(a)
            t1 = time.perf_counter_ns()
            a2 = zfpy.decompress_numpy(c)
            t2 = time.perf_counter_ns()
        except Exception:
            continue
        if a2.shape != a.shape or a2.dtype != np.float64:
            continue
        if not np.array_equal(a, a2):
            continue
        accumulate_double_list_loss(
            [float(x) for x in a.reshape(-1)],
            [float(x) for x in a2.reshape(-1)],
            loss_acc,
        )
        if isinstance(c, (bytes, bytearray)):
            clen = len(c)
        else:
            clen = int(c.nbytes)
        orig_sz += o
        comp_sz += clen
        comp_ns += t1 - t0
        decomp_ns += t2 - t1

    return orig_sz, comp_sz, comp_ns, decomp_ns, loss_acc[0], loss_acc[1]


def bench_bitshuffle_lz4_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int, float, float]:
    try:
        import numpy as np
        from bitshuffle import compress_lz4, decompress_lz4
    except ImportError:
        return 0, 0, 0, 0, 0.0, 0.0

    orig_sz = comp_sz = 0
    comp_ns = decomp_ns = 0
    loss_acc = [0.0, 0.0]

    for col in columns:
        if not col:
            continue
        a = np.ascontiguousarray(col, dtype=np.float32)
        o = int(a.nbytes)
        try:
            t0 = time.perf_counter_ns()
            c = compress_lz4(a)
            t1 = time.perf_counter_ns()
            a2 = decompress_lz4(c, a.shape, a.dtype)
            t2 = time.perf_counter_ns()
        except Exception:
            continue
        if a2.shape != a.shape or not np.array_equal(a, a2):
            continue
        accumulate_double_list_loss(
            [float(x) for x in a.reshape(-1)],
            [float(x) for x in a2.reshape(-1)],
            loss_acc,
        )
        clen = int(c.nbytes)
        orig_sz += o
        comp_sz += clen
        comp_ns += t1 - t0
        decomp_ns += t2 - t1

    return orig_sz, comp_sz, comp_ns, decomp_ns, loss_acc[0], loss_acc[1]
