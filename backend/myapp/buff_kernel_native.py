"""
Optional native BUFF pipeline helpers: NumPy-accelerated column stats (SIMD via BLAS on x86/ARM)
and Roaring-lite index set, aligned with ``cpp/src/buff_kernel`` semantics.

Enable in benchmarks with ``COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE=1`` (C++ ``benchmark_cpp``).
"""

from __future__ import annotations

import math
import struct
from typing import List, Sequence, Tuple


def filter_nan_values(values: Sequence[float | int]) -> list[float | int]:
    """Experiment preprocessing only; preserve non-NaN values without float casts."""
    return [value for value in values if not math.isnan(value)]


def pack_raw64_values(values: Sequence[float | int], kind: str, *, drop_nan: bool = True) -> bytes:
    """Pack integer values as int64, never through a precision-losing float view."""
    if kind not in {"int64", "float64"}:
        raise ValueError("BUFF native requires int64 or float64")
    selected = filter_nan_values(values) if drop_nan else values
    return struct.pack(f"<{len(selected)}{'q' if kind == 'int64' else 'd'}", *selected)


def encode_raw64(raw: bytes) -> bytes:
    """Byte-identical to current C++ buff_kernel::encode_doubles raw64 wire.

    Numeric interpretation is external: int64 lanes are bitcast, not converted.
    This exact storage format has 12 bytes overhead for every nonempty column.
    """
    if len(raw) % 8 or len(raw) // 8 > 0xffffffff:
        raise ValueError("BUFF native requires uint32-counted complete 64-bit lanes")
    if not raw:
        return b""
    out = bytearray(struct.pack("<III", len(raw) // 8, 64, 0))
    for index in range(0, len(raw), 8):
        out.extend(raw[index + 4:index + 8])
        out.extend(raw[index:index + 4])
    return bytes(out)


def decode_raw64(wire: bytes) -> bytes:
    """Recover original 64-bit lanes; reject legacy bounded/component formats."""
    if not wire:
        return b""
    if len(wire) < 12:
        raise ValueError("BUFF native truncated raw64 header")
    count, integer_bits, fractional_bits = struct.unpack_from("<III", wire)
    if count == 0 or integer_bits != 64 or fractional_bits != 0 or len(wire) != 12 + count * 8:
        raise ValueError("BUFF native raw64 header/length mismatch")
    out = bytearray()
    for index in range(12, len(wire), 8):
        out.extend(wire[index + 4:index + 8])
        out.extend(wire[index:index + 4])
    return bytes(out)


def column_stats_numpy(values: Sequence[float]) -> Tuple[float, float, float]:
    """Sum, min, max using NumPy (vectorized)."""
    import numpy as np

    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    return float(a.sum()), float(a.min()), float(a.max())


def count_gt_numpy(values: Sequence[float], threshold: float) -> int:
    import numpy as np

    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 0
    return int(np.sum(a > threshold))


class BuffRoaringLite:
    """Minimal sorted unique id set (same role as ``buff_kernel::BuffRoaringLite``)."""

    def __init__(self) -> None:
        self._ids: set[int] = set()

    def add(self, x: int) -> None:
        self._ids.add(int(x))

    def cardinality(self) -> int:
        return len(self._ids)
