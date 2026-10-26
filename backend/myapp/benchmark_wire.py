"""Little-endian column wire format for tri-language benchmark parity (int64 / float64)."""
from __future__ import annotations

import struct

BENCHMARK_INT_BYTES = 8
BENCHMARK_FLOAT_BYTES = 8


def pack_int64_le(values) -> bytes:
    return struct.pack("<" + "q" * len(values), *values)


def pack_float64_le(values) -> bytes:
    return struct.pack("<" + "d" * len(values), *values)


def column_value_bytes(data_type: str) -> int:
    return BENCHMARK_FLOAT_BYTES if data_type == "float" else BENCHMARK_INT_BYTES


def column_original_size(num_values: int, data_type: str) -> int:
    return num_values * column_value_bytes(data_type)
