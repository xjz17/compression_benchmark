"""Column list helpers for cross-type benchmark rows (int columns as float values)."""
from __future__ import annotations

from typing import List


def integers_as_float_columns(column_int_data: List[List[int]]) -> List[List[float]]:
    """Treat each int as float with zero fractional part, not intBitsToFloat."""
    return [[float(v) for v in col] for col in column_int_data]


def integers_as_double_columns(column_int_data: List[List[int]]) -> List[List[float]]:
    """Same as {@link integers_as_float_columns}; values are float64 on the wire."""
    return integers_as_float_columns(column_int_data)
