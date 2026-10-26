"""Shared input policy for numeric-column compression experiments.

Drop NaN values before computing bytes/counts/hashes. This does not modify the
source file, remove infinities, or cast integers through floating point.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import struct
from dataclasses import dataclass
from typing import Iterable

NAN_INPUT_POLICY = "drop_nan_per_numeric_column_v1"
_MISSING_CSV_TOKENS = frozenset({"", "nan", "+nan", "-nan", "na", "n/a", "null", "none", "<na>"})


def is_nan_numeric(value: object) -> bool:
    """Recognize NaN/missing numeric cells without losing integer precision."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_CSV_TOKENS
    if isinstance(value, numbers.Integral):
        return False
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        # pandas.NA is a missing scalar but deliberately rejects bool/float.
        return str(value) == "<NA>"


def drop_nan_numeric(values: Iterable) -> tuple[tuple, int]:
    """Return retained values unchanged and the number of discarded NaN cells."""
    retained = []
    removed = 0
    for value in values:
        if is_nan_numeric(value):
            removed += 1
        else:
            retained.append(value)
    return tuple(retained), removed


@dataclass(frozen=True)
class PreparedNumericColumn:
    values: tuple[int | float, ...]
    logical_type: str
    source_count: int
    removed_nan_count: int
    raw: bytes

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    def manifest(self) -> dict:
        return {
            "input_policy": NAN_INPUT_POLICY,
            "source_rows": self.source_count,
            "rows": self.count,
            "removed_nan_count": self.removed_nan_count,
            "raw_bytes": len(self.raw),
            "raw_sha256": self.raw_sha256,
        }


def prepare_numeric_column(values: Iterable, logical_type: str) -> PreparedNumericColumn:
    """Normalize integer/float input after per-element NaN removal.

    Accepts numeric values or CSV cells. Integer spellings are parsed directly
    with int(), preserving all int64 bits; invalid/noninteger cells raise.
    Empty/all-NaN inputs produce a valid empty result for callers to skip.
    """
    aliases = {"int": "integer", "integer": "integer", "int64": "integer", "long": "integer",
               "float": "float", "float64": "float", "double": "float"}
    try:
        kind = aliases[logical_type.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported numeric experiment type: {logical_type!r}") from exc
    retained, removed = drop_nan_numeric(values)
    if kind == "integer":
        converted_list = []
        for value in retained:
            integer = int(value)
            if not isinstance(value, (str, numbers.Integral)) and value != integer:
                raise ValueError(f"noninteger experiment value: {value!r}")
            converted_list.append(integer)
        converted = tuple(converted_list)
    else:
        converted = tuple(float(value) for value in retained)
    raw = struct.pack(f"<{len(converted)}{'q' if kind == 'integer' else 'd'}", *converted)
    return PreparedNumericColumn(converted, kind, len(retained) + removed, removed, raw)


def parse_experiment_float_cell(cell: object) -> float | None:
    """Streaming CSV parser: invalid/NaN cells skip; +/-Inf and -0.0 remain."""
    if is_nan_numeric(cell):
        return None
    try:
        value = float(cell)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value
