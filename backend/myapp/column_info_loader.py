"""Load catalog ``{key}_column_info.csv`` for patch benchmarks."""

from __future__ import annotations

import io
import csv
import os
import re
from typing import Dict, List, Sequence, Tuple


def load_column_info_rows(column_info_path: str) -> List[dict]:
    path = os.path.abspath(column_info_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def max_point_by_column_name(column_info_path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in load_column_info_rows(column_info_path):
        name = str(row.get("column_name", "")).strip()
        if not name:
            continue
        try:
            out[name] = max(0, int(row.get("max_point_number", 0)))
        except (TypeError, ValueError):
            out[name] = 0
    return out


def logical_type_by_column_name(column_info_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in load_column_info_rows(column_info_path):
        name = str(row.get("column_name", "")).strip()
        if not name:
            continue
        out[name] = str(row.get("logical_type", "")).strip().lower()
    return out


def _headers_for_logical_type(
    column_info_path: str, logical_want: str
) -> List[Tuple[str, int]]:
    headers: List[Tuple[str, int]] = []
    for row in load_column_info_rows(column_info_path):
        logical = str(row.get("logical_type", "")).strip().lower()
        if logical != logical_want:
            continue
        name = str(row.get("column_name", "")).strip()
        if not name:
            continue
        try:
            mp = max(0, int(row.get("max_point_number", 0)))
        except (TypeError, ValueError):
            mp = 0
        headers.append((name, mp))
    return headers


def _logical_type_sort_key(t: str) -> tuple[int, str]:
    low = t.lower()
    rank = {
        "integer": 0,
        "int": 0,
        "long": 0,
        "float": 1,
        "double": 1,
        "timestamp": 2,
        "datetime": 2,
        "date": 2,
        "time": 2,
        "boolean": 2,
        "bool": 2,
        "string": 3,
    }.get(low, 50)
    return (rank, low)


_TIMESTAMP_LOGICAL_TYPES = frozenset({"timestamp", "datetime", "date", "time"})
_TIMESTAMP_NAME_TOKENS = frozenset(
    {"timestamp", "datetime", "date", "time", "published"}
)
_COLUMN_NAME_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_SECONDS_TIMESTAMP_RANGE = (946_684_800, 4_102_444_800)
_MILLIS_TIMESTAMP_RANGE = (946_684_800_000, 4_102_444_800_000)


def column_info_row_is_timestamp(row: dict) -> bool:
    """True for catalog display columns that represent timestamps or dates."""
    logical = _normalize_logical_type(str(row.get("logical_type", "")))
    if logical in _TIMESTAMP_LOGICAL_TYPES:
        return True
    name = str(row.get("column_name", "")).strip().lower()
    if not name:
        return False
    if name.isdigit():
        value = int(name)
        if (
            _SECONDS_TIMESTAMP_RANGE[0] <= value <= _SECONDS_TIMESTAMP_RANGE[1]
            or _MILLIS_TIMESTAMP_RANGE[0] <= value <= _MILLIS_TIMESTAMP_RANGE[1]
        ):
            return True
    compact = re.sub(r"[^a-z0-9]+", "", name)
    if "timestamp" in compact or "datetime" in compact:
        return True
    tokens = {tok for tok in _COLUMN_NAME_TOKEN_RE.split(name) if tok}
    return bool(tokens & _TIMESTAMP_NAME_TOKENS)


def catalog_dtype_from_column_info_row(row: dict) -> str | None:
    """Catalog-page dtype: int / float / timestamp."""
    if column_info_row_is_timestamp(row):
        return "timestamp"
    logical = _normalize_logical_type(str(row.get("logical_type", "")))
    if logical in ("integer", "int", "long"):
        return "int"
    if logical in ("float", "double"):
        return "float"
    # String/byte columns are currently out of scope for benchmark and catalog dtype charts.
    return None


def distinct_logical_types_from_column_info(column_info_path: str) -> List[str]:
    """
    Distinct ``logical_type`` values from ``column_info`` / legacy column_types CSV.

    Returns values as stored in the file (e.g. ``integer``, ``float``); no int/float
    benchmark mapping or integer→float fallback.
    """
    path = os.path.abspath(column_info_path)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            body_lines = [ln for ln in f if not ln.lstrip().startswith("#")]
        if not body_lines:
            return []
        buf = io.StringIO("".join(body_lines))
        reader = csv.DictReader(buf)
        if not reader.fieldnames:
            return []
        lt_field = None
        for fn in reader.fieldnames:
            if fn and str(fn).strip().lower() == "logical_type":
                lt_field = fn
                break
        if not lt_field:
            return []
        seen: set[str] = set()
        for row in reader:
            raw = (row.get(lt_field) or "").strip()
            if raw:
                seen.add(raw)
        if not seen:
            return []
        return sorted(seen, key=_logical_type_sort_key)
    except Exception:
        return []


def benchmark_dtype_logical_aliases(data_type: str) -> frozenset[str]:
    """
    ``logical_type`` values that qualify for a benchmark dtype (regen / upload gating).

    Matches ``scripts/regenerate_staticdata_*.py`` column selection (float is float/double
    only, not integer wire columns).
    """
    if data_type == "int":
        return frozenset({"integer", "int", "long"})
    if data_type == "float":
        return frozenset({"float", "double"})
    raise ValueError(data_type)


def column_info_has_benchmark_dtype(column_info_path: str, data_type: str) -> bool:
    """True when ``column_info`` lists at least one column for this benchmark dtype."""
    if not os.path.isfile(column_info_path):
        return False
    wanted = benchmark_dtype_logical_aliases(data_type)
    for row in load_column_info_rows(column_info_path):
        logical = _normalize_logical_type(str(row.get("logical_type", "")))
        if logical in wanted:
            return True
    return False


def benchmark_dtypes_from_column_info(column_info_path: str) -> List[str]:
    """Return ``int`` / ``float`` dtypes that have benchmark columns in column_info."""
    if not os.path.isfile(column_info_path):
        return []
    out: List[str] = []
    for dtype in ("int", "float"):
        if column_info_has_benchmark_dtype(column_info_path, dtype):
            out.append(dtype)
    return out


def catalog_dtypes_from_column_info(column_info_path: str) -> List[str]:
    """Return catalog display dtypes, splitting timestamp/date columns from int."""
    if not os.path.isfile(column_info_path):
        return []
    order = ("int", "float", "timestamp")
    seen: set[str] = set()
    for row in load_column_info_rows(column_info_path):
        dtype = catalog_dtype_from_column_info_row(row)
        if dtype:
            seen.add(dtype)
    return [dtype for dtype in order if dtype in seen]


def _normalize_logical_type(raw: str) -> str:
    return str(raw).strip().lower()


def _is_integer_logical(logical: str) -> bool:
    return _normalize_logical_type(logical) in ("integer", "int", "long")


def _is_float_wire_logical(logical: str) -> bool:
    return _normalize_logical_type(logical) in ("integer", "int", "long", "float", "double")


def bench_headers_for_data_type(
    column_info_path: str, data_type: str
) -> List[Tuple[str, int]]:
    """
    Return (column_name, max_point_number) for columns to benchmark, in catalog order.

    ``int`` -> ``logical_type`` integer (and int/long aliases).
    ``float`` -> integer **and** float columns (integer series encoded as float64 wire).
    """
    if data_type == "int":
        headers: List[Tuple[str, int]] = []
        for row in load_column_info_rows(column_info_path):
            if not _is_integer_logical(str(row.get("logical_type", ""))):
                continue
            name = str(row.get("column_name", "")).strip()
            if not name:
                continue
            try:
                mp = max(0, int(row.get("max_point_number", 0)))
            except (TypeError, ValueError):
                mp = 0
            headers.append((name, mp))
        return headers

    if data_type == "float":
        headers = []
        seen: set[str] = set()
        for row in load_column_info_rows(column_info_path):
            logical = str(row.get("logical_type", ""))
            if not _is_float_wire_logical(logical):
                continue
            name = str(row.get("column_name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                mp = max(0, int(row.get("max_point_number", 0)))
            except (TypeError, ValueError):
                mp = 0
            headers.append((name, mp))
        return headers

    raise ValueError(data_type)


def _normalize_header_name(name: str) -> str:
    """Strip whitespace and UTF-8 BOM so column_info names match master CSV headers."""
    return str(name).strip().removeprefix("\ufeff")


def order_headers_by_master_csv(
    master_header: Sequence[str], bench_headers: Sequence[Tuple[str, int]]
) -> List[Tuple[str, int, int]]:
    """Map bench columns to master CSV column index; preserve master file column order."""
    index_by_name = {
        _normalize_header_name(h): i for i, h in enumerate(master_header)
    }
    ordered: List[Tuple[str, int, int]] = []
    for name, mp in bench_headers:
        idx = index_by_name.get(_normalize_header_name(name))
        if idx is not None:
            ordered.append((name, mp, idx))
    return ordered


def patch_bench_columns_for_data_type(
    master_header: Sequence[str],
    column_info_path: str,
    data_type: str,
) -> List[Tuple[str, int, int]]:
    """Map column_info bench columns onto master CSV indices (patch bench entry point)."""
    bench_meta = bench_headers_for_data_type(column_info_path, data_type)
    if not bench_meta:
        return []
    return order_headers_by_master_csv(master_header, bench_meta)
