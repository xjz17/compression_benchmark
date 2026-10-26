"""
Per-column logical typing and manifest sidecar for typed TsFile Java export.

Used by ``processor`` (upload pipeline) and by ``scripts/regenerate_dataset_info.py``
(which renames ``{key}_*`` sidecars to catalog names under ``backend/dataset/<key>/``).
No Django dependency.
"""
from __future__ import annotations

import csv
import os
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd



def _infer_column_logical_type(series: pd.Series) -> str:
    """
    Classify a column as integer, float, or string for downstream TsFile export.

    Uses pandas dtypes first, then best-effort numeric coercion for object columns.
    """
    if series is None:
        return "string"
    s = series
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_bool_dtype(s):
        return "string"
    if getattr(s.dtype, "name", "") == "string":
        return "string"

    non_null = s.dropna()
    if non_null.empty:
        return "string"

    num = pd.to_numeric(non_null, errors="coerce")
    if num.isna().any():
        return "string"

    as_float = num.astype("float64")
    frac = (as_float % 1).abs()
    if (frac > 1e-9).any():
        return "float"
    try:
        as_int = num.astype("Int64")
        if as_int.isna().any():
            return "float"
        return "integer"
    except (TypeError, ValueError):
        return "float"


def _fractional_decimal_places_from_cell(s: str) -> int:
    """
    For one CSV cell string, return how many decimal digits appear after the decimal point
    (trailing fractional zeros stripped). Scientific notation uses Decimal's exponent.
    Non-numeric / empty -> 0.
    """
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return 0
    if s[0] in "+-":
        s = s[1:].strip()
    if not s:
        return 0
    if "e" in s.lower():
        try:
            d = Decimal(s)
        except InvalidOperation:
            return 0
        exp = d.as_tuple().exponent
        return max(0, -exp) if exp < 0 else 0
    if "." not in s:
        return 0
    _, frac = s.split(".", 1)
    frac_digits = "".join(ch for ch in frac if ch.isdigit())
    if not frac_digits:
        return 0
    stripped = frac_digits.rstrip("0")
    return len(stripped)


def write_csv_column_type_manifest(csv_file_path: str) -> str:
    """
    Read a CSV with pandas, infer per-column logical type (integer / float / string),
    and write a sidecar manifest next to the input CSV using LF newlines.

    The first line is a comment: ``# max_point_number=<n>`` where ``n`` is the maximum
    number of fractional decimal digits observed in any cell of columns inferred as
    ``float`` (from a parallel string read of the CSV, so formatting matches the file).
    If there are no float columns, ``n`` is 0. Java typed TsFile export reads this for
    ``Encoder.MAX_POINT_NUMBER``.

    For every column classified as ``integer`` or ``float``, also writes numeric
    statistics and charts next to the input CSV (same naming style as ``views`` /
    ``statistic.py``): ``{base}_numeric_statistic.csv``,
    ``{base}_numeric_scaling.png``, ``{base}_numeric_xor_leading.png``,
    ``{base}_numeric_xor_trailing.png``.

    Manifest columns: column_index,column_name,logical_type
    Returns the manifest file path.
    """
    csv_path = os.path.abspath(csv_file_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df_raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    max_point_number = 0
    for col in df.columns:
        if _infer_column_logical_type(df[col]) != "float":
            continue
        if col not in df_raw.columns:
            continue
        for cell in df_raw[col].values:
            max_point_number = max(max_point_number, _fractional_decimal_places_from_cell(cell))

    parent = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    manifest_path = os.path.join(parent, f"{base}_column_types.csv")

    rows_out = []
    for idx, col in enumerate(df.columns):
        logical = _infer_column_logical_type(df[col])
        rows_out.append({"column_index": idx, "column_name": str(col), "logical_type": logical})

    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# max_point_number={max_point_number}\n")
        w = csv.DictWriter(
            f,
            fieldnames=["column_index", "column_name", "logical_type"],
            lineterminator="\n",
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    numeric_cols = [col for col in df.columns if _infer_column_logical_type(df[col]) in ("integer", "float")]
    if numeric_cols:
        numeric_df = df[numeric_cols].copy()
        numeric_df = numeric_df.dropna(axis=1, how="all")
        if numeric_df.shape[1] > 0:
            stat_csv = os.path.join(parent, f"{base}_numeric_statistic.csv")
            scaling_png = os.path.join(parent, f"{base}_numeric_scaling.png")
            xor_leading_png = os.path.join(parent, f"{base}_numeric_xor_leading.png")
            xor_trailing_png = os.path.join(parent, f"{base}_numeric_xor_trailing.png")
            from .statistic import (
                statistic_bit_width_distribution,
                statistic_df,
                statistic_xor_distribution,
            )

            statistic_bit_width_distribution(numeric_df, scaling_png, 'float')
            statistic_xor_distribution(numeric_df, xor_leading_png, xor_trailing_png, 'float')
            statistic_df(numeric_df, stat_csv)

    return manifest_path


def _source_cells_for_max_points(
    df_source: pd.DataFrame,
    column_name: str,
    column_index: int,
) -> pd.Series:
    """Resolve a column in the source CSV (by name, else by index)."""
    if column_name in df_source.columns:
        return df_source[column_name]
    if column_index < len(df_source.columns):
        return df_source.iloc[:, column_index]
    raise KeyError(f'Column {column_name!r} not found in source CSV')


def write_column_max_points_sidecar(
    csv_file_path: str,
    data_type: str,
    *,
    source_csv_path: str | None = None,
) -> str:
    """
    Write per-column max fractional decimal places for TsFile FLOAT encoder props.

    ``csv_file_path`` is the processed CSV passed to the Java/C++ bench (column set and
    order). Fractional digits are read from ``source_csv_path`` when given (typically the
    original upload / ``dataset/<key>/<key>.csv``). Arrow-derived processed CSV may use
    ``float_format='%.8g'``; CSV uploads keep full text precision in the processed file.
    Otherwise cells come from ``csv_file_path``.

    Sidecar path: ``{base}_column_max_points.csv`` with columns
    ``column_index,column_name,max_point_number``.
    """
    csv_path = os.path.abspath(csv_file_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    df_proc = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if source_csv_path:
        src_path = os.path.abspath(source_csv_path)
        if not os.path.isfile(src_path):
            raise FileNotFoundError(src_path)
        df_source = pd.read_csv(src_path, dtype=str, keep_default_na=False)
    else:
        df_source = df_proc

    parent = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(parent, f"{base}_column_max_points.csv")

    rows_out = []
    for idx, col in enumerate(df_proc.columns):
        max_point = 0
        if data_type == "float":
            src_col = _source_cells_for_max_points(df_source, str(col), idx)
            for cell in src_col.values:
                max_point = max(max_point, _fractional_decimal_places_from_cell(cell))
        rows_out.append(
            {
                "column_index": idx,
                "column_name": str(col),
                "max_point_number": max_point,
            }
        )

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["column_index", "column_name", "max_point_number"],
            lineterminator="\n",
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    return out_path
