"""
Build catalog-sidecar files under ``backend/dataset/<key>/`` (regenerate scripts + API).

Master CSVs live in ``backend/dataset_csv/``; this module writes metadata and charts only.
Naming: ``{key}_column_info.csv`` (includes per-column CSV size), ``{key}_dataset_info.csv``
(whole-dataset row count and master CSV size), ``{key}_numeric_statistic.csv``,
``column_distribution_data/column_<idx>_{scaling|leading_zeros|trailing_zeros}.csv`` — bit-width / leading-trailing zero histograms for the Dataset feature matrix UI,
optional ``{key}_{int|float}_*``.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from .column_type_manifest import (
    _fractional_decimal_places_from_cell,
    _infer_column_logical_type,
)
from .dataset_bundle import dtype_asset_filename
from .column_info_loader import logical_type_by_column_name
from .statistic import (
    column_bit_width_distribution_data,
    column_xor_zero_distribution_data,
    statistic_bit_width_distribution,
    statistic_df,
    statistic_xor_distribution,
)

COLUMN_INFO_FIELDS = (
    "column_index",
    "column_name",
    "logical_type",
    "max_point_number",
    "column_csv_size_bytes",
)

DATASET_INFO_FIELDS = (
    "data_point_count",
    "csv_size_bytes",
)

# LeCo uint64 exports: one decimal integer per line, no CSV header. Pandas default read treats the
# first value as a column name and promotes the column to float64 when any value exceeds int64 max.
FORCED_INTEGER_CATALOG_KEYS = frozenset({"fb"})

# Single-column exports with one value per line and no header row.
HEADERLESS_CATALOG_KEYS = frozenset({"books", "fb", "linear", "normal", "wiki_ts"})
_TIMESTAMP_LOGICAL_TYPES = frozenset({"timestamp", "datetime", "date", "time"})


def _timestamp_series_to_epoch_microseconds(series: pd.Series) -> pd.Series:
    """Parse timestamp text as UTC and materialize integer epoch microseconds."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    epoch_us = parsed.astype("int64") // 1_000
    return pd.Series(epoch_us, index=series.index).where(parsed.notna())


def _read_catalog_master_csv(key: str, csv_path: str) -> pd.DataFrame:
    """Read a catalog master CSV, applying dataset-specific layout fixes."""
    if key in FORCED_INTEGER_CATALOG_KEYS:
        return pd.read_csv(csv_path, header=None, names=["0"])
    df = pd.read_csv(csv_path)
    if key == "medium_posts" and "published" in df.columns:
        df["published"] = _timestamp_series_to_epoch_microseconds(df["published"])
    return df


def _catalog_logical_type(key: str, series: pd.Series) -> str:
    if key in FORCED_INTEGER_CATALOG_KEYS:
        return "integer"
    if key == "medium_posts" and str(series.name).strip().lower() == "published":
        return "timestamp"
    return _infer_column_logical_type(series)


def _catalog_numeric_columns(key: str, df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if _catalog_logical_type(key, df[col])
        in ("integer", "float", *_TIMESTAMP_LOGICAL_TYPES)
    ]


def column_info_filename(key: str) -> str:
    return f"{key}_column_info.csv"


def dataset_info_filename(key: str) -> str:
    return f"{key}_dataset_info.csv"


def count_csv_data_points(csv_path: str, *, dataset_key: str | None = None) -> int:
    """Physical line count minus one header row (unless headerless catalog export)."""
    with open(csv_path, "rb") as f:
        lines = sum(1 for _ in f)
    if dataset_key in HEADERLESS_CATALOG_KEYS:
        return lines
    return max(0, lines - 1)


def write_dataset_info_csv(
    csv_file_path: str, *, out_path: str | None = None, dataset_key: str | None = None
) -> str:
    """Write ``{key}_dataset_info.csv``: data_point_count, csv_size_bytes."""
    csv_path = os.path.abspath(csv_file_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    parent = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    if out_path is None:
        out_path = os.path.join(parent, f"{base}_dataset_info.csv")
    out_path = os.path.abspath(out_path)

    size_bytes = int(Path(csv_path).stat().st_size)
    data_points = count_csv_data_points(csv_path, dataset_key=dataset_key)
    row = {
        "data_point_count": data_points,
        "csv_size_bytes": size_bytes,
    }
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(DATASET_INFO_FIELDS), lineterminator="\n")
        w.writeheader()
        w.writerow(row)
    return out_path


def load_dataset_info(dataset_info_path: str) -> dict[str, int] | None:
    path = os.path.abspath(dataset_info_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f), None)
        if not row:
            return None
        return {
            "data_point_count": int(row["data_point_count"]),
            "csv_size_bytes": int(row["csv_size_bytes"]),
        }
    except (KeyError, TypeError, ValueError, OSError):
        return None


def catalog_numeric_filenames(key: str) -> dict[str, str]:
    return {
        "statistic": f"{key}_numeric_statistic.csv",
    }


def column_distribution_chart_dirname() -> str:
    """Legacy PNG charts (deprecated; kept for old bundles)."""
    return "column_distribution_charts"


def column_distribution_data_dirname() -> str:
    return "column_distribution_data"


def float_decimal_places_dirname() -> str:
    return "float_decimal_places"


def float_decimal_places_filename(column_index: int) -> str:
    return f"column_{int(column_index)}.csv"


FLOAT_DECIMAL_PLACES_FIELDS = (
    "decimal_places",
    "count",
    "fraction",
)


COLUMN_DISTRIBUTION_CHART_KINDS = frozenset({"scaling", "leading_zeros", "trailing_zeros"})

COLUMN_DISTRIBUTION_DATA_FIELDS = {
    "scaling": "scaling_bit_width_distribution",
    "leading_zeros": "leading_zeros_distribution",
    "trailing_zeros": "trailing_zeros_distribution",
}

COLUMN_DISTRIBUTION_ROW_FIELDS = (
    "category",
    "count",
    "percentage",
)

_LEGACY_COLUMN_DISTRIBUTION_CHART_KIND_BY_KIND = {
    "leading_zeros": "xor_leading",
    "trailing_zeros": "xor_trailing",
}


def column_distribution_chart_filename(column_index: int, chart_kind: str) -> str:
    if chart_kind not in COLUMN_DISTRIBUTION_CHART_KINDS:
        raise ValueError(chart_kind)
    return f"column_{int(column_index)}_{chart_kind}.png"


def column_distribution_data_filename(column_index: int, chart_kind: str) -> str:
    if chart_kind not in COLUMN_DISTRIBUTION_CHART_KINDS:
        raise ValueError(chart_kind)
    return f"column_{int(column_index)}_{chart_kind}.csv"


def _read_column_distribution_json(path: Path) -> list[dict[str, float | int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    out: list[dict[str, float | int]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                {
                    "category": int(row.get("category", 0)),
                    "count": int(row.get("count", 0)),
                    "percentage": float(row.get("percentage", 0.0)),
                }
            )
        except (TypeError, ValueError):
            continue
    return out


def _parse_column_distribution_row(row: dict[str, str]) -> dict[str, float | int] | None:
    try:
        return {
            "category": int(row.get("category", 0)),
            "count": int(row.get("count", 0)),
            "percentage": float(row.get("percentage", 0.0)),
        }
    except (TypeError, ValueError):
        return None


def column_distribution_data_path(
    bundle_dir: Path,
    column_index: int,
    chart_kind: str,
    *,
    operator_id: str = "",
) -> Path:
    root = bundle_dir
    if operator_id:
        from .dataset_operators import operator_bundle_dir

        root = operator_bundle_dir(bundle_dir, operator_id)
    return root / column_distribution_data_dirname() / column_distribution_data_filename(
        column_index,
        chart_kind,
    )


def read_column_distribution_data(path: Path) -> list[dict[str, float | int]]:
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                out: list[dict[str, float | int]] = []
                for row in csv.DictReader(f):
                    parsed = _parse_column_distribution_row(row)
                    if parsed is not None:
                        out.append(parsed)
                return out
        except OSError:
            return []
    legacy_json = path.with_suffix(".json")
    if legacy_json.is_file():
        return _read_column_distribution_json(legacy_json)
    return []


def write_column_distribution_data_file(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(COLUMN_DISTRIBUTION_ROW_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "category": int(row.get("category", 0)),
                    "count": int(row.get("count", 0)),
                    "percentage": float(row.get("percentage", 0.0)),
                }
            )


def column_distribution_chart_filename_candidates(
    column_index: int, chart_kind: str
) -> tuple[str, ...]:
    """Primary filename first, then legacy ``xor_*`` names when present on disk."""
    idx = int(column_index)
    primary = column_distribution_chart_filename(idx, chart_kind)
    legacy_kind = _LEGACY_COLUMN_DISTRIBUTION_CHART_KIND_BY_KIND.get(chart_kind)
    if legacy_kind:
        return (primary, f"column_{idx}_{legacy_kind}.png")
    return (primary,)


# Pre-{key}_prefix catalog files; removed when promoting/refreshing a bundle.
LEGACY_CATALOG_FILENAMES = (
    "column_types.csv",
    "numeric_statistic.csv",
    "numeric_scaling.png",
    "numeric_xor_leading.png",
    "numeric_xor_trailing.png",
)


def legacy_catalog_filenames(key: str) -> tuple[str, ...]:
    return (
        f"{key}_column_max_points.csv",
        f"{key}_int_info.csv",
        f"{key}_float_info.csv",
        *LEGACY_CATALOG_FILENAMES,
    )


def remove_legacy_catalog_files(bundle_dir: Path, key: str) -> None:
    for name in legacy_catalog_filenames(key):
        path = bundle_dir / name
        if path.is_file():
            path.unlink()


def filter_frame(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Match ``regenerate_staticdata_java.py`` / Home staticdata."""
    if data_type == "int":
        return df.select_dtypes(include=np.integer)
    if data_type == "float":
        out = df.select_dtypes(include=np.number)
        return out.dropna(axis=1, how="all")
    raise ValueError(data_type)


def write_column_info_csv(
    csv_file_path: str, *, out_path: str | None = None, dataset_key: str | None = None
) -> str:
    """
    Write ``{key}_column_info.csv``:
    column_index, column_name, logical_type, max_point_number, column_csv_size_bytes.
    Integer and string columns use max_point_number=0.
    ``column_csv_size_bytes`` is measured from a temporary single-column CSV.
    """
    import csv

    csv_path = os.path.abspath(csv_file_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    parent = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    if out_path is None:
        out_path = os.path.join(parent, f"{base}_column_info.csv")
    out_path = os.path.abspath(out_path)

    if dataset_key in FORCED_INTEGER_CATALOG_KEYS:
        rows_out = [
            {
                "column_index": 0,
                "column_name": "0",
                "logical_type": "integer",
                "max_point_number": 0,
                "column_csv_size_bytes": int(Path(csv_path).stat().st_size),
            }
        ]
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            w = csv.DictWriter(f, fieldnames=list(COLUMN_INFO_FIELDS), lineterminator="\n")
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        return out_path

    df = _read_catalog_master_csv(dataset_key or "", csv_path)
    df_raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    rows_out: list[dict[str, object]] = []
    for idx, col in enumerate(df.columns):
        logical = _catalog_logical_type(dataset_key or "", df[col])
        max_point = 0
        if logical == "float" and col in df_raw.columns:
            for cell in df_raw[col].values:
                max_point = max(max_point, _fractional_decimal_places_from_cell(cell))
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{base}_col{idx}.csv",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as tmp:
            df[[col]].to_csv(tmp.name, index=False, lineterminator="\n")
            tmp_path = Path(tmp.name)
        try:
            col_csv_size = int(tmp_path.stat().st_size)
        finally:
            tmp_path.unlink(missing_ok=True)
        rows_out.append(
            {
                "column_index": idx,
                "column_name": str(col),
                "logical_type": logical,
                "max_point_number": max_point,
                "column_csv_size_bytes": col_csv_size,
            }
        )

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMN_INFO_FIELDS), lineterminator="\n")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return out_path


def _float_columns_from_column_info(column_info_csv: Path) -> list[tuple[int, str]]:
    """(column_index, column_name) for columns with logical_type float/double."""
    out: list[tuple[int, str]] = []
    if not column_info_csv.is_file():
        return out
    with column_info_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logical = str(row.get("logical_type", "")).strip().lower()
            if logical not in ("float", "double"):
                continue
            name = str(row.get("column_name", "")).strip()
            if not name:
                continue
            try:
                idx = int(row.get("column_index", -1))
            except (TypeError, ValueError):
                continue
            out.append((idx, name))
    out.sort(key=lambda item: item[0])
    return out


def _decimal_places_distribution(cells) -> dict[int, int]:
    """Count non-empty cells by fractional decimal digits (trailing zeros stripped)."""
    from collections import Counter

    counts: Counter[int] = Counter()
    for cell in cells:
        s = str(cell).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            continue
        counts[_fractional_decimal_places_from_cell(s)] += 1
    return dict(counts)


def _format_fraction_fixed_decimal(count: int, total: int) -> str:
    """Write count/total as fixed-point decimal (never scientific notation)."""
    if total <= 0:
        return "0"
    frac_digits = len(str(total)) + len(str(count))
    with localcontext() as ctx:
        ctx.prec = frac_digits + 20
        q = Decimal(count) / Decimal(total)
        return format(q, f".{frac_digits}f")


def write_float_decimal_places_catalog_assets(
    master_csv: str,
    bundle_dir: Path,
    *,
    column_info_csv: Path,
) -> None:
    """
    Per float column, write ``float_decimal_places/column_<idx>.csv`` with
    decimal_places, count, fraction (0–1 share of non-empty cells in that column).
    """
    float_cols = _float_columns_from_column_info(column_info_csv)
    out_dir = bundle_dir / float_decimal_places_dirname()
    if not float_cols:
        if out_dir.is_dir():
            import shutil

            shutil.rmtree(out_dir)
        return

    df_raw = pd.read_csv(master_csv, dtype=str, keep_default_na=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_names = {float_decimal_places_filename(idx) for idx, _ in float_cols}
    for stale in out_dir.glob("column_*.csv"):
        if stale.name not in expected_names:
            stale.unlink()

    for col_idx, col_name in float_cols:
        if col_name not in df_raw.columns:
            continue
        dist = _decimal_places_distribution(df_raw[col_name].values)
        if not dist:
            out_path = out_dir / float_decimal_places_filename(col_idx)
            if out_path.is_file():
                out_path.unlink()
            continue
        total = sum(dist.values())
        out_path = out_dir / float_decimal_places_filename(col_idx)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            w = csv.DictWriter(
                f, fieldnames=list(FLOAT_DECIMAL_PLACES_FIELDS), lineterminator="\n"
            )
            w.writeheader()
            for places in sorted(dist):
                count = dist[places]
                w.writerow(
                    {
                        "decimal_places": places,
                        "count": count,
                        "fraction": _format_fraction_fixed_decimal(count, total),
                    }
                )


def _numeric_column_index_map(column_info_csv: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not column_info_csv.is_file():
        return out
    with column_info_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("column_name", "")).strip()
            logical = str(row.get("logical_type", "")).strip().lower()
            if not name or logical not in (
                "integer",
                "int",
                "long",
                "float",
                "double",
                *_TIMESTAMP_LOGICAL_TYPES,
            ):
                continue
            try:
                out[name] = int(row.get("column_index", -1))
            except (TypeError, ValueError):
                continue
    return out


def numeric_column_names_from_column_info(column_info_csv: Path) -> list[str]:
    """Column names with numeric or timestamp logical types, ordered by index."""
    indexed: list[tuple[int, str]] = []
    if not column_info_csv.is_file():
        return []
    with column_info_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("column_name", "")).strip()
            logical = str(row.get("logical_type", "")).strip().lower()
            if not name or logical not in (
                "integer",
                "int",
                "long",
                "float",
                "double",
                *_TIMESTAMP_LOGICAL_TYPES,
            ):
                continue
            try:
                idx = int(row.get("column_index", -1))
            except (TypeError, ValueError):
                continue
            indexed.append((idx, name))
    indexed.sort(key=lambda item: item[0])
    return [name for _, name in indexed]


def write_numeric_catalog_assets(
    key: str,
    master_csv: str,
    bundle_dir: Path,
    *,
    column_info_csv: Path,
    output_bundle_dir: Path | None = None,
    numeric_df: pd.DataFrame | None = None,
) -> None:
    """``{key}_numeric_statistic.csv`` + per-column distribution PNGs."""
    out_root = output_bundle_dir or bundle_dir
    if numeric_df is not None:
        working_df = numeric_df.copy().dropna(axis=1, how="all")
    else:
        df = _read_catalog_master_csv(key, master_csv)
        numeric_cols = _catalog_numeric_columns(key, df)
        if not numeric_cols:
            return
        working_df = df[numeric_cols].copy().dropna(axis=1, how="all")
    if working_df.shape[1] == 0:
        return
    numeric_df = working_df
    names = catalog_numeric_filenames(key)
    numeric_index_map = _numeric_column_index_map(column_info_csv)
    logical_types = logical_type_by_column_name(str(column_info_csv))
    chart_dir = out_root / column_distribution_data_dirname()
    chart_dir.mkdir(parents=True, exist_ok=True)
    for stale in chart_dir.glob("*.json"):
        stale.unlink()
    legacy_chart_dir = out_root / column_distribution_chart_dirname()
    if legacy_chart_dir.is_dir():
        for stale_png in legacy_chart_dir.glob("*.png"):
            stale_png.unlink()

    statistic_df(
        numeric_df,
        str(out_root / names["statistic"]),
        column_index_by_name=numeric_index_map,
    )
    for seq_idx, col in enumerate(numeric_df.columns):
        try:
            col_idx = int(numeric_index_map.get(str(col), seq_idx))
        except (TypeError, ValueError):
            col_idx = seq_idx
        logical_type = logical_types.get(str(col), 'float')
        single_col_df = numeric_df[[col]]
        scaling_rows = column_bit_width_distribution_data(single_col_df, logical_type)
        leading_rows, trailing_rows = column_xor_zero_distribution_data(single_col_df)
        write_column_distribution_data_file(
            chart_dir / column_distribution_data_filename(col_idx, "scaling"),
            scaling_rows,
        )
        write_column_distribution_data_file(
            chart_dir / column_distribution_data_filename(col_idx, "leading_zeros"),
            leading_rows,
        )
        write_column_distribution_data_file(
            chart_dir / column_distribution_data_filename(col_idx, "trailing_zeros"),
            trailing_rows,
        )


def remove_numeric_summary_chart_assets(key: str, bundle_dir: Path) -> None:
    """Remove deprecated dataset-wide numeric summary charts."""
    for name in (
        f"{key}_numeric_scaling.png",
        f"{key}_numeric_xor_leading.png",
        f"{key}_numeric_xor_trailing.png",
    ):
        path = bundle_dir / name
        if path.is_file():
            path.unlink()


def write_catalog_statistics_assets(key: str, master_csv: str, bundle_dir: Path) -> Path:
    """
    Write ``{key}_column_info.csv``, ``{key}_dataset_info.csv``, and ``{key}_numeric_*`` assets.
    Invoked from ``scripts/regenerate_dataset_info.py`` on every refresh.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    col_info = bundle_dir / column_info_filename(key)
    write_column_info_csv(master_csv, out_path=str(col_info), dataset_key=key)
    write_dataset_info_csv(
        master_csv,
        out_path=str(bundle_dir / dataset_info_filename(key)),
        dataset_key=key,
    )
    write_numeric_catalog_assets(
        key,
        master_csv,
        bundle_dir,
        column_info_csv=col_info,
    )
    write_float_decimal_places_catalog_assets(
        master_csv,
        bundle_dir,
        column_info_csv=col_info,
    )
    # Remove legacy prefixed directory from earlier versions.
    legacy_prefixed_chart_dir = bundle_dir / f"{key}_column_distribution_charts"
    if legacy_prefixed_chart_dir.is_dir():
        import shutil

        shutil.rmtree(legacy_prefixed_chart_dir)
    remove_numeric_summary_chart_assets(key, bundle_dir)
    remove_legacy_catalog_files(bundle_dir, key)
    return col_info


def remove_dtype_catalog_assets(key: str, bundle_dir: Path) -> None:
    """Remove ``{key}_{int|float}_statistic/scaling/xor_*`` assets if present."""
    for data_type in ("int", "float"):
        for kind in ("statistic", "scaling", "xor_leading", "xor_trailing"):
            path = bundle_dir / dtype_asset_filename(key, data_type, kind)
            if path.is_file():
                path.unlink()


def write_dtype_catalog_assets(
    key: str, master_csv: str, bundle_dir: Path, data_type: str
) -> None:
    """``{key}_{int|float}_statistic/png`` for Home ``/api/staticdata/``."""
    raw = pd.read_csv(master_csv)
    sub = filter_frame(raw, data_type)
    if sub.empty:
        return
    stat_csv = bundle_dir / dtype_asset_filename(key, data_type, "statistic")
    scaling_png = bundle_dir / dtype_asset_filename(key, data_type, "scaling")
    xor_lead = bundle_dir / dtype_asset_filename(key, data_type, "xor_leading")
    xor_trail = bundle_dir / dtype_asset_filename(key, data_type, "xor_trailing")
    logical_type = 'integer' if data_type == 'int' else 'float'
    statistic_bit_width_distribution(sub, str(scaling_png), logical_type)
    statistic_xor_distribution(sub, str(xor_lead), str(xor_trail), logical_type)
    statistic_df(sub, str(stat_csv))


def build_catalog_bundle(
    key: str,
    source_csv: Path,
    bundle_dir: Path,
    *,
    generate_dtype_assets: bool = True,
) -> bool:
    """Write column_info + numeric assets from ``dataset_csv`` master (no CSV copy into bundle)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"  write catalog stats from {source_csv.name} + {column_info_filename(key)}")
    write_catalog_statistics_assets(key, str(source_csv), bundle_dir)
    if generate_dtype_assets:
        for dt in ("int", "float"):
            write_dtype_catalog_assets(key, str(source_csv), bundle_dir, dt)
    else:
        remove_dtype_catalog_assets(key, bundle_dir)

    return True


def _cleanup_typed_tsfile_export_artifacts(export_dir: Path) -> None:
    """Remove legacy encoding-grid export dir if present."""
    import shutil

    if export_dir.is_dir():
        shutil.rmtree(export_dir)


def build_catalog_tsfile(
    key: str,
    master_csv: Path,
    bundle_dir: Path,
    tsfile_dir: Path,
    *,
    overwrite: bool,
    run_selected_tsfile_export,
) -> bool:
    """
    Build ``{key}.tsfile`` under ``dataset_tsfile/<key>/`` using table-model export + column_info.

    ``run_selected_tsfile_export(...)`` is injected for tests/scripts.
    """
    from .catalog_tsfile_selection import fixed_catalog_tsfile_config
    from .typed_tsfile_export import catalog_typed_tsfile_paths

    col_info = bundle_dir / column_info_filename(key)
    if not master_csv.is_file() or not col_info.is_file():
        return False

    tsfile_dir.mkdir(parents=True, exist_ok=True)
    paths = catalog_typed_tsfile_paths(
        key,
        tsfile_bundle_dir=tsfile_dir,
        dataset_bundle_dir=bundle_dir,
    )
    prebuilt = paths["prebuilt"]
    info_csv = paths["info_csv"]

    if prebuilt.is_file() and not overwrite:
        print(f"  keep {prebuilt.name}")
        return True

    cfg = fixed_catalog_tsfile_config(key)
    print(
        f"  table-model export: model={cfg['tsfile_model']} table={cfg['table_name']} "
        f"tags={cfg['tag_column_count']} "
        f"int={cfg['int_encoding']} float={cfg['float_encoding']} string={cfg['string_encoding']} "
        f"time={cfg['time_encoding']} compression={cfg['compression']}"
    )

    run_selected_tsfile_export(
        str(master_csv),
        str(col_info),
        str(prebuilt),
        str(info_csv),
        key,
    )

    if not prebuilt.is_file():
        print(f"  failed {key}: Java export did not create {prebuilt.name}")
        return False

    legacy_export = tsfile_dir / f"{key}_typed_tsfile_export"
    _cleanup_typed_tsfile_export_artifacts(legacy_export)
    legacy_summary = bundle_dir / f"{key}_typed_tsfile_encoding_summary.csv"
    if legacy_summary.is_file():
        legacy_summary.unlink()
    return True
