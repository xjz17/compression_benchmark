"""Aggregate float column fixed vs non-fixed precision for the Dataset catalog page."""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from typing import MutableMapping

import pandas as pd

from .dataset_catalog_assets import (
    float_decimal_places_dirname,
    float_decimal_places_filename,
    load_dataset_info,
)
from .dataset_dtype_distribution import (
    _is_java_tsfile_encoding_algorithm,
    java_compression_report_path,
)
from .column_info_loader import load_column_info_rows

FLOAT_PRECISION_FIXED_LABEL = 'Fixed precision'
FLOAT_PRECISION_NON_FIXED_LABEL = 'Non-fixed precision'
FLOAT_PRECISION_CATEGORIES: tuple[str, ...] = (
    FLOAT_PRECISION_FIXED_LABEL,
    FLOAT_PRECISION_NON_FIXED_LABEL,
)
FIXED_PRECISION_DOMINANT_FRACTION_THRESHOLD = 0.85

# Home staticdata query param ``float_precision`` (dataset-level float runs only).
FLOAT_PRECISION_FILTER_ALL = 'all'
FLOAT_PRECISION_FILTER_FIXED = 'fixed'
FLOAT_PRECISION_FILTER_NON_FIXED = 'non_fixed'
FLOAT_PRECISION_FILTER_VALUES: tuple[str, ...] = (
    FLOAT_PRECISION_FILTER_ALL,
    FLOAT_PRECISION_FILTER_FIXED,
    FLOAT_PRECISION_FILTER_NON_FIXED,
)
REGER_FLOAT_PRECISION_REPORT_SUFFIX = '_float_cpp_reger_precision_result.csv'
REGER_FLOAT_PRECISION_REPORT_FIELDS: tuple[str, ...] = (
    'precision_scope',
    'column_index',
    'algorithm',
    'originalSize',
    'compressedSize',
    'compressionTime',
    'decompressionTime',
    'isLossy',
    'lossRate',
)


def _empty_precision_counters() -> dict[str, float]:
    return {label: 0.0 for label in FLOAT_PRECISION_CATEGORIES}


def _is_float_logical_type(logical: str) -> bool:
    return str(logical or '').strip().lower() in ('float', 'double')


def _decimal_places_csv_path(bundle_dir: str, column_index: int) -> str:
    return os.path.join(
        bundle_dir,
        float_decimal_places_dirname(),
        float_decimal_places_filename(column_index),
    )


def classify_column_fixed_precision(decimal_places_csv: str) -> str:
    """
    Fixed when the dominant decimal-places bucket has fraction >= 85%;
    otherwise non-fixed. Missing/empty stats -> non-fixed.
    """
    if not os.path.isfile(decimal_places_csv):
        return FLOAT_PRECISION_NON_FIXED_LABEL
    try:
        with open(decimal_places_csv, encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return FLOAT_PRECISION_NON_FIXED_LABEL
    if not rows:
        return FLOAT_PRECISION_NON_FIXED_LABEL

    best_fraction = -1.0
    for row in rows:
        try:
            fraction = float(row.get('fraction', 0) or 0)
        except (TypeError, ValueError):
            continue
        if fraction > best_fraction:
            best_fraction = fraction
    if best_fraction < 0:
        return FLOAT_PRECISION_NON_FIXED_LABEL
    if best_fraction >= FIXED_PRECISION_DOMINANT_FRACTION_THRESHOLD:
        return FLOAT_PRECISION_FIXED_LABEL
    return FLOAT_PRECISION_NON_FIXED_LABEL


def parse_float_precision_filter(raw: object) -> str:
    """Normalize API ``float_precision`` to ``all`` / ``fixed`` / ``non_fixed``."""
    s = str(raw or '').strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        '': FLOAT_PRECISION_FILTER_ALL,
        'all': FLOAT_PRECISION_FILTER_ALL,
        'fixed': FLOAT_PRECISION_FILTER_FIXED,
        'fixed_precision': FLOAT_PRECISION_FILTER_FIXED,
        'non_fixed': FLOAT_PRECISION_FILTER_NON_FIXED,
        'nonfixed': FLOAT_PRECISION_FILTER_NON_FIXED,
        'non_fixed_precision': FLOAT_PRECISION_FILTER_NON_FIXED,
    }
    return aliases.get(s, FLOAT_PRECISION_FILTER_ALL)


def precision_label_for_filter(float_precision: str) -> str | None:
    parsed = parse_float_precision_filter(float_precision)
    if parsed == FLOAT_PRECISION_FILTER_FIXED:
        return FLOAT_PRECISION_FIXED_LABEL
    if parsed == FLOAT_PRECISION_FILTER_NON_FIXED:
        return FLOAT_PRECISION_NON_FIXED_LABEL
    return None


def allowed_float_column_indexes_for_filter(
    column_info_path: str,
    bundle_dir: str,
    float_precision: str,
) -> set[int] | None:
    """Return allowed ``column_index`` values, or ``None`` when filter is ``all``."""
    if parse_float_precision_filter(float_precision) == FLOAT_PRECISION_FILTER_ALL:
        return None
    want = precision_label_for_filter(float_precision)
    if want is None:
        return None
    precision_by_idx = float_column_precision_by_index(column_info_path, bundle_dir)
    return {idx for idx, label in precision_by_idx.items() if label == want}


def filter_report_dataframe_for_float_precision(
    df: pd.DataFrame,
    *,
    column_info_path: str,
    bundle_dir: str,
    float_precision: str,
) -> pd.DataFrame:
    """Keep only float report rows whose ``column_index`` matches the precision filter."""
    if df is None or df.empty:
        return df
    allowed = allowed_float_column_indexes_for_filter(
        column_info_path,
        bundle_dir,
        float_precision,
    )
    if allowed is None:
        return df
    if 'column_index' not in df.columns:
        return df.iloc[0:0].copy()
    cur = df.copy()
    cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
    algorithm = cur['algorithm'].astype(str).str.strip().str.lower() if 'algorithm' in cur.columns else None
    if algorithm is not None:
        cur = cur[algorithm != 'reger']
    cur = cur[cur['column_index'].isin(allowed)] if allowed else cur.iloc[0:0].copy()
    reger_row = load_reger_float_precision_row(bundle_dir, float_precision)
    if reger_row is None:
        return cur
    aligned = {column: reger_row.get(column, '') for column in df.columns}
    return pd.concat([cur, pd.DataFrame([aligned], columns=df.columns)], ignore_index=True)


def reger_float_precision_report_path(bundle_dir: str) -> str:
    normalized = os.path.abspath(str(bundle_dir))
    dataset_key = os.path.basename(normalized.rstrip(os.sep))
    return os.path.join(normalized, f'{dataset_key}{REGER_FLOAT_PRECISION_REPORT_SUFFIX}')


def load_reger_float_precision_row(
    bundle_dir: str,
    float_precision: str,
) -> dict[str, object] | None:
    """Load one explicit REGER aggregate for a floating-point precision stratum."""
    scope = parse_float_precision_filter(float_precision)
    if scope not in (FLOAT_PRECISION_FILTER_FIXED, FLOAT_PRECISION_FILTER_NON_FIXED):
        return None
    report_path = reger_float_precision_report_path(bundle_dir)
    if not os.path.isfile(report_path):
        return None
    try:
        report = pd.read_csv(report_path)
    except Exception:
        return None
    if report.empty or 'precision_scope' not in report.columns:
        return None
    selected = report[report['precision_scope'].astype(str).map(parse_float_precision_filter) == scope]
    if selected.empty:
        return None
    row = selected.iloc[0].to_dict()
    row['column_index'] = -1
    row['algorithm'] = 'reger'
    return row


def float_precision_catalog_fields(
    column_info_path: str,
    bundle_dir: str,
) -> dict[str, object]:
    """Per-dataset float column counts and indexes for Dataset catalog filtering."""
    precision_by_idx = float_column_precision_by_index(column_info_path, bundle_dir)
    indexes: dict[str, list[int]] = {
        FLOAT_PRECISION_FILTER_FIXED: [],
        FLOAT_PRECISION_FILTER_NON_FIXED: [],
    }
    for idx, label in sorted(precision_by_idx.items()):
        if label == FLOAT_PRECISION_FIXED_LABEL:
            indexes[FLOAT_PRECISION_FILTER_FIXED].append(int(idx))
        elif label == FLOAT_PRECISION_NON_FIXED_LABEL:
            indexes[FLOAT_PRECISION_FILTER_NON_FIXED].append(int(idx))
    return {
        'float_precision_column_counts': {
            FLOAT_PRECISION_FILTER_FIXED: len(indexes[FLOAT_PRECISION_FILTER_FIXED]),
            FLOAT_PRECISION_FILTER_NON_FIXED: len(indexes[FLOAT_PRECISION_FILTER_NON_FIXED]),
        },
        'float_precision_column_indexes': indexes,
    }


def empty_float_precision_catalog_fields() -> dict[str, object]:
    return {
        'float_precision_column_counts': {
            FLOAT_PRECISION_FILTER_FIXED: 0,
            FLOAT_PRECISION_FILTER_NON_FIXED: 0,
        },
        'float_precision_column_indexes': {
            FLOAT_PRECISION_FILTER_FIXED: [],
            FLOAT_PRECISION_FILTER_NON_FIXED: [],
        },
    }


def float_column_precision_by_index(
    column_info_path: str,
    bundle_dir: str,
) -> dict[int, str]:
    """Map float/double ``column_index`` -> fixed / non-fixed precision label."""
    return _float_column_precision_by_index_cached(
        str(column_info_path),
        str(bundle_dir),
    )


@lru_cache(maxsize=512)
def _float_column_precision_by_index_cached(
    column_info_path: str,
    bundle_dir: str,
) -> dict[int, str]:
    if not os.path.isfile(column_info_path):
        return {}
    out: dict[int, str] = {}
    try:
        rows = load_column_info_rows(column_info_path)
    except Exception:
        return out
    for row in rows:
        if not _is_float_logical_type(str(row.get('logical_type', ''))):
            continue
        try:
            col_idx = int(row.get('column_index', -1))
        except (TypeError, ValueError):
            continue
        if col_idx < 0:
            continue
        out[col_idx] = classify_column_fixed_precision(_decimal_places_csv_path(bundle_dir, col_idx))
    return out


def _float_column_info_rows(column_info_path: str) -> list[dict]:
    if not os.path.isfile(column_info_path):
        return []
    try:
        rows = load_column_info_rows(column_info_path)
    except Exception:
        return []
    return [r for r in rows if _is_float_logical_type(str(r.get('logical_type', '')))]


def min_float_tsfile_compressed_by_column(report_path: str) -> dict[int, float]:
    """Per-column min TsFile-encoding ``compressedSize`` from float Java report."""
    if not os.path.isfile(report_path):
        return {}
    try:
        df = pd.read_csv(report_path)
    except Exception:
        return {}
    if df is None or df.empty or 'algorithm' not in df.columns or 'compressedSize' not in df.columns:
        return {}
    if 'column_index' not in df.columns:
        return {}

    cur = df.copy()
    cur['algorithm'] = cur['algorithm'].astype(str)
    cur['compressedSize'] = pd.to_numeric(cur['compressedSize'], errors='coerce').fillna(0)
    cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
    cur = cur[cur['compressedSize'] > 0]
    cur = cur[cur['column_index'].notna()]
    cur = cur[cur['algorithm'].map(_is_java_tsfile_encoding_algorithm)]
    if cur.empty:
        return {}

    out: dict[int, float] = {}
    for col_idx, group in cur.groupby('column_index', sort=False):
        try:
            out[int(col_idx)] = float(group['compressedSize'].min())
        except (TypeError, ValueError):
            continue
    return out


def accumulate_float_precision_metrics(
    *,
    column_info_path: str,
    dataset_info_path: str,
    bundle_dir: str,
    dataset_key: str,
    column_counts: MutableMapping[str, float],
    data_point_counts: MutableMapping[str, float],
    csv_size_bytes: MutableMapping[str, float],
    tsfile_size_bytes: MutableMapping[str, float],
) -> None:
    """Add one dataset's float columns (fixed vs non-fixed) into accumulators."""
    precision_by_idx = float_column_precision_by_index(column_info_path, bundle_dir)
    if not precision_by_idx:
        return

    float_rows = _float_column_info_rows(column_info_path)
    info = load_dataset_info(dataset_info_path) if os.path.isfile(dataset_info_path) else None
    row_count = max(0, int(info.get('data_point_count', 0)) if info else 0)

    for row in float_rows:
        try:
            col_idx = int(row.get('column_index', -1))
        except (TypeError, ValueError):
            continue
        label = precision_by_idx.get(col_idx)
        if not label:
            continue
        column_counts[label] = float(column_counts.get(label, 0.0)) + 1.0
        data_point_counts[label] = float(data_point_counts.get(label, 0.0)) + float(row_count)
        try:
            csv_size_bytes[label] = float(csv_size_bytes.get(label, 0.0)) + float(
                row.get('column_csv_size_bytes', 0) or 0
            )
        except (TypeError, ValueError):
            pass

    report_path = java_compression_report_path(bundle_dir, dataset_key, 'float')
    ts_by_col = min_float_tsfile_compressed_by_column(report_path)
    for col_idx, compressed in ts_by_col.items():
        label = precision_by_idx.get(col_idx)
        if not label or compressed <= 0:
            continue
        tsfile_size_bytes[label] = float(tsfile_size_bytes.get(label, 0.0)) + float(compressed)
