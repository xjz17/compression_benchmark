"""Catalog-page compression rate aggregates."""

from __future__ import annotations

import os
from typing import Any, MutableMapping

import pandas as pd

from .benchmark_loss import is_known_lossy_algorithm
from .column_info_loader import catalog_dtype_from_column_info_row, load_column_info_rows
from .dataset_bundle import column_info_path, dataset_dir
from .dataset_dtype_distribution import (
    BENCHMARK_DTYPES,
    CATALOG_DISTRIBUTION_DTYPES,
    java_compression_report_path,
    logical_type_to_benchmark_dtype,
)
from .dataset_float_precision_distribution import (
    FLOAT_PRECISION_CATEGORIES,
    float_column_precision_by_index,
)
from .upload_storage import (
    user_upload_abs_dir,
    user_upload_report_csv_path,
    user_upload_slug,
)
from .report_filters import is_hidden_report_algorithm
from .models import UploadedDataset

# Average-rate pies / top-level Average card: TS_2DIFF on int/float; zstd on string.
CATALOG_NUMERIC_COMPRESSION_ALGORITHM = 'TS_2DIFF'
CATALOG_STRING_COMPRESSION_ALGORITHM = 'zstd_level_3'

# Overall-rate pies / top-level Overall card: fixed Java bench algorithms per dtype.
CATALOG_OVERALL_INT_COMPRESSION_ALGORITHM = 'TS_2DIFF'
CATALOG_OVERALL_FLOAT_COMPRESSION_ALGORITHM = 'GORILLA'
# Java generic string bench row name for LZ4 (commons-compress LZ4_FRAMED).
CATALOG_OVERALL_STRING_COMPRESSION_ALGORITHM = 'lz4-framed'

CompressedSizeLookup = dict[tuple[int, str], float]
BestCompressedColumnLookup = dict[int, tuple[float, str, float | None]]
_REPORT_LOOKUP_CACHE: dict[str, CompressedSizeLookup] = {}
_BEST_LOSSLESS_REPORT_LOOKUP_CACHE: dict[str, BestCompressedColumnLookup] = {}


def clear_report_lookup_cache() -> None:
    """Drop cached Java report lookups (e.g. before a long batch recompute)."""
    _REPORT_LOOKUP_CACHE.clear()
    _BEST_LOSSLESS_REPORT_LOOKUP_CACHE.clear()


def catalog_overall_compression_algorithm_for_dtype(dtype: str) -> str | None:
    key = str(dtype or '').strip().lower()
    if key == 'int':
        return CATALOG_OVERALL_INT_COMPRESSION_ALGORITHM
    if key == 'float':
        return CATALOG_OVERALL_FLOAT_COMPRESSION_ALGORITHM
    if key == 'string':
        return CATALOG_OVERALL_STRING_COMPRESSION_ALGORITHM
    return None


def catalog_average_compression_algorithm_for_dtype(dtype: str) -> str | None:
    key = str(dtype or '').strip().lower()
    if key in ('int', 'float'):
        return CATALOG_NUMERIC_COMPRESSION_ALGORITHM
    if key == 'string':
        return CATALOG_STRING_COMPRESSION_ALGORITHM
    return None


def catalog_compression_algorithm_for_dtype(dtype: str) -> str | None:
    """Backward-compatible alias for overall-rate algorithm selection."""
    return catalog_overall_compression_algorithm_for_dtype(dtype)


def _report_dtype_for_catalog_compression_row(row: dict) -> str | None:
    logical = str(row.get('logical_type', '') or '').strip().lower()
    if logical in ('integer', 'int', 'long'):
        return 'int'
    if logical in ('float', 'double'):
        return 'float'
    if logical in ('string', 'str', 'text', 'byte', 'bytes', 'object'):
        return 'string'
    if logical in ('timestamp', 'datetime', 'date', 'time'):
        return 'int'
    return None


def _catalog_bucket_and_report_dtype(row: dict) -> tuple[str | None, str | None]:
    bucket_dtype = catalog_dtype_from_column_info_row(row)
    if bucket_dtype not in CATALOG_DISTRIBUTION_DTYPES:
        return None, None
    report_dtype = _report_dtype_for_catalog_compression_row(row)
    if not report_dtype:
        report_dtype = logical_type_to_benchmark_dtype(str(row.get('logical_type', '')))
    if report_dtype not in ('int', 'float', 'string'):
        return None, None
    return bucket_dtype, report_dtype


def _load_java_report(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _build_compressed_size_lookup(report_df: pd.DataFrame) -> CompressedSizeLookup:
    if report_df is None or report_df.empty:
        return {}
    required = ('column_index', 'algorithm', 'compressedSize')
    if not all(col in report_df.columns for col in required):
        return {}
    cur = report_df[list(required)].copy()
    cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
    cur = cur.dropna(subset=['column_index'])
    if cur.empty:
        return {}
    cur['column_index'] = cur['column_index'].astype(int)
    cur['algorithm'] = cur['algorithm'].astype(str)
    grouped = cur.groupby(['column_index', 'algorithm'], as_index=False)['compressedSize'].sum()
    out: CompressedSizeLookup = {}
    for row in grouped.itertuples(index=False):
        try:
            compressed = float(row.compressedSize)
        except (TypeError, ValueError):
            continue
        if compressed <= 0:
            continue
        out[(int(row.column_index), str(row.algorithm))] = compressed
    return out


def _report_lookup(report_path: str) -> CompressedSizeLookup:
    cached = _REPORT_LOOKUP_CACHE.get(report_path)
    if cached is not None:
        return cached
    lookup = _build_compressed_size_lookup(_load_java_report(report_path))
    _REPORT_LOOKUP_CACHE[report_path] = lookup
    return lookup


def _is_best_available_catalog_algorithm(algorithm: str) -> bool:
    key = str(algorithm or '').strip().removeprefix('\ufeff')
    if not key:
        return False
    if key.startswith('user_lossy:'):
        return False
    if is_known_lossy_algorithm(key):
        return False
    if is_hidden_report_algorithm(key):
        return False
    return True


def _build_best_lossless_compressed_lookup(
    report_df: pd.DataFrame,
) -> BestCompressedColumnLookup:
    if report_df is None or report_df.empty:
        return {}
    required = ('column_index', 'algorithm', 'compressedSize')
    if not all(col in report_df.columns for col in required):
        return {}

    optional_cols = [
        col for col in ('originalSize', 'isLossy') if col in report_df.columns
    ]
    cur = report_df[list(required) + optional_cols].copy()
    cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
    cur['compressedSize'] = pd.to_numeric(cur['compressedSize'], errors='coerce')
    if 'originalSize' in cur.columns:
        cur['originalSize'] = pd.to_numeric(cur['originalSize'], errors='coerce')
    else:
        cur['originalSize'] = pd.NA
    cur = cur.dropna(subset=['column_index', 'compressedSize'])
    cur = cur[cur['compressedSize'] > 0]
    if cur.empty:
        return {}

    cur['column_index'] = cur['column_index'].astype(int)
    cur['algorithm'] = cur['algorithm'].astype(str)
    cur = cur[cur['algorithm'].map(_is_best_available_catalog_algorithm)]
    if cur.empty:
        return {}

    if 'isLossy' in cur.columns:
        lossy_flags = pd.to_numeric(cur['isLossy'], errors='coerce')
        cur = cur[lossy_flags.isna() | (lossy_flags == 0)]
        if cur.empty:
            return {}

    grouped = cur.groupby(['column_index', 'algorithm'], as_index=False).agg({
        'compressedSize': 'sum',
        'originalSize': 'sum',
    })
    out: BestCompressedColumnLookup = {}
    for row in grouped.itertuples(index=False):
        column_index = int(row.column_index)
        try:
            compressed = float(row.compressedSize)
        except (TypeError, ValueError):
            continue
        if compressed <= 0:
            continue
        try:
            original = float(row.originalSize)
        except (TypeError, ValueError):
            original = 0.0
        original_for_compare = original if original > 0 else None
        current = out.get(column_index)
        if current is None:
            out[column_index] = (compressed, str(row.algorithm), original_for_compare)
            continue
        current_compressed, _current_algorithm, current_original = current
        if original_for_compare and current_original:
            candidate_rate = compressed / original_for_compare
            current_rate = current_compressed / current_original
            if candidate_rate < current_rate:
                out[column_index] = (compressed, str(row.algorithm), original_for_compare)
        elif compressed < current_compressed:
            out[column_index] = (compressed, str(row.algorithm), original_for_compare)
    return out


def _best_lossless_report_lookup(report_path: str) -> BestCompressedColumnLookup:
    cached = _BEST_LOSSLESS_REPORT_LOOKUP_CACHE.get(report_path)
    if cached is not None:
        return cached
    lookup = _build_best_lossless_compressed_lookup(_load_java_report(report_path))
    _BEST_LOSSLESS_REPORT_LOOKUP_CACHE[report_path] = lookup
    return lookup


def _compressed_size_for_column_algorithm(
    report_df: pd.DataFrame,
    column_index: int,
    algorithm: str,
) -> float | None:
    """Legacy helper; prefer ``_compressed_size_from_report_path`` for batch scans."""
    if report_df is None or report_df.empty:
        return None
    lookup = _build_compressed_size_lookup(report_df)
    return lookup.get((int(column_index), str(algorithm)))


def _compressed_size_from_report_path(
    report_path: str,
    column_index: int,
    algorithm: str,
) -> float | None:
    return _report_lookup(report_path).get((int(column_index), str(algorithm)))


def _load_report_lookup(
    report_cache: dict[str, CompressedSizeLookup],
    *,
    report_path: str,
) -> CompressedSizeLookup:
    if report_path not in report_cache:
        report_cache[report_path] = _report_lookup(report_path)
    return report_cache[report_path]


def _java_report_path(
    dataset_key: str,
    dtype: str,
    *,
    user_record: UploadedDataset | None = None,
) -> str:
    if user_record is not None:
        return user_upload_report_csv_path(user_record, dtype, 'java')
    return java_compression_report_path(
        dataset_dir(dataset_key),
        dataset_key,
        dtype,
    )


def _column_info_path(
    dataset_key: str,
    *,
    user_record: UploadedDataset | None = None,
) -> str:
    if user_record is not None:
        slug = user_upload_slug(user_record)
        return os.path.join(user_upload_abs_dir(user_record), f'{slug}_column_info.csv')
    return column_info_path(dataset_key)


def best_available_column_compression_metrics(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None = None,
) -> dict[str, Any]:
    """
    Dataset overview metrics from existing Java reports.

    For every benchmarkable column, choose the lossless row with the lowest
    ``compressedSize / originalSize`` currently present in that column's report.
    Columns without a usable report row are omitted because the Overview cards
    are scoped to data that already has benchmark results.
    """
    if not os.path.isfile(column_info_path):
        return {
            'original_size_bytes': 0.0,
            'compressed_size_bytes': 0.0,
            'compression_rate_sum': 0.0,
            'average_compression_rate': None,
            'used_column_count': 0,
            'missing_column_count': 0,
            'used_examples': [],
            'missing_examples': [],
        }
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return {
            'original_size_bytes': 0.0,
            'compressed_size_bytes': 0.0,
            'compression_rate_sum': 0.0,
            'average_compression_rate': None,
            'used_column_count': 0,
            'missing_column_count': 0,
            'used_examples': [],
            'missing_examples': [],
        }

    original_total = 0.0
    compressed_total = 0.0
    rates: list[float] = []
    used_examples: list[str] = []
    missing_examples: list[str] = []
    used_count = 0
    missing_count = 0
    report_cache: dict[str, BestCompressedColumnLookup] = {}

    for row in info_rows:
        _bucket_dtype, report_dtype = _catalog_bucket_and_report_dtype(row)
        if not report_dtype:
            continue
        try:
            column_index = int(row.get('column_index', -1))
            column_csv_size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if column_index < 0 or column_csv_size <= 0:
            continue

        report_path = _java_report_path(dataset_key, report_dtype, user_record=user_record)
        if report_path not in report_cache:
            report_cache[report_path] = _best_lossless_report_lookup(report_path)
        best = report_cache[report_path].get(column_index)
        if best is None:
            missing_count += 1
            if len(missing_examples) < 20:
                missing_examples.append(f'{dataset_key}:{report_dtype}:{column_index}')
            continue

        compressed, algorithm, report_original_size = best
        original_size = (
            float(report_original_size)
            if report_original_size is not None and float(report_original_size) > 0
            else column_csv_size
        )
        if original_size <= 0:
            continue
        original_total += original_size
        compressed_total += float(compressed)
        rates.append(float(compressed) / original_size)
        used_count += 1
        if len(used_examples) < 40:
            used_examples.append(f'{dataset_key}:{report_dtype}:{column_index}:{algorithm}')

    return {
        'original_size_bytes': original_total,
        'compressed_size_bytes': compressed_total,
        'compression_rate_sum': float(sum(rates)),
        'average_compression_rate': average_column_compression_rate(rates),
        'used_column_count': used_count,
        'missing_column_count': missing_count,
        'used_examples': used_examples,
        'missing_examples': missing_examples,
    }


def _accumulate_column_ratio_sums(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    csv_sums: MutableMapping[str, float],
    compressed_sums: MutableMapping[str, float],
    dtype_filter: str | None = None,
    precision_by_index: dict[int, str] | None = None,
    algorithm_for_dtype=catalog_overall_compression_algorithm_for_dtype,
) -> None:
    if not os.path.isfile(column_info_path):
        return
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return

    report_cache: dict[str, CompressedSizeLookup] = {}
    for row in info_rows:
        dtype = logical_type_to_benchmark_dtype(str(row.get('logical_type', '')))
        if not dtype:
            continue
        if dtype_filter is not None and dtype != dtype_filter:
            continue
        algorithm = algorithm_for_dtype(dtype)
        if not algorithm:
            continue
        try:
            column_index = int(row.get('column_index', -1))
            column_csv_size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if column_csv_size <= 0:
            continue

        bucket_key = dtype
        if precision_by_index is not None:
            label = precision_by_index.get(column_index)
            if not label:
                continue
            bucket_key = label

        report_path = _java_report_path(dataset_key, dtype, user_record=user_record)
        lookup = _load_report_lookup(report_cache, report_path=report_path)
        compressed = lookup.get((column_index, algorithm))
        if compressed is None:
            continue
        csv_sums[bucket_key] = float(csv_sums.get(bucket_key, 0.0)) + column_csv_size
        compressed_sums[bucket_key] = float(compressed_sums.get(bucket_key, 0.0)) + float(
            compressed
        )


def try_accumulate_dataset_overall_compression_rate_sums(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    bundle_dir: str | None,
    category_name: str,
    dtype_csv_sums: MutableMapping[str, float],
    dtype_compressed_sums: MutableMapping[str, float],
    float_precision_csv_sums: MutableMapping[str, float],
    float_precision_compressed_sums: MutableMapping[str, float],
    category_overall_csv_sum: MutableMapping[str, float],
    category_overall_compressed_sum: MutableMapping[str, float],
    global_overall_csv_total: MutableMapping[str, float],
    global_overall_compressed_total: MutableMapping[str, float],
) -> bool:
    """
    Add one dataset to overall compression aggregates when every exportable column has
    Java ``compressedSize`` for the catalog overall algorithm (int TS_2DIFF, float GORILLA,
    string lz4-framed). Returns False when the dataset is skipped (missing report or row).
    """
    if not os.path.isfile(column_info_path):
        return False
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return False

    columns: list[tuple[int, str, str, float, str, bool]] = []
    required_report_dtypes: set[str] = set()
    for row in info_rows:
        bucket_dtype, report_dtype = _catalog_bucket_and_report_dtype(row)
        if not bucket_dtype or not report_dtype:
            continue
        algorithm = catalog_overall_compression_algorithm_for_dtype(report_dtype)
        if not algorithm:
            continue
        try:
            column_index = int(row.get('column_index', -1))
            column_csv_size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if column_csv_size <= 0:
            continue
        required = bucket_dtype != 'timestamp'
        columns.append(
            (column_index, bucket_dtype, report_dtype, column_csv_size, algorithm, required)
        )
        if required:
            required_report_dtypes.add(report_dtype)

    if not columns:
        return False

    report_cache: dict[str, CompressedSizeLookup] = {}
    for report_dtype in required_report_dtypes:
        report_path = _java_report_path(dataset_key, report_dtype, user_record=user_record)
        if not os.path.isfile(report_path):
            return False
        _load_report_lookup(report_cache, report_path=report_path)

    precision_by_index: dict[int, str] = {}
    if bundle_dir:
        precision_by_index = float_column_precision_by_index(column_info_path, bundle_dir)

    per_dtype_csv = {d: 0.0 for d in CATALOG_DISTRIBUTION_DTYPES}
    per_dtype_compressed = {d: 0.0 for d in CATALOG_DISTRIBUTION_DTYPES}
    per_precision_csv = {label: 0.0 for label in FLOAT_PRECISION_CATEGORIES}
    per_precision_compressed = {label: 0.0 for label in FLOAT_PRECISION_CATEGORIES}
    dataset_csv_total = 0.0
    dataset_compressed_total = 0.0

    for column_index, bucket_dtype, report_dtype, column_csv_size, algorithm, required in columns:
        report_path = _java_report_path(dataset_key, report_dtype, user_record=user_record)
        if not os.path.isfile(report_path):
            if required:
                return False
            continue
        lookup = _load_report_lookup(report_cache, report_path=report_path)
        compressed = lookup.get((column_index, algorithm))
        if compressed is None:
            if required:
                return False
            continue
        compressed_val = float(compressed)
        per_dtype_csv[bucket_dtype] += column_csv_size
        per_dtype_compressed[bucket_dtype] += compressed_val
        dataset_csv_total += column_csv_size
        dataset_compressed_total += compressed_val
        if bucket_dtype == 'float' and report_dtype == 'float' and precision_by_index:
            label = precision_by_index.get(column_index)
            if label:
                per_precision_csv[label] += column_csv_size
                per_precision_compressed[label] += compressed_val

    if dataset_csv_total <= 0 or dataset_compressed_total <= 0:
        return False

    for dtype in CATALOG_DISTRIBUTION_DTYPES:
        if per_dtype_csv[dtype] > 0:
            dtype_csv_sums[dtype] = float(dtype_csv_sums.get(dtype, 0.0)) + per_dtype_csv[dtype]
            dtype_compressed_sums[dtype] = float(
                dtype_compressed_sums.get(dtype, 0.0),
            ) + per_dtype_compressed[dtype]
    for label in FLOAT_PRECISION_CATEGORIES:
        if per_precision_csv[label] > 0:
            float_precision_csv_sums[label] = float(
                float_precision_csv_sums.get(label, 0.0),
            ) + per_precision_csv[label]
            float_precision_compressed_sums[label] = float(
                float_precision_compressed_sums.get(label, 0.0),
            ) + per_precision_compressed[label]

    cat = str(category_name or '').strip() or 'Others'
    category_overall_csv_sum[cat] = float(category_overall_csv_sum.get(cat, 0.0)) + dataset_csv_total
    category_overall_compressed_sum[cat] = float(
        category_overall_compressed_sum.get(cat, 0.0),
    ) + dataset_compressed_total
    global_overall_csv_total['total'] = float(
        global_overall_csv_total.get('total', 0.0),
    ) + dataset_csv_total
    global_overall_compressed_total['total'] = float(
        global_overall_compressed_total.get('total', 0.0),
    ) + dataset_compressed_total
    return True


def collect_numeric_column_ts2diff_rates(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None = None,
) -> list[float]:
    """Per-column ``TS_2DIFF`` compression rate (compressedSize / column_csv_size_bytes) for int/float."""
    if not os.path.isfile(column_info_path):
        return []
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return []

    algorithm = CATALOG_NUMERIC_COMPRESSION_ALGORITHM
    report_cache: dict[str, CompressedSizeLookup] = {}
    rates: list[float] = []
    for row in info_rows:
        dtype = logical_type_to_benchmark_dtype(str(row.get('logical_type', '')))
        if dtype not in ('int', 'float'):
            continue
        try:
            column_index = int(row.get('column_index', -1))
            column_csv_size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if column_csv_size <= 0:
            continue
        report_path = _java_report_path(dataset_key, dtype, user_record=user_record)
        lookup = _load_report_lookup(report_cache, report_path=report_path)
        compressed = lookup.get((column_index, algorithm))
        if compressed is None:
            continue
        rates.append(float(compressed) / column_csv_size)
    return rates


def average_column_compression_rate(rates: list[float]) -> float | None:
    if not rates:
        return None
    return float(sum(rates)) / float(len(rates))


def average_compression_rates_by_bucket(
    bucket_rates: MutableMapping[str, list[float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, rates in bucket_rates.items():
        avg = average_column_compression_rate(rates)
        if avg is not None:
            out[str(name)] = avg
    return out


def extend_numeric_ts2diff_rates_by_bucket(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    bucket_rates: MutableMapping[str, list[float]],
    bucket_name: str,
    rates: list[float] | None = None,
) -> None:
    if rates is None:
        rates = collect_numeric_column_ts2diff_rates(
            column_info_path=column_info_path,
            dataset_key=dataset_key,
            user_record=user_record,
        )
    if not rates:
        return
    key = str(bucket_name or '').strip()
    if not key:
        return
    bucket_rates.setdefault(key, [])
    bucket_rates[key].extend(rates)


def extend_column_average_rates_by_dtype(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    dtype_rates: MutableMapping[str, list[float]],
) -> None:
    """Append per-column compression rates into catalog int/float/timestamp buckets."""
    if not os.path.isfile(column_info_path):
        return
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return

    report_cache: dict[str, CompressedSizeLookup] = {}
    for row in info_rows:
        bucket_dtype, report_dtype = _catalog_bucket_and_report_dtype(row)
        if not bucket_dtype or not report_dtype:
            continue
        algorithm = catalog_average_compression_algorithm_for_dtype(report_dtype)
        if not algorithm:
            continue
        try:
            column_index = int(row.get('column_index', -1))
            column_csv_size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if column_csv_size <= 0:
            continue
        report_path = _java_report_path(dataset_key, report_dtype, user_record=user_record)
        lookup = _load_report_lookup(report_cache, report_path=report_path)
        compressed = lookup.get((column_index, algorithm))
        if compressed is None:
            continue
        dtype_rates.setdefault(bucket_dtype, [])
        dtype_rates[bucket_dtype].append(float(compressed) / column_csv_size)


def extend_numeric_ts2diff_rates_by_dtype(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    dtype_rates: MutableMapping[str, list[float]],
) -> None:
    """Backward-compatible alias for dtype average pie charts."""
    extend_column_average_rates_by_dtype(
        column_info_path=column_info_path,
        dataset_key=dataset_key,
        user_record=user_record,
        dtype_rates=dtype_rates,
    )


def extend_float_ts2diff_rates_by_precision(
    *,
    column_info_path: str,
    bundle_dir: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    precision_rates: MutableMapping[str, list[float]],
) -> None:
    """Append per-column TS_2DIFF rates for float columns grouped by precision label."""
    precision_by_index = float_column_precision_by_index(column_info_path, bundle_dir)
    if not precision_by_index:
        return
    if not os.path.isfile(column_info_path):
        return
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return

    algorithm = CATALOG_NUMERIC_COMPRESSION_ALGORITHM
    report_path = _java_report_path(dataset_key, 'float', user_record=user_record)
    lookup = _report_lookup(report_path)
    for row in info_rows:
        dtype = logical_type_to_benchmark_dtype(str(row.get('logical_type', '')))
        if dtype != 'float':
            continue
        try:
            column_index = int(row.get('column_index', -1))
            column_csv_size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if column_csv_size <= 0:
            continue
        label = precision_by_index.get(column_index)
        if not label:
            continue
        compressed = lookup.get((column_index, algorithm))
        if compressed is None:
            continue
        precision_rates.setdefault(label, [])
        precision_rates[label].append(float(compressed) / column_csv_size)


def empty_dtype_rate_sums() -> tuple[dict[str, float], dict[str, float]]:
    zeros = {d: 0.0 for d in CATALOG_DISTRIBUTION_DTYPES}
    return dict(zeros), dict(zeros)


def empty_float_precision_rate_sums() -> tuple[dict[str, float], dict[str, float]]:
    zeros = {label: 0.0 for label in FLOAT_PRECISION_CATEGORIES}
    return dict(zeros), dict(zeros)


def accumulate_dtype_compression_rate_sums(
    *,
    column_info_path: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    csv_sums: MutableMapping[str, float],
    compressed_sums: MutableMapping[str, float],
) -> bool:
    """Legacy wrapper; prefer ``try_accumulate_dataset_overall_compression_rate_sums``."""
    holder_csv: dict[str, float] = {}
    holder_compressed: dict[str, float] = {}
    fp_csv, fp_compressed = empty_float_precision_rate_sums()
    ok = try_accumulate_dataset_overall_compression_rate_sums(
        column_info_path=column_info_path,
        dataset_key=dataset_key,
        user_record=user_record,
        bundle_dir=None,
        category_name='',
        dtype_csv_sums=csv_sums,
        dtype_compressed_sums=compressed_sums,
        float_precision_csv_sums=fp_csv,
        float_precision_compressed_sums=fp_compressed,
        category_overall_csv_sum=holder_csv,
        category_overall_compressed_sum=holder_compressed,
        global_overall_csv_total=holder_csv,
        global_overall_compressed_total=holder_compressed,
    )
    return ok


def accumulate_float_precision_compression_rate_sums(
    *,
    column_info_path: str,
    bundle_dir: str,
    dataset_key: str,
    user_record: UploadedDataset | None,
    csv_sums: MutableMapping[str, float],
    compressed_sums: MutableMapping[str, float],
) -> None:
    precision_by_index = float_column_precision_by_index(column_info_path, bundle_dir)
    if not precision_by_index:
        return
    _accumulate_column_ratio_sums(
        column_info_path=column_info_path,
        dataset_key=dataset_key,
        user_record=user_record,
        csv_sums=csv_sums,
        compressed_sums=compressed_sums,
        dtype_filter='float',
        precision_by_index=precision_by_index,
    )


def compression_rates_from_sums(
    csv_sums: MutableMapping[str, float],
    compressed_sums: MutableMapping[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, csv_total in csv_sums.items():
        try:
            csv_val = float(csv_total)
            compressed_val = float(compressed_sums.get(name, 0.0))
        except (TypeError, ValueError):
            continue
        if csv_val <= 0 or compressed_val <= 0:
            continue
        out[str(name)] = compressed_val / csv_val
    return out


def global_compression_rate(
    csv_sums: MutableMapping[str, float],
    compressed_sums: MutableMapping[str, float],
) -> float | None:
    try:
        csv_total = float(sum(float(v) for v in csv_sums.values()))
        compressed_total = float(sum(float(v) for v in compressed_sums.values()))
    except (TypeError, ValueError):
        return None
    if csv_total <= 0 or compressed_total <= 0:
        return None
    return compressed_total / csv_total


def category_compression_rates(
    catalog_entries: list[tuple[str, str, UploadedDataset | None]],
) -> dict[str, float]:
    """Per major category: sum(compressedSize) / sum(column_csv_size_bytes) with dtype-specific algos."""
    category_csv_sum: dict[str, float] = {}
    category_compressed_sum: dict[str, float] = {}
    for category, dataset_key, user_record in catalog_entries:
        col_info = _column_info_path(dataset_key, user_record=user_record)
        per_cat_csv: dict[str, float] = {}
        per_cat_compressed: dict[str, float] = {}
        _accumulate_column_ratio_sums(
            column_info_path=col_info,
            dataset_key=dataset_key,
            user_record=user_record,
            csv_sums=per_cat_csv,
            compressed_sums=per_cat_compressed,
        )
        if not per_cat_csv:
            continue
        try:
            csv_total = float(sum(per_cat_csv.values()))
            compressed_total = float(sum(per_cat_compressed.values()))
        except (TypeError, ValueError):
            continue
        if csv_total <= 0 or compressed_total <= 0:
            continue
        category_csv_sum[category] = category_csv_sum.get(category, 0.0) + csv_total
        category_compressed_sum[category] = category_compressed_sum.get(
            category, 0.0,
        ) + compressed_total
    return {
        name: category_compressed_sum[name] / category_csv_sum[name]
        for name in category_csv_sum
        if category_csv_sum.get(name, 0.0) > 0
    }
