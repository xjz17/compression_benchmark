"""Aggregate int / float / timestamp distributions for the Dataset catalog page."""

from __future__ import annotations

import os
from typing import MutableMapping

import pandas as pd

from .column_info_loader import catalog_dtype_from_column_info_row, load_column_info_rows
from .dataset_catalog_assets import load_dataset_info
from .native_lz_family import NATIVE_LZ_FAMILY_ALGORITHMS

# String/byte columns are disabled for now; keep dataset-level distributions numeric/time only.
BENCHMARK_DTYPES: tuple[str, ...] = ('int', 'float')
CATALOG_DISTRIBUTION_DTYPES: tuple[str, ...] = ('int', 'float', 'timestamp')

# TsFile TSEncodings present in Java dtype reports (see TsFile.vue algorithmsByType).
_JAVA_TSFILE_ENCODING_ALGORITHMS = frozenset(
    {
        'chimp',
        'gorilla',
        'rle',
        'rlbe',
        'sprintz',
        'ts_2diff',
        'ts_2diff_bos',
    }
)

_JAVA_TSFILE_STRING_ENCODING_ALGORITHMS = frozenset(
    {
        'plain',
        'dictionary',
    }
)

_JAVA_TSFILE_CHUNK_COMPRESSION_ALGORITHMS = frozenset(
    {
        'snappy',
        'gzip',
        'lz4',
        'zstd',
        'xz',
        'lzma',
    }
)

# String columns: generic byte compressors (see benchmarkAlgorithmCatalog STRING_GENERIC_*).
_JAVA_STRING_GENERIC_ALGORITHM_EXACT = frozenset(
    {
        'brotli',
        'gz',
        'lz4-framed',
        'snappy-framed',
        *NATIVE_LZ_FAMILY_ALGORITHMS,
    }
)


def logical_type_to_benchmark_dtype(logical: str) -> str | None:
    low = str(logical or '').strip().lower()
    if low in ('integer', 'int', 'long'):
        return 'int'
    if low in ('float', 'double'):
        return 'float'
    if low in ('timestamp', 'datetime', 'date', 'time'):
        return 'timestamp'
    # String/byte is intentionally excluded from the current benchmark scope.
    return None


def _logical_type_to_report_dtype(logical: str) -> str | None:
    low = str(logical or '').strip().lower()
    if low in ('integer', 'int', 'long'):
        return 'int'
    if low in ('float', 'double'):
        return 'float'
    # String/byte is intentionally excluded from the current benchmark scope.
    return None


def _empty_dtype_counters() -> dict[str, float]:
    return {d: 0.0 for d in CATALOG_DISTRIBUTION_DTYPES}


def _is_java_tsfile_encoding_algorithm(algo: str) -> bool:
    return str(algo or '').strip().lower() in _JAVA_TSFILE_ENCODING_ALGORITHMS


def _is_java_string_generic_algorithm(algo: str) -> bool:
    key = str(algo or '').strip().lower()
    if key in _JAVA_STRING_GENERIC_ALGORITHM_EXACT:
        return True
    if key in ('xz', 'lzma') or key.startswith('xz_level_') or key.startswith('lzma_level_'):
        return True
    if key == 'zstd' or key.startswith('zstd_level_'):
        return True
    return False


def _is_tsfile_staticdata_algorithm(algo: str, dtype: str) -> bool:
    key = str(algo or '').strip().lower()
    if key in _JAVA_TSFILE_CHUNK_COMPRESSION_ALGORITHMS:
        return True
    if dtype == 'string':
        return key in _JAVA_TSFILE_STRING_ENCODING_ALGORITHMS
    return _is_java_tsfile_encoding_algorithm(algo)


def _algorithm_filter_for_dtype_tsfile_size(dtype: str, *, tsfile_staticdata_report: bool = False):
    if tsfile_staticdata_report:
        return lambda algo: _is_tsfile_staticdata_algorithm(algo, dtype)
    if dtype == 'string':
        return _is_java_string_generic_algorithm
    return _is_java_tsfile_encoding_algorithm


def column_counts_by_dtype(
    column_info_path: str,
    *,
    force_one_timestamp_axis: bool = False,
) -> dict[str, int]:
    out = {d: 0 for d in CATALOG_DISTRIBUTION_DTYPES}
    if not os.path.isfile(column_info_path):
        return out
    try:
        rows = load_column_info_rows(column_info_path)
    except Exception:
        return out
    for row in rows:
        dt = catalog_dtype_from_column_info_row(row)
        if dt:
            out[dt] += 1
    if force_one_timestamp_axis:
        out['timestamp'] = 1
    return out


def data_point_counts_by_dtype(
    column_info_path: str,
    data_point_count: int,
    *,
    force_one_timestamp_axis: bool = False,
) -> dict[str, int]:
    cols = column_counts_by_dtype(
        column_info_path,
        force_one_timestamp_axis=force_one_timestamp_axis,
    )
    rows = max(0, int(data_point_count or 0))
    return {d: rows * cols[d] for d in CATALOG_DISTRIBUTION_DTYPES}


def column_csv_bytes_by_dtype(
    column_info_path: str,
    data_point_count: int,
    *,
    force_one_timestamp_axis: bool = False,
) -> dict[str, float]:
    """
    Dataset-page dtype CSV-size proxy.

    Numeric and timestamp slices are compared as dense 64-bit values, so their
    chart size is ``column_count * row_count * 8`` instead of CSV text bytes.
    String/byte data keeps the measured CSV text size, although it is hidden in
    the focused Dataset-page dtype charts.
    """
    out = _empty_dtype_counters()
    if not os.path.isfile(column_info_path):
        return out
    try:
        row_count = max(0, int(data_point_count or 0))
    except (TypeError, ValueError):
        row_count = 0
    cols = column_counts_by_dtype(
        column_info_path,
        force_one_timestamp_axis=force_one_timestamp_axis,
    )
    for dtype in CATALOG_DISTRIBUTION_DTYPES:
        out[dtype] = float(row_count * 8 * cols[dtype])
    return out


def java_compression_report_path(bundle_dir: str, dataset_key: str, dtype: str) -> str:
    return os.path.join(bundle_dir, f'{dataset_key}_{dtype}_java_compression_result.csv')


def tsfile_compression_report_path(bundle_dir: str, dataset_key: str, dtype: str) -> str:
    return os.path.join(bundle_dir, f'{dataset_key}_{dtype}_tsfile_compression_result.csv')


def sum_tsfile_column_mins_from_java_report(
    report_path: str,
    dtype: str,
    *,
    tsfile_staticdata_report: bool = False,
) -> float:
    """
    Per column in a dtype-scoped Java report: min ``compressedSize`` among eligible
    algorithms (TsFile encodings for int/float).
    """
    algo_filter = _algorithm_filter_for_dtype_tsfile_size(
        dtype, tsfile_staticdata_report=tsfile_staticdata_report
    )
    if not os.path.isfile(report_path):
        return 0.0
    try:
        df = pd.read_csv(report_path)
    except Exception:
        return 0.0
    if df is None or df.empty or 'algorithm' not in df.columns:
        return 0.0
    if 'compressedSize' not in df.columns:
        return 0.0

    cur = df.copy()
    cur['algorithm'] = cur['algorithm'].astype(str)
    cur['compressedSize'] = pd.to_numeric(cur['compressedSize'], errors='coerce').fillna(0)
    cur = cur[cur['compressedSize'] > 0]
    cur = cur[cur['algorithm'].map(algo_filter)]
    if cur.empty:
        return 0.0

    total = 0.0
    if 'column_index' in cur.columns:
        cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
        cur = cur[cur['column_index'].notna()]
        for _, group in cur.groupby('column_index', sort=False):
            try:
                total += float(group['compressedSize'].min())
            except (TypeError, ValueError):
                continue
    else:
        try:
            total = float(cur['compressedSize'].min())
        except (TypeError, ValueError):
            total = 0.0
    return float(total)


def min_tsfile_column_size_from_java_report(
    report_path: str,
    dtype: str,
    column_index: int,
    *,
    tsfile_staticdata_report: bool = False,
) -> float:
    """Min eligible compressed size for one source column in a dtype report."""
    algo_filter = _algorithm_filter_for_dtype_tsfile_size(
        dtype, tsfile_staticdata_report=tsfile_staticdata_report
    )
    if not os.path.isfile(report_path):
        return 0.0
    try:
        df = pd.read_csv(report_path)
    except Exception:
        return 0.0
    if df is None or df.empty or 'algorithm' not in df.columns:
        return 0.0
    if 'compressedSize' not in df.columns or 'column_index' not in df.columns:
        return 0.0

    cur = df.copy()
    cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
    cur = cur[cur['column_index'] == int(column_index)]
    if cur.empty:
        return 0.0
    cur['algorithm'] = cur['algorithm'].astype(str)
    cur['compressedSize'] = pd.to_numeric(cur['compressedSize'], errors='coerce').fillna(0)
    cur = cur[cur['compressedSize'] > 0]
    cur = cur[cur['algorithm'].map(algo_filter)]
    if cur.empty:
        return 0.0
    try:
        return float(cur['compressedSize'].min())
    except (TypeError, ValueError):
        return 0.0


def accumulate_dtype_metrics(
    *,
    column_info_path: str,
    dataset_info_path: str,
    bundle_dir: str,
    dataset_key: str,
    column_counts: MutableMapping[str, float],
    data_point_counts: MutableMapping[str, float],
    csv_size_bytes: MutableMapping[str, float],
    tsfile_size_bytes: MutableMapping[str, float],
    force_one_timestamp_axis: bool = False,
) -> None:
    """Add one dataset's column_info / dataset_info / compression reports into accumulators."""
    if not os.path.isfile(column_info_path):
        return

    cols = column_counts_by_dtype(
        column_info_path,
        force_one_timestamp_axis=force_one_timestamp_axis,
    )
    for dtype in CATALOG_DISTRIBUTION_DTYPES:
        column_counts[dtype] = float(column_counts.get(dtype, 0.0)) + float(cols[dtype])

    info = load_dataset_info(dataset_info_path) if os.path.isfile(dataset_info_path) else None
    row_count = int(info.get('data_point_count', 0)) if info else 0
    points = data_point_counts_by_dtype(
        column_info_path,
        row_count,
        force_one_timestamp_axis=force_one_timestamp_axis,
    )
    for dtype in CATALOG_DISTRIBUTION_DTYPES:
        data_point_counts[dtype] = float(data_point_counts.get(dtype, 0.0)) + float(points[dtype])

    col_csv = column_csv_bytes_by_dtype(
        column_info_path,
        row_count,
        force_one_timestamp_axis=force_one_timestamp_axis,
    )
    for dtype in CATALOG_DISTRIBUTION_DTYPES:
        size = float(col_csv.get(dtype, 0.0))
        if size > 0:
            csv_size_bytes[dtype] = float(csv_size_bytes.get(dtype, 0.0)) + size

    try:
        rows = load_column_info_rows(column_info_path)
    except Exception:
        rows = []
    for row in rows:
        bucket_dtype = catalog_dtype_from_column_info_row(row)
        report_dtype = _logical_type_to_report_dtype(str(row.get('logical_type', '')))
        if not bucket_dtype or not report_dtype:
            continue
        try:
            column_index = int(row.get('column_index', -1))
        except (TypeError, ValueError):
            continue
        if column_index < 0:
            continue
        tsfile_path = tsfile_compression_report_path(bundle_dir, dataset_key, report_dtype)
        java_path = java_compression_report_path(bundle_dir, dataset_key, report_dtype)
        if os.path.isfile(tsfile_path):
            report_path = tsfile_path
            from_tsfile_staticdata = True
        else:
            report_path = java_path
            from_tsfile_staticdata = False
        ts_sum = min_tsfile_column_size_from_java_report(
            report_path,
            report_dtype,
            column_index,
            tsfile_staticdata_report=from_tsfile_staticdata,
        )
        if ts_sum > 0:
            tsfile_size_bytes[bucket_dtype] = (
                float(tsfile_size_bytes.get(bucket_dtype, 0.0)) + ts_sum
            )
