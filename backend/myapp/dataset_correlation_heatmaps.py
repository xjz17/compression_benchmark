"""Pearson correlation matrices for Dataset page algorithm/operator heatmaps."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .column_info_loader import load_column_info_rows
from .column_encodings import is_legacy_column_rle_algorithm
from .dataset_operators import (
    SUPPORTED_OPERATORS,
    ensure_operator_catalog_assets,
    operator_bundle_dir,
)
from .encoding_operator_catalog import load_method_page_operators
from .home_leaderboard_boxplot_data import boxplot_group_label
from .method_catalog import compression_families_for_algorithm_key
from .statistic import statistic_row_byte_entropy, statistic_row_value_entropy

logger = logging.getLogger(__name__)

CORRELATION_FEATURE_KEYS: tuple[str, ...] = (
    'mean',
    'standard_deviation',
    'spread',
    'byte_entropy',
    'value_entropy',
    'repeat',
    'sortedness',
)

CORRELATION_FEATURE_LABELS: dict[str, str] = {
    'mean': 'Mean',
    'standard_deviation': 'Standard Deviation',
    'spread': 'Spread',
    'byte_entropy': 'Byte Entropy',
    'value_entropy': 'Value Entropy',
    'repeat': 'Repeat',
    'sortedness': 'Sortedness',
    'compression_vs_byte_entropy': 'Compression ↔ Byte entropy Δ',
    'compression_vs_value_entropy': 'Compression ↔ Value entropy Δ',
}

OPERATOR_ENTROPY_COLUMN_SEGMENTS: tuple[dict[str, str], ...] = (
    {'key': 'int', 'label': 'Int columns'},
    {'key': 'fixed_float', 'label': 'Fixed-precision float columns'},
    {'key': 'non_fixed_float', 'label': 'Variable-precision float columns'},
)

_MIN_CORRELATION_POINTS = 3
_HIDDEN_METHOD_PAGE_OPERATOR_IDS = frozenset({'rd'})


def _finite_series(values: list[float | None]) -> np.ndarray | None:
    nums = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(nums) < _MIN_CORRELATION_POINTS:
        return None
    return np.asarray(nums, dtype=float)


def _zscore(values: list[float | None]) -> list[float | None]:
    arr = np.asarray(
        [np.nan if v is None else float(v) for v in values],
        dtype=float,
    )
    mask = np.isfinite(arr)
    if mask.sum() < _MIN_CORRELATION_POINTS:
        return [None] * len(values)
    mean = float(np.mean(arr[mask]))
    std = float(np.std(arr[mask]))
    if std <= 0:
        return [0.0 if np.isfinite(v) else None for v in arr]
    out = ((arr - mean) / std).tolist()
    return [float(v) if np.isfinite(v) else None for v in out]


def pearson_correlation(
    xs: list[float | None],
    ys: list[float | None],
) -> float | None:
    paired_x: list[float] = []
    paired_y: list[float] = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        paired_x.append(float(x))
        paired_y.append(float(y))
    x_arr = _finite_series(paired_x)
    y_arr = _finite_series(paired_y)
    if x_arr is None or y_arr is None or len(x_arr) != len(y_arr):
        return None
    if np.std(x_arr) <= 0 or np.std(y_arr) <= 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _statistic_value(row: dict, key: str) -> float | None:
    if key == 'byte_entropy':
        return statistic_row_byte_entropy(row)
    if key == 'value_entropy':
        return statistic_row_value_entropy(row)
    raw = row.get(key)
    if raw is None or raw == '':
        return None
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    return num if np.isfinite(num) else None


def _rows_by_column_index(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows:
        try:
            col_idx = int(row.get('column_index'))
        except (TypeError, ValueError):
            continue
        out[col_idx] = row
    return out


def _operator_display_name(operator_id: str) -> str:
    return _method_operator_name_map().get(operator_id, operator_id)


@lru_cache(maxsize=1)
def _method_operator_name_map() -> dict[str, str]:
    try:
        return {
            str(row.get('id', '') or ''): str(row.get('name', '') or '')
            for row in load_method_page_operators()
            if str(row.get('id', '') or '') and str(row.get('name', '') or '')
        }
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _method_operator_ids() -> tuple[str, ...]:
    supported = set(SUPPORTED_OPERATORS)
    try:
        catalog_rows = load_method_page_operators()
    except Exception:
        catalog_rows = ()
    out: list[str] = []
    seen: set[str] = set()
    for row in catalog_rows:
        operator_id = str(row.get('id', '') or '').strip()
        if operator_id in _HIDDEN_METHOD_PAGE_OPERATOR_IDS:
            continue
        if operator_id and operator_id in supported and operator_id not in seen:
            out.append(operator_id)
            seen.add(operator_id)
    return tuple(out)


def _all_catalog_operator_ids(bundle_dir: Path) -> list[str]:
    """Dataset entropy operators aligned with the Methods page operator set."""
    return list(_method_operator_ids())


def _operator_entropy_column_segments(
    *,
    column_info_path: str,
    bundle_dir: Path,
    column_indices: list[int],
) -> list[dict[str, Any]]:
    wanted = set(column_indices)
    index_to_segment: dict[int, str] = {}
    try:
        rows = load_column_info_rows(column_info_path)
    except Exception:
        rows = []
    for row in rows:
        try:
            col_idx = int(row.get('column_index'))
        except (TypeError, ValueError):
            continue
        if col_idx not in wanted:
            continue
        logical = str(row.get('logical_type', '') or '').strip().lower()
        if logical in ('integer', 'int', 'long'):
            index_to_segment[col_idx] = 'int'
        elif logical in ('float', 'double'):
            try:
                max_point = int(row.get('max_point_number', 0) or 0)
            except (TypeError, ValueError):
                max_point = 0
            index_to_segment[col_idx] = 'fixed_float' if max_point > 0 else 'non_fixed_float'

    out: list[dict[str, Any]] = []
    for segment in OPERATOR_ENTROPY_COLUMN_SEGMENTS:
        key = segment['key']
        positions = [
            pos
            for pos, col_idx in enumerate(column_indices)
            if index_to_segment.get(col_idx) == key
        ]
        out.append({
            **segment,
            'positions': positions,
            'column_count': len(positions),
        })
    return out


def _compression_rate_for_column(
    report_df: pd.DataFrame,
    column_index: int,
    algorithm: str,
    column_csv_size: float,
) -> float | None:
    if report_df is None or report_df.empty:
        return None
    if 'column_index' not in report_df.columns or 'algorithm' not in report_df.columns:
        return None
    if column_csv_size <= 0:
        return None
    cur = report_df.copy()
    cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
    cur = cur[cur['column_index'] == int(column_index)]
    cur = cur[cur['algorithm'].astype(str) == str(algorithm)]
    if cur.empty or 'compressedSize' not in cur.columns:
        return None
    try:
        compressed = float(cur['compressedSize'].sum())
    except (TypeError, ValueError):
        return None
    if compressed <= 0:
        return None
    return compressed / float(column_csv_size)


@lru_cache(maxsize=None)
def _cached_boxplot_group_label(algorithm: str) -> str:
    return boxplot_group_label(algorithm) or ''


@lru_cache(maxsize=None)
def _cached_compression_families(
    algorithm: str,
    catalog_keys: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        compression_families_for_algorithm_key(
            algorithm,
            catalog_keys=catalog_keys,
        )
    )


def _collect_algorithm_column_rates(
    *,
    column_indices: list[int],
    column_sizes: dict[int, float],
    report_cache: dict[str, pd.DataFrame],
) -> tuple[dict[str, list[float | None]], dict[str, set[str]]]:
    algo_keys: set[str] = set()
    for report_df in report_cache.values():
        if report_df is None or report_df.empty or 'algorithm' not in report_df.columns:
            continue
        for algo in report_df['algorithm'].dropna().astype(str):
            key = algo.strip()
            if key and not is_legacy_column_rle_algorithm(key):
                algo_keys.add(key)

    catalog_keys = tuple(sorted(algo_keys))
    algorithm_labels = {
        algo: _cached_boxplot_group_label(algo)
        for algo in algo_keys
    }
    label_algo_keys: dict[str, set[str]] = {}
    grouped: dict[str, dict[int, list[float]]] = {}
    wanted_columns = {int(idx) for idx in column_indices}
    for report_df in report_cache.values():
        if report_df is None or report_df.empty:
            continue
        needed = {'column_index', 'algorithm', 'compressedSize'}
        if not needed.issubset(report_df.columns):
            continue
        cur = report_df.loc[:, ['column_index', 'algorithm', 'compressedSize']].copy()
        cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
        cur['compressedSize'] = pd.to_numeric(cur['compressedSize'], errors='coerce')
        cur['algorithm'] = cur['algorithm'].astype(str).str.strip()
        cur = cur[
            cur['column_index'].notna()
            & cur['compressedSize'].notna()
            & cur['algorithm'].isin(algo_keys)
        ]
        if cur.empty:
            continue
        cur['column_index'] = cur['column_index'].astype(int)
        cur = cur[cur['column_index'].isin(wanted_columns)]
        if cur.empty:
            continue
        aggregated = cur.groupby(
            ['column_index', 'algorithm'],
            as_index=False,
        )['compressedSize'].sum()
        for row in aggregated.itertuples(index=False):
            col_idx = int(row.column_index)
            size = float(column_sizes.get(col_idx, 0.0) or 0.0)
            compressed = float(row.compressedSize)
            if size <= 0 or compressed <= 0:
                continue
            algo = str(row.algorithm)
            label = algorithm_labels.get(algo)
            if not label:
                continue
            label_algo_keys.setdefault(label, set()).add(algo)
            grouped.setdefault(label, {}).setdefault(col_idx, []).append(compressed / size)

    out: dict[str, list[float | None]] = {}
    label_families: dict[str, set[str]] = {}
    for label, per_col in grouped.items():
        series: list[float | None] = []
        for col_idx in column_indices:
            vals = per_col.get(col_idx) or []
            series.append(float(min(vals)) if vals else None)
        if any(v is not None for v in series):
            out[label] = series
            families: set[str] = set()
            for algo in label_algo_keys.get(label, set()):
                families.update(_cached_compression_families(algo, catalog_keys))
            label_families[label] = families
    return out, label_families


def _reference_compression_rates(
    *,
    column_indices: list[int],
    column_sizes: dict[int, float],
    report_cache: dict[str, pd.DataFrame],
) -> list[float | None]:
    rates: list[float | None] = []
    for col_idx in column_indices:
        size = column_sizes.get(col_idx, 0.0)
        candidates: list[float] = []
        for report_df in report_cache.values():
            if report_df is None or report_df.empty:
                continue
            for algo in ('TS_2DIFF', 'ts_2diff_bos'):
                rate = _compression_rate_for_column(report_df, col_idx, algo, size)
                if rate is not None:
                    candidates.append(rate)
            if not candidates and 'algorithm' in report_df.columns:
                for algo in report_df['algorithm'].dropna().astype(str):
                    rate = _compression_rate_for_column(
                        report_df, col_idx, algo.strip(), size,
                    )
                    if rate is not None:
                        candidates.append(rate)
        rates.append(float(min(candidates)) if candidates else None)
    return rates


def _entropy_change_series(
    *,
    column_indices: list[int],
    baseline_rows: dict[int, dict],
    operator_rows: dict[int, dict],
    field: str,
) -> list[float | None]:
    out: list[float | None] = []
    for col_idx in column_indices:
        base_row = baseline_rows.get(col_idx)
        op_row = operator_rows.get(col_idx)
        if not base_row or not op_row:
            out.append(None)
            continue
        if field == 'byte_entropy_change':
            base_val = statistic_row_byte_entropy(base_row)
            op_val = statistic_row_byte_entropy(op_row)
        else:
            base_val = statistic_row_value_entropy(base_row)
            op_val = statistic_row_value_entropy(op_row)
        if base_val is None or op_val is None:
            out.append(None)
        else:
            out.append(float(op_val) - float(base_val))
    return out


def _entropy_before_after_series(
    *,
    column_indices: list[int],
    baseline_rows: dict[int, dict],
    operator_rows: dict[int, dict],
    field: str,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    before: list[float | None] = []
    after: list[float | None] = []
    change: list[float | None] = []
    for col_idx in column_indices:
        base_row = baseline_rows.get(col_idx)
        op_row = operator_rows.get(col_idx)
        if not base_row or not op_row:
            before.append(None)
            after.append(None)
            change.append(None)
            continue
        if field == 'byte_entropy':
            base_val = statistic_row_byte_entropy(base_row)
            op_val = statistic_row_byte_entropy(op_row)
        else:
            base_val = statistic_row_value_entropy(base_row)
            op_val = statistic_row_value_entropy(op_row)
        if base_val is None or op_val is None:
            before.append(None)
            after.append(None)
            change.append(None)
        else:
            base_f = float(base_val)
            op_f = float(op_val)
            before.append(base_f)
            after.append(op_f)
            change.append(op_f - base_f)
    return before, after, change


def _build_heatmap_matrix(
    *,
    y_labels: list[str],
    x_keys: list[str],
    cell_fn,
) -> tuple[list[list[float | None]], list[str]]:
    x_labels = [CORRELATION_FEATURE_LABELS.get(key, key) for key in x_keys]
    matrix: list[list[float | None]] = []
    for y_label in y_labels:
        row_vals: list[float | None] = []
        for x_key in x_keys:
            row_vals.append(cell_fn(y_label, x_key))
        matrix.append(row_vals)
    return matrix, x_labels


def _build_algorithm_family_heatmap(
    *,
    algo_rates: dict[str, list[float | None]],
    algo_labels: list[str],
    feature_series: dict[str, list[float | None]],
    title: str,
    description: str,
) -> dict[str, Any]:
    def algo_cell(algo_label: str, feature_key: str) -> float | None:
        rates = algo_rates.get(algo_label)
        feats = feature_series.get(feature_key)
        if rates is None or feats is None:
            return None
        return pearson_correlation(rates, feats)

    matrix, x_labels = _build_heatmap_matrix(
        y_labels=algo_labels,
        x_keys=list(CORRELATION_FEATURE_KEYS),
        cell_fn=algo_cell,
    )
    return {
        'title': title,
        'description': description,
        'x_labels': x_labels,
        'y_labels': algo_labels,
        'values': matrix,
    }


def build_dataset_correlation_heatmaps(
    *,
    bundle_dir: Path,
    dataset_key: str,
    file_key: str,
    column_info_path: str,
    column_indices: list[int],
    column_sizes: dict[int, float],
    report_cache: dict[str, pd.DataFrame],
    baseline_stat_rows: list[dict],
    operator_ids: list[str],
) -> dict[str, Any]:
    baseline_by_col = _rows_by_column_index(baseline_stat_rows)
    feature_series: dict[str, list[float | None]] = {}
    for key in CORRELATION_FEATURE_KEYS:
        raw = [
            _statistic_value(baseline_by_col.get(col_idx, {}), key)
            for col_idx in column_indices
        ]
        feature_series[key] = _zscore(raw)

    algo_rates, label_families = _collect_algorithm_column_rates(
        column_indices=column_indices,
        column_sizes=column_sizes,
        report_cache=report_cache,
    )
    algo_labels = sorted(algo_rates.keys(), key=lambda s: s.casefold())
    value_based_labels = sorted(
        (
            label
            for label in algo_labels
            if 'value_based' in label_families.get(label, set())
        ),
        key=lambda s: s.casefold(),
    )
    byte_based_labels = sorted(
        (
            label
            for label in algo_labels
            if 'byte_based' in label_families.get(label, set())
        ),
        key=lambda s: s.casefold(),
    )

    algorithm_heatmap = _build_algorithm_family_heatmap(
        algo_rates=algo_rates,
        algo_labels=algo_labels,
        feature_series=feature_series,
        title='Compression algorithms vs column features',
        description=(
            'Pearson r between per-column compression rate and z-score normalized '
            'column statistics for every benchmarked compression algorithm.'
        ),
    )
    value_based_algorithm_heatmap = _build_algorithm_family_heatmap(
        algo_rates=algo_rates,
        algo_labels=value_based_labels,
        feature_series=feature_series,
        title='Value-based compression vs column features',
        description=(
            'Pearson r between per-column compression rate and z-score normalized column '
            'statistics for value-based compression methods (column encoding operators).'
        ),
    )
    byte_based_algorithm_heatmap = _build_algorithm_family_heatmap(
        algo_rates=algo_rates,
        algo_labels=byte_based_labels,
        feature_series=feature_series,
        title='Byte-based compression vs column features',
        description=(
            'Pearson r between per-column compression rate and z-score normalized column '
            'statistics for byte-based compression methods (byte codec operators).'
        ),
    )

    operator_labels = [_operator_display_name(op_id) for op_id in operator_ids]

    operator_entropy: dict[str, dict[str, list[float | None]]] = {}
    for op_id in operator_ids:
        op_csv = operator_bundle_dir(bundle_dir, op_id) / f'{file_key}_numeric_statistic.csv'
        if not op_csv.is_file():
            continue
        try:
            op_df = pd.read_csv(op_csv)
        except Exception:
            continue
        op_by_col = _rows_by_column_index(op_df.to_dict(orient='records'))
        byte_before, byte_after, byte_change = _entropy_before_after_series(
            column_indices=column_indices,
            baseline_rows=baseline_by_col,
            operator_rows=op_by_col,
            field='byte_entropy',
        )
        value_before, value_after, value_change = _entropy_before_after_series(
            column_indices=column_indices,
            baseline_rows=baseline_by_col,
            operator_rows=op_by_col,
            field='value_entropy',
        )
        operator_entropy[op_id] = {
            'byte_entropy_before': byte_before,
            'byte_entropy_after': byte_after,
            'byte_entropy_change': byte_change,
            'value_entropy_before': value_before,
            'value_entropy_after': value_after,
            'value_entropy_change': value_change,
        }

    def operator_cell(operator_label: str, feature_key: str) -> float | None:
        op_id = operator_ids[operator_labels.index(operator_label)]
        ent_data = operator_entropy.get(op_id)
        if not ent_data:
            return None
        feats = feature_series.get(feature_key)
        if feats is None:
            return None
        return pearson_correlation(feats, ent_data['byte_entropy_change'])

    operator_matrix, operator_x_labels = _build_heatmap_matrix(
        y_labels=operator_labels,
        x_keys=list(CORRELATION_FEATURE_KEYS),
        cell_fn=operator_cell,
    )

    return {
        'column_count': len(column_indices),
        'algorithm_heatmap': algorithm_heatmap,
        'value_based_algorithm_heatmap': value_based_algorithm_heatmap,
        'byte_based_algorithm_heatmap': byte_based_algorithm_heatmap,
        'operator_heatmap': {
            'title': 'Operator entropy change vs column features',
            'description': (
                'Pearson r across columns between z-score normalized column statistics '
                'and per-column byte entropy change after each encoding operator.'
            ),
            'x_labels': operator_x_labels,
            'y_labels': operator_labels,
            'values': operator_matrix,
        },
        'operator_entropy_change_boxplots': _build_operator_entropy_change_boxplots(
            operator_ids=operator_ids,
            operator_labels=operator_labels,
            operator_entropy=operator_entropy,
            column_segments=_operator_entropy_column_segments(
                column_info_path=column_info_path,
                bundle_dir=bundle_dir,
                column_indices=column_indices,
            ),
        ),
    }


def _finite_values_at_positions(
    values: list[float | None],
    positions: list[int] | None = None,
) -> list[float]:
    if positions is None:
        iterable = values
    else:
        iterable = [values[pos] for pos in positions if 0 <= pos < len(values)]
    return [
        float(v)
        for v in iterable
        if v is not None and np.isfinite(v)
    ]


def _build_operator_entropy_chart(
    *,
    field: str,
    title: str,
    y_label: str,
    description: str,
    operator_ids: list[str],
    operator_labels: list[str],
    operator_entropy: dict[str, dict[str, list[float | None]]],
    positions: list[int] | None = None,
) -> dict[str, Any]:
    algorithms: list[str] = []
    values: list[list[float]] = []
    before_medians: list[float] = []
    after_medians: list[float] = []
    before_key = field.replace('_change', '_before')
    after_key = field.replace('_change', '_after')
    for op_id, label in zip(operator_ids, operator_labels):
        ent_data = operator_entropy.get(op_id) or {}
        series = ent_data.get(field)
        if not series:
            continue
        pts = _finite_values_at_positions(series, positions)
        if not pts:
            continue
        before_pts = _finite_values_at_positions(ent_data.get(before_key, []), positions)
        after_pts = _finite_values_at_positions(ent_data.get(after_key, []), positions)
        algorithms.append(label)
        values.append(pts)
        before_medians.append(float(np.median(before_pts)) if before_pts else float('nan'))
        after_medians.append(float(np.median(after_pts)) if after_pts else float('nan'))
    order = sorted(
        range(len(algorithms)),
        key=lambda idx: (
            float(np.median(values[idx])) if values[idx] else float('inf'),
            algorithms[idx],
        ),
    )
    algorithms = [algorithms[idx] for idx in order]
    values = [values[idx] for idx in order]
    before_medians = [before_medians[idx] for idx in order]
    after_medians = [after_medians[idx] for idx in order]
    if positions is not None:
        finite_before = [value for value in before_medians if np.isfinite(value)]
        if finite_before:
            common_before = float(np.median(finite_before))
            before_medians = [common_before for _ in before_medians]
    return {
        'title': title,
        'description': description,
        'algorithms': algorithms,
        'values': values,
        'before_medians': before_medians,
        'after_medians': after_medians,
        'y_label': y_label,
        'log_y': False,
    }


def _build_operator_entropy_change_boxplots(
    *,
    operator_ids: list[str],
    operator_labels: list[str],
    operator_entropy: dict[str, dict[str, list[float | None]]],
    column_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-operator distributions of entropy change across benchmarked columns."""
    charts: dict[str, Any] = {}
    for field, title, y_label, description in (
        (
            'byte_entropy_change',
            'Byte entropy change by operator',
            'Byte entropy change (bits)',
            (
                'Distribution of per-column byte entropy change '
                '(transformed minus raw) for each encoding operator.'
            ),
        ),
        (
            'value_entropy_change',
            'Value entropy change by operator',
            'Value entropy change (bits)',
            (
                'Distribution of per-column value entropy change '
                '(transformed minus raw) for each encoding operator.'
            ),
        ),
    ):
        chart = _build_operator_entropy_chart(
            field=field,
            title=title,
            y_label=y_label,
            description=description,
            operator_ids=operator_ids,
            operator_labels=operator_labels,
            operator_entropy=operator_entropy,
        )
        if not chart['algorithms']:
            continue
        segments: list[dict[str, Any]] = []
        for segment in column_segments or []:
            segment_chart = _build_operator_entropy_chart(
                field=field,
                title=f'{title} ({segment["label"]})',
                y_label=y_label,
                description=description,
                operator_ids=operator_ids,
                operator_labels=operator_labels,
                operator_entropy=operator_entropy,
                positions=segment.get('positions') or [],
            )
            segments.append({
                **segment_chart,
                'segment_key': segment['key'],
                'segment_label': segment['label'],
                'column_count': segment['column_count'],
            })
        if segments:
            chart['segments'] = segments
        charts[field] = chart
    return charts


def load_correlation_heatmap_context(
    *,
    bundle_dir: Path,
    dataset_key: str,
    column_info_path: str,
    filtered_column_indices: list[int] | None = None,
    file_key: str | None = None,
    master_csv: str | None = None,
    materialize_missing_operator_assets: bool = True,
) -> dict[str, Any] | None:
    storage_key = str(file_key or dataset_key).strip()
    column_indices = _load_benchmark_column_indices(
        column_info_path,
        allowed=set(filtered_column_indices) if filtered_column_indices else None,
    )
    if not column_indices:
        return None

    column_sizes = _load_column_sizes(column_info_path, column_indices)
    report_cache: dict[str, pd.DataFrame] = {}
    for dtype in ('int', 'float'):
        report_path = bundle_dir / f'{storage_key}_{dtype}_java_compression_result.csv'
        if report_path.is_file():
            try:
                report_cache[dtype] = pd.read_csv(report_path)
            except Exception:
                report_cache[dtype] = pd.DataFrame()

    baseline_csv = bundle_dir / f'{storage_key}_numeric_statistic.csv'
    if not baseline_csv.is_file():
        return None
    try:
        baseline_df = pd.read_csv(baseline_csv)
    except Exception:
        return None

    operator_ids = _all_catalog_operator_ids(bundle_dir)
    if master_csv and materialize_missing_operator_assets:
        for operator_id in operator_ids:
            try:
                ensure_operator_catalog_assets(
                    storage_key,
                    master_csv,
                    bundle_dir,
                    operator_id,
                )
            except Exception:
                logger.exception(
                    'Failed to generate missing operator statistics: dataset=%s operator=%s',
                    dataset_key,
                    operator_id,
                )
    return build_dataset_correlation_heatmaps(
        bundle_dir=bundle_dir,
        dataset_key=dataset_key,
        file_key=storage_key,
        column_info_path=column_info_path,
        column_indices=column_indices,
        column_sizes=column_sizes,
        report_cache=report_cache,
        baseline_stat_rows=baseline_df.to_dict(orient='records'),
        operator_ids=operator_ids,
    )


def _logical_type_to_report_dtype(logical_type: object) -> str | None:
  """Map column logical type to int/float Java report dtype."""
  logical = str(logical_type or '').strip().lower()
  if logical in ('integer', 'int', 'long'):
    return 'int'
  if logical in ('float', 'double'):
    return 'float'
  return None


def _load_benchmark_column_indices(
    column_info_path: str,
    *,
    allowed: set[int] | None,
) -> list[int]:
    indices: list[int] = []
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return indices
    for row in info_rows:
        if _logical_type_to_report_dtype(row.get('logical_type', '')) is None:
            continue
        try:
            col_idx = int(row.get('column_index'))
            size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if size <= 0:
            continue
        if allowed is not None and col_idx not in allowed:
            continue
        indices.append(col_idx)
    return sorted(set(indices))


def _load_column_sizes(column_info_path: str, column_indices: list[int]) -> dict[int, float]:
    sizes: dict[int, float] = {}
    try:
        info_rows = load_column_info_rows(column_info_path)
    except Exception:
        return sizes
    wanted = set(column_indices)
    for row in info_rows:
        try:
            col_idx = int(row.get('column_index'))
            size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if col_idx in wanted and size > 0:
            sizes[col_idx] = size
    return sizes
