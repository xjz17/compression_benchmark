"""Aggregate float Java benchmark metrics for Home leaderboard boxplots and Pareto chart."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable
from math import sqrt
from pathlib import Path

import pandas as pd

from .active_method_policy import is_active_algorithm_key, is_active_method_name
from .column_encodings import is_legacy_column_rle_algorithm
from .column_info_loader import load_column_info_rows
from .dataset_bundle import resolve_master_csv_path
from .lzbench_codec_catalog import (
    LZBENCH_CODEC_ENTRIES,
    canonical_report_algorithm_key,
    is_lzbench_catalog_algorithm,
    parse_lzbench_algorithm_key,
)
from .report_filters import filter_compression_report_dataframe, sanitize_compression_report_dataframe
from .report_filters import is_hidden_report_algorithm
from .dataset_float_precision_distribution import (
    FLOAT_PRECISION_FILTER_ALL,
    allowed_float_column_indexes_for_filter,
    filter_report_dataframe_for_float_precision,
    parse_float_precision_filter,
)

MIN_RATE_FOR_BOX_PLOT: float | None = 0.00001
BALANCED_SCORE_COMPRESSION_TIME_WEIGHT = 0.30
BALANCED_SCORE_DECOMPRESSION_TIME_WEIGHT = 0.30
BALANCED_SCORE_COMPRESSION_RATE_WEIGHT = 0.40

_REPORT_AGGREGATE_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}

_LOSSY_BOX_PLOT_LABELS = frozenset({
    'MOST',
    'Machete',
    'Sim-Piece',
    'SZ',
    'SZ3',
    'ZFP',
})

_HOME_HIDDEN_ALGORITHM_KEYS = frozenset(
    {
        *(f'ts_2diff_bos_l{level}' for level in range(6)),
        *(f'subcolumn_l{level}' for level in range(6)),
        *(f'oss_l{level}' for level in range(6)),
    },
)

_HOME_HIDDEN_ALGORITHM_LABELS = frozenset()

# String/byte benchmark paths are disabled for now; keep Home overall numeric-only.
_HOME_BENCHMARK_DATA_TYPES = ('int', 'float')

_PARAMETER_VARIANT_LABELS = {
    'lzo1': 'LZO1',
    'lzo1a': 'LZO1A',
    'lzo1b': 'LZO1B',
    'lzo1c': 'LZO1C',
    'lzo1f': 'LZO1F',
    'lzo1x_999': 'LZO1X',
    'lzo1y': 'LZO1Y',
    'lzo1z': 'LZO1Z',
    'slz_deflate_3': 'deflate',
    'slz_gzip_3': 'gzip',
    'slz_zlib_3': 'zlib',
    'ucl_nrv2b_9': 'NRV2b',
    'ucl_nrv2d': 'NRV2d',
    'ucl_nrv2e': 'NRV2e',
}

_ALGORITHM_KEY_LABEL: dict[str, str] = {
    'brotli': 'Brotli',
    'gz': 'GZIP',
    'gzip': 'GZIP',
    'lz4-framed': 'LZ4',
    'lz4': 'LZ4',
    'snappy-framed': 'SNAPPY',
    'snappy': 'SNAPPY',
    'fpzip': 'FPZIP',
    'blosc_blosclz': 'Blosc',
    'blosc2_blosclz': 'Blosc2',
    'density': 'Density',
    'gipfeli': 'Gipfeli',
    'glza': 'GLZA',
    'lzmat': 'LZMAT',
    'lzo1': 'LZO1',
    'lzo1a': 'LZO1',
    'lzo1b': 'LZO1',
    'lzo1c': 'LZO1',
    'lzo1f': 'LZO1',
    'lzo1y': 'LZO1',
    'lzo1z': 'LZO1',
    'lzo2a': 'LZO2A',
    'lzrw': 'LZRW',
    'lzsse2': 'LZSSE',
    'lzsse4': 'LZSSE',
    'lzsse4fast': 'LZSSE',
    'lzsse8': 'LZSSE',
    'lzsse8fast': 'LZSSE',
    'lzvn': 'LZVN',
    'pithy': 'Pithy',
    'shrinker': 'Shrinker',
    'slz_deflate': 'SLZ',
    'slz_gzip': 'SLZ',
    'slz_zlib': 'SLZ',
    'ucl_nrv2d': 'UCL NRV',
    'ucl_nrv2e': 'UCL NRV',
    'wflz': 'WFLZ',
    'xpack': 'XPACK',
    'yalz77': 'YALZ77',
    'yappy': 'Yappy',
    'ext_zfp_zfpy': 'ZFP',
    'fpcompress_dpratio': 'FPCompress DPratio',
    'fpcompress_dpspeed': 'FPCompress DPspeed',
    'ans': 'ANS',
    'fse': 'FSE',
    'huffman': 'Huffman',
    'huff0': 'Huff0',
    'arithmetic': 'Arithmetic',
    'CHIMP': 'CHIMP',
    'CHIMP128': 'Chimp128',
    'GORILLA': 'GORILLA',
    'RLE': 'RLE',
    'RLBE': 'RLBE',
    'SPRINTZ': 'SPRINTZ',
    'TS_2DIFF': 'TS_2DIFF',
    'ts_2diff_bos': 'BOS',
    'subcolumn': 'Sub-column',
    'reger': 'REGER',
    'col_dictionary': 'Dictionary',
    'col_bss_lz4': 'BSS+LZ4',
    'DELTA_BINARY_PACKED': 'DBP',
    'streamvbyte_delta1': 'Stream VByte',
    'fastpfor': 'FastPFor',
    'fpc': 'FPC',
    'pfpc': 'PFPC',
    'simple8b': 'Simple8b',
    'corad_tricklet': 'CORAD',
    'leco': 'LECO',
    'slz_deflate_3': 'SLZ',
    'slz_gzip_3': 'SLZ',
    'slz_zlib_3': 'SLZ',
    'alp': 'ALP',
    'buff_kernel_native': 'BUFF',
    'buff_scalar': 'BUFF',
    'buff_simd': 'BUFF',
    'buff_float_kernel_native': 'BUFF',
    'buff_float_scalar': 'BUFF',
    'buff_float_simd': 'BUFF',
    'most_minimal': 'MOST',
    'most_float_minimal': 'MOST',
    'elf_on_chimp': 'Elf',
    'elf_on_chimp_float': 'Elf',
    'elf_plus': 'Elf+',
    'elf_plus_float': 'Elf+',
    'camel': 'Camel',
    'camel_float': 'Camel',
    'chimp32': 'Chimp32',
    'chimp_int': 'Chimp',
    'chimp_float': 'Chimp',
}


def _lzbench_group_label(algorithm_key: str) -> str | None:
    key = str(algorithm_key or '').strip()
    if not key or not is_lzbench_catalog_algorithm(key):
        return None
    if key.startswith('lzbench_'):
        key = key[len('lzbench_'):]
    for entry in LZBENCH_CODEC_ENTRIES:
        if entry.presets is not None:
            presets = entry.presets
            if key in presets:
                return entry.display_name
            continue
        safe_id = entry.codec_id.replace('-', '_')
        match = re.fullmatch(re.escape(safe_id) + r'_(\d+)', key)
        if match:
            return entry.display_name
    return None


def boxplot_group_label(algorithm_key: str) -> str | None:
    """Map report ``algorithm`` key to a grouped display label for Home charts."""
    key = canonical_report_algorithm_key(algorithm_key)
    if not key:
        return None
    if is_hidden_report_algorithm(key):
        return None
    if is_legacy_column_rle_algorithm(key):
        return None
    if key in _ALGORITHM_KEY_LABEL:
        return _ALGORITHM_KEY_LABEL[key]
    if key.startswith('xz_level_') or key == 'xz':
        return 'XZ'
    if key.startswith('zstd_level_') or key == 'zstd':
        return 'ZSTD'
    if key.startswith('spdp_level_') or key.startswith('spdp_float_level_'):
        return 'SPDP'
    if key.startswith('machete_rel_') or key.startswith('machete_float_rel_'):
        return 'Machete'
    if key.startswith('simpiece_'):
        return 'Sim-Piece'
    if key.startswith('sz3_'):
        return 'SZ3'
    if key.startswith('sz_'):
        return 'SZ'
    if key.startswith('ext_zfp_tol_') or key == 'ext_zfp_zfpy':
        return 'ZFP'
    if key.startswith('buff_'):
        return 'BUFF'
    lz_label = _lzbench_group_label(key)
    if lz_label:
        return lz_label
    if key.startswith('serf_'):
        return 'Serf'
    return key


def is_home_leaderboard_algorithm_enabled(algorithm_key: str) -> bool:
    """Return whether a raw algorithm key participates in Home comparisons."""
    key = canonical_report_algorithm_key(algorithm_key)
    return (
        bool(key)
        and is_active_algorithm_key(key)
        and key not in _HOME_HIDDEN_ALGORITHM_KEYS
        and not is_hidden_report_algorithm(key)
    )


def is_home_leaderboard_label_enabled(label: str) -> bool:
    value = str(label or '').strip()
    return is_active_method_name(value) and value not in _HOME_HIDDEN_ALGORITHM_LABELS


def is_lossy_boxplot_label(label: str) -> bool:
    return str(label or '').strip() in _LOSSY_BOX_PLOT_LABELS


def clear_java_float_report_cache() -> None:
    """Drop cached aggregated float Java reports (e.g. before a batch recompute)."""
    _REPORT_AGGREGATE_CACHE.clear()


def iter_catalog_benchmark_datasets(backend: Path) -> list[tuple[str, Path]]:
    info_root = backend / 'dataset'
    if not info_root.is_dir():
        return []
    pairs: list[tuple[str, Path]] = []
    for child in sorted(info_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        slug = child.name
        col_info = child / f'{slug}_column_info.csv'
        if not col_info.is_file():
            continue
        master = resolve_master_csv_path(slug)
        if master is None:
            reports = sorted(child.glob(f'{slug}_*_compression_result.csv'))
            if not reports:
                continue
            pairs.append((slug, reports[0]))
        else:
            pairs.append((slug, Path(master)))
    return pairs


def read_float_file_size(bundle_dir: Path, slug: str) -> int | None:
    return read_benchmark_file_size(bundle_dir, slug, data_type='float')


def _logical_type_matches_data_type(logical_type: object, data_type: str) -> bool:
    logical = str(logical_type or '').strip().lower()
    if data_type == 'overall':
        return logical in ('integer', 'int', 'long', 'float', 'double')
    if data_type == 'int':
        return logical in ('integer', 'int', 'long')
    if data_type == 'float':
        return logical in ('float', 'double')
    return False


def _home_report_data_types(data_type: object) -> tuple[str, ...]:
    dtype = str(data_type or 'float').strip().lower()
    if dtype == 'overall':
        return _HOME_BENCHMARK_DATA_TYPES
    if dtype in _HOME_BENCHMARK_DATA_TYPES:
        return (dtype,)
    return ('float',)


def read_benchmark_file_size(
    bundle_dir: Path,
    slug: str,
    *,
    data_type: str = 'float',
    selected_column_index: int | None = None,
    float_precision: str = FLOAT_PRECISION_FILTER_ALL,
) -> int | None:
    column_info_path = bundle_dir / f'{slug}_column_info.csv'
    if not column_info_path.is_file():
        return None
    try:
        rows = load_column_info_rows(str(column_info_path))
    except Exception:
        return None
    allowed_float_columns: set[int] | None = None
    if (
        selected_column_index is None
        and data_type == 'float'
        and parse_float_precision_filter(float_precision) != FLOAT_PRECISION_FILTER_ALL
    ):
        allowed_float_columns = allowed_float_column_indexes_for_filter(
            str(column_info_path),
            str(bundle_dir),
            float_precision,
        )
        if not allowed_float_columns:
            return None
    total = 0
    for row in rows:
        logical = row.get('logical_type', '')
        if not _logical_type_matches_data_type(logical, data_type):
            continue
        row_idx = None
        if selected_column_index is not None or allowed_float_columns is not None:
            try:
                row_idx = int(row.get('column_index', -1))
            except (TypeError, ValueError):
                continue
        if selected_column_index is not None:
            if row_idx != int(selected_column_index):
                continue
        if allowed_float_columns is not None:
            if row_idx not in allowed_float_columns:
                continue
        try:
            total += int(float(row.get('column_csv_size_bytes', 0) or 0))
        except (TypeError, ValueError):
            continue
    return total if total > 0 else None


def aggregate_report_rows_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'algorithm' not in df.columns:
        return df.iloc[0:0].copy()
    cols = ['algorithm', 'originalSize', 'compressedSize', 'compressionTime', 'decompressionTime']
    work = df[cols].copy()
    work['algorithm'] = work['algorithm'].astype(str)
    for col in cols[1:]:
        work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0.0)
    return work.groupby('algorithm', as_index=False)[cols[1:]].sum()


def _read_raw_benchmark_report(report_path: Path) -> pd.DataFrame | None:
    if not report_path.is_file():
        return None
    cache_key = str(report_path.resolve())
    try:
        mtime = report_path.stat().st_mtime
    except OSError:
        mtime = None
    if mtime is not None:
        cached = _REPORT_AGGREGATE_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1].copy()
    needed = (
        'column_index',
        'algorithm',
        'originalSize',
        'compressedSize',
        'compressionTime',
        'decompressionTime',
    )
    try:
        df = pd.read_csv(report_path, usecols=list(needed))
    except Exception:
        return None
    if not set(needed).issubset(df.columns):
        return None
    df = sanitize_compression_report_dataframe(df)
    if df.empty:
        return None
    if mtime is not None:
        _REPORT_AGGREGATE_CACHE[cache_key] = (mtime, df)
    return df.copy()


def read_java_float_report(bundle_dir: Path, slug: str) -> pd.DataFrame | None:
    return read_benchmark_report(bundle_dir, slug, data_type='float')


def _read_benchmark_report_for_data_type(
    bundle_dir: Path,
    slug: str,
    data_type: str,
    implement_language: str,
) -> pd.DataFrame | None:
    return _read_raw_benchmark_report(bundle_dir / f'{slug}_{data_type}_{implement_language}_compression_result.csv')


def read_benchmark_report(
    bundle_dir: Path,
    slug: str,
    *,
    data_type: str = 'float',
    implement_language: str = 'java',
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    language = str(implement_language or 'java').strip().lower() or 'java'
    for report_data_type in _home_report_data_types(data_type):
        report = _read_benchmark_report_for_data_type(bundle_dir, slug, report_data_type, language)
        if report is not None and not report.empty:
            frames.append(report)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def pareto_level_sort_key(algorithm_key: str) -> float | None:
    """Numeric order for connecting multi-level algorithm points; None if not leveled."""
    key = canonical_report_algorithm_key(algorithm_key) or str(algorithm_key or '').strip()
    if not key:
        return None
    parsed = parse_lzbench_algorithm_key(key)
    if parsed is not None:
        level = parsed['level']
        if isinstance(level, str):
            presets = parsed['entry'].presets or ()
            try:
                return float(presets.index(level))
            except ValueError:
                return None
        return float(level)
    for prefix in ('zstd_level_', 'xz_level_', 'spdp_level_', 'spdp_float_level_'):
        if key.startswith(prefix):
            try:
                return float(key[len(prefix):])
            except ValueError:
                return None
    return None


def pareto_point_label(algorithm_key: str) -> str:
    """Human-readable Pareto chart label including level when applicable."""
    key = canonical_report_algorithm_key(algorithm_key) or str(algorithm_key or '').strip()
    family = boxplot_group_label(key)
    parsed = parse_lzbench_algorithm_key(key)
    if parsed is not None:
        entry = parsed['entry']
        level = parsed['level']
        if entry.presets is not None:
            return f'{entry.display_name} ({_PARAMETER_VARIANT_LABELS.get(str(level), level)})'
        return f'{entry.display_name} (level {level})'
    if key.startswith('zstd_level_'):
        return f'ZSTD (level {key.rsplit("_", 1)[-1]})'
    if key.startswith('xz_level_'):
        return f'XZ (level {key.rsplit("_", 1)[-1]})'
    if key.startswith('spdp_float_level_'):
        return f'SPDP (level {key.rsplit("_", 1)[-1]})'
    if key.startswith('spdp_level_'):
        return f'SPDP (level {key.rsplit("_", 1)[-1]})'
    return family or key


def best_metrics_from_report(
    report: pd.DataFrame,
    *,
    file_size: float,
    implement_language: str = 'java',
    use_report_original_size_for_rate: bool = False,
    report_is_sanitized: bool = False,
    selected_column_index: int | None = None,
    **filter_params,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]]]:
    """Best-per-family and per-algorithm metrics after applying current UI parameters."""
    if report.empty or (file_size <= 0 and not use_report_original_size_for_rate):
        return {}, {}

    work = filter_compression_report_dataframe(
        report,
        implement_language=implement_language,
        sanitize_input=not report_is_sanitized,
        **filter_params,
    )
    if work.empty:
        return {}, {}
    if selected_column_index is not None:
        if 'column_index' not in work.columns:
            return {}, {}
        work = work[pd.to_numeric(work['column_index'], errors='coerce') == int(selected_column_index)].copy()
        if work.empty:
            return {}, {}

    work = work[work['algorithm'].apply(is_home_leaderboard_algorithm_enabled)].copy()
    if work.empty:
        return {}, {}

    work['label'] = work['algorithm'].apply(boxplot_group_label)
    work = work[work['label'].notna()]
    if work.empty:
        return {}, {}

    for col in ('originalSize', 'compressedSize', 'compressionTime', 'decompressionTime'):
        work[col] = pd.to_numeric(work[col], errors='coerce')
    work = work[(work['compressedSize'] > 0) & (work['originalSize'] > 0)]
    if work.empty:
        return {}, {}

    work = aggregate_report_rows_by_algorithm(work)
    work['label'] = work['algorithm'].apply(boxplot_group_label)
    work = work[work['label'].notna()]
    if work.empty:
        return {}, {}

    num_points = work['originalSize'] / 8.0
    work = work[num_points > 0]
    if work.empty:
        return {}, {}

    if use_report_original_size_for_rate:
        work = work[work['originalSize'] > 0].copy()
        if work.empty:
            return {}, {}
        work['rate'] = work['compressedSize'] / work['originalSize']
    else:
        work['rate'] = work['compressedSize'] / float(file_size)
    work['comp_ns'] = work['compressionTime'] / num_points
    work['decomp_ns'] = work['decompressionTime'] / num_points

    per_algo: dict[str, tuple[float, float, float]] = {}
    for row in work.itertuples(index=False):
        algo = str(row.algorithm)
        per_algo[algo] = (float(row.rate), float(row.comp_ns), float(row.decomp_ns))

    best_rows = work.loc[work.groupby('label', sort=False)['rate'].idxmin()]
    best: dict[str, tuple[float, float, float]] = {}
    for row in best_rows.itertuples(index=False):
        best[str(row.label)] = (float(row.rate), float(row.comp_ns), float(row.decomp_ns))
    return best, per_algo


def metrics_for_dataset(
    slug: str,
    bundle_dir: Path,
    *,
    implement_language: str = 'java',
    data_type: str = 'float',
    filter_params: dict | None = None,
    selected_column_index: int | None = None,
    float_precision: str = FLOAT_PRECISION_FILTER_ALL,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]]]:
    """Best-per-family and per-algorithm-key metrics for one catalog dataset."""
    normalized_float_precision = parse_float_precision_filter(float_precision)
    file_size = read_benchmark_file_size(
        bundle_dir,
        slug,
        data_type=data_type,
        selected_column_index=selected_column_index,
        float_precision=normalized_float_precision,
    )
    report = read_benchmark_report(
        bundle_dir,
        slug,
        data_type=data_type,
        implement_language=implement_language,
    )
    use_report_original_size = str(data_type or '').strip().lower() == 'overall'
    if report is None or report.empty:
        return {}, {}
    if file_size is None and not use_report_original_size:
        return {}, {}
    if (
        selected_column_index is None
        and str(data_type or '').strip().lower() == 'float'
        and normalized_float_precision != FLOAT_PRECISION_FILTER_ALL
    ):
        report = filter_report_dataframe_for_float_precision(
            report,
            column_info_path=str(bundle_dir / f'{slug}_column_info.csv'),
            bundle_dir=str(bundle_dir),
            float_precision=normalized_float_precision,
        )
        if report.empty:
            return {}, {}
    return best_metrics_from_report(
        report,
        file_size=float(file_size or 0),
        implement_language=implement_language,
        use_report_original_size_for_rate=use_report_original_size,
        report_is_sanitized=True,
        selected_column_index=selected_column_index,
        **(filter_params or {}),
    )


def best_metrics_for_dataset(
    slug: str,
    bundle_dir: Path,
) -> dict[str, tuple[float, float, float]]:
    best, _per_algo = metrics_for_dataset(slug, bundle_dir)
    return best


def all_metrics_for_dataset(
    slug: str,
    bundle_dir: Path,
) -> dict[str, tuple[float, float, float]]:
    _best, per_algo = metrics_for_dataset(slug, bundle_dir)
    return per_algo


def collect_metrics(
    backend: Path,
    *,
    min_rate: float | None = MIN_RATE_FOR_BOX_PLOT,
    implement_language: str = 'java',
    data_type: str = 'float',
    filter_params: dict | None = None,
    dataset_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    selected_column_index: int | None = None,
    float_precision: str = FLOAT_PRECISION_FILTER_ALL,
    excluded_families: set[str] | frozenset[str] | None = None,
    on_dataset_progress: Callable[[str, int, int, float], None] | Callable[[str, int, int], None] | None = None,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    int,
]:
    rates: dict[str, list[float]] = defaultdict(list)
    comp_times: dict[str, list[float]] = defaultdict(list)
    decomp_times: dict[str, list[float]] = defaultdict(list)
    balanced_costs: dict[str, list[float]] = defaultdict(list)
    balanced_ranks: dict[str, list[float]] = defaultdict(list)
    pareto_rates: dict[str, list[float]] = defaultdict(list)
    pareto_comp_times: dict[str, list[float]] = defaultdict(list)
    datasets_used = 0
    pairs = iter_catalog_benchmark_datasets(backend)
    if dataset_keys is not None:
        allowed_dataset_keys = {str(key).strip() for key in dataset_keys if str(key).strip()}
        pairs = [(slug, path) for slug, path in pairs if slug in allowed_dataset_keys]
    total = len(pairs)
    excluded = {str(label).strip() for label in (excluded_families or set()) if str(label).strip()}
    for index, (slug, _) in enumerate(pairs, start=1):
        bundle_dir = backend / 'dataset' / slug
        iter_started = time.perf_counter()
        per_ds, per_algo = metrics_for_dataset(
            slug,
            bundle_dir,
            implement_language=implement_language,
            data_type=data_type,
            filter_params=filter_params,
            selected_column_index=selected_column_index,
            float_precision=float_precision,
        )
        if on_dataset_progress is not None:
            elapsed = time.perf_counter() - iter_started
            try:
                on_dataset_progress(slug, index, total, elapsed)
            except TypeError:
                on_dataset_progress(slug, index, total)
        if not per_ds:
            continue
        if excluded:
            per_ds = {label: values for label, values in per_ds.items() if label not in excluded}
            per_algo = {label: values for label, values in per_algo.items() if label not in excluded}
            if not per_ds:
                continue
        datasets_used += 1
        ds_for_balanced = {
            label: (rate, comp_ns, decomp_ns)
            for label, (rate, comp_ns, decomp_ns) in per_ds.items()
            if (min_rate is None or rate >= min_rate)
        }
        ds_costs, ds_ranks = compute_balanced_score_metrics([ds_for_balanced])
        for label, pts in ds_costs.items():
            balanced_costs[label].extend(pts)
        for label, pts in ds_ranks.items():
            balanced_ranks[label].extend(pts)
        for label, (rate, comp_ns, decomp_ns) in per_ds.items():
            if min_rate is not None and rate < min_rate:
                continue
            rates[label].append(rate)
            comp_times[label].append(comp_ns)
            decomp_times[label].append(decomp_ns)
        for algo_key, (rate, comp_ns, _decomp_ns) in per_algo.items():
            if min_rate is not None and rate < min_rate:
                continue
            pareto_rates[algo_key].append(rate)
            pareto_comp_times[algo_key].append(comp_ns)
    return (
        rates,
        comp_times,
        decomp_times,
        balanced_costs,
        balanced_ranks,
        pareto_rates,
        pareto_comp_times,
        datasets_used,
    )


def collect_pareto_metrics(
    backend: Path,
    *,
    min_rate: float | None = MIN_RATE_FOR_BOX_PLOT,
    data_type: str = 'float',
    implement_language: str = 'java',
    dataset_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    selected_column_index: int | None = None,
    float_precision: str = FLOAT_PRECISION_FILTER_ALL,
    excluded_families: set[str] | frozenset[str] | None = None,
    on_dataset_progress: Callable[[str, int, int, float], None] | Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]], int]:
    """Aggregate per-algorithm-key metrics across catalog datasets for the Pareto chart."""
    (
        _rates,
        _comp,
        _decomp,
        _balanced_costs,
        _balanced_ranks,
        pareto_rates,
        pareto_comp_times,
        datasets_used,
    ) = collect_metrics(
        backend,
        min_rate=min_rate,
        implement_language=implement_language,
        data_type=data_type,
        dataset_keys=dataset_keys,
        selected_column_index=selected_column_index,
        float_precision=float_precision,
        excluded_families=excluded_families,
        on_dataset_progress=on_dataset_progress,
    )
    return pareto_rates, pareto_comp_times, datasets_used


def sort_labels_by_mean_rate(rates: dict[str, list[float]]) -> list[str]:
    with_data = [name for name, vals in rates.items() if vals]
    return sorted(
        with_data,
        key=lambda name: sum(rates[name]) / len(rates[name]),
        reverse=True,
    )


def sort_labels_by_mean_value(values: dict[str, list[float]]) -> list[str]:
    with_data = [name for name, vals in values.items() if vals]
    return sorted(with_data, key=lambda name: sum(values[name]) / len(values[name]))


def compute_balanced_score_metrics(
    datasets: list[dict[str, tuple[float, float, float]]] | tuple[dict[str, tuple[float, float, float]], ...],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return weighted compression ratio/time cost and per-dataset rank.

    ``rate``, ``compression_time``, and ``decompression_time`` are cost metrics,
    so lower is better. Compression and decompression time are min-max
    normalized across methods available in each dataset. Compression ratio is
    used directly.
    """
    costs: dict[str, list[float]] = defaultdict(list)
    ranks: dict[str, list[float]] = defaultdict(list)
    for per_dataset in datasets:
        usable = {
            label: (float(rate), float(comp_ns), float(decomp_ns))
            for label, (rate, comp_ns, decomp_ns) in per_dataset.items()
            if rate > 0 and comp_ns >= 0 and decomp_ns >= 0
        }
        if len(usable) < 2:
            continue

        comp_range = nonnegative_value_range(comp_ns for _rate, comp_ns, _decomp_ns in usable.values())
        decomp_range = nonnegative_value_range(decomp_ns for _rate, _comp_ns, decomp_ns in usable.values())
        dataset_costs = {}
        for label, (rate, comp_ns, decomp_ns) in usable.items():
            score = weighted_balanced_score(rate, comp_ns, decomp_ns, comp_range, decomp_range)
            if score is not None:
                dataset_costs[label] = score
        if len(dataset_costs) < 2:
            continue
        for label, value in dataset_costs.items():
            costs[label].append(float(value))

        ordered = sorted(dataset_costs.items(), key=lambda item: (item[1], item[0]))
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            avg_rank = (index + 1 + end) / 2.0
            for label, _value in ordered[index:end]:
                ranks[label].append(float(avg_rank))
            index = end
    return costs, ranks


def nonnegative_value_range(values) -> tuple[float, float] | None:
    nums: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 0:
            nums.append(numeric)
    if not nums:
        return None
    return min(nums), max(nums)


def normalized_by_range(value: object, value_range: tuple[float, float] | None) -> float | None:
    if value_range is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    low, high = value_range
    if high <= low:
        return 0.0
    return (numeric - low) / (high - low)


def weighted_balanced_score(
    compression_rate: object,
    compression_time: object,
    decompression_time: object,
    compression_time_range: tuple[float, float] | None,
    decompression_time_range: tuple[float, float] | None,
) -> float | None:
    try:
        rate = float(compression_rate)
    except (TypeError, ValueError):
        return None
    if rate <= 0:
        return None
    comp_norm = normalized_by_range(compression_time, compression_time_range)
    decomp_norm = normalized_by_range(decompression_time, decompression_time_range)
    if comp_norm is None or decomp_norm is None:
        return None
    return (
        BALANCED_SCORE_COMPRESSION_TIME_WEIGHT * comp_norm
        + BALANCED_SCORE_DECOMPRESSION_TIME_WEIGHT * decomp_norm
        + BALANCED_SCORE_COMPRESSION_RATE_WEIGHT * rate
    )


def _chart_payload(
    values: dict[str, list[float]],
    algorithm_order: list[str],
    *,
    key: str,
    title: str,
    y_label: str,
    log_y: bool,
    chart_type: str = 'boxplot',
) -> dict[str, object]:
    algorithms: list[str] = []
    series_values: list[list[float]] = []
    for name in algorithm_order:
        pts = values.get(name, [])
        if log_y:
            pts = [v for v in pts if v > 0]
        if not pts:
            continue
        algorithms.append(name)
        series_values.append([float(v) for v in pts])
    return {
        'key': key,
        'title': title,
        'y_label': y_label,
        'log_y': log_y,
        'chart_type': chart_type,
        'algorithms': algorithms,
        'values': series_values,
    }


def _mean(values: dict[str, list[float]], label: str) -> float | None:
    pts = values.get(label) or []
    if not pts:
        return None
    return float(sum(pts) / len(pts))


def _mean_value_samples(values: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        label: [float(sum(pts) / len(pts))]
        for label, pts in values.items()
        if pts
    }


def _critical_difference_threshold(algorithm_count: int, sample_count: int) -> float:
    if algorithm_count < 2 or sample_count <= 0:
        return 0.0
    nemenyi_q_alpha_05 = 3.314
    return float(nemenyi_q_alpha_05 * sqrt(algorithm_count * (algorithm_count + 1) / (6.0 * sample_count)))


def _critical_difference_groups(
    algorithms: list[str],
    average_ranks: list[float],
    critical_difference: float,
) -> list[dict[str, object]]:
    if critical_difference <= 0 or len(algorithms) < 2:
        return []
    groups: list[dict[str, object]] = []
    index = 0
    lane = 0
    while index < len(algorithms):
        end = index
        while (
            end + 1 < len(algorithms)
            and average_ranks[end + 1] - average_ranks[index] <= critical_difference
        ):
            end += 1
        if end > index:
            groups.append(
                {
                    'start_rank': float(average_ranks[index]),
                    'end_rank': float(average_ranks[end]),
                    'lane': int(lane),
                    'algorithms': algorithms[index:end + 1],
                },
            )
            lane = (lane + 1) % 4
            index = end + 1
        else:
            index += 1
    return groups


def _critical_difference_chart_payload(
    ranks: dict[str, list[float]],
    algorithm_order: list[str],
    *,
    key: str,
    title: str,
) -> dict[str, object]:
    algorithms: list[str] = []
    values: list[list[float]] = []
    average_ranks: list[float] = []
    sample_count = 0
    for name in algorithm_order:
        pts = ranks.get(name, [])
        if not pts:
            continue
        avg_rank = float(sum(pts) / len(pts))
        algorithms.append(name)
        values.append([avg_rank])
        average_ranks.append(avg_rank)
        sample_count = max(sample_count, len(pts))

    critical_difference = _critical_difference_threshold(len(algorithms), sample_count)
    return {
        'key': key,
        'title': title,
        'y_label': 'Average rank (lower is better)',
        'log_y': False,
        'chart_type': 'critical_difference',
        'algorithms': algorithms,
        'values': values,
        'critical_difference': critical_difference,
        'sample_count': sample_count,
        'rank_groups': _critical_difference_groups(
            algorithms,
            average_ranks,
            critical_difference,
        ),
    }


def _build_pareto_level_series(points: list[dict[str, object]]) -> list[dict[str, object]]:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for point in points:
        level_order = point.get('level_order')
        family = point.get('family')
        if level_order is None or not family:
            continue
        by_family[str(family)].append(point)

    series: list[dict[str, object]] = []
    for family in sorted(by_family):
        ordered = sorted(by_family[family], key=lambda row: float(row['level_order']))
        if len(ordered) < 2:
            continue
        series.append(
            {
                'family': family,
                'points': [
                    {
                        'algorithm': str(row['algorithm']),
                        'rate': float(row['rate']),
                        'compression_time': float(row['compression_time']),
                        'level_order': float(row['level_order']),
                    }
                    for row in ordered
                ],
            },
        )
    return series


def _build_actual_frontier_polyline(frontier: list[dict[str, object]]) -> list[list[float]]:
    ordered = sorted(
        frontier,
        key=lambda row: (float(row['rate']), float(row['compression_time'])),
    )
    path: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for point in ordered:
        x = float(point['rate'])
        y = float(point['compression_time'])
        coord = (x, y)
        if coord in seen:
            continue
        seen.add(coord)
        path.append([x, y])
    return path


def compute_pareto_chart(
    family_rates: dict[str, list[float]],
    family_comp_times: dict[str, list[float]],
    level_rates: dict[str, list[float]],
    level_comp_times: dict[str, list[float]],
    *,
    include_lossy: bool = False,
) -> dict[str, object]:
    """Build Pareto chart: family-best points (same as boxplots) plus per-level trajectories."""
    points: list[dict[str, object]] = []
    leveled_families: set[str] = set()

    for algo_key in level_rates:
        if not is_home_leaderboard_algorithm_enabled(algo_key):
            continue
        level_order = pareto_level_sort_key(algo_key)
        if level_order is None:
            continue
        family = boxplot_group_label(algo_key)
        if not family:
            continue
        if not include_lossy and is_lossy_boxplot_label(family):
            continue
        rate = _mean(level_rates, algo_key)
        comp_time = _mean(level_comp_times, algo_key)
        if rate is None or comp_time is None or comp_time <= 0:
            continue
        leveled_families.add(family)
        points.append(
            {
                'algorithm': pareto_point_label(algo_key),
                'algorithm_key': str(algo_key),
                'family': family,
                'level_order': level_order,
                'rate': rate,
                'compression_time': comp_time,
            },
        )

    for label in family_rates:
        if not is_home_leaderboard_label_enabled(label):
            continue
        if label in leveled_families:
            continue
        if not include_lossy and is_lossy_boxplot_label(label):
            continue
        rate = _mean(family_rates, label)
        comp_time = _mean(family_comp_times, label)
        if rate is None or comp_time is None or comp_time <= 0:
            continue
        points.append(
            {
                'algorithm': label,
                'algorithm_key': label,
                'family': label,
                'level_order': None,
                'rate': rate,
                'compression_time': comp_time,
            },
        )

    pareto_flags = {idx: False for idx in range(len(points))}
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (
                float(q['rate']) <= float(p['rate'])
                and float(q['compression_time']) <= float(p['compression_time'])
                and (
                    float(q['rate']) < float(p['rate'])
                    or float(q['compression_time']) < float(p['compression_time'])
                )
            ):
                dominated = True
                break
        pareto_flags[i] = not dominated

    for idx, flag in pareto_flags.items():
        points[idx]['pareto'] = bool(flag)

    frontier = [p for p in points if p.get('pareto')]

    return {
        'title': 'Pareto Frontier (Compression Rate vs Time)',
        'x_label': 'Compression rate',
        'x_min': 0.0,
        'x_max': 0.9,
        'y_label': 'Compression time (ns/point)',
        'points': points,
        'level_series': _build_pareto_level_series(points),
        'frontier_style': 'actual_polyline',
        'frontier': _build_actual_frontier_polyline(frontier),
    }


def build_home_leaderboard_boxplot_groups(
    backend: Path,
    *,
    min_rate: float | None = MIN_RATE_FOR_BOX_PLOT,
    implement_language: str = 'java',
    data_type: str = 'float',
    filter_params: dict | None = None,
    dataset_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    selected_column_index: int | None = None,
    float_precision: str = FLOAT_PRECISION_FILTER_ALL,
    leaderboard_mode: str = '',
    on_dataset_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object], int]:
    effective_language = str(implement_language or 'java').strip().lower() or 'java'
    effective_dataset_keys = dataset_keys
    excluded_families = None
    if str(leaderboard_mode or '').strip().lower() == 'paper_table3':
        from .paper_table3_leaderboard import (
            PAPER_TABLE3_EXCLUDED_FAMILIES,
            paper_table3_eligible_dataset_keys,
            paper_table3_scope_for_request,
        )

        if paper_table3_scope_for_request(data_type, float_precision) is not None:
            effective_language = 'cpp'
            excluded_families = PAPER_TABLE3_EXCLUDED_FAMILIES
            effective_dataset_keys = paper_table3_eligible_dataset_keys(
                backend,
                data_type=data_type,
                float_precision=float_precision,
                implement_language=effective_language,
                dataset_keys=dataset_keys,
            )

    (
        rates,
        comp_times,
        decomp_times,
        balanced_costs,
        balanced_ranks,
        pareto_rates,
        pareto_comp_times,
        datasets_used,
    ) = collect_metrics(
        backend,
        min_rate=min_rate,
        implement_language=effective_language,
        data_type=data_type,
        filter_params=filter_params,
        dataset_keys=effective_dataset_keys,
        selected_column_index=selected_column_index,
        float_precision=float_precision,
        excluded_families=excluded_families,
        on_dataset_progress=on_dataset_progress,
    )

    lossless_rate_labels = [
        label for label in sort_labels_by_mean_rate(rates) if not is_lossy_boxplot_label(label)
    ]
    lossy_rate_labels = [
        label for label in sort_labels_by_mean_rate(rates) if is_lossy_boxplot_label(label)
    ]
    lossless_time_labels = [
        label for label in sort_labels_by_mean_value(comp_times) if not is_lossy_boxplot_label(label)
    ]
    lossy_time_labels = [
        label for label in sort_labels_by_mean_value(comp_times) if is_lossy_boxplot_label(label)
    ]
    lossless_decomp_labels = [
        label for label in sort_labels_by_mean_value(decomp_times) if not is_lossy_boxplot_label(label)
    ]
    lossy_decomp_labels = [
        label for label in sort_labels_by_mean_value(decomp_times) if is_lossy_boxplot_label(label)
    ]
    lossless_balanced_labels = [
        label for label in sort_labels_by_mean_value(balanced_costs) if not is_lossy_boxplot_label(label)
    ]
    lossless_rank_labels = [
        label for label in sort_labels_by_mean_value(balanced_ranks) if not is_lossy_boxplot_label(label)
    ]
    lossy_balanced_labels = [
        label for label in sort_labels_by_mean_value(balanced_costs) if is_lossy_boxplot_label(label)
    ]
    lossy_rank_labels = [
        label for label in sort_labels_by_mean_value(balanced_ranks) if is_lossy_boxplot_label(label)
    ]

    groups = {
        'compression_rate': {
            'title': 'Compression Rate Boxplots',
            'charts': {
                'lossless': _chart_payload(
                    rates,
                    lossless_rate_labels,
                    key='lossless',
                    title='Lossless',
                    y_label='Compression rate',
                    log_y=False,
                ),
                'lossy': _chart_payload(
                    rates,
                    lossy_rate_labels,
                    key='lossy',
                    title='Lossy',
                    y_label='Compression rate',
                    log_y=True,
                ),
            },
        },
        'compression_time': {
            'title': 'Compression Time Boxplots',
            'charts': {
                'lossless': _chart_payload(
                    comp_times,
                    lossless_time_labels,
                    key='lossless',
                    title='Lossless',
                    y_label='Compression time (ns/point)',
                    log_y=True,
                ),
                'lossy': _chart_payload(
                    comp_times,
                    lossy_time_labels,
                    key='lossy',
                    title='Lossy',
                    y_label='Compression time (ns/point)',
                    log_y=True,
                ),
            },
        },
        'decompression_time': {
            'title': 'Decompression Time Boxplots',
            'charts': {
                'lossless': _chart_payload(
                    decomp_times,
                    lossless_decomp_labels,
                    key='lossless',
                    title='Lossless',
                    y_label='Decompression time (ns/point)',
                    log_y=True,
                ),
                'lossy': _chart_payload(
                    decomp_times,
                    lossy_decomp_labels,
                    key='lossy',
                    title='Lossy',
                    y_label='Decompression time (ns/point)',
                    log_y=True,
                ),
            },
        },
        'balanced_score': {
            'title': 'Balanced Compression-Time Score',
            'charts': {
                'lossless_score': _chart_payload(
                    _mean_value_samples(balanced_costs),
                    lossless_balanced_labels,
                    key='lossless_score',
                    title='Lossless score',
                    y_label='Weighted balanced score (lower is better)',
                    log_y=True,
                    chart_type='bar',
                ),
                'lossless_rank': _chart_payload(
                    balanced_ranks,
                    lossless_rank_labels,
                    key='lossless_rank',
                    title='Lossless rank',
                    y_label='Rank across datasets (lower is better)',
                    log_y=False,
                ),
                'lossless_critical_difference': _critical_difference_chart_payload(
                    balanced_ranks,
                    lossless_rank_labels,
                    key='lossless_critical_difference',
                    title='Critical difference diagram',
                ),
                'lossy_score': _chart_payload(
                    _mean_value_samples(balanced_costs),
                    lossy_balanced_labels,
                    key='lossy_score',
                    title='Lossy score',
                    y_label='Weighted balanced score (lower is better)',
                    log_y=True,
                    chart_type='bar',
                ),
                'lossy_rank': _chart_payload(
                    balanced_ranks,
                    lossy_rank_labels,
                    key='lossy_rank',
                    title='Lossy rank',
                    y_label='Rank across datasets (lower is better)',
                    log_y=False,
                ),
                'lossy_critical_difference': _critical_difference_chart_payload(
                    balanced_ranks,
                    lossy_rank_labels,
                    key='lossy_critical_difference',
                    title='Critical difference diagram',
                ),
            },
        },
    }
    pareto_chart = compute_pareto_chart(
        rates,
        comp_times,
        pareto_rates,
        pareto_comp_times,
        include_lossy=False,
    )
    return groups, pareto_chart, datasets_used
