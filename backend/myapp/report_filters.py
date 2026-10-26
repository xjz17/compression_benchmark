"""Shared compression report filtering (Home staticdata + Upload API)."""

from __future__ import annotations

import os
from typing import Any, Mapping

import pandas as pd

from .column_encodings import is_legacy_column_rle_algorithm
from .lzbench_codec_catalog import (
    canonical_report_algorithm_key,
    is_lzbench_catalog_algorithm,
    parse_lzbench_levels_param,
    selected_lzbench_algorithm_keys,
)

_RESEARCH_FAMILY_VARIANTS = {
    'simpiece': [
        ('simpiece_relRange_0p005', 'simpiece_float_relRange_0p005'),
        ('simpiece_relRange_0p01', 'simpiece_float_relRange_0p01'),
    ],
    'serf': [
        ('serf_float_ABS_1e-1',),
        ('serf_float_ABS_1e-2',),
        ('serf_float_ABS_1e-3',),
        ('serf_float_ABS_1e-4',),
    ],
}

_SZ_ABS_ORDER = [
    'sz_ABS_1e-1',
    'sz_ABS_1e-2',
    'sz_ABS_1e-3',
    'sz_ABS_1e-4',
]
_SZ_REL_ORDER = [
    'sz_REL_1e-2',
    'sz_REL_1e-3',
]
_SZ3_ABS_GROUPS = [
    ['sz3_ft_ABS_1e-1', 'sz3_ABS_1e-1'],
    ['sz3_ABS_1e-2'],
    ['sz3_ABS_1e-3'],
    ['sz3_ABS_1e-4'],
]
_SZ3_REL_ORDER = [
    'sz3_REL_1e-2',
    'sz3_REL_1e-3',
    'sz3_REL_1e-4',
]
_MACHETE_REL_ORDER = [
    'machete_rel_1e-1',
    'machete_rel_1e-2',
    'machete_rel_1e-3',
    'machete_rel_1e-4',
]
_ZFP_TOL_ORDER = [
    'ext_zfp_tol_1e-1',
    'ext_zfp_tol_1e-2',
    'ext_zfp_tol_1e-3',
]
_SPDP_LEVEL_ORDER = [f'spdp_level_{i}' for i in range(10)]

_CHIMP_HISTORY_VALUES = (1, 128)
_CHIMP_HISTORY_1_KEYS = ('CHIMP',)
_CHIMP_HISTORY_128_KEYS = ('CHIMP128',)
_CHIMP_LEGACY_KEYS = ('CHIMP',)

_HIDDEN_REPORT_ALGORITHM_KEYS = frozenset(
    {
        *(f'subcolumn_l{level}' for level in range(6)),
        *(f'ts_2diff_bos_l{level}' for level in range(6)),
        *(f'oss_l{level}' for level in range(6)),
    },
)


def is_hidden_report_algorithm(algorithm_key: str) -> bool:
    key = canonical_report_algorithm_key(algorithm_key)
    return bool(key) and key in _HIDDEN_REPORT_ALGORITHM_KEYS


def parse_report_filter_params(data: Mapping[str, Any]) -> dict:
    xz_raw = data.get('xz_level')
    lzma_raw = data.get('lzma_level')
    zstd_raw = data.get('zstd_level')
    xz_level = int(xz_raw if xz_raw is not None else lzma_raw) if (
        xz_raw is not None or lzma_raw is not None
    ) else 9
    zstd_level = int(zstd_raw) if zstd_raw is not None else 9
    sz_absrel = (str(data.get('sz_absrel', 'ABS') or 'ABS')).upper()
    if sz_absrel not in ('ABS', 'REL'):
        sz_absrel = 'ABS'
    sz3_absrel = (str(data.get('sz3_absrel', 'ABS') or 'ABS')).upper()
    if sz3_absrel not in ('ABS', 'REL'):
        sz3_absrel = 'ABS'
    try:
        chimp_history = int(data.get('chimp_history', 128) or 128)
    except (TypeError, ValueError):
        chimp_history = 128
    if chimp_history not in _CHIMP_HISTORY_VALUES:
        chimp_history = 128
    return {
        'xz_level': xz_level,
        'lzma_level': xz_level,
        'zstd_level': zstd_level,
        'chimp_history': chimp_history,
        'simpiece_param': int(data.get('simpiece_param', 0) or 0),
        'serf_param': int(data.get('serf_param', 0) or 0),
        'sz_absrel': sz_absrel,
        'sz_eps': int(data.get('sz_eps', 0) or 0),
        'sz3_absrel': sz3_absrel,
        'sz3_eps': int(data.get('sz3_eps', 0) or 0),
        'machete_eps': int(data.get('machete_eps', 2) or 2),
        'spdp_level': max(0, min(int(data.get('spdp_level', 9) or 9), 9)),
        'zfp_tol_eps': max(0, min(int(data.get('zfp_tol_eps', 2) or 2), 2)),
        'lzbench_levels': parse_lzbench_levels_param(data.get('lzbench_levels')),
    }


def _research_family_predicates(algo: pd.Series):
    return {
        'simpiece': algo.str.startswith('simpiece_', na=False),
        'sz': algo.str.startswith('sz_', na=False),
        'sz3': algo.str.startswith('sz3_', na=False),
        'serf': algo.str.startswith('serf_float_', na=False),
    }


def _pick_slider_algorithm_key(algo: pd.Series, ordered_keys: list[str], index: int) -> str | None:
    if not ordered_keys:
        return None
    i = max(0, min(int(index), len(ordered_keys) - 1))
    for k in ordered_keys[i::-1]:
        if (algo == k).any():
            return k
    for k in ordered_keys[i + 1:]:
        if (algo == k).any():
            return k
    return None


def _is_sz_algorithm(algo: str) -> bool:
    s = str(algo or '')
    return s.startswith('sz_') and not s.startswith('sz3_')


def _filter_sz_sample(df: pd.DataFrame, absrel: str, eps_idx: int) -> pd.DataFrame:
    if df.empty:
        return df
    algo = df['algorithm'].astype(str)
    pred = algo.map(_is_sz_algorithm)
    if not pred.any():
        return df
    absrel = (absrel or 'ABS').upper()
    order = _SZ_REL_ORDER if absrel == 'REL' else _SZ_ABS_ORDER
    chosen = _pick_slider_algorithm_key(algo, order, eps_idx)
    if not chosen:
        return df
    drop = pred & (algo != chosen)
    return df.loc[~drop]


def _filter_sz3_sample(df: pd.DataFrame, absrel: str, eps_idx: int) -> pd.DataFrame:
    if df.empty:
        return df
    algo = df['algorithm'].astype(str)
    pred = algo.str.startswith('sz3_', na=False)
    if not pred.any():
        return df
    absrel = (absrel or 'ABS').upper()
    if absrel == 'REL':
        chosen = _pick_slider_algorithm_key(algo, _SZ3_REL_ORDER, eps_idx)
    else:
        groups = _SZ3_ABS_GROUPS
        gi = max(0, min(int(eps_idx), len(groups) - 1))
        chosen = None
        for gix in range(gi, -1, -1):
            for k in groups[gix]:
                if (algo == k).any():
                    chosen = k
                    break
            if chosen:
                break
        if not chosen:
            for gix in range(gi + 1, len(groups)):
                for k in groups[gix]:
                    if (algo == k).any():
                        chosen = k
                        break
                if chosen:
                    break
    if not chosen:
        return df
    drop = pred & (algo != chosen)
    return df.loc[~drop]


def _filter_machete_sample(df: pd.DataFrame, eps_idx: int) -> pd.DataFrame:
    if df.empty:
        return df
    algo = df['algorithm'].astype(str)
    pred = algo.str.startswith('machete_rel_', na=False) | algo.str.startswith(
        'machete_float_rel_', na=False
    )
    if not pred.any():
        return df
    chosen = _pick_slider_algorithm_key(algo, list(_MACHETE_REL_ORDER), eps_idx)
    if not chosen:
        legacy_order = [k.replace('machete_rel_', 'machete_float_rel_', 1) for k in _MACHETE_REL_ORDER]
        chosen = _pick_slider_algorithm_key(algo, legacy_order, eps_idx)
    if not chosen:
        return df
    drop = pred & (algo != chosen)
    return df.loc[~drop]


def _filter_spdp_sample(df: pd.DataFrame, level_idx: int) -> pd.DataFrame:
    if df.empty:
        return df
    algo = df['algorithm'].astype(str)
    pred = algo.str.startswith('spdp_level_', na=False) | algo.str.startswith(
        'spdp_float_level_', na=False
    )
    if not pred.any():
        return df
    chosen = _pick_slider_algorithm_key(algo, list(_SPDP_LEVEL_ORDER), level_idx)
    if not chosen:
        return df
    drop = pred & (algo != chosen)
    return df.loc[~drop]


def _filter_zfp_tol_sample(df: pd.DataFrame, eps_idx: int) -> pd.DataFrame:
    if df.empty:
        return df
    algo = df['algorithm'].astype(str)
    pred = algo.str.startswith('ext_zfp_tol_', na=False)
    if not pred.any():
        return df
    chosen = _pick_slider_algorithm_key(algo, list(_ZFP_TOL_ORDER), eps_idx)
    if not chosen:
        return df
    drop = pred & (algo != chosen)
    return df.loc[~drop]


def _filter_one_research_variant(df: pd.DataFrame, family_key: str, param_index: int) -> pd.DataFrame:
    if df.empty or family_key not in _RESEARCH_FAMILY_VARIANTS:
        return df
    variant_groups = _RESEARCH_FAMILY_VARIANTS[family_key]
    algo = df['algorithm'].astype(str)
    preds = _research_family_predicates(algo)
    pred = preds[family_key]
    if not pred.any():
        return df
    i = max(0, min(int(param_index), len(variant_groups) - 1))
    chosen = None
    for k in variant_groups[i]:
        if (algo == k).any():
            chosen = k
            break
    if not chosen:
        for gi in range(i - 1, -1, -1):
            for k in variant_groups[gi]:
                if (algo == k).any():
                    chosen = k
                    break
            if chosen:
                break
    if not chosen:
        for gi in range(i + 1, len(variant_groups)):
            for k in variant_groups[gi]:
                if (algo == k).any():
                    chosen = k
                    break
            if chosen:
                break
    if not chosen:
        return df
    drop = pred & (algo != chosen)
    return df.loc[~drop]


def _filter_lzbench_sample(df: pd.DataFrame, lzbench_levels: Mapping[str, int | str] | None) -> pd.DataFrame:
    if df.empty:
        return df
    algo = df['algorithm'].astype(str)
    has_lzbench = algo.apply(is_lzbench_catalog_algorithm)
    if not has_lzbench.any():
        return df
    keep = selected_lzbench_algorithm_keys(lzbench_levels)
    drop = has_lzbench & ~algo.isin(keep)
    return df.loc[~drop]


def _filter_chimp_sample(df: pd.DataFrame, chimp_history: int) -> pd.DataFrame:
    if df.empty:
        return df
    algo = df['algorithm'].astype(str)
    is_chimp = algo.isin(_CHIMP_HISTORY_1_KEYS + _CHIMP_HISTORY_128_KEYS)
    if not is_chimp.any():
        return df
    if int(chimp_history) == 1:
        keep = set(_CHIMP_HISTORY_1_KEYS)
    else:
        keep = set(_CHIMP_HISTORY_128_KEYS)
        if not algo.isin(keep).any():
            keep.update(_CHIMP_LEGACY_KEYS)
    drop = is_chimp & ~algo.isin(keep)
    return df.loc[~drop]


def apply_research_codec_filters(
    df: pd.DataFrame,
    *,
    simpiece_param: int,
    serf_param: int,
    sz_absrel: str,
    sz_eps: int,
    sz3_absrel: str,
    sz3_eps: int,
    machete_eps: int,
    spdp_level: int,
    zfp_tol_eps: int,
) -> pd.DataFrame:
    cur = _filter_machete_sample(df, machete_eps)
    cur = _filter_spdp_sample(cur, spdp_level)
    cur = _filter_zfp_tol_sample(cur, zfp_tol_eps)
    cur = _filter_one_research_variant(cur, 'simpiece', simpiece_param)
    cur = _filter_sz_sample(cur, sz_absrel, sz_eps)
    cur = _filter_sz3_sample(cur, sz3_absrel, sz3_eps)
    cur = _filter_one_research_variant(cur, 'serf', serf_param)
    return cur


def sanitize_compression_report_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid lzbench placeholder rows and normalize raw lzbench stdout names."""
    if df is None or df.empty or 'algorithm' not in df.columns:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()
    cur = df.dropna(subset=['algorithm']).copy()
    algo = cur['algorithm'].astype(str)
    cur = cur.loc[~algo.str.lower().str.startswith('lzbench selector', na=False)]
    if cur.empty:
        return cur
    unique_names = algo.loc[cur.index].unique()
    name_map = {name: canonical_report_algorithm_key(name) for name in unique_names}
    cur['algorithm'] = cur['algorithm'].map(name_map)
    cur = cur[cur['algorithm'].notna()]
    if cur.empty:
        return cur
    if 'column_index' in cur.columns and 'originalSize' in cur.columns:
        cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
        cur['originalSize'] = pd.to_numeric(cur['originalSize'], errors='coerce')
        cur = cur.dropna(subset=['column_index', 'originalSize'])
        if cur.empty:
            return cur
        cur['column_index'] = cur['column_index'].astype(int)
        cur = cur[~((cur['algorithm'].astype(str) == 'reger') & (cur['column_index'] != -1))]
        if cur.empty:
            return cur
        keep_idx = cur.groupby(['column_index', 'algorithm'], sort=False)['originalSize'].idxmax()
        cur = cur.loc[keep_idx]
    return cur


def filter_compression_report_dataframe(
    df: pd.DataFrame,
    *,
    implement_language: str,
    sanitize_input: bool = True,
    lzma_level: int = 9,
    xz_level: int | None = None,
    zstd_level: int = 9,
    chimp_history: int = 128,
    simpiece_param: int = 0,
    serf_param: int = 0,
    sz_absrel: str = 'ABS',
    sz_eps: int = 0,
    sz3_absrel: str = 'ABS',
    sz3_eps: int = 0,
    machete_eps: int = 2,
    spdp_level: int = 9,
    zfp_tol_eps: int = 2,
    lzbench_levels: Mapping[str, int | str] | None = None,
) -> pd.DataFrame:
    cur_df = sanitize_compression_report_dataframe(df) if sanitize_input else df.copy()
    if cur_df.empty:
        return cur_df
    cur_df = cur_df.copy()
    cur_df = cur_df[~cur_df['algorithm'].map(is_hidden_report_algorithm)]
    if cur_df.empty:
        return cur_df
    if xz_level is None:
        xz_level = lzma_level
    algo = cur_df['algorithm'].astype(str)
    cur_df = cur_df[
        (~algo.str.startswith('xz_level_', na=False))
        & (~algo.str.startswith('zstd_level_', na=False))
        | (algo == f'xz_level_{xz_level}')
        | (algo == f'zstd_level_{zstd_level}')
    ]
    cur_df = _filter_chimp_sample(cur_df, chimp_history)
    cur_df = _filter_lzbench_sample(cur_df, lzbench_levels)
    if implement_language in ('java', 'python', 'cpp'):
        cur_df = apply_research_codec_filters(
            cur_df,
            simpiece_param=simpiece_param,
            serf_param=serf_param,
            sz_absrel=sz_absrel,
            sz_eps=sz_eps,
            sz3_absrel=sz3_absrel,
            sz3_eps=sz3_eps,
            machete_eps=machete_eps,
            spdp_level=spdp_level,
            zfp_tol_eps=zfp_tol_eps,
        )
    algo = cur_df['algorithm'].astype(str)
    cur_df = cur_df[~algo.map(is_legacy_column_rle_algorithm)]
    return cur_df


def rows_to_benchmark_items(
    filtered_df: pd.DataFrame,
    unified_original: float,
    *,
    tsfile_dir: str | None = None,
    process_id: str | None = None,
) -> list[dict]:
    items = []
    numer = float(unified_original) if unified_original and float(unified_original) > 0 else 0.0
    for _, row in filtered_df.iterrows():
        try:
            algorithm_name = str(row['algorithm'])
            if is_hidden_report_algorithm(algorithm_name):
                continue
            compressed_size = int(row['compressedSize'])
            if tsfile_dir and os.path.isdir(tsfile_dir):
                for filename in os.listdir(tsfile_dir):
                    file_basename = os.path.splitext(filename)[0]
                    if file_basename.endswith(algorithm_name):
                        file_path = os.path.join(tsfile_dir, filename)
                        compressed_size = os.path.getsize(file_path)
                        break
            if compressed_size <= 0:
                compressed_size = 1
            if numer <= 0:
                continue
            entry = {
                'algorithm': algorithm_name,
                'compression_rate': compressed_size / numer,
                'compression_time': float(row['compressionTime']) / 1e9,
                'decompression_time': float(row['decompressionTime']) / 1e9,
                'space_cost': compressed_size,
            }
            if process_id is not None:
                entry['process_id'] = process_id
            items.append(entry)
        except (TypeError, ValueError, KeyError):
            continue
    return items
