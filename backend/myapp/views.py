from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404, HttpResponse

# Create your views here.

from pathlib import Path
from collections.abc import Callable
from statistics import median

from rest_framework import viewsets
from rest_framework import status

from rest_framework.authentication import TokenAuthentication
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import UploadedAlgorithm, UploadedDataset
from .report_filters import (
    apply_research_codec_filters as _apply_research_codec_filters,
    filter_compression_report_dataframe,
    parse_report_filter_params,
    rows_to_benchmark_items,
)
from .column_info_loader import (
    benchmark_dtypes_from_column_info,
    catalog_dtypes_from_column_info,
    distinct_logical_types_from_column_info,
    load_column_info_rows,
)
from .feature_flags import HIDE_LOSSY_ALGORITHMS
from .active_method_policy import (
    active_method_policy_summary,
    is_active_method_name,
)
from .method_catalog import (
    compression_mode_for_method_name,
    load_method_operator_pipelines,
    resolve_method_operator_steps,
)
from .dataset_categories import normalize_major_category
from .dataset_dtype_distribution import accumulate_dtype_metrics
from .dataset_float_precision_distribution import (
    FLOAT_PRECISION_FILTER_ALL,
    accumulate_float_precision_metrics,
    empty_float_precision_catalog_fields,
    filter_report_dataframe_for_float_precision,
    float_precision_catalog_fields,
    parse_float_precision_filter,
)
from .dataset_catalog_compression_rate import (
    average_column_compression_rate,
    average_compression_rates_by_bucket,
    best_available_column_compression_metrics,
    collect_numeric_column_ts2diff_rates,
    compression_rates_from_sums,
    empty_dtype_rate_sums,
    empty_float_precision_rate_sums,
    extend_column_average_rates_by_dtype,
    extend_float_ts2diff_rates_by_precision,
    extend_numeric_ts2diff_rates_by_bucket,
    global_compression_rate,
    try_accumulate_dataset_overall_compression_rate_sums,
)
from .dataset_catalog_response_cache import (
    catalog_response_cache_key,
    get_or_build_catalog_response,
)
from .dataset_catalog_filters import parse_catalog_part
from .dataset_catalog_result_store import (
    catalog_items_path,
    catalog_summary_path,
    ledger_public_compress_summary_path,
    load_precomputed_catalog_items,
    load_precomputed_catalog_summary,
)
from .upload_storage import (
    build_user_upload_asset_urls,
    create_user_upload,
    parse_bool_form_value,
    parse_user_upload_dataset_key,
    resolve_user_upload_chart_path,
    resolve_user_upload_catalog_asset_file,
    resolve_user_upload_column_types_csv,
    resolve_user_upload_master_csv,
    resolve_user_upload_statistic_csv,
    user_upload_dataset_key,
    user_upload_logical_types,
    user_upload_abs_dir,
    user_upload_report_csv_path,
    user_upload_slug,
    user_upload_tsfile_artifact_dir,
    resolve_user_upload_tsfile_artifact_path,
    user_upload_tsfile_report_path,
    user_upload_has_tsfile_benchmark,
    user_upload_has_asset_bundle,
    user_upload_tsfile_total_bytes,
    user_upload_has_prebuilt_tsfile,
    resolve_user_upload_prebuilt_tsfile_path,
    get_user_upload_by_dataset_key,
)
from .dataset_bundle import (
    catalog_asset_filename,
    catalog_has_statistics_bundle,
    catalog_servable_asset_filenames,
    column_info_path,
    dataset_dir,
    dataset_info_path,
    dataset_root,
    dtype_asset_filename,
    list_catalog_operator_ids,
    refresh_catalog_items_operators,
    resolve_master_csv_path,
    prebuilt_tsfile_path,
    resolve_prebuilt_tsfile_path,
    tsfile_bundle_dir,
    catalog_tsfile_info_path,
    typed_tsfile_encoding_summary_path,
)
from .dataset_operators import (
    SUPPORTED_OPERATORS,
    ensure_operator_catalog_assets,
    operator_bundle_dir,
    operator_statistic_csv_path,
)
from .encoding_operator_catalog import (
    load_byte_codec_operators,
    load_encoding_operators,
    load_method_page_operators,
    operator_row_for_api,
)
from .typed_tsfile_export import file_size_bytes_from_catalog_tsfile_info
from .user_upload_pipeline import start_user_upload_pipeline
from .user_algorithm_storage import user_algorithm_result_path
from .user_algorithm_targets import benchmark_column_types_for_algorithm
from .dataset_catalog_assets import (
    column_distribution_chart_dirname,
    column_distribution_chart_filename_candidates,
    count_csv_data_points,
    dataset_info_filename,
    load_dataset_info,
)
from .processor import *
from .statistic import *

import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import uuid
import tempfile
from contextlib import contextmanager
import mimetypes
import csv
import io
import zipfile
import json
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Always resolve data dirs from the Django project root (backend/), never from the shell cwd.
_BACKEND_ROOT = str(settings.BASE_DIR)

def _sanitize_for_json(obj):
    """RFC 8259–friendly payload: no NaN/Inf (browsers' JSON.parse rejects them)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, 'item'):
        try:
            return _sanitize_for_json(obj.item())
        except Exception:
            pass
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _java_benchmark_command(jar_path, *main_args):
    """Run the shaded benchmark entrypoint on a classpath (optional extras via env/file)."""
    from .bench_executable_paths import iter_java_classpath_append_files, repo_root as _bench_repo_root

    parts = [os.path.abspath(jar_path)]
    extra = os.environ.get('COMPRESSION_BENCHMARK_JAVA_CLASSPATH_APPEND', '').strip()
    if extra:
        parts.extend(p for p in extra.split(os.pathsep) if p.strip())
    for append_file in iter_java_classpath_append_files(Path(parts[0])):
        with open(append_file, encoding='utf-8') as af:
            for line in af:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts.append(line)
    # Same as scripts/regenerate_staticdata_java.py: Elf+ JAR (+ deps) after benchmark JAR.
    repo_root = str(_bench_repo_root())
    elf_jar = os.path.join(
        repo_root, 'external', 'compression', 'elf', 'target', 'start-compress-1.0-SNAPSHOT.jar')
    if os.path.isfile(elf_jar):
        parts.append(elf_jar)
    camel_jar = os.path.join(
        repo_root, 'external', 'compression', 'camel', 'target', 'start-compress-1.0-SNAPSHOT.jar')
    camel_dep = os.path.join(
        repo_root, 'external', 'compression', 'camel', 'target', 'dependency')
    if os.path.isfile(camel_jar):
        parts.append(camel_jar)
        if os.path.isdir(camel_dep):
            for name in sorted(os.listdir(camel_dep)):
                if name.endswith('.jar'):
                    parts.append(os.path.join(camel_dep, name))
    _raw = os.environ.get('COMPRESSION_BENCHMARK_JAVA_OPTS')
    if _raw is None:
        jvm_opts = '-Xmx8g'
    else:
        jvm_opts = _raw.strip()
    jvm_prefix = shlex.split(jvm_opts) if jvm_opts else []
    return ['java', *jvm_prefix, '-cp', os.pathsep.join(parts), 'org.example.Main', *main_args]


# Upload / staticdata parity (see scripts/regenerate_staticdata_java.py).
_BENCHMARK_ENV_DEFAULTS = {
    'COMPRESSION_BENCHMARK_CMIX_BENCH': '0',
    'COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS': '1',
    'COMPRESSION_BENCHMARK_CMIX_MAX_COLUMN_BYTES': str(64 * 1024),
    'COMPRESSION_BENCHMARK_CMIX_TIMEOUT_SEC': '180',
    'COMPRESSION_BENCHMARK_MSDZIP_BENCH': '0',
    'COMPRESSION_BENCHMARK_MSDLC_BENCH': '0',
    'COMPRESSION_BENCHMARK_SERF_BENCH': '0',
    'COMPRESSION_BENCHMARK_SZ_BENCH': '1',
    'COMPRESSION_BENCHMARK_SZ3_BENCH': '1',
    'COMPRESSION_BENCHMARK_BUFF_BENCH': '0',
    'COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE': '1',
    'COMPRESSION_BENCHMARK_FPZIP_BENCH': '1',
    'COMPRESSION_BENCHMARK_STATICDATA_REGEN': '1',
}


def _apply_benchmark_env_defaults(env=None):
    env = dict(os.environ if env is None else env)
    for key, value in _BENCHMARK_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    for key in list(env.keys()):
        if key.startswith('COMPRESSION_BENCHMARK_CPP_NATIVE'):
            del env[key]
    return env


@contextmanager
def _use_benchmark_env_defaults():
    touched = set(_BENCHMARK_ENV_DEFAULTS.keys())
    for key in list(os.environ.keys()):
        if key.startswith('COMPRESSION_BENCHMARK_CPP_NATIVE'):
            touched.add(key)
    saved = {k: os.environ.get(k) for k in touched}
    try:
        _apply_benchmark_env_defaults(os.environ)
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _java_bench_subprocess_env():
    return _apply_benchmark_env_defaults()


def _repo_root():
    return os.path.abspath(os.path.join(_BACKEND_ROOT, '..'))


def _focused_original_dataset_keys() -> set[str]:
    """Restrict built-in catalog UI to timestamped public Compress datasets."""
    compress_keys = _ledger_public_compress_ready_dataset_keys()
    if compress_keys:
        return compress_keys

    root = Path(_repo_root())
    focused: set[str] = set()
    for dirname in ('original_datasets', 'orignial_datasets', 'original_dataset'):
        folder = root / dirname
        if not folder.is_dir():
            continue
        focused.update(
            path.stem
            for path in folder.glob('*.csv')
            if path.is_file() and path.stem
        )
    return focused


def _ledger_public_compress_ready_dataset_keys() -> set[str]:
    path = ledger_public_compress_summary_path()
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return set()
    meta = payload.get('meta') if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        return set()
    keys = meta.get('ready_dataset_keys') or meta.get('dataset_keys') or []
    if not isinstance(keys, list):
        return set()
    return {
        str(key or '').strip()
        for key in keys
        if str(key or '').strip()
    }


def _focused_dataset_filter_enabled() -> bool:
    return bool(_focused_original_dataset_keys())


def _filter_focused_builtin_dataset_keys(keys: list[str]) -> list[str]:
    focused = _focused_original_dataset_keys()
    if not focused:
        return keys
    return [
        key for key in keys
        if parse_user_upload_dataset_key(key) is not None or key in focused
    ]


def _cpp_benchmark_executable():
    """Native driver: ``benchmark_cpp`` (see ``cpp/``). Override with ``COMPRESSION_BENCHMARK_BENCHMARK_CPP``."""
    from .bench_executable_paths import benchmark_cpp_executable_path

    exe = benchmark_cpp_executable_path()
    return str(exe) if exe is not None else None


def _cpp_bench_subprocess_env():
    env = _apply_benchmark_env_defaults()
    env.setdefault('COMPRESSION_BENCHMARK_REPO_ROOT', _repo_root())
    env.setdefault('COMPRESSION_BENCHMARK_PYTHON', sys.executable)
    return env


_DATASET_ROOT = dataset_root()

_LOSSY_ALGO_PREFIXES = (
    'sz_',
    'sz3_',
    'serf_',
    'simpiece_',
    'ext_zfp_tol_',
    'machete_rel_',
    'machete_float_rel_',
    'most_minimal',
    'most_float_minimal',
)
_LOSSY_ALGO_LABELS = frozenset({'most', 'machete', 'simpiece', 'sz', 'sz3'})

# String/byte benchmark paths are disabled for now; keep API overall numeric-only.
_STATICDATA_BENCHMARK_DATA_TYPES = ('int', 'float')


def _is_lossy_algorithm_key(key: str) -> bool:
    algo = str(key or '').strip().removeprefix('\ufeff')
    if not algo:
        return False
    label = ''.join(ch for ch in algo.lower() if ch.isalnum())
    if label in _LOSSY_ALGO_LABELS:
        return True
    if algo.startswith('user_lossy:'):
        return True
    if algo.startswith('user_lossless:'):
        return False
    return any(algo.startswith(prefix) for prefix in _LOSSY_ALGO_PREFIXES)


def _filter_staticdata_items_hide_lossy(items: list) -> list:
    """Drop lossy algorithm rows when ``HIDE_LOSSY_ALGORITHMS`` is enabled."""
    if not HIDE_LOSSY_ALGORITHMS:
        return items
    return [
        item for item in items
        if not _is_lossy_algorithm_key(str(item.get('algorithm', '')))
    ]


def _distribution_rows_from_values(
    values_by_category: dict[str, float],
    *,
    value_field: str,
) -> list[dict]:
    rows: list[dict] = []
    positives = {
        k: float(v)
        for k, v in values_by_category.items()
        if float(v) > 0
    }
    total = float(sum(positives.values()))
    for name, value in sorted(positives.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append({
            'category': name,
            value_field: value,
            'value': value,
            'percentage': (value * 100.0 / total) if total > 0 else 0.0,
        })
    return rows


def _dataset_bundle_dir(dataset_key: str) -> str:
    return dataset_dir(dataset_key)


def _catalog_column_logical_types(dataset_key: str) -> list[str]:
    """Distinct logical_type values from ``{key}_column_info.csv`` (or legacy column_types.csv)."""
    path = column_info_path(dataset_key)
    if os.path.isfile(path):
        return distinct_logical_types_from_column_info(path)
    legacy = os.path.join(_dataset_bundle_dir(dataset_key), 'column_types.csv')
    if os.path.isfile(legacy):
        return distinct_logical_types_from_column_info(legacy)
    return []


def _prebuilt_typed_tsfile_path(dataset_key: str) -> str:
    return prebuilt_tsfile_path(dataset_key)


def _discovered_catalog_bundle_keys():
    """Dataset keys with catalog statistics sidecars under ``backend/dataset/<key>/``."""
    root = _DATASET_ROOT
    if not os.path.isdir(root):
        return []
    keys = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if catalog_has_statistics_bundle(name):
            keys.append(name)
    return keys


def _datasets_reference_dir() -> Path:
    return Path(_repo_root()) / 'benchmark' / 'reference' / 'datasets'


def _load_dataset_metadata_ordered() -> tuple[list[str], dict[str, dict]]:
    """Dataset keys from dataset_metadata.json sorted by key, plus map key -> metadata fields."""
    path = _datasets_reference_dir() / 'dataset_metadata.json'
    order: list[str] = []
    by_key: dict[str, dict] = {}
    if not path.is_file():
        logger.warning('Dataset metadata file missing: %s', path)
        return order, by_key
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        logger.exception('Failed to parse dataset metadata file: %s', path)
        return order, by_key
    entries = payload.get('datasets', [])
    if not isinstance(entries, list):
        return order, by_key
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get('key', '')).strip()
        if not key:
            continue
        order.append(key)
        surl = str(entry.get('source_url', '') or '').strip()
        by_key[key] = {
            'key': key,
            'category': str(entry.get('category', '') or '').strip(),
            'data_type': str(entry.get('data_type', '') or '').strip(),
            'source': str(entry.get('source', '') or '').strip(),
            'source_url': surl if surl else None,
        }
    order.sort()
    return order, by_key


def _all_dataset_keys_for_ui():
    """Metadata keys (sorted by name) first, then any bundle-only keys from backend/dataset."""
    bundle = set(_discovered_catalog_bundle_keys())
    order, _meta = _load_dataset_metadata_ordered()
    seen = set()
    out = []
    for k in order:
        if k not in seen:
            seen.add(k)
            out.append(k)
    for k in sorted(bundle):
        if k not in seen:
            seen.add(k)
            out.append(k)
    return _filter_focused_builtin_dataset_keys(out)


def _major_category_from_meta(meta: dict) -> str:
    raw = str(meta.get('category', '') or '').strip()
    if not raw:
        return 'Uncategorized'
    if '/' in raw:
        return raw.split('/', 1)[0].strip() or raw
    return raw


def _dataset_keys_for_major_category(major_category: str) -> list[str]:
    """Dataset keys whose metadata ``category`` matches ``major_category``."""
    target = str(major_category or '').strip()
    if not target:
        return []
    _, meta_map = _load_dataset_metadata_ordered()
    out = []
    for key in _all_dataset_keys_for_ui():
        meta = meta_map.get(key)
        if meta is None:
            continue
        if _major_category_from_meta(meta) == target:
            out.append(key)
    return out


def _resolve_dataset_csv_path(dataset_key: str, user=None):
    """Built-in master CSV (external ``backend_dataset_csv`` first, then in-repo), then user uploads."""
    bundled = resolve_master_csv_path(dataset_key)
    if bundled:
        return bundled
    record = get_user_upload_by_dataset_key(dataset_key, user)
    if record is not None:
        return resolve_user_upload_master_csv(record)
    return None


def _csv_download_available(dataset_key: str, user=None) -> bool:
    """Whether the master CSV file exists on disk (CSV download / on-demand TsFile export)."""
    return bool(_resolve_dataset_csv_path(dataset_key, user))


def _dataset_papers_path() -> Path:
    return _datasets_reference_dir() / 'dataset_papers.json'


def _method_metadata_path() -> Path:
    return _datasets_reference_dir().parent / 'methods' / 'method_metadata.json'


def _load_dataset_paper_usage_map():
    path = _dataset_papers_path()
    if not path.is_file():
        logger.warning('Dataset paper-usage file missing: %s', path)
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        logger.exception('Failed to parse dataset paper-usage file: %s', path)
        return {}

    result = {}
    entries = payload.get('datasets', [])
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dataset = str(entry.get('dataset', '')).strip()
        if not dataset:
            continue
        papers = []
        raw_papers = entry.get('papers', [])
        if isinstance(raw_papers, list):
            for paper in raw_papers:
                if not isinstance(paper, dict):
                    continue
                title = str(paper.get('title', '')).strip()
                if not title:
                    continue
                paper_obj = {'title': title}
                year = paper.get('year')
                if year not in (None, ''):
                    try:
                        paper_obj['year'] = int(year)
                    except (TypeError, ValueError):
                        paper_obj['year'] = str(year)
                url = str(paper.get('url', '')).strip()
                if url:
                    paper_obj['url'] = url
                venue = str(paper.get('venue', '')).strip()
                if venue:
                    paper_obj['venue'] = venue
                papers.append(paper_obj)
        result[dataset] = papers
    return result


def _resolve_dataset_info_csv_path(dataset_key: str, user=None):
    if parse_user_upload_dataset_key(dataset_key) is not None:
        rec = get_user_upload_by_dataset_key(dataset_key, user)
        if rec is None:
            return None
        path = os.path.join(
            user_upload_abs_dir(rec),
            dataset_info_filename(user_upload_slug(rec)),
        )
        return path if os.path.isfile(path) else None
    path = dataset_info_path(dataset_key)
    return path if os.path.isfile(path) else None


def _built_in_dataset_catalog_ready(dataset_key: str) -> bool:
    """Built-in catalog entries require ``{key}_dataset_info.csv``; skip others for now."""
    if parse_user_upload_dataset_key(dataset_key) is not None:
        return True
    return bool(_resolve_dataset_info_csv_path(dataset_key, None))


def _dataset_info_catalog_totals(dataset_key: str, user=None) -> tuple[int, int]:
    """
    ``(data_point_count, csv_size_bytes)`` from ``{key}_dataset_info.csv`` only.
    Used for Dataset page overview totals (no ``dataset_csv`` gate).
    """
    info_path = _resolve_dataset_info_csv_path(dataset_key, user)
    if not info_path:
        return 0, 0
    cached = load_dataset_info(info_path)
    if cached is None:
        return 0, 0
    return (
        int(cached.get('data_point_count', 0) or 0),
        int(cached.get('csv_size_bytes', 0) or 0),
    )


def _dataset_has_prebuilt_tsfile(dataset_key: str, user=None) -> bool:
    if parse_user_upload_dataset_key(dataset_key) is not None:
        rec = get_user_upload_by_dataset_key(dataset_key, user)
        return rec is not None and user_upload_has_prebuilt_tsfile(rec)
    return resolve_prebuilt_tsfile_path(dataset_key) is not None


def _dataset_counts_toward_catalog_size_stats(dataset_key: str, user=None) -> bool:
    """
    Overview CSV/TsFile size totals and category size pies.

    In the focused original-dataset mode, the Dataset page should summarize the
    selected CSVs directly even when no prebuilt TsFile bundle exists yet.
    """
    if not _resolve_dataset_info_csv_path(dataset_key, user):
        return False
    focused = _focused_original_dataset_keys()
    if focused and parse_user_upload_dataset_key(dataset_key) is None:
        return dataset_key in focused
    return _dataset_has_prebuilt_tsfile(dataset_key, user)


def _resolve_catalog_tsfile_info_path(dataset_key: str, user=None) -> str | None:
    """``{key}_catalog_tsfile_info.csv`` under catalog ``dataset/<key>/`` or user upload dir."""
    if parse_user_upload_dataset_key(dataset_key) is not None:
        rec = get_user_upload_by_dataset_key(dataset_key, user)
        if rec is None:
            return None
        from .upload_storage import user_upload_catalog_tsfile_info_path

        path = user_upload_catalog_tsfile_info_path(rec)
        return path if os.path.isfile(path) else None
    path = catalog_tsfile_info_path(dataset_key)
    if os.path.isfile(path):
        return path
    legacy = typed_tsfile_encoding_summary_path(dataset_key)
    return legacy if os.path.isfile(legacy) else None


def _tsfile_size_bytes_from_summary(dataset_key: str, user=None) -> int:
    """
    ``file_size_bytes`` from ``*_catalog_tsfile_info.csv`` (legacy encoding summary fallback).
    User uploads fall back to on-disk prebuilt ``.tsfile`` size when the info CSV was removed post-export.
    """
    info_path = _resolve_catalog_tsfile_info_path(dataset_key, user)
    if info_path:
        size = file_size_bytes_from_catalog_tsfile_info(info_path)
        if size > 0:
            return size
    if parse_user_upload_dataset_key(dataset_key) is not None:
        rec = get_user_upload_by_dataset_key(dataset_key, user)
        if rec is not None and user_upload_has_prebuilt_tsfile(rec):
            return user_upload_tsfile_total_bytes(rec)
    tsfile_path = resolve_prebuilt_tsfile_path(dataset_key)
    if tsfile_path:
        try:
            return int(os.path.getsize(tsfile_path))
        except OSError:
            pass
    return 0


def _dataset_stats(dataset_key: str, user=None):
    csv_path = _resolve_dataset_csv_path(dataset_key, user)
    # ``has_csv`` is only for download/export gating in the API/UI, not for row/size stats.
    has_csv = _csv_download_available(dataset_key, user)

    def _count_columns_from_header(path: str) -> int:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, [])
        return len(header)

    def _column_info_csv_path() -> str | None:
        if parse_user_upload_dataset_key(dataset_key) is not None:
            rec = get_user_upload_by_dataset_key(dataset_key, user)
            if rec is None:
                return None
            return resolve_user_upload_column_types_csv(rec)
        path = column_info_path(dataset_key)
        return path if os.path.isfile(path) else None

    col_info_path = _column_info_csv_path()
    cols = 0
    numeric_cols = 0
    if col_info_path:
        try:
            col_rows = load_column_info_rows(col_info_path)
            cols = len(col_rows)
            numeric_logicals = {'integer', 'int', 'long', 'float', 'double'}
            numeric_cols = sum(
                1
                for row in col_rows
                if str(row.get('logical_type', '')).strip().lower() in numeric_logicals
            )
        except Exception:
            cols = 0
            numeric_cols = 0
    elif csv_path:
        try:
            cols = _count_columns_from_header(csv_path)
        except Exception:
            cols = 0

    info_path = _resolve_dataset_info_csv_path(dataset_key, user)
    cached = load_dataset_info(info_path) if info_path else None
    try:
        if cached is not None:
            rows = int(cached['data_point_count'])
            size_bytes = int(cached['csv_size_bytes'])
        elif csv_path:
            rows = 0
            size_bytes = 0
        else:
            rows = 0
            size_bytes = 0
    except Exception:
        if cached is not None:
            rows = int(cached.get('data_point_count', 0) or 0)
            size_bytes = int(cached.get('csv_size_bytes', 0) or 0)
        elif csv_path:
            rows = 0
            try:
                size_bytes = int(os.path.getsize(csv_path))
            except OSError:
                size_bytes = 0
        else:
            rows = 0
            size_bytes = 0

    return {
        'row_count': int(rows),
        'column_count': int(cols),
        'numeric_column_count': numeric_cols,
        'numeric_points': int(rows) * numeric_cols,
        'size_bytes': int(size_bytes),
        'has_csv': has_csv,
    }


def _dataset_lossless_java_float_metrics(dataset_key: str, user_record: UploadedDataset | None = None):
    """
    Per dataset (Java float report only):
    - min compressedSize among lossless algorithms
    - best-lossless compression rate (min compressedSize / file_size)
    """
    if user_record is not None:
        report_path = user_upload_report_csv_path(user_record, 'float', 'java')
        slug = user_upload_slug(user_record)
        column_info_csv = os.path.join(user_upload_abs_dir(user_record), f'{slug}_column_info.csv')
    else:
        base = _dataset_bundle_dir(dataset_key)
        report_path = os.path.join(base, f'{dataset_key}_float_java_compression_result.csv')
        column_info_csv = os.path.join(base, f'{dataset_key}_column_info.csv')

    if not os.path.isfile(report_path):
        return {'min_compressed_size': None, 'avg_rate': None}

    file_size = _sum_column_csv_size_bytes(column_info_csv, 'float')
    if not (file_size > 0):
        return {'min_compressed_size': None, 'avg_rate': None}

    try:
        report_df = pd.read_csv(report_path)
    except Exception:
        return {'min_compressed_size': None, 'avg_rate': None}

    grouped = _aggregate_report_rows_by_algorithm(report_df)
    min_compressed = None
    for _, row in grouped.iterrows():
        algo = str(row.get('algorithm', '')).strip()
        if _is_lossy_algorithm_key(algo):
            continue
        try:
            compressed = float(row.get('compressedSize', 0))
        except (TypeError, ValueError):
            continue
        if compressed <= 0:
            continue
        if min_compressed is None or compressed < min_compressed:
            min_compressed = compressed
    best_rate = (min_compressed / file_size) if (min_compressed is not None and file_size > 0) else None

    return {
        'min_compressed_size': float(min_compressed) if min_compressed is not None else None,
        'avg_rate': float(best_rate) if best_rate is not None else None,
    }


# Java per-column report name for IoTDB TS_2DIFF (catalog compression-rate pie chart).
_CATALOG_CATEGORY_COMPRESSION_RATE_ALGORITHM = 'TS_2DIFF'


def _logical_type_to_java_report_dtype(logical_type: object) -> str | None:
    """Map column logical type to int/float Java report; string columns are excluded."""
    logical = _normalize_logical_type(logical_type)
    if logical in ('integer', 'int', 'long'):
        return 'int'
    if logical in ('float', 'double'):
        return 'float'
    return None


def _java_compression_report_path(
    dataset_key: str,
    dtype: str,
    *,
    user_record: UploadedDataset | None = None,
) -> str:
    if user_record is not None:
        return user_upload_report_csv_path(user_record, dtype, 'java')
    return os.path.join(_dataset_bundle_dir(dataset_key), f'{dataset_key}_{dtype}_java_compression_result.csv')


def _load_java_compression_report(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _compressed_size_for_column_algorithm(
    report_df: pd.DataFrame,
    column_index: int,
    algorithm: str,
) -> float | None:
    if report_df is None or report_df.empty:
        return None
    if 'column_index' not in report_df.columns or 'algorithm' not in report_df.columns:
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
    return compressed


def _iter_catalog_column_compression_rows(
    dataset_key: str,
    column_info_csv: str,
    *,
    user_record: UploadedDataset | None = None,
):
    """Yield ``(column_index, column_csv_size_bytes, java_report_df)`` per benchmarked column."""
    if not os.path.isfile(column_info_csv):
        return
    try:
        info_rows = load_column_info_rows(column_info_csv)
    except Exception:
        return
    report_cache: dict[str, pd.DataFrame] = {}
    for row in info_rows:
        dtype = _logical_type_to_java_report_dtype(row.get('logical_type', ''))
        if dtype is None:
            continue
        report_path = _java_compression_report_path(
            dataset_key, dtype, user_record=user_record,
        )
        if dtype not in report_cache:
            report_cache[dtype] = _load_java_compression_report(report_path)
        report_df = report_cache[dtype]
        if report_df.empty:
            continue
        try:
            column_index = int(row.get('column_index', -1))
            column_csv_size = float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
        if column_csv_size <= 0:
            continue
        yield column_index, column_csv_size, report_df


def _category_avg_lossless_compression_rates(
    catalog_entries: list[tuple[str, str, UploadedDataset | None]],
) -> dict[str, float]:
    """
    Per major category: ``sum(compressedSize) / sum(column_csv_size_bytes)`` using
    ``_CATALOG_CATEGORY_COMPRESSION_RATE_ALGORITHM`` on int/float columns only.
    """
    algorithm = _CATALOG_CATEGORY_COMPRESSION_RATE_ALGORITHM
    category_csv_sum: dict[str, float] = {}
    category_compressed_sum: dict[str, float] = {}
    for category, dataset_key, user_record in catalog_entries:
        col_info = (
            os.path.join(
                user_upload_abs_dir(user_record),
                f'{user_upload_slug(user_record)}_column_info.csv',
            )
            if user_record is not None
            else column_info_path(dataset_key)
        )
        for column_index, column_csv_size, report_df in _iter_catalog_column_compression_rows(
            dataset_key, col_info, user_record=user_record,
        ):
            compressed = _compressed_size_for_column_algorithm(
                report_df, column_index, algorithm,
            )
            if compressed is None:
                continue
            category_csv_sum[category] = category_csv_sum.get(category, 0.0) + float(
                column_csv_size
            )
            category_compressed_sum[category] = category_compressed_sum.get(
                category, 0.0,
            ) + float(compressed)
    return {
        name: category_compressed_sum[name] / category_csv_sum[name]
        for name in category_csv_sum
        if category_csv_sum.get(name, 0.0) > 0
    }


def _parse_metric_aggregation_mode(raw: object) -> str:
    mode = str(raw or 'overall').strip().lower()
    if mode in ('average', 'avg', 'mean'):
        return 'average'
    if mode in ('median', 'med'):
        return 'median'
    return 'overall'


def _parse_compression_rate_mode(raw: object) -> str:
    return _parse_metric_aggregation_mode(raw)


def _balanced_score_samples_from_dataset_rows(rows: pd.DataFrame) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    if rows is None or rows.empty:
        return samples
    needed = {
        'dataset_key',
        'algorithm',
        'compression_rate_value',
        'compression_time_value',
        'decompression_time_value',
    }
    if not needed.issubset(rows.columns):
        return samples
    for _dataset_key, group in rows.groupby('dataset_key'):
        usable = group[
            (pd.to_numeric(group['compression_rate_value'], errors='coerce') > 0)
            & (pd.to_numeric(group['compression_time_value'], errors='coerce') >= 0)
            & (pd.to_numeric(group['decompression_time_value'], errors='coerce') >= 0)
        ]
        if len(usable) < 2:
            continue
        from .home_leaderboard_boxplot_data import nonnegative_value_range, weighted_balanced_score

        comp_range = nonnegative_value_range(usable['compression_time_value'])
        decomp_range = nonnegative_value_range(usable['decompression_time_value'])
        for row in usable.itertuples(index=False):
            score = weighted_balanced_score(
                row.compression_rate_value,
                row.compression_time_value,
                row.decompression_time_value,
                comp_range,
                decomp_range,
            )
            if score is not None:
                samples.setdefault(str(row.algorithm), []).append(float(score))
    return samples


def _attach_balanced_scores_to_staticdata_items(
    items: list[dict],
    *,
    score_samples: dict[str, list[float]] | None = None,
) -> list[dict]:
    if not items:
        return items
    samples = score_samples or {}
    if samples:
        for item in items:
            vals = samples.get(str(item.get('algorithm')))
            if vals:
                item['balanced_score'] = float(median(vals))
        return items

    usable = []
    for item in items:
        try:
            rate = float(item.get('compression_rate'))
            comp_time = float(item.get('compression_time'))
            decomp_time = float(item.get('decompression_time'))
        except (TypeError, ValueError):
            continue
        if rate > 0 and comp_time >= 0 and decomp_time >= 0:
            usable.append((item, rate, comp_time, decomp_time))
    if len(usable) < 2:
        return items
    from .home_leaderboard_boxplot_data import nonnegative_value_range, weighted_balanced_score

    comp_range = nonnegative_value_range(comp_time for _item, _rate, comp_time, _decomp_time in usable)
    decomp_range = nonnegative_value_range(decomp_time for _item, _rate, _comp_time, decomp_time in usable)
    for item, rate, comp_time, decomp_time in usable:
        score = weighted_balanced_score(
            rate,
            comp_time,
            decomp_time,
            comp_range,
            decomp_range,
        )
        if score is not None:
            item['balanced_score'] = score
    return items


def _staticdata_items_from_metric_rows(
    filtered_df: pd.DataFrame,
    *,
    rate_mode: str = 'overall',
    compression_time_mode: str = 'overall',
    decompression_time_mode: str = 'overall',
    dataset_original_sizes: dict[str, float] | None = None,
) -> list[dict]:
    """Aggregate benchmark rows by algorithm with independent metric modes."""
    if filtered_df is None or filtered_df.empty or 'algorithm' not in filtered_df.columns:
        return []

    cur = filtered_df.copy()
    cur['algorithm'] = cur['algorithm'].astype(str)
    from .home_leaderboard_boxplot_data import is_home_leaderboard_algorithm_enabled

    cur = cur[cur['algorithm'].apply(is_home_leaderboard_algorithm_enabled)].copy()
    if cur.empty:
        return []
    has_dataset_key = 'dataset_key' in cur.columns
    if not has_dataset_key:
        cur['dataset_key'] = [f'__row_{i}' for i in range(len(cur))]
    cur['dataset_key'] = cur['dataset_key'].astype(str)

    for key in ('originalSize', 'compressedSize'):
        if key not in cur.columns:
            cur[key] = 0
        cur[key] = pd.to_numeric(cur[key], errors='coerce').fillna(0)
    # Timing availability is independent from a valid rate measurement.  Keep
    # missing timing as NaN so it is excluded from timing aggregation without
    # dropping the dataset's original/compressed byte counts or inventing a
    # zero-time result.
    for key in ('compressionTime', 'decompressionTime'):
        if key not in cur.columns:
            cur[key] = float('nan')
        cur[key] = pd.to_numeric(cur[key], errors='coerce')

    cur = cur[cur['compressedSize'] > 0]
    if cur.empty:
        return []

    by_dataset = (
        cur
        .groupby(['dataset_key', 'algorithm'], as_index=False)[
            ['originalSize', 'compressedSize', 'compressionTime', 'decompressionTime']
        ]
        .sum(min_count=1)
    )
    if dataset_original_sizes:
        def _original_for_dataset(row):
            if not has_dataset_key:
                return row['originalSize']
            value = dataset_original_sizes.get(str(row['dataset_key']))
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return row['originalSize']
            return numeric if numeric > 0 else row['originalSize']

        by_dataset['originalSize'] = by_dataset.apply(_original_for_dataset, axis=1)

    overall_original_size = None
    if dataset_original_sizes and not has_dataset_key:
        try:
            raw_original = dataset_original_sizes.get('__dataset__')
            overall_original_size = float(raw_original)
        except (TypeError, ValueError):
            overall_original_size = None
        if overall_original_size is not None and overall_original_size <= 0:
            overall_original_size = None

    by_dataset = by_dataset[by_dataset['compressedSize'] > 0].copy()
    if _parse_metric_aggregation_mode(rate_mode) != 'overall' or overall_original_size is None:
        by_dataset = by_dataset[by_dataset['originalSize'] > 0].copy()
    if by_dataset.empty:
        return []

    by_dataset['compression_rate_value'] = 0.0
    valid_original = by_dataset['originalSize'] > 0
    by_dataset.loc[valid_original, 'compression_rate_value'] = (
        by_dataset.loc[valid_original, 'compressedSize']
        / by_dataset.loc[valid_original, 'originalSize']
    )
    num_points = by_dataset['originalSize'] / 8.0
    valid_points = num_points > 0
    by_dataset = by_dataset[valid_points].copy()
    num_points = num_points[valid_points]
    if by_dataset.empty:
        return []
    by_dataset['compression_time_value'] = by_dataset['compressionTime'] / num_points
    by_dataset['decompression_time_value'] = by_dataset['decompressionTime'] / num_points
    if has_dataset_key:
        from .home_leaderboard_boxplot_data import boxplot_group_label

        by_dataset['family_label'] = by_dataset['algorithm'].apply(boxplot_group_label)
        by_dataset = by_dataset[by_dataset['family_label'].notna()].copy()
        if by_dataset.empty:
            return []
        best_idx = by_dataset.groupby(
            ['dataset_key', 'family_label'],
            sort=False,
        )['compression_rate_value'].idxmin()
        by_dataset = by_dataset.loc[best_idx].copy()
        by_dataset['algorithm'] = by_dataset['family_label'].astype(str)
    score_samples = (
        _balanced_score_samples_from_dataset_rows(by_dataset)
        if has_dataset_key
        else None
    )

    def _series_aggregate(values: pd.Series, mode: str) -> float:
        values = pd.to_numeric(values, errors='coerce').dropna()
        if values.empty:
            return float('nan')
        mode = _parse_metric_aggregation_mode(mode)
        if mode == 'median':
            return float(values.median())
        if mode == 'average':
            return float(values.mean())
        return float(values.sum())

    items: list[dict] = []
    for algorithm, group in by_dataset.groupby('algorithm'):
        try:
            compressed_sum = float(group['compressedSize'].sum())
            original_sum = (
                overall_original_size
                if overall_original_size is not None
                else float(group['originalSize'].sum())
            )
            if compressed_sum <= 0 or original_sum <= 0:
                continue
            if _parse_metric_aggregation_mode(rate_mode) == 'overall':
                compression_rate = compressed_sum / original_sum
            else:
                compression_rate = _series_aggregate(group['compression_rate_value'], rate_mode)

            compression_time = _series_aggregate(
                group['compression_time_value'],
                compression_time_mode,
            )
            decompression_time = _series_aggregate(
                group['decompression_time_value'],
                decompression_time_mode,
            )
            items.append({
                'algorithm': str(algorithm),
                'compression_rate': compression_rate,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'space_cost': int(compressed_sum),
            })
        except (TypeError, ValueError, KeyError):
            continue
    _attach_balanced_scores_to_staticdata_items(items, score_samples=score_samples)
    return _filter_staticdata_items_hide_lossy(items)


def _staticdata_items_average_rate(filtered_df: pd.DataFrame) -> list[dict]:
    """Per-column compressed/original, then arithmetic mean per algorithm."""
    return _staticdata_items_from_metric_rows(filtered_df, rate_mode='average')


def _aggregate_report_rows_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize per-column/per-algorithm report rows into one row per algorithm."""
    if df is None or df.empty or 'algorithm' not in df.columns:
        return pd.DataFrame(columns=['algorithm', 'originalSize', 'compressedSize', 'compressionTime', 'decompressionTime'])
    cur = df.copy()
    cur['algorithm'] = cur['algorithm'].astype(str)
    for key in ('originalSize', 'compressedSize', 'compressionTime', 'decompressionTime'):
        if key not in cur.columns:
            cur[key] = 0
        cur[key] = pd.to_numeric(cur[key], errors='coerce').fillna(0)
    grouped = (
        cur.groupby('algorithm', as_index=False)[
            ['originalSize', 'compressedSize', 'compressionTime', 'decompressionTime']
        ]
        .sum()
    )
    return grouped


def _normalize_logical_type(raw: object) -> str:
    return str(raw or '').strip().lower()


def _logical_matches_data_type(logical_type: object, data_type: str) -> bool:
    logical = _normalize_logical_type(logical_type)
    if data_type == 'overall':
        return logical in ('integer', 'int', 'long', 'float', 'double')
    if data_type == 'int':
        return logical in ('integer', 'int', 'long')
    if data_type == 'float':
        return logical in ('float', 'double')
    return False


def _staticdata_report_data_types(data_type: object) -> tuple[str, ...]:
    dtype = str(data_type or '').strip().lower()
    if dtype == 'overall':
        return _STATICDATA_BENCHMARK_DATA_TYPES
    if dtype in _STATICDATA_BENCHMARK_DATA_TYPES:
        return (dtype,)
    return ()


def _sum_column_csv_size_bytes(
    column_info_csv: str,
    data_type: str,
    selected_column_index: int | None = None,
    *,
    include_integer_for_float: bool = False,
    float_precision: str = 'all',
    bundle_dir: str | None = None,
    apply_float_precision_filter: bool = False,
) -> float:
    if not os.path.isfile(column_info_csv):
        return 0.0
    try:
        rows = load_column_info_rows(column_info_csv)
    except Exception:
        return 0.0
    allowed_float_cols: set[int] | None = None
    if (
        apply_float_precision_filter
        and data_type == 'float'
        and parse_float_precision_filter(float_precision) != 'all'
        and bundle_dir
    ):
        from .dataset_float_precision_distribution import allowed_float_column_indexes_for_filter

        allowed_float_cols = allowed_float_column_indexes_for_filter(
            column_info_csv,
            bundle_dir,
            float_precision,
        )
    total = 0.0
    for row in rows:
        logical = row.get('logical_type', '')
        if not _logical_matches_data_type(logical, data_type):
            if not (
                include_integer_for_float
                and data_type == 'float'
                and _normalize_logical_type(logical) in ('integer', 'int', 'long')
            ):
                continue
        if selected_column_index is not None:
            try:
                row_idx = int(row.get('column_index', -1))
            except (TypeError, ValueError):
                continue
            if row_idx != int(selected_column_index):
                continue
        if allowed_float_cols is not None:
            try:
                row_idx = int(row.get('column_index', -1))
            except (TypeError, ValueError):
                continue
            if row_idx not in allowed_float_cols:
                continue
        try:
            total += float(row.get('column_csv_size_bytes', 0) or 0)
        except (TypeError, ValueError):
            continue
    return float(total)


def _apply_dataset_level_float_precision_filter(
    df: pd.DataFrame,
    *,
    data_type: str,
    data_dimension: str,
    float_precision: str,
    column_info_path: str,
    bundle_dir: str,
) -> pd.DataFrame:
    if data_type != 'float' or data_dimension == 'single_column':
        return df
    return filter_report_dataframe_for_float_precision(
        df,
        column_info_path=column_info_path,
        bundle_dir=bundle_dir,
        float_precision=float_precision,
    )


def _max_original_size_from_report(df: pd.DataFrame) -> float:
    grouped = _aggregate_report_rows_by_algorithm(df)
    if grouped.empty or 'originalSize' not in grouped.columns:
        return 0.0
    try:
        return float(grouped['originalSize'].max())
    except (TypeError, ValueError, KeyError):
        return 0.0


def _filter_report_for_column(df: pd.DataFrame, column_index: int | None) -> pd.DataFrame:
    """Column-scoped view for single_column mode; returns empty frame when report lacks column_index."""
    if column_index is None:
        return df
    if df is None or df.empty:
        return df
    if 'column_index' not in df.columns:
        return df.iloc[0:0].copy()
    cur = df.copy()
    cur['column_index'] = pd.to_numeric(cur['column_index'], errors='coerce')
    return cur[cur['column_index'] == int(column_index)]


def _lite_catalog_item(key: str, meta: dict) -> dict:
    """Minimal catalog row for Home dropdowns (``?lite=1``); avoids per-dataset stats scans."""
    col_info_path = column_info_path(key)
    logical_types: list[str] = []
    benchmark_dtypes: list[str] = []
    catalog_dtypes: list[str] = []
    if os.path.isfile(col_info_path):
        logical_types = distinct_logical_types_from_column_info(col_info_path)
        benchmark_dtypes = benchmark_dtypes_from_column_info(col_info_path)
        catalog_dtypes = _focused_time_series_catalog_dtypes(
            key,
            catalog_dtypes_from_column_info(col_info_path),
        )
    return {
        'key': key,
        'category': meta.get('category', ''),
        'data_type': '',
        'logical_types': logical_types,
        'benchmark_dtypes': benchmark_dtypes,
        'catalog_dtypes': catalog_dtypes,
        'source': meta.get('source', key.replace('-', ' ')),
        'source_url': meta.get('source_url'),
    }


def _focused_time_series_catalog_dtypes(key: str, dtypes: list[str]) -> list[str]:
    """Catalog display dtypes for focused time-series datasets."""
    out = [dtype for dtype in dtypes if dtype in ('int', 'float', 'timestamp')]
    if key in _focused_original_dataset_keys() and 'timestamp' not in out:
        out.append('timestamp')
    order = {'int': 0, 'float': 1, 'timestamp': 2}
    return sorted(dict.fromkeys(out), key=lambda dtype: order.get(dtype, 99))


class DatasetCatalogView(APIView):
    """Dataset inventory + provenance mapping + summary numbers for frontend Dataset page."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        lite = parse_bool_form_value(request.query_params.get('lite'))
        include_user_uploads = parse_bool_form_value(
            request.query_params.get('include_user_uploads'),
        )
        part = parse_catalog_part(request.query_params.get('part'))
        user = (
            request.user
            if getattr(request, 'user', None) and request.user.is_authenticated
            else None
        )
        uploads_revision = ''
        if include_user_uploads and user is not None:
            uploads_revision = _user_uploads_catalog_revision(user)
        user_id = user.pk if user is not None else None

        focused_filter_enabled = _focused_dataset_filter_enabled()
        catalog_revision = uploads_revision
        if focused_filter_enabled:
            focused_revision = ','.join(sorted(_focused_original_dataset_keys()))
            catalog_revision = f'{catalog_revision}|focused:{focused_revision}'

        if part == 'summary':
            if not include_user_uploads and not lite:
                precomputed_summary = load_precomputed_catalog_summary()
                if precomputed_summary is not None:
                    return JsonResponse(
                        _sanitize_for_json({'summary': precomputed_summary}),
                        status=200,
                    )
                if catalog_summary_path().is_file():
                    logger.warning(
                        'Precomputed catalog summary missing or stale (version/schema); '
                        'falling back to live summary build.',
                    )
            cache_key = catalog_response_cache_key(
                lite,
                include_user_uploads,
                user_id,
                catalog_revision,
                'summary',
            )
            payload = get_or_build_catalog_response(
                cache_key,
                lambda: self._build_catalog_payload(
                    lite,
                    include_user_uploads,
                    user,
                    build_summary=True,
                    build_items=False,
                ),
            )
            return JsonResponse(
                _sanitize_for_json({'summary': payload.get('summary') or {}}),
                status=200,
            )

        if part == 'items':
            if not include_user_uploads and not lite and not focused_filter_enabled:
                precomputed_items = load_precomputed_catalog_items()
                if precomputed_items is not None:
                    return JsonResponse(
                        _sanitize_for_json({
                            'items': refresh_catalog_items_operators(
                                precomputed_items.get('items') or [],
                            ),
                            'paper_usage': precomputed_items.get('paper_usage') or [],
                        }),
                        status=200,
                    )
                if catalog_items_path().is_file():
                    logger.warning(
                        'Precomputed catalog items missing or stale (version/schema); '
                        'falling back to live items build.',
                    )
            cache_key = catalog_response_cache_key(
                lite,
                include_user_uploads,
                user_id,
                catalog_revision,
                'items',
            )
            payload = get_or_build_catalog_response(
                cache_key,
                lambda: self._build_catalog_payload(
                    lite,
                    include_user_uploads,
                    user,
                    build_summary=False,
                    build_items=True,
                ),
            )
            items = refresh_catalog_items_operators(payload.get('items') or [])
            return JsonResponse(
                _sanitize_for_json({
                    'items': items,
                    'paper_usage': payload.get('paper_usage') or [],
                }),
                status=200,
            )

        # part == 'full' (default; backward compatible)
        if not include_user_uploads and not lite and not focused_filter_enabled:
            precomputed_summary = load_precomputed_catalog_summary()
            precomputed_items = load_precomputed_catalog_items()
            if precomputed_summary is not None and precomputed_items is not None:
                return JsonResponse(
                    _sanitize_for_json({
                        'summary': precomputed_summary,
                        'items': refresh_catalog_items_operators(
                            precomputed_items.get('items') or [],
                        ),
                        'paper_usage': precomputed_items.get('paper_usage') or [],
                    }),
                    status=200,
                )
            if precomputed_summary is not None:
                items_key = catalog_response_cache_key(
                    lite,
                    False,
                    None,
                    '',
                    'items',
                )
                detail = get_or_build_catalog_response(
                    items_key,
                    lambda: self._build_catalog_payload(
                        lite,
                        include_user_uploads=False,
                        user=None,
                        build_summary=False,
                        build_items=True,
                    ),
                )
                return JsonResponse(
                    _sanitize_for_json({
                        'summary': precomputed_summary,
                        'items': detail['items'],
                        'paper_usage': detail['paper_usage'],
                    }),
                    status=200,
                )
            if catalog_summary_path().is_file():
                logger.warning(
                    'Precomputed catalog summary missing or stale (version/schema); '
                    'falling back to live summary build.',
                )
        cache_key = catalog_response_cache_key(
            lite,
            include_user_uploads,
            user_id,
            catalog_revision,
            'full',
        )
        payload = get_or_build_catalog_response(
            cache_key,
            lambda: self._build_catalog_payload(lite, include_user_uploads, user),
        )
        return JsonResponse(_sanitize_for_json(payload), status=200)

    def _build_catalog_payload(
        self,
        lite,
        include_user_uploads,
        user,
        *,
        build_summary: bool = True,
        build_items: bool = True,
        progress_label: str = '',
        on_dataset_progress: Callable[[str, int, int, str], None] | None = None,
    ):
        # Full catalog summary scans Java compression CSVs per dataset (seconds to a few
        # minutes depending on catalog size). ``build_summary=False`` skips that work when
        # overview/pie data is loaded from disk.
        items = []
        category_counter = {}
        category_size_bytes_sum = {}
        category_data_point_sum = {}
        category_tsfile_size_sum = {}
        dtype_column_counts: dict[str, float] = {}
        dtype_data_point_counts: dict[str, float] = {}
        dtype_csv_size_bytes: dict[str, float] = {}
        dtype_tsfile_size_bytes: dict[str, float] = {}
        dtype_compression_csv_sums, dtype_compression_compressed_sums = empty_dtype_rate_sums()
        float_precision_column_counts: dict[str, float] = {}
        float_precision_data_point_counts: dict[str, float] = {}
        float_precision_csv_size_bytes: dict[str, float] = {}
        float_precision_tsfile_size_bytes: dict[str, float] = {}
        (
            float_precision_compression_csv_sums,
            float_precision_compression_compressed_sums,
        ) = empty_float_precision_rate_sums()
        total_rows = 0
        total_bytes = 0
        total_tsfile_bytes = 0
        total_compressed_size_bytes = 0.0
        best_available_original_total = 0.0
        best_available_compressed_total = 0.0
        best_available_rate_sum = 0.0
        best_available_column_count = 0
        numeric_column_compression_rates: list[float] = []
        category_average_rate_lists: dict[str, list[float]] = {}
        dtype_average_rate_lists: dict[str, list[float]] = {}
        float_precision_average_rate_lists: dict[str, list[float]] = {}
        global_overall_csv_total: dict[str, float] = {}
        global_overall_compressed_total: dict[str, float] = {}
        category_overall_csv_sum: dict[str, float] = {}
        category_overall_compressed_sum: dict[str, float] = {}
        paper_usage_map = _load_dataset_paper_usage_map()
        paper_usage = []
        from .bench_executable_paths import benchmark_java_jar_path

        jar_path = str(benchmark_java_jar_path())
        has_java_export = os.path.isfile(jar_path)
        _, meta_map = _load_dataset_metadata_ordered()
        available_categories: set[str] = set()
        for key in _all_dataset_keys_for_ui():
            if not _built_in_dataset_catalog_ready(key):
                continue
            meta = meta_map.get(key)
            if meta is not None:
                cat = _major_category_from_meta(meta)
                if cat:
                    available_categories.add(cat)
        if include_user_uploads and user is not None:
            for rec in UploadedDataset.objects.filter(
                owner=user,
                processing_status=UploadedDataset.PROCESSING_COMPLETED,
            ):
                available_categories.add(normalize_major_category(rec.major_category))

        catalog_keys = [
            key for key in _all_dataset_keys_for_ui()
            if _built_in_dataset_catalog_ready(key)
        ]
        catalog_total = len(catalog_keys)
        catalog_index = 0

        for key in catalog_keys:
            catalog_index += 1
            if on_dataset_progress is not None:
                on_dataset_progress(
                    key,
                    catalog_index,
                    catalog_total,
                    progress_label or 'catalog',
                )
            meta = meta_map.get(key)
            if meta is None:
                meta = {
                    'key': key,
                    'category': '',
                    'data_type': '',
                    'source': key.replace('-', ' '),
                    'source_url': None,
                }
            major_category = _major_category_from_meta(meta)
            if build_summary or build_items:
                category_counter[major_category] = int(category_counter.get(major_category, 0)) + 1
            if lite:
                if build_items:
                    items.append(_lite_catalog_item(key, meta))
                continue

            if build_items:
                stats = _dataset_stats(key)
                has_prebuilt_tsfile = _dataset_has_prebuilt_tsfile(key)
                csv_download = stats['has_csv']
                has_bundle = catalog_has_statistics_bundle(key)
                asset_urls = None
                if has_bundle:
                    base = f'/api/datasets/catalog-assets/{key}'
                    available_operators = list_catalog_operator_ids(key)
                    asset_urls = {
                        'statistics_json': f'{base}/statistics-json/',
                        'correlation_heatmaps_json': f'{base}/correlation-heatmaps/',
                        'operator_step_entropy_json': f'{base}/operator-step-entropy/',
                        'numeric_statistic_csv': (
                            f'{base}/{catalog_asset_filename(key, "numeric_statistic")}'
                        ),
                        'column_types_csv': f'{base}/{catalog_asset_filename(key, "column_info")}',
                        'dataset_info_csv': f'{base}/{catalog_asset_filename(key, "dataset_info")}',
                        'available_operators': available_operators,
                    }
                prebuilt_tsfile_size = _tsfile_size_bytes_from_summary(key)
                logical_types = _catalog_column_logical_types(key)
                stats_public = {k: v for k, v in stats.items() if k != 'numeric_points'}
                col_info_path = column_info_path(key)
                benchmark_dtypes = (
                    benchmark_dtypes_from_column_info(col_info_path)
                    if os.path.isfile(col_info_path)
                    else []
                )
                catalog_dtypes = (
                    _focused_time_series_catalog_dtypes(
                        key,
                        catalog_dtypes_from_column_info(col_info_path),
                    )
                    if os.path.isfile(col_info_path)
                    else []
                )
            else:
                stats = None
                col_info_path = column_info_path(key)

            if build_summary:
                info_rows, info_size = _dataset_info_catalog_totals(key)
                counts_toward_size_stats = _dataset_counts_toward_catalog_size_stats(key)
                total_rows += info_rows
                if counts_toward_size_stats:
                    total_bytes += info_size
                prebuilt_tsfile_size = _tsfile_size_bytes_from_summary(key)
                if counts_toward_size_stats and prebuilt_tsfile_size > 0:
                    total_tsfile_bytes += prebuilt_tsfile_size
                if counts_toward_size_stats:
                    category_size_bytes_sum[major_category] = float(
                        category_size_bytes_sum.get(major_category, 0.0)
                    ) + float(info_size)
                category_data_point_sum[major_category] = float(
                    category_data_point_sum.get(major_category, 0.0)
                ) + float(info_rows)
                if counts_toward_size_stats and prebuilt_tsfile_size > 0:
                    category_tsfile_size_sum[major_category] = float(
                        category_tsfile_size_sum.get(major_category, 0.0)
                    ) + float(prebuilt_tsfile_size)
                if os.path.isfile(col_info_path):
                    if counts_toward_size_stats:
                        best_metrics = best_available_column_compression_metrics(
                            column_info_path=col_info_path,
                            dataset_key=key,
                            user_record=None,
                        )
                        best_original = float(best_metrics.get('original_size_bytes') or 0.0)
                        best_compressed = float(
                            best_metrics.get('compressed_size_bytes') or 0.0
                        )
                        best_columns = int(best_metrics.get('used_column_count') or 0)
                        if best_original > 0 and best_compressed > 0 and best_columns > 0:
                            total_compressed_size_bytes += best_compressed
                            best_available_original_total += best_original
                            best_available_compressed_total += best_compressed
                            best_available_rate_sum += float(
                                best_metrics.get('compression_rate_sum') or 0.0,
                            )
                            best_available_column_count += best_columns
                        ts2diff_rates = collect_numeric_column_ts2diff_rates(
                            column_info_path=col_info_path,
                            dataset_key=key,
                            user_record=None,
                        )
                        if ts2diff_rates:
                            numeric_column_compression_rates.extend(ts2diff_rates)
                            extend_numeric_ts2diff_rates_by_bucket(
                                column_info_path=col_info_path,
                                dataset_key=key,
                                user_record=None,
                                bucket_rates=category_average_rate_lists,
                                bucket_name=major_category,
                                rates=ts2diff_rates,
                            )
                    accumulate_dtype_metrics(
                        column_info_path=col_info_path,
                        dataset_info_path=dataset_info_path(key),
                        bundle_dir=_dataset_bundle_dir(key),
                        dataset_key=key,
                        column_counts=dtype_column_counts,
                        data_point_counts=dtype_data_point_counts,
                        csv_size_bytes=dtype_csv_size_bytes,
                        tsfile_size_bytes=dtype_tsfile_size_bytes,
                        force_one_timestamp_axis=(key in _focused_original_dataset_keys()),
                    )
                    accumulate_float_precision_metrics(
                        column_info_path=col_info_path,
                        dataset_info_path=dataset_info_path(key),
                        bundle_dir=_dataset_bundle_dir(key),
                        dataset_key=key,
                        column_counts=float_precision_column_counts,
                        data_point_counts=float_precision_data_point_counts,
                        csv_size_bytes=float_precision_csv_size_bytes,
                        tsfile_size_bytes=float_precision_tsfile_size_bytes,
                    )
                    try_accumulate_dataset_overall_compression_rate_sums(
                        column_info_path=col_info_path,
                        dataset_key=key,
                        user_record=None,
                        bundle_dir=_dataset_bundle_dir(key),
                        category_name=major_category,
                        dtype_csv_sums=dtype_compression_csv_sums,
                        dtype_compressed_sums=dtype_compression_compressed_sums,
                        float_precision_csv_sums=float_precision_compression_csv_sums,
                        float_precision_compressed_sums=(
                            float_precision_compression_compressed_sums
                        ),
                        category_overall_csv_sum=category_overall_csv_sum,
                        category_overall_compressed_sum=category_overall_compressed_sum,
                        global_overall_csv_total=global_overall_csv_total,
                        global_overall_compressed_total=global_overall_compressed_total,
                    )
                    if counts_toward_size_stats:
                        extend_column_average_rates_by_dtype(
                            column_info_path=col_info_path,
                            dataset_key=key,
                            user_record=None,
                            dtype_rates=dtype_average_rate_lists,
                        )
                    if counts_toward_size_stats:
                        extend_float_ts2diff_rates_by_precision(
                            column_info_path=col_info_path,
                            bundle_dir=_dataset_bundle_dir(key),
                            dataset_key=key,
                            user_record=None,
                            precision_rates=float_precision_average_rate_lists,
                        )

            if build_items:
                items.append({
                    'key': key,
                    'category': meta['category'],
                    'data_type': '',
                    'logical_types': logical_types,
                    'benchmark_dtypes': benchmark_dtypes,
                    'catalog_dtypes': catalog_dtypes,
                    **_float_precision_fields_for_catalog_item(
                        lite=lite,
                        column_info_path=col_info_path,
                        bundle_dir=_dataset_bundle_dir(key),
                    ),
                    'source': meta['source'],
                    'source_url': meta['source_url'],
                    **stats_public,
                    'has_asset_bundle': has_bundle,
                    'asset_urls': asset_urls,
                    'prebuilt_tsfile_size_bytes': prebuilt_tsfile_size,
                    'can_export_tsfile': bool(
                        has_prebuilt_tsfile or (csv_download and has_java_export)
                    ),
                    'csv_download_url': f'/api/datasets/download/{key}/csv/',
                    'tsfile_download_url': f'/api/datasets/download/{key}/tsfile/',
                })
                paper_usage.append({
                    'dataset': key,
                    'papers': paper_usage_map.get(key, []),
                })

        if include_user_uploads and user is not None:
            for rec in UploadedDataset.objects.filter(
                owner=user,
                processing_status=UploadedDataset.PROCESSING_COMPLETED,
            ).order_by('-created_at'):
                key = user_upload_dataset_key(rec)
                stats = _dataset_stats(key, user)
                upload_rows, upload_size = _dataset_info_catalog_totals(key, user)
                upload_counts_toward_size_stats = _dataset_counts_toward_catalog_size_stats(
                    key, user,
                )
                logical_types = user_upload_logical_types(rec)
                major_category = normalize_major_category(rec.major_category)
                upload_slug = user_upload_slug(rec)
                upload_dir = user_upload_abs_dir(rec)
                upload_col_info = os.path.join(upload_dir, f'{upload_slug}_column_info.csv')
                upload_benchmark_dtypes = (
                    benchmark_dtypes_from_column_info(upload_col_info)
                    if os.path.isfile(upload_col_info)
                    else []
                )
                upload_catalog_dtypes = (
                    catalog_dtypes_from_column_info(upload_col_info)
                    if os.path.isfile(upload_col_info)
                    else []
                )
                category_counter[major_category] = int(category_counter.get(major_category, 0)) + 1
                if lite:
                    items.append({
                        'key': key,
                        'display_name': rec.display_name,
                        'is_public': bool(rec.is_public),
                        'category': major_category,
                        'data_type': '',
                        'logical_types': logical_types,
                        'benchmark_dtypes': upload_benchmark_dtypes,
                        'catalog_dtypes': upload_catalog_dtypes,
                        'source': '',
                        'source_url': None,
                        'is_user_upload': True,
                    })
                    continue
                if upload_counts_toward_size_stats:
                    category_size_bytes_sum[major_category] = float(
                        category_size_bytes_sum.get(major_category, 0.0)
                    ) + float(upload_size)
                category_data_point_sum[major_category] = float(
                    category_data_point_sum.get(major_category, 0.0)
                ) + float(upload_rows)
                prebuilt_tsfile_size = _tsfile_size_bytes_from_summary(key, user)
                if upload_counts_toward_size_stats and prebuilt_tsfile_size > 0:
                    category_tsfile_size_sum[major_category] = float(
                        category_tsfile_size_sum.get(major_category, 0.0)
                    ) + float(prebuilt_tsfile_size)
                if not lite:
                    if os.path.isfile(upload_col_info):
                        if upload_counts_toward_size_stats:
                            upload_best_metrics = best_available_column_compression_metrics(
                                column_info_path=upload_col_info,
                                dataset_key=upload_slug,
                                user_record=rec,
                            )
                            upload_best_original = float(
                                upload_best_metrics.get('original_size_bytes') or 0.0
                            )
                            upload_best_compressed = float(
                                upload_best_metrics.get('compressed_size_bytes') or 0.0
                            )
                            upload_best_columns = int(
                                upload_best_metrics.get('used_column_count') or 0
                            )
                            if (
                                upload_best_original > 0
                                and upload_best_compressed > 0
                                and upload_best_columns > 0
                            ):
                                total_compressed_size_bytes += upload_best_compressed
                                best_available_original_total += upload_best_original
                                best_available_compressed_total += upload_best_compressed
                                best_available_rate_sum += float(
                                    upload_best_metrics.get('compression_rate_sum') or 0.0,
                                )
                                best_available_column_count += upload_best_columns
                            upload_ts2diff_rates = collect_numeric_column_ts2diff_rates(
                                column_info_path=upload_col_info,
                                dataset_key=upload_slug,
                                user_record=rec,
                            )
                            if upload_ts2diff_rates:
                                numeric_column_compression_rates.extend(upload_ts2diff_rates)
                                extend_numeric_ts2diff_rates_by_bucket(
                                    column_info_path=upload_col_info,
                                    dataset_key=upload_slug,
                                    user_record=rec,
                                    bucket_rates=category_average_rate_lists,
                                    bucket_name=major_category,
                                    rates=upload_ts2diff_rates,
                                )
                        try_accumulate_dataset_overall_compression_rate_sums(
                            column_info_path=upload_col_info,
                            dataset_key=upload_slug,
                            user_record=rec,
                            bundle_dir=upload_dir,
                            category_name=major_category,
                            dtype_csv_sums=dtype_compression_csv_sums,
                            dtype_compressed_sums=dtype_compression_compressed_sums,
                            float_precision_csv_sums=float_precision_compression_csv_sums,
                            float_precision_compressed_sums=(
                                float_precision_compression_compressed_sums
                            ),
                            category_overall_csv_sum=category_overall_csv_sum,
                            category_overall_compressed_sum=category_overall_compressed_sum,
                            global_overall_csv_total=global_overall_csv_total,
                            global_overall_compressed_total=global_overall_compressed_total,
                        )
                        if upload_counts_toward_size_stats:
                            extend_column_average_rates_by_dtype(
                                column_info_path=upload_col_info,
                                dataset_key=upload_slug,
                                user_record=rec,
                                dtype_rates=dtype_average_rate_lists,
                            )
                        if upload_counts_toward_size_stats:
                            extend_float_ts2diff_rates_by_precision(
                                column_info_path=upload_col_info,
                                bundle_dir=upload_dir,
                                dataset_key=upload_slug,
                                user_record=rec,
                                precision_rates=float_precision_average_rate_lists,
                            )
                total_rows += upload_rows
                if upload_counts_toward_size_stats:
                    total_bytes += upload_size
                asset_urls = build_user_upload_asset_urls(key, rec)
                has_bundle = user_upload_has_asset_bundle(rec)
                upload_csv_download = stats['has_csv']
                can_tsfile = bool(
                    user_upload_has_prebuilt_tsfile(rec)
                    or (upload_csv_download and has_java_export)
                )
                if upload_counts_toward_size_stats and prebuilt_tsfile_size > 0:
                    total_tsfile_bytes += prebuilt_tsfile_size
                items.append({
                    'key': key,
                    'display_name': rec.display_name,
                    'is_public': bool(rec.is_public),
                    'category': major_category,
                    'data_type': '',
                    'logical_types': logical_types,
                    'benchmark_dtypes': upload_benchmark_dtypes,
                    'catalog_dtypes': upload_catalog_dtypes,
                    **_float_precision_fields_for_catalog_item(
                        lite=lite,
                        column_info_path=upload_col_info,
                        bundle_dir=upload_dir,
                    ),
                    'source': '',
                    'source_url': None,
                    **{k: v for k, v in stats.items() if k != 'numeric_points'},
                    'has_asset_bundle': has_bundle,
                    'asset_urls': asset_urls,
                    'prebuilt_tsfile_size_bytes': prebuilt_tsfile_size,
                    'can_export_tsfile': can_tsfile,
                    'csv_download_url': f'/api/datasets/download/{key}/csv/',
                    'tsfile_download_url': f'/api/datasets/download/{key}/tsfile/',
                    'is_user_upload': True,
                })
                if not lite:
                    accumulate_dtype_metrics(
                        column_info_path=upload_col_info,
                        dataset_info_path=os.path.join(upload_dir, f'{upload_slug}_dataset_info.csv'),
                        bundle_dir=upload_dir,
                        dataset_key=upload_slug,
                        column_counts=dtype_column_counts,
                        data_point_counts=dtype_data_point_counts,
                        csv_size_bytes=dtype_csv_size_bytes,
                        tsfile_size_bytes=dtype_tsfile_size_bytes,
                    )
                    accumulate_float_precision_metrics(
                        column_info_path=upload_col_info,
                        dataset_info_path=os.path.join(upload_dir, f'{upload_slug}_dataset_info.csv'),
                        bundle_dir=upload_dir,
                        dataset_key=upload_slug,
                        column_counts=float_precision_column_counts,
                        data_point_counts=float_precision_data_point_counts,
                        csv_size_bytes=float_precision_csv_size_bytes,
                        tsfile_size_bytes=float_precision_tsfile_size_bytes,
                    )
                paper_usage.append({'dataset': key, 'papers': []})

        if build_items:
            total_datasets = len(items)
        else:
            total_datasets = int(sum(category_counter.values()))
        if not build_summary:
            return {
                'summary': {},
                'items': items,
                'paper_usage': paper_usage,
            }
        category_distribution = []
        if total_datasets > 0:
            for name, cnt in sorted(category_counter.items(), key=lambda kv: (-kv[1], kv[0])):
                category_distribution.append({
                    'category': name,
                    'count': int(cnt),
                    'value': float(cnt),
                    'percentage': (float(cnt) * 100.0 / float(total_datasets)),
                })
        if lite:
            category_size_distribution = []
            category_data_point_distribution = []
            category_tsfile_size_distribution = []
            category_overall_compression_rate_distribution = []
            category_average_compression_rate_distribution = []
            overall_compression_rate = None
            average_compression_rate = None
            total_compression_rate = None
            dtype_column_distribution = []
            dtype_data_point_distribution = []
            dtype_csv_size_distribution = []
            dtype_tsfile_size_distribution = []
            dtype_overall_compression_rate_distribution = []
            dtype_average_compression_rate_distribution = []
            float_precision_column_distribution = []
            float_precision_data_point_distribution = []
            float_precision_csv_size_distribution = []
            float_precision_tsfile_size_distribution = []
            float_precision_overall_compression_rate_distribution = []
            float_precision_average_compression_rate_distribution = []
        else:
            category_size_distribution = _distribution_rows_from_values(
                category_size_bytes_sum,
                value_field='size_bytes',
            )
            category_data_point_distribution = _distribution_rows_from_values(
                category_data_point_sum,
                value_field='data_points',
            )
            category_tsfile_size_distribution = _distribution_rows_from_values(
                category_tsfile_size_sum,
                value_field='tsfile_size_bytes',
            )
            category_overall_compression_rate_values = compression_rates_from_sums(
                category_overall_csv_sum,
                category_overall_compressed_sum,
            )
            category_overall_compression_rate_distribution = _distribution_rows_from_values(
                category_overall_compression_rate_values,
                value_field='compression_rate',
            )
            category_average_compression_rate_distribution = _distribution_rows_from_values(
                average_compression_rates_by_bucket(category_average_rate_lists),
                value_field='compression_rate',
            )
            fixed_overall_compression_rate = global_compression_rate(
                global_overall_csv_total,
                global_overall_compressed_total,
            )
            fixed_average_compression_rate = average_column_compression_rate(
                numeric_column_compression_rates,
            )
            if best_available_original_total > 0 and best_available_compressed_total > 0:
                overall_compression_rate = (
                    best_available_compressed_total / best_available_original_total
                )
            else:
                overall_compression_rate = fixed_overall_compression_rate
            if best_available_column_count > 0:
                average_compression_rate = (
                    best_available_rate_sum / float(best_available_column_count)
                )
            else:
                average_compression_rate = fixed_average_compression_rate
            total_compression_rate = overall_compression_rate
            dtype_column_distribution = _distribution_rows_from_values(
                dtype_column_counts,
                value_field='count',
            )
            dtype_data_point_distribution = _distribution_rows_from_values(
                dtype_data_point_counts,
                value_field='data_points',
            )
            dtype_csv_size_distribution = _distribution_rows_from_values(
                dtype_csv_size_bytes,
                value_field='csv_size_bytes',
            )
            dtype_tsfile_size_distribution = _distribution_rows_from_values(
                dtype_tsfile_size_bytes,
                value_field='tsfile_size_bytes',
            )
            dtype_overall_compression_rate_distribution = _distribution_rows_from_values(
                compression_rates_from_sums(
                    dtype_compression_csv_sums,
                    dtype_compression_compressed_sums,
                ),
                value_field='compression_rate',
            )
            dtype_average_compression_rate_distribution = _distribution_rows_from_values(
                average_compression_rates_by_bucket(dtype_average_rate_lists),
                value_field='compression_rate',
            )
            float_precision_column_distribution = _distribution_rows_from_values(
                float_precision_column_counts,
                value_field='count',
            )
            float_precision_data_point_distribution = _distribution_rows_from_values(
                float_precision_data_point_counts,
                value_field='data_points',
            )
            float_precision_csv_size_distribution = _distribution_rows_from_values(
                float_precision_csv_size_bytes,
                value_field='csv_size_bytes',
            )
            float_precision_tsfile_size_distribution = _distribution_rows_from_values(
                float_precision_tsfile_size_bytes,
                value_field='tsfile_size_bytes',
            )
            float_precision_overall_compression_rate_distribution = _distribution_rows_from_values(
                compression_rates_from_sums(
                    float_precision_compression_csv_sums,
                    float_precision_compression_compressed_sums,
                ),
                value_field='compression_rate',
            )
            float_precision_average_compression_rate_distribution = _distribution_rows_from_values(
                average_compression_rates_by_bucket(float_precision_average_rate_lists),
                value_field='compression_rate',
            )

        return {
            'summary': {
                'dataset_count': total_datasets,
                'total_rows': total_rows,
                'total_size_bytes': total_bytes,
                'total_tsfile_size_bytes': int(round(total_compressed_size_bytes)),
                'total_compressed_size_bytes': int(round(total_compressed_size_bytes)),
                'total_catalog_tsfile_size_bytes': total_tsfile_bytes,
                'overall_compression_rate': overall_compression_rate,
                'average_compression_rate': average_compression_rate,
                'total_compression_rate': total_compression_rate,
                'category_distribution': category_distribution,
                'category_size_distribution': category_size_distribution,
                'category_data_point_distribution': category_data_point_distribution,
                'category_tsfile_size_distribution': category_tsfile_size_distribution,
                'category_overall_compression_rate_distribution': (
                    category_overall_compression_rate_distribution
                ),
                'category_average_compression_rate_distribution': (
                    category_average_compression_rate_distribution
                ),
                'dtype_column_distribution': dtype_column_distribution,
                'dtype_data_point_distribution': dtype_data_point_distribution,
                'dtype_csv_size_distribution': dtype_csv_size_distribution,
                'dtype_tsfile_size_distribution': dtype_tsfile_size_distribution,
                'dtype_overall_compression_rate_distribution': (
                    dtype_overall_compression_rate_distribution
                ),
                'dtype_average_compression_rate_distribution': (
                    dtype_average_compression_rate_distribution
                ),
                'float_precision_column_distribution': float_precision_column_distribution,
                'float_precision_data_point_distribution': (
                    float_precision_data_point_distribution
                ),
                'float_precision_csv_size_distribution': (
                    float_precision_csv_size_distribution
                ),
                'float_precision_tsfile_size_distribution': (
                    float_precision_tsfile_size_distribution
                ),
                'float_precision_overall_compression_rate_distribution': (
                    float_precision_overall_compression_rate_distribution
                ),
                'float_precision_average_compression_rate_distribution': (
                    float_precision_average_compression_rate_distribution
                ),
                'available_categories': sorted(available_categories),
            },
            'items': items,
            'paper_usage': paper_usage,
        }


def _user_uploads_catalog_revision(user) -> str:
    from django.db.models import Count, Max

    row = UploadedDataset.objects.filter(
        owner=user,
        processing_status=UploadedDataset.PROCESSING_COMPLETED,
    ).aggregate(updated=Max('updated_at'), count=Count('id'))
    updated = row.get('updated')
    stamp = updated.isoformat() if updated is not None else ''
    return f"{int(row.get('count') or 0)}:{stamp}"


class MethodCatalogView(APIView):
    """Method metadata list for the Methods page."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        path = _method_metadata_path()
        encoding_operator_rows = load_method_page_operators()
        encoding_operators = [
            operator_row_for_api(row) for row in encoding_operator_rows
        ]
        byte_codec_operators = [
            operator_row_for_api(row) for row in load_byte_codec_operators()
        ]
        valid_operator_ids = frozenset(row['id'] for row in encoding_operator_rows)
        pipeline_overlay = load_method_operator_pipelines()
        if not path.is_file():
            return JsonResponse(
                {
                    'items': [],
                    'active_method_policy': _sanitize_for_json(active_method_policy_summary()),
                    'encoding_operators': _sanitize_for_json(encoding_operators),
                    'byte_codec_operators': _sanitize_for_json(byte_codec_operators),
                },
                status=200,
            )
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            logger.exception('Failed to parse method metadata file: %s', path)
            return JsonResponse(
                {
                    'items': [],
                    'active_method_policy': _sanitize_for_json(active_method_policy_summary()),
                    'encoding_operators': _sanitize_for_json(encoding_operators),
                    'byte_codec_operators': _sanitize_for_json(byte_codec_operators),
                },
                status=200,
            )

        items = []
        for row in payload.get('methods', []):
            if not isinstance(row, dict):
                continue
            name = str(row.get('name', '')).strip()
            if not name or not is_active_method_name(name):
                continue
            data_types = row.get('data_types', [])
            if isinstance(data_types, str):
                data_types = [t.strip() for t in data_types.split(',') if t.strip()]
            elif isinstance(data_types, list):
                data_types = [str(t).strip() for t in data_types if str(t).strip()]
            else:
                data_types = []
            inline_pipeline = row.get('operator_pipeline')
            if inline_pipeline is not None:
                operator_pipeline = inline_pipeline
            else:
                operator_pipeline = pipeline_overlay.get(name, [])
            primary_dtype = data_types[0].lower() if data_types else ''
            items.append(
                {
                    'name': name,
                    'paper': str(row.get('paper', '')).strip(),
                    'paper_url': str(row.get('paper_url', '')).strip(),
                    'description': str(row.get('description', '')).strip(),
                    'compression_mode': compression_mode_for_method_name(name),
                    'data_types': data_types,
                    'operator_pipeline': operator_pipeline,
                    'operator_steps': resolve_method_operator_steps(
                        name,
                        row_pipeline=operator_pipeline,
                        valid_operator_ids=valid_operator_ids,
                        data_type=primary_dtype,
                    ),
                }
            )
        items.sort(key=lambda row: row['name'].casefold())
        if HIDE_LOSSY_ALGORITHMS:
            items = [row for row in items if row.get('compression_mode') != 'lossy']
        return JsonResponse(
            {
                'items': _sanitize_for_json(items),
                'active_method_policy': _sanitize_for_json(active_method_policy_summary()),
                'encoding_operators': _sanitize_for_json(encoding_operators),
                'byte_codec_operators': _sanitize_for_json(byte_codec_operators),
            },
            status=200,
        )


class EncodingOperatorCatalogView(APIView):
    """Operator metadata for Methods and Dataset pages."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        operators = [
            operator_row_for_api(row)
            for row in load_method_page_operators()
        ]
        return JsonResponse({'operators': _sanitize_for_json(operators)}, status=200)


class DatasetCsvDownloadView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, dataset_key, *args, **kwargs):
        user_record = get_user_upload_by_dataset_key(dataset_key, request.user)
        csv_path = _resolve_dataset_csv_path(dataset_key, request.user)
        if not csv_path:
            if dataset_key not in _all_dataset_keys_for_ui() and user_record is None:
                raise Http404('Dataset not found')
            raise Http404('CSV file is not available on this deployment')
        download_name = f'{dataset_key}.csv'
        if user_record is not None:
            download_name = f'{user_record.display_name}.csv'
        response = FileResponse(open(csv_path, 'rb'), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{download_name}"'
        return response


def _iter_tsfile_zip_entries(root: str, arc_prefix: str = '') -> list[tuple[str, str]]:
    """Collect (absolute_path, zip_arcname) for each ``.tsfile`` under ``root``."""
    entries: list[tuple[str, str]] = []
    if not root or not os.path.isdir(root):
        return entries
    root_real = os.path.realpath(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith('.tsfile'):
                continue
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, root_real).replace(os.sep, '/')
            arc = f'{arc_prefix}/{rel}' if arc_prefix else rel
            entries.append((path, arc))
    return entries


def _http_zip_tsfile_bundle(entries: list[tuple[str, str]], download_filename: str) -> HttpResponse:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in entries:
            zf.write(path, arcname=arcname)
    response = HttpResponse(buf.getvalue(), content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename="{download_filename}"'
    return response


def _curated_dataset_tsfile_zip_entries(dataset_key: str) -> list[tuple[str, str]]:
    """Per-algorithm TsFiles under ``dataset_tsfile/.../tsfile_artifacts/`` plus catalog prebuilt."""
    entries = _iter_tsfile_zip_entries(
        os.path.join(tsfile_bundle_dir(dataset_key), 'tsfile_artifacts'),
        'tsfile_artifacts',
    )
    prebuilt = _prebuilt_typed_tsfile_path(dataset_key)
    if os.path.isfile(prebuilt):
        arc = f'{dataset_key}.tsfile'
        if not any(a == arc for _, a in entries):
            entries.append((prebuilt, arc))
    return entries


def _user_upload_tsfile_zip_entries(user_record: UploadedDataset) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for dtype in ('int', 'float'):
        entries.extend(
            _iter_tsfile_zip_entries(
                user_upload_tsfile_artifact_dir(user_record, dtype),
                f'tsfile_artifacts/{dtype}',
            )
        )
    prebuilt = resolve_user_upload_prebuilt_tsfile_path(user_record)
    if prebuilt:
        arc = f'{user_record.display_name}.tsfile'
        if not any(a == arc for _, a in entries):
            entries.append((prebuilt, arc))
    return entries


class DatasetTsFileDownloadView(APIView):
    """ZIP of TsFile artifacts (and catalog prebuilt when present); on-demand Java export if none on disk."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def _download_user_upload_tsfile_zip(self, request, dataset_key: str):
        user_record = get_user_upload_by_dataset_key(
            dataset_key, request.user, owner_only=True,
        )
        if user_record is None:
            raise Http404('Dataset not found')
        if user_record.processing_status != UploadedDataset.PROCESSING_COMPLETED:
            raise Http404('TSFile export is not ready for this upload')
        entries = _user_upload_tsfile_zip_entries(user_record)
        if not entries:
            raise Http404(
                'No TsFile artifacts for this upload. '
                'Re-upload or wait for background processing to finish.'
            )
        slug = user_record.display_name
        return _http_zip_tsfile_bundle(entries, f'{slug}_tsfile_bundle.zip')

    def get(self, request, dataset_key, *args, **kwargs):
        if parse_user_upload_dataset_key(dataset_key) is not None:
            return self._download_user_upload_tsfile_zip(request, dataset_key)

        if dataset_key not in _all_dataset_keys_for_ui():
            raise Http404('Dataset not found')

        zip_entries = _curated_dataset_tsfile_zip_entries(dataset_key)
        if zip_entries:
            return _http_zip_tsfile_bundle(zip_entries, f'{dataset_key}_tsfile_bundle.zip')

        csv_path = _resolve_dataset_csv_path(dataset_key, request.user)
        if not csv_path:
            raise Http404('CSV source is not available for this dataset')

        from .bench_executable_paths import benchmark_java_jar_path

        jar = benchmark_java_jar_path()
        if not jar.is_file():
            return JsonResponse(
                {
                    'error': (
                        'Java benchmark JAR missing. Build: cd java && mvn package -DskipTests '
                        f'(expected java/target/ or backend/lib/{jar.name})'
                    ),
                },
                status=500,
            )
        jar_path = str(jar)

        column_info = column_info_path(dataset_key)
        if not os.path.isfile(column_info):
            return JsonResponse(
                {
                    'error': (
                        f'Missing {os.path.basename(column_info)} for this dataset; '
                        'run regenerate_dataset_info.py first.'
                    ),
                },
                status=500,
            )

        try:
            from .column_info_loader import bench_headers_for_data_type

            with tempfile.TemporaryDirectory(prefix=f'tsfile_export_{dataset_key}_') as export_root:
                generated_files = []
                for dtype in ('int', 'float'):
                    if not bench_headers_for_data_type(column_info, dtype):
                        continue
                    tsfile_dir = os.path.join(export_root, f'tsfiles_{dtype}')
                    os.makedirs(tsfile_dir, exist_ok=True)
                    report_path = os.path.join(
                        export_root,
                        f'{dataset_key}_{dtype}_tsfile_compression_result.csv',
                    )

                    java_cmd = _java_benchmark_command(
                        jar_path,
                        '--input',
                        csv_path,
                        '--data-type',
                        dtype,
                        '--column-info',
                        column_info,
                        '--tsfile-dir',
                        tsfile_dir,
                        '--output-report',
                        report_path,
                    )
                    subprocess.run(java_cmd, check=True, timeout=None, env=_java_bench_subprocess_env())

                    for root, _, files in os.walk(tsfile_dir):
                        for name in files:
                            generated_files.append(os.path.join(root, name))

                if not generated_files:
                    return JsonResponse({'error': 'No TSFile artifacts were generated for this dataset'}, status=500)

                buf = io.BytesIO()
                with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for path in generated_files:
                        arcname = os.path.relpath(path, export_root)
                        zf.write(path, arcname=arcname)
                payload = buf.getvalue()
                response = HttpResponse(payload, content_type='application/zip')
                response['Content-Disposition'] = f'attachment; filename="{dataset_key}_tsfile_bundle.zip"'
                return response
        except subprocess.CalledProcessError as e:
            logger.exception('Failed to export dataset TSFile bundle for %s', dataset_key)
            return JsonResponse({'error': f'TSFile export failed: {e}'}, status=500)


def _catalog_asset_data_type(request) -> str:
    raw = (request.query_params.get('data_type') or '').strip().lower()
    return raw if raw in ('int', 'float') else ''


def _parse_catalog_operator(request) -> str:
    raw = (request.query_params.get('operator') or '').strip()
    if not raw:
        return ''
    return raw if raw in SUPPORTED_OPERATORS else ''


def _resolve_catalog_column_info_and_bundle(dataset_key: str, user):
    if dataset_key in _discovered_catalog_bundle_keys():
        path_csv = column_info_path(dataset_key)
        if not os.path.isfile(path_csv):
            legacy = os.path.join(_dataset_bundle_dir(dataset_key), 'column_types.csv')
            path_csv = legacy if os.path.isfile(legacy) else path_csv
        return path_csv, _dataset_bundle_dir(dataset_key)
    record = get_user_upload_by_dataset_key(dataset_key, user)
    if record is None:
        return None, None
    upload_dir = user_upload_abs_dir(record)
    slug = user_upload_slug(record)
    return os.path.join(upload_dir, f'{slug}_column_info.csv'), upload_dir


def _resolve_builtin_master_csv_for_operator_assets(dataset_key: str) -> str | None:
    master_csv = resolve_master_csv_path(dataset_key)
    if master_csv:
        return master_csv
    local_original = Path(_repo_root()) / 'orignial_datasets' / f'{dataset_key}.csv'
    return str(local_original) if local_original.is_file() else None


def _resolve_catalog_operator_asset_context(dataset_key: str, user):
    if dataset_key in _discovered_catalog_bundle_keys():
        master_csv = _resolve_builtin_master_csv_for_operator_assets(dataset_key)
        return dataset_key, Path(_dataset_bundle_dir(dataset_key)), master_csv
    record = get_user_upload_by_dataset_key(dataset_key, user)
    if record is None:
        return None
    return (
        user_upload_slug(record),
        Path(user_upload_abs_dir(record)),
        resolve_user_upload_master_csv(record),
    )


def _ensure_catalog_operator_statistic_csv(dataset_key: str, user, operator: str) -> Path | None:
    context = _resolve_catalog_operator_asset_context(dataset_key, user)
    if context is None:
        return None
    file_key, bundle_dir, master_csv = context
    if not master_csv:
        return None
    try:
        return ensure_operator_catalog_assets(file_key, master_csv, bundle_dir, operator)
    except Exception:
        logger.exception(
            'Failed to generate missing operator statistics: dataset=%s operator=%s',
            dataset_key,
            operator,
        )
        return None


def _float_precision_fields_for_catalog_item(
    *,
    lite: bool,
    column_info_path: str,
    bundle_dir: str,
) -> dict[str, object]:
    if lite or not os.path.isfile(column_info_path):
        return empty_float_precision_catalog_fields()
    return float_precision_catalog_fields(column_info_path, bundle_dir)


def _resolve_catalog_statistic_csv(
    dataset_key: str,
    user,
    data_type: str = '',
    operator: str = '',
):
    if dataset_key in _discovered_catalog_bundle_keys():
        base = _dataset_bundle_dir(dataset_key)
        if operator:
            path_csv = operator_statistic_csv_path(Path(base), dataset_key, operator)
            if not path_csv.is_file():
                generated = _ensure_catalog_operator_statistic_csv(
                    dataset_key,
                    user,
                    operator,
                )
                if generated is not None:
                    path_csv = generated
            return str(path_csv) if path_csv.is_file() else None
        if data_type in ('int', 'float'):
            path_csv = os.path.join(
                base,
                dtype_asset_filename(dataset_key, data_type, 'statistic'),
            )
            if os.path.isfile(path_csv):
                return path_csv
        path_csv = os.path.join(
            base,
            catalog_asset_filename(dataset_key, 'numeric_statistic'),
        )
        if not os.path.isfile(path_csv):
            legacy = os.path.join(base, 'numeric_statistic.csv')
            path_csv = legacy if os.path.isfile(legacy) else path_csv
        if not os.path.isfile(path_csv):
            for dt in ('float', 'int'):
                candidate = os.path.join(
                    base,
                    dtype_asset_filename(dataset_key, dt, 'statistic'),
                )
                if os.path.isfile(candidate):
                    return candidate
        return path_csv if os.path.isfile(path_csv) else None
    record = get_user_upload_by_dataset_key(dataset_key, user)
    if record is None:
        return None
    if operator:
        bundle_dir = Path(user_upload_abs_dir(record))
        file_key = user_upload_slug(record)
        path_csv = operator_statistic_csv_path(bundle_dir, file_key, operator)
        if not path_csv.is_file():
            generated = _ensure_catalog_operator_statistic_csv(
                dataset_key,
                user,
                operator,
            )
            if generated is not None:
                path_csv = generated
        return str(path_csv) if path_csv.is_file() else None
    return resolve_user_upload_statistic_csv(record, data_type or None)


def _resolve_catalog_asset_file(dataset_key: str, filename: str, user, data_type: str = ''):
    if dataset_key in _discovered_catalog_bundle_keys():
        allowed = catalog_servable_asset_filenames(dataset_key) | {f'{dataset_key}.csv'}
        if filename not in allowed:
            return None
        if filename == f'{dataset_key}.csv':
            return resolve_master_csv_path(dataset_key)
        base = os.path.realpath(_dataset_bundle_dir(dataset_key))
        target = os.path.realpath(os.path.join(base, filename))
        if target.startswith(base + os.sep) and os.path.isfile(target):
            return target
        return None
    record = get_user_upload_by_dataset_key(dataset_key, user)
    if record is None:
        return None
    return resolve_user_upload_catalog_asset_file(
        record,
        filename,
        data_type or None,
    )


_COLUMN_DISTRIBUTION_CHART_NAME_RE = re.compile(
    r'^column_\d+_(scaling|leading_zeros|trailing_zeros|xor_leading|xor_trailing)\.png$'
)


def _resolve_catalog_column_distribution_chart_file(
    dataset_key: str,
    chart_name: str,
    user,
    operator: str = '',
):
    if not chart_name or not _COLUMN_DISTRIBUTION_CHART_NAME_RE.match(chart_name):
        return None
    if dataset_key in _discovered_catalog_bundle_keys():
        base = os.path.realpath(_dataset_bundle_dir(dataset_key))
        if operator:
            chart_dir = os.path.realpath(
                str(operator_bundle_dir(Path(base), operator) / column_distribution_chart_dirname())
            )
        else:
            chart_dir = os.path.realpath(
                os.path.join(base, column_distribution_chart_dirname())
            )
        target = os.path.realpath(os.path.join(chart_dir, chart_name))
        if target.startswith(chart_dir + os.sep) and os.path.isfile(target):
            return target
        return None
    record = get_user_upload_by_dataset_key(dataset_key, user)
    if record is None:
        return None
    base = os.path.realpath(user_upload_abs_dir(record))
    chart_dir = os.path.realpath(
        os.path.join(base, column_distribution_chart_dirname())
    )
    target = os.path.realpath(os.path.join(chart_dir, chart_name))
    if target.startswith(chart_dir + os.sep) and os.path.isfile(target):
        return target
    return None


class DatasetCatalogStatisticJsonView(APIView):
    """JSON table for numeric column statistics (catalog bundles and user uploads)."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, dataset_key, *args, **kwargs):
        from . import dataset_catalog_assets as catalog_assets

        operator = _parse_catalog_operator(request)
        path_csv = _resolve_catalog_statistic_csv(
            dataset_key,
            request.user,
            _catalog_asset_data_type(request),
            operator,
        )
        if not path_csv:
            raise Http404('Statistic file not found')
        if not os.path.isfile(path_csv):
            return JsonResponse({'columns': [], 'rows': []})
        try:
            df = pd.read_csv(path_csv)
        except Exception:
            logger.exception('Failed to read numeric_statistic.csv for %s', dataset_key)
            raise Http404('Statistic file is not readable')
        float_precision = parse_float_precision_filter(
            request.query_params.get('float_precision'),
        )
        if float_precision != FLOAT_PRECISION_FILTER_ALL:
            col_info_path, bundle_dir = _resolve_catalog_column_info_and_bundle(
                dataset_key,
                request.user,
            )
            if col_info_path and bundle_dir and os.path.isfile(col_info_path):
                df = filter_report_dataframe_for_float_precision(
                    df,
                    column_info_path=col_info_path,
                    bundle_dir=bundle_dir,
                    float_precision=float_precision,
                )
        baseline_rows = None
        if operator:
            baseline_path = _resolve_catalog_statistic_csv(
                dataset_key,
                request.user,
                _catalog_asset_data_type(request),
                '',
            )
            if baseline_path and os.path.isfile(baseline_path):
                try:
                    baseline_df = pd.read_csv(baseline_path)
                    if float_precision != FLOAT_PRECISION_FILTER_ALL:
                        col_info_path, bundle_dir = _resolve_catalog_column_info_and_bundle(
                            dataset_key,
                            request.user,
                        )
                        if col_info_path and bundle_dir and os.path.isfile(col_info_path):
                            baseline_df = filter_report_dataframe_for_float_precision(
                                baseline_df,
                                column_info_path=col_info_path,
                                bundle_dir=bundle_dir,
                                float_precision=float_precision,
                            )
                    baseline_rows = baseline_df.to_dict(orient='records')
                except Exception:
                    logger.exception(
                        'Failed to read baseline numeric_statistic.csv for %s',
                        dataset_key,
                    )
        rows = df.to_dict(orient='records')
        bundle_dir: Path | None = None
        if dataset_key in _discovered_catalog_bundle_keys():
            bundle_dir = Path(_dataset_bundle_dir(dataset_key))
        else:
            upload_record = get_user_upload_by_dataset_key(dataset_key, request.user)
            if upload_record is not None:
                bundle_dir = Path(user_upload_abs_dir(upload_record))
        for row in rows:
            raw_idx = row.get('column_index', None)
            try:
                col_idx = int(raw_idx)
            except (TypeError, ValueError):
                for field in catalog_assets.COLUMN_DISTRIBUTION_DATA_FIELDS.values():
                    row[field] = []
                continue
            if bundle_dir is None:
                for field in catalog_assets.COLUMN_DISTRIBUTION_DATA_FIELDS.values():
                    row[field] = []
                continue
            for kind, field in catalog_assets.COLUMN_DISTRIBUTION_DATA_FIELDS.items():
                data_path = catalog_assets.column_distribution_data_path(
                    bundle_dir,
                    col_idx,
                    kind,
                    operator_id=operator,
                )
                row[field] = catalog_assets.read_column_distribution_data(data_path)
        from .statistic import enrich_statistic_rows_for_api

        rows = enrich_statistic_rows_for_api(
            rows,
            baseline_rows=baseline_rows if operator else None,
        )
        response_columns = list(df.columns)
        for extra_col in (
            'byte_entropy',
            'value_entropy',
            'byte_entropy_change',
            'value_entropy_change',
        ):
            if extra_col not in response_columns:
                response_columns.append(extra_col)
        return JsonResponse(
            _sanitize_for_json(
                {
                    'columns': response_columns,
                    'rows': rows,
                }
            ),
            status=200,
        )


def _operator_step_entropy_csv_path(dataset_key: str) -> Path:
    return Path(_dataset_bundle_dir(dataset_key)) / f'{dataset_key}_operator_step_byte_entropy.csv'


def _operator_step_markov3_entropy_rate_csv_path(dataset_key: str) -> Path:
    return Path(_dataset_bundle_dir(dataset_key)) / f'{dataset_key}_operator_step_byte_markov3_entropy_rate.csv'


OPERATOR_ENTROPY_SNAPSHOT_PATH = (
    Path(settings.BASE_DIR) / 'dataset' / 'operator_entropy_all_datasets.csv'
)


def _overall_catalog_bundle_keys() -> list[str]:
    focused = _focused_original_dataset_keys()
    keys = [
        key for key in _all_dataset_keys_for_ui()
        if (not focused or key in focused)
        and _built_in_dataset_catalog_ready(key)
        and catalog_has_statistics_bundle(key)
    ]
    if focused:
        return keys
    entropy_ready = [
        key for key in keys
        if _operator_step_entropy_csv_path(key).is_file()
    ]
    if entropy_ready:
        return entropy_ready
    return keys


def _read_operator_step_rows_for_path(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    with path.open('r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            data_type = str(row.get('data_type', '') or '').strip().lower()
            if data_type not in {'int', 'float'}:
                continue
            algorithm = str(row.get('algorithm', '') or '').strip()
            operator_id = str(row.get('operator_id', '') or '').strip()
            if not algorithm or not operator_id:
                continue
            key = (
                data_type,
                algorithm,
                str(row.get('column_index', '') or '').strip(),
                str(row.get('step_index', '') or '').strip(),
                str(row.get('step_variant', '') or '').strip(),
                operator_id,
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _operator_step_float(raw: object) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _operator_step_entropy_float(
    row: dict,
    value_field: str,
    *,
    count_field: str | None = None,
) -> float | None:
    if count_field is not None:
        try:
            count = int(float(row.get(count_field, '') or 0))
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            return None
    value = _operator_step_float(row.get(value_field))
    if value is None or abs(value) < 1e-12:
        return None
    return value


def _median_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(median(finite)) if finite else None


def _operator_display_name_map() -> dict[str, str]:
    out: dict[str, str] = {
        'bitpack_int': 'Bit packing',
        'output_bytes': 'Output bytes',
    }
    try:
        rows = list(load_method_page_operators())
    except Exception:
        rows = []
    for row in rows:
        operator_id = str(row.get('id', '') or '').strip()
        name = str(row.get('name', '') or '').strip()
        if operator_id and name:
            out[operator_id] = name
    return out


def _operator_step_label(operator_id: str, display_names: dict[str, str]) -> str:
    return display_names.get(operator_id) or str(operator_id or '').replace('_', ' ')


def _method_page_visible_operator_ids() -> tuple[str, ...]:
    try:
        from .dataset_correlation_heatmaps import _method_operator_ids

        return tuple(_method_operator_ids())
    except Exception:
        supported = set(SUPPORTED_OPERATORS)
        out: list[str] = []
        seen: set[str] = set()
        for row in load_method_page_operators():
            operator_id = str(row.get('id', '') or '').strip()
            if operator_id == 'rd':
                continue
            if operator_id and operator_id in supported and operator_id not in seen:
                out.append(operator_id)
                seen.add(operator_id)
        return tuple(out)


def _status_for_step(rows: list[dict]) -> str:
    statuses = [str(row.get('operator_status', '') or '').strip() for row in rows]
    if statuses and all(status == 'ok' for status in statuses):
        return 'ok'
    for candidate in (
        'unsupported_transform',
        'blocked_by_unsupported_previous_step',
    ):
        if candidate in statuses:
            return candidate
    return statuses[0] if statuses else ''


def _transition_state_after_operator(
    operator_id: str,
    state: str,
    *,
    terminal: bool,
    byte_operator_ids: set[str],
) -> str:
    op = str(operator_id or '').strip()
    cur = str(state or '').strip().lower() or 'int'
    if cur in {'mixed', 'unknown'}:
        return cur
    if op == 'scaling':
        return 'int' if cur == 'float' else cur
    if op in {'eraser', 'ease', 'pongo_eraser'}:
        return 'byte' if cur == 'float' else cur
    if op in {'frame_of_reference', 'delta', 'rd', 'outlier_separation', 'subcolumn', 'xor'}:
        return cur
    if op in {'bitpacking', 'bitpack_int', 'trailing_zero_pack'} or op in byte_operator_ids:
        return 'byte'
    if op == 'dictionary':
        return 'byte' if terminal else 'int'
    if op == 'zigzag':
        return 'int'
    if op in {'rle', 'run_length'}:
        return 'mixed'
    return cur


def _read_operator_step_entropy_rows(dataset_key: str) -> list[dict]:
    return _read_operator_step_rows_for_path(_operator_step_entropy_csv_path(dataset_key))


def _read_operator_step_markov3_entropy_rate_rows(dataset_key: str) -> list[dict]:
    return _read_operator_step_rows_for_path(_operator_step_markov3_entropy_rate_csv_path(dataset_key))


def _read_overall_operator_step_entropy_rows() -> list[dict]:
    rows: list[dict] = []
    for dataset_key in _overall_catalog_bundle_keys():
        for row in _read_operator_step_entropy_rows(dataset_key):
            next_row = dict(row)
            next_row['dataset_key'] = dataset_key
            rows.append(next_row)
    return rows


def _read_overall_operator_step_markov3_entropy_rate_rows() -> list[dict]:
    rows: list[dict] = []
    for dataset_key in _overall_catalog_bundle_keys():
        for row in _read_operator_step_markov3_entropy_rate_rows(dataset_key):
            next_row = dict(row)
            next_row['dataset_key'] = dataset_key
            rows.append(next_row)
    return rows


def _mean_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _dataset_equal_mean_or_none(
    grouped_values: dict[str, list[float]],
) -> float | None:
    per_dataset_means = []
    for values in grouped_values.values():
        mean = _mean_or_none(values)
        if mean is not None:
            per_dataset_means.append(mean)
    return _mean_or_none(per_dataset_means)


_OVERALL_BYTE_ENTROPY_OPERATOR_ALIASES = {
    'rle': 'run_length',
}

_OVERALL_BYTE_ENTROPY_OPERATOR_LABELS = {
    'run_length': 'run length',
}


def _overall_byte_entropy_operator_id(operator_id: str) -> str:
    return _OVERALL_BYTE_ENTROPY_OPERATOR_ALIASES.get(operator_id, operator_id)


def _overall_byte_entropy_operator_label(
    operator_id: str,
    display_names: dict[str, str],
) -> str:
    return (
        _OVERALL_BYTE_ENTROPY_OPERATOR_LABELS.get(operator_id)
        or _operator_step_label(operator_id, display_names)
    )


def _build_overall_operator_step_change_chart(
    rows: list[dict],
    *,
    title: str,
    description: str,
    y_label: str,
    display_names: dict[str, str] | None = None,
    operator_ids: list[str] | tuple[str, ...] | None = None,
) -> dict | None:
    display = display_names or _operator_display_name_map()
    catalog_operator_ids = (
        tuple(operator_ids)
        if operator_ids is not None
        else _method_page_visible_operator_ids()
    )
    source_operator_ids = set(catalog_operator_ids)
    merged_catalog_operator_ids: list[str] = []
    merged_seen: set[str] = set()
    for operator_id in catalog_operator_ids:
        merged_operator_id = _overall_byte_entropy_operator_id(operator_id)
        if merged_operator_id in merged_seen:
            continue
        merged_seen.add(merged_operator_id)
        merged_catalog_operator_ids.append(merged_operator_id)
    by_operator: dict[str, dict[str, list[float]]] = {}
    for row_index, row in enumerate(rows):
        operator_id = str(row.get('operator_id', '') or '').strip()
        if not operator_id or operator_id not in source_operator_ids:
            continue
        change = _operator_step_entropy_float(row, 'byte_entropy_change')
        if change is None:
            continue
        dataset_key = str(row.get('dataset_key', '') or '').strip()
        if not dataset_key:
            dataset_key = f'__row_{row_index}'
        merged_operator_id = _overall_byte_entropy_operator_id(operator_id)
        # The paper uses dataset-equal aggregation. Keep each dataset as one
        # observation, then average those dataset means across datasets.
        by_operator.setdefault(merged_operator_id, {}).setdefault(dataset_key, []).append(-change)
    rows_out = []
    for operator_id in merged_catalog_operator_ids:
        dataset_values = by_operator.get(operator_id, {})
        avg_change = _dataset_equal_mean_or_none(dataset_values)
        rows_out.append((
            operator_id,
            _overall_byte_entropy_operator_label(operator_id, display),
            avg_change,
            len(dataset_values),
        ))
    if not any(
        sample_count
        for _operator_id, _label, _reduction, sample_count in rows_out
    ):
        return None
    rows_out.sort(
        key=lambda row: (
            0 if row[2] is not None else 1,
            -(row[2] if row[2] is not None else 0.0),
            row[1].casefold(),
        )
    )
    return {
        'chart_type': 'bar',
        'title': title,
        'description': description,
        'operator_ids': [
            operator_id
            for operator_id, _label, _reduction, _sample_count in rows_out
        ],
        'algorithms': [
            label
            for _operator_id, label, _reduction, _sample_count in rows_out
        ],
        'mean_changes': [
            change
            for _operator_id, _label, change, _sample_count in rows_out
        ],
        'sample_counts': [
            sample_count
            for _operator_id, _label, _reduction, sample_count in rows_out
        ],
        'sample_count_label': 'Datasets',
        'y_label': y_label,
        'log_y': False,
    }


def _build_overall_operator_step_byte_entropy_chart(
    rows: list[dict],
    *,
    display_names: dict[str, str] | None = None,
    operator_ids: list[str] | tuple[str, ...] | None = None,
) -> dict | None:
    return _build_overall_operator_step_change_chart(
        rows,
        title='Overall byte entropy reduction by operator',
        description=(
            'Mean byte entropy reduction per operator after dataset-equal '
            'aggregation across all datasets, algorithms, and numeric data '
            'types in the operator pipeline automata.'
        ),
        y_label='Mean byte entropy reduction (bits/byte)',
        display_names=display_names,
        operator_ids=operator_ids,
    )


def _build_overall_operator_step_markov3_entropy_rate_chart(
    rows: list[dict],
    *,
    display_names: dict[str, str] | None = None,
    operator_ids: list[str] | tuple[str, ...] | None = None,
    title: str | None = None,
    description: str | None = None,
) -> dict | None:
    return _build_overall_operator_step_change_chart(
        rows,
        title=title or 'Overall Markov-3 byte entropy-rate reduction by operator',
        description=description or (
            'Mean Markov-3 byte entropy-rate reduction per operator after '
            'dataset-equal aggregation across all datasets, algorithms, and '
            'numeric data types in the operator pipeline automata.'
        ),
        y_label='Mean Markov-3 byte entropy-rate reduction (bits/byte)',
        display_names=display_names,
        operator_ids=operator_ids,
    )


def _read_operator_entropy_snapshot_rows() -> list[dict]:
    path = OPERATOR_ENTROPY_SNAPSHOT_PATH
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _build_operator_entropy_reduction_chart(
    *,
    dataset_key: str | None = None,
) -> dict | None:
    display = _operator_display_name_map()
    snapshot_rows = _read_operator_entropy_snapshot_rows()
    if not snapshot_rows:
        return None
    by_operator: dict[str, list[float]] = defaultdict(list)
    for row in snapshot_rows:
        if dataset_key and str(row.get('dataset_key', '') or '').strip() != dataset_key:
            continue
        operator_id = str(row.get('operator_id', '') or '').strip()
        if not operator_id:
            continue
        try:
            value = float(row.get('mean_entropy_reduction', '') or '')
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        by_operator[operator_id].append(value)
    if not by_operator:
        return None

    rows_out = []
    for operator_id, values in by_operator.items():
        mean_reduction = sum(values) / len(values)
        if abs(mean_reduction) < 0.005:
            mean_reduction = 0.0
        rows_out.append((
            operator_id,
            _overall_byte_entropy_operator_label(operator_id, display),
            mean_reduction,
            len(values),
        ))
    rows_out.sort(
        key=lambda row: (
            0 if row[2] is not None else 1,
            -(row[2] if row[2] is not None else 0.0),
            row[1].casefold(),
        )
    )
    selected_label = dataset_key or 'Overall'
    return {
        'chart_type': 'bar',
        'title': f'{selected_label} byte entropy reduction by operator',
        'description': (
            'Mean byte entropy reduction per operator from the standalone '
            'operator sample used by the paper.'
        ),
        'operator_ids': [
            operator_id
            for operator_id, _label, _reduction, _sample_count in rows_out
        ],
        'algorithms': [
            label
            for _operator_id, label, _reduction, _sample_count in rows_out
        ],
        'mean_changes': [
            reduction
            for _operator_id, _label, reduction, _sample_count in rows_out
        ],
        'sample_counts': [
            sample_count
            for _operator_id, _label, _reduction, sample_count in rows_out
        ],
        'sample_count_label': 'Datasets' if dataset_key is None else 'Columns',
        'y_label': 'Mean byte entropy reduction (bits/byte)',
        'log_y': False,
    }


def _operator_step_level_sort_key(algorithm: str) -> float | None:
    try:
        from .home_leaderboard_boxplot_data import pareto_level_sort_key
    except Exception:
        return None
    return pareto_level_sort_key(algorithm)


def _operator_step_result_path(bundle_dir: Path | str | None, result_file: object) -> Path | None:
    filename = str(result_file or '').strip()
    if not filename:
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    if bundle_dir is None:
        return None
    return Path(bundle_dir) / filename


def _operator_step_algorithm_compression_rates(
    rows: list[dict],
    *,
    bundle_dir: Path | str | None = None,
) -> dict[str, float]:
    wanted = {
        str(row.get('algorithm', '') or '').strip()
        for row in rows
        if str(row.get('algorithm', '') or '').strip()
    }
    if not wanted or bundle_dir is None:
        return {}

    totals: dict[str, list[float]] = {}
    seen_paths: set[Path] = set()
    for row in rows:
        path = _operator_step_result_path(bundle_dir, row.get('result_file'))
        if path is None or path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        try:
            with path.open('r', encoding='utf-8', newline='') as handle:
                for rec in csv.DictReader(handle):
                    algorithm = str(rec.get('algorithm', '') or '').strip()
                    if algorithm not in wanted:
                        continue
                    compressed = _operator_step_float(rec.get('compressedSize'))
                    original = _operator_step_float(rec.get('originalSize'))
                    if compressed is None or original is None or original <= 0 or compressed <= 0:
                        continue
                    current = totals.setdefault(algorithm, [0.0, 0.0])
                    current[0] += compressed
                    current[1] += original
        except OSError:
            continue

    rates = {}
    for algorithm, (compressed, original) in totals.items():
        if original > 0:
            rates[algorithm] = compressed / original
    return rates


def _best_operator_step_level_algorithms(
    grouped: dict[str, dict],
    rates: dict[str, float],
) -> set[str]:
    by_family: dict[str, list[tuple[str, float | None, float]]] = {}
    for algorithm, item in grouped.items():
        level_order = _operator_step_level_sort_key(algorithm)
        if level_order is None:
            continue
        method_name = str(item.get('method_name', '') or '').strip()
        family = method_name.casefold() if method_name else algorithm.casefold()
        by_family.setdefault(family, []).append((algorithm, rates.get(algorithm), float(level_order)))

    selected = set()
    for candidates in by_family.values():
        if len(candidates) == 1:
            selected.add(candidates[0][0])
            continue
        best = min(
            candidates,
            key=lambda row: (
                0 if row[1] is not None and math.isfinite(row[1]) else 1,
                row[1] if row[1] is not None and math.isfinite(row[1]) else math.inf,
                -row[2],
                row[0].casefold(),
            ),
        )
        selected.add(best[0])
    return selected


def _operator_step_algorithm_options(
    rows: list[dict],
    *,
    bundle_dir: Path | str | None = None,
) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        algorithm = str(row.get('algorithm', '') or '').strip()
        if not algorithm:
            continue
        item = grouped.setdefault(
            algorithm,
            {
                'algorithm': algorithm,
                'label': algorithm,
                'method_name': str(row.get('method_name', '') or '').strip(),
                'data_types': set(),
            },
        )
        dtype = str(row.get('data_type', '') or '').strip().lower()
        if dtype in {'int', 'float'}:
            item['data_types'].add(dtype)
        if not item.get('method_name'):
            item['method_name'] = str(row.get('method_name', '') or '').strip()
    rates = _operator_step_algorithm_compression_rates(rows, bundle_dir=bundle_dir)
    selected_level_algorithms = _best_operator_step_level_algorithms(grouped, rates)
    out = []
    for algorithm, item in grouped.items():
        if _operator_step_level_sort_key(algorithm) is not None and algorithm not in selected_level_algorithms:
            continue
        method_name = str(item.get('method_name', '') or '').strip()
        label = method_name or algorithm
        out.append({
            'algorithm': algorithm,
            'label': label,
            'method_name': method_name,
            'data_types': sorted(item['data_types']),
        })
    out.sort(key=lambda row: (str(row.get('method_name') or row['algorithm']).casefold(), row['algorithm'].casefold()))
    return out


def _finite_chart_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _merge_heatmap_blocks(payloads: list[dict], key: str) -> dict | None:
    blocks = [
        payload.get(key)
        for payload in payloads
        if isinstance(payload.get(key), dict)
        and isinstance(payload.get(key, {}).get('x_labels'), list)
        and isinstance(payload.get(key, {}).get('y_labels'), list)
        and isinstance(payload.get(key, {}).get('values'), list)
    ]
    if not blocks:
        return None
    x_labels = list(blocks[0].get('x_labels') or [])
    if not x_labels:
        return None
    by_label: dict[str, list[list[float]]] = {}
    label_order: list[str] = []
    for block in blocks:
        block_x = list(block.get('x_labels') or [])
        if block_x != x_labels:
            continue
        for label, values in zip(block.get('y_labels') or [], block.get('values') or []):
            name = str(label or '').strip()
            if not name or not isinstance(values, list):
                continue
            if name not in by_label:
                by_label[name] = [[] for _ in x_labels]
                label_order.append(name)
            for idx, value in enumerate(values[:len(x_labels)]):
                number = _finite_chart_number(value)
                if number is not None:
                    by_label[name][idx].append(number)
    if not label_order:
        return None
    first = blocks[0]
    return {
        'title': first.get('title') or '',
        'description': first.get('description') or '',
        'x_labels': x_labels,
        'y_labels': label_order,
        'values': [
            [
                float(median(values)) if values else None
                for values in by_label[label]
            ]
            for label in label_order
        ],
    }


def _merge_entropy_chart_blocks(charts: list[dict], *, normalize_before: bool = False) -> dict | None:
    if not charts:
        return None
    by_label: dict[str, dict[str, list[float]]] = {}
    label_order: list[str] = []
    for chart in charts:
        algorithms = chart.get('algorithms') or []
        values = chart.get('values') or []
        before_medians = chart.get('before_medians') or []
        after_medians = chart.get('after_medians') or []
        for idx, label in enumerate(algorithms):
            name = str(label or '').strip()
            if not name:
                continue
            bucket = by_label.setdefault(
                name,
                {'values': [], 'before_medians': [], 'after_medians': []},
            )
            if name not in label_order:
                label_order.append(name)
            for value in values[idx] if idx < len(values) and isinstance(values[idx], list) else []:
                number = _finite_chart_number(value)
                if number is not None:
                    bucket['values'].append(number)
            for field, source in (
                ('before_medians', before_medians),
                ('after_medians', after_medians),
            ):
                if idx < len(source):
                    number = _finite_chart_number(source[idx])
                    if number is not None:
                        bucket[field].append(number)
    if not label_order:
        return None
    first = charts[0]
    rows = []
    for label in label_order:
        bucket = by_label[label]
        values = bucket['values']
        if not values:
            median_change = float('inf')
        else:
            median_change = float(median(values))
        before_median = (
            float(median(bucket['before_medians']))
            if bucket['before_medians']
            else float('nan')
        )
        after_median = (
            float(median(bucket['after_medians']))
            if bucket['after_medians']
            else float('nan')
        )
        rows.append((label, values, before_median, after_median, median_change))
    rows.sort(key=lambda row: (row[4], row[0]))
    common_before = None
    if normalize_before:
        finite_before = [
            before_median
            for _label, _values, before_median, _after_median, _median_change in rows
            if _finite_chart_number(before_median) is not None
        ]
        if finite_before:
            common_before = float(median(finite_before))
            rows.sort(
                key=lambda row: (
                    (
                        row[3] - common_before
                        if _finite_chart_number(row[3]) is not None
                        else float('inf')
                    ),
                    row[0],
                )
            )
    return {
        'title': first.get('title') or '',
        'description': first.get('description') or '',
        'algorithms': [label for label, _values, _before, _after, _change in rows],
        'values': [values for _label, values, _before, _after, _change in rows],
        'before_medians': [
            common_before if common_before is not None else before_median
            for _label, _values, before_median, _after, _change in rows
        ],
        'after_medians': [
            after_median for _label, _values, _before, after_median, _change in rows
        ],
        'y_label': first.get('y_label') or '',
        'log_y': bool(first.get('log_y', False)),
    }


def _merge_entropy_boxplot_payloads(payloads: list[dict]) -> dict[str, dict]:
    by_key: dict[str, list[dict]] = {}
    for payload in payloads:
        block = payload.get('operator_entropy_change_boxplots')
        if not isinstance(block, dict):
            continue
        for key, chart in block.items():
            if isinstance(chart, dict):
                by_key.setdefault(str(key), []).append(chart)
    out: dict[str, dict] = {}
    for key, charts in by_key.items():
        merged = _merge_entropy_chart_blocks(charts)
        if not merged:
            continue
        segment_groups: dict[str, list[dict]] = {}
        for chart in charts:
            for segment in chart.get('segments') or []:
                if isinstance(segment, dict):
                    segment_key = str(segment.get('segment_key') or '')
                    if segment_key:
                        segment_groups.setdefault(segment_key, []).append(segment)
        segments: list[dict] = []
        for segment_key, segment_charts in segment_groups.items():
            segment = _merge_entropy_chart_blocks(segment_charts, normalize_before=True)
            if not segment:
                continue
            first = segment_charts[0]
            segment['segment_key'] = segment_key
            segment['segment_label'] = first.get('segment_label') or segment_key
            segment['column_count'] = sum(
                int(float(chart.get('column_count') or 0))
                for chart in segment_charts
                if _finite_chart_number(chart.get('column_count')) is not None
            )
            segments.append(segment)
        if segments:
            merged['segments'] = segments
        out[key] = merged
    return out


def _catalog_correlation_payload_for_dataset(
    *,
    dataset_key: str,
    user,
    float_precision: str,
    materialize_missing_operator_assets: bool = True,
) -> dict | None:
    from .dataset_correlation_heatmaps import load_correlation_heatmap_context
    from .dataset_float_precision_distribution import (
        FLOAT_PRECISION_FILTER_ALL,
        allowed_float_column_indexes_for_filter,
    )
    from .upload_storage import get_user_upload_by_dataset_key, user_upload_slug

    col_info_path, bundle_dir = _resolve_catalog_column_info_and_bundle(
        dataset_key,
        user,
    )
    if not col_info_path or not bundle_dir:
        return None

    file_key = dataset_key
    upload_record = get_user_upload_by_dataset_key(dataset_key, user)
    if upload_record is not None:
        file_key = user_upload_slug(upload_record)
    operator_context = _resolve_catalog_operator_asset_context(dataset_key, user)
    master_csv = operator_context[2] if operator_context is not None else None

    filtered_indices: list[int] | None = None
    if float_precision != FLOAT_PRECISION_FILTER_ALL:
        allowed = allowed_float_column_indexes_for_filter(
            col_info_path,
            bundle_dir,
            float_precision,
        )
        filtered_indices = sorted(int(idx) for idx in (allowed or set()))

    payload = load_correlation_heatmap_context(
        bundle_dir=Path(bundle_dir),
        dataset_key=dataset_key,
        column_info_path=col_info_path,
        filtered_column_indices=filtered_indices,
        file_key=file_key,
        master_csv=master_csv,
        materialize_missing_operator_assets=materialize_missing_operator_assets,
    )
    if not payload:
        return None

    snapshot_byte_chart = _build_operator_entropy_reduction_chart(
        dataset_key=dataset_key,
    )
    if snapshot_byte_chart is not None:
        payload.setdefault('operator_entropy_change_boxplots', {})[
            'byte_entropy_change'
        ] = snapshot_byte_chart

    markov3_chart = _build_overall_operator_step_markov3_entropy_rate_chart(
        _read_operator_step_markov3_entropy_rate_rows(dataset_key),
        title=f'{dataset_key} Markov-3 byte entropy-rate reduction by operator',
        description=(
            'Mean Markov-3 byte entropy-rate reduction per operator for the '
            'selected dataset.'
        ),
    )
    if markov3_chart is not None:
        payload['operator_step_markov3_entropy_rate_chart'] = markov3_chart
    return payload


def _overall_correlation_heatmap_payload(*, user, float_precision: str) -> dict:
    payloads = [
        payload
        for dataset_key in _overall_catalog_bundle_keys()
        for payload in [
            _catalog_correlation_payload_for_dataset(
                dataset_key=dataset_key,
                user=user,
                float_precision=float_precision,
                materialize_missing_operator_assets=False,
            )
        ]
        if payload
    ]
    entropy_boxplots = _merge_entropy_boxplot_payloads(payloads)
    overall_operator_step_byte_chart = _build_operator_entropy_reduction_chart()
    if overall_operator_step_byte_chart is None:
        overall_operator_step_byte_chart = _build_overall_operator_step_byte_entropy_chart(
            _read_overall_operator_step_entropy_rows(),
        )
    if overall_operator_step_byte_chart is not None:
        entropy_boxplots['byte_entropy_change'] = overall_operator_step_byte_chart
    overall_operator_step_markov3_chart = _build_overall_operator_step_markov3_entropy_rate_chart(
        _read_overall_operator_step_markov3_entropy_rate_rows(),
    )
    return {
        'dataset_key': 'overall',
        'column_count': sum(int(payload.get('column_count') or 0) for payload in payloads),
        'algorithm_heatmap': _merge_heatmap_blocks(payloads, 'algorithm_heatmap'),
        'value_based_algorithm_heatmap': _merge_heatmap_blocks(payloads, 'value_based_algorithm_heatmap'),
        'byte_based_algorithm_heatmap': _merge_heatmap_blocks(payloads, 'byte_based_algorithm_heatmap'),
        'operator_heatmap': _merge_heatmap_blocks(payloads, 'operator_heatmap'),
        'operator_entropy_change_boxplots': entropy_boxplots,
        'operator_step_markov3_entropy_rate_chart': overall_operator_step_markov3_chart,
    }


def _build_operator_step_entropy_automaton(
    *,
    rows: list[dict],
    data_type: str,
    algorithm: str,
    display_names: dict[str, str],
    byte_operator_ids: set[str],
) -> dict | None:
    selected = [
        row for row in rows
        if str(row.get('data_type', '') or '').strip().lower() == data_type
        and str(row.get('algorithm', '') or '').strip() == algorithm
    ]
    if not selected:
        return None

    by_step: dict[int, list[dict]] = {}
    for row in selected:
        try:
            step_index = int(row.get('step_index', '') or 0)
        except (TypeError, ValueError):
            continue
        if step_index <= 0:
            continue
        by_step.setdefault(step_index, []).append(row)
    if not by_step:
        return None

    method_name = str(selected[0].get('method_name', '') or '').strip()
    first_rows = by_step[min(by_step)]
    start_entropy = _median_or_none([
        value for value in (
            _operator_step_entropy_float(
                row,
                'before_byte_entropy',
                count_field='before_count',
            ) for row in first_rows
        ) if value is not None
    ])
    states = [{
        'id': 's0',
        'label': data_type,
        'data_type': data_type,
        'byte_entropy': start_entropy,
        'unknown_entropy': start_entropy is None,
    }]
    edges = []
    current_state_type = data_type

    for ordinal, step_index in enumerate(sorted(by_step), start=1):
        step_rows = by_step[step_index]
        is_terminal = step_index == max(by_step)
        next_state_id = f's{ordinal}'
        operator_groups: dict[tuple[str, str], list[dict]] = {}
        for row in step_rows:
            operator_id = str(row.get('operator_id', '') or '').strip()
            variant = str(row.get('step_variant', '') or '').strip()
            if not operator_id:
                continue
            operator_groups.setdefault((variant, operator_id), []).append(row)

        next_types: set[str] = set()
        all_after_values: list[float] = []
        for (variant, operator_id), op_rows in sorted(operator_groups.items()):
            transform_operator = str(op_rows[0].get('transform_operator_id', '') or operator_id).strip()
            before_values = [
                value for value in (
                    _operator_step_entropy_float(
                        row,
                        'before_byte_entropy',
                        count_field='before_count',
                    ) for row in op_rows
                ) if value is not None
            ]
            after_values = [
                value for value in (
                    _operator_step_entropy_float(
                        row,
                        'after_byte_entropy',
                        count_field='after_count',
                    ) for row in op_rows
                ) if value is not None
            ]
            change_values = [
                value for value in (
                    _operator_step_entropy_float(row, 'byte_entropy_change') for row in op_rows
                ) if value is not None
            ]
            all_after_values.extend(after_values)
            next_types.add(_transition_state_after_operator(
                transform_operator,
                current_state_type,
                terminal=is_terminal,
                byte_operator_ids=byte_operator_ids,
            ))
            edges.append({
                'id': f'e{step_index}-{variant or "main"}-{operator_id}',
                'from': f's{ordinal - 1}',
                'to': next_state_id,
                'step_index': step_index,
                'step_variant': variant,
                'operator_id': operator_id,
                'operator_label': _operator_step_label(operator_id, display_names),
                'transform_operator_id': transform_operator,
                'status': _status_for_step(op_rows),
                'before_byte_entropy': _median_or_none(before_values),
                'after_byte_entropy': _median_or_none(after_values),
                'byte_entropy_change': _median_or_none(change_values),
                'sample_count': len(op_rows),
            })

        if len(next_types) == 1:
            current_state_type = next(iter(next_types))
        elif len(next_types) > 1:
            current_state_type = 'mixed'
        after_entropy = _median_or_none(all_after_values)
        states.append({
            'id': next_state_id,
            'label': current_state_type,
            'data_type': current_state_type,
            'byte_entropy': after_entropy,
            'unknown_entropy': after_entropy is None,
        })

    return {
        'data_type': data_type,
        'algorithm': algorithm,
        'method_name': method_name,
        'states': states,
        'edges': edges,
    }


class DatasetCatalogOperatorStepEntropyView(APIView):
    """Per-algorithm operator pipeline automata from ``*_operator_step_byte_entropy.csv``."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, dataset_key, *args, **kwargs):
        is_overall = str(dataset_key or '').strip().lower() == 'overall'
        focused = _focused_original_dataset_keys()
        if focused and not is_overall and dataset_key not in focused:
            raise Http404('Dataset is hidden by the current focused dataset set')
        if not is_overall and dataset_key not in _discovered_catalog_bundle_keys():
            raise Http404('Dataset bundle not found')
        rows = (
            _read_overall_operator_step_entropy_rows()
            if is_overall
            else _read_operator_step_entropy_rows(dataset_key)
        )
        algorithms = _operator_step_algorithm_options(
            rows,
            bundle_dir=None if is_overall else _operator_step_entropy_csv_path(dataset_key).parent,
        )
        selected_algorithm = str(request.query_params.get('algorithm', '') or '').strip()
        algorithm_keys = {row['algorithm'] for row in algorithms}
        if not selected_algorithm or selected_algorithm not in algorithm_keys:
            selected_algorithm = algorithms[0]['algorithm'] if algorithms else ''

        display_names = _operator_display_name_map()
        byte_operator_ids = {
            str(row.get('id', '') or '').strip()
            for row in load_byte_codec_operators()
            if str(row.get('id', '') or '').strip()
        }
        automata = {}
        if selected_algorithm:
            for dtype in ('int', 'float'):
                automata[dtype] = _build_operator_step_entropy_automaton(
                    rows=rows,
                    data_type=dtype,
                    algorithm=selected_algorithm,
                    display_names=display_names,
                    byte_operator_ids=byte_operator_ids,
                )
        return JsonResponse(
            _sanitize_for_json({
                'dataset_key': 'overall' if is_overall else dataset_key,
                'algorithms': algorithms,
                'selected_algorithm': selected_algorithm,
                'automata': automata,
            }),
            status=200,
        )


class DatasetCatalogCorrelationHeatmapsView(APIView):
    """Pearson correlation heatmaps for algorithm/operator vs column features."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, dataset_key, *args, **kwargs):
        from .dataset_float_precision_distribution import parse_float_precision_filter

        float_precision = parse_float_precision_filter(
            request.query_params.get('float_precision'),
        )
        is_overall = str(dataset_key or '').strip().lower() == 'overall'
        payload = (
            _overall_correlation_heatmap_payload(
                user=request.user,
                float_precision=float_precision,
            )
            if is_overall
            else _catalog_correlation_payload_for_dataset(
                dataset_key=dataset_key,
                user=request.user,
                float_precision=float_precision,
            )
        )
        if not payload:
            return JsonResponse(
                _sanitize_for_json(
                    {
                        'column_count': 0,
                        'value_based_algorithm_heatmap': None,
                        'byte_based_algorithm_heatmap': None,
                        'operator_heatmap': None,
                        'operator_entropy_change_boxplots': {},
                    }
                ),
                status=200,
            )
        return JsonResponse(_sanitize_for_json(payload), status=200)


class DatasetCatalogColumnDistributionChartView(APIView):
    """Serve per-column distribution PNG charts for dataset feature matrix cells."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, dataset_key, filename, *args, **kwargs):
        target = _resolve_catalog_column_distribution_chart_file(
            dataset_key,
            filename,
            request.user,
            _parse_catalog_operator(request),
        )
        if not target:
            raise Http404('Column chart not found')
        response = FileResponse(open(target, 'rb'), content_type='image/png')
        response['Content-Disposition'] = 'inline'
        return response


class DatasetColumnInfoView(APIView):
    """Column list from ``{key}_column_info.csv`` (built-in and user-upload datasets)."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, dataset_key, *args, **kwargs):
        path_csv = None
        if parse_user_upload_dataset_key(dataset_key) is not None:
            rec = get_user_upload_by_dataset_key(dataset_key, request.user, owner_only=True)
            if rec is None:
                raise Http404('Dataset not found')
            path_csv = resolve_user_upload_column_types_csv(rec)
        else:
            if dataset_key not in _all_dataset_keys_for_ui():
                raise Http404('Dataset not found')
            path_csv = column_info_path(dataset_key)
            if not os.path.isfile(path_csv):
                legacy = os.path.join(_dataset_bundle_dir(dataset_key), 'column_types.csv')
                path_csv = legacy if os.path.isfile(legacy) else None

        if not path_csv or not os.path.isfile(path_csv):
            return JsonResponse({'columns': []}, status=200)

        rows = load_column_info_rows(path_csv)
        columns = []
        for row in rows:
            name = str(row.get('column_name', '')).strip()
            if not name:
                continue
            logical = str(row.get('logical_type', '')).strip().lower()
            benchmark_dtype = None
            if logical in ('integer', 'int', 'long'):
                benchmark_dtype = 'int'
            elif logical in ('float', 'double'):
                benchmark_dtype = 'float'
            else:
                continue
            raw_idx = row.get('column_index')
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                idx = None
            label = f'{idx}: {name}' if idx is not None else name
            try:
                column_csv_size_bytes = int(float(row.get('column_csv_size_bytes', 0) or 0))
            except (TypeError, ValueError):
                column_csv_size_bytes = 0
            columns.append(
                {
                    'column_index': idx,
                    'column_name': name,
                    'logical_type': logical,
                    'column_csv_size_bytes': column_csv_size_bytes,
                    'benchmark_data_type': benchmark_dtype,
                    'label': label,
                }
            )
        return JsonResponse({'columns': _sanitize_for_json(columns)}, status=200)


class DatasetCatalogAssetFileView(APIView):
    """Serve catalog bundle assets and user-upload statistic/chart files."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, dataset_key, filename, *args, **kwargs):
        if not filename or '/' in filename or '\\' in filename or filename.startswith('.'):
            raise Http404('Invalid file name')
        if dataset_key in _discovered_catalog_bundle_keys():
            allowed = catalog_servable_asset_filenames(dataset_key) | {f'{dataset_key}.csv'}
            if filename not in allowed:
                raise Http404('File not available')
        elif parse_user_upload_dataset_key(dataset_key) is None:
            raise Http404('Dataset bundle not found')
        target = _resolve_catalog_asset_file(
            dataset_key,
            filename,
            request.user,
            _catalog_asset_data_type(request),
        )
        if not target:
            raise Http404('File not found')
        content_type, _ = mimetypes.guess_type(target)
        if not content_type:
            content_type = 'application/octet-stream'
        disposition = 'inline'
        if filename.endswith('.csv'):
            content_type = 'text/csv'
            disposition = f'attachment; filename="{filename}"'
        response = FileResponse(open(target, 'rb'), content_type=content_type)
        if disposition != 'inline':
            response['Content-Disposition'] = disposition
        return response


def _merge_user_algorithm_items(
    owner,
    dataset_key: str,
    data_type: str,
    implement_language: str,
    items: list[dict],
    unified_original: float,
    *,
    rate_mode: str = 'overall',
    compression_time_mode: str = 'overall',
    decompression_time_mode: str = 'overall',
) -> None:
    """Append rows from completed user algorithms (same language) into Home leaderboard items."""
    if unified_original <= 0:
        return
    for algo in UploadedAlgorithm.objects.filter(
        owner=owner,
        implement_language=implement_language,
        processing_status=UploadedAlgorithm.PROCESSING_COMPLETED,
    ):
        if data_type not in benchmark_column_types_for_algorithm(algo, [data_type]):
            continue
        path = user_algorithm_result_path(algo, dataset_key, data_type)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            logger.exception('Failed to read user algorithm result %s', path)
            continue
        items.extend(_staticdata_items_from_metric_rows(
            df,
            rate_mode=rate_mode,
            compression_time_mode=compression_time_mode,
            decompression_time_mode=decompression_time_mode,
            dataset_original_sizes={'__dataset__': unified_original},
        ))


def _staticdata_items_from_filtered_report(
    filtered_df,
    unified_original: float,
    *,
    rate_mode: str = 'overall',
    compression_time_mode: str = 'overall',
    decompression_time_mode: str = 'overall',
):
    if unified_original <= 0:
        return []
    return _staticdata_items_from_metric_rows(
        filtered_df,
        rate_mode=rate_mode,
        compression_time_mode=compression_time_mode,
        decompression_time_mode=decompression_time_mode,
        dataset_original_sizes={'__dataset__': unified_original},
    )


def _user_upload_staticdata_response(
    request,
    dataset_key: str,
    query_params,
    *,
    selected_column_index: int | None = None,
) -> JsonResponse | None:
    """Build staticdata JSON for ``user-upload-<pk>`` owned by the signed-in user."""
    pk = parse_user_upload_dataset_key(dataset_key)
    if pk is None:
        return None
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    try:
        record = UploadedDataset.objects.get(pk=pk, owner=request.user)
    except UploadedDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
    if record.processing_status != UploadedDataset.PROCESSING_COMPLETED:
        return JsonResponse(
            {'error': 'Dataset processing is not complete yet.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    data_type = query_params.get('data_type')
    implement_language = query_params.get('implement_language')
    if data_type not in ('int', 'float'):
        return JsonResponse({'error': 'Invalid data_type'}, status=status.HTTP_400_BAD_REQUEST)
    if implement_language not in ('java', 'python', 'cpp'):
        return JsonResponse({'error': 'Invalid implement_language'}, status=status.HTTP_400_BAD_REQUEST)

    parsed_filter_params = parse_report_filter_params(query_params)
    xz_level = parsed_filter_params.get('xz_level', 9)
    zstd_level = int(query_params.get('zstd_level', 9) or 9)
    simpiece_param = int(query_params.get('simpiece_param', 0) or 0)
    serf_param = int(query_params.get('serf_param', 0) or 0)
    sz_absrel = (query_params.get('sz_absrel', 'ABS') or 'ABS').upper()
    if sz_absrel not in ('ABS', 'REL'):
        sz_absrel = 'ABS'
    sz_eps = int(query_params.get('sz_eps', 0) or 0)
    sz3_absrel = (query_params.get('sz3_absrel', 'ABS') or 'ABS').upper()
    if sz3_absrel not in ('ABS', 'REL'):
        sz3_absrel = 'ABS'
    sz3_eps = int(query_params.get('sz3_eps', 0) or 0)
    machete_eps = int(query_params.get('machete_eps', 2) or 2)
    spdp_level = max(0, min(int(query_params.get('spdp_level', 9) or 9), 9))
    zfp_tol_eps = max(0, min(int(query_params.get('zfp_tol_eps', 2) or 2), 2))
    chimp_history = parsed_filter_params.get('chimp_history', 128)
    lzbench_levels = parsed_filter_params.get('lzbench_levels')
    float_precision = parse_float_precision_filter(query_params.get('float_precision'))
    rate_mode = _parse_compression_rate_mode(query_params.get('compression_rate_mode'))
    compression_time_mode = _parse_metric_aggregation_mode(
        query_params.get('compression_time_mode'),
    )
    decompression_time_mode = _parse_metric_aggregation_mode(
        query_params.get('decompression_time_mode'),
    )

    report_path = user_upload_report_csv_path(record, data_type, implement_language)
    if not os.path.isfile(report_path):
        return JsonResponse({'error': 'Compression result file not found'}, status=status.HTTP_404_NOT_FOUND)

    try:
        cur_df = pd.read_csv(report_path)
    except Exception as e:
        logger.exception('Failed to read user upload report %s', report_path)
        return JsonResponse({'error': f'Invalid compression result CSV: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    filtered_df = filter_compression_report_dataframe(
        cur_df,
        implement_language=implement_language,
        xz_level=xz_level,
        zstd_level=zstd_level,
        chimp_history=chimp_history,
        simpiece_param=simpiece_param,
        serf_param=serf_param,
        sz_absrel=sz_absrel,
        sz_eps=sz_eps,
        sz3_absrel=sz3_absrel,
        sz3_eps=sz3_eps,
        machete_eps=machete_eps,
        spdp_level=spdp_level,
        zfp_tol_eps=zfp_tol_eps,
        lzbench_levels=lzbench_levels,
    )
    filtered_df = _filter_report_for_column(filtered_df, selected_column_index)

    upload_dir = user_upload_abs_dir(record)
    column_info_csv = os.path.join(
        upload_dir,
        f'{user_upload_slug(record)}_column_info.csv',
    )
    if selected_column_index is None:
        filtered_df = _apply_dataset_level_float_precision_filter(
            filtered_df,
            data_type=data_type,
            data_dimension='single_dataset',
            float_precision=float_precision,
            column_info_path=column_info_csv,
            bundle_dir=upload_dir,
        )
    unified_original = _sum_column_csv_size_bytes(
        column_info_csv,
        data_type,
        selected_column_index=selected_column_index,
        float_precision=float_precision,
        bundle_dir=upload_dir,
        apply_float_precision_filter=(selected_column_index is None),
    )
    if unified_original <= 0 and 'originalSize' in filtered_df.columns:
        unified_original = _max_original_size_from_report(filtered_df)

    items = _staticdata_items_from_filtered_report(
        filtered_df,
        unified_original,
        rate_mode=rate_mode,
        compression_time_mode=compression_time_mode,
        decompression_time_mode=decompression_time_mode,
    )
    _merge_user_algorithm_items(
        record.owner,
        dataset_key,
        data_type,
        implement_language,
        items,
        unified_original,
        rate_mode=rate_mode,
        compression_time_mode=compression_time_mode,
        decompression_time_mode=decompression_time_mode,
    )
    items = _filter_staticdata_items_hide_lossy(items)

    statistic_data = []
    stat_path = os.path.join(
        user_upload_abs_dir(record),
        f'{record.display_name}_{data_type}_statistic.csv',
    )
    if os.path.isfile(stat_path):
        try:
            statistic_data = pd.read_csv(stat_path).to_dict(orient='records')
        except Exception:
            logger.exception('Failed to read statistic CSV %s', stat_path)

    chart_urls = []
    chart_paths = {
        'scaling_chart': os.path.join(
            user_upload_abs_dir(record), f'{record.display_name}_{data_type}_scaling.png'
        ),
        'xor_leading_chart': os.path.join(
            user_upload_abs_dir(record), f'{record.display_name}_{data_type}_xor_leading.png'
        ),
        'xor_trailing_chart': os.path.join(
            user_upload_abs_dir(record), f'{record.display_name}_{data_type}_xor_trailing.png'
        ),
    }
    for chart_name, chart_path in chart_paths.items():
        if chart_path and os.path.exists(chart_path):
            chart_urls.append({
                'name': chart_name,
                'url': f'/api/my-uploads/{record.pk}/charts/{chart_name}/?data_type={data_type}',
            })

    return JsonResponse(
        _sanitize_for_json({
            'items': items,
            'charts': chart_urls,
            'statistics': statistic_data,
        }),
        status=200,
    )


class UploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def _post_user_dataset_upload(self, request, file):
        """Save upload and run full py/java/cpp benchmark pipeline in the background."""
        from .upload_serializers import UploadedDatasetSerializer

        process_id = str(uuid.uuid4())[:8]
        rel_dir = os.path.join('user_datasets', str(request.user.pk), process_id)
        dest_dir = os.path.join(settings.MEDIA_ROOT, rel_dir)
        os.makedirs(dest_dir, exist_ok=True)
        uploaded_file_path = os.path.join(dest_dir, file.name)
        with open(uploaded_file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        file_name = os.path.splitext(os.path.basename(file.name))[0]
        dataset_ext = os.path.splitext(uploaded_file_path)[1][1:].lower()
        if dataset_ext not in ('csv', 'arrow'):
            return Response(
                {'error': 'Invalid file format. Only CSV and Arrow files are allowed.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if dataset_ext == 'csv':
            try:
                pd.read_csv(uploaded_file_path, nrows=5)
            except Exception:
                return Response({'error': 'Error reading CSV file.'}, status=status.HTTP_400_BAD_REQUEST)

        is_public = parse_bool_form_value(request.data.get('is_public'))
        major_category = normalize_major_category(request.data.get('major_category'))
        try:
            record = create_user_upload(
                user=request.user,
                storage_id=process_id,
                display_name=file_name,
                original_filename=file.name,
                file_format=dataset_ext,
                is_public=is_public,
                major_category=major_category,
                uploaded_path=uploaded_file_path,
            )
        except Exception as exc:
            logger.exception('Failed to create user upload for %s', request.user.email)
            if os.path.isdir(dest_dir):
                shutil.rmtree(dest_dir, ignore_errors=True)
            return Response({'error': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        start_user_upload_pipeline(record.pk)
        payload = {
            'saved_upload': UploadedDatasetSerializer(record).data,
            'message': (
                'Upload saved. Benchmarks (Python, Java, C++, TsFile) are running in the background. '
                'Home and TsFile pages will show results when processing completes.'
            ),
        }
        return Response(payload, status=status.HTTP_202_ACCEPTED)

    def post(self, request, *args, **kwargs):
        file = request.data.get('file', None)
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        return self._post_user_dataset_upload(request, file)


class StaticDataView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        data_dimension = request.query_params.get('data_dimension', 'single_dataset')
        column_index_raw = (request.query_params.get('column_index') or '').strip()
        selected_column_index = None
        if column_index_raw:
            try:
                selected_column_index = int(column_index_raw)
            except (TypeError, ValueError):
                return JsonResponse({'error': 'Invalid column_index'}, status=status.HTTP_400_BAD_REQUEST)
        dataset = request.query_params.get('dataset', None)
        category = (request.query_params.get('category') or '').strip()
        logger.debug(
            'staticdata dimension=%s dataset=%s category=%s column_index=%s',
            data_dimension,
            dataset,
            category,
            column_index_raw,
        )

        if data_dimension == 'single_field':
            if not category:
                return JsonResponse({'error': 'Missing category parameter'}, status=status.HTTP_400_BAD_REQUEST)
        elif data_dimension == 'single_dataset':
            if dataset is None:
                return JsonResponse({'error': 'Missing dataset parameter'}, status=status.HTTP_400_BAD_REQUEST)
        elif data_dimension == 'single_column':
            if dataset is None:
                return JsonResponse({'error': 'Missing dataset parameter'}, status=status.HTTP_400_BAD_REQUEST)
            if selected_column_index is None:
                return JsonResponse({'error': 'Missing column_index parameter'}, status=status.HTTP_400_BAD_REQUEST)
        elif data_dimension != 'all_datasets':
            return JsonResponse({'error': f'Invalid data_dimension: {data_dimension}'}, status=status.HTTP_400_BAD_REQUEST)

        data_type = str(request.query_params.get('data_type', '') or '').strip().lower()
        report_data_types = _staticdata_report_data_types(data_type)
        if not report_data_types:
            return JsonResponse({'error': 'Invalid data_type'}, status=status.HTTP_400_BAD_REQUEST)
        if data_type == 'overall' and data_dimension == 'single_column':
            return JsonResponse(
                {'error': 'Overall data_type is not available for single_column.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        implement_language = request.query_params.get('implement_language', None)
        parsed_filter_params = parse_report_filter_params(request.query_params)
        leaderboard_mode = str(request.query_params.get('leaderboard_mode') or '').strip().lower()
        float_precision = parse_float_precision_filter(
            request.query_params.get('float_precision'),
        )
        if (
            leaderboard_mode == 'paper_table3'
            and data_dimension == 'all_datasets'
        ):
            from .paper_table3_leaderboard import (
                paper_table3_display_rows,
                paper_table3_scope_for_request,
            )

            scope = paper_table3_scope_for_request(data_type, float_precision)
            if scope is not None:
                try:
                    table_rows = paper_table3_display_rows(
                        Path(settings.BASE_DIR),
                        data_type=data_type,
                        float_precision=float_precision,
                        implement_language='cpp',
                        dataset_keys=_all_dataset_keys_for_ui(),
                    )
                    items = [
                        {
                            'algorithm': str(row.get('method', '')),
                            'compression_rate': float(row['average_rate']),
                            'compression_time': float(row['average_compression_ns']),
                            'decompression_time': float(row['average_decompression_ns']),
                            'balanced_score': float(row['median_balance_score']),
                            'average_rank': float(row['rank']),
                        }
                        for row in table_rows
                    ]
                except Exception:
                    logger.exception('Failed to build paper Table 3 leaderboard rows')
                    return JsonResponse(
                        {'error': 'Failed to build paper Table 3 leaderboard rows'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )
                return JsonResponse(
                    _sanitize_for_json({
                        'items': items,
                        'charts': [],
                        'statistics': [],
                    }),
                    status=200,
                )
        xz_level = parsed_filter_params.get('xz_level', 9)
        zstd_level = request.query_params.get('zstd_level', None)

        # 设置默认值
        zstd_level = int(zstd_level) if zstd_level is not None else 9

        simpiece_param = int(request.query_params.get('simpiece_param', 0))
        serf_param = int(request.query_params.get('serf_param', 0))
        sz_absrel = (request.query_params.get('sz_absrel', 'ABS') or 'ABS').upper()
        if sz_absrel not in ('ABS', 'REL'):
            sz_absrel = 'ABS'
        sz_eps = int(request.query_params.get('sz_eps', 0))
        sz3_absrel = (request.query_params.get('sz3_absrel', 'ABS') or 'ABS').upper()
        if sz3_absrel not in ('ABS', 'REL'):
            sz3_absrel = 'ABS'
        sz3_eps = int(request.query_params.get('sz3_eps', 0))
        machete_eps = int(request.query_params.get('machete_eps', 2))
        spdp_level = max(0, min(int(request.query_params.get('spdp_level', 9) or 9), 9))
        zfp_tol_eps = max(0, min(int(request.query_params.get('zfp_tol_eps', 2) or 2), 2))
        chimp_history = parsed_filter_params.get('chimp_history', 128)
        lzbench_levels = parsed_filter_params.get('lzbench_levels')
        rate_mode = _parse_compression_rate_mode(
            request.query_params.get('compression_rate_mode'),
        )
        compression_time_mode = _parse_metric_aggregation_mode(
            request.query_params.get('compression_time_mode'),
        )
        decompression_time_mode = _parse_metric_aggregation_mode(
            request.query_params.get('decompression_time_mode'),
        )
        apply_float_precision = (
            data_type == 'float'
            and data_dimension != 'single_column'
            and float_precision != 'all'
        )

        def _read_and_filter_report(report_path):
            try:
                cur_df = pd.read_csv(report_path)
            except Exception as e:
                logger.exception('Failed to read compression report %s', report_path)
                raise ValueError(f'Invalid compression result CSV: {e}')
            return filter_compression_report_dataframe(
                cur_df,
                implement_language=implement_language,
                xz_level=xz_level,
                zstd_level=zstd_level,
                chimp_history=chimp_history,
                simpiece_param=simpiece_param,
                serf_param=serf_param,
                sz_absrel=sz_absrel,
                sz_eps=sz_eps,
                sz3_absrel=sz3_absrel,
                sz3_eps=sz3_eps,
                machete_eps=machete_eps,
                spdp_level=spdp_level,
                zfp_tol_eps=zfp_tol_eps,
                lzbench_levels=lzbench_levels,
            )

        chart_urls = []
        statistic_data = []

        def _aggregate_staticdata_frames(dataset_names: list[str]):
            frames = []
            for ds_name in dataset_names:
                ds_dir = _dataset_bundle_dir(ds_name)
                if not os.path.isdir(ds_dir):
                    continue
                column_info_csv = os.path.join(ds_dir, f'{ds_name}_column_info.csv')
                for report_data_type in report_data_types:
                    report_name = f'{ds_name}_{report_data_type}_{implement_language}_compression_result.csv'
                    report_path = os.path.join(ds_dir, report_name)
                    if not os.path.isfile(report_path):
                        continue
                    try:
                        cur = _read_and_filter_report(report_path)
                    except ValueError:
                        continue
                    if (
                        report_data_type == 'float'
                        and data_dimension != 'single_column'
                        and float_precision != 'all'
                    ):
                        cur = _apply_dataset_level_float_precision_filter(
                            cur,
                            data_type=report_data_type,
                            data_dimension=data_dimension,
                            float_precision=float_precision,
                            column_info_path=column_info_csv,
                            bundle_dir=ds_dir,
                        )
                    if cur.empty:
                        continue
                    cur = cur.copy()
                    cur['algorithm'] = cur['algorithm'].astype(str)
                    cur['dataset_key'] = ds_name
                    frames.append(cur[[
                        'dataset_key',
                        'algorithm',
                        'originalSize',
                        'compressedSize',
                        'compressionTime',
                        'decompressionTime',
                    ]])

            if not frames:
                return []

            filtered_df = pd.concat(frames, ignore_index=True)
            return _staticdata_items_from_metric_rows(
                filtered_df,
                rate_mode=rate_mode,
                compression_time_mode=compression_time_mode,
                decompression_time_mode=decompression_time_mode,
            )

        if data_dimension == 'all_datasets':
            if not os.path.isdir(_DATASET_ROOT):
                return JsonResponse({'error': 'Dataset root not found'}, status=status.HTTP_404_NOT_FOUND)

            dataset_names = [
                name for name in _all_dataset_keys_for_ui()
                if os.path.isdir(_dataset_bundle_dir(name))
            ]
            items = _aggregate_staticdata_frames(dataset_names)
            return JsonResponse(
                _sanitize_for_json({
                    'items': items,
                    'charts': chart_urls,
                    'statistics': statistic_data,
                }),
                status=200,
            )

        if data_dimension == 'single_field':
            dataset_names = _dataset_keys_for_major_category(category)
            items = _aggregate_staticdata_frames(dataset_names)
            return JsonResponse(
                _sanitize_for_json({
                    'items': items,
                    'charts': chart_urls,
                    'statistics': statistic_data,
                    'category': category,
                    'datasets_in_category': dataset_names,
                }),
                status=200,
            )
        # single_dataset / single_column
        if data_type == 'overall':
            items = _aggregate_staticdata_frames([dataset])
            return JsonResponse(
                _sanitize_for_json({
                    'items': items,
                    'charts': chart_urls,
                    'statistics': statistic_data,
                }),
                status=200,
            )

        user_upload_resp = _user_upload_staticdata_response(
            request,
            dataset,
            request.query_params,
            selected_column_index=(selected_column_index if data_dimension == 'single_column' else None),
        )
        if user_upload_resp is not None:
            return user_upload_resp

        dataset_dir = _dataset_bundle_dir(dataset)
        if not os.path.isdir(dataset_dir):
            return JsonResponse({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

        output_report_name = f'{dataset}_{data_type}_{implement_language}_compression_result.csv'
        output_report_path = os.path.join(dataset_dir, output_report_name)
        if not os.path.exists(output_report_path):
            return JsonResponse({'error': 'Compression result file not found'}, status=status.HTTP_404_NOT_FOUND)

        column_info_csv = os.path.join(dataset_dir, f'{dataset}_column_info.csv')

        scaling_chart_path = os.path.join(dataset_dir, f'{dataset}_{data_type}_scaling.png')
        xor_leading_chart_path = os.path.join(dataset_dir, f'{dataset}_{data_type}_xor_leading.png')
        xor_trailing_chart_path = os.path.join(dataset_dir, f'{dataset}_{data_type}_xor_trailing.png')

        chart_paths = {
            'scaling_chart': scaling_chart_path,
            'xor_leading_chart': xor_leading_chart_path,
            'xor_trailing_chart': xor_trailing_chart_path
        }
        try:
            filtered_df = _read_and_filter_report(output_report_path)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        if data_dimension == 'single_column':
            filtered_df = _filter_report_for_column(filtered_df, selected_column_index)
        elif apply_float_precision:
            filtered_df = _apply_dataset_level_float_precision_filter(
                filtered_df,
                data_type=data_type,
                data_dimension=data_dimension,
                float_precision=float_precision,
                column_info_path=column_info_csv,
                bundle_dir=dataset_dir,
            )

        # One unified original size for compression rate on every row (fair cross-algorithm ranking).
        # Primary: dtype-filtered sum(column_csv_size_bytes) from column_info. Fallback: max(originalSize).
        has_row_original = 'originalSize' in filtered_df.columns
        unified_original = _sum_column_csv_size_bytes(
            column_info_csv,
            data_type,
            selected_column_index=(selected_column_index if data_dimension == 'single_column' else None),
            float_precision=float_precision,
            bundle_dir=dataset_dir,
            apply_float_precision_filter=apply_float_precision,
        )
        if unified_original <= 0 and has_row_original:
            unified_original = _max_original_size_from_report(filtered_df)
        if unified_original <= 0:
            logger.warning(
                'StaticDataView: unified original size is 0; check %s and CSV originalSize columns.',
                column_info_csv,
            )

        items = _staticdata_items_from_filtered_report(
            filtered_df,
            unified_original,
            rate_mode=rate_mode,
            compression_time_mode=compression_time_mode,
            decompression_time_mode=decompression_time_mode,
        )
        if request.user.is_authenticated:
            _merge_user_algorithm_items(
                request.user,
                dataset,
                data_type,
                implement_language,
                items,
                unified_original,
                rate_mode=rate_mode,
                compression_time_mode=compression_time_mode,
                decompression_time_mode=decompression_time_mode,
            )
        items = _filter_staticdata_items_hide_lossy(items)

        # 读取统计数据
        statistic_csv_name = f'{dataset}_{data_type}_statistic.csv'
        statistic_csv_path = os.path.join(dataset_dir, statistic_csv_name)

        if os.path.exists(statistic_csv_path):
            statistic_df_data = pd.read_csv(statistic_csv_path)
            statistic_data = statistic_df_data.to_dict(orient='records')

        # 准备图表URL列表
        chart_urls = []
        for chart_name, chart_path in chart_paths.items():
            if chart_path and os.path.exists(chart_path):
                relative_path = os.path.relpath(chart_path, _DATASET_ROOT)
                chart_url = f"/static-chart/{relative_path.replace(os.sep, '/')}"
                chart_urls.append({
                    'name': chart_name,
                    'url': chart_url
                })

        return JsonResponse(
            _sanitize_for_json({
                'items': items,
                'charts': chart_urls,
                'statistics': statistic_data,
            }),
            status=200,
        )


def _staticdata_dataset_keys_with_java_report():
    """Dataset keys under backend/dataset/ with at least one ``*_java_compression_result.csv``."""
    root = _DATASET_ROOT
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            files = os.listdir(path)
        except OSError:
            continue
        if any(f.endswith('_java_compression_result.csv') for f in files):
            out.append(name)
    return _filter_focused_builtin_dataset_keys(out)


_TSFILE_STATIC_ARTIFACT_ALGO_RE = re.compile(r'^[A-Za-z0-9_.+-]+$')


def _tsfile_benchmark_items_from_report(output_report_path: str, unified_original: float) -> list[dict]:
    try:
        df = pd.read_csv(output_report_path)
    except Exception as e:
        raise ValueError(f'Invalid TsFile benchmark CSV: {e}') from e

    df = df.dropna(subset=['algorithm'])
    algo = df['algorithm'].astype(str)
    filtered_df = df[
        (~algo.str.startswith('xz_level_', na=False))
        & (~algo.str.startswith('lzma_level_', na=False))
        & (~algo.str.startswith('zstd_level_', na=False))
    ]

    has_row_original = 'originalSize' in filtered_df.columns
    if unified_original <= 0 and has_row_original:
        try:
            unified_original = float(filtered_df['originalSize'].max())
        except (TypeError, ValueError, KeyError):
            unified_original = 0.0

    items = []
    for _, row in filtered_df.iterrows():
        try:
            csize = row['compressedSize']
            if pd.isna(csize) or float(csize) <= 0:
                continue
            if unified_original <= 0:
                continue
            items.append({
                'algorithm': row['algorithm'],
                'compression_rate': float(csize) / unified_original,
                'compression_time': float(row['compressionTime']) / 1e9,
                'decompression_time': float(row['decompressionTime']) / 1e9,
                'space_cost': int(float(csize)),
            })
        except (TypeError, ValueError, KeyError):
            continue
    return items


def _user_upload_tsfile_dataset_keys(user) -> list[str]:
    if user is None or not getattr(user, 'is_authenticated', False):
        return []
    keys = []
    for rec in UploadedDataset.objects.filter(
        owner=user,
        processing_status=UploadedDataset.PROCESSING_COMPLETED,
    ).order_by('-created_at'):
        if user_upload_has_tsfile_benchmark(rec):
            keys.append(user_upload_dataset_key(rec))
    return keys


def _staticdata_tsfile_dataset_keys():
    """Dataset keys under backend/dataset/ with ``*_tsfile_compression_result.csv`` (TsFile page)."""
    root = _DATASET_ROOT
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            files = os.listdir(path)
        except OSError:
            continue
        if any(f.endswith('_tsfile_compression_result.csv') for f in files):
            out.append(name)
    return _filter_focused_builtin_dataset_keys(out)


class StaticDataManifestView(APIView):
    """List dataset keys that have precomputed Java compression CSVs (same tree as Home staticdata)."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        tsfile_keys = list(_staticdata_tsfile_dataset_keys())
        include_user = parse_bool_form_value(
            request.query_params.get('include_user_uploads'),
        )
        if include_user:
            for key in _user_upload_tsfile_dataset_keys(request.user):
                if key not in tsfile_keys:
                    tsfile_keys.append(key)
        return JsonResponse(
            _sanitize_for_json({
                'datasets': _staticdata_dataset_keys_with_java_report(),
                'tsfile_datasets': tsfile_keys,
            }),
            status=200,
        )


class TsFileStaticDataView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        dataset = request.query_params.get('dataset', None)
        if dataset is None:
            return JsonResponse({'error': 'Missing dataset parameter'}, status=status.HTTP_400_BAD_REQUEST)

        data_type = request.query_params.get('data_type', None)
        if data_type not in ('int', 'float'):
            return JsonResponse({'error': 'Invalid data_type'}, status=status.HTTP_400_BAD_REQUEST)

        user_record = get_user_upload_by_dataset_key(dataset, request.user, owner_only=True)
        if user_record is not None:
            if user_record.processing_status != UploadedDataset.PROCESSING_COMPLETED:
                return JsonResponse(
                    {'error': 'TsFile benchmarks are still processing for this upload.'},
                    status=status.HTTP_404_NOT_FOUND,
                )
            output_report_path = user_upload_tsfile_report_path(user_record, data_type)
            if not os.path.isfile(output_report_path):
                return JsonResponse(
                    {'error': f'No TsFile benchmark for {data_type} columns in this upload.'},
                    status=status.HTTP_404_NOT_FOUND,
                )
            slug = user_upload_slug(user_record)
            column_info_csv = os.path.join(user_upload_abs_dir(user_record), f'{slug}_column_info.csv')
        else:
            dataset_dir = _dataset_bundle_dir(dataset)
            if not os.path.isdir(dataset_dir):
                return JsonResponse({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

            output_report_name = f'{dataset}_{data_type}_tsfile_compression_result.csv'
            output_report_path = os.path.join(dataset_dir, output_report_name)
            if not os.path.exists(output_report_path):
                return JsonResponse({'error': 'TsFile benchmark file not found'}, status=status.HTTP_404_NOT_FOUND)

            column_info_csv = os.path.join(dataset_dir, f'{dataset}_column_info.csv')

        unified_original = _sum_column_csv_size_bytes(
            column_info_csv,
            data_type,
            include_integer_for_float=True,
        )
        try:
            items = _tsfile_benchmark_items_from_report(output_report_path, unified_original)
        except ValueError as e:
            logger.exception('Failed to read TsFile benchmark report %s', output_report_path)
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        items = _filter_staticdata_items_hide_lossy(items)

        return JsonResponse(
            _sanitize_for_json({
                'items': items,
                'charts': [],
                'statistics': [],
            }),
            status=200,
        )


class TsFileStaticArtifactDownloadView(APIView):
    """Download one per-algorithm ``.tsfile`` from staticdata or user upload ``tsfile_artifacts/``."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        dataset = request.query_params.get('dataset')
        data_type = request.query_params.get('data_type')
        algorithm = request.query_params.get('algorithm')
        if not dataset or not data_type or not algorithm:
            return JsonResponse({'error': 'Missing dataset, data_type, or algorithm'}, status=400)
        if data_type not in ('int', 'float'):
            return JsonResponse({'error': 'Invalid data_type'}, status=400)
        if not _TSFILE_STATIC_ARTIFACT_ALGO_RE.match(algorithm):
            return JsonResponse({'error': 'Invalid algorithm'}, status=400)

        user_record = get_user_upload_by_dataset_key(dataset, request.user, owner_only=True)
        if user_record is not None:
            if user_record.processing_status != UploadedDataset.PROCESSING_COMPLETED:
                return JsonResponse({'error': 'Upload processing is not complete.'}, status=404)
            path = resolve_user_upload_tsfile_artifact_path(
                user_record, data_type, algorithm,
            )
            if not path:
                return JsonResponse({'error': 'TsFile artifact not found for this algorithm.'}, status=404)
            fname = os.path.basename(path)
        else:
            if dataset not in _staticdata_tsfile_dataset_keys():
                return JsonResponse(
                    {'error': 'Dataset has no TsFile artifacts (missing tsfile_compression_result CSV).'},
                    status=404,
                )
            fname = f'{dataset}_{algorithm}.tsfile'
            path = os.path.join(
                tsfile_bundle_dir(dataset), 'tsfile_artifacts', data_type, fname,
            )

        if not os.path.isfile(path):
            return JsonResponse(
                {
                    'error': (
                        'TsFile artifact not found on the server. For curated datasets run '
                        'scripts/regenerate_tsfile_staticdata.py; for uploads wait until '
                        'background processing completes.'
                    ),
                },
                status=404,
            )

        response = FileResponse(open(path, 'rb'), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="{fname}"'
        return response


class ResearchCompressionCatalogView(APIView):
    """Literature-linked compressor repos (lossy/lossless, language, data domain)."""

    def get(self, request, *args, **kwargs):
        repo_root = Path(__file__).resolve().parent.parent.parent
        csv_path = repo_root / 'benchmark' / 'reference' / 'research_algo_repos_classified.csv'
        if not csv_path.is_file():
            return JsonResponse({'error': 'Catalog file missing'}, status=404)

        df = pd.read_csv(csv_path)
        lossy = request.query_params.get('lossy_lossless')
        lang = request.query_params.get('language')
        cat = request.query_params.get('benchmark_category')

        filtered = df
        if lossy:
            filtered = filtered[filtered['lossy_lossless'] == lossy]
        if lang:
            filtered = filtered[filtered['language'].str.contains(lang, case=False, na=False)]
        if cat:
            filtered = filtered[filtered['benchmark_category'] == cat]

        items = filtered.fillna('').to_dict(orient='records')
        filter_options = {
            'lossy_lossless': sorted(df['lossy_lossless'].dropna().unique().tolist()),
            'language': sorted(df['language'].dropna().unique().tolist()),
            'benchmark_category': sorted(df['benchmark_category'].dropna().unique().tolist()),
        }
        return JsonResponse({'items': items, 'filter_options': filter_options}, status=200)


class StaticChartImageView(APIView):
    def get(self, request, chart_path, *args, **kwargs):
        try:
            # 构造完整的静态文件路径
            full_chart_path = os.path.join(_DATASET_ROOT, chart_path)

            logger.debug(f"Serving static chart image from: {full_chart_path}")

            if not os.path.exists(full_chart_path):
                raise Http404("Static chart image not found")

            # 确定文件的 MIME 类型
            content_type, _ = mimetypes.guess_type(full_chart_path)
            if content_type is None:
                content_type = 'image/png'

            # 返回文件响应
            response = FileResponse(
                open(full_chart_path, 'rb'),
                content_type=content_type
            )
            return response

        except Exception as e:
            logger.error(f"Static chart image error: {str(e)}")
            raise Http404(f"Static chart image not found: {str(e)}")


class BenchmarkBoxplotManifestView(APIView):
    """Return leaderboard boxplot chart data keyed by ranking criterion."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        from .benchmark_boxplot_result_store import (
            boxplot_groups_for_api,
            load_precomputed_home_leaderboard_boxplots,
            pareto_chart_for_api,
        )
        from .home_leaderboard_boxplot_data import build_home_leaderboard_boxplot_groups

        payload = load_precomputed_home_leaderboard_boxplots()
        groups = None
        dynamic_pareto = None
        query_params = request.query_params
        if query_params:
            data_dimension = str(query_params.get('data_dimension') or 'single_dataset').strip() or 'single_dataset'
            dataset = query_params.get('dataset', None)
            category = str(query_params.get('category') or '').strip()
            column_index_raw = str(query_params.get('column_index') or '').strip()
            selected_column_index = None
            if column_index_raw:
                try:
                    selected_column_index = int(column_index_raw)
                except (TypeError, ValueError):
                    return JsonResponse({'error': 'Invalid column_index'}, status=status.HTTP_400_BAD_REQUEST)
            if data_dimension == 'single_field':
                if not category:
                    return JsonResponse({'error': 'Missing category parameter'}, status=status.HTTP_400_BAD_REQUEST)
                dataset_keys = _dataset_keys_for_major_category(category)
            elif data_dimension == 'single_dataset':
                if dataset is None:
                    return JsonResponse({'error': 'Missing dataset parameter'}, status=status.HTTP_400_BAD_REQUEST)
                dataset_keys = [str(dataset)]
            elif data_dimension == 'single_column':
                if dataset is None:
                    return JsonResponse({'error': 'Missing dataset parameter'}, status=status.HTTP_400_BAD_REQUEST)
                if selected_column_index is None:
                    return JsonResponse({'error': 'Missing column_index parameter'}, status=status.HTTP_400_BAD_REQUEST)
                dataset_keys = [str(dataset)]
            elif data_dimension == 'all_datasets':
                dataset_keys = _all_dataset_keys_for_ui()
            else:
                return JsonResponse(
                    {'error': f'Invalid data_dimension: {data_dimension}'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            data_type = str(query_params.get('data_type') or 'float').strip().lower() or 'float'
            if not _staticdata_report_data_types(data_type):
                return JsonResponse({'error': 'Invalid data_type'}, status=status.HTTP_400_BAD_REQUEST)
            if data_type == 'overall' and data_dimension == 'single_column':
                return JsonResponse(
                    {'error': 'Overall data_type is not available for single_column.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                filter_params = parse_report_filter_params(query_params)
                leaderboard_mode = str(query_params.get('leaderboard_mode') or '').strip().lower()
                if leaderboard_mode == 'paper_table3' and data_dimension == 'all_datasets':
                    from .paper_table3_leaderboard import paper_table3_eligible_dataset_keys

                    dataset_keys = paper_table3_eligible_dataset_keys(
                        Path(settings.BASE_DIR),
                        data_type=data_type,
                        float_precision=parse_float_precision_filter(query_params.get('float_precision')),
                        implement_language='cpp',
                        dataset_keys=dataset_keys,
                    )
                groups_obj, dynamic_pareto, _datasets_used = build_home_leaderboard_boxplot_groups(
                    Path(settings.BASE_DIR),
                    implement_language='cpp' if leaderboard_mode == 'paper_table3' else (str(query_params.get('implement_language') or 'java').strip().lower() or 'java'),
                    data_type=data_type,
                    filter_params=filter_params,
                    dataset_keys=dataset_keys,
                    selected_column_index=selected_column_index,
                    float_precision=parse_float_precision_filter(query_params.get('float_precision')),
                    leaderboard_mode=leaderboard_mode,
                )
                groups = boxplot_groups_for_api(
                    {'groups': groups_obj},
                    hide_lossy=HIDE_LOSSY_ALGORITHMS,
                )
            except Exception:
                logger.exception('Failed to build parameterized leaderboard boxplots')
                return JsonResponse(
                    {'error': 'Failed to build parameterized leaderboard boxplots'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        if groups is None:
            groups = boxplot_groups_for_api(payload, hide_lossy=HIDE_LOSSY_ALGORITHMS)
        pareto_payload = {'pareto_chart': dynamic_pareto} if isinstance(dynamic_pareto, dict) else payload
        pareto_chart = pareto_chart_for_api(pareto_payload)
        meta = payload.get('meta') if isinstance(payload, dict) else None
        return JsonResponse(
            {
                'groups': _sanitize_for_json(groups),
                'pareto_chart': _sanitize_for_json(pareto_chart),
                'meta': _sanitize_for_json(meta if isinstance(meta, dict) else {}),
            },
            status=200,
        )
