"""Read/write precomputed Dataset catalog summary (overview + pie charts) under ``backend/dataset_result/``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings

# Bump when summary schema changes (e.g. pie-chart compression-rate fields).
CATALOG_RESULT_VERSION = 5
CATALOG_ITEMS_VERSION = 1
RESULT_DIRNAME = 'dataset_result'
FULL_SUMMARY_FILENAME = 'catalog_public_full.json'
ITEMS_FILENAME = 'catalog_public_items.json'
MANIFEST_FILENAME = 'manifest.json'
LEDGER_SUMMARY_FILENAME = 'ledger_public_timestamp_summary.json'
LEDGER_COMPRESS_SUMMARY_FILENAME = 'ledger_public_compress_summary.json'

# Overview stat cards 5–6 (Overall / Average compression rate) and pie-chart rate rows.
CATALOG_SUMMARY_OVERVIEW_RATE_KEYS = (
    'overall_compression_rate',
    'average_compression_rate',
)

CATALOG_SUMMARY_COMPRESSION_DISTRIBUTION_KEYS = (
    'category_overall_compression_rate_distribution',
    'category_average_compression_rate_distribution',
    'dtype_overall_compression_rate_distribution',
    'dtype_average_compression_rate_distribution',
    'float_precision_overall_compression_rate_distribution',
    'float_precision_average_compression_rate_distribution',
)

CATALOG_SUMMARY_REQUIRED_KEYS = (
    'dataset_count',
    'total_rows',
    'total_size_bytes',
    'total_tsfile_size_bytes',
    'total_compressed_size_bytes',
    *CATALOG_SUMMARY_OVERVIEW_RATE_KEYS,
    'category_distribution',
    'category_size_distribution',
    'category_data_point_distribution',
    'category_tsfile_size_distribution',
    *CATALOG_SUMMARY_COMPRESSION_DISTRIBUTION_KEYS,
    'dtype_column_distribution',
    'dtype_data_point_distribution',
    'dtype_csv_size_distribution',
    'dtype_tsfile_size_distribution',
    'float_precision_column_distribution',
    'float_precision_data_point_distribution',
    'float_precision_csv_size_distribution',
    'float_precision_tsfile_size_distribution',
    'available_categories',
)

CATALOG_SUMMARY_STRUCTURAL_DTYPE_KEYS = (
    'dtype_column_distribution',
    'dtype_data_point_distribution',
    'dtype_csv_size_distribution',
    'dtype_tsfile_size_distribution',
)


def dataset_result_dir() -> Path:
    return Path(settings.BASE_DIR) / RESULT_DIRNAME


def catalog_summary_path() -> Path:
    return dataset_result_dir() / FULL_SUMMARY_FILENAME


def catalog_items_path() -> Path:
    return dataset_result_dir() / ITEMS_FILENAME


def manifest_path() -> Path:
    return dataset_result_dir() / MANIFEST_FILENAME


def ledger_catalog_summary_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / 'benchmark'
        / 'reference'
        / 'datasets'
        / LEDGER_SUMMARY_FILENAME
    )


def ledger_public_compress_summary_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / 'benchmark'
        / 'reference'
        / 'datasets'
        / LEDGER_COMPRESS_SUMMARY_FILENAME
    )


def _is_finite_number(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    return num == num and num != float('inf') and num != float('-inf')


def validate_catalog_summary(summary: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors; empty list means OK."""
    errors: list[str] = []
    if not isinstance(summary, dict):
        return ['summary must be a dict']

    for key in CATALOG_SUMMARY_REQUIRED_KEYS:
        if key not in summary:
            errors.append(f'missing key: {key}')

    for key in CATALOG_SUMMARY_OVERVIEW_RATE_KEYS:
        if key in summary and not _is_finite_number(summary.get(key)):
            errors.append(f'{key} must be a finite number')

    for key in CATALOG_SUMMARY_COMPRESSION_DISTRIBUTION_KEYS:
        rows = summary.get(key)
        if rows is None:
            continue
        if not isinstance(rows, list):
            errors.append(f'{key} must be a list')
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f'{key}[{idx}] must be an object')
                continue
            if not _is_finite_number(row.get('compression_rate')):
                if not _is_finite_number(row.get('value')):
                    errors.append(
                        f'{key}[{idx}] missing compression_rate/value number',
                    )

    return errors


def _normalized_distribution_rows(rows: object, *, value_field: str) -> object:
    if not isinstance(rows, list):
        return rows
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            normalized.append(row)
            continue
        out = dict(row)
        name = str(out.get('category') or out.get('name') or '').strip()
        if name and 'category' not in out:
            out['category'] = name
        if value_field not in out and 'value' in out:
            out[value_field] = out['value']
        if 'value' not in out and value_field in out:
            out['value'] = out[value_field]
        normalized.append(out)
    return normalized


def _normalized_catalog_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return summary with legacy ledger field names mapped to current API keys."""
    out = dict(summary)
    alias_map = {
        'data_point_count': 'total_rows',
        'csv_size_bytes': 'total_size_bytes',
        'tsfile_size_bytes': 'total_tsfile_size_bytes',
        'compressed_size_bytes': 'total_compressed_size_bytes',
    }
    for old_key, new_key in alias_map.items():
        if new_key not in out and old_key in out:
            out[new_key] = out[old_key]
    if 'total_compressed_size_bytes' not in out and 'total_tsfile_size_bytes' in out:
        out['total_compressed_size_bytes'] = out['total_tsfile_size_bytes']

    distribution_value_fields = {
        'category_distribution': 'count',
        'category_size_distribution': 'size_bytes',
        'category_data_point_distribution': 'data_points',
        'category_tsfile_size_distribution': 'tsfile_size_bytes',
        'category_overall_compression_rate_distribution': 'compression_rate',
        'category_average_compression_rate_distribution': 'compression_rate',
        'dtype_column_distribution': 'count',
        'dtype_data_point_distribution': 'data_points',
        'dtype_csv_size_distribution': 'csv_size_bytes',
        'dtype_tsfile_size_distribution': 'tsfile_size_bytes',
        'dtype_overall_compression_rate_distribution': 'compression_rate',
        'dtype_average_compression_rate_distribution': 'compression_rate',
        'float_precision_column_distribution': 'count',
        'float_precision_data_point_distribution': 'data_points',
        'float_precision_csv_size_distribution': 'csv_size_bytes',
        'float_precision_tsfile_size_distribution': 'tsfile_size_bytes',
        'float_precision_overall_compression_rate_distribution': 'compression_rate',
        'float_precision_average_compression_rate_distribution': 'compression_rate',
    }
    for key, value_field in distribution_value_fields.items():
        if key in out:
            out[key] = _normalized_distribution_rows(
                out[key],
                value_field=value_field,
            )

    if 'available_categories' not in out:
        categories = []
        seen = set()
        rows = out.get('category_distribution')
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get('category') or '').strip()
                if name and name not in seen:
                    seen.add(name)
                    categories.append(name)
        out['available_categories'] = categories
    return out


def _merge_structural_dtype_distributions(
    summary: dict[str, Any],
    *,
    structural_source: dict[str, Any] | None,
) -> dict[str, Any]:
    if not structural_source:
        return summary
    if int(summary.get('dataset_count', 0) or 0) != int(
        structural_source.get('dataset_count', 0) or 0
    ):
        return summary

    out = dict(summary)
    for key in CATALOG_SUMMARY_STRUCTURAL_DTYPE_KEYS:
        rows = structural_source.get(key)
        if isinstance(rows, list) and rows:
            # These four charts are catalog structure stats: split int, float, and
            # timestamp. Compression-rate dtype charts remain algorithm-report stats.
            out[key] = rows
    return out


def _load_catalog_summary_from_path(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    meta = data.get('meta')
    summary = data.get('summary')
    if not isinstance(meta, dict) or not isinstance(summary, dict):
        return None
    if int(meta.get('version', 0) or 0) != CATALOG_RESULT_VERSION:
        return None
    summary = _normalized_catalog_summary(summary)
    if validate_catalog_summary(summary):
        return None
    return summary


def load_precomputed_catalog_summary() -> dict[str, Any] | None:
    # The benchmark's current Dataset overview scope is timestamped public Compress rows.
    # Keep the ignored backend/dataset_result cache as a fallback for ad-hoc rebuilds,
    # but never substitute the broader timestamped ledger for the current 19-dataset scope.
    catalog_summary = _load_catalog_summary_from_path(catalog_summary_path())
    public_compress_summary = _load_catalog_summary_from_path(
        ledger_public_compress_summary_path(),
    )
    if public_compress_summary is not None:
        return _merge_structural_dtype_distributions(
            public_compress_summary,
            structural_source=catalog_summary,
        )
    return catalog_summary


def write_precomputed_catalog_summary(*, summary: dict[str, Any]) -> Path:
    errors = validate_catalog_summary(summary)
    if errors:
        raise ValueError(
            'Refusing to write invalid catalog summary: ' + '; '.join(errors),
        )

    root = dataset_result_dir()
    root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    envelope = {
        'meta': {
            'version': CATALOG_RESULT_VERSION,
            'include_user_uploads': False,
            'generated_at': generated_at,
            'overview_rate_keys': list(CATALOG_SUMMARY_OVERVIEW_RATE_KEYS),
        },
        'summary': summary,
    }
    path = catalog_summary_path()
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    _write_manifest(generated_at=generated_at)
    return path


def _write_manifest(*, generated_at: str) -> None:
    manifest = {
        'version': CATALOG_RESULT_VERSION,
        'generated_at': generated_at,
        'overview_rate_keys': list(CATALOG_SUMMARY_OVERVIEW_RATE_KEYS),
        'files': {
            'public_full_summary': FULL_SUMMARY_FILENAME,
            'public_items': ITEMS_FILENAME,
        },
    }
    manifest_path().write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def precomputed_catalog_summary_available() -> bool:
    return load_precomputed_catalog_summary() is not None


def validate_catalog_items_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ['payload must be a dict']
    items = payload.get('items')
    paper_usage = payload.get('paper_usage')
    if not isinstance(items, list):
        errors.append('items must be a list')
    if not isinstance(paper_usage, list):
        errors.append('paper_usage must be a list')
    return errors


def load_precomputed_catalog_items() -> dict[str, Any] | None:
    path = catalog_items_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    meta = data.get('meta')
    if not isinstance(meta, dict):
        return None
    if int(meta.get('version', 0) or 0) != CATALOG_ITEMS_VERSION:
        return None
    body = {
        'items': data.get('items'),
        'paper_usage': data.get('paper_usage'),
    }
    if validate_catalog_items_payload(body):
        return None
    return body


def write_precomputed_catalog_items(
    *,
    items: list[Any],
    paper_usage: list[Any],
) -> Path:
    body = {'items': items, 'paper_usage': paper_usage}
    errors = validate_catalog_items_payload(body)
    if errors:
        raise ValueError(
            'Refusing to write invalid catalog items: ' + '; '.join(errors),
        )

    root = dataset_result_dir()
    root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    envelope = {
        'meta': {
            'version': CATALOG_ITEMS_VERSION,
            'include_user_uploads': False,
            'generated_at': generated_at,
        },
        'items': items,
        'paper_usage': paper_usage,
    }
    path = catalog_items_path()
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    _write_manifest(generated_at=generated_at)
    return path


def precomputed_catalog_items_available() -> bool:
    return load_precomputed_catalog_items() is not None
