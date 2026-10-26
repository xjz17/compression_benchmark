"""Fixed catalog TsFile export settings (table model, TsFile ≥ 2.3.x)."""

from __future__ import annotations

import csv
import os

from .column_info_loader import load_column_info_rows
from .dataset_bundle import dataset_dir, dataset_root, resolve_prebuilt_tsfile_path

CATALOG_TSFILE_MODEL = 'table'
CATALOG_TSFILE_INT_ENCODING = 'TS_2DIFF'
CATALOG_TSFILE_FLOAT_ENCODING = 'GORILLA'
CATALOG_TSFILE_STRING_ENCODING = 'PLAIN'
CATALOG_TSFILE_TIME_ENCODING = 'TS_2DIFF'
CATALOG_TSFILE_COMPRESSION = 'LZ4'
CATALOG_TSFILE_TAG_COLUMN_COUNT = '0'

CATALOG_TSFILE_INFO_FIELDS = (
    'tsfile_model',
    'table_name',
    'tag_column_count',
    'int_encoding',
    'float_encoding',
    'string_encoding',
    'time_encoding',
    'compression',
    'file_size_bytes',
)


def catalog_tsfile_info_path(dataset_key: str) -> str:
    return os.path.join(dataset_dir(dataset_key), f'{dataset_key}_catalog_tsfile_info.csv')


def catalog_table_name(dataset_key: str) -> str:
    """Lowercase table name written into the TsFile (matches Java export)."""
    sanitized = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in str(dataset_key).strip().lower())
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized.strip('_') or 'catalog'


def fixed_catalog_tsfile_config(dataset_key: str) -> dict[str, str]:
    """Return the fixed table-model TsFile layout for catalog export."""
    return {
        'tsfile_model': CATALOG_TSFILE_MODEL,
        'table_name': catalog_table_name(dataset_key),
        'tag_column_count': CATALOG_TSFILE_TAG_COLUMN_COUNT,
        'int_encoding': CATALOG_TSFILE_INT_ENCODING,
        'float_encoding': CATALOG_TSFILE_FLOAT_ENCODING,
        'string_encoding': CATALOG_TSFILE_STRING_ENCODING,
        'time_encoding': CATALOG_TSFILE_TIME_ENCODING,
        'compression': CATALOG_TSFILE_COMPRESSION,
    }


def select_catalog_tsfile_encodings(
    dataset_key: str,
    *,
    bundle_dir: str | None = None,
) -> dict[str, str]:
    """Legacy name: returns fixed encodings (Java bench CSVs are no longer consulted)."""
    _ = bundle_dir
    cfg = fixed_catalog_tsfile_config(dataset_key)
    return {
        'int_encoding': cfg['int_encoding'],
        'float_encoding': cfg['float_encoding'],
        'string_encoding': cfg['string_encoding'],
        'time_encoding': cfg['time_encoding'],
        'compression': cfg['compression'],
        'tsfile_model': cfg['tsfile_model'],
        'table_name': cfg['table_name'],
        'tag_column_count': cfg['tag_column_count'],
    }


def read_catalog_tsfile_info_row(path: str) -> dict[str, str] | None:
    if not os.path.isfile(path):
        return None
    with open(path, newline='', encoding='utf-8') as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        return None
    return {k: (v if v is not None else '') for k, v in row.items()}


def file_size_bytes_from_catalog_tsfile_info(path: str) -> int:
    row = read_catalog_tsfile_info_row(path)
    if not row:
        return 0
    try:
        return int(str(row.get('file_size_bytes', '')).strip())
    except ValueError:
        return 0


def write_catalog_tsfile_info_csv(
    path: str,
    *,
    dataset_key: str,
    file_size_bytes: int,
    config: dict[str, str] | None = None,
) -> None:
    cfg = dict(config or fixed_catalog_tsfile_config(dataset_key))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CATALOG_TSFILE_INFO_FIELDS))
        writer.writeheader()
        writer.writerow(
            {
                'tsfile_model': cfg.get('tsfile_model', CATALOG_TSFILE_MODEL),
                'table_name': cfg.get('table_name', catalog_table_name(dataset_key)),
                'tag_column_count': cfg.get('tag_column_count', CATALOG_TSFILE_TAG_COLUMN_COUNT),
                'int_encoding': cfg.get('int_encoding', CATALOG_TSFILE_INT_ENCODING),
                'float_encoding': cfg.get('float_encoding', CATALOG_TSFILE_FLOAT_ENCODING),
                'string_encoding': cfg.get('string_encoding', CATALOG_TSFILE_STRING_ENCODING),
                'time_encoding': cfg.get('time_encoding', CATALOG_TSFILE_TIME_ENCODING),
                'compression': cfg.get('compression', CATALOG_TSFILE_COMPRESSION),
                'file_size_bytes': int(file_size_bytes),
            }
        )


def ensure_catalog_tsfile_info_for_dataset(dataset_key: str) -> bool:
    """
    When a prebuilt ``{key}.tsfile`` exists but ``{key}_catalog_tsfile_info.csv`` is
    missing or has no ``file_size_bytes``, write the info CSV with fixed settings.
    """
    tsfile_path = resolve_prebuilt_tsfile_path(dataset_key)
    if not tsfile_path:
        return False
    info_path = catalog_tsfile_info_path(dataset_key)
    if file_size_bytes_from_catalog_tsfile_info(info_path) > 0:
        return False
    try:
        file_size = int(os.path.getsize(tsfile_path))
    except OSError:
        return False
    if file_size <= 0:
        return False
    existing = read_catalog_tsfile_info_row(info_path) or {}
    cfg = fixed_catalog_tsfile_config(dataset_key)
    for key in (
        'int_encoding',
        'float_encoding',
        'string_encoding',
        'time_encoding',
        'compression',
        'tsfile_model',
        'table_name',
        'tag_column_count',
    ):
        if existing.get(key):
            cfg[key] = str(existing[key]).strip()
    write_catalog_tsfile_info_csv(info_path, dataset_key=dataset_key, file_size_bytes=file_size, config=cfg)
    return True


def backfill_missing_catalog_tsfile_info(dataset_keys: list[str] | None = None) -> list[str]:
    """Backfill info CSVs for datasets with on-disk prebuilt TsFiles. Returns updated keys."""
    keys = list(dataset_keys) if dataset_keys is not None else _discovered_dataset_keys()
    updated: list[str] = []
    for key in keys:
        if ensure_catalog_tsfile_info_for_dataset(key):
            updated.append(key)
    return updated


def _discovered_dataset_keys() -> list[str]:
    root = dataset_root()
    if not os.path.isdir(root):
        return []
    keys: list[str] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            keys.append(name)
    return keys


# Legacy helpers kept for callers that still import them; export no longer requires Java CSVs.
def columns_by_dtype(dataset_key: str, *, bundle_dir: str | None = None) -> dict[str, list[int]]:
    path = os.path.join(
        bundle_dir if bundle_dir is not None else dataset_dir(dataset_key),
        f'{dataset_key}_column_info.csv',
    )
    out: dict[str, list[int]] = {'int': [], 'float': [], 'string': []}
    if not os.path.isfile(path):
        return out
    for row in load_column_info_rows(path):
        logical = str(row.get('logical_type', '')).strip().lower()
        if logical in ('integer', 'int', 'long'):
            bucket = 'int'
        elif logical in ('float', 'double'):
            bucket = 'float'
        elif logical == 'string':
            bucket = 'string'
        else:
            continue
        try:
            out[bucket].append(int(row.get('column_index', -1)))
        except (TypeError, ValueError):
            continue
    return out


def missing_java_compression_reports_for_tsfile(
    dataset_key: str,
    *,
    bundle_dir: str | None = None,
) -> list[str]:
    _ = dataset_key, bundle_dir
    return []
