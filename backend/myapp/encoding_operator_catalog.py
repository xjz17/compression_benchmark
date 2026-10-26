"""Load column and byte-codec operator metadata for the Methods page and Dataset catalog."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

_METADATA_FILENAME = 'encoding_operators.json'
_BYTE_CODEC_METADATA_FILENAME = 'byte_codec_operators.json'


def _methods_reference_dir() -> Path:
    return Path(settings.BASE_DIR).resolve().parent / 'benchmark' / 'reference' / 'methods'


def encoding_operators_metadata_path() -> Path:
    return _methods_reference_dir() / _METADATA_FILENAME


def byte_codec_operators_metadata_path() -> Path:
    return _methods_reference_dir() / _BYTE_CODEC_METADATA_FILENAME


def _normalize_operator_row(row: object, *, category: str) -> dict[str, str] | None:
    if not isinstance(row, dict):
        return None
    operator_id = str(row.get('id', '')).strip()
    if not operator_id:
        return None
    name = str(row.get('name', operator_id)).strip() or operator_id
    description = str(row.get('description', '')).strip()
    example = str(row.get('example', '')).strip()
    numeric_example = str(row.get('numeric_example', '')).strip()
    good_ratio_example = str(row.get('good_ratio_example', '')).strip()
    poor_ratio_example = str(row.get('poor_ratio_example', '')).strip()
    out: dict[str, str] = {
        'id': operator_id,
        'name': name,
        'description': description,
        'category': category,
    }
    if example:
        out['example'] = example
    if numeric_example:
        out['numeric_example'] = numeric_example
    if good_ratio_example:
        out['good_ratio_example'] = good_ratio_example
    if poor_ratio_example:
        out['poor_ratio_example'] = poor_ratio_example
    return out


def operator_row_for_api(row: dict[str, str]) -> dict[str, str]:
    """Serialize one operator row for Methods API responses."""
    payload = {
        'id': row['id'],
        'name': row['name'],
        'description': row['description'],
        'category': row.get('category', 'column'),
    }
    example = str(row.get('example', '')).strip()
    if example:
        payload['example'] = example
    numeric_example = str(row.get('numeric_example', '')).strip()
    if numeric_example:
        payload['numeric_example'] = numeric_example
    good_ratio_example = str(row.get('good_ratio_example', '')).strip()
    if good_ratio_example:
        payload['good_ratio_example'] = good_ratio_example
    poor_ratio_example = str(row.get('poor_ratio_example', '')).strip()
    if poor_ratio_example:
        payload['poor_ratio_example'] = poor_ratio_example
    return payload


def _load_operator_rows(path: Path, *, category: str) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        logger.warning('Operator metadata missing: %s', path)
        return ()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        logger.exception('Failed to parse operator metadata: %s', path)
        return ()
    rows = payload.get('operators')
    if not isinstance(rows, list):
        logger.warning('Invalid operator metadata (operators list): %s', path)
        return ()
    out: list[dict[str, str]] = []
    for row in rows:
        normalized = _normalize_operator_row(row, category=category)
        if normalized is not None:
            out.append(normalized)
    return tuple(out)


@lru_cache(maxsize=1)
def load_encoding_operators() -> tuple[dict[str, str], ...]:
    """Column encoding operators for Dataset statistics and Methods page."""
    return _load_operator_rows(encoding_operators_metadata_path(), category='column')


@lru_cache(maxsize=1)
def load_byte_codec_operators() -> tuple[dict[str, str], ...]:
    """General-purpose byte-codec operators for the Methods page matrix."""
    return _load_operator_rows(byte_codec_operators_metadata_path(), category='byte_codec')


@lru_cache(maxsize=1)
def load_method_page_operators() -> tuple[dict[str, str], ...]:
    """Column operators first, then byte-codec operators (Methods page column order)."""
    return load_encoding_operators() + load_byte_codec_operators()
