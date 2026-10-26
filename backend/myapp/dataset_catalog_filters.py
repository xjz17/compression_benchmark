"""Query-parameter filters for ``DatasetCatalogView``."""

from __future__ import annotations

import os

from .column_info_loader import benchmark_dtypes_from_column_info

BENCHMARK_DTYPE_VALUES = frozenset({'int', 'float'})
CATALOG_PART_VALUES = frozenset({'full', 'summary', 'items'})


def parse_catalog_part(raw: object) -> str:
    """``summary`` = overview + pie charts; ``items`` = table rows; ``full`` = both (default)."""
    part = str(raw or 'full').strip().lower()
    if part in CATALOG_PART_VALUES:
        return part
    return 'full'


def parse_dataset_catalog_filters(request) -> tuple[str, str]:
    """Return ``(category_filter, benchmark_dtype_filter)``; empty strings mean no filter."""
    category = str(request.query_params.get('category', '') or '').strip()
    dtype = str(
        request.query_params.get('benchmark_dtype', '')
        or request.query_params.get('data_type', '')
        or '',
    ).strip().lower()
    if category.lower() in ('', 'all'):
        category = ''
    if dtype in ('', 'all'):
        dtype = ''
    elif dtype not in BENCHMARK_DTYPE_VALUES:
        dtype = ''
    return category, dtype


def dataset_has_benchmark_dtype(column_info_path: str, dtype: str) -> bool:
    if not dtype:
        return True
    if not os.path.isfile(column_info_path):
        return False
    return dtype in benchmark_dtypes_from_column_info(column_info_path)


def dataset_matches_category_filter(major_category: str, category_filter: str) -> bool:
    if not category_filter:
        return True
    return str(major_category or '').strip() == category_filter
