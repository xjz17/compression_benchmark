"""Unified built-in dataset layout: CSV in ``dataset_csv/``, sidecars in ``dataset/``, TsFile in ``dataset_tsfile/``."""

from __future__ import annotations

import os

from django.conf import settings

# Optional large-dataset store outside the repo (override with env on other machines).
_DEFAULT_EXTERNAL_DATASET_CSV_ROOT = r'E:/xjz/compression_benchmark/backend_dataset_csv'
_EXTERNAL_DATASET_CSV_ROOT_ENV = 'COMPRESSION_BENCHMARK_EXTERNAL_DATASET_CSV_ROOT'
_DEFAULT_EXTERNAL_DATASET_TSFILE_ROOT = r'E:/xjz/compression_benchmark/backend_dataset_tsfile'
_EXTERNAL_DATASET_TSFILE_ROOT_ENV = 'COMPRESSION_BENCHMARK_EXTERNAL_DATASET_TSFILE_ROOT'


def dataset_root() -> str:
    return os.path.join(settings.BASE_DIR, 'dataset')


def builtin_dataset_csv_root() -> str:
    """In-repo master CSV directory: ``backend/dataset_csv/``."""
    return os.path.join(settings.BASE_DIR, 'dataset_csv')


def external_dataset_csv_root() -> str | None:
    """
    External master CSV directory when configured and present on disk.

    Default: ``E:/xjz/compression_benchmark/backend_dataset_csv``.
    Override: ``COMPRESSION_BENCHMARK_EXTERNAL_DATASET_CSV_ROOT``.
    """
    raw = os.environ.get(_EXTERNAL_DATASET_CSV_ROOT_ENV, '').strip()
    path = os.path.expanduser(raw) if raw else _DEFAULT_EXTERNAL_DATASET_CSV_ROOT
    return path if os.path.isdir(path) else None


def dataset_csv_root() -> str:
    """Preferred master CSV root (external when available, else in-repo)."""
    ext = external_dataset_csv_root()
    return ext if ext else builtin_dataset_csv_root()


def resolve_master_csv_path(dataset_key: str) -> str | None:
    """
    Resolve ``{key}.csv``: external store first, then ``backend/dataset_csv/``.

    Returns ``None`` when neither location has the file.
    """
    fname = f'{dataset_key}.csv'
    ext_root = external_dataset_csv_root()
    if ext_root:
        ext_path = os.path.join(ext_root, fname)
        if os.path.isfile(ext_path):
            return ext_path
    builtin_path = os.path.join(builtin_dataset_csv_root(), fname)
    if os.path.isfile(builtin_path):
        return builtin_path
    return None


def master_csv_path(dataset_key: str) -> str:
    """Best-effort path to ``{key}.csv`` (existing file, else primary lookup location)."""
    resolved = resolve_master_csv_path(dataset_key)
    if resolved:
        return resolved
    return os.path.join(dataset_csv_root(), f'{dataset_key}.csv')


def builtin_dataset_tsfile_root() -> str:
    """In-repo TsFile bundle directory: ``backend/dataset_tsfile/``."""
    return os.path.join(settings.BASE_DIR, 'dataset_tsfile')


def external_dataset_tsfile_root() -> str | None:
    """
    External TsFile bundle directory when configured and present on disk.

    Default: ``E:/xjz/compression_benchmark/backend_dataset_tsfile``.
    Override: ``COMPRESSION_BENCHMARK_EXTERNAL_DATASET_TSFILE_ROOT``.
    """
    raw = os.environ.get(_EXTERNAL_DATASET_TSFILE_ROOT_ENV, '').strip()
    path = os.path.expanduser(raw) if raw else _DEFAULT_EXTERNAL_DATASET_TSFILE_ROOT
    return path if os.path.isdir(path) else None


def dataset_tsfile_root() -> str:
    """Preferred TsFile bundle root (external when available, else in-repo)."""
    ext = external_dataset_tsfile_root()
    return ext if ext else builtin_dataset_tsfile_root()


def _iter_dataset_tsfile_roots() -> list[str]:
    roots: list[str] = []
    ext = external_dataset_tsfile_root()
    if ext:
        roots.append(ext)
    builtin = builtin_dataset_tsfile_root()
    if builtin not in roots:
        roots.append(builtin)
    return roots


def _tsfile_bundle_dir_under(root: str, dataset_key: str) -> str:
    return os.path.join(root, dataset_key)


def resolve_tsfile_bundle_dir(dataset_key: str) -> str | None:
    """Return ``<root>/<key>/`` when the bundle directory exists under external or builtin."""
    for root in _iter_dataset_tsfile_roots():
        path = _tsfile_bundle_dir_under(root, dataset_key)
        if os.path.isdir(path):
            return path
    return None


def tsfile_bundle_dir(dataset_key: str) -> str:
    """Best-effort TsFile bundle dir (existing bundle, else primary lookup location)."""
    resolved = resolve_tsfile_bundle_dir(dataset_key)
    if resolved:
        return resolved
    return _tsfile_bundle_dir_under(dataset_tsfile_root(), dataset_key)


def resolve_prebuilt_tsfile_path(dataset_key: str) -> str | None:
    """Resolve ``{key}.tsfile``: external store first, then ``backend/dataset_tsfile/``."""
    fname = f'{dataset_key}.tsfile'
    for root in _iter_dataset_tsfile_roots():
        path = os.path.join(_tsfile_bundle_dir_under(root, dataset_key), fname)
        if os.path.isfile(path):
            return path
    return None


def prebuilt_tsfile_path(dataset_key: str) -> str:
    """Best-effort path to catalog prebuilt ``{key}.tsfile``."""
    resolved = resolve_prebuilt_tsfile_path(dataset_key)
    if resolved:
        return resolved
    return os.path.join(tsfile_bundle_dir(dataset_key), f'{dataset_key}.tsfile')


def dataset_dir(dataset_key: str) -> str:
    return os.path.join(dataset_root(), dataset_key)


def catalog_tsfile_info_path(dataset_key: str) -> str:
    return os.path.join(dataset_dir(dataset_key), f'{dataset_key}_catalog_tsfile_info.csv')


def typed_tsfile_encoding_summary_path(dataset_key: str) -> str:
    """Deprecated alias for ``catalog_tsfile_info_path``."""
    return catalog_tsfile_info_path(dataset_key)


def column_info_path(dataset_key: str) -> str:
    return os.path.join(dataset_dir(dataset_key), f'{dataset_key}_column_info.csv')


def dataset_info_path(dataset_key: str) -> str:
    return os.path.join(dataset_dir(dataset_key), f'{dataset_key}_dataset_info.csv')


def catalog_asset_filename(dataset_key: str, asset: str) -> str:
    """
    Catalog bundle asset names (``asset``):

    column_info, dataset_info, numeric_statistic,
    or master csv key ``csv`` -> ``{key}.csv`` (served from ``dataset_csv/``).
    """
    if asset == 'csv':
        return f'{dataset_key}.csv'
    if asset == 'column_info':
        return f'{dataset_key}_column_info.csv'
    if asset == 'dataset_info':
        return f'{dataset_key}_dataset_info.csv'
    if asset.startswith('numeric_'):
        suffix = asset.replace('numeric_', '', 1)
        ext = 'csv' if suffix == 'statistic' else 'png'
        return f'{dataset_key}_numeric_{suffix}.{ext}'
    raise ValueError(asset)


def dtype_asset_filename(dataset_key: str, data_type: str, kind: str) -> str:
    """kind: statistic | scaling | xor_leading | xor_trailing | info (legacy)"""
    if kind in ('info', 'statistic'):
        return f'{dataset_key}_{data_type}_{kind}.csv'
    return f'{dataset_key}_{data_type}_{kind}.png'


def catalog_dtype_asset_filenames(dataset_key: str) -> frozenset[str]:
    names: set[str] = set()
    for data_type in ('int', 'float'):
        for kind in ('statistic', 'scaling', 'xor_leading', 'xor_trailing'):
            names.add(dtype_asset_filename(dataset_key, data_type, kind))
    return frozenset(names)


def catalog_bundle_asset_filenames(dataset_key: str) -> frozenset[str]:
    return frozenset(
        {
            catalog_asset_filename(dataset_key, 'csv'),
            catalog_asset_filename(dataset_key, 'column_info'),
            catalog_asset_filename(dataset_key, 'dataset_info'),
            catalog_asset_filename(dataset_key, 'numeric_statistic'),
        }
    ) | catalog_dtype_asset_filenames(dataset_key)


# Legacy names (pre-{key}_ prefix bundles); API may still resolve if present.
LEGACY_BUNDLE_ASSET_FILENAMES = frozenset(
    {
        'numeric_statistic.csv',
        'column_types.csv',
        'numeric_scaling.png',
        'numeric_xor_leading.png',
        'numeric_xor_trailing.png',
    }
)

BUNDLE_ASSET_FILENAMES = LEGACY_BUNDLE_ASSET_FILENAMES


def catalog_servable_asset_filenames(dataset_key: str) -> frozenset[str]:
    return catalog_bundle_asset_filenames(dataset_key) | LEGACY_BUNDLE_ASSET_FILENAMES


def list_catalog_operator_ids(dataset_key: str) -> list[str]:
    from pathlib import Path

    from .dataset_operators import list_available_operators

    bundle = dataset_dir(dataset_key)
    if not os.path.isdir(bundle):
        return []
    return list_available_operators(Path(bundle))


def refresh_catalog_items_operators(items: list[dict]) -> list[dict]:
    """Return a copy of catalog items with asset URLs refreshed from disk."""
    refreshed: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            refreshed.append(item)
            continue
        key = str(item.get('key') or '').strip()
        asset_urls = item.get('asset_urls')
        if not key or not isinstance(asset_urls, dict):
            refreshed.append(item)
            continue
        next_item = dict(item)
        next_urls = dict(asset_urls)
        base = f'/api/datasets/catalog-assets/{key}'
        if next_urls.get('statistics_json'):
            next_urls.setdefault('statistics_json', f'{base}/statistics-json/')
            next_urls.setdefault('correlation_heatmaps_json', f'{base}/correlation-heatmaps/')
        next_urls['available_operators'] = list_catalog_operator_ids(key)
        next_item['asset_urls'] = next_urls
        refreshed.append(next_item)
    return refreshed


def catalog_has_statistics_bundle(dataset_key: str) -> bool:
    """True when ``backend/dataset/<key>/`` has catalog sidecars."""
    bundle = dataset_dir(dataset_key)
    if not os.path.isdir(bundle):
        return False
    if os.path.isfile(column_info_path(dataset_key)):
        return True
    if os.path.isfile(
        os.path.join(bundle, catalog_asset_filename(dataset_key, 'numeric_statistic'))
    ):
        return True
    for data_type in ('int', 'float'):
        if os.path.isfile(
            os.path.join(bundle, dtype_asset_filename(dataset_key, data_type, 'statistic'))
        ):
            return True
    if os.path.isfile(os.path.join(bundle, 'numeric_statistic.csv')):
        return True
    if resolve_prebuilt_tsfile_path(dataset_key):
        return True
    return False
