"""Persist authenticated user uploads under MEDIA_ROOT."""

from __future__ import annotations

import os
import shutil
from typing import Optional

import pandas as pd
from django.conf import settings

from .models import UploadedDataset, User

USER_UPLOAD_KEY_PREFIX = 'user-upload-'


def parse_bool_form_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ('1', 'true', 'yes', 'on')


def user_upload_slug(record: UploadedDataset) -> str:
    return record.display_name


def user_upload_column_info_path(record: UploadedDataset) -> str:
    slug = user_upload_slug(record)
    return os.path.join(user_upload_abs_dir(record), f'{slug}_column_info.csv')


def user_upload_abs_dir(record: UploadedDataset) -> str:
    return os.path.join(settings.MEDIA_ROOT, record.storage_path)


def user_upload_dataset_key(record: UploadedDataset) -> str:
    return f'{USER_UPLOAD_KEY_PREFIX}{record.pk}'


def parse_user_upload_dataset_key(dataset_key: str) -> Optional[int]:
    if not dataset_key.startswith(USER_UPLOAD_KEY_PREFIX):
        return None
    try:
        return int(dataset_key[len(USER_UPLOAD_KEY_PREFIX):])
    except ValueError:
        return None


def user_upload_bundle_data_type(record: UploadedDataset) -> Optional[str]:
    dtypes = user_upload_benchmark_dtypes(record)
    if 'float' in dtypes:
        return 'float'
    if 'int' in dtypes:
        return 'int'
    return None


def resolve_user_upload_column_types_csv(record: UploadedDataset) -> Optional[str]:
    base = user_upload_abs_dir(record)
    if not os.path.isdir(base):
        return None
    slug = user_upload_slug(record)
    for name in (
        f'{slug}_column_info.csv',
        f'{record.display_name}_column_info.csv',
        f'{slug}_column_types.csv',
        f'{record.display_name}_column_types.csv',
    ):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    for name in os.listdir(base):
        if name.endswith('_column_info.csv') or name.endswith('_column_types.csv'):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    return None


def user_upload_has_asset_bundle(record: UploadedDataset) -> bool:
    if os.path.isfile(user_upload_column_info_path(record)):
        return True
    return resolve_user_upload_statistic_csv(record) is not None


def build_user_upload_asset_urls(dataset_key: str, record: UploadedDataset) -> Optional[dict]:
    """Catalog-style asset URLs for bundled statistics/charts on the Dataset page."""
    if not user_upload_has_asset_bundle(record):
        return None
    slug = user_upload_slug(record)
    base = f'/api/datasets/catalog-assets/{dataset_key}'
    from .dataset_bundle import catalog_asset_filename

    urls = {
        'statistics_json': f'{base}/statistics-json/',
        'correlation_heatmaps_json': f'{base}/correlation-heatmaps/',
        'numeric_statistic_csv': f'{base}/{catalog_asset_filename(slug, "numeric_statistic")}',
        'column_types_csv': f'{base}/{catalog_asset_filename(slug, "column_info")}',
        'dataset_info_csv': f'{base}/{catalog_asset_filename(slug, "dataset_info")}',
    }
    return urls


def create_user_upload(
    *,
    user: User,
    storage_id: str,
    display_name: str,
    original_filename: str,
    file_format: str,
    is_public: bool,
    major_category: str,
    uploaded_path: str,
) -> UploadedDataset:
    rel_dir = os.path.join('user_datasets', str(user.pk), storage_id)
    dest_dir = os.path.join(settings.MEDIA_ROOT, rel_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, original_filename)
    if os.path.normpath(os.path.abspath(uploaded_path)) != os.path.normpath(
        os.path.abspath(dest_file)
    ):
        shutil.copy2(uploaded_path, dest_file)
    return UploadedDataset.objects.create(
        owner=user,
        display_name=display_name[:255],
        storage_id=storage_id,
        original_filename=original_filename[:512],
        file_format=file_format[:16],
        data_type='',
        implement_language='',
        is_public=is_public,
        major_category=major_category,
        storage_path=rel_dir,
        processing_status=UploadedDataset.PROCESSING_PENDING,
    )


def user_upload_master_csv_path(record: UploadedDataset) -> str:
    """Primary CSV for benchmarks and download (upload file or Arrow-derived ``{slug}.csv``)."""
    base = user_upload_abs_dir(record)
    if record.file_format == 'csv':
        return os.path.join(base, record.original_filename)
    return os.path.join(base, f'{user_upload_slug(record)}.csv')


def resolve_user_upload_master_csv(record: UploadedDataset) -> Optional[str]:
    path = user_upload_master_csv_path(record)
    return path if os.path.isfile(path) else None


def remove_legacy_user_upload_slice_csvs(record: UploadedDataset) -> None:
    """Remove deprecated ``{slug}_{format}_{int|float}.csv`` sidecars (custom algo uses scratch)."""
    base = user_upload_abs_dir(record)
    slug = user_upload_slug(record)
    for dtype in ('int', 'float'):
        path = os.path.join(base, f'{slug}_{record.file_format}_{dtype}.csv')
        if os.path.isfile(path):
            os.unlink(path)


def user_upload_report_csv_path(
    record: UploadedDataset,
    data_type: str,
    implement_language: str,
) -> str:
    slug = user_upload_slug(record)
    return os.path.join(
        user_upload_abs_dir(record),
        f'{slug}_{data_type}_{implement_language}_compression_result.csv',
    )


def user_upload_benchmark_dtypes(record: UploadedDataset) -> list[str]:
    """Benchmark dtypes (``int`` / ``float``) for reports and custom algorithms."""
    from .column_info_loader import benchmark_dtypes_from_column_info

    info = resolve_user_upload_column_types_csv(record)
    if info:
        return benchmark_dtypes_from_column_info(info)
    return []


def user_upload_logical_types(record: UploadedDataset) -> list[str]:
    """Distinct ``logical_type`` values from ``{slug}_column_info.csv`` (display/API)."""
    from .column_info_loader import distinct_logical_types_from_column_info

    info = resolve_user_upload_column_types_csv(record)
    if info:
        return distinct_logical_types_from_column_info(info)
    return []


def resolve_user_upload_report_csv(
    record: UploadedDataset,
    *,
    data_type: Optional[str] = None,
    implement_language: Optional[str] = None,
) -> Optional[str]:
    if data_type and implement_language:
        path = user_upload_report_csv_path(record, data_type, implement_language)
        return path if os.path.isfile(path) else None
    base = user_upload_abs_dir(record)
    if not os.path.isdir(base):
        return None
    if record.data_type and record.implement_language:
        return resolve_user_upload_report_csv(
            record,
            data_type=record.data_type,
            implement_language=record.implement_language,
        )
    preferred = []
    if record.implement_language and record.data_type:
        preferred.append(
            f'_{record.data_type}_{record.implement_language}_compression_result.csv'
        )
    for suffix in preferred:
        for name in sorted(os.listdir(base)):
            if name.endswith(suffix):
                path = os.path.join(base, name)
                if os.path.isfile(path):
                    return path
    for name in sorted(os.listdir(base)):
        if name.endswith('_compression_result.csv'):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    return None


def resolve_user_upload_statistic_csv(
    record: UploadedDataset,
    data_type: Optional[str] = None,
) -> Optional[str]:
    base = user_upload_abs_dir(record)
    if not os.path.isdir(base):
        return None
    slug = user_upload_slug(record)
    dtype = data_type or record.data_type
    if dtype:
        path = os.path.join(base, f'{slug}_{dtype}_statistic.csv')
        if os.path.isfile(path):
            return path
    for name in (
        f'{slug}_numeric_statistic.csv',
        f'{record.display_name}_numeric_statistic.csv',
        'numeric_statistic.csv',
    ):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    for dt in ('float', 'int'):
        path = os.path.join(base, f'{slug}_{dt}_statistic.csv')
        if os.path.isfile(path):
            return path
    for name in os.listdir(base):
        if name.endswith('_statistic.csv'):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    return None


def resolve_user_upload_original_size(
    record: UploadedDataset,
    data_type: Optional[str] = None,
) -> int:
    dtype = data_type
    column_info_csv = user_upload_column_info_path(record)
    if os.path.isfile(column_info_csv):
        try:
            info_df = pd.read_csv(column_info_csv)
            if {'logical_type', 'column_csv_size_bytes'}.issubset(info_df.columns):
                logical = info_df['logical_type'].astype(str).str.strip().str.lower()
                if dtype == 'int':
                    mask = logical.isin(['integer', 'int', 'long'])
                elif dtype == 'float':
                    mask = logical.isin(['float', 'double'])
                else:
                    mask = logical.isin(['integer', 'int', 'long', 'float', 'double'])
                return int(pd.to_numeric(info_df.loc[mask, 'column_csv_size_bytes'], errors='coerce').fillna(0).sum())
        except Exception:
            pass
    master = resolve_user_upload_master_csv(record)
    if master:
        return int(os.path.getsize(master))
    return 0


_CHART_FILE_SUFFIX = {
    'scaling_chart': '_scaling.png',
    'xor_leading_chart': '_xor_leading.png',
    'xor_trailing_chart': '_xor_trailing.png',
}


def resolve_user_upload_catalog_asset_file(
    record: UploadedDataset,
    filename: str,
    data_type: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a catalog-asset URL filename under the user's upload directory.

    Supports ``{slug}_numeric_*.png``, ``{slug}_{int|float}_*.png``, and legacy unprefixed names.
    """
    if not filename or '/' in filename or '\\' in filename:
        return None
    base = user_upload_abs_dir(record)
    if not os.path.isdir(base):
        return None
    slug = user_upload_slug(record)

    def _safe_join(name: str) -> Optional[str]:
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            return None
        real_base = os.path.realpath(base)
        real_target = os.path.realpath(path)
        if real_target.startswith(real_base + os.sep):
            return real_target
        return None

    if filename.endswith('_statistic.csv') or filename == 'numeric_statistic.csv':
        direct = _safe_join(filename)
        if direct:
            return direct
        return resolve_user_upload_statistic_csv(record, data_type or None)

    if filename.endswith('_column_info.csv') or filename == 'column_types.csv':
        direct = _safe_join(filename)
        if direct:
            return direct
        return resolve_user_upload_column_types_csv(record)

    if filename.endswith('.png'):
        direct = _safe_join(filename)
        if direct:
            return direct
        return None

    if filename == f'{record.display_name}.csv' or filename.endswith('.csv'):
        direct = _safe_join(filename)
        if direct:
            return direct
    return resolve_user_upload_master_csv(record)


def resolve_user_upload_chart_path(
    record: UploadedDataset,
    chart_key: str,
    data_type: Optional[str] = None,
) -> Optional[str]:
    suffix = _CHART_FILE_SUFFIX.get(chart_key)
    if not suffix:
        return None
    dtypes = [data_type] if data_type else list(user_upload_benchmark_dtypes(record)) or ['float', 'int']
    base = user_upload_abs_dir(record)
    if not os.path.isdir(base):
        return None
    for dt in dtypes:
        dtype_suffix = f'_{dt}{suffix}'
        for name in os.listdir(base):
            if name.endswith(dtype_suffix):
                path = os.path.join(base, name)
                if os.path.isfile(path):
                    return path
    return None


def user_upload_tsfile_report_path(record: UploadedDataset, data_type: str) -> str:
    slug = user_upload_slug(record)
    return os.path.join(
        user_upload_abs_dir(record),
        f'{slug}_{data_type}_tsfile_compression_result.csv',
    )


def user_upload_tsfile_artifact_dir(record: UploadedDataset, data_type: str) -> str:
    return os.path.join(user_upload_abs_dir(record), 'tsfile_artifacts', data_type)


def user_upload_tsfile_artifact_basename(record: UploadedDataset, data_type: str) -> str:
    """Stem of the master CSV used for TsFile benchmarks (artifact name prefix)."""
    master = resolve_user_upload_master_csv(record)
    if master:
        return os.path.splitext(os.path.basename(master))[0]
    return f'{user_upload_slug(record)}_{data_type}'


def resolve_user_upload_tsfile_artifact_path(
    record: UploadedDataset, data_type: str, algorithm: str
) -> Optional[str]:
    artifact_dir = user_upload_tsfile_artifact_dir(record, data_type)
    if not os.path.isdir(artifact_dir):
        return None
    base = user_upload_tsfile_artifact_basename(record, data_type)
    primary = os.path.join(artifact_dir, f'{base}_{algorithm}.tsfile')
    if os.path.isfile(primary):
        return primary
    needle = f'_{algorithm}'
    for name in sorted(os.listdir(artifact_dir)):
        if name.endswith('.tsfile') and needle in name:
            path = os.path.join(artifact_dir, name)
            if os.path.isfile(path):
                return path
    return None


def user_upload_has_tsfile_benchmark(record: UploadedDataset) -> bool:
    for dt in ('int', 'float'):
        if os.path.isfile(user_upload_tsfile_report_path(record, dt)):
            return True
    return False


def user_upload_prebuilt_tsfile_path(record: UploadedDataset) -> str:
    """Catalog-style single TsFile: ``<display_name>.tsfile`` at the upload bundle root."""
    slug = user_upload_slug(record)
    return os.path.join(user_upload_abs_dir(record), f'{slug}.tsfile')


def user_upload_typed_tsfile_export_dir(record: UploadedDataset) -> str:
    """Grid TsFile export dir for typed catalog export (removed after prebuilt copy)."""
    slug = user_upload_slug(record)
    return os.path.join(user_upload_abs_dir(record), f'{slug}_typed_tsfile_export')


def user_upload_catalog_tsfile_info_path(record: UploadedDataset) -> str:
    """Selected encodings + TsFile size for catalog-style export."""
    slug = user_upload_slug(record)
    return os.path.join(user_upload_abs_dir(record), f'{slug}_catalog_tsfile_info.csv')


def user_upload_typed_tsfile_encoding_summary_path(record: UploadedDataset) -> str:
    """Deprecated alias for ``user_upload_catalog_tsfile_info_path``."""
    return user_upload_catalog_tsfile_info_path(record)


def resolve_user_upload_prebuilt_tsfile_path(record: UploadedDataset) -> Optional[str]:
    path = user_upload_prebuilt_tsfile_path(record)
    return path if os.path.isfile(path) else None


def user_upload_has_prebuilt_tsfile(record: UploadedDataset) -> bool:
    return resolve_user_upload_prebuilt_tsfile_path(record) is not None


def user_upload_tsfile_total_bytes(record: UploadedDataset) -> int:
    """Size of the catalog-style prebuilt TsFile (smallest typed-export encoding), not per-dtype artifacts."""
    path = resolve_user_upload_prebuilt_tsfile_path(record)
    return int(os.path.getsize(path)) if path else 0


def resolve_user_upload_tsfile_dir(record: UploadedDataset) -> Optional[str]:
    """Legacy on-disk layout (``tsfiles/`` under media); prefer ``tsfile_artifacts/<dtype>/``."""
    d = os.path.join(user_upload_abs_dir(record), 'tsfiles')
    return d if os.path.isdir(d) else None


def list_user_upload_chart_keys(record: UploadedDataset, data_type: Optional[str] = None) -> list[str]:
    return [
        key
        for key in _CHART_FILE_SUFFIX
        if resolve_user_upload_chart_path(record, key, data_type=data_type) is not None
    ]


def user_upload_accessible(record: UploadedDataset, user: Optional[User]) -> bool:
    if record.is_public:
        return True
    if user is None or not getattr(user, 'is_authenticated', False):
        return False
    return record.owner_id == user.pk


def get_user_upload_by_dataset_key(
    dataset_key: str,
    user: Optional[User],
    *,
    owner_only: bool = False,
) -> Optional[UploadedDataset]:
    pk = parse_user_upload_dataset_key(dataset_key)
    if pk is None:
        return None
    try:
        record = UploadedDataset.objects.get(pk=pk)
    except UploadedDataset.DoesNotExist:
        return None
    if owner_only:
        if user is None or not user.is_authenticated or record.owner_id != user.pk:
            return None
        return record
    if user_upload_accessible(record, user):
        return record
    return None


def delete_user_upload(record: UploadedDataset) -> None:
    abs_dir = user_upload_abs_dir(record)
    if os.path.isdir(abs_dir):
        shutil.rmtree(abs_dir, ignore_errors=True)
    record.delete()
