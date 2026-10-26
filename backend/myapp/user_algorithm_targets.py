"""Which column types to benchmark per algorithm target and where to read dataset CSVs."""

from __future__ import annotations

import os
from typing import Iterator

from django.conf import settings

from .models import UploadedAlgorithm, UploadedDataset
from .processor import csv_to_df
from .column_info_loader import benchmark_dtypes_from_column_info
from .dataset_bundle import column_info_path, dataset_dir, dataset_root, resolve_master_csv_path
from .upload_storage import (
    resolve_user_upload_column_types_csv,
    user_upload_dataset_key,
    user_upload_master_csv_path,
)


def benchmark_column_types_for_algorithm(
    algo: UploadedAlgorithm,
    dataset_logical_types: list[str],
) -> list[str]:
    """
    Integer-target codecs run on ``int`` only.
    Float-target codecs run on ``int`` and ``float`` when the dataset has those columns.
    """
    available = [t for t in dataset_logical_types if t in ('int', 'float')]
    if not available:
        return []
    if algo.target_column_type == UploadedAlgorithm.TARGET_INT:
        return ['int'] if 'int' in available else []
    dtypes: list[str] = []
    if 'int' in available:
        dtypes.append('int')
    if 'float' in available:
        dtypes.append('float')
    return dtypes


def _catalog_bundle_keys() -> list[str]:
    root = dataset_root()
    if not os.path.isdir(root):
        return []
    keys = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        master = os.path.join(path, f'{name}.csv')
        if os.path.isfile(master):
            keys.append(name)
    return keys


def _upload_benchmark_dtypes(record: UploadedDataset) -> list[str]:
    info = resolve_user_upload_column_types_csv(record)
    if info:
        return benchmark_dtypes_from_column_info(info)
    return []


def _catalog_benchmark_dtypes(dataset_key: str) -> list[str]:
    info = column_info_path(dataset_key)
    if os.path.isfile(info):
        return benchmark_dtypes_from_column_info(info)
    master = resolve_master_csv_path(dataset_key)
    if not master:
        return []
    types: list[str] = []
    for dtype in ('int', 'float'):
        try:
            df = csv_to_df(master, dtype)
        except Exception:
            continue
        if df is not None and not df.empty:
            types.append(dtype)
    return types


def write_dtype_slice_csv(master_csv: str, data_type: str, dest_path: str) -> bool:
    """
    Write a dtype-filtered CSV slice for custom algorithm runs (catalog + user uploads).

    Same rules as legacy ``csv_to_df``: int columns only, or all numeric columns for float.
    Output lives in a scratch directory and is deleted after the pipeline finishes.
    """
    if not os.path.isfile(master_csv):
        return False
    try:
        df = csv_to_df(master_csv, data_type)
    except Exception:
        return False
    if df is None or df.empty:
        return False
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(dest_path, index=False, lineterminator='\n')
    return True


def iter_algorithm_benchmark_jobs(
    owner,
    algo: UploadedAlgorithm,
    *,
    dataset_id: int | None = None,
    scratch_dir: str,
) -> Iterator[tuple[str, str, str]]:
    """
    Yield ``(dataset_key, data_type, processed_csv_path)`` for one algorithm run.

    ``processed_csv_path`` is a temporary dtype slice under ``scratch_dir`` (not persisted).
    Includes completed user uploads and built-in ``backend/dataset`` bundles.
    """
    ds_qs = UploadedDataset.objects.filter(
        owner=owner,
        processing_status=UploadedDataset.PROCESSING_COMPLETED,
    )
    if dataset_id is not None:
        ds_qs = ds_qs.filter(pk=dataset_id)

    for ds in ds_qs.order_by('-created_at'):
        key = user_upload_dataset_key(ds)
        logical = _upload_benchmark_dtypes(ds)
        master = user_upload_master_csv_path(ds)
        if not os.path.isfile(master):
            continue
        for dtype in benchmark_column_types_for_algorithm(algo, logical):
            dest = os.path.join(scratch_dir, f'upload_{ds.pk}_{dtype}.csv')
            if write_dtype_slice_csv(master, dtype, dest):
                yield key, dtype, dest

    if dataset_id is not None:
        return

    for key in _catalog_bundle_keys():
        logical = _catalog_benchmark_dtypes(key)
        for dtype in benchmark_column_types_for_algorithm(algo, logical):
            dest = os.path.join(scratch_dir, f'catalog_{key}_{dtype}.csv')
            master = resolve_master_csv_path(key)
            if not master:
                continue
            if write_dtype_slice_csv(master, dtype, dest):
                yield key, dtype, dest
