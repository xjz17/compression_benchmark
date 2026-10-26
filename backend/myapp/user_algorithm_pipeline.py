"""Background evaluation of user algorithms on catalog + user-upload datasets."""

from __future__ import annotations

import logging
import os
import tempfile
import traceback

import pandas as pd

from .models import UploadedAlgorithm, UploadedDataset
from .user_algorithm_runner import benchmark_user_algorithm_on_csv
from .user_algorithm_storage import (
    ENTRY_BY_LANGUAGE,
    user_algorithm_result_path,
    user_algorithm_results_dir,
)
from .user_algorithm_targets import iter_algorithm_benchmark_jobs

logger = logging.getLogger(__name__)


def _algorithm_exists(record_id: int) -> bool:
    return UploadedAlgorithm.objects.filter(pk=record_id).exists()


def _update_algorithm(record_id: int, **fields) -> bool:
    """Update status fields; return False if the upload was deleted."""
    if not _algorithm_exists(record_id):
        return False
    UploadedAlgorithm.objects.filter(pk=record_id).update(**fields)
    return True


def _write_result_row(record: UploadedAlgorithm, dataset_key: str, data_type: str, row: dict) -> None:
    if not _algorithm_exists(record.pk):
        return
    os.makedirs(user_algorithm_results_dir(record), exist_ok=True)
    path = user_algorithm_result_path(record, dataset_key, data_type)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df = df[df['algorithm'].astype(str) != str(row['algorithm'])]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False, lineterminator='\n')


def _run_one_job(
    record: UploadedAlgorithm, key: str, dtype: str, processed_csv: str
) -> bool:
    if not _algorithm_exists(record.pk):
        return False
    with tempfile.TemporaryDirectory(prefix='user_algo_run_') as tmp:
        row = benchmark_user_algorithm_on_csv(record, processed_csv, dtype, tmp)
    if not _algorithm_exists(record.pk):
        return False
    _write_result_row(record, key, dtype, row)
    return True


def _backfill_results_during_run(record: UploadedAlgorithm, scratch_dir: str) -> bool:
    """Benchmark datasets that finished while the full catalog run was in progress."""
    ran = False
    for key, dtype, processed_csv in iter_algorithm_benchmark_jobs(
        record.owner,
        record,
        scratch_dir=scratch_dir,
    ):
        if os.path.isfile(user_algorithm_result_path(record, key, dtype)):
            continue
        if _run_one_job(record, key, dtype, processed_csv):
            ran = True
        logger.info('User algorithm %s backfill: %s %s', record.pk, key, dtype)
    return ran


def run_user_algorithm_pipeline(record_id: int, *, dataset_id: int | None = None) -> None:
    try:
        record = UploadedAlgorithm.objects.get(pk=record_id)
    except UploadedAlgorithm.DoesNotExist:
        return

    if not record.entry_script:
        _update_algorithm(
            record_id,
            processing_status=UploadedAlgorithm.PROCESSING_FAILED,
            processing_error=(
                f'Missing entry: {ENTRY_BY_LANGUAGE.get(record.implement_language, "main")}'
            ),
        )
        return

    if not _update_algorithm(
        record_id,
        processing_status=UploadedAlgorithm.PROCESSING_RUNNING,
        processing_error='',
    ):
        return

    try:
        ran = False
        with tempfile.TemporaryDirectory(prefix='user_algo_scratch_') as scratch:
            for key, dtype, processed_csv in iter_algorithm_benchmark_jobs(
                record.owner,
                record,
                dataset_id=dataset_id,
                scratch_dir=scratch,
            ):
                if not _algorithm_exists(record_id):
                    return
                if _run_one_job(record, key, dtype, processed_csv):
                    ran = True
                    logger.info(
                        'User algorithm %s (%s target): %s %s',
                        record.pk,
                        record.target_column_type,
                        key,
                        dtype,
                    )

            if dataset_id is None and _algorithm_exists(record_id):
                ran = _backfill_results_during_run(record, scratch) or ran

        if not _algorithm_exists(record_id):
            return
        if not ran:
            raise ValueError(
                'No datasets with matching column types. '
                'Integer algorithms need int columns; float algorithms need numeric columns. '
                'Upload a dataset or wait for catalog bundles.'
            )
        _update_algorithm(
            record_id,
            processing_status=UploadedAlgorithm.PROCESSING_COMPLETED,
            processing_error='',
        )
    except Exception:
        if not _algorithm_exists(record_id):
            return
        logger.exception('User algorithm %s pipeline failed', record_id)
        _update_algorithm(
            record_id,
            processing_status=UploadedAlgorithm.PROCESSING_FAILED,
            processing_error=traceback.format_exc()[-4000:],
        )


def start_user_algorithm_pipeline(record_id: int, *, dataset_id: int | None = None) -> None:
    import threading

    threading.Thread(
        target=run_user_algorithm_pipeline,
        args=(record_id,),
        kwargs={'dataset_id': dataset_id},
        daemon=True,
        name=f'user-algorithm-{record_id}',
    ).start()


def refresh_algorithms_for_dataset(dataset_id: int) -> None:
    """Run every ready user algorithm on one newly completed user upload."""
    try:
        ds = UploadedDataset.objects.get(pk=dataset_id)
    except UploadedDataset.DoesNotExist:
        return
    qs = UploadedAlgorithm.objects.filter(
        owner=ds.owner,
        processing_status__in=(
            UploadedAlgorithm.PROCESSING_COMPLETED,
            UploadedAlgorithm.PROCESSING_FAILED,
        ),
    ).exclude(entry_script='')
    for algo in qs:
        start_user_algorithm_pipeline(algo.pk, dataset_id=dataset_id)
