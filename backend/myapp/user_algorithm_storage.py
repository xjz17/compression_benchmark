"""Persist user algorithm uploads under MEDIA_ROOT/user_algorithms/."""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
import zipfile
from typing import Optional

from django.conf import settings

from .models import UploadedAlgorithm

logger = logging.getLogger(__name__)

ALGO_REPORT_PREFIX_LOSSLESS = 'user_lossless:'
ALGO_REPORT_PREFIX_LOSSY = 'user_lossy:'


def algorithm_report_key(record: UploadedAlgorithm) -> str:
    prefix = (
        ALGO_REPORT_PREFIX_LOSSY
        if record.lossy_lossless == UploadedAlgorithm.MODE_LOSSY
        else ALGO_REPORT_PREFIX_LOSSLESS
    )
    slug = re.sub(r'[^A-Za-z0-9_.+-]+', '_', record.display_name.strip())[:120]
    return f'{prefix}{slug or "codec"}'


def user_algorithm_abs_dir(record: UploadedAlgorithm) -> str:
    return os.path.join(settings.MEDIA_ROOT, record.storage_path)


def user_algorithm_results_dir(record: UploadedAlgorithm) -> str:
    return os.path.join(user_algorithm_abs_dir(record), 'results')


def user_algorithm_result_path(
    record: UploadedAlgorithm, dataset_key: str, data_type: str
) -> str:
    safe_key = dataset_key.replace('/', '_')
    return os.path.join(
        user_algorithm_results_dir(record),
        f'{safe_key}_{data_type}_user_algorithm.csv',
    )


def create_user_algorithm(
    *,
    user,
    display_name: str,
    implement_language: str,
    lossy_lossless: str,
    target_column_type: str,
    uploaded_path: str,
    original_filename: str,
) -> UploadedAlgorithm:
    storage_id = str(uuid.uuid4())[:8]
    rel_dir = os.path.join('user_algorithms', str(user.pk), storage_id)
    dest_dir = os.path.join(settings.MEDIA_ROOT, rel_dir)
    os.makedirs(dest_dir, exist_ok=True)

    lower = original_filename.lower()
    if lower.endswith('.zip'):
        with zipfile.ZipFile(uploaded_path, 'r') as zf:
            zf.extractall(dest_dir)
    else:
        shutil.copy2(uploaded_path, os.path.join(dest_dir, os.path.basename(original_filename)))

    entry = _detect_entry_script(dest_dir, implement_language)
    if not entry:
        shutil.rmtree(dest_dir, ignore_errors=True)
        expected = ENTRY_BY_LANGUAGE.get(implement_language, 'main')
        raise ValueError(f'Missing entry file: {expected}')

    return UploadedAlgorithm.objects.create(
        owner=user,
        display_name=display_name[:255],
        storage_id=storage_id,
        original_filename=original_filename[:512],
        implement_language=implement_language,
        lossy_lossless=lossy_lossless,
        target_column_type=target_column_type,
        entry_script=entry,
        storage_path=rel_dir,
        processing_status=UploadedAlgorithm.PROCESSING_PENDING,
    )


ENTRY_BY_LANGUAGE = {
    UploadedAlgorithm.LANGUAGE_PYTHON: 'main.py',
    UploadedAlgorithm.LANGUAGE_JAVA: 'Main.java',
    UploadedAlgorithm.LANGUAGE_CPP: 'main.cpp',
}


def _detect_entry_script(root: str, language: str) -> Optional[str]:
    """Fixed entry filename per language (searched at package root, then recursively)."""
    expected = ENTRY_BY_LANGUAGE.get(language)
    if not expected:
        return None
    direct = os.path.join(root, expected)
    if os.path.isfile(direct):
        return expected
    for dirpath, _, files in os.walk(root):
        if expected in files:
            return os.path.relpath(os.path.join(dirpath, expected), root)
    return None


def delete_user_algorithm(record: UploadedAlgorithm) -> None:
    """Remove DB row and all files under MEDIA_ROOT/user_algorithms/<user>/<storage_id>/."""
    abs_dir = user_algorithm_abs_dir(record)
    pk = record.pk
    record.delete()
    if os.path.isdir(abs_dir):
        shutil.rmtree(abs_dir, ignore_errors=True)
    logger.info('Deleted user algorithm %s at %s', pk, abs_dir)
