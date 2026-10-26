"""Run a user-uploaded codec against one processed dataset CSV."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from .models import UploadedAlgorithm
from .user_algorithm_storage import algorithm_report_key, user_algorithm_abs_dir

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _command_for_entry(
    record: UploadedAlgorithm, mode: str, inp: str, out: str, data_type: str
) -> list[str]:
    root = user_algorithm_abs_dir(record)
    entry = (record.entry_script or '').strip()
    if not entry:
        raise FileNotFoundError('No entry script configured for this algorithm upload.')
    entry_path = os.path.join(root, entry)
    lang = record.implement_language

    if entry.endswith('.py'):
        return [
            'python3',
            entry_path,
            '--mode',
            mode,
            '--input',
            inp,
            '--data-type',
            data_type,
            '--output',
            out,
        ]
    if lang == UploadedAlgorithm.LANGUAGE_JAVA and entry.endswith('.java'):
        classes = os.path.join(root, '_javac_classes')
        os.makedirs(classes, exist_ok=True)
        subprocess.run(
            ['javac', '-d', classes, entry_path],
            check=True,
            cwd=root,
            timeout=120,
        )
        class_name = Path(entry).stem
        if class_name != 'Main':
            raise ValueError('Java entry must be Main.java (public class Main).')
        return [
            'java',
            '-cp',
            classes,
            'Main',
            '--mode',
            mode,
            '--input',
            inp,
            '--data-type',
            data_type,
            '--output',
            out,
        ]
    if lang == UploadedAlgorithm.LANGUAGE_CPP and entry.endswith('.cpp'):
        exe = os.path.join(root, '_user_codec_bin')
        subprocess.run(
            ['c++', '-O2', '-std=c++17', entry_path, '-o', exe],
            check=True,
            cwd=root,
            timeout=180,
        )
        return [exe, '--mode', mode, '--input', inp, '--data-type', data_type, '--output', out]
    if os.path.isfile(entry_path) and os.access(entry_path, os.X_OK):
        return [entry_path, '--mode', mode, '--input', inp, '--data-type', data_type, '--output', out]
    raise FileNotFoundError(f'Unsupported entry for run: {entry}')


def benchmark_user_algorithm_on_csv(
    record: UploadedAlgorithm,
    processed_csv: str,
    data_type: str,
    work_dir: str,
) -> dict:
    """Return compression report row fields after compress (+ optional decompress check)."""
    os.makedirs(work_dir, exist_ok=True)
    compressed = os.path.join(work_dir, 'out.bin')
    restored = os.path.join(work_dir, 'restored.csv')

    original_size = os.path.getsize(processed_csv)
    t0 = time.perf_counter_ns()
    cmd = _command_for_entry(record, 'compress', processed_csv, compressed, data_type)
    subprocess.run(cmd, check=True, cwd=user_algorithm_abs_dir(record), timeout=3600)
    compress_ns = time.perf_counter_ns() - t0

    if not os.path.isfile(compressed):
        raise RuntimeError('Codec did not create --output file')

    compressed_size = os.path.getsize(compressed)

    t1 = time.perf_counter_ns()
    dec_cmd = _command_for_entry(record, 'decompress', compressed, restored, data_type)
    try:
        subprocess.run(
            dec_cmd, check=True, cwd=user_algorithm_abs_dir(record), timeout=3600
        )
        decompress_ns = time.perf_counter_ns() - t1
    except (subprocess.CalledProcessError, FileNotFoundError):
        decompress_ns = 0

    return {
        'algorithm': algorithm_report_key(record),
        'originalSize': int(original_size),
        'compressedSize': int(compressed_size),
        'compressionTime': int(compress_ns),
        'decompressionTime': int(decompress_ns),
    }
