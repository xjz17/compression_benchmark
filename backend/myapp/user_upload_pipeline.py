"""Background benchmark pipeline for authenticated user uploads (catalog parity)."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import pandas as pd

from django.conf import settings

from .dataset_catalog_assets import (
    remove_dtype_catalog_assets,
    write_catalog_statistics_assets,
)
from .models import UploadedDataset
from .compression_report_io import append_compression_report_row, existing_algorithms_from_report
from .processor import (
    arrow_to_df,
    validate_arrow_file,
)
from .processor_patch_bench import process_csv_single_column_patch
from .typed_tsfile_export import run_java_selected_tsfile_export
from .catalog_tsfile_selection import (
    select_catalog_tsfile_encodings,
)
from .column_info_loader import (
    benchmark_dtype_logical_aliases,
    column_info_has_benchmark_dtype,
)
from .upload_storage import (
    remove_legacy_user_upload_slice_csvs,
    user_upload_abs_dir,
    user_upload_master_csv_path,
    user_upload_prebuilt_tsfile_path,
    user_upload_report_csv_path,
    user_upload_slug,
    user_upload_catalog_tsfile_info_path,
    user_upload_typed_tsfile_export_dir,
    user_upload_tsfile_artifact_dir,
    user_upload_tsfile_report_path,
)

logger = logging.getLogger(__name__)


def _ensure_matplotlib_cache_dir() -> None:
    mpl_cfg = Path(settings.BASE_DIR) / '.matplotlib-cache'
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(mpl_cfg.resolve()))


def _read_raw_dataframe_for_arrow(record: UploadedDataset) -> pd.DataFrame:
    base = user_upload_abs_dir(record)
    path = os.path.join(base, record.original_filename)
    for dtype in ('float', 'int'):
        ok, _err = validate_arrow_file(path, dtype)
        if ok:
            sub = arrow_to_df(path, dtype)
            if sub is not None and not sub.empty:
                return sub
    raise ValueError('Arrow file has no usable numeric columns in target lists.')


def _ensure_master_csv(record: UploadedDataset) -> str:
    """Master CSV path: original CSV or Arrow-converted ``{slug}.csv``."""
    master = user_upload_master_csv_path(record)
    if os.path.isfile(master):
        return master
    if record.file_format != 'arrow':
        raise FileNotFoundError(f'Master CSV missing for upload {record.pk}')
    raw = _read_raw_dataframe_for_arrow(record)
    raw.to_csv(master, index=False, float_format='%.8g', lineterminator='\n')
    return master


def _dtype_has_columns(column_info_path: str, data_type: str) -> bool:
    return column_info_has_benchmark_dtype(column_info_path, data_type)


def _column_indexes_for_data_type(
    master_csv: str,
    column_info_path: str,
    data_type: str,
) -> list[int]:
    with open(column_info_path, 'r', encoding='utf-8', newline='') as handle:
        info_rows = list(csv.DictReader(handle))
    raw_df = pd.read_csv(master_csv)
    wanted = benchmark_dtype_logical_aliases(data_type)
    col_indexes: list[int] = []
    for row in info_rows:
        logical = str(row.get('logical_type', '')).strip().lower()
        if logical not in wanted:
            continue
        try:
            idx = int(row.get('column_index'))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= raw_df.shape[1]:
            continue
        col_indexes.append(idx)
    return sorted(set(col_indexes))


def _run_java_benchmark(
    master_csv: str, data_type: str, report_path: str, column_info_path: str
) -> None:
    import sys
    from pathlib import Path

    _scripts = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from java_bench_common import run_java_benchmark  # noqa: E402

    raw_df = pd.read_csv(master_csv)
    col_indexes = _column_indexes_for_data_type(master_csv, column_info_path, data_type)
    out_rows: list[dict] = []
    for idx in col_indexes:
        one_col_df = raw_df.iloc[:, [idx]]
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'_java_{data_type}_col{idx}.csv',
            delete=False,
            encoding='utf-8',
            newline='\n',
        ) as tmp_csv:
            one_col_df.to_csv(tmp_csv.name, index=False, lineterminator='\n')
            tmp_csv_path = Path(tmp_csv.name)
        tmp_report = tmp_csv_path.with_name(f'{tmp_csv_path.stem}_report.csv')
        tmp_report.unlink(missing_ok=True)
        try:
            run_java_benchmark(
                tmp_csv_path,
                data_type,
                tmp_report,
                column_info=Path(column_info_path),
                column_index=idx,
            )
            bench_df = pd.read_csv(tmp_report)
            for _, r in bench_df.iterrows():
                algo = str(r.get('algorithm', '')).strip()
                if not algo:
                    continue
                try:
                    row_col_idx = int(r.get('column_index'))
                except (TypeError, ValueError):
                    row_col_idx = idx
                out_rows.append(
                    {
                        'column_index': row_col_idx,
                        'algorithm': algo,
                        'originalSize': int(float(r.get('originalSize', 0) or 0)),
                        'compressedSize': int(float(r.get('compressedSize', 0) or 0)),
                        'compressionTime': int(float(r.get('compressionTime', 0) or 0)),
                        'decompressionTime': int(float(r.get('decompressionTime', 0) or 0)),
                    }
                )
        finally:
            tmp_csv_path.unlink(missing_ok=True)
            tmp_report.unlink(missing_ok=True)
    out_rows.sort(key=lambda row: (int(row['column_index']), str(row['algorithm'])))
    with open(report_path, 'w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                'column_index',
                'algorithm',
                'originalSize',
                'compressedSize',
                'compressionTime',
                'decompressionTime',
            ),
            lineterminator='\n',
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


def _run_cpp_benchmark(
    master_csv: str, data_type: str, report_path: str, column_info_path: str
) -> None:
    raw = pd.read_csv(master_csv)
    column_indexes = _column_indexes_for_data_type(master_csv, column_info_path, data_type)
    if not column_indexes:
        with open(report_path, 'w', encoding='utf-8', newline='\n') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    'column_index',
                    'algorithm',
                    'originalSize',
                    'compressedSize',
                    'compressionTime',
                    'decompressionTime',
                ),
                lineterminator='\n',
            )
            writer.writeheader()
        return
    from .views import _cpp_bench_subprocess_env, _cpp_benchmark_executable, _repo_root

    bench_exe = _cpp_benchmark_executable()
    if not bench_exe:
        raise FileNotFoundError('benchmark_cpp executable not found')
    out_rows: list[dict] = []
    for idx in column_indexes:
        one_col = raw.iloc[:, [idx]]
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'_cpp_{data_type}_col{idx}.csv',
            delete=False,
            encoding='utf-8',
            newline='\n',
        ) as tmp_csv:
            one_col.to_csv(tmp_csv.name, index=False, lineterminator='\n')
            tmp_csv_path = Path(tmp_csv.name)
        tmp_report = tmp_csv_path.with_name(f'{tmp_csv_path.stem}_report.csv')
        tmp_report.unlink(missing_ok=True)
        cpp_cmd = [
            bench_exe,
            '-i',
            str(tmp_csv_path),
            '--data-type',
            data_type,
            '--output-report',
            str(tmp_report),
            '--column-info',
            column_info_path,
            '--column-index',
            str(idx),
        ]
        try:
            subprocess.run(
                cpp_cmd,
                check=True,
                timeout=None,
                env=_cpp_bench_subprocess_env(),
                cwd=_repo_root(),
            )
            cur_df = pd.read_csv(tmp_report)
            for _, r in cur_df.iterrows():
                algo = str(r.get('algorithm', '')).strip()
                if not algo:
                    continue
                try:
                    row_col_idx = int(r.get('column_index'))
                except (TypeError, ValueError):
                    row_col_idx = idx
                out_rows.append(
                    {
                        'column_index': row_col_idx,
                        'algorithm': algo,
                        'originalSize': int(float(r.get('originalSize', 0) or 0)),
                        'compressedSize': int(float(r.get('compressedSize', 0) or 0)),
                        'compressionTime': int(float(r.get('compressionTime', 0) or 0)),
                        'decompressionTime': int(float(r.get('decompressionTime', 0) or 0)),
                    }
                )
        finally:
            tmp_csv_path.unlink(missing_ok=True)
            tmp_report.unlink(missing_ok=True)

    out_rows.sort(key=lambda row: (int(row['column_index']), str(row['algorithm'])))
    with open(report_path, 'w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                'column_index',
                'algorithm',
                'originalSize',
                'compressedSize',
                'compressionTime',
                'decompressionTime',
            ),
            lineterminator='\n',
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


def _float_columns_from_csv(csv_path: str, column_info_path: str) -> list[list[float]]:
    from .column_info_loader import bench_headers_for_data_type

    header_meta = bench_headers_for_data_type(column_info_path, 'float')
    headers = [name for name, _mp in header_meta]
    if not headers:
        return []
    df = pd.read_csv(csv_path, usecols=headers)
    cols: list[list[float]] = []
    for name in headers:
        series = pd.to_numeric(df[name], errors='coerce').dropna()
        if series.empty:
            continue
        cols.append(series.astype(float).tolist())
    return cols


def _ensure_buff_kernel_native_python_row(
    master_csv: str,
    column_info_path: str,
    data_type: str,
    report_path: str,
) -> None:
    if data_type != 'float':
        return
    if 'buff_kernel_native' in existing_algorithms_from_report(report_path):
        return
    os.environ['COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE'] = '1'
    from .native_research_cli_python import bench_buff_float_kernel_native_columns

    columns = _float_columns_from_csv(master_csv, column_info_path)
    if not columns:
        logger.warning('buff_kernel_native skipped: no float columns in %s', master_csv)
        return
    logger.info('Appending missing buff_kernel_native row to %s', report_path)
    o, c, te, td = bench_buff_float_kernel_native_columns(columns)
    if o <= 0:
        logger.warning('buff_kernel_native produced no data for %s', master_csv)
        return
    append_compression_report_row(
        report_path,
        {
            'algorithm': 'buff_kernel_native',
            'originalSize': int(o),
            'compressedSize': int(c),
            'compressionTime': int(te),
            'decompressionTime': int(td),
        },
    )


def _run_python_benchmark(
    master_csv: str,
    data_type: str,
    report_path: str,
    column_info_path: str,
) -> None:
    from .views import _use_benchmark_env_defaults

    with _use_benchmark_env_defaults():
        prior_supplement = os.environ.get('COMPRESSION_BENCHMARK_PYTHON_JAVA_SUPPLEMENT')
        os.environ['COMPRESSION_BENCHMARK_PYTHON_JAVA_SUPPLEMENT'] = '0'
        os.environ['COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE'] = '1'
        try:
            raw = pd.read_csv(master_csv)
            col_indexes = _column_indexes_for_data_type(
                master_csv, column_info_path, data_type
            )

            rows: list[dict] = []
            for idx in col_indexes:
                one_col_df = raw.iloc[:, [idx]]
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix=f'_python_{data_type}_col{idx}.csv',
                    delete=False,
                    encoding='utf-8',
                    newline='\n',
                ) as tmp_csv:
                    one_col_df.to_csv(tmp_csv.name, index=False, lineterminator='\n')
                    tmp_csv_path = Path(tmp_csv.name)
                tmp_report = tmp_csv_path.with_name(f'{tmp_csv_path.stem}_report.csv')
                tmp_report.unlink(missing_ok=True)
                try:
                    process_csv_single_column_patch(
                        str(tmp_csv_path),
                        data_type,
                        str(tmp_report),
                        column_info_path,
                        int(idx),
                    )
                    with open(tmp_report, 'r', encoding='utf-8', newline='') as handle:
                        for rec in csv.DictReader(handle):
                            algo = str(rec.get('algorithm', '')).strip()
                            if not algo:
                                continue
                            try:
                                row_col_idx = int(rec.get('column_index'))
                            except (TypeError, ValueError):
                                row_col_idx = int(idx)
                            rows.append(
                                {
                                    'column_index': row_col_idx,
                                    'algorithm': algo,
                                    'originalSize': int(float(rec.get('originalSize', 0) or 0)),
                                    'compressedSize': int(float(rec.get('compressedSize', 0) or 0)),
                                    'compressionTime': int(float(rec.get('compressionTime', 0) or 0)),
                                    'decompressionTime': int(float(rec.get('decompressionTime', 0) or 0)),
                                }
                            )
                finally:
                    tmp_csv_path.unlink(missing_ok=True)
                    tmp_report.unlink(missing_ok=True)

            rows.sort(key=lambda row: (int(row['column_index']), str(row['algorithm'])))
            with open(report_path, 'w', encoding='utf-8', newline='\n') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        'column_index',
                        'algorithm',
                        'originalSize',
                        'compressedSize',
                        'compressionTime',
                        'decompressionTime',
                    ),
                    lineterminator='\n',
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            _ensure_buff_kernel_native_python_row(
                master_csv,
                column_info_path,
                data_type,
                report_path,
            )
        finally:
            if prior_supplement is None:
                os.environ.pop('COMPRESSION_BENCHMARK_PYTHON_JAVA_SUPPLEMENT', None)
            else:
                os.environ['COMPRESSION_BENCHMARK_PYTHON_JAVA_SUPPLEMENT'] = prior_supplement


def _run_tsfile_benchmark(
    record: UploadedDataset,
    data_type: str,
    master_csv: str,
    column_info_path: str,
) -> None:
    """Per-dtype TsFile bench (``regenerate_tsfile_staticdata.py`` parity)."""
    artifact_dir = user_upload_tsfile_artifact_dir(record, data_type)
    if os.path.isdir(artifact_dir):
        shutil.rmtree(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)
    report_path = user_upload_tsfile_report_path(record, data_type)

    import sys
    from pathlib import Path

    _scripts = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from java_bench_common import (  # noqa: E402
        JAR,
        JAVA_BENCH_CWD,
        _buff_env,
        _serf_env,
        java_jvm_prefix,
        resolve_benchmark_io_path,
    )

    if not JAR.is_file():
        raise FileNotFoundError(f'Missing Java benchmark JAR: {JAR}')

    java_cmd = [
        'java',
        *java_jvm_prefix(),
        '-jar',
        str(JAR),
        '--input',
        str(resolve_benchmark_io_path(master_csv)),
        '--data-type',
        data_type,
        '--column-info',
        str(resolve_benchmark_io_path(column_info_path)),
        '--tsfile-dir',
        str(resolve_benchmark_io_path(artifact_dir)),
        '--output-report',
        str(resolve_benchmark_io_path(report_path)),
    ]
    env = _buff_env(_serf_env(dict(os.environ)))
    from .views import _repo_root

    env.setdefault('COMPRESSION_BENCHMARK_REPO_ROOT', _repo_root())
    env.setdefault('COMPRESSION_BENCHMARK_CSV_PATCH_ROWS', '10000')
    env.setdefault('COMPRESSION_BENCHMARK_BOS_ENCODE_ROWS', '2048')
    logger.info(
        'User upload %s: running TsFile %s benchmark (master CSV, column_info)',
        record.pk,
        data_type,
    )
    subprocess.run(java_cmd, check=True, timeout=None, cwd=str(JAVA_BENCH_CWD), env=env)


def _run_typed_tsfile_catalog_export(
    record: UploadedDataset, master_csv: str, column_info_path: str
) -> None:
    slug = user_upload_slug(record)
    base = user_upload_abs_dir(record)
    encodings = select_catalog_tsfile_encodings(slug, bundle_dir=base)
    dest = user_upload_prebuilt_tsfile_path(record)
    info_csv = user_upload_catalog_tsfile_info_path(record)
    logger.info(
        'User upload %s: table-model TsFile export model=%s table=%s compression=%s -> %s',
        record.pk,
        encodings['tsfile_model'],
        encodings['table_name'],
        encodings['compression'],
        dest,
    )
    run_java_selected_tsfile_export(
        master_csv,
        column_info_path,
        output_tsfile_path=dest,
        info_csv_path=info_csv,
        dataset_key=slug,
        int_encoding=encodings['int_encoding'],
        float_encoding=encodings['float_encoding'],
        string_encoding=encodings['string_encoding'],
    )
    if not os.path.isfile(dest):
        raise RuntimeError(
            'Typed TsFile export produced no usable .tsfile; '
            f'expected prebuilt at {dest}'
        )
    legacy_export = user_upload_typed_tsfile_export_dir(record)
    if os.path.isdir(legacy_export):
        shutil.rmtree(legacy_export)
    logger.info(
        'User upload %s: prebuilt TsFile %s (%s bytes)',
        record.pk,
        dest,
        os.path.getsize(dest),
    )


def _run_language_benchmarks(
    record: UploadedDataset,
    data_type: str,
    master_csv: str,
    column_info_path: str,
) -> None:
    report_paths: list[str] = []
    for lang, runner in (
        ('java', _run_java_benchmark),
        ('python', _run_python_benchmark),
        ('cpp', _run_cpp_benchmark),
    ):
        report_path = user_upload_report_csv_path(record, data_type, lang)
        report_paths.append(report_path)
        logger.info(
            'User upload %s: running %s %s benchmark (patch, column_info)',
            record.pk,
            data_type,
            lang,
        )
        runner(master_csv, data_type, report_path, column_info_path)


def run_user_upload_pipeline(record_id: int) -> None:
    try:
        record = UploadedDataset.objects.get(pk=record_id)
    except UploadedDataset.DoesNotExist:
        logger.warning('User upload pipeline: record %s missing', record_id)
        return

    record.processing_status = UploadedDataset.PROCESSING_RUNNING
    record.processing_error = ''
    record.save(update_fields=['processing_status', 'processing_error'])

    try:
        from .views import _use_benchmark_env_defaults

        base = user_upload_abs_dir(record)
        if not os.path.isdir(base):
            raise FileNotFoundError(f'Upload storage missing: {base}')

        with _use_benchmark_env_defaults():
            os.environ.setdefault('COMPRESSION_BENCHMARK_CSV_PATCH_ROWS', '10000')
            os.environ.setdefault('COMPRESSION_BENCHMARK_BOS_ENCODE_ROWS', '2048')

            master_csv = _ensure_master_csv(record)
            slug = user_upload_slug(record)
            _ensure_matplotlib_cache_dir()
            col_info = write_catalog_statistics_assets(
                slug,
                master_csv,
                Path(base),
            )
            remove_dtype_catalog_assets(slug, Path(base))
            column_info_path = str(col_info)

            active_dtypes: list[str] = []
            for data_type in ('int', 'float'):
                if not _dtype_has_columns(column_info_path, data_type):
                    logger.info(
                        'User upload %s: skip %s (no columns in column_info)',
                        record.pk,
                        data_type,
                    )
                    continue
                active_dtypes.append(data_type)

            if not active_dtypes:
                raise ValueError(
                    'No integer or floating-point columns found in this dataset.'
                )

            for data_type in active_dtypes:
                _run_language_benchmarks(record, data_type, master_csv, column_info_path)

            _run_typed_tsfile_catalog_export(record, master_csv, column_info_path)

            for data_type in active_dtypes:
                _run_tsfile_benchmark(record, data_type, master_csv, column_info_path)

            remove_legacy_user_upload_slice_csvs(record)

        record.processing_status = UploadedDataset.PROCESSING_COMPLETED
        record.processing_error = ''
        record.save(update_fields=['processing_status', 'processing_error'])
        logger.info('User upload %s: processing completed', record.pk)
        from .user_algorithm_pipeline import refresh_algorithms_for_dataset

        refresh_algorithms_for_dataset(record.pk)
    except Exception:
        logger.exception('User upload %s: processing failed', record_id)
        record.processing_status = UploadedDataset.PROCESSING_FAILED
        record.processing_error = traceback.format_exc()[-4000:]
        record.save(update_fields=['processing_status', 'processing_error'])


def start_user_upload_pipeline(record_id: int) -> None:
    import threading

    thread = threading.Thread(
        target=run_user_upload_pipeline,
        args=(record_id,),
        daemon=True,
        name=f'user-upload-{record_id}',
    )
    thread.start()
