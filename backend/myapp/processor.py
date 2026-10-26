import logging
import os
import struct
import csv

import numpy as np
import pandas as pd
import pyarrow as pa

from .active_method_policy import is_active_algorithm_key

from .compression import *
from .entropy_cpp import bench_ans_bytes
from .entropy_cpp import bench_huffman_bytes
from .entropy_cpp import bench_arithmetic_bytes
from .simple8b_int_python import (
    bench_simple8b_double_columns,
    bench_simple8b_float_columns,
    bench_simple8b_int64_columns,
)
from .column_encodings import (
    bench_bss_lz4_float_columns,
    bench_delta_zigzag_varint_columns,
    DELTA_BINARY_PACKED_ALGORITHM,
    is_delta_binary_packed_algorithm,
    bench_dictionary_float_columns,
    bench_dictionary_int_columns,
)
from .external_compression_python import (
    bench_bitshuffle_lz4_float_columns,
    bench_zfp_lossy_columns,
    bench_zfp_zfpy_columns,
)
from .iotdb_style_python import (
    bench_chimp_float_columns,
    bench_chimp_tsfile_int_columns,
    bench_gorilla_float_columns,
    bench_gorilla_int_columns,
    bench_rle_float_tsfile_columns,
    bench_rle_int_iotdb_key,
    bench_rlbe_float_columns,
    bench_rlbe_int_columns,
    bench_sprintz_float_columns,
    bench_sprintz_int_columns,
    bench_ts_2diff_bos_float_columns,
    bench_ts_2diff_bos_int_columns,
    bench_ts_2diff_float_columns,
    bench_ts_2diff_int_columns,
)
from .column_type_manifest import _fractional_decimal_places_from_cell

from .native_research_cli_python import native_cli_float_benchmark_pairs
from .submodule_lossless_benchmark_python import submodule_benchmark_float_pairs
from .simpiece_python import (
    bench_simpiece_double_columns,
    bench_simpiece_float_columns,
    bench_simpiece_int64_columns,
)
from .bench_record import record_bench
from .benchmark_columns import integers_as_float_columns
from .research_codec_python import (
    bench_alp_double_columns,
    bench_alp_float_columns,
    bench_alp_int64_columns,
    bench_fastpfor_float_columns,
    bench_leco_float_columns,
    scale_float_columns_to_int,
)
from .research_extended_codecs import (
    bench_corad_tricklet_double_columns,
    bench_corad_tricklet_float_columns,
    bench_corad_tricklet_int64_columns,
    bench_fpc_double_columns,
    bench_fpc_int64_columns,
    bench_pfpc_double_columns,
    bench_pfpc_int64_columns,
    bench_machete_double_rel_columns,
    bench_machete_float_rel_columns,
    bench_machete_int64_rel_columns,
    bench_most_double_minimal_columns,
    bench_most_float_minimal_columns,
    bench_most_int64_minimal_columns,
    bench_spdp_double_columns,
    bench_spdp_int64_columns,
    bench_spdp_float_columns,
    bench_streamvbyte_delta1_int_columns,
    levels_to_bench,
    row_name,
    ROW_NAME,
    PFPC_ROW_NAME,
)
from .compression_report_io import (
    append_compression_report_row,
    existing_algorithms_from_report,
)
from .java_research_supplement import merge_java_float_research_rows_into_records
from .column_type_manifest import write_csv_column_type_manifest

logger = logging.getLogger(__name__)


from .benchmark_wire import pack_float64_le, pack_int64_le
from .experiment_data_policy import parse_experiment_float_cell, prepare_numeric_column
from .grouping_native import ROW_NAME as GROUPING_ROW_NAME, bench_grouping_streams, streams_from_patch_cells


def _parse_float_cell_strict(cell) -> float | None:
    return parse_experiment_float_cell(cell)


def _parse_int_cell_strict(cell) -> int | None:
    s = str(cell).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


_RESEARCH_CODEC_BENCHMARK_EXTENSIONS = tuple(native_cli_float_benchmark_pairs()) + tuple(
    submodule_benchmark_float_pairs()
)


def _run_elf64_production_benchmarks(columns, should_run, record_fn) -> None:
    """Genuine double Elf benches; never substitute a generic byte LZ preset.

    The existing Java bridge measures encode/decode inside the JVM (not process
    launch). Its historical metric sums chunk bodies, unlike benchmark-86's
    self-contained EL64 frame. Require all input bytes to survive Java's
    adaptive round-trip checks before recording any result.
    """
    from . import elf_double_java

    selected = [name for name in ("elf_on_chimp", "elf_plus") if should_run(name)]
    if not selected:
        return
    if not elf_double_java._elf_jar() or not elf_double_java._benchmark_jar():
        logger.warning("Elf64 unsupported: missing Elf or benchmark Java JAR; no fallback")
        return
    prepared = [prepare_numeric_column(column, "float64") for column in columns]
    exact_columns = [list(column.values) for column in prepared if column.count]
    expected_bytes = sum(len(column.raw) for column in prepared)
    if not expected_bytes:
        return
    methods = {
        "elf_on_chimp": elf_double_java.bench_elf_on_chimp_double_columns,
        "elf_plus": elf_double_java.bench_elf_plus_double_columns,
    }
    for name in selected:
        result = methods[name](exact_columns)
        if int(result[0]) != expected_bytes or int(result[1]) <= 0:
            logger.warning(
                "%s unsupported for this input or incomplete lossless round trip: "
                "verified %s of %s original bytes; no result/fallback",
                name, result[0], expected_bytes,
            )
            continue
        record_bench(record_fn, name, result)


def _run_shared_codec_benchmarks(
    codec_cols: list,
    should_run,
    record_fn,
    *,
    double_native_cols: list | None = None,
    int_native_cols: list | None = None,
    column_float_max_point: list[int] | None = None,
    buff_kernel_wire: str = "float",
) -> None:
    """Research codecs; Elf only receives explicitly supplied full double columns."""
    if not codec_cols:
        return
    if int_native_cols:
        if should_run("elf_on_chimp") or should_run("elf_plus"):
            logger.warning("Elf64 unsupported for int64 input: no lossy integer-to-double cast")
    elif double_native_cols is not None:
        _run_elf64_production_benchmarks(double_native_cols, should_run, record_fn)
    for _spdp_lvl in levels_to_bench():
        _spdp_name = row_name(_spdp_lvl)
        if not should_run(_spdp_name):
            continue
        if int_native_cols:
            result = bench_spdp_int64_columns(int_native_cols, _spdp_lvl)
        elif double_native_cols:
            result = bench_spdp_double_columns(double_native_cols, _spdp_lvl)
        else:
            result = bench_spdp_float_columns(codec_cols, _spdp_lvl)
        if int(result[0]) > 0:
            record_bench(record_fn, _spdp_name, result)
    if should_run(ROW_NAME):
        if int_native_cols:
            result = bench_fpc_int64_columns(int_native_cols)
            if int(result[0]) > 0:
                record_bench(record_fn, ROW_NAME, result)
        elif double_native_cols:
            result = bench_fpc_double_columns(double_native_cols)
            if int(result[0]) > 0:
                record_bench(record_fn, ROW_NAME, result)
    if should_run(PFPC_ROW_NAME):
        if int_native_cols:
            result = bench_pfpc_int64_columns(int_native_cols)
            if int(result[0]) > 0:
                record_bench(record_fn, PFPC_ROW_NAME, result)
        elif double_native_cols:
            result = bench_pfpc_double_columns(double_native_cols)
            if int(result[0]) > 0:
                record_bench(record_fn, PFPC_ROW_NAME, result)
    if should_run("most_minimal"):
        if int_native_cols:
            result = bench_most_int64_minimal_columns(int_native_cols)
        elif double_native_cols:
            result = bench_most_double_minimal_columns(double_native_cols)
        else:
            result = bench_most_float_minimal_columns(codec_cols)
        if int(result[0]) > 0:
            record_bench(record_fn, "most_minimal", result)
    if should_run("corad_tricklet"):
        if int_native_cols:
            result = bench_corad_tricklet_int64_columns(int_native_cols)
        elif double_native_cols:
            result = bench_corad_tricklet_double_columns(double_native_cols)
        else:
            result = bench_corad_tricklet_float_columns(codec_cols)
        if int(result[0]) > 0:
            record_bench(record_fn, "corad_tricklet", result)
    for _machete_rel, _machete_name in (
        (1e-1, "machete_rel_1e-1"),
        (1e-2, "machete_rel_1e-2"),
        (1e-3, "machete_rel_1e-3"),
        (1e-4, "machete_rel_1e-4"),
    ):
        if not should_run(_machete_name):
            continue
        if int_native_cols:
            result = bench_machete_int64_rel_columns(int_native_cols, _machete_rel)
        elif double_native_cols:
            result = bench_machete_double_rel_columns(double_native_cols, _machete_rel)
        else:
            result = bench_machete_float_rel_columns(codec_cols, _machete_rel)
        if int(result[0]) > 0:
            record_bench(record_fn, _machete_name, result)
    for _simpiece_rel, _simpiece_name in (
        (0.005, "simpiece_relRange_0p005"),
        (0.01, "simpiece_relRange_0p01"),
    ):
        if not should_run(_simpiece_name):
            continue
        if int_native_cols:
            result = bench_simpiece_int64_columns(int_native_cols, _simpiece_rel)
        elif double_native_cols:
            result = bench_simpiece_double_columns(double_native_cols, _simpiece_rel)
        else:
            result = bench_simpiece_float_columns(codec_cols, _simpiece_rel)
        if int(result[0]) > 0:
            record_bench(record_fn, _simpiece_name, result)
    if should_run("simple8b"):
        if int_native_cols:
            result = bench_simple8b_int64_columns(int_native_cols)
        elif double_native_cols:
            result = bench_simple8b_double_columns(
                double_native_cols, column_float_max_point
            )
        else:
            result = bench_simple8b_float_columns(
                codec_cols, column_float_max_point
            )
        if int(result[0]) > 0:
            record_bench(record_fn, "simple8b", result)
    if should_run("alp"):
        if int_native_cols:
            result = bench_alp_int64_columns(int_native_cols)
        elif double_native_cols:
            result = bench_alp_double_columns(double_native_cols)
        else:
            result = bench_alp_float_columns(codec_cols)
        if int(result[0]) > 0:
            record_bench(record_fn, "alp", result)
    for name, bench_fn in (
        ("fastpfor", bench_fastpfor_float_columns),
        ("leco", bench_leco_float_columns),
    ) + _RESEARCH_CODEC_BENCHMARK_EXTENSIONS:
        if not should_run(name):
            continue
        if name == "buff_kernel_native" and buff_kernel_wire == "int":
            from .native_research_cli_python import bench_buff_int_kernel_native_columns

            result = bench_buff_int_kernel_native_columns(int_native_cols if int_native_cols else codec_cols)
        elif name == "leco" and column_float_max_point is not None:
            result = bench_fn(codec_cols, column_float_max_point)
        else:
            result = bench_fn(codec_cols)
        if int(result[0]) > 0:
            record_bench(record_fn, name, result)


def _bench_fpzip_columns(column_float_lists: list) -> tuple:
    """fpzip on float32 columns; int benchmarks pass cast-to-float column views."""
    total_original_size = total_compressed_size = total_comp_time = total_decomp_time = 0
    total_loss_err = total_loss_ref = 0.0
    for column_list in column_float_lists:
        result = test_fpzip_compression(column_list)
        total_original_size += result[0]
        total_compressed_size += result[1]
        total_comp_time += result[2]
        total_decomp_time += result[3]
        if len(result) >= 6:
            total_loss_err += float(result[4])
            total_loss_ref += float(result[5])
    return (
        total_original_size,
        total_compressed_size,
        total_comp_time,
        total_decomp_time,
        total_loss_err,
        total_loss_ref,
    )


def process_csv(csv_path: str,
                data_type: str,
                lzma_level: int,
                zstd_level: int,
                output_csv_path: str):
    df = pd.read_csv(csv_path, dtype=str)

    # 打印 DataFrame 的各个列的数据类型
    logger.debug("DataFrame columns and their data types:")
    logger.debug(df.dtypes)
    logger.debug("DataFrame info:")
    logger.debug(df.info())

    output_records = []

    # Parse the source spelling once, before any pandas numeric inference can
    # round an int64 column containing missing values through float64.
    prepared_columns = {}
    if data_type in ('int', 'float'):
        for name in df.columns:
            try:
                prepared = prepare_numeric_column(df[name].tolist(), data_type)
            except (ValueError, TypeError, OverflowError, struct.error):
                continue
            if prepared.count:
                prepared_columns[name] = prepared
    target_columns = list(prepared_columns)

    logger.debug(target_columns)

    column_byte_data = []
    for col_name in target_columns:
        column = pd.Series(prepared_columns[col_name].values)
        if data_type == 'int':
            column_byte_data.append(pack_int64_le(column.astype(int).tolist()))
        elif data_type == 'float':
            column_byte_data.append(pack_float64_le(column.astype(float).tolist()))

    algos = [
        'brotli',
        'gz',
        'lz4-framed',
        'snappy-framed',
    ]

    for algo in algos:
        logger.debug(f'Processing {algo}...')
        total_original_size, total_compressed_size, total_comp_time, total_decomp_time = 0, 0, 0, 0

        for col_name in target_columns:
            logger.debug(f'  Processing column: {col_name}')
            column = pd.Series(prepared_columns[col_name].values)
            column_bytes = None

            if data_type == 'int':
                column_list = column.astype(int).tolist()
                column_bytes = pack_int64_le(column_list)

            elif data_type == 'float':
                column_list = column.astype(float).tolist()
                column_bytes = pack_float64_le(column_list)

            result = test_common_compression(column_bytes, algo)

            total_original_size += result[0]
            total_compressed_size += result[1]
            total_comp_time += result[2]
            total_decomp_time += result[3]

        output_records.append({
            'algorithm': algo,
            'originalSize': total_original_size,
            'compressedSize': total_compressed_size,
            'compressionTime': total_comp_time,
            'decompressionTime': total_decomp_time,
        })

    from .native_lz_family import bench_native_lz_family_bytes

    for algorithm in (
        'blosc_blosclz',
        'blosc2_blosclz',
        'fpzip',
        'fpcompress_dpratio',
        'fpcompress_dpspeed',
    ):
        if column_byte_data:
            o, c, te, td = bench_native_lz_family_bytes(column_byte_data, algorithm)[:4]
            if o > 0:
                output_records.append({
                    'algorithm': algorithm,
                    'originalSize': o,
                    'compressedSize': c,
                    'compressionTime': te,
                    'decompressionTime': td,
                })

    # Camel is intentionally excluded pending a lossless implementation.
    # Elf/Elf+ are typed float64 codecs, not native_lz_family aliases.
    if data_type == 'float':
        def record_elf(name, original, compressed, enc_ns, dec_ns, *_loss):
            output_records.append({
                'algorithm': name, 'originalSize': original,
                'compressedSize': compressed, 'compressionTime': enc_ns,
                'decompressionTime': dec_ns,
            })
        _run_elf64_production_benchmarks(
            [list(prepared_columns[name].values) for name in target_columns],
            lambda _name: True,
            record_elf,
        )

    # process XZ/LZMA2
    total_original_size, total_compressed_size, total_comp_time, total_decomp_time = 0, 0, 0, 0
    for col_name in target_columns:
        column = pd.Series(prepared_columns[col_name].values)
        column_bytes = None

        if data_type == 'int':
            column_list = column.astype(int).tolist()
            column_bytes = pack_int64_le(column_list)

        elif data_type == 'float':
            column_list = column.astype(float).tolist()
            column_bytes = pack_float64_le(column_list)

        result = test_lzma_compression(column_bytes, lzma_level)

        total_original_size += result[0]
        total_compressed_size += result[1]
        total_comp_time += result[2]
        total_decomp_time += result[3]

    output_records.append({
        'algorithm': 'xz',
        'originalSize': total_original_size,
        'compressedSize': total_compressed_size,
        'compressionTime': total_comp_time,
        'decompressionTime': total_decomp_time,
    })

    # process zstd
    total_original_size, total_compressed_size, total_comp_time, total_decomp_time = 0, 0, 0, 0
    for col_name in target_columns:
        column = pd.Series(prepared_columns[col_name].values)
        column_bytes = None

        if data_type == 'int':
            column_list = column.astype(int).tolist()
            column_bytes = pack_int64_le(column_list)

        elif data_type == 'float':
            column_list = column.astype(float).tolist()
            column_bytes = pack_float64_le(column_list)

        result = test_zstd_compression(column_bytes, zstd_level)

        total_original_size += result[0]
        total_compressed_size += result[1]
        total_comp_time += result[2]
        total_decomp_time += result[3]

    output_records.append({
        'algorithm': 'zstd',
        'originalSize': total_original_size,
        'compressedSize': total_compressed_size,
        'compressionTime': total_comp_time,
        'decompressionTime': total_decomp_time,
    })

    col_bytes_ans = []
    for col_name in target_columns:
        column = pd.Series(prepared_columns[col_name].values)
        if data_type == 'int':
            column_list = column.astype(int).tolist()
            col_bytes_ans.append(pack_int64_le(column_list))
        else:
            column_list = column.astype(float).tolist()
            col_bytes_ans.append(pack_float64_le(column_list))
    o, c, te, td = bench_ans_bytes(col_bytes_ans)
    output_records.append({
        'algorithm': 'ans',
        'originalSize': o,
        'compressedSize': c,
        'compressionTime': te,
        'decompressionTime': td,
    })
    o, c, te, td = bench_huffman_bytes(col_bytes_ans)
    output_records.append({
        'algorithm': 'huffman',
        'originalSize': o,
        'compressedSize': c,
        'compressionTime': te,
        'decompressionTime': td,
    })
    o, c, te, td = bench_arithmetic_bytes(col_bytes_ans)
    output_records.append({
        'algorithm': 'arithmetic',
        'originalSize': o,
        'compressedSize': c,
        'compressionTime': te,
        'decompressionTime': td,
    })

    output_df = pd.DataFrame(output_records)
    output_df.to_csv(output_csv_path, index=False, lineterminator='\n')


def process_csv_simple(csv_path: str,
                      data_type: str,
                      output_csv_path: str):
    df = pd.read_csv(csv_path, dtype=str)

    logger.debug("DataFrame columns and their data types:")
    logger.debug(df.dtypes)
    logger.debug("DataFrame info:")
    logger.debug(df.info())

    if os.environ.get("WEB_COMPRESSION_STATICDATA_REGEN", "").strip() == "1" and os.path.isfile(
        output_csv_path
    ):
        os.remove(output_csv_path)

    existing_algorithms = existing_algorithms_from_report(output_csv_path)

    def should_run(algo: str) -> bool:
        if not is_active_algorithm_key(algo):
            return False
        if algo in ("chimp_int", "chimp32"):
            return False
        if is_delta_binary_packed_algorithm(algo):
            return not any(
                is_delta_binary_packed_algorithm(existing) for existing in existing_algorithms
            )
        return algo not in existing_algorithms

    def append_record(algo: str, o: int, c: int, te: int, td: int) -> None:
        record = {
            'algorithm': algo,
            'originalSize': o,
            'compressedSize': c,
            'compressionTime': te,
            'decompressionTime': td,
        }
        append_compression_report_row(output_csv_path, record)
        existing_algorithms.add(algo)

    headers = df.columns.tolist()
    logger.debug(headers)

    column_byte_data = []
    column_int_data = []
    column_float_data = []
    column_float_max_point = []
    valid_headers = []
    grouping_bench_columns = []

    for col_idx, col_name in enumerate(headers):
        logger.debug(f'Preprocessing column: {col_name}')
        column = df[col_name].dropna()

        if data_type == 'int':
            column_list = []
            for cell in column:
                v = _parse_int_cell_strict(cell)
                if v is not None:
                    column_list.append(v)
            if not column_list:
                continue
            column_bytes = pack_int64_le(column_list)
        elif data_type == 'float':
            column_list = []
            max_point = 0
            for cell in column:
                v = _parse_float_cell_strict(cell)
                if v is None:
                    continue
                column_list.append(v)
                max_point = max(max_point, _fractional_decimal_places_from_cell(str(cell)))
            if not column_list:
                continue
            column_bytes = pack_float64_le(column_list)
        else:
            continue

        valid_headers.append(col_name)
        grouping_bench_columns.append((col_name, max_point if data_type == "float" else 0, col_idx))
        column_byte_data.append(column_bytes)
        if data_type == 'int':
            column_int_data.append(column_list)
        else:
            column_float_data.append(column_list)
            column_float_max_point.append(max_point)

    from .processor_patch_bench import accumulate_benchmarks_for_columns

    if should_run(GROUPING_ROW_NAME):
        patch_cells = df.astype(str).values.tolist()
        streams = streams_from_patch_cells(
            data_type,
            grouping_bench_columns,
            patch_cells,
            row_offset=0,
        )
        if streams:
            result = bench_grouping_streams(streams)
            if int(result[0]) > 0:
                append_record(GROUPING_ROW_NAME, *result)

    accumulate_benchmarks_for_columns(
        data_type,
        column_byte_data,
        column_int_data,
        column_float_data,
        column_float_max_point,
        valid_headers,
        record_fn=append_record,
        should_run=should_run,
        use_catalog_max_points=False,
    )
    if data_type == "float":
        merge_java_float_research_rows_into_records(
            existing_algorithms, csv_path, output_csv_path
        )


def validate_arrow_file(file_path, data_type):
    try:
        with pa.memory_map(file_path, 'r') as source:
            table = pa.ipc.open_stream(source).read_all()

        if 'target' not in table.column_names:
            return False, 'Arrow file must contain a "target" column.'

        target_column = table.schema.field('target')
        target_type = target_column.type

        if not pa.types.is_list(target_type):
            return False, 'Target column must be a list type'

        item_type = target_type.value_type

        if data_type == 'float':
            if not (pa.types.is_integer(item_type) or pa.types.is_floating(item_type)):
                return False, 'Target column must contain numeric data (int or float) for float data type'

        elif data_type == 'int':
            if not pa.types.is_integer(item_type):
                return False, 'Target column must contain integer data for int data type'

        return True, None

    except Exception as e:
        return False, f'Error reading Arrow file: {str(e)}'


def arrow_to_df(file_path: str, data_type: str):
    
    with pa.memory_map(file_path, 'r') as source:
        table = pa.ipc.open_stream(source).read_all()

    target_column = table['target']
    has_item_id = 'item_id' in table.column_names
    item_id_column = table['item_id'] if has_item_id else None

    # 获取最大列表长度
    max_length = 0
    for i in range(len(target_column)):
        if target_column[i].is_valid:
            current_length = len(target_column[i].values)
            max_length = max(max_length, current_length)

    logger.debug(f"Maximum list length: {max_length}")
    logger.debug(f"Has item_id column: {has_item_id}")

    processed_columns = {}
    column_counter = 1

    for i in range(len(target_column)):
        if not target_column[i].is_valid:
            continue

        target_list = target_column[i].values.to_pylist()

        # 根据data_type转换数据类型并填充
        if data_type == 'int':
            converted_list = [int(x) for x in target_list] + \
                [0] * (max_length - len(target_list))
        elif data_type == 'float':
            converted_list = [float(x) for x in target_list] + \
                [0.0] * (max_length - len(target_list))

        # 根据是否有item_id列来决定列名
        if has_item_id:
            item_id = item_id_column[i].as_py()
            column_name = str(item_id)
        else:
            column_name = f"column_{column_counter}"
            column_counter += 1

        processed_columns[column_name] = converted_list

    # 创建新的DataFrame
    processed_df = pd.DataFrame(processed_columns)

    return drop_invalid_numeric_rows(processed_df)


def drop_invalid_numeric_rows(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drop rows with any NaN so exported numeric CSVs have no empty quoted cells."""
    if df is None or df.empty:
        return df
    return df.dropna(how='any')


def csv_to_df(file_path: str, data_type: str):
    df = pd.read_csv(file_path)

    target_df = None

    if data_type == 'int':
        target_df = df.select_dtypes(include=np.integer)
    elif data_type == 'float':
        target_df = df.select_dtypes(include=np.number)

        target_df = target_df.dropna(axis=1, how='all')

    return drop_invalid_numeric_rows(target_df)
