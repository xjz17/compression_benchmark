"""Catalog CSV patch benchmark (stream rows, accumulate metrics)."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import pickle
from typing import Callable, List, Sequence, Tuple

from .algorithm_allow_list import (
    from_allowlist_env,
    from_incremental_skip_env,
    has_allowlist,
    should_run as allowlist_should_run,
    should_skip,
)
from .bench_record import record_bench
from .benchmark_columns import integers_as_float_columns
from .benchmark_totals import BenchmarkTotals
from .column_info_loader import load_column_info_rows, patch_bench_columns_for_data_type
from .compression_report_io import (
    append_compression_report_row,
    existing_algorithms_from_report,
)
from .native_lz_family import (
    NATIVE_LZ_FAMILY_ALGORITHMS,
    bench_native_lz_family_bytes,
    bench_native_lz_family_literal_bytes,
    native_lz_literal_fast_enabled,
)
from .entropy_cpp import bench_ans_bytes
from .entropy_cpp import bench_arithmetic_bytes
from .benchmark_wire import pack_float64_le, pack_int64_le
from .experiment_data_policy import NAN_INPUT_POLICY, parse_experiment_float_cell
from .column_encodings import (
    bench_bss_lz4_float_columns,
    bench_delta_zigzag_varint_columns,
    bench_dictionary_float_columns,
    bench_dictionary_int_columns,
    DELTA_BINARY_PACKED_ALGORITHM,
    is_delta_binary_packed_algorithm,
)
from .dbp_float64_codec import bench_delta_zigzag_varint_float_columns
from .external_compression_python import (
    bench_bitshuffle_lz4_float_columns,
    bench_zfp_lossy_columns,
)
from .entropy_cpp import bench_huffman_bytes
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
from .research_codec_python import scale_float_columns_to_int
from .research_extended_codecs import bench_streamvbyte_delta1_int_columns
from .reger_codec import bench_reger_double_columns, bench_reger_int_columns
from .subcolumn_python import (
    bench_subcolumn_double_columns,
    bench_subcolumn_int_columns,
)
from .grouping_native import ROW_NAME as GROUPING_ROW_NAME, bench_grouping_streams, streams_from_patch_cells

logger = logging.getLogger(__name__)
REGER_TIME_COLUMN = "__reger_time__"
_COLUMN_REPORT_FIELDS = (
    "column_index",
    "algorithm",
    "originalSize",
    "compressedSize",
    "compressionTime",
    "decompressionTime",
    "isLossy",
    "lossRate",
)


def _incremental_skip_for_bench() -> set[str] | None:
    if has_allowlist():
        return None
    skip = from_incremental_skip_env()
    return skip if skip else None


def _patch_bench_verbose() -> bool:
    return os.environ.get("COMPRESSION_BENCHMARK_PATCH_BENCH_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _bench_should_run(algo: str, incremental_skip: set[str] | None) -> bool:
    if algo in ("chimp_int", "chimp32"):
        return False
    return allowlist_should_run(incremental_skip, algo)


_TYPED_OR_EXCLUDED_SPECIALISTS = frozenset({"elf_on_chimp", "elf_plus", "camel"})
_BYTE_ONLY_ALGORITHMS = frozenset(
    (*NATIVE_LZ_FAMILY_ALGORITHMS, "ans", "huffman", "arithmetic")
) - _TYPED_OR_EXCLUDED_SPECIALISTS


def _byte_only_allowlist() -> bool:
    allowlist = from_allowlist_env()
    return bool(allowlist) and all(algo in _BYTE_ONLY_ALGORITHMS for algo in allowlist)


def _flush_should_run_append(
    algo: str,
    *,
    existing_algorithms: set[str],
    incremental_skip: set[str] | None,
) -> bool:
    if has_allowlist():
        return _bench_should_run(algo, incremental_skip)
    if algo in ("chimp_int", "chimp32"):
        return False
    if is_delta_binary_packed_algorithm(algo):
        return not any(
            is_delta_binary_packed_algorithm(existing) for existing in existing_algorithms
        )
    return algo not in existing_algorithms


def accumulate_grouping_for_patch(
    data_type: str,
    bench_columns: Sequence[Tuple[str, int, int]],
    patch_cells: Sequence[Sequence[str]],
    *,
    row_offset: int,
    record_fn: Callable[..., None],
    should_run: Callable[[str], bool],
) -> None:
    if not should_run(GROUPING_ROW_NAME) or data_type not in ("int", "float"):
        return
    streams = streams_from_patch_cells(data_type, bench_columns, patch_cells, row_offset=row_offset)
    if not streams:
        return
    result = bench_grouping_streams(streams)
    if int(result[0]) > 0:
        record_bench(record_fn, GROUPING_ROW_NAME, result)


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


def _parse_reger_time_cell(cell, row_idx: int) -> int:
    s = str(cell).strip()
    if not s:
        return row_idx
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return row_idx


def _normalize_string_cell(cell) -> str:
    s = str(cell).strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def accumulate_generic_byte_compression(
    column_byte_data: list,
    valid_headers: list,
    *,
    record_fn: Callable[..., None],
    should_run: Callable[[str], bool],
) -> None:
    """Native lossless byte-codec report rows shared by Python, Java, and C++."""
    if native_lz_literal_fast_enabled():
        literal_result = None
        for algorithm in NATIVE_LZ_FAMILY_ALGORITHMS:
            if algorithm in _TYPED_OR_EXCLUDED_SPECIALISTS:
                continue
            if not should_run(algorithm):
                continue
            if literal_result is None:
                literal_result = bench_native_lz_family_literal_bytes(column_byte_data)
            record_bench(record_fn, algorithm, literal_result)
        return
    for algorithm in NATIVE_LZ_FAMILY_ALGORITHMS:
        if algorithm in _TYPED_OR_EXCLUDED_SPECIALISTS:
            continue
        if not should_run(algorithm):
            continue
        record_bench(record_fn, algorithm, bench_native_lz_family_bytes(column_byte_data, algorithm))


def _run_shared_codec_benchmarks(codec_cols, should_run, record_fn, **kwargs) -> None:
    from .processor import _run_shared_codec_benchmarks as _shared

    _shared(codec_cols, should_run, record_fn, **kwargs)


DEFAULT_PATCH_ROWS = 10_000


def patch_row_limit() -> int:
    raw = os.environ.get("COMPRESSION_BENCHMARK_CSV_PATCH_ROWS", "").strip()
    if not raw:
        return DEFAULT_PATCH_ROWS
    try:
        n = int(raw)
        return n if n > 0 else DEFAULT_PATCH_ROWS
    except ValueError:
        return DEFAULT_PATCH_ROWS


def _build_patch_column_arrays(
    data_type: str,
    bench_columns: Sequence[Tuple[str, int, int]],
    patch_cells: Sequence[Sequence[str]],
) -> tuple:
    """Return typed patch arrays plus optional REGER row-time data."""
    column_byte_data: list = []
    column_int_data: list = []
    column_float_data: list = []
    column_float_max_point: list = []
    valid_headers: list = []
    row_time_data: list[int] = []

    for col_name, max_point, _idx in bench_columns:
        cells = [row[_idx] if _idx < len(row) else "" for row in patch_cells]
        if str(col_name).strip() == REGER_TIME_COLUMN:
            row_time_data = [
                _parse_reger_time_cell(cell, row_idx)
                for row_idx, cell in enumerate(cells)
            ]
            continue
        if data_type == "string":
            parts: list[bytes] = []
            for cell in cells:
                parts.append(_normalize_string_cell(cell).encode("utf-8"))
            column_bytes = b"".join(parts)
            if not column_bytes:
                continue
            valid_headers.append(col_name)
            column_byte_data.append(column_bytes)
            continue
        if data_type == "int":
            column_list = []
            for cell in cells:
                v = _parse_int_cell_strict(cell)
                if v is not None:
                    column_list.append(v)
            if not column_list:
                continue
            column_bytes = pack_int64_le(column_list)
        else:
            column_list = []
            for cell in cells:
                v = _parse_float_cell_strict(cell)
                if v is not None:
                    column_list.append(v)
            if not column_list:
                continue
            column_bytes = pack_float64_le(column_list)

        valid_headers.append(col_name)
        column_byte_data.append(column_bytes)
        if data_type == "int":
            column_int_data.append(column_list)
        else:
            column_float_data.append(column_list)
            column_float_max_point.append(max_point)

    return (
        column_byte_data,
        column_int_data,
        column_float_data,
        column_float_max_point,
        valid_headers,
        row_time_data,
    )


def _patch_byte_cache_root() -> str:
    raw = os.environ.get("COMPRESSION_BENCHMARK_PATCH_BYTE_CACHE_DIR", "").strip()
    if raw:
        return raw
    column_cache = os.environ.get("COMPRESSION_BENCHMARK_COLUMN_CSV_CACHE_DIR", "").strip()
    if not column_cache:
        return ""
    return column_cache.rstrip(os.sep) + "_patch_bytes"


def _single_column_patch_byte_cache_path(
    csv_path: str,
    data_type: str,
    column_index: int,
    max_point: int,
    limit: int,
) -> str:
    root = _patch_byte_cache_root()
    if not root:
        return ""
    stat = os.stat(csv_path)
    source_sig = {
        "source": os.path.abspath(csv_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "data_type": data_type,
        "column_index": int(column_index),
        "max_point": int(max_point),
        "patch_rows": int(limit),
        "version": 2,
        "input_policy": NAN_INPUT_POLICY,
    }
    key = hashlib.sha256(json.dumps(source_sig, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    directory = os.path.join(root, data_type)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"col{int(column_index)}_{key}.pkl")


def _load_or_build_single_column_byte_patches(
    csv_path: str,
    data_type: str,
    column_index: int,
    max_point: int,
    limit: int,
) -> tuple[str, list[bytes]]:
    cache_path = _single_column_patch_byte_cache_path(csv_path, data_type, column_index, max_point, limit)
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
        return str(cached["header"]), list(cached["patches"])

    patches: list[bytes] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            master_header = next(reader)
        except StopIteration:
            logger.error("CSV file is empty: %s", csv_path)
            return "column_0", []
        if not master_header:
            logger.error("CSV header is empty: %s", csv_path)
            return "column_0", []
        one_col_name = str(master_header[0]).strip() or "column_0"
        bench_columns = [(one_col_name, max_point, 0)]
        patch_cells: list[list[str]] = []

        def flush_patch() -> None:
            nonlocal patch_cells
            if not patch_cells:
                return
            byte_d, _int_d, _float_d, _max_pt, _headers, _row_time_d = _build_patch_column_arrays(
                data_type, bench_columns, patch_cells
            )
            if byte_d:
                patches.append(bytes(byte_d[0]))
            patch_cells = []

        for row in reader:
            patch_cells.append([row[0] if row else ""])
            if len(patch_cells) >= limit:
                flush_patch()
        flush_patch()

    if cache_path and patches:
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as handle:
            pickle.dump({"header": one_col_name, "patches": patches}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    return one_col_name, patches


def accumulate_benchmarks_for_columns(
    data_type: str,
    column_byte_data: list,
    column_int_data: list,
    column_float_data: list,
    column_float_max_point: list,
    valid_headers: list,
    *,
    record_fn: Callable[..., None],
    should_run: Callable[[str], bool],
    use_catalog_max_points: bool = False,
    row_time_data: Sequence[int] | None = None,
) -> None:
    """Run the same codec suite as ``process_csv_simple``; merge via ``record_fn``."""

    if data_type == "string":
        accumulate_generic_byte_compression(
            column_byte_data,
            valid_headers,
            record_fn=record_fn,
            should_run=should_run,
        )
        return

    accumulate_generic_byte_compression(
        column_byte_data,
        valid_headers,
        record_fn=record_fn,
        should_run=should_run,
    )

    if should_run("ans"):
        result = bench_ans_bytes(column_byte_data)
        if int(result[0]) > 0:
            record_bench(record_fn, "ans", result)
    if should_run("huffman"):
        record_bench(record_fn, "huffman", bench_huffman_bytes(column_byte_data))
    if should_run("arithmetic"):
        record_bench(record_fn, "arithmetic", bench_arithmetic_bytes(column_byte_data))

    codec_cols: list = []
    if data_type == "int" and column_int_data:
        codec_cols = integers_as_float_columns(column_int_data)
        for name, bench_fn in (
            (DELTA_BINARY_PACKED_ALGORITHM, bench_delta_zigzag_varint_columns),
            ("col_dictionary", bench_dictionary_int_columns),
        ):
            if not should_run(name):
                continue
            result = bench_fn(column_int_data)
            if int(result[0]) > 0:
                record_bench(record_fn, name, result)
        if should_run("streamvbyte_delta1"):
            result = bench_streamvbyte_delta1_int_columns(column_int_data)
            if int(result[0]) > 0:
                record_bench(record_fn, "streamvbyte_delta1", result)
        if should_run("col_bss_lz4"):
            result = bench_bss_lz4_float_columns(codec_cols)
            if int(result[0]) > 0:
                record_bench(record_fn, "col_bss_lz4", result)
        for name, bench_fn in (
            ("TS_2DIFF", bench_ts_2diff_int_columns),
            ("RLE", bench_rle_int_iotdb_key),
            ("GORILLA", bench_gorilla_int_columns),
            ("CHIMP", bench_chimp_tsfile_int_columns),
            ("SPRINTZ", bench_sprintz_int_columns),
            ("RLBE", bench_rlbe_int_columns),
        ):
            if not should_run(name):
                continue
            result = bench_fn(column_int_data)
            if int(result[0]) > 0:
                record_bench(record_fn, name, result)
        if should_run("ts_2diff_bos"):
            result = bench_ts_2diff_bos_int_columns(column_int_data)
            if int(result[0]) > 0:
                record_bench(record_fn, "ts_2diff_bos", result)
        if should_run("subcolumn"):
            result = bench_subcolumn_int_columns(column_int_data)
            if int(result[0]) > 0:
                record_bench(record_fn, "subcolumn", result)
        if should_run("reger"):
            result = bench_reger_int_columns(column_int_data, times=row_time_data)
            if int(result[0]) > 0:
                record_bench(record_fn, "reger", result)
        int_as_double = [[float(v) for v in col] for col in column_int_data]
        _run_shared_codec_benchmarks(
            codec_cols,
            should_run,
            record_fn,
            double_native_cols=int_as_double,
            int_native_cols=column_int_data,
            buff_kernel_wire="int",
        )
        for tol, name in (
            (1e-1, "ext_zfp_tol_1e-1"),
            (1e-2, "ext_zfp_tol_1e-2"),
            (1e-3, "ext_zfp_tol_1e-3"),
        ):
            if not should_run(name):
                continue
            result = bench_zfp_lossy_columns(int_as_double, tol, wire_data_type="int")
            if int(result[0]) > 0:
                record_bench(record_fn, name, result)
    elif data_type == "float" and column_float_data:
        codec_cols = column_float_data
        mp_arg = column_float_max_point if use_catalog_max_points else None
        if should_run(DELTA_BINARY_PACKED_ALGORITHM):
            result = bench_delta_zigzag_varint_float_columns(
                column_float_data, mp_arg
            )
            if int(result[0]) > 0:
                record_bench(record_fn, DELTA_BINARY_PACKED_ALGORITHM, result)
        scaled_int_cols = scale_float_columns_to_int(column_float_data, mp_arg)
        if scaled_int_cols:
            if should_run("streamvbyte_delta1"):
                result = bench_streamvbyte_delta1_int_columns(scaled_int_cols)
                if int(result[0]) > 0:
                    record_bench(record_fn, "streamvbyte_delta1", result)
            if should_run("reger"):
                result = bench_reger_int_columns(scaled_int_cols, times=row_time_data)
                if int(result[0]) > 0:
                    record_bench(record_fn, "reger", result)
        for name, bench_fn in (
            ("col_dictionary", bench_dictionary_float_columns),
            ("col_bss_lz4", bench_bss_lz4_float_columns),
        ):
            if not should_run(name):
                continue
            result = bench_fn(column_float_data)
            if int(result[0]) > 0:
                record_bench(record_fn, name, result)
        if should_run("TS_2DIFF"):
            result = bench_ts_2diff_float_columns(
                column_float_data, column_float_max_point
            )
            if int(result[0]) > 0:
                record_bench(record_fn, "TS_2DIFF", result)
        if should_run("RLE"):
            result = bench_rle_float_tsfile_columns(
                column_float_data, column_float_max_point
            )
            if int(result[0]) > 0:
                record_bench(record_fn, "RLE", result)
        for name, bench_fn in (
            ("GORILLA", bench_gorilla_float_columns),
            ("CHIMP", bench_chimp_float_columns),
            ("SPRINTZ", bench_sprintz_float_columns),
            ("RLBE", bench_rlbe_float_columns),
        ):
            if not should_run(name):
                continue
            result = bench_fn(column_float_data)
            if int(result[0]) > 0:
                record_bench(record_fn, name, result)
        if should_run("ts_2diff_bos"):
            result = bench_ts_2diff_bos_float_columns(
                column_float_data, column_float_max_point
            )
            if int(result[0]) > 0:
                record_bench(record_fn, "ts_2diff_bos", result)
        if should_run("subcolumn"):
            result = bench_subcolumn_double_columns(
                column_float_data, column_float_max_point
            )
            if int(result[0]) > 0:
                record_bench(record_fn, "subcolumn", result)
        _run_shared_codec_benchmarks(
            codec_cols,
            should_run,
            record_fn,
            double_native_cols=column_float_data,
            column_float_max_point=(
                column_float_max_point if use_catalog_max_points else None
            ),
        )
        if should_run("ext_bitshuffle_lz4"):
            result = bench_bitshuffle_lz4_float_columns(column_float_data)
            if int(result[0]) > 0:
                record_bench(record_fn, "ext_bitshuffle_lz4", result)
        for tol, name in (
            (1e-1, "ext_zfp_tol_1e-1"),
            (1e-2, "ext_zfp_tol_1e-2"),
            (1e-3, "ext_zfp_tol_1e-3"),
        ):
            if not should_run(name):
                continue
            result = bench_zfp_lossy_columns(column_float_data, tol, wire_data_type="float")
            if int(result[0]) > 0:
                record_bench(record_fn, name, result)



def process_csv_simple_patch(
    csv_path: str,
    data_type: str,
    output_csv_path: str,
    column_info_path: str,
) -> None:
    """
    Stream ``csv_path`` in patches; column set from ``column_info_path``:

    - ``int``: ``logical_type`` integer columns only
    - ``float``: integer + float columns (integer wire as float64)

    ``max_point_number`` still comes from column_info per column.
    """
    from .java_research_supplement import merge_java_float_research_rows_into_records

    if data_type not in ("int", "float"):
        raise ValueError(data_type)

    if os.environ.get("COMPRESSION_BENCHMARK_STATICDATA_REGEN", "").strip() == "1" and os.path.isfile(
        output_csv_path
    ):
        os.remove(output_csv_path)

    existing_algorithms = existing_algorithms_from_report(output_csv_path)
    skip_algorithms = _incremental_skip_for_bench()

    def bench_should_run(algo: str) -> bool:
        return _bench_should_run(algo, skip_algorithms)

    def flush_should_run(algo: str) -> bool:
        return _flush_should_run_append(
            algo,
            existing_algorithms=existing_algorithms,
            incremental_skip=skip_algorithms,
        )

    def append_record(
        algo: str,
        o: int,
        c: int,
        te: int,
        td: int,
        is_lossy: int = 0,
        loss_rate: str = "0",
    ) -> None:
        append_compression_report_row(
            output_csv_path,
            {
                "algorithm": algo,
                "originalSize": o,
                "compressedSize": c,
                "compressionTime": te,
                "decompressionTime": td,
                "isLossy": int(is_lossy),
                "lossRate": loss_rate,
            },
        )
        existing_algorithms.add(algo)

    if not os.path.isfile(column_info_path):
        logger.warning("Missing column_info for patch bench: %s", column_info_path)
        return

    totals = BenchmarkTotals()
    limit = patch_row_limit()
    patch_cells: List[List[str]] = []
    patch_index = 0
    rows_processed = 0

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            master_header = next(reader)
        except StopIteration:
            logger.error("CSV file is empty: %s", csv_path)
            return

        bench_columns = patch_bench_columns_for_data_type(
            master_header, column_info_path, data_type
        )
        if not bench_columns:
            logger.warning(
                "No %s columns in column_info for %s", data_type, column_info_path
            )
            return

        def flush_patch() -> None:
            nonlocal patch_cells, patch_index, rows_processed
            if not patch_cells:
                return
            rows_in_patch = len(patch_cells)
            patch_row_offset = rows_processed
            patch_index += 1
            rows_processed += rows_in_patch
            if _patch_bench_verbose():
                print(
                    f"ProcessorPatchBench patch {patch_index}: {rows_in_patch} rows "
                    f"(limit={limit}, total={rows_processed}, data_type={data_type}, "
                    f"column_index=-1)",
                    flush=True,
                )
            built = _build_patch_column_arrays(data_type, bench_columns, patch_cells)
            byte_d, int_d, float_d, max_pt, _headers, row_time_d = built
            if not byte_d:
                patch_cells = []
                return

            def merge_fn(
                algo: str,
                o: int,
                c: int,
                te: int,
                td: int,
                loss_err: float = 0.0,
                loss_ref: float = 0.0,
            ) -> None:
                totals.merge(algo, o, c, te, td, loss_err, loss_ref)

            accumulate_grouping_for_patch(
                data_type,
                bench_columns,
                patch_cells,
                row_offset=patch_row_offset,
                record_fn=merge_fn,
                should_run=bench_should_run,
            )

            accumulate_benchmarks_for_columns(
                data_type,
                byte_d,
                int_d,
                float_d,
                max_pt,
                _headers,
                record_fn=merge_fn,
                should_run=bench_should_run,
                use_catalog_max_points=True,
                row_time_data=row_time_d or None,
            )
            patch_cells = []

        for row in reader:
            patch_cells.append(row)
            if len(patch_cells) >= limit:
                flush_patch()
        flush_patch()

    totals.flush(append_record, flush_should_run)

    if data_type == "float":
        merge_java_float_research_rows_into_records(
            existing_algorithms, csv_path, output_csv_path
        )


def _column_meta_by_index(column_info_path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in load_column_info_rows(column_info_path):
        try:
            idx = int(row.get("column_index"))
        except (TypeError, ValueError):
            continue
        out[idx] = row
    return out


def _logical_type_matches_dtype(logical_type: str, data_type: str) -> bool:
    logical = str(logical_type or "").strip().lower()
    if data_type == "int":
        return logical in ("integer", "int", "long")
    if data_type == "float":
        return logical in ("float", "double")
    if data_type == "string":
        return logical == "string"
    return True


def process_csv_single_column_patch(
    csv_path: str,
    data_type: str,
    output_csv_path: str,
    column_info_path: str,
    column_index: int,
) -> None:
    """
    Single-column patch benchmark.

    The input CSV is expected to contain one data column only. Results are written with
    ``column_index`` in each row.
    """
    if data_type not in ("int", "float"):
        raise ValueError(data_type)
    if not os.path.isfile(column_info_path):
        logger.warning("Missing column_info for patch bench: %s", column_info_path)
        return

    by_idx = _column_meta_by_index(column_info_path)
    row_meta = by_idx.get(int(column_index))
    max_point = 0
    if row_meta is not None:
        logical = str(row_meta.get("logical_type", "")).strip().lower()
        if not _logical_type_matches_dtype(logical, data_type):
            logger.warning(
                "column_info logical_type mismatch: idx=%s logical=%s data_type=%s",
                column_index,
                logical,
                data_type,
            )
        try:
            max_point = max(0, int(row_meta.get("max_point_number", 0)))
        except (TypeError, ValueError):
            max_point = 0

    totals = BenchmarkTotals()
    limit = patch_row_limit()
    incremental_skip = _incremental_skip_for_bench()

    def bench_should_run(algo: str) -> bool:
        return _bench_should_run(algo, incremental_skip)

    if _byte_only_allowlist():
        one_col_name, byte_patches = _load_or_build_single_column_byte_patches(
            csv_path,
            data_type,
            column_index,
            max_point,
            limit,
        )
        for patch_index, payload in enumerate(byte_patches, start=1):
            if _patch_bench_verbose():
                print(
                    f"ProcessorPatchBench byte-cache patch {patch_index}: "
                    f"{len(payload)} bytes (limit={limit}, data_type={data_type}, "
                    f"column_index={column_index})",
                    flush=True,
                )

            def merge_fn(
                algo: str,
                o: int,
                c: int,
                te: int,
                td: int,
                loss_err: float = 0.0,
                loss_ref: float = 0.0,
            ) -> None:
                totals.merge(algo, o, c, te, td, loss_err, loss_ref)

            accumulate_benchmarks_for_columns(
                data_type,
                [payload],
                [],
                [],
                [],
                [one_col_name],
                record_fn=merge_fn,
                should_run=bench_should_run,
                use_catalog_max_points=True,
                row_time_data=None,
            )
    else:
        patch_cells: List[List[str]] = []
        patch_index = 0
        rows_processed = 0

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                master_header = next(reader)
            except StopIteration:
                logger.error("CSV file is empty: %s", csv_path)
                return
            if not master_header:
                logger.error("CSV header is empty: %s", csv_path)
                return
            if len(master_header) > 1:
                logger.warning(
                    "single-column patch bench got %d columns; only first column will be used",
                    len(master_header),
                )
            one_col_name = str(master_header[0]).strip() or "column_0"
            bench_columns = [(one_col_name, max_point, 0)]

            def flush_patch() -> None:
                nonlocal patch_cells, patch_index, rows_processed
                if not patch_cells:
                    return
                rows_in_patch = len(patch_cells)
                patch_index += 1
                rows_processed += rows_in_patch
                if _patch_bench_verbose():
                    print(
                        f"ProcessorPatchBench patch {patch_index}: {rows_in_patch} rows "
                        f"(limit={limit}, total={rows_processed}, data_type={data_type}, "
                        f"column_index={column_index})",
                        flush=True,
                    )
                byte_d, int_d, float_d, max_pt, headers, row_time_d = _build_patch_column_arrays(
                    data_type, bench_columns, patch_cells
                )
                if not byte_d:
                    patch_cells = []
                    return

                def merge_fn(
                    algo: str,
                    o: int,
                    c: int,
                    te: int,
                    td: int,
                    loss_err: float = 0.0,
                    loss_ref: float = 0.0,
                ) -> None:
                    totals.merge(algo, o, c, te, td, loss_err, loss_ref)

                accumulate_benchmarks_for_columns(
                    data_type,
                    byte_d,
                    int_d,
                    float_d,
                    max_pt,
                    headers,
                    record_fn=merge_fn,
                    should_run=bench_should_run,
                    use_catalog_max_points=True,
                    row_time_data=row_time_d or None,
                )
                patch_cells = []

            for row in reader:
                first_cell = row[0] if row else ""
                patch_cells.append([first_cell])
                if len(patch_cells) >= limit:
                    flush_patch()
            flush_patch()

    rows: list[dict] = []
    for algo, (o, c, te, td) in sorted(totals.snapshot().items()):
        if should_skip(incremental_skip, algo):
            continue
        rows.append(
            {
                "column_index": int(column_index),
                "algorithm": algo,
                "originalSize": int(o),
                "compressedSize": int(c),
                "compressionTime": int(te),
                "decompressionTime": int(td),
                "isLossy": totals.is_lossy(algo),
                "lossRate": totals.loss_rate_csv(algo),
            }
        )
    with open(output_csv_path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_COLUMN_REPORT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
