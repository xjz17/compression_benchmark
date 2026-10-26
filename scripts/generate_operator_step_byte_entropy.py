#!/usr/bin/env python3
"""
Generate per-algorithm operator-step byte entropy rows for selected datasets.

The script is intentionally additive. It never edits existing compression result
CSVs. For each dataset key it writes/merges:

    backend/dataset/<key>/<key>_operator_step_byte_entropy.csv

Existing rows are preserved. New rows with the same identity key are skipped,
and an existing output file is copied to a timestamped backup before merging.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "orignial_datasets"
DEFAULT_BACKEND_DATASET_ROOT = BACKEND_DIR / "dataset"
DEFAULT_SOURCE_SAMPLE_ROWS = int(os.environ.get("COMPRESSION_BENCHMARK_OPERATOR_STEP_SAMPLE_ROWS", "10000") or "0")
DEFAULT_ENTROPY_SAMPLE_VALUES = int(os.environ.get("COMPRESSION_BENCHMARK_OPERATOR_STEP_ENTROPY_VALUES", "10000") or "0")

NUMERIC_LOGICAL_TYPES = frozenset({"integer", "int", "long", "float", "double"})
DEFAULT_DATA_TYPES = frozenset({"int", "float"})
OPERATOR_ALIASES = {
    "bitpack_int": "bitpacking",
    "rle": "run_length",
    "ease": "eraser",
    "pongo_eraser": "eraser",
}

FIELDNAMES = [
    "dataset_key",
    "source_csv",
    "result_file",
    "data_type",
    "language",
    "algorithm",
    "method_name",
    "column_index",
    "column_name",
    "logical_type",
    "step_index",
    "step_variant",
    "operator_id",
    "transform_operator_id",
    "operator_status",
    "before_byte_entropy",
    "after_byte_entropy",
    "byte_entropy_change",
    "before_count",
    "after_count",
]

IDENTITY_COLUMNS = [
    "dataset_key",
    "result_file",
    "data_type",
    "language",
    "algorithm",
    "column_index",
    "step_index",
    "step_variant",
    "operator_id",
]


@dataclass(frozen=True)
class ColumnInfo:
    index: int
    name: str
    logical_type: str
    max_point_number: int


@dataclass(frozen=True)
class CompressionResultFile:
    path: Path
    data_type: str
    language: str


@dataclass(frozen=True)
class PipelineStreams:
    """Logical value stream plus side streams emitted by a branching operator."""

    primary: pd.Series
    auxiliary: tuple[pd.Series, ...] = ()


def _configure_backend_imports() -> None:
    backend = str(BACKEND_DIR.resolve())
    if backend not in sys.path:
        sys.path.insert(0, backend)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    try:
        import django

        django.setup()
    except Exception:
        # The imports used below only require configured settings. Let the
        # concrete import/call fail with a useful message if settings are broken.
        pass


def _read_column_info(path: Path) -> dict[int, ColumnInfo]:
    out: dict[int, ColumnInfo] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                index = int(row.get("column_index", ""))
            except (TypeError, ValueError):
                continue
            name = str(row.get("column_name", "")).strip()
            logical = str(row.get("logical_type", "")).strip().lower()
            try:
                max_point = int(row.get("max_point_number", 0) or 0)
            except (TypeError, ValueError):
                max_point = 0
            out[index] = ColumnInfo(
                index=index,
                name=name,
                logical_type=logical,
                max_point_number=max(0, max_point),
            )
    return out


def _iter_dataset_csvs(datasets_root: Path, dataset_filter: set[str]) -> list[Path]:
    csv_paths = sorted(path for path in datasets_root.glob("*.csv") if path.is_file())
    if dataset_filter:
        csv_paths = [path for path in csv_paths if path.stem in dataset_filter]
    return csv_paths


def _parse_compression_result_file(path: Path, dataset_key: str) -> CompressionResultFile | None:
    pattern = re.compile(
        rf"^{re.escape(dataset_key)}_(?P<data_type>[^_]+)_(?P<language>.+)_compression_result\.csv$"
    )
    match = pattern.match(path.name)
    if not match:
        return None
    return CompressionResultFile(
        path=path,
        data_type=match.group("data_type").strip().lower(),
        language=match.group("language").strip(),
    )


def _compression_result_files(
    bundle_dir: Path,
    dataset_key: str,
    allowed_data_types: set[str],
) -> list[CompressionResultFile]:
    files: list[CompressionResultFile] = []
    for path in sorted(bundle_dir.glob(f"{dataset_key}_*_compression_result.csv")):
        parsed = _parse_compression_result_file(path, dataset_key)
        if parsed is None:
            continue
        if allowed_data_types and parsed.data_type not in allowed_data_types:
            continue
        files.append(parsed)
    return files


def _read_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _needed_column_indices(
    result_files: list[CompressionResultFile],
    column_info: dict[int, ColumnInfo],
) -> set[int]:
    numeric_indices = {
        index
        for index, info in column_info.items()
        if info.logical_type in NUMERIC_LOGICAL_TYPES
    }
    needed: set[int] = set()
    for result in result_files:
        for row in _read_result_rows(result.path):
            try:
                index = int(row.get("column_index", ""))
            except (TypeError, ValueError):
                continue
            if index in numeric_indices:
                needed.add(index)
    return needed


def _read_source_columns(csv_path: Path, column_indices: set[int]) -> dict[int, pd.Series]:
    ordered = sorted(column_indices)
    if not ordered:
        return {}
    df = pd.read_csv(
        csv_path,
        usecols=ordered,
        low_memory=False,
        nrows=DEFAULT_SOURCE_SAMPLE_ROWS if DEFAULT_SOURCE_SAMPLE_ROWS > 0 else None,
    )
    columns_by_index: dict[int, pd.Series] = {}
    for index, actual_name in zip(ordered, df.columns):
        columns_by_index[index] = df[actual_name]
    return columns_by_index


def _flatten_streams(streams: PipelineStreams) -> pd.Series:
    parts = (streams.primary, *streams.auxiliary)
    nonempty = [part for part in parts if len(part) > 0]
    if not nonempty:
        return pd.Series([], dtype=float)
    if len(nonempty) == 1:
        return nonempty[0]
    out = pd.concat(nonempty, ignore_index=True)
    if all(getattr(part, "attrs", {}).get("byte_tokens") for part in nonempty):
        out.attrs["byte_tokens"] = True
    return out


def _byte_entropy(series: pd.Series | PipelineStreams) -> tuple[float | None, int]:
    if isinstance(series, PipelineStreams):
        series = _flatten_streams(series)
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, 0
    if DEFAULT_ENTROPY_SAMPLE_VALUES > 0 and arr.size > DEFAULT_ENTROPY_SAMPLE_VALUES:
        arr = arr[:DEFAULT_ENTROPY_SAMPLE_VALUES]
    if getattr(series, "attrs", {}).get("byte_tokens"):
        byte_values = np.clip(np.rint(arr), 0, 255).astype(np.uint8, copy=False)
    else:
        byte_values = np.frombuffer(arr.astype("<f4", copy=False).tobytes(), dtype=np.uint8)
    counts = np.bincount(byte_values, minlength=256)
    total = int(counts.sum())
    probs = counts[counts > 0].astype(float) / float(total)
    entropy = float(-np.sum(probs * np.log2(probs)))
    if abs(entropy) < 1e-12:
        return None, int(arr.size)
    return entropy, int(arr.size)


def _finite_array(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _infer_decimal_places_fast(arr: np.ndarray) -> int:
    sample = arr[np.isfinite(arr)]
    if sample.size == 0:
        return 0
    if sample.size > 10000:
        sample = sample[:10000]
    max_places = 0
    for value in sample:
        text = f"{float(value):.15f}".rstrip("0").rstrip(".")
        if "." in text:
            max_places = max(max_places, len(text.split(".", 1)[1]))
    return max_places


def _unsigned_bit_width(value: int) -> int:
    return max(1, max(0, int(value)).bit_length())


def _outlier_candidate_indices(unique_count: int, limit: int = 33) -> np.ndarray:
    if unique_count <= limit:
        return np.arange(unique_count, dtype=int)
    return np.unique(np.linspace(0, unique_count - 1, num=limit, dtype=int))


def _separate_outlier_streams(series: pd.Series) -> PipelineStreams:
    """Approximate BOS separation while retaining its branch semantics.

    BOS selects a central interval by balancing the packed widths of normal and
    exceptional values against position metadata. The catalog proxy evaluates
    the same storage-cost terms over deterministic quantile candidates. It
    keeps the normal stream separate so the following frame-of-reference step
    does not accidentally operate on a zero-valued exception bitmap.
    """

    values = _finite_array(series)
    if values.size == 0:
        return PipelineStreams(pd.Series([], dtype=float))
    info = np.iinfo(np.int64)
    lower_limit = np.nextafter(float(info.min), 0.0)
    upper_limit = np.nextafter(float(info.max), 0.0)
    ints = np.rint(np.clip(values, lower_limit, upper_limit)).astype(np.int64, copy=False)
    unique, counts = np.unique(ints, return_counts=True)
    if unique.size == 1:
        return PipelineStreams(pd.Series(ints.astype(float, copy=False), dtype=float))

    cumulative = np.cumsum(counts, dtype=np.int64)
    n = int(ints.size)
    min_value = int(unique[0])
    max_value = int(unique[-1])
    position_width = _unsigned_bit_width(n - 1)
    candidates = _outlier_candidate_indices(int(unique.size))

    best_cost = n * _unsigned_bit_width(max_value - min_value)
    best_bounds = (min_value, max_value)
    for lower_index in candidates:
        lower = int(unique[lower_index])
        left_count = int(cumulative[lower_index - 1]) if lower_index > 0 else 0
        for upper_index in candidates[candidates >= lower_index]:
            upper = int(unique[upper_index])
            normal_count = int(cumulative[upper_index]) - left_count
            if normal_count <= 0:
                continue
            right_count = n - int(cumulative[upper_index])
            outlier_count = left_count + right_count
            position_cost = min(
                outlier_count * position_width,
                n + outlier_count,
            )
            cost = position_cost
            cost += normal_count * _unsigned_bit_width(upper - lower)
            if left_count:
                cost += left_count * _unsigned_bit_width(lower - min_value)
            if right_count:
                cost += right_count * _unsigned_bit_width(max_value - upper)
            if cost < best_cost:
                best_cost = cost
                best_bounds = (lower, upper)

    lower, upper = best_bounds
    left_mask = ints < lower
    right_mask = ints > upper
    normal_mask = ~(left_mask | right_mask)
    markers = np.zeros(n, dtype=float)
    markers[left_mask] = 1.0
    markers[right_mask] = 2.0

    primary = pd.Series(ints[normal_mask].astype(float, copy=False), dtype=float)
    auxiliary: list[pd.Series] = []
    if np.any(left_mask | right_mask):
        auxiliary.append(pd.Series(markers, dtype=float))
    if np.any(left_mask):
        auxiliary.append(
            pd.Series(ints[left_mask].astype(float) - float(min_value), dtype=float)
        )
    if np.any(right_mask):
        auxiliary.append(
            pd.Series(ints[right_mask].astype(float) - float(upper), dtype=float)
        )
    return PipelineStreams(primary=primary, auxiliary=tuple(auxiliary))


def _zigzag_encode_array(arr: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(arr)
    ints = np.zeros(arr.shape[0], dtype=np.int64)
    if np.any(finite_mask):
        info = np.iinfo(np.int64)
        lower = np.nextafter(float(info.min), 0.0)
        upper = np.nextafter(float(info.max), 0.0)
        clipped = np.clip(arr[finite_mask], lower, upper)
        ints[finite_mask] = clipped.astype(np.int64, copy=False)
    encoded = ((ints << np.int64(1)) ^ (ints >> np.int64(63))).astype(np.uint64)
    return encoded.view(np.int64).astype(float, copy=False)


def _trailing_zero_count_uint64(value: int) -> int:
    value &= 0xFFFFFFFFFFFFFFFF
    if value == 0:
        return 64
    return (value & -value).bit_length() - 1


def _fast_apply_operator_to_series(
    series: pd.Series,
    operator_id: str,
    *,
    meta: Any,
) -> pd.Series:
    """Fast local transforms for entropy bookkeeping.

    These mirror ``myapp.dataset_operators`` semantics closely enough for the
    catalog entropy curves, while avoiding Python-per-value loops on GB-scale
    source CSVs.
    """

    as_float = str(getattr(meta, "logical_type", "")).strip().lower() in {"float", "double"}

    if operator_id == "delta":
        values = _finite_array(series)
        if values.size < 2:
            return pd.Series([], dtype=float)
        return pd.Series(np.diff(values), dtype=float)

    if operator_id == "rd":
        values = _finite_array(series)
        if values.size == 0:
            return pd.Series([], dtype=float)
        if values.size == 1:
            return pd.Series(values, dtype=float)
        out = np.empty(values.size, dtype=float)
        out[:-1] = values[:-1] - values[1:]
        out[-1] = values[-1]
        return pd.Series(out, dtype=float)

    if operator_id == "xor":
        values = _finite_array(series)
        if values.size < 2:
            return pd.Series([], dtype=float)
        bits = values.astype("<f8", copy=False).view("<u8")
        out = np.bitwise_xor(bits[:-1], bits[1:]).astype(float, copy=False)
        return pd.Series(out, dtype=float)

    if operator_id == "run_length":
        values = _finite_array(series)
        if values.size == 0:
            return pd.Series([], dtype=float)
        equal = np.isclose(values[1:], values[:-1], rtol=1e-15, atol=1e-15)
        starts = np.concatenate(([0], np.nonzero(~equal)[0] + 1))
        counts = np.diff(np.concatenate((starts, [values.size])))
        payload = np.empty(starts.size * 2, dtype=float)
        payload[0::2] = counts.astype(float)
        payload[1::2] = values[starts]
        return pd.Series(payload, dtype=float)

    if operator_id == "dictionary":
        values = _finite_array(series)
        if values.size == 0:
            return pd.Series([], dtype=float)
        codes, _ = pd.factorize(values, sort=False)
        return pd.Series(codes.astype(float, copy=False), dtype=float)

    if operator_id == "bitpacking":
        from myapp.dataset_operators import apply_bitpacking_series

        return apply_bitpacking_series(series)

    if operator_id == "trailing_zero_pack":
        values = _finite_array(series)
        if values.size == 0:
            return pd.Series([], dtype=float)
        bits = values.astype("<f8", copy=False).view("<u8")
        payload = np.empty(bits.size, dtype=np.uint64)
        out_pos = 0
        for start in range(0, bits.size, 8):
            block = bits[start : start + 8]
            if block.size == 1:
                payload[out_pos] = block[0]
                out_pos += 1
                continue
            xor_bits = np.empty(block.size, dtype=np.uint64)
            xor_bits[0] = block[0]
            xor_bits[1:] = np.bitwise_xor(block[:-1], block[1:])
            shift = min(_trailing_zero_count_uint64(int(value)) for value in xor_bits[1:])
            payload[out_pos : out_pos + block.size] = np.right_shift(xor_bits, shift)
            out_pos += block.size
        return pd.Series(payload[:out_pos].astype(float, copy=False), dtype=float)

    if operator_id == "eraser":
        values = _finite_array(series)
        if values.size == 0:
            return pd.Series([], dtype=float)
        bits = values.astype("<f8", copy=False).view("<u8").copy()
        bits &= np.uint64(0xFFFFFFFFFFFFF000)
        return pd.Series(bits.view(np.uint8).astype(float, copy=False), dtype=float)

    if operator_id == "scaling":
        arr = _numeric_array(series)
        if not as_float:
            return pd.Series(arr, index=series.index)
        decimal_places = int(getattr(meta, "max_point_number", 0) or 0)
        if decimal_places <= 0:
            decimal_places = _infer_decimal_places_fast(arr)
        if decimal_places <= 0:
            return pd.Series(arr, index=series.index)
        return pd.Series(np.round(arr * (10.0**decimal_places)), index=series.index)

    if operator_id == "frame_of_reference":
        arr = _numeric_array(series)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return pd.Series([], dtype=float)
        return pd.Series(arr - float(np.min(finite)), index=series.index)

    if operator_id == "value_reorder":
        values = _finite_array(series)
        if values.size == 0:
            return pd.Series([], dtype=float)
        return pd.Series(np.sort(values, kind="stable"), dtype=float)

    if operator_id == "regression_predictor":
        from myapp.dataset_operators import apply_regression_predictor_series

        return apply_regression_predictor_series(
            series,
            as_float=as_float,
            max_point_number=int(getattr(meta, "max_point_number", 0) or 0),
        )

    if operator_id == "zigzag":
        arr = _numeric_array(series)
        return pd.Series(_zigzag_encode_array(arr), index=series.index)

    from myapp.dataset_operators import _apply_operator_to_series

    return _apply_operator_to_series(series, operator_id, meta=meta)


def _apply_operator_to_streams(
    streams: PipelineStreams,
    operator_id: str,
    *,
    meta: Any,
) -> PipelineStreams:
    if operator_id == "outlier_separation":
        return _separate_outlier_streams(_flatten_streams(streams))

    if operator_id == "frame_of_reference" and streams.auxiliary:
        return PipelineStreams(
            primary=_fast_apply_operator_to_series(
                streams.primary,
                operator_id,
                meta=meta,
            ),
            auxiliary=streams.auxiliary,
        )

    if operator_id == "bitpacking" and streams.auxiliary:
        packed_parts = tuple(
            _fast_apply_operator_to_series(part, operator_id, meta=meta)
            for part in (streams.primary, *streams.auxiliary)
            if len(part) > 0
        )
        if not packed_parts:
            return PipelineStreams(pd.Series([], dtype=float))
        packed = pd.concat(packed_parts, ignore_index=True)
        packed.attrs["byte_tokens"] = True
        return PipelineStreams(packed)

    return PipelineStreams(
        _fast_apply_operator_to_series(
            _flatten_streams(streams),
            operator_id,
            meta=meta,
        )
    )


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.12g}"


def _pipeline_for_method(
    pipelines: dict[str, Any],
    method_name: str,
    data_type: str,
) -> list[Any]:
    raw = pipelines.get(method_name)
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []
    dtype = str(data_type or "").strip().lower()
    dtype_aliases = {
        "integer": "int",
        "long": "int",
        "double": "float",
    }
    dtype = dtype_aliases.get(dtype, dtype)
    if dtype in raw and isinstance(raw[dtype], list):
        return raw[dtype]
    for fallback in ("int", "float"):
        if fallback in raw and isinstance(raw[fallback], list):
            return raw[fallback]
    return []


def _operator_choices(step: Any) -> list[tuple[str, str]]:
    if isinstance(step, str):
        operator_id = step.strip()
        return [("", operator_id)] if operator_id else []
    if isinstance(step, dict):
        any_of = step.get("any_of")
        if isinstance(any_of, list):
            out: list[tuple[str, str]] = []
            for idx, raw in enumerate(any_of, start=1):
                operator_id = str(raw).strip()
                if operator_id:
                    out.append((str(idx), operator_id))
            return out
    return []


def _method_name_for_algorithm(
    algorithm: str,
    *,
    catalog_keys: tuple[str, ...],
) -> str | None:
    from myapp.method_catalog import method_name_for_algorithm_key

    method_name = method_name_for_algorithm_key(algorithm, catalog_keys=catalog_keys)
    if method_name:
        return method_name
    normalized = str(algorithm or "").strip()
    if normalized == "DICTIONARY":
        return "col_dictionary"
    if normalized == "col_rle":
        return "RLE"
    return None


def _build_step_rows_for_pair(
    *,
    dataset_key: str,
    source_csv: Path,
    result_file: CompressionResultFile,
    algorithm: str,
    method_name: str,
    column_info: ColumnInfo,
    source_series: pd.Series,
    pipeline: list[Any],
) -> list[dict[str, str]]:
    from myapp.dataset_operators import (
        SUPPORTED_OPERATORS,
        _ColumnCatalogMeta,
    )

    supported = set(SUPPORTED_OPERATORS)
    meta = _ColumnCatalogMeta(
        logical_type=column_info.logical_type,
        max_point_number=column_info.max_point_number,
    )
    current = PipelineStreams(source_series)
    blocked = False
    rows: list[dict[str, str]] = []

    for step_index, step in enumerate(pipeline, start=1):
        choices = _operator_choices(step)
        if not choices:
            continue
        for step_variant, operator_id in choices:
            transform_operator_id = OPERATOR_ALIASES.get(operator_id, operator_id)
            before_entropy, before_count = _byte_entropy(current)
            after_entropy: float | None = None
            after_count = 0
            status = "ok"
            next_current = current

            if blocked:
                status = "blocked_by_unsupported_previous_step"
            elif transform_operator_id not in supported:
                status = "unsupported_transform"
                blocked = True
            else:
                try:
                    transformed = _apply_operator_to_streams(
                        current,
                        transform_operator_id,
                        meta=meta,
                    )
                    after_entropy, after_count = _byte_entropy(transformed)
                    next_current = transformed
                except Exception as exc:
                    status = f"transform_error:{type(exc).__name__}"
                    blocked = True

            change = None
            if after_entropy is not None and before_entropy is not None:
                change = after_entropy - before_entropy
                if abs(change) < 1e-12:
                    change = None

            rows.append(
                {
                    "dataset_key": dataset_key,
                    "source_csv": str(source_csv),
                    "result_file": result_file.path.name,
                    "data_type": result_file.data_type,
                    "language": result_file.language,
                    "algorithm": algorithm,
                    "method_name": method_name,
                    "column_index": str(column_info.index),
                    "column_name": column_info.name,
                    "logical_type": column_info.logical_type,
                    "step_index": str(step_index),
                    "step_variant": step_variant,
                    "operator_id": operator_id,
                    "transform_operator_id": transform_operator_id,
                    "operator_status": status,
                    "before_byte_entropy": _fmt_float(before_entropy),
                    "after_byte_entropy": _fmt_float(after_entropy),
                    "byte_entropy_change": _fmt_float(change),
                    "before_count": str(before_count),
                    "after_count": str(after_count) if after_entropy is not None else "",
                }
            )
            if status == "ok" and step_variant == "":
                current = next_current
        if blocked:
            # Keep later steps in the output, but do not mutate the stream after
            # an unsupported transform because downstream inputs are unknown.
            continue

    return rows


def _existing_keys(path: Path) -> tuple[list[dict[str, str]], set[tuple[str, ...]]]:
    if not path.is_file():
        return [], set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        keys = set()
        for row in reader:
            normalized = {field: row.get(field, "") for field in FIELDNAMES}
            rows.append(normalized)
            keys.add(tuple(normalized.get(col, "") for col in IDENTITY_COLUMNS))
    return rows, keys


def _write_merged_rows(path: Path, new_rows: list[dict[str, str]]) -> tuple[int, int, int, Path | None]:
    existing_rows, keys = _existing_keys(path)
    row_index = {
        tuple(row.get(col, "") for col in IDENTITY_COLUMNS): idx
        for idx, row in enumerate(existing_rows)
    }
    appended: list[dict[str, str]] = []
    updated = 0
    for row in new_rows:
        normalized = {field: row.get(field, "") for field in FIELDNAMES}
        key = tuple(normalized.get(col, "") for col in IDENTITY_COLUMNS)
        if key in keys:
            old = existing_rows[row_index[key]]
            old_status = str(old.get("operator_status", ""))
            new_status = str(normalized.get("operator_status", ""))
            if normalized != old and (
                old_status != new_status
                or old_status.startswith("transform_error")
                or old_status in {"unsupported_transform", "blocked_by_unsupported_previous_step", "ok"}
            ):
                existing_rows[row_index[key]] = normalized
                updated += 1
            continue
        keys.add(key)
        row_index[key] = len(existing_rows) + len(appended)
        appended.append(normalized)

    if not appended and not updated and path.is_file():
        return 0, 0, len(existing_rows), None

    backup_path: Path | None = None
    if path.is_file():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_name(f"{path.stem}.bak-{stamp}{path.suffix}")
        shutil.copy2(path, backup_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(appended)
    os.replace(tmp, path)
    return len(appended), updated, len(existing_rows) + len(appended), backup_path


def process_dataset(
    *,
    dataset_csv: Path,
    backend_dataset_root: Path,
    allowed_data_types: set[str],
    allowed_algorithms: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    from myapp.method_catalog import load_method_operator_pipelines
    from myapp.lzbench_codec_catalog import all_catalog_algorithm_keys

    dataset_key = dataset_csv.stem
    bundle_dir = backend_dataset_root / dataset_key
    column_info_path = bundle_dir / f"{dataset_key}_column_info.csv"
    result_files = _compression_result_files(bundle_dir, dataset_key, allowed_data_types)
    summary: dict[str, Any] = {
        "dataset": dataset_key,
        "status": "ok",
        "result_files": len(result_files),
        "needed_columns": 0,
        "generated_rows": 0,
        "appended_rows": 0,
        "updated_rows": 0,
        "total_rows": 0,
        "backup": "",
        "skipped": "",
    }

    if not bundle_dir.is_dir():
        summary["status"] = "skipped"
        summary["skipped"] = "missing backend dataset directory"
        return summary
    if not column_info_path.is_file():
        summary["status"] = "skipped"
        summary["skipped"] = "missing column_info"
        return summary
    if not result_files:
        summary["status"] = "skipped"
        summary["skipped"] = "no matching compression_result files"
        return summary

    column_info = _read_column_info(column_info_path)
    needed_indices = _needed_column_indices(result_files, column_info)
    summary["needed_columns"] = len(needed_indices)
    if not needed_indices:
        summary["status"] = "skipped"
        summary["skipped"] = "no numeric columns in matching compression results"
        return summary

    columns = _read_source_columns(dataset_csv, needed_indices)
    pipelines = load_method_operator_pipelines()
    catalog_keys = all_catalog_algorithm_keys()
    method_cache: dict[str, str | None] = {}
    pipeline_cache: dict[tuple[str, str], list[Any]] = {}
    step_cache: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    generated_rows: list[dict[str, str]] = []

    for result_file in result_files:
        result_rows = _read_result_rows(result_file.path)
        pairs = sorted(
            {
                (str(row.get("algorithm", "")).strip(), int(row.get("column_index", -1)))
                for row in result_rows
                if str(row.get("algorithm", "")).strip()
                and (
                    not allowed_algorithms
                    or str(row.get("algorithm", "")).strip() in allowed_algorithms
                )
                and str(row.get("column_index", "")).strip().lstrip("-").isdigit()
            }
        )
        for algorithm, column_index in pairs:
            info = column_info.get(column_index)
            source_series = columns.get(column_index)
            if info is None or source_series is None:
                continue
            if info.logical_type not in NUMERIC_LOGICAL_TYPES:
                continue
            if algorithm not in method_cache:
                method_cache[algorithm] = _method_name_for_algorithm(
                    algorithm,
                    catalog_keys=catalog_keys,
                )
            method_name = method_cache[algorithm]
            if not method_name:
                continue
            pipeline_key = (method_name, result_file.data_type)
            if pipeline_key not in pipeline_cache:
                pipeline_cache[pipeline_key] = _pipeline_for_method(
                    pipelines,
                    method_name,
                    result_file.data_type,
                )
            pipeline = pipeline_cache[pipeline_key]
            if not pipeline:
                continue
            cache_key = (result_file.data_type, method_name, column_index)
            if cache_key not in step_cache:
                step_cache[cache_key] = _build_step_rows_for_pair(
                    dataset_key=dataset_key,
                    source_csv=dataset_csv,
                    result_file=result_file,
                    algorithm=algorithm,
                    method_name=method_name,
                    column_info=info,
                    source_series=source_series,
                    pipeline=pipeline,
                )
            for base_row in step_cache[cache_key]:
                row = dict(base_row)
                row.update(
                    {
                        "result_file": result_file.path.name,
                        "data_type": result_file.data_type,
                        "language": result_file.language,
                        "algorithm": algorithm,
                        "method_name": method_name,
                    }
                )
                generated_rows.append(row)

    summary["generated_rows"] = len(generated_rows)
    output_path = bundle_dir / f"{dataset_key}_operator_step_byte_entropy.csv"
    if dry_run:
        summary["status"] = "dry-run"
        return summary

    appended, updated, total, backup_path = _write_merged_rows(output_path, generated_rows)
    summary["appended_rows"] = appended
    summary["updated_rows"] = updated
    summary["total_rows"] = total
    summary["backup"] = str(backup_path) if backup_path else ""
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Directory containing the selected original dataset CSVs.",
    )
    parser.add_argument(
        "--backend-dataset-root",
        type=Path,
        default=DEFAULT_BACKEND_DATASET_ROOT,
        help="Directory containing benchmark dataset result bundles.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Only process this dataset key. Can be passed multiple times.",
    )
    parser.add_argument(
        "--data-type",
        action="append",
        choices=("int", "float"),
        default=[],
        help="Compression result data type to process. Default: int and float.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        default=[],
        help="Only process this algorithm key. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute rows and print the summary without writing output files.",
    )
    args = parser.parse_args()

    _configure_backend_imports()

    datasets_root = args.datasets_root.resolve()
    backend_dataset_root = args.backend_dataset_root.resolve()
    dataset_filter = {str(item).strip() for item in args.dataset if str(item).strip()}
    allowed_data_types = (
        {str(item).strip().lower() for item in args.data_type if str(item).strip()}
        if args.data_type
        else set(DEFAULT_DATA_TYPES)
    )
    allowed_algorithms = {
        str(item).strip() for item in args.algorithm if str(item).strip()
    }

    csv_paths = _iter_dataset_csvs(datasets_root, dataset_filter)
    if not csv_paths:
        print(f"No dataset CSVs found in {datasets_root}", file=sys.stderr)
        return 1

    print(
        "Mode: additive operator-step byte entropy "
        f"(datasets={len(csv_paths)}, data_types={','.join(sorted(allowed_data_types))})"
        ,
        flush=True,
    )
    failures = 0
    for csv_path in csv_paths:
        print(f"Processing {csv_path.stem} ...", flush=True)
        try:
            summary = process_dataset(
                dataset_csv=csv_path,
                backend_dataset_root=backend_dataset_root,
                allowed_data_types=allowed_data_types,
                allowed_algorithms=allowed_algorithms,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failures += 1
            print(
                f"FAILED {csv_path.stem}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            "{dataset}: {status}; files={result_files}; columns={needed_columns}; "
            "generated={generated_rows}; appended={appended_rows}; updated={updated_rows}; "
            "total={total_rows}; "
            "skip={skipped}".format(**summary),
            flush=True,
        )
        if summary["status"] == "skipped":
            continue
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
