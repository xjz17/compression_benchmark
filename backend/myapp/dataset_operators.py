"""Column-wise encoding operators and catalog statistics under ``operators/<id>/``."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_catalog_assets import (
    _read_catalog_master_csv,
    column_info_filename,
    numeric_column_names_from_column_info,
    write_numeric_catalog_assets,
)
from .statistic import count_trailing_zeros64, value_to_bits64

OPERATOR_DELTA = "delta"
OPERATOR_RD = "rd"
OPERATOR_FRAME_OF_REFERENCE = "frame_of_reference"
OPERATOR_VALUE_REORDER = "value_reorder"
OPERATOR_REGRESSION_PREDICTOR = "regression_predictor"
OPERATOR_ZIGZAG = "zigzag"
OPERATOR_XOR = "xor"
OPERATOR_RLE = "rle"
OPERATOR_RUN_LENGTH = "run_length"
OPERATOR_DICTIONARY = "dictionary"
OPERATOR_BITPACKING = "bitpacking"
OPERATOR_BITPACK_INT = "bitpack_int"
OPERATOR_TRAILING_ZERO_PACK = "trailing_zero_pack"
OPERATOR_SCALING = "scaling"
OPERATOR_OUTLIER_SEPARATION = "outlier_separation"
OPERATOR_SUBCOLUMN = "subcolumn"
OPERATOR_ERASER = "eraser"
OPERATOR_EASE = "ease"
OPERATOR_PONGO_ERASER = "pongo_eraser"
OPERATOR_DICTIONARY_FILTER = "dictionary_filter"
OPERATOR_BYTE_DELTA_FILTER = "byte_delta_filter"
OPERATOR_BLOCK_TRANSFORM = "block_transform"
OPERATOR_SYMBOL_RECODE = "symbol_recode"
OPERATOR_LZ_MATCH = "lz_match"
OPERATOR_MATCH_PARSE_POLICY = "match_parse_policy"
OPERATOR_MATCH_WINDOW_MODEL = "match_window_model"
OPERATOR_MATCH_TOKEN_FORMAT = "match_token_format"
OPERATOR_INTEGER_CODE = "integer_code"
OPERATOR_BITSTREAM_PACK = "bitstream_pack"
OPERATOR_ENTROPY_HUFFMAN = "entropy_huffman"
OPERATOR_ENTROPY_RANGE = "entropy_range"
OPERATOR_ENTROPY_FSE = "entropy_fse"
OPERATOR_ENTROPY_SPECIAL = "entropy_special"
OPERATOR_LITERAL_ENTROPY = "literal_entropy"
OPERATOR_BLOCK_STATE_CONTROL = "block_state_control"
OPERATOR_CONTAINER_FRAME = "container_frame"

SUPPORTED_OPERATORS: tuple[str, ...] = (
    OPERATOR_DELTA,
    OPERATOR_RD,
    OPERATOR_FRAME_OF_REFERENCE,
    OPERATOR_VALUE_REORDER,
    OPERATOR_REGRESSION_PREDICTOR,
    OPERATOR_ZIGZAG,
    OPERATOR_XOR,
    OPERATOR_RUN_LENGTH,
    OPERATOR_DICTIONARY,
    OPERATOR_BITPACKING,
    OPERATOR_BITPACK_INT,
    OPERATOR_TRAILING_ZERO_PACK,
    OPERATOR_SCALING,
    OPERATOR_OUTLIER_SEPARATION,
    OPERATOR_SUBCOLUMN,
    OPERATOR_ERASER,
    OPERATOR_EASE,
    OPERATOR_PONGO_ERASER,
    OPERATOR_DICTIONARY_FILTER,
    OPERATOR_BYTE_DELTA_FILTER,
    OPERATOR_BLOCK_TRANSFORM,
    OPERATOR_SYMBOL_RECODE,
    OPERATOR_LZ_MATCH,
    OPERATOR_MATCH_PARSE_POLICY,
    OPERATOR_MATCH_WINDOW_MODEL,
    OPERATOR_MATCH_TOKEN_FORMAT,
    OPERATOR_INTEGER_CODE,
    OPERATOR_BITSTREAM_PACK,
    OPERATOR_ENTROPY_HUFFMAN,
    OPERATOR_ENTROPY_RANGE,
    OPERATOR_ENTROPY_FSE,
    OPERATOR_ENTROPY_SPECIAL,
    OPERATOR_LITERAL_ENTROPY,
    OPERATOR_BLOCK_STATE_CONTROL,
    OPERATOR_CONTAINER_FRAME,
)

OPERATORS_DIRNAME = "operators"
LEGACY_TRANSFORMED_CSV_SUFFIX = "_transformed.csv"
CATALOG_OPERATOR_BLOCK_SIZE = 8

_FLOAT_LOGICAL_TYPES = frozenset({"float", "double"})


@dataclass(frozen=True)
class _ColumnCatalogMeta:
    logical_type: str
    max_point_number: int


def operators_root(bundle_dir: Path) -> Path:
    return bundle_dir / OPERATORS_DIRNAME


def operator_bundle_dir(bundle_dir: Path, operator_id: str) -> Path:
    return operators_root(bundle_dir) / operator_id


def operator_statistic_csv_path(bundle_dir: Path, key: str, operator_id: str) -> Path:
    return operator_bundle_dir(bundle_dir, operator_id) / f"{key}_numeric_statistic.csv"


def list_available_operators(bundle_dir: Path) -> list[str]:
    root = operators_root(bundle_dir)
    if not root.is_dir():
        return []
    out: list[str] = []
    for operator_id in SUPPORTED_OPERATORS:
        op_dir = root / operator_id
        if op_dir.is_dir() and any(op_dir.glob("*_numeric_statistic.csv")):
            out.append(operator_id)
    return out


def ensure_operator_catalog_assets(
    key: str,
    master_csv: str,
    bundle_dir: Path,
    operator_id: str,
) -> Path:
    stat_csv = operator_statistic_csv_path(bundle_dir, key, operator_id)
    if stat_csv.is_file():
        return stat_csv
    write_operator_catalog_assets(key, master_csv, bundle_dir, operator_id)
    if not stat_csv.is_file():
        raise FileNotFoundError(f"Operator statistic file was not generated: {stat_csv}")
    return stat_csv


def _remove_legacy_transformed_csv(op_dir: Path, dataset_key: str) -> None:
    legacy = op_dir / f"{dataset_key}{LEGACY_TRANSFORMED_CSV_SUFFIX}"
    if legacy.is_file():
        legacy.unlink()


def _uint64_to_signed_int64(value: int) -> int:
    return struct.unpack("<q", struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))[0]


def _infer_max_decimal_places(arr: np.ndarray) -> int:
    max_places = 0
    for raw in arr:
        if not np.isfinite(raw):
            continue
        text = format(Decimal(str(float(raw))), "f").rstrip("0").rstrip(".")
        if "." in text:
            max_places = max(max_places, len(text.split(".", 1)[1]))
    return max_places


def _finite_float_values(series: pd.Series) -> list[float]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return [float(x) for x in finite.to_numpy(dtype=float)]


def _iter_value_blocks(values: list[float], block_size: int = CATALOG_OPERATOR_BLOCK_SIZE):
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        if block:
            yield block


def _values_equal(a: float, b: float) -> bool:
    return a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-15 * max(1.0, abs(a), abs(b)))


def apply_delta_series(series: pd.Series) -> pd.Series:
    """Catalog delta: ``np.diff`` on finite values (length ``n−1``)."""
    values = _finite_float_values(series)
    if len(values) < 2:
        return pd.Series([], dtype=float)
    return pd.Series(np.diff(np.asarray(values, dtype=float)))


def apply_rd_series(series: pd.Series) -> pd.Series:
    """Reverse first-order difference (thesis §3.4)."""
    values = _finite_float_values(series)
    n = len(values)
    if n == 0:
        return pd.Series([], dtype=float)
    if n == 1:
        return pd.Series([values[0]], dtype=float)
    out = np.empty(n, dtype=float)
    out[-1] = values[-1]
    for idx in range(n - 2, -1, -1):
        out[idx] = values[idx] - values[idx + 1]
    return pd.Series(out)


def apply_xor_series(series: pd.Series) -> pd.Series:
    """Consecutive IEEE754 float64 XOR (catalog charts)."""
    values = _finite_float_values(series)
    if len(values) < 2:
        return pd.Series([], dtype=float)
    xor_values: list[float] = []
    pre_value_b = value_to_bits64(values[0], as_float=True)
    for idx in range(1, len(values)):
        b_value = value_to_bits64(values[idx], as_float=True)
        xor_values.append(float(pre_value_b ^ b_value))
        pre_value_b = b_value
    return pd.Series(xor_values, dtype=float)


def apply_scaling_series(
    series: pd.Series,
    *,
    as_float: bool,
    max_point_number: int = 0,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not as_float:
        return numeric
    arr = numeric.to_numpy(dtype=float)
    decimal_places = max_point_number if max_point_number > 0 else _infer_max_decimal_places(arr)
    if decimal_places <= 0:
        return numeric
    scale = float(10**decimal_places)
    out = np.round(arr.astype(np.float64, copy=False) * scale)
    return pd.Series(out, index=series.index)


def apply_frame_of_reference_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return pd.Series(np.zeros(len(arr), dtype=float), index=series.index)
    col_min = float(np.min(finite))
    return pd.Series(arr - col_min, index=series.index)


def apply_value_reorder_series(series: pd.Series) -> pd.Series:
    values = np.asarray(_finite_float_values(series), dtype=float)
    if values.size == 0:
        return pd.Series([], dtype=float)
    return pd.Series(np.sort(values, kind="stable"), dtype=float)


def apply_regression_predictor_series(
    series: pd.Series,
    *,
    as_float: bool = False,
    max_point_number: int = 0,
) -> pd.Series:
    """REGER-style integer prediction residuals.

    REGER predicts over an integer representation: floating-point columns are
    decimal-scaled first, coefficients are stored as float32, and predictions
    are truncated to int64 before residual calculation. Keeping fractional OLS
    errors as floats would measure random mantissa bytes rather than the encoded
    residual stream.
    """
    scaled = apply_scaling_series(
        series,
        as_float=as_float,
        max_point_number=max_point_number,
    )
    values = np.asarray(_finite_float_values(scaled), dtype=float)
    if values.size < 2:
        return pd.Series([], dtype=float)

    info = np.iinfo(np.int64)
    lower = np.nextafter(float(info.min), 0.0)
    upper = np.nextafter(float(info.max), 0.0)
    ints = np.rint(np.clip(values, lower, upper)).astype(np.int64, copy=False)
    x = ints[:-1]
    y = ints[1:]
    n = len(x)
    sx = math.fsum(float(value) for value in x)
    sy = math.fsum(float(value) for value in y)
    sxx = math.fsum(float(value) * float(value) for value in x)
    sxy = math.fsum(float(left) * float(right) for left, right in zip(x, y))
    denom = float(n) * sxx - sx * sx
    if abs(denom) <= 1e-12:
        theta0, theta1 = sy / float(n), 0.0
    else:
        theta1 = (float(n) * sxy - sx * sy) / denom
        theta0 = (sy - theta1 * sx) / float(n)
    theta0 = struct.unpack("<f", struct.pack("<f", theta0))[0]
    theta1 = struct.unpack("<f", struct.pack("<f", theta1))[0]

    residuals: list[float] = []
    for previous, current in zip(x, y):
        raw_prediction = theta0 + theta1 * float(previous)
        if raw_prediction <= float(info.min):
            prediction = int(info.min)
        elif raw_prediction >= float(info.max):
            prediction = int(info.max)
        else:
            prediction = int(raw_prediction)
        residual = int(current) - prediction
        residuals.append(float(min(int(info.max), max(int(info.min), residual))))
    return pd.Series(residuals, dtype=float)


def _zigzag_encode_int64(value: int) -> int:
    n = int(value)
    encoded = (n << 1) ^ (n >> 63)
    return _uint64_to_signed_int64(encoded & 0xFFFFFFFFFFFFFFFF)


def apply_zigzag_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = np.zeros(len(numeric), dtype=np.int64)
    for idx, raw in enumerate(numeric.to_numpy()):
        if not np.isfinite(raw):
            out[idx] = 0
            continue
        out[idx] = _zigzag_encode_int64(int(raw))
    return pd.Series(out, index=series.index)


def apply_rle_series(series: pd.Series) -> pd.Series:
    """Run-length payload ``[count, value, ...]`` (thesis §3.10)."""
    values = _finite_float_values(series)
    if not values:
        return pd.Series([], dtype=float)
    payload: list[float] = []
    idx = 0
    while idx < len(values):
        value = values[idx]
        count = 1
        while idx + count < len(values) and _values_equal(values[idx + count], value):
            count += 1
        payload.append(float(count))
        payload.append(float(value))
        idx += count
    return pd.Series(payload)


def apply_dictionary_series(series: pd.Series) -> pd.Series:
    """Dictionary indices in encounter order (thesis §3.11)."""
    values = _finite_float_values(series)
    if not values:
        return pd.Series([], dtype=float)
    mapping: dict[float, int] = {}
    out: list[float] = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(float(mapping[value]))
    return pd.Series(out)


def apply_bitpacking_series(series: pd.Series) -> pd.Series:
    """Approximate bit-pack payload as the minimal little-endian integer bytes."""
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return pd.Series([], dtype=float)
    info = np.iinfo(np.int64)
    clipped = np.clip(finite, np.nextafter(float(info.min), 0.0), np.nextafter(float(info.max), 0.0))
    ints = np.rint(clipped).astype(np.int64, copy=False)
    min_val = int(np.min(ints))
    shifted = (ints - min_val).astype(np.uint64, copy=False)
    max_val = int(np.max(shifted)) if shifted.size else 0
    width = max(1, (max_val.bit_length() + 7) // 8)
    raw = shifted.astype("<u8", copy=False).view(np.uint8).reshape(-1, 8)[:, :width].ravel()
    return _series_from_byte_values(raw)


def apply_trailing_zero_pack_series(series: pd.Series) -> pd.Series:
    """Block-wise XOR trailing-zero shift payload (thesis §3.15)."""
    values = _finite_float_values(series)
    payload: list[float] = []
    for block in _iter_value_blocks(values):
        if len(block) == 1:
            payload.append(float(value_to_bits64(block[0], as_float=True)))
            continue
        xor_bits: list[int] = []
        prev = value_to_bits64(block[0], as_float=True)
        xor_bits.append(prev)
        for idx in range(1, len(block)):
            cur = value_to_bits64(block[idx], as_float=True)
            xor_bits.append(prev ^ cur)
            prev = cur
        shift = min(count_trailing_zeros64(bits) for bits in xor_bits[1:]) if len(xor_bits) > 1 else 0
        payload.extend(float(bits >> shift) for bits in xor_bits)
    return pd.Series(payload)


def _byte_values_from_series(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.asarray([], dtype=np.uint8)
    if getattr(series, "attrs", {}).get("byte_tokens"):
        return np.clip(np.rint(finite), 0, 255).astype(np.uint8, copy=False)
    return finite.astype("<f4", copy=False).view(np.uint8).copy()


def _series_from_byte_values(values: np.ndarray | list[int]) -> pd.Series:
    arr = np.asarray(values, dtype=np.uint8)
    out = pd.Series(arr.astype(float, copy=False), dtype=float)
    out.attrs["byte_tokens"] = True
    return out


def _byte_delta(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.size == 0:
        return values
    out = np.empty(values.size, dtype=np.uint8)
    out[0] = values[0]
    if values.size > 1:
        out[1:] = ((values[1:].astype(np.int16) - values[:-1].astype(np.int16)) & 0xFF).astype(np.uint8)
    return out


def _frequency_ranked_bytes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.size == 0:
        return values
    counts = np.bincount(values, minlength=256)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(256, dtype=np.uint8)
    rank[order] = np.arange(256, dtype=np.uint8)
    return rank[values]


def _entropy_code_bytes(values: np.ndarray, *, salt: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.size == 0:
        return values
    counts = np.bincount(values, minlength=256)
    total = float(values.size)
    probs = counts.astype(float) / total
    lengths = np.zeros(256, dtype=np.uint8)
    nonzero = probs > 0
    lengths[nonzero] = np.maximum(1, np.ceil(-np.log2(probs[nonzero]))).astype(np.uint8)
    ranked = _frequency_ranked_bytes(values)
    bucket = max(2, 2 + (salt % 7))
    coded = ((lengths[values].astype(np.uint16) * (17 + salt)) + (ranked.astype(np.uint16) // bucket) + salt) & 0xFF
    return coded.astype(np.uint8)


def apply_byte_delta_filter_series(series: pd.Series) -> pd.Series:
    return _series_from_byte_values(_byte_delta(_byte_values_from_series(series)))


def apply_dictionary_filter_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    ranked = _frequency_ranked_bytes(values)
    return _series_from_byte_values(((ranked.astype(np.uint16) * 3 + values.astype(np.uint16)) & 0xFF).astype(np.uint8))


def apply_block_transform_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    chunks = []
    for start in range(0, values.size, 64):
        block = np.sort(values[start:start + 64])
        chunks.append(_byte_delta(block))
    return _series_from_byte_values(np.concatenate(chunks) if chunks else np.asarray([], dtype=np.uint8))


def apply_symbol_recode_series(series: pd.Series) -> pd.Series:
    ranked = _frequency_ranked_bytes(_byte_values_from_series(series))
    return _series_from_byte_values((ranked.astype(np.uint16) // 2).astype(np.uint8))


def apply_lz_match_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    ranked = _frequency_ranked_bytes(values)
    deltas = _byte_delta(values)
    positions = (np.arange(values.size, dtype=np.uint16) & 0xFF).astype(np.uint8)
    token_kind = (ranked < 64).astype(np.uint8)
    tokens = np.empty(values.size * 3, dtype=np.uint8)
    tokens[0::3] = token_kind
    tokens[1::3] = ((ranked.astype(np.uint16) * 5 + positions.astype(np.uint16)) & 0xFF).astype(np.uint8)
    tokens[2::3] = deltas
    return _series_from_byte_values(tokens)


def apply_match_parse_policy_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    idx = np.arange(values.size, dtype=np.uint16)
    return _series_from_byte_values(((values.astype(np.uint16) * 3 + idx % 251) & 0xFF).astype(np.uint8))


def apply_match_window_model_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    window = ((np.arange(values.size, dtype=np.uint16) * 131) & 0xFF).astype(np.uint8)
    return _series_from_byte_values(np.bitwise_xor(values, window))


def apply_match_token_format_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    if values.size % 2:
        values = np.concatenate((values, np.asarray([0], dtype=np.uint8)))
    tokens = (((values[0::2] >> 4).astype(np.uint16) << 4) | (values[1::2].astype(np.uint16) & 0x0F)).astype(np.uint8)
    return _series_from_byte_values(tokens)


def apply_integer_code_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return pd.Series([], dtype=float)
    finite = np.clip(np.abs(finite), 0.0, float(np.iinfo(np.uint64).max // 2))
    finite = np.rint(finite).astype(np.uint64, copy=False)
    out = np.empty(finite.size * 2, dtype=np.uint8)
    out[0::2] = (finite & np.uint64(0x7F)).astype(np.uint8)
    out[1::2] = ((finite >> np.uint64(7)) & np.uint64(0x7F)).astype(np.uint8)
    return _series_from_byte_values(out)


def apply_bitstream_pack_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    if values.size % 2:
        values = np.concatenate((values, np.asarray([0], dtype=np.uint8)))
    packed = (((values[0::2] & 0x0F).astype(np.uint16) << 4) | (values[1::2].astype(np.uint16) & 0x0F)).astype(np.uint8)
    return _series_from_byte_values(packed)


def apply_entropy_coding_series(series: pd.Series, *, salt: int = 0) -> pd.Series:
    return _series_from_byte_values(_entropy_code_bytes(_byte_values_from_series(series), salt=salt))


def apply_block_state_control_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    block_ids = np.arange(values.size, dtype=np.uint32) // np.uint32(64)
    markers = ((block_ids * np.uint32(29)) & np.uint32(0xFF)).astype(np.uint8)
    return _series_from_byte_values(np.bitwise_xor(values, markers))


def apply_container_frame_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    checksum = int(np.sum(values, dtype=np.uint64) & np.uint64(0xFF))
    header = np.asarray([31, 139, values.size & 0xFF, (values.size >> 8) & 0xFF], dtype=np.uint8)
    trailer = np.asarray([checksum, checksum ^ 0xFF], dtype=np.uint8)
    return _series_from_byte_values(np.concatenate((header, values, trailer)))


def apply_outlier_separation_series(series: pd.Series) -> pd.Series:
    values = np.asarray(_finite_float_values(series), dtype=float)
    if values.size == 0:
        return pd.Series([], dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = max(q3 - q1, 1.0)
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask = ((values < low) | (values > high)).astype(float)
    clipped = np.clip(values, low, high)
    return pd.Series(np.concatenate((mask, clipped)), dtype=float)


def apply_subcolumn_series(series: pd.Series) -> pd.Series:
    values = _byte_values_from_series(series)
    if values.size == 0:
        return pd.Series([], dtype=float)
    pad = (-values.size) % 4
    if pad:
        values = np.concatenate((values, np.zeros(pad, dtype=np.uint8)))
    subcolumns = values.reshape(-1, 4).T.reshape(-1)
    return _series_from_byte_values(_byte_delta(subcolumns))


def apply_ease_series(series: pd.Series) -> pd.Series:
    values = np.asarray(_finite_float_values(series), dtype=float)
    if values.size == 0:
        return pd.Series([], dtype=float)
    bits = values.astype("<f8", copy=False).view("<u8").copy()
    bits &= np.uint64(0xFFFFFFFFFFFFF000)
    return _series_from_byte_values(bits.view(np.uint8))


def apply_pongo_eraser_series(series: pd.Series, *, max_point_number: int = 0) -> pd.Series:
    values = np.asarray(_finite_float_values(series), dtype=float)
    if values.size == 0:
        return pd.Series([], dtype=float)
    bits = values.astype("<f8", copy=False).view("<u8").copy()
    inferred_places = _infer_max_decimal_places(values)
    places = max_point_number if max_point_number > 0 else inferred_places
    places = max(0, min(15, int(places)))
    erase_bits = max(0, min(52, (15 - places) * 3))
    if erase_bits:
        bits &= np.uint64(0xFFFFFFFFFFFFFFFF ^ ((1 << erase_bits) - 1))
    return _series_from_byte_values(bits.view(np.uint8))


def _column_catalog_meta(bundle_dir: Path, key: str) -> dict[str, _ColumnCatalogMeta]:
    import csv

    path = bundle_dir / column_info_filename(key)
    out: dict[str, _ColumnCatalogMeta] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("column_name", "")).strip()
            if not name:
                continue
            logical = str(row.get("logical_type", "")).strip().lower()
            try:
                max_point = int(row.get("max_point_number", 0) or 0)
            except (TypeError, ValueError):
                max_point = 0
            out[name] = _ColumnCatalogMeta(
                logical_type=logical,
                max_point_number=max(0, max_point),
            )
    return out


def _normalize_operator_id(operator_id: str) -> str:
    if operator_id == OPERATOR_BITPACK_INT:
        return OPERATOR_BITPACKING
    if operator_id == OPERATOR_RLE:
        return OPERATOR_RUN_LENGTH
    if operator_id in {OPERATOR_EASE, OPERATOR_PONGO_ERASER}:
        return OPERATOR_ERASER
    return operator_id


def _apply_operator_to_series(
    series: pd.Series,
    operator_id: str,
    *,
    meta: _ColumnCatalogMeta,
) -> pd.Series:
    operator_id = _normalize_operator_id(operator_id)
    as_float = meta.logical_type in _FLOAT_LOGICAL_TYPES
    if operator_id == OPERATOR_DELTA:
        return apply_delta_series(series)
    if operator_id == OPERATOR_RD:
        return apply_rd_series(series)
    if operator_id == OPERATOR_XOR:
        return apply_xor_series(series)
    if operator_id == OPERATOR_RUN_LENGTH:
        return apply_rle_series(series)
    if operator_id == OPERATOR_DICTIONARY:
        return apply_dictionary_series(series)
    if operator_id == OPERATOR_BITPACKING:
        return apply_bitpacking_series(series)
    if operator_id == OPERATOR_TRAILING_ZERO_PACK:
        return apply_trailing_zero_pack_series(series)
    if operator_id == OPERATOR_SCALING:
        return apply_scaling_series(
            series,
            as_float=as_float,
            max_point_number=meta.max_point_number,
        )
    if operator_id == OPERATOR_FRAME_OF_REFERENCE:
        return apply_frame_of_reference_series(series)
    if operator_id == OPERATOR_VALUE_REORDER:
        return apply_value_reorder_series(series)
    if operator_id == OPERATOR_REGRESSION_PREDICTOR:
        return apply_regression_predictor_series(
            series,
            as_float=as_float,
            max_point_number=meta.max_point_number,
        )
    if operator_id == OPERATOR_ZIGZAG:
        return apply_zigzag_series(series)
    if operator_id == OPERATOR_OUTLIER_SEPARATION:
        return apply_outlier_separation_series(series)
    if operator_id == OPERATOR_SUBCOLUMN:
        return apply_subcolumn_series(series)
    if operator_id == OPERATOR_ERASER:
        return apply_ease_series(series)
    if operator_id == OPERATOR_DICTIONARY_FILTER:
        return apply_dictionary_filter_series(series)
    if operator_id == OPERATOR_BYTE_DELTA_FILTER:
        return apply_byte_delta_filter_series(series)
    if operator_id == OPERATOR_BLOCK_TRANSFORM:
        return apply_block_transform_series(series)
    if operator_id == OPERATOR_SYMBOL_RECODE:
        return apply_symbol_recode_series(series)
    if operator_id == OPERATOR_LZ_MATCH:
        return apply_lz_match_series(series)
    if operator_id == OPERATOR_MATCH_PARSE_POLICY:
        return apply_match_parse_policy_series(series)
    if operator_id == OPERATOR_MATCH_WINDOW_MODEL:
        return apply_match_window_model_series(series)
    if operator_id == OPERATOR_MATCH_TOKEN_FORMAT:
        return apply_match_token_format_series(series)
    if operator_id == OPERATOR_INTEGER_CODE:
        return apply_integer_code_series(series)
    if operator_id == OPERATOR_BITSTREAM_PACK:
        return apply_bitstream_pack_series(series)
    if operator_id == OPERATOR_ENTROPY_HUFFMAN:
        return apply_entropy_coding_series(series, salt=1)
    if operator_id == OPERATOR_ENTROPY_RANGE:
        return apply_entropy_coding_series(series, salt=3)
    if operator_id == OPERATOR_ENTROPY_FSE:
        return apply_entropy_coding_series(series, salt=5)
    if operator_id == OPERATOR_ENTROPY_SPECIAL:
        return apply_entropy_coding_series(series, salt=7)
    if operator_id == OPERATOR_LITERAL_ENTROPY:
        return apply_entropy_coding_series(series, salt=9)
    if operator_id == OPERATOR_BLOCK_STATE_CONTROL:
        return apply_block_state_control_series(series)
    if operator_id == OPERATOR_CONTAINER_FRAME:
        return apply_container_frame_series(series)
    raise ValueError(f"Unsupported operator: {operator_id!r}")


def build_operator_transformed_dataframe(
    key: str,
    master_csv: str,
    bundle_dir: Path,
    operator_id: str,
    *,
    column_info_csv: Path,
) -> pd.DataFrame:
    if operator_id not in SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported operator: {operator_id!r}")

    numeric_names = numeric_column_names_from_column_info(column_info_csv)
    if not numeric_names:
        return pd.DataFrame()

    df = _read_catalog_master_csv(key, master_csv)
    meta_by_name = _column_catalog_meta(bundle_dir, key)
    transformed: dict[str, pd.Series] = {}
    for col_name in numeric_names:
        if col_name not in df.columns:
            continue
        meta = meta_by_name.get(
            col_name,
            _ColumnCatalogMeta(logical_type="float", max_point_number=0),
        )
        transformed[col_name] = _apply_operator_to_series(
            df[col_name],
            operator_id,
            meta=meta,
        )
    return pd.DataFrame(transformed)


def write_operator_catalog_assets(
    key: str,
    master_csv: str,
    bundle_dir: Path,
    operator_id: str,
) -> Path:
    if operator_id not in SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported operator: {operator_id!r}")

    col_info = bundle_dir / column_info_filename(key)
    if not col_info.is_file():
        raise FileNotFoundError(f"Missing column info: {col_info}")

    op_dir = operator_bundle_dir(bundle_dir, operator_id)
    transformed_df = build_operator_transformed_dataframe(
        key,
        master_csv,
        bundle_dir,
        operator_id,
        column_info_csv=col_info,
    )
    if transformed_df.empty:
        raise ValueError(f"No numeric columns to transform for {key!r}")

    op_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_transformed_csv(op_dir, key)

    write_numeric_catalog_assets(
        key,
        master_csv,
        bundle_dir,
        column_info_csv=col_info,
        output_bundle_dir=op_dir,
        numeric_df=transformed_df,
    )
    return op_dir
