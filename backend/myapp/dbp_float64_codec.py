"""DBF1/DBF2 self-describing, bit-exact floating DBP wrappers.

The wire contract lives in
``analysis_outputs/dbp_float_e2e_20261009/WIRE_CONTRACT.md``.  DBF1 keeps the
project's delta / ZigZag / unsigned LEB128 body, but records the information a
standalone decoder needs to reconstruct the original binary64 or binary32 lanes.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Iterable, Sequence

from .benchmark_loss import accumulate_double_list_loss
from .benchmark_wire import BENCHMARK_FLOAT_BYTES


MAGIC = b"DBF1"
MAGIC32 = b"DBF2"
MODE_DECIMAL = 0
MODE_IEEE_BITS = 1
RAW_BITS_SCALE = 0xFF
HEADER_SIZE = 10
MAX_SCALE = 18
MAX_VALUES = 10_000

_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_HEADER = struct.Struct("<4sBBI")
_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_F64 = struct.Struct("<d")
_F32 = struct.Struct("<f")


def _float_bits(value: float) -> int:
    return _U64.unpack(_F64.pack(float(value)))[0]


def _bits_float(word: int) -> float:
    return _F64.unpack(_U64.pack(word & _MASK64))[0]


def _bits_float32(word: int) -> float:
    return _F32.unpack(_U32.pack(word & 0xFFFFFFFF))[0]


def _i64_word(value: int) -> int:
    return int(value) & _MASK64


def _word_i64(word: int) -> int:
    word &= _MASK64
    return word - (1 << 64) if word & _SIGN64 else word


def _zigzag_i64(value: int) -> int:
    return ((value << 1) ^ (value >> 63)) & _MASK64


def _unzigzag_i64(value: int) -> int:
    value &= _MASK64
    return (value >> 1) ^ -(value & 1)


def _append_uleb128_u64(out: bytearray, value: int) -> None:
    value &= _MASK64
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _read_uleb128_u64(blob: bytes, pos: int) -> tuple[int, int]:
    """Read one canonical uint64 ULEB128 value."""

    value = 0
    for byte_index in range(10):
        if pos >= len(blob):
            raise ValueError("truncated DBF1 varint")
        byte = blob[pos]
        pos += 1

        # A uint64 has only one payload bit in byte ten.  This also rejects a
        # continuation bit in that byte.
        if byte_index == 9 and (byte & 0xFE):
            raise ValueError("overflowing DBF1 varint")
        value |= (byte & 0x7F) << (7 * byte_index)
        if not byte & 0x80:
            if byte_index and (byte & 0x7F) == 0:
                raise ValueError("noncanonical DBF1 varint")
            return value, pos
    raise ValueError("overflowing DBF1 varint")


def _encode_words(words: Sequence[int]) -> bytes:
    if not words:
        return b""
    out = bytearray(_U64.pack(words[0] & _MASK64))
    previous = words[0] & _MASK64
    for raw_word in words[1:]:
        current = raw_word & _MASK64
        delta_word = (current - previous) & _MASK64
        signed_delta = _word_i64(delta_word)
        _append_uleb128_u64(out, _zigzag_i64(signed_delta))
        previous = current
    return bytes(out)


def _decode_words(body: bytes, count: int) -> list[int]:
    if count == 0:
        if body:
            raise ValueError("trailing bytes after empty DBF1 frame")
        return []
    if len(body) < 8:
        raise ValueError("truncated DBF1 first word")

    current = _U64.unpack_from(body, 0)[0]
    words = [current]
    pos = 8
    for _ in range(1, count):
        encoded_delta, pos = _read_uleb128_u64(body, pos)
        current = (current + _unzigzag_i64(encoded_delta)) & _MASK64
        words.append(current)
    if pos != len(body):
        raise ValueError("trailing bytes after DBF1 body")
    return words


def _scale_limit(max_point: int | None) -> int:
    if max_point is None:
        return MAX_SCALE
    try:
        value = int(max_point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_point must be an integer or None") from exc
    if value < 0:
        return MAX_SCALE
    return min(value, MAX_SCALE)


def _exact_decimal_codes(values: Sequence[float], scale: int) -> list[int] | None:
    factor = 10.0**scale
    codes: list[int] = []
    for value in values:
        if not math.isfinite(value):
            return None
        product = value * factor
        if not math.isfinite(product):
            return None
        code = round(product)  # Python rounds binary64 ties to even.
        if code < _I64_MIN or code > _I64_MAX:
            return None
        restored = float(code) / factor
        if _float_bits(restored) != _float_bits(value):
            return None
        codes.append(code)
    return codes


def _exact_decimal_codes32(words: Sequence[int], scale: int) -> list[int] | None:
    factor = 10.0**scale
    codes: list[int] = []
    for word in words:
        value = _bits_float32(word)
        if not math.isfinite(value):
            return None
        product = value * factor
        if not math.isfinite(product):
            return None
        code = round(product)
        if code < _I64_MIN or code > _I64_MAX:
            return None
        restored = float(code) / factor
        try:
            restored_word = _U32.unpack(_F32.pack(restored))[0]
        except OverflowError:
            return None
        if restored_word != word:
            return None
        codes.append(code)
    return codes


def encode(values: Iterable[float], max_point: int | None = 18) -> bytes:
    """Encode at most 10,000 float64 values into one complete DBF1 frame.

    Decimal scales are tried from ``min(max_point, 18)`` down to zero.  If no
    scale reconstructs every binary64 lane bit-for-bit, the whole block stores
    the original IEEE-754 words.
    """

    floats = [float(value) for value in values]
    count = len(floats)
    if count > MAX_VALUES:
        raise ValueError(f"DBF1 block has {count} values; maximum is {MAX_VALUES}")
    if count == 0:
        return _HEADER.pack(MAGIC, MODE_IEEE_BITS, RAW_BITS_SCALE, 0)

    for scale in range(_scale_limit(max_point), -1, -1):
        codes = _exact_decimal_codes(floats, scale)
        if codes is not None:
            words = [_i64_word(code) for code in codes]
            return _HEADER.pack(MAGIC, MODE_DECIMAL, scale, count) + _encode_words(
                words
            )

    words = [_float_bits(value) for value in floats]
    return _HEADER.pack(MAGIC, MODE_IEEE_BITS, RAW_BITS_SCALE, count) + _encode_words(
        words
    )


def encode32(
    raw_bytes: bytes | bytearray | memoryview,
    max_point: int | None = 18,
) -> bytes:
    """Encode raw little-endian float32 lanes into one complete DBF2 frame.

    The raw-byte API is deliberate: it preserves signaling/quiet NaN payloads
    and signed zero even when Python's float conversion would alter them.
    """

    raw = bytes(raw_bytes)
    if len(raw) % 4:
        raise ValueError("raw float32 input length must be divisible by 4")
    count = len(raw) // 4
    if count > MAX_VALUES:
        raise ValueError(f"DBF2 block has {count} values; maximum is {MAX_VALUES}")
    if count == 0:
        return _HEADER.pack(MAGIC32, MODE_IEEE_BITS, RAW_BITS_SCALE, 0)

    words = list(struct.unpack(f"<{count}I", raw))
    for scale in range(_scale_limit(max_point), -1, -1):
        codes = _exact_decimal_codes32(words, scale)
        if codes is not None:
            code_words = [_i64_word(code) for code in codes]
            return _HEADER.pack(MAGIC32, MODE_DECIMAL, scale, count) + _encode_words(
                code_words
            )
    return _HEADER.pack(MAGIC32, MODE_IEEE_BITS, RAW_BITS_SCALE, count) + _encode_words(
        words
    )


def decode(blob: bytes | bytearray | memoryview) -> list[float]:
    """Decode one complete DBF1 frame, rejecting malformed or trailing data."""

    frame = bytes(blob)
    if len(frame) < HEADER_SIZE:
        raise ValueError("truncated DBF1 header")
    magic, mode, scale, count = _HEADER.unpack_from(frame, 0)
    if magic != MAGIC:
        raise ValueError("invalid DBF1 magic")
    if count > MAX_VALUES:
        raise ValueError(f"impossible DBF1 count {count}")
    if mode == MODE_DECIMAL:
        if scale > MAX_SCALE:
            raise ValueError("invalid DBF1 decimal scale")
        if count == 0:
            raise ValueError("noncanonical DBF1 empty mode")
    elif mode == MODE_IEEE_BITS:
        if scale != RAW_BITS_SCALE:
            raise ValueError("invalid DBF1 raw-bits scale")
    else:
        raise ValueError("invalid DBF1 mode")

    words = _decode_words(frame[HEADER_SIZE:], count)
    if mode == MODE_IEEE_BITS:
        return [_bits_float(word) for word in words]

    factor = 10.0**scale
    return [float(_word_i64(word)) / factor for word in words]


def decode32(blob: bytes | bytearray | memoryview) -> bytes:
    """Decode one complete DBF2 frame to raw little-endian float32 lanes."""

    frame = bytes(blob)
    if len(frame) < HEADER_SIZE:
        raise ValueError("truncated DBF2 header")
    magic, mode, scale, count = _HEADER.unpack_from(frame, 0)
    if magic != MAGIC32:
        raise ValueError("invalid DBF2 magic")
    if count > MAX_VALUES:
        raise ValueError(f"impossible DBF2 count {count}")
    if mode == MODE_DECIMAL:
        if scale > MAX_SCALE:
            raise ValueError("invalid DBF2 decimal scale")
        if count == 0:
            raise ValueError("noncanonical DBF2 empty mode")
    elif mode == MODE_IEEE_BITS:
        if scale != RAW_BITS_SCALE:
            raise ValueError("invalid DBF2 raw-bits scale")
    else:
        raise ValueError("invalid DBF2 mode")

    words = _decode_words(frame[HEADER_SIZE:], count)
    if mode == MODE_IEEE_BITS:
        if any(word > 0xFFFFFFFF for word in words):
            raise ValueError("DBF2 raw-bits word exceeds uint32")
        return b"".join(_U32.pack(word) for word in words)

    factor = 10.0**scale
    output = bytearray()
    for word in words:
        restored = float(_word_i64(word)) / factor
        try:
            output.extend(_F32.pack(restored))
        except OverflowError as exc:
            raise ValueError("DBF2 scaled value overflows float32") from exc
    return bytes(output)


def _packed_float64(values: Sequence[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"<{len(values)}d", *values)


def bench_delta_zigzag_varint_float_columns(
    columns: Sequence[Sequence[float]],
    max_points: Sequence[int] | None = None,
) -> tuple[int, int, int, int, float, float]:
    """Benchmark complete DBF1 frames and require a bit-exact round trip.

    Encode timing includes float conversion, scale selection and the complete
    header/body allocation.  Decode timing includes parsing and float
    reconstruction.  The independent byte-for-byte guard runs outside both
    timing intervals and raises immediately on failure.
    """

    total_original = total_compressed = total_encode_ns = total_decode_ns = 0
    loss_acc = [0.0, 0.0]
    for column_index, column in enumerate(columns):
        if not column:
            continue
        max_point = (
            max_points[column_index]
            if max_points is not None and column_index < len(max_points)
            else None
        )

        for block_start in range(0, len(column), MAX_VALUES):
            block = column[block_start : block_start + MAX_VALUES]
            encode_start = time.perf_counter_ns()
            original = [float(value) for value in block]
            compressed = encode(original, max_point=max_point)
            encode_end = time.perf_counter_ns()
            restored = decode(compressed)
            decode_end = time.perf_counter_ns()

            original_bytes = _packed_float64(original)
            restored_bytes = _packed_float64(restored)
            if restored_bytes != original_bytes:
                raise RuntimeError(
                    "DBF1 bitwise round-trip failure in "
                    f"column {column_index}, block offset {block_start}"
                )

            total_original += len(original) * BENCHMARK_FLOAT_BYTES
            total_compressed += len(compressed)
            total_encode_ns += encode_end - encode_start
            total_decode_ns += decode_end - encode_end
            accumulate_double_list_loss(original, restored, loss_acc)

    return (
        total_original,
        total_compressed,
        total_encode_ns,
        total_decode_ns,
        loss_acc[0],
        loss_acc[1],
    )


__all__ = [
    "HEADER_SIZE",
    "MAGIC",
    "MAGIC32",
    "MAX_SCALE",
    "MAX_VALUES",
    "MODE_DECIMAL",
    "MODE_IEEE_BITS",
    "RAW_BITS_SCALE",
    "bench_delta_zigzag_varint_float_columns",
    "decode",
    "decode32",
    "encode",
    "encode32",
]
