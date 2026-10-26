#!/usr/bin/env python3
"""Deterministic reversible byte carriers for the 85 x 97 joint benchmark.

The downstream object is the complete O2R1 representation of one operator
block.  General-purpose codecs receive the bytes directly.  Algorithms whose
native contract is numeric receive bounded exact integer values; this is a
declared transport adapter, not a claim that the algorithm natively consumes
an arbitrary operator port graph.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import struct
from typing import Final


MAGIC: Final = b"DBC1"
VERSION: Final = 1
HEADER = struct.Struct("<4sBBBBQQQ32s")

DIRECT_U8: Final = "direct_u8"
INT64_U56: Final = "int64_u56"
FLOAT64_U48: Final = "float64_u48"
FLOAT64_U8: Final = "float64_u8"

_KIND_TO_ID = {DIRECT_U8: 1, INT64_U56: 2, FLOAT64_U48: 3, FLOAT64_U8: 4}
_ID_TO_KIND = {value: key for key, value in _KIND_TO_ID.items()}
_KIND_SHAPE = {
    DIRECT_U8: ("uint8", 1, 1),
    INT64_U56: ("int64", 7, 8),
    FLOAT64_U48: ("float64", 6, 8),
    FLOAT64_U8: ("float64", 1, 8),
}


BYTE_ALGORITHMS: Final = frozenset(
    {
        "ANS", "Arithmetic", "BSC", "Blosc", "Blosc2", "BriefLZ",
        "Brotli", "Bzip2", "Bzip3", "Dict.", "FSE", "FastLZ",
        "FastLZMA2", "Huff0", "Huffman", "Kanzi", "LZ4", "LZ4 Fast",
        "LZ4HC", "LZAV", "LZF", "LZFSE", "LZHAM", "LZMAT", "LZO1",
        "LZRW", "LZVN", "Libdeflate", "Lizard", "NRV", "PPMD8",
        "Pithy", "QuickLZ", "RLE", "SLZ", "Snappy", "TAMP", "WFLZ",
        "XPACK", "XZ", "YALZ77", "Yappy", "ZXC", "Zlib", "Zlib-ng",
        "Zling", "Zstd", "gzip", "CSC", "Crush", "Density", "Gipfeli",
        "LZG", "LZJB", "LZO2A", "LZSSE", "Lzlib", "Shrinker", "Tornado",
    }
)

# These methods admit an exact int64 carrier in the current reviewed routes.
# Seven payload bytes per value keep every value in [0, 2^56), so adjacent
# deltas cannot overflow int64 and reserved negative sentinels cannot occur.
INT64_CARRIER_ALGORITHMS: Final = frozenset(
    {
        "2DIFF", "BOS", "REGER", "SVB", "FastPFor", "Simple8b", "RLBE",
        "DBP", "LECO", "Chimp128", "Pongo", "Gorilla", "Sprintz",
    }
)

# Six payload bytes become one exactly representable finite binary64 integer
# in [0, 2^48).  This excludes NaN, infinity, signed-zero ambiguity and codec
# sentinel bit patterns without an uncharged exception side channel.
FLOAT64_CARRIER_ALGORITHMS: Final = frozenset(
    {
        "ALP", "Chimp", "Elf", "Elf+", "FPC", "FPCmp-R",
        "FPCmp-S", "FPZIP", "PFPC", "SPDP", "CORAD", "BUFF",
    }
)

# Camel's integer-delta wire field is intentionally narrower than the adjacent
# delta of arbitrary uint48 words.  One byte per finite binary64 value keeps
# every adjacent delta in [-255, 255] without changing Camel, hiding bytes, or
# invoking a substitute codec.  The 8x carrier expansion and DBC1 metadata are
# charged in the complete frame just like the other numeric carriers.
FLOAT64_U8_ALGORITHMS: Final = frozenset({"Camel"})

ACTIVE85: Final = (BYTE_ALGORITHMS | INT64_CARRIER_ALGORITHMS |
                   FLOAT64_CARRIER_ALGORITHMS | FLOAT64_U8_ALGORITHMS)

if len(BYTE_ALGORITHMS) != 59:
    raise RuntimeError("byte algorithm partition no longer has 59 entries")
if len(INT64_CARRIER_ALGORITHMS) != 13:
    raise RuntimeError("int64 carrier partition no longer has 13 entries")
if len(FLOAT64_CARRIER_ALGORITHMS) != 12:
    raise RuntimeError("float64_u48 carrier partition no longer has 12 entries")
if len(FLOAT64_U8_ALGORITHMS) != 1:
    raise RuntimeError("float64_u8 carrier partition no longer has one entry")
if len(ACTIVE85) != 85:
    raise RuntimeError("algorithm carrier partitions are not an exact 85-way union")


@dataclasses.dataclass(frozen=True)
class PackedCarrier:
    kind: str
    dtype: str
    row_count: int
    values: bytes
    missing: bytes
    metadata: bytes
    source_bytes: int
    lane_bytes: int
    storage_width: int
    padding_bytes: int

    @property
    def carrier_numeric_bytes(self) -> int:
        return len(self.values)

    @property
    def carrier_metadata_bytes(self) -> int:
        return len(self.metadata)

    @property
    def missing_bitmap_bytes(self) -> int:
        return len(self.missing)


def carrier_kind_for_algorithm(algorithm: str) -> str:
    if algorithm in BYTE_ALGORITHMS:
        return DIRECT_U8
    if algorithm in INT64_CARRIER_ALGORITHMS:
        return INT64_U56
    if algorithm in FLOAT64_CARRIER_ALGORITHMS:
        return FLOAT64_U48
    if algorithm in FLOAT64_U8_ALGORITHMS:
        return FLOAT64_U8
    raise ValueError(f"algorithm is outside frozen active85 carrier map: {algorithm}")


def _zero_missing(row_count: int) -> bytes:
    return bytes((row_count + 7) // 8)


def _metadata(kind: str, original: bytes, carrier_count: int,
              padding_bytes: int) -> bytes:
    dtype, lane_bytes, storage_width = _KIND_SHAPE[kind]
    del dtype
    return HEADER.pack(
        MAGIC,
        VERSION,
        _KIND_TO_ID[kind],
        lane_bytes,
        storage_width,
        len(original),
        carrier_count,
        padding_bytes,
        hashlib.sha256(original).digest(),
    )


def pack_bytes(payload: bytes, kind: str) -> PackedCarrier:
    if kind not in _KIND_SHAPE:
        raise ValueError(f"unknown byte carrier: {kind}")
    dtype, lane_bytes, storage_width = _KIND_SHAPE[kind]
    carrier_count = (len(payload) + lane_bytes - 1) // lane_bytes
    padding = carrier_count * lane_bytes - len(payload)

    if kind == DIRECT_U8:
        values = payload
    elif kind == INT64_U56:
        values_out = bytearray()
        for offset in range(0, len(payload), lane_bytes):
            chunk = payload[offset : offset + lane_bytes]
            word = int.from_bytes(chunk.ljust(lane_bytes, b"\0"), "little")
            if not 0 <= word < (1 << 56):
                raise AssertionError("uint56 carrier construction overflowed")
            values_out.extend(struct.pack("<q", word))
        values = bytes(values_out)
    elif kind == FLOAT64_U48:
        values_out = bytearray()
        for offset in range(0, len(payload), lane_bytes):
            chunk = payload[offset : offset + lane_bytes]
            word = int.from_bytes(chunk.ljust(lane_bytes, b"\0"), "little")
            if not 0 <= word < (1 << 48):
                raise AssertionError("uint48 carrier construction overflowed")
            value = float(word)
            if not math.isfinite(value) or int(value) != word:
                raise AssertionError("uint48 is not exact in binary64")
            values_out.extend(struct.pack("<d", value))
        values = bytes(values_out)
    else:
        # One source byte becomes one exactly representable finite double.
        values = b"".join(struct.pack("<d", float(byte)) for byte in payload)

    if len(values) != carrier_count * storage_width:
        raise AssertionError("carrier numeric byte count is inconsistent")
    metadata = _metadata(kind, payload, carrier_count, padding)
    return PackedCarrier(
        kind=kind,
        dtype=dtype,
        row_count=carrier_count,
        values=values,
        missing=_zero_missing(carrier_count),
        metadata=metadata,
        source_bytes=len(payload),
        lane_bytes=lane_bytes,
        storage_width=storage_width,
        padding_bytes=padding,
    )


def _parse_metadata(metadata: bytes) -> tuple[str, int, int, int, bytes]:
    if len(metadata) != HEADER.size:
        raise ValueError("DBC1 metadata length differs from the fixed contract")
    magic, version, kind_id, lane_bytes, storage_width, original_length, \
        carrier_count, padding, digest = HEADER.unpack(metadata)
    if magic != MAGIC or version != VERSION or kind_id not in _ID_TO_KIND:
        raise ValueError("invalid DBC1 header")
    kind = _ID_TO_KIND[kind_id]
    _, expected_lane, expected_storage = _KIND_SHAPE[kind]
    if lane_bytes != expected_lane or storage_width != expected_storage:
        raise ValueError("DBC1 carrier shape differs from its kind")
    if original_length + padding != carrier_count * lane_bytes:
        raise ValueError("DBC1 length/padding identity failed")
    if padding >= lane_bytes and carrier_count != 0:
        raise ValueError("DBC1 padding is non-canonical")
    if carrier_count == 0 and (original_length != 0 or padding != 0):
        raise ValueError("empty DBC1 carrier has nonzero lengths")
    return kind, original_length, carrier_count, padding, digest


def unpack_bytes(values: bytes, missing: bytes, metadata: bytes) -> bytes:
    kind, original_length, carrier_count, padding, digest = _parse_metadata(metadata)
    _, lane_bytes, storage_width = _KIND_SHAPE[kind]
    if len(values) != carrier_count * storage_width:
        raise ValueError("carrier value byte count differs from DBC1")
    if missing != _zero_missing(carrier_count):
        raise ValueError("byte carrier missing bitmap is not canonical all-present")

    if kind == DIRECT_U8:
        reconstructed = values
    elif kind == INT64_U56:
        output = bytearray()
        for (word,) in struct.iter_unpack("<q", values):
            if not 0 <= word < (1 << 56):
                raise ValueError("int64 carrier value is outside uint56")
            output.extend(word.to_bytes(lane_bytes, "little"))
        reconstructed = bytes(output)
    elif kind == FLOAT64_U48:
        output = bytearray()
        for (value,) in struct.iter_unpack("<d", values):
            if not math.isfinite(value) or value < 0 or value >= (1 << 48) or \
                    value != math.floor(value):
                raise ValueError("float64 carrier is not an exact uint48")
            output.extend(int(value).to_bytes(lane_bytes, "little"))
        reconstructed = bytes(output)
    else:
        output = bytearray()
        for (value,) in struct.iter_unpack("<d", values):
            if not math.isfinite(value) or value < 0 or value >= 256 or \
                    value != math.floor(value):
                raise ValueError("float64_u8 carrier is not an exact byte")
            output.append(int(value))
        reconstructed = bytes(output)

    if padding and any(reconstructed[original_length:]):
        raise ValueError("byte carrier has nonzero canonical padding")
    result = reconstructed[:original_length]
    if hashlib.sha256(result).digest() != digest:
        raise ValueError("byte carrier payload SHA-256 mismatch")
    return result
