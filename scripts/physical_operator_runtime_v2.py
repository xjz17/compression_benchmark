#!/usr/bin/env python3
"""Legacy PV2F codec-candidate runtime retained for experiment replay.

This module's 105 registered entries mix transforms, packers, and complete
catalog codecs.  They are valid historical PV2F wire candidates, but complete
codec names are not primitive RL actions.  New operator-composition work uses
``primitive_operator_action_runtime.py``, where algorithm names are pipeline
provenance and the selectable actions are independently reversible operators.

Every entry here has an executable implementation and a stable wire ID.
Catalog names remain separate codec candidates even when the normalized-int64
path shares code.
The final byte stream contains transform metadata, action IDs, the packer
payload (optionally wrapped by a real byte compressor), an outer frame, and a
mandatory CRC32 over the original normalized-int64 bytes.

This is a research wire format, not a production codec or a claim that the
prototype implementations are binary-compatible with Parquet, Gorilla, Chimp,
or FastPFOR.  ``implementation_source`` in the registry states exactly what is
executed.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import math
import struct
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import brotli
import lz4.frame
import numpy as np
import zstandard


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from myapp.active_method_policy import (  # noqa: E402
    active_method_policy_summary,
    is_active_algorithm_key,
    is_active_method_name,
)


MAGIC = b"PV2F"
VERSION = 2
FLAG_CRC32 = 1
HEADER = struct.Struct("<4sBBBBQQII")
U16 = struct.Struct("<H")
U32 = struct.Struct("<I")
I64 = struct.Struct("<q")
MASK64 = (1 << 64) - 1


class PhysicalCodecError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionSpec:
    wire_id: int
    name: str
    phase: str
    family: str
    implementation_source: str
    conceptual_stages: tuple[str, ...]
    block_size: int = 0
    dictionary_size: int = 0
    window_size: int = 0
    exception_stream: bool = False
    inner_framing: str = "none"
    inner_checksum: str = "none"
    catalog_method: str = ""
    codec_key: str = ""


TRANSFORMS: tuple[ActionSpec, ...] = (
    ActionSpec(1, "delta.parquet_prev.global", "transform", "delta", "physical_operator_runtime_v2.apply_delta_global", ("delta",)),
    ActionSpec(2, "delta.tsfile_prev.block128", "transform", "delta", "physical_operator_runtime_v2.apply_delta_block", ("block_state", "delta"), block_size=128),
    ActionSpec(3, "delta.tsfile_prev.block1024", "transform", "delta", "physical_operator_runtime_v2.apply_delta_block", ("block_state", "delta"), block_size=1024),
    ActionSpec(4, "delta2.tsfile.block128", "transform", "delta2", "physical_operator_runtime_v2.apply_delta2_block", ("block_state", "delta", "delta_of_delta"), block_size=128),
    ActionSpec(5, "for.bos.block128", "transform", "frame_of_reference", "physical_operator_runtime_v2.apply_for_block", ("block_state", "frame_of_reference"), block_size=128),
    ActionSpec(6, "for.bos.block1024", "transform", "frame_of_reference", "physical_operator_runtime_v2.apply_for_block", ("block_state", "frame_of_reference"), block_size=1024),
    ActionSpec(7, "xor.gorilla.prev", "transform", "xor", "physical_operator_runtime_v2.apply_xor_global", ("xor_predictor",)),
    ActionSpec(8, "outlier.bos.iqr.block128", "transform", "outlier_separation", "physical_operator_runtime_v2.apply_outlier_block", ("block_state", "outlier_separation", "exception_stream"), block_size=128, exception_stream=True),
    ActionSpec(9, "outlier.bos.iqr.block1024", "transform", "outlier_separation", "physical_operator_runtime_v2.apply_outlier_block", ("block_state", "outlier_separation", "exception_stream"), block_size=1024, exception_stream=True),
    ActionSpec(10, "dictionary.parquet.freq256", "transform", "dictionary", "physical_operator_runtime_v2.apply_dictionary", ("dictionary", "dictionary_index", "escape_stream"), dictionary_size=256, exception_stream=True),
    ActionSpec(11, "dictionary.parquet.freq4096", "transform", "dictionary", "physical_operator_runtime_v2.apply_dictionary", ("dictionary", "dictionary_index", "escape_stream"), dictionary_size=4096, exception_stream=True),
    ActionSpec(12, "rle.parquet.runs", "transform", "run_length", "physical_operator_runtime_v2.apply_rle_transform", ("run_length",)),
)


BASE_PACKERS: tuple[ActionSpec, ...] = (
    ActionSpec(101, "raw.i64", "packer", "raw", "physical_operator_runtime_v2.pack_raw", ("literal_i64",)),
    ActionSpec(102, "varint.parquet.zigzag", "packer", "integer_code", "physical_operator_runtime_v2.pack_varint", ("zigzag", "leb128")),
    ActionSpec(103, "bitpack.dbp.global", "packer", "bitpack", "physical_operator_runtime_v2.pack_bitpack", ("frame_of_reference", "bitpack")),
    ActionSpec(104, "bitpack.dbp.block128", "packer", "bitpack", "physical_operator_runtime_v2.pack_bitpack", ("block_state", "frame_of_reference", "bitpack"), block_size=128),
    ActionSpec(105, "bitpack.dbp.block1024", "packer", "bitpack", "physical_operator_runtime_v2.pack_bitpack", ("block_state", "frame_of_reference", "bitpack"), block_size=1024),
    ActionSpec(106, "pfor.prototype.block128.p90", "packer", "pfor", "physical_operator_runtime_v2.pack_pfor", ("block_state", "frame_of_reference", "bitpack", "exception_stream"), block_size=128, exception_stream=True),
    ActionSpec(107, "pfor.prototype.block256.p95", "packer", "pfor", "physical_operator_runtime_v2.pack_pfor", ("block_state", "frame_of_reference", "bitpack", "exception_stream"), block_size=256, exception_stream=True),
    ActionSpec(108, "tsfile.delta128", "packer", "delta_binary_packed", "backend.myapp.tsfile_encoding.p02.encode_ts_2diff_int64", ("block_state", "delta", "min_delta", "bitpack"), block_size=128),
    ActionSpec(109, "tsfile.gorilla64", "packer", "xor", "backend.myapp.tsfile_encoding.p03_i64.encode_gorilla_int64", ("xor_predictor", "leading_trailing_zero", "bitstream_pack")),
    ActionSpec(110, "tsfile.chimp128", "packer", "xor", "backend.myapp.tsfile_encoding.p03_i64.encode_chimp_int64", ("chimp_history_128", "xor_predictor", "leading_trailing_zero", "bitstream_pack"), window_size=128),
    ActionSpec(111, "tsfile.sprintz", "packer", "sprintz", "backend.myapp.tsfile_encoding.p04_i64.encode_sprintz_int64", ("block_state", "predictor", "bitpack")),
    ActionSpec(112, "streamvbyte.delta1.limb32", "packer", "streamvbyte", "backend.myapp.int64_column_codec.stream_vbyte_encode_int64", ("limb_split", "delta", "streamvbyte")),
    ActionSpec(113, "simple8b.limb32", "packer", "simple8b", "backend.myapp.int64_column_codec.simple8b_encode_int64", ("limb_split", "simple8b")),
)


CATALOG_PACKER_METHODS: tuple[tuple[str, str], ...] = (
    ("Chimp", "catalog.specialist.chimp"),
    ("Pongo", "catalog.specialist.pongo"),
    ("BOS", "catalog.specialist.bos"),
    ("REGER", "catalog.specialist.reger"),
    ("FastPFor", "catalog.specialist.fastpfor"),
    ("RLE", "catalog.specialist.rle"),
    ("RLBE", "catalog.specialist.rlbe"),
    ("Dict.", "catalog.specialist.dictionary"),
    ("DBP", "catalog.specialist.dbp"),
    ("ALP", "catalog.specialist.alp"),
    ("LECO", "catalog.specialist.leco"),
    ("Camel", "catalog.specialist.camel"),
    ("Elf", "catalog.specialist.elf"),
    ("Elf+", "catalog.specialist.elf_plus"),
    ("FPC", "catalog.specialist.fpc"),
    ("PFPC", "catalog.specialist.pfpc"),
    ("SPDP", "catalog.specialist.spdp.level9"),
    ("BUFF", "catalog.specialist.buff.int64"),
    ("FPZIP", "catalog.specialist.fpzip.precision0"),
    ("FPCmp-R", "catalog.specialist.fpcmp_ratio"),
    ("FPCmp-S", "catalog.specialist.fpcmp_speed"),
)

CATALOG_PACKERS: tuple[ActionSpec, ...] = tuple(
    ActionSpec(
        114 + index,
        action_name,
        "packer",
        "catalog_specialist",
        f"backend.myapp.catalog_action_codecs.pack_specialist_int64:{method}",
        ("catalog_specialist", method),
        catalog_method=method,
    )
    for index, (method, action_name) in enumerate(CATALOG_PACKER_METHODS)
)

PACKERS: tuple[ActionSpec, ...] = BASE_PACKERS + CATALOG_PACKERS


IDENTITY_COMPRESSOR = ActionSpec(201, "none", "compressor", "none", "identity", ("none",))

# Kept for decoding old PV2F streams. They are not active search actions because
# the catalog below retains one empirically selected level per method.
LEGACY_COMPRESSORS: tuple[ActionSpec, ...] = (
    ActionSpec(202, "zlib.deflate.level1", "compressor", "deflate", "stdlib.zlib.compress(level=1)", ("lz77", "huffman", "zlib_frame"), window_size=32768, inner_framing="zlib", inner_checksum="adler32"),
    ActionSpec(203, "zlib.deflate.level6", "compressor", "deflate", "stdlib.zlib.compress(level=6)", ("lz77", "huffman", "zlib_frame"), window_size=32768, inner_framing="zlib", inner_checksum="adler32"),
    ActionSpec(204, "zstd.level3", "compressor", "zstd", "python-zstandard.ZstdCompressor(level=3)", ("lz_match", "literal_huffman", "sequence_fse", "zstd_frame"), window_size=0, inner_framing="zstd", inner_checksum="xxh64_32"),
    ActionSpec(205, "zstd.level9", "compressor", "zstd", "python-zstandard.ZstdCompressor(level=9)", ("lz_match", "literal_huffman", "sequence_fse", "zstd_frame"), window_size=0, inner_framing="zstd", inner_checksum="xxh64_32"),
    ActionSpec(206, "lzma2.xz.dict1m", "compressor", "lzma", "stdlib.lzma FORMAT_XZ LZMA2 dict_size=1MiB", ("lzma_match", "range_coder", "xz_frame"), dictionary_size=1 << 20, inner_framing="xz", inner_checksum="crc64"),
    ActionSpec(207, "lzma2.xz.dict8m", "compressor", "lzma", "stdlib.lzma FORMAT_XZ LZMA2 dict_size=8MiB", ("lzma_match", "range_coder", "xz_frame"), dictionary_size=1 << 23, inner_framing="xz", inner_checksum="crc64"),
    ActionSpec(208, "brotli.quality5.window22", "compressor", "brotli", "brotli.compress(quality=5,lgwin=22)", ("context_model", "lz_match", "prefix_code", "brotli_stream"), window_size=1 << 22, inner_framing="brotli", inner_checksum="none"),
    ActionSpec(209, "bzip2.level9", "compressor", "bzip2", "stdlib.bz2.compress(compresslevel=9)", ("block_sort", "move_to_front", "run_length", "huffman", "bzip2_frame"), block_size=900_000, inner_framing="bzip2", inner_checksum="crc32"),
    ActionSpec(210, "lz4.frame.64k", "compressor", "lz4", "lz4.frame.compress(64KiB linked blocks, checksums)", ("lz4_match", "lz4_frame"), block_size=65_536, window_size=65_536, inner_framing="lz4-frame", inner_checksum="block+content"),
)


GENERAL_BEST_KEYS: tuple[tuple[str, str], ...] = (
    ("gzip", "gz"),
    ("Brotli", "brotli"),
    ("Zstd", "zstd_level_9"),
    ("LZ4", "lz4-framed"),
    ("XZ", "xz_level_6"),
    ("Density", "density"),
    ("Gipfeli", "gipfeli"),
    ("GLZA", "glza"),
    ("LZMAT", "lzmat"),
    ("LZO1", "lzo1x_112"),
    ("LZO2A", "lzo2a"),
    ("LZRW", "lzrw"),
    ("LZSSE", "lzsse8"),
    ("LZVN", "lzvn"),
    ("Pithy", "pithy"),
    ("Shrinker", "shrinker"),
    ("NRV", "ucl_nrv2b_9"),
    ("WFLZ", "wflz"),
    ("XPACK", "xpack"),
    ("YALZ77", "yalz77"),
    ("Yappy", "yappy"),
    ("Snappy", "snappy-framed"),
    ("ANS", "ans"),
    ("FSE", "fse"),
    ("Huffman", "huffman"),
    ("Huff0", "huff0"),
    ("Arithmetic", "arithmetic"),
    ("Blosc", "blosc_blosclz"),
    ("Blosc2", "blosc2_blosclz"),
    ("BriefLZ", "brieflz_9"),
    ("BSC", "bsc4"),
    ("Bzip2", "bzip2_1"),
    ("Bzip3", "bzip3_10"),
    ("Crush", "crush_2"),
    ("CSC", "csc_4"),
    ("FastLZ", "fastlz_2"),
    ("FastLZMA2", "fastlzma2_10"),
    ("Kanzi", "kanzi_4"),
    ("Libdeflate", "libdeflate_12"),
    ("Lizard", "lizard_10"),
    ("LZ4 Fast", "lz4fast_17"),
    ("LZ4HC", "lz4hc_10"),
    ("LZAV", "lzav_2"),
    ("LZF", "lzf_1"),
    ("LZFSE", "lzfse"),
    ("LZG", "lzg_4"),
    ("LZHAM", "lzham_4"),
    ("LZJB", "lzjb"),
    ("Lzlib", "lzlib_4"),
    ("PPMD8", "ppmd8_4"),
    ("QuickLZ", "quicklz_3"),
    ("SLZ", "slz_deflate_2"),
    ("Tornado", "tornado_11"),
    ("Zlib", "zlib_9"),
    ("Zlib-ng", "zlib_ng_8"),
    ("Zling", "zling_4"),
    ("ZXC", "zxc_6"),
    ("TAMP", "tamp_10"),
)


def _action_slug(text: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in text).strip("_")


ALL_CATALOG_COMPRESSORS: tuple[ActionSpec, ...] = tuple(
    ActionSpec(
        301 + index,
        f"catalog.general.{_action_slug(method)}.{key}",
        "compressor",
        "catalog_general",
        f"backend.myapp.catalog_action_codecs.compress_general:{key}",
        ("byte_codec", key),
        catalog_method=method,
        codec_key=key,
    )
    for index, (method, key) in enumerate(GENERAL_BEST_KEYS)
)

# Keep retired wire IDs decodable, but do not expose them to new searches.
CATALOG_COMPRESSORS: tuple[ActionSpec, ...] = tuple(
    spec
    for spec in ALL_CATALOG_COMPRESSORS
    if is_active_method_name(spec.catalog_method)
    and is_active_algorithm_key(spec.codec_key)
)
RETIRED_CATALOG_COMPRESSORS: tuple[ActionSpec, ...] = tuple(
    spec for spec in ALL_CATALOG_COMPRESSORS if spec not in CATALOG_COMPRESSORS
)

COMPRESSORS: tuple[ActionSpec, ...] = (IDENTITY_COMPRESSOR,) + CATALOG_COMPRESSORS

SPECIALIST_ACTION_BY_METHOD: dict[str, str | None] = {
    "Chimp": "catalog.specialist.chimp",
    "Chimp128": "tsfile.chimp128",
    "Pongo": "catalog.specialist.pongo",
    "Gorilla": "tsfile.gorilla64",
    "2DIFF": "tsfile.delta128",
    "BOS": "catalog.specialist.bos",
    "REGER": "catalog.specialist.reger",
    "SVB": "streamvbyte.delta1.limb32",
    "FastPFor": "catalog.specialist.fastpfor",
    "Simple8b": "simple8b.limb32",
    "RLE": "catalog.specialist.rle",
    "RLBE": "catalog.specialist.rlbe",
    "Sprintz": "tsfile.sprintz",
    "Dict.": "catalog.specialist.dictionary",
    "DBP": "catalog.specialist.dbp",
    "ALP": "catalog.specialist.alp",
    "LECO": "catalog.specialist.leco",
    "Camel": "catalog.specialist.camel",
    "Elf": "catalog.specialist.elf",
    "Elf+": "catalog.specialist.elf_plus",
    "FPC": "catalog.specialist.fpc",
    "PFPC": "catalog.specialist.pfpc",
    "SPDP": "catalog.specialist.spdp.level9",
    "CORAD": None,
    "BUFF": "catalog.specialist.buff.int64",
    "FPZIP": "catalog.specialist.fpzip.precision0",
    "FPCmp-R": "catalog.specialist.fpcmp_ratio",
    "FPCmp-S": "catalog.specialist.fpcmp_speed",
}
GENERAL_ACTION_BY_METHOD = {spec.catalog_method: spec.name for spec in CATALOG_COMPRESSORS}


ALL_ACTIONS = TRANSFORMS + PACKERS + COMPRESSORS
DECODABLE_ACTIONS = ALL_ACTIONS + LEGACY_COMPRESSORS + RETIRED_CATALOG_COMPRESSORS
ACTION_BY_NAME = {spec.name: spec for spec in ALL_ACTIONS}
ACTION_BY_ID = {spec.wire_id: spec for spec in DECODABLE_ACTIONS}
if len(ACTION_BY_NAME) != len(ALL_ACTIONS) or len(ACTION_BY_ID) != len(DECODABLE_ACTIONS):
    raise RuntimeError("duplicate action name or wire ID")


@dataclass(frozen=True)
class TransformRecord:
    spec: ActionSpec
    metadata: bytes


@dataclass(frozen=True)
class TransformState:
    values: np.ndarray
    records: tuple[TransformRecord, ...] = ()


@dataclass(frozen=True)
class EncodedFrame:
    data: bytes
    components: dict[str, int]
    pipeline: tuple[str, ...]


def as_i64(values: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def raw_i64(values: np.ndarray) -> bytes:
    return as_i64(values).astype("<i8", copy=False).tobytes()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def zigzag64(value: int) -> int:
    value = int(value)
    return ((value << 1) ^ (value >> 63)) & MASK64


def unzigzag64(value: int) -> int:
    value = int(value) & MASK64
    return (value >> 1) ^ -(value & 1)


def append_uvarint(out: bytearray, value: int) -> None:
    value = int(value)
    if value < 0:
        raise PhysicalCodecError("uvarint cannot encode a negative value")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def read_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift <= 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
    raise PhysicalCodecError("truncated or oversized uvarint")


def append_svarint(out: bytearray, value: int) -> None:
    append_uvarint(out, zigzag64(int(value)))


def read_svarint(data: bytes, offset: int) -> tuple[int, int]:
    value, offset = read_uvarint(data, offset)
    return unzigzag64(value), offset


def read_exact(data: bytes, offset: int, count: int) -> tuple[bytes, int]:
    if offset < 0 or count < 0 or offset + count > len(data):
        raise PhysicalCodecError(f"truncated stream at {offset}, need {count}, size {len(data)}")
    return data[offset : offset + count], offset + count


def split_blocks(values: np.ndarray, block_size: int) -> Iterable[np.ndarray]:
    if block_size <= 0:
        yield values
        return
    for start in range(0, values.size, block_size):
        yield values[start : start + block_size]


def _pack_u64_bits(values: np.ndarray, width: int) -> bytes:
    values = np.asarray(values, dtype=np.uint64)
    if width < 0 or width > 64:
        raise PhysicalCodecError(f"invalid bit width {width}")
    if values.size == 0 or width == 0:
        return b""
    shifts = np.arange(width, dtype=np.uint64)
    bits = ((values[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8, copy=False)
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def _unpack_u64_bits(payload: bytes, count: int, width: int) -> np.ndarray:
    if count < 0 or width < 0 or width > 64:
        raise PhysicalCodecError("invalid packed bit dimensions")
    expected = (count * width + 7) // 8
    if len(payload) != expected:
        raise PhysicalCodecError(f"packed bit length mismatch: {len(payload)} != {expected}")
    if count == 0 or width == 0:
        return np.zeros(count, dtype=np.uint64)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[: count * width]
    bit_matrix = bits.reshape(count, width).astype(np.uint64, copy=False)
    weights = np.left_shift(np.uint64(1), np.arange(width, dtype=np.uint64))
    return np.sum(bit_matrix * weights[None, :], axis=1, dtype=np.uint64)


def _metadata_i64_list(values: Sequence[int]) -> bytes:
    out = bytearray()
    append_uvarint(out, len(values))
    for value in values:
        append_svarint(out, int(value))
    return bytes(out)


def _read_metadata_i64_list(data: bytes, offset: int = 0) -> tuple[list[int], int]:
    count, offset = read_uvarint(data, offset)
    values: list[int] = []
    for _ in range(count):
        value, offset = read_svarint(data, offset)
        values.append(value)
    return values, offset


def apply_delta_global(values: np.ndarray) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    if values.size == 0:
        return values.copy(), b""
    return np.diff(values).astype(np.int64, copy=False), I64.pack(int(values[0]))


def invert_delta_global(values: np.ndarray, metadata: bytes) -> np.ndarray:
    if not metadata:
        if values.size:
            raise PhysicalCodecError("delta metadata missing")
        return np.asarray([], dtype=np.int64)
    if len(metadata) != I64.size:
        raise PhysicalCodecError("delta metadata length")
    first = I64.unpack(metadata)[0]
    return np.concatenate((np.asarray([first], dtype=np.int64), first + np.cumsum(values, dtype=np.int64)))


def apply_delta_block(values: np.ndarray, block_size: int) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    outputs: list[np.ndarray] = []
    firsts: list[int] = []
    lengths: list[int] = []
    for block in split_blocks(values, block_size):
        lengths.append(int(block.size))
        if block.size:
            firsts.append(int(block[0]))
            outputs.append(np.diff(block).astype(np.int64, copy=False))
    meta = bytearray()
    append_uvarint(meta, block_size)
    meta.extend(_metadata_i64_list(lengths))
    meta.extend(_metadata_i64_list(firsts))
    return (np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)), bytes(meta)


def invert_delta_block(values: np.ndarray, metadata: bytes) -> np.ndarray:
    block_size, offset = read_uvarint(metadata, 0)
    lengths, offset = _read_metadata_i64_list(metadata, offset)
    firsts, offset = _read_metadata_i64_list(metadata, offset)
    if offset != len(metadata) or len(lengths) != len(firsts) or any(length < 0 or length > block_size for length in lengths):
        raise PhysicalCodecError("invalid block-delta metadata")
    outputs: list[np.ndarray] = []
    position = 0
    for length, first in zip(lengths, firsts):
        diff_count = max(0, int(length) - 1)
        diffs = values[position : position + diff_count]
        if diffs.size != diff_count:
            raise PhysicalCodecError("truncated block-delta values")
        outputs.append(np.concatenate((np.asarray([first], dtype=np.int64), first + np.cumsum(diffs, dtype=np.int64))))
        position += diff_count
    if position != values.size:
        raise PhysicalCodecError("trailing block-delta values")
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def apply_delta2_block(values: np.ndarray, block_size: int) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    outputs: list[np.ndarray] = []
    lengths: list[int] = []
    firsts: list[int] = []
    first_deltas: list[int] = []
    for block in split_blocks(values, block_size):
        lengths.append(int(block.size))
        firsts.append(int(block[0]) if block.size else 0)
        if block.size >= 2:
            deltas = np.diff(block).astype(np.int64, copy=False)
            first_deltas.append(int(deltas[0]))
            outputs.append(np.diff(deltas).astype(np.int64, copy=False))
        else:
            first_deltas.append(0)
    meta = bytearray()
    append_uvarint(meta, block_size)
    meta.extend(_metadata_i64_list(lengths))
    meta.extend(_metadata_i64_list(firsts))
    meta.extend(_metadata_i64_list(first_deltas))
    return (np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)), bytes(meta)


def invert_delta2_block(values: np.ndarray, metadata: bytes) -> np.ndarray:
    block_size, offset = read_uvarint(metadata, 0)
    lengths, offset = _read_metadata_i64_list(metadata, offset)
    firsts, offset = _read_metadata_i64_list(metadata, offset)
    first_deltas, offset = _read_metadata_i64_list(metadata, offset)
    if offset != len(metadata) or not (len(lengths) == len(firsts) == len(first_deltas)):
        raise PhysicalCodecError("invalid delta2 metadata")
    outputs: list[np.ndarray] = []
    position = 0
    for length, first, first_delta in zip(lengths, firsts, first_deltas):
        length = int(length)
        if length < 0 or length > block_size:
            raise PhysicalCodecError("invalid delta2 block length")
        if length == 0:
            continue
        if length == 1:
            outputs.append(np.asarray([first], dtype=np.int64))
            continue
        dd_count = length - 2
        dd = values[position : position + dd_count]
        if dd.size != dd_count:
            raise PhysicalCodecError("truncated delta2 payload")
        deltas = np.concatenate((np.asarray([first_delta], dtype=np.int64), first_delta + np.cumsum(dd, dtype=np.int64)))
        outputs.append(np.concatenate((np.asarray([first], dtype=np.int64), first + np.cumsum(deltas, dtype=np.int64))))
        position += dd_count
    if position != values.size:
        raise PhysicalCodecError("trailing delta2 payload")
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def apply_for_block(values: np.ndarray, block_size: int) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    outputs: list[np.ndarray] = []
    bases: list[int] = []
    lengths: list[int] = []
    for block in split_blocks(values, block_size):
        base = int(block.min()) if block.size else 0
        bases.append(base)
        lengths.append(int(block.size))
        outputs.append((block - base).astype(np.int64, copy=False))
    meta = bytearray()
    append_uvarint(meta, block_size)
    meta.extend(_metadata_i64_list(lengths))
    meta.extend(_metadata_i64_list(bases))
    return (np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)), bytes(meta)


def invert_for_block(values: np.ndarray, metadata: bytes) -> np.ndarray:
    block_size, offset = read_uvarint(metadata, 0)
    lengths, offset = _read_metadata_i64_list(metadata, offset)
    bases, offset = _read_metadata_i64_list(metadata, offset)
    if offset != len(metadata) or len(lengths) != len(bases):
        raise PhysicalCodecError("invalid FOR metadata")
    outputs: list[np.ndarray] = []
    position = 0
    for length, base in zip(lengths, bases):
        length = int(length)
        if length < 0 or length > block_size:
            raise PhysicalCodecError("invalid FOR block length")
        block = values[position : position + length]
        if block.size != length:
            raise PhysicalCodecError("truncated FOR payload")
        outputs.append((block + int(base)).astype(np.int64, copy=False))
        position += length
    if position != values.size:
        raise PhysicalCodecError("trailing FOR payload")
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def apply_xor_global(values: np.ndarray) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    if values.size == 0:
        return values.copy(), b""
    bits = values.astype("<i8", copy=False).view("<u8")
    encoded = np.bitwise_xor(bits[1:], bits[:-1]).view("<i8")
    return encoded, I64.pack(int(values[0]))


def invert_xor_global(values: np.ndarray, metadata: bytes) -> np.ndarray:
    if not metadata:
        if values.size:
            raise PhysicalCodecError("xor metadata missing")
        return np.asarray([], dtype=np.int64)
    if len(metadata) != I64.size:
        raise PhysicalCodecError("xor metadata length")
    first = I64.unpack(metadata)[0]
    restored = np.empty(values.size + 1, dtype=np.uint64)
    restored[0] = np.asarray([first], dtype="<i8").view("<u8")[0]
    encoded = values.astype("<i8", copy=False).view("<u8")
    for index, value in enumerate(encoded, start=1):
        restored[index] = restored[index - 1] ^ value
    return restored.view("<i8")


def apply_outlier_block(values: np.ndarray, block_size: int) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    output = values.copy()
    meta = bytearray()
    append_uvarint(meta, block_size)
    blocks = list(split_blocks(values, block_size))
    append_uvarint(meta, len(blocks))
    base_position = 0
    for block in blocks:
        if block.size:
            q1, q3 = np.percentile(block.astype(np.float64), [25.0, 75.0])
            spread = max(float(q3 - q1), 1.0)
            low = max(-(1 << 63), int(math.floor(q1 - 1.5 * spread)))
            high = min((1 << 63) - 1, int(math.ceil(q3 + 1.5 * spread)))
            mask = (block < low) | (block > high)
            positions = np.nonzero(mask)[0]
            originals = block[mask]
            target = output[base_position : base_position + block.size]
            np.clip(target, low, high, out=target)
        else:
            low = high = 0
            positions = np.asarray([], dtype=np.int64)
            originals = np.asarray([], dtype=np.int64)
        append_uvarint(meta, int(block.size))
        append_svarint(meta, low)
        append_svarint(meta, high)
        append_uvarint(meta, int(positions.size))
        previous = -1
        for position, original in zip(positions, originals):
            append_uvarint(meta, int(position) - previous - 1)
            append_svarint(meta, int(original))
            previous = int(position)
        base_position += block.size
    return output, bytes(meta)


def invert_outlier_block(values: np.ndarray, metadata: bytes) -> np.ndarray:
    block_size, offset = read_uvarint(metadata, 0)
    block_count, offset = read_uvarint(metadata, offset)
    output = as_i64(values).copy()
    base_position = 0
    for _ in range(block_count):
        length, offset = read_uvarint(metadata, offset)
        _low, offset = read_svarint(metadata, offset)
        _high, offset = read_svarint(metadata, offset)
        exception_count, offset = read_uvarint(metadata, offset)
        if length > block_size or base_position + length > output.size:
            raise PhysicalCodecError("invalid outlier block length")
        previous = -1
        for _exception in range(exception_count):
            gap, offset = read_uvarint(metadata, offset)
            original, offset = read_svarint(metadata, offset)
            position = previous + 1 + gap
            if position >= length:
                raise PhysicalCodecError("invalid outlier position")
            output[base_position + position] = original
            previous = position
        base_position += length
    if base_position != output.size or offset != len(metadata):
        raise PhysicalCodecError("invalid outlier metadata boundary")
    return output


def apply_dictionary(values: np.ndarray, dictionary_limit: int) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    if values.size == 0:
        meta = bytearray()
        append_uvarint(meta, dictionary_limit)
        append_uvarint(meta, 0)
        append_uvarint(meta, 0)
        return values.copy(), bytes(meta)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    order = np.lexsort((unique, -counts))
    selected_indices = order[:dictionary_limit]
    dictionary = unique[selected_indices].astype(np.int64, copy=False)
    code_for_unique = np.zeros(unique.size, dtype=np.int64)
    code_for_unique[selected_indices] = np.arange(1, dictionary.size + 1, dtype=np.int64)
    codes = code_for_unique[inverse]
    exception_positions = np.nonzero(codes == 0)[0]
    meta = bytearray()
    append_uvarint(meta, dictionary_limit)
    append_uvarint(meta, int(dictionary.size))
    for value in dictionary:
        append_svarint(meta, int(value))
    append_uvarint(meta, int(exception_positions.size))
    previous = -1
    for position in exception_positions:
        append_uvarint(meta, int(position) - previous - 1)
        append_svarint(meta, int(values[position]))
        previous = int(position)
    return codes.astype(np.int64, copy=False), bytes(meta)


def invert_dictionary(values: np.ndarray, metadata: bytes) -> np.ndarray:
    _limit, offset = read_uvarint(metadata, 0)
    dictionary_size, offset = read_uvarint(metadata, offset)
    dictionary: list[int] = []
    for _ in range(dictionary_size):
        value, offset = read_svarint(metadata, offset)
        dictionary.append(value)
    codes = as_i64(values)
    output = np.empty(codes.size, dtype=np.int64)
    for index, code in enumerate(codes):
        code = int(code)
        if code < 0 or code > dictionary_size:
            raise PhysicalCodecError("dictionary code out of range")
        output[index] = 0 if code == 0 else dictionary[code - 1]
    exception_count, offset = read_uvarint(metadata, offset)
    previous = -1
    for _ in range(exception_count):
        gap, offset = read_uvarint(metadata, offset)
        value, offset = read_svarint(metadata, offset)
        position = previous + 1 + gap
        if position >= output.size or int(codes[position]) != 0:
            raise PhysicalCodecError("invalid dictionary exception")
        output[position] = value
        previous = position
    if offset != len(metadata) or int(np.sum(codes == 0)) != exception_count:
        raise PhysicalCodecError("dictionary exception count mismatch")
    return output


def apply_rle_transform(values: np.ndarray) -> tuple[np.ndarray, bytes]:
    values = as_i64(values)
    output: list[int] = []
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[end] == values[position]:
            end += 1
        output.extend((end - position, int(values[position])))
        position = end
    meta = bytearray()
    append_uvarint(meta, int(values.size))
    return as_i64(output), bytes(meta)


def invert_rle_transform(values: np.ndarray, metadata: bytes) -> np.ndarray:
    expected, offset = read_uvarint(metadata, 0)
    if offset != len(metadata) or values.size % 2:
        raise PhysicalCodecError("invalid RLE transform metadata or payload")
    outputs: list[np.ndarray] = []
    total = 0
    for index in range(0, values.size, 2):
        run = int(values[index])
        if run <= 0:
            raise PhysicalCodecError("invalid RLE run length")
        outputs.append(np.full(run, int(values[index + 1]), dtype=np.int64))
        total += run
    if total != expected:
        raise PhysicalCodecError("RLE transformed length mismatch")
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def apply_transform(values: np.ndarray, spec: ActionSpec) -> tuple[np.ndarray, bytes]:
    if spec.name == "delta.parquet_prev.global":
        return apply_delta_global(values)
    if spec.name.startswith("delta.tsfile_prev.block"):
        return apply_delta_block(values, spec.block_size)
    if spec.name.startswith("delta2.tsfile.block"):
        return apply_delta2_block(values, spec.block_size)
    if spec.name.startswith("for.bos.block"):
        return apply_for_block(values, spec.block_size)
    if spec.name == "xor.gorilla.prev":
        return apply_xor_global(values)
    if spec.name.startswith("outlier.bos.iqr.block"):
        return apply_outlier_block(values, spec.block_size)
    if spec.name.startswith("dictionary.parquet.freq"):
        return apply_dictionary(values, spec.dictionary_size)
    if spec.name == "rle.parquet.runs":
        return apply_rle_transform(values)
    raise PhysicalCodecError(f"unknown transform {spec.name}")


def invert_transform(values: np.ndarray, spec: ActionSpec, metadata: bytes) -> np.ndarray:
    if spec.name == "delta.parquet_prev.global":
        return invert_delta_global(values, metadata)
    if spec.name.startswith("delta.tsfile_prev.block"):
        return invert_delta_block(values, metadata)
    if spec.name.startswith("delta2.tsfile.block"):
        return invert_delta2_block(values, metadata)
    if spec.name.startswith("for.bos.block"):
        return invert_for_block(values, metadata)
    if spec.name == "xor.gorilla.prev":
        return invert_xor_global(values, metadata)
    if spec.name.startswith("outlier.bos.iqr.block"):
        return invert_outlier_block(values, metadata)
    if spec.name.startswith("dictionary.parquet.freq"):
        return invert_dictionary(values, metadata)
    if spec.name == "rle.parquet.runs":
        return invert_rle_transform(values, metadata)
    raise PhysicalCodecError(f"unknown inverse transform {spec.name}")


def apply_transform_pipeline(values: np.ndarray, names: Sequence[str]) -> TransformState:
    state = TransformState(as_i64(values).copy(), ())
    seen: set[str] = set()
    for name in names:
        spec = ACTION_BY_NAME.get(name)
        if spec is None or spec.phase != "transform":
            raise PhysicalCodecError(f"not a transform action: {name}")
        if name in seen:
            raise PhysicalCodecError(f"repeated exact transform is not legal: {name}")
        transformed, metadata = apply_transform(state.values, spec)
        state = TransformState(as_i64(transformed), state.records + (TransformRecord(spec, metadata),))
        seen.add(name)
    return state


def pack_raw(values: np.ndarray, _spec: ActionSpec) -> bytes:
    return raw_i64(values)


def unpack_raw(payload: bytes, count: int, _spec: ActionSpec) -> np.ndarray:
    if len(payload) != count * 8:
        raise PhysicalCodecError("raw i64 payload length mismatch")
    return np.frombuffer(payload, dtype="<i8", count=count).astype(np.int64, copy=True)


def pack_varint(values: np.ndarray, _spec: ActionSpec) -> bytes:
    out = bytearray()
    for value in as_i64(values):
        append_uvarint(out, zigzag64(int(value)))
    return bytes(out)


def unpack_varint(payload: bytes, count: int, _spec: ActionSpec) -> np.ndarray:
    values: list[int] = []
    offset = 0
    for _ in range(count):
        value, offset = read_uvarint(payload, offset)
        values.append(unzigzag64(value))
    if offset != len(payload):
        raise PhysicalCodecError("trailing varint payload")
    return as_i64(values)


def pack_bitpack(values: np.ndarray, spec: ActionSpec) -> bytes:
    values = as_i64(values)
    block_size = spec.block_size or max(1, int(values.size))
    blocks = list(split_blocks(values, block_size))
    out = bytearray()
    append_uvarint(out, block_size)
    append_uvarint(out, len(blocks))
    for block in blocks:
        base = int(block.min()) if block.size else 0
        residuals = np.asarray([int(value) - base for value in block], dtype=np.uint64)
        width = max((int(value).bit_length() for value in residuals), default=0)
        packed = _pack_u64_bits(residuals, width)
        append_uvarint(out, int(block.size))
        append_svarint(out, base)
        out.append(width)
        append_uvarint(out, len(packed))
        out.extend(packed)
    return bytes(out)


def unpack_bitpack(payload: bytes, count: int, _spec: ActionSpec) -> np.ndarray:
    block_size, offset = read_uvarint(payload, 0)
    block_count, offset = read_uvarint(payload, offset)
    outputs: list[np.ndarray] = []
    total = 0
    for _ in range(block_count):
        length, offset = read_uvarint(payload, offset)
        base, offset = read_svarint(payload, offset)
        width_raw, offset = read_exact(payload, offset, 1)
        width = width_raw[0]
        packed_size, offset = read_uvarint(payload, offset)
        packed, offset = read_exact(payload, offset, packed_size)
        if length > block_size:
            raise PhysicalCodecError("invalid bitpack block length")
        residuals = _unpack_u64_bits(packed, length, width)
        outputs.append(np.asarray([int(value) + base for value in residuals], dtype=np.int64))
        total += length
    if offset != len(payload) or total != count:
        raise PhysicalCodecError("bitpack payload boundary mismatch")
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def pack_pfor(values: np.ndarray, spec: ActionSpec) -> bytes:
    values = as_i64(values)
    percentile = 0.90 if ".p90" in spec.name else 0.95
    blocks = list(split_blocks(values, spec.block_size))
    out = bytearray()
    append_uvarint(out, spec.block_size)
    append_uvarint(out, len(blocks))
    for block in blocks:
        base = int(block.min()) if block.size else 0
        residuals = np.asarray([int(value) - base for value in block], dtype=np.uint64)
        widths = np.asarray([int(value).bit_length() for value in residuals], dtype=np.int16)
        width = int(np.quantile(widths, percentile, method="higher")) if widths.size else 0
        width = max(0, min(64, width))
        mask = MASK64 if width == 64 else ((1 << width) - 1 if width else 0)
        lows = np.asarray([int(value) & mask for value in residuals], dtype=np.uint64)
        exception_positions = np.nonzero(widths > width)[0]
        packed = _pack_u64_bits(lows, width)
        append_uvarint(out, int(block.size))
        append_svarint(out, base)
        out.append(width)
        append_uvarint(out, len(packed))
        out.extend(packed)
        append_uvarint(out, int(exception_positions.size))
        previous = -1
        for position in exception_positions:
            append_uvarint(out, int(position) - previous - 1)
            append_uvarint(out, int(residuals[position]) >> width if width < 64 else 0)
            previous = int(position)
    return bytes(out)


def unpack_pfor(payload: bytes, count: int, _spec: ActionSpec) -> np.ndarray:
    block_size, offset = read_uvarint(payload, 0)
    block_count, offset = read_uvarint(payload, offset)
    outputs: list[np.ndarray] = []
    total = 0
    for _ in range(block_count):
        length, offset = read_uvarint(payload, offset)
        base, offset = read_svarint(payload, offset)
        width_raw, offset = read_exact(payload, offset, 1)
        width = width_raw[0]
        packed_size, offset = read_uvarint(payload, offset)
        packed, offset = read_exact(payload, offset, packed_size)
        if length > block_size or width > 64:
            raise PhysicalCodecError("invalid PFOR block")
        residuals = _unpack_u64_bits(packed, length, width)
        exception_count, offset = read_uvarint(payload, offset)
        previous = -1
        for _exception in range(exception_count):
            gap, offset = read_uvarint(payload, offset)
            high, offset = read_uvarint(payload, offset)
            position = previous + 1 + gap
            if position >= length:
                raise PhysicalCodecError("invalid PFOR exception position")
            residuals[position] |= np.uint64(high << width)
            previous = position
        outputs.append(np.asarray([int(value) + base for value in residuals], dtype=np.int64))
        total += length
    if offset != len(payload) or total != count:
        raise PhysicalCodecError("PFOR payload boundary mismatch")
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def _project_pack(values: np.ndarray, spec: ActionSpec) -> bytes:
    sequence = [int(value) for value in as_i64(values)]
    if not sequence:
        return b""
    if spec.name == "tsfile.delta128":
        from backend.myapp.tsfile_encoding.p02 import encode_ts_2diff_int64

        return encode_ts_2diff_int64(sequence)
    if spec.name == "tsfile.gorilla64":
        from backend.myapp.tsfile_encoding.p03_i64 import encode_gorilla_int64

        return encode_gorilla_int64(sequence)
    if spec.name == "tsfile.chimp128":
        from backend.myapp.tsfile_encoding.p03_i64 import encode_chimp_int64

        return encode_chimp_int64(sequence)
    if spec.name == "tsfile.sprintz":
        from backend.myapp.tsfile_encoding.p04_i64 import encode_sprintz_int64

        return encode_sprintz_int64(sequence)
    if spec.name == "streamvbyte.delta1.limb32":
        from backend.myapp.int64_column_codec import stream_vbyte_encode_int64

        return stream_vbyte_encode_int64(sequence)
    if spec.name == "simple8b.limb32":
        from backend.myapp.int64_column_codec import simple8b_encode_int64

        return simple8b_encode_int64(sequence)
    raise PhysicalCodecError(f"unknown project packer {spec.name}")


def _project_unpack(payload: bytes, count: int, spec: ActionSpec) -> np.ndarray:
    if count == 0:
        if payload:
            raise PhysicalCodecError("non-empty project payload for empty column")
        return np.asarray([], dtype=np.int64)
    if spec.name == "tsfile.delta128":
        from backend.myapp.tsfile_encoding.p02 import decode_ts_2diff_int64

        values = decode_ts_2diff_int64(payload)
    elif spec.name == "tsfile.gorilla64":
        from backend.myapp.tsfile_encoding.p03_i64 import decode_gorilla_int64

        values = decode_gorilla_int64(payload)
    elif spec.name == "tsfile.chimp128":
        from backend.myapp.tsfile_encoding.p03_i64 import decode_chimp_int64

        values = decode_chimp_int64(payload)
    elif spec.name == "tsfile.sprintz":
        from backend.myapp.tsfile_encoding.p04_i64 import decode_sprintz_int64

        values = decode_sprintz_int64(payload)
    elif spec.name == "streamvbyte.delta1.limb32":
        from backend.myapp.int64_column_codec import stream_vbyte_decode_int64

        values = stream_vbyte_decode_int64(payload, count)
    elif spec.name == "simple8b.limb32":
        from backend.myapp.int64_column_codec import simple8b_decode_int64

        values = simple8b_decode_int64(payload, count)
    else:
        raise PhysicalCodecError(f"unknown project unpacker {spec.name}")
    if len(values) != count:
        raise PhysicalCodecError(f"project packer count mismatch: {len(values)} != {count}")
    return as_i64(values)


def pack_values(values: np.ndarray, spec: ActionSpec) -> bytes:
    if spec.catalog_method:
        from backend.myapp.catalog_action_codecs import pack_specialist_int64

        return pack_specialist_int64([int(value) for value in as_i64(values)], spec.catalog_method)
    if spec.name == "raw.i64":
        return pack_raw(values, spec)
    if spec.name == "varint.parquet.zigzag":
        return pack_varint(values, spec)
    if spec.name.startswith("bitpack.dbp"):
        return pack_bitpack(values, spec)
    if spec.name.startswith("pfor.prototype"):
        return pack_pfor(values, spec)
    return _project_pack(values, spec)


def unpack_values(payload: bytes, count: int, spec: ActionSpec) -> np.ndarray:
    if spec.catalog_method:
        from backend.myapp.catalog_action_codecs import unpack_specialist_int64

        return as_i64(unpack_specialist_int64(payload, count, spec.catalog_method))
    if spec.name == "raw.i64":
        return unpack_raw(payload, count, spec)
    if spec.name == "varint.parquet.zigzag":
        return unpack_varint(payload, count, spec)
    if spec.name.startswith("bitpack.dbp"):
        return unpack_bitpack(payload, count, spec)
    if spec.name.startswith("pfor.prototype"):
        return unpack_pfor(payload, count, spec)
    return _project_unpack(payload, count, spec)


def _lzma_filters(dictionary_size: int) -> list[dict[str, int]]:
    return [{
        "id": lzma.FILTER_LZMA2,
        "dict_size": int(dictionary_size),
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 64,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }]


def compress_payload(payload: bytes, spec: ActionSpec) -> bytes:
    if spec.codec_key:
        from backend.myapp.catalog_action_codecs import compress_general

        return compress_general(payload, spec.codec_key)
    if spec.name == "none":
        return payload
    if spec.name == "zlib.deflate.level1":
        return zlib.compress(payload, level=1)
    if spec.name == "zlib.deflate.level6":
        return zlib.compress(payload, level=6)
    if spec.name == "zstd.level3":
        return zstandard.ZstdCompressor(level=3, write_checksum=True, write_content_size=True).compress(payload)
    if spec.name == "zstd.level9":
        return zstandard.ZstdCompressor(level=9, write_checksum=True, write_content_size=True).compress(payload)
    if spec.name.startswith("lzma2.xz"):
        return lzma.compress(payload, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, filters=_lzma_filters(spec.dictionary_size))
    if spec.name == "brotli.quality5.window22":
        return brotli.compress(payload, quality=5, lgwin=22)
    if spec.name == "bzip2.level9":
        return bz2.compress(payload, compresslevel=9)
    if spec.name == "lz4.frame.64k":
        return lz4.frame.compress(
            payload,
            compression_level=0,
            block_size=lz4.frame.BLOCKSIZE_MAX64KB,
            block_linked=True,
            content_checksum=True,
            block_checksum=True,
            store_size=True,
        )
    raise PhysicalCodecError(f"unknown compressor {spec.name}")


def decompress_payload(payload: bytes, spec: ActionSpec) -> bytes:
    if spec.codec_key:
        from backend.myapp.catalog_action_codecs import decompress_general

        return decompress_general(payload, spec.codec_key)
    if spec.name == "none":
        return payload
    if spec.name.startswith("zlib.deflate"):
        return zlib.decompress(payload)
    if spec.name.startswith("zstd.level"):
        return zstandard.ZstdDecompressor().decompress(payload)
    if spec.name.startswith("lzma2.xz"):
        return lzma.decompress(payload, format=lzma.FORMAT_XZ)
    if spec.name.startswith("brotli."):
        return brotli.decompress(payload)
    if spec.name.startswith("bzip2."):
        return bz2.decompress(payload)
    if spec.name.startswith("lz4.frame"):
        return lz4.frame.decompress(payload)
    raise PhysicalCodecError(f"unknown decompressor {spec.name}")


def encode_from_packed(
    original_values: np.ndarray,
    state: TransformState,
    packer_name: str,
    compressor_name: str,
    packed: bytes,
) -> EncodedFrame:
    original_values = as_i64(original_values)
    packer = ACTION_BY_NAME.get(packer_name)
    compressor = ACTION_BY_NAME.get(compressor_name)
    if packer is None or packer.phase != "packer":
        raise PhysicalCodecError(f"not a packer: {packer_name}")
    if compressor is None or compressor.phase != "compressor":
        raise PhysicalCodecError(f"not a compressor: {compressor_name}")
    compressed = compress_payload(packed, compressor)
    action_ids = [record.spec.wire_id for record in state.records] + [packer.wire_id, compressor.wire_id]
    action_table = b"".join(U16.pack(wire_id) for wire_id in action_ids)
    metadata_parts = [U32.pack(len(record.metadata)) + record.metadata for record in state.records]
    metadata = b"".join(metadata_parts)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        FLAG_CRC32,
        len(state.records),
        0,
        int(original_values.size),
        int(state.values.size),
        len(metadata),
        len(compressed),
    )
    checksum = U32.pack(zlib.crc32(raw_i64(original_values)) & 0xFFFFFFFF)
    data = header + action_table + metadata + compressed + checksum
    components = {
        "outer_header_bytes": len(header),
        "action_table_bytes": len(action_table),
        "metadata_length_prefix_bytes": U32.size * len(state.records),
        "transform_metadata_bytes": sum(len(record.metadata) for record in state.records),
        "packer_payload_bytes": len(packed),
        "compressed_payload_bytes": len(compressed),
        "checksum_bytes": len(checksum),
        "outer_frame_bytes": len(header) + len(action_table) + len(metadata) + len(checksum),
        "final_bytes": len(data),
    }
    expected = components["outer_frame_bytes"] + components["compressed_payload_bytes"]
    if expected != len(data):
        raise AssertionError((expected, len(data), components))
    pipeline = tuple(record.spec.name for record in state.records) + (packer.name, compressor.name)
    return EncodedFrame(data, components, pipeline)


def encode_from_state(original_values: np.ndarray, state: TransformState, packer_name: str, compressor_name: str) -> EncodedFrame:
    packer = ACTION_BY_NAME.get(packer_name)
    if packer is None or packer.phase != "packer":
        raise PhysicalCodecError(f"not a packer: {packer_name}")
    packed = pack_values(state.values, packer)
    return encode_from_packed(original_values, state, packer_name, compressor_name, packed)


def encode_pipeline(values: np.ndarray, transform_names: Sequence[str], packer_name: str, compressor_name: str) -> EncodedFrame:
    values = as_i64(values)
    return encode_from_state(values, apply_transform_pipeline(values, transform_names), packer_name, compressor_name)


def decode_frame(data: bytes) -> tuple[np.ndarray, tuple[str, ...]]:
    if len(data) < HEADER.size + 2 * U16.size + U32.size:
        raise PhysicalCodecError("frame too short")
    magic, version, flags, transform_count, reserved, original_count, transformed_count, metadata_size, payload_size = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or flags != FLAG_CRC32 or reserved != 0:
        raise PhysicalCodecError("invalid physical frame header")
    offset = HEADER.size
    action_count = transform_count + 2
    action_ids: list[int] = []
    for _ in range(action_count):
        raw, offset = read_exact(data, offset, U16.size)
        action_ids.append(U16.unpack(raw)[0])
    try:
        specs = [ACTION_BY_ID[wire_id] for wire_id in action_ids]
    except KeyError as exc:
        raise PhysicalCodecError(f"unknown wire action ID {exc.args[0]}") from exc
    if any(spec.phase != "transform" for spec in specs[:transform_count]) or specs[-2].phase != "packer" or specs[-1].phase != "compressor":
        raise PhysicalCodecError("action phase mismatch")
    metadata_end = offset + metadata_size
    if metadata_end > len(data):
        raise PhysicalCodecError("truncated metadata section")
    records: list[tuple[ActionSpec, bytes]] = []
    for spec in specs[:transform_count]:
        raw, offset = read_exact(data, offset, U32.size)
        length = U32.unpack(raw)[0]
        metadata, offset = read_exact(data, offset, length)
        records.append((spec, metadata))
    if offset != metadata_end:
        raise PhysicalCodecError("metadata section size mismatch")
    compressed, offset = read_exact(data, offset, payload_size)
    checksum_raw, offset = read_exact(data, offset, U32.size)
    if offset != len(data):
        raise PhysicalCodecError("trailing bytes after checksum")
    packed = decompress_payload(compressed, specs[-1])
    values = unpack_values(packed, transformed_count, specs[-2])
    for spec, metadata in reversed(records):
        values = invert_transform(values, spec, metadata)
    if values.size != original_count:
        raise PhysicalCodecError(f"decoded count mismatch: {values.size} != {original_count}")
    expected_checksum = U32.unpack(checksum_raw)[0]
    actual_checksum = zlib.crc32(raw_i64(values)) & 0xFFFFFFFF
    if actual_checksum != expected_checksum:
        raise PhysicalCodecError("CRC32 mismatch")
    return as_i64(values), tuple(spec.name for spec in specs)


def registry() -> dict[str, object]:
    unavailable_specialists = sorted(method for method, action in SPECIALIST_ACTION_BY_METHOD.items() if action is None)
    return {
        "format": "PV2F",
        "version": VERSION,
        "outer_header_bytes": HEADER.size,
        "mandatory_checksum": "CRC32 over original normalized-int64 bytes",
        "action_semantics": "legacy mixed-phase codec candidates, not the primitive RL action space",
        "rl_search_eligible": False,
        "active_method_policy": active_method_policy_summary(),
        "replacement_runtime": "scripts/primitive_operator_action_runtime.py",
        "max_transforms": 2,
        "actions": [asdict(spec) for spec in ALL_ACTIONS],
        "legacy_decode_only_actions": [
            asdict(spec) for spec in LEGACY_COMPRESSORS + RETIRED_CATALOG_COMPRESSORS
        ],
        "counts": {
            "transforms": len(TRANSFORMS),
            "packers": len(PACKERS),
            "compressors": len(COMPRESSORS),
            "total_actions": len(ALL_ACTIONS),
        },
        "catalog_coverage": {
            "deduplication_rule": "same action name and same measured compression ratio",
            "specialist_methods_total": len(SPECIALIST_ACTION_BY_METHOD),
            "specialist_methods_executable": len(SPECIALIST_ACTION_BY_METHOD) - len(unavailable_specialists),
            "general_methods_total": len(GENERAL_ACTION_BY_METHOD),
            "general_methods_executable": len(GENERAL_ACTION_BY_METHOD),
            "method_implementations_total": len(SPECIALIST_ACTION_BY_METHOD) + len(GENERAL_ACTION_BY_METHOD),
            "method_implementations_executable": len(SPECIALIST_ACTION_BY_METHOD) + len(GENERAL_ACTION_BY_METHOD) - len(unavailable_specialists),
            "unavailable_methods": unavailable_specialists,
            "specialist_action_by_method": SPECIALIST_ACTION_BY_METHOD,
            "general_action_by_method": GENERAL_ACTION_BY_METHOD,
            "general_best_level_key_by_method": dict(GENERAL_BEST_KEYS),
            "specialist_best_level_key_by_method": {"SPDP": "spdp_level_9"},
            "level_selection": {
                "general": "modal minimum-byte winner over 448 verified specialist payload pairs from the 20-column benchmark",
                "SPDP": "level 9: minimum aggregate bytes and 15/20 per-column wins on the same normalized-int64 columns",
            },
        },
    }


def self_check(repo_root: Path | None = None) -> dict[str, object]:
    if repo_root is not None:
        repo_text = str(repo_root.resolve())
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
    samples = (
        np.asarray([], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([-7, -7, -3, 0, 1, 1, 1, 18, 18, 99], dtype=np.int64),
        np.arange(-257, 259, dtype=np.int64),
        np.asarray([0, 1, -1, 2**31 - 1, -(2**31), 2**40, -(2**40)], dtype=np.int64),
    )
    checked = 0
    packer_checks = 0
    compressor_checks = 0
    transform_checks = 0

    # Check each packer against every contract sample without multiplying by
    # every compressor. Packer and compressor stages meet at an opaque byte API.
    for sample in samples:
        for packer in PACKERS:
            try:
                packed = pack_values(sample, packer)
                unpacked = unpack_values(packed, sample.size, packer)
            except Exception as exc:
                raise AssertionError(f"packer self-check failed {packer.name}: {exc}") from exc
            if not np.array_equal(unpacked, sample):
                raise AssertionError(f"packer mismatch {packer.name}")
            frame = encode_pipeline(sample, (), packer.name, "none")
            restored, pipeline = decode_frame(frame.data)
            if pipeline != frame.pipeline or not np.array_equal(restored, sample):
                raise AssertionError(f"packer frame mismatch {frame.pipeline}")
            packer_checks += 1
            checked += 1

    catalog_edge_sample = np.asarray(
        [-(2**63), -(2**40), -1, 0, 1, 2**31 - 1, 2**40, 2**63 - 1],
        dtype=np.int64,
    )
    for packer in CATALOG_PACKERS:
        packed = pack_values(catalog_edge_sample, packer)
        unpacked = unpack_values(packed, catalog_edge_sample.size, packer)
        if not np.array_equal(unpacked, catalog_edge_sample):
            raise AssertionError(f"catalog edge-case packer mismatch {packer.name}")
        packer_checks += 1
        checked += 1

    transform_sample = samples[-1]
    transform_pipelines: list[tuple[str, ...]] = [(spec.name,) for spec in TRANSFORMS]
    transform_pipelines.extend(
        (TRANSFORMS[index].name, TRANSFORMS[(index + 5) % len(TRANSFORMS)].name)
        for index in range(len(TRANSFORMS))
        if TRANSFORMS[index].name != TRANSFORMS[(index + 5) % len(TRANSFORMS)].name
    )
    for transforms in transform_pipelines:
        frame = encode_pipeline(transform_sample, transforms, "raw.i64", "none")
        restored, pipeline = decode_frame(frame.data)
        if pipeline != frame.pipeline or not np.array_equal(restored, transform_sample):
            raise AssertionError(f"transform frame mismatch {frame.pipeline}")
        transform_checks += 1
        checked += 1

    compressor_sample = np.resize(np.asarray([-9, -1, 0, 1, 7, 7, 1024], dtype=np.int64), 64)
    for compressor in COMPRESSORS:
        frame = encode_pipeline(compressor_sample, (), "raw.i64", compressor.name)
        restored, pipeline = decode_frame(frame.data)
        if pipeline != frame.pipeline or not np.array_equal(restored, compressor_sample):
            raise AssertionError(f"compressor frame mismatch {frame.pipeline}")
        if frame.components["outer_frame_bytes"] + frame.components["compressed_payload_bytes"] != frame.components["final_bytes"]:
            raise AssertionError("component byte sum mismatch")
        compressor_checks += 1
        checked += 1
    probe = encode_pipeline(np.arange(16, dtype=np.int64), (), "raw.i64", "none")
    corrupted = bytearray(probe.data)
    corrupted[-1] ^= 1
    try:
        decode_frame(bytes(corrupted))
    except PhysicalCodecError:
        corruption_detected = True
    else:
        corruption_detected = False
    if not corruption_detected:
        raise AssertionError("checksum corruption was not detected")
    return {
        "status": "PASS",
        "round_trips": checked,
        "packer_round_trips": packer_checks,
        "compressor_round_trips": compressor_checks,
        "transform_round_trips": transform_checks,
        "corruption_detected": corruption_detected,
        "registry_sha256": sha256_bytes(json.dumps(registry(), sort_keys=True).encode("utf-8")),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(self_check(root), indent=2, sort_keys=True))
