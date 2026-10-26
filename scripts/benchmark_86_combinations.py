#!/usr/bin/env python3
"""Screen paper-catalog specialist + general lossless codec compositions.

The experiment is deliberately stricter than the legacy aggregate benchmark:
both stages must materialize bytes and pass an end-to-end round trip.  Method
families with several tested levels/presets keep the smallest verified stream.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import json
import lzma
import os
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
os.environ.setdefault("COMPRESSION_BENCHMARK_NATIVE_LZ_FAST_BYTES", "0")

from myapp.entropy_cpp import ans_decode, ans_encode  # noqa: E402
from myapp.entropy_cpp import arithmetic_decode, arithmetic_encode  # noqa: E402
from myapp.catalog_action_codecs import _pongo_decode, _pongo_encode  # noqa: E402
from myapp.camel_python import CamelCompressor, CamelDecompressor  # noqa: E402
from myapp.experiment_data_policy import NAN_INPUT_POLICY, prepare_numeric_column  # noqa: E402
from myapp.column_encodings import (  # noqa: E402
    decode_delta_zigzag_varint_int,
    decode_dictionary_float,
    decode_dictionary_int,
    encode_delta_zigzag_varint_int,
    encode_dictionary_float,
    encode_dictionary_int,
)
from myapp.elf32_python import (  # noqa: E402
    ElfCompressor32,
    ElfDecompressor32,
    ElfOnChimpCompressor32,
    ElfOnChimpDecompressor32,
)
from myapp.entropy_cpp import huffman_decode, huffman_encode  # noqa: E402
from myapp.int64_column_codec import (  # noqa: E402
    fast_pfor_decode_int64,
    fast_pfor_encode_int64,
    simple8b_decode_int64,
    simple8b_encode_int64,
    stream_vbyte_decode_int64,
    stream_vbyte_encode_int64,
)
from myapp.iotdb_style_python import (  # noqa: E402
    _decode_long_delta_bitpack,
    _encode_long_delta_bitpack,
)
from myapp.method_catalog import algorithm_keys_for_method_name  # noqa: E402
from myapp.native_lz_family import (  # noqa: E402
    compress_native_lz_family,
    decompress_native_lz_family,
)
from myapp.reger_codec import (  # noqa: E402
    decode_float_column as reger_decode_float,
    decode_int_column as reger_decode_int,
    encode_float_column as reger_encode_float,
    encode_int_column as reger_encode_int,
)
from myapp.research_codec_python import (  # noqa: E402
    _alp_d64_best_fe,
    _alp_d64_dec_val,
    _alp_d64_enc_val,
    _double_bits,
    scale_float_column_to_ints,
    scale_float_column_to_ints_with_max_point,
)
from myapp.research_extended_codecs import (  # noqa: E402
    _fpc_compress_memory,
    _fpc_decompress_memory,
    _pfpc_compress_memory,
    _pfpc_decompress_memory,
    _spdp_decompress_core,
    _spdp_roundtrip,
)
from myapp.tsfile_encoding.p02 import (  # noqa: E402
    decode_rle_double,
    decode_rle_int64,
    decode_ts_2diff_double,
    decode_ts_2diff_int64,
    encode_rle_double,
    encode_rle_int64,
    encode_ts_2diff_double,
    encode_ts_2diff_int64,
)
from myapp.tsfile_encoding.p03_i64 import (  # noqa: E402
    decode_chimp_double,
    decode_chimp_int64,
    decode_gorilla_double,
    decode_gorilla_int64,
    encode_chimp_double,
    encode_chimp_int64,
    encode_gorilla_double,
    encode_gorilla_int64,
)
from myapp.tsfile_encoding.p04_i64 import (  # noqa: E402
    decode_rlbe_double,
    decode_rlbe_int64,
    decode_sprintz_double,
    decode_sprintz_int64,
    encode_rlbe_double,
    encode_rlbe_int64,
    encode_sprintz_double,
    encode_sprintz_int64,
)


SPECIALIST_METHODS = (
    "Chimp", "Chimp128", "Pongo", "Gorilla", "2DIFF", "BOS", "REGER",
    "SVB", "FastPFor", "Simple8b", "RLE", "RLBE", "Sprintz", "Dict.",
    "DBP", "ALP", "LECO", "Camel", "Elf", "Elf+", "FPC", "PFPC",
    "SPDP", "CORAD", "BUFF", "FPZIP", "FPCmp-R", "FPCmp-S",
)

GENERAL_METHODS = (
    "gzip", "Brotli", "Zstd", "LZ4", "XZ", "Density", "Gipfeli", "GLZA",
    "LZMAT", "LZO1", "LZO2A", "LZRW", "LZSSE", "LZVN", "Pithy",
    "Shrinker", "NRV", "WFLZ", "XPACK", "YALZ77", "Yappy", "Snappy",
    "ANS", "FSE", "Huffman", "Huff0", "Arithmetic", "Blosc", "Blosc2",
    "BriefLZ", "BSC", "Bzip2", "Bzip3", "Crush", "CSC", "FastLZ",
    "FastLZMA2", "Kanzi", "Libdeflate", "Lizard", "LZ4 Fast", "LZ4HC",
    "LZAV", "LZF", "LZFSE", "LZG", "LZHAM", "LZJB", "Lzlib", "PPMD8",
    "QuickLZ", "SLZ", "Tornado", "Zlib", "Zlib-ng", "Zling", "ZXC", "TAMP",
)

CATALOG_NAMES = {
    "Brotli": "brotli", "Zstd": "zstd", "Snappy": "snappy", "ANS": "ANS",
    "Huffman": "Huffman", "Arithmetic": "Arithmetic", "NRV": "UCL NRV",
    "Libdeflate": "libdeflate", "Zlib": "zlib", "Zlib-ng": "zlib-ng",
}
BENCHMARK_CONTRACT = "lossless64_drop_nan_v2_20260905"
IGNORED_SPECIALIST_METHODS = {"Camel": "excluded by user pending lossless Camel implementation"}


@dataclass(frozen=True)
class Column:
    dataset: str
    index: int
    name: str
    logical_type: str
    max_point: int
    values: tuple[int | float, ...]
    source_count: int = field(init=False)
    removed_nan_count: int = field(init=False)

    def __post_init__(self) -> None:
        # Apply at the common Column boundary, including directly constructed
        # experiment columns, before any codec computes bytes or a denominator.
        prepared = prepare_numeric_column(self.values, self.logical_type)
        object.__setattr__(self, "values", prepared.values)
        object.__setattr__(self, "source_count", prepared.source_count)
        object.__setattr__(self, "removed_nan_count", prepared.removed_nan_count)

    @property
    def raw(self) -> bytes:
        dtype = "<i8" if self.logical_type == "integer" else "<f8"
        return np.asarray(self.values, dtype=dtype).tobytes(order="C")


@dataclass
class Artifact:
    method: str
    payload: bytes
    implementation: str
    decode: Callable[[bytes], Sequence[int | float] | bytes]
    note: str = ""


def _bits(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values)


def _values_equal(col: Column, got: Sequence[int | float] | bytes) -> bool:
    if isinstance(got, (bytes, bytearray)):
        return bytes(got) == col.raw
    if len(got) != len(col.values):
        return False
    if col.logical_type == "integer":
        return tuple(int(x) for x in got) == tuple(int(x) for x in col.values)
    return _bits([float(x) for x in got]) == _bits([float(x) for x in col.values])


def _scale_float(col: Column) -> tuple[int, list[int]] | None:
    vals = [float(v) for v in col.values]
    ints = scale_float_column_to_ints_with_max_point(vals, col.max_point)
    if ints is None:
        ints = scale_float_column_to_ints(vals)
    if ints is None:
        return None
    # Recover the actual scale selected by the helper and require a bitwise inverse.
    for scale in range(max(18, col.max_point), -1, -1):
        power = 10.0 ** scale
        candidate = [int(round(v * power)) for v in vals]
        if candidate != ints:
            continue
        recovered = [float(v) / power for v in ints]
        if _bits(recovered) == _bits(vals):
            return scale, ints
    return None


def _scaled_artifact(
    col: Column,
    method: str,
    encode_int: Callable[[Sequence[int]], bytes],
    decode_int: Callable[[bytes], Sequence[int]],
    implementation: str,
) -> Artifact | None:
    if col.logical_type == "integer":
        vals = [int(v) for v in col.values]
        body = encode_int(vals)
        return Artifact(method, b"I" + body, implementation, lambda p: decode_int(p[1:]))
    scaled = _scale_float(col)
    if scaled is None:
        return None
    scale, vals = scaled
    body = encode_int(vals)

    def decode(payload: bytes) -> Sequence[float]:
        s = struct.unpack_from("<i", payload, 1)[0]
        recovered = decode_int(payload[5:])
        power = 10.0 ** s
        return [float(v) / power for v in recovered]

    return Artifact(method, b"F" + struct.pack("<i", scale) + body, implementation, decode)


def _direct_ts_artifact(col: Column, method: str) -> Artifact:
    vals = list(col.values)
    if method in {"Chimp", "Chimp128"}:
        if col.logical_type == "integer":
            body = encode_chimp_int64([int(v) for v in vals])
            dec = decode_chimp_int64
        else:
            body = encode_chimp_double([float(v) for v in vals])
            dec = decode_chimp_double
        impl = "project_tsfile_chimp" if method == "Chimp" else "project_alias_CHIMP128_to_CHIMP"
        note = "当前 Java/Python benchmark 的 Chimp128 路径与 Chimp 相同" if method == "Chimp128" else ""
    elif method == "Gorilla":
        if col.logical_type == "integer":
            body = encode_gorilla_int64([int(v) for v in vals]); dec = decode_gorilla_int64
        else:
            body = encode_gorilla_double([float(v) for v in vals]); dec = decode_gorilla_double
        impl, note = "project_tsfile_gorilla", ""
    elif method == "2DIFF":
        if col.logical_type == "integer":
            body = encode_ts_2diff_int64([int(v) for v in vals]); dec = decode_ts_2diff_int64
        else:
            body = encode_ts_2diff_double([float(v) for v in vals], col.max_point); dec = decode_ts_2diff_double
        impl, note = "project_tsfile_2diff", ""
    elif method == "RLE":
        if col.logical_type == "integer":
            body = encode_rle_int64([int(v) for v in vals]); dec = decode_rle_int64
        else:
            body = encode_rle_double([float(v) for v in vals], col.max_point); dec = decode_rle_double
        impl, note = "project_tsfile_rle", ""
    elif method == "RLBE":
        if col.logical_type == "integer":
            body = encode_rlbe_int64([int(v) for v in vals]); dec = decode_rlbe_int64
        else:
            body = encode_rlbe_double([float(v) for v in vals]); dec = decode_rlbe_double
        impl, note = "project_tsfile_rlbe", ""
    elif method == "Sprintz":
        if col.logical_type == "integer":
            body = encode_sprintz_int64([int(v) for v in vals]); dec = decode_sprintz_int64
        else:
            body = encode_sprintz_double([float(v) for v in vals]); dec = decode_sprintz_double
        impl, note = "project_tsfile_sprintz", ""
    else:
        raise KeyError(method)
    return Artifact(method, body, impl, dec, note)


def _camel_artifact(col: Column) -> Artifact:
    out = bytearray(b"CAM1" + struct.pack("<I", len(col.values)))
    for start in range(0, len(col.values), 1000):
        chunk = [float(v) for v in col.values[start : start + 1000]]
        enc = CamelCompressor()
        for value in chunk:
            enc.add_value(value)
        enc.close()
        body = enc.get_bytes()
        out.extend(struct.pack("<II", len(chunk), len(body)))
        out.extend(body)

    def decode(payload: bytes) -> Sequence[float]:
        n = struct.unpack_from("<I", payload, 4)[0]
        pos, values = 8, []
        while len(values) < n:
            count, size = struct.unpack_from("<II", payload, pos); pos += 8
            dec = CamelDecompressor(payload[pos : pos + size]); pos += size
            values.extend(dec.read_pair() for _ in range(count))
        return values

    return Artifact("Camel", bytes(out), "project_camel_lossless", decode)


def _elf32_legacy_artifact(col: Column, method: str) -> Artifact:
    on_chimp = method == "Elf"
    out = bytearray(b"ELF1" + struct.pack("<I", len(col.values)))
    for start in range(0, len(col.values), 2048):
        chunk = [float(v) for v in col.values[start : start + 2048]]
        enc = ElfOnChimpCompressor32() if on_chimp else ElfCompressor32()
        for value in chunk:
            enc.add_value(value)
        enc.close(); body = enc.get_bytes()
        out.extend(struct.pack("<II", len(chunk), len(body))); out.extend(body)

    def decode(payload: bytes) -> Sequence[float]:
        n = struct.unpack_from("<I", payload, 4)[0]
        pos, values = 8, []
        while len(values) < n:
            count, size = struct.unpack_from("<II", payload, pos); pos += 8
            body = payload[pos : pos + size]; pos += size
            dec = ElfOnChimpDecompressor32(body) if on_chimp else ElfDecompressor32(body)
            values.extend(dec.decompress(count))
        return values

    return Artifact(method, bytes(out), "project_elf32_on_chimp" if on_chimp else "project_elf32_plus", decode)


def _elf_artifact(col: Column, method: str) -> Artifact:
    from myapp import elf64_exact_stages
    return Artifact(method, elf64_exact_stages.encode(col.raw, method),
                    "project_elf64_on_chimp" if method == "Elf" else "project_elf64_plus",
                    lambda payload: elf64_exact_stages.decode(payload, method),
                    "binary64 source encoder; EL64 self-describing count; NaN removed at Column boundary")


def _buff_native_artifact(col: Column) -> Artifact:
    cli = ROOT / "cpp/build/buff_native_exact_stages_cli"

    def invoke(command: str, data: bytes) -> bytes:
        process = subprocess.run([str(cli), command], input=data, capture_output=True, check=True)
        return process.stdout

    return Artifact("BUFF", invoke("original-encode", col.raw), "project_buff_native_raw64",
                    lambda payload: invoke("original-decode", payload),
                    "lossless native raw64 storage; integer lanes preserve their bits; not legacy precision-bounded BUFF")


def _alp_artifact(col: Column) -> Artifact | None:
    if col.logical_type == "integer":
        return _scaled_artifact(col, "ALP", fast_pfor_encode_int64, lambda p: fast_pfor_decode_int64(p, len(col.values)), "project_alp_int64_path")
    out = bytearray(b"ALP2" + struct.pack("<I", len(col.values)))
    vals = [float(v) for v in col.values]
    for start in range(0, len(vals), 1024):
        window = vals[start : start + 1024]
        fe = _alp_d64_best_fe(window, len(window))
        if fe is None:
            raw = np.asarray(window, dtype="<f8").tobytes()
            out.extend(struct.pack("<BHI", 0, len(window), len(raw))); out.extend(raw)
            continue
        fac, exp = fe
        encoded, exc, placeholder = [], [], None
        for pos, value in enumerate(window):
            code = _alp_d64_enc_val(value, fac, exp)
            if _double_bits(_alp_d64_dec_val(code, fac, exp)) == _double_bits(value):
                encoded.append(code)
                if placeholder is None:
                    placeholder = code
            else:
                encoded.append(0); exc.append((pos, _double_bits(value)))
        if placeholder is None:
            raw = np.asarray(window, dtype="<f8").tobytes()
            out.extend(struct.pack("<BHI", 0, len(window), len(raw))); out.extend(raw)
            continue
        for pos, _ in exc:
            encoded[pos] = placeholder
        body = fast_pfor_encode_int64(encoded)
        out.extend(struct.pack("<BHbbIH", 1, len(window), fac, exp, len(body), len(exc)))
        out.extend(body)
        for pos, bits in exc:
            out.extend(struct.pack("<HQ", pos, bits))

    def decode(payload: bytes) -> Sequence[float]:
        n = struct.unpack_from("<I", payload, 4)[0]
        pos, values = 8, []
        while len(values) < n:
            mode = payload[pos]
            if mode == 0:
                _, count, size = struct.unpack_from("<BHI", payload, pos); pos += 7
                values.extend(np.frombuffer(payload[pos : pos + size], dtype="<f8").tolist()); pos += size
            else:
                _, count, fac, exp, size, exc_n = struct.unpack_from("<BHbbIH", payload, pos); pos += 11
                codes = fast_pfor_decode_int64(payload[pos : pos + size], count); pos += size
                chunk = [_alp_d64_dec_val(v, fac, exp) for v in codes]
                for _ in range(exc_n):
                    idx, bits = struct.unpack_from("<HQ", payload, pos); pos += 10
                    chunk[idx] = struct.unpack("<d", struct.pack("<Q", bits))[0]
                values.extend(chunk)
        return values

    return Artifact("ALP", bytes(out), "project_alp_double_materialized", decode)


def specialist_artifact(col: Column, method: str) -> tuple[Artifact | None, str]:
    if method in IGNORED_SPECIALIST_METHODS:
        return None, IGNORED_SPECIALIST_METHODS[method]
    try:
        if method in {"Chimp", "Chimp128", "Gorilla", "2DIFF", "RLE", "RLBE", "Sprintz"}:
            art = _direct_ts_artifact(col, method)
        elif method == "REGER":
            if col.logical_type == "integer":
                art = Artifact(method, reger_encode_int(col.values), "project_reger", reger_decode_int)
            else:
                art = Artifact(method, reger_encode_float(col.values), "project_reger", reger_decode_float)
        elif method == "Dict.":
            if col.logical_type == "integer":
                art = Artifact(method, encode_dictionary_int(list(map(int, col.values))), "project_dictionary", decode_dictionary_int)
            else:
                art = Artifact(method, encode_dictionary_float(list(map(float, col.values))), "project_dictionary", decode_dictionary_float)
        elif method == "DBP":
            art = _scaled_artifact(col, method, encode_delta_zigzag_varint_int, decode_delta_zigzag_varint_int, "project_delta_zigzag_varint")
        elif method == "SVB":
            art = _scaled_artifact(col, method, stream_vbyte_encode_int64, lambda p: stream_vbyte_decode_int64(p, len(col.values)), "project_stream_vbyte_int64")
        elif method in {"FastPFor", "LECO"}:
            art = _scaled_artifact(col, method, fast_pfor_encode_int64, lambda p: fast_pfor_decode_int64(p, len(col.values)), "project_fastpfor_int64")
        elif method == "Simple8b":
            art = _scaled_artifact(col, method, simple8b_encode_int64, lambda p: simple8b_decode_int64(p, len(col.values)), "project_simple8b_int64")
        elif method == "BOS":
            art = _scaled_artifact(col, method, _encode_long_delta_bitpack, _decode_long_delta_bitpack, "project_BOS64_delta_bitpack_payload")
        elif method == "ALP":
            art = _alp_artifact(col)
        elif method == "Camel":
            art = _camel_artifact(col)
        elif method in {"Elf", "Elf+"}:
            if col.logical_type == "integer":
                return None, "Elf/Elf+ accepts floating-point float64 input"
            art = _elf_artifact(col, method)
        elif method == "FPC":
            raw = col.raw; body = _fpc_compress_memory(raw, 16)
            art = Artifact(method, body, "project_FPC", _fpc_decompress_memory)
        elif method == "PFPC":
            raw = col.raw; body = _pfpc_compress_memory(raw, 16, 1, max(1, min(len(raw), 1 << 20)))
            art = Artifact(method, body, "project_PFPC_single_thread", _pfpc_decompress_memory)
        elif method == "SPDP":
            body, _, _, decoded = _spdp_roundtrip(col.raw, 5, 8)
            if decoded != col.raw:
                return None, "SPDP roundtrip mismatch"
            def decode_spdp(payload: bytes) -> bytes:
                out = bytearray(len(col.raw))
                _spdp_decompress_core(5, len(payload), payload, out, 8)
                return bytes(out)
            art = Artifact(method, body, "project_SPDP_level_5", decode_spdp)
        elif method == "FPZIP":
            if col.logical_type == "integer":
                return None, "FPZIP accepts floating-point arrays"
            import fpzip
            arr = np.asarray(col.values, dtype="<f8")
            body = fpzip.compress(arr, precision=0, order="C")
            art = Artifact(method, body, "upstream_fpzip_python_precision_0", lambda p: fpzip.decompress(p, order="C").reshape(-1).tolist())
        elif method in {"FPCmp-R", "FPCmp-S"}:
            key = "fpcompress_dpratio" if method == "FPCmp-R" else "fpcompress_dpspeed"
            body = compress_native_lz_family(col.raw, key)
            art = Artifact(method, body, f"project_native_reference:{key}", lambda p: decompress_native_lz_family(p, key, len(col.raw)))
        elif method == "Pongo":
            if col.logical_type == "integer":
                lanes = [
                    struct.unpack("<d", struct.pack("<q", int(value)))[0]
                    for value in col.values
                ]
                body = _pongo_encode(lanes)

                def decode_pongo_int(payload: bytes) -> Sequence[int]:
                    return [
                        struct.unpack("<q", struct.pack("<d", value))[0]
                        for value in _pongo_decode(payload)
                    ]

                art = Artifact(
                    method, body, "project_pongo_cpp_python_bit_exact",
                    decode_pongo_int,
                )
            else:
                body = _pongo_encode([float(value) for value in col.values])
                art = Artifact(
                    method, body, "project_pongo_cpp_python_bit_exact",
                    _pongo_decode,
                )
        elif method == "CORAD":
            from myapp import corad_lossless_stages
            kind = "int64" if col.logical_type == "integer" else "float64"
            art = Artifact(method, corad_lossless_stages.encode(col.raw, kind),
                           "project_corad_lossless_crd2",
                           lambda payload: corad_lossless_stages.decode(payload, kind),
                           "revised lossless format includes normalization parameters, exact-bit corrections and tails")
        elif method == "BUFF":
            art = _buff_native_artifact(col)
        else:
            return None, "no materialized adapter"
        if art is None:
            return None, "not applicable or exact decimal scaling unavailable"
        recovered = art.decode(art.payload)
        if not _values_equal(col, recovered):
            return None, "specialist roundtrip mismatch under strict 64-bit comparison"
        return art, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _generic_variants(method: str) -> list[str]:
    catalog_name = CATALOG_NAMES.get(method, method)
    keys = algorithm_keys_for_method_name(catalog_name)
    overrides = {
        "ANS": ["ans"], "Huffman": ["huffman"], "Arithmetic": ["arithmetic"],
        "Brotli": ["brotli"], "Zstd": [f"zstd_level_{i}" for i in range(1, 10)],
        "Snappy": ["snappy-framed"], "NRV": algorithm_keys_for_method_name("UCL NRV"),
    }
    return overrides.get(method, keys)


def _compress_generic(data: bytes, key: str) -> tuple[bytes, Callable[[bytes], bytes], str]:
    if key == "ans":
        return ans_encode(data), ans_decode, "project_rANS"
    if key == "huffman":
        return huffman_encode(data), huffman_decode, "project_Huffman"
    if key == "arithmetic":
        return arithmetic_encode(data), arithmetic_decode, "project_range_coder"
    if key == "gz":
        return gzip.compress(data, compresslevel=6, mtime=0), gzip.decompress, "stdlib_gzip_level_6"
    if key == "brotli":
        import brotli
        return brotli.compress(data, quality=6), brotli.decompress, "upstream_brotli_quality_6"
    if key.startswith("zstd_level_"):
        import zstandard as zstd
        level = int(key.rsplit("_", 1)[1]); cctx = zstd.ZstdCompressor(level=level); dctx = zstd.ZstdDecompressor()
        return cctx.compress(data), dctx.decompress, f"upstream_zstd_level_{level}"
    if key.startswith("xz_level_"):
        level = int(key.rsplit("_", 1)[1])
        return lzma.compress(data, format=lzma.FORMAT_XZ, preset=level), lambda p: lzma.decompress(p, format=lzma.FORMAT_XZ), f"stdlib_xz_preset_{level}"
    if key == "lz4-framed":
        import lz4.frame
        return lz4.frame.compress(data), lz4.frame.decompress, "upstream_lz4_frame"
    if key == "snappy-framed":
        import snappy
        return snappy.compress(data), snappy.decompress, "upstream_snappy"
    if key.startswith("bzip2_"):
        level = int(key.rsplit("_", 1)[1])
        return bz2.compress(data, compresslevel=level), bz2.decompress, f"stdlib_bzip2_level_{level}"
    if key.startswith("zlib_") and not key.startswith("zlib_ng_"):
        level = int(key.rsplit("_", 1)[1])
        return zlib.compress(data, level), zlib.decompress, f"stdlib_zlib_level_{level}"
    if key == "blosc_blosclz":
        import blosc
        return blosc.compress(data, typesize=1, cname="blosclz", clevel=5), blosc.decompress, "upstream_blosc_blosclz"
    if key == "blosc2_blosclz":
        import blosc2
        return blosc2.compress(data, typesize=1, codec=blosc2.Codec.BLOSCLZ, clevel=5), blosc2.decompress, "upstream_blosc2_blosclz"
    comp = compress_native_lz_family(data, key)
    return comp, lambda p: decompress_native_lz_family(p, key, len(data)), f"project_native_reference:{key}"


def best_generic(data: bytes, method: str) -> tuple[dict | None, list[dict]]:
    attempts: list[dict] = []
    for key in _generic_variants(method):
        started = time.perf_counter()
        try:
            comp, decode, impl = _compress_generic(data, key)
            ok = decode(comp) == data
            attempts.append({"key": key, "bytes": len(comp), "ok": ok, "implementation": impl, "sec": time.perf_counter() - started})
        except Exception as exc:
            attempts.append({"key": key, "bytes": None, "ok": False, "implementation": "error", "error": f"{type(exc).__name__}: {exc}", "sec": time.perf_counter() - started})
    valid = [row for row in attempts if row["ok"]]
    return (min(valid, key=lambda row: (row["bytes"], row["key"])) if valid else None), attempts


def load_columns(dataset: str) -> list[Column]:
    data_path = ROOT / "backend" / "dataset_csv" / f"{dataset}.csv"
    metadata_path = ROOT / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv"
    # Parse numeric strings only after filtering missing/NaN cells. Letting
    # pandas infer float columns first can round large int64 values with holes.
    frame = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    metadata = list(csv.DictReader(metadata_path.open()))
    columns = []
    for row in metadata:
        index = int(row["column_index"]); name = row["column_name"]; logical = row["logical_type"]
        series = frame[name]
        col = Column(dataset, index, name, logical, int(row["max_point_number"]), tuple(series.tolist()))
        if col.values:
            columns.append(col)
    return columns


def run_column(col: Column) -> dict:
    if not col.values:
        raise ValueError("empty/all-NaN experimental columns must be skipped")
    started = time.time()
    specialists, exclusions, pairs = [], [], []
    for specialist in SPECIALIST_METHODS:
        print(f"  SPECIALIST {col.dataset} col={col.index} {specialist}", flush=True)
        art, reason = specialist_artifact(col, specialist)
        if art is None:
            exclusions.append({"method": specialist, "reason": reason})
            print(f"  SKIP {col.dataset} col={col.index} {specialist}: {reason}", flush=True)
            continue
        specialists.append({"method": specialist, "bytes": len(art.payload), "implementation": art.implementation, "note": art.note})
        for general in GENERAL_METHODS:
            best, attempts = best_generic(art.payload, general)
            if best is None:
                pairs.append({"specialist": specialist, "general": general, "ok": False, "attempts": attempts})
                continue
            pairs.append({
                "specialist": specialist, "specialist_bytes": len(art.payload), "specialist_implementation": art.implementation,
                "general": general, "level": best["key"], "general_implementation": best["implementation"],
                "compressed_bytes": best["bytes"], "rate": best["bytes"] / len(col.raw),
                "factor": len(col.raw) / best["bytes"], "ok": True,
                "levels_tested": len(attempts), "levels_verified": sum(bool(x["ok"]) for x in attempts),
            })
        print(f"  MATERIALIZED {col.dataset} col={col.index} {specialist} bytes={len(art.payload)}", flush=True)
    valid_pairs = [p for p in pairs if p.get("ok")]
    valid_pairs.sort(key=lambda p: (p["compressed_bytes"], p["specialist"], p["general"], p["level"]))
    result = {
        "dataset": col.dataset, "column_index": col.index, "column_name": col.name,
        "logical_type": col.logical_type, "max_point": col.max_point, "rows": len(col.values),
        "source_rows": col.source_count, "removed_nan_count": col.removed_nan_count,
        "input_policy": NAN_INPUT_POLICY, "benchmark_contract": BENCHMARK_CONTRACT,
        "original_bytes": len(col.raw), "raw_sha256": hashlib.sha256(col.raw).hexdigest(),
        "specialists_materialized": specialists, "specialist_exclusions": exclusions,
        "pairs": pairs, "best": valid_pairs[0] if valid_pairs else None, "top5": valid_pairs[:5],
        "elapsed_sec": time.time() - started,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Metro-Traffic", "Air-Quality"])
    parser.add_argument("--columns", nargs="*", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "specialist_methods": list(SPECIALIST_METHODS), "general_methods": list(GENERAL_METHODS),
        "paper_codec_count": len(SPECIALIST_METHODS) + len(GENERAL_METHODS),
        "family_pairs_per_column": len(SPECIALIST_METHODS) * len(GENERAL_METHODS),
        "grouping_excluded": True, "level_policy": "minimum verified compressed bytes among project-tested variants",
        "input_policy": NAN_INPUT_POLICY, "benchmark_contract": BENCHMARK_CONTRACT,
        "ignored_specialists": IGNORED_SPECIALIST_METHODS,
        "columns": [],
    }
    for dataset in args.datasets:
        for col in load_columns(dataset):
            if args.columns and col.index not in args.columns:
                continue
            print(f"START {dataset} col={col.index} {col.name}", flush=True)
            result = run_column(col)
            target = output / f"{dataset}__{col.index:03d}.json"
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["columns"].append({k: result[k] for k in ("dataset", "column_index", "column_name", "logical_type", "rows", "original_bytes", "elapsed_sec")})
            manifest["columns"][-1]["best"] = result["best"]
            manifest["columns"][-1]["specialist_count"] = len(result["specialists_materialized"])
            manifest["columns"][-1]["verified_pair_count"] = sum(bool(x.get("ok")) for x in result["pairs"])
            manifest["columns"][-1]["path"] = target.name
            (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"DONE {dataset} col={col.index} specialists={manifest['columns'][-1]['specialist_count']} pairs={manifest['columns'][-1]['verified_pair_count']} best={result['best']}", flush=True)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
