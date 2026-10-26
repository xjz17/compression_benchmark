"""int64 lo/hi limb codecs for double-wire research rows (matches Java Int64ColumnCodec)."""

from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

from .research_extended_codecs import _svb_decode_d1, _svb_encode_d1
from .simple8b_int_python import simple8b_decode_bytes_to_i32_column, simple8b_encode_bytes_from_i32_column
from .native_lz_family import compress_native_lz_family, decompress_native_lz_family


def lo32(v: int) -> int:
    w = int(v) & 0xFFFFFFFF
    return w if w < 0x80000000 else w - 0x100000000


def hi32(v: int) -> int:
    w = (int(v) >> 32) & 0xFFFFFFFF
    return w if w < 0x80000000 else w - 0x100000000


def reconstruct(lo: int, hi: int) -> int:
    u = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    return u if u < (1 << 63) else u - (1 << 64)


def _split_lo_hi(data: Sequence[int]) -> Tuple[List[int], List[int]]:
    lo = [lo32(int(v)) for v in data]
    hi = [hi32(int(v)) for v in data]
    return lo, hi


def fast_pfor_encode_int64(data: Sequence[int]) -> bytes:
    """Wire: LE u32 loWordCount, lo FastPFOR words, hi FastPFOR words (via zstd on limbs)."""
    lo, hi = _split_lo_hi(data)
    lo_raw = struct.pack("<%di" % len(lo), *lo)
    hi_raw = struct.pack("<%di" % len(hi), *hi)
    c_lo = compress_native_lz_family(lo_raw, "zstd_level_3")
    c_hi = compress_native_lz_family(hi_raw, "zstd_level_3")
    return struct.pack("<I", len(c_lo)) + c_lo + c_hi


def fast_pfor_decode_int64(wire: bytes, n: int) -> List[int]:
    if len(wire) < 4:
        raise ValueError("FastPFOR int64: truncated header")
    lo_len = struct.unpack_from("<I", wire, 0)[0]
    if lo_len > len(wire) - 4:
        raise ValueError("FastPFOR int64: truncated body")
    c_lo = wire[4 : 4 + lo_len]
    c_hi = wire[4 + lo_len :]
    lo = struct.unpack("<%di" % n, decompress_native_lz_family(c_lo, "zstd_level_3", n * 4))
    hi = struct.unpack("<%di" % n, decompress_native_lz_family(c_hi, "zstd_level_3", n * 4))
    return [reconstruct(lo[i], hi[i]) for i in range(n)]


def stream_vbyte_encode_int64(data: Sequence[int]) -> bytes:
    lo, hi = _split_lo_hi(data)
    e_lo = _svb_encode_d1([v & 0xFFFFFFFF for v in lo])
    e_hi = _svb_encode_d1([v & 0xFFFFFFFF for v in hi])
    return struct.pack("<I", len(e_lo)) + e_lo + e_hi


def stream_vbyte_decode_int64(wire: bytes, n: int) -> List[int]:
    if len(wire) < 4:
        raise ValueError("streamvbyte int64: truncated header")
    lo_len = struct.unpack_from("<I", wire, 0)[0]
    if lo_len > len(wire) - 4:
        raise ValueError("streamvbyte int64: truncated lo payload")
    e_lo = wire[4 : 4 + lo_len]
    e_hi = wire[4 + lo_len :]
    lo = _svb_decode_d1(e_lo, n)
    hi = _svb_decode_d1(e_hi, n)
    return [reconstruct(lo[i], hi[i]) for i in range(n)]


def simple8b_encode_int64(data: Sequence[int]) -> bytes:
    lo, hi = _split_lo_hi(data)
    u32 = lo + hi
    raw = struct.pack("<%dI" % len(u32), *[v & 0xFFFFFFFF for v in u32])
    return simple8b_encode_bytes_from_i32_column(raw)


def simple8b_decode_int64(body: bytes, n: int) -> List[int]:
    raw = simple8b_decode_bytes_to_i32_column(body)
    if len(raw) != 2 * n * 4:
        raise ValueError("simple8b int64: limb length mismatch")
    u32 = struct.unpack("<%dI" % (2 * n), raw)
    lo = u32[:n]
    hi = u32[n:]
    return [reconstruct(lo[i], hi[i]) for i in range(n)]
