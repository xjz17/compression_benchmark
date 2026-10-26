"""
In-process BUFF kernel (bounded double + split column packing), aligned with upstream
``external/compression/buff/database/src/methods/prec_double.rs`` and
``compress/split_double.rs``, and with ``org.example.buff`` (Java) / ``cpp/src/buff_kernel`` (C++).

This is **not** the subprocess ``comp_profiler`` benchmark; use for algorithm experiments.
"""

from __future__ import annotations

import math
import struct
from typing import List, Tuple


EXP_MASK = 0x7FF0000000000000
FIRST_ONE = 0x8000000000000000
NEG_ONE = 0xFFFFFFFFFFFFFFFF


def _double_to_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _bits_to_double(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & 0xFFFFFFFFFFFFFFFF))[0]


def get_precision_bound(precision: int) -> float:
    s = "0." + "0" * precision + "49"
    return float(s)


def _countr_zero_u64(xu: int) -> int:
    if xu == 0:
        return 64
    return (xu & -xu).bit_length() - 1


def _u64(x: int) -> int:
    return x & 0xFFFFFFFFFFFFFFFF


def _u64_shl(x: int, n: int) -> int:
    return _u64(_u64(x) << n)


class BuffPrecisionBound:
    """Port of Rust ``PrecisionBound`` / Java ``BuffPrecisionBound``."""

    __slots__ = ("position", "precision", "precision_exp", "int_length", "decimal_length")

    def __init__(self, prec_delta: float) -> None:
        self.position = 0
        self.precision = prec_delta
        xu = _double_to_bits(prec_delta)
        self.precision_exp = ((xu & EXP_MASK) >> 52) - 1023
        self.int_length = 0
        self.decimal_length = 0

    def precision_bound(self, orig: float) -> float:
        origu = _u64(_double_to_bits(orig))
        curu = _u64(origu & _u64_shl(0xFFFFFFFFFFFFFFFF, self.position) | _u64_shl(1, self.position))
        cur = _bits_to_double(curu)
        pre = cur
        bounded = self.is_bounded(orig, cur)
        if bounded:
            while bounded:
                if self.position == 52:
                    return pre
                self.position += 1
                curu = _u64(origu & _u64_shl(0xFFFFFFFFFFFFFFFF, self.position) | _u64_shl(1, self.position))
                cur = _bits_to_double(curu)
                if not self.is_bounded(orig, cur):
                    bounded = False
                    break
                pre = cur
        else:
            while not bounded:
                if self.position == 0:
                    break
                self.position -= 1
                curu = _u64(origu & _u64_shl(0xFFFFFFFFFFFFFFFF, self.position) | _u64_shl(1, self.position))
                cur = _bits_to_double(curu)
                if self.is_bounded(orig, cur):
                    bounded = True
                    pre = cur
                    break
        return pre

    def cal_length(self, x: float) -> None:
        xu = _double_to_bits(x)
        trailing_zeros = _countr_zero_u64(xu)
        exp = ((xu & EXP_MASK) >> 52) - 1023
        dec_length = 0
        if 52 <= trailing_zeros:
            if exp < 0:
                dec_length = -exp
                if exp < self.precision_exp:
                    dec_length = 0
        elif (52 - trailing_zeros) > exp:
            dec_length = (52 - trailing_zeros) - exp
        if exp >= 0:
            if (exp + 1) > self.int_length:
                self.int_length = exp + 1
        if dec_length > self.decimal_length:
            self.decimal_length = dec_length

    def set_length(self, ilen: int, dlen: int) -> None:
        self.int_length = ilen
        self.decimal_length = dlen

    def fetch_components(self, bd: float) -> Tuple[int, int]:
        bdu = _u64(_double_to_bits(bd))
        exp = ((bdu & EXP_MASK) >> 52) - 1023
        sign = bdu & FIRST_ONE
        int_part = 0
        dec_part = 0
        if exp >= 0:
            dec_part = _u64(bdu << (12 + exp))
            int_part = _u64(((bdu << 11) | FIRST_ONE) >> (63 - exp))
            if sign != 0:
                int_part = _u64(~int_part)
                dec_part = _u64(~dec_part + 1)
        elif exp < self.precision_exp:
            dec_part = 0
            if sign != 0:
                int_part = NEG_ONE
                dec_part = _u64(~dec_part)
        else:
            dec_part = _u64(((bdu << 11) | FIRST_ONE) >> (-exp - 1))
            if sign != 0:
                int_part = NEG_ONE
                dec_part = _u64(~dec_part)
        signed_int = struct.unpack("<q", struct.pack("<Q", _u64(int_part)))[0]
        shift = 64 - self.decimal_length
        dec_out = (dec_part >> shift) if shift > 0 else dec_part
        return signed_int, int(_u64(dec_out) & 0xFFFFFFFF)

    def is_bounded(self, a: float, b: float) -> bool:
        return abs(a - b) < self.precision


class BuffBitPackReader:
    """LSB-first bit reader (matches ``BuffBitPackWriter`` / Rust ``BitPack::read``)."""

    BYTE_BITS = 8

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._bit_pos = 0

    def read(self, bits: int) -> int:
        v = 0
        for i in range(bits):
            bi = self._bit_pos // self.BYTE_BITS
            off = self._bit_pos % self.BYTE_BITS
            if bi >= len(self._data):
                raise ValueError("BuffBitPackReader past end")
            b = (self._data[bi] >> off) & 1
            v |= b << i
            self._bit_pos += 1
        return v


def _sign_extend_u32(v: int, bits: int) -> int:
    if bits <= 0:
        return 0
    v &= (1 << bits) - 1
    sign = 1 << (bits - 1)
    if v & sign:
        v -= 1 << bits
    return v


class BuffBitPackWriter:
    """Port of Rust ``BitPack::write`` for ``Vec<u8>``."""

    MAX_BITS = 32
    BYTE_BITS = 8

    def __init__(self) -> None:
        self._buf = bytearray(64)
        self._cursor = 0
        self._bits_in_byte = 0

    def write(self, value: int, bits: int) -> None:
        v = value & 0xFFFFFFFF
        mut_bits = bits
        if mut_bits < self.MAX_BITS:
            v &= (1 << mut_bits) - 1
        while mut_bits > 0:
            bits_left = self.BYTE_BITS - self._bits_in_byte
            if mut_bits <= bits_left:
                self._buf[self._cursor] |= (v & 0xFF) << self._bits_in_byte
                self._bits_in_byte += mut_bits
                if self._bits_in_byte >= self.BYTE_BITS:
                    self._cursor += 1
                    self._bits_in_byte = 0
                break
            take = bits_left
            bb = v & ((1 << take) - 1)
            self._buf[self._cursor] |= bb << self._bits_in_byte
            self._cursor += 1
            self._bits_in_byte = 0
            v >>= take
            mut_bits -= take
            self._ensure_capacity()
        self._ensure_capacity()

    def _ensure_capacity(self) -> None:
        if self._cursor >= len(self._buf) - 4:
            self._buf.extend(bytes(len(self._buf)))

    def to_bytes(self) -> bytes:
        n = self._cursor
        if self._bits_in_byte > 0:
            n += 1
        return bytes(self._buf[:n])


def decimal_bits_for_precision(prec: int) -> int:
    m = {
        1: 5,
        2: 8,
        3: 11,
        4: 15,
        5: 18,
        6: 21,
        7: 25,
        8: 28,
        9: 31,
        10: 35,
        11: 38,
        12: 50,
    }
    return m.get(prec, 10)


class DecodedColumn:
    __slots__ = ("t", "ilen", "dlen", "int_parts", "dec_parts")

    def __init__(
        self,
        t: int,
        ilen: int,
        dlen: int,
        int_parts: List[int],
        dec_parts: List[int],
    ) -> None:
        self.t = t
        self.ilen = ilen
        self.dlen = dlen
        self.int_parts = int_parts
        self.dec_parts = dec_parts


def decode_doubles(data: bytes) -> DecodedColumn:
    """Inverse of :func:`encode_doubles` at component level (no float reconstruction)."""
    if not data:
        return DecodedColumn(0, 0, 0, [], [])
    rd = BuffBitPackReader(data)
    t = rd.read(32)
    ilen = rd.read(32)
    dlen = rd.read(32)
    int_parts = [_sign_extend_u32(rd.read(ilen), ilen) for _ in range(t)]
    dec_parts = [rd.read(dlen) & 0xFFFFFFFF for _ in range(t)]
    return DecodedColumn(t, ilen, dlen, int_parts, dec_parts)


def encode_doubles(values: List[float], scale: int) -> bytes:
    """Same contract as Java ``BuffSplitDoubleColumnCodec.encodeDoubles``."""
    if not values:
        return b""
    prec = int(math.floor(math.log10(scale)))
    prec_delta = get_precision_bound(prec)
    bound = BuffPrecisionBound(prec_delta)
    bd_vec: List[float] = []
    for val in values:
        v = bound.precision_bound(val)
        bd_vec.append(v)
        bound.cal_length(v)
    ilen = int(bound.int_length)
    dlen = int(bound.decimal_length)
    bp = BuffBitPackWriter()
    t = len(values)
    bp.write(t, 32)
    bp.write(ilen, 32)
    bp.write(dlen, 32)
    int_parts: List[int] = []
    dec_parts: List[int] = []
    for bd in bd_vec:
        ip, dp = bound.fetch_components(bd)
        int_parts.append(ip)
        dec_parts.append(dp)
    for ip in int_parts:
        bp.write(ip & 0xFFFFFFFF, ilen)
    for dp in dec_parts:
        bp.write(dp & 0xFFFFFFFF, dlen)
    return bp.to_bytes()
