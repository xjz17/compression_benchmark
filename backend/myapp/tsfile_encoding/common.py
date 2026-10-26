"""Shared bit/byte helpers for TsFile encoding ports (``cpp/src/tsfile_encoding``)."""
from __future__ import annotations

import struct
from typing import List, Tuple


def to_i32(value: int) -> int:
    return struct.unpack("<i", struct.pack("<I", int(value) & 0xFFFFFFFF))[0]


def clz_u32(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    if x == 0:
        return 32
    return 32 - x.bit_length()


def ctz_u32(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    if x == 0:
        return 32
    return (x & -x).bit_length() - 1


def append_uvarint(out: bytearray, v: int) -> None:
    v = int(v) & 0xFFFFFFFF
    while (v & 0xFFFFFF80) != 0:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)


def append_u32_be(out: bytearray, v: int) -> None:
    v = int(v)
    out.extend([(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])


def append_bytes(dst: bytearray, src: bytes) -> None:
    dst.extend(src)


def append_int_le_padded(out: bytearray, value: int, bit_width: int) -> None:
    padded = (bit_width + 7) // 8
    offset = 0
    value = int(value)
    while padded > 0:
        out.append((value >> offset) & 0xFF)
        offset += 8
        padded -= 1


def append_f32_be(out: bytearray, value: float) -> None:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    append_u32_be(out, bits)


def read_uvarint(data: bytes, pos: int) -> Tuple[int, int]:
    v = 0
    s = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        v |= (b & 0x7F) << s
        if (b & 0x80) == 0:
            return v, pos
        s += 7
    return 0, -1


def bytes_to_int_at_bits(data: bytes, pos_bits: int, width: int) -> int:
    ret = 0
    cnt = pos_bits & 7
    index = pos_bits >> 3
    while width > 0:
        m = (8 - cnt) if (width + cnt >= 8) else width
        width -= m
        ret <<= m
        y = data[index] & (0xFF >> cnt)
        y = (y & 0xFF) >> (8 - cnt - m)
        ret |= y & 0xFF
        cnt += m
        if cnt == 8:
            index += 1
            cnt = 0
    return ret


class ByteCursor:
    def __init__(self, data: bytes) -> None:
        self.p = data
        self.len = len(data)
        self.pos = 0
        self.ok = True

    def has(self, n: int) -> bool:
        return self.pos + n <= self.len

    def get_u8(self) -> int:
        if not self.has(1):
            self.ok = False
            return 0
        b = self.p[self.pos]
        self.pos += 1
        return b

    def read_uvarint(self) -> int:
        v = 0
        s = 0
        while True:
            b = self.get_u8()
            if not self.ok:
                return 0
            v |= (b & 0x7F) << s
            if (b & 0x80) == 0:
                return v
            s += 7

    def read_i32_be(self) -> int:
        if not self.has(4):
            self.ok = False
            return 0
        u = (
            (self.p[self.pos] << 24)
            | (self.p[self.pos + 1] << 16)
            | (self.p[self.pos + 2] << 8)
            | self.p[self.pos + 3]
        )
        self.pos += 4
        return struct.unpack(">i", struct.pack(">I", u & 0xFFFFFFFF))[0]

    def read_f32_be(self) -> float:
        return struct.unpack(">f", struct.pack(">i", self.read_i32_be()))[0]

    def read_i64_be(self) -> int:
        if not self.has(8):
            self.ok = False
            return 0
        u = 0
        for i in range(8):
            u = (u << 8) | self.p[self.pos + i]
        self.pos += 8
        return struct.unpack(">q", struct.pack(">Q", u & 0xFFFFFFFFFFFFFFFF))[0]

    def read_int_le_padded(self, bit_width: int) -> int:
        n = (bit_width + 7) // 8
        result = 0
        offset = 0
        while n > 0:
            ch = self.get_u8()
            if not self.ok:
                return 0
            result += ch << offset
            offset += 8
            n -= 1
        return result


def unpack8_values(width: int, buf: bytes, offset: int, values: List[int]) -> None:
    byte_idx = offset
    buffer = 0
    total_bits = 0
    value_idx = 0
    while value_idx < 8:
        while total_bits < width:
            if byte_idx >= offset + width:
                return
            buffer = (buffer << 8) | (buf[byte_idx] & 0xFF)
            byte_idx += 1
            total_bits += 8
        while total_bits >= width and value_idx < 8:
            values[value_idx] = buffer >> (total_bits - width)
            value_idx += 1
            total_bits -= width
            if total_bits > 0:
                buffer &= (1 << total_bits) - 1
            else:
                buffer = 0
