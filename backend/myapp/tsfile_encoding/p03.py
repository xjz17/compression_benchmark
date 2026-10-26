"""
GORILLA (V2) and CHIMP — port of ``cpp/src/tsfile_encoding/tsfile_encode_p03.cpp`` /
``tsfile_decode_p03.cpp``.
"""
from __future__ import annotations

import struct
from typing import List, Sequence

from .common import clz_u32, ctz_u32, to_i32

_K_VALUE_BITS = 32
_K_LEADING_ZERO_BITS = 5
_K_MEANINGFUL_XOR_BITS = 5
_K_GORILLA_END_INT = -2147483648
_K_GORILLA_END_FLOAT_BITS = 0x7FC00000
_K_CHIMP_PREVIOUS = 64
_K_CHIMP_PREV_LOG2 = 6
_K_CHIMP_THRESHOLD = 5 + _K_CHIMP_PREV_LOG2
_K_CHIMP_SET_LSB = (1 << (_K_CHIMP_THRESHOLD + 1)) - 1
_K_CASE_ZERO_META = _K_CHIMP_PREV_LOG2 + 2
_K_CASE_ONE_META = _K_CHIMP_PREV_LOG2 + 10
_K_CASE_ONE_META_FLOAT_DEC = _K_CHIMP_PREV_LOG2 + 8
_K_CHIMP_END_FLOAT = 0x7FC00000

_K_LEADING_REP = (
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
)
_K_LEADING_ROUND = (
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
)
_K_CHIMP_DECODE_LEADING = (0, 8, 12, 16, 18, 20, 22, 24)


class _GorillaBitStream:
    def __init__(self) -> None:
        self.stored_leading_zeros = 0x7FFFFFFF
        self.stored_trailing_zeros = 0
        self.buffer = 0
        self.bits_left = 8
        self.out = bytearray()

    def flip_byte(self) -> None:
        if self.bits_left == 0:
            self.out.append(self.buffer & 0xFF)
            self.buffer = 0
            self.bits_left = 8

    def skip_bit(self) -> None:
        self.bits_left -= 1
        self.flip_byte()

    def write_bit(self) -> None:
        self.buffer = (self.buffer | (1 << (self.bits_left - 1))) & 0xFF
        self.bits_left -= 1
        self.flip_byte()

    def write_bits(self, value: int, bits: int) -> None:
        value = int(value) & 0xFFFFFFFFFFFFFFFF
        while bits > 0:
            shift = bits - self.bits_left
            if shift >= 0:
                mask = (1 << self.bits_left) - 1
                self.buffer = (self.buffer | ((value >> shift) & mask)) & 0xFF
                bits -= self.bits_left
                self.bits_left = 0
            else:
                shift = self.bits_left - bits
                self.buffer = (self.buffer | ((value << shift) & 0xFF)) & 0xFF
                self.bits_left -= bits
                bits = 0
            self.flip_byte()

    def finish_stream(self) -> None:
        self.bits_left = 0
        self.flip_byte()


class _IntGorillaEncoder:
    def __init__(self) -> None:
        self.b = _GorillaBitStream()
        self.first_written = False
        self.stored_value = 0

    def write_first(self, value: int) -> None:
        self.stored_value = to_i32(value)
        self.b.write_bits(self.stored_value & 0xFFFFFFFF, _K_VALUE_BITS)

    def write_existing_leading(self, xorv: int) -> None:
        self.b.skip_bit()
        sig = _K_VALUE_BITS - self.b.stored_leading_zeros - self.b.stored_trailing_zeros
        self.b.write_bits(xorv >> self.b.stored_trailing_zeros, sig)

    def write_new_leading(self, xorv: int, lz: int, tz: int) -> None:
        self.b.write_bit()
        sig = _K_VALUE_BITS - lz - tz
        self.b.write_bits(lz, _K_LEADING_ZERO_BITS)
        self.b.write_bits(sig - 1, _K_MEANINGFUL_XOR_BITS)
        self.b.write_bits(xorv >> tz, sig)
        self.b.stored_leading_zeros = lz
        self.b.stored_trailing_zeros = tz

    def compress_value(self, value: int) -> None:
        xors = to_i32(self.stored_value ^ to_i32(value))
        xoru = xors & 0xFFFFFFFF
        self.stored_value = to_i32(value)
        if xoru == 0:
            self.b.skip_bit()
        else:
            self.b.write_bit()
            lz = clz_u32(xoru)
            tz = ctz_u32(xoru)
            if lz >= self.b.stored_leading_zeros and tz >= self.b.stored_trailing_zeros:
                self.write_existing_leading(xoru)
            else:
                self.write_new_leading(xoru, lz, tz)

    def encode(self, value: int) -> None:
        if self.first_written:
            self.compress_value(value)
        else:
            self.write_first(value)
            self.first_written = True

    def flush_ending(self, end_int: int) -> bytes:
        self.encode(end_int)
        self.b.finish_stream()
        return bytes(self.b.out)


class _IntChimpEncoder:
    def __init__(self) -> None:
        self.b = _GorillaBitStream()
        self.first_written = False
        self.stored_values = [0] * _K_CHIMP_PREVIOUS
        self.indices = [0] * (1 << (_K_CHIMP_THRESHOLD + 1))
        self.index = 0
        self.current = 0

    def write_first(self, value: int) -> None:
        v = to_i32(value)
        self.stored_values[self.current] = v
        self.b.write_bits(v & 0xFFFFFFFF, _K_VALUE_BITS)
        self.indices[v & _K_CHIMP_SET_LSB] = self.index

    def compress_value(self, value: int) -> None:
        value = to_i32(value)
        key = value & _K_CHIMP_SET_LSB
        trailing_zeros = 0
        curr_index = self.indices[key]
        if (self.index - curr_index) < _K_CHIMP_PREVIOUS:
            temp_xor = to_i32(value ^ self.stored_values[curr_index % _K_CHIMP_PREVIOUS])
            trailing_zeros = _K_VALUE_BITS if temp_xor == 0 else ctz_u32(temp_xor & 0xFFFFFFFF)
            if trailing_zeros > _K_CHIMP_THRESHOLD:
                previous_index = curr_index % _K_CHIMP_PREVIOUS
                xorv = temp_xor & 0xFFFFFFFF
            else:
                previous_index = self.index % _K_CHIMP_PREVIOUS
                xorv = (self.stored_values[previous_index] ^ value) & 0xFFFFFFFF
        else:
            previous_index = self.index % _K_CHIMP_PREVIOUS
            xorv = (self.stored_values[previous_index] ^ value) & 0xFFFFFFFF

        if xorv == 0:
            self.b.write_bits(previous_index, _K_CASE_ZERO_META)
            self.b.stored_leading_zeros = _K_VALUE_BITS + 1
        else:
            nlz = clz_u32(xorv)
            leading_zeros = _K_LEADING_ROUND[nlz]
            if trailing_zeros > _K_CHIMP_THRESHOLD:
                sig = _K_VALUE_BITS - leading_zeros - trailing_zeros
                meta = (
                    256 * (_K_CHIMP_PREVIOUS + previous_index)
                    + 32 * _K_LEADING_REP[leading_zeros]
                    + sig
                )
                self.b.write_bits(meta, _K_CASE_ONE_META)
                self.b.write_bits(xorv >> trailing_zeros, sig)
                self.b.stored_leading_zeros = _K_VALUE_BITS + 1
            elif leading_zeros == self.b.stored_leading_zeros:
                self.b.write_bit()
                self.b.skip_bit()
                sig = _K_VALUE_BITS - leading_zeros
                self.b.write_bits(xorv, sig)
            else:
                self.b.stored_leading_zeros = leading_zeros
                sig = _K_VALUE_BITS - leading_zeros
                self.b.write_bits(24 + _K_LEADING_REP[leading_zeros], 5)
                self.b.write_bits(xorv, sig)

        self.current = (self.current + 1) % _K_CHIMP_PREVIOUS
        self.stored_values[self.current] = value
        self.index += 1
        self.indices[key] = self.index

    def encode(self, value: int) -> None:
        if self.first_written:
            self.compress_value(value)
        else:
            self.write_first(value)
            self.first_written = True

    def flush_ending(self, end_marker: int) -> bytes:
        self.encode(end_marker)
        self.b.finish_stream()
        return bytes(self.b.out)


class _GorillaReader:
    def __init__(self, enc: bytes) -> None:
        self.p = enc
        self.len = len(enc)
        self.pos = 0
        self.buffer = 0
        self.bits_left = 0
        self.ok = True

    def flip_byte(self) -> None:
        if self.bits_left == 0:
            if self.pos >= self.len:
                self.ok = False
                return
            self.buffer = self.p[self.pos]
            self.pos += 1
            self.bits_left = 8

    def read_bit(self) -> bool:
        self.flip_byte()
        if not self.ok:
            return False
        bit = ((self.buffer >> (self.bits_left - 1)) & 1) == 1
        self.bits_left -= 1
        return bit

    def read_long(self, bits: int) -> int:
        value = 0
        while bits > 0:
            self.flip_byte()
            if not self.ok:
                return 0
            if bits > self.bits_left or bits == 8:
                d = self.buffer & ((1 << self.bits_left) - 1)
                value = (value << self.bits_left) + d
                bits -= self.bits_left
                self.bits_left = 0
            else:
                d = (self.buffer >> (self.bits_left - bits)) & ((1 << bits) - 1)
                value = (value << bits) + d
                self.bits_left -= bits
                bits = 0
        return value

    def read_next_clear_bit(self, max_bits: int) -> int:
        value = 0
        for _ in range(max_bits):
            value = (value << 1) & 0xFF
            if self.read_bit():
                value |= 1
            else:
                break
        return value

    def read_next_n_bits(self, n: int) -> int:
        value = 0
        for _ in range(n):
            value = (value << 1) & 0xFF
            if self.read_bit():
                value |= 1
        return value


class _GorillaIntDecoder:
    def __init__(self, enc: bytes, end_u32: int) -> None:
        self.r = _GorillaReader(enc)
        self.first_read = False
        self.stored_leading_zeros = 0x7FFFFFFF
        self.stored_trailing_zeros = 0
        self.stored_value = 0
        self.has_next = True
        self.end_bits_u32 = end_u32 & 0xFFFFFFFF

    def read_next_after_control(self, control_bits: int) -> None:
        if control_bits == 3:
            self.stored_leading_zeros = int(self.r.read_long(_K_LEADING_ZERO_BITS))
            if not self.r.ok:
                return
            sig64 = self.r.read_long(_K_MEANINGFUL_XOR_BITS)
            if not self.r.ok:
                return
            significant_bits = (sig64 & 0xFF) + 1
            self.stored_trailing_zeros = _K_VALUE_BITS - significant_bits - self.stored_leading_zeros
        if control_bits in (2, 3):
            sig = _K_VALUE_BITS - self.stored_leading_zeros - self.stored_trailing_zeros
            xorv = int(self.r.read_long(sig))
            if not self.r.ok:
                return
            xorv <<= self.stored_trailing_zeros
            self.stored_value = to_i32(self.stored_value ^ xorv)
        if (self.stored_value & 0xFFFFFFFF) == self.end_bits_u32:
            self.has_next = False

    def read_int(self) -> int:
        ret = self.stored_value
        if not self.first_read:
            self.r.flip_byte()
            self.stored_value = to_i32(self.r.read_long(_K_VALUE_BITS))
            if not self.r.ok:
                return 0
            ret = self.stored_value
            self.first_read = True
        else:
            ret = self.stored_value
        ctrl = self.r.read_next_clear_bit(2)
        if not self.r.ok:
            return ret
        self.read_next_after_control(ctrl)
        return ret


class _IntChimpDecoder:
    def __init__(self, enc: bytes, end_u32: int) -> None:
        self.r = _GorillaReader(enc)
        self.first_read = False
        self.stored_leading_zeros = 0x7FFFFFFF
        self.stored_trailing_zeros = 0
        self.stored_value = 0
        self.stored_values = [0] * _K_CHIMP_PREVIOUS
        self.current = 0
        self.has_next = True
        self.end_bits_u32 = end_u32 & 0xFFFFFFFF
        self.case_one_meta_bits = (
            _K_CASE_ONE_META_FLOAT_DEC if end_u32 == _K_CHIMP_END_FLOAT else _K_CASE_ONE_META
        )

    def read_next(self) -> None:
        control = self.r.read_next_n_bits(2)
        if not self.r.ok:
            return
        if control == 3:
            self.stored_leading_zeros = _K_CHIMP_DECODE_LEADING[int(self.r.read_long(3))]
            if not self.r.ok:
                return
            xor_bits = _K_VALUE_BITS - self.stored_leading_zeros
            if xor_bits <= 0:
                self.r.ok = False
                self.has_next = False
                return
            xorv = int(self.r.read_long(xor_bits))
            if not self.r.ok:
                return
            self.stored_value = to_i32(self.stored_value ^ xorv)
            self.current = (self.current + 1) % _K_CHIMP_PREVIOUS
            self.stored_values[self.current] = self.stored_value
        elif control == 2:
            xor_bits = _K_VALUE_BITS - self.stored_leading_zeros
            if xor_bits <= 0:
                self.r.ok = False
                self.has_next = False
                return
            xorv = int(self.r.read_long(xor_bits))
            if not self.r.ok:
                return
            self.stored_value = to_i32(self.stored_value ^ xorv)
            self.current = (self.current + 1) % _K_CHIMP_PREVIOUS
            self.stored_values[self.current] = self.stored_value
        elif control == 1:
            fill = self.case_one_meta_bits
            temp = int(self.r.read_long(fill))
            if not self.r.ok:
                return
            index = (temp >> (fill - _K_CHIMP_PREV_LOG2)) & ((1 << _K_CHIMP_PREV_LOG2) - 1)
            fill -= _K_CHIMP_PREV_LOG2
            self.stored_leading_zeros = _K_CHIMP_DECODE_LEADING[(temp >> (fill - 3)) & 7]
            fill -= 3
            significant_bits = (temp >> (fill - 5)) & 31
            self.stored_value = self.stored_values[index]
            if significant_bits == 0:
                significant_bits = _K_VALUE_BITS
            self.stored_trailing_zeros = _K_VALUE_BITS - significant_bits - self.stored_leading_zeros
            xor_bits = _K_VALUE_BITS - self.stored_leading_zeros - self.stored_trailing_zeros
            if self.stored_trailing_zeros < 0 or xor_bits <= 0:
                self.r.ok = False
                self.has_next = False
                return
            xorv = int(self.r.read_long(xor_bits))
            if not self.r.ok:
                return
            xorv <<= self.stored_trailing_zeros
            self.stored_value = to_i32(self.stored_value ^ xorv)
            self.current = (self.current + 1) % _K_CHIMP_PREVIOUS
            self.stored_values[self.current] = self.stored_value
        else:
            previous_index = int(self.r.read_long(_K_CHIMP_PREV_LOG2))
            if not self.r.ok:
                self.has_next = False
                return
            self.stored_value = self.stored_values[previous_index]
            self.current = (self.current + 1) % _K_CHIMP_PREVIOUS
            self.stored_values[self.current] = self.stored_value
        if not self.r.ok:
            self.has_next = False
        elif (self.stored_value & 0xFFFFFFFF) == self.end_bits_u32:
            self.has_next = False
        elif (self.stored_values[self.current] & 0xFFFFFFFF) == self.end_bits_u32:
            self.has_next = False

    def read_int(self) -> int:
        ret = self.stored_value
        if not self.first_read:
            self.r.flip_byte()
            self.stored_value = to_i32(self.r.read_long(_K_VALUE_BITS))
            if not self.r.ok:
                return 0
            self.stored_values[self.current] = self.stored_value
            self.first_read = True
            ret = self.stored_value
        else:
            ret = self.stored_value
        self.read_next()
        return ret


def encode_gorilla_int32(values: Sequence[int]) -> bytes:
    enc = _IntGorillaEncoder()
    for v in values:
        enc.encode(int(v))
    return enc.flush_ending(_K_GORILLA_END_INT)


def encode_gorilla_float(values: Sequence[float]) -> bytes:
    enc = _IntGorillaEncoder()
    for f in values:
        bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]
        enc.encode(to_i32(bits))
    return enc.flush_ending(to_i32(_K_GORILLA_END_FLOAT_BITS))


def encode_chimp_int32(values: Sequence[int]) -> bytes:
    enc = _IntChimpEncoder()
    for v in values:
        enc.encode(int(v))
    return enc.flush_ending(_K_GORILLA_END_INT)


def encode_chimp_float(values: Sequence[float]) -> bytes:
    enc = _IntChimpEncoder()
    for f in values:
        bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]
        enc.encode(to_i32(bits))
    return enc.flush_ending(to_i32(_K_CHIMP_END_FLOAT))


def decode_gorilla_int32(enc: bytes) -> List[int]:
    dec = _GorillaIntDecoder(enc, _K_GORILLA_END_INT & 0xFFFFFFFF)
    out: List[int] = []
    while dec.has_next:
        out.append(dec.read_int())
        if not dec.r.ok:
            return []
    return out if dec.r.ok else []


def decode_gorilla_float(enc: bytes) -> List[float]:
    dec = _GorillaIntDecoder(enc, _K_GORILLA_END_FLOAT_BITS)
    out: List[float] = []
    while dec.has_next:
        bits = dec.read_int()
        if not dec.r.ok:
            return []
        out.append(struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0])
    return out if dec.r.ok else []


def decode_chimp_int32(enc: bytes) -> List[int]:
    end_u32 = _K_GORILLA_END_INT & 0xFFFFFFFF
    dec = _IntChimpDecoder(enc, end_u32)
    out: List[int] = []
    while dec.has_next and dec.r.ok:
        v = dec.read_int()
        if not dec.r.ok:
            return []
        if (v & 0xFFFFFFFF) == end_u32:
            break
        out.append(v)
    return out if dec.r.ok else []


def decode_chimp_float(enc: bytes) -> List[float]:
    end_u32 = _K_CHIMP_END_FLOAT
    dec = _IntChimpDecoder(enc, end_u32)
    out: List[float] = []
    while dec.has_next and dec.r.ok:
        bits = dec.read_int()
        if not dec.r.ok:
            return []
        if (bits & 0xFFFFFFFF) == end_u32:
            break
        out.append(struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0])
    return out if dec.r.ok else []
