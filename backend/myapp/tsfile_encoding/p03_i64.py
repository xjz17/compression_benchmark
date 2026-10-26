"""GORILLA/CHIMP INT64 and DOUBLE — port of ``tsfile_encode_p03_int64.cpp``."""
from __future__ import annotations

import struct
from typing import List, Sequence

from .p03 import _GorillaBitStream, _GorillaReader, _K_LEADING_REP, _K_LEADING_ROUND

_K_VALUE_BITS = 64
_K_LEADING_ZERO_BITS = 6
_K_MEANINGFUL_XOR_BITS = 6
_K_GORILLA_END_LONG = -(1 << 63)
_K_GORILLA_END_DOUBLE_BITS = 0x7FF8000000000000
_K_CHIMP_PREVIOUS = 128
_K_CHIMP_PREV_LOG2 = 7
_K_CHIMP_THRESHOLD = 6 + _K_CHIMP_PREV_LOG2
_K_CHIMP_SET_LSB = (1 << (_K_CHIMP_THRESHOLD + 1)) - 1
_K_CASE_ZERO_META = _K_CHIMP_PREV_LOG2 + 2
_K_CASE_ONE_META = _K_CHIMP_PREV_LOG2 + 11
_K_CHIMP_DECODE_LEADING = (0, 8, 12, 16, 18, 20, 22, 24)


def _clz64(x: int) -> int:
    x = int(x) & 0xFFFFFFFFFFFFFFFF
    if x == 0:
        return 64
    return (64 - x.bit_length())


def _ctz64(x: int) -> int:
    x = int(x) & 0xFFFFFFFFFFFFFFFF
    if x == 0:
        return 64
    return (x & -x).bit_length() - 1


def _to_i64(v: int) -> int:
    v = int(v) & 0xFFFFFFFFFFFFFFFF
    if v >= 0x8000000000000000:
        return v - (1 << 64)
    return v


class _LongGorillaEncoder:
    def __init__(self) -> None:
        self.b = _GorillaBitStream()
        self.first_written = False
        self.stored_value = 0

    def write_first(self, value: int) -> None:
        self.stored_value = int(value) & 0xFFFFFFFFFFFFFFFF
        self.b.write_bits(self.stored_value, _K_VALUE_BITS)

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
        value = int(value) & 0xFFFFFFFFFFFFFFFF
        xors = self.stored_value ^ value
        self.stored_value = value
        if xors == 0:
            self.b.skip_bit()
        else:
            self.b.write_bit()
            lz = _clz64(xors)
            tz = _ctz64(xors)
            if lz >= self.b.stored_leading_zeros and tz >= self.b.stored_trailing_zeros:
                self.write_existing_leading(xors)
            else:
                self.write_new_leading(xors, lz, tz)

    def encode(self, value: int) -> None:
        if self.first_written:
            self.compress_value(value)
        else:
            self.write_first(value)
            self.first_written = True

    def flush_ending(self, end_val: int) -> bytes:
        self.encode(end_val)
        self.b.finish_stream()
        return bytes(self.b.out)


class _LongChimpEncoder:
    def __init__(self) -> None:
        self.b = _GorillaBitStream()
        self.first_written = False
        self.stored_values = [0] * _K_CHIMP_PREVIOUS
        self.indices = [0] * (1 << (_K_CHIMP_THRESHOLD + 1))
        self.index = 0
        self.current = 0

    def encode(self, value: int) -> None:
        value = int(value) & 0xFFFFFFFFFFFFFFFF
        if not self.first_written:
            self.stored_values[self.current] = value
            self.b.write_bits(value, _K_VALUE_BITS)
            self.indices[value & _K_CHIMP_SET_LSB] = self.index
            self.first_written = True
            return
        key = value & _K_CHIMP_SET_LSB
        xorv = 0
        previous_index = 0
        trailing_zeros = 0
        curr_index = self.indices[key]
        if (self.index - curr_index) < _K_CHIMP_PREVIOUS:
            prev_v = self.stored_values[curr_index % _K_CHIMP_PREVIOUS]
            temp_xor = value ^ prev_v
            trailing_zeros = _K_VALUE_BITS if temp_xor == 0 else _ctz64(temp_xor)
            if trailing_zeros > _K_CHIMP_THRESHOLD:
                previous_index = curr_index % _K_CHIMP_PREVIOUS
                xorv = temp_xor
            else:
                previous_index = self.index % _K_CHIMP_PREVIOUS
                xorv = self.stored_values[previous_index] ^ value
        else:
            previous_index = self.index % _K_CHIMP_PREVIOUS
            xorv = self.stored_values[previous_index] ^ value
        if xorv == 0:
            self.b.write_bits(previous_index, _K_CASE_ZERO_META)
            self.b.stored_leading_zeros = _K_VALUE_BITS + 1
        else:
            leading_zeros = _K_LEADING_ROUND[min(_clz64(xorv), 63)]
            if trailing_zeros > _K_CHIMP_THRESHOLD:
                sig = _K_VALUE_BITS - leading_zeros - trailing_zeros
                meta = (
                    512 * (_K_CHIMP_PREVIOUS + previous_index)
                    + 64 * _K_LEADING_REP[leading_zeros]
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

    def flush_ending(self, end_marker: int) -> bytes:
        self.encode(end_marker)
        self.b.finish_stream()
        return bytes(self.b.out)


class _GorillaLongDecoder:
    """Same control flow as ``p03._GorillaIntDecoder`` with 64-bit value width."""

    def __init__(self, enc: bytes, end_u64: int) -> None:
        self.r = _GorillaReader(enc)
        self.first_read = False
        self.stored_leading_zeros = 0x7FFFFFFF
        self.stored_trailing_zeros = 0
        self.stored_value = 0
        self.has_next = True
        self.end_bits_u64 = end_u64 & 0xFFFFFFFFFFFFFFFF

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
            if sig <= 0 or sig > _K_VALUE_BITS:
                self.r.ok = False
                self.has_next = False
                return
            xorv = self.r.read_long(sig)
            if not self.r.ok:
                return
            xorv = (xorv << self.stored_trailing_zeros) & 0xFFFFFFFFFFFFFFFF
            self.stored_value ^= xorv
        if (self.stored_value & 0xFFFFFFFFFFFFFFFF) == self.end_bits_u64:
            self.has_next = False

    def read_value(self) -> int:
        ret = self.stored_value
        if not self.first_read:
            self.r.flip_byte()
            self.stored_value = self.r.read_long(_K_VALUE_BITS) & 0xFFFFFFFFFFFFFFFF
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


class _LongChimpDecoder:
    def __init__(self, enc: bytes, end_u64: int) -> None:
        self.r = _GorillaReader(enc)
        self.first_read = False
        self.stored_leading_zeros = 0x7FFFFFFF
        self.stored_trailing_zeros = 0
        self.stored_value = 0
        self.stored_values = [0] * _K_CHIMP_PREVIOUS
        self.current = 0
        self.has_next = True
        self.end_bits_u64 = end_u64 & 0xFFFFFFFFFFFFFFFF

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
            self.stored_value ^= xorv
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
            self.stored_value ^= xorv
            self.current = (self.current + 1) % _K_CHIMP_PREVIOUS
            self.stored_values[self.current] = self.stored_value
        elif control == 1:
            fill = _K_CHIMP_PREV_LOG2 + 9  # Java LongChimpDecoder.CASE_ONE_METADATA_LENGTH
            temp = int(self.r.read_long(fill))
            if not self.r.ok:
                return
            index = (temp >> (fill := fill - _K_CHIMP_PREV_LOG2)) & ((1 << _K_CHIMP_PREV_LOG2) - 1)
            self.stored_leading_zeros = _K_CHIMP_DECODE_LEADING[(temp >> (fill := fill - 3)) & 7]
            significant_bits = int((temp >> (fill := fill - 6)) & ((1 << 6) - 1))
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
            xorv = (xorv << self.stored_trailing_zeros) & 0xFFFFFFFFFFFFFFFF
            self.stored_value ^= xorv
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
        elif (self.stored_value & 0xFFFFFFFFFFFFFFFF) == self.end_bits_u64:
            self.has_next = False
        elif (self.stored_values[self.current] & 0xFFFFFFFFFFFFFFFF) == self.end_bits_u64:
            self.has_next = False

    def read_value(self) -> int:
        ret = self.stored_value
        if not self.first_read:
            self.r.flip_byte()
            self.stored_value = self.r.read_long(_K_VALUE_BITS) & 0xFFFFFFFFFFFFFFFF
            if not self.r.ok:
                return 0
            self.stored_values[self.current] = self.stored_value
            self.first_read = True
            ret = self.stored_value
        else:
            ret = self.stored_value
        self.read_next()
        return ret


def encode_gorilla_int64(values: Sequence[int]) -> bytes:
    enc = _LongGorillaEncoder()
    for v in values:
        enc.encode(int(v))
    return enc.flush_ending(_K_GORILLA_END_LONG)


def encode_gorilla_double(values: Sequence[float]) -> bytes:
    enc = _LongGorillaEncoder()
    for d in values:
        bits = struct.unpack("<Q", struct.pack("<d", float(d)))[0]
        enc.encode(bits)
    return enc.flush_ending(_K_GORILLA_END_DOUBLE_BITS)


def encode_chimp_int64(values: Sequence[int]) -> bytes:
    enc = _LongChimpEncoder()
    for v in values:
        enc.encode(int(v))
    return enc.flush_ending(_K_GORILLA_END_LONG)


def encode_chimp_double(values: Sequence[float]) -> bytes:
    enc = _LongChimpEncoder()
    for d in values:
        bits = struct.unpack("<Q", struct.pack("<d", float(d)))[0]
        enc.encode(bits)
    return enc.flush_ending(_K_GORILLA_END_DOUBLE_BITS)


def decode_gorilla_int64(enc: bytes) -> List[int]:
    end_u64 = _K_GORILLA_END_LONG & 0xFFFFFFFFFFFFFFFF
    dec = _GorillaLongDecoder(enc, end_u64)
    out: List[int] = []
    while dec.has_next:
        v = dec.read_value() & 0xFFFFFFFFFFFFFFFF
        if v == end_u64:
            break
        out.append(_to_i64(v))
        if not dec.r.ok:
            break
    return out


def decode_gorilla_double(enc: bytes) -> List[float]:
    dec = _GorillaLongDecoder(enc, _K_GORILLA_END_DOUBLE_BITS)
    out: List[float] = []
    while dec.has_next:
        bits = dec.read_value() & 0xFFFFFFFFFFFFFFFF
        if bits == _K_GORILLA_END_DOUBLE_BITS:
            break
        out.append(struct.unpack("<d", struct.pack("<Q", bits))[0])
        if not dec.r.ok:
            break
    return out


def decode_chimp_int64(enc: bytes) -> List[int]:
    end_u64 = _K_GORILLA_END_LONG & 0xFFFFFFFFFFFFFFFF
    dec = _LongChimpDecoder(enc, end_u64)
    out: List[int] = []
    while dec.has_next:
        v = dec.read_value() & 0xFFFFFFFFFFFFFFFF
        if v == end_u64:
            break
        out.append(_to_i64(v))
        if not dec.r.ok:
            break
    return out


def decode_chimp_double(enc: bytes) -> List[float]:
    dec = _LongChimpDecoder(enc, _K_GORILLA_END_DOUBLE_BITS)
    out: List[float] = []
    while dec.has_next:
        bits = dec.read_value() & 0xFFFFFFFFFFFFFFFF
        if bits == _K_GORILLA_END_DOUBLE_BITS:
            break
        out.append(struct.unpack("<d", struct.pack("<Q", bits))[0])
        if not dec.r.ok:
            break
    return out
