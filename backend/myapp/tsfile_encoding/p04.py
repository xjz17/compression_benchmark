"""
SPRINTZ and RLBE — port of ``cpp/src/tsfile_encoding/tsfile_encode_p04.cpp`` /
``tsfile_decode_p04.cpp``.
"""
from __future__ import annotations

import math
import struct
from typing import List, Sequence

from . import p02, p03
from .common import (
    ByteCursor,
    append_bytes,
    append_f32_be,
    append_int_le_padded,
    append_uvarint,
    clz_u32,
    to_i32,
    unpack8_values,
)

_K_SPRINTZ_BLOCK = 8
_K_SPRINTZ_GROUP_MAX = 16
_K_RLBE_BLOCK = 10000


def _int_value_bit_width(num: int) -> int:
    return 32 - clz_u32(int(num) & 0xFFFFFFFF)


def _int_max_bit_width(values: Sequence[int]) -> int:
    w = 1
    for num in values:
        w = max(w, _int_value_bit_width(int(num)))
    return w


def _pack8_values(width: int, values: Sequence[int], offset: int, buf: bytearray) -> None:
    buf_idx = 0
    value_idx = offset
    left_bit = 0
    while value_idx < 8 + offset:
        buffer = 0
        left_size = 32
        if left_bit > 0:
            buffer |= (int(values[value_idx]) << (32 - left_bit))
            left_size -= left_bit
            left_bit = 0
            value_idx += 1
        while left_size >= width and value_idx < 8 + offset:
            buffer |= int(values[value_idx]) << (left_size - width)
            left_size -= width
            value_idx += 1
        if left_size > 0 and value_idx < 8 + offset:
            buffer |= int(int(values[value_idx]) & 0xFFFFFFFF) >> (width - left_size)
            left_bit = width - left_size
        for j in range(4):
            buf[buf_idx] = (buffer >> ((3 - j) * 8)) & 0xFF
            buf_idx += 1
            if buf_idx >= width:
                return


def _sprintz_zigzag(pred: int) -> int:
    if pred <= 0:
        return -2 * pred
    return 2 * pred - 1


class _IntFire:
    def __init__(self) -> None:
        self.learn_shift = 2
        self.bit_width = 8
        self.accumulator = 0
        self.delta = 0

    def reset(self) -> None:
        self.accumulator = 0
        self.delta = 0

    def predict(self, value: int) -> int:
        # Match C++/Java: 32-bit signed wrap on multiply (see IntFire in tsfile_encode_p04.cpp).
        value = to_i32(value)
        alpha = to_i32(self.accumulator) >> self.learn_shift
        diff = to_i32(to_i32(alpha * to_i32(self.delta)) >> self.bit_width)
        return to_i32(value + diff)

    def train(self, pre: int, val: int, err: int) -> None:
        pre = to_i32(pre)
        val = to_i32(val)
        err = to_i32(err)
        gradient = to_i32(-self.delta if err > 0 else self.delta)
        self.accumulator = to_i32(to_i32(self.accumulator) - gradient)
        self.delta = to_i32(val - pre)


class _IntSprintzEncoder:
    def __init__(self) -> None:
        self.fire = _IntFire()
        self.values: List[int] = []
        self.is_first_cached = False
        self.group_num = 0
        self.byte_cache = bytearray()

    def fire_residual(self, value: int, pre_value: int) -> int:
        pr = self.fire.predict(pre_value)
        err = int(value) - pr
        self.fire.train(pre_value, value, err)
        return err

    def predict_stanza(self, value: int, pre_value: int) -> int:
        return _sprintz_zigzag(self.fire_residual(value, pre_value))

    def bit_pack(self) -> None:
        pre_value = self.values[0]
        rest = self.values[1:]
        bw = _int_max_bit_width(rest)
        bytes_arr = bytearray(bw)
        _pack8_values(bw, rest, 0, bytes_arr)
        append_int_le_padded(self.byte_cache, bw, 1)
        append_uvarint(self.byte_cache, pre_value)
        append_bytes(self.byte_cache, bytes(bytes_arr))

    def flush_all(self, out: bytearray) -> None:
        if self.byte_cache:
            append_bytes(out, bytes(self.byte_cache))
            self.byte_cache.clear()
        if self.values:
            sz = len(self.values) | (1 << 7)
            append_int_le_padded(out, sz, 1)
            append_bytes(out, p02.encode_rle_int32_legacy(self.values))
            self.values.clear()
        self.is_first_cached = False
        self.group_num = 0

    def encode(self, value: int, out: bytearray) -> None:
        if not self.is_first_cached:
            self.values.append(int(value))
            self.is_first_cached = True
            return
        self.values.append(int(value))
        if len(self.values) == _K_SPRINTZ_BLOCK + 1:
            pre = self.values[0]
            self.fire.reset()
            tmp_vals = list(self.values)
            for i in range(1, _K_SPRINTZ_BLOCK + 1):
                v = tmp_vals[i]
                tmp_vals[i] = self.predict_stanza(v, pre)
                pre = v
            self.values = tmp_vals
            self.bit_pack()
            self.is_first_cached = False
            self.values.clear()
            self.group_num += 1
            if self.group_num == _K_SPRINTZ_GROUP_MAX:
                self.flush_all(out)


class _FloatSprintzEncoder:
    def __init__(self) -> None:
        self.fire = _IntFire()
        self.values: List[float] = []
        self.convert_buffer = [0] * _K_SPRINTZ_BLOCK
        self.is_first_cached = False
        self.group_num = 0
        self.byte_cache = bytearray()

    def fire_residual(self, value: float, pre_value: float) -> int:
        pb = struct.unpack("<I", struct.pack("<f", float(pre_value)))[0]
        vb = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        pred = self.fire.predict(pb)
        err = to_i32(to_i32(vb) - pred)
        self.fire.train(pb, vb, err)
        return err

    def predict_stanza(self, value: float, pre_value: float) -> int:
        return _sprintz_zigzag(self.fire_residual(value, pre_value))

    def bit_pack(self) -> None:
        pre_value = self.values[0]
        rest = list(self.convert_buffer)
        bw = _int_max_bit_width(rest)
        bytes_arr = bytearray(bw)
        _pack8_values(bw, rest, 0, bytes_arr)
        append_int_le_padded(self.byte_cache, bw, 1)
        append_f32_be(self.byte_cache, pre_value)
        append_bytes(self.byte_cache, bytes(bytes_arr))

    def flush_all(self, out: bytearray) -> None:
        if self.byte_cache:
            append_bytes(out, bytes(self.byte_cache))
            self.byte_cache.clear()
        if self.values:
            sz = len(self.values) | (1 << 7)
            append_int_le_padded(out, sz, 1)
            append_bytes(out, p03.encode_gorilla_float(self.values))
            self.values.clear()
        self.is_first_cached = False
        self.group_num = 0

    def encode(self, value: float, out: bytearray) -> None:
        if not self.is_first_cached:
            self.values.append(float(value))
            self.is_first_cached = True
            return
        self.values.append(float(value))
        if len(self.values) == _K_SPRINTZ_BLOCK + 1:
            self.fire.reset()
            for i in range(1, _K_SPRINTZ_BLOCK + 1):
                self.convert_buffer[i - 1] = self.predict_stanza(
                    self.values[i], self.values[i - 1]
                )
            self.bit_pack()
            self.is_first_cached = False
            self.values.clear()
            self.group_num += 1
            if self.group_num == _K_SPRINTZ_GROUP_MAX:
                self.flush_all(out)


class _IntRlbEncoder:
    def __init__(self) -> None:
        self.block_bits = bytearray()
        self.diff_value = [0] * (_K_RLBE_BLOCK + 1)
        self.length_code = [0] * (_K_RLBE_BLOCK + 1)
        self.leng_rle = [0] * (_K_RLBE_BLOCK + 1)
        self.write_index = -1
        self.previous_value = 0
        self.byte_buffer = 0
        self.number_left_in_buffer = 0

    @staticmethod
    def cal_binary_length(val: int) -> int:
        if val == 0:
            return 1
        return 32 - clz_u32(val & 0xFFFFFFFF)

    @staticmethod
    def calc_fibonacci(val: int) -> int:
        fib = [0] * 200
        fib[0] = 1
        fib[1] = 1
        i = 2
        while fib[i - 1] <= val:
            fib[i] = fib[i - 1] + fib[i - 2]
            i += 1
        i -= 1
        valfib = 0
        while val > 0:
            while fib[i] > val and i >= 1:
                i -= 1
            valfib |= 1 << (i - 1)
            val -= fib[i]
        return valfib

    def write_bit(self, b: bool) -> None:
        self.byte_buffer = (self.byte_buffer << 1) & 0xFF
        if b:
            self.byte_buffer |= 1
        self.number_left_in_buffer += 1
        if self.number_left_in_buffer == 8:
            self.clear_buffer()

    def clear_buffer(self) -> None:
        if self.number_left_in_buffer == 0:
            return
        self.byte_buffer = (self.byte_buffer << (8 - self.number_left_in_buffer)) & 0xFF
        self.block_bits.append(self.byte_buffer)
        self.number_left_in_buffer = 0
        self.byte_buffer = 0

    def rle_on_length_code(self) -> None:
        i = 0
        while i <= self.write_index:
            j = i
            temprlecal = 0
            while j <= self.write_index and self.length_code[j] == self.length_code[i]:
                j += 1
                temprlecal += 1
            self.leng_rle[i] = temprlecal
            i = j

    def flush_segment(self, i: int) -> None:
        for j in range(5, -1, -1):
            self.write_bit((self.length_code[i] & (1 << j)) != 0)
        fib = self.calc_fibonacci(self.leng_rle[i])
        fiblen = self.cal_binary_length(fib)
        for j in range(fiblen):
            self.write_bit((fib & (1 << j)) != 0)
        self.write_bit(True)
        j = i
        while True:
            temp_difflen = self.cal_binary_length(self.diff_value[j])
            du = self.diff_value[j] & 0xFFFFFFFF
            for k in range(temp_difflen - 1, -1, -1):
                self.write_bit((du & (1 << k)) != 0)
            j += 1
            if j > self.write_index or self.leng_rle[j] != 0:
                break

    def write_write_index_bits(self) -> None:
        n = (self.write_index + 1) & 0xFFFFFFFF
        for i in range(31, -1, -1):
            self.write_bit((n & (1 << i)) != 0)

    def flush_block(self) -> None:
        if self.write_index == -1:
            return
        self.write_write_index_bits()
        self.rle_on_length_code()
        for i in range(self.write_index + 1):
            if self.leng_rle[i] > 0:
                self.flush_segment(i)
        self.clear_buffer()
        self.write_index = -1
        self.byte_buffer = 0
        self.number_left_in_buffer = 0
        self.diff_value = [0] * (_K_RLBE_BLOCK + 1)
        self.length_code = [0] * (_K_RLBE_BLOCK + 1)
        self.leng_rle = [0] * (_K_RLBE_BLOCK + 1)

    def encode_value(self, value: int) -> None:
        if self.write_index == -1:
            self.write_index = 0
            self.diff_value[0] = int(value)
            self.length_code[0] = self.cal_binary_length(value)
            self.previous_value = int(value)
            return
        self.write_index += 1
        self.diff_value[self.write_index] = int(value) - self.previous_value
        self.length_code[self.write_index] = self.cal_binary_length(self.diff_value[self.write_index])
        self.previous_value = int(value)
        if self.write_index == _K_RLBE_BLOCK - 1:
            self.flush_block()


class _FloatRlbEncoder:
    def __init__(self) -> None:
        self.block_bits = bytearray()
        self.diff_value: List[float] = [0.0] * (_K_RLBE_BLOCK + 1)
        self.length_code = [0] * (_K_RLBE_BLOCK + 1)
        self.leng_rle = [0] * (_K_RLBE_BLOCK + 1)
        self.write_index = -1
        self.previous_value = 0.0
        self.byte_buffer = 0
        self.number_left_in_buffer = 0

    @staticmethod
    def cal_binary_length_float(val: float) -> int:
        bits = struct.unpack("<I", struct.pack("<f", float(val)))[0]
        if bits == 0:
            return 1
        return 32 - clz_u32(bits)

    def write_bit(self, b: bool) -> None:
        self.byte_buffer = (self.byte_buffer << 1) & 0xFF
        if b:
            self.byte_buffer |= 1
        self.number_left_in_buffer += 1
        if self.number_left_in_buffer == 8:
            self.clear_buffer()

    def clear_buffer(self) -> None:
        if self.number_left_in_buffer == 0:
            return
        self.byte_buffer = (self.byte_buffer << (8 - self.number_left_in_buffer)) & 0xFF
        self.block_bits.append(self.byte_buffer)
        self.number_left_in_buffer = 0
        self.byte_buffer = 0

    def rle_on_length_code(self) -> None:
        i = 0
        while i <= self.write_index:
            j = i
            temprlecal = 0
            while j <= self.write_index and self.length_code[j] == self.length_code[i]:
                j += 1
                temprlecal += 1
            self.leng_rle[i] = temprlecal
            i = j

    def flush_segment(self, i: int) -> None:
        for j in range(5, -1, -1):
            self.write_bit((self.length_code[i] & (1 << j)) != 0)
        fib = _IntRlbEncoder.calc_fibonacci(self.leng_rle[i])
        fiblen = _IntRlbEncoder.cal_binary_length(fib)
        for j in range(fiblen):
            self.write_bit((fib & (1 << j)) != 0)
        self.write_bit(True)
        j = i
        while True:
            temp_difflen = self.cal_binary_length_float(self.diff_value[j])
            du = struct.unpack("<I", struct.pack("<f", self.diff_value[j]))[0]
            for k in range(temp_difflen - 1, -1, -1):
                self.write_bit((du & (1 << k)) != 0)
            j += 1
            if j > self.write_index or self.leng_rle[j] != 0:
                break

    def write_write_index_bits(self) -> None:
        n = (self.write_index + 1) & 0xFFFFFFFF
        for i in range(31, -1, -1):
            self.write_bit((n & (1 << i)) != 0)

    def flush_block(self) -> None:
        if self.write_index == -1:
            return
        self.write_write_index_bits()
        self.rle_on_length_code()
        for i in range(self.write_index + 1):
            if self.leng_rle[i] > 0:
                self.flush_segment(i)
        self.clear_buffer()
        self.write_index = -1
        self.byte_buffer = 0
        self.number_left_in_buffer = 0
        self.diff_value = [0.0] * (_K_RLBE_BLOCK + 1)
        self.length_code = [0] * (_K_RLBE_BLOCK + 1)
        self.leng_rle = [0] * (_K_RLBE_BLOCK + 1)

    def encode_value(self, value: float) -> None:
        if self.write_index == -1:
            self.write_index = 0
            self.diff_value[0] = float(value)
            self.length_code[0] = self.cal_binary_length_float(value)
            self.previous_value = float(value)
            return
        self.write_index += 1
        self.diff_value[self.write_index] = float(value) - self.previous_value
        self.length_code[self.write_index] = self.cal_binary_length_float(self.diff_value[self.write_index])
        self.previous_value = float(value)
        if self.write_index == _K_RLBE_BLOCK - 1:
            self.flush_block()


def encode_sprintz_int32_legacy(values: Sequence[int]) -> bytes:
    out = bytearray()
    enc = _IntSprintzEncoder()
    for v in values:
        enc.encode(int(v), out)
    enc.flush_all(out)
    return bytes(out)


def encode_sprintz_float_legacy(values: Sequence[float]) -> bytes:
    out = bytearray()
    enc = _FloatSprintzEncoder()
    for f in values:
        enc.encode(float(f), out)
    enc.flush_all(out)
    return bytes(out)


def encode_rlbe_int32_legacy(values: Sequence[int]) -> bytes:
    enc = _IntRlbEncoder()
    for v in values:
        enc.encode_value(int(v))
    enc.flush_block()
    return bytes(enc.block_bits)


def encode_rlbe_float_legacy(values: Sequence[float]) -> bytes:
    enc = _FloatRlbEncoder()
    for f in values:
        enc.encode_value(float(f))
    enc.flush_block()
    return bytes(enc.block_bits)


def _recalc_int_fire(buf: List[int]) -> None:
    for i in range(1, _K_SPRINTZ_BLOCK + 1):
        if (buf[i] & 1) == 0:
            buf[i] = int(math.trunc(-buf[i] / 2))
        else:
            buf[i] = (buf[i] + 1) // 2
    fire = _IntFire()
    pre = buf[0]
    for i in range(1, _K_SPRINTZ_BLOCK + 1):
        pred = fire.predict(buf[i - 1])
        err = buf[i]
        buf[i] = pred + err
        fire.train(buf[i - 1], buf[i], err)


def _recalc_float_fire(pre_value: float, convert_buf: List[int], out_buf: List[float]) -> None:
    out_buf[0] = pre_value
    for i in range(_K_SPRINTZ_BLOCK):
        if (convert_buf[i] & 1) == 0:
            convert_buf[i] = int(math.trunc(-convert_buf[i] / 2))
        else:
            convert_buf[i] = (convert_buf[i] + 1) // 2
    fire = _IntFire()
    pb = struct.unpack("<I", struct.pack("<f", pre_value))[0]
    p = fire.predict(pb)
    e = convert_buf[0]
    convert_buf[0] = to_i32(convert_buf[0] + p)
    out_buf[1] = struct.unpack("<f", struct.pack("<I", convert_buf[0] & 0xFFFFFFFF))[0]
    fire.train(pb, convert_buf[0], e)
    for i in range(1, _K_SPRINTZ_BLOCK):
        pred = fire.predict(convert_buf[i - 1])
        err = convert_buf[i]
        convert_buf[i] = to_i32(convert_buf[i] + pred)
        out_buf[i + 1] = struct.unpack("<f", struct.pack("<I", convert_buf[i] & 0xFFFFFFFF))[0]
        fire.train(convert_buf[i - 1], convert_buf[i], err)


def _decode_sprintz_int_blocks(c: ByteCursor, out: List[int]) -> bool:
    while c.ok and c.pos < c.len:
        bit_width_byte = c.read_int_le_padded(1)
        if not c.ok:
            return False
        if (bit_width_byte & 0x80) != 0:
            decode_size = bit_width_byte & ~(1 << 7)
            chunk = c.p[c.pos : c.len]
            c.pos = c.len
            tail = p02.decode_rle_int32_legacy(bytes(chunk))
            if len(tail) != decode_size:
                return False
            out.extend(tail)
            break
        pre = c.read_uvarint()
        if not c.ok:
            return False
        if not c.has(bit_width_byte):
            c.ok = False
            return False
        packc = bytes(c.p[c.pos : c.pos + bit_width_byte])
        c.pos += bit_width_byte
        tmp = [0] * 8
        unpack8_values(bit_width_byte, packc, 0, tmp)
        buf = [pre] + tmp
        _recalc_int_fire(buf)
        out.extend(buf[: _K_SPRINTZ_BLOCK + 1])
    return c.ok and c.pos == c.len


def _decode_sprintz_float_blocks(c: ByteCursor, out: List[float]) -> bool:
    while c.ok and c.pos < c.len:
        bit_width_byte = c.read_int_le_padded(1)
        if not c.ok:
            return False
        if (bit_width_byte & 0x80) != 0:
            decode_size = bit_width_byte & ~(1 << 7)
            chunk = c.p[c.pos : c.len]
            c.pos = c.len
            tail = p03.decode_gorilla_float(bytes(chunk))
            if len(tail) != decode_size:
                return False
            out.extend(tail)
            break
        pre = c.read_f32_be()
        if not c.ok:
            return False
        if not c.has(bit_width_byte):
            c.ok = False
            return False
        packc = bytes(c.p[c.pos : c.pos + bit_width_byte])
        c.pos += bit_width_byte
        tmp = [0] * 8
        unpack8_values(bit_width_byte, packc, 0, tmp)
        buf: List[float] = [0.0] * 9
        convert_buf = list(tmp)
        _recalc_float_fire(pre, convert_buf, buf)
        out.extend(buf[: _K_SPRINTZ_BLOCK + 1])
    return c.ok and c.pos == c.len


class _RlbBitReader:
    def __init__(self, enc: bytes) -> None:
        self.p = enc
        self.len = len(enc)
        self.pos = 0
        self.byte_buffer = 0
        self.number_left_in_buffer = 0
        self.ok = True

    def readbit(self) -> int:
        if self.number_left_in_buffer == 0:
            if self.pos >= self.len:
                self.ok = False
                return 0
            self.byte_buffer = self.p[self.pos]
            self.pos += 1
            self.number_left_in_buffer = 8
        top = (self.byte_buffer >> 7) & 1
        self.byte_buffer = (self.byte_buffer << 1) & 0xFF
        self.number_left_in_buffer -= 1
        return top

    def clear_buffer(self) -> None:
        while self.number_left_in_buffer > 0:
            self.readbit()

    def read_block_size_from_stream(self) -> int:
        blocksize = 0
        for i in range(31, -1, -1):
            if self.readbit():
                blocksize |= 1 << i
        return blocksize


def _rlbe_read_block_int(enc: bytes) -> List[int]:
    br = _RlbBitReader(enc)
    br.clear_buffer()
    blocksize = br.read_block_size_from_stream()
    if not br.ok or blocksize < 1:
        return []
    data = [0] * (blocksize * 2 + 1)
    fib = [0] * 4096
    writeindex = -1
    while writeindex < blocksize - 1 and br.ok:
        seglength = 0
        for j in range(5, -1, -1):
            seglength |= br.readbit() << j
        now = br.readbit()
        next_bit = br.readbit()
        j = 1
        runlength = 0
        fib[0] = 1
        fib[1] = 1
        while True:
            if j > 1:
                fib[j] = fib[j - 1] + fib[j - 2]
            if now == 1:
                runlength += fib[j]
            if now == 1 and next_bit == 1:
                break
            j += 1
            now = next_bit
            next_bit = br.readbit()
        for _ in range(1, runlength + 1):
            if writeindex >= blocksize - 1 or not br.ok:
                break
            readtemp = 0
            for k in range(seglength - 1, -1, -1):
                readtemp += br.readbit() << k
            if seglength == 32:
                readtemp -= 1 << 31
            if writeindex == -1:
                writeindex += 1
                data[writeindex] = readtemp
            else:
                writeindex += 1
                data[writeindex] = data[writeindex - 1] + readtemp
    return data[:blocksize] if br.ok else []


def _rlbe_read_block_float(enc: bytes) -> List[float]:
    br = _RlbBitReader(enc)
    br.clear_buffer()
    blocksize = br.read_block_size_from_stream()
    if not br.ok or blocksize < 1:
        return []
    data: List[float] = [0.0] * (blocksize * 2 + 1)
    fib = [0] * 4096
    writeindex = -1
    while writeindex < blocksize - 1 and br.ok:
        seglength = 0
        for j in range(5, -1, -1):
            seglength |= br.readbit() << j
        now = br.readbit()
        next_bit = br.readbit()
        j = 1
        runlength = 0
        fib[0] = 1
        fib[1] = 1
        while True:
            if j > 1:
                fib[j] = fib[j - 1] + fib[j - 2]
            if now == 1:
                runlength += fib[j]
            if now == 1 and next_bit == 1:
                break
            j += 1
            now = next_bit
            next_bit = br.readbit()
        for _ in range(1, runlength + 1):
            if writeindex >= blocksize - 1 or not br.ok:
                break
            readtemp = 0
            for k in range(seglength - 1, -1, -1):
                readtemp += br.readbit() << k
            if seglength == 32:
                readtemp -= 1 << 31
            f = struct.unpack("<f", struct.pack("<I", readtemp & 0xFFFFFFFF))[0]
            if writeindex == -1:
                writeindex += 1
                data[writeindex] = f
            else:
                writeindex += 1
                data[writeindex] = data[writeindex - 1] + f
    return data[:blocksize] if br.ok else []


def decode_sprintz_int32_legacy(enc: bytes) -> List[int]:
    c = ByteCursor(enc)
    out: List[int] = []
    if not _decode_sprintz_int_blocks(c, out):
        return []
    return out


def decode_sprintz_float_legacy(enc: bytes) -> List[float]:
    c = ByteCursor(enc)
    out: List[float] = []
    if not _decode_sprintz_float_blocks(c, out):
        return []
    return out


def decode_rlbe_int32_legacy(enc: bytes) -> List[int]:
    return _rlbe_read_block_int(enc)


def decode_rlbe_float_legacy(enc: bytes) -> List[float]:
    return _rlbe_read_block_float(enc)


# Current codecs execute in C++; *_legacy helpers preserve historical Python APIs.
from .. import unified_rle_native as _unified_rle_native

def encode_rlbe_int32(values: Sequence[int]) -> bytes:
    return _unified_rle_native.encode_values(values, "rlbe", "i32")

def decode_rlbe_int32(blob: bytes) -> List[int]:
    return _unified_rle_native.decode_values(blob, "rlbe", "i32")

def encode_rlbe_float(values: Sequence[float]) -> bytes:
    return _unified_rle_native.encode_values(values, "rlbe", "f32")

def decode_rlbe_float(blob: bytes) -> List[float]:
    return _unified_rle_native.decode_values(blob, "rlbe", "f32")

def encode_sprintz_int32(values: Sequence[int]) -> bytes:
    return _unified_rle_native.encode_values(values, "sprintz", "i32")

def decode_sprintz_int32(blob: bytes) -> List[int]:
    return _unified_rle_native.decode_values(blob, "sprintz", "i32")

def encode_sprintz_float(values: Sequence[float]) -> bytes:
    return _unified_rle_native.encode_values(values, "sprintz", "f32")

def decode_sprintz_float(blob: bytes) -> List[float]:
    return _unified_rle_native.decode_values(blob, "sprintz", "f32")
