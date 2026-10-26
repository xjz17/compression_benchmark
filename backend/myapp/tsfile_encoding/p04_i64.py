"""Current C++ SPRINTZ/RLBE bindings and explicit historical Python helpers."""
from __future__ import annotations

import struct
from typing import List, Sequence

from . import p02, p03_i64
from .p03_i64 import _to_i64
from .common import ByteCursor, append_bytes, append_f32_be, append_int_le_padded
from .p04 import _K_RLBE_BLOCK, _K_SPRINTZ_BLOCK, _K_SPRINTZ_GROUP_MAX, _RlbBitReader

_K_SPRINTZ_BLOCK = 8
_K_SPRINTZ_GROUP_MAX = 16


def _clz_u64(u: int) -> int:
    u = int(u) & 0xFFFFFFFFFFFFFFFF
    if u == 0:
        return 64
    return 64 - u.bit_length()


def _long_value_bit_width(num: int) -> int:
    u = int(num) & 0xFFFFFFFFFFFFFFFF
    if u == 0:
        return 1
    return 64 - _clz_u64(u)


def _i64_add(a: int, b: int) -> int:
    return _to_i64(_to_i64(a) + _to_i64(b))


def _long_max_bit_width(values: Sequence[int]) -> int:
    w = 1
    for num in values:
        w = max(w, _long_value_bit_width(int(num)))
    return w


def _pack8_values_long(width: int, values: Sequence[int], offset: int, buf: bytearray) -> None:
    buf_idx = 0
    value_idx = offset
    left_bit = 0
    while value_idx < 8 + offset:
        buffer = 0
        left_size = 64
        if left_bit > 0:
            buffer |= (_to_i64(values[value_idx]) & 0xFFFFFFFFFFFFFFFF) << (64 - left_bit)
            left_size -= left_bit
            left_bit = 0
            value_idx += 1
        while left_size >= width and value_idx < 8 + offset:
            buffer |= (_to_i64(values[value_idx]) & 0xFFFFFFFFFFFFFFFF) << (left_size - width)
            left_size -= width
            value_idx += 1
        if left_size > 0 and value_idx < 8 + offset:
            shift = width - left_size
            if shift >= 64:
                part = 0
            else:
                part = (_to_i64(values[value_idx]) >> shift) & 0xFFFFFFFFFFFFFFFF
            buffer |= part
            left_bit = width - left_size
        for j in range(8):
            buf[buf_idx] = (buffer >> ((7 - j) * 8)) & 0xFF
            buf_idx += 1
            if buf_idx >= width:
                return


def append_i64_be(out: bytearray, v: int) -> None:
    v = int(v) & 0xFFFFFFFFFFFFFFFF
    for i in range(7, -1, -1):
        out.append((v >> (i * 8)) & 0xFF)


def _long_sprintz_zigzag(pred: int) -> int:
    u = int(pred) & 0xFFFFFFFFFFFFFFFF
    sign = 0xFFFFFFFFFFFFFFFF if _to_i64(pred) < 0 else 0
    return _to_i64(((u << 1) & 0xFFFFFFFFFFFFFFFF) ^ sign)


class _LongFire:
    def __init__(self) -> None:
        self.learn_shift = 2
        self.bit_width = 16
        self.accumulator = 0
        self.delta = 0

    def reset(self) -> None:
        self.accumulator = 0
        self.delta = 0

    def predict(self, value: int) -> int:
        value = _to_i64(value)
        alpha = _to_i64(self.accumulator) >> self.learn_shift
        diff = _to_i64(_to_i64(alpha) * _to_i64(self.delta)) >> self.bit_width
        return _i64_add(value, diff)

    def train(self, pre: int, val: int, err: int) -> None:
        pre = _to_i64(pre)
        val = _to_i64(val)
        err = _to_i64(err)
        gradient = _to_i64(-self.delta if err > 0 else self.delta)
        self.accumulator = _i64_add(self.accumulator, -gradient)
        self.delta = _i64_add(val, -pre)


class _LongSprintzEncoder:
    def __init__(self) -> None:
        self.fire = _LongFire()
        self.values: List[int] = []
        self.is_first_cached = False
        self.group_num = 0
        self.byte_cache = bytearray()

    def fire_residual(self, value: int, pre_value: int) -> int:
        pre = _to_i64(pre_value)
        val = _to_i64(value)
        pr = self.fire.predict(pre)
        err = _i64_add(val, -pr)
        self.fire.train(pre, val, err)
        return err

    def predict_stanza(self, value: int, pre_value: int) -> int:
        return _long_sprintz_zigzag(self.fire_residual(value, pre_value))

    def bit_pack(self) -> None:
        pre_value = self.values[0]
        rest = self.values[1:]
        bw = _long_max_bit_width(rest)
        bytes_arr = bytearray(bw)
        _pack8_values_long(bw, rest, 0, bytes_arr)
        append_int_le_padded(self.byte_cache, bw, 1)
        append_i64_be(self.byte_cache, pre_value)
        append_bytes(self.byte_cache, bytes(bytes_arr))

    def flush_all(self, out: bytearray) -> None:
        if self.byte_cache:
            append_bytes(out, bytes(self.byte_cache))
            self.byte_cache.clear()
        if self.values:
            sz = len(self.values) | (1 << 7)
            append_int_le_padded(out, sz, 1)
            append_bytes(out, p02.encode_rle_int64_legacy(self.values))
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


class _DoubleSprintzEncoder:
    def __init__(self) -> None:
        self.fire = _LongFire()
        self.values: List[float] = []
        self.convert_buffer = [0] * _K_SPRINTZ_BLOCK
        self.is_first_cached = False
        self.group_num = 0
        self.byte_cache = bytearray()

    def fire_residual(self, value: float, pre_value: float) -> int:
        pb = struct.unpack("<Q", struct.pack("<d", float(pre_value)))[0]
        vb = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        prev = _to_i64(pb)
        val = _to_i64(vb)
        pred = self.fire.predict(prev)
        err = _i64_add(val, -pred)
        self.fire.train(prev, val, err)
        return err

    def predict_stanza(self, value: float, pre_value: float) -> int:
        return _long_sprintz_zigzag(self.fire_residual(value, pre_value))

    def bit_pack(self) -> None:
        pre_value = self.values[0]
        bw = _long_max_bit_width(self.convert_buffer)
        bytes_arr = bytearray(bw)
        _pack8_values_long(bw, self.convert_buffer, 0, bytes_arr)
        append_int_le_padded(self.byte_cache, bw, 1)
        append_f64_be(self.byte_cache, pre_value)
        append_bytes(self.byte_cache, bytes(bytes_arr))

    def flush_all(self, out: bytearray) -> None:
        if self.byte_cache:
            append_bytes(out, bytes(self.byte_cache))
            self.byte_cache.clear()
        if self.values:
            sz = len(self.values) | (1 << 7)
            append_int_le_padded(out, sz, 1)
            append_bytes(out, p03_i64.encode_gorilla_double(self.values))
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


def append_f64_be(out: bytearray, d: float) -> None:
    b = struct.unpack("<Q", struct.pack("<d", float(d)))[0]
    append_i64_be(out, b)


def _unpack8_values_long(width: int, buf: bytes, offset: int, values: List[int]) -> None:
    byte_idx = offset
    buffer = 0
    total_bits = 0
    value_idx = 0
    while value_idx < 8:
        while total_bits < width:
            buffer = (buffer << 8) | (buf[byte_idx] & 0xFF)
            byte_idx += 1
            total_bits += 8
        while total_bits >= width and value_idx < 8:
            raw = buffer >> (total_bits - width)
            if width < 64:
                raw &= (1 << width) - 1
            values[value_idx] = _to_i64(raw)
            value_idx += 1
            total_bits -= width
            if total_bits > 0:
                buffer &= (1 << total_bits) - 1
            else:
                buffer = 0


def _inv_sprintz_zigzag(v: int) -> int:
    zz = int(v) & 0xFFFFFFFFFFFFFFFF
    return _to_i64((zz >> 1) ^ (0xFFFFFFFFFFFFFFFF if (zz & 1) else 0))


def _recalc_long_fire(buf: List[int]) -> None:
    for i in range(1, _K_SPRINTZ_BLOCK + 1):
        buf[i] = _inv_sprintz_zigzag(buf[i])
    fire = _LongFire()
    for i in range(1, _K_SPRINTZ_BLOCK + 1):
        pred = fire.predict(buf[i - 1])
        err = _to_i64(buf[i])
        buf[i] = _i64_add(pred, err)
        fire.train(buf[i - 1], buf[i], err)


def _i64_bits_as_double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", _to_i64(bits) & 0xFFFFFFFFFFFFFFFF))[0]


def _recalc_double_fire(pre_value: float, convert_buf: List[int], out_buf: List[float]) -> None:
    out_buf[0] = pre_value
    cb = [_to_i64(convert_buf[i]) for i in range(_K_SPRINTZ_BLOCK)]
    for i in range(_K_SPRINTZ_BLOCK):
        cb[i] = _inv_sprintz_zigzag(cb[i])
    fire = _LongFire()
    pb = struct.unpack("<Q", struct.pack("<d", pre_value))[0]
    prev = _to_i64(pb)
    pred = fire.predict(prev)
    e = cb[0]
    cb[0] = _i64_add(pred, e)
    out_buf[1] = _i64_bits_as_double(cb[0])
    fire.train(prev, cb[0], e)
    for i in range(1, _K_SPRINTZ_BLOCK):
        pred = fire.predict(cb[i - 1])
        e = cb[i]
        cb[i] = _i64_add(pred, e)
        out_buf[i + 1] = _i64_bits_as_double(cb[i])
        fire.train(cb[i - 1], cb[i], e)


def decode_sprintz_int64_legacy(enc: bytes) -> List[int]:
    c = ByteCursor(enc)
    out: List[int] = []
    while c.ok and c.pos < c.len:
        bit_width_byte = c.read_int_le_padded(1)
        if not c.ok:
            return []
        if bit_width_byte & 0x80:
            decode_size = bit_width_byte & ~(1 << 7)
            chunk = bytes(c.p[c.pos : c.len])
            c.pos = c.len
            tail = p02.decode_rle_int64_legacy(chunk)
            if len(tail) != decode_size:
                return []
            out.extend(_to_i64(v) for v in tail)
            break
        pre = c.read_i64_be()
        if not c.ok or not c.has(bit_width_byte):
            return []
        packc = bytes(c.p[c.pos : c.pos + bit_width_byte])
        c.pos += bit_width_byte
        tmp = [0] * 8
        _unpack8_values_long(bit_width_byte, packc, 0, tmp)
        buf = [pre] + tmp
        _recalc_long_fire(buf)
        out.extend(_to_i64(v) for v in buf[: _K_SPRINTZ_BLOCK + 1])
    return out


def decode_sprintz_double_legacy(enc: bytes) -> List[float]:
    c = ByteCursor(enc)
    out: List[float] = []
    while c.ok and c.pos < c.len:
        bit_width_byte = c.read_int_le_padded(1)
        if not c.ok:
            return []
        if bit_width_byte & 0x80:
            decode_size = bit_width_byte & ~(1 << 7)
            chunk = bytes(c.p[c.pos : c.len])
            c.pos = c.len
            tail = p03_i64.decode_gorilla_double(chunk)
            if len(tail) != decode_size:
                return []
            out.extend(tail)
            break
        pre_bits = c.read_i64_be()
        pre = struct.unpack("<d", struct.pack("<q", pre_bits))[0]
        if not c.ok or not c.has(bit_width_byte):
            return []
        packc = bytes(c.p[c.pos : c.pos + bit_width_byte])
        c.pos += bit_width_byte
        tmp = [0] * 8
        _unpack8_values_long(bit_width_byte, packc, 0, tmp)
        buf: List[float] = [0.0] * 9
        _recalc_double_fire(pre, list(tmp), buf)
        out.extend(buf[: _K_SPRINTZ_BLOCK + 1])
    return out


class _LongRlbEncoder:
    def __init__(self) -> None:
        self.block_bits = bytearray()
        self.diff_value: List[int] = [0] * (_K_RLBE_BLOCK + 1)
        self.length_code = [0] * (_K_RLBE_BLOCK + 1)
        self.leng_rle = [0] * (_K_RLBE_BLOCK + 1)
        self.write_index = -1
        self.previous_value = 0
        self.byte_buffer = 0
        self.number_left_in_buffer = 0

    @staticmethod
    def _cal_binary_length(val: int) -> int:
        val = int(val) & 0xFFFFFFFFFFFFFFFF
        if val == 0:
            return 1
        return val.bit_length()

    def write_bit(self, b: bool) -> None:
        self.byte_buffer = (self.byte_buffer << 1) & 0xFF
        if b:
            self.byte_buffer |= 1
        self.number_left_in_buffer += 1
        if self.number_left_in_buffer == 8:
            self.block_bits.append(self.byte_buffer)
            self.number_left_in_buffer = 0
            self.byte_buffer = 0

    def flush_segment(self, i: int) -> None:
        for j in range(6, -1, -1):
            self.write_bit((self.length_code[i] & (1 << j)) != 0)
        fib = self._calc_fibonacci(self.leng_rle[i])
        fiblen = self._cal_binary_length(fib)
        for j in range(fiblen):
            self.write_bit((fib & (1 << j)) != 0)
        self.write_bit(True)
        j = i
        while True:
            temp_difflen = self._cal_binary_length(self.diff_value[j])
            du = int(self.diff_value[j]) & 0xFFFFFFFFFFFFFFFF
            for k in range(temp_difflen - 1, -1, -1):
                self.write_bit((du & (1 << k)) != 0)
            j += 1
            if not (j <= self.write_index and self.leng_rle[j] == 0):
                break

    @staticmethod
    def _calc_fibonacci(val: int) -> int:
        fib = [1, 1]
        i = 2
        while fib[-1] <= val:
            fib.append(fib[-1] + fib[-2])
        i = len(fib) - 2
        valfib = 0
        v = int(val)
        while v > 0:
            while fib[i] > v and i >= 1:
                i -= 1
            valfib |= 1 << (i - 1)
            v -= fib[i]
        return valfib

    def flush_block(self) -> None:
        if self.write_index == -1:
            return
        n = self.write_index + 1
        for i in range(31, -1, -1):
            self.write_bit((n & (1 << i)) != 0)
        i = 0
        while i <= self.write_index:
            j = i
            temprlecal = 0
            while j <= self.write_index and self.length_code[j] == self.length_code[i]:
                j += 1
                temprlecal += 1
            self.leng_rle[i] = temprlecal
            i = j
        for i in range(self.write_index + 1):
            if self.leng_rle[i] > 0:
                self.flush_segment(i)
        if self.number_left_in_buffer:
            self.block_bits.append(self.byte_buffer << (8 - self.number_left_in_buffer))
        self.write_index = -1
        self.byte_buffer = 0
        self.number_left_in_buffer = 0

    def encode_value(self, value: int) -> None:
        if self.write_index == -1:
            self.write_index = 0
            self.diff_value[0] = int(value)
            self.length_code[0] = self._cal_binary_length(value)
            self.previous_value = int(value)
            return
        self.write_index += 1
        self.diff_value[self.write_index] = int(value) - self.previous_value
        self.length_code[self.write_index] = self._cal_binary_length(self.diff_value[self.write_index])
        self.previous_value = int(value)
        if self.write_index == _K_RLBE_BLOCK - 1:
            self.flush_block()


def _rlbe_read_block_long(enc: bytes) -> List[int]:
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
        for j in range(6, -1, -1):
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
                readtemp |= br.readbit() << k
            if writeindex == -1:
                writeindex += 1
                data[writeindex] = readtemp
            else:
                writeindex += 1
                data[writeindex] = data[writeindex - 1] + readtemp
    return data[:blocksize] if br.ok else []


def encode_sprintz_int64_legacy(values: Sequence[int]) -> bytes:
    out = bytearray()
    enc = _LongSprintzEncoder()
    for v in values:
        enc.encode(int(v), out)
    enc.flush_all(out)
    return bytes(out)


def encode_sprintz_double_legacy(values: Sequence[float]) -> bytes:
    out = bytearray()
    enc = _DoubleSprintzEncoder()
    for d in values:
        enc.encode(float(d), out)
    enc.flush_all(out)
    return bytes(out)


def encode_rlbe_int64_legacy(values: Sequence[int]) -> bytes:
    enc = _LongRlbEncoder()
    for v in values:
        enc.encode_value(int(v))
    enc.flush_block()
    return bytes(enc.block_bits)


def encode_rlbe_double_legacy(values: Sequence[float]) -> bytes:
    enc = _LongRlbEncoder()
    for d in values:
        bits = struct.unpack("<Q", struct.pack("<d", float(d)))[0]
        enc.encode_value(bits)
    enc.flush_block()
    return bytes(enc.block_bits)


def decode_rlbe_int64_legacy(enc: bytes) -> List[int]:
    return [_to_i64(v) for v in _rlbe_read_block_long(enc)]


def decode_rlbe_double_legacy(enc: bytes) -> List[float]:
    bits = _rlbe_read_block_long(enc)
    return [struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0] for b in bits]


# Current codecs execute in C++; *_legacy helpers preserve historical Python APIs.
from .. import unified_rle_native as _unified_rle_native

def encode_rlbe_int64(values: Sequence[int]) -> bytes:
    return _unified_rle_native.encode_values(values, "rlbe", "i64")

def decode_rlbe_int64(blob: bytes) -> List[int]:
    return _unified_rle_native.decode_values(blob, "rlbe", "i64")

def encode_rlbe_double(values: Sequence[float]) -> bytes:
    return _unified_rle_native.encode_values(values, "rlbe", "f64")

def decode_rlbe_double(blob: bytes) -> List[float]:
    return _unified_rle_native.decode_values(blob, "rlbe", "f64")

def encode_sprintz_int64(values: Sequence[int]) -> bytes:
    return _unified_rle_native.encode_values(values, "sprintz_reference", "i64")

def decode_sprintz_int64(blob: bytes) -> List[int]:
    return _unified_rle_native.decode_values(blob, "sprintz_reference", "i64")

def encode_sprintz_double(values: Sequence[float]) -> bytes:
    return _unified_rle_native.encode_values(values, "sprintz_reference", "f64")

def decode_sprintz_double(blob: bytes) -> List[float]:
    return _unified_rle_native.decode_values(blob, "sprintz_reference", "f64")
