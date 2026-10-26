"""
TsFile p02: TS_2DIFF Python port and current C++ URL1 RLE bindings.
Historical RLE helpers remain explicitly named ``*_legacy``.
"""
from __future__ import annotations

import math
import struct
from typing import List, Sequence, Tuple


def _to_i32(value: int) -> int:
    return struct.unpack("<i", struct.pack("<I", int(value) & 0xFFFFFFFF))[0]

_BLOCK_DEFAULT_SIZE = 128
_K_RLE_MIN_REPEATED = 8
_K_RLE_MAX_REPEATED = 0x7FFF
_K_RLE_MAX_BIT_PACKED_GROUPS = 63


def _append_uvarint(out: bytearray, v: int) -> None:
    v = int(v) & 0xFFFFFFFF
    while (v & 0xFFFFFF80) != 0:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)


def _append_u32_be(out: bytearray, v: int) -> None:
    v = int(v)
    out.extend([(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])


def _append_bytes(dst: bytearray, src: bytes) -> None:
    dst.extend(src)


def _clz_u32(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    if x == 0:
        return 32
    return 32 - x.bit_length()


def _int_value_bit_width(num: int) -> int:
    return 32 - _clz_u32(int(num))


def _int_max_bit_width(values: Sequence[int]) -> int:
    w = 1
    for num in values:
        w = max(w, _int_value_bit_width(int(num)))
    return w


def _int_to_bytes(src_num: int, result: bytearray, pos: int, width: int) -> None:
    cnt = pos & 7
    index = pos >> 3
    src_num = int(src_num)
    while width > 0:
        m = (8 - cnt) if (width + cnt >= 8) else width
        width -= m
        mask = 1 << (8 - cnt)
        cnt += m
        y = (src_num >> width) & 0xFF
        y = (y << (8 - cnt)) & 0xFF
        mask = ~(mask - (1 << (8 - cnt))) & 0xFF
        while index >= len(result):
            result.append(0)
        result[index] = (result[index] & mask) | y
        src_num &= ~(-1 << width)
        if cnt == 8:
            index += 1
            cnt = 0


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
            buffer |= (int(values[value_idx]) << (left_size - width))
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


def _append_int_le_padded(out: bytearray, value: int, bit_width: int) -> None:
    padded = (bit_width + 7) // 8
    offset = 0
    value = int(value)
    while padded > 0:
        out.append((value >> offset) & 0xFF)
        offset += 8
        padded -= 1


def encode_int_delta_block(values: Sequence[int]) -> bytes:
    stream = bytearray()
    delta_block = [0] * _BLOCK_DEFAULT_SIZE
    write_index = -1
    first_value = 0
    previous_value = 0
    min_delta_base = 0x7FFFFFFF

    def flush_block() -> None:
        nonlocal write_index, min_delta_base, first_value, previous_value
        if write_index == -1:
            return
        for i in range(write_index):
            delta_block[i] -= min_delta_base
        write_width = 0
        for i in range(write_index):
            u = delta_block[i] & 0xFFFFFFFF
            write_width = max(write_width, 32 - _clz_u32(u))
        _append_u32_be(stream, write_index)
        _append_u32_be(stream, write_width)
        _append_u32_be(stream, min_delta_base)
        _append_u32_be(stream, first_value)
        if write_width > 0:
            enc_len = (write_index * write_width + 7) // 8
            encoding_block = bytearray(enc_len)
            for i in range(write_index):
                _int_to_bytes(delta_block[i], encoding_block, write_width * i, write_width)
            _append_bytes(stream, bytes(encoding_block))
        write_index = -1
        min_delta_base = 0x7FFFFFFF
        first_value = 0
        previous_value = 0

    for value in values:
        value = int(value)
        if write_index == -1:
            write_index = 0
            first_value = value
            previous_value = first_value
            continue
        delta = value - previous_value
        min_delta_base = min(min_delta_base, delta)
        delta_block[write_index] = delta
        write_index += 1
        previous_value = value
        if write_index == _BLOCK_DEFAULT_SIZE:
            flush_block()
    flush_block()
    return bytes(stream)


class _IntRleEncoder:
    def __init__(self) -> None:
        self.bit_width = 1
        self.repeat_count = 0
        self.bit_packed_group_count = 0
        self.num_buffered = 0
        self.is_bit_pack_run = False
        self.is_bit_width_saved = False
        self.pre_value = 0
        self.buffered = [0] * _K_RLE_MIN_REPEATED
        self.bytes_buffer: List[bytearray] = []
        self.byte_cache = bytearray()

    def _clear_buffer(self) -> None:
        for i in range(self.num_buffered, _K_RLE_MIN_REPEATED):
            self.buffered[i] = 0

    def _convert_buffer(self) -> None:
        bytes_arr = bytearray(self.bit_width)
        _pack8_values(self.bit_width, self.buffered, 0, bytes_arr)
        self.bytes_buffer.append(bytes_arr)

    def _end_previous_bit_packed_run(self, last_bit_packed_num: int) -> None:
        if not self.is_bit_pack_run:
            return
        bit_pack_header = (self.bit_packed_group_count << 1) | 1
        self.byte_cache.append(bit_pack_header & 0xFF)
        self.byte_cache.append(last_bit_packed_num & 0xFF)
        for b in self.bytes_buffer:
            _append_bytes(self.byte_cache, bytes(b))
        self.bytes_buffer.clear()
        self.is_bit_pack_run = False
        self.bit_packed_group_count = 0

    def _write_or_append_bit_packed_run(self) -> None:
        if self.bit_packed_group_count >= _K_RLE_MAX_BIT_PACKED_GROUPS:
            self._end_previous_bit_packed_run(_K_RLE_MIN_REPEATED)
        if not self.is_bit_pack_run:
            self.is_bit_pack_run = True
        self._convert_buffer()
        self.num_buffered = 0
        self.repeat_count = 0
        self.bit_packed_group_count += 1

    def _write_rle_run(self) -> None:
        self._end_previous_bit_packed_run(_K_RLE_MIN_REPEATED)
        _append_uvarint(self.byte_cache, self.repeat_count << 1)
        _append_int_le_padded(self.byte_cache, self.pre_value, self.bit_width)
        self.repeat_count = 0
        self.num_buffered = 0

    def _encode_value(self, value: int) -> None:
        value = int(value)
        if not self.is_bit_width_saved:
            self.byte_cache.append(self.bit_width & 0xFF)
            self.is_bit_width_saved = True
        if value == self.pre_value:
            self.repeat_count += 1
            if _K_RLE_MIN_REPEATED <= self.repeat_count <= _K_RLE_MAX_REPEATED:
                return
            if self.repeat_count == _K_RLE_MAX_REPEATED + 1:
                self.repeat_count = _K_RLE_MAX_REPEATED
                self._write_rle_run()
                self.repeat_count = 1
                self.pre_value = value
        else:
            if self.repeat_count >= _K_RLE_MIN_REPEATED:
                self._write_rle_run()
            self.repeat_count = 1
            self.pre_value = value
        self.buffered[self.num_buffered] = value
        self.num_buffered += 1
        if self.num_buffered == _K_RLE_MIN_REPEATED:
            self._write_or_append_bit_packed_run()

    def flush(self, values: Sequence[int]) -> bytes:
        vals = [int(v) for v in values]
        self.bit_width = _int_max_bit_width(vals)
        self.byte_cache.clear()
        self.num_buffered = 0
        self.repeat_count = 0
        self.bit_packed_group_count = 0
        self.bytes_buffer.clear()
        self.is_bit_pack_run = False
        self.is_bit_width_saved = False
        self.pre_value = 0
        for v in vals:
            self._encode_value(v)
        last_bit_packed_num = self.num_buffered
        if self.repeat_count >= _K_RLE_MIN_REPEATED:
            self._write_rle_run()
        elif self.num_buffered > 0:
            self._clear_buffer()
            self._write_or_append_bit_packed_run()
            self._end_previous_bit_packed_run(last_bit_packed_num)
        else:
            self._end_previous_bit_packed_run(_K_RLE_MIN_REPEATED)
        out = bytearray()
        _append_uvarint(out, len(self.byte_cache))
        _append_bytes(out, bytes(self.byte_cache))
        return bytes(out)


class _LongRleEncoder(_IntRleEncoder):
    def _convert_buffer(self) -> None:
        bytes_arr = bytearray(self.bit_width)
        _pack8_values_long(self.bit_width, self.buffered, 0, bytes_arr)
        self.bytes_buffer.append(bytes_arr)

    def flush(self, values: Sequence[int]) -> bytes:
        vals = [int(v) for v in values]
        self.bit_width = _long_max_bit_width(vals)
        self.byte_cache.clear()
        self.num_buffered = 0
        self.repeat_count = 0
        self.bit_packed_group_count = 0
        self.bytes_buffer.clear()
        self.is_bit_pack_run = False
        self.is_bit_width_saved = False
        self.pre_value = 0
        for v in vals:
            self._encode_value(v)
        last_bit_packed_num = self.num_buffered
        if self.repeat_count >= _K_RLE_MIN_REPEATED:
            self._write_rle_run()
        elif self.num_buffered > 0:
            self._clear_buffer()
            self._write_or_append_bit_packed_run()
            self._end_previous_bit_packed_run(last_bit_packed_num)
        else:
            self._end_previous_bit_packed_run(_K_RLE_MIN_REPEATED)
        out = bytearray()
        _append_uvarint(out, len(self.byte_cache))
        _append_bytes(out, bytes(self.byte_cache))
        return bytes(out)


def _sign_extend(value: int, bit_width: int) -> int:
    """TsFile little-endian padded fields are ``bit_width``-bit signed."""
    if bit_width <= 0:
        return int(value)
    if bit_width >= 64:
        v = int(value) & 0xFFFFFFFFFFFFFFFF
        if v & (1 << 63):
            return v - (1 << 64)
        return v
    if bit_width >= 32:
        return _to_i32(value)
    mask = (1 << bit_width) - 1
    v = int(value) & mask
    if v & (1 << (bit_width - 1)):
        v -= 1 << bit_width
    return v


def _long_max_bit_width(values: Sequence[int]) -> int:
    w = 1
    for num in values:
        u = int(num) & 0xFFFFFFFFFFFFFFFF
        if u == 0:
            w = max(w, 1)
        else:
            w = max(w, u.bit_length())
    return w


def _pack8_values_long(width: int, values: Sequence[int], offset: int, buf: bytearray) -> None:
    buf_idx = 0
    value_idx = offset
    left_bit = 0
    while value_idx < 8 + offset:
        buffer = 0
        left_size = 64
        if left_bit > 0:
            buffer |= (int(values[value_idx]) & 0xFFFFFFFFFFFFFFFF) << (64 - left_bit)
            left_size -= left_bit
            left_bit = 0
            value_idx += 1
        while left_size >= width and value_idx < 8 + offset:
            buffer |= (int(values[value_idx]) & 0xFFFFFFFFFFFFFFFF) << (left_size - width)
            left_size -= width
            value_idx += 1
        if left_size > 0 and value_idx < 8 + offset:
            buffer |= (int(values[value_idx]) & 0xFFFFFFFFFFFFFFFF) >> (width - left_size)
            left_bit = width - left_size
        for j in range(8):
            buf[buf_idx] = (buffer >> ((7 - j) * 8)) & 0xFF
            buf_idx += 1
            if buf_idx >= width:
                return


def _unpack8_values_long(width: int, buf: bytes, offset: int, values: List[int]) -> None:
    from .p03_i64 import _to_i64

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


def _read_uvarint(data: bytes, pos: int) -> Tuple[int, int]:
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


def _bytes_to_int_at_bits(data: bytes, pos_bits: int, width: int) -> int:
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
            cnt = 0
            index += 1
    return ret


def _unpack8_values(width: int, buf: bytes, offset: int, values: List[int]) -> None:
    from .common import to_i32

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
            raw = buffer >> (total_bits - width)
            if width < 32:
                raw &= (1 << width) - 1
            values[value_idx] = to_i32(raw)
            value_idx += 1
            total_bits -= width
            if total_bits > 0:
                buffer &= (1 << total_bits) - 1
            else:
                buffer = 0


class _IntRleDecoder:
    def __init__(self, enc: bytes) -> None:
        self.outer = enc
        self.outer_pos = 0
        self.outer_ok = True
        self.pattern_bytes = b""
        self.inner_pos = 0
        self.inner_ok = True
        self.bit_width = 0
        self.current_count = 0
        self.current_value = 0
        self.current_buffer: List[int] = []
        self.bit_packing_num = 0
        self.pkg_header_read = False
        self.mode = "rle"

    def _read_length_and_width(self) -> bool:
        length, pos = _read_uvarint(self.outer, self.outer_pos)
        if pos < 0 or length == 0:
            self.outer_ok = False
            return False
        if self.outer_pos + length > len(self.outer):
            self.outer_ok = False
            return False
        self.pattern_bytes = self.outer[pos : pos + length]
        self.outer_pos = pos + length
        self.inner_pos = 0
        if len(self.pattern_bytes) < 1:
            self.inner_ok = False
            return False
        self.bit_width = self.pattern_bytes[0]
        self.inner_pos = 1
        self.pkg_header_read = True
        self.current_buffer.clear()
        return True

    def _read_number_rle(self) -> bool:
        from .common import to_i32

        n = (self.bit_width + 7) // 8
        if self.inner_pos + n > len(self.pattern_bytes):
            self.inner_ok = False
            return False
        result = 0
        offset = 0
        for _ in range(n):
            ch = self.pattern_bytes[self.inner_pos]
            self.inner_pos += 1
            result += ch << offset
            offset += 8
        if self.bit_width < 32:
            result &= (1 << self.bit_width) - 1
        self.current_value = to_i32(result)
        return True

    def _unpack_all_bitpacked(self, bit_packed_group_count: int, last_bit_packed_num: int) -> None:
        total_count = (bit_packed_group_count - 1) * _K_RLE_MIN_REPEATED + last_bit_packed_num
        self.bit_packing_num = total_count
        self.current_buffer = [0] * total_count
        bytes_to_read = bit_packed_group_count * self.bit_width
        if self.inner_pos + bytes_to_read > len(self.pattern_bytes):
            self.inner_ok = False
            return
        k = 0
        idx = self.inner_pos
        while k < total_count:
            tv = [0] * 8
            _unpack8_values(self.bit_width, self.pattern_bytes, idx, tv)
            for t in range(8):
                if k >= total_count:
                    break
                self.current_buffer[k] = tv[t]
                k += 1
            idx += self.bit_width
        self.inner_pos = idx

    def _read_next(self) -> bool:
        header, pos = _read_uvarint(self.pattern_bytes, self.inner_pos)
        if pos < 0:
            self.inner_ok = False
            return False
        self.inner_pos = pos
        if (header & 1) == 0:
            self.mode = "rle"
            self.current_count = header >> 1
            return self._read_number_rle()
        self.mode = "bitpacked"
        bit_packed_group_count = header >> 1
        if self.inner_pos >= len(self.pattern_bytes):
            self.inner_ok = False
            return False
        last_bit_packed_num = self.pattern_bytes[self.inner_pos]
        self.inner_pos += 1
        self._unpack_all_bitpacked(bit_packed_group_count, last_bit_packed_num)
        self.current_count = self.bit_packing_num
        return self.inner_ok

    def _has_next_pkg(self) -> bool:
        return self.current_count > 0 or self.inner_pos < len(self.pattern_bytes)

    def read_int(self) -> int:
        if not self.pkg_header_read:
            if not self._read_length_and_width():
                self.outer_ok = False
                return 0
        if self.current_count == 0:
            if not self._read_next():
                self.outer_ok = False
                return 0
        self.current_count -= 1
        if self.mode == "rle":
            result = self.current_value
        else:
            result = self.current_buffer[self.bit_packing_num - self.current_count - 1]
        if not self._has_next_pkg():
            self.pkg_header_read = False
        return result

    def _has_more_outer(self) -> bool:
        return self.outer_pos < len(self.outer) and self.outer_ok


class _LongRleDecoder(_IntRleDecoder):
    def _read_number_rle(self) -> bool:
        from .p03_i64 import _to_i64

        n = (self.bit_width + 7) // 8
        if self.inner_pos + n > len(self.pattern_bytes):
            self.inner_ok = False
            return False
        result = 0
        offset = 0
        for _ in range(n):
            ch = self.pattern_bytes[self.inner_pos]
            self.inner_pos += 1
            result += ch << offset
            offset += 8
        if self.bit_width < 64:
            result &= (1 << self.bit_width) - 1
        self.current_value = _to_i64(result)
        return True

    def _unpack_all_bitpacked(self, bit_packed_group_count: int, last_bit_packed_num: int) -> None:
        total_count = (bit_packed_group_count - 1) * _K_RLE_MIN_REPEATED + last_bit_packed_num
        self.bit_packing_num = total_count
        self.current_buffer = [0] * total_count
        bytes_to_read = bit_packed_group_count * self.bit_width
        if self.inner_pos + bytes_to_read > len(self.pattern_bytes):
            self.inner_ok = False
            return
        k = 0
        idx = self.inner_pos
        while k < total_count:
            tv = [0] * 8
            _unpack8_values_long(self.bit_width, self.pattern_bytes, idx, tv)
            for t in range(8):
                if k >= total_count:
                    break
                self.current_buffer[k] = tv[t]
                k += 1
            idx += self.bit_width
        self.inner_pos = idx


def decode_rle_int32_legacy(enc: bytes) -> List[int]:
    out: List[int] = []
    dec = _IntRleDecoder(enc)
    while dec._has_more_outer():
        if not dec.pkg_header_read:
            if not dec._read_length_and_width():
                return []
        while dec._has_next_pkg():
            out.append(dec.read_int())
            if not dec.inner_ok:
                return []
        dec.pkg_header_read = False
    if not dec.outer_ok or dec.outer_pos != len(dec.outer):
        return []
    return out


def encode_rle_int32_legacy(values: Sequence[int]) -> bytes:
    return _IntRleEncoder().flush(values)


def convert_float_to_int(value: float, max_point_number: int, max_point_value: float) -> int:
    if value * max_point_value > 2147483647.0 or value * max_point_value < -2147483648.0:
        if value > 2147483647.0 or value < -2147483648.0:
            return 0xFFFFFFFF
        return int(math.floor(float(value) + 0.5)) if float(value) >= 0 else int(math.ceil(float(value) - 0.5))
    if math.isnan(value):
        return 0xFFFFFFFF
    return int(math.floor(float(value) * max_point_value + 0.5)) if float(value) >= 0 else int(
        math.ceil(float(value) * max_point_value - 0.5)
    )


def encode_rle_float_legacy(values: Sequence[float], max_point_number: int = 0) -> bytes:
    max_point = max(0, int(max_point_number))
    max_point_value = 10.0**max_point
    ints = [convert_float_to_int(float(f), max_point, max_point_value) for f in values]
    out = bytearray()
    _append_uvarint(out, max_point)
    body = encode_rle_int32_legacy(ints)
    _append_bytes(out, body)
    return bytes(out)


def decode_rle_float_legacy(blob: bytes) -> List[float]:
    max_point, pos = _read_uvarint(blob, 0)
    if pos < 0:
        return []
    ints = decode_rle_int32_legacy(blob[pos:])
    if not ints:
        return []
    if max_point <= 0:
        return [float(v) for v in ints]
    scale = 10.0 ** int(max_point)
    return [float(v) / scale for v in ints]


def encode_ts_2diff_int32(values: Sequence[int]) -> bytes:
    return encode_int_delta_block(values)


def encode_ts_2diff_float(values: Sequence[float], max_point_number: int = 0) -> bytes:
    max_point = max(0, int(max_point_number))
    max_point_value = 10.0**max_point
    ints = [convert_float_to_int(float(f), max_point, max_point_value) for f in values]
    out = bytearray()
    _append_uvarint(out, max_point)
    _append_bytes(out, encode_int_delta_block(ints))
    return bytes(out)


def decode_ts_2diff_int32(blob: bytes) -> List[int]:
    out: List[int] = []
    pos = 0
    n = len(blob)
    while pos < n:
        if pos + 16 > n:
            return []
        pack_num = struct.unpack(">i", blob[pos : pos + 4])[0]
        pos += 4
        pack_width = struct.unpack(">i", blob[pos : pos + 4])[0]
        pos += 4
        min_delta_base = struct.unpack(">i", blob[pos : pos + 4])[0]
        pos += 4
        first_value = struct.unpack(">i", blob[pos : pos + 4])[0]
        pos += 4
        enc_len = ((pack_num * pack_width) + 7) // 8
        if pos + enc_len > n:
            return []
        delta_buf = blob[pos : pos + enc_len]
        pos += enc_len
        out.append(first_value)
        previous = first_value
        for i in range(pack_num):
            v = _bytes_to_int_at_bits(delta_buf, pack_width * i, pack_width)
            cur = previous + min_delta_base + v
            out.append(cur)
            previous = cur
    return out


def decode_ts_2diff_float(blob: bytes) -> List[float]:
    max_point, pos = _read_uvarint(blob, 0)
    if pos < 0:
        return []
    ints = decode_ts_2diff_int32(blob[pos:])
    if not ints:
        return []
    if max_point <= 0:
        return [float(v) for v in ints]
    scale = 10.0 ** int(max_point)
    return [float(v) / scale for v in ints]


def _clz_u64(x: int) -> int:
    x = int(x) & ((1 << 64) - 1)
    if x == 0:
        return 64
    return 64 - x.bit_length()


def encode_int64_delta_block(values: Sequence[int]) -> bytes:
    stream = bytearray()
    delta_block = [0] * _BLOCK_DEFAULT_SIZE
    write_index = -1
    first_value = 0
    previous_value = 0
    min_delta_base = (1 << 63) - 1

    def flush_block() -> None:
        nonlocal write_index, min_delta_base, first_value, previous_value
        if write_index == -1:
            return
        for i in range(write_index):
            delta_block[i] -= min_delta_base
        write_width = 0
        for i in range(write_index):
            u = int(delta_block[i]) & ((1 << 64) - 1)
            write_width = max(write_width, 64 - _clz_u64(u))
        _append_u32_be(stream, write_index)
        _append_u32_be(stream, write_width)
        stream.extend(struct.pack(">q", min_delta_base))
        stream.extend(struct.pack(">q", first_value))
        if write_width > 0:
            enc_len = (write_index * write_width + 7) // 8
            encoding_block = bytearray(enc_len)
            for i in range(write_index):
                _int_to_bytes(int(delta_block[i]), encoding_block, write_width * i, write_width)
            _append_bytes(stream, bytes(encoding_block))
        write_index = -1
        min_delta_base = (1 << 63) - 1
        first_value = 0
        previous_value = 0

    for value in values:
        value = int(value)
        if write_index == -1:
            write_index = 0
            first_value = value
            previous_value = first_value
            continue
        delta = value - previous_value
        min_delta_base = min(min_delta_base, delta)
        delta_block[write_index] = delta
        write_index += 1
        previous_value = value
        if write_index == _BLOCK_DEFAULT_SIZE:
            flush_block()
    flush_block()
    return bytes(stream)


def convert_double_to_long(value: float, max_point_number: int, max_point_value: float) -> int:
    v = float(value)
    if v * max_point_value > float((1 << 63) - 1) or v * max_point_value < float(-(1 << 63)):
        if v > float((1 << 63) - 1) or v < float(-(1 << 63)):
            return struct.unpack("<q", struct.pack("<d", v))[0]
        return int(math.floor(v + 0.5)) if v >= 0 else int(math.ceil(v - 0.5))
    if math.isnan(v):
        return struct.unpack("<q", struct.pack("<d", v))[0]
    return int(math.floor(v * max_point_value + 0.5)) if v >= 0 else int(math.ceil(v * max_point_value - 0.5))


_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


def _int64_delta_chain_fits(longs: Sequence[int]) -> bool:
    """True when scaled values and every consecutive delta fit signed int64 (p02 wire format)."""
    prev: int | None = None
    for raw in longs:
        v = int(raw)
        if v < _I64_MIN or v > _I64_MAX:
            return False
        if prev is not None:
            delta = v - prev
            if delta < _I64_MIN or delta > _I64_MAX:
                return False
        prev = v
    return True


def effective_max_point_for_i64_chain(
    values: Sequence[float], requested_max_point: int
) -> int:
    """
    Reduce ``max_point_number`` until TsFile TS_2DIFF/RLE int64 delta blocks can encode the series.

    Catalog ``column_info`` may record many fractional digits (e.g. POI-lat with no CSV header);
    scaling by 10^19 can make a single latitude step exceed int64 delta range.
    """
    mp = max(0, int(requested_max_point))
    while mp > 0:
        scale = 10.0**mp
        longs = [convert_double_to_long(float(f), mp, scale) for f in values]
        if _int64_delta_chain_fits(longs):
            return mp
        mp -= 1
    return 0


def encode_ts_2diff_int64(values: Sequence[int]) -> bytes:
    return encode_int64_delta_block(values)


def encode_ts_2diff_double(values: Sequence[float], max_point_number: int = 0) -> bytes:
    max_point = effective_max_point_for_i64_chain(values, max_point_number)
    max_point_value = 10.0**max_point
    longs = [convert_double_to_long(float(f), max_point, max_point_value) for f in values]
    out = bytearray()
    _append_uvarint(out, max_point)
    _append_bytes(out, encode_int64_delta_block(longs))
    return bytes(out)


def encode_rle_int64_legacy(values: Sequence[int]) -> bytes:
    return _LongRleEncoder().flush([int(x) for x in values])


def encode_rle_double_legacy(values: Sequence[float], max_point_number: int = 0) -> bytes:
    max_point = effective_max_point_for_i64_chain(values, max_point_number)
    max_point_value = 10.0**max_point
    longs = [convert_double_to_long(float(f), max_point, max_point_value) for f in values]
    out = bytearray()
    _append_uvarint(out, max_point)
    _append_bytes(out, encode_rle_int64_legacy(longs))
    return bytes(out)


def decode_ts_2diff_int64(blob: bytes) -> List[int]:
    out: List[int] = []
    pos = 0
    n = len(blob)
    while pos < n:
        if pos + 24 > n:
            return []
        pack_num = struct.unpack(">i", blob[pos : pos + 4])[0]
        pos += 4
        pack_width = struct.unpack(">i", blob[pos : pos + 4])[0]
        pos += 4
        min_delta_base = struct.unpack(">q", blob[pos : pos + 8])[0]
        pos += 8
        first_value = struct.unpack(">q", blob[pos : pos + 8])[0]
        pos += 8
        enc_len = ((pack_num * pack_width) + 7) // 8
        if pos + enc_len > n:
            return []
        delta_buf = blob[pos : pos + enc_len]
        pos += enc_len
        out.append(first_value)
        previous = first_value
        for i in range(pack_num):
            v = _bytes_to_int_at_bits(delta_buf, pack_width * i, pack_width)
            cur = previous + min_delta_base + v
            out.append(cur)
            previous = cur
    return out


def decode_ts_2diff_double(blob: bytes) -> List[float]:
    max_point, pos = _read_uvarint(blob, 0)
    if pos < 0:
        return []
    longs = decode_ts_2diff_int64(blob[pos:])
    if not longs:
        return []
    if max_point <= 0:
        return [float(v) for v in longs]
    scale = 10.0 ** int(max_point)
    return [float(v) / scale for v in longs]


def decode_rle_int64_legacy(blob: bytes) -> List[int]:
    out: List[int] = []
    dec = _LongRleDecoder(blob)
    while dec._has_more_outer():
        if not dec.pkg_header_read:
            if not dec._read_length_and_width():
                return []
        while dec._has_next_pkg():
            out.append(dec.read_int())
            if not dec.inner_ok:
                return []
        dec.pkg_header_read = False
    if not dec.outer_ok or dec.outer_pos != len(dec.outer):
        return []
    return out


def decode_rle_double_legacy(blob: bytes) -> List[float]:
    max_point, pos = _read_uvarint(blob, 0)
    if pos < 0:
        return []
    longs = decode_rle_int64_legacy(blob[pos:])
    if not longs:
        return []
    if max_point <= 0:
        return [float(v) for v in longs]
    scale = 10.0 ** int(max_point)
    return [float(v) / scale for v in longs]


# Current codecs execute in C++; *_legacy helpers preserve historical Python APIs.
from .. import unified_rle_native as _unified_rle_native

def encode_rle_int32(values: Sequence[int]) -> bytes:
    return _unified_rle_native.encode_values(values, "rle", "i32")

def decode_rle_int32(blob: bytes) -> List[int]:
    return _unified_rle_native.decode_values(blob, "rle", "i32")

def encode_rle_int64(values: Sequence[int]) -> bytes:
    return _unified_rle_native.encode_values(values, "rle", "i64")

def decode_rle_int64(blob: bytes) -> List[int]:
    return _unified_rle_native.decode_values(blob, "rle", "i64")

def encode_rle_float(values: Sequence[float], max_point_number: int = 0) -> bytes:
    return _unified_rle_native.encode_scaled_values(values, "f32", max_point_number)

def decode_rle_float(blob: bytes) -> List[float]:
    return _unified_rle_native.decode_values(blob, "rle", "f32")

def encode_rle_double(values: Sequence[float], max_point_number: int = 0) -> bytes:
    return _unified_rle_native.encode_scaled_values(values, "f64", max_point_number)

def decode_rle_double(blob: bytes) -> List[float]:
    return _unified_rle_native.decode_values(blob, "rle", "f64")
