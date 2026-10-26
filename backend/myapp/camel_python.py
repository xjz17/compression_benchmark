"""
Pure Python port of ``yyy.ts.compress.camel.Camel`` + ``CamelDecompressor`` (double API, float columns
cast to ``double`` like ``ExternalCompressionBench.benchCamelFloat``). Bit stream:
``gr.aueb.delorean.chimp.OutputBitStream`` / ``InputBitStream`` via ``dsi_vigna_bitstream``.
"""
from __future__ import annotations

from .benchmark_wire import column_original_size

import math
import struct
import time
from typing import List, Tuple

from .dsi_vigna_bitstream import InputBitStreamBytes, OutputBitStreamArray

_OUT_STREAM_SIZE = 100_000
_DECIMAL_MAX_COUNT = 3
M_VALUE_BITS = (3, 5, 7, 10, 15)
THRESHOLD = (5, 25, 125, 625)
_POWERS = (1, 10, 100, 1000, 10_000, 100_000)


def _double_to_long_bits(d: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(d)))[0]


def _long_bits_to_double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & ((1 << 64) - 1)))[0]


def _java_round_double(x: float) -> int:
    """``Math.round(double)`` in Java (half away from zero for typical values)."""
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


class CamelCompressor:
    """Port of ``yyy.ts.compress.camel.Camel`` with lossless 64-bit literal fallback."""

    __slots__ = ("stored_val", "stored_double_bits", "first", "size", "_buf", "out")

    def __init__(self) -> None:
        self._buf = bytearray(_OUT_STREAM_SIZE)
        self.out = OutputBitStreamArray(self._buf)
        self.stored_val = 0
        self.stored_double_bits = 0
        self.first = True
        self.size = 0

    @staticmethod
    def _adjusted_first_value(value: float, dc: int, decimal_value: int) -> float:
        sign = -1.0 if value < 0 else 1.0
        p = float(_POWERS[dc])
        return sign * ((int(abs(value)) * _POWERS[dc] + decimal_value) / p)

    @staticmethod
    def _reconstruct_decimal_path(value: float) -> float:
        int_signal = 0 if value < 0 else 1
        int_val = int(value)
        dec = CamelCompressor._cal_decimal_count(value)
        scale = float(_POWERS[dec[0]])
        sign = -1.0 if int_signal == 0 else 1.0
        decimal_val = float(dec[1]) / scale
        decimal_val = _java_round_double(decimal_val * scale) / scale
        return sign * ((int_val * scale + decimal_val * scale) / scale)

    @staticmethod
    def _decimal_path_lossless(value: float) -> bool:
        return _double_to_long_bits(value) == _double_to_long_bits(
            CamelCompressor._reconstruct_decimal_path(value)
        )

    def add_value(self, value: float) -> None:
        if self.first:
            dec = self._cal_decimal_count(value)
            dc = int(dec[0])
            adjusted = self._adjusted_first_value(value, dc, int(dec[1]))
            if _double_to_long_bits(adjusted) == _double_to_long_bits(float(value)):
                self._write_first(_double_to_long_bits(adjusted))
            else:
                self._write_first(_double_to_long_bits(float(value)))
        else:
            self._compress_value(float(value))

    def _write_first(self, value_bits: int) -> None:
        self.first = False
        self.stored_double_bits = value_bits & ((1 << 64) - 1)
        self.stored_val = int(_long_bits_to_double(self.stored_double_bits))
        self.out.write_long(self.stored_double_bits & ((1 << 64) - 1), 64)
        self.size += 64

    def _compress_literal(self, value: float) -> None:
        self.out.write_bit(1)
        bits = _double_to_long_bits(value)
        xor = bits ^ self.stored_double_bits
        self.out.write_int((xor >> 32) & 0xFFFFFFFF, 32)
        self.out.write_int(xor & 0xFFFFFFFF, 32)
        self.stored_double_bits = bits
        self.stored_val = int(value)

    def _compress_value(self, value: float) -> None:
        if not self._decimal_path_lossless(value):
            self._compress_literal(value)
            return
        self.out.write_bit(0)
        int_signal = 0 if value < 0 else 1
        self.size = self._compress_integer_value(int(value), int_signal)
        dec = self._cal_decimal_count(value)
        self.size = self._compress_decimal_value(int(dec[1]), int(dec[0]))
        self.stored_double_bits = _double_to_long_bits(value)

    def close(self) -> None:
        self.out.write_bit(0)
        self.out.flush()

    def get_bytes(self) -> bytes:
        wb = self.out.written_bits
        n = (wb + 7) // 8
        return bytes(self._buf[:n])

    @staticmethod
    def _cal_decimal_count(value: float) -> tuple:
        factor = 1.0
        decimal_count = 0
        value = abs(float(value))
        epsilon = 1e-7
        while abs(value * factor - _java_round_double(value * factor)) > epsilon:
            factor *= 10.0
            decimal_count += 1
        if decimal_count == 0:
            decimal_count = 1
        if 0 < decimal_count <= _DECIMAL_MAX_COUNT:
            dc = decimal_count
            decimal_value = _java_round_double(value * _POWERS[dc]) % _POWERS[dc]
        else:
            dc = _DECIMAL_MAX_COUNT
            decimal_value = _java_round_double(value * _POWERS[dc]) % _POWERS[dc]
        return (dc, decimal_value)

    def _compress_decimal_value(self, decimal_value: int, decimal_count: int) -> int:
        if decimal_count == 0:
            return self.size
        self.out.write_int(decimal_count - 1, 2)
        self.size += 2
        thread = THRESHOLD[decimal_count - 1]
        m = int(decimal_value)
        self.size += 1
        if decimal_value - thread >= 0:
            self.out.write_bit(1)
            m = int(decimal_value % thread)
            p = float(_POWERS[decimal_count])
            xor = _double_to_long_bits(decimal_value / p + 1.0) ^ _double_to_long_bits(
                float(m) / p + 1.0
            )
            self.out.write_long(xor >> (52 - decimal_count), decimal_count)
            self.size += decimal_count
        else:
            self.out.write_bit(0)

        if decimal_count == 1:
            self.out.write_int(m, 3)
            self.size += 3
        elif decimal_count == 2:
            if m < 8:
                self.out.write_int(0, 1)
                self.out.write_int(m, 3)
                self.size += 4
            else:
                self.out.write_int(1, 1)
                self.out.write_int(m, 5)
                self.size += 6
        elif decimal_count == 3:
            if m < 2:
                self.out.write_int(0, 2)
                self.out.write_int(m, 1)
                self.size += 3
            elif m < 8:
                self.out.write_int(1, 2)
                self.out.write_int(m, 3)
                self.size += 5
            elif m < 32:
                self.out.write_int(2, 2)
                self.out.write_int(m, 5)
                self.size += 7
            else:
                self.out.write_int(3, 2)
                self.out.write_int(m, M_VALUE_BITS[decimal_count - 1])
                self.size += 2 + M_VALUE_BITS[decimal_count - 1]
        else:
            if m < 16:
                self.out.write_int(0, 2)
                self.out.write_int(m, 4)
                self.size += 6
            elif m < 64:
                self.out.write_int(1, 2)
                self.out.write_int(m, 6)
                self.size += 8
            elif m < 256:
                self.out.write_int(2, 2)
                self.out.write_int(m, 8)
                self.size += 10
            else:
                self.out.write_int(3, 2)
                self.out.write_int(m, M_VALUE_BITS[decimal_count - 1])
                self.size += 2 + M_VALUE_BITS[decimal_count - 1]
        return self.size

    def _compress_integer_value(self, int_value: int, int_signal: int) -> int:
        diff = int(int_value - self.stored_val)
        self.out.write_int(int_signal, 1)
        self.size += 2
        if -1 <= diff <= 1:
            self.out.write_int(diff + 1, 2)
        else:
            self.out.write_int(3, 2)
            if diff < 0:
                self.out.write_bit(0)
                diff = -diff
            else:
                self.out.write_bit(1)
            self.size += 1
            if 2 <= diff < 8:
                self.out.write_int(0, 1)
                self.out.write_int(diff, 3)
                self.size += 4
            else:
                self.out.write_int(1, 1)
                self.out.write_int(diff, 16)
                self.size += 17
        self.stored_val = int_value
        return self.size


_DEFAULT_BLOCK = 1000


class CamelDecompressor:
    """Port of ``yyy.ts.compress.camel.CamelDecompressor`` (``readPair`` semantics)."""

    __slots__ = ("_in", "stored_val", "stored_double_bits", "current_val", "first", "eos", "read_num")

    def __init__(self, data: bytes) -> None:
        self._in = InputBitStreamBytes(data)
        self.stored_val = 0
        self.stored_double_bits = 0
        self.current_val = 0.0
        self.first = True
        self.eos = False
        self.read_num = 0

    def read_pair(self) -> float | None:
        if self.eos:
            return None
        self.read_num += 1
        if self.first:
            self.first = False
            hi = self._in.read_int(32) & 0xFFFFFFFF
            lo = self._in.read_int(32) & 0xFFFFFFFF
            first_long = (hi << 32) | lo
            first_val = _long_bits_to_double(first_long)
            self.stored_val = int(first_val)
            self.stored_double_bits = first_long
            self.current_val = first_val
        else:
            if self.read_num - 1 == _DEFAULT_BLOCK:
                self.eos = True
                return None
            self.current_val = self._next_value()
        return self.current_val

    def _next_value(self) -> float:
        if self._in.read_bit() == 1:
            hi = self._in.read_int(32) & 0xFFFFFFFF
            lo = self._in.read_int(32) & 0xFFFFFFFF
            xor = (hi << 32) | lo
            bits = xor ^ self.stored_double_bits
            cur = _long_bits_to_double(bits)
            self.stored_double_bits = bits
            self.stored_val = int(cur)
            return cur
        int_val = self._read_int_part()
        dec = self._read_decimal()
        dc = int(dec[0])
        scale = float(_POWERS[dc])
        sign = -1.0 if int_val[0] == 0 else 1.0
        cur = sign * ((int_val[1] * _POWERS[dc] + dec[1] * _POWERS[dc]) / scale)
        self.stored_double_bits = _double_to_long_bits(cur)
        return cur

    def _read_int_part(self) -> tuple:
        int_signal = self._in.read_int(1)
        integer_num = self._in.read_int(2)
        if integer_num < 3:
            diff_val = integer_num - 1
        else:
            diff_symbol = self._in.read_bit()
            rng = self._in.read_int(1)
            if rng == 0:
                diff_val = self._in.read_int(3)
            else:
                diff_val = self._in.read_int(16)
            diff_val = (-1 if diff_symbol == 0 else 1) * diff_val
        self.stored_val = diff_val + self.stored_val
        sig = 1 if self.stored_val >= 0 else 0
        if self.stored_val == 0 and int_signal == 0:
            sig = 0
        return (sig, abs(self.stored_val))

    def _read_decimal(self) -> tuple:
        decimal_count = self._in.read_int(2) + 1
        is_m = self._in.read_int(1)
        xor_string = 0
        if is_m == 1:
            xor = self._in.read_long(decimal_count)
            shifted = xor << (52 - decimal_count)
            xor_string = shifted
        if decimal_count <= 1:
            m_int = self._in.read_int(3)
        elif decimal_count == 2:
            temp = self._in.read_int(1)
            if temp == 0:
                m_int = self._in.read_int(3)
            else:
                m_int = self._in.read_int(5)
        elif decimal_count == 3:
            temp = self._in.read_int(2)
            if temp == 0:
                m_int = self._in.read_int(1)
            elif temp == 1:
                m_int = self._in.read_int(3)
            elif temp == 2:
                m_int = self._in.read_int(5)
            else:
                m_int = self._in.read_int(M_VALUE_BITS[decimal_count - 1])
        else:
            temp = self._in.read_int(2)
            if temp == 0:
                m_int = self._in.read_int(4)
            elif temp == 1:
                m_int = self._in.read_int(6)
            elif temp == 2:
                m_int = self._in.read_int(8)
            else:
                m_int = self._in.read_int(M_VALUE_BITS[decimal_count - 1])

        scale = float(_POWERS[decimal_count])
        if is_m == 1:
            m = float(m_int) / scale + 1.0
            m_prime = _double_to_long_bits(m)
            dec_long = xor_string ^ m_prime
            decimal_val = _long_bits_to_double(dec_long) - 1.0
        else:
            m = float(m_int) / float(_POWERS[decimal_count])
            decimal_val = m
        decimal_val = _java_round_double(decimal_val * scale) / scale
        return (decimal_count, decimal_val)


def bench_camel_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    """Match ``benchCamelFloat``: 1000-value chunks, raw ``double`` equality vs input float."""
    chunk = 1000
    tot_o = tot_c = te = td = 0
    for col in columns:
        n = len(col)
        if n == 0:
            continue
        tot_o += column_original_size(n, 'float')
        try:
            col_c = col_te = col_td = 0
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                c = CamelCompressor()
                t0 = time.perf_counter_ns()
                for i in range(start, end):
                    c.add_value(float(col[i]))
                c.close()
                t1 = time.perf_counter_ns()
                wb = c.out.written_bits
                raw = c._buf
                comp_b = (wb + 7) // 8
                if comp_b <= 0 or comp_b > len(raw):
                    raise ValueError("camel size")
                payload = bytes(raw[:comp_b])
                dec = CamelDecompressor(payload)
                t2 = time.perf_counter_ns()
                for i in range(start, end):
                    v = dec.read_pair()
                    if v is None:
                        raise ValueError("camel early eos")
                    exp = float(col[i])
                    if _double_to_long_bits(v) != _double_to_long_bits(exp):
                        raise ValueError("camel mismatch")
                t3 = time.perf_counter_ns()
                clen = end - start
                col_c += comp_b
                col_te += t1 - t0
                col_td += t3 - t2
            tot_c += col_c
            te += col_te
            td += col_td
        except (ValueError, struct.error, OverflowError):
            continue
    return tot_o, tot_c, te, td
