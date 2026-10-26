"""
Paper Chimp32 (IEEE float columns) — pure Python port matching ``gr.aueb.delorean.chimp.Chimp32``
and ``OutputBitStream`` array mode bit-for-bit with the Java reference used by
``ExternalCompressionBench.benchChimp32Paper`` (optional golden export may build
``external/compression/chimp``; benchmark regen does not require that submodule).

Other “JAR research” codecs (Elf, Camel, Sim-P-piece, SZ/Serf) are **not** reimplemented here:
Elf/Camel/Sim-Piece bundle proprietary layouts across multiple shaded JARs; SZ/Serf compressed
bytes are defined by **libSZ / sfz native binaries**, not by Java bytecode — “bit-identical to
current bench” for those rows implies the **same native output**, which is not obtainable from a
standalone Python reimplementation without embedding or subprocess to those libraries.
"""
from __future__ import annotations

from .benchmark_wire import column_original_size

import struct
import time
from typing import List, Tuple, Union

from .dsi_vigna_bitstream import InputBitStreamBytes, OutputBitStreamArray

# --- Same tables as Chimp32.java ---
THRESHOLD = 5

_LEADING_REPRESENTATION_COMPRESSOR: Tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
)

_LEADING_ROUND: Tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    8,
    8,
    8,
    8,
    12,
    12,
    12,
    12,
    16,
    16,
    18,
    18,
    20,
    20,
    22,
    22,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
)

_LEADING_REPRESENTATION_DECODER: Tuple[int, ...] = (0, 8, 12, 16, 18, 20, 22, 24)
NAN_INT = 0x7FC00000


def _number_of_leading_zeros_uint32(x: int) -> int:
    x &= 0xFFFFFFFF
    if x == 0:
        return 32
    return 32 - x.bit_length()


def _number_of_trailing_zeros_uint32(x: int) -> int:
    x &= 0xFFFFFFFF
    if x == 0:
        return 32
    return (x & -x).bit_length() - 1


class Chimp32Compressor:
    """Port of gr.aueb.delorean.chimp.Chimp32 (float path)."""

    __slots__ = ("out", "stored_leading_zeros", "stored_val", "first", "size", "_buf")

    def __init__(self) -> None:
        self._buf = bytearray(16 * 1024 * 1024)
        self.out = OutputBitStreamArray(self._buf)
        self.stored_leading_zeros = 2147483647  # Integer.MAX_VALUE
        self.stored_val = 0
        self.first = True
        self.size = 0

    def add_value(self, value: float) -> None:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0] & 0xFFFFFFFF
        self.add_raw_int32(bits)

    def add_raw_int32(self, bits: int) -> None:
        bits &= 0xFFFFFFFF
        if self.first:
            self._write_first(bits)
        else:
            self._compress_value(bits)

    def _write_first(self, value: int) -> None:
        self.first = False
        self.stored_val = value & 0xFFFFFFFF
        self.out.write_int(self.stored_val, 32)
        self.size += 32

    def close(self) -> None:
        # Match Float.floatToRawIntBits(Float.NaN) == 0x7fc00000 used as stream terminator.
        self.add_value(struct.unpack("<f", struct.pack("<I", NAN_INT))[0])
        self.out.write_bit(0)
        self.out.flush()

    def _compress_value(self, value: int) -> None:
        value &= 0xFFFFFFFF
        xor = (self.stored_val ^ value) & 0xFFFFFFFF
        if xor == 0:
            self.out.write_bit(0)
            self.out.write_bit(0)
            self.size += 2
            self.stored_leading_zeros = 33
        else:
            lz_idx = _number_of_leading_zeros_uint32(xor)
            leading_zeros = _LEADING_ROUND[lz_idx]
            trailing_zeros = _number_of_trailing_zeros_uint32(xor)

            if trailing_zeros > THRESHOLD:
                significant_bits = 32 - leading_zeros - trailing_zeros
                self.out.write_bit(0)
                self.out.write_bit(1)
                self.out.write_int(_LEADING_REPRESENTATION_COMPRESSOR[leading_zeros], 3)
                self.out.write_int(significant_bits, 5)
                self.out.write_int((xor >> trailing_zeros) & 0xFFFFFFFF, significant_bits)
                self.size += 10 + significant_bits
                self.stored_leading_zeros = 33
            elif leading_zeros == self.stored_leading_zeros:
                self.out.write_bit(1)
                self.out.write_bit(0)
                significant_bits = 32 - leading_zeros
                self.out.write_int(xor & 0xFFFFFFFF, significant_bits)
                self.size += 2 + significant_bits
            else:
                self.stored_leading_zeros = leading_zeros
                significant_bits = 32 - leading_zeros
                self.out.write_int(24 + _LEADING_REPRESENTATION_COMPRESSOR[leading_zeros], 5)
                self.out.write_int(xor & 0xFFFFFFFF, significant_bits)
                self.size += 5 + significant_bits
        self.stored_val = value

    def get_size(self) -> int:
        return self.out.written_bits

    def get_compressed_bytes(self) -> bytes:
        bits = self.out.written_bits
        n = (bits + 7) // 8
        return bytes(self._buf[:n])


class Chimp32Decompressor:
    """Port of gr.aueb.delorean.chimp.ChimpDecompressor32 (sanity check)."""

    __slots__ = ("_in", "stored_leading_zeros", "stored_trailing_zeros", "stored_val", "first", "eos")

    def __init__(self, bs: Union[bytes, InputBitStreamBytes]) -> None:
        self._in = bs if isinstance(bs, InputBitStreamBytes) else InputBitStreamBytes(bs)
        self.stored_leading_zeros = 2147483647
        self.stored_trailing_zeros = 0
        self.stored_val = 0
        self.first = True
        self.eos = False

    def read_pair(self) -> int | None:
        if self.eos:
            return None
        self._next()
        if self.eos:
            return None
        return self.stored_val

    def _next(self) -> None:
        if self.first:
            self.first = False
            self.stored_val = self._in.read_int(32) & 0xFFFFFFFF
            if self.stored_val == NAN_INT:
                self.eos = True
            return
        self._next_value()

    def _next_value(self) -> None:
        inn = self._in
        if inn.read_bit() == 1:
            if inn.read_bit() == 1:
                idx = inn.read_int(3)
                self.stored_leading_zeros = _LEADING_REPRESENTATION_DECODER[idx]
            significant_bits = 32 - self.stored_leading_zeros
            if significant_bits == 0:
                significant_bits = 32
            value = inn.read_int(32 - self.stored_leading_zeros) & 0xFFFFFFFF
            value = (self.stored_val ^ value) & 0xFFFFFFFF
            if value == NAN_INT:
                self.eos = True
                return
            self.stored_val = value
        elif inn.read_bit() == 1:
            idx = inn.read_int(3)
            self.stored_leading_zeros = _LEADING_REPRESENTATION_DECODER[idx]
            significant_bits = inn.read_int(5)
            if significant_bits == 0:
                significant_bits = 32
            self.stored_trailing_zeros = 32 - significant_bits - self.stored_leading_zeros
            value = inn.read_int(32 - self.stored_leading_zeros - self.stored_trailing_zeros) & 0xFFFFFFFF
            value = (value << self.stored_trailing_zeros) & 0xFFFFFFFF
            value = (self.stored_val ^ value) & 0xFFFFFFFF
            if value == NAN_INT:
                self.eos = True
                return
            self.stored_val = value
        # else: two leading 0s — same value as previous (Java compressor used 00).


def decompress_chimp32_float_column(compressed: bytes, n: int) -> List[float]:
    dec = Chimp32Decompressor(compressed)
    out: List[float] = []
    for _ in range(n):
        v = dec.read_pair()
        if v is None:
            raise ValueError("Chimp32 decompress: early EOS")
        out.append(struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0])
    if dec.read_pair() is not None:
        raise ValueError("Chimp32 decompress: expected terminator after n values")
    return out


def bench_chimp32_paper_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    """
    Same aggregation contract as other column benches: sums over columns.
    Returns (orig_bytes, comp_bytes, enc_ns, dec_ns). Skips columns that fail round-trip.
    """
    tot_o = tot_c = te = td = 0
    for col in columns:
        if not col:
            continue
        n = len(col)
        t0 = time.perf_counter_ns()
        c = Chimp32Compressor()
        for f in col:
            c.add_value(float(f))
        c.close()
        raw = c.get_compressed_bytes()
        t1 = time.perf_counter_ns()
        try:
            t2 = time.perf_counter_ns()
            dec = decompress_chimp32_float_column(raw, n)
            t3 = time.perf_counter_ns()
        except ValueError:
            continue
        ok = True
        for i in range(n):
            exp = struct.unpack("<I", struct.pack("<f", float(col[i])))[0] & 0xFFFFFFFF
            got = struct.unpack("<I", struct.pack("<f", dec[i]))[0] & 0xFFFFFFFF
            if exp != got:
                ok = False
                break
        if not ok:
            continue
        tot_o += column_original_size(n, 'float')
        tot_c += len(raw)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td
