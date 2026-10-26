"""
Elf32 float paths from ``external/compression/elf``: erasing codec + ``ElfXORCompressor32`` or Chimp32 XOR,
matching ``ExternalCompressionBench.benchElfVariant32Float`` (2048-value chunks).
"""
from __future__ import annotations

from .benchmark_wire import column_original_size

import math
import struct
import time
from typing import Callable, List, Optional, Tuple

from .chimp32_python import Chimp32Compressor, Chimp32Decompressor
from .dsi_vigna_bitstream import InputBitStreamBytes, OutputBitStreamArray
from .elf32_utils import get_alpha_and_beta_star, get_f_alpha, get_sp, get10i_n, round_up

END_SIGN = 0x7FC00000
MAX_BETA = 2147483647

_XOR_LR = (
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
)
_XOR_ROUND = (
    0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
)
_XOR_DEC_LR = (0, 6, 10, 12, 14, 16, 18, 20)


def _nlz(x: int) -> int:
    x &= 0xFFFFFFFF
    return 32 if x == 0 else 32 - x.bit_length()


def _ntz(x: int) -> int:
    x &= 0xFFFFFFFF
    return 32 if x == 0 else (x & -x).bit_length() - 1


def _raw_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(f)))[0] & 0xFFFFFFFF


def _bits_to_float(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


class ElfXORCompressor32:
    __slots__ = ("out", "sl", "st", "sv", "first", "_buf")

    def __init__(self, buf: Optional[bytearray] = None) -> None:
        # 2048-float chunks can exceed 5 KiB in worst-case bit patterns; keep headroom for Elf+.
        self._buf = buf if buf is not None else bytearray(512 * 1024)
        self.out = OutputBitStreamArray(self._buf)
        self.sl = MAX_BETA
        self.st = MAX_BETA
        self.sv = 0
        self.first = True

    def add_raw(self, value: int) -> None:
        v = value & 0xFFFFFFFF
        if self.first:
            self.first = False
            self.sv = v
            tz = _ntz(v)
            self.out.write_int(tz, 6)
            if tz < 32:
                self.out.write_long((self.sv >> (tz + 1)) & ((1 << (31 - tz)) - 1), 31 - tz)
            return
        xor = (self.sv ^ v) & 0xFFFFFFFF
        if xor == 0:
            self.out.write_int(1, 2)
            return
        lz = _XOR_ROUND[_nlz(xor)]
        tz = _ntz(xor)
        if lz == self.sl and tz >= self.st:
            cb = 32 - self.sl - self.st
            ln = 2 + cb
            if ln > 32:
                self.out.write_int(0, 2)
                self.out.write_int((xor >> self.st) & 0xFFFFFFFF, cb)
            else:
                self.out.write_int((xor >> self.st) & ((1 << ln) - 1), ln)
        else:
            self.sl, self.st = lz, tz
            cb = 32 - self.sl - self.st
            if cb <= 8:
                w = (((0x2 << 3) | _XOR_LR[self.sl]) << 3) | (cb & 0x7)
                self.out.write_int(w, 8)
                self.out.write_int((xor >> (self.st + 1)) & ((1 << max(0, cb - 1)) - 1), max(0, cb - 1))
            else:
                w = (((0x3 << 3) | _XOR_LR[self.sl]) << 5) | (cb & 0x1F)
                self.out.write_int(w, 10)
                self.out.write_int((xor >> (self.st + 1)) & ((1 << max(0, cb - 1)) - 1), max(0, cb - 1))
        self.sv = v

    def close_xor(self) -> None:
        self.add_raw(END_SIGN)
        self.out.write_bit(0)
        self.out.flush()

    def get_out(self) -> bytes:
        wb = self.out.written_bits
        return bytes(self._buf[: (wb + 7) // 8])


class ElfXORDecompressor32:
    __slots__ = ("inn", "sv", "sl", "st", "first", "eos")

    def __init__(self, inn: InputBitStreamBytes) -> None:
        self.inn = inn
        self.sv = 0
        self.sl = MAX_BETA
        self.st = MAX_BETA
        self.first = True
        self.eos = False

    def read_value(self) -> Optional[int]:
        if self.eos:
            return None
        if self.first:
            self.first = False
            tz = self.inn.read_int(6)
            if tz < 32:
                mid = self.inn.read_int(31 - tz)
                self.sv = (((mid << 1) + 1) << tz) & 0xFFFFFFFF
            else:
                self.sv = 0
            if self.sv == END_SIGN:
                self.eos = True
                return None
            return self.sv
        fl = self.inn.read_int(2)
        if fl == 3:
            lc = self.inn.read_int(8)
            self.sl = _XOR_DEC_LR[lc >> 5]
            cb = lc & 0x1F
            if cb == 0:
                cb = 32
            self.st = 32 - self.sl - cb
            mid = self.inn.read_int(cb - 1)
            val = (((mid << 1) + 1) << self.st) & 0xFFFFFFFF
            val = (self.sv ^ val) & 0xFFFFFFFF
            if val == END_SIGN:
                self.eos = True
                return None
            self.sv = val
            return self.sv
        if fl == 2:
            lc = self.inn.read_int(6)
            self.sl = _XOR_DEC_LR[lc >> 3]
            cb = lc & 0x7
            if cb == 0:
                cb = 8
            self.st = 32 - self.sl - cb
            mid = self.inn.read_int(cb - 1)
            val = (((mid << 1) + 1) << self.st) & 0xFFFFFFFF
            val = (self.sv ^ val) & 0xFFFFFFFF
            if val == END_SIGN:
                self.eos = True
                return None
            self.sv = val
            return self.sv
        if fl == 1:
            return self.sv
        cb = 32 - self.sl - self.st
        mid = self.inn.read_int(cb)
        val = ((mid << self.st) & 0xFFFFFFFF) ^ self.sv
        val &= 0xFFFFFFFF
        if val == END_SIGN:
            self.eos = True
            return None
        self.sv = val
        return self.sv


def _elf_erase_prefix_write(
    out: OutputBitStreamArray, v: float, last_beta: List[int]
) -> Tuple[int, int]:
    v_int = _raw_bits(v)
    v_prime = v_int
    extra = 0
    if v == 0.0 or math.isinf(v):
        out.write_int(2, 2)
        extra = 2
        v_prime = v_int
    elif math.isnan(v):
        out.write_int(2, 2)
        extra = 2
        v_prime = 0x7FC00000
    else:
        ab = get_alpha_and_beta_star(v, last_beta[0])
        e = (v_int >> 23) & 0xFF
        g_alpha = get_f_alpha(ab[0]) + e - 127
        erase_bits = 23 - g_alpha
        # Java ``int`` shift distance is masked with 0x1f.
        es = erase_bits & 31
        mask = ((0xFFFFFFFF << es) & 0xFFFFFFFF)
        delta = ((~mask) & 0xFFFFFFFF) & v_int
        if delta != 0 and erase_bits > 3:
            if ab[1] == last_beta[0]:
                out.write_bit(0)
                extra = 1
            else:
                out.write_int(ab[1] | 0x18, 5)
                extra = 5
                last_beta[0] = ab[1]
            v_prime = mask & v_int
        else:
            out.write_int(2, 2)
            extra = 2
            v_prime = v_int
    return v_prime, extra


def _elf_next_decoded(
    inn: InputBitStreamBytes, xor_read: Callable[[], Optional[int]], last_beta: List[int]
) -> Optional[float]:
    b0 = inn.read_int(1)
    if b0 == 0:
        v_prime_b = xor_read()
        if v_prime_b is None:
            return None
        v_prime = _bits_to_float(v_prime_b)
        sp = get_sp(abs(v_prime))
        if last_beta[0] == 0:
            vv = get10i_n(-sp - 1)
            if v_prime < 0:
                vv = -vv
        else:
            alpha = last_beta[0] - sp - 1
            vv = round_up(v_prime, alpha)
        return vv
    b1 = inn.read_int(1)
    if b1 == 0:
        vb = xor_read()
        if vb is None:
            return None
        return _bits_to_float(vb)
    last_beta[0] = inn.read_int(3)
    v_prime_b = xor_read()
    if v_prime_b is None:
        return None
    v_prime = _bits_to_float(v_prime_b)
    sp = get_sp(abs(v_prime))
    if last_beta[0] == 0:
        vv = get10i_n(-sp - 1)
        if v_prime < 0:
            vv = -vv
    else:
        alpha = last_beta[0] - sp - 1
        vv = round_up(v_prime, alpha)
    return vv


class ElfOnChimpCompressor32:
    __slots__ = ("_c", "_lb")

    def __init__(self) -> None:
        self._c = Chimp32Compressor()
        self._lb = [MAX_BETA]

    def add_value(self, v: float) -> None:
        vp, _ = _elf_erase_prefix_write(self._c.out, v, self._lb)
        self._c.add_raw_int32(vp)

    def close(self) -> None:
        self._c.out.write_int(2, 2)
        self._c.close()

    def get_bytes(self) -> bytes:
        return self._c.get_compressed_bytes()


class ElfOnChimpDecompressor32:
    __slots__ = ("_inn", "_ch", "_lb")

    def __init__(self, data: bytes) -> None:
        self._inn = InputBitStreamBytes(data)
        self._ch = Chimp32Decompressor(self._inn)
        self._lb = [MAX_BETA]

    def decompress(self, n: int) -> List[float]:
        out: List[float] = []
        for _ in range(n):
            v = _elf_next_decoded(self._inn, self._ch.read_pair, self._lb)
            if v is None:
                raise ValueError("elf-on-chimp early eos")
            out.append(v)
        return out


class ElfCompressor32:
    __slots__ = ("_xor", "_lb")

    def __init__(self) -> None:
        self._xor = ElfXORCompressor32()
        self._lb = [MAX_BETA]

    def add_value(self, v: float) -> None:
        vp, _ = _elf_erase_prefix_write(self._xor.out, v, self._lb)
        self._xor.add_raw(vp)

    def close(self) -> None:
        self._xor.out.write_int(2, 2)
        self._xor.close_xor()

    def get_bytes(self) -> bytes:
        return self._xor.get_out()


class ElfDecompressor32:
    __slots__ = ("_xor", "_lb")

    def __init__(self, data: bytes) -> None:
        self._xor = ElfXORDecompressor32(InputBitStreamBytes(data))
        self._lb = [MAX_BETA]

    def decompress(self, n: int) -> List[float]:
        out: List[float] = []
        for _ in range(n):
            v = _elf_next_decoded(self._xor.inn, self._xor.read_value, self._lb)
            if v is None:
                raise ValueError("elf+ early eos")
            out.append(v)
        return out


_ELF_CHUNK = 2048


def _bench_elf_variant(columns: List[List[float]], enc: str) -> Tuple[int, int, int, int]:
    tot_o = tot_c = te = td = 0
    for col in columns:
        n = len(col)
        if n == 0:
            continue
        try:
            co = cc = cte = ctd = 0
            for start in range(0, n, _ELF_CHUNK):
                end = min(n, start + _ELF_CHUNK)
                chunk = end - start
                if enc == "onchimp":
                    c = ElfOnChimpCompressor32()
                else:
                    c = ElfCompressor32()
                t0 = time.perf_counter_ns()
                for i in range(start, end):
                    c.add_value(float(col[i]))
                c.close()
                t1 = time.perf_counter_ns()
                raw = c.get_bytes()
                t2 = time.perf_counter_ns()
                if enc == "onchimp":
                    dec = ElfOnChimpDecompressor32(raw)
                else:
                    dec = ElfDecompressor32(raw)
                got = dec.decompress(chunk)
                t3 = time.perf_counter_ns()
                if len(got) != chunk:
                    raise ValueError("len")
                for i in range(chunk):
                    if _raw_bits(got[i]) != _raw_bits(col[start + i]):
                        raise ValueError("bits")
                co += column_original_size(chunk, 'float')
                cc += len(raw)
                cte += t1 - t0
                ctd += t3 - t2
            tot_o += co
            tot_c += cc
            te += cte
            td += ctd
        except (ValueError, struct.error, OverflowError):
            continue
    return tot_o, tot_c, te, td


def bench_elf_on_chimp_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    return _bench_elf_variant(columns, "onchimp")


def bench_elf_plus_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    return _bench_elf_variant(columns, "plus")