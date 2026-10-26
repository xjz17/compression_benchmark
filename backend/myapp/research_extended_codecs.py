"""
SPDP (Burtscher), FPC v1.1 (Burtscher), pFPC v1.0 (Burtscher), Stream VByte delta-1,
MOST-inspired lossy minimal (segment OLS + int16 residuals + ZSTD),
CORAD-style tricklets, Machete-style lossy quant — Python parity with C++/Java benches.
"""
from __future__ import annotations

from .benchmark_loss import accumulate_byte_loss, accumulate_double_list_loss, accumulate_long_list_loss
from .benchmark_wire import column_original_size

import os
import struct
import threading
import time
from typing import List, Tuple

import numpy as np
import zstandard

_BUFFER = 1 << 23
_MAX_TABLE = 1 << 18
MIN_SPDP_LEVEL = 0
MAX_SPDP_LEVEL = 9


def row_name(level: int) -> str:
    return f"spdp_level_{level}"


def levels_to_bench() -> List[int]:
    """Levels to benchmark (always 0..9; partial regen uses ``--algorithms spdp_level_N``)."""
    return list(range(MIN_SPDP_LEVEL, MAX_SPDP_LEVEL + 1))


def _spdp_compress_core(
    level: int, length: int, buf1: bytearray, buf2: bytearray, word_bytes: int = 4
) -> int:
    if word_bytes not in (4, 8) or length % word_bytes != 0:
        return 0
    ln = length // word_bytes
    if word_bytes == 4:
        prev2 = 0
        prev1 = 0
        for pos in range(ln):
            curr = struct.unpack_from("<I", buf1, pos * 4)[0]
            struct.pack_into("<I", buf2, pos * 4, (curr - prev2) & 0xFFFFFFFF)
            prev2 = prev1
            prev1 = curr
    else:
        prev2 = 0
        prev1 = 0
        for pos in range(ln):
            curr = struct.unpack_from("<Q", buf1, pos * 8)[0]
            struct.pack_into("<Q", buf2, pos * 8, (curr - prev2) & 0xFFFFFFFFFFFFFFFF)
            prev2 = prev1
            prev1 = curr
    for pos in range(ln * word_bytes, length):
        buf2[pos] = buf1[pos]
    prev = 0
    wpos = 0
    for _d in range(8):
        rpos = _d
        while rpos < length:
            curr = buf2[rpos] & 0xFF
            buf1[wpos] = (curr - prev) & 0xFF
            prev = curr
            wpos += 1
            rpos += 8
    predtabsize = min(1 << (level + 9), _MAX_TABLE)
    predtabsizem1 = predtabsize - 1
    lastpos = [0] * predtabsize
    rpos = 0
    wpos = 0
    hist = 0
    while rpos < length:
        val = buf1[rpos] & 0xFF
        lpos = lastpos[hist]
        if lpos >= 6:
            if (
                buf1[lpos - 6] == buf1[rpos - 6]
                and buf1[lpos - 5] == buf1[rpos - 5]
                and buf1[lpos - 4] == buf1[rpos - 4]
                and buf1[lpos - 3] == buf1[rpos - 3]
                and buf1[lpos - 2] == buf1[rpos - 2]
                and buf1[lpos - 1] == buf1[rpos - 1]
            ):
                cnt = 0
                while val == (buf1[lpos] & 0xFF) and cnt < 255 and rpos < length - 1:
                    lastpos[hist] = rpos
                    hist = ((hist << 2) ^ val) & predtabsizem1
                    rpos += 1
                    lpos += 1
                    cnt += 1
                    val = buf1[rpos] & 0xFF
                buf2[wpos] = cnt
                wpos += 1
        buf2[wpos] = val
        wpos += 1
        lastpos[hist] = rpos
        hist = ((hist << 2) ^ val) & predtabsizem1
        rpos += 1
    return wpos


def _spdp_decompress_core(
    level: int, length: int, buf2: bytes, buf1: bytearray, word_bytes: int = 4
) -> None:
    if word_bytes not in (4, 8):
        return
    predtabsize = min(1 << (level + 9), _MAX_TABLE)
    predtabsizem1 = predtabsize - 1
    lastpos = [0] * predtabsize
    buf2 = bytearray(buf2)
    rpos = 0
    wpos = 0
    hist = 0
    while rpos < length:
        lpos = lastpos[hist]
        if lpos >= 6:
            if (
                buf1[lpos - 6] == buf1[wpos - 6]
                and buf1[lpos - 5] == buf1[wpos - 5]
                and buf1[lpos - 4] == buf1[wpos - 4]
                and buf1[lpos - 3] == buf1[wpos - 3]
                and buf1[lpos - 2] == buf1[wpos - 2]
                and buf1[lpos - 1] == buf1[wpos - 1]
            ):
                cnt = buf2[rpos]
                rpos += 1
                for _j in range(cnt):
                    val = buf1[lpos]
                    buf1[wpos] = val
                    lastpos[hist] = wpos
                    hist = ((hist << 2) ^ (val & 0xFF)) & predtabsizem1
                    wpos += 1
                    lpos += 1
        val = buf2[rpos]
        buf1[wpos] = val
        lastpos[hist] = wpos
        hist = ((hist << 2) ^ (val & 0xFF)) & predtabsizem1
        wpos += 1
        rpos += 1
    usize = wpos
    # Second phase writes buf2[0..usize); compressed input may be shorter than usize.
    if len(buf2) < usize:
        buf2.extend(b"\x00" * (usize - len(buf2)))
    acc = 0
    rpos = 0
    for d in range(8):
        wpos2 = d
        while wpos2 < usize:
            acc = (acc + (buf1[rpos] & 0xFF)) & 0xFF
            buf2[wpos2] = acc
            rpos += 1
            wpos2 += 8
    wlen = usize // word_bytes
    p2 = 0
    p1 = 0
    if word_bytes == 4:
        for i in range(wlen):
            curr = (struct.unpack_from("<I", buf2, i * 4)[0] + p2) & 0xFFFFFFFF
            struct.pack_into("<I", buf1, i * 4, curr)
            p2 = p1
            p1 = curr
        tail = wlen * 4
    else:
        for i in range(wlen):
            curr = (struct.unpack_from("<Q", buf2, i * 8)[0] + p2) & 0xFFFFFFFFFFFFFFFF
            struct.pack_into("<Q", buf1, i * 8, curr)
            p2 = p1
            p1 = curr
        tail = wlen * 8
    for i in range(tail, usize):
        buf1[i] = buf2[i]


def _spdp_roundtrip(raw: bytes, level: int, word_bytes: int = 4) -> Tuple[bytes, int, int, bytes]:
    n = len(raw)
    if word_bytes not in (4, 8) or n % word_bytes != 0:
        return b"", 0, 0, b""
    buf1 = bytearray(raw)
    buf2 = bytearray(n * 2 + 9)
    t0 = time.perf_counter_ns()
    csize = _spdp_compress_core(level, n, buf1, buf2, word_bytes)
    comp = bytes(buf2[:csize])
    t1 = time.perf_counter_ns()
    t2 = time.perf_counter_ns()
    out = bytearray(n)
    _spdp_decompress_core(level, csize, comp, out, word_bytes)
    t3 = time.perf_counter_ns()
    if bytes(out) != raw:
        return b"", 0, 0, b""
    return comp, t1 - t0, t3 - t2, bytes(out)


def _bench_spdp_bytes_columns(
    columns: List[List[float | int]],
    level: int,
    *,
    wire_data_type: str,
    pack_item: str,
    word_bytes: int,
) -> Tuple[int, int, int, int, float, float]:
    if level < MIN_SPDP_LEVEL or level > MAX_SPDP_LEVEL:
        return 0, 0, 0, 0, 0.0, 0.0
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        raw = struct.pack(pack_item % len(col), *col)
        if len(raw) > _BUFFER or len(raw) % word_bytes != 0:
            continue
        comp, ct, dt, decoded = _spdp_roundtrip(raw, level, word_bytes)
        if not comp:
            continue
        accumulate_byte_loss(raw, decoded, loss_acc)
        tot_o += column_original_size(len(col), wire_data_type)
        tot_c += len(comp)
        te += ct
        td += dt
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_spdp_float_columns(columns: List[List[float]], level: int = 5) -> Tuple[int, int, int, int]:
    return _bench_spdp_bytes_columns(
        columns, level, wire_data_type="float", pack_item="<%df", word_bytes=4
    )


def bench_spdp_double_columns(columns: List[List[float]], level: int = 5) -> Tuple[int, int, int, int]:
    return _bench_spdp_bytes_columns(
        columns, level, wire_data_type="float", pack_item="<%dd", word_bytes=8
    )


def bench_spdp_int64_columns(columns: List[List[int]], level: int = 5) -> Tuple[int, int, int, int]:
    return _bench_spdp_bytes_columns(
        columns, level, wire_data_type="int", pack_item="<%dq", word_bytes=8
    )


ROW_NAME = "fpc"
DEFAULT_FPC_PRED_SIZE_LG2 = 16
_FPC_SIZE = 32768
_FPC_MASK = (
    0x0000000000000000,
    0x00000000000000FF,
    0x000000000000FFFF,
    0x0000000000FFFFFF,
    0x000000FFFFFFFFFF,
    0x0000FFFFFFFFFFFF,
    0x00FFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
)


def pred_size_lg2_to_bench() -> int:
    """FCM/dFCM table size exponent (``1 << lg2`` entries). See ``COMPRESSION_BENCHMARK_FPC_PRED_SIZE_LG2``."""
    env = os.environ.get("COMPRESSION_BENCHMARK_FPC_PRED_SIZE_LG2")
    if env:
        try:
            v = int(env.strip())
        except ValueError:
            v = -1
        if 0 <= v <= 24:
            return v
    return DEFAULT_FPC_PRED_SIZE_LG2


def _fpc_u64(x: int) -> int:
    return x & 0xFFFFFFFFFFFFFFFF


def _fpc_ushr(x: int, bits: int) -> int:
    return _fpc_u64(x) >> bits


def _fpc_bcode(xor1: int) -> int:
    xor1 = _fpc_u64(xor1)
    bcode = 7
    if xor1 >> 56 == 0:
        bcode = 6
    if xor1 >> 48 == 0:
        bcode = 5
    if xor1 >> 40 == 0:
        bcode = 4
    if xor1 >> 24 == 0:
        bcode = 3
    if xor1 >> 16 == 0:
        bcode = 2
    if xor1 >> 8 == 0:
        bcode = 1
    if xor1 == 0:
        bcode = 0
    return bcode


def _fpc_write_xor(outbuf: bytearray, out: int, xor1: int) -> None:
    xor1 = _fpc_u64(xor1)
    shift = (out & 0x7) << 3
    base = (out >> 3) << 3
    need = base + 16
    if len(outbuf) < need:
        outbuf.extend(b"\x00" * (need - len(outbuf)))
    lo = struct.unpack_from("<Q", outbuf, base)[0]
    lo = _fpc_u64(lo | (xor1 << shift))
    struct.pack_into("<Q", outbuf, base, lo)
    if shift == 0:
        xor1 = 0
    struct.pack_into("<Q", outbuf, base + 8, _fpc_ushr(xor1, 64 - shift))


def _fpc_read_xor(inbuf: bytes | bytearray, inp: int) -> int:
    base = (inp >> 3) << 3
    lo = struct.unpack_from("<Q", inbuf, base)[0]
    nxt = struct.unpack_from("<Q", inbuf, base + 8)[0]
    tmp = (inp & 0x7) << 3
    val = _fpc_ushr(lo, tmp)
    if tmp == 0:
        nxt = 0
    else:
        nxt = (nxt << (64 - tmp)) & 0xFFFFFFFFFFFFFFFF
    return _fpc_u64(val | nxt)


def _fpc_compress_memory(raw: bytes, pred_size_lg2: int) -> bytes:
    """Port of ``FPC_Compress_Memory`` (fpc.c / FpcDouble.java)."""
    in_bytes = len(raw)
    if in_bytes == 0 or in_bytes % 8 != 0:
        return b""
    predsizem1 = pred_size_lg2
    out_stream = bytearray([predsizem1 & 0xFF])
    predsizem1 = (1 << pred_size_lg2) - 1
    fcm = [0] * (predsizem1 + 1)
    dfcm = [0] * (predsizem1 + 1)
    hash_v = 0
    dhash = 0
    lastval = 0
    pred1 = 0
    pred2 = 0
    in_cursor = 0
    while in_cursor < in_bytes:
        chunk_end = min(in_cursor + _FPC_SIZE * 8, in_bytes)
        intot = (chunk_end - in_cursor) // 8
        inbuf = [
            struct.unpack_from("<Q", raw, in_cursor + j * 8)[0] for j in range(intot)
        ]
        while len(inbuf) < intot + 2:
            inbuf.append(0)
        out = 6 + ((intot + 1) >> 1)
        outbuf = bytearray(6 + (_FPC_SIZE // 2) + _FPC_SIZE * 8 + 2)
        struct.pack_into("<Q", outbuf, (out >> 3) << 3, 0)
        val = inbuf[0]
        bcode = 0
        i = 0
        while i < intot:
            xor1 = _fpc_u64(val ^ pred1)
            fcm[hash_v] = val
            hash_v = ((hash_v << 6) ^ _fpc_ushr(val, 48)) & predsizem1
            pred1 = fcm[hash_v]

            stride = _fpc_u64(val - lastval)
            xor2 = _fpc_u64(val ^ _fpc_u64(lastval + pred2))
            lastval = val
            val = inbuf[i + 1]
            dfcm[dhash] = stride
            dhash = ((dhash << 2) ^ _fpc_ushr(stride, 40)) & predsizem1
            pred2 = dfcm[dhash]

            code = 0
            if xor1 > xor2:
                code = 0x80
                xor1 = xor2
            bcode = _fpc_bcode(xor1)
            _fpc_write_xor(outbuf, out, xor1)
            out += bcode + (bcode >> 2)
            code |= bcode << 4

            xor1 = _fpc_u64(val ^ pred1)
            fcm[hash_v] = val
            hash_v = ((hash_v << 6) ^ _fpc_ushr(val, 48)) & predsizem1
            pred1 = fcm[hash_v]

            stride = _fpc_u64(val - lastval)
            xor2 = _fpc_u64(val ^ _fpc_u64(lastval + pred2))
            lastval = val
            val = inbuf[i + 2]
            dfcm[dhash] = stride
            dhash = ((dhash << 2) ^ _fpc_ushr(stride, 40)) & predsizem1
            pred2 = dfcm[dhash]

            bcode = code | 0x8
            if xor1 > xor2:
                code = bcode
                xor1 = xor2
            bcode = _fpc_bcode(xor1)
            _fpc_write_xor(outbuf, out, xor1)
            out += bcode + (bcode >> 2)
            outbuf[6 + (i >> 1)] = (code | bcode) & 0xFF
            i += 2
        if intot & 1:
            out -= bcode + (bcode >> 2)
        outbuf[0] = intot & 0xFF
        outbuf[1] = (intot >> 8) & 0xFF
        outbuf[2] = (intot >> 16) & 0xFF
        outbuf[3] = out & 0xFF
        outbuf[4] = (out >> 8) & 0xFF
        outbuf[5] = (out >> 16) & 0xFF
        out_stream.extend(outbuf[:out])
        in_cursor += intot * 8
    return bytes(out_stream)


def _fpc_decompress_memory(comp: bytes) -> bytes:
    """Port of ``FPC_Decompress_Memory`` (fpc.c / FpcDouble.java)."""
    in_bytes = len(comp)
    if in_bytes < 7:
        return b""
    in_cursor = 0
    out_parts: List[bytes] = []
    inbuf = bytearray(((_FPC_SIZE // 2) + _FPC_SIZE * 8 + 6 + 2))

    inbuf[0:7] = comp[0:7]
    in_cursor = 7
    predsizem1 = inbuf[0]
    predsizem1 = (1 << predsizem1) - 1
    fcm = [0] * (predsizem1 + 1)
    dfcm = [0] * (predsizem1 + 1)
    hash_v = 0
    dhash = 0
    lastval = 0
    pred1 = 0
    pred2 = 0
    intot = inbuf[3] & 0xFF
    intot = (intot << 8) | (inbuf[2] & 0xFF)
    intot = (intot << 8) | (inbuf[1] & 0xFF)
    inp = inbuf[6] & 0xFF
    inp = (inp << 8) | (inbuf[5] & 0xFF)
    inp = (inp << 8) | (inbuf[4] & 0xFF)

    while intot > 0:
        end = min(inp, in_bytes - in_cursor)
        if end + 6 < inp:
            return b""
        inbuf[0:end] = comp[in_cursor : in_cursor + end]
        in_cursor += end
        inp = (intot + 1) >> 1
        outbuf = [0] * (intot + 1)
        i = 0
        while i < intot:
            code = inbuf[i >> 1] & 0xFF
            val = _fpc_read_xor(inbuf, inp)
            bcode_hi = (code >> 4) & 0x7
            inp += bcode_hi + (bcode_hi >> 2)
            val &= _FPC_MASK[bcode_hi]
            if code & 0x80:
                pred1 = pred2
            val = _fpc_u64(val ^ pred1)
            fcm[hash_v] = val
            hash_v = ((hash_v << 6) ^ _fpc_ushr(val, 48)) & predsizem1
            pred1 = fcm[hash_v]
            stride = _fpc_u64(val - lastval)
            dfcm[dhash] = stride
            dhash = ((dhash << 2) ^ _fpc_ushr(stride, 40)) & predsizem1
            pred2 = _fpc_u64(val + dfcm[dhash])
            lastval = val
            outbuf[i] = val

            val = _fpc_read_xor(inbuf, inp)
            bcode_lo = code & 0x7
            inp += bcode_lo + (bcode_lo >> 2)
            val &= _FPC_MASK[bcode_lo]
            if code & 0x8:
                pred1 = pred2
            val = _fpc_u64(val ^ pred1)
            fcm[hash_v] = val
            hash_v = ((hash_v << 6) ^ _fpc_ushr(val, 48)) & predsizem1
            pred1 = fcm[hash_v]
            stride = _fpc_u64(val - lastval)
            dfcm[dhash] = stride
            dhash = ((dhash << 2) ^ _fpc_ushr(stride, 40)) & predsizem1
            pred2 = _fpc_u64(val + dfcm[dhash])
            lastval = val
            outbuf[i + 1] = val
            i += 2
        out_parts.append(b"".join(struct.pack("<Q", v) for v in outbuf[:intot]))
        intot = 0
        if (end - 6) >= inp:
            intot = inbuf[inp + 2] & 0xFF
            intot = (intot << 8) | (inbuf[inp + 1] & 0xFF)
            intot = (intot << 8) | (inbuf[inp] & 0xFF)
            end = inbuf[inp + 5] & 0xFF
            end = (end << 8) | (inbuf[inp + 4] & 0xFF)
            end = (end << 8) | (inbuf[inp + 3] & 0xFF)
            inp = end
    return b"".join(out_parts)


def _fpc_roundtrip(raw: bytes, pred_size_lg2: int) -> Tuple[bytes, int, int, bytes]:
    n = len(raw)
    if n == 0 or n % 8 != 0:
        return b"", 0, 0, b""
    t0 = time.perf_counter_ns()
    comp = _fpc_compress_memory(raw, pred_size_lg2)
    t1 = time.perf_counter_ns()
    if not comp:
        return b"", 0, 0, b""
    t2 = time.perf_counter_ns()
    dec = _fpc_decompress_memory(comp)
    t3 = time.perf_counter_ns()
    if dec != raw:
        return b"", 0, 0, b""
    return comp, t1 - t0, t3 - t2, dec


def _bench_fpc_bytes_columns(
    columns: List[List[float | int]],
    pred_size_lg2: int,
    *,
    wire_data_type: str,
    pack_item: str,
) -> Tuple[int, int, int, int, float, float]:
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        raw = struct.pack(pack_item % len(col), *col)
        if len(raw) > _BUFFER or len(raw) % 8 != 0:
            continue
        comp, ct, dt, decoded = _fpc_roundtrip(raw, pred_size_lg2)
        if not comp:
            continue
        accumulate_byte_loss(raw, decoded, loss_acc)
        tot_o += column_original_size(len(col), wire_data_type)
        tot_c += len(comp)
        te += ct
        td += dt
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_fpc_double_columns(
    columns: List[List[float]], pred_size_lg2: int | None = None
) -> Tuple[int, int, int, int, float, float]:
    lg2 = pred_size_lg2 if pred_size_lg2 is not None else pred_size_lg2_to_bench()
    return _bench_fpc_bytes_columns(
        columns, lg2, wire_data_type="float", pack_item="<%dd"
    )


def bench_fpc_int64_columns(
    columns: List[List[int]], pred_size_lg2: int | None = None
) -> Tuple[int, int, int, int, float, float]:
    lg2 = pred_size_lg2 if pred_size_lg2 is not None else pred_size_lg2_to_bench()
    return _bench_fpc_bytes_columns(
        columns, lg2, wire_data_type="int", pack_item="<%dq"
    )


PFPC_ROW_NAME = "pfpc"
DEFAULT_PFPC_PRED_SIZE_LG2 = 20
DEFAULT_PFPC_THREADS = 8
DEFAULT_PFPC_CHUNKSIZE = 2048
_PFPC_BUFSIZE = 8 * 1024 * 1024
_PFPC_HLSHIFT = 6
_PFPC_HRSHIFT = 48
_PFPC_DLSHIFT = 2
_PFPC_DRSHIFT = 40


def pred_size_lg2_to_bench_pfpc() -> int:
    env = os.environ.get("COMPRESSION_BENCHMARK_PFPC_PRED_SIZE_LG2")
    if env:
        try:
            v = int(env.strip())
        except ValueError:
            v = -1
        if 0 <= v <= 24:
            return v
    return DEFAULT_PFPC_PRED_SIZE_LG2


def threads_to_bench_pfpc() -> int:
    env = os.environ.get("COMPRESSION_BENCHMARK_PFPC_THREADS")
    if env:
        try:
            v = int(env.strip())
        except ValueError:
            v = -1
        if 1 <= v <= 128:
            return v
    return DEFAULT_PFPC_THREADS


def chunk_size_to_bench_pfpc() -> int:
    env = os.environ.get("COMPRESSION_BENCHMARK_PFPC_CHUNKSIZE")
    if env:
        try:
            v = int(env.strip())
        except ValueError:
            v = -1
        if 1 <= v <= 65536:
            return v
    return DEFAULT_PFPC_CHUNKSIZE


def _pfpc_bcode(xor1: int) -> int:
    return _fpc_bcode(xor1)


def _pfpc_write_xor(outbuf: bytearray, out: int, xor1: int) -> None:
    xor1 = _fpc_u64(xor1)
    shift = (out & 0x7) << 3
    base = (out >> 3) << 3
    need = base + 16
    if len(outbuf) < need:
        outbuf.extend(b"\x00" * (need - len(outbuf)))
    lo = struct.unpack_from("<Q", outbuf, base)[0]
    lo = _fpc_u64(lo | (xor1 << shift))
    struct.pack_into("<Q", outbuf, base, lo)
    hi = (_fpc_ushr(xor1, 63 - shift) >> 1) if shift != 0 else 0
    struct.pack_into("<Q", outbuf, base + 8, hi)


def _pfpc_read_xor(inbuf: bytes | bytearray, inp: int) -> int:
    base = (inp >> 3) << 3
    lo = struct.unpack_from("<Q", inbuf, base)[0]
    nxt = struct.unpack_from("<Q", inbuf, base + 8)[0]
    tmp = (inp & 0x7) << 3
    val = _fpc_ushr(lo, tmp)
    if tmp == 0:
        nxt = 0
    else:
        nxt = (nxt << (63 - tmp)) & 0xFFFFFFFFFFFFFFFF
    return _fpc_u64(val | (nxt + nxt))


def _pfpc_compress_chunk(
    threads: int,
    chunksize: int,
    end: int,
    inbuf: List[int],
    ibuf_start: int,
    obuf: bytearray,
    predsizem1: int,
    fcm: List[int],
    dfcm: List[int],
) -> None:
    full = end // (chunksize * threads)
    part = end - full * chunksize * threads
    if part > chunksize:
        part = chunksize
    out = full * chunksize + part
    out = ((out + 1) >> 1) + 4
    base_align = out & -8
    if len(obuf) <= base_align + 8:
        obuf.extend(b"\x00" * (base_align + 16 - len(obuf)))
    struct.pack_into("<Q", obuf, base_align, 0)

    hash_v = 0
    dhash = 0
    lastval = 0
    pred1 = 0
    pred2 = 0
    save = 0
    o = 8
    offset = 0
    increment = threads * chunksize
    local_chunksize = chunksize
    while end - offset > 0:
        if local_chunksize > end - offset:
            local_chunksize = end - offset
        for i in range(local_chunksize):
            val = inbuf[ibuf_start + offset + i]
            xor1 = _fpc_u64(val ^ pred1)
            fcm[hash_v] = val
            hash_v = ((hash_v << _PFPC_HLSHIFT) ^ _fpc_ushr(val, _PFPC_HRSHIFT)) & predsizem1
            pred1 = fcm[hash_v]

            stride = _fpc_u64(val - lastval)
            xor2 = _fpc_u64(val ^ _fpc_u64(lastval + pred2))
            lastval = val
            dfcm[dhash] = stride
            dhash = ((dhash << _PFPC_DLSHIFT) ^ _fpc_ushr(stride, _PFPC_DRSHIFT)) & predsizem1
            pred2 = dfcm[dhash]

            code = 0
            if xor1 > xor2:
                code = 0x8
                xor1 = xor2
            bcode = _pfpc_bcode(xor1)
            _pfpc_write_xor(obuf, out, xor1)
            code |= bcode
            out += bcode + (bcode >> 2)
            need_byte = (o >> 1) + 1
            if len(obuf) < need_byte:
                obuf.extend(b"\x00" * (need_byte - len(obuf)))
            obuf[o >> 1] = (save | code) & 0xFF
            save = (code << 4) & 0xFF
            o += 1
        offset += increment
    if o & 1:
        if len(obuf) <= o >> 1:
            obuf.extend(b"\x00")
        obuf[o >> 1] = save & 0xFF
    struct.pack_into("<I", obuf, 0, out)


def _pfpc_decompress_chunk(
    threads: int,
    chunksize: int,
    end: int,
    ibuf: bytes | bytearray,
    obuf: List[int],
    obuf_start: int,
    predsizem1: int,
    fcm: List[int],
    dfcm: List[int],
) -> None:
    hash_v = 0
    dhash = 0
    lastval = 0
    pred1 = 0
    pred2 = 0
    full = end // (chunksize * threads)
    part = end - full * chunksize * threads
    if part > chunksize:
        part = chunksize
    inp = full * chunksize + part
    inp = ((inp + 1) >> 1) + 4
    i = 8
    alt = 4
    offset = 0
    increment = threads * chunksize
    local_chunksize = chunksize
    while end - offset > 0:
        if local_chunksize > end - offset:
            local_chunksize = end - offset
        for o in range(local_chunksize):
            bcode = (ibuf[i >> 1] >> alt) & 0xF
            i += 1
            alt ^= 4
            val = _pfpc_read_xor(ibuf, inp)
            if bcode & 0x8:
                pred1 = pred2
            bcode &= 0x7
            val &= _FPC_MASK[bcode]
            inp += bcode + (bcode >> 2)
            val = _fpc_u64(val ^ pred1)
            fcm[hash_v] = val
            hash_v = ((hash_v << _PFPC_HLSHIFT) ^ _fpc_ushr(val, _PFPC_HRSHIFT)) & predsizem1
            pred1 = fcm[hash_v]
            stride = _fpc_u64(val - lastval)
            dfcm[dhash] = stride
            dhash = ((dhash << _PFPC_DLSHIFT) ^ _fpc_ushr(stride, _PFPC_DRSHIFT)) & predsizem1
            pred2 = _fpc_u64(val + dfcm[dhash])
            lastval = val
            obuf[obuf_start + offset + o] = val
        offset += increment


def _pfpc_chunkrepeat(threads: int, chunk_size: int) -> int:
    chunkrepeat = _PFPC_BUFSIZE // (8 * threads * chunk_size)
    if chunkrepeat * 8 * threads * chunk_size < _PFPC_BUFSIZE:
        chunkrepeat += 1
    return chunkrepeat


def _pfpc_compress_block(
    num: int,
    inbuf: List[int],
    threads: int,
    chunk_size: int,
    pred_size_lg2: int,
) -> bytes:
    predsizem1 = (1 << pred_size_lg2) - 1
    chunkrepeat = _pfpc_chunkrepeat(threads, chunk_size)
    inter: List[bytearray] = []
    fcm_tables: List[List[int]] = []
    dfcm_tables: List[List[int]] = []
    for _ in range(threads):
        inter.append(
            bytearray(8 + ((chunk_size * chunkrepeat + 1) // 2) + (chunk_size * chunkrepeat * 8) + 7)
        )
        fcm_tables.append([0] * (predsizem1 + 1))
        dfcm_tables.append([0] * (predsizem1 + 1))

    def worker(tid: int) -> None:
        end = num - tid * chunk_size
        if end <= 0:
            struct.pack_into("<I", inter[tid], 0, 4)
            return
        _pfpc_compress_chunk(
            threads,
            chunk_size,
            end,
            inbuf,
            tid * chunk_size,
            inter[tid],
            predsizem1,
            fcm_tables[tid],
            dfcm_tables[tid],
        )

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()

    parts = bytearray()
    parts.extend(struct.pack("<I", num))
    for t in range(threads):
        cnt = struct.unpack_from("<I", inter[t], 0)[0]
        parts.extend(inter[t][:cnt])
    return bytes(parts)


def _pfpc_compress_memory(raw: bytes, pred_size_lg2: int, threads: int, chunk_size: int) -> bytes:
    in_bytes = len(raw)
    if in_bytes == 0 or in_bytes % 8 != 0:
        return b""
    if threads < 1 or threads >= 256 or chunk_size < 1:
        return b""
    out_stream = bytearray()
    out_stream.append(pred_size_lg2 & 0xFF)
    out_stream.append(threads & 0xFF)
    out_stream.extend(struct.pack("<I", chunk_size))
    max_block = _PFPC_BUFSIZE >> 3
    in_cursor = 0
    while in_cursor < in_bytes:
        block_bytes = min(in_bytes - in_cursor, max_block * 8)
        num = block_bytes // 8
        inbuf = list(struct.unpack("<" + "Q" * num, raw[in_cursor : in_cursor + block_bytes]))
        out_stream.extend(_pfpc_compress_block(num, inbuf, threads, chunk_size, pred_size_lg2))
        in_cursor += block_bytes
    return bytes(out_stream)


def _pfpc_decompress_memory(comp: bytes) -> bytes:
    if len(comp) < 6:
        return b""
    pred_lg2 = comp[0]
    threads = comp[1]
    chunk_size = struct.unpack_from("<I", comp, 2)[0]
    if threads < 1 or threads >= 256 or chunk_size < 1 or pred_lg2 >= 256:
        return b""
    predsizem1 = (1 << pred_lg2) - 1
    chunkrepeat = _pfpc_chunkrepeat(threads, chunk_size)
    inter: List[bytearray] = []
    fcm_tables: List[List[int]] = []
    dfcm_tables: List[List[int]] = []
    for _ in range(threads):
        inter.append(
            bytearray(8 + ((chunk_size * chunkrepeat + 1) // 2) + (chunk_size * chunkrepeat * 8) + 7)
        )
        fcm_tables.append([0] * (predsizem1 + 1))
        dfcm_tables.append([0] * (predsizem1 + 1))
    in_off = 6
    out_parts: List[bytes] = []
    max_out = _PFPC_BUFSIZE >> 3
    while in_off + 4 <= len(comp):
        num = struct.unpack_from("<I", comp, in_off)[0]
        in_off += 4
        if num <= 0:
            break
        for t in range(threads):
            if in_off + 4 > len(comp):
                return b""
            chunk_total = struct.unpack_from("<I", comp, in_off)[0]
            if chunk_total < 4 or in_off + chunk_total > len(comp):
                return b""
            inter[t][:chunk_total] = comp[in_off : in_off + chunk_total]
            in_off += chunk_total

        output = [0] * num

        def worker(tid: int) -> None:
            end = num - tid * chunk_size
            if end <= 0:
                return
            _pfpc_decompress_chunk(
                threads,
                chunk_size,
                end,
                inter[tid],
                output,
                tid * chunk_size,
                predsizem1,
                fcm_tables[tid],
                dfcm_tables[tid],
            )

        ths = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        out_parts.append(b"".join(struct.pack("<Q", v) for v in output[:num]))
    return b"".join(out_parts)


def _pfpc_roundtrip(
    raw: bytes, pred_size_lg2: int, threads: int, chunk_size: int
) -> Tuple[bytes, int, int, bytes]:
    n = len(raw)
    if n == 0 or n % 8 != 0:
        return b"", 0, 0, b""
    t0 = time.perf_counter_ns()
    comp = _pfpc_compress_memory(raw, pred_size_lg2, threads, chunk_size)
    t1 = time.perf_counter_ns()
    if not comp:
        return b"", 0, 0, b""
    t2 = time.perf_counter_ns()
    dec = _pfpc_decompress_memory(comp)
    t3 = time.perf_counter_ns()
    if dec != raw:
        return b"", 0, 0, b""
    return comp, t1 - t0, t3 - t2, dec


def _bench_pfpc_bytes_columns(
    columns: List[List[float | int]],
    pred_size_lg2: int,
    threads: int,
    chunk_size: int,
    *,
    wire_data_type: str,
    pack_item: str,
) -> Tuple[int, int, int, int, float, float]:
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        raw = b"".join(struct.pack(pack_item, v) for v in col)
        comp, ct, dt, dec = _pfpc_roundtrip(raw, pred_size_lg2, threads, chunk_size)
        if not comp:
            continue
        tot_o += column_original_size(len(col), wire_data_type)
        tot_c += len(comp)
        te += ct
        td += dt
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_pfpc_double_columns(
    columns: List[List[float]], pred_size_lg2: int | None = None
) -> Tuple[int, int, int, int, float, float]:
    lg2 = pred_size_lg2 if pred_size_lg2 is not None else pred_size_lg2_to_bench_pfpc()
    th = threads_to_bench_pfpc()
    cs = chunk_size_to_bench_pfpc()
    return _bench_pfpc_bytes_columns(
        columns, lg2, th, cs, wire_data_type="float", pack_item="<d"
    )


def bench_pfpc_int64_columns(
    columns: List[List[int]], pred_size_lg2: int | None = None
) -> Tuple[int, int, int, int, float, float]:
    lg2 = pred_size_lg2 if pred_size_lg2 is not None else pred_size_lg2_to_bench_pfpc()
    th = threads_to_bench_pfpc()
    cs = chunk_size_to_bench_pfpc()
    return _bench_pfpc_bytes_columns(
        columns, lg2, th, cs, wire_data_type="int", pack_item="<q"
    )


def _svb_code_bits(val: int) -> int:
    v = val & 0xFFFFFFFF
    if v < (1 << 8):
        return 0
    if v < (1 << 16):
        return 1
    if v < (1 << 24):
        return 2
    return 3


def _svb_write_data(val: int, out: bytearray, off: int) -> int:
    v = val & 0xFFFFFFFF
    if v < (1 << 8):
        out[off] = v & 0xFF
        return 1
    if v < (1 << 16):
        struct.pack_into("<H", out, off, v & 0xFFFF)
        return 2
    if v < (1 << 24):
        struct.pack_into("<H", out, off, v & 0xFFFF)
        out[off + 2] = (v >> 16) & 0xFF
        return 3
    struct.pack_into("<I", out, off, v)
    return 4


def _svb_encode_d1(u32: List[int]) -> bytes:
    n = len(u32)
    key_len = (n + 3) // 4
    prev = 0
    data_sz = 0
    tmp = bytearray(n * 5)
    for c in range(n):
        val = (u32[c] - prev) & 0xFFFFFFFF
        prev = u32[c] & 0xFFFFFFFF
        data_sz += _svb_write_data(val, tmp, data_sz)
    out = bytearray(4 + key_len + data_sz)
    struct.pack_into("<I", out, 0, n)
    key_ptr = 4
    data_ptr = 4 + key_len
    shift = 0
    key = 0
    prev = 0
    for c in range(n):
        if shift == 8:
            shift = 0
            out[key_ptr] = key & 0xFF
            key_ptr += 1
            key = 0
        val = (u32[c] - prev) & 0xFFFFFFFF
        prev = u32[c] & 0xFFFFFFFF
        nb = _svb_write_data(val, out, data_ptr)
        data_ptr += nb
        key |= _svb_code_bits(val) << shift
        shift += 2
    out[key_ptr] = key & 0xFF
    return bytes(out[:data_ptr])


def _svb_decode_d1(buf: bytes, n: int) -> List[int]:
    count = struct.unpack_from("<I", buf, 0)[0]
    if count != n:
        raise ValueError("count")
    key_len = (n + 3) // 4
    key_ptr = 4
    data_ptr = 4 + key_len
    shift = 0
    key = buf[key_ptr]
    key_ptr += 1
    prev = 0
    out: List[int] = []
    for c in range(n):
        if shift == 8:
            shift = 0
            key = buf[key_ptr]
            key_ptr += 1
        code = (key >> shift) & 3
        if code == 0:
            d = buf[data_ptr]
            data_ptr += 1
        elif code == 1:
            d = struct.unpack_from("<H", buf, data_ptr)[0]
            data_ptr += 2
        elif code == 2:
            d = struct.unpack_from("<H", buf, data_ptr)[0] | (buf[data_ptr + 2] << 16)
            data_ptr += 3
        else:
            d = struct.unpack_from("<I", buf, data_ptr)[0]
            data_ptr += 4
        val = (d + prev) & 0xFFFFFFFF
        out.append(val)
        prev = val
        shift += 2
    return out


def bench_streamvbyte_delta1_int64_columns(columns: List[List[int]]) -> Tuple[int, int, int, int, float, float]:
    from .int64_column_codec import stream_vbyte_decode_int64, stream_vbyte_encode_int64

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        tot_o += len(col) * 8
        data = [int(v) for v in col]
        t0 = time.perf_counter_ns()
        try:
            enc = stream_vbyte_encode_int64(data)
        except Exception:
            continue
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        try:
            dec = stream_vbyte_decode_int64(enc, len(data))
        except Exception:
            continue
        t3 = time.perf_counter_ns()
        if dec != data:
            continue
        accumulate_long_list_loss(data, dec, loss_acc)
        tot_c += len(enc)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_streamvbyte_delta1_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    """Legacy int32 path; float/double scaled columns should use int64 bench."""
    return bench_streamvbyte_delta1_int64_columns(columns)


_K_SEG = 128
_MOST_REL_ERROR = 1e-3


def _ols_ab_segment(seg: np.ndarray) -> Tuple[float, float]:
    """OLS on segment indices 0..m-1 (matches C++/Java)."""
    m = int(seg.size)
    sx = sy = sxx = sxy = 0.0
    for i in range(m):
        x = float(i)
        y = float(seg[i])
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
    dm = float(m)
    den = dm * sxx - sx * sx
    if abs(den) < 1e-30:
        return (float(seg[0]) if m else 0.0, 0.0)
    b = (dm * sxy - sx * sy) / den
    a = (sy - b * sx) / dm
    return float(a), float(b)


def _pack_most_column_raw(col: np.ndarray, *, mode: str) -> bytes:
    """Segment OLS + per-segment int16 residual quant (lossy; matches most_float_codec.cpp)."""
    if mode == "float32":
        arr = np.asarray(col, dtype=np.float32)
        pack_head = "<Ifff"
        cast = np.float32
    elif mode == "double":
        arr = np.asarray(col, dtype=np.float64)
        pack_head = "<Iddd"
        cast = float
    elif mode == "int64":
        arr = np.asarray(col, dtype=np.int64)
        pack_head = "<Iddd"
        cast = float
    else:
        raise ValueError("unsupported most mode")
    n = int(arr.size)
    col_max = float(np.max(np.abs(arr))) if n else 1.0
    if col_max == 0.0:
        col_max = 1.0
    parts: List[bytes] = []
    for off in range(0, n, _K_SEG):
        m = min(_K_SEG, n - off)
        seg = arr[off : off + m]
        a, b = _ols_ab_segment(seg)
        res = np.empty(m, dtype=np.float64)
        for i in range(m):
            res[i] = float(seg[i]) - float(a + b * cast(i))
        res_max = float(np.max(np.abs(res))) if m else 0.0
        scale = res_max * _MOST_REL_ERROR if res_max > 0.0 else 1.0
        inv = 1.0 / scale
        q = np.clip(np.rint(res.astype(np.float64) * inv), -32768, 32767).astype(np.int16)
        parts.append(struct.pack(pack_head, m, cast(a), cast(b), cast(scale)) + q.tobytes())
    return b"".join(parts)


def _verify_most_column_raw(col: np.ndarray, raw: bytes, *, mode: str) -> bool:
    """Lossy reconstruction within relative error (same bound as Machete bench)."""
    if mode == "float32":
        arr = np.asarray(col, dtype=np.float32)
        unpack_head = "<Ifff"
        head_size = 16
        cast = np.float32
        eps = 1e-6
    elif mode == "double":
        arr = np.asarray(col, dtype=np.float64)
        unpack_head = "<Iddd"
        head_size = 28
        cast = float
        eps = 1e-9
    elif mode == "int64":
        arr = np.asarray(col, dtype=np.int64)
        unpack_head = "<Iddd"
        head_size = 28
        cast = float
        eps = 1e-6
    else:
        raise ValueError("unsupported most mode")
    n = int(arr.size)
    col_max = float(np.max(np.abs(arr))) if n else 1.0
    if col_max == 0.0:
        col_max = 1.0
    pos = 0
    off = 0
    while off < n:
        if pos + head_size > len(raw):
            return False
        m, a, b, scale = struct.unpack_from(unpack_head, raw, pos)
        pos += head_size
        if m > _K_SEG or off + m > n or pos + 2 * m > len(raw):
            return False
        for i in range(m):
            q = struct.unpack_from("<h", raw, pos)[0]
            pos += 2
            dec = cast(a + b * cast(i) + cast(q) * cast(scale))
            orig = float(arr[off + i])
            err = abs(float(dec) - orig)
            bound = _MOST_REL_ERROR * max(abs(orig), col_max)
            if err > bound * 1.0001 + eps:
                return False
        off += m
    return pos == len(raw) and off == n


def _accumulate_most_raw_loss(col: np.ndarray, raw: bytes, mode: str, acc: List[float]) -> None:
    if mode == "float32":
        unpack_head = "<Ifff"
        head_size = 16
        cast = np.float32
    elif mode in ("double", "int64"):
        unpack_head = "<Iddd"
        head_size = 28
        cast = float
    else:
        return
    n = int(col.size)
    pos = 0
    off = 0
    orig: List[float] = []
    decoded: List[float] = []
    while off < n:
        if pos + head_size > len(raw):
            return
        m, a, b, scale = struct.unpack_from(unpack_head, raw, pos)
        pos += head_size
        if m > _K_SEG or off + m > n or pos + 2 * m > len(raw):
            return
        for i in range(m):
            q = struct.unpack_from("<h", raw, pos)[0]
            pos += 2
            orig.append(float(col[off + i]))
            decoded.append(float(cast(a + b * cast(i) + cast(q) * cast(scale))))
        off += m
    if off == n and pos == len(raw):
        if mode == "int64":
            accumulate_long_list_loss(
                [int(col[i]) for i in range(n)],
                [int(round(decoded[i])) for i in range(n)],
                acc,
            )
        else:
            accumulate_double_list_loss(orig, decoded, acc)


def bench_most_float_minimal_columns(columns: List[List[float]]) -> Tuple[int, int, int, int, float, float]:
    zc = zstandard.ZstdCompressor(level=3)
    zd = zstandard.ZstdDecompressor()
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        arr = np.array(col, dtype=np.float32)
        n = int(arr.size)
        raw = _pack_most_column_raw(arr, mode="float32")
        if not _verify_most_column_raw(arr, raw, mode="float32"):
            continue
        t0 = time.perf_counter_ns()
        c = zc.compress(raw)
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        d = zd.decompress(c, max_output_size=len(raw))
        t3 = time.perf_counter_ns()
        if d != raw or not _verify_most_column_raw(arr, d, mode="float32"):
            continue
        _accumulate_most_raw_loss(arr, d, "float32", loss_acc)
        tot_o += column_original_size(n, 'float')
        tot_c += len(c)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_most_double_minimal_columns(columns: List[List[float]]) -> Tuple[int, int, int, int, float, float]:
    zc = zstandard.ZstdCompressor(level=3)
    zd = zstandard.ZstdDecompressor()
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        arr = np.array(col, dtype=np.float64)
        n = int(arr.size)
        raw = _pack_most_column_raw(arr, mode="double")
        if not _verify_most_column_raw(arr, raw, mode="double"):
            continue
        t0 = time.perf_counter_ns()
        c = zc.compress(raw)
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        d = zd.decompress(c, max_output_size=len(raw))
        t3 = time.perf_counter_ns()
        if d != raw or not _verify_most_column_raw(arr, d, mode="double"):
            continue
        _accumulate_most_raw_loss(arr, d, "double", loss_acc)
        tot_o += column_original_size(n, "float")
        tot_c += len(c)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_most_int64_minimal_columns(columns: List[List[int]]) -> Tuple[int, int, int, int, float, float]:
    zc = zstandard.ZstdCompressor(level=3)
    zd = zstandard.ZstdDecompressor()
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        arr = np.array(col, dtype=np.int64)
        n = int(arr.size)
        raw = _pack_most_column_raw(arr, mode="int64")
        if not _verify_most_column_raw(arr, raw, mode="int64"):
            continue
        t0 = time.perf_counter_ns()
        c = zc.compress(raw)
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        d = zd.decompress(c, max_output_size=len(raw))
        t3 = time.perf_counter_ns()
        if d != raw or not _verify_most_column_raw(arr, d, mode="int64"):
            continue
        _accumulate_most_raw_loss(arr, d, "int64", loss_acc)
        tot_o += column_original_size(n, "int")
        tot_c += len(c)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_corad_tricklet_float_columns(
    columns: List[List[float]], trick: int = 40, atoms: int = 4
) -> Tuple[int, int, int, int, float, float]:
    """Legacy entrypoint name; preserve supplied float values in CRD2 float64."""
    return _bench_corad_tricklet_columns(columns, trick, atoms, mode="float64")


def _bench_corad_tricklet_columns(
    columns: List[List[float | int]],
    trick: int = 40,
    atoms: int = 4,
    *,
    mode: str,
) -> Tuple[int, int, int, int, float, float]:
    """Versioned CORAD-lossless v2; verify original lanes before counting them."""
    from . import corad_lossless_stages as corad

    if trick != corad.TRICK or atoms != corad.MAX_ATOMS:
        raise ValueError("CORAD-lossless v2 supports only trick=40 and atoms=4")
    if mode not in {"int64", "float64", "float32"}:
        raise ValueError("unsupported CORAD mode")
    kind = "int64" if mode == "int64" else "float64"
    dtype = "<i8" if kind == "int64" else "<f8"
    wire_data_type = "int" if kind == "int64" else "float"
    tot_o = tot_c = te = td = count = 0
    for col in columns:
        x = np.asarray(col, dtype=dtype)
        if kind == "float64":
            x = x[~np.isnan(x)]
        n = int(x.size)
        if n == 0:
            continue
        raw = x.tobytes()
        t0 = time.perf_counter_ns()
        comp = corad.encode(raw, kind)
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        dec = corad.decode(comp, kind)
        t3 = time.perf_counter_ns()
        if dec != raw:
            raise RuntimeError("CORAD-lossless v2 failed original-lane bitwise verification")
        tot_o += column_original_size(n, wire_data_type)
        tot_c += len(comp)
        te += t1 - t0
        td += t3 - t2
        count += n
    return tot_o, tot_c, te, td, 0.0, float(count)


def bench_corad_tricklet_double_columns(
    columns: List[List[float]],
    trick: int = 40,
    atoms: int = 4,
) -> Tuple[int, int, int, int, float, float]:
    return _bench_corad_tricklet_columns(columns, trick, atoms, mode="float64")


def bench_corad_tricklet_int64_columns(
    columns: List[List[int]],
    trick: int = 40,
    atoms: int = 4,
) -> Tuple[int, int, int, int, float, float]:
    return _bench_corad_tricklet_columns(columns, trick, atoms, mode="int64")


def _float32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def bench_machete_float_rel_columns(columns: List[List[float]], rel: float = 1e-3) -> Tuple[int, int, int, int, float, float]:
    """Legacy float32 wire path; preferred path is double/int64 variants."""
    zc = zstandard.ZstdCompressor(level=3)
    zd = zstandard.ZstdDecompressor()
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        arr = np.array(col, dtype=np.float32)
        n = int(arr.size)
        if n < 1:
            continue
        mx = float(np.max(np.abs(arr)))
        if mx == 0.0:
            mx = 1.0
        scale = float(mx * rel)
        if not (scale > 0.0):
            continue
        inv = 1.0 / scale
        q = np.clip(np.rint(arr * inv), -32768, 32767).astype(np.int16)
        raw = struct.pack("<fI", scale, n) + q.tobytes()
        t0 = time.perf_counter_ns()
        c = zc.compress(raw)
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        d = zd.decompress(c, max_output_size=len(raw))
        t3 = time.perf_counter_ns()
        if len(d) != len(raw):
            continue
        s2, n2 = struct.unpack_from("<fI", d, 0)
        if n2 != n:
            continue
        if _float32_bits(float(s2)) != _float32_bits(scale):
            continue
        q2 = np.frombuffer(d[8: 8 + 2 * n], dtype=np.int16)
        ok = True
        dec_vals: List[float] = []
        orig_vals = [float(arr[i]) for i in range(n)]
        for i in range(n):
            dec = float(q2[i]) * float(s2)
            dec_vals.append(dec)
            err = abs(dec - float(arr[i]))
            bound_abs = float(rel * max(abs(float(arr[i])), mx))
            if err > bound_abs * 1.0001 + 1e-6:
                ok = False
                break
        if not ok:
            continue
        accumulate_double_list_loss(orig_vals, dec_vals, loss_acc)
        tot_o += column_original_size(n, "float")
        tot_c += len(c)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def _bench_machete_rel_columns(
    columns: List[List[float | int]],
    rel: float,
    *,
    mode: str,
) -> Tuple[int, int, int, int, float, float]:
    zc = zstandard.ZstdCompressor(level=3)
    zd = zstandard.ZstdDecompressor()
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    if not (rel > 0.0) or rel >= 1.0:
        return 0, 0, 0, 0, 0.0, 0.0
    is_float_input = mode in ("float64", "float32")
    if mode == "float64":
        arr_dtype = np.float64
        data_type_name = "float"
    elif mode == "int64":
        arr_dtype = np.int64
        data_type_name = "int"
    elif mode == "float32":
        arr_dtype = np.float32
        data_type_name = "float"
    else:
        raise ValueError("unsupported machete mode")
    for col in columns:
        if not col:
            continue
        arr = np.array(col, dtype=arr_dtype)
        n = int(arr.size)
        if n < 1:
            continue
        mx = float(np.max(np.abs(arr.astype(np.float64))))
        if mx == 0.0:
            mx = 1.0
        scale = float(mx * rel)
        if not (scale > 0.0):
            continue
        inv = 1.0 / scale
        q = np.clip(np.rint(arr.astype(np.float64) * inv), -32768, 32767).astype(np.int16)
        raw = struct.pack("<dI", scale, n) + q.tobytes()
        t0 = time.perf_counter_ns()
        c = zc.compress(raw)
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        d = zd.decompress(c, max_output_size=len(raw))
        t3 = time.perf_counter_ns()
        if len(d) != len(raw):
            continue
        s2, n2 = struct.unpack_from("<dI", d, 0)
        if n2 != n:
            continue
        if s2 != scale:
            continue
        q2 = np.frombuffer(d[12: 12 + 2 * n], dtype=np.int16)
        ok = True
        dec_vals: List[float] = []
        orig_vals = [float(arr[i]) for i in range(n)]
        for i in range(n):
            dec = float(q2[i]) * float(s2)
            dec_vals.append(dec)
            err = abs(dec - float(arr[i]))
            bound_abs = float(rel * max(abs(float(arr[i])), mx))
            eps = 1e-6 if is_float_input else 1.0
            if not is_float_input:
                bound_abs = max(bound_abs, 1.0)
            if err > bound_abs * 1.0001 + eps:
                ok = False
                break
        if not ok:
            continue
        if mode == "int64":
            accumulate_long_list_loss(
                [int(arr[i]) for i in range(n)],
                [int(round(dec_vals[i])) for i in range(n)],
                loss_acc,
            )
        else:
            accumulate_double_list_loss(orig_vals, dec_vals, loss_acc)
        tot_o += column_original_size(n, data_type_name)
        tot_c += len(c)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_machete_double_rel_columns(
    columns: List[List[float]],
    rel: float = 1e-3,
) -> Tuple[int, int, int, int]:
    return _bench_machete_rel_columns(columns, rel, mode="float64")


def bench_machete_int64_rel_columns(
    columns: List[List[int]],
    rel: float = 1e-3,
) -> Tuple[int, int, int, int]:
    return _bench_machete_rel_columns(columns, rel, mode="int64")
