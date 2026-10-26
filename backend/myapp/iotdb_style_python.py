"""
IoTDB / TsFile-style encodings in Python (best-effort; not binary-compatible with libtsfile).

- ts_2diff_bos_int / ts_2diff_bos_float: native Python TS_2DIFF+BOS-style
  first-difference payloads. Float path scales each column to int64 before BOS.
- TS_2DIFF (int/float): TsFile second-difference encoder payload (delta blocks + float scale),
  ``compressedSize`` aligned with Java ``CompressionUtils.testEncoding`` / C++ ``encode_ts_2diff_*``.
- RLE / GORILLA / CHIMP / SPRINTZ / RLBE: native TsFile ports in ``myapp.tsfile_encoding`` (aligned with C++/Java).
"""
from __future__ import annotations

from .benchmark_wire import column_original_size

import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from decimal import Decimal, InvalidOperation
from typing import List, Sequence, Tuple

import zstandard

try:
    import numpy as _np
except Exception:  # pragma: no cover - optional acceleration only
    _np = None

_Z = zstandard.ZstdCompressor(level=3)
_ZD = zstandard.ZstdDecompressor()

_REPO_ROOT = os.environ.get("WEB_COMPRESSION_REPO_ROOT", "").strip()
if not _REPO_ROOT:
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TSFILE_PY_ROOT = os.path.join(_REPO_ROOT, "external", "compression", "tsfile", "python")
if os.path.isdir(_TSFILE_PY_ROOT) and _TSFILE_PY_ROOT not in sys.path:
    sys.path.insert(0, _TSFILE_PY_ROOT)

try:
    from tsfile import (
        Compressor as TsFileCompressor,
        TSDataType as TsFileDataType,
        TSEncoding as TsFileEncoding,
        TimeseriesSchema as TsFileTimeseriesSchema,
        TsFileReader,
        TsFileWriter,
        get_tsfile_config,
        set_tsfile_config,
    )
    from tsfile.field import Field
    from tsfile.row_record import RowRecord

    _HAS_TSFILE_PY = True
except Exception:
    _HAS_TSFILE_PY = False


from .tsfile_encoding import bench as _tsfile_bench
from .tsfile_encoding import p02 as _tsfile_p02
from .codec_level_wrappers import PayloadRow, bench_payload_groups

bench_ts_2diff_int_columns = _tsfile_bench.bench_ts_2diff_int_columns
bench_ts_2diff_float_columns = _tsfile_bench.bench_ts_2diff_float_columns
bench_rle_int_iotdb_key = _tsfile_bench.bench_rle_int_columns
bench_rle_float_tsfile_columns = _tsfile_bench.bench_rle_float_columns
bench_gorilla_int_columns = _tsfile_bench.bench_gorilla_int_columns
bench_gorilla_float_columns = _tsfile_bench.bench_gorilla_float_columns
bench_chimp_tsfile_int_columns = _tsfile_bench.bench_chimp_int_columns
bench_chimp_float_columns = _tsfile_bench.bench_chimp_float_columns
bench_sprintz_int_columns = _tsfile_bench.bench_sprintz_int_columns
bench_sprintz_float_columns = _tsfile_bench.bench_sprintz_float_columns
bench_rlbe_int_columns = _tsfile_bench.bench_rlbe_int_columns
bench_rlbe_float_columns = _tsfile_bench.bench_rlbe_float_columns


_BLOCK_DEFAULT_SIZE = 128


def _append_uvarint(out: bytearray, v: int) -> None:
    v = int(v) & 0xFFFFFFFF
    while (v & 0xFFFFFF80) != 0:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)


def _append_u32_be(out: bytearray, v: int) -> None:
    v = int(v)
    out.extend([(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])


def _clz_u32(x: int) -> int:
    if x == 0:
        return 32
    return 32 - x.bit_length()


def _int_to_bytes(src_num: int, result: bytearray, pos: int, width: int) -> None:
    cnt = pos & 7
    index = pos >> 3
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


def _encode_int_delta_block(values: Sequence[int]) -> bytes:
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
            stream.extend(encoding_block)
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


def _convert_float_to_int(value: float, max_point: int, max_point_value: float) -> int:
    if value * max_point_value > 2147483647.0 or value * max_point_value < -2147483648.0:
        if value > 2147483647.0 or value < -2147483648.0:
            return 0xFFFFFFFF
        return int(round(value))
    if math.isnan(value):
        return 0xFFFFFFFF
    return int(round(float(value) * max_point_value))


def encode_ts_2diff_int(values: Sequence[int]) -> bytes:
    return _encode_int_delta_block(values)


def encode_ts_2diff_float(values: Sequence[float], max_point_number: int = 0) -> bytes:
    max_point = max(0, int(max_point_number))
    max_point_value = 10.0**max_point
    ints = [_convert_float_to_int(float(f), max_point, max_point_value) for f in values]
    out = bytearray()
    _append_uvarint(out, max_point)
    out.extend(_encode_int_delta_block(ints))
    return bytes(out)


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
            index += 1
            cnt = 0
    return ret


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


def decode_ts_2diff_int(blob: bytes) -> List[int]:
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
    ints = decode_ts_2diff_int(blob[pos:])
    if not ints:
        return []
    if max_point <= 0:
        return [float(v) for v in ints]
    scale = 10.0 ** int(max_point)
    return [float(v) / scale for v in ints]


def _float_columns_match(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if struct.unpack("<f", struct.pack("<f", float(x)))[0] != struct.unpack(
            "<f", struct.pack("<f", float(y))
        )[0]:
            return False
    return True


def _ts2diff_bos_jar_path() -> str:
    from .bench_executable_paths import benchmark_java_jar_path

    return str(benchmark_java_jar_path())


def _pick_bos_or_ts2diff(
    bos: Tuple[int, int, int, int],
    ts2: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Mirror Java ``Ts2DiffBosImprove``: use TS_2DIFF metrics when smaller than BOS."""
    _bo, bc, _bte, _btd = bos
    to, tc, tte, ttd = ts2
    if tc > 0 and (bc <= 0 or tc < bc):
        return to, tc, tte, ttd
    if bc > 0:
        return bos
    if tc > 0:
        return ts2
    return 0, 0, 0, 0


def _run_ts2diff_bos_jvm(main_class: str, payload: bytes) -> Tuple[int, int, int, int]:
    jar = _ts2diff_bos_jar_path()
    if not os.path.isfile(jar):
        return 0, 0, 0, 0
    env = dict(os.environ)
    env.setdefault("WEB_COMPRESSION_REPO_ROOT", _REPO_ROOT)
    try:
        proc = subprocess.run(
            ["java", "-cp", jar, main_class],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=600,
            check=False,
            env=env,
        )
    except (subprocess.SubprocessError, OSError):
        return 0, 0, 0, 0
    if proc.returncode != 0 or len(proc.stdout) < 32:
        return 0, 0, 0, 0
    o, c, te, td = struct.unpack("<qqqq", proc.stdout[:32])
    if c <= 0 or o <= 0:
        return 0, 0, 0, 0
    return int(o), int(c), int(te), int(td)


def _run_ts2diff_bos_payloads_jvm(
    main_class: str, payload: bytes
) -> List[PayloadRow | None]:
    jar = _ts2diff_bos_jar_path()
    if not os.path.isfile(jar):
        return []
    env = dict(os.environ)
    env.setdefault("WEB_COMPRESSION_REPO_ROOT", _REPO_ROOT)
    env["WEB_COMPRESSION_BOS_PAYLOADS"] = "1"
    try:
        proc = subprocess.run(
            ["java", "-cp", jar, main_class],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=600,
            check=False,
            env=env,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if proc.returncode != 0 or len(proc.stdout) < 4:
        return []
    data = proc.stdout
    try:
        count = struct.unpack_from("<i", data, 0)[0]
    except struct.error:
        return []
    if count < 0:
        return []
    pos = 4
    rows: List[PayloadRow | None] = []
    for _ in range(count):
        if pos + 28 > len(data):
            return []
        o, te, td, payload_len = struct.unpack_from("<qqqi", data, pos)
        pos += 28
        if payload_len <= 0:
            rows.append(None)
            continue
        if pos + payload_len > len(data):
            return []
        body = data[pos : pos + payload_len]
        pos += payload_len
        rows.append(PayloadRow(int(o), body, int(te), int(td)) if o > 0 else None)
    return rows


_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_U64_MASK = (1 << 64) - 1


def _to_i64(v: int) -> int:
    u = int(v) & _U64_MASK
    return u - (1 << 64) if u >= (1 << 63) else u


def _validate_i64(v: int) -> int:
    iv = int(v)
    if iv < _I64_MIN or iv > _I64_MAX:
        raise OverflowError("BOS int64 value out of range")
    return iv


def _zigzag_encode64(v: int) -> int:
    x = _to_i64(v)
    return ((x << 1) ^ (x >> 63)) & _U64_MASK


def _zigzag_decode64(v: int) -> int:
    u = int(v) & _U64_MASK
    return _to_i64((u >> 1) ^ (-(u & 1)))


def _bit_width64(v: int) -> int:
    u = int(v) & _U64_MASK
    return max(1, u.bit_length())


def _mask_bits64(width: int) -> int:
    return _U64_MASK if width >= 64 else (1 << width) - 1


def _write_i32_be(out: bytearray, v: int) -> None:
    out.extend(struct.pack(">i", int(v)))


def _patch_i32_be(out: bytearray, pos: int, v: int) -> None:
    struct.pack_into(">i", out, pos, int(v))


def _write_i64_be(out: bytearray, v: int) -> None:
    out.extend(struct.pack(">q", _to_i64(v)))


def _read_i32_be(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 4 > len(data):
        raise ValueError("truncated BOS int32")
    return struct.unpack(">i", data[pos : pos + 4])[0], pos + 4


def _read_i64_be(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 8 > len(data):
        raise ValueError("truncated BOS int64")
    return struct.unpack(">q", data[pos : pos + 8])[0], pos + 8


def _append_packed_delta_width4(out: bytearray, vals: Sequence[int], off: int, blen: int) -> None:
    prev = vals[off]
    i = 1
    while i + 1 < blen:
        cur = vals[off + i]
        z0 = _zigzag_encode64(_to_i64(cur - prev)) & 0xF
        prev = cur
        cur = vals[off + i + 1]
        z1 = _zigzag_encode64(_to_i64(cur - prev)) & 0xF
        prev = cur
        out.append((z0 << 4) | z1)
        i += 2
    if i < blen:
        cur = vals[off + i]
        z0 = _zigzag_encode64(_to_i64(cur - prev)) & 0xF
        out.append(z0 << 4)


def _append_packed_delta_width3(out: bytearray, vals: Sequence[int], off: int, blen: int) -> None:
    prev = vals[off]
    i = 1
    full_end = 1 + ((blen - 1) // 8) * 8
    while i < full_end:
        zs = [0] * 8
        for zi in range(8):
            cur = vals[off + i + zi]
            zs[zi] = _zigzag_encode64(_to_i64(cur - prev)) & 0x7
            prev = cur
        out.append((zs[0] << 5) | (zs[1] << 2) | (zs[2] >> 1))
        out.append(((zs[2] & 0x1) << 7) | (zs[3] << 4) | (zs[4] << 1) | (zs[5] >> 2))
        out.append(((zs[5] & 0x3) << 6) | (zs[6] << 3) | zs[7])
        i += 8
    buf = 0
    bits = 0
    while i < blen:
        cur = vals[off + i]
        z = _zigzag_encode64(_to_i64(cur - prev)) & 0x7
        prev = cur
        buf = (buf << 3) | z
        bits += 3
        if bits >= 8:
            out.append((buf >> (bits - 8)) & 0xFF)
            bits -= 8
            buf = 0 if bits == 0 else buf & ((1 << bits) - 1)
        i += 1
    if bits > 0:
        out.append((buf << (8 - bits)) & 0xFF)


def _delta_zigzag_numpy(vals: Sequence[int], off: int, blen: int):
    if _np is None or blen < 4096:
        return None
    arr = _np.asarray(vals[off : off + blen], dtype=_np.int64)
    deltas = arr[1:] - arr[:-1]
    return ((deltas << _np.int64(1)) ^ (deltas >> _np.int64(63))).astype(_np.uint64, copy=False)


def _append_packed_delta_numpy(out: bytearray, z, width: int, expected_payload_len: int) -> bool:
    if _np is None or width not in (3, 4):
        return False
    count = int(z.size)
    if count <= 0:
        return False
    z8 = z.astype(_np.uint8, copy=False)
    if width == 4:
        padded_len = ((count + 1) // 2) * 2
        if padded_len != count:
            padded = _np.zeros(padded_len, dtype=_np.uint8)
            padded[:count] = z8
            z8 = padded
        packed = ((z8[0::2] << _np.uint8(4)) | z8[1::2]).astype(_np.uint8, copy=False)
    else:
        padded_len = ((count + 7) // 8) * 8
        if padded_len != count:
            padded = _np.zeros(padded_len, dtype=_np.uint8)
            padded[:count] = z8
            z8 = padded
        groups = z8.reshape((-1, 8))
        packed = _np.empty(groups.shape[0] * 3, dtype=_np.uint8)
        packed[0::3] = (groups[:, 0] << _np.uint8(5)) | (groups[:, 1] << _np.uint8(2)) | (groups[:, 2] >> _np.uint8(1))
        packed[1::3] = (
            ((groups[:, 2] & _np.uint8(1)) << _np.uint8(7))
            | (groups[:, 3] << _np.uint8(4))
            | (groups[:, 4] << _np.uint8(1))
            | (groups[:, 5] >> _np.uint8(2))
        )
        packed[2::3] = ((groups[:, 5] & _np.uint8(3)) << _np.uint8(6)) | (groups[:, 6] << _np.uint8(3)) | groups[:, 7]
    out.extend(packed[:expected_payload_len].tobytes())
    return True


def _encode_long_delta_bitpack(
    values: Sequence[int], segment_rows: int | None = None, *, validate: bool = True
) -> bytes:
    vals = [_validate_i64(v) for v in values] if validate else values
    n = len(vals)
    seg = max(1, int(segment_rows or n or 1))
    out = bytearray()
    _write_i32_be(out, n)
    _write_i32_be(out, seg)
    for off in range(0, n, seg):
        blen = min(seg, n - off)
        _write_i32_be(out, blen)
        first = vals[off]
        _write_i64_be(out, first)
        if blen == 1:
            out.append(1)
            _write_i32_be(out, 0)
            continue

        np_z = _delta_zigzag_numpy(vals, off, blen)
        if np_z is not None:
            max_z = int(np_z.max(initial=0))
            width = max(1, max_z.bit_length())
        else:
            width = 1
            prev = first
            for i in range(1, blen):
                cur = vals[off + i]
                delta = _to_i64(cur - prev)
                z = _zigzag_encode64(delta)
                width = max(width, _bit_width64(z))
                prev = cur
        packed_width = 64 if width > 56 else width
        out.append(packed_width & 0xFF)
        value_count = blen - 1
        expected_payload_len = (
            value_count * 8
            if packed_width >= 64
            else (value_count * packed_width + 7) // 8
        )
        payload_len_pos = len(out)
        _write_i32_be(out, 0)
        payload_start = len(out)
        prev = first
        if packed_width >= 64:
            for i in range(1, blen):
                cur = vals[off + i]
                z = _zigzag_encode64(_to_i64(cur - prev))
                out.extend(struct.pack(">Q", z & _U64_MASK))
                prev = cur
        elif np_z is not None and _append_packed_delta_numpy(out, np_z, packed_width, expected_payload_len):
            pass
        elif packed_width == 4:
            _append_packed_delta_width4(out, vals, off, blen)
        elif packed_width == 3:
            _append_packed_delta_width3(out, vals, off, blen)
        else:
            buf = 0
            bits = 0
            mask = _mask_bits64(packed_width)
            for i in range(1, blen):
                cur = vals[off + i]
                z = _zigzag_encode64(_to_i64(cur - prev))
                buf = (buf << packed_width) | (z & mask)
                bits += packed_width
                while bits >= 8:
                    out.append((buf >> (bits - 8)) & 0xFF)
                    bits -= 8
                    if bits == 0:
                        buf = 0
                    else:
                        buf &= (1 << bits) - 1
                prev = cur
            if bits > 0:
                out.append((buf << (8 - bits)) & 0xFF)
        written = len(out) - payload_start
        if written != expected_payload_len:
            raise ValueError("internal BOS payload length mismatch")
        _patch_i32_be(out, payload_len_pos, written)
    return bytes(out)


def _decode_long_delta_bitpack(blob: bytes) -> list[int]:
    pos = 0
    n, pos = _read_i32_be(blob, pos)
    seg, pos = _read_i32_be(blob, pos)
    if n < 0 or seg <= 0:
        raise ValueError("invalid BOS header")
    out: list[int] = [0] * n
    out_pos = 0
    blob_len = len(blob)
    while out_pos < n:
        blen, pos = _read_i32_be(blob, pos)
        if blen <= 0 or out_pos + blen > n:
            raise ValueError("invalid BOS segment length")
        first, pos = _read_i64_be(blob, pos)
        if pos >= blob_len:
            raise ValueError("truncated BOS width")
        width = blob[pos]
        pos += 1
        payload_len, pos = _read_i32_be(blob, pos)
        if payload_len < 0 or pos + payload_len > blob_len:
            raise ValueError("invalid BOS payload length")
        payload_start = pos
        payload_end = pos + payload_len
        pos += payload_len

        out[out_pos] = first
        if blen > 1:
            if width <= 0 or width > 64:
                raise ValueError("invalid BOS bit width")
            prev = first
            if width >= 64:
                need = 8 * (blen - 1)
                if payload_len < need:
                    raise ValueError("truncated BOS 64-bit payload")
                p = payload_start
                for _ in range(1, blen):
                    z = struct.unpack_from(">Q", blob, p)[0]
                    p += 8
                    prev = _to_i64(prev + _zigzag_decode64(z))
                    out[out_pos + _] = prev
            elif width == 4:
                p = payload_start
                i = 1
                while i + 1 < blen:
                    b = blob[p]
                    p += 1
                    z = b >> 4
                    prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                    out[out_pos + i] = prev
                    z = b & 0xF
                    prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                    out[out_pos + i + 1] = prev
                    i += 2
                if i < blen:
                    if p >= payload_end:
                        raise ValueError("truncated BOS payload")
                    z = blob[p] >> 4
                    prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                    out[out_pos + i] = prev
            elif width == 1:
                p = payload_start
                i = 1
                while i + 7 < blen:
                    if p >= payload_end:
                        raise ValueError("truncated BOS payload")
                    b = blob[p]
                    p += 1
                    for shift in (7, 6, 5, 4, 3, 2, 1, 0):
                        z = (b >> shift) & 0x1
                        prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                        out[out_pos + i] = prev
                        i += 1
                if i < blen:
                    if p >= payload_end:
                        raise ValueError("truncated BOS payload")
                    b = blob[p]
                    for shift in range(7, -1, -1):
                        if i >= blen:
                            break
                        z = (b >> shift) & 0x1
                        prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                        out[out_pos + i] = prev
                        i += 1
            elif width == 2:
                p = payload_start
                i = 1
                while i + 3 < blen:
                    if p >= payload_end:
                        raise ValueError("truncated BOS payload")
                    b = blob[p]
                    p += 1
                    for shift in (6, 4, 2, 0):
                        z = (b >> shift) & 0x3
                        prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                        out[out_pos + i] = prev
                        i += 1
                if i < blen:
                    if p >= payload_end:
                        raise ValueError("truncated BOS payload")
                    b = blob[p]
                    for shift in range(6, -1, -2):
                        if i >= blen:
                            break
                        z = (b >> shift) & 0x3
                        prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                        out[out_pos + i] = prev
                        i += 1
            elif width == 3:
                p = payload_start
                i = 1
                full_end = 1 + ((blen - 1) // 8) * 8
                while i < full_end:
                    if p + 3 > payload_end:
                        raise ValueError("truncated BOS payload")
                    b0 = blob[p]
                    b1 = blob[p + 1]
                    b2 = blob[p + 2]
                    p += 3
                    zs = (
                        (b0 >> 5) & 0x7,
                        (b0 >> 2) & 0x7,
                        ((b0 & 0x3) << 1) | (b1 >> 7),
                        (b1 >> 4) & 0x7,
                        (b1 >> 1) & 0x7,
                        ((b1 & 0x1) << 2) | (b2 >> 6),
                        (b2 >> 3) & 0x7,
                        b2 & 0x7,
                    )
                    for z in zs:
                        prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                        out[out_pos + i] = prev
                        i += 1
                buf = 0
                bits = 0
                while i < blen:
                    while bits < 3:
                        if p >= payload_end:
                            raise ValueError("truncated BOS payload")
                        buf = (buf << 8) | blob[p]
                        bits += 8
                        p += 1
                    z = (buf >> (bits - 3)) & 0x7
                    bits -= 3
                    buf = 0 if bits == 0 else buf & ((1 << bits) - 1)
                    prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                    out[out_pos + i] = prev
                    i += 1
            elif width == 8:
                need = blen - 1
                if payload_len < need:
                    raise ValueError("truncated BOS payload")
                p = payload_start
                i = 1
                while i + 3 < blen:
                    z0 = blob[p]
                    z1 = blob[p + 1]
                    z2 = blob[p + 2]
                    z3 = blob[p + 3]
                    p += 4
                    prev = _to_i64(prev + ((z0 >> 1) ^ -(z0 & 1)))
                    out[out_pos + i] = prev
                    prev = _to_i64(prev + ((z1 >> 1) ^ -(z1 & 1)))
                    out[out_pos + i + 1] = prev
                    prev = _to_i64(prev + ((z2 >> 1) ^ -(z2 & 1)))
                    out[out_pos + i + 2] = prev
                    prev = _to_i64(prev + ((z3 >> 1) ^ -(z3 & 1)))
                    out[out_pos + i + 3] = prev
                    i += 4
                while i < blen:
                    z = blob[p]
                    p += 1
                    prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                    out[out_pos + i] = prev
                    i += 1
            elif width == 16:
                need = (blen - 1) * 2
                if payload_len < need:
                    raise ValueError("truncated BOS payload")
                p = payload_start
                for i in range(1, blen):
                    z = (blob[p] << 8) | blob[p + 1]
                    p += 2
                    prev = _to_i64(prev + ((z >> 1) ^ -(z & 1)))
                    out[out_pos + i] = prev
            else:
                mask = _mask_bits64(width)
                buf = 0
                bits = 0
                p = payload_start
                for _ in range(1, blen):
                    while bits < width:
                        if p >= payload_end:
                            raise ValueError("truncated BOS payload")
                        buf = (buf << 8) | blob[p]
                        bits += 8
                        p += 1
                    z = (buf >> (bits - width)) & mask
                    bits -= width
                    buf = 0 if bits == 0 else buf & ((1 << bits) - 1)
                    prev = _to_i64(prev + _zigzag_decode64(z))
                    out[out_pos + _] = prev
        out_pos += blen
    if pos != blob_len:
        raise ValueError("trailing BOS payload bytes")
    return out


def _raw_int_payload_rows(columns: List[List[int]]) -> List[PayloadRow | None]:
    rows: List[PayloadRow | None] = []
    for col in columns:
        if not col:
            rows.append(None)
            continue
        try:
            vals = [_validate_i64(x) for x in col]
            t0 = time.perf_counter_ns()
            enc = struct.pack(">i", len(vals)) + b"".join(struct.pack(">q", v) for v in vals)
            t1 = time.perf_counter_ns()
            n = struct.unpack_from(">i", enc, 0)[0]
            dec = [struct.unpack_from(">q", enc, 4 + i * 8)[0] for i in range(n)]
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if dec != vals:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(vals), "int"), enc, t1 - t0, t2 - t1))
    return rows


def _raw_float_payload_rows(columns: List[List[float]]) -> List[PayloadRow | None]:
    rows: List[PayloadRow | None] = []
    for col in columns:
        if not col:
            rows.append(None)
            continue
        try:
            vals = [float(v) for v in col]
            t0 = time.perf_counter_ns()
            enc = struct.pack(">i", len(vals)) + b"".join(struct.pack(">d", v) for v in vals)
            t1 = time.perf_counter_ns()
            n = struct.unpack_from(">i", enc, 0)[0]
            dec = [struct.unpack_from(">d", enc, 4 + i * 8)[0] for i in range(n)]
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if dec != vals:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(vals), "float"), enc, t1 - t0, t2 - t1))
    return rows


def _encode_int_rle(vals: Sequence[int]) -> bytes:
    if not vals:
        return struct.pack(">ii", 0, 0)
    runs: list[tuple[int, int]] = []
    cur = _validate_i64(vals[0])
    count = 1
    for raw in vals[1:]:
        v = _validate_i64(raw)
        if v == cur and count < 0x7FFFFFFF:
            count += 1
        else:
            runs.append((cur, count))
            cur = v
            count = 1
    runs.append((cur, count))
    out = bytearray(struct.pack(">ii", len(vals), len(runs)))
    for value, run_len in runs:
        out.extend(struct.pack(">qi", value, run_len))
    return bytes(out)


def _decode_int_rle(blob: bytes) -> list[int]:
    pos = 0
    n, runs = struct.unpack_from(">ii", blob, pos)
    pos += 8
    out: list[int] = []
    for _ in range(runs):
        value, run_len = struct.unpack_from(">qi", blob, pos)
        pos += 12
        out.extend([value] * run_len)
    if len(out) != n:
        raise ValueError("invalid BOS RLE payload")
    return out


def _rle_int_payload_rows(columns: List[List[int]]) -> List[PayloadRow | None]:
    rows: List[PayloadRow | None] = []
    for col in columns:
        if not col:
            rows.append(None)
            continue
        try:
            vals = [_validate_i64(x) for x in col]
            t0 = time.perf_counter_ns()
            enc = _encode_int_rle(vals)
            t1 = time.perf_counter_ns()
            dec = _decode_int_rle(enc)
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if dec != vals:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(vals), "int"), enc, t1 - t0, t2 - t1))
    return rows


def _scaled_rle_float_payload_rows(
    columns: List[List[float]], max_point_per_column: Sequence[int] | None = None
) -> List[PayloadRow | None]:
    from .research_codec_python import (
        scale_float_column_to_ints,
        scale_float_column_to_ints_with_max_point,
    )

    rows: List[PayloadRow | None] = []
    for ci, col in enumerate(columns):
        if not col:
            rows.append(None)
            continue
        if max_point_per_column is not None and ci < len(max_point_per_column):
            ints = scale_float_column_to_ints_with_max_point(col, int(max_point_per_column[ci]))
            if ints is None:
                ints = scale_float_column_to_ints(col)
        else:
            ints = scale_float_column_to_ints(col)
        if ints is None:
            rows.append(None)
            continue
        try:
            vals = [_validate_i64(x) for x in ints]
            t0 = time.perf_counter_ns()
            enc = _encode_int_rle(vals)
            t1 = time.perf_counter_ns()
            dec = _decode_int_rle(enc)
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if dec != vals:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(col), "float"), enc, t1 - t0, t2 - t1))
    return rows


def _bos_int_payload_rows(columns: List[List[int]]) -> List[PayloadRow | None]:
    rows: List[PayloadRow | None] = []
    for col in columns:
        if not col:
            rows.append(None)
            continue
        try:
            vals = [_validate_i64(x) for x in col]
            t0 = time.perf_counter_ns()
            enc = _encode_long_delta_bitpack(vals, validate=False)
            t1 = time.perf_counter_ns()
            dec = _decode_long_delta_bitpack(enc)
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if dec != vals:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(vals), "int"), enc, t1 - t0, t2 - t1))
    return rows


def _bos_float_payload_rows(
    columns: List[List[float]], max_point_per_column: Sequence[int] | None = None
) -> List[PayloadRow | None]:
    from .research_codec_python import (
        scale_float_column_to_ints,
        scale_float_column_to_ints_with_max_point,
    )

    rows: List[PayloadRow | None] = []
    for ci, col in enumerate(columns):
        if not col:
            rows.append(None)
            continue
        if max_point_per_column is not None and ci < len(max_point_per_column):
            ints = scale_float_column_to_ints_with_max_point(col, int(max_point_per_column[ci]))
            if ints is None:
                ints = scale_float_column_to_ints(col)
        else:
            ints = scale_float_column_to_ints(col)
        if ints is None:
            rows.append(None)
            continue
        try:
            vals = [_validate_i64(x) for x in ints]
            t0 = time.perf_counter_ns()
            enc = _encode_long_delta_bitpack(vals, validate=False)
            t1 = time.perf_counter_ns()
            dec = _decode_long_delta_bitpack(enc)
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if dec != vals:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(col), "float"), enc, t1 - t0, t2 - t1))
    return rows


def _ts2diff_int_payload_rows(columns: List[List[int]]) -> List[PayloadRow | None]:
    rows: List[PayloadRow | None] = []
    for col in columns:
        if not col:
            rows.append(None)
            continue
        vals = [int(x) for x in col]
        try:
            t0 = time.perf_counter_ns()
            enc = _tsfile_p02.encode_ts_2diff_int64(vals)
            t1 = time.perf_counter_ns()
            dec = _tsfile_p02.decode_ts_2diff_int64(enc)
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if dec != vals:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(vals), "int"), enc, t1 - t0, t2 - t1))
    return rows


def _ts2diff_float_payload_rows(
    columns: List[List[float]], max_point_per_column: Sequence[int] | None = None
) -> List[PayloadRow | None]:
    rows: List[PayloadRow | None] = []
    for ci, col in enumerate(columns):
        if not col:
            rows.append(None)
            continue
        if max_point_per_column is not None and ci < len(max_point_per_column):
            mp = int(max_point_per_column[ci])
        else:
            mp = _max_decimal_scale_in_column(col)
        try:
            eff_mp = _tsfile_bench._clamp_ts2diff_max_point_for_double(col, mp)
            t0 = time.perf_counter_ns()
            enc = _tsfile_p02.encode_ts_2diff_double(col, eff_mp)
            t1 = time.perf_counter_ns()
            dec = _tsfile_p02.decode_ts_2diff_double(enc)
            t2 = time.perf_counter_ns()
        except Exception:
            rows.append(None)
            continue
        if not dec:
            rows.append(None)
            continue
        rows.append(PayloadRow(column_original_size(len(col), "float"), enc, t1 - t0, t2 - t1))
    return rows


def _group_payload_rows(
    *row_sets: Sequence[PayloadRow | None],
) -> list[tuple[PayloadRow | None, ...]]:
    n = max((len(rows) for rows in row_sets), default=0)
    return [
        tuple(rows[i] if i < len(rows) else None for rows in row_sets)
        for i in range(n)
    ]


def bench_ts_2diff_bos_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    """Native INT64 BOS payloads, floored by native TS_2DIFF INT64."""
    groups = _group_payload_rows(_bos_int_payload_rows(columns), _ts2diff_int_payload_rows(columns))
    picked = bench_payload_groups(groups, ("raw",), policy="size", bytes_per_ns=0.0)
    if picked[1] <= 0:
        return 0, 0, 0, 0
    return picked


def bench_ts_2diff_bos_float_columns(
    columns: List[List[float]], max_point_per_column: Sequence[int] | None = None
) -> Tuple[int, int, int, int]:
    """Native DOUBLE BOS payloads after int64 scaling, floored by native TS_2DIFF DOUBLE."""
    groups = _group_payload_rows(
        _bos_float_payload_rows(columns, max_point_per_column),
        _ts2diff_float_payload_rows(columns, max_point_per_column),
    )
    picked = bench_payload_groups(groups, ("raw",), policy="size", bytes_per_ns=0.0)
    if picked[1] <= 0:
        return 0, 0, 0, 0
    return picked


def bench_ts_2diff_bos_level_int_columns(
    columns: List[List[int]], _level: int | None = None
) -> Tuple[int, int, int, int]:
    return bench_ts_2diff_bos_int_columns(columns)


def bench_ts_2diff_bos_level_float_columns(
    columns: List[List[float]],
    _level: int | None = None,
    max_point_per_column: Sequence[int] | None = None,
) -> Tuple[int, int, int, int]:
    return bench_ts_2diff_bos_float_columns(columns, max_point_per_column)


def encode_sprintz_approx(values: List[int]) -> bytes:
    v = [int(x) for x in values]
    if len(v) < 3:
        return struct.pack("<i", len(v)) + struct.pack("<" + "i" * len(v), *v)
    d1 = [v[i] - v[i - 1] for i in range(1, len(v))]
    d2 = [d1[i] - d1[i - 1] for i in range(1, len(d1))]
    return struct.pack("<3i", len(v), v[0], v[1]) + struct.pack(
        "<" + "i" * len(d2), *d2
    )


def decode_sprintz_approx(blob: bytes) -> List[int]:
    if len(blob) < 4:
        return []
    n = struct.unpack("<i", blob[:4])[0]
    if n <= 0:
        return []
    if n < 3:
        return list(struct.unpack("<" + "i" * n, blob[4: 4 + 4 * n]))
    a, b = struct.unpack("<2i", blob[4:12])
    rest = blob[12:]
    need = (n - 2) * 4
    if len(rest) < need:
        return []
    d2 = list(struct.unpack("<" + "i" * (n - 2), rest[:need]))
    out = [a, b]
    d1_cur = b - a
    for j in range(n - 2):
        d1_next = d1_cur + d2[j]
        out.append(out[-1] + d1_next)
        d1_cur = d1_next
    return out


bench_sprintz_approx_int_columns = bench_sprintz_int_columns


def encode_rlbe_int(values: List[int]) -> bytes:
    raw = struct.pack("<" + "i" * len(values), *[int(x) for x in values])
    n = len(values)
    if n == 0:
        return struct.pack("<i", 0)
    out = bytearray(struct.pack("<i", n))
    for lane in range(4):
        plane = bytes(raw[lane + i * 4] for i in range(n))
        c = _Z.compress(plane)
        out.extend(struct.pack("<i", len(c)))
        out.extend(c)
    return bytes(out)


def decode_rlbe_int(blob: bytes) -> List[int]:
    if len(blob) < 4:
        return []
    n = struct.unpack("<i", blob[:4])[0]
    pos = 4
    planes: List[bytes] = []
    for _ in range(4):
        if pos + 4 > len(blob):
            return []
        ln = struct.unpack("<i", blob[pos : pos + 4])[0]
        pos += 4
        comp = blob[pos : pos + ln]
        pos += ln
        planes.append(_ZD.decompress(comp))
    raw = bytearray(n * 4)
    for i in range(n):
        for lane in range(4):
            if i >= len(planes[lane]):
                return []
            raw[i * 4 + lane] = planes[lane][i]
    return list(struct.unpack("<" + "i" * n, bytes(raw)))


bench_rlbe_approx_int_columns = bench_rlbe_int_columns


def _decimal_scale_from_float(v: float) -> int:
    if not isinstance(v, (float, int)):
        return 0
    fv = float(v)
    if not (fv == fv) or fv in (float("inf"), float("-inf")):
        return 0
    try:
        bd = Decimal(str(fv)).normalize()
    except (InvalidOperation, ValueError):
        return 0
    exp = bd.as_tuple().exponent
    if exp >= 0:
        return 0
    # Keep a practical bound same as Java Ts2DiffBosImprove.
    return min(24, -exp)


def _max_decimal_scale_in_column(col: List[float]) -> int:
    m = 0
    for v in col:
        m = max(m, _decimal_scale_from_float(float(v)))
    return m


bench_tsfile_rle_float_columns = bench_rle_float_tsfile_columns
bench_tsfile_gorilla_float_columns = bench_gorilla_float_columns
bench_tsfile_chimp_float_columns = bench_chimp_float_columns
bench_tsfile_sprintz_float_columns = bench_sprintz_float_columns
bench_tsfile_rlbe_float_columns = bench_rlbe_float_columns
