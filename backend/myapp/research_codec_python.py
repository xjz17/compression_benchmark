"""
Research-style codecs in pure Python (parity targets Java ExternalCompressionBench rows).

- leco_float / alp_float: match Java scalar logic; integer stage uses zstd on packed int32
  (Java uses FastPFOR+VariableByte).
- fastpfor_int / fastpfor_float: zstd on little-endian int32 payload (Java: FastPFOR+VB).
"""
from __future__ import annotations

from .benchmark_wire import column_original_size
from .native_lz_family import compress_native_lz_family, decompress_native_lz_family

import math
import os
import struct
import time
from typing import List, Optional, Tuple

_LECO_MAX_SCALE = 15


def _zstd3_compress(raw: bytes) -> bytes:
    return compress_native_lz_family(raw, "zstd_level_3")


def _zstd3_decompress(blob: bytes, expected_size: int) -> bytes:
    return decompress_native_lz_family(blob, "zstd_level_3", expected_size)


def _alp_force_raw_fallback() -> bool:
    return os.environ.get("WEB_COMPRESSION_ALP_FORCE_RAW_FALLBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

# --- ALP (same tables / logic as java AlpFloatBench) ---
_MAX_EXP = 10
_MAGIC = 12582912.0
_EXC_COST = 32 + 16
_ENC_HI = 9223372036854774784.0
_ENC_LO = -9223372036854774784.0
_D64_MAGIC = 6755399441055744.0
_D64_EXC_COST = 64 + 16
_FRAC = (
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
)
_EXP = (
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
    1_000_000_000.0,
    10_000_000_000.0,
)
_FACT = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000)
_MAX_FAC = len(_FACT) - 1


def _double_bits(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def _u64_to_i64(bits: int) -> int:
    u = int(bits) & 0xFFFFFFFFFFFFFFFF
    return u - (1 << 64) if u >= (1 << 63) else u


def _double_bits_i64(v: float) -> int:
    """Java/C++ long-compatible signed int64 view of double raw bits."""
    return _u64_to_i64(_double_bits(v))


def _float_bits(v: float) -> int:
    """@deprecated Use {@link _double_bits} for benchmark wire format."""
    return _double_bits(v)


def _double_to_java_int(d: float) -> int:
    """Narrowing double → int32 per JLS (NaN → 0; out of range → clamp)."""
    if math.isnan(d):
        return 0
    if d >= 2**31:
        return 2**31 - 1
    if d < -(2**31):
        return -(2**31)
    return int(d)


def _alp_impossible(n: float) -> bool:
    if not (n == n) or n in (float("inf"), float("-inf")):
        return True
    if float(n) > _ENC_HI or float(n) < _ENC_LO:
        return True
    return False


def _alp_enc_val(value: float, fac: int, exp: int) -> int:
    tmp = float(value) * _EXP[exp] * _FRAC[fac]
    if _alp_impossible(tmp):
        return 2**31 - 1
    tmp = tmp + _MAGIC - _MAGIC
    return _double_to_java_int(tmp)


def _alp_dec_val(enc: int, fac: int, exp: int) -> float:
    return float(enc * _FACT[fac] * _FRAC[exp])


def _u32_span(min_e: int, max_e: int) -> int:
    d = (max_e & 0xFFFFFFFF) - (min_e & 0xFFFFFFFF)
    if d <= 0:
        return 0
    return d.bit_length()


def _alp_best_fe(vals: List[float], n: int) -> tuple[int, int] | None:
    if n == 1:
        v = vals[0]
        if _alp_impossible(v):
            return None
        bt = _float_bits(v)
        for exp in range(_MAX_EXP, -1, -1):
            for fac in range(min(exp, _MAX_FAC), -1, -1):
                enc = _alp_enc_val(v, fac, exp)
                dec = _alp_dec_val(enc, fac, exp)
                if _float_bits(dec) == bt:
                    return fac, exp
        return None
    best_cost = 1 << 62
    best_pair: tuple[int, int] | None = None
    for exp in range(_MAX_EXP, -1, -1):
        for fac in range(min(exp, _MAX_FAC), -1, -1):
            non_exc = 0
            min_e = 2**31 - 1
            max_e = -2**31
            bad = False
            for v in vals:
                if _alp_impossible(v):
                    bad = True
                    break
                enc = _alp_enc_val(v, fac, exp)
                dec = _alp_dec_val(enc, fac, exp)
                if _float_bits(dec) == _float_bits(v):
                    non_exc += 1
                    min_e = min(min_e, enc)
                    max_e = max(max_e, enc)
            if bad or non_exc < 2:
                continue
            bits = _u32_span(min_e, max_e)
            cost = n * bits + (n - non_exc) * _EXC_COST
            if cost < best_cost:
                best_cost = cost
                best_pair = (fac, exp)
    return best_pair


def _alp_d64_impossible(n: float) -> bool:
    if not (n == n) or n in (float("inf"), float("-inf")):
        return True
    return float(n) > _ENC_HI or float(n) < _ENC_LO


def _alp_d64_enc_val(value: float, fac: int, exp: int) -> int:
    tmp = float(value) * _EXP[exp] * _FRAC[fac]
    if _alp_d64_impossible(tmp):
        return 2**63 - 1
    tmp = tmp + _D64_MAGIC - _D64_MAGIC
    return int(tmp)


def _alp_d64_dec_val(enc: int, fac: int, exp: int) -> float:
    return float(enc * _FACT[fac] * _FRAC[exp])


def _u64_span_i64(min_e: int, max_e: int) -> int:
    delta = (int(max_e) & 0xFFFFFFFFFFFFFFFF) - (int(min_e) & 0xFFFFFFFFFFFFFFFF)
    if delta <= 0:
        return 0
    return delta.bit_length()


def _alp_d64_best_fe(vals: List[float], n: int) -> tuple[int, int] | None:
    if n == 1:
        v = vals[0]
        if _alp_d64_impossible(v):
            return None
        vb = _double_bits(v)
        for exp in range(_MAX_EXP, -1, -1):
            for fac in range(min(exp, _MAX_FAC), -1, -1):
                enc = _alp_d64_enc_val(v, fac, exp)
                dec = _alp_d64_dec_val(enc, fac, exp)
                if _double_bits(dec) == vb:
                    return fac, exp
        return None
    best_cost = 1 << 62
    best_pair: tuple[int, int] | None = None
    for exp in range(_MAX_EXP, -1, -1):
        for fac in range(min(exp, _MAX_FAC), -1, -1):
            non_exc = 0
            min_e = -1
            max_e = 0
            bad = False
            for v in vals:
                if _alp_d64_impossible(v):
                    bad = True
                    break
                enc = _alp_d64_enc_val(v, fac, exp)
                dec = _alp_d64_dec_val(enc, fac, exp)
                if _double_bits(dec) == _double_bits(v):
                    non_exc += 1
                    if (enc & 0xFFFFFFFFFFFFFFFF) < (min_e & 0xFFFFFFFFFFFFFFFF):
                        min_e = enc
                    if (enc & 0xFFFFFFFFFFFFFFFF) > (max_e & 0xFFFFFFFFFFFFFFFF):
                        max_e = enc
            if bad or non_exc < 2:
                continue
            bits = _u64_span_i64(min_e, max_e)
            cost = n * bits + (n - non_exc) * _D64_EXC_COST
            if cost < best_cost:
                best_cost = cost
                best_pair = (fac, exp)
    return best_pair


def _pack_i32(vals: List[int]) -> bytes:
    return struct.pack("<%di" % len(vals), *vals)


def _pack_i64(vals: List[int]) -> bytes:
    return struct.pack("<%dq" % len(vals), *vals)


def _pack_le_u32(vals: List[int]) -> bytes:
    """Little-endian uint32 payload (e.g. raw float bits)."""
    return struct.pack("<%dI" % len(vals), *[int(v) & 0xFFFFFFFF for v in vals])


def _unpack_i32(blob: bytes, n: int) -> List[int]:
    return list(struct.unpack("<%di" % n, blob))


def _alp_raw_float64_fallback(window: List[float]) -> tuple[int, int, int] | None:
    bits = [_double_bits(float(v)) for v in window]
    raw = struct.pack("<%dQ" % len(bits), *bits)
    t0 = time.perf_counter_ns()
    comp = _zstd3_compress(raw)
    t1 = time.perf_counter_ns()
    try:
        dec = _zstd3_decompress(comp, len(raw))
    except Exception:
        return None
    t2 = time.perf_counter_ns()
    if dec != raw:
        return None
    return len(comp) + 8, t1 - t0, t2 - t1


_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def min_decimal_scale_exact_float(v: float) -> int:
    if v != v or v in (float("inf"), float("-inf")):
        return -1
    if float(v) == 0.0:
        return 0
    bits = _double_bits(v)
    for s in range(_LECO_MAX_SCALE + 1):
        pow10 = 10.0**s
        m = int(round(float(v) * pow10))
        if m < _I64_MIN or m > _I64_MAX:
            return -1
        r = float(m) / pow10
        if _double_bits(r) == bits:
            return s
    return -1


def scale_float_column_to_ints(col: List[float]) -> Optional[List[int]]:
    """Lossless decimal scale per column (same rules as ``bench_leco_float_columns``)."""
    if not col:
        return None
    mx = 0
    for v in col:
        s = min_decimal_scale_exact_float(float(v))
        if s < 0:
            return None
        mx = max(mx, s)
    pow10 = 10.0**mx
    data: List[int] = []
    for v in col:
        fv = float(v)
        if fv == 0.0:
            data.append(_double_bits_i64(fv))
            continue
        m = int(round(fv * pow10))
        if m < _I64_MIN or m > _I64_MAX:
            return None
        r = float(m) / pow10
        if _double_bits(r) != _double_bits(fv):
            return None
        data.append(m)
    return data


def scale_float_column_to_ints_with_max_point(
    col: List[float], max_point: int
) -> Optional[List[int]]:
    """Scale using catalog max_point_number with downward fallback; no bit-exact guard."""
    if not col:
        return None
    start_scale = max(0, int(max_point))
    for col_scale in range(start_scale, -1, -1):
        pow10 = 10.0**col_scale
        data: List[int] = []
        ok = True
        for v in col:
            fv = float(v)
            m = int(round(fv * pow10))
            if m < _I64_MIN or m > _I64_MAX:
                ok = False
                break
            data.append(m)
        if ok:
            return data
    return None


def scale_float_columns_to_int(
    columns: List[List[float]],
    max_points_per_column: Optional[List[int]] = None,
) -> List[List[int]]:
    out: List[List[int]] = []
    for i, col in enumerate(columns):
        if max_points_per_column is not None and i < len(max_points_per_column):
            scaled = scale_float_column_to_ints_with_max_point(
                col, max_points_per_column[i]
            )
            if scaled is None:
                scaled = scale_float_column_to_ints(col)
        else:
            scaled = scale_float_column_to_ints(col)
        if scaled is not None:
            out.append(scaled)
    return out


def bench_leco_float_columns(
    columns: List[List[float]], max_points_per_column: Optional[List[int]] = None
) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_long_list_loss
    from .int64_column_codec import fast_pfor_decode_int64, fast_pfor_encode_int64

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for i, col in enumerate(columns):
        if not col:
            continue
        if max_points_per_column is not None and i < len(max_points_per_column):
            data = scale_float_column_to_ints_with_max_point(col, max_points_per_column[i])
            if data is None:
                data = scale_float_column_to_ints(col)
        else:
            data = scale_float_column_to_ints(col)
        if data is None:
            continue
        t0 = time.perf_counter_ns()
        try:
            payload = fast_pfor_encode_int64(data)
        except Exception:
            continue
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        try:
            recovered = fast_pfor_decode_int64(payload, len(data))
        except Exception:
            continue
        t3 = time.perf_counter_ns()
        if recovered != data:
            continue
        accumulate_long_list_loss(data, recovered, loss_acc)
        tot_o += column_original_size(len(col), 'float')
        tot_c += len(payload) + 4
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_alp_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_double_list_loss

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        tot_o += column_original_size(len(col), 'float')
        col_ok = True
        cc = ct_enc = ct_dec = 0
        off = 0
        n = len(col)
        while off < n and col_ok:
            vec_len = min(1024, n - off)
            window = [float(col[off + i]) for i in range(vec_len)]
            if _alp_force_raw_fallback():
                fallback = _alp_raw_float64_fallback(window)
                if fallback is None:
                    col_ok = False
                    break
                fc, fte, ftd = fallback
                accumulate_double_list_loss(window, window, loss_acc)
                cc += fc
                ct_enc += fte
                ct_dec += ftd
                off += vec_len
                continue
            fe = _alp_best_fe(window, vec_len)
            if fe is None:
                fallback = _alp_raw_float64_fallback(window)
                if fallback is None:
                    col_ok = False
                    break
                fc, fte, ftd = fallback
                accumulate_double_list_loss(window, window, loss_acc)
                cc += fc
                ct_enc += fte
                ct_dec += ftd
                off += vec_len
                continue
            fac, exp = fe
            encoded = [0] * vec_len
            exc_pos: List[int] = []
            exc_val: List[float] = []
            ph = 0
            found_ph = False
            for i in range(vec_len):
                v = window[i]
                enc = _alp_enc_val(v, fac, exp)
                dec = _alp_dec_val(enc, fac, exp)
                if _float_bits(dec) == _float_bits(v):
                    encoded[i] = enc
                    if not found_ph:
                        ph = enc
                        found_ph = True
                else:
                    exc_pos.append(i)
                    exc_val.append(v)
            if not found_ph:
                fallback = _alp_raw_float64_fallback(window)
                if fallback is None:
                    col_ok = False
                    break
                fc, fte, ftd = fallback
                accumulate_double_list_loss(window, window, loss_acc)
                cc += fc
                ct_enc += fte
                ct_dec += ftd
                off += vec_len
                continue
            for p in exc_pos:
                encoded[p] = ph
            raw = _pack_i32(encoded)
            t0 = time.perf_counter_ns()
            comp = _zstd3_compress(raw)
            t1 = time.perf_counter_ns()
            try:
                dec_b = _zstd3_decompress(comp, len(raw))
            except Exception:
                col_ok = False
                break
            t2 = time.perf_counter_ns()
            rec_enc = _unpack_i32(dec_b, vec_len)
            if len(rec_enc) != vec_len:
                col_ok = False
                break
            round_trip = [_alp_dec_val(rec_enc[i], fac, exp) for i in range(vec_len)]
            for j, p in enumerate(exc_pos):
                round_trip[p] = exc_val[j]
            for i in range(vec_len):
                if _float_bits(round_trip[i]) != _float_bits(window[i]):
                    col_ok = False
                    break
            if not col_ok:
                break
            accumulate_double_list_loss(window, round_trip, loss_acc)
            exc_count = len(exc_pos)
            cc += len(comp) + 6 + exc_count * (2 + 4)
            ct_enc += t1 - t0
            ct_dec += t2 - t1
            off += vec_len
        if col_ok:
            tot_c += cc
            te += ct_enc
            td += ct_dec
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_alp_double_columns(columns: List[List[float]]) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_double_list_loss

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    from .int64_column_codec import fast_pfor_decode_int64, fast_pfor_encode_int64

    tot_o = tot_c = te = td = 0
    for col in columns:
        if not col:
            continue
        col_ok = True
        cc = ct_enc = ct_dec = 0
        off = 0
        n = len(col)
        while off < n and col_ok:
            vec_len = min(1024, n - off)
            window = [float(col[off + i]) for i in range(vec_len)]
            if _alp_force_raw_fallback():
                fallback = _alp_raw_float64_fallback(window)
                if fallback is None:
                    col_ok = False
                    break
                fc, fte, ftd = fallback
                accumulate_double_list_loss(window, window, loss_acc)
                cc += fc
                ct_enc += fte
                ct_dec += ftd
                off += vec_len
                continue
            fe = _alp_d64_best_fe(window, vec_len)
            if fe is None:
                fallback = _alp_raw_float64_fallback(window)
                if fallback is None:
                    col_ok = False
                    break
                fc, fte, ftd = fallback
                accumulate_double_list_loss(window, window, loss_acc)
                cc += fc
                ct_enc += fte
                ct_dec += ftd
                off += vec_len
                continue
            fac, exp = fe
            encoded = [0] * vec_len
            exc_pos: List[int] = []
            exc_val: List[float] = []
            placeholder = 0
            found_placeholder = False
            for i in range(vec_len):
                v = window[i]
                enc = _alp_d64_enc_val(v, fac, exp)
                dec = _alp_d64_dec_val(enc, fac, exp)
                if _double_bits(dec) == _double_bits(v):
                    encoded[i] = enc
                    if not found_placeholder:
                        placeholder = enc
                        found_placeholder = True
                else:
                    exc_pos.append(i)
                    exc_val.append(v)
            if not found_placeholder:
                fallback = _alp_raw_float64_fallback(window)
                if fallback is None:
                    col_ok = False
                    break
                fc, fte, ftd = fallback
                accumulate_double_list_loss(window, window, loss_acc)
                cc += fc
                ct_enc += fte
                ct_dec += ftd
                off += vec_len
                continue
            for p in exc_pos:
                encoded[p] = placeholder
            try:
                t0 = time.perf_counter_ns()
                payload = fast_pfor_encode_int64(encoded)
                t1 = time.perf_counter_ns()
                t2 = time.perf_counter_ns()
                recovered_enc = fast_pfor_decode_int64(payload, vec_len)
                t3 = time.perf_counter_ns()
            except Exception:
                col_ok = False
                break
            if len(recovered_enc) != vec_len:
                col_ok = False
                break
            round_trip = [_alp_d64_dec_val(recovered_enc[i], fac, exp) for i in range(vec_len)]
            for j, p in enumerate(exc_pos):
                round_trip[p] = exc_val[j]
            for i in range(vec_len):
                if _double_bits(round_trip[i]) != _double_bits(window[i]):
                    col_ok = False
                    break
            if not col_ok:
                break
            accumulate_double_list_loss(window, round_trip, loss_acc)
            exc_count = len(exc_pos)
            cc += len(payload) + 6 + exc_count * (2 + 8)
            ct_enc += t1 - t0
            ct_dec += t3 - t2
            off += vec_len
        if col_ok:
            tot_o += column_original_size(n, "float")
            tot_c += cc
            te += ct_enc
            td += ct_dec
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_alp_int64_columns(columns: List[List[int]]) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_long_list_loss

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    from .int64_column_codec import fast_pfor_decode_int64, fast_pfor_encode_int64

    tot_o = tot_c = te = td = 0
    for col in columns:
        if not col:
            continue
        data = [int(v) for v in col]
        t0 = time.perf_counter_ns()
        try:
            payload = fast_pfor_encode_int64(data)
        except Exception:
            continue
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        try:
            recovered = fast_pfor_decode_int64(payload, len(data))
        except Exception:
            continue
        t3 = time.perf_counter_ns()
        if recovered != data:
            continue
        accumulate_long_list_loss(data, recovered, loss_acc)
        tot_o += column_original_size(len(col), "int")
        tot_c += len(payload)
        te += t1 - t0
        td += t3 - t2
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_fastpfor_int_columns(columns: List[List[int]]) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_byte_loss

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        data = [int(x) for x in col]
        raw = _pack_i64(data)
        t0 = time.perf_counter_ns()
        comp = _zstd3_compress(raw)
        t1 = time.perf_counter_ns()
        try:
            dec = _zstd3_decompress(comp, len(raw))
        except Exception:
            continue
        t2 = time.perf_counter_ns()
        if dec != raw:
            continue
        accumulate_byte_loss(raw, dec, loss_acc)
        tot_o += len(data) * 8
        tot_c += len(comp)
        te += t1 - t0
        td += t2 - t1
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_fastpfor_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_byte_loss

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        if not col:
            continue
        bits = [_double_bits(float(v)) for v in col]
        raw = struct.pack("<%dQ" % len(bits), *bits)
        t0 = time.perf_counter_ns()
        comp = _zstd3_compress(raw)
        t1 = time.perf_counter_ns()
        try:
            dec = _zstd3_decompress(comp, len(raw))
        except Exception:
            continue
        t2 = time.perf_counter_ns()
        if dec != raw:
            continue
        accumulate_byte_loss(raw, dec, loss_acc)
        tot_o += len(col) * 8
        tot_c += len(comp)
        te += t1 - t0
        td += t2 - t1
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]
