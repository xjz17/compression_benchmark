"""Port of ``org.urbcomp.startdb.compress.elf.utils.Elf32Utils`` (SIGMOD Elf / Elf+ float)."""
from __future__ import annotations

import math
import struct
from typing import Tuple

_F: Tuple[int, ...] = (
    0,
    4,
    7,
    10,
    14,
    17,
    20,
    24,
    27,
    30,
    34,
    37,
    40,
    44,
    47,
    50,
    54,
    57,
    60,
    64,
    67,
)


def _f32(x: float) -> float:
    """Round to IEEE float32 like Java ``float`` (used in ``getSignificantCount`` / ``map10iP``)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _f32_table(vals: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(_f32(v) for v in vals)


# Match ``map10iP`` / ``map10iN`` ``float`` literals in Elf32Utils.java
_MAP10IP: Tuple[float, ...] = _f32_table(
    (
        1.0,
        1.0e1,
        1.0e2,
        1.0e3,
        1.0e4,
        1.0e5,
        1.0e6,
        1.0e7,
        1.0e8,
        1.0e9,
        1.0e10,
        1.0e11,
        1.0e12,
        1.0e13,
        1.0e14,
        1.0e15,
        1.0e16,
        1.0e17,
        1.0e18,
        1.0e19,
        1.0e20,
    )
)

_MAP10IN: Tuple[float, ...] = _f32_table(
    (
        1.0,
        1.0e-1,
        1.0e-2,
        1.0e-3,
        1.0e-4,
        1.0e-5,
        1.0e-6,
        1.0e-7,
        1.0e-8,
        1.0e-9,
        1.0e-10,
        1.0e-11,
        1.0e-12,
        1.0e-13,
        1.0e-14,
        1.0e-15,
        1.0e-16,
        1.0e-17,
        1.0e-18,
        1.0e-19,
        1.0e-20,
    )
)

_MAP_SP_GT1: Tuple[int, ...] = (1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000)

_MAP_SP_LT1: Tuple[float, ...] = _f32_table(
    (1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001)
)

LOG_2_10 = math.log(10) / math.log(2)


def get_f_alpha(alpha: int) -> int:
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if alpha >= len(_F):
        return int(math.ceil(alpha * LOG_2_10))
    return _F[alpha]


def get10i_p(i: int) -> float:
    if i < 0:
        raise ValueError("i must be >= 0")
    if i >= len(_MAP10IP):
        return _f32(float(f"1.0E{i}"))
    return _MAP10IP[i]


def get10i_n(i: int) -> float:
    if i < 0:
        raise ValueError("i must be >= 0")
    if i >= len(_MAP10IN):
        return _f32(float(f"1.0E-{i}"))
    return _MAP10IN[i]


def get_sp(v: float) -> int:
    if v >= 1:
        i = 0
        while i < len(_MAP_SP_GT1) - 1:
            if v < _MAP_SP_GT1[i + 1]:
                return i
            i += 1
    else:
        i = 1
        while i < len(_MAP_SP_LT1):
            if v >= _MAP_SP_LT1[i]:
                return -i
            i += 1
    return int(math.floor(math.log10(v)))


def _get_sp_and_10in_flag(v: float) -> Tuple[int, int]:
    """Java widens ``float`` to ``double`` for ``getSPAnd10iNFlag(double v)``."""
    sp_and = [0, 0]
    if v >= 1:
        i = 0
        while i < len(_MAP_SP_GT1) - 1:
            if v < float(_MAP_SP_GT1[i + 1]):
                sp_and[0] = i
                return sp_and[0], sp_and[1]
            i += 1
    else:
        i = 1
        while i < len(_MAP_SP_LT1):
            boundary = float(_MAP_SP_LT1[i])
            if v >= boundary:
                sp_and[0] = -i
                sp_and[1] = 1 if v == boundary else 0
                return sp_and[0], sp_and[1]
            i += 1
    log10v = math.log10(v)
    sp_and[0] = int(math.floor(log10v))
    sp_and[1] = 1 if log10v == int(log10v) else 0
    return sp_and[0], sp_and[1]


def _get_significant_count(v: float, sp: int, last_beta_star: int) -> int:
    if last_beta_star != 2147483647 and last_beta_star != 0:
        i = max(last_beta_star - sp - 1, 1)
    elif last_beta_star == 2147483647:
        i = 8 - sp - 1
    elif sp >= 0:
        i = 1
    else:
        i = -sp

    temp = _f32(v * get10i_p(i))
    temp_int = int(temp)
    while temp_int != temp:
        i += 1
        temp = _f32(v * get10i_p(i))
        temp_int = int(temp)

    if _f32(temp / get10i_p(i)) != v:
        return 8
    while i > 0 and temp_int % 10 == 0:
        i -= 1
        temp_int //= 10
    return sp + i + 1


def get_alpha_and_beta_star(v: float, last_beta_star: int) -> Tuple[int, int]:
    if v < 0:
        v = -v
    vf = _f32(v)
    sp, flag = _get_sp_and_10in_flag(float(vf))
    beta = _get_significant_count(vf, sp, last_beta_star)
    alpha = beta - sp - 1
    beta_star = 0 if flag == 1 else beta
    return alpha, beta_star


def round_up(v: float, alpha: int) -> float:
    scale = get10i_p(alpha)
    vv = _f32(v)
    if v < 0:
        r = _f32(math.floor(vv * scale) / scale)
    else:
        r = _f32(math.ceil(vv * scale) / scale)
    return float(r)
