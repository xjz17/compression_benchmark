#!/usr/bin/env python3
"""Verify Python buff_kernel encode ↔ decode component parity (and optional byte length)."""

from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND = os.path.join(_REPO, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from myapp.buff_kernel import (  # noqa: E402
    BuffPrecisionBound,
    decode_doubles,
    encode_doubles,
    get_precision_bound,
)


def _sign_extend_u32(v: int, bits: int) -> int:
    if bits <= 0:
        return 0
    v &= (1 << bits) - 1
    sign = 1 << (bits - 1)
    if v & sign:
        v -= 1 << bits
    return v


def _truncated_int_for_stream(ip: int, ilen: int) -> int:
    """Low ``ilen`` bits of ``ip`` as ``u32``, then sign-extended (matches encode write)."""
    raw = (ip & 0xFFFFFFFF) & ((1 << ilen) - 1)
    return _sign_extend_u32(raw, ilen)


def _expected_components(values: list[float], scale: int):
    import math

    prec = int(math.floor(math.log10(scale)))
    prec_delta = get_precision_bound(prec)
    bound = BuffPrecisionBound(prec_delta)
    bd_vec = []
    for val in values:
        v = bound.precision_bound(val)
        bd_vec.append(v)
        bound.cal_length(v)
    int_parts = []
    dec_parts = []
    for bd in bd_vec:
        ip, dp = bound.fetch_components(bd)
        int_parts.append(ip)
        dec_parts.append(dp)
    return bound.int_length, bound.decimal_length, int_parts, dec_parts


def main() -> None:
    values = [1.125, -0.25, 3.14159265]
    scale = 10000
    packed = encode_doubles(values, scale)
    dec = decode_doubles(packed)
    ilen_e, dlen_e, int_e, dec_e = _expected_components(values, scale)
    assert dec.t == len(values)
    assert dec.ilen == ilen_e
    assert dec.dlen == dlen_e
    int_trunc = [_truncated_int_for_stream(ip, ilen_e) for ip in int_e]
    assert dec.int_parts == int_trunc
    assert dec.dec_parts == dec_e
    print("buff_kernel roundtrip OK:", len(packed), "bytes,", dec.t, "points")


if __name__ == "__main__":
    main()
