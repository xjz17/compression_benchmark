"""Round-trip loss metrics for benchmark CSV reports (``isLossy``, ``lossRate``)."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

LOSS_RATE_EPS = 1e-15

_KNOWN_LOSSY_PREFIXES = (
    "sz_",
    "sz3_",
    "serf_",
    "simpiece_",
    "ext_zfp_tol_",
    "machete_rel_",
    "machete_float_rel_",
    "most_minimal",
    "most_float_minimal",
)


def is_known_lossy_algorithm(algorithm: str) -> bool:
    key = str(algorithm or "").strip().lower()
    if not key:
        return False
    return any(key.startswith(prefix) for prefix in _KNOWN_LOSSY_PREFIXES)


def accumulate_byte_loss(original: bytes | None, decoded: bytes | None, acc: List[float]) -> None:
    if len(acc) < 2 or original is None:
        return
    ref = float(len(original))
    err = 0.0
    if decoded is None:
        err = ref
    else:
        n = min(len(original), len(decoded))
        for i in range(n):
            if original[i] != decoded[i]:
                err += 1.0
        err += float(abs(len(original) - len(decoded)))
    acc[0] += err
    acc[1] += ref


def accumulate_long_list_loss(
    original: Sequence[int] | None,
    decoded: Sequence[int] | None,
    acc: List[float],
) -> None:
    if len(acc) < 2 or original is None or decoded is None:
        return
    n = min(len(original), len(decoded))
    err = 0.0
    ref = 0.0
    for i in range(n):
        o = float(original[i])
        d = float(decoded[i])
        err += abs(o - d)
        ref += abs(o)
    if len(original) != len(decoded):
        err += float(abs(len(original) - len(decoded)))
        ref += float(len(original))
    acc[0] += err
    acc[1] += ref


def accumulate_double_list_loss(
    original: Sequence[float] | None,
    decoded: Sequence[float] | None,
    acc: List[float],
) -> None:
    if len(acc) < 2 or original is None or decoded is None:
        return
    n = min(len(original), len(decoded))
    err = 0.0
    ref = 0.0
    for i in range(n):
        o = float(original[i])
        d = float(decoded[i])
        if math.isfinite(o) and math.isfinite(d):
            err += abs(o - d)
            ref += abs(o)
    if len(original) != len(decoded):
        err += float(abs(len(original) - len(decoded)))
        ref += float(len(original))
    acc[0] += err
    acc[1] += ref


def accumulate_long_array_loss(
    original: Sequence[int],
    decoded: Sequence[int],
    count: int,
    acc: List[float],
) -> None:
    if len(acc) < 2 or count <= 0:
        return
    n = min(count, len(original), len(decoded))
    err = 0.0
    ref = 0.0
    for i in range(n):
        o = float(original[i])
        d = float(decoded[i])
        err += abs(o - d)
        ref += abs(o)
    if count != n:
        err += float(abs(count - n))
        ref += float(count)
    acc[0] += err
    acc[1] += ref


def accumulate_double_array_loss(
    original: Sequence[float],
    decoded: Sequence[float],
    count: int,
    acc: List[float],
) -> None:
    if len(acc) < 2 or count <= 0:
        return
    n = min(count, len(original), len(decoded))
    err = 0.0
    ref = 0.0
    for i in range(n):
        o = float(original[i])
        d = float(decoded[i])
        if math.isfinite(o) and math.isfinite(d):
            err += abs(o - d)
            ref += abs(o)
    if count != n:
        err += float(abs(count - n))
        ref += float(count)
    acc[0] += err
    acc[1] += ref


def loss_rate(sum_abs_error: float, sum_abs_ref: float) -> float:
    if sum_abs_ref <= 0.0:
        return 1.0 if sum_abs_error > 0.0 else 0.0
    return sum_abs_error / sum_abs_ref


def is_lossy_flag(algorithm: str, sum_abs_error: float, sum_abs_ref: float) -> int:
    if is_known_lossy_algorithm(algorithm):
        return 1
    return 1 if loss_rate(sum_abs_error, sum_abs_ref) > LOSS_RATE_EPS else 0


def format_loss_rate_csv(rate: float) -> str:
    """Fixed decimal form for CSV (never scientific notation)."""
    if rate == 0.0 or not math.isfinite(rate):
        return "0"
    text = f"{rate:.15f}".rstrip("0").rstrip(".")
    return text or "0"
