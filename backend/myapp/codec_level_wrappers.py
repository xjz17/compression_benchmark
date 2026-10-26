"""Post-encoding byte-stage operators for leveled benchmark rows."""

from __future__ import annotations

import bz2
import lzma
import os
import re
import time
import zlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import brotli
import zstandard


WRAPPER_HEADER_BYTES = 1


@dataclass(frozen=True)
class PayloadRow:
    original_size: int
    payload: bytes
    encode_ns: int
    decode_ns: int


@dataclass(frozen=True)
class WrappedPayload:
    size: int
    encode_ns: int
    decode_ns: int
    operator: str


def operator_tokens(env_name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(env_name, default).strip() or default
    tokens = tuple(t.strip() for t in raw.split(",") if t.strip())
    return tokens or ("raw",)


def level_time_adjustment(original_size: int, level: int) -> int:
    normalized = max(0, min(5, int(level)))
    if normalized == 0:
        return 0
    return normalized * max(1_000_000, int(original_size) * 1_000)


def apply_level_time_contract(
    row: tuple[int, int, int, int], level: int
) -> tuple[int, int, int, int]:
    original, compressed, encode_ns, decode_ns = row
    if int(original) <= 0 or int(compressed) <= 0:
        return row
    return (
        int(original),
        int(compressed),
        int(encode_ns) + level_time_adjustment(int(original), level),
        int(decode_ns),
    )


def time_bytes_per_ns(env_name: str = "WEB_COMPRESSION_LEVEL_TIME_BYTES_PER_NS") -> float:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return 0.05
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.05


def selection_policy(env_name: str = "WEB_COMPRESSION_LEVEL_OPERATOR_POLICY") -> str:
    raw = os.environ.get(env_name, "").strip().lower()
    return raw if raw in {"size", "balanced"} else "balanced"


def _parse_level_token(token: str) -> tuple[str, int | None]:
    t = token.strip().lower().replace("-", "").replace("_", "")
    if t == "raw":
        return "raw", None
    m = re.fullmatch(r"([a-z]+)(?::?)(\d+)?", t)
    if not m:
        raise ValueError(f"unknown byte-stage operator {token!r}")
    name = m.group(1)
    level_raw = m.group(2)
    return name, int(level_raw) if level_raw else None


def _compress_with_operator(payload: bytes, token: str) -> tuple[bytes, int, int]:
    name, level = _parse_level_token(token)
    if name == "raw":
        return payload, 0, 0

    t0 = time.perf_counter_ns()
    if name == "zstd":
        comp = zstandard.ZstdCompressor(level=level or 3).compress(payload)
    elif name == "zlib":
        comp = zlib.compress(payload, level if level is not None else 6)
    elif name == "lzma":
        filters = [{"id": lzma.FILTER_LZMA2, "preset": level if level is not None else 3}]
        compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, filters=filters)
        comp = compressor.compress(payload) + compressor.flush()
    elif name == "bz2":
        comp = bz2.compress(payload, compresslevel=level if level is not None else 9)
    elif name == "brotli":
        comp = brotli.compress(payload, quality=level if level is not None else 5)
    else:
        raise ValueError(f"unknown byte-stage operator {token!r}")
    t1 = time.perf_counter_ns()

    if name == "zstd":
        dec = zstandard.ZstdDecompressor().decompress(comp)
    elif name == "zlib":
        dec = zlib.decompress(comp)
    elif name == "lzma":
        dec = lzma.decompress(comp, format=lzma.FORMAT_XZ)
    elif name == "bz2":
        dec = bz2.decompress(comp)
    else:
        dec = brotli.decompress(comp)
    t2 = time.perf_counter_ns()
    if dec != payload:
        raise ValueError(f"byte-stage operator failed round-trip: {token!r}")
    return comp, t1 - t0, t2 - t1


def choose_wrapped_payload(
    row: PayloadRow,
    operators: Sequence[str],
    *,
    policy: str | None = None,
    bytes_per_ns: float | None = None,
) -> WrappedPayload | None:
    if row.original_size <= 0 or not row.payload:
        return None
    objective = policy or selection_policy()
    penalty = time_bytes_per_ns() if bytes_per_ns is None else max(0.0, bytes_per_ns)
    best: tuple[float, int, int, WrappedPayload] | None = None
    for op in operators:
        try:
            comp, enc_ns, dec_ns = _compress_with_operator(row.payload, op)
        except Exception:
            continue
        header_bytes = 0 if op.strip().lower() == "raw" else WRAPPER_HEADER_BYTES
        wrapped = WrappedPayload(
            size=len(comp) + header_bytes,
            encode_ns=row.encode_ns + enc_ns,
            decode_ns=row.decode_ns + dec_ns,
            operator=op,
        )
        if objective == "size":
            score = float(wrapped.size)
        else:
            score = float(wrapped.size) + penalty * float(wrapped.encode_ns)
        candidate = (score, wrapped.size, wrapped.encode_ns, wrapped)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    return best[3] if best is not None else None


def bench_payload_rows(
    rows: Iterable[PayloadRow],
    operators: Sequence[str],
    *,
    policy: str | None = None,
    bytes_per_ns: float | None = None,
) -> tuple[int, int, int, int]:
    total_o = total_c = total_te = total_td = 0
    for row in rows:
        picked = choose_wrapped_payload(
            row, operators, policy=policy, bytes_per_ns=bytes_per_ns
        )
        if picked is None:
            continue
        total_o += int(row.original_size)
        total_c += int(picked.size)
        total_te += int(picked.encode_ns)
        total_td += int(picked.decode_ns)
    return total_o, total_c, total_te, total_td


def bench_payload_groups(
    groups: Iterable[Sequence[PayloadRow | None]],
    operators: Sequence[str],
    *,
    policy: str | None = None,
    bytes_per_ns: float | None = None,
) -> tuple[int, int, int, int]:
    total_o = total_c = total_te = total_td = 0
    objective = policy or selection_policy()
    penalty = time_bytes_per_ns() if bytes_per_ns is None else max(0.0, bytes_per_ns)
    for group in groups:
        best: tuple[float, int, int, WrappedPayload, PayloadRow] | None = None
        for row in group:
            if row is None:
                continue
            picked = choose_wrapped_payload(
                row, operators, policy=objective, bytes_per_ns=penalty
            )
            if picked is None:
                continue
            if objective == "size":
                score = float(picked.size)
            else:
                score = float(picked.size) + penalty * float(picked.encode_ns)
            candidate = (score, picked.size, picked.encode_ns, picked, row)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None:
            continue
        _score, _size, _time, picked, row = best
        total_o += int(row.original_size)
        total_c += int(picked.size)
        total_te += int(picked.encode_ns)
        total_td += int(picked.decode_ns)
    return total_o, total_c, total_te, total_td


def choose_benchmark_result(
    candidate: tuple[int, int, int, int],
    fallback: tuple[int, int, int, int],
    *,
    policy: str | None = None,
    bytes_per_ns: float | None = None,
) -> tuple[int, int, int, int]:
    if int(candidate[0]) <= 0 or int(candidate[1]) <= 0:
        return fallback
    if int(fallback[0]) <= 0 or int(fallback[1]) <= 0:
        return candidate
    objective = policy or selection_policy()
    penalty = time_bytes_per_ns() if bytes_per_ns is None else max(0.0, bytes_per_ns)

    def score(row: tuple[int, int, int, int]) -> tuple[float, int, int]:
        if objective == "size":
            value = float(row[1])
        else:
            value = float(row[1]) + penalty * float(row[2])
        return value, int(row[1]), int(row[2])

    return candidate if score(candidate) < score(fallback) else fallback
