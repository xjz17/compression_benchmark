"""Typed process bridge for the C++ URL1/URB1 and numeric codec adapters.

Python performs data marshaling only. Timings around these functions include
process creation and IPC; they are not native kernel latency measurements.
"""
from __future__ import annotations

import array
import sys
from typing import Sequence

from .cpp_action_bridge import run

CLI = "unified_rle_numeric_cli"
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
TIMING_SCOPE = "python_cli_wall_including_process_launch_ipc"
_FORMATS = {"i32": "i", "i64": "q", "f32": "f", "f64": "d"}
_WIDTHS = {"i32": 4, "i64": 8, "f32": 4, "f64": 8}


def pack_values(values: Sequence[int] | Sequence[float], kind: str) -> bytes:
    if kind not in _FORMATS:
        raise ValueError(f"unsupported numeric kind: {kind}")
    packed = array.array(_FORMATS[kind], values)
    if packed.itemsize != _WIDTHS[kind]:
        raise RuntimeError("platform array width differs from codec contract")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def unpack_values(raw: bytes, kind: str) -> list[int] | list[float]:
    if kind not in _FORMATS or len(raw) % _WIDTHS[kind]:
        raise ValueError("numeric decode output width mismatch")
    values = array.array(_FORMATS[kind])
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def encode(raw: bytes, codec: str = "rle", kind: str = "i64") -> bytes:
    return run(CLI, "encode", raw, codec, kind)


def decode(frame: bytes, codec: str = "rle", kind: str = "i64", *,
           max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> bytes:
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 0:
        raise ValueError("max_output_bytes must be a nonnegative integer")
    return run(CLI, "decode", frame, codec, kind, max_output_bytes)


def encode_values(values: Sequence[int] | Sequence[float], codec: str, kind: str) -> bytes:
    return encode(pack_values(values, kind), codec, kind)


def encode_scaled_values(values: Sequence[float], kind: str, max_point_number: int) -> bytes:
    if isinstance(max_point_number, bool) or not isinstance(max_point_number, int) or not 0 <= max_point_number <= 18:
        raise ValueError("RLE max_point must be an integer in [0,18]")
    if kind not in ("f32", "f64"):
        raise ValueError("scaled RLE requires floating input")
    return run(CLI, "encode", pack_values(values, kind), "rle_scaled", kind, max_point_number)


def decode_values(frame: bytes, codec: str, kind: str, *,
                  max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> list[int] | list[float]:
    return unpack_values(decode(frame, codec, kind, max_output_bytes=max_output_bytes), kind)
