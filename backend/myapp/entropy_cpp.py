"""C++ entropy-codec bindings; Python contains no entropy coding kernel.

Public encode/decode and six-item bench APIs are preserved. Bench durations
are wall-clock nanoseconds INCLUDING process launch and IPC, not native kernel
latency. The exact historical Huffman/range/rANS wires are emitted in C++.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from .cpp_action_bridge import run

logger = logging.getLogger(__name__)
TIMING_SCOPE = "python_cli_wall_including_process_launch_ipc"
_CLI = "general_codec_reference_stages_cli"


def huffman_encode(data: bytes) -> bytes:
    return run(_CLI, "huffman-forward", data)


def huffman_decode(comp: bytes) -> bytes:
    return run(_CLI, "huffman-inverse", comp)


def arithmetic_encode(data: bytes) -> bytes:
    return run(_CLI, "range-forward", data)


def arithmetic_decode(comp: bytes) -> bytes:
    return run(_CLI, "range-inverse", comp)


def ans_encode(data: bytes) -> bytes:
    return run(_CLI, "rans-forward", data)


def ans_decode(comp: bytes) -> bytes:
    return run(_CLI, "rans-inverse", comp)


def _bench(
    columns: list[bytes], encode: Callable[[bytes], bytes],
    decode: Callable[[bytes], bytes], label: str, *, skip_failures: bool = False,
) -> tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_byte_loss

    original = compressed = encode_ns = decode_ns = 0
    loss_acc = [0.0, 0.0]
    for index, raw in enumerate(columns):
        try:
            t0 = time.perf_counter_ns()
            wire = encode(raw)
            t1 = time.perf_counter_ns()
            restored = decode(wire)
            t2 = time.perf_counter_ns()
            if restored != raw:
                raise RuntimeError(f"{label} roundtrip failed")
        except Exception as error:
            if not skip_failures:
                raise
            logger.warning("%s: skip column %d (%d bytes): %s", label, index, len(raw), error)
            continue
        accumulate_byte_loss(raw, restored, loss_acc)
        original += len(raw)
        compressed += len(wire)
        encode_ns += t1 - t0
        decode_ns += t2 - t1
    return original, compressed, encode_ns, decode_ns, loss_acc[0], loss_acc[1]


def bench_huffman_bytes(column_bytes: list[bytes]) -> tuple[int, int, int, int, float, float]:
    return _bench(column_bytes, huffman_encode, huffman_decode, "huffman")


def bench_arithmetic_bytes(column_bytes: list[bytes]) -> tuple[int, int, int, int, float, float]:
    return _bench(column_bytes, arithmetic_encode, arithmetic_decode, "arithmetic")


def bench_ans_bytes(column_bytes: list[bytes]) -> tuple[int, int, int, int, float, float]:
    # Preserve the former ANS benchmark's per-column failure handling.
    return _bench(column_bytes, ans_encode, ans_decode, "ANS", skip_failures=True)
