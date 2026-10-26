"""Opt-in in-process C++ layout bridge, not the full graph timing backend.

The active graph runtime is deliberately unchanged. This shared-library bridge
preserves its existing OLS1 wire representation and has no subprocess fallback.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "cpp/build-native-layout-v1" / (
    "libnative_layout_bridge_v1.dylib" if sys.platform == "darwin"
    else "libnative_layout_bridge_v1.so")


class NativeLayoutBridge:
    def __init__(self, library: str | Path | None = None):
        self.path = Path(library or os.environ.get("NATIVE_LAYOUT_BRIDGE_V1", LIBRARY)).resolve()
        if not self.path.is_file():
            raise RuntimeError(f"missing native layout shared library: {self.path}")
        self._library = ctypes.CDLL(str(self.path))
        pointer = ctypes.c_void_p
        parameters = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), pointer,
                      ctypes.c_size_t]
        self._library.wc_layout_abi_version_v1.restype = ctypes.c_uint32
        self._library.wc_layout_abi_version_v1.argtypes = []
        if self._library.wc_layout_abi_version_v1() != 1:
            raise RuntimeError("native layout ABI version mismatch")
        self._library.wc_layout_run_v1.argtypes = parameters
        self._library.wc_layout_run_v1.restype = pointer
        self._library.wc_layout_benchmark_v1.argtypes = parameters + [ctypes.c_uint32] * 2
        self._library.wc_layout_benchmark_v1.restype = pointer
        for name, result in (("data", pointer), ("size", ctypes.c_size_t),
                             ("error", ctypes.c_char_p), ("free", None)):
            function = getattr(self._library, f"wc_layout_result_{name}_v1")
            function.argtypes = [pointer]
            function.restype = result

    def _call(self, name: str, args: tuple, data: bytes, *extra) -> bytes:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("layout input must be bytes-like")
        strings = tuple(str(value).encode("ascii") for value in args)
        if any(b"\0" in value for value in strings):
            raise ValueError("layout argument contains a NUL")
        argv = (ctypes.c_char_p * len(strings))(*strings)
        raw = bytes(data)
        buffer = ctypes.create_string_buffer(raw)
        result = getattr(self._library, name)(len(strings), argv, buffer, len(raw), *extra)
        if not result:
            raise MemoryError("native layout result allocation failed")
        try:
            error = self._library.wc_layout_result_error_v1(result)
            if error:
                raise ValueError(error.decode("utf-8", errors="replace"))
            size = self._library.wc_layout_result_size_v1(result)
            return ctypes.string_at(self._library.wc_layout_result_data_v1(result), size)
        finally:
            self._library.wc_layout_result_free_v1(result)

    def run(self, mode: str, data: bytes, *args) -> bytes:
        return self._call("wc_layout_run_v1", (mode, *args), data)

    def benchmark_stage(self, data: bytes, *args, warmups: int = 1,
                        repeats: int = 5) -> dict:
        if (type(warmups) is not int or not 0 <= warmups <= 31
                or type(repeats) is not int or not 1 <= repeats <= 1001):
            raise ValueError("invalid timing repetitions")
        return json.loads(self._call("wc_layout_benchmark_v1", ("encode", *args),
                                     data, warmups, repeats))
