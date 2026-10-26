"""Thin interface to the C++ CORAD-lossless v2 operator kernels.

The wire is the project's exact-repair CORAD variant, not upstream OMP.
Historical action names remain replay-compatible; no Python codec is executed.
"""
from __future__ import annotations
import struct
from .cpp_action_bridge import run as _cpp_run

NORMALIZE = "op.scaling.corad_normalize_exact.python_v2"
TRICKLET = "op.dictionary.corad_tricklet_xor.python_v2"
ENTROPY = "op.block_state_control.corad_deflate9.python_v2"
FRAME = "op.container_frame.corad_lossless.python_v2"
ACTION_NAMES = (NORMALIZE, TRICKLET, ENTROPY, FRAME)
ACTION_OPERATOR_IDS = dict(zip(ACTION_NAMES, (
    "scaling", "dictionary", "block_state_control", "container_frame",
)))
TRICK = 40
MAX_ATOMS = 4
_COMMANDS = dict(zip(ACTION_NAMES, ("normalize", "tricklet", "entropy", "frame")))

def _words(raw: bytes) -> tuple[int, ...]:
    """Inspection-only raw64 transport adapter used by counterexample tests."""
    if len(raw) % 8:
        raise ValueError("CORAD requires complete 64-bit lanes")
    return struct.unpack(f"<{len(raw) // 8}Q", raw)

def _float(word: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", word))[0]

def _pack_words(words) -> bytes:
    return struct.pack(f"<{len(words)}Q", *words)

def normalize_forward(raw: bytes, kind: str) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "normalize-forward", raw, kind)

def normalize_inverse(raw: bytes) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "normalize-inverse", raw)

def tricklet_forward(raw: bytes) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "tricklet-forward", raw)

def tricklet_inverse(raw: bytes) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "tricklet-inverse", raw)

def entropy_forward(raw: bytes) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "entropy-forward", raw)

def entropy_inverse(raw: bytes) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "entropy-inverse", raw)

def frame_forward(raw: bytes) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "frame-forward", raw)

def frame_inverse(raw: bytes) -> bytes:
    return _cpp_run("corad_lossless_cpp_cli", "frame-inverse", raw)

def forward(action: str, raw: bytes, kind: str = "bytes") -> bytes:
    if action not in _COMMANDS:
        raise ValueError(f"unknown CORAD action: {action}")
    return _cpp_run("corad_lossless_cpp_cli", _COMMANDS[action] + "-forward", raw, kind)

def inverse(action: str, raw: bytes) -> bytes:
    if action not in _COMMANDS:
        raise ValueError(f"unknown CORAD action: {action}")
    return _cpp_run("corad_lossless_cpp_cli", _COMMANDS[action] + "-inverse", raw)

def encode(raw: bytes, kind: str) -> bytes:
    return frame_forward(entropy_forward(tricklet_forward(normalize_forward(raw, kind))))

def decode(wire: bytes, kind: str | None = None) -> bytes:
    normalized = tricklet_inverse(entropy_inverse(frame_inverse(wire)))
    if kind is not None and (kind not in ("float64", "int64")
                            or normalized[4] != {"float64": 0, "int64": 1}[kind]):
        raise ValueError("CORAD original kind mismatch")
    return normalize_inverse(normalized)
