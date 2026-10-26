"""Thin stage API for the C++ port of the project's legacy REGER wire."""
from .cpp_action_bridge import run as _cpp_run

def block_core_forward(data: bytes) -> bytes:
    return _cpp_run("reger_legacy_cpp_cli", "core-forward", data)

def block_core_inverse(data: bytes) -> bytes:
    return _cpp_run("reger_legacy_cpp_cli", "core-inverse", data)

def frame_forward(data: bytes) -> bytes:
    return _cpp_run("reger_legacy_cpp_cli", "frame-forward", data)

def frame_inverse(data: bytes) -> bytes:
    return _cpp_run("reger_legacy_cpp_cli", "frame-inverse", data)
