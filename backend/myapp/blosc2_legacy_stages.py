"""Thin stage API for the pinned native Blosc2 binding and B2P1 wire.

C++ links C-Blosc2 directly; Python supplies no codec or framing kernel.
"""
from .cpp_action_bridge import run as _cpp_run

def binding_core_forward(data: bytes) -> bytes:
    return _cpp_run("blosc2_binding_cpp_cli", "core-forward", data)

def binding_core_inverse(data: bytes) -> bytes:
    return _cpp_run("blosc2_binding_cpp_cli", "core-inverse", data)

def frame_forward(data: bytes) -> bytes:
    return _cpp_run("blosc2_binding_cpp_cli", "frame-forward", data)

def frame_inverse(data: bytes) -> bytes:
    return _cpp_run("blosc2_binding_cpp_cli", "frame-inverse", data)
