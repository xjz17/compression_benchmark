"""Thin C++ adapter preserving the legacy snap 1.1.1 SNM1 match packet.

The BSD-3-Clause attribution and ported implementation are retained in
cpp/src/snappy_exact_stages.cpp. No Python match-selection kernel remains.
"""
from .cpp_action_bridge import run as _cpp_run

def lz_match_forward(data: bytes) -> bytes:
    return _cpp_run("snappy_exact_stages_cli", "snap111-forward", data)

def lz_match_inverse(data: bytes) -> bytes:
    return _cpp_run("snappy_exact_stages_cli", "lz-inverse", data)
