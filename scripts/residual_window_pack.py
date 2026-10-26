"""Native A035/A039 residual-window packing bridge.

The C++ CLI consumes already-computed XOR residual words.  It does not derive
the residual, own the first source word, or silently change the word width.
"""
from __future__ import annotations

from pathlib import Path
import subprocess


ELF_PLUS_ACTION = "op.trailing_zero_pack.elf_plus_residual.cpp_v1"
CHIMP_ACTION = "op.trailing_zero_pack.chimp_residual.cpp_v1"
BINARY = Path(__file__).resolve().parents[1] / "cpp/build/residual_window_pack_cli"


def _profile(*, word_bytes: int, elf_plus: bool) -> str:
    if word_bytes not in (4, 8):
        raise ValueError("residual window packing requires 32-bit or 64-bit words")
    return ("elfplus" if elf_plus else "chimp") + str(word_bytes * 8)


def run(command: str, payload: bytes, *, word_bytes: int, elf_plus: bool) -> bytes:
    if command not in ("encode", "decode"):
        raise ValueError("command must be encode or decode")
    if not BINARY.is_file():
        raise RuntimeError(f"missing C++ action binary {BINARY}; build residual_window_pack_cli")
    result = subprocess.run(
        [str(BINARY), command, _profile(word_bytes=word_bytes, elf_plus=elf_plus)],
        input=bytes(payload), capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def forward(payload: bytes, *, word_bytes: int, elf_plus: bool = True) -> bytes:
    if len(payload) % word_bytes:
        raise ValueError("residual payload is not aligned to the declared word width")
    return run("encode", payload, word_bytes=word_bytes, elf_plus=elf_plus)


def inverse(packet: bytes, *, word_bytes: int, elf_plus: bool = True) -> bytes:
    result = run("decode", packet, word_bytes=word_bytes, elf_plus=elf_plus)
    if len(result) % word_bytes:
        raise ValueError("native residual decoder returned an unaligned payload")
    return result
