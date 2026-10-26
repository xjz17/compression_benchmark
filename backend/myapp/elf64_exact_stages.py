"""IEEE-754 binary64 Elf/Elf+ source encoders and independently callable stages."""
from pathlib import Path
import subprocess

_CLI = Path(__file__).resolve().parents[2] / "cpp/build/elf64_exact_stages_cli"


def _run(command: str, data: bytes) -> bytes:
    result = subprocess.run([str(_CLI), command], input=bytes(data), capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _plus(method: str | bool) -> bool:
    if method in ("Elf", False):
        return False
    if method in ("Elf+", True):
        return True
    raise ValueError("Elf64 method must be Elf or Elf+")


def encode(data: bytes, method: str = "Elf") -> bytes:
    """Original source encoder (EL64 framing, 1600 values/chunk), not the stage composition."""
    return _run("elf-plus-original-encode" if _plus(method) else "elf-original-encode", data)


def decode(data: bytes, method: str = "Elf") -> bytes:
    return _run("elf-plus-original-decode" if _plus(method) else "elf-original-decode", data)


def eraser_forward(data: bytes) -> bytes:
    return _run("eraser-forward", data)


def eraser_inverse(data: bytes) -> bytes:
    return _run("eraser-inverse", data)


def xor_forward(data: bytes, method: str = "Elf") -> bytes:
    return _run("elf-plus-xor-forward" if _plus(method) else "elf-xor-forward", data)


def xor_inverse(data: bytes, method: str = "Elf") -> bytes:
    return _run("elf-plus-xor-inverse" if _plus(method) else "elf-xor-inverse", data)
