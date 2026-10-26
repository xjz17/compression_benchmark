"""Process/typed-data adapter only; all codec kernels execute in C++."""
from pathlib import Path
import subprocess

BUILD = Path(__file__).resolve().parents[2] / "cpp/build"


def run(binary: str, command: str, data: bytes, *args) -> bytes:
    path = BUILD / binary
    if not path.is_file():
        raise RuntimeError(f"missing C++ action binary {path}; build target {binary}")
    result = subprocess.run([str(path), command, *(str(a) for a in args)],
                            input=bytes(data), capture_output=True)
    if result.returncode:
        raise ValueError(f"{binary} {command}: " + result.stderr.decode(errors="replace").strip())
    return result.stdout
