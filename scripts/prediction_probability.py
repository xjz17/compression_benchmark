#!/usr/bin/env python3
"""Real upstream probability predictors, separate from entropy coding.

PPMd, DMC and PAQ are explicitly the cmix v21 variants/submodels. cmix is
the real full predictor and is memory-gated. NNCP exposes its real upstream
Transformer through a separate byte-symbol API; LibNC requires Linux x86-64.
Prediction output is an auxiliary probability trace, NOT a reversible codec.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CMIX = ROOT / "external/compression/cmix"
DEFAULT_DRIVER = ROOT / "build/prediction_probability/prediction_probability_cli"
PROVIDERS = {
    "ppmd": {"name": "cmix v21 mod_ppmd_v2 byte model", "status": "implemented",
             "source": "src/models/ppmd.cpp:1322-1337", "unit": "bit P(1); optional 256-byte distribution",
             "license": "cmix GPL-3.0 combined work; see upstream notices", "defaults": {"order": 6, "memory_mib": 16}},
    "dmc": {"name": "cmix PAQ dmcModel count branch (not DMC forest)", "status": "implemented",
            "source": "src/models/paq8.cpp:7638-7736", "unit": "bit P(1), original 12-bit integer / 4096",
            "license": "upstream PAQ GPL-2.0-or-later; combined driver GPL-3.0",
            "defaults": {"extra_nodes": 65536, "cloning_threshold_fixed_point": 1024}},
    "paq": {"name": "cmix v21 PAQ8l/PAQ8px/PAQ8pxd hybrid final predictor", "status": "implemented",
            "source": "src/models/paq8.cpp:8228-8383", "unit": "bit P(1), final p() / 4096",
            "license": "upstream PAQ GPL-2.0-or-later; combined driver GPL-3.0", "defaults": {"memory_level": 0}},
    "cmix": {"name": "cmix v21 full Predictor, all-byte vocabulary, no preprocessing/pretraining",
             "status": "implemented_resource_gated_not_runtime_verified", "source": "src/predictor.cpp:361-468",
             "unit": "bit P(1), upstream float", "license": "GPL-3.0 combined work",
             "reason": "upstream recommends at least 32 GB RAM; explicit allow_large_memory required"},
    "nncp": {"name": "Fabrice Bellard NNCP v3.3 Transformer stream1 profile", "status": "implemented_linux_x86_64_dependency",
             "source_url": "https://bellard.org/nncp/nncp-2024-06-05.tar.gz",
             "source": "cpp/src/nncp_probability_bridge.c; external/prediction_probability/nncp-2024-06-05/nncp.c",
             "unit": "256 float32 probabilities for the NEXT raw byte; not bit P(1)",
             "license": "NNCP source MIT; bundled LibNC binary separately freely redistributable",
             "defaults": {"profile": "transformer-stream1-v1", "seed": 123, "batch_size": 1,
                          "seg_len": 1, "vocabulary": 256, "max_symbols": 4096},
             "reason": "native Linux x86-64 LibNC or explicitly enabled Docker; separate NNCPSymbolSession API"},
}


class PredictorUnavailable(RuntimeError):
    pass


def native_sources() -> list[Path]:
    """Sources for either standalone build or an equivalent CMake target."""
    sources = [ROOT / "cpp/src/prediction_probability_model.cpp",
               ROOT / "cpp/src/prediction_probability_cli.cpp",
               CMIX / "src/predictor.cpp", CMIX / "src/context-manager.cpp"]
    for group in ("contexts", "mixer", "states", "models"):
        sources.extend(p for p in sorted((CMIX / "src" / group).glob("*.cpp"))
                       if p.name not in {"paq8.cpp", "ppmd.cpp"})
    return sources


def build_driver(*, force: bool = False) -> Path:
    """Build only this isolated executable, never modify cmix source/builds."""
    override = os.environ.get("COMPRESSION_BENCHMARK_PREDICTION_DRIVER")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise PredictorUnavailable(f"configured prediction driver is missing: {path}")
        return path
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise PredictorUnavailable("a C++14 compiler is required for real upstream models")
    sources = native_sources()
    dependencies = sources + list((CMIX / "src").rglob("*.h")) + [
        CMIX / "src/models/paq8.cpp", CMIX / "src/models/ppmd.cpp",
        ROOT / "cpp/src/prediction_probability_model.hpp"]
    digest = hashlib.sha256((compiler + "|-std=c++14|-O2|-ffp-contract=off").encode())
    for path in sorted(dependencies):
        if not path.is_file():
            raise PredictorUnavailable(f"required upstream source missing: {path}")
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    stamp = DEFAULT_DRIVER.with_suffix(".sha256")
    if not force and DEFAULT_DRIVER.is_file() and stamp.is_file() and stamp.read_text().strip() == fingerprint:
        return DEFAULT_DRIVER
    DEFAULT_DRIVER.parent.mkdir(parents=True, exist_ok=True)
    # Serialize concurrent test/session builds. No lock is held while running a predictor.
    import fcntl
    with (DEFAULT_DRIVER.parent / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and DEFAULT_DRIVER.is_file() and stamp.is_file() and stamp.read_text().strip() == fingerprint:
            return DEFAULT_DRIVER
        fd, output = tempfile.mkstemp(prefix="prediction_probability_", dir=DEFAULT_DRIVER.parent)
        os.close(fd)
        try:
            command = [compiler, "-std=c++14", "-O2", "-ffp-contract=off", *map(str, sources), "-o", output]
            result = subprocess.run(command, capture_output=True, text=True, timeout=240)
            if result.returncode:
                raise PredictorUnavailable("native predictor build failed:\n" + result.stderr[-12000:])
            os.replace(output, DEFAULT_DRIVER)
            stamp.write_text(fingerprint + "\n")
        finally:
            if os.path.exists(output):
                os.unlink(output)
    return DEFAULT_DRIVER


def _command(provider: str, *, allow_large_memory: bool = False, ppmd_order: int = 6,
             ppmd_memory_mib: int = 16, paq_level: int = 0) -> list[str]:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown probability provider: {provider}")
    if provider == "nncp":
        raise ValueError("NNCP predicts byte symbols: use NNCPSymbolSession or predict_symbol_probabilities")
    if provider == "cmix" and not allow_large_memory:
        raise PredictorUnavailable(PROVIDERS[provider]["reason"])
    if not 2 <= ppmd_order <= 64 or not 4 <= ppmd_memory_mib <= 256 or not 0 <= paq_level <= 3:
        raise ValueError("bounded parameters: PPMd order 2..64, memory 4..256 MiB, PAQ level 0..3")
    command = [str(build_driver()), provider, "--ppmd-order", str(ppmd_order),
               "--ppmd-memory-mib", str(ppmd_memory_mib), "--paq-level", str(paq_level)]
    if allow_large_memory:
        command.append("--allow-large-memory")
    return command


class PredictionSession:
    """One fresh process per model: upstream PAQ/DMC global state cannot leak.

    predict() is idempotent until observe(bit). Always predict before observing.
    Probabilities are the source model outputs, not a quantized entropy CDF.
    """
    def __init__(self, provider: str, *, timeout: float = 120, **options):
        self.provider = provider
        self.timeout = timeout
        self._pending = False
        self._bit_count = 0
        self._process = subprocess.Popen(_command(provider, **options), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _write(self, data: bytes) -> None:
        if self._process.poll() is not None:
            raise PredictorUnavailable(self._process.stderr.read().decode(errors="replace"))
        try:
            self._process.stdin.write(data)
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise PredictorUnavailable(self._process.stderr.read().decode(errors="replace")) from exc

    def _read(self, count: int) -> bytes:
        data = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(data) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._process.stdout], [], [], max(0, remaining))[0]:
                self.close()
                raise TimeoutError(f"{self.provider} prediction exceeded {self.timeout}s")
            chunk = os.read(self._process.stdout.fileno(), count - len(data))
            if not chunk:
                error = self._process.stderr.read().decode(errors="replace")
                raise PredictorUnavailable(f"native probability stream ended early: {error}")
            data.extend(chunk)
        return bytes(data)

    def predict(self) -> float:
        self._write(b"P")
        probability = struct.unpack("<d", self._read(8))[0]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("invalid upstream probability")
        self._pending = True
        return probability

    def observe(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        if not self._pending:
            raise ValueError("predict must precede observe")
        self._write(b"O" + bytes([bit]))
        self._pending = False
        self._bit_count += 1

    def byte_distribution(self) -> list[float]:
        if self.provider != "ppmd" or self._bit_count % 8:
            raise ValueError("only ppmd at a byte boundary exposes a byte distribution")
        self._write(b"D")
        return list(struct.unpack("<256d", self._read(256 * 8)))

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._process.stdin.write(b"Q")
                self._process.stdin.close()
                self._process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait()
        for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
            if pipe and not pipe.closed:
                pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def predict_probabilities(data: bytes, provider: str = "ppmd", *, timeout: float = 120,
                          **options) -> list[float]:
    """P(bit=1) BEFORE each observed bit, MSB-first; no entropy coding occurs."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    data = bytes(data)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("probability trace limited to 16 MiB input")
    result = subprocess.run(_command(provider, **options), input=b"T" + struct.pack("<Q", len(data)) + data + b"Q",
                            capture_output=True, timeout=timeout)
    if result.returncode:
        detail = result.stderr.decode(errors="replace")
        if provider == "cmix":
            detail += " Full cmix retains its large upstream allocation; check memory/resource limits."
        raise PredictorUnavailable(f"{provider} predictor exited with status {result.returncode}: {detail}")
    if len(result.stdout) != len(data) * 8 * 8:
        raise RuntimeError("native predictor returned an invalid trace length")
    values = array.array("d")
    values.frombytes(result.stdout)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise RuntimeError("native predictor returned an invalid probability")
    return values.tolist()


class NNCPSymbolSession(PredictionSession):
    """Real NNCP next-byte probabilities, with one model update after observe.

    The fixed transformer-stream1-v1 profile retains the upstream default
    network and uses a one-symbol segment. It is not the default file profile.
    """
    def __init__(self, *, timeout: float = 120):
        from build_nncp_probability import nncp_command
        self.provider = "nncp"
        self.timeout = timeout
        self._pending = False
        self._symbol_count = 0
        try:
            command = nncp_command()
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise PredictorUnavailable(str(exc)) from exc
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0)

    def predict(self) -> list[float]:
        if self._symbol_count >= 4096:
            raise ValueError("NNCP session is limited to 4096 observed symbols")
        self._write(b"D")
        values = list(struct.unpack("<256f", self._read(1024)))
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values) or abs(sum(values) - 1) > 0.0001:
            raise RuntimeError("invalid upstream NNCP distribution")
        self._pending = True
        return values

    def observe(self, symbol: int) -> None:
        if not isinstance(symbol, int) or not 0 <= symbol <= 255:
            raise ValueError("NNCP symbol must be an integer in 0..255")
        if not self._pending:
            raise ValueError("predict must precede observe")
        self._write(b"O" + bytes([symbol]))
        if self._read(1) != b"\x01":
            raise RuntimeError("NNCP observation failed")
        self._pending = False
        self._symbol_count += 1

    def byte_distribution(self) -> list[float]:
        return self.predict()


def predict_symbol_probabilities(data: bytes, *, timeout: float = 120) -> list[list[float]]:
    """One 256-float distribution BEFORE each byte, including online updates."""
    from build_nncp_probability import nncp_command
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    data = bytes(data)
    if len(data) > 4096:
        raise ValueError("NNCP probability trace is limited to 4096 bytes")
    try:
        command = nncp_command()
    except (RuntimeError, subprocess.SubprocessError) as exc:
        raise PredictorUnavailable(str(exc)) from exc
    result = subprocess.run(command, input=b"T" + struct.pack("<Q", len(data)) + data + b"Q",
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise PredictorUnavailable("NNCP predictor failed: " + result.stderr.decode(errors="replace"))
    if len(result.stdout) != len(data) * 1024:
        raise RuntimeError("invalid NNCP trace length")
    values = [list(row) for row in struct.iter_unpack("<256f", result.stdout)]
    if any(any(not math.isfinite(p) or not 0 <= p <= 1 for p in row) or abs(sum(row) - 1) > 0.0001 for row in values):
        raise RuntimeError("invalid NNCP trace probability")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("info")
    sub.add_parser("build")
    predict = sub.add_parser("predict")
    predict.add_argument("provider", choices=PROVIDERS)
    predict.add_argument("input", type=Path)
    predict.add_argument("--allow-large-memory", action="store_true")
    args = parser.parse_args()
    if args.action == "info":
        print(json.dumps(PROVIDERS, indent=2, ensure_ascii=False))
    elif args.action == "build":
        print(build_driver())
    else:
        if args.provider == "nncp":
            print(json.dumps({"provider": "nncp", "unit": "P(next raw byte=symbol), symbol 0..255",
                              "profile": "transformer-stream1-v1",
                              "probabilities": predict_symbol_probabilities(args.input.read_bytes())}))
            return
        print(json.dumps({"provider": args.provider, "unit": "P(bit=1)", "bit_order": "MSB-first",
                          "probabilities": predict_probabilities(args.input.read_bytes(), args.provider,
                                                                  allow_large_memory=args.allow_large_memory)}))


if __name__ == "__main__":
    main()
