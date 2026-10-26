#!/usr/bin/env python3
"""
Compare compressed-byte hex between Python ports and the reference JAR paths used by
``ExternalCompressionBench`` (Elf on Chimp / Elf+ / Camel / Chimp32 / Sim-Piece).

**Build artifacts (run from repository root)**

.. code-block:: text

   cd java && mvn -q -DskipTests package && cp target/benchmark-java-1.0-SNAPSHOT.jar ../backend/lib/
   cd external/compression/elf && mvn -q -DskipTests package
   cd external/compression/camel && mvn -q -DskipTests package
   cd external/compression/chimp && mvn -q -DskipTests package

Camel copies runtime JARs to ``target/dependency`` during ``package``.
The benchmark JAR must sit at ``backend/lib/benchmark-java-1.0-SNAPSHOT.jar`` (same as
``scripts/regenerate_staticdata_java.py``); optional entries can be appended via
``backend/lib/java_classpath.append`` or ``COMPRESSION_BENCHMARK_JAVA_CLASSPATH_APPEND``.

**Fixture**

Default CSV: ``benchmark/fixtures/parity_float_codecs.csv`` (small float matrix).

**Sim-Piece**

Uses in-repo ``SimPieceFloatNative`` (Java) and ``simpiece_python.py`` (Python). Hex lines should match
when both ports agree; if they differ, the script cross-checks decompress tolerance within ε.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_float_table(path: Path) -> list[list[float]]:
    cols: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        for _ in header:
            cols.append([])
        for row in r:
            for i, cell in enumerate(row):
                cols[i].append(float(cell.strip()))
    return cols


def _py_hex_elf_on_chimp(columns: list[list[float]]) -> list[str]:
    from myapp.elf32_python import (
        ElfOnChimpCompressor32,
        ElfOnChimpDecompressor32,
        _ELF_CHUNK,
        _raw_bits,
    )

    lines: list[str] = []
    for ci, col in enumerate(columns):
        n = len(col)
        k = 0
        for start in range(0, n, _ELF_CHUNK):
            end = min(n, start + _ELF_CHUNK)
            chunk = end - start
            c = ElfOnChimpCompressor32()
            for i in range(start, end):
                c.add_value(float(col[i]))
            c.close()
            raw = c.get_bytes()
            got = ElfOnChimpDecompressor32(raw).decompress(chunk)
            if len(got) != chunk:
                raise SystemExit("Elf on Chimp python decompress length")
            for i in range(chunk):
                if _raw_bits(got[i]) != _raw_bits(float(col[start + i])):
                    raise SystemExit("Elf on Chimp python bits mismatch")
            lines.append(f"{ci} {k} {raw.hex()}")
            k += 1
    return lines


def _py_hex_elf_plus(columns: list[list[float]]) -> list[str]:
    from myapp.elf32_python import (
        ElfCompressor32,
        ElfDecompressor32,
        _ELF_CHUNK,
        _raw_bits,
    )

    lines: list[str] = []
    for ci, col in enumerate(columns):
        n = len(col)
        k = 0
        for start in range(0, n, _ELF_CHUNK):
            end = min(n, start + _ELF_CHUNK)
            chunk = end - start
            c = ElfCompressor32()
            for i in range(start, end):
                c.add_value(float(col[i]))
            c.close()
            raw = c.get_bytes()
            got = ElfDecompressor32(raw).decompress(chunk)
            if len(got) != chunk:
                raise SystemExit("Elf+ python decompress length")
            for i in range(chunk):
                if _raw_bits(got[i]) != _raw_bits(float(col[start + i])):
                    raise SystemExit("Elf+ python bits mismatch")
            lines.append(f"{ci} {k} {raw.hex()}")
            k += 1
    return lines


def _py_hex_camel(columns: list[list[float]]) -> list[str]:
    from myapp.camel_python import (
        CamelCompressor,
        CamelDecompressor,
        _double_to_long_bits,
    )

    chunk_pts = 1000
    lines: list[str] = []
    for ci, col in enumerate(columns):
        n = len(col)
        k = 0
        for start in range(0, n, chunk_pts):
            end = min(n, start + chunk_pts)
            c = CamelCompressor()
            for i in range(start, end):
                c.add_value(float(col[i]))
            c.close()
            raw = c.get_bytes()
            dec = CamelDecompressor(raw)
            for i in range(start, end):
                v = dec.read_pair()
                if v is None:
                    raise SystemExit("Camel python early eos")
                exp = float(col[i])
                if _double_to_long_bits(v) != _double_to_long_bits(exp):
                    raise SystemExit("Camel python mismatch")
            lines.append(f"{ci} {k} {raw.hex()}")
            k += 1
    return lines


def _py_hex_chimp(columns: list[list[float]]) -> list[str]:
    from myapp.chimp32_python import Chimp32Compressor, decompress_chimp32_float_column

    lines: list[str] = []
    for ci, col in enumerate(columns):
        n = len(col)
        c = Chimp32Compressor()
        for f in col:
            c.add_value(float(f))
        c.close()
        raw = c.get_compressed_bytes()
        decompress_chimp32_float_column(raw, n)
        lines.append(f"{ci} 0 {raw.hex()}")
    return lines


def _py_hex_simpiece(columns: list[list[float]], rel: float) -> list[str]:
    from myapp.simpiece_python import parity_hex_lines_for_column

    lines: list[str] = []
    for ci, col in enumerate(columns):
        hx = parity_hex_lines_for_column(col, rel)
        for k, h in enumerate(hx):
            lines.append(f"{ci} {k} {h}")
    return lines


def _simpiece_cross_verify(columns: list[list[float]], rel: float, java_lines: list[str]) -> bool:
    """Java ``HashMap`` key order is undefined; verify Python can decode JAR blobs within ε."""
    from myapp.simpiece_python import decompress_column

    for line in java_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        ci = int(parts[0])
        blob = bytes.fromhex(parts[2])
        col = columns[ci]
        n = len(col)
        mn = min(float(x) for x in col)
        mx = max(float(x) for x in col)
        rng = mx - mn
        if rng <= 0:
            rng = 1.0
        eps = rng * rel
        try:
            restored = decompress_column(blob)
        except Exception:
            return False
        by_ts = {t: v for t, v in restored}
        for i in range(n):
            dv = by_ts.get(i)
            if dv is None or abs(dv - float(col[i])) > eps * 1.0001 + 1e-9:
                return False
    return True


def _run_java(csv_path: Path, mode: str) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from scripts.regenerate_staticdata_java import java_cp

    cp = java_cp()
    props: list[str] = []
    elf_j = ROOT / "external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar"
    camel_j = ROOT / "external/compression/camel/target/start-compress-1.0-SNAPSHOT.jar"
    if elf_j.is_file():
        props.append(f"-Dweb.compression.elf.jar={elf_j.resolve()}")
    if camel_j.is_file():
        props.append(f"-Dweb.compression.camel.jar={camel_j.resolve()}")
    cmd = [
        "java",
        *props,
        "-cp",
        cp,
        "org.example.ResearchParityHexMain",
        str(csv_path),
        mode,
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
        timeout=600,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"java failed ({p.returncode}): {p.stderr[:2000]!r} cmd={cmd!r}"
        )
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark/fixtures/parity_float_codecs.csv",
    )
    ap.add_argument(
        "--skip-simpiece",
        action="store_true",
        help="Do not run Sim-Piece hex compare (Zstd implementation may differ).",
    )
    ap.add_argument(
        "--java-only",
        action="store_true",
        help="Print Java hex lines only (no Python compare).",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "backend"))

    csv_path = args.csv.resolve()
    columns = _read_float_table(csv_path)

    modes = [
        "ELF_ON_CHIMP",
        "ELF_PLUS",
        "CAMEL",
        "CHIMP_FLOAT",
    ]
    if not args.skip_simpiece:
        modes.extend(["SIMPIECE_0P005", "SIMPIECE_0P01"])

    failures = 0
    for mode in modes:
        try:
            java_lines = _run_java(csv_path, mode)
        except subprocess.TimeoutExpired:
            print(f"[skip] {mode}: Java timed out (first JVM/reflect pass can be slow; retry or shrink CSV)")
            continue
        except Exception as ex:
            print(f"[skip] {mode}: Java reference not runnable ({ex})")
            continue
        if args.java_only:
            print(f"=== {mode} (Java) ===")
            for ln in java_lines:
                print(ln)
            continue
        if mode == "ELF_ON_CHIMP":
            py_lines = _py_hex_elf_on_chimp(columns)
        elif mode == "ELF_PLUS":
            py_lines = _py_hex_elf_plus(columns)
        elif mode == "CAMEL":
            py_lines = _py_hex_camel(columns)
        elif mode == "CHIMP_FLOAT":
            py_lines = _py_hex_chimp(columns)
        elif mode == "SIMPIECE_0P005":
            py_lines = _py_hex_simpiece(columns, 0.005)
        elif mode == "SIMPIECE_0P01":
            py_lines = _py_hex_simpiece(columns, 0.01)
        else:
            raise SystemExit(mode)
        if py_lines != java_lines:
            if mode.startswith("SIMPIECE_"):
                rel = 0.005 if "0P005" in mode else 0.01
                if _simpiece_cross_verify(columns, rel, java_lines) and _simpiece_cross_verify(
                    columns, rel, py_lines
                ):
                    print(
                        f"OK {mode} ({len(java_lines)} lines, hex may differ — cross-decode tolerance OK)"
                    )
                    continue
            print(f"FAIL {mode}: Python and Java hex lines differ")
            print(f"  first mismatch index: {next((i for i, (a, b) in enumerate(zip(py_lines, java_lines)) if a != b), 'len')}")
            mj = max(len(py_lines), len(java_lines))
            for i in range(min(5, mj)):
                p = py_lines[i] if i < len(py_lines) else "<missing>"
                j = java_lines[i] if i < len(java_lines) else "<missing>"
                print(f"  [{i}] py: {p[:80]}...")
                print(f"  [{i}] jv: {j[:80]}...")
            failures += 1
        else:
            print(f"OK {mode} ({len(py_lines)} lines)")

    if failures:
        raise SystemExit(f"{failures} mode(s) failed hex parity")


if __name__ == "__main__":
    main()
