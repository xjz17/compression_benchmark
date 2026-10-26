#!/usr/bin/env python3
"""Run lzbench on one patch payload file; emit CSV rows to stdout.

Used by Java/C++ patch benches (same parser/catalog as ``myapp.lzbench_external``).

Output columns: algorithm,originalSize,compressedSize,compressionTime,decompressionTime
(times in nanoseconds).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from myapp.lzbench_external import (  # noqa: E402
    _run_lzbench_on_bytes,
    lzbench_bench_enabled,
    lzbench_executable_path,
    allowlist_wants_lzbench,
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: lzbench_patch_cli.py <payload.bin>", file=sys.stderr)
        return 2
    if not lzbench_bench_enabled():
        return 0
    if not allowlist_wants_lzbench():
        return 0
    exe = lzbench_executable_path()
    if exe is None:
        return 1
    payload = Path(sys.argv[1]).read_bytes()
    if not payload:
        return 0
    rows = _run_lzbench_on_bytes(payload, exe)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    for row in rows:
        writer.writerow(
            [
                row["algorithm"],
                row["originalSize"],
                row["compressedSize"],
                row["compressionTime"],
                row["decompressionTime"],
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
