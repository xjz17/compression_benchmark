#!/usr/bin/env python3
"""
Build Blosc2 column CLI for Java/C++ benchmarks.

From repository root::

    python scripts/build_blosc2_col_bench.py

Output: ``external/compression/c-blosc2/build/bin/blosc2_col_bench`` (``.exe`` on Windows).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blosc_col_bench_build import BLOSC2_SPEC, build_col_bench


def main() -> None:
    build_col_bench(BLOSC2_SPEC)


if __name__ == "__main__":
    main()
