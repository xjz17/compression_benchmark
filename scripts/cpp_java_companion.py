#!/usr/bin/env python3
"""
Append ExternalCompressionBench rows for the hybrid C++ driver (`benchmark_cpp` invokes
`org.example.CppBenchmarkCompanion` with `--phase external-only`). TSFile encodings are produced
natively in C++.

Same classpath / cwd conventions as ``java_bench_common.run_java_benchmark``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from java_bench_common import ROOT, _serf_env, java_cp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--data-type", required=True, choices=("int", "float"))
    ap.add_argument("--append", required=True, help="Existing CSV to append rows to (no header)")
    ap.add_argument(
        "--phase",
        required=True,
        choices=("external-only",),
        help="external-only: optional future JAR-only rows; TSFile / Camel / Sim-Piece / ELF / … are native in C++",
    )
    args = ap.parse_args()

    cmd = [
        "java",
        "-cp",
        java_cp(),
        "org.example.CppBenchmarkCompanion",
        "--input",
        args.input,
        "--data-type",
        args.data_type,
        "--append",
        args.append,
        "--phase",
        args.phase,
    ]
    env = _serf_env(None)
    env = {
        **env,
        "WEB_COMPRESSION_CPP_NATIVE_FASTPFOR": "1",
        "WEB_COMPRESSION_CPP_NATIVE_CHIMP32": "1",
        "WEB_COMPRESSION_CPP_NATIVE_SIMPLE8B": "1",
        "WEB_COMPRESSION_CPP_NATIVE_LECO_FLOAT": "1",
        "WEB_COMPRESSION_CPP_NATIVE_ALP_FLOAT": "1",
        "WEB_COMPRESSION_CPP_NATIVE_ELF_ON_CHIMP_FLOAT": "1",
        "WEB_COMPRESSION_CPP_NATIVE_ELF_PLUS_FLOAT": "1",
        "WEB_COMPRESSION_CPP_NATIVE_CAMEL_FLOAT": "1",
        "WEB_COMPRESSION_CPP_NATIVE_SIMPIECE_FLOAT": "1",
        "WEB_COMPRESSION_CPP_NATIVE_SPDP": "1",
        "WEB_COMPRESSION_CPP_NATIVE_MOST": "1",
        "WEB_COMPRESSION_CPP_NATIVE_MACHETE": "1",
        "WEB_COMPRESSION_CPP_NATIVE_CORAD": "1",
    }
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    main()
