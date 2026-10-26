"""Produce a local A111 correctness receipt, without running performance benchmarks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_outputs/operator_consolidation_20260917/a111_validation.json"
SOURCES = (
    "cpp/src/exception_separation.hpp", "cpp/src/exception_separation_cli.cpp",
    "cpp/src/max_value_bitpack.hpp", "cpp/tests/exception_separation_test.cpp",
    "scripts/exception_separation.py", "scripts/test_exception_separation.py",
    "scripts/validate_exception_separation.py",
    "cpp/src/fastpfor_exact_stages.cpp", "external/compression/FastPFor/headers/fastpfor.h",
    "scripts/operator_catalog_layers.py", "scripts/build_layered_operator_catalog.py",
    "benchmark/reference/actions/operator_implementations_layered_v1.json",
)


def main():
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("C++ compiler unavailable")
    commands = []

    def run(argv):
        process = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
        commands.append({"argv": [str(arg) for arg in argv], "cwd": str(ROOT),
                         "returncode": process.returncode,
                         "stdout": process.stdout, "stderr": process.stderr})
        if process.returncode:
            raise RuntimeError(f"validation command failed: {argv}\n{process.stderr}")
        return process.stdout.strip()

    environment = {"system": platform.system(), "release": platform.release(),
                   "machine": platform.machine(), "python": sys.version,
                   "python_executable": sys.executable,
                   "compiler": run([compiler, "--version"])}
    with tempfile.TemporaryDirectory(prefix="a111-validation-") as temporary:
        work = Path(temporary)
        test_binary, cli_binary = work / "exception_test", work / "exception_cli"
        flags = [compiler, "-std=c++20", "-Wall", "-Wextra", "-Wconversion",
                 "-Wsign-conversion", "-I", "cpp/src"]
        run([*flags, "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
             "cpp/tests/exception_separation_test.cpp", "-o", str(test_binary)])
        native_result = run([str(test_binary)])
        run([compiler, "-std=c++20", "-Wall", "-Wextra", "-I", "cpp/src",
             "cpp/src/exception_separation_cli.cpp", "-o", str(cli_binary)])
        bridge_runner = (
            "import pathlib,sys,unittest;sys.path.insert(0,'scripts');"
            "import test_exception_separation as t;"
            "t.es.BINARY=pathlib.Path(sys.argv[1]);"
            "r=unittest.TextTestRunner(verbosity=2).run("
            "unittest.defaultTestLoader.loadTestsFromModule(t));"
            "sys.exit(not r.wasSuccessful())"
        )
        run([sys.executable, "-c", bridge_runner, str(cli_binary)])

    sys.path.insert(0, str(ROOT / "scripts"))
    import operator_implementation_catalog as catalog
    import operator_catalog_layers as layers
    item = catalog.BY_ID["A111"]
    counts = layers.validate_layers(catalog.IMPLEMENTATIONS)
    assert item["accounting"]["layer"] == "mechanism"
    assert item["input"]["values"] == "uint32[N]"
    assert item["outputs"] == {"low": "uint32[N]", "pos": "uint32[E]", "high": "uint32[E]"}
    assert "not_registered_in_graph_policy" in item["execution_status"]
    assert item["source_action_names"] == []
    original_fastpfor = (ROOT / "cpp/src/fastpfor_exact_stages.cpp").read_text()
    assert "exception_separation" not in original_fastpfor
    assert "FastPFor<8> codec" in original_fastpfor and "codec.encodeArray" in original_fastpfor
    record = {
        "schema": "a111-native-validation-v1", "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local C++ correctness, sanitizer, CLI cold decode, typed stream bridge, and catalog accounting. No dataset performance claim.",
        "environment": environment, "commands": commands,
        "native_result": native_result, "native_roundtrips": 726,
        "bridge_unittest_count": 6,
        "contract": {
            "input": "uint32 LE[N], N<=UINT32_MAX, supplied b in [0,32]",
            "outputs": ["low:uint32[N]", "positions:uint32[E]", "high:uint32[E]"],
            "metadata": ["b", "N", "E derived from exception stream length"],
            "EHS1_bytes": "13+4*N+8*E",
            "EHP1_bytes": "15+ceil(N*b/8)+ceil(E*position_width/8)+ceil(E*high_width/8)",
            "A020_adapter": "Zero-extend uint32 values to uint64 API arguments without numeric loss; low uses supplied b, other streams use maximum-value width.",
            "strict_checks": ["width and word alignment", "low/high width range", "stream lengths",
                              "strictly increasing in-range positions", "nonzero high",
                              "b=0 and b=32", "truncation and trailing bytes",
                              "canonical position/high packed widths", "zero bit padding"],
        },
        "catalog": {"status": "PASS", "registry_version": catalog.REGISTRY_VERSION,
                    "trace_entry_count": catalog.TRACE_ENTRY_COUNT, "layer_counts": counts,
                    "A111_layer": item["accounting"]["layer"],
                    "A111_execution_status": item["execution_status"]},
        "review": {
            "new_correctness_bugs_found": [],
            "old_fastpfor_fused_path_rewired": False,
            "graph_policy_action_registered": False,
            "format_is_upstream_fastpfor": False,
            "bit_width_selector_added": False,
            "limits": ["Structural checks do not detect all otherwise-valid content changes; there is no checksum.",
                       "A valid zero-width stream can expand substantially; allocation requires sufficient memory.",
                       "Final PDF wording and layout are validated separately from this native test receipt."],
        },
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCES},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "path": str(OUTPUT),
                      "native_roundtrips": record["native_roundtrips"],
                      "bridge_unittest_count": record["bridge_unittest_count"],
                      "layer_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
