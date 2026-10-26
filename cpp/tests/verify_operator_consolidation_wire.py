"""Compare rebuilt C++ stage entries with captured pre-consolidation binaries."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess


SOURCES = [
    "operator_layout_primitives.hpp", "dictionary_recode.hpp",
    "reference_operator_actions.cpp", "dictionary64_action_cli.cpp",
    "openzl_layout_stages.cpp", "legacy_numeric_action_stages.cpp",
    "learning_codec_exact_stages.cpp", "learning_codec_exact_stages.hpp",
    "p04_exact_stages.cpp", "spdp_exact_stages.cpp",
    "general_codec_reference_stages.cpp", "general_codec_reference_stages.hpp",
    "native_lz_action_stages.cpp", "native_lz_action_stages.hpp",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    counts = Counter()
    binaries = set()
    random_source = random.Random(20260917)

    def execute(directory, binary, command, payload):
        result = subprocess.run(
            [str(directory / binary), *command], input=payload, capture_output=True,
            check=False, timeout=30,
        )
        if result.returncode:
            raise AssertionError(f"{binary} {command}: {result.stderr.decode(errors='replace')}")
        return result.stdout

    def same(binary, command, payload):
        before = execute(args.baseline, binary, command, payload)
        after = execute(args.build, binary, command, payload)
        if before != after:
            raise AssertionError(f"wire changed: {binary} {command}, input sha256={hashlib.sha256(payload).hexdigest()}")
        counts[" ".join([binary, *command])] += 1
        binaries.add(binary)
        return after

    def roundtrip(binary, forward, inverse, raw):
        wire = same(binary, forward, raw)
        if same(binary, inverse, wire) != raw:
            raise AssertionError(f"roundtrip failed: {binary} {forward}")
        return wire

    def words(values):
        return b"".join(struct.pack("<Q", value & ((1 << 64) - 1)) for value in values)

    special = [0, (1 << 63), (1 << 64) - 1, (1 << 63) - 1,
               0x7ff0000000000000, 0xfff0000000000000, 1,
               0x7ff8000000001234, 0x7ff8000000005678]
    fixtures = [b"", words(special), words(special[:3] * 11)]
    for size in [1,2,3,7,8,9,10,17,18,19,63,64,65,127,128,129,257]:
        fixtures.append(words([random_source.getrandbits(64) for _ in range(size)]))
    for raw in fixtures:
        for action in ["op.eraser.reference_v1", "op.match_token_format.reference_v1",
                       "op.bitstream_pack.reference_v1", "op.dictionary.reference_v1",
                       "op.symbol_recode.reference_v1"]:
            roundtrip("reference_operator_actions_cli", ["state", action], ["invert-state", action], raw)
        roundtrip("dictionary64_action_cli", ["encode"], ["decode"], raw)
        roundtrip("learning_codec_exact_stages_cli", ["alp-dictionary-forward"], ["alp-dictionary-inverse"], raw)
        roundtrip("spdp_exact_stages_cli", ["compose"], ["decompose", str(len(raw))], raw)
        delta = same("spdp_exact_stages_cli", ["delta-forward"], raw)
        roundtrip("spdp_exact_stages_cli", ["block-forward"], ["block-inverse"], delta)
        buff_scale = b"BSC1" + bytes([1,3]) + struct.pack("<I", len(raw)//8) + raw
        buff_split = roundtrip("learning_codec_exact_stages_cli", ["buff-column-split-forward"], ["buff-column-split-inverse"], buff_scale)
        roundtrip("learning_codec_exact_stages_cli", ["buff-bitpack-forward"], ["buff-bitpack-inverse"], buff_split)
        codes = same("legacy_numeric_action_stages_cli", ["forward", "op.scaling.legacy_exact_decimal.python_v1", "int64", "0"], raw)
        roundtrip("legacy_numeric_action_stages_cli", ["forward", "op.block_transform.int64_lo_hi32.python_v1"],
                  ["inverse", "op.block_transform.int64_lo_hi32.python_v1", str(len(raw)//8)], codes)

    for width in range(1,65):
        residuals = [random_source.getrandbits(width) for _ in range(8)]
        residuals[0] = 1 << (width-1)
        packet = b"SZI1" + struct.pack("<Q",9) + words([random_source.getrandbits(64),*residuals])
        roundtrip("p04_exact_stages_cli", ["sprintz-bitpack-forward"], ["sprintz-bitpack-inverse"], packet)
    for tail in range(1,9):
        raw = words([13,0,1,3,7,15,31,63,127] + list(range(tail)))
        for prefix, inverse in [(b"SZI1", "sprintz-bitpack-inverse"),
                                (b"SZF1", "sprintz-bitpack-inverse-double")]:
            packet = prefix + struct.pack("<Q", 9+tail) + raw
            roundtrip("p04_exact_stages_cli", ["sprintz-bitpack-forward"], [inverse], packet)
        packet = b"SZI1" + struct.pack("<Q",9+tail) + raw
        roundtrip("p04_exact_stages_cli", ["sprintz-bitpack-forward-current"], ["sprintz-bitpack-inverse-current"], packet)

    byte_fixtures = [b"", b"x", b"abc"*100, bytes(range(256))*3,
                     bytes(random_source.randrange(256) for _ in range(1027))]
    for raw in byte_fixtures:
        for width in [1,4,8]:
            roundtrip("general_codec_reference_stages_cli", [f"shuffle{width}-forward"], ["shuffle-inverse"], raw)
        for policy in ["blosclz-forward", "zstd-lz-forward"]:
            semantic = roundtrip("general_codec_reference_stages_cli", [policy], ["lz-inverse"], raw)
            roundtrip("general_codec_reference_stages_cli", ["parse-forward"], ["parse-inverse"], semantic)
        window = same("native_lz_action_stages_cli", ["forward", "op.match_window_model.native_lz.window65535"], raw)
        limits = same("native_lz_action_stages_cli", ["forward", "op.lz_match.native_lz.aligned8.min3.max258"], window)
        roundtrip("native_lz_action_stages_cli", ["forward", "op.match_parse_policy.native_lz.aligned8_greedy"],
                  ["inverse", "op.match_parse_policy.native_lz.aligned8_greedy"], limits)

    test_names = ["operator_consolidation_test", "learning_codec_exact_stages_test",
                  "p04_exact_stages_test", "spdp_exact_stages_test",
                  "general_codec_reference_stages_test", "openzl_layout_stages_test",
                  "native_legacy_action_stages_test"]
    command = ["ctest", "--test-dir", str(args.build), "--output-on-failure",
               "-R", "^(" + "|".join(test_names) + ")$"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    report = {
        "schema": "operator-consolidation-core-validation-v1",
        "date": "2026-09-17", "status": "PASS" if result.returncode == 0 else "FAIL",
        "scope": "Local C++ mechanism/profile/recipe consolidation; no benchmark or graph-registration claim.",
        "baseline_directory": str(args.baseline), "build_directory": str(args.build),
        "wire_comparisons": sum(counts.values()), "wire_commands": dict(sorted(counts.items())),
        "baseline_binary_sha256": {name: digest(args.baseline/name) for name in sorted(binaries)},
        "current_binary_sha256": {name: digest(args.build/name) for name in sorted(binaries)},
        "source_sha256": {"cpp/src/"+name: digest(root/"src"/name) for name in SOURCES},
        "ctest": {"command": command, "returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr},
        "coverage": ["A001/A051/A066 signed and unsigned bit-field profiles",
                     "A041/A042/A053 ordering profiles and A043 ID packing",
                     "A047 column order and cross-column uint8 Delta state",
                     "A022 all 64 block widths and integer/float tails",
                     "A027/A063 framing-only adapters; A057/A065 configuration adapters",
                     "A058/A059 retained lazy/greedy policy profiles"],
        "limits": ["No server benchmark or compression-time comparison.",
                   "Old action names and wire IDs remain compatibility entry points.",
                   "Unsigned fields need explicit signed views and packet carriers.",
                   "Graph search registration and all-data validation are separate."],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(result.stdout, end="")
    print(f"wire_comparisons={sum(counts.values())} status={report['status']} report={args.report}")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
