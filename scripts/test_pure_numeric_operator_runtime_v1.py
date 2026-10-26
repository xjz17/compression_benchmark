#!/usr/bin/env python3
"""Small deterministic unit tests; no RL experiments or legacy CLI calls."""
import argparse
import hashlib
import json
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pure_numeric_operator_runtime_v1 as r


class PureNumericTests(unittest.TestCase):
    def test_mapping_and_full_domain_boundaries(self):
        values = [0, -1, 1, -2, 2, -(1 << 63), (1 << 63) - 1, -(1 << 62), 1 << 62]
        source = r.TypedData.ints(values)
        encoded = r.zigzag_forward(source)
        self.assertEqual(encoded.kind, "uint64")
        self.assertEqual(encoded.values(), tuple(2 * x if x >= 0 else -2 * x - 1 for x in values))
        self.assertEqual(len(encoded.payload), 8 * len(values))
        self.assertEqual(r.zigzag_inverse(encoded), source)

    def test_random_full_width_bitwise_roundtrip(self):
        rng = random.Random(20260906)
        values = [rng.randrange(-(1 << 63), 1 << 63) for _ in range(4096)]
        source = r.TypedData.ints(values)
        encoded = r.zigzag_forward(source)
        self.assertEqual(encoded.values(), tuple(2 * x if x >= 0 else -2 * x - 1 for x in values))
        self.assertEqual(r.zigzag_inverse(encoded).payload, source.payload)

    def test_independent_unsigned_inverse(self):
        unsigned = [0, 1, 2, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
        source = r.TypedData.uints(unsigned)
        recovered = r.zigzag_inverse(source)
        self.assertEqual(recovered.values(), tuple((u >> 1) ^ -(u & 1) for u in unsigned))
        self.assertEqual(r.zigzag_forward(recovered), source)

    def test_no_implicit_backend(self):
        source = r.TypedData.ints([0, -1, 1, -2, 2])
        original = r._run
        with patch.object(r, "_run", wraps=original) as call:
            result = r.zigzag_forward(source)
        self.assertEqual([x.args[0] for x in call.call_args_list], ["zigzag-forward"])
        self.assertEqual(len(result.payload), 40)

    def test_compact_widths_and_uint64_max(self):
        for maximum, width in [(0, 1), (255, 1), (256, 2), (65535, 2), (65536, 4),
                               ((1 << 32) - 1, 4), (1 << 32, 8), ((1 << 64) - 1, 8)]:
            with self.subTest(maximum=maximum):
                source = r.TypedData.uints([0, maximum])
                packed = r.run_action("compact_uint64", source)
                self.assertEqual(len(packed.payload), 13 + 2 * width)
                self.assertEqual(packed.payload[12], width)
                self.assertEqual(r.run_action("compact_uint64", packed, reverse=True), source)

    def test_empty(self):
        source = r.TypedData.ints([])
        result, _ = r.run_pipeline(source, ["zigzag", "compact_uint64", "zlib6"])
        restored, _ = r.run_pipeline(result, ["zigzag", "compact_uint64", "zlib6"], reverse=True)
        self.assertEqual(restored, source)

    def test_explicit_stages_and_frame(self):
        source = r.TypedData.ints([0, -1, 1, -2, 2] * 100)
        actions = ["zigzag", "compact_uint64", "zlib6"]
        result, stages = r.run_pipeline(source, actions)
        self.assertEqual([x["action"] for x in stages], actions)
        self.assertEqual(stages[0]["output_bytes"], len(source.payload))
        self.assertEqual(stages[1]["output_bytes"], 513)
        frame = r.frame_wrap(source, result, actions)
        self.assertGreater(len(frame), len(result.payload))
        self.assertEqual(r.decode_frame(frame), source)
        pure = r.zigzag_forward(source)
        self.assertEqual(r.decode_frame(r.frame_wrap(source, pure, ["zigzag"])), source)

    def test_types_lengths_and_bad_packets(self):
        with self.assertRaises(TypeError): r.zigzag_forward(r.TypedData.uints([1]))
        with self.assertRaises(TypeError): r.zigzag_inverse(r.TypedData.ints([1]))
        with self.assertRaises(TypeError): r.run_action("compact_uint64", r.TypedData.ints([1]))
        with self.assertRaises(TypeError): r.run_action("zlib6", r.TypedData.uints([1]))
        with self.assertRaises(ValueError): r.TypedData("int64", b"123", 0)
        with self.assertRaises(ValueError): r._run("zigzag-forward", b"123")
        for raw in [b"", b"UCV1" + struct.pack("<QB", 1, 0), b"UCV1" + struct.pack("<QB", 1, 3) + b"abc",
                    b"UCV1" + struct.pack("<QB", (1 << 64) - 1, 8)]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                r.run_action("compact_uint64", r.TypedData("bytes", raw, len(raw)), reverse=True)

    def test_zlib_trailing_and_frame_integrity(self):
        source = r.TypedData("bytes", b"hello", 5)
        encoded = r.run_action("zlib6", source)
        self.assertEqual(r.run_action("zlib6", encoded, reverse=True), source)
        raw = encoded.payload + b"trailing"
        with self.assertRaises(ValueError):
            r.run_action("zlib6", r.TypedData("bytes", raw, len(raw)), reverse=True)
        numeric = r.TypedData.ints([1, -1])
        pure = r.zigzag_forward(numeric)
        frame = bytearray(r.frame_wrap(numeric, pure, ["zigzag"]))
        frame[-1] ^= 1
        with self.assertRaises(ValueError): r.decode_frame(bytes(frame))

    def test_default_cli_is_pure_zigzag(self):
        source = r.TypedData.ints([0, -1, 1, -(1 << 63), (1 << 63) - 1])
        with tempfile.TemporaryDirectory(prefix="pure-zigzag-test-") as folder:
            original, mapped, restored = [Path(folder) / name for name in ("input.i64", "mapped.u64", "restored.i64")]
            original.write_bytes(source.payload)
            script = str(Path(r.__file__))
            subprocess.run([sys.executable, script, "transform", "--input", str(original), "--output", str(mapped)], check=True)
            self.assertEqual(mapped.stat().st_size, original.stat().st_size)
            self.assertEqual(r.TypedData("uint64", mapped.read_bytes(), source.count).values(),
                             (0, 1, 2, (1 << 64) - 1, (1 << 64) - 2))
            subprocess.run([sys.executable, script, "transform", "--reverse", "--input", str(mapped), "--output", str(restored)], check=True)
            self.assertEqual(restored.read_bytes(), source.payload)

    def test_legacy_frozen_sources_and_binaries_untouched(self):
        manifest = json.loads((r.ROOT / "analysis_outputs/canonical206_beam29_ppo_grpo_smallest2_20260906_v1/manifest.json").read_text())
        for relative, expected in manifest["source_hashes"].items():
            self.assertEqual(hashlib.sha256((r.ROOT / relative).read_bytes()).hexdigest(), expected, relative)
        for name, expected in manifest["runtime_cli_hashes"].items():
            self.assertEqual(hashlib.sha256((r.ROOT / "cpp/build" / name).read_bytes()).hexdigest(), expected, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("report exists; choose a new path")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PureNumericTests))
    if args.report:
        files = [Path(r.__file__), Path(__file__), r.SOURCE, r.CLI, r.ROOT / "docs/pure_numeric_operator_v1.md"]
        report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "created": datetime.now().astimezone().isoformat(),
                  "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "random_full_int64_domain_cases": 4096, "legacy_python_hashes_verified": 8,
                  "legacy_cli_hashes_verified": 37, "registry_version": r.REGISTRY_VERSION,
                  "rl_rerun": False, "legacy_search_migrated": False,
                  "files": {str(p.relative_to(r.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
