"""Current CLI wire parity and independent-process shared-library decoding."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_layout_bridge_v1 import NativeLayoutBridge, ROOT
from typed_graph_runtime import Stream, pack_bundle, unpack_bundle


class NativeLayoutBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = NativeLayoutBridge()
        cls.cli = ROOT / "cpp/build/openzl_layout_stages_cli"

    def cli_run(self, mode, data, *args):
        return subprocess.run([str(self.cli), mode, *map(str, args)], input=data,
                              capture_output=True, check=True).stdout

    def test_all_layout_profiles_wire_parity(self):
        float_bits = struct.pack("<8I", 0, 0x80000000, 0x7f800000, 0xff800000,
                                 0x7fc01234, 1, 0x00800000, 0x3f800000)
        raw16 = struct.pack("<6H", 0, 0x8000, 0x7e15, 0xfc00, 1, 0x3c00)
        cases = [
            (float_bits, ("transpose", 4)),
            (float_bits, ("split-struct", "1,3")),
            (float_bits, ("float-deconstruct", "fp32")),
            (raw16, ("float-deconstruct", "fp16")),
            (raw16, ("float-deconstruct", "bf16")),
            (float_bits, ("bitsplit", 4, "23,8,1")),
            (float_bits, ("bitsplit-top8", 4)),
            (float_bits, ("bitsplit-fp", 4)),
            (raw16, ("bitsplit-bf16",)),
            (b"\x91\xfe\xc5", ("bitunpack", 5)),
            (float_bits, ("split-segments", "numeric", 4, "2,0,6")),
            (struct.pack("<64I", *range(64)), ("split-by-range", 4, 2)),
            (pack_bundle([Stream("numeric", 4, float_bits[:8]),
                          Stream("numeric", 4, float_bits[8:])]), ("concat",)),
        ]
        for raw, args in cases:
            with self.subTest(args=args):
                encoded = self.bridge.run("encode", raw, *args)
                self.assertEqual(encoded, self.cli_run("encode", raw, *args))
                for mode in ("decode", "decode-bundle"):
                    self.assertEqual(self.bridge.run(mode, encoded),
                                     self.cli_run(mode, encoded))

    def test_distinct_children_reencode_without_process_spawn(self):
        raw = struct.pack("<8I", *range(8))
        with patch.object(subprocess, "run", side_effect=AssertionError("unexpected IPC")):
            frame = self.bridge.run("encode", raw, "split-segments", "numeric", 4, "3,5")
            children, meta = unpack_bundle(frame)
            first = self.bridge.run("encode", children[0].data, "bitsplit", 4, "16,16")
            second = self.bridge.run("encode", children[1].data, "transpose", 4)
            restored = [Stream(child.kind, child.width, self.bridge.run("decode", wire))
                        for child, wire in zip(children, (first, second))]
            self.assertEqual(self.bridge.run("decode", pack_bundle(restored, **meta)), raw)

    def test_independent_cold_decode(self):
        raw = bytes(range(120))
        frame = self.bridge.run("encode", raw, "transpose", 3)
        script = ("import sys;from native_layout_bridge_v1 import NativeLayoutBridge;"
                  "sys.stdout.buffer.write(NativeLayoutBridge().run('decode',sys.stdin.buffer.read()))")
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "scripts"),
                           NATIVE_LAYOUT_BRIDGE_V1=str(self.bridge.path))
        result = subprocess.run([sys.executable, "-c", script], input=frame,
                                capture_output=True, env=environment, check=True)
        self.assertEqual(result.stdout, raw)

    def test_empty_and_all_truncated_frames(self):
        self.assertEqual(self.bridge.run("decode", self.bridge.run("encode", b"", "transpose", 3)), b"")
        frame = self.bridge.run("encode", b"abcdef", "transpose", 3)
        for size in range(len(frame)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.bridge.run("decode", frame[:size])
        with self.assertRaises(ValueError):
            self.bridge.run("decode", frame + b"x")

    def test_rejects_invalid_arguments_without_shared_error_state(self):
        for args in (("encode", "bitsplit", 3, "12,12"),
                     ("encode", "bitsplit", 4, "64"),
                     ("encode", "float-deconstruct", "fp64"),
                     ("encode", "transpose", -1), ("encode", "transpose", 1 << 40),
                     ("encode", "bitunpack", 0), ("decode", "unused"),
                     ("encode", "split-by-range", 4, 0), ("encode", "unknown")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.bridge.run(args[0], bytes(12), *args[1:])
        with self.assertRaises(TypeError):
            self.bridge.run("encode", 10, "transpose", 1)
        with self.assertRaises(ValueError):
            self.bridge.run("encode", b"", "transpose\0", 1)
        self.assertEqual(self.bridge.run("decode", self.bridge.run("encode", b"abc", "transpose", 1)), b"abc")

    def test_concurrent_result_ownership(self):
        def run(index):
            raw = bytes([index]) * 30
            return self.bridge.run("decode", self.bridge.run("encode", raw, "transpose", 3)) == raw
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertTrue(all(pool.map(run, range(32))))

    def test_stage_timer_explicitly_not_graph_or_formal(self):
        for raw, args in ((bytes(range(96)), ("transpose", 3)),
                          (pack_bundle([Stream("serial", 1, b"abc"),
                                        Stream("serial", 1, b"def")]), ("concat",))):
            report = self.bridge.benchmark_stage(raw, *args, warmups=0, repeats=3)
            self.assertFalse(report["whole_graph_measurement"])
            self.assertFalse(report["formal_eligible"])
            self.assertEqual(len(report["compression_samples_ns"]), 3)
            self.assertEqual(len(report["decompression_samples_ns"]), 3)
            self.assertTrue(all(value >= 0 for value in report["compression_samples_ns"]))
            self.assertEqual(report["complete_ols1_bytes"], len(self.bridge.run("encode", raw, *args)))
            self.assertNotIn("compression_ratio", report)
        for warmups, repeats in ((-1, 3), (32, 3), (0, 0), (0, 1002), (True, 3)):
            with self.assertRaises(ValueError):
                self.bridge.benchmark_stage(b"abc", "transpose", 1, warmups=warmups, repeats=repeats)


if __name__ == "__main__":
    unittest.main()
