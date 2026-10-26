"""Lossless native graph integration, persistence and malformed frame tests."""
import hashlib
import json
from pathlib import Path
import pickle
import random
import struct
import subprocess
import sys
import unittest
from unittest import mock
import zlib

import exhaustive_graph_runtime_v1 as g


class ExhaustiveGraphRuntimeTests(unittest.TestCase):
    def source(self, raw=None, max_depth=10, view="int64", max_point=None):
        if raw is None:
            raw = struct.pack("<128q", *range(128))
        return g.GraphState.from_streams([g.Stream("numeric", 8, raw)],
            source_views=[view], max_depth=max_depth, max_point=max_point)

    def check(self, state, original):
        self.assertEqual(g.restore_frame(state.stop()), original.streams)

    def test_real_numeric_ports_then_independent_backends_and_cold_inverse(self):
        original = self.source()
        state = original.apply("op.xor.gorilla.cpp_v1", ["s0"], input_view="int64")
        ids = list(state.streams)
        self.assertEqual(len(ids), 2)
        self.assertEqual([s.kind for s in state.streams.values()], ["numeric", "numeric"])
        state = state.apply("op.entropy_huffman.project_bytes.cpp_v1", [ids[0]])
        state = state.apply("op.entropy_range.project_bytes.cpp_v1", [ids[1]])
        self.assertEqual(sorted(state.depths.values()), [2, 2])
        self.assertEqual(state.codec_count, 3)
        self.check(state, original)
        decoder = "import sys;from exhaustive_graph_runtime_v1 import restore_frame;sys.stdout.buffer.write(restore_frame(sys.stdin.buffer.read())['s0'].data)"
        result = subprocess.run([sys.executable, "-c", decoder], input=state.stop(),
            capture_output=True, cwd=Path(__file__).resolve().parent, check=True)
        self.assertEqual(result.stdout, original.streams["s0"].data)

    def test_concat_is_one_active_stream_and_preserves_explicit_order(self):
        original = self.source()
        state = original.apply("op.delta.stream_vbyte.cpp_v1", ["s0"], input_view="int64")
        ids = list(state.streams)
        self.assertEqual(len(ids), 2)
        state = state.apply("concat", list(reversed(ids)))
        self.assertEqual(len(state.streams), 1)
        self.assertEqual(list(state.depths.values()), [1])
        self.assertEqual(state.codec_count, 1)
        self.assertIn("concat_boundaries", state.steps[-1])
        self.check(state, original)

    def test_max_depth_is_per_path_not_total_nodes(self):
        original = self.source(max_depth=2)
        state = original.apply("columnize", ["s0"])
        self.assertEqual(len(state.streams), 8)
        for key in list(state.streams):
            state = state.apply("op.entropy_huffman.project_bytes.cpp_v1", [key])
        self.assertEqual(state.codec_count, 9)
        self.assertEqual(set(state.depths.values()), {2})
        with self.assertRaises(g.GraphInputError):
            state.apply("op.entropy_huffman.project_bytes.cpp_v1", [next(iter(state.streams))])
        joined = state.apply("concat", list(state.streams))
        self.assertEqual(list(joined.depths.values()), [2])
        self.check(joined, original)

    def test_minimum_for_ports_and_persistence(self):
        original = self.source()
        state = original.apply("op.frame_of_reference.minimum.cpp_v1", ["s0"], input_view="int64")
        self.assertEqual(len(state.streams), 2)
        self.assertEqual(state.steps[-1]["port_specs"][0]["name"], "minimum")
        saved = g.import_state(json.loads(json.dumps(g.export_state(state))))
        self.assertEqual(saved.stop(), state.stop())
        self.assertEqual(pickle.loads(pickle.dumps(state)).stop(), state.stop())
        self.check(saved, original)

    def test_max_point_is_forwarded_and_reconstructible(self):
        original = self.source(struct.pack("<64d", *([1.25, 2.5, -3.75, 0.0] * 16)),
                               view="float64", max_point=2)
        state = original.apply("op.scaling.ts_2diff.cpp_v1", ["s0"], input_view="float64")
        self.assertEqual(state.steps[-1]["max_point"], 2)
        self.check(state, original)

    def test_float_special_bit_patterns_are_preserved(self):
        words = [0, 0x8000000000000000, 0x7ff8000000001234,
                 0x7ff0000000000000, 0xfff0000000000000, 1]
        original = self.source(struct.pack("<6Q", *words), view="float64")
        state = original.apply("columnize", ["s0"])
        self.check(state, original)
        # Gorilla XOR works on the original binary64 words, including NaN payload.
        state = original.apply("op.xor.gorilla.cpp_v1", ["s0"], input_view="float64")
        self.check(state, original)

    def test_reference_exposes_native_ports_without_python_zlib(self):
        original = self.source()
        with mock.patch.object(zlib, "compress", side_effect=AssertionError("Python zlib encode called")), \
                mock.patch.object(zlib, "decompressobj", side_effect=AssertionError("Python zlib decode called")):
            state = original.apply("op.delta.reference_v1", ["s0"], input_view="int64")
            self.assertEqual(state.steps[-1]["kind"], "ports")
            self.assertEqual(state.steps[-1]["port_metadata"]["schema"], "reference.opa1")
            self.assertGreaterEqual(len(state.streams), 2)
            for i, key in enumerate(list(state.streams)):
                state = state.apply("op.entropy_huffman.project_bytes.cpp_v1" if i % 2 else
                                    "op.entropy_range.project_bytes.cpp_v1", [key])
            self.check(state, original)
        decoder = "import sys;from exhaustive_graph_runtime_v1 import restore_frame;sys.stdout.buffer.write(restore_frame(sys.stdin.buffer.read())['s0'].data)"
        result = subprocess.run([sys.executable, "-c", decoder], input=state.stop(),
            capture_output=True, cwd=Path(__file__).resolve().parent, check=True)
        self.assertEqual(result.stdout, original.streams["s0"].data)

    def test_native_carrier_matches_level6_and_rejects_bad_streams(self):
        rng = random.Random(291)
        for data in (b"", b"a" * 131072, rng.randbytes(65536), rng.randbytes(1024 * 1024)):
            encoded = g._carrier("encode", data)
            self.assertEqual(encoded, zlib.compress(data, 6))
            self.assertEqual(g._carrier("decode", encoded), data)
            for corrupted in (encoded[:-1], encoded + b"x"):
                with self.assertRaises(ValueError):
                    g._carrier("decode", corrupted)

    def test_repeated_operators_are_legal_and_multiple_root_order_survives(self):
        original = g.GraphState.from_streams([
            g.Stream("serial", 1, b"abc" * 64),
            g.Stream("serial", 1, b"def" * 64)], max_depth=2)
        state = original.apply("op.entropy_huffman.project_bytes.cpp_v1", ["s0"])
        state = state.apply("op.entropy_huffman.project_bytes.cpp_v1", ["s2"])
        state = state.apply("concat", ["s3", "s1"])
        self.assertEqual(list(g.restore_frame(state.stop())), ["s0", "s1"])
        self.check(state, original)

    def test_frozen_shuffle_profile_remains_executable(self):
        original = self.source()
        import canonical127_registry as c
        variant = c.GROUPS[59]["io_variants"][0]["name"]
        state = original.apply(59, ["s0"], io_variant=variant, output_mode="wire")
        self.assertEqual(state.steps[-1]["kind"], "wire")
        self.check(state, original)

    def test_inventory_includes_providers_but_does_not_claim_complete(self):
        inventory = g.registry_inventory()
        self.assertEqual(inventory["operator_count"], 135)
        self.assertEqual(inventory["codec_count"], 130)
        self.assertFalse(inventory["search_scope_complete"])
        self.assertEqual(sum(r["kind"] == "model_provider" for r in inventory["operators"]), 5)
        with self.assertRaises(g.UnsupportedActionError):
            self.source().apply(130, ["s0"])

    def test_reject_bad_inputs_and_damaged_or_duplicate_edges(self):
        original = self.source()
        with self.assertRaises(g.GraphInputError):
            original.apply("concat", ["s0"])
        state = original.apply("columnize", ["s0"])
        with self.assertRaises(g.GraphInputError):
            state.apply("concat", ["s1", "s1"])
        frame = state.stop()
        for corrupted in (frame[:-1], frame + b"x", b"bad"):
            with self.assertRaises(ValueError):
                g.restore_frame(corrupted)
        n = struct.unpack_from("<Q", frame, 4)[0]
        header = json.loads(frame[12:12 + n])
        header["steps"][0]["outputs"] *= 2
        metadata = json.dumps(header).encode()
        body = g.FRAME_MAGIC + struct.pack("<Q", len(metadata)) + metadata + frame[12 + n:-32]
        with self.assertRaises(ValueError):
            g.restore_frame(body + hashlib.sha256(body).digest())


if __name__ == "__main__":
    unittest.main()
