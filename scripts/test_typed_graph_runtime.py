"""Native multi-stream integration and cold-frame tests (no cached input)."""
import json
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest

import typed_graph_runtime as g


class GraphTests(unittest.TestCase):
    def check_roundtrip(self, source, action, parameters=None):
        state = g.GraphState.from_streams([source])
        result = state.apply(action, ["s0"], parameters=parameters)
        self.assertEqual(g.restore_frame(result.stop()), {"s0": source})
        return result

    def test_column_modes(self):
        source = g.Stream("struct", 4, bytes(range(40)))
        result = self.check_roundtrip(source, "columnize", {"width": 4})
        self.assertEqual([s.data for s in result.streams.values()],
                         [source.data[i::4] for i in range(4)])
        result = self.check_roundtrip(source, "columnize", {"mode": "split-struct", "fields": [1, 3]})
        self.assertEqual([s.width for s in result.streams.values()], [1, 3])

    def test_float_bit_patterns(self):
        for fmt, width, values in [
            ("fp32", 4, [0, 0x80000000, 0x7fc01234, 0x7f800000, 1, 0xffffffff]),
            ("fp16", 2, [0, 0x8000, 0x7e12, 0x7c00, 1, 0xffff]),
            ("bf16", 2, [0, 0x8000, 0x7fc2, 0x7f80, 1, 0xffff]),
        ]:
            with self.subTest(format=fmt):
                raw = b"".join(v.to_bytes(width, "little") for v in values)
                result = self.check_roundtrip(g.Stream("numeric", width, raw),
                    "float_deconstruct", {"format": fmt})
                self.assertEqual(sum(len(s.data) for s in result.streams.values()),
                                 len(values) * (3 if fmt == "fp16" else width))

    def test_bitsplit_modes(self):
        source = g.Stream("numeric", 4, struct.pack("<4I", 0, 1, 0xffffffff, 0x80000000))
        for p in ({"bits": [7, 17, 8]}, {"mode": "fp"}, {"mode": "top8"}):
            self.check_roundtrip(source, "bitsplit", p)
        self.check_roundtrip(g.Stream("numeric", 2, b"\x00\x80\xff\xff"),
                             "bitsplit", {"mode": "bf16"})
        with self.assertRaises(ValueError):
            self.check_roundtrip(source, "bitsplit", {"bits": [8]})

    def test_unpack_tail(self):
        for nbits in (1, 2, 3, 5, 7, 8):
            self.check_roundtrip(g.Stream("serial", 1, b"\xff"), "bitunpack", {"bits": nbits})
        with self.assertRaises(ValueError):
            self.check_roundtrip(g.Stream("serial", 1, b"\xff"), "bitunpack", {"bits": 9})

    def test_specials(self):
        for kind, width in (("serial", 1), ("struct", 3), ("numeric", 2)):
            streams = [g.Stream(kind, width, b""), g.Stream(kind, width, bytes(range(width * 4))),
                       g.Stream(kind, width, b"\xff" * width)]
            state = g.GraphState.from_streams(streams)
            joined = state.apply("concat", list(state.streams))
            self.assertEqual(list(g.restore_frame(joined.stop()).values()), streams)
            self.check_roundtrip(streams[1], "split_segments", {"sizes": [1, 0]})
        raw = struct.pack("<256I", *([1] * 128 + [100000] * 128))
        self.check_roundtrip(g.Stream("numeric", 4, raw), "split_segments", {"mode": "range"})

    def test_independent_leaf_codecs_and_cold_decode(self):
        source = g.Stream("numeric", 4, struct.pack("<256I", *range(256)))
        state = g.GraphState.from_streams([source]).apply("columnize", ["s0"])
        ids = list(state.streams)
        state = state.apply("op.entropy_huffman.project_bytes.cpp_v1", [ids[0]])
        state = state.apply("op.entropy_range.project_bytes.cpp_v1", [ids[1]])
        self.assertEqual(g.restore_frame(state.stop()), {"s0": source})
        # New interpreter has no encoder object or access to original data.
        script = "import sys; from typed_graph_runtime import restore_frame; sys.stdout.buffer.write(restore_frame(sys.stdin.buffer.read())['s0'].data)"
        proc = subprocess.run([sys.executable, "-c", script], input=state.stop(), capture_output=True,
                              cwd=Path(__file__).resolve().parent, check=True)
        self.assertEqual(proc.stdout, source.data)

    def test_nested_split_concat(self):
        source = g.Stream("struct", 2, bytes(range(64)))
        state = self.check_roundtrip(source, "columnize")
        ids = list(state.streams)
        state = state.apply("split_segments", [ids[0]], parameters={"sizes": [3, 0]})
        state = state.apply("concat", list(state.streams))
        self.assertEqual(g.restore_frame(state.stop()), {"s0": source})

    def test_a111_exception_three_live_data_flows(self):
        import exception_separation
        values = [1, 2, 3, 1025]
        source = g.Stream("numeric", 4, struct.pack("<4I", *values))
        state = g.GraphState.from_streams([source]).apply(
            exception_separation.ALGORITHM_NAME, ["s0"], parameters={"width": 2})
        self.assertEqual(len(state.streams), 3)
        self.assertEqual([len(s.data) for s in state.streams.values()], [16, 4, 4])
        self.assertEqual(g.restore_frame(state.stop()), {"s0": source})

    def test_a035_residual_window_pack(self):
        import residual_window_pack
        words = [0, 1, 0x100, 0x10000, 0xffffffffffffffff]
        source = g.Stream("numeric", 8, struct.pack("<5Q", *words))
        state = g.GraphState.from_streams([source]).apply(
            residual_window_pack.ELF_PLUS_ACTION, ["s0"])
        self.assertEqual(len(state.streams), 1)
        self.assertEqual(next(iter(state.streams.values())).kind, "serial")
        self.assertEqual(g.restore_frame(state.stop()), {"s0": source})

    def test_raw_reachability_for_pdf_aligned_downstream_mechanisms(self):
        words = g.Stream("numeric", 8, struct.pack("<16q", *([1, 2, 3, 4] * 4)))
        state = g.GraphState.from_streams([words]).apply(
            g.ADJACENT_XOR_PORT_ACTION, ["s0"], input_view="int64"
        )
        self.assertEqual([len(stream.data) for stream in state.streams.values()], [8, 120])
        residual_id = list(state.streams)[1]
        state = state.apply("op.trailing_zero_pack.elf_plus_residual.cpp_v1", [residual_id])
        self.assertEqual(g.restore_frame(state.stop()), {"s0": words})

        raw = g.Stream("serial", 1, b"ABCDEFGH" * 64)
        for actions in (
            ("op.match_window_model.native_lz.window4096",
             "op.lz_match.native_lz.aligned8.min3.max18",
             "op.match_parse_policy.native_lz.aligned8_greedy",
             "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3"),
            ("op.lz_match.blosclz.greedy.cpp_reference_v1",
             "op.match_token_format.blosclz.ctrl8.cpp_reference_v1"),
        ):
            candidate = g.GraphState.from_streams([raw])
            for action in actions:
                candidate = candidate.apply(action, [next(iter(candidate.streams))])
            self.assertEqual(g.restore_frame(candidate.stop()), {"s0": raw})

        state = g.GraphState.from_streams([words])
        for action in ("op.delta.spdp.cpp_v1", "op.block_transform.spdp.cpp_v1",
                       "op.entropy_special.spdp.cpp_v1"):
            state = state.apply(action, [next(iter(state.streams))])
        self.assertEqual(g.restore_frame(state.stop()), {"s0": words})

    def test_rejections(self):
        state = g.GraphState.from_streams([g.Stream("numeric", 2, b"\x00\x80")])
        for ids in (["missing"], ["s0", "s0"], []):
            with self.assertRaises(ValueError):
                state.apply("columnize", ids)
        with self.assertRaises(ValueError):
            state.apply("bitsplit", ["s0"], parameters={"width": 4, "bits": [32]})
        with self.assertRaises(TypeError):
            g.Stream("serial", 1, 3)
        for action, parameters in (
            ("columnize", {"fields": [65, 66]}),
            ("bitsplit", {"mode": "fp", "bits": [65, 66]}),
            ("split_segments", {"mode": "range", "sizes": [65, 66]}),
        ):
            with self.assertRaises(ValueError):
                state.apply(action, ["s0"], parameters=parameters)
        with self.assertRaises(ValueError):
            state.apply("op.entropy_huffman.project_bytes.cpp_v1", ["s0"], input_view="float64")
        frame = state.apply("columnize", ["s0"]).stop()
        for corrupt in (frame[:-1], frame + b"X", b"garbage"):
            with self.assertRaises(ValueError):
                g.restore_frame(corrupt)
        n = struct.unpack_from("<Q", frame, 4)[0]
        header = json.loads(frame[12:12 + n])
        header["steps"][0]["outputs"] *= 2
        h = json.dumps(header).encode()
        body = b"TGF1" + struct.pack("<Q", len(h)) + h + frame[12 + n:-32]
        with self.assertRaises(ValueError):
            g.restore_frame(body + hashlib.sha256(body).digest())

    def test_many_roots_keep_order_and_integrity(self):
        streams = [g.Stream("serial", 1, bytes([i])) for i in range(15)]
        state = g.GraphState.from_streams(streams).apply("concat", [f"s{i}" for i in range(15)])
        self.assertEqual(list(g.restore_frame(state.stop()).values()), streams)
        frame = bytearray(state.stop())
        frame[-33] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            g.restore_frame(frame)

    def test_versioned_unified_entrypoint(self):
        import operator_taxonomy_v30 as taxonomy
        import unified_action_runtime as uar
        source = g.Stream("numeric", 2, b"\x00\x80\xff\xff")
        state = g.GraphState.from_streams([source])
        result = uar.run_graph_action(state, 143, ["s0"],
            registry_version=taxonomy.REGISTRY_VERSION, parameters={"format": "bf16"})
        frame = uar.run_graph_action(result, taxonomy.STOP_ACTION_ID,
                                    registry_version=taxonomy.REGISTRY_VERSION)
        self.assertEqual(g.restore_frame(frame), {"s0": source})
        with self.assertRaises(uar.ActionRuntimeError):
            uar.run_graph_action(state, 143, registry_version=taxonomy.LEGACY_REGISTRY_VERSION)
        entropy_state = uar.run_graph_action(
            state, 146, ["s0"], registry_version=taxonomy.REGISTRY_VERSION)
        entropy_frame = uar.run_graph_action(
            entropy_state, taxonomy.STOP_ACTION_ID,
            registry_version=taxonomy.REGISTRY_VERSION)
        self.assertEqual(g.restore_frame(entropy_frame), {"s0": source})
        with self.assertRaises(uar.ActionNotExecutableError):
            uar.run_graph_action(state, 150, ["s0"], registry_version=taxonomy.REGISTRY_VERSION)
        registry = uar.graph_registry()
        self.assertEqual(registry["counts"]["operator_families"], 30)
        self.assertEqual(registry["counts"]["special_action_kinds"], 3)
        with uar.prediction_model(146, registry_version=taxonomy.REGISTRY_VERSION) as model:
            self.assertTrue(0 <= model.predict() <= 1)
            model.observe(1)

    def test_empty_and_raw_frame(self):
        for stream in (g.Stream("serial", 1, b""), g.Stream("struct", 3, b"")):
            state = g.GraphState.from_streams([stream])
            self.assertEqual(g.restore_frame(state.stop()), {"s0": stream})
            self.check_roundtrip(stream, "columnize")


if __name__ == "__main__":
    unittest.main()
