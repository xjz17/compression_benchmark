#!/usr/bin/env python3
"""Cold, bit-exact typed-graph tests for the complete A110 NNP1 path."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

import nncp_entropy_codec
import operator_taxonomy_v30_v7 as taxonomy
import typed_graph_runtime as graph
import unified_action_runtime as runtime


class NNCPEntropyGraphTests(unittest.TestCase):
    def test_direct_graph_and_fresh_interpreter_restore(self):
        for source in (
            graph.Stream("serial", 1, b""),
            graph.Stream("serial", 1, b"ABA\x00\xff"),
            graph.Stream("numeric", 4, bytes(range(64))),
        ):
            with self.subTest(source=source.spec()):
                state = graph.GraphState.from_streams([source]).apply(
                    nncp_entropy_codec.ACTION, ["s0"]
                )
                packet = next(iter(state.streams.values())).data
                self.assertEqual(packet[:4], b"NNP1")
                frame = state.stop()
                self.assertEqual(graph.restore_frame(frame), {"s0": source})
                script = (
                    "import sys; from typed_graph_runtime import restore_frame; "
                    "sys.stdout.buffer.write(restore_frame(sys.stdin.buffer.read())['s0'].data)"
                )
                restored = subprocess.run(
                    [sys.executable, "-c", script], input=frame, capture_output=True,
                    cwd=Path(__file__).resolve().parent, check=True, timeout=180,
                )
                self.assertEqual(restored.stdout, source.data)

    def test_versioned_action_keeps_a110_identity(self):
        source = graph.Stream("serial", 1, b"identity")
        state = graph.GraphState.from_streams([source])
        encoded = runtime.run_graph_action(
            state, 134, ["s0"], registry_version=taxonomy.REGISTRY_VERSION
        )
        frame = encoded.stop()
        header_size = int.from_bytes(frame[4:12], "little")
        header = json.loads(frame[12:12 + header_size])
        self.assertEqual(header["steps"][0]["action"], nncp_entropy_codec.ACTION)
        self.assertEqual(header["steps"][0]["kind"], "nncp_entropy")
        self.assertEqual(header["steps"][0]["provider"], "nncp")
        self.assertEqual(header["steps"][0]["profile"], "transformer-stream1-v1")
        self.assertEqual(graph.restore_frame(frame), {"s0": source})

    def test_wrong_identity_and_corruption_are_rejected(self):
        packet = nncp_entropy_codec.forward(b"abc")
        with self.assertRaises(ValueError):
            nncp_entropy_codec.forward(b"abc", "op.prediction_probability.paq.provider_v1")
        with self.assertRaises(ValueError):
            nncp_entropy_codec.inverse(packet, expected_action="wrong")
        for invalid in (packet[:-1], packet + b"x", b"PEC1" + packet[4:]):
            with self.assertRaises(ValueError):
                nncp_entropy_codec.inverse(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)

