"""FOR leaves are independently executable data, including in cold graph frames."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest

import minimum_for as f
import typed_graph_runtime as g


def frame_parts(frame):
    size = struct.unpack_from('<Q', frame, 4)[0]
    return json.loads(frame[12:12 + size]), frame[12 + size:-32]


def edited_frame(frame, mutate):
    header, leaves = frame_parts(frame)
    mutate(header)
    raw = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    body = b'TGF1' + struct.pack('<Q', len(raw)) + raw + leaves
    return body + hashlib.sha256(body).digest()


class MinimumFORGraphTests(unittest.TestCase):
    def transform(self, values, signed=True):
        raw = struct.pack('<' + ('q' if signed else 'Q') * len(values), *values)
        source = g.Stream('numeric', 8, raw)
        state = g.GraphState.from_streams([source]).apply(
            f.ALGORITHM_NAME, ['s0'], input_view='int64' if signed else 'bytes')
        self.assertEqual(len(state.streams), 2)
        minimum, offsets = state.streams.values()
        self.assertEqual((minimum.kind, minimum.width, len(minimum.data)), ('numeric', 8, 8))
        self.assertEqual(struct.unpack('<q' if signed else '<Q', minimum.data)[0],
                         min(values, default=0))
        self.assertEqual(list(struct.unpack('<' + 'Q' * len(values), offsets.data)),
                         [v - min(values) for v in values])
        self.assertEqual(g.restore_frame(state.stop()), {'s0': source})
        return source, state

    def test_values_and_full_domains_have_two_real_leaves(self):
        for values, signed in [([], True), ([0, 0], True), ([-10, -7, 3], True),
                               ([-2**63, 2**63 - 1], True),
                               ([2**63, 2**64 - 1, 0], False), ([], False)]:
            with self.subTest(values=values, signed=signed):
                self.transform(values, signed)

    def test_different_leaf_codecs_and_cold_inverse(self):
        values = [-1000 + i % 11 for i in range(300)]
        source, state = self.transform(values)
        minimum_id, offsets_id = list(state.streams)
        state = state.apply('op.entropy_huffman.project_bytes.cpp_v1', [minimum_id])
        state = state.apply('op.bitpack_int.max_value_width.cpp_v1', [offsets_id])
        packed_id = list(state.streams)[-1]
        state = state.apply('op.entropy_range.project_bytes.cpp_v1', [packed_id])
        frame = state.stop()
        header, _ = frame_parts(frame)
        # The first edge retains shape/header metadata; the values reside only
        # in the two leaves (which now have distinct downstream codecs).
        edge = header['steps'][0]
        self.assertEqual(edge['kind'], 'minimum_for')
        self.assertEqual([p['name'] for p in edge['port_specs']], ['minimum', 'offsets'])
        self.assertNotIn('payload', edge)
        self.assertNotIn(source.data.hex(), json.dumps(edge))
        script = "import sys; from typed_graph_runtime import restore_frame; sys.stdout.buffer.write(restore_frame(sys.stdin.buffer.read())['s0'].data)"
        result = subprocess.run([sys.executable, '-c', script], input=frame,
            capture_output=True, cwd=Path(__file__).parent, check=True)
        self.assertEqual(result.stdout, source.data)
        self.assertEqual(g.restore_frame(frame), {'s0': source})

    def test_missing_duplicate_wrong_domain_and_v1_rejected(self):
        _, state = self.transform([-10, 1, 2])
        frame = state.stop()
        mutations = [
            lambda h: h['steps'][0]['outputs'].pop(),
            lambda h: h['steps'][0]['outputs'].__setitem__(1, h['steps'][0]['outputs'][0]),
            lambda h: h['steps'][0]['port_specs'][1].__setitem__('kind', 'int64'),
            lambda h: h['steps'][0]['port_specs'][0].__setitem__('count', 2),
            lambda h: h['steps'][0].__setitem__('action', 'op.delta.reference_v1'),
            lambda h: h.__setitem__('registry_version', 'operator-taxonomy-30-v1'),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i), self.assertRaises(ValueError):
                g.restore_frame(edited_frame(frame, mutate))
        for corrupt in (frame[:-1], frame + b'X'):
            with self.assertRaises(ValueError):
                g.restore_frame(corrupt)

    def test_v1_legacy_for_boundary_still_restores(self):
        source = g.Stream('numeric', 8, struct.pack('<3q', -3, 4, 20))
        state = g.GraphState.from_streams([source], registry_version='operator-taxonomy-30-v1')
        state = state.apply('op.frame_of_reference.reference_v1', ['s0'], input_view='int64')
        self.assertEqual(len(state.streams), 1)
        self.assertEqual(state.steps[0]['kind'], 'legacy')
        self.assertEqual(frame_parts(state.stop())[0]['registry_version'], 'operator-taxonomy-30-v1')
        self.assertEqual(g.restore_frame(state.stop()), {'s0': source})
        with self.assertRaisesRegex(ValueError, 'registry v2'):
            g.GraphState.from_streams([source], registry_version='operator-taxonomy-30-v1').apply(
                f.ALGORITHM_NAME, ['s0'], input_view='int64')
        with self.assertRaisesRegex(ValueError, '64-bit'):
            g.GraphState.from_streams([g.Stream('numeric', 4, struct.pack('<2I', 1, 2))]).apply(
                f.ALGORITHM_NAME, ['s0'])


if __name__ == '__main__':
    unittest.main()
