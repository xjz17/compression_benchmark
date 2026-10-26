#!/usr/bin/env python3
"""ACF1 integration tests; all tests use synthetic byte-exact source values."""
import base64
import hashlib
import json
import random
import struct
import unittest
from unittest import mock
import zlib

import atomic_composition_frame_v1 as frame
import atomic_transform_runtime_v1 as transforms


def integers(values):
    return struct.pack('<' + 'q' * len(values), *values)


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.raw = integers([0, 0, -1, -1, 2, 2, 2, 3, 100, 0] * 32)

    def test_metadata_generic_roundtrip_and_canonical(self):
        palette = base64.b64encode(integers([7, 8, 9])).decode('ascii')
        value = {'b': [-17, 0, (1 << 64) - 1, None, True, False, 0.125, b'raw'],
                 'palette_b64': palette, 'op': 'dictionary', 'non_symbol': 'hello 中文'}
        encoded = frame.serialize_metadata(value)
        self.assertEqual(frame.deserialize_metadata(encoded), value)
        self.assertEqual(encoded, frame.serialize_metadata(dict(reversed(list(value.items())))))
        self.assertIn(integers([7, 8, 9]), encoded)
        self.assertNotIn(palette.encode(), encoded)

    def test_metadata_smaller_than_json(self):
        _, metadata = transforms.apply_ops(self.raw, [{'op': 'zigzag'}, {'op': 'bitpack'}])
        self.assertLess(len(frame.serialize_metadata(metadata)), len(json.dumps(metadata).encode()) // 2)

    def test_empty_specs_no_implicit_codec(self):
        encoded = frame.encode_frame(self.raw, 'int64', [], [])
        info = frame.inspect_frame(encoded)
        self.assertEqual(frame.decode_frame(encoded), self.raw)
        self.assertEqual(info['terminals'], [])
        self.assertEqual(info['metadata'], [])
        self.assertEqual(info['payload_bytes'], len(self.raw))
        self.assertEqual(info['encoded_body_bytes'], info['uncompressed_body_bytes'])
        self.assertEqual(encoded[info['header_bytes']:], frame.make_body(self.raw, []))
        self.assertEqual(len(encoded), info['header_bytes'] + info['encoded_body_bytes'])

    def test_all_explicit_single_terminals_and_empty_source(self):
        for raw in (b'', self.raw):
            for terminal in frame.TERMINALS:
                with self.subTest(terminal=terminal, rawbytes=len(raw)):
                    encoded = frame.encode_frame(raw, 'int64', [], [terminal])
                    self.assertEqual(frame.decode_frame(encoded), raw)
                    info = frame.inspect_frame(encoded)
                    self.assertEqual(info['terminals'], [terminal])
                    self.assertEqual(info['terminal_ids'], [frame.TERMINALS[terminal]])
                    self.assertEqual(info['raw_sha256'], hashlib.sha256(raw).hexdigest())

    def test_two_complete_algorithms(self):
        for terminals in (['zlib6', 'xz'], ['bzip2', 'zstd'], ['brotli', 'zlib6']):
            encoded = frame.encode_frame(self.raw, 'int64', [], terminals)
            self.assertEqual(frame.decode_frame(encoded), self.raw)
            self.assertEqual(frame.inspect_frame(encoded)['comparison_group'], 'complete_algorithms_only')

    def test_pure_zigzag_preserves_payload_size(self):
        encoded = frame.encode_frame(self.raw, 'int64', [{'op': 'zigzag'}], [])
        self.assertEqual(frame.decode_frame(encoded), self.raw)
        info = frame.inspect_frame(encoded)
        self.assertEqual(info['payload_bytes'], len(self.raw))
        self.assertGreater(len(encoded), len(self.raw))

    def test_transform_and_terminal_compositions(self):
        chains = [
            [{'op': 'zigzag'}, {'op': 'dictionary'}, {'op': 'bitpack'}],
            [{'op': 'delta', 'lag': 2, 'order': 2}, {'op': 'zigzag'}, {'op': 'compact'}, {'op': 'shuffle', 'width': 8}],
            [{'op': 'xor', 'lag': 1}, {'op': 'rle'}, {'op': 'sparsezero'}, {'op': 'bittranspose', 'width': 8}],
            [{'op': 'for'}, {'op': 'gcd'}, {'op': 'compact'}],
        ]
        for specs in chains:
            for terminals in ([], ['range0'], ['sparse1'], ['zlib6'], ['xz', 'zstd']):
                with self.subTest(specs=specs, terminals=terminals):
                    encoded = frame.encode_frame(self.raw, 'int64', specs, terminals)
                    self.assertEqual(frame.decode_frame(encoded), self.raw)
                    info = frame.inspect_frame(encoded)
                    self.assertEqual(info['registered_depth'], len(specs) + len(terminals))
                    self.assertFalse(info['metadata_compressed_separately'])

    def test_bitpattern_float_and_exact_float_scale(self):
        # Arbitrary float64 bit patterns, including NaNs and signed zero.
        raw = struct.pack('<6Q', 0, 1 << 63, 0x7ff0000000000001, 0xfff0000000000000, 1, (1 << 64) - 1)
        encoded = frame.encode_frame(raw, 'float64', [{'op': 'xor'}, {'op': 'shuffle'}], ['range1'])
        self.assertEqual(frame.decode_frame(encoded), raw)
        values = struct.pack('<4d', -1.5, 0.0, 1.25, 12.75)
        specs = [{'op': 'float_scale', 'factor': 100}, {'op': 'zigzag'}, {'op': 'bitpack'}]
        self.assertEqual(frame.decode_frame(frame.encode_frame(values, 'float64', specs, ['zstd'])), values)

    def test_full_int64_random_and_boundaries(self):
        rng = random.Random(20260906)
        values = [-(1 << 63), -1, 0, 1, (1 << 63) - 1] + [rng.randrange(-(1 << 63), 1 << 63) for _ in range(128)]
        raw = integers(values)
        for op in ('zigzag', 'delta', 'xor', 'for', 'gcd'):
            self.assertEqual(frame.decode_frame(frame.encode_frame(raw, 'int64', [{'op': op}], ['zlib6'])), raw)

    def test_prepared_identical_to_cold_and_no_transform_call(self):
        specs = [{'op': 'zigzag'}, {'op': 'bitpack', 'bits': 'auto'}]
        payload, metadata = transforms.apply_ops(self.raw, specs)
        cold = frame.encode_frame(self.raw, 'int64', specs, ['range0'])
        with mock.patch.object(transforms, 'apply_ops', side_effect=AssertionError('unexpected recompute')):
            prepared = frame.encode_prepared(self.raw, 'int64', specs, ['range0'], payload, metadata)
        self.assertEqual(prepared, cold)
        self.assertEqual(frame.decode_frame(prepared), self.raw)

    def test_complete_body_not_metadata_separate_compression(self):
        specs = [{'op': 'dictionary'}, {'op': 'bitpack'}]
        payload, metadata = transforms.apply_ops(self.raw, specs)
        encoded = frame.encode_prepared(self.raw, 'int64', specs, ['zlib6'], payload, metadata)
        info = frame.inspect_frame(encoded)
        terminal_output = encoded[info['header_bytes']:]
        self.assertEqual(terminal_output, zlib.compress(frame.make_body(payload, metadata), 6))
        self.assertEqual(info['uncompressed_body_bytes'], info['metadata_length_prefix_bytes'] + info['metadata_bytes'] + info['payload_bytes'])

    def test_prepared_spec_mismatch_rejected(self):
        payload, metadata = transforms.apply_ops(self.raw, [{'op': 'delta', 'lag': 2}])
        for specs in ([{'op': 'delta'}], [{'op': 'xor', 'lag': 2}], []):
            with self.assertRaises(ValueError):
                frame.encode_prepared(self.raw, 'int64', specs, [], payload, metadata)
        payload, metadata = transforms.apply_ops(self.raw, [{'op': 'dictionary'}])
        with self.assertRaises(ValueError):
            frame.encode_prepared(self.raw, 'int64', [{'op': 'dictionary', 'max_entries': 1}], [], payload, metadata)

    def test_malformed_metadata_and_raw_rejected(self):
        for metadata in (b'', b'\xff', b'\x08\x80\x00', b'\x08\x00extra'):
            with self.assertRaises(ValueError):
                frame.deserialize_metadata(metadata)
        for raw, kind in ((b'123', 'int64'), (self.raw, 'bytes'), (bytearray(self.raw), 'int64')):
            with self.assertRaises(ValueError):
                frame.encode_frame(raw, kind, [], [])
        payload, metadata = transforms.apply_ops(self.raw, [{'op': 'zigzag'}])
        metadata[0]['in_nbytes'] += 8
        with self.assertRaises(ValueError):
            frame.encode_prepared(self.raw, 'int64', [{'op': 'zigzag'}], [], payload, metadata)

    def test_depth_and_terminal_guards(self):
        for specs, terminals in (([{'op': 'zigzag'}] * 11, []), ([{'op': 'zigzag'}] * 10, ['zlib6']),
                                 ([], ['zlib6'] * 3), ([], ['range0', 'zlib6']), ([], ['unknown'])):
            with self.assertRaises(ValueError):
                frame.encode_frame(self.raw, 'int64', specs, terminals)
        encoded = frame.encode_frame(self.raw, 'int64', [{'op': 'zigzag'}] * 9, ['zlib6'])
        self.assertEqual(frame.decode_frame(encoded), self.raw)

    def test_bitflip_truncation_and_trailing_rejected(self):
        encoded = frame.encode_frame(self.raw, 'int64', [{'op': 'zigzag'}, {'op': 'bitpack'}], ['zlib6'])
        for index in range(len(encoded)):
            damaged = bytearray(encoded)
            damaged[index] ^= 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                frame.decode_frame(bytes(damaged))
        for size in range(len(encoded)):
            with self.assertRaises(ValueError):
                frame.decode_frame(encoded[:size])
        with self.assertRaises(ValueError):
            frame.decode_frame(encoded + b'extra')

    def test_prepared_wrong_source_caught_by_decode_sha(self):
        encoded = frame.encode_prepared(self.raw, 'int64', [], [], bytes(len(self.raw)), [])
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            frame.decode_frame(encoded)

    def test_limits_before_codec_call(self):
        with mock.patch.object(frame, 'MAX_RAW_BYTES', 8), mock.patch.object(frame, '_compress', side_effect=AssertionError('must not execute')):
            with self.assertRaises(ValueError):
                frame.encode_frame(self.raw, 'int64', [], ['zlib6'])
        with mock.patch.object(frame, 'MAX_BODY_BYTES', 8):
            with self.assertRaises(ValueError):
                frame.make_body(self.raw, [])

    def test_unknown_version_and_codec_id(self):
        encoded = bytearray(frame.encode_frame(b'', 'int64', [], ['zlib6']))
        encoded[3] = ord('2')
        with self.assertRaisesRegex(ValueError, 'version'):
            frame.decode_frame(bytes(encoded))
        encoded[3] = ord('1')
        # Empty raw length occupies one byte, then SHA32, terminal count1.
        encoded[4 + 1 + 1 + 32 + 1] = 255
        with self.assertRaisesRegex(ValueError, 'terminal ID'):
            frame.decode_frame(bytes(encoded))


if __name__ == '__main__':
    unittest.main(verbosity=2)
