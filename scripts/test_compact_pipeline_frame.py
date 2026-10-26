"""Independent CCF1 transport tests; synthetic payloads do not invoke codecs.

The wire fixtures below spell out the format instead of relying on the tested
encoder to create malformed frames. These tests establish transport identity,
bounds and corruption detection, not losslessness of a particular codec.
"""
import json
import struct
import unittest
from unittest.mock import patch
import zlib

import compact_pipeline_frame as compact


LEGACY = struct.Struct('<4sBBQI')
PREFIX = struct.Struct('<4sIIQI')
RAW = b'12345678'


def legacy_frame(metadata=b'{}', payload=b'opaque codec payload', layers=1):
    header = LEGACY.pack(b'C206', 1, layers, len(RAW) // 8, zlib.crc32(RAW))
    if layers == 0:
        return header + RAW
    return header + b'UAR1' + struct.pack('<I', len(metadata)) + metadata + payload


def compact_fixture(metadata, payload=b'opaque codec payload', *, encoded=None,
                    json_size=None, compressed_size=None, payload_size=None,
                    header=None, checksum=None):
    original = legacy_frame(metadata, payload)
    if encoded is None:
        encoded = zlib.compress(metadata, 9)
    if header is None:
        header = original[:LEGACY.size]
    return PREFIX.pack(
        b'CCF1', len(metadata) if json_size is None else json_size,
        len(encoded) if compressed_size is None else compressed_size,
        len(payload) if payload_size is None else payload_size,
        zlib.crc32(original) if checksum is None else checksum,
    ) + header + encoded + payload


class CompactFrameTests(unittest.TestCase):
    def setUp(self):
        # Valid JSON with intentionally noncanonical whitespace, key ordering,
        # Unicode text and alternate number spellings. Exact bytes must survive.
        self.metadata = (
            '{ "z" : 1.00, "unicode" : "算子", "a" : -0, "notes" : "'
            + 'repeatable metadata field; ' * 100 + '" }'
        ).encode('utf-8')
        json.loads(self.metadata)
        self.payload = bytes(range(256)) * 4
        self.original = legacy_frame(self.metadata, self.payload)
        self.wire = compact_fixture(self.metadata, self.payload)

    def test_exact_legacy_frame_identity_and_unchanged_payload(self):
        result = compact.compact_frame(self.original)
        self.assertEqual(result[:4], b'CCF1')
        self.assertEqual(compact.expand_frame(result), self.original)
        magic, raw_len, stored_len, payload_len, checksum = PREFIX.unpack_from(result)
        self.assertEqual((magic, raw_len, payload_len, checksum),
                         (b'CCF1', len(self.metadata), len(self.payload),
                          zlib.crc32(self.original)))
        self.assertEqual(result[PREFIX.size:PREFIX.size + LEGACY.size],
                         self.original[:LEGACY.size])
        self.assertEqual(result[PREFIX.size + LEGACY.size + stored_len:], self.payload)
        self.assertLess(len(result), len(self.original))
        self.assertEqual(len(self.original) - len(result),
                         len(self.metadata) - stored_len - 16)

    def test_independent_wire_fixture_expands_exactly(self):
        self.assertEqual(compact.expand_frame(self.wire), self.original)

    def test_nonnormalized_json_bytes_are_not_reserialized(self):
        wire = compact.compact_frame(self.original)
        expanded = compact.expand_frame(wire)
        self.assertEqual(expanded[LEGACY.size + 8:
                                  LEGACY.size + 8 + len(self.metadata)], self.metadata)
        canonical = json.dumps(json.loads(self.metadata), sort_keys=True,
                               separators=(',', ':')).encode('utf-8')
        self.assertNotEqual(canonical, self.metadata)

    def test_tiny_metadata_falls_back_without_any_extra_byte(self):
        original = legacy_frame(b'{}', self.payload)
        self.assertGreater(len(compact_fixture(b'{}', self.payload)), len(original))
        self.assertEqual(compact.compact_frame(original), original)
        self.assertEqual(compact.expand_frame(original), original)

    def test_size_never_expands_for_varied_valid_metadata(self):
        for repetitions in (0, 1, 2, 5, 10, 20, 100, 1000):
            with self.subTest(repetitions=repetitions):
                metadata = json.dumps({'note': 'abc' * repetitions}).encode('utf-8')
                original = legacy_frame(metadata, self.payload)
                wire = compact.compact_frame(original)
                self.assertLessEqual(len(wire), len(original))
                self.assertEqual(compact.expand_frame(wire), original)

    def test_zero_layer_raw_frame_identity(self):
        original = legacy_frame(layers=0)
        self.assertEqual(compact.compact_frame(original), original)
        self.assertEqual(compact.expand_frame(original), original)

    def test_zero_layer_raw_wrong_length_and_crc_rejected(self):
        original = legacy_frame(layers=0)
        for bad in (original[:-1], original + b'x', original[:-1] + b'x'):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    compact.expand_frame(bad)
                with self.assertRaises(ValueError):
                    compact.compact_frame(bad)

    def test_unknown_magic_and_too_short_frames_rejected(self):
        for bad in (b'', b'CCF1', b'C206', b'CCF2' + self.wire[4:],
                    b'XXXX' + self.original[4:]):
            with self.subTest(length=len(bad), magic=bad[:4]):
                with self.assertRaises(ValueError):
                    compact.expand_frame(bad)

    def test_every_compact_truncation_rejected(self):
        for size in range(len(self.wire)):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    compact.expand_frame(self.wire[:size])

    def test_trailing_wire_data_rejected(self):
        with self.assertRaises(ValueError):
            compact.expand_frame(self.wire + b'trailing')

    def test_full_original_frame_crc_covers_payload_and_c206_header(self):
        for offset in (len(self.wire) - 1, PREFIX.size + 14, PREFIX.size + 8):
            with self.subTest(offset=offset):
                bad = bytearray(self.wire)
                bad[offset] ^= 1
                with self.assertRaises(ValueError):
                    compact.expand_frame(bytes(bad))

    def test_wrong_outer_crc_rejected(self):
        bad = compact_fixture(self.metadata, self.payload,
                              checksum=zlib.crc32(self.original) ^ 1)
        with self.assertRaises(ValueError):
            compact.expand_frame(bad)

    def test_corrupt_zlib_stream_rejected(self):
        encoded = bytearray(zlib.compress(self.metadata, 9))
        encoded[-1] ^= 1
        with self.assertRaises(ValueError):
            compact.expand_frame(compact_fixture(self.metadata, self.payload,
                                                 encoded=bytes(encoded)))

    def test_concatenated_zlib_stream_rejected_even_with_correct_declared_extent(self):
        encoded = zlib.compress(self.metadata, 9) + zlib.compress(b'another stream', 9)
        with self.assertRaises(ValueError):
            compact.expand_frame(compact_fixture(self.metadata, self.payload, encoded=encoded))

    def test_zlib_trailing_junk_rejected_even_with_correct_declared_extent(self):
        encoded = zlib.compress(self.metadata, 9) + b'not a stream'
        with self.assertRaises(ValueError):
            compact.expand_frame(compact_fixture(self.metadata, self.payload, encoded=encoded))

    def test_truncated_zlib_stream_with_correct_total_extent_rejected(self):
        encoded = zlib.compress(self.metadata, 9)[:-1]
        with self.assertRaises(ValueError):
            compact.expand_frame(compact_fixture(self.metadata, self.payload, encoded=encoded))

    def test_external_zlib_dictionary_is_not_accepted(self):
        compressor = zlib.compressobj(level=9, zdict=b'repeatable metadata field; ')
        encoded = compressor.compress(self.metadata) + compressor.flush()
        self.assertTrue(encoded[1] & 32)  # RFC1950 FDICT, explicit external dependency.
        with self.assertRaises(ValueError):
            compact.expand_frame(compact_fixture(self.metadata, self.payload, encoded=encoded))

    def test_declared_uncompressed_length_must_match_exactly(self):
        for length in (0, 1, len(self.metadata) - 1, len(self.metadata) + 1):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    compact.expand_frame(compact_fixture(self.metadata, self.payload,
                                                         json_size=length))

    def test_decompression_bomb_cannot_exceed_small_declared_length(self):
        encoded = zlib.compress(b'A' * (1024 * 1024), 9)
        with self.assertRaises(ValueError):
            compact.expand_frame(compact_fixture(self.metadata, self.payload,
                                                 encoded=encoded, json_size=32))

    def test_declared_metadata_limit_rejected_before_large_allocation(self):
        with self.assertRaises(ValueError):
            compact.expand_frame(compact_fixture(self.metadata, self.payload,
                                                 json_size=compact.MAX_METADATA_BYTES + 1))

    def test_declared_payload_and_compressed_extents_must_match_wire(self):
        encoded_size = len(zlib.compress(self.metadata, 9))
        for kwargs in ({'payload_size': len(self.payload) + 1},
                       {'payload_size': 2 ** 63}, {'compressed_size': 0},
                       {'compressed_size': encoded_size - 1},
                       {'compressed_size': encoded_size + 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    compact.expand_frame(compact_fixture(self.metadata, self.payload, **kwargs))

    def test_reconstructed_frame_limit_checked(self):
        # Use a reduced limit to test exact allocation policy with a tiny fixture.
        with patch.object(compact, 'MAX_FRAME_BYTES', len(self.original) - 1):
            with self.assertRaises(ValueError):
                compact.expand_frame(self.wire)

    def test_legacy_metadata_extent_and_magic_validated(self):
        metadata_start = LEGACY.size
        bad_magic = self.original[:metadata_start] + b'UAR2' + self.original[metadata_start + 4:]
        bad_length = (self.original[:metadata_start + 4]
                      + struct.pack('<I', len(self.original))
                      + self.original[metadata_start + 8:])
        for bad in (bad_magic, bad_length, self.original[:LEGACY.size + 7]):
            with self.subTest(length=len(bad), prefix=bad[:30]):
                with self.assertRaises(ValueError):
                    compact.compact_frame(bad)
                with self.assertRaises(ValueError):
                    compact.expand_frame(bad)

    def test_invalid_legacy_kind_or_depth_rejected(self):
        for kind, layers in ((0, 1), (3, 1), (1, 11), (1, 255)):
            bad_header = LEGACY.pack(b'C206', kind, layers, 1, zlib.crc32(RAW))
            bad = bad_header + self.original[LEGACY.size:]
            with self.subTest(kind=kind, layers=layers):
                with self.assertRaises(ValueError):
                    compact.compact_frame(bad)
                with self.assertRaises(ValueError):
                    compact.expand_frame(bad)


if __name__ == '__main__':
    unittest.main()
