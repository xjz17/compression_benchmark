"""Native A111 process/port contract, including independent cold decoding."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exception_separation as es


def words(values):
    return struct.pack(f"<{len(values)}I", *values)


class ExceptionSeparationTests(unittest.TestCase):
    def test_known_semantic_streams(self):
        raw = words([0, 7, 8, 255, 0xffffffff])
        packet = es.forward(raw, 3)
        ports = es.to_ports(packet)
        self.assertEqual((ports.width, ports.count), (3, 5))
        self.assertEqual(ports.low, words([0, 7, 0, 7, 7]))
        self.assertEqual(ports.positions, words([2, 3, 4]))
        self.assertEqual(ports.high, words([1, 31, 536870911]))
        self.assertEqual(es.from_ports(ports), packet)
        self.assertEqual(es.inverse(packet), raw)

    def test_cold_decode_and_a020_recipe(self):
        raw = words([0, 1, 7, 8, 255, 65535, 0xffffffff])
        for width in (0, 1, 8, 16, 31, 32):
            with self.subTest(width=width):
                packet = es.forward(raw, width)
                packed = es.pack(packet)
                self.assertEqual(es.unpack(packed), packet)
                decoded = subprocess.run([str(es.BINARY), "decompose"],
                                         input=packed, capture_output=True, check=True)
                self.assertEqual(decoded.stdout, raw)
                self.assertEqual(es.run("compose", raw, width), packed)

    def test_boundaries_and_empty(self):
        for width in (0, 32):
            self.assertEqual(es.inverse(es.forward(b"", width)), b"")
            self.assertEqual(es.run("decompose", es.run("compose", b"", width)), b"")
        ports = es.to_ports(es.forward(words([0, 0xffffffff]), 32))
        self.assertEqual(ports.positions, b"")
        self.assertEqual(ports.high, b"")
        zero = es.to_ports(es.forward(words([0, 0xffffffff]), 0))
        self.assertEqual(zero.low, words([0, 0]))
        self.assertEqual(zero.high, words([0xffffffff]))

    def test_independent_stream_reencoding(self):
        raw = words([9, 1, 1000, 0xffffffff, 9])
        ports = es.to_ports(es.forward(raw, 4))
        packed_low = es.forward(ports.low, 2)
        packed_positions = es.forward(ports.positions, 1)
        packed_high = es.forward(ports.high, 16)
        restored = replace(ports, low=es.inverse(packed_low),
                           positions=es.inverse(packed_positions), high=es.inverse(packed_high))
        self.assertEqual(es.inverse(es.from_ports(restored)), raw)

    def test_missing_duplicate_unsorted_and_overflow_streams(self):
        ports = es.to_ports(es.forward(words([8, 16, 32]), 3))
        bad_ports = [replace(ports, high=ports.high[:-4]),
                     replace(ports, low=ports.low[:-4]),
                     replace(ports, positions=words([0, 0, 2])),
                     replace(ports, positions=words([1, 0, 2])),
                     replace(ports, positions=words([0, 1, 3])),
                     replace(ports, high=words([0, 2, 4])),
                     replace(ports, high=words([1 << 29, 2, 4])),
                     replace(ports, low=words([8, 0, 0]))]
        for invalid in bad_ports:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                es.from_ports(invalid)

    def test_bad_packets_and_widths(self):
        wire = es.forward(words([1, 6]), 2)
        packed = es.pack(wire)
        for packet, decode in ((wire, es.inverse), (packed, es.unpack)):
            for length in range(len(packet)):
                with self.subTest(length=length), self.assertRaises(ValueError):
                    decode(packet[:length])
            with self.assertRaises(ValueError):
                decode(packet + b"\0")
        for width in (-1, 33, 1.5, True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                es.forward(b"", width)
        with self.assertRaises(ValueError):
            es.forward(b"\1", 3)
        with self.assertRaises(ValueError):
            es.inverse(b"PFS1" + bytes(64))


if __name__ == "__main__":
    unittest.main()
