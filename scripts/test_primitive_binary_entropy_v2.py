#!/usr/bin/env python3
"""Round-trip, canonical-format, corruption and timing tests for pure PBV2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import struct
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cpp/src/primitive_binary_entropy_v2.cpp"
BINARY = ROOT / "cpp/build/primitive_binary_entropy_v2/primitive_binary_entropy_v2"
MODES = (4, 8, 12, 16)


def build():
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    temporary = BINARY.with_name(BINARY.name + ".next")
    subprocess.run(["clang++", "-std=c++20", "-O3", "-Wall", "-Wextra", "-pedantic",
                    str(SOURCE), "-o", str(temporary)], check=True)
    os.replace(temporary, BINARY)


def invoke(command, data, success=True):
    result = subprocess.run([str(BINARY), command], input=data, capture_output=True, timeout=30)
    if success and result.returncode != 0:
        raise AssertionError(result.stderr.decode())
    if not success and result.returncode == 0:
        raise AssertionError("corrupt or noncanonical input accepted")
    return result.stdout


def fnv64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def repaired_frame(original, body):
    """Change the body with VALID length/hash while keeping correct raw hash."""
    header = bytearray(original[:40])
    struct.pack_into("<Q", header, 16, len(body))
    struct.pack_into("<Q", header, 32, fnv64(body))
    return bytes(header) + body


class PrimitiveBinaryEntropyTest(unittest.TestCase):
    def check_roundtrip(self, raw):
        for bits in MODES:
            command = f"bit{bits}"
            encoded = invoke(command + "-encode", raw)
            self.assertEqual(invoke(command + "-decode", encoded), raw)
            self.assertEqual(encoded[:4], b"PBV2")
            self.assertEqual(encoded[4], bits)
            self.assertEqual(encoded[5:8], b"\x00\x00\x00")
            self.assertEqual(struct.unpack_from("<Q", encoded, 8)[0], len(raw))
            self.assertEqual(struct.unpack_from("<Q", encoded, 16)[0] + 40, len(encoded))
            self.assertEqual(struct.unpack_from("<Q", encoded, 24)[0], fnv64(raw))
            self.assertEqual(struct.unpack_from("<Q", encoded, 32)[0], fnv64(encoded[40:]))
            self.assertEqual(encoded, invoke(command + "-encode", raw))

    def test_empty_short_and_all_byte_values(self):
        for raw in (b"", b"\x00", b"\xff", b"ab", b"abc", bytes(range(256))):
            self.check_roundtrip(raw)
        for bits in MODES:
            self.assertEqual(len(invoke(f"bit{bits}-encode", b"")), 41)

    def test_every_previous_16bit_context(self):
        self.check_roundtrip(bytes(x for a in range(256) for b in range(256) for x in (a, b)))

    def test_rescale_and_repetition(self):
        for raw in (b"\x00" * 200000, b"\xff" * 200000, b"abcabcabdabca" * 20000):
            self.check_roundtrip(raw)

    def test_seeded_uniform_random(self):
        rng = random.Random(20260906)
        for size in (3, 31, 257, 4095, 4096, 32768, 65536, 385000):
            self.check_roundtrip(rng.randbytes(size))

    def test_biased_random(self):
        rng = random.Random(42)
        for alphabet in (2, 3, 7, 16, 127):
            self.check_roundtrip(bytes(rng.randrange(alphabet) for _ in range(40000)))

    def test_truncation_trailing_wrong_mode_and_limit(self):
        for bits in MODES:
            command = f"bit{bits}"
            encoded = invoke(command + "-encode", bytes(range(256)) * 10)
            for size in (0, 1, 4, 8, 16, 24, 32, 39, 40, len(encoded) - 1):
                invoke(command + "-decode", encoded[:size], False)
            invoke(command + "-decode", encoded + b"\x00", False)
            wrong = 8 if bits == 4 else 4
            invoke(f"bit{wrong}-decode", encoded, False)
            damaged = bytearray(encoded)
            struct.pack_into("<Q", damaged, 8, 256 * 1024 * 1024 + 1)
            invoke(command + "-decode", damaged, False)

    def test_header_payload_corruption(self):
        for bits in MODES:
            command = f"bit{bits}"
            encoded = invoke(command + "-encode", bytes(range(256)) * 50)
            for index in (0, 3, 4, 5, 7, 8, 16, 24, 31, 32, 39, 40, 50, len(encoded) - 1):
                damaged = bytearray(encoded)
                damaged[index] ^= 1
                invoke(command + "-decode", damaged, False)

    def test_valid_hash_noncanonical_termination_and_trailing(self):
        for bits in MODES:
            command = f"bit{bits}"
            for raw in (b"", b"A", b"AB", b"\x00" * 5000, b"hello arithmetic" * 100):
                encoded = invoke(command + "-encode", raw)
                body = encoded[40:]
                altered_bodies = [body[:-1], body + b"junk", body + b"\x00", body + b"\x00" * 32]
                altered_bodies += [body[:-1] + bytes([body[-1] ^ (1 << bit)]) for bit in range(8)]
                for altered in altered_bodies:
                    invoke(command + "-decode", repaired_frame(encoded, altered), False)
            encoded_empty = invoke(command + "-encode", b"")
            for arbitrary in (b"", b"\x00", b"\xff", b"arbitrary"):
                invoke(command + "-decode", repaired_frame(encoded_empty, arbitrary), False)

    def test_short_fuzz(self):
        rng = random.Random(801)
        for bits in MODES:
            for _ in range(20):
                raw = rng.randbytes(rng.randrange(1, 400))
                encoded = invoke(f"bit{bits}-encode", raw)
                self.assertEqual(invoke(f"bit{bits}-decode", encoded), raw)

    def test_no_complete_codec_dependency(self):
        output = subprocess.run(["otool", "-L", str(BINARY)], capture_output=True, check=True).stdout.decode()
        for forbidden in ("libz.", "libbz2", "liblzma", "libzstd", "libbrotli", "liblz4"):
            self.assertNotIn(forbidden, output)


def benchmark():
    rng = random.Random(17)
    results = []
    for size in (40000, 385000):
        inputs = {"uniform": rng.randbytes(size),
                  "small_alphabet": bytes(rng.randrange(16) for _ in range(size)),
                  "zero": bytes(size)}
        for label, raw in inputs.items():
            for bits in MODES:
                encode_times, decode_times = [], []
                for _ in range(3):
                    start = time.perf_counter()
                    encoded = invoke(f"bit{bits}-encode", raw)
                    encode_times.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    restored = invoke(f"bit{bits}-decode", encoded)
                    decode_times.append(time.perf_counter() - start)
                    assert restored == raw
                elapsed = statistics.median(encode_times)
                results.append(dict(input=label, context_bits=bits, raw_bytes=size, encoded_bytes=len(encoded),
                                    encode_seconds_median3=elapsed,
                                    decode_including_reencode_seconds_median3=statistics.median(decode_times),
                                    encode_MBps=size / elapsed / 1e6))
    return {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            "header_bytes": 40, "complete_codec_calls": False,
            "semantics": "MSB-first bits, previous-k-bit adaptive [1,1], rescale at 4096, no hidden transforms",
            "benchmark": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        build()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrimitiveBinaryEntropyTest))
    if args.benchmark:
        print(json.dumps(benchmark(), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
