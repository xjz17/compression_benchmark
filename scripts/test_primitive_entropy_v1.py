#!/usr/bin/env python3
"""Independent round-trip, corruption, framing and throughput tests for PEV1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cpp/src/primitive_entropy_v1.cpp"
BINARY = ROOT / "cpp/build/primitive_entropy_v1/primitive_entropy_v1"
MODES = ("range0", "range1", "sparse0", "sparse1")


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
        raise AssertionError("corrupt input unexpectedly accepted")
    return result.stdout


class PrimitiveEntropyTest(unittest.TestCase):
    def check_roundtrip(self, raw):
        for mode, command in enumerate(MODES):
            encoded = invoke(command + "-encode", raw)
            self.assertEqual(invoke(command + "-decode", encoded), raw)
            self.assertEqual(encoded[:4], b"PEV1")
            self.assertEqual(encoded[4], mode)
            self.assertEqual(struct.unpack_from("<Q", encoded, 8)[0], len(raw))
            self.assertEqual(struct.unpack_from("<Q", encoded, 16)[0] + 40, len(encoded))
            self.assertEqual(encoded, invoke(command + "-encode", raw))

    def test_empty_and_short(self):
        for raw in (b"", b"\x00", b"\xff", b"ab", b"abc", bytes(range(32))):
            self.check_roundtrip(raw)

    def test_all_symbols_and_contexts(self):
        self.check_roundtrip(bytes(range(256)) * 256)
        self.check_roundtrip(bytes(x for a in range(256) for b in range(256) for x in (a, b)))

    def test_repetition_and_rescale(self):
        for raw in (b"\x00" * 200000, b"\xff" * 200000, b"abcabcabdabca" * 20000):
            self.check_roundtrip(raw)

    def test_seeded_random(self):
        rng = random.Random(20260906)
        for n in (3, 7, 31, 257, 4096, 32767, 32768, 65536, 262144):
            self.check_roundtrip(rng.randbytes(n))

    def test_biased_random(self):
        rng = random.Random(42)
        for alphabet in (2, 3, 7, 16, 127):
            self.check_roundtrip(bytes(rng.randrange(alphabet) for _ in range(40000)))

    def test_truncation_trailing_and_mode(self):
        encoded = invoke("range0-encode", bytes(range(256)) * 10)
        for n in (0, 1, 4, 8, 16, 24, 32, 39, 40, len(encoded) - 1):
            invoke("range0-decode", encoded[:n], False)
        invoke("range0-decode", encoded + b"\x00", False)
        invoke("range1-decode", encoded, False)

    def test_header_and_payload_corruption(self):
        for command in MODES:
            encoded = invoke(command + "-encode", bytes(range(256)) * 50)
            for index in (0, 3, 4, 5, 7, 8, 16, 24, 31, 32, 39, 40, 50, len(encoded) - 1):
                damaged = bytearray(encoded)
                damaged[index] ^= 1
                invoke(command + "-decode", damaged, False)
            oversized = bytearray(encoded)
            struct.pack_into("<Q", oversized, 8, 256 * 1024 * 1024 + 1)
            invoke(command + "-decode", oversized, False)

    def test_rehashed_corrupt_payload_and_short_fuzz(self):
        rng = random.Random(801)
        for command in MODES:
            original = invoke(command + "-encode", b"hello arithmetic" * 100)
            for n in (0, 1, 2, 4, 10, 32):
                body = rng.randbytes(n)
                damaged = bytearray(original[:40]) + body
                body_hash = 14695981039346656037
                for byte in body:
                    body_hash = ((body_hash ^ byte) * 1099511628211) & ((1 << 64) - 1)
                struct.pack_into("<QQ", damaged, 16, n, struct.unpack_from("<Q", original, 24)[0])
                struct.pack_into("<Q", damaged, 32, body_hash)
                invoke(command + "-decode", damaged, False)
            for _ in range(20):
                raw = rng.randbytes(rng.randrange(1, 400))
                encoded = invoke(command + "-encode", raw)
                self.assertEqual(invoke(command + "-decode", encoded), raw)

    def test_no_complete_codec_dependency(self):
        result = subprocess.run(["otool", "-L", str(BINARY)], capture_output=True, check=True).stdout.decode()
        for forbidden in ("libz.", "libbz2", "liblzma", "libzstd", "libbrotli", "liblz4"):
            self.assertNotIn(forbidden, result)

    def test_repaired_header_with_noncanonical_body(self):
        def repair(original, body):
            value = 14695981039346656037
            for byte in body:
                value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
            header = bytearray(original[:40])
            struct.pack_into('<Q', header, 16, len(body))
            struct.pack_into('<Q', header, 32, value)
            # Preserve the CORRECT original raw length and raw checksum.
            return bytes(header) + body
        for command in MODES:
            for raw in (b'', b'A', b'AB', b'hello arithmetic' * 100):
                encoded = invoke(command + '-encode', raw)
                body = encoded[40:]
                for altered in (body[:-1], body + b'junk', body + b'\x00'):
                    invoke(command + '-decode', repair(encoded, altered), False)
            empty = invoke(command + '-encode', b'')
            for arbitrary in (b'\x00', b'\xff', b'arbitrary'):
                if arbitrary != empty[40:]:
                    invoke(command + '-decode', repair(empty, arbitrary), False)


def benchmark():
    rng = random.Random(17)
    inputs = {"uniform": rng.randbytes(1000000), "small_alphabet": bytes(rng.randrange(16) for _ in range(1000000)),
              "repeated": b"abcabcabdabca" * 83333}
    results = []
    for label, raw in inputs.items():
        for mode, command in enumerate(MODES):
            start = time.perf_counter()
            encoded = invoke(command + "-encode", raw)
            elapsed = time.perf_counter() - start
            start = time.perf_counter()
            restored = invoke(command + "-decode", encoded)
            decode_elapsed = time.perf_counter() - start
            assert restored == raw
            results.append(dict(input=label, mode=mode, command=command, raw_bytes=len(raw), encoded_bytes=len(encoded),
                                encode_seconds=elapsed, decode_seconds=decode_elapsed,
                                encode_MBps=len(raw) / elapsed / 1e6))
    return {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            "header_bytes": 40, "model": "adaptive order-0/order-1 byte arithmetic, uniform or sparse escape priors",
            "complete_codec_calls": False, "benchmark": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        build()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrimitiveEntropyTest))
    if args.benchmark:
        print(json.dumps(benchmark(), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
