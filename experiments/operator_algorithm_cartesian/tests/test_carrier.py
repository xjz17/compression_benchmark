#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from carrier import (  # noqa: E402
    ACTIVE85,
    BYTE_ALGORITHMS,
    DIRECT_U8,
    FLOAT64_CARRIER_ALGORITHMS,
    FLOAT64_U8_ALGORITHMS,
    FLOAT64_U8,
    FLOAT64_U48,
    INT64_CARRIER_ALGORITHMS,
    INT64_U56,
    carrier_kind_for_algorithm,
    pack_bytes,
    unpack_bytes,
)
from joint_frame import build_frame, parse_frame  # noqa: E402
from shard_runner import load_routes  # noqa: E402


class CarrierTest(unittest.TestCase):
    def test_partition_is_exact_active85(self) -> None:
        self.assertEqual(len(ACTIVE85), 85)
        self.assertEqual(len(BYTE_ALGORITHMS), 59)
        self.assertEqual(len(INT64_CARRIER_ALGORITHMS), 13)
        self.assertEqual(len(FLOAT64_CARRIER_ALGORITHMS), 12)
        self.assertEqual(len(FLOAT64_U8_ALGORITHMS), 1)
        self.assertFalse(BYTE_ALGORITHMS & INT64_CARRIER_ALGORITHMS)
        self.assertFalse(BYTE_ALGORITHMS & FLOAT64_CARRIER_ALGORITHMS)
        self.assertFalse(INT64_CARRIER_ALGORITHMS & FLOAT64_CARRIER_ALGORITHMS)
        self.assertFalse(ACTIVE85 - (BYTE_ALGORITHMS | INT64_CARRIER_ALGORITHMS |
                                    FLOAT64_CARRIER_ALGORITHMS |
                                    FLOAT64_U8_ALGORITHMS))
        self.assertEqual(carrier_kind_for_algorithm("Zstd"), DIRECT_U8)
        self.assertEqual(carrier_kind_for_algorithm("BOS"), INT64_U56)
        self.assertEqual(carrier_kind_for_algorithm("Elf"), FLOAT64_U48)
        self.assertEqual(carrier_kind_for_algorithm("Camel"), FLOAT64_U8)

    def test_roundtrip_boundaries_and_random(self) -> None:
        rng = random.Random(20261026)
        payloads = [
            b"", b"\0", b"\xff", bytes(range(256)),
            bytes(rng.randrange(256) for _ in range(10003)),
        ]
        for size in range(0, 33):
            payloads.append(bytes(rng.randrange(256) for _ in range(size)))
        for kind in (DIRECT_U8, INT64_U56, FLOAT64_U48, FLOAT64_U8):
            for payload in payloads:
                with self.subTest(kind=kind, size=len(payload)):
                    packed = pack_bytes(payload, kind)
                    self.assertEqual(
                        unpack_bytes(packed.values, packed.missing, packed.metadata),
                        payload,
                    )

    def test_numeric_domains_are_safe(self) -> None:
        payload = bytes(range(256)) * 11
        ints = pack_bytes(payload, INT64_U56)
        for (value,) in struct.iter_unpack("<q", ints.values):
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 1 << 56)
        floats = pack_bytes(payload, FLOAT64_U48)
        for (value,) in struct.iter_unpack("<d", floats.values):
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 1 << 48)
            self.assertEqual(value, float(int(value)))
        byte_floats = pack_bytes(payload, FLOAT64_U8)
        for source, (value,) in zip(payload, struct.iter_unpack("<d", byte_floats.values)):
            self.assertEqual(value, float(source))

    def test_tampering_fails_closed(self) -> None:
        packed = pack_bytes(b"abcdefghijk", INT64_U56)
        changed = bytearray(packed.values)
        changed[0] ^= 1
        with self.assertRaises(ValueError):
            unpack_bytes(bytes(changed), packed.missing, packed.metadata)
        changed_meta = bytearray(packed.metadata)
        changed_meta[-1] ^= 1
        with self.assertRaises(ValueError):
            unpack_bytes(packed.values, packed.missing, bytes(changed_meta))
        with self.assertRaises(ValueError):
            unpack_bytes(packed.values, b"\x01", packed.metadata)

    def test_complete_joint_frame_and_tamper(self) -> None:
        payload = b"O2R1example"
        packed = pack_bytes(payload, FLOAT64_U48)
        binding = {
            "schema": "operator-algorithm-joint-binding-v1",
            "operator_id": "A013",
            "algorithm": "Elf",
            "carrier_kind": FLOAT64_U48,
        }
        source_sha = hashlib.sha256(b"source-column").hexdigest()
        frame = build_frame(binding, packed.metadata, b"algorithm-frame", source_sha)
        parsed = parse_frame(frame)
        self.assertEqual(parsed["binding"], binding)
        self.assertEqual(parsed["source_column_sha256"], source_sha)
        self.assertEqual(parsed["carrier_metadata"], packed.metadata)
        self.assertEqual(parsed["algorithm_frame"], b"algorithm-frame")
        changed = bytearray(frame)
        changed[-1] ^= 1
        with self.assertRaises(ValueError):
            parse_frame(bytes(changed))

    def test_frozen_active85_route_manifest_shape(self) -> None:
        records = [{"algorithm": name} for name in sorted(ACTIVE85)]
        record_sha = hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
        manifest = {
            "schema": "operator-algorithm-active85-carrier-routes-v1",
            "status": "frozen",
            "algorithms": 85,
            "records": records,
            "records_canonical_sha256": record_sha,
            "contract": {
                "complete_algorithm_frame_charged": True,
                "carrier_metadata_and_missing_bitmap_charged": True,
                "native_W1_R3_T1": True,
                "cold_decode": True,
                "independent_optima_combined": False,
                "no_substitute_codec": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            routes = load_routes(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(set(routes), ACTIVE85)
            manifest["records"][0]["algorithm"] = "tampered"
            path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            with self.assertRaises(RuntimeError):
                load_routes(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
