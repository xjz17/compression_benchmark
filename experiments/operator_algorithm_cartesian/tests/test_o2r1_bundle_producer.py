#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "o2r1_bundle_cli"
CATALOG = REPO / "analysis_outputs/operator102_current135_20261017_v1/native/RESOLVED_CATALOG.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class O2R1BundleProducerIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(BINARY.exists() and CATALOG.exists(),
                         "local production producer/catalog absent")
    def test_real_operator_produce_parse_inverse_and_timing(self) -> None:
        catalog = json.loads(CATALOG.read_text())
        for operator_id in ("A013", "A065"):
            with self.subTest(operator_id=operator_id), tempfile.TemporaryDirectory(
                    prefix="o2r1-producer-test-") as temp:
                implementation = next(
                    row for row in catalog["resolved_implementations"]
                    if row["implementation_id"] == operator_id)
                root = Path(temp)
                values = root / "values.bin"
                values.write_bytes(b"".join(struct.pack("<q", value)
                                            for value in (1, 1, 2, 3, 5, 8, 13, 21)))
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({
                    "schema": "fixture-column-v1",
                    "dataset_id": "fixture", "column_id": "value",
                    "byte_order": "little", "values_are_valid_only": True,
                    "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
                    "blocks": [{
                        "block_id": "entity0", "dtype": "int64", "width": 8,
                        "valid_count": 8, "row_count": 8, "missing_count": 0,
                        "values_path": str(values), "values_sha256": sha(values),
                    }],
                }))
                output = root / "bundle"
                request = {
                    "schema": "operator-algorithm-o2r1-column-request-v1",
                    "request_id": f"fixture-{operator_id}", "dataset_id": "fixture",
                    "column_id": "value", "manifest_path": str(manifest),
                    "expected_manifest_sha256": sha(manifest),
                    "input_contract": {
                        "byte_order": "little", "values_are_valid_only": True,
                        "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
                    },
                    "implementation": implementation, "output_dir": str(output),
                    "provenance": {"test": True},
                }
                completed = subprocess.run(
                    [str(BINARY), "--jsonl"], input=json.dumps(request) + "\n",
                    text=True, capture_output=True, timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                response = json.loads(completed.stdout)
                self.assertEqual(response["status"], "passed")
                receipt = json.loads((output / "BUNDLE.json").read_text())
                self.assertTrue(receipt["bit_exact"])
                self.assertTrue(receipt["strict"])
                self.assertFalse(receipt["fallback_used"])
                self.assertEqual(receipt["block_count"], 1)
                block = receipt["blocks"][0]
                o2r1 = output / block["o2r1_path"]
                self.assertTrue(o2r1.read_bytes().startswith(b"O2R1"))
                self.assertEqual(sha(o2r1), block["o2r1_sha256"])
                inverse = json.loads(
                    (output / block["operator_inverse_receipt_path"]).read_text())
                self.assertEqual(inverse["status"], "passed")
                self.assertTrue(inverse["bit_exact"])
                self.assertTrue(inverse["strict"])
                self.assertTrue(inverse["inverse_executed_from_parsed_o2r1"])
                self.assertEqual(len(receipt["timing"][
                    "full_encode_including_o2r1_pack"]["samples_ns"]), 3)
                self.assertEqual(len(receipt["timing"][
                    "full_decode_including_o2r1_parse"]["samples_ns"]), 3)


if __name__ == "__main__":
    unittest.main()
