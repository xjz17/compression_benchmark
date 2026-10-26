#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_clean200_column_plan_v1 import canonical_sha, load_scope


class Clean200PlanTest(unittest.TestCase):
    def test_canonical_sha_is_order_independent_for_objects(self) -> None:
        self.assertEqual(canonical_sha({"b": 2, "a": 1}),
                         canonical_sha({"a": 1, "b": 2}))

    def test_normalizes_job_manifest_and_verifies_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "column.json"
            manifest.write_text('{"schema":"fixture"}\n')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            plan = {
                "schema": "fixture",
                "input_contract": {
                    "byte_order": "little",
                    "values_are_valid_only": True,
                    "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
                },
                "jobs": [{
                    "dataset_id": f"d{index % 15}",
                    "column_id": f"c{index}",
                    "job_id": f"j{index}",
                    "manifest_path": str(manifest),
                    "manifest_sha256": digest,
                } for index in range(125)],
            }
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            source, rows = load_scope("new15", path, True)
            self.assertEqual(source["columns"], 125)
            self.assertEqual(len(rows), 125)
            self.assertEqual(rows[0]["manifest_bindings"][0]["sha256"], digest)

    def test_rejects_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "column.json"
            manifest.write_text('{}\n')
            plan = {
                "schema": "fixture",
                "input_contract": {
                    "byte_order": "little",
                    "values_are_valid_only": True,
                    "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
                },
                "jobs": [{
                    "dataset_id": f"d{index % 15}",
                    "column_id": f"c{index}",
                    "manifest_path": str(manifest),
                    "manifest_sha256": "0" * 64,
                } for index in range(125)],
            }
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, "manifest SHA mismatch"):
                load_scope("new15", path, True)


if __name__ == "__main__":
    unittest.main()
