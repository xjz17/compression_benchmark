#!/usr/bin/env python3
"""Split/source-boundary checks using the actual prepared-data verifier."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import prepare_weighted_grpo_timeseries_v1 as preparation
from run_weighted_grpo_realdata_smoke_v1 import SmokeError, choose_training_block, load_training_input
import test_prepare_weighted_grpo_timeseries_v1 as fixtures


class RealDataSmokeBoundaryTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, dict]:
        cohort = preparation.load_cohort_manifest()
        entry = next(row for row in cohort["datasets"] if row["id"] == "national_illness")
        source = fixtures.PreparationIntegrationTest._write_fixture(root)
        spec = fixtures.PreparationIntegrationTest._write_spec(
            root, source, dataset_id=entry["id"], domain=entry["domain"],
            source_group=entry["source_group"],
        )
        spec_document = json.loads(spec.read_text())
        spec_document["source"]["upstream_locator"] = entry["locator"]
        spec.write_text(json.dumps(spec_document))
        prepared = root / "prepared"
        manifest = preparation.prepare_dataset(
            preparation.load_preparation_contract(spec), prepared, cohort_manifest=cohort,
        )
        admission = root / "admission.json"
        admission.write_text(json.dumps({"sources": [{
            "dataset_id": entry["id"], "source_group": entry["source_group"],
            "status": "approved", "revision": "synthetic-v1",
        }]}))
        return prepared, admission, manifest

    def test_real_verifier_and_loader_use_only_complete_train_block(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, admission, manifest = self.fixture(Path(directory))
            source, values, bitmap = load_training_input(
                prepared, admission, block_id=None, max_block_bytes=16384,
            )
            self.assertEqual(source["block"]["split"], "train")
            self.assertEqual(len(values), source["block"]["original_numeric_bytes"])
            self.assertEqual(len(bitmap), source["block"]["missing_bitmap_bytes"])
            held_out = next(row for row in manifest["blocks"] if row["split"] == "validation")
            with self.assertRaisesRegex(SmokeError, "no eligible complete training block"):
                load_training_input(prepared, admission, block_id=held_out["block_id"],
                                    max_block_bytes=16384)

    def test_modified_values_fail_standard_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, admission, manifest = self.fixture(Path(directory))
            row = next(row for row in manifest["blocks"] if row["split"] == "train")
            path = prepared / row["values_path"]
            payload = path.read_bytes()
            path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
            with self.assertRaises(preparation.PreparationError):
                load_training_input(prepared, admission, block_id=row["block_id"],
                                    max_block_bytes=16384)

    def test_blocked_or_changed_revision_cannot_enter_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, admission, _manifest = self.fixture(Path(directory))
            document = json.loads(admission.read_text())
            document["sources"][0]["status"] = "blocked"
            admission.write_text(json.dumps(document))
            with self.assertRaisesRegex(SmokeError, "not approved"):
                load_training_input(prepared, admission, block_id=None, max_block_bytes=16384)
            document["sources"][0]["status"] = "approved"
            document["sources"][0]["revision"] = "another-revision"
            admission.write_text(json.dumps(document))
            with self.assertRaisesRegex(SmokeError, "revision differs"):
                load_training_input(prepared, admission, block_id=None, max_block_bytes=16384)

    def test_test_role_and_oversize_do_not_fallback_or_truncate(self):
        with tempfile.TemporaryDirectory() as directory:
            _prepared, _admission, manifest = self.fixture(Path(directory))
            manifest["cohort"]["role"] = "test_candidate"
            with self.assertRaisesRegex(SmokeError, "train_validation"):
                choose_training_block(manifest, block_id=None, max_block_bytes=16384)
            manifest["cohort"]["role"] = "train_validation"
            with self.assertRaisesRegex(SmokeError, "no eligible"):
                choose_training_block(manifest, block_id=None, max_block_bytes=1)


if __name__ == "__main__":
    unittest.main()
