#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import calibrate_weighted_grpo_references_v1 as calibration

from calibrate_weighted_grpo_references_v1 import (
    SCHEMA,
    SMOKE_SCHEMA,
    block_size_bucket,
)
import weighted_grpo_experiment_v1 as experiment


class CalibrationReferenceTests(unittest.TestCase):
    def test_power_of_two_block_buckets(self):
        self.assertEqual(block_size_bucket(1), 1)
        self.assertEqual(block_size_bucket(8), 8)
        self.assertEqual(block_size_bucket(9), 16)
        self.assertEqual(block_size_bucket(1 << 20), 1 << 20)

    def test_invalid_block_sizes_are_rejected(self):
        for value in (0, -1, (1 << 20) + 1, True):
            with self.assertRaises(ValueError):
                block_size_bucket(value)

    def test_smoke_schema_is_distinct_and_formal_loader_rejects_it(self):
        self.assertNotEqual(SCHEMA, SMOKE_SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            path.write_text(json.dumps({
                "schema": SMOKE_SCHEMA,
                "calibration_sha256": "not-a-formal-table",
            }), encoding="utf-8")
            with self.assertRaises(experiment.ExperimentContractError):
                experiment._load_reference_table(path)

    def test_progress_preserves_train_only_scope_and_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values, bitmap = b"12345678", b"\x00"
            (root / "values.bin").write_bytes(values)
            (root / "bitmap.bin").write_bytes(bitmap)
            block = {
                "split": "train", "block_id": "train-1", "dtype": "int64",
                "original_numeric_bytes": len(values), "values_path": "values.bin",
                "missing_bitmap_path": "bitmap.bin",
                "values_sha256": hashlib.sha256(values).hexdigest(),
                "missing_bitmap_sha256": hashlib.sha256(bitmap).hexdigest(),
            }
            manifest = {
                "dataset_id": "test-fixture", "provenance_sha256": "fixture",
                "blocks": [block, {**block, "split": "validation", "block_id": "held-out"}],
            }
            native = {"compression_median_ns": 80, "decompression_median_ns": 40,
                      "complete_frame_bytes": 16}
            events = []
            with patch.object(calibration, "_load_manifest", return_value=(root, manifest)), \
                    patch.object(calibration.preparation, "validate_prepared_cohort_manifests"), \
                    patch.object(calibration, "_run_native", return_value=native) as measure, \
                    patch.object(calibration.subprocess, "run", return_value=
                                 subprocess.CompletedProcess([], 0, b"test-version", b"")):
                document = calibration.calibrate(
                    [root], cli=root / "cli", require_complete_cohort=False,
                    progress=events.append)
            measure.assert_called_once_with(root / "cli", values + bitmap, warmups=3, repeats=15)
            self.assertEqual(document["schema"], SMOKE_SCHEMA)
            self.assertFalse(document["formal_training_eligible"])
            self.assertEqual(document["training_block_count"], 1)
            self.assertEqual(document["references"][0]["E0_compression_ns_per_original_byte"], 10)
            self.assertEqual(events[-1]["measured_blocks"], 1)
            self.assertEqual(events[-1]["original_numeric_bytes"], 8)
            self.assertEqual(events[-1]["training_blocks"], 1)

    def test_native_measurement_has_a_bounded_process_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            cli = Path(directory) / "cli"
            cli.touch()
            with patch.object(calibration.subprocess, "run", side_effect=
                              subprocess.TimeoutExpired("cli", 120)) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    calibration._run_native(cli, b"1234", warmups=1, repeats=3)
            self.assertEqual(launch.call_args.kwargs["timeout"], 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
