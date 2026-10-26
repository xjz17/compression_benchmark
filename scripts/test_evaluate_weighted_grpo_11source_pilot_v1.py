from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest import mock
import zlib

import evaluate_weighted_grpo_11source_pilot_v1 as evaluation


class PilotValidationTests(unittest.TestCase):
    def record(self):
        return {"dataset_id": "source", "E0": 1, "D0": 2,
                "block": {"block_id": "validation1", "split": "validation", "dtype": "float32",
                          "original_numeric_bytes": 8, "row_count": 3}}

    def test_validation_only_and_unique(self):
        record = self.record()
        self.assertEqual(evaluation.validate_panel({"blocks": [record]}), [record])
        for split in ("train", "test"):
            altered = copy.deepcopy(record)
            altered["block"]["split"] = split
            with self.assertRaisesRegex(ValueError, "training or test"):
                evaluation.validate_panel({"blocks": [altered]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluation.validate_panel({"blocks": [record, record]})

    def test_validation_requires_exact11_sources_and_train_library(self):
        records = [{**self.record(), "dataset_id": f"source{i}"} for i in range(11)]
        receipt = {"sources": [{"dataset_id": row["dataset_id"]} for row in records]}
        library = {"source_scope": "training_only", "measurements": [{"source_split": "train"}]}
        evaluation.validate_panel_scope(records, receipt, library)
        with self.assertRaisesRegex(ValueError, "same 11"):
            evaluation.validate_panel_scope(records[:-1], receipt, library)
        library["measurements"][0]["source_split"] = "validation"
        with self.assertRaisesRegex(ValueError, "nontraining"):
            evaluation.validate_panel_scope(records, receipt, library)

    def test_no_oversize_or_bad_references(self):
        for field, value in (("E0", 0), ("D0", float("nan"))):
            record = self.record()
            record[field] = value
            with self.assertRaises(ValueError):
                evaluation.validate_panel({"blocks": [record]})
        record = self.record()
        record["block"]["original_numeric_bytes"] = 1048577
        with self.assertRaises(ValueError):
            evaluation.validate_panel({"blocks": [record]})

    def test_zlib_wrapper_carries_root_types_lengths_and_mask(self):
        def fake_native(_cli, command, payload):
            return zlib.compress(payload) if command == "encode" else zlib.decompress(payload)
        # Original float bit patterns include a negative zero and a NaN payload.
        values = struct.pack("<II", 0x80000000, 0x7FC12345)
        bitmap = b"\x02"
        with mock.patch.object(evaluation, "_zlib_cli", side_effect=fake_native):
            frame = evaluation.encode_zlib6_frame(Path("unused"), "float32", 3, values, bitmap)
            self.assertEqual(evaluation.decode_zlib6_frame(Path("unused"), frame),
                             ("float32", 3, values, bitmap))
            self.assertGreater(len(frame), len(zlib.compress(values + bitmap)))
            with self.assertRaisesRegex(ValueError, "integrity"):
                evaluation.decode_zlib6_frame(Path("unused"), frame[:-1] + bytes([frame[-1] ^ 1]))
            with self.assertRaises(ValueError):
                evaluation.decode_zlib6_frame(Path("unused"), frame[:12])

    def test_zlib_partition_rejects_rehashed_invalid_length(self):
        payload = evaluation.ZLIB_HEADER.pack(b"ZP11", 3, 3, 16, 1) + b"invalid_native"
        frame = payload + hashlib.sha256(payload).digest()
        with self.assertRaisesRegex(ValueError, "valid-value length"):
            evaluation.decode_zlib6_frame(Path("unused"), frame)

    def test_isolated_timeout_is_missing_not_zero_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json"
            result = evaluation.run_isolated(lambda: time.sleep(10), path, 0.05)
            self.assertEqual(result["status"], "timeout")
            self.assertNotIn("J", result)
            self.assertNotIn("rho", result)
            self.assertLess(result["elapsed_seconds"], 2)

    def test_isolated_failure_keeps_reason(self):
        def broken():
            raise RuntimeError("native rejected candidate")
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation.run_isolated(broken, Path(directory) / "row.json", 2)
            self.assertEqual(result["status"], "failed")
            self.assertIn("native rejected", result["error"])
            self.assertNotIn("rho", result)

    def test_summary_pairs_only_matching_complete_samegraph_rows(self):
        common = {"dataset_id": "source", "block_id": "b", "weight": [1, 0, 0],
                  "status": "complete", "same_graph_format": True, "rho": .5}
        rows = [{**common, "method": "initial", "J": 1},
                {**common, "method": "trained", "J": .5},
                {**common, "block_id": "other", "method": "trained", "J": .1},
                {**common, "block_id": "bad", "method": "trained", "status": "timeout"}]
        summary = evaluation.paired_summary(rows)
        self.assertEqual(summary["paired_rows"], 1)
        self.assertEqual(summary["trained_vs_initial_same_graph"], {"win": 1, "tie": 0, "loss": 0})


if __name__ == "__main__":
    unittest.main()
