#!/usr/bin/env python3
"""Focused tests for the prepared-block fixed-86 baseline protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepared_block_baseline86_v1 as baseline  # noqa: E402


class ToyExecutor:
    def __init__(self, *, reject=(), corrupt=(), non_cold=()):
        self.reject = frozenset(reject)
        self.corrupt = frozenset(corrupt)
        self.non_cold = frozenset(non_cold)

    def support(self, request):
        return baseline.SupportDecision(
            supported=request.label not in self.reject,
            reason=("toy rejected label" if request.label in self.reject else "toy exact path"),
            executor_identity="toy-native-executor-v1",
            implementation_evidence={"toy": True, "label": request.label},
        )

    def encode(self, request):
        label = request.label.encode("utf-8")
        frame = (
            b"TOY1" + struct.pack("<HII", len(label), len(request.values),
                                   len(request.missing_bitmap))
            + label + request.values + request.missing_bitmap
        )
        return baseline.EncodeReceipt(
            complete_frame=frame,
            elapsed_ns=80 + len(request.label),
            self_contained=True,
            metadata={"boundary": "toy-complete-frame"},
        )

    def decode(self, request, frame):
        if frame[:4] != b"TOY1":
            raise ValueError("bad toy frame")
        label_size, value_size, bitmap_size = struct.unpack("<HII", frame[4:14])
        cursor = 14 + label_size
        values = frame[cursor:cursor + value_size]
        bitmap = frame[cursor + value_size:cursor + value_size + bitmap_size]
        if request.label in self.corrupt:
            values = bytes([values[0] ^ 1]) + values[1:]
        return baseline.DecodeReceipt(
            values=values,
            missing_bitmap=bitmap,
            elapsed_ns=40 + len(request.label),
            independent_process=request.label not in self.non_cold,
            decoder_identity="toy-cold-decoder-v1",
            metadata={"boundary": "toy-independent-decode"},
        )


def toy_block() -> baseline.PreparedBlock:
    values = struct.pack("<qq", -1, 7)
    bitmap = b"\x04"
    return baseline.PreparedBlock(
        dataset_id="toy-dataset", dataset_provenance_sha256="a" * 64,
        block_id="validation__entity__value__000000", entity_id="entity",
        split="validation", column="value", dtype="int64", row_count=3,
        valid_count=2, block_index=0, values=values, missing_bitmap=bitmap,
        original_numeric_bytes=len(values), values_sha256=hashlib.sha256(values).hexdigest(),
        missing_bitmap_sha256=hashlib.sha256(bitmap).hexdigest(),
    )


class CatalogTests(unittest.TestCase):
    def test_catalog_classifies_every_label_without_upstream_native_claim(self):
        catalog = baseline.build_catalog_snapshot()
        self.assertEqual(len(catalog.labels), 86)
        self.assertEqual(len({row.label for row in catalog.labels}), 86)
        chimp = next(row for row in catalog.labels if row.label == "Chimp")
        self.assertEqual(chimp.implementation_class, baseline.IMPLEMENTATION_PROXY)
        self.assertIn("Chimp128", chimp.classification_reason)
        self.assertTrue(all(not row.upstream_native_claim for row in catalog.labels))
        self.assertEqual(
            sum(row.implementation_class == baseline.IMPLEMENTATION_NATIVE_CURRENT
                for row in catalog.labels),
            85,
        )

    def test_float_only_label_is_explicitly_unsupported_for_int64(self):
        catalog = baseline.build_catalog_snapshot()
        alp = next(row for row in catalog.labels if row.label == "ALP")
        request, reason = baseline.make_execution_request(catalog, alp, toy_block())
        self.assertIsNone(request)
        self.assertIn("int64", reason)


class MeasurementTests(unittest.TestCase):
    def test_complete_frame_metrics_use_values_only_as_denominator(self):
        catalog = baseline.build_catalog_snapshot()
        label = next(row for row in catalog.labels if row.label == "gzip")
        request, reason = baseline.make_execution_request(catalog, label, toy_block())
        self.assertIsNone(reason)
        assert request is not None
        result = baseline.benchmark_request(ToyExecutor(), request, warmups=1, repeats=3)
        expected_frame = ToyExecutor().encode(request).complete_frame
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["complete_compressed_bytes"], len(expected_frame))
        self.assertEqual(result["original_numeric_bytes"], len(request.values))
        self.assertEqual(result["missing_bitmap_bytes_in_complete_frame_contract"], 1)
        self.assertEqual(
            result["rho_complete_bytes_per_original_numeric_byte"],
            len(expected_frame) / len(request.values),
        )
        self.assertEqual(len(result["compression_elapsed_ns_repeats"]), 3)
        self.assertTrue(result["independent_cold_decode"])

    def test_non_cold_or_corrupt_decode_is_saved_as_per_label_failure(self):
        catalog = baseline.build_catalog_snapshot()
        result = baseline.benchmark_block(
            ToyExecutor(corrupt={"gzip"}, non_cold={"Zlib"}),
            catalog, toy_block(), warmups=0, repeats=1,
        )
        by_label = {row["label"]: row for row in result["label_results"]}
        self.assertEqual(by_label["gzip"]["status"], "failed")
        self.assertIn("bit-exact", by_label["gzip"]["error"])
        self.assertEqual(by_label["Zlib"]["status"], "failed")
        self.assertIn("independent cold", by_label["Zlib"]["error"])
        self.assertEqual(result["counts"]["labels"], 86)
        self.assertEqual(result["counts"]["failed"], 2)
        self.assertEqual({row["label"] for row in result["failures"]}, {"gzip", "Zlib"})

    def test_executor_rejection_is_unsupported_not_a_measurement(self):
        catalog = baseline.build_catalog_snapshot()
        result = baseline.benchmark_block(
            ToyExecutor(reject={"gzip"}), catalog, toy_block(), warmups=0, repeats=1
        )
        row = next(row for row in result["label_results"] if row["label"] == "gzip")
        self.assertEqual(row["status"], "unsupported")
        self.assertNotIn("rho_complete_bytes_per_original_numeric_byte", row)


class OutputProtocolTests(unittest.TestCase):
    def test_run_writes_catalog_summary_and_per_block_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            block = toy_block()
            (prepared / "values.bin").write_bytes(block.values)
            (prepared / "bitmap.bin").write_bytes(block.missing_bitmap)
            manifest = {
                "dataset_id": block.dataset_id,
                "provenance_sha256": block.dataset_provenance_sha256,
                "blocks": [{
                    "block_id": block.block_id, "entity_id": block.entity_id,
                    "split": block.split, "column": block.column,
                    "dtype": block.dtype, "row_count": block.row_count,
                    "valid_count": block.valid_count, "block_index": block.block_index,
                    "values_path": "values.bin", "values_sha256": block.values_sha256,
                    "missing_bitmap_path": "bitmap.bin",
                    "missing_bitmap_sha256": block.missing_bitmap_sha256,
                    "original_numeric_bytes": block.original_numeric_bytes,
                }],
            }
            (prepared / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            verification = {
                "dataset_id": block.dataset_id, "blocks": 1,
                "original_numeric_bytes": block.original_numeric_bytes,
                "provenance_sha256": block.dataset_provenance_sha256,
                "source_verified": False,
            }
            output = root / "output"
            with mock.patch.object(
                baseline.preparation, "verify_prepared_dataset", return_value=verification
            ):
                summary = baseline.run_prepared_baselines(
                    ToyExecutor(reject={"gzip"}), [prepared], output,
                    verify_source=False, warmups=0, repeats=1,
                )
            self.assertEqual(summary["counts"]["blocks"], 1)
            self.assertEqual(summary["counts"]["label_block_pairs"], 86)
            self.assertTrue((output / "catalog.json").is_file())
            self.assertTrue((output / "summary.json").is_file())
            receipt_path = output / summary["blocks"][0]["receipt_path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["counts"]["labels"], 86)
            gzip = next(row for row in receipt["label_results"] if row["label"] == "gzip")
            self.assertEqual(gzip["status"], "unsupported")
            measured = next(
                row for row in receipt["label_results"] if row["status"] == "measured"
            )
            artifact = measured["complete_frame_artifact"]
            artifact_path = Path(summary["complete_frame_artifacts"]["root"]) / artifact["relative_path"]
            self.assertTrue(artifact_path.is_file())
            self.assertEqual(
                hashlib.sha256(artifact_path.read_bytes()).hexdigest(), artifact["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
