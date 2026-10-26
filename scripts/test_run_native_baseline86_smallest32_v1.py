#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import run_native_baseline86_smallest32_v1 as runner

FRAME = b"frame-bytes!"
FRAME_SHA256 = hashlib.sha256(FRAME).hexdigest()


def inventory() -> dict:
    methods = []
    for index in range(86):
        methods.append({
            "label": f"m{index}", "linked": index == 0,
            "implementation_class": "direct_current" if index == 0 else "unsupported",
            "primary_comparison": index == 0, "implementation": "fixture",
            "configuration": "fixed", "dtypes": ["int64"] if index == 0 else [],
        })
    return {"methods": methods}


def manifest(root: Path) -> dict:
    return {
        "dataset_id": "fixture", "domain": "science", "path": (root / "dataset_manifest.json").resolve(),
        "sha256": "a" * 64, "provenance_sha256": "b" * 64, "block_count": 1,
        "original_numeric_bytes": 16,
        "blocks": {"block0": {"block_id": "block0", "dtype": "int64", "row_count": 2,
                              "original_numeric_bytes": 16}},
    }


def row(method: int, status: str = "unsupported") -> dict:
    linked = method == 0
    result = {
        "schema": "native-baseline86-result-v1", "dataset_id": "fixture",
        "manifest": "", "block_id": "block0", "method": f"m{method}",
        "domain": "science", "column": "x", "source_column": "x", "dtype": "int64",
        "row_count": 2, "original_numeric_bytes": 16,
        "implementation_class": "direct_current" if linked else "unsupported",
        "primary_comparison": linked, "implementation": "fixture", "configuration": "fixed",
        "status": status,
    }
    if status == "ok":
        result.update(complete_frame_bytes=len(FRAME), complete_frame_sha256=FRAME_SHA256,
                      complete_frame_relative_path=f"{FRAME_SHA256[:2]}/{FRAME_SHA256}.ngf1",
                      complete_frame_storage="content_addressed_ngf1",
                      compression_ns_samples=[10, 11, 12, 13, 14],
                      decompression_ns_samples=[20, 21, 22, 23, 24], compression_ns=12.0,
                      decompression_ns=22.0, compression_ratio=.75,
                      compression_ns_per_original_numeric_byte=.75,
                      decompression_ns_per_original_numeric_byte=1.375)
    else:
        result["reason"] = "not linked"
    return result


def write_frame(root: Path) -> Path:
    path = root / FRAME_SHA256[:2] / f"{FRAME_SHA256}.ngf1"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FRAME)
    return path


class RunnerTests(unittest.TestCase):
    def test_manifest_matrix_and_checkpoint_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_dir = root / "frames"
            write_frame(frame_dir)
            item = manifest(root)
            rows = [row(0, "ok")] + [row(i) for i in range(1, 86)]
            for value in rows:
                value["manifest"] = str(item["path"])
            counts = runner.validate_manifest_rows(
                rows, item, inventory(), repeats=5, frame_dir=frame_dir)
            self.assertEqual(counts["ok"], 1)
            self.assertEqual(counts["unsupported"], 85)
            raw = root / "raw.jsonl"
            raw.write_text("".join(json.dumps(value) + "\n" for value in rows))
            identity = {"x": 1}
            checkpoint = root / "checkpoint.json"
            runner.atomic_json(checkpoint, {
                "schema": runner.CHECKPOINT_SCHEMA, "status": "verified", "identity": identity,
                "raw_jsonl_sha256": runner.sha256_file(raw), "status_counts": counts,
                "complete_frames": [{"relative_path": f"{FRAME_SHA256[:2]}/{FRAME_SHA256}.ngf1",
                                     "sha256": FRAME_SHA256, "bytes": len(FRAME)}],
            })
            reusable, loaded = runner.checkpoint_reusable(
                checkpoint, raw, identity, item, inventory(), repeats=5, frame_dir=frame_dir)
            self.assertTrue(reusable)
            self.assertEqual(len(loaded or []), 86)
            raw.write_text(raw.read_text() + "{}\n")
            self.assertFalse(runner.checkpoint_reusable(
                checkpoint, raw, identity, item, inventory(), repeats=5,
                frame_dir=frame_dir)[0])
            raw.write_text("".join(json.dumps(value) + "\n" for value in rows))
            write_frame(frame_dir).write_bytes(b"corrupt")
            self.assertFalse(runner.checkpoint_reusable(
                checkpoint, raw, identity, item, inventory(), repeats=5,
                frame_dir=frame_dir)[0])

    def test_missing_method_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);frame_dir = root / "frames"
            item = manifest(root)
            rows = [row(i) for i in range(85)]
            for value in rows:
                value["manifest"] = str(item["path"])
            with self.assertRaises(runner.CampaignError):
                runner.validate_manifest_rows(
                    rows, item, inventory(), repeats=5, frame_dir=frame_dir)

    def test_column_aggregate_uses_sum_of_block_medians(self) -> None:
        first = row(0, "ok")
        second = dict(first, block_id="block1", original_numeric_bytes=8,
                      complete_frame_bytes=6, compression_ns=20.0, decompression_ns=30.0,
                      compression_ratio=.75, compression_ns_per_original_numeric_byte=2.5,
                      decompression_ns_per_original_numeric_byte=3.75)
        result = runner.summarize_columns([first, second])[0]
        self.assertTrue(result["main_metrics_admitted"])
        self.assertEqual(result["complete_frame_bytes_sum"], 18)
        self.assertEqual(result["original_numeric_bytes_sum"], 24)
        self.assertEqual(result["compression_ns_sum_of_block_medians"], 32)
        self.assertEqual(result["compression_ratio"], .75)
        self.assertEqual(result["compression_ns_per_original_numeric_byte"], 32 / 24)

    def test_partial_failure_has_no_main_metric(self) -> None:
        good = row(0, "ok")
        failed = dict(good, block_id="block1", status="failed", reason="fixture")
        for name in ("complete_frame_bytes", "compression_ns", "decompression_ns",
                     "compression_ratio", "compression_ns_per_original_numeric_byte",
                     "decompression_ns_per_original_numeric_byte"):
            failed.pop(name, None)
        result = runner.summarize_columns([good, failed])[0]
        self.assertEqual(result["status"], "partial_failed")
        self.assertFalse(result["main_metrics_admitted"])
        self.assertIsNone(result["compression_ratio"])
        self.assertIsNone(result["compression_ns_per_original_numeric_byte"])

    def test_compatibility_columns_json_and_cpu_parser(self) -> None:
        result = runner.summarize_columns([row(0, "ok")])[0]
        expected = {
            "dataset_id", "column", "method", "implementation_class",
            "primary_comparison", "status", "block_count", "entity_count",
            "original_numeric_bytes", "frame_bytes", "compression_time_ns",
            "decompression_time_ns", "compression_ratio", "compression_ns_per_byte",
            "decompression_ns_per_byte",
        }
        self.assertTrue(expected.issubset(result))
        self.assertEqual(runner.parse_cpus("25,26"), (25, 26))
        with self.assertRaises(runner.CampaignError):
            runner.parse_cpus("25,25")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner.write_column_outputs(output, [result])
            document = json.loads((output / "columns.json").read_bytes())
            self.assertEqual(document["columns"][0]["compression_time_ns"], 12.0)

    def test_frame_index_is_content_addressed_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);frame_dir=output/"frames";write_frame(frame_dir)
            document=runner.write_frame_index(output,frame_dir,[row(0,"ok")])
            self.assertEqual(document["frame_count"],1)
            self.assertEqual(document["frame_bytes"],len(FRAME))
            self.assertEqual(document["frames"][0]["path"],
                             f"frames/{FRAME_SHA256[:2]}/{FRAME_SHA256}.ngf1")


if __name__ == "__main__":
    unittest.main()
