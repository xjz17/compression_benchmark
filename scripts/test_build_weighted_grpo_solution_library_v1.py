#!/usr/bin/env python3
"""Tests for the measured, training-only weighted-GRPO solution library."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import build_weighted_grpo_solution_library_v1 as builder
import calibrate_weighted_grpo_references_v1 as calibration
import formal_weighted_grpo_training_v1 as training
import typed_graph_runtime as graph_runtime
import weight_conditioned_graph_task_v1 as graph_task
import weighted_grpo_graph_task_factory_v1 as factory_module


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class SolutionLibraryBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        (self.prepared / "blocks").mkdir(parents=True)
        self.values = struct.pack("<16q", *range(16))
        self.bitmap = b"\x00\x00"
        self.values_path = self.prepared / "blocks" / "block.values.bin"
        self.bitmap_path = self.prepared / "blocks" / "block.missing.bitmap"
        self.values_path.write_bytes(self.values)
        self.bitmap_path.write_bytes(self.bitmap)
        self.provenance = "a" * 64
        self.block = {
            "block_id": "train-block-0",
            "entity_id": "entity",
            "column": "value",
            "split": "train",
            "block_index": 0,
            "dtype": "int64",
            "row_count": 16,
            "valid_count": 16,
            "missing_count": 0,
            "values_path": "blocks/block.values.bin",
            "missing_bitmap_path": "blocks/block.missing.bitmap",
            "values_sha256": _sha(self.values),
            "missing_bitmap_sha256": _sha(self.bitmap),
            "values_bytes": len(self.values),
            "missing_bitmap_bytes": len(self.bitmap),
            "prepared_block_bytes": len(self.values) + len(self.bitmap),
            "original_numeric_bytes": len(self.values),
        }
        self.manifest = {
            "format_version": "weighted-grpo-prepared-timeseries-v1",
            "dataset_id": "training-dataset",
            "cohort": {"role": "train_validation"},
            "source": {"sha256": "b" * 64},
            "provenance_sha256": self.provenance,
            "blocks": [self.block],
        }
        self.manifest_path = self.prepared / "dataset_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.reference_path = self.root / "reference.json"
        reference = {
            "schema": calibration.SCHEMA,
            "formal_training_eligible": True,
            "objective_normalization": "E/(E+E0), D/(D+D0)",
            "references": [{
                "dtype": "int64",
                "block_size_bucket_bytes": 256,
                "E0_compression_ns_per_original_byte": 10.0,
                "D0_decompression_ns_per_original_byte": 5.0,
            }],
        }
        reference["calibration_sha256"] = _sha(_canonical(reference))
        self.reference_path.write_text(json.dumps(reference), encoding="utf-8")
        self.delta = graph_task.find_action("op.delta.reference_v1", input_view="int64")
        self.zigzag = graph_task.find_action(
            "op.zigzag.signed64.raw.cpp_v2", input_view="int64"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _frame(self, recipe: tuple[int, ...]) -> bytes:
        state = graph_runtime.GraphState.from_streams(
            (
                graph_runtime.Stream("numeric", 8, self.values),
                graph_runtime.Stream("serial", 1, self.bitmap),
            )
        )
        for action in recipe:
            spec = graph_task.ACTION_CATALOG[action]
            state = state.apply(
                spec.runtime_action, ["s0"], input_view=spec.input_view,
                io_variant=spec.io_variant,
            )
        return state.stop()

    def _measurement(self, recipe: tuple[int, ...], *, tag: str, encode_ns: int | None = None):
        frame = self._frame(recipe)
        frame_path = self.root / f"{tag}.tgf"
        frame_path.write_bytes(frame)
        compression_ns = encode_ns or 1024
        decompression_ns = 768
        row = {
            "status": "passed",
            "source_split": "train",
            "dataset_id": "training-dataset",
            "dataset_provenance_sha256": self.provenance,
            "block_id": "train-block-0",
            "dtype": "int64",
            "values_sha256": _sha(self.values),
            "missing_bitmap_sha256": _sha(self.bitmap),
            "recipe_action_ids": list(recipe),
            "operator_graph": {
                "registry_fingerprint": graph_task.REGISTRY_FINGERPRINT,
                "typed_graph_registry": graph_runtime.CURRENT_GRAPH_REGISTRY,
                "recipe_action_ids": list(recipe),
                "independent_cold_validation": True,
                "complete_frame_includes_all_root_flows": True,
            },
            "complete_frame_path": str(frame_path),
            "complete_frame_sha256": _sha(frame),
            "complete_frame_bytes": len(frame),
            "original_numeric_bytes": len(self.values),
            "compression_time_ns": compression_ns,
            "decompression_time_ns": decompression_ns,
            "rho_complete_bytes_per_original_numeric_byte": (
                len(frame) / len(self.values)
            ),
            "E_compression_ns_per_original_numeric_byte": (
                compression_ns / len(self.values)
            ),
            "D_decompression_ns_per_original_numeric_byte": (
                decompression_ns / len(self.values)
            ),
        }
        row["measurement_sha256"] = builder._measurement_digest(row)
        return row

    def _receipt(self, rows, *, name="receipt.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps({
            "schema": builder.RECEIPT_SCHEMA,
            "source_scope": "training_only",
            "action_registry_fingerprint": graph_task.REGISTRY_FINGERPRINT,
            "measurements": rows,
        }), encoding="utf-8")
        return path

    def _build(self, receipt: Path):
        with mock.patch.object(
            builder.preparation, "verify_prepared_dataset",
            return_value={"dataset_id": "training-dataset"},
        ):
            return builder.build_library(
                [self.prepared], [receipt], self.reference_path
            )

    def test_build_deduplicates_recipe_and_factory_selects_deterministically(self):
        delta_fast = self._measurement((self.delta,), tag="delta-fast", encode_ns=128)
        delta_repeat = dict(delta_fast)
        delta_repeat["measurement_sha256"] = builder._measurement_digest(delta_repeat)
        zigzag = self._measurement((self.zigzag,), tag="zigzag", encode_ns=12800)
        document = self._build(self._receipt([delta_fast, delta_repeat, zigzag]))
        self.assertEqual(document["schema"], factory_module.SOLUTION_LIBRARY_SCHEMA)
        self.assertEqual(document["library_format"], factory_module.SOLUTION_LIBRARY_FORMAT)
        self.assertEqual(document["counts"]["unique_dtype_recipe_pairs"], 2)
        delta_recipe = next(
            recipe for recipe in document["recipes"]
            if recipe["recipe_action_ids"] == [self.delta]
        )
        self.assertEqual(len(delta_recipe["bucket_evidence"]), 1)
        self.assertEqual(len(delta_recipe["bucket_evidence"][0]["measurements"]), 2)

        library = self.root / "library.json"
        builder.write_immutable(library, document)
        production = factory_module.ProductionTaskFactory(
            library, independent_cold_validation=False,
            require_formal_training_ready=False,
        )
        validation = builder.validate_library_manifest(library)
        self.assertEqual(validation["recipe_count"], 2)
        self.assertEqual(validation["action_registry_fingerprint"],
                         graph_task.REGISTRY_FINGERPRINT)
        for weight in ((0.0, 1.0, 0.0), (0.2, 0.7, 0.1)):
            request = self._request(weight=weight)
            with mock.patch.object(
                graph_task, "mechanism_probe_report", return_value=self._probe_report()
            ):
                first = production(request)
                second = production(request)
            self.assertEqual(first.previous_recipe, (self.delta,))
            self.assertEqual(first.previous_recipe, second.previous_recipe)
            self.assertIsNotNone(first.solution_library_feature_bucket_id)
            self.assertIsInstance(first.solution_library_selection_objective, float)

    def _request(self, *, weight=(0.4, 0.3, 0.3)):
        return training.TrainingTaskFactoryRequest(
            stage="train", dataset_id="training-dataset",
            dataset_manifest_path=str(self.manifest_path),
            prepared_root=str(self.prepared), provenance_sha256=self.provenance,
            block_id="train-block-0", entity_id="entity", split="train",
            column="value", dtype="int64", block_index=0,
            series_block_ordinal=0, values_path=str(self.values_path),
            missing_bitmap_path=str(self.bitmap_path),
            values_sha256=_sha(self.values), missing_bitmap_sha256=_sha(self.bitmap),
            original_numeric_bytes=len(self.values),
            prepared_block_bytes=len(self.values) + len(self.bitmap),
            compression_reference_ns_per_byte=10.0,
            decompression_reference_ns_per_byte=5.0,
            weight=weight, seed=0, iteration=0, group_index=0, episode_index=0,
            previous_recipe=(), previous_recipe_source_block_id=None,
            evaluation_budget=64, max_local_edits=None,
            reuse_previous_recipe=False,
            start_recipe_mode="training_solution_library_only",
            expected_action_registry_fingerprint=graph_task.REGISTRY_FINGERPRINT,
        )

    def _probe_report(self):
        rows = []
        keys = []
        for action in (self.delta, self.zigzag):
            key = graph_task._probe_key(graph_task.ACTION_CATALOG[action])
            keys.append(key)
            rows.append({"probe_key": key, "passed": True, "output_flow_count": 1})
        return {
            "passed_action_probe_keys": keys,
            "mechanisms": [{"action_probes": rows}],
        }

    def test_rejects_validation_leakage(self):
        row = self._measurement((self.delta,), tag="leak")
        row["source_split"] = "validation"
        row["measurement_sha256"] = builder._measurement_digest(row)
        with self.assertRaisesRegex(builder.SolutionLibraryBuildError, "TRAIN-only"):
            self._build(self._receipt([row]))

    def test_rejects_stale_registry(self):
        row = self._measurement((self.delta,), tag="stale")
        receipt = self._receipt([row])
        document = json.loads(receipt.read_text())
        document["action_registry_fingerprint"] = "stale"
        receipt.write_text(json.dumps(document))
        with self.assertRaisesRegex(builder.SolutionLibraryBuildError, "stale action registry"):
            self._build(receipt)

    def test_rejects_bad_hash(self):
        row = self._measurement((self.delta,), tag="bad-hash")
        row["complete_frame_sha256"] = "0" * 64
        row["measurement_sha256"] = builder._measurement_digest(row)
        with self.assertRaisesRegex(builder.SolutionLibraryBuildError, "frame hash/size"):
            self._build(self._receipt([row]))

    def test_rejects_wrong_recipe(self):
        row = self._measurement((self.delta,), tag="wrong-recipe")
        row["recipe_action_ids"] = [self.zigzag]
        row["operator_graph"]["recipe_action_ids"] = [self.zigzag]
        row["measurement_sha256"] = builder._measurement_digest(row)
        with self.assertRaisesRegex(builder.SolutionLibraryBuildError, "embedded graph"):
            self._build(self._receipt([row]))

    def test_rejects_stop_only_placeholder(self):
        row = self._measurement((self.delta,), tag="stop-only")
        row["recipe_action_ids"] = []
        row["operator_graph"]["recipe_action_ids"] = []
        row["measurement_sha256"] = builder._measurement_digest(row)
        with self.assertRaisesRegex(builder.SolutionLibraryBuildError, "STOP-only"):
            self._build(self._receipt([row]))

    def test_factory_rejects_tampered_extended_manifest(self):
        row = self._measurement((self.delta,), tag="tamper")
        document = self._build(self._receipt([row]))
        document["default_recipe_by_dtype"]["int64"] = "not-a-recipe"
        library = self.root / "tampered.json"
        library.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(
            factory_module.ProductionTaskFactoryError, "content fingerprint"
        ):
            factory_module.ProductionTaskFactory(library)


if __name__ == "__main__":
    unittest.main()
