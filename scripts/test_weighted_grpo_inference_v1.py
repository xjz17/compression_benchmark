#!/usr/bin/env python3
"""Focused tests for frozen single-block weighted-GRPO inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import weighted_grpo_experiment_v1 as experiment  # noqa: E402
import weighted_grpo_inference_v1 as inference  # noqa: E402
from test_weighted_grpo_experiment_v1 import RecordingFactory, ToyTask  # noqa: E402
from weight_conditioned_grpo_v1 import (  # noqa: E402
    GRPOConfig,
    TrainingResult,
    initialize_policy,
    save_checkpoint,
)
from weighted_objective_pruning_v1 import BoundState  # noqa: E402


def _zero_bound(_task, _state):
    return BoundState(
        original_numeric_bytes=8,
        fixed_frame_bytes=0,
        mandatory_metadata_bytes=0,
        exhausted_leaf_logical_bits=(),
    )


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class FrozenInferenceTest(unittest.TestCase):
    def _fixture(self, root: Path):
        prepared = root / "prepared"
        blocks_dir = prepared / "blocks"
        blocks_dir.mkdir(parents=True)
        blocks = []
        for index in range(2):
            values = bytes([index + 1]) * 8
            bitmap = b"\x00"
            values_path = f"blocks/values-{index}.bin"
            bitmap_path = f"blocks/bitmap-{index}.bin"
            (prepared / values_path).write_bytes(values)
            (prepared / bitmap_path).write_bytes(bitmap)
            blocks.append({
                "block_id": f"test__entity__value__{index}",
                "entity_id": "entity",
                "split": "test",
                "column": "value",
                "source_column": "value",
                "dtype": "int64",
                "block_index": index,
                "values_path": values_path,
                "values_bytes": 8,
                "values_sha256": hashlib.sha256(values).hexdigest(),
                "missing_bitmap_path": bitmap_path,
                "missing_bitmap_bytes": 1,
                "missing_bitmap_sha256": hashlib.sha256(bitmap).hexdigest(),
                "original_numeric_bytes": 8,
                "prepared_block_bytes": 9,
            })
        manifest = {
            "dataset_id": "toy-inference",
            "provenance_sha256": "a" * 64,
            "blocks": blocks,
        }
        manifest_path = prepared / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        reference = {
            "schema": "weighted-grpo-zlib6-reference-table-v1",
            "objective_normalization": "E/(E+E0), D/(D+D0)",
            "reference_method": "native zlib level 6 complete ZR61 frame",
            "references": [{
                "dtype": "int64",
                "block_size_bucket_bytes": 16,
                "E0_compression_ns_per_original_byte": 2.0,
                "D0_decompression_ns_per_original_byte": 3.0,
            }],
        }
        reference["calibration_sha256"] = hashlib.sha256(
            _canonical_json(reference)
        ).hexdigest()
        reference_path = root / "reference.json"
        reference_path.write_text(json.dumps(reference), encoding="utf-8")

        request = experiment.TaskFactoryRequest(
            stage="online_selection", dataset_id="training-toy",
            dataset_manifest_path="/tmp/dataset_manifest.json", prepared_root="/tmp",
            provenance_sha256="b" * 64, block_id="train-block", entity_id="entity",
            split="train", column="value", dtype="int64", block_index=0,
            series_block_ordinal=0, values_path="/tmp/values",
            missing_bitmap_path="/tmp/bitmap", values_sha256="c" * 64,
            missing_bitmap_sha256="d" * 64, original_numeric_bytes=8,
            prepared_block_bytes=9, compression_reference_ns_per_byte=2.0,
            decompression_reference_ns_per_byte=3.0, method="grpo",
            weight=(1.0, 0.0, 0.0), previous_recipe=(), selected_recipe=(),
            max_local_edits=None, evaluation_budget=64,
            reuse_previous_recipe=False, ablation=experiment.AblationConfig(),
        )
        policy = initialize_policy([ToyTask(request)], seed=3, hidden_sizes=(6, 5))
        checkpoint = root / "checkpoint.json"
        save_checkpoint(
            TrainingResult(
                policy, (), GRPOConfig(hidden_sizes=(6, 5), groups_per_iteration=2),
                seed=3, completed_iterations=1,
            ),
            checkpoint,
        )
        return manifest_path, reference_path, checkpoint, blocks

    @staticmethod
    def _verification(blocks):
        return {
            "dataset_id": "toy-inference", "blocks": len(blocks),
            "original_numeric_bytes": 16, "provenance_sha256": "a" * 64,
            "source_verified": False,
        }

    def test_arbitrary_weight_writes_independent_complete_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, reference, checkpoint, blocks = self._fixture(root)
            factory = RecordingFactory()
            output = root / "compressed" / "frame.bin"
            with mock.patch.object(
                experiment.preparation, "verify_prepared_dataset",
                return_value=self._verification(blocks),
            ):
                document = inference.run_single_block_inference(
                    prepared, blocks[0]["block_id"], reference, factory,
                    checkpoint, (0.2, 0.3, 0.5), output,
                    verify_source=False,
                    bound_provider=_zero_bound,
                )

            payload = output.read_bytes()
            artifact = document["final_independent_encoding"][
                "complete_payload_artifact"
            ]
            self.assertEqual(document["status"], "complete")
            self.assertEqual(document["weight"], [0.2, 0.3, 0.5])
            self.assertEqual(artifact["bytes"], len(payload))
            self.assertEqual(artifact["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(len(factory.requests), 2)
            self.assertEqual(factory.requests[0].evaluation_budget, 64)
            self.assertEqual(factory.requests[1].stage, "final_independent_encoding")
            timing = document["online_selection"]["prediction_timing_ns"]
            self.assertEqual(
                timing["total_prediction_time_ns"],
                timing["online_task_factory_time_ns"]
                + timing["search_only_total_prediction_time_ns"],
            )

    def test_continuation_uses_16_evaluations_and_two_edit_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, reference, checkpoint, blocks = self._fixture(root)
            first_factory = RecordingFactory()
            with mock.patch.object(
                experiment.preparation, "verify_prepared_dataset",
                return_value=self._verification(blocks),
            ):
                previous = inference.run_single_block_inference(
                    prepared, blocks[0]["block_id"], reference, first_factory,
                    checkpoint, (0.4, 0.3, 0.3), root / "first.bin",
                    verify_source=False,
                    bound_provider=_zero_bound,
                )
                previous_receipt = root / "first-result.json"
                experiment.write_result_atomic(previous_receipt, previous)
                factory = RecordingFactory()
                document = inference.run_single_block_inference(
                    prepared, blocks[1]["block_id"], reference, factory,
                    checkpoint, (0.4, 0.3, 0.3), root / "later.bin",
                    previous_recipe=previous["selected_recipe_action_ids"],
                    previous_block_id=blocks[0]["block_id"],
                    previous_result_receipt=previous_receipt,
                    previous_result_sha256=previous["result_sha256"],
                    series_block_ordinal=1, verify_source=False,
                    bound_provider=_zero_bound,
                )

            request = factory.requests[0]
            self.assertEqual(request.evaluation_budget, 16)
            self.assertEqual(request.max_local_edits, 2)
            self.assertTrue(request.reuse_previous_recipe)
            self.assertEqual(
                request.previous_recipe,
                tuple(previous["selected_recipe_action_ids"]),
            )
            self.assertTrue(document["continuation"]["enabled"])
            evidence = document["continuation"]["previous_result_receipt"]
            self.assertEqual(
                evidence["verified_result_sha256"], previous["result_sha256"]
            )

    def test_continuation_receipt_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, reference, checkpoint, blocks = self._fixture(root)
            with mock.patch.object(
                experiment.preparation, "verify_prepared_dataset",
                return_value=self._verification(blocks),
            ):
                previous = inference.run_single_block_inference(
                    prepared, blocks[0]["block_id"], reference, RecordingFactory(),
                    checkpoint, (0.4, 0.3, 0.3), root / "first.bin",
                    verify_source=False, bound_provider=_zero_bound,
                )
                receipt = root / "first-result.json"
                experiment.write_result_atomic(receipt, previous)
                cases = (
                    {
                        "previous_result_sha256": "0" * 64,
                        "previous_recipe": previous["selected_recipe_action_ids"],
                    },
                    {
                        "previous_result_sha256": previous["result_sha256"],
                        "previous_recipe": [0],
                    },
                )
                for case in cases:
                    factory = RecordingFactory()
                    with self.subTest(case=case), self.assertRaises(
                        inference.InferenceContractError
                    ):
                        inference.run_single_block_inference(
                            prepared, blocks[1]["block_id"], reference, factory,
                            checkpoint, (0.4, 0.3, 0.3), root / "rejected.bin",
                            previous_recipe=case["previous_recipe"],
                            previous_block_id=blocks[0]["block_id"],
                            previous_result_receipt=receipt,
                            previous_result_sha256=case["previous_result_sha256"],
                            series_block_ordinal=1, verify_source=False,
                            bound_provider=_zero_bound,
                        )
                    self.assertEqual(factory.requests, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
