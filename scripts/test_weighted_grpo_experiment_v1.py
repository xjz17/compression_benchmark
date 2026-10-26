#!/usr/bin/env python3
"""Focused tests for the independent weighted-GRPO result protocol."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from weight_conditioned_grpo_v1 import (  # noqa: E402
    CompressionMetrics,
    FrozenPolicy,
    GRPOConfig,
    TerminalOutcome,
    TrainingResult,
    initialize_policy,
    save_checkpoint,
)
import weighted_grpo_experiment_v1 as experiment  # noqa: E402


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class ToyTask:
    registry_fingerprint = "weighted-experiment-toy-registry"
    action_count = 3
    stop_action = 2

    def __init__(self, request: experiment.TaskFactoryRequest, *, corrupt=False):
        self.request = request
        self.task_id = f"{request.block_id}:{request.method}:{request.stage}"
        self.data_fingerprint = request.values_sha256
        self.root_fingerprint = hashlib.sha256(
            canonical_json({
                "block": request.block_id,
                "previous": request.previous_recipe,
                "selected": request.selected_recipe,
            })
        ).hexdigest()
        self._root = (
            request.selected_recipe
            if request.stage == "final_independent_encoding"
            else request.previous_recipe
        )
        edits = request.max_local_edits
        self.max_steps = 1 if edits == 0 else (int(edits or 2) + 1)
        self._base_len = len(self._root)
        self.corrupt = corrupt

    def task_features(self):
        return np.asarray([1.0, self.request.block_index / 10.0])

    def initial_state(self):
        return self._root

    def state_features(self, state):
        return np.asarray([len(state) / 8.0, float(sum(state) if state else 0.0)])

    def legal_actions(self, state):
        if self.request.stage == "final_independent_encoding":
            return (self.stop_action,)
        used = len(state) - self._base_len
        if used >= int(self.request.max_local_edits or 2):
            return (self.stop_action,)
        return (0, 1, self.stop_action)

    def transition(self, state, action):
        if action not in (0, 1):
            return None
        return state + (action,)

    def evaluate_terminal(self, state):
        last = state[-1] if state else None
        complete_bytes = {None: 8, 0: 6, 1: 2}[last]
        compression_ns = {None: 80, 0: 8, 1: 40}[last]
        decompression_ns = {None: 80, 0: 16, 1: 32}[last]
        payload_byte = (sum(state) + len(state)) % 251
        if self.corrupt:
            payload_byte = (payload_byte + 1) % 251
        return TerminalOutcome(
            operator_graph={"recipe_action_ids": list(state)},
            compressed_payload=bytes([payload_byte]) * complete_bytes,
            metrics=CompressionMetrics(
                self.request.original_numeric_bytes,
                compression_ns,
                decompression_ns,
                self.request.compression_reference_ns_per_byte,
                self.request.decompression_reference_ns_per_byte,
            ),
        )


class RecordingFactory:
    def __init__(
        self, *, corrupt_final_block: str | None = None,
        solution_library_identity: dict[str, str] | None = None,
    ):
        self.requests = []
        self.corrupt_final_block = corrupt_final_block
        self._solution_library_identity = solution_library_identity

    def __call__(self, request):
        self.requests.append(request)
        corrupt = (
            request.stage == "final_independent_encoding"
            and request.block_id == self.corrupt_final_block
        )
        task = ToyTask(request, corrupt=corrupt)
        if self._solution_library_identity is not None:
            task.solution_library_path = self._solution_library_identity["path"]
            task.solution_library_sha256 = self._solution_library_identity["file_sha256"]
        return task

    def solution_library_identity(self):
        if self._solution_library_identity is None:
            raise RuntimeError("test factory has no formal solution library")
        return dict(self._solution_library_identity)


class WeightedExperimentProtocolTests(unittest.TestCase):
    def _fixture(
        self, root: Path, *, block_count=2, split="validation", start_index=0
    ):
        prepared = root / "prepared"
        blocks_dir = prepared / "blocks"
        blocks_dir.mkdir(parents=True)
        blocks = []
        for offset in range(block_count):
            index = start_index + offset
            values = bytes([index + 1]) * 8
            bitmap = b"\x00"
            values_relative = f"blocks/values-{index}.bin"
            bitmap_relative = f"blocks/bitmap-{index}.bin"
            (prepared / values_relative).write_bytes(values)
            (prepared / bitmap_relative).write_bytes(bitmap)
            blocks.append({
                "block_id": f"{split}__e-entity__c-value__b-{index:06d}",
                "entity_id": "entity",
                "split": split,
                "column": "value",
                "source_column": "value",
                "dtype": "int64",
                "block_index": index,
                "values_path": values_relative,
                "values_bytes": len(values),
                "values_sha256": hashlib.sha256(values).hexdigest(),
                "missing_bitmap_path": bitmap_relative,
                "missing_bitmap_bytes": len(bitmap),
                "missing_bitmap_sha256": hashlib.sha256(bitmap).hexdigest(),
                "original_numeric_bytes": 8,
                "prepared_block_bytes": 9,
            })
        manifest = {
            "dataset_id": "toy-dataset",
            "provenance_sha256": "a" * 64,
            "blocks": blocks,
        }
        if split == "test":
            cohort = experiment.preparation.load_cohort_manifest()
            entry = next(row for row in cohort["datasets"]
                         if row["id"] == "enedis-with-holidays")
            manifest.update({
                "dataset_id": entry["id"], "domain": entry["domain"],
                "source_group": entry["source_group"],
                "source": {"sha256": "f" * 64},
                "cohort": {
                    "version": experiment.preparation.COHORT_VERSION,
                    "assignment_sha256": experiment.preparation.COHORT_ASSIGNMENT_SHA256,
                    "policy_sha256": experiment.preparation.COHORT_POLICY_SHA256,
                    "role": "test_candidate",
                },
                "split_contract": experiment.preparation.SPLIT_CONTRACTS["test_candidate"],
            })
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
            canonical_json(reference)
        ).hexdigest()
        reference_path = root / "references.json"
        reference_path.write_text(json.dumps(reference), encoding="utf-8")
        return prepared, reference_path, blocks

    @staticmethod
    def _verification(blocks, dataset_id="toy-dataset"):
        return {
            "dataset_id": dataset_id,
            "blocks": len(blocks),
            "original_numeric_bytes": sum(
                block["original_numeric_bytes"] for block in blocks
            ),
            "provenance_sha256": "a" * 64,
            "source_verified": False,
        }

    @staticmethod
    def _policy():
        request = experiment.TaskFactoryRequest(
            stage="online_selection",
            dataset_id="training-toy",
            dataset_manifest_path="/tmp/dataset_manifest.json",
            prepared_root="/tmp",
            provenance_sha256="b" * 64,
            block_id="train-block",
            entity_id="entity",
            split="train",
            column="value",
            dtype="int64",
            block_index=0,
            series_block_ordinal=0,
            values_path="/tmp/values",
            missing_bitmap_path="/tmp/bitmap",
            values_sha256="c" * 64,
            missing_bitmap_sha256="d" * 64,
            original_numeric_bytes=8,
            prepared_block_bytes=9,
            compression_reference_ns_per_byte=2.0,
            decompression_reference_ns_per_byte=3.0,
            method="grpo",
            weight=(1.0, 0.0, 0.0),
            previous_recipe=(),
            selected_recipe=(),
            max_local_edits=None,
            evaluation_budget=64,
            reuse_previous_recipe=False,
            ablation=experiment.AblationConfig(),
        )
        policy = initialize_policy([ToyTask(request)], seed=7, hidden_sizes=(6, 5))
        policy.params[-1][:] = np.asarray([1.0, 0.5, -2.0])
        return policy

    def test_full_protocol_records_phases_reuse_statistics_and_frozen_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, references, blocks = self._fixture(root)
            factory = RecordingFactory()
            frozen = FrozenPolicy(self._policy())
            before = [value.copy() for value in frozen.params]
            with mock.patch.object(
                experiment.preparation,
                "verify_prepared_dataset",
                return_value=self._verification(blocks),
            ):
                result = experiment.run_experiment(
                    [prepared], references, factory, policy=frozen,
                    config=experiment.ExperimentConfig(verify_source=False),
                    payload_artifact_root=root / "payloads",
                )

            artifact_root = root / "payloads"
            artifact_bytes = {
                path.relative_to(artifact_root).as_posix(): path.read_bytes()
                for path in artifact_root.rglob("*.bin")
            }

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["attempted_records"], 24)
        self.assertEqual(result["successful_records"], 24)
        self.assertEqual(result["failed_records"], 0)
        self.assertEqual(result["payload_artifacts"]["successful_artifacts"], 24)
        self.assertEqual(len(artifact_bytes), 24)
        self.assertEqual(len(result["summaries"]), 12)
        self.assertTrue(result["training_provenance"]["model_parameters_unchanged"])
        self.assertEqual(result["training_provenance"]["optimizer_steps"], 0)
        for old, new in zip(before, frozen.params):
            np.testing.assert_array_equal(old, new)
            self.assertFalse(new.flags.writeable)

        later_online = [
            request for request in factory.requests
            if request.stage == "online_selection" and request.block_index == 1
        ]
        self.assertEqual(len(later_online), 12)
        self.assertTrue(all(request.max_local_edits == 2 for request in later_online))
        self.assertTrue(all(request.reuse_previous_recipe for request in later_online))
        self.assertTrue(all(request.previous_recipe for request in later_online))
        record = result["records"][0]
        self.assertIn("rho_complete_bytes_per_original_numeric_byte",
                      record["final_independent_encoding"])
        self.assertIn("candidate_evaluation_time_ns",
                      record["online_selection"]["prediction_timing_ns"])
        timing = record["online_selection"]["prediction_timing_ns"]
        self.assertEqual(
            timing["total_prediction_time_ns"],
            timing["online_task_factory_time_ns"]
            + timing["search_only_total_prediction_time_ns"],
        )
        self.assertGreaterEqual(record["online_selection"]["evaluations"], 1)
        artifact = record["final_independent_encoding"]["complete_payload_artifact"]
        self.assertEqual(artifact["bytes"], len(artifact_bytes[artifact["relative_path"]]))
        self.assertEqual(
            artifact["sha256"],
            hashlib.sha256(artifact_bytes[artifact["relative_path"]]).hexdigest(),
        )
        self.assertEqual(
            artifact["sha256"],
            record["final_independent_encoding"]["complete_payload_sha256"],
        )
        summary = result["summaries"][0]
        self.assertIn("mean", summary["metrics_across_successful_blocks"]["rho"])
        self.assertIn("p95", summary["metrics_across_successful_blocks"]["rho"])
        self.assertGreaterEqual(
            summary["combination_usage"]["distinct_selected_recipes"], 1
        )

    def test_ablation_fields_disable_reuse_and_final_mismatch_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, references, blocks = self._fixture(root)
            factory = RecordingFactory(corrupt_final_block=blocks[1]["block_id"])
            config = experiment.ExperimentConfig(
                methods=("beam30",),
                verify_source=False,
                ablation=experiment.AblationConfig(
                    no_certified_prune=True,
                    no_heuristic_order=True,
                    no_reuse=True,
                ),
            )
            with mock.patch.object(
                experiment.preparation,
                "verify_prepared_dataset",
                return_value=self._verification(blocks),
            ):
                result = experiment.run_experiment(
                    [prepared], references, factory, config=config,
                    payload_artifact_root=root / "payloads",
                )

            artifacts = list((root / "payloads").rglob("*.bin"))

        self.assertEqual(result["status"], "complete_with_failures")
        self.assertEqual(result["successful_records"], 6)
        self.assertEqual(result["failed_records"], 6)
        self.assertEqual(result["payload_artifacts"]["successful_artifacts"], 6)
        self.assertEqual(len(artifacts), 6)
        self.assertEqual(result["config"]["ablation"], {
            "no_certified_prune": True,
            "no_heuristic_order": True,
            "no_reuse": True,
        })
        self.assertFalse(result["config"]["effective_objective_guided_ordering"])
        self.assertTrue(all(
            record["online_selection"]["objective_guided_ordering"] is False
            for record in result["records"]
            if record["online_selection"] is not None
        ))
        later_online = [
            request for request in factory.requests
            if request.stage == "online_selection" and request.block_index == 1
        ]
        self.assertTrue(all(not request.previous_recipe for request in later_online))
        failures = [record for record in result["records"] if record["status"] == "failed"]
        self.assertTrue(all(
            record["failure"]["stage"] == "final_independent_encoding"
            for record in failures
        ))
        self.assertTrue(all(
            "payload differs" in record["failure"]["message"]
            for record in failures
        ))

    def test_checkpoint_is_loaded_then_frozen_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, references, blocks = self._fixture(root, block_count=1)
            policy = self._policy()
            config = GRPOConfig(hidden_sizes=(6, 5), groups_per_iteration=2)
            checkpoint = root / "policy.json"
            digest = save_checkpoint(
                TrainingResult(policy, (), config, seed=7, completed_iterations=0),
                checkpoint,
                extra={"purpose": "orchestration-test"},
            )
            with mock.patch.object(
                experiment.preparation,
                "verify_prepared_dataset",
                return_value=self._verification(blocks),
            ):
                result = experiment.run_experiment(
                    [prepared], references, RecordingFactory(),
                    checkpoint_path=checkpoint,
                    config=experiment.ExperimentConfig(
                        methods=("grpo",), verify_source=False
                    ),
                )

        training = result["training_provenance"]
        self.assertEqual(training["status"], "not_run_checkpoint_consumed")
        self.assertEqual(training["checkpoint_payload_sha256"], digest)
        self.assertEqual(training["optimizer_steps"], 0)
        self.assertTrue(training["model_parameters_frozen"])
        self.assertTrue(training["model_parameters_unchanged"])
        self.assertEqual(result["successful_records"], 6)

    def test_v1_rejects_weight_grids_outside_the_six_frozen_vectors(self):
        with self.assertRaisesRegex(ValueError, "six frozen"):
            replace(
                experiment.ExperimentConfig(),
                weights=((0.2, 0.4, 0.4),),
            )

    def test_formal_mode_rejects_outcomes_without_cold_roundtrip_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, references, blocks = self._fixture(
                root, block_count=1, split="test"
            )
            policy = self._policy()
            checkpoint = root / "delivery.json"
            library = root / "solution_library.json"
            library.write_bytes(b"solution-library")
            library_identity = {
                "path": str(library.resolve()),
                "file_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "action_registry_fingerprint": ToyTask.registry_fingerprint,
            }
            save_checkpoint(
                TrainingResult(
                    policy, (), GRPOConfig(hidden_sizes=(6, 5), groups_per_iteration=2),
                    seed=7, completed_iterations=0,
                ), checkpoint,
                extra={
                    "solution_library_identity": library_identity,
                    "run_identity": {
                        "solution_library_identity": library_identity,
                        "action_registry_fingerprint": ToyTask.registry_fingerprint,
                    },
                },
            )
            plan = {
                "schema": experiment.CAMPAIGN_PLAN_SCHEMA,
                "admission": {
                    "corpus_sha256": "a" * 64,
                    "calibration_sha256": "b" * 64,
                    "gate_bundle_sha256": "c" * 64,
                    "source_fingerprint": "d" * 64,
                    "action_registry_fingerprint": ToyTask.registry_fingerprint,
                    "test_cohort_sha256": "e" * 64,
                },
                "delivery_checkpoint": str(checkpoint),
                "solution_library_manifest": str(library),
            }
            plan["plan_sha256"] = hashlib.sha256(canonical_json(plan)).hexdigest()
            plan_path = root / "campaign_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with mock.patch.object(
                experiment.preparation, "verify_prepared_dataset",
                return_value=self._verification(blocks, "enedis-with-holidays"),
            ):
                with self.assertRaisesRegex(
                    experiment.ExperimentContractError, "cold-decode evidence"
                ):
                    experiment.run_experiment(
                        [prepared], references, RecordingFactory(
                            solution_library_identity=library_identity
                        ),
                        checkpoint_path=checkpoint,
                        config=experiment.ExperimentConfig(
                            methods=("grpo",), splits=("test",),
                            verify_source=True, fail_fast=True,
                            require_formal_evidence=True,
                        ),
                        payload_artifact_root=root / "frames",
                        campaign_plan_path=plan_path,
                    )

    def test_payload_artifacts_are_content_addressed_not_overwritten(self):
        identity = {
            "dataset_id": "d", "dataset_provenance_sha256": "a" * 64,
            "block_id": "b", "entity_id": "e", "split": "test",
            "column": "v", "dtype": "int64", "block_index": 0,
            "series_block_ordinal": 0, "method": "grpo", "weight_id": "433",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = experiment._write_payload_artifact_atomic(root, identity, b"first")
            second = experiment._write_payload_artifact_atomic(root, identity, b"second")
            self.assertNotEqual(first["relative_path"], second["relative_path"])
            self.assertEqual((root / first["relative_path"]).read_bytes(), b"first")
            self.assertEqual((root / second["relative_path"]).read_bytes(), b"second")

    def test_formal_measurement_rejects_hybrid_even_with_passing_correctness_flags(self):
        graph = {
            "schema": "weight-conditioned-graph-task-v1",
            "registry_fingerprint": "registry",
            "complete_frame_includes_all_root_flows": True,
            "independent_cold_validation": True,
            "measurement_protocol_schema": "typed-graph-timing-samples-v2",
            "measurement_contract": dict(experiment.FORMAL_MEASUREMENT_CONTRACT),
            "compression_timing_boundary": experiment.FORMAL_TIMING_BOUNDARY,
            "decompression_timing_boundary": experiment.FORMAL_TIMING_BOUNDARY,
        }
        outcome = TerminalOutcome(
            operator_graph=graph, compressed_payload=b"complete-frame",
            metrics=CompressionMetrics(64, 1000, 500, 2.0, 3.0),
        )
        # Correctness smoke still accepts this explicitly non-formal timing path.
        experiment._validate_outcome(outcome, 64, 2.0, 3.0, require_formal_evidence=False)
        with self.assertRaisesRegex(experiment.ExperimentContractError, "hybrid"):
            experiment._validate_outcome(outcome, 64, 2.0, 3.0, require_formal_evidence=True)
        graph["measurement_protocol_schema"] = experiment.FORMAL_MEASUREMENT_PROTOCOL_SCHEMA
        evidence = experiment._validate_outcome(
            outcome, 64, 2.0, 3.0, require_formal_evidence=True,
        )
        self.assertEqual(evidence["measurement_contract"], experiment.FORMAL_MEASUREMENT_CONTRACT)
        graph["decompression_timing_boundary"] = "Python cold worker first-use registry + CLI"
        with self.assertRaisesRegex(experiment.ExperimentContractError, "decompression_timing_boundary"):
            experiment._validate_outcome(outcome, 64, 2.0, 3.0, require_formal_evidence=True)
        graph["decompression_timing_boundary"] = experiment.FORMAL_TIMING_BOUNDARY
        graph["measurement_contract"] = {"native_complete_frame_timings": True}
        with self.assertRaisesRegex(experiment.ExperimentContractError, "measurement_contract"):
            experiment._validate_outcome(outcome, 64, 2.0, 3.0, require_formal_evidence=True)

    def test_campaign_binding_preserves_the_frozen_test_cohort_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            delivery = root / "delivery.json"
            delivery.write_bytes(b"delivery")
            library = root / "library.json"
            library.write_bytes(b"library")
            plan = {
                "schema": experiment.CAMPAIGN_PLAN_SCHEMA,
                "admission": {
                    "corpus_sha256": "a" * 64,
                    "calibration_sha256": "b" * 64,
                    "gate_bundle_sha256": "c" * 64,
                    "source_fingerprint": "d" * 64,
                    "action_registry_fingerprint": "e" * 64,
                    "test_cohort_sha256": "f" * 64,
                },
                "delivery_checkpoint": str(delivery),
                "solution_library_manifest": str(library),
            }
            plan["plan_sha256"] = hashlib.sha256(canonical_json(plan)).hexdigest()
            path = root / "campaign_plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            binding = experiment._load_campaign_binding(path, checkpoint_path=None)
        self.assertEqual(binding["test_cohort_sha256"], "f" * 64)
        self.assertEqual(
            binding["solution_library_file_sha256"],
            hashlib.sha256(b"library").hexdigest(),
        )

    def test_formal_checkpoint_rejects_wrong_solution_library(self):
        expected = {
            "path": "/tmp/library.json",
            "file_sha256": "a" * 64,
            "action_registry_fingerprint": ToyTask.registry_fingerprint,
        }
        training = {
            "checkpoint_extra": {
                "solution_library_identity": {**expected, "file_sha256": "b" * 64},
                "run_identity": {
                    "solution_library_identity": expected,
                    "action_registry_fingerprint": ToyTask.registry_fingerprint,
                },
            },
        }
        with self.assertRaisesRegex(
            experiment.ExperimentContractError, "different solution library"
        ):
            experiment._validate_checkpoint_solution_library(training, expected)

    def test_test_split_uses_series_ordinal_not_source_block_index_for_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, references, blocks = self._fixture(
                root, block_count=2, split="test", start_index=7
            )
            factory = RecordingFactory()
            with mock.patch.object(
                experiment.preparation,
                "verify_prepared_dataset",
                return_value=self._verification(blocks, "enedis-with-holidays"),
            ):
                result = experiment.run_experiment(
                    [prepared], references, factory,
                    config=experiment.ExperimentConfig(
                        methods=("beam30",), splits=("test",),
                        verify_source=False,
                    ),
                )

        self.assertEqual(result["status"], "complete")
        online = [
            request for request in factory.requests
            if request.stage == "online_selection"
        ]
        first = [request for request in online if request.block_index == 7]
        second = [request for request in online if request.block_index == 8]
        self.assertEqual(len(first), 6)
        self.assertEqual(len(second), 6)
        self.assertTrue(all(request.series_block_ordinal == 0 for request in first))
        self.assertTrue(all(request.evaluation_budget == 64 for request in first))
        self.assertTrue(all(request.max_local_edits is None for request in first))
        self.assertTrue(all(request.series_block_ordinal == 1 for request in second))
        self.assertTrue(all(request.evaluation_budget == 16 for request in second))
        self.assertTrue(all(request.max_local_edits == 2 for request in second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
