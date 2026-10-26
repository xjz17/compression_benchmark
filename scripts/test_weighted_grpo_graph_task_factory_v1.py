#!/usr/bin/env python3
"""Focused tests for the production weighted-GRPO task-factory bridge."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import formal_weighted_grpo_training_v1 as training
import typed_graph_runtime as graph_runtime
import weighted_grpo_experiment_v1 as experiment
import weighted_grpo_graph_task_factory_v1 as factory_module
import weight_conditioned_graph_task_v1 as graph_task


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class WeightedGRPOGraphTaskFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.values = struct.pack("<8q", 3, 4, 8, 9, 15, 16, 23, 42)
        self.bitmap = b"\x00"
        self.values_path = self.prepared / "values.bin"
        self.bitmap_path = self.prepared / "missing.bitmap"
        self.values_path.write_bytes(self.values)
        self.bitmap_path.write_bytes(self.bitmap)
        self.delta = graph_task.find_action(
            "op.delta.reference_v1", input_view="int64"
        )
        library = {
            "schema": factory_module.SOLUTION_LIBRARY_SCHEMA,
            "source_scope": "training_only",
            "action_registry_fingerprint": graph_task.REGISTRY_FINGERPRINT,
            "recipes": [{
                "recipe_id": "int64-delta-from-training",
                "dtype": "int64",
                "recipe_action_ids": [self.delta],
                "source_split": "train",
                "source_block_ids": ["train-source-block"],
            }],
            "default_recipe_by_dtype": {
                "int64": "int64-delta-from-training",
            },
        }
        self.library_path = self.root / "solution_library.json"
        self.library_path.write_text(
            json.dumps(library, sort_keys=True), encoding="utf-8"
        )
        self.factory = factory_module.ProductionTaskFactory(
            self.library_path,
            independent_cold_validation=False,
            require_formal_training_ready=False,
        )

    def tearDown(self) -> None:
        factory_module._factory_from_environment.cache_clear()
        self.temporary.cleanup()

    def training_request(
        self,
        *,
        stage: str = "train",
        split: str = "train",
        source_block_index: int = 7,
        series_block_ordinal: int = 0,
        previous_recipe: tuple[int, ...] = (),
        previous_source: str | None = None,
    ) -> training.TrainingTaskFactoryRequest:
        continuation = series_block_ordinal > 0
        return training.TrainingTaskFactoryRequest(
            stage=stage,
            dataset_id="dataset",
            dataset_manifest_path=str(self.root / "not-read-manifest.json"),
            prepared_root=str(self.prepared),
            provenance_sha256="a" * 64,
            block_id=f"block-{source_block_index}",
            entity_id="entity",
            split=split,
            column="value",
            dtype="int64",
            block_index=source_block_index,
            series_block_ordinal=series_block_ordinal,
            values_path=str(self.values_path),
            missing_bitmap_path=str(self.bitmap_path),
            values_sha256=_sha256(self.values),
            missing_bitmap_sha256=_sha256(self.bitmap),
            original_numeric_bytes=len(self.values),
            prepared_block_bytes=len(self.values) + len(self.bitmap),
            compression_reference_ns_per_byte=7.0,
            decompression_reference_ns_per_byte=5.0,
            weight=(0.4, 0.3, 0.3),
            seed=0,
            iteration=1,
            group_index=0,
            episode_index=0,
            previous_recipe=previous_recipe,
            previous_recipe_source_block_id=previous_source,
            evaluation_budget=(16 if continuation else 64),
            max_local_edits=(2 if continuation else None),
            reuse_previous_recipe=continuation,
            start_recipe_mode=(
                "previous_successful_complete_recipe" if continuation
                else "training_solution_library_only"
            ),
            expected_action_registry_fingerprint=graph_task.REGISTRY_FINGERPRINT,
        )

    def experiment_request(
        self,
        *,
        stage: str = "online_selection",
        split: str = "test",
        source_block_index: int = 7,
        series_block_ordinal: int = 0,
        previous_recipe: tuple[int, ...] = (),
        selected_recipe: tuple[int, ...] = (),
        reuse: bool = False,
    ) -> experiment.TaskFactoryRequest:
        final = stage == "final_independent_encoding"
        return experiment.TaskFactoryRequest(
            stage=stage,
            dataset_id="test-dataset",
            dataset_manifest_path=str(self.root / "not-read-test-manifest.json"),
            prepared_root=str(self.prepared),
            provenance_sha256="b" * 64,
            block_id=f"test-block-{source_block_index}",
            entity_id="entity",
            split=split,
            column="value",
            dtype="int64",
            block_index=source_block_index,
            series_block_ordinal=series_block_ordinal,
            values_path=str(self.values_path),
            missing_bitmap_path=str(self.bitmap_path),
            values_sha256=_sha256(self.values),
            missing_bitmap_sha256=_sha256(self.bitmap),
            original_numeric_bytes=len(self.values),
            prepared_block_bytes=len(self.values) + len(self.bitmap),
            compression_reference_ns_per_byte=11.0,
            decompression_reference_ns_per_byte=13.0,
            method="grpo",
            weight=(1.0, 0.0, 0.0),
            previous_recipe=previous_recipe,
            selected_recipe=selected_recipe,
            max_local_edits=(0 if final else None if series_block_ordinal == 0 else 2),
            evaluation_budget=(1 if final else 64 if series_block_ordinal == 0 else 16),
            reuse_previous_recipe=reuse,
            ablation=experiment.AblationConfig(no_reuse=not reuse),
        )

    def test_training_identity_train_and_validation_use_frozen_library(self):
        for request in (
            self.training_request(stage="identity"),
            self.training_request(stage="train"),
            self.training_request(stage="validation", split="validation"),
        ):
            task = self.factory(request)
            self.assertEqual(task.previous_recipe, (self.delta,))
            self.assertEqual(task.block_index, 0)
            self.assertEqual(task.source_block_index, 7)
            self.assertEqual(task.series_block_ordinal, 0)
            self.assertEqual(task.objective_weight, (0.4, 0.3, 0.3))
            self.assertEqual(task.original_numeric_bytes, len(self.values))
            self.assertEqual(task.candidate_budget.snapshot(), (0, 64))
            self.assertTrue(task.candidate_budget.hard_limit)
            self.assertEqual(task.config.timing_warmups, 1)
            self.assertEqual(task.config.timing_repetitions, 5)
            self.assertEqual(task.solution_library_recipe_id,
                             "int64-delta-from-training")
            self.assertEqual(task.initial_state().recipe, (self.delta,))
            self.assertTrue(task.initial_state().continuation_edit_mode)
            self.assertEqual(task._max_recipe_edits, graph_task.MAX_GRAPH_NODES)
            self.assertEqual(task.max_steps, graph_task.MAX_GRAPH_NODES * 3 + 1)

    def test_training_continuation_reuses_exact_recipe_and_budget(self):
        request = self.training_request(
            source_block_index=99,
            series_block_ordinal=1,
            previous_recipe=(self.delta,),
            previous_source="block-7",
        )
        task = self.factory(request)
        self.assertEqual(task.previous_recipe, request.previous_recipe)
        self.assertEqual(task.initial_state().recipe, request.previous_recipe)
        self.assertEqual(task.block_index, 1)
        self.assertEqual(task.source_block_index, 99)
        self.assertEqual(task.candidate_budget.snapshot(), (0, 16))
        self.assertEqual(task.max_steps, 7)

    def test_test_first_block_uses_ordinal_not_nonzero_source_index(self):
        request = self.experiment_request(source_block_index=7, series_block_ordinal=0)
        task = self.factory(request)
        self.assertEqual(task.block_index, 0)
        self.assertEqual(task.source_block_index, 7)
        self.assertEqual(task.previous_recipe, (self.delta,))
        self.assertEqual(task.candidate_budget.snapshot(), (0, 64))
        self.assertEqual(task.objective_weight, (1.0, 0.0, 0.0))
        self.assertEqual(task._max_recipe_edits, graph_task.MAX_GRAPH_NODES)

    def test_first_block_can_complete_more_than_two_recipe_edits(self):
        task = self.factory(self.experiment_request())
        zigzag = graph_task.find_action(
            "op.zigzag.signed64.raw.cpp_v2", input_view="int64"
        )
        state = task.initial_state()
        for replacement in (zigzag, self.delta, zigzag):
            state = task.transition(
                state, graph_task.EDIT_TYPE_ACTIONS["replace"]
            )
            assert state is not None
            state = task.transition(state, graph_task.EDIT_POSITION_ACTIONS[0])
            assert state is not None
            state = task.transition(state, replacement)
            assert state is not None
        self.assertEqual(state.edit_count, 3)
        self.assertIn(graph_task.STOP_ACTION, task.legal_actions(state))
        outcome = task.evaluate_terminal(state)
        self.assertEqual(outcome.operator_graph["recipe_edit_limit"], 64)
        self.assertTrue(
            outcome.operator_graph["first_block_unbounded_local_edit_mode"]
        )

    def test_test_continuation_reuses_exact_previous_recipe(self):
        request = self.experiment_request(
            source_block_index=8,
            series_block_ordinal=1,
            previous_recipe=(self.delta,),
            reuse=True,
        )
        task = self.factory(request)
        self.assertEqual(task.previous_recipe, request.previous_recipe)
        self.assertEqual(task.candidate_budget.snapshot(), (0, 16))
        self.assertIsNone(task.solution_library_recipe_id)
        self.assertEqual(task._max_recipe_edits, 2)

    def test_no_reuse_ablation_restarts_only_from_training_library(self):
        request = self.experiment_request(
            source_block_index=8,
            series_block_ordinal=1,
            previous_recipe=(),
            reuse=False,
        )
        task = self.factory(request)
        self.assertEqual(task.previous_recipe, (self.delta,))
        self.assertEqual(task.solution_library_recipe_id,
                         "int64-delta-from-training")

    def test_final_independent_encoding_replays_exact_selected_recipe(self):
        request = self.experiment_request(
            stage="final_independent_encoding",
            source_block_index=8,
            series_block_ordinal=1,
            previous_recipe=(self.delta,),
            selected_recipe=(self.delta,),
        )
        task = self.factory(request)
        self.assertEqual(task.previous_recipe, request.selected_recipe)
        self.assertEqual(task.initial_state().recipe, request.selected_recipe)
        outcome = task.evaluate_terminal(task.initial_state())
        self.assertEqual(outcome.operator_graph["recipe_action_ids"], [self.delta])
        self.assertEqual(outcome.operator_graph["objective_weight"], [1.0, 0.0, 0.0])
        self.assertEqual(outcome.operator_graph["timing_warmups"], 1)
        self.assertEqual(outcome.operator_graph["timing_repetitions"], 5)
        self.assertEqual(len(outcome.operator_graph["compression_repeat_time_ns"]), 5)
        self.assertEqual(len(outcome.operator_graph["decompression_repeat_time_ns"]), 5)
        self.assertFalse(outcome.operator_graph[
            "cold_worker_launch_in_decompression_time"
        ])
        self.assertEqual(outcome.metrics.original_numeric_bytes, len(self.values))
        self.assertGreater(len(outcome.compressed_payload), 0)
        restored = graph_runtime.restore_frame(outcome.compressed_payload)
        self.assertEqual(restored["s0"].data, self.values)
        self.assertEqual(restored["s0"].width, 8)
        self.assertEqual(restored["s1"].data, self.bitmap)
        self.assertEqual(restored["s1"].kind, "serial")

    def test_payload_hash_registry_and_dtype_fail_closed(self):
        request = self.training_request()
        with self.assertRaisesRegex(
            factory_module.ProductionTaskFactoryError, "values payload fingerprint"
        ):
            self.factory(replace(request, values_sha256="0" * 64))
        with self.assertRaisesRegex(
            factory_module.ProductionTaskFactoryError, "unsupported prepared dtype"
        ):
            self.factory(replace(request, dtype="complex64"))
        with self.assertRaisesRegex(
            factory_module.ProductionTaskFactoryError, "different action registries"
        ):
            self.factory(replace(
                request, expected_action_registry_fingerprint="wrong-registry"
            ))

    def test_module_entrypoint_requires_absolute_explicit_library(self):
        factory_module._factory_from_environment.cache_clear()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                factory_module.ProductionTaskFactoryError,
                factory_module.SOLUTION_LIBRARY_ENV,
            ):
                factory_module.task_factory(self.training_request())
        factory_module._factory_from_environment.cache_clear()
        with mock.patch.dict(os.environ, {
            factory_module.SOLUTION_LIBRARY_ENV: str(self.library_path),
        }, clear=False), mock.patch.object(
            graph_task, "assert_formal_training_ready", return_value={}
        ):
            task = factory_module.task_factory(self.training_request())
            self.assertEqual(task.previous_recipe, (self.delta,))

    def test_module_entrypoint_exposes_certified_bound_provider(self):
        self.assertIs(factory_module.task_factory.bound_provider,
                      factory_module.bound_provider)
        task = self.factory(self.training_request())
        bound = factory_module.task_factory.bound_provider(task, task.initial_state())
        self.assertIsInstance(bound, graph_task.BoundState)
        self.assertGreaterEqual(bound.unavoidable_bytes, 0)


if __name__ == "__main__":
    unittest.main()
