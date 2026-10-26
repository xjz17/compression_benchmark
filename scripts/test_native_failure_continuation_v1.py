#!/usr/bin/env python3
"""Bounded regressions for native continuation failures and state refresh.

The transport fixture supplies protocol evidence only.  It is deliberately not
used as compression or timing evidence.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import formal_weighted_grpo_training_v1 as training
import native_grpo_task_v1 as bridge
import native_training_factory_v1 as native_factory
import weight_conditioned_grpo_v1 as grpo


def _block(ordinal: int) -> training.TrainingBlock:
    digest = hashlib.sha256(f"block:{ordinal}".encode()).hexdigest()
    return training.TrainingBlock(
        dataset_id="fixture", dataset_manifest_path="/fixture/manifest.json",
        prepared_root="/fixture", provenance_sha256="p" * 64,
        block_id=f"fixture-{ordinal}", entity_id="entity", split="train",
        column="value", dtype="int64", block_index=ordinal,
        values_path=f"/fixture/{ordinal}.values",
        missing_bitmap_path=f"/fixture/{ordinal}.bitmap",
        values_sha256=digest, missing_bitmap_sha256="b" * 64,
        original_numeric_bytes=8, prepared_block_bytes=9,
        compression_reference_ns_per_byte=1.0,
        decompression_reference_ns_per_byte=1.0,
        block_fingerprint=digest,
    )


def _sequence(count: int) -> training._BlockSequence:
    return training._BlockSequence(
        sequence_id="fixture-sequence", dataset_id="fixture",
        entity_id="entity", split="train", column="value",
        blocks=tuple(_block(index) for index in range(count)),
    )


def _success(recipe: list[int], objective: float = 0.25) -> dict:
    outcome = grpo.TerminalOutcome(
        operator_graph={"recipe_action_ids": recipe},
        compressed_payload=b"frame",
        metrics=grpo.CompressionMetrics(8, 8, 8, 1.0, 1.0),
    )
    return _part(outcome, objective)


def _failure(recipe: list[int], objective: float = 2.0) -> dict:
    outcome = grpo.TerminalFailure(
        operator_graph={"recipe_action_ids": recipe},
        original_numeric_bytes=8, max_frame_bytes=1 << 30,
        error="fixture recipe is inapplicable to this block",
    )
    return _part(outcome, objective)


def _part(outcome, objective: float) -> dict:
    return {
        "features": np.zeros((1, 1), dtype=np.float64),
        "legal": np.ones((1, 1), dtype=bool),
        "actions": np.zeros(1, dtype=np.int64),
        "old_logp": np.zeros(1, dtype=np.float64),
        "objective": float(objective), "outcome": outcome,
    }


class _ClosedTask:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RecordingFactory:
    def __init__(self) -> None:
        self.requests = []
        self.tasks = []

    def __call__(self, request):
        self.requests.append(request)
        task = _ClosedTask()
        self.tasks.append(task)
        return task


class ContinuationEpisodeTests(unittest.TestCase):
    def _run(self, parts: list[dict]):
        factory = _RecordingFactory()
        side_effect = list(parts)
        with mock.patch.object(training, "_validate_fresh_task"), mock.patch.object(
            training.grpo, "_rollout", side_effect=side_effect
        ):
            result = training._sequence_rollout_job(
                factory, _sequence(len(parts)), (0.4, 0.3, 0.3), object(),
                seed=1, iteration=1, group_index=0, episode_index=0,
                rollout_seed=2, registry_fingerprint="fixture-registry",
            )
        self.assertTrue(all(task.closed for task in factory.tasks))
        return factory.requests, result

    def test_first_failure_opens_budgeted_empty_continuation(self) -> None:
        requests, result = self._run([_failure([]), _success([91])])
        continuation = requests[1]
        self.assertEqual(continuation.series_block_ordinal, 1)
        self.assertEqual(continuation.evaluation_budget, 16)
        self.assertEqual(continuation.max_local_edits, 2)
        self.assertEqual(continuation.previous_recipe, ())
        self.assertIsNone(continuation.previous_recipe_source_block_id)
        self.assertFalse(continuation.reuse_previous_recipe)
        self.assertEqual(
            continuation.start_recipe_mode,
            "no_successful_predecessor_empty_recipe",
        )
        self.assertEqual(
            [row["terminal_status"] for row in result["sequence_receipt"]],
            ["failed", "ok"],
        )

    def test_failure_preserves_last_successful_recipe_and_source(self) -> None:
        requests, result = self._run([
            _success([91]), _failure([17]), _success([103]),
        ])
        failed = requests[1]
        after_failure = requests[2]
        self.assertEqual(failed.previous_recipe, (91,))
        self.assertEqual(failed.previous_recipe_source_block_id, "fixture-0")
        self.assertEqual(after_failure.previous_recipe, (91,))
        self.assertEqual(after_failure.previous_recipe_source_block_id, "fixture-0")
        self.assertTrue(after_failure.reuse_previous_recipe)
        self.assertEqual(
            [row["terminal_status"] for row in result["sequence_receipt"]],
            ["ok", "failed", "ok"],
        )


class _NativeBlock:
    dataset_id = "fixture"
    split = "train"
    provenance_sha256 = "p" * 64
    fingerprint = "native-fixture-block"
    record = {
        "block_id": "native-fixture-1", "dtype": "int64",
        "values_sha256": "v" * 64, "missing_bitmap_sha256": "m" * 64,
        "original_numeric_bytes": 8,
    }

    def roots(self):
        return [{
            "kind": "numeric", "width": 8, "logical_view": "int64",
            "original_numeric_payload": True,
            "data_hex": "0100000000000000",
        }]


class _NativeTransport:
    def __init__(self) -> None:
        self.calls = []

    @staticmethod
    def _state(*, candidate_used: int, refreshed: bool = False) -> dict:
        return {
            "ok": True, "task_handle": "task-1", "state_handle": "state-1",
            "max_steps": 7,
            "catalog": {
                "action_count": bridge.ACTION_COUNT,
                "fingerprint": bridge.REGISTRY,
                "complete_native_graph_space": True,
            },
            "task_features": ([1.0] if refreshed else [0.0]) + [0.0] * 17,
            "state_features": ([1.0] if refreshed else [0.0]) + [0.0] * 27,
            "legal_action_ids": [0, 91], "terminal_evaluable": True,
            "candidate_used": candidate_used,
        }

    def call(self, request):
        self.calls.append(request)
        command = request["cmd"]
        if command == "reset":
            return self._state(candidate_used=0)
        if command == "evaluate_terminal":
            return {
                "terminal_status": "failed", "original_numeric_bytes": 8,
                "max_frame_bytes": 1 << 30, "candidate_budget": 16,
                "candidate_ordinal": 1, "recipe_action_ids": [91],
                "error": "previous recipe is invalid on this block",
            }
        if command == "inspect":
            return self._state(candidate_used=1, refreshed=True)
        if command == "release_task":
            return {"ok": True}
        raise AssertionError(f"unexpected native command: {command}")


class FactoryRefreshTests(unittest.TestCase):
    def test_previous_recipe_is_evaluated_then_inspected_in_same_task(self) -> None:
        block = _NativeBlock()
        transport = _NativeTransport()
        factory = object.__new__(native_factory.Factory)
        factory.by_identity = {(block.dataset_id, block.record["block_id"]): block}
        factory.clients = [transport]
        factory.lock = __import__("threading").Lock()
        factory.cursor = 0
        factory.allow_large_memory = False
        request = SimpleNamespace(
            expected_action_registry_fingerprint=bridge.REGISTRY,
            dataset_id=block.dataset_id, block_id=block.record["block_id"],
            split=block.split, dtype=block.record["dtype"],
            provenance_sha256=block.provenance_sha256,
            values_sha256=block.record["values_sha256"],
            missing_bitmap_sha256=block.record["missing_bitmap_sha256"],
            original_numeric_bytes=8, stage="train", weight=(0.4, 0.3, 0.3),
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            series_block_ordinal=1, previous_recipe=(91,),
        )

        task = factory(request)
        try:
            self.assertIsInstance(task.previous_recipe_evaluation, grpo.TerminalFailure)
            self.assertEqual(task.initial_state().payload["candidate_used"], 1)
            self.assertEqual(task.task_features()[0], 1.0)
            self.assertEqual(task.state_features(task.initial_state())[0], 1.0)
            self.assertEqual(
                [call["cmd"] for call in transport.calls],
                ["reset", "evaluate_terminal", "inspect"],
            )
            self.assertEqual(transport.calls[1]["state_handle"], "state-1")
            self.assertEqual(transport.calls[2]["state_handle"], "state-1")
        finally:
            task.close()
        self.assertEqual(transport.calls[-1]["cmd"], "release_task")


if __name__ == "__main__":
    unittest.main()
