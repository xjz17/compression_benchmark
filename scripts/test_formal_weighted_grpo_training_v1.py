#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import formal_weighted_grpo_training_v1 as driver
import weight_conditioned_grpo_v1 as grpo


REGISTRY = "toy-action-registry-v1"


class _ToyTask:
    action_count = 3
    stop_action = 2
    max_steps = 2

    def __init__(self, request: driver.TrainingTaskFactoryRequest):
        self.request = request
        self.task_id = request.block_id
        self.registry_fingerprint = REGISTRY
        self.data_fingerprint = request.values_sha256 + request.missing_bitmap_sha256
        self.root_fingerprint = "root:" + request.block_id

    def task_features(self) -> np.ndarray:
        return np.asarray([1.0, self.request.original_numeric_bytes / 64.0])

    def initial_state(self) -> tuple[int, ...]:
        return tuple(self.request.previous_recipe)

    def state_features(self, state: tuple[int, ...]) -> np.ndarray:
        return np.asarray([float(len(state)), float(state[-1] + 1 if state else 0)])

    def legal_actions(self, state: tuple[int, ...]):
        return (0, 1, 2) if not state else (2,)

    def transition(self, state: tuple[int, ...], action: int):
        return state + (int(action),) if action in (0, 1) and not state else None

    def evaluate_terminal(self, state: tuple[int, ...]) -> grpo.TerminalOutcome:
        choice = state[0] if state else -1
        size = {0: 4, 1: 7, -1: 11}[choice]
        encode = {0: 40, 1: 12, -1: 30}[choice]
        decode = {0: 32, 1: 10, -1: 20}[choice]
        payload = bytes([choice + 1]) * size
        return grpo.TerminalOutcome(
            operator_graph={"recipe_action_ids": list(state)},
            compressed_payload=payload,
            metrics=grpo.CompressionMetrics(
                original_numeric_bytes=self.request.original_numeric_bytes,
                compression_time_ns=encode,
                decompression_time_ns=decode,
                compression_reference_ns_per_byte=(
                    self.request.compression_reference_ns_per_byte
                ),
                decompression_reference_ns_per_byte=(
                    self.request.decompression_reference_ns_per_byte
                ),
            ),
        )


def _toy_factory(request: driver.TrainingTaskFactoryRequest) -> _ToyTask:
    return _ToyTask(request)


def _toy_solution_library_identity():
    path = Path(__file__).resolve()
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "action_registry_fingerprint": REGISTRY,
    }


_toy_factory.solution_library_identity = _toy_solution_library_identity


def _block(split: str, ordinal: int) -> driver.TrainingBlock:
    # Source indices intentionally begin above zero and need not equal the
    # sequence ordinal used by the continuation protocol.
    index = 7 + ordinal * 4
    basis = f"{split}:{index}".encode()
    digest = hashlib.sha256(basis).hexdigest()
    return driver.TrainingBlock(
        dataset_id="toy", dataset_manifest_path="/tmp/toy/dataset_manifest.json",
        prepared_root="/tmp/toy", provenance_sha256="p" * 64,
        block_id=f"toy-{split}-{index}", entity_id="e", split=split,
        column="x", dtype="int32", block_index=index,
        values_path=f"/tmp/toy/{split}-{index}.values",
        missing_bitmap_path=f"/tmp/toy/{split}-{index}.bitmap",
        values_sha256=digest, missing_bitmap_sha256="b" * 64,
        original_numeric_bytes=32, prepared_block_bytes=33,
        compression_reference_ns_per_byte=2.0,
        decompression_reference_ns_per_byte=1.0,
        block_fingerprint=digest,
    )


def _corpus() -> driver.TrainingCorpus:
    return driver.TrainingCorpus(
        train_blocks=(_block("train", 0), _block("train", 1)),
        validation_blocks=(_block("validation", 0), _block("validation", 1)),
        dataset_provenance_sha256=(("toy", "p" * 64),),
        calibration_sha256="c" * 64,
        corpus_sha256="d" * 64,
    )


def _gate() -> driver.GateEvidence:
    return driver.GateEvidence(
        source_fingerprint="source-v1", action_registry_fingerprint=REGISTRY,
        receipt_sha256=tuple((name, name[0] * 64)
                             for name in sorted(driver.REQUIRED_GATE_ASSERTIONS)),
        bundle_sha256="e" * 64,
    )


def _config(max_iterations: int) -> driver.TrainingConfig:
    return driver.TrainingConfig(
        seeds=(0, 1, 2), per_seed_budget_seconds=120.0,
        measurement_workers=2, validation_interval=1,
        checkpoint_interval=1, max_iterations=max_iterations,
    )


class FormalWeightedGRPOTrainingTest(unittest.TestCase):
    def test_supervised_phases_preserve_last_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = dict(corpus=_corpus(), gate=_gate(), factory=_toy_factory,
                        output_dir=root, seed=0, config=_config(1))
            with self.assertRaisesRegex(driver.TrainingAdmissionError, "initialized"):
                driver.train_one_seed(**args, require_initialized=True)
            initial = driver.train_one_seed(**args, initialize_only=True)
            self.assertEqual(initial["status"], "initialized")
            self.assertEqual(initial["completed_iterations"], 0)
            self.assertEqual(initial["cumulative_training_wall_time_ns"], 0)
            trained = driver.train_one_seed(**args, require_initialized=True)
            self.assertEqual(trained["completed_iterations"], 1)
            resume = (root / "resume_checkpoint.json").read_bytes()
            validation = driver.validate_last_complete_seed(**args)
            self.assertEqual(validation["optimizer_steps"], 0)
            self.assertFalse(validation["validation_in_training_budget"])
            self.assertFalse(validation["selection_used_test_data"])
            self.assertTrue(validation["model_parameters_frozen"])
            self.assertEqual(validation["completed_iterations"], 1)
            self.assertEqual((root / "resume_checkpoint.json").read_bytes(), resume)

    def test_gate_bundle_requires_all_explicit_evidence(self) -> None:
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptors = {}
            for gate, assertions in driver.REQUIRED_GATE_ASSERTIONS.items():
                receipt = {
                    "schema": driver.GATE_RECEIPT_SCHEMA,
                    "gate": gate, "passed": True,
                    "assertions": {name: True for name in assertions},
                }
                if gate in {"mechanisms", "graph"}:
                    receipt["source_fingerprint"] = "source-v1"
                    receipt["action_registry_fingerprint"] = REGISTRY
                if gate == "data":
                    receipt["dataset_provenance_sha256"] = {"toy": "p" * 64}
                if gate == "reference":
                    receipt["calibration_sha256"] = "c" * 64
                if gate in {"graph", "reference"}:
                    receipt["measurement_contract"] = dict(driver.experiment.FORMAL_MEASUREMENT_CONTRACT)
                if gate == "graph":
                    receipt["measurement_protocol_schema"] = driver.experiment.FORMAL_MEASUREMENT_PROTOCOL_SCHEMA
                    receipt["compression_timing_boundary"] = driver.experiment.FORMAL_TIMING_BOUNDARY
                    receipt["decompression_timing_boundary"] = driver.experiment.FORMAL_TIMING_BOUNDARY
                path = root / f"{gate}.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")
                descriptors[gate] = {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({
                "schema": driver.GATE_BUNDLE_SCHEMA, "receipts": descriptors,
            }), encoding="utf-8")
            with self.assertRaisesRegex(driver.TrainingAdmissionError, "hybrid"):
                driver._load_gate_bundle(bundle, corpus)
            import weight_conditioned_graph_task_v1 as actual_runtime
            with mock.patch.object(
                actual_runtime, "MEASUREMENT_PROTOCOL_SCHEMA",
                driver.experiment.FORMAL_MEASUREMENT_PROTOCOL_SCHEMA,
            ), mock.patch.object(
                actual_runtime, "FORMAL_MEASUREMENT_CONTRACT",
                dict(driver.experiment.FORMAL_MEASUREMENT_CONTRACT), create=True,
            ):
                evidence = driver._load_gate_bundle(bundle, corpus)
            self.assertEqual(evidence.action_registry_fingerprint, REGISTRY)
            mechanism_path = root / "mechanisms.json"
            bad = json.loads(mechanism_path.read_text(encoding="utf-8"))
            bad["assertions"]["cold_bit_exact_roundtrip"] = False
            mechanism_path.write_text(json.dumps(bad), encoding="utf-8")
            descriptors["mechanisms"]["sha256"] = hashlib.sha256(
                mechanism_path.read_bytes()
            ).hexdigest()
            bundle.write_text(json.dumps({
                "schema": driver.GATE_BUNDLE_SCHEMA, "receipts": descriptors,
            }), encoding="utf-8")
            with self.assertRaises(driver.TrainingAdmissionError):
                driver._load_gate_bundle(bundle, corpus)

    def test_one_seed_trains_with_fixed_and_continuous_weight_halves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt = driver.train_one_seed(
                corpus=_corpus(), gate=_gate(), factory=_toy_factory,
                output_dir=Path(temporary), seed=0, config=_config(1),
            )
            self.assertEqual(receipt["completed_iterations"], 1)
            self.assertFalse(receipt["test_data_used_for_selection"])
            row = receipt["trace"][0]
            self.assertEqual(row["fixed_weight_groups"], 4)
            self.assertEqual(row["continuous_simplex_groups"], 4)
            continuous = [
                detail["weight"] for detail in row["group_detail"]
                if detail["weight_source"] == "dirichlet"
            ]
            self.assertEqual(len(continuous), 4)
            self.assertTrue(all(abs(sum(weight) - 1.0) < 1.0e-12 for weight in continuous))
            self.assertTrue(all(tuple(weight) not in grpo.FIXED_WEIGHTS for weight in continuous))
            self.assertTrue(Path(receipt["best_policy_checkpoint"]).is_file())
            self.assertEqual(receipt["validations"][0]["weight_count"], 6)
            self.assertEqual(
                receipt["run_identity"]["solution_library_identity"],
                _toy_solution_library_identity(),
            )
            loaded = grpo.load_checkpoint(Path(receipt["best_policy_checkpoint"]))
            self.assertEqual(
                loaded.extra["solution_library_identity"],
                _toy_solution_library_identity(),
            )

    def test_nonzero_source_index_and_two_block_recipe_reuse(self) -> None:
        corpus = _corpus()
        gate = _gate()
        config = _config(1)
        sequences = driver._ordered_sequences(corpus.train_blocks)
        self.assertEqual(len(sequences), 1)
        self.assertEqual([block.block_index for block in sequences[0].blocks], [7, 11])
        identity = driver._build_policy_identity(corpus, _toy_factory, gate, config)
        frozen = grpo.FrozenPolicy(grpo.TwoHiddenMLP(identity, np.random.default_rng(2)))
        rollout = driver._sequence_rollout_job(
            _toy_factory, sequences[0], (0.4, 0.3, 0.3), frozen,
            seed=2, iteration=1, group_index=0, episode_index=0,
            rollout_seed=9, registry_fingerprint=REGISTRY,
        )
        first, second = rollout["sequence_receipt"]
        self.assertEqual(first["series_block_ordinal"], 0)
        self.assertEqual(first["source_block_index"], 7)
        self.assertEqual(first["evaluation_budget"], 64)
        self.assertFalse(first["reuse_previous_recipe"])
        self.assertIsNone(first["previous_recipe_source_block_id"])
        self.assertEqual(second["series_block_ordinal"], 1)
        self.assertEqual(second["source_block_index"], 11)
        self.assertEqual(second["evaluation_budget"], 16)
        self.assertEqual(second["max_local_edits"], 2)
        self.assertTrue(second["reuse_previous_recipe"])
        self.assertEqual(second["previous_recipe_source_block_id"], first["block_id"])
        self.assertEqual(second["previous_recipe"], first["selected_complete_recipe"])
        self.assertFalse(first["future_block_data_visible"])
        self.assertFalse(second["future_block_data_visible"])

        validation = driver._validate_policy(
            policy=frozen, factory=_toy_factory, corpus=corpus, seed=2,
            iteration=1, config=config, registry_fingerprint=REGISTRY,
        )
        self.assertEqual(validation["validation_sequence_count"], 1)
        self.assertEqual(
            validation["checkpoint_selection_inference_protocol"],
            "frozen_policy_budgeted_candidate_search_64_first_16_continuation",
        )
        rows = validation["sequence_rows"][0]["rows"]
        self.assertTrue(all(
            row["selection_protocol"] == "frozen_policy_budgeted_candidate_search"
            for row in rows
        ))
        self.assertGreater(rows[0]["evaluated_complete_candidates"], 1)
        self.assertEqual(rows[0]["evaluation_budget"], 64)
        self.assertEqual(rows[1]["evaluation_budget"], 16)
        self.assertEqual([row["series_block_ordinal"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["previous_recipe"], rows[0]["selected_complete_recipe"])
        self.assertEqual(rows[1]["previous_recipe_source_block_id"], rows[0]["block_id"])

    def test_failed_block_does_not_open_a_future_block(self) -> None:
        corpus = _corpus()
        gate = _gate()
        config = _config(1)
        three = driver._ordered_sequences((*corpus.train_blocks, _block("train", 2)))[0]
        identity = driver._build_policy_identity(corpus, _toy_factory, gate, config)
        frozen = grpo.FrozenPolicy(grpo.TwoHiddenMLP(identity, np.random.default_rng(4)))
        opened: list[str] = []

        def failing_factory(request: driver.TrainingTaskFactoryRequest) -> _ToyTask:
            opened.append(request.block_id)
            if request.series_block_ordinal == 1:
                raise RuntimeError("injected current-block failure")
            return _ToyTask(request)

        with self.assertRaises(RuntimeError):
            driver._sequence_rollout_job(
                failing_factory, three, (1.0, 0.0, 0.0), frozen,
                seed=4, iteration=1, group_index=0, episode_index=0,
                rollout_seed=5, registry_fingerprint=REGISTRY,
            )
        self.assertEqual(opened, [three.blocks[0].block_id, three.blocks[1].block_id])
        self.assertNotIn(three.blocks[2].block_id, opened)

    def test_resume_matches_uninterrupted_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = root / "full"
            resumed = root / "resumed"
            full_receipt = driver.train_one_seed(
                corpus=_corpus(), gate=_gate(), factory=_toy_factory,
                output_dir=full, seed=0, config=_config(2),
            )
            first = driver.train_one_seed(
                corpus=_corpus(), gate=_gate(), factory=_toy_factory,
                output_dir=resumed, seed=0, config=_config(1),
            )
            self.assertEqual(first["completed_iterations"], 1)
            resumed_receipt = driver.train_one_seed(
                corpus=_corpus(), gate=_gate(), factory=_toy_factory,
                output_dir=resumed, seed=0, config=_config(2),
            )
            self.assertEqual(resumed_receipt["completed_iterations"], 2)
            full_state = json.loads((full / "resume_checkpoint.json").read_text())
            resumed_state = json.loads((resumed / "resume_checkpoint.json").read_text())
            self.assertEqual(
                full_state["payload"]["policy_parameters"],
                resumed_state["payload"]["policy_parameters"],
            )
            self.assertEqual(
                full_state["payload"]["optimizer_first"],
                resumed_state["payload"]["optimizer_first"],
            )
            self.assertEqual(
                full_state["payload"]["optimizer_second"],
                resumed_state["payload"]["optimizer_second"],
            )
            self.assertEqual(
                full_state["payload"]["rng_state"],
                resumed_state["payload"]["rng_state"],
            )

    def test_formal_budget_is_three_seeds_by_sixteen_hours(self) -> None:
        config = driver.TrainingConfig()
        self.assertEqual(config.seeds, (0, 1, 2))
        self.assertEqual(config.per_seed_budget_seconds, 16 * 60 * 60)
        self.assertEqual(
            len(config.seeds) * config.per_seed_budget_seconds, 48 * 60 * 60
        )
        self.assertEqual(config.measurement_workers, 4)

    def test_campaign_selects_delivery_only_from_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = driver.run_campaign(
                corpus=_corpus(), gate=_gate(), factory=_toy_factory,
                output_dir=Path(temporary), config=_config(1),
            )
            self.assertIn(campaign["selected_seed"], (0, 1, 2))
            self.assertFalse(campaign["selection_used_test_data"])
            self.assertTrue(Path(campaign["delivery_checkpoint"]).is_file())
            self.assertEqual(len(campaign["seed_receipts"]), 3)
            self.assertEqual(
                campaign["selected_validation_score"],
                min(row["best_validation_score"] for row in campaign["seed_receipts"]),
            )
            self.assertEqual(
                campaign["solution_library_identity"],
                _toy_solution_library_identity(),
            )
            delivery = grpo.load_checkpoint(Path(campaign["delivery_checkpoint"]))
            self.assertEqual(
                delivery.extra["solution_library_identity"],
                campaign["solution_library_identity"],
            )

    def test_training_rejects_drifted_solution_library_identity(self) -> None:
        def bad_factory(request):
            return _toy_factory(request)

        bad_factory.solution_library_identity = lambda: {
            **_toy_solution_library_identity(),
            "file_sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                driver.TrainingAdmissionError, "bytes changed"
            ):
                driver.train_one_seed(
                    corpus=_corpus(), gate=_gate(), factory=bad_factory,
                    output_dir=Path(temporary), seed=0, config=_config(1),
                )


if __name__ == "__main__":
    unittest.main()
