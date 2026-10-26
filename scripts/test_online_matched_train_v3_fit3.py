#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import online_matched_train_v3_fit3 as target
import weight_conditioned_grpo_v1 as grpo


def block(sample: str, split: str = "train", *, dataset: str = "d0",
          domain: str = "energy", raw: int = 64) -> target.TrainingBlock:
    values = bytes(index % 256 for index in range(raw))
    roots = (
        {"kind": "numeric", "width": 8, "data_hex": values.hex(),
         "logical_view": "int64", "original_numeric_payload": True},
        {"kind": "serial", "width": 1, "data_hex": "00",
         "logical_view": "bytes", "original_numeric_payload": False},
    )
    return target.TrainingBlock(sample, dataset, domain, split, "cal/" + sample,
                                target.digest(sample.encode()), roots, raw, 1.0, 1.0)


class FakeBackend:
    def __init__(self):
        self.model_sha256 = "1" * 64
        self.current_block = None
        self.weight = None
        self.recipe = []
        self.steps = 0
        self.replay_calls = 0

    def load_model(self, model: Path) -> str:
        self.model_sha256 = target.digest(model.read_bytes())
        return self.model_sha256

    def _observation(self):
        result = np.zeros(target.OBSERVATION_DIM, dtype=np.float64)
        result[0] = self.steps / 10
        result[-3:] = self.weight
        return result

    def _snapshot(self, terminal=False):
        if terminal:
            return target.Snapshot(True, None, (),
                                   {"terminal": True,
                                    "recipe_action_ids": list(self.recipe)})
        legal = (1, 2, target.FLOW_STOP_BEGIN)
        return target.Snapshot(False, self._observation(), legal,
                               {"terminal": False, "recipe_action_ids": list(self.recipe)})

    def reset(self, item, weight, contract):
        self.current_block = item
        self.weight = target._weights(weight)
        self.recipe = []
        self.steps = 0
        return self._snapshot()

    def step(self, action_id):
        if action_id not in (1, 2, target.FLOW_STOP_BEGIN):
            raise ValueError("illegal fake action")
        self.recipe.append(action_id)
        self.steps += 1
        if action_id == target.FLOW_STOP_BEGIN or self.steps >= 3:
            if self.recipe[-1] != target.FLOW_STOP_BEGIN:
                self.recipe.append(target.FLOW_STOP_BEGIN)
            return self._snapshot(True)
        return self._snapshot()

    def finish(self):
        if not self.recipe or self.recipe[-1] != target.FLOW_STOP_BEGIN:
            self.recipe.append(target.FLOW_STOP_BEGIN)
        return {"terminal": True, "recipe_action_ids": list(self.recipe)}

    def replay(self, item, weight, contract, recipe, *, warmups, repetitions,
               capture_transitions):
        self.replay_calls += 1
        self.reset(item, weight, contract)
        transitions = []
        terminal = None
        for action in recipe:
            snap = self._snapshot()
            if snap.terminal:
                break
            transitions.append({"observation": snap.observation.tolist(),
                                "legal_action_ids": list(snap.legal_action_ids),
                                "action_id": action})
            terminal = self.step(action)
        if terminal is None or not terminal.terminal:
            terminal = target.Snapshot(True, None, (), self.finish())
        final_recipe = tuple(terminal.response["recipe_action_ids"])
        ordinary = [value for value in final_recipe if value < target.FLOW_STOP_BEGIN]
        divisor = 1 + sum(ordinary)
        frame = max(1, item.original_numeric_bytes // divisor) + 8
        reps = max(1, repetitions)
        return {"terminal": True, "recipe_action_ids": list(final_recipe),
                "transitions": transitions, "frame_bytes": frame,
                "frame_sha256": target.digest(bytes(frame)),
                "original_numeric_bytes": item.original_numeric_bytes,
                "encode_samples_ns": [100 + 25 * len(ordinary)] * reps,
                "decode_samples_ns": [80 + 10 * len(ordinary)] * reps,
                "reward_timing_eligible": True, "model_sha256": self.model_sha256,
                "canonical_recipe_key": target.digest(str(final_recipe).encode()),
                "timing_scope": "matched_policy_forced_action_total"}


def identity(rows):
    return target.policy_identity(rows, target.OnlineContract())


class OnlineMatchedTrainV3Test(unittest.TestCase):
    def test_fit2_stratified_teacher_mass_is_exact(self):
        records = (
            [{"step": 0, "action_id": target.FLOW_STOP_BEGIN}] * 4 +
            [{"step": 0, "action_id": 7}] +
            [{"step": 1, "action_id": target.FLOW_STOP_BEGIN}] * 3
        )
        weights, receipt = target.teacher_stratified_weights(records)
        self.assertAlmostEqual(weights[:4].sum() / weights.sum(), 0.375)
        self.assertAlmostEqual(weights[4:5].sum() / weights.sum(), 0.375)
        self.assertAlmostEqual(weights[5:].sum() / weights.sum(), 0.25)
        self.assertAlmostEqual(receipt["root_total_mass"], 0.75)
        self.assertEqual(receipt["counts"], {
            "root_stop": 4, "root_nonstop": 1, "continuation": 3})

    def test_fit2_root_metrics_include_precision_recall_and_weight_groups(self):
        rows = (block("a"),)
        policy = grpo.TwoHiddenMLP(identity(rows), np.random.default_rng(1))
        policy.params[5][:] = 0.0
        policy.params[5][target.FLOW_STOP_BEGIN] = 100.0
        observation = np.zeros(target.OBSERVATION_DIM, dtype=np.float64)
        legal = (1, target.FLOW_STOP_BEGIN)
        records = (
            {"step": 0, "observation": observation, "legal_action_ids": legal,
             "action_id": target.FLOW_STOP_BEGIN, "weight": [1, 0, 0]},
            {"step": 0, "observation": observation, "legal_action_ids": legal,
             "action_id": 1, "weight": [1, 0, 0]},
            {"step": 0, "observation": observation, "legal_action_ids": legal,
             "action_id": target.FLOW_STOP_BEGIN, "weight": [0, 1, 0]},
        )
        metrics = target.masked_teacher_accuracy(policy, records)
        self.assertEqual(metrics["root_stop_recall"], 1.0)
        self.assertEqual(metrics["root_nonstop_recall"], 0.0)
        self.assertEqual(metrics["root_balanced_accuracy"], 0.5)
        self.assertIsNone(metrics["root_nonstop_precision"])
        self.assertEqual(len(metrics["root_metrics_by_weight"]), 2)

    def test_contract_rejects_legacy_stop_and_uses_6373(self):
        contract = target.OnlineContract()
        self.assertEqual(contract.action_count, 6373)
        self.assertEqual(contract.flow_stop_begin, 6341)
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            backend.replay(block("a"), (1, 0, 0), contract, (0,), warmups=0,
                           repetitions=1, capture_transitions=False)

    def test_pareto_and_total_objective(self):
        item = block("a")
        rows = (
            target.RecipeMeasurement((6341,), 80, 64, 100, 100, "1" * 64,
                                     "2" * 64, "r0", True),
            target.RecipeMeasurement((1, 6341), 40, 64, 200, 120, "1" * 64,
                                     "3" * 64, "r1", True),
            target.RecipeMeasurement((2, 6341), 90, 64, 300, 300, "1" * 64,
                                     "4" * 64, "r2", True),
        )
        frontier = target.pareto_front(rows)
        self.assertEqual({row.recipe for row in frontier}, {(6341,), (1, 6341)})
        self.assertAlmostEqual(target.objective(rows[1], item, (1, 0, 0)), 40 / 64)

    def test_teacher_is_training_only_and_contains_legal_multistep_labels(self):
        backend = FakeBackend()
        contract = target.OnlineContract()
        records = target.build_teacher_records(
            backend, (block("a"),), contract, np.random.default_rng(2),
            candidate_count=4, teacher_weights=((1, 0, 0),), warmups=0,
            repetitions=1)
        self.assertTrue(records)
        self.assertTrue(all(row["split"] == "train" and
                            row["action_id"] in row["legal_action_ids"] for row in records))
        with self.assertRaises(ValueError):
            target.build_teacher_records(
                backend, (block("v", "validation"),), contract,
                np.random.default_rng(2), candidate_count=4)

    def test_masked_ce_is_finite_and_improves(self):
        rows = (block("a"),)
        policy = grpo.TwoHiddenMLP(identity(rows), np.random.default_rng(3))
        optimizer = grpo.Adam(policy.params, 1e-2, device="cpu")
        features = np.zeros((16, target.OBSERVATION_DIM), dtype=np.float64)
        features[:, -3] = 1
        legal = np.zeros((16, target.ACTION_COUNT), dtype=bool)
        legal[:, [1, target.FLOW_STOP_BEGIN]] = True
        actions = np.ones(16, dtype=np.int64)
        before = target._probabilities(policy, features[0], (1, target.FLOW_STOP_BEGIN))[0]
        metrics = target.masked_cross_entropy_update(
            policy, optimizer, features, legal, actions, batch_size=8,
            rng=np.random.default_rng(4), device="cpu")
        after = target._probabilities(policy, features[0], (1, target.FLOW_STOP_BEGIN))[0]
        self.assertTrue(np.isfinite(metrics["mean_masked_cross_entropy"]))
        self.assertGreater(after, before)

    def test_same_recipe_group_measured_once_and_has_zero_advantage(self):
        rows = (block("a"), block("b", dataset="d1", domain="network"))
        policy = grpo.TwoHiddenMLP(identity(rows), np.random.default_rng(5))
        policy.params[5][target.FLOW_STOP_BEGIN] = 100.0
        optimizer = grpo.Adam(policy.params, 3e-4, device="cpu")
        backend = FakeBackend()
        config = grpo.GRPOConfig(groups_per_iteration=2, epochs=1,
                                 entropy_coefficient=0.0)
        result = target.grpo_iteration(
            backend, policy, optimizer, target.StratifiedSampler(rows),
            target.OnlineContract(), np.random.default_rng(6), config,
            warmups=0, repetitions=1, group_size=4)
        self.assertEqual(result["same_recipe_groups"], 2)
        self.assertEqual(result["informative_groups"], 0)
        self.assertEqual(backend.replay_calls, 2)
        self.assertEqual(result["parameter_change_l2"], 0.0)

    def test_resume_roundtrip_and_contract_tamper(self):
        rows = (block("a"),)
        ident = identity(rows)
        policy = grpo.TwoHiddenMLP(ident, np.random.default_rng(7))
        optimizer = grpo.Adam(policy.params, 3e-4, device="cpu")
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.json"
            target.save_resume(path, policy=policy, optimizer=optimizer, rng=rng,
                               contract=target.OnlineContract(), seed=7,
                               pretrain_epochs=2, grpo_iterations=3,
                               cumulative_wall_ns=11, trace=({"ok": True},))
            loaded, loaded_optimizer, _loaded_rng, payload = target.load_resume(
                path, identity=ident, contract=target.OnlineContract(), device="cpu")
            self.assertEqual(payload["grpo_iterations"], 3)
            self.assertEqual(loaded_optimizer.steps, optimizer.steps)
            for left, right in zip(policy.params, loaded.params):
                np.testing.assert_array_equal(left, right)
            document = __import__("json").loads(path.read_text())
            document["payload"]["grpo_iterations"] = 4
            path.write_text(__import__("json").dumps(document))
            with self.assertRaises(ValueError):
                target.load_resume(path, identity=ident,
                                   contract=target.OnlineContract(), device="cpu")

    def test_quick_subset_caps_each_dataset_and_sampler_balances_domains(self):
        rows = tuple(block(f"a{i}", dataset="d0", raw=64 * (i + 1)) for i in range(4)) + tuple(
            block(f"b{i}", dataset="d1", domain="network", raw=128 * (i + 1))
            for i in range(4)) + (block("av", "validation"),
                                  block("bv", "validation", dataset="d1", domain="network"))
        quick = target.quick_subset(rows)
        self.assertEqual(sum(row.split == "train" and row.dataset_id == "d0" for row in quick), 2)
        self.assertEqual(sum(row.split == "validation" for row in quick), 2)
        receipt = target.StratifiedSampler(tuple(row for row in quick if row.split == "train")).receipt()
        self.assertEqual(set(receipt), {"energy", "network"})

    def test_deadline_is_hard_and_finite(self):
        self.assertLessEqual(target.remaining_deadline_ns(2_000_000_000, 1.0), 0)
        with self.assertRaises(ValueError):
            target.remaining_deadline_ns(0, float("nan"))

    def test_masked_ce_cpu_cuda_parity_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch absent")
        if not torch.cuda.is_available():
            self.skipTest("CUDA absent")
        rows = (block("a"),)
        base = grpo.TwoHiddenMLP(identity(rows), np.random.default_rng(9))
        cpu_policy = copy.deepcopy(base)
        gpu_policy = copy.deepcopy(base)
        cpu_optimizer = grpo.Adam(cpu_policy.params, 1e-3, device="cpu")
        gpu_optimizer = grpo.Adam(gpu_policy.params, 1e-3, device="cuda:0")
        features = np.zeros((4, target.OBSERVATION_DIM), dtype=np.float64)
        features[:, -3] = 1
        legal = np.zeros((4, target.ACTION_COUNT), dtype=bool)
        legal[:, [1, 2, target.FLOW_STOP_BEGIN]] = True
        actions = np.asarray([1, 2, 1, 2], dtype=np.int64)
        sample_weights = np.asarray([0.5, 1.5, 0.5, 1.5], dtype=np.float64)
        for policy, optimizer, device in ((cpu_policy, cpu_optimizer, "cpu"),
                                          (gpu_policy, gpu_optimizer, "cuda:0")):
            target.masked_cross_entropy_update(
                policy, optimizer, features, legal, actions, batch_size=4,
                rng=np.random.default_rng(10), device=device,
                sample_weights=sample_weights)
        for left, right in zip(cpu_policy.params, gpu_policy.params):
            np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-9)

    def test_real_cpp_session_and_reward_timing_when_built(self):
        root = Path(__file__).resolve().parents[1]
        server = root / "cpp/native_pipeline_v1/build/native_online_training_server"
        catalog = root / "cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json"
        profile = root / "cpp/native_pipeline_v1/experiment_no_cmix_nncp_v1.json"
        if not all(path.is_file() for path in (server, catalog, profile)):
            self.skipTest("native online training server is not built")
        item = block("native", raw=256)
        contract = target.OnlineContract()
        policy = grpo.TwoHiddenMLP(identity((item,)), np.random.default_rng(11))
        with tempfile.TemporaryDirectory() as directory:
            model, model_sha = target.export_current_model(
                policy, grpo.GRPOConfig(), 11, 0, Path(directory), contract,
                phase="native_probe")
            with target.OnlineTrainingServer(server, catalog, profile,
                                             timeout_seconds=15) as backend:
                self.assertEqual(backend.load_model(model), model_sha)
                recipe = target.stop_recipe(backend, item, contract)
                response = backend.replay(
                    item, (1, 0, 0), contract, recipe, warmups=0,
                    repetitions=1, capture_transitions=True)
                measured = target.parse_measurement(response, item, model_sha)
                self.assertTrue(measured.reward_timing_eligible)
                self.assertEqual(response["timing_scope"],
                                 "matched_policy_forced_action_total")
                self.assertEqual(len(response["transitions"]), len(recipe))

    def test_real_cpp_horizon_canonical_recipe_and_bad_suffix_when_built(self):
        root = Path(__file__).resolve().parents[1]
        server = root / "cpp/native_pipeline_v1/build/native_online_training_server"
        catalog = root / "cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json"
        profile = root / "cpp/native_pipeline_v1/experiment_no_cmix_nncp_v1.json"
        if not all(path.is_file() for path in (server, catalog, profile)):
            self.skipTest("native online training server is not built")
        item = block("native-horizon", raw=256)
        contract = target.OnlineContract()
        policy = grpo.TwoHiddenMLP(identity((item,)), np.random.default_rng(12))
        with tempfile.TemporaryDirectory() as directory:
            model, model_sha = target.export_current_model(
                policy, grpo.GRPOConfig(), 12, 0, Path(directory), contract,
                phase="native_horizon_probe")
            with target.OnlineTrainingServer(server, catalog, profile,
                                             timeout_seconds=15) as backend:
                self.assertEqual(backend.load_model(model), model_sha)
                canonical = (170,) * contract.max_policy_steps + (6342, 6341)
                response = backend.replay(
                    item, (1, 0, 0), contract, canonical, warmups=0,
                    repetitions=1, capture_transitions=True)
                self.assertEqual(tuple(response["recipe_action_ids"]), canonical)
                self.assertEqual(response["policy_calls"], contract.max_policy_steps)
                self.assertEqual(response["forced_flow_stops"], 2)
                self.assertEqual(len(response["transitions"]),
                                 contract.max_policy_steps)
                with self.assertRaisesRegex(ValueError, "suffix.*per-flow STOP"):
                    backend.replay(
                        item, (1, 0, 0), contract,
                        (170,) * contract.max_policy_steps + (170,),
                        warmups=0, repetitions=1, capture_transitions=False)
                with self.assertRaisesRegex(ValueError, "canonical recipe differs"):
                    backend.replay(
                        item, (1, 0, 0), contract,
                        (170,) * contract.max_policy_steps + (6341, 6342),
                        warmups=0, repetitions=1, capture_transitions=False)


if __name__ == "__main__":
    unittest.main()
