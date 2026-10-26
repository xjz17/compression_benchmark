#!/usr/bin/env python3
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np

import online_matched_train_v6 as core
import train_combo_policy_sixweight_v7_soft_targets as target
import weight_conditioned_grpo_v1 as grpo


WEIGHTS = tuple(tuple(float(value) for value in weight)
                for weight in core.FIXED_WEIGHTS)


def teacher_records():
    rows = []
    winner = (core.FLOW_STOP_BEGIN + 1, 10, core.FLOW_STOP_BEGIN)
    legal = (
        (10, 20, 30, 40, core.FLOW_STOP_BEGIN, core.FLOW_STOP_BEGIN + 1),
        (10, 20, 30, 40, core.FLOW_STOP_BEGIN),
        (core.FLOW_STOP_BEGIN,),
    )
    for weight in WEIGHTS:
        for step, action in enumerate(winner):
            rows.append({
                "sample_id": "sample", "weight": list(weight), "step": step,
                "action_id": action, "recipe_action_ids": list(winner),
                "teacher_objective": 1.0,
                "observation": np.zeros(core.OBSERVATION_DIM, dtype=np.float64),
                "legal_action_ids": legal[step],
            })
    return tuple(rows)


def candidate_rows(*, reused=False, illegal=False, omit_weight=None):
    recipes = (
        ((core.FLOW_STOP_BEGIN + 1, 10, core.FLOW_STOP_BEGIN), 1.00),
        ((core.FLOW_STOP_BEGIN + 1, 20, core.FLOW_STOP_BEGIN), 1.02),
        ((core.FLOW_STOP_BEGIN + 1, 40, core.FLOW_STOP_BEGIN), 1.10),
        ((core.FLOW_STOP_BEGIN + 1, 99 if illegal else 30,
          core.FLOW_STOP_BEGIN), 1.20),
    )
    rows = []
    for weight in WEIGHTS:
        if omit_weight == weight:
            continue
        for number, (recipe, objective) in enumerate(recipes):
            measured = WEIGHTS[0] if reused else weight
            rows.append({
                "schema": "native-online-offline-combo-measurement-v6",
                "sample_id": "sample", "split": "train", "status": "ok",
                "candidate": f"candidate-{number}", "weight": list(weight),
                "measured_timing_weight": list(measured),
                "time_reused_across_weights": reused,
                "approximate_teacher_timing": reused,
                "objective": objective, "recipe_action_ids": list(recipe),
            })
    return tuple(rows)


def identity():
    data = (("sample", "a" * 64),
            ("semantic_mask_contract",
             core.digest(core.SEMANTIC_MASK_CONTRACT.encode())))
    roots = (("sample", "b" * 64),)
    return grpo.PolicyIdentity(
        core.REGISTRY_FINGERPRINT, data, roots, core.ACTION_COUNT,
        core.FLOW_STOP_BEGIN, core.TASK_FEATURE_DIM, core.STATE_FEATURE_DIM,
        (128, 128), core.FEATURE_SCHEMA)


class TopKRegretSoftTargetsTest(unittest.TestCase):
    def test_topk_targets_follow_prefix_regret_and_keep_contract(self):
        data = target.build_topk_regret_targets(
            teacher_records(), candidate_rows(), top_k=2,
            relative_regret_temperature=0.05)
        self.assertEqual(data.targets.shape, (18, core.ACTION_COUNT))
        np.testing.assert_allclose(data.targets.sum(axis=1), 1.0)
        self.assertFalse(np.any(data.targets[~data.legal]))
        # Every weight has a mandatory bitmap STOP, then a two-way numeric target,
        # then a deterministic numeric-flow STOP.
        for offset in range(0, 18, 3):
            self.assertEqual(np.flatnonzero(data.targets[offset]).tolist(),
                             [core.FLOW_STOP_BEGIN + 1])
            self.assertEqual(np.flatnonzero(data.targets[offset + 1]).tolist(), [10, 20])
            self.assertGreater(data.targets[offset + 1, 10],
                               data.targets[offset + 1, 20])
            self.assertEqual(np.flatnonzero(data.targets[offset + 2]).tolist(),
                             [core.FLOW_STOP_BEGIN])
        self.assertEqual(data.receipt["action_count"], 6373)
        self.assertFalse(data.receipt["architecture_changed"])
        self.assertFalse(data.receipt["online_contract_changed"])
        self.assertEqual(data.receipt["multi_action_records"], 6)

    def test_exact_sixweight_gate_rejects_reuse_missing_and_illegal_branch(self):
        with self.assertRaisesRegex(ValueError, "exact finite"):
            target.build_topk_regret_targets(
                teacher_records(), candidate_rows(reused=True))
        with self.assertRaisesRegex(ValueError, "lacks successful"):
            target.build_topk_regret_targets(
                teacher_records(), candidate_rows(omit_weight=WEIGHTS[-1]))
        with self.assertRaisesRegex(ValueError, "illegal"):
            target.build_topk_regret_targets(
                teacher_records(), candidate_rows(illegal=True))
        good = {
            "schema": "native-online-offline-combo-space-v6",
            "train_only_teacher": True,
            "validation_evaluation_labels_written": False,
            "online_candidate_search": False,
            "reuse_candidate_time_weights": False,
            "approximate_teacher_timing": False,
            "measured_candidate_timing_weights": [list(weight) for weight in WEIGHTS],
        }
        target.validate_exact_search_metadata(good)
        bad = dict(good, approximate_teacher_timing=True)
        with self.assertRaisesRegex(ValueError, "exact timing"):
            target.validate_exact_search_metadata(bad)

    def test_soft_update_improves_loss_and_never_updates_illegal_output_columns(self):
        policy = grpo.TwoHiddenMLP(identity(), np.random.default_rng(7))
        optimizer = grpo.Adam(policy.params, 1e-2, device="cpu")
        features = np.zeros((32, core.OBSERVATION_DIM), dtype=np.float64)
        features[:, -3] = 1.0
        legal = np.zeros((32, core.ACTION_COUNT), dtype=bool)
        legal[:, [1, 2, core.FLOW_STOP_BEGIN]] = True
        soft = np.zeros((32, core.ACTION_COUNT), dtype=np.float64)
        soft[:, 1] = 0.70
        soft[:, 2] = 0.20
        soft[:, core.FLOW_STOP_BEGIN] = 0.10
        before = target.masked_soft_target_metrics(policy, features, legal, soft)
        illegal_w3 = policy.params[4][:, 3].copy()
        illegal_b3 = float(policy.params[5][3])
        update = target.masked_soft_cross_entropy_update(
            policy, optimizer, features, legal, soft, batch_size=8,
            rng=np.random.default_rng(8), device="cpu", stop_bce_weight=0.2)
        after = target.masked_soft_target_metrics(policy, features, legal, soft)
        self.assertLess(after["mean_masked_soft_cross_entropy"],
                        before["mean_masked_soft_cross_entropy"])
        np.testing.assert_array_equal(policy.params[4][:, 3], illegal_w3)
        self.assertEqual(float(policy.params[5][3]), illegal_b3)
        self.assertEqual(update["stop_auxiliary_eligible_records"], 32)
        self.assertTrue(np.isfinite(update["mean_total_distillation_loss"]))
        self.assertFalse(update["architecture_changed"])

    def test_cpu_cuda_update_parity_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch absent")
        if not torch.cuda.is_available():
            self.skipTest("CUDA absent")
        base = grpo.TwoHiddenMLP(identity(), np.random.default_rng(11))
        cpu_policy, gpu_policy = copy.deepcopy(base), copy.deepcopy(base)
        cpu_optimizer = grpo.Adam(cpu_policy.params, 1e-3, device="cpu")
        gpu_optimizer = grpo.Adam(gpu_policy.params, 1e-3, device="cuda:0")
        features = np.zeros((4, core.OBSERVATION_DIM), dtype=np.float64)
        legal = np.zeros((4, core.ACTION_COUNT), dtype=bool)
        legal[:, [1, 2, core.FLOW_STOP_BEGIN]] = True
        soft = np.zeros((4, core.ACTION_COUNT), dtype=np.float64)
        soft[:, 1], soft[:, 2], soft[:, core.FLOW_STOP_BEGIN] = 0.6, 0.3, 0.1
        for policy, optimizer, device in (
                (cpu_policy, cpu_optimizer, "cpu"),
                (gpu_policy, gpu_optimizer, "cuda:0")):
            target.masked_soft_cross_entropy_update(
                policy, optimizer, features, legal, soft, batch_size=4,
                rng=np.random.default_rng(12), device=device)
        for left, right in zip(cpu_policy.params, gpu_policy.params):
            np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
