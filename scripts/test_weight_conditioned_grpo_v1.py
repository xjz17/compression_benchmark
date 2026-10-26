#!/usr/bin/env python3
"""Focused deterministic tests; no real compression method or benchmark."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from weight_conditioned_grpo_v1 import (
    FIXED_WEIGHTS,
    CompressionMetrics,
    FrozenPolicy,
    GRPOConfig,
    GraphContractError,
    PolicyIdentity,
    TerminalOutcome,
    TwoHiddenMLP,
    group_advantages,
    initialize_policy,
    load_checkpoint,
    objective_value,
    propose_frozen,
    sample_training_weights,
    save_checkpoint,
    train_grpo,
    validate_weight,
)


class ToyTask:
    registry_fingerprint = "registry-sha256-v1"
    action_count = 3
    stop_action = 2
    max_steps = 3

    def __init__(self, task_id="toy", data="data-a", root="root-a", bias=0):
        self.task_id = task_id
        self.data_fingerprint = data
        self.root_fingerprint = root
        self.bias = bias
        self.initial_roots = []
        self.terminal_states = []

    def task_features(self):
        return np.asarray([self.bias, 1.0], dtype=np.float64)

    def initial_state(self):
        state = ()
        self.initial_roots.append(state)
        return state

    def state_features(self, state):
        return np.asarray([len(state) / 2.0, float(sum(state) if state else 0.0)])

    def legal_actions(self, state):
        return (0, 1, self.stop_action) if not state else (self.stop_action,)

    def transition(self, state, action):
        if action not in (0, 1) or state:
            return None
        return state + (action,)

    def evaluate_terminal(self, state):
        self.terminal_states.append(state)
        # Action 1 has the smaller payload; action 0 is faster.  Empty STOP is worst.
        size = {(): 8, (0,): 6, (1,): 2}[state]
        compression_ns = {(): 80, (0,): 8, (1,): 40}[state]
        decompression_ns = {(): 80, (0,): 16, (1,): 32}[state]
        return TerminalOutcome(
            operator_graph=state,
            compressed_payload=bytes([len(state)]) * size,
            metrics=CompressionMetrics(
                original_numeric_bytes=8,
                compression_time_ns=compression_ns,
                decompression_time_ns=decompression_ns,
                compression_reference_ns_per_byte=2.0,
                decompression_reference_ns_per_byte=2.0,
            ),
        )


class WeightConditionedGRPOTests(unittest.TestCase):
    def test_two_hidden_backward_matches_finite_differences(self):
        identity = PolicyIdentity(
            "registry", (("task", "data"),), (("task", "root"),),
            action_count=3, stop_action=2, task_feature_dim=2,
            state_feature_dim=2, hidden_sizes=(4, 5),
        )
        rng = np.random.default_rng(4)
        policy = TwoHiddenMLP(identity, rng)
        inputs = rng.normal(size=(3, identity.input_dim))
        output_gradient = rng.normal(size=(3, identity.action_count))
        logits, cache = policy.forward(inputs)
        gradients = policy.backward(cache, output_gradient)
        self.assertEqual(logits.shape, output_gradient.shape)
        epsilon = 1.0e-6
        for parameter, gradient in zip(policy.params, gradients):
            for index in list(np.ndindex(parameter.shape))[:3]:
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = float(np.sum(policy.forward(inputs)[0] * output_gradient))
                parameter[index] = original - epsilon
                minus = float(np.sum(policy.forward(inputs)[0] * output_gradient))
                parameter[index] = original
                self.assertAlmostEqual((plus - minus) / (2 * epsilon), gradient[index], places=5)

    def test_defaults_are_two_by_128_and_required_grpo_hyperparameters(self):
        config = GRPOConfig()
        self.assertEqual(config.hidden_sizes, (128, 128))
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.clip_ratio, 0.2)
        self.assertEqual(config.entropy_coefficient, 0.01)
        self.assertEqual(config.epochs, 4)
        self.assertEqual(config.group_size, 8)
        policy = initialize_policy([ToyTask()], seed=3)
        self.assertEqual([value.shape for value in policy.params], [
            (7, 128), (128,), (128, 128), (128,), (128, 3), (3,),
        ])
        self.assertFalse(hasattr(policy, "critic"))

    def test_weight_sampler_is_half_fixed_and_half_continuous(self):
        first = sample_training_weights(12, np.random.default_rng(7))
        second = sample_training_weights(12, np.random.default_rng(7))
        for (left, left_source), (right, right_source) in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(left_source, right_source)
        self.assertEqual(sum(source == "fixed" for _weight, source in first), 6)
        self.assertEqual(sum(source == "dirichlet" for _weight, source in first), 6)
        fixed = {tuple(row) for row, source in first if source == "fixed"}
        self.assertTrue(fixed.issubset(set(FIXED_WEIGHTS)))
        for row, source in first:
            self.assertAlmostEqual(float(row.sum()), 1.0)
            self.assertTrue(np.all(row >= 0.0))
            if source == "dirichlet":
                self.assertTrue(np.all(row > 0.0))

    def test_weight_validation_uses_absolute_simplex_tolerance(self):
        np.testing.assert_array_equal(
            validate_weight((0.2, 0.3, 0.5)), np.asarray([0.2, 0.3, 0.5])
        )
        with self.assertRaisesRegex(ValueError, "summing to one"):
            validate_weight((0.2, 0.3, 0.500001))

    def test_objective_uses_raw_ratio_and_normalized_times(self):
        outcome = TerminalOutcome(
            operator_graph=(1,),
            compressed_payload=b"ab",
            metrics=CompressionMetrics(8, 16, 8, 2.0, 1.0),
        )
        # rho=.25, Ehat=.5, Dhat=.5
        self.assertAlmostEqual(objective_value(outcome, (0.4, 0.3, 0.3)), 0.4)
        self.assertAlmostEqual(objective_value(outcome, (1.0, 0.0, 0.0)), 0.25)

    def test_groups_share_task_root_weight_and_reward_is_terminal_minus_j(self):
        task = ToyTask()
        config = GRPOConfig(hidden_sizes=(12, 10), groups_per_iteration=2)
        result = train_grpo([task], seed=11, iterations=1, config=config)
        row = result.trace[0]
        self.assertEqual(row["episodes"], 16)
        self.assertEqual(len(task.terminal_states), 16)
        self.assertEqual(row["reward_definition"], "terminal_-J_nonterminal_0")
        self.assertEqual(row["fixed_weight_groups"], 1)
        self.assertEqual(row["dirichlet_weight_groups"], 1)
        self.assertEqual(row["optimizer_steps"], 4)
        for group in row["groups_detail"]:
            self.assertEqual(group["episodes"], 8)
            self.assertEqual(group["task_id"], "toy")
            self.assertEqual(group["root_fingerprint"], "root-a")
            self.assertEqual(len(group["weight"]), 3)
        self.assertLessEqual(row["max_terminal_reward"], 0.0)

    def test_training_is_seed_reproducible_and_updates_without_a_critic(self):
        config = GRPOConfig(hidden_sizes=(16, 14), groups_per_iteration=4)
        initial = initialize_policy([ToyTask()], seed=19, hidden_sizes=(16, 14))
        before = [value.copy() for value in initial.params]
        first = train_grpo([ToyTask()], seed=19, iterations=3, config=config)
        second = train_grpo([ToyTask()], seed=19, iterations=3, config=config)
        self.assertTrue(any(np.any(a != b) for a, b in zip(before, first.policy.params)))
        for left, right in zip(first.policy.params, second.policy.params):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(
            [row["mean_terminal_reward"] for row in first.trace],
            [row["mean_terminal_reward"] for row in second.trace],
        )
        self.assertTrue(all(row["parameter_change_l2"] > 0.0 for row in first.trace))

    def test_checkpoint_is_deterministic_and_bound_to_registry_and_data(self):
        config = GRPOConfig(hidden_sizes=(10, 9), groups_per_iteration=2)
        result = train_grpo([ToyTask()], seed=5, iterations=1, config=config)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            digest1 = save_checkpoint(result, first, extra={"purpose": "unit-test"})
            digest2 = save_checkpoint(result, second, extra={"purpose": "unit-test"})
            self.assertEqual(digest1, digest2)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded = load_checkpoint(
                first,
                expected_registry_fingerprint="registry-sha256-v1",
                expected_data_fingerprints={"toy": "data-a"},
            )
            self.assertEqual(loaded.payload_sha256, digest1)
            self.assertEqual(loaded.extra, {"purpose": "unit-test"})
            for left, right in zip(result.policy.params, loaded.policy.params):
                np.testing.assert_array_equal(left, right)
            with self.assertRaisesRegex(GraphContractError, "registry fingerprint"):
                load_checkpoint(first, expected_registry_fingerprint="changed")
            with self.assertRaisesRegex(GraphContractError, "training-data fingerprints"):
                load_checkpoint(first, expected_data_fingerprints={"toy": "changed"})
            document = first.read_text()
            first.write_text(document.replace('"seed":5', '"seed":6'))
            with self.assertRaisesRegex(GraphContractError, "digest mismatch"):
                load_checkpoint(first)

    def test_frozen_proposal_returns_graph_payload_and_timing_without_mutation(self):
        task = ToyTask()
        policy = initialize_policy([task], seed=23, hidden_sizes=(8, 8))
        frozen = FrozenPolicy(policy)
        before = [value.copy() for value in frozen.params]
        proposal = propose_frozen(frozen, ToyTask(data="new-test-data"), (1.0, 0.0, 0.0))
        self.assertIsInstance(proposal.compressed_payload, bytes)
        self.assertEqual(proposal.operator_graph, proposal.outcome.operator_graph)
        self.assertGreaterEqual(proposal.prediction_time_ns, 0)
        self.assertGreaterEqual(proposal.feature_time_ns, 0)
        self.assertGreaterEqual(proposal.model_inference_time_ns, 0)
        self.assertGreaterEqual(proposal.terminal_evaluation_time_ns, 0)
        for old, new in zip(before, frozen.params):
            np.testing.assert_array_equal(old, new)
            self.assertFalse(new.flags.writeable)
        with self.assertRaises(TypeError):
            propose_frozen(policy, ToyTask(), (1.0, 0.0, 0.0))

    def test_advantages_are_group_local_and_constant_groups_are_zero(self):
        values = np.asarray([-2., -1., 0., 1., 2., 3., 4., 5.] + [7.] * 8)
        advantages = group_advantages(values)
        np.testing.assert_allclose(advantages[:8], group_advantages(values[:8]))
        np.testing.assert_array_equal(advantages[8:], np.zeros(8))
        np.testing.assert_allclose(group_advantages(values + 100.0), advantages, atol=1e-13)

    def test_final_step_requires_stop_and_registry_mismatch_is_rejected(self):
        class BrokenTask(ToyTask):
            max_steps = 1

            def legal_actions(self, state):
                return (0, 1)

        policy = initialize_policy([ToyTask()], seed=1, hidden_sizes=(4, 4))
        with self.assertRaisesRegex(GraphContractError, "STOP must be legal"):
            propose_frozen(FrozenPolicy(policy), BrokenTask(), (1.0, 0.0, 0.0))
        changed = ToyTask()
        changed.registry_fingerprint = "different-registry"
        with self.assertRaisesRegex(GraphContractError, "action/registry"):
            propose_frozen(FrozenPolicy(policy), changed, (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
