#!/usr/bin/env python3
"""Focused tests for budgeted policy/Beam30 graph search."""
from __future__ import annotations

import unittest

import numpy as np

from weight_conditioned_grpo_v1 import (
    CompressionMetrics,
    FrozenPolicy,
    TerminalOutcome,
    initialize_policy,
)
from weight_conditioned_search_v1 import (
    CONTINUATION_EVALUATIONS,
    FIRST_BLOCK_EVALUATIONS,
    SearchConfig,
    accept_continuation_only_if_better,
    beam30_search,
    policy_candidate_search,
)
from weighted_objective_pruning_v1 import BoundState


class ToyTask:
    registry_fingerprint = "search-toy-registry"
    action_count = 3
    stop_action = 2
    max_steps = 3

    def __init__(self):
        self.task_id = "search-toy"
        self.data_fingerprint = "data"
        self.root_fingerprint = "root"
        self.evaluations = 0

    def task_features(self):
        return np.asarray([1.0, 0.0])

    def initial_state(self):
        return ()

    def state_features(self, state):
        return np.asarray([len(state), sum(state) if state else 0.0], dtype=np.float64)

    def legal_actions(self, state):
        return (self.stop_action,) if state else (0, 1, self.stop_action)

    def transition(self, state, action):
        return state + (action,) if not state and action in (0, 1) else None

    def evaluate_terminal(self, state):
        self.evaluations += 1
        size = {(): 8, (0,): 6, (1,): 2}[state]
        encode = {(): 80, (0,): 8, (1,): 40}[state]
        decode = {(): 80, (0,): 16, (1,): 32}[state]
        return TerminalOutcome(
            operator_graph={"recipe_action_ids": list(state)},
            compressed_payload=b"x" * size,
            metrics=CompressionMetrics(8, encode, decode, 2.0, 2.0),
        )


class MeasuredToyTask(ToyTask):
    compression_reference_ns_per_byte = 2.0
    decompression_reference_ns_per_byte = 2.0

    def measured_action_cost(self, action):
        return {
            0: (0.75, 100.0, 2.0),
            1: (0.25, 1.0, 2.0),
        }.get(action)

    def action_measurement_count(self, action):
        return 1 if action in (0, 1) else 0

    def ordering_features(self):
        return {
            "smoothness": 0.0,
            "repeat_fraction": 0.0,
            "floating_fraction": 0.0,
        }


class PendingEditState:
    def __init__(self, recipe=(7,), pending_edit_kind=None, stage=0, edit_count=0):
        self.recipe = tuple(recipe)
        self.pending_edit_kind = pending_edit_kind
        self.stage = stage
        self.edit_count = edit_count
        self.continuation_edit_mode = True


class MultiTokenContinuationTask:
    """A two-token edit whose prefix cannot be evaluated or stopped."""

    registry_fingerprint = "search-continuation-registry"
    action_count = 3
    stop_action = 2
    max_steps = 3
    _max_recipe_edits = 1

    def __init__(self, *, hard_budget=2):
        self.task_id = "search-continuation"
        self.data_fingerprint = "data"
        self.root_fingerprint = "root"
        self.evaluations = 0
        self.hard_budget = hard_budget
        self.bound_calls = 0

    def task_features(self):
        return np.asarray([1.0, 0.0])

    def initial_state(self):
        return PendingEditState()

    def state_features(self, state):
        return np.asarray([state.stage, state.edit_count], dtype=np.float64)

    def legal_actions(self, state):
        if state.pending_edit_kind is not None:
            return (1,)
        if state.edit_count:
            return (self.stop_action,)
        return (0, self.stop_action)

    def transition(self, state, action):
        if state.stage == 0 and action == 0:
            return PendingEditState(state.recipe, "replace", 1, 0)
        if state.stage == 1 and action == 1:
            return PendingEditState((9,), None, 2, 1)
        return None

    def evaluate_terminal(self, state):
        if state.pending_edit_kind is not None:
            raise AssertionError("incomplete edit consumed a candidate evaluation")
        self.evaluations += 1
        if self.evaluations > self.hard_budget:
            raise AssertionError("complete-candidate budget exceeded")
        size = 8 if state.recipe == (7,) else 4
        return TerminalOutcome(
            operator_graph={"recipe_action_ids": list(state.recipe)},
            compressed_payload=b"x" * size,
            metrics=CompressionMetrics(8, 8, 8, 2.0, 2.0),
        )


class WeightConditionedSearchTests(unittest.TestCase):
    def test_block_budget_defaults_are_frozen(self):
        self.assertEqual(SearchConfig.for_block(0).max_evaluations,
                         FIRST_BLOCK_EVALUATIONS)
        self.assertEqual(SearchConfig.for_block(1).max_evaluations,
                         CONTINUATION_EVALUATIONS)

    def test_beam30_uses_exact_terminal_objective_with_shared_budget(self):
        task = ToyTask()
        result = beam30_search(
            task, (1.0, 0.0, 0.0),
            config=SearchConfig(max_evaluations=3, beam_width=30),
        )
        self.assertEqual(task.evaluations, 3)
        self.assertEqual(result.evaluations, 3)
        self.assertEqual(result.best_actions, (1,))
        self.assertAlmostEqual(result.best_objective, 0.25)
        self.assertGreaterEqual(result.timing.candidate_evaluation_time_ns, 0)
        self.assertEqual(result.method, "beam30")

    def test_frozen_policy_orders_but_exact_measurement_selects(self):
        task = ToyTask()
        policy = initialize_policy([task], seed=4, hidden_sizes=(4, 4))
        # Make both non-STOP actions outrank STOP; action 0 is deliberately first.
        policy.params[-1][:] = np.asarray([2.0, 1.0, -2.0])
        result = policy_candidate_search(
            FrozenPolicy(policy), task, (1.0, 0.0, 0.0),
            config=SearchConfig(max_evaluations=3, beam_width=2),
        )
        self.assertEqual(result.evaluations, 3)
        self.assertEqual(result.best_actions, (1,))
        self.assertAlmostEqual(result.best_objective, 0.25)
        self.assertGreater(result.timing.model_inference_time_ns, 0)

    def test_default_beam_order_uses_measured_weighted_cost(self):
        task = MeasuredToyTask()
        result = beam30_search(
            task, (0.0, 1.0, 0.0),
            config=SearchConfig(max_evaluations=2, beam_width=1),
        )
        self.assertEqual(result.best_actions, (1,))
        self.assertEqual(result.evaluations, 2)
        self.assertGreaterEqual(result.timing.action_ordering_time_ns, 0)
        self.assertTrue(result.objective_guided_ordering)

        ablated = beam30_search(
            MeasuredToyTask(), (0.0, 1.0, 0.0),
            config=SearchConfig(
                max_evaluations=2, beam_width=1,
                use_objective_guided_ordering=False,
            ),
        )
        self.assertEqual(ablated.best_actions, (0,))
        self.assertFalse(ablated.objective_guided_ordering)

    def test_certified_bound_prunes_only_after_an_incumbent_exists(self):
        task = ToyTask()

        def action_one_first(_task, _state, action, _weight):
            return 0.0 if action == 1 else 1.0

        def bound(_task, state):
            unavoidable = {(): 0, (0,): 6, (1,): 2}[state]
            return BoundState(8, unavoidable, 0, ())

        result = beam30_search(
            task, (1.0, 0.0, 0.0),
            config=SearchConfig(max_evaluations=3, beam_width=2),
            bound_provider=bound,
            action_ordering=action_one_first,
        )
        self.assertEqual(result.best_actions, (1,))
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(result.certified_pruned_states, 1)

    def test_continuation_rejects_non_improving_local_search(self):
        task_a = ToyTask()
        baseline = beam30_search(
            task_a, (1.0, 0.0, 0.0),
            config=SearchConfig(max_evaluations=3, beam_width=30),
        )
        task_b = ToyTask()
        worse = beam30_search(
            task_b, (0.0, 1.0, 0.0),
            config=SearchConfig(max_evaluations=1, beam_width=30),
        )
        # The helper refuses cross-objective comparisons.
        with self.assertRaises(ValueError):
            accept_continuation_only_if_better(baseline, worse)
        kept = accept_continuation_only_if_better(baseline, baseline)
        self.assertIs(kept, baseline)

    def test_multitoken_continuation_evaluates_only_complete_recipes(self):
        for method in ("beam", "policy"):
            task = MultiTokenContinuationTask()

            def unsafe_replayed_prefix_bound(_task, _state):
                task.bound_calls += 1
                return BoundState(8, 0, 0, ())

            if method == "beam":
                result = beam30_search(
                    task, (1.0, 0.0, 0.0),
                    config=SearchConfig(max_evaluations=2, beam_width=30),
                    bound_provider=unsafe_replayed_prefix_bound,
                )
            else:
                policy = initialize_policy([task], seed=11, hidden_sizes=(4, 4))
                result = policy_candidate_search(
                    FrozenPolicy(policy), task, (1.0, 0.0, 0.0),
                    config=SearchConfig(max_evaluations=2, beam_width=30),
                    bound_provider=unsafe_replayed_prefix_bound,
                )
            self.assertEqual(result.evaluations, 2)
            self.assertEqual(task.evaluations, 2)
            self.assertEqual(result.best_outcome.operator_graph["recipe_action_ids"], [9])
            self.assertEqual(result.evaluated_candidates[-1].actions, (9,))
            self.assertEqual(result.evaluated_candidates[-1].search_actions, (0, 1))
            # The current graph-step bound is unsafe while delete/replace edits
            # remain, so search must keep it out of certified pruning.
            self.assertEqual(task.bound_calls, 1)

    def test_policy_search_never_runs_past_the_complete_candidate_budget(self):
        task = ToyTask()
        policy = initialize_policy([task], seed=9, hidden_sizes=(4, 4))
        policy.params[-1][:] = np.asarray([2.0, 1.0, -2.0])
        result = policy_candidate_search(
            FrozenPolicy(policy), task, (1.0, 0.0, 0.0),
            config=SearchConfig(max_evaluations=2, beam_width=30),
        )
        self.assertEqual(task.evaluations, 2)
        self.assertEqual(result.evaluations, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
