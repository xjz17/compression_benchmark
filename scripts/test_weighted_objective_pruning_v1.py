import unittest

from weighted_objective_pruning_v1 import (
    BoundState,
    certified_prune,
    feature_priority,
    objective_lower_bound,
    objective_guided_ordering_score,
    ordering_score,
)


class WeightedObjectivePruningTests(unittest.TestCase):
    def test_bound_counts_bits_before_shared_padding(self):
        state = BoundState(100, 44, 5, (1, 1))
        self.assertEqual(state.unavoidable_bytes, 50)
        self.assertEqual(objective_lower_bound(state, (1, 0, 0)), 0.5)

    def test_a_zero_never_gets_a_positive_byte_bound(self):
        state = BoundState(100, 44, 500, (8000,))
        self.assertEqual(objective_lower_bound(state, (0, 1, 0)), 0.0)
        self.assertFalse(certified_prune(state, (0, 1, 0), 0.01).prune)

    def test_strict_rule_keeps_ties(self):
        state = BoundState(100, 40, 10, ())
        self.assertFalse(certified_prune(state, (1, 0, 0), 0.5).prune)
        self.assertTrue(certified_prune(state, (1, 0, 0), 0.5, keep_ties=False).prune)

    def test_exhaustive_toy_optimum_is_preserved(self):
        # Every completion is >= the certified bound.  Compare exhaustive and
        # pruned minima to ensure the branch removal cannot change the winner.
        branches = [
            (BoundState(100, 20, 0, ()), [0.40, 0.55]),
            (BoundState(100, 44, 20, (800,)), [1.70, 1.90]),
            (BoundState(100, 30, 0, ()), [0.35, 0.60]),
        ]
        incumbent = 0.40
        kept = []
        for state, completions in branches:
            self.assertLessEqual(objective_lower_bound(state, (1, 0, 0)), min(completions))
            if not certified_prune(state, (1, 0, 0), incumbent).prune:
                kept.extend(completions)
                incumbent = min(incumbent, min(completions))
        self.assertEqual(min(kept), min(value for _state, values in branches for value in values))

    def test_heuristics_are_ordering_only_and_keep_other_mass(self):
        priorities = feature_priority((1, 0, 0), {
            "smoothness": 1, "repeat_fraction": 0, "floating_fraction": 0})
        self.assertGreater(priorities["delta_zigzag_pack"], priorities["other_legal"])
        self.assertGreater(priorities["other_legal"], 0)
        self.assertAlmostEqual(sum(priorities.values()), 1.0)
        score = ordering_score(weight=(0.4, 0.3, 0.3), estimated_ratio=0.5,
            estimated_compression_ns_per_byte=2, estimated_decompression_ns_per_byte=1,
            compression_reference_ns_per_byte=2, decompression_reference_ns_per_byte=1)
        self.assertAlmostEqual(score, 0.5)

    def test_measured_cost_guides_order_but_unseen_action_keeps_exploration_bonus(self):
        fast = objective_guided_ordering_score(
            weight=(0, 1, 0), estimated_cost=(1.0, 1.0, 10.0),
            compression_reference_ns_per_byte=10.0,
            decompression_reference_ns_per_byte=10.0,
            feature_priority_mass=0.1, measurement_count=3,
        )
        slow = objective_guided_ordering_score(
            weight=(0, 1, 0), estimated_cost=(0.1, 100.0, 1.0),
            compression_reference_ns_per_byte=10.0,
            decompression_reference_ns_per_byte=10.0,
            feature_priority_mass=0.1, measurement_count=3,
        )
        unseen = objective_guided_ordering_score(
            weight=(0, 1, 0), estimated_cost=None,
            compression_reference_ns_per_byte=10.0,
            decompression_reference_ns_per_byte=10.0,
            feature_priority_mass=0.1, measurement_count=0,
        )
        self.assertLess(fast, slow)
        self.assertLess(unseen, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
