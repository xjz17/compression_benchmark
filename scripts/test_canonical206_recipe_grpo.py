#!/usr/bin/env python3
"""Fast deterministic tests; no real codec, data file, or measurement benchmark."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from canonical206_recipe_grpo import (
    _actor_update, _collect, group_advantages, initialize_policy, load_policy,
    propose, sample_uniform, save_policy, train,
)
from search_physical_operator_variants_v2 import Adam


class ToyEnv:
    action_count = 5
    stop_id = 4
    max_steps = 3

    def __init__(self, shift=0.0):
        self.shift = shift
        self.executed = []
        self.reward_calls = 0

    def features(self, prefix):
        return np.asarray([1.0, len(prefix) / 3.0,
                           float(0 in prefix), float(1 in prefix)])

    def legal(self, prefix):
        if len(prefix) == 0:
            return (0, 1, self.stop_id)
        if len(prefix) == 1:
            return (2, 3, self.stop_id)
        return (self.stop_id,)

    def transition(self, prefix, action):
        assert action in self.legal(prefix) and action != self.stop_id
        self.executed.append((prefix, action))
        return prefix + (action,)

    def reward(self, prefix):
        self.reward_calls += 1
        return self.shift + {(): -1.0, (0,): 0.1, (1,): 0.2,
                             (0, 2): 2.0, (0, 3): 0.3,
                             (1, 2): 0.4, (1, 3): 3.0}[prefix]


class BudgetExceeded(RuntimeError):
    pass


class GRPOTests(unittest.TestCase):
    def test_group_advantage_translation_and_constant(self):
        returns = np.asarray([-1., .1, .2, .7, 2., 2., 2., 2.])
        first = group_advantages(returns, 4)
        np.testing.assert_allclose(first, group_advantages(returns + 37.0, 4), atol=1e-13)
        np.testing.assert_array_equal(first[4:], np.zeros(4))
        self.assertTrue(np.all(first[1:4] > 0))

    def test_no_token_recentering_and_episode_equal_weights(self):
        env = ToyEnv()
        actor = initialize_policy(env, 4, 8)
        rollout = _collect(env, actor, 4, np.random.default_rng(6))
        lengths = np.asarray([1, 1, 1, 7])
        episode = np.repeat(np.arange(4), lengths)
        values = group_advantages([-1.0, .1, .2, .7], 4)
        self.assertLess((values[1] - values[episode].mean()), 0.0)
        # Construct legal repeated observations with deliberately unequal lengths.
        for key in ("features", "legal", "actions", "old_logp"):
            rollout[key] = np.repeat(rollout[key][:1], len(episode), axis=0)
        rollout["episode"] = episode
        rollout["lengths"] = lengths
        rollout["weights"] = 1.0 / lengths[episode]
        rollout["advantages"] = values[episode].copy()
        before = rollout["advantages"].copy()
        _actor_update(actor, Adam(actor.params, .001), rollout, 2, .2, .03)
        np.testing.assert_array_equal(before, rollout["advantages"])
        self.assertTrue(np.all(rollout["advantages"][episode == 1] > 0))
        np.testing.assert_allclose(np.bincount(episode, weights=rollout["weights"]), np.ones(4))

    def test_training_updates_legal_and_seed_reproducible(self):
        env = ToyEnv()
        initial = initialize_policy(env, 9, 16)
        before = [p.copy() for p in initial.params]
        actor, trace = train(env, 9, iterations=4, episodes=32, hidden=16, actor=initial)
        self.assertEqual(env.reward_calls, 128)
        self.assertEqual(len(trace), 4)
        self.assertTrue(any(np.any(old != new) for old, new in zip(before, actor.params)))
        self.assertTrue(all(row["invalid_fraction"] == 0 for row in trace))
        self.assertTrue(all(row["optimizer_steps"] == 4 for row in trace))
        self.assertTrue(all(row["informative_groups"] > 0 for row in trace))
        self.assertTrue(all(np.isfinite(row["loss_after"]) for row in trace))
        same, same_trace = train(ToyEnv(), 9, iterations=4, episodes=32, hidden=16)
        for left, right in zip(actor.params, same.params):
            np.testing.assert_array_equal(left, right)
        self.assertEqual([x["mean_return"] for x in trace],
                         [x["mean_return"] for x in same_trace])
        shifted, _ = train(ToyEnv(17.), 9, iterations=4, episodes=32, hidden=16)
        for left, right in zip(actor.params, shifted.params):
            np.testing.assert_allclose(left, right, atol=1e-13)

    def test_uniform_budget_callback_and_serialization(self):
        rows = sample_uniform(ToyEnv(), 12, 17)
        self.assertEqual(len(rows), 17)
        self.assertTrue(all(row["optimizer_steps"] == 0 and not row["invalid"] for row in rows))
        clean = lambda rs: [{k: v for k, v in r.items() if k != "seconds"} for r in rs]
        self.assertEqual(clean(rows), clean(sample_uniform(ToyEnv(), 12, 17)))
        env = ToyEnv()
        actor, trace = train(env, 11, episodes=32, max_rollouts=42)
        self.assertEqual([r["episodes"] for r in trace], [32, 8])
        self.assertEqual(env.reward_calls, 40)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actor.policy"
            save_policy(actor, path)
            restored = load_policy(path)
            for left, right in zip(actor.params, restored.params):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(propose(actor, ToyEnv(), np.random.default_rng(1)),
                             propose(restored, ToyEnv(), np.random.default_rng(1)))
        seen = []
        _, trace = train(ToyEnv(), 4, on_iteration=lambda actor, row: seen.append(row) or False)
        self.assertEqual(len(trace), 1)
        self.assertEqual(len(seen), 1)

    def test_budget_exception_not_a_completed_iteration(self):
        env = ToyEnv()
        original = env.reward

        def limited(prefix):
            if env.reward_calls >= 5:
                raise BudgetExceeded("actual measurement budget exhausted")
            return original(prefix)

        env.reward = limited
        seen = []
        actor = initialize_policy(env, 2)
        before = [p.copy() for p in actor.params]
        with self.assertRaisesRegex(BudgetExceeded, "actual measurement"):
            train(env, 2, actor=actor, on_iteration=lambda actor, row: seen.append(row))
        self.assertEqual(seen, [])
        for left, right in zip(before, actor.params):
            np.testing.assert_array_equal(left, right)

    def test_warm_start_separate_accounting(self):
        _, trace = train(ToyEnv(), 1, iterations=1, episodes=8,
                         warm_start_paths=[(1, 3)], warm_start_epochs=2)
        self.assertEqual(trace[0]["warm_start"]["tokens"], 3)
        self.assertEqual(trace[0]["warm_start"]["optimizer_steps"], 2)
        self.assertEqual(trace[0]["optimizer_steps"], 4)
        self.assertEqual(trace[0]["cumulative_grpo_optimizer_steps"], 4)

    def test_invalid_reward_hook_preserves_prefix_and_default(self):
        class InvalidEnv(ToyEnv):
            def legal(self, prefix):
                return (0,) if not prefix else (2,)

            def transition(self, prefix, action):
                return (0,) if not prefix else None

        env = InvalidEnv()
        default = _collect(env, None, 1, np.random.default_rng(1))
        self.assertTrue(default["invalid"][0])
        self.assertEqual(default["pipelines"], [(0,)])
        self.assertEqual(default["episode_returns"][0], -10.0)
        calls = []

        def hook(prefix):
            calls.append(prefix)
            return 1.3 - 0.02

        env.reward_on_invalid = hook
        hooked = _collect(env, None, 1, np.random.default_rng(1))
        self.assertEqual(calls, [(0,)])
        self.assertAlmostEqual(hooked["episode_returns"][0], 1.28)
        self.assertEqual(env.reward_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
