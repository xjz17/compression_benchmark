"""Toy-executor tests for depth/packet grammar; no compression codec is called.

Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python3.11 \
         -m unittest discover -s scripts -p test_canonical206_depth10_env.py -v

These tests verify search control and bookkeeping only.  Production codec and
wire-byte equivalence still require the runner's real round-trip/cold replay.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

import canonical206_recipe_env as recipe_module
from canonical206_depth10_env import BudgetExceeded, Depth10Env, build_catalog
from canonical206_recipe_env import build_catalog as build_recipe_catalog
from canonical206_search_env import DESC, EXCLUDED, SPECS


from canonical182_policy_paths import PolicyPath, concrete_path, path_from_record, path_record
from canonical206_search_env import VARIANT_PATHS, VARIANT_DESC
import primitive_operator_action_runtime as p


def old(ids):
    return path_from_record({'ids': ids, 'policy_registry_version': 'cpp-canonical-206-v1'})


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'analysis_outputs/canonical206_beam29_ppo_grpo_smallest2_20260906_v1/action_registry.json'
DBP = old((93, 94, 95, 96))
BZIP2 = old((165, 166, 167, 74, 181))
DEPTH9 = DBP + BZIP2
DEPTH10 = DEPTH9 + old((76,))


@dataclass
class ToyNode:
    ids: tuple[int, ...]
    frame_bytes: int


class ToyAtomicEnv:
    """Deterministic in-memory stand-in; intentionally does no real encoding."""

    def __init__(self, column, sample_rows=0, max_steps=10):
        self.column = column
        self.max_steps = max_steps
        self.feature_dim = 8
        self.nodes = {(): ToyNode((), 500)}
        self.records = []
        self.invalid = {}
        self.cache_hits = 0

    def transition(self, prefix, action, *, io_variant=None, parameters=None):
        child = PolicyPath(prefix) + concrete_path((action,), (io_variant,), (parameters or {},))
        if child in self.nodes:
            self.cache_hits += 1
            return child
        self.nodes[child] = ToyNode(child, 500 - 10 * len(child))
        self.records.append({'prefix': list(child), 'io_variants': list(child.io_variants), 'status': 'ok', 'error': '',
                             'frame_bytes': self.nodes[child].frame_bytes,
                             'forward_sec': 0.0, 'inverse_sec': 0.0})
        return child

    def features(self, prefix):
        return np.zeros(self.feature_dim)

    def summary(self):
        return {'unique_edges': len(self.records), 'cache_hits': self.cache_hits}


def toy_frame(_column, node):
    payload = b'TOY' + bytes(node.ids)
    return payload.ljust(node.frame_bytes, b'x')


class Depth10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = p.registry()
        cls.original_options = build_recipe_catalog(cls.registry)
        cls.options = build_catalog(cls.registry)
        cls.option_ids = {option.ids: option.option_id for option in cls.options}

    def setUp(self):
        self.column = {'kind': 'float64', 'raw': b'\0' * 512, 'max_point': 0,
                       'dataset': 'toy', 'name': 'control-only'}
        self.patchers = [
            patch.object(recipe_module, 'Env', ToyAtomicEnv),
            patch.object(recipe_module, 'make_frame', toy_frame),
            patch.object(recipe_module, 'decode_frame', lambda _frame: self.column['raw']),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def env(self, **kwargs):
        return Depth10Env(self.column, self.options, 450, **kwargs)

    def step(self, env, prefix, ids, provenance='search'):
        result = env.transition(prefix, self.option_ids[concrete_path(ids)], provenance=provenance)
        self.assertIsNotNone(result)
        return result

    def test_catalog_preserves_full_options_and_explicit_variant_singletons(self):
        self.assertEqual(self.options[:len(self.original_options)], self.original_options)
        self.assertEqual([o.option_id for o in self.options], list(range(len(self.options))))
        for variant, path in VARIANT_PATHS.items():
            if path[0] not in EXCLUDED and VARIANT_DESC[variant].execution_mode in {'standalone', 'stagewise'}:
                self.assertIn(path, self.option_ids)
        for option in self.options:
            self.assertFalse(EXCLUDED.intersection(option.ids))

    def test_unfinished_packet_only_allows_registered_continuation(self):
        env = self.env()
        self.assertNotIn(self.option_ids[old((166,))], env.legal(()))
        prefix = self.step(env, (), old((165,)))
        self.assertFalse(env.is_closed(prefix))
        self.assertIn(self.option_ids[old((166,))], env.legal(prefix))
        self.assertNotIn(self.option_ids[old((75,))], env.legal(prefix))
        self.assertNotIn(self.option_ids[BZIP2], env.legal(prefix))
        self.assertNotIn(self.option_ids[old((169,))], env.legal(prefix))
        prefix = self.step(env, prefix, old((166,)))
        self.assertIn(self.option_ids[old((167,))], env.legal(prefix))
        before = len(env.base.records)
        with self.assertRaises(ValueError):
            env.execute_ids(old((165, 75)))
        self.assertEqual(before, len(env.base.records))

    def test_whole_recipe_and_single_steps_have_identical_expansion(self):
        macro = self.env()
        macro_path = self.step(macro, (), BZIP2)
        atomic = self.env()
        atomic_path = ()
        for index in range(len(BZIP2)):
            atomic_path = self.step(atomic, atomic_path, BZIP2[index:index+1])
        self.assertEqual(macro.expanded(macro_path), atomic.expanded(atomic_path))
        self.assertEqual(macro.frame(macro_path), atomic.frame(atomic_path))
        self.assertTrue(macro.is_closed(macro_path))
        self.assertTrue(atomic.is_closed(atomic_path))

    def test_more_than_two_completed_byte_segments(self):
        env = self.env()
        prefix = ()
        for segment in (old((74,)), old((75,)), old((76,)), old((192, 127, 128))):
            prefix = self.step(env, prefix, segment)
            self.assertTrue(env.is_closed(prefix))
        self.assertEqual(len(prefix), 4)
        self.assertEqual(len(env.expanded(prefix)), 6)

    def test_two_reference_prefixes_without_third_or_repeated_id(self):
        self.column['kind'] = 'int64'
        env = self.env()
        prefix = self.step(env, (), old((7,)))
        prefix = self.step(env, prefix, old((11,)))
        self.assertNotIn(self.option_ids[old((1,))], env.legal(prefix))
        self.assertNotIn(self.option_ids[old((7,))], env.legal(prefix))
        self.assertIn(self.option_ids[BZIP2], env.legal(prefix))

    def test_typed_recipe_cannot_start_from_byte_packet(self):
        env = self.env()
        prefix = self.step(env, (), old((74,)))
        self.assertNotIn(self.option_ids[DBP], env.legal(prefix))

    def test_actual_depth9_and_10_and_direct_depth11_rejection(self):
        env = self.env()
        self.assertEqual(env.max_steps, 10)
        prefix = self.step(env, (), DBP)
        self.assertEqual(env.remaining_depth(prefix), 6)
        prefix = self.step(env, prefix, BZIP2)
        self.assertEqual(env.expanded(prefix), DEPTH9)
        self.assertEqual(env.remaining_depth(prefix), 1)
        self.assertNotIn(self.option_ids[old((192, 127, 128))], env.legal(prefix))
        prefix = self.step(env, prefix, old((76,)))
        self.assertEqual(env.expanded(prefix), DEPTH10)
        self.assertEqual(env.legal(prefix), (env.stop,))
        before = len(env.base.records)
        with self.assertRaises(ValueError):
            env.execute_ids(DEPTH9 + old((75, 76)))
        self.assertEqual(before, len(env.base.records))
        summary = env.summary()
        self.assertEqual(summary['max_atomic_depth_reached'], 10)
        self.assertEqual(summary['max_verified_candidate_depth'], 10)
        self.assertEqual(summary['candidate_depth_histogram_verified_unique']['9'], 1)
        self.assertEqual(summary['candidate_depth_histogram_verified_unique']['10'], 1)
        self.assertEqual(summary['depth_9_10_evidence']['10']['verified_examples'], [list(DEPTH10)])
        self.assertEqual(summary['depth_guard_rejections'], 1)

    def test_ten_single_decisions_reach_real_depth10(self):
        env = self.env()
        prefix = ()
        for index in range(len(DEPTH10)):
            prefix = self.step(env, prefix, DEPTH10[index:index+1])
        self.assertEqual(len(prefix), 10)
        self.assertEqual(env.expanded(prefix), DEPTH10)
        self.assertEqual(env.legal(prefix), (env.stop,))

    def test_seed_uses_original_option_and_starts_budget_after_replay(self):
        env = self.env(max_unique_edges=0)
        seeded = env.seed_fixed_recipes([path_record(BZIP2)])
        self.assertEqual(seeded[0]['compressed_bytes'], 450)
        self.assertIn((self.option_ids[BZIP2],), env.nodes)
        self.assertEqual(env.search_budget_origin, 5)
        self.assertEqual(env.best_bytes, 450)
        self.step(env, (), BZIP2)
        self.assertEqual(env.summary()['search_unique_atomic_edges'], 0)
        with self.assertRaises(BudgetExceeded):
            env.transition((), self.option_ids[DBP])

    def test_budget_interruption_is_not_verified_depth10(self):
        env = self.env(max_unique_edges=9)
        prefix = self.step(env, (), DBP)
        prefix = self.step(env, prefix, BZIP2)
        with self.assertRaises(BudgetExceeded):
            env.transition(prefix, self.option_ids[old((76,))])
        summary = env.summary()
        self.assertEqual(summary['max_candidate_depth_attempted'], 10)
        self.assertEqual(summary['max_atomic_depth_reached'], 9)
        self.assertEqual(summary['max_verified_candidate_depth'], 9)
        self.assertEqual(summary['depth_9_10_evidence']['10']['unique_verified_complete'], 0)
        self.assertFalse(summary['depth_9_10_evidence']['10']['atomic_stage_reached'])

    def test_no_automatic_completion_or_global_best_reward(self):
        env = self.env()
        prefix = self.step(env, (), old((74,)))
        self.assertIsNone(env.auto_complete_option)
        self.assertFalse(env.complete_on_transition)
        before = len(env.base.records)
        score = env.reward(prefix)
        env.best_bytes = 1
        self.assertEqual(score, env.reward(prefix))
        self.assertAlmostEqual(env.reward_on_invalid(prefix), score - 0.02)
        self.assertEqual(before, len(env.base.records))
        self.assertFalse(env._completion_cache)
        with self.assertRaises(ValueError):
            self.env(auto_complete_option=self.option_ids[BZIP2])

    def test_complete_decode_timeout_does_not_promote_atomic_cache(self):
        env = self.env()
        with patch.object(recipe_module, 'decode_frame', side_effect=subprocess.TimeoutExpired('toy', 0.3)):
            self.assertIsNone(env.transition((), self.option_ids[BZIP2]))
        self.assertIn(BZIP2, env.base.nodes)
        self.assertNotIn(BZIP2, env.completed)
        self.assertNotIn((self.option_ids[BZIP2],), env.nodes)
        self.assertEqual(env.best_prefix, ())
        self.assertEqual(env.records[-1]['failed_phase'], 'complete_decode')

    def test_depth_validation_and_feature_shape(self):
        for bad in (0, 11, 3.5, True):
            with self.assertRaises(ValueError):
                self.env(max_atomic_depth=bad)
        env = self.env(max_atomic_depth=9)
        self.assertEqual(env.features(()).shape, (env.feature_dim,))
        prefix = self.step(env, (), DBP)
        prefix = self.step(env, prefix, BZIP2)
        self.assertEqual(env.legal(prefix), (env.stop,))
        self.assertEqual(env.grammar_state(prefix)['registered_id_depth'], 9)

    def archive(self, ids, **extra):
        frame = toy_frame(self.column, ToyNode(tuple(ids), 500 - 10 * len(ids)))
        return {**path_record(ids), 'selected_ids': list(ids), 'compressed_bytes': len(frame),
                'frame_sha256': hashlib.sha256(frame).hexdigest(),
                'source': 'toy_archive.json', 'method': 'old_beam29', 'seed': 0, **extra}

    def test_seed_arbitrary_known_chain_checks_hash_and_only_promotes_endpoint(self):
        env = Depth10Env(self.column, self.options, 410, max_unique_edges=0)
        path = env.option_path_for_ids(DEPTH9)
        self.assertEqual(len(path), 9)
        self.assertEqual(env.expanded(path), DEPTH9)
        rows = env.seed_known_chains([self.archive(BZIP2), self.archive(DEPTH9)])
        self.assertEqual(len(rows), 2)
        self.assertEqual(env.best_prefix, path)
        self.assertEqual(env.best_bytes, 410)
        self.assertEqual(env.best_provenance, 'baseline')
        self.assertIsNone(env.best_new_prefix)
        self.assertIn(path, env.nodes)
        self.assertNotIn(path[:1], env.nodes)
        self.assertEqual(env.reward(path), 0.0)
        self.assertEqual(env.search_budget_origin, len(env.base.records))
        self.assertEqual(env.summary()['search_unique_atomic_edges'], 0)
        self.assertTrue(env.summary()['best_is_seeded_known_chain'])
        self.assertFalse(env.summary()['strict_new_win'])
        self.assertEqual(env.summary()['best_known_archive_source'][0]['method'], 'old_beam29')

    def test_search_replay_of_archive_chain_is_not_new_win(self):
        env = Depth10Env(self.column, self.options, 410)
        env.seed_known_chains([self.archive(DEPTH9)])
        prefix = ()
        for index in range(len(DEPTH9)):
            prefix = self.step(env, prefix, DEPTH9[index:index+1])
        self.assertEqual(env.expanded(prefix), DEPTH9)
        self.assertEqual(env.best_provenance, 'baseline')
        self.assertFalse(env.summary()['strict_new_win'])
        if env.best_new_prefix is not None:
            self.assertNotEqual(env.expanded(env.best_new_prefix), DEPTH9)

    def test_archive_wrong_hash_never_promotes_an_endpoint(self):
        env = Depth10Env(self.column, self.options, 410)
        with self.assertRaises(AssertionError):
            env.seed_known_chains([self.archive(DEPTH9, frame_sha256='0' * 64)])
        self.assertEqual(env.nodes, {(): ()})
        self.assertEqual(env.best_prefix, ())
        self.assertFalse(env.seeded_known_ids)

    def test_archive_requires_original_hash_and_rejects_depth11(self):
        env = self.env()
        with self.assertRaises(ValueError):
            env.seed_known_chains([{'ids': list(BZIP2)}])
        self.assertFalse(env.base.records)
        env = self.env()
        with self.assertRaises(ValueError):
            env.seed_known_chains([self.archive(DEPTH9 + old((75, 76)))])
        self.assertFalse(env.base.records)


if __name__ == '__main__':
    unittest.main()
