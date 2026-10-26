"""LZ parameter migration, true codec replay and conditional-policy gradients."""
import copy
import json
from pathlib import Path
import pickle
import struct
import tempfile
import unittest

import numpy as np
import primitive_operator_action_runtime as p
import unified_action_runtime as u
import native_lz_parameters as lz
import native_window_parameters as window_parameters
import canonical156_parameter_policy as conditional
from canonical182_policy_paths import PolicyPath, concrete_path, path_from_names, path_from_record, path_record
from canonical182_policy_checkpoint import save, load, validate
from canonical206_search_env import Env, encode_cold, decode_frame, SEQUENCES, SPECS, beam29
from canonical206_recipe_env import RecipeEnv, RecipeOption, build_catalog
from canonical206_recipe_grpo import initialize_policy, train as recipe_train
from canonical206_rl_policy import train as atomic_train


def column():
    return {'kind': 'int64', 'raw': struct.pack('<512q', *([1, 2, 3, 4] * 128)),
            'max_point': 0, 'name': 'lz', 'dataset': 'parameter-probe'}


def chain():
    return next(s for s in SEQUENCES if len(s) == 4 and SPECS[s[0]].name == window_parameters.ALGORITHM_NAME and SPECS[s[-1]].name.endswith('bias3'))


def configured(values):
    recipe = chain()
    return concrete_path(recipe, parameters=({}, values, {}, {}))


class ParameterTests(unittest.TestCase):
    def test_legacy_182_paths_migrate_exact_parameters(self):
        manifest = json.loads((Path(__file__).parents[1] / 'benchmark/reference/actions/canonical182.json').read_text())
        entries = [a for a in manifest['actions'] if lz.parameters_from_name(a['name'])]
        self.assertEqual(len(entries), 27)
        paths = []
        for row in entries:
            record = {'ids': [row['policy_action_id']], 'io_variants': [row['name']],
                      'policy_registry_version': 'cpp-canonical-182-v1'}
            path = path_from_record(record)
            self.assertEqual(path.io_variants, (lz.ALGORITHM_NAME,))
            self.assertEqual(path.parameters, (lz.parameters_from_name(row['name']),))
            self.assertEqual(path_from_names([row['name']]), path)
            self.assertEqual(path_from_record(path_record(path)), path)
            paths.append(path)
        self.assertEqual(len({tuple(v) for v in paths}), 1)
        self.assertEqual(len(set(paths)), 27)

    def test_immutable_path_params_roundtrip_and_default(self):
        a = configured({'min_match': 8, 'max_match': 16})
        b = configured({'min_match': 8, 'max_match': 256})
        self.assertNotEqual(a, b)
        self.assertEqual(len({a, b}), 2)
        self.assertEqual(a[:1] + a[1:], a)
        self.assertEqual(copy.deepcopy(a), a)
        self.assertEqual(pickle.loads(pickle.dumps(a)), a)
        self.assertEqual(path_from_record(json.loads(json.dumps(path_record(a)))), a)
        exposed = a.parameters; exposed[1]['max_match'] = 256
        self.assertEqual(a.parameters[1]['max_match'], 16)
        self.assertEqual(path_from_names([lz.ALGORITHM_NAME]).parameters, (lz.default_parameters(),))
        self.assertEqual(len(lz.parameter_candidates()), 147)

    def test_real_codec_distinct_parameter_cache_and_cold_replay(self):
        env = Env(column(), 0, max_steps=4)
        completed = []
        for values in ({'min_match': 8, 'max_match': 16}, {'min_match': 16, 'max_match': 256}):
            path = configured(values); prefix = ()
            for action, variant, params in zip(path, path.io_variants, path.parameters):
                prefix = env.transition(prefix, action, io_variant=variant, parameters=params)
                self.assertIsNotNone(prefix, env.records[-1])
            frame = encode_cold(column(), prefix)
            self.assertEqual(decode_frame(frame), column()['raw'])
            self.assertEqual(prefix.parameters[1], values)
            completed.append(prefix)
        self.assertNotEqual(completed[0], completed[1])
        self.assertNotEqual(env.nodes[completed[0]].payload_bytes, env.nodes[completed[1]].payload_bytes)
        self.assertEqual(len(env.nodes), 8)

    def test_beam_enumerates_parameters_under_one_algorithm_id(self):
        lz_id = p.POLICY_ACTION_BY_NAME[lz.ALGORITHM_NAME].action_id
        class BeamProbe:
            max_steps = 1
            best_prefix = ()
            def __init__(self): self.seen=[]; self.nodes={}
            def legal(self, prefix): return (lz_id,)
            def legal_io_variants(self, prefix, action): return (lz.ALGORITHM_NAME,)
            def parameter_choices(self, prefix, action): return lz.parameter_candidates()
            def transition(self, prefix, action, *, io_variant, parameters):
                self.seen.append((action, parameters)); return None
        probe = BeamProbe(); beam29(probe, width=1)
        self.assertEqual(len(probe.seen),147)
        self.assertEqual({i for i, _ in probe.seen}, {lz_id})

    def test_conditional_joint_gradient_matches_finite_difference(self):
        env = Env(column(),0,1); actor = initialize_policy(env,5,4)
        action = p.POLICY_ACTION_BY_NAME[lz.ALGORITHM_NAME].action_id
        features = np.vstack([env.features(()), env.features(())])
        legal = np.zeros((2, env.action_count), bool); legal[:, [action, env.stop_id]] = True
        selected = np.array([0,len(conditional.CANDIDATES)-1])
        from search_physical_operator_variants_v2 import distribution
        pp = distribution(actor.parameter_head,features,np.ones((2,len(conditional.CANDIDATES)),bool))[1]
        base = distribution(actor, features, legal)[1]
        rollout = {'features':features,'legal':legal,'actions':np.array([action,action]),
            'parameter_actions':selected, 'old_logp':np.log(pp[np.arange(2),selected]) + np.log(base[:, action]),
            'advantages':np.array([-1.,2.]),'weights':np.ones(2)}
        class Capture:
            def step(self,grads): self.grads=grads
        opt, popt = Capture(), Capture()
        conditional.update(actor,(opt,popt),rollout,1,.2,0)
        bias=actor.parameter_head.params[3]; numeric=[]; epsilon=1e-6
        for index in (0,146,73):
            bias[index]+=epsilon; upper=conditional.objective(actor,rollout,.2,0)['loss']
            bias[index]-=2*epsilon; lower=conditional.objective(actor,rollout,.2,0)['loss']
            bias[index]+=epsilon; numeric.append((upper-lower)/(2*epsilon))
        np.testing.assert_allclose(popt.grads[3][[0,146,73]],numeric,atol=1e-8,rtol=1e-5)
        self.assertGreater(np.linalg.norm(popt.grads[3]),0.1)
        for index in (action, env.stop_id):
            bias = actor.params[3]
            bias[index] += epsilon; upper = conditional.objective(actor, rollout, .2, 0)['loss']
            bias[index] -= 2*epsilon; lower = conditional.objective(actor, rollout, .2, 0)['loss']
            bias[index] += epsilon
            self.assertAlmostEqual(opt.grads[3][index], (upper-lower)/(2*epsilon), places=8)
        self.assertGreater(np.linalg.norm(opt.grads[3]), .01)
        # A joint ratio above the upper clip must disable BOTH positive-advantage gradients.
        rollout['advantages'] = np.ones(2)
        rollout['old_logp'] -= np.log(2)
        conditional.update(actor, (opt, popt), rollout, 1, .2, 0)
        for gradients in (opt.grads, popt.grads):
            for gradient in gradients: np.testing.assert_allclose(gradient, 0, atol=1e-15)

    def test_conditional_checkpoint_retains_both_heads_and_fingerprint(self):
        env=Env(column(),0,1);actor=initialize_policy(env,7,4)
        self.assertEqual(actor.params[2].shape[1],128)
        self.assertEqual(actor.parameter_head.params[2].shape[1],len(conditional.CANDIDATES))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.npz';save(actor,path); restored=load(path,env)
            for a,b in zip(actor.parameter_head.params,restored.parameter_head.params): np.testing.assert_array_equal(a,b)
            restored.policy_identity['parameter_fingerprint']='old-default'
            with self.assertRaisesRegex(ValueError,'mismatch'):validate(restored,env)

    def test_recipe_grpo_trains_parameters_with_real_codec_rewards(self):
        recipe=chain(); option=RecipeOption(0,'native-lz',recipe,('bytes',),False)
        env=RecipeEnv(column(),[option],5000,max_steps=1,complete_on_transition=False)
        env.legal=lambda _prefix:(0,)  # isolate parameters; no shortcut via STOP
        actor=initialize_policy(env,19,8)
        before=[v.copy() for v in actor.parameter_head.params]
        actor,trace=recipe_train(env,19,iterations=1,episodes=8,group_size=8,hidden=8,epochs=1,
                                entropy_coefficient=0,actor=actor)
        self.assertEqual(trace[0]['parameter_decisions'],8)
        self.assertGreater(trace[0]['std_return'],0)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(before,actor.parameter_head.params)))
        self.assertGreater(len({json.dumps(k.parameters,sort_keys=True) for k in env.nodes if k}),1)
        for prefix in env.nodes:
            if prefix: self.assertEqual(decode_frame(env.frame(prefix)),column()['raw'])

    def test_atomic_ppo_and_grpo_train_parameter_head_from_real_codec_rewards(self):
        for method in ('ppo', 'grpo'):
            with self.subTest(method=method):
                recipe = chain()
                env = Env(column(), 0, max_steps=4)
                env.legal = lambda prefix: (recipe[len(prefix)],) if len(prefix) < 4 else (env.stop_id,)
                initial = initialize_policy(env, 23, 8)
                actor, trace = atomic_train(method, [env], 23, iterations=1, episodes=8,
                    group_size=8, hidden=8, epochs=1, entropy_coefficient=0)
                self.assertEqual(trace[0]['parameter_decisions'], 16)
                self.assertEqual(trace[0]['invalid_episodes'], 0)
                self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(initial.parameter_head.params, actor.parameter_head.params)))
                completed = [path for path in env.nodes if len(path) == 4]
                self.assertGreater(len(completed), 1)
                self.assertGreater(len({env.nodes[path].frame_bytes for path in completed}), 1)
                for path in completed:
                    self.assertEqual(decode_frame(encode_cold(column(), path)), column()['raw'])

    def test_recipe_catalog_deduplicates_parameters_and_preserves_archived_paths(self):
        from canonical182_policy_paths import stage_signature
        from canonical206_depth10_env import Depth10Env, build_catalog as depth_catalog
        options = build_catalog(p.registry())
        self.assertEqual(len(options), len({stage_signature(o.ids) for o in options}))
        env = RecipeEnv(column(), options, 5000, max_steps=1, complete_on_transition=False)
        for recipe in SEQUENCES:
            if any(SPECS[i].name == lz.ALGORITHM_NAME for i in recipe):
                option_path = env.option_path_for_ids(recipe)
                self.assertEqual(env.expanded(option_path), recipe)
                self.assertEqual(next(q for q in options[option_path[0]].parameters if "min_match" in q), lz.default_parameters())
        depth = Depth10Env(column(), depth_catalog(p.registry()), 5000)
        path = configured({'min_match': 16, 'max_match': 80})
        self.assertEqual(depth.expanded(depth.option_path_for_ids(path)), path)

    def test_fixed_recipe_seed_preserves_historical_min_max(self):
        options = build_catalog(p.registry())
        probe = RecipeEnv(column(), options, 5000, max_steps=1, complete_on_transition=False)
        original = next(path for path in sorted(probe.fixed_recipe_ids)
                        if any('min_match' in q and q != lz.default_parameters() for q in path.parameters))
        frame = encode_cold(column(), original)
        env = RecipeEnv(column(), options, len(frame), max_steps=1, complete_on_transition=False)
        records = env.seed_fixed_recipes([path_record(original)])
        self.assertEqual(records[0]['parameters'], list(original.parameters))
        self.assertEqual(records[0]['compressed_bytes'], len(frame))
        self.assertEqual(env.expanded(env.option_path_for_ids(original)), original)
        self.assertIn(original, env.seeded_fixed_ids)

    def test_parameter_warm_start_and_restricted_mask_joint_probability(self):
        from canonical206_recipe_grpo import _collect
        from search_physical_operator_variants_v2 import distribution
        option = RecipeOption(0, 'native-lz', chain(), ('bytes',), False)
        env = RecipeEnv(column(), [option], 5000, max_steps=1, complete_on_transition=False)
        env.legal = lambda _: (0,)
        values = {'min_match': 8, 'max_match': 16, **window_parameters.default_parameters()}
        warm_path = PolicyPath((0,), parameters=(values,))
        actor = initialize_policy(env, 31, 8)
        features = env.features(())[None, :]
        mask = np.ones((1,len(conditional.CANDIDATES)), bool)
        target = conditional.CANDIDATES.index(values)
        before = distribution(actor.parameter_head, features, mask)[1][0, target]
        actor, trace = recipe_train(env, 31, actor=actor, iterations=1, episodes=8,
            max_rollouts=0, warm_start_paths=[warm_path], warm_start_epochs=1)
        after = distribution(actor.parameter_head, features, mask)[1][0, target]
        self.assertGreater(after, before)
        self.assertEqual(trace, [])
        self.assertEqual(env.expanded(warm_path).parameters[1], {k:values[k] for k in ("min_match", "max_match")})
        choices = (env.parameter_choices((), 0)[0], env.parameter_choices((), 0)[-1])
        env.parameter_choices = lambda _prefix, action: choices if action == 0 else ()
        rollout = _collect(env, actor, 4, np.random.default_rng(33))
        self.assertTrue(np.all(rollout['parameter_legal'].sum(axis=1) == 2))
        np.testing.assert_allclose(conditional._stats(actor, rollout)[2], 1, atol=1e-14)


if __name__ == '__main__': unittest.main()
