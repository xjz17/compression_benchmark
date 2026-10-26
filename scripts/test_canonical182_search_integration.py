"""Algorithm-space migration checks; tiny probes, not a new benchmark claim."""
import copy
import json
from pathlib import Path
import pickle
import struct
import tempfile
import unittest

import numpy as np
import optimal_rle_actions as rle
import primitive_operator_action_runtime as p
import native_lz_parameters as lz
import current_action_parameters as action_parameters
from canonical182_policy_paths import (PolicyPath, concrete_path, path_from_record,
    path_from_names, path_record, variants_for, require_current_registry)
from canonical182_policy_checkpoint import load, save, validate
from canonical206_search_env import (Env, STOP, SPECS, REFS, make_frame, decode_frame,
    encode_cold, grammar_manifest)
from canonical206_recipe_env import RecipeOption, RecipeEnv, build_catalog
from canonical206_recipe_grpo import initialize_policy
from canonical206_rl_policy import train, propose


def old(ids):
    return path_from_record({'ids': ids, 'policy_registry_version': 'cpp-canonical-206-v1'})


def column(kind='int64'):
    fmt = '<8q' if kind == 'int64' else '<8d'
    return {'kind':kind, 'raw':struct.pack(fmt, 1,2,3,4,4,4,5,6), 'max_point':0,
            'name':'tiny', 'dataset':'integration'}


class MigrationTests(unittest.TestCase):
    def test_all_old_steps_preserve_concrete_variant(self):
        covered=[]
        for spec in p.LEGACY_206_POLICY_ACTIONS:
            path=old([spec.action_id])
            canonical = p.ACTION_CANONICAL_NAMES.get(spec.name, spec.name)
            if canonical in p.RETIRED_IDENTITY_ACTION_NAMES:
                self.assertEqual(tuple(path), ())
                self.assertEqual(path_from_record(path_record(path)), path)
                continue
            if canonical in p.COMPOSITE_POLICY_REPLACEMENTS:
                replacement = p.COMPOSITE_POLICY_REPLACEMENTS[canonical]
                self.assertEqual(tuple(path), tuple(item[0] for item in replacement))
                self.assertEqual(path.io_variants, tuple(item[1] for item in replacement))
                self.assertEqual(path_from_record(path_record(path)), path)
                covered.extend(path.io_variants)
                continue
            expected = spec.name
            if 175 <= spec.action_id <= 180:
                which = 'blosc1' if spec.action_id <= 177 else 'blosc2'
                expected = f'op.block_state_control.{which}_residual_core.cpp_v2'
                self.assertEqual(path.parameters, ({'typesize': (1,4,8)[(spec.action_id-175)%3]},))
            elif spec.action_id in (182,183,204):
                mode,level = {182:('gzip_prefix',6),183:('zlib_prefix',9),204:('corad_ctk2',9)}[spec.action_id]
                expected = f'op.block_state_control.deflate_residual_core.{mode}.cpp_v2'
                self.assertEqual(path.parameters, ({'level':level},))
            elif action_parameters.module_for_name(spec.name):
                expected = action_parameters.module_for_name(spec.name).ALGORITHM_NAME
            self.assertEqual(path.io_variants, (rle.upgrade_variant(expected),))
            self.assertIn(path[0],SPECS)
            self.assertEqual(path_from_record(path_record(path)),path)
            covered.extend(path.io_variants)
        self.assertEqual(len(set(covered)),165)
        self.assertEqual(len(SPECS),122)
        self.assertEqual(STOP,127)
        self.assertEqual(grammar_manifest()['selectable_count'],119)
        self.assertIn(old([7])[0],REFS)
        self.assertNotEqual(old([7])[0],7)

    def test_variant_identity_survives_path_operations_and_pickle(self):
        first,second,third=old([158]),old([159]),old([160])
        self.assertEqual(list(first),list(second))
        self.assertNotEqual(first,second)
        self.assertEqual(len({first,second,third}),3)
        self.assertNotEqual(first,tuple(first))
        path=first+old([74])
        self.assertEqual(path[:1],first)
        self.assertEqual(path[1:],old([74]))
        self.assertEqual(copy.deepcopy(path),path)
        self.assertEqual(pickle.loads(pickle.dumps(path)),path)
        self.assertEqual(path_from_record(json.loads(json.dumps(path_record(path)))),path)
        with self.assertRaises(AttributeError): path.io_variants=()
        with self.assertRaises(ValueError): concrete_path(list(first))
        with self.assertRaises(ValueError): concrete_path(first,old([7]).io_variants)

    def test_old_ids_require_version_and_old_baselines_require_remeasurement(self):
        with self.assertRaises(ValueError): path_from_record({'ids':[80]})
        for bad in ([1.5], [True], [-1], [156]):
            with self.assertRaises(ValueError): concrete_path(bad)
        with self.assertRaisesRegex(ValueError,'remeasure'):
            require_current_registry({'policy_registry_version':'cpp-canonical-206-v1'})
        a,b=old([80]),old([86])
        self.assertEqual(list(a),list(b))
        self.assertNotEqual(a,b)

    def test_real_variant_cache_and_cold_frame_replay(self):
        c=column();env=Env(c,0)
        self.assertEqual(env.action_count,128)
        self.assertEqual(env.feature_dim,8+127+12)
        paths=[]
        for i in (158,159,160):
            path=old([i]);nxt=env.transition((),path[0],io_variant=path.io_variants[0])
            self.assertIsNotNone(nxt,env.records[-1])
            paths.append(nxt)
            frame=make_frame(c,env.nodes[nxt])
            self.assertEqual(decode_frame(frame),c['raw'])
            self.assertEqual(encode_cold(c,nxt),frame)
            self.assertEqual(env.transition((),path[0],io_variant=path.io_variants[0]),nxt)
        self.assertEqual(len(env.nodes),4)
        self.assertEqual(len(set(paths)),3)
        self.assertEqual(env.cache_hits,3)
        self.assertEqual(env.records[0]['policy_registry_version'],p.ACTION_REGISTRY_VERSION)
        self.assertEqual(env.records[0]['io_variants'],list(paths[0].io_variants))

    def test_ambiguous_variant_is_not_silently_selected(self):
        env=Env(column(),0);action=old([158])[0]
        self.assertEqual(len(env.legal_io_variants((),action)),3)
        self.assertIsNone(env.transition((),action))
        self.assertIn('io_variant',env.records[-1]['error'])
        self.assertEqual(len(env.nodes),1)
        self.assertIsNone(env.transition((),action))
        self.assertEqual(len(env.records),1)

    def test_two_reference_stages_use_original_profiles(self):
        c=column();env=Env(c,0);chain=old([7,11]);prefix=()
        for action,variant in zip(chain,chain.io_variants):
            prefix=env.transition(prefix,action,io_variant=variant)
            self.assertIsNotNone(prefix,env.records[-1])
        self.assertEqual(prefix,chain)
        self.assertEqual(env.nodes[prefix].layers,1)
        self.assertEqual(encode_cold(c,chain),make_frame(c,env.nodes[prefix]))
        self.assertEqual(decode_frame(encode_cold(c,chain)),c['raw'])

    def test_recipe_options_with_same_algorithm_keep_variants(self):
        options=[]
        for idx,i in enumerate((158,159,160)):
            path=old([i])
            options.append(RecipeOption(idx,'shuffle',path,('bytes',),False))
        env=RecipeEnv(column(),options,1000,max_steps=1,complete_on_transition=False)
        for option in options:
            prefix=env.transition((),option.option_id)
            self.assertIsNotNone(prefix)
            self.assertEqual(env.expanded(prefix),option.ids)
            self.assertEqual(env.records[-1]['io_variants'],list(option.io_variants))
        self.assertEqual(len(env.completed),3)
        self.assertEqual(len(env.base.nodes),4)

    def test_registered_nine_step_pipeline_cold_replay(self):
        from canonical206_depth10_env import Depth10Env, build_catalog as depth_catalog
        c=column('float64');chain=old([93,94,95,96,165,166,167,74,181])
        env=Depth10Env(c,depth_catalog(p.registry()),1000000)
        row=env.execute_ids(chain)
        self.assertEqual(row['status'],'ok',row)
        frame=encode_cold(c,chain)
        self.assertEqual(decode_frame(frame),c['raw'])
        self.assertEqual(len(frame),row['compressed_bytes'])
        self.assertEqual(row['io_variants'],list(chain.io_variants))

    def test_checkpoint_rejects_legacy_and_same_shape_changed_catalog(self):
        env=Env(column(),0)
        actor=initialize_policy(env,17,8)
        self.assertEqual(actor.params[2].shape[1],128)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.npz';save(actor,path)
            restored=load(path,env)
            for a,b in zip(actor.params,restored.params): np.testing.assert_array_equal(a,b)
            changed=Env(column(),0);changed.catalog_fingerprint='different-order-same-size'
            with self.assertRaisesRegex(ValueError,'catalog'): validate(restored,changed)
            old_version=Env(column(),0);old_version.policy_registry_version='cpp-canonical-137-v1'
            with self.assertRaisesRegex(ValueError,'catalog'): validate(restored,old_version)
            np.savez(path,*actor.params)
            with self.assertRaisesRegex(ValueError,'legacy'): load(path,env)

    def test_checkpoint_distinguishes_recipe_and_depth_feature_horizons(self):
        from canonical206_depth10_env import Depth10Env
        path=old([158])
        options=[RecipeOption(0,'shuffle',path,('bytes',),False)]
        recipe=RecipeEnv(column(),options,1000,max_steps=3)
        depth=Depth10Env(column(),options,1000,max_atomic_depth=10)
        self.assertEqual(recipe.action_count,depth.action_count)
        self.assertEqual(recipe.feature_dim,depth.feature_dim)
        actor=initialize_policy(recipe,19,8)
        with self.assertRaisesRegex(ValueError,'features mismatch'): validate(actor,depth)
        horizon=RecipeEnv(column(),options,1000,max_steps=2)
        with self.assertRaisesRegex(ValueError,'features mismatch'): validate(actor,horizon)

    def test_ppo_and_grpo_use_current_128_logit_head(self):
        # A STOP-only control isolates action-head sizing from codec sampling.
        for method in ('ppo','grpo'):
            env=Env(column(),0,max_steps=1)
            env.legal=lambda _prefix:(STOP,)
            actor,trace=train(method,[env],7,iterations=1,episodes=8,hidden=8)
            self.assertEqual(actor.params[2].shape[1],128)
            self.assertEqual(propose(actor,env,np.random.default_rng(3)),())
            self.assertEqual(len(trace),1)

    def test_block_recipe_parameter_profiles_replay_without_key_collisions(self):
        import block_state_parameters as block
        options=build_catalog(p.registry())
        env=RecipeEnv(column(),options,1000000)
        seen=set()
        for option in options:
            if not any(SPECS[i].name in block.BY_NAME for i in option.ids):
                continue
            for profile in option.fixed_parameter_profiles or (option.parameters,):
                chain=concrete_path(option.ids,parameters=profile)
                option_path=env.option_path_for_ids(chain)
                self.assertEqual(env.expanded(option_path),chain)
                scoped=dict(option_path.parameters[0])
                scoped['unexpected']=123
                with self.assertRaises(ValueError):
                    env.expanded(PolicyPath(option_path,parameters=(scoped,)))
            for i in option.ids:
                if SPECS[i].name in block.BY_NAME:seen.add(SPECS[i].name)
        self.assertEqual(seen,set(block.BY_NAME))

    def test_parameter_masks_respect_fixed_blosc_input_width(self):
        import block_state_parameters as block
        from canonical156_parameter_policy import legal_mask
        from canonical206_search_env import SEQUENCES
        env=Env(column(),0,max_steps=8)
        for sequence in SEQUENCES:
            if SPECS[sequence[-1]].name not in (block.BLOSC1,block.BLOSC2):
                continue
            name=SPECS[sequence[-1]].name
            width=sequence.parameters[-1]['typesize']
            key=block.BY_NAME[name].option_key
            self.assertEqual(env.parameter_choices(sequence[:-1],sequence[-1]),({key:width},))
            self.assertEqual(legal_mask(env,sequence[:-1],sequence[-1]).sum(),1)
        options=build_catalog(p.registry())
        recipe=RecipeEnv(column(),options,1000000)
        for option in options:
            choices=recipe.parameter_choices((),option.option_id)
            if choices:
                self.assertEqual(legal_mask(recipe,(),option.option_id).sum(),len(choices))
            if any(SPECS[i].name in (block.BLOSC1,block.BLOSC2) for i in option.ids):
                self.assertEqual(len(choices),1)
                self.assertEqual(recipe.expanded(recipe.option_path_for_ids(option.ids)),option.ids)

    def test_deflate_parameters_change_real_edges_features_and_cache(self):
        import block_state_parameters as block
        from canonical156_parameter_policy import legal_mask
        env=Env(column(),0,max_steps=8)
        source_path=path_from_names(p.GZIP_EXACT_ACTIONS[:2])
        prefix=()
        for i,v,q in zip(source_path,source_path.io_variants,source_path.parameters):
            prefix=env.transition(prefix,i,io_variant=v,parameters=q or None)
            self.assertIsNotNone(prefix)
        action=p.POLICY_ACTION_BY_NAME[block.DEFLATE].action_id
        self.assertEqual(legal_mask(env,prefix,action).sum(),10)
        a=env.transition(prefix,action,io_variant=block.DEFLATE_VARIANTS[0],parameters={'deflate_level':6})
        b=env.transition(prefix,action,io_variant=block.DEFLATE_VARIANTS[0],parameters={'deflate_level':9})
        self.assertIsNotNone(a);self.assertIsNotNone(b);self.assertNotEqual(a,b)
        self.assertAlmostEqual(env.features(a)[-10],6/9)
        self.assertAlmostEqual(env.features(b)[-10],1)
        before=env.cache_hits
        self.assertEqual(env.transition(prefix,action,io_variant=block.DEFLATE_VARIANTS[0],parameters={'deflate_level':6}),a)
        self.assertEqual(env.cache_hits,before+1)
        with self.assertRaises(ValueError):
            env.transition(prefix,action,parameters={'deflate_level':6,'extra':1})


if __name__=='__main__': unittest.main()
