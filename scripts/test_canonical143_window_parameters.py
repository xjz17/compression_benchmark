"""Window merging: independent migration oracles, packets, cache, and policy masks."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np

import optimal_rle_actions as rle
import primitive_operator_action_runtime as p
import unified_action_runtime as u
import native_lz_parameters as lz
import native_window_parameters as window
import current_action_parameters as parameters
import canonical156_parameter_policy as conditional
from canonical182_policy_paths import path_from_names, path_from_record, path_record, concrete_path
from canonical206_search_env import Env, encode_cold, decode_frame
from canonical206_recipe_env import RecipeEnv, RecipeOption
from canonical206_recipe_grpo import initialize_policy
from canonical182_policy_checkpoint import validate

ROOT = Path(__file__).parents[1]
VERSION = p.ACTION_REGISTRY_VERSION
NAMES = (window.ALGORITHM_NAME, lz.ALGORITHM_NAME,
         'op.match_parse_policy.native_lz.aligned8_greedy',
         'op.match_token_format.native_lz.ctrl8_offset16_len8.bias3')


class WindowParametersTest(unittest.TestCase):
    def test_manifest_and_all_156_migrations(self):
        current = json.loads((ROOT/'benchmark/reference/actions/canonical127.json').read_text())
        old = json.loads((ROOT/'benchmark/reference/actions/canonical156.json').read_text())
        self.assertEqual(len(p.POLICY_ACTIONS), 122)
        self.assertEqual(sum(map(len, p.POLICY_IO_VARIANTS.values())), 170)
        self.assertEqual(len({g['operator_id'] for g in current['actions']}), 28)
        self.assertEqual([g['document_id'] for g in current['actions']], [f'P{i+1:03d}' for i in range(127)])
        from137 = {old_id: g['policy_action_id'] for g in current['actions']
                   for old_id in g['previous_137_policy_ids']}
        for group in old['actions']:
            old_id = group['policy_action_id']
            expected_id = old_id if old_id < 87 else 87 if old_id <= 100 else old_id - 13
            expected_id = 25 if expected_id in (32,33,34) else expected_id-3 if expected_id>34 else expected_id
            expected_id = 14 if 14 <= expected_id <= 17 else expected_id-3 if expected_id>17 else expected_id
            expected_id = from137[expected_id]
            module = parameters.module_for_name(group['name'])
            for variant in group['io_variants']:
                with self.subTest(old_id=old_id, variant=variant['name']):
                    selected = module.parameters_from_name(group['name']) or module.default_parameters() if module else {}
                    record = {'ids':[old_id], 'io_variants':[variant['name']],
                              'parameters':[selected], 'policy_registry_version':'cpp-canonical-156-v1'}
                    path = path_from_record(record)
                    if group['name'] in p.RETIRED_IDENTITY_ACTION_NAMES:
                        self.assertEqual((tuple(path),path.parameters,path.io_variants),((),(),()))
                        self.assertEqual(path_from_record(path_record(path)),path)
                        continue
                    canonical=p.ACTION_CANONICAL_NAMES.get(group['name'],group['name'])
                    if canonical in p.COMPOSITE_POLICY_REPLACEMENTS:
                        replacement=p.COMPOSITE_POLICY_REPLACEMENTS[canonical]
                        self.assertEqual(tuple(path),tuple(item[0] for item in replacement))
                        self.assertEqual(path.io_variants,tuple(item[1] for item in replacement))
                        self.assertEqual(path.parameters,tuple({} for _ in replacement))
                        self.assertEqual(path_from_record(path_record(path)),path)
                        continue
                    self.assertEqual(tuple(path), (expected_id,))
                    self.assertEqual(path.parameters, (selected,))
                    self.assertEqual(path.io_variants, (rle.upgrade_variant(parameters.canonical_io_variant(variant['name'])),))
                    self.assertEqual(path_from_record(path_record(path)), path)
        self.assertEqual(p.migrate_policy_action(156, 'cpp-canonical-156-v1'),
                         {'policy_action_id':127,'io_variant':None,'parameters':{}})
        self.assertEqual(len([a for a in p.POLICY_ACTIONS if a.operator_id == 'match_window_model']), 2)

    def test_window_scalar_validation_and_stored_replay(self):
        src = u.ActionData.bytes(b'ABCDEFGH'*128)
        for value in (1, 7, 4096, 65000, 65535):
            with self.subTest(value=value):
                choice = {'window_bytes':value}
                encoded = u.run_policy_action(window.ALGORITHM_NAME, src, registry_version=VERSION, parameters=choice)
                header, packet = u._unpack_result(encoded)
                self.assertEqual(p.native_lz_stage_packet_context(packet)['window'], value)
                self.assertEqual(header['action_parameters'], choice)
                self.assertEqual(u.run_policy_action(window.ALGORITHM_NAME, encoded, registry_version=VERSION, reverse=True), src)
                ports = u.run_policy_action(window.ALGORITHM_NAME, src, registry_version=VERSION, parameters=choice, output='streams')
                self.assertEqual(u.run_policy_action(window.ALGORITHM_NAME, ports, registry_version=VERSION, reverse=True), src)
                with self.assertRaises(u.ActionRuntimeError):
                    u.run_policy_action(window.ALGORITHM_NAME, encoded, registry_version=VERSION, reverse=True,
                                        parameters={'window_bytes':2 if value != 2 else 3})
                for mutation in ({'window_bytes':2 if value != 2 else 3}, {}, {'window_bytes':True}):
                    corrupt = copy.deepcopy(header); corrupt['action_parameters'] = mutation
                    with self.assertRaises(u.ActionRuntimeError):
                        u.run_policy_action(window.ALGORITHM_NAME, u._pack_existing_result(corrupt, packet), registry_version=VERSION, reverse=True)
        for bad in ({}, {'window_bytes':False},{'window_bytes':0},{'window_bytes':65536},
                    {'window_bytes':1.5},{'window_bytes':5,'min_match':8}):
            with self.subTest(bad=bad), self.assertRaises((u.ActionRuntimeError, ValueError)):
                u.run_policy_action(window.ALGORITHM_NAME, src, registry_version=VERSION, parameters=bad)

    def test_historical_names_never_override_explicit_parameters(self):
        cases = [('op.match_window_model.native_lz.window32768', {'window_bytes':65535}),
                 ('op.lz_match.native_lz.aligned8.min3.max258', {'min_match':8,'max_match':256})]
        for name, choices in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError): path_from_names([name], [choices])
                same = parameters.parameters_from_name(name)
                self.assertEqual(path_from_names([name], [same]).parameters, (same,))
        with self.assertRaises(ValueError):
            p.encode_registered_exact_action_pipeline(b'ABCDEFGH'*32,
                ['op.match_window_model.native_lz.window32768', *NAMES[1:]],
                io_variants=['op.match_window_model.native_lz.window65535', None, None, None])
        for old_version in ('cpp-canonical-156-v1','cpp-canonical-182-v1','cpp-canonical-206-v1'):
            manifest = json.loads((ROOT/f"benchmark/reference/actions/canonical{old_version.split('-')[2]}.json").read_text()) if old_version != 'cpp-canonical-206-v1' else None
            row = next(g for g in manifest['actions'] if g['name'].endswith('native_lz.window32768')) if manifest else None
            old_id = row['policy_action_id'] if row else next(a.action_id for a in p.LEGACY_206_POLICY_ACTIONS if a.name.endswith('native_lz.window32768'))
            with self.assertRaises(ValueError):
                p.migrate_policy_action(old_id,old_version,parameters={'window_bytes':65535})

    def test_old156_window_packet_inverse_without_new_parameter_header(self):
        raw = u.ActionData.bytes(b'01234567'*200)
        for value in window.WINDOW_CANDIDATES:
            name = f'op.match_window_model.native_lz.window{value}'
            old = p.algorithm_io.previous156.GROUP_BY_NAME[name]
            packet = p.encode_exact_native_lz_stage(name, raw.payload)
            encoded = u._pack_result((name,), raw, packet, None,
                extra_metadata={'policy_registry_version':'cpp-canonical-156-v1',
                                'policy_algorithm':name,'policy_action_id':old['policy_action_id'], 'io_variant':name})
            self.assertEqual(u.run_policy_action(window.ALGORITHM_NAME,encoded,registry_version=VERSION,reverse=True),raw)

    def test_distance_changes_payload_and_cache_keeps_both_parameters(self):
        rng = np.random.default_rng(8)
        block = rng.bytes(12288)
        column = {'raw':block*2,'kind':'int64','max_point':0,'name':'window-distance','dataset':'synthetic'}
        env = Env(column,0,4)
        finished=[]
        for width in (4096,65535):
            path = path_from_names(NAMES, ({'window_bytes':width},lz.default_parameters(),{},{}))
            prefix=()
            for action,variant,choice in zip(path,path.io_variants,path.parameters):
                prefix=env.transition(prefix,action,io_variant=variant,parameters=choice)
                self.assertIsNotNone(prefix,env.records[-1])
            cold=encode_cold(column,prefix)
            self.assertEqual(decode_frame(cold),column['raw'])
            self.assertEqual(prefix.parameters[0],{'window_bytes':width})
            finished.append((prefix,len(cold)))
        self.assertNotEqual(finished[0][0],finished[1][0])
        self.assertGreater(finished[0][1],finished[1][1])
        self.assertNotEqual(env.features(finished[0][0]).tolist(),env.features(finished[1][0]).tolist())

    def test_masks_joint_recipe_and_checkpoint_identity(self):
        column={'raw':b'ABCDEFGH'*32,'kind':'int64','max_point':0,'name':'mask','dataset':'synthetic'}
        atomic=Env(column,0,4)
        w_id=p.POLICY_ACTION_BY_NAME[window.ALGORITHM_NAME].action_id
        l_id=p.POLICY_ACTION_BY_NAME[lz.ALGORITHM_NAME].action_id
        self.assertEqual(int(conditional.legal_mask(atomic,(),w_id).sum()),14)
        self.assertEqual(int(conditional.legal_mask(atomic,(),l_id).sum()),147)
        self.assertEqual(len(conditional.CANDIDATES),2235)
        path=path_from_names(NAMES,({'window_bytes':4096},{'min_match':16,'max_match':80},{},{}))
        option=RecipeOption(0,'native',path,('bytes',),False)
        recipe=RecipeEnv(column,[option],2000,max_steps=1,complete_on_transition=False)
        self.assertEqual(int(conditional.legal_mask(recipe,(),0).sum()),2058)
        flattened=recipe.option_path_for_ids(path)
        self.assertEqual(recipe.expanded(flattened),path)
        chosen=recipe.parameter_choices((),0)[-1]
        prefix=recipe.transition((),0,parameters=chosen)
        self.assertIsNotNone(prefix)
        self.assertEqual(recipe.expanded(prefix).parameters[0],{'window_bytes':chosen['window_bytes']})
        self.assertEqual(recipe.expanded(prefix).parameters[1],{'min_match':chosen['min_match'],'max_match':chosen['max_match']})
        actor=initialize_policy(atomic,10,4)
        actor.policy_identity['policy_registry_version']='cpp-canonical-156-v1'
        with self.assertRaises(ValueError):validate(actor,atomic)

if __name__ == '__main__': unittest.main()
