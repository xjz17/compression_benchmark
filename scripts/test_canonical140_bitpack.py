"""Shared maximum-input-width packing: wire, typed values, and versioned adapters."""
import copy
import json
from pathlib import Path
import struct
import unittest
import numpy as np

import max_value_bitpack as b
import current_action_parameters as parameters
import optimal_rle_actions as rle
import primitive_operator_action_runtime as p
import unified_action_runtime as u
from canonical182_policy_paths import path_from_names, path_from_record, path_record, concrete_path
from canonical206_search_env import Env, encode_cold, decode_frame, SEQUENCES
from canonical206_recipe_env import RecipeEnv, RecipeOption, build_catalog
from canonical206_recipe_grpo import initialize_policy
from canonical182_policy_checkpoint import validate
from qwen182_action_binding import QwenActionBinding

ROOT = Path(__file__).parents[1]
VERSION = p.ACTION_REGISTRY_VERSION


def independent_payload(values):
    width = max(values, default=0).bit_length()
    bits = ''.join(format(value, f'0{width}b') for value in values) if width else ''
    padded = bits + '0' * ((-len(bits)) % 8)
    body = bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
    return b'MVW1' + struct.pack('<QB', len(values), width) + body


def stages(names, source):
    for name in names: source=u.run_action(name,source)
    return source


def family_packet(variant):
    ints = u.ActionData.ints([100,102,104,106,109,110,115,120])
    if variant == b.SOURCE_VARIANTS[0]:
        return stages(p.TS2DIFF_INT_ACTIONS[:-1], ints), ints
    if variant == b.SOURCE_VARIANTS[1]:
        return stages(p.LECO_ACTIONS[:-1], ints), ints
    if variant == b.SOURCE_VARIANTS[2]:
        source = u.ActionData.floats([1.,1.25,2.5,3.5,4.,7.125,8.,9.])
        return stages(p.ALP_MAIN_ACTIONS[:2], source), source
    return stages(p.REGER_ACTIONS[:4], ints), ints


class MaximumWidthTest(unittest.TestCase):
    def test_counts_and_every_historical_profile_migration(self):
        current = json.loads((ROOT/'benchmark/reference/actions/canonical127.json').read_text())
        self.assertEqual((len(p.POLICY_ACTIONS),len(u.IO_VARIANT_ACTIONS),p.STOP_ACTION_ID),(122,170,127))
        self.assertEqual(len({s.operator_id for s in p.POLICY_ACTIONS}),28)
        self.assertEqual(len([s for s in p.POLICY_ACTIONS if s.operator_id=='bitpack_int']),8)
        self.assertEqual(len(p.POLICY_IO_VARIANTS[b.ALGORITHM_NAME]),5)
        self.assertFalse(set(b.SOURCE_VARIANTS)&set(p.POLICY_ACTION_BY_NAME))
        current_for_old143={old:g['policy_action_id'] for g in current['actions'] for old in g['previous_143_policy_ids']}
        self.assertEqual({i:current_for_old143[i] for i in (25,32,33,34)},{25:22,32:22,33:22,34:22})
        for version in (137,140,143,156,182):
            old = json.loads((ROOT/f'benchmark/reference/actions/canonical{version}.json').read_text())
            expected={old_id:g['policy_action_id'] for g in current['actions'] for old_id in g[f'previous_{version}_policy_ids']}
            covered=0
            for group in old['actions']:
                for variant in group['io_variants']:
                    selected=parameters.parameters_from_name(group['name'])
                    module=parameters.module_for_name(group['name'])
                    if module and selected is None: selected=module.default_parameters()
                    record={'ids':[group['policy_action_id']], 'io_variants':[variant['name']],
                            'parameters':[selected or {}], 'policy_registry_version':f'cpp-canonical-{version}-v1'}
                    path=path_from_record(record)
                    if group['name'] in p.RETIRED_IDENTITY_ACTION_NAMES:
                        self.assertEqual((tuple(path),path.parameters,path.io_variants),((),(),()))
                        self.assertEqual(path_from_record(path_record(path)),path)
                        covered+=1
                        continue
                    canonical=p.ACTION_CANONICAL_NAMES.get(group['name'],group['name'])
                    if canonical in p.COMPOSITE_POLICY_REPLACEMENTS:
                        replacement=p.COMPOSITE_POLICY_REPLACEMENTS[canonical]
                        self.assertEqual(tuple(path),tuple(item[0] for item in replacement))
                        self.assertEqual(path.io_variants,tuple(item[1] for item in replacement))
                        self.assertEqual(path.parameters,tuple({} for _ in replacement))
                        self.assertEqual(path_from_record(path_record(path)),path)
                        covered+=1
                        continue
                    self.assertEqual(tuple(path),(expected[group['policy_action_id']],))
                    self.assertEqual(path.parameters,(selected or {},))
                    self.assertEqual(path.io_variants,(rle.upgrade_variant(parameters.canonical_io_variant(variant['name'])),))
                    self.assertEqual(path_from_record(path_record(path)),path)
                    covered+=1
            self.assertEqual(covered,{137:169,140:168,143:167,156:180,182:206}[version])
        for version, stop in [('cpp-canonical-137-v1',137),('cpp-canonical-143-v1',143),('cpp-canonical-156-v1',156),('cpp-canonical-182-v1',182),('cpp-canonical-206-v1',206),('legacy-225-v1',225)]:
            self.assertEqual(p.migrate_policy_action(stop,version),{'policy_action_id':127,'io_variant':None,'parameters':{}})

    def test_general_values_have_exact_maximum_width(self):
        cases=[[],[0],[0]*100,[0,1,7,2],[100,101,102],[2**63-1],[2**63,2**64-1,0],list(range(129))]
        for values in cases:
            raw=struct.pack('<'+'Q'*len(values),*values)
            sources=[u.ActionData.bytes(raw)]
            if all(v<2**63 for v in values):sources.append(u.ActionData.ints(values))
            for source in sources:
                with self.subTest(values=values[:5],kind=source.kind):
                    encoded=u.run_policy_action(b.ALGORITHM_NAME,source,registry_version=VERSION)
                    header,payload=u._unpack_result(encoded)
                    self.assertEqual(header['io_variant'],b.PLAIN_VARIANT)
                    self.assertEqual(payload,independent_payload(values))
                    self.assertEqual(u.run_policy_action(b.ALGORITHM_NAME,encoded,registry_version=VERSION,reverse=True),source)
                    self.assertEqual(p.encode_registered_exact_action_pipeline(source.payload,[b.ALGORITHM_NAME],logical_type='int' if source.kind=='int64' else 'bytes'),payload)
                    ports=u.run_policy_action(b.ALGORITHM_NAME,source,registry_version=VERSION,output='streams')
                    self.assertEqual(ports.metadata['bit_width'],max(values,default=0).bit_length())
                    self.assertEqual(ports.metadata['value_count'],len(values))
                    self.assertEqual([(s.name,s.kind) for s in ports.streams],[('packed','bytes')])
                    self.assertEqual(u.run_policy_action(b.ALGORITHM_NAME,ports,registry_version=VERSION,reverse=True),source)
        # Demonstrates that no implicit minimum subtraction was inserted.
        encoded=u.run_policy_action(b.ALGORITHM_NAME,u.ActionData.ints([100,101]),registry_version=VERSION)
        self.assertEqual(u._unpack_result(encoded)[1][12],7)

    def test_negative_float_malformed_and_metadata_rejected(self):
        for bad in (u.ActionData.ints([-1]),u.ActionData.ints([-2**63,0]),u.ActionData.floats([1.]),u.ActionData.bytes(b'123')):
            with self.subTest(kind=bad.kind),self.assertRaises(u.ActionRuntimeError):
                u.run_policy_action(b.ALGORITHM_NAME,bad,registry_version=VERSION)
        good=u.run_policy_action(b.ALGORITHM_NAME,u.ActionData.ints([0,1,7,2]),registry_version=VERSION)
        header,payload=u._unpack_result(good)
        for corrupt in (payload[:-1],payload+b'\0',payload[:12]+b'\x41'+payload[13:],payload[:-1]+bytes([payload[-1]|1]),payload[:4]+struct.pack('<Q',2**64-1)+payload[12:]):
            with self.subTest(corrupt=corrupt.hex()),self.assertRaises(u.ActionRuntimeError):
                u.run_policy_action(b.ALGORITHM_NAME,u._pack_existing_result(header,corrupt),registry_version=VERSION,reverse=True)
        for key,value in [('source_count',100),('source_kind','float64'),('io_variant',b.SOURCE_VARIANTS[0])]:
            changed=copy.deepcopy(header);changed[key]=value
            with self.subTest(key=key),self.assertRaises(u.ActionRuntimeError):
                u.run_policy_action(b.ALGORITHM_NAME,u._pack_existing_result(changed,payload),registry_version=VERSION,reverse=True)

    def test_raw_magic_bytes_are_unsigned_values_not_source_packets(self):
        for magic in b.INPUT_MAGICS.values():
            raw=u.ActionData.bytes(magic+b'\0'*4)
            encoded=u.run_policy_action(b.ALGORITHM_NAME,raw,registry_version=VERSION)
            header,payload=u._unpack_result(encoded)
            self.assertEqual(header['io_variant'],b.PLAIN_VARIANT)
            self.assertEqual(payload,independent_payload(list(struct.unpack('<Q',raw.payload))))
            self.assertEqual(u.run_policy_action(b.ALGORITHM_NAME,encoded,registry_version=VERSION,reverse=True),raw)

    def test_four_source_profiles_and_old_headers_keep_packets(self):
        for variant in b.SOURCE_VARIANTS:
            before,_=family_packet(variant)
            auto=u.run_policy_action(b.ALGORITHM_NAME,before,registry_version=VERSION)
            explicit=u.run_policy_action(b.ALGORITHM_NAME,before,registry_version=VERSION,io_variant=variant)
            self.assertEqual(auto,explicit)
            header,payload=u._unpack_result(auto)
            self.assertEqual(header['io_variant'],variant)
            self.assertEqual(u.run_policy_action(b.ALGORITHM_NAME,auto,registry_version=VERSION,reverse=True),before)
            ports=u.run_policy_action(b.ALGORITHM_NAME,before,registry_version=VERSION,io_variant=variant,output='streams')
            self.assertEqual(u.run_policy_action(b.ALGORITHM_NAME,ports,registry_version=VERSION,reverse=True),before)
            wrong=next(v for v in b.SOURCE_VARIANTS if v!=variant)
            with self.assertRaises(u.ActionRuntimeError):u.run_policy_action(wrong,auto,registry_version=VERSION,reverse=True)
            for oldversion in (143,156,182):
                old=json.loads((ROOT/f'benchmark/reference/actions/canonical{oldversion}.json').read_text())
                old_id=next(g['policy_action_id'] for g in old['actions'] if g['name']==variant)
                h=copy.deepcopy(header);h.update(policy_registry_version=f'cpp-canonical-{oldversion}-v1',policy_action_id=old_id,policy_algorithm=variant)
                legacy=u._pack_existing_result(h,payload)
                self.assertEqual(u.run_policy_action(b.ALGORITHM_NAME,legacy,registry_version=VERSION,reverse=True),before)

    def test_fixed_recipes_keep_source_variants_and_no_cross_adapter_grammar(self):
        registry=p.registry()
        def walk(legacy,names,variants):
            if isinstance(legacy,dict):
                for k in legacy:walk(legacy[k],names[k],variants[k])
                return
            for old,name,variant in zip(legacy,names,variants):
                if old in b.SOURCE_VARIANTS:
                    self.assertEqual(name,b.ALGORITHM_NAME);self.assertEqual(variant,old)
        walk(registry['benchmark_86_legacy_action_pipelines'],registry['benchmark_86_canonical_action_pipelines'],registry['benchmark_86_io_variant_pipelines'])
        for path in SEQUENCES:
            for index,variant in enumerate(path.io_variants):
                if variant in b.SOURCE_VARIANTS and index:
                    self.assertEqual(path.io_variants[index-1],b.PRODUCER_VARIANTS[variant])
        profiles=[path_from_names([v]) for v in (*b.SOURCE_VARIANTS,b.PLAIN_VARIANT)]
        self.assertEqual(len({tuple(v) for v in profiles}),1)
        self.assertEqual(len(set(profiles)),5)
        self.assertEqual(path_from_names([b.ALGORITHM_NAME]).io_variants,(b.PLAIN_VARIANT,))

    def test_search_recipe_checkpoint_and_unsigned_qwen(self):
        col={'raw':struct.pack('<64q',*([0,1,7,2]*16)),'kind':'int64','max_point':0,'name':'plain','dataset':'bitpack'}
        env=Env(col,0,1);path=path_from_names([b.ALGORITHM_NAME])
        self.assertIn(path[0],env.legal(()))
        out=env.transition((),path[0],io_variant=b.PLAIN_VARIANT)
        self.assertIsNotNone(out,env.records[-1]);self.assertEqual(decode_frame(encode_cold(col,out)),col['raw'])
        options=build_catalog(p.registry());self.assertTrue(any(o.ids.io_variants==(b.PLAIN_VARIANT,) for o in options))
        actor=initialize_policy(env,7,4);self.assertEqual(actor.params[2].shape[1],128)
        actor.policy_identity['policy_registry_version']='cpp-canonical-143-v1'
        with self.assertRaises(ValueError):validate(actor,env)
        binding=QwenActionBinding();record=path_record(path)
        raw=struct.pack('<3Q',2**64-1,2**63,0)
        self.assertEqual(binding.root_data(raw,record).kind,'bytes')
        wire=binding.encode_recipe(raw,record)
        self.assertEqual(binding.decode_recipe(wire),raw)

if __name__=='__main__':unittest.main()
