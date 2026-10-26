"""Unified minimum FOR: numeric domains, real streams, old recipes, and versions."""
import copy
import json
from pathlib import Path
import struct
import unittest

import minimum_for as f
import max_value_bitpack as b
import native_action_parameters as params
import current_action_parameters as current_params
import optimal_rle_actions as rle
import primitive_operator_action_runtime as p
import unified_action_runtime as u
from action_semantic_ports import ActionPorts, TypedStream, PortSchemaError, from_ports
from canonical182_policy_paths import path_from_names, path_from_record, path_record
from canonical206_search_env import Env, encode_cold, decode_frame, SEQUENCES
from canonical206_recipe_env import build_catalog
from qwen182_action_binding import QwenActionBinding
import operator_taxonomy_v30 as graph1
import operator_taxonomy_v30_v2 as graph2

ROOT=Path(__file__).parents[1]
VERSION=p.ACTION_REGISTRY_VERSION

def packet(variant):
    ints=u.ActionData.ints([100,102,104,107,120,10000,10001,10002])
    if variant==f.SOURCE_VARIANTS[0]:return ints
    if variant==f.SOURCE_VARIANTS[1]:names=p.BOS_BOSA_ACTIONS[:1]
    elif variant==f.SOURCE_VARIANTS[2]:names=p.REGER_ACTIONS[:3]
    elif variant==f.SOURCE_VARIANTS[3]:names=p.TS2DIFF_INT_ACTIONS[:1]
    elif variant==f.SOURCE_VARIANTS[4]:names=p.BOS_BOSA_ACTIONS[:3]
    else:
        ints=u.ActionData.floats([0.5,1.25,2.5,3.75,9.0]);names=p.ALP_MAIN_ACTIONS[:1]
    for name in names:ints=u.run_action(name,ints)
    return ints


class MinimumForTests(unittest.TestCase):
    def test_inventory_and_all_historical_profiles(self):
        self.assertEqual((VERSION,len(p.POLICY_ACTIONS),len(u.IO_VARIANT_ACTIONS),p.STOP_ACTION_ID),('cpp-canonical-127-v3',122,170,127))
        self.assertEqual([s.name for s in p.POLICY_ACTIONS if s.operator_id=='frame_of_reference'],[f.ALGORITHM_NAME])
        self.assertEqual(set(p.POLICY_IO_VARIANTS[f.ALGORITHM_NAME]),set((*f.SOURCE_VARIANTS,f.PLAIN_VARIANT)))
        current=p.algorithm_io.GROUPS
        self.assertEqual([g['document_id'] for g in current],[f'P{i+1:03d}' for i in range(127)])
        for version,expected_count in [(137,169),(140,168),(143,167),(156,180),(182,206)]:
            old=json.loads((ROOT/f'benchmark/reference/actions/canonical{version}.json').read_text())
            crosswalk={old_id:g['policy_action_id'] for g in current for old_id in g[f'previous_{version}_policy_ids']}
            seen=0
            for group in old['actions']:
                for variant in group['io_variants']:
                    migrated=p.migrate_policy_action(group['policy_action_id'],f'cpp-canonical-{version}-v1',io_variant=variant['name'])
                    if group['name'] in p.RETIRED_IDENTITY_ACTION_NAMES:
                        self.assertEqual(migrated['migration_kind'],'retired_identity_bypass')
                        self.assertEqual(migrated['replacement_policy_action_ids'],[])
                        self.assertNotIn('policy_action_id',migrated)
                        seen+=1
                        continue
                    canonical=p.ACTION_CANONICAL_NAMES.get(group['name'],group['name'])
                    if canonical in p.COMPOSITE_POLICY_REPLACEMENTS:
                        replacement=p.COMPOSITE_POLICY_REPLACEMENTS[canonical]
                        self.assertEqual(migrated['migration_kind'],'replacement_recipe')
                        self.assertEqual(migrated['replacement_policy_action_ids'],[item[0] for item in replacement])
                        self.assertEqual(migrated['replacement_io_variants'],[item[1] for item in replacement])
                        seen+=1
                        continue
                    self.assertEqual(migrated['policy_action_id'],crosswalk[group['policy_action_id']])
                    module=current_params.module_for_name(group['name'])
                    self.assertEqual(migrated['io_variant'],rle.upgrade_variant(current_params.canonical_io_variant(variant['name'])))
                    self.assertEqual(migrated['parameters'],(module.parameters_from_name(group['name']) or module.default_parameters()) if module else {})
                    seen+=1
            self.assertEqual(seen,expected_count)
            self.assertEqual(p.migrate_policy_action(version,f'cpp-canonical-{version}-v1')['policy_action_id'],127)
        # No old ambiguous merged selector may silently acquire the new generic format.
        with self.assertRaises(ValueError):path_from_record({'ids':[14],'policy_registry_version':'cpp-canonical-140-v1'})
        self.assertEqual(path_from_names([f.ALGORITHM_NAME]).io_variants,(f.PLAIN_VARIANT,))

    def test_signed_unsigned_full_domains_and_exact_two_streams(self):
        cases=[[],[0],[1000,1001,1003],[-9,-1,-5],[-2**63,0,2**63-1],[2**64-1,0,2**63]]
        for values in cases:
            sources=[]
            if all(-2**63<=v<2**63 for v in values):sources.append(u.ActionData.ints(values))
            if all(0<=v<2**64 for v in values):sources.append(u.ActionData.bytes(struct.pack('<'+'Q'*len(values),*values)))
            for source in sources:
                with self.subTest(values=values,kind=source.kind):
                    wire=u.run_policy_action(f.ALGORITHM_NAME,source,registry_version=VERSION)
                    h,payload=u._unpack_result(wire);ports=u.run_policy_action(f.ALGORITHM_NAME,source,registry_version=VERSION,output='streams')
                    minimum=min(values,default=0);offsets=[v-minimum for v in values]
                    basekind='int64' if source.kind=='int64' else 'uint64'
                    self.assertEqual([(s.name,s.kind,s.count) for s in ports.streams],[('minimum',basekind,1),('offsets','uint64',len(values))])
                    self.assertEqual(struct.unpack('<q' if basekind=='int64' else '<Q',ports.streams[0].payload)[0],minimum)
                    self.assertEqual(list(x[0] for x in struct.iter_unpack('<Q',ports.streams[1].payload)),offsets)
                    expected=b'MFR1'+bytes([0 if basekind=='int64' else 1])+struct.pack('<Q',len(values))+struct.pack('<q' if basekind=='int64' else '<Q',minimum)+b''.join(struct.pack('<Q',v) for v in offsets)
                    self.assertEqual(payload,expected)
                    self.assertEqual(from_ports(ports),wire)
                    self.assertEqual(u.run_policy_action(f.ALGORITHM_NAME,ports,registry_version=VERSION,reverse=True),source)
                    self.assertEqual(p.encode_registered_exact_action_pipeline(source.payload,[f.ALGORITHM_NAME],logical_type='int64' if source.kind=='int64' else 'uint64'),expected)
                    self.assertNotIn('payload',ports.metadata)
                    self.assertLess(sum(len(s.get('literal',''))//2 for s in ports.metadata['segments']),21)

    def test_stream_edits_are_used_and_malformed_metadata_rejected(self):
        src=u.ActionData.ints([1000,1001,1003]);wire=u.run_policy_action(f.ALGORITHM_NAME,src,registry_version=VERSION)
        ports=u.run_policy_action(f.ALGORITHM_NAME,src,registry_version=VERSION,output='streams')
        changed=ActionPorts((TypedStream('minimum','int64',struct.pack('<q',1001),1),ports.streams[1]),ports.metadata,ports.header)
        self.assertEqual(u.run_policy_action(f.ALGORITHM_NAME,changed,registry_version=VERSION,reverse=True),u.ActionData.ints([1001,1002,1004]))
        for bad in [u.ActionData.floats([1.]),u.ActionData.bytes(b'123')]:
            with self.assertRaises(u.ActionRuntimeError):u.run_policy_action(f.ALGORITHM_NAME,bad,registry_version=VERSION)
        header,payload=u._unpack_result(wire)
        corruptions=[payload[:-1],payload+b'\0',payload[:4]+b'\x02'+payload[5:],payload[:5]+struct.pack('<Q',2**64-1)+payload[13:],payload[:21]+struct.pack('<3Q',1,2,3),payload[:13]+struct.pack('<q',2**63-1)+payload[21:]]
        for damaged in corruptions:
            with self.subTest(damaged=damaged.hex()),self.assertRaises(u.ActionRuntimeError):
                u.run_policy_action(f.ALGORITHM_NAME,u._pack_existing_result(header,damaged),registry_version=VERSION,reverse=True)
        for key,value in [('source_count',100),('source_kind','float64')]:
            h=copy.deepcopy(header);h[key]=value
            with self.assertRaises(u.ActionRuntimeError):u.run_policy_action(f.ALGORITHM_NAME,u._pack_existing_result(h,payload),registry_version=VERSION,reverse=True)
        with self.assertRaises(PortSchemaError):from_ports(ActionPorts(ports.streams[:1],ports.metadata,ports.header))

    def test_raw_magic_collisions_never_select_source_adapters(self):
        for magic in [b'MFR1',b'UAR1',b'S2D1',b'BDL1',b'BOS1',b'ASC1',b'RAR1']:
            src=u.ActionData.bytes(magic+b'\0'*4)
            encoded=u.run_policy_action(f.ALGORITHM_NAME,src,registry_version=VERSION)
            self.assertEqual(u._unpack_result(encoded)[0]['io_variant'],f.PLAIN_VARIANT)
            self.assertEqual(u.run_policy_action(f.ALGORITHM_NAME,encoded,registry_version=VERSION,reverse=True),src)

    def test_all_six_source_profiles_and_historical_replay(self):
        for variant in f.SOURCE_VARIANTS:
            before=packet(variant)
            explicit=u.run_policy_action(f.ALGORITHM_NAME,before,registry_version=VERSION,io_variant=variant)
            self.assertEqual(u.run_policy_action(variant,before,registry_version=VERSION),explicit)
            if variant!=f.SOURCE_VARIANTS[0]:
                self.assertEqual(u.run_policy_action(f.ALGORITHM_NAME,before,registry_version=VERSION),explicit)
            h,payload=u._unpack_result(explicit)
            self.assertEqual(u.run_policy_action(f.ALGORITHM_NAME,explicit,registry_version=VERSION,reverse=True),before)
            for version in (140,143,156,182):
                old=json.loads((ROOT/f'benchmark/reference/actions/canonical{version}.json').read_text())
                g=next(g for g in old['actions'] if variant in [v['name'] for v in g['io_variants']])
                oldh=copy.deepcopy(h);oldh.update(policy_registry_version=f'cpp-canonical-{version}-v1',policy_action_id=g['policy_action_id'],policy_algorithm=g['name'])
                legacy=u._pack_existing_result(oldh,payload)
                self.assertEqual(u.run_policy_action(f.ALGORITHM_NAME,legacy,registry_version=VERSION,reverse=True),before)
        for variant in f.SOURCE_VARIANTS[:3]:
            source=packet(variant);legacy=u.run_action(f.PREVIOUS_WHOLE_NAME,source)
            self.assertEqual(u._unpack_result(legacy)[0]['io_variant'],variant)
            self.assertEqual(u.run_action(f.PREVIOUS_WHOLE_NAME,legacy,reverse=True),source)

    def test_fixed_bos_double_for_and_private_recipe_profiles(self):
        src=u.ActionData.ints([100,101,102,104,108,120,1000,1001])
        for names in (p.BOS_BOSA_ACTIONS,p.TS2DIFF_INT_ACTIONS,p.REGER_ACTIONS):
            path=path_from_names(names)
            self.assertEqual(path_from_record(path_record(path)),path)
            canonical=tuple(p.POLICY_ACTION_BY_ID[i].name for i in path)
            old=p.encode_registered_exact_action_pipeline(src.payload,names,logical_type='int')
            now=p.encode_registered_exact_action_pipeline(src.payload,canonical,io_variants=path.io_variants,logical_type='int')
            self.assertEqual(now,old)
        path=path_from_names(p.BOS_BOSA_ACTIONS)
        self.assertEqual(path[1],path[3]);self.assertNotEqual(path.io_variants[1],path.io_variants[3])
        for path in SEQUENCES:
            for i,v in enumerate(path.io_variants):
                if v in f.PRODUCER_VARIANTS and i:
                    self.assertEqual(path.io_variants[i-1],f.PRODUCER_VARIANTS[v][0])

    def test_search_recipe_and_unsigned_qwen_root(self):
        src=u.ActionData.ints([-9,-8,-7,-6]*16)
        col={'raw':src.payload,'kind':'int64','max_point':0,'name':'for','dataset':'minimum'}
        env=Env(col,0,1);path=path_from_names([f.ALGORITHM_NAME]);self.assertIn(path[0],env.legal(()))
        out=env.transition((),path[0],io_variant=f.PLAIN_VARIANT)
        self.assertIsNotNone(out,env.records[-1]);self.assertEqual(decode_frame(encode_cold(col,out)),src.payload)
        self.assertTrue(any(o.ids.io_variants==(f.PLAIN_VARIANT,) for o in build_catalog(p.registry())))
        binding=QwenActionBinding();record=path_record(path);raw=struct.pack('<3Q',2**64-1,0,2**63)
        self.assertEqual(binding.root_data(raw,record).kind,'bytes')
        self.assertEqual(binding.decode_recipe(binding.encode_recipe(raw,record)),raw)

    def test_graph_namespaces_and_stop_migrations(self):
        self.assertEqual((graph1.ACTION_COUNT,graph1.STOP_ACTION_ID),(154,151))
        self.assertEqual((graph2.ACTION_COUNT,graph2.STOP_ACTION_ID),(148,145))
        self.assertEqual((len(graph2.FAMILIES),len(graph2.SPECIAL_ACTIONS)),(30,3))
        self.assertEqual(graph2.migrate_action(143,graph1.REGISTRY_VERSION)['name'],'op.split_columns.float_deconstruct.cpp_v1')
        for version in (graph1.REGISTRY_VERSION, 'cpp-canonical-140-v1'):
            with self.assertRaises(ValueError):graph2.migrate_action(14,version)
        for version,stop in [('cpp-canonical-137-v1',137),('cpp-canonical-140-v1',140),('cpp-canonical-143-v1',143),('cpp-canonical-156-v1',156),('cpp-canonical-182-v1',182),('cpp-canonical-206-v1',206),('legacy-225-v1',225),(graph1.REGISTRY_VERSION,151)]:
            self.assertEqual(graph2.migrate_action(stop,version)['name'],'stop')
        for old in graph1.CODEC_ACTIONS:
            if old.legacy_policy_action_id is None or old.operator_id=='split_columns':continue
            for variant in old.io_variants:
                new=graph2.migrate_action(old.action_id,graph1.REGISTRY_VERSION,io_variant=variant)
                self.assertEqual(new['operator_id'],old.operator_id)
                self.assertEqual(new['io_variant'],params.module_for_name(variant).ALGORITHM_NAME if params.module_for_name(variant) else variant)

if __name__=='__main__':unittest.main()
