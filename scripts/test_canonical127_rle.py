"""Current RLE replacement and independent historical wire replay boundaries."""
from __future__ import annotations
import copy
import hashlib
import json
import struct
import subprocess
import sys
import unittest
from pathlib import Path
import canonical129_registry as old
import optimal_rle_actions as r
import primitive_operator_action_runtime as p
import unified_action_runtime as u
import typed_graph_runtime as graph
import operator_taxonomy_v30_v3 as v3
import operator_taxonomy_v30_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
VERSION = p.ACTION_REGISTRY_VERSION
OLD_VERSION = 'cpp-canonical-129-v1'
RAW = struct.pack('<4q', 2, 2, 3, 3)


def cold(code, payload):
    result = subprocess.run([sys.executable, '-c', "import sys;sys.path.insert(0, 'scripts');" + code],
                            cwd=ROOT, input=payload, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr.decode(errors='replace'))
    return result.stdout


class RleReplacementTests(unittest.TestCase):
    def test_current_inventory_and_frozen_crosswalks(self):
        self.assertEqual((p.ACTION_REGISTRY_VERSION, len(p.POLICY_ACTIONS), p.STOP_ACTION_ID), (VERSION, 122, 127))
        self.assertEqual(len(u.IO_VARIANT_ACTIONS), 170)
        self.assertEqual([s.name for s in p.POLICY_ACTIONS if s.operator_id == 'run_length'], [r.ALGORITHM_NAME])
        self.assertEqual((len(v4.CODEC_ACTIONS), len(v4.MODEL_PROVIDERS), len(v4.ACTIONS)), (130, 5, 138))
        self.assertEqual((v4.STOP_ACTION_ID, v4.CONCAT_ACTION_ID, v4.SPLIT_SEGMENTS_ACTION_ID), (135, 136, 137))
        self.assertEqual((len(old.GROUPS), len(v3.ACTIONS), v3.STOP_ACTION_ID), (129, 140, 137))
        for action in (65, 66, 67):
            frozen = v3.migrate_action(action, OLD_VERSION)
            self.assertEqual(frozen['action_id'], action)
            upgraded = v4.migrate_action(action, v3.REGISTRY_VERSION)
            self.assertEqual(upgraded['action_id'], 65)
            self.assertEqual(upgraded['migration_kind'], 'implementation_replacement')
            self.assertTrue(upgraded['requires_remeasurement'])
        self.assertEqual(v3.migrate_action(129, OLD_VERSION)['action_id'], 137)
        self.assertEqual(v4.migrate_action(137, v3.REGISTRY_VERSION)['action_id'], 135)
        for index, group in enumerate(old.GROUPS):
            for variant in group['io_variants']:
                q = {'level': 6} if group['name'] == 'op.block_state_control.deflate_residual_core.cpp_v2' else None
                frozen = v3.migrate_action(index, OLD_VERSION, io_variant=variant['name'], parameters=q)
                self.assertEqual((frozen['action_id'], frozen['io_variant']), (index, variant['name']))
        self.assertEqual(p.registry()['counts']['callable_action_implementation_records'], 122)

    def test_old_four_forms_exact_wire_and_cold_inverse(self):
        sources = [u.ActionData.ints([2,2,3,3]), u.ActionData.bytes(bytes([2,2,3,3])),
                   u.ActionData.ints([2,2,3,3]), u.run_action('op.delta.rlbe.cpp_v1',u.ActionData.ints([2,5,7,10]))]
        variants = list(r.OLD_TO_CURRENT)
        ids = (65,65,66,67)
        goldens = ('789cf30f70346464e42b2acdcb49cd4b2fc960808150431620c9c4c4c40c008783058c',
                   '524c4231040000000000000002020000000302000000', '05020304af00', '00000004057760')
        for source, variant, action, golden in zip(sources, variants, ids, goldens):
            with self.subTest(variant=variant):
                wire = u.run_policy_action(action,source,registry_version=OLD_VERSION,io_variant=variant)
                self.assertEqual(u._unpack_result(wire)[1].hex(), golden)
                by_name=u.run_policy_action(variant,source,registry_version=OLD_VERSION)
                self.assertEqual(u._unpack_result(by_name)[1].hex(),golden)
                self.assertEqual(u.run_action(variant,source).payload, u.run_actions((variant,),source,_concrete=True).payload)
                code = f"import unified_action_runtime as u;w=u.ActionData.bytes(sys.stdin.buffer.read());sys.stdout.buffer.write(u.run_policy_action({action},w,registry_version='{OLD_VERSION}',reverse=True).payload)"
                self.assertEqual(cold(code,wire.payload),source.payload)
                header, payload = u._unpack_result(wire)
                bad = copy.deepcopy(header);bad['policy_action_id'] = 64
                with self.assertRaises(u.ActionRuntimeError):
                    u.run_policy_action(action,u._pack_existing_result(bad,payload),registry_version=OLD_VERSION,reverse=True)

    def test_current_four_profiles_execute_new_native_core(self):
        sources = [u.ActionData.ints([2,2,3,3]),u.ActionData.bytes(bytes([2,2,3,3])),
                   u.run_action('op.scaling.tsfile_rle.cpp_v1',u.ActionData.floats([1.25,1.25,2.5]),max_point=2),
                   u.run_action('op.delta.rlbe.cpp_v1',u.ActionData.ints([2,5,7,10]))]
        for variant, source, magic in zip(r.VARIANTS,sources,(b'URL1',b'URL1',b'URS1',b'URB1')):
            with self.subTest(variant=variant):
                wire = u.run_policy_action(r.ALGORITHM_NAME,source,registry_version=VERSION,io_variant=variant)
                header, payload = u._unpack_result(wire)
                self.assertTrue(payload.startswith(magic))
                self.assertEqual(header['policy_registry_version'], VERSION)
                self.assertEqual(header['actions'], [variant])
                self.assertEqual(u.run_policy_action(65,wire,registry_version=VERSION,reverse=True),source)
                self.assertEqual(cold(f"import unified_action_runtime as u;w=u.ActionData.bytes(sys.stdin.buffer.read());sys.stdout.buffer.write(u.run_policy_action(65,w,registry_version='{VERSION}',reverse=True).payload)",wire.payload),source.payload)

    def test_float_raw_bits_and_explicit_byte_profiles(self):
        raw = struct.pack('<7Q',0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000042,1,0xffffffffffffffff)
        source=u.ActionData('float64',raw,7)
        wire=u.run_action(r.ALGORITHM_NAME,source)
        self.assertEqual(u.run_action(r.ALGORITHM_NAME,wire,reverse=True),source)
        empty=u.ActionData.ints([])
        self.assertEqual(u.run_action(r.ALGORITHM_NAME,u.run_action(r.ALGORITHM_NAME,empty),reverse=True),empty)
        for bad in (u.ActionData.bytes(b'x'),u.ActionData.bytes(b'RLS1' + bytes(8))):
            with self.assertRaises(u.ActionRuntimeError):
                u.run_action(r.ALGORITHM_NAME,bad,io_variant=r.WORDS_VARIANT)
        wire=u.run_action(r.ALGORITHM_NAME,u.ActionData.bytes(RAW),io_variant=r.BYTES_VARIANT)
        header,payload=u._unpack_result(wire)
        header.update(actions=[r.WORDS_VARIANT],io_variant=r.WORDS_VARIANT)
        with self.assertRaises(u.ActionRuntimeError):
            u.run_policy_action(65,u._pack_existing_result(header,payload),registry_version=VERSION,reverse=True)

    def test_old_graph_v3_and_current_graph_v4_cold(self):
        for version, action in ((v3.REGISTRY_VERSION,65),(v3.REGISTRY_VERSION,66),(v4.REGISTRY_VERSION,65)):
            state=graph.GraphState.from_streams([graph.Stream('numeric',8,RAW)],registry_version=version)
            state=state.apply(action,['s0'],input_view='int64')
            frame=state.stop()
            self.assertEqual(cold("import typed_graph_runtime as g;sys.stdout.buffer.write(g.restore_frame(sys.stdin.buffer.read())['s0'].data)",frame),RAW)
            # Archived graph frames may store the source numeric ID rather than its name.
            size=struct.unpack_from('<Q',frame,4)[0]
            header=json.loads(frame[12:12+size]);header['steps'][0]['action']=action
            data=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
            body=b'TGF1'+struct.pack('<Q',len(data))+data+frame[12+size:-32]
            numeric=body+hashlib.sha256(body).digest()
            self.assertEqual(graph.restore_frame(numeric)['s0'].data,RAW)

    def test_bzip2_current_prefix_and_frozen_prefix(self):
        raw=b'AAAABBBBCCCCDDDDAAAABBBB'*3
        mappings=p.benchmark_86_io_variant_pipelines()['Bzip2']
        source=u.ActionData.bytes(raw);current=source
        for name in mappings:
            current=u.run_action(name,current)
        header,payload=u._unpack_result(current)
        self.assertEqual(header['rle_prefix_version'],'URL1')
        self.assertEqual(payload,p._run_general_codec_residual_stage_cli('original-bzip2-encode',raw))
        for name in reversed(mappings):current=u.run_action(name,current,reverse=True)
        self.assertEqual(current,source)
        legacy_names=p.benchmark_86_legacy_action_pipelines()['Bzip2']
        current=source
        for name in legacy_names: current=u.run_action(name,current)
        self.assertNotIn('rle_prefix_version',u._unpack_result(current)[0])
        for name in reversed(legacy_names):current=u.run_action(name,current,reverse=True)
        self.assertEqual(current,source)

    def test_current_method_dispatch_matches_actual_new_stage_chains(self):
        for method,kind,values in [('RLE','int',[2,2,3,3]),('RLE','float',[1.25,1.25,2.5]),
                                    ('RLBE','int',[2,5,7,10]),('RLBE','float',[1.0,1.0,2.0]),
                                    ('Sprintz','int',[7]*13),('Sprintz','float',[1.0]*13)]:
            with self.subTest(method=method,kind=kind):
                source=u.ActionData.ints(values) if kind=='int' else u.ActionData.floats(values)
                mappings=p.benchmark_86_io_variant_pipelines()[method]
                names=mappings['int' if kind=='int' else 'float_max_point_0_to_18' if method=='RLE' else 'float']
                current=source
                for name in names: current=u.run_action(name,current,max_point=2)
                self.assertEqual(u._unpack_result(current)[1],p.encode_exact_benchmark_method(source.payload,method,logical_type=kind,max_point=2))
                for name in reversed(names): current=u.run_action(name,current,reverse=True)
                self.assertEqual(current,source)

    def test_sprintz_frozen_packet_and_current_tail(self):
        source=u.ActionData.ints([7]*13)
        pred=u.run_action(p.SPRINTZ_ACTIONS[0],source)
        zig=u.run_action(p.SPRINTZ_ACTIONS[1],pred)
        old_id=next(g['policy_action_id'] for g in old.GROUPS if g['name']==r.OLD_SPRINTZ_VARIANT)
        frozen=u.run_policy_action(old_id,zig,registry_version=OLD_VERSION)
        current=u.run_action(r.SPRINTZ_VARIANT,zig)
        self.assertNotEqual(u._unpack_result(frozen)[1],u._unpack_result(current)[1])
        self.assertEqual(u.run_policy_action(old_id,frozen,registry_version=OLD_VERSION,reverse=True),zig)
        self.assertEqual(u.run_action(r.SPRINTZ_VARIANT,current,reverse=True),zig)
        code=f"import unified_action_runtime as u;w=u.ActionData.bytes(sys.stdin.buffer.read());sys.stdout.buffer.write(u.run_policy_action({old_id},w,registry_version='{OLD_VERSION}',reverse=True).payload)"
        self.assertEqual(cold(code,frozen.payload),zig.payload)
        whole=u._pack_result(p.SPRINTZ_ACTIONS,source,p.encode_exact_p04_action_pipeline(source.payload,'Sprintz','int'),None)
        self.assertEqual(u.run_actions(p.SPRINTZ_ACTIONS,whole,reverse=True,_concrete=True),source)
        for version in (v3.REGISTRY_VERSION,v4.REGISTRY_VERSION):
            state=graph.GraphState.from_streams([graph.Stream('numeric',8,source.payload)],registry_version=version)
            for index,name in enumerate(p.SPRINTZ_ACTIONS):
                state=state.apply(name,[next(iter(state.streams))],input_view='int64' if index==0 else 'bytes')
            self.assertEqual(cold("import typed_graph_runtime as g;sys.stdout.buffer.write(g.restore_frame(sys.stdin.buffer.read())['s0'].data)",state.stop()),source.payload)


if __name__ == '__main__': unittest.main()
