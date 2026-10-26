"""A048 must be FOR-free, parameterized, and independently branchable."""
from pathlib import Path
import json,struct,subprocess,sys,unittest
import primitive_operator_action_runtime as p
import unified_action_runtime as u
import action_semantic_ports as ports
import typed_graph_runtime as graph
import operator_taxonomy_v30_v7 as taxonomy
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'analysis_outputs/buff_a048_decomposition_20260915'
class BuffPlaneTests(unittest.TestCase):
    def stages(self, values=(1.25,1.5,2.0,-3.25), width=None):
        source=u.ActionData.floats(values)
        scaled=u.run_policy_action(10,source,registry_version=p.ACTION_REGISTRY_VERSION)
        normalized=u.run_policy_action(14,scaled,io_variant=p.BUFF_FOR_ACTION.name,registry_version=p.ACTION_REGISTRY_VERSION)
        split=u.run_policy_action(58,normalized,parameters=None if width is None else {'bit_width':width},registry_version=p.ACTION_REGISTRY_VERSION)
        return source,scaled,normalized,split
    def test_actual_streams_and_no_second_for(self):
        _,_,normalized,split=self.stages()
        view=ports.to_ports(normalized)
        self.assertEqual([(x.name,x.kind,x.count) for x in view.streams],[('minimum','int64',1),('offsets','uint64',4)])
        changed=ports.ActionPorts((view.streams[0],ports.TypedStream('offsets','uint64',struct.pack('<4Q',1,2,4,8),4)),view.metadata,view.header)
        shifted=ports.from_ports(changed)
        output=u.run_policy_action(58,shifted,parameters={'bit_width':4},registry_version=p.ACTION_REGISTRY_VERSION)
        v=ports.to_ports(output)
        self.assertEqual([(x.name,x.kind,x.count) for x in v.streams],[('minimum','int64',1),('byte_plane0','uint8',4)])
        self.assertEqual(v.streams[1].payload,b'\x10\x20\x40\x80') # offset 1 stays 1; no implicit subtract-min
        self.assertEqual(u.run_policy_action(58,output,registry_version=p.ACTION_REGISTRY_VERSION,reverse=True),shifted)
        self.assertEqual(ports.from_ports(ports.to_ports(split)),split)
    def test_explicit_width_default_and_rejection(self):
        _,_,n,x=self.stages(width=16)
        self.assertEqual(u._unpack_result(x)[0]['action_parameters'],{'bit_width':16})
        self.assertEqual(u.run_policy_action(58,x,reverse=True,registry_version=p.ACTION_REGISTRY_VERSION),n)
        for setting in [-1,65,True,1.5,0,1]:
            with self.subTest(width=setting),self.assertRaises(ValueError):u.run_policy_action(58,n,parameters={'bit_width':setting},registry_version=p.ACTION_REGISTRY_VERSION)
        with self.assertRaises(ValueError):u.run_policy_action(58,x,reverse=True,parameters={'bit_width':15},registry_version=p.ACTION_REGISTRY_VERSION)
        header,body=u._unpack_result(x);header['action_parameters']['bit_width']=15
        with self.assertRaises(ValueError):u.run_policy_action(58,u._pack_existing_result(header,body),reverse=True,registry_version=p.ACTION_REGISTRY_VERSION)
    def test_zero_and_empty(self):
        for values in [[],[3.25]*5]:
            s,_,n,x=self.stages(values)
            self.assertEqual(ports.to_ports(x).metadata['plane_count'],0)
            self.assertEqual(len(ports.to_ports(x).streams),1) # base bypass; no new data planes at width zero
            self.assertEqual(u.run_actions(p.BUFF_ATOMIC_ACTIONS,u.run_actions(p.BUFF_ATOMIC_ACTIONS,s),reverse=True),s)
    def test_legacy_uar_and_migration(self):
        saved=(AUDIT/'fixtures/legacy_a048.uar').read_bytes()
        actual=u.run_policy_action(58,u.ActionData.bytes(saved),registry_version='cpp-canonical-127-v3',reverse=True)
        self.assertEqual(actual.payload,(AUDIT/'fixtures/legacy_input.uar').read_bytes())
        for version in ['cpp-canonical-127-v1','cpp-canonical-127-v2','cpp-canonical-127-v3']:
            self.assertEqual(p.migrate_policy_action(58,version)['replacement_policy_action_ids'],[14,58])
        self.assertEqual(taxonomy.migrate_action(58,'operator-taxonomy-30-v6')['replacement_action_ids'],[14,58])
        self.assertEqual(p.migrate_policy_action(58,p.ACTION_REGISTRY_VERSION,parameters={'bit_width':17})['parameters'],{'bit_width':17})
        with self.assertRaises(ValueError):u.run_actions((p.BUFF_ACTIONS[1],),actual)
    def test_independent_plane_graph_and_cold_decode(self):
        source=u.ActionData.floats([1.25,1.5,2.0,-3.25]*8)
        g=graph.GraphState.from_streams([graph.Stream('numeric',8,source.payload)])
        for action,variant in [(10,None),(14,p.BUFF_FOR_ACTION.name),(58,None)]:
            g=g.apply(action,list(g.streams),input_view='float64' if action==10 else 'bytes',io_variant=variant)
        self.assertEqual(len(g.streams),3)
        ids=list(g.streams)
        g=g.apply(65,[ids[1]],io_variant='op.run_length.byte_optimal.uint8.cpp_v1')
        frame=g.stop();self.assertEqual(graph.restore_frame(frame)['s0'].data,source.payload)
        code="import sys;sys.path.insert(0,'scripts');import typed_graph_runtime as g;sys.stdout.buffer.write(g.restore_frame(sys.stdin.buffer.read())['s0'].data)"
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,input=frame,capture_output=True)
        self.assertEqual(done.returncode,0,done.stderr.decode());self.assertEqual(done.stdout,source.payload)
        (AUDIT/'fixtures/current_a048_planes.tgf').write_bytes(frame)
        with self.assertRaises(ValueError):graph.restore_frame(frame[:-1])
    def test_v6_graph_with_frozen_v3_payload(self):
        original=(AUDIT/'fixtures/legacy_input.uar').read_bytes()
        old_result=(AUDIT/'fixtures/legacy_a048.uar').read_bytes()
        source=graph.Stream('serial',1,original);result=graph.Stream('serial',1,old_result)
        step={'action':p.BUFF_ACTIONS[1],'inputs':['s0'],'input_specs':[source.spec()],
              'kind':'legacy','input_view':'bytes','outputs':['s1'],'output_specs':[result.spec()]}
        old=graph.GraphState({'s1':result},{'s0':source.spec()},(step,),2,'operator-taxonomy-30-v6')
        self.assertEqual(graph.restore_frame(old.stop())['s0'].data,original)
        (AUDIT/'fixtures/legacy_v6_a048.tgf').write_bytes(old.stop())
    def test_native_86_buff_mapping_stays_truthful(self):
        spec=p.benchmark_86_exact_action_pipelines()['BUFF']
        self.assertEqual(spec,{'int':list(p.BUFF_NATIVE_ACTIONS),'float':list(p.BUFF_NATIVE_ACTIONS)})
        self.assertNotIn(p.BUFF_ACTIONS[1],p.BUFF_ATOMIC_ACTIONS)
if __name__=='__main__': unittest.main()
