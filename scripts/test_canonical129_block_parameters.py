"""Shared block codec parameters: exact old wires, new choices, and versioned replay."""
from __future__ import annotations
import copy,json,struct,subprocess,sys,unittest
from pathlib import Path
import primitive_operator_action_runtime as p
import unified_action_runtime as u
import block_state_parameters as b
import current_action_parameters as params
import typed_graph_runtime as graph
import operator_taxonomy_v30 as graph1
import operator_taxonomy_v30_v2 as graph2
import operator_taxonomy_v30_v3 as graph3
import canonical129_registry as frozen129
ROOT=Path(__file__).resolve().parents[1];VERSION=p.ACTION_REGISTRY_VERSION
RAW=struct.pack('<256I',*range(256))

def prefix(mode):
    if mode=='corad_ctk2':
        raw=struct.pack('<10q',*range(10))
        return p.corad_lossless_stages.tricklet_forward(p.corad_lossless_stages.normalize_forward(raw,'int64'))
    commands=('deflate-lz-forward','huffman-forward') if mode=='gzip_prefix' else ('deflate-lz-forward','parse-forward','huffman-forward')
    return p._run_general_reference_chain(RAW,commands)

def blosc_prefix(t):return p._run_general_reference_chain(RAW,(f'shuffle{t}-forward','blosclz-forward','token-forward'))

class CurrentBlockTests(unittest.TestCase):
    def test_inventory_and_namespace_frozen(self):
        self.assertEqual((VERSION,len(p.POLICY_ACTIONS),len(u.IO_VARIANT_ACTIONS),p.STOP_ACTION_ID),('cpp-canonical-127-v3',122,170,127))
        self.assertEqual(len(graph2.ACTIONS),148);self.assertEqual(graph2.STOP_ACTION_ID,145)
        self.assertEqual((len(graph3.CODEC_ACTIONS),len(graph3.MODEL_PROVIDERS),len(graph3.ACTIONS)),(132,5,140))
        self.assertEqual(graph3.registry()['operator_action_counts']['block_state_control'],13)
        self.assertEqual(graph3.registry()['operator_action_counts']['container_frame'],9)
        for i,old in enumerate(p.algorithm_io.previous137.GROUPS):
            for old_variant in old['io_variants']:
                inv=p.migrate_policy_action(i,'cpp-canonical-137-v1',io_variant=old_variant['name'])
                if old['name'] in p.RETIRED_IDENTITY_ACTION_NAMES:
                    self.assertEqual(inv['migration_kind'],'retired_identity_bypass')
                    self.assertEqual(inv['replacement_policy_action_ids'],[])
                    historical=graph3.migrate_action(i,graph2.REGISTRY_VERSION,io_variant=old_variant['name'])
                    self.assertEqual((historical['action_id'],historical['parameters']),(5,{}))
                    continue
                canonical=p.ACTION_CANONICAL_NAMES.get(old['name'],old['name'])
                if canonical in p.COMPOSITE_POLICY_REPLACEMENTS:
                    replacement=p.COMPOSITE_POLICY_REPLACEMENTS[canonical]
                    self.assertEqual(inv['migration_kind'],'replacement_recipe')
                    self.assertEqual(inv['replacement_policy_action_ids'],[item[0] for item in replacement])
                    self.assertEqual(inv['replacement_io_variants'],[item[1] for item in replacement])
                    continue
                name=p.POLICY_ACTION_BY_ID[inv['policy_action_id']].name
                self.assertEqual(name,p.ACTION_CANONICAL_NAMES[old['name']])
                self.assertIn(inv['io_variant'],p.POLICY_IO_VARIANTS[name])
                migrated=graph3.migrate_action(i,graph2.REGISTRY_VERSION,io_variant=old_variant['name'])
                old_name=frozen129.canonical(old['name'])
                frozen_group=frozen129.GROUP_BY_NAME[old_name]
                self.assertEqual((migrated['action_id'],migrated['parameters']),
                                 (frozen_group['policy_action_id'],inv['parameters']))
        # V2 migration remains in V2's original namespace after current IDs shrink.
        self.assertEqual(graph2.migrate_action(118,'cpp-canonical-140-v1')['action_id'],115)
        self.assertEqual(graph2.migrate_action(140,'cpp-canonical-140-v1')['action_id'],145)
        self.assertEqual(graph3.migrate_action(145,graph2.REGISTRY_VERSION)['action_id'],137)
        self.assertEqual(graph3.migrate_action(143,graph1.REGISTRY_VERSION)['action_id'],129)
        self.assertEqual(graph3.migrate_action(143,'cpp-canonical-143-v1')['action_id'],137)

    def test_all_blosc_widths_preserve_old_wire_and_inverse(self):
        for name,codec,base_id in [(b.BLOSC1,'blosc1',115),(b.BLOSC2,'blosc2',118)]:
            for offset,t in enumerate((1,4,8)):
                with self.subTest(codec=codec,typesize=t):
                    source=u.ActionData.bytes(blosc_prefix(t));settings={'typesize':t}
                    wire=u.run_policy_action(name,source,registry_version=VERSION,parameters=settings)
                    h,payload=u._unpack_result(wire)
                    self.assertEqual(payload,p._run_general_codec_residual_stage_cli(f'{codec}-forward-{t}',source.payload))
                    self.assertEqual(u.run_policy_action(name,wire,registry_version=VERSION,reverse=True),source)
                    wrong=next(value for value in (1,4,8) if value!=t)
                    with self.assertRaises(u.ActionRuntimeError):u.run_policy_action(name,source,registry_version=VERSION,parameters={'typesize':wrong})
                    bad_width=copy.deepcopy(h);bad_width['action_parameters']={'typesize':wrong}
                    with self.assertRaises(u.ActionRuntimeError):u.run_policy_action(name,u._pack_existing_result(bad_width,payload),registry_version=VERSION,reverse=True)
                    # Old ID invocation keeps the original width and wire.
                    old=u.run_policy_action(base_id+offset,source,registry_version='cpp-canonical-137-v1')
                    self.assertEqual(u._unpack_result(old)[1],payload)
                    old_name=p.algorithm_io.previous137.GROUPS[base_id+offset]['name']
                    old_header=copy.deepcopy(h);old_header.update(actions=[old_name],policy_algorithm=old_name,policy_registry_version='cpp-canonical-137-v1',policy_action_id=base_id+offset,io_variant=old_name);old_header.pop('action_parameters')
                    self.assertEqual(u.run_policy_action(name,u._pack_existing_result(old_header,payload),registry_version=VERSION,reverse=True),source)
                    bad=copy.deepcopy(h);bad['action_parameters']={'typesize':3}
                    with self.assertRaises((ValueError,u.ActionRuntimeError)):u.run_policy_action(name,u._pack_existing_result(bad,payload),registry_version=VERSION,reverse=True)

    def test_deflate_every_io_every_level_and_default_compatibility(self):
        for variant,mode,old_id,old_cmd in zip(b.DEFLATE_VARIANTS,('gzip_prefix','zlib_prefix','corad_ctk2'),(122,123,127),('gzip-forward','zlib-forward','entropy-forward')):
            source=u.ActionData.bytes(prefix(mode))
            for level in range(10):
                with self.subTest(mode=mode,level=level):
                    wire=u.run_policy_action(b.DEFLATE,source,registry_version=VERSION,io_variant=variant,parameters={'level':level})
                    self.assertEqual(u.run_policy_action(b.DEFLATE,wire,registry_version=VERSION,reverse=True),source)
            legacy=u.run_policy_action(old_id,source,registry_version='cpp-canonical-137-v1')
            h,payload=u._unpack_result(legacy)
            expected=(p.corad_lossless_stages.entropy_forward(source.payload) if mode=='corad_ctk2' else p._run_general_codec_residual_stage_cli(old_cmd,source.payload))
            self.assertEqual(payload,expected);self.assertEqual(h['action_parameters'],{'level':6 if mode=='gzip_prefix' else 9})
            # Historical UAR1 had no parameter record and kept its original name.
            old_name=p.algorithm_io.previous137.GROUPS[old_id]['name']
            h.update(actions=[old_name],policy_algorithm=old_name,policy_registry_version='cpp-canonical-137-v1',policy_action_id=old_id,io_variant=old_name);h.pop('action_parameters')
            self.assertEqual(u.run_policy_action(b.DEFLATE,u._pack_existing_result(h,payload),registry_version=VERSION,reverse=True),source)
            for bad in ({'level':True},{'level':-1},{'level':10},{'level':'9'},{'level':1,'extra':2}):
                with self.assertRaises((ValueError,u.ActionRuntimeError)):u.run_policy_action(b.DEFLATE,source,registry_version=VERSION,io_variant=variant,parameters=bad)
        # The primitive pipeline entry also honors the explicit parameter at its real codec step.
        names=(*p.GZIP_EXACT_ACTIONS[:2], b.DEFLATE, p.GZIP_EXACT_ACTIONS[-1])
        wire=p.encode_registered_exact_action_pipeline(RAW,names,
            io_variants=(None,None,b.DEFLATE_VARIANTS[0],None),
            parameters=(None,None,{'level':7},None))
        expected=b.run(b.DEFLATE,prefix('gzip_prefix'),{'level':7},io_variant=b.DEFLATE_VARIANTS[0])
        expected=p._run_deflate_container_stage_cli('gzip-forward',expected)
        self.assertEqual(wire,expected)
        with self.assertRaises((ValueError,u.ActionRuntimeError)):u.run_policy_action(b.DEFLATE,source,registry_version=VERSION,parameters={'level':9})
        with self.assertRaises((ValueError,u.ActionRuntimeError)):u.run_policy_action(b.DEFLATE_OLD[0],source,registry_version=VERSION,parameters={'level':9})

    def test_reference_merge_preserves_tags_and_magic_merge_preserves_wires(self):
        source=u.ActionData.ints([1,2,3,-5])
        name='op.match_window_model.reference_v1'
        for old_id in (80,107):
            old=p.algorithm_io.previous137.GROUPS[old_id]['name']
            wire=u.run_policy_action(old_id,source,registry_version='cpp-canonical-137-v1')
            self.assertEqual(u.run_policy_action(name,wire,registry_version=VERSION,reverse=True),source)
            self.assertEqual(u._unpack_result(wire)[1],u._unpack_result(u.run_actions((old,),source,_concrete=True))[1])
        name='op.container_frame.magic_length.cpp_v3'
        self.assertEqual(len(p.POLICY_IO_VARIANTS[name]),6)
        source=u.ActionData.bytes(b'example payload')
        for variant in p.POLICY_IO_VARIANTS[name]:
            source=u.ActionData.bytes(p.corad_lossless_stages.entropy_forward(prefix('corad_ctk2'))) if 'corad' in variant else u.ActionData.bytes(b'example payload')
            wire=u.run_policy_action(name,source,registry_version=VERSION,io_variant=variant)
            self.assertEqual(u.run_policy_action(name,wire,registry_version=VERSION,reverse=True),source)
            self.assertEqual(len(u._unpack_result(wire)[1]),len(source.payload)+12)

    def test_namespaced_options_never_overwrite_widths(self):
        names=(b.BLOSC1,b.BLOSC2,b.DEFLATE)
        choice={'blosc1_typesize':1,'blosc2_typesize':8,'deflate_level':7}
        self.assertEqual(params.normalize_option(names,choice),choice)
        self.assertEqual(params.for_stage(b.BLOSC2,choice),{'typesize':8})
        self.assertEqual(params.for_option(b.BLOSC2,{'typesize':8}),{'blosc2_typesize':8})
        self.assertEqual(params.for_option(b.DEFLATE,{'level':7}),{'deflate_level':7})
        with self.assertRaises(ValueError):params.for_option(b.DEFLATE,{'level':7,'extra':1})
        self.assertEqual(len(params.choices_for_names(names)),90)
        self.assertEqual(len({json.dumps(p,sort_keys=True) for p in params.choices_for_names(names)}),90)
        with self.assertRaises(ValueError):params.normalize_option(names,{'typesize':8,'level':7})

    def test_graph_versions_real_codecs_and_cold_restore(self):
        for version,selector in [(graph2.REGISTRY_VERSION,117),(graph3.REGISTRY_VERSION,b.BLOSC1)]:
            source=graph.Stream('serial',1,blosc_prefix(8));state=graph.GraphState.from_streams([source],registry_version=version)
            kwargs={'parameters':{'typesize':8}} if version==graph3.REGISTRY_VERSION else {}
            state=u.run_graph_action(state,selector,['s0'],registry_version=version,**kwargs)
            self.assertEqual(graph.restore_frame(state.stop()),{'s0':source})
            if version==graph2.REGISTRY_VERSION:
                direct=graph.GraphState.from_streams([source],registry_version=version).apply(selector,['s0'])
                self.assertEqual(graph.restore_frame(direct.stop()),{'s0':source})
            script='import sys; from typed_graph_runtime import restore_frame; sys.stdout.buffer.write(restore_frame(sys.stdin.buffer.read())["s0"].data)'
            result=subprocess.run([sys.executable,'-c',script],input=state.stop(),capture_output=True,check=True,cwd=ROOT/'scripts')
            self.assertEqual(result.stdout,source.data)
        source=graph.Stream('serial',1,prefix('gzip_prefix'));state=graph.GraphState.from_streams([source])
        state=u.run_graph_action(state,b.DEFLATE,['s0'],registry_version=graph3.REGISTRY_VERSION,io_variant=b.DEFLATE_VARIANTS[0],parameters={'level':7})
        self.assertEqual(graph.restore_frame(state.stop()),{'s0':source})
        for version in (graph2.REGISTRY_VERSION,graph3.REGISTRY_VERSION):
            source=graph.Stream('numeric',8,struct.pack('<3Q',100,103,108));state=graph.GraphState.from_streams([source],registry_version=version)
            state=state.apply('op.frame_of_reference.minimum.cpp_v1',['s0'])
            self.assertEqual(graph.restore_frame(state.stop()),{'s0':source})

if __name__=='__main__':unittest.main()
