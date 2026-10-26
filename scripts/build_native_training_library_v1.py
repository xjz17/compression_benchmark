#!/usr/bin/env python3
"""Predeclared seed recipes measured solely on the frozen training blocks.

The seed library is a starting repertoire, not a restriction of the online
6341-action space. No historical best-of-86 profile or test measurement enters.
"""
import argparse
import collections
import json
import os
from pathlib import Path

import native_prepared_cohort_v1 as data
import native_training_corpus_v1 as training
import calibrate_native_smallest32_v1 as calibration
from native_grpo_task_v1 import NativeClient,NativeGraphTask,REGISTRY

# Chosen before this run's measurements. Typed and bytes profiles are separate.
SEED_STAGES=(
    'op.frame_of_reference.minimum.int64_uint64_le.cpp_v1',
    'op.bitpack_int.max_value_width.cpp_v1',
    'op.xor.previous_word.cpp_v2',
    'op.dictionary.project_words64.cpp_v1',
    'op.run_length.byte_optimal.cpp_v1',
    'op.entropy_huffman.project_bytes.cpp_v1',
    'op.entropy_range.project_bytes.cpp_v1',
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','materialized','reference','cli','catalog','cold-cli','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    data.require(not a.output.exists(),'library already frozen')
    data.require(hasattr(os,'sched_getaffinity') and len(os.sched_getaffinity(0))==1,'library measurement requires one pinned CPU')
    corpus=data.load(a.cohort,a.materialized,role='train_validation')
    reference=training.load_reference(a.reference,corpus)
    actions=json.loads(a.catalog.read_bytes())['registry_payload']['actions']
    proposals=[r for r in actions if r['target_slot']==0 and r['runtime_action'] in SEED_STAGES]
    data.require(len(proposals)<64 and {r['runtime_action'] for r in proposals}==set(SEED_STAGES),'seed proposal catalog mismatch')
    measurements=[];failures=[]
    cli_sha=data.digest(a.cli.read_bytes())
    with NativeClient(a.cli,a.catalog,timeout=300) as client:
        catalog=client.call({'cmd':'catalog'})
        data.require(catalog['complete_native_graph_space'] is True and catalog['registry_fingerprint']==REGISTRY,
                     'complete native action space required before building formal training library')
        blocks=corpus.split('train')
        for index,block in enumerate(blocks):
            if not block.record['original_numeric_bytes']:continue
            group=calibration.reference_for(reference,block.record['dtype'],block.record['prepared_block_bytes'])
            with NativeGraphTask(client,block,weights=(0.4,0.3,0.3),E0=group['E0'],D0=group['D0'],warmups=1,repeats=5) as task:
                initial=task.initial_state();legal=set(task.legal_actions(initial))
                for proposal in proposals:
                    aid=proposal['action_id']
                    if aid not in legal:
                        failures.append({'block_fingerprint':block.fingerprint,'action_id':aid,'status':'inapplicable_by_native_mask'})
                        continue
                    try:
                        after=task.transition(initial,aid)
                        outcome=task.evaluate_terminal(after)
                        cold=calibration.native(a.cold_cli.resolve(),'decode',{'frame_hex':outcome.compressed_payload.hex()})
                        data.require(cold['roots']==block.roots(),'library independent cold decode mismatch')
                        measurements.append({**block.identity(),'recipe_action_ids':[aid],
                            'frame_sha256':data.digest(outcome.compressed_payload),'cold_decode_verified':True,
                            'rho':outcome.compression_ratio,'E':outcome.metrics.compression_ns_per_byte,
                            'D':outcome.metrics.decompression_ns_per_byte,'E0':group['E0'],'D0':group['D0'],
                            'native_receipt':outcome.operator_graph['native_receipt']})
                    except RuntimeError as error:
                        # Algorithm-specific range restrictions remain visible.
                        # Correctness/contract ValueErrors are fatal, never skipped.
                        failures.append({'block_fingerprint':block.fingerprint,'action_id':aid,'status':'native_execution_rejected','reason':str(error)})
            if (index+1)%50==0 or index+1==len(blocks):
                print(json.dumps({'event':'library_progress','blocks':index+1,'total':len(blocks),'measurements':len(measurements),'rejections':len(failures)}),flush=True)
    data.require(data.digest(a.cli.read_bytes())==cli_sha,'native library worker binary changed')
    grouped=collections.defaultdict(list)
    for r in measurements:grouped[r['dtype'],tuple(r['recipe_action_ids'])].append(r)
    recipes=[]
    for (dtype,ids),members in sorted(grouped.items()):
        # Keep every successful seed recipe; C++ evaluates compatible library
        # candidates on the current block within the 64-evaluation budget.
        recipes.append({'recipe_id':'native-seed-'+data.digest(data.canonical([dtype,ids]))[:24],
            'dtype':dtype,'recipe_action_ids':list(ids),'source_split':'train',
            'source_block_fingerprints':sorted({r['block_fingerprint'] for r in members}),
            'measurements_sha256':data.digest(data.canonical(members))})
    data.require({r['dtype'] for r in recipes}=={b.record['dtype'] for b in blocks if b.record['original_numeric_bytes']},'library lacks a training dtype')
    evidence_path=a.output.with_name(a.output.stem+'.measurements.json')
    evidence={'schema':'native-training-library-measurements-v1','corpus_fingerprint':corpus.fingerprint,
        'reference_sha256':reference['calibration_sha256'],'measurements':measurements,'rejections':failures}
    calibration.write_new(evidence_path,evidence)
    value={'schema':'weighted-grpo-training-solution-library-v1','library_format':'native-ngf1-smallest32-v1',
        'source_scope':'training_only','cohort_version':data.gate.VERSION,'corpus_fingerprint':corpus.fingerprint,
        'action_registry_fingerprint':REGISTRY,'native_catalog_sha256':data.digest(a.catalog.read_bytes()),
        'native_server_sha256':cli_sha,'reference_sha256':reference['calibration_sha256'],
        'source_selection':'all nonempty training blocks; validation and test excluded',
        'seed_stage_proposals':list(SEED_STAGES),'online_action_space_restricted':False,
        'selection_contract':'C++ current-block complete evaluation; all library evaluations charged within 64',
        'recipes':recipes,'measurement_receipt':{'path':evidence_path.name,
            'file_sha256':data.digest(evidence_path.read_bytes()),'measurement_count':len(measurements),
            'rejection_count':len(failures)},
        'test_used_for_training_statistics':False,'validation_used_for_library':False}
    value['library_content_sha256']=data.digest(data.canonical(value))
    calibration.write_new(a.output,value)
    print(json.dumps({'event':'library_complete','recipes':len(recipes),'measurements':len(measurements),'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
