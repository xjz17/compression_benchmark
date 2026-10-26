#!/usr/bin/env python3
"""Frozen C++ inference and full-frame measurements on chronological sequences.

Python reads input files and records receipts. Each search timing is the C++
outer clock, including all policy, candidate and final-encode work. Cold decode
is an independent process outside the measured interval.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import collections
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import native_prepared_cohort_v1 as data
import native_cold_decode_v1 as cold_decode
import calibrate_native_smallest32_v1 as calibration
import weight_conditioned_grpo_v1 as grpo
from native_grpo_task_v1 import NativeClient,NativeGraphTask
import native_experiment_scope_v1 as experiment_scope

VARIANTS={
    'model':('policy',True,True),
    'beam30':('beam30',True,True),
    'no_target_guidance':('policy',False,False),
    'no_reuse':('policy',True,True),
}


def search_options(variant: str) -> dict:
    data.require(variant in VARIANTS,'unknown inference variant')
    mode,pruning,ordering=VARIANTS[variant]
    return dict(search_mode=mode,enable_certified_pruning=pruning,
                enable_objective_guided_ordering=ordering)


def atomic_bytes(path: Path, value: bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as out:out.write(value);out.flush();os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def preserve_frame(directory: Path, frame: bytes) -> str:
    """Persist a chosen complete frame outside the C++ measurement interval."""
    data.require(frame.startswith(b'NGF1'), 'only complete NGF1 frames may be preserved')
    relative='frames/'+data.digest(frame)+'.ngf'
    path=directory/relative
    if path.exists():
        data.require(path.read_bytes()==frame,'preserved frame content changed')
    else:atomic_bytes(path,frame)
    return relative


def verify_frames(directory: Path, rows) -> None:
    checked={}
    for row in rows:
        if row['status']!='ok':
            data.require('frame_path' not in row,'failed or empty input cannot claim a compressed frame')
            continue
        expected='frames/'+row['frame_sha256']+'.ngf'
        data.require(row.get('frame_path')==expected,'preserved frame path/identity changed')
        if expected not in checked:
            path=directory/expected
            data.require(path.is_file() and (directory/'frames').resolve().parent==directory.resolve()
                         and path.resolve().parent==(directory/'frames').resolve(),
                         'preserved frame is missing or outside the frame directory')
            payload=path.read_bytes()
            data.require(payload.startswith(b'NGF1') and data.digest(payload)==row['frame_sha256'],
                         'preserved complete frame checksum changed')
            checked[expected]=len(payload)
        data.require(checked[expected]==row['frame_bytes'],'preserved complete frame size changed')


def verify_receipt(directory: Path, identity: dict) -> dict:
    receipt=json.loads((directory/'receipt.json').read_bytes())
    data.require(receipt.get('schema')=='native-frozen-evaluation-smallest32-v1','wrong completed receipt schema')
    data.require(receipt.get('run_identity')==identity,'completed frozen evaluation identity mismatch')
    unsigned={k:v for k,v in receipt.items() if k!='receipt_sha256'}
    data.require(receipt.get('receipt_sha256')==data.digest(data.canonical(unsigned)),'completed receipt checksum mismatch')
    data.require({r['path'] for r in receipt['worker_files']}=={f'worker_{i}.jsonl' for i in range(len(identity['cpus']))} and
                 len(receipt['worker_files'])==len(identity['cpus']),'completed evaluation worker coverage changed')
    rows=[]
    for row in receipt['worker_files']:
        data.require(data.digest((directory/row['path']).read_bytes())==row['sha256'],'completed evaluation rows changed')
        with (directory/row['path']).open() as f:rows.extend(json.loads(line) for line in f if line.strip())
    data.require(receipt['pair_count']==identity['expected_pair_count'],'completed evaluation coverage changed')
    validate_coverage(rows,identity)
    verify_frames(directory,rows)
    statuses=collections.Counter(r['status'] for r in rows)
    data.require(receipt['ok_pair_count']==statuses['ok'] and receipt['N_A_pair_count']==statuses['N/A'] and
                 receipt['failed_pair_count']==statuses['failed'],'completed evaluation status counts changed')
    costs=[r['J_selection'] if r['status']=='ok' else r['validation_constraint_cost'] for r in rows if r['status']!='N/A']
    data.require(receipt['mean_validation_cost']==float(np.mean(costs)),'completed validation summary changed')
    return receipt


def validate_coverage(rows,identity):
    pairs=[(r['block_fingerprint'],r['weight_index']) for r in rows]
    data.require(len(pairs)==len(set(pairs))==identity['expected_pair_count'],'missing or duplicate measured block/weight pairs')
    by_block=collections.defaultdict(set)
    for row in rows:
        wi=row['weight_index'];data.require(type(wi) is int and 0<=wi<6 and row['weight']==list(grpo.FIXED_WEIGHTS[wi]),'measurement weight changed')
        data.require(row.get('inference_variant')==identity['inference_variant'],'measurement inference variant changed')
        data.require(row['status'] in ('ok','failed','N/A'),'unrecognized model measurement status')
        data.require((row['status']=='N/A')==(row['original_numeric_bytes']==0),'zero-byte/N_A status mismatch')
        by_block[row['block_fingerprint']].add(wi)
    data.require(all(weights==set(range(6)) for weights in by_block.values()),'block missing one or more frozen weights')
    data.require(data.digest(data.canonical(sorted(by_block)))==identity['expected_block_fingerprints_sha256'],
                 'measured block cohort differs from expected inputs')


def sequences(blocks):
    grouped=collections.defaultdict(list)
    for block in blocks:
        r=block.record
        grouped[(block.dataset_id,r['entity_id'],r['split'],r['column'])].append(block)
    return [sorted(rows,key=lambda b:(b.record['block_index'],b.record['block_id']))
            for _,rows in sorted(grouped.items())]


def evaluate(corpus: data.Corpus, reference: dict, *, split: str, model: Path,
             cli: Path,catalog: Path,library: Path,cold_cli: Path,cpus: tuple[int,...],
             output: Path, warmups: int=0,repeats: int=1,allow_large_memory: bool=True,
             variant: str='model'):
    options=search_options(variant)
    data.require((corpus.role=='train_validation' and split=='validation') or
                 (corpus.role=='test_candidate' and split=='test'),'only isolated validation/test roles accepted')
    data.require(0<=warmups and repeats>0 and 1<=len(cpus)<=4,'invalid evaluation resources/repetitions')
    data.require(split!='test' or len(cpus)==1,'formal final model timing must be serial on the same CPU as baseline86')
    output.mkdir(parents=True,exist_ok=True)
    jobs=[(sequence,wi,w) for sequence in sequences(corpus.split(split))
                         for wi,w in enumerate(grpo.FIXED_WEIGHTS)]
    identity={'schema':'native-frozen-evaluation-identity-v1','split':split,'corpus_fingerprint':corpus.fingerprint,
        'experiment_scope':experiment_scope.load(catalog),
        'model_sha256':data.digest(model.read_bytes()),'library_sha256':data.digest(library.read_bytes()),
        'native_cli_sha256':data.digest(cli.read_bytes()),'catalog_sha256':data.digest(catalog.read_bytes()),
        'cold_cli_sha256':data.digest(cold_cli.read_bytes()),'reference_sha256':reference['calibration_sha256'],
        'driver_sha256':data.digest(Path(__file__).read_bytes()),'cpus':list(cpus),'warmups':warmups,'repeats':repeats,
        'allow_large_memory':allow_large_memory,'complete_frames_preserved':True,
        'inference_variant':variant,'search_options':options,
        'block_reuse':variant!='no_reuse',
        'candidate_budget_rule':'every_block_64' if variant=='no_reuse' else 'first_64_then_16',
        'expected_pair_count':len(corpus.split(split))*len(grpo.FIXED_WEIGHTS)}
    identity['expected_block_fingerprints_sha256']=data.digest(data.canonical(sorted(b.fingerprint for b in corpus.split(split))))
    identity_path=output/'run_identity.json'
    if identity_path.exists():data.require(json.loads(identity_path.read_bytes())==identity,'cannot reuse evaluation with different inputs/configuration')
    else:atomic_bytes(identity_path,data.canonical(identity)+b'\n')
    if (output/'receipt.json').exists():return verify_receipt(output,identity)
    # One worker owns its subprocess and CPU for the entire queue partition.
    started=time.monotonic_ns()
    def worker(index):
        rows=[]
        with NativeClient(cli,catalog,model=model,library=library,cpu=cpus[index],timeout=57600) as client:
            for job_index in range(index,len(jobs),len(cpus)):
                sequence,wi,weight=jobs[job_index]
                checkpoint=output/'jobs'/f'{job_index:08d}.json'
                job_identity={'run':identity,'blocks':[b.fingerprint for b in sequence],'weight_index':wi,'cpu':cpus[index]}
                if checkpoint.exists():
                    saved=json.loads(checkpoint.read_bytes())
                    data.require(saved['identity']==job_identity and data.digest(data.canonical(saved['rows']))==saved['rows_sha256'],
                                 'evaluation checkpoint identity/checksum mismatch')
                    data.require(len(saved['rows'])==len(sequence),'evaluation checkpoint is incomplete')
                    for measured,block in zip(saved['rows'],sequence):
                        data.require(all(measured.get(k)==v for k,v in block.identity().items()) and
                                     measured['weight_index']==wi and measured['weight']==list(weight),
                                     'evaluation checkpoint substituted source blocks or weights')
                    verify_frames(output,saved['rows'])
                    rows.extend(saved['rows']);continue
                row_start=len(rows)
                previous=();previous_source=None;ordinal=0
                for block in sequence:
                    r=block.record
                    native_ordinal=0 if variant=='no_reuse' else ordinal
                    if variant=='no_reuse':previous=();previous_source=None
                    if not r['original_numeric_bytes']:
                        rows.append({**block.identity(),'weight_index':wi,'weight':list(weight),
                                     'inference_variant':variant,'status':'N/A','reason':'zero_original_numeric_bytes'})
                        ordinal+=1
                        continue
                    ref=calibration.reference_for(reference,r['dtype'],r['prepared_block_bytes'])
                    results=[];frames=[];failure=None
                    for repeat in range(warmups+repeats):
                        with NativeGraphTask(client,block,weights=weight,E0=ref['E0'],D0=ref['D0'],
                             ordinal=native_ordinal,previous_recipe=previous,allow_large_memory=allow_large_memory,
                             warmups=1,repeats=5) as task:
                            result=client.call(dict(cmd='search',task_handle=task.task_handle,
                                beam_width=30,include_frame_hex=True,use_solution_library=native_ordinal==0,
                                dtype=r['dtype'],**options))
                        data.require(all(result.get(k)==v for k,v in options.items()),'native inference variant changed')
                        data.require(result['original_numeric_bytes']==r['original_numeric_bytes'],'evaluation denominator drift')
                        if result.get('terminal_status')=='failed':
                            data.require(result['max_frame_bytes']==1<<30 and
                                not any(k in result for k in ('frame_hex','frame_bytes','compression_ratio')),
                                'failed search must not fabricate compression metrics')
                            failed=grpo.TerminalFailure({'recipe_action_ids':result.get('recipe_action_ids',[])},
                                r['original_numeric_bytes'],result['max_frame_bytes'],result['error'])
                            failure={**block.identity(),'status':'failed','weight_index':wi,'weight':list(weight),
                                'inference_variant':variant,'native_block_index':native_ordinal,
                                'series_block_ordinal':ordinal,'previous_recipe_action_ids':list(previous),
                                'previous_recipe_source_block_id':previous_source,
                                'error':result['error'],'native_receipt':result,
                                'validation_constraint_cost':grpo.objective_value(failed,weight),
                                'compression_ratio':None,'compression_ns_per_byte':None,'decompression_ns_per_byte':None}
                            break
                        frame=bytes.fromhex(result['frame_hex'])
                        data.require(len(frame)==result['frame_bytes'] and data.digest(frame)==result['frame_sha256'],'full frame checksum/size mismatch')
                        data.require(1<=result['evaluations']<= (64 if native_ordinal==0 else 16),'search budget mismatch')
                        t=result['timing']
                        data.require(t['total_compression_ns']>=t['total_prediction_ns'] and t['total_compression_ns']>=t['final_encode_ns'],'C++ total interval invalid')
                        if repeat>=warmups:
                            results.append(result);frames.append(frame)
                    if failure is not None:
                        rows.append(failure);ordinal+=1
                        continue
                    # Timings may affect the selected candidate. Keep every
                    # paired size/time sample and select one whole observation
                    # at median E_total; never sum separate component medians.
                    representative=sorted(range(repeats),key=lambda k:results[k]['timing']['total_compression_ns'])[repeats//2]
                    chosen=results[representative];frame=frames[representative]
                    for unique in dict.fromkeys(frames):
                        cold=cold_decode.decode(cold_cli,unique,cpu=cpus[index],allow_large_memory=allow_large_memory)
                        data.require(cold['roots']==block.roots(),'independent native cold decode mismatch')
                    frame_path=preserve_frame(output,frame)
                    raw=r['original_numeric_bytes']
                    et=chosen['timing']['total_compression_ns']/raw
                    dt=statistics.median(chosen['decode_samples_ns'])/raw
                    rho=len(frame)/raw
                    objective=weight[0]*rho+weight[1]*et/(et+ref['E0'])+weight[2]*dt/(dt+ref['D0'])
                    sample_receipts=[{k:v for k,v in result.items() if k not in ('frame_hex','candidates')} for result in results]
                    row={**block.identity(),'status':'ok','weight_index':wi,'weight':list(weight),
                         'inference_variant':variant,'native_block_index':native_ordinal,
                         'series_block_ordinal':ordinal,'previous_recipe_action_ids':list(previous),
                         'previous_recipe_source_block_id':previous_source,
                         'recipe_action_ids':chosen['best_recipe_action_ids'],'frame_bytes':len(frame),
                         'frame_sha256':data.digest(frame),'frame_path':frame_path,'compression_ratio':rho,
                         'compression_time_ns':chosen['timing']['total_compression_ns'],
                         'decompression_time_ns':statistics.median(chosen['decode_samples_ns']),
                         'compression_ns_per_byte':et,'decompression_ns_per_byte':dt,
                         'J_selection':chosen['J_selection'],'J_total':objective,'reference':ref,
                         'representative_sample_index':representative,'samples':sample_receipts,
                         'cold_decode_verified':True,'timing_scope':'C++ features+frozen_policy+candidate_search+final_graph_encode',
                         'future_block_data_visible':False}
                    rows.append(row)
                    previous=tuple(chosen['best_recipe_action_ids']);previous_source=r['block_id'];ordinal+=1
                job_rows=rows[row_start:]
                data.require(len(job_rows)==len(sequence),'incomplete evaluation sequence')
                atomic_bytes(checkpoint,data.canonical({'identity':job_identity,'rows':job_rows,
                    'rows_sha256':data.digest(data.canonical(job_rows))})+b'\n')
                print(json.dumps({'event':'frozen_sequence','split':split,'job_index':job_index,'jobs':len(jobs),
                                  'worker':index,'rows':len(rows)}),flush=True)
        path=output/f'worker_{index}.jsonl'
        atomic_bytes(path,b''.join(data.canonical(row)+b'\n' for row in rows))
        return rows
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        partitions=list(pool.map(worker,range(len(cpus))))
    rows=[row for partition in partitions for row in partition]
    validate_coverage(rows,identity)
    verify_frames(output,rows)
    ok=[r for r in rows if r['status']=='ok']
    failed=[r for r in rows if r['status']=='failed']
    costs=[r['J_selection'] if r['status']=='ok' else r['validation_constraint_cost'] for r in rows if r['status']!='N/A']
    data.require(costs and all(math.isfinite(v) for v in costs),'no finite frozen validation costs')
    by_weight=collections.defaultdict(list)
    for row in ok:by_weight[str(row['weight_index'])].append(row['J_selection'])
    receipt={'schema':'native-frozen-evaluation-smallest32-v1','run_identity':identity,'split':split,'corpus_fingerprint':corpus.fingerprint,
             'inference_variant':variant,'search_options':options,
             'model_sha256':data.digest(model.read_bytes()),'library_sha256':data.digest(library.read_bytes()),
             'native_cli_sha256':data.digest(cli.read_bytes()),'catalog_sha256':data.digest(catalog.read_bytes()),
             'cpus':list(cpus),'warmups':warmups,'repeats':repeats,'pair_count':len(rows),'ok_pair_count':len(ok),
             'N_A_pair_count':sum(r['status']=='N/A' for r in rows),'failed_pair_count':len(failed),
             'mean_exact_J':float(np.mean([r['J_selection'] for r in ok])) if ok else None,
             'mean_validation_cost':float(np.mean(costs)),
             'mean_exact_J_by_weight':{k:float(np.mean(v)) for k,v in by_weight.items()},
             'selection_statistic':'macro_mean_J_selection_with_explicit_infeasible_constraint_penalties' if split=='validation' else None,
             'test_used_for_model_selection':False,'wall_ns':time.monotonic_ns()-started,
             'worker_files':[{ 'path':f'worker_{i}.jsonl','sha256':data.digest((output/f'worker_{i}.jsonl').read_bytes())} for i in range(len(cpus))]}
    receipt['receipt_sha256']=data.digest(data.canonical(receipt))
    atomic_bytes(output/'receipt.json',data.canonical(receipt)+b'\n')
    return receipt


def main():
    import argparse
    import native_training_corpus_v1 as binding
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','materialized','reference','model','cli','catalog','library','cold-cli','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('test',),required=True)
    p.add_argument('--cpus',required=True);p.add_argument('--warmups',type=int,default=1);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--variant',choices=tuple(VARIANTS),default='model')
    a=p.parse_args()
    source=data.load(a.cohort,a.materialized,role='train_validation')
    reference=binding.load_reference(a.reference,source)
    test=data.bind_receipt(source.receipt,role='test_candidate')
    print(json.dumps(evaluate(test,reference,split=a.split,model=a.model,cli=a.cli,catalog=a.catalog,
        library=a.library,cold_cli=a.cold_cli,cpus=tuple(map(int,a.cpus.split(','))),output=a.output,
        warmups=a.warmups,repeats=a.repeats,variant=a.variant),sort_keys=True))


if __name__=='__main__':main()
