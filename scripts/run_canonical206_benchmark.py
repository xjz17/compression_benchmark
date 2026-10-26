#!/usr/bin/env python3
"""Fresh Beam29/PPO/GRPO comparison on the two smallest complete datasets."""
from __future__ import annotations

import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_v]='1'

import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from canonical206_data import load_smallest_two, dataset_ranking_manifest
from canonical206_search_env import Env, SPECS, beam29, encode_cold, decode_frame, grammar_manifest
from canonical206_rl_policy import train, propose, save_policy, load_policy
from search_physical_operator_variants_v2 import MLP

from canonical182_policy_paths import (PolicyPath, concrete_path, path_from_names, path_from_record, path_record, require_current_registry)
import primitive_operator_action_runtime as policy_runtime

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_RUN=subprocess.run


def bounded_run(*args,**kwargs):
    kwargs.setdefault('timeout',30)
    return ORIGINAL_RUN(*args,**kwargs)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def key(column):return f"{column['dataset']}__{column['numeric_index']:02d}"


def output_rows(output):
    return [json.loads(p.read_text()) for p in sorted((output/'columns').glob('*.json'))]


def select_and_measure(column,method,seed,actor,args):
    begin=time.perf_counter()
    env=Env(column,args.sample_rows,args.max_steps)
    if method=='beam29':
        _prefix,levels=beam29(env,29,progress=True)
        proposal_count=0
    else:
        rng=np.random.default_rng(seed+10000+column['numeric_index'])
        propose(actor,env,rng,greedy=True)
        for _ in range(args.proposals-1):propose(actor,env,rng)
        levels=[];proposal_count=args.proposals
    sampled_sec=time.perf_counter()-begin
    finalists=sorted(env.nodes,key=lambda q:(env.nodes[q].frame_bytes,q))[:args.finalists]
    if () not in finalists:finalists.append(())
    full_attempts=[]
    best=None
    for prefix in finalists:
        t=time.perf_counter()
        try:
            frame=encode_cold(column,prefix)
            enc_s=time.perf_counter()-t
            back=decode_frame(frame)
            if back!=column['raw']:raise ValueError('full-column bitwise mismatch')
            full_attempts.append(dict(prefix=list(prefix),io_variants=list(concrete_path(prefix).io_variants),parameters=list(concrete_path(prefix).parameters),policy_registry_version=policy_runtime.ACTION_REGISTRY_VERSION,status='ok',frame_bytes=len(frame),encode_sec=enc_s,wall_sec=time.perf_counter()-t))
            if best is None or (len(frame),prefix)<(len(best[1]),best[0]):best=(prefix,frame)
        except Exception as exc:
            full_attempts.append(dict(prefix=list(prefix),io_variants=list(concrete_path(prefix).io_variants),parameters=list(concrete_path(prefix).parameters),policy_registry_version=policy_runtime.ACTION_REGISTRY_VERSION,status='rejected',error=str(exc)[:500],wall_sec=time.perf_counter()-t))
    if best is None:raise RuntimeError('even raw fallback failed')
    search_sec=time.perf_counter()-begin
    prefix,frame=best
    # Independent fresh encode calls after selection: no candidate/prefix cache.
    times=[];dec_times=[]
    for repeat in range(args.repeats+1):
        t=time.perf_counter(); fresh=encode_cold(column,prefix);dt=time.perf_counter()-t
        t=time.perf_counter();decoded=decode_frame(fresh);ddt=time.perf_counter()-t
        if fresh!=frame or decoded!=column['raw']:raise AssertionError('fresh timing replay differs')
        if repeat:times.append(dt);dec_times.append(ddt)
    rel=f"frames/{method}__seed{seed}__{key(column)}.c206"
    (args.output/rel).parent.mkdir(exist_ok=True)
    (args.output/rel).write_bytes(frame)
    row=dict(dataset=column['dataset'],column=column['name'],numeric_index=column['numeric_index'],
             source_index=column['source_index'],kind=column['kind'],values=column['count'],
             method=method,seed=seed,original_bytes=len(column['raw']),compressed_bytes=len(frame),
             compression_ratio=len(column['raw'])/len(frame),compressed_fraction=len(frame)/len(column['raw']),
             compose_sec=search_sec,sampled_search_sec=sampled_sec,full_rerank_sec=search_sec-sampled_sec,
             compress_sec=statistics.median(times),decompress_sec=statistics.median(dec_times),
             compress_sec_repeats=times,decompress_sec_repeats=dec_times,
             policy_registry_version=policy_runtime.ACTION_REGISTRY_VERSION,
             selected_io_variants=list(concrete_path(prefix).io_variants),selected_parameters=list(concrete_path(prefix).parameters),parameters=list(concrete_path(prefix).parameters),selected_ids=list(prefix),selected_pipeline=[SPECS[i].name for i in prefix],
             input_sha256=column['input_sha256'],frame_sha256=hashlib.sha256(frame).hexdigest(),
             frame_path=rel,sample_rows=min(column['count'],args.sample_rows),
             proposals=proposal_count,full_finalists=full_attempts,lossless=True,
             search_stats=env.summary(),beam_levels=levels,
             measured_action_ids=sorted({a for q in env.nodes for a in q}),
             rejected_action_ids=sorted({q[-1] for q in env.invalid}),
             claim='full typed numerical column; complete C206+UAR1 frame; encode/compose includes C++ CLI startup and IPC')
    dump(args.output/'columns'/f'{method}__seed{seed}__{key(column)}.json',row)
    dump(args.output/'traces'/f'{method}__seed{seed}__{key(column)}.json',env.records)
    print(json.dumps({k:row[k] for k in ('dataset','column','method','seed','compressed_bytes','compression_ratio','compose_sec','compress_sec','selected_ids')},ensure_ascii=False),flush=True)


def summarize(output):
    rows=output_rows(output);groups={}
    for r in rows:groups.setdefault((r['dataset'],r['method'],r['seed']),[]).append(r)
    per_seed=[]
    for (dataset,method,seed),rr in sorted(groups.items()):
        original=sum(r['original_bytes'] for r in rr);compressed=sum(r['compressed_bytes'] for r in rr)
        per_seed.append(dict(dataset=dataset,method=method,seed=seed,columns=len(rr),original_bytes=original,
                             compressed_bytes=compressed,compression_ratio=original/compressed,
                             compressed_fraction=compressed/original,compress_sec=sum(r['compress_sec'] for r in rr),
                             compose_sec=sum(r['compose_sec'] for r in rr),
                             decompress_sec=sum(r['decompress_sec'] for r in rr),
                             unique_edges=sum(r['search_stats']['unique_edges'] for r in rr),
                             invalid_edges=sum(r['search_stats']['rejected_edges'] for r in rr)))
    if not per_seed:return
    csv_write(output/'dataset_seed_results.csv',per_seed)
    summary=[]
    for dataset,method in sorted({(r['dataset'],r['method']) for r in per_seed}):
        rr=[r for r in per_seed if r['dataset']==dataset and r['method']==method]
        result=dict(dataset=dataset,method=method,seeds=len(rr),columns_per_seed=min(r['columns'] for r in rr))
        for field in ('original_bytes','compressed_bytes','compression_ratio','compressed_fraction','compress_sec','compose_sec','decompress_sec','unique_edges','invalid_edges'):
            result[field]=statistics.median(r[field] for r in rr)
        result['compressed_bytes_min']=min(r['compressed_bytes'] for r in rr)
        result['compressed_bytes_max']=max(r['compressed_bytes'] for r in rr)
        summary.append(result)
    csv_write(output/'dataset_summary.csv',summary)
    flat=[]
    for r in rows:
        flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))}|{'selected_ids':' -> '.join(map(str,r['selected_ids'])),'selected_pipeline':' -> '.join(r['selected_pipeline'])})
    csv_write(output/'per_column_results.csv',flat)
    dump(output/'summary.json',dict(rows=len(rows),all_lossless=all(r['lossless'] for r in rows),dataset_summary=summary,
                                  reached_actions=sorted({i for r in rows for i in r['measured_action_ids']})))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-rows',type=int,default=2048)
    parser.add_argument('--max-steps',type=int,default=7)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--episodes',type=int,default=64)
    parser.add_argument('--proposals',type=int,default=128)
    parser.add_argument('--finalists',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20260906,20260907,20260908])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    subprocess.run=bounded_run
    columns=load_smallest_two()
    if args.smoke:
        columns=[columns[0],next(c for c in columns if c['kind']=='float64')]
        args.iterations=1;args.episodes=8;args.proposals=4;args.seeds=[7];args.sample_rows=128;args.repeats=1
    sources=[ROOT/'scripts'/n for n in ('canonical206_search_env.py','canonical206_rl_policy.py','canonical206_data.py','canonical182_policy_paths.py','canonical182_policy_checkpoint.py','canonical182_registry.py','canonical156_registry.py','canonical156_parameter_policy.py','native_lz_parameters.py','native_window_parameters.py','native_action_parameters.py','canonical143_registry.py','canonical140_registry.py','max_value_bitpack.py','canonical137_registry.py','minimum_for.py','run_canonical206_benchmark.py','unified_action_runtime.py','primitive_operator_action_runtime.py','action_cpp_migration.py','search_physical_operator_variants_v2.py')]
    sources.extend(ROOT/'scripts'/n for n in ('canonical129_registry.py','canonical127_registry.py','optimal_rle_actions.py','current_action_parameters.py','block_state_parameters.py'))
    sources.extend(ROOT/'benchmark/reference/actions'/n for n in ('canonical182.json','canonical156.json','native_lz_defaults.json','native_window_defaults.json','canonical143.json','canonical140.json','canonical137.json','canonical129.json','canonical127.json'))
    sources.extend(ROOT/'cpp/src'/n for n in ('optimal_rle.cpp','optimal_rle.hpp','unified_rle_numeric.cpp','unified_rle_numeric.hpp'))
    hashes={str(q.relative_to(ROOT)):sha(q) for q in sources}
    config={k:v for k,v in vars(args).items() if k not in ('output','resume')}
    if args.output.exists() and not args.resume:raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['source_hashes']!=hashes or old['config']!=config:raise ValueError('resume source/config mismatch')
    else:
        (args.output/'code').mkdir()
        for source in sources:shutil.copy2(source,args.output/'code'/source.name)
        dump(args.output/'action_registry.json',policy_runtime.registry())
        dump(manifest_path,dict(policy_registry_version=policy_runtime.ACTION_REGISTRY_VERSION,created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),config=config,source_hashes=hashes,
                               machine=platform.platform(),python=sys.version,numpy=np.__version__,
                               cpu=platform.processor(),dataset_ranking=dataset_ranking_manifest(ROOT),
                               columns=[{k:v for k,v in c.items() if k!='raw'} for c in columns],grammar=grammar_manifest(),
                               split='two cross-dataset directions; held-out dataset never used in policy updates',
                               search='2048-row prefix unless configured; full column reranks best 8 measured prefixes plus raw',
                               timings='serial wall time; compress=median5 fresh full encodes after1 warmup; compose=features+all online codec/inverse calls+full rerank; RL training separate',
                               objective='minimize complete serialized C206+UAR1 bytes; latency measured, not part of reward',
                               budgets='PPO/GRPO same rollout/proposal budgets; Beam width29, not matched codec-call budget',
                               variant_comparison='Beam enumerates explicit variants; atomic actor uses unique context resolution and rejects ambiguity. These are not matched reachable search spaces.',
                               grpo='group size8; terminal group-relative normalization; clipped surrogate; beta0 no reference KL',
                               runtime_cli_hashes={q.name:sha(q) for q in (ROOT/'cpp/build').glob('*cli') if q.is_file()}))
    def done(method,seed,c):return (args.output/'columns'/f'{method}__seed{seed}__{key(c)}.json').exists()
    # Interleave methods for each dataset; no concurrent benchmarks on this host.
    datasets=sorted({c['dataset'] for c in columns})
    for dataset in datasets:
        test_columns=[c for c in columns if c['dataset']==dataset]
        for c in test_columns:
            if not done('beam29',0,c):select_and_measure(c,'beam29',0,None,args)
            summarize(args.output)
        train_columns=[c for c in columns if c['dataset']!=dataset]
        if args.smoke:train_columns=columns
        for seed in args.seeds:
            for method in ('grpo','ppo'):
                policy_path=args.output/'policies'/f'{method}__{seed}__test_{dataset}.npz'
                trace_path=policy_path.with_suffix('.json')
                if policy_path.exists():
                    actor=load_policy(policy_path, Env(train_columns[0],args.sample_rows,args.max_steps))
                else:
                    start=time.perf_counter()
                    envs=[Env(c,args.sample_rows,args.max_steps) for c in train_columns]
                    actor,trace=train(method,envs,seed,iterations=args.iterations,episodes=args.episodes,verbose=True)
                    training_sec=time.perf_counter()-start
                    policy_path.parent.mkdir(parents=True,exist_ok=True)
                    save_policy(actor,policy_path)
                    dump(trace_path,dict(method=method,seed=seed,heldout_dataset=dataset,training_datasets=sorted({c['dataset'] for c in train_columns}),
                                         training_sec=training_sec,trace=trace,training_envs=[e.summary() for e in envs],
                                         action_coverage=sorted({i for e in envs for q in e.nodes for i in q})))
                    print(json.dumps(dict(event='training_done',method=method,seed=seed,test_dataset=dataset,seconds=training_sec)),flush=True)
                    del envs
                for c in test_columns:
                    if not done(method,seed,c):select_and_measure(c,method,seed,actor,args)
                    summarize(args.output)
    expected=len(columns)*(1+2*len(args.seeds))
    rows=output_rows(args.output)
    if len(rows)!=expected:raise AssertionError((len(rows),expected))
    if hashes!={str(q.relative_to(ROOT)):sha(q) for q in sources}:raise AssertionError('source changed during benchmark')
    dump(args.output/'COMPLETE.json',dict(status='complete',expected_rows=expected,actual_rows=len(rows),source_hashes_unchanged=True))
    print(json.dumps(dict(event='complete',rows=len(rows),output=str(args.output))),flush=True)


if __name__=='__main__':main()
