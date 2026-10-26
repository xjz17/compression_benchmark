#!/usr/bin/env python3
"""Independent read-only completed 433 RL audit. Never executes any codec.

Checks logged reward/advantage input flow and actual saved actor parameters.
This is not a full gradient replay: features/legal tensors are hash receipts,
and PPO critic parameters have per-update hashes rather than a saved checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'analysis_outputs/raw_rate_multiobjective_rl_20260907_v1/benchmark/SUMMARY.json'


def load(path):return json.loads(Path(path).read_text())
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):return digest(Path(path).read_bytes())
def canonical(row):return json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def same(a,b):return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-11)
def require(value,label):
    if not value:raise AssertionError(label)
def vector(actual,expected,label):
    require(len(actual)==len(expected) and all(same(a,b) for a,b in zip(actual,expected)),label)
def frame(record):
    path=Path(record['frame_path'])
    require(path.stat().st_size==record['bytes'] and sha(path)==record['frame_sha256'],'frame hash/size')
def recipe(row):return {k:row[k] for k in ('specs','terminals','bytes','raw_bytes','frame_sha256')}
def rank(row):return (row['cost'],row['bytes'],len(row['specs']),len(row['terminals']),tuple(row.get('sequence',[])))
def rescored(row,ref):
    enc=row['encode_seconds_median'];dec=row['decode_seconds_median']
    parts=(row['bytes']/row['raw_bytes'],enc/(enc+ref['encode_seconds_median']),dec/(dec+ref['decode_seconds_median']))
    return {**row,'cost':sum(w*x for w,x in zip((4,3,3),parts))/10}


def metrics(row,reference,repeats=None):
    if repeats is not None:
        for name in ('encode','decode'):
            times=row[name+'_seconds_repeats']
            require(len(times)==repeats and all(math.isfinite(t) and t>0 for t in times),'positive repeat vector')
            require(same(statistics.median(times),row[name+'_seconds_median']),'repeat median')
    rho=row['bytes']/row['raw_bytes'];enc=row['encode_seconds_median'];dec=row['decode_seconds_median']
    en=enc/(enc+reference['encode_seconds_median']);dn=dec/(dec+reference['decode_seconds_median'])
    cost=.4*rho+.3*en+.3*dn
    vector([row['compression_rate'],row['encode_time_normalized'],row['decode_time_normalized'],row['cost']],
           [rho,en,dn,cost],'raw rho/normalized time/J formulas')
    if 'score' in row:require(same(row['score'],-cost),'negative score')
    if 'reward' in row:require(same(row['reward'],-cost),'negative reward')


def checkpoint(path,seed,hidden):
    # NumPy only loads/samples parameter tensors. No learner or optimizer imports.
    import numpy as np
    def phash(arrays):
        result=hashlib.sha256()
        for p in arrays:
            a=np.asarray(p,dtype='<f8');result.update(json.dumps(list(a.shape)).encode());result.update(a.tobytes())
        return result.hexdigest()
    def initial(out_dim,random_seed):
        rng=np.random.default_rng(random_seed)
        arrays=[rng.normal(0.,math.sqrt(2./198),(198,hidden)),np.zeros(hidden,dtype=np.float64),
                rng.normal(0.,.01,(hidden,out_dim)),np.zeros(out_dim,dtype=np.float64)]
        return phash(arrays)
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files)=={'arr_0','arr_1','arr_2','arr_3'},'checkpoint tensor keys')
        arrays=[saved[f'arr_{i}'] for i in range(4)]
    require([a.shape for a in arrays]==[(198,hidden),(hidden,),(hidden,77),(77,)],'198-hidden-77 checkpoint')
    require(all(a.dtype==np.float64 and np.isfinite(a).all() for a in arrays),'finite float64 checkpoint')
    return dict(final_sha256=phash(arrays),fresh_actor_sha256=initial(77,seed),
                fresh_critic_sha256=initial(1,seed+3000003),parameters=sum(a.size for a in arrays),
                raw_weight_bytes=sum(a.nbytes for a in arrays),checkpoint_bytes=Path(path).stat().st_size)


def trajectories(run,records,measurements,logs,config):
    receipt=run['trajectory_audit'];rollouts=[r for r in records if r['event']=='rollout']
    updates=[r for r in records if r['event']=='actual_update_inputs']
    require(len(rollouts)==len(updates)==len(logs)==64,'64 rollout/update/log iterations')
    require(records[0]['event']=='trajectory_audit' and records[-1]['event']=='audit_close'
            and records[-1]['status']=='PASS','trajectory receipt envelope')
    valid=invalid=decisions=nonzero=informative=0;own=set();valid_le_penalty=0
    for number,(r,u,log) in enumerate(zip(rollouts,updates,logs),1):
        require(r['iteration']==u['iteration']==log['iteration']==number,'iteration order')
        require(r['input_digests']==u['input_digests'],'collector/optimizer tensor identity')
        require(u['epochs']==4 and u['gamma']==1 and u['gae_lambda']==.95,'actual update configuration')
        returns,lengths,bad=r['episode_returns'],r['lengths'],r['invalid']
        require(len(returns)==len(lengths)==len(bad)==32,'32 trajectories per iteration')
        require(all(1<=n<=7 for n in lengths),'decision horizon')
        require(r['episode_ids']==[i for i,n in enumerate(lengths) for _ in range(n)],'decision episode assignment')
        require(sum(lengths)==len(r['actions'])==len(r['old_values'])==len(r['old_logp']),'decision array lengths')
        require(r['input_digests']['features']['shape']==[sum(lengths),198]
                and r['input_digests']['legal']['shape']==[sum(lengths),77],'actual input tensor shape receipts')
        require(r['reward_calls']==sum(not b for b in bad)==log['reward_calls'],'valid reward count')
        valid+=r['reward_calls'];invalid+=sum(bad);decisions+=sum(lengths)
        offset=0
        for ret,n,is_bad,seq in zip(returns,lengths,bad,r['pipelines']):
            acts=r['actions'][offset:offset+n];offset+=n
            if is_bad:
                require(same(ret,-5.) and seq==acts[:-1],'invalid trajectory penalty/sequence')
            else:
                key=tuple(seq);own.add(key)
                require(key in measurements,'own trajectory exact measurement exists')
                require(same(ret,measurements[key]['reward']),'trajectory actual weighted measurement reward')
                require(seq==(acts[:-1] if acts[-1]==76 else acts),'valid terminal/STOP sequence')
                valid_le_penalty+=int(ret<=-5)
        weights=[1/lengths[e] for e in r['episode_ids']]
        vector(u['weights'],weights,'episode-equal weights')
        vector([log['mean_return'],log['min_return'],log['max_return']],
               [statistics.mean(returns),min(returns),max(returns)],'logged reward summary')
        if run['method']=='grpo':
            episode_adv=[];stds=[]
            for start in range(0,32,8):
                group=returns[start:start+8];mean=statistics.mean(group);std=statistics.pstdev(group);stds.append(std)
                episode_adv.extend([(x-mean)/(std+1e-8) if std>1e-8 else 0. for x in group])
            vector(u['advantages'],[episode_adv[e] for e in r['episode_ids']],'actual GRPO advantage inputs')
            vector(r['group_return_std'],stds,'GRPO population std')
            groups=sum(x>1e-8 for x in stds);informative+=groups
            require(groups==r['informative_groups']==log['informative_groups'],'informative group count')
            require(u['targets'] is None and u['group_size']==8,'GRPO no critic/group8')
        else:
            values=r['old_values'];advantages=[0.]*len(values);targets=[0.]*len(values);offset=0
            for e,n in enumerate(lengths):
                next_value=next_adv=0.
                for step in range(n-1,-1,-1):
                    ix=offset+step;reward=returns[e] if step==n-1 else 0.
                    next_adv=reward+next_value-values[ix]+.95*next_adv
                    advantages[ix]=next_adv;targets[ix]=next_adv+values[ix];next_value=values[ix]
                offset+=n
            vector(u['raw_gae_advantages'],advantages,'PPO raw GAE')
            vector(u['targets'],targets,'PPO critic targets')
            mass=sum(weights);avg=sum(w*a for w,a in zip(weights,advantages))/mass
            std=math.sqrt(sum(w*(a-avg)**2 for w,a in zip(weights,advantages))/mass)
            norm=[(a-avg)/(std+1e-8) for a in advantages] if std>1e-8 else advantages
            vector(u['advantages'],norm,'actual PPO normalized GAE inputs')
        nonzero+=sum(x!=0 for x in u['advantages'])
    require(valid==receipt['valid_episodes'] and invalid==receipt['invalid_episodes']
            and valid+invalid==receipt['episodes']==2048,'2048 trajectory receipt')
    require(decisions==receipt['decisions'] and nonzero==receipt['nonzero_advantage_decisions'] and nonzero>0,'decision/advantage receipt')
    require(valid_le_penalty==run['budget']['valid_reward_calls_at_or_below_invalid_penalty'],'valid reward penalty caveat count')
    if run['method']=='grpo':require(informative==receipt['informative_groups'] and informative>0,'nonzero GRPO group signal')
    return own,dict(episodes=2048,valid=valid,invalid=invalid,decisions=decisions,nonzero_advantages=nonzero,
                   informative_groups=informative if run['method']=='grpo' else None)


def audit_run(path,column,config,reference,shared,identity,db):
    run=load(path);folder=Path(path).parent;budget=run['budget']
    require(run['experiment_identity']==identity and run['key']==column['key'] and run['weight_tenths']==[4,3,3],'run identity')
    require(run['reference']==reference and run['input_sha256']==column['input_sha256'],'run reference/input')
    require(run['historical_winner_injection'] is False and run['shared_cache_does_not_expand_candidate_pool'],'candidate isolation flags')
    receipt=run['trajectory_audit'];tp=Path(receipt['path'])
    require(sha(tp)==receipt['sha256'] and tp.stat().st_size==receipt['bytes'],'trajectory hash/size')
    with gzip.open(tp,'rt') as stream:records=[json.loads(x) for x in stream]
    require(len(records)==receipt['event_count']==130,'trajectory event count')
    with gzip.open(run['trace_path'],'rt') as stream:events=[json.loads(x) for x in stream]
    require(events[0]['event']=='environment' and events[-1]['event']=='close','candidate envelope')
    measurements={tuple(x['sequence']):x for x in events if x['event']=='measurement'}
    logs=load(folder/'TRAINING.json');own,counts=trajectories(run,records,measurements,logs,config)
    for seq,row in measurements.items():
        metrics(row,reference,1)
        require(row['status']=='PASS' and row['lossless'] and row['runtime_sha256']==identity,'candidate certificate')
        require(row['input_sha256']==column['input_sha256'] and row['raw_bytes']==column['raw_bytes'],'candidate exact input')
        require(0<=seq[0]<38 and row['terminals']==config['tail_catalog'][seq[0]],'tail dispatch')
        require(row['specs']==[config['action_catalog'][x-38] for x in seq[1:]],'prefix dispatch')
        mixed=len(seq)-1+sum(4 if t=='bzip2' else 1 for t in row['terminals'])
        require(len(seq[1:])==len(set(seq[1:]))<=6 and row['mixed_coarse_depth']==mixed<=10,'legal repeated-action/depth bound')
        cached=db.execute('SELECT record FROM vectors WHERE cache_key=?',(row['cache_key'],)).fetchone()
        require(cached is not None,'own exact oracle request exists')
        cert=json.loads(cached[0])
        require(all(row.get(k)==v for k,v in cert.items()),'candidate vs original vector certificate')
    require(len(own)==run['training_snapshot']['unique_sampled_candidates'],'own unique sample count')
    require(sum(measurements[k]['reward']<=-5 for k in own)==budget['valid_unique_candidates_at_or_below_invalid_penalty'],
            'unique valid reward below finite invalid penalty counter')
    cp=checkpoint(folder/'actor.npz',run['seed'],config['hidden'])
    require(cp['fresh_actor_sha256']==budget['initial_actor_sha256'] and cp['final_sha256']==budget['final_actor_sha256'],'fresh/final actor hashes')
    require(cp['final_sha256']!=cp['fresh_actor_sha256'] and cp['parameters']==13325,'actual nonzero learned actor')
    require(logs[-1]['actor_sha256']==cp['final_sha256'],'last optimizer log checkpoint identity')
    require(sum(x['optimizer_steps'] for x in logs)==budget['optimizer_steps']==256,'256 actor updates')
    require(sum(x['critic_optimizer_steps'] for x in logs)==budget['critic_optimizer_steps']==(256 if run['method']=='ppo' else 0),'critic optimizer steps')
    require(all(x['initial_actor_sha256']==cp['fresh_actor_sha256'] and x['warm_start_episodes']==0 for x in logs),'fresh training no warm start')
    require(any(x['actor_parameter_change_l2']>0 for x in logs),'actor update norms')
    require(all(x['clip_ratio']==.2 and x['entropy_coefficient']==.01 and x['reference_kl_beta']==0 for x in logs),'actual policy hyperparameters')
    if run['method']=='ppo':
        require(all(x['initial_critic_sha256']==cp['fresh_critic_sha256'] for x in logs),'fresh critic hash')
        require(any(x['critic_parameter_change_l2']>0 for x in logs) and logs[-1]['critic_sha256']!=cp['fresh_critic_sha256'],'actual critic updates')
    else:require(all(x['critic_sha256'] is None for x in logs),'GRPO no critic')
    require(same(run['optimizer_seconds'],sum(x['update_seconds'] for x in logs)),'optimizer time sum')
    require(run['optimizer_seconds']<=run['train_seconds']<=run['search_seconds'],'nested training costs')
    frozen=load(folder/'REFINEMENT_POOL_FROZEN.json')
    require(frozen['experiment_identity']==identity,'frozen refinement identity')
    sampled_top=sorted((measurements[k] for k in own),key=rank)[:8]
    expected_sampled={tuple(r['sequence']) for r in sampled_top}
    actual_sampled={tuple(e['candidate']['sequence']) for e in frozen['pool'] if 'training_sampled' in e['origins']}
    require(actual_sampled==expected_sampled,'exact own top8 sampled refinement')
    for entry in frozen['pool']:
        candidate=entry['candidate'];seq=tuple(candidate['sequence'])
        require(set(entry['origins'])<=set(['training_sampled','post_training_greedy','common_safety_baseline']),'refinement origins')
        if 'training_sampled' in entry['origins']:require(seq in own,'own sampled candidate')
        if 'post_training_greedy' in entry['origins']:require(seq in measurements,'greedy actually measured')
        if 'common_safety_baseline' in entry['origins']:
            require(any(recipe(candidate)==recipe(r) for r in shared['baselines']),'common root baseline only')
            # Frozen pools deduplicate by recipe. A root already sampled/greedily
            # visited keeps its training-phase certificate while adding the
            # baseline origin; its selection-phase cache key is legitimately
            # different because the measurement protocol is 0+1 vs 1+7.
            if not any(candidate['cache_key']==r['cache_key'] for r in shared['baselines']):
                require(seq in measurements and candidate['cache_key']==measurements[seq]['cache_key']
                        and bool(set(entry['origins']) & {'training_sampled','post_training_greedy'}),
                        'deduplicated baseline retains only its own training certificate')
            best_baseline=min((rescored(r,reference) for r in shared['baselines']),key=rank)
            require(recipe(candidate)==recipe(best_baseline),'common baseline truly minimizes same J')
    refined=run['refinement_pool'];require(len(refined)==len(frozen['pool'])==budget['refinement_candidates'],'refinement pool size')
    for index,(row,entry) in enumerate(zip(refined,frozen['pool'])):
        require(recipe(row)==recipe(entry['candidate']) and row['sequence']==entry['candidate']['sequence'],'frozen refinement recipe')
        metrics(row,reference,7)
        ev=load(folder/'refinement'/str(index)/'EVIDENCE.json')
        require(all(ev[k]==row[k] for k in ('cache_key','bytes','frame_sha256','cost','encode_seconds_repeats','decode_seconds_repeats')),'refinement receipt')
    selected=min(refined,key=rank)
    require(recipe(selected)==recipe(run['selected']) and same(selected['cost'],run['selected']['cost']),'selected actual best own/refinement')
    require(same(run['selection_score'],run['selected']['score']),'selected score receipt')
    frame(run['selected'])
    delta=run['oracle_run_delta']
    for budget_key,oracle_key in [('actual_new_training_encoder_calls','training_encoder_calls'),('actual_new_training_decoder_calls','training_decoder_calls'),('refinement_encoder_calls','selection_encoder_calls'),('refinement_decoder_calls','selection_decoder_calls')]:
        require(budget[budget_key]==delta.get(oracle_key,0),'actual oracle call accounting')
    require(budget['logical_refinement_encoder_calls']==budget['logical_refinement_decoder_calls']==8*len(refined),'logical refinement budget')
    return run,dict(path=str(path),method=run['method'],seed=run['seed'],column=run['key'],status='PASS',
                   trajectory_counts=counts,checkpoint=cp,own_candidates=len(own),refinement_candidates=len(refined))


def audit_completed(output):
    started=time.perf_counter();summary=load(output/'SUMMARY.json');manifest=load(output/'MANIFEST.json')
    require(summary['status']=='COMPLETE' and summary['main_433_protocol_complete'],'wait for complete protocol')
    require(summary['all_66_weights_complete'] is False,'433 does not claim 66 weights')
    config=manifest['config'];identity=manifest['experiment_identity'];columns=manifest['columns']
    require(len(columns)==20 and config['weights_tenths']==[[4,3,3]] and len(config['seeds'])==3,'20 columns / 3 seeds /433')
    require(config['iterations']==64 and config['episodes']==32 and config['action_count']==77,'frozen training budget/grammar')
    require(summary['experiment_identity']==identity,'summary identity')
    require(summary['config']==config and summary['columns']==columns,'summary/manifest exact protocol inputs')
    expected_runs={str(output/'runs'/c['key']/'a04_b03_c03'/f'{method}_{seed}'/'RESULT.json')
                   for c in columns for seed in config['seeds'] for method in ('ppo','grpo')}
    expected_benchmarks={str(output/'benchmark'/c['key']/'RESULT.json') for c in columns}
    require(len(summary['run_result_paths'])==120 and set(summary['run_result_paths'])==expected_runs,
            'summary exact run provenance paths')
    require(len(summary['benchmark_result_paths'])==20 and set(summary['benchmark_result_paths'])==expected_benchmarks,
            'summary exact benchmark provenance paths')
    provenance=manifest['provenance_before']
    require(all(Path(p).is_file() and sha(p)==h for p,h in provenance['file_sha256'].items()),'frozen source/input/dependency hashes')
    require(summary['provenance_after']['file_sha256']==provenance['file_sha256'],'before/after source hashes')
    require(summary['provenance_after']['dependencies']==provenance['dependencies'],'before/after dependency versions')
    history={c['key']:c for c in load(HISTORY)['columns']};beam=Path(config['beam_source']);beamrefs=load(beam/'REFERENCES.json')['columns']
    shared=load(output/'REFERENCES_AND_BASELINES.json');references=shared['references']
    require(digest(canonical(references).encode())==config['reference_digest'],'frozen reference digest')
    require(shared['experiment_identity']==identity,'shared reference/baseline identity')
    db=sqlite3.connect('file:'+str(output/'measurements.sqlite3')+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA query_only=ON')
    reports=[];runs=[];benchmark_rows=[];benchmarks=[];cold=0;checkpoints=0
    for column in columns:
        key=column['key'];ref=references[key]
        require(ref==history[key]['reference']==beamrefs[key]==shared['columns'][key]['reference'],'historical/Beam/RL exact references')
        require(sha(column['input_path'])==column['input_sha256'] and ref['input_sha256']==column['input_sha256'],'exact input')
        frame(ref)
        require(len(shared['columns'][key]['baselines'])==38,'38 common baseline candidates')
        for index,base in enumerate(shared['columns'][key]['baselines']):
            require(base['specs']==[] and base['sequence']==[index]
                    and base['terminals']==config['tail_catalog'][index]
                    and base['input_sha256']==column['input_sha256'] and base['kind']==column['kind'],'common baseline identity')
            cached=db.execute('SELECT record FROM vectors WHERE cache_key=?',(base['cache_key'],)).fetchone()
            require(cached is not None and all(base.get(k)==v for k,v in json.loads(cached[0]).items()),'common baseline exact oracle vector')
        colruns=[]
        for seed in config['seeds']:
            initial=None
            for method in ('ppo','grpo'):
                path=output/'runs'/key/'a04_b03_c03'/f'{method}_{seed}'/'RESULT.json'
                run,report=audit_run(path,column,config,ref,shared['columns'][key],identity,db)
                require(initial is None or initial==run['budget']['initial_actor_sha256'],'paired actor initialization')
                initial=run['budget']['initial_actor_sha256'];runs.append(run);reports.append(report);colruns.append(run);checkpoints+=1
        b=load(output/'benchmark'/key/'RESULT.json');frozen=load(output/'benchmark'/key/'FROZEN_SELECTION.json')
        require(b['experiment_identity']==frozen['experiment_identity']==identity,'final benchmark identity')
        require(b['column']==column,'final benchmark exact input column')
        require(len(b['rows'])==len(frozen['selections'])==21,'21 final methods')
        require(b['selection_not_changed_by_final_timing'] is True,'final no reselection')
        require([r['method'] for r in b['rows']]==[r['method'] for r in frozen['selections']],'frozen method order')
        expected_pool={canonical(recipe(r)) for r in [*shared['columns'][key]['baselines'],
                                                     *[x['recipe'] for x in frozen['selections']]]}
        require(expected_pool=={canonical(recipe(r)) for r in b['recipes'].values()},'final exact baseline/selection pool')
        require(len(b['round_orders'])==13 and all(len(order)==len(b['recipes'])
                and set(order)==set(b['recipes']) for order in b['round_orders']),
                '13 complete interleaved recipe permutations')
        for method in ('ppo','grpo'):
            winner=min((r for r in colruns if r['method']==method),key=lambda r:(-r['selection_score'],r['seed']))
            selected=next(x for x in frozen['selections'] if x['method']==method)
            require(selected['source_seed']==winner['seed'] and recipe(selected['recipe'])==recipe(winner['selected']),'frozen true best seed')
        for method,count in [('fresh_fixed5_J_best',5),('fresh_complete30_J_best',30),('common38_J_best',38)]:
            candidate=min((rescored(r,ref) for r in shared['columns'][key]['baselines'][:count]),key=rank)
            selected=next(x for x in frozen['selections'] if x['method']==method)
            require(recipe(candidate)==recipe(selected['recipe']),'fresh fixed catalog J optimum')
        for old in history[key]['methods']:
            selected=next(x for x in frozen['selections'] if x['method']==old['method'])
            require(recipe(selected['recipe'])==recipe(old) and 'NOT_retrained' in selected['selection'],'old recipes preserved/not trained')
        for scope in ('fixed_b4','open'):
            bm=load(beam/'runs'/key/'a04_b03_c03'/scope/'RESULT.json')['selected']
            selected=next(x for x in frozen['selections'] if x['method']=='beam30_'+scope)
            require(recipe(selected['recipe'])==recipe(bm),'Beam fixed before new RL/final timing')
        groups=defaultdict(list)
        for row,selection in zip(b['rows'],frozen['selections']):
            metrics(row,ref,11)
            require(row['input_sha256']==column['input_sha256'] and row['kind']==column['kind']
                    and row['raw_bytes']==column['raw_bytes'],'final measured input identity')
            require(row['weight_tenths']==[4,3,3] and row['weight_id']=='a04_b03_c03','final weight identity')
            require(recipe(row)==recipe(selection['recipe']),'final frozen recipe equality')
            require(row['lossless'] and row['deterministic'] and row['cold_decode_pass'],'real benchmark cold decode receipt')
            groups[canonical(recipe(row))].append(row)
        for aliases in groups.values():
            first=aliases[0]
            require(all(all(x[k]==first[k] for k in ('bytes','encode_seconds_repeats','decode_seconds_repeats','cost')) for x in aliases),'same-recipe aliases identical')
        for row in b['recipes'].values():
            frame(row);require(row['cold_decode_pass'] and row['warmups']==2 and row['repeats']==11,'final frame receipt')
            for name in ('encode','decode'):
                require(len(row[name+'_seconds_repeats'])==11 and same(statistics.median(row[name+'_seconds_repeats']),row[name+'_seconds_median']),'final median from11')
        require(b['complete_encoder_calls']==b['complete_decoder_calls']==13*len(b['recipes']),'final actual codec calls')
        require(b['cold_decoder_process_calls']==len(b['recipes']),'cold process count')
        cold+=b['cold_decoder_process_calls'];benchmarks.append(b);benchmark_rows.extend(b['rows'])
    grouped=defaultdict(list)
    for b in benchmarks:
        for row in b['rows']:
            for ds in (b['column']['dataset'],'ALL'):grouped[(ds,row['weight_id'],row['method'])].append(row)
    require(len(summary['aggregates'])==len(grouped),'aggregate group count')
    for a in summary['aggregates']:
        rows=grouped[(a['dataset'],a['weight_id'],a['method'])];raw=sum(r['raw_bytes'] for r in rows);size=sum(r['bytes'] for r in rows)
        require(a['raw_bytes']==raw and a['compressed_bytes']==size and a['columns']==len(rows),'aggregate bytes/columns')
        vector([a['compression_rate'],a['encode_seconds_sum_of_column_medians'],a['decode_seconds_sum_of_column_medians'],a['mean_normalized_encode_time'],a['mean_normalized_decode_time'],a['mean_J']],
            [size/raw,sum(r['encode_seconds_median'] for r in rows),sum(r['decode_seconds_median'] for r in rows),statistics.mean(r['encode_time_normalized'] for r in rows),statistics.mean(r['decode_time_normalized'] for r in rows),statistics.mean(r['cost'] for r in rows)],'weighted dataset aggregates')
    require(len(runs)==120 and len(benchmark_rows)==420,'120 runs/420 final records')
    require(sum(r['budget']['rollouts'] for r in runs)==245760,'245760 actual trajectories')
    require(sum(r['budget']['optimizer_steps'] for r in runs)==30720,'30720 actor updates')
    require(sum(r['budget']['critic_optimizer_steps'] for r in runs)==15360,'15360 PPO critic updates')
    for key,value in summary['completed_run_cost_sums'].items():require(same(value,sum(r[key] for r in runs)),'nested cost sum '+key)
    phase_rows=db.execute('SELECT phase,n,wall,enc,dec FROM phase_totals').fetchall();db.close()
    interrupted_attempts=list((output/'runs').glob('*/*/*.interrupted-*'))
    unassigned_interrupted_calls={}
    for phase,n,wall,enc,dec in phase_rows:
        reported=summary['oracle']['persistent_by_phase'][phase]
        require(n==reported['unique_vectors'] and enc==reported['complete_encoder_calls'] and dec==reported['complete_decoder_calls'],'oracle phase counters')
        require(same(wall,reported['measurement_wall_seconds']),'oracle wall sum')
        expected=sum(r['oracle_run_delta'].get(phase+'_encoder_calls',0) for r in runs)+(20*38*8 if phase=='selection' else 0)
        require(enc>=expected,'actual calls cover learners + baseline calibration')
        if enc!=expected:
            require(bool(interrupted_attempts),'extra oracle work requires preserved interrupted attempt')
            unassigned_interrupted_calls[phase]=enc-expected
    with (output/'PER_COLUMN.csv').open() as stream:csvrows=list(csv.DictReader(stream))
    require(len(csvrows)==420,'CSV row count')
    for a,b in zip(csvrows,benchmark_rows):
        require(a['method']==b['method'],'CSV method order')
        vector([float(a[k]) for k in ('compression_rate','encode_seconds_median','decode_seconds_median','cost')],
               [b[k] for k in ('compression_rate','encode_seconds_median','decode_seconds_median','cost')],'CSV metrics')
    return dict(schema='independent-433-rl-readonly-audit-v1',status='PASS',output=str(output),
        verifier_path=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),
        audited_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        summary_sha256=sha(output/'SUMMARY.json'),manifest_sha256=sha(output/'MANIFEST.json'),
        runs=120,trajectories=245760,actor_updates=30720,critic_updates=15360,checkpoint_count=checkpoints,
        valid_trajectories=sum(r['trajectory_counts']['valid'] for r in reports),
        invalid_trajectories=sum(r['trajectory_counts']['invalid'] for r in reports),
        valid_rewards_at_or_below_invalid_penalty=sum(r['budget']['valid_reward_calls_at_or_below_invalid_penalty'] for r in runs),
        columns=20,methods_per_column=21,method_column_records=420,cold_decode_receipts=cold,
        codec_calls_by_this_audit=0,gradient_replay=False,
        evidence_limitations=['Actual advantages and collector/optimizer input hashes verified; full feature/legal tensors were not retained, so this is not full gradient replay.',
            'Saved actor checkpoint compared with fresh seed initialization and final log hash; PPO critic updates checked via actual per-iteration norms/hashes/targets, not a saved critic checkpoint.',
            'Frame hashes certify retained artifacts; lossless/cold-decode claims refer to actual completed benchmark receipts, not new decoding by this audit.'],
        preserved_interrupted_attempts=[str(p) for p in interrupted_attempts],
        oracle_encoder_calls_outside_completed_run_deltas=unassigned_interrupted_calls,
        run_audits=reports,issues=[],audit_wall_seconds=time.perf_counter()-started)


def audit(output):
    # Refuse an active writer, even if SUMMARY appeared just before its exit.
    # immutable=1 below is safe only for this completed, checkpointed database.
    with (output/'RUN.lock').open('rb') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        try:return audit_completed(output)
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    try:result=audit(args.output.resolve())
    except Exception as exc:
        import traceback
        result=dict(schema='independent-433-rl-readonly-audit-v1',status='FAIL',error=str(exc),traceback=traceback.format_exc(),codec_calls_by_this_audit=0)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
