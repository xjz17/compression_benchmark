#!/usr/bin/env python3
"""Admitted C++ graph GRPO training on one GPU; frozen validation stays C++.

New cohort/native execution identity; no old data/gate loader is relabeled.
The external seed supervisor enforces the cumulative 16-hour hard deadline.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import native_prepared_cohort_v1 as data
import native_training_corpus_v1 as binding
from native_training_factory_v1 import Factory
from native_grpo_task_v1 import REGISTRY
import native_experiment_scope_v1 as experiment_scope
import native_frozen_evaluation_v1 as frozen
import formal_weighted_grpo_training_v1 as engine
import weight_conditioned_grpo_v1 as grpo
from export_weighted_grpo_native_policy_v1 import export_checkpoint

SCHEMA='native-smallest32-grpo-training-v1'


def admission(path: Path) -> dict:
    doc=json.loads(path.read_bytes())
    data.require(doc.get('schema')=='native-smallest32-training-admission-v2' and doc.get('passed') is True,'native training admission absent')
    data.require(doc['action_registry_fingerprint']==REGISTRY and doc['action_count']==6341 and
                 doc['catalog_fixed_configuration_count']==193 and
                 doc['fixed_configuration_count']==191,'admission action inventory drift')
    scope=experiment_scope.load()
    data.require(doc.get('experiment_scope')==scope,'admission experiment scope changed')
    data.require(doc.get('frozen_files',{}).get(scope['profile_path'])==scope['profile_sha256'],
                 'experiment profile missing from admitted frozen files')
    required={'all_fixed_configuration_roundtrips','independent_cold_decode','native_policy_parity',
              'native_feature_parity','typed_graph_and_corruption','probability_state_isolation',
              'baseline86_inventory','training_bridge_and_budget'}
    data.require(set(doc['gates'])==required and all(v is True for v in doc['gates'].values()),'incomplete native admission gates')
    for row in doc['receipts'].values():
        p=Path(row['path']);data.require(p.is_absolute() and p.is_file() and data.digest(p.read_bytes())==row['sha256'],'admission receipt changed')
    data.require(set(doc['receipts'])==required,'gate evidence missing')
    for name,sha in doc['frozen_files'].items():
        p=Path(name);data.require(p.is_absolute() and p.is_file() and data.digest(p.read_bytes())==sha,'admitted source/binary changed: '+name)
    return doc


def save_policy(policy,config,seed,iteration,path,identity):
    grpo.save_checkpoint(grpo.TrainingResult(policy,(),config.grpo_config,seed,iteration),path,
        extra={'native_run_identity':identity,'training_split_only':True})
    target=path.with_suffix('.wgpn')
    if target.exists():
        with tempfile.TemporaryDirectory() as tmp:
            check=Path(tmp)/'model.wgpn';export_checkpoint(path,check)
            data.require(check.read_bytes()==target.read_bytes(),'immutable frozen policy export changed')
    else:export_checkpoint(path,target)
    return target


def train(a):
    cpus=tuple(map(int,a.cpus.split(',')))
    data.require(len(set(cpus))==len(cpus) and 1<=len(cpus)<=4,'invalid CPU worker list')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        data.require(os.environ.get(name)=='1',name+' must be 1 before Python imports')
    if a.phase=='train':
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        data.require(visible.startswith('GPU-') and ',' not in visible,'training requires one explicit GPU UUID')
    gate=admission(a.admission)
    corpus=data.load(a.cohort,a.materialized,role='train_validation')
    ref=binding.load_reference(a.reference,corpus);bound=binding.bind(corpus,ref)
    data.require(gate['corpus_fingerprint']==corpus.fingerprint,'admitted data changed')
    data.require(gate['reference_sha256']==ref['calibration_sha256'],'admitted reference changed')
    for path in (a.cli,a.catalog,a.cold_cli,a.library):
        data.require(str(path.resolve()) in gate['frozen_files'],'unadmitted runtime artifact '+str(path))
    config=engine.TrainingConfig(measurement_workers=len(cpus),validation_interval=a.validation_interval,
                                max_iterations=a.max_iterations,policy_device='cuda:0')
    output=a.output.resolve()/f'seed_{a.seed}';output.mkdir(parents=True,exist_ok=True)
    with Factory(corpus,cli=a.cli,catalog=a.catalog,library=a.library,cpus=cpus) as factory:
        evidence=engine.GateEvidence(data.digest(data.canonical(gate['frozen_files'])),REGISTRY,
            tuple(sorted((k,v['sha256']) for k,v in gate['receipts'].items())),data.digest(a.admission.read_bytes()))
        policy_identity=engine._build_policy_identity(bound,factory,evidence,config)
        config_id=asdict(config);config_id.pop('max_iterations')
        # Device names are fixed for exact resume; initialization is CPU only,
        # but the optimizer is created on cuda only in the supervised phase.
        identity={'schema':SCHEMA,'policy_identity':policy_identity.to_json(),'corpus_sha256':bound.corpus_sha256,
            'native_data_fingerprint':corpus.fingerprint,'reference_sha256':ref['calibration_sha256'],
            'admission_sha256':data.digest(a.admission.read_bytes()),'configuration':config_id,
            'solution_library':factory.solution_library_identity(),'worker_cpus':list(cpus),
            'initialization':'iteration_zero_without_validation_or_gradient; not eligible for model selection',
            'selection':'C++ frozen macro mean J_selection over all validation block and six weight pairs; infeasible candidates use explicit bounded constraint penalty',
            'infeasible_candidate_cost':'1+a*max_frame_bytes/B_raw+b+c; max_frame_bytes=1073741824; no compression metrics fabricated',
            'training_budget':'external cumulative hard deadline 57600 seconds; periodic validation included'}
        identity=json.loads(data.canonical(identity))
        resume=output/'resume_checkpoint.json'
        # _load_resume's tensor records/RNG checks are cohort-independent.
        load_config=config if a.phase=='train' else engine.TrainingConfig(
            measurement_workers=len(cpus),validation_interval=a.validation_interval,policy_device='cpu')
        loaded=engine._load_resume(resume,expected_identity=identity,config=load_config)
        if loaded is None:
            data.require(a.phase=='initialize','initialize checkpoint before supervised training')
            policy=grpo.TwoHiddenMLP(policy_identity,np.random.default_rng(a.seed))
            optimizer=grpo.Adam(policy.params,config.grpo_config.learning_rate,device='cpu')
            rng=np.random.default_rng(a.seed+1_000_003)
            iteration=elapsed=0;best_score=best_checkpoint=None;trace=[];validations=[]
        else:
            policy,optimizer,rng,iteration,elapsed,best_score,best_checkpoint,trace,validations=loaded
        def persist(status):
            engine._atomic_json(resume,engine._resume_document(run_identity=identity,seed=a.seed,iteration=iteration,
                policy=policy,optimizer=optimizer,rng=rng,cumulative_wall_ns=elapsed,best_score=best_score,
                best_checkpoint=best_checkpoint,trace=trace,validations=validations,status=status))
        def validate():
            nonlocal best_score,best_checkpoint
            data.require(iteration>0,'untrained iteration zero cannot be selected')
            checkpoint=output/f'policy_iter_{iteration:08d}.json'
            model=save_policy(policy,config,a.seed,iteration,checkpoint,identity)
            directory=output/f'validation_iter_{iteration:08d}'
            # A completed, hash-checked receipt can be reused after interruption;
            # partial worker files are remeasured without entering test data.
            receipt_path=directory/'receipt.json'
            # Training has reached a batch barrier. Release its native pool
            # before starting the frozen pool so the worker memory allowance
            # never silently doubles during periodic or final validation.
            factory.close()
            value=frozen.evaluate(corpus,ref,split='validation',model=model,cli=a.cli,catalog=a.catalog,
                library=a.library,cold_cli=a.cold_cli,cpus=cpus,output=directory)
            if a.phase=='train':factory.reopen()
            value={**value,'iteration':iteration,'receipt_path':str(receipt_path)}
            if not any(v['iteration']==iteration for v in validations):validations.append(value)
            score=value['mean_validation_cost']
            if best_score is None or score<best_score:
                best_score=score;best_checkpoint=str(checkpoint)
            return value
        if a.phase=='initialize':
            data.require(iteration==0,'cannot initialize an already trained seed')
            save_policy(policy,config,a.seed,0,output/'policy_iter_00000000.json',identity)
            persist('initialized')
        elif a.phase=='frozen-validation':
            validate();persist('frozen_validation_complete')
        else:
            started=time.monotonic_ns();previous_elapsed=elapsed
            while elapsed<57600*10**9 and (a.max_iterations is None or iteration<a.max_iterations):
                next_iteration=iteration+1
                row=engine._training_iteration(policy=policy,optimizer=optimizer,factory=factory,corpus=bound,
                    rng=rng,seed=a.seed,iteration=next_iteration,config=config,registry_fingerprint=REGISTRY,
                    progress_callback=lambda event: print(json.dumps(event),flush=True))
                iteration=next_iteration;trace.append(row)
                elapsed=previous_elapsed+time.monotonic_ns()-started
                persist('running')
                save_policy(policy,config,a.seed,iteration,output/f'policy_iter_{iteration:08d}.json',identity)
                print(json.dumps({'event':'training_update','seed':a.seed,'iteration':iteration,
                    'elapsed_ns':elapsed,'parameter_change_l2':row['parameter_change_l2'],
                    'transitions':row['transitions'],'policy_execution':row['policy_execution']}),flush=True)
                if iteration%config.validation_interval==0:
                    validate();elapsed=previous_elapsed+time.monotonic_ns()-started;persist('running')
            persist('wall_clock_budget_exhausted' if elapsed>=57600*10**9 else 'max_iterations_reached')
        receipt={'schema':SCHEMA,'seed':a.seed,'phase':a.phase,'completed_iterations':iteration,
            'cumulative_training_wall_ns':elapsed,'best_validation_score':best_score,
            'best_policy_checkpoint':best_checkpoint,'test_used_for_selection':False,
            'policy_execution':dict(optimizer.execution_receipt),'run_identity':identity,
            'resume_checkpoint':str(resume),'validations':validations}
        engine._atomic_json(output/f'{a.phase}_receipt.json',receipt)
        print(json.dumps({'event':'seed_phase_complete','seed':a.seed,'phase':a.phase,'iterations':iteration}),flush=True)
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','materialized','reference','admission','cli','catalog','cold-cli','library','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpus',required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    p.add_argument('--phase',choices=('initialize','train','frozen-validation'),required=True)
    p.add_argument('--validation-interval',type=int,default=10)
    p.add_argument('--max-iterations',type=int)
    train(p.parse_args())


if __name__=='__main__':main()
