#!/usr/bin/env python3
"""Sequential single-GPU seeds, frozen selection, and native held-out comparison.

Intended to run in a dedicated tmux session. Every phase records its argv and
exit status. A failed gate/job stops the chain without touching unrelated jobs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Thread limits must precede imports of numerical libraries in all subprocesses.
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import native_prepared_cohort_v1 as data
import native_training_corpus_v1 as binding
import native_frozen_evaluation_v1 as frozen
import train_native_smallest32_v1 as training
import formal_weighted_grpo_training_v1 as engine


def resource_snapshot(gpu_uuid: str,workers: int):
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                       text=True,capture_output=True,check=True)
    rows=[r.strip().split(', ') for r in gpu.stdout.splitlines()]
    matches=[r for r in rows if r[0]==gpu_uuid]
    data.require(len(matches)==1,'selected GPU UUID not present')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],
                        text=True,capture_output=True,check=True)
    data.require(not any(row.strip().startswith(gpu_uuid+',') for row in apps.stdout.splitlines()),'selected GPU has a compute process; campaign stopped')
    data.require(int(matches[0][1])<=10 and int(matches[0][2])==0,'selected GPU is not idle')
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    data.require(1<=workers<=4,'native workers outside 1..4')
    required_gib=workers*64+16
    data.require(mem['MemAvailable']>=required_gib*1024*1024,
                 'campaign requires '+str(required_gib)+' GiB available for bounded native workers and shared-host headroom')
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),gpu_uuid=gpu_uuid,gpu_rows=rows,
                compute_processes=apps.stdout.splitlines(),memory_kib=mem)


def recover_owned_orphan(root: Path,seed: int,supervisor_argv: list[str],*,
                         gpu_uuid: str,workers: int,command):
    """Recover only a ledger-marked orphan before the strict fresh GPU gate."""
    state_path=root/'seed_supervision'/f'seed_{seed}.json'
    if not state_path.exists() or json.loads(state_path.read_bytes()).get('status')!='running':
        return None
    marker='--recover-only'
    data.require(marker not in supervisor_argv,'duplicate supervisor recovery flag')
    separator=supervisor_argv.index('--')
    recovery=[*supervisor_argv[:separator],marker,*supervisor_argv[separator:]]
    command(f'seed_{seed}_recover_owned_orphan',recovery)
    # This remains the same fail-closed check used before every new attempt.
    # A mismatched/unknown GPU process is never excused by the ledger recovery.
    snapshot=resource_snapshot(gpu_uuid,workers)
    engine._atomic_json(root/f'seed_{seed}_resource_after_recovery.json',snapshot)
    return snapshot


def run(a):
    data.require(sys.platform.startswith('linux'),'formal campaign runs on the configured Linux host')
    admitted=training.admission(a.admission)
    for artifact in (a.cli,a.catalog,a.cold_cli,a.baseline_cli,a.library):
        data.require(str(artifact.resolve()) in admitted['frozen_files'],
                     'campaign selected an unadmitted artifact: '+str(artifact))
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid)
    state_path=root/'campaign_state.json'
    identity={key:str(value.resolve()) if isinstance(value,Path) else value for key,value in vars(a).items()}
    state={'schema':'native-smallest32-campaign-v1','identity':identity,'status':'starting','phases':[]}
    if state_path.exists():
        state=json.loads(state_path.read_bytes());data.require(state['identity']==identity,'campaign identity changed')
    def persist(status):
        state['status']=status;state['updated_at']=datetime.now(timezone.utc).isoformat();engine._atomic_json(state_path,state)
    def command(phase,argv,allowed=(0,)):
        log=root/(phase+'.log')
        state['active_phase']=phase;persist('running')
        with log.open('ab') as out:
            result=subprocess.run(argv,env=environment,cwd=Path(__file__).resolve().parents[1],stdout=out,stderr=subprocess.STDOUT)
        row=dict(phase=phase,argv=list(map(str,argv)),returncode=result.returncode,log=str(log))
        state['phases'].append(row);persist('running' if result.returncode in allowed else 'failed')
        data.require(result.returncode in allowed,'campaign phase failed: '+phase+'; see '+str(log))
    script=Path(__file__).resolve().parent
    common=[]
    for key in ('cohort','materialized','reference','admission','cli','catalog','cold_cli','library','output'):
        common+=['--'+key.replace('_','-'),str(getattr(a,key).resolve())]
    common+=['--cpus',a.cpus]
    for seed in (0,1,2):
        directory=root/f'seed_{seed}'
        initialization=[sys.executable,str(script/'train_native_smallest32_v1.py'),*common,'--seed',str(seed),'--phase','initialize']
        if not (directory/'resume_checkpoint.json').exists():command(f'seed_{seed}_initialize',initialization)
        worker=[sys.executable,str(script/'train_native_smallest32_v1.py'),*common,
                '--seed',str(seed),'--phase','train']
        supervisor_argv=[sys.executable,str(script/'supervise_weighted_grpo_seed_v1.py'),
            '--run-root',str(root),'--seed',str(seed),'--cwd',str(script.parent),
            '--budget-seconds','57600','--',*worker]
        recover_owned_orphan(root,seed,supervisor_argv,gpu_uuid=a.gpu_uuid,
                             workers=len(a.cpus.split(',')),command=command)
        supervisor=root/'seed_supervision'/f'seed_{seed}.json'
        consumed=json.loads(supervisor.read_bytes())['consumed_wall_ns'] if supervisor.exists() else 0
        if consumed<57600*10**9:
            snapshot=resource_snapshot(a.gpu_uuid,len(a.cpus.split(',')))
            engine._atomic_json(root/f'seed_{seed}_resource_start.json',snapshot)
            command(f'seed_{seed}_supervised_training',supervisor_argv,allowed=(0,124))
            result=json.loads(supervisor.read_bytes())
            data.require(result['status'] in ('complete','hard_deadline_reached','budget_exhausted','abandoned_attempt_budget_exhausted'),
                         'seed process failed; preserve last complete checkpoint for diagnosis')
        command(f'seed_{seed}_frozen_validation',[sys.executable,str(script/'train_native_smallest32_v1.py'),
                    *common,'--seed',str(seed),'--phase','frozen-validation'])
    receipts=[json.loads((root/f'seed_{s}'/'frozen-validation_receipt.json').read_bytes()) for s in (0,1,2)]
    selected=min(receipts,key=lambda r:(r['best_validation_score'],r['seed']))
    checkpoint=Path(selected['best_policy_checkpoint']);model=checkpoint.with_suffix('.wgpn')
    selection={'schema':'native-smallest32-frozen-model-selection-v1','seed':selected['seed'],
        'checkpoint':str(checkpoint),'model':str(model),'model_sha256':data.digest(model.read_bytes()),
        'selection_cost':selected['best_validation_score'],'test_used_for_selection':False,
        'seed_scores':{str(r['seed']):r['best_validation_score'] for r in receipts}}
    engine._atomic_json(root/'frozen_selection.json',selection)
    # Final timings on both sides are serial on this same physical CPU.
    test_cpu=a.cpus.split(',')[0]
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    data.require(mem['MemAvailable']>=80*1024*1024,'final CPU evaluation requires 80 GiB available')
    engine._atomic_json(root/'test_resource_start.json',{'timestamp':datetime.now(timezone.utc).isoformat(),
        'cpu':int(test_cpu),'memory_kib':mem,'thread_limit':1,'measurement_concurrency':1})
    # Test inputs are bound only by the child after model selection completes.
    command('frozen_test',[sys.executable,str(script/'native_frozen_evaluation_v1.py'),
        '--cohort',str(a.cohort),'--materialized',str(a.materialized),'--reference',str(a.reference),
        '--model',str(model),'--cli',str(a.cli),'--catalog',str(a.catalog),'--library',str(a.library),
        '--cold-cli',str(a.cold_cli),'--output',str(root/'model_test'),'--split','test',
        '--cpus',test_cpu,'--warmups','1','--repeats','5'])
    variant_directories=[]
    for variant in ('beam30','no_target_guidance','no_reuse'):
        directory=root/('test_'+variant)
        variant_directories.extend(['--variant',str(directory)])
        command('frozen_test_'+variant,[sys.executable,str(script/'native_frozen_evaluation_v1.py'),
            '--cohort',str(a.cohort),'--materialized',str(a.materialized),'--reference',str(a.reference),
            '--model',str(model),'--cli',str(a.cli),'--catalog',str(a.catalog),'--library',str(a.library),
            '--cold-cli',str(a.cold_cli),'--output',str(directory),'--split','test',
            '--cpus',test_cpu,'--warmups','1','--repeats','5','--variant',variant])
    command('baseline86_test',[sys.executable,str(script/'run_native_baseline86_smallest32_v1.py'),
        '--cohort',str(a.cohort),'--materialized',str(a.materialized),'--cli',str(a.baseline_cli),
        '--output',str(root/'baseline86_test'),'--cpus',test_cpu,'--warmups','1','--repeats','5'])
    command('comparison_summary',[sys.executable,str(script/'summarize_native_comparison_smallest32_v1.py'),
        '--model',str(root/'model_test'),'--baseline',str(root/'baseline86_test'),*variant_directories,
        '--output',str(root/'comparison')])
    persist('complete')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','materialized','reference','admission','cli','catalog','cold-cli','baseline-cli','library','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpus',required=True)
    p.add_argument('--gpu-uuid',required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        path=a.output/'campaign_state.json'
        if path.exists():
            state=json.loads(path.read_bytes());state.update(status='failed',error=str(error),error_type=type(error).__name__)
            engine._atomic_json(path,state)
        raise

if __name__=='__main__':main()
