#!/usr/bin/env python3
"""Authorized local sequence: remaining65 Beam -> RL -> unified 66-weight replay.

No simultaneous codec measurement phases. No automatic retry, timeout change,
candidate omission or old-run relabeling. A failed phase stops this scheduler.
Use --launch only after the completed 433 main-results PPT has been delivered.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import run_weight_grid_rl_remaining65_v1 as remaining

grid = remaining.grid
ROOT = grid.ROOT
SCHEMA = 'serial-remaining65-then-unified66-v1'


def prepare_ppt_snapshot(output, original, resume):
    """Freeze delivery evidence once; never bind long jobs to the user's live PPT."""
    folder=output/'delivery_snapshot'
    snapshot=folder/'main_delivered.pptx'
    receipt_path=folder/'SOURCE.json'
    if snapshot.exists() or receipt_path.exists():
        if not resume or not snapshot.is_file() or not receipt_path.is_file():
            raise AssertionError('existing/incomplete delivery snapshot requires verified resume')
        receipt=grid.read_json(receipt_path)
        if (receipt['original_path']!=str(original)
                or receipt['snapshot_sha256']!=grid.digest(snapshot.read_bytes())):
            raise AssertionError('immutable delivered-PPT snapshot identity mismatch')
        return snapshot,receipt_path,receipt
    if not original.is_file() or original.suffix.lower()!='.pptx':
        raise RuntimeError('delivered main PPTX path required for initial snapshot')
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise RuntimeError('new delivery snapshot folder must be empty')
    original_sha=grid.digest(original.read_bytes())
    shutil.copy2(original,snapshot)
    if grid.digest(snapshot.read_bytes())!=original_sha:
        raise AssertionError('main PPT changed while its immutable snapshot was copied')
    receipt=dict(original_path=str(original),original_delivered_sha256=original_sha,
        snapshot_path=str(snapshot),snapshot_sha256=original_sha,created_at=grid.now(),
        live_original_may_be_edited_without_invalidating_experiments=True)
    grid.dump(receipt_path,receipt)
    return snapshot,receipt_path,receipt


def phase_commands(output):
    weights = [','.join(map(str,w)) for w in remaining.remaining_weights()]
    return {
        'beam65': [sys.executable,'-B',str(ROOT/'scripts/run_weight_grid_beam30_v1.py'),
            '--output',str(output/'beam65'),'--weights',*weights,'--tail-scopes','open',
            '--reference-source','historical','--candidate-cap','6144'],
        'rl65': [sys.executable,'-B',str(ROOT/'scripts/run_weight_grid_rl_remaining65_v1.py'),
            '--output',str(output/'rl65'),'--beam-output',str(output/'beam65')]
    }


def phase_complete(folder, name):
    if not (folder/'SUMMARY.json').exists():return False
    summary = grid.read_json(folder/'SUMMARY.json')
    if summary.get('status') != 'COMPLETE':return False
    if (folder/'STATUS.json').exists() and grid.read_json(folder/'STATUS.json').get('status') != 'COMPLETE':
        return False
    expected = set(remaining.remaining_weights())
    if set(map(tuple,summary['config']['weights_tenths'])) != expected:
        raise AssertionError(name+' completed weight set mismatch')
    if name=='beam65':
        return (summary['completed_searches']==1300 and len(summary['columns'])==20
            and summary['config']['tail_scopes']==['open'] and summary['config']['width']==30
            and summary['config']['reference_source']=='historical')
    return (summary['completed_runs']==7800 and len(summary['columns'])==20
        and summary['remaining65_protocol_complete'] and len(summary['benchmark_result_paths'])==20)


def require_main_433(folder):
    summary = grid.read_json(folder/'SUMMARY.json')
    if (summary['status']!='COMPLETE' or not summary['main_433_protocol_complete']
            or summary['completed_runs']!=120 or summary['config']['weights_tenths']!=[[4,3,3]]
            or len(summary['columns'])!=20 or len(summary['benchmark_result_paths'])!=20):
        raise AssertionError('completed exact 120-run 433 source required before background continuation')
    if grid.read_json(folder/'STATUS.json')['status'] != 'COMPLETE':
        raise AssertionError('main 433 STATUS is not COMPLETE')
    return summary


def compact_progress(folder):
    path=folder/'PROGRESS.json'
    if not path.exists():path=folder/'STATUS.json'
    if not path.exists():return {'status':'STARTING'}
    row=grid.read_json(path)
    keys=('status','phase','completed_searches','expected_searches','completed_runs','expected_runs',
          'column','key','weight','weight_id','method','seed','elapsed_seconds')
    return {key:row[key] for key in keys if key in row}


def run_phase(name, command, output, identity, resume):
    folder=output/name
    if phase_complete(folder,name):return {'phase':name,'status':'PREVIOUSLY_COMPLETE_NOT_REEXECUTED'}
    if (folder/'MANIFEST.json').exists():
        if not resume:raise RuntimeError('existing partial phase requires --resume')
        command=[*command,'--resume']
    log_path=output/(name+'.runner.log')
    with log_path.open('ab',buffering=0) as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        receipt=dict(phase=name,pid=child.pid,command=command,started_at=grid.now(),log_path=str(log_path))
        grid.dump(output/(name+'.launch.json'),receipt)
        try:
            while child.poll() is None:
                grid.dump(output/'STATUS.json',dict(status='RUNNING',phase=name,experiment_identity=identity,
                    child_pid=child.pid,at=grid.now(),child_progress=compact_progress(folder)))
                time.sleep(30)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    if child.returncode != 0 or not phase_complete(folder,name):
        raise RuntimeError(f'{name} failed/coverage incomplete, exit={child.returncode}; inspect {log_path}; no automatic retry')
    return {**receipt,'status':'COMPLETE','ended_at':grid.now(),'returncode':child.returncode}


def verify_source_benchmark(benchmark, frozen, source_summary, weights, methods):
    if (benchmark['experiment_identity'] != source_summary['experiment_identity']
            or frozen['experiment_identity'] != source_summary['experiment_identity']):
        raise AssertionError('source benchmark/frozen/summary experiment identity mismatch')
    expected={(grid.weight_id(weight),method) for weight in weights for method in methods}
    rows={(row['weight_id'],row['method']):row for row in benchmark['rows']}
    selections={(row['weight_id'],row['method']):row for row in frozen['selections']}
    if (len(rows)!=len(benchmark['rows']) or len(selections)!=len(frozen['selections'])
            or set(rows)!=expected or set(selections)!=expected):
        raise AssertionError('source benchmark precise weight/method coverage mismatch')
    for key,row in rows.items():
        selection=selections[key]
        for name,value in selection.items():
            if name!='recipe' and row.get(name)!=value:
                raise AssertionError('source benchmark alias differs from frozen selection')
        for name in ('specs','terminals','bytes','raw_bytes','frame_sha256'):
            if row[name]!=selection['recipe'][name]:
                raise AssertionError('source benchmark recipe/frame differs from frozen selection')
        if (tuple(row['weight_tenths']) not in weights
                or grid.weight_id(tuple(row['weight_tenths']))!=row['weight_id']):
            raise AssertionError('source benchmark weight values inconsistent with IDs')


def unified66(main_output, other_output, output, resume):
    """Freeze every source recipe, then remeasure all 66 weights together."""
    main=require_main_433(main_output)
    other=grid.read_json(other_output/'SUMMARY.json')
    if not phase_complete(other_output,'rl65'):raise AssertionError('remaining65 not complete')
    if main['columns'] != other['columns']:raise AssertionError('433/remaining65 exact input columns differ')
    main_refs=grid.read_json(main_output/'REFERENCES_AND_BASELINES.json')['references']
    other_refs=grid.read_json(other_output/'REFERENCES_AND_BASELINES.json')['references']
    if main_refs != other_refs:raise AssertionError('433/remaining65 reference constants differ')
    columns=main['columns']
    source_paths=[Path(__file__),Path(__file__).with_name('test_weight_grid_remaining65_v1.py')]
    source_paths += [folder/name for folder in (main_output,other_output) for name in
                     ('SUMMARY.json','MANIFEST.json','REFERENCES_AND_BASELINES.json')]
    source_paths += [Path(path) for summary in (main,other) for path in summary['benchmark_result_paths']]
    source_paths += [folder/'benchmark'/column['key']/'FROZEN_SELECTION.json'
                     for folder in (main_output,other_output) for column in columns]
    _,gate=grid.prior.legacy_runner.input_columns()
    runtime_before=grid.source_snapshot(columns,gate)
    before={**runtime_before['file_sha256'],**{str(path):grid.digest(path.read_bytes()) for path in source_paths}}
    identity=grid.fingerprint(dict(schema='unified66-replay-v1',source_sha256=before,
        dependencies=runtime_before['dependencies'],references=main_refs,
        main_experiment=main['experiment_identity'],remaining_experiment=other['experiment_identity']))
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'MANIFEST.json'
    if manifest.exists():
        if not resume or grid.read_json(manifest)['experiment_identity']!=identity:
            raise AssertionError('unified66 resume source/reference drift')
    else:
        if any(output.iterdir()):raise RuntimeError('new unified66 output must be empty')
        grid.dump(manifest,dict(schema='unified66-replay-v1',experiment_identity=identity,
            source_sha256=before,dependencies=runtime_before['dependencies'],columns=columns,reference_digest=grid.fingerprint(main_refs),
            selection_frozen_from_independent_sources=True,created_at=grid.now()))
    benchmarks=[]
    historical={row['key']:row for row in grid.read_json(grid.HISTORICAL_BENCHMARK)['columns']}
    for column in columns:
        key=column['key'];selections=[]
        for folder,source_role,source_summary,source_weights in (
                (main_output,'independently_completed_433',main,{(4,3,3)}),
                (other_output,'fresh_remaining65',other,set(remaining.remaining_weights()))):
            benchmark=grid.read_json(folder/'benchmark'/key/'RESULT.json')
            if benchmark['column']!=column:raise AssertionError('source column drift')
            frozen=grid.read_json(folder/'benchmark'/key/'FROZEN_SELECTION.json')
            methods={row['method'] for row in historical[key]['methods']} | {
                'ppo','grpo','fresh_fixed5_J_best','fresh_complete30_J_best','common38_J_best','beam30_open'}
            if source_role=='independently_completed_433':methods.add('beam30_fixed_b4')
            verify_source_benchmark(benchmark,frozen,source_summary,source_weights,methods)
            for row in benchmark['rows']:
                grid.prior.verify_frame(row)
                selections.append(dict(method=row['method'],weight_id=row['weight_id'],
                    weight_tenths=row['weight_tenths'],selection='frozen_source_recipe_unified66_replay_NO_reselection',
                    source_role=source_role,source_experiment_identity=benchmark['experiment_identity'],
                    source_selection=row.get('selection'),source_seed=row.get('source_seed'),recipe=grid.raw_recipe(row)))
        if len({(r['weight_id'],r['method']) for r in selections}) != len(selections):
            raise AssertionError('source weight/method duplicate')
        for method in ('ppo','grpo','beam30_open'):
            if {tuple(r['weight_tenths']) for r in selections if r['method']==method}!=set(grid.weight_grid()):
                raise AssertionError('unified66 main-method weight coverage incomplete')
        result=grid.benchmark_column(column,selections,[],main_refs[key],output/'benchmark'/key,identity,resume)
        benchmarks.append(result)
        grid.dump(output/'STATUS.json',dict(status='RUNNING',phase='unified66_remeasurement',
            completed_columns=len(benchmarks),expected_columns=20,experiment_identity=identity))
    runtime_after=grid.source_snapshot(columns,gate)
    after={**runtime_after['file_sha256'],**{str(path):grid.digest(path.read_bytes()) for path in source_paths}}
    if before!=after or runtime_before['dependencies']!=runtime_after['dependencies']:
        raise AssertionError('unified66 source/runtime changed during replay')
    grid.dump(output/'SUMMARY.json',dict(schema='unified66-replay-v1',status='COMPLETE',all_66_weights_complete=True,
        experiment_identity=identity,columns=columns,weights_tenths=[list(w) for w in grid.weight_grid()],
        source_runs={'433':str(main_output),'remaining65':str(other_output)},source_sha256=after,
        dependencies=runtime_after['dependencies'],
        source_training_runs={'433':120,'remaining65':7800},source_main_Beam30_open_searches={'433':20,'remaining65':1300},
        reference_constants_identical=True,all_methods_final_timed_together_per_column=True,
        final_selection_not_changed_by_remeasurement=True,
        source_note='433 is independently completed source, never copied/relabelled as remaining65 training',
        fixed_b4_scope='Beam30 fixed_b4 measured only at 433; open-tail Beam30 covers all66 weights',
        aggregates=grid.aggregate_results(benchmarks),
        benchmark_result_paths=[str(output/'benchmark'/c['key']/'RESULT.json') for c in columns],
        J_cross_weight_warning='Different weights are different objectives; compare methods within a weight, not absolute J across weights'))
    remaining.fresh.export_csv(output/'PER_COLUMN.csv',benchmarks)
    grid.dump(output/'STATUS.json',dict(status='COMPLETE',completed_columns=20,expected_columns=20,
                                       experiment_identity=identity,all_66_weights_complete=True))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--main433-output',type=Path,required=True)
    parser.add_argument('--main-ppt',type=Path,required=True)
    parser.add_argument('--authorized-after-ppt-delivery',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args(argv)
    if not args.authorized_after_ppt_delivery:parser.error('explicit post-delivery authorization flag required')
    output=args.output.resolve();main_output=args.main433_output.resolve();ppt=args.main_ppt.resolve()
    require_main_433(main_output)
    snapshot_exists=(output/'delivery_snapshot'/'main_delivered.pptx').is_file()
    if not (args.resume and snapshot_exists) and (not ppt.is_file() or ppt.suffix.lower()!='.pptx'):
        raise RuntimeError('delivered main PPTX path required')
    if args.launch:
        output.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(output),
            '--main433-output',str(main_output),'--main-ppt',str(ppt),'--authorized-after-ppt-delivery']
        if args.resume:command.append('--resume')
        log_path=output.parent/(output.name+'.scheduler.log')
        with log_path.open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        receipt=dict(status='DISPATCHED_NOT_YET_VERIFIED_RUNNING',pid=child.pid,command=command,
            started_at=grid.now(),log_path=str(log_path),source_main_ppt_path=str(ppt),
            immutable_delivery_snapshot=str(output/'delivery_snapshot'/'main_delivered.pptx'))
        grid.dump(output.parent/(output.name+'.launch.json'),receipt);print(grid.canonical(receipt));return 0
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'SCHEDULER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (output/'MANIFEST.json').exists() and any(
            path.name not in ('SCHEDULER.lock','delivery_snapshot') for path in output.iterdir()):
        raise RuntimeError('new scheduler output must be empty')
    ppt_snapshot,ppt_receipt,delivery=prepare_ppt_snapshot(output,ppt,args.resume)
    commands=phase_commands(output)
    files=[Path(__file__),Path(remaining.__file__),Path(remaining.fresh.__file__),
        Path(__file__).with_name('test_weight_grid_remaining65_v1.py'),
        ROOT/'scripts/run_weight_grid_beam30_v1.py',ROOT/'scripts/test_weight_grid_beam30_v1.py',
        ROOT/'scripts/run_weight_grid_rl_v1.py',ROOT/'scripts/test_weight_grid_rl_v1.py',
        main_output/'SUMMARY.json',ppt_snapshot,ppt_receipt]
    signatures={str(path):grid.digest(path.read_bytes()) for path in files}
    columns,gate=grid.prior.legacy_runner.input_columns()
    runtime=grid.source_snapshot(columns,gate)
    signatures={**runtime['file_sha256'],**signatures}
    plan=dict(schema=SCHEMA,commands=commands,source_sha256=signatures,dependencies=runtime['dependencies'],
        main433_output=str(main_output),main_ppt_original_path=str(ppt),
        main_ppt_original_delivered_sha256=delivery['original_delivered_sha256'],
        main_ppt_snapshot=str(ppt_snapshot),live_original_ppt_not_a_runtime_dependency=True,
        expected_searches=1300,expected_new_training_runs=7800,
        weights_tenths=[list(w) for w in remaining.remaining_weights()],
        phases=['beam65','rl65','unified66'],workers=1,parallel_codec_measurement=False,
        no_timeout_change=True,no_automatic_retry=True)
    identity=grid.fingerprint(plan);manifest=output/'MANIFEST.json'
    if manifest.exists():
        if not args.resume or grid.read_json(manifest)['experiment_identity']!=identity:
            raise AssertionError('scheduler source/plan identity drift')
    else:
        if any(path.name not in ('SCHEDULER.lock','delivery_snapshot') for path in output.iterdir()):
            raise RuntimeError('new scheduler output must be empty')
        grid.dump(manifest,{**plan,'experiment_identity':identity,'created_at':grid.now()})
    phase_receipts=[];status='RUNNING';error=None;began=time.perf_counter()
    def verify_runtime():
        current=grid.source_snapshot(columns,gate)
        if (current['dependencies']!=runtime['dependencies']
                or current['file_sha256']!=runtime['file_sha256']
                or any(grid.digest(Path(path).read_bytes())!=sha for path,sha in signatures.items())):
            raise AssertionError('scheduler frozen source/runtime changed between phases')
    try:
        for phase in ('beam65','rl65'):
            verify_runtime()
            phase_receipts.append(run_phase(phase,commands[phase],output,identity,args.resume))
            verify_runtime()
        verify_runtime()
        grid.dump(output/'STATUS.json',dict(status='RUNNING',phase='unified66',experiment_identity=identity))
        unified66(main_output,output/'rl65',output/'combined66',args.resume)
        verify_runtime()
        status='COMPLETE'
        grid.dump(output/'SUMMARY.json',dict(status=status,experiment_identity=identity,phases=phase_receipts,
            all_66_weights_complete=True,combined_summary=str(output/'combined66'/'SUMMARY.json'),
            main433_not_retrained=True,invocation_wall_seconds=time.perf_counter()-began))
    except BaseException as exc:
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        grid.dump(output/'FAILURE.json',dict(status=status,error=error,at=grid.now()));raise
    finally:
        receipt=dict(status=status,experiment_identity=identity,ended_at=grid.now(),phases=phase_receipts,
            invocation_wall_seconds=time.perf_counter()-began,error=error)
        grid.dump(output/'STATUS.json',receipt)
        with (output/'INVOCATIONS.jsonl').open('a') as stream:stream.write(grid.canonical(receipt)+'\n')
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
