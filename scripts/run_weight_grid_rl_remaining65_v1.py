#!/usr/bin/env python3
"""Fresh paired RL on the 65 integer-tenth weights excluding (4,3,3).

Historical zlib6 constants are identical to the independently completed 433 run.
Beam30 source is open-tail, completed per-weight search. No source/checkpoint or
winner is imported into a learner; only final replay sees Beam and old recipes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import run_weight_grid_rl_433_retrain_v1 as fresh

grid = fresh.grid
SCHEMA = 'historical-reference-remaining65-fresh-rl-v1'


def remaining_weights():
    return [weight for weight in grid.weight_grid() if weight != (4,3,3)]


def load_sources(columns, weights, beam_output):
    historical = {row['key']:row for row in grid.read_json(grid.HISTORICAL_BENCHMARK)['columns']}
    summary = grid.read_json(beam_output/'SUMMARY.json')
    expected_config = dict(width=30, reference_source='historical')
    if summary['status'] != 'COMPLETE' or any(summary['config'][k] != v for k,v in expected_config.items()):
        raise AssertionError('completed historical-reference Beam30 source required')
    if set(map(tuple,summary['config']['weights_tenths'])) != set(weights) or summary['config']['tail_scopes'] != ['open']:
        raise AssertionError('Beam65 exact weights/open-tail coverage mismatch')
    if summary['completed_searches'] != len(columns)*len(weights):
        raise AssertionError('Beam65 search count mismatch')
    beam_refs = grid.read_json(beam_output/'REFERENCES.json')['columns']
    extras = [beam_output/name for name in ('SUMMARY.json','MANIFEST.json','REFERENCES.json')]
    references, beam_rows = {}, {}
    for column in columns:
        key = column['key']
        if historical[key]['kind'] != column['kind']:
            raise AssertionError('historical/current column kind mismatch')
        ref = copy.deepcopy(historical[key]['reference'])
        fresh.check_reference(column,ref)
        if beam_refs[key] != ref:
            raise AssertionError('Beam/RL historical reference constant drift')
        references[key] = ref
        extras.append(Path(ref['frame_path']))
        beam_rows[key] = {}
        for weight in weights:
            folder = beam_output/'runs'/key/grid.weight_id(weight)/'open'
            row = grid.read_json(folder/'RESULT.json')
            frozen = grid.read_json(folder/'FROZEN_SELECTION.json')
            if (row['method'] != 'beam30_open' or row['weight_tenths'] != list(weight)
                    or row['column'] != column or row['reference'] != ref
                    or row['budget']['width'] != 30 or row['selected']['mixed_coarse_depth'] > 10
                    or row['experiment_identity'] != summary['experiment_identity']
                    or frozen['experiment_identity'] != row['experiment_identity']):
                raise AssertionError('Beam65 exact selected search identity mismatch')
            for name in ('specs','terminals','sequence','frame_sha256','bytes','raw_bytes','cost'):
                if row['selected'][name] != frozen['selected'][name]:
                    raise AssertionError('Beam65 selected/frozen mismatch')
            if row['receipt_sha256'][str(folder/'FROZEN_SELECTION.json')] != grid.digest((folder/'FROZEN_SELECTION.json').read_bytes()):
                raise AssertionError('Beam65 frozen selection checksum mismatch')
            grid.prior.verify_frame(row['selected'])
            # Do not retain all frontier/trace data in RAM; the learner never sees it.
            beam_rows[key][grid.weight_id(weight)] = dict(method='beam30_open',selected=row['selected'],
                experiment_identity=row['experiment_identity'],weight_tenths=list(weight))
            extras += [folder/'RESULT.json',folder/'FROZEN_SELECTION.json',Path(row['selected']['frame_path'])]
    return historical,references,beam_rows,extras


def build_selections(run_rows, shared, historical, beam_rows, weights):
    selections = []
    for weight in weights:
        for method in ('ppo','grpo'):
            candidates = [r for r in run_rows if r['method']==method and r['weight_tenths']==list(weight)]
            winner = grid.prior.select_across_seeds(candidates)
            selections.append(dict(method=method,weight_tenths=list(weight),weight_id=grid.weight_id(weight),
                selection='fresh_remaining65_training_best_of_requested_seeds',source_seed=winner['seed'],
                before_benchmark_selection_cost=winner['selected']['cost'],
                source_selection_origin=winner['selected']['selection_origin'],
                seed_metrics=[dict(seed=r['seed'],cost=r['selected']['cost'],rho=r['selected']['compression_rate'],
                    E=r['selected']['encode_seconds_median'],D=r['selected']['decode_seconds_median']) for r in candidates],
                recipe=winner['selected']))
        baselines = [grid.score(row,shared['reference'],weight) for row in shared['baselines']]
        for method,count in (('fresh_fixed5_J_best',5),('fresh_complete30_J_best',30),('common38_J_best',38)):
            chosen = min(baselines[:count],key=grid.prior.score_rank)
            selections.append(dict(method=method,weight_tenths=list(weight),weight_id=grid.weight_id(weight),
                selection='new_common_tail_catalog_measurement_and_J_selection',
                before_benchmark_selection_cost=chosen['cost'],recipe=chosen))
        for old in historical['methods']:
            selections.append(dict(method=old['method'],weight_tenths=list(weight),weight_id=grid.weight_id(weight),
                selection='frozen_historical_recipe_replay_NOT_retrained',recipe=grid.raw_recipe(old)))
        beam = beam_rows[grid.weight_id(weight)]
        selections.append(dict(method='beam30_open',weight_tenths=list(weight),weight_id=grid.weight_id(weight),
            selection='frozen_new_per_weight_Beam30_recipe_replayed_NOT_reselected',
            source_experiment_identity=beam['experiment_identity'],
            before_benchmark_selection_cost=beam['selected']['cost'],recipe=grid.raw_recipe(beam['selected'])))
    if len({(r['weight_id'],r['method']) for r in selections}) != len(selections):
        raise AssertionError('method/weight alias collision')
    return selections


def source_snapshot(columns, gate, extras):
    return fresh.source_snapshot(columns,gate,[*extras,Path(__file__),
        Path(__file__).with_name('test_weight_grid_remaining65_v1.py')])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--beam-output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--free-space-floor-gib',type=float,default=15)
    args = parser.parse_args(argv)
    if args.free_space_floor_gib < 0:parser.error('nonnegative disk floor required')
    columns,gate = grid.prior.legacy_runner.input_columns()
    weights = remaining_weights()
    historical,references,beam_rows,extras = load_sources(columns,weights,args.beam_output.resolve())
    config = {**grid.BASE_CONFIG,'seeds':list(grid.SEEDS),'weights_tenths':[list(w) for w in weights],
        'workers':1,'action_catalog':grid.rp.legacy_tail.catalog(),'tail_catalog':grid.TAILS,
        'action_count':77,'grammar':'17 families; 38 prefix choices; 38 tails; STOP',
        'not_30_family_graph_policy':True,'reference_source':str(grid.HISTORICAL_BENCHMARK),
        'reference_digest':grid.fingerprint(references),'reference_constants_recalibrated':False,
        'normalization':'raw rho; E/(E+E0), D/(D+D0); same historical zlib6 constants as independent 433',
        'training_per_weight':'new actor/critic/optimizer per column,method,seed,weight; no warm-start',
        'cache':'new raw metric vectors shared across own 65 weights; no old,Beam,or other learner candidate exposure',
        'common_baselines':'38 new empty-prefix measurements frozen before any learner starts',
        'measurement_protocol':'training 0+1; own-pool selection 1+7; final frozen recipes 2+11 interleaved plus cold decode',
        'invalid_reward':-5.0,'J_cross_weight_warning':'different weights define different objectives; never rank weights by absolute J',
        'no_heldout_generalization_claim':True,'excluded_weight_tenths':[4,3,3],
        'beam_source':str(args.beam_output.resolve()),'history_and_Beam_final_replay_only':True}
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    lock = (output/'RUN.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    grid.check_space(output,args.free_space_floor_gib)
    before = source_snapshot(columns,gate,extras)
    identity = grid.fingerprint(dict(schema=SCHEMA,config=config,columns=columns,
        file_sha256=before['file_sha256'],dependencies=before['dependencies']))
    manifest = output/'MANIFEST.json'
    if manifest.exists():
        if not args.resume or grid.read_json(manifest)['experiment_identity'] != identity:
            raise AssertionError('resume source/input/reference/protocol identity mismatch')
    else:
        if any(p.name!='RUN.lock' for p in output.iterdir()):raise RuntimeError('new output must be empty')
        grid.dump(manifest,dict(schema=SCHEMA,experiment_identity=identity,config=config,columns=columns,
            provenance_before=before,created_at=grid.now(),command=sys.argv))
    oracle = grid.MeasurementOracle(output/'measurements.sqlite3',identity,floor_gib=args.free_space_floor_gib)
    began = time.perf_counter();status='RUNNING';error=None;paths=[];benchmarks=[];costs=Counter()
    grid.dump(output/'STATUS.json',dict(status=status,phase='common_baselines',at=grid.now(),experiment_identity=identity))
    try:
        shared = fresh.prepare_shared(columns,references,output,oracle,identity,args.resume)
        for column in columns:
            raw = Path(column['input_path']).read_bytes()
            if grid.digest(raw) != column['input_sha256']:raise AssertionError('input drift')
            runs=[]
            for weight_index,weight in enumerate(weights):
                for seed_index,seed in enumerate(grid.SEEDS):
                    methods=('ppo','grpo') if (weight_index+seed_index+column['numeric_index'])%2==0 else ('grpo','ppo')
                    initial=None
                    for method in methods:
                        folder=output/'runs'/column['key']/grid.weight_id(weight)/f'{method}_{seed}'
                        folder.parent.mkdir(parents=True,exist_ok=True)
                        row=grid.run_training(column,raw,references[column['key']],shared[column['key']]['baselines'],
                            method,seed,weight,config,folder,oracle,identity,args.resume)
                        if initial is not None and initial != row['budget']['initial_actor_sha256']:
                            raise AssertionError('paired initial actor mismatch')
                        initial=row['budget']['initial_actor_sha256'];runs.append(row);paths.append(str(folder/'RESULT.json'))
                        for name in ('train_seconds','search_seconds','optimizer_seconds','refinement_seconds','run_seconds'):
                            costs[name]+=row[name]
                        progress=dict(status='RUNNING',phase='training',completed_runs=len(paths),expected_runs=7800,
                            column=column['key'],weight_id=grid.weight_id(weight),method=method,seed=seed,
                            elapsed_seconds=time.perf_counter()-began,oracle=oracle.totals())
                        grid.dump(output/'STATUS.json',progress);grid.dump(output/'PROGRESS.json',progress);grid.prior.emit(**progress)
            selections=build_selections(runs,shared[column['key']],historical[column['key']],beam_rows[column['key']],weights)
            grid.dump(output/'selections'/(column['key']+'.json'),dict(experiment_identity=identity,
                selections=selections,frozen_at=grid.now()))
            benchmarks.append(grid.benchmark_column(column,selections,shared[column['key']]['baselines'],references[column['key']],
                output/'benchmark'/column['key'],identity,args.resume))
        after=source_snapshot(columns,gate,extras);grid.prior.legacy_runner.verify_unchanged(before,after)
        if len(paths)!=7800 or len(benchmarks)!=20:raise AssertionError('remaining65 formal coverage incomplete')
        status='COMPLETE'
        previous=[json.loads(l) for l in (output/'INVOCATIONS.jsonl').read_text().splitlines()] if (output/'INVOCATIONS.jsonl').exists() else []
        grid.dump(output/'SUMMARY.json',dict(schema=SCHEMA,status=status,remaining65_protocol_complete=True,
            all_66_weights_complete=False,excluded_weight_tenths=[4,3,3],experiment_identity=identity,config=config,
            columns=columns,run_result_paths=paths,completed_runs=len(paths),expected_runs=7800,
            benchmark_result_paths=[str(output/'benchmark'/c['key']/'RESULT.json') for c in columns],
            aggregates=grid.aggregate_results(benchmarks),oracle=oracle.totals(),completed_run_cost_sums=dict(costs),
            provenance_after=after,invocation_wall_seconds=time.perf_counter()-began,
            all_recorded_invocation_wall_seconds=sum(r['invocation_wall_seconds'] for r in previous)+time.perf_counter()-began,
            historical_methods_preserved_not_retrained=True,final_selection_not_changed_by_remeasurement=True))
        fresh.export_csv(output/'PER_COLUMN.csv',benchmarks)
    except BaseException as exc:
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        grid.dump(output/'FAILURE.json',dict(status=status,at=grid.now(),error=error));raise
    finally:
        receipt=dict(status=status,ended_at=grid.now(),experiment_identity=identity,completed_runs=len(paths),
            expected_runs=7800,completed_benchmarks=len(benchmarks),invocation_wall_seconds=time.perf_counter()-began,
            oracle=oracle.totals(),error=error)
        with (output/'INVOCATIONS.jsonl').open('a') as stream:stream.write(grid.canonical(receipt)+'\n')
        grid.dump(output/'STATUS.json',receipt);grid.dump(output/'PROGRESS.json',receipt)
        oracle.close();fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
