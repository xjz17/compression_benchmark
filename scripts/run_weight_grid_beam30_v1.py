#!/usr/bin/env python3
"""Versioned full-frame, J-ranked Beam30; replay frozen historical methods.

No frozen RL/codec sources, policy checkpoints, or historical outputs are edited.
Each weight gets its own level-synchronous beam and private candidate exposure.
Only exact-request raw measurement vectors are shared across these searches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import fcntl
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_weight_grid_rl_v1 as grid

SCHEMA = 'full-frame-J-beam30-v1'
ROOT = grid.ROOT
rp, prior, tail = grid.rp, grid.prior, grid.rp.legacy_tail
HISTORICAL = grid.HISTORICAL_BENCHMARK
FORMAL_REFERENCES = ROOT / 'analysis_outputs/weight_grid_rl_20260909_v1/formal_v1/REFERENCES_AND_BASELINES.json'


def beam_search(env, budget=6144, width=30, roots=None):
    """Tail roots then complete prefix layers, all ranked by env.rank (J).

    Every attempted child (even invalid) consumes the cap. Every evaluated
    state already has its tail, so STOP requires no hidden completion/evaluation.
    A capped partial final layer can update the incumbent but is not expanded.
    """
    if budget < 1 or width < 1:
        raise ValueError('positive beam width and candidate cap required')
    roots = tuple(env.legal(())) if roots is None else tuple(roots)
    if any(root not in env.legal(()) for root in roots) or len(set(roots)) != len(roots):
        raise ValueError('distinct legal tail roots required')
    attempts = valid = invalid = deepest = 0
    frontier, layers, incumbent = [], [], None
    candidates = []
    for depth in range(7):
        expanded = [((), roots)] if depth == 0 else [(p, tuple(a for a in env.legal(p) if a != env.stop)) for p in frontier]
        layer_possible = sum(len(actions) for _, actions in expanded)
        if not layer_possible:
            break
        layer_attempts = layer_valid = layer_invalid = 0
        candidates = []
        for parent, actions in expanded:
            for action in actions:
                if attempts >= budget:
                    break
                attempts += 1; layer_attempts += 1
                child = env.transition(parent, action)
                if child is None:
                    invalid += 1; layer_invalid += 1
                    continue
                row = env.evaluate(child, origin='beam30_sampled', sampled=True)
                valid += 1; layer_valid += 1
                deepest = max(deepest, len(child)-1)
                candidates.append((env.rank(row), child))
                if incumbent is None or env.rank(row) < env.rank(incumbent):
                    incumbent = row
            if attempts >= budget:
                break
        candidates.sort(key=lambda item: (item[0], item[1]))
        frontier = [seq for _, seq in candidates[:width]]
        layers.append(dict(prefix_depth=depth, legal_child_slots=layer_possible,
                           candidate_attempts=layer_attempts, valid=layer_valid, invalid=layer_invalid,
                           completed_layer=layer_attempts == layer_possible, survivors=[list(s) for s in frontier],
                           survivor_ranks=[list(r) for r, _ in candidates[:width]],
                           incumbent_sequence=incumbent['sequence'] if incumbent else None))
        if attempts >= budget or not frontier:
            break
    if incumbent is None:
        raise RuntimeError('beam generated no valid complete recipe')
    return dict(candidate_attempts=attempts, valid_terminal_requests=valid, invalid_attempts=invalid,
                width=width, budget=budget, deepest_prefix_evaluated=deepest,
                root_count=len(roots), roots_attempted=layers[0]['candidate_attempts'], layers=layers,
                last_layer_complete=layers[-1]['completed_layer'],
                stop_reason='candidate_cap_reached' if attempts >= budget else 'frontier_or_depth_exhausted',
                ranking='(J, frame_bytes, prefix_count, tail_count, sequence)',
                implicit_stop_at_every_measured_state=True,
                best_sampled=incumbent, learned_parameters=0, optimizer_updates=0)


def verify_run(row, identity, weight, scope):
    if (row['schema'] != SCHEMA or row['experiment_identity'] != identity
            or row['weight_tenths'] != list(weight) or row['tail_scope'] != scope):
        raise AssertionError('completed beam identity drift')
    prior.verify_frame(row['selected'])
    for path, expected in row['receipt_sha256'].items():
        if grid.digest(Path(path).read_bytes()) != expected:
            raise AssertionError('completed beam receipt drift: '+path)
    frozen_path=Path(row['selected']['frame_path']).parent/'FROZEN_SELECTION.json'
    frozen=grid.read_json(frozen_path)
    if frozen['experiment_identity']!=identity or any(
        frozen['selected'][key]!=row['selected'][key] for key in
        ('specs','terminals','sequence','frame_sha256','bytes','raw_bytes','cost','weight_tenths')):
        raise AssertionError('selected result differs from frozen selection')
    if row['budget']['width'] != 30 or row['selected']['mixed_coarse_depth'] > 10:
        raise AssertionError('beam width/depth mismatch')


def run_search(column, raw, reference, weight, scope, config, folder, oracle, identity, resume):
    result_path = folder/'RESULT.json'
    if result_path.exists():
        if not resume: raise RuntimeError('completed output requires --resume')
        row = grid.read_json(result_path); verify_run(row, identity, weight, scope)
        return row
    grid.check_space(folder.parent, oracle.floor_gib)
    prior.prepare_incomplete(folder, resume)
    started = time.perf_counter(); before = oracle.stats.copy()
    env = grid.GridEnv(raw, column['kind'], reference, folder/'candidates.jsonl', weight=weight,
                       oracle=oracle, key=column['key'], method='beam30_'+scope, seed=None)
    setup_seconds = time.perf_counter()-started
    try:
        roots = None if scope == 'open' else (tail.TAILS.index(('bzip2',)),)
        began = time.perf_counter()
        budget = beam_search(env, config['candidate_cap'], 30, roots)
        search_seconds = time.perf_counter()-began
        snapshot = env.snapshot()
        own = sorted((env.measured_results[s] for s in env._sampled), key=prior.score_rank)
        root_best = min((r for r in own if len(r['sequence']) == 1), key=prior.score_rank)
        pool = {tuple(r['sequence']): r for r in [*own[:config['refine_top_k']], root_best]}
        if any(tuple(r['sequence']) not in env._sampled for r in pool.values()):
            raise AssertionError('foreign candidate in refinement pool')
        pool_list = list(pool.values())
        grid.dump(folder/'FROZEN_REFINEMENT.json', dict(frozen_at=grid.now(), experiment_identity=identity,
                  own_sampled_only=True, candidates=pool_list, no_historical_winner_injection=True))
        began = time.perf_counter(); refined=[]
        for index, candidate in enumerate(pool_list):
            row, _ = oracle.measure(raw, column['kind'], candidate['specs'], candidate['terminals'],
                                     phase='selection', warmups=1, repeats=7, expected=candidate['frame_sha256'])
            row = prior.decorate(grid.score(row, reference, weight), candidate['sequence'], 'beam30_own_refinement')
            row.update(selection_origin='beam30_own_refinement', sampled_score=candidate['score'])
            refined.append(row)
        refinement_seconds = time.perf_counter()-began
        selected = min(refined, key=prior.score_rank)
        grid.dump(folder/'FROZEN_SELECTION.json', dict(experiment_identity=identity,
                  selected=selected, frozen_at=grid.now(), final_benchmark_may_not_reselect=True))
        selected = grid.persist_selected(raw, column['kind'], selected, folder/'selected.acf3')
        row = dict(schema=SCHEMA, experiment_identity=identity, column=column, method='beam30_'+scope,
                   tail_scope=scope, weight_tenths=list(weight), weight_id=grid.weight_id(weight),
                   reference=reference, budget=budget, search_snapshot=snapshot,
                   selected=selected, refined_candidates=refined, sampled_only=budget['best_sampled'],
                   own_unique_candidates=len(own), refinement_candidates=len(refined),
                   setup_seconds=setup_seconds, search_seconds=search_seconds,
                   refinement_seconds=refinement_seconds, run_seconds=time.perf_counter()-started,
                   oracle_run_delta=dict(oracle.stats-before), shared_cache_only_raw_vectors=True,
                   no_actor_or_critic=True, no_historical_candidate_injection=True)
    finally:
        env.close()
    row['receipt_sha256']={str(p):grid.digest(p.read_bytes()) for p in
                          (env.trace_path, folder/'FROZEN_REFINEMENT.json', folder/'FROZEN_SELECTION.json')}
    grid.dump(result_path,row); verify_run(row,identity,weight,scope)
    return row


def reference_for(column, historical, reference_source):
    if reference_source == 'historical':
        record=historical['reference']
    else:
        record=grid.read_json(FORMAL_REFERENCES)['columns'][column['key']]['reference']
    if (record['input_sha256'] != column['input_sha256'] or record['raw_bytes'] != column['raw_bytes']
            or record['specs'] != [] or record['terminals'] != ['zlib6']):
        raise AssertionError('reference does not match requested exact input/zlib6')
    prior.verify_frame(record)
    return copy.deepcopy(record)


def replay_selections(historical, searches, weights):
    selections=[]
    for weight in weights:
        for item in historical['methods']:
            selections.append(dict(method=item['method'], weight_tenths=list(weight),
                weight_id=grid.weight_id(weight), selection='frozen_historical_recipe_replay_NOT_retrained',
                original_selection_criterion=item.get('selection_criterion'),
                original_source_group=item.get('source_group'), original_source_seed=item.get('source_seed'),
                recipe=grid.raw_recipe(item)))
        for row in searches:
            if row['weight_tenths']!=list(weight):continue
            selections.append(dict(method=row['method'], weight_tenths=list(weight),
                weight_id=grid.weight_id(weight), selection='new_actual_J_ranked_width30_search',
                before_benchmark_selection_cost=row['selected']['cost'],
                source_selection_origin=row['selected']['selection_origin'], recipe=row['selected']))
    return selections


def source_snapshot(columns, gate, reference_source):
    base=grid.source_snapshot(columns,gate)
    extra=[Path(__file__),Path(__file__).with_name('test_weight_grid_beam30_v1.py')]
    if reference_source=='formal':extra.append(FORMAL_REFERENCES)
    for p in extra:base['file_sha256'][str(p.resolve())]=grid.digest(p.read_bytes())
    return base


def export_csv(path, benchmarks):
    fields=['dataset','column','method','weight_id','raw_bytes','bytes','compression_rate',
            'encode_seconds_median','decode_seconds_median','encode_time_normalized',
            'decode_time_normalized','cost','chain','selection','lossless','cold_decode_pass']
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for bench in benchmarks:
            for row in bench['rows']:
                writer.writerow({key:({**row,'dataset':bench['column']['dataset'],
                                       'column':bench['column']['name']}).get(key) for key in fields})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--keys',nargs='+')
    parser.add_argument('--weights',nargs='+',default=['4,3,3'])
    parser.add_argument('--all-weights',action='store_true')
    parser.add_argument('--tail-scopes',nargs='+',choices=['open','fixed_b4'],default=['open','fixed_b4'])
    parser.add_argument('--candidate-cap',type=int,default=6144)
    parser.add_argument('--reference-source',choices=['historical','formal'],default='historical')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--skip-benchmark',action='store_true')
    parser.add_argument('--free-space-floor-gib',type=float,default=15)
    args=parser.parse_args(argv)
    weights=grid.weight_grid() if args.all_weights else [grid.validate_weight(tuple(map(int,w.split(',')))) for w in args.weights]
    if len(weights)!=len(set(weights)) or len(args.tail_scopes)!=len(set(args.tail_scopes)) or args.candidate_cap<1:
        parser.error('weights/scopes distinct; positive cap required')
    columns,gate=prior.legacy_runner.input_columns(args.keys)
    history=grid.read_json(HISTORICAL)
    historical={c['key']:c for c in history['columns']}
    references={c['key']:reference_for(c,historical[c['key']],args.reference_source) for c in columns}
    config=dict(width=30,candidate_cap=args.candidate_cap,refine_top_k=8,
        weights_tenths=[list(w) for w in weights],tail_scopes=args.tail_scopes,
        reference_source=args.reference_source,reference_constants_frozen_not_recalibrated=True,
        reference_digest=grid.fingerprint(references),catalog=tail.catalog(),tails=tail.TAILS,
        action_count=77,max_prefix=6,max_mixed_coarse_depth=10,
        grammar='17 families / 38 prefix choices / 38 tails / STOP; fixed_b4 ablation restricts only root tail',
        not_30_family_graph=True,workers=1,skip_benchmark=args.skip_benchmark,
        search_measurement='0 warmups + 1 fresh full encode/decode; exact raw-vector memoization',
        refinement_measurement='own top8 and own best root; 1 warmup + 7 repeats',
        final_measurement='frozen selections; 2 warmups + 11 randomized interleaved repeats; cold frame-only decode',
        baseline_replay='all 14 historical methods preserved, not newly trained or newly selected',
        cache='private raw vectors only, shared across own weights/scopes; no preexisting RL cache injection',
        no_cross_weight_absolute_J_ranking=True)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'RUN.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    grid.check_space(output,args.free_space_floor_gib)
    before=source_snapshot(columns,gate,args.reference_source)
    identity=grid.fingerprint(dict(schema=SCHEMA,config=config,columns=columns,
                                  file_sha256=before['file_sha256'],dependencies=before['dependencies']))
    manifest=output/'MANIFEST.json'
    if manifest.exists():
        if not args.resume or grid.read_json(manifest)['experiment_identity']!=identity:
            raise AssertionError('resume source/input/config drift')
    else:
        if any(p.name!='RUN.lock' for p in output.iterdir()):raise RuntimeError('new output must be empty')
        grid.dump(manifest,dict(schema=SCHEMA,experiment_identity=identity,config=config,columns=columns,
                               created_at=grid.now(),provenance_before=before,command=sys.argv))
        grid.dump(output/'REFERENCES.json',dict(reference_source=args.reference_source,
                  columns=references,digest=grid.fingerprint(references)))
    if grid.read_json(output/'REFERENCES.json')['columns']!=references:
        raise AssertionError('frozen reference copy drift')
    oracle=grid.MeasurementOracle(output/'measurements.sqlite3',identity,floor_gib=args.free_space_floor_gib)
    start=time.perf_counter();status='RUNNING';runs=[];benchmarks=[];error=None
    try:
        for column in columns:
            raw=Path(column['input_path']).read_bytes()
            if grid.digest(raw)!=column['input_sha256']:raise AssertionError('input drift')
            searches=[]
            for weight in weights:
                for scope in args.tail_scopes:
                    folder=output/'runs'/column['key']/grid.weight_id(weight)/scope
                    folder.parent.mkdir(parents=True,exist_ok=True)
                    row=run_search(column,raw,references[column['key']],weight,scope,config,
                                   folder,oracle,identity,args.resume)
                    runs.append(row);searches.append(row)
                    progress=dict(status='RUNNING',completed_searches=len(runs),
                        expected_searches=len(columns)*len(weights)*len(args.tail_scopes),
                        key=column['key'],scope=scope,weight=grid.weight_id(weight),
                        elapsed_seconds=time.perf_counter()-start,oracle=oracle.totals(),
                        latest_search_seconds=row['search_seconds'],latest_budget={k:v for k,v in row['budget'].items() if k not in ['layers','best_sampled']})
                    grid.dump(output/'PROGRESS.json',progress);prior.emit(**progress)
            if not args.skip_benchmark:
                selections=replay_selections(historical[column['key']],searches,weights)
                benchmark=grid.benchmark_column(column,selections,[],references[column['key']],
                    output/'benchmark'/column['key'],identity,args.resume)
                benchmarks.append(benchmark)
        after=source_snapshot(columns,gate,args.reference_source);prior.legacy_runner.verify_unchanged(before,after)
        status='COMPLETE'
        totals=Counter()
        for row in runs:
            for k in ('search_seconds','refinement_seconds','setup_seconds','run_seconds'):totals[k]+=row[k]
        grid.dump(output/'SUMMARY.json',dict(schema=SCHEMA,status=status,experiment_identity=identity,
            columns=columns,config=config,searches=[dict(column=r['column']['key'],method=r['method'],
                weight_id=r['weight_id'],candidate_attempts=r['budget']['candidate_attempts'],
                search_seconds=r['search_seconds'],refinement_seconds=r['refinement_seconds'],
                own_unique_candidates=r['own_unique_candidates'],deepest_prefix=r['budget']['deepest_prefix_evaluated'],
                last_layer_complete=r['budget']['last_layer_complete'],stop_reason=r['budget']['stop_reason'],
                selected=r['selected']) for r in runs],aggregates=grid.aggregate_results(benchmarks),
            benchmark_result_paths=[str(output/'benchmark'/c['key']/'RESULT.json') for c in columns] if benchmarks else [],
            completed_searches=len(runs),oracle=oracle.totals(),search_cost_sums=dict(totals),
            invocation_wall_seconds=time.perf_counter()-start,provenance_after=after,
            all_66_weights_complete=len(weights)==66 and len(columns)==20 and 'open' in args.tail_scopes
                and not args.skip_benchmark and len(benchmarks)==20,
            reference_source=args.reference_source,reference_constants_frozen=True,
            historical_reference_constants_frozen=args.reference_source=='historical',
            old_policies_replayed_not_retrained=True))
        if benchmarks:export_csv(output/'PER_COLUMN.csv',benchmarks)
    except BaseException as exc:
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        grid.dump(output/'FAILURE.json',dict(status=status,at=grid.now(),error=error));raise
    finally:
        receipt=dict(status=status,ended_at=grid.now(),invocation_wall_seconds=time.perf_counter()-start,
                     completed_searches=len(runs),oracle=oracle.totals(),error=error)
        with (output/'INVOCATIONS.jsonl').open('a') as stream:stream.write(grid.canonical(receipt)+'\n')
        grid.dump(output/'STATUS.json',receipt)
        oracle.close();fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
