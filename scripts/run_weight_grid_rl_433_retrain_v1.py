#!/usr/bin/env python3
"""Fresh 0.4/0.3/0.3 PPO/GRPO with identical historical reference constants.

Versioned wrapper; never edits the frozen grid, policies, codec implementation,
old outputs or Beam outputs. New common baselines and learner measurements only.
Beam and old-method recipes enter final replay, never learner candidate pools.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import fcntl
import math
from pathlib import Path
import sys
import time
import traceback

import run_weight_grid_rl_v1 as grid

SCHEMA = 'historical-reference-433-fresh-rl-v1'
WEIGHT = (4, 3, 3)
BEAM_METHODS = ('beam30_fixed_b4', 'beam30_open')
DEFAULT_BEAM = grid.ROOT / 'analysis_outputs/weight_grid_beam30_20260909_v1/formal20_v1'


def check_reference(column, record):
    if (record['input_sha256'] != column['input_sha256']
            or record['raw_bytes'] != column['raw_bytes']
            or ('kind' in record and record['kind'] != column['kind'])
            or record['specs'] != [] or record['terminals'] != ['zlib6']
            or any(not math.isfinite(record[name]) or record[name] <= 0
                   for name in ('encode_seconds_median', 'decode_seconds_median'))):
        raise AssertionError('historical reference is not the exact input/zlib6 record')
    grid.prior.verify_frame(record)


def load_sources(columns, beam_output):
    history = grid.read_json(grid.HISTORICAL_BENCHMARK)
    historical = {row['key']: row for row in history['columns']}
    beam_summary = grid.read_json(beam_output / 'SUMMARY.json')
    if beam_summary['status'] != 'COMPLETE' or beam_summary['config']['width'] != 30:
        raise AssertionError('Beam30 source must be completed width-30 search')
    if beam_summary['config']['reference_source'] != 'historical':
        raise AssertionError('Beam30 must have used the identical historical reference constants')
    source_paths = [beam_output / name for name in ('SUMMARY.json', 'MANIFEST.json', 'REFERENCES.json')]
    beam_reference = grid.read_json(beam_output / 'REFERENCES.json')['columns']
    references, beam_rows = {}, {}
    for column in columns:
        key = column['key']
        if historical[key]['kind'] != column['kind']:
            raise AssertionError('historical/current outer column kinds differ')
        reference = copy.deepcopy(historical[key]['reference'])
        check_reference(column, reference)
        if beam_reference[key] != reference:
            raise AssertionError('Beam and learner reference records differ')
        references[key] = reference
        source_paths.append(Path(reference['frame_path']))
        beam_rows[key] = []
        for scope, method in (('fixed_b4', BEAM_METHODS[0]), ('open', BEAM_METHODS[1])):
            folder = beam_output / 'runs' / key / grid.weight_id(WEIGHT) / scope
            row = grid.read_json(folder / 'RESULT.json')
            frozen = grid.read_json(folder / 'FROZEN_SELECTION.json')
            if (row['method'] != method or row['weight_tenths'] != list(WEIGHT)
                    or row['column'] != column or row['reference'] != reference
                    or row['budget']['width'] != 30
                    or row['experiment_identity'] != beam_summary['experiment_identity']
                    or frozen['experiment_identity'] != row['experiment_identity']):
                raise AssertionError('Beam input/weight/reference/identity mismatch')
            for name in ('specs', 'terminals', 'sequence', 'frame_sha256', 'bytes', 'raw_bytes', 'cost'):
                if row['selected'][name] != frozen['selected'][name]:
                    raise AssertionError('Beam selected recipe differs from frozen selection')
            if row['selected']['mixed_coarse_depth'] > 10:
                raise AssertionError('Beam depth exceeds common limit')
            grid.prior.verify_frame(row['selected'])
            for filename, sha in row['receipt_sha256'].items():
                if grid.digest(Path(filename).read_bytes()) != sha:
                    raise AssertionError('Beam receipt checksum mismatch')
            source_paths += [folder / 'RESULT.json', folder / 'FROZEN_SELECTION.json',
                             Path(row['selected']['frame_path'])]
            source_paths += [Path(filename) for filename in row['receipt_sha256']]
            beam_rows[key].append(row)
    return historical, references, beam_rows, source_paths


def source_snapshot(columns, gate, extra_paths):
    snapshot = grid.source_snapshot(columns, gate)
    paths = [Path(__file__), Path(__file__).with_name('test_weight_grid_rl_433_retrain_v1.py'),
             *extra_paths]
    for path in paths:
        snapshot['file_sha256'][str(path.resolve())] = grid.digest(path.read_bytes())
    return snapshot


def prepare_shared(columns, references, output, oracle, identity, resume):
    """Keep E0/D0 unchanged; freshly measure the 38 common safety tails."""
    frozen = output / 'REFERENCES_AND_BASELINES.json'
    if frozen.exists():
        obj = grid.read_json(frozen)
        if not resume or obj['experiment_identity'] != identity or obj['references'] != references:
            raise AssertionError('reference/baseline resume identity drift')
        if set(obj['columns']) != {c['key'] for c in columns}:
            raise AssertionError('baseline column coverage drift')
        for column in columns:
            key = column['key']
            check_reference(column, obj['references'][key])
            if obj['columns'][key]['reference'] != references[key]:
                raise AssertionError('duplicated per-column baseline reference drift')
            rows = obj['columns'][key]['baselines']
            if len(rows) != len(grid.TAILS):
                raise AssertionError('common tail catalog count drift')
            for index, row in enumerate(rows):
                oracle.verify_record(row)
                if (row['sequence'] != [index] or row['specs'] != []
                        or row['terminals'] != list(grid.TAILS[index])
                        or row['input_sha256'] != column['input_sha256']
                        or row['raw_bytes'] != column['raw_bytes'] or row['kind'] != column['kind']):
                    raise AssertionError('common baseline is not the exact requested tail/input')
        return obj['columns']
    shared = {}
    for column in columns:
        raw = Path(column['input_path']).read_bytes()
        if grid.digest(raw) != column['input_sha256']:
            raise AssertionError('input drift during baseline calibration')
        rows = []
        for index, terminals in enumerate(grid.TAILS):
            row, _ = oracle.measure(raw, column['kind'], [], terminals,
                                    phase='selection', warmups=1, repeats=7)
            rows.append(grid.prior.decorate(row, [index], 'fresh_common_safety_baseline'))
        shared[column['key']] = {'reference': references[column['key']], 'baselines': rows}
    grid.dump(frozen, dict(experiment_identity=identity, frozen_at=grid.now(),
                          reference_origin='historical_exact_constants_not_recalibrated',
                          references=references, columns=shared,
                          baselines_new_in_this_experiment=True,
                          frozen_before_any_new_actor_training=True))
    return shared


def build_selections(column, run_rows, shared, historical, beam_rows):
    """Only this stage admits replay recipes; it runs after both learners finish."""
    selections = []
    for method in ('ppo', 'grpo'):
        candidates = [r for r in run_rows if r['method'] == method]
        winner = grid.prior.select_across_seeds(candidates)
        selections.append(dict(method=method, weight_tenths=list(WEIGHT),
            weight_id=grid.weight_id(WEIGHT), selection='fresh_433_training_best_of_requested_seeds',
            source_seed=winner['seed'], before_benchmark_selection_cost=winner['selected']['cost'],
            source_selection_origin=winner['selected']['selection_origin'],
            seed_metrics=[dict(seed=r['seed'], cost=r['selected']['cost'],
                rho=r['selected']['compression_rate'], E=r['selected']['encode_seconds_median'],
                D=r['selected']['decode_seconds_median']) for r in candidates],
            recipe=winner['selected']))
    baseline = [grid.score(r, shared['reference'], WEIGHT) for r in shared['baselines']]
    for method, count in (('fresh_fixed5_J_best', 5), ('fresh_complete30_J_best', 30),
                          ('common38_J_best', 38)):
        chosen = min(baseline[:count], key=grid.prior.score_rank)
        selections.append(dict(method=method, weight_tenths=list(WEIGHT), weight_id=grid.weight_id(WEIGHT),
            selection='new_common_tail_catalog_measurement_and_J_selection',
            before_benchmark_selection_cost=chosen['cost'], recipe=chosen))
    for old in historical['methods']:
        selections.append(dict(method=old['method'], weight_tenths=list(WEIGHT),
            weight_id=grid.weight_id(WEIGHT), selection='frozen_historical_recipe_replay_NOT_retrained',
            original_source_seed=old.get('source_seed'), recipe=grid.raw_recipe(old)))
    for beam in beam_rows:
        selections.append(dict(method=beam['method'], weight_tenths=list(WEIGHT),
            weight_id=grid.weight_id(WEIGHT), selection='frozen_new_Beam30_recipe_replayed_NOT_reselected',
            source_experiment_identity=beam['experiment_identity'],
            before_benchmark_selection_cost=beam['selected']['cost'], recipe=grid.raw_recipe(beam['selected'])))
    if len({r['method'] for r in selections}) != len(selections):
        raise AssertionError('fresh/replay method IDs collide')
    return selections


def export_csv(path, benchmarks):
    fields = ['dataset', 'column', 'method', 'weight_id', 'raw_bytes', 'bytes', 'compression_rate',
              'encode_seconds_median', 'decode_seconds_median', 'encode_time_normalized',
              'decode_time_normalized', 'cost', 'chain', 'selection', 'source_seed',
              'lossless', 'cold_decode_pass']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for benchmark in benchmarks:
            for row in benchmark['rows']:
                record = {**row, 'dataset': benchmark['column']['dataset'],
                          'column': benchmark['column']['name']}
                writer.writerow({name: record.get(name) for name in fields})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beam-output', type=Path, default=DEFAULT_BEAM)
    parser.add_argument('--keys', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(grid.SEEDS))
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--free-space-floor-gib', type=float, default=15)
    args = parser.parse_args(argv)
    if (args.iterations < 1 or args.episodes < 8 or args.episodes % 8
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or min(args.seeds) < 0 or args.free_space_floor_gib < 0):
        parser.error('positive budgets, episodes multiple of 8, distinct nonnegative seeds required')
    columns, gate = grid.prior.legacy_runner.input_columns(args.keys)
    historical, references, beam_rows, extras = load_sources(columns, args.beam_output.resolve())
    config = {**grid.BASE_CONFIG, 'iterations': args.iterations, 'episodes': args.episodes,
        'seeds': args.seeds, 'weights_tenths': [list(WEIGHT)], 'workers': 1,
        'action_catalog': grid.rp.legacy_tail.catalog(), 'tail_catalog': grid.TAILS,
        'grammar': '17 families; 38 prefix choices; 38 tails; tail-first decision, prefix-then-tail execution',
        'action_count': 77, 'not_30_family_graph_policy': True,
        'reference_source': str(grid.HISTORICAL_BENCHMARK),
        'reference_digest': grid.fingerprint(references),
        'normalization': 'raw rho; E/(E+E0), D/(D+D0); unchanged historical column zlib6 constants',
        'training_per_weight': 'fresh actor, fresh PPO critic and fresh optimizer; no imported checkpoints',
        'cache': 'new raw metric vectors only; no old, Beam or other learner candidates/winners exposed',
        'common_baselines': '38 freshly measured empty-prefix tails; same safety baseline catalog for both methods',
        'training_measurement': '0 warmups + 1 complete-frame encode/decode',
        'selection_measurement': 'frozen own top8 + own greedy + common safety baseline; 1 warmup + 7 repeats',
        'final_measurement': 'freeze selections; 2 warmups + 11 random interleaved repeats; cold frame-only decode',
        'invalid_reward': -5.0, 'invalid_caveat': 'not a lower bound on valid rewards; retain counters',
        'no_heldout_generalization_claim': True, 'all_66_weights_complete': False,
        'beam_source': str(args.beam_output.resolve()), 'history_recipes_final_replay_only': True}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'RUN.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    grid.check_space(output, args.free_space_floor_gib)
    before = source_snapshot(columns, gate, extras)
    identity = grid.fingerprint(dict(schema=SCHEMA, config=config, columns=columns,
        file_sha256=before['file_sha256'], dependencies=before['dependencies']))
    manifest_path = output / 'MANIFEST.json'
    if manifest_path.exists():
        if not args.resume or grid.read_json(manifest_path)['experiment_identity'] != identity:
            raise AssertionError('resume source/protocol/reference/input identity mismatch')
    else:
        if any(path.name != 'RUN.lock' for path in output.iterdir()):
            raise RuntimeError('new output must be empty; never copy runs from another experiment')
        grid.dump(manifest_path, dict(schema=SCHEMA, experiment_identity=identity, config=config,
            columns=columns, provenance_before=before, created_at=grid.now(), command=sys.argv))
    oracle = grid.MeasurementOracle(output / 'measurements.sqlite3', identity,
                                   floor_gib=args.free_space_floor_gib)
    began = time.perf_counter()
    status, error, paths, benchmarks = 'RUNNING', None, [], []
    costs = Counter()
    grid.dump(output / 'STATUS.json', dict(status=status, at=grid.now(),
                                          experiment_identity=identity, phase='common_baselines'))
    try:
        shared = prepare_shared(columns, references, output, oracle, identity, args.resume)
        for column in columns:
            raw = Path(column['input_path']).read_bytes()
            if grid.digest(raw) != column['input_sha256']:
                raise AssertionError('input drift before learner')
            runs = []
            for seed_index, seed in enumerate(args.seeds):
                methods = ('ppo', 'grpo') if (seed_index + column['numeric_index']) % 2 == 0 else ('grpo', 'ppo')
                initial = None
                for method in methods:
                    folder = output / 'runs' / column['key'] / grid.weight_id(WEIGHT) / f'{method}_{seed}'
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    row = grid.run_training(column, raw, references[column['key']],
                        shared[column['key']]['baselines'], method, seed, WEIGHT, config,
                        folder, oracle, identity, args.resume)
                    if initial is not None and initial != row['budget']['initial_actor_sha256']:
                        raise AssertionError('paired learner initial actors differ')
                    initial = row['budget']['initial_actor_sha256']
                    runs.append(row)
                    paths.append(str(folder / 'RESULT.json'))
                    for key in ('train_seconds', 'search_seconds', 'optimizer_seconds', 'refinement_seconds', 'run_seconds'):
                        costs[key] += row[key]
                    progress = dict(status='RUNNING', phase='training', completed_runs=len(paths),
                        expected_runs=len(columns) * len(args.seeds) * 2, column=column['key'],
                        method=method, seed=seed, elapsed_seconds=time.perf_counter()-began,
                        oracle=oracle.totals())
                    grid.dump(output / 'PROGRESS.json', progress)
                    grid.dump(output / 'STATUS.json', progress)
                    grid.prior.emit(**progress)
            selections = build_selections(column, runs, shared[column['key']],
                                          historical[column['key']], beam_rows[column['key']])
            grid.dump(output / 'selections' / (column['key'] + '.json'),
                      dict(experiment_identity=identity, selections=selections, frozen_at=grid.now()))
            benchmarks.append(grid.benchmark_column(column, selections,
                shared[column['key']]['baselines'], references[column['key']],
                output / 'benchmark' / column['key'], identity, args.resume))
        after = source_snapshot(columns, gate, extras)
        grid.prior.legacy_runner.verify_unchanged(before, after)
        complete = (len(columns) == 20 and tuple(args.seeds) == grid.SEEDS
                    and args.iterations == 64 and args.episodes == 32
                    and len(paths) == 120 and len(benchmarks) == 20)
        status = 'COMPLETE' if complete else 'PARTIAL_PROTOCOL_COMPLETE'
        old_invocations = []
        if (output / 'INVOCATIONS.jsonl').exists():
            import json
            old_invocations = [json.loads(line) for line in (output / 'INVOCATIONS.jsonl').read_text().splitlines()]
        grid.dump(output / 'SUMMARY.json', dict(schema=SCHEMA, status=status,
            main_433_protocol_complete=complete, all_66_weights_complete=False,
            experiment_identity=identity, config=config, columns=columns,
            run_result_paths=paths, completed_runs=len(paths), expected_runs=len(columns)*len(args.seeds)*2,
            benchmark_result_paths=[str(output / 'benchmark' / c['key'] / 'RESULT.json') for c in columns],
            aggregates=grid.aggregate_results(benchmarks), oracle=oracle.totals(),
            completed_run_cost_sums=dict(costs), provenance_after=after,
            invocation_wall_seconds=time.perf_counter()-began,
            all_recorded_invocation_wall_seconds=sum(x['invocation_wall_seconds'] for x in old_invocations)+time.perf_counter()-began,
            new_learners=['ppo', 'grpo'], final_replay_Beam=list(BEAM_METHODS),
            historical_methods_preserved_not_retrained=True,
            final_selection_not_changed_by_remeasurement=True,
            training_cost_note='optimizer time is included in training; shared oracle counted once globally; invocation receipts include interrupted attempts'))
        export_csv(output / 'PER_COLUMN.csv', benchmarks)
    except BaseException as exc:
        status = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        grid.dump(output / 'FAILURE.json', dict(status=status, at=grid.now(), error=error))
        raise
    finally:
        receipt = dict(status=status, ended_at=grid.now(), experiment_identity=identity,
            invocation_wall_seconds=time.perf_counter()-began, completed_runs=len(paths),
            expected_runs=len(columns)*len(args.seeds)*2, completed_benchmarks=len(benchmarks),
            oracle=oracle.totals(), error=error)
        with (output / 'INVOCATIONS.jsonl').open('a') as stream:
            stream.write(grid.canonical(receipt)+'\n')
        grid.dump(output / 'STATUS.json', receipt)
        grid.dump(output / 'PROGRESS.json', receipt)
        oracle.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
