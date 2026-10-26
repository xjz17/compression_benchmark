#!/usr/bin/env python3
"""Full-column depth-10 coverage probe and paired online GRPO/frozen search.

The current registry and complete C206/UAR1 frames define the experiment.
Depth counts registered IDs, not hidden primitives in compound reference APIs.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import statistics
import subprocess
import time

import numpy as np

import run_canonical206_recipe_search as common
from canonical206_data import load_smallest_two
from canonical206_depth10_env import Depth10Env, build_catalog
from canonical206_recipe_env import BudgetExceeded
from canonical206_recipe_grpo import initialize_policy, train, _collect, save_policy
from canonical206_search_env import SPECS, ENTROPY, EXCLUDED, encode_cold, decode_frame, is_reference

from canonical182_policy_paths import (PolicyPath, concrete_path, path_from_names, path_from_record, path_record, require_current_registry, same_stages)
import primitive_operator_action_runtime as policy_runtime

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('grpo_depth10', 'frozen_depth10')
GENERAL = {'Bzip2', 'Brotli', 'XZ', 'Zstd', 'LZ4', 'Snappy', 'Blosc2', 'Deflate', 'Zlib', 'Gzip'}


def dump(path, value):
    common.dump(path, value)


def sha(path):
    return common.sha(path)


def key(column):
    return common.key(column)


def log(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False), flush=True)


def stats():
    return dict(common.CLI_STATS)


def delta(before):
    return {k: common.CLI_STATS[k] - before[k] for k in before}


def prior_rows(column, args):
    fixed_path = args.parent / 'fixed_algorithm_baseline_v1/columns' / (key(column) + '.json')
    fixed = json.loads(fixed_path.read_text())
    pool = [{**fixed, 'source': str(fixed_path), 'source_label': 'fixed', 'source_sha256': sha(fixed_path)}]
    for run in (args.parent, args.previous):
        for path in sorted((run / 'columns').glob('*.json')):
            row = json.loads(path.read_text())
            if (row['dataset'], row['numeric_index']) == (column['dataset'], column['numeric_index']):
                pool.append({**row, 'source': str(path),
                             'source_label': f"{run.name}:{row['method']}:seed{row['seed']}",
                             'source_sha256': sha(path)})
    for row in pool:
        require_current_registry(row)
        if row['input_sha256'] != column['input_sha256'] or not row['lossless']:
            raise AssertionError('prior data mismatch / non-lossless row')
    pool = [r for r in pool if 0 < len(r['selected_ids']) <= args.max_depth
            and not EXCLUDED.intersection(r['selected_ids'])
            and len(set(r['selected_ids'])) == len(r['selected_ids'])]
    if not pool:
        raise ValueError('no in-scope known result')
    pool.sort(key=lambda r: (r['compressed_bytes'], r['selected_ids'], r['source']))
    unique = {}
    for row in pool:
        unique.setdefault(path_from_record(row), row)
    # Known top-four plus the fixed winner. This is shared prior knowledge,
    # never counted as a newly discovered compression improvement.
    seed_rows = list(unique.values())[:4]
    if path_from_record(fixed) not in {path_from_record(r) for r in seed_rows}:
        seed_rows.append({**fixed, 'source': str(fixed_path), 'source_label': 'fixed', 'source_sha256': sha(fixed_path)})
    return fixed, pool[0], seed_rows, pool


def apply_chain(env, ids, provenance):
    """Execute an exact candidate using existing options; no synthetic shortcut."""
    target = concrete_path(ids)
    prefix = ()
    offset = 0
    while offset < len(target):
        candidates = [env.options[a] for a in env.legal(prefix) if a != env.stop
                      and same_stages(target[offset:offset + len(env.options[a].ids)], env.options[a].ids)]
        if not candidates:
            return None
        option = min(candidates, key=lambda o: (-len(o.ids), o.option_id))
        parameters = next((q for q in target.parameters[offset:offset + len(option.ids)] if q), {})
        following = env.transition(prefix, option.option_id, parameters=parameters, provenance=provenance)
        if following is None:
            return None
        prefix = following
        offset += len(option.ids)
    if env.expanded(prefix) != target:
        raise AssertionError('candidate path changed')
    return prefix


def deep_witness_candidates(env, column, args):
    """Predeclared recipe combinations at depth 9/10, diversified by backend.

    These are coverage witnesses, not claims of likely compression gains. They
    must pass the same actual packet grammar, budget and full-file verification.
    """
    rows = json.loads((args.parent / 'fixed_algorithm_baseline_v1/candidates' / (key(column) + '.json')).read_text())
    by_ids = {o.ids: o for o in env.options}
    outer_rows = [r for r in rows if r['status'] == 'ok' and path_from_record(r) in by_ids
                  and 'bytes' in by_ids[path_from_record(r)].accepted_input_kinds
                  and GENERAL.intersection(r['aliases'])]
    outer_rows.sort(key=lambda r: (r['compressed_bytes'], r['ids']))
    outer = []
    for row in outer_rows:
        ids = path_from_record(row)
        if ids not in outer:
            outer.append(ids)
    outer = outer[:8]
    roots = [o.ids for o in [o for o in env.options if len(o.ids) > 1 or o.labels]
             if (column['kind'] in o.accepted_input_kinds or 'bytes' in o.accepted_input_kinds)
             and 1 <= len(o.ids) <= 6]
    roots.sort(key=lambda ids: (SPECS[ids[0]].name.endswith('reference_v1'), -len(ids), ids))
    tails = [concrete_path((i,)) for i in sorted(ENTROPY)] + [ids for ids in outer if len(ids) <= 4]
    buckets = defaultdict(list)
    seen = set()
    for backend_index, backend in enumerate(outer):
        for first in roots:
            for last in [()] + tails:
                ids = first + backend + last
                if len(ids) not in (9, 10) or len(ids) > args.max_depth or len(set(ids)) != len(ids) or ids in seen:
                    continue
                seen.add(ids)
                buckets[(len(ids), backend_index)].append(ids)
    output = []
    # Round-robin length and backend, not one large Bzip2-first block.
    while any(buckets.values()):
        for bucket in sorted(buckets):
            if buckets[bucket]:
                output.append(buckets[bucket].pop(0))
    return output


def shared_probe(column, fixed, prior_best, seed_rows, all_prior, options, args):
    common.CALL_TIMEOUT = 30.0
    before = stats()
    started = time.perf_counter()
    env = Depth10Env(column, options, prior_best['compressed_bytes'], max_atomic_depth=args.max_depth,
                     max_unique_edges=args.probe_edges)
    seeded = env.seed_known_chains(seed_rows)
    if env.best_bytes != prior_best['compressed_bytes']:
        raise AssertionError('known archive replay did not retain its exact best file')
    seed_sec = time.perf_counter() - started
    seed_cli = delta(before)
    search_before = stats()
    began = time.perf_counter()
    env.deadline = began + args.probe_seconds
    common.CALL_TIMEOUT = args.search_timeout
    witness = []
    successful_depths = Counter()
    stop_reason = 'candidate_frontier_exhausted'
    try:
        for ids in deep_witness_candidates(env, column, args):
            if time.perf_counter() >= began + args.probe_seconds * .6:
                break
            if successful_depths[len(ids)] >= 6:
                continue
            result = apply_chain(env, ids, 'shared_depth_witness')
            witness.append({**path_record(ids), 'depth': len(ids), 'success': result is not None,
                            'compressed_bytes': len(env.frame(result)) if result is not None else None})
            if result is not None:
                successful_depths[len(ids)] += 1
            if all(successful_depths[d] >= 6 for d in (9, 10)):
                break
        # A small depth-stratified beam explores alternatives to the handpicked
        # witnesses. Each round attempts one child at each available depth.
        tried = set()
        while True:
            buckets = defaultdict(dict)
            for prefix in env.nodes:
                ids = env.expanded(prefix)
                buckets[len(ids)].setdefault(ids, prefix)
            made_attempt = False
            for depth in sorted(buckets, reverse=True):
                if depth >= args.max_depth:
                    continue
                frontier = sorted(buckets[depth].values(),
                                  key=lambda p: (len(env.frame(p)), env.expanded(p)))[:args.frontier_width]
                for prefix in frontier:
                    choices = [a for a in env.legal(prefix) if a != env.stop
                               and (env.expanded(prefix), a) not in tried]
                    if not choices:
                        continue
                    choices.sort(key=lambda a: (not bool(env.options[a].labels), -len(env.options[a].ids), a))
                    action = choices[0]
                    tried.add((env.expanded(prefix), action))
                    env.transition(prefix, action, provenance='shared_depth_frontier')
                    made_attempt = True
                    break
            if not made_attempt:
                break
    except BudgetExceeded as exc:
        stop_reason = str(exc)
    probe_sec = time.perf_counter() - began
    env.deadline = None
    info = {
        'dataset': column['dataset'], 'column': column['name'], 'numeric_index': column['numeric_index'],
        'input_sha256': column['input_sha256'], 'original_bytes': len(column['raw']),
        'fixed_algorithm': fixed['algorithm'], 'fixed_bytes': fixed['compressed_bytes'],
        'prior_best_bytes': prior_best['compressed_bytes'], 'prior_best_source': prior_best['source'],
        'prior_sources': [{k: row.get(k) for k in ('source', 'source_sha256', 'source_label', 'selected_ids', 'selected_io_variants', 'selected_parameters', 'policy_registry_version', 'compressed_bytes', 'frame_sha256')} for row in all_prior],
        'seeded': seeded, 'prior_verification_sec': seed_sec, 'shared_probe_sec': probe_sec,
        'seed_cli': seed_cli, 'probe_cli': delta(search_before), 'stop_reason': stop_reason,
        'shared_best_bytes': env.best_bytes, 'shared_best_ids': list(env.expanded(env.best_prefix)),
        'shared_best_io_variants': list(env.expanded(env.best_prefix).io_variants), 'shared_best_parameters': list(env.expanded(env.best_prefix).parameters),
        'policy_registry_version': policy_runtime.ACTION_REGISTRY_VERSION,
        'shared_best_provenance': env.best_provenance, 'depth_witness_attempts': witness,
        'search_stats': env.summary(),
    }
    snapshot_path = args.output / 'shared' / (key(column) + '.pickle')
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(pickle.dumps(env, protocol=5))
    info['snapshot_path'] = str(snapshot_path.relative_to(args.output))
    info['shared_initial_snapshot_sha256'] = sha(snapshot_path)
    dump(args.output / 'shared' / (key(column) + '.json'), info)
    dump(args.output / 'shared' / (key(column) + '.trace.json'), {'options': env.records, 'atomic': env.base.records})
    log('shared_complete', dataset=column['dataset'], column=column['name'], prior_bytes=prior_best['compressed_bytes'],
        best_bytes=env.best_bytes, depth_witness_successes=dict(successful_depths), seconds=probe_sec)
    return env, info


def run_method(column, fixed, prior_best, shared_env, shared, method, seed, args):
    env = copy.deepcopy(shared_env)
    env.max_unique_edges = args.max_edges
    env.search_budget_origin = len(env.base.records)
    option_origin = len(env.records)
    atomic_origin = len(env.base.records)
    actor = initialize_policy(env, seed, hidden=64)
    if env.best_prefix:
        actor.params[-1][env.best_prefix[0]] += 1.0
    initial_params = [value.copy() for value in actor.params]
    initial_snapshot_sha = shared['shared_initial_snapshot_sha256']
    common.CALL_TIMEOUT = args.search_timeout
    before = stats()
    started = time.perf_counter()
    env.deadline = started + args.max_seconds
    updates = []
    episodes = 0
    stop_reason = 'iteration_limit'

    def updated(_actor, row):
        updates.append(dict(row))
        log('iteration', dataset=column['dataset'], column=column['name'], method=method, seed=seed,
            iteration=len(updates), best_bytes=env.best_bytes)

    try:
        if method == 'grpo_depth10':
            actor, _trace = train(env, seed, actor=actor, iterations=args.iterations,
                                  episodes=4, group_size=4, epochs=4, lr=.003,
                                  entropy_coefficient=.03, on_iteration=updated)
            episodes = len(updates) * 4
        else:
            rng = np.random.default_rng(seed + 1_000_003)
            for _ in range(args.iterations * 4):
                _collect(env, actor, 1, rng)
                episodes += 1
    except BudgetExceeded as exc:
        stop_reason = str(exc)
        if method == 'grpo_depth10':
            episodes = len(updates) * 4
    search_sec = time.perf_counter() - started
    search_cli = delta(before)
    changed = any(not np.array_equal(a, b) for a, b in zip(actor.params, initial_params))
    if method == 'frozen_depth10' and changed:
        raise AssertionError('frozen weights changed')
    selected = env.expanded(env.best_prefix)
    frame = env.frame(env.best_prefix)
    if len(selected) > args.max_depth or len(frame) > prior_best['compressed_bytes']:
        raise AssertionError('depth or known-incumbent protection violation')
    common.CALL_TIMEOUT = 30.0
    encode_times, decode_times = [], []
    for repeat in range(args.repeats + 1):
        t = time.perf_counter()
        fresh = encode_cold(column, selected)
        encode_sec = time.perf_counter() - t
        t = time.perf_counter()
        recovered = decode_frame(fresh)
        decode_sec = time.perf_counter() - t
        if fresh != frame or recovered != column['raw']:
            raise AssertionError('independent full-column replay mismatch')
        if repeat:
            encode_times.append(encode_sec)
            decode_times.append(decode_sec)
    stem = f'{method}__seed{seed}__{key(column)}'
    frame_path = args.output / 'frames' / (stem + '.c206')
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(frame)
    policy_path = args.output / 'policies' / (stem + '.npz')
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    save_policy(actor, policy_path)
    summary = env.summary()
    online_records = env.records[option_origin:]
    online_depths = Counter(len(r['ids']) for r in online_records)
    online_ok = Counter(len(r['ids']) for r in online_records if r['status'] == 'ok')
    row = {
        'dataset': column['dataset'], 'column': column['name'], 'numeric_index': column['numeric_index'],
        'seed': seed, 'method': method, 'original_bytes': len(column['raw']), 'input_sha256': column['input_sha256'],
        'fixed_algorithm': fixed['algorithm'], 'fixed_bytes': fixed['compressed_bytes'], 'fixed_ratio': fixed['compression_ratio'],
        'prior_best_bytes': prior_best['compressed_bytes'], 'prior_best_source': prior_best['source'],
        'shared_best_bytes': shared['shared_best_bytes'], 'compressed_bytes': len(frame), 'compression_ratio': len(column['raw']) / len(frame),
        'bytes_saved_vs_fixed': fixed['compressed_bytes'] - len(frame),
        'bytes_saved_vs_prior': prior_best['compressed_bytes'] - len(frame),
        'bytes_saved_vs_shared': shared['shared_best_bytes'] - len(frame),
        'policy_registry_version': policy_runtime.ACTION_REGISTRY_VERSION,
        'selected_io_variants': list(selected.io_variants), 'selected_parameters': list(selected.parameters), 'selected_ids': list(selected), 'selected_pipeline': [SPECS[i].name for i in selected],
        'selected_option_ids': list(env.best_prefix), 'selected_provenance': env.best_provenance,
        'selected_registered_depth': len(selected), 'depth_limit': args.max_depth,
        'reference_hidden_backend': {'present': bool(selected and is_reference(selected[:1])), 'description': 'initial reference group includes compact-width packing + zlib6; registered IDs are not bare primitives'},
        'compress_sec': statistics.median(encode_times), 'compress_sec_repeats': encode_times,
        'decompress_sec': statistics.median(decode_times),
        'prior_verification_sec': shared['prior_verification_sec'], 'shared_probe_sec': shared['shared_probe_sec'],
        'search_train_sec': search_sec,
        'compose_sec': shared['prior_verification_sec'] + shared['shared_probe_sec'] + search_sec,
        'search_cli': search_cli, 'search_stats': summary,
        'online_option_depth_attempts': dict(sorted(online_depths.items())),
        'online_option_depth_ok': dict(sorted(online_ok.items())),
        'online_unique_atomic_edges': len(env.base.records) - atomic_origin,
        'completed_update_iterations': len(updates), 'completed_rollouts_lower_bound': episodes,
        'weights_changed': changed, 'stop_reason': stop_reason,
        'shared_initial_snapshot_sha256': initial_snapshot_sha,
        'frame_sha256': sha(frame_path), 'frame_path': str(frame_path.relative_to(args.output)), 'lossless': True,
    }
    dump(args.output / 'columns' / (stem + '.json'), row)
    dump(args.output / 'traces' / (stem + '.json'), {'updates': updates, 'options': online_records,
         'atomic': env.base.records[atomic_origin:], 'shared_snapshot_sha256': initial_snapshot_sha})
    log('column_complete', **{k: row[k] for k in ('dataset', 'column', 'method', 'seed', 'compressed_bytes',
        'compression_ratio', 'bytes_saved_vs_prior', 'bytes_saved_vs_shared', 'selected_ids', 'completed_update_iterations')})


def progress(args):
    rows = [json.loads(p.read_text()) for p in (args.output / 'columns').glob('*.json')]
    dump(args.output / 'progress.json', {'rows': len(rows), 'method_rows': dict(Counter(r['method'] for r in rows)),
         'all_lossless': all(r['lossless'] for r in rows), 'all_depth_at_most_10': all(len(r['selected_ids']) <= args.max_depth for r in rows),
         'strict_new_rows': sum(r['bytes_saved_vs_prior'] > 0 for r in rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260906])
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--probe-seconds', type=float, default=20)
    parser.add_argument('--probe-edges', type=int, default=512)
    parser.add_argument('--max-seconds', type=float, default=40)
    parser.add_argument('--max-edges', type=int, default=1024)
    parser.add_argument('--search-timeout', type=float, default=.5)
    parser.add_argument('--frontier-width', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--column-keys', nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for field in ('parent', 'previous', 'output'):
        setattr(args, field, getattr(args, field).resolve())
    if not 1 <= args.max_depth <= 10 or min(args.probe_seconds, args.max_seconds, args.search_timeout, args.repeats) <= 0:
        raise ValueError('depth must be 1..10 and budgets positive')
    original = json.loads((args.parent / 'manifest.json').read_text())
    previous_complete = json.loads((args.previous / 'COMPLETE.json').read_text())
    if previous_complete['status'] != 'PASS':
        raise ValueError('previous run not complete')
    registry = json.loads((args.parent / 'action_registry.json').read_text())
    require_current_registry(registry)
    all_columns = load_smallest_two()
    frozen = common.assert_frozen(args.parent, original, all_columns)
    columns = all_columns
    if args.column_keys:
        columns = [c for c in columns if key(c) in args.column_keys]
        if set(args.column_keys) != {key(c) for c in columns}:
            raise ValueError('unknown column key')
    options = build_catalog(registry)
    log('preflight', frozen=frozen, columns=len(columns), options=len(options), max_registered_ids=args.max_depth)
    if args.dry_run:
        return
    source_names = ['run_canonical206_depth10.py', 'canonical206_depth10_env.py', 'canonical206_recipe_env.py',
                    'canonical206_recipe_grpo.py', 'run_canonical206_recipe_search.py']
    source_names += ['canonical182_policy_paths.py','canonical182_policy_checkpoint.py','canonical182_registry.py','canonical206_search_env.py','canonical156_registry.py','canonical156_parameter_policy.py','native_lz_parameters.py','native_window_parameters.py','native_action_parameters.py','canonical143_registry.py','canonical140_registry.py','max_value_bitpack.py','canonical137_registry.py','minimum_for.py']
    source_names += ['canonical129_registry.py','canonical127_registry.py','optimal_rle_actions.py','current_action_parameters.py','block_state_parameters.py']
    sources = {f'scripts/{n}': sha(ROOT / 'scripts' / n) for n in source_names}
    for filename in ('canonical182.json', 'canonical156.json', 'native_lz_defaults.json','native_window_defaults.json','canonical143.json','canonical140.json','canonical137.json','canonical129.json','canonical127.json'):
        sources['benchmark/reference/actions/' + filename] = sha(ROOT/'benchmark/reference/actions'/filename)
    for filename in ('optimal_rle.cpp','optimal_rle.hpp','unified_rle_numeric.cpp','unified_rle_numeric.hpp'):
        sources['cpp/src/' + filename] = sha(ROOT/'cpp/src'/filename)
    config = {k: v for k, v in vars(args).items() if k not in ('parent', 'previous', 'output', 'resume', 'dry_run')}
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['source_hashes'] != sources or previous['config'] != config:
            raise ValueError('resume source/config mismatch')
        if (previous['parent'] != str(args.parent) or previous['previous'] != str(args.previous)
                or previous['parent_manifest_sha256'] != sha(args.parent / 'manifest.json')
                or previous['previous_manifest_sha256'] != sha(args.previous / 'manifest.json')
                or previous['registry_sha256'] != sha(args.parent / 'action_registry.json')):
            raise ValueError('resume prior/registry provenance mismatch')
    else:
        for name in source_names:
            target = args.output / 'code' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT / 'scripts' / name, target)
        dump(manifest_path, {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'parent': str(args.parent),
            'previous': str(args.previous), 'parent_manifest_sha256': sha(args.parent / 'manifest.json'),
            'previous_manifest_sha256': sha(args.previous / 'manifest.json'), 'registry_sha256': sha(args.parent / 'action_registry.json'),
            'source_hashes': sources, 'config': config, 'frozen_before': frozen,
            'columns': [{k: v for k, v in c.items() if k != 'raw'} for c in columns],
            'machine': platform.platform(), 'numpy': np.__version__,
            'methods': list(METHODS), 'depth_definition': 'maximum 10 expanded current algorithm IDs with explicit I/O variants; reference APIs contain compact + zlib6',
            'shared_phase': 'one shared, budgeted depth-stratified coverage probe per column; identical cached environment snapshot for both policies',
            'prior': 'top-four distinct known chains plus fixed winner, from original Beam29/PPO/GRPO and v3 GRPO/frozen; separately verified, never new discovery',
            'online': 'GRPO groups of 4, beta0, per-column online updates; identical initialized frozen-policy control; no single forced Bzip2 completion',
            'claim_boundary': 'bounded one-seed coverage exploration, not exhaustive, not an RL superiority or stability claim'})
        dump(args.output / 'option_catalog.json', [o.to_dict() for o in options])
    subprocess.run = common.bounded_run
    for column in columns:
        fixed, prior_best, seed_rows, all_prior = prior_rows(column, args)
        shared_path = args.output / 'shared' / (key(column) + '.json')
        if shared_path.exists():
            shared = json.loads(shared_path.read_text())
            for source in shared['prior_sources']:
                if sha(Path(source['source'])) != source['source_sha256']:
                    raise AssertionError('archived result changed after shared probe')
            if (shared['input_sha256'] != column['input_sha256']
                    or shared['prior_best_bytes'] != prior_best['compressed_bytes']):
                raise AssertionError('shared checkpoint no longer matches input/prior')
            snapshot_path = args.output / shared['snapshot_path']
            if sha(snapshot_path) != shared['shared_initial_snapshot_sha256']:
                raise AssertionError('shared snapshot hash mismatch')
            # Only this run's self-generated, hash-checked checkpoint is read.
            shared_env = pickle.loads(snapshot_path.read_bytes())
        else:
            shared_env, shared = shared_probe(column, fixed, prior_best, seed_rows, all_prior, options, args)
        for seed in args.seeds:
            for method in METHODS:
                target = args.output / 'columns' / f'{method}__seed{seed}__{key(column)}.json'
                if not target.exists():
                    run_method(column, fixed, prior_best, shared_env, shared, method, seed, args)
                progress(args)
    after = common.assert_frozen(args.parent, original, load_smallest_two())
    if sources != {k: sha(ROOT / k) for k in sources}:
        raise AssertionError('new execution source changed')
    actual = len(list((args.output / 'columns').glob('*.json')))
    expected = len(columns) * len(args.seeds) * len(METHODS)
    if actual != expected:
        raise AssertionError('coverage mismatch')
    dump(args.output / 'COMPLETE.json', {'status': 'PASS', 'actual_rows': actual, 'expected_rows': expected,
         'frozen_after': after, 'new_sources_unchanged': True})
    log('complete', rows=actual, output=str(args.output))


if __name__ == '__main__':
    main()
