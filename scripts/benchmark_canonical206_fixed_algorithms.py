#!/usr/bin/env python3
"""Full-column fixed registered algorithms, using frozen canonical206 inputs."""
from __future__ import annotations

import os
for _thread_var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_var] = '1'

import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from canonical206_data import load_smallest_two
from canonical206_search_env import Env, SPECS, DESC, VARIANT_DESC, EXCLUDED, make_frame, encode_cold, decode_frame
import primitive_operator_action_runtime as p

from canonical182_policy_paths import (PolicyPath, concrete_path, path_from_names, path_from_record, path_record, require_current_registry)
import primitive_operator_action_runtime as policy_runtime

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = subprocess.run


def bounded_run(*args, **kwargs):
    kwargs.setdefault('timeout', 30)
    return ORIGINAL_RUN(*args, **kwargs)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def assert_frozen(parent, manifest, columns):
    failures = []
    for relative, expected in manifest['source_hashes'].items():
        if sha(ROOT / relative) != expected:
            failures.append(f'source changed: {relative}')
    for name, expected in manifest['runtime_cli_hashes'].items():
        if sha(ROOT / 'cpp/build' / name) != expected:
            failures.append(f'CLI changed: {name}')
    expected_columns = {(c['dataset'], c['name']): c for c in manifest['columns']}
    if set(expected_columns) != {(c['dataset'], c['name']) for c in columns}:
        failures.append('column set changed')
    for column in columns:
        original = expected_columns[(column['dataset'], column['name'])]
        for field in ('input_sha256', 'source_sha256', 'kind', 'count', 'max_point', 'source_index', 'numeric_index'):
            if column[field] != original[field]:
                failures.append(f"column {column['dataset']}/{column['name']} changed: {field}")
        if hashlib.sha256(column['raw']).hexdigest() != original['input_sha256']:
            failures.append('loaded raw hash changed')
        if sha(column['source_path']) != original['source_sha256']:
            failures.append('input file hash changed')
    if failures:
        raise AssertionError('; '.join(failures))
    return {'status': 'PASS', 'python_sources': len(manifest['source_hashes']),
            'cli_binaries': len(manifest['runtime_cli_hashes']), 'columns': len(columns)}


def select_recipes(registry, column):
    """Use the frozen method mapping, not the search grammar or stage aliases."""
    by_name = {row['name']: row['policy_action_id'] for row in registry['policy_actions']}
    recipes = {}
    excluded = []
    logical = 'int' if column['kind'] == 'int64' else 'float'
    for label, mapping in sorted(registry.get('benchmark_86_io_variant_pipelines', registry['benchmark_86_canonical_action_pipelines']).items()):
        variant = 'shared'
        if isinstance(mapping, dict):
            if logical in mapping:
                variant = logical
            elif column['kind'] in mapping:
                variant = column['kind']
            elif logical == 'float' and 'float_max_point_0_to_18' in mapping and 0 <= column['max_point'] <= 18:
                variant = 'float_max_point_0_to_18'
            elif 'bytes' in mapping:
                variant = 'bytes'
            else:
                excluded.append({'label': label, 'status': 'inapplicable', 'reason': 'no exact source-kind variant'})
                continue
            names = mapping[variant]
        else:
            names = mapping
        params = registry.get('benchmark_86_action_parameters', {}).get(label)
        params = params.get(variant) if isinstance(params, dict) else params
        ids = path_from_names(names, params)
        if EXCLUDED.intersection(ids):
            excluded.append({'label': label, 'status': 'excluded', 'reason': 'Camel excluded by original protocol', 'ids': list(ids)})
            continue
        if not ids:
            raise AssertionError(f'empty algorithm recipe: {label}')
        if column['kind'] not in VARIANT_DESC[ids.io_variants[0]].accepted_input_kinds and 'bytes' not in VARIANT_DESC[ids.io_variants[0]].accepted_input_kinds:
            excluded.append({'label': label, 'status': 'inapplicable', 'reason': 'first action rejects typed source and byte view', 'ids': list(ids)})
            continue
        if ids not in recipes:
            recipes[ids] = {**path_record(ids), 'names': [SPECS[i].name for i in ids], 'aliases': [], 'variants': {}}
        recipes[ids]['aliases'].append(label)
        recipes[ids]['variants'][label] = variant
    return sorted(recipes.values(), key=lambda row: (row['aliases'], row['ids'])), excluded


def measure_column(column, recipes, exclusions, output):
    started = time.perf_counter()
    # Direct transition calls only: no Env.legal(), max-depth cut, or search.
    env = Env(column, sample_rows=0, max_steps=max(len(r['ids']) for r in recipes))
    candidates = []
    best_recipe = {**path_record(PolicyPath()), 'names': [], 'aliases': ['RAW'], 'variants': {'RAW': 'raw64'}}
    best_frame = make_frame(column, env.nodes[()])
    assert decode_frame(best_frame) == column['raw']
    best_key = (len(best_frame), best_recipe['aliases'], best_recipe['ids'])
    candidates.append({**best_recipe, 'status': 'ok', 'compressed_bytes': len(best_frame),
                       'compression_ratio': len(column['raw']) / len(best_frame), 'frame_sha256': hashlib.sha256(best_frame).hexdigest(),
                       'lossless': True, 'candidate_wall_sec': 0.0})
    for recipe in recipes:
        candidate_started = time.perf_counter()
        prefix = ()
        try:
            for action, variant, params in zip(recipe['ids'], recipe['io_variants'], recipe['parameters']):
                child = env.transition(prefix, action, io_variant=variant, parameters=params)
                if child is None:
                    raise ValueError(env.records[-1]['error'])
                prefix = child
            frame = make_frame(column, env.nodes[prefix])
            if decode_frame(frame) != column['raw']:
                raise AssertionError('complete fixed-algorithm frame is not bit-exact')
            row = {**recipe, 'status': 'ok', 'compressed_bytes': len(frame),
                   'compression_ratio': len(column['raw']) / len(frame), 'frame_sha256': hashlib.sha256(frame).hexdigest(), 'lossless': True}
            candidate_key = (len(frame), recipe['aliases'], recipe['ids'])
            if candidate_key < best_key:
                best_key, best_recipe, best_frame = candidate_key, recipe, frame
        except Exception as exc:
            row = {**recipe, 'status': 'rejected', 'error': f'{type(exc).__name__}: {exc}'[:1000], 'lossless': False}
        row['candidate_wall_sec'] = time.perf_counter() - candidate_started
        candidates.append(row)
    selection_sec = time.perf_counter() - started
    prefix = path_from_record(best_recipe)
    times, decode_times = [], []
    for repeat in range(6):
        begin = time.perf_counter()
        fresh = encode_cold(column, prefix)
        encode_sec = time.perf_counter() - begin
        begin = time.perf_counter()
        recovered = decode_frame(fresh)
        decode_sec = time.perf_counter() - begin
        if fresh != best_frame or recovered != column['raw']:
            raise AssertionError('independent cold replay differs from cached fixed-chain frame')
        if repeat:
            times.append(encode_sec)
            decode_times.append(decode_sec)
    key = f"{column['dataset']}__{column['numeric_index']:02d}"
    frame_path = f'frames/{key}.c206'
    (output / 'frames').mkdir(exist_ok=True)
    (output / frame_path).write_bytes(best_frame)
    result = {'dataset': column['dataset'], 'column': column['name'], 'numeric_index': column['numeric_index'],
              'source_index': column['source_index'], 'kind': column['kind'], 'values': column['count'],
              'original_bytes': len(column['raw']), 'algorithm': best_recipe['aliases'][0],
              'algorithm_aliases': best_recipe['aliases'], 'selected_ids': best_recipe['ids'],
              'policy_registry_version': policy_runtime.ACTION_REGISTRY_VERSION,
              'selected_io_variants': best_recipe['io_variants'], 'selected_parameters': best_recipe['parameters'], 'selected_pipeline': best_recipe['names'], 'selected_variants': best_recipe['variants'],
              'compressed_bytes': len(best_frame), 'compression_ratio': len(column['raw']) / len(best_frame),
              'compress_sec': statistics.median(times), 'compress_sec_repeats': times,
              'decompress_sec': statistics.median(decode_times), 'decompress_sec_repeats': decode_times,
              'fixed_algorithm_selection_sec': selection_sec, 'input_sha256': column['input_sha256'],
              'source_sha256': column['source_sha256'], 'frame_sha256': hashlib.sha256(best_frame).hexdigest(),
              'frame_path': frame_path, 'lossless': True, 'candidate_unique_chains': len(recipes),
              'candidate_labels': sum(len(r['aliases']) for r in recipes),
              'successful_chains': sum(r['status'] == 'ok' for r in candidates if r['aliases'] != ['RAW']),
              'rejected_chains': sum(r['status'] == 'rejected' for r in candidates),
              'excluded_labels': exclusions, 'search_stats': env.summary(),
              'attempted_action_ids': sorted({r['prefix'][-1] for r in env.records}),
              'successful_action_ids': sorted({r['prefix'][-1] for r in env.records if r['status'] == 'ok'}),
              'candidate_results_path': f'candidates/{key}.json', 'trace_path': f'traces/{key}.json',
              'claim': 'best full framed file among frozen registered algorithm recipes; not an exhaustive parameter or upstream implementation oracle'}
    dump(output / 'columns' / f'{key}.json', result)
    dump(output / 'candidates' / f'{key}.json', candidates)
    dump(output / 'traces' / f'{key}.json', env.records)
    print(json.dumps({k: result[k] for k in ('dataset', 'column', 'algorithm', 'compressed_bytes', 'compression_ratio',
                                           'compress_sec', 'fixed_algorithm_selection_sec', 'successful_chains', 'rejected_chains')}, ensure_ascii=False), flush=True)
    return result


def write_csv(path, results):
    fields = ('dataset', 'column', 'numeric_index', 'source_index', 'kind', 'values', 'original_bytes', 'algorithm',
              'compressed_bytes', 'compression_ratio', 'compress_sec', 'decompress_sec', 'fixed_algorithm_selection_sec',
              'input_sha256', 'source_sha256', 'frame_sha256', 'frame_path', 'lossless', 'candidate_unique_chains',
              'candidate_labels', 'successful_chains', 'rejected_chains')
    rows = [{**{field: r[field] for field in fields}, 'algorithm_aliases': ' | '.join(r['algorithm_aliases']),
             'selected_ids': ' -> '.join(map(str, r['selected_ids'])), 'selected_pipeline': ' -> '.join(r['selected_pipeline']),
             'compress_sec_repeats': json.dumps(r['compress_sec_repeats'])} for r in results]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    parent = args.parent.resolve()
    if not (parent / 'COMPLETE.json').exists():
        raise ValueError('parent experiment must be complete')
    original = json.loads((parent / 'manifest.json').read_text())
    registry = json.loads((parent / 'action_registry.json').read_text())
    require_current_registry(registry)
    if registry['policy_registry_version'] != p.ACTION_REGISTRY_VERSION:
        raise ValueError('registry version changed')
    if [(r['policy_action_id'], r['name']) for r in registry['policy_actions']] != [(i, SPECS[i].name) for i in sorted(SPECS)]:
        raise ValueError('registry IDs/names differ from frozen experiment')
    columns = load_smallest_two()
    frozen_before = assert_frozen(parent, original, columns)
    specs = [(c, *select_recipes(registry, c)) for c in columns]
    scope = {'labels_in_frozen_mapping': len(registry['benchmark_86_canonical_action_pipelines']),
             'per_column': [{'dataset': c['dataset'], 'column': c['name'], 'kind': c['kind'],
                             'eligible_labels': sum(len(r['aliases']) for r in recipes), 'unique_chains': len(recipes),
                             'excluded': excluded} for c, recipes, excluded in specs],
             'total_unique_chain_column_pairs': sum(len(recipes) for _, recipes, _ in specs),
             'distinct_chains_across_kinds': len({path_from_record(r) for _, recipes, _ in specs for r in recipes})}
    print(json.dumps({'event': 'scope', **scope}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    output = parent / 'fixed_algorithm_baseline_v1'
    output.mkdir(exist_ok=False)
    sources = dict(original['source_hashes'])
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    for relative in sources:
        target = output / 'code' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    initial = {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'parent': str(parent),
               'parent_manifest_sha256': sha(parent / 'manifest.json'), 'registry_sha256': sha(parent / 'action_registry.json'),
               'source_hashes': sources, 'runtime_cli_hashes': original['runtime_cli_hashes'],
               'scope': scope, 'frozen_before': frozen_before, 'machine': platform.platform(),
               'input_policy': 'same original full typed numeric columns, exact original bits, no float conversion or rounding',
               'selection': 'minimum complete C206+UAR1 frame bytes; ties sorted algorithm aliases then IDs; RAW fallback',
               'execution': 'one registered full algorithm recipe only; shared full-column prefix cache; no search grammar filter or cross-algorithm composition',
               'timing': 'winner-only independent cold full encode, one warmup and five repeats median; inverse and equality outside encoder timer; C++ CLI and IPC included',
               'candidate_timing': 'selection cost includes shared cached encoding, all inverse checks and full-frame decode; not a per-algorithm encoding latency',
               'limits': '86 method labels are registered project implementations including aliases, proxies, reference stages and residual composite cores; no upstream-native86 claim',
               'timeout_sec_per_subprocess': 30}
    dump(output / 'manifest.json', initial)
    subprocess.run = bounded_run
    results = []
    for column, recipes, exclusions in specs:
        results.append(measure_column(column, recipes, exclusions, output))
        write_csv(output / 'best.csv', results)
    frozen_after = assert_frozen(parent, original, load_smallest_two())
    if sources[str(Path(__file__).resolve().relative_to(ROOT))] != sha(__file__):
        raise AssertionError('baseline runner changed during run')
    if initial['registry_sha256'] != sha(parent / 'action_registry.json') or initial['parent_manifest_sha256'] != sha(parent / 'manifest.json'):
        raise AssertionError('parent registry/manifest changed during baseline')
    summary = {'status': 'PASS', 'columns': len(results), 'all_lossless': all(r['lossless'] for r in results),
               'frozen_before': frozen_before, 'frozen_after': frozen_after,
               'candidate_chain_column_pairs': sum(r['candidate_unique_chains'] for r in results),
               'successful_chain_column_pairs': sum(r['successful_chains'] for r in results),
               'rejected_chain_column_pairs': sum(r['rejected_chains'] for r in results),
               'attempted_action_ids': sorted({i for r in results for i in r['attempted_action_ids']}),
               'successful_action_ids': sorted({i for r in results for i in r['successful_action_ids']}),
               'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    dump(output / 'COMPLETE.json', summary)
    print(json.dumps({'event': 'complete', **summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
