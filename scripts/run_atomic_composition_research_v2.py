#!/usr/bin/env python3
"""Targeted version-2 full-column operator expansion; no RL or hidden codec.

Screening estimates only allocate measurements. Every reported candidate is
actually encoded and decoded as an independent complete file. Historical data
and same-frame whole-codec cascades are separate comparator families.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import subprocess
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'analysis_outputs/primitive_composition_research_20260906_v1'
PURE = ['range0', 'range1', 'sparse0', 'sparse1', 'bit4', 'bit8', 'bit12', 'bit16']
ALGORITHMS = ['zlib6', 'bzip2', 'xz', 'zstd', 'brotli']


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def spec_key(specs):
    return json.dumps(specs, sort_keys=True, separators=(',', ':'))


def column_key(c):
    return f"{c['dataset']}__{c['numeric_index']:02d}"


def op(name, **kw):
    return {'op': name, **kw}


def generate_pool_v1_unused(raw, kind, budget, seed):
    """Domain-generic proposals, never dataset-name-specific parameters."""
    from atomic_transform_runtime_v1 import apply_ops
    rng = random.Random(seed)
    initials = [[]]
    if kind == 'float64':
        for factor in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            s = [op('float_scale', factor=factor)]
            try:
                apply_ops(raw, s)
                initials.append(s)
            except ValueError:
                pass
    predictors = [[]] + [[op('delta', order=order, lag=lag)]
        for order in [1, 2] for lag in [1, 2, 4, 8, 24]] + [
        [op('xor', lag=lag)] for lag in [1, 2, 4, 8, 24]]
    maps = [[], [op('zigzag')], [op('for')], [op('gcd')],
            [op('zigzag'), op('gcd')], [op('for'), op('gcd')],
            [op('zigzag'), op('for')]]
    sparse = [[], [op('rle')], [op('sparsezero')], [op('dictionary')]]
    packers = [[], [op('compact')], [op('bitpack')]]
    reorder = [[], *[[op('shuffle', width=w)] for w in [2, 4, 8]],
        *[[op('bittranspose', width=w)] for w in [1, 2, 4, 8]]]
    seen = set()
    result = []
    def add(s):
        k = spec_key(s)
        if len(s) <= 9 and k not in seen:
            seen.add(k)
            result.append(s)
    # A compact coverage core guarantees simple chains before random proposals.
    for init, pred, mapping, pack in itertools.product(initials, predictors, maps[:4], packers):
        add(init + pred + mapping + pack)
    for init, pred, squeeze, pack in itertools.product(initials, predictors, sparse[1:], packers):
        add(init + pred + squeeze + pack)
        add(init + pred + [op('zigzag')] + squeeze + pack)
    # Mutation-friendly extra numeric stages remain individually reversible.
    actions = [op('zigzag'), op('for'), op('gcd'), op('dictionary'),
        op('rle'), op('sparsezero')] + [s[0] for s in predictors[1:]]
    attempts = 0
    while len(result) < budget and attempts < budget * 20:
        attempts += 1
        initial = rng.choice(initials)
        s = initial + rng.choice(predictors) + rng.choice(maps)
        if rng.random() < .55:
            s += rng.choice(sparse)
        if rng.random() < .35:
            s += [rng.choice(actions)]
        if rng.random() < .15:
            s += [rng.choice(actions)]
        s += rng.choice(packers) + rng.choice(reorder)
        add(s)
    return result, initials


def generate_pool(raw,kind,budget,seed):
    from atomic_transform_extensions_v2 import apply_ops
    from run_atomic_composition_research_v1 import generate_pool as old_pool
    rng=random.Random(seed)
    old,initials=old_pool(raw,kind,min(3000,budget),seed)
    initials=initials.copy()
    if kind=='float64':
        for factor in [1,10,100,1000,10000,100000,1000000]:
            s=[op('float_scale_exceptions',factor=factor,max_fraction=.05)]
            try:apply_ops(raw,s);initials.append(s)
            except ValueError:pass
    seen={spec_key(s) for s in old}
    result=old.copy()
    def add(s):
        k=spec_key(s)
        if len(s)<=9 and k not in seen:seen.add(k);result.append(s)
    predictors=[[]]+[[op('delta',lag=l,order=o)] for l in [1,2,4,8,24] for o in [1,2]]+[[op('xor',lag=l)] for l in [1,2,4,8,24]]
    maps=[[],[op('zigzag')],[op('for')],[op('gcd')],[op('zigzag'),op('gcd')]]
    blocks=[16,32,64,128,256,512,1024]
    reorder=[[],[op('shuffle',width=2)],[op('shuffle',width=4)],[op('bittranspose',width=1)],[op('bittranspose',width=2)]]
    for init,pred,mapping,b in itertools.product(initials,predictors,maps[:3],blocks):
        add(init+pred+mapping+[op('block_bitpack',block_size=b)])
        add(init+pred+mapping+[op('block_for',block_size=b),op('block_bitpack',block_size=b)])
    attempts=0
    while len(result)<budget and attempts<budget*20:
        attempts+=1
        s=rng.choice(initials)+rng.choice(predictors)+rng.choice(maps)
        if rng.random()<.55:s+=[op('block_for',block_size=rng.choice(blocks))]
        if rng.random()<.25:s+=[op(rng.choice(['rle','sparsezero','dictionary']))]
        s += [op('block_bitpack',block_size=rng.choice(blocks))] if rng.random()<.75 else [op(rng.choice(['compact','bitpack']))]
        add(s+rng.choice(reorder))
    return result,initials


def entropy_estimates(body):
    """Full-body heuristics, not measured compressed lengths or result scores."""
    a = np.frombuffer(body, dtype=np.uint8)
    if not len(a):
        return (0., 0.)
    counts = np.bincount(a, minlength=256)
    p = counts[counts > 0]
    h0 = float((len(a) * math.log2(len(a)) - np.sum(p * np.log2(p))) / 8)
    # Penalize populated conditional cells to avoid ranking a huge free model.
    if len(a) < 2:
        return h0, h0
    pairs = np.bincount(a[:-1].astype(np.uint16) * 256 + a[1:], minlength=65536).reshape(256, 256)
    totals = pairs.sum(axis=1)
    nz = pairs[pairs > 0]
    nz_t = totals[totals > 0]
    h1 = float((np.sum(nz_t * np.log2(nz_t)) - np.sum(nz * np.log2(nz))) / 8)
    return h0 + .75 * len(p), h1 + .75 * len(nz)


def measure_column(c, args_dict):
    from atomic_transform_extensions_v2 import apply_ops, reverse_ops
    from atomic_composition_frame_v2 import encode_frame, encode_prepared, decode_frame, make_body, inspect_frame
    args = argparse.Namespace(**args_dict)
    out = Path(args.output)
    key = column_key(c)
    row_path = out / 'columns' / f'{key}.json'
    if row_path.exists():
        raise ValueError('fresh-output-only: existing row cannot be accepted as a replay')
    raw, kind = c['raw'], c['kind']
    start = time.perf_counter()
    log_path = out / 'traces' / f'{key}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # This worker owns this one column. Unfinished runs are not silently reused.
    if log_path.exists():
        raise ValueError(f'incomplete trace exists: {log_path}; use a new output version')
    counters = collections.Counter()
    transformed_hashes = collections.defaultdict(set)
    best = {}
    all_success = []
    seen = set()
    trace = log_path.open('x')
    def evaluate(specs, terminals, family, prepared=None, origin='search'):
        seqkey = spec_key([specs, terminals])
        if seqkey in seen:
            counters['duplicate_proposals_skipped'] += 1
            return None
        seen.add(seqkey)
        t0 = time.perf_counter()
        record = {'family': family, 'specs': specs, 'terminals': terminals,
                  'depth': len(specs) + len(terminals), 'origin': origin}
        counters[family + '_attempts'] += 1
        try:
            frame = (encode_prepared(raw, kind, specs, terminals, *prepared)
                     if prepared is not None else encode_frame(raw, kind, specs, terminals))
            encode_sec = time.perf_counter() - t0
            restored = decode_frame(frame)
            if restored != raw:
                raise AssertionError('full-file lossless comparison failed')
            record.update(status='ok', bytes=len(frame), ratio=len(raw)/len(frame),
                          encode_sec=encode_sec, measured_sec=time.perf_counter()-t0,
                          frame_sha256=sha(frame))
            transformed = prepared[0] if prepared is not None else apply_ops(raw, specs)[0]
            record['transformed_payload_sha256'] = sha(transformed)
            record['transformed_payload_bytes'] = len(transformed)
            transformed_hashes[family].add(record['transformed_payload_sha256'])
            counters[family + '_successes'] += 1
            all_success.append(record)
            if family not in best or (len(frame), record['depth'], seqkey) < (
                best[family]['bytes'], best[family]['depth'], spec_key([best[family]['specs'], best[family]['terminals']])):
                dst = out / 'frames' / f'{key}__{family}.acf2'
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(frame)
                best[family] = {**record, 'frame_path': str(dst)}
        except (ValueError, RuntimeError, OverflowError, subprocess.TimeoutExpired) as exc:
            record.update(status='rejected', error=str(exc), measured_sec=time.perf_counter()-t0)
            counters[family + '_rejected'] += 1
        trace.write(json.dumps(record, separators=(',', ':')) + '\n')
        return record

    # Same frame protocol, not raw codec payload compared against framed result.
    for n in [1, 2]:
        for terms in itertools.product(ALGORITHMS, repeat=n):
            evaluate([], list(terms), 'algorithm', origin='whole_codec_control')
    baseline_sec = time.perf_counter() - start
    pool, initials = generate_pool(raw, kind, args.pool_size, args.seed + c['numeric_index'])
    parent=json.loads((Path(args.parent_run)/'columns'/f'{key}.json').read_text())
    parent_records=[json.loads(line) for line in Path(parent['trace_path']).read_text().splitlines()]
    inherited=[]
    inherited_keys=set()
    for record in sorted((r for r in parent_records if r['status']=='ok' and r['family']=='pure'),key=lambda r:r['bytes']):
        k=spec_key(record['specs'])
        if k not in inherited_keys:
            inherited_keys.add(k);inherited.append(record['specs'])
        if len(inherited)>=100:break
    all_keys={spec_key(s) for s in pool}
    for specs in inherited:
        if spec_key(specs) not in all_keys:pool.append(specs);all_keys.add(spec_key(specs))
    screened = []
    screening_start = time.perf_counter()
    for i, specs in enumerate(pool):
        try:
            payload, meta = apply_ops(raw, specs)
            if len(payload) > len(raw) * 8 + 65536:
                counters['screen_expansion_rejected'] += 1
                continue
            # Metadata is an encoded part of the complete body, not free advice.
            body = make_body(payload, meta)
            if len(body) > len(raw) * 8 + 65536:
                counters['screen_expansion_rejected'] += 1
                continue
            h0, h1 = entropy_estimates(body)
            screened.append({'specs': specs, 'h0': h0, 'h1': h1,
                             'literal': len(body), 'index': i})
            counters['screen_valid'] += 1
        except (ValueError, OverflowError) as exc:
            counters['screen_invalid'] += 1
    screening_sec = time.perf_counter() - screening_start
    # Round-robin rankings plus uniform coverage prevent one proxy dominating.
    rng = random.Random(args.seed + c['numeric_index'])
    shuffled = screened.copy()
    rng.shuffle(shuffled)
    queues = [sorted(screened, key=lambda x:(x[k], x['index'])) for k in ['h0','h1','literal']] + [shuffled]
    selected, chosen = [], set()
    # Parent winners are explicitly remeasured under the NEW bit contexts. They
    # are not claimed as newly discovered transform chains or free prior wins.
    for item in screened:
        if spec_key(item['specs']) in inherited_keys:
            selected.append(item);chosen.add(item['index'])
    selected=selected[:args.representations]
    chosen={x['index'] for x in selected}
    for items in itertools.zip_longest(*queues):
        if len(selected)>=args.representations:
            break
        for item in items:
            if item is not None and item['index'] not in chosen:
                selected.append(item)
                chosen.add(item['index'])
                if len(selected) >= args.representations:
                    break
        if len(selected) >= args.representations:
            break
    dump(out / 'screening' / f'{key}.json', {'scope':'Heuristic allocation only; not measured compression results',
         'pool_count':len(pool), 'valid':len(screened), 'initials':initials,
         'selected':selected, 'screening_sec':screening_sec})
    for j, item in enumerate(selected):
        specs = item['specs']
        prepared = apply_ops(raw, specs)
        # Independently verify the transform stack as well as the final frame.
        if reverse_ops(*prepared) != raw:
            raise AssertionError('transform round trip failed')
        for term in [None] + PURE:
            evaluate(specs, [] if term is None else [term], 'pure', prepared)
        if (j+1) % 100 == 0:
            trace.flush()
            dump(out / 'progress' / f'{key}.json', {'key':key,'selected_done':j+1,
                'selected_total':len(selected),'counters':dict(counters),
                'best':best,'elapsed_sec':time.perf_counter()-start})
    # Mixed controls use genuinely retained transformed data, not reverse-prefix
    # bridges. Kept distinct from the operator-only primary result family.
    ranked_specs = []
    ranked_seen = set()
    for r in sorted((x for x in all_success if x['family']=='pure'), key=lambda x:x['bytes']):
        k = spec_key(r['specs'])
        if k not in ranked_seen:
            ranked_seen.add(k)
            ranked_specs.append(r['specs'])
        if len(ranked_specs) >= args.hybrid_representations:
            break
    for specs in ranked_specs:
        if not specs:
            continue
        prepared = apply_ops(raw, specs)
        for term in ALGORITHMS:
            evaluate(specs, [term], 'hybrid', prepared, origin='matched_terminal_control')
    trace.close()
    prior_all = json.loads((Path(args.prior) / 'prior_coverage.json').read_text())
    prior = next(r for r in prior_all['columns'] if r['dataset']==c['dataset'] and r['numeric_index']==c['numeric_index'])
    old = prior['old_algorithm_composition_best_including_history_and_frozen']
    # Cold encoding repeats do not use the prepared-state cache and are not
    # included in the number of different measured candidates.
    for family, winner in best.items():
        repeats = []
        for _ in range(args.repeats + 1):
            t0 = time.perf_counter()
            fresh = encode_frame(raw, kind, winner['specs'], winner['terminals'])
            repeats.append(time.perf_counter()-t0)
            if sha(fresh) != winner['frame_sha256'] or decode_frame(fresh) != raw:
                raise AssertionError('cold replay differs')
        winner['compress_sec_repeats'] = repeats[1:]
        winner['compress_sec'] = statistics.median(repeats[1:])
        winner['frame_inspection'] = inspect_frame(fresh)
        winner['saved_vs_historical_bytes'] = old['compressed_bytes'] - winner['bytes']
        winner['saved_vs_same_frame_algorithm_bytes'] = best['algorithm']['bytes'] - winner['bytes']
    row = {k:v for k,v in c.items() if k!='raw'}
    row.update(status='PASS', key=key, registry='atomic-composition-v2',
        historical_best=old, best=best, counts=dict(counters), distinct_measured_candidates=len(seen),
        screen_pool=len(pool), screening_sec=screening_sec, baseline_sec=baseline_sec,
        compose_sec=time.perf_counter()-start, trace_path=str(log_path),
        max_depth=10, rl_updates=0, scope='full-column offline search, no RL, one seed',
        input_sha256=sha(raw))
    row['unique_transformed_payloads_measured']={k:len(v) for k,v in transformed_hashes.items()}
    row['parent_round1_pure_bytes']=parent['best']['pure']['bytes']
    row['parent_round1_algorithm_bytes']=parent['best']['algorithm']['bytes']
    row['parent_transform_chains_remeasured']=len(inherited)
    dump(row_path, row)
    print(json.dumps({'event':'column_complete','key':key,'counts':dict(counters),
        'old':old['compressed_bytes'], 'best':{k:v['bytes'] for k,v in best.items()},
        'seconds':row['compose_sec']}), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(DEFAULT / 'round2'))
    p.add_argument('--parent-run',default=str(DEFAULT/'round1'))
    p.add_argument('--prior', default=str(DEFAULT))
    p.add_argument('--pool-size', type=int, default=12000)
    p.add_argument('--representations', type=int, default=800)
    p.add_argument('--hybrid-representations', type=int, default=80)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20260906)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--keys', nargs='*')
    p.add_argument('--smoke', action='store_true', help='explicitly permit a subset; never label it full20')
    p.add_argument('--targeted',action='store_true',help='full-column targeted extension, not a new full20 run')
    args = p.parse_args()
    if min(args.pool_size,args.representations,args.workers,args.repeats) < 1:
        raise ValueError('positive budgets required')
    from canonical206_data import load_smallest_two
    columns = load_smallest_two(ROOT)
    original = json.loads((ROOT/'analysis_outputs/depth10_probe_20260906_v1/manifest.json').read_text())
    expected = {column_key(c):c for c in original['columns']}
    if {column_key(c) for c in columns} != set(expected) or len(expected)!=20:
        raise AssertionError('full source column key set differs from frozen 20 columns')
    for c in columns:
        e=expected[column_key(c)]
        if any(c[k]!=e[k] for k in ['input_sha256','source_sha256','kind','count','raw_bytes']):
            raise AssertionError('original input drift')
    if args.keys:
        if not (args.smoke or args.targeted):
            raise ValueError('--keys requires --smoke or --targeted; full20 runs cannot silently omit columns')
        columns=[c for c in columns if column_key(c) in args.keys]
        if len(columns)!=len(set(args.keys)):
            raise ValueError('unknown/duplicate key')
    sources = ['scripts/run_atomic_composition_research_v1.py', 'scripts/run_atomic_composition_research_v2.py', 'scripts/atomic_transform_runtime_v1.py',
        'scripts/atomic_composition_frame_v1.py', 'cpp/src/primitive_entropy_v1.cpp',
        'cpp/build/primitive_entropy_v1/primitive_entropy_v1', 'scripts/canonical206_data.py',
        'scripts/atomic_transform_extensions_v2.py','scripts/atomic_composition_frame_v2.py',
        'cpp/src/primitive_binary_entropy_v2.cpp','cpp/build/primitive_binary_entropy_v2/primitive_binary_entropy_v2']
    hashes={s:sha((ROOT/s).read_bytes()) for s in sources}
    prior_path=(Path(args.prior)/'prior_coverage.json').resolve()
    prior_hash=sha(prior_path.read_bytes())
    prior_data=json.loads(prior_path.read_text())
    parent_path=Path(args.parent_run).resolve()
    parent_hashes={str(f):sha(f.read_bytes()) for c in columns for f in
        [parent_path/'columns'/f'{column_key(c)}.json',parent_path/'traces'/f'{column_key(c)}.jsonl']}
    if {column_key(c) for c in prior_data['columns']}!=set(expected):
        raise AssertionError('prior comparator must cover exact frozen 20-column set')
    legacy_manifest=json.loads((ROOT/'analysis_outputs/canonical206_beam29_ppo_grpo_smallest2_20260906_v1/manifest.json').read_text())
    legacy_hashes={**legacy_manifest['source_hashes'],
        **{'cpp/build/'+k:v for k,v in legacy_manifest['runtime_cli_hashes'].items()}}
    for s,h in legacy_hashes.items():
        if sha((ROOT/s).read_bytes())!=h:
            raise AssertionError('legacy executable/source drift: '+s)
    out=Path(args.output)
    manifest_path=out/'manifest.json'
    config=vars(args)
    if manifest_path.exists():
        raise ValueError('fresh-output-only: use a new output version instead of trusting existing results')
    else:
        dump(manifest_path, {'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'source_hashes':hashes,'config':config,'columns':[{k:v for k,v in c.items() if k!='raw'} for c in columns],
            'prior_path':str(prior_path),'prior_sha256':prior_hash,'legacy_hashes':legacy_hashes,
            'scope':'smoke_subset' if args.smoke else ('targeted_extension' if args.targeted else 'full20_offline_search'),
            'parent_run':str(parent_path),'parent_hashes':parent_hashes,
            'controls':'same ACF2 framing for codec baselines, pure operators, and hybrid',
            'pure_terminals':PURE,'whole_algorithms':ALGORITHMS,'rl_updates':0})
        for s in sources:
            dst=out/'code'/s
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/s,dst)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows=list(executor.map(measure_column,columns,itertools.repeat(config)))
    if any(sha((ROOT/s).read_bytes())!=h for s,h in hashes.items()):
        raise AssertionError('executed code changed during experiment')
    if sha(prior_path.read_bytes())!=prior_hash:
        raise AssertionError('historical comparator changed during experiment')
    if any(sha(Path(s).read_bytes())!=h for s,h in parent_hashes.items()):
        raise AssertionError('parent measured result changed during extension')
    for s,h in legacy_hashes.items():
        if sha((ROOT/s).read_bytes())!=h:
            raise AssertionError('legacy source/CLI changed during experiment')
    check=load_smallest_two(ROOT)
    if any(c['input_sha256']!=expected[column_key(c)]['input_sha256'] for c in check):
        raise AssertionError('input changed during experiment')
    if {column_key(c) for c in check}!=set(expected):
        raise AssertionError('column set changed during experiment')
    if not (args.smoke or args.targeted) and ({r['key'] for r in rows}!=set(expected) or len(rows)!=20):
        raise AssertionError('formal experiment is missing one or more of the 20 columns')
    dump(out/'COMPLETE.json', {'status':'PASS','columns':len(rows),'source_hashes_unchanged':True,
        'scope':'smoke_subset' if args.smoke else ('targeted_extension' if args.targeted else 'full20_offline_search'),
        'full20_coverage':not (args.smoke or args.targeted) and len(rows)==20,'prior_unchanged':True,
        'legacy_sources_and_clis_unchanged':True,
        'inputs_unchanged':True,'all_winners_cold_replayed':True,
        'distinct_measured_candidates':sum(r['distinct_measured_candidates'] for r in rows),
        'pure_wins_vs_historical':sum(r['best']['pure']['saved_vs_historical_bytes']>0 for r in rows),
        'pure_wins_vs_same_frame_algorithms':sum(r['best']['pure']['saved_vs_same_frame_algorithm_bytes']>0 for r in rows)})


if __name__=='__main__':
    main()
