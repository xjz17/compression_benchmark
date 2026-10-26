#!/usr/bin/env python3
"""Bounded full-column recipe GRPO and matched frozen-policy control."""
from __future__ import annotations

import os
for _thread in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread] = '1'

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

import numpy as np
from canonical206_data import load_smallest_two
from canonical206_search_env import SPECS, encode_cold, decode_frame
from canonical206_recipe_env import RecipeEnv, build_catalog, BudgetExceeded
from canonical206_recipe_grpo import initialize_policy, train, save_policy, _collect
from benchmark_canonical206_fixed_algorithms import assert_frozen

from canonical182_policy_paths import (PolicyPath, concrete_path, path_from_names, path_from_record, path_record, require_current_registry)
import primitive_operator_action_runtime as policy_runtime

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = subprocess.run
CALL_TIMEOUT = 30.0
CLI_STATS = {'calls': 0, 'timeouts': 0, 'wall_sec': 0.0}


def bounded_run(*args, **kwargs):
    started = time.perf_counter()
    CLI_STATS['calls'] += 1
    kwargs.setdefault('timeout', CALL_TIMEOUT)
    try:
        return ORIGINAL_RUN(*args, **kwargs)
    except subprocess.TimeoutExpired:
        CLI_STATS['timeouts'] += 1
        raise
    finally:
        CLI_STATS['wall_sec'] += time.perf_counter() - started


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def key(column):
    return f"{column['dataset']}__{column['numeric_index']:02d}"


def stats_delta(before):
    return {k: CLI_STATS[k] - before[k] for k in CLI_STATS}


def run_column(column, baseline, options, method, seed, args):
    global CALL_TIMEOUT
    before = dict(CLI_STATS)
    start = time.perf_counter()
    CALL_TIMEOUT = 30.0
    previous_candidates=json.loads((args.parent/'fixed_algorithm_baseline_v1/candidates'/(key(column)+'.json')).read_text())
    byte_options={o.ids:o.option_id for o in options if 'bytes' in o.accepted_input_kinds}
    general_labels={'Bzip2','Brotli','XZ','Zstd','LZ4','Snappy','Blosc2','Deflate','Zlib','Gzip'}
    byte_candidates=[r for r in previous_candidates if r['status']=='ok'
                     and path_from_record(r) in byte_options and general_labels.intersection(r['aliases'])]
    backend=min(byte_candidates,key=lambda r:(r['compressed_bytes'],r['ids']))
    env = RecipeEnv(column, options, baseline['compressed_bytes'], max_steps=2,
                    max_unique_edges=args.max_edges,
                    auto_complete_option=byte_options[path_from_record(backend)],
                    complete_on_transition=True)
    seeded = env.seed_fixed_recipes([baseline])
    if seeded[0]['frame_sha256'] != baseline['frame_sha256']:
        raise AssertionError('baseline full file differs from frozen result')
    actor = initialize_policy(env, seed, hidden=64)
    # Same explicit prior for GRPO and its no-update control. This is not SFT.
    # A modest logit prior keeps broad exploration; archive protection is separate.
    baseline_option = env.option_path_for_ids(path_from_record(baseline))[0]
    actor.params[-1][baseline_option] += 1.0
    initial_params = [p.copy() for p in actor.params]
    seed_sec = time.perf_counter() - start
    seed_cli = stats_delta(before)
    search_cli_before = dict(CLI_STATS)
    search_start = time.perf_counter()
    env.deadline = search_start + args.max_seconds
    CALL_TIMEOUT = args.search_timeout
    trace = []
    stop_reason = 'rollout_limit'
    completed_episodes = 0

    def completed_iteration(_actor, row):
        trace.append(dict(row))
        print(json.dumps({'event':'iteration','dataset':column['dataset'],'column':column['name'],
                          'method':method,'seed':seed,'iteration':len(trace),
                          'best_bytes':env.best_bytes,'new_edges':len(env.base.records)-env.search_budget_origin},
                         ensure_ascii=False), flush=True)

    try:
        if method == 'grpo_recipe':
            actor, _returned_trace = train(env, seed, actor=actor, iterations=args.iterations,
                                          episodes=8, group_size=8, epochs=4, lr=.003,
                                          entropy_coefficient=.03,
                                          on_iteration=completed_iteration)
            completed_episodes = len(trace) * 8
        else:
            rng = np.random.default_rng(seed + 1_000_003)
            for _ in range(args.iterations * 8):
                _collect(env, actor, 1, rng)
                completed_episodes += 1
                if completed_episodes % 8 == 0:
                    trace.append({'iteration':completed_episodes//8, 'episodes':8,
                                  'best_bytes':env.best_bytes, 'optimizer_steps':0})
    except BudgetExceeded as exc:
        stop_reason = str(exc)
        if method == 'grpo_recipe':
            completed_episodes = len(trace) * 8
    search_sec = time.perf_counter() - search_start
    search_cli = stats_delta(search_cli_before)
    changed = any(not np.array_equal(a, b) for a, b in zip(actor.params, initial_params))
    if method == 'frozen_recipe' and changed:
        raise AssertionError('frozen-policy control changed its weights')
    option_path = env.best_prefix
    selected_ids = env.expanded(option_path)
    selected_frame = env.frame(option_path)
    if len(selected_frame) > baseline['compressed_bytes']:
        raise AssertionError('incumbent guard regressed')
    CALL_TIMEOUT = 30.0
    encode_times, decode_times = [], []
    for repeat in range(args.repeats + 1):
        t = time.perf_counter()
        fresh = encode_cold(column, selected_ids)
        encode_sec = time.perf_counter() - t
        t = time.perf_counter()
        recovered = decode_frame(fresh)
        decode_sec = time.perf_counter() - t
        if fresh != selected_frame or recovered != column['raw']:
            raise AssertionError('independent complete-column replay mismatch')
        if repeat:
            encode_times.append(encode_sec)
            decode_times.append(decode_sec)
    stem = f'{method}__seed{seed}__{key(column)}'
    frame_path = args.output / 'frames' / (stem + '.c206')
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(selected_frame)
    policy_path = args.output / 'policies' / (stem + '.npz')
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    save_policy(actor, policy_path)
    summary = env.summary()
    gain = 1 - len(selected_frame) / baseline['compressed_bytes']
    row = {
        'dataset':column['dataset'], 'column':column['name'], 'numeric_index':column['numeric_index'],
        'method':method, 'seed':seed, 'original_bytes':len(column['raw']),
        'fixed_algorithm':baseline['algorithm'], 'fixed_bytes':baseline['compressed_bytes'],
        'fixed_ratio':baseline['compression_ratio'], 'fixed_compress_sec':baseline['compress_sec'],
        'fixed_ids':baseline['selected_ids'], 'fixed_io_variants':baseline['selected_io_variants'], 'fixed_parameters':baseline['selected_parameters'],
        'fixed_prior_selection_sec':baseline['fixed_algorithm_selection_sec'],
        'compressed_bytes':len(selected_frame), 'compression_ratio':len(column['raw'])/len(selected_frame),
        'byte_saving_vs_fixed':gain, 'strict_win_vs_fixed':gain>0,
        'strict_new_composition_win':summary['strict_new_win'],
        'policy_registry_version':policy_runtime.ACTION_REGISTRY_VERSION,
        'selected_io_variants':list(selected_ids.io_variants), 'selected_parameters':list(selected_ids.parameters), 'selected_ids':list(selected_ids), 'selected_option_ids':list(option_path),
        'selected_pipeline':[SPECS[i].name for i in selected_ids],
        'selected_provenance':summary['best_first_provenance'],
        'selected_is_registered_fixed_recipe':summary['best_is_registered_fixed_recipe'],
        'compress_sec':statistics.median(encode_times), 'compress_sec_repeats':encode_times,
        'decompress_sec':statistics.median(decode_times), 'seed_verification_sec':seed_sec,
        'search_train_sec':search_sec, 'compose_sec':seed_sec+search_sec,
        'whole_job_sec':time.perf_counter()-start,
        'seed_cli':seed_cli, 'search_cli':search_cli, 'all_cli':stats_delta(before),
        'stop_reason':stop_reason, 'completed_rollouts_in_finished_batches':(completed_episodes//8)*8,
        'completed_rollouts_lower_bound':completed_episodes,
        'fixed_backend_completion_ids':backend['ids'], 'fixed_backend_completion_io_variants':backend['io_variants'], 'fixed_backend_completion_label':backend['aliases'][0],
        'completed_update_iterations':len(trace) if method=='grpo_recipe' else 0,
        'weights_changed':changed, 'search_stats':summary,
        'input_sha256':column['input_sha256'], 'frame_sha256':sha(frame_path),
        'frame_path':str(frame_path.relative_to(args.output)), 'lossless':True,
        'claim':'full-column online search, not held-out generalization; given previously measured fixed winner; current-version complete C206+UAR1 frames; baselines remeasured under the same registry',
    }
    dump(args.output/'columns'/(stem+'.json'),row)
    dump(args.output/'traces'/(stem+'.json'),{'updates':trace,'options':env.records,'atomic':env.base.records})
    print(json.dumps({'event':'column_complete',**{k:row[k] for k in
        ('dataset','column','method','seed','compressed_bytes','compression_ratio','byte_saving_vs_fixed',
         'selected_ids','completed_update_iterations','compose_sec','stop_reason')}},ensure_ascii=False),flush=True)
    return row


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(output, parent):
    rows=[json.loads(p.read_text()) for p in sorted((output/'columns').glob('*.json'))]
    if not rows:
        return
    old=[json.loads(p.read_text()) for p in (parent/'columns').glob('*.json')]
    best=[]
    for dataset,idx in sorted({(r['dataset'],r['numeric_index']) for r in rows}):
        rr=[r for r in rows if (r['dataset'],r['numeric_index'])==(dataset,idx)]
        first=rr[0]
        result={'dataset':dataset,'column':first['column'],'numeric_index':idx,
                'fixed_algorithm':first['fixed_algorithm'],'fixed_bytes':first['fixed_bytes'],'fixed_ratio':first['fixed_ratio']}
        for method in ('grpo_recipe','frozen_recipe'):
            pool=[r for r in rr if r['method']==method]
            if pool:
                win=min(pool,key=lambda r:(r['compressed_bytes'],r['seed']))
                for field in ('compressed_bytes','compression_ratio','selected_ids','seed','byte_saving_vs_fixed',
                              'compress_sec','compose_sec','completed_update_iterations','weights_changed'):
                    result[f'{method}_{field}']=win[field]
                result[f'{method}_seeds']=len(pool)
        for method in ('beam29','ppo','grpo'):
            pool=[r for r in old if r['dataset']==dataset and r['numeric_index']==idx and r['method']==method]
            result['old_'+method+'_bytes']=min(r['compressed_bytes'] for r in pool)
        best.append(result)
    dump(output/'best_by_column.json',best)
    # Stable columns even while one method has not completed a row.
    fields=list(dict.fromkeys(k for r in best for k in r))
    csv_write(output/'best_by_column.csv',[{k:r.get(k,'') for k in fields} for r in best])
    totals=[]
    for dataset,method,seed in sorted({(r['dataset'],r['method'],r['seed']) for r in rows}):
        rr=[r for r in rows if (r['dataset'],r['method'],r['seed'])==(dataset,method,seed)]
        raw=sum(r['original_bytes'] for r in rr);size=sum(r['compressed_bytes'] for r in rr)
        baseline=sum(r['fixed_bytes'] for r in rr)
        totals.append({'dataset':dataset,'method':method,'seed':seed,'columns':len(rr),
                       'raw_bytes':raw,'fixed_bytes':baseline,'compressed_bytes':size,'compression_ratio':raw/size,
                       'byte_saving_vs_fixed':1-size/baseline,'strict_win_columns':sum(r['strict_win_vs_fixed'] for r in rr),
                       'new_composition_win_columns':sum(r['strict_new_composition_win'] for r in rr),
                       'compress_sec':sum(r['compress_sec'] for r in rr),
                       'compose_sec':sum(r['compose_sec'] for r in rr),
                       'updated_columns':sum(r['weights_changed'] for r in rr),
                       'unique_edges':sum(r['search_stats']['search_unique_atomic_edges'] for r in rr)})
    csv_write(output/'dataset_seed_results.csv',totals)
    lines=['# 完整链 GRPO 在线组合结果','',
           '本页随已完成结果更新；未完成不能解读为全覆盖。固定基线与旧实验不改。',
           '最终保留已复核固定胜者，保证不退步；严格提升按完整文件字节计算。',
           '两方法具有相同初始策略和基线先验；frozen_recipe 不做参数更新。',
           '两方法每成功提出一段都会额外试接先前已知最好的通用固定后端（若合法）；这项确定性 completion 计入预算，不冒称 GRPO 独自生成。',
           '学习奖励按本条探索路径实际测到的最小完整文件计算，后续无效尝试另减 0.02；不能读取其他轨迹的最佳值作为奖励。最终选优只看纯文件字节。',
           '本轮在目标完整列上在线学习，不与原跨数据集泛化实验混为一谈。','',
           '| 数据集 | 数据列 | 固定算法 CR | 原 GRPO CR | 新 GRPO CR | 不学习对照 CR | 新 GRPO 节省字节 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in best:
        source=next(q for q in rows if q['dataset']==r['dataset'] and q['numeric_index']==r['numeric_index'])
        new=r.get('grpo_recipe_compression_ratio');control=r.get('frozen_recipe_compression_ratio')
        gain=r.get('grpo_recipe_byte_saving_vs_fixed')
        lines.append(f"| {r['dataset']} | {r['column']} | {r['fixed_ratio']:.3f} | {source['original_bytes']/r['old_grpo_bytes']:.3f} | {new:.3f} | {control:.3f} | {gain:.2%} |" if new is not None and control is not None else f"| {r['dataset']} | {r['column']} | {r['fixed_ratio']:.3f} | — | 待齐 | 待齐 | — |")
    lines+=['','## 口径','',
            '- CR = 原始 typed 数值字节 / 完整 C206+UAR1 文件；越大越好。',
            '- 每列新结果为已完成 seeds 中最小文件；对应 seed、链、时间都在 CSV/JSON，不能把跨 seeds 的不同指标拼到一个结果。',
            f'- 完整算法作为搜索选项，并非新增 codec；所有选项展开后的{len(SPECS)}项算法ID及每步I/O变体、参数见 option_catalog.json。',
            '- 基线复核、在线策略更新、真实 codec 调用与逆校验都计入 compose_sec；此前固定算法全筛选成本单列 fixed_prior_selection_sec。',
            '- 两方法相同新增原子尝试上限与搜索时间上限，但可能因命中缓存、轨迹截断或墙钟先到而实际调用数不同；actual CLI、atomic edges 与时间都公开，不能当成严格等调用结果。',
            '- 每个胜者独立重新编码及完整解码。超时是本轮资源限制，不是判定 codec 本质不可用。',
            '- 保底提升和候选设计不等于 GRPO 学习贡献；必须与不学习对照比较。',
            '- GRPO 组内优势只归一化一次，按每条轨迹等权；原方法参见 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    dump(output/'progress.json',{'rows':len(rows),'columns':len(best),'dataset_seed_results':totals,
                                'all_lossless':all(r['lossless'] for r in rows),
                                'all_non_regressing':all(r['compressed_bytes']<=r['fixed_bytes'] for r in rows)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20260906,20260907,20260908])
    parser.add_argument('--max-edges',type=int,default=384)
    parser.add_argument('--max-seconds',type=float,default=30)
    parser.add_argument('--search-timeout',type=float,default=3)
    parser.add_argument('--iterations',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--column-keys',nargs='+')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();args.parent=args.parent.resolve();args.output=args.output.resolve()
    original=json.loads((args.parent/'manifest.json').read_text())
    registry=json.loads((args.parent/'action_registry.json').read_text())
    require_current_registry(registry)
    columns=load_smallest_two()
    frozen=assert_frozen(args.parent,original,columns)
    if args.column_keys:
        columns=[c for c in columns if key(c) in args.column_keys]
        if set(args.column_keys)!={key(c) for c in columns}:raise ValueError('unknown column key')
    options=build_catalog(registry)
    print(json.dumps({'event':'preflight','frozen':frozen,'columns':len(columns),'options':len(options)},ensure_ascii=False),flush=True)
    if args.dry_run:return
    source_names=['run_canonical206_recipe_search.py','canonical206_recipe_env.py','canonical206_recipe_grpo.py']
    source_names += ['canonical182_policy_paths.py','canonical182_policy_checkpoint.py','canonical182_registry.py','canonical206_search_env.py','canonical156_registry.py','canonical156_parameter_policy.py','native_lz_parameters.py','native_window_parameters.py','native_action_parameters.py','canonical143_registry.py','canonical140_registry.py','max_value_bitpack.py','canonical137_registry.py','minimum_for.py']
    source_names += ['canonical129_registry.py','canonical127_registry.py','optimal_rle_actions.py','current_action_parameters.py','block_state_parameters.py']
    sources={f'scripts/{n}':sha(ROOT/'scripts'/n) for n in source_names}
    for filename in ('canonical182.json', 'canonical156.json', 'native_lz_defaults.json','native_window_defaults.json','canonical143.json','canonical140.json','canonical137.json','canonical129.json','canonical127.json'):
        sources['benchmark/reference/actions/' + filename] = sha(ROOT/'benchmark/reference/actions'/filename)
    for filename in ('optimal_rle.cpp','optimal_rle.hpp','unified_rle_numeric.cpp','unified_rle_numeric.hpp'):
        sources['cpp/src/' + filename] = sha(ROOT/'cpp/src'/filename)
    config={k:v for k,v in vars(args).items() if k not in ('parent','output','resume','dry_run')}
    if args.output.exists() and not args.resume:raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['source_hashes']!=sources or previous['config']!=config:raise ValueError('resume source/config mismatch')
    else:
        for name in source_names:
            target=args.output/'code'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts'/name,target)
        dump(manifest_path,{'created':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'parent':str(args.parent),
                            'parent_manifest_sha256':sha(args.parent/'manifest.json'),
                            'registry_sha256':sha(args.parent/'action_registry.json'),
                            'source_hashes':sources,'config':config,'frozen_before':frozen,
                            'machine':platform.platform(),'numpy':np.__version__,
                            'columns':[{k:v for k,v in c.items() if k!='raw'} for c in columns],
                            'prior':'same frozen/GRPO MLP initialization; fixed winner logit + 1; external incumbent guard is separate',
                            'scope':'online full-column adaptation; not cross-dataset evaluation',
                            'completion':'after each successful option try one predeclared best successful general byte backend if legal; same for both methods, fully budgeted',
                            'objective':'best complete file actually measured along this trajectory; invalid suffix penalty .02; beta0 GRPO; at most 2 recipe decisions, expanded atomic length may exceed old7',
                            'baseline_cost':'prior fixed recipe oracle known to both methods, its selection cost reported separately'})
        dump(args.output/'option_catalog.json',[o.to_dict() for o in options])
    subprocess.run=bounded_run
    for seed in args.seeds:
        for column in columns:
            baseline=json.loads((args.parent/'fixed_algorithm_baseline_v1/columns'/(key(column)+'.json')).read_text())
            for method in ('grpo_recipe','frozen_recipe'):
                target=args.output/'columns'/f'{method}__seed{seed}__{key(column)}.json'
                if not target.exists():run_column(column,baseline,options,method,seed,args)
                summarize(args.output,args.parent)
    frozen_after=assert_frozen(args.parent,original,load_smallest_two())
    if sources!={k:sha(ROOT/k) for k in sources}:raise AssertionError('new source changed during run')
    expected=len(columns)*len(args.seeds)*2
    actual=len(list((args.output/'columns').glob('*.json')))
    if expected!=actual:raise AssertionError('result row count differs')
    dump(args.output/'COMPLETE.json',{'status':'PASS','expected_rows':expected,'actual_rows':actual,
                                     'frozen_after':frozen_after,'new_sources_unchanged':True})
    print(json.dumps({'event':'complete','rows':actual,'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
