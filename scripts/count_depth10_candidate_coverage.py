#!/usr/bin/env python3
"""Read-only depth10 coverage and full-input/baseline inventory; no codecs run.

Prints {data, markdown} as JSON. The caller may persist the generated report.
Unique candidate key is (dataset, numeric_index, tuple(expanded IDs)).
Shared trace is counted once, then each online trace exactly once.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from canonical206_data import load_smallest_two

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histogram(chains):
    return dict(sorted(Counter(len(chain) for chain in chains).items()))


def summarize(rows, atomic):
    logged = {tuple(row['ids']) for row in rows}
    attempted = {tuple(row['ids']) for row in rows if row['status'] != 'budget_exhausted'}
    successful = {tuple(row['ids']) for row in rows if row['status'] == 'ok'}
    budget = [row for row in rows if row['status'] == 'budget_exhausted']
    budget_only = logged - attempted
    budget_partial = {tuple(row['ids']) for row in budget if row['completed_atomic_prefix']}
    assert {row['status'] for row in rows} <= {'ok', 'rejected', 'budget_exhausted'}
    assert all(row.get('lossless') is True for row in rows if row['status'] == 'ok')
    return {
        'option_records': len(rows),
        'option_record_statuses': dict(Counter(row['status'] for row in rows)),
        'cached_complete_records': sum(row.get('cached_complete') is True for row in rows),
        'new_atomic_edges_in_option_records': sum(row.get('new_atomic_edges', 0) for row in rows),
        'unique_logged_candidates': len(logged),
        'unique_attempted_candidates': len(attempted),
        'unique_successful_candidates': len(successful),
        'budget_exhausted_records': len(budget),
        'budget_exhausted_records_with_partial_atomic_prefix': sum(bool(row['completed_atomic_prefix']) for row in budget),
        'budget_exhausted_records_empty_atomic_prefix': sum(not row['completed_atomic_prefix'] for row in budget),
        'budget_exhausted_records_matching_actual_attempt': sum(tuple(row['ids']) in attempted for row in budget),
        'unique_budget_only_candidates': len(budget_only),
        'unique_budget_only_with_partial_atomic_prefix': len(budget_only & budget_partial),
        'unique_budget_only_without_partial_atomic_prefix': len(budget_only - budget_partial),
        'unique_logged_depth10': sum(len(ids) == 10 for ids in logged),
        'unique_attempted_depth10': sum(len(ids) == 10 for ids in attempted),
        'unique_successful_depth10': sum(len(ids) == 10 for ids in successful),
        'unique_logged_depth_histogram': histogram(logged),
        'unique_attempted_depth_histogram': histogram(attempted),
        'unique_successful_depth_histogram': histogram(successful),
        'atomic_records': len(atomic),
        'atomic_record_statuses': dict(Counter(row['status'] for row in atomic)),
        'unique_atomic_prefixes': len({tuple(row['prefix']) for row in atomic}),
    }


def checked_result(path, expected_input_sha):
    row = read(path)
    assert row['input_sha256'] == expected_input_sha and row['lossless'] is True, path
    frame = Path(row['frame_path'])
    if not frame.is_absolute():
        frame = path.parent.parent / frame
    assert frame.stat().st_size == row['compressed_bytes'], frame
    assert digest(frame) == row['frame_sha256'], frame
    return {
        'source_json': str(path), 'method': row.get('method', 'fixed'),
        'seed': row.get('seed'), 'algorithm': row.get('algorithm'),
        'selected_ids': row['selected_ids'],
        'compressed_bytes': row['compressed_bytes'],
        'compression_ratio': row['compression_ratio'],
        'frame_path': str(frame), 'frame_sha256': row['frame_sha256'],
        'compress_sec': row.get('compress_sec'), 'lossless_recorded': True,
        'frame_size_and_sha256_reverified': True,
    }


def audit(run):
    manifest = read(run / 'manifest.json')
    complete = read(run / 'COMPLETE.json')
    assert complete['status'] == 'PASS' and complete['actual_rows'] == complete['expected_rows'] == 40
    current = {(c['dataset'], c['numeric_index']): c for c in load_smallest_two(ROOT)}
    parent = Path(manifest['parent'])
    methods = ('grpo_depth10', 'frozen_depth10')
    columns = []
    all_logged = set()
    all_attempted = set()
    all_successful = set()
    all_shapes = set()
    all_logged_shapes = set()
    sum_fields = Counter()
    phase_totals = {phase: Counter() for phase in ('shared', *methods)}
    for column in manifest['columns']:
        column_id = column['dataset'], column['numeric_index']
        live = current[column_id]
        for field in ('input_sha256', 'source_sha256', 'count', 'kind', 'raw_bytes', 'nan_removed'):
            assert live[field] == column[field], (column_id, field)
        assert hashlib.sha256(live['raw']).hexdigest() == column['input_sha256']
        key = '{}__{:02d}'.format(*column_id)
        shared_path = run / 'shared' / (key + '.json')
        shared = read(shared_path)
        shared_trace_path = run / 'shared' / (key + '.trace.json')
        shared_trace = read(shared_trace_path)
        phase_rows = {'shared': shared_trace['options']}
        phase_atomic = {'shared': shared_trace['atomic']}
        phases = {'shared': summarize(phase_rows['shared'], phase_atomic['shared'])}
        source_traces = [str(shared_trace_path)]
        method_results = {}
        shared_cache_hits = shared['search_stats']['atomic_runtime']['cache_hits']
        atomic_cache_hits = shared_cache_hits
        phases['shared']['atomic_cache_hits'] = shared_cache_hits
        for method in methods:
            stem = '{}__seed{}__{}'.format(method, manifest['config']['seeds'][0], key)
            trace_path = run / 'traces' / (stem + '.json')
            trace = read(trace_path)
            assert trace['shared_snapshot_sha256'] == shared['shared_initial_snapshot_sha256']
            phase_rows[method], phase_atomic[method] = trace['options'], trace['atomic']
            phases[method] = summarize(trace['options'], trace['atomic'])
            source_traces.append(str(trace_path))
            result_path = run / 'columns' / (stem + '.json')
            saved = read(result_path)
            method_results[method] = checked_result(result_path, column['input_sha256'])
            combined = summarize(phase_rows['shared'] + trace['options'], phase_atomic['shared'] + trace['atomic'])
            saved_stats = saved['search_stats']
            for field, source_field in (
                ('unique_logged_depth_histogram', 'candidate_depth_histogram_unique'),
                ('unique_successful_depth_histogram', 'candidate_depth_histogram_verified_unique'),
            ):
                assert combined[field] == {int(k): v for k, v in saved_stats[source_field].items()}, (key, method, field)
            cache_hits = saved_stats['atomic_runtime']['cache_hits'] - shared_cache_hits
            assert cache_hits >= 0
            atomic_cache_hits += cache_hits
            phases[method]['atomic_cache_hits'] = cache_hits
        options = [r for phase in phase_rows.values() for r in phase]
        atomic = [r for phase in phase_atomic.values() for r in phase]
        union = summarize(options, atomic)
        union['atomic_cache_hits'] = atomic_cache_hits
        union['summed_per_phase_unique_logged'] = sum(p['unique_logged_candidates'] for p in phases.values())
        union['cross_phase_duplicate_unique_logged'] = union['summed_per_phase_unique_logged'] - union['unique_logged_candidates']
        union['summed_per_phase_unique_attempts'] = sum(p['unique_attempted_candidates'] for p in phases.values())
        union['cross_phase_duplicate_unique_attempts'] = union['summed_per_phase_unique_attempts'] - union['unique_attempted_candidates']
        logged = {tuple(row['ids']) for row in options}
        attempted = {tuple(row['ids']) for row in options if row['status'] != 'budget_exhausted'}
        successful = {tuple(row['ids']) for row in options if row['status'] == 'ok'}
        all_logged.update((column_id[0], column_id[1], ids) for ids in logged)
        all_attempted.update((column_id[0], column_id[1], ids) for ids in attempted)
        all_successful.update((column_id[0], column_id[1], ids) for ids in successful)
        all_shapes.update(attempted)
        all_logged_shapes.update(logged)
        assert logged - attempted == {tuple(row['ids']) for row in options if row['status'] == 'budget_exhausted'} - attempted
        assert len(logged) == len(attempted) + union['unique_budget_only_candidates']
        budget_evidence = [
            {'phase': phase, 'trace_path': path, 'ids': row['ids'],
             'completed_atomic_prefix': row['completed_atomic_prefix'],
             'new_atomic_edges': row['new_atomic_edges'],
             'also_has_nonbudget_outcome': tuple(row['ids']) in attempted}
            for (phase, records), path in zip(phase_rows.items(), source_traces)
            for row in records if row['status'] == 'budget_exhausted'
        ]
        for field, value in union.items():
            if isinstance(value, int):
                sum_fields[field] += value
        for phase, summary in phases.items():
            for field, value in summary.items():
                if isinstance(value, int):
                    phase_totals[phase][field] += value
        fixed_path = parent / 'fixed_algorithm_baseline_v1/columns' / (key + '.json')
        fixed = checked_result(fixed_path, column['input_sha256'])
        archive_paths = {Path(row['source']) for row in shared['prior_sources']}
        archive_paths.add(fixed_path)
        archive_results = [checked_result(path, column['input_sha256']) for path in sorted(archive_paths)]
        old_pool = archive_results + list(method_results.values())
        best = min(old_pool, key=lambda row: (row['compressed_bytes'], row['selected_ids'], row['source_json']))
        best_ties = [r['source_json'] for r in old_pool if r['compressed_bytes'] == best['compressed_bytes']]
        columns.append({
            'dataset': column['dataset'], 'numeric_index': column['numeric_index'], 'column': column['name'],
            'input': {k: column[k] for k in ('source_path', 'source_index', 'kind', 'count', 'raw_bytes', 'input_sha256', 'source_sha256', 'source_rows', 'nan_removed', 'max_point', 'input_policy')},
            'source_traces': source_traces, 'phases': phases, 'union': union,
            'budget_interruption_evidence': budget_evidence,
            'fixed_algorithm_best': fixed, 'grpo_depth10': method_results['grpo_depth10'],
            'frozen_depth10': method_results['frozen_depth10'],
            'historical_best_before_depth10': min(archive_results, key=lambda r: r['compressed_bytes']),
            'old_algorithm_composition_best_including_history_and_frozen': best,
            'old_best_tied_result_jsons': best_ties,
            'old_archive_results_checked': len(archive_results),
        })
    assert len(columns) == 20
    assert len(all_logged) == sum_fields['unique_logged_candidates']
    assert len(all_attempted) == sum_fields['unique_attempted_candidates']
    assert len(all_successful) == sum_fields['unique_successful_candidates']
    result = {
        'status': 'PASS', 'run': str(run), 'manifest_sha256': digest(run / 'manifest.json'),
        'script_sha256': digest(Path(__file__)), 'codecs_executed': 0,
        'scope': 'Depth10 shared phase counted once plus GRPO/frozen online traces, seed 20260906. Historical winners are checked for baseline but historical search traces are not added to coverage.',
        'candidate_key': '(dataset, numeric_index, tuple(expanded_registered_ids))',
        'logged_definition': 'all option ids in saved traces, including budget-exhausted proposals',
        'attempt_definition': 'unique ids having at least one non-budget-exhausted outcome (ok or rejected). Budget-only target chains are excluded even if part of their prefix executed; those partial steps remain counted in atomic records.',
        'coverage_revision': 'budget-outcome-correction-20260906; earlier 11762 attempts label corrected to unique logged candidates',
        'success_definition': 'option status ok and lossless true; complete framed file verified by previous run, current audit rechecks saved winners sizes and hashes but does not decode or rerun codecs',
        'depth_definition': 'expanded canonical206 registered IDs, not pure primitive count; reference IDs 0..28 contain implicit packing/zlib6',
        'input_loader': str(ROOT / 'scripts/canonical206_data.py') + ':load_smallest_two',
        'input_location_note': 'Complete original CSV files plus deterministic typed loader; no standalone raw input file is needed. Loader returns raw bytes, int64/float64 little-endian, independently SHA-checked here.',
        'config': manifest['config'], 'totals': dict(sum_fields),
        'globally_distinct_id_sequences_ignoring_column': len(all_shapes),
        'globally_distinct_logged_id_sequences_ignoring_column': len(all_logged_shapes),
        'phase_totals': {k: dict(v) for k, v in phase_totals.items()}, 'columns': columns,
    }
    lines = [
        '# 上一轮 Depth10 组合覆盖与输入/基线清单', '',
        '只读审计；本脚本未执行任何 codec。覆盖口径：共享阶段只算一次，再加 GRPO 和 frozen 的在线日志。历史搜索不在本次覆盖分母中；历史最佳文件纳入比较基线。', '',
        f"- 候选记录：{sum_fields['option_records']:,} 条（含重复、缓存与失败）。",
        f"- 按 `(dataset, numeric_index, 完整展开 IDs)` 去重：{len(all_logged):,} 个被记录/提出的候选；其中 {len(all_attempted):,} 个有非预算中断执行结果（ok/rejected），{len(all_successful):,} 个成功完整文件。",
        f"- 旧稿把 {len(all_logged):,} 标成实际尝试，现已纠正。{sum_fields['budget_exhausted_records']} 条 budget_exhausted 记录中，{sum_fields['budget_exhausted_records_matching_actual_attempt']} 条形状在别处已有非预算中断结果，因此仍纳入实际候选。剩余 {sum_fields['unique_budget_only_candidates']} 个仅预算中断的不同目标链排除：{sum_fields['unique_budget_only_without_partial_atomic_prefix']} 个没有推进前缀、{sum_fields['unique_budget_only_with_partial_atomic_prefix']} 个只执行了部分前缀，不能说全部完全未执行。",
        f"- 恰好深度 10：{sum_fields['unique_logged_depth10']:,} 个被记录候选，{sum_fields['unique_attempted_depth10']:,} 个有非预算中断结果，{sum_fields['unique_successful_depth10']:,} 个成功。",
        f"- 完成文件缓存命中记录：{sum_fields['cached_complete_records']:,}；注册 ID 步骤记录：{sum_fields['atomic_records']:,}；步骤缓存命中：{sum_fields['atomic_cache_hits']:,}。这些都不是独立组合数。",
        f"- 跨列也忽略后：记录 {len(all_logged_shapes):,} 种 IDs 形状，非预算中断结果 {len(all_shapes):,} 种；此数不能代替逐列测量覆盖。",
        '- 深度是注册 ID 数，不是纯原语数。旧 0–28 ID 含隐式 compact/zlib6；不能把旧覆盖说成同样数量的纯算子组合。', '',
        '| 数据集 | 列 | 候选记录 | 不同记录候选 | 有执行结果的不同候选 | 成功组合 | 深度10成功 | 固定最佳 B | 历史+depth10最佳 B | GRPO B | frozen B |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for c in columns:
        u = c['union']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(c['dataset'], c['column'], u['option_records'], u['unique_logged_candidates'], u['unique_attempted_candidates'], u['unique_successful_candidates'], u['unique_successful_depth10'], c['fixed_algorithm_best']['compressed_bytes'], c['old_algorithm_composition_best_including_history_and_frozen']['compressed_bytes'], c['grpo_depth10']['compressed_bytes'], c['frozen_depth10']['compressed_bytes']))
    lines += ['', '## 完整输入', '', '`scripts/canonical206_data.py:load_smallest_two(ROOT)` 为原实验输入读取器；直接返回每列原始字节。当前重新读取，20 列 input SHA256 和两个 CSV source SHA256 全部与 frozen manifest 相同。所有行完整使用，没有采样；只按原策略逐列删除 NaN。本轮 20 列删除数量均为 0。', '', '| 数据集 | 列 | dtype | 数值个数 | 原始 B |', '|---|---|---|---:|---:|']
    for c in columns:
        i = c['input']
        lines.append(f"| {c['dataset']} | {c['column']} | {i['kind']} | {i['count']} | {i['raw_bytes']} |")
    lines += ['', '## 计算方法与边界', '',
        '1. 读取 `manifest.json`、`COMPLETE.json`，再遍历每列 `shared/{dataset}__{index:02d}.trace.json` 的 options/atomic。',
        '2. 分别读取 `traces/{grpo_depth10|frozen_depth10}__seed20260906__{dataset}__{index:02d}.json` 的 options/atomic。这些在线日志已经排除共享前缀，不把 columns 中累计直方图直接相加。',
        '3. options 的全部 ids 数组去重为 logged；仅 status!=budget_exhausted 去重为有实际执行结果的候选；status=ok 且 lossless=true 为成功。如果某形状一处预算中断、另一处已有实际结果，仍保留后者。每种方法的 shared+online 的 logged/成功直方图与旧 columns 保存统计逐项核对通过。',
        '4. 预算中断部分执行与完全未推进分开：60条budget记录中11条有非空completed_atomic_prefix；仅budget形状40种中10种部分执行。该部分执行消耗的步骤仍保留在atomic统计，但不将完整目标链作为已完成测量结果。逐条证据在JSON的budget_interruption_evidence。',
        '5. `cached_complete=true` 只计已完成候选缓存命中；atomic cache_hits 用 shared 一次加每方法终值减 shared 初值。atomic 记录是注册 ID 步骤，不能当成完整组合数。',
        '6. 固定 baseline 与 shared.prior_sources 中所有历史候选、当前 GRPO/frozen 胜者逐个复核 frame_path 的字节数和 SHA256；没有重新运行 codec 或把 frozen RH 的 10247B 写作 GRPO 结果。',
        '7. JSON 为逐列精确输入路径、SHA256、完整 IDs、结果 JSON 路径、真实 frame 路径和分阶段统计的机器可读入口。固定 baseline 是已登记算法 recipe 中的最好实测，不是所有算法与参数的理论最优。', '',
        f"来源：`{run}`。", '',
    ]
    return {'data': result, 'markdown': '\n'.join(lines)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'analysis_outputs/depth10_probe_20260906_v1')
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), ensure_ascii=False, indent=2))
