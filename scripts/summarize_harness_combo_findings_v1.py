#!/usr/bin/env python3
"""Add wall-time components and realized validation guards to audited pilot."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--previous-root', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    report = read(root / 'report/REPORT.json')
    assert report['status'] == 'complete_verified'
    lock = read(root / 'results/POLICY_LOCK.json')
    chosen = lock['selected_policy']
    names = list(dict.fromkeys(['fixed_zstd3', 'fixed_lz4', 'prior_top1', chosen]))
    catalog = {x['id']: x for x in read(root / 'results/CATALOG.json')['candidates']}
    data = defaultdict(dict)
    zero_cpu = Counter()
    zero_wall = Counter()
    stable = {'checked': 0, 'frame_sha_differences': [], 'byte_differences': []}
    for key, c in report['columns'].items():
        matrix = read(root / 'results/matrix' / key / 'COLUMN.json')
        items = {r['candidate_id']: r for r in matrix['results'] if r['status'] == 'ok' and r['strict']}
        for r in items.values():
            for part in ['encode', 'decode']:
                zero_cpu[part] += sum(t == 0 for t in r[part]['cpu_samples_ns'])
                zero_wall[part] += sum(t == 0 for t in r[part]['wall_samples_ns'])
        if args.previous_root:
            previous = read(args.previous_root / 'results/matrix' / key / 'COLUMN.json')
            for r in previous['results']:
                if r['status'] != 'ok' or not r['strict']:
                    continue
                current = items[r['candidate_id']]
                stable['checked'] += 1
                if current['frame_sha256'] != r['frame_sha256']:
                    stable['frame_sha_differences'].append([key, r['candidate_id']])
                if current['compressed_bytes'] != r['compressed_bytes']:
                    stable['byte_differences'].append([key, r['candidate_id']])
        if c['stage'] != 'locked_validation':
            continue
        for name in names:
            online = read(root / 'results/online' / key / name / 'RESULT.json')
            r = items[online['selected_id']]
            assert online['selected']['compressed_bytes'] == r['compressed_bytes']
            data[name][key] = dict(column=c, online=online, measurement=r)

    summary = {}
    for name, rows in data.items():
        groups = defaultdict(lambda: [0, 0])
        total = Counter()
        ids = Counter()
        for key, x in rows.items():
            c, o, m = x['column'], x['online'], x['measurement']
            total['compressed_bytes'] += m['compressed_bytes']
            total['online_total_cpu_ns'] += o['online_total_cpu_ns']
            total['online_total_wall_ns'] += o['online_total_wall_ns']
            total['probe_cpu_ns'] += o.get('probe_all', {}).get('cpu_ns', 0)
            for part in ['encode', 'decode']:
                total[part + '_warm_median_cpu_ns_sum'] += m[part]['cpu_ns']
                total[part + '_warm_median_wall_ns_sum'] += m[part]['wall_ns']
            groups[c['source_group']][0] += m['compressed_bytes']
            groups[c['source_group']][1] += data['fixed_zstd3'][key]['measurement']['compressed_bytes']
            ids[o['selected_id']] += 1
        total = dict(total)
        total['selected_ids'] = dict(ids)
        total['worst_source_bytes_ratio_to_zstd3'] = max(a / b for a, b in groups.values())
        summary[name] = total
    baseline = summary['fixed_zstd3']
    selected = summary[chosen]
    selected['online_cpu_ratio_to_zstd3'] = selected['online_total_cpu_ns'] / baseline['online_total_cpu_ns']
    selected['realized_cpu_guard_passed'] = selected['online_cpu_ratio_to_zstd3'] <= 3.0
    selected['realized_risk_vs_training_cap_passed'] = selected['worst_source_bytes_ratio_to_zstd3'] <= lock['worst_source_cap_relative_to_prior_top1']
    selected['realized_risk_vs_validation_prior_passed'] = selected['worst_source_bytes_ratio_to_zstd3'] <= summary['prior_top1']['worst_source_bytes_ratio_to_zstd3']
    oracle = report['matrix']['by_stage']['locked_validation']['oracle_totals']
    selected['byte_reduction_vs_zstd3_pct'] = 100 * (1 - selected['compressed_bytes'] / baseline['compressed_bytes'])
    selected['byte_reduction_vs_prior_pct'] = 100 * (1 - selected['compressed_bytes'] / summary['prior_top1']['compressed_bytes'])
    selected['byte_gap_to_same_pool_oracle_pct'] = 100 * (selected['compressed_bytes'] / oracle['full_pool_bytes'] - 1)
    examples = []
    for key, c in report['columns'].items():
        if c['stage'] != 'locked_validation' or not c['new_12_incremental_bytes']:
            continue
        old, best = c['old_control_18_oracle'], c['full_pool_oracle']
        examples.append({'dataset': c['dataset_id'], 'column': c['column'],
                         'old_id': old['candidate_id'], 'old_bytes': old['compressed_bytes'],
                         'best_id': best['candidate_id'], 'best_bytes': best['compressed_bytes'],
                         'incremental_bytes': c['new_12_incremental_bytes'],
                         'reduction_vs_old_pct': 100 * (1 - best['compressed_bytes'] / old['compressed_bytes']),
                         'online_selected_id': data[chosen][key]['online']['selected_id'],
                         'new_recipe_selected': catalog[data[chosen][key]['online']['selected_id']]['new_combination']})
    examples.sort(key=lambda x: -x['incremental_bytes'])
    result = {'status': 'verified_supplement', 'selected_policy': chosen, 'validation_columns': 9,
              'validation_methods': summary, 'same_pool_oracle': oracle,
              'new12_pool_byte_reduction_vs_old18_pct': 100 * (1 - oracle['full_pool_bytes'] / oracle['old_control_18_bytes']),
              'new_recipe_examples_retrospective': examples,
              'cpu_timer_zero_samples': dict(zero_cpu), 'wall_timer_zero_samples': dict(zero_wall),
              'timing_caveat': 'Small-call CPU samples sometimes quantize to zero/near milliseconds; zero is not zero execution cost. Prefer warmed wall-time components; one-shot online totals are exploratory shared-host measurements.',
              'previous_run_frame_stability': stable if args.previous_root else None,
              'no_confirm_used': True, 'not_fresh_external_validation': True}
    out = root / 'report/FINDINGS.json'
    assert not out.exists()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 组合发现、实际约束与时延补充', '',
             '以下为预留 development 9 列；时延是相同候选在矩阵中三次热测量的中位数之和。在线总时间另列，包含试压和独立验证。', '',
             '| 方法 | 完整帧 B | 热压缩 wall ms | 热解压 wall ms | 在线总 wall ms | 在线总 CPU ms |',
             '|---|---:|---:|---:|---:|---:|']
    for name in names:
        v = summary[name]
        lines.append(f"| {name} | {v['compressed_bytes']} | {v['encode_warm_median_wall_ns_sum']/1e6:.3f} | {v['decode_warm_median_wall_ns_sum']/1e6:.3f} | {v['online_total_wall_ns']/1e6:.3f} | {v['online_total_cpu_ns']/1e6:.3f} |")
    lines += ['', f"冻结策略相对 Zstd3 少 {selected['byte_reduction_vs_zstd3_pct']:.4f}% 字节，相对先验少 {selected['byte_reduction_vs_prior_pct']:.4f}%；距同池 oracle 多 {selected['byte_gap_to_same_pool_oracle_pct']:.4f}%。",
              f"验证总 CPU 为 Zstd3 的 {selected['online_cpu_ratio_to_zstd3']:.4f} 倍；3倍守卫通过={selected['realized_cpu_guard_passed']}。",
              f"验证最差来源字节/Zstd3={selected['worst_source_bytes_ratio_to_zstd3']:.6f}，训练冻结上界={lock['worst_source_cap_relative_to_prior_top1']:.6f}；通过={selected['realized_risk_vs_training_cap_passed']}；相对验证prior也未恶化={selected['realized_risk_vs_validation_prior_passed']}。",
              '', f"新增12配方使回看最优值相对18对照池减少 {result['new12_pool_byte_reduction_vs_old18_pct']:.4f}%；这是实测候选池潜力，不等同于在线推荐收益。", '',
              '| 数据集/列 | 对照最佳 B | 新组合最佳 B | 减少 | 在线是否选中新组合 |', '|---|---:|---:|---:|---|']
    for e in examples:
        lines.append(f"| {e['dataset']}/{e['column']} | {e['old_bytes']} | {e['best_bytes']} | {e['reduction_vs_old_pct']:.3f}% | {e['new_recipe_selected']} |")
    lines += ['', result['timing_caveat'], '',
              '确认组未参与。第二轮是由第一轮development调参结果启发的开发研究；两轮均保留，不将其称为新的外部盲测。完整135/3519历史oracle的不同计量口径见主报告。']
    (root / 'report/FINDINGS.zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'path': str(out), 'policy': chosen, 'selected': selected}, ensure_ascii=False))


if __name__ == '__main__':
    main()
