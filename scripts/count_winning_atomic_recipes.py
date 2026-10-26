#!/usr/bin/env python3
"""Read-only strict-winning logical pure recipe counts, main five runs only."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from count_atomic_composition_measurements import recipe_key

ROOT = Path(__file__).resolve().parents[1] / 'analysis_outputs/primitive_composition_research_20260906_v1'
RUNS = ('round1', 'round2', 'round3', 'ablation_round12', 'ablation_round3')


def read(path):
    return json.loads(path.read_text())


def main():
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    columns, candidates, controls, provenance = {}, {}, {}, []
    for name in RUNS:
        run = ROOT / name
        complete = read(run / 'COMPLETE.json')
        assert complete['status'] == 'PASS'
        provenance.append({'run': name, 'complete_sha256': hashlib.sha256((run / 'COMPLETE.json').read_bytes()).hexdigest(),
                           'manifest_sha256': hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest()})
        rows = [read(p) for p in sorted((run / 'columns').glob('*.json'))]
        assert len(rows) == complete['columns']
        for row in rows:
            key = row['key']
            if key not in columns:
                columns[key] = {'key': key, 'dataset': row['dataset'], 'name': row['name'], 'input_sha256': row['input_sha256'],
                                'raw_bytes': row['raw_bytes'], 'historical_best_bytes': row['historical_best']['compressed_bytes']}
                candidates[key], controls[key] = {}, []
            assert row['input_sha256'] == columns[key]['input_sha256']
            if name.startswith('round'):
                assert row['historical_best']['compressed_bytes'] == columns[key]['historical_best_bytes']
                algorithm = row['best']['algorithm']
                controls[key].append({'run': name, 'bytes': algorithm['bytes'], 'terminals': algorithm['terminals'],
                                      'source_json': str(run / 'columns' / (key + '.json'))})
            trace = run / 'traces' / (key + '.jsonl')
            for line_number, line in enumerate(trace.open(), 1):
                record = json.loads(line)
                if record.get('family') != 'pure' or record['status'] != 'ok':
                    continue
                logical_key = recipe_key(record['specs'], record['terminals'])
                prior = candidates[key].get(logical_key)
                value = (record['bytes'], record['depth'], name, line_number, record['frame_sha256'])
                if prior is None or value[0] < prior[0]:
                    candidates[key][logical_key] = value
    results = []
    for key in sorted(columns):
        row = columns[key]
        algorithm = min(controls[key], key=lambda c: c['bytes'])
        threshold = min(row['historical_best_bytes'], algorithm['bytes'])
        winning = sorted((value, logical_key) for logical_key, value in candidates[key].items() if value[0] < threshold)
        hist = Counter(value[1] for value, _ in winning)
        row.update(same_frame_algorithm_best_bytes=algorithm['bytes'], same_frame_algorithm_best_source=algorithm,
                   threshold_bytes=threshold, threshold_rule='strictly below min(historical_best, minimum same-frame algorithm winner over round1/2/3)',
                   successful_unique_pure_recipes=len(candidates[key]), strict_winning_recipes=len(winning),
                   strict_winning_multistage_recipes=sum(value[1] >= 2 for value, _ in winning),
                   strict_winning_zero_or_single_stage_recipes=sum(value[1] < 2 for value, _ in winning),
                   strict_winning_depth_histogram=dict(sorted(hist.items())), examples=[])
        for value, logical_key in winning[:5]:
            size, depth, name, line_number, digest = value
            specs, terminals = json.loads(logical_key)
            row['examples'].append({'bytes': size, 'saved_bytes': threshold - size, 'depth': depth, 'normalized_specs': specs,
                                    'terminals': terminals, 'source_trace': str(ROOT / name / 'traces' / (key + '.jsonl')),
                                    'source_line': line_number, 'frame_sha256': digest})
        results.append(row)
    assert len(results) == 20
    data = {'status': 'PASS', 'generated_at': started.isoformat(), 'codecs_run_by_audit': 0,
            'runs': list(RUNS), 'excluded': ['smoke_v1', 'round2_smoke', 'probe_no2_v3'],
            'key': '(column, recipe_key(normalized specs, ordered terminals)); frame version/SHA excluded',
            'equivalence_scope': 'distinct normalized operator sequences, not proven semantically distinct or irreducible encoders; only selected winners underwent deletion ablation',
            'per_recipe_size_rule': 'minimum actual successful bytes for that recipe across the five runs',
            'multistage_definition': 'len(specs) + len(terminals) >= 2; zero/single-stage candidates counted separately',
            'minimum_winning_recipes_per_column': min(r['strict_winning_recipes'] for r in results),
            'minimum_winning_multistage_recipes_per_column': min(r['strict_winning_multistage_recipes'] for r in results),
            'total_winning_recipes': sum(r['strict_winning_recipes'] for r in results),
            'total_winning_multistage_recipes': sum(r['strict_winning_multistage_recipes'] for r in results),
            'all_columns_have_multiple_winning_multistage_recipes': all(r['strict_winning_multistage_recipes'] >= 2 for r in results),
            'columns': results, 'sources': provenance,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    lines = ['# 每列严格胜出的纯算子逻辑组合数量', '', '审计时间：' + started.isoformat(), '',
             '范围：round1、round2、round3、ablation_round12、ablation_round3；不计 smoke/probe。按规范化逻辑链去重，不以不同帧版本扩充种数；每种链取跨轮最小成功实测文件。没有运行 codec。', '',
             '阈值＝min(历史最佳文件，round1/2/3 同封装完整算法对照最佳文件)，必须严格小于。此处“同封装完整算法”不包含混合组。', '',
             f"每列最少严格胜出 {data['minimum_winning_recipes_per_column']} 种纯候选；其中真正多阶段组合最少 {data['minimum_winning_multistage_recipes_per_column']} 种。", '',
             '| 列 | 历史最佳 B | 算法对照最佳 B | 严格阈值 B | 严格胜出种数 | 深度≥2组合数 | 零/单步数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('| {} {} | {} | {} | {} | {} | {} | {} |'.format(r['key'], r['name'], r['historical_best_bytes'], r['same_frame_algorithm_best_bytes'], r['threshold_bytes'], r['strict_winning_recipes'], r['strict_winning_multistage_recipes'], r['strict_winning_zero_or_single_stage_recipes']))
    lines += ['', f"合计 {data['total_winning_recipes']} 种严格胜出的逐列纯候选，{data['total_winning_multistage_recipes']} 种为深度≥2组合。", '',
              'JSON 附逐列深度分布和最多5个实测胜出例子（完整规范化参数、终端、trace行号和frame SHA）。数量不是理论最优保证，也不是RL独立收益；每列搜索范围不同。',
              '这些是默认参数规范化后的不同算子序列，不代表全部语义互不等价或不可删减；部分变换可能对该列无效。仅选中赢家做了删除消融，未对全部胜出链做消融。', '']
    stamp = started.strftime('%Y%m%d_%H%M%S_%f')
    snapshots = ROOT / 'codec_count_snapshots'
    snapshots.mkdir(exist_ok=True)
    base = 'WINNING_RECIPE_COUNTS_20260906'
    for suffix, content in (('.json', json.dumps(data, ensure_ascii=False, indent=2) + '\n'), ('.md', '\n'.join(lines))):
        with (snapshots / (base + '_' + stamp + suffix)).open('x') as handle:
            handle.write(content)
        temp = ROOT / (base + suffix + '.tmp')
        temp.write_text(content)
        temp.replace(ROOT / (base + suffix))
    print(json.dumps({k: data[k] for k in ('status', 'minimum_winning_recipes_per_column', 'minimum_winning_multistage_recipes_per_column', 'total_winning_recipes', 'total_winning_multistage_recipes', 'all_columns_have_multiple_winning_multistage_recipes')}, indent=2))


if __name__ == '__main__':
    main()
