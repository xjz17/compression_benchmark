"""Audit and render the separately measured fixed-algorithm column baseline."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

from canonical206_column_tables import _short_name


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def same(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9)


def escaped(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def build_fixed_tables(parent, manifest, registry):
    """Require the complete extension, validate it, then return Markdown lines."""
    parent = Path(parent).resolve()
    root = Path(__file__).resolve().parents[1]
    out = parent / 'fixed_algorithm_baseline_v1'
    complete = json.loads((out / 'COMPLETE.json').read_text())
    baseline_manifest = json.loads((out / 'manifest.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((out / 'columns').glob('*.json'))]
    cols = {(c['dataset'], c['numeric_index']): c for c in manifest['columns']}
    observed = {(r['dataset'], r['numeric_index']) for r in rows}
    assert observed == set(cols) and len(rows) == len(cols) == complete['columns'] == 20
    assert complete['status'] == 'PASS' and complete['all_lossless']
    assert sha(parent / 'manifest.json') == baseline_manifest['parent_manifest_sha256']
    assert sha(parent / 'action_registry.json') == baseline_manifest['registry_sha256']
    assert all(sha(root / rel) == digest for rel, digest in baseline_manifest['source_hashes'].items())
    assert all(sha(root / 'cpp/build' / name) == digest
               for name, digest in baseline_manifest['runtime_cli_hashes'].items())
    actions = {a['policy_action_id']: a['name'] for a in registry['policy_actions']}
    with (out / 'best.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_lookup = {(r['dataset'], int(r['numeric_index'])): r for r in csv_rows}
    assert set(csv_lookup) == set(cols) and len(csv_rows) == len(rows)
    expected_stems = {f"{d}__{i:02d}" for d, i in cols}
    for directory in ('columns', 'candidates', 'traces'):
        assert {p.stem for p in (out / directory).glob('*.json')} == expected_stems
    candidate_pairs = successful_pairs = rejected_pairs = 0
    for r in rows:
        key = r['dataset'], r['numeric_index']
        c = cols[key]
        assert r['lossless'] and r['input_sha256'] == c['input_sha256']
        assert r['column'] == c['name'] and r['kind'] == c['kind']
        assert r['source_index'] == c['source_index'] and r['values'] == c['count']
        assert r['source_sha256'] == c['source_sha256'] == sha(c['source_path'])
        assert r['original_bytes'] == c['raw_bytes'] and r['compressed_bytes'] > 0
        assert same(r['compression_ratio'], r['original_bytes'] / r['compressed_bytes'])
        assert r['selected_pipeline'] == [actions[i] for i in r['selected_ids']]
        frame = out / r['frame_path']
        assert frame.stat().st_size == r['compressed_bytes'] and sha(frame) == r['frame_sha256']
        with frame.open('rb') as stream:
            magic, kind, layers, count, _crc = struct.unpack('<4sBBQI', stream.read(18))
        assert magic == b'C206' and kind == {'int64': 1, 'float64': 2}[r['kind']]
        assert count == r['values']
        assert (1 <= layers <= len(r['selected_ids'])) if r['selected_ids'] else layers == 0
        for field in ('compress_sec', 'decompress_sec'):
            samples = r[field + '_repeats']
            assert len(samples) == 5 and all(math.isfinite(v) and v >= 0 for v in samples)
            assert same(r[field], statistics.median(samples))
        assert math.isfinite(r['fixed_algorithm_selection_sec']) and r['fixed_algorithm_selection_sec'] >= 0
        candidates = json.loads((out / r['candidate_results_path']).read_text())
        good = [x for x in candidates if x['status'] == 'ok']
        bad = [x for x in candidates if x['status'] == 'rejected']
        assert len(candidates) == len(good) + len(bad) == r['candidate_unique_chains'] + 1
        assert len({tuple(x['ids']) for x in candidates}) == len(candidates)
        raw = [x for x in good if not x['ids']]
        assert len(raw) == 1 and raw[0]['compressed_bytes'] == r['original_bytes'] + 18
        for x in candidates:
            assert x['names'] == [actions[i] for i in x['ids']]
            assert math.isfinite(x['candidate_wall_sec']) and x['candidate_wall_sec'] >= 0
            assert set(x['aliases']) == set(x['variants'])
            for label in x['aliases']:
                if label == 'RAW':
                    assert not x['ids']
                    continue
                mapping = registry['benchmark_86_canonical_action_pipelines'][label]
                recipe = mapping[x['variants'][label]] if isinstance(mapping, dict) else mapping
                assert x['names'] == recipe, (key, label, x['variants'])
        for x in good:
            assert x['lossless'] and same(x['compression_ratio'], r['original_bytes'] / x['compressed_bytes'])
        best = min(good, key=lambda x: (x['compressed_bytes'], x['aliases'], x['ids']))
        assert (r['selected_ids'], r['algorithm_aliases'], r['compressed_bytes'], r['frame_sha256']) == (
            best['ids'], best['aliases'], best['compressed_bytes'], best['frame_sha256'])
        assert r['algorithm'] == best['aliases'][0]
        assert r['selected_variants'] == best['variants']
        assert r['successful_chains'] == len(good) - 1 and r['rejected_chains'] == len(bad)
        labels = [label for x in candidates if x['ids'] for label in x['aliases']]
        excluded = [x['label'] for x in r['excluded_labels']]
        assert len(labels) == len(set(labels)) == r['candidate_labels']
        assert not set(labels).intersection(excluded)
        assert set(labels).union(excluded) == set(registry['benchmark_86_canonical_action_pipelines'])
        trace = json.loads((out / r['trace_path']).read_text())
        stats = r['search_stats']
        assert len(trace) == stats['unique_edges']
        assert len({tuple(x['prefix']) for x in trace}) == len(trace)
        assert sum(x['status'] == 'ok' for x in trace) == stats['valid_edges']
        assert sum(x['status'] == 'rejected' for x in trace) == stats['rejected_edges']
        for field in ('forward_sec', 'inverse_sec'):
            assert all(math.isfinite(x[field]) and x[field] >= 0 for x in trace)
            assert same(sum(x[field] for x in trace), stats[field])
        successful_trace = {tuple(x['prefix']): x for x in trace if x['status'] == 'ok'}
        for x in good:
            if x['ids']:
                assert successful_trace[tuple(x['ids'])]['frame_bytes'] == x['compressed_bytes']
        assert r['attempted_action_ids'] == sorted({x['prefix'][-1] for x in trace})
        assert r['successful_action_ids'] == sorted({x['prefix'][-1] for x in trace if x['status'] == 'ok'})
        assert sum(x['candidate_wall_sec'] for x in candidates) <= r['fixed_algorithm_selection_sec'] + 1e-8
        for field in ('original_bytes', 'compressed_bytes', 'compression_ratio', 'compress_sec',
                      'decompress_sec', 'candidate_unique_chains', 'candidate_labels',
                      'successful_chains', 'rejected_chains', 'fixed_algorithm_selection_sec'):
            assert same(csv_lookup[key][field], r[field]), (key, field)
        assert csv_lookup[key]['algorithm'] == r['algorithm']
        assert csv_lookup[key]['algorithm_aliases'] == ' | '.join(r['algorithm_aliases'])
        assert csv_lookup[key]['selected_ids'] == ' -> '.join(map(str, r['selected_ids']))
        assert csv_lookup[key]['selected_pipeline'] == ' -> '.join(r['selected_pipeline'])
        assert json.loads(csv_lookup[key]['compress_sec_repeats']) == r['compress_sec_repeats']
        for field in ('input_sha256', 'source_sha256', 'frame_sha256', 'frame_path'):
            assert csv_lookup[key][field] == r[field]
        candidate_pairs += r['candidate_unique_chains']
        successful_pairs += r['successful_chains']
        rejected_pairs += r['rejected_chains']
    assert candidate_pairs == complete['candidate_chain_column_pairs']
    assert successful_pairs == complete['successful_chain_column_pairs']
    assert rejected_pairs == complete['rejected_chain_column_pairs']
    audit = dict(status='PASS', columns=len(rows), candidate_chain_column_pairs=candidate_pairs,
                 successful_chain_column_pairs=successful_pairs, rejected_chain_column_pairs=rejected_pairs,
                 complete_frame_and_input_hashes=True, same_frozen_inputs_and_runtime=True,
                 minimum_successful_full_chain_selection=True, same_row_size_time_and_chain=True,
                 all_five_encoder_samples_and_csv_verified=True,
                 scope='registered fixed algorithm chains; raw fallback; not global or upstream-native oracle')
    (out / 'REPORT_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    row_lookup = {(r['dataset'], r['numeric_index']): r for r in rows}
    lines = ['## 每列最佳固定压缩算法：同输入、同封装补测', '',
             '“最佳算法”指冻结注册表中的**固定完整算法链**在本列压缩后字节最少，不是最快算法，也不是所有算法及参数的全局最优。这里没有额外拼接第二种算法。',
             '从 86 个方法标签出发，沿用 Camel 排除设置：整数列有 82 个适用标签、去重后 65 条链；浮点列有 85 个标签、去重后 69 条链。完全相同的链只测一次、保留别名。失败或超时候选不参与选优，成功数在表中列出。',
             '这些是当前项目的注册实现映射，包含 reference/proxy 和残余复合核心，不等同于 86 个上游原生库。', '',
             '这是对完整列逐个测试后的事后最佳基线，不是零成本选择器。全部算法筛选的墙钟保留在 CSV 的 `fixed_algorithm_selection_sec` 字段，不混入下表胜者的编码时间。', '',
             '补测保留原始 int64/float64 位模式，并使用相同 C206 + UAR1 完整封装、相同冻结的 C++ 接口。压缩时间为**选中算法自身** 1 次预热后 5 次新编码的中位数，不含算法筛选和解码；仍含 CLI/IPC 开销。补测与原实验分时运行，未证明细小计时差异有统计显著性。', '']
    for dataset in dict.fromkeys(c['dataset'] for c in manifest['columns']):
        lines += [f'### {dataset}：最佳固定算法', '',
                  '| 列 | 最佳固定算法（同链别名） | 完整固定算子链 | 压缩字节 | 压缩比 | 压缩时间（秒） | 成功 / 候选链 |',
                  '|---|---|---|---:|---:|---:|---:|']
        for key in cols:
            if key[0] != dataset:
                continue
            r = row_lookup[key]
            chain = ' → '.join(_short_name(i, actions[i]) for i in r['selected_ids']) or 'RAW'
            aliases = ' / '.join(r['algorithm_aliases'])
            lines.append(f"| {escaped(r['column'])} | {escaped(aliases)} | {escaped(chain)} | "
                         f"{r['compressed_bytes']:,} | {r['compression_ratio']:.3f}× | {r['compress_sec']:.5f} | "
                         f"{r['successful_chains']} / {r['candidate_unique_chains']} |")
        lines.append('')
    lines += [f'共补测 {candidate_pairs:,} 个不同链与列的配对，{successful_pairs:,} 个完整链通过精确解码，'
              f'{rejected_pairs:,} 个被拒绝。20 个最终优胜结果均完成 5 次独立编码计时和逐字节回放。', '',
              '[每列最佳固定算法 CSV](fixed_algorithm_baseline_v1/best.csv) · '
              '[补测清单](fixed_algorithm_baseline_v1/manifest.json) · '
              '[逐列 JSON](fixed_algorithm_baseline_v1/columns) · '
              '[所有固定链候选](fixed_algorithm_baseline_v1/candidates) · '
              '[补测审计](fixed_algorithm_baseline_v1/REPORT_AUDIT.json)', '']
    lines += ['固定算法表中的 ID 与原搜索使用同一注册表；完整实现名称保存在该表 CSV 的 `selected_pipeline` 字段，'
              '也可查 [冻结的 206 实现注册表](action_registry.json)。', '']
    comparisons = []
    for r in rows:
        beam = json.loads((parent / 'columns' /
            f"beam29__seed0__{r['dataset']}__{r['numeric_index']:02d}.json").read_text())
        assert beam['input_sha256'] == r['input_sha256']
        comparisons.append((r['compressed_bytes'] > beam['compressed_bytes']) -
                           (r['compressed_bytes'] < beam['compressed_bytes']))
    lines += [f'按逐列压缩字节比较：固定算法基线在 {comparisons.count(-1)}/20 列更小，'
              f'Beam29 在 {comparisons.count(1)}/20 列更小，{comparisons.count(0)} 列持平。'
              '因此原三种搜索方法之间的排序，不能扩展为“Beam29 每列都优于固定算法”。'
              '固定基线进行了全列全候选筛选，而搜索方法先筛前缀，这也不是等测量预算对照。', '']
    return lines
