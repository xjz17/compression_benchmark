#!/usr/bin/env python3
"""Read-only full-frame measurement audit; writes only derived audit reports.

Only finished runs are read. Screening proxies, transformed-body hashes, frame
versions, and frame hashes are never treated as measured logical recipe keys.
Import normalize_specs / recipe_key to share the exact identity convention.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'analysis_outputs/primitive_composition_research_20260906_v1'
DEFAULTS = {
    'delta': {'lag': 1, 'order': 1}, 'xor': {'lag': 1},
    'compact': {'width': 'auto'}, 'bitpack': {'bits': 'auto'},
    'dictionary': {'max_entries': 4096}, 'shuffle': {'width': 8},
    'bittranspose': {'width': 8}, 'float_scale': {'factor': 1},
    'float_scale_exceptions': {'factor': 1, 'max_fraction': 0.05},
    'block_for': {'block_size': 128}, 'block_bitpack': {'block_size': 128},
}
ALGORITHMS = frozenset(('zlib6', 'bzip2', 'xz', 'zstd', 'brotli'))
ENTROPY = frozenset(('range0', 'range1', 'sparse0', 'sparse1', 'bit4', 'bit8', 'bit12', 'bit16'))


def normalize_specs(specs):
    """Fill frozen runtime defaults, retaining order and all explicit values."""
    if not isinstance(specs, (list, tuple)) or any(not isinstance(s, dict) or not isinstance(s.get('op'), str) for s in specs):
        raise ValueError('specs must be an ordered list of operator dictionaries')
    result = [{**DEFAULTS.get(s['op'], {}), **s} for s in specs]
    for spec in result:
        if spec['op'] == 'float_scale_exceptions':
            spec['max_fraction'] = float(spec['max_fraction'])
    return result


def recipe_key(specs, terminals):
    """Canonical logical identity, excluding frame version/hash and run/family."""
    if not isinstance(terminals, (list, tuple)) or not all(isinstance(t, str) for t in terminals):
        raise ValueError('terminals must be an ordered string sequence')
    return json.dumps([normalize_specs(specs), list(terminals)], sort_keys=True, separators=(',', ':'), allow_nan=False)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_of(record):
    terminals = set(record['terminals'])
    if not terminals <= ALGORITHMS | ENTROPY:
        raise ValueError('unrecognized terminal in saved trace')
    inferred = ('hybrid' if record['specs'] else 'algorithm') if terminals & ALGORITHMS else 'pure'
    if record.get('family', inferred) != inferred:
        raise ValueError('saved family disagrees with explicit pipeline')
    return inferred


class Counts:
    def __init__(self):
        self.calls = Counter()
        self.keys = set()
        self.success = set()
        self.depths = {}
        self.replay_calls = 0

    def add(self, logical_key, record):
        status = record['status']
        if status not in ('ok', 'invalid', 'rejected'):
            raise ValueError('not an executed frame-encode attempt: ' + status)
        depth = len(record['specs']) + len(record['terminals'])
        assert record['depth'] == depth and depth <= 10
        self.calls[status] += 1
        self.keys.add(logical_key)
        self.depths[logical_key] = depth
        if status == 'ok':
            assert isinstance(record['bytes'], int) and record['bytes'] > 0 and len(record['frame_sha256']) == 64
            self.success.add(logical_key)

    def result(self):
        attempts = sum(self.calls.values())
        return {
            'trace_encode_attempt_calls': attempts,
            'trace_successful_frame_calls': self.calls['ok'],
            'trace_invalid_calls': self.calls['invalid'],
            'trace_rejected_calls': self.calls['rejected'],
            'unique_attempted_logical_candidates': len(self.keys),
            'unique_successful_logical_candidates': len(self.success),
            'unique_multistage_attempted_candidates': sum(self.depths[k] >= 2 for k in self.keys),
            'unique_multistage_successful_candidates': sum(self.depths[k] >= 2 for k in self.success),
            'duplicate_trace_remeasure_calls': attempts - len(self.keys),
            'cold_replay_encode_calls': self.replay_calls,
            'total_encode_attempt_calls': attempts + self.replay_calls,
            'total_verified_frame_encode_calls': self.calls['ok'] + self.replay_calls,
            'unique_attempted_depth_histogram': dict(sorted(Counter(self.depths[k] for k in self.keys).items())),
            'unique_successful_depth_histogram': dict(sorted(Counter(self.depths[k] for k in self.success).items())),
        }


class Ledger:
    def __init__(self):
        self.all = Counts()
        self.family = defaultdict(Counts)
        self.column = defaultdict(lambda: defaultdict(Counts))

    def add(self, column, record):
        fam = family_of(record)
        key = column, recipe_key(record['specs'], record['terminals'])
        for count in (self.all, self.family[fam], self.column[column]['all'], self.column[column][fam]):
            count.add(key, record)

    def replay(self, column, family, times):
        for count in (self.all, self.family[family], self.column[column]['all'], self.column[column][family]):
            count.replay_calls += times

    def result(self):
        return {'all': self.all.result(),
                'families': {k: v.result() for k, v in sorted(self.family.items())},
                'columns': {k: {f: c.result() for f, c in sorted(v.items())} for k, v in sorted(self.column.items())}}


def completed_runs(root):
    finished, excluded = [], []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if re.fullmatch(r'round\d+', name) or name.startswith('ablation_'):
            category = 'main'
        elif name in ('smoke_v1', 'round2_smoke'):
            category = 'smoke'
        elif name.startswith('probe_'):
            category = 'probes'
        else:
            continue
        marker = path / ('SUMMARY.json' if category == 'probes' else 'COMPLETE.json')
        expected_status = 'COMPLETE' if category == 'probes' else 'PASS'
        if not marker.is_file() or read(marker).get('status') != expected_status:
            excluded.append({'run': name, 'category': category, 'reason': 'no successful completion marker at snapshot start'})
            continue
        finished.append((marker.stat().st_mtime_ns, path, marker, category))
    return sorted(finished), excluded


def audit(root):
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    finished, excluded = completed_runs(root)
    ledgers = {k: Ledger() for k in ('main', 'smoke', 'probes', 'main_plus_probes', 'search_rounds_only')}
    inputs, runs, sources, failures = {}, [], [], []
    for _, path, marker, category in finished:
        manifest_path = path / 'manifest.json'
        manifest, complete = read(manifest_path), read(marker)
        is_ablation = path.name.startswith('ablation_')
        run_ledger = Ledger()
        destinations = [ledgers[category], run_ledger]
        if category in ('main', 'probes'):
            destinations.append(ledgers['main_plus_probes'])
        if category == 'main' and not is_ablation:
            destinations.append(ledgers['search_rounds_only'])
        run_source = {'run': path.name, 'category': category, 'path': str(path), 'completion_path': str(marker),
                      'completion_sha256': sha(marker), 'manifest_sha256': sha(manifest_path), 'trace_files': [],
                      'screening_excluded': {}, 'cold_replay_basis': 'none' if category == 'probes' else 'row repeats + one warmup per winner; ablation also one parent winner replay'}
        if category == 'probes':
            inp = manifest['input']
            column = '{}__{:02d}'.format(inp['dataset'], inp['numeric_index'])
            rows = [(column, inp, path / 'candidates.jsonl', None)]
            run_source['screening_excluded'] = {'unique_full_bodies_screened': manifest.get('unique_full_bodies_screened')}
        else:
            rows = []
            for row_path in sorted((path / 'columns').glob('*.json')):
                row = read(row_path)
                assert row['status'] == 'PASS'
                column = row['key']
                rows.append((column, row, path / 'traces' / (column + '.jsonl'), row))
            assert len(rows) == complete['columns']
        for column, inp, trace_path, row in rows:
            input_info = {'dataset': inp['dataset'], 'name': inp['name'], 'input_sha256': inp['input_sha256'], 'raw_bytes': inp['raw_bytes']}
            if column in inputs:
                assert inputs[column] == input_info, ('input mismatch', path.name, column)
            inputs[column] = input_info
            trace_sha = sha(trace_path)
            local_counts = Counter()
            count = 0
            with trace_path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    fam = family_of(record)
                    for ledger in destinations:
                        ledger.add(column, record)
                    local_counts[fam + '_attempts'] += 1
                    local_counts[fam + '_successes'] += record['status'] == 'ok'
                    count += 1
                    if record['status'] != 'ok':
                        failures.append({'run': path.name, 'column': column, 'trace_path': str(trace_path),
                                         'status': record['status'], 'error': record.get('error', record.get('reason')),
                                         'specs': record['specs'], 'terminals': record['terminals']})
            assert trace_sha == sha(trace_path), 'completed trace mutated while auditing'
            run_source['trace_files'].append({'path': str(trace_path), 'sha256': trace_sha, 'records': count})
            if row is not None:
                if is_ablation:
                    assert count == row['measured_candidates']
                    assert local_counts['pure_successes'] == row['successful_candidates']
                else:
                    assert count == row['distinct_measured_candidates']
                    for k, value in local_counts.items():
                        assert row['counts'][k] == value, (path.name, column, k)
                    for k, value in row['counts'].items():
                        if k.startswith('screen_') or k == 'duplicate_proposals_skipped':
                            run_source['screening_excluded'][k] = run_source['screening_excluded'].get(k, 0) + value
                for family, winner in row['best'].items():
                    repetitions = len(winner['compress_sec_repeats']) + 1
                    if not is_ablation:
                        assert len(winner['compress_sec_repeats']) == manifest['config']['repeats']
                    for ledger in destinations:
                        ledger.replay(column, family, repetitions)
                if is_ablation:
                    # Source code verifies the original selected winner once,
                    # before the deletion candidates and final warmup+3 repeats.
                    for ledger in destinations:
                        ledger.replay(column, 'pure', 1)
        run_stats = run_ledger.result()
        if category == 'probes':
            assert run_stats['all']['trace_encode_attempt_calls'] == complete['counters']['full_file_attempts']
            assert run_stats['all']['trace_successful_frame_calls'] == complete['counters']['full_file_verified']
            run_source['negative_result'] = {'best_bytes': complete['best']['bytes'], 'target_bytes': complete['target_bytes'], 'strict_win': complete['strict_win']}
        else:
            total_field = 'attempts' if is_ablation else 'distinct_measured_candidates'
            assert run_stats['all']['trace_encode_attempt_calls'] == complete[total_field]
        assert sha(marker) == run_source['completion_sha256'], 'completion marker changed during audit'
        runs.append({'run': path.name, 'category': category, 'stats': run_stats})
        sources.append(run_source)
    result = {
        'status': 'PASS', 'generated_at': started.isoformat(), 'root': str(root), 'script_sha256': sha(Path(__file__)),
        'codecs_run_by_this_audit': 0, 'normalization_defaults': DEFAULTS,
        'logical_key': '(dataset/numeric column, recipe_key(normalize_specs(specs), ordered terminals))',
        'attempt_definition': 'one saved call to top-level encode_frame/encode_prepared; includes rejected/invalid calls, does not imply a compressed file was produced',
        'success_definition': 'status ok: complete framed file encoded and decoded byte-exactly by the source run',
        'count_level': 'top-level frame encode API invocations, not subprocess/terminal-internal calls. A no-terminal frame also counts. Internal decoder canonical re-encoding is not a new candidate.',
        'replay_definition': 'cold validation/warmup/repetition calls outside candidate traces, inferred from completed runner protocol and saved compress_sec_repeats, never unique new candidates',
        'normalization_note': 'defaults materialized then ordered specs+terminals serialized with sorted map keys; frame_sha256 and ACF version excluded. Auto widths stay auto, not replaced by data-dependent realized widths.',
        'screening_definition': 'all entropy estimates, apply_ops-only pools and skipped duplicate proposals excluded from full-frame measurement counts',
        'completed_runs': [r['run'] for r in runs], 'excluded_incomplete_runs': excluded,
        'scopes': {name: ledger.result() for name, ledger in ledgers.items()},
        'per_run': runs, 'inputs': inputs, 'failed_candidate_records': failures, 'sources': sources,
    }
    return result


def markdown(data):
    main = data['scopes']['main']
    pure = main['families'].get('pure', {})
    lines = ['# 实际完整帧编码与逻辑组合覆盖审计', '', '时间：' + data['generated_at'], '',
             '统计完成标志成功的轮次；审计本身未调用任何 codec。主统计包含正式轮和消融，smoke 与补充 probe 分开。', '',
             '- 纳入完成轮：' + '、'.join(data['completed_runs']),
             '- 尚未纳入：' + ('、'.join(r['run'] for r in data['excluded_incomplete_runs']) or '无'), '',
             f"主结果纯原语：{pure.get('trace_encode_attempt_calls', 0):,} 次候选编码调用，跨轮/默认值归一化后 {pure.get('unique_attempted_logical_candidates', 0):,} 条不同链，成功 {pure.get('unique_successful_logical_candidates', 0):,} 条。",
             f"上面保留了零步identity/单步控制；只算深度≥2的真正多阶段组合：{pure.get('unique_multistage_attempted_candidates', 0):,} 条不同尝试，成功 {pure.get('unique_multistage_successful_candidates', 0):,} 条。",
             f"其中重复候选重测 {pure.get('duplicate_trace_remeasure_calls', 0):,} 次；另有冷验证/预热/重复 {pure.get('cold_replay_encode_calls', 0):,} 次。失败/invalid 不作成功文件；编码调用不等于已产出文件。", '',
             '## 分轮调用（未跨轮去重）', '',
             '| 轮次 | 类别 | 纯候选调用 | 纯成功调用 | 所有候选调用 | 冷重放调用 |',
             '|---|---|---:|---:|---:|---:|']
    for run in data['per_run']:
        stat = run['stats']; p = stat['families'].get('pure', {})
        lines.append(f"| {run['run']} | {run['category']} | {p.get('trace_encode_attempt_calls', 0)} | {p.get('trace_successful_frame_calls', 0)} | {stat['all']['trace_encode_attempt_calls']} | {stat['all']['cold_replay_encode_calls']} |")
    lines += ['', '## 逐列主结果累计', '',
              '| 列 | 纯调用 | 不同纯尝试 | 不同纯成功 | 重复候选重测 | 纯冷重放 | 全类别候选调用 | 全类别冷重放 | 总调用 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, stats in main['columns'].items():
        p, a = stats.get('pure', {}), stats['all']
        name = data['inputs'][key]['name']
        lines.append('| {} {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(key, name, p.get('trace_encode_attempt_calls', 0), p.get('unique_attempted_logical_candidates', 0), p.get('unique_successful_logical_candidates', 0), p.get('duplicate_trace_remeasure_calls', 0), p.get('cold_replay_encode_calls', 0), a['trace_encode_attempt_calls'], a['cold_replay_encode_calls'], a['total_encode_attempt_calls']))
    lines += ['', '## 去重后深度分布（主结果纯原语）', '', '| 深度 | 不同尝试 | 不同成功 |', '|---:|---:|---:|']
    for depth, count in pure.get('unique_attempted_depth_histogram', {}).items():
        lines.append(f"| {depth} | {count} | {pure['unique_successful_depth_histogram'].get(depth, 0)} |")
    lines += ['', '## 独立范围', '', '| 范围 | 纯编码调用 | 不同纯尝试 | 不同纯成功 | 冷重放（全类别） |', '|---|---:|---:|---:|---:|']
    for name in ('smoke', 'probes', 'main_plus_probes', 'search_rounds_only'):
        stat = data['scopes'][name]; p = stat['families'].get('pure', {})
        lines.append(f"| {name} | {p.get('trace_encode_attempt_calls', 0)} | {p.get('unique_attempted_logical_candidates', 0)} | {p.get('unique_successful_logical_candidates', 0)} | {stat['all']['cold_replay_encode_calls']} |")
    lines += ['', '## 精确口径', '',
              '- 逻辑键是列＋补齐已冻结默认值的 specs JSON＋有序 terminals。不以 ACF 版本或 frame SHA 变动算新链，也不把同链多次编码当不同组合。',
              '- attempt 是保存的完整帧 encode API 调用，包含未产出文件的 typed/data-dependent rejection；success 仅 status=ok 且源程序已完成逐字节无损回环。',
              '- 冷验证次数：正式/烟雾搜索每个赢家 1 次预热＋保存的重复次数；消融每列还有 1 次原赢家编码核验，最终 1 次预热＋3 次重复。probe 没有额外冷重放。',
              '- 总调用是顶层完整帧编码调用；不是熵 CLI 次数或算子调用次数。解码器内部 canonical re-encode 不算新候选。',
              '- screening 只测变换/熵估计，不计入完整编码测量。消融无效链明确排除成功数。',
              '- 同一逻辑链跨帧版本重测仍会影响文件字节数，去重只用于覆盖数，不能把不同版本的文件字节直接视作不变。',
              '- JSON 包含各 scope、各轮及逐列调用/去重/深度统计、失败链及完成标志/trace 的 SHA256；仅读取完成文件。', '']
    for source in data['sources']:
        if 'negative_result' in source:
            n = source['negative_result']
            lines.append(f"补充负结果：{source['run']} 最佳 {n['best_bytes']} B，严格目标 <{n['target_bytes']} B，strict_win={n['strict_win']}；不冒充 round3。")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    stamp = datetime.fromisoformat(result['generated_at']).strftime('%Y%m%d_%H%M%S_%f')
    basename = 'CODEC_COUNT_AUDIT_20260906'
    snapshots = args.root / 'codec_count_snapshots'
    snapshots.mkdir(exist_ok=True)
    outputs = {'.json': json.dumps(result, ensure_ascii=False, indent=2) + '\n', '.md': markdown(result)}
    for suffix, content in outputs.items():
        snapshot = snapshots / (basename + '_' + stamp + suffix)
        with snapshot.open('x') as handle:
            handle.write(content)
        latest = args.root / (basename + suffix)
        temporary = latest.with_suffix(suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(latest)
    print(json.dumps({'status': result['status'], 'completed_runs': result['completed_runs'],
                      'excluded': result['excluded_incomplete_runs'],
                      'main_pure': result['scopes']['main']['families'].get('pure', {}),
                      'snapshot_prefix': str(snapshots / (basename + '_' + stamp)),
                      'latest': str(args.root / (basename + '.json'))}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
