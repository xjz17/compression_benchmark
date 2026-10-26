#!/usr/bin/env python3
"""Verify concrete new files, then report pure and mixed results separately."""
import argparse
import collections
import csv
import hashlib
import html
import json
import importlib
import datetime
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def chain(row):
    parts=[]
    for spec in row['specs']:
        params=', '.join(f'{k}={v}' for k,v in spec.items() if k!='op')
        parts.append(spec['op'] + (f'({params})' if params else ''))
    return ' → '.join(parts + row['terminals']) or '原样'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args=p.parse_args()
    run=args.run.resolve()
    from canonical206_data import load_smallest_two
    sources={f"{c['dataset']}__{c['numeric_index']:02d}":c for c in load_smallest_two(ROOT)}
    manifest=json.loads((run/'manifest.json').read_text())
    is_v3=(run/'code/scripts/atomic_composition_frame_v3.py').is_file()
    is_v2=is_v3 or 'v2' in str(manifest.get('registry','')) or (run/'code/scripts/atomic_composition_frame_v2.py').is_file()
    frame_module=importlib.import_module('atomic_composition_frame_v3' if is_v3 else ('atomic_composition_frame_v2' if is_v2 else 'atomic_composition_frame_v1'))
    decode_frame,inspect_frame,encode_frame=frame_module.decode_frame,frame_module.inspect_frame,frame_module.encode_frame
    pure_terminals={'range0','range1','sparse0','sparse1','bit4','bit8','bit12','bit16'} if is_v2 else {'range0','range1','sparse0','sparse1'}
    frame_name='ACF3' if is_v3 else ('ACF2' if is_v2 else 'ACF1')
    expected={f"{c['dataset']}__{c['numeric_index']:02d}" for c in manifest['columns']}
    manifest_rows={f"{c['dataset']}__{c['numeric_index']:02d}":c for c in manifest['columns']}
    if len(expected)!=len(manifest['columns']):
        raise AssertionError('duplicate manifest column')
    scope=manifest.get('scope', 'legacy_smoke_subset' if manifest.get('config',{}).get('keys') else 'unverified_scope')
    formal=scope=='full20_offline_search'
    if formal and (expected!=set(sources) or len(expected)!=20):
        raise AssertionError('formal manifest does not cover the exact canonical 20 columns')
    if 'prior_path' in manifest:
        prior=Path(manifest['prior_path']).read_bytes()
        if hashlib.sha256(prior).hexdigest()!=manifest['prior_sha256']:
            raise AssertionError('frozen comparator hash changed')
        prior_rows={f"{c['dataset']}__{c['numeric_index']:02d}":c for c in json.loads(prior)['columns']}
    else:
        prior_rows=None
    rows=[]
    row_keys=set()
    for f in sorted((run/'columns').glob('*.json')):
        r=json.loads(f.read_text())
        if r['key'] not in expected or r['status']!='PASS':
            raise AssertionError('unexpected or failed row')
        if r['key'] in row_keys or f.name!=r['key']+'.json':
            raise AssertionError('duplicate or misnamed row')
        row_keys.add(r['key'])
        if any(r.get(k)!=v for k,v in manifest_rows[r['key']].items()):
            raise AssertionError('row identity differs from frozen manifest')
        if prior_rows is not None and r['historical_best']!=prior_rows[r['key']]['old_algorithm_composition_best_including_history_and_frozen']:
            raise AssertionError('row historical comparator differs from frozen prior')
        c=sources[r['key']]
        if hashlib.sha256(c['raw']).hexdigest()!=r['input_sha256']:
            raise AssertionError('input drift')
        for family,w in r['best'].items():
            frame_path=Path(w['frame_path']).resolve()
            if not frame_path.is_relative_to(run/'frames'):
                raise AssertionError('winner path escapes current run frames')
            frame=frame_path.read_bytes()
            if len(frame)!=w['bytes'] or hashlib.sha256(frame).hexdigest()!=w['frame_sha256']:
                raise AssertionError('frame changed')
            if decode_frame(frame)!=c['raw']:
                raise AssertionError('independent report replay failed')
            info=inspect_frame(frame)
            if w['terminals']!=info['terminals'] or encode_frame(c['raw'],c['kind'],w['specs'],w['terminals'])!=frame:
                raise AssertionError('advertised chain differs from the actual encoded file')
            if info['registered_depth']!=w['depth'] or w['depth']>10:
                raise AssertionError('depth mismatch')
            if family=='pure' and any(t not in pure_terminals for t in w['terminals']):
                raise AssertionError('whole codec in primary pure group')
            if w['ratio']!=len(c['raw'])/len(frame):
                raise AssertionError('ratio mismatch')
        trace_path=Path(r['trace_path']).resolve()
        if trace_path!=run/'traces'/(r['key']+'.jsonl'):
            raise AssertionError('trace path does not match current run column')
        trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
        keys=[json.dumps([x['specs'],x['terminals']],sort_keys=True) for x in trace]
        if len(set(keys))!=len(keys) or len(keys)!=r['distinct_measured_candidates']:
            raise AssertionError('duplicate or phantom candidate count')
        for family in ['algorithm','pure','hybrid']:
            records=[x for x in trace if x['family']==family]
            good=[x for x in records if x['status']=='ok']
            if len(records)!=r['counts'].get(family+'_attempts',0) or len(good)!=r['counts'].get(family+'_successes',0):
                raise AssertionError('count differs from trace')
            if good and min(x['bytes'] for x in good)!=r['best'][family]['bytes']:
                raise AssertionError('winner differs from trace')
        r['trace_depth_histogram']=dict(sorted(collections.Counter(x['depth'] for x in trace).items()))
        r['verified_by_report']=True
        rows.append(r)
    complete={r['key'] for r in rows}==expected and len(rows)==len(expected) and (run/'COMPLETE.json').is_file()
    if complete and json.loads((run/'COMPLETE.json').read_text())['status']!='PASS':
        raise AssertionError('failed COMPLETE')
    if complete and formal and not json.loads((run/'COMPLETE.json').read_text()).get('full20_coverage'):
        raise AssertionError('missing explicit complete full20 acceptance flag')
    flat=[]
    for r in rows:
        old=r['historical_best']['compressed_bytes']
        algorithm=r['best']['algorithm']['bytes']
        pure=r['best']['pure']; hybrid=r['best']['hybrid']
        target=min(old,algorithm)
        flat.append({'dataset':r['dataset'],'column':r['name'], 'key':r['key'],
            'raw_bytes':r['raw_bytes'], 'historical_bytes':old,
            'same_frame_algorithm_chain':chain(r['best']['algorithm']), 'same_frame_algorithm_bytes':algorithm,
            'pure_chain':chain(pure),'pure_bytes':pure['bytes'],'pure_ratio':pure['ratio'],
            'pure_saved_vs_stronger_control':target-pure['bytes'],
            'pure_compress_sec':pure['compress_sec'],
            'hybrid_chain':chain(hybrid),'hybrid_bytes':hybrid['bytes'],'hybrid_ratio':hybrid['ratio'],
            'pure_distinct_candidates':r['counts']['pure_attempts'],
            'pure_unique_transformed_payloads':r['unique_transformed_payloads_measured']['pure'],
            'compose_sec':r['compose_sec']})
    status=('COMPLETE' if formal else ('TARGETED_COMPLETE' if scope=='targeted_extension' else 'SMOKE_COMPLETE')) if complete else 'PARTIAL'
    summary={'status':status,'scope':scope,'scope_columns':len(rows),
        'expected_columns':len(sources),'expected_run_columns':len(expected),'rl_updates':0,
        'pure_wins_vs_historical':sum(x['pure_bytes']<x['historical_bytes'] for x in flat),
        'pure_wins_vs_same_frame':sum(x['pure_bytes']<x['same_frame_algorithm_bytes'] for x in flat),
        'pure_wins_vs_both':sum(x['pure_saved_vs_stronger_control']>0 for x in flat),
        'pure_candidates':sum(x['pure_distinct_candidates'] for x in flat),
        'all_measured_candidates':sum(r['distinct_measured_candidates'] for r in rows),
        'full_file_lossless_winner_replay':complete,'winner_replay_scope_columns':len(rows),
        'columns':flat, 'run':str(run)}
    latest=run/'analysis'
    latest.mkdir(exist_ok=True)
    dest=latest/('snapshot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    dest.mkdir()
    (dest/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if flat:
        with (dest/'per_column.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(flat[0])); writer.writeheader();writer.writerows(flat)
    lines=['# 真正独立算子组合：逐列实验结果','',
        f"状态：{summary['status']}（{scope}），当前{len(rows)}/正式目标20列，本次运行设定{len(expected)}列。纯组实际测量 {summary['pure_candidates']:,} 个不同完整候选；所有组 {summary['all_measured_candidates']:,} 个。",'',
        f"纯算子组超过历史最佳 {summary['pure_wins_vs_historical']}/{len(rows)}列；超过同封装算法单步/二级联 {summary['pure_wins_vs_same_frame']}/{len(rows)}列；同时超过两者 {summary['pure_wins_vs_both']}/{len(rows)}列。",'',
        '纯组只有独立数值变换、打包/转置和独立算术熵编码，没有Bzip2/XZ/Zstd后端，没有撤销前缀再调用完整算法。混合组明确另列，不冒充纯算子收益。', '',
        f'CR=同一份完整int64/float64原始字节÷完整独立可解码文件字节；metadata/palette/模型规则/长度/校验/外头全部计入。新三组使用相同{frame_name}封装；旧历史包格式不同，因此同时给新同封装控制。', '',
        '当前是单seed、逐完整列离线搜索，不是held-out泛化测试，没有训练GRPO/PPO。本轮不声称RL贡献或全局最优。筛选代理值仅分配测量，不计为真实压缩候选。随机参数序列可能在当前输入上等价，故另报不同transform payload数。', '']
    for dataset in sorted({x['dataset'] for x in flat}):
        lines += [f'## {dataset}', '', '| 列 | 历史最佳 B | 同封装算法最佳 B | 纯算子最佳 B / CR | 比更强控制省 B | 纯算子组合 | 混合最佳 B |', '|---|---:|---:|---:|---:|---|---:|']
        for x in flat:
            if x['dataset']==dataset:
                lines.append(f"| {x['column']} | {x['historical_bytes']:,} | {x['same_frame_algorithm_bytes']:,} | {x['pure_bytes']:,} / {x['pure_ratio']:.6f}× | {x['pure_saved_vs_stronger_control']:+,} | {x['pure_chain']} | {x['hybrid_bytes']:,} |")
        lines += ['', '| 列 | 纯组完整候选 | 不同变换载荷 | 纯组冷编码 ms | 组合时间 s |', '|---|---:|---:|---:|---:|']
        for x in flat:
            if x['dataset']==dataset:
                lines.append(f"| {x['column']} | {x['pure_distinct_candidates']:,} | {x['pure_unique_transformed_payloads']:,} | {x['pure_compress_sec']*1000:.3f} | {x['compose_sec']:.2f} |")
        lines.append('')
    lines += ['## 范围与文件', '',
        '- 算法控制为5种明确参数的完整codec，穷举长度1/2级联共30种；不代表所有可能算法或参数的理论最佳。历史最强控制另含旧固定/Beam29/PPO/GRPO/frozen结果。',
        '- 计时不是跨机器速度排名：每列冷编码为无prepared缓存的重复测量中位数；组合时间含筛选、全部实际尝试、验证和冷回放，并行worker时间不能相加当墙钟。',
        '- 每个胜者的实际完整文件、逆参数、sha、原始计时在columns/*.json与frames/；每次真实编码结果在traces/*.jsonl；筛选记录在screening/。',
        '- 删除算子消融和fresh-agent审计另行保存；没有相应已完成文件时，不能视为已验证机制归因。',
        f'- 运行路径：`{run}`。','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    # Self-contained readable HTML: table cells are escaped, not external assets.
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>独立算子组合结果</title>',
        '<style>body{font:16px system-ui;max-width:1800px;margin:32px auto;padding:0 24px;color:#18334d}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{border:1px solid #ccd4da;padding:10px;text-align:left}th{background:#18334d;color:white}.win{background:#e1f1ea}.loss{background:#fff1e7}code{word-break:break-all}p{line-height:1.6}</style>',
        '<h1>真正独立算子组合：逐列结果</h1>',
        f"<p>{html.escape(lines[2])}</p><p>{html.escape(lines[4])}</p>",
        ''.join('<p>'+html.escape(t)+'</p>' for t in lines[6:11] if t)]
    for dataset in sorted({x['dataset'] for x in flat}):
        body.append('<h2>'+html.escape(dataset)+'</h2><table><tr><th>列</th><th>历史最佳 B</th><th>同封装算法 B</th><th>纯算子 B / CR</th><th>省 B</th><th>纯算子链</th><th>混合 B</th><th>纯候选数</th></tr>')
        for x in flat:
            if x['dataset']!=dataset: continue
            cells=[x['column'],f"{x['historical_bytes']:,}",f"{x['same_frame_algorithm_bytes']:,}",f"{x['pure_bytes']:,} / {x['pure_ratio']:.4f}×",f"{x['pure_saved_vs_stronger_control']:+,}",x['pure_chain'],f"{x['hybrid_bytes']:,}",f"{x['pure_distinct_candidates']:,}"]
            body.append('<tr class="'+('win' if x['pure_saved_vs_stronger_control']>0 else 'loss')+'">'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>')
        body.append('</table>')
    body.append('<p>详细时序、文件校验与范围见同目录 REPORT.md 和 per_column.csv。负值表示尚未超越；混合组不冒充纯算子组。</p></html>')
    (dest/'REPORT.html').write_text('\n'.join(body))
    for artifact in dest.iterdir():
        shutil.copy2(artifact,latest/artifact.name)
    print(json.dumps({k:v for k,v in summary.items() if k!='columns'},ensure_ascii=False))


if __name__=='__main__':main()
