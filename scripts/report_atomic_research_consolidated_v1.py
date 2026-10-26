#!/usr/bin/env python3
"""Verify and consolidate full-column pure searches, keeping negative controls."""
import argparse
import collections
import csv
import datetime
import hashlib
import html
import importlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PURE={'range0','range1','sparse0','sparse1','bit4','bit8','bit12','bit16'}

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def label(w):
    return ' → '.join([s['op']+('('+','.join(f'{k}={v}' for k,v in s.items() if k!='op')+')' if len(s)>1 else '') for s in w['specs']]+w['terminals']) or 'raw'
def path_abs(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p
def verify(w,raw,kind):
    buf=path_abs(w['frame_path']).read_bytes()
    version=int(chr(buf[3]));mod=importlib.import_module(f'atomic_composition_frame_v{version}')
    assert len(buf)==w['bytes'] and hashlib.sha256(buf).hexdigest()==w['frame_sha256']
    assert mod.decode_frame(buf)==raw and mod.encode_frame(raw,kind,w['specs'],w['terminals'])==buf
    info=mod.inspect_frame(buf)
    assert info['registered_depth']==w['depth']<=10
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--ablations',nargs='*',type=Path,default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from canonical206_data import load_smallest_two
    from count_atomic_composition_measurements import recipe_key
    from atomic_transform_extensions_v3 import SUPPORTED_OPERATORS
    cols={f"{x['dataset']}__{x['numeric_index']:02d}":x for x in load_smallest_two(ROOT)}
    assert len(cols)==20
    rows=collections.defaultdict(list); reductions=collections.defaultdict(list)
    measured=collections.Counter();successful=collections.Counter();logical=collections.defaultdict(set);oklogical=collections.defaultdict(set)
    screen=collections.Counter();times=collections.Counter();depth=collections.defaultdict(collections.Counter)
    recipe_sizes=collections.defaultdict(dict)
    provenance=[]
    observed_ops=set()
    all_paths=a.runs+a.ablations
    for run_arg in all_paths:
        run=run_arg.resolve();complete=read(run/'COMPLETE.json');manifest=read(run/'manifest.json')
        assert complete['status']=='PASS'
        is_ablation=run_arg in a.ablations
        if not is_ablation:
            assert manifest['scope'] in ('full20_offline_search','targeted_extension'),'smokes cannot become formal result rows'
            for rel,h in manifest['source_hashes'].items():assert sha(ROOT/rel)==h
            assert sha(manifest['prior_path'])==manifest['prior_sha256']
        else:
            assert sha(run/'code/scripts/ablate_atomic_winners_v1.py')==manifest['source_code_sha256']
        provenance.append({'run':str(run),'scope':manifest['scope'],'manifest_sha256':sha(run/'manifest.json'),'complete_sha256':sha(run/'COMPLETE.json')})
        for rp in sorted((run/'columns').glob('*.json')):
            r=read(rp);k=r['key'];assert k in cols and r['status']=='PASS'
            assert r['input_sha256']==cols[k]['input_sha256']
            (reductions if is_ablation else rows)[k].append((r,run))
            times[k]+=r['compose_sec'];screen[k]+=r.get('screen_pool',0)
            for line in (run/'traces'/f'{k}.jsonl').read_text().splitlines():
                t=json.loads(line)
                if t['family']!='pure':continue
                seq=recipe_key(t['specs'],t['terminals'])
                measured[k]+=1;logical[k].add(seq);depth[k][t['depth']]+=1
                if t['status']=='ok':
                    successful[k]+=1;oklogical[k].add(seq)
                    observed_ops.update(s['op'] for s in t['specs'])
                    recipe_sizes[k][seq]=min(t['bytes'],recipe_sizes[k].get(seq,t['bytes']))
    assert set(rows)==set(cols)
    flat=[];ablations=[]
    for k,c in cols.items():
        candidates=rows[k]+reductions[k]
        selected,selected_run=min(candidates,key=lambda x:(x[0]['best']['pure']['bytes'],x[0]['best']['pure']['depth']))
        pure=selected['best']['pure']
        assert set(pure['terminals'])<=PURE
        info=verify(pure,c['raw'],c['kind'])
        old=rows[k][0][0]['historical_best']
        assert all(r['historical_best']==old for r,_ in rows[k])
        assert sha(old['frame_path'])==old['frame_sha256'] and Path(old['frame_path']).stat().st_size==old['compressed_bytes']
        algo_r,algo_run=min(rows[k],key=lambda x:x[0]['best']['algorithm']['bytes'])
        algorithm=algo_r['best']['algorithm'];verify(algorithm,c['raw'],c['kind'])
        hybrid_r,hybrid_run=min(rows[k],key=lambda x:x[0]['best']['hybrid']['bytes'])
        hybrid=hybrid_r['best']['hybrid'];verify(hybrid,c['raw'],c['kind'])
        target=min(old['compressed_bytes'],algorithm['bytes'])
        flat.append({'key':k,'dataset':c['dataset'],'column':c['name'],'raw_bytes':len(c['raw']),
            'historical_algorithm':old.get('algorithm') or old.get('method'),'historical_ids':json.dumps(old['selected_ids']),
            'historical_bytes':old['compressed_bytes'],'historical_CR':len(c['raw'])/old['compressed_bytes'],
            'algorithm_chain':label(algorithm),'algorithm_bytes':algorithm['bytes'],'algorithm_CR':len(c['raw'])/algorithm['bytes'],
            'stronger_control_bytes':target,'stronger_control_CR':len(c['raw'])/target,
            'pure_chain':label(pure),'pure_bytes':pure['bytes'],'pure_CR':len(c['raw'])/pure['bytes'],
            'saved_bytes':target-pure['bytes'],'saved_percent':100*(target-pure['bytes'])/target,
            'pure_depth':pure['depth'],'pure_compress_ms':1000*pure['compress_sec'],
            'hybrid_chain':label(hybrid),'hybrid_bytes':hybrid['bytes'],'hybrid_CR':len(c['raw'])/hybrid['bytes'],
            'pure_beats_hybrid':pure['bytes']<hybrid['bytes'],
            'actual_pure_attempts_including_remeasure':measured[k],'actual_pure_successes_including_remeasure':successful[k],
            'unique_pure_recipes':len(logical[k]),'unique_successful_pure_recipes':len(oklogical[k]),
            'strict_winning_pure_recipes':sum(size<target for size in recipe_sizes[k].values()),
            'strict_winning_multistep_pure_recipes':sum(size<target and sum(map(len,json.loads(seq)))>=2 for seq,size in recipe_sizes[k].items()),
            'screened_proposals_not_compression_tests':screen[k],
            'search_and_ablation_worker_seconds':times[k],
            'winner_source_run':str(selected_run),'winner_frame':str(path_abs(pure['frame_path'])),
            'winner_frame_sha256':pure['frame_sha256'],'input_sha256':c['input_sha256'],
            'frame_format':info['format'],'frame_header_bytes':info['header_bytes'],
            'metadata_bytes_before_explicit_entropy':info['metadata_bytes'],'payload_bytes_before_explicit_entropy':info['payload_bytes']})
        for rr,rd in reductions[k]:
            for line in (rd/'traces'/f'{k}.jsonl').read_text().splitlines():
                t=json.loads(line)
                ablations.append({'key':k,'column':c['name'],'round':t['round'],'removed':str(t['removed_stage']),
                    'parent_bytes':t['parent_bytes'],'status':t['status'],'bytes':t.get('bytes'),
                    'delta_bytes':t.get('delta_bytes'),'error':t.get('error',''),'source_run':str(rd)})
    summary={'status':'COMPLETE','scope':'20 same typed numeric columns; adaptive offline development, not held-out/RL',
        'pure_wins_vs_both_controls':sum(r['saved_bytes']>0 for r in flat),'columns_count':len(flat),
        'pure_wins_vs_hybrid':sum(r['pure_beats_hybrid'] for r in flat),
        'minimum_winning_multistep_recipes_per_column':min(r['strict_winning_multistep_pure_recipes'] for r in flat),
        'strict_winning_multistep_recipes_total':sum(r['strict_winning_multistep_pure_recipes'] for r in flat),
        'pure_encoding_attempts_including_remeasure':sum(measured.values()),'pure_encoding_successes_including_remeasure':sum(successful.values()),
        'unique_column_recipe_candidates':sum(map(len,logical.values())),
        'unique_successful_column_recipes':sum(map(len,oklogical.values())),
        'unique_multistep_pure_candidates':sum(sum(sum(map(len,json.loads(s)))>=2 for s in keys) for keys in logical.values()),
        'unique_successful_multistep_pure_candidates':sum(sum(sum(map(len,json.loads(s)))>=2 for s in keys) for keys in oklogical.values()),
        'maximum_measured_depth':max(d for counter in depth.values() for d in counter),
        'source_runs':provenance,'all_selected_winners_byte_exact_cold_replayed':True,
        'model_training_updates':0,'semantic_audit':'pending; see separate fresh-agent audit',
        'implemented_transform_families':list(SUPPORTED_OPERATORS),
        'measured_transform_families':sorted(observed_ops),
        'implemented_but_not_formally_measured':sorted(set(SUPPORTED_OPERATORS)-observed_ops),
        'columns':flat}
    audit_file=ROOT/'refine-logs/primitive_composition_research_20260906_v1/INDEPENDENT_AUDIT_v2.json'
    if audit_file.is_file():
        summary['semantic_audit']={'status':'reported_separately','path':str(audit_file),
            'sha256':sha(audit_file),'review_independence':'same-family-fresh-context',
            'acceptance_status':'provisional','note':'See reviewer-owned verdict and remaining warnings in the linked report; this exporter does not promote semantic acceptance.'}
    latest=a.output.resolve();latest.mkdir(parents=True,exist_ok=True)
    dest=latest/('snapshot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));dest.mkdir()
    (dest/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name,data in [('per_column.csv',flat),('deletion_ablation.csv',ablations)]:
        if not data:continue
        with (dest/name).open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    intro=[f"真正独立算子组合：{summary['pure_wins_vs_both_controls']}/20 列严格超过历史最佳和新同封装算法组合两个控制。",
        f"共实际纯组尝试 {summary['pure_encoding_attempts_including_remeasure']:,} 次（含跨轮重测、失败消融）；跨轮按列+参数链去重 {summary['unique_column_recipe_candidates']:,} 条，其中成功 {summary['unique_successful_column_recipes']:,} 条。筛选代理值不计，smoke不计入主结果。",
        f"排除零步/单步候选后，深度≥2的不同纯组合为 {summary['unique_multistep_pure_candidates']:,} 条，其中成功 {summary['unique_successful_multistep_pure_candidates']:,} 条。另有1000条补充旧原语排列probe，全部成功但没有改进，单独保留，不混入上述主计数。",
        f"真正严格超过双控制的不同多步参数链共 {summary['strict_winning_multistep_recipes_total']:,} 条，每列至少 {summary['minimum_winning_multistep_recipes_per_column']} 条。不是所有成功无损候选都更优，也不声称这些参数链彼此语义不等价或全部不可删减。",
        '纯组：显式数值变换、布局/打包、独立算术编码；不使用Bzip2/XZ/Zstd/Brotli/zlib完整codec。算法控制：5种固定参数完整codec的单步和两级联共30种，另保留历史206实现最强结果。',
        'CR=原始完整int64/float64字节÷完整可独立解码文件字节，越大越好。掩码、字典、异常值、模型规则、校验、外头全部计入；不是CSV原文压缩。',
        f"重要限制：已实现{len(SUPPORTED_OPERATORS)}类独立变换/布局原语，其中{len(observed_ops)}类进入本轮正式完整帧测量；另有 {', '.join(sorted(set(SUPPORTED_OPERATORS)-observed_ops)) or '无'} 仅有实现/合成单测覆盖。本轮配合8种独立熵模型有限搜索，不是全部206算子的任意排列穷举；跨轮链条去重不代表其输出彼此不等价。深度最多10。",
        '这轮没有训练GRPO/PPO。数据指导了新算子设计，因此只能说明当前20列的可达结果，不能证明RL收益或未见数据泛化，也不能保证任何列必然获益。',
        f"混合组=真实原语后接完整codec，明确另列。纯组只在 {summary['pure_wins_vs_hybrid']}/20 列严格超过混合组；不能把超过算法组合说成超过所有已测试方案。",
        '计时：胜者为无prepared缓存的三次冷编码中位数；不同轮在同机不同并行负载下测量，不能视为严谨的速度排名。搜索秒数含编码/逆验证/筛选，是累计worker时间，不是整体墙钟。']
    md=['# 独立算子组合研究结果','']
    md+=sum(([p,''] for p in intro),[])
    web=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>独立算子组合研究</title>',
        '<style>body{font:16px system-ui;color:#203447;margin:32px}h1{font-size:30px}p{line-height:1.65;max-width:1400px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #cad7df;text-align:left}th{background:#203447;color:#fff}.win{background:#e7f4ed}.loss{background:#fff0dc}td.chain{min-width:270px}small{color:#536776}</style>',
        '<h1>独立算子组合研究结果</h1>',*['<p>'+html.escape(p)+'</p>' for p in intro]]
    for dataset in sorted({r['dataset'] for r in flat}):
        md += [f'## {dataset}','','| 列 | 更强算法控制 B / CR | 纯算子 B / CR | 节省 | 纯算子组合 | 混合 B | 去重纯候选 / 胜出多步链 |','|---|---:|---:|---:|---|---:|---:|']
        web.append('<h2>'+dataset+'</h2><table><tr><th>列</th><th>历史最佳 B / CR</th><th>同封装算法组合 B / CR</th><th>純算子组合</th><th>纯算子 B / CR</th><th>相对更强控制节省</th><th>混合 B</th><th>去重纯候选 / 胜出多步链</th><th>编码 ms</th></tr>')
        for r in flat:
            if r['dataset']!=dataset:continue
            md.append(f"| {r['column']} | {r['stronger_control_bytes']:,} / {r['stronger_control_CR']:.4f}× | {r['pure_bytes']:,} / {r['pure_CR']:.4f}× | {r['saved_bytes']:+,} B ({r['saved_percent']:+.2f}%) | {r['pure_chain']} | {r['hybrid_bytes']:,} | {r['unique_pure_recipes']:,} / {r['strict_winning_multistep_pure_recipes']:,} |")
            cells=[r['column'],f"{r['historical_bytes']:,} / {r['historical_CR']:.4f}×",f"{r['algorithm_chain']}\n{r['algorithm_bytes']:,} / {r['algorithm_CR']:.4f}×",r['pure_chain'],f"{r['pure_bytes']:,} / {r['pure_CR']:.4f}×",f"{r['saved_bytes']:+,} B ({r['saved_percent']:+.2f}%)",f"{r['hybrid_bytes']:,}",f"{r['unique_pure_recipes']:,} / {r['strict_winning_multistep_pure_recipes']:,}",f"{r['pure_compress_ms']:.3f}"]
            web.append('<tr class="'+('win' if r['saved_bytes']>0 else 'loss')+'">'+''.join('<td>'+html.escape(str(x)).replace('\n','<br>')+'</td>' for x in cells)+'</tr>')
        web.append('</table>');md.append('')
    md += ['## 如何理解算子贡献','','- split_common：只取出最高频位模式并保存位置mask；其余数值才送后续delta。不是把mask免费丢给解码器。',
        '- delta：当前整数/64位表示上的可逆模2^64差分；浮点未经float_scale时操作的是位模式，不是有舍入的浮点相减。zigzag：把有符号表示映射到非负整数，本身不压缩。compact/bitpack：按实际位宽写出；转置：集中相似位/字节；算术编码：利用剩余统计规律。',
        '- 删除消融见 deletion_ablation.csv。负delta_bytes表示去掉该步骤更好，已允许进一步简化；类型非法的删除明确列为invalid，没有偷偷补回必需操作。',
        '- 时间、每列完整chain/CR/大小/sha/来源、混合方案、跨轮重测与去重计数在 per_column.csv。每个源run保留完整trace、代码快照和可逆文件。',
        '- 语义审计状态以独立审计文件为准；同模型家族审计只能是provisional。','']
    no2_winner=next(r for r in flat if r['key']=='Air-Quality__08')
    no2=[t for t in ablations if t['key']=='Air-Quality__08' and t['parent_bytes']==no2_winner['pure_bytes'] and t['round']==0]
    if no2:
        mechanism=f"原链：{no2_winner['pure_chain']}，完整文件 {no2_winner['pure_bytes']:,} B。算法组合控制为 {no2_winner['stronger_control_bytes']:,} B。以下每条均重新编码并逐字节恢复，其他步骤保持不变。"
        md += ['## NO2：不是算法换名字，而是拆除一步就能检验的组合','',mechanism,'','| 删除的步骤 | 完整文件 B | 相比原链多 B |','|---|---:|---:|']
        web.append('<h2>NO2：逐个删除，检查组合是否真的有贡献</h2><p>'+html.escape(mechanism)+'</p><table><tr><th>删除步骤</th><th>完整文件 B</th><th>比原链多 B</th></tr>')
        for t in no2:
            md.append(f"| {t['removed']} | {t['bytes']:,} | {t['delta_bytes']:+,} |")
            web.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [t['removed'],f"{t['bytes']:,}",f"{t['delta_bytes']:+,}"])+'</tr>')
        md += ['','这些是当前列、当前前后文下的边际贡献，不是“任何数据上A后面都必须接B”的规则。ZigZag仍然等长；它通过改变后续位模式分布帮助压缩。','']
        web.append('</table><p>这是当前数据与前后文下的贡献，不是普遍的“必须相邻”规则。ZigZag自身仍等长，收益来自后续位模式与熵编码。</p>')
    (dest/'REPORT.md').write_text('\n'.join(md))
    web.append('<p>逐步删除消融、完整时间与文件校验见 <a href="per_column.csv">逐列表</a>、<a href="deletion_ablation.csv">删除消融</a>、<a href="REPORT.md">完整说明</a>。独立语义审计另存，未默认为通过。</p></html>')
    (dest/'REPORT.html').write_text('\n'.join(web))
    for f in dest.iterdir():shutil.copy2(f,latest/f.name)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('columns','source_runs')},ensure_ascii=False))

if __name__=='__main__':main()
