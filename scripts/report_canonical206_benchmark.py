#!/usr/bin/env python3
"""Audit completed outputs and write a compact Chinese experiment report."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path
from canonical206_column_tables import build_column_tables
from canonical206_fixed_tables import build_fixed_tables


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):d.update(chunk)
    return d.hexdigest()


def close(a,b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
        float(a),float(b),rel_tol=1e-10,abs_tol=1e-9)


def row_stem(r):
    return f"{r['method']}__seed{r['seed']}__{r['dataset']}__{r['numeric_index']:02d}"


def audit_column_rows(out,rows,cols,config):
    """Check recorded byte/timing arithmetic without rerunning any codec."""
    failures=[]
    for r in rows:
        label=r['frame_path'];c=cols[(r['dataset'],r['numeric_index'])]
        frame_path=out/label
        if (not r['lossless'] or r['input_sha256']!=c['input_sha256'] or
            r['original_bytes']!=c['raw_bytes'] or r['values']!=c['count'] or
            r['kind']!=c['kind'] or frame_path.stat().st_size!=r['compressed_bytes'] or
            sha(frame_path)!=r['frame_sha256']):
            failures.append(f"frame/input {label}")
        with frame_path.open('rb') as f:header=f.read(18)
        if len(header)!=18:
            failures.append(f"truncated frame header {label}")
        else:
            magic,kind,layers,count,_crc=struct.unpack('<4sBBQI',header)
            if (magic!=b'C206' or kind!={'int64':1,'float64':2}[r['kind']] or
                count!=r['values'] or (not r['selected_ids'] and layers!=0) or
                (r['selected_ids'] and not 1<=layers<=len(r['selected_ids']))):
                failures.append(f"frame header {label}")
        if (r['original_bytes']<=0 or r['compressed_bytes']<=0 or
            not close(r['compression_ratio'],r['original_bytes']/r['compressed_bytes']) or
            not close(r['compressed_fraction'],r['compressed_bytes']/r['original_bytes'])):
            failures.append(f"ratio arithmetic {label}")
        for field in ('compress_sec','decompress_sec'):
            samples=r[f'{field}_repeats']
            if len(samples)!=config['repeats'] or not samples:
                failures.append(f"{field} repeat count {label}")
            elif (any(not math.isfinite(float(t)) or t<0 for t in samples) or
                  not close(statistics.median(samples),r[field])):
                failures.append(f"{field} median/samples {label}")
        for field in ('compress_sec','decompress_sec','compose_sec','sampled_search_sec','full_rerank_sec'):
            if not math.isfinite(float(r[field])) or r[field]<0:
                failures.append(f"non-finite/negative {field} {label}")
        if not close(r['compose_sec'],r['sampled_search_sec']+r['full_rerank_sec']):
            failures.append(f"compose timing decomposition {label}")
        finalists=r['full_finalists']
        if (any(not math.isfinite(float(f['wall_sec'])) or f['wall_sec']<0 for f in finalists) or
            sum(f['wall_sec'] for f in finalists)>r['full_rerank_sec']+1e-8):
            failures.append(f"full finalist wall time accounting {label}")
        valid=[f for f in finalists if f['status']=='ok']
        raw=[f for f in valid if not f['prefix']]
        if (len(raw)!=1 or raw[0]['frame_bytes']!=r['original_bytes']+18 or
            r['compressed_bytes']>r['original_bytes']+18):
            failures.append(f"raw fallback {label}")
        if not valid:
            failures.append(f"no successful full finalist {label}")
        else:
            best=min(valid,key=lambda f:(f['frame_bytes'],tuple(f['prefix'])))
            if r['compressed_bytes']!=best['frame_bytes'] or r['selected_ids']!=best['prefix']:
                failures.append(f"full finalist selection {label}")
    return failures


def audit_aggregates(rows,seed_rows,summary):
    """Independently reconstruct both CSV levels from the final column JSON."""
    failures=[];groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['method'],r['seed'])].append(r)
    observed={(r['dataset'],r['method'],int(r['seed'])):r for r in seed_rows}
    if set(observed)!=set(groups) or len(observed)!=len(seed_rows):
        failures.append('dataset_seed_results group coverage/duplicates')
    derived={}
    fields=('original_bytes','compressed_bytes','compress_sec','compose_sec','decompress_sec')
    for key,rr in groups.items():
        expected={f:sum(r[f] for r in rr) for f in fields}
        expected.update(columns=len(rr),compression_ratio=expected['original_bytes']/expected['compressed_bytes'],
                        compressed_fraction=expected['compressed_bytes']/expected['original_bytes'],
                        unique_edges=sum(r['search_stats']['unique_edges'] for r in rr),
                        invalid_edges=sum(r['search_stats']['rejected_edges'] for r in rr))
        derived[key]=expected
        if key in observed:
            for field,value in expected.items():
                if not close(observed[key][field],value):
                    failures.append(f'dataset_seed_results {key} {field}')
    groups2=defaultdict(list)
    for (dataset,method,_seed),r in derived.items():groups2[(dataset,method)].append(r)
    observed2={(r['dataset'],r['method']):r for r in summary}
    if set(observed2)!=set(groups2) or len(observed2)!=len(summary):
        failures.append('dataset_summary group coverage/duplicates')
    for key,rr in groups2.items():
        expected={f:statistics.median(r[f] for r in rr)
                  for f in (*fields,'compression_ratio','compressed_fraction','unique_edges','invalid_edges')}
        expected.update(seeds=len(rr),columns_per_seed=min(r['columns'] for r in rr),
                        compressed_bytes_min=min(r['compressed_bytes'] for r in rr),
                        compressed_bytes_max=max(r['compressed_bytes'] for r in rr))
        if key in observed2:
            for field,value in expected.items():
                if not close(observed2[key][field],value):
                    failures.append(f'dataset_summary {key} {field}')
    return failures


def audit_trace(row,edges):
    failures=[];label=row_stem(row);stats=row['search_stats']
    prefixes=[tuple(e['prefix']) for e in edges]
    valid=[e for e in edges if e['status']=='ok']
    rejected=[e for e in edges if e['status']=='rejected']
    if (len(edges)!=stats['unique_edges'] or len(set(prefixes))!=len(edges) or
        len(valid)!=stats['valid_edges'] or len(rejected)!=stats['rejected_edges'] or
        len(valid)+len(rejected)!=len(edges)):
        failures.append(f'trace edge accounting {label}')
    if any(not q or any(not 0<=a<206 for a in q) for q in prefixes):
        failures.append(f'trace action IDs {label}')
    for field in ('forward_sec','inverse_sec'):
        if (any(not math.isfinite(float(e[field])) or e[field]<0 for e in edges) or
            not close(sum(e[field] for e in edges),stats[field])):
            failures.append(f'trace {field} {label}')
    if (set(row['measured_action_ids'])!={a for e in valid for a in e['prefix']} or
        set(row['rejected_action_ids'])!={e['prefix'][-1] for e in rejected}):
        failures.append(f'trace action coverage {label}')
    measured={tuple(e['prefix']) for e in valid}|{()}
    if any(tuple(f['prefix']) not in measured for f in row['full_finalists']):
        failures.append(f'unmeasured full finalist {label}')
    return failures


def summarize_finalist_rejections(rows):
    """Attribute complete finalist attempt time using saved errors, no replay."""
    groups={};details=[];affected_columns=[]
    for r in rows:
        key=(r['dataset'],r['method'],r['seed'])
        if key not in groups:
            groups[key]=dict(dataset=key[0],method=key[1],seed=key[2],
                             full_finalist_count=0,rejected_finalist_count=0,
                             timeout_finalist_count=0,rejected_candidate_wall_sec=0.0,
                             timeout_candidate_wall_sec=0.0)
        group=groups[key]
        group['full_finalist_count']+=len(r['full_finalists'])
        timeout_count=0;timeout_wall=0.0
        for f in r['full_finalists']:
            if f['status']!='rejected':continue
            text=f.get('error','').lower()
            is_timeout='timed out' in text or 'timeoutexpired' in text
            group['rejected_finalist_count']+=1
            group['rejected_candidate_wall_sec']+=f['wall_sec']
            if is_timeout:
                group['timeout_finalist_count']+=1
                group['timeout_candidate_wall_sec']+=f['wall_sec']
                timeout_count+=1;timeout_wall+=f['wall_sec']
            details.append(dict(dataset=r['dataset'],column=r['column'],method=r['method'],
                                seed=r['seed'],prefix=f['prefix'],is_timeout=is_timeout,
                                candidate_wall_sec=f['wall_sec'],error=f.get('error','')))
        if timeout_count:
            affected_columns.append(dict(dataset=r['dataset'],column=r['column'],method=r['method'],
                                         seed=r['seed'],timeout_finalist_count=timeout_count,
                                         timeout_candidate_wall_sec=timeout_wall,
                                         compose_sec=r['compose_sec'],sampled_search_sec=r['sampled_search_sec'],
                                         full_rerank_sec=r['full_rerank_sec']))
    return dict(scope='full-column finalist attempts only; online sample-stage rejections are in traces',
                subprocess_timeout_seconds=30,
                classification='saved error text contains timed out or TimeoutExpired',
                wall_time_definition='whole rejected finalist attempt, including earlier stages and timeout cleanup; already included in compose_sec',
                groups=[groups[k] for k in sorted(groups)],
                worst_timeout_column=max(affected_columns,key=lambda r:r['timeout_candidate_wall_sec']) if affected_columns else None,
                rejected_finalists=details)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    out=args.output.resolve();root=Path(__file__).resolve().parents[1]
    manifest=json.loads((out/'manifest.json').read_text())
    complete=json.loads((out/'COMPLETE.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((out/'columns').glob('*.json'))]
    seeds=manifest['config']['seeds']
    cols={(c['dataset'],c['numeric_index']):c for c in manifest['columns']}
    expected={(d,i,m,s) for d,i in cols for m in ('beam29','grpo','ppo') for s in ([0] if m=='beam29' else seeds)}
    observed={(r['dataset'],r['numeric_index'],r['method'],r['seed']) for r in rows}
    assert observed==expected and len(rows)==len(expected)==complete['actual_rows']
    assert complete['status']=='complete' and complete['expected_rows']==len(expected)
    failures=audit_column_rows(out,rows,cols,manifest['config'])
    with (out/'dataset_summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (out/'dataset_seed_results.csv').open() as f:seed_rows=list(csv.DictReader(f))
    aggregate_failures=audit_aggregates(rows,seed_rows,summary)
    failures.extend(aggregate_failures)
    source_ok=all(sha(root/p)==h for p,h in manifest['source_hashes'].items())
    binary_ok=all(sha(root/'cpp/build'/p)==h for p,h in manifest['runtime_cli_hashes'].items())
    input_files={c['source_path']:c['source_sha256'] for c in manifest['columns']}
    data_ok=all(sha(Path(path))==digest for path,digest in input_files.items())
    training=[]
    datasets={c['dataset'] for c in manifest['columns']}
    expected_training={(d,m,s) for d in datasets for m in ('grpo','ppo') for s in seeds}
    for p in sorted((out/'policies').glob('*.json')):
        record=json.loads(p.read_text())
        assert set(record['training_datasets'])==datasets-{record['heldout_dataset']}
        assert p.with_suffix('.npz').is_file()
        assert [t['iteration'] for t in record['trace']]==list(range(1,manifest['config']['iterations']+1))
        assert all(t['episodes']==manifest['config']['episodes'] and
                   t['cumulative_episodes']==t['iteration']*manifest['config']['episodes']
                   for t in record['trace'])
        assert math.isfinite(record['training_sec']) and record['training_sec']>=0
        training.append(record)
    observed_training={(t['heldout_dataset'],t['method'],t['seed']) for t in training}
    assert observed_training==expected_training and len(training)==len(expected_training)
    attempts=Counter();success=set();selected=set();reject_counts=Counter();examples={}
    row_by_stem={row_stem(r):r for r in rows}
    trace_paths={p.stem:p for p in (out/'traces').glob('*.json')}
    column_stems={p.stem for p in (out/'columns').glob('*.json')}
    assert set(trace_paths)==set(row_by_stem)==column_stems
    trace_failures=[]
    for stem,p in sorted(trace_paths.items()):
        edges=json.loads(p.read_text())
        trace_failures.extend(audit_trace(row_by_stem[stem],edges))
        for edge in edges:
            attempts[edge['prefix'][-1]]+=1
            if edge['status']=='ok':success.update(edge['prefix'])
            else:
                reject_counts[edge['prefix'][-1]]+=1
                examples.setdefault(edge['prefix'][-1],edge['error'])
    failures.extend(trace_failures)
    for r in rows:selected.update(r['selected_ids'])
    grammar=manifest['grammar'];excluded=set(grammar['excluded_camel'])
    finalist_rejections=summarize_finalist_rejections(rows)
    audit=dict(status='PASS' if not failures and source_ok and binary_ok and data_ok else 'FAIL',
               rows=len(rows),expected_rows=len(expected),frames_and_input_checks=not failures,
               failures=failures,python_sources_unchanged=source_ok,cpp_binaries_unchanged=binary_ok,
               full_input_files_unchanged=data_ok,training_runs=len(training),
               frozen_core_python_file_count=len(manifest['source_hashes']),
               recorded_cli_file_count=len(manifest['runtime_cli_hashes']),source_csv_count=len(input_files),
               hash_scope='only manifest-listed core Python files, CLI files, and source CSVs; transitive Python dependencies not covered',
               aggregate_arithmetic_checks=not aggregate_failures,trace_checks=not trace_failures,
               trace_file_count=len(trace_paths),
               registered_actions=206,selectable_actions=203,excluded_ids=sorted(excluded),
               attempted_action_count=len(attempts),successful_action_count=len(success),
               selected_action_count=len(selected),attempted_ids=sorted(attempts),successful_ids=sorted(success),
               selected_ids=sorted(selected),not_attempted_ids=sorted(set(range(206))-set(attempts)-excluded),
               attempted_but_never_successful=sorted(set(attempts)-success),
               full_finalist_rejection_count=sum(g['rejected_finalist_count'] for g in finalist_rejections['groups']),
               full_finalist_timeout_count=sum(g['timeout_finalist_count'] for g in finalist_rejections['groups']))
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    if audit['status']!='PASS':raise AssertionError(audit)
    names={'beam29':'Beam29','ppo':'PPO','grpo':'GRPO（β=0）'}
    report=['# 206 个具体实现的算子组合实验','',
            '使用最小的 Air-Quality、Metro-Traffic 两个完整数据集，共 20 个数值列。原三种搜索方法有 140 条记录，另补测每列最佳固定算法；所有最终结果均精确解码。',
            '', '压缩比 = 原始数值字节 ÷ 压缩后完整文件字节，越大越好。时间越小越好。', '',
            '| 数据集 | 方法 | 压缩比 | 压缩后字节 | 压缩时间（秒） | 算子组合时间（秒） |',
            '|---|---|---:|---:|---:|---:|']
    for r in summary:
        report.append(f"| {r['dataset']} | {names[r['method']]} | {float(r['compression_ratio']):.3f}× | {float(r['compressed_bytes']):,.0f} | {float(r['compress_sec']):.4f} | {float(r['compose_sec']):.2f} |")
    report+=['','PPO/GRPO 为 3 个种子的中位数；各指标单独取中位数，可能不来自同一 seed。数据集时间为逐列时间之和。', '']
    for dataset in sorted({r['dataset'] for r in summary}):
        rr=[r for r in summary if r['dataset']==dataset]
        size_best=min(rr,key=lambda r:float(r['compressed_bytes']))
        encode_best=min(rr,key=lambda r:float(r['compress_sec']))
        search_best=min(rr,key=lambda r:float(r['compose_sec']))
        report.append(f"{dataset}：在 Beam29、GRPO、PPO 三种搜索方法中，本轮压缩后字节最少的是 {names[size_best['method']]}；全列编码最快的是 {names[encode_best['method']]}；组合搜索最快的是 {names[search_best['method']]}。这里不包含后文补测的固定算法基线，也不超出本轮固定配置。")
    report+=['',
             '## 两种时间分别包含什么','',
             '- 压缩时间：选好组合后，对完整列重新编码，1 次预热、5 次新编码取中位数；不包含搜索与解码检查。',
             '- 算子组合时间：特征、Beam/策略计算、真实候选压缩与解码校验、前 8 个候选的全列复核；各方法使用独立缓存。',
             '- 当前都是 C++ CLI 接口的墙钟，包含进程启动、IPC 和 Python 适配，不是纯 C++ 内核耗时。',
             '- 源 CSV 读取不包含在这两项时间中；离线训练时间单独报告。', '',
             '### 全列复核中的拒绝与超时','',
             '当前每次 C++ 子进程调用的超时上限为 30 秒；一条多阶段候选的总墙钟可超过 30 秒。所有被拒绝或超时的候选耗时仍完整计入算子组合时间。下表仅列出发生全列拒绝的 dataset/method/seed 组，未列组为零。超时秒数为整次候选尝试的墙钟合计，包含此前阶段及超时清理；不等同于策略神经网络推理时间。', '',
             '| 数据集 | 方法 | seed | 全列拒绝数 | 其中超时数 | 超时候选墙钟合计（秒） |',
             '|---|---|---:|---:|---:|---:|']
    rejected_groups=[g for g in finalist_rejections['groups'] if g['rejected_finalist_count']]
    for g in rejected_groups:
        report.append(f"| {g['dataset']} | {names[g['method']]} | {g['seed']} | {g['rejected_finalist_count']} | {g['timeout_finalist_count']} | {g['timeout_candidate_wall_sec']:.2f} |")
    if not rejected_groups:report.append('| 全部组 | — | — | 0 | 0 | 0.00 |')
    worst=finalist_rejections['worst_timeout_column']
    if worst:
        report+=['',f"超时候选累计耗时最多的单列是 {worst['dataset']} / {worst['column']}，{names[worst['method']]} seed {worst['seed']}：组合共 {worst['compose_sec']:.2f} 秒，其中样本搜索 {worst['sampled_search_sec']:.2f} 秒、全列复核 {worst['full_rerank_sec']:.2f} 秒；{worst['timeout_finalist_count']} 个超时候选累计 {worst['timeout_candidate_wall_sec']:.2f} 秒。"]
    report+=['','超时只说明该实现对本次输入未在当前调用时限内完成，不能据此认定 codec 本质无效，也不能将这部分时间归因于 RL 推理。逐候选记录见 [full_finalist_timeout_summary.json](full_finalist_timeout_summary.json)。', '',
             '## 训练成本','', '| 测试数据集 | 方法 | 训练数据集 | 训练秒数中位数 | 种子范围（秒） |', '|---|---|---|---:|---:|']
    for dataset in sorted({c['dataset'] for c in manifest['columns']}):
        for method in ('grpo','ppo'):
            rr=[r for r in training if r['heldout_dataset']==dataset and r['method']==method]
            vals=[r['training_sec'] for r in rr]
            report.append(f"| {dataset} | {names[method]} | {', '.join(rr[0]['training_datasets'])} | {statistics.median(vals):.2f} | {min(vals):.2f}–{max(vals):.2f} |")
    report+=['',f"每次训练 {manifest['config']['iterations']}×{manifest['config']['episodes']}={manifest['config']['iterations']*manifest['config']['episodes']} 条轨迹，这是预先固定的试验预算，尚未证明 PPO/GRPO 已收敛。两种 RL 的训练/测试 proposal 预算相同。Beam 固定宽度 29，与 RL 不是匹配实际调用数的对照，因此本轮不支持算法本质优劣或样本效率结论。", '',
             '## 3 个种子的完整结果','', '| 数据集 | 方法 | seed | 压缩字节 | 压缩比 | 压缩秒 | 组合秒 | 在线新执行边数 | 拒绝边数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in seed_rows:
        report.append(f"| {r['dataset']} | {names[r['method']]} | {r['seed']} | {int(r['compressed_bytes']):,} | {float(r['compression_ratio']):.3f} | {float(r['compress_sec']):.4f} | {float(r['compose_sec']):.2f} | {r['unique_edges']} | {r['invalid_edges']} |")
    report+=['','## 算子范围与边界','',
             f"- 注册库 206 个，沿用 Camel 排除设置后可选 203 个；在线搜索实际尝试 {len(attempts)} 个，实现精确逆向的有 {len(success)} 个，最终组合用到 {len(selected)} 个。未尝试或拒绝并不代表实现无用。",
             '- 使用已有阶段连接、可逆的跨链连接、最多两段和 7 步；完整注册库参与候选构建，但并未穷举全部排列。',
             '- 从每列前 2,048 行搜索，随后在完整列上复核前 8 名；可能漏掉采样排名低但全列更好的组合，没有全空间 oracle。',
             '- float64 保持原值和位模式，整数直接读为 int64；20 列实际没有 NaN 被删除。Metro 的 3 个字符串列不在数值压缩范围。',
             '- 大小包括 C206 和全部 UAR1 可解码元数据；不能与旧报告的裸 codec payload 大小混用。',
             '- 206 个实现中仍有 reference/proxy 和残余复合核心；本轮复用当前具体实现，不将方法标签等同于相应上游源码版本。',
             '- 奖励优化字节数，压缩时间和组合时间是测量指标，不是已训练的联合目标。', '']
    registry=json.loads((out/'action_registry.json').read_text())
    report+=build_fixed_tables(out,manifest,registry)
    report+=build_column_tables(rows,manifest,registry,out)
    never_successful=sorted(set(attempts)-success)
    if never_successful:
        registry=json.loads((out/'action_registry.json').read_text())
        action_names={r['policy_action_id']:r['name'] for r in registry['policy_actions']}
        report+=['','## 原搜索记录中未执行成功的实现','',
                 '下面仅统计原 Beam29、GRPO、PPO 的搜索轨迹，不含后来补测的固定完整算法链；这不是对所有输入或所有合法上游状态的否定。', '']
        for action in never_successful:
            report.append(f'- ID {action}，`{action_names[action]}`：{attempts[action]} 次尝试，成功 0 次。示例错误：`{examples[action]}`。')
            recovered=[]
            for q in sorted((out/'fixed_algorithm_baseline_v1'/'candidates').glob('*.json')):
                if any(c['status']=='ok' and action in c['ids'] for c in json.loads(q.read_text())):
                    recovered.append(q.stem)
            if recovered:
                report.append(f'  后续固定完整算法链补测在 {len(recovered)} 列中包含该 ID 且通过完整解码。因此“原搜索成功 0 次”不代表该实现本身不可用，区别在于实际尝试的上游路径。')
        report+=['','两组覆盖率必须分别理解，不能把局部搜索中的拒绝当作实现全局失败。']
    witness=next((r for r in rows if r['method']=='beam29' and r['dataset']=='Metro-Traffic' and r['column']=='snow_1h'),None)
    if witness:
        sample={tuple(e['prefix']):e['frame_bytes'] for e in json.loads(
            (out/'traces'/f'{row_stem(witness)}.json').read_text()) if e['status']=='ok'}
        full={tuple(f['prefix']):f['frame_bytes'] for f in witness['full_finalists'] if f['status']=='ok'}
        paths=[((143,),'ALP-RD 字典（143）'),((79,78),'小数定标 → RLE（79 → 78）')]
        if all(path in sample and path in full for path,_ in paths):
            report+=['','## 一个实际的前缀排名翻转','',
                     'Metro-Traffic / snow_1h：两个候选均经过完整列精确解码检查，大小含相同实验封装。', '',
                     '| 组合 | 前 2,048 行字节 | 完整 48,204 行字节 |', '|---|---:|---:|']
            for path,label in paths:report.append(f'| {label} | {sample[path]:,} | {full[path]:,} |')
            source=cols[(witness['dataset'],witness['numeric_index'])]['source_path']
            with Path(source).open(newline='') as f:
                reader=csv.reader(f);next(reader)
                values=[float(row[witness['source_index']]) for row in reader]
            prefix=values[:2048];suffix=values[2048:]
            nonzero=sum(v!=0 for v in suffix)
            runs=(1+sum(a!=b for a,b in zip(suffix,suffix[1:]))) if suffix else 0
            if prefix and all(v==0 for v in prefix):
                report+=['',f'前 2,048 行全为零；后面 {len(suffix):,} 行仅 {nonzero} 个非零值、{runs} 个连续相同值段。前缀的全零特例不能代表完整列的候选排名，而完整列仍有很长的重复段。',
                         '直观含义：不能只看前缀赢家就结束搜索。这证明本轮全列复核有用，并不证明 RL 已学会预测排名翻转。']
    report+=['','ID 对照见 [冻结的 206 实现注册表](action_registry.json)；逐列 JSON 保存完整算子名称、frame hash、候选记录和全部计时样本。', '',
             '## 验证','',
             f"原三种搜索方法 {len(rows)}/{len(expected)} 条记录完成；全部 frame 大小/hash、输入 hash、压缩比、编码/解码计时、完整列候选选优、trace 对应关系与数据集聚合算术核对通过。冻结的 {len(manifest['source_hashes'])} 个核心 Python 文件、清单记录的 {len(manifest['runtime_cli_hashes'])} 个 CLI 文件及 {len(input_files)} 个源 CSV 的哈希与运行前记录一致；未覆盖所有间接依赖的 Python 文件。详见 [AUDIT.json](AUDIT.json)、[PROTOCOL.md](PROTOCOL.md)。", '',
             '新增固定算法补测的 20/20 个优胜结果另行核对完整候选选优、算法名称与类型分支映射、全部 5 次编码计时、输入/文件哈希及 CSV 转录。原始 140 条搜索记录、压缩文件和数据集汇总不因新增逐列表而改写；新增 CSV 与明细是派生视图。详见 [固定算法审计](fixed_algorithm_baseline_v1/REPORT_AUDIT.json)。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'full_finalist_timeout_summary.json').write_text(json.dumps(finalist_rejections,indent=2,ensure_ascii=False)+'\n')
    (out/'training_summary.json').write_text(json.dumps([{k:v for k,v in t.items() if k!='trace'} for t in training],indent=2)+'\n')
    (out/'action_rejections.json').write_text(json.dumps({str(i):dict(count=n,example=examples[i]) for i,n in reject_counts.most_common()},indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
