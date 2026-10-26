#!/usr/bin/env python3
"""Aggregate paired native measurements per source/column, retaining failures."""
from __future__ import annotations
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

import native_prepared_cohort_v1 as data
import native_frozen_evaluation_v1 as frozen

METHODS={'model':'GRPO_C++_total','beam30':'Beam30_C++_total',
         'no_target_guidance':'GRPO_no_target_guidance_C++_total',
         'no_reuse':'GRPO_no_reuse_C++_total'}
METRICS=('compression_ratio','compression_ns_per_byte','decompression_ns_per_byte')


def model_measurements(directory:Path):
    identity=json.loads((directory/'run_identity.json').read_bytes())
    receipt=frozen.verify_receipt(directory,identity)
    data.require(receipt['split']=='test','comparison requires held-out model test')
    measured=[]
    for file in receipt['worker_files']:
        path=directory/file['path'];data.require(data.digest(path.read_bytes())==file['sha256'],'model test rows changed')
        for line in path.read_text().splitlines():
            measured.append(json.loads(line))
    return measured


def model_columns(directory:Path):
    return aggregate_model_rows(model_measurements(directory))


def chosen_sample(row):
    samples=row.get('samples',[])
    return samples[row['representative_sample_index']] if samples else None


def aggregate_model_rows(measured):
    groups=collections.defaultdict(list)
    for row in measured:
        groups[row['dataset_id'],row['column'],row['weight_index']].append(row)
    rows=[]
    for (dataset,column,wi),members in sorted(groups.items()):
        raw=sum(r['original_numeric_bytes'] for r in members)
        statuses=collections.Counter(r['status'] for r in members)
        valid=[r for r in members if r['status']=='ok']
        complete=statuses['ok']==len(members)
        variant=members[0].get('inference_variant','model')
        data.require(variant in METHODS and all(r.get('inference_variant','model')==variant for r in members),
                     'mixed inference variants within one column')
        row=dict(dataset_id=dataset,column=column,weight_index=wi,weight=members[0]['weight'],
            method=METHODS[variant],inference_variant=variant,
            implementation_class='native_beam_search_and_complete_graph' if variant=='beam30' else 'model_prediction_and_complete_graph',
            primary_comparison=True,block_count=len(members),entity_count=len({r['entity_id'] for r in members}),
            original_numeric_bytes=raw,status='N/A' if raw==0 else 'ok' if complete else 'failed_or_partial',
            block_status_counts=dict(statuses),compression_ratio=None,compression_ns_per_byte=None,decompression_ns_per_byte=None,
            failure_reasons=dict(collections.Counter(r.get('error',r.get('reason','unknown')) for r in members if r['status']!='ok')),
            recipe_usage=[{'recipe_action_ids':list(recipe),'block_count':count}
                for recipe,count in sorted(collections.Counter(tuple(r.get('recipe_action_ids',[])) for r in valid).items())])
        if complete and raw:
            row.update(frame_bytes=sum(r['frame_bytes'] for r in valid),
                compression_time_ns=sum(r['compression_time_ns'] for r in valid),
                decompression_time_ns=sum(r['decompression_time_ns'] for r in valid))
            row.update(compression_ratio=row['frame_bytes']/raw,
                compression_ns_per_byte=row['compression_time_ns']/raw,
                decompression_ns_per_byte=row['decompression_time_ns']/raw)
            selected=[chosen_sample(r) for r in valid]
            if all(sample is not None for sample in selected):
                prediction=[s['timing']['total_prediction_ns'] for s in selected]
                normalized=[v/r['original_numeric_bytes'] for v,r in zip(prediction,valid)]
                # Nearest-rank P95 on complete chosen block observations.
                p95=lambda values:sorted(values)[max(0,math.ceil(.95*len(values))-1)]
                row.update(prediction_time_ns=sum(prediction),prediction_ns_per_byte=sum(prediction)/raw,
                    prediction_block_mean_ns=statistics.mean(prediction),prediction_block_p95_ns=p95(prediction),
                    prediction_block_mean_ns_per_byte=statistics.mean(normalized),
                    prediction_block_p95_ns_per_byte=p95(normalized),
                    evaluation_count=sum(s['evaluations'] for s in selected),
                    certified_pruned_states=sum(s['certified_pruned_states'] for s in selected),
                    budget_unexplored_actions=sum(s['budget_unexplored_actions'] for s in selected),
                    subtiming_ns={key:sum(s['timing'][key] for s in selected) for key in selected[0]['timing']})
        rows.append(row)
    return rows


def amortization_rows(measured):
    groups=collections.defaultdict(list)
    for row in measured:
        groups[row.get('inference_variant','model'),row['dataset_id'],row['entity_id'],row['column'],row['weight_index']].append(row)
    result=[]
    for (variant,dataset,entity,column,wi),members in sorted(groups.items()):
        raw=packed=encode=decode=prediction=0;complete=True
        for count,row in enumerate(sorted(members,key=lambda r:(r['block_index'],r['block_id'])),1):
            raw+=row['original_numeric_bytes'];complete=complete and row['status']=='ok'
            sample=chosen_sample(row)
            if row['status']=='ok':
                packed+=row['frame_bytes'];encode+=row['compression_time_ns'];decode+=row['decompression_time_ns']
                if sample is not None:prediction+=sample['timing']['total_prediction_ns']
            item=dict(dataset_id=dataset,entity_id=entity,column=column,weight_index=wi,
                method=METHODS[variant],processed_blocks=count,
                through_block_id=row['block_id'],cumulative_original_numeric_bytes=raw,
                status='N/A' if raw==0 else 'ok' if complete else 'failed_or_partial',
                compression_ratio=None,compression_ns_per_byte=None,decompression_ns_per_byte=None,
                prediction_ns_per_byte=None)
            if complete and raw:
                item.update(compression_ratio=packed/raw,compression_ns_per_byte=encode/raw,
                    decompression_ns_per_byte=decode/raw,prediction_ns_per_byte=prediction/raw,
                    cumulative_frame_bytes=packed,cumulative_compression_ns=encode,
                    cumulative_decompression_ns=decode,cumulative_prediction_ns=prediction)
            result.append(item)
    return result


def win_tie_loss(model,baseline):
    index={(r['dataset_id'],r['column'],r['method']):r for r in baseline}
    labels=sorted({r['method'] for r in baseline});groups=collections.defaultdict(collections.Counter)
    primary={r['method']:r['primary_comparison'] for r in baseline}
    for row in model:
        for label in labels:
            other=index.get((row['dataset_id'],row['column'],label))
            for metric in METRICS:
                count=groups[row['method'],row['weight_index'],label,metric]
                if row['status']!='ok' or other is None or other['status']!='ok':
                    count['not_comparable']+=1;continue
                left,right=row[metric],other[metric]
                if left is None or right is None:count['not_comparable']+=1;continue
                outcome='tie' if math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12) else 'win' if left<right else 'loss'
                count[outcome]+=1
    return [dict(method=method,weight_index=wi,baseline=label,metric=metric,
        primary_comparison=primary[label],win=count['win'],tie=count['tie'],loss=count['loss'],
        not_comparable=count['not_comparable'],interpretation='numerical comparison; not statistical significance')
        for (method,wi,label,metric),count in sorted(groups.items())]


def write_amortization_plots(curves,output:Path):
    """Export one auditable longest-sequence example per source/column.

    The full per-entity measurements remain in amortization.json. Selecting by
    input block count, before considering timings, avoids selecting a favorable
    measured curve. A one-block source is explicitly a single-point example.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import html
    grouped=collections.defaultdict(list)
    for row in curves:grouped[row['dataset_id'],row['column']].append(row)
    folder=output/'amortization_plots';folder.mkdir(exist_ok=True)
    index=[];records=[]
    for (dataset,column),members in sorted(grouped.items()):
        lengths=collections.defaultdict(int)
        for row in members:lengths[row['entity_id']]=max(lengths[row['entity_id']],row['processed_blocks'])
        entity=min(lengths,key=lambda value:(-lengths[value],value))
        selected=[r for r in members if r['entity_id']==entity]
        fig,axes=plt.subplots(2,3,figsize=(12,6),squeeze=False)
        for wi,axis in enumerate(axes.flat):
            for method in METHODS.values():
                points=sorted((r for r in selected if r['weight_index']==wi and r['method']==method),
                              key=lambda r:r['processed_blocks'])
                valid=[r for r in points if r['status']=='ok']
                if valid:axis.plot([r['processed_blocks'] for r in valid],
                    [r['compression_ns_per_byte'] for r in valid],marker='o',markersize=2,label=method)
            axis.set_title(('100','010','001','550','505','433')[wi])
            axis.set_xlabel('Processed blocks');axis.set_ylabel('Cumulative compression ns / B_raw')
            axis.grid(alpha=.2)
        handles,labels=axes[0][0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=7)
        fig.suptitle(f'{dataset} / {column}\nLongest input sequence example; {lengths[entity]} blocks',fontsize=10)
        fig.tight_layout(rect=(0,.08,1,.93))
        stem=data.digest(data.canonical([dataset,column]))[:16]
        path=folder/(stem+'.svg');fig.savefig(path);plt.close(fig)
        record=dict(dataset_id=dataset,column=column,entity_id=entity,block_count=lengths[entity],
            selection='maximum input block count; ties by entity id; independent of measured performance',
            path=str(path.relative_to(output)),sha256=data.digest(path.read_bytes()))
        records.append(record)
        index.append('<h2>'+html.escape(dataset+' / '+column)+'</h2><p>Entity: '+html.escape(entity)+
            '</p><img style="max-width:100%" src="'+html.escape(path.name)+'" alt="Cumulative compression cost">')
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Amortization examples</title>'
        '<h1>Amortization examples</h1><p>One longest input sequence per source and numeric column. '
        'All entities are retained in ../amortization.json. Single-block inputs have no observed multi-block amortization.</p>'+
        ''.join(index))
    (output/'amortization_plot_manifest.json').write_bytes(data.canonical(records)+b'\n')


def validate_pairing(model_dir:Path,baseline_dir:Path,model,baseline):
    m=json.loads((model_dir/'receipt.json').read_bytes())
    b=json.loads((baseline_dir/'receipt.json').read_bytes())
    data.require(b['status']=='complete' and b['manifest_count']==16,'baseline test is incomplete')
    data.require(b['corpus']['role']=='test_candidate' and b['corpus']['corpus_fingerprint']==m['corpus_fingerprint'],
                 'model/baseline test cohort differs')
    data.require(len(m['cpus'])==1 and b['timing_contract']['available_cpu_affinity']==m['cpus'] and
                 b['timing_contract']['manifest_execution']=='serial' and b['timing_contract']['threads']==1,
                 'model/baseline timing concurrency or CPU differs')
    data.require((b['timing_contract']['warmups'],b['timing_contract']['repeats'])==(m['warmups'],m['repeats']),
                 'model/baseline repetitions differ')
    inventory=json.loads(Path(b['inventory']).read_bytes())
    data.require(data.digest(data.canonical(inventory))==b['inventory_sha256'],'baseline inventory changed')
    labels={r['label'] for r in inventory['methods']}
    data.require(len(labels)==86,'baseline label inventory incomplete')
    for row in b['outputs'].values():
        data.require(data.digest(Path(row['path']).read_bytes())==row['sha256'],'baseline summary artifact changed')
    data.require('frames_json' in b['outputs'] and
                 Path(b['outputs']['frames_json']['path']).resolve()==(baseline_dir/'frames.json').resolve(),
                 'baseline complete-frame index is missing')
    frames=json.loads((baseline_dir/'frames.json').read_bytes())
    indexed={row['path']:row for row in frames['frames']}
    data.require(len(indexed)==len(frames['frames'])==frames['frame_count']==b['complete_frame_count'],
                 'baseline complete-frame coverage changed')
    data.require(sum(row['bytes'] for row in indexed.values())==frames['frame_bytes']==b['complete_frame_bytes_on_disk'],
                 'baseline complete-frame byte sum changed')
    for relative,row in indexed.items():
        expected='frames/'+row['sha256'][:2]+'/'+row['sha256']+'.ngf1'
        path=baseline_dir/relative
        data.require(relative==expected and path.resolve().is_relative_to(baseline_dir.resolve()) and path.is_file(),
                     'baseline complete-frame path changed or missing')
        data.require(path.stat().st_size==row['bytes'] and data.digest(path.read_bytes())==row['sha256'],
                     'baseline complete-frame checksum or length changed')
    data.require(len(b['manifest_receipts'])==16 and
                 len({r['dataset_id'] for r in b['manifest_receipts']})==16,'baseline manifest coverage differs')
    raw_count=0;referenced=set()
    for row in b['manifest_receipts']:
        path=Path(row['raw_jsonl'])
        data.require(data.digest(path.read_bytes())==row['raw_jsonl_sha256'],'baseline raw measurements changed')
        count=0
        with path.open() as f:
            for line in f:
                if not line.strip():continue
                count+=1;measured=json.loads(line)
                if measured.get('status') in ('ok','N/A'):
                    relative='frames/'+measured['complete_frame_relative_path'];referenced.add(relative)
                    data.require(relative in indexed and indexed[relative]['sha256']==measured['complete_frame_sha256'] and
                        indexed[relative]['bytes']==measured['complete_frame_bytes'],'baseline row/frame identity changed')
        data.require(count==row['rows'],'baseline raw row count changed');raw_count+=count
    data.require(raw_count==b['raw_row_count'],'baseline raw coverage changed')
    data.require(referenced==set(indexed),'baseline frame index omits or adds unreferenced frames')
    mm=collections.defaultdict(dict);bb=collections.defaultdict(dict)
    for row in model:
        key=(row['dataset_id'],row['column']);wi=row['weight_index']
        data.require(wi not in mm[key],'duplicate model column/weight');mm[key][wi]=row
    for row in baseline:
        key=(row['dataset_id'],row['column']);label=row['method']
        data.require(label not in bb[key],'duplicate baseline column/method');bb[key][label]=row
    data.require(set(mm)==set(bb) and len({k[0] for k in mm})==16,'model/baseline dataset or column coverage differs')
    for key in mm:
        data.require(set(mm[key])==set(range(6)) and set(bb[key])==labels,'missing weight or method in column comparison')
        reference=mm[key][0]
        for row in [*mm[key].values(),*bb[key].values()]:
            data.require(all(row[f]==reference[f] for f in ('original_numeric_bytes','block_count','entity_count')),
                         'model/baseline original bytes or input coverage differs for '+str(key))
    data.require(sum(r['block_count'] for r in model)==m['pair_count']==m['run_identity']['expected_pair_count'],
                 'model pair coverage changed')
    return {'schema':'native-smallest32-comparison-pairing-v1','passed':True,
        'model_receipt_sha256':data.digest((model_dir/'receipt.json').read_bytes()),
        'baseline_receipt_sha256':data.digest((baseline_dir/'receipt.json').read_bytes()),
        'corpus_fingerprint':m['corpus_fingerprint'],'datasets':16,'numeric_columns':len(mm),
        'weights':6,'baseline_labels':86,'cpus':m['cpus'],'measurement_concurrency':1,
        'same_original_input_coverage':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--variant',type=Path,action='append',default=[],help='Additional completed inference variant directory')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    measured=model_measurements(a.model);model=aggregate_model_rows(measured)
    curves=amortization_rows(measured)
    del measured
    original=json.loads((a.model/'run_identity.json').read_bytes())
    data.require(original['inference_variant']=='model','--model must be the main frozen policy evaluation')
    # Baseline aggregator uses the same exact source/column group and denominator.
    baseline=json.loads((a.baseline/'columns.json').read_bytes())
    if isinstance(baseline,dict):baseline=baseline['columns']
    paired=validate_pairing(a.model,a.baseline,model,baseline)
    pairings=[paired]
    seen={'model'}
    for directory in a.variant:
        identity=json.loads((directory/'run_identity.json').read_bytes())
        variant=identity['inference_variant']
        data.require(variant in METHODS and variant not in seen,'duplicate or unknown comparison variant')
        data.require(all(identity[k]==original[k] for k in ('model_sha256','native_cli_sha256','catalog_sha256',
            'library_sha256','reference_sha256','cold_cli_sha256','driver_sha256','experiment_scope')),'inference variant changed frozen inputs/runtime')
        extra=model_measurements(directory);columns=aggregate_model_rows(extra)
        pairings.append({**validate_pairing(directory,a.baseline,columns,baseline),'inference_variant':variant})
        curves.extend(amortization_rows(extra));model.extend(columns);seen.add(variant)
        del extra
    rows=model+baseline
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'pairing_receipt.json').write_bytes(data.canonical(paired)+b'\n')
    (a.output/'variant_pairing_receipts.json').write_bytes(data.canonical(pairings)+b'\n')
    (a.output/'all_columns.json').write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
    (a.output/'amortization.json').write_bytes(data.canonical(curves)+b'\n')
    (a.output/'win_tie_loss.json').write_bytes(data.canonical(win_tie_loss(model,baseline))+b'\n')
    write_amortization_plots(curves,a.output)
    fields=['dataset_id','column','weight_index','method','implementation_class','primary_comparison',
        'status','block_count','entity_count','original_numeric_bytes','frame_bytes','compression_time_ns',
        'decompression_time_ns','compression_ratio','compression_ns_per_byte','decompression_ns_per_byte',
        'prediction_time_ns','prediction_ns_per_byte','prediction_block_mean_ns','prediction_block_p95_ns',
        'prediction_block_mean_ns_per_byte','prediction_block_p95_ns_per_byte','evaluation_count',
        'certified_pruned_states','budget_unexplored_actions']
    with (a.output/'all_columns.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    counts=collections.Counter(r['status'] for r in rows)
    note=('模型方法的压缩时间包含 C++ 特征提取、冻结模型预测、全部候选评价、图调度和最终完整帧编码。'
          '86方法按各自的实际原生实现测量完整帧编码和解码；代理、共享实现及未接入项保留各自标识。\n\n'
          '逐列指标采用完整帧字节之和/B_raw之和，以及各块代表耗时之和/B_raw之和。'
          '零数值字节为N/A；存在失败或不支持块的列不输出完整覆盖主指标。'
          '不把事后最佳算法称作免费选择，不从不完整行计算速度优势。\n\n'
          'amortization.json 按来源/实体/列/权重保留累计块曲线；win_tie_loss.json 是逐列数值比较，'
          '不代表统计显著性。预测耗时的 P95 使用各块完整代表观测的 nearest-rank 分位数。'
          'no_reuse 每块重新从训练组合库出发，显式使用64次预算；其余变体首块64、后续16。\n\n'
          f'结果行数：{len(rows)}；状态计数：{dict(counts)}。\n')
    (a.output/'REPORT.md').write_text(note)
    print(json.dumps({'columns':len(rows),'status_counts':dict(counts),'output':str(a.output)}))

if __name__=='__main__':main()
