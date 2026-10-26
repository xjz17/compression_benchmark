#!/usr/bin/env python3
"""Read-only completed Beam30 audit; never runs training or a codec.

Hash/receipt checks below do not claim a new decode. Cold-decode assertions are
the actual benchmark's fresh-process receipts, explicitly labeled as such.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    summary=read(output/'SUMMARY.json');manifest=read(output/'MANIFEST.json')
    checks=[];issues=[]
    def check(name,value,detail=None):
        checks.append({'name':name,'pass':bool(value),'detail':detail})
        if not value:issues.append(name)
    check('complete_status',summary['status']=='COMPLETE')
    check('identity',summary['experiment_identity']==manifest['experiment_identity'])
    check('all_source_fingerprints_unchanged',all(Path(p).is_file() and sha(p)==h for p,h in manifest['provenance_before']['file_sha256'].items()))
    refs=read(output/'REFERENCES.json')['columns']
    rows=[];runs=[];totals=Counter();cold_receipts=0
    for column in manifest['columns']:
        check(column['key']+':input_hash',sha(column['input_path'])==column['input_sha256'])
        for weight in manifest['config']['weights_tenths']:
            wid='a%02d_b%02d_c%02d'%tuple(weight)
            for scope in manifest['config']['tail_scopes']:
                folder=output/'runs'/column['key']/wid/scope;r=read(folder/'RESULT.json');runs.append(r)
                budget=r['budget'];name=column['key']+':'+wid+':'+scope
                check(name+':width30',budget['width']==30)
                check(name+':cap_count',budget['candidate_attempts']<=budget['budget'] and
                      budget['candidate_attempts']==budget['valid_terminal_requests']+budget['invalid_attempts'])
                check(name+':frontier_limit',all(len(layer['survivors'])<=30 for layer in budget['layers']))
                check(name+':source_ref_exact',r['reference']==refs[column['key']])
                check(name+':trace_hashes',all(sha(p)==h for p,h in r['receipt_sha256'].items()))
                frozen=read(folder/'FROZEN_SELECTION.json')['selected'];selected=r['selected']
                check(name+':frozen_selection',all(selected[k]==frozen[k] for k in ('specs','terminals','sequence','bytes','frame_sha256','cost')))
                check(name+':selected_frame_hash',sha(selected['frame_path'])==selected['frame_sha256'] and Path(selected['frame_path']).stat().st_size==selected['bytes'])
                with gzip.open(folder/'candidates.jsonl.gz','rt') as stream:
                    measurements=[json.loads(line) for line in stream if '"event": "measurement"' in line]
                own={tuple(m['sequence']) for m in measurements}
                pool=read(folder/'FROZEN_REFINEMENT.json')['candidates']
                check(name+':own_candidates_only',all(tuple(x['sequence']) in own for x in pool))
                check(name+':unique_candidate_count',len(own)==r['own_unique_candidates']==budget['valid_terminal_requests'])
                check(name+':sampled_depth_legal',all(m['mixed_coarse_depth']<=10 and m['prefix_depth']<=6 for m in measurements))
                check(name+':scope',scope!='fixed_b4' or all(m['terminals']==['bzip2'] for m in measurements))
                for key in ('candidate_attempts','valid_terminal_requests','invalid_attempts'):totals[key]+=budget[key]
        if manifest['config']['skip_benchmark']:continue
        b=read(output/'benchmark'/column['key']/'RESULT.json');rows+=b['rows']
        frozen=read(output/'benchmark'/column['key']/'FROZEN_SELECTION.json')
        check(column['key']+':final_frozen_aliases',[x['method'] for x in frozen['selections']]==[x['method'] for x in b['rows']])
        recipe_keys=set()
        for r in b['rows']:
            rho=r['bytes']/r['raw_bytes'];ref=refs[column['key']]
            en=r['encode_seconds_median']/(r['encode_seconds_median']+ref['encode_seconds_median'])
            dn=r['decode_seconds_median']/(r['decode_seconds_median']+ref['decode_seconds_median'])
            cost=sum(w*x for w,x in zip(r['weight_tenths'],(rho,en,dn)))/10
            check(column['key']+':'+r['method']+':formula',all(abs(x-y)<1e-12 for x,y in
                  [(rho,r['compression_rate']),(en,r['encode_time_normalized']),(dn,r['decode_time_normalized']),(cost,r['cost'])]))
            check(column['key']+':'+r['method']+':cold_decode_receipt',r['lossless'] and r['deterministic'] and r['cold_decode_pass'])
            key=json.dumps({'specs':r['specs'],'terminals':r['terminals']},sort_keys=True)
            same=[x for x in b['rows'] if x['specs']==r['specs'] and x['terminals']==r['terminals']]
            check(column['key']+':'+r['method']+':alias_same_measurement',all(x['bytes']==r['bytes'] and x['encode_seconds_repeats']==r['encode_seconds_repeats'] and x['decode_seconds_repeats']==r['decode_seconds_repeats'] for x in same))
            recipe_keys.add(key)
        check(column['key']+':final_unique_count',len(recipe_keys)==len(b['recipes'])==b['cold_decoder_process_calls'])
        cold_receipts+=b['cold_decoder_process_calls']
        for recipe in b['recipes'].values():
            check(column['key']+':final_frame_hash:'+recipe['frame_sha256'],sha(recipe['frame_path'])==recipe['frame_sha256'] and Path(recipe['frame_path']).stat().st_size==recipe['bytes'])
            check(column['key']+':repeats',len(recipe['encode_seconds_repeats'])==len(recipe['decode_seconds_repeats'])==11)
    check('expected_searches',len(runs)==len(manifest['columns'])*len(manifest['config']['weights_tenths'])*len(manifest['config']['tail_scopes']))
    return {'status':'PASS' if not issues else 'FAIL','schema':'beam30-independent-stdlib-audit-v1',
            'output':str(output),'manifest_sha256':sha(output/'MANIFEST.json'),'summary_sha256':sha(output/'SUMMARY.json'),
            'searches':len(runs),'method_column_rows':len(rows),'candidate_counts':dict(totals),
            'cold_decode_receipts':cold_receipts,'cold_decode_reexecuted_by_audit':False,
            'all_checks':checks,'issues':issues}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    result=audit(p.parse_args().output.resolve())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
