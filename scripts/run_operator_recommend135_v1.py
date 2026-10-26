#!/usr/bin/env python3
"""Run frozen C++ commands sequentially; never runs compression in Python."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import resource


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)


def finalize_groups(root,inputs):
    prior=root/'GROUPS_PREDECLARED.json';groups=json.loads(prior.read_text())
    parent={d:d for d in groups['dataset_to_group']}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for group in groups['groups']:
        for d in group['datasets'][1:]:union(group['datasets'][0],d)
    signatures=collections.defaultdict(list)
    for s in inputs['samples']:
        key=(s['dtype'],s['element_count'],*(s[p][k] for p in ('head','tail') for k in ('row_count','values_sha256','missing_sha256')))
        signatures[key].append(s)
    duplicate=[]
    for samples in signatures.values():
        datasets=sorted({s['dataset_id'] for s in samples})
        if len(datasets)>1:
            for d in datasets[1:]:union(datasets[0],d)
            duplicate.append({'datasets':datasets,'sample_ids':[s['sample_id'] for s in samples],
                              'reason':'exact head+tail value/mask SHA, dtype, element and row counts; conservative anti-leakage merge, not a provenance claim'})
    sets=collections.defaultdict(list)
    for d in sorted(parent):sets[find(d)].append(d)
    final=[];mapping={}
    for datasets in sets.values():
        originals=[g for g in groups['groups'] if set(g['datasets']) & set(datasets)]
        group_id=originals[0]['group_id'] if len(originals)==1 else 'fingerprint:'+hashlib.sha256('\n'.join(datasets).encode()).hexdigest()[:20]
        final.append({'group_id':group_id,'datasets':datasets,'confidence':originals[0]['confidence'] if len(originals)==1 else 'exact sample duplicates; conservative',
                      'evidence':[e for g in originals for e in g['evidence']]})
        mapping.update({d:group_id for d in datasets})
    groups['predeclared_file_sha256']=sha(prior);groups['groups']=final;groups['dataset_to_group']=mapping
    groups['scope']['group_count']=len(final)
    groups['sample_duplicate_merges']=duplicate;groups['group_count']=len(final)
    groups['input_file_sha256']=sha(root/'inputs/INPUTS.json')
    groups['finalization_timing']='before any candidate performance measurement'
    save(root/'GROUPS.json',groups);return groups


def run(root,stage,tag=''):
    if socket.gethostname()!='timecho-20':raise RuntimeError('wrong host')
    resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root=root.resolve(strict=True);binary=root/'build/operator_recommend135_cli'
    build=json.loads((root/'build/BUILD_RECEIPT.json').read_text())
    if sha(binary)!=build['binaries']['operator_recommend135_cli'] or build['NNCP']!='OFF':raise RuntimeError('binary binding/NNCP violation')
    inputs=json.loads((root/'inputs/INPUTS.json').read_text());datasets={s['dataset_id'] for s in inputs['samples']}
    if len(datasets)!=135:raise RuntimeError('not 135 dataset coverage')
    if stage=='witness':groups=finalize_groups(root,inputs)
    else:groups=json.loads((root/'GROUPS.json').read_text())
    if set(groups['dataset_to_group'])!=datasets:raise RuntimeError('dataset partition mismatch')
    suffix=('_'+tag) if tag else ''
    own=root/(stage+'_supervisor'+suffix);own.mkdir(exist_ok=False)
    observation={'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
                 'scheduler':os.sched_getscheduler(0),'nice':os.getpriority(os.PRIO_PROCESS,0),
                 'ionice':subprocess.check_output(['ionice','-p',str(os.getpid())],text=True).strip(),
                 'address_space_limit_bytes':list(resource.getrlimit(resource.RLIMIT_AS)),
                 'binary_sha256':sha(binary),'input_sha256':sha(root/'inputs/INPUTS.json'),
                 'groups_sha256':sha(root/'GROUPS.json'),'NNCP':'OFF'}
    if observation['affinity']!=[38] or observation['scheduler']!=os.SCHED_IDLE or observation['nice']!=19 or observation['ionice']!='idle':raise RuntimeError('resource isolation differs from frozen spec: '+str(observation))
    save(own/'START.json',observation)
    if stage=='witness':
        selected=[];seen=set()
        for s in inputs['samples']:
            if s['dtype'] not in seen:selected.append(s);seen.add(s['dtype'])
        witness=dict(inputs,samples=selected);input_path=root/'WITNESS_INPUTS.json';save(input_path,witness)
        output=root/('witness'+suffix)
    else:input_path=root/'inputs/INPUTS.json';output=root/'campaign'
    commands=[[str(binary),'campaign',str(input_path),str(root/'GROUPS.json'),str(output),'5']]
    for number,command in enumerate(commands):
        with (own/(str(number)+'.log')).open('xb') as f:
            completed=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        save(own/(str(number)+'_PROCESS.json'),{'command':command,'returncode':completed.returncode})
        if completed.returncode:raise RuntimeError('C++ campaign command failed; evidence retained')
    done=json.loads((output/'CAMPAIGN_COMPLETE.json').read_text())
    rows=json.loads((output/'ROWS.json').read_text())
    if done['failed'] or len(rows)!=len(witness['samples'] if stage=='witness' else inputs['samples']):raise RuntimeError('failed/missing measurement rows; do not fit an incomplete matrix')
    if stage=='production':
        command=[str(binary),'evaluate',str(output/'ROWS.json'),str(root/'model')]
        with (own/'evaluate.log').open('xb') as f:completed=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        save(own/'EVALUATE_PROCESS.json',{'command':command,'returncode':completed.returncode})
        if completed.returncode:raise RuntimeError('evaluation failed; evidence retained')
    save(own/'DONE.json',{'status':'complete','campaign':done,'observation':observation})
    print(json.dumps({'stage':stage,'status':'complete','campaign':done}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('stage',choices=['witness','production']);p.add_argument('--tag',default='');a=p.parse_args();run(a.root,a.stage,a.tag)
