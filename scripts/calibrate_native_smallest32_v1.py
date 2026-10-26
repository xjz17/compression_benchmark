#!/usr/bin/env python3
"""Training-only zlib6 time scales measured by the complete native graph engine.

This is a new NGF1 reference, not a relabeling of ZR61 or legacy-cohort results.
Source reading, JSON transport and independent verification are outside native
sample clocks. Unknown size buckets use the nearest frozen training bucket of
the SAME dtype (log2 distance, smaller bucket wins ties), never test timings.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
from typing import Any

import native_prepared_cohort_v1 as data

SCHEMA='native-smallest32-zlib6-reference-v1'
LOOKUP='exact dtype; nearest training log2 size bucket; smaller bucket wins ties; unknown dtype rejected'
RECIPE={'policy_registry':'native-smallest32-zlib6-reference-v1','steps':[
    {'kind':'stage','inputs':[0],'stage':{'action':'zlib.level6','profile':'bytes','parameters':{}}},
    {'kind':'stage','inputs':[1],'stage':{'action':'zlib.level6','profile':'bytes','parameters':{}}},
    {'kind':'stop'}]}


def bucket(size: int) -> int:
    data.require(type(size) is int and 0<size<=1048576,'prepared bytes outside 1..1MiB')
    return 1<<(size-1).bit_length()


def native(cli: Path, mode: str, request: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix='native-graph-request-') as d:
        p=Path(d)/'request.json'
        p.write_bytes(data.canonical({'schema':'native-graph-request-v1',**request}))
        result=subprocess.run([str(cli),mode,str(p)],capture_output=True,timeout=120,check=False)
    data.require(result.returncode==0,'native graph failed: '+result.stderr.decode(errors='replace')[:2000])
    value=json.loads(result.stdout)
    data.require(value.get('schema')=='native-graph-response-v1','wrong native response schema')
    return value


def measure(cli: Path, block: data.Block, warmups: int, repeats: int) -> dict:
    roots=block.roots()
    answer=native(cli,'benchmark',dict(roots=roots,recipe=RECIPE,warmups=warmups,repeats=repeats))
    raw=block.record['original_numeric_bytes']
    data.require(raw>0 and answer['original_numeric_bytes']==raw,'native numeric denominator mismatch')
    frame=bytes.fromhex(answer.pop('frame_hex'))
    data.require(len(frame)==answer['frame_bytes'] and answer['compression_ratio']==len(frame)/raw,'native frame size mismatch')
    cold=native(cli,'decode',{'frame_hex':frame.hex()})
    data.require(cold['roots']==roots and cold['original_numeric_bytes']==raw,'independent cold decode mismatch')
    for field in ('encode_ns','decode_ns'):
        samples=answer[field]
        data.require(len(samples)==repeats and all(type(x) is int and x>0 for x in samples),'invalid native time samples')
    return {**block.identity(),'block_size_bucket_bytes':bucket(block.record['prepared_block_bytes']),
            'frame_sha256':data.digest(frame),'cold_decode_verified':True,
            'compression_ns_per_original_byte':statistics.median(answer['encode_ns'])/raw,
            'decompression_ns_per_original_byte':statistics.median(answer['decode_ns'])/raw,
            'native_receipt':answer}


def references(rows: list[dict]) -> list[dict]:
    groups=collections.defaultdict(list)
    for row in rows:
        data.require(row['split']=='train','non-training observation in reference table')
        groups[row['dtype'],row['block_size_bucket_bytes']].append(row)
    result=[]
    for (dtype,size),members in sorted(groups.items()):
        e=statistics.median(r['compression_ns_per_original_byte'] for r in members)
        d=statistics.median(r['decompression_ns_per_original_byte'] for r in members)
        data.require(all(math.isfinite(x) and x>0 for x in (e,d)),'nonpositive time scale')
        result.append(dict(dtype=dtype,block_size_bucket_bytes=size,E0=e,D0=d,
            training_block_count=len(members),block_fingerprints_sha256=data.digest(data.canonical(sorted(r['block_fingerprint'] for r in members)))))
    return result


def reference_for(table: dict, dtype: str, prepared_bytes: int) -> dict:
    data.require(table['schema']==SCHEMA and table['lookup_rule']==LOOKUP,'reference identity mismatch')
    target=bucket(prepared_bytes)
    choices=[r for r in table['references'] if r['dtype']==dtype]
    data.require(bool(choices),'no training reference for original dtype '+dtype)
    row=min(choices,key=lambda r:(abs(r['block_size_bucket_bytes'].bit_length()-target.bit_length()),r['block_size_bucket_bytes']))
    return {**row,'requested_bucket_bytes':target,'exact_size_bucket':row['block_size_bucket_bytes']==target}


def write_new(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    # Never silently replace a frozen reference or completion receipt.
    with path.open('x',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--materialized',type=Path,required=True)
    p.add_argument('--cli',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--repeats',type=int,default=15)
    a=p.parse_args()
    data.require(not a.output.exists(),'output already exists; use an explicit new reference version')
    data.require(a.warmups>=1 and a.repeats>=3 and a.repeats%2==1,'invalid repetitions')
    data.require(platform.system()=='Linux' and hasattr(os,'sched_getaffinity'),'formal calibration requires Linux affinity')
    affinity=sorted(os.sched_getaffinity(0))
    data.require(len(affinity)==1,'calibration must run on exactly one pinned CPU')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        data.require(os.environ.get(key)=='1',key+' must be 1')
    cli=a.cli.resolve()
    binary=data.digest(cli.read_bytes())
    corpus=data.load(a.cohort,a.materialized,role='train_validation')
    train=corpus.split('train')
    expected=[b for b in train if b.record['original_numeric_bytes']>0]
    print(json.dumps({'event':'start','corpus':corpus.summary(),'nonempty_training_blocks':len(expected),'cpu_affinity':affinity}),flush=True)
    rows=[]
    started=time.monotonic()
    for b in expected:
        rows.append(measure(cli,b,a.warmups,a.repeats))
        if len(rows)%50==0 or len(rows)==len(expected):
            print(json.dumps({'event':'progress','blocks':len(rows),'total':len(expected),'elapsed_seconds':time.monotonic()-started}),flush=True)
    data.require(data.digest(cli.read_bytes())==binary,'native binary changed during calibration')
    value=dict(schema=SCHEMA,cohort_version=data.gate.VERSION,corpus_fingerprint=corpus.fingerprint,
        data_receipt_sha256=data.digest(data.canonical(corpus.receipt)),lookup_rule=LOOKUP,
        reference_method='zlib level 6 independently on values and missing bitmap; complete NGF1 frame',
        recipe=RECIPE,reference_role='training-only time scale; no required policy action',
        calibration_scope='all nonempty train blocks from frozen smallest32 16-source training role',
        validation_used_for_time_statistics=False,test_used_for_training_statistics=False,
        formal_training_admitted=False,warmups=a.warmups,repeats=a.repeats,
        aggregation='per-block median native ns / B_raw; group median of these ns/B_raw',
        native_cli=str(cli),native_binary_sha256=binary,
        driver_sha256=data.digest(Path(__file__).read_bytes()),binding_sha256=data.digest(Path(data.__file__).read_bytes()),
        platform=platform.platform(),hostname=platform.node(),cpu_affinity=affinity,
        thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        training_dataset_count=len({b.dataset_id for b in train}),training_block_count=len(rows),
        original_numeric_bytes=sum(r['original_numeric_bytes'] for r in rows),
        zero_B_raw_blocks=[b.identity() for b in train if not b.record['original_numeric_bytes']],
        references=references(rows),blocks=rows)
    value['calibration_sha256']=data.digest(data.canonical(value))
    write_new(a.output,value)
    print(json.dumps({'event':'complete','output':str(a.output),'sha256':value['calibration_sha256'],
                      'groups':len(value['references']),'blocks':len(rows)}),flush=True)


if __name__=='__main__': main()
