#!/usr/bin/env python3
"""Real C++ frame-only exec witness for the documented CPU environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(root):
    root=root.resolve(strict=True);build=json.loads((root/'build/BUILD_RECEIPT.json').read_text())
    binary=root/'build/operator_recommend135_cli';assert sha(binary)==build['binaries']['operator_recommend135_cli']
    index=json.loads((root/'witness_final/CASE_INDEX.json').read_text());selected={}
    for item in index:
        if item['status']!='strict_verified':continue
        result=json.loads(Path(item['result_path']).read_text())
        sample=json.loads((Path(item['result_path']).parent.parent/'SAMPLE.json').read_text())
        selected.setdefault((sample['dtype'],result['kind']),(result,sample))
    required={(dtype,kind) for dtype in ('float32','float64','int64') for kind in ('algorithm','operator','operator_tree','algorithm_combination')}
    assert required<=set(selected)
    output=root/'agent_doc_witness';output.mkdir(exist_ok=False);receipts=[];pids=set()
    for i,key in enumerate(sorted(required)):
        result,sample=selected[key];frame=Path(result['measurement']['frame_path']);assert sha(frame)==result['measurement']['frame_sha256']
        decoded=output/str(i)
        command=['chrt','--idle','0','nice','-n','19','ionice','-c','3','taskset','-c','38',str(binary),'decode',str(frame),str(decoded)]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        stdout,stderr=process.communicate(timeout=60);assert process.returncode==0,(command,stderr)
        receipt=json.loads((decoded/'DECODE.json').read_text());assert receipt['decode_pid']==process.pid
        assert process.pid not in pids and process.pid!=result['measurement_process']['pid'];pids.add(process.pid)
        assert sha(decoded/'values.bin')==sample['tail']['values_sha256']
        assert sha(decoded/'missing.bin')==sample['tail']['missing_sha256']
        assert receipt['dtype']==sample['dtype'] and receipt['row_count']==sample['tail']['row_count']
        assert receipt['frame_sha256']==sha(frame)
        receipts.append({'dtype':key[0],'kind':key[1],'command':command,'pid':process.pid,'source_case':result['sample_id'],
                         'candidate_id':result['candidate_id'],'frame_sha256':sha(frame),'verified':True})
    report={'status':'passed','binary_sha256':sha(binary),'cases':receipts,'fresh_exec_count':len(pids),
            'doc_invocation':'python3 source/scripts/verify_operator_recommend135_doc_witness.py '+str(root)}
    (output/'REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fresh_exec_count':len(pids)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();verify(a.root)
