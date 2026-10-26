#!/usr/bin/env python3
"""Document-following native witness; never receives any confirm costs."""
import argparse,datetime,hashlib,json,os,resource,subprocess
from pathlib import Path

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(root):
 root=root.resolve();assert not (root/'WITNESS.json').exists();os.sched_setaffinity(0,{38});os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));os.setpriority(os.PRIO_PROCESS,0,19);resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648));subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
 ready=json.loads((root/'BUILD_FEATURES_READY.json').read_text());assert ready['status']=='ready_for_independent_document_witness'
 build=json.loads((root/'build/BUILD_RECEIPT.json').read_text());assert build['status']=='built-and-tested';commands=[]
 for name,args,expected in [('quality_model_test',[],'quality_model_test: PASS'),('quality_data_test',[],'quality_data_test: PASS'),('quality_experiment',['selftest'],'quality_experiment selftest: PASS')]:
  path=root/'build'/name;assert sha(path)==build['binaries'][name]
  p=subprocess.run([str(path),*args],capture_output=True,text=True);commands.append({'argv':[str(path),*args],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0 and expected in p.stdout,(name,p.stdout,p.stderr)
 features=root/'features/FEATURES.jsonl';receipt=json.loads((root/'features/FEATURES_RECEIPT.json').read_text());assert sha(features)==receipt['output_sha256'] and receipt['column_count']==3519 and receipt['dataset_count']==135 and receipt['oracle_labels_read'] is False
 keys=set();datasets=set();dtypes=set()
 with features.open() as f:
  for line in f:
   d=json.loads(line);k=(d['dataset_id'],d['column_id']);assert k not in keys and len(d['features'])==24 and d['oracle_labels_read'] is False;keys.add(k);datasets.add(d['dataset_id']);dtypes.add(d['input_dtype'])
 assert len(keys)==3519 and len(datasets)==135 and dtypes=={'float32','float64','int64'}
 result={'schema':'operator-oracle-quality135-document-witness-v1','status':'pass','finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'profile':{'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),'scheduler':os.sched_getscheduler(0),'nice':os.getpriority(os.PRIO_PROCESS,0),'ionice':'idle','address_space_limit_bytes':list(resource.getrlimit(resource.RLIMIT_AS))},'commands':commands,'features_sha256':sha(features),'features_receipt_sha256':sha(root/'features/FEATURES_RECEIPT.json'),'build_receipt_sha256':sha(root/'build/BUILD_RECEIPT.json'),'protocol_sha256':sha(root/'PROTOCOL.json'),'frozen_real_feature_rows':len(keys),'confirm_costs_opened':False,'actual_codec_execution_claimed':False}
 (root/'WITNESS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'pass','native_commands':3,'feature_rows':3519}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
