#!/usr/bin/env python3
"""Protect prior immutable outputs and pre-existing user-space process settings."""
import argparse,datetime,hashlib,json,os,socket
from pathlib import Path

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(p,d):
 assert not p.exists(),'refusing replacement '+str(p);p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def job(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
 return {'pid':int(pid),'start_ticks':f[19],'comm':(p/'comm').read_text().strip(),'kernel_thread':bool(int(f[6])&0x00200000),'affinity':sorted(os.sched_getaffinity(int(pid))),'scheduler':os.sched_getscheduler(int(pid)),'nice':os.getpriority(os.PRIO_PROCESS,int(pid))}
def pre(root,old):
 assert socket.gethostname()=='timecho-20'
 prior=json.loads((old/'PROTECTION_PRE.json').read_text());expected=dict(prior['expected_files'])
 for n in ['SOURCE_FREEZE.json','CPP_SOURCE_FREEZE.json']:
  p=old/n;expected[str(p)]=sha(p)
  for rel,h in json.loads(p.read_text())['files'].items():expected[str(old/rel)]=h
 lock=json.loads((old/'LOCK.json').read_text())
 for k,a in lock['artifacts'].items():
  if k=='confirm_costs':continue
  expected[str(old/a['path'])]=a['sha256']
 for n in ['LOCK.json','FINALIZED.json','PROTECTION_POST.json','models/TRAINING_AUDIT.json','report/REPORT.json']:
  expected[str(old/n)]=sha(old/n)
 index=json.loads((old/'models/MODEL_INDEX.json').read_text())
 for t in index['tasks']:expected[str(old/'models'/t['model_path'])]=t['model_sha256']
 mismatch=[p for p,h in expected.items() if sha(p)!=h];assert not mismatch,mismatch
 users=[];kernels=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   j=job(p.name)
   (kernels if j['kernel_thread'] else users).append(j)
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 d={'schema':'operator-oracle-risk135-protection-pre-v1','observed_at':now(),'expected_files':expected,'protected_file_count':len(expected),'observed_userspace_jobs':users,'excluded_kernel_identities':kernels,'kernel_policy':'PF_KTHREAD excluded before run; user-space affinity/scheduler/nice strict','confirm_cost_payload_hashed_or_opened':False,'loadavg':Path('/proc/loadavg').read_text().strip()};write_new(root/'PROTECTION_PRE.json',d);return d

def post(root):
 d=json.loads((root/'PROTECTION_PRE.json').read_text());mismatch=[p for p,h in d['expected_files'].items() if not Path(p).is_file() or sha(p)!=h];changed=[];ended=[];surviving=0
 for before in d['observed_userspace_jobs']:
  try:
   after=job(before['pid'])
   if before['start_ticks']!=after['start_ticks']:ended.append(before['pid']);continue
   surviving+=1;fields=[k for k in ['affinity','scheduler','nice','kernel_thread'] if before[k]!=after[k]]
   if fields:changed.append({'before':before,'after':after,'changed_fields':fields})
  except (FileNotFoundError,ProcessLookupError):ended.append(before['pid'])
 result={'schema':'operator-oracle-risk135-protection-post-v1','status':'pass' if not mismatch and not changed else 'fail','observed_at':now(),'protected_file_count':len(d['expected_files']),'mismatch':mismatch,'settings_changed':changed,'surviving_user_jobs':surviving,'naturally_ended_or_pid_reused':ended,'kernel_policy':d['kernel_policy'],'confirm_cost_payload_hashed_or_opened':False,'limits':'unchanged hashes/settings do not prove absence of shared-host contention'};write_new(root/'PROTECTION_POST.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['pre','post']);p.add_argument('--root',required=True,type=Path);p.add_argument('--quality-root',type=Path);a=p.parse_args();d=pre(a.root,a.quality_root) if a.mode=='pre' else post(a.root);print(json.dumps({'status':d.get('status','snapshotted'),'files':d['protected_file_count']}));raise SystemExit(d.get('status')=='fail')
