#!/usr/bin/env python3
"""Snapshot/check immutable inputs and existing user-space process settings."""
import argparse,datetime,hashlib,json,os,resource,socket
from pathlib import Path

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def profile():
 return {'hostname':socket.gethostname(),'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),'scheduler':os.sched_getscheduler(0),'nice':os.getpriority(os.PRIO_PROCESS,0),'address_space_limit_bytes':list(resource.getrlimit(resource.RLIMIT_AS))}
def job(pid):
 p=Path('/proc')/str(pid); text=(p/'stat').read_text(); fields=text[text.rfind(')')+2:].split(); flags=int(fields[6]); identity={'pid':int(pid),'start_ticks':fields[19],'comm':(p/'comm').read_text().strip(),'kernel_thread':bool(flags&0x00200000)}
 return {**identity,'affinity':sorted(os.sched_getaffinity(int(pid))),'scheduler':os.sched_getscheduler(int(pid)),'nice':os.getpriority(os.PRIO_PROCESS,int(pid))}
def write(p,d):
 assert not p.exists(),'refusing overwrite '+str(p); p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def pre(root,old):
 assert socket.gethostname()=='timecho-20'; prior=json.loads((old/'PROTECTION_PRE.json').read_text()); build=json.loads((old/'build/BUILD_RECEIPT.json').read_text()); expected={**prior['immutable_files'],**prior['source_manifests'],**build['frozen_runtime_before'],**build['source_sha256']}
 expected[str(old/'build/BUILD_RECEIPT.json')]=sha(old/'build/BUILD_RECEIPT.json'); expected[str(old/'build/model.o')]=sha(old/'build/model.o')
 for n,h in build['binaries'].items():expected[str(old/'build'/n)]=h
 for p in (old/'models').glob('*.json'):expected[str(p)]=sha(p)
 observed={p:sha(p) for p in expected}; mismatch=[p for p in expected if observed[p]!=expected[p]]; assert not mismatch,('pre-existing frozen files drifted',mismatch)
 users=[]; skipped=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   d=job(p.name)
   if d['kernel_thread']:skipped.append({k:d[k] for k in ['pid','start_ticks','comm','kernel_thread']})
   else:users.append(d)
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 result={'schema':'operator-oracle-quality135-protection-pre-v1','observed_at':now(),'profile':profile(),'expected_files':expected,'protected_file_count':len(expected),'observed_userspace_jobs':users,'kernel_policy':'PF_KTHREAD identities excluded before experiment: kernel workqueue affinity is OS-managed; user-space settings remain strict','excluded_kernel_identities':skipped,'loadavg':Path('/proc/loadavg').read_text().strip()};write(root/'PROTECTION_PRE.json',result); return result

def post(root):
 d=json.loads((root/'PROTECTION_PRE.json').read_text()); mismatch=[]
 for p,h in d['expected_files'].items():
  if not Path(p).is_file() or sha(p)!=h:mismatch.append(p)
 changed=[]; ended=[]; surviving=[]
 for prior in d['observed_userspace_jobs']:
  try:
   current=job(prior['pid'])
   if current['start_ticks']!=prior['start_ticks']:ended.append(prior['pid']);continue
   surviving.append(current)
   fields=[k for k in ['affinity','scheduler','nice','kernel_thread'] if current[k]!=prior[k]]
   if fields:changed.append({'before':prior,'after':current,'changed_fields':fields})
  except (FileNotFoundError,ProcessLookupError):ended.append(prior['pid'])
 result={'schema':'operator-oracle-quality135-protection-post-v1','observed_at':now(),'status':'pass' if not mismatch and not changed else 'fail','protected_file_count':len(d['expected_files']),'mismatch':mismatch,'settings_changed':changed,'surviving_user_jobs':len(surviving),'naturally_ended_or_pid_reused':ended,'kernel_policy':d['kernel_policy'],'profile':profile(),'loadavg':Path('/proc/loadavg').read_text().strip(),'limits':'unchanged hashes/settings do not prove absence of shared-host resource contention'};write(root/'PROTECTION_POST.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['pre','post']);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared-root',type=Path);a=p.parse_args();d=pre(a.root,a.prepared_root) if a.mode=='pre' else post(a.root);print(json.dumps({'status':d.get('status','snapshotted'),'files':d['protected_file_count']}));raise SystemExit(d.get('status')=='fail')
