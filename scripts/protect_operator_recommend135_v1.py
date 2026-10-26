#!/usr/bin/env python3
"""Read-only post-run checks for prior runtimes, source closure and job settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(root):
    root=root.resolve(strict=True)
    if socket.gethostname()!='timecho-20':raise RuntimeError('wrong host')
    prior=json.loads((root/'PRE_STATE.json').read_text());build=json.loads((root/'build/BUILD_RECEIPT.json').read_text())
    frozen=json.loads((root/'PRODUCTION_FREEZE.json').read_text())
    checks={}
    for category,expected in [('old_experiment_files',prior['protected_files']),
                              ('reused_runtime_files',build['frozen_runtime_before']),
                              ('production_source_closure',frozen['files'])]:
        mismatch=[]
        for filename,expected_sha in expected.items():
            p=Path(filename)
            if not p.is_file() or sha(p)!=expected_sha:mismatch.append(filename)
        checks[category]={'count':len(expected),'mismatch':mismatch,'passed':not mismatch}
    inputs=json.loads((root/'inputs/INPUTS.json').read_text())
    plan=Path(inputs['plan_columns_path']);manifest=json.loads(plan.read_text())
    source_checks={str(plan):inputs['plan_columns_sha256'],**manifest['manifest_sha256']}
    mismatches=[p for p,s in source_checks.items() if not Path(p).is_file() or sha(p)!=s]
    checks['source_inventory']={'count':len(source_checks),'mismatch':mismatches,'passed':not mismatches}
    pre=json.loads((root/'HOST_DETAILS_PRE_MATRIX.json').read_text());jobs=[];changed=[];ended=[]
    for job in pre['observed_running_job_identities']:
        p=Path('/proc')/str(job['pid'])
        try:
            text=(p/'stat').read_text();fields=text[text.rfind(')')+2:].split()
            if fields[19]!=job['start_ticks']:ended.append(job['pid']);continue
            current={'pid':job['pid'],'start_ticks':fields[19],'comm':(p/'comm').read_text().strip(),
                     'affinity':sorted(os.sched_getaffinity(job['pid'])),
                     'scheduler':os.sched_getscheduler(job['pid']),'nice':os.getpriority(os.PRIO_PROCESS,job['pid'])}
            jobs.append(current)
            if any(current[key]!=job[key] for key in ('affinity','scheduler','nice')):changed.append({'before':job,'after':current})
        except (FileNotFoundError,ProcessLookupError):ended.append(job['pid'])
    checks['surviving_job_settings']={'before_count':len(pre['observed_running_job_identities']),'surviving_count':len(jobs),
                                     'ended_or_pid_reused':ended,'changed':changed,'passed':not changed,
                                     'scope':'matching PID+start_ticks only; natural exit is recorded, not attributed to this experiment'}
    current_profile=json.loads((root/'production_supervisor/START.json').read_text())
    checks['own_profile']={'observed':current_profile,'passed':current_profile['affinity']==[38] and
                           current_profile['scheduler']==os.SCHED_IDLE and current_profile['nice']==19 and
                           current_profile['ionice']=='idle' and current_profile.get('address_space_limit_bytes')==[2147483648,2147483648]}
    checks['frozen_bindings']={'passed':sha(root/'build/operator_recommend135_cli')==frozen['binary_sha256'] and
                               sha(root/'inputs/INPUTS.json')==frozen['input_sha256'] and sha(root/'GROUPS.json')==frozen['groups_sha256']}
    report={'schema':'operator-recommend135-protection-v1','hostname':socket.gethostname(),
            'checks':checks,'status':'pass' if all(x['passed'] for x in checks.values()) else 'fail',
            'post_loadavg':Path('/proc/loadavg').read_text().strip(),'surviving_jobs':jobs}
    (root/'PROTECTION_POST.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':{k:{x:y for x,y in v.items() if x not in ('observed','scope')} for k,v in checks.items()}}))
    return report['status']=='pass'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();raise SystemExit(0 if run(a.root) else 1)
