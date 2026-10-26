#!/usr/bin/env python3
"""Detach authorized local 20-column Beam30 .4/.3/.3 supplement, no remote jobs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    output=a.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    runner=root/'scripts/run_weight_grid_beam30_v1.py'
    command=[sys.executable,str(runner),'--output',str(output)]
    if a.resume:command.append('--resume')
    log=output.parent/(output.name+'.runner.log')
    with log.open('ab',buffering=0) as stream:
        child=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(status='DISPATCHED_NOT_YET_VERIFIED',pid=child.pid,command=command,
                 output=str(output),log_path=str(log),start_new_session=True,
                 runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                 requested_at=datetime.now(timezone.utc).isoformat())
    path=output.parent/(output.name+'.launch.json')
    path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
