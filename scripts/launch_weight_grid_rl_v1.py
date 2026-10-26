#!/usr/bin/env python3
"""Launch the full 66-weight runner in a separate local process session.

Only use after resource/experiment authorization. This does not change sleep
settings, create schedulers, use remote machines, or reduce the training budget.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root/'scripts/run_weight_grid_rl_v1.py'), '--output', str(output)]
    if args.resume: command.append('--resume')
    log = output.parent/(output.name+'.runner.log')
    receipt = output.parent/(output.name+'.launch.json')
    metadata = dict(command=command, cwd=str(root), log_path=str(log), output=str(output),
                    receipt=str(receipt), start_new_session=True, dry_run=args.dry_run,
                    requested_at=datetime.now(timezone.utc).isoformat())
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2)); return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Log and receipt intentionally live outside the new experiment directory.
    with log.open('ab', buffering=0) as stream:
        child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    metadata.update(pid=child.pid, status='DISPATCHED_NOT_YET_VERIFIED_RUNNING')
    receipt.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__=='__main__': main()
