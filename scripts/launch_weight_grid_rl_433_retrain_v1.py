#!/usr/bin/env python3
"""Launch only the authorized local, single-worker 120-run 433 retraining.

Call only after other experimental codec timing has ended. This launcher does
not alter sleep settings, create a scheduler, change budgets or use a remote host.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beam-output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, beam = args.output.resolve(), args.beam_output.resolve()
    runner = root/'scripts/run_weight_grid_rl_433_retrain_v1.py'
    command = [sys.executable, '-B', str(runner), '--output', str(output), '--beam-output', str(beam)]
    if args.resume:
        command.append('--resume')
    output.parent.mkdir(parents=True, exist_ok=True)
    logfile = output.parent/(output.name+'.runner.log')
    with logfile.open('ab', buffering=0) as stream:
        child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    receipt = dict(status='DISPATCHED_NOT_YET_VERIFIED_RUNNING', pid=child.pid,
        requested_at=datetime.now(timezone.utc).isoformat(), command=command, cwd=str(root),
        output=str(output), beam_output=str(beam), log_path=str(logfile), start_new_session=True,
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        expected_new_training_runs=120, reference_source='unchanged historical per-column zlib6')
    text = json.dumps(receipt, ensure_ascii=False, indent=2)+'\n'
    (output.parent/(output.name+'.launch.json')).write_text(text, encoding='utf-8')
    with (output.parent/(output.name+'.launches.jsonl')).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False)+'\n')
    print(text)


if __name__ == '__main__':
    main()
