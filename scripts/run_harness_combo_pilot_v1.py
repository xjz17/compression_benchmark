#!/usr/bin/env python3
"""Isolated CPU pilot coordinator; all numeric execution/selection is C++.

No dependency installs, existing process changes, source edits in deployments,
confirm labels, or GPU use. The existing unified deployment is read-only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)


def identities():
    rows = {}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / 'stat').read_text()
            after = stat[stat.rfind(')') + 2:].split()
            if int(after[1]) == 2:  # kernel workers can legitimately move CPUs
                continue
            status = (d / 'status').read_text().splitlines()
            affinity = next(x.split(':', 1)[1].strip() for x in status if x.startswith('Cpus_allowed_list:'))
            pid = int(d.name)
            rows[d.name] = {'starttime': after[19], 'nice': int(after[16]),
                            'scheduler': os.sched_getscheduler(pid), 'affinity': affinity}
        except (OSError, ValueError, StopIteration):
            continue
    return rows


def compare_processes(before, after):
    changed, ended = {}, []
    for pid, a in before.items():
        b = after.get(pid)
        if not b or b['starttime'] != a['starttime']:
            ended.append(pid)
        elif a != b:
            changed[pid] = {'before': a, 'after': b}
    return {'surviving_settings_changes': changed, 'ended_or_reused_pids': ended,
            'no_control_action_against_existing_processes': True,
            'zero_resource_interference_claimed': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--deployment', required=True)
    ap.add_argument('--mode', choices=['build', 'selftest', 'experiment', 'protect'], required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    dep = Path(args.deployment).resolve()
    assert root != dep and root.parent == dep.parent.parent
    assert dep.name == 'unified_compression_20261022_v1'
    assert os.uname().nodename == 'timecho-20'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    src = root / 'source'
    build = dep / 'build'
    binary = root / 'build/harness_combo_pilot'
    plan = root / 'PLAN.json'
    source_files = [src / ('cpp/harness_combo_pilot_v1/' + name) for name in ['runtime.hpp', 'runtime.cpp', 'main.cpp']]
    source_files.append(Path(__file__).resolve())
    source_hashes = {str(p): sha(p) for p in source_files}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def command(cmd, tag, cwd=root, timeout=1200):
        if args.attempt > 1:
            tag = args.mode + '_attempt' + str(args.attempt) + '_' + tag
        log = root / 'logs' / (tag + '.log')
        log.parent.mkdir(exist_ok=True)
        assert not log.exists(), 'retain prior attempts; use a new tag/root'
        started = time.time()
        with log.open('w') as f:
            result = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        receipt = {'argv': [str(x) for x in cmd], 'cwd': str(cwd), 'returncode': result.returncode,
                   'elapsed_wall_s': time.time() - started, 'log': str(log), 'log_sha256': sha(log)}
        write(root / 'commands' / (tag + '.json'), receipt)
        if result.returncode:
            raise RuntimeError('command failed: ' + tag)
        return receipt

    linkfile = build / 'CMakeFiles/unified_compression_cli.dir/link.txt'
    tokens = shlex.split(linkfile.read_text())
    marker = tokens.index('-o')
    libraries = tokens[marker + 2:]
    protected = {str(linkfile): sha(linkfile)}
    for token in libraries:
        q = (build / token).resolve() if not token.startswith('/') else Path(token)
        if q.is_file():
            protected[str(q)] = sha(q)
    protected[str(dep / 'src/benchmark/reference/actions/operator_harness_tree_catalog_v3.json')] = sha(dep / 'src/benchmark/reference/actions/operator_harness_tree_catalog_v3.json')
    before_path = root / 'PROTECTION_PRE.json'
    if not before_path.exists():
        write(before_path, {'files': protected, 'processes': identities(),
                            'loadavg': Path('/proc/loadavg').read_text(), 'timestamp': time.time()})
    spec = json.loads((root / 'ENV_SPEC.json').read_text())
    assert set(os.sched_getaffinity(0)) == {spec['cpu_affinity']}
    assert os.sched_getscheduler(0) == os.SCHED_IDLE and os.getpriority(os.PRIO_PROCESS, 0) == 19
    write(root / ('RESOURCE_' + args.mode + '.json'), {
        'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
        'scheduler': os.sched_getscheduler(0), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
        'address_space_limit': __import__('resource').getrlimit(__import__('resource').RLIMIT_AS),
        'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()})
    try:
        if args.mode == 'build':
            binary.parent.mkdir(exist_ok=True)
            # Relative upstream includes resolve through a read-only source link.
            ext = src / 'external'
            if not ext.exists():
                ext.symlink_to(dep / 'src/external', target_is_directory=True)
            inc = ['-I' + str(src / 'cpp/harness_combo_pilot_v1'),
                   '-I' + str(dep / 'src/third_party/include'),
                   '-I' + str(dep / 'src/cpp/operator_harness_v2'),
                   '-I' + str(dep / 'src/cpp/native_graph_v1'),
                   '-I' + str(dep / 'src/cpp/src')]
            objects = []
            for name in ['runtime', 'main']:
                obj = binary.parent / (name + '.o')
                command(['/usr/bin/c++', '-O3', '-DNDEBUG', '-std=c++20', '-fno-fast-math',
                         '-ffp-contract=off', '-Wall', '-Wextra', '-Wpedantic', *inc, '-c',
                         str(src / ('cpp/harness_combo_pilot_v1/' + name + '.cpp')), '-o', str(obj)], 'compile_' + name, timeout=300)
                objects.append(obj)
            command(['/usr/bin/c++', '-O3', *map(str, objects), '-o', str(binary), *libraries], 'link', cwd=build, timeout=180)
            command([str(binary), '--catalog'], 'catalog')
            write(root / 'BUILD.json', {'status': 'complete', 'binary_sha256': sha(binary),
                                       'source_hashes': source_hashes,
                                       'dependencies_rebuilt': False, 'nncp': 'OFF', 'protected_dependencies': protected})
        elif args.mode == 'selftest':
            target = root / ('selftest' if args.attempt == 1 else 'selftest_attempt' + str(args.attempt))
            command([str(binary), '--selftest', str(target)], 'selftest', timeout=300)
            write(root / 'SELFTEST_ACCEPTED.json', {'path': str(target / 'SELFTEST.json'),
                                                   'binary_sha256': sha(binary), 'receipt_sha256': sha(target / 'SELFTEST.json')})
        elif args.mode == 'experiment':
            built = json.loads((root / 'BUILD.json').read_text())
            assert built['source_hashes'] == source_hashes and built['binary_sha256'] == sha(binary)
            accepted = json.loads((root / 'SELFTEST_ACCEPTED.json').read_text())
            assert accepted['binary_sha256'] == sha(binary)
            assert accepted['receipt_sha256'] == sha(accepted['path'])
            assert json.loads(Path(accepted['path']).read_text())['failures'] == 0
            assert sha(plan) == spec['plan_sha256']
            assert (root / 'VERIFIED_WITNESS.json').exists(), 'independent agent must follow witness doc first'
            phases = [('--matrix', 'discovery'), ('--rank', None), ('--online', 'tuning'), ('--select-policy', None),
                      ('--online', 'locked_validation'), ('--matrix', 'tuning'), ('--matrix', 'locked_validation')]
            for mode, stage in phases:
                tag = mode[2:] + ('_' + stage if stage else '')
                write(root / 'PHASE.json', {'phase': tag, 'status': 'running', 'timestamp': time.time()})
                command([str(binary), mode, str(plan), str(root / 'results'), *([stage] if stage else [])], tag, timeout=1200)
            write(root / 'RUN_COMPLETE.json', {'status': 'complete', 'plan_sha256': sha(plan),
                                              'binary_sha256': sha(binary), 'timestamp': time.time()})
        before = json.loads(before_path.read_text())
        changed = [p for p, digest in before['files'].items() if not Path(p).is_file() or sha(p) != digest]
        proc = compare_processes(before['processes'], identities())
        receipt = {'protected_file_count': len(before['files']), 'changed_files': changed, **proc}
        receipt['status'] = 'passed' if not changed and not proc['surviving_settings_changes'] else 'review_required'
        write(root / ('PROTECTION_POST_' + args.mode + '.json'), receipt)
        assert not changed, 'a reused dependency changed'
        print(json.dumps({'mode': args.mode, 'status': 'complete', 'protection': receipt['status']}))
    except Exception as exc:
        write(root / ('FAILED_' + args.mode + '.json'), {'error': str(exc), 'timestamp': time.time()})
        raise


if __name__ == '__main__':
    main()
