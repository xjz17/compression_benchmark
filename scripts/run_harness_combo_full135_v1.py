#!/usr/bin/env python3
"""Isolated, resumable full-column coordinator. Numeric execution stays in C++."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import signal
import shlex
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def io_priority():
    # The pinned deployment is x86_64 Linux; ioprio_get(IOPRIO_WHO_PROCESS, 0).
    require(os.uname().machine == 'x86_64', 'unsupported ioprio syscall ABI')
    result = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    require(result >= 0, 'cannot read process IO priority')
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def identities():
    rows = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[1]) == 2:
                continue
            status = (entry / 'status').read_text().splitlines()
            affinity = next(x.split(':', 1)[1].strip() for x in status
                            if x.startswith('Cpus_allowed_list:'))
            pid = int(entry.name)
            rows[entry.name] = dict(starttime=fields[19], nice=int(fields[16]),
                                   scheduler=os.sched_getscheduler(pid), affinity=affinity)
        except (OSError, ValueError, StopIteration):
            continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['build', 'selftest', 'freeze', 'run', 'protect'])
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--column-timeout', type=int, default=14400)
    args = parser.parse_args()
    root, dep = args.root.resolve(), args.deployment.resolve()
    require(not sys.flags.optimize, 'Python optimization disables safety assertions in dependencies')
    require(root != dep and root.parent == dep.parent.parent, 'unsafe experiment root')
    require(dep.name == 'unified_compression_20261022_v1', 'unexpected deployment')
    require(os.uname().nodename == 'timecho-20', 'unexpected server')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = json.loads((root / 'ENV_SPEC.json').read_text())
    require(str(root) == spec['root'] and str(dep) == spec['deployment_read_only'], 'ENV paths differ')
    require(set(os.sched_getaffinity(0)) == {spec['cpu_affinity']}, 'CPU affinity differs')
    require(os.sched_getscheduler(0) == os.SCHED_IDLE, 'scheduler must be idle')
    require(os.getpriority(os.PRIO_PROCESS, 0) == spec['nice'] == 19, 'nice must be 19')
    require(spec['io_class'] == 'idle' and io_priority() >> 13 == 3, 'IO class must be idle')
    require(spec['warmups'] == 1 and spec['repeats'] == 3 and spec['workers'] == 1, 'measurement spec differs')
    limit = spec['address_space_limit_bytes']
    require(limit == 2 << 30, 'address-space spec differs')
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    src, build = root / 'source', dep / 'build'
    binary = root / 'build/harness_combo_full135'
    linkfile = build / 'CMakeFiles/unified_compression_cli.dir/link.txt'
    tokens = shlex.split(linkfile.read_text())
    libraries = tokens[tokens.index('-o') + 2:]
    protected = {str(linkfile): sha(linkfile)}
    for token in libraries:
        path = Path(token) if token.startswith('/') else (build / token).resolve()
        if path.is_file():
            protected[str(path)] = sha(path)
    catalog = dep / 'src/benchmark/reference/actions/operator_harness_tree_catalog_v3.json'
    protected[str(catalog)] = sha(catalog)
    protected[str(Path('/usr/bin/c++').resolve())] = sha(Path('/usr/bin/c++').resolve())
    for include_root in [dep / 'src/third_party/include', dep / 'src/cpp/operator_harness_v2',
                         dep / 'src/cpp/native_graph_v1', dep / 'src/cpp/src']:
        for header in include_root.rglob('*'):
            if header.is_file() and header.suffix in ('.h', '.hpp', '.inc'):
                protected[str(header)] = sha(header)
    for header in ['external/compression/lz4/lib/lz4frame.h', 'external/compression/lz4/lib/lz4.h',
                   'external/compression/zstd/lib/zstd.h']:
        path = dep / 'src' / header
        protected[str(path)] = sha(path)
    source_paths = sorted(p for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    sources = {str(p.relative_to(src)): sha(p) for p in source_paths}
    before_file = root / 'PROTECTION_PRE.json'
    if not before_file.exists():
        write(before_file, dict(files=protected, processes=identities(), timestamp=time.time()))
    tag_prefix = args.mode + '_attempt' + str(args.attempt)
    write(root / ('RESOURCE_' + tag_prefix + '.json'), dict(
        pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)),
        scheduler=os.sched_getscheduler(0), nice=os.getpriority(os.PRIO_PROCESS, 0),
        address_space_limit=resource.getrlimit(resource.RLIMIT_AS), io_priority=io_priority(), gpu_used=False,
        spec_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()))

    def command(argv, tag, cwd=root, timeout=1800):
        name = tag_prefix + '_' + tag
        log = root / 'logs' / (name + '.log')
        log.parent.mkdir(exist_ok=True)
        require(not log.exists(), 'retain previous attempts; increment --attempt')
        started = time.time()
        failure = None
        with log.open('w') as output:
            try:
                proc = subprocess.Popen(list(map(str, argv)), cwd=cwd, env=env,
                                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Only this newly-created session/process group belongs to us.
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc, failure = 124, 'wall_timeout'
        receipt = dict(argv=list(map(str, argv)), cwd=str(cwd), returncode=rc,
                       elapsed_wall_s=time.time() - started, log=str(log),
                       log_sha256=sha(log), failure=failure)
        write(root / 'commands' / (name + '.json'), receipt)
        return receipt

    def checked(argv, tag, **kwargs):
        receipt = command(argv, tag, **kwargs)
        require(receipt['returncode'] == 0, 'command failed: ' + tag)
        return receipt

    def protection():
        before = json.loads(before_file.read_text())
        after = identities()
        changed = {pid: dict(before=a, after=after[pid]) for pid, a in before['processes'].items()
                   if pid in after and a['starttime'] == after[pid]['starttime'] and a != after[pid]}
        changed_files = [p for p, digest in before['files'].items()
                         if not Path(p).is_file() or sha(p) != digest]
        result = dict(status='passed' if not changed and not changed_files else 'review_required',
                      protected_file_count=len(before['files']), changed_files=changed_files,
                      surviving_settings_changes=changed,
                      ended_or_reused_pids=[pid for pid, a in before['processes'].items()
                                            if pid not in after or a['starttime'] != after[pid]['starttime']],
                      no_control_action_against_existing_processes=True,
                      zero_resource_interference_claimed=False, timestamp=time.time())
        write(root / ('PROTECTION_POST_' + tag_prefix + '.json'), result)
        return result

    completion = None
    try:
        require(not (root / 'FREEZE.json').exists() or args.mode in ('run', 'protect'),
                'frozen build/selftest/protocol is immutable; use a new root')
        if args.mode == 'build':
            binary.parent.mkdir(exist_ok=True)
            ext = src / 'external'
            if not ext.exists():
                ext.symlink_to(dep / 'src/external', target_is_directory=True)
            includes = [src / 'cpp/harness_combo_pilot_v1', dep / 'src/third_party/include',
                        dep / 'src/cpp/operator_harness_v2', dep / 'src/cpp/native_graph_v1',
                        dep / 'src/cpp/src']
            objects = []
            for name, path in [('runtime', src / 'cpp/harness_combo_pilot_v1/runtime.cpp'),
                               ('main', src / 'cpp/harness_combo_full135_v1/main.cpp')]:
                obj = binary.parent / (name + '.o')
                checked(['/usr/bin/c++', '-O3', '-DNDEBUG', '-std=c++20', '-fno-fast-math',
                         '-ffp-contract=off', '-Wall', '-Wextra', '-Wpedantic',
                         *('-I' + str(p) for p in includes), '-c', path, '-o', obj],
                        'compile_' + name, timeout=600)
                objects.append(obj)
            checked(['/usr/bin/c++', '-O3', *objects, '-o', binary, *libraries],
                    'link', cwd=build, timeout=300)
            checked([binary, 'catalog', '--output', root / 'CANDIDATES.json'], 'catalog')
            write(root / 'BUILD.json', dict(status='complete', binary_sha256=sha(binary),
                                          source_hashes=sources, dependencies_rebuilt=False,
                                          nncp='OFF', protected_dependencies=protected))
        elif args.mode == 'selftest':
            target = root / ('selftest_attempt' + str(args.attempt))
            checked([binary, 'selftest', '--out', target], 'selftest', timeout=600)
            receipt = target / 'SELFTEST.json'
            test = json.loads(receipt.read_text())
            require(test.get('failures') == 0, 'selftest failed')
            write(root / 'SELFTEST_ACCEPTED.json', dict(path=str(receipt),
                  receipt_sha256=sha(receipt), binary_sha256=sha(binary)))
        elif args.mode == 'freeze':
            require(not (root / 'FREEZE.json').exists(), 'freeze already exists')
            tasks = [json.loads(line) for line in (root / 'plan/COLUMN_TASKS.jsonl').read_text().splitlines()]
            require(len(tasks) == 3519, 'wrong task count')
            require(len({t['task_id'] for t in tasks}) == 3519, 'duplicate task IDs')
            require(len({(t['column']['dataset_id'], t['column']['column_id']) for t in tasks}) == 3519,
                    'duplicate source column identities')
            for ordinal, task in enumerate(tasks):
                relative = Path(task['task_path'])
                require(not relative.is_absolute() and '..' not in relative.parts and
                        relative.as_posix() == task['task_path'] and
                        (root / 'plan' / relative).resolve().is_relative_to((root / 'plan').resolve()),
                        'unsafe task path')
                require(re.fullmatch(r't[0-9]{4}_[0-9a-f]{16}', task['task_id']) and
                        task['ordinal'] == ordinal, 'task identity/order differs')
            task_hashes = {task['task_path']: sha(root / 'plan' / task['task_path']) for task in tasks}
            require(len(task_hashes) == 3519, 'duplicate task paths')
            paths = ['plan/SCOPE.json', 'plan/COLUMN_TASKS.jsonl', 'CANDIDATES.json', 'PROTOCOL.json',
                     'ENV_SPEC.json', 'BUILD.json', 'SELFTEST_ACCEPTED.json', 'WITNESS.zh.md']
            write(root / 'FREEZE.json', dict(status='frozen', timestamp=time.time(),
                  files={p: sha(root / p) for p in paths}, sources=sources,
                  task_files=task_hashes,
                  binary_sha256=sha(binary), candidate_count=30,
                  candidate_scope='30 frozen typed recipes, not 97x84 Cartesian search',
                  confirm_labels_used=False, protocol_tuning_after_freeze=False))
        elif args.mode == 'run':
            frozen = json.loads((root / 'FREEZE.json').read_text())
            require(all(sha(root / p) == digest for p, digest in frozen['files'].items()), 'frozen files changed')
            require(all(sha(root / 'plan' / p) == digest for p, digest in frozen['task_files'].items()), 'frozen task files changed')
            require(frozen['sources'] == sources and frozen['binary_sha256'] == sha(binary), 'frozen executable changed')
            witness = json.loads((root / 'VERIFIED_WITNESS.json').read_text())
            require(witness['status'] == 'passed' and witness['binary_sha256'] == sha(binary), 'witness executable mismatch')
            require(witness['document_sha256'] == sha(root / 'WITNESS.zh.md'), 'witness document mismatch')
            require(witness['freeze_sha256'] == sha(root / 'FREEZE.json'), 'witness freeze mismatch')
            tasks = [json.loads(line) for line in (root / 'plan/COLUMN_TASKS.jsonl').read_text().splitlines()]
            require(len(tasks) == 3519, 'wrong task count')
            require([t['split']['name'] for t in tasks] == ['development'] * 3433 + ['confirm'] * 86, 'split order/count differs')
            scope = json.loads((root / 'plan/SCOPE.json').read_text())
            require(scope['inputs']['columns']['sha256'] == spec['source_columns_sha256'] and
                    scope['inputs']['split']['sha256'] == spec['split_sha256'], 'source/split binding differs')
            catalog_rows = json.loads((root / 'CANDIDATES.json').read_text())['candidates']
            candidate_ids = {c['id'] for c in catalog_rows}
            require(len(catalog_rows) == len(candidate_ids) == 30, 'catalog candidate IDs not unique')

            def validate_result(task, result_path):
                saved = json.loads(result_path.read_text())
                require(saved['task_sha256'] == frozen['task_files'][task['task_path']], 'result task mismatch')
                require(saved['binary_sha256'] == frozen['binary_sha256'], 'result executable mismatch')
                require(saved['status'] == 'complete', 'column execution incomplete')
                candidates = saved['candidates']
                require(len(candidates) == 30 and len({c['id'] for c in candidates}) == 30 and
                        {c['id'] for c in candidates} == candidate_ids,
                        'candidate inventory incomplete')
                for candidate in candidates:
                    require(candidate['status'] in ('success', 'ineligible', 'domain_rejected'), 'unresolved candidate failure')
                    if candidate['status'] == 'success':
                        require(candidate['strict'] and candidate['bit_exact'] and candidate['rank_eligible'] and
                                candidate['independent_decode']['verified'], 'unverified candidate')
                return sha(result_path)

            completed, errors = 0, []
            phase = None
            for ordinal, task in enumerate(tasks):
                split = task['split']['name']
                if split != phase:
                    if split == 'confirm':
                        require(phase == 'development' and ordinal == 3433 and not errors, 'development execution unresolved')
                        dev_hashes = {x['task_id']: validate_result(x, root / 'results' / x['task_id'] / 'RESULT.json')
                                      for x in tasks[:ordinal]}
                        write(root / 'DEVELOPMENT_ACCEPTED.json', dict(status='passed', columns=3433,
                              result_hashes=dev_hashes, freeze_sha256=sha(root / 'FREEZE.json'),
                              no_tuning_or_catalog_change=True, timestamp=time.time()))
                    phase = split
                    write(root / 'PHASE.json', dict(phase=phase, ordinal=ordinal, timestamp=time.time()))
                out = root / 'results' / task['task_id']
                result = out / 'RESULT.json'
                # The C++ resume path rechecks complete containers and receipts;
                # existence of RESULT alone never skips a column.
                write(root / 'PROGRESS.json', dict(status='running', completed_columns=completed,
                      total_columns=3519, current=task, timestamp=time.time()))
                call = command([binary, 'run-column', '--task', root / 'plan' / task['task_path'], '--out', out,
                                '--warmups', '1', '--repeats', '3'], 'column_' + str(ordinal),
                               timeout=args.column_timeout)
                validation_error = None
                if not call['returncode'] and result.exists():
                    try:
                        validate_result(task, result)
                    except Exception as exc:
                        validation_error = str(exc)
                if call['returncode'] or not result.exists() or validation_error:
                    errors.append(dict(ordinal=ordinal, task=task, command=call,
                                       validation_error=validation_error))
                    write(root / ('COLUMN_FAILURES_attempt' + str(args.attempt) + '.json'), errors)
                else:
                    completed += 1
                if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(tasks):
                    checked([sys.executable, src / 'scripts/report_harness_combo_full135_v1.py',
                             '--plan-root', root / 'plan', '--run-root', root,
                             '--output', root / 'report',
                             *(['--no-verify-frames'] if ordinal + 1 < len(tasks) else [])],
                            'report_' + str(ordinal), timeout=14400)
            summary_path = root / 'report/REPORT.json'
            summary = json.loads(summary_path.read_text())
            require(summary['coverage']['missing'] == 0 and summary['coverage']['failures'] == 0 and
                    summary['all_retained_strict_frames_verified'], 'final report has unresolved gaps')
            require(summary['status'] == 'complete' and summary['coverage']['bound_column_results'] == 3519 and
                    summary['coverage']['source_blocks_in_bound_results'] == 558035 and
                    summary['coverage']['observed_terminal_attempts'] == 105570,
                    'final report coverage differs')
            status = 'complete' if completed == len(tasks) and not errors else 'partial_with_gaps'
            write(root / 'RUN_FINISHED.json', dict(status=status, completed_columns=completed,
                  total_columns=len(tasks), failed_columns=len(errors), timestamp=time.time(),
                  report_sha256=sha(summary_path),
                  freeze_sha256=sha(root / 'FREEZE.json'), binary_sha256=sha(binary)))
            require(status == 'complete' and not errors, 'execution contains failed column attempts')
            checked([sys.executable, src / 'scripts/compare_harness_combo_full135_reference_v1.py',
                     '--root', root, '--oracle', spec['oracle_reference_path']],
                    'frozen_reference_context', timeout=1800)
            context = root / 'report/FROZEN_REFERENCE_CONTEXT.json'
            completion = dict(status='complete', columns=3519, datasets=135,
                  source_blocks=558035, candidate_attempts=105570,
                  run_finished_sha256=sha(root / 'RUN_FINISHED.json'),
                  report_sha256=sha(summary_path), reference_context_sha256=sha(context),
                  freeze_sha256=sha(root / 'FREEZE.json'), timestamp=time.time())
            write(root / 'EXECUTION_COMPLETE.json', completion)
        audit = protection()
        if completion is not None:
            require(audit['status'] == 'passed', 'final protection audit requires review')
            completion['protection_receipt_sha256'] = sha(root / ('PROTECTION_POST_' + tag_prefix + '.json'))
            write(root / 'COMPLETION.json', completion)
        print(json.dumps(dict(mode=args.mode, status='complete', protection=audit['status'])), flush=True)
    except Exception as exc:
        write(root / ('FAILED_' + tag_prefix + '.json'), dict(error=str(exc), timestamp=time.time()))
        protection()
        raise


if __name__ == '__main__':
    main()
