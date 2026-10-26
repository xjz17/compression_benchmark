#!/usr/bin/env python3
"""Bounded sequential orchestration; all numerical feature/model work is C++."""
import argparse, datetime, hashlib, json, os, resource, socket, subprocess, time
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(p):
    return json.loads(Path(p).read_text())

def write(p, d):
    p = Path(p)
    assert not p.exists(), 'refuse overwrite ' + str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')

def profile():
    assert socket.gethostname() == 'timecho-20'
    os.sched_setaffinity(0, {38})
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    return {'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
            'scheduler': os.sched_getscheduler(0), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
            'address_space_limit_bytes': list(resource.getrlimit(resource.RLIMIT_AS))}

def freeze(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted((root/'source').rglob('*')) if p.is_file()}

def run(cmd):
    start = time.monotonic()
    cpu = os.times()
    print(json.dumps({'started_at': now(), 'command': list(map(str, cmd))}), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)
    end = os.times()
    return {'command': list(map(str, cmd)), 'wall_seconds': time.monotonic()-start,
            'children_user_cpu_seconds': end.children_user-cpu.children_user,
            'children_system_cpu_seconds': end.children_system-cpu.children_system}

def feature_check(root):
    staged = read(root/'INPUTS_READY.json')
    old_path = Path(staged['old_features_path'])
    assert sha(old_path) == staged['old_features_sha256']
    old = {}
    with old_path.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row['dataset_id'], row.get('column_id', row.get('sample_id')))
            assert key not in old and len(row['features']) == 24
            old[key] = row['features']
    assert len(old) == 3519
    metadata = read(root/'costs/INPUT_METADATA.json')['columns']
    devkeys = {(r['dataset_id'], r['column_id']) for r in metadata if r['split'] == 'development'}
    confirmkeys = {(r['dataset_id'], r['column_id']) for r in metadata if r['split'] == 'confirm'}
    assert len(metadata) == len(old) == len(devkeys) + len(confirmkeys) == 3519
    assert len(devkeys) == 3433 and len(confirmkeys) == 86
    assert not (devkeys & confirmkeys) and set(old) == devkeys | confirmkeys
    observed = {}
    receipts = {}
    for variant, size in [('prefix24', 24), ('multi3', 98)]:
        path = root/'features'/variant/'FEATURES.jsonl'
        receipt = read(path.parent/'FEATURES_RECEIPT.json')
        assert receipt['output_sha256'] == sha(path)
        assert receipt['column_count'] == 3433 and receipt['dataset_count'] == 127
        assert receipt['confirm_column_count_processed'] == 0 and not receipt['oracle_labels_read']
        assert receipt['confirm_source_manifests_opened'] == 0
        assert receipt['feature_count'] == size
        rows = {}
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                key = (row['dataset_id'], row.get('column_id', row.get('sample_id')))
                assert key not in rows and len(row['features']) == size
                assert row['oracle_labels_read'] is False
                assert row['feature_variant'] == variant
                assert row['features'][:24] == old[key], ('prefix drift', key)
                rows[key] = row['features']
        assert len(rows) == 3433 and len({k[0] for k in rows}) == 127
        assert set(rows) == devkeys and not (set(rows) & confirmkeys)
        if observed:
            assert rows.keys() == observed.keys()
        else:
            observed = rows
        receipts[variant] = {'path': str(path), 'sha256': sha(path),
                             'receipt_sha256': sha(path.parent/'FEATURES_RECEIPT.json'),
                             'rows': len(rows), 'features': size}
    return {'status': 'pass', 'exact_old_prefix_parity': 3433,
            'multi3_first24_parity': 3433, 'confirm_payload_rows': 0, 'variants': receipts}

def build_features(root, worker):
    assert read(root/'INPUTS_READY.json')['status'] == 'ready_for_native_build'
    source = freeze(root)
    write(root/'SOURCE_FREEZE.json', {'schema': 'operator-oracle-risk135-source-freeze-v1',
          'frozen_at': now(), 'files': source, 'design_sha256': sha(root/'DESIGN.json')})
    build = run(['python3', root/'source/scripts/build_operator_oracle_risk135_v1.py',
                 '--root', root, '--prepared-root', read(root/'ENV_SPEC.json')['reuse_prepared_root']])
    features = run([root/'build/risk_features', read(root/'INPUTS_READY.json')['columns_path'],
                    root/'costs/INPUT_METADATA.json', root/'features'])
    check = feature_check(root)
    write(root/'FEATURE_PARITY.json', check)
    protocol = read(root/'DESIGN.json')
    protocol['source_freeze_sha256'] = sha(root/'SOURCE_FREEZE.json')
    protocol['build_receipt_sha256'] = sha(root/'build/BUILD_RECEIPT.json')
    protocol['feature_bindings'] = check['variants']
    write(root/'PROTOCOL.json', protocol)
    assert source == freeze(root)
    write(root/'BUILD_FEATURES_READY.json', {'status': 'awaiting_independent_witness',
          'finished_at': now(), 'worker': worker, 'build': build, 'feature_extraction': features,
          'source_freeze_sha256': sha(root/'SOURCE_FREEZE.json'),
          'protocol_sha256': sha(root/'PROTOCOL.json'), 'feature_parity_sha256': sha(root/'FEATURE_PARITY.json')})

def witness(root, worker):
    assert read(root/'BUILD_FEATURES_READY.json')['status'] == 'awaiting_independent_witness'
    assert freeze(root) == read(root/'SOURCE_FREEZE.json')['files']
    b = read(root/'build/BUILD_RECEIPT.json')
    for name, expected in b['binaries'].items():
        assert sha(root/'build'/name) == expected
    units = [run([root/'build'/name]) for name in ['quality_model_test', 'quality_data_test']]
    test = run([root/'build/risk_experiment', 'selftest'])
    check = feature_check(root)
    assert check == read(root/'FEATURE_PARITY.json')
    assert not (root/'costs/CONFIRM_COSTS.jsonl').exists()
    write(root/'WITNESS.json', {'schema': 'operator-oracle-risk135-witness-v1', 'status': 'pass',
          'finished_at': now(), 'worker': worker, 'units': units, 'native_selftest': test,
          'feature_check': check, 'doc_sha256': sha(root/'source/docs/operator_oracle_risk135_v1.md'),
          'source_freeze_sha256': sha(root/'SOURCE_FREEZE.json'),
          'build_receipt_sha256': sha(root/'build/BUILD_RECEIPT.json'),
          'confirm_cost_payload_read': False})

def train(root, worker):
    assert read(root/'WITNESS.json')['status'] == 'pass'
    assert freeze(root) == read(root/'SOURCE_FREEZE.json')['files']
    assert not (root/'costs/CONFIRM_COSTS.jsonl').exists()
    tune = run([root/'build/risk_experiment', 'tune', root/'costs/DEVELOPMENT_COSTS.jsonl',
                root/'features/prefix24/FEATURES.jsonl', root/'features/multi3/FEATURES.jsonl',
                root/'PROTOCOL.json', root/'models'])
    predict = run([root/'build/risk_experiment', 'predict', root/'models/MODEL_INDEX.json',
                   root/'features/prefix24/FEATURES.jsonl', root/'features/multi3/FEATURES.jsonl',
                   root/'DEVELOPMENT_PREDICTIONS.jsonl'])
    assert read(root/'models/COMPLETE.json')['confirm_rows_read'] == 0
    files = {str(p.relative_to(root)): sha(p) for p in sorted((root/'models').rglob('*')) if p.is_file()}
    files['DEVELOPMENT_PREDICTIONS.jsonl'] = sha(root/'DEVELOPMENT_PREDICTIONS.jsonl')
    write(root/'LOCK.json', {'schema': 'operator-oracle-risk135-lock-v1', 'status': 'locked_development_only',
          'locked_at': now(), 'files': files, 'worker': worker, 'tune': tune, 'predict': predict,
          'confirm_evaluation_performed': False, 'witness_sha256': sha(root/'WITNESS.json'),
          'protocol_sha256': sha(root/'PROTOCOL.json'), 'source_freeze_sha256': sha(root/'SOURCE_FREEZE.json')})

def finalize(root, worker):
    lock = read(root/'LOCK.json')
    for rel, h in lock['files'].items():
        assert sha(root/rel) == h
    assert freeze(root) == read(root/'SOURCE_FREEZE.json')['files']
    old_quality = Path(read(root/'ENV_SPEC.json')['reuse_quality_root'])
    report = run(['python3', root/'source/scripts/report_operator_oracle_risk135_v1.py', '--root', root,
                  '--baseline-oof', old_quality/'models/CHOSEN_OOF_FULL_RANKS.jsonl'])
    protection = run(['python3', root/'source/scripts/protect_operator_oracle_risk135_v1.py', 'post', '--root', root])
    assert read(root/'PROTECTION_POST.json')['status'] == 'pass'
    write(root/'FINALIZED.json', {'schema': 'operator-oracle-risk135-final-v1', 'status': 'complete',
          'finished_at': now(), 'worker': worker, 'report_run': report, 'protection_run': protection,
          'lock_sha256': sha(root/'LOCK.json'), 'report_sha256': sha(root/'report/REPORT.json'),
          'protection_sha256': sha(root/'PROTECTION_POST.json'), 'confirm_evaluation_performed': False,
          'NNCP': 'OFF'})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build-features', 'witness', 'train-predict', 'finalize'])
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    worker = profile()
    try:
        {'build-features': build_features, 'witness': witness, 'train-predict': train, 'finalize': finalize}[a.mode](root, worker)
    except Exception as e:
        write(root/(a.mode.upper().replace('-', '_')+'_FAILED.json'), {'status': 'failed', 'error': repr(e), 'at': now(), 'worker': worker})
        raise

if __name__ == '__main__':
    main()
