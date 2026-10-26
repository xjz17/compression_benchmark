#!/usr/bin/env python3
"""Package a named review/model subset; large matrices stay on the source host."""
import argparse, csv, hashlib, json, os, resource, subprocess, tarfile, statistics, math
from pathlib import Path

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main(root):
    os.sched_setaffinity(0, {38})
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    assert json.loads((root/'verification/VERIFIED_FINAL.json').read_text())['status'] == 'pass'
    output = root/'delivery'
    output.mkdir()
    groups = {}
    with (root/'DEVELOPMENT_PREDICTIONS.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            groups.setdefault(row['task_id'], []).append(row['selection_cpu_ns_median'])
    receipt = json.loads((root/'DEVELOPMENT_PREDICTIONS.jsonl.receipt.json').read_text())
    assert receipt['output']['sha256'] == sha(root/'DEVELOPMENT_PREDICTIONS.jsonl')
    measurement = {
        'schema':'operator-oracle-risk135-measurement-summary-v1',
        'scope':'new C++ selector only; no codec or end-to-end speed comparison',
        'selection_ns_medians_across_columns':{
            task:{'rows':len(values),'median':statistics.median(values),
                  'mean':statistics.mean(values),'p95':sorted(values)[math.ceil(.95*len(values))-1]}
            for task,values in groups.items()},
        'prediction_process_timing':receipt['timing'],
        'feature_joint_prefix24_and_multi3_timing':json.loads((root/'features/multi3/FEATURES_RECEIPT.json').read_text())['timings'],
        'feature_timing_limit':'joint feature extraction; no isolated marginal multi3 cost measurement',
        'fitting':json.loads((root/'LOCK.json').read_text())['tune'],
        'prediction_receipt_sha256':sha(root/'DEVELOPMENT_PREDICTIONS.jsonl.receipt.json'),
        'driver_verification_sha256':sha(root/'verification/DRIVER_VERIFIED_V2.json'),
        'confirm_cost_payload_read':False}
    (output/'MEASUREMENT_SUMMARY.json').write_text(json.dumps(measurement,indent=2)+'\n')
    cols = ['task_id','policy_id','fold','dataset_id','sample_id','dtype','original_numeric_bytes',
            'is_long_column','selected_candidate_id','selected_compressed_bytes',
            'allowed_oracle_compressed_bytes','absolute_excess_bytes']
    with (output/'OOF_DEVELOPMENT_TOP1.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        count = 0
        with (root/'models/SELECTED_OOF_FULL_RANKS.jsonl').open() as source:
            for line in source:
                row = json.loads(line)
                writer.writerow({k:row[k] for k in cols})
                count += 1
        assert count == 10299
    files = [p for p in (root/'source').rglob('*') if p.is_file()]
    files += [p for p in (root/'models').rglob('*.json') if p.is_file()]
    files += [p for p in (root/'report').rglob('*') if p.is_file()]
    files += [p for p in (root/'verification/source').glob('*') if p.is_file()]
    files += [p for p in (root/'verification').glob('*.json') if p.is_file()]
    files += [p for p in (root/'verification').glob('*.py') if p.is_file()]
    files += [p for p in (root/'verification').glob('*.log') if p.is_file()]
    files += [root/n for n in ['ENV_SPEC.json','DESIGN.json','PROTOCOL.json','SOURCE_FREEZE.json',
              'INPUTS_READY.json','FEATURE_PARITY.json','BUILD_FEATURES_READY.json','WITNESS.json',
              'LOCK.json','FINALIZED.json','PROTECTION_PRE.json','PROTECTION_POST.json',
              'DEVELOPMENT_PREDICTIONS.jsonl.receipt.json','build/BUILD_RECEIPT.json',
              'features/prefix24/FEATURES_RECEIPT.json','features/multi3/FEATURES_RECEIPT.json',
              'costs/CANDIDATES.json','inputs/split_config.json','inputs/GROUPS.json',
              'inputs/recommendation_ground_truth.json','delivery/OOF_DEVELOPMENT_TOP1.csv',
              'delivery/MEASUREMENT_SUMMARY.json']]
    files = sorted(set(files))
    assert all(p.is_file() for p in files)
    manifest = {'schema':'operator-oracle-risk135-local-subset-v1',
                'scope':'source, models, audit/report receipts and selected development OOF CSV; no full cost/feature/prediction/replay matrix closure',
                'omitted':['DEVELOPMENT_COSTS.jsonl','FEATURES.jsonl','DEVELOPMENT_PREDICTIONS.jsonl',
                           'OOF full-rank JSONL','55 policy replay JSONL','source data payloads','native binaries'],
                'confirm_cost_payload_read':False,'files':{str(p.relative_to(root)):sha(p) for p in files}}
    mp = output/'MANIFEST.json'
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    archive = output/'REVIEW_MODEL_SUBSET.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for p in files+[mp]:
            tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
    receipt = {'status':'complete_subset','archive_bytes':archive.stat().st_size,
               'archive_sha256':sha(archive),'payload_files':len(files),'manifest_sha256':sha(mp),
               'confirm_cost_payload_read':False}
    (output/'COMPLETE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve(strict=True))
