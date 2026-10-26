#!/usr/bin/env python3
"""Save a manifest-bound review subset; retain the full remote evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--require-review', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    assert (root / 'RUN_COMPLETE.json').is_file()
    report = json.loads((root / 'report/REPORT.json').read_text())
    assert report['status'] == 'complete_verified'
    if args.require_review:
        assert (root / 'FINAL_INDEPENDENT_REVIEW.json').is_file()
        review = json.loads((root / 'FINAL_INDEPENDENT_REVIEW.json').read_text())
        assert review['status'] == 'passed' and not review['differences']
    selected = set()
    for p in root.iterdir():
        if p.is_file() and p.suffix in {'.json', '.md', '.log'}:
            selected.add(p)
    for directory in ['report', 'commands', 'logs']:
        selected.update(p for p in (root / directory).rglob('*') if p.is_file())
    for directory in ['source/cpp/harness_combo_pilot_v1', 'source/scripts']:
        selected.update(p for p in (root / directory).glob('*') if p.is_file())
    selected.add(root / 'build/harness_combo_pilot')
    selected.update((root / 'results').rglob('*.json'))
    selected.add(root / 'independent_witness/SELFTEST.json')
    # Winning and fixed baseline validation frames are useful for inspection.
    # Every other full frame remains on the server and was audited in place.
    for c in report['columns'].values():
        if c['stage'] != 'locked_validation':
            continue
        key = c['dataset_id'] + '__' + c['column_id'][:16]
        selected.update((root / 'results/online' / key).glob('*/selected/frame.ngf'))
    files = [{'path': str(p.relative_to(root)), 'size': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(selected)]
    manifest = {'schema': 'harness-combo-review-subset-v1', 'run_root': str(root),
                'full_remote_evidence_retained': True, 'is_full_local_evidence_closure': False,
                'contents': 'all committed experiment JSON, reports, build/source/command receipts, binary, validation-selected frames; other frame payloads remain remote',
                'files': files}
    dest = root / 'delivery'
    dest.mkdir(exist_ok=True)
    m = dest / 'REVIEW_SUBSET_MANIFEST.json'
    assert not m.exists()
    m.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    archive = dest / 'review_subset.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tf:
        tf.add(m, arcname='REVIEW_SUBSET_MANIFEST.json', recursive=False)
        for f in files:
            tf.add(root / f['path'], arcname=f['path'], recursive=False)
    receipt = {'archive': str(archive), 'sha256': sha(archive), 'size': archive.stat().st_size,
               'manifest_sha256': sha(m), 'payload_files': len(files),
               'is_full_local_evidence_closure': False}
    (dest / 'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
