#!/usr/bin/env python3
"""Package hash-bound metadata; full production frames stay on the benchmark host."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def package(root):
    root=root.resolve(strict=True)
    audit=json.loads((root/'audit/AUDIT.json').read_text())
    protection=json.loads((root/'PROTECTION_POST.json').read_text())
    if audit['status']!='pass' or protection['status']!='pass':raise RuntimeError('audit/protection not passed')
    selected=set()
    for directory in ['source','delivery','report','model','audit','witness_final','agent_doc_witness','cli_example']:
        for p in (root/directory).rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='BAD_MODEL.json':selected.add(p)
    selected.add(root/'inputs/INPUTS.json')
    for p in (root/'campaign').rglob('*'):
        if p.is_file() and p.suffix in {'.json','.sha256'} and p.name!='PHASE.json':selected.add(p)
    for directory in ['build','serve_build','production_supervisor','witness_supervisor_final']:
        for p in (root/directory).glob('*'):
            if p.is_file() and p.suffix in {'.json','.map'}:selected.add(p)
    for p in root.glob('*.json'):selected.add(p)
    selected.add(root/'new_input.json')
    selected={p for p in selected if p.name not in {'DELIVERY_FILES_SHA256.json','DELIVERY_INDEX.json'}}
    files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(selected)}
    manifest={'schema':'operator-recommend135-delivery-files-v1','source_root':str(root),'files':files,
              'scope':'full metadata/receipts/models/reports and small witness bytes; production frame/value bytes remain on the benchmark host and are bound by the independent full-root audit'}
    manifest_path=root/'DELIVERY_FILES_SHA256.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=root/'EVIDENCE_FINAL.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as t:
        for p in sorted(selected):t.add(p,arcname=str(p.relative_to(root)),recursive=False)
        t.add(manifest_path,arcname=manifest_path.name,recursive=False)
    receipt={'schema':'operator-recommend135-delivery-index-v1','archive':str(archive),'bytes':archive.stat().st_size,
             'sha256':sha(archive),'manifest_sha256':sha(manifest_path),'payload_file_count':len(files),
             'total_uncompressed_payload_bytes':sum(x['bytes'] for x in files.values()),'scope':manifest['scope']}
    (root/'DELIVERY_INDEX.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();package(a.root)
