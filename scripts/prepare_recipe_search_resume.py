#!/usr/bin/env python3
"""Preserve every completed result when resuming after a decode-timeout fix.

No outcome filter: all completed source rows, frames, policies and traces are
copied with hashes and original code provenance. The original run is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from canonical182_policy_paths import require_current_registry, path_from_record

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prior=args.prior.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    if (prior/'COMPLETE.json').exists():raise ValueError('prior is already complete')
    old=json.loads((prior/'manifest.json').read_text())
    rows=list(sorted((prior/'columns').glob('*.json')))
    if not rows:raise ValueError('no completed rows to preserve')
    for path in rows:
        r=json.loads(path.read_text())
        require_current_registry(r)
        path_from_record(r)
        if not r['lossless'] or sha(prior/r['frame_path'])!=r['frame_sha256']:
            raise AssertionError('prior completed result failed its frame integrity check')
    output.mkdir()
    records=[]
    for folder in ('columns','frames','traces','policies'):
        source=prior/folder
        if not source.exists():continue
        (output/folder).mkdir()
        for path in sorted(source.iterdir()):
            if not path.is_file():raise ValueError('unexpected nested input directory')
            rel=str(path.relative_to(prior));target=output/rel
            shutil.copy2(path,target)
            digest=sha(path)
            if sha(target)!=digest:raise AssertionError('copy differs')
            records.append({'relative_path':rel,'origin':str(path),'sha256':digest})
    (output/'code').mkdir()
    for relative in old['source_hashes']:
        shutil.copy2(ROOT/relative,output/'code'/Path(relative).name)
    shutil.copy2(prior/'option_catalog.json',output/'option_catalog.json')
    shutil.copytree(prior/'code',output/'prior_code')
    shutil.copy2(prior/'manifest.json',output/'prior_manifest.json')
    manifest={**old,'created':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'source_hashes':{p:sha(ROOT/p) for p in old['source_hashes']},
              'resumed_from':str(prior),'adopted_completed_rows':len(rows),
              'prior_manifest_sha256':sha(prior/'manifest.json'),
              'adoption_policy':'all previously completed rows without outcome filtering; exact source/frame/trace copies; old code in prior_code',
              'source_change_reason':'Only handle full-frame validation exceptions as rejected candidate rather than aborting run. Original completed rows already passed this validation.'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output/'ADOPTED_RESULTS.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(output),'adopted_rows':len(rows),'copied_files':len(records)},ensure_ascii=False))

if __name__=='__main__':main()
