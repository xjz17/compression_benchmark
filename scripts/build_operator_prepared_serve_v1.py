#!/usr/bin/env python3
"""Link a resident CLI to the already frozen, measured C++ objects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(root, source, output):
    root = root.resolve(strict=True)
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(root / 'source') or output.is_relative_to(root / 'build'):
        raise ValueError('serving build must not overwrite the measured source or build')
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = root / 'build/BUILD_RECEIPT.json'
    old = json.loads(receipt_path.read_text())
    if old['NNCP'] != 'OFF' or old['excluded_action_ids'] != [106, 107, 108, 109, 110]:
        raise ValueError('frozen exclusion policy mismatch')
    protected = dict(old['frozen_runtime_before'])
    protected.update(old['source_sha256'])
    protected[str(receipt_path)] = sha(receipt_path)
    objects = [root / 'build' / (n + '.o') for n in ('campaign', 'runtime', 'model', 'input_builder')]
    protected.update({str(p): sha(p) for p in objects})
    for n, s in old['binaries'].items():
        protected[str(root / 'build' / n)] = s
    before = {p: sha(p) for p in protected}
    if before != protected:
        raise RuntimeError('frozen core differs from measured build')
    original = next(c for c in old['commands'] if '-c' in c and
                    Path(c[c.index('-c') + 1]).name == 'prepared_replay.cpp')
    flags = original[1:original.index('-c')]
    link = next(c for c in old['commands'] if '-o' in c and
                Path(c[c.index('-o') + 1]).name == 'prepared_replay')
    libraries = link[link.index('-Wl,--whole-archive'):]
    obj = output / 'prepared_serve.o'
    binary = output / 'prepared_serve'
    commands = [
        ['/usr/bin/c++', *flags, '-c', str(source), '-o', str(obj)],
        ['/usr/bin/c++', '-O3', str(obj), *(str(p) for p in objects),
         '-o', str(binary), '-Wl,-Map,' + str(output / 'prepared_serve.map'), *libraries],
    ]
    source_sha = sha(source)
    for command in commands:
        print(json.dumps({'command': command}), flush=True)
        subprocess.run(command, check=True)
    after = {p: sha(p) for p in protected}
    if after != before or sha(source) != source_sha:
        raise RuntimeError('serving build modified frozen dependencies or source')
    result = {'schema': 'operator-prepared135-resident-build-v1', 'status': 'built',
              'NNCP': 'OFF', 'excluded_action_ids': [106, 107, 108, 109, 110],
              'source': str(source), 'source_sha256': source_sha,
              'binary_sha256': sha(binary), 'commands': commands,
              'protected_files_before': before, 'protected_files_after': after,
              'link_map_sha256': sha(output / 'prepared_serve.map'),
              'scope': 'same frozen core objects; no claim of additional measurement until serving smoke receipts pass'}
    (output / 'BUILD_RECEIPT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'built', 'binary_sha256': result['binary_sha256']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.source, a.output)
