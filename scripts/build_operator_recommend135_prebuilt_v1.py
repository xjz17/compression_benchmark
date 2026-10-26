#!/usr/bin/env python3
"""Compile only new C++ sources against the frozen v4 archive; no old-tree writes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build(base, overlay, source, output):
    base, overlay, source = (p.resolve(strict=True) for p in (base, overlay, source))
    output = output.resolve()
    if output.is_relative_to(base) or output.is_relative_to(overlay):
        raise ValueError('new build must be outside both frozen runtime trees')
    output.mkdir(parents=True, exist_ok=True)
    cache = base / 'build/CMakeCache.txt'
    if 'NATIVE_GRAPH_ENABLE_NNCP:BOOL=OFF' not in cache.read_text():
        raise ValueError('NNCP must be OFF')
    receipt_path = overlay / 'build/BUILD_RECEIPT.json'
    if not receipt_path.exists():
        receipt_path = overlay / 'BUILD_RECEIPT.json'
    old = json.loads(receipt_path.read_text())
    if old['NNCP'] != 'OFF':
        raise ValueError('v4 NNCP state is not OFF')
    link = next(c for c in old['commands'] if '-o' in c and
                Path(c[c.index('-o')+1]).name == 'operator_harness_v2_cli')
    libraries = link[link.index('-Wl,--whole-archive'):]
    flag_file = base / 'build/CMakeFiles/operator_harness_v2.dir/flags.make'
    assignments = dict(line.split(' = ', 1) for line in flag_file.read_text().splitlines() if ' = ' in line)
    flags = sum((shlex.split(assignments[k]) for k in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS')), [])
    flags = ['-I'+str(source/'cpp/operator_recommend135_v1'),
             '-I'+str(overlay/'source/cpp/operator_harness_v2'),
             '-I'+str(overlay/'source/cpp/src'),
             '-I'+str(overlay/'source/cpp/native_graph_v1')] + flags
    flags += ['-I'+str(base/'source/external/compression/lz4/lib'),
              '-I'+str(base/'source/external/compression/zstd/lib'), '-ffp-contract=off']
    protected = {cache, flag_file, receipt_path}
    for token in libraries:
        if not token.startswith('-'):
            protected.add(Path(token).resolve(strict=True))
    for p in old.get('private_codec_plugins',{}).values():
        # Different receipt versions store either a path or structured binding.
        if isinstance(p,str) and Path(p).exists(): protected.add(Path(p))
    protected.update((overlay/'build').glob('liboperator102_blosc*_v4.so'))
    for root in (overlay/'source/cpp',base/'source/cpp'):
        for directory in ('operator_harness_v2','native_graph_v1','src'):
            protected.update((root/directory).rglob('*.hpp'))
            protected.update((root/directory).rglob('*.h'))
    for directory in (base/'source/external/compression/lz4/lib',
                      base/'source/external/compression/zstd/lib',base/'deps/include/nlohmann'):
        protected.update(directory.rglob('*.h'));protected.update(directory.rglob('*.hpp'))
    before = {str(p):sha(p) for p in sorted(protected)}
    paths = sorted((source/'cpp/operator_recommend135_v1').glob('*.cpp'))
    headers = sorted((source/'cpp/operator_recommend135_v1').glob('*.hpp'))
    source_hash = {str(p):sha(p) for p in paths+headers}
    previous_path=output/'BUILD_RECEIPT.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
    headers_same=all(previous.get('source_sha256',{}).get(str(p))==source_hash[str(p)] for p in headers)
    commands=[]
    def run(command):
        commands.append(command)
        print(json.dumps({'command':command}),flush=True)
        subprocess.run(command,check=True)
    objects={}
    for path in paths:
        obj=output/(path.stem+'.o')
        command=['/usr/bin/c++',*flags,'-c',str(path),'-o',str(obj)]
        if not (headers_same and obj.exists() and previous.get('source_sha256',{}).get(str(path))==source_hash[str(path)] and command in previous.get('commands',[])):
            run(command)
        else:commands.append(command)
        objects[path.stem]=str(obj)
    binaries={}
    targets={'operator_recommend135_cli':['main','campaign','runtime','model','input_builder'],
             'campaign_test':['campaign_test','campaign','runtime','model'],
             'runtime_test':['runtime_test','runtime'],
             'input_builder_test':['input_builder_test','input_builder'],
             'model_test':['model_test','model']}
    for target, members in targets.items():
        binary=output/target
        run(['/usr/bin/c++','-O3',*(objects[m] for m in members),'-o',str(binary),
             '-Wl,-Map,'+str(output/(target+'.map')),*libraries,'-pthread'])
        binaries[target]=sha(binary)
    after={str(p):sha(p) for p in sorted(protected)}
    if before!=after: raise RuntimeError('frozen runtime changed during build')
    if source_hash!={str(p):sha(p) for p in paths+headers}:raise RuntimeError('source changed during build')
    tests={}
    for name in ('input_builder_test','runtime_test','model_test','campaign_test'):
        tested=subprocess.run([str(output/name)],text=True,capture_output=True)
        tests[name]={'returncode':tested.returncode,'stdout':tested.stdout,'stderr':tested.stderr}
        if tested.returncode: raise RuntimeError(json.dumps(tests[name]))
    receipt={'schema':'operator-recommend135-build-v1','status':'built-and-tested',
             'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'NNCP':'OFF','excluded_action_ids':[106,107,108,109,110],
             'frozen_runtime_before':before,'frozen_runtime_after':after,
             'source_sha256':source_hash,'binaries':binaries,'commands':commands,'tests':tests,
             'maps_sha256':{p.name:sha(p) for p in output.glob('*.map')}}
    atomic_json(output/'BUILD_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('base','overlay','source','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    result=build(args.base,args.overlay,args.source,args.output)
    print(json.dumps({'status':result['status'],'binaries':result['binaries']}))
