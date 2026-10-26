#!/usr/bin/env python3
"""Build the non-raw serving/replay adapter without changing measured binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build(root):
    root=root.resolve(strict=True);prior=root/'build';output=root/'serve_build';output.mkdir(exist_ok=False)
    original=json.loads((prior/'BUILD_RECEIPT.json').read_text())
    compile_main=next(c for c in original['commands'] if '-c' in c and Path(c[c.index('-c')+1]).name=='main.cpp')
    flags=compile_main[1:compile_main.index('-c')]
    link=next(c for c in original['commands'] if '-o' in c and Path(c[c.index('-o')+1]).name=='operator_recommend135_cli')
    libraries=link[link.index('-Wl,--whole-archive'):]
    reused={str(prior/(s+'.o')):sha(prior/(s+'.o')) for s in ['campaign','model','runtime']}
    reused.update(original['frozen_runtime_before']);reused[str(prior/'operator_recommend135_cli')]=sha(prior/'operator_recommend135_cli')
    for p,s in reused.items():
        if sha(p)!=s:raise RuntimeError('prebuilt binding changed: '+p)
    sources=[root/'source/cpp/operator_recommend135_v1'/(name+'.cpp') for name in ['compression_serve','compression_serve_test']]
    source_hash={str(p):sha(p) for p in sources};commands=[]
    for source in sources:
        obj=output/(source.stem+'.o');command=['/usr/bin/c++',*flags,'-c',str(source),'-o',str(obj)]
        commands.append(command);subprocess.run(command,check=True)
        binary=output/source.stem
        command=['/usr/bin/c++','-O3',str(obj),*[str(prior/(s+'.o')) for s in ['campaign','runtime','model']],
                 '-o',str(binary),'-Wl,-Map,'+str(output/(source.stem+'.map')),*libraries,'-pthread']
        commands.append(command);subprocess.run(command,check=True)
    test=subprocess.run([str(output/'compression_serve_test')],text=True,capture_output=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    after={p:sha(p) for p in reused}
    if after!=reused:raise RuntimeError('measured runtime changed during adapter build')
    if source_hash!={str(p):sha(p) for p in sources}:raise RuntimeError('adapter source changed during build')
    receipt={'schema':'operator-recommend135-serve-build-v1','status':'built-and-tested','NNCP':'OFF',
             'source_sha256':source_hash,'reused_before':reused,'reused_after':after,
             'binaries':{p.name:sha(p) for p in output.iterdir() if p.name in ['compression_serve','compression_serve_test']},
             'maps_sha256':{p.name:sha(p) for p in output.glob('*.map')},'commands':commands,
             'tests':{'compression_serve_test':{'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr}}}
    (output/'SERVE_BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'binaries':receipt['binaries']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();build(a.root)
