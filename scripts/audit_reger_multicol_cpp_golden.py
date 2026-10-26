#!/usr/bin/env python3
"""Freeze original REGER3 multi-column selection before removing Python kernels."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))
DEFAULT=ROOT/"analysis_outputs/reger_multicol_cpp_port_20260905_v1"

def packet(columns,times=None):
    rows=min(map(len,columns),default=0)
    raw=bytearray(struct.pack("<4sQIB",b"RGMI",rows,len(columns),times is not None))
    if times is not None: raw.extend(struct.pack("<"+"q"*rows,*times[:rows]))
    for column in columns: raw.extend(struct.pack("<"+"q"*rows,*column[:rows]))
    return bytes(raw)

def freeze(output):
    raise RuntimeError("Historical Python REGER retired; use immutable pre-migration goldens with verify.")
    from myapp import reger_codec as original
    rng=random.Random(101)
    cases=[]
    for columns,rows in ((0,0),(1,0),(3,1),(3,17),(3,180),(2,512),(16,33),(32,17)):
        data=[[10000-(i//4)*11+(i%4)+c*17 for i in range(rows)] for c in range(columns)]
        cases.append((f"correlated_{columns}_{rows}",data,None,{}))
        if columns: cases.append((f"times_{columns}_{rows}",data,[1700000000000000+i*60000000 for i in range(rows)],{}))
    cases.append(("random",[[rng.randrange(-10000,10000) for _ in range(40)] for _ in range(3)],None,{}))
    cases.append(("times_precision",[[i*17 for i in range(41)],[(i%7)*11 for i in range(41)]],[(1<<53)+1+i*17 for i in range(41)],{}))
    cases.append(("custom_sizes",[[i*i for i in range(40)],[i%3 for i in range(40)]],None,{"COMPRESSION_BENCHMARK_REGER_BLOCK_SIZE":"17","COMPRESSION_BENCHMARK_REGER_SEGMENT_SIZE":"4"}))
    cases.append(("fast",[[i*i for i in range(40)],[i%3 for i in range(40)]],None,{"COMPRESSION_BENCHMARK_REGER_PROFILE":"fast"}))
    records=[]
    for name,columns,times,env in cases:
        previous={key:os.environ.get(key) for key in env}
        os.environ.update(env)
        try: wire=original.encode_int_columns(columns,times)
        finally:
            for key,value in previous.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value
        decoded=original.decode_int_columns(wire)
        expected=packet(decoded)
        records.append(dict(case=name,environment=env,input_b64=base64.b64encode(packet(columns,times)).decode(),wire_b64=base64.b64encode(wire).decode(),decoded_b64=base64.b64encode(expected).decode()))
    output.mkdir(parents=True,exist_ok=False)
    source=ROOT/"backend/myapp/reger_codec.py"
    (output/"golden.json").write_text(json.dumps(dict(records=records,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+"\n")
    print(json.dumps(dict(cases=len(records),output=str(output))))

def verify(output,binary):
    result=[]
    for record in json.loads((output/"golden.json").read_text())["records"]:
        row=dict(case=record["case"])
        for direction,source,target in (("forward","input_b64","wire_b64"),("inverse","wire_b64","decoded_b64")):
            expected=base64.b64decode(record[target]); proc=subprocess.run([binary,"columns-"+direction],input=base64.b64decode(record[source]),capture_output=True,env=os.environ|record["environment"])
            row[direction]=proc.returncode==0 and proc.stdout==expected
            if not row[direction]:row[direction+"_error"]=proc.stderr.decode(errors="replace");row[direction+"_sizes"]=[len(proc.stdout),len(expected)]
        result.append(row)
    summary=dict(records=len(result),exact=sum(r["forward"] and r["inverse"] for r in result),results=result)
    (output/"verification.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    if summary["exact"]!=summary["records"]:raise SystemExit(1)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("freeze","verify"));parser.add_argument("--output",type=Path,default=DEFAULT);parser.add_argument("--binary",default="/tmp/reger_legacy_cpp");args=parser.parse_args()
    if args.mode=="freeze":freeze(args.output)
    else:verify(args.output,args.binary)
