#!/usr/bin/env python3
"""Propose only same-min/effective-max Native-LZ aliases, with full-wire checks.

Read-only for the registry. Intermediate NLS1 maximum fields differ; aliases
are valid only after canonicalizing that configuration and inverse context.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import primitive_operator_action_runtime as primitive

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/"analysis_outputs/action_ratio_identity_225_20260905_v1"
INPUTS=ROOT/"analysis_outputs/python_stage_cpp_port_20260905_v1/golden.json"
DEFAULT=ROOT/"analysis_outputs/native_lz_conservative_aliases_20260905_v1"
sys.path.insert(0,str(ROOT/"backend"))
from myapp.cpp_action_bridge import run

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))
def write_csv(path,records):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
def bounds(name):
    found=re.fullmatch(r"op\.lz_match\.native_lz\.aligned8\.min(\d+)\.max(\d+)",name)
    if not found:raise ValueError(name)
    return tuple(map(int,found.groups()))

def chain(raw,window,match):
    minimum,maximum=bounds(match)
    actions=(window,match,"op.match_parse_policy.native_lz.aligned8_greedy",f"op.match_token_format.native_lz.ctrl8_offset16_len8.bias{minimum}")
    previous=raw;packets=[]
    for action in actions:
        current=run("native_lz_action_stages_cli","forward",previous,action)
        packets.append(current);previous=current
    ctx=struct.unpack_from("<4sBIIIQ",packets[-2])
    recorded=(ctx[2],ctx[3],ctx[4],ctx[5])
    current=packets[-1]
    for index in range(3,-1,-1):
        current=run("native_lz_action_stages_cli","inverse",current,actions[index],*(recorded if index==3 else ()))
        expected=raw if index==0 else packets[index-1]
        if current!=expected:raise AssertionError(f"inverse mismatch: {actions[index]}")
    return packets

def audit(output,workers):
    if output.exists():raise FileExistsError(f"refusing to overwrite {output}")
    frozen_hashes={name:sha((FROZEN/name).read_bytes()) for name in ("actions.csv","cases.csv","pairs.csv","summary.json")}
    matches=sorted(row["action"] for row in read_csv(FROZEN/"actions.csv") if row["action"].startswith("op.lz_match.native_lz.aligned8.min"))
    groups={}
    for name in matches:
        minimum,maximum=bounds(name);groups.setdefault((minimum,maximum//8*8),[]).append(name)
    if len(matches)!=42 or len(groups)!=27:raise AssertionError((len(matches),len(groups)))
    canonical={name:min(members,key=lambda n:bounds(n)[1]) for members in groups.values() for name in members}
    aliases=[dict(old_action=name,canonical_action=canonical[name],operator_id="lz_match",minimum=bounds(name)[0],
                  old_maximum=bounds(name)[1],canonical_maximum=bounds(canonical[name])[1],effective_maximum=bounds(name)[1]//8*8)
             for name in matches if canonical[name]!=name]
    if len(aliases)!=15:raise AssertionError(len(aliases))
    frozen_cases=read_csv(FROZEN/"cases.csv")
    cases={}
    for record in json.loads(INPUTS.read_text())["records"]:
        if record["module"]=="corad" and record["stage"]=="normalize" and ":" in record["case"]:
            raw=base64.b64decode(record["input_b64"])
            if sha(raw) not in {r["payload_sha256"] for r in frozen_cases}:raise AssertionError("real source changed")
            cases[record["case"]]=raw
    if len(cases)!=16:raise AssertionError("expected sixteen frozen real slices")
    # Every old match is exercised at a common maximal window. Also include its
    # real method-specific windows from all currently declared Native-LZ methods.
    contexts={("op.match_window_model.native_lz.window65535",name) for name in matches}
    methods=[]
    for method,key in primitive.EXACT_NATIVE_METHOD_KEYS.items():
        preset=primitive.native_lz_preset(key)
        window=f"op.match_window_model.native_lz.window{preset.window}"
        old=f"op.lz_match.native_lz.aligned8.min{preset.min_match}.max{preset.max_match}"
        if old not in canonical:raise AssertionError(f"method not in frozen42: {method}")
        contexts.add((window,old));methods.append(dict(method=method,key=key,window=window,old_action=old,canonical_action=canonical[old]))
    tasks={(case,window,match) for case in cases for window,old in contexts for match in (old,canonical[old])}
    measured={};measurements=[]
    def one(task):
        case,window,match=task;packets=chain(cases[case],window,match)
        return task,packets
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(one,task) for task in sorted(tasks)]
        for number,future in enumerate(as_completed(futures),1):
            task,packets=future.result();case,window,match=task
            measured[task]=packets
            measurements.append(dict(case=case,window=window,action=match,raw_bytes=len(cases[case]),raw_sha256=sha(cases[case]),
                                     match_packet_bytes=len(packets[1]),match_packet_sha256=sha(packets[1]),wire_bytes=len(packets[-1]),wire_sha256=sha(packets[-1]),
                                     exact_stage_inverses=4,full_chain_inverse=True))
            if number%200==0:print(f"native full-chain checks: {number}/{len(tasks)}",flush=True)
    comparisons=[]
    for case in sorted(cases):
        for window,old in sorted(contexts):
            new=canonical[old];left=measured[case,window,old];right=measured[case,window,new]
            row=dict(case=case,window=window,old_action=old,canonical_action=new,final_wire_identical=left[-1]==right[-1],
                     final_size_identical=len(left[-1])==len(right[-1]),parse_tokens_identical=left[-2][25:]==right[-2][25:],
                     intermediate_match_packet_identical=left[1]==right[1])
            if not row["final_wire_identical"] or not row["parse_tokens_identical"]:raise AssertionError(row)
            comparisons.append(row)
    # Cross-check the existing primitive entrypoint on every old and retained
    # action using one of the frozen 128-prefix inputs, not a new transport.
    case=next(name for name in sorted(cases) if name.endswith(":128"));raw=cases[case]
    primitive_checks=0
    for name in matches:
        window="op.match_window_model.native_lz.window65535";minimum,_=bounds(name)
        data=raw
        for action in (window,name,"op.match_parse_policy.native_lz.aligned8_greedy",f"op.match_token_format.native_lz.ctrl8_offset16_len8.bias{minimum}"):
            data=primitive.encode_exact_native_lz_stage(action,data)
        if data!=measured[case,window,name][-1]:raise AssertionError("primitive vs native full wire mismatch")
        primitive_checks+=1
    unchanged=all(sha((FROZEN/name).read_bytes())==digest for name,digest in frozen_hashes.items())
    for alias in aliases:
        rows=[r for r in comparisons if r["old_action"]==alias["old_action"]]
        alias.update(real_source_cases=len({r["case"] for r in rows}),tested_windows=len({r["window"] for r in rows}),
                     exact_full_wire_checks=len(rows),all_full_wires_identical=all(r["final_wire_identical"] for r in rows))
    summary=dict(status="PASS",old_match_actions=42,canonical_match_actions=27,redundant_actions=15,real_source_cases=16,
                 source_kind="frozen 8 real columns x requested128/2048",registered_method_contexts=len(methods),
                 distinct_old_window_match_contexts=len(contexts),distinct_executed_chains=len(tasks),individual_stage_inverse_checks=4*len(tasks),
                 full_wire_comparisons=len(comparisons),full_wire_comparisons_exact=len(comparisons),primitive_native_full_wire_checks=primitive_checks,
                 frozen_audit_unchanged=unchanged,registry_modified=False,aliases=aliases,
                 source_hashes={str(path.relative_to(ROOT)):sha(path.read_bytes()) for path in (Path(__file__),ROOT/"cpp/src/native_lz_action_stages.cpp",ROOT/"cpp/build/native_lz_action_stages_cli",INPUTS)},
                 scope=["Conservative equality key is exact original min and floor(max/8)*8; different min biases are NOT merged.",
                        "Current aligned-8 fast parser only uses the effective maximum. Chain/lazy mode is outside this alias proposal.",
                        "NLS1 intermediate maximum fields intentionally differ; final codec wire and parse tokens agree after valid full-chain execution.",
                        "Applying aliases requires canonicalizing action names and inverse-context parameters; this is not arbitrary packet byte identity.",
                        "Common65535 window covers all42 actions; additional windows are all current method-specific contexts, not a full14x42 Cartesian test.",
                        "This checks alias semantics of the newly native parser. Independent old-Python-to-C++ parity is a separate frozen-golden migration result."])
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/"aliases.csv",aliases);write_csv(output/"method_contexts.csv",methods)
    write_csv(output/"chain_measurements.csv",sorted(measurements,key=lambda r:(r["case"],r["window"],r["action"])))
    write_csv(output/"full_wire_comparisons.csv",comparisons)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({key:value for key,value in summary.items() if key not in {"source_hashes","aliases"}},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=DEFAULT);parser.add_argument("--workers",type=int,default=8);args=parser.parse_args();audit(args.output,args.workers)
