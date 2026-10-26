#!/usr/bin/env python3
"""Independent integration QA: canonical IDs, legacy replay, union dictionary.

Does not modify registry/dispatcher. Some codec domains intentionally reject
particular probes; only old-success/new-failure is treated as a migration gap.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct
import primitive_operator_action_runtime as p
import unified_action_runtime as u
from audit_current_action_ratio_identity import make_cases

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"analysis_outputs/canonical206_integration_review_v1"
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def audit(output):
    if output.exists():raise FileExistsError(output)
    source_paths=[ROOT/"scripts"/name for name in ("action_cpp_migration.py","primitive_operator_action_runtime.py","unified_action_runtime.py")]
    source_hashes={str(path.relative_to(ROOT)):sha(path.read_bytes()) for path in source_paths}
    with (ROOT/"analysis_outputs/action_ratio_identity_225_20260905_v1/actions.csv").open(encoding="utf-8-sig",newline="") as stream:
        old_rows=list(csv.DictReader(stream))
    old_names=[row["action"] for row in old_rows]
    assert old_names==[item.name for item in p.LEGACY_225_POLICY_ACTIONS]
    assert len(p.POLICY_ACTIONS)==len(u.CANONICAL_ACTIONS)==206
    assert len({item.name for item in p.POLICY_ACTIONS})==206
    assert [item.action_id for item in p.POLICY_ACTIONS]==list(range(206))
    for index,old in enumerate(old_names):
        assert p.POLICY_ACTION_BY_ID[p.LEGACY_225_POLICY_ID_TO_CURRENT[index]].name==p.ACTION_CANONICAL_NAMES[old]
    assert p.LEGACY_225_POLICY_ID_TO_CURRENT[225]==206
    cases,_=make_cases([128]);failures=[];rejections=[];covered=set();calls=0;old_replays=0;id_checks=0;sequence_checks=0
    excluded=set(p.CAMEL_ACTIONS)
    def invoke(name,data,point):
        nonlocal calls,id_checks
        output=u.run_action(name,data,max_point=point);calls+=1
        assert u.run_action(name,output,reverse=True)==data
        spec=p.POLICY_ACTION_BY_NAME[name]
        by_id=u.run_policy_action(spec.action_id,data,registry_version=p.ACTION_REGISTRY_VERSION,max_point=point)
        assert by_id==output
        assert u.run_policy_action(spec.action_id,by_id,registry_version=p.ACTION_REGISTRY_VERSION,reverse=True)==data
        id_checks+=1;covered.add(name);return output
    for label,typed,point in cases:
        roots=[typed] if typed.kind=="bytes" else [typed,u.ActionData.bytes(typed.payload)]
        for old_sequence in sorted(u._INDEPENDENT_STAGEWISE_SEQUENCES):
            if excluded.intersection(old_sequence):continue
            canonical=tuple(p.ACTION_CANONICAL_NAMES.get(name,name) for name in old_sequence)
            if any(name not in p.POLICY_ACTION_BY_NAME for name in canonical):continue
            first=u.describe_action(old_sequence[0])
            root=next((item for item in roots if item.kind in first.accepted_input_kinds),None)
            if root is None:continue
            # Keep prior harness mode mistake out of a C++ migration diagnosis.
            actual_point=None if old_sequence==p.BOS64_INT_ACTIONS and root.kind=="int64" else point
            old_current=new_current=root;old_ok=True
            for old,new in zip(old_sequence,canonical):
                try:
                    previous=old_current;old_current=u.run_action(old,previous,max_point=actual_point)
                    assert u.run_action(old,old_current,reverse=True)==previous
                    old_replays+=1
                except Exception as error:
                    rejections.append(dict(case=label,action=old,error=f"{type(error).__name__}: {error}"));old_ok=False;break
                try:new_current=invoke(new,new_current,actual_point)
                except Exception as error:
                    failures.append(dict(case=label,operation="canonical_stage",old=old,new=new,error=f"{type(error).__name__}: {error}"));old_ok=False;break
            if old_ok:
                if u._unpack_result(new_current)[1]!=u._unpack_result(old_current)[1]:
                    failures.append(dict(case=label,operation="chain_payload",old=" | ".join(old_sequence),new=" | ".join(canonical),error="final payload differs"))
                try:
                    whole=u.run_actions(canonical,root,max_point=actual_point)
                    assert u._unpack_result(whole)[1]==u._unpack_result(new_current)[1]
                    assert u.run_actions(canonical,whole,reverse=True,max_point=actual_point)==root
                    sequence_checks+=1
                except Exception as error:
                    failures.append(dict(case=label,operation="canonical_run_actions",old=" | ".join(old_sequence),new=" | ".join(canonical),error=f"{type(error).__name__}: {error}"))
        for spec in p.POLICY_ACTIONS:
            if spec.name in covered or spec.name in excluded:continue
            descriptor=u.describe_action(spec.name)
            for root in roots:
                if root.kind not in descriptor.accepted_input_kinds:continue
                try:invoke(spec.name,root,point)
                except Exception:continue
        print(f"integration {label}: covered {len(covered)}/206; gaps {len(failures)}",flush=True)
    words=[0,1<<63,0x3ff0000000000000,0x7ff0000000000000,(1<<64)-1-0xfffffffffffff]
    raw=struct.pack("<5Q",*words)
    sources=[u.ActionData("int64",raw,5),u.ActionData("float64",raw,5)]
    dictionary_outputs=[]
    for kind,source in zip(("int64","float64"),sources):
        canonical=u.run_action(p.cpp_migration.DICTIONARY,source)
        old=u.run_action(f"op.dictionary.project_{kind}.v1",source)
        assert u.run_action(p.cpp_migration.DICTIONARY,canonical,reverse=True)==source
        assert u.run_action(f"op.dictionary.project_{kind}.v1",old,reverse=True)==source
        assert u._unpack_result(old)[1]==u._unpack_result(canonical)[1]
        dictionary_outputs.append(u._unpack_result(canonical)[1])
    assert dictionary_outputs[0]==dictionary_outputs[1]
    rejected_guards=0
    for action,version in ((0,"old"),(206,p.ACTION_REGISTRY_VERSION),(225,p.ACTION_REGISTRY_VERSION)):
        try:u.run_policy_action(action,sources[0],registry_version=version)
        except u.ActionRuntimeError:rejected_guards+=1
        else:raise AssertionError("version/STOP guard failed")
    unchanged=all(sha(path.read_bytes())==source_hashes[str(path.relative_to(ROOT))] for path in source_paths)
    summary=dict(policy_actions=206,old_policy_actions=225,removed_implementations=19,canonical_covered=len(covered),
                 canonical_uncovered=sorted(set(p.POLICY_ACTION_BY_NAME)-covered),canonical_stage_calls=calls,
                 policy_id_name_parity_checks=id_checks,old_name_replay_roundtrips=old_replays,canonical_run_actions_checks=sequence_checks,
                 dictionary_union_bitwise=True,guard_rejections=rejected_guards,failures=failures,legacy_domain_rejections=rejections,
                 sources_unchanged_during_run=unchanged,source_hashes=source_hashes)
    output.mkdir(parents=True,exist_ok=False);(output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"failures","legacy_domain_rejections","source_hashes"}},indent=2))
    if failures:raise SystemExit(1)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=DEFAULT);args=parser.parse_args();audit(args.output)
