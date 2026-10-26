#!/usr/bin/env python3
"""Lossless single-stage deletion of each selected pure winner (not an RL run)."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--keys',nargs='*',help='Explicit targeted deletion scope, never labelled full20')
    args=p.parse_args()
    from canonical206_data import load_smallest_two
    sources={f"{r['dataset']}__{r['numeric_index']:02d}":r for r in load_smallest_two(ROOT)}
    grouped={k:[] for k in sources}
    manifests=[]
    for run in args.runs:
        run=run.resolve()
        comp=json.loads((run/'COMPLETE.json').read_text())
        assert comp['status']=='PASS'
        manifests.append({'path':str(run),'manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()})
        for path in sorted((run/'columns').glob('*.json')):
            row=json.loads(path.read_text())
            assert row['key'] in sources and row['status']=='PASS'
            assert row['input_sha256']==hashlib.sha256(sources[row['key']]['raw']).hexdigest()
            grouped[row['key']].append((row,run))
    assert all(grouped.values()),'Must cover all 20 canonical columns'
    if args.keys:
        assert len(set(args.keys))==len(args.keys) and set(args.keys)<=set(grouped)
        grouped={k:grouped[k] for k in args.keys}
    scope='targeted_winner_single_stage_deletion' if args.keys else 'full20_winner_single_stage_deletion'
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['columns','frames','traces']: (out/folder).mkdir()
    dump(out/'manifest.json',{'scope':scope,'source_runs':manifests,'keys':list(grouped),
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy':'Each deletion is a measured candidate; invalid typed chains are reported, not repaired. Iterative reduction only accepts strictly smaller byte-exact files. No entropy substitution or complete codecs.'})
    results=[]
    for key,parents in grouped.items():
        row,run=min(parents,key=lambda x:x[0]['best']['pure']['bytes'])
        original=row['best']['pure']
        frame_path=Path(original['frame_path'])
        if not frame_path.is_absolute(): frame_path=ROOT/frame_path
        original_frame=frame_path.read_bytes()
        version=int(chr(original_frame[3]))
        frame=importlib.import_module(f'atomic_composition_frame_v{min(version,2)}' if version<3 else 'atomic_composition_frame_v3')
        raw=sources[key]['raw']; kind=sources[key]['kind']
        assert frame.decode_frame(original_frame)==raw
        assert frame.encode_frame(raw,kind,original['specs'],original['terminals'])==original_frame
        specs=original['specs']; terminals=original['terminals']; current_bytes=len(original_frame)
        records=[]; seen=set(); rounds=0
        start=time.perf_counter()
        while True:
            best=None
            for i in range(len(specs)+len(terminals)):
                ns=specs[:i]+specs[i+1:] if i<len(specs) else list(specs)
                nt=list(terminals) if i<len(specs) else terminals[:i-len(specs)]+terminals[i-len(specs)+1:]
                recipe=json.dumps([ns,nt],sort_keys=True)
                if recipe in seen: continue
                seen.add(recipe)
                rec={'round':rounds,'removed_index':i,'removed_stage':specs[i] if i<len(specs) else terminals[i-len(specs)],
                    'parent_bytes':current_bytes,'specs':ns,'terminals':nt,'depth':len(ns)+len(nt),'family':'pure','origin':'single_stage_deletion'}
                begin=time.perf_counter()
                try:
                    buf=frame.encode_frame(raw,kind,ns,nt)
                    assert frame.decode_frame(buf)==raw
                    rec.update(status='ok',bytes=len(buf),delta_bytes=len(buf)-current_bytes,frame_sha256=hashlib.sha256(buf).hexdigest())
                    if len(buf)<current_bytes and (best is None or len(buf)<best[0]): best=(len(buf),ns,nt,buf)
                except (ValueError,OverflowError) as exc:
                    rec.update(status='invalid',error=str(exc))
                rec['measured_sec']=time.perf_counter()-begin
                records.append(rec)
            if best is None: break
            current_bytes,specs,terminals,original_frame=best
            rounds+=1
            assert rounds<=10
        # Re-measure the final recipe, with no prepared payload or cache.
        warm=frame.encode_frame(raw,kind,specs,terminals)
        times=[]
        for _ in range(3):
            begin=time.perf_counter(); buf=frame.encode_frame(raw,kind,specs,terminals)
            times.append(time.perf_counter()-begin)
            assert buf==warm and frame.decode_frame(buf)==raw
        target=out/'frames'/f'{key}.acf{version}'
        target.write_bytes(warm)
        best={'family':'pure','specs':specs,'terminals':terminals,'depth':len(specs)+len(terminals),
            'bytes':len(warm),'ratio':len(raw)/len(warm),'frame_path':str(target),
            'frame_sha256':hashlib.sha256(warm).hexdigest(),'compress_sec':statistics.median(times),
            'compress_sec_repeats':times,'frame_inspection':frame.inspect_frame(warm)}
        result={'key':key,'dataset':row['dataset'],'name':row['name'],'input_sha256':row['input_sha256'],
            'raw_bytes':len(raw),'status':'PASS','parent_run':str(run),'parent_pure_bytes':original['bytes'],
            'accepted_deletions':rounds,'measured_candidates':len(records),
            'successful_candidates':sum(r['status']=='ok' for r in records),
            'compose_sec':time.perf_counter()-start,'best':{'pure':best}}
        dump(out/'columns'/f'{key}.json',result)
        (out/'traces'/f'{key}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        results.append(result)
        print(json.dumps({'key':key,'bytes':best['bytes'],'deletions':rounds,'attempts':len(records)}),flush=True)
    dump(out/'COMPLETE.json',{'status':'PASS','scope':scope,'columns':len(results),
        'attempts':sum(r['measured_candidates'] for r in results),'successes':sum(r['successful_candidates'] for r in results),
        'improved_columns':sum(r['accepted_deletions']>0 for r in results)})

if __name__=='__main__': main()
