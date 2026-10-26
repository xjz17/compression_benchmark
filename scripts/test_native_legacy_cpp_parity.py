#!/usr/bin/env python3
"""Freeze old Python action payloads, then verify a pure C++ stage implementation.

The verify mode deliberately imports no retired Python codec kernels. Inputs and
expected bytes are content-addressed artifacts; source hashes record provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "analysis_outputs/native_legacy_cpp_parity_20260905_v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(output):
    raise RuntimeError("Historical Python kernels retired; verify the existing frozen oracle.")
    sys.path.insert(0, str(ROOT))
    os.environ["WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES"] = "0"
    import audit_86_composability_realdata as audit
    import primitive_operator_action_runtime as p
    from backend.myapp import legacy_numeric_stages as legacy
    from backend.myapp.native_lz_family import compress_native_lz_family, _preset

    if (output / "golden.json").exists():
        raise RuntimeError("frozen goldens already exist; use a new output directory")
    (output / "blobs").mkdir(parents=True, exist_ok=True)
    def blob(raw):
        key = sha(raw)
        path = output / "blobs" / key
        if not path.exists():
            path.write_bytes(raw)
        return key

    fixtures, datasets = [], []
    for requested in (128, 2048):
        cols, manifest = audit.load_external_columns(requested)
        datasets.extend(manifest)
        for col in cols:
            fixtures.append(dict(name=f"{col.dataset}/{col.name}@{requested}",
                                 kind="int64" if col.logical_type == "integer" else "float64",
                                 max_point=col.max_point, input=blob(col.raw), count=len(col.raw)//8,
                                 requested_rows=requested, actual_rows=len(col.raw)//8, real=True))
    rng = random.Random(1729)
    far = rng.randbytes(16392)
    edges = [
        ("empty-int", "int64", 0, b""),
        ("empty-float", "float64", 5, b""),
        ("i64-extremes", "int64", 0, struct.pack("<8q", -(1<<63), (1<<63)-1, 0, -1, 1, (1<<53)+1, -(1<<53)-1, 42)),
        ("f64-bits", "float64", 15, struct.pack("<9Q", 0, 1<<63, 1, 0x0010000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000001234, 0x3ff0000000000001, 0x7fefffffffffffff)),
        ("f64-decimal", "float64", 3, struct.pack("<8d", 0.0, 1.125, -1.125, 0.1, 0.2, 2.5, -2.5, 1000.125)),
        ("f64-single", "float64", 2, struct.pack("<d", 1.23)),
        ("f64-constant-tail", "float64", 2, struct.pack("<1025d", *([12.25]*1025))),
        ("raw-tail7", "bytes", None, b"abcdefg"),
        ("zeros1024", "bytes", None, bytes(1024)),
        ("repeat4096", "bytes", None, bytes(range(64))*64),
        ("far-window", "bytes", None, far+far[:256]),
    ]
    for name, kind, point, raw in edges:
        fixtures.append(dict(name=name, kind=kind, max_point=point, input=blob(raw), count=len(raw)//8, real=False))

    records, chains, rejected = [], [], []
    def record(family, fixture, action, before, after, context=None):
        item = dict(family=family, fixture=fixture["name"], action=action,
                    input=blob(before), output=blob(after), context=context or {},
                    kind=fixture["kind"], max_point=fixture["max_point"], count=fixture["count"])
        records.append(item)
    for fixture in fixtures:
        raw = (output/"blobs"/fixture["input"]).read_bytes()
        for key in sorted(set(p.EXACT_NATIVE_METHOD_KEYS.values())):
            actions = p.exact_native_lz_action_pipeline(key)
            current = raw
            for spec in actions:
                context = p.native_lz_stage_packet_context(current) if p.exact_native_lz_stage_role(spec.name) == "token_format" else {}
                encoded = p.encode_exact_native_lz_stage(spec.name, current)
                assert p.decode_exact_native_lz_stage(spec.name, encoded, context=context) == current
                record("native", fixture, spec.name, current, encoded, context)
                current = encoded
            original = compress_native_lz_family(raw, key)
            assert current == original
            preset = _preset(key)
            chains.append(dict(family="native", fixture=fixture["name"], algorithm=key,
                               input=fixture["input"], output=blob(original),
                               actions=[s.name for s in actions],
                               window=preset.window, minimum=preset.min_match, maximum=preset.max_match))
        if fixture["kind"] == "bytes":
            continue
        for family, actions in (("scaled", legacy.SCALED_ACTIONS), ("alp", legacy.ALP_ACTIONS)):
            if family == "alp" and fixture["kind"] != "float64":
                continue
            current = raw
            try:
                for action in actions:
                    encoded = legacy.forward(action, current, fixture["kind"], fixture["max_point"])
                    assert legacy.inverse(action, encoded, count=fixture["count"]) == current
                    record("legacy", fixture, action, current, encoded)
                    current = encoded
            except (ValueError, OverflowError) as error:
                rejected.append(dict(fixture=fixture["name"], family=family, action=action,
                                     input=blob(current), kind=fixture["kind"], max_point=fixture["max_point"],
                                     error=type(error).__name__+": "+str(error)))
                continue
            chains.append(dict(family=family, fixture=fixture["name"], input=fixture["input"],
                               output=blob(current), actions=list(actions)))
    sources = ["backend/myapp/native_lz_family.py", "backend/myapp/legacy_numeric_stages.py",
               "backend/myapp/research_codec_python.py", "scripts/primitive_operator_action_runtime.py"]
    result = dict(schema=1, fixtures=fixtures, datasets=datasets, records=records, chains=chains,
                  rejected=rejected, source_sha256={s:sha((ROOT/s).read_bytes()) for s in sources})
    (output/"golden.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(saved=str(output/"golden.json"), fixtures=len(fixtures), stages=len(records),
                          chains=len(chains), rejected=len(rejected),
                          native_actions=len({r["action"] for r in records if r["family"] == "native"}),
                          legacy_actions=len({r["action"] for r in records if r["family"] == "legacy"})), indent=2))


def verify(output, native_cli, legacy_cli):
    golden = json.loads((output/"golden.json").read_text())
    def run(row, inverse, raw):
        cli = native_cli if row["family"] == "native" else legacy_cli
        command = [str(cli), "inverse" if inverse else "forward", row["action"]]
        if row["family"] == "native":
            ctx = row["context"]
            if inverse and ctx:
                command += [str(ctx[k]) for k in ("window", "min_match", "max_match", "raw_size")]
        elif inverse:
            command += [str(row["count"])]
        else:
            command += [row["kind"], "none" if row["max_point"] is None else str(row["max_point"])]
        return subprocess.run(command, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    failures, tested = [], 0
    for index, row in enumerate(golden["records"]):
        for inverse in (False, True):
            before, after = (row["output"], row["input"]) if inverse else (row["input"], row["output"])
            process = run(row, inverse, (output/"blobs"/before).read_bytes())
            if process.returncode or sha(process.stdout) != after:
                failures.append(dict(index=index, fixture=row["fixture"], action=row["action"], inverse=inverse,
                                     stderr=process.stderr.decode(errors="replace"), expected=after,
                                     actual=sha(process.stdout), actual_bytes=len(process.stdout)))
            tested += 1
        if len(failures) >= 25:
            break
    for row in golden["rejected"]:
        params = dict(row, family="legacy", count=0, context={})
        process = run(params, False, (output/"blobs"/row["input"]).read_bytes())
        if process.returncode == 0:
            failures.append(dict(unexpected_acceptance=row))
    result = dict(tested_stage_directions=tested, failures=failures,
                  rejected_cases_checked=len(golden["rejected"]),
                  source_sha256=golden["source_sha256"],
                  native_cli_sha256=sha(native_cli.read_bytes()), legacy_cli_sha256=sha(legacy_cli.read_bytes()),
                  stage_count=len(golden["records"]), chain_count=len(golden["chains"]))
    (output/"cpp_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


def save_compat(output):
    raise RuntimeError("Historical Python kernels retired; verify the existing frozen compatibility oracle.")
    """Freeze non-action slow/lazy and literal paths before kernel removal."""
    sys.path.insert(0, str(ROOT))
    from backend.myapp import native_lz_family as native
    import primitive_operator_action_runtime as p
    golden = json.loads((output/"golden.json").read_text())
    assert sha((ROOT/"backend/myapp/native_lz_family.py").read_bytes()) == golden["source_sha256"]["backend/myapp/native_lz_family.py"]
    target = output/"compat_golden.json"
    if target.exists():
        raise RuntimeError("compat goldens already exist; do not overwrite frozen evidence")
    records = []
    os.environ["WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES"] = str(1<<60)
    os.environ["WEB_COMPRESSION_NATIVE_LZ_LITERAL_FAST"] = "0"
    for fixture in golden["fixtures"]:
        raw = (output/"blobs"/fixture["input"]).read_bytes()
        for key in sorted(set(p.EXACT_NATIVE_METHOD_KEYS.values())):
            preset = native._preset(key)
            tokens = native._parse_native_lz_family_chain(raw,preset)
            wire = native.pack_native_lz_tokens(tokens,key)
            assert wire == native.compress_native_lz_family(raw,key)
            encoded_tokens = p._serialize_native_tokens(tokens)
            for body in (wire,encoded_tokens):
                path = output/"blobs"/sha(body)
                if not path.exists():
                    path.write_bytes(body)
            records.append(dict(fixture=fixture["name"],input=fixture["input"],output=sha(wire),tokens=sha(encoded_tokens),
                                window=preset.window,minimum=preset.min_match,maximum=preset.max_match,
                                candidates=preset.max_candidates,lazy=int(preset.lazy),algorithm=key))
        literal = native._compress_native_lz_family_literal(raw)
        path = output/"blobs"/sha(literal)
        if not path.exists():
            path.write_bytes(literal)
        records.append(dict(fixture=fixture["name"],input=fixture["input"],output=sha(literal),literal=True))
    target.write_text(json.dumps(dict(records=records),indent=2)+"\n")
    print(json.dumps(dict(saved=str(target),cases=len(records))))


def verify_compat(output,native_cli):
    golden = json.loads((output/"compat_golden.json").read_text())
    tested, failures = 0, []
    for row in golden["records"]:
        raw = (output/"blobs"/row["input"]).read_bytes()
        args = [] if row.get("literal") else [str(row[k]) for k in ("window","minimum","maximum","candidates","lazy")]
        modes = [("literal",row["output"])] if row.get("literal") else [("compress-chain",row["output"]),("parse-chain",row["tokens"])]
        for mode, expected in modes:
            result = subprocess.run([str(native_cli),mode]+args,input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tested += 1
            if result.returncode or sha(result.stdout) != expected:
                failures.append(dict(fixture=row["fixture"],algorithm=row.get("algorithm"),mode=mode,
                                     stderr=result.stderr.decode(errors="replace"),expected=expected,actual=sha(result.stdout)))
        wire = (output/"blobs"/row["output"]).read_bytes()
        result = subprocess.run([str(native_cli),"decompress",str(row.get("minimum",3)),str(len(raw))],input=wire,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        tested += 1
        if result.returncode or result.stdout != raw:
            failures.append(dict(fixture=row["fixture"],mode="decompress",stderr=result.stderr.decode(errors="replace")))
        if len(failures) >= 25:
            break
    report = dict(tested=tested,failures=failures,cases=len(golden["records"]))
    (output/"compat_cpp_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if failures:
        raise SystemExit(1)


def verify_chains(output,native_cli,legacy_cli):
    golden = json.loads((output/"golden.json").read_text())
    fixtures = {f["name"]:f for f in golden["fixtures"]}
    failures, checked, directions = [], 0, 0
    def call(command, raw):
        nonlocal directions
        directions += 1
        result = subprocess.run(command,input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return result.stdout
    for chain in golden["chains"]:
        raw = (output/"blobs"/chain["input"]).read_bytes()
        fixture = fixtures[chain["fixture"]]
        try:
            if chain["family"] == "native":
                params = [str(chain[k]) for k in ("window","minimum","maximum")]
                wire = call([str(native_cli),"compress"]+params,raw)
                recovered = call([str(native_cli),"decompress",str(chain["minimum"]),str(len(raw))],wire)
            else:
                wire = raw
                for action in chain["actions"]:
                    wire = call([str(legacy_cli),"forward",action,fixture["kind"],str(fixture["max_point"])],wire)
                recovered = wire
                for action in reversed(chain["actions"]):
                    recovered = call([str(legacy_cli),"inverse",action,str(fixture["count"])],recovered)
            assert sha(wire) == chain["output"], "final wire differs from frozen Python output"
            assert recovered == raw, "complete reverse differs from original raw bits"
            checked += 1
        except (RuntimeError,AssertionError) as error:
            failures.append(dict(fixture=chain["fixture"],family=chain["family"],algorithm=chain.get("algorithm"),error=str(error)))
    report = dict(complete_chains=checked,cli_calls=directions,failures=failures,
                  native_final_oracle="frozen direct Python fast encoder (also matched original split stages)",
                  legacy_final_oracle="frozen Python split legacy numeric adapter")
    (output/"complete_cpp_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if failures:
        raise SystemExit(1)


def verify_bridge(output):
    """Exercise replacement public APIs against immutable pre-migration bytes."""
    sys.path.insert(0,str(ROOT))
    from backend.myapp import native_lz_family as native, legacy_numeric_stages as legacy
    golden = json.loads((output/"golden.json").read_text())
    compat = json.loads((output/"compat_golden.json").read_text())
    failures, checks = [], 0
    def check(actual,expected,row,label):
        nonlocal checks
        checks += 1
        if sha(actual) != expected:
            failures.append(dict(fixture=row["fixture"],label=label,expected=expected,actual=sha(actual)))
    os.environ["WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES"] = "0"
    os.environ["WEB_COMPRESSION_NATIVE_LZ_LITERAL_FAST"] = "0"
    for row in golden["chains"]:
        if row["family"] != "native":
            continue
        raw = (output/"blobs"/row["input"]).read_bytes()
        wire = native.compress_native_lz_family(raw,row["algorithm"])
        check(wire,row["output"],row,"configured-fast")
        check(native.decompress_native_lz_family(wire,row["algorithm"],len(raw)),row["input"],row,"decode")
        check(native.compress_native_lz_family_staged_fast(raw,row["algorithm"]),row["output"],row,"staged-fast")
        check(native.compress_native_lz_family_staged(raw,row["algorithm"]),row["output"],row,"public-parse-pack-fast")
    for row in golden["records"]:
        if row["family"] != "legacy":
            continue
        raw = (output/"blobs"/row["input"]).read_bytes()
        encoded = legacy.forward(row["action"],raw,row["kind"],row["max_point"])
        check(encoded,row["output"],row,"legacy-forward")
        check(legacy.inverse(row["action"],encoded,count=row["count"]),row["input"],row,"legacy-inverse")
    for row in compat["records"]:
        raw = (output/"blobs"/row["input"]).read_bytes()
        if row.get("literal"):
            os.environ["WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES"] = "0"
            os.environ["WEB_COMPRESSION_NATIVE_LZ_LITERAL_FAST"] = "1"
            check(native.compress_native_lz_family(raw,"zstd_level_3"),row["output"],row,"configured-literal")
            check(native.compress_native_lz_family_staged(raw,"zstd_level_3"),row["output"],row,"public-parse-pack-literal")
        else:
            os.environ["WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES"] = str(1<<60)
            os.environ["WEB_COMPRESSION_NATIVE_LZ_LITERAL_FAST"] = "0"
            check(native.compress_native_lz_family(raw,row["algorithm"]),row["output"],row,"configured-chain")
            check(native.compress_native_lz_family_staged(raw,row["algorithm"]),row["output"],row,"public-parse-pack-chain")
    os.environ["WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES"] = "0"
    os.environ["WEB_COMPRESSION_NATIVE_LZ_LITERAL_FAST"] = "0"
    for bench in (lambda data:native.bench_native_lz_family_bytes(data,"zstd_level_3"),native.bench_native_lz_family_literal_bytes):
        data = [b"",bytes(range(64))*64]
        result = bench(data)
        assert len(result) == 6 and result[0] == 4096 and result[1] > 0 and result[4] == 0 and result[5] == 4096
        checks += 1
    report = dict(public_api_checks=checks,failures=failures,
                  bridge_sha256={name:sha((ROOT/"backend/myapp"/name).read_bytes()) for name in ("native_lz_family.py","legacy_numeric_stages.py","cpp_action_bridge.py")})
    (output/"thin_bridge_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("save", "verify", "save-compat", "verify-compat", "verify-chains", "verify-bridge"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--native-cli", type=Path, default=ROOT/"cpp/build/native_lz_action_stages_cli")
    parser.add_argument("--legacy-cli", type=Path, default=ROOT/"cpp/build/legacy_numeric_action_stages_cli")
    args = parser.parse_args()
    if args.mode == "save":
        save(args.output_dir)
    elif args.mode == "verify":
        verify(args.output_dir, args.native_cli, args.legacy_cli)
    elif args.mode == "save-compat":
        save_compat(args.output_dir)
    elif args.mode == "verify-compat":
        verify_compat(args.output_dir,args.native_cli)
    elif args.mode == "verify-chains":
        verify_chains(args.output_dir,args.native_cli,args.legacy_cli)
    else:
        verify_bridge(args.output_dir)
