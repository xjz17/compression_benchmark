#!/usr/bin/env python3
"""Freeze Python stage wires, then verify native CLIs without importing originals."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import random
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
DEFAULT = ROOT / "analysis_outputs/python_stage_cpp_port_20260905_v1"

def sha(raw): return hashlib.sha256(raw).hexdigest()
def b64(raw): return base64.b64encode(raw).decode()

def freeze(output, stress=False):
    raise RuntimeError("Historical Python kernels retired; use immutable pre-migration goldens with verify.")
    from audit_86_composability_realdata import load_external_columns
    from myapp import corad_lossless_stages as corad
    from myapp import reger_legacy_stages as reger
    from myapp import blosc2_legacy_stages as blosc
    import blosc2
    cases = []
    for rows in (() if stress else (128, 2048)):
        columns, _ = load_external_columns(rows)
        for column in columns:
            kind = "int64" if column.logical_type == "integer" else "float64"
            raw = struct.pack("<" + ("q" if kind == "int64" else "d") * len(column.values), *column.values)
            cases.append((f"{column.dataset}:{column.name}:{rows}", kind, raw))
    for name, kind, values in [
        ("empty_int", "int64", []), ("one_int", "int64", [7]),
        ("int_bounds", "int64", [-(1 << 63), (1 << 63)-1, 0, -1, 1, (1 << 53)+1]),
        ("int_constant", "int64", [17]*513), ("int_ramp", "int64", list(range(-256, 257))),
        ("float_special", "float64", [0.0, -0.0, float("inf"), -float("inf"), 5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308]),
        ("float_cancel", "float64", [1e100, 1.0, -1e100, 1e-100, -1e-100, 0.1, -0.3]*9),
    ]:
        cases.append((name, kind, struct.pack("<" + ("q" if kind == "int64" else "d")*len(values), *values)))
    if stress:
        rng = random.Random(20260905)
        for count in (1, 17, 39, 40, 41, 80, 81, 512, 513, 514, 1027):
            for kind in ("int64", "float64"):
                if kind == "int64":
                    values = [rng.randrange(-(1<<63), 1<<63) for _ in range(count)]
                    raw = struct.pack("<"+"q"*count, *values)
                else:
                    words = [rng.getrandbits(64) for _ in range(count)]
                    # Experiment input excludes NaN, but retains infinity and exact lane bits.
                    words = [w for w in words if w & 0x7ff0000000000000 != 0x7ff0000000000000 or w & 0xfffffffffffff == 0]
                    raw = struct.pack("<"+"Q"*len(words), *words)
                cases.append((f"seeded_{kind}_{count}",kind,raw))
        for length in (1, 31, 32, 33, 131072, 131073, 1048576):
            for label, raw in (("zero",bytes(length)), ("random",rng.randbytes(length)),
                               ("pattern",(bytes(range(256))*((length+255)//256))[:length])):
                cases.append((f"raw_{label}_{length}", "bytes", raw))
    records, errors = [], []
    modules = {"corad": [("normalize", lambda raw, kind: corad.normalize_forward(raw, kind), corad.normalize_inverse),
                          ("tricklet", lambda raw, kind: corad.tricklet_forward(raw), corad.tricklet_inverse),
                          ("entropy", lambda raw, kind: corad.entropy_forward(raw), corad.entropy_inverse),
                          ("frame", lambda raw, kind: corad.frame_forward(raw), corad.frame_inverse)],
               "reger": [("core", lambda raw, kind: reger.block_core_forward(raw), reger.block_core_inverse),
                          ("frame", lambda raw, kind: reger.frame_forward(raw), reger.frame_inverse)],
               "blosc2": [("core", lambda raw, kind: blosc.binding_core_forward(raw), blosc.binding_core_inverse),
                           ("frame", lambda raw, kind: blosc.frame_forward(raw), blosc.frame_inverse)]}
    for label, kind, raw in cases:
        for module, stages in modules.items():
            if kind == "bytes" and module != "blosc2": continue
            value = raw
            for stage, forward, inverse in stages:
                try:
                    encoded = forward(value, kind)
                    if inverse(encoded) != value:
                        raise AssertionError("Python stage inverse mismatch")
                    records.append(dict(case=label, kind=kind, module=module, stage=stage,
                                        input_b64=b64(value), output_b64=b64(encoded),
                                        input_sha256=sha(value), output_sha256=sha(encoded)))
                    value = encoded
                except Exception as exc:
                    errors.append(dict(case=label, module=module, stage=stage, error=f"{type(exc).__name__}: {exc}"))
                    break
    paths = [ROOT / "backend/myapp" / f"{name}.py" for name in
             ("corad_lossless_stages", "reger_legacy_stages", "reger_codec", "blosc2_legacy_stages")]
    result = dict(cases=len(cases), records=records, errors=errors, python=sys.executable,
                  versions=dict(python_blosc2=blosc2.__version__, c_blosc2=blosc2.blosclib_version),
                  environment={k:v for k,v in os.environ.items() if k.startswith("COMPRESSION_BENCHMARK_REGER_")},
                  source_hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths})
    output.mkdir(parents=True, exist_ok=False)
    (output / "golden.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))

def verify(output, binaries):
    frozen = json.loads((output / "golden.json").read_text())
    results = []
    for record in frozen["records"]:
        if record["module"] not in binaries: continue
        raw = base64.b64decode(record["input_b64"])
        wire = base64.b64decode(record["output_b64"])
        row = {k:record[k] for k in ("case", "kind", "module", "stage")}
        for direction, source, expected in (("forward", raw, wire), ("inverse", wire, raw)):
            proc = subprocess.run([binaries[record["module"]], record["stage"]+"-"+direction, record["kind"]], input=source, capture_output=True)
            row[direction+"_exact"] = proc.returncode == 0 and proc.stdout == expected
            row[direction+"_bytes"] = len(proc.stdout)
            row[direction+"_expected_bytes"] = len(expected)
            row[direction+"_error"] = proc.stderr.decode(errors="replace")
            if not row[direction+"_exact"]:
                row[direction+"_first_difference"] = next((i for i,(a,b) in enumerate(zip(proc.stdout,expected)) if a != b), min(len(proc.stdout),len(expected)))
        results.append(row)
    summary = dict(records=len(results), exact=sum(r["forward_exact"] and r["inverse_exact"] for r in results), results=results,
                   binary_hashes={k:sha(Path(v).read_bytes()) for k,v in binaries.items()}, golden_sha256=sha((output/"golden.json").read_bytes()))
    name="verification_"+"_".join(sorted(binaries))+".json"
    (output / name).write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "results"}, indent=2))
    for row in results:
        if not row["forward_exact"] or not row["inverse_exact"]: print(json.dumps(row))
    if summary["exact"] != summary["records"]: raise SystemExit(1)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "stress", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--binary", action="append", default=[], help="MODULE=/absolute/CLI")
    args=parser.parse_args()
    if args.mode in ("freeze", "stress"): freeze(args.output, args.mode == "stress")
    else: verify(args.output, dict(value.split("=", 1) for value in args.binary))
