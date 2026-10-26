#!/usr/bin/env python3
"""Add verified PongoDouble + general-codec pairs to completed column results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from benchmark_86_combinations import GENERAL_METHODS, best_generic, load_columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--column", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--bridge-dir", default="/dev/shm/pongo_bridge_20260826", type=Path)
    parser.add_argument("--jar", default="backend/lib/benchmark-java-1.0-SNAPSHOT.jar", type=Path)
    args = parser.parse_args()
    column = next(c for c in load_columns(args.dataset) if c.index == args.column)
    if column.logical_type != "float":
        raise SystemExit("Pongo augmentation is only defined for float columns")
    stem = f"{args.dataset}_{args.column}"
    raw_path = args.bridge_dir / f"{stem}.f64le"
    payload_path = args.bridge_dir / f"{stem}.pongo"
    raw_path.write_bytes(column.raw)
    completed = subprocess.run(
        ["java", "-cp", f"{args.bridge_dir}:{args.jar.resolve()}", "PongoPayloadMain", str(raw_path), str(payload_path)],
        check=True, capture_output=True, text=True,
    )
    payload = payload_path.read_bytes()
    pairs = []
    for general in GENERAL_METHODS:
        best, attempts = best_generic(payload, general)
        if best is None:
            pairs.append({"specialist": "Pongo", "general": general, "ok": False, "attempts": attempts})
        else:
            pairs.append({
                "specialist": "Pongo", "specialist_bytes": len(payload),
                "specialist_implementation": "upstream_project_PongoDouble_java",
                "general": general, "level": best["key"],
                "general_implementation": best["implementation"], "compressed_bytes": best["bytes"],
                "rate": best["bytes"] / len(column.raw), "factor": len(column.raw) / best["bytes"],
                "ok": True, "levels_tested": len(attempts),
                "levels_verified": sum(bool(x["ok"]) for x in attempts),
            })
    result = json.loads(args.result.read_text(encoding="utf-8"))
    result["specialist_exclusions"] = [x for x in result["specialist_exclusions"] if x["method"] != "Pongo"]
    result["specialists_materialized"] = [x for x in result["specialists_materialized"] if x["method"] != "Pongo"]
    result["specialists_materialized"].append({
        "method": "Pongo", "bytes": len(payload), "implementation": "upstream_project_PongoDouble_java",
        "note": "Java encoder and decoder verified raw Double.doubleToRawLongBits for every row",
    })
    result["pairs"] = [x for x in result["pairs"] if x.get("specialist") != "Pongo"] + pairs
    valid = [x for x in result["pairs"] if x.get("ok")]
    valid.sort(key=lambda x: (x["compressed_bytes"], x["specialist"], x["general"], x["level"]))
    result["best"], result["top5"] = valid[0], valid[:5]
    result["pongo_augmentation"] = {
        "java_stdout": completed.stdout.strip(), "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "verified_pairs": sum(bool(x.get("ok")) for x in pairs),
    }
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "column": args.column, "pongo_bytes": len(payload), "best": result["best"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
