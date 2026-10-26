#!/usr/bin/env python3
"""Run one real W1/R3/cold-decode smoke for every frozen active85 route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--o2r1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    routes = json.loads(args.routes.read_text())
    records = routes.get("records")
    if not isinstance(records, list) or len(records) != 85:
        raise ValueError("route manifest is not exact active85")
    outcomes: list[dict] = []
    for index, record in enumerate(records):
        algorithm = record["algorithm"]
        target = output / f"{index:03d}_{algorithm}"
        completed = subprocess.run([
            sys.executable, str(args.runner.resolve()),
            "--routes", str(args.routes.resolve()),
            "--algorithm", algorithm,
            "--o2r1", str(args.o2r1.resolve()),
            "--output", str(target),
        ], text=True, capture_output=True, check=False)
        row = {
            "algorithm": algorithm,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if completed.returncode == 0:
            receipt = json.loads((target / "RESULT.json").read_text())
            row.update({
                "status": receipt.get("status"),
                "bit_exact": receipt.get("bit_exact"),
                "carrier_kind": receipt.get("carrier_kind"),
                "samples": len(receipt.get("measurement", {}).get(
                    "compression_ns_samples", [])),
                "result_sha256": sha256(target / "RESULT.json"),
            })
        else:
            row.update({"status": "failed", "bit_exact": False})
        outcomes.append(row)
    failed = [row["algorithm"] for row in outcomes
              if row["returncode"] != 0 or row.get("status") != "ok" or
              row.get("bit_exact") is not True or row.get("samples") != 3]
    summary = {
        "schema": "operator-algorithm-active85-route-smoke-v2",
        "status": "passed" if not failed else "failed",
        "routes_sha256": sha256(args.routes.resolve()),
        "runner_sha256": sha256(args.runner.resolve()),
        "carrier_source_sha256": sha256(args.runner.resolve().parents[1] /
                                         "source" / "carrier.py"),
        "o2r1_sha256": sha256(args.o2r1.resolve()),
        "algorithms_expected": 85,
        "algorithms_passed": 85 - len(failed),
        "algorithms_failed": failed,
        "all_cold_decode_bit_exact": not failed,
        "all_W1_R3_T1": not failed,
        "outcomes": outcomes,
    }
    receipt = output / "SMOKE.json"
    receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "status", "algorithms_expected", "algorithms_passed",
        "algorithms_failed", "routes_sha256")}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
