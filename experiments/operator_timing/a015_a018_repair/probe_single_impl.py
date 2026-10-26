#!/usr/bin/env python3
"""Run one exact Clean200 column through one repaired timing implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--implementation-id", required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    catalog = json.loads(args.catalog.read_text())
    job = plan["jobs"][args.ordinal]
    rows = catalog.get("resolved_implementations", catalog.get("implementations"))
    selected = [row for row in rows if row.get("implementation_id") == args.implementation_id]
    if len(selected) != 1:
        raise RuntimeError("implementation selection is not unique")
    original_manifest = job["manifests"][0]
    expected_manifest_sha = plan["manifest_sha256"][original_manifest]
    if sha256(Path(original_manifest)) != expected_manifest_sha:
        raise RuntimeError("manifest SHA mismatch")
    request = {
        "schema": "operator102-timing-column-request-v1",
        "request_id": f"repair-v2-probe-{args.ordinal}-{args.implementation_id}",
        "dataset_id": job["dataset_id"],
        "column_id": job["column_id"],
        "manifest_path": original_manifest,
        "expected_manifest_sha256": expected_manifest_sha,
        "input_contract": plan.get("input_contract", {}),
        "implementations": selected,
        "provenance": {
            "plan_sha256": sha256(args.plan),
            "catalog_sha256": sha256(args.catalog),
            "driver_sha256": sha256(args.driver),
            "global_column_ordinal": args.ordinal,
            "repair_probe": "a015_a018_repair_v2",
        },
    }
    process = subprocess.run(
        [str(args.driver), "--jsonl"],
        input=json.dumps(request, separators=(",", ":")) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"driver exit {process.returncode}: {process.stderr[-4000:]}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("driver did not return exactly one response")
    response = json.loads(lines[0])
    results = response.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("probe response does not contain one result")
    result = results[0]
    timing = result.get("timing")
    if result.get("implementation_id") != args.implementation_id:
        raise RuntimeError("probe implementation identity changed")
    if result.get("status") != "ok" or result.get("bitexact") is not True:
        raise RuntimeError(f"repaired implementation is not bit-exact ok: {result}")
    if not isinstance(timing, dict) or (
        timing.get("warmups_per_block"), timing.get("repeats"), timing.get("threads")
    ) != (1, 3, 1):
        raise RuntimeError("probe timing contract is not W1/R3/T1")
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
