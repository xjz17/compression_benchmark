#!/usr/bin/env python3
"""Select a deterministic small, typed pilot cohort from CLEAN200_PLAN.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DTYPES = ("int8", "int16", "int64", "float32", "float64")


def manifest_dtype_and_shape(path: Path) -> tuple[str, int, int]:
    value = json.loads(path.read_text())
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise RuntimeError(f"manifest has no blocks: {path}")
    dtype = value.get("dtype")
    if not isinstance(dtype, str):
        identity = value.get("column_identity", {})
        dtype = identity.get("dtype") if isinstance(identity, dict) else None
    if not isinstance(dtype, str):
        dtype = blocks[0].get("dtype")
    if dtype not in DTYPES:
        raise RuntimeError(f"unsupported dtype {dtype!r}: {path}")
    byte_count = sum(int(block.get("original_numeric_bytes", 0)) for block in blocks)
    entities = len({str(block.get("entity_id", block.get("entity_ordinal", "")))
                    for block in blocks})
    return dtype, byte_count, entities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dtype", type=int, default=1)
    parser.add_argument("--minimum-bytes", type=int, default=256)
    args = parser.parse_args()
    if args.per_dtype < 1:
        raise RuntimeError("per-dtype must be positive")
    plan = json.loads(args.plan.read_text())
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 3723:
        raise RuntimeError("exact CLEAN200 3,723-column plan required")
    typed: dict[str, list[tuple[int, int, str, dict]]] = {name: [] for name in DTYPES}
    for job in jobs:
        manifests = job.get("manifests")
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise RuntimeError(f"one manifest required: {job.get('job_id')}")
        dtype, byte_count, entities = manifest_dtype_and_shape(Path(manifests[0]))
        if byte_count >= args.minimum_bytes:
            typed[dtype].append((byte_count, -entities, str(job["job_id"]), job))
    selected: list[dict] = []
    for dtype in DTYPES:
        candidates = sorted(typed[dtype], key=lambda row: row[:3])
        if len(candidates) < args.per_dtype:
            raise RuntimeError(f"not enough {dtype} pilot candidates")
        selected.extend(row[3] for row in candidates[:args.per_dtype])
    selected.sort(key=lambda row: (row["dataset_id"], row["column_id"], row["job_id"]))
    manifest_paths = {path for job in selected for path in job["manifests"]}
    output = {
        **{key: value for key, value in plan.items()
           if key not in {"jobs", "manifest_sha256", "counts", "jobs_canonical_sha256"}},
        "schema": "clean200-pilot-column-plan-v1",
        "status": "frozen_verified_subset",
        "parent_plan": str(args.plan.resolve()),
        "parent_jobs_canonical_sha256": plan.get("jobs_canonical_sha256"),
        "selection": {
            "per_dtype": args.per_dtype,
            "minimum_bytes": args.minimum_bytes,
            "order": "smallest eligible original_numeric_bytes, then more entities, then job_id",
            "dtypes": list(DTYPES),
        },
        "counts": {
            "datasets": len({job["dataset_id"] for job in selected}),
            "columns": len(selected),
            "active_algorithms": 85,
            "active_operators": 97,
            "expected_operator_timing_keys": len(selected) * 97,
            "expected_cartesian_keys": len(selected) * 97 * 85,
        },
        "manifest_sha256": {
            path: digest for path, digest in plan["manifest_sha256"].items()
            if path in manifest_paths
        },
        "jobs": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True,
                                      ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "counts": output["counts"],
        "selected": [
            {"dataset_id": job["dataset_id"], "column_id": job["column_id"],
             "job_id": job["job_id"]} for job in selected
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

