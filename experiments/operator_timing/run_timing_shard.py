#!/usr/bin/env python3
"""Run one deterministic shard of the independent operator timing campaign.

The native child is persistent for the full shard.  One request is one logical
column; implementation order rotates by the column's global ordinal so an
implementation does not systematically receive the same thermal/cache slot.
Results are append-only JSONL and can be resumed without changing old output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REQUEST_SCHEMA = "operator102-timing-column-request-v1"
RESPONSE_SCHEMA = "operator102-timing-column-response-v1"
EXCLUDED_SLOW = {"A106", "A107", "A108", "A109", "A110"}
CAMEL_INT64_BOUNDARY_REASON = (
    "unsupported_input_domain: Camel int64-to-float64 exactness precondition is false"
)
CAMEL_INT64_BOUNDARY_REJECTION = {
    "class": "native_timing_error",
    "message": "Camel integer is not exactly representable as float64",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def catalog_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows = catalog.get("resolved_implementations", catalog.get("implementations"))
    require(isinstance(rows, list), "catalog implementations array is absent")
    active = [row for row in rows if row.get("implementation_id") not in EXCLUDED_SLOW]
    ids = [row.get("implementation_id") for row in active]
    require(len(active) == 97 and len(set(ids)) == 97 and all(isinstance(x, str) for x in ids),
            "catalog does not resolve the exact active97 set")
    require(not (set(ids) & EXCLUDED_SLOW), "paused slow implementation entered active97")
    return active


def rewrite_path(value: str, old: str | None, new: str | None) -> str:
    if old is None and new is None:
        return value
    require(bool(old) and new is not None, "both path-prefix-from/to are required")
    require(value.startswith(old), f"manifest path is outside remap prefix: {value}")
    return new + value[len(old):]


def completed_ids(path: Path) -> set[str]:
    complete: set[str] = set()
    if not path.exists():
        return complete
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(row.get("schema") == RESPONSE_SCHEMA,
                    f"non-timing row in existing output line {number}")
            request_id = row.get("request_id")
            require(isinstance(request_id, str) and request_id not in complete,
                    f"duplicate/invalid request in existing output line {number}")
            validate_response(row, None)
            complete.add(request_id)
    return complete


def validate_not_applicable_result(row: dict[str, Any]) -> None:
    """Accept only the frozen A004 int64-to-float64 support boundary.

    Active97 has no generic mapper-declared N/A entries.  In particular, the
    implementation ID alone is insufficient: the native exception class and
    message, non-rankable flag, and public reason must all match exactly.
    """
    require(row.get("implementation_id") == "A004",
            "only A004 may report not_applicable in active97 timing")
    require(row.get("bitexact") is False and row.get("rankable") is False,
            "A004 support boundary must be explicitly non-bitexact/non-rankable")
    require(row.get("reason") == CAMEL_INT64_BOUNDARY_REASON,
            "A004 support-boundary reason drift")
    require(row.get("native_domain_rejection") == CAMEL_INT64_BOUNDARY_REJECTION,
            "A004 native support-boundary evidence drift")


def validate_response(response: dict[str, Any], expected_ids: list[str] | None) -> None:
    require(response.get("schema") == RESPONSE_SCHEMA, "wrong native response schema")
    require(response.get("status") != "request_failed", f"native request failed: {response}")
    rows = response.get("results")
    require(isinstance(rows, list) and len(rows) == 97, "response is not exact active97")
    ids = [row.get("implementation_id") for row in rows]
    require(len(set(ids)) == 97, "response contains duplicate implementation IDs")
    if expected_ids is not None:
        require(ids == expected_ids, "native response changed rotated implementation order")
    for row in rows:
        status = row.get("status")
        require(status in {"ok", "not_applicable"},
                f"nonterminal operator result: {row.get('implementation_id')} {status}")
        if status == "not_applicable":
            validate_not_applicable_result(row)
            continue
        require(row.get("bitexact") is True, "timed result is not bit-exact")
        timing = row.get("timing")
        require(isinstance(timing, dict) and timing.get("warmups_per_block") == 1 and
                timing.get("repeats") == 3 and timing.get("threads") == 1 and
                timing.get("ranking_view") == "full_o2r1",
                "timing contract is not W1/R3/T1")
        for key in ("kernel_encode_ns", "kernel_decode_ns",
                    "full_encode_ns", "full_decode_ns"):
            values = timing.get(key)
            require(isinstance(values, list) and len(values) == 3 and
                    all(isinstance(x, int) and x >= 0 for x in values),
                    f"invalid {key} samples")
        require(timing.get("bitexact_each_warmup_and_repeat") is True,
                "per-repeat bit-exact evidence is absent")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--path-prefix-from")
    parser.add_argument("--path-prefix-to")
    args = parser.parse_args()
    require(args.shard_count > 0 and 0 <= args.shard_index < args.shard_count,
            "invalid shard index/count")
    plan_path, catalog_path, driver_path = (args.plan.resolve(), args.catalog.resolve(),
                                            args.driver.resolve())
    require(plan_path.is_file() and catalog_path.is_file() and driver_path.is_file() and
            os.access(driver_path, os.X_OK), "plan/catalog/executable driver is absent")
    plan, catalog = read_object(plan_path), read_object(catalog_path)
    jobs = plan.get("jobs")
    require(isinstance(jobs, list) and jobs, "plan jobs are absent")
    manifest_shas = plan.get("manifest_sha256", {})
    require(isinstance(manifest_shas, dict), "manifest SHA map is absent")
    implementations = catalog_rows(catalog)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output)

    child = subprocess.Popen([str(driver_path), "--jsonl"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             bufsize=1)
    assert child.stdin is not None and child.stdout is not None
    try:
        with output.open("a", buffering=1) as sink:
            for ordinal, job in enumerate(jobs):
                if ordinal % args.shard_count != args.shard_index:
                    continue
                request_id = str(job.get("job_id", ""))
                require(request_id, f"plan job {ordinal} has no job_id")
                if request_id in done:
                    continue
                manifests = job.get("manifests")
                require(isinstance(manifests, list) and len(manifests) == 1,
                        f"one manifest required: {request_id}")
                original_manifest = str(manifests[0])
                manifest = rewrite_path(original_manifest, args.path_prefix_from,
                                        args.path_prefix_to)
                expected_sha = manifest_shas.get(original_manifest)
                require(isinstance(expected_sha, str) and len(expected_sha) == 64,
                        f"manifest SHA absent: {request_id}")
                require(sha256(Path(manifest)) == expected_sha,
                        f"manifest SHA mismatch: {request_id}")
                offset = ordinal % len(implementations)
                rotated = implementations[offset:] + implementations[:offset]
                expected_ids = [row["implementation_id"] for row in rotated]
                request = {
                    "schema": REQUEST_SCHEMA,
                    "request_id": request_id,
                    "dataset_id": job["dataset_id"],
                    "column_id": job["column_id"],
                    "manifest_path": manifest,
                    "expected_manifest_sha256": expected_sha,
                    "input_contract": plan.get("input_contract", {}),
                    "implementations": rotated,
                    "provenance": {
                        "plan_sha256": sha256(plan_path),
                        "catalog_sha256": sha256(catalog_path),
                        "driver_sha256": sha256(driver_path),
                        "global_column_ordinal": ordinal,
                        "implementation_rotation_offset": offset,
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                    },
                }
                child.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                child.stdin.flush()
                line = child.stdout.readline()
                require(line != "", f"native driver exited during {request_id}")
                response = json.loads(line)
                validate_response(response, expected_ids)
                require(response.get("request_id") == request_id and
                        response.get("dataset_id") == job["dataset_id"] and
                        response.get("column_id") == job["column_id"],
                        "native response identity mismatch")
                sink.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
    finally:
        if child.stdin:
            child.stdin.close()
        return_code = child.wait(timeout=30)
        stderr = child.stderr.read() if child.stderr else ""
        require(return_code == 0, f"native driver exited {return_code}: {stderr[-4000:]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
