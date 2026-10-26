#!/usr/bin/env python3
"""Freeze the exact 200-dataset / 3,723-column input union.

This adapter does not rewrite numeric payloads.  It normalizes the three
already accepted plan schemas to one manifest whose jobs point at the original
immutable column manifests and bind their SHA-256 digests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "clean200-column-plan-v1"
EXPECTED_COUNTS = {
    "old135": (135, 3519),
    "new15": (15, 125),
    "additional50": (50, 79),
}
REQUIRED_CONTRACT = {
    "byte_order": "little",
    "values_are_valid_only": True,
    "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalize_contract(value: Any, source: str) -> dict[str, Any]:
    if isinstance(value, str):
        require(
            value == (
                "current135 value-column manifests only; original dtype, numeric "
                "payload and missing bitmap; metadata and timestamp columns excluded"
            ),
                f"unrecognized prose input contract for {source}")
        return {**REQUIRED_CONTRACT, "source_description": value}
    require(isinstance(value, dict), f"input contract object required for {source}")
    for key, expected in REQUIRED_CONTRACT.items():
        require(value.get(key) == expected,
                f"{source} input contract mismatch for {key}")
    result = dict(REQUIRED_CONTRACT)
    if isinstance(value.get("description"), str):
        result["source_description"] = value["description"]
    return result


def manifest_paths(job: dict[str, Any]) -> list[Path]:
    paths = job.get("manifests")
    if paths is None:
        one = job.get("manifest_path")
        paths = [one] if one is not None else None
    require(isinstance(paths, list) and paths and
            all(isinstance(value, str) and value for value in paths),
            f"job lacks manifest path(s): {job.get('job_id')}")
    return [Path(value).resolve() for value in paths]


def expected_manifest_sha(plan: dict[str, Any], job: dict[str, Any], path: Path) -> str:
    if isinstance(job.get("manifest_sha256"), str):
        require(len(manifest_paths(job)) == 1,
                "job-level manifest SHA can only bind one manifest")
        return job["manifest_sha256"]
    mapping = plan.get("manifest_sha256")
    require(isinstance(mapping, dict) and isinstance(mapping.get(str(path)), str),
            f"no frozen manifest SHA for {path}")
    return mapping[str(path)]


def load_scope(scope: str, plan_path: Path, verify_files: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = plan_path.resolve()
    plan = read_object(plan_path)
    jobs = plan.get("jobs")
    require(isinstance(jobs, list), f"jobs list required: {plan_path}")
    expected_datasets, expected_columns = EXPECTED_COUNTS[scope]
    require(len(jobs) == expected_columns,
            f"{scope}: expected {expected_columns} columns, got {len(jobs)}")
    dataset_ids = {job.get("dataset_id") for job in jobs if isinstance(job, dict)}
    require(None not in dataset_ids and len(dataset_ids) == expected_datasets,
            f"{scope}: expected {expected_datasets} dataset identities")
    contract = normalize_contract(plan.get("input_contract"), scope)

    normalized: list[dict[str, Any]] = []
    for source_ordinal, job in enumerate(jobs):
        require(isinstance(job, dict), f"{scope}: job must be an object")
        dataset_id = job.get("dataset_id")
        column_id = job.get("column_id")
        require(isinstance(dataset_id, str) and dataset_id and
                isinstance(column_id, str) and column_id,
                f"{scope}: invalid dataset/column identity")
        manifests = manifest_paths(job)
        bindings: list[dict[str, Any]] = []
        for manifest in manifests:
            frozen_sha = expected_manifest_sha(plan, job, manifest)
            require(len(frozen_sha) == 64,
                    f"invalid frozen manifest SHA for {manifest}")
            if verify_files:
                require(manifest.is_file(), f"missing manifest: {manifest}")
                actual = sha256(manifest)
                require(actual == frozen_sha,
                        f"manifest SHA mismatch: {manifest}; {actual} != {frozen_sha}")
            bindings.append({"path": str(manifest), "sha256": frozen_sha})
        identity_payload = {
            "dataset_id": dataset_id,
            "column_id": column_id,
            "manifests": bindings,
        }
        clean200_key = canonical_sha(identity_payload)
        normalized.append({
            "clean200_column_key": clean200_key,
            "job_id": f"clean200_{clean200_key[:24]}",
            "source_scope": scope,
            "source_ordinal": source_ordinal,
            "source_job_id": job.get("job_id"),
            "dataset_id": dataset_id,
            "column_id": column_id,
            "column_name": job.get("column_name"),
            "dtype_hint": job.get("dtype", job.get("canonical_dtype")),
            "valid_count_hint": job.get("valid_count"),
            "original_numeric_bytes": job.get(
                "scheduling_original_numeric_bytes",
                job.get("canonical_input_bytes", job.get("source_native_numeric_bytes")),
            ),
            "manifest_bindings": bindings,
            "manifests": [binding["path"] for binding in bindings],
            "manifest_bundle_sha256": canonical_sha(bindings),
        })
    source = {
        "scope": scope,
        "plan_path": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "plan_schema": plan.get("schema"),
        "datasets": expected_datasets,
        "columns": expected_columns,
        "input_contract": contract,
    }
    return source, normalized


def build(args: argparse.Namespace) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for scope, path in (
        ("old135", Path(args.old135_plan)),
        ("new15", Path(args.new15_plan)),
        ("additional50", Path(args.additional50_plan)),
    ):
        source, rows = load_scope(scope, path, not args.skip_file_verification)
        sources.append(source)
        jobs.extend(rows)

    require(len(jobs) == 3723, f"expected 3723 columns, got {len(jobs)}")
    dataset_ids = {row["dataset_id"] for row in jobs}
    require(len(dataset_ids) == 200, f"expected 200 datasets, got {len(dataset_ids)}")
    keys = [row["clean200_column_key"] for row in jobs]
    require(len(keys) == len(set(keys)), "duplicate clean200 column identity")
    dataset_overlap: dict[str, list[str]] = {}
    for source in sources:
        scope = source["scope"]
        for row in (value for value in jobs if value["source_scope"] == scope):
            dataset_overlap.setdefault(row["dataset_id"], []).append(scope)
    duplicated_datasets = {
        dataset_id: sorted(set(scopes))
        for dataset_id, scopes in dataset_overlap.items()
        if len(set(scopes)) > 1
    }
    require(not duplicated_datasets,
            f"dataset identity overlaps across scopes: {duplicated_datasets}")
    jobs.sort(key=lambda row: (row["dataset_id"], row["column_id"], row["clean200_column_key"]))
    plan = {
        "schema": SCHEMA,
        "status": "frozen_verified" if not args.skip_file_verification else "frozen_unverified_files",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "counts": {
            "datasets": 200,
            "columns": 3723,
            "active_algorithms": 85,
            "active_operators": 97,
            "expected_operator_timing_keys": 3723 * 97,
            "expected_cartesian_keys": 3723 * 97 * 85,
        },
        "input_contract": {
            **REQUIRED_CONTRACT,
            "entity_boundaries_preserved": True,
            "numeric_payloads_rewritten": False,
            "metadata_and_timestamps_excluded": True,
            "source_plan_contracts": {
                source["scope"]: source["input_contract"] for source in sources
            },
        },
        "sources": sources,
        "manifest_sha256": {
            binding["path"]: binding["sha256"]
            for row in jobs for binding in row["manifest_bindings"]
        },
        "jobs": jobs,
    }
    plan["jobs_canonical_sha256"] = canonical_sha(jobs)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old135-plan", required=True)
    parser.add_argument("--new15-plan", required=True)
    parser.add_argument("--additional50-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args()
    result = build(args)
    atomic_json(Path(args.output).resolve(), result)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "status": result["status"],
        "counts": result["counts"],
        "jobs_canonical_sha256": result["jobs_canonical_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

