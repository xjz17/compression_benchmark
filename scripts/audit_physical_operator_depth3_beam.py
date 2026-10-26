#!/usr/bin/env python3
"""Independent integrity and coverage audit for the TypedBeam depth-3 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from physical_operator_runtime_v2 import TRANSFORMS


FAMILY = {spec.name: spec.family for spec in TRANSFORMS}
WIDTHS = (29, 64, 256, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_csv(path: Path, expected_rows: int) -> int:
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if row["status"] != "ok" or int(row["lossless"]) != 1:
                raise AssertionError(f"non-lossless row in {path}: {rows}")
            pipeline = tuple(part.strip() for part in row["pipeline"].split(" -> ") if part.strip())
            transforms = pipeline[:-2]
            families = [FAMILY[name] for name in transforms]
            if len(transforms) > 3 or len(families) != len(set(families)):
                raise AssertionError(f"illegal transform grammar in {path}: {pipeline}")
    if rows != expected_rows:
        raise AssertionError(f"row count mismatch {path}: {rows} != {expected_rows}")
    return rows


def verify_output_hashes(directory: Path, manifest: dict[str, Any]) -> None:
    for name, expected in manifest["output_sha256"].items():
        actual = sha256_file(directory / name)
        if actual != expected:
            raise AssertionError(f"manifest output hash mismatch: {directory / name}")


def parse_remote_hashes(run_root: Path) -> int:
    count = 0
    for line in (run_root / "REMOTE_SHA256.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        actual = sha256_file(run_root / relative)
        if actual != expected:
            raise AssertionError(f"remote/local hash mismatch: {relative}")
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(args.existing_root.resolve().glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        existing[manifest["dataset"], int(manifest["column_index"])] = manifest
    if len(existing) != 20:
        raise AssertionError(f"expected 20 shallow references, got {len(existing)}")
    round_manifest = json.loads((run_root / "ROUND_MANIFEST.json").read_text(encoding="utf-8"))
    remote_hash_count = parse_remote_hashes(run_root)
    beam_manifests = sorted((run_root / "results" / "beam").glob("*/manifest.json"))
    exact_manifests = sorted((run_root / "results" / "exact").glob("*/manifest.json"))
    source_tuples: set[tuple[str, str, str, str]] = set()
    measured_rows = 0
    complete_keys: set[tuple[str, int]] = set()
    for manifest_path in beam_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        directory = manifest_path.parent
        verify_output_hashes(directory, manifest)
        key = manifest["dataset"], int(manifest["column_index"])
        if manifest["status"] != "complete" or manifest["mode"] != "beam":
            raise AssertionError(f"bad Beam completion state: {manifest_path}")
        if manifest["widths"] != list(WIDTHS) or key in complete_keys:
            raise AssertionError(f"bad/duplicate Beam widths: {manifest_path}")
        if manifest["input_sha256"] != existing[key]["input_sha256"]:
            raise AssertionError(f"input hash mismatch: {key}")
        for result in manifest["result"]:
            if int(result["failed_candidates"]) != 0 or not result["all_candidates_accounted"]:
                raise AssertionError(f"failed/unaccounted Beam result: {key}")
            measured_rows += verify_csv(
                directory / result["measurement_file"], int(result["written_measurements"])
            )
        source = manifest["source_sha256"]
        source_tuples.add(
            (source["runner"], source["runtime"], source["loader"], source["backend_myapp_python_tree"])
        )
        complete_keys.add(key)
    if len(source_tuples) != 1:
        raise AssertionError(f"mixed source hashes: {source_tuples}")
    runner_hash, runtime_hash, loader_hash, backend_hash = next(iter(source_tuples))
    round_sources = round_manifest["source_sha256"]
    if runner_hash != round_sources["search_physical_operator_depth3_beam.py"]:
        raise AssertionError("runner hash differs from round manifest")
    if runtime_hash != round_sources["physical_operator_runtime_v2.py"]:
        raise AssertionError("runtime hash differs from round manifest")
    if loader_hash != round_sources["enumerate_physical_operator_variants_v2.py"]:
        raise AssertionError("loader hash differs from round manifest")
    if backend_hash != round_manifest["backend_myapp_python_tree_sha256"]:
        raise AssertionError("backend tree differs from round manifest")
    exact_rows = 0
    for manifest_path in exact_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        directory = manifest_path.parent
        verify_output_hashes(directory, manifest)
        result = manifest["result"]
        if (
            manifest["status"] != "complete"
            or manifest["mode"] != "exact"
            or int(result["expected_candidates"]) != 143650
            or int(result["written_measurements"]) != 143650
            or int(result["failed_candidates"]) != 0
        ):
            raise AssertionError(f"bad exact completion: {manifest_path}")
        exact_rows += verify_csv(directory / result["measurement_file"], 143650)
        source = manifest["source_sha256"]
        if (source["runner"], source["runtime"], source["loader"], source["backend_myapp_python_tree"]) not in source_tuples:
            raise AssertionError(f"exact source differs from Beam source: {manifest_path}")
    expected_keys = set(existing)
    missing = sorted(expected_keys - complete_keys)
    all_beam_dirs = {
        path.name for path in (run_root / "results" / "beam").iterdir() if path.is_dir()
    }
    incomplete_dirs = sorted(
        name for name in all_beam_dirs if not (run_root / "results" / "beam" / name / "manifest.json").is_file()
    )
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_verdict": "PASS",
        "coverage_verdict": "FAIL_BUDGET_CENSORED" if missing else "PASS",
        "complete_beam_columns": len(complete_keys),
        "expected_beam_columns": len(expected_keys),
        "exact_anchors": len(exact_manifests),
        "missing_columns": [f"{dataset}:{index}" for dataset, index in missing],
        "incomplete_directories": incomplete_dirs,
        "remote_local_hashes_verified": remote_hash_count,
        "beam_measurement_rows_verified": measured_rows,
        "exact_measurement_rows_verified": exact_rows,
        "failed_measurement_rows": 0,
        "source_sha256": {
            "runner": runner_hash,
            "runtime": runtime_hash,
            "loader": loader_hash,
            "backend_myapp_python_tree": backend_hash,
        },
        "claim_boundary": "integrity passes for completed outputs; coverage failure makes teacher and GRPO gates inconclusive",
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f"""# TypedBeam depth-3 independent audit

- Integrity: **{audit['integrity_verdict']}**
- Coverage: **{audit['coverage_verdict']}** ({len(complete_keys)}/{len(expected_keys)} columns)
- Exact anchors: {len(exact_manifests)}/2
- Remote/local hashes: {remote_hash_count} verified
- Lossless candidate rows: {measured_rows + exact_rows:,} verified; failures: 0
- Missing: {', '.join(audit['missing_columns'])}

Completed artifacts are internally usable, but the four missing columns make
the all-column teacher gate and any decision to train robust GRPO inconclusive.
"""
    (output_dir / "AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run(parse_args())
