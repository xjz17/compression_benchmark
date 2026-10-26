#!/usr/bin/env python3
"""Build a relocation-aware post-hoc hash supplement for a PV2F result tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--remote-root",
        default="/dev/shm/physical_operator_variants_20260826_v1",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    output = args.output.resolve()
    files = sorted(
        path
        for scope in (
            "measurements",
            "search",
            "audit",
            "analysis",
            "code",
            "logs",
            "timings_20260826_v1",
            "timings_20260827_v2",
            "timings_20260827_v3",
            "sparse19_20260828_v3",
            "sparse19_latency_diagnosis_20260828_v1",
            "sparse19_prefix_batch_20260828_v1",
            "sparse19_prefix_batch_20260828_v2",
            "sparse19_prefix_batch_20260828_v3",
            "sparse19_prefix_batch_20260828_v4",
        )
        for path in (root / scope).rglob("*")
        if path.is_file() and path.resolve() != output
    )
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in files}
    policy_paths = sorted((root / "search/policies").glob("*.npz"))
    policy_arrays = 0
    all_policy_values_finite = True
    for path in policy_paths:
        with np.load(path) as archive:
            policy_arrays += len(archive.files)
            all_policy_values_finite = all_policy_values_finite and all(
                bool(np.all(np.isfinite(archive[name]))) for name in archive.files
            )
    manifest = json.loads((root / "search/manifest.json").read_text(encoding="utf-8"))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "run_root_local": str(root),
        "run_root_remote": args.remote_root,
        "relocation": {
            "historical_measurement_inputs": manifest["measurement_inputs"],
            "local_measurement_root": str(root / "measurements"),
            "mapping_rule": "match each historical final directory basename under the local measurement root",
        },
        "coverage": {
            "hashed_files": len(hashes),
            "policy_files": len(policy_paths),
            "policy_arrays": policy_arrays,
            "all_policy_values_finite": all_policy_values_finite,
            "training_curve_rows": csv_rows(root / "search/training_curve.csv"),
            "beam_level_rows": csv_rows(root / "search/beam29_levels.csv"),
            "column_feature_rows": csv_rows(root / "search/column_features.csv"),
        },
        "search_manifest_hash_supplement": {
            "reason": "The immutable search manifest did not hash policies, training_curve.csv, beam29_levels.csv, algorithm_summary.csv, or column_features.csv.",
            "files": {
                name: digest
                for name, digest in hashes.items()
                if name.startswith("search/")
            },
        },
        "all_artifact_sha256": hashes,
        "claim_boundary": "post-hoc integrity and relocation supplement; it does not retroactively change the immutable search manifest or expand decoder replay coverage",
    }
    if len(policy_paths) != 12 or policy_arrays != 48 or not all_policy_values_finite:
        result["status"] = "FAIL"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["coverage"]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    run(parse_args())
