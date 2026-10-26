#!/usr/bin/env python3
"""Combine disjoint 106D ablation shards and verify full dimension coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-columns", type=int, required=True)
    parser.add_argument("--expected-dimensions", type=int, default=106)
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260825, 20260826, 20260827))
    args = parser.parse_args()

    roots = [path.resolve() for path in args.input_roots]
    tables = {name: [] for name in ("evaluation.csv", "training_curve.csv", "run_status.csv", "feature_support.csv")}
    manifests = []
    catalogs = []
    input_hashes: dict[str, str] = {}
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"]["datasets"] != [args.dataset] or manifest["counts"]["columns"] != args.expected_columns:
            raise AssertionError(f"dataset/column mismatch in {manifest_path}")
        if manifest["counts"]["evaluation_rows"] != manifest["counts"]["expected_evaluation_rows"]:
            raise AssertionError(f"incomplete shard {manifest_path}")
        manifests.append(manifest)
        catalogs.append(read_csv(root / "feature_catalog.csv"))
        for name in tables:
            path = root / name
            rows = read_csv(path)
            tables[name].extend(rows)
            input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        input_hashes[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    catalog = catalogs[0]
    if any(rows != catalog for rows in catalogs[1:]):
        raise AssertionError("feature catalogs differ")
    dimensions = sorted({int(row["disabled_dimension"]) for row in tables["evaluation.csv"]})
    if dimensions != list(range(args.expected_dimensions)):
        raise AssertionError(f"dimension coverage mismatch: {dimensions}")
    expected = args.expected_dimensions * args.expected_columns * len(args.seeds)
    if len(tables["evaluation.csv"]) != expected or len(tables["run_status.csv"]) != expected:
        raise AssertionError({"expected": expected, "evaluation": len(tables["evaluation.csv"]), "status": len(tables["run_status.csv"])})
    keys = [(row["algorithm"], int(row["seed"]), int(row["column_index"])) for row in tables["evaluation.csv"]]
    if len(set(keys)) != expected:
        raise AssertionError("duplicate or missing evaluation keys")
    if {row["dataset"] for row in tables["evaluation.csv"]} != {args.dataset}:
        raise AssertionError("dataset mismatch")
    if any(int(row["completed_iterations"]) != int(row["requested_iterations"]) or row["early_stop_reason"] for row in tables["run_status.csv"]):
        raise AssertionError("incomplete training run")
    support_expected = args.expected_dimensions * args.expected_columns
    if len(tables["feature_support.csv"]) != support_expected:
        raise AssertionError(f"expected {support_expected} support rows")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "feature_catalog.csv", catalog)
    for name, rows in tables.items():
        sort_key = lambda row: (
            int(row.get("disabled_dimension", row.get("dimension", -1))),
            int(row.get("seed", 0)),
            int(row["fold"]),
            int(row.get("iteration", 0)),
        )
        write_csv(output_dir / name, sorted(rows, key=sort_key))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset": args.dataset,
        "counts": {
            "dimensions": args.expected_dimensions,
            "continuous_dimensions": sum(catalog[dim]["inference_class"] == "continuous" for dim in dimensions),
            "one_hot_dimensions": sum(catalog[dim]["inference_class"] != "continuous" for dim in dimensions),
            "columns": args.expected_columns,
            "seeds": len(args.seeds),
            "evaluation_rows": expected,
            "training_runs": expected,
            "training_curve_rows": len(tables["training_curve.csv"]),
            "feature_support_rows": support_expected,
        },
        "protocol": manifests[0]["protocol"],
        "claim_boundary": manifests[0]["claim_boundary"],
        "input_hashes": input_hashes,
    }
    for name in ("feature_catalog.csv", *tables.keys()):
        path = output_dir / name
        manifest.setdefault("output_hashes", {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
