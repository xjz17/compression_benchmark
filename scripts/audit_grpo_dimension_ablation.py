#!/usr/bin/env python3
"""Independent count, losslessness, finiteness, and hash audit for 106D LOFO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_manifest_outputs(root: Path, manifest_name: str) -> int:
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("output_hashes", manifest.get("outputs", {}))
    for name, expected in hashes.items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError((path, expected, actual))
    return len(hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metro-root", type=Path, required=True)
    parser.add_argument("--air-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dataset_specs = ((args.metro_root.resolve(), 6, 1908), (args.air_root.resolve(), 14, 4452))
    total_codec = 0
    unique_pairs = 0
    total_training_rows = 0
    for root, columns, expected in dataset_specs:
        evaluation = rows(root / "raw/evaluation.csv")
        status = rows(root / "raw/run_status.csv")
        training = rows(root / "raw/training_curve.csv")
        codec = rows(root / "codec/per_seed_codec_cost.csv")
        if not (len(evaluation) == len(status) == len(codec) == expected):
            raise AssertionError((root, len(evaluation), len(status), len(codec), expected))
        algorithms = {row["algorithm"] for row in evaluation}
        if algorithms != {f"dim_{dim:03d}" for dim in range(106)}:
            raise AssertionError(f"dimension coverage failed at {root}")
        keys = [(row["algorithm"], int(row["seed"]), int(row["column_index"])) for row in evaluation]
        if len(set(keys)) != expected:
            raise AssertionError(f"duplicate evaluation key at {root}")
        if any(int(row["completed_iterations"]) != int(row["requested_iterations"]) or row["early_stop_reason"] for row in status):
            raise AssertionError(f"incomplete run at {root}")
        if any(int(row["lossless"]) != 1 for row in codec):
            raise AssertionError(f"non-lossless row at {root}")
        numeric_fields = (
            "clip_fraction", "approx_kl", "policy_entropy", "effective_action_count",
            "mean_gradient_norm", "max_gradient_norm", "mean_group_dispersion", "elapsed_sec",
        )
        if any(not math.isfinite(float(row[field])) for row in training for field in numeric_fields):
            raise AssertionError(f"non-finite diagnostic at {root}")
        total_codec += len(codec)
        unique_pairs += len({(row["dataset"], int(row["column_index"]), row["safe_pipeline"]) for row in codec})
        total_training_rows += len(training)
        verify_manifest_outputs(root / "raw", "manifest.json")
        verify_manifest_outputs(root / "analysis", "MANIFEST.json")

    dimension_rows = rows(args.analysis_root.resolve() / "per_dimension.csv")
    if len(dimension_rows) != 106 or {int(row["dimension"]) for row in dimension_rows} != set(range(106)):
        raise AssertionError("per-dimension coverage failed")
    verdicts = Counter(row["verdict"] for row in dimension_rows)
    expected_verdicts = {
        "inconclusive": 44,
        "non_identifiable_zero_variance": 3,
        "non_identifiable_one_hot": 59,
    }
    if dict(verdicts) != expected_verdicts:
        raise AssertionError((dict(verdicts), expected_verdicts))
    verify_manifest_outputs(args.analysis_root.resolve(), "MANIFEST.json")

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "training_units": 6360,
        "training_curve_rows": total_training_rows,
        "codec_result_rows": total_codec,
        "unique_codec_column_pipeline_pairs": unique_pairs,
        "dimensions": len(dimension_rows),
        "verdict_counts": dict(verdicts),
        "all_lossless": True,
        "all_training_complete": True,
        "all_selected_diagnostics_finite": True,
    }
    args.output_json.resolve().write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
