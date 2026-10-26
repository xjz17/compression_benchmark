#!/usr/bin/env python3
"""Combine frozen development and fresh calibration measurements for GRPO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FRESH_KEYS = {
    *(('Bird-migration', index) for index in range(5)),
    *(('meteoblue', index) for index in range(0, 61, 5)),
}
EXCLUDED_KEYS = {('Bird-migration', 5), ('Bird-migration', 6)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-measurement-root", type=Path, required=True)
    parser.add_argument("--fresh-measurement-root", type=Path, required=True)
    parser.add_argument("--old-summary-csv", type=Path, required=True)
    parser.add_argument("--old-baseline-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    old_paths = sorted(args.old_measurement_root.resolve().glob("*/pipeline_measurements.csv"))
    all_fresh_paths = sorted(args.fresh_measurement_root.resolve().glob("*/pipeline_measurements.csv"))
    fresh_paths = []
    excluded_paths = []
    for path in all_fresh_paths:
        shard_keys = {
            (row["dataset"], int(row["column_index"]))
            for row in read_csv(path.parent / "summary.csv")
        }
        if shard_keys <= FRESH_KEYS:
            fresh_paths.append(path)
        elif shard_keys <= EXCLUDED_KEYS:
            excluded_paths.append(path)
        else:
            raise AssertionError(f"unexpected fresh shard keys: {path} {shard_keys}")
    if len(old_paths) != 4 or len(fresh_paths) != 18 or len(excluded_paths) != 2:
        raise AssertionError(
            f"expected 4 old + 18 eligible + 2 grammar-ineligible fresh shards, got "
            f"{len(old_paths)} + {len(fresh_paths)} + {len(excluded_paths)}"
        )

    fresh_counts: Counter[tuple[str, int]] = Counter()
    fresh_rows = 0
    for path in fresh_paths:
        for row in read_csv(path):
            key = row["dataset"], int(row["column_index"])
            if int(row["lossless"]) != 1 or row["status"] != "ok":
                raise AssertionError(f"non-lossless fresh row: {path} {key}")
            fresh_counts[key] += 1
            fresh_rows += 1
    if set(fresh_counts) != FRESH_KEYS or set(fresh_counts.values()) != {22764}:
        raise AssertionError(f"fresh coverage mismatch: {fresh_counts}")

    old_summary = read_csv(args.old_summary_csv.resolve())
    fresh_summary = [row for path in fresh_paths for row in read_csv(path.parent / "summary.csv")]
    summary = old_summary + fresh_summary
    summary_keys = [(row["dataset"], int(row["column_index"])) for row in summary]
    if len(summary_keys) != len(set(summary_keys)) or set(summary_keys[-len(fresh_summary):]) != FRESH_KEYS:
        raise AssertionError("summary duplicate or fresh coverage mismatch")

    baseline = read_csv(args.old_baseline_csv.resolve())
    for row in fresh_summary:
        baseline.append({
            "dataset": row["dataset"],
            "column_index": row["column_index"],
            "column_name": row["column_name"],
            "logical_type": row["logical_type"],
            "sample_values": row["sample_values"],
            "sample_sha256": row["sample_sha256"],
            "candidate_pipeline_depth_le_3": row["best_single_pipeline"],
            "candidate_rate": row["best_single_rate"],
            "baseline_algorithm": "calibration_best_single_proxy",
            "baseline_rate": row["best_single_rate"],
            "gain_vs_baseline_pct": 0.0,
            "candidate_strictly_better": 0,
            "baseline_original_size": int(row["sample_values"]) * 8,
            "baseline_compressed_size": "",
            "valid_baseline_algorithms": "",
            "excluded_baseline_rows": "",
            "baseline_report": "fresh calibration placeholder; not used for final baseline claims",
        })
    baseline_keys = {(row["dataset"], int(row["column_index"])) for row in baseline}
    if baseline_keys != set(summary_keys):
        raise AssertionError("baseline/summary coverage mismatch")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    shard_root = output_root / "measurement_shards"
    shard_root.mkdir()
    for index, source in enumerate(old_paths + fresh_paths):
        target_dir = shard_root / f"shard_{index:02d}"
        target_dir.mkdir()
        (target_dir / "pipeline_measurements.csv").symlink_to(source)
    write_csv(output_root / "combined_summary.csv", summary)
    write_csv(output_root / "combined_baseline.csv", baseline)
    write_csv(output_root / "fresh_split.csv", [
        {
            "dataset": dataset,
            "column_index": index,
            "calibration_row_start": 0,
            "calibration_values": 32768,
            "evaluation_row_start": 32768,
            "evaluation_values": 32768,
        }
        for dataset, index in sorted(FRESH_KEYS)
    ])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "old_shards": {str(path): sha256(path) for path in old_paths},
        "fresh_shards": {str(path): sha256(path) for path in fresh_paths},
        "excluded_shards": {str(path): sha256(path) for path in excluded_paths},
        "old_summary_sha256": sha256(args.old_summary_csv.resolve()),
        "old_baseline_sha256": sha256(args.old_baseline_csv.resolve()),
        "counts": {
            "development_columns": len(old_summary),
            "fresh_columns": len(fresh_summary),
            "fresh_pipeline_rows": fresh_rows,
            "fresh_lossless_rows": fresh_rows,
            "combined_columns": len(summary),
            "measurement_shards": len(old_paths) + len(fresh_paths),
        },
        "split_contract": "fresh rows [0,32768) calibration; [32768,65536) final codec evaluation",
        "fresh_selection": {
            "Bird-migration": list(range(5)),
            "meteoblue": list(range(0, 61, 5)),
            "selection_used_method_results": False,
            "grammar_ineligible_before_training": {
                "Bird-migration": [5, 6],
                "reason": "at least one fixed 28-operator depth<=3 sequence is not lossless in the executable proxy",
            },
        },
        "baseline_boundary": "fresh combined_baseline rows are calibration best-single placeholders and cannot support CompressionBench baseline claims",
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
