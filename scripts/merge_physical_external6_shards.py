#!/usr/bin/env python3
"""Merge disjoint-seed Bird/meteoblue external PPO feature shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(candidate: np.ndarray, baseline: np.ndarray, repeats: int = 20_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260827)
    values = np.empty(repeats)
    for index in range(repeats):
        take = rng.integers(0, candidate.size, candidate.size)
        values[index] = 100.0 * (candidate[take].sum() - baseline[take].sum()) / baseline[take].sum()
    return tuple(float(value) for value in np.percentile(values, (2.5, 50.0, 95.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shards = [path.resolve() for path in args.shard]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "policies").mkdir()

    evaluations: list[dict[str, str]] = []
    curves: list[dict[str, str]] = []
    paired: list[dict[str, str]] = []
    codec_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    input_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    seen_seeds: set[int] = set()
    shard_hashes: dict[str, str] = {}
    for shard in shards:
        if (shard / "exit.txt").read_text(encoding="utf-8").strip() != "0":
            raise AssertionError(f"failed shard: {shard}")
        result = shard / "results"
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        overlap = seen_seeds.intersection(int(seed) for seed in summary["seeds"])
        if overlap:
            raise AssertionError(f"overlapping seeds: {sorted(overlap)}")
        seen_seeds.update(int(seed) for seed in summary["seeds"])
        shard_hashes[str(shard)] = sha256(result / "manifest.json")
        evaluations.extend(read_csv(result / "evaluation.csv"))
        curves.extend(read_csv(result / "training_curve.csv"))
        paired.extend(read_csv(result / "paired_seed_summary.csv"))
        for row in read_csv(result / "codec_replay.csv"):
            key = row["dataset"], row["column_index"], row["pipeline"]
            if key in codec_by_key:
                old = codec_by_key[key]
                for field in ("final_bytes", "encoded_sha256", "lossless", "status"):
                    if old[field] != row[field]:
                        raise AssertionError((key, field, old[field], row[field]))
            else:
                codec_by_key[key] = row
        for row in read_csv(result / "external_inputs.csv"):
            input_groups.setdefault((row["dataset"], row["column_index"]), []).append(row)
        for policy in sorted((result / "policies").glob("*.npz")):
            target = output / "policies" / policy.name
            if target.exists():
                raise AssertionError(target)
            shutil.copy2(policy, target)

    evaluations.sort(key=lambda row: (row["variant"], int(row["seed"]), row["dataset"], int(row["column_index"])))
    paired.sort(key=lambda row: int(row["seed"]))
    if len(seen_seeds) != 10 or len(evaluations) != 10 * 2 * 18:
        raise AssertionError((len(seen_seeds), len(evaluations)))
    write_csv(output / "evaluation.csv", evaluations)
    write_csv(output / "training_curve.csv", curves)
    write_csv(output / "paired_seed_summary.csv", paired)
    write_csv(output / "codec_replay_deduplicated.csv", list(codec_by_key.values()))

    input_rows: list[dict[str, Any]] = []
    for key, rows in sorted(input_groups.items()):
        for field in ("column_name", "logical_type", "values", "original_bytes", "input_sha256"):
            if len({row[field] for row in rows}) != 1:
                raise AssertionError((key, field))
        exemplar = rows[0]
        input_rows.append(
            {
                "dataset": exemplar["dataset"],
                "column_index": exemplar["column_index"],
                "column_name": exemplar["column_name"],
                "logical_type": exemplar["logical_type"],
                "values": exemplar["values"],
                "original_bytes": exemplar["original_bytes"],
                "input_sha256": exemplar["input_sha256"],
                "full60_feature_ms_median": statistics.median(float(row["full60_feature_ms"]) for row in rows),
                "minimal6_feature_ms_median": statistics.median(float(row["candidate_feature_ms"]) for row in rows),
            }
        )
    write_csv(output / "external_inputs.csv", input_rows)

    keys = sorted({(row["dataset"], int(row["column_index"])) for row in evaluations})
    baseline_medians: list[float] = []
    candidate_medians: list[float] = []
    per_column: list[dict[str, Any]] = []
    for dataset, column in keys:
        full_values = [int(row["encoded_bytes"]) for row in evaluations if row["variant"] == "full60" and row["dataset"] == dataset and int(row["column_index"]) == column]
        candidate_values = [int(row["encoded_bytes"]) for row in evaluations if row["variant"] == "minimal6" and row["dataset"] == dataset and int(row["column_index"]) == column]
        full_median = float(statistics.median(full_values))
        candidate_median = float(statistics.median(candidate_values))
        baseline_medians.append(full_median)
        candidate_medians.append(candidate_median)
        per_column.append(
            {
                "dataset": dataset,
                "column_index": column,
                "full60_median_bytes": full_median,
                "minimal6_median_bytes": candidate_median,
                "minimal6_penalty_pct": 100.0 * (candidate_median - full_median) / full_median,
            }
        )
    write_csv(output / "per_column_comparison.csv", per_column)
    baseline_array = np.asarray(baseline_medians)
    candidate_array = np.asarray(candidate_medians)
    aggregate_penalty = 100.0 * (candidate_array.sum() - baseline_array.sum()) / baseline_array.sum()
    ci_low, ci_mid, ci_high = bootstrap(candidate_array, baseline_array)
    max_penalty = max(float(row["minimal6_penalty_pct"]) for row in per_column)
    strict = aggregate_penalty <= 0.25 and max_penalty <= 5.0 and ci_high <= 0.25
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if strict else "FAIL",
        "seeds": sorted(seen_seeds),
        "external_columns": len(keys),
        "evaluation_rows": len(evaluations),
        "codec_distinct_rows": len(codec_by_key),
        "codec_lossless_rows": sum(int(row["lossless"]) for row in codec_by_key.values()),
        "median_seed_full60_bytes": statistics.median(int(row["full60_bytes"]) for row in paired),
        "median_seed_minimal6_bytes": statistics.median(int(row["candidate_bytes"]) for row in paired),
        "median_seed_penalty_pct": statistics.median(float(row["candidate_penalty_pct"]) for row in paired),
        "aggregate_column_median_penalty_pct": aggregate_penalty,
        "max_column_median_penalty_pct": max_penalty,
        "paired_column_bootstrap_95pct": {"low": ci_low, "mid": ci_mid, "high": ci_high},
        "gate": "aggregate<=0.25%, max_column<=5%, bootstrap_upper<=0.25%",
        "claim_boundary": "Air+Metro-trained policies on fixed Bird/meteoblue external columns; actual PV2F proposal replay without external oracle enumeration",
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": summary["created_utc"],
        "source_sha256": sha256(Path(__file__).resolve()),
        "shard_manifest_sha256": shard_hashes,
        "summary_sha256": sha256(output / "SUMMARY.json"),
        "evaluation_sha256": sha256(output / "evaluation.csv"),
        "codec_sha256": sha256(output / "codec_replay_deduplicated.csv"),
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
