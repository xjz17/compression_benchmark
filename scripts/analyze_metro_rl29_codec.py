#!/usr/bin/env python3
"""Audit Metro PPO/GRPO codec replay and compare it with Beam-29."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct_gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--codec-csv", type=Path, required=True)
    parser.add_argument("--beam-csv", type=Path, required=True)
    parser.add_argument("--run-status-csv", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    evaluations = read_csv(args.evaluation_csv.resolve())
    codec = read_csv(args.codec_csv.resolve())
    beams = read_csv(args.beam_csv.resolve())
    status = read_csv(args.run_status_csv.resolve())
    if len(evaluations) != 36 or len(codec) != 36 or len(beams) != 6 or len(status) != 36:
        raise AssertionError({"evaluation": len(evaluations), "codec": len(codec), "beam": len(beams), "runs": len(status)})
    if any(int(row["lossless"]) != 1 for row in codec) or any(int(row["lossless"]) != 1 for row in beams):
        raise AssertionError("non-lossless row")
    if any(int(row["completed_iterations"]) != int(row["requested_iterations"]) or row["early_stop_reason"] for row in status):
        raise AssertionError("incomplete training run")

    eval_by_key = {
        (row["algorithm"], int(row["seed"]), int(row["column_index"])): row
        for row in evaluations
    }
    codec_by_key = {
        (row["algorithm"], int(row["seed"]), int(row["column_index"])): row
        for row in codec
    }
    if set(eval_by_key) != set(codec_by_key) or len(eval_by_key) != 36:
        raise AssertionError("evaluation/codec key mismatch")
    beam_by_column = {int(row["column_index"]): row for row in beams}
    if len(beam_by_column) != 6:
        raise AssertionError("beam column coverage mismatch")
    for key, eval_row in eval_by_key.items():
        codec_row = codec_by_key[key]
        beam = beam_by_column[key[2]]
        if eval_row["pipeline"] != codec_row["safe_pipeline"]:
            raise AssertionError(f"pipeline mismatch {key}")
        if codec_row["input_sha256"] != beam["input_sha256"]:
            raise AssertionError(f"input hash mismatch {key}")
        if int(codec_row["original_bytes"]) != int(beam["original_bytes"]):
            raise AssertionError(f"denominator mismatch {key}")

    per_column: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in codec:
        grouped[(row["algorithm"], int(row["column_index"]))].append(row)
    for (algorithm, column_index), rows in sorted(grouped.items()):
        if len(rows) != 3 or len({int(row["seed"]) for row in rows}) != 3:
            raise AssertionError(f"seed coverage mismatch {(algorithm, column_index)}")
        beam = beam_by_column[column_index]
        rates = [float(row["codec_rate"]) for row in rows]
        bytes_values = [int(row["encoded_bytes"]) for row in rows]
        baseline_rate = float(rows[0]["baseline_rate"])
        beam_rate = float(beam["beam_rate"])
        median_rate = statistics.median(rates)
        pipelines = [row["safe_pipeline"] for row in rows]
        median_row = sorted(
            rows,
            key=lambda row: (float(row["codec_rate"]), row["safe_pipeline"], int(row["seed"])),
        )[1]
        per_column.append({
            "algorithm": algorithm,
            "column_index": column_index,
            "column_name": rows[0]["column_name"],
            "baseline_algorithm": rows[0]["baseline_algorithm"],
            "baseline_rate": baseline_rate,
            "beam_pipeline": beam["beam_pipeline"],
            "beam_rate": beam_rate,
            "median_policy_pipeline": median_row["safe_pipeline"],
            "distinct_policy_pipelines": len(set(pipelines)),
            "median_encoded_bytes": int(statistics.median(bytes_values)),
            "median_codec_rate": median_rate,
            "min_codec_rate": min(rates),
            "max_codec_rate": max(rates),
            "gain_vs_baseline_pct": pct_gain(baseline_rate, median_rate),
            "gain_vs_beam29_pct": pct_gain(beam_rate, median_rate),
            "seeds_beating_baseline": sum(rate < baseline_rate for rate in rates),
            "seeds_beating_beam29": sum(rate < beam_rate for rate in rates),
            "median_encode_ms": statistics.median(float(row["encode_ms_median"]) for row in rows),
            "median_decode_ms": statistics.median(float(row["decode_ms_median"]) for row in rows),
            "lossless_seeds": sum(int(row["lossless"]) for row in rows),
        })

    per_seed: list[dict[str, object]] = []
    for algorithm in ("ppo", "grpo"):
        for seed in (20260825, 20260826, 20260827):
            rows = [row for row in codec if row["algorithm"] == algorithm and int(row["seed"]) == seed]
            if len(rows) != 6:
                raise AssertionError((algorithm, seed, len(rows)))
            encoded = sum(int(row["encoded_bytes"]) for row in rows)
            original = sum(int(row["original_bytes"]) for row in rows)
            beam_encoded = sum(int(beam_by_column[int(row["column_index"])]["beam_encoded_bytes"]) for row in rows)
            baseline_encoded = sum(int(beam_by_column[int(row["column_index"])]["baseline_compressed_bytes"]) for row in rows)
            rate = encoded / original
            per_seed.append({
                "algorithm": algorithm,
                "seed": seed,
                "encoded_bytes": encoded,
                "original_bytes": original,
                "aggregate_codec_rate": rate,
                "aggregate_beam29_rate": beam_encoded / original,
                "aggregate_baseline_rate": baseline_encoded / original,
                "gain_vs_beam29_pct": pct_gain(beam_encoded / original, rate),
                "gain_vs_baseline_pct": pct_gain(baseline_encoded / original, rate),
                "columns_beating_beam29": sum(float(row["codec_rate"]) < float(beam_by_column[int(row["column_index"])]["beam_rate"]) for row in rows),
                "columns_beating_baseline": sum(float(row["codec_rate"]) < float(row["baseline_rate"]) for row in rows),
            })

    algorithm_summary: list[dict[str, object]] = []
    original_total = sum(int(row["original_bytes"]) for row in beams)
    beam_bytes_total = sum(int(row["beam_encoded_bytes"]) for row in beams)
    baseline_bytes_total = sum(int(row["baseline_compressed_bytes"]) for row in beams)
    for algorithm in ("ppo", "grpo"):
        rows = [row for row in per_column if row["algorithm"] == algorithm]
        selected_bytes = sum(int(row["median_encoded_bytes"]) for row in rows)
        selected_rate = selected_bytes / original_total
        seed_rows = [row for row in per_seed if row["algorithm"] == algorithm]
        algorithm_summary.append({
            "algorithm": algorithm,
            "columns": 6,
            "seeds": 3,
            "aggregate_of_column_medians_rate": selected_rate,
            "aggregate_beam29_rate": beam_bytes_total / original_total,
            "aggregate_baseline_rate": baseline_bytes_total / original_total,
            "gain_vs_beam29_pct": pct_gain(beam_bytes_total / original_total, selected_rate),
            "gain_vs_baseline_pct": pct_gain(baseline_bytes_total / original_total, selected_rate),
            "median_seed_aggregate_rate": statistics.median(float(row["aggregate_codec_rate"]) for row in seed_rows),
            "min_seed_aggregate_rate": min(float(row["aggregate_codec_rate"]) for row in seed_rows),
            "max_seed_aggregate_rate": max(float(row["aggregate_codec_rate"]) for row in seed_rows),
            "columns_median_beating_beam29": sum(float(row["median_codec_rate"]) < float(row["beam_rate"]) for row in rows),
            "columns_all_seeds_beating_beam29": sum(int(row["seeds_beating_beam29"]) == 3 for row in rows),
            "columns_median_beating_baseline": sum(float(row["median_codec_rate"]) < float(row["baseline_rate"]) for row in rows),
            "median_encode_ms": statistics.median(float(row["median_encode_ms"]) for row in rows),
            "median_decode_ms": statistics.median(float(row["median_decode_ms"]) for row in rows),
            "policy_candidate_evaluations_upper_bound": 31,
            "beam29_candidate_evaluations": 1624,
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_column_comparison.csv", per_column)
    write_csv(output_dir / "per_seed_aggregate.csv", per_seed)
    write_csv(output_dir / "algorithm_summary.csv", algorithm_summary)
    inputs = [args.evaluation_csv, args.codec_csv, args.beam_csv, args.run_status_csv, args.raw_manifest]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "counts": {"evaluation_rows": 36, "codec_rows": 36, "beam_rows": 6, "training_runs": 36},
        "selection_contract": "raw held-out greedy policy; no best-single or Beam fallback",
        "split_contract": "six-fold leave-one-column-out, three seeds",
        "action_contract": "29 actions = 28 operators + absorbing STOP; one to three non-empty operators",
        "codec_contract": "standalone normalized-int64 Python codec; decoder receives only encoded bytes",
        "all_input_hashes_match_beam29": True,
        "all_lossless": True,
        "candidate_count_contract": "28 single-operator probes plus at most 3 policy actions versus 1,624 Beam-29 candidate evaluations",
        "production_claim_allowed": False,
        "algorithms": algorithm_summary,
        "inputs": {str(path.resolve()): hashlib.sha256(path.resolve().read_bytes()).hexdigest() for path in inputs},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
