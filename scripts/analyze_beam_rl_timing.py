#!/usr/bin/env python3
"""Recompute comparable timing layers for Beam-29 and held-out RL.

The report deliberately keeps three clocks separate:

1. selector work: proxy evaluator time used to acquire target-column probes;
2. offline training: wall time of all folds and seeds;
3. selected codec execution: standalone Python encode/decode time.

It never presents a sum across these layers as a production end-to-end latency.
"""

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
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_dataset_spec(text: str) -> tuple[str, Path, Path, Path, Path]:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("dataset spec must be DATASET=beam,evaluation,training,algorithm_summary")
    paths = parts[1].split(",")
    if len(paths) != 4:
        raise argparse.ArgumentTypeError("dataset spec needs four comma-separated paths")
    return parts[0], *(Path(path).resolve() for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    measurements: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    measurement_paths = sorted(args.measurement_root.resolve().glob("*/pipeline_measurements.csv"))
    if len(measurement_paths) != 4:
        raise AssertionError(f"expected four measurement shards, got {len(measurement_paths)}")
    for path in measurement_paths:
        for row in read_csv(path):
            if int(row["lossless"]) != 1 or row["status"] != "ok":
                continue
            measurements[(row["dataset"], int(row["column_index"]))][row["pipeline"]] = float(row["eval_ms"])

    search_rows: list[dict[str, object]] = []
    online_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    input_paths: list[Path] = list(measurement_paths)

    for dataset, beam_path, evaluation_path, training_path, algorithm_summary_path in args.dataset:
        input_paths.extend((beam_path, evaluation_path, training_path, algorithm_summary_path))
        beams = read_csv(beam_path)
        evaluations = read_csv(evaluation_path)
        training = read_csv(training_path)
        algorithm_summary = {row["algorithm"]: row for row in read_csv(algorithm_summary_path)}
        if {row["dataset"] for row in beams + evaluations} != {dataset}:
            raise AssertionError(f"dataset mismatch for {dataset}")

        for beam in beams:
            column_index = int(beam["column_index"])
            table = measurements[(dataset, column_index)]
            if len(table) != 22_764:
                raise AssertionError((dataset, column_index, len(table)))
            single_ms = sum(value for pipeline, value in table.items() if " -> " not in pipeline)
            exhaustive_ms = sum(table.values())
            beam_ms = float(beam["search_ms_single_run"])
            search_rows.append({
                "dataset": dataset,
                "column_index": column_index,
                "column_name": beam["column_name"],
                "exhaustive_candidates": 22_764,
                "beam29_candidates": int(beam["candidate_evaluations"]),
                "exhaustive_proxy_eval_ms_sum": exhaustive_ms,
                "beam29_observed_search_ms": beam_ms,
                "beam29_speedup_vs_exhaustive_eval_sum": exhaustive_ms / beam_ms,
                "single_probe_28_eval_ms_sum": single_ms,
            })

        for row in evaluations:
            column_index = int(row["column_index"])
            table = measurements[(dataset, column_index)]
            pipeline = [part.strip() for part in row["pipeline"].split(" -> ") if part.strip()]
            probe_ms = sum(value for key, value in table.items() if " -> " not in key)
            added_prefix_ms = 0.0
            for depth in range(2, len(pipeline) + 1):
                added_prefix_ms += table[" -> ".join(pipeline[:depth])]
            online_rows.append({
                "dataset": dataset,
                "algorithm": row["algorithm"],
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "column_index": column_index,
                "column_name": row["column_name"],
                "pipeline": row["pipeline"],
                "depth": int(row["depth"]),
                "single_probe_28_eval_ms_sum": probe_ms,
                "additional_prefix_eval_ms_sum": added_prefix_ms,
                "online_proxy_work_ms_sum": probe_ms + added_prefix_ms,
                "candidate_evaluations_upper_bound": 31,
            })

        final_elapsed: dict[tuple[str, int, int], float] = {}
        for row in training:
            key = row["algorithm"], int(row["seed"]), int(row["fold"])
            final_elapsed[key] = max(final_elapsed.get(key, 0.0), float(row["elapsed_sec"]))
        dataset_search = [row for row in search_rows if row["dataset"] == dataset]
        for algorithm in sorted({row["algorithm"] for row in evaluations}):
            online = [float(row["online_proxy_work_ms_sum"]) for row in online_rows if row["dataset"] == dataset and row["algorithm"] == algorithm]
            elapsed = [value for key, value in final_elapsed.items() if key[0] == algorithm]
            codec = algorithm_summary[algorithm]
            summary_rows.append({
                "dataset": dataset,
                "algorithm": algorithm,
                "columns": len(beams),
                "seeds": len({int(row["seed"]) for row in evaluations if row["algorithm"] == algorithm}),
                "beam29_total_search_sec": sum(float(row["beam29_observed_search_ms"]) for row in dataset_search) / 1000.0,
                "beam29_median_search_sec_per_column": statistics.median(float(row["beam29_observed_search_ms"]) for row in dataset_search) / 1000.0,
                "exhaustive_total_proxy_eval_sec_sum": sum(float(row["exhaustive_proxy_eval_ms_sum"]) for row in dataset_search) / 1000.0,
                "beam29_speedup_vs_exhaustive_total": sum(float(row["exhaustive_proxy_eval_ms_sum"]) for row in dataset_search) / sum(float(row["beam29_observed_search_ms"]) for row in dataset_search),
                "rl_online_proxy_work_median_ms_per_prediction": statistics.median(online),
                "rl_online_proxy_work_sum_of_column_medians_ms": sum(
                    statistics.median(
                        float(row["online_proxy_work_ms_sum"])
                        for row in online_rows
                        if row["dataset"] == dataset and row["algorithm"] == algorithm and int(row["column_index"]) == column
                    )
                    for column in sorted({int(row["column_index"]) for row in online_rows if row["dataset"] == dataset and row["algorithm"] == algorithm})
                ),
                "offline_training_sum_sec": sum(elapsed),
                "offline_training_median_sec_per_fold_seed": statistics.median(elapsed),
                "standalone_codec_median_encode_ms": float(codec["median_encode_ms"]),
                "standalone_codec_median_decode_ms": float(codec["median_decode_ms"]),
                "timing_scope": "proxy selector work + offline training + standalone normalized-int64 Python codec, reported separately",
            })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "per_column_search.csv", search_rows)
    write_csv(output_dir / "rl_online_proxy_cost.csv", online_rows)
    write_csv(output_dir / "dataset_algorithm_summary.csv", summary_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "timing_contract": {
            "beam": "observed one-run Beam-29 search wall time on compression-166",
            "exhaustive": "sum of stored per-pipeline Python proxy evaluator times; not observed serial process wall time",
            "rl_online": "sum of 28 stored single-probe evaluator times plus selected depth-2/depth-3 prefix evaluator times; actor inference and raw feature scan excluded",
            "training": "sum and median of stored fold-seed training elapsed time",
            "codec": "median standalone normalized-int64 Python codec time from five repetitions",
            "production_claim_allowed": False,
        },
        "counts": {
            "search_rows": len(search_rows),
            "online_rows": len(online_rows),
            "summary_rows": len(summary_rows),
        },
        "inputs": {str(path): sha256(path) for path in input_paths},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
