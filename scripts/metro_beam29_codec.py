#!/usr/bin/env python3
"""Run an actual-byte Beam search for one dataset with a standalone codec.

The search evaluates every depth-1 pipeline, expands the retained frontier at
depths 2 and 3, and selects the smallest encoded byte stream seen at any depth.
It is a falsification experiment for the normalized-int64 Python proxy codec,
not a production codec benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expected_evaluations(operator_count: int, width: int, max_depth: int) -> int:
    frontier = 1
    total = 0
    for _depth in range(1, max_depth + 1):
        expanded = frontier * operator_count
        total += expanded
        frontier = min(width, expanded)
    return total


def beam_search(
    values: np.ndarray,
    codec,
    operator_module,
    operators: tuple[str, ...],
    width: int,
    max_depth: int,
) -> tuple[tuple[int, ...], bytes, int, list[dict[str, int]]]:
    cache: dict[tuple[int, ...], bytes] = {}

    def encode(pipeline: tuple[int, ...]) -> bytes:
        if pipeline not in cache:
            names = tuple(operators[action] for action in pipeline)
            cache[pipeline] = codec.encode_stream(values, names, operator_module)
        return cache[pipeline]

    frontier: list[tuple[int, ...]] = [()]
    levels: list[dict[str, int]] = []
    for depth in range(1, max_depth + 1):
        expanded = [
            prefix + (action,)
            for prefix in frontier
            for action in range(len(operators))
        ]
        for pipeline in expanded:
            encode(pipeline)
        frontier = sorted(expanded, key=lambda pipeline: (len(cache[pipeline]), pipeline))[:width]
        levels.append(
            {
                "depth": depth,
                "expanded": len(expanded),
                "retained": len(frontier),
                "best_bytes_at_depth": len(cache[frontier[0]]),
            }
        )
    best = min(cache, key=lambda pipeline: (len(cache[pipeline]), len(pipeline), pipeline))
    return best, cache[best], len(cache), levels


def run(args: argparse.Namespace) -> None:
    data_repo = args.data_repo.resolve()
    baseline_csv = args.baseline_csv.resolve()
    output_dir = args.output_dir.resolve()
    codec_path = args.codec_path.resolve()
    codec = import_path(codec_path, "metro_beam_codec")
    operator_module = codec.load_operator_module(data_repo)
    operators = tuple(sorted(codec.OPERATOR_IDS))
    if len(operators) != 28 or len(set(operators)) != 28:
        raise AssertionError(f"expected 28 unique operators, got {len(operators)}")

    baseline_rows = [
        row for row in read_csv(baseline_csv) if row["dataset"] == args.dataset
    ]
    baseline_rows.sort(key=lambda row: int(row["column_index"]))
    if not baseline_rows:
        raise ValueError(f"no baseline rows for {args.dataset}")
    if len({int(row["column_index"]) for row in baseline_rows}) != len(baseline_rows):
        raise AssertionError("duplicate baseline column indices")

    input_rows = [
        {
            "dataset": row["dataset"],
            "column_index": row["column_index"],
            "column_name": row["column_name"],
        }
        for row in baseline_rows
    ]
    values_by_key = codec.load_values(data_repo, input_rows, operator_module)
    expected = expected_evaluations(len(operators), args.beam_width, args.max_depth)
    result_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []

    for baseline in baseline_rows:
        column_index = int(baseline["column_index"])
        key = (args.dataset, column_index)
        values = values_by_key[key]
        original_bytes = int(values.size * 8)
        input_sha256 = hashlib.sha256(
            values.astype("<i8", copy=False).tobytes()
        ).hexdigest()
        if int(baseline["sample_values"]) != int(values.size):
            raise AssertionError(f"sample count mismatch for {key}")
        if baseline["sample_sha256"] != input_sha256:
            raise AssertionError(f"sample hash mismatch for {key}")
        if int(baseline["baseline_original_size"]) != original_bytes:
            raise AssertionError(f"baseline denominator mismatch for {key}")

        started = time.perf_counter_ns()
        best, encoded, evaluations, levels = beam_search(
            values,
            codec,
            operator_module,
            operators,
            args.beam_width,
            args.max_depth,
        )
        search_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if evaluations != expected:
            raise AssertionError((evaluations, expected))
        restored, decoded_pipeline = codec.decode_stream(encoded)
        names = tuple(operators[action] for action in best)
        if decoded_pipeline != names or not np.array_equal(restored, values):
            raise AssertionError(f"standalone round trip failed for {key}")

        beam_rate = len(encoded) / float(original_bytes)
        baseline_rate = float(baseline["baseline_rate"])
        gain_pct = 100.0 * (baseline_rate - beam_rate) / baseline_rate
        result = {
            "dataset": args.dataset,
            "column_index": column_index,
            "column_name": baseline["column_name"],
            "logical_type": baseline["logical_type"],
            "sample_values": int(values.size),
            "original_bytes": original_bytes,
            "input_sha256": input_sha256,
            "beam_width": args.beam_width,
            "max_depth": args.max_depth,
            "candidate_evaluations": evaluations,
            "beam_pipeline": " -> ".join(names),
            "beam_pipeline_depth": len(best),
            "beam_encoded_bytes": len(encoded),
            "beam_rate": beam_rate,
            "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
            "baseline_algorithm": baseline["baseline_algorithm"],
            "baseline_compressed_bytes": int(baseline["baseline_compressed_size"]),
            "baseline_rate": baseline_rate,
            "gain_vs_baseline_pct": gain_pct,
            "beam_strictly_better": int(beam_rate < baseline_rate),
            "search_ms_single_run": search_ms,
            "lossless": 1,
        }
        result_rows.append(result)
        for level in levels:
            level_rows.append(
                {
                    "dataset": args.dataset,
                    "column_index": column_index,
                    "column_name": baseline["column_name"],
                    **level,
                }
            )
        print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_column.csv", result_rows)
    write_csv(output_dir / "search_levels.csv", level_rows)
    gains = [float(row["gain_vs_baseline_pct"]) for row in result_rows]
    total_original = sum(int(row["original_bytes"]) for row in result_rows)
    total_beam = sum(int(row["beam_encoded_bytes"]) for row in result_rows)
    total_baseline = sum(int(row["baseline_compressed_bytes"]) for row in result_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "supported_columns": len(result_rows),
        "beam_width": args.beam_width,
        "max_depth": args.max_depth,
        "operators": list(operators),
        "candidate_evaluations_per_column": expected,
        "total_candidate_evaluations": expected * len(result_rows),
        "lossless_columns": sum(int(row["lossless"]) for row in result_rows),
        "positive_columns": sum(int(row["beam_strictly_better"]) for row in result_rows),
        "median_column_gain_pct": statistics.median(gains),
        "mean_column_gain_pct": statistics.mean(gains),
        "min_column_gain_pct": min(gains),
        "max_column_gain_pct": max(gains),
        "aggregate_beam_rate": total_beam / total_original,
        "aggregate_baseline_rate": total_baseline / total_original,
        "aggregate_gain_pct": 100.0 * (total_baseline - total_beam) / total_baseline,
        "total_search_ms_single_run": sum(float(row["search_ms_single_run"]) for row in result_rows),
        "evaluation_type": "actual standalone normalized-int64 Python proxy codec",
        "decoder_contract": "decode_stream receives encoded bytes only",
        "baseline_contract": "best aligned per-column CompressionBench result from the supplied baseline CSV",
        "production_claim_allowed": False,
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "codec": sha256_file(codec_path),
            "baseline_csv": sha256_file(baseline_csv),
        },
    }
    with (output_dir / "MANIFEST.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"finished": manifest}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="Metro-Traffic")
    parser.add_argument("--beam-width", type=int, default=29)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()
    if args.beam_width <= 0 or args.max_depth <= 0:
        parser.error("--beam-width and --max-depth must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
