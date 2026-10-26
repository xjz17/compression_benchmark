#!/usr/bin/env python3
"""Measure deeper fixed-horizon beam search over all 28 operator actions.

The best result is retained across depths, which is equivalent to padding the
remaining positions with a STOP/no-op action.  At depths beyond the exhaustive
three-layer run, a wide beam is used only as a stronger reference, not as an
oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def load_operator_module(repo_root: Path):
    source = repo_root / "scripts" / "generate_air_operator_pair_heatmap.py"
    spec = importlib.util.spec_from_file_location("deeper_beam_operator_impl", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load operator implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source


def locate_dataset(repo_root: Path, dataset: str) -> tuple[Path, Path]:
    data_candidates = (
        repo_root / "orignial_datasets" / f"{dataset}.csv",
        repo_root / "backend" / "dataset_csv" / f"{dataset}.csv",
    )
    info_candidates = (
        repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv",
        repo_root / "dataset" / dataset / f"{dataset}_column_info.csv",
    )
    data_path = next((path for path in data_candidates if path.is_file()), None)
    info_path = next((path for path in info_candidates if path.is_file()), None)
    if data_path is None or info_path is None:
        raise FileNotFoundError(f"Missing dataset or column metadata for {dataset}")
    return data_path, info_path


def load_columns(module, repo_root: Path, dataset: str, column_indices: set[int]):
    data_path, info_path = locate_dataset(repo_root, dataset)
    columns = [
        row
        for row in module.numeric_columns(info_path)
        if int(row["column_index"]) in column_indices
    ]
    frame = pd.read_csv(
        data_path,
        usecols=[row["column_name"] for row in columns],
        low_memory=False,
        dtype=str,
    )
    for column in columns:
        values = module.column_values(frame[column["column_name"]], column)
        if values.size:
            yield data_path, info_path, column, values


def pipeline_text(pipeline: tuple[str, ...]) -> str:
    return " -> ".join(pipeline)


def push_best(heap, width: int, rate: float, serial: int, pipeline, state) -> None:
    item = (-rate, serial, pipeline, state)
    if len(heap) < width:
        heapq.heappush(heap, item)
    elif rate < -heap[0][0]:
        heapq.heapreplace(heap, item)


def run_beam(
    module,
    original: np.ndarray,
    operators: tuple[str, ...],
    width: int,
    max_depth: int,
    allow_repeated_operators: bool = True,
):
    original_size = len(module.i64_bytes(original))
    frontier = [((), module.initial_state(original))]
    best_rate = math.inf
    best_pipeline: tuple[str, ...] = ()
    evaluations = 0
    serial = 0
    rows = []
    started = time.perf_counter()
    for depth in range(1, max_depth + 1):
        heap = []
        lossless_count = 0
        depth_started = time.perf_counter()
        for prefix, state in frontier:
            for operator in operators:
                if not allow_repeated_operators and operator in prefix:
                    continue
                pipeline = prefix + (operator,)
                evaluations += 1
                try:
                    next_state = module.apply_operator(state, operator)
                    rate, lossless = module.checked_rate(
                        next_state, original, original_size
                    )
                except Exception:
                    continue
                if not lossless or not math.isfinite(rate):
                    continue
                lossless_count += 1
                if (rate, len(pipeline), pipeline) < (
                    best_rate,
                    len(best_pipeline),
                    best_pipeline,
                ):
                    best_rate = rate
                    best_pipeline = pipeline
                serial += 1
                push_best(heap, width, rate, serial, pipeline, next_state)
        frontier = [
            (pipeline, state)
            for _neg_rate, _serial, pipeline, state in sorted(
                heap, key=lambda item: (-item[0], item[2])
            )
        ]
        rows.append(
            {
                "depth": depth,
                "beam_width": width,
                "best_rate_through_depth": best_rate,
                "best_pipeline_through_depth": pipeline_text(best_pipeline),
                "frontier_best_rate": -max(heap)[0] if heap else math.nan,
                "frontier_size": len(frontier),
                "lossless_expansions": lossless_count,
                "cumulative_evaluations": evaluations,
                "depth_ms": (time.perf_counter() - depth_started) * 1000.0,
                "cumulative_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        if not frontier:
            break
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = (
        "dataset",
        "column_index",
        "column_name",
        "logical_type",
        "sample_values",
        "sample_sha256",
        "depth",
        "beam_width",
        "best_rate_through_depth",
        "best_pipeline_through_depth",
        "frontier_best_rate",
        "frontier_size",
        "lossless_expansions",
        "cumulative_evaluations",
        "depth_ms",
        "cumulative_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--column-indices", nargs="+", type=int, required=True)
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[4, 256])
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument(
        "--disallow-repeated-operators",
        action="store_true",
        help="Do not place the same operator more than once in one pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_depth < 1:
        raise ValueError("max-depth must be positive")
    if any(width < 1 for width in args.beam_widths):
        raise ValueError("beam widths must be positive")
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    module, operator_source = load_operator_module(repo_root)
    operators = tuple(item.operator_id for item in module.OPERATORS)
    if len(operators) != 28 or len(set(operators)) != 28:
        raise RuntimeError(f"Expected 28 unique operators, got {len(operators)}")

    all_rows = []
    inputs = []
    for data_path, info_path, column, original in load_columns(
        module, repo_root, args.dataset, set(args.column_indices)
    ):
        sample_hash = hashlib.sha256(module.i64_bytes(original)).hexdigest()
        inputs.append(
            {
                "dataset": args.dataset,
                "column_index": int(column["column_index"]),
                "column_name": column["column_name"],
                "sample_values": int(original.size),
                "sample_sha256": sample_hash,
                "data_path": str(data_path),
                "column_info_path": str(info_path),
            }
        )
        for width in sorted(set(args.beam_widths)):
            print(
                f"RUN dataset={args.dataset} column={column['column_index']} "
                f"name={column['column_name']} beam={width} depth={args.max_depth}",
                flush=True,
            )
            rows = run_beam(
                module,
                original,
                operators,
                width,
                args.max_depth,
                allow_repeated_operators=not args.disallow_repeated_operators,
            )
            for row in rows:
                row.update(
                    {
                        "dataset": args.dataset,
                        "column_index": int(column["column_index"]),
                        "column_name": column["column_name"],
                        "logical_type": str(column.get("logical_type", "")).strip().lower(),
                        "sample_values": int(original.size),
                        "sample_sha256": sample_hash,
                    }
                )
            all_rows.extend(rows)
            write_rows(output_dir / "beam_depth_results.csv", all_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operator_source": str(operator_source),
        "operator_source_sha256": hashlib.sha256(operator_source.read_bytes()).hexdigest(),
        "config": {
            "dataset": args.dataset,
            "column_indices": args.column_indices,
            "beam_widths": sorted(set(args.beam_widths)),
            "max_depth": args.max_depth,
            "operators": operators,
            "allow_repeated_operators": not args.disallow_repeated_operators,
            "stop_semantics": "best result through depth; unused suffix is STOP/no-op",
            "reference_semantics": "widest beam is a stronger reference, not an exhaustive oracle",
        },
        "inputs": inputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT {output_dir}")


if __name__ == "__main__":
    main()
