#!/usr/bin/env python3
"""Validate held-out RL predictions and adapt raw greedy pipelines for codec replay."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-columns", type=int, required=True)
    parser.add_argument("--algorithms", nargs="+", default=("ppo", "grpo"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260825, 20260826, 20260827))
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    algorithms = set(args.algorithms)
    seeds = set(args.seeds)
    rows = read_csv(args.evaluation_csv.resolve())
    expected_rows = args.expected_columns * len(algorithms) * len(seeds)
    if len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {len(rows)}")
    if {row["algorithm"] for row in rows} != algorithms:
        raise AssertionError("algorithm coverage mismatch")
    if {int(row["seed"]) for row in rows} != seeds:
        raise AssertionError("seed coverage mismatch")
    if {row["dataset"] for row in rows} != {args.dataset}:
        raise AssertionError("dataset coverage mismatch")

    columns = {(int(row["column_index"]), row["column_name"]) for row in rows}
    if len(columns) != args.expected_columns:
        raise AssertionError(f"expected {args.expected_columns} columns, got {columns}")
    keys = [
        (row["algorithm"], int(row["seed"]), int(row["column_index"]))
        for row in rows
    ]
    if len(set(keys)) != expected_rows or any(count != 1 for count in Counter(keys).values()):
        raise AssertionError("duplicate or missing algorithm/seed/column evaluation")

    fold_by_column: dict[int, int] = {}
    output: list[dict[str, object]] = []
    for row in rows:
        column_index = int(row["column_index"])
        fold = int(row["fold"])
        previous = fold_by_column.setdefault(column_index, fold)
        if previous != fold:
            raise AssertionError(f"column {column_index} has inconsistent folds")
        pipeline = [part.strip() for part in row["pipeline"].split(" -> ") if part.strip()]
        if len(pipeline) != int(row["depth"]) or not 1 <= len(pipeline) <= args.max_depth:
            raise AssertionError(f"invalid pipeline depth: {row}")
        adapted = dict(row)
        adapted["safe_pipeline"] = row["pipeline"]
        adapted["safe_proxy_rate"] = row["proxy_rate"]
        adapted["selection_contract"] = "raw_greedy_policy_no_fallback"
        output.append(adapted)

    folds = list(fold_by_column.values())
    if set(folds) != set(range(args.expected_columns)) or any(
        folds.count(fold) != 1 for fold in range(args.expected_columns)
    ):
        raise AssertionError(f"not leave-one-column-out: {fold_by_column}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, output)
    print(json.dumps({
        "status": "PASS",
        "dataset": args.dataset,
        "rows": len(output),
        "columns": len(columns),
        "algorithms": sorted(algorithms),
        "seeds": sorted(seeds),
        "fold_by_column": fold_by_column,
        "selection_contract": "raw_greedy_policy_no_fallback",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
