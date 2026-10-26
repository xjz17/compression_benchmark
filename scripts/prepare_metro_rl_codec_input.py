#!/usr/bin/env python3
"""Validate Metro held-out RL predictions and adapt them for codec replay."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ALGORITHMS = {"ppo", "grpo"}
SEEDS = {20260825, 20260826, 20260827}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.evaluation_csv.resolve())
    if len(rows) != 36:
        raise AssertionError(f"expected 36 rows, got {len(rows)}")
    if {row["algorithm"] for row in rows} != ALGORITHMS:
        raise AssertionError("algorithm coverage mismatch")
    if {int(row["seed"]) for row in rows} != SEEDS:
        raise AssertionError("seed coverage mismatch")
    if {row["dataset"] for row in rows} != {"Metro-Traffic"}:
        raise AssertionError("dataset coverage mismatch")
    columns = {(int(row["column_index"]), row["column_name"]) for row in rows}
    if len(columns) != 6:
        raise AssertionError(f"expected six columns, got {columns}")
    keys = [(row["algorithm"], int(row["seed"]), int(row["column_index"])) for row in rows]
    if any(count != 1 for count in Counter(keys).values()) or len(set(keys)) != 36:
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
        if len(pipeline) != int(row["depth"]) or not 1 <= len(pipeline) <= 3:
            raise AssertionError(f"invalid pipeline depth: {row}")
        adapted = dict(row)
        # The codec replayer historically calls this field safe_pipeline.  Here
        # it is exactly the raw greedy policy proposal: no fallback is applied.
        adapted["safe_pipeline"] = row["pipeline"]
        adapted["safe_proxy_rate"] = row["proxy_rate"]
        adapted["selection_contract"] = "raw_greedy_policy_no_fallback"
        output.append(adapted)
    if set(fold_by_column.values()) != set(range(6)) or any(
        list(fold_by_column.values()).count(fold) != 1 for fold in range(6)
    ):
        raise AssertionError(f"not leave-one-column-out: {fold_by_column}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, output)
    print(json.dumps({
        "status": "PASS",
        "rows": len(output),
        "columns": len(columns),
        "algorithms": sorted(ALGORITHMS),
        "seeds": sorted(SEEDS),
        "fold_by_column": fold_by_column,
        "selection_contract": "raw_greedy_policy_no_fallback",
    }, sort_keys=True))


if __name__ == "__main__":
    main()

