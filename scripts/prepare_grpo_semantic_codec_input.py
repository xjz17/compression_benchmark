#!/usr/bin/env python3
"""Prepare frozen GRPO variants and Beam-29 for disjoint suffix codec replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


VARIANTS = ("C0", "E1", "E2", "E3", "D1", "D2", "D3", "D4")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def beam29(rows: list[dict[str, str]]) -> tuple[str, float, int]:
    rates = {
        tuple(part.strip() for part in row["pipeline"].split(" -> ")): float(row["rate"])
        for row in rows if int(row["lossless"]) == 1 and row["status"] == "ok" and math.isfinite(float(row["rate"]))
    }
    operators = sorted(pipeline[0] for pipeline in rates if len(pipeline) == 1)
    frontier: list[tuple[str, ...]] = [()]
    evaluated: set[tuple[str, ...]] = set()
    candidates: list[tuple[float, tuple[str, ...]]] = []
    for _depth in range(3):
        expanded = []
        for prefix in frontier:
            for operator in operators:
                pipeline = prefix + (operator,)
                rate = rates[pipeline]
                evaluated.add(pipeline)
                expanded.append((rate, pipeline))
                candidates.append((rate, pipeline))
        expanded.sort()
        frontier = [pipeline for _rate, pipeline in expanded[:29]]
    rate, pipeline = min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))
    return " -> ".join(pipeline), rate, len(evaluated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--fresh-measurement-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    evaluations = read_csv(args.evaluation_csv.resolve())
    keys = Counter((row["algorithm"], int(row["seed"]), row["dataset"], int(row["column_index"])) for row in evaluations)
    if len(evaluations) != 432 or set(row["algorithm"] for row in evaluations) != set(VARIANTS) or set(keys.values()) != {1}:
        raise AssertionError("expected 8 variants x 3 seeds x 18 grammar-eligible fresh columns")

    output: list[dict[str, object]] = []
    for row in evaluations:
        adapted = dict(row)
        adapted["safe_pipeline"] = row["pipeline"]
        adapted["safe_proxy_rate"] = row["proxy_rate"]
        adapted["selection_contract"] = "frozen development-trained greedy policy on calibration prefix"
        output.append(adapted)

    evaluation_columns = {(row["dataset"], int(row["column_index"])) for row in evaluations}
    by_column: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(args.fresh_measurement_root.resolve().glob("*/pipeline_measurements.csv")):
        for row in read_csv(path):
            key = row["dataset"], int(row["column_index"])
            if key in evaluation_columns:
                by_column[key].append(row)
    if set(by_column) != evaluation_columns or len(by_column) != 18:
        raise AssertionError(f"expected 18 grammar-eligible fresh columns, got {len(by_column)}")

    exemplar = {(row["dataset"], int(row["column_index"])): row for row in evaluations}
    beam_diagnostics = []
    for key, rows in sorted(by_column.items()):
        pipeline, rate, evaluations_count = beam29(rows)
        reference = exemplar[key]
        output.append({
            "algorithm": "Beam-29",
            "seed": -1,
            "fold": 0,
            "dataset": key[0],
            "column_index": key[1],
            "column_name": reference["column_name"],
            "logical_type": reference["logical_type"],
            "safe_pipeline": pipeline,
            "safe_proxy_rate": rate,
            "baseline_algorithm": reference["baseline_algorithm"],
            "baseline_rate": reference["baseline_rate"],
            "selection_contract": "Beam-29 on calibration prefix only",
        })
        beam_diagnostics.append({"dataset": key[0], "column_index": key[1], "evaluations": evaluations_count})
    if {row["evaluations"] for row in beam_diagnostics} != {1624}:
        raise AssertionError(beam_diagnostics)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, output)
    print(json.dumps({
        "status": "PASS",
        "rl_rows": len(evaluations),
        "beam_rows": len(beam_diagnostics),
        "output_rows": len(output),
        "beam_evaluations_per_column": 1624,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
