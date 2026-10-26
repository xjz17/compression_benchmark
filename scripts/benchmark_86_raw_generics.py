#!/usr/bin/env python3
"""Best raw-column general-codec baseline for the 86-codec composition report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_86_combinations import GENERAL_METHODS, best_generic, load_columns
from myapp.experiment_data_policy import NAN_INPUT_POLICY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--column", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    column = next(c for c in load_columns(args.dataset) if c.index == args.column)
    rows = []
    for method in GENERAL_METHODS:
        best, attempts = best_generic(column.raw, method)
        if best:
            rows.append({"method": method, **best, "levels_tested": len(attempts)})
    rows.sort(key=lambda row: (row["bytes"], row["method"], row["key"]))
    result = {
        "dataset": args.dataset, "column_index": args.column, "column_name": column.name,
        "original_bytes": len(column.raw), "best": rows[0] if rows else None, "top5": rows[:5],
        "rows": len(column.values),
        "source_rows": getattr(column, "source_count", len(column.values)),
        "removed_nan_count": getattr(column, "removed_nan_count", 0),
        "input_policy": NAN_INPUT_POLICY,
        "raw_sha256": hashlib.sha256(column.raw).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
