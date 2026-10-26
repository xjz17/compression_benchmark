#!/usr/bin/env python3
"""Build frozen external codec-replay input for Full106 and compact policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--no-zigzag", type=Path, required=True)
    parser.add_argument("--no-probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = (
        ("full_106", [row for row in read(args.full.resolve()) if row["algorithm"] == "C0"]),
        ("compact_no_zigzag", read(args.no_zigzag.resolve())),
        ("compact_no_probes", read(args.no_probes.resolve())),
    )
    output = []
    for label, values in sources:
        if len(values) != 54:
            raise AssertionError((label, len(values)))
        for row in values:
            adapted = dict(row)
            adapted["algorithm"] = label
            adapted["safe_pipeline"] = row["pipeline"]
            adapted["safe_proxy_rate"] = row["proxy_rate"]
            adapted["selection_contract"] = "development-locked frozen policy; external calibration prefix only"
            output.append(adapted)
    keys = {(row["algorithm"], row["seed"], row["dataset"], row["column_index"]) for row in output}
    if len(output) != 162 or len(keys) != 162:
        raise AssertionError((len(output), len(keys)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output[0])
    fieldnames.extend(sorted(set().union(*(row.keys() for row in output)) - set(fieldnames)))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"status": "PASS", "rows": len(output), "algorithms": [item[0] for item in sources]}, sort_keys=True))


if __name__ == "__main__":
    main()
