#!/usr/bin/env python3
"""Freshly re-encode every distinct pipeline selected by final feature runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve()
    sys.path.insert(0, str(script.parent))
    base_path = script.with_name("search_physical_operator_variants_v2.py")
    runtime_path = script.with_name("physical_operator_runtime_v2.py")
    base = load_module(base_path, "selected_replay_base")
    runtime = load_module(runtime_path, "selected_replay_runtime")
    repo_root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tables = {table.key: table for table in base.load_tables(args.measurement_root.resolve())}

    claims: dict[tuple[str, int, tuple[str, ...]], dict[str, Any]] = {}
    source_rows = 0
    for path in args.evaluation:
        for row in read_csv(path.resolve()):
            source_rows += 1
            dataset = row.get("test_dataset") or row.get("dataset")
            column = int(row["column_index"])
            pipeline = base.parse_pipeline(row["pipeline"])
            key = (dataset, column, pipeline)
            encoded_bytes = int(row["encoded_bytes"])
            if key in claims and claims[key]["expected_bytes"] != encoded_bytes:
                raise AssertionError((key, claims[key], encoded_bytes))
            claims[key] = {
                "expected_bytes": encoded_bytes,
                "source_files": sorted(set(claims.get(key, {}).get("source_files", []) + [str(path.resolve())])),
            }

    values_cache: dict[tuple[str, int], np.ndarray] = {}
    replay_rows: list[dict[str, Any]] = []
    for (dataset, column, pipeline), claim in sorted(claims.items()):
        table = tables[(dataset, column)]
        if pipeline not in table.sizes:
            raise AssertionError((dataset, column, pipeline, "missing from frozen table"))
        if table.sizes[pipeline] != claim["expected_bytes"]:
            raise AssertionError((dataset, column, pipeline, table.sizes[pipeline], claim["expected_bytes"]))
        values = values_cache.get((dataset, column))
        if values is None:
            values, _metadata, _data, _info = base.load_column(repo_root, dataset, column, 0)
            values_cache[(dataset, column)] = values
        transforms, packer, compressor = base.split_prefix(pipeline)
        if packer is None or compressor is None:
            raise AssertionError(pipeline)
        encoded = runtime.encode_pipeline(values, transforms, packer, compressor)
        restored, decoded = runtime.decode_frame(encoded.data)
        lossless = int(decoded == pipeline and np.array_equal(restored, values))
        exact_bytes = int(len(encoded.data) == claim["expected_bytes"] == table.sizes[pipeline])
        replay_rows.append(
            {
                "dataset": dataset,
                "column_index": column,
                "column_name": table.column_name,
                "pipeline": " -> ".join(pipeline),
                "expected_bytes": claim["expected_bytes"],
                "replayed_bytes": len(encoded.data),
                "exact_bytes": exact_bytes,
                "lossless": lossless,
                "input_sha256": runtime.sha256_bytes(runtime.raw_i64(values)),
                "encoded_sha256": runtime.sha256_bytes(encoded.data),
                "source_files": "|".join(claim["source_files"]),
                "status": "PASS" if exact_bytes and lossless else "FAIL",
            }
        )
    write_csv(output / "selected_pipeline_replay.csv", replay_rows)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["status"] == "PASS" for row in replay_rows) else "FAIL",
        "source_rows": source_rows,
        "distinct_column_pipelines": len(replay_rows),
        "exact_byte_replays": sum(int(row["exact_bytes"]) for row in replay_rows),
        "lossless_replays": sum(int(row["lossless"]) for row in replay_rows),
        "source_sha256": {
            "runner": sha256(script),
            "base": sha256(base_path),
            "runtime": sha256(runtime_path),
            **{str(path.resolve()): sha256(path.resolve()) for path in args.evaluation},
        },
        "claim_boundary": "selected finite Air/Metro column-pipeline pairs only; does not repair the known general PV2F int64/frame-integrity audit failure",
    }
    (output / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise AssertionError(audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
