#!/usr/bin/env python3
"""Load the two smallest current source datasets for typed action searches.

Size ranking uses actual complete CSV files, not the pre-migration catalog
sizes. Numeric inputs retain int64/float64 precision and use the shared
per-column NaN policy. This module performs no source or output writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(DEFAULT_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from myapp.experiment_data_policy import prepare_numeric_column  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_ranking_manifest(repo_root: Path | str) -> list[dict]:
    """Rank all local source CSVs with a corresponding catalog metadata file.

    This is the complete locally available catalog/source intersection; it
    does not imply that every dataset from the public catalog exists locally.
    """
    root = Path(repo_root).resolve()
    entries = []
    for source in (root / "orignial_datasets").glob("*.csv"):
        metadata = root / "backend/dataset" / source.stem / f"{source.stem}_column_info.csv"
        if source.is_file() and metadata.is_file():
            entries.append({
                "dataset": source.stem,
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "metadata_path": str(metadata),
            })
    entries.sort(key=lambda entry: (entry["source_bytes"], entry["dataset"]))
    return [{"rank": rank, **entry} for rank, entry in enumerate(entries, start=1)]


def full_dataset_ranking_manifest(repo_root: Path | str) -> list[dict]:
    """Explicit alias for callers recording the full source-size ranking."""
    return dataset_ranking_manifest(repo_root)


def _load_dataset(entry: dict) -> list[dict]:
    source = Path(entry["source_path"])
    metadata_path = Path(entry["metadata_path"])
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        metadata = {row["column_name"]: row for row in csv.DictReader(handle)}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        if not names or len(names) != len(set(names)):
            raise ValueError(f"missing or duplicate source column names: {source}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty source dataset: {source}")
    if any(None in row for row in rows):
        raise ValueError(f"source row exceeds CSV header width: {source}")
    source_sha256 = _sha256_file(source)
    columns = []
    for source_index, name in enumerate(names):
        info = metadata.get(name)
        if name in {"timestamp", "date_time"}:
            # Source migration materializes these as Unix-epoch milliseconds;
            # their metadata can still say string or omit the combined column.
            logical_type = "integer"
        elif info is None:
            raise ValueError(f"missing column-name metadata: {entry['dataset']}:{name}")
        else:
            logical_type = info["logical_type"].strip().lower()
            if logical_type == "timestamp":
                logical_type = "integer"
        if logical_type not in {"integer", "int", "int64", "long", "float", "float64", "double"}:
            continue
        prepared = prepare_numeric_column((row[name] for row in rows), logical_type)
        if not prepared.count:
            continue
        kind = "int64" if prepared.logical_type == "integer" else "float64"
        max_point = None if kind == "int64" else int(info.get("max_point_number", 0) or 0)
        columns.append({
            "dataset": entry["dataset"],
            "name": name,
            "source_index": source_index,
            "numeric_index": len(columns),
            "kind": kind,
            "raw": prepared.raw,
            "count": prepared.count,
            "max_point": max_point,
            "source_path": str(source),
            "input_sha256": prepared.raw_sha256,
            "source_sha256": source_sha256,
            "nan_removed": prepared.removed_nan_count,
            "input_policy": prepared.manifest()["input_policy"],
            "source_rows": prepared.source_count,
            "raw_bytes": len(prepared.raw),
        })
    if not columns:
        raise ValueError(f"no nonempty numeric columns: {source}")
    return columns


def load_smallest_two(repo_root: Path | str = DEFAULT_REPO_ROOT) -> list[dict]:
    """Return full int64/float64 columns from the two smallest source CSVs.

    The caller owns any training/test row split or prefix sampling. String
    columns remain outside this numerical compression experiment.
    """
    ranking = dataset_ranking_manifest(repo_root)
    if len(ranking) < 2:
        raise ValueError(f"need at least two local catalog/source datasets: {repo_root}")
    return [column for entry in ranking[:2] for column in _load_dataset(entry)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    args = parser.parse_args()
    columns = load_smallest_two(args.repo_root)
    by_dataset = defaultdict(list)
    for column in columns:
        by_dataset[column["dataset"]].append(column)
    ranking = dataset_ranking_manifest(args.repo_root)
    for entry in ranking[:2]:
        selected = by_dataset[entry["dataset"]]
        print(json.dumps({
            "dataset": entry["dataset"],
            "source_bytes": entry["source_bytes"],
            "rows": selected[0]["source_rows"],
            "numeric_columns": len(selected),
            "raw_bytes": sum(column["raw_bytes"] for column in selected),
            "nan_removed": sum(column["nan_removed"] for column in selected),
            "source_sha256": selected[0]["source_sha256"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
