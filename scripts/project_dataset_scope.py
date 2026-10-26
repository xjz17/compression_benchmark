#!/usr/bin/env python3
"""Load and validate the current project-wide dataset scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPE_PATH = (
    ROOT / "benchmark/reference/datasets/project_dataset_scope_20260928.json"
)
EXPECTED_SCOPE_SHA256 = (
    "52d3891e8efe653a15d1f1535e7fa922302e6e97cdc797639caa3d289573444f"
)
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_project_dataset_scope(path: Path = DEFAULT_SCOPE_PATH) -> dict[str, Any]:
    scope = json.loads(path.read_text(encoding="utf-8"))
    if scope.get("schema") != "compression-benchmark-project-dataset-scope-v1":
        raise ValueError(f"unexpected project dataset scope schema in {path}")
    core = scope.get("core_csv_datasets")
    timestamp_rows = scope.get("timestamp_datasets")
    if not isinstance(core, list) or not isinstance(timestamp_rows, list):
        raise ValueError("project dataset scope lists are missing")
    timestamp = [str(row.get("id", "")) for row in timestamp_rows]
    if any(not value for value in [*core, *timestamp]):
        raise ValueError("project dataset scope contains an empty dataset id")
    if len(core) != len(set(core)) or len(timestamp) != len(set(timestamp)):
        raise ValueError("project dataset scope contains duplicate dataset ids")
    overlap = sorted(set(core) & set(timestamp))
    if overlap:
        raise ValueError(f"core/timestamp scope overlap: {overlap}")
    if len(core) != int(scope.get("core_csv_dataset_count", -1)):
        raise ValueError("core CSV dataset count mismatch")
    if len(timestamp) != int(scope.get("timestamp_dataset_count", -1)):
        raise ValueError("timestamp dataset count mismatch")
    if len(core) + len(timestamp) != int(scope.get("total_dataset_count", -1)):
        raise ValueError("total project dataset count mismatch")
    for row, dataset in zip(timestamp_rows, timestamp, strict=True):
        if row.get("repo_id") != f"BenchmarkHost/{dataset}":
            raise ValueError(f"unexpected BenchmarkHost repository for {dataset}")
        if not REVISION_RE.fullmatch(str(row.get("revision", ""))):
            raise ValueError(f"invalid frozen revision for {dataset}")
    retired = set(map(str, scope.get("retired_datasets", [])))
    if retired & (set(core) | set(timestamp)):
        raise ValueError("retired dataset is still active in project scope")
    return scope


if sha256_file(DEFAULT_SCOPE_PATH) != EXPECTED_SCOPE_SHA256:
    raise ValueError("default project dataset scope hash does not match its frozen version")
PROJECT_DATASET_SCOPE = load_project_dataset_scope()
CORE_CSV_DATASETS = tuple(PROJECT_DATASET_SCOPE["core_csv_datasets"])
TIMESTAMP_DATASETS = tuple(
    row["id"] for row in PROJECT_DATASET_SCOPE["timestamp_datasets"]
)
PROJECT_DATASETS = CORE_CSV_DATASETS + TIMESTAMP_DATASETS
