#!/usr/bin/env python3
"""Independently validate the capped THULab wide-CSV export.

This validator intentionally does not import the CSV exporter.  It validates
the published tree, frozen input bindings, numeric CSV grammar, exact series
mapping, byte cap, timestamp ordering, and the frozen 34/123/3 policy split.

An optional pilot source rescan is deliberately narrow.  It supports only
small, non-adapter table-model datasets whose primary timestamp is physical
``Time`` and whose emitted values are native numeric FIELDs.  Unsupported
sources fail closed rather than silently receiving a weaker audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import multiprocessing
import os
import operator
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


COLLECTION_SCHEMA = "project179-thulab-numeric-wide-csv-collection-v1"
RECEIPT_SCHEMA = "project179-thulab-numeric-wide-csv-receipt-v1"
SCOPE_SCHEMA = "web-compression-project-dataset-scope-v1"
POLICY_SCHEMA = "project179-thulab-utc-policy-v1"
SOURCE_MANIFEST_SCHEMA = "thulab-timestamp-scope-download-v1"
SOURCE_RECEIPT_SCHEMA = "thulab-timestamp-dataset-download-receipt-v1"
ROW_EXCEPTIONS_SCHEMA = "project179-thulab-csv-row-exceptions-v1"
REQUIRED_TSFILE_VERSION = "2.1.5"
ROW_COVERAGE_QUERY = (
    "all_schema_TAG_FIELD_ATTRIBUTE_columns_physical_Time_implicit_"
    "except_frozen_Africa9_string_fields"
)
NULL_TAG_ENCODING = "TYPE:null"
POSITIVE_INFINITY_ENCODING = "inf"
NAN_ENCODING = "nan"
MAX_DATASET_BYTES = 1_000_000_000
PRODUCTION_DECISION_COUNTS = {
    "admit_exact_utc": 34,
    "admit_stored_epoch_coordinate": 123,
    "admit_dataset_adapter": 3,
    "block": 0,
}
ALLOWED_DECISIONS = frozenset(PRODUCTION_DECISION_COUNTS)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
INT_RE = re.compile(r"-?(?:0|[1-9][0-9]*)")
UINT_RE = re.compile(r"(?:0|[1-9][0-9]*)")
NUMBER_RE = re.compile(
    r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
NUMERIC_TYPES = frozenset({"INT32", "INT64", "FLOAT", "DOUBLE"})

# The pinned 2.1.5 reader raises ColumnNotExistError when any of these nine
# Unicode identifiers is projected.  The exception is intentionally tied to
# the two immutable source objects below.  Metadata still has to expose every
# omitted name as an ordinary STRING FIELD, all candidate numeric columns are
# independently rebuilt from the complete schema, and no other omission is
# accepted.
AFRICA_DATASET_ID = "africa_senegal_mauritania_crops_spi"
AFRICA_COVERAGE_EXCEPTION_ID = "africa9-ordinary-string-sdk-column-not-found-v1"
AFRICA_OMITTED_QUERY_COLUMNS = (
    "ndvi_categorías",
    "sipi_categorías",
    "arvi_categorías",
    "evi_categorías",
    "spi_1_categorías",
    "spi_3_categorías",
    "spi_6_categorías",
    "spi_9_categorías",
    "spi_12_categorías",
)
AFRICA_FROZEN_SOURCES = {
    "test/test.tsfile": {
        "sha256": "8a42c37fb5c9ea9359e31b5655908cbb6e6c8ef74d0e1c8799564632389bdd3c",
        "rows": 5_415,
        "positive_infinity_cells": 12,
        "table_name": "africa_senegal_mauritania_crops_spi_test",
    },
    "train/train.tsfile": {
        "sha256": "7d4c84fed1e3ad52af66fe0b2d3e76e34bc17adca5cf4446e39f5433e087abca",
        "rows": 21_660,
        "positive_infinity_cells": 46,
        "table_name": "africa_senegal_mauritania_crops_spi_train",
    },
}
AFRICA_POSITIVE_INFINITY_SOURCE_NAME = "z_gpp"
AFRICA_POSITIVE_INFINITY_TOTAL = 58


def _africa_omitted_schema_rows() -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "tsfile_type": "STRING",
            "category": "FIELD",
            "decision": "drop_non_numeric_field",
        }
        for name in AFRICA_OMITTED_QUERY_COLUMNS
    ]

TRENDING_DATASET_ID = "trending_stocks_yahoo_finance"
TRENDING_SOURCE_PATH = "trending_stocks_yahoo_finance.tsfile"
TRENDING_SOURCE_SHA256 = (
    "1588e8c345bab4fc511dfb7daa9acbf72727b90e749bee766afbd9c25583a90d"
)
TRENDING_SOURCE_ROWS = 16_376
TRENDING_NULL_TAG_ROWS = 2

MACROECON_DATASET_ID = "macroecon"
MACROECON_ROW_EXCEPTION_ID = "macroecon-eight-all-null-int64-min-first-rows-v1"
MACROECON_SOURCE_FILE_COUNT = 64
MACROECON_CANDIDATE_VALUE_COLUMN_COUNT = 5_548
MACROECON_SOURCE_ROWS = 1_284_546
MACROECON_ELIGIBLE_ROWS = 1_284_538
MACROECON_DROPPED_ROWS = 8
MACROECON_AFFECTED_SOURCE_COUNT = 8

BITCOIN_DATASET_ID = "bitcoin"
BITCOIN_SOURCE_PATH = "bitcoin.tsfile"
BITCOIN_SOURCE_SHA256 = (
    "1b82eb13d27dd9c389a533a0a24a0f98f00f5e63a6387a8581e791416e6bb88a"
)
BITCOIN_SOURCE_ROWS = 82_443
BITCOIN_TRUE_NAN_CELLS = 8_073
BITCOIN_NULL_FLOAT_CELLS = 0

DMI_DATASET_ID = "dmi-aarhus-weather-data"
DMI_SOURCE_BOUND_CONVERSION = (
    "policy_source_bound_integer_epoch_to_int64_us"
)
DMI_REFERENCE_TIME_OUTPUT_NAME = "reference_time__timestamp_unix_us"
DMI_SOURCE_ROWS = 43_910
DMI_FROZEN_SOURCES = {
    "data/data.tsfile": {
        "sha256": "d40d9fdca20f018eb48676d9730033c439ab99e1a821d7b94cce01876b3a1ff6",
        "bytes": 64_184,
        "table_name": "dmi_aarhus_data",
        "schema_sha256": "a06fde8814d9018899ce0bc948c238096cc95d80089731810192dd32bd7ca878",
        "source_type": "INT64",
        "source_category": "FIELD",
        "rows": 2_382,
        "reference_epoch_s_min": 1_764_370_800,
        "reference_epoch_s_max": 1_772_773_200,
        "time_epoch_s_min": 1_764_374_400,
        "time_epoch_s_max": 1_772_946_000,
        "conversion": DMI_SOURCE_BOUND_CONVERSION,
    },
    "data/training-matrix.tsfile": {
        "sha256": "9938b0f54e1629e2dadbc49780641a3c2c8f589c1dcd59b8526be0a816082ae4",
        "bytes": 13_245_662,
        "table_name": "dmi_aarhus_training_matrix",
        "schema_sha256": "9aeba98632a965f55b2d440387d74ddfb6ef9423cc7ca524b0c960b43ca12cd1",
        "source_type": "STRING",
        "source_category": "TAG",
        "rows": 41_528,
        "reference_epoch_s_min": 1_768_950_000,
        "reference_epoch_s_max": 1_781_053_200,
        "time_epoch_s_min": 1_768_953_600,
        "time_epoch_s_max": 1_781_060_400,
        "conversion": "policy_temporal_string_to_int64_us",
    },
}

FORBUSH_DATASET_ID = "forbush_decreases"
FORBUSH_SOURCE_PATH = "forbush_decreases.tsfile"
FORBUSH_SOURCE_SHA256 = (
    "35f8df03af8e5fb1c46a6a69f39baad746cfb9575b613a2c91497a8e47b14f46"
)
FORBUSH_TABLE_NAME = "forbush_decreases"
FORBUSH_SCHEMA_SHA256 = (
    "8c32538e20c5df65737767e6c8695fbd9a3c880f092c792c77c2752ee198d6f1"
)
FORBUSH_SOURCE_ROWS = 8_726
FORBUSH_ALL_NULL_CELLS = 78_024
FORBUSH_CANDIDATE_VALUE_COLUMNS = 23
FORBUSH_IDENTITY_NUMERIC_COLUMNS = 21
FORBUSH_DOUBLE_COLUMNS = 20
FORBUSH_TEMPORAL_FIELDS = (
    "cme_datetime_utc",
    "solar_source_datetime_utc",
)
FORBUSH_NATIVE_QUERY_MODE = "rowwise_direct_source_bound"
FORBUSH_TEMPORAL_EMPTY_REPRESENTATION = (
    "csv_empty_source_non_null_empty_distinct_from_null"
)
FORBUSH_TEMPORAL_VALUE_COUNTS = {
    "cme_datetime_utc": {
        "source_non_null_empty_count": 1_629,
        "source_null_count": 6_441,
        "source_valid_count": 656,
    },
    "solar_source_datetime_utc": {
        "source_non_null_empty_count": 1_284,
        "source_null_count": 6_441,
        "source_valid_count": 1_001,
    },
}


class CsvValidationError(RuntimeError):
    """A fail-closed validation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CsvValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    _regular_file(path, context)
    try:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise CsvValidationError(f"duplicate JSON key in {context}: {key}")
                result[key] = item
            return result

        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CsvValidationError(f"cannot read {context}: {error}") from error
    _require(isinstance(value, Mapping), f"{context} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{context} keys differ: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}",
    )


def _row_exception_policy(
    path: Path,
    *,
    scope_sha256: str,
    timestamp_policy_sha256: str,
) -> tuple[Mapping[str, Any], str, dict[str, Mapping[str, Any]]]:
    """Load the one authorized row exception and reject any policy widening."""
    value = _read_json(path, "row-exception policy")
    _exact_keys(
        value,
        {
            "schema", "exception_id", "authorization", "scope_sha256",
            "timestamp_policy_file_sha256", "dataset_id",
            "source_receipt_sha256", "expected_source_file_count",
            "expected_candidate_value_column_count", "expected_source_rows",
            "expected_eligible_rows", "expected_dropped_rows", "predicate", "sources",
        },
        "row-exception policy",
    )
    _require(
        value.get("schema") == ROW_EXCEPTIONS_SCHEMA
        and value.get("exception_id") == MACROECON_ROW_EXCEPTION_ID
        and value.get("dataset_id") == MACROECON_DATASET_ID,
        "unsupported row-exception policy identity",
    )
    _require(
        value.get("scope_sha256") == scope_sha256
        and value.get("timestamp_policy_file_sha256") == timestamp_policy_sha256,
        "row-exception policy is not bound to scope and timestamp policy",
    )
    _require(
        value.get("expected_source_file_count") == MACROECON_SOURCE_FILE_COUNT
        and value.get("expected_candidate_value_column_count")
        == MACROECON_CANDIDATE_VALUE_COLUMN_COUNT
        and value.get("expected_source_rows") == MACROECON_SOURCE_ROWS
        and value.get("expected_eligible_rows") == MACROECON_ELIGIBLE_ROWS
        and value.get("expected_dropped_rows") == MACROECON_DROPPED_ROWS,
        "row-exception policy aggregate counts changed",
    )
    source_receipt_sha = value.get("source_receipt_sha256")
    _require(
        isinstance(source_receipt_sha, str)
        and SHA256_RE.fullmatch(source_receipt_sha) is not None,
        "row-exception source receipt hash is invalid",
    )
    predicate = value.get("predicate")
    _require(isinstance(predicate, Mapping), "row-exception predicate must be an object")
    _exact_keys(
        predicate,
        {
            "physical_time", "source_row_ordinal", "schema_column_category",
            "schema_column_type", "require_all_value_fields_null",
            "require_no_tag_or_attribute_columns",
        },
        "row-exception predicate",
    )
    _require(
        predicate
        == {
            "physical_time": INT64_MIN,
            "source_row_ordinal": 0,
            "schema_column_category": "FIELD",
            "schema_column_type": "FLOAT",
            "require_all_value_fields_null": True,
            "require_no_tag_or_attribute_columns": True,
        },
        "row-exception predicate changed",
    )
    raw_sources = value.get("sources")
    _require(
        isinstance(raw_sources, list)
        and len(raw_sources) == MACROECON_AFFECTED_SOURCE_COUNT,
        "row-exception affected-source count changed",
    )
    sources: dict[str, Mapping[str, Any]] = {}
    expected_source_keys = {
        "path", "table_name", "sha256", "expected_source_rows",
        "expected_eligible_rows", "expected_dropped_rows",
    }
    for index, item in enumerate(raw_sources):
        _require(isinstance(item, Mapping), f"row-exception source {index} is invalid")
        _exact_keys(item, expected_source_keys, f"row-exception source {index}")
        relative = _safe_relative(item.get("path"), f"row-exception source {index}")
        table_name = item.get("table_name")
        digest = item.get("sha256")
        source_rows = item.get("expected_source_rows")
        eligible_rows = item.get("expected_eligible_rows")
        dropped_rows = item.get("expected_dropped_rows")
        _require(
            relative.endswith(".tsfile")
            and relative not in sources
            and isinstance(table_name, str)
            and table_name == PurePosixPath(relative).stem
            and isinstance(digest, str)
            and SHA256_RE.fullmatch(digest) is not None
            and isinstance(source_rows, int)
            and not isinstance(source_rows, bool)
            and isinstance(eligible_rows, int)
            and not isinstance(eligible_rows, bool)
            and dropped_rows == 1
            and source_rows == eligible_rows + 1,
            f"invalid row-exception source binding: {relative}",
        )
        sources[relative] = item
    _require(
        list(sources) == sorted(sources),
        "row-exception affected sources are not in canonical path order",
    )
    return value, _sha256(path), sources


def _real_directory(path: Path, context: str) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise CsvValidationError(f"missing {context}: {path}") from error
    _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
             f"{context} must be a real directory: {path}")
    return path.resolve(strict=True)


def _regular_file(path: Path, context: str) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise CsvValidationError(f"missing {context}: {path}") from error
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode),
             f"{context} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _safe_relative(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{context} must be non-empty")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and "\\" not in value
        and path.as_posix() == value,
        f"unsafe relative path in {context}: {value!r}",
    )
    return value


def _safe_dataset_id(value: Any, context: str) -> str:
    dataset_id = _safe_relative(value, context)
    _require("/" not in dataset_id and dataset_id not in {"_receipts", "CSV_EXPORT_MANIFEST.json"},
             f"unsafe dataset id in {context}: {dataset_id!r}")
    return dataset_id


def _walk_no_links(root: Path) -> tuple[set[str], set[str]]:
    """Return relative directories/files while rejecting links and special files."""
    canonical = _real_directory(root, "export root")
    directories: set[str] = set()
    files: set[str] = set()
    pending = [canonical]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(canonical).as_posix()
            mode = os.lstat(child).st_mode
            _require(not stat.S_ISLNK(mode), f"output tree contains symlink: {relative}")
            _require(
                not child.name.startswith(".")
                and ".tmp" not in child.name
                and "candidate" not in child.name
                and "spool" not in child.name,
                f"output tree contains temporary path: {relative}",
            )
            if stat.S_ISDIR(mode):
                directories.add(relative)
                pending.append(child)
            elif stat.S_ISREG(mode):
                files.add(relative)
            else:
                raise CsvValidationError(f"output tree contains special file: {relative}")
    return directories, files


def _scope_rows(scope: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require(scope.get("schema") == SCOPE_SCHEMA, "unsupported scope schema")
    rows = scope.get("timestamp_datasets")
    _require(isinstance(rows, list) and bool(rows), "scope dataset list is invalid")
    _require(scope.get("timestamp_dataset_count") == len(rows), "scope dataset count mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"scope dataset {index} is not an object")
        dataset_id = _safe_dataset_id(row.get("id"), f"scope dataset {index}")
        _require(dataset_id not in result, f"duplicate scope dataset: {dataset_id}")
        _require(row.get("repo_id") == f"THULab/{dataset_id}",
                 f"scope repo identity mismatch for {dataset_id}")
        revision = row.get("revision")
        _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                 f"invalid scope revision for {dataset_id}")
        result[dataset_id] = row
    return result


def _policy_rows(
    policy: Mapping[str, Any],
    *,
    scope_id: str,
    scope_sha256: str,
    scope_rows: Mapping[str, Mapping[str, Any]],
    require_production_counts: bool,
) -> dict[str, Mapping[str, Any]]:
    _require(policy.get("schema") == POLICY_SCHEMA, "unsupported timestamp policy schema")
    _require(policy.get("scope_id") == scope_id and policy.get("scope_sha256") == scope_sha256,
             "timestamp policy is not bound to the supplied scope")
    values = policy.get("datasets")
    _require(isinstance(values, list), "timestamp policy datasets must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    counts = {name: 0 for name in ALLOWED_DECISIONS}
    for index, row in enumerate(values):
        _require(isinstance(row, Mapping), f"policy dataset {index} is not an object")
        dataset_id = _safe_dataset_id(row.get("dataset_id"), f"policy dataset {index}")
        _require(dataset_id in scope_rows and dataset_id not in result,
                 f"unknown or duplicate policy dataset: {dataset_id}")
        scope_row = scope_rows[dataset_id]
        _require(
            row.get("repo_id") == scope_row.get("repo_id")
            and row.get("revision") == scope_row.get("revision"),
            f"policy identity mismatch for {dataset_id}",
        )
        decision = row.get("decision")
        _require(decision in ALLOWED_DECISIONS, f"unsupported policy decision for {dataset_id}")
        counts[str(decision)] += 1
        result[dataset_id] = row
    _require(set(result) == set(scope_rows), "policy dataset set differs from frozen scope")
    declared = policy.get("decision_counts")
    if declared is not None:
        _require(declared == counts, "timestamp policy decision_counts do not recompute")
    if require_production_counts:
        _require(len(scope_rows) == 160, "production THULab scope must contain 160 datasets")
        _require(counts == PRODUCTION_DECISION_COUNTS,
                 f"production timestamp decision split changed: {counts}")
    return result


def _source_inventory(dataset_dir: Path) -> list[dict[str, Any]]:
    root = _real_directory(dataset_dir, "source dataset directory")
    rows: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root)
            mode = os.lstat(child).st_mode
            _require(not stat.S_ISLNK(mode),
                     f"source dataset contains symlink: {relative.as_posix()}")
            if stat.S_ISDIR(mode):
                if ".cache" not in relative.parts:
                    pending.append(child)
                continue
            _require(stat.S_ISREG(mode),
                     f"source dataset contains special file: {relative.as_posix()}")
            if ".cache" in relative.parts:
                continue
            relative_text = relative.as_posix()
            if child.suffix == ".tsfile" or relative_text in {"README.md", ".gitattributes"}:
                rows.append({
                    "path": relative_text,
                    "bytes": child.stat().st_size,
                    "sha256": _sha256(child),
                })
    return sorted(rows, key=lambda row: str(row["path"]))


def _validate_source_bindings(
    mirror: Path,
    *,
    scope: Mapping[str, Any],
    scope_sha256: str,
    scope_rows: Mapping[str, Mapping[str, Any]],
    exported_ids: Sequence[str],
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    mirror_root = _real_directory(mirror, "source mirror")
    top = _read_json(mirror_root / "DOWNLOAD_MANIFEST.json", "source mirror manifest")
    _require(top.get("schema") == SOURCE_MANIFEST_SCHEMA and top.get("status") == "complete",
             "source mirror manifest is not complete")
    _require(top.get("scope_id") == scope.get("scope_id") and top.get("scope_sha256") == scope_sha256,
             "source mirror manifest is not bound to scope")
    top_rows = top.get("datasets")
    _require(isinstance(top_rows, list) and len(top_rows) == len(scope_rows),
             "source mirror dataset inventory count mismatch")
    top_by_id = {
        str(row.get("dataset_id")): row
        for row in top_rows if isinstance(row, Mapping)
    }
    _require(set(top_by_id) == set(scope_rows), "source mirror dataset set differs from scope")
    source_receipts_root = _real_directory(
        mirror_root / "_receipts", "source receipt directory"
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for dataset_id in exported_ids:
        scope_row = scope_rows[dataset_id]
        top_row = top_by_id[dataset_id]
        _require(
            top_row.get("repo_id") == scope_row.get("repo_id")
            and top_row.get("revision") == scope_row.get("revision"),
            f"source mirror identity mismatch for {dataset_id}",
        )
        source_receipt_path = source_receipts_root / f"{dataset_id}.json"
        source_receipt = _read_json(source_receipt_path, f"source receipt for {dataset_id}")
        _require(
            source_receipt.get("schema") == SOURCE_RECEIPT_SCHEMA
            and source_receipt.get("status") == "complete"
            and source_receipt.get("dataset_id") == dataset_id
            and source_receipt.get("repo_id") == scope_row.get("repo_id")
            and source_receipt.get("revision") == scope_row.get("revision"),
            f"invalid source receipt for {dataset_id}",
        )
        files = source_receipt.get("files")
        _require(isinstance(files, list) and bool(files),
                 f"source receipt has no files for {dataset_id}")
        expected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(files):
            _require(isinstance(item, Mapping), f"invalid source file row {dataset_id}/{index}")
            relative = _safe_relative(item.get("path"), f"source file {dataset_id}/{index}")
            byte_count = item.get("bytes")
            digest = item.get("sha256")
            _require(
                relative not in seen
                and isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count >= 0
                and isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
                f"invalid source file binding for {dataset_id}/{relative}",
            )
            seen.add(relative)
            expected.append({"path": relative, "bytes": byte_count, "sha256": digest})
        expected.sort(key=lambda row: str(row["path"]))
        actual = _source_inventory(mirror_root / dataset_id)
        _require(actual == expected, f"source file inventory/hash mismatch for {dataset_id}")
        tsfiles = [row for row in expected if str(row["path"]).endswith(".tsfile")]
        _require(bool(tsfiles), f"source receipt has no TsFiles for {dataset_id}")
        output_receipt = receipts[dataset_id]
        _require(output_receipt.get("source_receipt_sha256") == _sha256(source_receipt_path),
                 f"output receipt is not bound to source receipt for {dataset_id}")
        _require(output_receipt.get("source_tsfiles") == tsfiles,
                 f"output receipt TsFile binding differs from source for {dataset_id}")
        scans = output_receipt.get("source_scans")
        _require(isinstance(scans, list), f"output receipt source_scans missing for {dataset_id}")
        scan_paths = [
            str(item.get("source_path")) for item in scans if isinstance(item, Mapping)
        ]
        _require(
            len(scan_paths) == len(scans)
            and scan_paths == [str(item["path"]) for item in tsfiles]
            and len(set(scan_paths)) == len(scan_paths),
            f"output source_scans do not exactly follow source TsFiles for {dataset_id}",
        )
        source_by_path = {str(item["path"]): item for item in tsfiles}
        scan_counts = [0, 0, 0]
        null_tag_counts = [0, 0]
        positive_infinity_cells = 0
        source_nan_cells = 0
        source_nan_by_output: Counter[str] = Counter()
        source_nan_dtype: dict[str, str] = {}
        for scan in scans:
            assert isinstance(scan, Mapping)
            _require(scan.get("source_action") in {"materialize", "exclude_entire_source_file"}
                     and isinstance(scan.get("table_name"), str)
                     and bool(scan.get("table_name")),
                     f"invalid source scan contract for {dataset_id}")
            counts = [scan.get("source_rows"), scan.get("eligible_rows"), scan.get("filtered_rows")]
            _require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                         for value in counts),
                     f"invalid source scan counts for {dataset_id}")
            _require(counts[0] == counts[1] + counts[2],
                     f"source scan row counts do not close for {dataset_id}")
            for index, value in enumerate(counts):
                scan_counts[index] += int(value)
            null_tag_rows = scan.get("null_tag_rows")
            null_tag_cells = scan.get("null_tag_cells")
            _require(
                isinstance(null_tag_rows, int)
                and not isinstance(null_tag_rows, bool)
                and 0 <= null_tag_rows <= counts[0]
                and isinstance(null_tag_cells, int)
                and not isinstance(null_tag_cells, bool)
                and null_tag_cells >= null_tag_rows,
                f"invalid NULL TAG counts for {dataset_id}/{scan.get('source_path')}",
            )
            null_tag_counts[0] += null_tag_rows
            null_tag_counts[1] += null_tag_cells
            scan_positive_infinity = scan.get("positive_infinity_cells")
            positive_by_column = scan.get("positive_infinity_by_column")
            _require(
                isinstance(scan_positive_infinity, int)
                and not isinstance(scan_positive_infinity, bool)
                and scan_positive_infinity >= 0
                and isinstance(positive_by_column, list),
                f"invalid positive-infinity evidence for {dataset_id}/{scan.get('source_path')}",
            )
            seen_infinity_columns: list[str] = []
            recomputed_infinity = 0
            for item in positive_by_column:
                _require(
                    isinstance(item, Mapping)
                    and isinstance(item.get("source_name"), str)
                    and re.fullmatch(r"c[0-9]{4,}", str(item.get("output_column")))
                    is not None
                    and item.get("source_type") in {"FLOAT", "DOUBLE"}
                    and isinstance(item.get("count"), int)
                    and not isinstance(item.get("count"), bool)
                    and item.get("count") > 0,
                    f"invalid positive-infinity column evidence for {dataset_id}",
                )
                seen_infinity_columns.append(str(item["output_column"]))
                recomputed_infinity += int(item["count"])
            _require(
                seen_infinity_columns == sorted(set(seen_infinity_columns))
                and recomputed_infinity == scan_positive_infinity,
                f"positive-infinity column totals/order differ for {dataset_id}",
            )
            positive_infinity_cells += scan_positive_infinity
            scan_nan_cells = scan.get("nan_cells")
            nan_by_column = scan.get("nan_by_column")
            _require(
                isinstance(scan_nan_cells, int)
                and not isinstance(scan_nan_cells, bool)
                and scan_nan_cells >= 0
                and isinstance(nan_by_column, list),
                f"invalid NaN evidence for {dataset_id}/{scan.get('source_path')}",
            )
            seen_nan_columns: list[str] = []
            recomputed_nan = 0
            for item in nan_by_column:
                _require(
                    isinstance(item, Mapping)
                    and isinstance(item.get("source_name"), str)
                    and bool(item.get("source_name"))
                    and item.get("source_type") in {"FLOAT", "DOUBLE"}
                    and re.fullmatch(
                        r"c[0-9]{4,}", str(item.get("output_column"))
                    ) is not None
                    and isinstance(item.get("count"), int)
                    and not isinstance(item.get("count"), bool)
                    and item.get("count") > 0,
                    f"invalid NaN column evidence for {dataset_id}",
                )
                seen_nan_columns.append(str(item["output_column"]))
                recomputed_nan += int(item["count"])
            _require(
                seen_nan_columns == sorted(set(seen_nan_columns))
                and recomputed_nan == scan_nan_cells,
                f"NaN column totals/order differ for {dataset_id}",
            )
            source_nan_cells += scan_nan_cells
            for item in nan_by_column:
                output_column = str(item["output_column"])
                source_nan_by_output[output_column] += int(item["count"])
                prior_dtype = source_nan_dtype.setdefault(
                    output_column, str(item["source_type"])
                )
                _require(
                    prior_dtype == item["source_type"],
                    f"NaN output dtype conflict for {dataset_id}/{output_column}",
                )
            relative = str(scan.get("source_path"))
            source_binding = source_by_path[relative]
            _require(
                scan.get("source_sha256") == source_binding.get("sha256"),
                f"source scan SHA-256 binding differs for {dataset_id}/{relative}",
            )
            if dataset_id == AFRICA_DATASET_ID:
                proof = AFRICA_FROZEN_SOURCES.get(relative)
                _require(proof is not None, f"unapproved Africa source path: {relative}")
                exception = scan.get("row_coverage_exception")
                _require(
                    source_binding.get("sha256") == proof["sha256"]
                    and scan.get("table_name") == proof["table_name"]
                    and counts[0] == counts[1] == proof["rows"]
                    and counts[2] == 0
                    and scan_positive_infinity == proof["positive_infinity_cells"]
                    and positive_by_column
                    == [{
                        "source_name": AFRICA_POSITIVE_INFINITY_SOURCE_NAME,
                        "output_column": positive_by_column[0].get("output_column")
                        if positive_by_column else None,
                        "source_type": "DOUBLE",
                        "count": proof["positive_infinity_cells"],
                    }]
                    and isinstance(exception, Mapping)
                    and exception.get("exception_id") == AFRICA_COVERAGE_EXCEPTION_ID
                    and exception.get("source_path") == relative
                    and exception.get("source_sha256") == proof["sha256"]
                    and exception.get("expected_source_rows") == proof["rows"]
                    and exception.get("verified_source_rows") == proof["rows"]
                    and exception.get("omitted_query_columns")
                    == list(AFRICA_OMITTED_QUERY_COLUMNS)
                    and exception.get("omitted_query_column_schema")
                    == _africa_omitted_schema_rows()
                    and exception.get("omitted_column_contract")
                    == "ordinary_STRING_FIELD_drop_non_numeric_field"
                    and exception.get("row_coverage_proof")
                    == "frozen_source_sha256_and_exact_source_row_count",
                    f"Africa row-coverage exception is not exactly frozen for {relative}",
                )
            else:
                _require(
                    scan.get("row_coverage_exception") is None,
                    f"unapproved row-coverage exception for {dataset_id}/{relative}",
                )
                _require(
                    scan_positive_infinity == 0 and positive_by_column == [],
                    f"positive infinity is not approved for {dataset_id}/{relative}",
                )
        _require(
            scan_counts[0] == output_receipt.get("source_rows")
            and scan_counts[1] == _fixed_uint20(
                output_receipt.get("eligible_rows_decimal_20"),
                f"{dataset_id} eligible rows",
            )
            and scan_counts[2] == output_receipt.get("policy_filtered_rows"),
            f"source scan totals differ from output receipt for {dataset_id}",
        )
        receipt_null_rows = output_receipt.get("null_tag_rows")
        receipt_null_cells = output_receipt.get("null_tag_cells")
        _require(
            isinstance(receipt_null_rows, int)
            and not isinstance(receipt_null_rows, bool)
            and isinstance(receipt_null_cells, int)
            and not isinstance(receipt_null_cells, bool)
            and receipt_null_rows == null_tag_counts[0]
            and receipt_null_cells == null_tag_counts[1],
            f"dataset NULL TAG totals differ from source scans for {dataset_id}",
        )
        receipt_positive_infinity = output_receipt.get("positive_infinity_cells")
        _require(
            isinstance(receipt_positive_infinity, int)
            and not isinstance(receipt_positive_infinity, bool)
            and receipt_positive_infinity == positive_infinity_cells,
            f"dataset positive-infinity total differs from source scans for {dataset_id}",
        )
        _require(
            output_receipt.get("source_nan_cells") == source_nan_cells,
            f"dataset source NaN total differs from source scans for {dataset_id}",
        )
        expected_source_nan_by_column = [
            {
                "output_column": column,
                "output_dtype": {
                    "FLOAT": "float32", "DOUBLE": "float64"
                }[source_nan_dtype[column]],
                "count": count,
            }
            for column, count in sorted(source_nan_by_output.items())
        ]
        _require(
            output_receipt.get("source_nan_by_column")
            == expected_source_nan_by_column,
            f"dataset source NaN column totals differ from source scans for {dataset_id}",
        )
        if dataset_id == AFRICA_DATASET_ID:
            _require(
                positive_infinity_cells == AFRICA_POSITIVE_INFINITY_TOTAL,
                "Africa positive-infinity total is not the approved exact 58 cells",
            )
        if dataset_id == TRENDING_DATASET_ID:
            _require(
                len(tsfiles) == 1
                and tsfiles[0].get("path") == TRENDING_SOURCE_PATH
                and tsfiles[0].get("sha256") == TRENDING_SOURCE_SHA256
                and scan_counts[0] == scan_counts[1] == TRENDING_SOURCE_ROWS
                and scan_counts[2] == 0
                and null_tag_counts == [TRENDING_NULL_TAG_ROWS, TRENDING_NULL_TAG_ROWS],
                "trending frozen source/row/NULL-TAG contract mismatch",
            )
        result[dataset_id] = tsfiles
    return result


def _validate_row_exception_bindings(
    *,
    manifest: Mapping[str, Any],
    dataset_ids: Sequence[str],
    receipts: Mapping[str, Mapping[str, Any]],
    source_tsfiles: Mapping[str, Sequence[Mapping[str, Any]]],
    exception_policy: Mapping[str, Any] | None,
    exception_policy_sha256: str | None,
    exception_sources: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind collection, receipts, and per-source evidence to the exception file."""
    _require(
        manifest.get("row_exception_policy_file_sha256") == exception_policy_sha256,
        "collection manifest row-exception policy binding mismatch",
    )
    for dataset_id in dataset_ids:
        receipt = receipts[dataset_id]
        _require(
            receipt.get("row_exception_policy_file_sha256") == exception_policy_sha256,
            f"dataset row-exception policy binding mismatch for {dataset_id}",
        )
        root_timestamp_filtered = receipt.get("timestamp_policy_filtered_rows")
        root_exception_filtered = receipt.get("row_exception_filtered_rows")
        root_total_filtered = receipt.get("policy_filtered_rows")
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (
                    root_timestamp_filtered,
                    root_exception_filtered,
                    root_total_filtered,
                )
            )
            and root_timestamp_filtered + root_exception_filtered == root_total_filtered,
            f"dataset filtered-row categories do not close for {dataset_id}",
        )
        scans = receipt.get("source_scans")
        assert isinstance(scans, list)
        observed_timestamp_filtered = 0
        observed_exception_filtered = 0
        for scan in scans:
            assert isinstance(scan, Mapping)
            scan_timestamp_filtered = scan.get("timestamp_policy_filtered_rows")
            scan_exception_filtered = scan.get("row_exception_filtered_rows")
            scan_total_filtered = scan.get("filtered_rows")
            _require(
                all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in (
                        scan_timestamp_filtered,
                        scan_exception_filtered,
                        scan_total_filtered,
                    )
                )
                and scan_timestamp_filtered + scan_exception_filtered
                == scan_total_filtered,
                f"source filtered-row categories do not close for "
                f"{dataset_id}/{scan.get('source_path')}",
            )
            observed_timestamp_filtered += int(scan_timestamp_filtered)
            observed_exception_filtered += int(scan_exception_filtered)
        _require(
            observed_timestamp_filtered == root_timestamp_filtered
            and observed_exception_filtered == root_exception_filtered,
            f"source/dataset filtered-row category totals differ for {dataset_id}",
        )
        if dataset_id != MACROECON_DATASET_ID:
            _require(
                receipt.get("row_exception_id") is None
                and root_exception_filtered == 0
                and all(scan.get("row_exception") is None for scan in scans),
                f"unapproved row exception for {dataset_id}",
            )

    if exception_policy is None:
        _require(
            all(receipts[dataset_id].get("row_exception_id") is None for dataset_id in dataset_ids),
            "row exception requires an explicit --row-exceptions policy",
        )
        return

    # Every subset export is bound to the same frozen exception-policy file,
    # even when the sole authorized dataset is outside that subset (for
    # example the DMI-only preflight).  The policy has already been parsed and
    # bound cryptographically above, and every present non-macroecon receipt
    # has been proved to apply no exception.  There is therefore no macroecon
    # evidence to validate until macroecon is actually included.
    if MACROECON_DATASET_ID not in dataset_ids:
        return
    receipt = receipts[MACROECON_DATASET_ID]
    tsfiles = source_tsfiles[MACROECON_DATASET_ID]
    _require(
        exception_policy.get("source_receipt_sha256")
        == receipt.get("source_receipt_sha256"),
        "row-exception policy source receipt binding mismatch",
    )
    _require(
        len(tsfiles) == MACROECON_SOURCE_FILE_COUNT
        and receipt.get("source_rows") == MACROECON_SOURCE_ROWS
        and _fixed_uint20(
            receipt.get("eligible_rows_decimal_20"), "macroecon eligible rows"
        )
        == MACROECON_ELIGIBLE_ROWS
        and receipt.get("policy_filtered_rows") == MACROECON_DROPPED_ROWS
        and receipt.get("timestamp_policy_filtered_rows") == 0
        and receipt.get("row_exception_id") == MACROECON_ROW_EXCEPTION_ID
        and receipt.get("row_exception_filtered_rows") == MACROECON_DROPPED_ROWS
        and len(receipt.get("candidate_value_columns", []))
        == MACROECON_CANDIDATE_VALUE_COLUMN_COUNT,
        "macroecon row-exception aggregate receipt mismatch",
    )
    source_by_path = {str(item["path"]): item for item in tsfiles}
    _require(
        set(exception_sources) <= set(source_by_path),
        "row-exception policy names a source outside the frozen macroecon receipt",
    )
    scan_by_path = {
        str(item.get("source_path")): item
        for item in receipt["source_scans"]
        if isinstance(item, Mapping)
    }
    observed_affected: list[str] = []
    expected_exception_keys = {
        "exception_id", "source_path", "source_sha256", "table_name",
        "expected_source_rows", "observed_source_rows", "expected_eligible_rows",
        "observed_eligible_rows", "expected_dropped_rows", "observed_dropped_rows",
        "dropped_source_row_ordinals", "physical_time", "all_value_fields_null",
    }
    for relative, scan in scan_by_path.items():
        declared = exception_sources.get(relative)
        evidence = scan.get("row_exception")
        if declared is None:
            _require(
                evidence is None and scan.get("row_exception_filtered_rows") == 0,
                f"unapproved macroecon row exception for {relative}",
            )
            continue
        observed_affected.append(relative)
        _require(isinstance(evidence, Mapping), f"missing row exception for {relative}")
        _exact_keys(evidence, expected_exception_keys, f"row exception for {relative}")
        expected_source_rows = int(declared["expected_source_rows"])
        expected_eligible_rows = int(declared["expected_eligible_rows"])
        expected_dropped_rows = int(declared["expected_dropped_rows"])
        _require(
            source_by_path[relative].get("sha256") == declared.get("sha256")
            and scan.get("table_name") == declared.get("table_name")
            and scan.get("source_rows") == expected_source_rows
            and scan.get("eligible_rows") == expected_eligible_rows
            and scan.get("filtered_rows") == expected_dropped_rows
            and scan.get("timestamp_policy_filtered_rows") == 0
            and scan.get("row_exception_filtered_rows") == expected_dropped_rows
            and evidence
            == {
                "exception_id": MACROECON_ROW_EXCEPTION_ID,
                "source_path": relative,
                "source_sha256": declared["sha256"],
                "table_name": declared["table_name"],
                "expected_source_rows": expected_source_rows,
                "observed_source_rows": expected_source_rows,
                "expected_eligible_rows": expected_eligible_rows,
                "observed_eligible_rows": expected_eligible_rows,
                "expected_dropped_rows": expected_dropped_rows,
                "observed_dropped_rows": expected_dropped_rows,
                "dropped_source_row_ordinals": [0],
                "physical_time": INT64_MIN,
                "all_value_fields_null": True,
            },
            f"macroecon row-exception evidence differs for {relative}",
        )
    _require(
        observed_affected == sorted(exception_sources),
        "macroecon affected-source receipt set/order differs from policy",
    )


def _fixed_uint20(value: Any, context: str) -> int:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9]{20}", value) is not None,
             f"{context} must be a fixed-width uint20 string")
    return int(value)


def _bound_file(dataset_dir: Path, binding: Any, expected_name: str) -> Path:
    _require(isinstance(binding, Mapping), f"missing binding for {expected_name}")
    _require(binding.get("path") == expected_name, f"unexpected path for {expected_name}")
    path = _regular_file(dataset_dir / expected_name, expected_name)
    byte_count = _fixed_uint20(binding.get("bytes_decimal_20"),
                               f"{expected_name} byte binding")
    digest = binding.get("sha256")
    _require(
        byte_count == path.stat().st_size,
        f"byte count mismatch for {path}",
    )
    _require(isinstance(digest, str) and digest == _sha256(path),
             f"SHA-256 mismatch for {path}")
    return path


def _canonical_int(text: str, *, unsigned: bool, context: str) -> int:
    expression = UINT_RE if unsigned else INT_RE
    _require(expression.fullmatch(text) is not None, f"non-canonical integer in {context}: {text!r}")
    value = int(text)
    if not unsigned:
        _require(INT64_MIN <= value <= INT64_MAX, f"int64 overflow in {context}")
    return value


def _numeric_cell(
    text: str,
    context: str,
    dtype: str | None = None,
    *,
    allow_positive_infinity: bool = False,
    allow_nan: bool = False,
) -> str | None:
    if dtype in {"int32", "int64"}:
        if text == "":
            return None
        value = _canonical_int(text, unsigned=False, context=context)
        if dtype == "int32":
            _require(-(1 << 31) <= value <= (1 << 31) - 1,
                     f"int32 overflow in {context}")
        return None
    if text == "":
        return None
    if text == NAN_ENCODING:
        _require(
            allow_nan and dtype in {"float32", "float64"},
            f"NaN is not approved in {context}",
        )
        decoded = float(text)
        _require(math.isnan(decoded), f"NaN did not round-trip in {context}")
        return NAN_ENCODING
    if text == "inf":
        _require(
            allow_positive_infinity and dtype in {"float32", "float64"},
            f"positive infinity is not approved in {context}",
        )
        decoded = float(text)
        _require(
            math.isinf(decoded) and decoded > 0,
            f"positive infinity did not round-trip in {context}",
        )
        return POSITIVE_INFINITY_ENCODING
    _require(text not in {"-inf", "+inf", "-nan", "+nan", "NaN", "NAN"},
             f"non-finite value is forbidden in {context}")
    _require(NUMBER_RE.fullmatch(text) is not None,
             f"non-numeric CSV cell in {context}: {text!r}")
    try:
        value = float(text)
    except ValueError as error:  # pragma: no cover - regex already excludes it
        raise CsvValidationError(f"invalid numeric CSV cell in {context}: {text!r}") from error
    _require(math.isfinite(value), f"unexpected non-finite spelling in {context}")
    return None


def _validate_data_csv(
    path: Path,
    receipt: Mapping[str, Any],
    column_dtypes: Mapping[str, str],
    allowed_positive_infinity_columns: frozenset[str] = frozenset(),
    allowed_nan_columns: frozenset[str] = frozenset(),
) -> tuple[list[str], set[int], Counter[int], Counter[str], Counter[str], int]:
    raw_start = path.read_bytes()[:3]
    _require(raw_start != b"\xef\xbb\xbf", "data.csv must not contain a UTF-8 BOM")
    used_ids: set[int] = set()
    series_row_counts: Counter[int] = Counter()
    positive_infinity_counts: Counter[str] = Counter()
    nan_counts: Counter[str] = Counter()
    previous_timestamp: int | None = None
    row_count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            _require(
                isinstance(header, list) and len(header) >= 3
                and header[:2] == ["timestamp_unix_us", "series_id"]
                and len(header) == len(set(header))
                and all(header),
                "data.csv header must be timestamp_unix_us,series_id plus unique value columns",
            )
            retained = receipt.get("retained_value_columns")
            _require(isinstance(retained, list) and header[2:] == retained,
                     "data.csv header differs from retained_value_columns")
            dropped = receipt.get("dropped_trailing_value_columns")
            _require(
                isinstance(dropped, list)
                and all(isinstance(value, str) and value for value in dropped)
                and len(set(retained + dropped)) == len(retained) + len(dropped),
                "retained/dropped value-column metadata is invalid",
            )
            for line_number, row in enumerate(reader, start=2):
                _require(len(row) == len(header), f"data.csv row width mismatch at line {line_number}")
                timestamp = _canonical_int(row[0], unsigned=False,
                                           context=f"data.csv line {line_number} timestamp")
                series_id = _canonical_int(row[1], unsigned=True,
                                           context=f"data.csv line {line_number} series_id")
                _require(previous_timestamp is None or timestamp >= previous_timestamp,
                         f"data.csv is not chronological at line {line_number}")
                previous_timestamp = timestamp
                used_ids.add(series_id)
                series_row_counts[series_id] += 1
                for column, value in zip(header[2:], row[2:]):
                    special = _numeric_cell(
                        value,
                        f"data.csv line {line_number} column {column}",
                        column_dtypes.get(column),
                        allow_positive_infinity=(
                            column in allowed_positive_infinity_columns
                        ),
                        allow_nan=(column in allowed_nan_columns),
                    )
                    if special == POSITIVE_INFINITY_ENCODING:
                        positive_infinity_counts[column] += 1
                    elif special == NAN_ENCODING:
                        nan_counts[column] += 1
                row_count += 1
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CsvValidationError(f"cannot parse data.csv: {error}") from error
    expected_rows = _fixed_uint20(
        receipt.get("emitted_rows_decimal_20"), "manifest emitted row count"
    )
    _require(row_count == expected_rows, "data.csv row count differs from receipt")
    return (
        header, used_ids, series_row_counts, positive_infinity_counts, nan_counts,
        row_count,
    )


def _json_list_cell(value: str, context: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise CsvValidationError(f"invalid JSON cell in {context}") from error
    _require(isinstance(parsed, list), f"{context} must contain a JSON list")
    _require(_canonical_json(parsed) == value, f"{context} is not canonical JSON")
    return parsed


def _validate_series_map(
    path: Path,
    receipt: Mapping[str, Any],
    used_ids: set[int],
    *,
    dataset_id: str,
) -> dict[int, dict[str, Any]]:
    expected_header = [
        "series_id", "table_name", "tag_names_json",
        "typed_tag_values_json", "source_paths_json",
    ]
    records: dict[int, dict[str, Any]] = {}
    previous_id: int | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            _require(next(reader, None) == expected_header, "series_map.csv header mismatch")
            for line_number, row in enumerate(reader, start=2):
                _require(len(row) == len(expected_header),
                         f"series_map.csv row width mismatch at line {line_number}")
                series_id = _canonical_int(
                    row[0], unsigned=True,
                    context=f"series_map.csv line {line_number} series_id",
                )
                _require(series_id not in records and (previous_id is None or series_id > previous_id),
                         "series_map.csv IDs must be unique and strictly increasing")
                previous_id = series_id
                _require(bool(row[1]), f"series_map.csv line {line_number} has empty table_name")
                tag_names = _json_list_cell(row[2], f"series_map.csv line {line_number} tag_names")
                typed_tags = _json_list_cell(row[3], f"series_map.csv line {line_number} typed_tags")
                source_paths = _json_list_cell(row[4], f"series_map.csv line {line_number} source_paths")
                _require(all(isinstance(value, str) and value for value in tag_names),
                         "series map tag names must be strings")
                _require(all(isinstance(value, str) and ":" in value for value in typed_tags),
                         "series map typed TAG values are invalid")
                _require(
                    len(tag_names) == len(typed_tags)
                    or (not tag_names and typed_tags == ["SINGLE:__single_entity__"]),
                    "series map TAG name/value arity mismatch",
                )
                for value in typed_tags:
                    if value == "SINGLE:__single_entity__":
                        continue
                    dtype, encoded = value.split(":", 1)
                    _require(bool(dtype), "series map typed TAG has an empty type")
                    try:
                        parsed = json.loads(encoded)
                    except json.JSONDecodeError as error:
                        raise CsvValidationError(
                            f"series map typed TAG is not canonical JSON: {value!r}"
                        ) from error
                    _require(
                        _canonical_json(parsed) == encoded,
                        f"series map typed TAG is not canonical JSON: {value!r}",
                    )
                    if parsed is None:
                        _require(
                            dtype in {"STRING", "TEXT"},
                            f"NULL TAG has unsupported type in {dataset_id}: {dtype}",
                        )
                _require(
                    bool(source_paths)
                    and all(isinstance(value, str) for value in source_paths)
                    and source_paths == sorted(set(source_paths)),
                    "series map source_paths must be a non-empty sorted unique list",
                )
                for source_path in source_paths:
                    _safe_relative(source_path, "series map source path")
                records[series_id] = {
                    "table_name": row[1],
                    "tag_names": tag_names,
                    "typed_tags": typed_tags,
                    "source_paths": source_paths,
                }
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CsvValidationError(f"cannot parse series_map.csv: {error}") from error
    _require(set(records) == used_ids,
             "series_map.csv ID set must exactly equal IDs referenced by data.csv")
    return records


def _validate_columns_csv(
    path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    expected_header = [
        "column", "logical_key_json", "table_name", "source_name",
        "source_type", "output_dtype", "conversion",
    ]
    records: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            _require(next(reader, None) == expected_header, "columns.csv header mismatch")
            for line_number, row in enumerate(reader, start=2):
                _require(len(row) == len(expected_header),
                         f"columns.csv row width mismatch at line {line_number}")
                column, logical_key, table, source_name, source_type, dtype, conversion = row
                _require(column == f"c{len(records):04d}" and column not in records,
                         "columns.csv column IDs must be contiguous and ordered")
                _require(table and source_name,
                         f"columns.csv source schema is invalid at line {line_number}")
                if conversion == "identity":
                    _require(source_type in NUMERIC_TYPES,
                             f"identity output is not a numeric source at line {line_number}")
                    expected_dtype = {
                        "INT32": "int32", "INT64": "int64",
                        "FLOAT": "float32", "DOUBLE": "float64",
                    }[source_type]
                else:
                    approved_temporal = (
                        conversion == "policy_temporal_string_to_int64_us"
                        and source_type in {"STRING", "TEXT"}
                    ) or (
                        conversion == DMI_SOURCE_BOUND_CONVERSION
                        and source_type in {"INT32", "INT64"}
                    )
                    _require(
                        approved_temporal,
                        f"unapproved conversion in columns.csv line {line_number}",
                    )
                    expected_dtype = "int64"
                _require(dtype == expected_dtype,
                         f"column output dtype is inconsistent at line {line_number}")
                parsed = _json_list_cell(logical_key, f"columns.csv line {line_number} logical key")
                _require(parsed == [table, source_name, source_type, conversion],
                         f"columns.csv logical key mismatch at line {line_number}")
                records[column] = {
                    "logical_key": logical_key,
                    "table_name": table,
                    "source_name": source_name,
                    "source_type": source_type,
                    "output_dtype": dtype,
                    "conversion": conversion,
                }
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CsvValidationError(f"cannot parse columns.csv: {error}") from error
    retained = receipt.get("retained_value_columns")
    _require(isinstance(retained, list) and list(records) == retained,
             "columns.csv rows differ from retained_value_columns")
    return records


def _validate_column_partition(
    receipt: Mapping[str, Any], dataset_id: str, cap: int
) -> tuple[list[str], list[str], list[str]]:
    """Validate the complete candidate list and its retained-prefix split."""

    candidate = receipt.get("candidate_value_columns")
    retained = receipt.get("retained_value_columns")
    dropped = receipt.get("dropped_trailing_value_columns")
    _require(
        isinstance(candidate, list)
        and isinstance(retained, list)
        and isinstance(dropped, list)
        and bool(retained)
        and all(isinstance(value, str) and value for value in candidate)
        and all(isinstance(value, str) and value for value in retained)
        and all(isinstance(value, str) and value for value in dropped),
        f"candidate/retained/dropped column metadata is invalid for {dataset_id}",
    )
    expected = [f"c{index:04d}" for index in range(len(candidate))]
    _require(candidate == expected and candidate == [*retained, *dropped],
             f"candidate columns are not an exact retained prefix/dropped suffix for {dataset_id}")
    _require(
        _fixed_uint20(
            receipt.get("candidate_value_column_count_decimal_20"),
            f"{dataset_id} candidate value-column count",
        ) == len(candidate)
        and _fixed_uint20(
            receipt.get("retained_value_column_count_decimal_20"),
            f"{dataset_id} retained value-column count",
        ) == len(retained)
        and _fixed_uint20(
            receipt.get("dropped_trailing_value_column_count_decimal_20"),
            f"{dataset_id} dropped value-column count",
        ) == len(dropped),
        f"fixed-width value-column counts do not match arrays for {dataset_id}",
    )
    _require(receipt.get("last_retained_value_column") == retained[-1],
             f"last retained value-column boundary is invalid for {dataset_id}")
    _require(
        receipt.get("first_dropped_trailing_value_column")
        == (dropped[0] if dropped else None),
        f"first dropped value-column boundary is invalid for {dataset_id}",
    )
    next_column_total = _fixed_uint20(
        receipt.get("next_trailing_column_would_total_bytes_decimal_20"),
        f"{dataset_id} next trailing-column total",
    )
    if dropped:
        _require(next_column_total > cap,
                 f"column-prefix maximality witness is invalid for {dataset_id}")
    else:
        _require(next_column_total == 0,
                 f"unexpected column-prefix witness for complete schema {dataset_id}")
    return candidate, retained, dropped


def _validate_receipt_identity(
    receipt: Mapping[str, Any],
    *,
    dataset_id: str,
    scope_row: Mapping[str, Any],
    policy_row: Mapping[str, Any],
    scope: Mapping[str, Any],
    scope_sha256: str,
    policy_sha256: str,
) -> None:
    _require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == "complete",
             f"invalid output receipt for {dataset_id}")
    _require(
        receipt.get("dataset_id") == dataset_id
        and receipt.get("repo_id") == scope_row.get("repo_id")
        and receipt.get("revision") == scope_row.get("revision"),
        f"output receipt identity mismatch for {dataset_id}",
    )
    _require(
        receipt.get("scope_id") == scope.get("scope_id")
        and receipt.get("scope_sha256") == scope_sha256
        and receipt.get("timestamp_policy_file_sha256") == policy_sha256,
        f"output receipt input binding mismatch for {dataset_id}",
    )
    expected_policy_sha = hashlib.sha256(_canonical_json(dict(policy_row)).encode("utf-8")).hexdigest()
    _require(receipt.get("timestamp_policy_sha256") == expected_policy_sha,
             f"dataset policy hash mismatch for {dataset_id}")
    _require(receipt.get("timestamp_policy_decision") == policy_row.get("decision"),
             f"timestamp policy decision mismatch for {dataset_id}")
    _require(receipt.get("tsfile_version") == REQUIRED_TSFILE_VERSION,
             f"TsFile version binding mismatch for {dataset_id}")
    _require(receipt.get("row_coverage_query") == ROW_COVERAGE_QUERY,
             f"row coverage query contract mismatch for {dataset_id}")
    _require(
        receipt.get("null_tag_encoding") == NULL_TAG_ENCODING
        and receipt.get("null_tag_values_preserved") is True,
        f"NULL TAG preservation contract mismatch for {dataset_id}",
    )
    _require(
        receipt.get("positive_infinity_encoding") == POSITIVE_INFINITY_ENCODING
        and receipt.get("positive_infinity_values_preserved") is True,
        f"positive-infinity preservation contract mismatch for {dataset_id}",
    )
    _require(
        receipt.get("nan_encoding") == NAN_ENCODING
        and receipt.get("nan_values_preserved") is True,
        f"NaN preservation contract mismatch for {dataset_id}",
    )
    _require(receipt.get("ordinary_strings_emitted_to_data_csv") is False,
             f"ordinary string emission flag is unsafe for {dataset_id}")
    _require(receipt.get("series_mapping_is_payload") is False,
             f"series mapping payload flag changed for {dataset_id}")
    _require(isinstance(receipt.get("exporter_sha256"), str)
             and SHA256_RE.fullmatch(str(receipt.get("exporter_sha256"))) is not None,
             f"exporter hash binding is missing for {dataset_id}")


def _validate_dataset(
    output: Path,
    dataset_id: str,
    receipt: Mapping[str, Any],
    *,
    cap: int,
    policy_row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, dict[str, str]]]:
    dataset_dir = _real_directory(output / dataset_id, f"dataset directory {dataset_id}")
    entries = {item.name for item in dataset_dir.iterdir()}
    expected_names = {"data.csv", "series_map.csv", "columns.csv", "dataset_manifest.json"}
    _require(entries == expected_names,
             f"dataset output whitelist violation for {dataset_id}: {sorted(entries)}")
    raw_files = receipt.get("files")
    _require(isinstance(raw_files, list) and len(raw_files) == 4,
             f"dataset file manifest is invalid for {dataset_id}")
    _require(
        [item.get("path") for item in raw_files if isinstance(item, Mapping)]
        == ["data.csv", "series_map.csv", "columns.csv", "dataset_manifest.json"],
        f"dataset file manifest order changed for {dataset_id}",
    )
    file_rows = {
        str(item.get("path")): item for item in raw_files if isinstance(item, Mapping)
    }
    _require(set(file_rows) == expected_names and len(file_rows) == len(raw_files),
             f"dataset file manifest whitelist mismatch for {dataset_id}")
    data_path = _bound_file(dataset_dir, file_rows["data.csv"], "data.csv")
    map_path = _bound_file(dataset_dir, file_rows["series_map.csv"], "series_map.csv")
    columns_path = _bound_file(dataset_dir, file_rows["columns.csv"], "columns.csv")
    manifest_path = _regular_file(dataset_dir / "dataset_manifest.json", "dataset manifest")
    manifest_row = file_rows["dataset_manifest.json"]
    _require(_fixed_uint20(manifest_row.get("bytes_decimal_20"), "dataset manifest bytes")
             == manifest_path.stat().st_size
             and manifest_row.get("sha256") is None,
             f"dataset self-binding is invalid for {dataset_id}")
    actual_bytes = sum((dataset_dir / name).stat().st_size for name in expected_names)
    _require(receipt.get("cap_accounting") == "all four files in this dataset directory",
             f"dataset cap accounting contract mismatch for {dataset_id}")
    _require(_fixed_uint20(receipt.get("cap_bytes_decimal_20"), "dataset cap") == cap,
             f"dataset cap mismatch for {dataset_id}")
    _require(_fixed_uint20(receipt.get("dataset_manifest_bytes_decimal_20"),
                           "dataset manifest byte count") == manifest_path.stat().st_size,
             f"dataset manifest byte count mismatch for {dataset_id}")
    _require(_fixed_uint20(receipt.get("actual_total_bytes_decimal_20"),
                           "actual dataset byte count") == actual_bytes,
             f"actual capped byte count mismatch for {dataset_id}")
    _require(actual_bytes <= cap <= MAX_DATASET_BYTES,
             f"dataset byte cap violated for {dataset_id}: {actual_bytes}>{cap}")
    eligible = _fixed_uint20(receipt.get("eligible_rows_decimal_20"), "eligible rows")
    emitted = _fixed_uint20(receipt.get("emitted_rows_decimal_20"), "emitted rows")
    lost = _fixed_uint20(receipt.get("lost_rows_due_to_byte_cap_decimal_20"), "lost rows")
    source_rows = receipt.get("source_rows")
    filtered = receipt.get("policy_filtered_rows")
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (eligible, emitted, lost, source_rows, filtered))
        and emitted <= eligible
        and eligible - emitted == lost
        and source_rows - eligible == filtered,
        f"row accounting does not close for {dataset_id}",
    )
    candidate_columns, retained_columns, dropped_columns = _validate_column_partition(
        receipt, dataset_id, cap
    )
    column_records = _validate_columns_csv(columns_path, receipt)
    _require(list(column_records) == retained_columns,
             f"columns.csv is not the declared retained prefix for {dataset_id}")
    temporal_rules = {
        (str(item.get("source_name")), str(item.get("source_type")))
        for item in policy_row.get("temporal_string_fields", [])
        if isinstance(item, Mapping)
    }
    source_bound_rules = {
        (str(item.get("source_name")), str(item.get("source_type")))
        for item in _source_bound_temporal_rules(policy_row)
    }
    for item in column_records.values():
        if item["conversion"] == "policy_temporal_string_to_int64_us":
            _require((item["source_name"], item["source_type"]) in temporal_rules,
                     f"columns.csv emits an unapproved temporal string for {dataset_id}")
        elif item["conversion"] == DMI_SOURCE_BOUND_CONVERSION:
            _require(
                (item["source_name"], item["source_type"]) in source_bound_rules,
                f"columns.csv emits an unapproved source-bound temporal field for {dataset_id}",
            )
    expected_positive_infinity: Counter[str] = Counter()
    for scan in receipt.get("source_scans", []):
        assert isinstance(scan, Mapping)
        for item in scan.get("positive_infinity_by_column", []):
            assert isinstance(item, Mapping)
            output_column = str(item.get("output_column"))
            expected_positive_infinity[output_column] += int(item.get("count", 0))
    if dataset_id == AFRICA_DATASET_ID:
        expected_infinity_column_count = sum(
            int(item["positive_infinity_cells"]) > 0
            for item in AFRICA_FROZEN_SOURCES.values()
        )
        _require(
            sum(expected_positive_infinity.values()) == AFRICA_POSITIVE_INFINITY_TOTAL
            and len(expected_positive_infinity) == expected_infinity_column_count,
            "Africa positive-infinity output-column evidence is not exact",
        )
        for scan in receipt.get("source_scans", []):
            assert isinstance(scan, Mapping)
            for item in scan.get("positive_infinity_by_column", []):
                assert isinstance(item, Mapping)
                infinity_meta = column_records.get(str(item.get("output_column")))
                _require(
                    infinity_meta is not None
                    and infinity_meta["table_name"] == scan.get("table_name")
                    and infinity_meta["source_name"]
                    == AFRICA_POSITIVE_INFINITY_SOURCE_NAME
                    and infinity_meta["source_type"] == "DOUBLE"
                    and infinity_meta["conversion"] == "identity",
                    "Africa positive infinity is not bound to each source's retained "
                    "DOUBLE z_gpp column",
                )
    else:
        _require(
            not expected_positive_infinity,
            f"positive infinity is not approved for dataset {dataset_id}",
        )
    emitted_nan_by_column = receipt.get("emitted_nan_by_column")
    emitted_nan_cells = receipt.get("emitted_nan_cells")
    _require(
        isinstance(emitted_nan_cells, int)
        and not isinstance(emitted_nan_cells, bool)
        and emitted_nan_cells >= 0
        and isinstance(emitted_nan_by_column, list),
        f"invalid emitted NaN evidence for {dataset_id}",
    )
    expected_nan: Counter[str] = Counter()
    nan_columns_seen: list[str] = []
    for item in emitted_nan_by_column:
        _require(isinstance(item, Mapping), f"invalid emitted NaN row for {dataset_id}")
        column = str(item.get("output_column"))
        count = item.get("count")
        meta = column_records.get(column)
        _require(
            meta is not None
            and meta["source_type"] in {"FLOAT", "DOUBLE"}
            and item.get("output_dtype") == meta["output_dtype"]
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0,
            f"emitted NaN is not bound to a retained FLOAT/DOUBLE column for {dataset_id}",
        )
        nan_columns_seen.append(column)
        expected_nan[column] += int(count)
    _require(
        nan_columns_seen == sorted(set(nan_columns_seen))
        and sum(expected_nan.values()) == emitted_nan_cells,
        f"emitted NaN totals/order differ for {dataset_id}",
    )
    if lost == 0:
        source_retained_nan: Counter[str] = Counter()
        for item in receipt.get("source_nan_by_column", []):
            if isinstance(item, Mapping) and item.get("output_column") in retained_columns:
                source_retained_nan[str(item["output_column"])] += int(item.get("count", 0))
        _require(
            expected_nan == source_retained_nan,
            f"complete-row export did not preserve every retained source NaN for {dataset_id}",
        )
    (
        header,
        used_ids,
        series_row_counts,
        observed_positive_infinity,
        observed_nan,
        rows,
    ) = _validate_data_csv(
        data_path,
        receipt,
        {name: row["output_dtype"] for name, row in column_records.items()},
        frozenset(expected_positive_infinity),
        frozenset(expected_nan),
    )
    _require(
        observed_positive_infinity == expected_positive_infinity,
        f"CSV positive-infinity counts differ from receipt for {dataset_id}",
    )
    _require(
        observed_nan == expected_nan,
        f"CSV NaN counts differ from receipt for {dataset_id}",
    )
    mapping = _validate_series_map(
        map_path, receipt, used_ids, dataset_id=dataset_id
    )
    null_tag_series = {
        series_id: record
        for series_id, record in mapping.items()
        if any(value.endswith(":null") for value in record["typed_tags"])
    }
    if dataset_id == TRENDING_DATASET_ID:
        _require(
            len(null_tag_series) == 1,
            "trending export must contain exactly one explicit NULL symbol series",
        )
        null_series_id, null_series = next(iter(null_tag_series.items()))
        _require(
            null_series["table_name"] == TRENDING_DATASET_ID
            and null_series["tag_names"] == ["symbol"]
            and null_series["typed_tags"] == ["STRING:null"]
            and null_series["source_paths"] == [TRENDING_SOURCE_PATH]
            and series_row_counts[null_series_id] == TRENDING_NULL_TAG_ROWS
            and emitted == eligible == TRENDING_SOURCE_ROWS
            and lost == 0,
            "trending NULL symbol rows were not preserved exactly",
        )
    next_row_bytes = _fixed_uint20(receipt.get("next_row_bytes_decimal_20"), "next row bytes")
    next_total = _fixed_uint20(
        receipt.get("next_row_would_total_bytes_decimal_20"), "next-row total"
    )
    if lost:
        _require(len(header[2:]) == 1 and next_row_bytes > 0 and next_total > cap,
                 f"row-prefix maximality witness is invalid for {dataset_id}")
    else:
        _require(next_row_bytes == 0 and next_total == 0,
                 f"unexpected row-prefix witness for complete dataset {dataset_id}")
    return {
        "dataset_id": dataset_id,
        "actual_bytes": actual_bytes,
        "emitted_rows": rows,
        "value_columns": header[2:],
        "candidate_value_column_count": len(candidate_columns),
        "dropped_value_column_count": len(dropped_columns),
        "series_count": len(mapping),
    }, mapping, column_records


def _type_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _category_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _schema_columns(schema: Any) -> list[tuple[str, str, str]]:
    raw = schema.get_columns() if hasattr(schema, "get_columns") else getattr(schema, "columns", ())
    result: list[tuple[str, str, str]] = []
    for column in raw:
        name = str(column.get_column_name() if hasattr(column, "get_column_name")
                   else getattr(column, "column_name", ""))
        dtype = _type_name(column.get_data_type() if hasattr(column, "get_data_type")
                           else getattr(column, "data_type", ""))
        category = _category_name(column.get_category() if hasattr(column, "get_category")
                                  else getattr(column, "category", ""))
        _require(name and name != "Time", "pilot source schema has invalid column name")
        result.append((name, dtype, category))
    _require(result and len({row[0] for row in result}) == len(result),
             "pilot source schema is empty or has duplicate columns")
    return result


def _schema_sha256(schema_columns: Sequence[tuple[str, str, str]]) -> str:
    """Fingerprint a table schema exactly as the frozen producer does."""
    basis = [
        {"name": name, "tsfile_type": dtype, "category": category}
        for name, dtype, category in schema_columns
    ]
    return hashlib.sha256(_canonical_json(basis).encode("utf-8")).hexdigest()


def _forbush_direct_rowwise_binding(
    *,
    dataset_id: str,
    source: Mapping[str, Any],
    table_name: str,
    schema_columns: Sequence[tuple[str, str, str]],
) -> bool:
    """Bind the native-safety exception to one immutable Forbush object.

    The pinned 2.1.5 SDK can segfault after its DataFrame path encounters a
    nullable integer and the same query is reopened rowwise in that process.
    A fresh rowwise query is exact, so this binding selects it *before* any
    DataFrame call.  No other dataset/source/schema receives the exception.
    """

    if dataset_id != FORBUSH_DATASET_ID:
        return False
    _require(
        source.get("path") == FORBUSH_SOURCE_PATH
        and source.get("sha256") == FORBUSH_SOURCE_SHA256
        and table_name == FORBUSH_TABLE_NAME
        and _schema_sha256(schema_columns) == FORBUSH_SCHEMA_SHA256,
        "Forbush direct-rowwise source/table/schema binding differs",
    )
    return True


def _source_bound_temporal_rules(
    policy_row: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw = policy_row.get("source_bound_temporal_fields", [])
    _require(isinstance(raw, list), "source_bound_temporal_fields must be a list")
    rules: list[Mapping[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, item in enumerate(raw):
        _require(
            isinstance(item, Mapping),
            f"source-bound temporal rule {index} is not an object",
        )
        path = _safe_relative(
            item.get("source_path"), f"source-bound temporal rule {index} path"
        )
        source_name = item.get("source_name")
        source_sha = item.get("source_sha256")
        schema_sha = item.get("source_schema_sha256")
        table_name = item.get("table_name")
        identity = (path, str(table_name), str(source_name))
        _require(
            identity not in identities
            and isinstance(source_name, str)
            and bool(source_name)
            and isinstance(table_name, str)
            and bool(table_name)
            and isinstance(source_sha, str)
            and SHA256_RE.fullmatch(source_sha) is not None
            and isinstance(schema_sha, str)
            and SHA256_RE.fullmatch(schema_sha) is not None,
            f"source-bound temporal rule {index} identity is invalid",
        )
        identities.add(identity)
        _require(
            item.get("source_type") in {"INT32", "INT64"}
            and item.get("source_category") == "FIELD"
            and item.get("source_unit") in {"s", "ms", "us"}
            and item.get("output_name") == f"{source_name}__timestamp_unix_us"
            and item.get("output_type") == "int64"
            and item.get("output_unit") == "us"
            and item.get("parse_kind") == "integer_epoch"
            and item.get("timezone") == "UTC"
            and item.get("event_utc_instant_claim") is True
            and item.get("semantic_kind") == "unix_epoch_instant"
            and item.get("timezone_basis") == "upstream_encoded_epoch",
            f"source-bound temporal rule {index} conversion contract is unsupported",
        )
        rules.append(item)
    return rules


def _applicable_temporal_rules(
    *,
    dataset_id: str,
    policy_row: Mapping[str, Any],
    source: Mapping[str, Any],
    table_name: str,
    schema_columns: Sequence[tuple[str, str, str]],
) -> tuple[dict[tuple[str, str, str], Mapping[str, Any]],
           dict[tuple[str, str, str], Mapping[str, Any]]]:
    """Resolve temporal rules without relaxing a same-name schema mismatch.

    A generic STRING/TAG rule may coexist with a different representation only
    when the latter has an exact path, object hash, table, and schema binding.
    """
    schema_by_name = {
        name: (dtype, category) for name, dtype, category in schema_columns
    }
    relative = str(source.get("path"))
    digest = str(source.get("sha256"))
    fingerprint = _schema_sha256(schema_columns)
    bound: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for rule in _source_bound_temporal_rules(policy_row):
        if rule.get("source_path") != relative:
            continue
        _require(
            rule.get("source_sha256") == digest
            and rule.get("table_name") == table_name
            and rule.get("source_schema_sha256") == fingerprint,
            f"source-bound temporal identity differs for {dataset_id}/{relative}",
        )
        key = (
            str(rule.get("source_name")),
            str(rule.get("source_type")),
            str(rule.get("source_category")),
        )
        _require(
            schema_by_name.get(key[0]) == key[1:],
            f"source-bound temporal schema differs for {dataset_id}/{relative}/{key[0]}",
        )
        _require(key not in bound, f"duplicate source-bound temporal rule: {key}")
        bound[key] = rule

    generic: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for item in policy_row.get("temporal_string_fields", []):
        _require(isinstance(item, Mapping), "temporal_string_fields entry is invalid")
        name = str(item.get("source_name"))
        key = (
            name,
            str(item.get("source_type")),
            str(item.get("source_category")),
        )
        actual = schema_by_name.get(name)
        if actual is None:
            continue
        if actual == key[1:]:
            _require(key not in generic, f"duplicate temporal string rule: {key}")
            generic[key] = item
            continue
        _require(
            any(bound_key[0] == name for bound_key in bound),
            f"temporal field {name!r} schema mismatch is not source-bound for "
            f"{dataset_id}/{relative}: actual={actual!r}, expected={key[1:]!r}",
        )
    _require(
        not (set(generic) & set(bound)),
        f"temporal field has both generic and source-bound rules in {dataset_id}/{relative}",
    )
    return generic, bound


def _convert_source_bound_temporal(value: Any, rule: Mapping[str, Any]) -> int:
    _require(
        rule.get("source_type") in {"INT32", "INT64"}
        and rule.get("source_category") == "FIELD"
        and rule.get("parse_kind") == "integer_epoch"
        and rule.get("source_unit") in {"s", "ms", "us"}
        and rule.get("output_type") == "int64"
        and rule.get("output_unit") == "us"
        and rule.get("timezone") == "UTC",
        "source-bound integer temporal conversion contract is unsupported",
    )
    _require(
        not isinstance(value, bool),
        f"source-bound temporal value is not an INT64 integer: {value!r}",
    )
    try:
        integer = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CsvValidationError(
            f"source-bound temporal value is not an INT64 integer: {value!r}"
        ) from error
    _require(
        INT64_MIN <= integer <= INT64_MAX,
        "source-bound temporal input is outside int64",
    )
    multiplier = {"s": 1_000_000, "ms": 1_000, "us": 1}[
        str(rule.get("source_unit"))
    ]
    result = integer * multiplier
    _require(
        INT64_MIN <= result <= INT64_MAX,
        "source-bound temporal seconds-to-microseconds conversion overflows int64",
    )
    return result


def _typed_tag(dtype: str, value: Any) -> str:
    if value is None:
        normalized: Any = None
    elif isinstance(value, bytes):
        normalized: Any = {"hex": value.hex()}
    elif isinstance(value, (str, int, float, bool)):
        normalized = value
    else:
        normalized = str(value)
    try:
        encoded = _canonical_json(normalized)
    except ValueError as error:
        raise CsvValidationError("pilot source TAG contains NaN/infinity") from error
    return f"{dtype}:{encoded}"


def _convert_time(value: Any, unit: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CsvValidationError(f"pilot source Time is not an integer: {value!r}") from error
    multiplier = {"s": 1_000_000, "ms": 1_000, "us": 1}.get(unit)
    _require(multiplier is not None, f"pilot rescan does not support Time unit {unit!r}")
    result = integer * multiplier
    _require(INT64_MIN <= result <= INT64_MAX, "pilot converted timestamp overflows int64")
    return result


def _format_source_value(
    value: Any,
    dtype: str,
    *,
    allow_positive_infinity: bool = False,
    allow_nan: bool = False,
) -> str:
    if value is None:
        return ""
    if dtype in {"INT32", "INT64"}:
        return str(int(value))
    number = float(value)
    if math.isnan(number):
        _require(
            allow_nan and _is_true_ieee_nan(value),
            "source rescan found unapproved or non-numeric NaN",
        )
        return NAN_ENCODING
    if math.isinf(number):
        _require(
            allow_positive_infinity and number > 0,
            "source rescan found unapproved infinity",
        )
        return "inf"
    return repr(number)


def _is_true_ieee_nan(value: Any) -> bool:
    """Accept a numeric IEEE NaN, never a string/bytes/bool spelling."""

    if isinstance(value, (str, bytes, bytearray, bool)):
        return False
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _convert_temporal_string(value: Any, rule: Mapping[str, Any]) -> int:
    _require(isinstance(value, str) and rule.get("output_unit") == "us",
             "source temporal string contract is unsupported")
    text = value.strip()
    _require(bool(text), "source temporal string is empty")
    if rule.get("parse_kind") == "integer_epoch":
        _require(
            rule.get("source_unit") == "d" and re.fullmatch(r"[+-]?\d+", text) is not None,
            f"source temporal integer epoch is invalid: {value!r}",
        )
        result = int(text) * 86_400_000_000
        _require(INT64_MIN <= result <= INT64_MAX,
                 "source temporal integer epoch overflows int64")
        return result
    _require(rule.get("parse_kind") == "strptime",
             "source temporal parse kind is unsupported")
    raw_formats = rule.get("accepted_formats")
    formats = (
        list(raw_formats)
        if isinstance(raw_formats, list) and raw_formats
        else [rule.get("format")]
    )
    _require(all(isinstance(item, str) and item for item in formats),
             "source temporal format contract is invalid")
    parsed = None
    for pattern in formats:
        try:
            parsed = datetime.strptime(text, pattern)
            break
        except ValueError:
            continue
    _require(parsed is not None, f"source temporal string parse failed: {value!r}")
    timezone_name = rule.get("timezone")
    if timezone_name == "embedded_offset":
        _require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
                 "source temporal string lacks its required embedded offset")
        aware = parsed
    else:
        _require(isinstance(timezone_name, str) and timezone_name,
                 "source temporal timezone contract is invalid")
        _require(parsed.tzinfo is None,
                 "source temporal string unexpectedly embeds an offset")
        zone = ZoneInfo(timezone_name)
        fold = rule.get("ambiguous_fold")
        _require(fold in {None, 0, 1}, "source temporal ambiguous_fold is invalid")
        offset0 = parsed.replace(tzinfo=zone, fold=0).utcoffset()
        offset1 = parsed.replace(tzinfo=zone, fold=1).utcoffset()
        _require(offset0 == offset1 or fold is not None,
                 "source temporal string is DST-ambiguous without ambiguous_fold")
        aware = parsed.replace(tzinfo=zone, fold=0 if fold is None else fold)
        roundtrip = aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
        _require(roundtrip == parsed, "source temporal string is a nonexistent civil time")
    parsed = aware.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    result = (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    _require(INT64_MIN <= result <= INT64_MAX, "pilot temporal string overflows int64")
    return result


def _load_tsfile_module() -> Any:
    try:
        installed = package_version("tsfile")
    except PackageNotFoundError as error:
        raise CsvValidationError(
            f"source rescan requires pinned tsfile {REQUIRED_TSFILE_VERSION}"
        ) from error
    _require(installed == REQUIRED_TSFILE_VERSION,
             f"pilot source rescan requires tsfile {REQUIRED_TSFILE_VERSION}, found {installed}")
    try:
        import tsfile  # type: ignore
    except Exception as error:  # pragma: no cover - environment dependent
        raise CsvValidationError(f"cannot import tsfile for pilot rescan: {error}") from error
    return tsfile


def _resolve_table_schema(
    schemas: Any, configured: Any, source_path: str
) -> tuple[str, Any]:
    _require(isinstance(schemas, Mapping) and bool(schemas),
             f"schema metadata pass found no table schema in {source_path}")
    if configured == "__single_table__":
        _require(len(schemas) == 1,
                 f"schema metadata pass expected one table in {source_path}")
        name = next(iter(schemas))
        return str(name), schemas[name]
    _require(isinstance(configured, str) and configured in schemas,
             f"configured table {configured!r} is absent in {source_path}")
    return configured, schemas[configured]


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _coverage_query_columns(
    *,
    dataset_id: str,
    source: Mapping[str, Any],
    schema_columns: Sequence[tuple[str, str, str]],
) -> list[str]:
    """Return the only approved query projection for one frozen source."""

    full = [name for name, _dtype, _category in schema_columns]
    if dataset_id != AFRICA_DATASET_ID:
        return full
    relative = str(source.get("path"))
    proof = AFRICA_FROZEN_SOURCES.get(relative)
    _require(proof is not None, f"unapproved Africa source path: {relative}")
    _require(
        source.get("sha256") == proof["sha256"],
        f"Africa safe projection source hash mismatch: {relative}",
    )
    schema_by_name = {
        name: (dtype, category) for name, dtype, category in schema_columns
    }
    _require(
        all(schema_by_name.get(name) == ("STRING", "FIELD")
            for name in AFRICA_OMITTED_QUERY_COLUMNS),
        f"Africa safe projection metadata contract changed: {relative}",
    )
    omitted = set(AFRICA_OMITTED_QUERY_COLUMNS)
    return [name for name in full if name not in omitted]


def _adapter_source_rule(
    policy_row: Mapping[str, Any], table_name: str, dataset_id: str
) -> Mapping[str, Any]:
    adapter = policy_row.get("mixed_axis_adapter")
    _require(isinstance(adapter, Mapping) and isinstance(adapter.get("adapter_id"), str),
             f"adapter policy is missing for {dataset_id}")
    rules = adapter.get("source_rules")
    _require(isinstance(rules, list) and bool(rules),
             f"adapter source rules are missing for {dataset_id}")
    matches = [
        rule for rule in rules
        if isinstance(rule, Mapping)
        and isinstance(rule.get("table_names"), list)
        and table_name in rule["table_names"]
    ]
    _require(len(matches) == 1,
             f"table {table_name!r} matches {len(matches)} adapter rules for {dataset_id}")
    return matches[0]


def _row_filter_keeps_values(
    raw: Mapping[str, Any],
    nulls: Mapping[str, bool],
    row_filter: Mapping[str, Any] | None,
) -> bool:
    """Apply only the frozen adapter's supported all-of row predicates."""

    if row_filter is None:
        return True
    _require(row_filter.get("kind") == "all", "source row filter kind is unsupported")
    conditions = row_filter.get("conditions")
    _require(isinstance(conditions, list) and bool(conditions),
             "source row filter conditions are invalid")
    for condition in conditions:
        _require(isinstance(condition, Mapping), "source row filter condition is invalid")
        name = str(condition.get("source_name"))
        _require(name in raw and name in nulls,
                 f"source row filter references absent column {name!r}")
        if nulls[name]:
            return False
        operator = condition.get("operator")
        if operator == "not_null":
            continue
        if operator == "int64_gte":
            threshold = condition.get("value")
            _require(isinstance(threshold, int) and not isinstance(threshold, bool),
                     "source row-filter threshold is invalid")
            if int(raw[name]) < threshold:
                return False
            continue
        raise CsvValidationError(f"unsupported frozen row filter operator: {operator!r}")
    return True


def _schema_metadata_pass(
    *,
    dataset_ids: Sequence[str],
    mirror: Path,
    policies: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    source_tsfiles: Mapping[str, Sequence[Mapping[str, Any]]],
    mappings: Mapping[str, Mapping[int, Mapping[str, Any]]],
    columns_by_dataset: Mapping[str, Mapping[str, Mapping[str, str]]],
    tsfile_module: Any | None,
) -> list[dict[str, Any]]:
    """Independently rebuild schema-level plans without reading source rows."""
    module = _load_tsfile_module() if tsfile_module is None else tsfile_module
    reports: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        policy = policies[dataset_id]
        receipt = receipts[dataset_id]
        scans = receipt["source_scans"]
        temporal_rules = {
            (
                str(item.get("source_name")),
                str(item.get("source_type")),
                str(item.get("source_category")),
            ): item
            for item in policy.get("temporal_string_fields", [])
            if isinstance(item, Mapping)
        }
        source_bound_rule_counts = {
            (str(item.get("source_path")), str(item.get("source_name"))): 0
            for item in _source_bound_temporal_rules(policy)
        }
        logical: dict[str, tuple[str, str, str, str, str]] = {}
        source_layouts: dict[str, tuple[str, list[str], str]] = {}
        adapter_rule_counts: dict[str, int] = {}
        float_source_count = 0
        for source, scan in zip(source_tsfiles[dataset_id], scans):
            relative = str(source["path"])
            reader = None
            try:
                reader = module.TsFileReader(str(mirror / dataset_id / relative))
                table_name, schema = _resolve_table_schema(
                    reader.get_all_table_schemas(), policy.get("table_name"), relative
                )
                schema_columns = _schema_columns(schema)
                applicable_temporal, applicable_source_bound = (
                    _applicable_temporal_rules(
                        dataset_id=dataset_id,
                        policy_row=policy,
                        source=source,
                        table_name=table_name,
                        schema_columns=schema_columns,
                    )
                )
                for item in applicable_source_bound.values():
                    source_bound_rule_counts[
                        (str(item["source_path"]), str(item["source_name"]))
                    ] += 1
                if any(
                    category == "FIELD" and dtype in {"FLOAT", "DOUBLE"}
                    for _name, dtype, category in schema_columns
                ):
                    float_source_count += 1
                _require(all(category in {"TAG", "FIELD", "ATTRIBUTE"}
                             for _name, _dtype, category in schema_columns),
                         f"unsupported schema category in {dataset_id}/{relative}")
                if policy.get("decision") == "admit_dataset_adapter":
                    rule = _adapter_source_rule(policy, table_name, dataset_id)
                    rule_id = str(rule.get("rule_id", ""))
                    _require(bool(rule_id), f"adapter rule has no ID for {dataset_id}")
                    adapter_rule_counts[rule_id] = adapter_rule_counts.get(rule_id, 0) + 1
                    source_action = rule.get("action")
                    timestamp = rule.get("timestamp")
                else:
                    source_action = "materialize"
                    timestamp = policy.get("timestamp")
                _require(source_action in {"materialize", "exclude_entire_source_file"},
                         f"unsupported source action in {dataset_id}/{relative}")
                _require(scan.get("table_name") == table_name
                         and scan.get("source_action") == source_action,
                         f"source scan table/action differs from schema plan in {dataset_id}/{relative}")
                if source_action == "exclude_entire_source_file":
                    _require(scan.get("query_columns") == [],
                             f"excluded source unexpectedly declares query columns: {relative}")
                    source_layouts[relative] = (table_name, [], source_action)
                    continue

                _require(isinstance(timestamp, Mapping),
                         f"materialized source lacks timestamp policy: {dataset_id}/{relative}")
                schema_by_name = {
                    name: (dtype, category) for name, dtype, category in schema_columns
                }
                primary = timestamp.get("source_column")
                if primary != "Time":
                    actual = schema_by_name.get(str(primary))
                    _require(
                        actual is not None
                        and (str(primary), actual[0], actual[1]) in temporal_rules,
                        f"primary temporal source is not independently admitted: {dataset_id}/{relative}",
                    )
                tags = [name for name, _dtype, category in schema_columns if category == "TAG"]
                emitted_fields: list[str] = []
                for name, dtype, category in schema_columns:
                    key_schema = (name, dtype, category)
                    temporal = applicable_temporal.get(key_schema)
                    source_bound = applicable_source_bound.get(key_schema)
                    if temporal is not None:
                        conversion = "policy_temporal_string_to_int64_us"
                        output_dtype = "int64"
                    elif source_bound is not None:
                        conversion = DMI_SOURCE_BOUND_CONVERSION
                        output_dtype = "int64"
                    elif category == "FIELD" and dtype in NUMERIC_TYPES:
                        conversion = "identity"
                        output_dtype = {
                            "INT32": "int32", "INT64": "int64",
                            "FLOAT": "float32", "DOUBLE": "float64",
                        }[dtype]
                    else:
                        continue
                    emitted_fields.append(name)
                    key = _canonical_json([table_name, name, dtype, conversion])
                    logical[key] = (table_name, name, dtype, conversion, output_dtype)
                coverage_columns = _coverage_query_columns(
                    dataset_id=dataset_id,
                    source=source,
                    schema_columns=schema_columns,
                )
                expected_query = _ordered_unique([
                    *tags,
                    *emitted_fields,
                    *coverage_columns,
                ])
                _require(scan.get("query_columns") == expected_query,
                         f"source scan projection differs from full schema in {dataset_id}/{relative}")
                source_layouts[relative] = (table_name, tags, source_action)
            finally:
                if reader is not None:
                    reader.close()

        _require(
            all(count == 1 for count in source_bound_rule_counts.values()),
            f"source-bound temporal rules do not each match exactly one source for {dataset_id}: "
            f"{source_bound_rule_counts}",
        )

        if policy.get("decision") == "admit_dataset_adapter":
            adapter = policy["mixed_axis_adapter"]
            for rule in adapter["source_rules"]:
                _require(
                    adapter_rule_counts.get(str(rule.get("rule_id")), 0)
                    == rule.get("expected_source_file_count"),
                    f"adapter source-file count differs for {dataset_id}/{rule.get('rule_id')}",
                )
        ordered = sorted(
            logical.items(),
            key=lambda item: (1 if item[1][3] != "identity" else 0, item[0]),
        )
        expected_columns = {
            f"c{index:04d}": {
                "logical_key": key,
                "table_name": value[0],
                "source_name": value[1],
                "source_type": value[2],
                "conversion": value[3],
                "output_dtype": value[4],
            }
            for index, (key, value) in enumerate(ordered)
        }
        retained = list(receipt.get("retained_value_columns", []))
        dropped = list(receipt.get("dropped_trailing_value_columns", []))
        _require([*retained, *dropped] == list(expected_columns),
                 f"schema metadata column prefix/suffix differs for {dataset_id}")
        actual_columns = columns_by_dataset[dataset_id]
        _require(
            all(actual_columns[name] == expected_columns[name] for name in retained),
            f"columns.csv differs from independently rebuilt metadata for {dataset_id}",
        )
        for record in mappings[dataset_id].values():
            for source_path in record["source_paths"]:
                _require(source_path in source_layouts,
                         f"series mapping names unknown source in {dataset_id}")
                table_name, tags, action = source_layouts[source_path]
                _require(
                    action == "materialize"
                    and record["table_name"] == table_name
                    and record["tag_names"] == tags,
                    f"series TAG layout differs from source schema in {dataset_id}/{source_path}",
                )
        reports.append({
            "dataset_id": dataset_id,
            "source_file_count": len(source_tsfiles[dataset_id]),
            "float_source_count": float_source_count,
            "candidate_value_column_count": len(expected_columns),
            "retained_value_column_count": len(retained),
            "dropped_value_column_count": len(dropped),
            "status": "passed",
        })
    return reports


def _read_output_tokens(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, list(reader)


def _pilot_rescan(
    *,
    dataset_id: str,
    mirror: Path,
    output: Path,
    policy_row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_tsfiles: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, Mapping[str, Any]],
    columns: Mapping[str, Mapping[str, str]],
    max_source_files: int,
    max_source_bytes: int,
    tsfile_module: Any | None,
) -> Mapping[str, Any]:
    _require(policy_row.get("decision") in {"admit_exact_utc", "admit_stored_epoch_coordinate"},
             f"pilot rescan unsupported for adapter/block dataset {dataset_id}")
    _require(not policy_row.get("dropped_timestamp_like_string_fields")
             and not policy_row.get("mixed_axis_adapter"),
             f"pilot rescan does not support dropped-temporal-string/adapter policy for {dataset_id}")
    timestamp = policy_row.get("timestamp")
    _require(isinstance(timestamp, Mapping) and timestamp.get("source_column") == "Time"
             and timestamp.get("output_unit", "us") == "us",
             f"pilot rescan requires a physical Time -> us policy for {dataset_id}")
    source_unit = timestamp.get("source_unit")
    _require(source_unit in {"s", "ms", "us"},
             f"pilot rescan unsupported source unit for {dataset_id}")
    _require(0 < len(source_tsfiles) <= max_source_files,
             f"pilot source-file bound exceeded for {dataset_id}")
    total_source_bytes = sum(int(row["bytes"]) for row in source_tsfiles)
    _require(total_source_bytes <= max_source_bytes,
             f"pilot source-byte bound exceeded for {dataset_id}")
    scans = receipt.get("source_scans")
    _require(isinstance(scans, list), f"missing source_scans for {dataset_id}")
    scan_by_path = {
        str(row.get("source_path")): row for row in scans if isinstance(row, Mapping)
    }
    expected_paths = [str(row["path"]) for row in source_tsfiles]
    _require(set(scan_by_path) == set(expected_paths) and len(scan_by_path) == len(scans),
             f"source_scans do not exactly cover TsFiles for {dataset_id}")
    module = _load_tsfile_module() if tsfile_module is None else tsfile_module
    if dataset_id == TRENDING_DATASET_ID:
        _require(
            len(source_tsfiles) == 1
            and source_tsfiles[0].get("path") == TRENDING_SOURCE_PATH
            and source_tsfiles[0].get("sha256") == TRENDING_SOURCE_SHA256,
            "trending pilot is not bound to the frozen source object",
        )
    if dataset_id == FORBUSH_DATASET_ID:
        _require(
            len(source_tsfiles) == 1
            and source_tsfiles[0].get("path") == FORBUSH_SOURCE_PATH
            and source_tsfiles[0].get("sha256") == FORBUSH_SOURCE_SHA256,
            "Forbush source rescan is not bound to the frozen source object",
        )
    output_header, output_rows = _read_output_tokens(output / dataset_id / "data.csv")
    retained = output_header[2:]
    temporal_by_name = {
        str(item.get("source_name")): item
        for item in policy_row.get("temporal_string_fields", [])
        if isinstance(item, Mapping)
    }
    source_bound_rule_counts = {
        (str(item.get("source_path")), str(item.get("source_name"))): 0
        for item in _source_bound_temporal_rules(policy_row)
    }
    map_key_to_id = {
        _canonical_json([record["table_name"], record["typed_tags"]]): series_id
        for series_id, record in mapping.items()
    }
    expected_rows: list[tuple[int, int, int, list[str]]] = []
    all_entity_keys: set[str] = set()
    entity_sources: dict[str, set[str]] = {}
    logical_columns: dict[str, tuple[str, str, str, str]] = {}
    observed_source_rows = 0
    observed_null_tag_rows = 0
    observed_null_tag_cells = 0
    observed_positive_infinity_cells = 0
    observed_nan_cells = 0
    observed_float_null_cells = 0
    observed_all_null_cells = 0
    temporal_unit_audits: list[dict[str, Any]] = []
    native_query_audits: list[dict[str, Any]] = []
    temporal_empty_string_audits: list[dict[str, Any]] = []
    for source_ordinal, source in enumerate(source_tsfiles):
        relative = str(source["path"])
        source_path = mirror / dataset_id / relative
        reader = result = None
        try:
            reader = module.TsFileReader(str(source_path))
            schemas = reader.get_all_table_schemas()
            _require(isinstance(schemas, Mapping) and bool(schemas),
                     f"pilot source has no table schema: {dataset_id}/{relative}")
            configured = policy_row.get("table_name")
            if configured == "__single_table__":
                _require(len(schemas) == 1, f"pilot source is not single-table: {relative}")
                table_name = str(next(iter(schemas)))
            else:
                _require(isinstance(configured, str) and configured in schemas,
                         f"pilot policy table is absent: {relative}")
                table_name = str(configured)
            schema_columns = _schema_columns(schemas[table_name])
            direct_rowwise = _forbush_direct_rowwise_binding(
                dataset_id=dataset_id,
                source=source,
                table_name=table_name,
                schema_columns=schema_columns,
            )
            native_query_audit: dict[str, Any] | None = None
            if direct_rowwise:
                source_sha_before = _sha256(source_path)
                _require(
                    source_sha_before == source.get("sha256"),
                    "Forbush source SHA differs before direct-rowwise rescan",
                )
                native_query_audit = {
                    "source_path": relative,
                    "source_sha256": source.get("sha256"),
                    "source_sha256_before": source_sha_before,
                    "table_name": table_name,
                    "source_schema_sha256": _schema_sha256(schema_columns),
                    "query_mode": FORBUSH_NATIVE_QUERY_MODE,
                    "dataframe_attempted": False,
                    "rowwise_fallback": False,
                }
                native_query_audits.append(native_query_audit)
            applicable_temporal, applicable_source_bound = (
                _applicable_temporal_rules(
                    dataset_id=dataset_id,
                    policy_row=policy_row,
                    source=source,
                    table_name=table_name,
                    schema_columns=schema_columns,
                )
            )
            for item in applicable_source_bound.values():
                source_bound_rule_counts[
                    (str(item["source_path"]), str(item["source_name"]))
                ] += 1
            _require(all(category in {"TAG", "FIELD", "ATTRIBUTE"}
                         for _name, _dtype, category in schema_columns),
                     f"pilot source contains unsupported schema category: {relative}")
            query_columns = _coverage_query_columns(
                dataset_id=dataset_id,
                source=source,
                schema_columns=schema_columns,
            )
            tags = [(name, dtype) for name, dtype, category in schema_columns if category == "TAG"]
            numeric = [
                (name, dtype)
                for name, dtype, category in schema_columns
                if category == "FIELD"
                and dtype in NUMERIC_TYPES
                and (name, dtype, category) not in applicable_source_bound
            ]
            temporal = []
            for name, dtype, category in schema_columns:
                key_schema = (name, dtype, category)
                if key_schema in applicable_temporal:
                    temporal.append((
                        name,
                        dtype,
                        applicable_temporal[key_schema],
                        "policy_temporal_string_to_int64_us",
                    ))
                elif key_schema in applicable_source_bound:
                    temporal.append((
                        name,
                        dtype,
                        applicable_source_bound[key_schema],
                        DMI_SOURCE_BOUND_CONVERSION,
                    ))
            for name, dtype in numeric:
                key = _canonical_json([table_name, name, dtype, "identity"])
                logical_columns[key] = (table_name, name, dtype, "identity")
            for name, dtype, _rule, conversion in temporal:
                key = _canonical_json([table_name, name, dtype, conversion])
                logical_columns[key] = (
                    table_name, name, dtype, conversion
                )
            if direct_rowwise:
                _require(
                    tuple(sorted(name for name, _dtype, _rule, _conversion in temporal))
                    == FORBUSH_TEMPORAL_FIELDS,
                    "Forbush approved temporal-field schema differs",
                )
            forbush_temporal_states: dict[str, dict[str, Any]] = {}
            if direct_rowwise:
                for name in FORBUSH_TEMPORAL_FIELDS:
                    output_matches = [
                        header
                        for header, meta in columns.items()
                        if meta["table_name"] == table_name
                        and meta["source_name"] == name
                        and meta["source_type"] == "STRING"
                        and meta["output_dtype"] == "int64"
                        and meta["conversion"]
                        == "policy_temporal_string_to_int64_us"
                    ]
                    _require(
                        len(output_matches) == 1,
                        f"Forbush temporal field lacks one retained output column: {name}",
                    )
                    forbush_temporal_states[name] = {
                        "output_column": output_matches[0],
                        "source_non_null_empty_row_ordinals": [],
                        "source_null_row_ordinals": [],
                        "source_valid_count": 0,
                    }
            result = reader.query_table(table_name, query_columns, INT64_MIN, INT64_MAX)
            metadata = result.get_metadata()
            names = [str(value) for value in metadata.get_column_list()]
            types = [_type_name(value) for value in metadata.get_data_type_list()]
            _require(bool(names) and names[0].casefold() == "time"
                     and names[1:] == query_columns and len(types) == len(names)
                     and types[0] == "INT64",
                     f"pilot query metadata differs from full schema: {relative}")
            indexes = {name: index + 1 for index, name in enumerate(names)}
            row_ordinal = 0
            source_null_tag_rows = 0
            source_null_tag_cells = 0
            source_positive_infinity: Counter[str] = Counter()
            source_nan: Counter[str] = Counter()
            source_nan_total = 0
            source_float_null_cells = 0
            dmi_proof = None
            dmi_reference_nulls = 0
            dmi_lead_nulls = 0
            dmi_overflows = 0
            dmi_relation_mismatches = 0
            dmi_reference_min: int | None = None
            dmi_reference_max: int | None = None
            dmi_time_min: int | None = None
            dmi_time_max: int | None = None
            dmi_temporal_rule: Mapping[str, Any] | None = None
            dmi_conversion: str | None = None
            if dataset_id == DMI_DATASET_ID:
                dmi_proof = DMI_FROZEN_SOURCES.get(relative)
                _require(dmi_proof is not None, f"unapproved DMI source path: {relative}")
                _require(
                    source.get("sha256") == dmi_proof["sha256"]
                    and source.get("bytes") == dmi_proof["bytes"]
                    and table_name == dmi_proof["table_name"]
                    and _schema_sha256(schema_columns) == dmi_proof["schema_sha256"],
                    f"DMI source identity/schema differs for {relative}",
                )
                reference_schema = next(
                    (
                        (dtype, category)
                        for name, dtype, category in schema_columns
                        if name == "reference_time"
                    ),
                    None,
                )
                _require(
                    reference_schema
                    == (dmi_proof["source_type"], dmi_proof["source_category"]),
                    f"DMI reference_time schema differs for {relative}",
                )
                matching_dmi = [
                    (rule, conversion)
                    for name, _dtype, rule, conversion in temporal
                    if name == "reference_time"
                ]
                _require(
                    len(matching_dmi) == 1
                    and matching_dmi[0][1] == dmi_proof["conversion"],
                    f"DMI reference_time temporal rule differs for {relative}",
                )
                dmi_temporal_rule, dmi_conversion = matching_dmi[0]
                _require(
                    source_unit == "s"
                    and "lead_time_hours" in {name for name, _dtype, _category in schema_columns},
                    f"DMI unit-relation fields differ for {relative}",
                )
            while result.next():
                raw: dict[str, Any] = {}
                for name in names:
                    index = indexes[name]
                    raw[name] = None if result.is_null_by_index(index) else result.get_value_by_index(index)
                # tsfile 2.1.4 exposes the implicit physical time column as
                # lowercase ``time`` in query metadata.  The frozen policy
                # intentionally names that logical source coordinate ``Time``.
                # Normalize only this first, implicit column; projected schema
                # column names above remain exact and case-sensitive.
                raw["Time"] = raw[names[0]]
                timestamp_us = _convert_time(raw["Time"], str(source_unit))
                observed_all_null_cells += sum(
                    raw[name] is None for name in names[1:]
                )
                for name, state in forbush_temporal_states.items():
                    value = raw.get(name)
                    if value is None:
                        state["source_null_row_ordinals"].append(row_ordinal)
                    else:
                        _require(
                            isinstance(value, str),
                            f"Forbush temporal field is not STRING: {name}",
                        )
                        if value == "":
                            state["source_non_null_empty_row_ordinals"].append(
                                row_ordinal
                            )
                        else:
                            state["source_valid_count"] += 1
                if dmi_proof is not None:
                    reference = raw.get("reference_time")
                    lead = raw.get("lead_time_hours")
                    if reference is None:
                        dmi_reference_nulls += 1
                    if lead is None:
                        dmi_lead_nulls += 1
                    if reference is not None:
                        assert dmi_temporal_rule is not None
                        if dmi_conversion == DMI_SOURCE_BOUND_CONVERSION:
                            reference_us = _convert_source_bound_temporal(
                                reference, dmi_temporal_rule
                            )
                        else:
                            reference_us = _convert_temporal_string(
                                reference, dmi_temporal_rule
                            )
                        _require(
                            reference_us % 1_000_000 == 0,
                            f"DMI reference_time is not exact epoch seconds in {relative}",
                        )
                        reference_s = reference_us // 1_000_000
                        dmi_reference_min = (
                            reference_s
                            if dmi_reference_min is None
                            else min(dmi_reference_min, reference_s)
                        )
                        dmi_reference_max = (
                            reference_s
                            if dmi_reference_max is None
                            else max(dmi_reference_max, reference_s)
                        )
                        if lead is not None and (
                            operator.index(raw["Time"]) - reference_s
                            != operator.index(lead) * 3_600
                        ):
                            dmi_relation_mismatches += 1
                    time_s = operator.index(raw["Time"])
                    dmi_time_min = time_s if dmi_time_min is None else min(dmi_time_min, time_s)
                    dmi_time_max = time_s if dmi_time_max is None else max(dmi_time_max, time_s)
                null_tag_cells = sum(raw[name] is None for name, _dtype in tags)
                if null_tag_cells:
                    source_null_tag_rows += 1
                    source_null_tag_cells += null_tag_cells
                typed_tags = [_typed_tag(dtype, raw[name]) for name, dtype in tags]
                if not typed_tags:
                    typed_tags = ["SINGLE:__single_entity__"]
                entity_key = _canonical_json([table_name, typed_tags])
                all_entity_keys.add(entity_key)
                entity_sources.setdefault(entity_key, set()).add(relative)
                for name, dtype in numeric:
                    value = raw.get(name)
                    if dtype not in {"FLOAT", "DOUBLE"}:
                        continue
                    if value is None:
                        source_float_null_cells += 1
                        continue
                    number = float(value)
                    if not math.isfinite(number):
                        if math.isnan(number):
                            _require(
                                _is_true_ieee_nan(value),
                                f"source rescan found non-numeric NaN in "
                                f"{dataset_id}/{relative}/{name}",
                            )
                            source_nan_total += 1
                            output_matches = [
                                header
                                for header, meta in columns.items()
                                if meta["table_name"] == table_name
                                and meta["source_name"] == name
                                and meta["source_type"] == dtype
                                and meta["conversion"] == "identity"
                            ]
                            _require(
                                len(output_matches) <= 1,
                                f"source NaN maps ambiguously in {dataset_id}/{relative}/{name}",
                            )
                            if output_matches:
                                source_nan[output_matches[0]] += 1
                            continue
                        allowed = (
                            dataset_id == AFRICA_DATASET_ID
                            and relative in AFRICA_FROZEN_SOURCES
                            and name == AFRICA_POSITIVE_INFINITY_SOURCE_NAME
                            and dtype == "DOUBLE"
                            and math.isinf(number)
                            and number > 0
                        )
                        _require(
                            allowed,
                            f"source rescan found unapproved non-finite value in "
                            f"{dataset_id}/{relative}/{name}",
                        )
                        output_matches = [
                            header
                            for header, meta in columns.items()
                            if meta["table_name"] == table_name
                            and meta["source_name"] == name
                            and meta["source_type"] == dtype
                            and meta["conversion"] == "identity"
                        ]
                        _require(
                            len(output_matches) == 1,
                            f"approved source infinity lacks one retained output column: "
                            f"{dataset_id}/{relative}/{name}",
                        )
                        source_positive_infinity[output_matches[0]] += 1
                cells = []
                for header in retained:
                    meta = columns[header]
                    if meta["table_name"] != table_name:
                        cells.append("")
                    elif meta["conversion"] == "policy_temporal_string_to_int64_us":
                        value = raw.get(meta["source_name"])
                        cells.append(
                            ""
                            if value is None
                            or (
                                meta["source_name"] in forbush_temporal_states
                                and value == ""
                            )
                            else str(_convert_temporal_string(
                                value, temporal_by_name[meta["source_name"]]
                            ))
                        )
                    elif meta["conversion"] == DMI_SOURCE_BOUND_CONVERSION:
                        value = raw.get(meta["source_name"])
                        rule = applicable_source_bound.get((
                            meta["source_name"], meta["source_type"], "FIELD"
                        ))
                        _require(
                            rule is not None,
                            f"source-bound output column does not apply to {relative}",
                        )
                        cells.append(
                            "" if value is None else str(
                                _convert_source_bound_temporal(value, rule)
                            )
                        )
                    else:
                        cells.append(_format_source_value(
                            raw.get(meta["source_name"]),
                            meta["source_type"],
                            allow_positive_infinity=(
                                dataset_id == AFRICA_DATASET_ID
                                and relative in AFRICA_FROZEN_SOURCES
                                and meta["source_name"]
                                == AFRICA_POSITIVE_INFINITY_SOURCE_NAME
                                and meta["source_type"] == "DOUBLE"
                            ),
                            allow_nan=meta["source_type"] in {"FLOAT", "DOUBLE"},
                        ))
                expected_rows.append((timestamp_us, source_ordinal, row_ordinal,
                                      [str(timestamp_us), entity_key, *cells]))
                row_ordinal += 1
                observed_source_rows += 1
            if dmi_proof is not None:
                dmi_audit = {
                    "source_path": relative,
                    "source_sha256": source.get("sha256"),
                    "table_name": table_name,
                    "source_schema_sha256": _schema_sha256(schema_columns),
                    "reference_time_source_type": dmi_proof["source_type"],
                    "reference_time_source_category": dmi_proof["source_category"],
                    "conversion": dmi_conversion,
                    "source_rows": row_ordinal,
                    "reference_time_nulls": dmi_reference_nulls,
                    "lead_time_hours_nulls": dmi_lead_nulls,
                    "int64_us_overflow_count": dmi_overflows,
                    "lead_relation_mismatch_count": dmi_relation_mismatches,
                    "reference_epoch_s_min": dmi_reference_min,
                    "reference_epoch_s_max": dmi_reference_max,
                    "time_epoch_s_min": dmi_time_min,
                    "time_epoch_s_max": dmi_time_max,
                }
                _require(
                    row_ordinal == dmi_proof["rows"]
                    and dmi_reference_nulls == 0
                    and dmi_lead_nulls == 0
                    and dmi_overflows == 0
                    and dmi_relation_mismatches == 0
                    and dmi_reference_min == dmi_proof["reference_epoch_s_min"]
                    and dmi_reference_max == dmi_proof["reference_epoch_s_max"]
                    and dmi_time_min == dmi_proof["time_epoch_s_min"]
                    and dmi_time_max == dmi_proof["time_epoch_s_max"],
                    f"DMI full-source temporal unit audit differs for {relative}: {dmi_audit}",
                )
                temporal_unit_audits.append(dmi_audit)
            observed_null_tag_rows += source_null_tag_rows
            observed_null_tag_cells += source_null_tag_cells
            observed_positive_infinity_cells += sum(
                source_positive_infinity.values()
            )
            observed_nan_cells += source_nan_total
            observed_float_null_cells += source_float_null_cells
            scan = scan_by_path[relative]
            _require(scan.get("source_action") == "materialize"
                     and scan.get("source_rows") == row_ordinal
                     and scan.get("eligible_rows") == row_ordinal
                     and scan.get("filtered_rows") == 0,
                     f"pilot source scan counts differ for {relative}")
            if direct_rowwise:
                assert native_query_audit is not None
                source_sha_after = _sha256(source_path)
                _require(
                    source_sha_after == source.get("sha256")
                    and scan.get("source_schema_sha256") == FORBUSH_SCHEMA_SHA256
                    and scan.get("read_mode") == FORBUSH_NATIVE_QUERY_MODE
                    and scan.get("dataframe_attempted") is False
                    and scan.get("rowwise_fallback") is False,
                    "Forbush producer/direct-rowwise receipt binding differs",
                )
                native_query_audit["source_sha256_after"] = source_sha_after
                field_audits = []
                for name in sorted(forbush_temporal_states):
                    state = forbush_temporal_states[name]
                    empty_ordinals = state[
                        "source_non_null_empty_row_ordinals"
                    ]
                    null_ordinals = state["source_null_row_ordinals"]
                    valid_count = state["source_valid_count"]
                    _require(
                        empty_ordinals == sorted(set(empty_ordinals))
                        and null_ordinals == sorted(set(null_ordinals))
                        and set(empty_ordinals).isdisjoint(null_ordinals)
                        and all(0 <= value < row_ordinal for value in empty_ordinals)
                        and all(0 <= value < row_ordinal for value in null_ordinals),
                        f"Forbush temporal empty/NULL row ordinals differ for {name}",
                    )
                    field_audit = {
                        "source_name": name,
                        "source_type": "STRING",
                        "source_category": "FIELD",
                        "output_column": state["output_column"],
                        "conversion": "policy_temporal_string_to_int64_us",
                        "representation": FORBUSH_TEMPORAL_EMPTY_REPRESENTATION,
                        "source_non_null_empty_count": len(empty_ordinals),
                        "source_non_null_empty_row_ordinals": empty_ordinals,
                        "source_null_count": len(null_ordinals),
                        "source_null_row_ordinals": null_ordinals,
                        "source_valid_count": valid_count,
                    }
                    _require(
                        len(empty_ordinals) + len(null_ordinals) + valid_count
                        == row_ordinal
                        and {
                            key: field_audit[key]
                            for key in FORBUSH_TEMPORAL_VALUE_COUNTS[name]
                        }
                        == FORBUSH_TEMPORAL_VALUE_COUNTS[name],
                        f"Forbush temporal empty/NULL counts differ for {name}",
                    )
                    field_audits.append(field_audit)
                _require(
                    scan.get("temporal_empty_string_audits") == field_audits,
                    "Forbush producer temporal empty-string audit differs",
                )
                temporal_empty_string_audits.append({
                    "source_path": relative,
                    "source_sha256": source.get("sha256"),
                    "table_name": table_name,
                    "source_schema_sha256": _schema_sha256(schema_columns),
                    "source_rows": row_ordinal,
                    "fields": field_audits,
                })
            _require(
                scan.get("null_tag_rows") == source_null_tag_rows
                and scan.get("null_tag_cells") == source_null_tag_cells,
                f"pilot NULL TAG counts differ for {relative}",
            )
            expected_positive_by_column = [
                {
                    "source_name": AFRICA_POSITIVE_INFINITY_SOURCE_NAME,
                    "output_column": output_column,
                    "source_type": "DOUBLE",
                    "count": count,
                }
                for output_column, count in sorted(source_positive_infinity.items())
            ]
            _require(
                scan.get("positive_infinity_cells")
                == sum(source_positive_infinity.values())
                and scan.get("positive_infinity_by_column")
                == expected_positive_by_column,
                f"pilot positive-infinity evidence differs for {relative}",
            )
            _require(
                scan.get("source_sha256") == source.get("sha256")
                and scan.get("nan_cells") == source_nan_total,
                f"pilot NaN evidence differs for {relative}",
            )
        finally:
            if result is not None:
                result.close()
            if reader is not None:
                reader.close()
    _require(
        all(count == 1 for count in source_bound_rule_counts.values()),
        f"source-bound temporal rules do not each match exactly one pilot source for "
        f"{dataset_id}: {source_bound_rule_counts}",
    )
    ordered_keys = sorted(
        logical_columns,
        key=lambda key: (
            1 if logical_columns[key][3] != "identity" else 0,
            key,
        ),
    )
    expected_columns = {
        f"c{index:04d}": {
            "logical_key": key,
            "table_name": value[0],
            "source_name": value[1],
            "source_type": value[2],
            "output_dtype": {
                "INT32": "int32", "INT64": "int64",
                "FLOAT": "float32", "DOUBLE": "float64",
            }.get(value[2], "int64"),
            "conversion": value[3],
        }
        for index, key in enumerate(ordered_keys)
        for value in [logical_columns[key]]
    }
    dropped = receipt.get("dropped_trailing_value_columns")
    _require([*retained, *dropped] == list(expected_columns),
             "pilot retained columns are not the exact source-schema prefix")
    _require(all(columns[name] == expected_columns[name] for name in retained),
             "pilot columns.csv metadata differs from source schema")
    stable_ids = {key: index for index, key in enumerate(sorted(all_entity_keys))}
    for series_id, record in mapping.items():
        key = _canonical_json([record["table_name"], record["typed_tags"]])
        _require(stable_ids.get(key) == series_id,
                 f"pilot series_id is not deterministic for {dataset_id}")
        _require(record["source_paths"] == sorted(entity_sources.get(key, set())),
                 f"pilot series source-path mapping differs for {dataset_id}")
    expected_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    expected_prefix: list[list[str]] = []
    for _timestamp, _source, _ordinal, row in expected_rows[:len(output_rows)]:
        entity_key = row[1]
        _require(entity_key in map_key_to_id,
                 "pilot output prefix references a series absent from mapping")
        expected_prefix.append([row[0], str(map_key_to_id[entity_key]), *row[2:]])
    _require(output_rows == expected_prefix,
             f"pilot output rows differ from independent full-schema rescan for {dataset_id}")
    _require(
        observed_source_rows == receipt.get("source_rows")
        == _fixed_uint20(receipt.get("eligible_rows_decimal_20"), "pilot eligible rows"),
             f"pilot total row coverage differs for {dataset_id}")
    _require(
        receipt.get("null_tag_rows") == observed_null_tag_rows
        and receipt.get("null_tag_cells") == observed_null_tag_cells,
        f"pilot NULL TAG totals differ for {dataset_id}",
    )
    _require(
        receipt.get("positive_infinity_cells")
        == observed_positive_infinity_cells,
        f"pilot positive-infinity total differs for {dataset_id}",
    )
    _require(
        receipt.get("source_nan_cells") == observed_nan_cells,
        f"pilot source NaN total differs for {dataset_id}",
    )
    if dataset_id == AFRICA_DATASET_ID:
        _require(
            observed_positive_infinity_cells == AFRICA_POSITIVE_INFINITY_TOTAL
            and len(output_rows) == sum(
                int(item["rows"]) for item in AFRICA_FROZEN_SOURCES.values()
            ),
            "Africa targeted source rescan did not prove all 58 +Inf cells and rows",
        )
    if dataset_id == TRENDING_DATASET_ID:
        _require(
            observed_source_rows == TRENDING_SOURCE_ROWS
            and observed_null_tag_rows == TRENDING_NULL_TAG_ROWS
            and observed_null_tag_cells == TRENDING_NULL_TAG_ROWS
            and len(output_rows) == TRENDING_SOURCE_ROWS,
            "trending targeted source rescan did not prove all rows and NULL TAGs",
        )
    if dataset_id == BITCOIN_DATASET_ID:
        _require(
            len(source_tsfiles) == 1
            and source_tsfiles[0].get("path") == BITCOIN_SOURCE_PATH
            and source_tsfiles[0].get("sha256") == BITCOIN_SOURCE_SHA256
            and observed_source_rows == BITCOIN_SOURCE_ROWS
            and len(output_rows) == BITCOIN_SOURCE_ROWS
            and observed_nan_cells == BITCOIN_TRUE_NAN_CELLS
            and observed_float_null_cells == BITCOIN_NULL_FLOAT_CELLS
            and observed_positive_infinity_cells == 0,
            "bitcoin targeted source rescan did not prove all rows/NaNs/NULLs",
        )
    if dataset_id == DMI_DATASET_ID:
        _require(
            len(source_tsfiles) == len(DMI_FROZEN_SOURCES) == 2
            and [str(item["path"]) for item in source_tsfiles]
            == sorted(DMI_FROZEN_SOURCES)
            and observed_source_rows == DMI_SOURCE_ROWS
            and len(output_rows) == DMI_SOURCE_ROWS
            and len(temporal_unit_audits) == 2,
            "DMI targeted source rescan did not prove all source/output rows",
        )
    compared_numeric_fields = sorted({
        str(meta["source_name"])
        for meta in columns.values()
        if meta["conversion"] == "identity"
        and meta["source_type"] in NUMERIC_TYPES
    })
    compared_temporal_fields = sorted({
        str(meta["source_name"])
        for meta in columns.values()
        if meta["conversion"] in {
            "policy_temporal_string_to_int64_us",
            DMI_SOURCE_BOUND_CONVERSION,
        }
    })
    if dataset_id == FORBUSH_DATASET_ID:
        double_fields = [
            meta for meta in columns.values()
            if meta["conversion"] == "identity" and meta["source_type"] == "DOUBLE"
        ]
        _require(
            observed_source_rows == FORBUSH_SOURCE_ROWS
            and len(output_rows) == FORBUSH_SOURCE_ROWS
            and observed_all_null_cells == FORBUSH_ALL_NULL_CELLS
            and len(expected_columns) == FORBUSH_CANDIDATE_VALUE_COLUMNS
            and len(compared_numeric_fields) == FORBUSH_IDENTITY_NUMERIC_COLUMNS
            and len(double_fields) == FORBUSH_DOUBLE_COLUMNS
            and tuple(compared_temporal_fields) == FORBUSH_TEMPORAL_FIELDS
            and dropped == []
            and len(retained) == len(expected_columns)
            and native_query_audits == [{
                "source_path": FORBUSH_SOURCE_PATH,
                "source_sha256": FORBUSH_SOURCE_SHA256,
                "source_sha256_before": FORBUSH_SOURCE_SHA256,
                "table_name": FORBUSH_TABLE_NAME,
                "source_schema_sha256": FORBUSH_SCHEMA_SHA256,
                "query_mode": FORBUSH_NATIVE_QUERY_MODE,
                "dataframe_attempted": False,
                "rowwise_fallback": False,
                "source_sha256_after": FORBUSH_SOURCE_SHA256,
            }]
            and len(temporal_empty_string_audits) == 1
            and {
                key: value
                for key, value in temporal_empty_string_audits[0].items()
                if key != "fields"
            } == {
                "source_path": FORBUSH_SOURCE_PATH,
                "source_sha256": FORBUSH_SOURCE_SHA256,
                "table_name": FORBUSH_TABLE_NAME,
                "source_schema_sha256": FORBUSH_SCHEMA_SHA256,
                "source_rows": FORBUSH_SOURCE_ROWS,
            }
            and [
                item["source_name"]
                for item in temporal_empty_string_audits[0]["fields"]
            ] == list(FORBUSH_TEMPORAL_FIELDS),
            "Forbush direct-rowwise full-value audit contract differs",
        )
    report = {
        "dataset_id": dataset_id,
        "source_files": len(source_tsfiles),
        "source_bytes": total_source_bytes,
        "source_rows": observed_source_rows,
        "null_tag_rows": observed_null_tag_rows,
        "null_tag_cells": observed_null_tag_cells,
        "positive_infinity_cells": observed_positive_infinity_cells,
        "nan_cells": observed_nan_cells,
        "float_null_cells": observed_float_null_cells,
        "all_null_cells": observed_all_null_cells,
        "compared_output_rows": len(output_rows),
        "candidate_value_column_count": len(expected_columns),
        "retained_value_column_count": len(retained),
        "dropped_value_column_count": len(dropped),
        "compared_numeric_fields": compared_numeric_fields,
        "compared_temporal_fields": compared_temporal_fields,
        "native_query_audits": native_query_audits,
        "no_rows_or_value_columns_dropped": (
            len(output_rows) == observed_source_rows and dropped == []
        ),
        "temporal_unit_audits": temporal_unit_audits,
        "status": "passed",
    }
    if dataset_id == FORBUSH_DATASET_ID:
        report["temporal_empty_string_audits"] = temporal_empty_string_audits
    return report


def _audit_macroecon_row_exception(
    *,
    mirror: Path,
    policy_row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_tsfiles: Sequence[Mapping[str, Any]],
    exception_policy: Mapping[str, Any],
    exception_sources: Mapping[str, Mapping[str, Any]],
    tsfile_module: Any | None,
) -> Mapping[str, Any]:
    """Rescan every macroecon row independently and prove the eight-row filter."""
    timestamp = policy_row.get("timestamp")
    _require(
        policy_row.get("decision") == "admit_stored_epoch_coordinate"
        and policy_row.get("model") == "table"
        and policy_row.get("table_name") == "__single_table__"
        and isinstance(timestamp, Mapping)
        and timestamp.get("source_column") == "Time"
        and timestamp.get("source_unit") == "ms"
        and timestamp.get("output_unit") == "us",
        "macroecon base timestamp policy changed",
    )
    _require(
        len(source_tsfiles) == int(exception_policy["expected_source_file_count"]),
        "macroecon source-file count differs from exception policy",
    )
    scans = receipt.get("source_scans")
    _require(isinstance(scans, list), "macroecon receipt lacks source scans")
    scan_by_path = {
        str(item.get("source_path")): item
        for item in scans
        if isinstance(item, Mapping)
    }
    expected_paths = [str(item["path"]) for item in source_tsfiles]
    _require(
        list(scan_by_path) == expected_paths and len(scan_by_path) == len(scans),
        "macroecon source scans do not exactly follow frozen sources",
    )
    module = _load_tsfile_module() if tsfile_module is None else tsfile_module
    total_source_rows = 0
    total_eligible_rows = 0
    total_dropped_rows = 0
    candidate_value_columns = 0
    affected_reports: list[dict[str, Any]] = []
    for source in source_tsfiles:
        relative = str(source["path"])
        declared = exception_sources.get(relative)
        reader = result = None
        try:
            reader = module.TsFileReader(str(mirror / MACROECON_DATASET_ID / relative))
            schemas = reader.get_all_table_schemas()
            _require(
                isinstance(schemas, Mapping) and len(schemas) == 1,
                f"macroecon source is not single-table: {relative}",
            )
            table_name = str(next(iter(schemas)))
            columns = _schema_columns(schemas[table_name])
            _require(
                all(dtype == "FLOAT" and category == "FIELD"
                    for _name, dtype, category in columns),
                f"macroecon source schema is not all FLOAT FIELD: {relative}",
            )
            candidate_value_columns += len(columns)
            query_columns = [name for name, _dtype, _category in columns]
            scan = scan_by_path[relative]
            _require(
                scan.get("table_name") == table_name
                and scan.get("query_columns") == query_columns,
                f"macroecon source query/schema receipt differs for {relative}",
            )
            result = reader.query_table(
                table_name, query_columns, INT64_MIN, INT64_MAX
            )
            metadata = result.get_metadata()
            names = [str(value) for value in metadata.get_column_list()]
            types = [_type_name(value) for value in metadata.get_data_type_list()]
            _require(
                bool(names)
                and names[0].casefold() == "time"
                and names[1:] == query_columns
                and types == ["INT64", *("FLOAT" for _ in query_columns)],
                f"macroecon query metadata differs from full schema: {relative}",
            )
            source_rows = 0
            eligible_rows = 0
            dropped_ordinals: list[int] = []
            while result.next():
                physical_time = (
                    None
                    if result.is_null_by_index(1)
                    else result.get_value_by_index(1)
                )
                _require(
                    physical_time is not None,
                    f"macroecon physical Time is NULL: {relative}/{source_rows}",
                )
                try:
                    physical_time_int = int(physical_time)
                except (TypeError, ValueError, OverflowError) as error:
                    raise CsvValidationError(
                        f"macroecon physical Time is not integer: {relative}/{source_rows}"
                    ) from error
                if physical_time_int == INT64_MIN:
                    all_null = all(
                        result.is_null_by_index(index)
                        for index in range(2, len(names) + 1)
                    )
                    _require(
                        declared is not None
                        and source_rows == 0
                        and all_null,
                        f"unapproved macroecon INT64_MIN row: {relative}/{source_rows}",
                    )
                    dropped_ordinals.append(source_rows)
                else:
                    _convert_time(physical_time_int, "ms")
                    for index in range(2, len(names) + 1):
                        if result.is_null_by_index(index):
                            continue
                        value = result.get_value_by_index(index)
                        try:
                            number = float(value)
                        except (TypeError, ValueError, OverflowError) as error:
                            raise CsvValidationError(
                                f"macroecon FLOAT value is invalid: "
                                f"{relative}/{source_rows}/{names[index - 1]}"
                            ) from error
                        _require(
                            math.isfinite(number),
                            f"macroecon retained FLOAT is non-finite: "
                            f"{relative}/{source_rows}/{names[index - 1]}",
                        )
                    eligible_rows += 1
                source_rows += 1
            dropped_rows = len(dropped_ordinals)
            _require(
                scan.get("source_rows") == source_rows
                and scan.get("eligible_rows") == eligible_rows
                and scan.get("filtered_rows") == dropped_rows
                and scan.get("timestamp_policy_filtered_rows") == 0
                and scan.get("row_exception_filtered_rows") == dropped_rows,
                f"macroecon independent row counts differ for {relative}",
            )
            if declared is None:
                _require(
                    not dropped_ordinals and scan.get("row_exception") is None,
                    f"unexpected macroecon dropped row in {relative}",
                )
            else:
                _require(
                    source.get("sha256") == declared.get("sha256")
                    and table_name == declared.get("table_name")
                    and source_rows == declared.get("expected_source_rows")
                    and eligible_rows == declared.get("expected_eligible_rows")
                    and dropped_rows == declared.get("expected_dropped_rows")
                    and dropped_ordinals == [0],
                    f"macroecon affected source differs from policy: {relative}",
                )
                affected_reports.append({
                    "path": relative,
                    "sha256": source["sha256"],
                    "source_rows": source_rows,
                    "eligible_rows": eligible_rows,
                    "dropped_rows": dropped_rows,
                    "dropped_source_row_ordinals": dropped_ordinals,
                })
            total_source_rows += source_rows
            total_eligible_rows += eligible_rows
            total_dropped_rows += dropped_rows
        finally:
            if result is not None:
                result.close()
            if reader is not None:
                reader.close()
    _require(
        candidate_value_columns
        == int(exception_policy["expected_candidate_value_column_count"])
        and total_source_rows == int(exception_policy["expected_source_rows"])
        and total_eligible_rows == int(exception_policy["expected_eligible_rows"])
        and total_dropped_rows == int(exception_policy["expected_dropped_rows"])
        and total_source_rows == total_eligible_rows + total_dropped_rows
        and [item["path"] for item in affected_reports] == sorted(exception_sources),
        "macroecon independent row-exception audit totals differ",
    )
    return {
        "dataset_id": MACROECON_DATASET_ID,
        "status": "passed",
        "source_files": len(source_tsfiles),
        "source_rows": total_source_rows,
        "eligible_rows": total_eligible_rows,
        "dropped_rows": total_dropped_rows,
        "affected_sources": affected_reports,
    }


class _ExactNanFallback(RuntimeError):
    pass


def _dataframe_missing_marker(value: Any) -> bool:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _scan_nan_source_exact(
    *,
    module: Any,
    source_path: Path,
    table_name: str,
    query_columns: Sequence[str],
    float_columns: Mapping[str, str],
    row_filter: Mapping[str, Any] | None = None,
    materialize: bool = True,
    row_exception: Mapping[str, Any] | None = None,
    direct_rowwise: bool = False,
) -> tuple[int, int, int, Counter[str], Counter[str], Counter[str]]:
    """Count non-finite values, optionally selecting rowwise before DataFrame.

    ``direct_rowwise`` is reserved for an identity already proven by the
    caller.  It must not be described as a DataFrame fallback because no
    DataFrame query is opened in that mode.
    """

    def open_result() -> tuple[Any, Any, list[str], list[str]]:
        reader = module.TsFileReader(str(source_path))
        result = reader.query_table(table_name, list(query_columns), INT64_MIN, INT64_MAX)
        metadata = result.get_metadata()
        names = [str(value) for value in metadata.get_column_list()]
        types = [_type_name(value) for value in metadata.get_data_type_list()]
        _require(
            bool(names)
            and names[0].casefold() == "time"
            and names[1:] == list(query_columns)
            and len(types) == len(names)
            and types[0] == "INT64",
            f"NaN audit query metadata mismatch: {source_path}",
        )
        return reader, result, names, types

    reader = result = None
    try:
        if (
            direct_rowwise
            or row_filter is not None
            or not materialize
            or row_exception is not None
        ):
            raise _ExactNanFallback()
        reader, result, names, types = open_result()
        rows = 0
        positive: Counter[str] = Counter()
        negative: Counter[str] = Counter()
        while result.next():
            if not hasattr(result, "read_data_frame"):
                raise _ExactNanFallback()
            frame = result.read_data_frame(8192)
            if frame is None or len(frame) == 0:
                raise CsvValidationError("NaN audit data frame was unexpectedly empty")
            _require(
                [str(value) for value in frame.columns] == names,
                f"NaN audit data-frame metadata mismatch: {source_path}",
            )
            tuples = list(frame.itertuples(index=False, name=None))
            if any(_dataframe_missing_marker(value) for row in tuples for value in row):
                raise _ExactNanFallback()
            indexes = {name: index for index, name in enumerate(names)}
            for values in tuples:
                for name in float_columns:
                    number = float(values[indexes[name]])
                    if math.isinf(number):
                        (positive if number > 0 else negative)[name] += 1
            rows += len(tuples)
        return rows, rows, 0, Counter(), positive, negative
    except _ExactNanFallback:
        pass
    finally:
        if result is not None:
            result.close()
        if reader is not None:
            reader.close()

    reader = result = None
    try:
        reader, result, names, types = open_result()
        indexes = {name: index + 1 for index, name in enumerate(names)}
        rows = 0
        eligible_rows = 0
        null_cells = 0
        nan_counts: Counter[str] = Counter()
        positive: Counter[str] = Counter()
        negative: Counter[str] = Counter()
        while result.next():
            raw: dict[str, Any] = {}
            nulls: dict[str, bool] = {}
            for name in names:
                index = indexes[name]
                is_null = bool(result.is_null_by_index(index))
                nulls[name] = is_null
                raw[name] = None if is_null else result.get_value_by_index(index)
            raw["Time"] = raw[names[0]]
            nulls["Time"] = nulls[names[0]]
            exception_drop = False
            physical_time = raw["Time"]
            if row_exception is not None:
                is_sentinel = (
                    not nulls["Time"] and int(physical_time) == INT64_MIN
                )
                if rows == 0 and is_sentinel:
                    _require(
                        names[1:] == list(float_columns)
                        and all(types[index] == "FLOAT" for index in range(1, len(types)))
                        and all(nulls[name] for name in names[1:]),
                        f"NaN audit macro row exception payload/schema differs: {source_path}",
                    )
                    exception_drop = True
                else:
                    _require(
                        not is_sentinel,
                        f"NaN audit found unexpected macro INT64_MIN row: {source_path}",
                    )
            keep = (
                materialize and not exception_drop
                and _row_filter_keeps_values(raw, nulls, row_filter)
            )
            for name in float_columns:
                index = indexes[name]
                if nulls[name]:
                    if keep:
                        null_cells += 1
                    continue
                raw_value = raw[name]
                number = float(raw_value)
                if math.isnan(number):
                    _require(
                        _is_true_ieee_nan(raw_value),
                        f"NaN audit found non-numeric NaN in {source_path}/{name}",
                    )
                    if keep:
                        nan_counts[name] += 1
                elif math.isinf(number):
                    if keep:
                        (positive if number > 0 else negative)[name] += 1
            if keep:
                eligible_rows += 1
            rows += 1
        return rows, eligible_rows, null_cells, nan_counts, positive, negative
    finally:
        if result is not None:
            result.close()
        if reader is not None:
            reader.close()


def _expected_source_nan_rows(
    *,
    table_name: str,
    nan_by_name: Mapping[str, int],
    float_columns: Mapping[str, str],
    output_by_key: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for name, count in nan_by_name.items():
        dtype = float_columns[name]
        key = _canonical_json([table_name, name, dtype, "identity"])
        rows.append({
            "source_name": name,
            "source_type": dtype,
            "output_column": output_by_key[key],
            "count": count,
        })
    return sorted(rows, key=lambda item: (item["output_column"], item["source_name"]))


def _audit_nan_sources(
    *,
    dataset_ids: Sequence[str],
    mirror: Path,
    policies: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    source_tsfiles: Mapping[str, Sequence[Mapping[str, Any]]],
    exception_sources: Mapping[str, Mapping[str, Any]],
    tsfile_module: Any | None,
) -> list[dict[str, Any]]:
    """Independently count true IEEE NaNs in every FLOAT/DOUBLE-bearing source."""
    module = _load_tsfile_module() if tsfile_module is None else tsfile_module
    reports: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        policy = policies[dataset_id]
        scans = receipts[dataset_id]["source_scans"]
        scan_by_path = {str(item["source_path"]): item for item in scans}
        layouts: list[
            tuple[
                Mapping[str, Any], str, list[tuple[str, str, str]], str,
                Mapping[str, Any] | None, bool,
            ]
        ] = []
        logical: dict[str, tuple[str, str, str]] = {}
        for source in source_tsfiles[dataset_id]:
            relative = str(source["path"])
            reader = None
            try:
                reader = module.TsFileReader(str(mirror / dataset_id / relative))
                table_name, schema = _resolve_table_schema(
                    reader.get_all_table_schemas(), policy.get("table_name"), relative
                )
                schema_columns = _schema_columns(schema)
                _generic_temporal, source_bound_temporal = (
                    _applicable_temporal_rules(
                        dataset_id=dataset_id,
                        policy_row=policy,
                        source=source,
                        table_name=table_name,
                        schema_columns=schema_columns,
                    )
                )
                if policy.get("decision") == "admit_dataset_adapter":
                    rule = _adapter_source_rule(policy, table_name, dataset_id)
                    source_action = str(rule.get("action"))
                    row_filter = (
                        rule.get("row_filter")
                        if isinstance(rule.get("row_filter"), Mapping) else None
                    )
                else:
                    source_action = "materialize"
                    row_filter = None
                direct_rowwise = _forbush_direct_rowwise_binding(
                    dataset_id=dataset_id,
                    source=source,
                    table_name=table_name,
                    schema_columns=schema_columns,
                )
                layouts.append((
                    source, table_name, schema_columns, source_action, row_filter,
                    direct_rowwise,
                ))
                if source_action == "materialize":
                    for name, dtype, category in schema_columns:
                        if (
                            category == "FIELD"
                            and dtype in NUMERIC_TYPES
                            and (name, dtype, category) not in source_bound_temporal
                        ):
                            key = _canonical_json([table_name, name, dtype, "identity"])
                            logical[key] = (table_name, name, dtype)
            finally:
                if reader is not None:
                    reader.close()
        ordered_keys = sorted(logical)
        output_by_key = {key: f"c{index:04d}" for index, key in enumerate(ordered_keys)}
        # Temporal-string outputs sort after all identity outputs and therefore
        # cannot change the numeric identity-column indexes above.
        _require(
            len(receipts[dataset_id].get("candidate_value_columns", [])) >= len(ordered_keys),
            f"NaN audit candidate-column mapping is incomplete for {dataset_id}",
        )
        dataset_nan = 0
        dataset_null = 0
        dataset_positive = 0
        dataset_negative = 0
        verified_rows = 0
        float_sources = 0
        affected: list[dict[str, Any]] = []
        source_query_modes: list[dict[str, Any]] = []
        for (
            source, table_name, schema_columns, source_action, row_filter,
            direct_rowwise,
        ) in layouts:
            relative = str(source["path"])
            float_columns = {
                name: dtype
                for name, dtype, category in schema_columns
                if category == "FIELD" and dtype in {"FLOAT", "DOUBLE"}
            }
            if not float_columns:
                continue
            float_sources += 1
            source_query_modes.append({
                "path": relative,
                "sha256": source["sha256"],
                "table_name": table_name,
                "source_schema_sha256": _schema_sha256(schema_columns),
                "query_mode": (
                    FORBUSH_NATIVE_QUERY_MODE
                    if direct_rowwise
                    else "dataframe_then_rowwise_if_exactness_requires"
                ),
            })
            coverage = _coverage_query_columns(
                dataset_id=dataset_id,
                source=source,
                schema_columns=schema_columns,
            )
            row_exception = (
                exception_sources.get(relative)
                if dataset_id == MACROECON_DATASET_ID else None
            )
            if row_exception is not None:
                _require(
                    row_exception.get("sha256") == source.get("sha256")
                    and row_exception.get("table_name") == table_name,
                    f"NaN audit row-exception binding differs for {relative}",
                )
            (
                rows, eligible_rows, null_cells, nan_by_name, positive, negative,
            ) = _scan_nan_source_exact(
                module=module,
                source_path=mirror / dataset_id / relative,
                table_name=table_name,
                query_columns=coverage,
                float_columns=float_columns,
                row_filter=row_filter,
                materialize=source_action == "materialize",
                row_exception=row_exception,
                direct_rowwise=direct_rowwise,
            )
            verified_rows += rows
            nan_cells = sum(nan_by_name.values())
            positive_cells = sum(positive.values())
            negative_cells = sum(negative.values())
            dataset_nan += nan_cells
            dataset_null += null_cells
            dataset_positive += positive_cells
            dataset_negative += negative_cells
            _require(
                negative_cells == 0,
                f"NaN audit found negative infinity in {dataset_id}/{relative}",
            )
            if positive_cells:
                _require(
                    dataset_id == AFRICA_DATASET_ID
                    and relative in AFRICA_FROZEN_SOURCES
                    and set(positive) == {AFRICA_POSITIVE_INFINITY_SOURCE_NAME},
                    f"NaN audit found unapproved positive infinity in {dataset_id}/{relative}",
                )
            scan = scan_by_path[relative]
            if source_action == "materialize":
                expected_nan_rows = _expected_source_nan_rows(
                    table_name=table_name,
                    nan_by_name=nan_by_name,
                    float_columns=float_columns,
                    output_by_key=output_by_key,
                )
                _require(
                    scan.get("source_sha256") == source.get("sha256")
                    and scan.get("source_rows") == rows
                    and scan.get("eligible_rows") == eligible_rows
                    and scan.get("nan_cells") == nan_cells
                    and scan.get("nan_by_column") == expected_nan_rows,
                    f"independent source NaN evidence differs for {dataset_id}/{relative}",
                )
            else:
                _require(
                    scan.get("source_action") == "exclude_entire_source_file"
                    and scan.get("source_rows") == 0
                    and scan.get("eligible_rows") == 0
                    and scan.get("nan_cells") == 0
                    and scan.get("nan_by_column") == [],
                    f"excluded-source NaN evidence differs for {dataset_id}/{relative}",
                )
            if nan_cells:
                affected.append({
                    "path": relative,
                    "sha256": source["sha256"],
                    "source_rows": rows,
                    "null_cells": null_cells,
                    "positive_infinity_cells": positive_cells,
                    "nan_cells": nan_cells,
                    "nan_by_column": [
                        {
                            "source_name": name,
                            "source_type": float_columns[name],
                            "count": count,
                        }
                        for name, count in sorted(nan_by_name.items())
                    ],
                })
        _require(
            dataset_negative == 0,
            f"NaN audit found negative infinity in {dataset_id}",
        )
        _require(
            receipts[dataset_id].get("source_nan_cells") == dataset_nan,
            f"independent dataset source NaN total differs for {dataset_id}",
        )
        report = {
            "dataset_id": dataset_id,
            "status": "passed",
            "source_count": len(source_tsfiles[dataset_id]),
            "float_source_count": float_sources,
            "source_rows_verified": verified_rows,
            "null_cells": dataset_null,
            "positive_infinity_cells": dataset_positive,
            "nan_cells": dataset_nan,
            "affected_sources": affected,
        }
        if dataset_id == BITCOIN_DATASET_ID:
            _require(
                len(source_tsfiles[dataset_id]) == 1
                and float_sources == 1
                and source_tsfiles[dataset_id][0].get("path") == BITCOIN_SOURCE_PATH
                and source_tsfiles[dataset_id][0].get("sha256") == BITCOIN_SOURCE_SHA256
                and verified_rows == BITCOIN_SOURCE_ROWS
                and dataset_nan == BITCOIN_TRUE_NAN_CELLS
                and dataset_null == BITCOIN_NULL_FLOAT_CELLS
                and dataset_positive == 0
                and len(affected) == 1,
                "bitcoin independent NaN audit contract mismatch",
            )
        if dataset_id == FORBUSH_DATASET_ID:
            report["source_query_modes"] = source_query_modes
            _require(
                len(source_tsfiles[dataset_id]) == 1
                and float_sources == 1
                and source_query_modes == [{
                    "path": FORBUSH_SOURCE_PATH,
                    "sha256": FORBUSH_SOURCE_SHA256,
                    "table_name": FORBUSH_TABLE_NAME,
                    "source_schema_sha256": FORBUSH_SCHEMA_SHA256,
                    "query_mode": FORBUSH_NATIVE_QUERY_MODE,
                }],
                "Forbush NaN audit did not use the exact direct-rowwise binding",
            )
        reports.append(report)
    return reports


def _audit_nan_source_dataset_worker(
    payload: tuple[
        str,
        Path,
        Mapping[str, Any],
        Mapping[str, Any],
        Sequence[Mapping[str, Any]],
        Mapping[str, Mapping[str, Any]],
    ],
) -> dict[str, Any]:
    """Spawn-safe, read-only wrapper for one dataset's exact NaN audit."""
    (
        dataset_id,
        mirror,
        policy,
        receipt,
        source_tsfiles,
        exception_sources,
    ) = payload
    reports = _audit_nan_sources(
        dataset_ids=[dataset_id],
        mirror=mirror,
        policies={dataset_id: policy},
        receipts={dataset_id: receipt},
        source_tsfiles={dataset_id: source_tsfiles},
        exception_sources=exception_sources,
        tsfile_module=None,
    )
    _require(len(reports) == 1, f"NaN worker returned no report for {dataset_id}")
    return reports[0]


def _audit_nan_sources_bounded(
    *,
    dataset_ids: Sequence[str],
    mirror: Path,
    policies: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    source_tsfiles: Mapping[str, Sequence[Mapping[str, Any]]],
    exception_sources: Mapping[str, Mapping[str, Any]],
    tsfile_module: Any | None,
    workers: int,
) -> list[dict[str, Any]]:
    """Audit datasets in deterministic order with at most ten read-only workers."""
    _require(1 <= workers <= 10, "validation workers must be in 1..10")
    if workers == 1:
        return _audit_nan_sources(
            dataset_ids=dataset_ids,
            mirror=mirror,
            policies=policies,
            receipts=receipts,
            source_tsfiles=source_tsfiles,
            exception_sources=exception_sources,
            tsfile_module=tsfile_module,
        )
    _require(
        tsfile_module is None,
        "parallel source audits require the pinned importable TsFile runtime",
    )
    payloads = [
        (
            dataset_id,
            mirror,
            policies[dataset_id],
            receipts[dataset_id],
            source_tsfiles[dataset_id],
            exception_sources,
        )
        for dataset_id in dataset_ids
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, len(payloads)), mp_context=context
    ) as executor:
        # map preserves the frozen dataset order.  The context manager drains
        # every submitted read-only worker before an exception is propagated.
        return list(executor.map(_audit_nan_source_dataset_worker, payloads))


def _output_cutoff(path: Path, expected_rows: int) -> int | None:
    rows = 0
    cutoff: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        _require(header[:2] == ["timestamp_unix_us", "series_id"],
                 "NaN prefix audit found an invalid CSV header")
        for row in reader:
            rows += 1
            cutoff = int(row[0])
    _require(rows == expected_rows, "NaN prefix audit output row count differs")
    return cutoff


def _source_timestamp_us(
    raw: Mapping[str, Any],
    nulls: Mapping[str, bool],
    timestamp: Mapping[str, Any],
    temporal_rules: Mapping[str, Mapping[str, Any]],
) -> int:
    source_name = str(timestamp.get("source_column"))
    _require(source_name in raw and not nulls.get(source_name, True),
             f"NaN prefix audit primary timestamp {source_name!r} is null or absent")
    if source_name == "Time":
        return _convert_time(raw[source_name], str(timestamp.get("source_unit")))
    rule = temporal_rules.get(source_name)
    _require(rule is not None,
             f"NaN prefix audit lacks temporal rule for {source_name!r}")
    return _convert_temporal_string(raw[source_name], rule)


def _audit_capped_nan_prefixes(
    *,
    dataset_ids: Sequence[str],
    mirror: Path,
    output: Path,
    policies: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    source_tsfiles: Mapping[str, Sequence[Mapping[str, Any]]],
    columns_by_dataset: Mapping[str, Mapping[str, Mapping[str, str]]],
    exception_sources: Mapping[str, Mapping[str, Any]],
    tsfile_module: Any | None,
) -> list[dict[str, Any]]:
    """Prove NaNs in every capped chronological output prefix without sorting rows."""

    candidates: list[str] = []
    retained_source_nan_by_dataset: dict[str, Counter[str]] = {}
    for dataset_id in dataset_ids:
        receipt = receipts[dataset_id]
        lost = _fixed_uint20(
            receipt.get("lost_rows_due_to_byte_cap_decimal_20"),
            f"{dataset_id} lost rows",
        )
        retained = set(receipt.get("retained_value_columns", []))
        retained_source_nan: Counter[str] = Counter()
        for item in receipt.get("source_nan_by_column", []):
            if isinstance(item, Mapping) and item.get("output_column") in retained:
                retained_source_nan[str(item["output_column"])] += int(item.get("count", 0))
        if lost > 0 and sum(retained_source_nan.values()) > 0:
            candidates.append(dataset_id)
            retained_source_nan_by_dataset[dataset_id] = retained_source_nan
    if not candidates:
        return []

    module = _load_tsfile_module() if tsfile_module is None else tsfile_module
    reports: list[dict[str, Any]] = []
    for dataset_id in candidates:
        policy = policies[dataset_id]
        receipt = receipts[dataset_id]
        columns = columns_by_dataset[dataset_id]
        emitted_rows = _fixed_uint20(
            receipt.get("emitted_rows_decimal_20"), f"{dataset_id} emitted rows"
        )
        cutoff = _output_cutoff(output / dataset_id / "data.csv", emitted_rows)
        retained_source_nan = retained_source_nan_by_dataset[dataset_id]
        if cutoff is None:
            _require(
                emitted_rows == 0
                and receipt.get("emitted_nan_cells") == 0
                and receipt.get("emitted_nan_by_column") == [],
                f"empty capped NaN prefix evidence differs for {dataset_id}",
            )
            reports.append({
                "dataset_id": dataset_id,
                "status": "passed",
                "emitted_rows": 0,
                "cutoff_timestamp_unix_us": None,
                "rows_before_cutoff": 0,
                "rows_at_cutoff": 0,
                "cutoff_tie_rows_included": 0,
                "retained_source_nan_cells": sum(retained_source_nan.values()),
                "expected_emitted_nan_cells": 0,
                "expected_emitted_nan_by_column": [],
            })
            continue

        temporal_rules = {
            str(item.get("source_name")): item
            for item in policy.get("temporal_string_fields", [])
            if isinstance(item, Mapping)
        }
        before_rows = 0
        before_nan: Counter[str] = Counter()
        total_retained_nan: Counter[str] = Counter()
        eligible_rows = 0
        # One entry per frozen source ordinal.  This is the exact secondary
        # ORDER BY key after timestamp; query row order is the tertiary key.
        tie_sources: list[tuple[int, dict[str, list[int]]]] = []
        for source in source_tsfiles[dataset_id]:
            relative = str(source["path"])
            reader = result = None
            try:
                reader = module.TsFileReader(str(mirror / dataset_id / relative))
                table_name, schema = _resolve_table_schema(
                    reader.get_all_table_schemas(), policy.get("table_name"), relative
                )
                schema_columns = _schema_columns(schema)
                if policy.get("decision") == "admit_dataset_adapter":
                    rule = _adapter_source_rule(policy, table_name, dataset_id)
                    source_action = str(rule.get("action"))
                    timestamp = rule.get("timestamp")
                    row_filter = (
                        rule.get("row_filter")
                        if isinstance(rule.get("row_filter"), Mapping) else None
                    )
                else:
                    source_action = "materialize"
                    timestamp = policy.get("timestamp")
                    row_filter = None
                if source_action != "materialize":
                    tie_sources.append((0, {}))
                    continue
                _require(isinstance(timestamp, Mapping),
                         f"NaN prefix audit lacks timestamp policy for {dataset_id}/{relative}")
                retained_float: dict[str, tuple[str, str]] = {}
                schema_by_name = {
                    name: (dtype, category) for name, dtype, category in schema_columns
                }
                for output_column, meta in columns.items():
                    if (
                        meta["table_name"] == table_name
                        and meta["conversion"] == "identity"
                        and meta["source_type"] in {"FLOAT", "DOUBLE"}
                    ):
                        source_name = meta["source_name"]
                        _require(
                            schema_by_name.get(source_name)
                            == (meta["source_type"], "FIELD")
                            and source_name not in retained_float,
                            f"NaN prefix audit float mapping is ambiguous for "
                            f"{dataset_id}/{relative}/{source_name}",
                        )
                        retained_float[source_name] = (meta["source_type"], output_column)
                coverage = _coverage_query_columns(
                    dataset_id=dataset_id,
                    source=source,
                    schema_columns=schema_columns,
                )
                result = reader.query_table(table_name, coverage, INT64_MIN, INT64_MAX)
                metadata = result.get_metadata()
                names = [str(value) for value in metadata.get_column_list()]
                types = [_type_name(value) for value in metadata.get_data_type_list()]
                _require(
                    bool(names) and names[0].casefold() == "time"
                    and names[1:] == coverage and len(types) == len(names)
                    and types[0] == "INT64",
                    f"NaN prefix audit query metadata mismatch: {dataset_id}/{relative}",
                )
                indexes = {name: index + 1 for index, name in enumerate(names)}
                source_tie_rows = 0
                source_tie_nan: dict[str, list[int]] = {}
                source_row_ordinal = 0
                row_exception = (
                    exception_sources.get(relative)
                    if dataset_id == MACROECON_DATASET_ID else None
                )
                if row_exception is not None:
                    _require(
                        row_exception.get("sha256") == source.get("sha256")
                        and row_exception.get("table_name") == table_name,
                        f"NaN prefix row-exception binding differs for {relative}",
                    )
                while result.next():
                    raw: dict[str, Any] = {}
                    nulls: dict[str, bool] = {}
                    for name in names:
                        index = indexes[name]
                        is_null = bool(result.is_null_by_index(index))
                        nulls[name] = is_null
                        raw[name] = None if is_null else result.get_value_by_index(index)
                    raw["Time"] = raw[names[0]]
                    nulls["Time"] = nulls[names[0]]
                    if row_exception is not None:
                        is_sentinel = (
                            not nulls["Time"] and int(raw["Time"]) == INT64_MIN
                        )
                        if source_row_ordinal == 0 and is_sentinel:
                            _require(
                                all(
                                    category == "FIELD" and dtype == "FLOAT"
                                    for _name, dtype, category in schema_columns
                                )
                                and all(nulls[name] for name in coverage),
                                f"NaN prefix macro exception payload/schema differs: {relative}",
                            )
                            source_row_ordinal += 1
                            continue
                        _require(
                            not is_sentinel,
                            f"NaN prefix found unexpected macro INT64_MIN row: {relative}",
                        )
                    source_row_ordinal += 1
                    if not _row_filter_keeps_values(raw, nulls, row_filter):
                        continue
                    timestamp_us = _source_timestamp_us(
                        raw, nulls, timestamp, temporal_rules
                    )
                    row_nan: list[str] = []
                    for source_name, (_dtype, output_column) in retained_float.items():
                        if nulls.get(source_name, True):
                            continue
                        value = raw[source_name]
                        number = float(value)
                        if math.isnan(number):
                            _require(
                                _is_true_ieee_nan(value),
                                f"NaN prefix audit found non-numeric NaN in "
                                f"{dataset_id}/{relative}/{source_name}",
                            )
                            row_nan.append(output_column)
                            total_retained_nan[output_column] += 1
                    eligible_rows += 1
                    if timestamp_us < cutoff:
                        before_rows += 1
                        before_nan.update(row_nan)
                    elif timestamp_us == cutoff:
                        for output_column in row_nan:
                            source_tie_nan.setdefault(output_column, []).append(source_tie_rows)
                        source_tie_rows += 1
                tie_sources.append((source_tie_rows, source_tie_nan))
            finally:
                if result is not None:
                    result.close()
                if reader is not None:
                    reader.close()

        _require(
            eligible_rows
            == _fixed_uint20(receipt.get("eligible_rows_decimal_20"),
                             f"{dataset_id} eligible rows")
            and total_retained_nan == retained_source_nan,
            f"NaN prefix audit source totals differ for {dataset_id}",
        )
        rows_at_cutoff = sum(count for count, _positions in tie_sources)
        included_ties = emitted_rows - before_rows
        _require(
            0 <= included_ties <= rows_at_cutoff,
            f"NaN prefix audit cutoff does not identify the emitted prefix for {dataset_id}",
        )
        expected_nan = before_nan.copy()
        remaining = included_ties
        for tie_rows, positions_by_column in tie_sources:
            take = min(remaining, tie_rows)
            if take:
                for output_column, positions in positions_by_column.items():
                    expected_nan[output_column] += sum(position < take for position in positions)
            remaining -= take
        _require(remaining == 0, f"NaN prefix audit tie allocation failed for {dataset_id}")
        expected_rows = [
            {
                "output_column": output_column,
                "output_dtype": columns[output_column]["output_dtype"],
                "count": count,
            }
            for output_column, count in sorted(expected_nan.items())
            if count
        ]
        _require(
            receipt.get("emitted_nan_cells") == sum(expected_nan.values())
            and receipt.get("emitted_nan_by_column") == expected_rows,
            f"capped CSV did not preserve the exact chronological NaN prefix for {dataset_id}",
        )
        reports.append({
            "dataset_id": dataset_id,
            "status": "passed",
            "emitted_rows": emitted_rows,
            "cutoff_timestamp_unix_us": cutoff,
            "rows_before_cutoff": before_rows,
            "rows_at_cutoff": rows_at_cutoff,
            "cutoff_tie_rows_included": included_ties,
            "retained_source_nan_cells": sum(retained_source_nan.values()),
            "expected_emitted_nan_cells": sum(expected_nan.values()),
            "expected_emitted_nan_by_column": expected_rows,
        })
    return reports


def validate_export(
    *,
    scope_path: Path,
    mirror: Path,
    policy_path: Path,
    row_exceptions_path: Path | None = None,
    output: Path,
    expected_dataset_byte_cap: int = MAX_DATASET_BYTES,
    require_production_policy_counts: bool = True,
    require_all_160: bool = False,
    schema_metadata_pass: bool = False,
    pilot_rescan_dataset_ids: Sequence[str] = (),
    pilot_max_source_files: int = 8,
    pilot_max_source_bytes: int = 64 * 1024 * 1024,
    validation_workers: int = 1,
    tsfile_module: Any | None = None,
) -> Mapping[str, Any]:
    _require(0 < expected_dataset_byte_cap <= MAX_DATASET_BYTES,
             "expected dataset byte cap is outside 1..1,000,000,000")
    _require(1 <= validation_workers <= 10, "validation workers must be in 1..10")
    output = _real_directory(output, "CSV export root")
    directories, files = _walk_no_links(output)
    manifest = _read_json(output / "CSV_EXPORT_MANIFEST.json", "CSV export manifest")
    _require(manifest.get("schema") == COLLECTION_SCHEMA and manifest.get("status") == "complete",
             "CSV export manifest is not complete")
    _require(manifest.get("tsfile_version") == REQUIRED_TSFILE_VERSION,
             "collection TsFile version binding mismatch")
    _require(manifest.get("dataset_byte_cap") == expected_dataset_byte_cap,
             "collection dataset byte cap differs from expected exact cap")

    scope = _read_json(scope_path, "frozen scope")
    scope_sha = _sha256(scope_path)
    scope_rows = _scope_rows(scope)
    _require(manifest.get("scope_id") == scope.get("scope_id")
             and manifest.get("scope_sha256") == scope_sha,
             "collection manifest is not bound to frozen scope")
    policy = _read_json(policy_path, "timestamp policy")
    policy_sha = _sha256(policy_path)
    _require(manifest.get("timestamp_policy_file_sha256") == policy_sha,
             "collection manifest is not bound to timestamp policy")
    policies = _policy_rows(
        policy,
        scope_id=str(scope.get("scope_id")),
        scope_sha256=scope_sha,
        scope_rows=scope_rows,
        require_production_counts=require_production_policy_counts,
    )
    exception_policy: Mapping[str, Any] | None = None
    exception_policy_sha: str | None = None
    exception_sources: dict[str, Mapping[str, Any]] = {}
    if row_exceptions_path is not None:
        exception_policy, exception_policy_sha, exception_sources = _row_exception_policy(
            row_exceptions_path,
            scope_sha256=scope_sha,
            timestamp_policy_sha256=policy_sha,
        )

    dataset_rows = manifest.get("datasets")
    _require(isinstance(dataset_rows, list) and bool(dataset_rows),
             "collection manifest has no datasets")
    _require(manifest.get("dataset_count") == len(dataset_rows),
             "collection dataset count mismatch")
    manifest_by_id: dict[str, Mapping[str, Any]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(dataset_rows):
        _require(isinstance(row, Mapping), f"collection dataset row {index} is invalid")
        dataset_id = _safe_dataset_id(row.get("dataset_id"), f"collection dataset row {index}")
        _require(dataset_id in scope_rows and dataset_id in policies and dataset_id not in manifest_by_id,
                 f"unknown or duplicate exported dataset: {dataset_id}")
        expected_receipt = f"{dataset_id}/dataset_manifest.json"
        _require(row.get("receipt") == expected_receipt,
                 f"unexpected receipt path for {dataset_id}")
        receipt = _read_json(output / expected_receipt, f"output receipt for {dataset_id}")
        _validate_receipt_identity(
            receipt,
            dataset_id=dataset_id,
            scope_row=scope_rows[dataset_id],
            policy_row=policies[dataset_id],
            scope=scope,
            scope_sha256=scope_sha,
            policy_sha256=policy_sha,
        )
        _require(
            row.get("actual_capped_bytes")
            == _fixed_uint20(receipt.get("actual_total_bytes_decimal_20"),
                             f"{dataset_id} actual bytes")
            and row.get("emitted_rows")
            == _fixed_uint20(receipt.get("emitted_rows_decimal_20"),
                             f"{dataset_id} emitted rows")
            and row.get("lost_rows_due_to_byte_cap")
            == _fixed_uint20(receipt.get("lost_rows_due_to_byte_cap_decimal_20"),
                             f"{dataset_id} lost rows"),
            f"collection/receipt summary mismatch for {dataset_id}",
        )
        manifest_by_id[dataset_id] = row
        receipts[dataset_id] = receipt

    dataset_ids = list(manifest_by_id)
    if require_all_160:
        _require(
            len(scope_rows) == 160
            and len(dataset_ids) == 160
            and set(dataset_ids) == set(scope_rows),
            "formal completion requires the exact frozen set of all 160 THULab datasets",
        )
    expected_directories = set(dataset_ids)
    expected_files = {
        "CSV_EXPORT_MANIFEST.json",
        *(f"{dataset_id}/data.csv" for dataset_id in dataset_ids),
        *(f"{dataset_id}/series_map.csv" for dataset_id in dataset_ids),
        *(f"{dataset_id}/columns.csv" for dataset_id in dataset_ids),
        *(f"{dataset_id}/dataset_manifest.json" for dataset_id in dataset_ids),
    }
    _require(directories == expected_directories,
             f"output directory whitelist mismatch: {sorted(directories ^ expected_directories)}")
    _require(files == expected_files,
             f"output file whitelist mismatch: {sorted(files ^ expected_files)}")

    source_tsfiles = _validate_source_bindings(
        mirror,
        scope=scope,
        scope_sha256=scope_sha,
        scope_rows=scope_rows,
        exported_ids=dataset_ids,
        receipts=receipts,
    )
    _validate_row_exception_bindings(
        manifest=manifest,
        dataset_ids=dataset_ids,
        receipts=receipts,
        source_tsfiles=source_tsfiles,
        exception_policy=exception_policy,
        exception_policy_sha256=exception_policy_sha,
        exception_sources=exception_sources,
    )
    dataset_reports: list[dict[str, Any]] = []
    mappings: dict[str, dict[int, dict[str, Any]]] = {}
    columns_by_dataset: dict[str, dict[str, dict[str, str]]] = {}
    for dataset_id in dataset_ids:
        report, mapping, columns = _validate_dataset(
            output,
            dataset_id,
            receipts[dataset_id],
            cap=expected_dataset_byte_cap,
            policy_row=policies[dataset_id],
        )
        dataset_reports.append(report)
        mappings[dataset_id] = mapping
        columns_by_dataset[dataset_id] = columns
        allowed_source_paths = {str(item["path"]) for item in source_tsfiles[dataset_id]}
        for series_id, record in mapping.items():
            _require(set(record["source_paths"]) <= allowed_source_paths,
                     f"series {series_id} names a source outside the frozen binding for {dataset_id}")

    requested_pilots = list(pilot_rescan_dataset_ids)
    _require(len(requested_pilots) == len(set(requested_pilots)),
             "pilot rescan dataset IDs must be unique")
    unknown_pilots = sorted(set(requested_pilots) - set(dataset_ids))
    _require(not unknown_pilots, f"pilot rescan requested unknown exports: {unknown_pilots}")
    if require_all_160:
        for required_pilot in (
            TRENDING_DATASET_ID, AFRICA_DATASET_ID, BITCOIN_DATASET_ID,
            MACROECON_DATASET_ID, DMI_DATASET_ID,
        ):
            if required_pilot not in requested_pilots:
                requested_pilots.append(required_pilot)
    macro_audit_requested = MACROECON_DATASET_ID in requested_pilots
    if macro_audit_requested:
        _require(
            exception_policy is not None,
            "macroecon source rescan requires explicit --row-exceptions policy",
        )
    run_schema_pass = schema_metadata_pass or require_all_160
    schema_reports = (
        _schema_metadata_pass(
            dataset_ids=dataset_ids,
            mirror=mirror,
            policies=policies,
            receipts=receipts,
            source_tsfiles=source_tsfiles,
            mappings=mappings,
            columns_by_dataset=columns_by_dataset,
            tsfile_module=tsfile_module,
        )
        if run_schema_pass else []
    )
    nan_source_audits = _audit_nan_sources_bounded(
        dataset_ids=dataset_ids,
        mirror=mirror,
        policies=policies,
        receipts=receipts,
        source_tsfiles=source_tsfiles,
        exception_sources=exception_sources,
        tsfile_module=tsfile_module,
        workers=validation_workers,
    )
    nan_prefix_audits = _audit_capped_nan_prefixes(
        dataset_ids=dataset_ids,
        mirror=mirror,
        output=output,
        policies=policies,
        receipts=receipts,
        source_tsfiles=source_tsfiles,
        columns_by_dataset=columns_by_dataset,
        exception_sources=exception_sources,
        tsfile_module=tsfile_module,
    )
    pilot_reports = [
        _pilot_rescan(
            dataset_id=dataset_id,
            mirror=mirror,
            output=output,
            policy_row=policies[dataset_id],
            receipt=receipts[dataset_id],
            source_tsfiles=source_tsfiles[dataset_id],
            mapping=mappings[dataset_id],
            columns=columns_by_dataset[dataset_id],
            max_source_files=pilot_max_source_files,
            max_source_bytes=pilot_max_source_bytes,
            tsfile_module=tsfile_module,
        )
        for dataset_id in requested_pilots
        if dataset_id != MACROECON_DATASET_ID
    ]
    row_exception_audits = (
        [_audit_macroecon_row_exception(
            mirror=mirror,
            policy_row=policies[MACROECON_DATASET_ID],
            receipt=receipts[MACROECON_DATASET_ID],
            source_tsfiles=source_tsfiles[MACROECON_DATASET_ID],
            exception_policy=exception_policy,
            exception_sources=exception_sources,
            tsfile_module=tsfile_module,
        )]
        if macro_audit_requested and exception_policy is not None else []
    )
    return {
        "schema": "project179-thulab-numeric-wide-csv-validation-v1",
        "status": "passed",
        "scope_id": scope.get("scope_id"),
        "dataset_byte_cap": expected_dataset_byte_cap,
        "dataset_count": len(dataset_reports),
        "validation_workers": validation_workers,
        "all_160_required": require_all_160,
        "policy_decision_counts": policy.get("decision_counts"),
        "row_exception_policy_file_sha256": exception_policy_sha,
        "datasets": dataset_reports,
        "schema_metadata_pass": schema_reports,
        "pilot_source_rescans": pilot_reports,
        "row_exception_audits": row_exception_audits,
        "nan_source_audits": nan_source_audits,
        "nan_prefix_audits": nan_prefix_audits,
        "proof_boundary": (
            "Structure, hashes, source bindings, policy, schema metadata, and capped CSV "
            "content grammar are validated for the selected export. Full source-row/value "
            "equivalence is established only for datasets listed in pilot_source_rescans. "
            "The frozen Africa exception is gated by two exact source SHA-256 values, "
            "complete schema metadata, exact 5,415/21,660 source-row counts, and the exact "
            "nine ordinary STRING FIELD omissions; the prior external Java 2.4 full-versus-"
            "safe projection equivalence audit is not recomputed by this validator. "
            "Africa z_gpp is the sole infinity exception: exactly 12+46 positive "
            "infinities are read back as float `inf`; all other infinities are rejected. "
            "True non-NULL IEEE FLOAT/DOUBLE NaNs are independently counted from every "
            "floating-point source and emitted only as lowercase `nan`; NULL remains an "
            "empty CSV cell. Bitcoin is additionally fixed at 82,443 rows, 8,073 true "
            "NaNs, zero FLOAT NULLs, and a full source/output row comparison. For cap-"
            "truncated datasets, retained-column NaNs in the emitted chronological prefix "
            "are independently recomputed by timestamp, frozen source ordinal, and source "
            "row ordinal without materializing or sorting every source row. DMI Aarhus is "
            "additionally bound to two exact source objects and table-schema hashes: its "
            "INT64 FIELD epoch-seconds and STRING TAG UTC forms are independently converted "
            "to exact int64 microseconds, checked over all 43,910 rows for NULLs, overflow, "
            "range, and Time-reference_time=lead_time_hours*3600, then compared row-for-row "
            "with the CSV output. "
            "When supplied, the macroecon row-exception policy is independently checked "
            "against all 64 full-schema sources and permits only eight first-row INT64_MIN "
            "timestamps whose 5,548 FLOAT value fields are all NULL."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--timestamp-policies", type=Path, required=True)
    parser.add_argument("--row-exceptions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-dataset-byte-cap", type=int, default=MAX_DATASET_BYTES)
    parser.add_argument("--pilot-source-rescan", action="append", default=[])
    parser.add_argument(
        "--require-all-160",
        action="store_true",
        help="require the export to contain the exact frozen set of all 160 datasets",
    )
    parser.add_argument(
        "--schema-metadata-pass",
        action="store_true",
        help="independently rebuild source table/TAG/projection/value-column metadata",
    )
    parser.add_argument("--pilot-max-source-files", type=int, default=8)
    parser.add_argument("--pilot-max-source-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="read-only dataset workers for independent full-source NaN audits (1..10)",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_export(
            scope_path=args.scope,
            mirror=args.mirror,
            policy_path=args.timestamp_policies,
            row_exceptions_path=args.row_exceptions,
            output=args.output,
            expected_dataset_byte_cap=args.expected_dataset_byte_cap,
            require_production_policy_counts=True,
            require_all_160=args.require_all_160,
            schema_metadata_pass=args.schema_metadata_pass,
            pilot_rescan_dataset_ids=args.pilot_source_rescan,
            pilot_max_source_files=args.pilot_max_source_files,
            pilot_max_source_bytes=args.pilot_max_source_bytes,
            validation_workers=args.workers,
        )
    except (CsvValidationError, OSError, ValueError, csv.Error, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
