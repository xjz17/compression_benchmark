#!/usr/bin/env python3
"""Prepare audited numeric time-series blocks for weighted-GRPO experiments.

The input contract is deliberately explicit.  A preparation specification must
name the entity columns, timestamp representation, timezone, Unix output unit,
and every numeric value column's original dtype.  The preparer never guesses a
timestamp unit, substitutes a row number, widens a numeric dtype, or silently
drops an invalid row.

The output is a directory containing one manifest and, for each
entity/split/column block, a little-endian value payload plus a missing-value
bitmap.  Floating-point NaN is a value and therefore has a clear bitmap bit;
it is not treated as missing.  Train/validation boundaries are established per
entity before block formation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import uuid
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


FORMAT_VERSION = "weighted-grpo-timeseries-preparation-v1"
COHORT_VERSION = "weighted-grpo-timeseries-cohort-v1"
DEFAULT_COHORT_MANIFEST = Path(__file__).with_name(
    "weighted_grpo_timeseries_cohort_v1.json"
)
MAX_DATASET_BYTES = 256 * 1024 * 1024
MAX_BLOCK_BYTES = 1 * 1024 * 1024
TRAIN_NUMERATOR = 4
TRAIN_DENOMINATOR = 5
MIN_SELECTED_ROWS_PER_ENTITY = 5
COHORT_ROLES = frozenset({"train_validation", "test_candidate"})
SPLIT_CONTRACTS = {
    "train_validation": {
        "cohort_role": "train_validation",
        "allowed_splits": ["train", "validation"],
        "partition": "per_entity_chronological_80_20_before_blocks",
        "train_boundary_rule": "floor(selected_rows * 4 / 5)",
    },
    "test_candidate": {
        "cohort_role": "test_candidate",
        "allowed_splits": ["test"],
        "partition": "all_selected_rows_are_test_before_blocks",
        "train_boundary_rule": None,
    },
}
SUPPORTED_DTYPES = {
    "int8": (1, "<b"),
    "uint8": (1, "<B"),
    "int16": (2, "<h"),
    "uint16": (2, "<H"),
    "int32": (4, "<i"),
    "uint32": (4, "<I"),
    "int64": (8, "<q"),
    "uint64": (8, "<Q"),
    "float16": (2, "<e"),
    "float32": (4, "<f"),
    "float64": (8, "<d"),
}
UNIX_UNIT_EXPONENT = {"s": 0, "ms": 3, "us": 6, "ns": 9}
DEFAULT_MISSING_TOKENS = ("", "null", "none", "<na>")
EXPECTED_TRAIN_VALIDATION = frozenset({
    "ETDataset-ett",
    "electricity_load_2011_2014",
    "weatherAUS",
    "eea_pm25_forecasting",
    "METR-LA",
    "finland_traffic",
    "exchange_rate",
    "bitcoin_price_timeseries",
    "freshretailnet_50k",
    "online_retail_2_uci",
    "national_illness",
    "gluco_tsfm_benchmark",
    "app_flow",
    "scaphandre_cpu_usage",
    "deep-space-probes",
    "geomagnetic_kp_index",
})
COHORT_ASSIGNMENT_SHA256 = (
    "d08557a6d38e2d575834dba46e76849cfd3fe8c82923cf80e6736610fecd282c"
)
COHORT_POLICY_SHA256 = (
    "b31fb16b4c9b3828b238a72d29ba875e48c43c5c6e2f981aae823fb06bd54278"
)


class PreparationError(ValueError):
    """Raised when an input would violate the experiment data contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _split_contract_for_binding(
    cohort_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the exact split contract for a frozen cohort role.

    Direct, non-cohort unit materializations retain the historical
    train/validation behavior.  Formal callers pass a cohort binding and are
    checked against it again by :func:`verify_prepared_dataset`.
    """
    role = (
        str(cohort_binding.get("role"))
        if isinstance(cohort_binding, Mapping)
        else "train_validation"
    )
    if role not in COHORT_ROLES:
        raise PreparationError(f"unsupported cohort role for splitting: {role!r}")
    return dict(SPLIT_CONTRACTS[role])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_local_source(
    path: Path,
    *,
    source_format: str = "auto",
    expected_sha256: str | None = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> dict[str, Any]:
    """Fingerprint a local CSV/TsFile and report only directly observable schema facts."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise PreparationError(f"source file does not exist: {resolved}")
    if source_format == "auto":
        suffix = resolved.suffix.casefold()
        if suffix == ".csv":
            source_format = "csv"
        elif suffix == ".tsfile":
            source_format = "tsfile"
        else:
            raise PreparationError(
                "cannot infer source format; use --format csv or --format tsfile"
            )
    if source_format not in {"csv", "tsfile"}:
        raise PreparationError("source format must be auto, csv, or tsfile")
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.casefold()
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise PreparationError("expected sha256 must be 64 lowercase hex digits")
    source_sha256 = sha256_file(resolved)
    result: dict[str, Any] = {
        "path": str(resolved),
        "format": source_format,
        "bytes": resolved.stat().st_size,
        "sha256": source_sha256,
        "expected_sha256": expected_sha256,
        "fingerprint_matches": (
            expected_sha256 is None or expected_sha256 == source_sha256
        ),
        "timestamp_unit": None,
        "timestamp_timezone": None,
        "timestamp_contract_status": (
            "unresolved_must_be_supplied_from_upstream_metadata_no_precision_guess"
        ),
    }
    if expected_sha256 is not None and expected_sha256 != source_sha256:
        result["schema_status"] = "blocked_fingerprint_mismatch"
        return result

    if source_format == "csv":
        if len(delimiter) != 1:
            raise PreparationError("CSV audit delimiter must be one character")
        with resolved.open(encoding=encoding, newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration as error:
                raise PreparationError("CSV source is empty") from error
        if not header or any(not name for name in header) or len(header) != len(set(header)):
            raise PreparationError("CSV header is empty, duplicated, or contains blank names")
        result.update({
            "schema_status": "header_observed_types_require_explicit_spec",
            "columns": header,
            "materialization_support": "supported_after_explicit_contract",
        })
        return result

    with resolved.open("rb") as handle:
        header_magic = handle.read(6)
        if resolved.stat().st_size >= 6:
            handle.seek(-6, os.SEEK_END)
            trailer_magic = handle.read(6)
        else:
            trailer_magic = b""
    if header_magic != b"TsFile" or trailer_magic != b"TsFile":
        result.update({
            "schema_status": "blocked_invalid_tsfile_magic",
            "materialization_support": "audit_only_not_materialized_by_csv_v1",
            "materialization_entrypoint": "prepare_weighted_grpo_tsfile_v1.py",
        })
        return result

    try:
        import tsfile  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - environment dependent
        result.update({
            "schema_status": "blocked_missing_tsfile_python_dependency",
            "materialization_support": "audit_only_not_materialized_by_csv_v1",
            "materialization_entrypoint": "prepare_weighted_grpo_tsfile_v1.py",
            "dependency_error": f"{type(error).__name__}: {error}",
        })
        return result

    reader = None
    try:
        reader = tsfile.TsFileReader(str(resolved))
        raw_schemas = reader.get_all_table_schemas()
        tables: list[dict[str, Any]] = []
        if isinstance(raw_schemas, Mapping):
            schema_values = raw_schemas.values()
        elif isinstance(raw_schemas, Sequence):
            schema_values = raw_schemas
        else:
            schema_values = ()
        for schema in schema_values:
            table_name = (
                schema.get_table_name()
                if hasattr(schema, "get_table_name")
                else getattr(schema, "table_name", None)
            )
            columns = (
                schema.get_columns()
                if hasattr(schema, "get_columns")
                else getattr(schema, "columns", ())
            )
            def enum_name(value: Any) -> str:
                return str(getattr(value, "name", value))

            tables.append({
                "table_name": str(table_name),
                "columns": [
                    {
                        "name": str(
                            column.get_column_name()
                            if hasattr(column, "get_column_name")
                            else getattr(column, "column_name", "")
                        ),
                        "data_type": enum_name(
                            column.get_data_type()
                            if hasattr(column, "get_data_type")
                            else getattr(column, "data_type", "")
                        ),
                        "category": enum_name(
                            column.get_category()
                            if hasattr(column, "get_category")
                            else getattr(column, "category", "")
                        ),
                    }
                    for column in columns
                ],
            })
        result.update({
            "schema_status": (
                "table_schema_observed_timestamp_unit_still_unresolved"
                if tables
                else "no_table_schema_observed_tree_model_or_unsupported_schema_discovery"
            ),
            "tables": tables,
            "materialization_support": "audit_only_not_materialized_by_csv_v1",
            "materialization_entrypoint": "prepare_weighted_grpo_tsfile_v1.py",
            "next_step": (
                "obtain upstream timestamp precision/timezone and entity semantics, then "
                "use the schema-driven TsFile entrypoint without dtype widening"
            ),
        })
    except Exception as error:  # pragma: no cover - depends on external TsFile bytes
        result.update({
            "schema_status": "blocked_unreadable_tsfile",
            "materialization_support": "audit_only_not_materialized_by_csv_v1",
            "materialization_entrypoint": "prepare_weighted_grpo_tsfile_v1.py",
            "reader_error": f"{type(error).__name__}: {error}",
        })
    finally:
        if reader is not None:
            reader.close()
    return result


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _required_text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, context: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PreparationError(f"{context} must be a list of strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result):
        raise PreparationError(f"{context} cannot contain empty names")
    if not allow_empty and not result:
        raise PreparationError(f"{context} cannot be empty")
    if len(result) != len(set(result)):
        raise PreparationError(f"{context} contains duplicates")
    return result


def _cohort_assignment_basis(datasets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(entry) for entry in datasets), key=lambda entry: entry["id"])


@dataclass(frozen=True)
class NumericColumn:
    source_name: str
    output_name: str
    dtype: str

    @property
    def width(self) -> int:
        return SUPPORTED_DTYPES[self.dtype][0]


@dataclass(frozen=True)
class TimestampContract:
    source_name: str
    output_name: str
    source_kind: str
    source_unit: str | None
    output_unit: str
    timezone_name: str
    formats: tuple[str, ...]
    ambiguous_fold: int | None
    duplicate_policy: str


@dataclass(frozen=True)
class PreparationContract:
    dataset_id: str
    domain: str
    source_group: str
    upstream_revision: str
    upstream_locator: str | None
    expected_source_sha256: str
    source_path: Path
    encoding: str
    delimiter: str
    entity_columns: tuple[str, ...]
    timestamp: TimestampContract
    value_columns: tuple[NumericColumn, ...]
    missing_tokens: frozenset[str]
    dataset_byte_cap: int
    block_byte_cap: int
    spec_sha256: str

    @property
    def numeric_columns(self) -> tuple[NumericColumn, ...]:
        timestamp = NumericColumn(
            self.timestamp.source_name,
            self.timestamp.output_name,
            "int64",
        )
        return (timestamp, *self.value_columns)

    @property
    def logical_row_bytes(self) -> int:
        return sum(column.width for column in self.numeric_columns)


def validate_cohort_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate domain balance and source-group isolation in the cohort manifest."""
    if manifest.get("version") != COHORT_VERSION:
        raise PreparationError(
            f"cohort version must be {COHORT_VERSION!r}, got {manifest.get('version')!r}"
        )
    domains = manifest.get("domains")
    datasets = manifest.get("datasets")
    if not isinstance(domains, list) or not isinstance(datasets, list):
        raise PreparationError("cohort domains and datasets must be lists")
    if any(not isinstance(entry, Mapping) for entry in domains):
        raise PreparationError("every cohort domain must be an object")
    if any(not isinstance(entry, Mapping) for entry in datasets):
        raise PreparationError("every cohort dataset must be an object")
    assignment_sha256 = _sha256_bytes(
        _canonical_json(_cohort_assignment_basis(datasets))
    )
    if manifest.get("assignment_sha256") != COHORT_ASSIGNMENT_SHA256:
        raise PreparationError(
            "cohort assignment_sha256 does not match the frozen v1 assignment"
        )
    policy_basis = {
        key: manifest.get(key)
        for key in ("defaults", "leakage_rules", "same_source_exclusions")
    }
    policy_sha256 = _sha256_bytes(_canonical_json(policy_basis))
    if manifest.get("policy_sha256") != COHORT_POLICY_SHA256:
        raise PreparationError(
            "cohort policy_sha256 does not match the frozen v1 policy"
        )
    domain_ids = [_required_text(item, "id", "domain") for item in domains]
    if len(domain_ids) != 8 or len(domain_ids) != len(set(domain_ids)):
        raise PreparationError("cohort must define exactly eight unique domains")

    seen_ids: set[str] = set()
    train_groups: set[str] = set()
    test_groups: set[str] = set()
    train_ids: set[str] = set()
    train_counts = {domain: 0 for domain in domain_ids}
    test_counts = {domain: 0 for domain in domain_ids}
    for index, entry in enumerate(datasets):
        dataset_id = _required_text(entry, "id", f"datasets[{index}]")
        domain = _required_text(entry, "domain", f"datasets[{index}]")
        role = _required_text(entry, "role", f"datasets[{index}]")
        source_group = _required_text(entry, "source_group", f"datasets[{index}]")
        if dataset_id in seen_ids:
            raise PreparationError(f"duplicate cohort dataset id: {dataset_id}")
        seen_ids.add(dataset_id)
        if domain not in train_counts:
            raise PreparationError(f"unknown domain for {dataset_id}: {domain}")
        if entry.get("audit_status") != "audit_required":
            raise PreparationError(
                f"frozen cohort entry {dataset_id} must remain audit_required; "
                "record passed audits in a run-specific manifest"
            )
        if role == "train_validation":
            train_counts[domain] += 1
            train_groups.add(source_group)
            train_ids.add(dataset_id)
        elif role == "test_candidate":
            test_counts[domain] += 1
            test_groups.add(source_group)
        else:
            raise PreparationError(f"unknown role for {dataset_id}: {role}")
    if any(count != 2 for count in train_counts.values()):
        raise PreparationError(
            f"each domain must contain two train/validation datasets: {train_counts}"
        )
    if train_ids != EXPECTED_TRAIN_VALIDATION:
        raise PreparationError(
            "train/validation dataset set differs from the frozen v1 cohort: "
            f"missing={sorted(EXPECTED_TRAIN_VALIDATION - train_ids)}, "
            f"extra={sorted(train_ids - EXPECTED_TRAIN_VALIDATION)}"
        )
    if len(train_groups) != len(train_ids):
        raise PreparationError("v1 train/validation datasets must have independent source groups")
    if any(count == 0 for count in test_counts.values()):
        raise PreparationError(f"each domain needs an explicit test candidate: {test_counts}")
    if len(test_groups) != sum(test_counts.values()):
        raise PreparationError("v1 test candidates must have independent source groups")
    overlap = train_groups & test_groups
    if overlap:
        raise PreparationError(f"source groups cross train/test roles: {sorted(overlap)}")

    exclusions = manifest.get("same_source_exclusions")
    if not isinstance(exclusions, list):
        raise PreparationError("cohort same_source_exclusions must be a list")
    roles_by_id = {
        _required_text(entry, "id", "cohort dataset"): _required_text(
            entry, "role", "cohort dataset"
        )
        for entry in datasets
    }
    exclusion_groups: set[str] = set()
    for index, exclusion in enumerate(exclusions):
        if not isinstance(exclusion, Mapping):
            raise PreparationError(f"same_source_exclusions[{index}] must be an object")
        exclusion_group = _required_text(
            exclusion, "source_group", f"same_source_exclusions[{index}]"
        )
        if exclusion_group in exclusion_groups:
            raise PreparationError(
                f"duplicate same-source exclusion group: {exclusion_group}"
            )
        exclusion_groups.add(exclusion_group)
        members = _string_list(
            exclusion.get("members"),
            f"same_source_exclusions[{index}].members",
            allow_empty=False,
        )
        member_roles = {roles_by_id[member] for member in members if member in roles_by_id}
        if len(member_roles) > 1:
            raise PreparationError(
                f"same-source exclusion {exclusion_group} crosses cohort roles"
            )
    if assignment_sha256 != COHORT_ASSIGNMENT_SHA256:
        raise PreparationError(
            "cohort dataset assignment or source metadata differs from frozen v1"
        )
    if policy_sha256 != COHORT_POLICY_SHA256:
        raise PreparationError("cohort defaults or leakage policy differ from frozen v1")
    return {
        "version": COHORT_VERSION,
        "assignment_sha256": assignment_sha256,
        "policy_sha256": policy_sha256,
        "domain_count": len(domain_ids),
        "train_validation_count": sum(train_counts.values()),
        "test_candidate_count": sum(test_counts.values()),
        "train_counts_by_domain": train_counts,
        "test_counts_by_domain": test_counts,
        "train_source_groups": len(train_groups),
        "test_source_groups": len(test_groups),
    }


def load_cohort_manifest(path: Path = DEFAULT_COHORT_MANIFEST) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_cohort_manifest(manifest)
    return manifest


def validate_contract_against_cohort(
    contract: PreparationContract,
    cohort_manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Bind a preparation contract to one immutable cohort assignment."""
    validate_cohort_manifest(cohort_manifest)
    matches = [
        entry
        for entry in cohort_manifest["datasets"]
        if entry.get("id") == contract.dataset_id
    ]
    if len(matches) != 1:
        raise PreparationError(
            f"dataset_id {contract.dataset_id!r} is not a unique member of the frozen cohort"
        )
    entry = matches[0]
    for field, actual in (
        ("domain", contract.domain),
        ("source_group", contract.source_group),
        ("locator", contract.upstream_locator),
    ):
        expected = entry[field]
        if actual != expected:
            raise PreparationError(
                f"preparation spec {field} for {contract.dataset_id} must be "
                f"{expected!r}, got {actual!r}"
            )
    return {
        "version": COHORT_VERSION,
        "assignment_sha256": COHORT_ASSIGNMENT_SHA256,
        "policy_sha256": COHORT_POLICY_SHA256,
        "role": str(entry["role"]),
    }


def validate_prepared_cohort_manifests(
    manifests: Sequence[Mapping[str, Any]],
    cohort_manifest: Mapping[str, Any],
    *,
    require_complete_train: bool = True,
) -> dict[str, Any]:
    """Reject cross-role family/hash leakage in a collection of prepared sources."""
    cohort_summary = validate_cohort_manifest(cohort_manifest)
    cohort_by_id = {entry["id"]: entry for entry in cohort_manifest["datasets"]}
    seen_ids: set[str] = set()
    hash_owner: dict[str, tuple[str, str]] = {}
    role_counts = {"train_validation": 0, "test_candidate": 0}
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, Mapping):
            raise PreparationError(f"prepared manifests[{index}] must be an object")
        dataset_id = _required_text(manifest, "dataset_id", f"prepared manifests[{index}]")
        if dataset_id in seen_ids:
            raise PreparationError(f"duplicate prepared dataset_id: {dataset_id}")
        seen_ids.add(dataset_id)
        cohort_entry = cohort_by_id.get(dataset_id)
        if cohort_entry is None:
            raise PreparationError(f"prepared dataset is outside the frozen cohort: {dataset_id}")
        for field in ("domain", "source_group"):
            if manifest.get(field) != cohort_entry[field]:
                raise PreparationError(
                    f"prepared {field} differs from frozen cohort for {dataset_id}"
                )
        cohort_binding = manifest.get("cohort")
        expected_binding = {
            "version": COHORT_VERSION,
            "assignment_sha256": COHORT_ASSIGNMENT_SHA256,
            "policy_sha256": COHORT_POLICY_SHA256,
            "role": cohort_entry["role"],
        }
        if cohort_binding != expected_binding:
            raise PreparationError(
                f"prepared cohort binding differs from frozen cohort for {dataset_id}"
            )
        role = str(cohort_entry["role"])
        split_contract = manifest.get("split_contract")
        if split_contract != SPLIT_CONTRACTS[role]:
            raise PreparationError(
                f"prepared split contract differs from frozen cohort role for {dataset_id}"
            )
        blocks = manifest.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise PreparationError(f"prepared blocks are missing for {dataset_id}")
        observed_splits = {
            _required_text(block, "split", f"prepared block for {dataset_id}")
            for block in blocks
            if isinstance(block, Mapping)
        }
        if len(observed_splits) == 0 or not observed_splits.issubset(
            set(SPLIT_CONTRACTS[role]["allowed_splits"])
        ):
            raise PreparationError(
                f"prepared block splits differ from frozen cohort role for {dataset_id}: "
                f"{sorted(observed_splits)}"
            )
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise PreparationError(f"prepared source is missing for {dataset_id}")
        source_sha256 = str(source.get("sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
            raise PreparationError(f"invalid source sha256 for {dataset_id}")
        prior = hash_owner.get(source_sha256)
        if prior is not None:
            prior_id, prior_role = prior
            if prior_role != role:
                raise PreparationError(
                    "exact source hash crosses train/test roles: "
                    f"{prior_id} ({prior_role}) and {dataset_id} ({role})"
                )
            raise PreparationError(
                f"exact source hash is reused by cohort datasets {prior_id} and {dataset_id}"
            )
        hash_owner[source_sha256] = (dataset_id, role)
        role_counts[role] += 1

    if require_complete_train:
        present_train = {
            dataset_id
            for dataset_id in seen_ids
            if cohort_by_id[dataset_id]["role"] == "train_validation"
        }
        if present_train != EXPECTED_TRAIN_VALIDATION:
            raise PreparationError(
                "prepared run does not contain the complete frozen 16-source "
                f"train/validation cohort: missing={sorted(EXPECTED_TRAIN_VALIDATION - present_train)}, "
                f"extra={sorted(present_train - EXPECTED_TRAIN_VALIDATION)}"
            )
    return {
        "version": COHORT_VERSION,
        "assignment_sha256": cohort_summary["assignment_sha256"],
        "policy_sha256": cohort_summary["policy_sha256"],
        "prepared_count": len(manifests),
        "role_counts": role_counts,
        "unique_source_hashes": len(hash_owner),
        "complete_train_validation": (
            role_counts["train_validation"] == len(EXPECTED_TRAIN_VALIDATION)
        ),
    }


def _parse_contract(spec: Mapping[str, Any], base_dir: Path) -> PreparationContract:
    if spec.get("format_version") != FORMAT_VERSION:
        raise PreparationError(f"format_version must be {FORMAT_VERSION!r}")
    dataset_id = _required_text(spec, "dataset_id", "spec")
    domain = _required_text(spec, "domain", "spec")
    source_group = _required_text(spec, "source_group", "spec")

    source = spec.get("source")
    if not isinstance(source, Mapping):
        raise PreparationError("spec.source must be an object")
    if source.get("format") != "csv":
        raise PreparationError(
            "CSV-v1 materialization supports only explicit local CSV; use "
            "prepare_weighted_grpo_tsfile_v1.py for schema-driven TsFile extraction"
        )
    upstream_revision = _required_text(source, "upstream_revision", "spec.source")
    expected_source_sha256 = _required_text(
        source, "expected_sha256", "spec.source"
    ).casefold()
    if re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256) is None:
        raise PreparationError("spec.source.expected_sha256 must be 64 lowercase hex digits")
    upstream_locator_value = source.get("upstream_locator")
    if upstream_locator_value is not None and (
        not isinstance(upstream_locator_value, str) or not upstream_locator_value.strip()
    ):
        raise PreparationError("spec.source.upstream_locator must be a non-empty string when set")
    upstream_locator = upstream_locator_value.strip() if upstream_locator_value else None
    path_text = _required_text(source, "path", "spec.source")
    source_path = Path(path_text)
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve()
    if not source_path.is_file():
        raise PreparationError(f"source file does not exist: {source_path}")
    encoding = str(source.get("encoding", "utf-8"))
    delimiter = str(source.get("delimiter", ","))
    if len(delimiter) != 1:
        raise PreparationError("spec.source.delimiter must be one character")

    entity_columns = _string_list(spec.get("entity_columns", []), "entity_columns")
    timestamp_value = spec.get("timestamp")
    if not isinstance(timestamp_value, Mapping):
        raise PreparationError("spec.timestamp must be an object")
    source_kind = _required_text(timestamp_value, "source_kind", "timestamp")
    if source_kind not in {"epoch", "datetime"}:
        raise PreparationError("timestamp.source_kind must be epoch or datetime")
    output_unit = _required_text(timestamp_value, "output_unit", "timestamp")
    if output_unit not in UNIX_UNIT_EXPONENT:
        raise PreparationError("timestamp.output_unit must be s, ms, us, or ns")
    source_unit: str | None = None
    if source_kind == "epoch":
        source_unit = _required_text(timestamp_value, "source_unit", "timestamp")
        if source_unit not in UNIX_UNIT_EXPONENT:
            raise PreparationError("timestamp.source_unit must be s, ms, us, or ns")
    timezone_name = _required_text(timestamp_value, "timezone", "timestamp")
    formats_value = timestamp_value.get("formats", ["iso8601"])
    formats = _string_list(formats_value, "timestamp.formats", allow_empty=False)
    if source_kind == "epoch" and formats != ("iso8601",):
        raise PreparationError("epoch timestamps do not use timestamp.formats")
    if source_kind == "epoch" and timezone_name != "UTC":
        raise PreparationError(
            "epoch timestamps are absolute Unix values; timestamp.timezone must be UTC"
        )
    ambiguous_fold = timestamp_value.get("ambiguous_fold")
    if ambiguous_fold not in {None, 0, 1}:
        raise PreparationError("timestamp.ambiguous_fold must be 0, 1, or omitted")
    duplicate_policy = _required_text(timestamp_value, "duplicate_policy", "timestamp")
    if duplicate_policy not in {"reject", "allow_preserve_order"}:
        raise PreparationError(
            "timestamp.duplicate_policy must be reject or allow_preserve_order"
        )
    timestamp = TimestampContract(
        source_name=_required_text(timestamp_value, "source_name", "timestamp"),
        output_name=str(timestamp_value.get("output_name", "timestamp_unix")).strip(),
        source_kind=source_kind,
        source_unit=source_unit,
        output_unit=output_unit,
        timezone_name=timezone_name,
        formats=formats,
        ambiguous_fold=ambiguous_fold,
        duplicate_policy=duplicate_policy,
    )
    if not timestamp.output_name:
        raise PreparationError("timestamp.output_name cannot be empty")

    raw_columns = spec.get("numeric_columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise PreparationError("numeric_columns must be a non-empty list")
    value_columns: list[NumericColumn] = []
    for index, value in enumerate(raw_columns):
        if not isinstance(value, Mapping):
            raise PreparationError(f"numeric_columns[{index}] must be an object")
        source_name = _required_text(value, "source_name", f"numeric_columns[{index}]")
        output_name = str(value.get("output_name", source_name)).strip()
        dtype = _required_text(value, "dtype", f"numeric_columns[{index}]")
        if not output_name:
            raise PreparationError(f"numeric_columns[{index}].output_name cannot be empty")
        if dtype not in SUPPORTED_DTYPES:
            raise PreparationError(
                f"numeric_columns[{index}].dtype must be one of {sorted(SUPPORTED_DTYPES)}"
            )
        if source_name == timestamp.source_name:
            raise PreparationError(
                "timestamp is included automatically; do not duplicate it in numeric_columns"
            )
        value_columns.append(NumericColumn(source_name, output_name, dtype))
    source_names = [timestamp.source_name, *(column.source_name for column in value_columns)]
    output_names = [timestamp.output_name, *(column.output_name for column in value_columns)]
    if len(source_names) != len(set(source_names)):
        raise PreparationError("numeric source column names must be unique")
    if len(output_names) != len(set(output_names)):
        raise PreparationError("numeric output column names must be unique")
    if set(entity_columns) & set(source_names):
        raise PreparationError("entity columns cannot also be numeric value columns")

    missing_values = spec.get("missing_tokens", list(DEFAULT_MISSING_TOKENS))
    if not isinstance(missing_values, list) or any(
        not isinstance(token, str) for token in missing_values
    ):
        raise PreparationError("missing_tokens must be a list of strings")
    missing_tokens = frozenset(token.casefold() for token in missing_values)
    if "nan" in missing_tokens or "+nan" in missing_tokens or "-nan" in missing_tokens:
        raise PreparationError("NaN is a floating-point value and cannot be a missing token")

    dataset_byte_cap = int(spec.get("dataset_byte_cap", MAX_DATASET_BYTES))
    block_byte_cap = int(spec.get("block_byte_cap", MAX_BLOCK_BYTES))
    if not 0 < dataset_byte_cap <= MAX_DATASET_BYTES:
        raise PreparationError(f"dataset_byte_cap must be in 1..{MAX_DATASET_BYTES}")
    if not 0 < block_byte_cap <= MAX_BLOCK_BYTES:
        raise PreparationError(f"block_byte_cap must be in 1..{MAX_BLOCK_BYTES}")
    for column in (
        NumericColumn(timestamp.source_name, timestamp.output_name, "int64"),
        *value_columns,
    ):
        max_rows_per_block(column.width, block_byte_cap)

    return PreparationContract(
        dataset_id=dataset_id,
        domain=domain,
        source_group=source_group,
        upstream_revision=upstream_revision,
        upstream_locator=upstream_locator,
        expected_source_sha256=expected_source_sha256,
        source_path=source_path,
        encoding=encoding,
        delimiter=delimiter,
        entity_columns=entity_columns,
        timestamp=timestamp,
        value_columns=tuple(value_columns),
        missing_tokens=missing_tokens,
        dataset_byte_cap=dataset_byte_cap,
        block_byte_cap=block_byte_cap,
        spec_sha256=_sha256_bytes(_canonical_json(spec)),
    )


def load_preparation_contract(path: Path) -> PreparationContract:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise PreparationError("preparation spec must be a JSON object")
    return _parse_contract(value, path.resolve().parent)


def _parse_int(text: str, dtype: str) -> bytes:
    if not text or any(character in text for character in ".eE"):
        raise PreparationError(f"invalid {dtype} token: {text!r}")
    try:
        value = int(text, 10)
        return struct.pack(SUPPORTED_DTYPES[dtype][1], value)
    except (ValueError, struct.error) as error:
        raise PreparationError(f"value outside {dtype}: {text!r}") from error


def _parse_float_bits(text: str, dtype: str) -> bytes | None:
    width = SUPPORTED_DTYPES[dtype][0]
    prefix = f"bits{width * 8}:"
    if not text.casefold().startswith(prefix):
        return None
    raw = text[len(prefix):]
    try:
        bits = int(raw, 0)
    except ValueError as error:
        raise PreparationError(f"invalid {prefix[:-1]} token: {text!r}") from error
    if not 0 <= bits < 1 << (8 * width):
        raise PreparationError(f"bit pattern outside {dtype}: {text!r}")
    return bits.to_bytes(width, "little")


def pack_numeric_token(token: str, dtype: str, missing_tokens: frozenset[str]) -> bytes | None:
    """Return little-endian value bytes, or None only for declared missing tokens."""
    text = token.strip()
    if text.casefold() in missing_tokens:
        return None
    if dtype.startswith(("int", "uint")):
        return _parse_int(text, dtype)
    exact_bits = _parse_float_bits(text, dtype)
    if exact_bits is not None:
        return exact_bits
    try:
        value = float(text)
        normalized = text.casefold()
        explicit_infinity = normalized in {
            "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"
        }
        if math.isinf(value) and not explicit_infinity:
            raise PreparationError(f"value outside {dtype}: {text!r}")
        return struct.pack(SUPPORTED_DTYPES[dtype][1], value)
    except PreparationError:
        raise
    except (ValueError, OverflowError, struct.error) as error:
        raise PreparationError(f"invalid {dtype} token: {text!r}") from error


def _convert_epoch_units(value: int, source_unit: str, output_unit: str) -> int:
    difference = UNIX_UNIT_EXPONENT[output_unit] - UNIX_UNIT_EXPONENT[source_unit]
    if difference >= 0:
        result = value * (10 ** difference)
    else:
        divisor = 10 ** (-difference)
        if value % divisor:
            raise PreparationError(
                f"timestamp {value} {source_unit} cannot be represented exactly in {output_unit}"
            )
        result = value // divisor
    if not -(1 << 63) <= result < (1 << 63):
        raise PreparationError("converted Unix timestamp is outside int64")
    return result


def _parse_datetime(text: str, formats: Sequence[str]) -> datetime:
    errors: list[str] = []
    for format_value in formats:
        try:
            if format_value == "iso8601":
                fraction_match = re.search(r"[\.,](\d+)(?=(?:Z|z|[+-]\d\d(?::?\d\d)?)?$)", text)
                if fraction_match and len(fraction_match.group(1)) > 6:
                    raise PreparationError(
                        "datetime timestamps finer than microseconds must be supplied as "
                        "integer epoch values with source_unit us or ns"
                    )
                normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
                return datetime.fromisoformat(normalized)
            return datetime.strptime(text, format_value)
        except PreparationError:
            raise
        except ValueError as error:
            errors.append(str(error))
    raise PreparationError(
        f"timestamp {text!r} does not match declared formats: {'; '.join(errors)}"
    )


def _localize_naive(value: datetime, contract: TimestampContract) -> datetime:
    if contract.timezone_name == "from_value":
        raise PreparationError("naive timestamp requires an explicit IANA timezone, not from_value")
    try:
        zone = ZoneInfo(contract.timezone_name)
    except ZoneInfoNotFoundError as error:
        raise PreparationError(f"unknown timestamp timezone: {contract.timezone_name}") from error
    candidates = [value.replace(tzinfo=zone, fold=fold) for fold in (0, 1)]
    valid = [
        candidate
        for candidate in candidates
        if candidate.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == value
    ]
    if not valid:
        raise PreparationError(f"nonexistent local timestamp: {value.isoformat()}")
    offsets = {candidate.utcoffset() for candidate in valid}
    if len(offsets) > 1 and contract.ambiguous_fold is None:
        raise PreparationError(
            f"ambiguous local timestamp requires ambiguous_fold: {value.isoformat()}"
        )
    fold = contract.ambiguous_fold or 0
    return value.replace(tzinfo=zone, fold=fold)


def parse_timestamp_token(token: str, contract: TimestampContract) -> int:
    text = token.strip()
    if not text:
        raise PreparationError("timestamp is missing; row-index substitution is forbidden")
    if contract.source_kind == "epoch":
        try:
            value = int(text, 10)
        except ValueError as error:
            raise PreparationError(f"invalid integer epoch timestamp: {text!r}") from error
        assert contract.source_unit is not None
        return _convert_epoch_units(value, contract.source_unit, contract.output_unit)

    parsed = _parse_datetime(text, contract.formats)
    if parsed.tzinfo is None:
        parsed = _localize_naive(parsed, contract)
    elif contract.timezone_name != "from_value":
        try:
            declared_zone = ZoneInfo(contract.timezone_name)
        except ZoneInfoNotFoundError as error:
            raise PreparationError(f"unknown timestamp timezone: {contract.timezone_name}") from error
        declared_offset = parsed.astimezone(declared_zone).utcoffset()
        if parsed.utcoffset() != declared_offset:
            raise PreparationError(
                f"timestamp offset does not match declared timezone {contract.timezone_name}: "
                f"{text!r}"
            )
    utc = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta: timedelta = utc - epoch
    microseconds = (
        delta.days * 86_400 * 1_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    return _convert_epoch_units(microseconds, "us", contract.output_unit)


def equal_prefix_allocation(counts: Mapping[tuple[str, ...], int], row_budget: int) -> dict[tuple[str, ...], int]:
    """Max-min fair row allocation; every selected interval is an entity prefix."""
    if row_budget < 0 or any(count < 0 for count in counts.values()):
        raise PreparationError("row counts and budget must be non-negative")
    ordered = sorted(counts)
    total = sum(counts.values())
    if row_budget >= total:
        return dict(counts)
    nonempty = [key for key in ordered if counts[key] > 0]
    if row_budget < len(nonempty):
        raise PreparationError(
            "dataset byte cap cannot select at least one row from every non-empty entity"
        )
    low, high = 0, max(counts.values(), default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(count, middle) for count in counts.values()) <= row_budget:
            low = middle
        else:
            high = middle - 1
    allocation = {key: min(counts[key], low) for key in ordered}
    remaining = row_budget - sum(allocation.values())
    for key in ordered:
        if remaining == 0:
            break
        if allocation[key] < counts[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("fair allocation failed to spend the row budget")
    return allocation


def max_rows_per_block(width: int, byte_cap: int) -> int:
    """Largest row count whose value payload plus null bitmap fits byte_cap."""
    if width <= 0 or byte_cap <= 0:
        raise PreparationError("block width and byte cap must be positive")
    low, high = 0, byte_cap // width
    while low < high:
        middle = (low + high + 1) // 2
        if middle * width + (middle + 7) // 8 <= byte_cap:
            low = middle
        else:
            high = middle - 1
    if low == 0:
        raise PreparationError(
            f"block_byte_cap {byte_cap} cannot hold one {width}-byte value and bitmap"
        )
    return low


def _entity_key(row: Mapping[str | None, str | list[str] | None], columns: Sequence[str]) -> tuple[str, ...]:
    if not columns:
        return ("__single_entity__",)
    values: list[str] = []
    for column in columns:
        value = row.get(column)
        if not isinstance(value, str) or not value.strip():
            raise PreparationError(f"entity column {column!r} is missing")
        values.append(value)
    return tuple(values)


def _entity_digest(key: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_json(list(key)))


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "column"
    return f"{normalized[:48]}-{_sha256_bytes(value.encode('utf-8'))[:16]}"


def _validate_header(
    fieldnames: Sequence[str] | None,
    contract: PreparationContract,
) -> list[str]:
    if not fieldnames or len(fieldnames) != len(set(fieldnames)):
        raise PreparationError("CSV header is empty or contains duplicate names")
    required = {
        *contract.entity_columns,
        contract.timestamp.source_name,
        *(column.source_name for column in contract.value_columns),
    }
    missing = required - set(fieldnames)
    if missing:
        raise PreparationError(f"CSV is missing required columns: {sorted(missing)}")
    return list(fieldnames)


def _iter_csv(contract: PreparationContract) -> Iterable[tuple[int, dict[str | None, Any]]]:
    with contract.source_path.open(encoding=contract.encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=contract.delimiter)
        _validate_header(reader.fieldnames, contract)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise PreparationError(f"CSV row {row_number} has more cells than the header")
            if any(value is None for value in row.values()):
                raise PreparationError(f"CSV row {row_number} has fewer cells than the header")
            yield row_number, row


def _source_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _audit_source(contract: PreparationContract) -> tuple[dict[tuple[str, ...], int], int]:
    counts: dict[tuple[str, ...], int] = {}
    previous_timestamp: dict[tuple[str, ...], int] = {}
    source_rows = 0
    for row_number, row in _iter_csv(contract):
        source_rows += 1
        key = _entity_key(row, contract.entity_columns)
        timestamp = parse_timestamp_token(str(row[contract.timestamp.source_name]), contract.timestamp)
        previous = previous_timestamp.get(key)
        if previous is not None and timestamp < previous:
            raise PreparationError(
                f"entity {key!r} is not chronological at CSV row {row_number}: "
                f"{timestamp} < {previous}"
            )
        if (
            previous is not None
            and timestamp == previous
            and contract.timestamp.duplicate_policy == "reject"
        ):
            raise PreparationError(
                f"entity {key!r} has a duplicate timestamp at CSV row {row_number}: "
                f"{timestamp}"
            )
        previous_timestamp[key] = timestamp
        for column in contract.value_columns:
            try:
                pack_numeric_token(str(row[column.source_name]), column.dtype, contract.missing_tokens)
            except PreparationError as error:
                raise PreparationError(
                    f"{contract.dataset_id}:{column.source_name}:row {row_number}: {error}"
                ) from error
        counts[key] = counts.get(key, 0) + 1
    if not source_rows:
        raise PreparationError("source CSV contains no data rows")
    return counts, source_rows


class _BlockBuilder:
    def __init__(
        self,
        *,
        column: NumericColumn,
        entity_key: tuple[str, ...],
        split: str,
        block_index: int,
        max_rows: int,
    ) -> None:
        self.column = column
        self.entity_key = entity_key
        self.split = split
        self.block_index = block_index
        self.max_rows = max_rows
        self.values = bytearray()
        self.bitmap = bytearray()
        self.row_count = 0
        self.valid_count = 0
        self.first_timestamp: int | None = None
        self.last_timestamp: int | None = None

    @property
    def full(self) -> bool:
        return self.row_count >= self.max_rows

    def add(self, packed: bytes | None, timestamp: int) -> None:
        if self.full:
            raise AssertionError("cannot append to a full block")
        bitmap_index, bit_index = divmod(self.row_count, 8)
        if bitmap_index == len(self.bitmap):
            self.bitmap.append(0)
        if packed is None:
            self.bitmap[bitmap_index] |= 1 << bit_index
        else:
            if len(packed) != self.column.width:
                raise AssertionError("packed value has the wrong width")
            self.values.extend(packed)
            self.valid_count += 1
        self.row_count += 1
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        self.last_timestamp = timestamp

    def write(self, root: Path) -> dict[str, Any]:
        entity_id = _entity_digest(self.entity_key)
        stem = (
            f"{self.split}__e-{entity_id}__c-{_safe_component(self.column.output_name)}"
            f"__b-{self.block_index:06d}"
        )
        value_relative = Path("blocks") / f"{stem}.values.bin"
        bitmap_relative = Path("blocks") / f"{stem}.missing.bitmap"
        value_path = root / value_relative
        bitmap_path = root / bitmap_relative
        value_path.write_bytes(self.values)
        bitmap_path.write_bytes(self.bitmap)
        return {
            "block_id": stem,
            "entity_id": entity_id,
            "split": self.split,
            "column": self.column.output_name,
            "source_column": self.column.source_name,
            "dtype": self.column.dtype,
            "byte_order": "little",
            "block_index": self.block_index,
            "row_count": self.row_count,
            "valid_count": self.valid_count,
            "missing_count": self.row_count - self.valid_count,
            "logical_slot_bytes": self.row_count * self.column.width,
            "original_numeric_bytes": self.valid_count * self.column.width,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "values_path": value_relative.as_posix(),
            "values_bytes": len(self.values),
            "values_sha256": _sha256_bytes(bytes(self.values)),
            "missing_bitmap_path": bitmap_relative.as_posix(),
            "missing_bitmap_bytes": len(self.bitmap),
            "missing_bitmap_sha256": _sha256_bytes(bytes(self.bitmap)),
            "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
            "prepared_block_bytes": len(self.values) + len(self.bitmap),
        }


def _flush_builder(
    builder: _BlockBuilder,
    root: Path,
    blocks: list[dict[str, Any]],
) -> None:
    if builder.row_count:
        blocks.append(builder.write(root))


def _manifest_fingerprint_basis(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "provenance_sha256"
    }


def prepare_dataset(
    contract: PreparationContract,
    output_dir: Path,
    *,
    cohort_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one dataset, atomically publishing a new output directory."""
    cohort_binding = (
        validate_contract_against_cohort(contract, cohort_manifest)
        if cohort_manifest is not None
        else None
    )
    split_contract = _split_contract_for_binding(cohort_binding)
    cohort_role = str(split_contract["cohort_role"])
    initial_signature = _source_signature(contract.source_path)
    source_sha256 = sha256_file(contract.source_path)
    if source_sha256 != contract.expected_source_sha256:
        raise PreparationError(
            "source sha256 does not match spec.source.expected_sha256: "
            f"expected={contract.expected_source_sha256}, actual={source_sha256}"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise PreparationError(f"output already exists; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    (temporary / "blocks").mkdir()

    try:
        counts, source_rows = _audit_source(contract)
        if _source_signature(contract.source_path) != initial_signature:
            raise PreparationError("source changed during the audit pass")
        row_budget = contract.dataset_byte_cap // contract.logical_row_bytes
        allocation = equal_prefix_allocation(counts, row_budget)
        if cohort_role == "train_validation":
            short_entities = {
                key: selected
                for key, selected in allocation.items()
                if selected < MIN_SELECTED_ROWS_PER_ENTITY
            }
            if short_entities:
                raise PreparationError(
                    "chronological 80/20 splitting requires at least five selected rows "
                    f"per entity; short entities={short_entities}"
                )
            train_counts = {
                key: allocation[key] * TRAIN_NUMERATOR // TRAIN_DENOMINATOR
                for key in allocation
            }
        else:
            train_counts = {key: 0 for key in allocation}

        selected_positions = {key: 0 for key in allocation}
        builders: dict[tuple[tuple[str, ...], str, str], _BlockBuilder] = {}
        next_block_index: dict[tuple[tuple[str, ...], str, str], int] = {}
        blocks: list[dict[str, Any]] = []
        selected_rows = 0

        for _row_number, row in _iter_csv(contract):
            key = _entity_key(row, contract.entity_columns)
            position = selected_positions[key]
            if position >= allocation[key]:
                continue
            timestamp = parse_timestamp_token(
                str(row[contract.timestamp.source_name]), contract.timestamp
            )
            split = (
                "test"
                if cohort_role == "test_candidate"
                else ("train" if position < train_counts[key] else "validation")
            )
            selected_positions[key] += 1
            selected_rows += 1

            packed_values: list[tuple[NumericColumn, bytes | None]] = [
                (
                    contract.numeric_columns[0],
                    struct.pack("<q", timestamp),
                )
            ]
            packed_values.extend(
                (
                    column,
                    pack_numeric_token(
                        str(row[column.source_name]), column.dtype, contract.missing_tokens
                    ),
                )
                for column in contract.value_columns
            )
            for column, packed in packed_values:
                stream_key = (key, split, column.output_name)
                builder = builders.get(stream_key)
                if builder is None:
                    block_index = next_block_index.get(stream_key, 0)
                    builder = _BlockBuilder(
                        column=column,
                        entity_key=key,
                        split=split,
                        block_index=block_index,
                        max_rows=max_rows_per_block(
                            column.width, contract.block_byte_cap
                        ),
                    )
                    builders[stream_key] = builder
                if builder.full:
                    _flush_builder(builder, temporary, blocks)
                    block_index = builder.block_index + 1
                    next_block_index[stream_key] = block_index
                    builder = _BlockBuilder(
                        column=column,
                        entity_key=key,
                        split=split,
                        block_index=block_index,
                        max_rows=max_rows_per_block(
                            column.width, contract.block_byte_cap
                        ),
                    )
                    builders[stream_key] = builder
                builder.add(packed, timestamp)

        for builder in builders.values():
            _flush_builder(builder, temporary, blocks)
        if selected_positions != allocation:
            raise PreparationError("source row counts changed between audit and materialization")
        if _source_signature(contract.source_path) != initial_signature:
            raise PreparationError("source changed during materialization")
        if sha256_file(contract.source_path) != source_sha256:
            raise PreparationError("source content changed during preparation")

        blocks.sort(key=lambda block: (
            block["entity_id"], block["split"], block["column"], block["block_index"]
        ))
        entities = []
        for key in sorted(allocation):
            selected = allocation[key]
            train = train_counts[key]
            validation = selected - train if cohort_role == "train_validation" else 0
            test = selected if cohort_role == "test_candidate" else 0
            entities.append({
                "entity_id": _entity_digest(key),
                "entity_values": list(key),
                "source_rows": counts[key],
                "selected_rows": selected,
                "selection": "chronological_prefix",
                "train_rows": train,
                "validation_rows": validation,
                "test_rows": test,
                "train_boundary_rule": split_contract["train_boundary_rule"],
            })

        numeric_columns = [
            {
                "source_name": column.source_name,
                "output_name": column.output_name,
                "dtype": column.dtype,
                "width_bytes": column.width,
                "role": "timestamp" if index == 0 else "value",
            }
            for index, column in enumerate(contract.numeric_columns)
        ]
        block_original_bytes = sum(block["original_numeric_bytes"] for block in blocks)
        block_slot_bytes = sum(block["logical_slot_bytes"] for block in blocks)
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "dataset_id": contract.dataset_id,
            "domain": contract.domain,
            "source_group": contract.source_group,
            "cohort": cohort_binding,
            "split_contract": split_contract,
            "source": {
                "path": str(contract.source_path),
                "format": "csv",
                "bytes": initial_signature[0],
                "sha256": source_sha256,
                "upstream_revision": contract.upstream_revision,
                "upstream_locator": contract.upstream_locator,
            },
            "preparation_spec_sha256": contract.spec_sha256,
            "input_contract": {
                "entity_columns": list(contract.entity_columns),
                "numeric_columns": numeric_columns,
                "timestamp": {
                    "source_name": contract.timestamp.source_name,
                    "source_kind": contract.timestamp.source_kind,
                    "source_unit": contract.timestamp.source_unit,
                    "output_name": contract.timestamp.output_name,
                    "output_unit": contract.timestamp.output_unit,
                    "timezone": contract.timestamp.timezone_name,
                    "formats": list(contract.timestamp.formats),
                    "ambiguous_fold": contract.timestamp.ambiguous_fold,
                    "duplicate_policy": contract.timestamp.duplicate_policy,
                    "invalid_policy": "fail_no_row_index_fallback",
                },
                "missing_tokens": sorted(contract.missing_tokens),
                "nan_policy": "retain_as_float_value_distinct_from_missing",
            },
            "selection": {
                "source_rows": source_rows,
                "selected_rows": selected_rows,
                "entity_balance": "max_min_fair_contiguous_prefix_rows",
                "logical_row_bytes": contract.logical_row_bytes,
                "dataset_byte_cap": contract.dataset_byte_cap,
                "dataset_byte_cap_semantics": (
                    "selected_fixed_width_numeric_slots_including_converted_timestamp_"
                    "before_null_compaction"
                ),
                "selected_logical_slot_bytes": selected_rows * contract.logical_row_bytes,
                "block_byte_cap": contract.block_byte_cap,
                "block_byte_cap_semantics": (
                    "value_payload_plus_missing_bitmap_per_column_block"
                ),
                "split": split_contract["partition"],
            },
            "entities": entities,
            "blocks": blocks,
            "totals": {
                "blocks": len(blocks),
                "logical_slot_bytes": block_slot_bytes,
                "original_numeric_bytes": block_original_bytes,
                "value_payload_bytes": sum(block["values_bytes"] for block in blocks),
                "missing_bitmap_bytes": sum(
                    block["missing_bitmap_bytes"] for block in blocks
                ),
                "prepared_block_bytes": sum(
                    block["prepared_block_bytes"] for block in blocks
                ),
            },
        }
        manifest["metric_contract"] = {
            "denominator_name": "B_raw",
            "denominator_bytes": block_original_bytes,
            "denominator_definition": (
                "sum(valid numeric values times declared original dtype width) after "
                "the explicit Unix timestamp conversion and before compression operators"
            ),
            "null_policy": (
                "null slots have no numeric payload and are excluded from B_raw; every "
                "required null bitmap byte is included in complete compressed bytes"
            ),
            "nan_policy": (
                "floating-point NaN is a present fixed-width numeric value included in B_raw"
            ),
            "compression_ratio": "complete_compressed_bytes / B_raw",
            "compression_time": "compression_elapsed_ns / B_raw",
            "decompression_time": "decompression_elapsed_ns / B_raw",
            "time_unit": "ns/byte",
            "optimization_direction": "all_three_lower_is_better",
            "zero_denominator_policy": "N/A",
        }
        manifest["provenance_sha256"] = _sha256_bytes(
            _canonical_json(_manifest_fingerprint_basis(manifest))
        )
        _atomic_json(temporary / "dataset_manifest.json", manifest)
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verified_child(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparationError(f"prepared manifest contains an unsafe path: {relative_text}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PreparationError(f"prepared path leaves the output directory: {relative_text}") from error
    return resolved


def verify_prepared_dataset(
    output_dir: Path,
    *,
    verify_source: bool = True,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Recompute block, aggregate, source, and provenance checks for prepared data."""
    root = output_dir.resolve()
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise PreparationError(f"missing prepared manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise PreparationError("prepared manifest has an unsupported format_version")
    if expected_role is not None and expected_role not in COHORT_ROLES:
        raise PreparationError(f"unsupported expected_role: {expected_role!r}")
    split_contract = manifest.get("split_contract")
    cohort_binding = manifest.get("cohort")
    if split_contract is None and expected_role is None:
        # Compatibility for previously prepared train/validation artifacts.
        # A formal role-bound call always requires the explicit contract, and
        # a legacy test_candidate artifact is never accepted by this fallback.
        manifest_role = "train_validation"
        if isinstance(cohort_binding, Mapping) and cohort_binding.get("role") != manifest_role:
            raise PreparationError(
                "legacy prepared manifest without split_contract is not train_validation"
            )
    else:
        if not isinstance(split_contract, Mapping):
            raise PreparationError("prepared manifest split_contract must be an object")
        manifest_role = _required_text(split_contract, "cohort_role", "split_contract")
        if manifest_role not in COHORT_ROLES:
            raise PreparationError(f"unsupported prepared cohort role: {manifest_role!r}")
        if dict(split_contract) != SPLIT_CONTRACTS[manifest_role]:
            raise PreparationError(
                f"prepared split_contract differs from the {manifest_role} contract"
            )
    if cohort_binding is not None:
        if not isinstance(cohort_binding, Mapping):
            raise PreparationError("prepared cohort binding must be an object or null")
        if cohort_binding.get("role") != manifest_role:
            raise PreparationError("prepared cohort role and split_contract role differ")
    if expected_role is not None:
        if manifest_role != expected_role:
            raise PreparationError(
                f"prepared role mismatch: expected {expected_role}, got {manifest_role}"
            )
        if not isinstance(cohort_binding, Mapping):
            raise PreparationError(
                "expected_role verification requires an explicit frozen cohort binding"
            )
    allowed_splits = tuple(SPLIT_CONTRACTS[manifest_role]["allowed_splits"])
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list):
        raise PreparationError("prepared manifest blocks must be a list")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise PreparationError("prepared manifest selection must be an object")
    block_byte_cap = int(selection.get("block_byte_cap", 0))
    if not 0 < block_byte_cap <= MAX_BLOCK_BYTES:
        raise PreparationError("prepared manifest has an invalid block_byte_cap")
    if selection.get("block_byte_cap_semantics") != (
        "value_payload_plus_missing_bitmap_per_column_block"
    ):
        raise PreparationError("prepared manifest has unsupported block cap semantics")
    dataset_byte_cap = int(selection.get("dataset_byte_cap", 0))
    if not 0 < dataset_byte_cap <= MAX_DATASET_BYTES:
        raise PreparationError("prepared manifest has an invalid dataset_byte_cap")
    if selection.get("dataset_byte_cap_semantics") != (
        "selected_fixed_width_numeric_slots_including_converted_timestamp_"
        "before_null_compaction"
    ):
        raise PreparationError("prepared manifest has unsupported dataset cap semantics")
    if selection.get("split") != SPLIT_CONTRACTS[manifest_role]["partition"]:
        raise PreparationError("prepared selection split rule differs from split_contract")

    totals = {
        "blocks": len(blocks),
        "logical_slot_bytes": 0,
        "original_numeric_bytes": 0,
        "value_payload_bytes": 0,
        "missing_bitmap_bytes": 0,
        "prepared_block_bytes": 0,
    }
    seen_block_ids: set[str] = set()
    seen_paths: set[str] = set()
    rows_by_stream: dict[tuple[str, str, str], int] = {}
    indexes_by_stream: dict[tuple[str, str, str], list[int]] = {}
    timestamp_blocks: dict[tuple[str, str], list[tuple[int, list[int]]]] = {}
    block_schemas: list[tuple[str, str, str, str]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise PreparationError("prepared block record must be an object")
        dtype = block.get("dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise PreparationError(f"prepared block has unsupported dtype: {dtype!r}")
        width = SUPPORTED_DTYPES[dtype][0]
        row_count = int(block["row_count"])
        valid_count = int(block["valid_count"])
        if not 0 <= valid_count <= row_count:
            raise PreparationError(f"invalid valid_count in block {block.get('block_id')}")
        missing_count = row_count - valid_count
        if int(block.get("missing_count", -1)) != missing_count:
            raise PreparationError(f"missing_count mismatch in {block.get('block_id')}")
        expected_values_bytes = valid_count * width
        expected_bitmap_bytes = (row_count + 7) // 8
        expected_prepared_bytes = expected_values_bytes + expected_bitmap_bytes
        if int(block["values_bytes"]) != expected_values_bytes:
            raise PreparationError(f"value byte count mismatch in {block.get('block_id')}")
        if int(block["missing_bitmap_bytes"]) != expected_bitmap_bytes:
            raise PreparationError(f"bitmap byte count mismatch in {block.get('block_id')}")
        if int(block["logical_slot_bytes"]) != row_count * width:
            raise PreparationError(f"logical slot byte count mismatch in {block.get('block_id')}")
        if int(block["original_numeric_bytes"]) != expected_values_bytes:
            raise PreparationError(f"original numeric byte count mismatch in {block.get('block_id')}")
        if int(block.get("prepared_block_bytes", -1)) != expected_prepared_bytes:
            raise PreparationError(f"prepared block byte count mismatch in {block.get('block_id')}")
        if expected_prepared_bytes > block_byte_cap:
            raise PreparationError(f"prepared block exceeds byte cap: {block.get('block_id')}")
        if block.get("missing_bitmap_semantics") != "one_bit_means_missing_lsb_first":
            raise PreparationError(f"unsupported bitmap semantics in {block.get('block_id')}")
        if block.get("byte_order") != "little":
            raise PreparationError(f"unsupported byte order in {block.get('block_id')}")
        block_id = _required_text(block, "block_id", "prepared block")
        if block_id in seen_block_ids:
            raise PreparationError(f"duplicate prepared block_id: {block_id}")
        seen_block_ids.add(block_id)

        file_bytes: dict[str, bytes] = {}
        for path_key, size_key, hash_key in (
            ("values_path", "values_bytes", "values_sha256"),
            ("missing_bitmap_path", "missing_bitmap_bytes", "missing_bitmap_sha256"),
        ):
            path = _verified_child(root, str(block[path_key]))
            relative_path = str(block[path_key])
            if relative_path in seen_paths:
                raise PreparationError(f"prepared block path is reused: {relative_path}")
            seen_paths.add(relative_path)
            if not path.is_file():
                raise PreparationError(f"missing prepared block file: {path}")
            if path.stat().st_size != int(block[size_key]):
                raise PreparationError(f"prepared file size mismatch: {path}")
            payload = path.read_bytes()
            if _sha256_bytes(payload) != block[hash_key]:
                raise PreparationError(f"prepared file hash mismatch: {path}")
            file_bytes[path_key] = payload

        bitmap = file_bytes["missing_bitmap_path"]
        bitmap_missing = sum(byte.bit_count() for byte in bitmap)
        if bitmap_missing != missing_count:
            raise PreparationError(f"bitmap missing count mismatch in {block_id}")
        if row_count % 8 and bitmap:
            unused_mask = ~((1 << (row_count % 8)) - 1) & 0xFF
            if bitmap[-1] & unused_mask:
                raise PreparationError(f"bitmap has nonzero unused bits in {block_id}")

        entity_id = _required_text(block, "entity_id", "prepared block")
        split = _required_text(block, "split", "prepared block")
        if split not in allowed_splits:
            raise PreparationError(f"invalid split in {block_id}: {split}")
        column = _required_text(block, "column", "prepared block")
        source_column = _required_text(block, "source_column", "prepared block")
        block_schemas.append((block_id, column, source_column, str(dtype)))
        stream_key = (entity_id, split, column)
        rows_by_stream[stream_key] = rows_by_stream.get(stream_key, 0) + row_count
        indexes_by_stream.setdefault(stream_key, []).append(int(block["block_index"]))

        input_contract = manifest.get("input_contract")
        timestamp_contract = (
            input_contract.get("timestamp") if isinstance(input_contract, Mapping) else None
        )
        if isinstance(timestamp_contract, Mapping) and column == timestamp_contract.get("output_name"):
            if dtype != "int64" or missing_count:
                raise PreparationError(f"timestamp block must be non-null int64: {block_id}")
            values = [
                value[0]
                for value in struct.iter_unpack("<q", file_bytes["values_path"])
            ]
            if values != sorted(values):
                raise PreparationError(f"timestamp block is not chronological: {block_id}")
            if values and (
                int(block["first_timestamp"]) != values[0]
                or int(block["last_timestamp"]) != values[-1]
            ):
                raise PreparationError(f"timestamp range mismatch in {block_id}")
            timestamp_blocks.setdefault((entity_id, split), []).append(
                (int(block["block_index"]), values)
            )

        totals["logical_slot_bytes"] += row_count * width
        totals["original_numeric_bytes"] += expected_values_bytes
        totals["value_payload_bytes"] += expected_values_bytes
        totals["missing_bitmap_bytes"] += expected_bitmap_bytes
        totals["prepared_block_bytes"] += expected_prepared_bytes
    for stream_key, indexes in indexes_by_stream.items():
        ordered_indexes = sorted(indexes)
        if ordered_indexes != list(range(len(ordered_indexes))):
            raise PreparationError(
                "block indexes are not contiguous for "
                f"{stream_key[0]}/{stream_key[1]}/{stream_key[2]}"
            )
    if manifest.get("totals") != totals:
        raise PreparationError(
            f"prepared totals mismatch: recorded={manifest.get('totals')}, computed={totals}"
        )

    input_contract = manifest.get("input_contract")
    if not isinstance(input_contract, Mapping):
        raise PreparationError("prepared input_contract must be an object")
    numeric_columns = input_contract.get("numeric_columns")
    if not isinstance(numeric_columns, list) or not numeric_columns:
        raise PreparationError("prepared numeric_columns must be a non-empty list")
    output_names: list[str] = []
    column_contracts: dict[str, tuple[str, str]] = {}
    logical_row_bytes = 0
    for index, column in enumerate(numeric_columns):
        if not isinstance(column, Mapping):
            raise PreparationError(f"prepared numeric_columns[{index}] must be an object")
        output_name = _required_text(column, "output_name", f"numeric_columns[{index}]")
        dtype = _required_text(column, "dtype", f"numeric_columns[{index}]")
        if dtype not in SUPPORTED_DTYPES:
            raise PreparationError(f"unsupported prepared dtype: {dtype}")
        if int(column.get("width_bytes", -1)) != SUPPORTED_DTYPES[dtype][0]:
            raise PreparationError(f"prepared width mismatch for {output_name}")
        output_names.append(output_name)
        column_contracts[output_name] = (
            _required_text(column, "source_name", f"numeric_columns[{index}]"),
            dtype,
        )
        logical_row_bytes += SUPPORTED_DTYPES[dtype][0]
    if len(output_names) != len(set(output_names)):
        raise PreparationError("prepared numeric output names are not unique")
    if int(selection.get("logical_row_bytes", -1)) != logical_row_bytes:
        raise PreparationError("prepared logical_row_bytes mismatch")
    for block_id, output_name, source_name, dtype in block_schemas:
        expected = column_contracts.get(output_name)
        if expected is None:
            raise PreparationError(f"prepared block uses unknown column: {block_id}")
        if (source_name, dtype) != expected:
            raise PreparationError(f"prepared block schema mismatch: {block_id}")

    entities = manifest.get("entities")
    if not isinstance(entities, list) or not entities:
        raise PreparationError("prepared entities must be a non-empty list")
    seen_entities: set[str] = set()
    selected_rows = 0
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise PreparationError("prepared entity record must be an object")
        entity_id = _required_text(entity, "entity_id", "prepared entity")
        if entity_id in seen_entities:
            raise PreparationError(f"duplicate prepared entity_id: {entity_id}")
        seen_entities.add(entity_id)
        entity_selected = int(entity["selected_rows"])
        train_rows = int(entity["train_rows"])
        validation_rows = int(entity["validation_rows"])
        test_rows = int(entity.get("test_rows", 0 if split_contract is None else -1))
        if entity_selected <= 0:
            raise PreparationError(f"prepared entity has no selected rows: {entity_id}")
        if manifest_role == "train_validation":
            if entity_selected < MIN_SELECTED_ROWS_PER_ENTITY:
                raise PreparationError(f"prepared entity has too few selected rows: {entity_id}")
            if train_rows != entity_selected * TRAIN_NUMERATOR // TRAIN_DENOMINATOR:
                raise PreparationError(f"prepared train boundary mismatch: {entity_id}")
            if train_rows + validation_rows != entity_selected or test_rows != 0:
                raise PreparationError(f"prepared split row count mismatch: {entity_id}")
            if entity.get("train_boundary_rule") != SPLIT_CONTRACTS[manifest_role]["train_boundary_rule"]:
                raise PreparationError(f"prepared train boundary rule mismatch: {entity_id}")
            expected_rows_by_split = (
                ("train", train_rows),
                ("validation", validation_rows),
            )
        else:
            if train_rows != 0 or validation_rows != 0 or test_rows != entity_selected:
                raise PreparationError(f"prepared test split row count mismatch: {entity_id}")
            if entity.get("train_boundary_rule") is not None:
                raise PreparationError(f"test entity must not define a train boundary: {entity_id}")
            expected_rows_by_split = (("test", test_rows),)
        selected_rows += entity_selected
        for split, expected_rows in expected_rows_by_split:
            for output_name in output_names:
                actual_rows = rows_by_stream.get((entity_id, split, output_name), 0)
                if actual_rows != expected_rows:
                    raise PreparationError(
                        f"prepared stream row count mismatch for {entity_id}/{split}/{output_name}: "
                        f"{actual_rows} != {expected_rows}"
                    )

        timestamp_contract = input_contract.get("timestamp")
        if not isinstance(timestamp_contract, Mapping):
            raise PreparationError("prepared timestamp contract must be an object")
        duplicate_policy = timestamp_contract.get("duplicate_policy")
        split_values: dict[str, list[int]] = {}
        for split in allowed_splits:
            ordered_blocks = sorted(timestamp_blocks.get((entity_id, split), []))
            if [index for index, _values in ordered_blocks] != list(range(len(ordered_blocks))):
                raise PreparationError(
                    f"timestamp block indexes are not contiguous for {entity_id}/{split}"
                )
            values = [value for _index, block_values in ordered_blocks for value in block_values]
            if values != sorted(values):
                raise PreparationError(
                    f"timestamps are not chronological for {entity_id}/{split}"
                )
            if duplicate_policy == "reject" and len(values) != len(set(values)):
                raise PreparationError(f"duplicate prepared timestamps for {entity_id}/{split}")
            split_values[split] = values
        if manifest_role == "train_validation" and split_values["train"] and split_values["validation"]:
            if split_values["train"][-1] > split_values["validation"][0]:
                raise PreparationError(f"train/validation chronology overlaps for {entity_id}")
            if (
                duplicate_policy == "reject"
                and split_values["train"][-1] == split_values["validation"][0]
            ):
                raise PreparationError(f"train/validation timestamp duplicates for {entity_id}")

    for entity_id, _split, output_name in rows_by_stream:
        if entity_id not in seen_entities:
            raise PreparationError(f"prepared block uses unknown entity_id: {entity_id}")
        if output_name not in column_contracts:
            raise PreparationError(f"prepared block uses unknown output column: {output_name}")

    if int(selection.get("selected_rows", -1)) != selected_rows:
        raise PreparationError("prepared selected_rows mismatch")
    if int(selection.get("selected_logical_slot_bytes", -1)) != selected_rows * logical_row_bytes:
        raise PreparationError("prepared selected logical byte count mismatch")
    if int(selection.get("selected_logical_slot_bytes", -1)) > dataset_byte_cap:
        raise PreparationError("prepared dataset exceeds its numeric byte cap")

    metric_contract = manifest.get("metric_contract")
    if not isinstance(metric_contract, Mapping):
        raise PreparationError("prepared metric_contract must be an object")
    if metric_contract.get("denominator_name") != "B_raw":
        raise PreparationError("prepared metric denominator must be B_raw")
    if int(metric_contract.get("denominator_bytes", -1)) != totals["original_numeric_bytes"]:
        raise PreparationError("prepared metric denominator byte count mismatch")
    if metric_contract.get("compression_ratio") != "complete_compressed_bytes / B_raw":
        raise PreparationError("prepared compression ratio contract mismatch")
    if metric_contract.get("compression_time") != "compression_elapsed_ns / B_raw":
        raise PreparationError("prepared compression time contract mismatch")
    if metric_contract.get("decompression_time") != "decompression_elapsed_ns / B_raw":
        raise PreparationError("prepared decompression time contract mismatch")

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise PreparationError("prepared manifest source must be an object")
    if verify_source:
        if source.get("format") == "tsfile_collection":
            source_files = source.get("files")
            if not isinstance(source_files, list) or not source_files:
                raise PreparationError(
                    "prepared TsFile collection source files must be a non-empty list"
                )
            identity_basis = []
            total_source_bytes = 0
            seen_paths: set[str] = set()
            seen_repo_paths: set[str] = set()
            for index, item in enumerate(source_files):
                if not isinstance(item, Mapping):
                    raise PreparationError(
                        f"prepared collection source[{index}] must be an object"
                    )
                source_path_text = _required_text(
                    item, "path", f"prepared collection source[{index}]"
                )
                repo_path = _required_text(
                    item, "repo_path", f"prepared collection source[{index}]"
                )
                if source_path_text in seen_paths or repo_path in seen_repo_paths:
                    raise PreparationError(
                        "prepared collection reuses a source path or repo_path"
                    )
                seen_paths.add(source_path_text)
                seen_repo_paths.add(repo_path)
                source_path = Path(source_path_text)
                if not source_path.is_file():
                    raise PreparationError(
                        f"prepared collection source is unavailable: {source_path}"
                    )
                expected_bytes = int(item["bytes"])
                if source_path.stat().st_size != expected_bytes:
                    raise PreparationError(
                        f"prepared collection source size changed: {source_path}"
                    )
                expected_sha256 = _required_text(
                    item, "sha256", f"prepared collection source[{index}]"
                )
                if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
                    raise PreparationError(
                        f"prepared collection source hash is invalid: {source_path}"
                    )
                if sha256_file(source_path) != expected_sha256:
                    raise PreparationError(
                        f"prepared collection source hash changed: {source_path}"
                    )
                total_source_bytes += expected_bytes
                identity_basis.append({
                    "repo_path": repo_path,
                    "sha256": expected_sha256,
                })
            if int(source.get("file_count", -1)) != len(source_files):
                raise PreparationError("prepared collection source file_count mismatch")
            if int(source.get("bytes", -1)) != total_source_bytes:
                raise PreparationError("prepared collection source byte total mismatch")
            collection_sha256 = _sha256_bytes(_canonical_json(identity_basis))
            if source.get("sha256") != collection_sha256:
                raise PreparationError("prepared collection source identity mismatch")
        else:
            source_path = Path(str(source.get("path", "")))
            if not source_path.is_file():
                raise PreparationError(f"prepared source is unavailable: {source_path}")
            if source_path.stat().st_size != int(source["bytes"]):
                raise PreparationError("prepared source size changed")
            if sha256_file(source_path) != source["sha256"]:
                raise PreparationError("prepared source hash changed")

    fingerprint = _sha256_bytes(
        _canonical_json(_manifest_fingerprint_basis(manifest))
    )
    if fingerprint != manifest.get("provenance_sha256"):
        raise PreparationError("prepared provenance fingerprint mismatch")
    return {
        "dataset_id": manifest["dataset_id"],
        "cohort_role": manifest_role,
        "blocks": totals["blocks"],
        "original_numeric_bytes": totals["original_numeric_bytes"],
        "provenance_sha256": fingerprint,
        "source_verified": verify_source,
    }


def _load_contract_from_json(path: Path) -> PreparationContract:
    return load_preparation_contract(path.resolve())


def _load_prepared_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "dataset_manifest.json" if path.is_dir() else path
    with manifest_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PreparationError(f"prepared manifest must be an object: {manifest_path}")
    return value


def _load_and_verify_prepared_manifest(path: Path) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    expected_manifest = root / "dataset_manifest.json"
    if not path.is_dir() and path.resolve() != expected_manifest.resolve():
        raise PreparationError(
            "validate-run manifest files must be named dataset_manifest.json"
        )
    verify_prepared_dataset(root)
    return _load_prepared_manifest(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cohort_parser = subparsers.add_parser("validate-cohort")
    cohort_parser.add_argument("--manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--spec", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_MANIFEST)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--skip-source", action="store_true")
    verify_parser.add_argument("--expected-role", choices=sorted(COHORT_ROLES))
    run_parser = subparsers.add_parser("validate-run")
    run_parser.add_argument("--manifest", type=Path, action="append", required=True)
    run_parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_MANIFEST)
    run_parser.add_argument("--allow-partial-train", action="store_true")
    audit_parser = subparsers.add_parser("audit-source")
    audit_parser.add_argument("--input", type=Path, required=True)
    audit_parser.add_argument("--format", choices=("auto", "csv", "tsfile"), default="auto")
    audit_parser.add_argument("--expected-sha256")
    audit_parser.add_argument("--encoding", default="utf-8")
    audit_parser.add_argument("--delimiter", default=",")
    args = parser.parse_args(argv)

    if args.command == "validate-cohort":
        with args.manifest.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(json.dumps(validate_cohort_manifest(manifest), sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(
            verify_prepared_dataset(
                args.input,
                verify_source=not args.skip_source,
                expected_role=args.expected_role,
            ),
            sort_keys=True,
        ))
        return 0
    if args.command == "validate-run":
        cohort = load_cohort_manifest(args.cohort)
        manifests = [_load_and_verify_prepared_manifest(path) for path in args.manifest]
        print(json.dumps(
            validate_prepared_cohort_manifests(
                manifests,
                cohort,
                require_complete_train=not args.allow_partial_train,
            ),
            sort_keys=True,
        ))
        return 0
    if args.command == "audit-source":
        print(json.dumps(
            audit_local_source(
                args.input,
                source_format=args.format,
                expected_sha256=args.expected_sha256,
                encoding=args.encoding,
                delimiter=args.delimiter,
            ),
            sort_keys=True,
        ))
        return 0
    contract = _load_contract_from_json(args.spec)
    cohort = load_cohort_manifest(args.cohort)
    result = prepare_dataset(contract, args.output, cohort_manifest=cohort)
    print(json.dumps({
        "dataset_id": result["dataset_id"],
        "output": str(args.output.resolve()),
        "blocks": result["totals"]["blocks"],
        "original_numeric_bytes": result["totals"]["original_numeric_bytes"],
        "provenance_sha256": result["provenance_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
