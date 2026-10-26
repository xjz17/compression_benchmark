#!/usr/bin/env python3
"""Build the scope-bound timestamp policy for the 160 project179 THULab datasets.

The schema audit, rather than README prose alone, is authoritative for the
stored Time unit.  Exact-UTC admissions retain an event-instant claim only when
the audit classified the frozen dataset README as explicit UTC evidence.
Timezone-unspecified and synthetic axes are admitted only as stored epoch
coordinates; they explicitly make no event-UTC-instant claim.  Mixed or
surrogate axes remain blocked pending a dataset-specific adapter.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid
from typing import Any, Mapping, Sequence


SCRIPTS = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPTS.parent
DEFAULT_SCOPE = (
    REPOSITORY_ROOT
    / "benchmark/reference/datasets/project_dataset_scope_20260928.json"
)
DEFAULT_AUDIT = (
    REPOSITORY_ROOT
    / "analysis_outputs/project179_numeric_utc_20260928/THULAB_TSFILE_SCHEMA_AUDIT.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "benchmark/reference/datasets/project179_thulab_utc_policy_20260928.json"
)
DEFAULT_TEMPORAL_SAMPLES_DIR = (
    REPOSITORY_ROOT
    / "analysis_outputs/project179_numeric_utc_20260928/temporal_samples_by_dataset"
)
DEFAULT_MIXED_AXIS_ADAPTER_AUDIT = (
    REPOSITORY_ROOT
    / "analysis_outputs/project179_numeric_utc_20260928/MIXED_AXIS_ADAPTER_AUDIT.json"
)

POLICY_SCHEMA = "project179-thulab-utc-policy-v1"
SCOPE_SCHEMA = "web-compression-project-dataset-scope-v1"
AUDIT_SCHEMA = "web-compression-thulab-tsfile-schema-audit-v1"
GENERATOR_VERSION = "project179-thulab-utc-policy-generator-v1"
TEMPORAL_SAMPLE_SCHEMA = "project179-thulab-temporal-string-samples-v1"
MIXED_AXIS_ADAPTER_AUDIT_SCHEMA = "project179-thulab-mixed-axis-adapter-audit-v1"
EXPECTED_DATASET_COUNT = 160
EXPECTED_TEMPORAL_CANDIDATE_DEFINITION_COUNT = 158
OUTPUT_UNIT = "us"
OUTPUT_TYPE = "int64"
TABLE_NAME = "__single_table__"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

EXACT_UTC_CLASS = "explicit_utc_in_dataset_readme"
STORED_COORDINATE_CLASSES = frozenset({
    "integer_epoch_or_civil_axis_timezone_unspecified",
    "synthetic_start_plus_frequency_timezone_unspecified",
})
BLOCKED_CLASS = "mixed_or_surrogate_axis_requires_dataset_adapter"
KNOWN_CLASSES = frozenset({EXACT_UTC_CLASS, *STORED_COORDINATE_CLASSES, BLOCKED_CLASS})
MIXED_ADAPTER_DATASETS = frozenset({
    "finance1k",
    "fulus_lyd_rates_2025",
    "knowable",
})
OBSERVED_UNIT_TO_POLICY_UNIT = {
    "epoch_s": "s",
    "epoch_ms": "ms",
    "epoch_us": "us",
}

# A same-named field has two representations in the frozen DMI repository:
# data/data.tsfile stores reference_time as an INT64 epoch-second FIELD, while
# data/training-matrix.tsfile stores it as an ISO-8601 STRING TAG.  The numeric
# form is admitted only through this exact path/content/table/schema binding;
# the dataset-level STRING rule continues to cover only the second source.
SOURCE_BOUND_TEMPORAL_FIELDS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "dmi-aarhus-weather-data": ({
        "source_path": "data/data.tsfile",
        "source_sha256": (
            "d40d9fdca20f018eb48676d9730033c439ab99e1a821d7b94cce01876b3a1ff6"
        ),
        "table_name": "dmi_aarhus_data",
        "source_schema_sha256": (
            "a06fde8814d9018899ce0bc948c238096cc95d80089731810192dd32bd7ca878"
        ),
        "source_name": "reference_time",
        "source_type": "INT64",
        "source_category": "FIELD",
        "source_unit": "s",
        "output_name": "reference_time__timestamp_unix_us",
        "output_type": "int64",
        "output_unit": "us",
        "parse_kind": "integer_epoch",
        "timezone": "UTC",
        "event_utc_instant_claim": True,
        "semantic_kind": "unix_epoch_instant",
        "timezone_basis": "upstream_encoded_epoch",
        "evidence": [
            "Frozen source data/data.tsfile sha256="
            "d40d9fdca20f018eb48676d9730033c439ab99e1a821d7b94cce01876b3a1ff6 "
            "has table dmi_aarhus_data with schema sha256="
            "a06fde8814d9018899ce0bc948c238096cc95d80089731810192dd32bd7ca878; "
            "reference_time is INT64 FIELD and the frozen DMI policy records epoch seconds.",
            "Read-only samples satisfy Time-reference_time=lead_time_hours*3600, "
            "consistent with the README UTC issue-time semantics; the formal validator "
            "must independently scan all source values, NULLs, and conversion bounds.",
        ],
    },),
}

# The per-file schema scan for the frozen revisions found these timestamp-like
# strings in TAG columns.  Every other candidate name below is a FIELD.  The
# materializer uses TAG values to partition entities first, then emits the
# parsed numeric value; the original string is never retained in the payload.
TAG_TEMPORAL_FIELDS = frozenset({"start_timestamp", "reference_time"})

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("integer_epoch_days", re.compile(r"^[+-]?\d+$")),
    ("date_only_utc_midnight_coordinate", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("clock_only_drop", re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")),
    (
        "embedded_offset_fractional",
        re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2}$"),
    ),
    (
        "embedded_offset",
        re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"),
    ),
    (
        "explicit_z_fractional",
        re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$"),
    ),
    (
        "explicit_z",
        re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
    ),
    (
        "naive_assume_utc_coordinate",
        re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$"),
    ),
)

_FORMAT_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "integer_epoch_days": {
        "parse_kind": "integer_epoch",
        "source_unit": "d",
        "timezone": "UTC",
        "semantic_kind": "unix_epoch_day_coordinate",
        "event_utc_instant_claim": False,
        "timezone_basis": "utc_midnight_coordinate_convention",
    },
    "date_only_utc_midnight_coordinate": {
        "parse_kind": "strptime",
        "format": "%Y-%m-%d",
        "timezone": "UTC",
        "semantic_kind": "civil_date_utc_midnight_coordinate",
        "event_utc_instant_claim": False,
        "timezone_basis": "utc_midnight_coordinate_convention",
    },
    "embedded_offset_fractional": {
        "parse_kind": "strptime",
        "format": "%Y-%m-%d %H:%M:%S.%f%z",
        "timezone": "embedded_offset",
        "semantic_kind": "unix_epoch_instant",
        "event_utc_instant_claim": True,
        "timezone_basis": "embedded_numeric_offset",
    },
    "embedded_offset": {
        "parse_kind": "strptime",
        "format": "%Y-%m-%d %H:%M:%S%z",
        "timezone": "embedded_offset",
        "semantic_kind": "unix_epoch_instant",
        "event_utc_instant_claim": True,
        "timezone_basis": "embedded_numeric_offset",
    },
    "explicit_z_fractional": {
        "parse_kind": "strptime",
        "format": "%Y-%m-%dT%H:%M:%S.%fZ",
        "timezone": "UTC",
        "semantic_kind": "unix_epoch_instant",
        "event_utc_instant_claim": True,
        "timezone_basis": "explicit_z_suffix",
    },
    "explicit_z": {
        "parse_kind": "strptime",
        "format": "%Y-%m-%dT%H:%M:%SZ",
        "timezone": "UTC",
        "semantic_kind": "unix_epoch_instant",
        "event_utc_instant_claim": True,
        "timezone_basis": "explicit_z_suffix",
    },
    "naive_assume_utc_coordinate": {
        "parse_kind": "strptime",
        "format": "%Y-%m-%d %H:%M:%S",
        "timezone": "UTC",
        "semantic_kind": "naive_civil_utc_coordinate",
        "event_utc_instant_claim": False,
        "timezone_basis": "project_utc_coordinate_convention",
    },
}


class PolicyBuildError(ValueError):
    """Raised when a frozen policy input violates the generation contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PolicyBuildError(f"JSON document must be an object: {path}")
    return value


def _classify_temporal_samples(values: Sequence[str], context: str) -> str:
    if not values:
        raise PolicyBuildError(f"{context} has no sampled values")
    classes: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise PolicyBuildError(f"{context} samples must be non-empty strings")
        matched = next((name for name, pattern in _PATTERNS if pattern.fullmatch(value)), None)
        if matched is None:
            raise PolicyBuildError(f"unsupported temporal sample in {context}: {value!r}")
        classes.add(matched)
    if len(classes) != 1:
        raise PolicyBuildError(f"mixed temporal sample formats in {context}: {sorted(classes)}")
    return next(iter(classes))


def _validate_temporal_parse_contract(
    values: Sequence[str], format_class: str, context: str
) -> None:
    contract = _FORMAT_CONTRACTS[format_class]
    if contract["parse_kind"] == "integer_epoch":
        if contract.get("source_unit") != "d":
            raise AssertionError("only frozen Unix-day string coordinates are expected here")
        for value in values:
            epoch_us = int(value) * 86_400_000_000
            if not INT64_MIN <= epoch_us <= INT64_MAX:
                raise PolicyBuildError(f"Unix-day sample exceeds int64 us in {context}: {value}")
        return
    datetime_format = str(contract["format"])
    for value in values:
        try:
            datetime.strptime(value, datetime_format)
        except ValueError as error:
            raise PolicyBuildError(
                f"sample does not satisfy parse contract in {context}: {value!r}"
            ) from error


def _load_temporal_samples(
    samples_dir: Path,
    *,
    audit_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not samples_dir.is_dir():
        raise PolicyBuildError(f"temporal sample directory is absent: {samples_dir}")
    rows: dict[str, Mapping[str, Any]] = {}
    bindings: list[Mapping[str, Any]] = []
    for path in sorted(samples_dir.glob("*.json")):
        raw = _load_object(path)
        if raw.get("schema") != TEMPORAL_SAMPLE_SCHEMA:
            raise PolicyBuildError(f"unsupported temporal sample schema: {path}")
        if raw.get("source_audit_sha256") != audit_sha256:
            raise PolicyBuildError(f"temporal sample is not bound to the schema audit: {path}")
        datasets = raw.get("datasets")
        if not isinstance(datasets, Mapping) or len(datasets) != 1:
            raise PolicyBuildError(f"temporal sample must contain one dataset: {path}")
        dataset_id, record = next(iter(datasets.items()))
        if not isinstance(dataset_id, str) or not isinstance(record, Mapping):
            raise PolicyBuildError(f"invalid temporal sample dataset record: {path}")
        if path.stem != dataset_id:
            raise PolicyBuildError(
                f"temporal sample filename/dataset mismatch: {path.stem!r} != {dataset_id!r}"
            )
        if dataset_id in rows:
            raise PolicyBuildError(f"duplicate temporal sample dataset: {dataset_id}")
        distinct = record.get("distinct_samples")
        if not isinstance(distinct, Mapping) or not distinct:
            raise PolicyBuildError(f"temporal sample has no distinct_samples: {path}")
        rows[dataset_id] = record
        bindings.append({"path": path.name, "sha256": _sha256(path)})
    return rows, bindings


def _temporal_rules(
    dataset_id: str,
    audit_row: Mapping[str, Any],
    sample_row: Mapping[str, Any] | None,
    *,
    sample_binding_by_name: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], Counter[str]]:
    candidates = audit_row.get("timestamp_string_candidates", {})
    if not isinstance(candidates, Mapping):
        raise PolicyBuildError(f"timestamp candidates for {dataset_id} must be an object")
    if not candidates:
        if sample_row is not None:
            raise PolicyBuildError(f"unexpected temporal samples for {dataset_id}")
        return [], [], Counter(), Counter()
    if sample_row is None:
        raise PolicyBuildError(f"missing temporal samples for {dataset_id}")
    distinct = sample_row.get("distinct_samples")
    if not isinstance(distinct, Mapping):
        raise PolicyBuildError(f"invalid distinct_samples for {dataset_id}")
    if set(distinct) != set(candidates):
        raise PolicyBuildError(
            f"temporal sample/candidate columns differ for {dataset_id}: "
            f"samples={sorted(distinct)}, candidates={sorted(candidates)}"
        )
    sample_name = f"{dataset_id}.json"
    sample_sha256 = sample_binding_by_name.get(sample_name)
    if sample_sha256 is None:
        raise PolicyBuildError(f"missing temporal sample binding for {dataset_id}")

    rules: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    converted_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    for source_name in sorted(candidates):
        count = candidates[source_name]
        if not isinstance(count, int) or count <= 0:
            raise PolicyBuildError(f"invalid candidate definition count for {dataset_id}.{source_name}")
        values = distinct[source_name]
        if not isinstance(values, list):
            raise PolicyBuildError(f"samples for {dataset_id}.{source_name} must be a list")
        format_class = _classify_temporal_samples(
            values, f"{dataset_id}.{source_name}"
        )
        source_category = "TAG" if source_name in TAG_TEMPORAL_FIELDS else "FIELD"
        sample_evidence = (
            f"Frozen temporal sample {sample_name} sha256={sample_sha256}: "
            f"{source_name} matched format_class={format_class} across "
            f"{len(values)} distinct sampled value(s)."
        )
        if format_class == "clock_only_drop":
            dropped.append({
                "source_name": source_name,
                "source_type": "STRING",
                "source_category": source_category,
                "format_class": format_class,
                "decision": "drop_clock_only_without_calendar_date",
                "reason": (
                    "A clock-of-day string has no independent Unix instant; it is removed "
                    "instead of attaching an invented date or rollover convention."
                ),
                "candidate_definition_count": count,
                "evidence": [sample_evidence],
            })
            dropped_counts[format_class] += count
            continue

        contract = _FORMAT_CONTRACTS[format_class]
        _validate_temporal_parse_contract(
            values, format_class, f"{dataset_id}.{source_name}"
        )
        rule = {
            "source_name": source_name,
            "source_type": "STRING",
            "source_category": source_category,
            "output_name": f"{source_name}__timestamp_unix_us",
            "output_type": "int64",
            "output_unit": "us",
            "format_class": format_class,
            **contract,
            "candidate_definition_count": count,
            "applicability": (
                "each frozen source file whose table schema contains the matching "
                "name, STRING type, and source_category; absence in other heterogeneous "
                "source files is allowed"
            ),
            "evidence": [sample_evidence],
        }
        rules.append(rule)
        converted_counts[format_class] += count
    return rules, dropped, converted_counts, dropped_counts


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise PolicyBuildError(f"{context}.{key} must be a non-empty string")
    return result.strip()


def _relative_input_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        # Tests and explicit callers may supply frozen inputs outside the repo.
        # The basename is deterministic and does not leak a machine-local path.
        return path.name


def _scope_rows(scope: Mapping[str, Any]) -> tuple[str, dict[str, Mapping[str, Any]]]:
    if scope.get("schema") != SCOPE_SCHEMA:
        raise PolicyBuildError(f"scope schema must be {SCOPE_SCHEMA!r}")
    scope_id = _required_text(scope, "scope_id", "scope")
    values = scope.get("timestamp_datasets")
    if not isinstance(values, list):
        raise PolicyBuildError("scope.timestamp_datasets must be a list")
    expected_count = scope.get("timestamp_dataset_count")
    if expected_count != EXPECTED_DATASET_COUNT or len(values) != EXPECTED_DATASET_COUNT:
        raise PolicyBuildError(
            f"scope must contain exactly {EXPECTED_DATASET_COUNT} timestamp datasets"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        context = f"scope.timestamp_datasets[{index}]"
        if not isinstance(value, Mapping):
            raise PolicyBuildError(f"{context} must be an object")
        dataset_id = _required_text(value, "id", context)
        repo_id = _required_text(value, "repo_id", context)
        revision = _required_text(value, "revision", context)
        if dataset_id in result:
            raise PolicyBuildError(f"duplicate scope dataset_id: {dataset_id}")
        if repo_id != f"THULab/{dataset_id}":
            raise PolicyBuildError(f"unexpected repository identity for {dataset_id}")
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise PolicyBuildError(f"invalid frozen revision for {dataset_id}: {revision}")
        result[dataset_id] = value
    return scope_id, result


def _audit_rows(audit: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise PolicyBuildError(f"schema audit must use {AUDIT_SCHEMA!r}")
    values = audit.get("datasets")
    if not isinstance(values, list):
        raise PolicyBuildError("schema audit datasets must be a list")
    if audit.get("dataset_count") != EXPECTED_DATASET_COUNT or len(values) != EXPECTED_DATASET_COUNT:
        raise PolicyBuildError(
            f"schema audit must contain exactly {EXPECTED_DATASET_COUNT} datasets"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        context = f"audit.datasets[{index}]"
        if not isinstance(value, Mapping):
            raise PolicyBuildError(f"{context} must be an object")
        dataset_id = _required_text(value, "dataset_id", context)
        if dataset_id in result:
            raise PolicyBuildError(f"duplicate audit dataset_id: {dataset_id}")
        evidence_class = _required_text(value, "utc_evidence_class", context)
        if evidence_class not in KNOWN_CLASSES:
            raise PolicyBuildError(
                f"unsupported utc_evidence_class for {dataset_id}: {evidence_class}"
            )
        if value.get("time_axis_is_int64") is not True:
            raise PolicyBuildError(f"{dataset_id} Time axis is not audited as INT64")
        result[dataset_id] = value
    return result


def _evidence_strings(
    dataset_id: str,
    audit_row: Mapping[str, Any],
    *,
    audit_sha256: str,
) -> list[str]:
    observed_unit = _required_text(
        audit_row, "observed_time_axis_unit", f"audit dataset {dataset_id}"
    )
    evidence_class = _required_text(
        audit_row, "utc_evidence_class", f"audit dataset {dataset_id}"
    )
    evidence = [
        (
            f"Frozen schema audit sha256={audit_sha256}: dataset_id={dataset_id}, "
            f"observed_time_axis_unit={observed_unit}, time_axis_is_int64=true."
        ),
        (
            f"Frozen schema audit classification: dataset_id={dataset_id}, "
            f"utc_evidence_class={evidence_class}."
        ),
    ]
    readme_evidence = audit_row.get("readme_evidence", [])
    if not isinstance(readme_evidence, list):
        raise PolicyBuildError(f"audit README evidence for {dataset_id} must be a list")
    for index, item in enumerate(readme_evidence):
        if not isinstance(item, Mapping):
            raise PolicyBuildError(
                f"audit README evidence {dataset_id}[{index}] must be an object"
            )
        line = item.get("line")
        text = item.get("text")
        if not isinstance(line, int) or line <= 0 or not isinstance(text, str) or not text.strip():
            raise PolicyBuildError(
                f"invalid README evidence entry for {dataset_id} at index {index}"
            )
        evidence.append(f"Frozen README.md line {line}: {text.strip()}")
    discrepancy = audit_row.get("timestamp_unit_contract_discrepancy")
    if discrepancy is not None:
        if not isinstance(discrepancy, str) or not discrepancy.strip():
            raise PolicyBuildError(f"invalid timestamp unit discrepancy for {dataset_id}")
        evidence.append(
            "Observed-value unit overrides README unit prose for this frozen policy: "
            + discrepancy.strip()
        )
    return evidence


def _mixed_axis_adapter_rows(
    path: Path,
    *,
    scope_id: str,
    scope_sha256: str,
    scope_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    resolved = path.resolve()
    raw = _load_object(resolved)
    if raw.get("schema") != MIXED_AXIS_ADAPTER_AUDIT_SCHEMA:
        raise PolicyBuildError("unsupported mixed-axis adapter audit schema")
    if raw.get("scope_id") != scope_id or raw.get("scope_sha256") != scope_sha256:
        raise PolicyBuildError("mixed-axis adapter audit is not bound to the frozen scope")
    rows = raw.get("datasets")
    if not isinstance(rows, list):
        raise PolicyBuildError("mixed-axis adapter audit datasets must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        context = f"mixed-axis adapter audit datasets[{index}]"
        if not isinstance(row, Mapping):
            raise PolicyBuildError(f"{context} must be an object")
        dataset_id = _required_text(row, "dataset_id", context)
        if dataset_id in by_id or dataset_id not in MIXED_ADAPTER_DATASETS:
            raise PolicyBuildError(f"unexpected or duplicate mixed adapter dataset: {dataset_id}")
        scope_row = scope_by_id.get(dataset_id)
        if scope_row is None:
            raise PolicyBuildError(f"mixed adapter dataset is outside scope: {dataset_id}")
        if (
            row.get("repo_id") != scope_row.get("repo_id")
            or row.get("revision") != scope_row.get("revision")
        ):
            raise PolicyBuildError(f"mixed adapter identity mismatch for {dataset_id}")
        if row.get("recommended_decision") != "admit_dataset_adapter":
            raise PolicyBuildError(f"mixed adapter decision not admitted for {dataset_id}")
        receipt = row.get("source_receipt")
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(receipt.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256"))) is None
            or not isinstance(receipt.get("tsfile_count"), int)
            or int(receipt.get("tsfile_count", 0)) <= 0
        ):
            raise PolicyBuildError(f"invalid frozen source receipt binding for {dataset_id}")
        adapter = row.get("policy_adapter")
        if not isinstance(adapter, Mapping):
            raise PolicyBuildError(f"missing policy adapter for {dataset_id}")
        by_id[dataset_id] = row
    if set(by_id) != MIXED_ADAPTER_DATASETS:
        raise PolicyBuildError(
            "mixed-axis adapter dataset set differs: "
            f"expected={sorted(MIXED_ADAPTER_DATASETS)}, actual={sorted(by_id)}"
        )
    return by_id, _sha256(resolved)


def _dataset_policy(
    dataset_id: str,
    scope_row: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    *,
    audit_sha256: str,
    temporal_string_fields: Sequence[Mapping[str, Any]],
    source_bound_temporal_fields: Sequence[Mapping[str, Any]],
    dropped_timestamp_like_string_fields: Sequence[Mapping[str, Any]],
    mixed_axis_adapter_row: Mapping[str, Any] | None,
    mixed_axis_adapter_audit_sha256: str,
) -> dict[str, Any]:
    evidence_class = _required_text(
        audit_row, "utc_evidence_class", f"audit dataset {dataset_id}"
    )
    observed_unit = _required_text(
        audit_row, "observed_time_axis_unit", f"audit dataset {dataset_id}"
    )
    evidence = _evidence_strings(dataset_id, audit_row, audit_sha256=audit_sha256)
    base: dict[str, Any] = {
        "dataset_id": dataset_id,
        "repo_id": _required_text(scope_row, "repo_id", f"scope dataset {dataset_id}"),
        "revision": _required_text(scope_row, "revision", f"scope dataset {dataset_id}"),
        "temporal_string_fields": list(temporal_string_fields),
        "source_bound_temporal_fields": list(source_bound_temporal_fields),
        "dropped_timestamp_like_string_fields": list(dropped_timestamp_like_string_fields),
    }

    if evidence_class == BLOCKED_CLASS:
        if mixed_axis_adapter_row is not None:
            receipt = mixed_axis_adapter_row["source_receipt"]
            evidence.extend([
                "Frozen mixed-axis adapter audit "
                f"sha256={mixed_axis_adapter_audit_sha256}: "
                f"dataset_id={dataset_id}, exact_rows={mixed_axis_adapter_row['exact_rows']}.",
                "Frozen source receipt binding: "
                f"path={receipt['path']}, sha256={receipt['sha256']}, "
                f"tsfile_count={receipt['tsfile_count']}, "
                f"tsfile_bytes={receipt['tsfile_bytes']}.",
            ])
            base.update({
                "decision": "admit_dataset_adapter",
                "model": "table",
                "table_name": TABLE_NAME,
                "reason": (
                    "A scope/revision/receipt-bound full scan supplies explicit per-table "
                    "timestamp and exclusion/filter rules for this mixed physical Time axis."
                ),
                "mixed_axis_adapter": dict(mixed_axis_adapter_row["policy_adapter"]),
                "evidence": evidence,
            })
            return base
        base.update({
            "decision": "block",
            "reason": (
                "The frozen Time axis mixes epoch values with row-index or surrogate values; "
                "a dataset-specific adapter and independent validation are required before admission."
            ),
            "evidence": evidence,
        })
        return base

    source_unit = OBSERVED_UNIT_TO_POLICY_UNIT.get(observed_unit)
    if source_unit is None:
        raise PolicyBuildError(
            f"admitted dataset {dataset_id} has unsupported observed Time unit {observed_unit!r}"
        )
    base.update({
        "model": "table",
        "table_name": TABLE_NAME,
    })
    if evidence_class == EXACT_UTC_CLASS:
        base.update({
            "decision": "admit_exact_utc",
            "reason": (
                "The frozen audit records explicit UTC evidence; the observed stored Time unit "
                "is converted exactly to signed int64 Unix epoch microseconds."
            ),
            "timestamp": {
                "source_column": "Time",
                "source_unit": source_unit,
                "output_unit": OUTPUT_UNIT,
                "output_column": "timestamp_unix_us",
                "timezone": "UTC",
                "semantic_kind": "unix_epoch_instant",
                "source_unit_verified": True,
                "utc_instant_verified": True,
                "timezone_inferred": False,
            },
            "evidence": evidence,
        })
    elif evidence_class in STORED_COORDINATE_CLASSES:
        base.update({
            "decision": "admit_stored_epoch_coordinate",
            "reason": (
                "The frozen Time axis has an observed epoch-scale unit but lacks sufficient "
                "timezone evidence for an event-UTC-instant claim; preserve it exactly as a "
                "stored epoch coordinate in microseconds."
            ),
            "timestamp": {
                "source_column": "Time",
                "source_unit": source_unit,
                "output_unit": OUTPUT_UNIT,
                "output_column": "timestamp_unix_us",
                "semantic_kind": "stored_epoch_coordinate",
                "source_unit_verified": True,
                "event_utc_instant_claim": False,
                "timezone_basis": "upstream_encoded_epoch",
                "timezone_inferred": False,
            },
            "evidence": evidence,
        })
    else:  # Defensive: _audit_rows already rejects unknown classes.
        raise PolicyBuildError(f"unsupported evidence class for {dataset_id}: {evidence_class}")
    return base


def build_policy(
    *,
    scope_path: Path,
    audit_path: Path,
    temporal_samples_dir: Path = DEFAULT_TEMPORAL_SAMPLES_DIR,
    mixed_axis_adapter_audit_path: Path = DEFAULT_MIXED_AXIS_ADAPTER_AUDIT,
    repository_root: Path = REPOSITORY_ROOT,
) -> Mapping[str, Any]:
    scope_path = scope_path.resolve()
    audit_path = audit_path.resolve()
    scope = _load_object(scope_path)
    audit = _load_object(audit_path)
    scope_id, scope_by_id = _scope_rows(scope)
    audit_by_id = _audit_rows(audit)
    if set(scope_by_id) != set(audit_by_id):
        missing_audit = sorted(set(scope_by_id) - set(audit_by_id))
        outside_scope = sorted(set(audit_by_id) - set(scope_by_id))
        raise PolicyBuildError(
            "scope/audit dataset identities differ: "
            f"missing_audit={missing_audit}, outside_scope={outside_scope}"
        )

    scope_sha256 = _sha256(scope_path)
    audit_sha256 = _sha256(audit_path)
    mixed_adapter_rows, mixed_adapter_audit_sha256 = _mixed_axis_adapter_rows(
        mixed_axis_adapter_audit_path,
        scope_id=scope_id,
        scope_sha256=scope_sha256,
        scope_by_id=scope_by_id,
    )
    sample_rows, sample_bindings = _load_temporal_samples(
        temporal_samples_dir.resolve(), audit_sha256=audit_sha256
    )
    sample_binding_by_name = {
        str(row["path"]): str(row["sha256"]) for row in sample_bindings
    }
    candidate_dataset_ids = {
        dataset_id
        for dataset_id, row in audit_by_id.items()
        if row.get("timestamp_string_candidates")
    }
    if set(sample_rows) != candidate_dataset_ids:
        raise PolicyBuildError(
            "temporal sample/audit dataset identities differ: "
            f"missing_samples={sorted(candidate_dataset_ids - set(sample_rows))}, "
            f"outside_audit={sorted(set(sample_rows) - candidate_dataset_ids)}"
        )
    rules_by_id: dict[str, list[dict[str, Any]]] = {}
    dropped_by_id: dict[str, list[dict[str, Any]]] = {}
    converted_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    for dataset_id in sorted(audit_by_id):
        rules, dropped, converted, dropped_temporal = _temporal_rules(
            dataset_id,
            audit_by_id[dataset_id],
            sample_rows.get(dataset_id),
            sample_binding_by_name=sample_binding_by_name,
        )
        rules_by_id[dataset_id] = rules
        dropped_by_id[dataset_id] = dropped
        converted_counts.update(converted)
        dropped_counts.update(dropped_temporal)

    for dataset_id, adapter_row in mixed_adapter_rows.items():
        overrides = adapter_row.get("temporal_string_overrides", {})
        if not isinstance(overrides, Mapping):
            raise PolicyBuildError(
                f"mixed adapter temporal overrides must be an object for {dataset_id}"
            )
        rules_by_name = {
            str(rule["source_name"]): rule for rule in rules_by_id[dataset_id]
        }
        for source_name, override in overrides.items():
            if not isinstance(source_name, str) or not isinstance(override, Mapping):
                raise PolicyBuildError(
                    f"invalid temporal override for {dataset_id}/{source_name}"
                )
            rule = rules_by_name.get(source_name)
            if rule is None:
                raise PolicyBuildError(
                    f"temporal override names an absent rule: {dataset_id}/{source_name}"
                )
            allowed = {
                "format",
                "accepted_formats",
                "format_class",
                "event_utc_instant_claim",
                "semantic_kind",
                "timezone_basis",
            }
            unknown = set(override) - allowed - {"additional_evidence"}
            if unknown:
                raise PolicyBuildError(
                    f"unsupported temporal override keys for {dataset_id}/{source_name}: "
                    f"{sorted(unknown)}"
                )
            for key in allowed:
                if key in override:
                    rule[key] = override[key]
            additional_evidence = override.get("additional_evidence")
            if not isinstance(additional_evidence, str) or not additional_evidence.strip():
                raise PolicyBuildError(
                    f"temporal override lacks additional evidence: {dataset_id}/{source_name}"
                )
            rule["evidence"] = [*rule["evidence"], additional_evidence.strip()]

    audited_candidate_count = audit.get("timestamp_string_candidate_definition_count")
    if audited_candidate_count != EXPECTED_TEMPORAL_CANDIDATE_DEFINITION_COUNT:
        raise PolicyBuildError(
            "schema audit timestamp candidate count changed: "
            f"{audited_candidate_count!r} != {EXPECTED_TEMPORAL_CANDIDATE_DEFINITION_COUNT}"
        )
    if sum(converted_counts.values()) + sum(dropped_counts.values()) != audited_candidate_count:
        raise PolicyBuildError("temporal candidate coverage does not close")
    datasets = [
        _dataset_policy(
            dataset_id,
            scope_by_id[dataset_id],
            audit_by_id[dataset_id],
            audit_sha256=audit_sha256,
            temporal_string_fields=rules_by_id[dataset_id],
            source_bound_temporal_fields=SOURCE_BOUND_TEMPORAL_FIELDS.get(dataset_id, ()),
            dropped_timestamp_like_string_fields=dropped_by_id[dataset_id],
            mixed_axis_adapter_row=mixed_adapter_rows.get(dataset_id),
            mixed_axis_adapter_audit_sha256=mixed_adapter_audit_sha256,
        )
        for dataset_id in sorted(scope_by_id)
    ]
    decision_counts = {
        decision: sum(row["decision"] == decision for row in datasets)
        for decision in (
            "admit_exact_utc",
            "admit_stored_epoch_coordinate",
            "admit_dataset_adapter",
            "block",
        )
    }
    if sum(decision_counts.values()) != EXPECTED_DATASET_COUNT:
        raise AssertionError("internal decision count mismatch")

    return {
        "schema": POLICY_SCHEMA,
        "generated_date": str(audit.get("audit_date", scope.get("effective_date", "2026-09-28"))),
        "generator": {
            "version": GENERATOR_VERSION,
            "path": "scripts/build_project179_thulab_utc_policy_v1.py",
        },
        "scope_id": scope_id,
        "scope_sha256": scope_sha256,
        "scope_timestamp_dataset_count": EXPECTED_DATASET_COUNT,
        "scope_effective_date": scope.get("effective_date"),
        "schema_audit": {
            "schema": AUDIT_SCHEMA,
            "audit_date": audit.get("audit_date"),
            "path": _relative_input_path(audit_path, repository_root),
            "sha256": audit_sha256,
        },
        "mixed_axis_adapter_audit": {
            "schema": MIXED_AXIS_ADAPTER_AUDIT_SCHEMA,
            "path": _relative_input_path(
                mixed_axis_adapter_audit_path.resolve(), repository_root
            ),
            "sha256": mixed_adapter_audit_sha256,
            "dataset_count": len(mixed_adapter_rows),
        },
        "temporal_string_evidence": {
            "schema": TEMPORAL_SAMPLE_SCHEMA,
            "directory": _relative_input_path(
                temporal_samples_dir.resolve(), repository_root
            ),
            "file_count": len(sample_bindings),
            "files": sample_bindings,
        },
        "temporal_string_coverage": {
            "candidate_definition_count": audited_candidate_count,
            "candidate_dataset_count": len(candidate_dataset_ids),
            "candidate_rule_count": sum(len(rows) for rows in rules_by_id.values())
            + sum(len(rows) for rows in dropped_by_id.values()),
            "converted_rule_count": sum(len(rows) for rows in rules_by_id.values()),
            "converted_definition_count": sum(converted_counts.values()),
            "dropped_clock_only_rule_count": sum(
                len(rows) for rows in dropped_by_id.values()
            ),
            "dropped_clock_only_definition_count": sum(dropped_counts.values()),
            "converted_definition_counts_by_format_class": dict(sorted(converted_counts.items())),
            "dropped_definition_counts_by_format_class": dict(sorted(dropped_counts.items())),
            "tag_definition_count": sum(
                int(rule["candidate_definition_count"])
                for rules in rules_by_id.values()
                for rule in rules
                if rule["source_category"] == "TAG"
            ),
            "field_definition_count": sum(
                int(rule["candidate_definition_count"])
                for rules in rules_by_id.values()
                for rule in rules
                if rule["source_category"] == "FIELD"
            ),
            "ordinary_string_contract": (
                "STRING/TEXT columns absent from temporal_string_fields are dropped; "
                "TAG strings may partition entities but are not retained as payload"
            ),
            "heterogeneous_schema_contract": (
                "rules are a dataset-level union and apply independently only to frozen "
                "source-file schemas containing the declared name/type/category"
            ),
        },
        "default_output_type": OUTPUT_TYPE,
        "default_output_unit": OUTPUT_UNIT,
        "output_contract": {
            "column": "timestamp_unix_us",
            "integer_type": "INT64",
            "unit": "Unix epoch microseconds",
            "conversion": "exact integer scaling only; no precision downsampling",
            "stored_coordinate_boundary": (
                "admit_stored_epoch_coordinate preserves the upstream encoded epoch coordinate "
                "and does not claim that it is a verified event UTC instant"
            ),
            "mixed_axis_adapter_boundary": (
                "admit_dataset_adapter requires an exact table/file-count match and applies "
                "only its receipt-bound primary timestamp, whole-file exclusions, and row filters"
            ),
        },
        "input_bindings": {
            "scope": {
                "path": _relative_input_path(scope_path, repository_root),
                "sha256": scope_sha256,
            },
            "schema_audit": {
                "path": _relative_input_path(audit_path, repository_root),
                "sha256": audit_sha256,
            },
            "temporal_samples": {
                "directory": _relative_input_path(
                    temporal_samples_dir.resolve(), repository_root
                ),
                "file_count": len(sample_bindings),
                "files": sample_bindings,
            },
            "mixed_axis_adapter_audit": {
                "path": _relative_input_path(
                    mixed_axis_adapter_audit_path.resolve(), repository_root
                ),
                "sha256": mixed_adapter_audit_sha256,
            },
        },
        "decision_counts": decision_counts,
        "datasets": datasets,
    }


def policy_bytes(policy: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_policy(path: Path, policy: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(policy_bytes(policy))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--temporal-samples-dir",
        type=Path,
        default=DEFAULT_TEMPORAL_SAMPLES_DIR,
    )
    parser.add_argument(
        "--mixed-axis-adapter-audit",
        type=Path,
        default=DEFAULT_MIXED_AXIS_ADAPTER_AUDIT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output already equals the deterministic generated bytes",
    )
    args = parser.parse_args(argv)
    try:
        policy = build_policy(
            scope_path=args.scope,
            audit_path=args.audit,
            temporal_samples_dir=args.temporal_samples_dir,
            mixed_axis_adapter_audit_path=args.mixed_axis_adapter_audit,
            repository_root=args.repository_root,
        )
        expected = policy_bytes(policy)
        if args.check:
            if not args.output.is_file() or args.output.read_bytes() != expected:
                raise PolicyBuildError(
                    f"generated policy is missing or stale: {args.output.resolve()}"
                )
        else:
            write_policy(args.output, policy)
    except (OSError, json.JSONDecodeError, PolicyBuildError) as error:
        print(f"PROJECT179_THULAB_UTC_POLICY_BLOCKER: {error}", file=sys.stderr)
        return 2
    summary = {
        "output": str(args.output.resolve()),
        "bytes": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
        "decision_counts": policy["decision_counts"],
        "temporal_string_coverage": policy["temporal_string_coverage"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
