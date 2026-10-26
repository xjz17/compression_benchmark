#!/usr/bin/env python3
"""Materialize the frozen project179 THULab mirror as numeric-only streams.

This program deliberately separates *storage eligibility* from *UTC semantic
admission*.  A physical ``Time`` coordinate or explicitly listed temporal
STRING/TEXT column is emitted as a signed int64 Unix-microsecond coordinate
only when a scope/revision/receipt-bound policy supplies its exact conversion
and semantic label.  Event-instant claims remain distinct from conservative
stored/civil coordinate labels.  A missing or negative policy creates a
blocked receipt; it never falls back to a row index, guessed unit, guessed
timezone, or unlisted civil-time convention.

Only table-model INT32, INT64, FLOAT, and DOUBLE FIELD columns are retained.
TAG columns partition entity streams and are removed from the payload, except
that an explicitly admitted temporal TAG is both retained in the partition and
emitted as its parsed int64 value stream.  Each selected stream is complete
(timestamp plus one retained value column), and the dataset cap retains the
longest deterministic whole-stream prefix in frozen schema order.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import operator
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import struct
import sys
import tempfile
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from prepare_weighted_grpo_timeseries_v1 import (
    PreparationError,
    _atomic_json,
    _canonical_json,
    _entity_digest,
    _sha256_bytes,
    sha256_file,
)
from prepare_weighted_grpo_tsfile_v1 import (
    TsFileValueColumn,
    _load_tsfile_module,
    _pack_value,
    _table_schema_columns,
    _type_name,
    _typed_entity_component,
    _validate_tsfile_container,
)


FORMAT_VERSION = "project179-thulab-numeric-materialization-v1"
COLLECTION_VERSION = "project179-thulab-numeric-collection-v1"
POLICY_VERSION = "project179-thulab-utc-policy-v1"
RECEIPT_VERSION = "project179-thulab-numeric-receipt-v1"
PLANNING_RECEIPT_VERSION = "project179-thulab-sharded-planning-receipt-v1"
DEFAULT_DATASET_BYTE_CAP = 1 << 30
DEFAULT_CHUNK_BYTE_CAP = 64 << 20
SPOOL_BUCKET_STREAMS = 4096
ENTITY_COUNT_CACHE_LIMIT = 65536
ENTITY_LOOKUP_CACHE_LIMIT = 65536
DEFAULT_METADATA_COMPRESSED_BYTE_CAP = 1 << 30
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SUPPORTED_FIELDS = {
    "INT32": ("int32", 4),
    "INT64": ("int64", 8),
    "FLOAT": ("float32", 4),
    "DOUBLE": ("float64", 8),
}
KNOWN_NON_NUMERIC_FIELDS = frozenset({"STRING", "TEXT", "BLOB", "BOOLEAN", "DATE"})
SOURCE_UNITS = frozenset({"d", "s", "ms", "us", "ns"})
UNIT_EXPONENT = {"s": 0, "ms": 3, "us": 6, "ns": 9}
OUTPUT_UNITS = frozenset(UNIT_EXPONENT)
STORAGE_LAYOUTS = frozenset({"legacy_v1", "sharded_v1"})
DEFAULT_STORAGE_LAYOUT = "legacy_v1"
ALLOWED_POLICY_DECISIONS = frozenset({
    "admit_exact_utc",
    "admit_stored_epoch_coordinate",
    "admit_dataset_adapter",
    "block",
})
STREAM_INDEX_VERSION = "project179-thulab-stream-index-v1"
ENTITY_DICTIONARY_SEMANTICS = (
    "metadata_only_typed_TAG_partition_provenance_not_numeric_payload"
)
PRODUCER_PATH = Path(__file__).resolve()
HELPER_TIMESERIES_PATH = PRODUCER_PATH.with_name("prepare_weighted_grpo_timeseries_v1.py")
HELPER_TSFILE_PATH = PRODUCER_PATH.with_name("prepare_weighted_grpo_tsfile_v1.py")


@dataclass(frozen=True)
class ScopeRow:
    dataset_id: str
    repo_id: str
    revision: str


@dataclass(frozen=True)
class TimestampPolicy:
    dataset_id: str
    repo_id: str
    revision: str
    decision: str
    reason: str
    table_name: str | None
    source_column: str | None
    source_unit: str | None
    output_unit: str | None
    output_column: str | None
    temporal_string_fields: tuple["TemporalStringField", ...]
    source_bound_temporal_fields: tuple["SourceBoundTemporalField", ...]
    dropped_timestamp_like_string_fields: tuple["DroppedTimestampLikeField", ...]
    mixed_axis_adapter_id: str | None
    source_rules: tuple["SourceRule", ...]
    evidence: tuple[str, ...]
    policy_sha256: str


@dataclass(frozen=True)
class FieldSpec:
    name: str
    tsfile_type: str
    output_name: str | None = None
    temporal: "TemporalStringField | SourceBoundTemporalField | None" = None

    @property
    def dtype(self) -> str:
        if self.temporal is not None:
            return "int64"
        return SUPPORTED_FIELDS[self.tsfile_type][0]

    @property
    def width(self) -> int:
        if self.temporal is not None:
            return 8
        return SUPPORTED_FIELDS[self.tsfile_type][1]

    @property
    def packer(self) -> TsFileValueColumn:
        if self.temporal is not None:
            raise AssertionError("temporal strings do not use native numeric packing")
        return TsFileValueColumn(self.name, self.name, self.tsfile_type)

    @property
    def emitted_name(self) -> str:
        return self.output_name or self.name


@dataclass(frozen=True)
class TemporalStringField:
    source_name: str
    source_type: str
    source_category: str
    output_name: str
    parse_kind: str
    datetime_format: str | None
    accepted_datetime_formats: tuple[str, ...]
    timezone_name: str
    source_unit: str | None
    ambiguous_fold: int | None
    event_utc_instant_claim: bool
    semantic_kind: str
    timezone_basis: str
    format_class: str
    applicability: str
    candidate_definition_count: int
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class SourceBoundTemporalField:
    """An integer temporal value admitted only for one frozen source schema.

    Unlike ``TemporalStringField`` rules, which are a dataset-level union over
    heterogeneous source files, this rule is content- and schema-addressed.
    Every binding component must match before the field is converted.  This is
    intentionally strict so a same-named numeric column in any other source is
    never assigned time semantics by inference.
    """

    source_path: str
    source_sha256: str
    table_name: str
    source_schema_sha256: str
    source_name: str
    source_type: str
    source_category: str
    source_unit: str
    output_name: str
    output_unit: str
    parse_kind: str
    timezone_name: str
    event_utc_instant_claim: bool
    semantic_kind: str
    timezone_basis: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class DroppedTimestampLikeField:
    source_name: str
    source_type: str
    source_category: str
    decision: str
    format_class: str
    reason: str
    applicability: str | None
    candidate_definition_count: int
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RowFilterCondition:
    source_name: str
    source_type: str
    source_category: str
    operator: str
    value: int | None
    drop_reason: str


@dataclass(frozen=True)
class RowFilter:
    conditions: tuple[RowFilterCondition, ...]
    expected_input_rows: int
    expected_output_rows: int
    expected_drop_rows: int
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class SourceRule:
    rule_id: str
    table_names: tuple[str, ...]
    action: str
    expected_source_file_count: int
    reason: str
    evidence: tuple[str, ...]
    source_column: str | None
    source_unit: str | None
    output_unit: str | None
    output_column: str | None
    event_utc_instant_claim: bool | None
    utc_instant_verified: bool | None
    semantic_kind: str | None
    timezone_basis: str | None
    timezone_inferred: bool | None
    row_filter: RowFilter | None


@dataclass(frozen=True)
class SourcePlan:
    source_path: str
    path: Path
    expected_sha256: str
    table_name: str
    schema_sha256: str
    tag_names: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    primary_temporal: TemporalStringField | None
    dropped_columns: tuple[Mapping[str, Any], ...]
    source_action: str
    source_action_reason: str | None
    adapter_rule_id: str | None
    timestamp_source_column: str | None
    timestamp_source_unit: str | None
    timestamp_output_unit: str | None
    timestamp_output_column: str | None
    timestamp_event_utc_instant_claim: bool | None
    timestamp_semantic_kind: str | None
    timestamp_timezone_basis: str | None
    timestamp_timezone_inferred: bool | None
    row_filter: RowFilter | None


@dataclass(frozen=True)
class DatasetBinding:
    row: ScopeRow
    source_receipt: Mapping[str, Any]
    source_receipt_sha256: str
    source_files: tuple[tuple[str, Path, int, str], ...]
    policy: TimestampPolicy | None
    scope_id: str
    scope_sha256: str
    policy_file_sha256: str
    dataset_byte_cap: int
    chunk_byte_cap: int
    metadata_compressed_byte_cap: int
    storage_layout: str
    producer_sha256: str
    helper_timeseries_sha256: str
    helper_tsfile_sha256: str

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(_canonical_json({
            "format_version": FORMAT_VERSION,
            "scope_id": self.scope_id,
            "scope_sha256": self.scope_sha256,
            "dataset_id": self.row.dataset_id,
            "repo_id": self.row.repo_id,
            "revision": self.row.revision,
            "source_receipt_sha256": self.source_receipt_sha256,
            "source_files": [
                {"path": relative, "bytes": size, "sha256": digest}
                for relative, _path, size, digest in self.source_files
            ],
            "policy_file_sha256": self.policy_file_sha256,
            "policy_sha256": None if self.policy is None else self.policy.policy_sha256,
            "dataset_byte_cap": self.dataset_byte_cap,
            "chunk_byte_cap": self.chunk_byte_cap,
            "metadata_compressed_byte_cap": self.metadata_compressed_byte_cap,
            "storage_layout": self.storage_layout,
            "producer_sha256": self.producer_sha256,
            "helper_timeseries_sha256": self.helper_timeseries_sha256,
            "helper_tsfile_sha256": self.helper_tsfile_sha256,
        }))


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise PreparationError(f"{context}.{key} must be a non-empty string")
    return result.strip()


def _parse_temporal_fields(value: Any, context: str) -> tuple[TemporalStringField, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PreparationError(f"{context} must be a list")
    result: list[TemporalStringField] = []
    seen_source: set[str] = set()
    seen_output: set[str] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, Mapping):
            raise PreparationError(f"{item_context} must be an object")
        source_name = _required_text(item, "source_name", item_context)
        source_type = str(item.get("source_type", "STRING")).strip().upper()
        if source_type not in {"STRING", "TEXT"}:
            raise PreparationError(f"{item_context}.source_type must be STRING or TEXT")
        source_category = str(item.get("source_category", "FIELD")).strip().upper()
        if source_category not in {"TAG", "FIELD"}:
            raise PreparationError(f"{item_context}.source_category must be TAG or FIELD")
        output_name = str(
            item.get("output_name", f"{source_name}__timestamp_unix_us")
        ).strip()
        if output_name != f"{source_name}__timestamp_unix_us":
            raise PreparationError(
                f"{item_context}.output_name must be {source_name}__timestamp_unix_us"
            )
        if source_name in seen_source or output_name in seen_output:
            raise PreparationError(f"duplicate temporal source/output name in {context}")
        parse_kind = _required_text(item, "parse_kind", item_context)
        if parse_kind not in {"strptime", "integer_epoch"}:
            raise PreparationError(f"unsupported temporal parse_kind in {item_context}")
        timezone_name = _required_text(item, "timezone", item_context)
        event_claim = item.get("event_utc_instant_claim")
        if not isinstance(event_claim, bool):
            raise PreparationError(
                f"{item_context}.event_utc_instant_claim must be a boolean"
            )
        semantic_kind = _required_text(item, "semantic_kind", item_context)
        timezone_basis = _required_text(item, "timezone_basis", item_context)
        format_class = _required_text(item, "format_class", item_context)
        applicability = _required_text(item, "applicability", item_context)
        candidate_definition_count = item.get("candidate_definition_count")
        if (
            not isinstance(candidate_definition_count, int)
            or isinstance(candidate_definition_count, bool)
            or candidate_definition_count <= 0
        ):
            raise PreparationError(
                f"{item_context}.candidate_definition_count must be a positive integer"
            )
        if item.get("output_type") != "int64" or item.get("output_unit") != "us":
            raise PreparationError(
                f"{item_context} output must be explicit signed int64 Unix microseconds"
            )
        datetime_format: str | None = None
        accepted_datetime_formats: tuple[str, ...] = ()
        source_unit: str | None = None
        ambiguous_fold: int | None = None
        if parse_kind == "strptime":
            datetime_format = _required_text(item, "format", item_context)
            raw_formats = item.get("accepted_formats")
            if raw_formats is None:
                accepted_datetime_formats = (datetime_format,)
            else:
                if (
                    not isinstance(raw_formats, list)
                    or not raw_formats
                    or any(
                        not isinstance(entry, str) or not entry.strip()
                        for entry in raw_formats
                    )
                ):
                    raise PreparationError(
                        f"{item_context}.accepted_formats must be a non-empty string list"
                    )
                accepted_datetime_formats = tuple(entry.strip() for entry in raw_formats)
                if (
                    len(accepted_datetime_formats) != len(set(accepted_datetime_formats))
                    or datetime_format != accepted_datetime_formats[0]
                ):
                    raise PreparationError(
                        f"{item_context}.format must be the first unique accepted format"
                    )
            raw_fold = item.get("ambiguous_fold")
            if raw_fold is not None:
                if raw_fold not in {0, 1}:
                    raise PreparationError(f"{item_context}.ambiguous_fold must be 0 or 1")
                ambiguous_fold = int(raw_fold)
            if timezone_name != "embedded_offset":
                try:
                    ZoneInfo(timezone_name)
                except ZoneInfoNotFoundError as error:
                    raise PreparationError(
                        f"unknown explicit timezone in {item_context}: {timezone_name}"
                    ) from error
        else:
            source_unit = _required_text(item, "source_unit", item_context)
            if source_unit not in SOURCE_UNITS:
                raise PreparationError(f"unsupported epoch source unit in {item_context}")
            if source_unit != "d" and UNIT_EXPONENT["us"] < UNIT_EXPONENT[source_unit]:
                raise PreparationError(
                    f"{item_context} cannot convert {source_unit} to us without downsampling"
                )
            if timezone_name != "UTC":
                raise PreparationError(f"integer_epoch timezone must be UTC in {item_context}")
        if event_claim:
            if semantic_kind != "unix_epoch_instant" or timezone_basis not in {
                "embedded_numeric_offset",
                "explicit_z_suffix",
            }:
                raise PreparationError(
                    f"{item_context} UTC-instant claim lacks an exact offset/Z semantic basis"
                )
            if (
                (timezone_basis == "embedded_numeric_offset" and timezone_name != "embedded_offset")
                or (timezone_basis == "explicit_z_suffix" and timezone_name != "UTC")
            ):
                raise PreparationError(
                    f"{item_context} timezone and exact UTC evidence basis disagree"
                )
        elif semantic_kind not in {
            "naive_civil_utc_coordinate",
            "civil_date_utc_midnight_coordinate",
            "unix_epoch_day_coordinate",
            "stored_timestamp_coordinate",
        }:
            raise PreparationError(
                f"{item_context} non-instant temporal rule needs an explicit coordinate semantic"
            )
        elif {
            "naive_civil_utc_coordinate": "project_utc_coordinate_convention",
            "civil_date_utc_midnight_coordinate": "utc_midnight_coordinate_convention",
            "unix_epoch_day_coordinate": "utc_midnight_coordinate_convention",
            "stored_timestamp_coordinate": "embedded_numeric_offset",
        }[semantic_kind] != timezone_basis:
            raise PreparationError(
                f"{item_context} coordinate semantic and timezone_basis disagree"
            )
        if parse_kind == "integer_epoch" and source_unit == "d" and (
            semantic_kind != "unix_epoch_day_coordinate"
            or timezone_basis != "utc_midnight_coordinate_convention"
            or event_claim
        ):
            raise PreparationError(
                f"{item_context} day ordinals must be labeled as a non-instant UTC-midnight coordinate"
            )
        evidence_raw = item.get("evidence")
        if (
            not isinstance(evidence_raw, list)
            or not evidence_raw
            or any(not isinstance(entry, str) or not entry.strip() for entry in evidence_raw)
        ):
            raise PreparationError(f"{item_context}.evidence must be a non-empty string list")
        result.append(TemporalStringField(
            source_name=source_name,
            source_type=source_type,
            source_category=source_category,
            output_name=output_name,
            parse_kind=parse_kind,
            datetime_format=datetime_format,
            accepted_datetime_formats=accepted_datetime_formats,
            timezone_name=timezone_name,
            source_unit=source_unit,
            ambiguous_fold=ambiguous_fold,
            event_utc_instant_claim=event_claim,
            semantic_kind=semantic_kind,
            timezone_basis=timezone_basis,
            format_class=format_class,
            applicability=applicability,
            candidate_definition_count=candidate_definition_count,
            evidence=tuple(entry.strip() for entry in evidence_raw),
        ))
        seen_source.add(source_name)
        seen_output.add(output_name)
    return tuple(result)


def _parse_source_bound_temporal_fields(
    value: Any, context: str
) -> tuple[SourceBoundTemporalField, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PreparationError(f"{context} must be a list")
    result: list[SourceBoundTemporalField] = []
    seen_bindings: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, Mapping):
            raise PreparationError(f"{item_context} must be an object")
        source_path = _required_text(item, "source_path", item_context)
        pure_path = Path(source_path)
        if (
            pure_path.is_absolute()
            or source_path != pure_path.as_posix()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise PreparationError(f"{item_context}.source_path must be a safe relative path")
        source_sha256 = _required_text(item, "source_sha256", item_context)
        if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
            raise PreparationError(f"{item_context}.source_sha256 must be lowercase sha256")
        table_name = _required_text(item, "table_name", item_context)
        source_schema_sha256 = _required_text(
            item, "source_schema_sha256", item_context
        )
        if re.fullmatch(r"[0-9a-f]{64}", source_schema_sha256) is None:
            raise PreparationError(
                f"{item_context}.source_schema_sha256 must be lowercase sha256"
            )
        source_name = _required_text(item, "source_name", item_context)
        binding_key = (source_path, source_name)
        if binding_key in seen_bindings:
            raise PreparationError(
                f"duplicate source-bound temporal field in {context}: {binding_key!r}"
            )
        source_type = _required_text(item, "source_type", item_context).upper()
        if source_type not in {"INT32", "INT64"}:
            raise PreparationError(
                f"{item_context}.source_type must be INT32 or INT64"
            )
        source_category = _required_text(
            item, "source_category", item_context
        ).upper()
        if source_category != "FIELD":
            raise PreparationError(f"{item_context}.source_category must be FIELD")
        source_unit = _required_text(item, "source_unit", item_context)
        if source_unit not in {"s", "ms", "us"}:
            raise PreparationError(f"{item_context}.source_unit must be s, ms, or us")
        output_name = _required_text(item, "output_name", item_context)
        if output_name != f"{source_name}__timestamp_unix_us":
            raise PreparationError(
                f"{item_context}.output_name must be "
                f"{source_name}__timestamp_unix_us"
            )
        if item.get("output_type") != "int64" or item.get("output_unit") != "us":
            raise PreparationError(
                f"{item_context} output must be explicit signed int64 Unix microseconds"
            )
        parse_kind = _required_text(item, "parse_kind", item_context)
        if parse_kind != "integer_epoch":
            raise PreparationError(f"{item_context}.parse_kind must be integer_epoch")
        timezone_name = _required_text(item, "timezone", item_context)
        if timezone_name != "UTC":
            raise PreparationError(f"{item_context}.timezone must be UTC")
        event_claim = item.get("event_utc_instant_claim")
        if not isinstance(event_claim, bool):
            raise PreparationError(
                f"{item_context}.event_utc_instant_claim must be a boolean"
            )
        semantic_kind = _required_text(item, "semantic_kind", item_context)
        timezone_basis = _required_text(item, "timezone_basis", item_context)
        if event_claim:
            if (
                semantic_kind != "unix_epoch_instant"
                or timezone_basis != "upstream_encoded_epoch"
            ):
                raise PreparationError(
                    f"{item_context} exact integer epoch semantics disagree"
                )
        elif (
            semantic_kind != "stored_timestamp_coordinate"
            or timezone_basis != "upstream_encoded_epoch"
        ):
            raise PreparationError(
                f"{item_context} stored integer epoch semantics disagree"
            )
        evidence_raw = item.get("evidence")
        if (
            not isinstance(evidence_raw, list)
            or not evidence_raw
            or any(
                not isinstance(entry, str) or not entry.strip()
                for entry in evidence_raw
            )
        ):
            raise PreparationError(f"{item_context}.evidence must be a non-empty string list")
        result.append(SourceBoundTemporalField(
            source_path=source_path,
            source_sha256=source_sha256,
            table_name=table_name,
            source_schema_sha256=source_schema_sha256,
            source_name=source_name,
            source_type=source_type,
            source_category=source_category,
            source_unit=source_unit,
            output_name=output_name,
            output_unit="us",
            parse_kind=parse_kind,
            timezone_name=timezone_name,
            event_utc_instant_claim=event_claim,
            semantic_kind=semantic_kind,
            timezone_basis=timezone_basis,
            evidence=tuple(entry.strip() for entry in evidence_raw),
        ))
        seen_bindings.add(binding_key)
    return tuple(result)


def _parse_dropped_timestamp_like_fields(
    value: Any, context: str
) -> tuple[DroppedTimestampLikeField, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PreparationError(f"{context} must be a list")
    result: list[DroppedTimestampLikeField] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, Mapping):
            raise PreparationError(f"{item_context} must be an object")
        source_name = _required_text(item, "source_name", item_context)
        if source_name in seen:
            raise PreparationError(f"duplicate explicit temporal drop in {context}: {source_name}")
        source_type = _required_text(item, "source_type", item_context).upper()
        if source_type not in {"STRING", "TEXT"}:
            raise PreparationError(f"{item_context}.source_type must be STRING or TEXT")
        source_category = _required_text(item, "source_category", item_context).upper()
        if source_category not in {"TAG", "FIELD"}:
            raise PreparationError(f"{item_context}.source_category must be TAG or FIELD")
        decision = _required_text(item, "decision", item_context)
        if decision != "drop_clock_only_without_calendar_date":
            raise PreparationError(f"unsupported explicit temporal drop in {item_context}")
        format_class = _required_text(item, "format_class", item_context)
        if format_class != "clock_only_drop":
            raise PreparationError(f"{item_context}.format_class must be clock_only_drop")
        reason = _required_text(item, "reason", item_context)
        raw_applicability = item.get("applicability")
        applicability = None
        if raw_applicability is not None:
            if not isinstance(raw_applicability, str) or not raw_applicability.strip():
                raise PreparationError(f"{item_context}.applicability must be a non-empty string")
            applicability = raw_applicability.strip()
        candidate_definition_count = item.get("candidate_definition_count")
        if (
            not isinstance(candidate_definition_count, int)
            or isinstance(candidate_definition_count, bool)
            or candidate_definition_count <= 0
        ):
            raise PreparationError(
                f"{item_context}.candidate_definition_count must be a positive integer"
            )
        evidence_raw = item.get("evidence")
        if (
            not isinstance(evidence_raw, list)
            or not evidence_raw
            or any(not isinstance(entry, str) or not entry.strip() for entry in evidence_raw)
        ):
            raise PreparationError(f"{item_context}.evidence must be a non-empty string list")
        result.append(DroppedTimestampLikeField(
            source_name=source_name,
            source_type=source_type,
            source_category=source_category,
            decision=decision,
            format_class=format_class,
            reason=reason,
            applicability=applicability,
            candidate_definition_count=candidate_definition_count,
            evidence=tuple(entry.strip() for entry in evidence_raw),
        ))
        seen.add(source_name)
    return tuple(result)


def _positive_count(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PreparationError(f"{context} must be a non-negative integer")
    return value


def _parse_source_rules(value: Any, context: str) -> tuple[str, tuple[SourceRule, ...]]:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{context} must be an object")
    adapter_id = _required_text(value, "adapter_id", context)
    raw_rules = value.get("source_rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise PreparationError(f"{context}.source_rules must be a non-empty list")
    rules: list[SourceRule] = []
    seen_rule_ids: set[str] = set()
    seen_tables: set[str] = set()
    for index, item in enumerate(raw_rules):
        item_context = f"{context}.source_rules[{index}]"
        if not isinstance(item, Mapping):
            raise PreparationError(f"{item_context} must be an object")
        rule_id = _required_text(item, "rule_id", item_context)
        if rule_id in seen_rule_ids:
            raise PreparationError(f"duplicate mixed-axis rule id: {rule_id}")
        raw_tables = item.get("table_names")
        if (
            not isinstance(raw_tables, list)
            or not raw_tables
            or any(not isinstance(name, str) or not name.strip() for name in raw_tables)
        ):
            raise PreparationError(f"{item_context}.table_names must be a non-empty string list")
        table_names = tuple(name.strip() for name in raw_tables)
        if len(table_names) != len(set(table_names)) or seen_tables & set(table_names):
            raise PreparationError(f"mixed-axis source rules overlap table names in {item_context}")
        seen_tables.update(table_names)
        action = _required_text(item, "action", item_context)
        if action not in {"materialize", "exclude_entire_source_file"}:
            raise PreparationError(f"unsupported mixed-axis action in {item_context}: {action}")
        expected_file_count = _positive_count(
            item.get("expected_source_file_count"),
            f"{item_context}.expected_source_file_count",
        )
        if expected_file_count == 0:
            raise PreparationError(f"{item_context}.expected_source_file_count must be positive")
        reason = _required_text(item, "reason", item_context)
        raw_evidence = item.get("evidence")
        if (
            not isinstance(raw_evidence, list)
            or not raw_evidence
            or any(not isinstance(entry, str) or not entry.strip() for entry in raw_evidence)
        ):
            raise PreparationError(f"{item_context}.evidence must be a non-empty string list")
        evidence = tuple(entry.strip() for entry in raw_evidence)
        source_column = source_unit = output_unit = output_column = None
        event_claim = None
        utc_instant_verified = None
        semantic_kind = timezone_basis = None
        timezone_inferred = None
        row_filter: RowFilter | None = None
        if action == "materialize":
            timestamp = item.get("timestamp")
            if not isinstance(timestamp, Mapping):
                raise PreparationError(f"{item_context}.timestamp must be an object")
            source_column = _required_text(timestamp, "source_column", f"{item_context}.timestamp")
            output_unit = _required_text(timestamp, "output_unit", f"{item_context}.timestamp")
            output_column = _required_text(timestamp, "output_column", f"{item_context}.timestamp")
            if output_unit != "us" or output_column != "timestamp_unix_us":
                raise PreparationError(f"{item_context} adapter output must be int64 Unix us")
            event_claim = timestamp.get("event_utc_instant_claim")
            if not isinstance(event_claim, bool):
                raise PreparationError(
                    f"{item_context}.timestamp.event_utc_instant_claim must be boolean"
                )
            raw_utc_verified = timestamp.get("utc_instant_verified")
            if event_claim:
                if raw_utc_verified is not True:
                    raise PreparationError(
                        f"{item_context}.timestamp.utc_instant_verified must be true "
                        "for an event-instant claim"
                    )
                utc_instant_verified = True
            else:
                if raw_utc_verified is True:
                    raise PreparationError(
                        f"{item_context}.timestamp cannot verify an event instant while "
                        "event_utc_instant_claim is false"
                    )
                utc_instant_verified = False
            semantic_kind = _required_text(
                timestamp, "semantic_kind", f"{item_context}.timestamp"
            )
            timezone_basis = _required_text(
                timestamp, "timezone_basis", f"{item_context}.timestamp"
            )
            timezone_inferred = timestamp.get("timezone_inferred")
            if timezone_inferred is not False:
                raise PreparationError(
                    f"{item_context}.timestamp.timezone_inferred must be false"
                )
            if source_column == "Time":
                source_unit = _required_text(
                    timestamp, "source_unit", f"{item_context}.timestamp"
                )
                if source_unit not in {"s", "ms", "us"}:
                    raise PreparationError(
                        f"{item_context} physical Time adapter requires exact s/ms/us"
                    )
                if timestamp.get("source_unit_verified") is not True:
                    raise PreparationError(
                        f"{item_context}.timestamp.source_unit_verified must be true"
                    )
            elif timestamp.get("source_format_verified") is not True:
                raise PreparationError(
                    f"{item_context}.timestamp.source_format_verified must be true"
                )
            raw_filter = item.get("row_filter")
            if raw_filter is not None:
                filter_context = f"{item_context}.row_filter"
                if not isinstance(raw_filter, Mapping) or raw_filter.get("kind") != "all":
                    raise PreparationError(f"{filter_context}.kind must be all")
                raw_conditions = raw_filter.get("conditions")
                if not isinstance(raw_conditions, list) or not raw_conditions:
                    raise PreparationError(
                        f"{filter_context}.conditions must be a non-empty list"
                    )
                conditions: list[RowFilterCondition] = []
                seen_conditions: set[str] = set()
                for condition_index, condition in enumerate(raw_conditions):
                    condition_context = (
                        f"{filter_context}.conditions[{condition_index}]"
                    )
                    if not isinstance(condition, Mapping):
                        raise PreparationError(f"{condition_context} must be an object")
                    condition_name = _required_text(
                        condition, "source_name", condition_context
                    )
                    if condition_name in seen_conditions:
                        raise PreparationError(
                            f"duplicate row-filter source in {filter_context}: {condition_name}"
                        )
                    condition_type = _required_text(
                        condition, "source_type", condition_context
                    ).upper()
                    condition_category = _required_text(
                        condition, "source_category", condition_context
                    ).upper()
                    operator = _required_text(condition, "operator", condition_context)
                    if operator not in {"int64_gte", "not_null"}:
                        raise PreparationError(
                            f"unsupported row-filter operator in {condition_context}"
                        )
                    raw_value = condition.get("value")
                    condition_value: int | None = None
                    if operator == "int64_gte":
                        if condition_type != "INT64" or condition_category != "TIME":
                            raise PreparationError(
                                f"{condition_context} int64_gte requires INT64 TIME"
                            )
                        if (
                            not isinstance(raw_value, int)
                            or isinstance(raw_value, bool)
                            or not INT64_MIN <= raw_value <= INT64_MAX
                        ):
                            raise PreparationError(
                                f"{condition_context}.value must be signed int64"
                            )
                        condition_value = raw_value
                    elif raw_value is not None:
                        raise PreparationError(
                            f"{condition_context}.value is only valid for int64_gte"
                        )
                    conditions.append(RowFilterCondition(
                        source_name=condition_name,
                        source_type=condition_type,
                        source_category=condition_category,
                        operator=operator,
                        value=condition_value,
                        drop_reason=_required_text(
                            condition, "drop_reason", condition_context
                        ),
                    ))
                    seen_conditions.add(condition_name)
                expected_input = _positive_count(
                    raw_filter.get("expected_input_rows"),
                    f"{filter_context}.expected_input_rows",
                )
                expected_output = _positive_count(
                    raw_filter.get("expected_output_rows"),
                    f"{filter_context}.expected_output_rows",
                )
                expected_drop = _positive_count(
                    raw_filter.get("expected_drop_rows"),
                    f"{filter_context}.expected_drop_rows",
                )
                if expected_input != expected_output + expected_drop:
                    raise PreparationError(
                        f"{filter_context} row counts do not close"
                    )
                filter_evidence = raw_filter.get("evidence")
                if (
                    not isinstance(filter_evidence, list)
                    or not filter_evidence
                    or any(
                        not isinstance(entry, str) or not entry.strip()
                        for entry in filter_evidence
                    )
                ):
                    raise PreparationError(
                        f"{filter_context}.evidence must be a non-empty string list"
                    )
                row_filter = RowFilter(
                    conditions=tuple(conditions),
                    expected_input_rows=expected_input,
                    expected_output_rows=expected_output,
                    expected_drop_rows=expected_drop,
                    evidence=tuple(entry.strip() for entry in filter_evidence),
                )
        elif item.get("timestamp") is not None or item.get("row_filter") is not None:
            raise PreparationError(
                f"{item_context} excluded files cannot declare timestamp/filter output"
            )
        rules.append(SourceRule(
            rule_id=rule_id,
            table_names=table_names,
            action=action,
            expected_source_file_count=expected_file_count,
            reason=reason,
            evidence=evidence,
            source_column=source_column,
            source_unit=source_unit,
            output_unit=output_unit,
            output_column=output_column,
            event_utc_instant_claim=event_claim,
            utc_instant_verified=utc_instant_verified,
            semantic_kind=semantic_kind,
            timezone_basis=timezone_basis,
            timezone_inferred=timezone_inferred,
            row_filter=row_filter,
        ))
        seen_rule_ids.add(rule_id)
    return adapter_id, tuple(rules)


def load_scope(path: Path) -> tuple[dict[str, Any], tuple[ScopeRow, ...], str]:
    resolved = path.resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("schema") != "web-compression-project-dataset-scope-v1":
        raise PreparationError("unsupported project dataset scope schema")
    scope_id = _required_text(raw, "scope_id", "scope")
    values = raw.get("timestamp_datasets")
    if not isinstance(values, list) or not values:
        raise PreparationError("scope.timestamp_datasets must be a non-empty list")
    rows: list[ScopeRow] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise PreparationError(f"scope.timestamp_datasets[{index}] must be an object")
        dataset_id = _required_text(value, "id", f"scope.timestamp_datasets[{index}]")
        repo_id = _required_text(value, "repo_id", f"scope.timestamp_datasets[{index}]")
        revision = _required_text(value, "revision", f"scope.timestamp_datasets[{index}]")
        if dataset_id in seen or "/" in dataset_id or dataset_id in {".", ".."}:
            raise PreparationError(f"duplicate or unsafe dataset id: {dataset_id!r}")
        if repo_id != f"THULab/{dataset_id}":
            raise PreparationError(f"unexpected repository identity for {dataset_id}: {repo_id}")
        if REVISION_RE.fullmatch(revision) is None:
            raise PreparationError(f"invalid frozen revision for {dataset_id}: {revision}")
        seen.add(dataset_id)
        rows.append(ScopeRow(dataset_id, repo_id, revision))
    if len(rows) != int(raw.get("timestamp_dataset_count", -1)):
        raise PreparationError("scope timestamp dataset count mismatch")
    return raw, tuple(rows), sha256_file(resolved)


def load_policies(
    path: Path, *, scope_id: str, scope_sha256: str, scope_rows: Sequence[ScopeRow]
) -> tuple[dict[str, TimestampPolicy], str]:
    resolved = path.resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("schema") != POLICY_VERSION:
        raise PreparationError(f"timestamp policy schema must be {POLICY_VERSION!r}")
    if raw.get("scope_id") != scope_id or raw.get("scope_sha256") != scope_sha256:
        raise PreparationError("timestamp policy is not bound to the supplied frozen scope")
    values = raw.get("datasets")
    if not isinstance(values, list):
        raise PreparationError("timestamp policy datasets must be a list")
    scope_by_id = {row.dataset_id: row for row in scope_rows}
    policies: dict[str, TimestampPolicy] = {}
    for index, value in enumerate(values):
        context = f"policies.datasets[{index}]"
        if not isinstance(value, Mapping):
            raise PreparationError(f"{context} must be an object")
        dataset_id = _required_text(value, "dataset_id", context)
        row = scope_by_id.get(dataset_id)
        if row is None:
            raise PreparationError(f"timestamp policy names an out-of-scope dataset: {dataset_id}")
        if dataset_id in policies:
            raise PreparationError(f"duplicate timestamp policy for {dataset_id}")
        repo_id = _required_text(value, "repo_id", context)
        revision = _required_text(value, "revision", context)
        if repo_id != row.repo_id or revision != row.revision:
            raise PreparationError(f"timestamp policy identity mismatch for {dataset_id}")
        decision = _required_text(value, "decision", context)
        if decision not in ALLOWED_POLICY_DECISIONS:
            raise PreparationError(f"unsupported timestamp policy decision for {dataset_id}")
        reason = _required_text(value, "reason", context)
        table_name: str | None = None
        source_column: str | None = None
        source_unit: str | None = None
        output_unit: str | None = None
        output_column: str | None = None
        mixed_axis_adapter_id: str | None = None
        source_rules: tuple[SourceRule, ...] = ()
        temporal_string_fields = _parse_temporal_fields(
            value.get("temporal_string_fields"), f"{context}.temporal_string_fields"
        )
        source_bound_temporal_fields = _parse_source_bound_temporal_fields(
            value.get("source_bound_temporal_fields"),
            f"{context}.source_bound_temporal_fields",
        )
        dropped_timestamp_like_string_fields = _parse_dropped_timestamp_like_fields(
            value.get("dropped_timestamp_like_string_fields"),
            f"{context}.dropped_timestamp_like_string_fields",
        )
        temporal_names = {field.source_name for field in temporal_string_fields}
        temporal_names.update(
            field.source_name for field in source_bound_temporal_fields
        )
        explicit_drop_names = {
            field.source_name for field in dropped_timestamp_like_string_fields
        }
        overlap = temporal_names & explicit_drop_names
        if overlap:
            raise PreparationError(
                f"{dataset_id} temporal conversion/drop rules overlap: {sorted(overlap)}"
            )
        raw_evidence = value.get("evidence")
        if (
            not isinstance(raw_evidence, list)
            or not raw_evidence
            or any(not isinstance(item, str) or not item.strip() for item in raw_evidence)
        ):
            raise PreparationError(f"{dataset_id} timestamp policy needs explicit evidence")
        evidence = tuple(item.strip() for item in raw_evidence)
        if decision in {
            "admit_exact_utc",
            "admit_stored_epoch_coordinate",
            "admit_dataset_adapter",
        }:
            if value.get("model") != "table":
                raise PreparationError(f"{dataset_id} UTC policy must explicitly declare table model")
            table_name = _required_text(value, "table_name", context)
        if decision == "admit_dataset_adapter":
            mixed_axis_adapter_id, source_rules = _parse_source_rules(
                value.get("mixed_axis_adapter"), f"{context}.mixed_axis_adapter"
            )
            temporal_by_name = {
                field.source_name: field for field in temporal_string_fields
            }
            for rule in source_rules:
                if rule.action != "materialize":
                    continue
                assert rule.source_column is not None
                if rule.source_column == "Time":
                    if rule.event_utc_instant_claim:
                        if (
                            rule.semantic_kind != "unix_epoch_instant"
                            or rule.timezone_basis != "upstream_encoded_epoch"
                        ):
                            raise PreparationError(
                                f"{dataset_id}/{rule.rule_id} exact physical Time semantics disagree"
                            )
                    elif (
                        rule.semantic_kind != "stored_epoch_coordinate"
                        or rule.timezone_basis != "upstream_encoded_epoch"
                    ):
                        raise PreparationError(
                            f"{dataset_id}/{rule.rule_id} stored Time semantics disagree"
                        )
                else:
                    temporal = temporal_by_name.get(rule.source_column)
                    if temporal is None:
                        raise PreparationError(
                            f"{dataset_id}/{rule.rule_id} primary string timestamp is not "
                            "listed in temporal_string_fields"
                        )
                    if (
                        temporal.event_utc_instant_claim
                        != rule.event_utc_instant_claim
                        or temporal.semantic_kind != rule.semantic_kind
                        or temporal.timezone_basis != rule.timezone_basis
                    ):
                        raise PreparationError(
                            f"{dataset_id}/{rule.rule_id} primary temporal provenance disagrees"
                        )
        elif decision in {"admit_exact_utc", "admit_stored_epoch_coordinate"}:
            timestamp = value.get("timestamp")
            if not isinstance(timestamp, Mapping):
                raise PreparationError(f"{context}.timestamp must be an object")
            source_column = _required_text(timestamp, "source_column", f"{context}.timestamp")
            if decision == "admit_stored_epoch_coordinate" and source_column != "Time":
                raise PreparationError(
                    f"{dataset_id} stored epoch coordinate admission requires source_column='Time'"
                )
            if source_column == "Time":
                source_unit = _required_text(timestamp, "source_unit", f"{context}.timestamp")
                if source_unit not in SOURCE_UNITS:
                    raise PreparationError(f"unsupported timestamp source unit for {dataset_id}")
                output_unit = str(
                    timestamp.get("output_unit", raw.get("default_output_unit", "us"))
                ).strip()
                if output_unit != "us":
                    raise PreparationError(
                        f"{dataset_id} physical Time output_unit must be us"
                    )
                if source_unit != "d" and UNIT_EXPONENT[output_unit] < UNIT_EXPONENT[source_unit]:
                    raise PreparationError(
                        f"{dataset_id} timestamp conversion {source_unit}->{output_unit} would "
                        "downsample precision; select an equal or finer output unit"
                    )
            else:
                temporal_by_name = {
                    field.source_name: field for field in temporal_string_fields
                }
                if source_column not in temporal_by_name:
                    raise PreparationError(
                        f"{dataset_id} primary string timestamp {source_column!r} is not "
                        "listed in temporal_string_fields"
                    )
                if (
                    decision == "admit_exact_utc"
                    and not temporal_by_name[source_column].event_utc_instant_claim
                ):
                    raise PreparationError(
                        f"{dataset_id} exact-UTC primary string timestamp lacks an "
                        "event_utc_instant_claim"
                    )
                output_unit = "us"
            output_column = str(timestamp.get("output_column", "timestamp_unix_us")).strip()
            if output_column != "timestamp_unix_us":
                raise PreparationError(
                    f"{dataset_id} output_column must be timestamp_unix_us"
                )
            if decision == "admit_exact_utc":
                required = {
                    "timezone": "UTC",
                    "semantic_kind": "unix_epoch_instant",
                    "utc_instant_verified": True,
                    "timezone_inferred": False,
                    (
                        "source_unit_verified"
                        if source_column == "Time"
                        else "source_format_verified"
                    ): True,
                }
            else:
                if source_unit not in {"s", "ms", "us"} or output_unit != "us":
                    raise PreparationError(
                        f"{dataset_id} stored epoch coordinate requires exact s/ms/us -> us"
                    )
                required = {
                    "semantic_kind": "stored_epoch_coordinate",
                    "source_unit_verified": True,
                    "event_utc_instant_claim": False,
                    "timezone_basis": "upstream_encoded_epoch",
                    "timezone_inferred": False,
                }
            for key, expected in required.items():
                if timestamp.get(key) != expected:
                    raise PreparationError(
                        f"{dataset_id} timestamp policy requires {key}={expected!r}"
                    )
        policy_hash = _sha256_bytes(_canonical_json(dict(value)))
        policies[dataset_id] = TimestampPolicy(
            dataset_id=dataset_id,
            repo_id=repo_id,
            revision=revision,
            decision=decision,
            reason=reason,
            table_name=table_name,
            source_column=source_column,
            source_unit=source_unit,
            output_unit=output_unit,
            output_column=output_column,
            temporal_string_fields=temporal_string_fields,
            source_bound_temporal_fields=source_bound_temporal_fields,
            dropped_timestamp_like_string_fields=dropped_timestamp_like_string_fields,
            mixed_axis_adapter_id=mixed_axis_adapter_id,
            source_rules=source_rules,
            evidence=evidence,
            policy_sha256=policy_hash,
        )
    return policies, sha256_file(resolved)


def _resolved_within(path: Path, root: Path, context: str) -> Path:
    """Resolve an already lstat-validated source object and enforce containment."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PreparationError(f"cannot resolve {context}: {path}") from error
    if resolved != root and root not in resolved.parents:
        raise PreparationError(f"{context} escapes its canonical source root: {path}")
    return resolved


def _real_source_directory(
    path: Path, context: str, *, root: Path | None = None
) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise PreparationError(f"missing or unreadable {context}: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PreparationError(f"{context} must be a real directory, not a symlink")
    containment_root = path.resolve(strict=True) if root is None else root
    return _resolved_within(path, containment_root, context)


def _regular_source_file(path: Path, context: str, *, root: Path) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise PreparationError(f"missing or unreadable {context}: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise PreparationError(f"{context} must be a regular file, not a symlink")
    return _resolved_within(path, root, context)


def _eligible_files(
    dataset_dir: Path, *, mirror_root: Path | None = None
) -> list[dict[str, Any]]:
    """Inventory admitted files without following any source-tree symlink."""
    dataset_root = _real_source_directory(
        dataset_dir,
        "source dataset directory",
        root=mirror_root,
    )
    canonical_mirror = dataset_root if mirror_root is None else mirror_root
    records: list[dict[str, Any]] = []
    pending = [dataset_root]
    while pending:
        directory = pending.pop()
        _real_source_directory(
            directory,
            "source dataset tree directory",
            root=dataset_root,
        )
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise PreparationError(
                f"cannot enumerate source dataset directory: {directory}"
            ) from error
        for path in children:
            relative = path.relative_to(dataset_root)
            try:
                mode = os.lstat(path).st_mode
            except OSError as error:
                raise PreparationError(
                    f"cannot inspect source dataset tree entry: {relative.as_posix()}"
                ) from error
            if stat.S_ISLNK(mode):
                raise PreparationError(
                    "source dataset tree entry must not be a symlink: "
                    f"{relative.as_posix()}"
                )
            if stat.S_ISDIR(mode):
                resolved = _real_source_directory(
                    path,
                    "source dataset tree directory",
                    root=dataset_root,
                )
                _resolved_within(
                    resolved, canonical_mirror, "source dataset tree directory"
                )
                if ".cache" not in relative.parts:
                    pending.append(resolved)
                continue
            if not stat.S_ISREG(mode):
                raise PreparationError(
                    "source dataset tree entry must be a regular file or directory: "
                    f"{relative.as_posix()}"
                )
            resolved = _regular_source_file(
                path, "source dataset tree entry", root=dataset_root
            )
            _resolved_within(resolved, canonical_mirror, "source dataset tree entry")
            if ".cache" in relative.parts:
                continue
            relative_text = relative.as_posix()
            if path.suffix == ".tsfile" or relative_text in {
                "README.md",
                ".gitattributes",
            }:
                records.append({
                    "path": relative_text,
                    "bytes": os.lstat(resolved).st_size,
                    "sha256": sha256_file(resolved),
                })
    return sorted(records, key=lambda item: str(item["path"]))


def bind_mirror(
    mirror: Path,
    *,
    scope: Mapping[str, Any],
    rows: Sequence[ScopeRow],
    scope_sha256: str,
    policies: Mapping[str, TimestampPolicy],
    policy_file_sha256: str,
    dataset_byte_cap: int,
    chunk_byte_cap: int,
    metadata_compressed_byte_cap: int,
    storage_layout: str,
    producer_sha256: str,
    helper_timeseries_sha256: str,
    helper_tsfile_sha256: str,
    dataset_ids: Sequence[str] | None = None,
) -> tuple[DatasetBinding, ...]:
    scope_ids = tuple(row.dataset_id for row in rows)
    scope_id_set = set(scope_ids)
    if dataset_ids is None:
        selected_id_set = scope_id_set
    else:
        if isinstance(dataset_ids, (str, bytes)):
            raise PreparationError("requested dataset IDs must be a sequence of strings")
        requested_ids = tuple(dataset_ids)
        if not requested_ids:
            raise PreparationError("requested dataset IDs must not be empty")
        if any(not isinstance(dataset_id, str) or not dataset_id for dataset_id in requested_ids):
            raise PreparationError("requested dataset IDs must be non-empty strings")
        if len(set(requested_ids)) != len(requested_ids):
            raise PreparationError("requested dataset IDs contain duplicates")
        unknown_ids = sorted(set(requested_ids) - scope_id_set)
        if unknown_ids:
            raise PreparationError(
                f"requested datasets are outside scope: {unknown_ids}"
            )
        selected_id_set = set(requested_ids)

    root = _real_source_directory(mirror, "source mirror")
    top_path = _regular_source_file(
        root / "DOWNLOAD_MANIFEST.json", "source mirror manifest", root=root
    )
    top = json.loads(top_path.read_text(encoding="utf-8"))
    if top.get("schema") != "thulab-timestamp-scope-download-v1" or top.get("status") != "complete":
        raise PreparationError("THULab mirror manifest is not complete")
    if top.get("scope_id") != scope.get("scope_id") or top.get("scope_sha256") != scope_sha256:
        raise PreparationError("THULab mirror manifest is not bound to the supplied scope")
    top_rows = top.get("datasets")
    if not isinstance(top_rows, list) or len(top_rows) != len(rows):
        raise PreparationError("THULab mirror dataset inventory count mismatch")
    top_by_id = {str(item.get("dataset_id")): item for item in top_rows if isinstance(item, Mapping)}
    if set(top_by_id) != {row.dataset_id for row in rows}:
        raise PreparationError("THULab mirror dataset inventory differs from scope")
    receipts_root = _real_source_directory(
        root / "_receipts", "source receipt directory", root=root
    )
    try:
        receipt_entries = sorted(receipts_root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise PreparationError("cannot enumerate source receipt directory") from error
    for path in receipt_entries:
        _regular_source_file(path, "source receipt entry", root=receipts_root)
    bindings: list[DatasetBinding] = []
    for row in rows:
        top_row = top_by_id[row.dataset_id]
        if top_row.get("repo_id") != row.repo_id or top_row.get("revision") != row.revision:
            raise PreparationError(f"THULab mirror identity mismatch for {row.dataset_id}")
        receipt_path = _regular_source_file(
            receipts_root / f"{row.dataset_id}.json",
            f"source receipt for {row.dataset_id}",
            root=receipts_root,
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "thulab-timestamp-dataset-download-receipt-v1"
            or receipt.get("status") != "complete"
            or receipt.get("dataset_id") != row.dataset_id
            or receipt.get("repo_id") != row.repo_id
            or receipt.get("revision") != row.revision
        ):
            raise PreparationError(f"invalid source receipt identity for {row.dataset_id}")
        if row.dataset_id not in selected_id_set:
            continue
        dataset_dir = _real_source_directory(
            root / row.dataset_id,
            f"source dataset directory for {row.dataset_id}",
            root=root,
        )
        actual_files = _eligible_files(dataset_dir, mirror_root=root)
        if actual_files != receipt.get("files"):
            raise PreparationError(f"source receipt/file inventory mismatch for {row.dataset_id}")
        tsfiles = tuple(
            (item["path"], dataset_dir / item["path"], int(item["bytes"]), item["sha256"])
            for item in actual_files if str(item["path"]).endswith(".tsfile")
        )
        if not tsfiles:
            raise PreparationError(f"source receipt contains no TsFiles for {row.dataset_id}")
        bindings.append(DatasetBinding(
            row=row,
            source_receipt=receipt,
            source_receipt_sha256=sha256_file(receipt_path),
            source_files=tsfiles,
            policy=policies.get(row.dataset_id),
            scope_id=str(scope["scope_id"]),
            scope_sha256=scope_sha256,
            policy_file_sha256=policy_file_sha256,
            dataset_byte_cap=dataset_byte_cap,
            chunk_byte_cap=chunk_byte_cap,
            metadata_compressed_byte_cap=metadata_compressed_byte_cap,
            storage_layout=storage_layout,
            producer_sha256=producer_sha256,
            helper_timeseries_sha256=helper_timeseries_sha256,
            helper_tsfile_sha256=helper_tsfile_sha256,
        ))
    return tuple(bindings)


def _convert_timestamp(value: Any, source_unit: str, output_unit: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PreparationError(f"TsFile Time is not an integer: {value!r}") from error
    if source_unit == "d":
        if output_unit != "us":
            raise PreparationError("integer epoch day ordinals may only convert to us")
        result = integer * 86_400 * 1_000_000
    else:
        source_exponent = UNIT_EXPONENT[source_unit]
        output_exponent = UNIT_EXPONENT[output_unit]
        if output_exponent < source_exponent:
            raise PreparationError(
                f"timestamp conversion {source_unit}->{output_unit} would downsample precision"
            )
        result = integer * (10 ** (output_exponent - source_exponent))
    if not INT64_MIN <= result <= INT64_MAX:
        raise PreparationError(
            f"converted Unix epoch {output_unit} timestamp exceeds signed int64"
        )
    return result


def _parse_temporal_string(value: Any, spec: TemporalStringField) -> int:
    if not isinstance(value, str):
        raise PreparationError(
            f"temporal field {spec.source_name!r} expected string, got {type(value).__name__}"
        )
    text = value.strip()
    if not text:
        raise PreparationError(f"temporal field {spec.source_name!r} contains an empty value")
    if spec.parse_kind == "integer_epoch":
        if re.fullmatch(r"[+-]?\d+", text) is None:
            raise PreparationError(
                f"temporal field {spec.source_name!r} is not an exact integer epoch: {value!r}"
            )
        assert spec.source_unit is not None
        return _convert_timestamp(int(text), spec.source_unit, "us")
    assert spec.datetime_format is not None
    parsed: datetime | None = None
    for accepted_format in spec.accepted_datetime_formats:
        try:
            parsed = datetime.strptime(text, accepted_format)
            break
        except ValueError:
            continue
    if parsed is None:
        raise PreparationError(
            f"temporal field {spec.source_name!r} does not match frozen format contract "
            f"{list(spec.accepted_datetime_formats)!r}: {value!r}"
        )
    if spec.timezone_name == "embedded_offset":
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise PreparationError(
                f"temporal field {spec.source_name!r} policy requires an embedded UTC offset"
            )
        aware = parsed
    else:
        if parsed.tzinfo is not None:
            raise PreparationError(
                f"temporal field {spec.source_name!r} unexpectedly embeds an offset"
            )
        zone = ZoneInfo(spec.timezone_name)
        offset0 = parsed.replace(tzinfo=zone, fold=0).utcoffset()
        offset1 = parsed.replace(tzinfo=zone, fold=1).utcoffset()
        if offset0 != offset1 and spec.ambiguous_fold is None:
            raise PreparationError(
                f"temporal field {spec.source_name!r} is DST-ambiguous without ambiguous_fold"
            )
        aware = parsed.replace(tzinfo=zone, fold=spec.ambiguous_fold or 0)
        roundtrip = aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
        if roundtrip != parsed:
            raise PreparationError(
                f"temporal field {spec.source_name!r} is a nonexistent civil time"
            )
    utc_value = aware.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_value - epoch
    result = ((delta.days * 86400 + delta.seconds) * 1_000_000) + delta.microseconds
    if not INT64_MIN <= result <= INT64_MAX:
        raise PreparationError(
            f"temporal field {spec.source_name!r} exceeds signed int64 Unix microseconds"
        )
    return result


def _convert_temporal_value(
    value: Any, spec: TemporalStringField | SourceBoundTemporalField
) -> int:
    if isinstance(spec, TemporalStringField):
        return _parse_temporal_string(value, spec)
    if isinstance(value, bool):
        raise PreparationError(
            f"source-bound temporal field {spec.source_name!r} expected integer, got bool"
        )
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise PreparationError(
            f"source-bound temporal field {spec.source_name!r} expected integer, "
            f"got {type(value).__name__}"
        ) from error
    return _convert_timestamp(integer, spec.source_unit, spec.output_unit)


def _pack_field_value(value: Any, field: FieldSpec) -> bytes:
    if field.temporal is not None:
        return struct.pack("<q", _convert_temporal_value(value, field.temporal))
    return _pack_value(value, field.packer)


def _field_conversion_label(field: FieldSpec) -> str:
    if isinstance(field.temporal, TemporalStringField):
        return "explicit_temporal_string_to_int64_unix_us"
    if isinstance(field.temporal, SourceBoundTemporalField):
        return "source_bound_integer_epoch_to_int64_unix_us"
    return "native_numeric_little_endian"


def _resolve_table_schema(
    schemas: Any, configured_name: str, source_label: str
) -> tuple[str, Any]:
    if not isinstance(schemas, Mapping) or not schemas:
        raise PreparationError(
            f"table model required but no table schema was found in {source_label}"
        )
    if configured_name == "__single_table__":
        if len(schemas) != 1:
            raise PreparationError(
                f"{source_label} must contain exactly one table for __single_table__, "
                f"found {sorted(map(str, schemas))}"
            )
        actual_name = next(iter(schemas))
        return str(actual_name), schemas[actual_name]
    schema = schemas.get(configured_name)
    if schema is None:
        raise PreparationError(
            f"policy table {configured_name!r} absent in {source_label}; "
            f"available={sorted(map(str, schemas))}"
        )
    return configured_name, schema


def _schema_for_binding(binding: DatasetBinding) -> tuple[SourcePlan, ...]:
    """Build one independently validated plan per frozen source file/table."""
    policy = binding.policy
    assert policy is not None and policy.table_name is not None
    tsfile = _load_tsfile_module()
    temporal_by_name = {
        field.source_name: field for field in policy.temporal_string_fields
    }
    plans: list[SourcePlan] = []
    matched_rule_counts = {rule.rule_id: 0 for rule in policy.source_rules}
    matched_source_temporal_counts = {
        (field.source_path, field.source_name): 0
        for field in policy.source_bound_temporal_fields
    }
    for relative, path, _size, digest in binding.source_files:
        reader = None
        try:
            _validate_tsfile_container(path)
            reader = tsfile.TsFileReader(str(path))
            actual_table_name, schema = _resolve_table_schema(
                reader.get_all_table_schemas(), policy.table_name, relative
            )
            columns = _table_schema_columns(schema)
            schema_basis = [
                {"name": name, "tsfile_type": dtype, "category": category}
                for name, (dtype, category) in columns.items()
            ]
            fingerprint = _sha256_bytes(_canonical_json(schema_basis))
            source_bound_temporal: list[SourceBoundTemporalField] = []
            for temporal_field in policy.source_bound_temporal_fields:
                if temporal_field.source_path != relative:
                    continue
                if temporal_field.source_sha256 != digest:
                    raise PreparationError(
                        f"source-bound temporal field {temporal_field.source_name!r} "
                        "source sha256 differs from the frozen binding"
                    )
                if temporal_field.table_name != actual_table_name:
                    raise PreparationError(
                        f"source-bound temporal field {temporal_field.source_name!r} "
                        f"table is {actual_table_name!r}, expected "
                        f"{temporal_field.table_name!r}"
                    )
                if temporal_field.source_schema_sha256 != fingerprint:
                    raise PreparationError(
                        f"source-bound temporal field {temporal_field.source_name!r} "
                        "schema fingerprint differs from the frozen binding"
                    )
                actual = columns.get(temporal_field.source_name)
                if actual != (
                    temporal_field.source_type,
                    temporal_field.source_category,
                ):
                    raise PreparationError(
                        f"source-bound temporal field {temporal_field.source_name!r} "
                        f"schema is {actual!r}, expected "
                        f"({temporal_field.source_type!r}, "
                        f"{temporal_field.source_category!r})"
                    )
                source_bound_temporal.append(temporal_field)
                matched_source_temporal_counts[
                    (temporal_field.source_path, temporal_field.source_name)
                ] += 1
            source_rule: SourceRule | None = None
            if policy.decision == "admit_dataset_adapter":
                matches = [
                    rule for rule in policy.source_rules
                    if actual_table_name in rule.table_names
                ]
                if len(matches) != 1:
                    raise PreparationError(
                        f"actual table {actual_table_name!r} matched {len(matches)} "
                        "mixed-axis source rules"
                    )
                source_rule = matches[0]
                matched_rule_counts[source_rule.rule_id] += 1
            source_action = (
                source_rule.action if source_rule is not None else "materialize"
            )
            timestamp_source_column = (
                source_rule.source_column
                if source_rule is not None
                else policy.source_column
            )
            timestamp_source_unit = (
                source_rule.source_unit
                if source_rule is not None
                else policy.source_unit
            )
            timestamp_output_unit = (
                source_rule.output_unit
                if source_rule is not None
                else policy.output_unit
            )
            timestamp_output_column = (
                source_rule.output_column
                if source_rule is not None
                else policy.output_column
            )
            applicable_temporal: list[
                TemporalStringField | SourceBoundTemporalField
            ] = list(source_bound_temporal)
            source_bound_temporal_by_name = {
                field.source_name: field for field in source_bound_temporal
            }
            for temporal_field in policy.temporal_string_fields:
                actual = columns.get(temporal_field.source_name)
                if actual is None:
                    continue
                if temporal_field.source_name in source_bound_temporal_by_name:
                    continue
                if actual != (
                    temporal_field.source_type, temporal_field.source_category
                ):
                    raise PreparationError(
                        f"temporal field {temporal_field.source_name!r} schema is {actual!r}, "
                        f"expected ({temporal_field.source_type!r}, "
                        f"{temporal_field.source_category!r})"
                    )
                applicable_temporal.append(temporal_field)
            applicable_explicit_drops: list[DroppedTimestampLikeField] = []
            for dropped_field in policy.dropped_timestamp_like_string_fields:
                actual = columns.get(dropped_field.source_name)
                if actual is None:
                    continue
                if actual != (
                    dropped_field.source_type, dropped_field.source_category
                ):
                    raise PreparationError(
                        f"explicit temporal drop {dropped_field.source_name!r} schema is {actual!r}, "
                        f"expected ({dropped_field.source_type!r}, "
                        f"{dropped_field.source_category!r})"
                    )
                applicable_explicit_drops.append(dropped_field)
            tag_names = tuple(
                name for name, (_dtype, category) in columns.items()
                if category == "TAG"
            )
            primary_temporal: TemporalStringField | None = None
            if source_action == "materialize" and timestamp_source_column != "Time":
                primary_temporal = temporal_by_name.get(str(timestamp_source_column))
                if primary_temporal is None or primary_temporal not in applicable_temporal:
                    raise PreparationError(
                        f"primary temporal field {timestamp_source_column!r} is absent"
                    )
            applicable_temporal_by_name = {
                field.source_name: field for field in applicable_temporal
            }
            if source_rule is not None and source_rule.row_filter is not None:
                for condition in source_rule.row_filter.conditions:
                    if condition.source_category == "TIME":
                        if condition.source_name != "Time" or condition.source_type != "INT64":
                            raise PreparationError("physical TIME filter must name INT64 Time")
                        continue
                    actual = columns.get(condition.source_name)
                    if actual != (condition.source_type, condition.source_category):
                        raise PreparationError(
                            f"row-filter field {condition.source_name!r} schema is {actual!r}, "
                            f"expected ({condition.source_type!r}, "
                            f"{condition.source_category!r})"
                        )
            fields_list: list[FieldSpec] = []
            for name, (dtype, category) in columns.items():
                if source_action != "materialize":
                    continue
                temporal = applicable_temporal_by_name.get(name)
                if temporal is not None:
                    fields_list.append(FieldSpec(
                        name,
                        dtype,
                        output_name=temporal.output_name,
                        temporal=temporal,
                    ))
                elif category == "FIELD" and dtype in SUPPORTED_FIELDS:
                    fields_list.append(FieldSpec(name, dtype))
            fields = tuple(fields_list)
            output_names = [field.emitted_name for field in fields]
            if len(output_names) != len(set(output_names)):
                raise PreparationError("retained output column names are not unique")
            dropped: list[dict[str, Any]] = []
            applicable_explicit_drop_by_name = {
                field.source_name: field for field in applicable_explicit_drops
            }
            for name, (dtype, category) in columns.items():
                if source_action == "exclude_entire_source_file":
                    assert source_rule is not None
                    dropped.append({
                        "name": name,
                        "tsfile_type": dtype,
                        "category": category,
                        "decision": "exclude_entire_source_file_by_dataset_adapter",
                        "adapter_rule_id": source_rule.rule_id,
                        "reason": source_rule.reason,
                        "evidence": list(source_rule.evidence),
                    })
                    continue
                if name in {field.source_name for field in applicable_temporal}:
                    decision = (
                        "partition_entity_and_parse_as_primary_int64_unix_us_"
                        "timestamp_and_value_field"
                        if category == "TAG" and name == timestamp_source_column
                        else "partition_entity_and_parse_as_int64_unix_us_value_field"
                        if category == "TAG"
                        else "parse_as_primary_int64_unix_us_timestamp_and_value_field"
                        if name == timestamp_source_column
                        else "parse_as_int64_unix_us_value_field"
                    )
                elif name in applicable_explicit_drop_by_name:
                    explicit_drop = applicable_explicit_drop_by_name[name]
                    dropped.append({
                        "name": name,
                        "tsfile_type": dtype,
                        "category": category,
                        "decision": explicit_drop.decision,
                        "format_class": explicit_drop.format_class,
                        "reason": explicit_drop.reason,
                        "applicability": explicit_drop.applicability,
                        "candidate_definition_count": (
                            explicit_drop.candidate_definition_count
                        ),
                        "evidence": list(explicit_drop.evidence),
                        "partition_role": (
                            "entity" if category == "TAG" else "none"
                        ),
                    })
                    continue
                elif category == "TAG":
                    decision = "partition_entity_then_drop_from_payload"
                elif category == "FIELD" and dtype not in SUPPORTED_FIELDS:
                    decision = (
                        "drop_non_numeric_field"
                        if dtype in KNOWN_NON_NUMERIC_FIELDS
                        else "drop_unsupported_field_type"
                    )
                elif category != "FIELD":
                    decision = "drop_non_field_column"
                else:
                    continue
                dropped.append({
                    "name": name,
                    "tsfile_type": dtype,
                    "category": category,
                    "decision": decision,
                })
            plans.append(SourcePlan(
                source_path=relative,
                path=path,
                expected_sha256=digest,
                table_name=actual_table_name,
                schema_sha256=fingerprint,
                tag_names=tag_names,
                fields=fields,
                primary_temporal=primary_temporal,
                dropped_columns=tuple(dropped),
                source_action=source_action,
                source_action_reason=(
                    source_rule.reason if source_rule is not None else None
                ),
                adapter_rule_id=(
                    source_rule.rule_id if source_rule is not None else None
                ),
                timestamp_source_column=timestamp_source_column,
                timestamp_source_unit=timestamp_source_unit,
                timestamp_output_unit=timestamp_output_unit,
                timestamp_output_column=timestamp_output_column,
                timestamp_event_utc_instant_claim=(
                    source_rule.event_utc_instant_claim
                    if source_rule is not None
                    else policy.decision == "admit_exact_utc"
                ),
                timestamp_semantic_kind=(
                    source_rule.semantic_kind
                    if source_rule is not None
                    else (
                        "unix_epoch_instant"
                        if policy.decision == "admit_exact_utc"
                        else "stored_epoch_coordinate"
                    )
                ),
                timestamp_timezone_basis=(
                    source_rule.timezone_basis
                    if source_rule is not None
                    else (
                        "explicit_utc_event_instant"
                        if policy.decision == "admit_exact_utc"
                        else "upstream_encoded_epoch"
                    )
                ),
                timestamp_timezone_inferred=(
                    source_rule.timezone_inferred
                    if source_rule is not None
                    else False
                ),
                row_filter=(
                    source_rule.row_filter if source_rule is not None else None
                ),
            ))
        except PreparationError as error:
            raise PreparationError(
                f"dataset={binding.row.dataset_id}, source_file={relative}: {error}"
            ) from error
        except Exception as error:
            raise PreparationError(
                f"TsFile reader failure for {relative}: {type(error).__name__}: {error}"
            ) from error
        finally:
            if reader is not None:
                reader.close()
    for rule in policy.source_rules:
        if matched_rule_counts[rule.rule_id] != rule.expected_source_file_count:
            raise PreparationError(
                f"dataset={binding.row.dataset_id}, adapter_rule={rule.rule_id}: expected "
                f"{rule.expected_source_file_count} source files but matched "
                f"{matched_rule_counts[rule.rule_id]}"
            )
    for binding_key, matched_count in matched_source_temporal_counts.items():
        if matched_count != 1:
            raise PreparationError(
                f"dataset={binding.row.dataset_id}, source_bound_temporal={binding_key!r}: "
                f"expected exactly one frozen source match, found {matched_count}"
            )
    return tuple(plans)


def _result_layout(
    result: Any, *, tag_names: Sequence[str], fields: Sequence[FieldSpec]
) -> tuple[list[tuple[int, str]], list[tuple[int, FieldSpec]]]:
    metadata = result.get_metadata()
    names = list(metadata.get_column_list())
    types = [_type_name(value) for value in metadata.get_data_type_list()]
    if not names or names[0].casefold() != "time" or types[0] != "INT64":
        raise PreparationError("query result must begin with an INT64 Time coordinate")
    indexes = {name: index + 1 for index, name in enumerate(names)}
    type_by_name = dict(zip(names, types))
    tags: list[tuple[int, str]] = []
    for name in tag_names:
        if name not in indexes:
            raise PreparationError(f"TAG {name!r} is missing from query result")
        tags.append((indexes[name], type_by_name[name]))
    values: list[tuple[int, FieldSpec]] = []
    for field in fields:
        if field.name not in indexes or type_by_name[field.name] != field.tsfile_type:
            raise PreparationError(f"FIELD {field.name!r} query type differs from schema")
        values.append((indexes[field.name], field))
    return tags, values


def _iter_source_rows(
    binding: DatasetBinding,
    plan: SourcePlan,
    filter_stats: dict[str, Any] | None = None,
    *,
    pack_values: bool = True,
) -> Iterable[tuple[tuple[str, ...], int, tuple[bytes | None, ...]]]:
    policy = binding.policy
    assert policy is not None and policy.table_name is not None
    if plan.source_action != "materialize":
        raise PreparationError("excluded source plan cannot be queried for materialization")
    assert plan.timestamp_source_column is not None
    assert plan.timestamp_output_unit is not None
    tsfile = _load_tsfile_module()
    if sha256_file(plan.path) != plan.expected_sha256:
        raise PreparationError(
            f"dataset={binding.row.dataset_id}, source_file={plan.source_path}: "
            "source changed after mirror binding"
        )
    reader = None
    result = None
    try:
        reader = tsfile.TsFileReader(str(plan.path))
        actual_table_name, _schema = _resolve_table_schema(
            reader.get_all_table_schemas(), policy.table_name, plan.source_path
        )
        if actual_table_name != plan.table_name:
            raise PreparationError("source table changed after schema audit")
        query_columns = list(dict.fromkeys([
            *plan.tag_names,
            *(field.name for field in plan.fields),
            *(
                condition.source_name
                for condition in (() if plan.row_filter is None else plan.row_filter.conditions)
                if condition.source_category != "TIME"
            ),
        ]))
        result = reader.query_table(
            plan.table_name,
            query_columns,
            INT64_MIN,
            INT64_MAX,
        )
        tag_layout, value_layout = _result_layout(
            result, tag_names=plan.tag_names, fields=plan.fields
        )
        metadata_names = list(result.get_metadata().get_column_list())
        result_indexes = {
            name: index + 1 for index, name in enumerate(metadata_names)
        }
        value_indexes = {field.name: index for index, field in value_layout}
        while result.next():
            if filter_stats is not None:
                filter_stats["input_rows"] += 1
            failed_conditions: list[RowFilterCondition] = []
            if plan.row_filter is not None:
                for condition in plan.row_filter.conditions:
                    index = 1 if condition.source_category == "TIME" else result_indexes[
                        condition.source_name
                    ]
                    is_null = result.is_null_by_index(index)
                    keep = not is_null
                    if condition.operator == "int64_gte" and keep:
                        assert condition.value is not None
                        keep = int(result.get_value_by_index(index)) >= condition.value
                    if not keep:
                        failed_conditions.append(condition)
                if failed_conditions:
                    if filter_stats is not None:
                        filter_stats["dropped_rows"] += 1
                        first_reason = failed_conditions[0].drop_reason
                        first_counts = filter_stats["first_failed_condition_counts"]
                        first_counts[first_reason] = first_counts.get(first_reason, 0) + 1
                        failure_counts = filter_stats["condition_failure_counts"]
                        for condition in failed_conditions:
                            reason = condition.drop_reason
                            failure_counts[reason] = failure_counts.get(reason, 0) + 1
                    continue
            if filter_stats is not None:
                filter_stats["output_rows"] += 1
            if plan.primary_temporal is None:
                assert plan.timestamp_source_unit is not None
                timestamp = _convert_timestamp(
                    result.get_value_by_index(1),
                    plan.timestamp_source_unit,
                    plan.timestamp_output_unit,
                )
            else:
                primary_index = value_indexes[plan.primary_temporal.source_name]
                if result.is_null_by_index(primary_index):
                    raise PreparationError(
                        f"primary temporal field {plan.primary_temporal.source_name!r} is null"
                    )
                timestamp = _parse_temporal_string(
                    result.get_value_by_index(primary_index), plan.primary_temporal
                )
            if tag_layout:
                components: list[str] = []
                for index, dtype in tag_layout:
                    if result.is_null_by_index(index):
                        raise PreparationError("TAG values cannot be null")
                    components.append(
                        _typed_entity_component(dtype, result.get_value_by_index(index))
                    )
                entity = tuple(components)
            else:
                entity = ("__single_entity__",)
            packed: list[bytes | None] = []
            for index, field in value_layout:
                if result.is_null_by_index(index):
                    if pack_values:
                        packed.append(None)
                    continue
                value = result.get_value_by_index(index)
                # A count-only pass still validates every admitted temporal
                # conversion, but deliberately avoids constructing native
                # numeric payload bytes.  The materialization pass performs
                # the authoritative native value packing.
                if not pack_values:
                    if field.temporal is not None:
                        _convert_temporal_value(value, field.temporal)
                    continue
                packed.append(_pack_field_value(value, field))
            yield entity, timestamp, tuple(packed)
    except PreparationError as error:
        raise PreparationError(
            f"dataset={binding.row.dataset_id}, source_file={plan.source_path}: {error}"
        ) from error
    except Exception as error:
        raise PreparationError(
            f"TsFile query failure for {plan.source_path}: {type(error).__name__}: {error}"
        ) from error
    finally:
        if result is not None:
            result.close()
        if reader is not None:
            reader.close()
    if sha256_file(plan.path) != plan.expected_sha256:
        raise PreparationError(
            f"dataset={binding.row.dataset_id}, source_file={plan.source_path}: "
            "source changed during materialization"
        )


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized[:80] or "column"


class _SqliteEntityStore:
    """Disk-backed entity counts and prefix selections with bounded caches."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript("""
            CREATE TABLE entities (
                plan_ordinal INTEGER NOT NULL,
                entity_key BLOB NOT NULL,
                row_count INTEGER NOT NULL,
                PRIMARY KEY (plan_ordinal, entity_key)
            ) WITHOUT ROWID;
            CREATE TABLE selected_entities (
                plan_ordinal INTEGER NOT NULL,
                entity_key BLOB NOT NULL,
                entity_ordinal INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                selected_field_count INTEGER NOT NULL,
                PRIMARY KEY (plan_ordinal, entity_key),
                UNIQUE (plan_ordinal, entity_ordinal)
            );
            CREATE TABLE plan_selections (
                plan_ordinal INTEGER PRIMARY KEY,
                base_stream_index INTEGER NOT NULL,
                selected_stream_count INTEGER NOT NULL,
                selected_entity_count INTEGER NOT NULL
            );
        """)
        self.pending: OrderedDict[tuple[int, bytes], int] = OrderedDict()
        self.peak_count_cache_entries = 0

    def increment(self, plan_ordinal: int, entity: tuple[str, ...]) -> None:
        key = (plan_ordinal, _canonical_json(list(entity)))
        self.pending[key] = self.pending.get(key, 0) + 1
        self.peak_count_cache_entries = max(
            self.peak_count_cache_entries, len(self.pending)
        )
        if len(self.pending) >= ENTITY_COUNT_CACHE_LIMIT:
            self.flush_counts()

    def flush_counts(self) -> None:
        if not self.pending:
            return
        self.connection.executemany(
            """
            INSERT INTO entities(plan_ordinal, entity_key, row_count)
            VALUES (?, ?, ?)
            ON CONFLICT(plan_ordinal, entity_key)
            DO UPDATE SET row_count = row_count + excluded.row_count
            """,
            ((plan, key, count) for (plan, key), count in self.pending.items()),
        )
        self.connection.commit()
        self.pending.clear()

    def entity_count(self, plan_ordinal: int) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM entities WHERE plan_ordinal = ?", (plan_ordinal,)
        ).fetchone()
        assert row is not None
        return int(row[0])

    def iter_entities(self, plan_ordinal: int) -> Iterable[tuple[bytes, int]]:
        cursor = self.connection.execute(
            """
            SELECT entity_key, row_count FROM entities
            WHERE plan_ordinal = ? ORDER BY entity_key
            """,
            (plan_ordinal,),
        )
        for key, count in cursor:
            yield bytes(key), int(count)

    def add_selected_entity(
        self,
        plan_ordinal: int,
        entity_key: bytes,
        entity_ordinal: int,
        row_count: int,
        selected_field_count: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO selected_entities(
                plan_ordinal, entity_key, entity_ordinal, row_count,
                selected_field_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                plan_ordinal, entity_key, entity_ordinal, row_count,
                selected_field_count,
            ),
        )

    def add_plan_selection(
        self,
        plan_ordinal: int,
        base_stream_index: int,
        selected_stream_count: int,
        selected_entity_count: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO plan_selections(
                plan_ordinal, base_stream_index, selected_stream_count,
                selected_entity_count
            ) VALUES (?, ?, ?, ?)
            """,
            (
                plan_ordinal, base_stream_index, selected_stream_count,
                selected_entity_count,
            ),
        )
        self.connection.commit()

    def iter_plan_selections(self) -> Iterable[tuple[int, int, int, int]]:
        for row in self.connection.execute(
            """
            SELECT plan_ordinal, base_stream_index, selected_stream_count,
                   selected_entity_count
            FROM plan_selections ORDER BY base_stream_index
            """
        ):
            yield int(row[0]), int(row[1]), int(row[2]), int(row[3])

    def selected_entity_by_key(
        self, plan_ordinal: int, entity_key: bytes
    ) -> tuple[int, int, int] | None:
        row = self.connection.execute(
            """
            SELECT entity_ordinal, row_count, selected_field_count
            FROM selected_entities
            WHERE plan_ordinal = ? AND entity_key = ?
            """,
            (plan_ordinal, entity_key),
        ).fetchone()
        if row is None:
            return None
        return int(row[0]), int(row[1]), int(row[2])

    def selected_entities_by_ordinal(
        self, plan_ordinal: int, first: int, last: int
    ) -> dict[int, tuple[bytes, int, int]]:
        return {
            int(ordinal): (bytes(key), int(row_count), int(field_count))
            for key, ordinal, row_count, field_count in self.connection.execute(
                """
                SELECT entity_key, entity_ordinal, row_count,
                       selected_field_count
                FROM selected_entities
                WHERE plan_ordinal = ? AND entity_ordinal BETWEEN ? AND ?
                ORDER BY entity_ordinal
                """,
                (plan_ordinal, first, last),
            )
        }

    def total_entity_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM entities").fetchone()
        assert row is not None
        return int(row[0])

    def entity_storage_basis(self) -> tuple[int, int]:
        """Return deterministic entity count/key bytes for temp-space bounds."""
        row = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(entity_key)), 0) FROM entities"
        ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1])

    def iter_selected_entities(
        self, plan_ordinal: int
    ) -> Iterable[tuple[bytes, int, int, int]]:
        for key, ordinal, row_count, field_count in self.connection.execute(
            """
            SELECT entity_key, entity_ordinal, row_count, selected_field_count
            FROM selected_entities
            WHERE plan_ordinal = ? ORDER BY entity_ordinal
            """,
            (plan_ordinal,),
        ):
            yield bytes(key), int(ordinal), int(row_count), int(field_count)

    def close(self) -> None:
        self.flush_counts()
        self.connection.close()


class _SpoolHandles:
    def __init__(self, paths: Mapping[Any, Path], limit: int = 64) -> None:
        self.paths = paths
        self.limit = limit
        self.handles: OrderedDict[Any, Any] = OrderedDict()

    def write(self, key: Any, payload: bytes) -> None:
        handle = self.handles.pop(key, None)
        if handle is None:
            if len(self.handles) >= self.limit:
                _old_key, old_handle = self.handles.popitem(last=False)
                old_handle.close()
            handle = self.paths[key].open("ab")
        handle.write(payload)
        self.handles[key] = handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def _write_chunk(
    root: Path,
    *,
    stream_index: int,
    chunk_index: int,
    field: FieldSpec,
    timestamps: bytearray,
    values: bytearray,
    bitmap: bytearray,
    rows: int,
    valid: int,
    first_timestamp: int,
    last_timestamp: int,
    timestamp_unit: str,
) -> dict[str, Any]:
    directory = root / "streams" / f"s-{stream_index:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"chunk-{chunk_index:06d}"
    payloads = {
        "timestamps": (directory / f"{stem}.timestamps.i64le", bytes(timestamps)),
        "values": (directory / f"{stem}.values.{field.dtype}le", bytes(values)),
        "missing_bitmap": (directory / f"{stem}.missing.bitmap", bytes(bitmap)),
    }
    record: dict[str, Any] = {
        "chunk_index": chunk_index,
        "row_count": rows,
        "valid_count": valid,
        "missing_count": rows - valid,
        f"first_timestamp_unix_{timestamp_unit}": first_timestamp,
        f"last_timestamp_unix_{timestamp_unit}": last_timestamp,
        "timestamp_unit": timestamp_unit,
        "timestamp_dtype": "int64",
        "value_dtype": field.dtype,
        "byte_order": "little",
        "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
    }
    for name, (path, payload) in payloads.items():
        path.write_bytes(payload)
        record[f"{name}_path"] = path.relative_to(root).as_posix()
        record[f"{name}_bytes"] = len(payload)
        record[f"{name}_sha256"] = _sha256_bytes(payload)
    return record


class _ShardWriter:
    """Append immutable payload segments to bounded, dataset-level shards."""

    def __init__(self, root: Path, byte_cap: int) -> None:
        self.root = root
        self.byte_cap = byte_cap
        self.directory = root / "payload"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle: Any | None = None
        self.relative: str | None = None
        self.size = 0
        self.digest: Any | None = None
        self.shards: list[dict[str, Any]] = []

    def _open(self) -> None:
        shard_index = len(self.shards)
        self.relative = f"payload/shard-{shard_index:06d}.bin"
        self.handle = (self.root / self.relative).open("xb")
        self.size = 0
        self.digest = hashlib.sha256()

    def _close_current(self) -> None:
        if self.handle is None:
            return
        assert self.relative is not None and self.digest is not None
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.shards.append({
            "shard_index": len(self.shards),
            "path": self.relative,
            "bytes": self.size,
            "sha256": self.digest.hexdigest(),
        })
        self.handle = None
        self.relative = None
        self.size = 0
        self.digest = None

    def append(self, payload: bytes) -> dict[str, Any]:
        if len(payload) > self.byte_cap:
            raise PreparationError("payload segment exceeds shard byte cap")
        if self.handle is None:
            self._open()
        elif payload and self.size + len(payload) > self.byte_cap:
            self._close_current()
            self._open()
        assert self.handle is not None and self.relative is not None and self.digest is not None
        offset = self.size
        if payload:
            self.handle.write(payload)
            self.digest.update(payload)
            self.size += len(payload)
        return {
            "path": self.relative,
            "offset": offset,
            "length": len(payload),
            "sha256": _sha256_bytes(payload),
        }

    def close(self) -> list[dict[str, Any]]:
        self._close_current()
        return list(self.shards)

    def abort(self) -> None:
        """Close an unfinished shard without publishing a shard receipt."""
        if self.handle is None:
            return
        try:
            self.handle.close()
        finally:
            self.handle = None
            self.relative = None
            self.size = 0
            self.digest = None


def _write_sharded_chunk(
    shard_writer: _ShardWriter,
    *,
    chunk_index: int,
    field: FieldSpec,
    timestamps: bytearray,
    values: bytearray,
    bitmap: bytearray,
    rows: int,
    valid: int,
    first_timestamp: int,
    last_timestamp: int,
    timestamp_unit: str,
) -> dict[str, Any]:
    payloads = {
        "timestamps": bytes(timestamps),
        "values": bytes(values),
        "missing_bitmap": bytes(bitmap),
    }
    record: dict[str, Any] = {
        "chunk_index": chunk_index,
        "row_count": rows,
        "valid_count": valid,
        "missing_count": rows - valid,
        f"first_timestamp_unix_{timestamp_unit}": first_timestamp,
        f"last_timestamp_unix_{timestamp_unit}": last_timestamp,
        "timestamp_unit": timestamp_unit,
        "timestamp_dtype": "int64",
        "value_dtype": field.dtype,
        "byte_order": "little",
        "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
    }
    for name, payload in payloads.items():
        record[f"{name}_segment"] = shard_writer.append(payload)
        record[f"{name}_bytes"] = len(payload)
        record[f"{name}_sha256"] = _sha256_bytes(payload)
    return record


def _emit_stream(
    root: Path,
    *,
    stream_index: int,
    entity: tuple[str, ...],
    field: FieldSpec,
    spool_path: Path,
    expected_rows: int,
    chunk_byte_cap: int,
    timestamp_unit: str,
    source_path: str,
    table_name: str,
    schema_sha256: str,
    shard_writer: _ShardWriter | None = None,
    include_entity_values: bool = True,
    spool_offset: int = 0,
    fixed_width_records: bool = False,
) -> dict[str, Any]:
    max_rows = max(1, min(chunk_byte_cap // 8, chunk_byte_cap // field.width, chunk_byte_cap * 8))
    chunks: list[dict[str, Any]] = []
    rows_total = valid_total = 0
    timestamps = bytearray()
    values = bytearray()
    bitmap = bytearray()
    rows = valid = 0
    first_timestamp: int | None = None
    last_timestamp: int | None = None

    def flush() -> None:
        nonlocal timestamps, values, bitmap, rows, valid, first_timestamp, last_timestamp
        if not rows:
            return
        assert first_timestamp is not None and last_timestamp is not None
        if shard_writer is None:
            chunk = _write_chunk(
                root,
                stream_index=stream_index,
                chunk_index=len(chunks),
                field=field,
                timestamps=timestamps,
                values=values,
                bitmap=bitmap,
                rows=rows,
                valid=valid,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                timestamp_unit=timestamp_unit,
            )
        else:
            chunk = _write_sharded_chunk(
                shard_writer,
                chunk_index=len(chunks),
                field=field,
                timestamps=timestamps,
                values=values,
                bitmap=bitmap,
                rows=rows,
                valid=valid,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                timestamp_unit=timestamp_unit,
            )
        chunks.append(chunk)
        timestamps, values, bitmap = bytearray(), bytearray(), bytearray()
        rows = valid = 0
        first_timestamp = last_timestamp = None

    with spool_path.open("rb") as handle:
        handle.seek(spool_offset)
        while rows_total < expected_rows:
            raw_timestamp = handle.read(8)
            if len(raw_timestamp) != 8:
                raise PreparationError("truncated internal timestamp spool")
            flag = handle.read(1)
            if flag not in {b"\x00", b"\x01"}:
                raise PreparationError("corrupt internal presence flag")
            timestamp = struct.unpack("<q", raw_timestamp)[0]
            raw_value = (
                handle.read(field.width)
                if fixed_width_records or flag == b"\x01"
                else b""
            )
            if (fixed_width_records or flag == b"\x01") and len(raw_value) != field.width:
                raise PreparationError("truncated internal value spool")
            value = raw_value if flag == b"\x01" else b""
            if rows >= max_rows:
                flush()
            bitmap_index, bit_index = divmod(rows, 8)
            if bitmap_index == len(bitmap):
                bitmap.append(0)
            if flag == b"\x00":
                bitmap[bitmap_index] |= 1 << bit_index
            else:
                values.extend(value)
                valid += 1
            timestamps.extend(raw_timestamp)
            first_timestamp = timestamp if first_timestamp is None else first_timestamp
            last_timestamp = timestamp
            rows += 1
            rows_total += 1
            valid_total += int(flag == b"\x01")
        if not fixed_width_records and handle.read(1):
            raise PreparationError("internal stream spool contains trailing records")
    flush()
    if rows_total != expected_rows:
        raise PreparationError("internal selected stream row count mismatch")
    record = {
        "stream_index": stream_index,
        "stream_id": f"s-{stream_index:08d}",
        "entity_id": _entity_digest(entity),
        "source_path": source_path,
        "table_name": table_name,
        "schema_sha256": schema_sha256,
        "field": field.emitted_name,
        "source_field": field.name,
        "tsfile_type": field.tsfile_type,
        "dtype": field.dtype,
        "width_bytes": field.width,
        "conversion": _field_conversion_label(field),
        "row_count": rows_total,
        "valid_count": valid_total,
        "missing_count": rows_total - valid_total,
        "logical_bytes": rows_total * (8 + field.width),
        "chunks": chunks,
    }
    if include_entity_values:
        record["entity_values"] = list(entity)
    return record


class _StreamIndexWriter:
    """Write a deterministic gzip JSONL stream index without retaining it in RAM."""

    def __init__(self, root: Path, binding: DatasetBinding) -> None:
        self.relative = "stream_index.jsonl.gz"
        self.path = root / self.relative
        self.raw = self.path.open("xb")
        self.gzip = gzip.GzipFile(
            filename="stream_index.jsonl",
            mode="wb",
            fileobj=self.raw,
            mtime=0,
        )
        self.uncompressed_digest = hashlib.sha256()
        self.uncompressed_bytes = 0
        self.record_count = 0
        self.closed = False
        self.metadata_compressed_byte_cap = binding.metadata_compressed_byte_cap
        try:
            self.write({
                "record_type": "header",
                "schema": STREAM_INDEX_VERSION,
                "storage_layout": "sharded_v1",
                "binding_sha256": binding.binding_sha256,
                "dataset_id": binding.row.dataset_id,
                "producer_sha256": binding.producer_sha256,
                "helper_timeseries_sha256": binding.helper_timeseries_sha256,
                "helper_tsfile_sha256": binding.helper_tsfile_sha256,
            })
        except BaseException:
            self.abort()
            raise

    def write(self, record: Mapping[str, Any]) -> None:
        if self.closed:
            raise PreparationError("cannot write a closed stream index")
        payload = _canonical_json(record) + b"\n"
        self.gzip.write(payload)
        self.uncompressed_digest.update(payload)
        self.uncompressed_bytes += len(payload)
        self.record_count += 1
        # close() performs the authoritative exact check because the compressor
        # can retain a small buffer.  This check stops an already-over-cap index
        # from continuing to grow during generation.
        if self.raw.tell() > self.metadata_compressed_byte_cap:
            raise PreparationError(
                "compressed stream index exceeds metadata byte cap"
            )

    def close(self, totals: Mapping[str, int], entity_record_count: int) -> dict[str, Any]:
        if self.closed:
            raise PreparationError("stream index already closed")
        self.write({
            "record_type": "footer",
            "schema": STREAM_INDEX_VERSION,
            "entity_record_count": entity_record_count,
            "stream_record_count": int(totals["stream_count"]),
            "totals": dict(totals),
        })
        self.gzip.close()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        self.closed = True
        compressed_bytes = self.path.stat().st_size
        if compressed_bytes > self.metadata_compressed_byte_cap:
            raise PreparationError(
                "compressed stream index exceeds metadata byte cap"
            )
        return {
            "schema": STREAM_INDEX_VERSION,
            "path": self.relative,
            "compression": "gzip",
            "record_encoding": "canonical_json_lines_utf8",
            "record_count": self.record_count,
            "entity_record_count": entity_record_count,
            "stream_record_count": int(totals["stream_count"]),
            "compressed_bytes": compressed_bytes,
            "compressed_sha256": sha256_file(self.path),
            "uncompressed_bytes": self.uncompressed_bytes,
            "uncompressed_sha256": self.uncompressed_digest.hexdigest(),
            "compressed_byte_cap": self.metadata_compressed_byte_cap,
        }

    def abort(self) -> None:
        """Close an unfinished gzip/index pair before its temporary tree is removed."""
        if self.closed:
            return
        try:
            self.gzip.close()
        finally:
            self.raw.close()
            self.closed = True


def _safe_materialized_path(relative: Any) -> str:
    value = str(relative)
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise PreparationError("unsafe materialized file path")
    return value


def _regular_file_within(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise PreparationError(f"materialized file is missing: {relative}") from error
    if not stat.S_ISREG(mode):
        raise PreparationError(f"materialized path is not a regular file: {relative}")
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise PreparationError(f"materialized file escapes dataset root: {relative}")
    return path


def _verify_sharded_index(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    binding: DatasetBinding,
    seen_paths: set[str],
) -> dict[str, int]:
    if manifest.get("storage_layout") != "sharded_v1":
        raise PreparationError("sharded manifest storage-layout mismatch")
    raw_shards = manifest.get("payload_shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise PreparationError("sharded manifest has no payload shards")
    shards: dict[str, tuple[Path, int]] = {}
    shard_order: dict[str, int] = {}
    for expected_index, row in enumerate(raw_shards):
        if not isinstance(row, Mapping) or row.get("shard_index") != expected_index:
            raise PreparationError("payload shard index/order mismatch")
        relative = _safe_materialized_path(row.get("path"))
        if relative in seen_paths:
            raise PreparationError("duplicate materialized file path")
        seen_paths.add(relative)
        path = _regular_file_within(dataset_dir, relative)
        size = int(row.get("bytes", -1))
        if size <= 0 or size > binding.chunk_byte_cap:
            raise PreparationError("payload shard size is outside the configured cap")
        if not path.is_file() or path.stat().st_size != size:
            raise PreparationError(f"payload shard size mismatch: {relative}")
        if sha256_file(path) != row.get("sha256"):
            raise PreparationError(f"payload shard hash mismatch: {relative}")
        shards[relative] = (path, size)
        shard_order[relative] = expected_index

    index = manifest.get("stream_index")
    if not isinstance(index, Mapping) or index.get("schema") != STREAM_INDEX_VERSION:
        raise PreparationError("missing or unsupported stream index")
    relative = _safe_materialized_path(index.get("path"))
    if relative in seen_paths:
        raise PreparationError("duplicate stream-index path")
    seen_paths.add(relative)
    index_path = _regular_file_within(dataset_dir, relative)
    if index.get("compression") != "gzip" or index.get("record_encoding") != "canonical_json_lines_utf8":
        raise PreparationError("unsupported stream-index encoding")
    if int(index.get("compressed_byte_cap", -1)) != binding.metadata_compressed_byte_cap:
        raise PreparationError("stream-index metadata cap binding mismatch")
    compressed_bytes = index_path.stat().st_size
    if (
        compressed_bytes > binding.metadata_compressed_byte_cap
        or compressed_bytes != int(index.get("compressed_bytes", -1))
    ):
        raise PreparationError("stream-index compressed size mismatch")
    if sha256_file(index_path) != index.get("compressed_sha256"):
        raise PreparationError("stream-index compressed hash mismatch")

    totals = {
        "stream_count": 0,
        "chunk_count": 0,
        "rows": 0,
        "valid_values": 0,
        "missing_values": 0,
        "logical_bytes": 0,
    }
    uncompressed_digest = hashlib.sha256()
    uncompressed_bytes = 0
    record_count = 0
    entity_record_count = 0
    current_entity_key: tuple[str, str, str] | None = None
    current_entity_had_stream = False
    previous_entity_sort_key: tuple[str, str, bytes] | None = None
    footer: Mapping[str, Any] | None = None
    consumed = {path: 0 for path in shards}
    last_shard_index = 0
    segment_handle: Any | None = None
    current_segment_path: str | None = None
    try:
        with gzip.open(index_path, "rb") as handle:
            for raw_line in handle:
                record_count += 1
                uncompressed_digest.update(raw_line)
                uncompressed_bytes += len(raw_line)
                if not raw_line.endswith(b"\n"):
                    raise PreparationError("stream-index record lacks newline terminator")
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PreparationError("invalid stream-index JSON record") from error
                if not isinstance(record, Mapping) or raw_line != _canonical_json(record) + b"\n":
                    raise PreparationError("stream-index record is not canonical JSONL")
                kind = record.get("record_type")
                if record_count == 1:
                    if record != {
                        "record_type": "header",
                        "schema": STREAM_INDEX_VERSION,
                        "storage_layout": "sharded_v1",
                        "binding_sha256": binding.binding_sha256,
                        "dataset_id": binding.row.dataset_id,
                        "producer_sha256": binding.producer_sha256,
                        "helper_timeseries_sha256": binding.helper_timeseries_sha256,
                        "helper_tsfile_sha256": binding.helper_tsfile_sha256,
                    }:
                        raise PreparationError("stream-index header binding mismatch")
                    continue
                if footer is not None:
                    raise PreparationError("stream-index contains records after footer")
                if kind == "entity":
                    source_path = str(record.get("source_path", ""))
                    table_name = str(record.get("table_name", ""))
                    entity_id = str(record.get("entity_id", ""))
                    entity_values = record.get("entity_values")
                    entity_sort_key = (
                        source_path, table_name, _canonical_json(entity_values)
                    ) if isinstance(entity_values, list) else None
                    if (
                        not source_path or not table_name or not entity_id
                        or not isinstance(entity_values, list)
                        or _entity_digest(tuple(str(value) for value in entity_values)) != entity_id
                        or (current_entity_key is not None and not current_entity_had_stream)
                        or (
                            previous_entity_sort_key is not None
                            and entity_sort_key is not None
                            and entity_sort_key <= previous_entity_sort_key
                        )
                    ):
                        raise PreparationError("invalid stream-index entity dictionary record")
                    entity_key = (source_path, table_name, entity_id)
                    current_entity_key = entity_key
                    current_entity_had_stream = False
                    previous_entity_sort_key = entity_sort_key
                    entity_record_count += 1
                    continue
                if kind == "footer":
                    if current_entity_key is not None and not current_entity_had_stream:
                        raise PreparationError("stream-index entity has no streams")
                    footer = record
                    continue
                if kind != "stream" or not isinstance(record.get("stream"), Mapping):
                    raise PreparationError("unsupported stream-index record type")
                stream = record["stream"]
                if "entity_values" in stream:
                    raise PreparationError("sharded stream redundantly embeds entity string values")
                if int(stream.get("stream_index", -1)) != totals["stream_count"]:
                    raise PreparationError("stream index/order mismatch")
                if stream.get("stream_id") != f"s-{totals['stream_count']:08d}":
                    raise PreparationError("stream id/order mismatch")
                entity_key = (
                    str(stream.get("source_path", "")),
                    str(stream.get("table_name", "")),
                    str(stream.get("entity_id", "")),
                )
                if entity_key != current_entity_key:
                    raise PreparationError("stream is not adjacent to its entity dictionary record")
                current_entity_had_stream = True
                dtype = str(stream.get("dtype", ""))
                width = int(stream.get("width_bytes", -1))
                if dtype not in {"int32", "int64", "float32", "float64"}:
                    raise PreparationError("stream contains a non-numeric dtype")
                if width != {"int32": 4, "int64": 8, "float32": 4, "float64": 8}[dtype]:
                    raise PreparationError("stream dtype/width mismatch")
                conversion = stream.get("conversion")
                if conversion == "explicit_temporal_string_to_int64_unix_us":
                    source_field = str(stream.get("source_field", ""))
                    if (
                        dtype != "int64"
                        or stream.get("field")
                        != f"{source_field}__timestamp_unix_us"
                        or stream.get("tsfile_type") not in {"STRING", "TEXT"}
                    ):
                        raise PreparationError("temporal stream output contract mismatch")
                elif conversion != "native_numeric_little_endian":
                    raise PreparationError("unsupported stream conversion")
                chunks = stream.get("chunks")
                if not isinstance(chunks, list) or not chunks:
                    raise PreparationError("stream contains no chunks")
                stream_rows = stream_valid = 0
                for expected_chunk_index, chunk in enumerate(chunks):
                    if not isinstance(chunk, Mapping) or int(chunk.get("chunk_index", -1)) != expected_chunk_index:
                        raise PreparationError("chunk index/order mismatch")
                    rows = int(chunk.get("row_count", -1))
                    valid = int(chunk.get("valid_count", -1))
                    missing = int(chunk.get("missing_count", -1))
                    if rows <= 0 or valid < 0 or missing < 0 or valid + missing != rows:
                        raise PreparationError("chunk row/valid/missing counts are inconsistent")
                    if chunk.get("timestamp_unit") != "us" or chunk.get("timestamp_dtype") != "int64":
                        raise PreparationError("chunk timestamp is not signed int64 Unix microseconds")
                    payloads: dict[str, bytes] = {}
                    for name in ("timestamps", "values", "missing_bitmap"):
                        segment = chunk.get(f"{name}_segment")
                        if not isinstance(segment, Mapping):
                            raise PreparationError("chunk lacks a payload segment reference")
                        segment_path = _safe_materialized_path(segment.get("path"))
                        if segment_path not in shards:
                            raise PreparationError("chunk references an unknown payload shard")
                        offset = int(segment.get("offset", -1))
                        length = int(segment.get("length", -1))
                        shard_size = shards[segment_path][1]
                        shard_index = shard_order[segment_path]
                        if (
                            offset < 0 or length < 0 or offset + length > shard_size
                            or shard_index < last_shard_index
                            or offset != consumed[segment_path]
                        ):
                            raise PreparationError("payload segment boundary/order mismatch")
                        last_shard_index = shard_index
                        if current_segment_path != segment_path:
                            if segment_handle is not None:
                                segment_handle.close()
                            segment_handle = shards[segment_path][0].open("rb")
                            current_segment_path = segment_path
                        assert segment_handle is not None
                        segment_handle.seek(offset)
                        payload = segment_handle.read(length)
                        if len(payload) != length or _sha256_bytes(payload) != segment.get("sha256"):
                            raise PreparationError("payload segment hash mismatch")
                        if length != int(chunk.get(f"{name}_bytes", -1)) or segment.get("sha256") != chunk.get(f"{name}_sha256"):
                            raise PreparationError("payload segment metadata mismatch")
                        consumed[segment_path] += length
                        payloads[name] = payload
                    if len(payloads["timestamps"]) != rows * 8:
                        raise PreparationError("timestamp byte count mismatch")
                    if len(payloads["values"]) != valid * width:
                        raise PreparationError("value byte count mismatch")
                    if len(payloads["missing_bitmap"]) != (rows + 7) // 8:
                        raise PreparationError("missing bitmap byte count mismatch")
                    if sum(byte.bit_count() for byte in payloads["missing_bitmap"]) != missing:
                        raise PreparationError("missing bitmap population mismatch")
                    if rows % 8 and payloads["missing_bitmap"][-1] >> (rows % 8):
                        raise PreparationError("missing bitmap has nonzero padding bits")
                    first = struct.unpack_from("<q", payloads["timestamps"], 0)[0]
                    last = struct.unpack_from("<q", payloads["timestamps"], (rows - 1) * 8)[0]
                    if first != int(chunk.get("first_timestamp_unix_us", INT64_MIN)) or last != int(chunk.get("last_timestamp_unix_us", INT64_MIN)):
                        raise PreparationError("chunk timestamp boundary metadata mismatch")
                    totals["chunk_count"] += 1
                    stream_rows += rows
                    stream_valid += valid
                if (
                    stream_rows != int(stream.get("row_count", -1))
                    or stream_valid != int(stream.get("valid_count", -1))
                    or stream_rows - stream_valid != int(stream.get("missing_count", -1))
                    or stream_rows * (8 + width) != int(stream.get("logical_bytes", -1))
                ):
                    raise PreparationError("stream/chunk aggregate mismatch")
                totals["stream_count"] += 1
                totals["rows"] += stream_rows
                totals["valid_values"] += stream_valid
                totals["missing_values"] += stream_rows - stream_valid
                totals["logical_bytes"] += stream_rows * (8 + width)
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        raise PreparationError("cannot decode stream-index gzip") from error
    finally:
        if segment_handle is not None:
            segment_handle.close()

    if footer is None or footer.get("schema") != STREAM_INDEX_VERSION:
        raise PreparationError("stream-index footer is missing")
    if footer.get("totals") != totals or int(footer.get("stream_record_count", -1)) != totals["stream_count"]:
        raise PreparationError("stream-index footer totals mismatch")
    if int(footer.get("entity_record_count", -1)) != entity_record_count:
        raise PreparationError("stream-index footer entity count mismatch")
    if any(consumed[path] != size for path, (_file, size) in shards.items()):
        raise PreparationError("payload shards contain gaps or unreferenced trailing bytes")
    if (
        record_count != int(index.get("record_count", -1))
        or totals["stream_count"] != int(index.get("stream_record_count", -1))
        or entity_record_count != int(index.get("entity_record_count", -1))
        or uncompressed_bytes != int(index.get("uncompressed_bytes", -1))
        or uncompressed_digest.hexdigest() != index.get("uncompressed_sha256")
    ):
        raise PreparationError("stream-index byte/count binding mismatch")
    if entity_record_count != int(manifest.get("selected_entity_dictionary_count", -1)):
        raise PreparationError("stream-index entity dictionary count mismatch")
    if manifest.get("entity_dictionary_semantics") != ENTITY_DICTIONARY_SEMANTICS:
        raise PreparationError("stream-index entity dictionary payload boundary mismatch")
    return totals


def _blocked_receipt(binding: DatasetBinding, code: str, detail: str) -> dict[str, Any]:
    policy = binding.policy
    file_receipts = []
    for relative, _path, size, digest in binding.source_files:
        is_named_blocker = f"source_file={relative}" in detail
        file_receipts.append({
            "path": relative,
            "bytes": size,
            "sha256": digest,
            "status": (
                "blocked_at_this_file"
                if is_named_blocker
                else "not_materialized_due_dataset_blocker"
            ),
            "blocker_code": code,
        })
    return {
        "schema": RECEIPT_VERSION,
        "status": "blocked",
        "dataset_id": binding.row.dataset_id,
        "repo_id": binding.row.repo_id,
        "revision": binding.row.revision,
        "binding_sha256": binding.binding_sha256,
        "storage_layout": binding.storage_layout,
        "scope_id": binding.scope_id,
        "scope_sha256": binding.scope_sha256,
        "source_receipt_sha256": binding.source_receipt_sha256,
        "runtime_code_bindings": {
            "producer_sha256": binding.producer_sha256,
            "helper_timeseries_sha256": binding.helper_timeseries_sha256,
            "helper_tsfile_sha256": binding.helper_tsfile_sha256,
        },
        "policy_file_sha256": binding.policy_file_sha256,
        "policy_sha256": None if policy is None else policy.policy_sha256,
        "blocker": {"code": code, "detail": detail},
        "files": file_receipts,
        "output_claim": "no Unix epoch integer data materialized",
    }


def materialize_dataset(binding: DatasetBinding, output_root: Path) -> dict[str, Any]:
    policy = binding.policy
    if policy is None:
        return _blocked_receipt(
            binding,
            "missing_explicit_timestamp_policy",
            "No scope/revision-bound UTC timestamp policy exists for this dataset.",
        )
    if policy.decision not in {
        "admit_exact_utc",
        "admit_stored_epoch_coordinate",
        "admit_dataset_adapter",
    }:
        return _blocked_receipt(binding, "timestamp_policy_not_admitted", policy.reason)
    dataset_dir = output_root / binding.row.dataset_id
    if dataset_dir.exists():
        return _blocked_receipt(
            binding,
            "materialization_safety_blocker",
            f"output exists without a reusable verified receipt: {dataset_dir}",
        )
    working = Path(tempfile.mkdtemp(
        prefix=f".{binding.row.dataset_id}.entity-store-", dir=output_root
    ))
    entity_store = _SqliteEntityStore(working / "entities.sqlite3")
    try:
        plans = _schema_for_binding(binding)
        ordered_plans = sorted(plans, key=lambda item: (item.source_path, item.table_name))
        plan_ordinal_by_key = {
            (plan.source_path, plan.table_name): ordinal
            for ordinal, plan in enumerate(ordered_plans)
        }
        rows_by_source: dict[str, int] = {}
        filter_stats_by_source: dict[str, dict[str, Any]] = {}
        source_rows = 0
        for plan in plans:
            if plan.source_action == "exclude_entire_source_file":
                continue
            rows_by_source[plan.source_path] = 0
            filter_stats = {
                "input_rows": 0,
                "output_rows": 0,
                "dropped_rows": 0,
                "first_failed_condition_counts": {},
                "condition_failure_counts": {},
            }
            filter_stats_by_source[plan.source_path] = filter_stats
            for entity, _timestamp, _values in _iter_source_rows(
                binding, plan, filter_stats, pack_values=False
            ):
                source_rows += 1
                rows_by_source[plan.source_path] += 1
                entity_store.increment(
                    plan_ordinal_by_key[(plan.source_path, plan.table_name)], entity
                )
            entity_store.flush_counts()
        for rule in policy.source_rules:
            if rule.row_filter is None:
                continue
            matching_paths = [
                plan.source_path for plan in plans if plan.adapter_rule_id == rule.rule_id
            ]
            aggregate = {
                name: sum(int(filter_stats_by_source[path][name]) for path in matching_paths)
                for name in ("input_rows", "output_rows", "dropped_rows")
            }
            expected = {
                "input_rows": rule.row_filter.expected_input_rows,
                "output_rows": rule.row_filter.expected_output_rows,
                "dropped_rows": rule.row_filter.expected_drop_rows,
            }
            if aggregate != expected:
                raise PreparationError(
                    f"dataset={binding.row.dataset_id}, adapter_rule={rule.rule_id}: "
                    f"row-filter counts {aggregate} differ from policy {expected}"
                )
        if not source_rows:
            raise PreparationError("explicit table query returned no rows")
        candidate_stream_count = 0
        selected_stream_count = 0
        selected_bytes = 0
        prefix_closed = False
        first_unselected_logical_bytes: int | None = None
        selected_plan_count = 0
        for plan_ordinal, plan in enumerate(ordered_plans):
            if plan.source_action != "materialize":
                continue
            plan_entity_count = entity_store.entity_count(plan_ordinal)
            candidate_stream_count += plan_entity_count * len(plan.fields)
            if prefix_closed or not plan.fields:
                continue
            plan_base_stream_index = selected_stream_count
            plan_selected_stream_count = 0
            plan_selected_entity_count = 0
            full_entity_width = sum(8 + field.width for field in plan.fields)
            for entity_ordinal, (entity_key, row_count) in enumerate(
                entity_store.iter_entities(plan_ordinal)
            ):
                full_entity_bytes = row_count * full_entity_width
                if selected_bytes + full_entity_bytes <= binding.dataset_byte_cap:
                    entity_store.add_selected_entity(
                        plan_ordinal, entity_key, entity_ordinal, row_count,
                        len(plan.fields),
                    )
                    selected_bytes += full_entity_bytes
                    selected_stream_count += len(plan.fields)
                    plan_selected_stream_count += len(plan.fields)
                    plan_selected_entity_count += 1
                    continue
                selected_field_count = 0
                for field in plan.fields:
                    stream_bytes = row_count * (8 + field.width)
                    if selected_bytes + stream_bytes > binding.dataset_byte_cap:
                        first_unselected_logical_bytes = stream_bytes
                        prefix_closed = True
                        break
                    selected_bytes += stream_bytes
                    selected_stream_count += 1
                    plan_selected_stream_count += 1
                    selected_field_count += 1
                if selected_field_count:
                    entity_store.add_selected_entity(
                        plan_ordinal, entity_key, entity_ordinal, row_count,
                        selected_field_count,
                    )
                    plan_selected_entity_count += 1
                # The prefix cannot advance to another entity once a field does
                # not fit, even if a later narrower field or entity would fit.
                prefix_closed = True
                break
            if plan_selected_stream_count:
                entity_store.add_plan_selection(
                    plan_ordinal, plan_base_stream_index,
                    plan_selected_stream_count, plan_selected_entity_count,
                )
                selected_plan_count += 1
            if plan_selected_entity_count < plan_entity_count:
                prefix_closed = True
        if not selected_plan_count:
            raise PreparationError("no complete entity/FIELD stream fits the dataset byte cap")
        temporary = output_root / f".{binding.row.dataset_id}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True)
        shard_writer: _ShardWriter | None = None
        index_writer: _StreamIndexWriter | None = None
        try:
            # The first pass has already validated every frozen source file and
            # computed every candidate stream.  Once the whole-stream prefix is
            # fixed, source files that contribute no selected stream need not be
            # decoded a second time.  This is especially important for datasets
            # such as finance1k, where the 1 GiB cap can stop before hundreds of
            # later source files.  The optimization cannot change selection or
            # provenance because the complete first-pass counts and source-file
            # receipts remain in the manifest.
            streams: list[dict[str, Any]] = []
            totals = {
                "stream_count": 0,
                "chunk_count": 0,
                "rows": 0,
                "valid_values": 0,
                "missing_values": 0,
                "logical_bytes": 0,
            }
            shard_writer = (
                _ShardWriter(temporary, binding.chunk_byte_cap)
                if binding.storage_layout == "sharded_v1"
                else None
            )
            index_writer = (
                _StreamIndexWriter(temporary, binding)
                if binding.storage_layout == "sharded_v1"
                else None
            )
            selected_entity_count = 0
            peak_lookup_cache_entries = 0
            for (
                plan_ordinal, plan_base_stream_index, plan_stream_count,
                plan_selected_entity_count,
            ) in entity_store.iter_plan_selections():
                plan = ordered_plans[plan_ordinal]
                spool_dir = Path(tempfile.mkdtemp(prefix="spool-plan-", dir=temporary))
                try:
                    bucket_count = (
                        plan_stream_count + SPOOL_BUCKET_STREAMS - 1
                    ) // SPOOL_BUCKET_STREAMS
                    bucket_paths = {
                        bucket: spool_dir / f"bucket-{bucket:06d}.bin"
                        for bucket in range(bucket_count)
                    }
                    lookup_cache: OrderedDict[
                        bytes, tuple[int, int, int] | None
                    ] = OrderedDict()
                    handles = _SpoolHandles(bucket_paths)
                    try:
                        for entity, timestamp, values in _iter_source_rows(binding, plan):
                            entity_key = _canonical_json(list(entity))
                            if entity_key in lookup_cache:
                                selected_entity = lookup_cache.pop(entity_key)
                            else:
                                selected_entity = entity_store.selected_entity_by_key(
                                    plan_ordinal, entity_key
                                )
                            lookup_cache[entity_key] = selected_entity
                            if len(lookup_cache) > ENTITY_LOOKUP_CACHE_LIMIT:
                                lookup_cache.popitem(last=False)
                            peak_lookup_cache_entries = max(
                                peak_lookup_cache_entries, len(lookup_cache)
                            )
                            if selected_entity is None:
                                continue
                            entity_ordinal, _expected_rows, field_limit = selected_entity
                            for field_ordinal in range(field_limit):
                                packed = values[field_ordinal]
                                stream_index = (
                                    plan_base_stream_index
                                    + entity_ordinal * len(plan.fields)
                                    + field_ordinal
                                )
                                bucket = (
                                    stream_index - plan_base_stream_index
                                ) // SPOOL_BUCKET_STREAMS
                                payload = struct.pack("<QqB", stream_index, timestamp, packed is not None)
                                if packed is not None:
                                    payload += packed
                                handles.write(bucket, payload)
                    finally:
                        handles.close()

                    last_emitted_entity_ordinal = -1
                    for bucket, bucket_path in bucket_paths.items():
                        bucket_first = plan_base_stream_index + bucket * SPOOL_BUCKET_STREAMS
                        bucket_last = min(
                            plan_base_stream_index + plan_stream_count,
                            bucket_first + SPOOL_BUCKET_STREAMS,
                        )
                        first_entity_ordinal = (
                            bucket_first - plan_base_stream_index
                        ) // len(plan.fields)
                        last_entity_ordinal = (
                            bucket_last - 1 - plan_base_stream_index
                        ) // len(plan.fields)
                        bucket_entities = entity_store.selected_entities_by_ordinal(
                            plan_ordinal, first_entity_ordinal, last_entity_ordinal
                        )
                        observed: dict[int, int] = {}
                        offsets: dict[int, int] = {}
                        aggregate_bytes = 0
                        for stream_index in range(bucket_first, bucket_last):
                            relative_index = stream_index - plan_base_stream_index
                            entity_ordinal, field_ordinal = divmod(
                                relative_index, len(plan.fields)
                            )
                            if entity_ordinal not in bucket_entities:
                                raise PreparationError(
                                    "selected entity ordinal is missing from SQLite store"
                                )
                            _entity_key, expected_rows, field_count = bucket_entities[
                                entity_ordinal
                            ]
                            if field_ordinal >= field_count:
                                raise PreparationError(
                                    "selected stream exceeds its prefix field count"
                                )
                            field = plan.fields[field_ordinal]
                            offsets[stream_index] = aggregate_bytes
                            observed[stream_index] = 0
                            aggregate_bytes += expected_rows * (9 + field.width)
                        aggregate_path = spool_dir / f"aggregate-{bucket:06d}.bin"
                        with aggregate_path.open("w+b") as aggregate_file, bucket_path.open("rb") as handle:
                            aggregate_file.truncate(aggregate_bytes)
                            while True:
                                header = handle.read(17)
                                if not header:
                                    break
                                if len(header) != 17:
                                    raise PreparationError("truncated internal bucket spool header")
                                stream_index, _timestamp, present = struct.unpack("<QqB", header)
                                if stream_index not in offsets or present not in {0, 1}:
                                    raise PreparationError("corrupt internal bucket spool record")
                                relative_index = stream_index - plan_base_stream_index
                                entity_ordinal, field_ordinal = divmod(
                                    relative_index, len(plan.fields)
                                )
                                field = plan.fields[field_ordinal]
                                value = handle.read(field.width) if present else b""
                                if present and len(value) != field.width:
                                    raise PreparationError("truncated internal bucket spool value")
                                _entity_key, expected_rows, _field_count = bucket_entities[
                                    entity_ordinal
                                ]
                                if observed[stream_index] >= expected_rows:
                                    raise PreparationError(
                                        "source row count grew between audit and bucket passes"
                                    )
                                record_width = 9 + field.width
                                aggregate_file.seek(
                                    offsets[stream_index]
                                    + observed[stream_index] * record_width
                                )
                                aggregate_file.write(header[8:17])
                                aggregate_file.write(value if present else bytes(field.width))
                                observed[stream_index] += 1
                            aggregate_file.flush()
                            os.fsync(aggregate_file.fileno())
                        bucket_path.unlink()

                        for stream_index in range(bucket_first, bucket_last):
                            relative_index = stream_index - plan_base_stream_index
                            entity_ordinal, field_ordinal = divmod(
                                relative_index, len(plan.fields)
                            )
                            entity_key, expected_rows, field_count = bucket_entities[
                                entity_ordinal
                            ]
                            if field_ordinal >= field_count:
                                raise PreparationError(
                                    "selected stream exceeds its prefix field count"
                                )
                            entity_values = json.loads(entity_key)
                            if not isinstance(entity_values, list):
                                raise PreparationError("invalid SQLite entity encoding")
                            entity = tuple(str(value) for value in entity_values)
                            field = plan.fields[field_ordinal]
                            if observed[stream_index] != expected_rows:
                                raise PreparationError(
                                    "source row count changed between audit and bucket output passes"
                                )
                            if (
                                index_writer is not None
                                and entity_ordinal != last_emitted_entity_ordinal
                            ):
                                index_writer.write({
                                    "record_type": "entity",
                                    "source_path": plan.source_path,
                                    "table_name": plan.table_name,
                                    "entity_id": _entity_digest(entity),
                                    "entity_values": list(entity),
                                })
                                selected_entity_count += 1
                                last_emitted_entity_ordinal = entity_ordinal
                            stream_record = _emit_stream(
                                temporary,
                                stream_index=stream_index,
                                entity=entity,
                                field=field,
                                spool_path=aggregate_path,
                                expected_rows=observed[stream_index],
                                chunk_byte_cap=binding.chunk_byte_cap,
                                timestamp_unit=str(plan.timestamp_output_unit),
                                source_path=plan.source_path,
                                table_name=plan.table_name,
                                schema_sha256=plan.schema_sha256,
                                shard_writer=shard_writer,
                                include_entity_values=index_writer is None,
                                spool_offset=offsets[stream_index],
                                fixed_width_records=True,
                            )
                            totals["stream_count"] += 1
                            totals["chunk_count"] += len(stream_record["chunks"])
                            totals["rows"] += int(stream_record["row_count"])
                            totals["valid_values"] += int(stream_record["valid_count"])
                            totals["missing_values"] += int(stream_record["missing_count"])
                            totals["logical_bytes"] += int(stream_record["logical_bytes"])
                            if index_writer is None:
                                streams.append(stream_record)
                            else:
                                index_writer.write({
                                    "record_type": "stream",
                                    "stream": stream_record,
                                })
                        aggregate_path.unlink()
                    if (
                        index_writer is not None
                        and last_emitted_entity_ordinal + 1
                        != plan_selected_entity_count
                    ):
                        raise PreparationError("selected entity index emission is incomplete")
                finally:
                    shutil.rmtree(spool_dir, ignore_errors=True)

            if (
                totals["stream_count"] != selected_stream_count
                or totals["logical_bytes"] != selected_bytes
            ):
                raise PreparationError("compressed prefix selection/output totals diverged")
            all_candidates_selected = selected_stream_count == candidate_stream_count
            if (
                (all_candidates_selected and first_unselected_logical_bytes is not None)
                or (
                    not all_candidates_selected
                    and (
                        first_unselected_logical_bytes is None
                        or selected_bytes + first_unselected_logical_bytes
                        <= binding.dataset_byte_cap
                    )
                )
            ):
                raise PreparationError("longest-prefix witness is incomplete")
            stream_index_binding = None
            payload_shards: list[dict[str, Any]] = []
            if index_writer is not None:
                stream_index_binding = index_writer.close(totals, selected_entity_count)
                assert shard_writer is not None
                payload_shards = shard_writer.close()
            manifest: dict[str, Any] = {
                "schema": FORMAT_VERSION,
                "status": "materialized",
                "storage_layout": binding.storage_layout,
                "dataset_id": binding.row.dataset_id,
                "repo_id": binding.row.repo_id,
                "revision": binding.row.revision,
                "binding_sha256": binding.binding_sha256,
                "scope_id": binding.scope_id,
                "scope_sha256": binding.scope_sha256,
                "source_receipt_sha256": binding.source_receipt_sha256,
                "runtime_code_bindings": {
                    "producer_sha256": binding.producer_sha256,
                    "helper_timeseries_sha256": binding.helper_timeseries_sha256,
                    "helper_tsfile_sha256": binding.helper_tsfile_sha256,
                },
                "source_tsfiles": [
                    {"path": relative, "bytes": size, "sha256": digest}
                    for relative, _path, size, digest in binding.source_files
                ],
                "timestamp_policy": {
                    "policy_file_sha256": binding.policy_file_sha256,
                    "policy_sha256": policy.policy_sha256,
                    "admission_decision": policy.decision,
                    "mixed_axis_adapter_id": policy.mixed_axis_adapter_id,
                    "source_column": policy.source_column,
                    "source_unit": policy.source_unit,
                    "output_column": policy.output_column,
                    "output_unit": policy.output_unit,
                    "output_dtype": "int64",
                    "timezone": (
                        "UTC" if policy.decision == "admit_exact_utc" else None
                    ),
                    "timezone_basis": (
                        "explicit_utc_event_instant"
                        if policy.decision == "admit_exact_utc"
                        else "upstream_encoded_epoch"
                        if policy.decision == "admit_stored_epoch_coordinate"
                        else None
                    ),
                    "semantic_kind": (
                        "unix_epoch_instant"
                        if policy.decision == "admit_exact_utc"
                        else "stored_epoch_coordinate"
                        if policy.decision == "admit_stored_epoch_coordinate"
                        else None
                    ),
                    "event_utc_instant_claim": (
                        True
                        if policy.decision == "admit_exact_utc"
                        else False
                        if policy.decision == "admit_stored_epoch_coordinate"
                        else None
                    ),
                    "precision_inferred": False,
                    "timezone_inferred": False,
                    "row_index_substitution": False,
                    "evidence": list(policy.evidence),
                    "temporal_string_fields": [
                        {
                            "source_name": field.source_name,
                            "source_type": field.source_type,
                            "source_category": field.source_category,
                            "output_name": field.output_name,
                            "parse_kind": field.parse_kind,
                            "format": field.datetime_format,
                            "accepted_formats": list(field.accepted_datetime_formats),
                            "timezone": field.timezone_name,
                            "source_unit": field.source_unit,
                            "ambiguous_fold": field.ambiguous_fold,
                            "output_unit": "us",
                            "output_dtype": "int64",
                            "event_utc_instant_claim": (
                                field.event_utc_instant_claim
                            ),
                            "semantic_kind": field.semantic_kind,
                            "timezone_basis": field.timezone_basis,
                            "format_class": field.format_class,
                            "applicability": field.applicability,
                            "candidate_definition_count": (
                                field.candidate_definition_count
                            ),
                            "evidence": list(field.evidence),
                        }
                        for field in policy.temporal_string_fields
                    ],
                    "source_bound_temporal_fields": [
                        {
                            "source_path": field.source_path,
                            "source_sha256": field.source_sha256,
                            "table_name": field.table_name,
                            "source_schema_sha256": field.source_schema_sha256,
                            "source_name": field.source_name,
                            "source_type": field.source_type,
                            "source_category": field.source_category,
                            "source_unit": field.source_unit,
                            "output_name": field.output_name,
                            "output_unit": field.output_unit,
                            "output_dtype": "int64",
                            "parse_kind": field.parse_kind,
                            "timezone": field.timezone_name,
                            "event_utc_instant_claim": (
                                field.event_utc_instant_claim
                            ),
                            "semantic_kind": field.semantic_kind,
                            "timezone_basis": field.timezone_basis,
                            "evidence": list(field.evidence),
                        }
                        for field in policy.source_bound_temporal_fields
                    ],
                    "dropped_timestamp_like_string_fields": [
                        {
                            "source_name": field.source_name,
                            "source_type": field.source_type,
                            "source_category": field.source_category,
                            "decision": field.decision,
                            "format_class": field.format_class,
                            "reason": field.reason,
                            "applicability": field.applicability,
                            "candidate_definition_count": (
                                field.candidate_definition_count
                            ),
                            "evidence": list(field.evidence),
                        }
                        for field in policy.dropped_timestamp_like_string_fields
                    ],
                    "source_rules": [
                        {
                            "rule_id": rule.rule_id,
                            "table_names": list(rule.table_names),
                            "action": rule.action,
                            "expected_source_file_count": (
                                rule.expected_source_file_count
                            ),
                            "reason": rule.reason,
                            "evidence": list(rule.evidence),
                            "timestamp": (
                                None
                                if rule.action != "materialize"
                                else {
                                    "source_column": rule.source_column,
                                    "source_unit": rule.source_unit,
                                    "output_column": rule.output_column,
                                    "output_unit": rule.output_unit,
                                    "output_dtype": "int64",
                                    "event_utc_instant_claim": (
                                        rule.event_utc_instant_claim
                                    ),
                                    "utc_instant_verified": (
                                        rule.utc_instant_verified
                                    ),
                                    "semantic_kind": rule.semantic_kind,
                                    "timezone_basis": rule.timezone_basis,
                                    "timezone_inferred": rule.timezone_inferred,
                                }
                            ),
                            "row_filter": (
                                None
                                if rule.row_filter is None
                                else {
                                    "kind": "all",
                                    "conditions": [
                                        {
                                            "source_name": condition.source_name,
                                            "source_type": condition.source_type,
                                            "source_category": condition.source_category,
                                            "operator": condition.operator,
                                            "value": condition.value,
                                            "drop_reason": condition.drop_reason,
                                        }
                                        for condition in rule.row_filter.conditions
                                    ],
                                    "expected_input_rows": (
                                        rule.row_filter.expected_input_rows
                                    ),
                                    "expected_output_rows": (
                                        rule.row_filter.expected_output_rows
                                    ),
                                    "expected_drop_rows": (
                                        rule.row_filter.expected_drop_rows
                                    ),
                                    "evidence": list(rule.row_filter.evidence),
                                }
                            ),
                        }
                        for rule in policy.source_rules
                    ],
                },
                "table": {
                    "table_name_policy": policy.table_name,
                    "source_file_tables": [
                        {
                            "source_path": plan.source_path,
                            "table_name": plan.table_name,
                            "schema_sha256": plan.schema_sha256,
                            "source_action": plan.source_action,
                            "source_action_reason": plan.source_action_reason,
                            "adapter_rule_id": plan.adapter_rule_id,
                            "partition_tags": list(plan.tag_names),
                            "timestamp": (
                                None
                                if plan.source_action != "materialize"
                                else {
                                    "source_column": plan.timestamp_source_column,
                                    "source_unit": plan.timestamp_source_unit,
                                    "output_column": plan.timestamp_output_column,
                                    "output_unit": plan.timestamp_output_unit,
                                    "output_dtype": "int64",
                                    "event_utc_instant_claim": (
                                        plan.timestamp_event_utc_instant_claim
                                    ),
                                    "semantic_kind": plan.timestamp_semantic_kind,
                                    "timezone_basis": plan.timestamp_timezone_basis,
                                    "timezone_inferred": plan.timestamp_timezone_inferred,
                                }
                            ),
                            "retained_fields": [
                                {
                                    "source_name": field.name,
                                    "output_name": field.emitted_name,
                                    "tsfile_type": field.tsfile_type,
                                    "dtype": field.dtype,
                                    "conversion": _field_conversion_label(field),
                                }
                                for field in plan.fields
                            ],
                            "dropped_columns": list(plan.dropped_columns),
                            "source_rows": rows_by_source.get(plan.source_path),
                            "row_filter_observed": filter_stats_by_source.get(
                                plan.source_path
                            ),
                        }
                        for plan in plans
                    ],
                    "partition_policy": (
                        "source_path plus actual table name plus exact typed TAG tuple; "
                        "TAG payload is not emitted except an explicitly admitted temporal "
                        "TAG is also parsed into its int64-us value stream"
                    ),
                },
                "selection": {
                    "dataset_byte_cap": binding.dataset_byte_cap,
                    "cap_label": "1 GiB" if binding.dataset_byte_cap == DEFAULT_DATASET_BYTE_CAP else "custom",
                    "policy": (
                        "longest deterministic whole (source path, table, typed TAG entity, "
                        "retained schema column) stream prefix"
                    ),
                    "ordering": (
                        "source path, actual table name, canonical typed TAG tuple, then the "
                        "frozen source schema column order; temporal replacements retain their "
                        "source column position"
                    ),
                    "candidate_stream_count": candidate_stream_count,
                    "selected_stream_count": selected_stream_count,
                    "selected_logical_bytes": selected_bytes,
                    "unselected_stream_count": candidate_stream_count - selected_stream_count,
                    "chunk_byte_cap": binding.chunk_byte_cap,
                    "metadata_compressed_byte_cap": (
                        binding.metadata_compressed_byte_cap
                    ),
                    "longest_prefix_witness": {
                        "candidate_stream_count": candidate_stream_count,
                        "selected_stream_count": selected_stream_count,
                        "selected_logical_bytes": selected_bytes,
                        "all_candidates_selected": all_candidates_selected,
                        "first_unselected_logical_bytes": (
                            first_unselected_logical_bytes
                        ),
                        "proof_boundary": (
                            "all_candidates_selected"
                            if all_candidates_selected
                            else "selected_bytes_plus_first_unselected_exceeds_cap"
                        ),
                    },
                },
                "source_rows": source_rows,
                "entity_count": entity_store.total_entity_count(),
                "bounded_working_set": {
                    "entity_counts": "sqlite_backed",
                    "entity_count_cache_limit": ENTITY_COUNT_CACHE_LIMIT,
                    "peak_entity_count_cache_entries": (
                        entity_store.peak_count_cache_entries
                    ),
                    "entity_lookup_cache_limit": ENTITY_LOOKUP_CACHE_LIMIT,
                    "peak_entity_lookup_cache_entries": peak_lookup_cache_entries,
                    "spool_bucket_stream_limit": SPOOL_BUCKET_STREAMS,
                },
                "totals": totals,
            }
            if binding.storage_layout == "legacy_v1":
                manifest["streams"] = streams
            else:
                assert stream_index_binding is not None
                manifest["stream_index"] = stream_index_binding
                manifest["payload_shards"] = payload_shards
                manifest["selected_entity_dictionary_count"] = selected_entity_count
                manifest["entity_dictionary_semantics"] = ENTITY_DICTIONARY_SEMANTICS
            manifest["provenance_sha256"] = _sha256_bytes(_canonical_json(manifest))
            _atomic_json(temporary / "dataset_manifest.json", manifest)
            os.replace(temporary, dataset_dir)
            return {
                "schema": RECEIPT_VERSION,
                "status": "materialized",
                "dataset_id": binding.row.dataset_id,
                "repo_id": binding.row.repo_id,
                "revision": binding.row.revision,
                "binding_sha256": binding.binding_sha256,
                "scope_id": binding.scope_id,
                "scope_sha256": binding.scope_sha256,
                "source_receipt_sha256": binding.source_receipt_sha256,
                "runtime_code_bindings": {
                    "producer_sha256": binding.producer_sha256,
                    "helper_timeseries_sha256": binding.helper_timeseries_sha256,
                    "helper_tsfile_sha256": binding.helper_tsfile_sha256,
                },
                "policy_file_sha256": binding.policy_file_sha256,
                "policy_sha256": policy.policy_sha256,
                "storage_layout": binding.storage_layout,
                "output_directory": binding.row.dataset_id,
                "manifest_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
                "selected_stream_count": totals["stream_count"],
                "selected_logical_bytes": selected_bytes,
                "candidate_stream_count": candidate_stream_count,
                "longest_prefix_witness": manifest["selection"][
                    "longest_prefix_witness"
                ],
                "files": [
                    {
                        "path": relative,
                        "bytes": size,
                        "sha256": digest,
                        "status": (
                            "excluded_entire_source_file_by_dataset_adapter"
                            if next(
                                plan for plan in plans
                                if plan.source_path == relative
                            ).source_action == "exclude_entire_source_file"
                            else "verified_and_materialized"
                        ),
                    }
                    for relative, _path, size, digest in binding.source_files
                ],
            }
        except BaseException:
            if index_writer is not None:
                index_writer.abort()
            if shard_writer is not None:
                shard_writer.abort()
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    except PreparationError as error:
        return _blocked_receipt(binding, "materialization_safety_blocker", str(error))
    finally:
        entity_store.close()
        shutil.rmtree(working, ignore_errors=True)


def verify_dataset_output(binding: DatasetBinding, output_root: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != RECEIPT_VERSION:
        raise PreparationError(f"unsupported materialization receipt for {binding.row.dataset_id}")
    if (
        receipt.get("dataset_id") != binding.row.dataset_id
        or receipt.get("binding_sha256") != binding.binding_sha256
        or receipt.get("storage_layout") != binding.storage_layout
        or (
            receipt.get("status") == "materialized"
            and receipt.get("output_directory") != binding.row.dataset_id
        )
    ):
        raise PreparationError(f"materialization receipt binding mismatch for {binding.row.dataset_id}")
    expected_runtime = {
        "producer_sha256": binding.producer_sha256,
        "helper_timeseries_sha256": binding.helper_timeseries_sha256,
        "helper_tsfile_sha256": binding.helper_tsfile_sha256,
    }
    if receipt.get("runtime_code_bindings") != expected_runtime:
        raise PreparationError(f"runtime code binding mismatch for {binding.row.dataset_id}")
    status = receipt.get("status")
    if status == "blocked":
        if (
            binding.policy is not None
            and binding.policy.decision
            in {
                "admit_exact_utc",
                "admit_stored_epoch_coordinate",
                "admit_dataset_adapter",
            }
        ):
            # An admitted dataset may remain blocked on schema/data safety.  Preserve
            # that evidence, but do not pretend it was materialized.
            if receipt.get("blocker", {}).get("code") != "materialization_safety_blocker":
                raise PreparationError(f"stale blocked receipt for {binding.row.dataset_id}")
        return dict(receipt) | {"resume_status": "verified_existing"}
    if status != "materialized":
        raise PreparationError(f"invalid materialization status for {binding.row.dataset_id}")
    dataset_dir = output_root / binding.row.dataset_id
    try:
        dataset_mode = os.lstat(dataset_dir).st_mode
    except OSError as error:
        raise PreparationError(f"materialized dataset directory is missing: {binding.row.dataset_id}") from error
    if not stat.S_ISDIR(dataset_mode) or dataset_dir.resolve().parent != output_root.resolve():
        raise PreparationError(f"unsafe materialized dataset directory: {binding.row.dataset_id}")
    manifest_path = _regular_file_within(dataset_dir, "dataset_manifest.json")
    if sha256_file(manifest_path) != receipt.get("manifest_sha256"):
        raise PreparationError(f"dataset manifest hash mismatch for {binding.row.dataset_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != FORMAT_VERSION
        or manifest.get("binding_sha256") != binding.binding_sha256
        or manifest.get("storage_layout") != binding.storage_layout
        or manifest.get("runtime_code_bindings") != expected_runtime
    ):
        raise PreparationError(f"dataset manifest binding mismatch for {binding.row.dataset_id}")
    provenance = manifest.pop("provenance_sha256", None)
    if provenance != _sha256_bytes(_canonical_json(manifest)):
        raise PreparationError(f"dataset provenance hash mismatch for {binding.row.dataset_id}")
    expected_sources = [
        {"path": relative, "bytes": size, "sha256": digest}
        for relative, _path, size, digest in binding.source_files
    ]
    if manifest.get("source_tsfiles") != expected_sources:
        raise PreparationError(f"dataset source-file binding mismatch for {binding.row.dataset_id}")
    seen_paths: set[str] = {"dataset_manifest.json"}
    if binding.storage_layout == "sharded_v1":
        if "streams" in manifest:
            raise PreparationError("sharded manifest must not embed stream metadata")
        totals = _verify_sharded_index(dataset_dir, manifest, binding, seen_paths)
    else:
        if "stream_index" in manifest or "payload_shards" in manifest:
            raise PreparationError("legacy manifest contains sharded-layout metadata")
        totals = {"stream_count": 0, "chunk_count": 0, "rows": 0, "valid_values": 0, "missing_values": 0, "logical_bytes": 0}
        streams = manifest.get("streams")
        if not isinstance(streams, list) or not streams:
            raise PreparationError("legacy manifest contains no streams")
        for stream in streams:
            totals["stream_count"] += 1
            totals["rows"] += int(stream["row_count"])
            totals["valid_values"] += int(stream["valid_count"])
            totals["missing_values"] += int(stream["missing_count"])
            totals["logical_bytes"] += int(stream["logical_bytes"])
            stream_rows = stream_valid = 0
            for chunk in stream.get("chunks", []):
                totals["chunk_count"] += 1
                stream_rows += int(chunk["row_count"])
                stream_valid += int(chunk["valid_count"])
                if int(chunk["missing_count"]) != int(chunk["row_count"]) - int(chunk["valid_count"]):
                    raise PreparationError("chunk missing-count mismatch")
                for name in ("timestamps", "values", "missing_bitmap"):
                    relative = _safe_materialized_path(chunk[f"{name}_path"])
                    if relative in seen_paths:
                        raise PreparationError("unsafe or duplicate materialized file path")
                    seen_paths.add(relative)
                    path = _regular_file_within(dataset_dir, relative)
                    if not path.is_file() or path.stat().st_size != int(chunk[f"{name}_bytes"]):
                        raise PreparationError(f"materialized file size mismatch: {relative}")
                    if sha256_file(path) != chunk[f"{name}_sha256"]:
                        raise PreparationError(f"materialized file hash mismatch: {relative}")
                timestamp_bytes = int(chunk["timestamps_bytes"])
                value_bytes = int(chunk["values_bytes"])
                bitmap_bytes = int(chunk["missing_bitmap_bytes"])
                width = int(stream["width_bytes"])
                if timestamp_bytes != int(chunk["row_count"]) * 8:
                    raise PreparationError("timestamp byte count mismatch")
                if value_bytes != int(chunk["valid_count"]) * width:
                    raise PreparationError("value byte count mismatch")
                if bitmap_bytes != (int(chunk["row_count"]) + 7) // 8:
                    raise PreparationError("missing bitmap byte count mismatch")
                bitmap = (dataset_dir / chunk["missing_bitmap_path"]).read_bytes()
                if sum(byte.bit_count() for byte in bitmap) != int(chunk["missing_count"]):
                    raise PreparationError("missing bitmap population mismatch")
            if stream_rows != int(stream["row_count"]) or stream_valid != int(stream["valid_count"]):
                raise PreparationError("stream/chunk aggregate mismatch")
    if totals != manifest.get("totals"):
        raise PreparationError(f"dataset totals mismatch for {binding.row.dataset_id}")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise PreparationError("dataset selection receipt is missing")
    candidate_count = int(selection.get("candidate_stream_count", -1))
    selected_count = int(selection.get("selected_stream_count", -1))
    if (
        int(selection.get("dataset_byte_cap", -1)) != binding.dataset_byte_cap
        or int(selection.get("chunk_byte_cap", -1)) != binding.chunk_byte_cap
        or int(selection.get("metadata_compressed_byte_cap", -1))
        != binding.metadata_compressed_byte_cap
        or selected_count != totals["stream_count"]
        or int(selection.get("selected_logical_bytes", -1)) != totals["logical_bytes"]
        or candidate_count < selected_count
        or int(selection.get("unselected_stream_count", -1)) != candidate_count - selected_count
        or receipt.get("selected_stream_count") != totals["stream_count"]
        or receipt.get("selected_logical_bytes") != totals["logical_bytes"]
        or receipt.get("candidate_stream_count") != candidate_count
    ):
        raise PreparationError(f"dataset selection/totals mismatch for {binding.row.dataset_id}")
    witness = selection.get("longest_prefix_witness")
    if not isinstance(witness, Mapping) or receipt.get("longest_prefix_witness") != witness:
        raise PreparationError("longest-prefix witness receipt mismatch")
    all_selected = selected_count == candidate_count
    first_unselected = witness.get("first_unselected_logical_bytes")
    if (
        int(witness.get("candidate_stream_count", -1)) != candidate_count
        or int(witness.get("selected_stream_count", -1)) != selected_count
        or int(witness.get("selected_logical_bytes", -1)) != totals["logical_bytes"]
        or witness.get("all_candidates_selected") is not all_selected
        or (
            all_selected
            and (
                first_unselected is not None
                or witness.get("proof_boundary") != "all_candidates_selected"
            )
        )
        or (
            not all_selected
            and (
                not isinstance(first_unselected, int)
                or isinstance(first_unselected, bool)
                or first_unselected <= 0
                or totals["logical_bytes"] + first_unselected
                <= binding.dataset_byte_cap
                or witness.get("proof_boundary")
                != "selected_bytes_plus_first_unselected_exceeds_cap"
            )
        )
    ):
        raise PreparationError("invalid longest-prefix witness")
    actual_paths: set[str] = set()
    for path in dataset_dir.rglob("*"):
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise PreparationError("materialized dataset contains a symlink")
        if stat.S_ISREG(mode):
            actual_paths.add(path.relative_to(dataset_dir).as_posix())
        elif not stat.S_ISDIR(mode):
            raise PreparationError("materialized dataset contains a non-regular path")
    if actual_paths != seen_paths:
        raise PreparationError(f"unexpected or missing materialized files for {binding.row.dataset_id}")
    return dict(receipt) | {"resume_status": "verified_existing"}


class _CountingSink:
    """Minimal file-like sink used to size deterministic gzip output in memory."""

    def __init__(self) -> None:
        self.bytes_written = 0

    def write(self, payload: bytes) -> int:
        self.bytes_written += len(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def tell(self) -> int:
        return self.bytes_written


def _selected_stream_specs(
    entity_store: _SqliteEntityStore,
    ordered_plans: Sequence[SourcePlan],
) -> Iterable[tuple[int, SourcePlan, tuple[str, ...], FieldSpec, int]]:
    """Yield the frozen selected prefix without constructing payload records."""
    for plan_ordinal, base_stream_index, _stream_count, _entity_count in (
        entity_store.iter_plan_selections()
    ):
        plan = ordered_plans[plan_ordinal]
        for entity_key, entity_ordinal, row_count, field_count in (
            entity_store.iter_selected_entities(plan_ordinal)
        ):
            decoded = json.loads(entity_key)
            if not isinstance(decoded, list):
                raise PreparationError("invalid SQLite entity encoding during planning")
            entity = tuple(str(value) for value in decoded)
            for field_ordinal in range(field_count):
                yield (
                    base_stream_index
                    + entity_ordinal * len(plan.fields)
                    + field_ordinal,
                    plan,
                    entity,
                    plan.fields[field_ordinal],
                    row_count,
                )


def _planning_chunk_rows(row_count: int, field: FieldSpec, chunk_cap: int) -> Iterable[int]:
    max_rows = max(1, min(chunk_cap // 8, chunk_cap // field.width, chunk_cap * 8))
    remaining = row_count
    while remaining:
        rows = min(remaining, max_rows)
        yield rows
        remaining -= rows


def _planning_stream_record(
    *,
    stream_index: int,
    plan: SourcePlan,
    entity: tuple[str, ...],
    field: FieldSpec,
    row_count: int,
    chunk_cap: int,
    largest_shard_index: int,
) -> dict[str, Any]:
    """Build a size-conservative sharded index record, not a payload record."""
    shard_digits = max(6, len(str(max(0, largest_shard_index))))
    shard_path = f"payload/shard-{'9' * shard_digits}.bin"
    chunks: list[dict[str, Any]] = []
    for chunk_index, rows in enumerate(_planning_chunk_rows(row_count, field, chunk_cap)):
        lengths = {
            "timestamps": rows * 8,
            "values": rows * field.width,
            "missing_bitmap": (rows + 7) // 8,
        }
        chunk: dict[str, Any] = {
            "chunk_index": chunk_index,
            "row_count": rows,
            # Setting both counts to the maximum is intentionally internally
            # inconsistent: it makes their decimal encodings no shorter than
            # either possible observed count.
            "valid_count": rows,
            "missing_count": rows,
            f"first_timestamp_unix_{plan.timestamp_output_unit}": INT64_MIN,
            f"last_timestamp_unix_{plan.timestamp_output_unit}": INT64_MIN,
            "timestamp_unit": plan.timestamp_output_unit,
            "timestamp_dtype": "int64",
            "value_dtype": field.dtype,
            "byte_order": "little",
            "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
        }
        for name, length in lengths.items():
            chunk[f"{name}_segment"] = {
                "path": shard_path,
                "offset": chunk_cap,
                "length": length,
                "sha256": "f" * 64,
            }
            chunk[f"{name}_bytes"] = length
            chunk[f"{name}_sha256"] = "f" * 64
        chunks.append(chunk)
    stream = {
        "stream_index": stream_index,
        "stream_id": f"s-{stream_index:08d}",
        "entity_id": _entity_digest(entity),
        "source_path": plan.source_path,
        "table_name": plan.table_name,
        "schema_sha256": plan.schema_sha256,
        "field": field.emitted_name,
        "source_field": field.name,
        "tsfile_type": field.tsfile_type,
        "dtype": field.dtype,
        "width_bytes": field.width,
        "conversion": _field_conversion_label(field),
        "row_count": row_count,
        "valid_count": row_count,
        "missing_count": row_count,
        "logical_bytes": row_count * (8 + field.width),
        "chunks": chunks,
    }
    return {"record_type": "stream", "stream": stream}


def _atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file while refusing every overwrite."""
    payload = _canonical_json(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PreparationError(f"planning receipt already exists: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_planning_receipt_path(
    receipt_path: Path, *, output: Path, mirror: Path
) -> Path:
    requested = receipt_path.absolute()
    parent = requested.parent
    try:
        parent_mode = os.lstat(parent).st_mode
    except OSError as error:
        raise PreparationError("planning receipt parent must already exist") from error
    if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
        raise PreparationError("planning receipt parent must be a real directory")
    resolved_parent = parent.resolve(strict=True)
    target = resolved_parent / requested.name
    if target.exists() or target.is_symlink():
        raise PreparationError(f"planning receipt already exists: {target}")

    def resolved_candidate(path: Path) -> Path:
        absolute = path.absolute()
        unresolved: list[str] = []
        probe = absolute
        while not probe.exists() and not probe.is_symlink():
            if probe.parent == probe:
                raise PreparationError(f"cannot resolve planning boundary: {absolute}")
            unresolved.append(probe.name)
            probe = probe.parent
        resolved = probe.resolve(strict=True)
        for name in reversed(unresolved):
            resolved = resolved / name
        return resolved

    output_candidate = resolved_candidate(output)
    mirror_root = mirror.resolve(strict=True)
    if target == output_candidate or target.is_relative_to(output_candidate):
        raise PreparationError("planning receipt must be outside the materialization output tree")
    if target == mirror_root or target.is_relative_to(mirror_root):
        raise PreparationError("planning receipt must be outside the source mirror tree")
    return target


def _plan_dataset(binding: DatasetBinding, working_root: Path) -> dict[str, Any]:
    policy = binding.policy
    if policy is None:
        raise PreparationError(
            f"dataset={binding.row.dataset_id}: missing explicit timestamp policy"
        )
    if policy.decision not in {
        "admit_exact_utc",
        "admit_stored_epoch_coordinate",
        "admit_dataset_adapter",
    }:
        raise PreparationError(
            f"dataset={binding.row.dataset_id}: timestamp policy is not admitted"
        )
    dataset_working = working_root / binding.row.dataset_id
    dataset_working.mkdir(exist_ok=False)
    entity_store = _SqliteEntityStore(dataset_working / "entities.sqlite3")
    try:
        plans = _schema_for_binding(binding)
        ordered_plans = sorted(plans, key=lambda item: (item.source_path, item.table_name))
        plan_ordinal_by_key = {
            (plan.source_path, plan.table_name): ordinal
            for ordinal, plan in enumerate(ordered_plans)
        }
        rows_by_source: dict[str, int] = {}
        filter_stats_by_source: dict[str, dict[str, Any]] = {}
        source_rows = 0
        for plan in plans:
            if plan.source_action == "exclude_entire_source_file":
                continue
            rows_by_source[plan.source_path] = 0
            filter_stats = {
                "input_rows": 0,
                "output_rows": 0,
                "dropped_rows": 0,
                "first_failed_condition_counts": {},
                "condition_failure_counts": {},
            }
            filter_stats_by_source[plan.source_path] = filter_stats
            for entity, _timestamp, values in _iter_source_rows(
                binding, plan, filter_stats, pack_values=False
            ):
                if values:
                    raise PreparationError("count-only planning unexpectedly packed values")
                source_rows += 1
                rows_by_source[plan.source_path] += 1
                entity_store.increment(
                    plan_ordinal_by_key[(plan.source_path, plan.table_name)], entity
                )
            entity_store.flush_counts()
        for rule in policy.source_rules:
            if rule.row_filter is None:
                continue
            matching_paths = [
                plan.source_path for plan in plans if plan.adapter_rule_id == rule.rule_id
            ]
            observed = {
                name: sum(
                    int(filter_stats_by_source[path][name]) for path in matching_paths
                )
                for name in ("input_rows", "output_rows", "dropped_rows")
            }
            expected = {
                "input_rows": rule.row_filter.expected_input_rows,
                "output_rows": rule.row_filter.expected_output_rows,
                "dropped_rows": rule.row_filter.expected_drop_rows,
            }
            if observed != expected:
                raise PreparationError(
                    f"dataset={binding.row.dataset_id}, adapter_rule={rule.rule_id}: "
                    f"row-filter counts {observed} differ from policy {expected}"
                )
        if not source_rows:
            raise PreparationError(
                f"dataset={binding.row.dataset_id}: explicit table query returned no rows"
            )

        candidate_stream_count = 0
        selected_stream_count = 0
        selected_logical_bytes = 0
        first_unselected_logical_bytes: int | None = None
        prefix_closed = False
        selected_plan_count = 0
        for plan_ordinal, plan in enumerate(ordered_plans):
            if plan.source_action != "materialize":
                continue
            plan_entity_count = entity_store.entity_count(plan_ordinal)
            candidate_stream_count += plan_entity_count * len(plan.fields)
            if prefix_closed or not plan.fields:
                continue
            base_stream_index = selected_stream_count
            plan_stream_count = 0
            plan_selected_entities = 0
            full_entity_width = sum(8 + field.width for field in plan.fields)
            for entity_ordinal, (entity_key, row_count) in enumerate(
                entity_store.iter_entities(plan_ordinal)
            ):
                full_entity_bytes = row_count * full_entity_width
                if selected_logical_bytes + full_entity_bytes <= binding.dataset_byte_cap:
                    entity_store.add_selected_entity(
                        plan_ordinal, entity_key, entity_ordinal, row_count, len(plan.fields)
                    )
                    selected_logical_bytes += full_entity_bytes
                    selected_stream_count += len(plan.fields)
                    plan_stream_count += len(plan.fields)
                    plan_selected_entities += 1
                    continue
                selected_fields = 0
                for field in plan.fields:
                    stream_bytes = row_count * (8 + field.width)
                    if selected_logical_bytes + stream_bytes > binding.dataset_byte_cap:
                        first_unselected_logical_bytes = stream_bytes
                        prefix_closed = True
                        break
                    selected_logical_bytes += stream_bytes
                    selected_stream_count += 1
                    plan_stream_count += 1
                    selected_fields += 1
                if selected_fields:
                    entity_store.add_selected_entity(
                        plan_ordinal, entity_key, entity_ordinal, row_count, selected_fields
                    )
                    plan_selected_entities += 1
                prefix_closed = True
                break
            if plan_stream_count:
                entity_store.add_plan_selection(
                    plan_ordinal, base_stream_index, plan_stream_count,
                    plan_selected_entities,
                )
                selected_plan_count += 1
            if plan_selected_entities < plan_entity_count:
                prefix_closed = True
        if not selected_plan_count:
            raise PreparationError(
                f"dataset={binding.row.dataset_id}: no complete stream fits the byte cap"
            )
        all_selected = selected_stream_count == candidate_stream_count
        if (
            (all_selected and first_unselected_logical_bytes is not None)
            or (
                not all_selected
                and (
                    first_unselected_logical_bytes is None
                    or selected_logical_bytes + first_unselected_logical_bytes
                    <= binding.dataset_byte_cap
                )
            )
        ):
            raise PreparationError(
                f"dataset={binding.row.dataset_id}: incomplete longest-prefix witness"
            )

        expected_shards = 0
        shard_bytes = 0
        payload_bytes_upper = 0
        selected_rows = 0
        selected_chunk_count = 0
        selected_entity_count = 0
        spool_bytes_by_plan: dict[str, int] = {}
        aggregate_bytes_by_plan_bucket: dict[tuple[str, int], int] = {}
        last_entity_by_plan: dict[str, str] = {}
        stream_ordinal_by_plan: dict[str, int] = {}
        for stream_index, plan, entity, field, row_count in _selected_stream_specs(
            entity_store, ordered_plans
        ):
            plan_key = f"{plan.source_path}\0{plan.table_name}"
            entity_key = _canonical_json(list(entity)).decode("utf-8")
            if last_entity_by_plan.get(plan_key) != entity_key:
                selected_entity_count += 1
                last_entity_by_plan[plan_key] = entity_key
            selected_rows += row_count
            spool_bytes_by_plan[plan_key] = (
                spool_bytes_by_plan.get(plan_key, 0) + row_count * (17 + field.width)
            )
            plan_stream_ordinal = stream_ordinal_by_plan.get(plan_key, 0)
            stream_ordinal_by_plan[plan_key] = plan_stream_ordinal + 1
            bucket = plan_stream_ordinal // SPOOL_BUCKET_STREAMS
            aggregate_key = (plan_key, bucket)
            aggregate_bytes_by_plan_bucket[aggregate_key] = (
                aggregate_bytes_by_plan_bucket.get(aggregate_key, 0)
                + row_count * (9 + field.width)
            )
            for rows in _planning_chunk_rows(row_count, field, binding.chunk_byte_cap):
                selected_chunk_count += 1
                for segment_bytes in (
                    rows * 8,
                    rows * field.width,
                    (rows + 7) // 8,
                ):
                    payload_bytes_upper += segment_bytes
                    if expected_shards == 0:
                        expected_shards = 1
                    elif segment_bytes and shard_bytes + segment_bytes > binding.chunk_byte_cap:
                        expected_shards += 1
                        shard_bytes = 0
                    shard_bytes += segment_bytes

        header = {
            "record_type": "header",
            "schema": STREAM_INDEX_VERSION,
            "storage_layout": "sharded_v1",
            "binding_sha256": binding.binding_sha256,
            "dataset_id": binding.row.dataset_id,
            "producer_sha256": binding.producer_sha256,
            "helper_timeseries_sha256": binding.helper_timeseries_sha256,
            "helper_tsfile_sha256": binding.helper_tsfile_sha256,
        }
        sink = _CountingSink()
        compressor = gzip.GzipFile(
            filename="stream_index.jsonl", mode="wb", fileobj=sink, mtime=0
        )
        index_uncompressed_upper = 0

        def add_index_record(record: Mapping[str, Any]) -> None:
            nonlocal index_uncompressed_upper
            payload = _canonical_json(dict(record)) + b"\n"
            index_uncompressed_upper += len(payload)
            compressor.write(payload)

        add_index_record(header)
        last_entity_by_plan.clear()
        for stream_index, plan, entity, field, row_count in _selected_stream_specs(
            entity_store, ordered_plans
        ):
            plan_key = f"{plan.source_path}\0{plan.table_name}"
            entity_key = _canonical_json(list(entity)).decode("utf-8")
            if last_entity_by_plan.get(plan_key) != entity_key:
                add_index_record({
                    "record_type": "entity",
                    "source_path": plan.source_path,
                    "table_name": plan.table_name,
                    "entity_id": _entity_digest(entity),
                    "entity_values": list(entity),
                })
                last_entity_by_plan[plan_key] = entity_key
            add_index_record(_planning_stream_record(
                stream_index=stream_index,
                plan=plan,
                entity=entity,
                field=field,
                row_count=row_count,
                chunk_cap=binding.chunk_byte_cap,
                largest_shard_index=max(0, expected_shards - 1),
            ))
        upper_totals = {
            "stream_count": selected_stream_count,
            "chunk_count": selected_chunk_count,
            "rows": selected_rows,
            "valid_values": selected_rows,
            "missing_values": selected_rows,
            "logical_bytes": selected_logical_bytes,
        }
        add_index_record({
            "record_type": "footer",
            "schema": STREAM_INDEX_VERSION,
            "entity_record_count": selected_entity_count,
            "stream_record_count": selected_stream_count,
            "totals": upper_totals,
        })
        compressor.close()
        index_compressed_expected_upper = sink.bytes_written
        # A deliberately loose zlib-compressBound-style expansion allowance,
        # plus gzip framing and rounding.  This is a storage upper bound, not
        # an assumption that the highly repetitive index will compress well.
        index_compressed_worst_case_upper = (
            index_uncompressed_upper
            + ((index_uncompressed_upper + 4_095) // 4_096)
            + ((index_uncompressed_upper + 16_383) // 16_384)
            + ((index_uncompressed_upper + 33_554_431) // 33_554_432)
            + 128
        )

        entity_count, entity_key_bytes = entity_store.entity_storage_basis()
        sqlite_temp_upper = (
            1 << 20
        ) + 32 * (entity_key_bytes + entity_count * 512)
        spool_temp_upper = 0
        for plan_key, bucket_bytes in spool_bytes_by_plan.items():
            largest_aggregate = max(
                (
                    size for (candidate_plan, _bucket), size
                    in aggregate_bytes_by_plan_bucket.items()
                    if candidate_plan == plan_key
                ),
                default=0,
            )
            spool_temp_upper = max(
                spool_temp_upper, bucket_bytes + largest_aggregate
            )
        manifest_and_receipt_reserve = 4 << 20
        peak_temporary_bytes_upper = (
            sqlite_temp_upper
            + spool_temp_upper
            + payload_bytes_upper
            + index_compressed_worst_case_upper
            + manifest_and_receipt_reserve
        )
        expected_regular_files = expected_shards + 2
        expected_directories = 2
        return {
            "dataset_id": binding.row.dataset_id,
            "repo_id": binding.row.repo_id,
            "revision": binding.row.revision,
            "binding_sha256": binding.binding_sha256,
            "policy_sha256": policy.policy_sha256,
            "source_receipt_sha256": binding.source_receipt_sha256,
            "source_files": [
                {"path": relative, "bytes": size, "sha256": digest}
                for relative, _path, size, digest in binding.source_files
            ],
            "source_validation": {
                "status": "validated_count_only",
                "source_rows": source_rows,
                "entity_count": entity_count,
                "source_rows_by_file": rows_by_source,
                "payload_values_packed": False,
            },
            "candidate_stream_count": candidate_stream_count,
            "selected_stream_count": selected_stream_count,
            "unselected_stream_count": candidate_stream_count - selected_stream_count,
            "selected_logical_bytes": selected_logical_bytes,
            "longest_prefix_witness": {
                "all_candidates_selected": all_selected,
                "first_unselected_logical_bytes": first_unselected_logical_bytes,
                "proof_boundary": (
                    "all_candidates_selected"
                    if all_selected
                    else "selected_bytes_plus_first_unselected_exceeds_cap"
                ),
            },
            "expected_payload": {
                "byte_upper_bound": payload_bytes_upper,
                "shard_count_upper_bound": expected_shards,
                "shard_byte_cap": binding.chunk_byte_cap,
                "method": (
                    "replay next-fit shard append order over every selected chunk; assume every "
                    "value is present and include one missing bitmap bit per row"
                ),
            },
            "expected_stream_index": {
                "record_count": 2 + selected_entity_count + selected_stream_count,
                "entity_record_count": selected_entity_count,
                "stream_record_count": selected_stream_count,
                "uncompressed_bytes_upper_bound": index_uncompressed_upper,
                "compressed_bytes_conservative_skeleton": (
                    index_compressed_expected_upper
                ),
                "compressed_bytes_worst_case_upper_bound": (
                    index_compressed_worst_case_upper
                ),
                "compressed_byte_cap": binding.metadata_compressed_byte_cap,
                "conservative_method": (
                    "canonical JSONL with all-value-present chunk payload lengths, maximum "
                    "count fields, signed-int64 timestamp text, maximum shard path width, and "
                    "deterministic gzip mtime=0; worst-case additionally uses a loose "
                    "zlib-compressBound-style expansion allowance plus gzip framing"
                ),
            },
            "peak_temporary_bytes_upper_bound": peak_temporary_bytes_upper,
            "peak_temporary_method": {
                "sqlite_entity_store_upper_bound": sqlite_temp_upper,
                "plan_spool_and_aggregate_upper_bound": spool_temp_upper,
                "payload_upper_bound": payload_bytes_upper,
                "index_worst_case_upper_bound": index_compressed_worst_case_upper,
                "manifest_and_receipt_reserve": manifest_and_receipt_reserve,
            },
            "expected_filesystem_objects": {
                "regular_files": expected_regular_files,
                "directories": expected_directories,
                "total": expected_regular_files + expected_directories,
                "formula": "payload_shards + stream_index + dataset_manifest + dataset_dir + payload_dir",
            },
            "layout_assertions": {
                "per_stream_payload_files": 0,
                "per_stream_payload_directories": 0,
                "payload_files_are_dataset_level_shards": True,
                "stream_metadata_is_in_single_compressed_index": True,
            },
        }
    finally:
        entity_store.close()


def run_planning_receipt(
    *,
    scope_path: Path,
    mirror: Path,
    policy_path: Path,
    output: Path,
    planning_receipt: Path,
    dataset_byte_cap: int = DEFAULT_DATASET_BYTE_CAP,
    chunk_byte_cap: int = DEFAULT_CHUNK_BYTE_CAP,
    metadata_compressed_byte_cap: int = DEFAULT_METADATA_COMPRESSED_BYTE_CAP,
    storage_layout: str = "sharded_v1",
    dataset_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate sources and atomically publish a deterministic no-payload plan."""
    if storage_layout != "sharded_v1":
        raise PreparationError("planning receipts are supported only for sharded_v1")
    if dataset_byte_cap <= 0 or dataset_byte_cap > DEFAULT_DATASET_BYTE_CAP:
        raise PreparationError("dataset byte cap must be in [1, 1 GiB]")
    if chunk_byte_cap < 8 or chunk_byte_cap > min(
        dataset_byte_cap, DEFAULT_CHUNK_BYTE_CAP
    ):
        raise PreparationError("sharded chunk byte cap must be in [8, min(dataset cap, 64 MiB)]")
    if metadata_compressed_byte_cap <= 0 or metadata_compressed_byte_cap > (1 << 30):
        raise PreparationError("metadata compressed byte cap must be in [1, 1 GiB]")
    target = _validated_planning_receipt_path(
        planning_receipt, output=output, mirror=mirror
    )
    mirror_manifest_path = mirror.resolve(strict=True) / "DOWNLOAD_MANIFEST.json"
    mirror_manifest_sha256 = sha256_file(mirror_manifest_path)
    producer_sha256 = sha256_file(PRODUCER_PATH)
    helper_timeseries_sha256 = sha256_file(HELPER_TIMESERIES_PATH)
    helper_tsfile_sha256 = sha256_file(HELPER_TSFILE_PATH)
    scope, rows, scope_sha256 = load_scope(scope_path)
    policies, policy_file_sha256 = load_policies(
        policy_path,
        scope_id=str(scope["scope_id"]),
        scope_sha256=scope_sha256,
        scope_rows=rows,
    )
    bindings = bind_mirror(
        mirror,
        scope=scope,
        rows=rows,
        scope_sha256=scope_sha256,
        policies=policies,
        policy_file_sha256=policy_file_sha256,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=chunk_byte_cap,
        metadata_compressed_byte_cap=metadata_compressed_byte_cap,
        storage_layout=storage_layout,
        producer_sha256=producer_sha256,
        helper_timeseries_sha256=helper_timeseries_sha256,
        helper_tsfile_sha256=helper_tsfile_sha256,
    )
    if dataset_ids:
        requested = set(dataset_ids)
        known = {binding.row.dataset_id for binding in bindings}
        if requested - known:
            raise PreparationError(
                f"requested datasets are outside scope: {sorted(requested - known)}"
            )
        bindings = tuple(
            binding for binding in bindings if binding.row.dataset_id in requested
        )
    if not bindings:
        raise PreparationError("planning selection contains no datasets")
    if target.exists() or target.is_symlink():
        raise PreparationError(f"planning receipt already exists: {target}")
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.planning-", dir=target.parent
    ) as working_name:
        plans = [
            _plan_dataset(binding, Path(working_name)) for binding in bindings
        ]
    if sha256_file(scope_path.resolve(strict=True)) != scope_sha256:
        raise PreparationError("scope changed during count-only planning")
    if sha256_file(policy_path.resolve(strict=True)) != policy_file_sha256:
        raise PreparationError("timestamp policy changed during count-only planning")
    if sha256_file(mirror_manifest_path) != mirror_manifest_sha256:
        raise PreparationError("mirror manifest changed during count-only planning")
    if (
        sha256_file(PRODUCER_PATH) != producer_sha256
        or sha256_file(HELPER_TIMESERIES_PATH) != helper_timeseries_sha256
        or sha256_file(HELPER_TSFILE_PATH) != helper_tsfile_sha256
    ):
        raise PreparationError("runtime code changed during count-only planning")
    for binding in bindings:
        source_receipt_path = (
            mirror.resolve(strict=True) / "_receipts" / f"{binding.row.dataset_id}.json"
        )
        if sha256_file(source_receipt_path) != binding.source_receipt_sha256:
            raise PreparationError(
                f"source receipt changed during planning for {binding.row.dataset_id}"
            )
        for relative, path, _size, digest in binding.source_files:
            if sha256_file(path) != digest:
                raise PreparationError(
                    f"source changed during planning for {binding.row.dataset_id}/{relative}"
                )
    total_shards = sum(
        int(plan["expected_payload"]["shard_count_upper_bound"])
        for plan in plans
    )
    total_dataset_regular_files = sum(
        int(plan["expected_filesystem_objects"]["regular_files"])
        for plan in plans
    )
    total_dataset_directories = sum(
        int(plan["expected_filesystem_objects"]["directories"])
        for plan in plans
    )
    collection_regular_files = total_dataset_regular_files + len(plans) + 1
    collection_directories = total_dataset_directories + 1
    receipt: dict[str, Any] = {
        "schema": PLANNING_RECEIPT_VERSION,
        "status": "complete",
        "mode": "count_only_no_payload_planning",
        "scope_id": scope["scope_id"],
        "scope_sha256": scope_sha256,
        "policy_file_sha256": policy_file_sha256,
        "mirror_manifest_sha256": mirror_manifest_sha256,
        "selected_dataset_count": len(bindings),
        "full_scope_dataset_count": len(rows),
        "selected_dataset_ids": [binding.row.dataset_id for binding in bindings],
        "caps": {
            "dataset_logical_byte_cap": dataset_byte_cap,
            "payload_shard_byte_cap": chunk_byte_cap,
            "metadata_compressed_byte_cap": metadata_compressed_byte_cap,
        },
        "storage_layout": storage_layout,
        "runtime_code_bindings": {
            "producer_sha256": producer_sha256,
            "helper_timeseries_sha256": helper_timeseries_sha256,
            "helper_tsfile_sha256": helper_tsfile_sha256,
        },
        "datasets": plans,
        "totals": {
            "candidate_stream_count": sum(
                int(plan["candidate_stream_count"]) for plan in plans
            ),
            "selected_stream_count": sum(
                int(plan["selected_stream_count"]) for plan in plans
            ),
            "selected_logical_bytes": sum(
                int(plan["selected_logical_bytes"]) for plan in plans
            ),
            "expected_payload_shard_count_upper_bound": total_shards,
            "expected_index_uncompressed_bytes_upper_bound": sum(
                int(plan["expected_stream_index"]["uncompressed_bytes_upper_bound"])
                for plan in plans
            ),
            "expected_index_compressed_bytes_conservative_skeleton": sum(
                int(plan["expected_stream_index"]["compressed_bytes_conservative_skeleton"])
                for plan in plans
            ),
            "expected_index_compressed_bytes_worst_case_upper_bound": sum(
                int(plan["expected_stream_index"]["compressed_bytes_worst_case_upper_bound"])
                for plan in plans
            ),
            "peak_temporary_bytes_upper_bound": max(
                int(plan["peak_temporary_bytes_upper_bound"]) for plan in plans
            ),
            "expected_collection_filesystem_objects": {
                "regular_files": collection_regular_files,
                "directories": collection_directories,
                "total": collection_regular_files + collection_directories,
                "includes": (
                    "dataset shards/indexes/manifests plus one dataset receipt per dataset, "
                    "the collection manifest, and the collection receipt directory"
                ),
            },
            "per_stream_payload_files": 0,
            "per_stream_payload_directories": 0,
        },
        "determinism": {
            "wall_clock_fields_included": False,
            "dataset_order": "frozen scope order filtered by exact requested IDs",
            "source_scan": (
                "full admitted row scan validates source hashes before and after each query, "
                "schema, timestamp conversion, row filters, typed TAG identity, and row counts"
            ),
            "payload_materialized": False,
        },
    }
    _atomic_json_no_clobber(target, receipt)
    return receipt


def run_collection(
    *,
    scope_path: Path,
    mirror: Path,
    policy_path: Path,
    output: Path,
    dataset_byte_cap: int = DEFAULT_DATASET_BYTE_CAP,
    chunk_byte_cap: int = DEFAULT_CHUNK_BYTE_CAP,
    metadata_compressed_byte_cap: int = DEFAULT_METADATA_COMPRESSED_BYTE_CAP,
    storage_layout: str = DEFAULT_STORAGE_LAYOUT,
    verify_only: bool = False,
    dataset_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if dataset_byte_cap <= 0 or dataset_byte_cap > DEFAULT_DATASET_BYTE_CAP:
        raise PreparationError("dataset byte cap must be in [1, 1 GiB]")
    if chunk_byte_cap < 8 or chunk_byte_cap > dataset_byte_cap:
        raise PreparationError("chunk byte cap must be in [8, dataset byte cap]")
    if storage_layout not in STORAGE_LAYOUTS:
        raise PreparationError(f"unsupported storage layout: {storage_layout}")
    if storage_layout == "sharded_v1" and chunk_byte_cap > DEFAULT_CHUNK_BYTE_CAP:
        raise PreparationError("sharded_v1 payload shards are capped at 64 MiB")
    if metadata_compressed_byte_cap <= 0 or metadata_compressed_byte_cap > (1 << 30):
        raise PreparationError("metadata compressed byte cap must be in [1, 1 GiB]")
    requested_output = output.absolute()
    try:
        requested_output_mode = os.lstat(requested_output).st_mode
    except FileNotFoundError:
        requested_output_mode = None
    except OSError as error:
        raise PreparationError(f"cannot inspect requested output path: {requested_output}") from error
    if requested_output_mode is not None and (
        stat.S_ISLNK(requested_output_mode)
        or not stat.S_ISDIR(requested_output_mode)
    ):
        raise PreparationError(
            f"requested output path must be a real directory, not a symlink or other path: "
            f"{requested_output}"
        )
    producer_sha256 = sha256_file(PRODUCER_PATH)
    helper_timeseries_sha256 = sha256_file(HELPER_TIMESERIES_PATH)
    helper_tsfile_sha256 = sha256_file(HELPER_TSFILE_PATH)
    scope, rows, scope_sha256 = load_scope(scope_path)
    policies, policy_file_sha256 = load_policies(
        policy_path,
        scope_id=str(scope["scope_id"]),
        scope_sha256=scope_sha256,
        scope_rows=rows,
    )
    bindings = bind_mirror(
        mirror,
        scope=scope,
        rows=rows,
        scope_sha256=scope_sha256,
        policies=policies,
        policy_file_sha256=policy_file_sha256,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=chunk_byte_cap,
        metadata_compressed_byte_cap=metadata_compressed_byte_cap,
        storage_layout=storage_layout,
        producer_sha256=producer_sha256,
        helper_timeseries_sha256=helper_timeseries_sha256,
        helper_tsfile_sha256=helper_tsfile_sha256,
    )
    if dataset_ids:
        requested = set(dataset_ids)
        known = {binding.row.dataset_id for binding in bindings}
        if requested - known:
            raise PreparationError(f"requested datasets are outside scope: {sorted(requested - known)}")
        bindings = tuple(binding for binding in bindings if binding.row.dataset_id in requested)
    # Recheck immediately before resolving so an output-path swap cannot turn a
    # previously inspected real directory into a symlink (including dangling).
    try:
        output_mode = os.lstat(requested_output).st_mode
    except FileNotFoundError:
        if verify_only:
            raise PreparationError(
                f"verify-only output directory does not exist: {requested_output}"
            )
        requested_output.mkdir(parents=True, exist_ok=False)
        output_mode = os.lstat(requested_output).st_mode
    except OSError as error:
        raise PreparationError(f"cannot inspect requested output path: {requested_output}") from error
    if stat.S_ISLNK(output_mode) or not stat.S_ISDIR(output_mode):
        raise PreparationError(
            f"requested output path must be a real directory, not a symlink or other path: "
            f"{requested_output}"
        )
    root = requested_output.resolve(strict=True)
    if verify_only:
        if not root.is_dir():
            raise PreparationError(f"verify-only output directory does not exist: {root}")
    requested_receipts_dir = root / "_receipts"
    try:
        receipts_mode = os.lstat(requested_receipts_dir).st_mode
    except FileNotFoundError:
        if verify_only:
            raise PreparationError("verify-only receipt directory does not exist")
        requested_receipts_dir.mkdir(exist_ok=False)
        receipts_mode = os.lstat(requested_receipts_dir).st_mode
    except OSError as error:
        raise PreparationError("cannot inspect materialization receipt directory") from error
    if stat.S_ISLNK(receipts_mode) or not stat.S_ISDIR(receipts_mode):
        raise PreparationError("receipt directory must be a real directory, not a symlink")
    receipts_dir = requested_receipts_dir.resolve(strict=True)
    if receipts_dir.parent != root:
        raise PreparationError("receipt directory escapes the materialization output root")
    results: list[dict[str, Any]] = []
    for binding in bindings:
        receipt_path = receipts_dir / f"{binding.row.dataset_id}.json"
        try:
            receipt_mode = os.lstat(receipt_path).st_mode
        except FileNotFoundError:
            receipt_mode = None
        except OSError as error:
            raise PreparationError(f"cannot inspect receipt: {receipt_path.name}") from error
        if receipt_mode is not None and not stat.S_ISREG(receipt_mode):
            raise PreparationError(f"receipt is not a regular file: {receipt_path.name}")
        if receipt_mode is not None:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            result = verify_dataset_output(binding, root, receipt)
        elif verify_only:
            raise PreparationError(f"missing materialization receipt for {binding.row.dataset_id}")
        else:
            result = materialize_dataset(binding, root)
            _atomic_json(receipt_path, result)
            result = dict(result) | {"resume_status": "created"}
        results.append(result)
    status_counts = {
        "materialized": sum(result["status"] == "materialized" for result in results),
        "blocked": sum(result["status"] == "blocked" for result in results),
    }
    receipt_rows = [
        {
            "dataset_id": result["dataset_id"],
            "status": result["status"],
            "receipt": f"_receipts/{result['dataset_id']}.json",
            "receipt_sha256": sha256_file(receipts_dir / f"{result['dataset_id']}.json"),
        }
        for result in results
    ]
    top: dict[str, Any] = {
        "schema": COLLECTION_VERSION,
        "status": "complete" if not status_counts["blocked"] else "complete_with_blocked_datasets",
        "mode": "verify_only" if verify_only else "prepare_or_resume",
        "scope_id": scope["scope_id"],
        "scope_sha256": scope_sha256,
        "policy_file_sha256": policy_file_sha256,
        "selected_dataset_count": len(bindings),
        "full_scope_dataset_count": len(rows),
        "status_counts": status_counts,
        "dataset_byte_cap": dataset_byte_cap,
        "chunk_byte_cap": chunk_byte_cap,
        "metadata_compressed_byte_cap": metadata_compressed_byte_cap,
        "storage_layout": storage_layout,
        "runtime_code_bindings": {
            "producer_sha256": producer_sha256,
            "helper_timeseries_sha256": helper_timeseries_sha256,
            "helper_tsfile_sha256": helper_tsfile_sha256,
        },
        "datasets": receipt_rows,
    }
    if verify_only:
        stored_path = root / "MATERIALIZATION_MANIFEST.json"
        if (
            not stored_path.exists()
            or not stat.S_ISREG(os.lstat(stored_path).st_mode)
        ):
            raise PreparationError("verify-only top-level materialization manifest is missing")
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        for key in (
            "schema", "status", "scope_id", "scope_sha256", "policy_file_sha256",
            "selected_dataset_count", "full_scope_dataset_count", "status_counts",
            "dataset_byte_cap", "chunk_byte_cap", "storage_layout", "datasets",
            "metadata_compressed_byte_cap", "runtime_code_bindings",
        ):
            if stored.get(key) != top.get(key):
                raise PreparationError(f"top-level materialization manifest mismatch: {key}")
        if len(bindings) == len(rows):
            expected_children = {"_receipts", "MATERIALIZATION_MANIFEST.json"} | {
                result["dataset_id"]
                for result in results if result["status"] == "materialized"
            }
            actual_children: set[str] = set()
            for path in root.iterdir():
                mode = os.lstat(path).st_mode
                expected_directory = path.name == "_receipts" or path.name in {
                    result["dataset_id"]
                    for result in results if result["status"] == "materialized"
                }
                if (
                    stat.S_ISLNK(mode)
                    or (expected_directory and not stat.S_ISDIR(mode))
                    or (not expected_directory and not stat.S_ISREG(mode))
                ):
                    raise PreparationError("top-level materialization path type mismatch")
                actual_children.add(path.name)
            if actual_children != expected_children:
                raise PreparationError("unexpected or missing top-level materialization paths")
            expected_receipts = {
                f"{binding.row.dataset_id}.json" for binding in bindings
            }
            actual_receipts: set[str] = set()
            for path in receipts_dir.iterdir():
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    raise PreparationError("receipt tree contains a non-regular file")
                actual_receipts.add(path.name)
            if actual_receipts != expected_receipts:
                raise PreparationError("unexpected or missing materialization receipt files")
    else:
        top["completed_unix_seconds"] = time.time()
        _atomic_json(root / "MATERIALIZATION_MANIFEST.json", top)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--timestamp-policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-byte-cap", type=int, default=DEFAULT_DATASET_BYTE_CAP)
    parser.add_argument("--chunk-byte-cap", type=int, default=DEFAULT_CHUNK_BYTE_CAP)
    parser.add_argument(
        "--metadata-compressed-byte-cap",
        type=int,
        default=DEFAULT_METADATA_COMPRESSED_BYTE_CAP,
    )
    parser.add_argument(
        "--storage-layout",
        choices=sorted(STORAGE_LAYOUTS),
        default=DEFAULT_STORAGE_LAYOUT,
    )
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate/count sources and write no payload output",
    )
    parser.add_argument(
        "--planning-receipt",
        type=Path,
        help="new, outside-output JSON receipt path required by --plan-only",
    )
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            if args.verify_only:
                raise PreparationError("--plan-only and --verify-only are mutually exclusive")
            if args.planning_receipt is None:
                raise PreparationError("--plan-only requires --planning-receipt")
            result = run_planning_receipt(
                scope_path=args.scope,
                mirror=args.mirror,
                policy_path=args.timestamp_policies,
                output=args.output,
                planning_receipt=args.planning_receipt,
                dataset_byte_cap=args.dataset_byte_cap,
                chunk_byte_cap=args.chunk_byte_cap,
                metadata_compressed_byte_cap=args.metadata_compressed_byte_cap,
                storage_layout=args.storage_layout,
                dataset_ids=args.dataset,
            )
        else:
            if args.planning_receipt is not None:
                raise PreparationError("--planning-receipt requires --plan-only")
            result = run_collection(
                scope_path=args.scope,
                mirror=args.mirror,
                policy_path=args.timestamp_policies,
                output=args.output,
                dataset_byte_cap=args.dataset_byte_cap,
                chunk_byte_cap=args.chunk_byte_cap,
                metadata_compressed_byte_cap=args.metadata_compressed_byte_cap,
                storage_layout=args.storage_layout,
                verify_only=args.verify_only,
                dataset_ids=args.dataset,
            )
    except (OSError, ValueError, PreparationError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
