#!/usr/bin/env python3
"""Prepare audited Apache TsFile inputs for weighted-GRPO experiments.

This extractor is intentionally schema-driven.  It never infers a timestamp
unit, timezone, entity key, device name, measurement type, or table/tree model.
Only native TsFile INT32, INT64, FLOAT, and DOUBLE measurements are admitted.
The emitted directory uses the same block, bitmap, fingerprint, split, and
``B_raw`` contract as ``prepare_weighted_grpo_timeseries_v1.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import sys
import tempfile
import uuid
from typing import Any, Iterable, Mapping, Sequence

from prepare_weighted_grpo_timeseries_v1 import (
    COHORT_ASSIGNMENT_SHA256,
    COHORT_POLICY_SHA256,
    COHORT_VERSION,
    DEFAULT_COHORT_MANIFEST,
    FORMAT_VERSION,
    MAX_BLOCK_BYTES,
    MAX_DATASET_BYTES,
    MIN_SELECTED_ROWS_PER_ENTITY,
    TRAIN_DENOMINATOR,
    TRAIN_NUMERATOR,
    NumericColumn,
    PreparationContract,
    PreparationError,
    TimestampContract,
    _BlockBuilder,
    _atomic_json,
    _canonical_json,
    _convert_epoch_units,
    _entity_digest,
    _flush_builder,
    _manifest_fingerprint_basis,
    _sha256_bytes,
    _split_contract_for_binding,
    _source_signature,
    equal_prefix_allocation,
    load_cohort_manifest,
    max_rows_per_block,
    sha256_file,
    validate_contract_against_cohort,
    verify_prepared_dataset,
)


SPEC_VERSION = "weighted-grpo-tsfile-preparation-v1"
COLLECTION_SPEC_VERSION = "weighted-grpo-tsfile-collection-preparation-v1"
ENGINE_VERSION = "weighted-grpo-tsfile-extractor-v1"
COLLECTION_ENGINE_VERSION = "weighted-grpo-tsfile-collection-extractor-v1"
TSFILE_TYPE_TO_DTYPE = {
    "INT32": "int32",
    "INT64": "int64",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
TIMESTAMP_COORDINATE_SEMANTICS = {
    "calendar_date_utc_midnight_encoding": {
        "timezone": "UTC",
        "source_kind": "calendar_date_coordinate",
        "required_false_flags": ("source_event_timezone_inferred",),
    },
    "portuguese_civil_time_coordinate_encoding": {
        "timezone": "Europe/Lisbon source civil-time labels",
        "source_kind": "civil_time_coordinate",
        "required_false_flags": (
            "source_event_timezone_inferred",
            "event_utc_instant_claim",
            "event_time_mapping_is_injective",
        ),
    },
    "helsinki_civil_time_coordinate_encoding": {
        "timezone": "Europe/Helsinki source civil-time labels",
        "source_kind": "civil_time_coordinate",
        "required_false_flags": (
            "source_event_timezone_inferred",
            "event_utc_instant_claim",
            "event_time_mapping_is_injective",
        ),
    },
}


@dataclass(frozen=True)
class TsFileValueColumn:
    source_name: str
    output_name: str
    tsfile_type: str

    @property
    def dtype(self) -> str:
        return TSFILE_TYPE_TO_DTYPE[self.tsfile_type]

    @property
    def numeric_column(self) -> NumericColumn:
        return NumericColumn(self.source_name, self.output_name, self.dtype)


@dataclass(frozen=True)
class TsFileSource:
    path: Path
    expected_sha256: str
    repo_path: str | None


@dataclass(frozen=True)
class TsFilePreparationSpec:
    dataset_id: str
    domain: str
    source_group: str
    source_format: str
    source_files: tuple[TsFileSource, ...]
    schema_policy: str | None
    upstream_revision: str
    upstream_locator: str
    model: str
    table_name: str | None
    entity_columns: tuple[str, ...]
    devices: tuple[str, ...]
    value_columns: tuple[TsFileValueColumn, ...]
    timestamp: TimestampContract
    timestamp_semantic_kind: str | None
    timestamp_semantic_flags: tuple[tuple[str, bool], ...]
    query_start: int
    query_end: int
    dataset_byte_cap: int
    block_byte_cap: int
    spec_sha256: str

    @property
    def is_collection(self) -> bool:
        return self.source_format == "tsfile_collection"

    @property
    def source_path(self) -> Path:
        """Compatibility accessor for the unchanged single-file path."""
        return self.source_files[0].path

    @property
    def expected_source_sha256(self) -> str:
        """Compatibility accessor for the unchanged single-file fingerprint."""
        return self.source_files[0].expected_sha256

    @property
    def source_identity_sha256(self) -> str:
        if not self.is_collection:
            return self.expected_source_sha256
        return _sha256_bytes(_canonical_json([
            {
                "repo_path": source.repo_path,
                "sha256": source.expected_sha256,
            }
            for source in self.source_files
        ]))

    @property
    def timestamp_column(self) -> NumericColumn:
        return NumericColumn(
            self.timestamp.source_name,
            self.timestamp.output_name,
            "int64",
        )

    @property
    def numeric_columns(self) -> tuple[NumericColumn, ...]:
        return (
            self.timestamp_column,
            *(column.numeric_column for column in self.value_columns),
        )

    @property
    def logical_row_bytes(self) -> int:
        return sum(column.width for column in self.numeric_columns)


def _required_text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, context: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PreparationError(f"{context} must be a list of strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result):
        raise PreparationError(f"{context} cannot contain blank values")
    if not allow_empty and not result:
        raise PreparationError(f"{context} cannot be empty")
    if len(result) != len(set(result)):
        raise PreparationError(f"{context} contains duplicates")
    return result


def _parse_value_columns(value: Any, context: str) -> tuple[TsFileValueColumn, ...]:
    if not isinstance(value, list) or not value:
        raise PreparationError(f"{context} must be a non-empty list")
    columns: list[TsFileValueColumn] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, Mapping):
            raise PreparationError(f"{item_context} must be an object")
        source_name = _required_text(item, "source_name", item_context)
        output_name = str(item.get("output_name", source_name)).strip()
        tsfile_type = _required_text(item, "tsfile_type", item_context).upper()
        if not output_name:
            raise PreparationError(f"{item_context}.output_name cannot be blank")
        if tsfile_type not in TSFILE_TYPE_TO_DTYPE:
            raise PreparationError(
                f"{item_context}.tsfile_type must be one of "
                f"{sorted(TSFILE_TYPE_TO_DTYPE)}"
            )
        columns.append(TsFileValueColumn(source_name, output_name, tsfile_type))
    if len({column.source_name for column in columns}) != len(columns):
        raise PreparationError(f"{context} contains duplicate source names")
    if len({column.output_name for column in columns}) != len(columns):
        raise PreparationError(f"{context} contains duplicate output names")
    return tuple(columns)


def load_tsfile_spec(path: Path) -> TsFilePreparationSpec:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise PreparationError("TsFile preparation spec must be a JSON object")
    format_version = raw.get("format_version")
    if format_version not in {SPEC_VERSION, COLLECTION_SPEC_VERSION}:
        raise PreparationError(
            "format_version must be "
            f"{SPEC_VERSION!r} or {COLLECTION_SPEC_VERSION!r}"
        )

    dataset_id = _required_text(raw, "dataset_id", "spec")
    domain = _required_text(raw, "domain", "spec")
    source_group = _required_text(raw, "source_group", "spec")
    source = raw.get("source")
    if not isinstance(source, Mapping):
        raise PreparationError("spec.source must be an object")
    source_format = _required_text(source, "format", "spec.source")
    expected_format = (
        "tsfile_collection"
        if format_version == COLLECTION_SPEC_VERSION
        else "tsfile"
    )
    if source_format != expected_format:
        raise PreparationError(
            f"{format_version} requires spec.source.format={expected_format!r}"
        )

    source_files: list[TsFileSource] = []
    schema_policy: str | None = None
    if source_format == "tsfile":
        path_text = _required_text(source, "path", "spec.source")
        source_path = Path(path_text)
        if not source_path.is_absolute():
            source_path = (path.resolve().parent / source_path).resolve()
        if not source_path.is_file():
            raise PreparationError(f"source TsFile does not exist: {source_path}")
        expected_source_sha256 = _required_text(
            source, "expected_sha256", "spec.source"
        ).casefold()
        if re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256) is None:
            raise PreparationError(
                "spec.source.expected_sha256 must be 64 lowercase hex digits"
            )
        source_files.append(TsFileSource(
            path=source_path,
            expected_sha256=expected_source_sha256,
            repo_path=None,
        ))
    else:
        schema_policy = _required_text(source, "schema_policy", "spec.source")
        if schema_policy != "identical_across_all_files":
            raise PreparationError(
                "TsFile collections require schema_policy=identical_across_all_files"
            )
        raw_files = source.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise PreparationError("spec.source.files must be a non-empty ordered list")
        seen_paths: set[Path] = set()
        seen_repo_paths: set[str] = set()
        seen_hashes: set[str] = set()
        series_indexes: list[int] = []
        for index, item in enumerate(raw_files):
            context = f"spec.source.files[{index}]"
            if not isinstance(item, Mapping):
                raise PreparationError(f"{context} must be an object")
            repo_path = _required_text(item, "repo_path", context)
            if repo_path in seen_repo_paths:
                raise PreparationError(f"duplicate collection repo_path: {repo_path}")
            seen_repo_paths.add(repo_path)
            match = re.search(r"_series_(\d+)\.tsfile$", repo_path)
            if match is None:
                raise PreparationError(
                    f"collection repo_path lacks a numeric series suffix: {repo_path}"
                )
            series_indexes.append(int(match.group(1)))
            path_text = _required_text(item, "path", context)
            source_path = Path(path_text)
            if not source_path.is_absolute():
                source_path = (path.resolve().parent / source_path).resolve()
            if source_path in seen_paths:
                raise PreparationError(f"duplicate collection source path: {source_path}")
            seen_paths.add(source_path)
            if not source_path.is_file():
                raise PreparationError(f"source TsFile does not exist: {source_path}")
            expected_source_sha256 = _required_text(
                item, "expected_sha256", context
            ).casefold()
            if re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256) is None:
                raise PreparationError(
                    f"{context}.expected_sha256 must be 64 lowercase hex digits"
                )
            if expected_source_sha256 in seen_hashes:
                raise PreparationError(
                    "collection repeats an exact source fingerprint; file substitution "
                    f"or duplication is not allowed: {repo_path}"
                )
            seen_hashes.add(expected_source_sha256)
            source_files.append(TsFileSource(
                path=source_path,
                expected_sha256=expected_source_sha256,
                repo_path=repo_path,
            ))
        if series_indexes != sorted(series_indexes) or len(series_indexes) != len(set(series_indexes)):
            raise PreparationError(
                "spec.source.files must be ordered by unique numeric series suffix"
            )
    upstream_revision = _required_text(
        source, "upstream_revision", "spec.source"
    )
    upstream_locator = _required_text(source, "upstream_locator", "spec.source")

    model = _required_text(raw, "model", "spec").lower()
    if model not in {"table", "tree"}:
        raise PreparationError("spec.model must be table or tree")
    model_config = raw.get(model)
    if not isinstance(model_config, Mapping):
        raise PreparationError(f"spec.{model} must be an object")
    value_columns = _parse_value_columns(
        model_config.get("value_columns"), f"spec.{model}.value_columns"
    )
    table_name: str | None = None
    devices: tuple[str, ...] = ()
    if model == "table":
        table_name = _required_text(model_config, "table_name", "spec.table")
        entity_columns = _string_list(
            model_config.get("entity_columns", []),
            "spec.table.entity_columns",
            allow_empty=True,
        )
        if set(entity_columns) & {column.source_name for column in value_columns}:
            raise PreparationError("table entity columns cannot also be value columns")
    else:
        devices = _string_list(
            model_config.get("devices"), "spec.tree.devices", allow_empty=False
        )
        entity_columns = ("__device__",)

    timestamp_raw = raw.get("timestamp")
    if not isinstance(timestamp_raw, Mapping):
        raise PreparationError("spec.timestamp must be an object")
    source_unit = _required_text(timestamp_raw, "source_unit", "spec.timestamp")
    output_unit = _required_text(timestamp_raw, "output_unit", "spec.timestamp")
    if source_unit not in {"s", "ms", "us", "ns"}:
        raise PreparationError("spec.timestamp.source_unit must be s, ms, us, or ns")
    if output_unit not in {"s", "ms", "us", "ns"}:
        raise PreparationError("spec.timestamp.output_unit must be s, ms, us, or ns")
    timezone_name = _required_text(timestamp_raw, "timezone", "spec.timestamp")
    semantic_kind_value = timestamp_raw.get("semantic_kind")
    if semantic_kind_value is None:
        semantic_kind = None
        timestamp_source_kind = "epoch"
        semantic_flags: tuple[tuple[str, bool], ...] = ()
        if timezone_name != "UTC":
            raise PreparationError(
                "TsFile epoch time requires UTC; non-UTC labels need an audited "
                "timestamp.semantic_kind"
            )
    else:
        semantic_kind = str(semantic_kind_value).strip()
        semantic_rule = TIMESTAMP_COORDINATE_SEMANTICS.get(semantic_kind)
        if semantic_rule is None:
            raise PreparationError(
                f"unsupported spec.timestamp.semantic_kind: {semantic_kind!r}"
            )
        if timezone_name != semantic_rule["timezone"]:
            raise PreparationError(
                f"{semantic_kind} requires timestamp.timezone="
                f"{semantic_rule['timezone']!r}"
            )
        if source_unit != output_unit:
            raise PreparationError(
                "audited timestamp coordinates must preserve their stored unit"
            )
        required_false = tuple(semantic_rule["required_false_flags"])
        for flag in required_false:
            if timestamp_raw.get(flag) is not False:
                raise PreparationError(
                    f"{semantic_kind} requires spec.timestamp.{flag}=false"
                )
        timestamp_source_kind = str(semantic_rule["source_kind"])
        semantic_flags = tuple((flag, False) for flag in required_false)
    duplicate_policy = _required_text(
        timestamp_raw, "duplicate_policy", "spec.timestamp"
    )
    if duplicate_policy not in {"reject", "allow_preserve_order"}:
        raise PreparationError(
            "spec.timestamp.duplicate_policy must be reject or allow_preserve_order"
        )
    output_name = str(timestamp_raw.get("output_name", "timestamp_unix")).strip()
    if not output_name:
        raise PreparationError("spec.timestamp.output_name cannot be blank")
    if output_name in {column.output_name for column in value_columns}:
        raise PreparationError("timestamp output name conflicts with a value column")
    timestamp_source_name = "Time" if source_format == "tsfile_collection" else "time"
    timestamp = TimestampContract(
        source_name=timestamp_source_name,
        output_name=output_name,
        source_kind=timestamp_source_kind,
        source_unit=source_unit,
        output_unit=output_unit,
        timezone_name=timezone_name,
        formats=("iso8601",),
        ambiguous_fold=None,
        duplicate_policy=duplicate_policy,
    )

    query_range = raw.get("query_range", {})
    if not isinstance(query_range, Mapping):
        raise PreparationError("spec.query_range must be an object")
    query_start = int(query_range.get("start_inclusive", INT64_MIN))
    query_end = int(query_range.get("end_inclusive", INT64_MAX))
    if not INT64_MIN <= query_start <= query_end <= INT64_MAX:
        raise PreparationError("query_range must be an ordered int64 interval")

    if source_format == "tsfile_collection":
        row_contract = raw.get("row_contract")
        if not isinstance(row_contract, Mapping):
            raise PreparationError("collection spec.row_contract must be an object")
        expected_row_contract = {
            "ordering_key": "Time",
            "emit_ordering_key_as_numeric_column": True,
            "ordering_key_output_dtype": "INT64",
            "sort_scope": "global_entity_timestamp_ascending",
            "duplicate_comparison_scope": "all_listed_source_files",
            "cross_file_entity_merge": True,
            "source_file_order": "numeric_series_suffix_ascending",
            "split_after_global_merge": "per_entity_chronological_80_20",
        }
        for key, expected in expected_row_contract.items():
            if row_contract.get(key) != expected:
                raise PreparationError(
                    f"collection row_contract.{key} must be {expected!r}"
                )

    dataset_byte_cap = int(raw.get("dataset_byte_cap", MAX_DATASET_BYTES))
    block_byte_cap = int(raw.get("block_byte_cap", MAX_BLOCK_BYTES))
    if not 0 < dataset_byte_cap <= MAX_DATASET_BYTES:
        raise PreparationError(f"dataset_byte_cap must be in 1..{MAX_DATASET_BYTES}")
    if not 0 < block_byte_cap <= MAX_BLOCK_BYTES:
        raise PreparationError(f"block_byte_cap must be in 1..{MAX_BLOCK_BYTES}")
    for column in (
        NumericColumn("time", output_name, "int64"),
        *(column.numeric_column for column in value_columns),
    ):
        max_rows_per_block(column.width, block_byte_cap)

    return TsFilePreparationSpec(
        dataset_id=dataset_id,
        domain=domain,
        source_group=source_group,
        source_format=source_format,
        source_files=tuple(source_files),
        schema_policy=schema_policy,
        upstream_revision=upstream_revision,
        upstream_locator=upstream_locator,
        model=model,
        table_name=table_name,
        entity_columns=entity_columns,
        devices=devices,
        value_columns=value_columns,
        timestamp=timestamp,
        timestamp_semantic_kind=semantic_kind,
        timestamp_semantic_flags=semantic_flags,
        query_start=query_start,
        query_end=query_end,
        dataset_byte_cap=dataset_byte_cap,
        block_byte_cap=block_byte_cap,
        spec_sha256=_sha256_bytes(_canonical_json(raw)),
    )


def _load_tsfile_module() -> Any:
    try:
        import tsfile  # type: ignore[import-not-found]
    except Exception as error:
        raise PreparationError(
            "Apache tsfile Python package is unavailable; install the pinned reader "
            f"before extraction: {type(error).__name__}: {error}"
        ) from error
    if not hasattr(tsfile, "TsFileReader"):
        raise PreparationError("installed tsfile package has no TsFileReader")
    return tsfile


def _validate_tsfile_container(path: Path) -> None:
    """Reject a mislabeled or truncated file before invoking the native reader."""
    size = path.stat().st_size
    if size < 12:
        raise PreparationError(
            f"TSFILE_CONTAINER_BLOCKER: file is too short to be a TsFile: {size} bytes"
        )
    with path.open("rb") as handle:
        header = handle.read(6)
        handle.seek(-6, os.SEEK_END)
        trailer = handle.read(6)
    if header != b"TsFile" or trailer != b"TsFile":
        raise PreparationError(
            "TSFILE_CONTAINER_BLOCKER: expected TsFile magic at both file boundaries"
        )


def _tsfile_version(tsfile_module: Any) -> str:
    try:
        from importlib.metadata import version

        return version("tsfile")
    except Exception:
        return str(getattr(tsfile_module, "__version__", "unknown"))


def _type_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _category_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def inspect_tsfile_source(path: Path) -> dict[str, Any]:
    """Fingerprint a small local TsFile and expose only API-backed schema facts."""
    source_path = path.resolve()
    if not source_path.is_file():
        raise PreparationError(f"source TsFile does not exist: {source_path}")
    _validate_tsfile_container(source_path)
    tsfile = _load_tsfile_module()
    reader = None
    try:
        reader = tsfile.TsFileReader(str(source_path))
        schemas = reader.get_all_table_schemas()
        if not isinstance(schemas, Mapping):
            raise PreparationError(
                "TSFILE_SCHEMA_BLOCKER: reader returned an invalid table-schema mapping"
            )
        table_schemas = []
        for table_name in sorted(schemas, key=str):
            columns = _table_schema_columns(schemas[table_name])
            table_schemas.append({
                "table_name": str(table_name),
                "columns": [
                    {"name": name, "tsfile_type": data_type, "category": category}
                    for name, (data_type, category) in columns.items()
                ],
            })
        result: dict[str, Any] = {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "tsfile_python_version": _tsfile_version(tsfile),
            "timestamp_unit": None,
            "timestamp_timezone": None,
            "timestamp_contract_status": "must_be_supplied_from_trusted_source_metadata",
            "precision_inferred": False,
            "table_schemas": table_schemas,
        }
        if table_schemas:
            result["model"] = "table"
            result["next_step"] = (
                "copy exact table/tag/field names and native types into a table spec; "
                "supply timestamp unit/timezone separately"
            )
        else:
            result["model"] = "tree_or_legacy"
            result["audit_blocker"] = (
                "TSFILE_TREE_DISCOVERY_BLOCKER: this tsfile Python API exposes no stable "
                "tree device/measurement enumeration; supply exact devices, measurement "
                "names/types, timestamp unit, and timezone from trusted source metadata"
            )
        return result
    except PreparationError:
        raise
    except Exception as error:
        raise PreparationError(
            f"TSFILE_READER_BLOCKER: {type(error).__name__}: {error}"
        ) from error
    finally:
        if reader is not None:
            reader.close()


def _typed_entity_component(type_name: str, value: Any) -> str:
    if value is None:
        raise PreparationError("entity/tag values cannot be null")
    if isinstance(value, bytes):
        normalized: Any = {"hex": value.hex()}
    elif isinstance(value, (str, int, float, bool)):
        normalized = value
    else:
        normalized = str(value)
    try:
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as error:
        raise PreparationError("entity/tag values cannot contain NaN or infinity") from error
    return f"{type_name}:{encoded}"


def _pack_value(value: Any, column: TsFileValueColumn) -> bytes:
    try:
        if column.tsfile_type == "INT32":
            return struct.pack("<i", int(value))
        if column.tsfile_type == "INT64":
            return struct.pack("<q", int(value))
        if column.tsfile_type == "FLOAT":
            return struct.pack("<f", float(value))
        if column.tsfile_type == "DOUBLE":
            return struct.pack("<d", float(value))
    except (ValueError, TypeError, OverflowError, struct.error) as error:
        raise PreparationError(
            f"value cannot be represented as native {column.tsfile_type}: {value!r}"
        ) from error
    raise AssertionError(column.tsfile_type)


def _table_schema_columns(schema: Any) -> dict[str, tuple[str, str]]:
    columns = (
        schema.get_columns()
        if hasattr(schema, "get_columns")
        else getattr(schema, "columns", ())
    )
    result: dict[str, tuple[str, str]] = {}
    for column in columns:
        name = str(
            column.get_column_name()
            if hasattr(column, "get_column_name")
            else getattr(column, "column_name", "")
        )
        data_type = _type_name(
            column.get_data_type()
            if hasattr(column, "get_data_type")
            else getattr(column, "data_type", "")
        )
        category = _category_name(
            column.get_category()
            if hasattr(column, "get_category")
            else getattr(column, "category", "")
        )
        result[name] = (data_type, category)
    return result


def _validate_model_and_schema(reader: Any, spec: TsFilePreparationSpec) -> None:
    schemas = reader.get_all_table_schemas()
    if not isinstance(schemas, Mapping):
        raise PreparationError("TsFile reader returned an invalid table-schema mapping")
    if spec.model == "table":
        if not schemas:
            raise PreparationError(
                "TSFILE_MODEL_BLOCKER: spec declares table model but the reader found no "
                "table schemas; use an explicit tree-model spec if this is a legacy file"
            )
        assert spec.table_name is not None
        schema = schemas.get(spec.table_name)
        if schema is None:
            raise PreparationError(
                f"TSFILE_SCHEMA_BLOCKER: table {spec.table_name!r} is absent; "
                f"available={sorted(str(name) for name in schemas)}"
            )
        columns = _table_schema_columns(schema)
        for entity_column in spec.entity_columns:
            actual = columns.get(entity_column)
            if actual is None:
                raise PreparationError(
                    f"TSFILE_SCHEMA_BLOCKER: entity column {entity_column!r} is absent"
                )
            if actual[1] != "TAG":
                raise PreparationError(
                    f"TSFILE_SCHEMA_BLOCKER: entity column {entity_column!r} must be TAG, "
                    f"got category {actual[1]}"
                )
        for column in spec.value_columns:
            actual = columns.get(column.source_name)
            if actual is None:
                raise PreparationError(
                    f"TSFILE_SCHEMA_BLOCKER: value column {column.source_name!r} is absent"
                )
            if actual[1] != "FIELD":
                raise PreparationError(
                    f"TSFILE_SCHEMA_BLOCKER: value column {column.source_name!r} must be "
                    f"FIELD, got category {actual[1]}"
                )
            if actual[0] != column.tsfile_type:
                raise PreparationError(
                    f"TSFILE_SCHEMA_BLOCKER: value column {column.source_name!r} type is "
                    f"{actual[0]}, expected {column.tsfile_type}"
                )
    elif schemas:
        raise PreparationError(
            "TSFILE_MODEL_BLOCKER: spec declares tree model but table schemas are present; "
            "use a table-model spec"
        )


def _reader_schema_fingerprint(reader: Any) -> str:
    schemas = reader.get_all_table_schemas()
    if not isinstance(schemas, Mapping):
        raise PreparationError("TsFile reader returned an invalid table-schema mapping")
    basis = []
    for table_name in sorted(schemas, key=str):
        columns = _table_schema_columns(schemas[table_name])
        basis.append({
            "table_name": str(table_name),
            "columns": [
                {
                    "name": name,
                    "tsfile_type": columns[name][0],
                    "category": columns[name][1],
                }
                for name in sorted(columns)
            ],
        })
    return _sha256_bytes(_canonical_json(basis))


def _verify_collection_sources(spec: TsFilePreparationSpec) -> list[dict[str, Any]]:
    """Verify every frozen collection member and its full table schema."""
    if not spec.is_collection:
        raise AssertionError("collection source verification used for a single TsFile")
    tsfile = _load_tsfile_module()
    records: list[dict[str, Any]] = []
    expected_schema: str | None = None
    for index, source in enumerate(spec.source_files):
        if not source.path.is_file():
            raise PreparationError(
                f"collection source[{index}] is unavailable: {source.path}"
            )
        _validate_tsfile_container(source.path)
        actual_sha256 = sha256_file(source.path)
        if actual_sha256 != source.expected_sha256:
            raise PreparationError(
                "collection source sha256 does not match the frozen spec: "
                f"repo_path={source.repo_path!r}, expected={source.expected_sha256}, "
                f"actual={actual_sha256}"
            )
        reader = None
        try:
            reader = tsfile.TsFileReader(str(source.path))
            _validate_model_and_schema(reader, spec)
            schema_sha256 = _reader_schema_fingerprint(reader)
        except PreparationError:
            raise
        except Exception as error:
            raise PreparationError(
                f"TSFILE_READER_BLOCKER: {type(error).__name__}: {error}"
            ) from error
        finally:
            if reader is not None:
                reader.close()
        if expected_schema is None:
            expected_schema = schema_sha256
        elif schema_sha256 != expected_schema:
            raise PreparationError(
                "TSFILE_SCHEMA_BLOCKER: collection schemas are not identical: "
                f"repo_path={source.repo_path!r}"
            )
        signature = _source_signature(source.path)
        records.append({
            "order": index,
            "repo_path": source.repo_path,
            "path": str(source.path),
            "bytes": signature[0],
            "sha256": actual_sha256,
            "schema_sha256": schema_sha256,
            "filesystem_signature": list(signature),
        })
    return records


def _result_layout(
    result: Any,
    *,
    expected_entity_names: Sequence[str],
    expected_value_columns: Sequence[TsFileValueColumn],
    tree_device: str | None,
) -> tuple[list[tuple[int, str]], list[tuple[int, TsFileValueColumn]]]:
    metadata = result.get_metadata()
    names = list(metadata.get_column_list())
    types = [_type_name(value) for value in metadata.get_data_type_list()]
    if not names or names[0].casefold() != "time" or types[0] != "INT64":
        raise PreparationError(
            "TSFILE_SCHEMA_BLOCKER: query result must start with an INT64 time column"
        )
    indexes = {name: index + 1 for index, name in enumerate(names)}
    type_by_name = dict(zip(names, types))

    entity_layout: list[tuple[int, str]] = []
    for name in expected_entity_names:
        if name not in indexes:
            raise PreparationError(
                f"TSFILE_SCHEMA_BLOCKER: entity column {name!r} is absent from query result"
            )
        entity_layout.append((indexes[name], type_by_name[name]))

    value_layout: list[tuple[int, TsFileValueColumn]] = []
    for column in expected_value_columns:
        result_name = (
            f"{tree_device}.{column.source_name}"
            if tree_device is not None
            else column.source_name
        )
        if result_name not in indexes:
            raise PreparationError(
                f"TSFILE_SCHEMA_BLOCKER: measurement {result_name!r} is absent from query result"
            )
        actual_type = type_by_name[result_name]
        if actual_type != column.tsfile_type:
            raise PreparationError(
                f"TSFILE_SCHEMA_BLOCKER: measurement {result_name!r} type is {actual_type}, "
                f"expected {column.tsfile_type}"
            )
        value_layout.append((indexes[result_name], column))
    return entity_layout, value_layout


def _iter_result_rows(
    result: Any,
    spec: TsFilePreparationSpec,
    *,
    entity_layout: Sequence[tuple[int, str]],
    value_layout: Sequence[tuple[int, TsFileValueColumn]],
    tree_device: str | None,
    row_limit: int | None,
) -> Iterable[tuple[tuple[str, ...], int, tuple[bytes | None, ...]]]:
    observed = 0
    while result.next():
        if row_limit is not None and observed >= row_limit:
            break
        observed += 1
        raw_timestamp = int(result.get_value_by_index(1))
        assert spec.timestamp.source_unit is not None
        timestamp = _convert_epoch_units(
            raw_timestamp,
            spec.timestamp.source_unit,
            spec.timestamp.output_unit,
        )
        if not INT64_MIN <= timestamp <= INT64_MAX:
            raise PreparationError(
                "converted TsFile timestamp is outside the signed INT64 output range"
            )
        if tree_device is not None:
            entity_key = (f"DEVICE:{json.dumps(tree_device, ensure_ascii=False)}",)
        elif entity_layout:
            components = []
            for index, type_name in entity_layout:
                if result.is_null_by_index(index):
                    raise PreparationError("table entity/tag values cannot be null")
                components.append(
                    _typed_entity_component(type_name, result.get_value_by_index(index))
                )
            entity_key = tuple(components)
        else:
            entity_key = ("__single_entity__",)
        packed_values: list[bytes | None] = []
        for index, column in value_layout:
            if result.is_null_by_index(index):
                packed_values.append(None)
            else:
                packed_values.append(_pack_value(result.get_value_by_index(index), column))
        yield entity_key, timestamp, tuple(packed_values)


def _iter_tsfile_file_rows(
    spec: TsFilePreparationSpec,
    source_path: Path,
    *,
    row_limit: int | None = None,
) -> Iterable[tuple[tuple[str, ...], int, tuple[bytes | None, ...]]]:
    _validate_tsfile_container(source_path)
    tsfile = _load_tsfile_module()
    reader = None
    try:
        reader = tsfile.TsFileReader(str(source_path))
        _validate_model_and_schema(reader, spec)
        if spec.model == "table":
            assert spec.table_name is not None
            query_columns = [
                *spec.entity_columns,
                *(column.source_name for column in spec.value_columns),
            ]
            result = reader.query_table(
                spec.table_name,
                query_columns,
                spec.query_start,
                spec.query_end,
            )
            try:
                entity_layout, value_layout = _result_layout(
                    result,
                    expected_entity_names=spec.entity_columns,
                    expected_value_columns=spec.value_columns,
                    tree_device=None,
                )
                yield from _iter_result_rows(
                    result,
                    spec,
                    entity_layout=entity_layout,
                    value_layout=value_layout,
                    tree_device=None,
                    row_limit=row_limit,
                )
            finally:
                result.close()
        else:
            for device in spec.devices:
                result = reader.query_timeseries(
                    device,
                    [column.source_name for column in spec.value_columns],
                    spec.query_start,
                    spec.query_end,
                )
                try:
                    _entity_layout, value_layout = _result_layout(
                        result,
                        expected_entity_names=(),
                        expected_value_columns=spec.value_columns,
                        tree_device=device,
                    )
                    rows = 0
                    for row in _iter_result_rows(
                        result,
                        spec,
                        entity_layout=(),
                        value_layout=value_layout,
                        tree_device=device,
                        row_limit=row_limit,
                    ):
                        rows += 1
                        yield row
                    if rows == 0:
                        raise PreparationError(
                            "TSFILE_DATA_BLOCKER: explicit tree device query returned no "
                            f"rows: {device!r}"
                        )
                finally:
                    result.close()
    except PreparationError:
        raise
    except Exception as error:
        raise PreparationError(
            f"TSFILE_READER_BLOCKER: {type(error).__name__}: {error}"
        ) from error
    finally:
        if reader is not None:
            reader.close()


def _encode_packed_values(values: Sequence[bytes | None]) -> bytes:
    encoded = bytearray()
    for value in values:
        if value is None:
            encoded.append(0)
        else:
            encoded.append(1)
            encoded.extend(value)
    return bytes(encoded)


def _decode_packed_values(
    payload: bytes,
    columns: Sequence[TsFileValueColumn],
) -> tuple[bytes | None, ...]:
    offset = 0
    values: list[bytes | None] = []
    for column in columns:
        if offset >= len(payload):
            raise PreparationError("corrupt internal collection row payload")
        present = payload[offset]
        offset += 1
        if present == 0:
            values.append(None)
            continue
        if present != 1:
            raise PreparationError("corrupt internal collection presence flag")
        width = column.numeric_column.width
        end = offset + width
        if end > len(payload):
            raise PreparationError("truncated internal collection row payload")
        values.append(payload[offset:end])
        offset = end
    if offset != len(payload):
        raise PreparationError("internal collection row payload has trailing bytes")
    return tuple(values)


def _iter_collection_rows(
    spec: TsFilePreparationSpec,
    *,
    row_limit: int | None,
    verify_sources: bool = True,
) -> Iterable[tuple[tuple[str, ...], int, tuple[bytes | None, ...]]]:
    """Spill all source rows, then emit the global (entity, Time) order."""
    if verify_sources:
        _verify_collection_sources(spec)
    with tempfile.TemporaryDirectory(prefix="weighted-grpo-tsfile-merge-") as directory:
        database_path = Path(directory) / "rows.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE rows ("
                "entity_key TEXT NOT NULL, timestamp INTEGER NOT NULL, "
                "source_ordinal INTEGER NOT NULL, row_ordinal INTEGER NOT NULL, "
                "packed_values BLOB NOT NULL)"
            )
            if spec.timestamp.duplicate_policy == "reject":
                connection.execute(
                    "CREATE UNIQUE INDEX unique_entity_time "
                    "ON rows(entity_key, timestamp)"
                )
            for source_ordinal, source in enumerate(spec.source_files):
                source_rows = 0
                for row_ordinal, (entity_key, timestamp, packed_values) in enumerate(
                    _iter_tsfile_file_rows(spec, source.path),
                ):
                    source_rows += 1
                    entity_text = _canonical_json(list(entity_key)).decode("utf-8")
                    try:
                        connection.execute(
                            "INSERT INTO rows VALUES (?, ?, ?, ?, ?)",
                            (
                                entity_text,
                                timestamp,
                                source_ordinal,
                                row_ordinal,
                                sqlite3.Binary(_encode_packed_values(packed_values)),
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise PreparationError(
                            "TSFILE_ORDER_BLOCKER: duplicate (entity, Time) key across "
                            "the frozen collection: "
                            f"entity={entity_key!r}, timestamp={timestamp}, "
                            f"repo_path={source.repo_path!r}"
                        ) from error
                if source_rows == 0:
                    raise PreparationError(
                        "TSFILE_DATA_BLOCKER: collection source returned no rows: "
                        f"{source.repo_path!r}"
                    )
            connection.commit()
            query = (
                "SELECT entity_key, timestamp, packed_values FROM rows "
                "ORDER BY entity_key, timestamp, source_ordinal, row_ordinal"
            )
            parameters: tuple[int, ...] = ()
            if row_limit is not None:
                query += " LIMIT ?"
                parameters = (row_limit,)
            for entity_text, timestamp, payload in connection.execute(query, parameters):
                decoded_entity = json.loads(entity_text)
                if not isinstance(decoded_entity, list) or any(
                    not isinstance(value, str) for value in decoded_entity
                ):
                    raise PreparationError("corrupt internal collection entity key")
                yield (
                    tuple(decoded_entity),
                    int(timestamp),
                    _decode_packed_values(bytes(payload), spec.value_columns),
                )
        finally:
            connection.close()


def iter_tsfile_rows(
    spec: TsFilePreparationSpec,
    *,
    row_limit: int | None = None,
) -> Iterable[tuple[tuple[str, ...], int, tuple[bytes | None, ...]]]:
    if row_limit is not None and row_limit <= 0:
        raise PreparationError("row_limit must be positive when set")
    if spec.is_collection:
        yield from _iter_collection_rows(spec, row_limit=row_limit)
    else:
        yield from _iter_tsfile_file_rows(
            spec,
            spec.source_path,
            row_limit=row_limit,
        )


def _cohort_binding(
    spec: TsFilePreparationSpec,
    cohort_manifest: Mapping[str, Any],
) -> dict[str, str]:
    dummy = PreparationContract(
        dataset_id=spec.dataset_id,
        domain=spec.domain,
        source_group=spec.source_group,
        upstream_revision=spec.upstream_revision,
        upstream_locator=spec.upstream_locator,
        expected_source_sha256=spec.source_identity_sha256,
        source_path=spec.source_path,
        encoding="binary",
        delimiter=",",
        entity_columns=spec.entity_columns,
        timestamp=spec.timestamp,
        value_columns=tuple(column.numeric_column for column in spec.value_columns),
        missing_tokens=frozenset(),
        dataset_byte_cap=spec.dataset_byte_cap,
        block_byte_cap=spec.block_byte_cap,
        spec_sha256=spec.spec_sha256,
    )
    return validate_contract_against_cohort(dummy, cohort_manifest)


def audit_tsfile_spec(
    spec: TsFilePreparationSpec,
    *,
    cohort_manifest: Mapping[str, Any],
    row_limit: int = 32,
) -> dict[str, Any]:
    if row_limit <= 0:
        raise PreparationError("row_limit must be positive")
    binding = _cohort_binding(spec, cohort_manifest)
    collection_records: list[dict[str, Any]] | None = None
    if spec.is_collection:
        collection_records = _verify_collection_sources(spec)
        source_sha256 = spec.source_identity_sha256
        row_iterator = _iter_collection_rows(
            spec,
            row_limit=row_limit,
            verify_sources=False,
        )
    else:
        source_sha256 = sha256_file(spec.source_path)
        if source_sha256 != spec.expected_source_sha256:
            raise PreparationError(
                "source sha256 does not match spec.source.expected_sha256: "
                f"expected={spec.expected_source_sha256}, actual={source_sha256}"
            )
        row_iterator = iter_tsfile_rows(spec, row_limit=row_limit)
    counts: dict[tuple[str, ...], int] = {}
    previous: dict[tuple[str, ...], int] = {}
    rows = 0
    for entity_key, timestamp, _packed in row_iterator:
        prior = previous.get(entity_key)
        if prior is not None and timestamp < prior:
            raise PreparationError(
                f"TSFILE_ORDER_BLOCKER: entity {entity_key!r} is not chronological"
            )
        if (
            prior is not None
            and timestamp == prior
            and spec.timestamp.duplicate_policy == "reject"
        ):
            raise PreparationError(
                f"TSFILE_ORDER_BLOCKER: entity {entity_key!r} has duplicate timestamps"
            )
        previous[entity_key] = timestamp
        counts[entity_key] = counts.get(entity_key, 0) + 1
        rows += 1
    if rows == 0:
        raise PreparationError(
            "TSFILE_DATA_BLOCKER: the explicit model/schema query returned no rows"
        )
    tsfile = _load_tsfile_module()
    source_record: dict[str, Any]
    if collection_records is None:
        source_record = {
            "path": str(spec.source_path),
            "bytes": spec.source_path.stat().st_size,
            "sha256": source_sha256,
            "upstream_revision": spec.upstream_revision,
        }
    else:
        source_record = {
            "format": "tsfile_collection",
            "files": collection_records,
            "bytes": sum(int(record["bytes"]) for record in collection_records),
            "sha256": source_sha256,
            "upstream_revision": spec.upstream_revision,
        }
    return {
        "engine_version": (
            COLLECTION_ENGINE_VERSION if spec.is_collection else ENGINE_VERSION
        ),
        "dataset_id": spec.dataset_id,
        "model": spec.model,
        "source": source_record,
        "cohort": binding,
        "tsfile_python_version": _tsfile_version(tsfile),
        "timestamp": {
            "source_unit": spec.timestamp.source_unit,
            "output_unit": spec.timestamp.output_unit,
            "timezone": spec.timestamp.timezone_name,
            "source_kind": spec.timestamp.source_kind,
            "semantic_kind": spec.timestamp_semantic_kind,
            **dict(spec.timestamp_semantic_flags),
            "precision_inferred": False,
        },
        "sampled_rows": rows,
        "sampled_entities": len(counts),
        "value_columns": [
            {
                "source_name": column.source_name,
                "output_name": column.output_name,
                "tsfile_type": column.tsfile_type,
                "dtype": column.dtype,
            }
            for column in spec.value_columns
        ],
        "status": "schema_and_bounded_rows_verified",
    }


def prepare_tsfile_dataset(
    spec: TsFilePreparationSpec,
    output_dir: Path,
    *,
    cohort_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _cohort_binding(spec, cohort_manifest)
    split_contract = _split_contract_for_binding(binding)
    cohort_role = str(split_contract["cohort_role"])
    collection_records: list[dict[str, Any]] | None = None
    if spec.is_collection:
        collection_records = _verify_collection_sources(spec)
        initial_signature: tuple[int, int, int] | None = None
        source_sha256 = spec.source_identity_sha256

        def rows() -> Iterable[tuple[tuple[str, ...], int, tuple[bytes | None, ...]]]:
            return _iter_collection_rows(
                spec,
                row_limit=None,
                verify_sources=False,
            )
    else:
        initial_signature = _source_signature(spec.source_path)
        source_sha256 = sha256_file(spec.source_path)
        if source_sha256 != spec.expected_source_sha256:
            raise PreparationError(
                "source sha256 does not match spec.source.expected_sha256: "
                f"expected={spec.expected_source_sha256}, actual={source_sha256}"
            )

        def rows() -> Iterable[tuple[tuple[str, ...], int, tuple[bytes | None, ...]]]:
            return iter_tsfile_rows(spec)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise PreparationError(f"output already exists; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / (
        f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    (temporary / "blocks").mkdir()

    try:
        counts: dict[tuple[str, ...], int] = {}
        previous_timestamp: dict[tuple[str, ...], int] = {}
        source_rows = 0
        for entity_key, timestamp, _packed in rows():
            source_rows += 1
            previous = previous_timestamp.get(entity_key)
            if previous is not None and timestamp < previous:
                raise PreparationError(
                    f"entity {entity_key!r} is not chronological: {timestamp} < {previous}"
                )
            if (
                previous is not None
                and timestamp == previous
                and spec.timestamp.duplicate_policy == "reject"
            ):
                raise PreparationError(
                    f"entity {entity_key!r} has duplicate timestamp {timestamp}"
                )
            previous_timestamp[entity_key] = timestamp
            counts[entity_key] = counts.get(entity_key, 0) + 1
        if not source_rows:
            raise PreparationError("explicit TsFile query returned no rows")
        if not spec.is_collection and _source_signature(spec.source_path) != initial_signature:
            raise PreparationError("source changed during the TsFile audit pass")

        row_budget = spec.dataset_byte_cap // spec.logical_row_bytes
        allocation = equal_prefix_allocation(counts, row_budget)
        if cohort_role == "train_validation":
            short_entities = {
                key: count
                for key, count in allocation.items()
                if count < MIN_SELECTED_ROWS_PER_ENTITY
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
        blocks: list[dict[str, Any]] = []
        selected_rows = 0
        for entity_key, timestamp, packed_values in rows():
            position = selected_positions[entity_key]
            if position >= allocation[entity_key]:
                continue
            split = (
                "test"
                if cohort_role == "test_candidate"
                else ("train" if position < train_counts[entity_key] else "validation")
            )
            selected_positions[entity_key] += 1
            selected_rows += 1
            values = (
                (spec.timestamp_column, struct.pack("<q", timestamp)),
                *tuple(
                    (column.numeric_column, packed)
                    for column, packed in zip(spec.value_columns, packed_values)
                ),
            )
            for column, packed in values:
                stream_key = (entity_key, split, column.output_name)
                builder = builders.get(stream_key)
                if builder is None:
                    builder = _BlockBuilder(
                        column=column,
                        entity_key=entity_key,
                        split=split,
                        block_index=0,
                        max_rows=max_rows_per_block(column.width, spec.block_byte_cap),
                    )
                    builders[stream_key] = builder
                if builder.full:
                    _flush_builder(builder, temporary, blocks)
                    builder = _BlockBuilder(
                        column=column,
                        entity_key=entity_key,
                        split=split,
                        block_index=builder.block_index + 1,
                        max_rows=max_rows_per_block(column.width, spec.block_byte_cap),
                    )
                    builders[stream_key] = builder
                builder.add(packed, timestamp)
        for builder in builders.values():
            _flush_builder(builder, temporary, blocks)
        if selected_positions != allocation:
            raise PreparationError("TsFile row counts changed between passes")
        if spec.is_collection:
            final_records = _verify_collection_sources(spec)
            if final_records != collection_records:
                raise PreparationError(
                    "collection source files changed during TsFile materialization"
                )
        else:
            if _source_signature(spec.source_path) != initial_signature:
                raise PreparationError("source changed during TsFile materialization")
            if sha256_file(spec.source_path) != source_sha256:
                raise PreparationError("source content changed during TsFile materialization")

        blocks.sort(
            key=lambda block: (
                block["entity_id"],
                block["split"],
                block["column"],
                block["block_index"],
            )
        )
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
            for index, column in enumerate(spec.numeric_columns)
        ]
        original_numeric_bytes = sum(
            block["original_numeric_bytes"] for block in blocks
        )
        tsfile = _load_tsfile_module()
        if collection_records is None:
            source_manifest: dict[str, Any] = {
                "path": str(spec.source_path),
                "format": "tsfile",
                "model": spec.model,
                "bytes": initial_signature[0] if initial_signature is not None else 0,
                "sha256": source_sha256,
                "upstream_revision": spec.upstream_revision,
                "upstream_locator": spec.upstream_locator,
                "tsfile_python_version": _tsfile_version(tsfile),
            }
        else:
            source_manifest = {
                "format": "tsfile_collection",
                "model": spec.model,
                "schema_policy": spec.schema_policy,
                "files": collection_records,
                "file_count": len(collection_records),
                "bytes": sum(int(record["bytes"]) for record in collection_records),
                "sha256": source_sha256,
                "identity_definition": (
                    "sha256(canonical ordered list of repo_path and expected sha256)"
                ),
                "upstream_revision": spec.upstream_revision,
                "upstream_locator": spec.upstream_locator,
                "tsfile_python_version": _tsfile_version(tsfile),
            }
        tsfile_contract: dict[str, Any] = {
            "model": spec.model,
            "table_name": spec.table_name,
            "devices": list(spec.devices),
            "query_start_inclusive": spec.query_start,
            "query_end_inclusive": spec.query_end,
            "native_types": [
                column.tsfile_type for column in spec.value_columns
            ],
        }
        if spec.is_collection:
            tsfile_contract.update({
                "source_format": spec.source_format,
                "source_file_count": len(spec.source_files),
                "schema_policy": spec.schema_policy,
            })
        selection: dict[str, Any] = {
            "source_rows": source_rows,
            "selected_rows": selected_rows,
            "entity_balance": "max_min_fair_contiguous_prefix_rows",
            "logical_row_bytes": spec.logical_row_bytes,
            "dataset_byte_cap": spec.dataset_byte_cap,
            "dataset_byte_cap_semantics": (
                "selected_fixed_width_numeric_slots_including_converted_timestamp_"
                "before_null_compaction"
            ),
            "selected_logical_slot_bytes": selected_rows * spec.logical_row_bytes,
            "block_byte_cap": spec.block_byte_cap,
            "block_byte_cap_semantics": (
                "value_payload_plus_missing_bitmap_per_column_block"
            ),
            "split": split_contract["partition"],
        }
        if spec.is_collection:
            selection["source_ordering"] = (
                "global_entity_timestamp_ascending_before_selection_and_split"
            )
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "preparation_engine": (
                COLLECTION_ENGINE_VERSION if spec.is_collection else ENGINE_VERSION
            ),
            "dataset_id": spec.dataset_id,
            "domain": spec.domain,
            "source_group": spec.source_group,
            "cohort": binding,
            "split_contract": split_contract,
            "source": source_manifest,
            "preparation_spec_sha256": spec.spec_sha256,
            "input_contract": {
                "entity_columns": list(spec.entity_columns),
                "numeric_columns": numeric_columns,
                "timestamp": {
                    "source_name": "time",
                    "source_kind": spec.timestamp.source_kind,
                    "source_unit": spec.timestamp.source_unit,
                    "output_name": spec.timestamp.output_name,
                    "output_unit": spec.timestamp.output_unit,
                    "timezone": spec.timestamp.timezone_name,
                    "formats": [
                        "integer_epoch_from_tsfile"
                        if spec.timestamp_semantic_kind is None
                        else "integer_coordinate_from_tsfile"
                    ],
                    "semantic_kind": spec.timestamp_semantic_kind,
                    **dict(spec.timestamp_semantic_flags),
                    "ambiguous_fold": None,
                    "duplicate_policy": spec.timestamp.duplicate_policy,
                    "invalid_policy": "fail_no_unit_or_timezone_inference",
                },
                "missing_tokens": [],
                "nan_policy": "retain_native_float_bits_distinct_from_tsfile_null",
                "float_bit_contract": {
                    "packing": (
                        "Apache API FLOAT/DOUBLE value immediately repacked as native-width "
                        "little-endian IEEE bytes without arithmetic normalization"
                    ),
                    "verified_cases": ["negative_zero", "quiet_nan_payload"],
                    "api_boundary": (
                        "the Python API exposes typed values rather than encoded cell bytes; "
                        "signaling-NaN source-bit identity is not independently observable"
                    ),
                },
                "tsfile": tsfile_contract,
            },
            "selection": selection,
            "entities": entities,
            "blocks": blocks,
            "totals": {
                "blocks": len(blocks),
                "logical_slot_bytes": sum(
                    block["logical_slot_bytes"] for block in blocks
                ),
                "original_numeric_bytes": original_numeric_bytes,
                "value_payload_bytes": sum(block["values_bytes"] for block in blocks),
                "missing_bitmap_bytes": sum(
                    block["missing_bitmap_bytes"] for block in blocks
                ),
                "prepared_block_bytes": sum(
                    block["prepared_block_bytes"] for block in blocks
                ),
            },
            "metric_contract": {
                "denominator_name": "B_raw",
                "denominator_bytes": original_numeric_bytes,
                "denominator_definition": (
                    "sum(valid numeric values times native TsFile width) after explicit "
                    "Unix timestamp unit conversion and before compression operators"
                ),
                "null_policy": (
                    "TsFile nulls have no numeric payload and are excluded from B_raw; "
                    "all required null bitmap bytes are included in complete compressed bytes"
                ),
                "nan_policy": (
                    "FLOAT/DOUBLE NaN is a present fixed-width value; IEEE bits exposed by "
                    "the Apache Python API are retained and included in B_raw"
                ),
                "compression_ratio": "complete_compressed_bytes / B_raw",
                "compression_time": "compression_elapsed_ns / B_raw",
                "decompression_time": "decompression_elapsed_ns / B_raw",
                "time_unit": "ns/byte",
                "optimization_direction": "all_three_lower_is_better",
                "zero_denominator_policy": "N/A",
            },
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--source", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--spec", type=Path, required=True)
    audit_parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_MANIFEST)
    audit_parser.add_argument("--row-limit", type=int, default=32)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--spec", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_MANIFEST)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--skip-source", action="store_true")
    verify_parser.add_argument(
        "--expected-role", choices=("train_validation", "test_candidate")
    )
    args = parser.parse_args(argv)

    if args.command == "inspect":
        print(json.dumps(inspect_tsfile_source(args.source), sort_keys=True))
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
    spec = load_tsfile_spec(args.spec)
    cohort = load_cohort_manifest(args.cohort)
    if args.command == "audit":
        print(json.dumps(
            audit_tsfile_spec(spec, cohort_manifest=cohort, row_limit=args.row_limit),
            sort_keys=True,
        ))
        return 0
    manifest = prepare_tsfile_dataset(
        spec,
        args.output,
        cohort_manifest=cohort,
    )
    print(json.dumps({
        "dataset_id": manifest["dataset_id"],
        "output": str(args.output.resolve()),
        "blocks": manifest["totals"]["blocks"],
        "original_numeric_bytes": manifest["totals"]["original_numeric_bytes"],
        "provenance_sha256": manifest["provenance_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
