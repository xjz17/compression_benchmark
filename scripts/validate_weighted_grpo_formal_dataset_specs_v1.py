#!/usr/bin/env python3
"""Validate frozen weighted-GRPO dataset admissions and formal TsFile specs.

This gate is deliberately source-static: it checks exact dataset identity,
revision, file fingerprints, types, timestamp semantics, and collection rules
without opening, downloading, or preparing source data.  Native TsFile schema
and row checks remain a separate materialization gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import rebuild_weighted_grpo_deep_space_v1 as deep_space


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    ROOT
    / "analysis_outputs/weight_conditioned_grpo_20260917_v1/data_audit/"
    "training_source_admissibility_v1.json"
)
DEFAULT_COHORT = ROOT / "scripts/weighted_grpo_timeseries_cohort_v1.json"
DEFAULT_REVISIONS = ROOT / "scripts/weighted_grpo_training_source_revisions_v1.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
NUMERIC_TYPES = {"INT32", "INT64", "FLOAT", "DOUBLE"}
SPEC_FORMATS = {
    "weighted-grpo-tsfile-preparation-v1",
    "weighted-grpo-tsfile-collection-preparation-v1",
}
CAMPAIGN_DATE = "2026-09-17"
REMEDIATION_SCHEMA = "weighted-grpo-blocked-source-remediation-audit-v1"


class SpecGateError(ValueError):
    """A frozen admission or formal spec violates the data contract."""


def _load(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpecGateError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise SpecGateError(f"{label} must be a JSON object: {path}")
    return value


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecGateError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_campaign_date(mapping: Mapping[str, Any], context: str) -> None:
    if mapping.get("created_date") != CAMPAIGN_DATE:
        raise SpecGateError(
            f"{context}.created_date must remain {CAMPAIGN_DATE} for the 20260917 campaign"
        )


def _unique_records(
    value: Any, *, key: str, context: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise SpecGateError(f"{context} must be a list of objects")
    result: dict[str, Mapping[str, Any]] = {}
    for item in value:
        identifier = _text(item, key, context)
        if identifier in result:
            raise SpecGateError(f"duplicate {context} {key}: {identifier}")
        result[identifier] = item
    return result


def _resolve_repo_path(text: str, context: str) -> Path:
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise SpecGateError(f"{context} must be a safe repository-relative path")
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise SpecGateError(f"{context} leaves the repository") from error
    if not resolved.is_file():
        raise SpecGateError(f"{context} does not exist: {resolved}")
    return resolved


def _validate_source_file(
    record: Mapping[str, Any], *, dataset_id: str, context: str
) -> dict[str, str]:
    repo_path = _text(record, "repo_path", context)
    path = _text(record, "path", context)
    digest = _text(record, "expected_sha256", context).casefold()
    if SHA256.fullmatch(digest) is None:
        raise SpecGateError(f"{context}.expected_sha256 must be 64 lowercase hex digits")
    expected_fragment = f"/training_raw/{dataset_id}/{repo_path}"
    if not path.endswith(expected_fragment):
        raise SpecGateError(
            f"{context}.path silently substitutes the frozen source layout; "
            f"expected suffix {expected_fragment!r}"
        )
    return {"repo_path": repo_path, "path": path, "expected_sha256": digest}


def validate_formal_spec(
    spec: Mapping[str, Any],
    *,
    cohort_entry: Mapping[str, Any],
    revision: str,
    allow_legacy_preflight_path: bool = False,
    expected_numeric_schema_sha256: str | None = None,
) -> dict[str, Any]:
    dataset_id = _text(spec, "dataset_id", "spec")
    _require_campaign_date(spec, f"spec[{dataset_id}]")
    if spec.get("format_version") not in SPEC_FORMATS:
        raise SpecGateError(f"unsupported formal spec format for {dataset_id}")
    for field in ("domain", "source_group"):
        if spec.get(field) != cohort_entry.get(field):
            raise SpecGateError(f"spec {field} substitution for {dataset_id}")
    if spec.get("model") != "table":
        raise SpecGateError(f"formal spec must declare table model: {dataset_id}")

    timestamp = spec.get("timestamp")
    if not isinstance(timestamp, Mapping):
        raise SpecGateError(f"spec timestamp is missing: {dataset_id}")
    semantic_kind = timestamp.get("semantic_kind")
    if semantic_kind == "calendar_date_utc_midnight_encoding":
        if timestamp.get("timezone") != "UTC":
            raise SpecGateError(f"calendar-date coordinate must use UTC: {dataset_id}")
        if timestamp.get("source_event_timezone_inferred") is not False:
            raise SpecGateError(
                f"calendar-date encoding must not infer an event timezone: {dataset_id}"
            )
        evidence = spec.get("evidence")
        if not isinstance(evidence, Mapping) or "not an inferred event timezone" not in str(
            evidence.get("timezone_boundary", "")
        ):
            raise SpecGateError(
                f"calendar-date encoding lacks its timezone boundary: {dataset_id}"
            )
    elif semantic_kind == "portuguese_civil_time_coordinate_encoding":
        if timestamp.get("timezone") != "Europe/Lisbon source civil-time labels":
            raise SpecGateError(f"Portuguese civil-time source semantics changed: {dataset_id}")
        if timestamp.get("source_event_timezone_inferred") is not False:
            raise SpecGateError(f"Portuguese source timezone must be documented, not inferred: {dataset_id}")
        if timestamp.get("event_utc_instant_claim") is not False:
            raise SpecGateError(f"Portuguese civil coordinate must not claim exact UTC instants: {dataset_id}")
        if timestamp.get("event_time_mapping_is_injective") is not False:
            raise SpecGateError(f"Portuguese DST aggregation must remain non-injective: {dataset_id}")
    elif semantic_kind == "helsinki_civil_time_coordinate_encoding":
        if timestamp.get("timezone") != "Europe/Helsinki source civil-time labels":
            raise SpecGateError(f"Finland civil-time source semantics changed: {dataset_id}")
        if timestamp.get("source_event_timezone_inferred") is not False:
            raise SpecGateError(f"Finland source timezone must be documented, not inferred: {dataset_id}")
        if timestamp.get("event_utc_instant_claim") is not False:
            raise SpecGateError(f"Finland civil coordinate must not claim exact UTC instants: {dataset_id}")
        if timestamp.get("event_time_mapping_is_injective") is not False:
            raise SpecGateError(f"Finland DST ambiguity must remain non-injective: {dataset_id}")
    elif semantic_kind is not None:
        raise SpecGateError(f"unsupported timestamp semantic kind: {dataset_id}")
    elif timestamp.get("timezone") != "UTC":
        raise SpecGateError(f"timestamp timezone must be UTC: {dataset_id}")
    if timestamp.get("source_unit") != timestamp.get("output_unit"):
        raise SpecGateError(f"timestamp source precision is not preserved: {dataset_id}")
    if timestamp.get("duplicate_policy") != "reject":
        raise SpecGateError(f"duplicate timestamps must be rejected: {dataset_id}")
    if not _text(timestamp, "output_name", f"spec[{dataset_id}].timestamp").startswith(
        "timestamp_unix_"
    ):
        raise SpecGateError(f"timestamp output is not an explicit Unix column: {dataset_id}")

    table = spec.get("table")
    if not isinstance(table, Mapping):
        raise SpecGateError(f"table contract is missing: {dataset_id}")
    entity_columns = table.get("entity_columns")
    if not isinstance(entity_columns, list) or any(
        not isinstance(value, str) or not value for value in entity_columns
    ):
        raise SpecGateError(f"entity_columns must be an explicit string list: {dataset_id}")
    columns = table.get("value_columns")
    if not isinstance(columns, list) or not columns:
        raise SpecGateError(f"value_columns must be non-empty: {dataset_id}")
    source_names: set[str] = set()
    output_names: set[str] = {str(timestamp["output_name"])}
    type_counts: dict[str, int] = {}
    for index, column in enumerate(columns):
        if not isinstance(column, Mapping):
            raise SpecGateError(f"value column {index} is not an object: {dataset_id}")
        source_name = _text(column, "source_name", f"spec[{dataset_id}].value_columns[{index}]")
        output_name = _text(column, "output_name", f"spec[{dataset_id}].value_columns[{index}]")
        tsfile_type = _text(column, "tsfile_type", f"spec[{dataset_id}].value_columns[{index}]")
        if tsfile_type not in NUMERIC_TYPES:
            raise SpecGateError(f"non-numeric or widened source type in {dataset_id}: {tsfile_type}")
        if source_name.casefold() == "time":
            raise SpecGateError(f"Time must be emitted by the timestamp contract: {dataset_id}")
        if source_name in source_names or output_name in output_names:
            raise SpecGateError(f"duplicate source/output numeric column in {dataset_id}")
        source_names.add(source_name)
        output_names.add(output_name)
        type_counts[tsfile_type] = type_counts.get(tsfile_type, 0) + 1
    numeric_schema = [
        {
            "source_name": column["source_name"],
            "output_name": column["output_name"],
            "tsfile_type": column["tsfile_type"],
        }
        for column in columns
    ]
    numeric_schema_sha256 = hashlib.sha256(
        json.dumps(
            numeric_schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if expected_numeric_schema_sha256 is not None:
        if SHA256.fullmatch(expected_numeric_schema_sha256) is None:
            raise SpecGateError(f"invalid frozen numeric schema fingerprint: {dataset_id}")
        if numeric_schema_sha256 != expected_numeric_schema_sha256:
            raise SpecGateError(f"numeric source type/schema substitution for {dataset_id}")

    source = spec.get("source")
    if not isinstance(source, Mapping):
        raise SpecGateError(f"source contract is missing: {dataset_id}")
    if source.get("upstream_revision") != revision:
        raise SpecGateError(f"upstream revision substitution for {dataset_id}")
    if source.get("upstream_locator") != cohort_entry.get("locator"):
        raise SpecGateError(f"upstream locator substitution for {dataset_id}")

    evidence = spec.get("evidence")
    if not isinstance(evidence, Mapping):
        raise SpecGateError(f"source evidence is missing: {dataset_id}")
    card_url = evidence.get("hf_readme", evidence.get("thulab_readme"))
    if not isinstance(card_url, str) or f"/blob/{revision}/README.md" not in card_url:
        raise SpecGateError(f"frozen source-card evidence is missing: {dataset_id}")
    for claim in ("timestamp_claim", "schema_claim"):
        _text(evidence, claim, f"spec[{dataset_id}].evidence")

    source_files: list[dict[str, str]] = []
    is_collection = spec.get("format_version") == "weighted-grpo-tsfile-collection-preparation-v1"
    if is_collection:
        if source.get("format") != "tsfile_collection":
            raise SpecGateError(f"collection source format mismatch: {dataset_id}")
        if source.get("schema_policy") != "identical_across_all_files":
            raise SpecGateError(f"collection schema policy is not strict: {dataset_id}")
        files = source.get("files")
        if not isinstance(files, list) or len(files) < 2:
            raise SpecGateError(f"collection must list every frozen file: {dataset_id}")
        for index, record in enumerate(files):
            if not isinstance(record, Mapping):
                raise SpecGateError(f"collection source file {index} is not an object")
            source_files.append(_validate_source_file(
                record, dataset_id=dataset_id, context=f"spec[{dataset_id}].source.files[{index}]"
            ))
    else:
        if source.get("format") != "tsfile":
            raise SpecGateError(f"single source format mismatch: {dataset_id}")
        single = {
            "repo_path": source.get("repo_path"),
            "path": source.get("path"),
            "expected_sha256": source.get("expected_sha256"),
        }
        if allow_legacy_preflight_path:
            path = _text(source, "path", f"spec[{dataset_id}].source")
            digest = _text(source, "expected_sha256", f"spec[{dataset_id}].source")
            if SHA256.fullmatch(digest) is None or f"/{dataset_id}/" not in path:
                raise SpecGateError(f"invalid legacy source identity for {dataset_id}")
            source_files.append({"repo_path": Path(path).name, "path": path, "expected_sha256": digest})
        else:
            source_files.append(_validate_source_file(
                single, dataset_id=dataset_id, context=f"spec[{dataset_id}].source"
            ))

    paths = [record["repo_path"] for record in source_files]
    hashes = [record["expected_sha256"] for record in source_files]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise SpecGateError(f"collection reuses a source path or fingerprint: {dataset_id}")

    row_contract = spec.get("row_contract")
    if row_contract is None and allow_legacy_preflight_path:
        row_contract = {
            "ordering_key": "Time",
            "emit_ordering_key_as_numeric_column": True,
            "ordering_key_output_dtype": "INT64",
            "sort_scope": "global_entity_timestamp_ascending",
            "duplicate_comparison_scope": "all_listed_source_files",
            "cross_file_entity_merge": True,
        }
    if not isinstance(row_contract, Mapping):
        raise SpecGateError(f"row_contract is missing: {dataset_id}")
    expected_rows = {
        "ordering_key": "Time",
        "emit_ordering_key_as_numeric_column": True,
        "ordering_key_output_dtype": "INT64",
        "sort_scope": "global_entity_timestamp_ascending",
        "duplicate_comparison_scope": "all_listed_source_files",
        "cross_file_entity_merge": True,
    }
    for key, expected in expected_rows.items():
        if row_contract.get(key) != expected:
            raise SpecGateError(f"row contract {key} mismatch for {dataset_id}")
    return {
        "dataset_id": dataset_id,
        "source_files": len(source_files),
        "value_columns": len(columns),
        "numeric_columns_including_time": len(columns) + 1,
        "native_type_counts": type_counts,
        "numeric_schema_sha256": numeric_schema_sha256,
        "collection": is_collection,
    }


def validate_calendar_audit_evidence(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    audit_path_text: str,
) -> dict[str, Any]:
    """Bind a calendar-date admission to replayable source and TsFile evidence."""
    dataset_id = _text(spec, "dataset_id", "spec")
    if record.get("dataset_id") != dataset_id:
        raise SpecGateError(f"calendar audit dataset substitution for {dataset_id}")
    if record.get("status") != "resolved_calendar_date_encoding":
        raise SpecGateError(f"calendar audit did not resolve {dataset_id}")
    if record.get("timestamp_semantics") != "calendar_date_not_wall_clock_instant":
        raise SpecGateError(f"calendar audit semantics changed for {dataset_id}")
    if record.get("exact_tsfile_time_match") is not True or record.get("unique_ascending") is not True:
        raise SpecGateError(f"calendar audit lacks exact ordered Time equality for {dataset_id}")
    if "not inferred" not in str(record.get("timezone_claim", "")):
        raise SpecGateError(f"calendar audit infers an event timezone for {dataset_id}")
    row_count = record.get("row_count")
    if not isinstance(row_count, int) or row_count <= 0:
        raise SpecGateError(f"calendar audit row count is invalid for {dataset_id}")

    source = spec.get("source")
    if not isinstance(source, Mapping):
        raise SpecGateError(f"source contract is missing: {dataset_id}")
    if record.get("frozen_tsfile_sha256") != source.get("expected_sha256"):
        raise SpecGateError(f"calendar audit TsFile fingerprint mismatch for {dataset_id}")

    columns = spec.get("table", {}).get("value_columns", [])
    native_types = {
        str(column.get("source_name")): str(column.get("tsfile_type"))
        for column in columns
        if isinstance(column, Mapping)
    }
    if record.get("native_field_types") != native_types:
        raise SpecGateError(f"calendar audit native type evidence mismatch for {dataset_id}")

    evidence = spec.get("evidence")
    if not isinstance(evidence, Mapping):
        raise SpecGateError(f"calendar source evidence is missing for {dataset_id}")
    if evidence.get("audit_path") != audit_path_text:
        raise SpecGateError(f"calendar spec points to a different audit for {dataset_id}")
    source_csv_sha256 = _text(
        evidence, "source_csv_sha256", f"spec[{dataset_id}].evidence"
    ).casefold()
    if SHA256.fullmatch(source_csv_sha256) is None:
        raise SpecGateError(f"calendar source CSV fingerprint is invalid for {dataset_id}")
    if record.get("source_csv_sha256") != source_csv_sha256:
        raise SpecGateError(f"calendar source CSV fingerprint mismatch for {dataset_id}")
    source_revision = _text(record, "source_revision", f"audit[{dataset_id}]")
    source_repo_path = _text(record, "source_repo_path", f"audit[{dataset_id}]")
    source_csv_url = _text(evidence, "source_csv", f"spec[{dataset_id}].evidence")
    if f"/blob/{source_revision}/" not in source_csv_url or not source_csv_url.endswith(
        source_repo_path
    ):
        raise SpecGateError(f"calendar upstream source identity mismatch for {dataset_id}")
    return {
        "row_count": row_count,
        "source_csv_sha256": source_csv_sha256,
        "frozen_tsfile_sha256": str(record["frozen_tsfile_sha256"]),
    }


def validate_portuguese_civil_audit_evidence(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    audit_path_text: str,
) -> dict[str, Any]:
    dataset_id = _text(spec, "dataset_id", "spec")
    if record.get("dataset_id") != dataset_id:
        raise SpecGateError(f"Portuguese civil audit dataset substitution for {dataset_id}")
    if record.get("status") != "resolved_portuguese_civil_coordinate_encoding":
        raise SpecGateError(f"Portuguese civil-time audit did not resolve {dataset_id}")
    if record.get("event_utc_instant_claim") is not False:
        raise SpecGateError(f"Portuguese civil-time audit claims exact UTC instants: {dataset_id}")
    if record.get("event_time_mapping_is_injective") is not False:
        raise SpecGateError(f"Portuguese DST mapping was silently made injective: {dataset_id}")
    if record.get("rows") != 140256 or record.get("numeric_fields") != 370:
        raise SpecGateError(f"Portuguese civil-grid shape changed for {dataset_id}")
    if record.get("civil_step_ms") != 900000:
        raise SpecGateError(f"Portuguese civil-grid cadence changed for {dataset_id}")
    source = spec.get("source")
    if not isinstance(source, Mapping) or record.get("frozen_tsfile_sha256") != source.get(
        "expected_sha256"
    ):
        raise SpecGateError(f"Portuguese civil audit TsFile fingerprint mismatch for {dataset_id}")
    columns = spec.get("table", {}).get("value_columns", [])
    if len(columns) != 370 or any(
        not isinstance(column, Mapping) or column.get("tsfile_type") != "FLOAT"
        for column in columns
    ):
        raise SpecGateError(f"Portuguese civil audit native type mismatch for {dataset_id}")
    evidence = spec.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("audit_path") != audit_path_text:
        raise SpecGateError(f"Portuguese civil spec points to a different audit for {dataset_id}")
    if "not an exact UTC event instant" not in str(evidence.get("timezone_boundary", "")):
        raise SpecGateError(f"Portuguese civil spec omits the event-time boundary: {dataset_id}")
    if "aggregated" not in str(evidence.get("dst_claim", "")):
        raise SpecGateError(f"Portuguese civil spec omits the DST aggregation caveat: {dataset_id}")
    return {
        "row_count": 140256,
        "numeric_fields": 370,
        "frozen_tsfile_sha256": str(record["frozen_tsfile_sha256"]),
        "exact_event_utc": False,
    }


def validate_helsinki_civil_audit_evidence(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    audit_path_text: str,
) -> dict[str, Any]:
    dataset_id = _text(spec, "dataset_id", "spec")
    if record.get("dataset_id") != dataset_id:
        raise SpecGateError(f"Finland civil audit dataset substitution for {dataset_id}")
    if record.get("status") != "resolved_helsinki_civil_coordinate_encoding":
        raise SpecGateError(f"Finland civil-time audit did not resolve {dataset_id}")
    if record.get("event_utc_instant_claim") is not False:
        raise SpecGateError(f"Finland civil-time audit claims exact UTC instants: {dataset_id}")
    if record.get("event_time_mapping_is_injective") is not False:
        raise SpecGateError(f"Finland DST mapping was silently made injective: {dataset_id}")
    if record.get("rows") != 35136 or record.get("numeric_fields") != 1:
        raise SpecGateError(f"Finland civil-grid shape changed for {dataset_id}")
    if record.get("civil_step_ms") != 900000:
        raise SpecGateError(f"Finland civil-grid cadence changed for {dataset_id}")
    if record.get("exact_tsfile_time_match") is not True:
        raise SpecGateError(f"Finland source labels no longer match frozen Time: {dataset_id}")
    dst_audit = record.get("dst_audit")
    if not isinstance(dst_audit, Mapping) or any(
        dst_audit.get(key) != expected
        for key, expected in {
            "nonexistent_spring_labels": 4,
            "ambiguous_autumn_labels": 4,
            "rows_on_2024_03_31": 96,
            "rows_on_2024_10_27": 96,
            "nonexistent_spring_values_are_zero": True,
        }.items()
    ):
        raise SpecGateError(f"Finland DST-day evidence changed for {dataset_id}")
    source = spec.get("source")
    if not isinstance(source, Mapping) or record.get("frozen_tsfile_sha256") != source.get(
        "expected_sha256"
    ):
        raise SpecGateError(f"Finland civil audit TsFile fingerprint mismatch for {dataset_id}")
    columns = spec.get("table", {}).get("value_columns", [])
    if columns != [
        {
            "source_name": "measurement",
            "output_name": "measurement",
            "tsfile_type": "FLOAT",
        }
    ]:
        raise SpecGateError(f"Finland civil audit native type mismatch for {dataset_id}")
    evidence = spec.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("audit_path") != audit_path_text:
        raise SpecGateError(f"Finland civil spec points to a different audit for {dataset_id}")
    source_csv_sha256 = _text(
        evidence, "source_csv_sha256", f"spec[{dataset_id}].evidence"
    ).casefold()
    if record.get("source_csv_sha256") != source_csv_sha256:
        raise SpecGateError(f"Finland source CSV fingerprint mismatch for {dataset_id}")
    source_csv_url = _text(evidence, "source_csv", f"spec[{dataset_id}].evidence")
    if (
        f"/blob/{record.get('source_revision')}/" not in source_csv_url
        or not source_csv_url.endswith(str(record.get("source_repo_path")))
    ):
        raise SpecGateError(f"Finland upstream source identity mismatch for {dataset_id}")
    if "not asserted to be an exact UTC event instant" not in str(
        evidence.get("timezone_boundary", "")
    ):
        raise SpecGateError(f"Finland civil spec omits the event-time boundary: {dataset_id}")
    return {
        "row_count": 35136,
        "source_csv_sha256": source_csv_sha256,
        "frozen_tsfile_sha256": str(record["frozen_tsfile_sha256"]),
        "exact_event_utc": False,
    }
def _linked_remediation_record(
    audit_path_text: str,
    dataset_id: str,
    cache: dict[str, dict[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    path = _resolve_repo_path(audit_path_text, f"audit_path[{dataset_id}]")
    key = str(path)
    if key not in cache:
        report = _load(path, "blocked-source remediation audit")
        if report.get("schema") != REMEDIATION_SCHEMA:
            raise SpecGateError(f"unexpected remediation audit schema: {path}")
        _require_campaign_date(report, "blocked-source remediation audit")
        records = _unique_records(
            report.get("records"), key="dataset_id", context="remediation record"
        )
        approved = {
            item_id
            for item_id, item in records.items()
            if str(item.get("status", "")).startswith("resolved_")
        }
        still_blocked = {
            item_id
            for item_id, item in records.items()
            if str(item.get("status", "")).startswith("still_blocked")
        }
        if set(report.get("approved_delta", [])) != approved:
            raise SpecGateError("remediation approved_delta does not match its records")
        if set(report.get("still_blocked", [])) != still_blocked:
            raise SpecGateError("remediation still_blocked does not match its records")
        cache[key] = records
    record = cache[key].get(dataset_id)
    if record is None:
        raise SpecGateError(f"linked remediation record is missing: {dataset_id}")
    return record


def validate_deep_space_formal_evidence(
    spec_path: Path,
    spec: Mapping[str, Any],
    *,
    cohort_entry: Mapping[str, Any],
    revision_entry: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact-upstream overlay and its completed rebuild receipts.

    The frozen THULab entry remains the logical cohort mirror only.  It is not
    an admissible byte source and cannot be repaired by changing its timestamp
    unit.  Formal bytes must descend from the exact juliensimon Parquet object.
    """
    dataset_id = deep_space.DATASET_ID
    _require_campaign_date(spec, f"spec[{dataset_id}]")
    if spec.get("format_version") != deep_space.SPEC_VERSION:
        raise SpecGateError("deep-space formal spec version changed")
    if spec.get("dataset_id") != dataset_id:
        raise SpecGateError("deep-space formal spec dataset substitution")
    if spec.get("domain") != cohort_entry.get("domain") or spec.get(
        "source_group"
    ) != cohort_entry.get("source_group"):
        raise SpecGateError("deep-space cohort assignment substitution")
    if cohort_entry.get("source_group") != "hf:juliensimon/deep-space-probes":
        raise SpecGateError("deep-space cohort source group changed")
    if cohort_entry.get("locator") != "https://huggingface.co/datasets/THULab/deep-space-probes":
        raise SpecGateError("deep-space rejected mirror identity changed")

    expected_revision_fields = {
        "repo_id": deep_space.UPSTREAM_REPO,
        "revision": deep_space.UPSTREAM_REVISION,
        "source_overlay": "exact_upstream_parquet_replaces_invalid_thulab_tsfile_mirror",
        "rejected_mirror_repo_id": "THULab/deep-space-probes",
        "rejected_mirror_revision": "2ce6a7c5b0e22c3b6aa19459e4fa16a433f7bc8c",
    }
    if any(revision_entry.get(key) != value for key, value in expected_revision_fields.items()):
        raise SpecGateError("deep-space exact-upstream revision overlay changed")

    source = spec.get("source")
    if not isinstance(source, Mapping):
        raise SpecGateError("deep-space source contract is missing")
    expected_source = {
        "format": "parquet",
        "repo_id": deep_space.UPSTREAM_REPO,
        "repo_path": deep_space.UPSTREAM_REPO_PATH,
        "upstream_revision": deep_space.UPSTREAM_REVISION,
        "upstream_locator": "https://huggingface.co/datasets/juliensimon/deep-space-probes",
        "cohort_mirror_locator": "https://huggingface.co/datasets/THULab/deep-space-probes",
        "expected_bytes": deep_space.SOURCE_BYTES,
        "expected_sha256": deep_space.SOURCE_SHA256,
        "git_oid": deep_space.SOURCE_GIT_OID,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise SpecGateError("deep-space exact-upstream source identity changed")
    execution_path = Path(str(source.get("path", "")))
    if (
        not execution_path.is_absolute()
        or execution_path.name != Path(deep_space.UPSTREAM_REPO_PATH).name
        or "THULab" in execution_path.parts
    ):
        raise SpecGateError(
            "deep-space execution path must name the exact-upstream Parquet, not the rejected mirror"
        )
    row_contract = spec.get("row_contract")
    if not isinstance(row_contract, Mapping) or any(
        row_contract.get(key) != value
        for key, value in {
            "source_rows": deep_space.EXPECTED_ROWS,
            "retain_all_source_rows": True,
            "retain_timestamp_only_rows": True,
            "expected_timestamp_only_rows": deep_space.EXPECTED_TIMESTAMP_ONLY_ROWS,
            "invalid_mirror_policy": "do not use or rescale the frozen THULab TsFiles",
        }.items()
    ):
        raise SpecGateError("deep-space full-row or invalid-mirror contract changed")

    audit_path_text = _text(admission, "source_audit_path", f"admission[{dataset_id}]")
    audit_path = _resolve_repo_path(audit_path_text, f"source_audit_path[{dataset_id}]")
    audit_sha256 = _text(
        admission, "source_audit_sha256", f"admission[{dataset_id}]"
    ).casefold()
    if _sha256(audit_path) != audit_sha256:
        raise SpecGateError("deep-space source-audit fingerprint mismatch")
    try:
        contract = deep_space.load_spec(spec_path)
        source_evidence = deep_space.validate_audit_receipt(contract, audit_path)
    except (deep_space.PreparationError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SpecGateError(f"deep-space source evidence is invalid: {error}") from error
    if admission.get("numeric_schema_sha256") != "f052ef6cd34ef786ff6ce8a6ec7dd924dc4cd1460e80dfd5dd850bfba4359a9f":
        raise SpecGateError("deep-space numeric schema fingerprint changed")

    receipt_path_text = _text(
        admission, "rebuild_receipt_path", f"admission[{dataset_id}]"
    )
    receipt_path = _resolve_repo_path(
        receipt_path_text, f"rebuild_receipt_path[{dataset_id}]"
    )
    receipt_sha256 = _text(
        admission, "rebuild_receipt_sha256", f"admission[{dataset_id}]"
    ).casefold()
    if _sha256(receipt_path) != receipt_sha256:
        raise SpecGateError("deep-space rebuild-receipt fingerprint mismatch")
    receipt = _load(receipt_path, "deep-space rebuild receipt")
    if (
        receipt.get("schema") != "weighted-grpo-deep-space-rebuild-receipts-v1"
        or receipt.get("status") != "passed"
        or receipt.get("dataset_id") != dataset_id
    ):
        raise SpecGateError("deep-space rebuild receipt identity/status changed")
    _require_campaign_date(receipt, "deep-space rebuild receipt")
    if receipt.get("formal_spec") != {
        "path": str(spec_path.relative_to(ROOT)),
        "file_sha256": _sha256(spec_path),
        "contract_sha256": source_evidence["spec_sha256"],
    }:
        raise SpecGateError("deep-space rebuild receipt does not bind the formal spec")
    source_audit_receipt = receipt.get("source_audit")
    if not isinstance(source_audit_receipt, Mapping) or any(
        source_audit_receipt.get(key) != value
        for key, value in {
            "path": audit_path_text,
            "file_sha256": audit_sha256,
            "rows": deep_space.EXPECTED_ROWS,
            "timestamp_only_rows": deep_space.EXPECTED_TIMESTAMP_ONLY_ROWS,
            "numeric_schema_sha256": admission.get("numeric_schema_sha256"),
        }.items()
    ):
        raise SpecGateError("deep-space rebuild receipt does not bind the source audit")
    overlay = receipt.get("source_overlay")
    if not isinstance(overlay, Mapping) or any(
        overlay.get(key) != value
        for key, value in {
            "repo_id": deep_space.UPSTREAM_REPO,
            "revision": deep_space.UPSTREAM_REVISION,
            "repo_path": deep_space.UPSTREAM_REPO_PATH,
            "bytes": deep_space.SOURCE_BYTES,
            "sha256": deep_space.SOURCE_SHA256,
            "git_oid": deep_space.SOURCE_GIT_OID,
        }.items()
    ):
        raise SpecGateError("deep-space rebuild receipt source overlay changed")
    rejected = receipt.get("invalid_mirror_rejection")
    admission_rejected = admission.get("rejected_mirror")
    if (
        not isinstance(rejected, Mapping)
        or not isinstance(admission_rejected, Mapping)
        or rejected.get("repo_id") != "THULab/deep-space-probes"
        or rejected.get("revision") != expected_revision_fields["rejected_mirror_revision"]
        or admission_rejected.get("repo_id") != rejected.get("repo_id")
        or admission_rejected.get("revision") != rejected.get("revision")
        or admission_rejected.get("policy") != "never_use_rescale_repair_or_admit"
    ):
        raise SpecGateError("deep-space invalid THULab mirror is not explicitly rejected")
    canonical = receipt.get("verified_local_canonical_rebuild")
    prepared = receipt.get("verified_local_prepared_artifact")
    if not isinstance(canonical, Mapping) or any(
        canonical.get(key) != value
        for key, value in {
            "blocks": 528,
            "rows": deep_space.EXPECTED_ROWS,
            "timestamp_only_rows": deep_space.EXPECTED_TIMESTAMP_ONLY_ROWS,
            "original_numeric_bytes": 148470360,
            "source_verified": True,
        }.items()
    ):
        raise SpecGateError("deep-space canonical rebuild receipt changed")
    if not isinstance(prepared, Mapping) or any(
        prepared.get(key) != value
        for key, value in {
            "blocks": 576,
            "source_rows": deep_space.EXPECTED_ROWS,
            "selected_rows": 699050,
            "original_numeric_bytes": 85082728,
            "prepared_block_bytes": 89277352,
            "source_verified": True,
            "split": "per_entity_chronological_80_20_before_blocks",
        }.items()
    ):
        raise SpecGateError("deep-space prepared-artifact receipt changed")
    for record, label in ((canonical, "canonical"), (prepared, "prepared")):
        for key in ("manifest_file_sha256", "provenance_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or SHA256.fullmatch(value) is None:
                raise SpecGateError(f"deep-space {label} {key} is invalid")
    return {
        "dataset_id": dataset_id,
        "source_files": 1,
        "value_columns": len(deep_space.VALUE_COLUMNS),
        "numeric_columns_including_time": len(deep_space.VALUE_COLUMNS) + 1,
        "native_type_counts": {"DOUBLE": len(deep_space.VALUE_COLUMNS)},
        "numeric_schema_sha256": admission["numeric_schema_sha256"],
        "collection": False,
        "source_overlay": f"{deep_space.UPSTREAM_REPO}@{deep_space.UPSTREAM_REVISION}",
        "source_audit_sha256": audit_sha256,
        "rebuild_receipt_sha256": receipt_sha256,
        "invalid_thulab_mirror_rejected": True,
        "canonical_rebuild_provenance_sha256": canonical["provenance_sha256"],
        "prepared_artifact_provenance_sha256": prepared["provenance_sha256"],
    }


def _validate_deep_space_prepared_manifest(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source")
    canonical = manifest.get("canonical_rebuild")
    selection = manifest.get("selection")
    timestamp = (manifest.get("input_contract") or {}).get("timestamp")
    totals = manifest.get("totals")
    if not all(isinstance(item, Mapping) for item in (source, canonical, selection, timestamp, totals)):
        raise SpecGateError("deep-space prepared artifact is missing its evidence chain")
    expected_source = {
        "repo_id": deep_space.UPSTREAM_REPO,
        "repo_path": deep_space.UPSTREAM_REPO_PATH,
        "revision": deep_space.UPSTREAM_REVISION,
        "bytes": deep_space.SOURCE_BYTES,
        "sha256": deep_space.SOURCE_SHA256,
        "git_oid": deep_space.SOURCE_GIT_OID,
        "format": "parquet",
        "upstream_locator": "https://huggingface.co/datasets/juliensimon/deep-space-probes",
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise SpecGateError("deep-space prepared artifact does not descend from the overlay")
    if "THULab/deep-space-probes" in json.dumps(source, sort_keys=True):
        raise SpecGateError("deep-space prepared artifact references the rejected THULab mirror")
    if manifest.get("source_group") != "hf:juliensimon/deep-space-probes":
        raise SpecGateError("deep-space prepared source group changed")
    expected_contract_sha256 = deep_space.logical_spec_sha256(
        deep_space.formal_spec_document(
            Path("/__execution_local__/data/deep_space_probes.parquet")
        )
    )
    if manifest.get("preparation_spec_sha256") != expected_contract_sha256:
        raise SpecGateError("deep-space prepared formal-spec binding changed")
    if any(
        canonical.get(key) != value
        for key, value in {
            "rows": deep_space.EXPECTED_ROWS,
            "timestamp_only_rows": deep_space.EXPECTED_TIMESTAMP_ONLY_ROWS,
            "selection_occurs_after_full_row_rebuild": True,
        }.items()
    ):
        raise SpecGateError("deep-space prepared canonical rebuild binding changed")
    if not isinstance(canonical.get("provenance_sha256"), str) or SHA256.fullmatch(
        str(canonical.get("provenance_sha256"))
    ) is None:
        raise SpecGateError("deep-space canonical provenance is invalid")
    if any(
        selection.get(key) != value
        for key, value in {
            "source_rows": deep_space.EXPECTED_ROWS,
            "selected_rows": 699050,
            "dataset_byte_cap": 268435456,
            "block_byte_cap": 1048576,
            "split": "per_entity_chronological_80_20_before_blocks",
        }.items()
    ):
        raise SpecGateError("deep-space prepared sampling/split contract changed")
    if any(
        timestamp.get(key) != value
        for key, value in {
            "source_kind": "datetime64[us]",
            "source_unit": "us",
            "output_name": "timestamp_unix_us",
            "output_unit": "us",
            "timezone": "UTC",
            "duplicate_policy": "reject",
            "invalid_policy": "fail_no_row_index_fallback",
        }.items()
    ):
        raise SpecGateError("deep-space prepared timestamp contract changed")
    if any(
        totals.get(key) != value
        for key, value in {
            "blocks": 576,
            "logical_slot_bytes": 268435200,
            "original_numeric_bytes": 85082728,
            "value_payload_bytes": 85082728,
            "missing_bitmap_bytes": 4194624,
            "prepared_block_bytes": 89277352,
        }.items()
    ):
        raise SpecGateError("deep-space prepared totals changed")


def validate_bundle(
    audit_path: Path = DEFAULT_AUDIT,
    cohort_path: Path = DEFAULT_COHORT,
    revisions_path: Path = DEFAULT_REVISIONS,
) -> dict[str, Any]:
    audit = _load(audit_path, "admissibility manifest")
    cohort = _load(cohort_path, "cohort manifest")
    revisions = _load(revisions_path, "source revision manifest")
    if audit.get("schema") != "weighted-grpo-training-source-admissibility-v1":
        raise SpecGateError("unexpected admissibility schema")
    _require_campaign_date(audit, "admissibility manifest")
    _require_campaign_date(cohort, "cohort manifest")
    _require_campaign_date(revisions, "source revision manifest")
    if audit.get("cohort_version") != cohort.get("version"):
        raise SpecGateError("admissibility cohort version mismatch")
    if audit.get("cohort_assignment_sha256") != cohort.get("assignment_sha256"):
        raise SpecGateError("admissibility cohort assignment mismatch")
    if audit.get("cohort_policy_sha256") != cohort.get("policy_sha256"):
        raise SpecGateError("admissibility cohort policy mismatch")
    if audit.get("source_revision_manifest_sha256") != _sha256(revisions_path):
        raise SpecGateError("source revision manifest fingerprint mismatch")

    cohort_train = _unique_records(
        [entry for entry in cohort.get("datasets", []) if entry.get("role") == "train_validation"],
        key="id",
        context="cohort train source",
    )
    revision_map = _unique_records(
        revisions.get("sources"), key="dataset_id", context="source revision"
    )
    sources = _unique_records(audit.get("sources"), key="dataset_id", context="admission source")
    if set(cohort_train) != set(revision_map) or set(cohort_train) != set(sources):
        raise SpecGateError(
            "fixed source substitution detected: cohort, revision, and admission IDs differ"
        )

    summary = audit.get("summary")
    if not isinstance(summary, Mapping):
        raise SpecGateError("admissibility summary is missing")
    new_ids = summary.get("new_formal_spec_dataset_ids")
    if not isinstance(new_ids, list) or len(new_ids) != len(set(new_ids)):
        raise SpecGateError("new formal spec IDs must be a unique list")
    expected_new_ids = {
        "electricity_load_2011_2014",
        "eea_pm25_forecasting",
        "bitcoin_price_timeseries",
        "scaphandre_cpu_usage",
        "geomagnetic_kp_index",
        "exchange_rate",
        "national_illness",
        "online_retail_2_uci",
        "finland_traffic",
        "deep-space-probes",
    }
    if set(new_ids) != expected_new_ids:
        raise SpecGateError("new formal spec bundle silently changed its ten sources")

    approved = 0
    blocked = 0
    approved_ids: list[str] = []
    blocked_ids: list[str] = []
    checked_specs: dict[str, dict[str, Any]] = {}
    remediation_cache: dict[str, dict[str, Mapping[str, Any]]] = {}
    linked_remediation_ids: set[str] = set()
    for dataset_id, admission in sources.items():
        cohort_entry = cohort_train[dataset_id]
        revision_entry = revision_map[dataset_id]
        for key in ("domain", "source_group"):
            if admission.get(key) != cohort_entry.get(key):
                raise SpecGateError(f"admission {key} substitution for {dataset_id}")
        revision = _text(revision_entry, "revision", f"revision[{dataset_id}]")
        if admission.get("revision") != revision:
            raise SpecGateError(f"admission revision substitution for {dataset_id}")
        status = admission.get("status")
        audit_path_text = admission.get("audit_path")
        remediation_record = None
        if audit_path_text is not None:
            if not isinstance(audit_path_text, str) or not audit_path_text:
                raise SpecGateError(f"audit_path must be a non-empty string: {dataset_id}")
            remediation_record = _linked_remediation_record(
                audit_path_text, dataset_id, remediation_cache
            )
            linked_remediation_ids.add(dataset_id)
        if status == "blocked":
            blocked += 1
            blocked_ids.append(dataset_id)
            blockers = admission.get("blockers")
            if not isinstance(blockers, list) or not blockers or "spec_path" in admission:
                raise SpecGateError(f"blocked source must keep explicit blockers and no spec: {dataset_id}")
            if remediation_record is not None:
                if not str(remediation_record.get("status", "")).startswith("still_blocked"):
                    raise SpecGateError(f"blocked source links a resolved audit: {dataset_id}")
                if remediation_record.get("frozen_revision") != revision:
                    raise SpecGateError(f"blocked-source audit revision mismatch: {dataset_id}")
                resolved = admission.get("resolved_evidence")
                if not isinstance(resolved, list) or not resolved:
                    raise SpecGateError(f"audited blocked source lacks resolved evidence: {dataset_id}")
            continue
        if status != "approved":
            raise SpecGateError(f"unsupported admission status for {dataset_id}: {status!r}")
        approved += 1
        approved_ids.append(dataset_id)
        evidence = admission.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise SpecGateError(f"approved source lacks evidence: {dataset_id}")
        spec_path_text = admission.get("spec_path")
        if dataset_id in expected_new_ids and not isinstance(spec_path_text, str):
            raise SpecGateError(f"newly approved source lacks its formal spec: {dataset_id}")
        if isinstance(spec_path_text, str):
            spec_path = _resolve_repo_path(spec_path_text, f"spec_path[{dataset_id}]")
            spec = _load(spec_path, f"formal spec {dataset_id}")
            if spec.get("dataset_id") != dataset_id:
                raise SpecGateError(f"formal spec dataset substitution for {dataset_id}")
            if dataset_id == deep_space.DATASET_ID:
                checked = validate_deep_space_formal_evidence(
                    spec_path,
                    spec,
                    cohort_entry=cohort_entry,
                    revision_entry=revision_entry,
                    admission=admission,
                )
            else:
                checked = validate_formal_spec(
                    spec,
                    cohort_entry=cohort_entry,
                    revision=revision,
                    allow_legacy_preflight_path=(dataset_id == "weatherAUS"),
                    expected_numeric_schema_sha256=admission.get("numeric_schema_sha256"),
                )
            timestamp = spec.get("timestamp", {})
            if timestamp.get("semantic_kind") == "calendar_date_utc_midnight_encoding":
                if remediation_record is None or not isinstance(audit_path_text, str):
                    raise SpecGateError(
                        f"calendar-date admission lacks linked source audit: {dataset_id}"
                    )
                checked["source_backed_calendar_audit"] = validate_calendar_audit_evidence(
                    spec,
                    remediation_record,
                    audit_path_text=audit_path_text,
                )
            elif timestamp.get("semantic_kind") == "portuguese_civil_time_coordinate_encoding":
                if remediation_record is None or not isinstance(audit_path_text, str):
                    raise SpecGateError(
                        f"Portuguese civil-time admission lacks linked source audit: {dataset_id}"
                    )
                checked["source_backed_portuguese_civil_audit"] = (
                    validate_portuguese_civil_audit_evidence(
                        spec,
                        remediation_record,
                        audit_path_text=audit_path_text,
                    )
                )
            elif timestamp.get("semantic_kind") == "helsinki_civil_time_coordinate_encoding":
                if remediation_record is None or not isinstance(audit_path_text, str):
                    raise SpecGateError(
                        f"Finland civil-time admission lacks linked source audit: {dataset_id}"
                    )
                checked["source_backed_helsinki_civil_audit"] = (
                    validate_helsinki_civil_audit_evidence(
                        spec,
                        remediation_record,
                        audit_path_text=audit_path_text,
                    )
                )
            checked_specs[dataset_id] = checked

    if approved != int(summary.get("timestamp_and_identity_approved", -1)):
        raise SpecGateError("approved source count mismatch")
    if blocked != int(summary.get("blocked", -1)):
        raise SpecGateError("blocked source count mismatch")
    if len(sources) != int(summary.get("fixed_train_validation_sources", -1)):
        raise SpecGateError("fixed source count mismatch")
    if set(checked_specs) & expected_new_ids != expected_new_ids:
        raise SpecGateError("not all ten new formal specs were checked")
    expected_linked_remediation_ids = {
        "exchange_rate",
        "national_illness",
        "online_retail_2_uci",
        "electricity_load_2011_2014",
        "finland_traffic",
        "gluco_tsfm_benchmark",
        "METR-LA",
    }
    if linked_remediation_ids != expected_linked_remediation_ids:
        raise SpecGateError("source-backed remediation links are incomplete or substituted")
    return {
        "schema": "weighted-grpo-formal-dataset-spec-gate-result-v1",
        "fixed_train_validation_sources": len(sources),
        "approved": approved,
        "blocked": blocked,
        "approved_dataset_ids": sorted(approved_ids),
        "blocked_dataset_ids": sorted(blocked_ids),
        "checked_formal_specs": checked_specs,
        "complete_cohort_trainable": approved == len(sources),
        "silent_substitution_checks": "passed",
    }


def validate_prepared_admissions(
    manifests: Sequence[Mapping[str, Any]],
    *,
    audit_path: Path = DEFAULT_AUDIT,
) -> dict[str, Any]:
    """Reject prepared artifacts for every source not admitted by the frozen audit.

    This check precedes prepared-byte/provenance verification.  A previously
    materialized directory cannot make a blocked source formally eligible.
    """
    audit = _load(audit_path, "admissibility manifest")
    sources = _unique_records(
        audit.get("sources"), key="dataset_id", context="admission source"
    )
    admitted: list[str] = []
    seen: set[str] = set()
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, Mapping):
            raise SpecGateError(f"prepared manifest {index} must be an object")
        dataset_id = _text(manifest, "dataset_id", f"prepared manifest[{index}]")
        if dataset_id in seen:
            raise SpecGateError(f"duplicate prepared dataset_id: {dataset_id}")
        seen.add(dataset_id)
        admission = sources.get(dataset_id)
        if admission is None:
            raise SpecGateError(
                f"prepared dataset is outside the fixed admission manifest: {dataset_id}"
            )
        if admission.get("status") != "approved":
            blockers = admission.get("blockers", [])
            raise SpecGateError(
                "FORMAL_DATASET_ADMISSION_BLOCKER: prepared artifact cannot admit "
                f"blocked dataset {dataset_id}; blockers={blockers}"
            )
        if dataset_id == deep_space.DATASET_ID:
            _validate_deep_space_prepared_manifest(manifest)
        admitted.append(dataset_id)
    return {
        "prepared_manifest_count": len(admitted),
        "admitted_dataset_ids": sorted(admitted),
        "blocked_artifacts_rejected_before_byte_verification": True,
    }


def _load_prepared_manifest(path: Path) -> Mapping[str, Any]:
    manifest_path = path / "dataset_manifest.json" if path.is_dir() else path
    return _load(manifest_path, "prepared dataset manifest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--revisions", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--prepared-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = validate_bundle(args.audit, args.cohort, args.revisions)
        if args.prepared_manifest:
            result["prepared_admission"] = validate_prepared_admissions(
                [_load_prepared_manifest(path) for path in args.prepared_manifest],
                audit_path=args.audit,
            )
    except SpecGateError as error:
        print(f"FORMAL_DATASET_SPEC_GATE_BLOCKER: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
