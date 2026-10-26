#!/usr/bin/env python3
"""Rebuild the fixed deep-space-probes source without the lossy TsFile mirror.

The frozen BenchmarkHost TsFiles are deliberately not inputs to this program.  The
only admitted source is ``juliensimon/deep-space-probes`` at commit
``ed64a3ba6e24baccb33574d3b36562ba5c1b7332``.  Its Parquet timestamp is a
documented UTC ``datetime64[us]``.  The rebuild stores that physical int64
value directly as Unix microseconds; it never guesses a unit and never
multiplies values from the invalid mirror.

The canonical output is a deterministic column/block package.  Every source
row has one timestamp slot and one slot in every numeric column, including
rows for which every non-timestamp value is null.  Float64 payloads are copied
from Arrow's physical value buffer so NaN payloads and signed zero are not
round-tripped through Python floats.  Null positions are retained in an
LSB-first bitmap and are distinct from present floating-point NaNs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request
import uuid
from typing import Any, Mapping, Sequence

from prepare_weighted_grpo_timeseries_v1 import (
    COHORT_ASSIGNMENT_SHA256,
    COHORT_POLICY_SHA256,
    COHORT_VERSION,
    DEFAULT_COHORT_MANIFEST,
    FORMAT_VERSION as PREPARED_FORMAT_VERSION,
    MAX_BLOCK_BYTES,
    MAX_DATASET_BYTES,
    TRAIN_DENOMINATOR,
    TRAIN_NUMERATOR,
    PreparationError,
    _canonical_json,
    _entity_digest,
    _safe_component,
    _sha256_bytes,
    equal_prefix_allocation,
    load_cohort_manifest,
    max_rows_per_block,
    sha256_file,
    verify_prepared_dataset,
)


FORMAT_VERSION = "weighted-grpo-deep-space-canonical-v1"
SPEC_VERSION = "weighted-grpo-deep-space-parquet-spec-v1"
AUDIT_VERSION = "weighted-grpo-deep-space-upstream-audit-v1"
DATASET_ID = "deep-space-probes"
UPSTREAM_REPO = "juliensimon/deep-space-probes"
UPSTREAM_REVISION = "ed64a3ba6e24baccb33574d3b36562ba5c1b7332"
UPSTREAM_REPO_PATH = "data/deep_space_probes.parquet"
UPSTREAM_README_PATH = "README.md"
SOURCE_BYTES = 31_839_367
SOURCE_SHA256 = "dd2b043ad74d1e049e7d1d6a85a51feb5a12244ef0cee6327e8cc6d80f0c76e8"
SOURCE_GIT_OID = "0e3da6bb2120cc92cdb65acae1de8c6c2d0ffb3b"
README_BYTES = 16_099
README_SHA256 = "1308dd25c5f37a05471899f8a7fecaa6bae8cd168c4a0be15a56aa9f0c8f732e"
README_GIT_OID = "cf8058719699bff61b64ea88a42689c01a62a28d"
EXPECTED_ROWS = 1_183_368
EXPECTED_TIMESTAMP_ONLY_ROWS = 38_167
EXPECTED_ENTITY_ROWS = {
    "pioneer_10": 210_384,
    "pioneer_11": 184_056,
    "voyager_1": 403_224,
    "voyager_2": 385_704,
}
EXPECTED_ENTITY_TIME_US = {
    "pioneer_10": (63_072_000_000_000, 820_450_800_000_000),
    "pioneer_11": (94_694_400_000_000, 788_914_800_000_000),
    "voyager_1": (220_924_800_000_000, 1_767_222_000_000_000),
    "voyager_2": (220_924_800_000_000, 1_609_455_600_000_000),
}
EXPECTED_ENTITY_ORDER = ("pioneer_10", "pioneer_11", "voyager_1", "voyager_2")
TIMESTAMP_SOURCE_NAME = "datetime"
TIMESTAMP_OUTPUT_NAME = "timestamp_unix_us"
VALUE_COLUMNS = (
    "heliocentric_distance_au",
    "hgi_latitude_deg",
    "hgi_longitude_deg",
    "b_magnitude_nt",
    "br_rtn_nt",
    "bt_rtn_nt",
    "bn_rtn_nt",
    "flow_speed_kms",
    "flow_elevation_deg",
    "flow_azimuth_deg",
    "proton_density_cm3",
    "proton_temperature_k",
    "b_magnitude_avg_nt",
    "flux_h_crs_12p8_17p9_mev",
    "flux_h_crs_130p3_154p2_mev",
    "flux_h_crs_132p8_154p9_mev",
    "flux_h_crs_154p2_171p3_mev",
    "flux_h_crs_154p9_174p9_mev",
    "flux_h_crs_171p3_193p6_mev",
    "flux_h_crs_174p9_187p7_mev",
    "flux_h_crs_17p9_30p0_mev",
    "flux_h_crs_187p7_220p5_mev",
    "flux_h_crs_193p6_208p2_mev",
    "flux_h_crs_208p2_245p7_mev",
    "flux_h_crs_220p5_270p1_mev",
    "flux_h_crs_245p7_272p3_mev",
    "flux_h_crs_270p1_346p0_mev",
    "flux_h_crs_272p3_344p0_mev",
    "flux_h_crs_30p0_48p0_mev",
    "flux_h_crs_344p0_478p6_mev",
    "flux_h_crs_3p0_4p6_mev",
    "flux_h_crs_478p6_598p7_mev",
    "flux_h_crs_48p0_56p0_mev",
    "flux_h_crs_4p6_6p2_mev",
    "flux_h_crs_6p2_7p7_mev",
    "flux_h_crs_74p5_83p7_mev",
    "flux_h_crs_75p9_82p6_mev",
    "flux_h_crs_7p7_12p8_mev",
    "flux_h_crt_120p7_227p3_mev",
    "flux_h_crt_30p55_56p47_mev",
    "flux_h_crt_3p45_5p15_mev",
    "flux_h_lecp_0p52_1p45_mev",
    "flux_h_lecp_0p57_1p78_mev",
    "flux_h_lecp_22p0_30p0_mev",
    "flux_h_lecp_22p0_31p0_mev",
    "flux_h_lecp_3p04_17p3_mev",
    "flux_h_lecp_3p40_17p6_mev",
)


@dataclass(frozen=True)
class DeepSpaceSpec:
    path: Path
    expected_sha256: str
    expected_bytes: int
    block_byte_cap: int
    spec_sha256: str


def formal_spec_document(source_path: Path) -> dict[str, Any]:
    """Return the complete frozen formal spec; only the storage location varies."""
    return {
        "format_version": SPEC_VERSION,
        "spec_id": "deep-space-probes-upstream-parquet-v1",
        "created_date": "2026-09-17",
        "purpose": "exact-upstream full-row canonical rebuild before train-validation sampling",
        "dataset_id": DATASET_ID,
        "domain": "science",
        "source_group": "hf:juliensimon/deep-space-probes",
        "model": "columnar_parquet",
        "block_byte_cap": MAX_BLOCK_BYTES,
        "source": {
            "format": "parquet",
            "path": str(source_path),
            "repo_id": UPSTREAM_REPO,
            "repo_path": UPSTREAM_REPO_PATH,
            "upstream_revision": UPSTREAM_REVISION,
            "upstream_locator": (
                "https://huggingface.co/datasets/juliensimon/deep-space-probes"
            ),
            "cohort_mirror_locator": (
                "https://huggingface.co/datasets/BenchmarkHost/deep-space-probes"
            ),
            "expected_bytes": SOURCE_BYTES,
            "expected_sha256": SOURCE_SHA256,
            "git_oid": SOURCE_GIT_OID,
            "license": "cc-by-4.0",
        },
        "evidence": {
            "hf_readme": (
                "https://huggingface.co/datasets/juliensimon/deep-space-probes/"
                f"blob/{UPSTREAM_REVISION}/README.md"
            ),
            "hf_readme_bytes": README_BYTES,
            "hf_readme_sha256": README_SHA256,
            "hf_readme_git_oid": README_GIT_OID,
            "source_card_timestamp_claim": "datetime64[us], UTC, hourly cadence",
            "source_card_row_claim": EXPECTED_ROWS,
            "source_card_entity_counts": EXPECTED_ENTITY_ROWS,
            "source_api_lfs_sha256": SOURCE_SHA256,
            "source_api_lfs_bytes": SOURCE_BYTES,
        },
        "timestamp": {
            "source_name": TIMESTAMP_SOURCE_NAME,
            "source_type": "timestamp[us]",
            "source_timezone": "UTC documented by frozen source card",
            "output_name": TIMESTAMP_OUTPUT_NAME,
            "output_dtype": "int64",
            "output_unit": "us",
            "conversion": (
                "physical datetime64[us] int64 -> Unix microseconds without scaling"
            ),
            "duplicate_policy": "reject_per_entity",
        },
        "row_contract": {
            "entity_column": "spacecraft",
            "source_order": "contiguous entity runs, then strict timestamp order",
            "source_rows": EXPECTED_ROWS,
            "retain_all_source_rows": True,
            "retain_timestamp_only_rows": True,
            "expected_timestamp_only_rows": EXPECTED_TIMESTAMP_ONLY_ROWS,
            "invalid_mirror_policy": "do not use or rescale the frozen BenchmarkHost TsFiles",
        },
        "table": {
            "entity_column": "spacecraft",
            "entity_type": "large_string",
            "value_columns": [
                {
                    "source_name": name,
                    "output_name": name,
                    "parquet_type": "double",
                    "dtype": "float64",
                }
                for name in VALUE_COLUMNS
            ],
        },
    }


def logical_spec_sha256(document: Mapping[str, Any]) -> str:
    """Fingerprint the frozen contract without binding a machine-local path.

    ``source.path`` is an execution locator, not source identity.  The exact
    repository revision, repository-relative path, byte count, SHA-256, git
    object id, schema, and timestamp contract remain in the fingerprint.  This
    lets an independently rebuilt artifact on the experiment server bind the
    same formal contract as a local audit without pretending the two machines
    share a filesystem path.
    """
    normalized = json.loads(json.dumps(document, ensure_ascii=False))
    source = normalized.get("source")
    if not isinstance(source, dict):
        raise PreparationError("deep-space formal spec source is missing")
    source["path"] = "<execution-local-exact-source-path>"
    return _sha256_bytes(_canonical_json(normalized))


def _load_arrow():
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as error:  # pragma: no cover - environment gate
        raise PreparationError(
            f"deep-space rebuild requires numpy and pyarrow: {error}"
        ) from error
    return np, pa, pc, pq


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def load_spec(path: Path, *, require_frozen_source: bool = True) -> DeepSpaceSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("format_version") != SPEC_VERSION:
        raise PreparationError(f"spec format_version must be {SPEC_VERSION!r}")
    for key, expected in (
        ("dataset_id", DATASET_ID),
        ("domain", "science"),
        ("source_group", "hf:juliensimon/deep-space-probes"),
    ):
        if raw.get(key) != expected:
            raise PreparationError(f"deep-space spec {key} must be {expected!r}")
    source = raw.get("source")
    if not isinstance(source, Mapping) or source.get("format") != "parquet":
        raise PreparationError("deep-space spec source must be Parquet")
    source_path = Path(_text(source, "path", "spec.source"))
    if not source_path.is_absolute():
        source_path = (path.resolve().parent / source_path).resolve()
    expected_sha256 = _text(source, "expected_sha256", "spec.source").casefold()
    expected_bytes = int(source.get("expected_bytes", -1))
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None or expected_bytes <= 0:
        raise PreparationError("deep-space source fingerprint is invalid")
    if require_frozen_source:
        required = {
            "repo_id": UPSTREAM_REPO,
            "repo_path": UPSTREAM_REPO_PATH,
            "upstream_revision": UPSTREAM_REVISION,
            "expected_sha256": SOURCE_SHA256,
            "expected_bytes": SOURCE_BYTES,
            "git_oid": SOURCE_GIT_OID,
            "license": "cc-by-4.0",
        }
        for key, expected in required.items():
            if source.get(key) != expected:
                raise PreparationError(
                    f"deep-space source {key} must remain {expected!r}"
                )
        timestamp = raw.get("timestamp")
        expected_timestamp = {
            "source_name": TIMESTAMP_SOURCE_NAME,
            "source_type": "timestamp[us]",
            "source_timezone": "UTC documented by frozen source card",
            "output_name": TIMESTAMP_OUTPUT_NAME,
            "output_dtype": "int64",
            "output_unit": "us",
            "conversion": "physical datetime64[us] int64 -> Unix microseconds without scaling",
            "duplicate_policy": "reject_per_entity",
        }
        if timestamp != expected_timestamp:
            raise PreparationError("deep-space timestamp contract changed")
        table = raw.get("table")
        if not isinstance(table, Mapping) or table.get("entity_column") != "spacecraft":
            raise PreparationError("deep-space entity contract changed")
        columns = table.get("value_columns")
        if not isinstance(columns, list):
            raise PreparationError("deep-space value_columns must be a list")
        expected_columns = [
            {"source_name": name, "output_name": name, "parquet_type": "double", "dtype": "float64"}
            for name in VALUE_COLUMNS
        ]
        if columns != expected_columns:
            raise PreparationError("deep-space numeric schema changed")
    block_byte_cap = int(raw.get("block_byte_cap", MAX_BLOCK_BYTES))
    if not 0 < block_byte_cap <= MAX_BLOCK_BYTES:
        raise PreparationError(f"block_byte_cap must be in 1..{MAX_BLOCK_BYTES}")
    max_rows_per_block(8, block_byte_cap)
    return DeepSpaceSpec(
        path=source_path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        block_byte_cap=block_byte_cap,
        spec_sha256=logical_spec_sha256(raw),
    )


def with_source_path(spec: DeepSpaceSpec, source_path: Path | None) -> DeepSpaceSpec:
    """Override only the local location; the frozen size/hash remain unchanged."""
    if source_path is None:
        return spec
    return DeepSpaceSpec(
        path=source_path.resolve(),
        expected_sha256=spec.expected_sha256,
        expected_bytes=spec.expected_bytes,
        block_byte_cap=spec.block_byte_cap,
        spec_sha256=spec.spec_sha256,
    )


def _source_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _validity_numpy(array, np):
    return array.is_valid().to_numpy(zero_copy_only=False).astype(np.bool_, copy=False)


def _physical_int64_numpy(array, np):
    if array.buffers()[1] is None:
        raise PreparationError("numeric Arrow array has no physical value buffer")
    return np.frombuffer(
        array.buffers()[1], dtype="<i8", count=len(array), offset=array.offset * 8
    )


def _physical_uint64_numpy(array, np):
    if array.buffers()[1] is None:
        raise PreparationError("numeric Arrow array has no physical value buffer")
    return np.frombuffer(
        array.buffers()[1], dtype="<u8", count=len(array), offset=array.offset * 8
    )


def _schema_summary(schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)}
        for field in schema
    ]


def audit_source(
    spec: DeepSpaceSpec,
    *,
    enforce_production_shape: bool = True,
) -> dict[str, Any]:
    """Audit all rows; a non-null measurement is never required to retain a row."""
    np, pa, _pc, pq = _load_arrow()
    source = spec.path.resolve()
    if not source.is_file():
        raise PreparationError(f"deep-space source is missing: {source}")
    signature = _source_signature(source)
    if signature[0] != spec.expected_bytes:
        raise PreparationError(
            f"deep-space source size mismatch: expected={spec.expected_bytes}, actual={signature[0]}"
        )
    source_sha256 = sha256_file(source)
    if source_sha256 != spec.expected_sha256:
        raise PreparationError(
            "deep-space source SHA-256 mismatch: "
            f"expected={spec.expected_sha256}, actual={source_sha256}"
        )
    parquet = pq.ParquetFile(source)
    expected_names = ["spacecraft", TIMESTAMP_SOURCE_NAME, *VALUE_COLUMNS]
    if parquet.schema_arrow.names != expected_names:
        raise PreparationError("deep-space Parquet column names/order changed")
    schema = parquet.schema_arrow
    if not pa.types.is_large_string(schema.field("spacecraft").type):
        raise PreparationError("deep-space spacecraft must remain large_string")
    timestamp_type = schema.field(TIMESTAMP_SOURCE_NAME).type
    if not pa.types.is_timestamp(timestamp_type) or timestamp_type.unit != "us" or timestamp_type.tz is not None:
        raise PreparationError("deep-space datetime must remain timezone-naive timestamp[us]")
    for name in VALUE_COLUMNS:
        if not pa.types.is_float64(schema.field(name).type):
            raise PreparationError(f"deep-space native numeric width changed: {name}")

    rows = 0
    entity_counts: dict[str, int] = {}
    first_timestamp: dict[str, int] = {}
    last_timestamp: dict[str, int] = {}
    previous_timestamp: dict[str, int] = {}
    entity_order: list[str] = []
    current_entity: str | None = None
    closed_entities: set[str] = set()
    null_counts = {name: 0 for name in VALUE_COLUMNS}
    timestamp_only_rows = 0
    timestamp_nulls = 0
    entity_nulls = 0
    selected_columns = ["spacecraft", TIMESTAMP_SOURCE_NAME, *VALUE_COLUMNS]
    for batch in parquet.iter_batches(batch_size=65_536, columns=selected_columns):
        spacecraft = batch.column(0).to_pylist()
        timestamp_array = batch.column(1)
        timestamp_valid = _validity_numpy(timestamp_array, np)
        timestamp_values = _physical_int64_numpy(timestamp_array, np)
        any_numeric = np.zeros(len(batch), dtype=np.bool_)
        for offset, name in enumerate(VALUE_COLUMNS, start=2):
            array = batch.column(offset)
            valid = _validity_numpy(array, np)
            null_counts[name] += int(len(array) - int(valid.sum()))
            any_numeric |= valid
        timestamp_only_rows += int((~any_numeric).sum())
        for index, entity in enumerate(spacecraft):
            rows += 1
            if entity is None:
                entity_nulls += 1
                continue
            if not timestamp_valid[index]:
                timestamp_nulls += 1
                continue
            entity = str(entity)
            timestamp = int(timestamp_values[index])
            if entity != current_entity:
                if current_entity is not None:
                    closed_entities.add(current_entity)
                if entity in closed_entities:
                    raise PreparationError(
                        f"deep-space entity rows are not contiguous: {entity}"
                    )
                entity_order.append(entity)
                current_entity = entity
            prior = previous_timestamp.get(entity)
            if prior is not None and timestamp <= prior:
                relation = "duplicate" if timestamp == prior else "regression"
                raise PreparationError(
                    f"deep-space {relation} timestamp for {entity}: {timestamp} <= {prior}"
                )
            previous_timestamp[entity] = timestamp
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
            first_timestamp.setdefault(entity, timestamp)
            last_timestamp[entity] = timestamp

    if _source_signature(source) != signature or sha256_file(source) != source_sha256:
        raise PreparationError("deep-space source changed during full-row audit")
    if timestamp_nulls or entity_nulls:
        raise PreparationError(
            f"deep-space key columns contain nulls: spacecraft={entity_nulls}, datetime={timestamp_nulls}"
        )
    if enforce_production_shape:
        if rows != EXPECTED_ROWS:
            raise PreparationError(f"deep-space row count changed: {rows} != {EXPECTED_ROWS}")
        if entity_counts != EXPECTED_ENTITY_ROWS:
            raise PreparationError(f"deep-space entity counts changed: {entity_counts}")
        if tuple(entity_order) != EXPECTED_ENTITY_ORDER:
            raise PreparationError(f"deep-space entity order changed: {entity_order}")
        observed_ranges = {
            key: (first_timestamp[key], last_timestamp[key]) for key in entity_counts
        }
        if observed_ranges != EXPECTED_ENTITY_TIME_US:
            raise PreparationError(f"deep-space timestamp ranges changed: {observed_ranges}")
        if timestamp_only_rows != EXPECTED_TIMESTAMP_ONLY_ROWS:
            raise PreparationError(
                "deep-space timestamp-only row count changed: "
                f"{timestamp_only_rows} != {EXPECTED_TIMESTAMP_ONLY_ROWS}"
            )
    return {
        "schema": AUDIT_VERSION,
        "created_date": "2026-09-17",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "preparation_spec_sha256": spec.spec_sha256,
        "source": {
            "path": str(source),
            "repo_id": UPSTREAM_REPO,
            "repo_path": UPSTREAM_REPO_PATH,
            "revision": UPSTREAM_REVISION,
            "bytes": signature[0],
            "sha256": source_sha256,
            "git_oid": SOURCE_GIT_OID,
            "license": "cc-by-4.0",
        },
        "parquet": {
            "rows": rows,
            "row_groups": parquet.metadata.num_row_groups,
            "schema": _schema_summary(schema),
        },
        "timestamp": {
            "source_type": "timestamp[us]",
            "source_timezone": "UTC documented by frozen source card",
            "output_name": TIMESTAMP_OUTPUT_NAME,
            "output_dtype": "int64",
            "output_unit": "us",
            "conversion": "physical datetime64[us] int64 -> Unix microseconds without scaling",
            "null_count": timestamp_nulls,
            "strictly_increasing_per_entity": True,
        },
        "entities": [
            {
                "spacecraft": entity,
                "rows": entity_counts[entity],
                "first_timestamp_unix_us": first_timestamp[entity],
                "last_timestamp_unix_us": last_timestamp[entity],
            }
            for entity in entity_order
        ],
        "numeric_columns": {
            name: {
                "dtype": "float64",
                "width_bytes": 8,
                "null_count": null_counts[name],
                "valid_count": rows - null_counts[name],
            }
            for name in VALUE_COLUMNS
        },
        "numeric_schema_sha256": _sha256_bytes(_canonical_json([
            {
                "source_name": name,
                "output_name": name,
                "dtype": "float64",
                "width_bytes": 8,
            }
            for name in VALUE_COLUMNS
        ])),
        "timestamp_only_rows": timestamp_only_rows,
        "row_preservation": {
            "all_source_rows_audited": True,
            "rows_are_not_filtered_by_measurement_presence": True,
            "timestamp_only_rows_are_retained": True,
        },
        "invalid_mirror_policy": (
            "frozen BenchmarkHost TsFiles are excluded; their Time values are never scaled or repaired"
        ),
    }


def validate_audit_receipt(
    spec: DeepSpaceSpec,
    audit_path: Path,
) -> dict[str, Any]:
    """Bind a persisted audit receipt to the exact source/spec contract."""
    raw = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PreparationError("deep-space audit receipt must be an object")
    required_top = {
        "schema": AUDIT_VERSION,
        "created_date": "2026-09-17",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "preparation_spec_sha256": spec.spec_sha256,
        "timestamp_only_rows": EXPECTED_TIMESTAMP_ONLY_ROWS,
    }
    for key, expected in required_top.items():
        if raw.get(key) != expected:
            raise PreparationError(f"deep-space audit receipt {key} changed")
    source = raw.get("source")
    if not isinstance(source, Mapping) or any(
        source.get(key) != expected
        for key, expected in {
            "repo_id": UPSTREAM_REPO,
            "repo_path": UPSTREAM_REPO_PATH,
            "revision": UPSTREAM_REVISION,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "git_oid": SOURCE_GIT_OID,
            "license": "cc-by-4.0",
        }.items()
    ):
        raise PreparationError("deep-space audit source identity changed")
    parquet = raw.get("parquet")
    if not isinstance(parquet, Mapping) or parquet.get("rows") != EXPECTED_ROWS:
        raise PreparationError("deep-space audit full-row count changed")
    timestamp = raw.get("timestamp")
    if not isinstance(timestamp, Mapping) or any(
        timestamp.get(key) != expected
        for key, expected in {
            "source_type": "timestamp[us]",
            "output_dtype": "int64",
            "output_unit": "us",
            "null_count": 0,
            "strictly_increasing_per_entity": True,
        }.items()
    ):
        raise PreparationError("deep-space audit timestamp contract changed")
    entities = raw.get("entities")
    if not isinstance(entities, list):
        raise PreparationError("deep-space audit entity evidence is missing")
    counts = {
        str(item.get("spacecraft")): int(item.get("rows", -1))
        for item in entities
        if isinstance(item, Mapping)
    }
    ranges = {
        str(item.get("spacecraft")): (
            int(item.get("first_timestamp_unix_us", -1)),
            int(item.get("last_timestamp_unix_us", -1)),
        )
        for item in entities
        if isinstance(item, Mapping)
    }
    if counts != EXPECTED_ENTITY_ROWS or ranges != EXPECTED_ENTITY_TIME_US:
        raise PreparationError("deep-space audit entity shape/ranges changed")
    numeric = raw.get("numeric_columns")
    if not isinstance(numeric, Mapping) or set(numeric) != set(VALUE_COLUMNS):
        raise PreparationError("deep-space audit numeric schema changed")
    for name, record in numeric.items():
        if (
            not isinstance(record, Mapping)
            or record.get("dtype") != "float64"
            or record.get("width_bytes") != 8
            or int(record.get("null_count", -1))
            + int(record.get("valid_count", -1)) != EXPECTED_ROWS
        ):
            raise PreparationError(f"deep-space audit width/null evidence changed: {name}")
    expected_schema_sha256 = _sha256_bytes(_canonical_json([
        {
            "source_name": name,
            "output_name": name,
            "dtype": "float64",
            "width_bytes": 8,
        }
        for name in VALUE_COLUMNS
    ]))
    if raw.get("numeric_schema_sha256") != expected_schema_sha256:
        raise PreparationError("deep-space audit numeric schema fingerprint changed")
    preservation = raw.get("row_preservation")
    if not isinstance(preservation, Mapping) or any(
        preservation.get(key) is not True
        for key in (
            "all_source_rows_audited",
            "rows_are_not_filtered_by_measurement_presence",
            "timestamp_only_rows_are_retained",
        )
    ):
        raise PreparationError("deep-space audit row-preservation proof changed")
    return {
        "schema": "weighted-grpo-deep-space-formal-evidence-gate-v1",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "rows": EXPECTED_ROWS,
        "timestamp_only_rows": EXPECTED_TIMESTAMP_ONLY_ROWS,
        "numeric_columns": len(VALUE_COLUMNS),
        "source_sha256": SOURCE_SHA256,
        "spec_sha256": spec.spec_sha256,
    }


class _StreamBuilder:
    def __init__(self, entity: str, column: str, role: str, block_index: int, max_rows: int):
        self.entity = entity
        self.column = column
        self.role = role
        self.block_index = block_index
        self.max_rows = max_rows
        self.values = bytearray()
        self.missing = bytearray()
        self.row_count = 0
        self.valid_count = 0
        self.first_timestamp: int | None = None
        self.last_timestamp: int | None = None

    @property
    def remaining(self) -> int:
        return self.max_rows - self.row_count

    @property
    def full(self) -> bool:
        return self.row_count == self.max_rows


def _append_missing_bits(destination: bytearray, prior_rows: int, missing, np) -> None:
    count = int(len(missing))
    if count == 0:
        return
    target_rows = prior_rows + count
    needed = (target_rows + 7) // 8
    if len(destination) < needed:
        destination.extend(b"\0" * (needed - len(destination)))
    positions = np.flatnonzero(missing)
    for position in positions.tolist():
        absolute = prior_rows + int(position)
        destination[absolute // 8] |= 1 << (absolute % 8)


def _append_array(builder: _StreamBuilder, array, timestamps, np) -> None:
    valid = _validity_numpy(array, np)
    missing = ~valid
    raw = _physical_uint64_numpy(array, np)
    selected = raw[valid].astype("<u8", copy=False).tobytes()
    builder.values.extend(selected)
    _append_missing_bits(builder.missing, builder.row_count, missing, np)
    builder.row_count += len(array)
    builder.valid_count += int(valid.sum())
    if len(timestamps):
        if builder.first_timestamp is None:
            builder.first_timestamp = int(timestamps[0])
        builder.last_timestamp = int(timestamps[-1])


def _append_timestamp(builder: _StreamBuilder, array, np) -> None:
    valid = _validity_numpy(array, np)
    if not bool(valid.all()):
        raise PreparationError("canonical timestamp stream contains a null")
    values = _physical_int64_numpy(array, np)
    builder.values.extend(values.astype("<i8", copy=False).tobytes())
    _append_missing_bits(builder.missing, builder.row_count, ~valid, np)
    builder.row_count += len(array)
    builder.valid_count += len(array)
    if len(values):
        if builder.first_timestamp is None:
            builder.first_timestamp = int(values[0])
        builder.last_timestamp = int(values[-1])


def _write_builder(builder: _StreamBuilder, root: Path) -> dict[str, Any]:
    entity_id = _entity_digest((builder.entity,))
    stem = (
        f"source__e-{entity_id}__c-{_safe_component(builder.column)}"
        f"__b-{builder.block_index:06d}"
    )
    values_relative = Path("blocks") / f"{stem}.values.bin"
    bitmap_relative = Path("blocks") / f"{stem}.missing.bitmap"
    values_path = root / values_relative
    bitmap_path = root / bitmap_relative
    values_path.write_bytes(builder.values)
    bitmap_path.write_bytes(builder.missing)
    return {
        "block_id": stem,
        "entity_id": entity_id,
        "entity_value": builder.entity,
        "column": builder.column,
        "role": builder.role,
        "dtype": "int64" if builder.role == "timestamp" else "float64",
        "byte_order": "little",
        "block_index": builder.block_index,
        "row_count": builder.row_count,
        "valid_count": builder.valid_count,
        "missing_count": builder.row_count - builder.valid_count,
        "logical_slot_bytes": builder.row_count * 8,
        "original_numeric_bytes": builder.valid_count * 8,
        "first_timestamp_unix_us": builder.first_timestamp,
        "last_timestamp_unix_us": builder.last_timestamp,
        "values_path": values_relative.as_posix(),
        "values_bytes": len(builder.values),
        "values_sha256": _sha256_bytes(bytes(builder.values)),
        "missing_bitmap_path": bitmap_relative.as_posix(),
        "missing_bitmap_bytes": len(builder.missing),
        "missing_bitmap_sha256": _sha256_bytes(bytes(builder.missing)),
        "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
        "canonical_block_bytes": len(builder.values) + len(builder.missing),
    }


def rebuild_canonical(
    spec: DeepSpaceSpec,
    output_dir: Path,
    *,
    enforce_production_shape: bool = True,
) -> dict[str, Any]:
    """Write a full-row canonical package atomically."""
    np, _pa, _pc, pq = _load_arrow()
    audit = audit_source(spec, enforce_production_shape=enforce_production_shape)
    source = spec.path.resolve()
    signature = _source_signature(source)
    output = output_dir.resolve()
    if output.exists():
        raise PreparationError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    (temporary / "blocks").mkdir()
    parquet = pq.ParquetFile(source)
    max_rows = max_rows_per_block(8, spec.block_byte_cap)
    builders: dict[tuple[str, str], _StreamBuilder] = {}
    blocks: list[dict[str, Any]] = []

    def get_builder(entity: str, column: str, role: str) -> _StreamBuilder:
        key = (entity, column)
        builder = builders.get(key)
        if builder is None:
            builder = _StreamBuilder(entity, column, role, 0, max_rows)
            builders[key] = builder
        return builder

    def flush_full(entity: str, column: str, role: str) -> None:
        key = (entity, column)
        builder = builders[key]
        if not builder.full:
            return
        blocks.append(_write_builder(builder, temporary))
        builders[key] = _StreamBuilder(
            entity, column, role, builder.block_index + 1, max_rows
        )

    try:
        selected_columns = ["spacecraft", TIMESTAMP_SOURCE_NAME, *VALUE_COLUMNS]
        for batch in parquet.iter_batches(batch_size=65_536, columns=selected_columns):
            entities = batch.column(0).to_pylist()
            start = 0
            while start < len(batch):
                entity = str(entities[start])
                end = start + 1
                while end < len(batch) and entities[end] == entities[start]:
                    end += 1
                position = start
                while position < end:
                    timestamp_builder = get_builder(
                        entity, TIMESTAMP_OUTPUT_NAME, "timestamp"
                    )
                    take = min(end - position, timestamp_builder.remaining)
                    stream_start_rows = timestamp_builder.row_count
                    timestamp_slice = batch.column(1).slice(position, take)
                    _append_timestamp(timestamp_builder, timestamp_slice, np)
                    flush_full(entity, TIMESTAMP_OUTPUT_NAME, "timestamp")
                    timestamp_values = _physical_int64_numpy(timestamp_slice, np)
                    for offset, column in enumerate(VALUE_COLUMNS, start=2):
                        builder = get_builder(entity, column, "value")
                        if builder.row_count != stream_start_rows:
                            raise AssertionError("canonical stream block boundaries diverged")
                        _append_array(
                            builder,
                            batch.column(offset).slice(position, take),
                            timestamp_values,
                            np,
                        )
                        flush_full(entity, column, "value")
                    position += take
                start = end
        for builder in builders.values():
            if builder.row_count:
                blocks.append(_write_builder(builder, temporary))
        if _source_signature(source) != signature or sha256_file(source) != spec.expected_sha256:
            raise PreparationError("deep-space source changed during canonical rebuild")
        blocks.sort(
            key=lambda block: (
                block["entity_value"], block["column"], block["block_index"]
            )
        )
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "dataset_id": DATASET_ID,
            "source": audit["source"],
            "preparation_spec_sha256": spec.spec_sha256,
            "timestamp_contract": audit["timestamp"],
            "entity_column": "spacecraft",
            "entity_encoding": (
                "one explicit manifest entity value per stream; source row order retained"
            ),
            "numeric_columns": [
                {
                    "source_name": TIMESTAMP_SOURCE_NAME,
                    "output_name": TIMESTAMP_OUTPUT_NAME,
                    "dtype": "int64",
                    "width_bytes": 8,
                    "role": "timestamp",
                },
                *[
                    {
                        "source_name": name,
                        "output_name": name,
                        "dtype": "float64",
                        "width_bytes": 8,
                        "role": "value",
                    }
                    for name in VALUE_COLUMNS
                ],
            ],
            "row_contract": {
                "source_rows": audit["parquet"]["rows"],
                "canonical_rows": audit["parquet"]["rows"],
                "timestamp_only_rows": audit["timestamp_only_rows"],
                "selection": "none_all_source_rows_retained",
                "ordering": "source_entity_order_then_strict_timestamp_ascending",
                "timestamp_only_rows_are_retained": True,
            },
            "entities": audit["entities"],
            "block_byte_cap": spec.block_byte_cap,
            "block_byte_cap_semantics": "value_payload_plus_missing_bitmap",
            "blocks": blocks,
            "totals": {
                "blocks": len(blocks),
                "rows": audit["parquet"]["rows"],
                "timestamp_only_rows": audit["timestamp_only_rows"],
                "logical_slot_bytes": sum(block["logical_slot_bytes"] for block in blocks),
                "original_numeric_bytes": sum(block["original_numeric_bytes"] for block in blocks),
                "value_payload_bytes": sum(block["values_bytes"] for block in blocks),
                "missing_bitmap_bytes": sum(block["missing_bitmap_bytes"] for block in blocks),
                "canonical_block_bytes": sum(block["canonical_block_bytes"] for block in blocks),
            },
            "metric_boundary": {
                "canonical_package_is_not_a_compressed_result": True,
                "future_B_raw": (
                    "sum of present numeric values times native width; null positions excluded"
                ),
                "future_complete_compressed_bytes_must_include_null_bitmaps": True,
            },
            "invalid_mirror_policy": audit["invalid_mirror_policy"],
        }
        manifest["provenance_sha256"] = _sha256_bytes(_canonical_json(manifest))
        _atomic_json(temporary / "canonical_manifest.json", manifest)
        os.replace(temporary, output)
        verify_canonical(output, verify_source=True)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _safe_child(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparationError(f"unsafe canonical child path: {relative_text}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PreparationError(f"canonical child leaves output: {relative_text}") from error
    return path


def _unpack_missing(payload: bytes, rows: int, np):
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if len(bits) < rows:
        raise PreparationError("canonical missing bitmap is truncated")
    if bool(bits[rows:].any()):
        raise PreparationError("canonical missing bitmap has nonzero unused bits")
    return bits[:rows].astype(np.bool_, copy=False)


def verify_canonical(output_dir: Path, *, verify_source: bool = True) -> dict[str, Any]:
    np, _pa, _pc, _pq = _load_arrow()
    root = output_dir.resolve()
    path = root / "canonical_manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("format_version") != FORMAT_VERSION:
        raise PreparationError("unsupported deep-space canonical manifest")
    fingerprint_basis = {key: value for key, value in raw.items() if key != "provenance_sha256"}
    fingerprint = _sha256_bytes(_canonical_json(fingerprint_basis))
    if raw.get("provenance_sha256") != fingerprint:
        raise PreparationError("deep-space canonical provenance mismatch")
    source = raw.get("source")
    if not isinstance(source, Mapping):
        raise PreparationError("deep-space canonical source is missing")
    if verify_source:
        source_path = Path(str(source.get("path", "")))
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(source.get("bytes", -1))
            or sha256_file(source_path) != source.get("sha256")
        ):
            raise PreparationError("deep-space canonical source fingerprint changed")
    entities = raw.get("entities")
    if not isinstance(entities, list) or not entities:
        raise PreparationError("deep-space canonical entities are missing")
    row_count_by_entity = {
        str(item["spacecraft"]): int(item["rows"])
        for item in entities
        if isinstance(item, Mapping)
    }
    blocks = raw.get("blocks")
    if not isinstance(blocks, list):
        raise PreparationError("deep-space canonical blocks must be a list")
    by_stream: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    computed_totals = {
        "blocks": len(blocks),
        "rows": sum(row_count_by_entity.values()),
        "timestamp_only_rows": 0,
        "logical_slot_bytes": 0,
        "original_numeric_bytes": 0,
        "value_payload_bytes": 0,
        "missing_bitmap_bytes": 0,
        "canonical_block_bytes": 0,
    }
    seen_paths: set[str] = set()
    for block in blocks:
        if not isinstance(block, Mapping):
            raise PreparationError("deep-space canonical block is not an object")
        rows = int(block.get("row_count", -1))
        valid = int(block.get("valid_count", -1))
        missing = int(block.get("missing_count", -1))
        if rows < 0 or valid < 0 or missing != rows - valid:
            raise PreparationError("deep-space canonical block row counts are invalid")
        expected_value_bytes = valid * 8
        expected_bitmap_bytes = (rows + 7) // 8
        if int(block.get("values_bytes", -1)) != expected_value_bytes:
            raise PreparationError("deep-space canonical value byte count changed")
        if int(block.get("missing_bitmap_bytes", -1)) != expected_bitmap_bytes:
            raise PreparationError("deep-space canonical bitmap byte count changed")
        if int(block.get("canonical_block_bytes", -1)) > int(raw["block_byte_cap"]):
            raise PreparationError("deep-space canonical block exceeds its byte cap")
        payloads: dict[str, bytes] = {}
        for path_key, size_key, hash_key in (
            ("values_path", "values_bytes", "values_sha256"),
            ("missing_bitmap_path", "missing_bitmap_bytes", "missing_bitmap_sha256"),
        ):
            relative = str(block[path_key])
            if relative in seen_paths:
                raise PreparationError(f"canonical path reused: {relative}")
            seen_paths.add(relative)
            child = _safe_child(root, relative)
            if not child.is_file() or child.stat().st_size != int(block[size_key]):
                raise PreparationError(f"canonical block file changed: {child}")
            payload = child.read_bytes()
            if _sha256_bytes(payload) != block[hash_key]:
                raise PreparationError(f"canonical block hash changed: {child}")
            payloads[path_key] = payload
        missing_bits = _unpack_missing(payloads["missing_bitmap_path"], rows, np)
        if int(missing_bits.sum()) != missing:
            raise PreparationError("canonical bitmap missing count changed")
        entity = str(block.get("entity_value"))
        column = str(block.get("column"))
        by_stream.setdefault((entity, column), []).append(block)
        computed_totals["logical_slot_bytes"] += rows * 8
        computed_totals["original_numeric_bytes"] += expected_value_bytes
        computed_totals["value_payload_bytes"] += expected_value_bytes
        computed_totals["missing_bitmap_bytes"] += expected_bitmap_bytes
        computed_totals["canonical_block_bytes"] += expected_value_bytes + expected_bitmap_bytes

    expected_columns = [TIMESTAMP_OUTPUT_NAME, *VALUE_COLUMNS]
    timestamp_only_total = 0
    for entity, entity_rows in row_count_by_entity.items():
        all_missing = np.ones(entity_rows, dtype=np.bool_)
        for column in expected_columns:
            ordered = sorted(
                by_stream.get((entity, column), []), key=lambda item: int(item["block_index"])
            )
            if [int(item["block_index"]) for item in ordered] != list(range(len(ordered))):
                raise PreparationError(f"canonical block indexes are not contiguous: {entity}/{column}")
            if sum(int(item["row_count"]) for item in ordered) != entity_rows:
                raise PreparationError(f"canonical stream lost rows: {entity}/{column}")
            if column == TIMESTAMP_OUTPUT_NAME:
                values = []
                for item in ordered:
                    payload = _safe_child(root, str(item["values_path"])).read_bytes()
                    values.extend(np.frombuffer(payload, dtype="<i8").tolist())
                    if int(item["missing_count"]) != 0:
                        raise PreparationError("canonical timestamp stream contains missing values")
                if values != sorted(values) or len(values) != len(set(values)):
                    raise PreparationError(f"canonical timestamps are not strictly increasing: {entity}")
                continue
            offset = 0
            for item in ordered:
                rows = int(item["row_count"])
                bitmap = _safe_child(root, str(item["missing_bitmap_path"])).read_bytes()
                all_missing[offset:offset + rows] &= _unpack_missing(bitmap, rows, np)
                offset += rows
        timestamp_only_total += int(all_missing.sum())
    computed_totals["timestamp_only_rows"] = timestamp_only_total
    if raw.get("totals") != computed_totals:
        raise PreparationError(
            f"deep-space canonical totals changed: recorded={raw.get('totals')}, computed={computed_totals}"
        )
    if int(raw["row_contract"]["timestamp_only_rows"]) != timestamp_only_total:
        raise PreparationError("deep-space timestamp-only row proof changed")
    return {
        "schema": "weighted-grpo-deep-space-canonical-verification-v1",
        "status": "verified",
        "dataset_id": DATASET_ID,
        "rows": computed_totals["rows"],
        "timestamp_only_rows": timestamp_only_total,
        "blocks": computed_totals["blocks"],
        "original_numeric_bytes": computed_totals["original_numeric_bytes"],
        "provenance_sha256": fingerprint,
        "source_verified": verify_source,
    }


def _load_canonical_stream_prefix(
    canonical_root: Path,
    blocks: Sequence[Mapping[str, Any]],
    selected_rows: int,
    np,
) -> tuple[Any, Any]:
    """Return (missing bool slots, compact uint64 values) for one prefix."""
    missing_parts = []
    value_parts = []
    consumed = 0
    for block in sorted(blocks, key=lambda item: int(item["block_index"])):
        if consumed >= selected_rows:
            break
        block_rows = int(block["row_count"])
        take = min(block_rows, selected_rows - consumed)
        bitmap = _safe_child(
            canonical_root, str(block["missing_bitmap_path"])
        ).read_bytes()
        block_missing = _unpack_missing(bitmap, block_rows, np)
        raw = np.frombuffer(
            _safe_child(canonical_root, str(block["values_path"])).read_bytes(),
            dtype="<u8",
        )
        prefix_valid = int((~block_missing[:take]).sum())
        missing_parts.append(block_missing[:take].copy())
        value_parts.append(raw[:prefix_valid].copy())
        consumed += take
    if consumed != selected_rows:
        raise PreparationError(
            f"canonical stream is shorter than selected prefix: {consumed} != {selected_rows}"
        )
    missing = np.concatenate(missing_parts) if missing_parts else np.zeros(0, dtype=np.bool_)
    values = np.concatenate(value_parts) if value_parts else np.zeros(0, dtype="<u8")
    if int((~missing).sum()) != len(values):
        raise PreparationError("canonical prefix bitmap/value cardinality mismatch")
    return missing, values


def _write_prepared_stream(
    root: Path,
    *,
    entity: str,
    column: str,
    source_column: str,
    dtype: str,
    split: str,
    missing,
    compact_values,
    timestamps,
    block_byte_cap: int,
    np,
) -> list[dict[str, Any]]:
    if len(missing) != len(timestamps):
        raise PreparationError("prepared stream timestamp alignment changed")
    output: list[dict[str, Any]] = []
    entity_id = _entity_digest((entity,))
    max_rows = max_rows_per_block(8, block_byte_cap)
    valid_offset = 0
    for block_index, start in enumerate(range(0, len(missing), max_rows)):
        end = min(len(missing), start + max_rows)
        block_missing = missing[start:end]
        block_valid = int((~block_missing).sum())
        block_values = compact_values[valid_offset:valid_offset + block_valid]
        valid_offset += block_valid
        value_payload = block_values.astype("<u8", copy=False).tobytes()
        bitmap_payload = np.packbits(block_missing, bitorder="little").tobytes()
        stem = (
            f"{split}__e-{entity_id}__c-{_safe_component(column)}"
            f"__b-{block_index:06d}"
        )
        values_relative = Path("blocks") / f"{stem}.values.bin"
        bitmap_relative = Path("blocks") / f"{stem}.missing.bitmap"
        (root / values_relative).write_bytes(value_payload)
        (root / bitmap_relative).write_bytes(bitmap_payload)
        output.append({
            "block_id": stem,
            "entity_id": entity_id,
            "split": split,
            "column": column,
            "source_column": source_column,
            "dtype": dtype,
            "byte_order": "little",
            "block_index": block_index,
            "row_count": end - start,
            "valid_count": block_valid,
            "missing_count": end - start - block_valid,
            "logical_slot_bytes": (end - start) * 8,
            "original_numeric_bytes": block_valid * 8,
            "first_timestamp": int(timestamps[start]),
            "last_timestamp": int(timestamps[end - 1]),
            "values_path": values_relative.as_posix(),
            "values_bytes": len(value_payload),
            "values_sha256": _sha256_bytes(value_payload),
            "missing_bitmap_path": bitmap_relative.as_posix(),
            "missing_bitmap_bytes": len(bitmap_payload),
            "missing_bitmap_sha256": _sha256_bytes(bitmap_payload),
            "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
            "prepared_block_bytes": len(value_payload) + len(bitmap_payload),
        })
    if valid_offset != len(compact_values):
        raise PreparationError("prepared stream did not consume every compact value")
    return output


def prepare_experiment_dataset(
    canonical_dir: Path,
    output_dir: Path,
    *,
    cohort_path: Path = DEFAULT_COHORT_MANIFEST,
    dataset_byte_cap: int = MAX_DATASET_BYTES,
    block_byte_cap: int = MAX_BLOCK_BYTES,
) -> dict[str, Any]:
    """Create the capped per-entity 80/20 experiment artifact from the full rebuild."""
    if not 0 < dataset_byte_cap <= MAX_DATASET_BYTES:
        raise PreparationError(f"dataset_byte_cap must be in 1..{MAX_DATASET_BYTES}")
    if not 0 < block_byte_cap <= MAX_BLOCK_BYTES:
        raise PreparationError(f"block_byte_cap must be in 1..{MAX_BLOCK_BYTES}")
    np, _pa, _pc, _pq = _load_arrow()
    canonical_root = canonical_dir.resolve()
    canonical_verification = verify_canonical(canonical_root, verify_source=True)
    canonical = json.loads(
        (canonical_root / "canonical_manifest.json").read_text(encoding="utf-8")
    )
    cohort = load_cohort_manifest(cohort_path)
    cohort_entries = [
        item for item in cohort["datasets"] if item.get("id") == DATASET_ID
    ]
    if len(cohort_entries) != 1:
        raise PreparationError("deep-space dataset is not a unique frozen cohort member")
    cohort_entry = cohort_entries[0]
    if (
        cohort_entry.get("role") != "train_validation"
        or cohort_entry.get("domain") != "science"
        or cohort_entry.get("source_group") != "hf:juliensimon/deep-space-probes"
    ):
        raise PreparationError("deep-space frozen cohort assignment changed")
    logical_row_bytes = (len(VALUE_COLUMNS) + 1) * 8
    row_budget = dataset_byte_cap // logical_row_bytes
    counts = {
        (str(item["spacecraft"]),): int(item["rows"])
        for item in canonical["entities"]
    }
    allocation = equal_prefix_allocation(counts, row_budget)
    train_counts = {
        key: rows * TRAIN_NUMERATOR // TRAIN_DENOMINATOR
        for key, rows in allocation.items()
    }
    by_stream: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for block in canonical["blocks"]:
        by_stream.setdefault(
            (str(block["entity_value"]), str(block["column"])), []
        ).append(block)

    output = output_dir.resolve()
    if output.exists():
        raise PreparationError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    (temporary / "blocks").mkdir()
    prepared_blocks: list[dict[str, Any]] = []
    try:
        entity_records = []
        for (entity,), selected_rows in sorted(allocation.items()):
            timestamp_missing, timestamp_compact = _load_canonical_stream_prefix(
                canonical_root,
                by_stream[(entity, TIMESTAMP_OUTPUT_NAME)],
                selected_rows,
                np,
            )
            if bool(timestamp_missing.any()) or len(timestamp_compact) != selected_rows:
                raise PreparationError("selected canonical timestamp prefix contains a null")
            timestamps = timestamp_compact.view("<i8")
            train_rows = train_counts[(entity,)]
            for column in [TIMESTAMP_OUTPUT_NAME, *VALUE_COLUMNS]:
                if column == TIMESTAMP_OUTPUT_NAME:
                    missing, values = timestamp_missing, timestamp_compact
                    source_column, dtype = TIMESTAMP_SOURCE_NAME, "int64"
                else:
                    missing, values = _load_canonical_stream_prefix(
                        canonical_root, by_stream[(entity, column)], selected_rows, np
                    )
                    source_column, dtype = column, "float64"
                valid_prefix = np.concatenate(([0], np.cumsum(~missing, dtype=np.int64)))
                for split, start, end in (
                    ("train", 0, train_rows),
                    ("validation", train_rows, selected_rows),
                ):
                    value_start = int(valid_prefix[start])
                    value_end = int(valid_prefix[end])
                    prepared_blocks.extend(_write_prepared_stream(
                        temporary,
                        entity=entity,
                        column=column,
                        source_column=source_column,
                        dtype=dtype,
                        split=split,
                        missing=missing[start:end],
                        compact_values=values[value_start:value_end],
                        timestamps=timestamps[start:end],
                        block_byte_cap=block_byte_cap,
                        np=np,
                    ))
            entity_records.append({
                "entity_id": _entity_digest((entity,)),
                "entity_values": [entity],
                "source_rows": counts[(entity,)],
                "selected_rows": selected_rows,
                "selection": "chronological_prefix",
                "train_rows": train_rows,
                "validation_rows": selected_rows - train_rows,
                "train_boundary_rule": "floor(selected_rows * 4 / 5)",
            })

        prepared_blocks.sort(key=lambda block: (
            block["entity_id"], block["split"], block["column"], block["block_index"]
        ))
        original_numeric_bytes = sum(
            block["original_numeric_bytes"] for block in prepared_blocks
        )
        manifest: dict[str, Any] = {
            "format_version": PREPARED_FORMAT_VERSION,
            "dataset_id": DATASET_ID,
            "domain": "science",
            "source_group": "hf:juliensimon/deep-space-probes",
            "cohort": {
                "version": COHORT_VERSION,
                "assignment_sha256": COHORT_ASSIGNMENT_SHA256,
                "policy_sha256": COHORT_POLICY_SHA256,
                "role": "train_validation",
            },
            "source": {
                **canonical["source"],
                "format": "parquet",
                "upstream_locator": (
                    "https://huggingface.co/datasets/juliensimon/deep-space-probes"
                ),
            },
            "preparation_spec_sha256": canonical["preparation_spec_sha256"],
            "canonical_rebuild": {
                "path": str(canonical_root),
                "provenance_sha256": canonical_verification["provenance_sha256"],
                "rows": canonical_verification["rows"],
                "timestamp_only_rows": canonical_verification["timestamp_only_rows"],
                "selection_occurs_after_full_row_rebuild": True,
            },
            "input_contract": {
                "entity_columns": ["spacecraft"],
                "numeric_columns": [
                    {
                        "source_name": TIMESTAMP_SOURCE_NAME,
                        "output_name": TIMESTAMP_OUTPUT_NAME,
                        "dtype": "int64",
                        "width_bytes": 8,
                        "role": "timestamp",
                    },
                    *[
                        {
                            "source_name": name,
                            "output_name": name,
                            "dtype": "float64",
                            "width_bytes": 8,
                            "role": "value",
                        }
                        for name in VALUE_COLUMNS
                    ],
                ],
                "timestamp": {
                    "source_name": TIMESTAMP_SOURCE_NAME,
                    "source_kind": "datetime64[us]",
                    "source_unit": "us",
                    "output_name": TIMESTAMP_OUTPUT_NAME,
                    "output_unit": "us",
                    "timezone": "UTC",
                    "conversion": (
                        "physical datetime64[us] int64 -> Unix microseconds without scaling"
                    ),
                    "duplicate_policy": "reject",
                    "invalid_policy": "fail_no_row_index_fallback",
                },
                "missing_tokens": [],
                "nan_policy": "retain_source_float64_bits_distinct_from_missing",
            },
            "selection": {
                "source_rows": sum(counts.values()),
                "selected_rows": sum(allocation.values()),
                "entity_balance": "max_min_fair_contiguous_prefix_rows",
                "logical_row_bytes": logical_row_bytes,
                "dataset_byte_cap": dataset_byte_cap,
                "dataset_byte_cap_semantics": (
                    "selected_fixed_width_numeric_slots_including_converted_timestamp_"
                    "before_null_compaction"
                ),
                "selected_logical_slot_bytes": sum(allocation.values()) * logical_row_bytes,
                "block_byte_cap": block_byte_cap,
                "block_byte_cap_semantics": (
                    "value_payload_plus_missing_bitmap_per_column_block"
                ),
                "split": "per_entity_chronological_80_20_before_blocks",
            },
            "entities": entity_records,
            "blocks": prepared_blocks,
            "totals": {
                "blocks": len(prepared_blocks),
                "logical_slot_bytes": sum(
                    block["logical_slot_bytes"] for block in prepared_blocks
                ),
                "original_numeric_bytes": original_numeric_bytes,
                "value_payload_bytes": sum(
                    block["values_bytes"] for block in prepared_blocks
                ),
                "missing_bitmap_bytes": sum(
                    block["missing_bitmap_bytes"] for block in prepared_blocks
                ),
                "prepared_block_bytes": sum(
                    block["prepared_block_bytes"] for block in prepared_blocks
                ),
            },
            "metric_contract": {
                "denominator_name": "B_raw",
                "denominator_bytes": original_numeric_bytes,
                "denominator_definition": (
                    "sum(valid numeric values times declared original dtype width) after "
                    "the exact Unix-microsecond timestamp conversion and before compression operators"
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
            },
        }
        manifest["provenance_sha256"] = _sha256_bytes(_canonical_json(manifest))
        _atomic_json(temporary / "dataset_manifest.json", manifest)
        os.replace(temporary, output)
        verify_prepared_dataset(output, verify_source=True)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _download_exact(url: str, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == expected_bytes and sha256_file(destination) == expected_sha256:
            return
        raise PreparationError(f"existing frozen source does not match: {destination}")
    part = destination.with_name(destination.name + ".part")
    if part.exists() and part.stat().st_size == expected_bytes:
        if sha256_file(part) != expected_sha256:
            raise PreparationError(f"complete partial source has the wrong hash: {part}")
        os.replace(part, destination)
        return
    if part.exists() and part.stat().st_size > expected_bytes:
        raise PreparationError(f"partial source is larger than the frozen object: {part}")
    request = urllib.request.Request(url, headers={"User-Agent": "weighted-grpo-source-freezer/1"})
    mode = "ab" if part.exists() else "wb"
    offset = part.stat().st_size if part.exists() else 0
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=60) as response, part.open(mode) as handle:
        if offset and response.status != 206:
            raise PreparationError("upstream did not honor the resume byte range")
        shutil.copyfileobj(response, handle, length=1 << 20)
        handle.flush()
        os.fsync(handle.fileno())
    if part.stat().st_size != expected_bytes or sha256_file(part) != expected_sha256:
        raise PreparationError(f"downloaded frozen source fingerprint mismatch: {destination}")
    os.replace(part, destination)


def download_source(root: Path) -> dict[str, Any]:
    """Download only exact-revision source objects, then verify both fingerprints."""
    base = root.resolve() / DATASET_ID
    source = base / UPSTREAM_REPO_PATH
    readme = base / UPSTREAM_README_PATH
    prefix = f"https://huggingface.co/datasets/{UPSTREAM_REPO}/resolve/{UPSTREAM_REVISION}"
    _download_exact(
        f"{prefix}/{UPSTREAM_REPO_PATH}?download=true",
        source,
        SOURCE_BYTES,
        SOURCE_SHA256,
    )
    _download_exact(
        f"{prefix}/{UPSTREAM_README_PATH}?download=true",
        readme,
        README_BYTES,
        README_SHA256,
    )
    receipt = {
        "schema": "weighted-grpo-deep-space-source-lock-v1",
        "dataset_id": DATASET_ID,
        "repo_id": UPSTREAM_REPO,
        "revision": UPSTREAM_REVISION,
        "license": "cc-by-4.0",
        "files": [
            {
                "repo_path": UPSTREAM_REPO_PATH,
                "path": str(source),
                "bytes": SOURCE_BYTES,
                "sha256": SOURCE_SHA256,
                "git_oid": SOURCE_GIT_OID,
            },
            {
                "repo_path": UPSTREAM_README_PATH,
                "path": str(readme),
                "bytes": README_BYTES,
                "sha256": README_SHA256,
                "git_oid": README_GIT_OID,
            },
        ],
    }
    receipt["lock_sha256"] = _sha256_bytes(_canonical_json(receipt))
    _atomic_json(base / "deep_space_source_lock.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--root", type=Path, required=True)
    generate = commands.add_parser("generate-spec")
    generate.add_argument("--source", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--spec", type=Path, required=True)
    audit.add_argument("--source", type=Path)
    audit.add_argument("--output", type=Path)
    evidence = commands.add_parser("validate-receipt")
    evidence.add_argument("--spec", type=Path, required=True)
    evidence.add_argument("--audit", type=Path, required=True)
    rebuild = commands.add_parser("rebuild")
    rebuild.add_argument("--spec", type=Path, required=True)
    rebuild.add_argument("--source", type=Path)
    rebuild.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--canonical", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_MANIFEST)
    prepare.add_argument("--dataset-byte-cap", type=int, default=MAX_DATASET_BYTES)
    prepare.add_argument("--block-byte-cap", type=int, default=MAX_BLOCK_BYTES)
    verify = commands.add_parser("verify")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--skip-source", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = download_source(args.root)
        elif args.command == "generate-spec":
            result = formal_spec_document(args.source)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(args.output, result)
        elif args.command == "prepare":
            prepared = prepare_experiment_dataset(
                args.canonical,
                args.output,
                cohort_path=args.cohort,
                dataset_byte_cap=args.dataset_byte_cap,
                block_byte_cap=args.block_byte_cap,
            )
            result = {
                "dataset_id": prepared["dataset_id"],
                "output": str(args.output.resolve()),
                "source_rows": prepared["selection"]["source_rows"],
                "selected_rows": prepared["selection"]["selected_rows"],
                "blocks": prepared["totals"]["blocks"],
                "original_numeric_bytes": prepared["totals"]["original_numeric_bytes"],
                "provenance_sha256": prepared["provenance_sha256"],
            }
        elif args.command == "validate-receipt":
            result = validate_audit_receipt(load_spec(args.spec), args.audit)
        elif args.command == "verify":
            result = verify_canonical(args.input, verify_source=not args.skip_source)
        else:
            spec = with_source_path(load_spec(args.spec), args.source)
            if args.command == "audit":
                result = audit_source(spec)
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_json(args.output, result)
            else:
                result = rebuild_canonical(spec, args.output)
                result = {
                    "dataset_id": result["dataset_id"],
                    "output": str(args.output.resolve()),
                    "rows": result["totals"]["rows"],
                    "timestamp_only_rows": result["totals"]["timestamp_only_rows"],
                    "blocks": result["totals"]["blocks"],
                    "provenance_sha256": result["provenance_sha256"],
                }
    except (PreparationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"DEEP_SPACE_REBUILD_BLOCKER: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
