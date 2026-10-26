#!/usr/bin/env python3
"""Source-backed audits for fixed weighted-GRPO data admissions.

The audit deliberately treats a calendar date as a date, not as an inferred
wall-clock instant.  A source column whose every label is a unique Gregorian
date (or midnight label) may be encoded reversibly as Unix days at 00:00 UTC.
This is an explicit numeric encoding convention; it is not a claim about the
timezone in which the source event occurred.  Sub-daily naive timestamps remain
blocked unless their timezone/epoch basis is documented by the source.

The script also checks the four frozen GlucoFM TsFiles.  It can prove ordering,
cross-file ranges, and duplicate absence, but it does not override the upstream
card's statement that timestamps may be either Unix seconds or seconds from a
recording start.
"""

from __future__ import annotations

import argparse
import calendar
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
REPORT_SCHEMA = "weighted-grpo-blocked-source-remediation-audit-v1"


class AuditError(ValueError):
    """Frozen source evidence violates the declared remediation contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calendar_midnight_unix_ms(label: str, formats: Iterable[str]) -> int:
    """Encode a date/midnight label at UTC midnight without timezone inference."""
    for format_string in formats:
        try:
            parsed = datetime.strptime(label, format_string)
        except ValueError:
            continue
        if parsed.hour or parsed.minute or parsed.second or parsed.microsecond:
            raise AuditError(f"timestamp is not a calendar-date midnight label: {label!r}")
        return calendar.timegm(parsed.timetuple()) * 1000
    raise AuditError(f"timestamp does not match the frozen calendar formats: {label!r}")


@dataclass(frozen=True)
class CalendarDataset:
    dataset_id: str
    source_csv: Path
    source_csv_sha256: str
    timestamp_column: str
    timestamp_formats: tuple[str, ...]
    tsfile_path: Path
    tsfile_sha256: str
    table_name: str
    query_columns: tuple[str, ...]
    expected_field_types: Mapping[str, str]
    source_revision: str
    source_repo_path: str


def _type_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _category_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _load_tsfile_module():
    try:
        import tsfile
    except Exception as error:  # pragma: no cover - environment gate
        raise AuditError(f"Apache TsFile Python reader is unavailable: {error}") from error
    return tsfile


def audit_calendar_dataset(spec: CalendarDataset) -> dict[str, Any]:
    for path, expected, label in (
        (spec.source_csv, spec.source_csv_sha256, "upstream CSV"),
        (spec.tsfile_path, spec.tsfile_sha256, "frozen TsFile"),
    ):
        if not path.is_file():
            raise AuditError(f"{label} is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise AuditError(
                f"{label} fingerprint mismatch for {spec.dataset_id}: "
                f"expected={expected}, actual={actual}"
            )

    source_times: list[int] = []
    with spec.source_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if spec.timestamp_column not in (reader.fieldnames or ()):
            raise AuditError(
                f"source timestamp column {spec.timestamp_column!r} is missing"
            )
        for row_number, row in enumerate(reader, start=2):
            label = row.get(spec.timestamp_column)
            if label is None:
                raise AuditError(f"missing timestamp at CSV row {row_number}")
            try:
                source_times.append(
                    calendar_midnight_unix_ms(label.strip(), spec.timestamp_formats)
                )
            except AuditError as error:
                raise AuditError(f"CSV row {row_number}: {error}") from error
    if not source_times:
        raise AuditError(f"source CSV is empty: {spec.dataset_id}")
    if len(source_times) != len(set(source_times)):
        raise AuditError(f"source calendar labels are not unique: {spec.dataset_id}")
    if source_times != sorted(source_times):
        raise AuditError(f"source calendar labels are not ascending: {spec.dataset_id}")

    tsfile = _load_tsfile_module()
    reader = tsfile.TsFileReader(str(spec.tsfile_path))
    result = None
    try:
        schemas = reader.get_all_table_schemas()
        schema = schemas.get(spec.table_name)
        if schema is None:
            raise AuditError(
                f"frozen table {spec.table_name!r} is absent: {sorted(schemas)}"
            )
        fields: dict[str, tuple[str, str]] = {}
        for column in schema.get_columns():
            fields[str(column.get_column_name())] = (
                _type_name(column.get_data_type()),
                _category_name(column.get_category()),
            )
        for name, expected_type in spec.expected_field_types.items():
            if fields.get(name) != (expected_type, "FIELD"):
                raise AuditError(
                    f"native field contract mismatch for {spec.dataset_id}.{name}: "
                    f"expected={(expected_type, 'FIELD')}, actual={fields.get(name)}"
                )

        result = reader.query_table(
            spec.table_name,
            list(spec.query_columns),
            INT64_MIN,
            INT64_MAX,
        )
        tsfile_times: list[int] = []
        while result.next():
            tsfile_times.append(int(result.get_value_by_index(1)))
    finally:
        if result is not None:
            result.close()
        reader.close()
    if tsfile_times != source_times:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(tsfile_times, source_times))
                if pair[0] != pair[1]
            ),
            min(len(tsfile_times), len(source_times)),
        )
        raise AuditError(
            f"frozen Time does not exactly match UTC-midnight calendar encoding for "
            f"{spec.dataset_id}: source_rows={len(source_times)}, "
            f"tsfile_rows={len(tsfile_times)}, first_mismatch={mismatch}"
        )
    return {
        "dataset_id": spec.dataset_id,
        "status": "resolved_calendar_date_encoding",
        "source_revision": spec.source_revision,
        "source_repo_path": spec.source_repo_path,
        "source_csv_sha256": spec.source_csv_sha256,
        "frozen_tsfile_sha256": spec.tsfile_sha256,
        "row_count": len(source_times),
        "first_unix_ms": source_times[0],
        "last_unix_ms": source_times[-1],
        "unique_ascending": True,
        "exact_tsfile_time_match": True,
        "timestamp_semantics": "calendar_date_not_wall_clock_instant",
        "conversion": "Gregorian source date -> Unix milliseconds at 00:00:00 UTC",
        "timezone_claim": (
            "UTC is an explicit reversible encoding convention for date-only labels; "
            "the source event timezone is not inferred"
        ),
        "native_field_types": dict(spec.expected_field_types),
    }


def audit_electricity_portuguese_civil_grid(source_root: Path) -> dict[str, Any]:
    """Verify the frozen table preserves the documented Portuguese civil grid.

    UCI explicitly states that every civil day has 96 labels: the nonexistent
    March hour is represented by zeros and the repeated October hour is
    aggregated.  Therefore these labels are ordered civil-time coordinates,
    not a one-to-one mapping to UTC event instants.
    """
    path = source_root / "electricity_load_2011_2014/electricity_load_2011_2014.tsfile"
    expected_sha256 = "85f942fc7a9ea558c41337710459bfa3c1527a3d79825980cd3eafd6f9aa0507"
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise AuditError("frozen electricity_load_2011_2014 source is missing or substituted")

    tsfile = _load_tsfile_module()
    reader = tsfile.TsFileReader(str(path))
    result = None
    try:
        schemas = reader.get_all_table_schemas()
        schema = schemas.get("electricity_load")
        if schema is None:
            raise AuditError("frozen electricity_load table is absent")
        fields = {
            str(column.get_column_name()): (
                _type_name(column.get_data_type()),
                _category_name(column.get_category()),
            )
            for column in schema.get_columns()
        }
        expected_fields = {
            f"mt_{index:03d}": ("FLOAT", "FIELD") for index in range(1, 371)
        }
        if fields != expected_fields:
            raise AuditError("electricity_load native 370-column FLOAT schema changed")

        result = reader.query_table(
            "electricity_load", ["mt_001"], INT64_MIN, INT64_MAX
        )
        timestamps: list[int] = []
        while result.next():
            timestamps.append(int(result.get_value_by_index(1)))
    finally:
        if result is not None:
            result.close()
        reader.close()

    first = calendar.timegm(datetime(2011, 1, 1, 0, 15).timetuple()) * 1000
    last = calendar.timegm(datetime(2015, 1, 1, 0, 0).timetuple()) * 1000
    if len(timestamps) != 140256 or not timestamps:
        raise AuditError("electricity_load row count no longer matches the UCI civil grid")
    if timestamps[0] != first or timestamps[-1] != last:
        raise AuditError("electricity_load first/last civil labels changed")
    if any(right - left != 900000 for left, right in zip(timestamps, timestamps[1:])):
        raise AuditError("electricity_load civil labels are not a complete 15-minute grid")
    return {
        "dataset_id": "electricity_load_2011_2014",
        "status": "resolved_portuguese_civil_coordinate_encoding",
        "frozen_revision": "f15d9f598e23ee7d57df58f7d0f17ad6f1a3f104",
        "frozen_tsfile_sha256": expected_sha256,
        "rows": len(timestamps),
        "numeric_fields": len(fields),
        "native_field_type": "FLOAT",
        "first_civil_coordinate_unix_ms": timestamps[0],
        "last_civil_coordinate_unix_ms": timestamps[-1],
        "civil_step_ms": 900000,
        "source_timezone": "Portuguese hour (Europe/Lisbon civil-time labels)",
        "event_utc_instant_claim": False,
        "event_time_mapping_is_injective": False,
        "dst_policy": (
            "Every civil day retains 96 labels: March 01:00-02:00 values are zero "
            "and October 01:00-02:00 values aggregate both repeated hours."
        ),
        "coordinate_encoding": (
            "The Gregorian civil label fields are encoded as a Unix-millisecond-shaped "
            "integer on a UTC coordinate grid solely to preserve ordering and labels; "
            "the integer is not asserted to be the exact UTC instant of measurement."
        ),
        "uci_evidence_url": (
            "https://archive.ics.uci.edu/dataset/321/"
            "electricityloaddiagrams20112014"
        ),
        "uci_api_url": "https://archive.ics.uci.edu/api/dataset?id=321",
        "uci_api_sha256_observed": (
            "b2dcc4da3b2bb104f2bab064bdf5f1da280fb2446a1a9d464f4319963023f56f"
        ),
    }


def audit_finland_helsinki_civil_grid(
    source_root: Path, upstream_root: Path
) -> dict[str, Any]:
    """Bind the frozen TIME table to its documented Finland civil labels.

    Fintraffic documents the raw TMS time fields as current Finland time
    (EET/EEST).  The frozen Real-TSF CSV has exactly 96 quarter-hour labels on
    every 2024 civil day, including both DST transition days.  We therefore
    preserve the processed labels as a reversible civil-coordinate grid and do
    not reinterpret them as exact UTC event instants.
    """
    source_csv = upstream_root / "finland_traffic__1.csv"
    source_csv_sha256 = (
        "6c8e7bdb4563bdd0de024e05cef90faebc3cb5ba33eab633cf42a9f28d24d17e"
    )
    tsfile_path = source_root / "finland_traffic/finland_traffic.tsfile"
    tsfile_sha256 = (
        "0d96ab13536f0608796e9cb37b127c262299883eb7f88b341f099f6c665e6571"
    )
    for path, expected, label in (
        (source_csv, source_csv_sha256, "Finland frozen source CSV"),
        (tsfile_path, tsfile_sha256, "Finland frozen TsFile"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(f"{label} is missing or substituted")

    source_times: list[int] = []
    source_values: list[float] = []
    source_labels: list[datetime] = []
    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["timestamp", "volume"]:
            raise AuditError(f"Finland source columns changed: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            try:
                label = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                value = float(row["volume"])
            except (KeyError, TypeError, ValueError) as error:
                raise AuditError(
                    f"invalid Finland source row {row_number}: {error}"
                ) from error
            source_labels.append(label)
            source_times.append(calendar.timegm(label.timetuple()) * 1000)
            source_values.append(value)
    if len(source_times) != 35136:
        raise AuditError(f"Finland source row count changed: {len(source_times)}")
    if any(right - left != 900000 for left, right in zip(source_times, source_times[1:])):
        raise AuditError("Finland processed civil labels are not a complete 15-minute grid")

    helsinki = ZoneInfo("Europe/Helsinki")
    nonexistent_labels = []
    ambiguous_labels = []
    for label in source_labels:
        roundtrips = []
        offsets = set()
        for fold in (0, 1):
            aware = label.replace(tzinfo=helsinki, fold=fold)
            offsets.add(aware.utcoffset())
            roundtrips.append(
                aware.astimezone(ZoneInfo("UTC"))
                .astimezone(helsinki)
                .replace(tzinfo=None)
                == label
            )
        if not any(roundtrips):
            nonexistent_labels.append(label)
        elif len(offsets) == 2 and all(roundtrips):
            ambiguous_labels.append(label)
    if len(nonexistent_labels) != 4 or len(ambiguous_labels) != 4:
        raise AuditError(
            "Finland 2024 DST-label audit changed: "
            f"nonexistent={len(nonexistent_labels)}, ambiguous={len(ambiguous_labels)}"
        )
    index_by_label = {label: index for index, label in enumerate(source_labels)}
    if any(source_values[index_by_label[label]] != 0 for label in nonexistent_labels):
        raise AuditError("Finland nonexistent spring labels are no longer zero-filled")

    tsfile = _load_tsfile_module()
    reader = tsfile.TsFileReader(str(tsfile_path))
    result = None
    try:
        schema = reader.get_all_table_schemas().get("finland_traffic")
        if schema is None:
            raise AuditError("frozen finland_traffic table is absent")
        fields = {
            str(column.get_column_name()): (
                _type_name(column.get_data_type()),
                _category_name(column.get_category()),
            )
            for column in schema.get_columns()
        }
        if fields != {
            "freq": ("STRING", "TAG"),
            "series": ("STRING", "TAG"),
            "channel": ("STRING", "TAG"),
            "measurement": ("FLOAT", "FIELD"),
        }:
            raise AuditError(f"Finland native schema changed: {fields}")
        result = reader.query_table(
            "finland_traffic", ["measurement"], INT64_MIN, INT64_MAX
        )
        tsfile_times: list[int] = []
        while result.next():
            tsfile_times.append(int(result.get_value_by_index(1)))
    finally:
        if result is not None:
            result.close()
        reader.close()
    if tsfile_times != source_times:
        raise AuditError("Finland frozen Time does not exactly preserve source civil labels")
    return {
        "dataset_id": "finland_traffic",
        "status": "resolved_helsinki_civil_coordinate_encoding",
        "frozen_revision": "93b678deb80b12a480c1d9d6ad7dad58c79b3bd5",
        "source_revision": "683bb21b470f5a26586e1ab2295f0a2a1529ef0f",
        "source_repo_path": "Finland_Traffic/15T/1.csv",
        "source_csv_sha256": source_csv_sha256,
        "frozen_tsfile_sha256": tsfile_sha256,
        "rows": len(source_times),
        "numeric_fields": 1,
        "native_field_types": {"measurement": "FLOAT"},
        "exact_tsfile_time_match": True,
        "unique_ascending": True,
        "first_civil_coordinate_unix_ms": source_times[0],
        "last_civil_coordinate_unix_ms": source_times[-1],
        "civil_step_ms": 900000,
        "source_timezone": "Europe/Helsinki civil-time labels (EET/EEST)",
        "event_utc_instant_claim": False,
        "event_time_mapping_is_injective": False,
        "dst_audit": {
            "nonexistent_spring_labels": len(nonexistent_labels),
            "ambiguous_autumn_labels": len(ambiguous_labels),
            "rows_on_2024_03_31": sum(
                label.date().isoformat() == "2024-03-31" for label in source_labels
            ),
            "rows_on_2024_10_27": sum(
                label.date().isoformat() == "2024-10-27" for label in source_labels
            ),
            "nonexistent_spring_values_are_zero": True,
            "processed_series_policy": (
                "The frozen processed CSV retains a regular 96-label civil grid on "
                "both DST days; no exact event-UTC interpretation is made."
            ),
        },
        "coordinate_encoding": (
            "The naive Finland civil labels are encoded on a Unix-millisecond-shaped "
            "coordinate grid solely to preserve source ordering and precision."
        ),
        "fintraffic_evidence_url": (
            "https://github.com/tmfg/digitraffic/blob/"
            "d74da63acab03d74dd8df97864edd42579daf8bb/pages/lam-en.md"
        ),
    }


def audit_gluco_collection(root: Path) -> dict[str, Any]:
    expected = {
        "gluco_tsfm_benchmark_test.tsfile":
            "d147e39cda16c717ec7b9008b6b953e6512b903abab0e81dbe991fa17ad973bc",
        "gluco_tsfm_benchmark_train_1.tsfile":
            "cbc186a8513367747bd4322b293691ba934e5ea2b78a41e309264c525b8139be",
        "gluco_tsfm_benchmark_train_2.tsfile":
            "1e1759678bc89320c20718589b5d0296b4b9d22e78cfc3fc68135e45fcf2924c",
        "gluco_tsfm_benchmark_train_3.tsfile":
            "2877f23cd753ad7de41883edaf20243bc29d408f300fa001b384dc78e5a0c0ee",
    }
    tsfile = _load_tsfile_module()
    entity_ranges: dict[tuple[str, str], list[tuple[str, int, int, int]]] = {}
    rows = 0
    adjacent_duplicates = 0
    ordering_regressions = 0
    for filename, digest in expected.items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise AuditError(f"frozen GlucoFM member is missing or substituted: {filename}")
        reader = tsfile.TsFileReader(str(path))
        result = None
        try:
            schemas = reader.get_all_table_schemas()
            if len(schemas) != 1:
                raise AuditError(f"unexpected GlucoFM table count in {filename}")
            table_name, schema = next(iter(schemas.items()))
            columns = {
                str(column.get_column_name()): (
                    _type_name(column.get_data_type()),
                    _category_name(column.get_category()),
                )
                for column in schema.get_columns()
            }
            if columns != {
                "dataset": ("STRING", "TAG"),
                "subject_id": ("STRING", "TAG"),
                "bgvalue": ("DOUBLE", "FIELD"),
            }:
                raise AuditError(f"GlucoFM native schema changed in {filename}: {columns}")
            result = reader.query_table(
                table_name,
                ["dataset", "subject_id", "bgvalue"],
                INT64_MIN,
                INT64_MAX,
            )
            current: dict[tuple[str, str], list[int]] = {}
            while result.next():
                timestamp = int(result.get_value_by_index(1))
                key = (
                    str(result.get_value_by_index(2)),
                    str(result.get_value_by_index(3)),
                )
                state = current.get(key)
                if state is None:
                    state = [1, timestamp, timestamp, timestamp]
                    current[key] = state
                else:
                    if timestamp == state[3]:
                        adjacent_duplicates += 1
                    elif timestamp < state[3]:
                        ordering_regressions += 1
                    state[0] += 1
                    state[1] = min(state[1], timestamp)
                    state[2] = max(state[2], timestamp)
                    state[3] = timestamp
                rows += 1
            for key, state in current.items():
                entity_ranges.setdefault(key, []).append(
                    (filename, state[0], state[1], state[2])
                )
        finally:
            if result is not None:
                result.close()
            reader.close()

    range_overlaps = []
    multi_file_entities = 0
    for key, ranges in sorted(entity_ranges.items()):
        if len(ranges) < 2:
            continue
        multi_file_entities += 1
        ordered = sorted(ranges, key=lambda item: (item[2], item[3], item[0]))
        for left, right in zip(ordered, ordered[1:]):
            if right[2] <= left[3]:
                range_overlaps.append({"entity": list(key), "left": left, "right": right})
    return {
        "dataset_id": "gluco_tsfm_benchmark",
        "status": "still_blocked_timestamp_semantics",
        "frozen_revision": "d0840cc1f87ed2fb43d0e22245aea948eb38eef8",
        "files": len(expected),
        "rows": rows,
        "entities": len(entity_ranges),
        "entities_spanning_multiple_files": multi_file_entities,
        "adjacent_duplicate_entity_time_keys": adjacent_duplicates,
        "ordering_regressions": ordering_regressions,
        "cross_file_entity_time_range_overlaps": len(range_overlaps),
        "merge_audit_resolved": (
            adjacent_duplicates == 0
            and ordering_regressions == 0
            and not range_overlaps
        ),
        "remaining_blocker": (
            "The exact upstream card at cf9a9c3471e0e0ada92d604ae54807cfb746036c "
            "states that timestamp arrays may be Unix seconds or seconds since start, "
            "while the frozen THULab card claims all rows are epoch seconds.  Per-cohort "
            "origins are not documented, so magnitude cannot be used to choose semantics."
        ),
        "upstream_card_sha256":
            "8fc783e07ae8aa59b3016c532d72152a4928b02e7fc76b418afe93065d4827f3",
    }


METR_FEATURES = tuple(
    [
        f"x_t_minus_{offset}_d{dimension}"
        for offset in range(11, 0, -1)
        for dimension in range(2)
    ]
    + [f"x_t_plus_0_d{dimension}" for dimension in range(2)]
    + [
        f"y_t_plus_{offset}_d{dimension}"
        for offset in range(1, 13)
        for dimension in range(2)
    ]
)
METR_ROW = struct.Struct("<48f")


def metr_overlap_matches(previous: bytes, current: bytes) -> bool:
    """Adjacent 24-step windows agree on their 23 shared physical steps."""
    if len(previous) != METR_ROW.size or len(current) != METR_ROW.size:
        raise AuditError("METR-LA row payload must contain exactly 48 FLOAT values")
    return previous[2 * 4 :] == current[: 46 * 4]


def audit_metr_la_dewindow(root: Path) -> dict[str, Any]:
    expected = [
        (
            "metr_la_train.tsfile",
            "621e6ff0d651d4268f9f54e7de62dfbd208996df8c6825c17ab25c228b3034c8",
        ),
        (
            "metr_la_val.tsfile",
            "121f7642157c440ce4c55c991515cfe1b7fb1ef1561629c4eb794d651a8fc30f",
        ),
        (
            "metr_la_test.tsfile",
            "13b1a2a33e4ae2161968ac857426a23549ce8f2bf82cd4914c0d11ac96c544f4",
        ),
    ]
    tsfile = _load_tsfile_module()
    split_edges: dict[str, dict[str, tuple[int, bytes]]] = {}
    total_rows = 0
    adjacent_comparisons = 0
    non_finite_values = 0
    node_ids: set[str] = set()
    for filename, digest in expected:
        path = root / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise AuditError(f"frozen METR-LA member is missing or substituted: {filename}")
        reader = tsfile.TsFileReader(str(path))
        result = None
        try:
            schemas = reader.get_all_table_schemas()
            if len(schemas) != 1:
                raise AuditError(f"unexpected METR-LA table count in {filename}")
            table_name, schema = next(iter(schemas.items()))
            columns = {
                str(column.get_column_name()): (
                    _type_name(column.get_data_type()),
                    _category_name(column.get_category()),
                )
                for column in schema.get_columns()
            }
            if columns.get("node_id") != ("STRING", "TAG"):
                raise AuditError(f"METR-LA node_id schema changed in {filename}")
            for feature in METR_FEATURES:
                if columns.get(feature) != ("FLOAT", "FIELD"):
                    raise AuditError(
                        f"METR-LA feature schema changed in {filename}: {feature}"
                    )
            result = reader.query_table(
                table_name,
                ["node_id", *METR_FEATURES],
                INT64_MIN,
                INT64_MAX,
            )
            first: dict[str, tuple[int, bytes]] = {}
            last: dict[str, tuple[int, bytes]] = {}
            previous_node: str | None = None
            previous_time: int | None = None
            previous_payload: bytes | None = None
            while result.next():
                timestamp = int(result.get_value_by_index(1))
                node_id = str(result.get_value_by_index(2))
                values = [float(result.get_value_by_index(index)) for index in range(3, 51)]
                if any(not (value == value and abs(value) != float("inf")) for value in values):
                    non_finite_values += sum(
                        not (value == value and abs(value) != float("inf"))
                        for value in values
                    )
                payload = METR_ROW.pack(*values)
                node_ids.add(node_id)
                if node_id == previous_node:
                    assert previous_time is not None and previous_payload is not None
                    if timestamp - previous_time != 300000:
                        raise AuditError(
                            f"METR-LA non-5-minute step in {filename}, node={node_id}: "
                            f"previous={previous_time}, current={timestamp}"
                        )
                    if not metr_overlap_matches(previous_payload, payload):
                        raise AuditError(
                            f"METR-LA overlapping windows disagree in {filename}, "
                            f"node={node_id}, previous={previous_time}, current={timestamp}"
                        )
                    adjacent_comparisons += 1
                else:
                    if node_id in last:
                        raise AuditError(
                            f"METR-LA node rows are not contiguous in {filename}: {node_id}"
                        )
                    first[node_id] = (timestamp, payload)
                last[node_id] = (timestamp, payload)
                previous_node = node_id
                previous_time = timestamp
                previous_payload = payload
                total_rows += 1
            if set(first) != set(last) or len(first) != 207:
                raise AuditError(
                    f"METR-LA split {filename} does not contain exactly 207 nodes"
                )
            split_edges[filename] = {
                node: (first[node][0], first[node][1], last[node][0], last[node][1])
                for node in first
            }
        finally:
            if result is not None:
                result.close()
            reader.close()

    boundary_comparisons = 0
    for left_name, right_name in zip(
        [item[0] for item in expected], [item[0] for item in expected][1:]
    ):
        left = split_edges[left_name]
        right = split_edges[right_name]
        if set(left) != set(right):
            raise AuditError(f"METR-LA split node sets differ: {left_name}, {right_name}")
        for node_id in left:
            left_last_time, left_last_payload = left[node_id][2], left[node_id][3]
            right_first_time, right_first_payload = right[node_id][0], right[node_id][1]
            if right_first_time - left_last_time != 300000:
                raise AuditError(
                    f"METR-LA split boundary is not consecutive for node {node_id}: "
                    f"{left_name}={left_last_time}, {right_name}={right_first_time}"
                )
            if not metr_overlap_matches(left_last_payload, right_first_payload):
                raise AuditError(
                    f"METR-LA split-boundary windows disagree for node {node_id}: "
                    f"{left_name} -> {right_name}"
                )
            boundary_comparisons += 1
    if non_finite_values:
        raise AuditError(
            f"METR-LA contains {non_finite_values} non-finite FLOAT assignments; "
            "the public reader cannot certify their payload bits"
        )
    return {
        "dataset_id": "METR-LA",
        "status": "still_blocked_timestamp_timezone",
        "frozen_revision": "d5895c93e8f9d9d1ef20540b10e0c8d75bfddf81",
        "files": len(expected),
        "rows": total_rows,
        "nodes": len(node_ids),
        "numeric_window_fields": len(METR_FEATURES),
        "adjacent_window_comparisons": adjacent_comparisons,
        "split_boundary_comparisons": boundary_comparisons,
        "non_finite_values": non_finite_values,
        "dewindow_value_agreement_audit": "passed_exact_float32_bits",
        "reconstruction_rule": (
            "For each node, keep the first 24 physical offsets and append the newest "
            "y_t_plus_12 value from each later row; train/val/test are first merged in "
            "chronological order before the experiment's per-entity 80/20 split."
        ),
        "remaining_blocker": (
            "The ISO-8601 t0_timestamp source labels and frozen card do not document "
            "a timezone or UTC epoch basis; numeric magnitude is not used to infer one."
        ),
    }
def build_report(source_root: Path, upstream_root: Path) -> dict[str, Any]:
    datasets = [
        CalendarDataset(
            dataset_id="exchange_rate",
            source_csv=upstream_root / "exchange_rate__exchange_rate.csv",
            source_csv_sha256=
                "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842",
            timestamp_column="date",
            timestamp_formats=("%Y/%m/%d %H:%M",),
            tsfile_path=source_root / "exchange_rate/exchange_rate.tsfile",
            tsfile_sha256=
                "ed2a30fb607e99b3616c3864081767ab887e39e56596921b60e8a79c6c149eb4",
            table_name="exchange_rate",
            query_columns=("c_0", "c_1", "c_2", "c_3", "c_4", "c_5", "c_6", "ot"),
            expected_field_types={
                "c_0": "FLOAT", "c_1": "FLOAT", "c_2": "FLOAT", "c_3": "FLOAT",
                "c_4": "FLOAT", "c_5": "FLOAT", "c_6": "FLOAT", "ot": "FLOAT",
            },
            source_revision="2b66e59ee19dac8f6f19fb5d4997f289fdfea357",
            source_repo_path="exchange_rate/exchange_rate.csv",
        ),
        CalendarDataset(
            dataset_id="national_illness",
            source_csv=upstream_root / "illness__national_illness.csv",
            source_csv_sha256=
                "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a",
            timestamp_column="date",
            timestamp_formats=("%Y-%m-%d %H:%M:%S",),
            tsfile_path=source_root / "national_illness/national_illness.tsfile",
            tsfile_sha256=
                "bcc1926130d4e97ccc9e4da892e730f25ef22bc175203120e7b28518629e6f7d",
            table_name="national_illness",
            query_columns=(
                "pct_weighted_ili", "pctunweighted_ili", "age_0_4", "age_5_24",
                "ilitotal", "num_of_providers", "ot",
            ),
            expected_field_types={
                "pct_weighted_ili": "FLOAT", "pctunweighted_ili": "FLOAT",
                "age_0_4": "FLOAT", "age_5_24": "FLOAT", "ilitotal": "FLOAT",
                "num_of_providers": "FLOAT", "ot": "FLOAT",
            },
            source_revision="2b66e59ee19dac8f6f19fb5d4997f289fdfea357",
            source_repo_path="illness/national_illness.csv",
        ),
        CalendarDataset(
            dataset_id="online_retail_2_uci",
            source_csv=upstream_root / "Online_Retail_2_UCI__D__item0.csv",
            source_csv_sha256=
                "6f5629179e889a422271a1ad41d040df69234af88295c02e7e34ca1fbecba185",
            timestamp_column="timestamp",
            timestamp_formats=("%Y-%m-%d",),
            tsfile_path=source_root / "online_retail_2_uci/online_retail_2_uci.tsfile",
            tsfile_sha256=
                "ee76c15c063ce755f021e3d78cafead80d634ae4c35cb37707ab58ede1e6922c",
            table_name="online_retail_2_uci",
            query_columns=("freq", "series", "channel", "measurement"),
            expected_field_types={"measurement": "FLOAT"},
            source_revision="683bb21b470f5a26586e1ab2295f0a2a1529ef0f",
            source_repo_path="Online_Retail_2_UCI/D/item0.csv",
        ),
    ]
    records = [audit_calendar_dataset(spec) for spec in datasets]
    records.append(audit_electricity_portuguese_civil_grid(source_root))
    records.append(audit_finland_helsinki_civil_grid(source_root, upstream_root))
    records.append(audit_gluco_collection(source_root / "gluco_tsfm_benchmark"))
    records.append(audit_metr_la_dewindow(source_root / "METR-LA"))
    return {
        "schema": REPORT_SCHEMA,
        "created_date": "2026-09-17",
        "policy": (
            "Date-only labels may use an explicit UTC-midnight Unix encoding without "
            "inferring an event timezone. Sub-daily naive labels remain blocked."
        ),
        "records": records,
        "approved_delta": [
            record["dataset_id"]
            for record in records
            if record["status"].startswith("resolved_")
        ],
        "still_blocked": [
            record["dataset_id"]
            for record in records
            if record["status"].startswith("still_blocked")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.source_root.resolve(), args.upstream_root.resolve())
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
