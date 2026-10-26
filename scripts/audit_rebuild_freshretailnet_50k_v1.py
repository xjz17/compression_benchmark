#!/usr/bin/env python3
"""Audit and width-normalize the frozen FreshRetailNet-50K Parquet source.

The upstream README declares ``hours_stock_status`` as Sequence(int32), while
the frozen Parquet files physically store list<int64>.  This script permits the
documented narrowing only after scanning every list element and proving that it
fits INT32.  It also audits the complete repository inventory and every
entity-date key across train and eval before optionally writing normalized
Parquet files.  The upstream split labels are provenance only; downstream code
must merge both files before making its own chronological 80/20 split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCHEMA = "freshretailnet-50k-frozen-parquet-audit-v1"
CREATED_DATE = "2026-09-17"
UPSTREAM_REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
THULAB_REVISION = "fd0aa204f48c7d55e55ef3be3e9947a492f16d9c"
EXPECTED_FILES = {
    ".gitattributes": (2570, "e4d0ebd3fe6bd5490da38d0bc76a4fb6a6581b5a394274f77461fc6a3deb737b"),
    ".gitignore": (99, "1e15ee344a6cd92c2ac9936353e0fe5a409ec5057042a8ad6824b92dde01f33c"),
    "README.md": (4871, "6e2e9bef4562855b7ea0ffec30d2cb5de485f87f33856d3dcd084eb0b71ddec9"),
    "data/eval.parquet": (8440124, "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e"),
    "data/train.parquet": (106436287, "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d"),
}
ENTITY_COLUMNS = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
)
DOUBLE_COLUMNS = (
    "sale_amount",
    "discount",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
)
INT32_COLUMNS = ("stock_hour6_22_cnt", "holiday_flag", "activity_flag")
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class FreshRetailAuditError(ValueError):
    """The frozen source violates the audited rebuild contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_schema() -> pa.Schema:
    fields = [pa.field(name, pa.int64()) for name in ENTITY_COLUMNS]
    fields.extend(
        [
            pa.field("dt", pa.string()),
            pa.field("sale_amount", pa.float64()),
            pa.field("hours_sale", pa.list_(pa.float64())),
            pa.field("stock_hour6_22_cnt", pa.int32()),
            pa.field("hours_stock_status", pa.list_(pa.int64())),
            pa.field("discount", pa.float64()),
            pa.field("holiday_flag", pa.int32()),
            pa.field("activity_flag", pa.int32()),
            pa.field("precpt", pa.float64()),
            pa.field("avg_temperature", pa.float64()),
            pa.field("avg_humidity", pa.float64()),
            pa.field("avg_wind_level", pa.float64()),
        ]
    )
    return pa.schema(fields)


def _normalized_schema(metadata: dict[bytes, bytes] | None) -> pa.Schema:
    schema = _physical_schema().set(
        _physical_schema().get_field_index("hours_stock_status"),
        pa.field("hours_stock_status", pa.list_(pa.int32())),
    )
    if metadata and b"huggingface" in metadata:
        payload = json.loads(metadata[b"huggingface"])
        payload["info"]["features"]["hours_stock_status"]["feature"]["dtype"] = "int32"
        schema = schema.with_metadata(
            {b"huggingface": json.dumps(payload, separators=(",", ":")).encode()}
        )
    return schema


def audit_inventory(root: Path) -> list[dict[str, Any]]:
    observed = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )
    if observed != sorted(EXPECTED_FILES):
        raise FreshRetailAuditError(
            f"frozen repository inventory changed: expected={sorted(EXPECTED_FILES)}, "
            f"observed={observed}"
        )
    result = []
    for relative, (expected_size, expected_hash) in EXPECTED_FILES.items():
        path = root / relative
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if (actual_size, actual_hash) != (expected_size, expected_hash):
            raise FreshRetailAuditError(
                f"frozen file changed: {relative}, size={actual_size}, sha256={actual_hash}"
            )
        result.append({"path": relative, "bytes": actual_size, "sha256": actual_hash})
    readme = (root / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "hours_sale|Sequence(float64)",
        "hours_stock_status|Sequence(int32)",
        "stock_hour6_22_cnt|int32",
        "holiday_flag|int32",
        "activity_flag|int32",
    ):
        if phrase not in readme:
            raise FreshRetailAuditError(f"README logical-width evidence is missing: {phrase}")
    return result


@dataclass
class EntityState:
    count: int
    first_day: int
    last_day: int


def _min_max(array: pa.Array) -> tuple[Any, Any]:
    result = pc.min_max(array).as_py()
    return result["min"], result["max"]


def _list_values(array: pa.Array) -> pa.Array:
    if not pa.types.is_list(array.type):
        raise FreshRetailAuditError(f"expected list array, got {array.type}")
    return array.values


def _date_days(array: pa.Array) -> np.ndarray:
    labels = np.asarray(array.to_pylist(), dtype="U10")
    if np.any(np.char.str_len(labels) != 10):
        raise FreshRetailAuditError("dt contains a non-YYYY-MM-DD label")
    try:
        days = labels.astype("datetime64[D]")
    except ValueError as error:
        raise FreshRetailAuditError(f"dt contains an invalid calendar date: {error}") from error
    if np.any(np.datetime_as_string(days, unit="D") != labels):
        raise FreshRetailAuditError("dt is not a canonical Gregorian YYYY-MM-DD label")
    return days.astype(np.int64)


def audit_parquet(
    path: Path,
    *,
    expected_rows: int,
    expected_days_per_entity: int,
    normalized_path: Path | None,
) -> tuple[dict[str, Any], dict[tuple[int, ...], EntityState]]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if schema.remove_metadata() != _physical_schema():
        raise FreshRetailAuditError(f"physical Parquet schema changed: {path.name}: {schema}")
    if parquet.metadata.num_rows != expected_rows:
        raise FreshRetailAuditError(
            f"row count changed for {path.name}: {parquet.metadata.num_rows}"
        )

    writer = None
    normalized_schema = _normalized_schema(schema.metadata)
    if normalized_path is not None:
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(normalized_path, normalized_schema, compression="zstd")

    entities: dict[tuple[int, ...], EntityState] = {}
    current_key: tuple[int, ...] | None = None
    current_count = 0
    current_first = 0
    current_last = 0
    last_finished_key: tuple[int, ...] | None = None
    null_counts: dict[str, int] = {name: 0 for name in schema.names}
    status_min = INT32_MAX
    status_max = INT32_MIN
    rows = 0

    def finish_current() -> None:
        nonlocal current_key, current_count, current_first, current_last, last_finished_key
        if current_key is None:
            return
        if current_key in entities:
            raise FreshRetailAuditError(f"entity run repeats non-contiguously: {current_key}")
        if last_finished_key is not None and current_key <= last_finished_key:
            raise FreshRetailAuditError(
                f"entity order is not strictly ascending: {last_finished_key} then {current_key}"
            )
        if current_count != expected_days_per_entity:
            raise FreshRetailAuditError(
                f"entity {current_key} has {current_count}, expected {expected_days_per_entity}"
            )
        entities[current_key] = EntityState(current_count, current_first, current_last)
        last_finished_key = current_key

    try:
        for row_group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_index)
            batch = table.combine_chunks().to_batches()[0]
            rows += batch.num_rows
            for name, column in zip(batch.schema.names, batch.columns):
                null_counts[name] += column.null_count

            sale_lengths = pc.list_value_length(batch.column(batch.schema.get_field_index("hours_sale")))
            status_array = batch.column(batch.schema.get_field_index("hours_stock_status"))
            status_lengths = pc.list_value_length(status_array)
            if _min_max(sale_lengths) != (24, 24) or _min_max(status_lengths) != (24, 24):
                raise FreshRetailAuditError("hourly sequences must contain exactly 24 elements")
            status_values = _list_values(status_array)
            if status_values.null_count:
                raise FreshRetailAuditError("hours_stock_status contains null list elements")
            local_min, local_max = _min_max(status_values)
            status_min = min(status_min, int(local_min))
            status_max = max(status_max, int(local_max))
            if local_min < INT32_MIN or local_max > INT32_MAX:
                raise FreshRetailAuditError("hours_stock_status cannot be narrowed to INT32")

            days = _date_days(batch.column(batch.schema.get_field_index("dt")))
            ids = np.column_stack(
                [
                    batch.column(batch.schema.get_field_index(name)).to_numpy(
                        zero_copy_only=False
                    )
                    for name in ENTITY_COLUMNS
                ]
            )
            changes = np.flatnonzero(np.any(ids[1:] != ids[:-1], axis=1)) + 1
            starts = np.concatenate(([0], changes))
            ends = np.concatenate((changes, [batch.num_rows]))
            for start, end in zip(starts.tolist(), ends.tolist()):
                key = tuple(int(value) for value in ids[start])
                run_days = days[start:end]
                if run_days.size > 1 and np.any(np.diff(run_days) != 1):
                    raise FreshRetailAuditError(f"entity-date labels are not daily: {key}")
                if current_key == key:
                    if int(run_days[0]) != current_last + 1:
                        raise FreshRetailAuditError(f"entity-date duplicate or gap: {key}")
                    current_count += int(run_days.size)
                    current_last = int(run_days[-1])
                else:
                    finish_current()
                    current_key = key
                    current_count = int(run_days.size)
                    current_first = int(run_days[0])
                    current_last = int(run_days[-1])

            if writer is not None:
                columns = list(batch.columns)
                status_index = batch.schema.get_field_index("hours_stock_status")
                columns[status_index] = pc.cast(status_array, pa.list_(pa.int32()), safe=True)
                normalized = pa.Table.from_arrays(columns, schema=normalized_schema)
                writer.write_table(normalized, row_group_size=batch.num_rows)
    finally:
        if writer is not None:
            writer.close()
    finish_current()
    if rows != expected_rows:
        raise FreshRetailAuditError(f"scanned row count changed: {path.name}: {rows}")
    if any(null_counts.values()):
        raise FreshRetailAuditError(f"source contains nulls: {null_counts}")
    return (
        {
            "path": path.name,
            "rows": rows,
            "row_groups": parquet.num_row_groups,
            "entities": len(entities),
            "days_per_entity": expected_days_per_entity,
            "sequence_length": 24,
            "hours_sale_source_type": "Sequence(float64)",
            "hours_stock_status_physical_type": "Sequence(int64)",
            "hours_stock_status_documented_type": "Sequence(int32)",
            "hours_stock_status_observed_min": status_min,
            "hours_stock_status_observed_max": status_max,
            "hours_stock_status_safe_int32_cast": True,
            "scalar_int32_columns": list(INT32_COLUMNS),
            "null_counts": null_counts,
        },
        entities,
    )


def _day_text(value: int) -> str:
    return np.datetime_as_string(np.datetime64(value, "D"), unit="D")


def audit_and_rebuild(
    source_root: Path, normalized_root: Path | None = None
) -> dict[str, Any]:
    inventory = audit_inventory(source_root)
    split_results = []
    entity_maps = {}
    for split, rows, days in (("train", 4_500_000, 90), ("eval", 350_000, 7)):
        normalized = (
            normalized_root / "data" / f"{split}.parquet"
            if normalized_root is not None
            else None
        )
        result, entities = audit_parquet(
            source_root / "data" / f"{split}.parquet",
            expected_rows=rows,
            expected_days_per_entity=days,
            normalized_path=normalized,
        )
        result["split"] = split
        if normalized is not None:
            result["normalized_path"] = f"data/{split}.parquet"
            result["normalized_bytes"] = normalized.stat().st_size
            result["normalized_sha256"] = sha256_file(normalized)
        split_results.append(result)
        entity_maps[split] = entities

    train = entity_maps["train"]
    evaluate = entity_maps["eval"]
    if set(train) != set(evaluate) or len(train) != 50_000:
        raise FreshRetailAuditError(
            f"train/eval entity sets changed: train={len(train)}, eval={len(evaluate)}"
        )
    overlaps = 0
    gaps = 0
    for key in train:
        train_state = train[key]
        eval_state = evaluate[key]
        if eval_state.first_day <= train_state.last_day:
            overlaps += 1
        if eval_state.first_day != train_state.last_day + 1:
            gaps += 1
    if overlaps or gaps:
        raise FreshRetailAuditError(
            f"cross-file entity-date audit failed: overlaps={overlaps}, gaps={gaps}"
        )

    if normalized_root is not None:
        (normalized_root / "README.md").write_bytes((source_root / "README.md").read_bytes())
    first_state = next(iter(train.values()))
    return {
        "schema": SCHEMA,
        "created_date": CREATED_DATE,
        "dataset_id": "freshretailnet_50k",
        "thulab_revision": THULAB_REVISION,
        "direct_upstream_revision": UPSTREAM_REVISION,
        "upstream_repository_inventory": inventory,
        "splits": split_results,
        "entity_key": list(ENTITY_COLUMNS),
        "entities": len(train),
        "total_rows": sum(item["rows"] for item in split_results),
        "entity_time_duplicates_within_or_across_files": overlaps,
        "cross_file_date_gaps": gaps,
        "date_range": {
            "first": _day_text(first_state.first_day),
            "train_last": _day_text(first_state.last_day),
            "eval_first": _day_text(evaluate[next(iter(evaluate))].first_day),
            "last": _day_text(evaluate[next(iter(evaluate))].last_day),
        },
        "date_semantics": (
            "dt is a date-only Gregorian label. UTC midnight is an explicit reversible "
            "numeric encoding convention and not an inferred event timezone."
        ),
        "rebuild_contract": {
            "merge_before_split": True,
            "ignore_upstream_train_eval_as_experiment_split": True,
            "own_split": "per-entity chronological first 80% train, last 20% validation",
            "hours_sale_output": "24 DOUBLE values, preserving Sequence(float64)",
            "hours_stock_status_output": (
                "24 INT32 values after full-range checked cast from physical list<int64>, "
                "preserving the README Sequence(int32) contract"
            ),
            "scalar_int32_output": list(INT32_COLUMNS),
            "entity_id_output": "seven source INT64 identifiers",
        },
        "status": "source_audit_and_width_normalization_passed",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_and_rebuild(
        args.source_root.resolve(),
        args.normalized_root.resolve() if args.normalized_root else None,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
