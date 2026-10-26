#!/usr/bin/env python3
"""Equivalence and deterministic performance fixtures for the v5 prototype."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from thulab_v5_planning_fastpath import (  # noqa: E402
    TupleEntityCountCache,
    canonical_entity_key,
    flush_entity_counts_to_sqlite,
    planning_projection,
)


@dataclass(frozen=True)
class _Temporal:
    source_name: str


@dataclass(frozen=True)
class _Field:
    name: str
    temporal: _Temporal | None = None


@dataclass(frozen=True)
class _Condition:
    source_name: str
    source_category: str


@dataclass(frozen=True)
class _Filter:
    conditions: tuple[_Condition, ...]


@dataclass(frozen=True)
class _Plan:
    tag_names: tuple[str, ...]
    fields: tuple[_Field, ...]
    primary_temporal: _Temporal | None = None
    row_filter: _Filter | None = None


def _new_entity_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE entities (
            plan_ordinal INTEGER NOT NULL,
            entity_key BLOB NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (plan_ordinal, entity_key)
        ) WITHOUT ROWID
        """
    )
    return connection


class PlanningProjectionTest(unittest.TestCase):
    def test_fresh_style_projection_omits_all_native_value_fields(self) -> None:
        tags = tuple(f"entity_{index}" for index in range(7))
        native_fields = tuple(_Field(f"value_{index}") for index in range(57))
        projection = planning_projection(_Plan(tags, native_fields))

        self.assertEqual(projection.query_columns, tags)
        self.assertEqual(projection.omitted_native_fields, tuple(
            field.name for field in native_fields
        ))
        self.assertEqual(len(projection.legacy_query_columns), 64)
        self.assertEqual(projection.removed_column_count, 57)

    def test_temporal_and_filter_columns_are_retained_once_in_legacy_order(self) -> None:
        start = _Temporal("start_timestamp")
        event = _Temporal("event_time")
        plan = _Plan(
            tag_names=("series", "start_timestamp"),
            fields=(
                _Field("value"),
                _Field("start_timestamp", start),
                _Field("event_time", event),
                _Field("quality"),
            ),
            primary_temporal=event,
            row_filter=_Filter((
                _Condition("Time", "TIME"),
                _Condition("quality", "FIELD"),
                _Condition("series", "TAG"),
            )),
        )

        projection = planning_projection(plan)

        self.assertEqual(
            projection.query_columns,
            ("series", "start_timestamp", "event_time", "quality"),
        )
        self.assertEqual(
            projection.temporal_columns, ("start_timestamp", "event_time")
        )
        self.assertEqual(projection.filter_columns, ("quality", "series"))
        self.assertEqual(projection.omitted_native_fields, ("value",))

    def test_primary_temporal_is_kept_even_if_not_repeated_in_fields(self) -> None:
        projection = planning_projection(_Plan(
            tag_names=("entity",),
            fields=(_Field("value"),),
            primary_temporal=_Temporal("event_time"),
        ))
        self.assertEqual(projection.query_columns, ("entity", "event_time"))
        self.assertEqual(projection.temporal_columns, ("event_time",))

    def test_representative_counting_result_is_projection_independent(self) -> None:
        plan = _Plan(
            tag_names=("shop", "item"),
            fields=tuple(_Field(f"value_{index}") for index in range(6)),
            row_filter=_Filter((_Condition("quality", "FIELD"),)),
        )
        projection = planning_projection(plan)
        rows = [
            {"Time": index, "shop": f"s{index % 2}", "item": f"i{index % 3}",
             "quality": 1 if index % 5 else 0,
             **{f"value_{field}": index * field for field in range(6)}}
            for index in range(120)
        ]

        def count_entities(columns: tuple[str, ...]) -> Counter[tuple[str, str]]:
            # Accessing only ``columns`` models projection cost; row admission and
            # entity construction deliberately mirror the same planning rules.
            accessed = [{name: row[name] for name in columns} for row in rows]
            return Counter(
                (row["shop"], row["item"])
                for row in accessed
                if row["quality"] >= 1
            )

        legacy = count_entities(projection.legacy_query_columns)
        minimal = count_entities(projection.query_columns)
        self.assertEqual(minimal, legacy)
        self.assertEqual(sum(minimal.values()), 96)


class TupleEntityCountCacheTest(unittest.TestCase):
    def test_sqlite_rows_match_legacy_per_row_canonical_keys(self) -> None:
        events = [
            (0, ("alpha", "1")),
            (0, ("alpha", "1")),
            (0, ("beta", "2")),
            (1, ("alpha", "1")),
            (0, ("alpha", "1")),
        ]
        expected = Counter(
            (plan_ordinal, canonical_entity_key(entity))
            for plan_ordinal, entity in events
        )
        connection = _new_entity_connection()
        try:
            counter = TupleEntityCountCache(
                lambda rows: flush_entity_counts_to_sqlite(connection, rows),
                cache_limit=3,
            )
            for plan_ordinal, entity in events:
                counter.increment(plan_ordinal, entity)
            counter.flush()
            actual = {
                (int(plan), bytes(key)): int(count)
                for plan, key, count in connection.execute(
                    "SELECT plan_ordinal, entity_key, row_count FROM entities"
                )
            }
            self.assertEqual(actual, dict(expected))
            self.assertEqual(counter.peak_entries, 3)
            self.assertEqual(counter.flush_count, 2)
        finally:
            connection.close()

    def test_encoding_work_scales_with_unique_entities_not_rows(self) -> None:
        encode_calls = 0

        def recording_encoder(entity):
            nonlocal encode_calls
            encode_calls += 1
            return canonical_entity_key(entity)

        flushed = []
        counter = TupleEntityCountCache(
            lambda rows: flushed.extend(rows),
            cache_limit=65_536,
            encode_entity=recording_encoder,
        )
        row_count = 10_000
        unique_entities = 50
        for index in range(row_count):
            counter.increment(0, (f"entity-{index % unique_entities:03d}",))
        counter.flush()

        self.assertEqual(encode_calls, unique_entities)
        self.assertEqual(counter.encoded_key_count, unique_entities)
        self.assertEqual(sum(count for _plan, _key, count in flushed), row_count)
        self.assertEqual(len(flushed), unique_entities)
        self.assertEqual(row_count // encode_calls, 200)

    def test_fresh_projection_has_ninefold_deterministic_access_reduction(self) -> None:
        rows = 4_850_000
        plan = _Plan(
            tag_names=tuple(f"entity_{index}" for index in range(7)),
            fields=tuple(_Field(f"value_{index}") for index in range(57)),
        )
        projection = planning_projection(plan)
        legacy_projected_cells = rows * len(projection.legacy_query_columns)
        minimal_projected_cells = rows * len(projection.query_columns)

        self.assertEqual(legacy_projected_cells, 310_400_000)
        self.assertEqual(minimal_projected_cells, 33_950_000)
        self.assertGreaterEqual(
            legacy_projected_cells // minimal_projected_cells, 9
        )

    def test_invalid_cache_limit_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cache_limit must be positive"):
            TupleEntityCountCache(lambda _rows: None, cache_limit=0)


if __name__ == "__main__":
    unittest.main()
