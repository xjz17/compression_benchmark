#!/usr/bin/env python3
"""Isolated THULAB v5 planning fast-path helpers.

This module is intentionally not imported by the v4 producer.  It prototypes
two internal optimizations that leave the planning receipt semantics unchanged:

* query only columns needed to identify entities, enforce row filters, and
  validate admitted temporal conversions during the count-only pass; and
* keep entity tuples as dictionary keys until a bounded batch is flushed,
  avoiding canonical JSON serialization on every source row.

Before integrating the projection into a producer, its row-presence semantics
must be checked against the pinned TsFile SDK.  Some columnar readers omit rows
when none of the projected FIELD values exists, so a producer must prove that
the minimal projection returns the same rows as the legacy all-FIELD query.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import sqlite3
from typing import Callable, Iterable, Protocol, Sequence


DEFAULT_ENTITY_COUNT_CACHE_LIMIT = 65_536


class _TemporalLike(Protocol):
    source_name: str


class _FieldLike(Protocol):
    name: str
    temporal: _TemporalLike | None


class _RowFilterConditionLike(Protocol):
    source_name: str
    source_category: str


class _RowFilterLike(Protocol):
    conditions: Sequence[_RowFilterConditionLike]


class _SourcePlanLike(Protocol):
    tag_names: Sequence[str]
    fields: Sequence[_FieldLike]
    primary_temporal: _TemporalLike | None
    row_filter: _RowFilterLike | None


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class PlanningProjection:
    """Auditable column projection for the count-only planning pass.

    ``Time`` is not listed because the TsFile table result exposes it as the
    mandatory first coordinate independently of ``query_columns``.
    """

    query_columns: tuple[str, ...]
    tag_columns: tuple[str, ...]
    temporal_columns: tuple[str, ...]
    filter_columns: tuple[str, ...]
    omitted_native_fields: tuple[str, ...]
    legacy_query_columns: tuple[str, ...]

    @property
    def removed_column_count(self) -> int:
        return len(self.legacy_query_columns) - len(self.query_columns)


def planning_projection(plan: _SourcePlanLike) -> PlanningProjection:
    """Build the smallest count-only projection implied by a source plan.

    The legacy producer projects every retained FIELD even though native values
    are not packed during planning.  A count-only pass needs only:

    * every TAG, to construct the entity key;
    * every admitted temporal STRING/TEXT field, to preserve fail-closed parse
      validation (including a primary temporal coordinate);
    * every non-Time row-filter source, to preserve admission decisions.

    The returned audit also records the legacy projection and omitted native
    fields so integration code can fail closed and report what changed.
    """

    tags = _ordered_unique(plan.tag_names)
    fields = tuple(plan.fields)
    field_names = tuple(field.name for field in fields)
    temporal = [field.name for field in fields if field.temporal is not None]
    if plan.primary_temporal is not None:
        temporal.append(plan.primary_temporal.source_name)
    temporal_columns = _ordered_unique(temporal)

    filter_columns = _ordered_unique(
        condition.source_name
        for condition in (() if plan.row_filter is None else plan.row_filter.conditions)
        if condition.source_category.upper() != "TIME"
    )
    legacy_query_columns = _ordered_unique((*tags, *field_names, *filter_columns))
    query_columns = _ordered_unique((*tags, *temporal_columns, *filter_columns))
    required = set(query_columns)
    omitted_native_fields = tuple(
        field.name
        for field in fields
        if field.temporal is None and field.name not in required
    )
    return PlanningProjection(
        query_columns=query_columns,
        tag_columns=tags,
        temporal_columns=temporal_columns,
        filter_columns=filter_columns,
        omitted_native_fields=omitted_native_fields,
        legacy_query_columns=legacy_query_columns,
    )


def canonical_entity_key(entity: Sequence[str]) -> bytes:
    """Encode an entity exactly as the v4 producer's canonical JSON helper."""

    return json.dumps(
        list(entity),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


EncodedEntityCount = tuple[int, bytes, int]
FlushRows = Callable[[Iterable[EncodedEntityCount]], None]
EntityEncoder = Callable[[Sequence[str]], bytes]


def flush_entity_counts_to_sqlite(
    connection: sqlite3.Connection,
    rows: Iterable[EncodedEntityCount],
) -> None:
    """Upsert one deferred entity-count batch using the v4 SQL semantics."""

    connection.executemany(
        """
        INSERT INTO entities(plan_ordinal, entity_key, row_count)
        VALUES (?, ?, ?)
        ON CONFLICT(plan_ordinal, entity_key)
        DO UPDATE SET row_count = row_count + excluded.row_count
        """,
        rows,
    )
    connection.commit()


class TupleEntityCountCache:
    """Bounded entity counter that serializes once per pending unique entity.

    The v4 key is ``(plan_ordinal, canonical_json(entity))`` and therefore
    canonicalizes on every row.  Equal tuples already have stable Python value
    semantics, so this cache defers encoding until ``flush`` while keeping the
    same SQLite key bytes and additive upsert behavior.
    """

    def __init__(
        self,
        flush_rows: FlushRows,
        *,
        cache_limit: int = DEFAULT_ENTITY_COUNT_CACHE_LIMIT,
        encode_entity: EntityEncoder = canonical_entity_key,
    ) -> None:
        if cache_limit <= 0:
            raise ValueError("cache_limit must be positive")
        self._flush_rows = flush_rows
        self._cache_limit = cache_limit
        self._encode_entity = encode_entity
        self.pending: OrderedDict[tuple[int, tuple[str, ...]], int] = OrderedDict()
        self.peak_entries = 0
        self.flush_count = 0
        self.encoded_key_count = 0

    def increment(self, plan_ordinal: int, entity: Sequence[str]) -> None:
        key = (plan_ordinal, tuple(entity))
        self.pending[key] = self.pending.get(key, 0) + 1
        self.peak_entries = max(self.peak_entries, len(self.pending))
        if len(self.pending) >= self._cache_limit:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        pending_size = len(self.pending)
        rows = (
            (plan_ordinal, self._encode_entity(entity), row_count)
            for (plan_ordinal, entity), row_count in self.pending.items()
        )
        self._flush_rows(rows)
        self.encoded_key_count += pending_size
        self.flush_count += 1
        self.pending.clear()

