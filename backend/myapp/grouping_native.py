"""Native Grouping layout benchmark.

Grouping stores a timestamp union shared by several typed series plus one
presence bitmap and per-column value streams.  The implementation is intentionally
self-contained so the paper's Grouping row is not a wrapper around the standalone
``analysis_outputs`` benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import time
from typing import Iterable, Sequence
from .experiment_data_policy import parse_experiment_float_cell


ROW_NAME = "grouping"
TIME_BYTES = 8
VALUE_BYTES = 8


@dataclass
class ColumnStream:
    timestamps: list[int]
    values: list[int]


@dataclass
class GroupedLayout:
    timestamps: list[int]
    bitmap: list[int]
    values: list[list[int]]


def _float_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _int_bits(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def group_layout(columns: Sequence[ColumnStream]) -> GroupedLayout:
    values = [[] for _ in columns]
    if not columns:
        return GroupedLayout([], [], values)
    if len(columns) == 1:
        return GroupedLayout(list(columns[0].timestamps), [], [list(columns[0].values)])

    positions = [0] * len(columns)
    timestamps: list[int] = []
    bitmap: list[int] = []
    while True:
        next_ts: int | None = None
        for idx, col in enumerate(columns):
            if positions[idx] < len(col.timestamps):
                ts = col.timestamps[positions[idx]]
                next_ts = ts if next_ts is None or ts < next_ts else next_ts
        if next_ts is None:
            break
        row = len(timestamps)
        timestamps.append(next_ts)
        need_words = (len(timestamps) * len(columns) + 63) // 64
        while len(bitmap) < need_words:
            bitmap.append(0)
        for idx, col in enumerate(columns):
            pos = positions[idx]
            if pos < len(col.timestamps) and col.timestamps[pos] == next_ts:
                bit = row * len(columns) + idx
                bitmap[bit // 64] |= 1 << (bit % 64)
                values[idx].append(col.values[pos])
                positions[idx] += 1
    return GroupedLayout(timestamps, bitmap, values)


def recover_layout(layout: GroupedLayout) -> list[ColumnStream]:
    columns = [ColumnStream([], []) for _ in layout.values]
    if not layout.values:
        return columns
    if len(layout.values) == 1:
        return [ColumnStream(list(layout.timestamps), list(layout.values[0]))]

    positions = [0] * len(layout.values)
    for row, ts in enumerate(layout.timestamps):
        for idx in range(len(layout.values)):
            bit = row * len(layout.values) + idx
            if ((layout.bitmap[bit // 64] >> (bit % 64)) & 1) == 0:
                continue
            pos = positions[idx]
            columns[idx].timestamps.append(ts)
            columns[idx].values.append(layout.values[idx][pos])
            positions[idx] += 1
    return columns


def verify_round_trip(columns: Sequence[ColumnStream], layout: GroupedLayout) -> None:
    recovered = recover_layout(layout)
    if len(recovered) != len(columns):
        raise RuntimeError("grouping column count mismatch")
    for idx, (want, got) in enumerate(zip(columns, recovered)):
        if want.timestamps != got.timestamps or want.values != got.values:
            raise RuntimeError(f"grouping lossless check failed at column {idx}")


def grouped_size_bytes(layout: GroupedLayout) -> int:
    value_count = sum(len(col) for col in layout.values)
    bitmap_bytes = 0 if len(layout.values) <= 1 else (len(layout.timestamps) * len(layout.values) + 7) // 8
    return len(layout.timestamps) * TIME_BYTES + value_count * VALUE_BYTES + bitmap_bytes


def original_size_bytes(columns: Sequence[ColumnStream]) -> int:
    return sum(len(col.values) for col in columns) * (TIME_BYTES + VALUE_BYTES)


def bench_grouping_streams(columns: Sequence[ColumnStream]) -> tuple[int, int, int, int]:
    original = original_size_bytes(columns)
    if original <= 0:
        return 0, 0, 0, 0
    t0 = time.perf_counter_ns()
    layout = group_layout(columns)
    t1 = time.perf_counter_ns()
    recovered = recover_layout(layout)
    t2 = time.perf_counter_ns()
    if len(recovered) != len(columns):
        raise RuntimeError("grouping recovered column count mismatch")
    for idx, (want, got) in enumerate(zip(columns, recovered)):
        if want.timestamps != got.timestamps or want.values != got.values:
            raise RuntimeError(f"grouping lossless check failed at column {idx}")
    return original, grouped_size_bytes(layout), t1 - t0, t2 - t1


def streams_from_dense_columns(columns: Sequence[Sequence[int | float]], *, data_type: str) -> list[ColumnStream]:
    streams: list[ColumnStream] = []
    for col in columns:
        timestamps = list(range(len(col)))
        if data_type == "int":
            values = [_int_bits(int(v)) for v in col]
        else:
            values = [_float_bits(float(v)) for v in col if math.isfinite(float(v))]
            if len(values) != len(timestamps):
                timestamps = [i for i, v in enumerate(col) if math.isfinite(float(v))]
        if values:
            streams.append(ColumnStream(timestamps, values))
    return streams


def streams_from_patch_cells(
    data_type: str,
    bench_columns: Sequence[tuple[str, int, int]],
    patch_cells: Sequence[Sequence[str]],
    *,
    row_offset: int = 0,
) -> list[ColumnStream]:
    streams = [ColumnStream([], []) for _ in bench_columns]
    for local_row, row in enumerate(patch_cells):
        ts = row_offset + local_row
        for slot, (_name, _max_point, col_idx) in enumerate(bench_columns):
            raw = row[col_idx] if col_idx < len(row) else ""
            text = str(raw).strip()
            if not text:
                continue
            try:
                if data_type == "int":
                    value = _int_bits(int(text))
                elif data_type == "float":
                    parsed = parse_experiment_float_cell(text)
                    if parsed is None:
                        continue
                    value = _float_bits(parsed)
                else:
                    continue
            except ValueError:
                continue
            streams[slot].timestamps.append(ts)
            streams[slot].values.append(value)
    return [stream for stream in streams if stream.values]
