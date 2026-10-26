"""Helpers to record benchmark rows with optional round-trip loss metrics."""

from __future__ import annotations

from typing import Callable, Sequence, Union

RecordFn = Callable[..., None]
BenchNumbers = Sequence[Union[int, float]]


def record_bench(record_fn: RecordFn, algo: str, result: BenchNumbers) -> None:
    o = int(result[0])
    c = int(result[1])
    te = int(result[2])
    td = int(result[3])
    if len(result) >= 6:
        record_fn(algo, o, c, te, td, float(result[4]), float(result[5]))
    else:
        record_fn(algo, o, c, te, td, 0.0, 0.0)
