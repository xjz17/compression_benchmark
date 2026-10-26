"""Accumulate compression benchmark metrics across CSV patches."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from .benchmark_loss import format_loss_rate_csv, is_lossy_flag, loss_rate

TotalsRow = Tuple[int, int, int, int]


class BenchmarkTotals:
    def __init__(self) -> None:
        self._by_algorithm: Dict[str, list] = {}
        self._loss_by_algorithm: Dict[str, list] = {}

    def merge(
        self,
        algorithm: str,
        original: int,
        compressed: int,
        comp_time: int,
        decomp_time: int,
        loss_abs_error: float = 0.0,
        loss_abs_ref: float = 0.0,
    ) -> None:
        if not algorithm or original <= 0:
            return
        acc = self._by_algorithm.setdefault(algorithm, [0, 0, 0, 0])
        acc[0] += int(original)
        acc[1] += int(compressed)
        acc[2] += int(comp_time)
        acc[3] += int(decomp_time)
        if loss_abs_error != 0.0 or loss_abs_ref != 0.0:
            loss = self._loss_by_algorithm.setdefault(algorithm, [0.0, 0.0])
            loss[0] += float(loss_abs_error)
            loss[1] += float(loss_abs_ref)

    def merge_tuple(self, algorithm: str, row: Tuple[int, ...]) -> None:
        if len(row) >= 4:
            if len(row) >= 6:
                self.merge(
                    algorithm,
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    float(row[4]),
                    float(row[5]),
                )
            else:
                self.merge(algorithm, row[0], row[1], row[2], row[3])

    def snapshot(self) -> Dict[str, TotalsRow]:
        return {
            k: (v[0], v[1], v[2], v[3])
            for k, v in self._by_algorithm.items()
            if len(v) >= 4 and v[0] > 0
        }

    def is_lossy(self, algorithm: str) -> int:
        loss = self._loss_by_algorithm.get(algorithm)
        sum_err = loss[0] if loss else 0.0
        sum_ref = loss[1] if loss else 0.0
        return is_lossy_flag(algorithm, sum_err, sum_ref)

    def loss_rate(self, algorithm: str) -> float:
        loss = self._loss_by_algorithm.get(algorithm)
        if not loss:
            return 0.0
        return loss_rate(loss[0], loss[1])

    def loss_rate_csv(self, algorithm: str) -> str:
        return format_loss_rate_csv(self.loss_rate(algorithm))

    def flush(
        self,
        append_row: Callable[..., None],
        should_run: Callable[[str], bool],
    ) -> None:
        for algo, (o, c, te, td) in sorted(self.snapshot().items()):
            if should_run(algo):
                append_row(algo, o, c, te, td, self.is_lossy(algo), self.loss_rate_csv(algo))
