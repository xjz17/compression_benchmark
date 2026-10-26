"""Thin CLI wrapper; logic lives in ``myapp.home_leaderboard_boxplot_data``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_BACKEND = _REPO / "backend"


def _make_progress_printer() -> Callable[[str, int, int, float], None]:
    phase_started = time.perf_counter()

    def _on_dataset(slug: str, index: int, total: int, elapsed: float) -> None:
        total_elapsed = time.perf_counter() - phase_started
        print(
            f"  [{index}/{total}] {slug} "
            f"({elapsed:.1f}s this dataset, {total_elapsed:.1f}s total)",
            flush=True,
        )

    return _on_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute Home leaderboard boxplot distributions and Pareto chart "
            "(float Java reports across catalog datasets)."
        ),
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=_BACKEND,
        help="Path to backend/ (default: repo backend/)",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=None,
        metavar="R",
        help="Skip dataset/algorithm points with rate < R (default: script constant; use 0 to disable)",
    )
    args = parser.parse_args()
    backend = args.backend.resolve()

    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    import django

    django.setup()

    from myapp.benchmark_boxplot_result_store import (
        BOXPLOT_RESULT_VERSION,
        write_precomputed_home_leaderboard_boxplots,
    )
    from myapp.home_leaderboard_boxplot_data import (
        MIN_RATE_FOR_BOX_PLOT,
        build_home_leaderboard_boxplot_groups,
        clear_java_float_report_cache,
    )

    if args.min_rate is not None:
        min_rate = None if args.min_rate <= 0 else args.min_rate
    else:
        min_rate = MIN_RATE_FOR_BOX_PLOT

    t0 = time.perf_counter()
    print(
        f"Building Home leaderboard boxplots (schema v{BOXPLOT_RESULT_VERSION})…",
        flush=True,
    )
    clear_java_float_report_cache()
    groups, pareto_chart, datasets_used = build_home_leaderboard_boxplot_groups(
        backend,
        min_rate=min_rate,
        on_dataset_progress=_make_progress_printer(),
    )
    out_path = write_precomputed_home_leaderboard_boxplots(
        groups=groups,
        pareto_chart=pareto_chart,
        meta_extra={
            'datasets_used': datasets_used,
            'min_rate': min_rate,
        },
    )
    elapsed = time.perf_counter() - t0
    print(f"Wrote {out_path}", flush=True)
    print(f"  datasets_used={datasets_used}", flush=True)
    print(f"  min_rate={min_rate!r}", flush=True)
    for metric_key, group in groups.items():
        charts = group.get("charts") or {}
        for chart_key, chart in charts.items():
            algo_count = len(chart.get("algorithms") or [])
            point_count = sum(len(row) for row in (chart.get("values") or []))
            print(
                f"  {metric_key}/{chart_key}: {algo_count} algorithms, {point_count} points",
                flush=True,
            )
    pareto_points = pareto_chart.get("points") or []
    pareto_count = sum(1 for p in pareto_points if p.get("pareto"))
    level_series = pareto_chart.get("level_series") or []
    print(
        f"  pareto: {len(pareto_points)} points, {pareto_count} on frontier, "
        f"{len(level_series)} level trajectories",
        flush=True,
    )
    print(f"Done in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
