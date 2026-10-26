#!/usr/bin/env python3
"""
Precompute Dataset page overview + pie-chart aggregates (``summary``) and the
classification table payload (``items`` + ``paper_usage``).

Writes under ``backend/dataset_result/``:

- ``catalog_public_full.json`` — ``GET /api/datasets/?part=summary``
- ``catalog_public_items.json`` — ``GET /api/datasets/?part=items``

Overall compression rate (summary card + dtype/category overall pies) is derived
from Java ``*_compression_result.csv`` rows: int → ``TS_2DIFF``, float →
``GORILLA``, string → ``lz4-framed``. A dataset is included only when every
exportable column has the matching ``compressedSize``; datasets without a
complete Java report are excluded from overall denominators. Average rate
(int/float ``TS_2DIFF``, string ``zstd_level_3``) and TsFile size stats are
unchanged.

From repository root::

    python scripts/update_dataset_catalog_result.py
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_BACKEND = _REPO / "backend"


def _make_progress_printer(phase: str) -> Callable[[str, int, int, str], None]:
    phase_started = time.perf_counter()
    dataset_started = time.perf_counter()

    def _on_dataset(key: str, index: int, total: int, label: str) -> None:
        nonlocal dataset_started
        now = time.perf_counter()
        elapsed = now - dataset_started
        total_elapsed = now - phase_started
        print(
            f"  [{label}] {index}/{total} {key} "
            f"({elapsed:.1f}s this dataset, {total_elapsed:.1f}s phase total)",
            flush=True,
        )
        dataset_started = now

    return _on_dataset


def main() -> int:
    if str(_BACKEND.resolve()) not in sys.path:
        sys.path.insert(0, str(_BACKEND.resolve()))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    import django

    django.setup()

    from myapp.dataset_catalog_compression_rate import clear_report_lookup_cache
    from myapp.dataset_catalog_result_store import (
        CATALOG_ITEMS_VERSION,
        CATALOG_RESULT_VERSION,
        CATALOG_SUMMARY_OVERVIEW_RATE_KEYS,
        validate_catalog_summary,
        write_precomputed_catalog_items,
        write_precomputed_catalog_summary,
    )
    from myapp.views import DatasetCatalogView, _sanitize_for_json

    view = DatasetCatalogView()
    t0 = time.perf_counter()
    print(
        f"Building public catalog summary (schema v{CATALOG_RESULT_VERSION})…",
        flush=True,
    )
    clear_report_lookup_cache()
    summary_payload = _sanitize_for_json(
        view._build_catalog_payload(
            lite=False,
            include_user_uploads=False,
            user=None,
            build_summary=True,
            build_items=False,
            progress_label='summary',
            on_dataset_progress=_make_progress_printer('summary'),
        ),
    )
    summary = summary_payload.get("summary") or {}
    errors = validate_catalog_summary(summary)
    if errors:
        print("Summary validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    summary_path = write_precomputed_catalog_summary(summary=summary)
    summary_elapsed = time.perf_counter() - t0
    print(f"Wrote {summary_path}", flush=True)
    print(f"  dataset_count={summary.get('dataset_count')}", flush=True)
    for key in CATALOG_SUMMARY_OVERVIEW_RATE_KEYS:
        print(f"  {key}={summary.get(key)}", flush=True)
    for key in (
        "category_overall_compression_rate_distribution",
        "category_average_compression_rate_distribution",
        "dtype_overall_compression_rate_distribution",
        "dtype_average_compression_rate_distribution",
    ):
        print(f"  {key}={len(summary.get(key) or [])} rows", flush=True)
    print(f"Summary done in {summary_elapsed:.1f}s", flush=True)

    t1 = time.perf_counter()
    print(
        f"Building public catalog items (schema v{CATALOG_ITEMS_VERSION})…",
        flush=True,
    )
    clear_report_lookup_cache()
    items_payload = _sanitize_for_json(
        view._build_catalog_payload(
            lite=False,
            include_user_uploads=False,
            user=None,
            build_summary=False,
            build_items=True,
            progress_label='items',
            on_dataset_progress=_make_progress_printer('items'),
        ),
    )
    items = items_payload.get("items") or []
    paper_usage = items_payload.get("paper_usage") or []
    items_path = write_precomputed_catalog_items(items=items, paper_usage=paper_usage)
    items_elapsed = time.perf_counter() - t1
    print(f"Wrote {items_path}", flush=True)
    print(f"  items={len(items)} paper_usage={len(paper_usage)}", flush=True)
    print(f"Items done in {items_elapsed:.1f}s", flush=True)
    print(f"Total done in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
