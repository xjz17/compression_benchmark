#!/usr/bin/env python3
"""Export a self-contained, public compression benchmark snapshot for HF Spaces.

The exporter reads the existing benchmark result CSVs through the same parsing,
sanitization, family-label, and float-precision helpers used by the Django app.
It writes only aggregate measurements and public dataset metadata; raw datasets,
uploaded user data, credentials, logs, and executable artifacts are excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


EXPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPORT_DIR.parents[1]
BACKEND_DIR = REPO_ROOT / "backend"

SOURCE_PATHS = {
    "paper_table3": REPO_ROOT / "backend/myapp/paper_table3_snapshot.json",
    "dataset_scope": REPO_ROOT / "benchmark/reference/datasets/ledger_public_compress_summary.json",
    "dataset_catalog": REPO_ROOT / "backend/dataset_result/catalog_public_items.json",
    "method_pipelines": REPO_ROOT / "benchmark/reference/methods/method_operator_pipelines.json",
}

SCOPES = (
    {"key": "integer", "label": "Integer", "data_type": "int", "float_precision": "all"},
    {"key": "float", "label": "All float", "data_type": "float", "float_precision": "all"},
    {"key": "fixed_float", "label": "Fixed-precision float", "data_type": "float", "float_precision": "fixed"},
    {"key": "nonfixed_float", "label": "Non-fixed-precision float", "data_type": "float", "float_precision": "non_fixed"},
    {"key": "overall", "label": "All numeric", "data_type": "overall", "float_precision": "all"},
)

RUNTIMES = ("cpp", "java", "python")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _finite(value: Any, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"Invalid numeric value: {value!r}")
    return number


def _timing_stats_from_report(rows: Any) -> dict[str, Any]:
    """Return a column mean or an explicit whole-dataset fallback."""
    import pandas as pd

    if rows is None or rows.empty:
        return {"compression": None, "decompression": None, "column_count": 0,
                "aggregation": "unavailable"}
    work = rows.copy()
    if work.get("_timing_source_type") is None:
        work["_timing_source_type"] = "single_report"
    column_groups = []
    aggregate_groups = []
    for source_type, source_rows in work.groupby("_timing_source_type", sort=True):
        columns = source_rows[pd.to_numeric(source_rows["column_index"], errors="coerce") >= 0]
        if not columns.empty:
            grouped = columns.groupby("column_index", as_index=False).agg(
                originalSize=("originalSize", "sum"),
                compressionTime=("compressionTime", "sum"),
                decompressionTime=("decompressionTime", "sum"),
            )
            grouped["_timing_source_type"] = source_type
            column_groups.append(grouped)
        else:
            aggregate = source_rows[pd.to_numeric(source_rows["column_index"], errors="coerce") < 0]
            if not aggregate.empty:
                aggregate_groups.append(aggregate)
    if column_groups and not aggregate_groups:
        grouped = pd.concat(column_groups, ignore_index=True)
        points = grouped["originalSize"] / 8.0
        return {"compression": float((grouped["compressionTime"] / points).mean()),
                "decompression": float((grouped["decompressionTime"] / points).mean()),
                "column_count": int(len(grouped)), "aggregation": "column_mean"}
    elapsed_parts = [*column_groups, *aggregate_groups]
    if elapsed_parts:
        # Use real columns for a typed report when present; use its -1 aggregate
        # only when that typed report lacks column rows, avoiding double count.
        elapsed = pd.concat(elapsed_parts, ignore_index=True)
        points = float(elapsed["originalSize"].sum() / 8.0)
        if points > 0:
            return {"compression": float(elapsed["compressionTime"].sum()) / points,
                    "decompression": float(elapsed["decompressionTime"].sum()) / points,
                    "column_count": 0, "aggregation": "dataset_aggregate"}
    return {"compression": None, "decompression": None, "column_count": 0,
            "aggregation": "unavailable"}


@lru_cache(maxsize=None)
def _algorithm_metadata(algorithm_key: str) -> tuple[bool, str | None, bool]:
    from myapp.home_leaderboard_boxplot_data import (
        boxplot_group_label,
        is_home_leaderboard_algorithm_enabled,
        is_lossy_boxplot_label,
    )

    family = boxplot_group_label(algorithm_key)
    return (
        is_home_leaderboard_algorithm_enabled(algorithm_key),
        family,
        is_lossy_boxplot_label(family or ""),
    )


def _public_datasets(catalog_payload: dict[str, Any], dataset_keys: list[str]) -> list[dict[str, Any]]:
    by_key = {str(item["key"]): item for item in catalog_payload.get("items", [])}
    if set(by_key) != set(dataset_keys):
        raise RuntimeError(
            "Dataset catalog and frozen 19-dataset scope differ: "
            f"catalog_only={sorted(set(by_key) - set(dataset_keys))}, "
            f"scope_only={sorted(set(dataset_keys) - set(by_key))}"
        )
    fields = (
        "key",
        "category",
        "row_count",
        "column_count",
        "numeric_column_count",
        "size_bytes",
        "prebuilt_tsfile_size_bytes",
        "source",
        "source_url",
    )
    return [{field: by_key[key].get(field) for field in fields} for key in dataset_keys]


def _paper_rows(snapshot: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    scope_keys = [str(value) for value in snapshot.get("scopes", [])]
    expected_scope_keys = ["integer", "fixed_float", "nonfixed_float", "overall"]
    if scope_keys != expected_scope_keys:
        raise RuntimeError(f"Unexpected paper snapshot scopes: {scope_keys}")

    methods: list[str] = []
    rows: list[dict[str, Any]] = []
    metric_fields = (
        ("rank", "rank"),
        ("average_rate", "average_compression_rate"),
        ("average_compression_ns", "average_compression_time_ns_per_point"),
        ("average_decompression_ns", "average_decompression_time_ns_per_point"),
        ("median_balance_score", "median_balanced_score"),
    )
    for source_row in snapshot.get("rows", []):
        method = str(source_row["method"])
        methods.append(method)
        for scope_index, scope_key in enumerate(scope_keys):
            row: dict[str, Any] = {"scope": scope_key, "method": method}
            for source_field, output_field in metric_fields:
                values = source_row.get(source_field, [])
                if len(values) != len(scope_keys):
                    raise RuntimeError(f"{method}.{source_field} does not cover all scopes")
                row[output_field] = _finite(values[scope_index], positive=True)
            rows.append(row)
    if len(methods) != len(set(methods)) or len(methods) != 20:
        raise RuntimeError(f"Expected 20 unique public methods, found {len(methods)}")
    return methods, rows


def _configuration_rows(
    *,
    dataset_key: str,
    scope: dict[str, str],
    runtime: str,
) -> list[dict[str, Any]]:
    import pandas as pd

    from myapp.dataset_float_precision_distribution import filter_report_dataframe_for_float_precision
    from myapp.home_leaderboard_boxplot_data import (
        aggregate_report_rows_by_algorithm,
        read_benchmark_report,
    )

    bundle_dir = BACKEND_DIR / "dataset" / dataset_key
    if scope["data_type"] == "overall":
        report_parts = []
        for report_type in ("int", "float"):
            part = read_benchmark_report(
                bundle_dir, dataset_key, data_type=report_type, implement_language=runtime
            )
            if part is not None and not part.empty:
                part = part.copy()
                part["_timing_source_type"] = report_type
                report_parts.append(part)
        report = pd.concat(report_parts, ignore_index=True) if report_parts else None
    else:
        report = read_benchmark_report(
            bundle_dir,
            dataset_key,
            data_type=scope["data_type"],
            implement_language=runtime,
        )
        if report is not None and not report.empty:
            report = report.copy()
            report["_timing_source_type"] = scope["data_type"]
    if report is None or report.empty:
        return []
    if scope["data_type"] == "float":
        report = filter_report_dataframe_for_float_precision(
            report,
            column_info_path=str(bundle_dir / f"{dataset_key}_column_info.csv"),
            bundle_dir=str(bundle_dir),
            float_precision=scope["float_precision"],
        )
    if report is None or report.empty:
        return []

    work = report.copy()
    algorithm_keys = [str(value) for value in work["algorithm"].dropna().unique()]
    enabled_by_key = {
        key: _algorithm_metadata(key)[0]
        for key in algorithm_keys
    }
    family_by_key = {key: _algorithm_metadata(key)[1] for key in algorithm_keys}
    lossy_by_key = {key: _algorithm_metadata(key)[2] for key in algorithm_keys}
    work = work[work["algorithm"].astype(str).map(enabled_by_key).fillna(False)].copy()
    if work.empty:
        return []
    work["family"] = work["algorithm"].astype(str).map(family_by_key)
    work = work[
        work["family"].notna()
        & ~work["algorithm"].astype(str).map(lossy_by_key).fillna(False)
    ].copy()
    if work.empty:
        return []
    for field in ("originalSize", "compressedSize", "compressionTime", "decompressionTime"):
        work[field] = pd.to_numeric(work[field], errors="coerce")
    work = work[
        (work["originalSize"] > 0)
        & (work["compressedSize"] > 0)
        & (work["compressionTime"] >= 0)
        & (work["decompressionTime"] >= 0)
    ].copy()
    if work.empty:
        return []

    aggregate = aggregate_report_rows_by_algorithm(work)
    aggregate["family"] = aggregate["algorithm"].astype(str).map(family_by_key)
    aggregate = aggregate[aggregate["family"].notna()].copy()
    aggregate["compression_rate"] = aggregate["compressedSize"] / aggregate["originalSize"]
    best_indexes = aggregate.groupby("family", sort=False)["compression_rate"].idxmin()
    best_algorithms = set(aggregate.loc[best_indexes, "algorithm"].astype(str))

    rows: list[dict[str, Any]] = []
    for record in aggregate.itertuples(index=False):
        original_bytes = _finite(record.originalSize, positive=True)
        compressed_bytes = _finite(record.compressedSize, positive=True)
        points = original_bytes / 8.0
        timing_rows = work[
            work["algorithm"].astype(str) == str(record.algorithm)
        ].copy()
        timing = _timing_stats_from_report(timing_rows)
        rows.append(
            {
                "dataset": dataset_key,
                "scope": scope["key"],
                "runtime": runtime,
                "method": str(record.algorithm),
                "family": str(record.family),
                "algorithm_variant": str(record.algorithm),
                "is_best_rate_in_family": str(record.algorithm) in best_algorithms,
                "original_size_bytes": int(round(original_bytes)),
                "compressed_size_bytes": int(round(compressed_bytes)),
                "point_count": int(round(points)),
                "compression_time_ns": _finite(record.compressionTime),
                "decompression_time_ns": _finite(record.decompressionTime),
                "compression_rate": compressed_bytes / original_bytes,
                "compression_time_ns_per_point": timing["compression"],
                "decompression_time_ns_per_point": timing["decompression"],
                "timing_column_count": timing["column_count"],
                "timing_aggregation": timing["aggregation"],
            }
        )
    return sorted(rows, key=lambda row: (row["method"], row["algorithm_variant"]))


def _aggregate_current_rows(dataset_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configuration_aggregates: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    configurations = sorted(
        {
            (row["runtime"], row["scope"], row["family"], row["method"])
            for row in dataset_rows
        }
    )
    for runtime, scope_key, family, method in configurations:
        rows = [
            row
            for row in dataset_rows
            if row["runtime"] == runtime
            and row["scope"] == scope_key
            and row["method"] == method
        ]
        configuration_aggregates.append(
            _aggregate_rows(
                rows,
                runtime=runtime,
                scope_key=scope_key,
                method=method,
                family=family,
                algorithm_variant=method,
            )
        )

    families = sorted(
        {(row["runtime"], row["scope"], row["family"]) for row in dataset_rows}
    )
    for runtime, scope_key, family in families:
            rows = [
                row
                for row in dataset_rows
                if row["runtime"] == runtime
                and row["scope"] == scope_key
                and row["family"] == family
                and row["is_best_rate_in_family"]
            ]
            if not rows:
                continue
            aggregates.append(
                _aggregate_rows(
                    rows,
                    runtime=runtime,
                    scope_key=scope_key,
                    method=family,
                    family=family,
                    algorithm_variant=None,
                )
            )
    return configuration_aggregates, aggregates


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    runtime: str,
    scope_key: str,
    method: str,
    family: str,
    algorithm_variant: str | None,
) -> dict[str, Any]:
    original_sum = sum(row["original_size_bytes"] for row in rows)
    compressed_sum = sum(row["compressed_size_bytes"] for row in rows)
    point_sum = sum(row["point_count"] for row in rows)
    elapsed_rows = [
        row for row in rows
        if row.get("point_count", 0) > 0
        and row.get("compression_time_ns") is not None
        and row.get("decompression_time_ns") is not None
    ]
    elapsed_point_sum = sum(row["point_count"] for row in elapsed_rows)
    comp_time_sum = sum(row["compression_time_ns"] for row in elapsed_rows)
    decomp_time_sum = sum(row["decompression_time_ns"] for row in elapsed_rows)
    elapsed_complete = len(elapsed_rows) == len(rows)
    timing_column_count = sum(row.get("timing_column_count", 0) for row in rows)
    modes = [row.get("timing_aggregation", "unavailable") for row in rows]
    valid_times = all(
        row.get("compression_time_ns_per_point") is not None
        and row.get("decompression_time_ns_per_point") is not None
        for row in rows
    )
    if not valid_times or "unavailable" in modes:
        average_compression_time = average_decompression_time = None
        timing_aggregation = "unavailable"
    elif "dataset_aggregate" in modes:
        average_compression_time = sum(row["compression_time_ns_per_point"] for row in rows) / len(rows)
        average_decompression_time = sum(row["decompression_time_ns_per_point"] for row in rows) / len(rows)
        timing_aggregation = "dataset_mean"
    elif timing_column_count:
        average_compression_time = sum(
            row["compression_time_ns_per_point"] * row["timing_column_count"] for row in rows
        ) / timing_column_count
        average_decompression_time = sum(
            row["decompression_time_ns_per_point"] * row["timing_column_count"] for row in rows
        ) / timing_column_count
        timing_aggregation = "column_mean"
    else:
        average_compression_time = average_decompression_time = None
        timing_aggregation = "unavailable"
    return {
        "runtime": runtime,
        "scope": scope_key,
        "method": method,
        "family": family,
        "algorithm_variant": algorithm_variant,
        "dataset_count": len(rows),
        "overall_compression_rate": compressed_sum / original_sum,
        "average_compression_rate": sum(row["compression_rate"] for row in rows) / len(rows),
        "weighted_compression_time_ns_per_point": (
            comp_time_sum / elapsed_point_sum if elapsed_point_sum and elapsed_complete else None
        ),
        "average_compression_time_ns_per_point": average_compression_time,
        "weighted_decompression_time_ns_per_point": (
            decomp_time_sum / elapsed_point_sum if elapsed_point_sum and elapsed_complete else None
        ),
        "average_decompression_time_ns_per_point": average_decompression_time,
        "timing_column_count": timing_column_count,
        "timing_aggregation": timing_aggregation,
        "timing_dataset_count": len(rows),
        "original_size_bytes": original_sum,
        "compressed_size_bytes": compressed_sum,
        "point_count": point_sum,
    }


def _source_status(path: Path) -> str:
    relpath = path.relative_to(REPO_ROOT).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relpath],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not tracked:
        return "generated_or_ignored"
    return "tracked_modified" if _git("status", "--porcelain", "--", relpath) else "tracked_clean"


def main() -> int:
    sys.path.insert(0, str(BACKEND_DIR))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    import django

    django.setup()

    paper_snapshot = _read_json(SOURCE_PATHS["paper_table3"])
    dataset_scope = _read_json(SOURCE_PATHS["dataset_scope"])
    dataset_catalog = _read_json(SOURCE_PATHS["dataset_catalog"])
    method_pipelines = _read_json(SOURCE_PATHS["method_pipelines"])

    dataset_keys = [str(value) for value in dataset_scope["meta"]["dataset_keys"]]
    if len(dataset_keys) != 19 or len(dataset_keys) != len(set(dataset_keys)):
        raise RuntimeError(f"Expected 19 unique datasets, found {len(dataset_keys)}")
    methods, paper_rows = _paper_rows(paper_snapshot)
    datasets = _public_datasets(dataset_catalog, dataset_keys)

    dataset_rows: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        for scope in SCOPES:
            for runtime in RUNTIMES:
                dataset_rows.extend(
                    _configuration_rows(
                        dataset_key=dataset_key,
                        scope=scope,
                        runtime=runtime,
                    )
                )
    configuration_aggregates, family_best_aggregates = _aggregate_current_rows(dataset_rows)
    # Retain the source elapsed-time and point totals. A per-column arithmetic
    # mean cannot be expanded back into elapsed time from a dataset point total,
    # so later scope transforms must use these measured totals rather than infer
    # them from the displayed mean.

    families = sorted({row["family"] for row in dataset_rows})
    configurations = sorted(
        {(row["runtime"], row["method"], row["family"]) for row in dataset_rows}
    )
    method_definitions = sorted({(row["method"], row["family"]) for row in dataset_rows})
    report_availability = []
    report_paths_used: set[str] = set()
    missing_report_count = 0
    for dataset_key in dataset_keys:
        for runtime in RUNTIMES:
            available_types = []
            for data_type in ("int", "float"):
                path = BACKEND_DIR / "dataset" / dataset_key / f"{dataset_key}_{data_type}_{runtime}_compression_result.csv"
                if path.is_file():
                    available_types.append(data_type)
                    report_paths_used.add(path.relative_to(REPO_ROOT).as_posix())
                else:
                    missing_report_count += 1
            report_availability.append(
                {
                    "dataset": dataset_key,
                    "runtime": runtime,
                    "available_data_types": available_types,
                }
            )
    source_files = [
        {
            "role": role,
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(path),
            "status": _source_status(path),
        }
        for role, path in SOURCE_PATHS.items()
    ]
    report_files = [
        {
            "path": relpath,
            "size_bytes": (REPO_ROOT / relpath).stat().st_size,
            "sha256": _sha256(REPO_ROOT / relpath),
        }
        for relpath in sorted(report_paths_used)
    ]

    pipeline_map = method_pipelines.get("pipelines", {})
    method_rows = [
        {
            "name": method,
            "family": family,
            "runtimes": sorted(
                runtime
                for runtime, configuration, configuration_family in configurations
                if configuration == method and configuration_family == family
            ),
            "operator_pipeline": pipeline_map.get(family),
            "operator_pipeline_status": (
                "interpretive_coarse_mapping" if family in pipeline_map else "not_available"
            ),
        }
        for method, family in method_definitions
    ]

    scope_summary = dataset_scope["summary"]
    current_cpp_family_keys = {
        (row["scope"], row["family"])
        for row in family_best_aggregates
        if row["runtime"] == "cpp"
    }
    paper_keys = {(row["scope"], row["method"]) for row in paper_rows}
    dirty_inputs = [
        row["path"] for row in source_files if row["status"] == "tracked_modified"
    ]
    data = {
        "schema_version": 1,
        "benchmark": {
            "title": "THULab Time Series Compression Benchmark",
            "implementation_runtimes": list(RUNTIMES),
            "lossless_design_methods_only": True,
            "current_result_selection": "visible configurations of methods categorized as lossless-design by the project; result-level float round trips are not established",
            "paper_result_selection": "the separate paper_table3 section contains its frozen 20-method C++ snapshot",
            "scope_note": "Current result rows cover numeric int/float report evidence for the retained 19 public datasets. They reflect the columns present in each report; they do not assert full-column coverage beyond those files.",
            "aggregation_note": "Dataset rows with column evidence use an arithmetic mean of per-column nanoseconds per point. Rows backed only by whole-report timing use source dataset elapsed time divided by source point totals and retain a zero column count. If any selected row uses that fallback, method timing is an equal-dataset arithmetic mean; otherwise it is pooled equally over measured columns.",
            "selection_note": "Configuration views retain measured configurations passing the project's method-name filter. This does not certify every result as lossless. Family views choose the lowest-rate configuration independently for each dataset, scope, and runtime.",
            "notes": [
                "Missing combinations are unavailable rather than zero.",
                "Hardware identity and round-trip verification are not reported by these source CSVs.",
                "The frozen paper table is labeled separately from the current report-tree recomputation.",
            ],
            "metric_definitions": {
                "compression_rate": "compressed bytes divided by original numeric bytes; lower is better",
                "average_compression_rate": "arithmetic mean of per-dataset compression rates",
                "overall_compression_rate": "sum of compressed bytes divided by sum of original numeric bytes",
                "average_time_ns_per_point": "column-backed rows use an arithmetic mean of per-column nanoseconds per numeric point; whole-report-only rows use source elapsed nanoseconds divided by source point totals; legacy CSV rows retain their 8-byte-equivalent point denominator",
                "weighted_time_ns_per_point": "sum of measured nanoseconds divided by sum of 8-byte numeric points",
                "paper_median_balanced_score": "stored paper snapshot value; normalization is paper-scope specific and lower is better",
            },
            "measurement_boundaries": [
                "No hardware identity is attached because the selected source files do not provide one.",
                "No round-trip verification claim is exported from timing/result CSVs alone.",
                "Family-level aggregates select the measured variant with the lowest compression rate within each dataset and method family, matching the backend family-selection rule.",
                "dataset_results contains every visible measured configuration; is_best_rate_in_family marks the configuration used for family-level aggregation.",
                "Whole-report timing rows (column_index=-1) remain dataset-level timing evidence with timing_column_count=0; they are never treated as columns.",
                "Paper Table 3 values and current report recomputations are separate result sets because the report tree has evolved since the paper snapshot was frozen.",
            ],
        },
        "source": {
            "generated_at": dataset_scope.get("generated_at") or dataset_scope["meta"].get("generated_at"),
            "git_commit": _git("rev-parse", "HEAD"),
            "source_files": source_files,
            "worktree_dirty_inputs": dirty_inputs,
            "result_csv_pattern": "backend/dataset/<dataset>/<dataset>_<int|float>_<cpp|java|python>_compression_result.csv",
            "result_csv_files_used": report_files,
        },
        "coverage": {
            "dataset_count": len(dataset_keys),
            "runtime_count": len(RUNTIMES),
            "runtimes": list(RUNTIMES),
            "method_family_count": len(families),
            "method_configuration_count": len(method_definitions),
            "runtime_configuration_count": len(configurations),
            "scope_count": len(SCOPES),
            "paper_leaderboard_row_count": len(paper_rows),
            "configuration_aggregate_row_count": len(configuration_aggregates),
            "family_best_aggregate_row_count": len(family_best_aggregates),
            "dataset_result_row_count": len(dataset_rows),
            "timing_column_observation_count": sum(row["timing_column_count"] for row in dataset_rows),
            "timing_column_mean_dataset_result_count": sum(
                row["timing_aggregation"] == "column_mean" for row in dataset_rows
            ),
            "timing_dataset_aggregate_result_count": sum(
                row["timing_aggregation"] == "dataset_aggregate" for row in dataset_rows
            ),
            "timing_unavailable_dataset_result_count": sum(
                row["timing_aggregation"] == "unavailable" for row in dataset_rows
            ),
            "source_report_file_count": len(report_files),
            "missing_source_report_count": missing_report_count,
            "excluded_datasets": dataset_scope["meta"].get("excluded_dataset_keys", []),
            "missing_measurement_policy": "Missing report and dataset/scope/runtime/configuration combinations are omitted; the UI must display them as unavailable, never as zero.",
        },
        "dataset_overview": {
            "dataset_count": scope_summary["dataset_count"],
            "total_rows": scope_summary["total_rows"],
            "total_size_bytes": scope_summary["total_size_bytes"],
            "total_compressed_size_bytes": scope_summary["total_compressed_size_bytes"],
            "overall_compression_rate": scope_summary["overall_compression_rate"],
            "average_compression_rate": scope_summary["average_compression_rate"],
            "compression_rate_used_column_count": dataset_scope["meta"].get("compression_rate_used_column_count"),
            "compression_rate_missing_column_count": dataset_scope["meta"].get("compression_rate_missing_column_count"),
            "compression_rate_missing_examples": dataset_scope["meta"].get("compression_rate_missing_examples", []),
            "provenance_note": dataset_scope["meta"].get("compression_rate_recompute"),
        },
        "scopes": list(SCOPES),
        "report_availability": report_availability,
        "datasets": datasets,
        "methods": method_rows,
        "paper_table3": {
            "source_note": paper_snapshot.get("source"),
            "comparison_to_current_reports": {
                "status": "separate_frozen_view",
                "exact_name_overlap_row_count": len(paper_keys & current_cpp_family_keys),
                "paper_row_count": len(paper_keys),
                "unmatched_paper_method_names": sorted(
                    {method for _scope, method in paper_keys - current_cpp_family_keys}
                ),
                "note": "The frozen table is not used to fill or overwrite current report values; method labels and measurements have evolved.",
            },
            "rows": paper_rows,
        },
        "configuration_aggregates": configuration_aggregates,
        "family_best_aggregates": family_best_aggregates,
        "dataset_results": dataset_rows,
    }

    data_path = EXPORT_DIR / "data.json"
    data_path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "data_file": "data.json",
        "data_sha256": _sha256(data_path),
        "counts": data["coverage"],
        "source_files": source_files,
        "result_csv_files": report_files,
        "excluded_content": [
            "raw datasets",
            "user uploads",
            "credentials and environment variables",
            "server logs",
            "binaries and model artifacts",
            "external or vendored benchmark results",
        ],
    }
    (EXPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
