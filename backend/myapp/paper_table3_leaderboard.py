"""Home leaderboard helpers for the paper Table 3 mode."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from django.conf import settings

from .dataset_float_precision_distribution import (
    filter_report_dataframe_for_float_precision,
    parse_float_precision_filter,
)
from .home_leaderboard_boxplot_data import (
    aggregate_report_rows_by_algorithm,
    best_metrics_from_report,
    boxplot_group_label,
    compute_balanced_score_metrics,
    is_home_leaderboard_algorithm_enabled,
    is_lossy_boxplot_label,
    read_benchmark_report,
)

PAPER_TABLE3_TOP_K = 20
PAPER_TABLE3_REFERENCE_METHOD = "REGER"
PAPER_TABLE3_EXCLUDED_FAMILIES = frozenset(
    {
        "BSS+LZ4",
        "CORAD",
        "Machete",
        "MOST",
        "Serf",
        "Sim-Piece",
        "Sub-column",
        "SZ",
        "SZ3",
        "ZFP",
    }
)

PAPER_TABLE3_SCOPES = {
    "integer": {
        "data_type": "int",
        "float_precision": "all",
        "key": "integer",
    },
    "fixed_float": {
        "data_type": "float",
        "float_precision": "fixed",
        "key": "fixed_float",
    },
    "nonfixed_float": {
        "data_type": "float",
        "float_precision": "non_fixed",
        "key": "nonfixed_float",
    },
    "overall": {
        "data_type": "overall",
        "float_precision": "all",
        "key": "overall",
    },
}

PAPER_TABLE3_SNAPSHOT_PATH = Path(__file__).resolve().parent / "paper_table3_snapshot.json"

_PAPER_TABLE3_ELIGIBLE_CACHE: dict[tuple, list[str]] = {}
_PAPER_TABLE3_ROWS_CACHE: dict[tuple, list[dict[str, float | str]]] = {}


def _repo_root() -> Path:
    return Path(settings.BASE_DIR).parent


def _dataset_summary_path() -> Path:
    return _repo_root() / "benchmark" / "reference" / "datasets" / "ledger_public_compress_summary.json"


def paper_table3_dataset_keys() -> list[str]:
    payload = json.loads(_dataset_summary_path().read_text(encoding="utf-8"))
    keys = [str(key) for key in payload["meta"]["dataset_keys"]]
    if len(keys) != 19:
        raise RuntimeError(f"Expected 19 paper datasets, found {len(keys)}")
    if "poisson_timestamps" in keys:
        raise RuntimeError("Retired dataset poisson_timestamps is still in the paper scope")
    return keys


def paper_table3_scope_for_request(data_type: object, float_precision: object) -> dict[str, str] | None:
    dtype = str(data_type or "").strip().lower()
    if dtype == "overall":
        return PAPER_TABLE3_SCOPES["overall"]
    if dtype == "int":
        return PAPER_TABLE3_SCOPES["integer"]
    if dtype == "float":
        fp = parse_float_precision_filter(float_precision)
        if fp == "fixed":
            return PAPER_TABLE3_SCOPES["fixed_float"]
        if fp == "non_fixed":
            return PAPER_TABLE3_SCOPES["nonfixed_float"]
    return None


def _candidate_dataset_keys(dataset_keys: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    return (
        [str(key).strip() for key in dataset_keys if str(key).strip()]
        if dataset_keys is not None
        else paper_table3_dataset_keys()
    )


def _scope_report_data_types(scope: dict[str, str]) -> tuple[str, ...]:
    if scope["data_type"] == "overall":
        return ("int", "float")
    return (scope["data_type"],)


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _paper_table3_cache_key(
    backend: Path,
    *,
    scope: dict[str, str],
    implement_language: str,
    dataset_keys: list[str],
    include_rank_samples: bool = False,
) -> tuple:
    backend_root = str(backend.resolve())
    mtimes = []
    for dataset_key in dataset_keys:
        bundle_dir = backend / "dataset" / dataset_key
        for report_data_type in _scope_report_data_types(scope):
            path = bundle_dir / f"{dataset_key}_{report_data_type}_{implement_language}_compression_result.csv"
            mtimes.append((dataset_key, report_data_type, _mtime_ns(path)))
        if scope["data_type"] == "float":
            mtimes.append((dataset_key, "column_info", _mtime_ns(bundle_dir / f"{dataset_key}_column_info.csv")))
    return (
        backend_root,
        scope["key"],
        implement_language,
        tuple(dataset_keys),
        include_rank_samples,
        tuple(mtimes),
    )


def _collect_column_rank_samples(report):
    if report is None or report.empty or "column_index" not in report.columns:
        return {}, 0

    work = report[report["algorithm"].apply(is_home_leaderboard_algorithm_enabled)].copy()
    if work.empty:
        return {}, 0

    for col in ("originalSize", "compressedSize", "compressionTime", "decompressionTime"):
        work[col] = work[col].apply(lambda value: float(value))
    work = work[(work["compressedSize"] > 0) & (work["originalSize"] > 0)]
    if work.empty:
        return {}, 0

    rank_samples: dict[str, list[float]] = defaultdict(list)
    ranked_columns = 0
    for _column_index, column_report in work.groupby("column_index", sort=True):
        column_work = aggregate_report_rows_by_algorithm(column_report)
        column_work["label"] = column_work["algorithm"].apply(boxplot_group_label)
        column_work = column_work[column_work["label"].notna()]
        if column_work.empty:
            continue
        num_points = column_work["originalSize"] / 8.0
        column_work = column_work[num_points > 0].copy()
        if column_work.empty:
            continue
        column_work["rate"] = column_work["compressedSize"] / column_work["originalSize"]
        column_work["comp_ns"] = column_work["compressionTime"] / num_points
        column_work["decomp_ns"] = column_work["decompressionTime"] / num_points

        best_rows = column_work.loc[column_work.groupby("label", sort=False)["rate"].idxmin()]
        retained = {
            str(row.label): (float(row.rate), float(row.comp_ns), float(row.decomp_ns))
            for row in best_rows.itertuples(index=False)
            if str(row.label) not in PAPER_TABLE3_EXCLUDED_FAMILIES
            and not is_lossy_boxplot_label(str(row.label))
        }
        if len(retained) < 2:
            continue
        _scores, ranks = compute_balanced_score_metrics([retained])
        if not ranks:
            continue
        ranked_columns += 1
        for family, values in ranks.items():
            rank_samples[family].extend(values)
    return rank_samples, ranked_columns


def paper_table3_eligible_dataset_keys(
    backend: Path,
    *,
    data_type: object,
    float_precision: object,
    implement_language: str = "cpp",
    dataset_keys: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    scope = paper_table3_scope_for_request(data_type, float_precision)
    candidate_keys = _candidate_dataset_keys(dataset_keys)
    if scope is None:
        return candidate_keys

    cache_key = _paper_table3_cache_key(
        backend,
        scope=scope,
        implement_language=implement_language,
        dataset_keys=candidate_keys,
    )
    cached = _PAPER_TABLE3_ELIGIBLE_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    eligible: list[str] = []
    for dataset_key in candidate_keys:
        bundle_dir = backend / "dataset" / dataset_key
        report = read_benchmark_report(
            bundle_dir,
            dataset_key,
            implement_language=implement_language,
            data_type=scope["data_type"],
        )
        if report is None or report.empty:
            continue
        if scope["data_type"] == "float":
            report = filter_report_dataframe_for_float_precision(
                report,
                column_info_path=str(bundle_dir / f"{dataset_key}_column_info.csv"),
                bundle_dir=str(bundle_dir),
                float_precision=scope["float_precision"],
            )
        if report is None or report.empty:
            continue
        metrics, _ = best_metrics_from_report(
            report,
            file_size=1.0,
            implement_language=implement_language,
            use_report_original_size_for_rate=True,
        )
        retained = {
            family: values
            for family, values in metrics.items()
            if family not in PAPER_TABLE3_EXCLUDED_FAMILIES
            and not is_lossy_boxplot_label(family)
        }
        if len(retained) >= PAPER_TABLE3_TOP_K:
            eligible.append(dataset_key)
    _PAPER_TABLE3_ELIGIBLE_CACHE[cache_key] = list(eligible)
    return eligible


def rank_overall_from_average_metrics(rows: list[dict[str, float | str]]) -> None:
    average_metrics = {
        str(row["method"]): (
            float(row["average_rate"]),
            float(row["average_compression_ns"]),
            float(row["average_decompression_ns"]),
        )
        for row in rows
    }
    scores, _ = compute_balanced_score_metrics([average_metrics])
    for row in rows:
        method_scores = scores.get(str(row["method"]), [])
        if len(method_scores) != 1:
            raise RuntimeError(f"Missing average-metric overall score for {row['method']}")
        row["median_balance_score"] = method_scores[0]
    rows.sort(
        key=lambda row: (
            float(row["median_balance_score"]),
            str(row["method"]),
        )
    )


def paper_table3_static_rows(
    backend: Path,
    *,
    data_type: object,
    float_precision: object,
    implement_language: str = "cpp",
    dataset_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    include_rank_samples: bool = False,
) -> list[dict[str, float | str]]:
    scope = paper_table3_scope_for_request(data_type, float_precision)
    if scope is None:
        return []

    candidate_keys = _candidate_dataset_keys(dataset_keys)
    cache_key = _paper_table3_cache_key(
        backend,
        scope=scope,
        implement_language=implement_language,
        dataset_keys=candidate_keys,
        include_rank_samples=include_rank_samples,
    )
    cached = _PAPER_TABLE3_ROWS_CACHE.get(cache_key)
    if cached is not None:
        return [dict(row) for row in cached]

    per_dataset: dict[str, dict[str, tuple[float, float, float]]] = {}
    column_ranks: dict[str, list[float]] = defaultdict(list)
    ranked_column_count = 0
    for dataset_key in candidate_keys:
        bundle_dir = backend / "dataset" / dataset_key
        report = read_benchmark_report(
            bundle_dir,
            dataset_key,
            implement_language=implement_language,
            data_type=scope["data_type"],
        )
        if report is None or report.empty:
            continue
        if scope["data_type"] == "float":
            report = filter_report_dataframe_for_float_precision(
                report,
                column_info_path=str(bundle_dir / f"{dataset_key}_column_info.csv"),
                bundle_dir=str(bundle_dir),
                float_precision=scope["float_precision"],
            )
        if report is None or report.empty:
            continue
        metrics, _ = best_metrics_from_report(
            report,
            file_size=1.0,
            implement_language=implement_language,
            use_report_original_size_for_rate=True,
        )
        retained = {
            family: values
            for family, values in metrics.items()
            if family not in PAPER_TABLE3_EXCLUDED_FAMILIES
            and not is_lossy_boxplot_label(family)
        }
        if len(retained) < PAPER_TABLE3_TOP_K:
            continue
        per_dataset[dataset_key] = retained
        if include_rank_samples:
            dataset_column_ranks, dataset_ranked_column_count = _collect_column_rank_samples(report)
            ranked_column_count += dataset_ranked_column_count
            for family, samples in dataset_column_ranks.items():
                column_ranks[family].extend(samples)

    if not per_dataset:
        raise RuntimeError(f"No C++ results found for {scope['key']}")
    if include_rank_samples and ranked_column_count <= 0:
        raise RuntimeError(f"No column-level ranks found for {scope['key']}")

    balanced_scores, dataset_ranks = compute_balanced_score_metrics(list(per_dataset.values()))
    rows: list[dict[str, float | str]] = []
    if scope["key"] == "overall":
        available_families = set.intersection(*(set(metrics) for metrics in per_dataset.values()))
    else:
        available_families = set().union(*(set(metrics) for metrics in per_dataset.values()))

    for family in available_families:
        values = [metrics[family] for metrics in per_dataset.values() if family in metrics]
        scores = balanced_scores.get(family, [])
        if not values or len(scores) != len(values):
            continue
        rank_samples = dataset_ranks.get(family, [])
        if include_rank_samples and family != PAPER_TABLE3_REFERENCE_METHOD:
            rank_samples = column_ranks.get(family, [])
        if not rank_samples:
            rank_samples = dataset_ranks.get(family, [])
        rows.append(
            {
                "method": family,
                "rank": statistics.fmean(rank_samples),
                "average_rate": statistics.fmean(value[0] for value in values),
                "average_compression_ns": statistics.fmean(value[1] for value in values),
                "average_decompression_ns": statistics.fmean(value[2] for value in values),
                "median_balance_score": statistics.median(scores),
                "dataset_count": len(values),
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["median_balance_score"]),
            str(row["method"]),
        )
    )
    if len(rows) < PAPER_TABLE3_TOP_K:
        raise RuntimeError(f"Only {len(rows)} methods have results for {scope['key']}")
    if scope["key"] == "overall":
        rank_overall_from_average_metrics(rows)
    _PAPER_TABLE3_ROWS_CACHE[cache_key] = [dict(row) for row in rows]
    return rows


def paper_table3_display_rows(
    backend: Path,
    *,
    data_type: object,
    float_precision: object,
    implement_language: str = "cpp",
    dataset_keys: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, float | str]]:
    """Return the exact method rows displayed in paper Table 3."""
    scope = paper_table3_scope_for_request(data_type, float_precision)
    if scope is None:
        return []
    snapshot = json.loads(PAPER_TABLE3_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    scopes = [str(scope_key) for scope_key in snapshot.get("scopes", [])]
    try:
        scope_index = scopes.index(scope["key"])
    except ValueError:
        return []

    rows: list[dict[str, float | str]] = []
    for source_row in snapshot.get("rows", []):
        def pick(field: str) -> float | None:
            values = source_row.get(field, [])
            if scope_index < len(values):
                return float(values[scope_index])
            return None

        rows.append(
            {
                "method": str(source_row["method"]),
                "rank": pick("rank"),
                "average_rate": pick("average_rate"),
                "average_compression_ns": pick("average_compression_ns"),
                "average_decompression_ns": pick("average_decompression_ns"),
                "median_balance_score": pick("median_balance_score"),
            }
        )
    return rows
