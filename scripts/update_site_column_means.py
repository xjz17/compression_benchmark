#!/usr/bin/env python3
"""Recalculate published timing metrics with explicit evidence-aware modes.

Compression sizes, rates, configuration choices, and row identities are copied
unchanged. Column-backed rows use an arithmetic mean over columns. Imported
whole-report rows (column_index=-1) remain dataset aggregates with a zero column
count; they are never relabeled as columns. Missing measurements remain null.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "deploy/huggingface-compressionbenchmark"
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(SITE))
sys.path.insert(0, str(REPO / "scripts"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return row["dataset"], row["scope"], row["runtime"], row["algorithm_variant"]


def aggregate_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return row["runtime"], row["scope"], row["family"], row["method"]


def valid_measurement_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    work = rows.copy()
    for field in ("column_index", "originalSize", "compressedSize", "compressionTime", "decompressionTime"):
        work[field] = pd.to_numeric(work[field], errors="coerce")
    return work[
        (work["originalSize"] > 0)
        & (work["compressedSize"] > 0)
        & (work["compressionTime"] >= 0)
        & (work["decompressionTime"] >= 0)
    ].copy()


def timing_stats(rows: pd.DataFrame, *, context: str = "") -> dict[str, Any]:
    if rows is None or rows.empty:
        return {"compression": None, "decompression": None, "column_count": 0,
                "aggregation": "unavailable", "point_count": 0.0,
                "compression_time_ns": None, "decompression_time_ns": None}
    work = valid_measurement_rows(rows)
    if work.empty:
        return {"compression": None, "decompression": None, "column_count": 0,
                "aggregation": "unavailable", "point_count": 0.0,
                "compression_time_ns": None, "decompression_time_ns": None}
    if work.get("_timing_source_type") is None:
        work["_timing_source_type"] = "single_report"
    column_groups = []
    aggregate_groups = []
    for source_type, source_rows in work.groupby("_timing_source_type", sort=True):
        columns = source_rows[source_rows["column_index"] >= 0]
        if not columns.empty:
            grouped = columns.groupby("column_index", as_index=False).agg(
                originalSize=("originalSize", "sum"),
                compressionTime=("compressionTime", "sum"),
                decompressionTime=("decompressionTime", "sum"),
            )
            grouped["_timing_source_type"] = source_type
            column_groups.append(grouped)
            continue
        aggregate = source_rows[source_rows["column_index"] < 0]
        if not aggregate.empty:
            aggregate_groups.append(aggregate)
    if column_groups and not aggregate_groups:
        grouped = pd.concat(column_groups, ignore_index=True)
        points = grouped["originalSize"] / 8.0
        return {
            "compression": float((grouped["compressionTime"] / points).mean()),
            "decompression": float((grouped["decompressionTime"] / points).mean()),
            "column_count": int(len(grouped)), "aggregation": "column_mean",
            "point_count": float(points.sum()),
            "compression_time_ns": float(grouped["compressionTime"].sum()),
            "decompression_time_ns": float(grouped["decompressionTime"].sum()),
        }
    # If any typed report has only a whole-report timing, use source elapsed
    # totals for one dataset-level value. Column evidence from other typed
    # reports is folded into the same dataset total, never assigned a fake count.
    elapsed_parts = [*column_groups, *aggregate_groups]
    if elapsed_parts:
        # For each typed report, use real columns when present and otherwise its
        # whole-report aggregate. This avoids counting a -1 aggregate again when
        # that same typed report already provides the constituent columns.
        elapsed = pd.concat(elapsed_parts, ignore_index=True)
        points = float(elapsed["originalSize"].sum() / 8.0)
        compression_time_ns = float(elapsed["compressionTime"].sum())
        decompression_time_ns = float(elapsed["decompressionTime"].sum())
        if points > 0:
            return {
                "compression": compression_time_ns / points,
                "decompression": decompression_time_ns / points,
                "column_count": 0, "aggregation": "dataset_aggregate",
                "point_count": points, "compression_time_ns": compression_time_ns,
                "decompression_time_ns": decompression_time_ns,
            }
    return {"compression": None, "decompression": None, "column_count": 0,
            "aggregation": "unavailable", "point_count": 0.0,
            "compression_time_ns": None, "decompression_time_ns": None}


def audit_column_means(audit: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    from integrate_dbp_leaderboard import precision_map

    allowed = {d["dataset"] for d in audit["datasets"]}
    classifications = {dataset: precision_map(dataset)[0] for dataset in sorted(allowed)}
    observations: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    byte_totals: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    elapsed_totals: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    seen: set[tuple[str, str]] = set()
    for column in audit["columns"]:
        key = (column["aggregate_id"], column["runtime"])
        if key in seen:
            raise ValueError(f"Duplicate frozen DBP column/runtime: {key}")
        seen.add(key)
        category = classifications[column["dataset"]].get(column["source_column"], "nonfixed_float")
        for scope in ("float", category, "overall"):
            group_key = column["dataset"], scope, column["runtime"]
            byte_totals[group_key][0] += int(column["native_bytes"])
            byte_totals[group_key][1] += int(column["frame_bytes"])
        if column["status"] != "ok":
            continue
        points = int(column["valid_count"])
        if points <= 0 or int(column["verified_values"]) != points:
            raise ValueError(f"Invalid frozen DBP column receipt: {key}")
        comp = float(column["encode_ns"]) / points
        decomp = float(column["decode_ns"]) / points
        if not math.isfinite(comp) or not math.isfinite(decomp):
            raise ValueError(f"Non-finite frozen DBP timing: {key}")
        for scope in ("float", category, "overall"):
            group_key = column["dataset"], scope, column["runtime"]
            observations[group_key].append((comp, decomp))
            elapsed_totals[group_key][0] += points
            elapsed_totals[group_key][1] += float(column["encode_ns"])
            elapsed_totals[group_key][2] += float(column["decode_ns"])
    result = {}
    for key, totals in byte_totals.items():
        values = observations.get(key, [])
        elapsed = elapsed_totals[key]
        result[key] = {
            "compression": sum(value[0] for value in values) / len(values) if values else None,
            "decompression": sum(value[1] for value in values) / len(values) if values else None,
            "column_count": len(values), "aggregation": "column_mean" if values else "unavailable",
            "original_size_bytes": totals[0], "compressed_size_bytes": totals[1],
            "point_count": elapsed[0], "compression_time_ns": elapsed[1],
            "decompression_time_ns": elapsed[2],
        }
    return result


def combine_timing_components(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts or any(part["aggregation"] == "unavailable" for part in parts):
        return {"compression": None, "decompression": None, "column_count": 0,
                "aggregation": "unavailable", "point_count": 0.0,
                "compression_time_ns": None, "decompression_time_ns": None}
    if any(part["aggregation"] == "dataset_aggregate" for part in parts):
        points = sum(part["point_count"] for part in parts)
        if points <= 0 or any(part["compression_time_ns"] is None or part["decompression_time_ns"] is None
                              for part in parts):
            return {"compression": None, "decompression": None, "column_count": 0,
                    "aggregation": "unavailable", "point_count": 0.0,
                    "compression_time_ns": None, "decompression_time_ns": None}
        compression_time_ns = sum(part["compression_time_ns"] for part in parts)
        decompression_time_ns = sum(part["decompression_time_ns"] for part in parts)
        return {"compression": compression_time_ns / points,
                "decompression": decompression_time_ns / points,
                "column_count": 0, "aggregation": "dataset_aggregate",
                "point_count": points, "compression_time_ns": compression_time_ns,
                "decompression_time_ns": decompression_time_ns}
    count = sum(part["column_count"] for part in parts)
    if count <= 0:
        return {"compression": None, "decompression": None, "column_count": 0,
                "aggregation": "unavailable", "point_count": 0.0,
                "compression_time_ns": None, "decompression_time_ns": None}
    return {
        "compression": sum(part["compression"] * part["column_count"] for part in parts) / count,
        "decompression": sum(part["decompression"] * part["column_count"] for part in parts) / count,
        "column_count": count, "aggregation": "column_mean",
        "point_count": sum(part["point_count"] for part in parts),
        "compression_time_ns": sum(part["compression_time_ns"] for part in parts),
        "decompression_time_ns": sum(part["decompression_time_ns"] for part in parts),
    }


def refresh_current_dbp_aggregates(site: Path, receipt_path: Path) -> None:
    """Refresh only DBP aggregate rows after a verified narrow row overlay."""
    from export_results import _aggregate_rows

    data_path = site / "data.json"
    manifest_path = site / "manifest.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before_sha = sha(data_path)
    dbp_rows = []
    for source_row in data["dataset_results"]:
        if source_row["family"] != "DBP":
            continue
        row = dict(source_row)
        row.setdefault("point_count", row["original_size_bytes"] / 8.0)
        dbp_rows.append(row)
    configurations: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    families: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dbp_rows:
        configurations[aggregate_key(row)].append(row)
        if row["is_best_rate_in_family"]:
            families[row["runtime"], row["scope"], row["family"], row["family"]].append(row)

    refreshed_counts = {}
    for name, items, groups in (
        ("configuration_aggregates", data["configuration_aggregates"], configurations),
        ("family_best_aggregates", data["family_best_aggregates"], families),
    ):
        by_key = {aggregate_key(item): item for item in items}
        if not set(groups).issubset(by_key):
            raise ValueError(f"Missing DBP aggregate identities in {name}")
        for key, rows in groups.items():
            runtime, scope, family, method = key
            refreshed = _aggregate_rows(
                rows, runtime=runtime, scope_key=scope, family=family, method=method,
                algorithm_variant=method if method != family else None,
            )
            by_key[key].clear()
            by_key[key].update(refreshed)
        refreshed_counts[name] = len(groups)

    write_json(data_path, data)
    manifest["data_sha256"] = sha(data_path)
    manifest["counts"] = data["coverage"]
    manifest["measurement_runs"] = data["source"].get("measurement_runs", [])
    write_json(manifest_path, manifest)
    receipt = {
        "status": "passed", "operation": "refresh_current_dbp_aggregates",
        "baseline_data_sha256": before_sha, "data_sha256": sha(data_path),
        "manifest_sha256": sha(manifest_path), "dbp_dataset_result_count": len(dbp_rows),
        **refreshed_counts,
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def update(
    site: Path, receipt_path: Path, backup_dir: Path | None = None,
    *, baseline_from_backup: bool = False,
    dbp_medium_payload: Path | None = None,
    dbp_medium_payload_sha256: str | None = None,
    dbp_medium_receipt: Path | None = None,
    dbp_medium_receipt_sha256: str | None = None,
) -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    import django

    django.setup()
    from myapp.dataset_float_precision_distribution import filter_report_dataframe_for_float_precision
    from myapp.home_leaderboard_boxplot_data import read_benchmark_report

    data_path = site / "data.json"
    manifest_path = site / "manifest.json"
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in ("data.json", "ui-data.json", "manifest.json"):
            source = site / name
            destination = backup_dir / f"{name}.before_dataset_fallback"
            if not destination.exists():
                shutil.copy2(source, destination)
    input_data_path = data_path
    input_manifest_path = manifest_path
    if baseline_from_backup:
        if backup_dir is None:
            raise ValueError("--baseline-from-backup requires --backup-dir")
        input_data_path = backup_dir / "data.json.before_dataset_fallback"
        input_manifest_path = backup_dir / "manifest.json.before_dataset_fallback"
        if not input_data_path.is_file() or not input_manifest_path.is_file():
            raise ValueError("Requested fallback baseline backup is missing")
    data = json.loads(input_data_path.read_text(encoding="utf-8"))
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    before_sha = sha(input_data_path)
    before_rows = data["dataset_results"]
    immutable = {
        row_key(row): (
            row["original_size_bytes"], row["compressed_size_bytes"], row["compression_rate"],
            row["is_best_rate_in_family"], row["family"], row["method"],
        )
        for row in before_rows
    }
    if len(immutable) != len(before_rows):
        raise ValueError("Published dataset result identities are not unique")

    source_files = data["source"]["result_csv_files_used"]
    mismatches = []
    for source in source_files:
        path = REPO / source["path"]
        actual = sha(path) if path.is_file() else None
        if actual != source["sha256"] or path.stat().st_size != source["size_bytes"]:
            mismatches.append({"path": source["path"], "expected": source["sha256"], "actual": actual})
    if mismatches:
        raise ValueError(f"Published result CSV sources drifted: {mismatches[:3]}")

    measurement_run = next(
        run for run in data["source"].get("measurement_runs", [])
        if run.get("method") == "DELTA_BINARY_PACKED"
    )
    audit_path = REPO / "analysis_outputs/dbp_float_e2e_20261009/publish_payload/dbp-audit.json"
    if sha(audit_path) != measurement_run["source_sha256"]:
        raise ValueError("Frozen DBP audit source differs from published measurement metadata")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("Frozen DBP audit is incomplete")
    integration_receipt_path = REPO / "analysis_outputs/dbp_main_leaderboard_20261009/integration_receipt.json"
    integration_receipt = json.loads(integration_receipt_path.read_text(encoding="utf-8"))
    if integration_receipt["measurement_source_sha256"] != sha(audit_path):
        raise ValueError("DBP integration receipt does not bind the frozen audit payload")
    precision_mismatches = []
    for source in integration_receipt["precision_source_files"]:
        path = REPO / source["path"]
        actual = sha(path) if path.is_file() else None
        if actual != source["sha256"]:
            precision_mismatches.append({"path": source["path"], "expected": source["sha256"], "actual": actual})
    if precision_mismatches:
        raise ValueError(f"DBP precision classification sources drifted: {precision_mismatches[:3]}")
    dbp_audit_means = audit_column_means(audit)

    overlay_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    overlay_metadata: dict[str, Any] | None = None
    if dbp_medium_payload is not None:
        if not all((dbp_medium_payload_sha256, dbp_medium_receipt, dbp_medium_receipt_sha256)):
            raise ValueError("DBP medium_posts overlay requires payload/receipt paths and both SHA-256 values")
        if sha(dbp_medium_payload) != dbp_medium_payload_sha256:
            raise ValueError("DBP medium_posts integration payload SHA-256 mismatch")
        if sha(dbp_medium_receipt) != dbp_medium_receipt_sha256:
            raise ValueError("DBP medium_posts measurement receipt SHA-256 mismatch")
        overlay = json.loads(dbp_medium_payload.read_text(encoding="utf-8"))
        overlay_receipt = json.loads(dbp_medium_receipt.read_text(encoding="utf-8"))
        if overlay.get("status") != "ready" or overlay.get("schema") != "dbp-medium-posts-integration-v2":
            raise ValueError("Unexpected DBP medium_posts integration payload contract")
        if overlay_receipt.get("status") != "passed" or overlay_receipt.get("dataset") != "medium_posts":
            raise ValueError("DBP medium_posts measurement receipt is not passed")
        for source_row in overlay.get("rows", []):
            row = dict(source_row)
            key = row_key(row)
            if key in overlay_by_key:
                raise ValueError(f"Duplicate DBP medium_posts overlay row: {key}")
            if (row["dataset"] != "medium_posts" or row["family"] != "DBP"
                    or row["method"] != "DELTA_BINARY_PACKED"
                    or row["algorithm_variant"] != "DELTA_BINARY_PACKED"
                    or row["runtime"] not in ("cpp", "java", "python")
                    or row["scope"] not in ("integer", "overall")
                    or row["timing_aggregation"] != "column_mean"
                    or row["timing_column_count"] != 1
                    or row.get("validation_status") != "passed_byte_for_byte_lossless"):
                raise ValueError(f"Invalid DBP medium_posts overlay row: {key}")
            points = row["point_count"]
            if points <= 0 or row["original_size_bytes"] != points * 8:
                raise ValueError(f"Invalid DBP medium_posts point/byte count: {key}")
            if not math.isclose(row["compression_time_ns_per_point"], row["compression_time_ns"] / points,
                                rel_tol=1e-15, abs_tol=0.0):
                raise ValueError(f"Invalid DBP medium_posts compression time: {key}")
            if not math.isclose(row["decompression_time_ns_per_point"], row["decompression_time_ns"] / points,
                                rel_tol=1e-15, abs_tol=0.0):
                raise ValueError(f"Invalid DBP medium_posts decompression time: {key}")
            overlay_by_key[key] = row
        expected_overlay_keys = {
            ("medium_posts", scope, runtime, "DELTA_BINARY_PACKED")
            for runtime in ("cpp", "java", "python") for scope in ("integer", "overall")
        }
        if set(overlay_by_key) != expected_overlay_keys:
            raise ValueError("DBP medium_posts overlay must contain exactly three runtimes x two scopes")
        overlay_metadata = {
            "method": "DELTA_BINARY_PACKED", "family": "DBP", "dataset": "medium_posts",
            "scopes": ["integer", "overall"], "runtimes": ["cpp", "java", "python"],
            "source_sha256": dbp_medium_receipt_sha256,
            "integration_payload_sha256": dbp_medium_payload_sha256,
            "validation_status": "passed_byte_for_byte_lossless",
            "timing_aggregation": "column_mean", "timing_column_count_per_row": 1,
            "environment": overlay_receipt.get("environment"),
            "benchmark_contract": overlay_receipt.get("benchmark_contract"),
            "cross_runtime_golden": overlay_receipt.get("cross_runtime_golden"),
        }

    report_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    def report(dataset: str, scope: str, runtime: str) -> pd.DataFrame:
        cache_key = dataset, scope, runtime
        if cache_key in report_cache:
            return report_cache[cache_key]
        bundle = BACKEND / "dataset" / dataset
        data_type = "int" if scope == "integer" else "float" if scope != "overall" else "overall"
        if data_type == "overall":
            frames = []
            for report_type in ("int", "float"):
                part = read_benchmark_report(bundle, dataset, data_type=report_type, implement_language=runtime)
                if part is not None and not part.empty:
                    part = part.copy()
                    part["_timing_source_type"] = report_type
                    frames.append(part)
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            frame = read_benchmark_report(bundle, dataset, data_type=data_type, implement_language=runtime)
            if frame is None:
                frame = pd.DataFrame()
            elif not frame.empty:
                frame = frame.copy()
                frame["_timing_source_type"] = data_type
        if not frame.empty and scope in ("fixed_float", "nonfixed_float"):
            frame = filter_report_dataframe_for_float_precision(
                frame,
                column_info_path=str(bundle / f"{dataset}_column_info.csv"),
                bundle_dir=str(bundle),
                float_precision="fixed" if scope == "fixed_float" else "non_fixed",
            )
        report_cache[cache_key] = frame
        return frame

    output_rows = []
    unavailable = []
    dataset_aggregates = []
    changed_timing = 0
    published_dbp_integers = {
        (row["dataset"], row["runtime"]): row
        for row in before_rows if row["family"] == "DBP" and row["scope"] == "integer"
    }
    for original in before_rows:
        row = dict(original)
        key = row_key(row)
        if key in overlay_by_key:
            replacement = dict(overlay_by_key[key])
            if (row.get("compression_time_ns_per_point"), row.get("decompression_time_ns_per_point")) != (
                replacement["compression_time_ns_per_point"], replacement["decompression_time_ns_per_point"]
            ):
                changed_timing += 1
            output_rows.append(replacement)
            continue
        if row["family"] == "DBP" and row["scope"] != "integer":
            components = []
            integer_baseline = published_dbp_integers.get((row["dataset"], row["runtime"]))
            if row["scope"] == "overall" and integer_baseline is not None:
                integer_frame = report(row["dataset"], "integer", row["runtime"])
                if not integer_frame.empty:
                    integer_rows = integer_frame[
                        integer_frame["algorithm"].astype(str) == row["algorithm_variant"]
                    ]
                    integer_rows = valid_measurement_rows(integer_rows)
                    components.append(timing_stats(
                        integer_rows,
                        context=f"{row['dataset']}/integer/{row['runtime']}/{row['algorithm_variant']}",
                    ))
                else:
                    components.append(timing_stats(pd.DataFrame()))
            audit_mean = dbp_audit_means.get((row["dataset"], row["scope"], row["runtime"]))
            if audit_mean:
                components.append(audit_mean)
            timing = combine_timing_components(components)
        else:
            frame = report(row["dataset"], row["scope"], row["runtime"])
            selected = frame[frame["algorithm"].astype(str) == row["algorithm_variant"]] if not frame.empty else frame
            selected = valid_measurement_rows(selected)
            timing = timing_stats(
                selected,
                context="/".join(row_key(row)),
            )
            selected_original = int(round(pd.to_numeric(selected["originalSize"], errors="coerce").sum()))
            selected_compressed = int(round(pd.to_numeric(selected["compressedSize"], errors="coerce").sum()))
            if (selected_original, selected_compressed) != (
                row["original_size_bytes"], row["compressed_size_bytes"]
            ):
                raise ValueError(
                    f"Published byte totals do not match selected source rows for {row_key(row)}: "
                    f"{(selected_original, selected_compressed)} != "
                    f"{(row['original_size_bytes'], row['compressed_size_bytes'])}"
                )
        if row["family"] == "DBP" and row["scope"] != "integer":
            audit_totals = dbp_audit_means.get((row["dataset"], row["scope"], row["runtime"]))
            expected_original = audit_totals["original_size_bytes"] if audit_totals else 0
            expected_compressed = audit_totals["compressed_size_bytes"] if audit_totals else 0
            if row["scope"] == "overall" and integer_baseline is not None:
                expected_original += integer_baseline["original_size_bytes"]
                expected_compressed += integer_baseline["compressed_size_bytes"]
            if (expected_original, expected_compressed) != (
                row["original_size_bytes"], row["compressed_size_bytes"]
            ):
                raise ValueError(
                    f"Published DBP byte totals do not match frozen components for {row_key(row)}: "
                    f"{(expected_original, expected_compressed)} != "
                    f"{(row['original_size_bytes'], row['compressed_size_bytes'])}"
                )
        if timing["aggregation"] == "unavailable":
            unavailable.append({
                "dataset": row["dataset"], "scope": row["scope"], "runtime": row["runtime"],
                "method": row["method"], "family": row["family"],
            })
        elif timing["aggregation"] == "dataset_aggregate":
            dataset_aggregates.append({
                "dataset": row["dataset"], "scope": row["scope"], "runtime": row["runtime"],
                "method": row["method"], "family": row["family"],
            })
        old_pair = (row.get("compression_time_ns_per_point"), row.get("decompression_time_ns_per_point"))
        new_pair = (timing["compression"], timing["decompression"])
        if old_pair != new_pair:
            changed_timing += 1
        row["compression_time_ns_per_point"] = timing["compression"]
        row["decompression_time_ns_per_point"] = timing["decompression"]
        row["timing_column_count"] = timing["column_count"]
        row["timing_aggregation"] = timing["aggregation"]
        output_rows.append(row)

    existing_output_keys = {row_key(row) for row in output_rows}
    for key in sorted(set(overlay_by_key) - existing_output_keys):
        output_rows.append(dict(overlay_by_key[key]))
        changed_timing += 1

    expected_keys = set(immutable) | set(overlay_by_key)
    if len(output_rows) != len(expected_keys) or {row_key(row) for row in output_rows} != expected_keys:
        raise ValueError("Dataset result identities changed during timing recalculation")
    for row in output_rows:
        if row_key(row) in overlay_by_key:
            if row != overlay_by_key[row_key(row)]:
                raise ValueError(f"DBP medium_posts overlay row changed: {row_key(row)}")
            continue
        if immutable[row_key(row)] != (
            row["original_size_bytes"], row["compressed_size_bytes"], row["compression_rate"],
            row["is_best_rate_in_family"], row["family"], row["method"],
        ):
            raise ValueError(f"Non-timing fields changed for {row_key(row)}")

    def timing_aggregate(rows: list[dict[str, Any]]) -> tuple[float | None, float | None, int, str]:
        count = sum(row["timing_column_count"] for row in rows)
        if any(row["timing_aggregation"] == "unavailable" for row in rows):
            return None, None, count, "unavailable"
        if any(row["timing_aggregation"] == "dataset_aggregate" for row in rows):
            return (
                sum(row["compression_time_ns_per_point"] for row in rows) / len(rows),
                sum(row["decompression_time_ns_per_point"] for row in rows) / len(rows),
                count, "dataset_mean",
            )
        if not count:
            return None, None, 0, "unavailable"
        return (
            sum(row["compression_time_ns_per_point"] * row["timing_column_count"] for row in rows) / count,
            sum(row["decompression_time_ns_per_point"] * row["timing_column_count"] for row in rows) / count,
            count, "column_mean",
        )

    configurations: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    families: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        configurations[aggregate_key(row)].append(row)
        if row["is_best_rate_in_family"]:
            families[row["runtime"], row["scope"], row["family"], row["family"]].append(row)

    def update_aggregates(items: list[dict[str, Any]], groups: dict) -> None:
        from export_results import _aggregate_rows

        if len(items) != len(groups):
            raise ValueError(f"Aggregate identity count changed: {len(items)} != {len(groups)}")
        by_key = {aggregate_key(item): item for item in items}
        if set(by_key) != set(groups):
            raise ValueError("Aggregate identities changed during timing recalculation")
        for key, rows in groups.items():
            if key[2] == "DBP":
                runtime, scope, family, method = key
                aggregate_rows = []
                for source_row in rows:
                    normalized = dict(source_row)
                    normalized.setdefault("point_count", normalized["original_size_bytes"] / 8.0)
                    aggregate_rows.append(normalized)
                refreshed = _aggregate_rows(
                    aggregate_rows, runtime=runtime, scope_key=scope, family=family, method=method,
                    algorithm_variant=method if method != family else None,
                )
                by_key[key].clear()
                by_key[key].update(refreshed)
                continue
            comp, decomp, count, mode = timing_aggregate(rows)
            by_key[key]["average_compression_time_ns_per_point"] = comp
            by_key[key]["average_decompression_time_ns_per_point"] = decomp
            by_key[key]["timing_column_count"] = count
            by_key[key]["timing_aggregation"] = mode
            by_key[key]["timing_dataset_count"] = len(rows)

    update_aggregates(data["configuration_aggregates"], configurations)
    update_aggregates(data["family_best_aggregates"], families)
    data["dataset_results"] = output_rows
    benchmark = data["benchmark"]
    benchmark["aggregation_note"] = (
        "Dataset rows with column evidence use an arithmetic mean of per-column nanoseconds per point. "
        "Rows backed only by whole-report timing use the source dataset elapsed-time/point total and "
        "retain a zero column count. If any selected row uses that fallback, method timing is an "
        "equal-dataset arithmetic mean; otherwise it is pooled equally over measured columns."
    )
    benchmark["metric_definitions"]["average_time_ns_per_point"] = (
        "column-backed rows use an arithmetic mean of per-column nanoseconds per numeric value; "
        "whole-report-only rows use source elapsed nanoseconds divided by source point totals; "
        "legacy CSV rows retain their 8-byte-equivalent point denominator"
    )
    boundary = (
        "Whole-report timing rows (column_index=-1) are not treated as columns and keep a zero column "
        "count. They provide a dataset-level timing fallback, while genuinely missing measurements "
        "remain unavailable."
    )
    benchmark["measurement_boundaries"] = [
        note for note in benchmark["measurement_boundaries"]
        if not note.startswith("Whole-report timing rows (column_index=-1)")
    ]
    if boundary not in benchmark["measurement_boundaries"]:
        benchmark["measurement_boundaries"].append(boundary)
    coverage = data["coverage"]
    coverage["timing_column_observation_count"] = sum(row["timing_column_count"] for row in output_rows)
    coverage["timing_column_mean_dataset_result_count"] = sum(
        row["timing_aggregation"] == "column_mean" for row in output_rows
    )
    coverage["timing_dataset_aggregate_result_count"] = len(dataset_aggregates)
    coverage["timing_unavailable_dataset_result_count"] = len(unavailable)
    coverage["dataset_result_row_count"] = len(output_rows)
    coverage["configuration_aggregate_row_count"] = len(data["configuration_aggregates"])
    coverage["family_best_aggregate_row_count"] = len(data["family_best_aggregates"])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_receipt = {
        "policy": "column_mean_with_dataset_aggregate_fallback",
        "generated_at": now,
        "result_csv_file_count": len(source_files),
        "result_csv_sha256_status": "all_match",
        "dbp_float_audit_path": str(audit_path.relative_to(REPO)),
        "dbp_float_audit_sha256": sha(audit_path),
        "dbp_precision_source_file_count": len(integration_receipt["precision_source_files"]),
        "dbp_precision_source_hash_status": "all_match",
        "aggregate_only_dataset_fallback_row_count": len(dataset_aggregates),
        "timing_unavailable_row_count": len(unavailable),
    }
    data["source"]["timing_recalculation"] = source_receipt
    if overlay_metadata is not None:
        overlay_metadata.update({
            "integrated_at": now,
            "integration_payload_path": str(dbp_medium_payload.relative_to(REPO)),
            "measurement_receipt_path": str(dbp_medium_receipt.relative_to(REPO)),
        })
        runs = [
            run for run in data["source"].get("measurement_runs", [])
            if not (run.get("method") == "DELTA_BINARY_PACKED" and run.get("dataset") == "medium_posts")
        ]
        data["source"]["measurement_runs"] = runs + [overlay_metadata]
    write_json(data_path, data)
    manifest["data_sha256"] = sha(data_path)
    manifest["counts"] = coverage
    manifest["timing_recalculation"] = source_receipt
    manifest["measurement_runs"] = data["source"].get("measurement_runs", [])
    write_json(manifest_path, manifest)
    unavailable_families = Counter(row["family"] for row in unavailable)
    unavailable_methods = Counter(row["method"] for row in unavailable)
    dataset_aggregate_families = Counter(row["family"] for row in dataset_aggregates)
    dataset_aggregate_methods = Counter(row["method"] for row in dataset_aggregates)
    receipt = {
        "status": "passed",
        "generated_at": now,
        "baseline_data_sha256": before_sha,
        "data_sha256": sha(data_path),
        "manifest_sha256": sha(manifest_path),
        "dataset_result_count": len(output_rows),
        "unchanged_dataset_result_identity_count": len(set(immutable) - set(overlay_by_key)),
        "unchanged_size_rate_selection_count": len(set(immutable) - set(overlay_by_key)),
        "changed_timing_row_count": changed_timing,
        "column_mean_timing_row_count": coverage["timing_column_mean_dataset_result_count"],
        "dataset_aggregate_timing_row_count": len(dataset_aggregates),
        "timing_unavailable_row_count": len(unavailable),
        "timing_column_observation_count": coverage["timing_column_observation_count"],
        "result_csv_files_verified": len(source_files),
        "result_csv_hash_mismatch_count": 0,
        "dbp_float_audit_sha256": sha(audit_path),
        "dbp_valid_column_count_per_runtime": audit["summary"]["nonempty_aggregate_count"],
        "unavailable_family_counts": dict(sorted(unavailable_families.items())),
        "unavailable_method_counts": dict(sorted(unavailable_methods.items())),
        "unavailable_examples": unavailable[:20],
        "dataset_aggregate_family_counts": dict(sorted(dataset_aggregate_families.items())),
        "dataset_aggregate_method_counts": dict(sorted(dataset_aggregate_methods.items())),
        "dataset_aggregate_examples": dataset_aggregates[:20],
        "dbp_medium_posts_overlay_row_count": len(overlay_by_key),
        "dbp_medium_posts_replaced_row_count": len(set(immutable) & set(overlay_by_key)),
        "dbp_medium_posts_new_row_count": len(set(overlay_by_key) - set(immutable)),
        "dbp_medium_posts_integration_payload_sha256": dbp_medium_payload_sha256,
        "dbp_medium_posts_measurement_receipt_sha256": dbp_medium_receipt_sha256,
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=SITE)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--baseline-from-backup", action="store_true")
    parser.add_argument("--dbp-medium-payload", type=Path)
    parser.add_argument("--dbp-medium-payload-sha256")
    parser.add_argument("--dbp-medium-receipt", type=Path)
    parser.add_argument("--dbp-medium-receipt-sha256")
    parser.add_argument("--refresh-dbp-aggregates-only", action="store_true")
    args = parser.parse_args()
    if args.refresh_dbp_aggregates_only:
        refresh_current_dbp_aggregates(args.site.resolve(), args.receipt.resolve())
        return
    update(
        args.site.resolve(), args.receipt.resolve(),
        args.backup_dir.resolve() if args.backup_dir else None,
        baseline_from_backup=args.baseline_from_backup,
        dbp_medium_payload=args.dbp_medium_payload.resolve() if args.dbp_medium_payload else None,
        dbp_medium_payload_sha256=args.dbp_medium_payload_sha256,
        dbp_medium_receipt=args.dbp_medium_receipt.resolve() if args.dbp_medium_receipt else None,
        dbp_medium_receipt_sha256=args.dbp_medium_receipt_sha256,
    )


if __name__ == "__main__":
    main()
