#!/usr/bin/env python3
"""Replace the published DBP float rows with verified full-frame measurements.

The input is the frozen campaign payload, retained outside the deployed site.
Existing integer rows and all other methods are preserved. Overall rows combine
the retained integer totals with the replacement float totals before ranking.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "deploy/huggingface-compressionbenchmark"
sys.path.insert(0, str(SITE))
from export_results import _aggregate_rows


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n")


def precision_map(dataset: str) -> tuple[dict[str, str], list[dict]]:
    """Use the same >=85% decimal-frequency rule and fallback as the backend."""
    bundle = REPO / "backend/dataset" / dataset
    info = bundle / f"{dataset}_column_info.csv"
    mapping, sources = {}, []
    if not info.is_file():
        return mapping, sources
    sources.append({"path": str(info.relative_to(REPO)), "sha256": sha(info)})
    with info.open(newline="", encoding="utf-8-sig") as handle:
        for column in csv.DictReader(handle):
            name = column["column_name"]
            stats = bundle / "float_decimal_places" / f"column_{int(column['column_index'])}.csv"
            classification = "nonfixed_float"
            if stats.is_file():
                with stats.open(newline="", encoding="utf-8-sig") as stats_handle:
                    fractions = [float(r.get("fraction") or 0) for r in csv.DictReader(stats_handle)]
                if fractions and max(fractions) >= 0.85:
                    classification = "fixed_float"
                sources.append({"path": str(stats.relative_to(REPO)), "sha256": sha(stats)})
            mapping[name] = classification
    return mapping, sources


def make_row(dataset: str, runtime: str, scope: str, parts: list[dict]) -> dict:
    original = sum(p["original_size_bytes"] for p in parts)
    compressed = sum(p["compressed_size_bytes"] for p in parts)
    points = sum(p["point_count"] for p in parts)
    elapsed_complete = all(
        p.get("compression_time_ns") is not None and p.get("decompression_time_ns") is not None
        for p in parts if p.get("point_count", 0) > 0
    )
    encode = sum(p["compression_time_ns"] for p in parts) if elapsed_complete else None
    decode = sum(p["decompression_time_ns"] for p in parts) if elapsed_complete else None
    positive_parts = [p for p in parts if p.get("point_count", 0) > 0]
    modes = [p.get("timing_aggregation", "unavailable") for p in positive_parts]
    timing_column_count = sum(p.get("timing_column_count", 0) for p in positive_parts)
    if "unavailable" in modes:
        compression_time_ns_per_point = decompression_time_ns_per_point = None
        timing_column_count = 0
        timing_aggregation = "unavailable"
    elif "dataset_aggregate" in modes:
        compression_time_ns_per_point = encode / points if elapsed_complete else None
        decompression_time_ns_per_point = decode / points if elapsed_complete else None
        timing_column_count = 0
        timing_aggregation = "dataset_aggregate" if elapsed_complete else "unavailable"
    elif timing_column_count:
        compression_time_ns_per_point = sum(
            p["compression_time_ns_per_point"] * p["timing_column_count"] for p in positive_parts
        ) / timing_column_count
        decompression_time_ns_per_point = sum(
            p["decompression_time_ns_per_point"] * p["timing_column_count"] for p in positive_parts
        ) / timing_column_count
        timing_aggregation = "column_mean"
    else:
        compression_time_ns_per_point = decompression_time_ns_per_point = None
        timing_aggregation = "unavailable"
    assert original > 0 and compressed > 0 and points > 0
    return {"dataset": dataset, "runtime": runtime, "scope": scope,
            "method": "DELTA_BINARY_PACKED", "algorithm_variant": "DELTA_BINARY_PACKED",
            "family": "DBP", "is_best_rate_in_family": True,
            "original_size_bytes": original, "compressed_size_bytes": compressed,
            "point_count": points, "compression_time_ns": encode,
            "decompression_time_ns": decode, "compression_rate": compressed / original,
            "compression_time_ns_per_point": compression_time_ns_per_point,
            "decompression_time_ns_per_point": decompression_time_ns_per_point,
            "timing_column_count": timing_column_count,
            "timing_aggregation": timing_aggregation}


def with_totals(row: dict) -> dict:
    row = dict(row)
    points = row.get("point_count", row["original_size_bytes"] / 8.0)
    row["point_count"] = points
    row.setdefault("timing_column_count", 0)
    if "timing_aggregation" not in row:
        if row["timing_column_count"] > 0:
            row["timing_aggregation"] = "column_mean"
        elif (row.get("compression_time_ns_per_point") is not None
              and row.get("decompression_time_ns_per_point") is not None):
            row["timing_aggregation"] = "dataset_aggregate"
        else:
            row["timing_aggregation"] = "unavailable"
    return row


def integrate(source: Path, site: Path, receipt_path: Path, expected_sha256: str) -> None:
    if sha(source) != expected_sha256:
        raise ValueError("Frozen measurement source SHA-256 differs from its validated receipt")
    audit = json.loads(source.read_text())
    assert audit["status"] == "complete"
    assert audit["summary"]["dataset_receipt_count"] == 169
    assert audit["summary"]["nonempty_aggregate_count"] == 4096
    assert all(r["status"] == "complete" for r in audit["runtimes"])
    assert audit["evidence"]["campaign_binding"]["status"] == "passed"
    data_path = site / "data.json"
    data = json.loads(data_path.read_text())
    baseline_sha = sha(data_path)
    catalog_path = site / "dataset-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    allowed = {d["key"] for d in catalog["datasets"]}
    assert len(allowed) == 169
    assert {d["dataset"] for d in audit["datasets"]} == allowed
    original_rows = data["dataset_results"]
    untouched = [r for r in original_rows if r["family"] != "DBP"]
    integers = [r for r in original_rows if r["family"] == "DBP" and r["scope"] == "integer"]
    assert integers and len({(r["dataset"], r["runtime"]) for r in integers}) == len(integers)
    grouped = defaultdict(list)
    precision_sources, classifications = [], {}
    for dataset in sorted(allowed):
        classifications[dataset], files = precision_map(dataset)
        precision_sources.extend(files)
    used_columns = Counter()
    exact_frames = defaultdict(dict)
    seen_columns = set()
    runtime_totals = defaultdict(Counter)
    for c in audit["columns"]:
        assert c["dataset"] in allowed and c["runtime"] in ("cpp", "java", "python")
        column_key = (c["aggregate_id"], c["runtime"])
        if column_key in seen_columns:
            raise ValueError(f"Duplicate measurement column/runtime: {column_key}")
        seen_columns.add(column_key)
        exact_frames[c["aggregate_id"]][c["runtime"]] = c["frame_sequence_sha256"]
        if c["status"] != "ok":
            assert c["status"] == "n_a" and c["valid_count"] == 0
        else:
            assert c["valid_count"] > 0 and c["verified_values"] == c["valid_count"]
        assert c["native_bytes"] == c["valid_count"] * c["width"]
        assert c["header_bytes"] == c["blocks"] * audit["measurement"]["header_bytes_per_block"]
        assert c["frame_bytes"] >= c["header_bytes"]
        assert all(math.isfinite(c[k]) and c[k] >= 0 for k in ("encode_ns", "decode_ns"))
        part = {"original_size_bytes": c["native_bytes"], "compressed_size_bytes": c["frame_bytes"],
                "point_count": c["valid_count"], "compression_time_ns": c["encode_ns"],
                "decompression_time_ns": c["decode_ns"],
                "timing_column_count": int(c["status"] == "ok"),
                "compression_time_ns_per_point": (
                    c["encode_ns"] / c["valid_count"] if c["status"] == "ok" else None
                ),
                "decompression_time_ns_per_point": (
                    c["decode_ns"] / c["valid_count"] if c["status"] == "ok" else None
                ),
                "timing_aggregation": "column_mean" if c["status"] == "ok" else "unavailable"}
        category = classifications[c["dataset"]].get(c["source_column"], "nonfixed_float")
        for scope in ("float", category):
            grouped[c["dataset"], c["runtime"], scope].append(part)
        if c["status"] == "ok":
            used_columns[c["runtime"], category] += 1
        runtime_totals[c["runtime"]].update({"columns": int(c["status"] == "ok"), "values": c["valid_count"],
                                           "schema_aggregate_count": 1,
                                           "nonempty_aggregate_count": int(c["status"] == "ok"),
                                           "n_a_aggregate_count": int(c["status"] == "n_a"),
                                           "blocks": c["blocks"],
                                           "native_bytes": c["native_bytes"], "frame_bytes": c["frame_bytes"],
                                           "encode_ns": c["encode_ns"], "decode_ns": c["decode_ns"]})
    assert len(exact_frames) == 4190
    assert all(set(v) == {"cpp", "java", "python"} and len(set(v.values())) == 1 for v in exact_frames.values())
    for runtime in ("cpp", "java", "python"):
        assert runtime_totals[runtime]["columns"] == 4096
        assert runtime_totals[runtime]["values"] == audit["summary"]["valid_value_count"]
        assert runtime_totals[runtime]["native_bytes"] == audit["summary"]["native_values_bytes"]
    floats = [make_row(ds, rt, sc, parts) for (ds, rt, sc), parts in sorted(grouped.items())
              if sum(p["point_count"] for p in parts) > 0]
    overall_parts = defaultdict(list)
    for row in integers + [r for r in floats if r["scope"] == "float"]:
        overall_parts[row["dataset"], row["runtime"]].append(with_totals(row))
    overall = [make_row(ds, rt, "overall", parts) for (ds, rt), parts in sorted(overall_parts.items())]
    rows = untouched + integers + floats + overall
    assert len({(r["dataset"], r["runtime"], r["scope"], r["algorithm_variant"]) for r in rows}) == len(rows)
    data["dataset_results"] = rows
    configs, families = defaultdict(list), defaultdict(list)
    for row in rows:
        r = with_totals(row)
        configs[r["runtime"], r["scope"], r["family"], r["method"]].append(r)
        if r["is_best_rate_in_family"]:
            families[r["runtime"], r["scope"], r["family"]].append(r)
    data["configuration_aggregates"] = [_aggregate_rows(rs, runtime=rt, scope_key=sc, family=fa, method=me, algorithm_variant=me)
                                         for (rt, sc, fa, me), rs in sorted(configs.items())]
    data["family_best_aggregates"] = [_aggregate_rows(rs, runtime=rt, scope_key=sc, family=fa, method=fa, algorithm_variant=None)
                                       for (rt, sc, fa), rs in sorted(families.items())]
    measured = {r["dataset"] for r in rows}
    note = (f"The catalog contains 169 reviewed datasets; published measurements cover {len(measured)}. "
            "Availability varies by method, runtime, and data type. Missing measurements remain unavailable.")
    catalog["summary"].update(published_result_dataset_count=len(measured),
                              catalog_only_dataset_count=169-len(measured), measurement_note=note)
    for d in catalog["datasets"]:
        if d["key"] in measured:
            d["has_published_results"] = True
            d["published_result_scope"] = "published_measurements"
            # An aggregate measurement does not certify every imported report's
            # row count. Preserve the historical comparison and leave new
            # datasets unasserted unless a dedicated row-count receipt exists.
            d.setdefault("published_result_row_count", None)
            d.setdefault("published_result_row_count_matches_catalog", None)
    write(catalog_path, catalog)
    data["datasets"] = [{**d, "has_published_measurements": d["key"] in measured} for d in catalog["datasets"]]
    data["dataset_summary"] = {**catalog["summary"], "published_measurement_dataset_count": len(measured)}
    data["dataset_overview"]["measured_dataset_count"] = len(measured)
    benchmark = data["benchmark"]
    benchmark.pop("method_audits", None)
    benchmark["scope_note"] = note
    benchmark["publication_note"] = "Available measurements are integrated into the shared leaderboard; availability and measurement context are recorded in source metadata."
    benchmark["current_result_selection"] = "Available measured configurations from methods categorized as lossless-design; verification coverage follows individual source receipts."
    benchmark["hardware"] = "Available hardware and runtime details are recorded per source in manifest.measurement_runs; imported result CSVs may omit them."
    benchmark["roundtrip_evidence"] = "Available verification receipts are recorded per source in manifest.measurement_runs; imported result CSVs alone do not establish round trips."
    benchmark["measurement_boundaries"] = [n for n in benchmark["measurement_boundaries"]
                                           if n != "No hardware identity is attached because the selected source files do not provide one."]
    timing_note = "The leaderboard combines available source runs and imported reports. Timing scores summarize recorded measurements and do not establish controlled performance comparisons across hardware or measurement contracts."
    if timing_note not in benchmark["measurement_boundaries"]:
        benchmark["measurement_boundaries"].append(timing_note)
    for scope in data["scopes"]:
        if scope["key"] == "overall":
            scope["label"] = "Available numeric"
    precision_note = "Precision categories use decimal-place frequency statistics: a dominant fraction of at least 85% is fixed; columns without statistics follow the existing non-fixed fallback."
    if precision_note not in benchmark["measurement_boundaries"]:
        benchmark["measurement_boundaries"].append(precision_note)
    benchmark["metric_definitions"]["compression_rate"] = "compressed numeric codec bytes divided by original numeric bytes; shared validity sidecars are excluded; lower is better"
    benchmark["metric_definitions"]["average_time_ns_per_point"] = "column-backed rows use an arithmetic mean of per-column nanoseconds per numeric value; whole-report-only rows use source elapsed nanoseconds divided by source point totals; legacy CSV rows retain their 8-byte-equivalent point denominator"
    benchmark["metric_definitions"]["weighted_time_ns_per_point"] = "sum of measured nanoseconds divided by sum of measured point counts"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    measurement_source = {"method": "DELTA_BINARY_PACKED", "family": "DBP", "runtimes": ["cpp", "java", "python"],
                          "source_sha256": sha(source), "source_scope_sha256": audit["evidence"]["scope_sha256"],
                          "generated_at": audit["generated_at"], "integrated_at": now,
                          "verified_column_count_per_runtime": 4096,
                          "measured_dataset_count_per_runtime": 162,
                          "encode_boundary": audit["measurement"]["encode_boundary"],
                          "decode_boundary": audit["measurement"]["decode_boundary"],
                          "size_boundary": "complete codec frame including reconstruction header; shared validity sidecars excluded",
                          "point_count_basis": "valid native-width numeric values",
                          "precision_rule": "dominant decimal-place fraction >=0.85; absent statistics use the existing non-fixed fallback",
                          "environment": audit["environment"], "evidence": audit["evidence"]}
    runs = [r for r in data["source"].get("measurement_runs", []) if r.get("method") != "DELTA_BINARY_PACKED"]
    data["source"]["measurement_runs"] = runs + [measurement_source]
    data["source"]["scope_catalog_sha256"] = sha(catalog_path)
    for f in data["source"]["source_files"]:
        if f.get("role") == "reviewed_dataset_catalog":
            f["sha256"] = sha(catalog_path)
    coverage = data["coverage"]
    coverage.update(measured_dataset_count=len(measured), dataset_result_row_count=len(rows),
                    configuration_aggregate_row_count=len(data["configuration_aggregates"]),
                    family_best_aggregate_row_count=len(data["family_best_aggregates"]))
    write(data_path, data)
    manifest_path = site / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("dbp_native_float_audit", None)
    manifest.update(data_sha256=sha(data_path), dataset_catalog_sha256=sha(catalog_path),
                    counts=coverage, source_files=data["source"]["source_files"], measurement_runs=runs+[measurement_source])
    write(manifest_path, manifest)
    receipt = {"status": "passed", "created_at_utc": now, "baseline_data_sha256": baseline_sha,
               "data_sha256": sha(data_path), "measurement_source_sha256": sha(source),
               "unchanged_other_method_rows": len(untouched), "unchanged_integer_rows": len(integers),
               "removed_previous_float_and_overall_rows": len(original_rows)-len(untouched)-len(integers),
               "new_float_scope_rows": len(floats), "new_overall_rows": len(overall),
               "measured_dataset_count": len(measured), "runtime_totals": runtime_totals,
               "precision_column_counts": {f"{rt}/{sc}": n for (rt, sc), n in sorted(used_columns.items())},
               "precision_source_files": precision_sources,
               "overall_aggregates": [r for r in data["family_best_aggregates"] if r["family"]=="DBP" and r["scope"]=="overall"]}
    write(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "unchanged_other_method_rows", "unchanged_integer_rows",
                                             "new_float_scope_rows", "new_overall_rows", "measured_dataset_count", "data_sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True,
                        help="SHA-256 from the independent frozen campaign validation receipt")
    parser.add_argument("--site", type=Path, default=SITE)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    integrate(args.measurements, args.site, args.receipt, args.expected_sha256)
