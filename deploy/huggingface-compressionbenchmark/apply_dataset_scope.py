"""Apply the reviewed 164-dataset public scope to a measured snapshot.

This is a scope filter, not a benchmark runner: retained individual measurements
are copied unchanged, excluded measurements are removed, and aggregates are
recomputed. Catalog-only datasets receive no fabricated measurement rows.
"""
from __future__ import annotations

from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path

from export_results import _aggregate_rows

ROOT = Path(__file__).resolve().parent

# These five catalog-only entries were absent from the deployed leaderboard's
# effective measured scope.  Keep the public benchmark fixed to the reviewed
# 164 real datasets even if a later input catalog happens to contain them.
OUTSIDE_WEBSITE164_SCOPE = frozenset({
    "app_flow",
    "ethereum_signals_sample",
    "github_top_developers",
    "tiktok_trending_hashtags_music",
    "youtube_trending_2019_2020",
})


def _is_glza(row: dict) -> bool:
    return str(row.get("family", "")).casefold() == "glza" or "glza" in str(
        row.get("method", "")
    ).casefold()


def _contains_glza(value: object) -> bool:
    if isinstance(value, dict):
        return any("glza" in str(key).casefold() or _contains_glza(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_glza(item) for item in value)
    return "glza" in str(value).casefold()


def _filter_reference_catalog(path: Path) -> str:
    """Remove the excluded method from every public documentation domain."""
    catalog = json.loads(path.read_text())
    removed = [item for item in catalog.get("methods", []) if _is_glza(item)]
    catalog["methods"] = [
        item for item in catalog.get("methods", []) if not _is_glza(item)
    ]
    source = catalog.setdefault("source", {})
    removed.extend(item for item in source.get("ppt_method_aliases", []) if _is_glza(item))
    source["ppt_method_aliases"] = [
        item for item in source.get("ppt_method_aliases", []) if not _is_glza(item)
    ]
    removed.extend(item for item in source.get("additional_measured_families", []) if _is_glza(item))
    source["additional_measured_families"] = [
        item for item in source.get("additional_measured_families", []) if not _is_glza(item)
    ]
    removed_reference_ids = {
        str(value) for item in removed for value in item.get("reference_ids", [])
    }
    referenced = set()
    for item in catalog.get("methods", []) + catalog.get("operators", []):
        referenced.update(str(value) for value in item.get("reference_ids", []))
        for implementation in item.get("implementations", []):
            referenced.update(str(value) for value in implementation.get("reference_ids", []))
    for item in source.get("ppt_method_aliases", []) + source.get("additional_measured_families", []):
        referenced.update(str(value) for value in item.get("reference_ids", []))
    catalog["references"] = [
        item for item in catalog.get("references", [])
        if not (
            str(item.get("id")) in removed_reference_ids
            and str(item.get("id")) not in referenced
        ) and not _contains_glza(item)
    ]
    if _contains_glza(catalog):
        raise ValueError("GLZA remains in the filtered public reference catalog")
    path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return sha(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply() -> None:
    catalog_path = ROOT / "dataset-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    data_path = ROOT / "data.json"
    data = json.loads(data_path.read_text())
    reference_catalog_path = ROOT / "reference-catalog.json"
    reference_catalog_sha256 = _filter_reference_catalog(reference_catalog_path)
    original_catalog = list(catalog["datasets"])
    original_keys = {d["key"] for d in original_catalog}
    if len(original_keys) != len(original_catalog):
        raise ValueError("The public catalog contains duplicate dataset keys")
    if len(original_keys) == 169 and not OUTSIDE_WEBSITE164_SCOPE <= original_keys:
        raise ValueError("The 169-dataset catalog does not identify the website164 exclusions")
    if len(original_keys) == 164 and original_keys & OUTSIDE_WEBSITE164_SCOPE:
        raise ValueError("The 164-dataset catalog still contains an excluded catalog-only dataset")
    if len(original_keys) not in {164, 169}:
        raise ValueError(f"Expected a 164- or 169-dataset input catalog, found {len(original_keys)}")
    catalog["datasets"] = [
        item for item in original_catalog if item["key"] not in OUTSIDE_WEBSITE164_SCOPE
    ]
    allowed = {d["key"] for d in catalog["datasets"]}
    if len(allowed) != 164 or len(catalog["datasets"]) != 164:
        raise ValueError(f"The reviewed public scope must contain 164 unique datasets, found {len(allowed)}")
    excluded = {d["key"] for d in catalog.get("excluded", [])}
    if allowed & excluded:
        raise ValueError("Excluded datasets are present in the public catalog")
    old_rows = data["dataset_results"]
    rows = [r for r in old_rows if r["dataset"] in allowed and not _is_glza(r)]
    if not rows:
        raise ValueError("The snapshot has no measurements in the reviewed scope")
    measured = {r["dataset"] for r in rows}
    old_sha = sha(data_path)
    data["dataset_results"] = rows
    data["datasets"] = [{**d, "has_published_measurements": d["key"] in measured}
                        for d in catalog["datasets"]]
    configurations = defaultdict(list)
    families = defaultdict(list)
    for original in rows:
        r = dict(original)
        # Preserve the exporter's 8-byte-equivalent denominator, including
        # historical rows whose source byte totals are not divisible by eight.
        r.setdefault("point_count", r["original_size_bytes"] / 8.0)
        # A per-column arithmetic mean cannot be expanded back into elapsed
        # time from a dataset point total. Leave weighted elapsed-time metrics
        # unavailable unless the source row still carries the original totals.
        configurations[r["runtime"], r["scope"], r["family"], r["method"]].append(r)
        if r["is_best_rate_in_family"]:
            families[r["runtime"], r["scope"], r["family"]].append(r)
    data["configuration_aggregates"] = [
        _aggregate_rows(rs, runtime=rt, scope_key=sc, family=fa, method=me, algorithm_variant=me)
        for (rt, sc, fa, me), rs in sorted(configurations.items())]
    data["family_best_aggregates"] = [
        _aggregate_rows(rs, runtime=rt, scope_key=sc, family=fa, method=fa, algorithm_variant=None)
        for (rt, sc, fa), rs in sorted(families.items())]
    method_keys = {(r["method"], r["family"]) for r in rows}
    data["methods"] = [m for m in data["methods"] if (m["name"], m["family"]) in method_keys]
    data["report_availability"] = [r for r in data["report_availability"] if r["dataset"] in allowed]
    # Retire the unrelated older paper aggregation instead of presenting it as
    # evidence for the new catalog scope.
    data.pop("paper_table3", None)
    category_labels = {
        item["category"]: item.get("category_label", item["category"])
        for item in catalog["datasets"]
    }
    category_counts = {
        key: sum(item["category"] == key for item in catalog["datasets"])
        for key in category_labels
    }
    catalog["summary"].update(
        dataset_count=164,
        catalog_only_dataset_count=0,
        published_result_dataset_count=len(measured),
        published_measurement_dataset_count=len(measured),
        empty_dataset_count=sum((item.get("row_count") or 0) == 0 for item in catalog["datasets"]),
        legacy_count=sum(item.get("cohort") == "legacy" for item in catalog["datasets"]),
        thulab_count=sum(item.get("cohort") == "thulab" for item in catalog["datasets"]),
        categories=[
            {"key": key, "label": category_labels[key], "count": category_counts[key]}
            for key in category_labels
        ],
        origin_classes={
            key: sum(item.get("origin_class") == key for item in catalog["datasets"])
            for key in ("real_measured", "derived_from_real")
        },
        provenance_statuses={
            key: sum(item.get("provenance_status") == key for item in catalog["datasets"])
            for key in ("source_supported", "incomplete")
        },
        measurement_note=(
            "The reviewed scope contains 89 real-measured and 75 derived-from-real datasets; "
            "15 have incomplete source provenance. Availability varies by method, runtime, "
            "and data type; missing measurements remain unavailable."
        ),
    )
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    data["dataset_overview"] = {
        "dataset_count": 164, "measured_dataset_count": len(measured),
        "total_rows": sum(d.get("row_count") or 0 for d in catalog["datasets"]),
        "total_rows_available_dataset_count": sum(d.get("row_count") is not None for d in catalog["datasets"]),
        "provenance_note": "Dataset catalog statistics follow the reviewed 164-real-dataset scope. Compression metrics use only published measurement rows."}
    benchmark = data["benchmark"]
    benchmark.pop("paper_result_selection", None)
    benchmark["metric_definitions"] = {k: v for k, v in benchmark["metric_definitions"].items() if not k.startswith("paper_")}
    for field in ("notes", "measurement_boundaries"):
        benchmark[field] = [s for s in benchmark.get(field, []) if "paper" not in s.lower()]
    benchmark["scope_note"] = (
        f"The benchmark contains 164 reviewed datasets: 89 real-measured and 75 derived-from-real; "
        f"15 have incomplete source provenance. Published measurement rows currently cover {len(measured)} "
        "of them. Missing combinations remain unavailable rather than zero."
    )
    benchmark["publication_note"] = (
        "All Native86 rows require hash-bound matching benchmark and fresh-cold results. "
        "The server-192 cohort also carries independent pair-verification receipts; the "
        "server-166 cohort carries strict source/admission or fresh-cold recovery receipts."
    )
    source = data["source"]
    source["source_files"] = [s for s in source["source_files"] if s.get("role") not in ("paper_table3", "dataset_scope", "dataset_catalog")]
    source["source_files"].append({"role": "reviewed_dataset_catalog", "path": "dataset-catalog.json", "sha256": sha(catalog_path), "status": "reviewed_public_catalog"})
    source["result_csv_files_used"] = [s for s in source["result_csv_files_used"] if len(Path(s["path"]).parts) >= 3 and Path(s["path"]).parts[2] in allowed]
    source["scope_applied_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source.setdefault("scope_input_snapshot_sha256", old_sha)
    source["scope_catalog_sha256"] = sha(catalog_path)
    coverage = data["coverage"]
    coverage.update(dataset_count=164, catalog_dataset_count=164, measured_dataset_count=len(measured),
                    dataset_result_row_count=len(rows), configuration_aggregate_row_count=len(data["configuration_aggregates"]),
                    family_best_aggregate_row_count=len(data["family_best_aggregates"]),
                    method_family_count=len({r["family"] for r in rows}), method_configuration_count=len({r["method"] for r in rows}),
                    runtime_configuration_count=len({(r["runtime"], r["method"]) for r in rows}),
                    source_report_file_count=len(source["result_csv_files_used"]), paper_leaderboard_row_count=0,
                    excluded_datasets=sorted(excluded | OUTSIDE_WEBSITE164_SCOPE | set(coverage.get("excluded_datasets", []))))
    coverage.update(
        timing_column_observation_count=sum(r.get("timing_column_count", 0) for r in rows),
        timing_column_mean_dataset_result_count=sum(
            r.get("timing_aggregation") == "column_mean" for r in rows
        ),
        timing_dataset_aggregate_result_count=sum(
            r.get("timing_aggregation") == "dataset_aggregate" for r in rows
        ),
        timing_unavailable_dataset_result_count=sum(
            r.get("timing_aggregation") == "unavailable" for r in rows
        ),
    )
    data["dataset_summary"] = {**catalog["summary"], "published_measurement_dataset_count": len(measured)}
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(counts=coverage, data_sha256=sha(data_path), source_files=source["source_files"],
                    result_csv_files=source["result_csv_files_used"], dataset_catalog_sha256=sha(catalog_path),
                    reference_catalog_sha256=reference_catalog_sha256)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"catalog_datasets": 164, "measured_datasets": len(measured), "retained_rows": len(rows),
                      "removed_rows": len(old_rows) - len(rows), "data_sha256": sha(data_path)}))


if __name__ == "__main__":
    apply()
