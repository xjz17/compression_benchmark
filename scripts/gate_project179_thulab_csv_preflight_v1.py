#!/usr/bin/env python3
"""Close the complete-schema and independently rescanned DMI launch gate."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SHA = re.compile(r"[0-9a-f]{64}")
DMI = "dmi-aarhus-weather-data"
DMI_SOURCES = {
    "data/data.tsfile": {
        "sha256": "d40d9fdca20f018eb48676d9730033c439ab99e1a821d7b94cce01876b3a1ff6",
        "schema_sha256": "a06fde8814d9018899ce0bc948c238096cc95d80089731810192dd32bd7ca878",
        "rows": 2382, "source_type": "INT64", "source_category": "FIELD",
        "reference_epoch_s_min": 1764370800, "reference_epoch_s_max": 1772773200,
        "time_epoch_s_min": 1764374400, "time_epoch_s_max": 1772946000,
        "conversion": "policy_source_bound_integer_epoch_to_int64_us",
    },
    "data/training-matrix.tsfile": {
        "sha256": "9938b0f54e1629e2dadbc49780641a3c2c8f589c1dcd59b8526be0a816082ae4",
        "schema_sha256": "9aeba98632a965f55b2d440387d74ddfb6ef9423cc7ca524b0c960b43ca12cd1",
        "rows": 41528, "source_type": "STRING", "source_category": "TAG",
        "reference_epoch_s_min": 1768950000, "reference_epoch_s_max": 1781053200,
        "time_epoch_s_min": 1768953600, "time_epoch_s_max": 1781060400,
        "conversion": "policy_temporal_string_to_int64_us",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("schema/DMI gate failed: " + message)


def digest(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    digest(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def check_schema(report: dict, ids: list[str], scope_sha: str,
                 policy_sha: str, exporter_sha: str) -> None:
    require(len(ids) == 160 and len(set(ids)) == 160, "not exact 160 scope")
    require(report.get("status") == "passed" and report.get("error_count") == 0
            and report.get("errors") == [], "schema preflight did not pass")
    require(report.get("dataset_count") == 160 and report.get("source_count") == 1524
            and report.get("dataset_ids") == ids, "preflight coverage is incomplete")
    require(report.get("scope_sha256") == scope_sha
            and report.get("timestamp_policy_file_sha256") == policy_sha
            and report.get("exporter_sha256") == exporter_sha
            and report.get("tsfile_version") == "2.1.5", "preflight hash/runtime mismatch")
    rows = report.get("datasets")
    require(isinstance(rows, list) and [r.get("dataset_id") for r in rows] == ids,
            "preflight dataset order/set mismatch")
    total = 0
    for row in rows:
        sources = row.get("sources")
        require(row.get("status") == "passed" and isinstance(sources, list)
                and len(sources) == row.get("source_file_count") and bool(sources),
                "dataset schema sources incomplete")
        require(SHA.fullmatch(str(row.get("binding_sha256", ""))) is not None,
                "missing frozen binding hash")
        expected = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode()).hexdigest()
        require(row.get("schema_sha256") == expected, "schema aggregate hash mismatch")
        require(len({s.get("source_path") for s in sources}) == len(sources),
                "duplicate schema source")
        for source in sources:
            require(SHA.fullmatch(str(source.get("source_sha256", ""))) is not None
                    and SHA.fullmatch(str(source.get("schema_sha256", ""))) is not None,
                    "source SHA/schema SHA missing")
        total += len(sources)
        if row["dataset_id"] == DMI:
            require(len(sources) == 2, "DMI source set changed")
            for source in sources:
                frozen = DMI_SOURCES.get(source["source_path"])
                require(frozen is not None and source["source_sha256"] == frozen["sha256"]
                        and source["schema_sha256"] == frozen["schema_sha256"],
                        "DMI frozen source/schema mismatch")
    require(total == 1524, "source coverage counter mismatch")


def check_dmi(report: dict, output: Path, scope_sha: str,
              policy_sha: str, exporter_sha: str) -> dict:
    require(report.get("status") == "passed" and report.get("dataset_count") == 1,
            "independent DMI validation did not pass")
    require([r.get("dataset_id") for r in report.get("datasets", [])] == [DMI],
            "DMI validation dataset mismatch")
    rescans = report.get("pilot_source_rescans", [])
    require(len(rescans) == 1, "missing independent DMI full-value scan")
    scan = rescans[0]
    require(scan.get("dataset_id") == DMI and scan.get("status") == "passed"
            and scan.get("source_files") == 2 and scan.get("source_rows") == 43910
            and scan.get("compared_output_rows") == 43910, "DMI full-row coverage mismatch")
    audits = scan.get("temporal_unit_audits", [])
    require(len(audits) == 2 and [r.get("source_path") for r in audits] == list(DMI_SOURCES),
            "DMI per-source temporal audit missing or duplicate")
    for audit in audits:
        frozen = DMI_SOURCES[audit["source_path"]]
        require(audit.get("source_sha256") == frozen["sha256"]
                and audit.get("source_schema_sha256") == frozen["schema_sha256"]
                and audit.get("source_rows") == frozen["rows"]
                and audit.get("reference_time_source_type") == frozen["source_type"]
                and audit.get("reference_time_source_category") == frozen["source_category"]
                and audit.get("conversion") == frozen["conversion"], "DMI audit binding mismatch")
        require(all(audit.get(k) == 0 for k in (
            "reference_time_nulls", "lead_time_hours_nulls", "int64_us_overflow_count",
            "lead_relation_mismatch_count")), "DMI NULL/overflow/unit proof mismatch")
        require(all(audit.get(k) == frozen[k] for k in (
            "reference_epoch_s_min", "reference_epoch_s_max", "time_epoch_s_min", "time_epoch_s_max")),
            "DMI full-source time ranges differ")
    require(output.is_dir() and not output.is_symlink(), "DMI output is not a real directory")
    receipt_path = output / DMI / "dataset_manifest.json"
    receipt = load(receipt_path)
    require(receipt.get("status") == "complete" and receipt.get("dataset_id") == DMI
            and receipt.get("scope_sha256") == scope_sha
            and receipt.get("timestamp_policy_file_sha256") == policy_sha
            and receipt.get("exporter_sha256") == exporter_sha, "DMI receipt is not bound")
    require(int(receipt["emitted_rows_decimal_20"]) == 43910
            and int(receipt["lost_rows_due_to_byte_cap_decimal_20"]) == 0
            and receipt.get("row_exception_filtered_rows") == 0,
            "DMI conversion lost or filtered rows")
    source_rows = receipt.get("source_tsfiles", [])
    require(len(source_rows) == 2 and {s["path"]: s["sha256"] for s in source_rows}
            == {p: s["sha256"] for p, s in DMI_SOURCES.items()}, "DMI source binding changed")
    files = receipt.get("files", [])
    require([f["path"] for f in files] ==
            ["data.csv", "series_map.csv", "columns.csv", "dataset_manifest.json"],
            "DMI output file set mismatch")
    total = 0
    hashes = {}
    for f in files:
        path = output / DMI / f["path"]
        actual = digest(path)
        size = path.stat().st_size
        require(size == int(f["bytes_decimal_20"]), "DMI output size mismatch")
        require(f["path"] == "dataset_manifest.json" or actual == f["sha256"],
                "DMI output hash mismatch")
        hashes[f["path"]] = actual
        total += size
    require(total <= 1_000_000_000, "DMI four-file cap exceeded")
    with (output / DMI / "columns.csv").open("r", encoding="utf-8", newline="") as f:
        references = [row for row in csv.DictReader(f) if row.get("source_name") == "reference_time"]
    require(len(references) == 2, "DMI reference_time column missing or duplicated")
    require({(r.get("table_name"), r.get("source_type"), r.get("output_dtype"), r.get("conversion"))
             for r in references} == {
                 ("dmi_aarhus_data", "INT64", "int64", "policy_source_bound_integer_epoch_to_int64_us"),
                 ("dmi_aarhus_training_matrix", "STRING", "int64", "policy_temporal_string_to_int64_us")},
            "DMI reference_time retained raw seconds or wrong conversion")
    return {"source_rows": 43910, "output_rows": 43910, "output_hashes": hashes,
            "temporal_unit_audits": audits,
            "actual_capped_bytes": total}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("schema-report", "dmi-report", "dmi-output", "scope", "timestamp-policies", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-exporter-sha256", required=True)
    p.add_argument("--expected-validator-sha256", required=True)
    a = p.parse_args()
    scope_sha, policy_sha = digest(a.scope), digest(a.timestamp_policies)
    ids = [r["id"] for r in load(a.scope)["timestamp_datasets"]]
    check_schema(load(a.schema_report), ids, scope_sha, policy_sha, a.expected_exporter_sha256)
    evidence = check_dmi(load(a.dmi_report), a.dmi_output, scope_sha, policy_sha,
                         a.expected_exporter_sha256)
    gate = {"schema": "project179-thulab-csv-schema-dmi-gate-v1", "status": "complete",
            "gate": "schema_and_dmi_compatibility_verified",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "dataset_count": 160, "source_count": 1524,
            "scope_sha256": scope_sha, "timestamp_policy_file_sha256": policy_sha,
            "exporter_sha256": a.expected_exporter_sha256,
            "validator_sha256": a.expected_validator_sha256,
            "schema_report_sha256": digest(a.schema_report),
            "dmi_validation_sha256": digest(a.dmi_report), "dmi_evidence": evidence}
    require(not a.output.exists() and not a.output.is_symlink(), "gate already exists")
    fd, name = tempfile.mkstemp(prefix=".schema-dmi-gate-", dir=a.output.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(gate, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, a.output)
        directory_fd = os.open(a.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"status": "complete", "gate": str(a.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
