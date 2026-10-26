#!/usr/bin/env python3
"""Audit the small Fitbit source as a *candidate* independent test source.

This is intentionally a bootstrap audit, not an admission into the frozen
cohort.  It verifies the exact BenchmarkHost revision, source bytes, README evidence,
and the table schema exposed by the native TsFile reader.  Timestamp semantics
come only from the frozen README claim (UTC milliseconds); no unit, timezone,
or ordering is inferred from values.  The script never changes the cohort
manifest and never marks the candidate as formal test evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

from prepare_weighted_grpo_tsfile_v1 import PreparationError, inspect_tsfile_source


DATASET_ID = "sleep_score_fitbit"
REPO_ID = "BenchmarkHost/sleep_score_fitbit"
REVISION = "24b55bb0692886173e16ecd9507926c20a3afb45"
SOURCE_REPO_PATH = "sleep_score_fitbit.tsfile"
SOURCE_BYTES = 5299
SOURCE_SHA256 = "435836189bc5937b4aca850b2e28734500ae5d5959767f391c0df881bf4ead00"
README_REPO_PATH = "README.md"
README_SHA256 = "b0a50d7b67f33931f74dc8b5076b916850bc95a0758dccde8b4c13c56e14a605"
SOURCE_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/"
    f"{SOURCE_REPO_PATH}"
)
README_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/raw/{REVISION}/{README_REPO_PATH}"
)

EXPECTED_FIELDS = {
    "overall_score": "INT64",
    "revitalization_score": "INT64",
    "deep_sleep_in_minutes": "INT64",
    "resting_heart_rate": "INT64",
    "restlessness": "DOUBLE",
}
COHORT_MANIFEST = Path(__file__).with_name("weighted_grpo_timeseries_cohort_v1.json")
SOURCE_GROUP = "hf:aai530-group6/sleep-score-fitbit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "weighted-grpo-source-audit/1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def validate_readme_evidence(text: str) -> dict[str, Any]:
    """Validate only explicit source-card claims; do not infer from numbers."""
    required = {
        "utc_timestamp": "parsed as UTC" in text
        and "Time | TIME | INT64 (ms)" in text,
        "record_count": "291 observations" in text,
        "source_range": "2022-12-23T09:20:30Z" in text
        and "2023-11-18T07:59:00Z" in text,
        "all_rows_retained": "All five measurements and all 291 rows are retained" in text,
        "sorted_by_time": "sorted by Time" in text,
    }
    return {
        "claims": required,
        "passed": all(required.values()),
        "timestamp_policy": (
            "explicit_utc_epoch_milliseconds_from_source_card"
            if required["utc_timestamp"]
            else "missing_explicit_utc_claim"
        ),
    }


def source_group_overlaps_frozen_cohort(
    manifest: Mapping[str, Any], source_group: str = SOURCE_GROUP
) -> bool:
    """Check overlap only with admitted train/validation source groups.

    The cohort manifest intentionally contains test-candidate placeholders.  A
    candidate matching its own placeholder is not an overlap with the frozen
    assignment; only a source group already assigned to train/validation is.
    """
    frozen_groups = {
        str(entry.get("source_group"))
        for entry in manifest.get("datasets", [])
        if isinstance(entry, Mapping)
        and entry.get("role") == "train_validation"
    }
    return source_group in frozen_groups


def _validate_source_identity(source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise PreparationError(f"Fitbit source is missing: {source}")
    actual_bytes = source.stat().st_size
    actual_sha = sha256_file(source)
    if actual_bytes != SOURCE_BYTES or actual_sha != SOURCE_SHA256:
        raise PreparationError(
            "FITBIT_SOURCE_IDENTITY_BLOCKER: exact revision bytes/hash mismatch: "
            f"expected=({SOURCE_BYTES}, {SOURCE_SHA256}), "
            f"actual=({actual_bytes}, {actual_sha})"
        )
    with source.open("rb") as handle:
        header = handle.read(6)
        handle.seek(-6, 2)
        trailer = handle.read(6)
    if header != b"TsFile" or trailer != b"TsFile":
        raise PreparationError("FITBIT_CONTAINER_BLOCKER: invalid TsFile magic")
    return {"bytes": actual_bytes, "sha256": actual_sha}


def audit_fitbit(source: Path, readme: Path) -> dict[str, Any]:
    identity = _validate_source_identity(source)
    if not readme.is_file():
        raise PreparationError(f"Fitbit README is missing: {readme}")
    readme_sha = sha256_file(readme)
    if readme_sha != README_SHA256:
        raise PreparationError(
            "FITBIT_README_IDENTITY_BLOCKER: exact revision README hash mismatch: "
            f"expected={README_SHA256}, actual={readme_sha}"
        )
    evidence = validate_readme_evidence(readme.read_text(encoding="utf-8"))
    if not evidence["passed"]:
        raise PreparationError(
            "FITBIT_TIMESTAMP_EVIDENCE_BLOCKER: README lacks one or more explicit "
            f"claims: {evidence['claims']}"
        )
    inspection = inspect_tsfile_source(source)
    tables = inspection.get("table_schemas", [])
    matching = [table for table in tables if table.get("table_name") == DATASET_ID]
    if len(matching) != 1:
        raise PreparationError(
            f"FITBIT_SCHEMA_BLOCKER: expected one table named {DATASET_ID!r}"
        )
    fields = {
        str(column["name"]): str(column["tsfile_type"])
        for column in matching[0].get("columns", [])
    }
    if fields != EXPECTED_FIELDS:
        raise PreparationError(
            f"FITBIT_SCHEMA_BLOCKER: expected fields {EXPECTED_FIELDS}, got {fields}"
        )
    cohort = json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))
    cohort_overlap = source_group_overlaps_frozen_cohort(cohort)
    if cohort_overlap:
        raise PreparationError(
            "FITBIT_LINEAGE_BLOCKER: source group overlaps a frozen train/validation group"
        )
    return {
        "schema": "weighted-grpo-fitbit-test-source-audit-v1",
        "dataset_id": DATASET_ID,
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "repo_path": SOURCE_REPO_PATH,
            "url": SOURCE_URL,
            **identity,
        },
        "readme": {
            "repo_path": README_REPO_PATH,
            "url": README_URL,
            "sha256": readme_sha,
            "evidence": evidence,
        },
        "inspection": inspection,
        "timestamp": {
            "source_unit": "ms",
            "timezone": "UTC",
            "semantic_basis": "explicit_source_card_claim",
            "value_inference": False,
        },
        "cohort_overlap_check": {
            "source_group": SOURCE_GROUP,
            "frozen_cohort_source_group_overlap": cohort_overlap,
            "manifest_path": str(COHORT_MANIFEST),
        },
        "status": {
            "schema_timestamp_audit": "passed",
            "independent_source_candidate": "qualified_for_manifest_review",
            "formal_test_admission": False,
            "reason_not_admitted": (
                "The frozen cohort manifest and assignment fingerprint are unchanged; "
                "a candidate audit cannot silently add a test source."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.download:
        _download(SOURCE_URL, args.source)
        _download(README_URL, args.readme)
    receipt = audit_fitbit(args.source, args.readme)
    encoded = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
