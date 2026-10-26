#!/usr/bin/env python3
"""Build a read-only evidence audit for the seven terminal r12 exclusions.

The script reads the frozen raw mirror and r12 receipt tree, re-hashes every
excluded dataset's source TsFile and every pinned failure-evidence reference,
then creates three new receipt files.  It never invokes the TsFile SDK and
does not change raw data, attempt metadata, runtime files, or runner gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


RUN_ID = "csv-xjz-20261001-r12"
FAILURE_IDS = [
    "bls_us_tech_employment_monthly",
    "cic-ids-2017-flowprep",
    "finance1k",
    "forbush_decreases",
    "qiskit_calibration_drift",
    "reg_forecasting",
    "weatherAUS",
]
EXPECTED_ARTIFACT_SHA256 = {
    "ATTEMPT_RESULTS.json": "abba87bbad5d7d8cb5095009992cbbbcaf7c6de0fda3da31b587bbb1759dd522",
    "PASS_COMPLETION_GATE.json": "02820814e097785654615adf42684f1d5feb93c5245bb369490f849ddb83cdf2",
    "LAUNCHER_FINISHED.json": "5314c37473839e1c6aeaf63fd87647b7df578a6994fc086e8149b890c878107c",
    "FINANCE1K_USER_CANCEL_INTENT_20261003T070643Z.json": "161ac9644f2674bffb17dfca9670ba4705d8955234ea3325e8105f70fb59a4a4",
    "FINANCE1K_USER_CANCEL_RECEIPT_20261003T070643Z.json": "766cc2ecf3c6b7a730b1b0c34e23fe3cee07e3a65d0ef83e7bfe715b785e90ae",
}

CLASSIFICATION = {
    "bls_us_tech_employment_monthly": {
        "exclusion_category": "conversion_or_validation_failure",
        "failure_class": "validation_type_conversion_failure",
        "reason": "validation rejected a non-finite-to-integer cast for column yoy_change_jobs",
        "root_cause": "non_finite_to_integer_cast_rejected",
        "context": "validation",
    },
    "cic-ids-2017-flowprep": {
        "exclusion_category": "conversion_or_validation_failure",
        "failure_class": "validation_type_conversion_failure",
        "reason": "validation raised TypeError while converting NoneType with int()",
        "root_cause": "none_to_integer_cast_rejected",
        "context": "validation",
    },
    "finance1k": {
        "exclusion_category": "user_cancelled",
        "failure_class": "cancelled_by_user_not_data_failure",
        "reason": "export was terminated by the user's explicit request after 153 datasets were usable",
        "root_cause": "not_applicable_user_cancellation",
        "context": "export",
    },
    "forbush_decreases": {
        "exclusion_category": "conversion_or_validation_failure",
        "failure_class": "validation_native_process_crash",
        "reason": "validation process exited on SIGSEGV in the _pilot_rescan stage",
        "root_cause": "not_identified",
        "context": "validate_project179_thulab_csv_v1.py:_pilot_rescan",
    },
    "qiskit_calibration_drift": {
        "exclusion_category": "conversion_or_validation_failure",
        "failure_class": "export_native_process_crash",
        "reason": "export worker ended with SIGSEGV; the parent reported BrokenProcessPool",
        "root_cause": "not_identified",
        "context": "rowwise_scan_start for qiskit_calibration_drift_1.tsfile",
    },
    "reg_forecasting": {
        "exclusion_category": "conversion_or_validation_failure",
        "failure_class": "export_duplicate_column_name_failure",
        "reason": "export rejected duplicate query-result column name time",
        "root_cause": "duplicate_column_names_in_query_result",
        "context": "export",
    },
    "weatherAUS": {
        "exclusion_category": "conversion_or_validation_failure",
        "failure_class": "export_native_process_crash",
        "reason": "export worker ended with SIGSEGV; the parent reported BrokenProcessPool",
        "root_cause": "not_identified",
        "context": "rowwise_scan_start for data/weather_aus.tsfile",
    },
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_under(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe relative path: {relative}")
    candidate = root.joinpath(*rel.parts)
    require(candidate.resolve().is_relative_to(root.resolve()), f"path escapes root: {relative}")
    return candidate


def check_ref(root: Path, ref: Mapping[str, Any], role: str) -> dict[str, Any]:
    require(isinstance(ref.get("path"), str), f"{role} has no path")
    require(isinstance(ref.get("sha256"), str), f"{role} has no SHA-256")
    require(isinstance(ref.get("bytes"), int), f"{role} has no byte size")
    path = safe_under(root, str(ref["path"]))
    actual_bytes = path.stat().st_size if path.exists() else -1
    actual_sha = sha256_file(path)
    require(actual_bytes == ref["bytes"], f"{role} byte mismatch: {ref['path']}")
    require(actual_sha == ref["sha256"], f"{role} SHA mismatch: {ref['path']}")
    return {
        "role": role,
        "path": ref["path"],
        "bytes": actual_bytes,
        "expected_sha256": ref["sha256"],
        "actual_sha256": actual_sha,
        "verified": True,
    }


def expected_failure_summary(attempt: Mapping[str, Any]) -> dict[str, Any]:
    failure = attempt["failure"]
    return {
        "dataset_id": attempt["dataset_id"],
        "stage": attempt["stage"],
        "error_type": failure["error_type"],
        "error_message": failure["error_message"],
        "native_crash_source": failure.get("native_crash_source"),
        "known_source_files": failure.get("known_source_files", []),
        "evidence": failure["evidence"],
    }


def collect_last_worker_events(
    output_root: Path, stage: Mapping[str, Any]
) -> list[dict[str, Any]]:
    last_events: list[dict[str, Any]] = []
    for ref in stage.get("worker_diagnostics", []):
        if not str(ref.get("path", "")).endswith(".events.jsonl"):
            continue
        path = safe_under(output_root, str(ref["path"]))
        last: dict[str, Any] | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    require(isinstance(value, dict), f"non-object diagnostic event: {path}")
                    last = value
        if last is not None:
            last_events.append(
                {
                    key: last[key]
                    for key in (
                        "event",
                        "phase",
                        "source_ordinal",
                        "source_path",
                        "source_sha256",
                        "table_name",
                    )
                    if key in last
                }
            )
    return last_events


def raw_receipt_sources(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = receipt.get("files")
    require(isinstance(files, list), "raw receipt files is not a list")
    return [
        {"path": item["path"], "sha256": item["sha256"]}
        for item in files
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and item["path"].casefold().endswith(".tsfile")
    ]


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    fields = [
        "dataset_id",
        "repo_id",
        "revision",
        "exclusion_category",
        "failure_class",
        "stage",
        "reason",
        "root_cause",
        "source_file_count",
        "source_bytes",
        "source_sha256_verified_count",
        "source_set_sha256",
        "attempt_binding_sha256",
        "failure_evidence_verified_count",
        "attempt_file_sha256",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    return stream.getvalue().encode("utf-8")


def markdown_bytes(document: Mapping[str, Any]) -> bytes:
    lines = [
        "# r12 final exclusions audit (2026-10-03)",
        "",
        "- Final run state: `complete_with_failures`.",
        "- Attempted: 160; usable/validated: 153; excluded: 7.",
        "- Exclusion accounting: 6 conversion/validation failures + 1 explicit user cancellation (`finance1k`).",
        f"- Source SHA audit: {document['totals']['source_sha256_verified_count']} / {document['totals']['source_file_count']} excluded-source TsFiles matched their pinned SHA-256 values.",
        f"- Non-finance failed-source TsFiles: {document['totals']['non_finance_source_file_count']}; finance1k source TsFiles: {document['totals']['finance1k_source_file_count']}.",
        "- Missing/cleaned transient CSV or spool files are outside this audit and are not interpreted as source corruption. No file was restored.",
        "- Native-crash entries report only the pinned stage/last context; their root cause remains `not_identified`.",
        "",
        "| dataset | category | stage | reason | source SHA | root cause |",
        "|---|---|---|---|---:|---|",
    ]
    for item in document["exclusions"]:
        reason = str(item["reason"]).replace("|", "\\|")
        lines.append(
            f"| `{item['dataset_id']}` | `{item['exclusion_category']}` | `{item['stage']}` | "
            f"{reason} | {item['source_sha256_verified_count']}/{item['source_file_count']} | "
            f"`{item['root_cause']}` |"
        )
    lines.extend(
        [
            "",
            "## Pinned final receipts",
            "",
        ]
    )
    for artifact in document["pinned_artifacts"]:
        lines.append(f"- `{artifact['name']}`: `{artifact['actual_sha256']}`")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    output_root = args.output_root.resolve()
    receipt_dir = args.receipt_dir.resolve()
    mirror = args.mirror.resolve()

    artifact_paths = {
        "ATTEMPT_RESULTS.json": output_root / "ATTEMPT_RESULTS.json",
        "PASS_COMPLETION_GATE.json": receipt_dir / "PASS_COMPLETION_GATE.json",
        "LAUNCHER_FINISHED.json": receipt_dir / "LAUNCHER_FINISHED.json",
        "FINANCE1K_USER_CANCEL_INTENT_20261003T070643Z.json": receipt_dir
        / "FINANCE1K_USER_CANCEL_INTENT_20261003T070643Z.json",
        "FINANCE1K_USER_CANCEL_RECEIPT_20261003T070643Z.json": receipt_dir
        / "FINANCE1K_USER_CANCEL_RECEIPT_20261003T070643Z.json",
    }
    pinned_artifacts: list[dict[str, Any]] = []
    for name, expected_sha in EXPECTED_ARTIFACT_SHA256.items():
        path = artifact_paths[name]
        actual_sha = sha256_file(path)
        require(actual_sha == expected_sha, f"pinned artifact SHA mismatch: {name}")
        pinned_artifacts.append(
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "verified": True,
            }
        )

    attempts = load_json(artifact_paths["ATTEMPT_RESULTS.json"])
    gate = load_json(artifact_paths["PASS_COMPLETION_GATE.json"])
    launcher = load_json(artifact_paths["LAUNCHER_FINISHED.json"])
    cancel_intent = load_json(
        artifact_paths["FINANCE1K_USER_CANCEL_INTENT_20261003T070643Z.json"]
    )
    cancel_receipt = load_json(
        artifact_paths["FINANCE1K_USER_CANCEL_RECEIPT_20261003T070643Z.json"]
    )
    require(attempts.get("run_id") == RUN_ID, "ATTEMPT_RESULTS run_id mismatch")
    require(
        attempts.get("status") == "attempts_complete_with_failures"
        and attempts.get("attempted_dataset_count") == 160
        and attempts.get("success_count") == 153
        and attempts.get("failure_count") == 7
        and attempts.get("failure_dataset_ids") == FAILURE_IDS,
        "ATTEMPT_RESULTS terminal counts or failure list mismatch",
    )
    require(
        gate.get("run_id") == RUN_ID
        and gate.get("status") == "complete_with_failures"
        and gate.get("attempted_dataset_count") == 160
        and gate.get("validated_success_count") == 153
        and gate.get("failed_dataset_count") == 7
        and gate.get("attempt_results_sha256")
        == EXPECTED_ARTIFACT_SHA256["ATTEMPT_RESULTS.json"],
        "completion gate terminal state mismatch",
    )
    require(launcher.get("exit_code") == 0, "launcher did not finish with exit 0")
    for value, label in ((cancel_intent, "cancel intent"), (cancel_receipt, "cancel receipt")):
        require(
            value.get("run_id") == RUN_ID
            and value.get("dataset_id") == "finance1k"
            and value.get("classification") == "cancelled_by_user_not_data_failure"
            and value.get("signal") == "SIGTERM",
            f"{label} identity/classification mismatch",
        )
    require(cancel_receipt.get("status") == "terminated", "cancel receipt is not terminal")

    manifest_path = mirror / "DOWNLOAD_MANIFEST.json"
    manifest_sha = sha256_file(manifest_path)
    require(
        manifest_sha == attempts.get("raw_manifest_sha256") == gate.get("raw_manifest_sha256"),
        "raw manifest SHA is not bound to final attempts/gate",
    )

    result_by_id = {item["dataset_id"]: item for item in attempts["results"]}
    summary_by_id = {item["dataset_id"]: item for item in attempts["failures"]}
    exclusions: list[dict[str, Any]] = []
    all_verified_sources = 0
    all_source_bytes = 0
    all_evidence_files = 0

    for dataset_id in FAILURE_IDS:
        attempt_path = output_root / "attempts" / dataset_id / "ATTEMPT.json"
        attempt = load_json(attempt_path)
        terminal_result = dict(result_by_id[dataset_id])
        terminal_evidence = terminal_result.pop("evidence", None)
        require(attempt == terminal_result, f"ATTEMPT/result mismatch: {dataset_id}")
        require(
            isinstance(terminal_evidence, dict)
            and terminal_evidence.get("attempt", {}).get("path")
            == f"attempts/{dataset_id}/ATTEMPT.json",
            f"terminal result attempt evidence mismatch: {dataset_id}",
        )
        require(
            expected_failure_summary(attempt) == summary_by_id[dataset_id],
            f"failure summary mismatch: {dataset_id}",
        )
        require(attempt.get("status") == "failed", f"attempt is not failed: {dataset_id}")

        raw_receipt_path = mirror / "_receipts" / f"{dataset_id}.json"
        raw_receipt = load_json(raw_receipt_path)
        receipt_sources = raw_receipt_sources(raw_receipt)
        require(
            raw_receipt.get("dataset_id") == dataset_id
            and raw_receipt.get("repo_id") == attempt.get("repo_id")
            and raw_receipt.get("revision") == attempt.get("revision")
            and raw_receipt.get("tsfile_count") == attempt.get("source_file_count")
            and raw_receipt.get("tsfile_bytes") == attempt.get("source_bytes")
            and receipt_sources == attempt.get("source_files"),
            f"raw receipt/attempt binding mismatch: {dataset_id}",
        )

        mirror_dataset = mirror / dataset_id
        actual_paths = sorted(
            str(path.relative_to(mirror_dataset).as_posix())
            for path in mirror_dataset.rglob("*")
            if path.is_file() and path.name.casefold().endswith(".tsfile")
        )
        expected_paths = sorted(item["path"] for item in receipt_sources)
        require(actual_paths == expected_paths, f"live source path set mismatch: {dataset_id}")
        source_files: list[dict[str, Any]] = []
        for source in receipt_sources:
            source_path = safe_under(mirror_dataset, str(source["path"]))
            actual_sha = sha256_file(source_path)
            require(actual_sha == source["sha256"], f"source SHA mismatch: {dataset_id}/{source['path']}")
            source_files.append(
                {
                    "path": source["path"],
                    "bytes": source_path.stat().st_size,
                    "expected_sha256": source["sha256"],
                    "actual_sha256": actual_sha,
                    "verified": True,
                }
            )
        require(
            sum(item["bytes"] for item in source_files) == attempt["source_bytes"],
            f"live source byte total mismatch: {dataset_id}",
        )

        evidence_files: list[dict[str, Any]] = []
        for name, ref in sorted(terminal_evidence.items()):
            evidence_files.append(check_ref(output_root, ref, f"terminal_result:{name}"))
        require(
            {key: terminal_evidence[key] for key in attempt.get("log_evidence", {})}
            == attempt.get("log_evidence", {}),
            f"terminal result/log evidence mismatch: {dataset_id}",
        )
        for index, ref in enumerate(attempt["failure"].get("evidence", [])):
            evidence_files.append(check_ref(output_root, ref, f"failure_evidence[{index}]"))

        failed_stage_receipts: list[dict[str, Any]] = []
        for stage_index, stage in enumerate(attempt["failure"].get("failed_export_stages", [])):
            stage_receipt = check_ref(
                output_root, stage["failure_receipt"], f"failed_stage[{stage_index}]:receipt"
            )
            evidence_files.append(stage_receipt)
            actual_stage_value = load_json(safe_under(output_root, stage["failure_receipt"]["path"]))
            require(
                actual_stage_value == stage["failure_receipt_value"],
                f"embedded failed-stage receipt mismatch: {dataset_id}",
            )
            worker_refs: list[dict[str, Any]] = []
            for diag_index, ref in enumerate(stage.get("worker_diagnostics", [])):
                checked = check_ref(
                    output_root,
                    ref,
                    f"failed_stage[{stage_index}]:worker_diagnostic[{diag_index}]",
                )
                evidence_files.append(checked)
                worker_refs.append(checked)
            relative_worker_refs = stage["failure_receipt_value"].get("worker_diagnostics", [])
            require(
                len(relative_worker_refs) == len(worker_refs),
                f"worker diagnostic reference count mismatch: {dataset_id}",
            )
            for full_ref, relative_ref in zip(stage.get("worker_diagnostics", []), relative_worker_refs):
                require(
                    full_ref["sha256"] == relative_ref["sha256"]
                    and full_ref["bytes"] == relative_ref["bytes"]
                    and PurePosixPath(full_ref["path"]).name
                    == PurePosixPath(relative_ref["path"]).name,
                    f"worker diagnostic reference mismatch: {dataset_id}",
                )
            failed_stage_receipts.append(
                {
                    "stage_path": stage["stage_path"],
                    "failure_receipt": stage_receipt,
                    "worker_diagnostics": worker_refs,
                    "last_worker_events": collect_last_worker_events(output_root, stage),
                }
            )

        classification = CLASSIFICATION[dataset_id]
        if classification["root_cause"] == "not_identified":
            require(
                dataset_id in {"forbush_decreases", "qiskit_calibration_drift", "weatherAUS"},
                f"unexpected unidentified cause: {dataset_id}",
            )
        if dataset_id == "finance1k":
            require(
                attempt["failure"].get("error_message") == "ERROR: termination requested by signal 15"
                and attempt.get("stage") == "export",
                "finance1k attempt is not the expected SIGTERM terminal record",
            )

        source_set_payload = [
            {"path": item["path"], "bytes": item["bytes"], "sha256": item["actual_sha256"]}
            for item in source_files
        ]
        exclusion = {
            "dataset_id": dataset_id,
            "repo_id": attempt["repo_id"],
            "revision": attempt["revision"],
            **classification,
            "stage": attempt["stage"],
            "attempt_error_type": attempt["failure"]["error_type"],
            "attempt_error_message": attempt["failure"]["error_message"],
            "attempt_returncode": attempt["failure"].get("returncode"),
            "attempt_signal": attempt["failure"].get("signal"),
            "attempt_native_crash_source": attempt["failure"].get("native_crash_source"),
            "attempt_binding_sha256": attempt["binding_sha256"],
            "attempt_file_sha256": sha256_file(attempt_path),
            "raw_receipt_path": str(raw_receipt_path),
            "raw_receipt_sha256": sha256_file(raw_receipt_path),
            "source_file_count": len(source_files),
            "source_bytes": sum(item["bytes"] for item in source_files),
            "source_sha256_verified_count": len(source_files),
            "source_set_sha256": canonical_sha256(source_set_payload),
            "source_files": source_files,
            "failure_evidence_verified_count": len(evidence_files),
            "failure_evidence": evidence_files,
            "failed_export_stage_evidence": failed_stage_receipts,
            "transient_output_policy": "partial CSV/spool files are not required evidence and are not restored",
        }
        exclusions.append(exclusion)
        all_verified_sources += len(source_files)
        all_source_bytes += exclusion["source_bytes"]
        all_evidence_files += len(evidence_files)

    require(all_verified_sources == 953, f"excluded source total is {all_verified_sources}, expected 953")
    require(
        sum(item["source_file_count"] for item in exclusions if item["dataset_id"] != "finance1k")
        == 60,
        "non-finance failed source count is not 60",
    )
    require(
        next(item for item in exclusions if item["dataset_id"] == "finance1k")["source_file_count"]
        == 893,
        "finance1k source count is not 893",
    )

    document = {
        "schema": "thulab-r12-final-exclusions-audit-v1",
        "status": "verified",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "scope": "seven terminal non-usable datasets only",
        "method": "pure Python SHA-256 and JSON/path binding checks; no TsFile SDK or value scan",
        "final_state": {
            "status": "complete_with_failures",
            "attempted_dataset_count": 160,
            "validated_usable_dataset_count": 153,
            "excluded_dataset_count": 7,
            "conversion_or_validation_failure_count": 6,
            "user_cancelled_count": 1,
        },
        "interpretation": {
            "finance1k": "cancelled_by_user_not_data_failure",
            "native_crash_root_cause": "not_identified unless pinned evidence states otherwise",
            "source_integrity": "all excluded source paths and SHA-256 values matched; this does not diagnose failure root cause",
            "transient_outputs": "missing or cleaned partial CSV/spool files are expected and were neither required nor restored",
        },
        "raw_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "bound_to_attempt_results_and_gate": True,
        },
        "pinned_artifacts": pinned_artifacts,
        "totals": {
            "source_file_count": all_verified_sources,
            "source_sha256_verified_count": all_verified_sources,
            "source_bytes": all_source_bytes,
            "non_finance_source_file_count": 60,
            "finance1k_source_file_count": 893,
            "failure_evidence_verified_count": all_evidence_files,
        },
        "exclusions": exclusions,
    }
    json_payload = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    csv_payload = csv_bytes(exclusions)
    md_payload = markdown_bytes(document)
    return document, json_payload, csv_payload, md_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--basename", default="FINAL_EXCLUSIONS_20261003")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document, json_payload, csv_payload, md_payload = audit(args)
    outputs: list[dict[str, Any]] = []
    if not args.verify_only:
        output_dir = (args.output_dir or args.receipt_dir).resolve()
        for suffix, payload in (("json", json_payload), ("csv", csv_payload), ("md", md_payload)):
            path = output_dir / f"{args.basename}.{suffix}"
            write_exclusive(path, payload)
            outputs.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    print(
        json.dumps(
            {
                "status": document["status"],
                "source_sha256_verified_count": document["totals"]["source_sha256_verified_count"],
                "failure_evidence_verified_count": document["totals"]["failure_evidence_verified_count"],
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
