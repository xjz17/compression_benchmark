#!/usr/bin/env python3
"""Build evidence-only coverage reports for operator-harness-v2 receipts.

This program never executes a codec.  It accepts one deliberately selected
local results tree, verifies each durable RESULT.json receipt, and reports only
what those receipts prove.  Catalog ``planned`` flags are descriptive input;
they are never treated as execution evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "operator-harness-coverage-v2"
EXPECTED_CASE_CONTRACT = 214
DISABLED_IDS = ("A106", "A107", "A108", "A109", "A110")
WRAPPER_ID = "operator102.lossless_domain_fallback.selector_raw.v1"


class CoverageError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoverageError(message)


def canonical_bytes(value: Any) -> bytes:
    """Match nlohmann::json::dump(): compact UTF-8 with ordered object keys."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CoverageError(f"cannot read JSON {path}: {error}") from error


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def median(values: Iterable[int | float]) -> int | float | None:
    items = list(values)
    if not items:
        return None
    result = statistics.median(items)
    return int(result) if isinstance(result, float) and result.is_integer() else result


def nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def timing_samples(value: Any, expected: Any) -> bool:
    return (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and expected > 0
        and isinstance(value, list)
        and len(value) == expected
        and all(nonnegative_integer(item) for item in value)
    )


def load_catalog(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    catalog = load_json(path)
    require(isinstance(catalog, dict), "catalog must be an object")
    require(
        catalog.get("schema") == "operator-harness-batch-catalog-v2",
        "wrong catalog schema",
    )
    rows = catalog.get("operators")
    require(isinstance(rows, list) and len(rows) == 102, "catalog must contain 102 rows")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "catalog operator row must be an object")
        implementation_id = row.get("implementation_id")
        require(
            isinstance(implementation_id, str)
            and len(implementation_id) == 4
            and implementation_id.startswith("A")
            and implementation_id[1:].isdigit(),
            "catalog has an invalid implementation ID",
        )
        require(implementation_id not in by_id, f"duplicate catalog row {implementation_id}")
        by_id[implementation_id] = row
    disabled = tuple(catalog.get("policy", {}).get("disabled_ids", []))
    require(disabled == DISABLED_IDS, "catalog disabled set is not exact A106-A110")
    require(
        sum(bool(row.get("selection", {}).get("selectable")) for row in rows) == 97,
        "catalog selectable count is not 97",
    )
    unhashed = dict(catalog)
    expected_content_sha = unhashed.pop("catalog_sha256", None)
    require(
        isinstance(expected_content_sha, str)
        and canonical_sha256(unhashed) == expected_content_sha,
        "catalog canonical content SHA-256 mismatch",
    )
    return catalog, by_id, file_sha256(path)


def discover_expectations(results_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    plans: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    issues: list[str] = []

    for path in sorted(results_root.rglob("PLAN.json")):
        document = load_json(path)
        if not isinstance(document, dict) or document.get("schema") != "operator-harness-batch-plan-v2":
            continue
        cases = document.get("cases")
        if not isinstance(cases, list):
            issues.append(f"{relative(path, results_root)}: plan cases is not an array")
            continue
        sources.append({"kind": "plan", "path": relative(path, results_root), "case_count": len(cases)})
        for item in cases:
            if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
                issues.append(f"{relative(path, results_root)}: malformed plan case")
                continue
            case_id = item["case_id"]
            if case_id in plans and canonical_bytes(plans[case_id]) != canonical_bytes(item):
                issues.append(f"conflicting PLAN.json definitions for {case_id}")
            else:
                plans.setdefault(case_id, item)

    for path in sorted(results_root.rglob("SUMMARY.json")):
        document = load_json(path)
        if not isinstance(document, dict) or document.get("schema") != "operator-harness-native-batch-summary-v2":
            continue
        cases = document.get("cases")
        if not isinstance(cases, list):
            issues.append(f"{relative(path, results_root)}: summary cases is not an array")
            continue
        if document.get("case_count") != len(cases):
            issues.append(f"{relative(path, results_root)}: summary case_count mismatch")
        actual_counts = Counter(
            item.get("status") for item in cases if isinstance(item, dict)
        )
        if document.get("status_counts") != dict(actual_counts):
            issues.append(f"{relative(path, results_root)}: summary status_counts mismatch")
        sources.append({"kind": "summary", "path": relative(path, results_root), "case_count": len(cases)})
        for item in cases:
            if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
                issues.append(f"{relative(path, results_root)}: malformed summary case")
                continue
            case_id = item["case_id"]
            normalized = {
                "case_id": case_id,
                "implementation_id": item.get("implementation_id"),
                "status": item.get("status"),
            }
            if case_id in summaries and summaries[case_id] != normalized:
                issues.append(f"conflicting SUMMARY.json entries for {case_id}")
            else:
                summaries.setdefault(case_id, normalized)

    expected = plans if plans else summaries
    if plans and summaries:
        plan_ids, summary_ids = set(plans), set(summaries)
        for case_id in sorted(plan_ids - summary_ids):
            issues.append(f"planned case absent from summaries: {case_id}")
        for case_id in sorted(summary_ids - plan_ids):
            issues.append(f"summary case absent from plans: {case_id}")
    return expected, sources, issues


def verify_run_response(
    receipt: dict[str, Any], catalog_row: dict[str, Any],
    catalog_file_sha256: str, catalog_content_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    case_id = receipt.get("case_id")
    implementation_id = receipt.get("implementation_id")
    run = receipt.get("native_run")
    cold = receipt.get("native_cold_decode")

    if not isinstance(run, dict):
        return ["ok receipt lacks native_run"], {}
    if run.get("schema") != "operator-harness-response-v2" or run.get("phase") != "run":
        errors.append("native_run schema/phase mismatch")
    if run.get("status") != "ok" or run.get("request_id") != case_id:
        errors.append("native_run status/request identity mismatch")
    if run.get("implementation_ids") != [implementation_id]:
        errors.append("native_run implementation binding mismatch")
    if run.get("in_process_bitexact") is not True:
        errors.append("native_run lacks in-process bit-exact evidence")

    input_sha = receipt.get("input_sha256")
    frame_sha = receipt.get("frame_sha256")
    if not isinstance(input_sha, str) or run.get("input_sha256") != input_sha:
        errors.append("native_run input SHA-256 mismatch")
    if not isinstance(frame_sha, str) or run.get("frame_sha256") != frame_sha:
        errors.append("native_run frame SHA-256 mismatch")
    input_bytes = run.get("input_bytes")
    frame_bytes = run.get("frame_bytes")
    if not nonnegative_integer(input_bytes):
        errors.append("native_run input_bytes is invalid")
    if not nonnegative_integer(frame_bytes):
        errors.append("native_run frame_bytes is invalid")

    repeats = run.get("repeats")
    if not timing_samples(run.get("encode_ns"), repeats):
        errors.append("native_run encode_ns samples are invalid")
    if not timing_samples(run.get("decode_ns"), repeats):
        errors.append("native_run decode_ns samples are invalid")
    if not isinstance(run.get("timing_boundary"), str) or not run["timing_boundary"]:
        errors.append("native_run timing boundary is missing")

    binding = run.get("execution_binding")
    if not isinstance(binding, dict):
        errors.append("native_run execution binding is missing")
    else:
        expected_binding = {
            "implementation_ids": [implementation_id],
            "batch_id": catalog_row.get("batch_id"),
            "layer": catalog_row.get("layer"),
            "initial_kind": catalog_row.get("initial_kind"),
            "initial_width": catalog_row.get("initial_width"),
            "input_view": catalog_row.get("input_view"),
            "pipeline_stage_count": len(catalog_row.get("present_pipeline", [])),
            "catalog_sha256": catalog_file_sha256,
            "catalog_content_sha256": catalog_content_sha256,
        }
        for field, expected in expected_binding.items():
            if binding.get(field) != expected:
                errors.append(f"execution binding differs from catalog: {field}")

    fallback = run.get("fallback")
    if not isinstance(fallback, dict):
        errors.append("native_run fallback audit is missing")
        fallback = {}
    wrapped = fallback.get("wrapped_stage_count")
    raw_fallback = fallback.get("raw_fallback_stage_count")
    reasons = fallback.get("reasons")
    if not nonnegative_integer(wrapped) or not nonnegative_integer(raw_fallback):
        errors.append("native_run fallback counters are invalid")
        wrapped = raw_fallback = 0
    elif raw_fallback > wrapped:
        errors.append("raw fallback count exceeds wrapped-stage count")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        errors.append("native_run fallback reasons are invalid")
        reasons = []
    wrapper_expected = catalog_row.get("domain_execution", {}).get("measurement_wrapper") == WRAPPER_ID
    if wrapper_expected and wrapped == 0:
        errors.append("catalog wrapper case has no wrapped-stage audit")
    if not wrapper_expected and (wrapped != 0 or raw_fallback != 0):
        errors.append("non-wrapper catalog case reports wrapper execution")

    if receipt.get("cold_decode_verified") is not True or not isinstance(cold, dict):
        errors.append("fresh-process decode evidence is missing")
    else:
        if cold.get("schema") != "operator-harness-response-v2" or cold.get("phase") != "decode":
            errors.append("fresh decoder schema/phase mismatch")
        if cold.get("status") != "ok" or cold.get("request_id") != case_id:
            errors.append("fresh decoder status/request identity mismatch")
        if cold.get("implementation_ids") != [implementation_id]:
            errors.append("fresh decoder implementation binding mismatch")
        if cold.get("frame_verified_bitexact") is not True:
            errors.append("fresh decoder lacks bit-exact frame evidence")
        if cold.get("frame_sha256") != frame_sha or cold.get("frame_bytes") != frame_bytes:
            errors.append("fresh decoder frame identity mismatch")
        if cold.get("input_sha256") != input_sha or cold.get("output_sha256") != input_sha:
            errors.append("fresh decoder restored SHA-256 mismatch")
        if cold.get("output_bytes") != input_bytes:
            errors.append("fresh decoder restored byte count mismatch")
        if cold.get("warmups") != 0 or cold.get("repeats") != 1:
            errors.append("fresh decoder is not the one-repeat zero-warmup check")
        if not timing_samples(cold.get("decode_ns"), cold.get("repeats")):
            errors.append("fresh decoder timing sample is invalid")
        cold_binding = cold.get("source_schema", {}).get("execution_binding")
        if cold_binding != binding:
            errors.append("fresh decoder execution binding differs from native_run")

    metrics = {
        "input_bytes": input_bytes if nonnegative_integer(input_bytes) else None,
        "frame_bytes": frame_bytes if nonnegative_integer(frame_bytes) else None,
        "encode_ns_median": median(run.get("encode_ns", [])) if isinstance(run.get("encode_ns"), list) else None,
        "decode_ns_median": median(run.get("decode_ns", [])) if isinstance(run.get("decode_ns"), list) else None,
        "fresh_cold_decode_ns": median(cold.get("decode_ns", [])) if isinstance(cold, dict) and isinstance(cold.get("decode_ns"), list) else None,
        "whole_graph_timing_boundary": run.get("timing_boundary"),
        "fresh_cold_timing_boundary": cold.get("timing_boundary") if isinstance(cold, dict) else None,
        "wrapped_stage_count": wrapped if nonnegative_integer(wrapped) else 0,
        "raw_fallback_stage_count": raw_fallback if nonnegative_integer(raw_fallback) else 0,
        "raw_fallback_reasons": reasons,
        "recipe_sha256": binding.get("recipe_sha256") if isinstance(binding, dict) else None,
        "dispatcher": run.get("dispatcher"),
        "bit_adapter": run.get("bit_adapter"),
    }
    return errors, metrics


def artifact_check(receipt_path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("status") != "ok":
        return {"state": "not_applicable"}
    frame = receipt_path.parent / "frame.ngf"
    restored = receipt_path.parent / "restored.bin"
    if not frame.exists() and not restored.exists():
        return {
            "state": "not_mirrored",
            "frame_present": False,
            "restored_present": False,
        }
    if not frame.exists() or not restored.exists():
        return {
            "state": "incomplete_mirror",
            "frame_present": frame.exists(),
            "restored_present": restored.exists(),
        }
    frame_bytes = receipt.get("native_run", {}).get("frame_bytes")
    input_sha = receipt.get("input_sha256")
    ok = (
        frame.stat().st_size == frame_bytes
        and file_sha256(frame) == receipt.get("frame_sha256")
        and file_sha256(restored) == input_sha
    )
    return {
        "state": "verified" if ok else "mismatch",
        "frame_present": True,
        "restored_present": True,
    }


def inspect_receipt(
    path: Path,
    results_root: Path,
    catalog_rows: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    catalog_file_sha256: str,
    catalog_content_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        receipt = load_json(path)
    except CoverageError as error:
        return {
            "path": relative(path, results_root),
            "case_id": None,
            "implementation_id": None,
            "receipt_valid": False,
            "evidence_status": "invalid_receipt",
            "errors": [str(error)],
        }
    if not isinstance(receipt, dict):
        return {
            "path": relative(path, results_root),
            "case_id": None,
            "implementation_id": None,
            "receipt_valid": False,
            "evidence_status": "invalid_receipt",
            "errors": ["receipt is not an object"],
        }
    stored_sha = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    computed_sha = canonical_sha256(unhashed)
    if not isinstance(stored_sha, str) or stored_sha != computed_sha:
        errors.append("receipt_sha256 mismatch")
    if receipt.get("schema") != "operator-harness-native-case-result-v2":
        errors.append("wrong receipt schema")
    case_id = receipt.get("case_id")
    implementation_id = receipt.get("implementation_id")
    if not isinstance(case_id, str) or not case_id:
        errors.append("invalid case_id")
    if implementation_id not in catalog_rows:
        errors.append("implementation_id is absent from the 102-row catalog")
    row = catalog_rows.get(implementation_id, {})

    planned = expected.get(case_id) if isinstance(case_id, str) else None
    if planned is not None:
        if planned.get("implementation_id") != implementation_id:
            errors.append("receipt implementation differs from plan/summary")
        if "case_sha256" in receipt and "request" in planned:
            if receipt.get("case_sha256") != canonical_sha256(planned):
                errors.append("case_sha256 differs from PLAN.json case")
        if planned.get("batch_id") is not None and receipt.get("batch_id") != planned.get("batch_id"):
            errors.append("receipt batch differs from PLAN.json")
        if planned.get("sample_id") is not None and receipt.get("sample_id") != planned.get("sample_id"):
            errors.append("receipt sample differs from PLAN.json")
        if planned.get("status") is not None and receipt.get("status") != planned.get("status"):
            errors.append("receipt status differs from SUMMARY.json")
        if planned.get("sample_sha256") is not None and receipt.get("input_sha256") is not None:
            if receipt.get("input_sha256") != planned.get("sample_sha256"):
                errors.append("receipt input SHA-256 differs from PLAN.json sample")

    run_document = None
    cursor = path.parent
    while cursor == results_root or results_root in cursor.parents:
        candidate = cursor / "RUN.json"
        if candidate.is_file():
            run_document = load_json(candidate)
            break
        if cursor == results_root:
            break
        cursor = cursor.parent
    if run_document is not None:
        if not isinstance(run_document, dict) or run_document.get("schema") != "operator-harness-native-batch-run-v2":
            errors.append("nearest RUN.json has the wrong schema")
        else:
            run_identity = run_document.get("run_sha256")
            unhashed_run = dict(run_document)
            unhashed_run.pop("run_sha256", None)
            if not isinstance(run_identity, str) or canonical_sha256(unhashed_run) != run_identity:
                errors.append("nearest RUN.json identity hash mismatch")
            if receipt.get("run_sha256") != run_identity:
                errors.append("receipt run_sha256 differs from nearest RUN.json")

    status = receipt.get("status")
    if not isinstance(status, str) or not status:
        errors.append("receipt status is invalid")
        status = "invalid"
    metrics: dict[str, Any] = {}
    execution_errors: list[str] = []
    if status == "ok" and row:
        execution_errors, metrics = verify_run_response(
            receipt, row, catalog_file_sha256, catalog_content_sha256
        )
        errors.extend(execution_errors)
        if planned is not None and isinstance(planned.get("request"), dict):
            expected_recipe_sha = canonical_sha256(planned["request"].get("recipe"))
            if metrics.get("recipe_sha256") != expected_recipe_sha:
                errors.append("executed recipe SHA-256 differs from PLAN.json")

    artifacts = artifact_check(path, receipt)
    if artifacts.get("state") == "mismatch":
        errors.append("mirrored frame/restored artifacts differ from receipt")

    receipt_valid = not errors
    raw_fallback = metrics.get("raw_fallback_stage_count", 0)
    wrapped = metrics.get("wrapped_stage_count", 0)
    if status == "ok" and receipt_valid:
        if raw_fallback:
            evidence_status = "verified_wrapper_fallback"
        elif wrapped:
            evidence_status = "verified_wrapper_strict"
        else:
            evidence_status = "verified_strict"
    elif not receipt_valid:
        evidence_status = "invalid_receipt"
    elif status == "disabled":
        evidence_status = "disabled"
    else:
        evidence_status = f"execution_{status}"

    return {
        "path": relative(path, results_root),
        "case_id": case_id,
        "case_sha256": receipt.get("case_sha256"),
        "receipt_sha256": stored_sha,
        "implementation_id": implementation_id,
        "name": row.get("name"),
        "batch_id": receipt.get("batch_id") or row.get("batch_id"),
        "sample_id": receipt.get("sample_id"),
        "status": status,
        "receipt_valid": receipt_valid,
        "evidence_status": evidence_status,
        "graph_roundtrip_verified": status == "ok" and receipt_valid,
        "strict_core_verified": status == "ok" and receipt_valid and raw_fallback == 0,
        "wrapper_fallback_used": bool(status == "ok" and receipt_valid and raw_fallback),
        "fresh_cold_verified": status == "ok" and receipt_valid,
        "metrics": metrics,
        "artifact_check": artifacts,
        "reason": receipt.get("reason"),
        "error": receipt.get("error"),
        "errors": errors,
    }


def choose_receipts(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anonymous: list[dict[str, Any]] = []
    for case in cases:
        case_id = case.get("case_id")
        if isinstance(case_id, str):
            grouped[case_id].append(case)
        else:
            anonymous.append(case)
    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for case_id, values in sorted(grouped.items()):
        if len(values) == 1:
            selected.append(values[0])
            continue
        hashes = {value.get("receipt_sha256") for value in values}
        if len(hashes) == 1 and None not in hashes:
            selected.append(sorted(values, key=lambda value: value["path"])[0])
            duplicates.append({
                "case_id": case_id,
                "state": "identical_collapsed",
                "paths": sorted(value["path"] for value in values),
                "receipt_sha256": next(iter(hashes)),
            })
        else:
            duplicates.append({
                "case_id": case_id,
                "state": "conflicting_receipts_no_result_selected",
                "paths": sorted(value["path"] for value in values),
                "receipt_sha256s": sorted(str(item) for item in hashes),
            })
    selected.extend(anonymous)
    selected.sort(key=lambda value: (value.get("case_id") or "", value["path"]))
    return selected, duplicates


def pipeline_text(row: dict[str, Any]) -> str:
    return " -> ".join(step.get("action", "?") for step in row.get("present_pipeline", []))


def io_variant(row: dict[str, Any]) -> str:
    base = f"{row.get('initial_kind')}/{row.get('initial_width')} view={row.get('input_view')}"
    if row.get("implementation_id") == "A048":
        return base + "; registered frozen-packet composite path, not the v1 pure typed-plane bridge"
    binding = row.get("execution_semantics", {}).get("v1_registry_binding")
    return base + (f"; {binding}" if binding else "")


def operator_rows(
    catalog: dict[str, Any],
    expected: dict[str, dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_by_id: dict[str, set[str]] = defaultdict(set)
    duplicate_by_id: Counter[str] = Counter()
    for case in selected_cases:
        if isinstance(case.get("implementation_id"), str):
            cases_by_id[case["implementation_id"]].append(case)
    for case_id, item in expected.items():
        implementation_id = item.get("implementation_id")
        if isinstance(implementation_id, str):
            expected_by_id[implementation_id].add(case_id)
    for duplicate in duplicates:
        if duplicate.get("state") != "conflicting_receipts_no_result_selected":
            continue
        case_id = duplicate.get("case_id", "")
        if isinstance(case_id, str) and len(case_id) >= 4:
            duplicate_by_id[case_id[:4]] += 1

    result: list[dict[str, Any]] = []
    for row in catalog["operators"]:
        implementation_id = row["implementation_id"]
        cases = cases_by_id.get(implementation_id, [])
        expected_ids = expected_by_id.get(implementation_id, set())
        observed_ids = {case["case_id"] for case in cases if isinstance(case.get("case_id"), str)}
        missing = sorted(expected_ids - observed_ids)
        verified = [case for case in cases if case.get("graph_roundtrip_verified")]
        strict = [case for case in verified if case.get("strict_core_verified")]
        fallback = [case for case in verified if case.get("wrapper_fallback_used")]
        invalid = [case for case in cases if not case.get("receipt_valid")]
        failed = [case for case in cases if case.get("receipt_valid") and not case.get("graph_roundtrip_verified")]
        selectable = bool(row.get("selection", {}).get("selectable"))

        expected_count = len(expected_ids)
        complete = (
            selectable
            and expected_count > 0
            and len(verified) == expected_count
            and not missing
            and not invalid
            and not failed
            and not duplicate_by_id[implementation_id]
        )
        if not selectable:
            bounded_status = "disabled_by_policy"
        elif complete:
            bounded_status = "all_planned_cases_roundtrip_verified"
        elif verified:
            bounded_status = "partial_roundtrip_verification"
        elif invalid or duplicate_by_id[implementation_id]:
            bounded_status = "evidence_invalid"
        elif failed:
            bounded_status = "observed_no_passing_case"
        else:
            bounded_status = "not_observed"

        if not selectable:
            strict_status = "not_applicable_disabled"
        elif not verified:
            strict_status = "not_verified"
        elif fallback and strict:
            strict_status = "mixed_strict_and_wrapper_fallback_cases"
        elif fallback:
            strict_status = "wrapper_fallback_on_all_verified_cases"
        elif complete:
            strict_status = "strict_on_all_planned_cases"
        else:
            strict_status = "strict_on_verified_subset"

        metrics = [case.get("metrics", {}) for case in verified]
        reason_counts: Counter[str] = Counter()
        for item in metrics:
            for reason in item.get("raw_fallback_reasons", []):
                reason_counts[reason] += 1
        result.append({
            "implementation_id": implementation_id,
            "name": row.get("name"),
            "layer": row.get("layer"),
            "batch_id": row.get("batch_id"),
            "selectable": selectable,
            "policy_family": row.get("selection", {}).get("family"),
            "bounded_case_status": bounded_status,
            "strict_domain_status": strict_status,
            "domain_claim": "bounded_real_data_cases_only_not_full_domain",
            "expected_case_count": expected_count,
            "observed_case_count": len(cases),
            "verified_case_count": len(verified),
            "strict_case_count": len(strict),
            "wrapper_fallback_case_count": len(fallback),
            "failed_case_count": len(failed),
            "invalid_receipt_count": len(invalid),
            "conflicting_duplicate_count": duplicate_by_id[implementation_id],
            "missing_case_ids": missing,
            "failed_cases": [
                {
                    "case_id": case.get("case_id"),
                    "status": case.get("status"),
                    "reason": case.get("reason"),
                    "error": case.get("error"),
                }
                for case in failed
            ],
            "fallback": {
                "measurement_wrapper": row.get("domain_execution", {}).get("measurement_wrapper"),
                "wrapped_stage_count": sum(item.get("wrapped_stage_count", 0) for item in metrics),
                "raw_fallback_stage_count": sum(item.get("raw_fallback_stage_count", 0) for item in metrics),
                "reason_case_counts": dict(sorted(reason_counts.items())),
            },
            "whole_graph_metrics_median": {
                "input_bytes": median(item.get("input_bytes") for item in metrics if item.get("input_bytes") is not None),
                "frame_bytes": median(item.get("frame_bytes") for item in metrics if item.get("frame_bytes") is not None),
                "encode_ns": median(item.get("encode_ns_median") for item in metrics if item.get("encode_ns_median") is not None),
                "decode_ns": median(item.get("decode_ns_median") for item in metrics if item.get("decode_ns_median") is not None),
                "fresh_cold_decode_ns": median(item.get("fresh_cold_decode_ns") for item in metrics if item.get("fresh_cold_decode_ns") is not None),
            },
            "fresh_cold_verified_case_count": sum(bool(case.get("fresh_cold_verified")) for case in cases),
            "initial_kind": row.get("initial_kind"),
            "initial_width": row.get("initial_width"),
            "input_view": row.get("input_view"),
            "input_adapter_kind": row.get("input_adapter_kind"),
            "focal_step_index": row.get("focal_step_index"),
            "pipeline": pipeline_text(row),
            "io_variant": io_variant(row),
            "execution_semantics": row.get("execution_semantics"),
        })
    return result


def build_coverage(results_root: Path, catalog_path: Path) -> dict[str, Any]:
    results_root = results_root.resolve()
    catalog_path = catalog_path.resolve()
    require(results_root.is_dir(), f"results root is not a directory: {results_root}")
    require(catalog_path.is_file(), f"catalog is not a file: {catalog_path}")
    catalog, catalog_rows, catalog_file_sha = load_catalog(catalog_path)
    expected, expectation_sources, expectation_issues = discover_expectations(results_root)

    receipt_paths = sorted(results_root.rglob("RESULT.json"))
    inspected = [
        inspect_receipt(
            path, results_root, catalog_rows, expected,
            catalog_file_sha, catalog["catalog_sha256"],
        )
        for path in receipt_paths
    ]
    selected, duplicates = choose_receipts(inspected)
    operators = operator_rows(catalog, expected, selected, duplicates)

    valid = [case for case in selected if case.get("receipt_valid")]
    verified = [case for case in selected if case.get("graph_roundtrip_verified")]
    strict = [case for case in verified if case.get("strict_core_verified")]
    wrapper_fallback = [case for case in verified if case.get("wrapper_fallback_used")]
    metrics = [case.get("metrics", {}) for case in verified]
    all_planned = [row for row in operators if row["bounded_case_status"] == "all_planned_cases_roundtrip_verified"]
    partial = [
        row for row in operators
        if row["selectable"] and row["bounded_case_status"] != "all_planned_cases_roundtrip_verified"
    ]
    excluded = [row for row in operators if not row["selectable"]]
    artifact_states = Counter(case.get("artifact_check", {}).get("state", "unknown") for case in selected)
    result_status_counts = Counter(case.get("status", "invalid") for case in valid)
    evidence_status_counts = Counter(case.get("evidence_status", "invalid") for case in selected)
    receipt_issues = [
        {"path": case["path"], "case_id": case.get("case_id"), "errors": case.get("errors", [])}
        for case in selected
        if not case.get("receipt_valid")
    ]

    coverage: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_policy": {
            "catalog_planned_flags_are_execution_evidence": False,
            "pass_source": "verified RESULT.json receipt plus successful in-process and fresh-process bit-exact responses",
            "receipt_hash": "sha256 of compact UTF-8 canonical JSON with ordered keys after removing receipt_sha256",
            "retry_policy": "select one final attempt root; identical duplicate receipts collapse, conflicting duplicates produce no selected result",
            "domain_boundary": "small real-data cases validate only their exact source and IO recipes; they do not establish full-domain support",
            "timing_boundary": "whole registered graph and NGF1 frame as recorded by native_run; no per-kernel attribution",
            "timing_aggregation": "median within each native repeat vector, then median of case medians for each operator",
            "fresh_cold_boundary": "decode runs in a new worker process, while its recorded nanoseconds exclude process startup and file IO",
            "performance_claim": "descriptive shared-host measurements only; no cross-operator ranking",
        },
        "remaining_validation_scope": [
            "Only contiguous 128-value valid-value payload slices were tested; null bitmaps, timestamp associations, whole-column blocks and large inputs remain untested.",
            "Negative integers, INT64 boundaries, NaN/Inf/signed-zero patterns, malformed frames and broad parameter combinations need additional domain coverage.",
            "A003/A006/A007/A008 raw fallback cases do not extend strict-kernel domains; A048's separate pure typed-plane IO bridge was not tested by these frozen-packet recipes.",
            "Registered fixed graphs run serially on one shared-host core; arbitrary graph composition, parallel scheduling, memory reuse and isolated production performance remain future validation work.",
        ],
        "inputs": {
            "results_root": str(results_root),
            "catalog": str(catalog_path),
            "catalog_file_sha256": catalog_file_sha,
            "catalog_content_sha256": catalog.get("catalog_sha256"),
            "expectation_sources": expectation_sources,
        },
        "counts": {
            "catalog_operators": 102,
            "eligible_operators": 97,
            "disabled_operators": 5,
            "expected_case_contract": EXPECTED_CASE_CONTRACT,
            "planned_or_summarized_unique_cases": len(expected),
            "discovered_receipt_files": len(receipt_paths),
            "selected_unique_cases": len(selected),
            "valid_receipts": len(valid),
            "invalid_receipts": len(selected) - len(valid),
            "roundtrip_verified_cases": len(verified),
            "strict_cases": len(strict),
            "wrapper_fallback_cases": len(wrapper_fallback),
            "fresh_cold_verified_cases": sum(bool(case.get("fresh_cold_verified")) for case in selected),
            "operators_all_planned_cases_verified": len(all_planned),
            "operators_with_gaps": len(partial),
            "result_status_counts": dict(sorted(result_status_counts.items())),
            "evidence_status_counts": dict(sorted(evidence_status_counts.items())),
            "artifact_states": dict(sorted(artifact_states.items())),
        },
        "whole_graph_metrics_median_across_verified_cases": {
            "input_bytes": median(item.get("input_bytes") for item in metrics if item.get("input_bytes") is not None),
            "frame_bytes": median(item.get("frame_bytes") for item in metrics if item.get("frame_bytes") is not None),
            "encode_ns": median(item.get("encode_ns_median") for item in metrics if item.get("encode_ns_median") is not None),
            "decode_ns": median(item.get("decode_ns_median") for item in metrics if item.get("decode_ns_median") is not None),
            "fresh_cold_decode_ns": median(item.get("fresh_cold_decode_ns") for item in metrics if item.get("fresh_cold_decode_ns") is not None),
        },
        "operators": operators,
        "cases": selected,
        "excluded": excluded,
        "gaps": partial,
        "duplicate_receipts": duplicates,
        "expectation_issues": expectation_issues,
        "receipt_issues": receipt_issues,
        "recipe_io_note_A048": {
            "registered_v2_path": pipeline_text(catalog_rows["A048"]),
            "registered_input": io_variant(catalog_rows["A048"]),
            "claim_boundary": "This report measures the registered frozen-packet composite recipe. It is not evidence for the separate harness-v1 pure typed-plane bridge.",
        },
    }
    coverage["coverage_sha256"] = canonical_sha256(coverage)
    return coverage


def tsv_text(coverage: dict[str, Any]) -> str:
    fields = [
        "implementation_id", "name", "layer", "batch_id", "bounded_case_status",
        "strict_domain_status", "expected_case_count", "observed_case_count",
        "verified_case_count", "strict_case_count", "wrapper_fallback_case_count",
        "failed_case_count", "invalid_receipt_count", "missing_case_count",
        "fresh_cold_verified_case_count", "input_bytes_median", "frame_bytes_median",
        "encode_ns_median", "decode_ns_median", "fresh_cold_decode_ns_median",
        "initial_kind", "initial_width", "input_view", "focal_step_index",
        "io_variant", "pipeline",
    ]
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in coverage["operators"]:
        metrics = row["whole_graph_metrics_median"]
        writer.writerow({
            "implementation_id": row["implementation_id"],
            "name": row["name"],
            "layer": row["layer"],
            "batch_id": row["batch_id"] or "",
            "bounded_case_status": row["bounded_case_status"],
            "strict_domain_status": row["strict_domain_status"],
            "expected_case_count": row["expected_case_count"],
            "observed_case_count": row["observed_case_count"],
            "verified_case_count": row["verified_case_count"],
            "strict_case_count": row["strict_case_count"],
            "wrapper_fallback_case_count": row["wrapper_fallback_case_count"],
            "failed_case_count": row["failed_case_count"],
            "invalid_receipt_count": row["invalid_receipt_count"],
            "missing_case_count": len(row["missing_case_ids"]),
            "fresh_cold_verified_case_count": row["fresh_cold_verified_case_count"],
            "input_bytes_median": metrics["input_bytes"],
            "frame_bytes_median": metrics["frame_bytes"],
            "encode_ns_median": metrics["encode_ns"],
            "decode_ns_median": metrics["decode_ns"],
            "fresh_cold_decode_ns_median": metrics["fresh_cold_decode_ns"],
            "initial_kind": row["initial_kind"],
            "initial_width": row["initial_width"],
            "input_view": row["input_view"],
            "focal_step_index": row["focal_step_index"],
            "io_variant": row["io_variant"],
            "pipeline": row["pipeline"],
        })
    return buffer.getvalue()


def display_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def report_text(coverage: dict[str, Any]) -> str:
    counts = coverage["counts"]
    lines = [
        "# Operator harness v2 coverage report",
        "",
        "This report is derived only from verified `RESULT.json` receipts in the selected results root. Catalog planning flags do not count as execution evidence. Every success below is limited to the exact small real-data source and IO recipes that ran; it is not a full-domain proof.",
        "",
        "## Evidence totals",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| Catalog operators | {counts['catalog_operators']} |",
        f"| Eligible / disabled | {counts['eligible_operators']} / {counts['disabled_operators']} |",
        f"| Planned case contract | {counts['planned_or_summarized_unique_cases']} / {counts['expected_case_contract']} |",
        f"| Selected unique receipts | {counts['selected_unique_cases']} |",
        f"| Valid / invalid receipts | {counts['valid_receipts']} / {counts['invalid_receipts']} |",
        f"| Bit-exact graph + fresh-process decode | {counts['roundtrip_verified_cases']} |",
        f"| Strict / wrapper-fallback verified cases | {counts['strict_cases']} / {counts['wrapper_fallback_cases']} |",
        f"| Eligible operators passing all planned cases | {counts['operators_all_planned_cases_verified']} / 97 |",
        "",
        "Receipt hashes are recomputed from compact canonical JSON after removing `receipt_sha256`. A passing case must also contain a successful in-process bit-exact run and a one-repeat, zero-warmup fresh-process decode with matching input/frame hashes and byte counts.",
        "",
        "## Per-operator evidence",
        "",
        "`strict` counts verified cases that exercised the strict target path. `fallback` counts bit-exact wrapper raw fallbacks; those cases do not prove strict-core domain support.",
        "",
        "| ID | Name | Bounded status | Strict-domain status | Cases verified/planned | strict/fallback/failed | median input->frame B | median encode/decode/cold ns |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in coverage["operators"]:
        metrics = row["whole_graph_metrics_median"]
        size = f"{display_number(metrics['input_bytes'])}->{display_number(metrics['frame_bytes'])}"
        timing = "/".join(display_number(metrics[key]) for key in ("encode_ns", "decode_ns", "fresh_cold_decode_ns"))
        cases = f"{row['verified_case_count']}/{row['expected_case_count']}"
        split = f"{row['strict_case_count']}/{row['wrapper_fallback_case_count']}/{row['failed_case_count']}"
        name = str(row["name"]).replace("|", "\\|")
        lines.append(
            f"| {row['implementation_id']} | {name} | {row['bounded_case_status']} | "
            f"{row['strict_domain_status']} | {cases} | {split} | {size} | {timing} |"
        )

    lines.extend([
        "",
        "## Remaining gaps",
        "",
    ])
    if coverage["gaps"]:
        for row in coverage["gaps"]:
            details: list[str] = []
            if row["failed_case_count"]:
                labels = [f"{item['case_id']} ({item['status']})" for item in row["failed_cases"]]
                details.append("failed: " + ", ".join(labels))
            if row["missing_case_ids"]:
                details.append("missing: " + ", ".join(row["missing_case_ids"]))
            if row["invalid_receipt_count"]:
                details.append(f"invalid receipts: {row['invalid_receipt_count']}")
            if row["conflicting_duplicate_count"]:
                details.append(f"conflicting retries: {row['conflicting_duplicate_count']}")
            if not details:
                details.append("no verified result receipt")
            lines.append(f"- **{row['implementation_id']} {row['name']}** — {'; '.join(details)}")
    else:
        lines.append("- No missing, failed, invalid, or conflicting planned case receipts.")

    lines.extend(["", "Validation coverage still remains bounded:", ""])
    lines.extend("- " + item for item in coverage["remaining_validation_scope"])

    lines.extend([
        "",
        "## Wrapper fallback accounting",
        "",
        f"The `{WRAPPER_ID}` wrapper is measurement infrastructure and is not a 103rd operator. A003, A006, A007, and A008 are reported with strict and raw-fallback cases separated. A raw fallback can establish lossless graph roundtrip for that sample, but it does not extend the strict focal operator's admitted domain.",
        "",
    ])
    wrapper_rows = [row for row in coverage["operators"] if row["fallback"]["measurement_wrapper"]]
    for row in wrapper_rows:
        reasons = row["fallback"]["reason_case_counts"] or {"none": 0}
        reason_text = ", ".join(f"{key}={value}" for key, value in reasons.items())
        lines.append(
            f"- **{row['implementation_id']} {row['name']}**: strict cases {row['strict_case_count']}; "
            f"fallback cases {row['wrapper_fallback_case_count']}; raw fallback stages "
            f"{row['fallback']['raw_fallback_stage_count']}; reasons {reason_text}."
        )

    a048 = coverage["recipe_io_note_A048"]
    lines.extend([
        "",
        "## Recipe and IO boundary",
        "",
        "Each `COVERAGE.tsv` row records the initial kind/width/view, focal index, full registered pipeline, and IO variant. Timings are whole-graph medians and include the registered prefixes plus NGF1 framing; they are not focal-kernel timings or cross-operator rankings. Fresh-cold decode runs in a separate worker process, but its native timing excludes process startup and file IO.",
        "",
        f"A048 uses `{a048['registered_v2_path']}` with `{a048['registered_input']}`. {a048['claim_boundary']}",
        "",
        "## Explicit exclusions",
        "",
        "| ID | Name | Family | State |",
        "|---|---|---|---|",
    ])
    for row in coverage["excluded"]:
        excluded_name = str(row["name"]).replace("|", "\\|")
        lines.append(
            f"| {row['implementation_id']} | {excluded_name} | "
            f"{row['policy_family']} | disabled_by_policy |"
        )

    issues = coverage["expectation_issues"] + [
        f"{item['case_id'] or item['path']}: {', '.join(item['errors'])}"
        for item in coverage["receipt_issues"]
    ]
    issues.extend(
        f"{item['case_id']}: {item['state']}" for item in coverage["duplicate_receipts"]
        if item["state"] == "conflicting_receipts_no_result_selected"
    )
    lines.extend(["", "## Evidence integrity issues", ""])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "The reported frame/input sizes come from `native_run`; fresh-cold latency comes from the independent `native_cold_decode` response. Mirrored frame/restored artifacts are checked when present and otherwise marked `not_mirrored` in `COVERAGE.json`.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(coverage: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir = output_dir.resolve()
    coverage_path = output_dir / "COVERAGE.json"
    tsv_path = output_dir / "COVERAGE.tsv"
    report_path = output_dir / "REPORT.md"
    atomic_text(coverage_path, json.dumps(coverage, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    atomic_text(tsv_path, tsv_text(coverage))
    atomic_text(report_path, report_text(coverage))
    return {
        "coverage": str(coverage_path),
        "tsv": str(tsv_path),
        "report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path,
                        help="one deliberately selected local final-attempt result tree")
    parser.add_argument("--catalog", required=True, type=Path,
                        help="operator-harness-batch-catalog-v2 JSON")
    parser.add_argument("--output-dir", type=Path,
                        help="destination; defaults to --results-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_root
    coverage = build_coverage(args.results_root, args.catalog)
    outputs = write_outputs(coverage, output_dir)
    print(json.dumps({
        "schema": SCHEMA,
        "counts": coverage["counts"],
        "outputs": outputs,
        "coverage_sha256": coverage["coverage_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
