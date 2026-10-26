#!/usr/bin/env python3
"""Audit and summarize the operator-tree stress receipts.

This program deliberately treats a plan as an expectation, never as execution
evidence.  A successful case needs a hash-bound durable receipt, exact native
timing vectors, an in-process bit-exact result, and a frame-only decode in a
different process.  Expected domain rejection is reported separately from a
successful strict path or lossless wrapper path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


CATALOG_SCHEMAS = {
    "operator-harness-batch-catalog-v2",
    "operator-harness-tree-catalog-v3",
}
PLAN_SCHEMA = "operator-harness-batch-plan-v2"
RECEIPT_SCHEMA = "operator-harness-native-case-result-v2"
RUN_SCHEMA = "operator-harness-native-batch-run-v2"
RESPONSE_SCHEMA = "operator-harness-response-v2"
REPORT_SCHEMA = "operator-tree-stress-coverage-v3"
SUITE_MARKER = "operator_tree_stress_v3"
DISABLED_IDS = {"A106", "A107", "A108", "A109", "A110"}
WRAPPER_ID = "operator102.lossless_domain_fallback.selector_raw.v1"
WRAPPER_IDS = {"A003", "A006", "A007", "A008"}
WRAPPER_REASONS_BY_ID = {
    "A003": {
        "Elf32 exact action requires finite float64 values",
        "Elf32 exact action requires a bitwise lossless float32 cast",
    },
    "A006": {
        "RLE scaling requires finite float64 values",
        "RLE decimal scaling is not bitwise reversible",
    },
    "A007": {
        "TS_2DIFF scaling requires finite values",
        "TS_2DIFF scaled value exceeds int64",
        "TS_2DIFF decimal scaling is not bitwise reversible",
    },
    "A008": {
        "DBP scaling requires finite values",
        "DBP scaled value exceeds int64",
        "DBP decimal scaling exceeds int64",
        "DBP decimal scaling is not bitwise reversible",
    },
}

# Domain rejection is accepted only when native code emits a stable code from
# this list.  An expected_outcome label in PLAN.json is never sufficient.
DOMAIN_REJECTION_REASONS = {
    "A048_VALUE_EXCEEDS_BIT_WIDTH": (
        "A048 value exceeds the requested pure typed-plane bit_width"
    ),
    "CAMEL_INT64_NOT_EXACTLY_REPRESENTABLE": (
        "Camel integer is not exactly representable as float64"
    ),
    "CAMEL_NONFINITE_INPUT": "Camel decimal path requires finite input",
    "CAMEL_DECIMAL_SCALE_UNREPRESENTABLE": "Camel decimal scale did not converge",
    "FPC_EMPTY_INPUT": "FPC requires non-empty 64-bit-word bytes",
    "GORILLA_RESERVED_MARKER_INPUT": (
        "Gorilla input contains reserved end-marker bit pattern"
    ),
}
UNSUPPORTED_CODES = {
    "UNSUPPORTED_NATIVE_STAGE",
    "INPUT_WIDTH_ADAPTATION_NOT_BIT_PRESERVING",
}
CRASH_CODES = {"CHILD_SIGNAL", "NATIVE_CHILD_CRASH"}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def nonnegative_samples(value: Any, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in value)
    )


def normalized_parameter(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    raise AuditError("catalog parameter is not scalar")


def load_catalog(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str, str]:
    catalog = load_json(path)
    require(isinstance(catalog, dict), "catalog is not an object")
    require(catalog.get("schema") in CATALOG_SCHEMAS, "wrong catalog schema")
    rows = catalog.get("operators")
    require(isinstance(rows, list) and len(rows) == 102,
            "catalog must contain exactly 102 rows")
    by_id: dict[str, dict[str, Any]] = {}
    enabled: set[str] = set()
    disabled: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), "catalog row is not an object")
        implementation_id = row.get("implementation_id")
        require(isinstance(implementation_id, str) and implementation_id not in by_id,
                "catalog has an invalid or duplicate implementation ID")
        by_id[implementation_id] = row
        selectable = row.get("selection", {}).get("selectable")
        require(isinstance(selectable, bool),
                f"{implementation_id}: selectable flag is absent")
        (enabled if selectable else disabled).add(implementation_id)
    require(len(enabled) == 97 and disabled == DISABLED_IDS,
            "catalog is not exact102/exact97/exact5")
    policy_disabled = set(catalog.get("policy", {}).get("disabled_ids", []))
    require(policy_disabled == DISABLED_IDS,
            "catalog policy does not disable exactly A106-A110")
    require(catalog.get("policy", {}).get("runtime_override_allowed") is False,
            "catalog exclusion policy is overridable")
    file_digest = file_sha256(path)
    content_digest = catalog.get("catalog_sha256")
    require(isinstance(content_digest, str) and len(content_digest) == 64,
            "catalog content SHA-256 is absent")
    unhashed = dict(catalog)
    unhashed.pop("catalog_sha256", None)
    require(canonical_sha256(unhashed) == content_digest,
            "catalog canonical content SHA-256 mismatch")
    # Private audit context; add only after the canonical catalog digest check.
    for row in by_id.values():
        row["_catalog_schema"] = catalog["schema"]
    return catalog, by_id, file_digest, content_digest


def suite_marker(plan: Mapping[str, Any]) -> Any:
    for field in ("suite", "suite_marker", "campaign"):
        if field in plan:
            return plan[field]
    return None


def validate_case_shape(case: dict[str, Any], row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    implementation_id = case.get("implementation_id")
    if case.get("suite") != SUITE_MARKER:
        errors.append("case suite marker is absent")
    if case.get("case_class") not in {"large_real", "boundary"}:
        errors.append("case_class is not large_real or boundary")
    if case.get("category") != case.get("case_class"):
        errors.append("category differs from case_class")
    if case.get("case_class") == "large_real" and case.get("boundary_family") is not None:
        errors.append("large_real case has a boundary_family")
    if case.get("case_class") == "boundary" and not isinstance(case.get("boundary_family"), str):
        errors.append("boundary case lacks boundary_family")
    if case.get("expected_outcome") not in {"roundtrip", "domain_rejected"}:
        errors.append("expected_outcome is invalid")
    if case.get("expected_outcome") == "domain_rejected":
        expected_code = case.get("expected_reason_code")
        if (not isinstance(expected_code, str) or
                case.get("expected_reason") != DOMAIN_REJECTION_REASONS.get(expected_code) or
                case.get("expected_reason_category") != "STRICT_INPUT_DOMAIN_REJECTED"):
            errors.append("expected domain rejection is not source-anchored")
    elif (case.get("expected_reason_code") is not None or
          case.get("expected_reason") is not None or
          case.get("expected_reason_category") is not None):
        errors.append("roundtrip case unexpectedly declares a rejection reason")
    if case.get("planned_path") not in {"strict_required", "wrapper_allowed"}:
        errors.append("planned_path is invalid")
    if case.get("expected_execution_path") != case.get("planned_path"):
        errors.append("expected_execution_path differs from planned_path")
    sample_sha = case.get("sample_sha256")
    if (not isinstance(sample_sha, str) or len(sample_sha) != 64 or
            any(character not in "0123456789abcdef" for character in sample_sha)):
        errors.append("sample_sha256 is malformed")
    provenance = case.get("provenance")
    if (not isinstance(provenance, dict) or
            provenance.get("sample_id") != case.get("sample_id") or
            provenance.get("sample_sha256") != sample_sha or
            provenance.get("case_class") != case.get("case_class") or
            provenance.get("category") != case.get("category")):
        errors.append("case provenance does not bind the selected sample")
    wrapper = row.get("domain_execution", {}).get("measurement_wrapper")
    if case.get("planned_path") == "wrapper_allowed" and wrapper != WRAPPER_ID:
        errors.append("wrapper_allowed used by a catalog row without the wrapper")
    if implementation_id not in WRAPPER_IDS and wrapper == WRAPPER_ID:
        errors.append("unexpected wrapper declaration")
    request = case.get("request")
    if not isinstance(request, dict):
        return errors + ["request is absent"]
    if request.get("implementation_ids") != [implementation_id]:
        errors.append("request implementation binding differs from case")
    if request.get("warmups") != 1 or request.get("repeats") != 5:
        errors.append("request does not specify actual 1 warmup + 5 repeats")
    if request.get("initial_kind") != row.get("initial_kind"):
        errors.append("request initial_kind differs from catalog")
    if request.get("initial_width") != row.get("initial_width"):
        errors.append("request initial_width differs from catalog")
    if request.get("input_view") != row.get("input_view"):
        errors.append("request input_view differs from catalog")
    if (not isinstance(request.get("element_count"), int) or
            isinstance(request.get("element_count"), bool) or
            request.get("element_count", -1) < 0):
        errors.append("request element_count is not a nonnegative integer")
    if request.get("request_id") not in {None, case.get("case_id")}:
        errors.append("request_id differs from case_id")
    if request.get("original_numeric_payload") is not (
        case.get("case_class") == "large_real"
    ):
        errors.append("original_numeric_payload differs from case class")
    return errors


def discover_plans(
    root: Path, catalog_rows: dict[str, dict[str, Any]], catalog_file_sha: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], list[str]]:
    cases: dict[str, dict[str, Any]] = {}
    origins: dict[str, Path] = {}
    issues: list[str] = []
    plan_paths = sorted(root.rglob("PLAN.json"))
    require(bool(plan_paths), "no PLAN.json was found")
    for path in plan_paths:
        try:
            plan = load_json(path)
        except Exception as error:
            issues.append(f"{path}: unreadable plan: {error}")
            continue
        if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
            issues.append(f"{path}: wrong plan schema")
            continue
        if suite_marker(plan) != SUITE_MARKER:
            issues.append(f"{path}: missing suite marker {SUITE_MARKER}")
        if plan.get("catalog_file_sha256") != catalog_file_sha:
            issues.append(f"{path}: catalog file binding mismatch")
        plan_cases = plan.get("cases")
        if not isinstance(plan_cases, list) or not plan_cases:
            issues.append(f"{path}: plan has no cases")
            continue
        for case in plan_cases:
            if not isinstance(case, dict):
                issues.append(f"{path}: case is not an object")
                continue
            case_id = case.get("case_id")
            implementation_id = case.get("implementation_id")
            if not isinstance(case_id, str) or not case_id:
                issues.append(f"{path}: case_id is absent")
                continue
            if case_id in cases:
                issues.append(f"{case_id}: duplicate planned case")
                continue
            if not isinstance(implementation_id, str):
                issues.append(f"{case_id}: implementation_id is absent")
                continue
            row = catalog_rows.get(implementation_id)
            if row is None or not row.get("selection", {}).get("selectable"):
                issues.append(f"{case_id}: plan selects absent or disabled ID")
                continue
            case_errors = validate_case_shape(case, row)
            request = case.get("request")
            if (isinstance(request, dict) and
                    request.get("catalog_sha256") != catalog_file_sha):
                case_errors.append("request catalog SHA-256 differs from selected catalog")
            issues.extend(f"{case_id}: {error}" for error in case_errors)
            cases[case_id] = case
            origins[case_id] = path
    return cases, origins, issues


def stage_rows(recipe: Any) -> list[dict[str, Any]] | None:
    if not isinstance(recipe, dict) or not isinstance(recipe.get("steps"), list):
        return None
    steps = recipe["steps"]
    if not steps or steps[-1] != {"kind": "stop"}:
        return None
    result: list[dict[str, Any]] = []
    for step in steps[:-1]:
        if not isinstance(step, dict) or step.get("kind") != "stage":
            return None
        stage = step.get("stage")
        if not isinstance(stage, dict):
            return None
        result.append(stage)
    return result


def validate_recipe(request: dict[str, Any], row: dict[str, Any]) -> list[str]:
    implementation_id = row["implementation_id"]
    actual = stage_rows(request.get("recipe"))
    if actual is None:
        return ["recipe is not a terminal-stop stage tree"]
    expected = row.get("present_pipeline")
    if not isinstance(expected, list):
        return ["catalog present_pipeline is absent"]
    errors: list[str] = []
    if len(actual) != len(expected):
        return ["recipe stage count differs from catalog"]
    for index, (observed, source) in enumerate(zip(actual, expected)):
        if observed.get("action") != source.get("action"):
            errors.append(f"stage {index} action differs from catalog")
        if observed.get("profile") != source.get("profile"):
            errors.append(f"stage {index} profile differs from catalog")
        parameters = observed.get("parameters")
        if not isinstance(parameters, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in parameters.items()
        ):
            errors.append(f"stage {index} parameters are not string-valued")
            continue
        expected_parameters = {
            key: normalized_parameter(value)
            for key, value in source.get("parameters", {}).items()
        }
        if implementation_id != "A048" and parameters != expected_parameters:
            errors.append(f"stage {index} parameters differ from catalog")
    if implementation_id == "A048":
        if len(actual) != 1 or actual[0].get("action") != "harness.a048.byte_planes.v1":
            errors.append("A048 is not bound to the pure typed-plane stage")
        elif actual[0].get("profile") != "uint64":
            errors.append("A048 pure typed-plane profile is not uint64")
        else:
            parameters = actual[0].get("parameters", {})
            text = parameters.get("bit_width") if isinstance(parameters, dict) else None
            if not isinstance(text, str) or not text.isdigit() or (
                text != "0" and text.startswith("0")
            ) or int(text) > 64:
                errors.append("A048 bit_width is not canonical 0..64")
            domain = row.get("harness_parameter_domains", {}).get("bit_width")
            if not isinstance(domain, dict):
                errors.append("A048 catalog lacks bit_width parameter domain")
            else:
                minimum = domain.get("minimum", domain.get("min"))
                maximum = domain.get("maximum", domain.get("max"))
                if (minimum != 0 or maximum != 64 or
                        domain.get("action") != "harness.a048.byte_planes.v1" or
                        domain.get("profile") != "uint64" or
                        domain.get("step_index") != 0 or
                        domain.get("type") != "canonical_decimal_integer_string" or
                        domain.get("default") != "64"):
                    errors.append("A048 catalog bit_width domain is not 0..64")
    return errors


def validate_binding(
    case: dict[str, Any], row: dict[str, Any], response: dict[str, Any],
    catalog_file_sha: str, catalog_content_sha: str
) -> list[str]:
    request = case["request"]
    errors = validate_recipe(request, row)
    binding = response.get("execution_binding")
    if not isinstance(binding, dict):
        return errors + ["native run execution_binding is absent"]
    expected = {
        "schema": (
            "operator-harness-execution-binding-v3"
            if case["implementation_id"] == "A048"
            else "operator-harness-execution-binding-v2"
        ),
        "implementation_ids": [case["implementation_id"]],
        "catalog_schema": row.get("_catalog_schema"),
        "catalog_sha256": catalog_file_sha,
        "catalog_content_sha256": catalog_content_sha,
        "recipe_sha256": canonical_sha256(request.get("recipe")),
        "batch_id": row.get("batch_id"),
        "layer": row.get("layer"),
        "dispatcher": row.get("dispatcher"),
        "initial_kind": request.get("initial_kind"),
        "initial_width": request.get("initial_width"),
        "input_view": request.get("input_view"),
        "input_adapter_kind": row.get("input_adapter_kind"),
        "pipeline_stage_count": len(row.get("present_pipeline", [])),
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            errors.append(f"execution binding {field} mismatch")
    implementation_id = case["implementation_id"]
    expected_variant = "pure_typed_plane" if implementation_id == "A048" else "catalog_exact"
    if binding.get("execution_variant") != expected_variant:
        errors.append("execution binding variant mismatch")
    if implementation_id == "A048":
        stages = stage_rows(request["recipe"])
        if stages is None or len(stages) != 1:
            return errors + ["A048 recipe cannot supply the binding parameter"]
        bit_width = stages[0]["parameters"]["bit_width"]
        if binding.get("harness_parameters") != {"bit_width": bit_width}:
            errors.append("A048 binding lacks the actual bit_width")
    elif binding.get("harness_parameters") not in (None, {}):
        errors.append("non-A048 binding unexpectedly relaxes harness parameters")
    source_binding = response.get("source_schema", {}).get("execution_binding")
    if source_binding != binding:
        errors.append("run source-schema binding differs from execution_binding")
    source_schema = response.get("source_schema", {})
    source_expected = {
        "source_dtype": request.get("dtype"),
        "source_byte_width": request.get("source_byte_width"),
        "source_element_count": request.get("element_count"),
        "initial_kind": request.get("initial_kind"),
        "initial_width": request.get("initial_width"),
        "input_view": request.get("input_view"),
    }
    for field, value in source_expected.items():
        if source_schema.get(field) != value:
            errors.append(f"source-schema {field} mismatch")
    return errors


def validate_a048_proof(case: dict[str, Any], response: dict[str, Any]) -> list[str]:
    if case["implementation_id"] != "A048":
        return []
    proof = response.get("typed_plane_contract")
    if proof is None:
        proof = response.get("source_schema", {}).get("typed_plane_contract")
    if not isinstance(proof, dict):
        return ["A048 typed_plane_contract is absent"]
    stages = stage_rows(case["request"]["recipe"])
    if stages is None or len(stages) != 1:
        return ["A048 typed-plane recipe is malformed"]
    stage = stages[0]
    bit_width = int(stage["parameters"]["bit_width"])
    count_value = case["request"].get("element_count")
    if not isinstance(count_value, int) or isinstance(count_value, bool) or count_value < 0:
        return ["A048 typed-plane element count is malformed"]
    count = count_value
    planes = (bit_width + 7) // 8
    physical = max(1, planes)
    errors: list[str] = []
    expected_scalars = {
        "schema": "operator-harness-a048-typed-plane-contract-v1",
        "bit_width": bit_width,
        "bit_width_parameter": str(bit_width),
        "source_word_width_bytes": 8,
        "plane_width_bytes": 1,
        "semantic_plane_count": planes,
        "physical_output_count": physical,
        "element_count": count,
        "total_plane_payload_bytes": count * planes,
        "input_bytes": count * 8,
        "plane_payload_equals_input_bytes": count * planes == count * 8,
        "inverse_consumes_all_planes": True,
        "plane_order": "width_aligned_most_significant_byte_to_least_significant_byte",
    }
    for field, value in expected_scalars.items():
        if proof.get(field) != value:
            errors.append(f"A048 typed-plane proof {field} mismatch")
    if proof.get("zero_plane_physical_carrier") is not (planes == 0):
        errors.append("A048 zero-plane carrier proof mismatch")
    shapes = proof.get("output_shapes")
    if not isinstance(shapes, list) or len(shapes) != planes:
        errors.append("A048 output_shapes count mismatch")
    else:
        for plane, shape in enumerate(shapes):
            if (not isinstance(shape, dict) or shape.get("plane_index") != plane or
                    shape.get("kind") != "serial" or shape.get("width") != 1):
                errors.append("A048 output shape is not a serial byte plane")
                continue
            if shape.get("bytes") != count:
                errors.append("A048 output plane byte count mismatch")
    physical_shapes = proof.get("physical_output_shapes")
    if not isinstance(physical_shapes, list) or len(physical_shapes) != physical:
        errors.append("A048 physical_output_shapes count mismatch")
    elif planes == 0:
        if physical_shapes != [{"plane_index": None, "kind": "serial", "width": 1,
                                "bytes": 0, "carrier_only": True}]:
            errors.append("A048 zero-width physical carrier shape mismatch")
    elif physical_shapes != shapes:
        errors.append("A048 physical and semantic plane shapes differ")
    source_proof = response.get("source_schema", {}).get("typed_plane_contract")
    if source_proof != proof:
        errors.append("A048 response proof differs from source-schema proof")
    return errors


def validate_a048_actual_proof(
    case: dict[str, Any], response: dict[str, Any], receipt_path: Path
) -> list[str]:
    if case["implementation_id"] != "A048":
        return []
    proof = response.get("a048_typed_plane_proof")
    if not isinstance(proof, dict):
        return ["A048 actual-port typed-plane proof is absent"]
    stages = stage_rows(case["request"]["recipe"])
    if stages is None or len(stages) != 1:
        return ["A048 actual-port recipe is malformed"]
    stage = stages[0]
    bit_width = int(stage["parameters"]["bit_width"])
    count_value = case["request"].get("element_count")
    if not isinstance(count_value, int) or isinstance(count_value, bool) or count_value < 0:
        return ["A048 actual-port element count is malformed"]
    count = count_value
    planes = (bit_width + 7) // 8
    physical = max(1, planes)
    errors: list[str] = []
    expected = {
        "schema": "operator-harness-a048-typed-plane-proof-v1",
        "stage_result_source": "untimed_policy_dispatcher_forward",
        "bit_width": bit_width,
        "element_count": count,
        "semantic_plane_count": planes,
        "physical_output_count": physical,
        "total_plane_payload_bytes": count * planes,
        "zero_plane_physical_carrier_valid": planes == 0,
        "plane_order": "width_aligned_most_significant_byte_to_least_significant_byte",
        "expected_shift_bytes_match": True,
        "inverse_from_outputs_and_shape_metadata_bitexact": True,
        "metadata_schema": "operator-harness-a048-typed-plane-v2",
        "metadata_is_shape_only": True,
    }
    for field, value in expected.items():
        if proof.get(field) != value:
            errors.append(f"A048 actual-port proof {field} mismatch")
    per_plane = proof.get("per_plane")
    if not isinstance(per_plane, list) or len(per_plane) != planes:
        errors.append("A048 actual-port per_plane count mismatch")
    else:
        for plane, shape in enumerate(per_plane):
            digest = shape.get("sha256") if isinstance(shape, dict) else None
            if (not isinstance(shape, dict) or shape.get("plane_index") != plane or
                    shape.get("kind") != "serial" or shape.get("width") != 1 or
                    shape.get("bytes") != count or not isinstance(digest, str) or
                    len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                errors.append("A048 actual-port plane evidence is malformed")
    restored_path = receipt_path.parent / "restored.bin"
    if restored_path.is_file():
        raw = restored_path.read_bytes()
        if len(raw) != count * 8:
            errors.append("A048 restored input size differs from the typed-plane proof")
        else:
            values = [
                int.from_bytes(raw[offset:offset + 8], "little")
                for offset in range(0, len(raw), 8)
            ]
            padding = planes * 8 - bit_width
            expected_digests: list[str] = []
            for plane in range(planes):
                shift = (planes - 1 - plane) * 8
                payload = bytes(
                    ((((value << padding) & ((1 << 64) - 1)) >> shift) & 0xff)
                    for value in values
                )
                expected_digests.append(hashlib.sha256(payload).hexdigest())
            if isinstance(per_plane, list) and len(per_plane) == planes:
                for plane, digest in enumerate(expected_digests):
                    shape = per_plane[plane]
                    if not isinstance(shape, dict) or shape.get("sha256") != digest:
                        errors.append(
                            f"A048 actual-port plane {plane} digest differs from restored input"
                        )
    else:
        errors.append("A048 restored input is absent for independent plane verification")
    metadata_digest = proof.get("metadata_sha256")
    if (not isinstance(proof.get("metadata_bytes"), int) or proof["metadata_bytes"] <= 0 or
            not isinstance(metadata_digest, str) or len(metadata_digest) != 64 or
            any(c not in "0123456789abcdef" for c in metadata_digest)):
        errors.append("A048 actual-port shape metadata evidence is malformed")
    metadata = {
        "schema": "operator-harness-a048-typed-plane-v2",
        "count": count,
        "bit_width": bit_width,
        "semantic_plane_count": planes,
        "zero_plane_carrier": planes == 0,
    }
    metadata_bytes = canonical_bytes(metadata)
    if (proof.get("metadata_bytes") != len(metadata_bytes) or
            proof.get("metadata_sha256") != hashlib.sha256(metadata_bytes).hexdigest()):
        errors.append("A048 actual-port metadata digest differs from canonical shape metadata")
    return errors


def validate_success(
    case: dict[str, Any], row: dict[str, Any], receipt: dict[str, Any],
    receipt_path: Path, catalog_file_sha: str, catalog_content_sha: str
) -> tuple[list[str], str | None]:
    errors: list[str] = []
    run = receipt.get("native_run")
    cold = receipt.get("native_cold_decode")
    if not isinstance(run, dict):
        return ["successful receipt lacks native_run"], None
    if not isinstance(cold, dict):
        return ["successful receipt lacks native_cold_decode"], None
    if run.get("schema") != RESPONSE_SCHEMA or run.get("phase") != "run" or run.get("status") != "ok":
        errors.append("native run schema/phase/status mismatch")
    if run.get("request_id") != case["case_id"]:
        errors.append("native run request_id mismatch")
    if run.get("warmups") != 1 or run.get("repeats") != 5:
        errors.append("native run did not execute actual 1+5 timing contract")
    if not nonnegative_samples(run.get("encode_ns"), 5):
        errors.append("native run lacks five encode samples")
    if not nonnegative_samples(run.get("decode_ns"), 5):
        errors.append("native run lacks five in-process decode samples")
    if run.get("in_process_bitexact") is not True:
        errors.append("native run is not in-process bit exact")
    if run.get("input_sha256") != case.get("sample_sha256"):
        errors.append("native run input digest differs from plan")
    errors.extend(validate_binding(
        case, row, run, catalog_file_sha, catalog_content_sha
    ))
    errors.extend(validate_a048_proof(case, run))
    errors.extend(validate_a048_actual_proof(case, run, receipt_path))
    if cold.get("schema") != RESPONSE_SCHEMA or cold.get("phase") != "decode" or cold.get("status") != "ok":
        errors.append("cold decode schema/phase/status mismatch")
    if cold.get("request_id") != case["case_id"]:
        errors.append("cold decode request_id mismatch")
    if cold.get("warmups") != 0 or cold.get("repeats") != 1 or not nonnegative_samples(cold.get("decode_ns"), 1):
        errors.append("cold decode does not provide the actual 0+1 sample")
    if cold.get("frame_verified_bitexact") is not True or receipt.get("cold_decode_verified") is not True:
        errors.append("cold decode is not bit-exact verified")
    if cold.get("output_sha256") != case.get("sample_sha256"):
        errors.append("cold decode output digest differs from plan")
    if cold.get("frame_sha256") != receipt.get("frame_sha256"):
        errors.append("cold decode frame digest mismatch")
    if cold.get("source_schema", {}).get("execution_binding") != run.get("execution_binding"):
        errors.append("cold decode execution binding differs from run")
    if case["implementation_id"] == "A048":
        if cold.get("typed_plane_contract") != run.get("typed_plane_contract"):
            errors.append("cold decode typed-plane proof differs from run")
        if cold.get("source_schema", {}).get("typed_plane_contract") != run.get("typed_plane_contract"):
            errors.append("cold source-schema typed-plane proof differs from run")
    run_pid = receipt.get("native_run_pid")
    cold_pid = receipt.get("native_decode_pid")
    if not isinstance(run_pid, int) or run_pid <= 0 or not isinstance(cold_pid, int) or cold_pid <= 0 or run_pid == cold_pid:
        errors.append("run and frame-only decode do not have distinct process IDs")
    frame = receipt_path.parent / "frame.ngf"
    restored = receipt_path.parent / "restored.bin"
    if not frame.is_file() or file_sha256(frame) != receipt.get("frame_sha256"):
        errors.append("durable frame artifact is absent or changed")
    if not restored.is_file() or file_sha256(restored) != case.get("sample_sha256"):
        errors.append("durable restored artifact is absent or changed")
    fallback = run.get("fallback")
    path: str | None = None
    if not isinstance(fallback, dict):
        errors.append("fallback audit is absent")
    else:
        raw = fallback.get("raw_fallback_stage_count")
        wrapped = fallback.get("wrapped_stage_count")
        if (not isinstance(raw, int) or isinstance(raw, bool) or raw < 0 or
                not isinstance(wrapped, int) or isinstance(wrapped, bool) or wrapped < raw):
            errors.append("fallback counters are invalid")
        else:
            wrapper_case = case["implementation_id"] in WRAPPER_IDS
            expected_target = 1 if wrapper_case else 0
            reason_counts = fallback.get("raw_fallback_reason_counts")
            reasons = fallback.get("reasons")
            if (wrapped != expected_target or
                    fallback.get("target_block_count") != expected_target or
                    fallback.get("raw_fallback_block_count") != raw or
                    fallback.get("strict_target_block_count") != wrapped - raw or
                    not isinstance(reason_counts, dict) or
                    not isinstance(reasons, list) or
                    any(not isinstance(reason, str) for reason in reasons)):
                errors.append("fallback block accounting differs from the catalog contract")
            elif raw:
                path = "wrapper_verified"
                if raw != 1 or not wrapper_case:
                    errors.append("raw fallback used by a non-wrapper or multiple-stage tree")
                allowed = WRAPPER_REASONS_BY_ID.get(case["implementation_id"], set())
                if (len(reasons) != 1 or reasons[0] not in allowed or
                        reason_counts != {reasons[0]: 1}):
                    errors.append("raw fallback reason is not in the exact source allowlist")
                if case.get("planned_path") != "wrapper_allowed":
                    errors.append("raw fallback violated strict_required plan")
            else:
                path = "strict_verified"
                if reasons or reason_counts:
                    errors.append("strict path unexpectedly reports fallback reasons")
    return errors, path


def nearest_run_and_plan(receipt_path: Path, root: Path) -> tuple[Path | None, Path | None]:
    current = receipt_path.parent
    while current == root or root in current.parents:
        plan = current / "PLAN.json"
        run = current / "RUN.json"
        if plan.is_file() or run.is_file():
            return (run if run.is_file() else None, plan if plan.is_file() else None)
        if current == root:
            break
        current = current.parent
    return None, None


def classify_non_success(receipt: dict[str, Any]) -> str:
    status = receipt.get("status")
    code = receipt.get("reason_code")
    if status == "timeout":
        return "timeout"
    if status in {"gap", "unsupported"} or code in UNSUPPORTED_CODES:
        return "unsupported"
    if status == "domain_rejected" and code in DOMAIN_REJECTION_REASONS:
        return "domain_rejected"
    if status == "crash" or code in CRASH_CODES:
        return "crash"
    run_exit = receipt.get("native_run_exit_code")
    decode_exit = receipt.get("native_decode_exit_code")
    if ((isinstance(run_exit, int) and 129 <= run_exit <= 192) or
            (isinstance(decode_exit, int) and 129 <= decode_exit <= 192)):
        return "crash"
    return "failed"


def inspect_receipt(
    path: Path, root: Path, case: dict[str, Any], row: dict[str, Any],
    catalog_file_sha: str, catalog_content_sha: str
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        receipt = load_json(path)
    except Exception as error:
        return {"case_id": case["case_id"], "implementation_id": case["implementation_id"],
                "classification": "invalid_receipt", "errors": [str(error)],
                "receipt_path": str(path)}
    if not isinstance(receipt, dict):
        return {"case_id": case["case_id"], "implementation_id": case["implementation_id"],
                "classification": "invalid_receipt", "errors": ["receipt is not an object"],
                "receipt_path": str(path)}
    stored_sha = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if not isinstance(stored_sha, str) or canonical_sha256(unhashed) != stored_sha:
        errors.append("receipt_sha256 mismatch")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        errors.append("wrong receipt schema")
    if receipt.get("case_id") != case["case_id"] or receipt.get("implementation_id") != case["implementation_id"]:
        errors.append("receipt case/implementation identity mismatch")
    if receipt.get("case_sha256") != canonical_sha256(case):
        errors.append("receipt case_sha256 differs from plan")
    for field in (
        "suite", "category", "case_class", "boundary_family",
        "expected_execution_path", "expected_outcome", "expected_reason",
        "expected_reason_category", "expected_reason_code", "planned_path", "provenance",
    ):
        if receipt.get(field) != case.get(field):
            errors.append(f"receipt does not preserve plan field {field}")
    for field in ("batch_id", "sample_id"):
        if receipt.get(field) != case.get(field):
            errors.append(f"receipt does not preserve case identity field {field}")
    if receipt.get("input_sha256") != case.get("sample_sha256"):
        errors.append("receipt lacks the execution-time input SHA-256")
    if "sample_sha256" in receipt and receipt.get("sample_sha256") != case.get("sample_sha256"):
        errors.append("receipt sample_sha256 differs from plan")
    run_path, plan_path = nearest_run_and_plan(path, root)
    if plan_path is None or load_json(plan_path) is None:
        errors.append("receipt has no enclosing immutable PLAN.json")
    if run_path is None:
        errors.append("receipt has no enclosing RUN.json")
    else:
        run_record = load_json(run_path)
        run_identity = run_record.get("run_sha256") if isinstance(run_record, dict) else None
        run_unhashed = dict(run_record) if isinstance(run_record, dict) else {}
        run_unhashed.pop("run_sha256", None)
        if run_record.get("schema") != RUN_SCHEMA or canonical_sha256(run_unhashed) != run_identity:
            errors.append("RUN.json identity is invalid")
        if receipt.get("run_sha256") != run_identity:
            errors.append("receipt run_sha256 differs from RUN.json")
        if plan_path is not None and run_record.get("plan_sha256") != canonical_sha256(load_json(plan_path)):
            errors.append("RUN.json plan digest mismatch")
    reference = (run_path.parent if run_path else path.parent) / "operators" / case["implementation_id"] / (canonical_sha256(case) + ".json")
    if not reference.is_file():
        errors.append("per-operator durable receipt reference is absent")
    else:
        index = load_json(reference)
        if (not isinstance(index, dict) or index.get("case_id") != case["case_id"] or
                index.get("receipt_sha256") != stored_sha):
            errors.append("per-operator durable receipt reference mismatch")
    status = receipt.get("status")
    path_class: str | None = None
    if status == "ok":
        success_errors, path_class = validate_success(
            case, row, receipt, path, catalog_file_sha, catalog_content_sha
        )
        errors.extend(success_errors)
        classification = path_class or "failed"
    else:
        classification = classify_non_success(receipt)
        if classification == "domain_rejected":
            native = receipt.get("native_run")
            reason_code = receipt.get("reason_code")
            if (not isinstance(native, dict) or
                    native.get("schema") != RESPONSE_SCHEMA or
                    native.get("phase") != "run" or
                    native.get("request_id") != case["case_id"] or
                    native.get("status") != "domain_rejected" or
                    native.get("reason_code") != receipt.get("reason_code") or
                    not isinstance(reason_code, str) or
                    native.get("error") != DOMAIN_REJECTION_REASONS.get(reason_code) or
                    receipt.get("reason") != native.get("error") or
                    native.get("reason_category") != "STRICT_INPUT_DOMAIN_REJECTED"):
                errors.append("domain rejection lacks matching native response code")
            run_pid = receipt.get("native_run_pid")
            if (not isinstance(run_pid, int) or isinstance(run_pid, bool) or
                    run_pid <= 0 or
                    receipt.get("native_run_exit_code") != 0 or
                    receipt.get("native_run_timed_out") is not False):
                errors.append("domain rejection lacks a clean native worker exit")
    if errors:
        classification = "invalid_receipt"
    expectation_matched = (
        (case.get("expected_outcome") == "roundtrip" and classification in {"strict_verified", "wrapper_verified"})
        or (
            case.get("expected_outcome") == "domain_rejected" and
            classification == "domain_rejected" and
            receipt.get("reason_code") == case.get("expected_reason_code") and
            receipt.get("reason") == case.get("expected_reason") and
            receipt.get("native_run", {}).get("reason_category") ==
            case.get("expected_reason_category")
        )
    )
    metrics: dict[str, Any] = {}
    if classification in {"strict_verified", "wrapper_verified"}:
        native = receipt["native_run"]
        cold = receipt["native_cold_decode"]
        metrics = {
            "input_bytes": native.get("input_bytes"),
            "frame_bytes": native.get("frame_bytes"),
            "encode_ns": median(native["encode_ns"]),
            "decode_ns": median(native["decode_ns"]),
            "fresh_decode_ns": median(cold["decode_ns"]),
        }
    return {
        "case_id": case["case_id"],
        "implementation_id": case["implementation_id"],
        "case_class": case.get("case_class"),
        "boundary_family": case.get("boundary_family"),
        "expected_outcome": case.get("expected_outcome"),
        "expected_reason_code": case.get("expected_reason_code"),
        "planned_path": case.get("planned_path"),
        "status": status,
        "classification": classification,
        "expectation_matched": expectation_matched,
        "reason_code": receipt.get("reason_code"),
        "reason": receipt.get("reason", receipt.get("error")),
        "receipt_sha256": stored_sha,
        "receipt_path": str(path),
        "errors": errors,
        "metrics": metrics,
    }


def find_receipts(root: Path) -> tuple[dict[str, list[Path]], list[str]]:
    result: dict[str, list[Path]] = defaultdict(list)
    malformed: list[str] = []
    for path in sorted(root.rglob("RESULT.json")):
        try:
            document = load_json(path)
        except Exception as error:
            malformed.append(f"{path}: unreadable RESULT.json: {error}")
            continue
        if isinstance(document, dict) and isinstance(document.get("case_id"), str):
            result[document["case_id"]].append(path)
        else:
            malformed.append(f"{path}: RESULT.json lacks a string case_id")
    return result, malformed


def blank_bucket() -> dict[str, int]:
    return {key: 0 for key in (
        "strict_verified", "wrapper_verified", "domain_rejected", "unsupported",
        "crash", "timeout", "failed", "invalid_receipt", "missing"
    )}


def summarize(
    root: Path, catalog_path: Path
) -> dict[str, Any]:
    catalog, rows, catalog_file_sha, catalog_content_sha = load_catalog(catalog_path)
    plans, _, plan_issues = discover_plans(root, rows, catalog_file_sha)
    receipts, malformed_receipts = find_receipts(root)
    cases: list[dict[str, Any]] = []
    duplicate_issues: list[str] = []
    for case_id, case in sorted(plans.items()):
        paths = receipts.get(case_id, [])
        if len(paths) == 0:
            cases.append({
                "case_id": case_id,
                "implementation_id": case["implementation_id"],
                "case_class": case.get("case_class"),
                "boundary_family": case.get("boundary_family"),
                "expected_outcome": case.get("expected_outcome"),
                "planned_path": case.get("planned_path"),
                "status": None,
                "classification": "missing",
                "expectation_matched": False,
                "reason_code": None,
                "receipt_sha256": None,
                "receipt_path": None,
                "errors": ["planned case has no durable receipt"],
                "metrics": {},
            })
            continue
        if len(paths) > 1:
            duplicate_issues.append(f"{case_id}: {len(paths)} receipts in selected results root")
            cases.append({
                "case_id": case_id,
                "implementation_id": case["implementation_id"],
                "case_class": case.get("case_class"),
                "classification": "invalid_receipt",
                "expectation_matched": False,
                "errors": ["duplicate receipts; no result selected"],
                "receipt_paths": [str(path) for path in paths],
                "metrics": {},
            })
            continue
        cases.append(inspect_receipt(
            paths[0], root, case, rows[case["implementation_id"]],
            catalog_file_sha, catalog_content_sha
        ))
    unplanned = sorted(set(receipts) - set(plans))
    eligible_ids = sorted(
        implementation_id for implementation_id, row in rows.items()
        if row.get("selection", {}).get("selectable")
    )
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_id[case["implementation_id"]].append(case)
    operators: list[dict[str, Any]] = []
    for implementation_id in sorted(rows):
        row = rows[implementation_id]
        selected = by_id.get(implementation_id, [])
        per_class: dict[str, dict[str, int]] = {}
        for case_class in ("large_real", "boundary"):
            bucket = blank_bucket()
            for case in selected:
                if case.get("case_class") == case_class:
                    bucket[case["classification"]] += 1
            per_class[case_class] = bucket
        attempts = [case for case in selected if case["classification"] not in {"missing", "invalid_receipt"}]
        operators.append({
            "implementation_id": implementation_id,
            "name": row.get("name"),
            "selectable": row.get("selection", {}).get("selectable"),
            "large_real_attempted": any(case.get("case_class") == "large_real" for case in attempts),
            "boundary_attempted": any(case.get("case_class") == "boundary" for case in attempts),
            "case_count": len(selected),
            "outcomes": per_class,
            "remaining_domain_holes": sorted({
                case.get("boundary_family") for case in selected
                if case.get("case_class") == "boundary" and
                case.get("classification") in {"domain_rejected", "unsupported", "crash", "timeout", "failed", "invalid_receipt", "missing"} and
                isinstance(case.get("boundary_family"), str)
            }),
            "remaining_domain_hole_cases": [
                {
                    "case_id": case.get("case_id"),
                    "boundary_family": case.get("boundary_family"),
                    "classification": case.get("classification"),
                    "reason_code": case.get("reason_code"),
                    "reason": case.get("reason"),
                }
                for case in selected
                if case.get("case_class") == "boundary" and
                case.get("classification") in {
                    "domain_rejected", "unsupported", "crash", "timeout",
                    "failed", "invalid_receipt", "missing",
                }
            ],
        })
    counts = Counter(case["classification"] for case in cases)
    real_missing_ids = [row["implementation_id"] for row in operators if row["selectable"] and not row["large_real_attempted"]]
    boundary_missing_ids = [row["implementation_id"] for row in operators if row["selectable"] and not row["boundary_attempted"]]
    expectation_mismatches = [case["case_id"] for case in cases if not case.get("expectation_matched")]
    receipt_issues = [
        {"case_id": case["case_id"], "errors": case.get("errors", [])}
        for case in cases if case.get("errors")
    ]
    hard_failures = sum(counts[key] for key in (
        "unsupported", "crash", "timeout", "failed", "invalid_receipt", "missing"
    ))
    acceptance_issues = (
        plan_issues + malformed_receipts + duplicate_issues +
        [f"unplanned receipt: {case_id}" for case_id in unplanned] +
        [f"eligible ID lacks large_real attempt: {implementation_id}" for implementation_id in real_missing_ids] +
        [f"eligible ID lacks boundary attempt: {implementation_id}" for implementation_id in boundary_missing_ids] +
        [f"expectation mismatch: {case_id}" for case_id in expectation_mismatches]
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "terminology": {"user_facing": "operator tree / 算子树", "internal_api_legacy": "graph"},
        "status": "pass" if not acceptance_issues and hard_failures == 0 else "fail",
        "catalog": {
            "path": str(catalog_path),
            "schema": catalog.get("schema"),
            "file_sha256": catalog_file_sha,
            "content_sha256": catalog_content_sha,
            "registered": 102,
            "enabled": 97,
            "disabled": 5,
            "disabled_ids": sorted(DISABLED_IDS),
        },
        "evidence_contract": {
            "plan_is_execution_evidence": False,
            "success": "durable hash-bound receipt + five encode + five in-process decode + one distinct-process frame-only decode",
            "domain_rejected_is_roundtrip_pass": False,
            "shared_host_timings_are_rankings": False,
        },
        "counts": {
            "planned_cases": len(plans),
            "planned_large_real_cases": sum(
                case.get("case_class") == "large_real" for case in plans.values()
            ),
            "planned_boundary_cases": sum(
                case.get("case_class") == "boundary" for case in plans.values()
            ),
            "receipt_files": sum(len(value) for value in receipts.values()),
            "selected_cases": len(cases),
            "strict_verified": counts["strict_verified"],
            "wrapper_verified": counts["wrapper_verified"],
            "domain_rejected": counts["domain_rejected"],
            "unsupported": counts["unsupported"],
            "crash": counts["crash"],
            "timeout": counts["timeout"],
            "failed": counts["failed"],
            "invalid_receipt": counts["invalid_receipt"],
            "missing": counts["missing"],
            "eligible_with_large_real_attempt": 97 - len(real_missing_ids),
            "eligible_with_boundary_attempt": 97 - len(boundary_missing_ids),
            "expectation_mismatches": len(expectation_mismatches),
            "roundtrip_verified": counts["strict_verified"] + counts["wrapper_verified"],
        },
        "roundtrip_complete": (
            counts["domain_rejected"] == 0 and hard_failures == 0 and
            counts["strict_verified"] + counts["wrapper_verified"] == len(cases)
        ),
        "acceptance_issues": acceptance_issues,
        "plan_issues": plan_issues,
        "malformed_receipts": malformed_receipts,
        "receipt_issues": receipt_issues,
        "unplanned_receipt_case_ids": unplanned,
        "operators": operators,
        "cases": cases,
    }
    unhashed = dict(report)
    report["coverage_sha256"] = canonical_sha256(unhashed)
    return report


def tsv_text(report: dict[str, Any]) -> str:
    columns = [
        "implementation_id", "name", "selectable", "large_real_attempted",
        "boundary_attempted", "real_strict", "real_wrapper", "real_domain_rejected",
        "real_failed", "boundary_strict", "boundary_wrapper",
        "boundary_domain_rejected", "boundary_failed", "remaining_domain_holes",
    ]
    rows: list[dict[str, Any]] = []
    for row in report["operators"]:
        real = row["outcomes"]["large_real"]
        boundary = row["outcomes"]["boundary"]
        failure_keys = ("unsupported", "crash", "timeout", "failed", "invalid_receipt", "missing")
        rows.append({
            "implementation_id": row["implementation_id"],
            "name": row["name"],
            "selectable": str(row["selectable"]).lower(),
            "large_real_attempted": str(row["large_real_attempted"]).lower(),
            "boundary_attempted": str(row["boundary_attempted"]).lower(),
            "real_strict": real["strict_verified"],
            "real_wrapper": real["wrapper_verified"],
            "real_domain_rejected": real["domain_rejected"],
            "real_failed": sum(real[key] for key in failure_keys),
            "boundary_strict": boundary["strict_verified"],
            "boundary_wrapper": boundary["wrapper_verified"],
            "boundary_domain_rejected": boundary["domain_rejected"],
            "boundary_failed": sum(boundary[key] for key in failure_keys),
            "remaining_domain_holes": ",".join(row["remaining_domain_holes"]),
        })
    from io import StringIO
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def cases_tsv_text(report: dict[str, Any]) -> str:
    columns = [
        "case_id", "implementation_id", "case_class", "boundary_family",
        "planned_path", "expected_outcome", "expected_reason_code",
        "classification", "expectation_matched", "reason_code",
        "input_bytes", "frame_bytes", "encode_ns_median", "decode_ns_median",
        "fresh_decode_ns", "receipt_sha256", "receipt_path",
    ]
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        metrics = case.get("metrics", {})
        rows.append({
            "case_id": case.get("case_id"),
            "implementation_id": case.get("implementation_id"),
            "case_class": case.get("case_class"),
            "boundary_family": case.get("boundary_family"),
            "planned_path": case.get("planned_path"),
            "expected_outcome": case.get("expected_outcome"),
            "expected_reason_code": case.get("expected_reason_code"),
            "classification": case.get("classification"),
            "expectation_matched": str(bool(case.get("expectation_matched"))).lower(),
            "reason_code": case.get("reason_code"),
            "input_bytes": metrics.get("input_bytes"),
            "frame_bytes": metrics.get("frame_bytes"),
            "encode_ns_median": metrics.get("encode_ns"),
            "decode_ns_median": metrics.get("decode_ns"),
            "fresh_decode_ns": metrics.get("fresh_decode_ns"),
            "receipt_sha256": case.get("receipt_sha256"),
            "receipt_path": case.get("receipt_path"),
        })
    from io import StringIO
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def markdown_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Operator tree stress validation",
        "",
        "This report is derived from durable C++ runner receipts. The plan defines expected coverage but is not counted as execution evidence. Domain rejection is recorded as a remaining domain boundary, not as a successful roundtrip.",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- Registered/enabled/disabled: 102/97/5; disabled A106-A110.",
        f"- Strict verified: {counts['strict_verified']}; lossless wrapper verified: {counts['wrapper_verified']}.",
        f"- Domain rejected: {counts['domain_rejected']}; unsupported/crash/timeout/failed: {counts['unsupported']}/{counts['crash']}/{counts['timeout']}/{counts['failed']}.",
        f"- Durable receipt invalid/missing: {counts['invalid_receipt']}/{counts['missing']}.",
        f"- Eligible IDs with large-real/boundary attempts: {counts['eligible_with_large_real_attempt']}/97 and {counts['eligible_with_boundary_attempt']}/97.",
        f"- Roundtrip complete across selected cases: {'yes' if report['roundtrip_complete'] else 'no'}.",
        "- Per-case median encode, in-process decode and fresh-process decode timings are in CASES.tsv; shared-host values are measurements, not rankings.",
        "",
        "| ID | Larger real strict/wrapper/rejected/failed | Boundary strict/wrapper/rejected/failed | Remaining domain holes |",
        "|---|---:|---:|---|",
    ]
    failure_keys = ("unsupported", "crash", "timeout", "failed", "invalid_receipt", "missing")
    for row in report["operators"]:
        if not row["selectable"]:
            continue
        real = row["outcomes"]["large_real"]
        boundary = row["outcomes"]["boundary"]
        real_failed = sum(real[key] for key in failure_keys)
        boundary_failed = sum(boundary[key] for key in failure_keys)
        holes = ", ".join(row["remaining_domain_holes"]) or "-"
        lines.append(
            f"| {row['implementation_id']} | {real['strict_verified']}/{real['wrapper_verified']}/{real['domain_rejected']}/{real_failed} | "
            f"{boundary['strict_verified']}/{boundary['wrapper_verified']}/{boundary['domain_rejected']}/{boundary_failed} | {holes} |"
        )
    if report["acceptance_issues"]:
        lines.extend(["", "## Acceptance issues", ""])
        lines.extend(f"- {issue}" for issue in report["acceptance_issues"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = summarize(arguments.results_root.resolve(), arguments.catalog.resolve())
    output = arguments.output_dir.resolve()
    atomic_text(output / "COVERAGE.json", json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n")
    atomic_text(output / "COVERAGE.tsv", tsv_text(report))
    atomic_text(output / "CASES.tsv", cases_tsv_text(report))
    atomic_text(output / "REPORT.md", markdown_text(report))
    print(json.dumps({
        "schema": REPORT_SCHEMA,
        "status": report["status"],
        "counts": report["counts"],
        "coverage_sha256": report["coverage_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
