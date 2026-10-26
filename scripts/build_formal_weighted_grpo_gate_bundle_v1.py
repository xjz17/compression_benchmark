#!/usr/bin/env python3
"""Build the four fail-closed receipts required by formal GRPO training.

The assembler does not run mechanism probes or data preparation.  It consumes
their JSON artifacts, re-verifies every prepared byte and the frozen zlib6
table, binds the mechanism and graph evidence to the currently imported action
registry, and records exact SHA-256 identities for every input artifact and
runtime file used for admission.

The output directory is published atomically and contains four receipts plus
``gate_bundle.json``.  Nothing is emitted when any admission check fails.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Mapping, Sequence

import calibrate_weighted_grpo_references_v1 as calibration
import formal_weighted_grpo_training_v1 as training
import nncp_entropy_codec
import operator_implementation_catalog
import operator_taxonomy_v30_v7
import prediction_entropy_codec
import prepare_weighted_grpo_timeseries_v1 as preparation
import prepare_weighted_grpo_tsfile_v1 as tsfile_preparation
import primitive_operator_action_runtime
import typed_graph_runtime
import unified_action_runtime
import weight_conditioned_graph_task_v1 as graph_task
import weight_conditioned_operator_gate_v1 as operator_gate
import weighted_grpo_experiment_v1 as experiment
import validate_weighted_grpo_formal_dataset_specs_v1 as dataset_spec_gate


SCHEMA = "weighted-grpo-formal-gate-assembler-v1"
NATIVE_REFERENCE_TIMING_BOUNDARY = (
    "in-process complete ZR61 frame; excludes process startup and stdin/stdout transfer"
)
EXPECTED_PDF_SHA256 = operator_gate.EXPECTED_PDF_SHA256
EXPECTED_LAYER_COUNTS = {
    "mechanism": 67, "profile": 7, "recipe": 14, "adapter": 14,
}
SPECIAL_CATEGORIES = {"stop", "concat", "split_segments"}


class GateAssemblyError(RuntimeError):
    """An input artifact is incomplete, stale, or not formal evidence."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateAssemblyError(f"cannot read {label}: {resolved}") from error
    if not isinstance(value, dict):
        raise GateAssemblyError(f"{label} must be a JSON object")
    return value, _sha256_bytes(raw)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateAssemblyError(message)


@dataclass(frozen=True)
class RuntimeIdentity:
    source_fingerprint: str
    action_registry_fingerprint: str
    action_count: int
    stop_action: int
    max_depth: int
    max_nodes: int
    max_streams: int
    special_categories: tuple[str, ...]
    edit_types: tuple[str, ...]
    runtime_files: tuple[tuple[str, str], ...]
    measurement_protocol_schema: str | None = None
    measurement_contract: Mapping[str, Any] | None = None


def _runtime_file_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    modules = (
        Path(operator_gate.__file__), Path(graph_task.__file__),
        Path(typed_graph_runtime.__file__), Path(unified_action_runtime.__file__),
        Path(operator_implementation_catalog.__file__),
        Path(operator_taxonomy_v30_v7.__file__),
        Path(primitive_operator_action_runtime.__file__),
        Path(prediction_entropy_codec.__file__), Path(nncp_entropy_codec.__file__),
        Path(dataset_spec_gate.__file__),
    )
    candidates = {
        *(path.resolve() for path in modules),
        Path(operator_implementation_catalog.MANIFEST_PATH).resolve(),
        (root / "cpp/CMakeLists.txt").resolve(),
    }
    source_root = root / "cpp/src"
    if source_root.is_dir():
        for suffix in ("*.c", "*.cc", "*.cpp", "*.h", "*.hpp"):
            candidates.update(path.resolve() for path in source_root.glob(suffix))
    build_root = root / "cpp/build"
    if build_root.is_dir():
        candidates.update(
            path.resolve() for path in build_root.glob("*_cli")
            if path.is_file()
        )
    existing = tuple(sorted(path for path in candidates if path.is_file()))
    if not existing:
        raise GateAssemblyError("no runtime files were found for source binding")
    return existing


def discover_runtime_identity() -> RuntimeIdentity:
    """Freeze current imported registry plus exact source/binary file hashes."""
    root = Path(__file__).resolve().parents[1]
    files: list[tuple[str, str]] = []
    for path in _runtime_file_paths():
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        files.append((label, _sha256_file(path)))
    source_fingerprint = _sha256_bytes(_canonical_json({
        "schema": "weighted-grpo-runtime-source-lock-v1",
        "files": [list(item) for item in files],
    }))
    categories = {row.category for row in graph_task.ACTION_CATALOG}
    specials = tuple(sorted(categories & SPECIAL_CATEGORIES))
    return RuntimeIdentity(
        source_fingerprint=source_fingerprint,
        action_registry_fingerprint=graph_task.REGISTRY_FINGERPRINT,
        action_count=graph_task.ACTION_COUNT,
        stop_action=graph_task.STOP_ACTION,
        max_depth=graph_task.MAX_ORDINARY_DEPTH,
        max_nodes=graph_task.MAX_GRAPH_NODES,
        max_streams=graph_task.MAX_ACTIVE_STREAMS,
        special_categories=specials,
        edit_types=tuple(sorted(graph_task.EDIT_TYPE_ACTIONS)),
        runtime_files=tuple(files),
        measurement_protocol_schema=getattr(graph_task, "MEASUREMENT_PROTOCOL_SCHEMA", None),
        measurement_contract=getattr(graph_task, "FORMAL_MEASUREMENT_CONTRACT", None),
    )


def _validate_operator_gate(report: Mapping[str, Any]) -> tuple[str, ...]:
    _require(report.get("schema") == operator_gate.SCHEMA,
             "unsupported operator-gate schema")
    counts = report.get("counts")
    _require(isinstance(counts, Mapping), "operator gate lacks counts")
    _require(counts.get("trace_entries") == 102, "operator gate must cover 102 trace entries")
    _require(counts.get("mechanisms") == 67, "operator gate must cover 67 mechanisms")
    _require(counts.get("layers") == EXPECTED_LAYER_COUNTS,
             "operator layer accounting must be 67/7/14/14")
    _require(counts.get("operator_groups") == 30, "operator gate must retain 30 groups")
    _require(counts.get("special_operations") == 3,
             "operator gate must name exactly three special operations")
    pdf = report.get("pdf")
    _require(isinstance(pdf, Mapping), "operator gate lacks PDF evidence")
    _require(pdf.get("sha256") == EXPECTED_PDF_SHA256 and pdf.get("matches_expected") is True,
             "operator PDF fingerprint is absent or mismatched")
    _require(report.get("catalog_sha256") == operator_implementation_catalog.MANIFEST_SHA256,
             "operator catalog fingerprint differs from the current catalog")
    _require(report.get("catalog_registry_version") == operator_implementation_catalog.REGISTRY_VERSION,
             "operator catalog version differs from the current catalog")
    _require(report.get("structural_gate_passed") is True,
             "operator gate does not pass its structural PDF/catalog checks")
    entries = report.get("entries")
    _require(isinstance(entries, list), "operator gate entries must be a list")
    mechanisms = [row for row in entries
                  if isinstance(row, Mapping) and row.get("layer") == "mechanism"]
    ids = [str(row.get("implementation_id", "")) for row in mechanisms]
    _require(len(mechanisms) == 67 and len(set(ids)) == 67 and all(ids),
             "operator gate must contain 67 unique mechanism rows")
    _require(all(row.get("selectable") is True for row in mechanisms),
             "every retained mechanism must be selectable")
    probability_ids = set(operator_gate.PROBABILITY_MECHANISMS)
    _require(all(
        row.get("formal_evidence_ready") is True
        for row in mechanisms
        if row.get("implementation_id") in probability_ids
    ), "every probability mechanism must have a current complete entropy/cold-decode receipt")
    # The current operator-gate schema deliberately labels ordinary mechanisms
    # as unprobed.  Their formal evidence comes from the separate exhaustive
    # graph-probe artifact validated below; do not require the operator report's
    # aggregate formal_training_gate_passed flag or duplicate that evidence.
    return tuple(sorted(ids))


def _validate_graph_evidence(
    catalog: Mapping[str, Any], probes: Mapping[str, Any], runtime: RuntimeIdentity,
    mechanism_ids: Sequence[str],
) -> None:
    _require(catalog.get("schema") == graph_task.CATALOG_SCHEMA,
             "unsupported graph-catalog schema")
    _require(catalog.get("registry_fingerprint") == runtime.action_registry_fingerprint,
             "graph catalog action-registry fingerprint is stale")
    _require(probes.get("schema") == "weight-conditioned-mechanism-probes-v1",
             "unsupported mechanism-probe schema")
    _require(probes.get("registry_fingerprint") == runtime.action_registry_fingerprint,
             "mechanism probes do not match the current action registry")
    _require(catalog.get("fixed_policy_action_count") == runtime.action_count,
             "graph action count differs from the current registry")
    _require(catalog.get("stop_action") == runtime.stop_action == 0,
             "STOP must be the frozen action zero")
    _require((runtime.max_depth, runtime.max_nodes, runtime.max_streams) == (10, 64, 32),
             "graph budgets are not the required depth10/nodes64/streams32")
    _require(set(runtime.special_categories) == SPECIAL_CATEGORIES,
             "runtime special categories are not exactly STOP/Concat/continuous Split")
    _require(set(runtime.edit_types) == {"insert", "delete", "replace", "parameter_change"},
             "continuation edit types are incomplete")
    edit = catalog.get("continuation_edit_action_space")
    _require(isinstance(edit, Mapping), "graph catalog lacks continuation edit semantics")
    _require(set((edit.get("type_action_ids") or {}).keys()) == set(runtime.edit_types),
             "graph catalog edit actions differ from the runtime")
    _require(edit.get("max_completed_edits_per_block") == 2,
             "continuation must enforce at most two completed edits")
    _require(edit.get("protocol") == (
        "edit_type -> edit_position -> fixed operator/parameter action; "
        "delete completes after edit_position"
    ), "continuation edit protocol is missing or changed")
    _require(edit.get("replay_contract") == (
        "after every completed edit the entire edited recipe is replayed "
        "from the original root data flows before it becomes a legal state"
    ), "continuation edits are not proven to replay from original roots")
    _require(catalog.get("formal_training_ready") is True,
             "graph catalog does not explicitly pass formal training")
    _require(catalog.get("formal_training_blockers") == [],
             "graph catalog retains formal-training blockers")
    _validate_runtime_measurement_contract(runtime)
    try:
        experiment._validate_formal_measurement_evidence(catalog)
    except experiment.ExperimentContractError as error:
        raise GateAssemblyError(str(error)) from error
    coverage = catalog.get("mechanism_coverage")
    _require(isinstance(coverage, Mapping), "graph catalog lacks mechanism coverage")
    _require(coverage.get("mechanism_count") == 67,
             "graph coverage must contain 67 mechanisms")
    _require(coverage.get("cataloged_mechanism_count") == 67,
             "not all mechanisms have fixed graph configurations")
    _require(coverage.get("probe_passed_mechanism_count") == 67,
             "graph audit lacks 67 bit-exact cold probe passes")
    _require(coverage.get("exhaustive_bit_exact_probe_receipt") is True,
             "graph coverage is not an exhaustive bit-exact cold receipt")
    for field in (
        "omitted_selectable_mechanisms", "probe_failed_mechanisms",
        "policy_reachability_blockers", "upstream_blocked_mechanisms",
    ):
        _require(coverage.get(field) == [], f"graph coverage has blockers in {field}")
    _require(coverage.get("probe_registry_fingerprint") == runtime.action_registry_fingerprint,
             "embedded coverage probe fingerprint is stale")

    rows = probes.get("mechanisms")
    _require(isinstance(rows, list), "mechanism probes must contain rows")
    ids = [str(row.get("implementation_id", "")) for row in rows
           if isinstance(row, Mapping)]
    _require(probes.get("mechanism_count") == 67 and len(rows) == 67,
             "mechanism probe count must be 67")
    _require(set(ids) == set(mechanism_ids) and len(set(ids)) == 67,
             "mechanism probes do not cover the PDF-aligned mechanism IDs")
    _require(probes.get("passed_mechanism_count") == 67
             and probes.get("failed_mechanism_count") == 0,
             "one or more mechanisms lack a passing probe")
    for row in rows:
        _require(row.get("passed") is True, "mechanism probe row is not passed")
        actions = row.get("action_probes")
        _require(isinstance(actions, list) and any(
            isinstance(action, Mapping)
            and action.get("passed") is True
            and int(action.get("frame_bytes", 0)) > 0
            for action in actions
        ), "mechanism row lacks a complete-frame in-process and cold probe")


def _validate_runtime_measurement_contract(runtime: RuntimeIdentity) -> None:
    try:
        experiment._validate_formal_measurement_contract(
            runtime.measurement_protocol_schema, runtime.measurement_contract,
        )
    except experiment.ExperimentContractError as error:
        raise GateAssemblyError(f"current graph runtime: {error}") from error


def _manifest_path(value: Path) -> Path:
    resolved = value.resolve()
    return resolved / "dataset_manifest.json" if resolved.is_dir() else resolved


def _validate_timestamp_contract(timestamp: Mapping[str, Any], dataset_id: str) -> None:
    """Accept only audited instant or explicitly labelled coordinate semantics."""
    source_kind = timestamp.get("source_kind")
    output_unit = timestamp.get("output_unit")
    _require(output_unit in {"s", "ms", "us", "ns"},
             "timestamp output unit is unresolved")
    _require(isinstance(timestamp.get("timezone"), str) and timestamp.get("timezone"),
             "timestamp timezone is unresolved")
    if source_kind in {"calendar_date_coordinate", "civil_time_coordinate"}:
        semantic_kind = timestamp.get("semantic_kind")
        rule = tsfile_preparation.TIMESTAMP_COORDINATE_SEMANTICS.get(semantic_kind)
        _require(rule is not None, "timestamp coordinate semantic kind is not audited")
        _require(rule["source_kind"] == source_kind,
                 "timestamp coordinate source kind changed")
        _require(rule["timezone"] == timestamp.get("timezone"),
                 "timestamp coordinate timezone label changed")
        _require(timestamp.get("source_unit") == output_unit,
                 "timestamp coordinate precision changed")
        _require(timestamp.get("formats") == ["integer_coordinate_from_tsfile"],
                 "timestamp coordinate representation changed")
        _require(timestamp.get("ambiguous_fold") is None,
                 "timestamp coordinate unexpectedly resolves an ambiguous fold")
        _require(
            timestamp.get("invalid_policy") == "fail_no_unit_or_timezone_inference",
            "timestamp coordinate permits an unapproved inference or fallback",
        )
        for flag in rule["required_false_flags"]:
            _require(timestamp.get(flag) is False,
                     f"timestamp coordinate audit flag changed: {flag}")
        return

    allowed_source_kinds = {"epoch", "datetime"}
    if dataset_id == dataset_spec_gate.deep_space.DATASET_ID:
        allowed_source_kinds.add("datetime64[us]")
    _require(source_kind in allowed_source_kinds,
             "timestamp source kind is unresolved")
    if source_kind == "epoch":
        _require(timestamp.get("source_unit") in {"s", "ms", "us", "ns"}
                 and timestamp.get("timezone") == "UTC",
                 "epoch timestamp precision/timezone is unresolved")
    _require(timestamp.get("invalid_policy") == "fail_no_row_index_fallback",
             "timestamp invalid-value policy permits an unapproved fallback")


def _validate_dataset_admission(prepared: Sequence[Path]) -> dict[str, Any]:
    """Run the frozen source/spec gate before touching prepared payload bytes."""
    manifests = [_load_json(_manifest_path(path), "prepared dataset manifest")[0]
                 for path in prepared]
    try:
        admission = dataset_spec_gate.validate_bundle()
        prepared_admission = dataset_spec_gate.validate_prepared_admissions(manifests)
    except dataset_spec_gate.SpecGateError as error:
        raise GateAssemblyError(str(error)) from error
    _require(admission.get("complete_cohort_trainable") is True,
             "formal dataset admission is incomplete; blocked cohort sources cannot be bypassed")
    _require(admission.get("blocked") == 0,
             "formal dataset admission retains blocked sources")
    frozen_paths = {
        "admissibility_manifest": dataset_spec_gate.DEFAULT_AUDIT,
        "cohort_manifest": dataset_spec_gate.DEFAULT_COHORT,
        "source_revision_manifest": dataset_spec_gate.DEFAULT_REVISIONS,
    }
    audit, _audit_sha = _load_json(dataset_spec_gate.DEFAULT_AUDIT, "admissibility manifest")
    for row in audit.get("sources", []):
        if isinstance(row, Mapping) and row.get("status") == "approved" and row.get("spec_path"):
            path = (dataset_spec_gate.ROOT / str(row["spec_path"])).resolve()
            _require(path.is_file(), f"approved formal dataset spec is missing: {path}")
            frozen_paths[f"formal_spec:{row['dataset_id']}"] = path
            for field, label in (
                ("source_audit_path", "source_audit"),
                ("rebuild_receipt_path", "rebuild_receipt"),
            ):
                linked = row.get(field)
                if linked is None:
                    continue
                linked_path = (dataset_spec_gate.ROOT / str(linked)).resolve()
                _require(
                    linked_path.is_file(),
                    f"approved {label} is missing for {row['dataset_id']}: {linked_path}",
                )
                frozen_paths[f"{label}:{row['dataset_id']}"] = linked_path
    return {
        "source_spec_gate": admission,
        "prepared_admission": prepared_admission,
        "input_artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
            for name, path in sorted(frozen_paths.items())
        },
    }


def _validate_data(prepared: Sequence[Path]) -> tuple[dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    _require(bool(prepared), "at least one prepared dataset is required")
    manifests: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for value in prepared:
        manifest_path = _manifest_path(value)
        root = manifest_path.parent
        verification = preparation.verify_prepared_dataset(root, verify_source=True)
        manifest, manifest_sha = _load_json(manifest_path, "prepared dataset manifest")
        _require(verification.get("provenance_sha256") == manifest.get("provenance_sha256"),
                 "prepared provenance verification mismatch")
        timestamp = (manifest.get("input_contract") or {}).get("timestamp")
        _require(isinstance(timestamp, Mapping), "prepared timestamp contract is missing")
        _validate_timestamp_contract(timestamp, str(manifest.get("dataset_id", "")))
        selection = manifest.get("selection")
        _require(isinstance(selection, Mapping)
                 and selection.get("split") == "per_entity_chronological_80_20_before_blocks",
                 "prepared split is not per-entity chronological 80/20")
        manifests.append(manifest)
        artifacts.append({
            "path": str(manifest_path), "sha256": manifest_sha,
            "dataset_id": manifest["dataset_id"],
            "provenance_sha256": manifest["provenance_sha256"],
        })
    cohort = preparation.load_cohort_manifest()
    audit = preparation.validate_prepared_cohort_manifests(
        manifests, cohort, require_complete_train=True,
    )
    _require(audit.get("complete_train_validation") is True,
             "prepared cohort is not the complete frozen train/validation set")
    _require(audit.get("prepared_count") == 16,
             "formal data gate requires all 16 train/validation sources")
    provenance = {
        str(manifest["dataset_id"]): str(manifest["provenance_sha256"])
        for manifest in manifests
    }
    _require(len(provenance) == 16, "prepared dataset IDs are not unique")
    return provenance, audit, artifacts


def _block_bucket(byte_count: int) -> int:
    return 1 << (byte_count - 1).bit_length()


def _validate_reference(
    reference: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]],
) -> str:
    _require(reference.get("schema") == calibration.SCHEMA,
             "zlib6 reference is smoke/non-formal or has an unsupported schema")
    _require(reference.get("formal_training_eligible") is True,
             "zlib6 reference is not formal-training eligible")
    _require(reference.get("calibration_scope") == "complete_frozen_16_source_train_cohort",
             "zlib6 reference does not cover the frozen train cohort")
    _require(reference.get("reference_method") == "native zlib level 6 complete ZR61 frame",
             "reference timing is not the native complete-frame zlib6 path")
    _require(reference.get("objective_normalization") == "E/(E+E0), D/(D+D0)",
             "reference objective normalization is incompatible")
    expected_hash = reference.get("calibration_sha256")
    basis = dict(reference)
    basis.pop("calibration_sha256", None)
    _require(expected_hash == _sha256_bytes(_canonical_json(basis)),
             "zlib6 calibration fingerprint mismatch")

    expected: dict[tuple[str, str], tuple[Mapping[str, Any], Path]] = {}
    for manifest in manifests:
        root = Path(str(manifest["__root__"]))
        for block in manifest.get("blocks", []):
            if block.get("split") == "train" and int(block.get("original_numeric_bytes", 0)) > 0:
                expected[(str(manifest["dataset_id"]), str(block["block_id"]))] = (block, root)
    rows = reference.get("blocks")
    _require(isinstance(rows, list) and len(rows) == len(expected),
             "reference rows do not match all non-empty training blocks")
    _require(reference.get("prepared_dataset_count") == 16,
             "formal reference must cover all 16 prepared datasets")
    _require(reference.get("training_block_count") == len(rows),
             "reference training block count is inconsistent")
    warmups = reference.get("warmups")
    repeats = reference.get("repeats")
    _require(isinstance(warmups, int) and not isinstance(warmups, bool) and warmups >= 1,
             "reference warmup count is invalid")
    _require(isinstance(repeats, int) and not isinstance(repeats, bool)
             and repeats >= 1 and repeats % 2 == 1,
             "reference repeat count must be positive and odd")
    actual_keys: set[tuple[str, str]] = set()
    group_rows: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "reference block row must be an object")
        key = (str(row.get("dataset_id", "")), str(row.get("block_id", "")))
        _require(key in expected and key not in actual_keys,
                 "reference includes a non-training, duplicate, or unknown block")
        actual_keys.add(key)
        block, root = expected[key]
        values = (root / str(block["values_path"])).read_bytes()
        bitmap = (root / str(block["missing_bitmap_path"])).read_bytes()
        payload = values + bitmap
        _require(row.get("provenance_sha256") == next(
            manifest["provenance_sha256"] for manifest in manifests
            if manifest["dataset_id"] == key[0]
        ), "reference row provenance mismatch")
        _require(row.get("dtype") == block["dtype"], "reference dtype mismatch")
        _require(row.get("prepared_block_bytes") == len(payload),
                 "reference prepared byte count mismatch")
        _require(row.get("original_numeric_bytes") == block["original_numeric_bytes"],
                 "reference denominator mismatch")
        _require(row.get("calibration_payload_sha256") == _sha256_bytes(payload),
                 "reference payload fingerprint mismatch")
        bucket = _block_bucket(len(payload))
        _require(row.get("block_size_bucket_bytes") == bucket,
                 "reference block-size bucket mismatch")
        native = row.get("native_receipt")
        _require(isinstance(native, Mapping)
                 and native.get("schema") == calibration.BENCHMARK_SCHEMA
                 and native.get("level") == 6
                 and native.get("raw_bytes") == len(payload)
                 and native.get("warmups") == warmups
                 and native.get("repeats") == repeats
                 and int(native.get("complete_frame_bytes", 0)) > 0,
                 "reference row lacks a native complete-frame receipt")
        _require(native.get("timing_boundary") == NATIVE_REFERENCE_TIMING_BOUNDARY,
                 "reference timing boundary is missing or not native in-process complete-frame")
        for field in (
            "compression_ns_per_original_byte", "decompression_ns_per_original_byte",
        ):
            value = row.get(field)
            _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(float(value)) and float(value) > 0.0,
                     f"reference row has invalid {field}")
        denominator = int(block["original_numeric_bytes"])
        _require(math.isclose(
            float(row["compression_ns_per_original_byte"]),
            int(native.get("compression_median_ns", -1)) / denominator,
            rel_tol=1.0e-12, abs_tol=0.0,
        ), "reference compression ns/B_raw differs from its native receipt")
        _require(math.isclose(
            float(row["decompression_ns_per_original_byte"]),
            int(native.get("decompression_median_ns", -1)) / denominator,
            rel_tol=1.0e-12, abs_tol=0.0,
        ), "reference decompression ns/B_raw differs from its native receipt")
        group_rows.setdefault((str(row["dtype"]), bucket), []).append(row)
    _require(actual_keys == set(expected), "reference omits training blocks")

    refs = reference.get("references")
    _require(isinstance(refs, list) and len(refs) == len(group_rows),
             "reference group table is incomplete")
    seen_groups: set[tuple[str, int]] = set()
    for row in refs:
        _require(isinstance(row, Mapping), "reference group must be an object")
        key = (str(row.get("dtype", "")), int(row.get("block_size_bucket_bytes", 0)))
        _require(key in group_rows and key not in seen_groups,
                 "reference group is duplicate or unknown")
        seen_groups.add(key)
        _require(row.get("training_block_count") == len(group_rows[key]),
                 "reference group training block count mismatch")
        expected_block_hash = _sha256_bytes(_canonical_json(sorted(
            (str(member["dataset_id"]), str(member["block_id"]))
            for member in group_rows[key]
        )))
        _require(row.get("block_ids_sha256") == expected_block_hash,
                 "reference group block-ID fingerprint mismatch")
        expected_e0 = float(statistics.median(
            float(member["compression_ns_per_original_byte"])
            for member in group_rows[key]
        ))
        expected_d0 = float(statistics.median(
            float(member["decompression_ns_per_original_byte"])
            for member in group_rows[key]
        ))
        for field in (
            "E0_compression_ns_per_original_byte", "D0_decompression_ns_per_original_byte",
        ):
            value = row.get(field)
            _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(float(value)) and float(value) > 0.0,
                     "reference groups require finite positive E0/D0")
        _require(math.isclose(
            float(row["E0_compression_ns_per_original_byte"]), expected_e0,
            rel_tol=1.0e-12, abs_tol=0.0,
        ), "reference E0 is not the group median")
        _require(math.isclose(
            float(row["D0_decompression_ns_per_original_byte"]), expected_d0,
            rel_tol=1.0e-12, abs_tol=0.0,
        ), "reference D0 is not the group median")
    _require(seen_groups == set(group_rows), "reference groups omit a dtype/size bucket")
    return str(expected_hash)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("wb") as handle:
        handle.write(json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
        ).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def assemble(
    *, operator_gate_path: Path, graph_catalog_path: Path,
    graph_probes_path: Path, prepared: Sequence[Path], reference_path: Path,
    output_dir: Path, runtime: RuntimeIdentity | None = None,
    live_reports: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate all evidence and atomically publish a training-gate bundle."""
    runtime = runtime or discover_runtime_identity()
    operator_report, operator_sha = _load_json(operator_gate_path, "operator gate")
    graph_report, graph_sha = _load_json(graph_catalog_path, "graph catalog")
    probe_report, probe_sha = _load_json(graph_probes_path, "mechanism probes")
    reference, reference_file_sha = _load_json(reference_path, "zlib6 reference")
    mechanism_ids = _validate_operator_gate(operator_report)
    if live_reports is None:
        live_operator = operator_gate.build_report()
        live_probes = graph_task.mechanism_probe_report()
        live_graph = graph_task.catalog_report()
    else:
        live_operator, live_graph, live_probes = live_reports
    _require(_canonical_json(operator_report) == _canonical_json(live_operator),
             "operator-gate artifact is not the current live report")
    _require(_canonical_json(graph_report) == _canonical_json(live_graph),
             "graph-catalog artifact is not the current live audit")
    _require(_canonical_json(probe_report) == _canonical_json(live_probes),
             "mechanism-probe artifact is not the current live probe report")
    if live_reports is None:
        runtime_after = discover_runtime_identity()
        _require(runtime_after == runtime,
                 "runtime sources, binaries, or action registry changed during live revalidation")
    _validate_graph_evidence(
        graph_report, probe_report, runtime, mechanism_ids,
    )
    dataset_admission = _validate_dataset_admission(prepared)
    provenance, cohort_audit, data_artifacts = _validate_data(prepared)
    manifests: list[dict[str, Any]] = []
    for value in prepared:
        path = _manifest_path(value)
        manifest, _digest = _load_json(path, "prepared dataset manifest")
        manifest["__root__"] = str(path.parent.resolve())
        manifests.append(manifest)
    calibration_sha = _validate_reference(reference, manifests)

    common = {
        "schema": training.GATE_RECEIPT_SCHEMA,
        "passed": True,
        "assembler_schema": SCHEMA,
    }
    artifacts = {
        "operator_gate": {"path": str(operator_gate_path.resolve()), "sha256": operator_sha},
        "graph_catalog": {"path": str(graph_catalog_path.resolve()), "sha256": graph_sha},
        "mechanism_probes": {"path": str(graph_probes_path.resolve()), "sha256": probe_sha},
        "reference": {"path": str(reference_path.resolve()), "sha256": reference_file_sha},
    }
    receipts: dict[str, dict[str, Any]] = {
        "mechanisms": {
            **common, "gate": "mechanisms",
            "assertions": {name: True for name in training.REQUIRED_GATE_ASSERTIONS["mechanisms"]},
            "source_fingerprint": runtime.source_fingerprint,
            "action_registry_fingerprint": runtime.action_registry_fingerprint,
            "mechanism_count": 67, "mechanism_ids": list(mechanism_ids),
            "runtime_files": dict(runtime.runtime_files),
            "input_artifacts": {key: artifacts[key] for key in ("operator_gate", "mechanism_probes")},
        },
        "graph": {
            **common, "gate": "graph",
            "assertions": {name: True for name in training.REQUIRED_GATE_ASSERTIONS["graph"]},
            "source_fingerprint": runtime.source_fingerprint,
            "action_registry_fingerprint": runtime.action_registry_fingerprint,
            "budgets": {"ordinary_depth": 10, "nodes": 64, "active_data_flows": 32},
            "special_operations": ["stop", "concat", "split_segments"],
            "continuation_edit_types": list(runtime.edit_types),
            "measurement_protocol_schema": runtime.measurement_protocol_schema,
            "measurement_contract": dict(runtime.measurement_contract),
            "compression_timing_boundary": experiment.FORMAL_TIMING_BOUNDARY,
            "decompression_timing_boundary": experiment.FORMAL_TIMING_BOUNDARY,
            "input_artifacts": {key: artifacts[key] for key in ("graph_catalog", "mechanism_probes")},
        },
        "data": {
            **common, "gate": "data",
            "assertions": {name: True for name in training.REQUIRED_GATE_ASSERTIONS["data"]},
            "dataset_provenance_sha256": provenance,
            "dataset_admission": dataset_admission,
            "cohort_audit": cohort_audit,
            "prepared_manifests": data_artifacts,
        },
        "reference": {
            **common, "gate": "reference",
            "assertions": {name: True for name in training.REQUIRED_GATE_ASSERTIONS["reference"]},
            "calibration_sha256": calibration_sha,
            "input_artifact": artifacts["reference"],
            "reference_group_count": len(reference["references"]),
            "training_block_count": len(reference["blocks"]),
            "timing_boundary": NATIVE_REFERENCE_TIMING_BOUNDARY,
            "measurement_contract": dict(runtime.measurement_contract),
        },
    }

    target = output_dir.resolve()
    _require(not target.exists(), f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".", dir=target.parent))
    try:
        descriptors: dict[str, dict[str, str]] = {}
        for gate in sorted(receipts):
            path = staging / f"{gate}_receipt.json"
            _atomic_json(path, receipts[gate])
            descriptors[gate] = {"path": path.name, "sha256": _sha256_file(path)}
        bundle = {
            "schema": training.GATE_BUNDLE_SCHEMA,
            "assembler_schema": SCHEMA,
            "receipts": descriptors,
        }
        bundle_path = staging / "gate_bundle.json"
        _atomic_json(bundle_path, bundle)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "schema": SCHEMA,
        "output_dir": str(target),
        "bundle": str(target / "gate_bundle.json"),
        "bundle_sha256": _sha256_file(target / "gate_bundle.json"),
        "source_fingerprint": runtime.source_fingerprint,
        "action_registry_fingerprint": runtime.action_registry_fingerprint,
        "dataset_count": len(provenance),
        "training_block_count": len(reference["blocks"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-gate", type=Path, required=True)
    parser.add_argument("--graph-catalog", type=Path, required=True)
    parser.add_argument("--graph-probes", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = assemble(
            operator_gate_path=args.operator_gate,
            graph_catalog_path=args.graph_catalog,
            graph_probes_path=args.graph_probes,
            prepared=args.prepared,
            reference_path=args.reference,
            output_dir=args.output_dir,
        )
    except GateAssemblyError as error:
        parser.exit(2, f"formal gate assembly failed: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
