#!/usr/bin/env python3
"""Build and validate the project179 numeric/UTC derived-view contract.

The contract is deliberately separate from the frozen dataset-scope file.  It
binds the scope, both conversion policies, both producers, the independent
Core19 cold auditor, and the two local runtime helper modules imported by the
THULab producer by SHA-256.  A contract may be created before materialization;
in that case its evidence state is explicitly ``not_attached`` and it makes no
completion claim.  Core19 evidence is accepted only when its top-level
manifest and independent cold-audit PASS receipt are supplied together.
Supplying verified Core19 and/or THULab evidence upgrades only the
corresponding evidence state.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPE = ROOT / "benchmark/reference/datasets/project_dataset_scope_20260928.json"
DEFAULT_CORE_POLICY = ROOT / "benchmark/reference/datasets/project179_core19_utc_policy_20260928.json"
DEFAULT_THULAB_POLICY = ROOT / "benchmark/reference/datasets/project179_thulab_utc_policy_20260928.json"
DEFAULT_CORE_PRODUCER = ROOT / "scripts/prepare_project179_core19_numeric_v1.py"
DEFAULT_CORE_AUDITOR = ROOT / "scripts/audit_project179_core19_numeric_v1.py"
DEFAULT_THULAB_PRODUCER = ROOT / "scripts/prepare_project179_thulab_numeric_v1.py"
DEFAULT_THULAB_HELPER_TIMESERIES = ROOT / "scripts/prepare_weighted_grpo_timeseries_v1.py"
DEFAULT_THULAB_HELPER_TSFILE = ROOT / "scripts/prepare_weighted_grpo_tsfile_v1.py"
DEFAULT_OUTPUT = ROOT / "benchmark/reference/datasets/project179_numeric_utc_contract_20260928.json"
ONE_GIB = 1 << 30
EXPECTED_CORE_AUDITOR_SHA256 = "be86e3421f43659feabcd31e391282f50091adfa56b4b9cc19a9cf3b4cf6cc74"
EXPECTED_CORE_ZSTD_PATH = "/usr/bin/zstd"
EXPECTED_CORE_ZSTD_SHA256 = "d304445daa7e6429293dc02035063b7993fb6a489ee90d8851bff497952836dc"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CORE_DTYPES = frozenset({"int64", "float64"})
THULAB_DTYPES = frozenset({"int32", "int64", "float32", "float64"})
SERVER192_STATES = frozenset({
    "not_checked_by_this_run",
    "authentication_blocked",
    "not_propagated",
})
THULAB_POLICY_DECISIONS = frozenset({
    "admit_exact_utc",
    "admit_stored_epoch_coordinate",
    "admit_dataset_adapter",
    "block",
})
CORE_AUDIT_SCHEMA = "project179-core19-numeric-utc-cold-audit-v1"
CORE_AUDIT_TRUE_CHECKS = (
    "exact_ordered_core19",
    "pinned_scope_policy_producer_and_zstd",
    "poisson_timestamps_rejected",
    "logical_payload_integer_float_or_explicit_timestamp_only",
    "timestamp_signed_int64_unix_microseconds",
    "longest_whole_column_prefix_per_dataset",
    "all_source_output_and_chunk_hashes_streamed",
    "ordered_chunks_reconstruct_raw",
    "strict_output_eligible_file_set",
    "output_payload_bindings_stable_through_audit_end",
    "metadata_single_fd_hash_parse_and_final_recheck",
    "backup_single_fd_raw_hash_csv_parse_and_final_recheck",
    "final_output_tree_rescan_after_all_stable_bindings",
)
CORE_AUDIT_RELEASE_CHECK_KEYS = frozenset(CORE_AUDIT_TRUE_CHECKS) | {
    "timestamp_blocked", "limit_bytes_per_dataset",
}


class ContractError(ValueError):
    """Raised when an input cannot support the requested contract claim."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"binding file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_scope(scope: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    if scope.get("schema") != "web-compression-project-dataset-scope-v1":
        raise ContractError("unsupported project scope schema")
    core = scope.get("core_csv_datasets")
    timestamp = scope.get("timestamp_datasets")
    if not isinstance(core, list) or not isinstance(timestamp, list):
        raise ContractError("scope dataset lists are missing")
    core_ids = [str(value) for value in core]
    timestamp_rows = [dict(value) for value in timestamp if isinstance(value, Mapping)]
    if len(timestamp_rows) != len(timestamp):
        raise ContractError("timestamp scope contains a non-object row")
    timestamp_ids = [str(row.get("id", "")) for row in timestamp_rows]
    if len(core_ids) != 19 or len(timestamp_ids) != 160:
        raise ContractError("scope must contain exactly 19 Core and 160 THULab datasets")
    if len(set(core_ids)) != 19 or len(set(timestamp_ids)) != 160:
        raise ContractError("scope contains duplicate dataset identities")
    if set(core_ids) & set(timestamp_ids):
        raise ContractError("Core and THULab identities overlap")
    if int(scope.get("total_dataset_count", -1)) != 179:
        raise ContractError("scope total_dataset_count must be 179")
    if "poisson_timestamps" in {*core_ids, *timestamp_ids}:
        raise ContractError("retired poisson_timestamps remains in active scope")
    for row, dataset_id in zip(timestamp_rows, timestamp_ids, strict=True):
        if row.get("repo_id") != f"THULab/{dataset_id}":
            raise ContractError(f"repository binding mismatch for {dataset_id}")
        if not REVISION_RE.fullmatch(str(row.get("revision", ""))):
            raise ContractError(f"unfrozen revision for {dataset_id}")
    return core_ids, timestamp_rows


def _validate_core_policy(
    policy: Mapping[str, Any], *, scope_id: str, core_ids: Sequence[str]
) -> dict[str, int]:
    if policy.get("schema") != "project179-core19-utc-policy-v2":
        raise ContractError("unsupported Core19 UTC policy schema")
    if policy.get("scope_id") != scope_id:
        raise ContractError("Core19 policy is bound to a different scope id")
    target = policy.get("target_timestamp", {})
    if target.get("dtype") != "int64" or target.get("unit") != "unix_epoch_microseconds":
        raise ContractError("Core19 timestamp target must be int64 Unix microseconds")
    datasets = policy.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(core_ids):
        raise ContractError("Core19 policy identities do not equal the Core19 scope")
    entries: list[Mapping[str, Any]] = []
    for dataset_id in core_ids:
        rows = datasets[dataset_id].get("timestamps")
        if not isinstance(rows, list):
            raise ContractError(f"Core19 timestamp list is missing: {dataset_id}")
        if sum(isinstance(row, Mapping) for row in rows) != len(rows):
            raise ContractError(f"Core19 timestamp policy has a non-object row: {dataset_id}")
        for row in rows:
            assert isinstance(row, Mapping)
            if row.get("source") == "backup_string":
                has_format = "format" in row
                has_formats = "formats" in row
                if has_format == has_formats:
                    raise ContractError(
                        f"Core19 timestamp must define exactly one format form: {dataset_id}"
                    )
                if has_format:
                    value = row.get("format")
                    if not isinstance(value, str) or not value:
                        raise ContractError(
                            f"Core19 timestamp format is invalid: {dataset_id}"
                        )
                else:
                    values = row.get("formats")
                    if (
                        not isinstance(values, list)
                        or not values
                        or any(not isinstance(value, str) or not value for value in values)
                        or len(values) != len(set(values))
                    ):
                        raise ContractError(
                            f"Core19 timestamp formats are invalid: {dataset_id}"
                        )
            entries.append(row)
    derived = {
        "timestamp_output_definition_count": len(entries),
        "converted_definition_count": sum(row.get("action") != "drop" for row in entries),
        "exact_event_utc_instant_definition_count": sum(
            row.get("action") != "drop" and row.get("event_utc_instant_claim") is True
            for row in entries
        ),
        "coordinate_only_definition_count": sum(
            row.get("action") != "drop" and row.get("event_utc_instant_claim") is False
            for row in entries
        ),
        "dropped_time_only_definition_count": sum(row.get("action") == "drop" for row in entries),
    }
    counts = policy.get("policy_counts", {})
    expected = {
        "timestamp_entries": derived["timestamp_output_definition_count"],
        "convertible_if_sources_validate": derived["converted_definition_count"],
        "exact_utc_instant_claim_true": derived["exact_event_utc_instant_definition_count"],
        "coordinate_only_instant_claim_false": derived["coordinate_only_definition_count"],
        "time_only_dropped": derived["dropped_time_only_definition_count"],
    }
    if any(int(counts.get(key, -1)) != value for key, value in expected.items()):
        raise ContractError("Core19 policy_counts do not match its timestamp entries")
    return derived


def _validate_thulab_policy(
    policy: Mapping[str, Any], *, scope_id: str, scope_sha256: str,
    timestamp_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if policy.get("schema") != "project179-thulab-utc-policy-v1":
        raise ContractError("unsupported THULab UTC policy schema")
    if policy.get("scope_id") != scope_id or policy.get("scope_sha256") != scope_sha256:
        raise ContractError("THULab policy scope binding mismatch")
    output = policy.get("output_contract", {})
    if output.get("integer_type") != "INT64" or output.get("unit") != "Unix epoch microseconds":
        raise ContractError("THULab timestamp target must be INT64 Unix microseconds")
    raw_datasets = policy.get("datasets")
    if not isinstance(raw_datasets, list) or len(raw_datasets) != 160:
        raise ContractError("THULab policy must contain exactly 160 dataset rows")
    datasets = [row for row in raw_datasets if isinstance(row, Mapping)]
    if len(datasets) != len(raw_datasets):
        raise ContractError("THULab policy contains a non-object row")
    expected = {
        (str(row["id"]), str(row["repo_id"]), str(row["revision"]))
        for row in timestamp_rows
    }
    observed = {
        (str(row.get("dataset_id", "")), str(row.get("repo_id", "")), str(row.get("revision", "")))
        for row in datasets
    }
    if observed != expected or len(observed) != 160:
        raise ContractError("THULab policy identities/revisions do not equal the frozen scope")
    observed_decisions = [row.get("decision") for row in datasets]
    unsupported_decisions = sorted({
        repr(decision)
        for decision in observed_decisions
        if decision not in THULAB_POLICY_DECISIONS
    })
    if unsupported_decisions:
        raise ContractError(
            "THULab policy contains an unsupported decision: "
            + ", ".join(unsupported_decisions)
        )
    decisions = {
        "primary_exact_event_utc_dataset_count": sum(
            row.get("decision") == "admit_exact_utc" for row in datasets
        ),
        "primary_stored_coordinate_dataset_count": sum(
            row.get("decision") == "admit_stored_epoch_coordinate" for row in datasets
        ),
        "primary_dataset_adapter_count": sum(
            row.get("decision") == "admit_dataset_adapter" for row in datasets
        ),
        "primary_blocked_dataset_count": sum(row.get("decision") == "block" for row in datasets),
    }
    if sum(decisions.values()) != 160:
        raise ContractError("THULab policy decision categories must sum to exactly 160")
    declared = policy.get("decision_counts", {})
    declared_expected = {
        "admit_exact_utc": decisions["primary_exact_event_utc_dataset_count"],
        "admit_stored_epoch_coordinate": decisions["primary_stored_coordinate_dataset_count"],
        "admit_dataset_adapter": decisions["primary_dataset_adapter_count"],
        "block": decisions["primary_blocked_dataset_count"],
    }
    if (
        not isinstance(declared, Mapping)
        or set(declared) != THULAB_POLICY_DECISIONS
        or any(int(declared.get(key, -1)) != value for key, value in declared_expected.items())
        or sum(int(declared[key]) for key in THULAB_POLICY_DECISIONS) != 160
    ):
        raise ContractError("THULab decision_counts do not match its dataset rows")
    if decisions["primary_dataset_adapter_count"] != 3:
        raise ContractError("THULab policy must contain exactly three explicit mixed-dataset adapters")
    if decisions["primary_blocked_dataset_count"] != 0:
        raise ContractError("THULab policy still contains blocked datasets")
    for row in datasets:
        dataset_id = str(row.get("dataset_id", ""))
        decision = row.get("decision")
        temporal_fields = row.get("temporal_string_fields", [])
        if not isinstance(temporal_fields, list):
            raise ContractError(f"THULab temporal field policy is not a list: {dataset_id}")
        temporal_by_name: dict[str, Mapping[str, Any]] = {}
        for field in temporal_fields:
            if not isinstance(field, Mapping):
                raise ContractError(f"THULab temporal field policy is invalid: {dataset_id}")
            source_name = str(field.get("source_name", ""))
            if (
                not source_name or source_name in temporal_by_name
                or field.get("output_type") != "int64"
                or field.get("output_unit") != "us"
                or field.get("output_name") != f"{source_name}__timestamp_unix_us"
                or not isinstance(field.get("event_utc_instant_claim"), bool)
                or not isinstance(field.get("semantic_kind"), str)
                or not isinstance(field.get("timezone_basis"), str)
            ):
                raise ContractError(f"THULab temporal output contract is invalid: {dataset_id}")
            temporal_by_name[source_name] = field
        if decision in {"admit_exact_utc", "admit_stored_epoch_coordinate"}:
            timestamp = row.get("timestamp")
            if not isinstance(timestamp, Mapping):
                raise ContractError(f"THULab primary timestamp policy is missing: {dataset_id}")
            source_column = str(timestamp.get("source_column", ""))
            if (
                timestamp.get("output_unit") != "us"
                or timestamp.get("output_column") != "timestamp_unix_us"
                or not source_column
            ):
                raise ContractError(f"THULab primary timestamp output is not int64-us: {dataset_id}")
            if source_column != "Time" and source_column not in temporal_by_name:
                raise ContractError(f"THULab primary temporal source is unbound: {dataset_id}")
            if decision == "admit_exact_utc":
                if (
                    timestamp.get("semantic_kind") != "unix_epoch_instant"
                    or timestamp.get("timezone") != "UTC"
                    or timestamp.get("utc_instant_verified") is not True
                    or timestamp.get("timezone_inferred") is not False
                ):
                    raise ContractError(f"THULab exact UTC semantics are invalid: {dataset_id}")
                if source_column == "Time":
                    if (
                        timestamp.get("source_unit") not in {"s", "ms", "us"}
                        or timestamp.get("source_unit_verified") is not True
                    ):
                        raise ContractError(f"THULab exact Time unit is invalid: {dataset_id}")
                elif timestamp.get("source_format_verified") is not True:
                    raise ContractError(f"THULab exact string format is unverified: {dataset_id}")
            elif (
                source_column != "Time"
                or timestamp.get("source_unit") not in {"s", "ms", "us"}
                or timestamp.get("source_unit_verified") is not True
                or timestamp.get("semantic_kind") != "stored_epoch_coordinate"
                or timestamp.get("event_utc_instant_claim") is not False
                or timestamp.get("timezone_basis") != "upstream_encoded_epoch"
                or timestamp.get("timezone_inferred") is not False
            ):
                raise ContractError(f"THULab stored-coordinate semantics are invalid: {dataset_id}")
        elif decision == "admit_dataset_adapter":
            adapter = row.get("mixed_axis_adapter")
            rules = adapter.get("source_rules") if isinstance(adapter, Mapping) else None
            if not isinstance(rules, list) or not rules:
                raise ContractError(f"THULab dataset adapter rules are missing: {dataset_id}")
            seen_rule_ids: set[str] = set()
            for rule in rules:
                if not isinstance(rule, Mapping):
                    raise ContractError(f"THULab dataset adapter rule is invalid: {dataset_id}")
                rule_id = str(rule.get("rule_id", ""))
                if not rule_id or rule_id in seen_rule_ids:
                    raise ContractError(f"THULab dataset adapter rule id is invalid: {dataset_id}")
                seen_rule_ids.add(rule_id)
                action = rule.get("action")
                timestamp = rule.get("timestamp")
                if action == "exclude_entire_source_file":
                    if timestamp is not None or rule.get("row_filter") is not None:
                        raise ContractError(f"THULab excluded adapter rule emits data: {dataset_id}:{rule_id}")
                    continue
                if action != "materialize" or not isinstance(timestamp, Mapping):
                    raise ContractError(f"THULab adapter action is invalid: {dataset_id}:{rule_id}")
                source_column = str(timestamp.get("source_column", ""))
                if (
                    timestamp.get("output_unit") != "us"
                    or timestamp.get("output_column") != "timestamp_unix_us"
                    or not source_column
                    or not isinstance(timestamp.get("event_utc_instant_claim"), bool)
                    or timestamp.get("timezone_inferred") is not False
                ):
                    raise ContractError(f"THULab adapter output is not int64-us: {dataset_id}:{rule_id}")
                if source_column == "Time":
                    expected_semantic = (
                        "unix_epoch_instant"
                        if timestamp.get("event_utc_instant_claim") is True
                        else "stored_epoch_coordinate"
                    )
                    if (
                        timestamp.get("source_unit") not in {"s", "ms", "us"}
                        or timestamp.get("source_unit_verified") is not True
                        or timestamp.get("semantic_kind") != expected_semantic
                        or timestamp.get("timezone_basis") != "upstream_encoded_epoch"
                        or (
                            timestamp.get("event_utc_instant_claim") is True
                            and timestamp.get("utc_instant_verified") is not True
                        )
                    ):
                        raise ContractError(f"THULab adapter Time unit is invalid: {dataset_id}:{rule_id}")
                else:
                    temporal = temporal_by_name.get(source_column)
                    if (
                        temporal is None
                        or timestamp.get("source_format_verified") is not True
                        or timestamp.get("event_utc_instant_claim")
                        != temporal.get("event_utc_instant_claim")
                        or timestamp.get("semantic_kind") != temporal.get("semantic_kind")
                        or timestamp.get("timezone_basis") != temporal.get("timezone_basis")
                    ):
                        raise ContractError(f"THULab adapter temporal source is unbound: {dataset_id}:{rule_id}")
    temporal_fields = [
        field
        for row in datasets
        for field in row.get("temporal_string_fields", [])
        if isinstance(field, Mapping)
    ]
    dropped_fields = [
        field
        for row in datasets
        for field in row.get("dropped_timestamp_like_string_fields", [])
        if isinstance(field, Mapping)
    ]
    decisions.update({
        "temporal_string_exact_definition_count": sum(
            int(field.get("candidate_definition_count", 0))
            for field in temporal_fields if field.get("event_utc_instant_claim") is True
        ),
        "temporal_string_coordinate_definition_count": sum(
            int(field.get("candidate_definition_count", 0))
            for field in temporal_fields if field.get("event_utc_instant_claim") is False
        ),
        "temporal_string_dropped_definition_count": sum(
            int(field.get("candidate_definition_count", 0)) for field in dropped_fields
        ),
    })
    coverage = policy.get("temporal_string_coverage", {})
    if int(coverage.get("converted_definition_count", -1)) != (
        decisions["temporal_string_exact_definition_count"]
        + decisions["temporal_string_coordinate_definition_count"]
    ):
        raise ContractError("THULab temporal-string converted count drift")
    if int(coverage.get("dropped_clock_only_definition_count", -1)) != decisions[
        "temporal_string_dropped_definition_count"
    ]:
        raise ContractError("THULab temporal-string dropped count drift")
    return decisions


def _validate_binding_shape(binding: Any, *, role: str) -> None:
    if not isinstance(binding, Mapping):
        raise ContractError(f"{role} binding is missing")
    path = binding.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ContractError(f"{role} binding path is not absolute")
    size = binding.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ContractError(f"{role} binding byte count is invalid")
    if not SHA256_RE.fullmatch(str(binding.get("sha256", ""))):
        raise ContractError(f"{role} binding SHA-256 is invalid")


def _require_exact_binding(
    actual: Any, expected: Mapping[str, Any], *, role: str,
) -> None:
    _validate_binding_shape(actual, role=role)
    if dict(actual) != dict(expected):
        raise ContractError(f"Core19 audit {role} path/hash binding mismatch")


def _regular_file_binding(path: Path, *, role: str) -> dict[str, Any]:
    absolute = Path(os.path.abspath(path))
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        raise ContractError(f"cannot stat {role}: {absolute}") from exc
    if not stat.S_ISREG(mode):
        raise ContractError(f"{role} must be a regular non-symlink file: {absolute}")
    return _file_binding(absolute)


def _is_contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _run_fresh_core_audit(
    *, auditor_path: Path, scope_path: Path, policy_path: Path,
    producer_path: Path, active_root: Path, backup_root: Path,
    output_root: Path, zstd_path: Path,
) -> dict[str, Any]:
    auditor_binding = _regular_file_binding(auditor_path, role="Core19 auditor")
    if auditor_binding["sha256"] != EXPECTED_CORE_AUDITOR_SHA256:
        raise ContractError("Core19 auditor is not the hard-pinned formal auditor")
    spec = importlib.util.spec_from_file_location(
        f"project179_core_auditor_{EXPECTED_CORE_AUDITOR_SHA256}",
        auditor_binding["path"],
    )
    if spec is None or spec.loader is None:
        raise ContractError("cannot load the hard-pinned Core19 auditor")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        audit_function = getattr(module, "audit")
        result = audit_function(
            scope_path=scope_path,
            policy_path=policy_path,
            producer_path=producer_path,
            active_root=active_root,
            backup_root=backup_root,
            output_root=output_root,
            zstd_path=zstd_path,
        )
    except Exception as exc:
        raise ContractError(f"fresh Core19 cold audit failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ContractError("fresh Core19 cold audit did not return a receipt object")
    return result


def _validate_core_audit_receipt(
    receipt_path: Path, *, manifest_path: Path, top: Mapping[str, Any],
    scope: Mapping[str, Any], scope_path: Path, scope_binding: Mapping[str, Any],
    policy_path: Path,
    policy_binding: Mapping[str, Any], producer_binding: Mapping[str, Any],
    producer_path: Path, auditor_path: Path, auditor_binding: Mapping[str, Any],
    active_root: Path, backup_root: Path, zstd_path: Path,
    zstd_binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = receipt_path.resolve()
    receipt_binding = _regular_file_binding(path, role="Core19 cold-audit receipt")
    receipt = _json(path)
    if receipt.get("schema") != CORE_AUDIT_SCHEMA:
        raise ContractError("unsupported Core19 cold-audit receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("mode") != "cold_read_only":
        raise ContractError("Core19 cold-audit receipt is not a cold PASS")
    if receipt.get("scope_id") != scope.get("scope_id"):
        raise ContractError("Core19 cold-audit scope id mismatch")
    if int(receipt.get("dataset_count", -1)) != 19:
        raise ContractError("Core19 cold-audit dataset count must be 19")

    input_bindings = receipt.get("input_bindings")
    required_input_bindings = {
        "scope", "policy", "producer", "auditor", "zstd",
        "active_source_contract", "active_top_manifest",
    }
    if not isinstance(input_bindings, Mapping) or set(input_bindings) != required_input_bindings:
        raise ContractError("Core19 cold-audit input binding set is incomplete")
    for name, expected in (
        ("scope", scope_binding),
        ("policy", policy_binding),
        ("producer", producer_binding),
        ("auditor", auditor_binding),
        ("zstd", zstd_binding),
    ):
        _require_exact_binding(input_bindings.get(name), expected, role=name)

    active_root = Path(os.path.abspath(active_root))
    source_contract_path = active_root / "SOURCE_CONTRACT.json"
    source_top_path = active_root / "manifest.json"
    source_contract_binding = _regular_file_binding(
        source_contract_path, role="active SOURCE_CONTRACT"
    )
    source_top_binding = _regular_file_binding(source_top_path, role="active top manifest")
    _require_exact_binding(
        input_bindings.get("active_source_contract"),
        source_contract_binding,
        role="active SOURCE_CONTRACT",
    )
    _require_exact_binding(
        input_bindings.get("active_top_manifest"),
        source_top_binding,
        role="active top manifest",
    )
    source_contract = _json(source_contract_path)
    if (
        source_contract.get("active_manifest_path") != str(source_top_path)
        or source_contract.get("active_manifest_sha256") != source_top_binding["sha256"]
    ):
        raise ContractError("active SOURCE_CONTRACT does not bind the rehashed active top manifest")

    manifest_binding = _regular_file_binding(manifest_path, role="Core19 top manifest")
    output_binding = receipt.get("output_binding")
    if not isinstance(output_binding, Mapping):
        raise ContractError("Core19 cold-audit output binding is missing")
    if output_binding.get("root") != str(manifest_path.parent):
        raise ContractError("Core19 cold-audit output root does not equal the Core19 manifest root")
    _require_exact_binding(
        output_binding.get("top_manifest"), manifest_binding, role="output top manifest"
    )

    core_ids = [str(value) for value in scope["core_csv_datasets"]]
    datasets = receipt.get("datasets")
    if (
        not isinstance(datasets, list)
        or len(datasets) != 19
        or any(not isinstance(row, Mapping) for row in datasets)
        or [str(row.get("dataset", "")) for row in datasets] != core_ids
        or any(row.get("status") != "PASS" for row in datasets)
    ):
        raise ContractError("Core19 cold-audit dataset identities/statuses differ from scope")
    top_rows = top.get("dataset_manifests")
    if not isinstance(top_rows, list) or len(top_rows) != 19:
        raise ContractError("Core19 top dataset manifest index is incomplete")
    output_root = manifest_path.parent
    receipt_count_fields = {
        "timestamp_admitted": "timestamp_admitted",
        "timestamp_selected": "timestamp_selected",
        "timestamp_blocked": "timestamp_blocked",
        "candidate_columns": "numeric_candidates",
        "selected_columns": "numeric_selected",
        "chunks": "chunks",
        "chunk_bytes": "chunk_bytes",
    }
    for dataset_id, top_row, audit_row in zip(core_ids, top_rows, datasets, strict=True):
        dataset_path = output_root / "datasets" / dataset_id / "manifest.json"
        dataset_binding = _regular_file_binding(
            dataset_path, role=f"Core19 dataset manifest {dataset_id}"
        )
        if not _is_contained(dataset_path, output_root):
            raise ContractError(f"Core19 dataset manifest escapes output root: {dataset_id}")
        if top_row.get("path") != str(dataset_path) or top_row.get("sha256") != dataset_binding["sha256"]:
            raise ContractError(f"Core19 top dataset path/hash mismatch: {dataset_id}")
        _require_exact_binding(
            audit_row.get("output_manifest"),
            dataset_binding,
            role=f"receipt dataset output manifest {dataset_id}",
        )
        dataset_manifest = _json(dataset_path)
        counts = dataset_manifest.get("counts")
        if not isinstance(counts, Mapping) or top_row.get("counts") != counts:
            raise ContractError(f"Core19 top/dataset counts mismatch: {dataset_id}")
        if any(audit_row.get(receipt_name) != counts.get(count_name) for receipt_name, count_name in receipt_count_fields.items()):
            raise ContractError(f"Core19 receipt/top/dataset counts mismatch: {dataset_id}")
        if audit_row.get("selected_bytes") != dataset_manifest.get("selection", {}).get("selected_bytes"):
            raise ContractError(f"Core19 receipt selected-byte count mismatch: {dataset_id}")
    if receipt.get("totals") != top.get("totals"):
        raise ContractError("Core19 cold-audit totals differ from the Core19 manifest")

    checks = receipt.get("release_checks")
    if not isinstance(checks, Mapping):
        raise ContractError("Core19 cold-audit release checks are missing")
    if set(checks) != CORE_AUDIT_RELEASE_CHECK_KEYS:
        raise ContractError("Core19 cold-audit release check key set is not exact")
    failed = [name for name in CORE_AUDIT_TRUE_CHECKS if checks.get(name) is not True]
    if failed:
        raise ContractError(
            "Core19 cold-audit release checks are not all true: " + ", ".join(failed)
        )
    if checks.get("limit_bytes_per_dataset") != ONE_GIB:
        raise ContractError("Core19 cold-audit release limit is not exactly 1 GiB")
    if checks.get("timestamp_blocked") != 0:
        raise ContractError("Core19 cold-audit release has blocked timestamps")
    fresh_receipt = _run_fresh_core_audit(
        auditor_path=auditor_path,
        scope_path=scope_path,
        policy_path=policy_path,
        producer_path=producer_path,
        active_root=active_root,
        backup_root=backup_root,
        output_root=output_root,
        zstd_path=zstd_path,
    )
    stored_comparable = dict(receipt)
    fresh_comparable = dict(fresh_receipt)
    stored_completed_at = stored_comparable.pop("completed_at", None)
    fresh_completed_at = fresh_comparable.pop("completed_at", None)
    if not isinstance(stored_completed_at, str) or not isinstance(fresh_completed_at, str):
        raise ContractError("Core19 cold-audit receipt lacks a completion timestamp")
    if stored_comparable != fresh_comparable:
        raise ContractError("stored Core19 audit receipt differs from a fresh cold rerun")
    return {
        "independent_audit_receipt": receipt_binding,
        "auditor_sha256": auditor_binding["sha256"],
        "zstd_path": zstd_binding["path"],
        "zstd_sha256": input_bindings["zstd"]["sha256"],
    }


def _validate_core_evidence(
    manifest_path: Path, audit_receipt_path: Path, *, scope: Mapping[str, Any],
    scope_path: Path, scope_binding: Mapping[str, Any], policy_path: Path,
    policy_binding: Mapping[str, Any], producer_path: Path,
    producer_binding: Mapping[str, Any], auditor_path: Path,
    auditor_binding: Mapping[str, Any], active_root: Path, backup_root: Path,
    zstd_path: Path, zstd_binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = manifest_path.resolve()
    output_root = path.parent
    top = _json(path)
    core_ids = [str(value) for value in scope["core_csv_datasets"]]
    if top.get("schema") != "project179-core19-numeric-utc-scope-v1":
        raise ContractError("unsupported Core19 materialization manifest")
    if top.get("status") != "complete" or int(top.get("dataset_count", -1)) != 19:
        raise ContractError("Core19 materialization is not a complete 19-dataset run")
    if top.get("datasets") != core_ids:
        raise ContractError("Core19 materialization order/identities differ from scope")
    if top.get("policy_sha256") != policy_binding["sha256"] or int(top.get("limit_bytes", -1)) != ONE_GIB:
        raise ContractError("Core19 materialization policy or 1 GiB cap binding mismatch")
    producer = top.get("producer", {})
    if (
        producer.get("path") != producer_binding["path"]
        or producer.get("sha256") != producer_binding["sha256"]
    ):
        raise ContractError("Core19 materialization producer path/hash mismatch")
    if (
        top.get("active_root") != str(Path(os.path.abspath(active_root)))
        or top.get("backup_root") != str(Path(os.path.abspath(backup_root)))
    ):
        raise ContractError("Core19 materialization source-root binding mismatch")
    rows = top.get("dataset_manifests")
    if (
        not isinstance(rows, list)
        or len(rows) != 19
        or any(not isinstance(row, Mapping) for row in rows)
        or [str(row.get("dataset", "")) for row in rows] != core_ids
    ):
        raise ContractError("Core19 dataset manifest index must exactly follow scope order")
    selected_columns = selected_bytes = timestamp_columns = 0
    for dataset_id, row in zip(core_ids, rows, strict=True):
        if row.get("status") != "complete":
            raise ContractError(f"invalid Core19 dataset receipt row: {dataset_id}")
        dataset_path = output_root / "datasets" / dataset_id / "manifest.json"
        if row.get("path") != str(dataset_path) or not _is_contained(dataset_path, output_root):
            raise ContractError(f"Core19 dataset manifest path mismatch: {dataset_id}")
        dataset_binding = _regular_file_binding(
            dataset_path, role=f"Core19 dataset manifest {dataset_id}"
        )
        if dataset_binding["sha256"] != row.get("sha256"):
            raise ContractError(f"Core19 dataset manifest hash mismatch: {dataset_id}")
        dataset = _json(dataset_path)
        if dataset.get("schema") != "project179-core19-numeric-utc-dataset-v1":
            raise ContractError(f"unsupported Core19 dataset manifest: {dataset_id}")
        if dataset.get("status") != "complete" or dataset.get("dataset") != dataset_id:
            raise ContractError(f"Core19 dataset is not complete: {dataset_id}")
        identity = dataset.get("identity", {})
        if identity.get("policy_sha256") != policy_binding["sha256"] or identity.get("producer_sha256") != producer_binding["sha256"]:
            raise ContractError(f"Core19 dataset policy/producer binding mismatch: {dataset_id}")
        if not isinstance(dataset.get("counts"), Mapping) or row.get("counts") != dataset.get("counts"):
            raise ContractError(f"Core19 dataset count receipt mismatch: {dataset_id}")
        selection = dataset.get("selection", {})
        if int(selection.get("limit_bytes", -1)) != ONE_GIB:
            raise ContractError(f"Core19 dataset cap mismatch: {dataset_id}")
        dataset_selected_bytes = int(selection.get("selected_bytes", -1))
        if dataset_selected_bytes < 0 or dataset_selected_bytes > ONE_GIB:
            raise ContractError(f"Core19 selected bytes exceed 1 GiB: {dataset_id}")
        columns = dataset.get("columns", [])
        if not isinstance(columns, list) or any(row.get("dtype") not in CORE_DTYPES for row in columns):
            raise ContractError(f"Core19 emitted a non numeric payload dtype: {dataset_id}")
        for column in columns:
            if "timestamp_unit" in column:
                if column.get("timestamp_unit") != "unix_epoch_microseconds" or column.get("dtype") != "int64":
                    raise ContractError(f"Core19 timestamp is not int64 Unix microseconds: {dataset_id}")
                timestamp_columns += 1
        selected_columns += len(columns)
        selected_bytes += dataset_selected_bytes
    if int(top.get("totals", {}).get("timestamp_blocked", -1)) != 0:
        raise ContractError("Core19 materialization contains blocked timestamps")
    audit_evidence = _validate_core_audit_receipt(
        audit_receipt_path,
        manifest_path=path,
        top=top,
        scope=scope,
        scope_path=scope_path,
        scope_binding=scope_binding,
        policy_path=policy_path,
        policy_binding=policy_binding,
        producer_path=producer_path,
        producer_binding=producer_binding,
        auditor_path=auditor_path,
        auditor_binding=auditor_binding,
        active_root=active_root,
        backup_root=backup_root,
        zstd_path=zstd_path,
        zstd_binding=zstd_binding,
    )
    return {
        "state": "materialized_and_verified",
        "completion_claim": True,
        "output_root": str(output_root),
        "manifest": str(path),
        "manifest_bytes": path.stat().st_size,
        "manifest_sha256": sha256_file(path),
        "dataset_count": 19,
        "selected_column_count": selected_columns,
        "selected_logical_bytes": selected_bytes,
        "selected_timestamp_column_count": timestamp_columns,
        **audit_evidence,
    }


def _validate_thulab_adapter_receipts(
    dataset_id: str, dataset: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    """Validate per-file time axes, filters, and exclusions for one mixed dataset."""
    timestamp_policy = dataset.get("timestamp_policy", {})
    rules = timestamp_policy.get("source_rules")
    if not isinstance(rules, list) or not rules:
        raise ContractError(f"THULab adapter has no explicit source rules: {dataset_id}")
    rule_by_id: dict[str, Mapping[str, Any]] = {}
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise ContractError(f"THULab adapter contains a non-object rule: {dataset_id}")
        rule_id = str(rule.get("rule_id", ""))
        if not rule_id or rule_id in rule_by_id:
            raise ContractError(f"THULab adapter rule id is empty/duplicate: {dataset_id}")
        action = rule.get("action")
        if action not in {"materialize", "exclude_entire_source_file"}:
            raise ContractError(f"THULab adapter rule has an unsupported action: {dataset_id}:{rule_id}")
        timestamp = rule.get("timestamp")
        if action == "materialize":
            if (
                not isinstance(timestamp, Mapping)
                or timestamp.get("output_dtype") != "int64"
                or timestamp.get("output_unit") != "us"
                or timestamp.get("output_column") != "timestamp_unix_us"
            ):
                raise ContractError(
                    f"THULab adapter rule timestamp is not int64 Unix microseconds: {dataset_id}:{rule_id}"
                )
        elif timestamp is not None:
            raise ContractError(f"excluded THULab adapter rule unexpectedly has a timestamp: {dataset_id}:{rule_id}")
        expected_files = rule.get("expected_source_file_count")
        if not isinstance(expected_files, int) or isinstance(expected_files, bool) or expected_files <= 0:
            raise ContractError(f"THULab adapter rule lacks a positive file count: {dataset_id}:{rule_id}")
        row_filter = rule.get("row_filter")
        if row_filter is not None:
            if action != "materialize" or not isinstance(row_filter, Mapping):
                raise ContractError(f"invalid THULab adapter row filter: {dataset_id}:{rule_id}")
            expected_counts = [
                int(row_filter.get(name, -1))
                for name in ("expected_input_rows", "expected_output_rows", "expected_drop_rows")
            ]
            if (
                min(expected_counts) < 0
                or expected_counts[0] != expected_counts[1] + expected_counts[2]
            ):
                raise ContractError(f"inconsistent THULab adapter row-filter policy counts: {dataset_id}:{rule_id}")
        rule_by_id[rule_id] = rule

    table = dataset.get("table", {})
    entries = table.get("source_file_tables")
    if not isinstance(entries, list) or not entries:
        raise ContractError(f"THULab adapter has no per-source-file receipts: {dataset_id}")
    source_rows = dataset.get("source_tsfiles")
    source_paths = {
        str(row.get("path", "")) for row in source_rows or [] if isinstance(row, Mapping)
    }
    if len(source_paths) != len(source_rows or []) or "" in source_paths:
        raise ContractError(f"THULab adapter source-file identity list is invalid: {dataset_id}")
    receipt_files = receipt.get("files")
    receipt_by_path = {
        str(row.get("path", "")): row
        for row in receipt_files or [] if isinstance(row, Mapping)
    }
    if len(receipt_by_path) != len(receipt_files or []) or set(receipt_by_path) != source_paths:
        raise ContractError(f"THULab adapter receipt files do not equal source files: {dataset_id}")

    entries_by_rule: dict[str, list[Mapping[str, Any]]] = {name: [] for name in rule_by_id}
    table_paths: set[str] = set()
    observed_output_rows = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ContractError(f"THULab adapter contains a non-object table receipt: {dataset_id}")
        source_path = str(entry.get("source_path", ""))
        rule_id = str(entry.get("adapter_rule_id", ""))
        if not source_path or source_path in table_paths or rule_id not in rule_by_id:
            raise ContractError(f"THULab adapter table receipt is unbound/duplicate: {dataset_id}:{source_path}")
        table_paths.add(source_path)
        entries_by_rule[rule_id].append(entry)
        rule = rule_by_id[rule_id]
        action = entry.get("source_action")
        if action != rule.get("action"):
            raise ContractError(f"THULab adapter source action differs from policy: {dataset_id}:{source_path}")
        timestamp = entry.get("timestamp")
        retained = entry.get("retained_fields")
        if not isinstance(retained, list):
            raise ContractError(f"THULab adapter retained-field receipt is missing: {dataset_id}:{source_path}")
        file_receipt = receipt_by_path.get(source_path, {})
        if action == "exclude_entire_source_file":
            if timestamp is not None or retained or entry.get("source_rows") is not None or entry.get("row_filter_observed") is not None:
                raise ContractError(f"excluded THULab source file contains output claims: {dataset_id}:{source_path}")
            if file_receipt.get("status") != "excluded_entire_source_file_by_dataset_adapter":
                raise ContractError(f"excluded THULab source file lacks an exclusion receipt: {dataset_id}:{source_path}")
            continue
        if (
            not isinstance(timestamp, Mapping)
            or timestamp.get("output_dtype") != "int64"
            or timestamp.get("output_unit") != "us"
            or timestamp.get("output_column") != "timestamp_unix_us"
        ):
            raise ContractError(f"THULab adapter source timestamp is not int64 Unix microseconds: {dataset_id}:{source_path}")
        if file_receipt.get("status") != "verified_and_materialized":
            raise ContractError(f"materialized THULab source file lacks a receipt: {dataset_id}:{source_path}")
        rule_timestamp = rule.get("timestamp")
        if not isinstance(rule_timestamp, Mapping) or any(
            timestamp.get(key) != rule_timestamp.get(key)
            for key in (
                "source_column", "source_unit", "output_column", "output_unit",
                "output_dtype", "event_utc_instant_claim", "semantic_kind",
                "timezone_basis", "timezone_inferred",
            )
        ):
            raise ContractError(
                f"THULab adapter source timestamp differs from its rule: {dataset_id}:{source_path}"
            )
        observed = entry.get("row_filter_observed")
        if not isinstance(observed, Mapping):
            raise ContractError(f"THULab adapter source lacks observed row counts: {dataset_id}:{source_path}")
        input_rows = int(observed.get("input_rows", -1))
        output_rows = int(observed.get("output_rows", -1))
        dropped_rows = int(observed.get("dropped_rows", -1))
        if min(input_rows, output_rows, dropped_rows) < 0 or input_rows != output_rows + dropped_rows:
            raise ContractError(f"THULab adapter observed row counts are inconsistent: {dataset_id}:{source_path}")
        if int(entry.get("source_rows", -1)) != output_rows:
            raise ContractError(f"THULab adapter retained row count differs from filter receipt: {dataset_id}:{source_path}")
        observed_output_rows += output_rows

    if table_paths != source_paths:
        raise ContractError(f"THULab adapter table receipts do not cover every source file: {dataset_id}")
    if int(dataset.get("source_rows", -1)) != observed_output_rows:
        raise ContractError(f"THULab adapter dataset source_rows do not match file receipts: {dataset_id}")
    for rule_id, rule in rule_by_id.items():
        matching = entries_by_rule[rule_id]
        if len(matching) != int(rule["expected_source_file_count"]):
            raise ContractError(f"THULab adapter rule file count differs from policy: {dataset_id}:{rule_id}")
        row_filter = rule.get("row_filter")
        if row_filter is None:
            continue
        aggregate = {
            name: sum(int(entry["row_filter_observed"][name]) for entry in matching)
            for name in ("input_rows", "output_rows", "dropped_rows")
        }
        expected = {
            "input_rows": int(row_filter["expected_input_rows"]),
            "output_rows": int(row_filter["expected_output_rows"]),
            "dropped_rows": int(row_filter["expected_drop_rows"]),
        }
        if aggregate != expected:
            raise ContractError(f"THULab adapter row-filter receipt differs from policy: {dataset_id}:{rule_id}")


def _validate_thulab_dataset_policy_evidence(
    dataset_id: str,
    dataset: Mapping[str, Any],
    expected_policy: Mapping[str, Any],
) -> None:
    """Match one emitted policy receipt to its frozen dataset/rule contract."""
    emitted = dataset.get("timestamp_policy")
    if not isinstance(emitted, Mapping):
        raise ContractError(f"THULab dataset timestamp policy is missing: {dataset_id}")
    decision = expected_policy.get("decision")
    if emitted.get("admission_decision") != decision:
        raise ContractError(f"THULab dataset admission decision drift: {dataset_id}")

    raw_temporal = expected_policy.get("temporal_string_fields", [])
    observed_temporal = emitted.get("temporal_string_fields", [])
    if not isinstance(raw_temporal, list) or not isinstance(observed_temporal, list):
        raise ContractError(f"THULab temporal policy receipt is invalid: {dataset_id}")
    raw_by_name = {
        str(field.get("source_name", "")): field
        for field in raw_temporal if isinstance(field, Mapping)
    }
    observed_by_name = {
        str(field.get("source_name", "")): field
        for field in observed_temporal if isinstance(field, Mapping)
    }
    if (
        len(raw_by_name) != len(raw_temporal)
        or len(observed_by_name) != len(observed_temporal)
        or set(raw_by_name) != set(observed_by_name)
        or "" in raw_by_name
    ):
        raise ContractError(f"THULab temporal policy field set drift: {dataset_id}")
    for source_name, expected in raw_by_name.items():
        observed = observed_by_name[source_name]
        comparisons = {
            "source_name": source_name,
            "source_type": expected.get("source_type"),
            "source_category": expected.get("source_category"),
            "output_name": expected.get("output_name"),
            "output_unit": "us",
            "output_dtype": "int64",
            "event_utc_instant_claim": expected.get("event_utc_instant_claim"),
            "semantic_kind": expected.get("semantic_kind"),
            "timezone_basis": expected.get("timezone_basis"),
        }
        if any(observed.get(key) != value for key, value in comparisons.items()):
            raise ContractError(
                f"THULab temporal output/semantic receipt drift: {dataset_id}:{source_name}"
            )

    table = dataset.get("table")
    source_tables = table.get("source_file_tables") if isinstance(table, Mapping) else None
    if not isinstance(source_tables, list) or not source_tables:
        raise ContractError(f"THULab source table receipts are missing: {dataset_id}")
    for source_table in source_tables:
        retained = source_table.get("retained_fields") if isinstance(source_table, Mapping) else None
        if not isinstance(retained, list):
            raise ContractError(f"THULab retained-field receipt is invalid: {dataset_id}")
        for field in retained:
            if not isinstance(field, Mapping):
                raise ContractError(f"THULab retained-field row is invalid: {dataset_id}")
            source_name = str(field.get("source_name", ""))
            if source_name in raw_by_name:
                expected = raw_by_name[source_name]
                if (
                    field.get("output_name") != expected.get("output_name")
                    or field.get("dtype") != "int64"
                    or field.get("conversion")
                    != "explicit_temporal_string_to_int64_unix_us"
                ):
                    raise ContractError(
                        f"THULab temporal retained-field receipt drift: {dataset_id}:{source_name}"
                    )
            elif (
                not source_name
                or field.get("output_name") != source_name
                or field.get("dtype") not in THULAB_DTYPES
                or field.get("conversion") != "native_numeric_little_endian"
            ):
                raise ContractError(
                    f"THULab retained payload is not numeric-only: {dataset_id}:{source_name}"
                )

    if decision in {"admit_exact_utc", "admit_stored_epoch_coordinate"}:
        expected_timestamp = expected_policy.get("timestamp")
        if not isinstance(expected_timestamp, Mapping):
            raise ContractError(f"THULab frozen timestamp policy is missing: {dataset_id}")
        comparisons = {
            "source_column": expected_timestamp.get("source_column"),
            "source_unit": expected_timestamp.get("source_unit"),
            "output_column": "timestamp_unix_us",
            "output_unit": "us",
            "output_dtype": "int64",
            "semantic_kind": expected_timestamp.get("semantic_kind"),
            "timezone_inferred": False,
        }
        if any(emitted.get(key) != value for key, value in comparisons.items()):
            raise ContractError(f"THULab primary timestamp receipt drift: {dataset_id}")
        if decision == "admit_exact_utc":
            if (
                emitted.get("event_utc_instant_claim") is not True
                or emitted.get("timezone") != "UTC"
                or emitted.get("timezone_basis") != "explicit_utc_event_instant"
            ):
                raise ContractError(f"THULab exact UTC receipt semantics drift: {dataset_id}")
        elif (
            emitted.get("event_utc_instant_claim") is not False
            or emitted.get("timezone_basis") != "upstream_encoded_epoch"
        ):
            raise ContractError(f"THULab stored-coordinate receipt semantics drift: {dataset_id}")
        if emitted.get("source_rules") != []:
            raise ContractError(f"THULab non-adapter dataset has source rules: {dataset_id}")
        return

    if decision != "admit_dataset_adapter":
        raise ContractError(f"THULab dataset policy is not admitted: {dataset_id}")
    adapter = expected_policy.get("mixed_axis_adapter")
    expected_rules = adapter.get("source_rules") if isinstance(adapter, Mapping) else None
    observed_rules = emitted.get("source_rules")
    if not isinstance(expected_rules, list) or not isinstance(observed_rules, list):
        raise ContractError(f"THULab adapter rule receipt is missing: {dataset_id}")
    expected_by_id = {
        str(rule.get("rule_id", "")): rule
        for rule in expected_rules if isinstance(rule, Mapping)
    }
    observed_by_id = {
        str(rule.get("rule_id", "")): rule
        for rule in observed_rules if isinstance(rule, Mapping)
    }
    if (
        len(expected_by_id) != len(expected_rules)
        or len(observed_by_id) != len(observed_rules)
        or set(expected_by_id) != set(observed_by_id)
        or "" in expected_by_id
    ):
        raise ContractError(f"THULab adapter rule set drift: {dataset_id}")
    for rule_id, expected in expected_by_id.items():
        observed = observed_by_id[rule_id]
        for key in ("action", "table_names", "expected_source_file_count"):
            if observed.get(key) != expected.get(key):
                raise ContractError(f"THULab adapter rule binding drift: {dataset_id}:{rule_id}")
        expected_timestamp = expected.get("timestamp")
        observed_timestamp = observed.get("timestamp")
        if expected.get("action") == "exclude_entire_source_file":
            if expected_timestamp is not None or observed_timestamp is not None:
                raise ContractError(f"THULab excluded rule timestamp drift: {dataset_id}:{rule_id}")
            continue
        if not isinstance(expected_timestamp, Mapping) or not isinstance(observed_timestamp, Mapping):
            raise ContractError(f"THULab adapter timestamp receipt is missing: {dataset_id}:{rule_id}")
        comparisons = {
            "source_column": expected_timestamp.get("source_column"),
            "source_unit": expected_timestamp.get("source_unit"),
            "output_column": "timestamp_unix_us",
            "output_unit": "us",
            "output_dtype": "int64",
            "event_utc_instant_claim": expected_timestamp.get("event_utc_instant_claim"),
            "semantic_kind": expected_timestamp.get("semantic_kind"),
            "timezone_basis": expected_timestamp.get("timezone_basis"),
            "timezone_inferred": False,
        }
        if any(observed_timestamp.get(key) != value for key, value in comparisons.items()):
            raise ContractError(f"THULab adapter timestamp/semantic drift: {dataset_id}:{rule_id}")


def _expected_thulab_binding_sha256(
    *, dataset: Mapping[str, Any], receipt: Mapping[str, Any],
    scope_id: str, scope_sha256: str, policy_file_sha256: str,
    policy_sha256: str, runtime: Mapping[str, str],
) -> str:
    selection = dataset.get("selection")
    source_files = dataset.get("source_tsfiles")
    if not isinstance(selection, Mapping) or not isinstance(source_files, list):
        raise ContractError("THULab binding inputs are missing")
    normalized_sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in source_files:
        if not isinstance(row, Mapping):
            raise ContractError("THULab source binding contains a non-object row")
        relative = _safe_relative_path(row.get("path"), str(dataset.get("dataset_id", "")))
        size = row.get("bytes")
        digest = str(row.get("sha256", ""))
        if (
            relative in seen_sources
            or not isinstance(size, int) or isinstance(size, bool) or size < 0
            or not SHA256_RE.fullmatch(digest)
        ):
            raise ContractError("THULab source binding row is invalid")
        seen_sources.add(relative)
        normalized_sources.append({"path": relative, "bytes": size, "sha256": digest})
    source_receipt_sha256 = str(receipt.get("source_receipt_sha256", ""))
    if not SHA256_RE.fullmatch(source_receipt_sha256):
        raise ContractError("THULab source receipt hash is invalid")
    payload = {
        "format_version": "project179-thulab-numeric-materialization-v1",
        "scope_id": scope_id,
        "scope_sha256": scope_sha256,
        "dataset_id": dataset.get("dataset_id"),
        "repo_id": dataset.get("repo_id"),
        "revision": dataset.get("revision"),
        "source_receipt_sha256": source_receipt_sha256,
        "source_files": normalized_sources,
        "policy_file_sha256": policy_file_sha256,
        "policy_sha256": policy_sha256,
        "dataset_byte_cap": selection.get("dataset_byte_cap"),
        "chunk_byte_cap": selection.get("chunk_byte_cap"),
        "metadata_compressed_byte_cap": selection.get("metadata_compressed_byte_cap"),
        "storage_layout": dataset.get("storage_layout"),
        "producer_sha256": runtime["producer_sha256"],
        "helper_timeseries_sha256": runtime["helper_timeseries_sha256"],
        "helper_tsfile_sha256": runtime["helper_tsfile_sha256"],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _safe_relative_path(value: Any, context: str) -> str:
    relative = str(value)
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ContractError(f"unsafe THULab materialized path: {context}")
    return relative


def _regular_file_within(root: Path, relative: str, context: str) -> Path:
    path = root / relative
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ContractError(f"missing THULab file: {context}:{relative}") from error
    if not stat.S_ISREG(mode):
        raise ContractError(f"THULab path is not a regular file: {context}:{relative}")
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ContractError(f"THULab file escapes its root: {context}:{relative}")
    return path


def _validate_sharded_thulab_dataset(
    dataset_id: str, dataset_path: Path, dataset: Mapping[str, Any]
) -> tuple[int, int]:
    """Stream-validate sharded metadata and whole-shard byte bindings."""
    raw_shards = dataset.get("payload_shards")
    index = dataset.get("stream_index")
    if "streams" in dataset or not isinstance(raw_shards, list) or not raw_shards:
        raise ContractError(f"invalid sharded THULab manifest shape: {dataset_id}")
    if not isinstance(index, Mapping) or index.get("schema") != "project179-thulab-stream-index-v1":
        raise ContractError(f"missing THULab sharded stream index: {dataset_id}")
    seen_paths = {"dataset_manifest.json"}
    shard_sizes: dict[str, int] = {}
    shard_order: dict[str, int] = {}
    shard_paths: dict[str, Path] = {}
    chunk_cap = int(dataset.get("selection", {}).get("chunk_byte_cap", -1))
    metadata_cap = int(
        dataset.get("selection", {}).get("metadata_compressed_byte_cap", -1)
    )
    if chunk_cap < 8 or chunk_cap > 64 << 20:
        raise ContractError(f"invalid THULab sharded payload cap: {dataset_id}")
    if metadata_cap <= 0 or metadata_cap > 1 << 30:
        raise ContractError(f"invalid THULab metadata cap: {dataset_id}")
    for expected_index, shard in enumerate(raw_shards):
        if not isinstance(shard, Mapping) or int(shard.get("shard_index", -1)) != expected_index:
            raise ContractError(f"THULab payload shard order mismatch: {dataset_id}")
        relative = _safe_relative_path(shard.get("path"), dataset_id)
        size = int(shard.get("bytes", -1))
        path = _regular_file_within(dataset_path.parent, relative, dataset_id)
        if relative in seen_paths or size <= 0 or size > chunk_cap:
            raise ContractError(f"invalid THULab payload shard metadata: {dataset_id}")
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != shard.get("sha256"):
            raise ContractError(f"THULab payload shard byte binding mismatch: {dataset_id}:{relative}")
        seen_paths.add(relative)
        shard_sizes[relative] = size
        shard_order[relative] = expected_index
        shard_paths[relative] = path

    index_relative = _safe_relative_path(index.get("path"), dataset_id)
    index_path = _regular_file_within(dataset_path.parent, index_relative, dataset_id)
    if index_relative in seen_paths:
        raise ContractError(f"duplicate THULab stream-index path: {dataset_id}")
    seen_paths.add(index_relative)
    if (
        index.get("compression") != "gzip"
        or index.get("record_encoding") != "canonical_json_lines_utf8"
        or int(index.get("compressed_byte_cap", -1)) != metadata_cap
        or not index_path.is_file()
        or index_path.stat().st_size > metadata_cap
        or index_path.stat().st_size != int(index.get("compressed_bytes", -1))
        or sha256_file(index_path) != index.get("compressed_sha256")
    ):
        raise ContractError(f"THULab stream-index byte binding mismatch: {dataset_id}")

    totals = {
        "stream_count": 0, "chunk_count": 0, "rows": 0,
        "valid_values": 0, "missing_values": 0, "logical_bytes": 0,
    }
    digest = hashlib.sha256()
    byte_count = record_count = entity_count = temporal_count = 0
    footer: Mapping[str, Any] | None = None
    current_entity_key: tuple[str, str, str] | None = None
    current_entity_had_stream = False
    previous_entity_sort_key: tuple[str, str, bytes] | None = None
    consumed = {path: 0 for path in shard_sizes}
    last_shard_index = 0
    segment_handle: Any | None = None
    current_segment_path: str | None = None
    try:
        with gzip.open(index_path, "rb") as handle:
            for raw_line in handle:
                record_count += 1
                byte_count += len(raw_line)
                digest.update(raw_line)
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ContractError(f"invalid THULab stream-index JSON: {dataset_id}") from error
                if not isinstance(record, Mapping) or raw_line != _canonical_json(record) + b"\n":
                    raise ContractError(f"noncanonical THULab stream-index record: {dataset_id}")
                kind = record.get("record_type")
                if record_count == 1:
                    if (
                        kind != "header"
                        or record.get("schema") != "project179-thulab-stream-index-v1"
                        or record.get("storage_layout") != "sharded_v1"
                        or record.get("dataset_id") != dataset_id
                        or record.get("binding_sha256") != dataset.get("binding_sha256")
                        or {
                            "producer_sha256": record.get("producer_sha256"),
                            "helper_timeseries_sha256": record.get("helper_timeseries_sha256"),
                            "helper_tsfile_sha256": record.get("helper_tsfile_sha256"),
                        } != dataset.get("runtime_code_bindings")
                    ):
                        raise ContractError(f"THULab stream-index header mismatch: {dataset_id}")
                    continue
                if footer is not None:
                    raise ContractError(f"THULab stream-index record follows footer: {dataset_id}")
                if kind == "entity":
                    key = (
                        str(record.get("source_path", "")),
                        str(record.get("table_name", "")),
                        str(record.get("entity_id", "")),
                    )
                    entity_values = record.get("entity_values")
                    entity_sort_key = (
                        key[0], key[1], _canonical_json(entity_values)
                    ) if isinstance(entity_values, list) else None
                    if (
                        "" in key or not isinstance(entity_values, list)
                        or hashlib.sha256(_canonical_json(entity_values)).hexdigest() != key[2]
                        or (current_entity_key is not None and not current_entity_had_stream)
                        or (
                            previous_entity_sort_key is not None
                            and entity_sort_key is not None
                            and entity_sort_key <= previous_entity_sort_key
                        )
                    ):
                        raise ContractError(f"invalid THULab entity dictionary record: {dataset_id}")
                    current_entity_key = key
                    current_entity_had_stream = False
                    previous_entity_sort_key = entity_sort_key
                    entity_count += 1
                    continue
                if kind == "footer":
                    if current_entity_key is not None and not current_entity_had_stream:
                        raise ContractError(f"THULab entity has no adjacent streams: {dataset_id}")
                    footer = record
                    continue
                stream = record.get("stream")
                if kind != "stream" or not isinstance(stream, Mapping):
                    raise ContractError(f"unsupported THULab stream-index record: {dataset_id}")
                if "entity_values" in stream:
                    raise ContractError(f"THULab sharded stream repeats string entity values: {dataset_id}")
                if int(stream.get("stream_index", -1)) != totals["stream_count"]:
                    raise ContractError(f"THULab stream order mismatch: {dataset_id}")
                entity_key = (
                    str(stream.get("source_path", "")), str(stream.get("table_name", "")),
                    str(stream.get("entity_id", "")),
                )
                if entity_key != current_entity_key or stream.get("dtype") not in THULAB_DTYPES:
                    raise ContractError(f"THULab stream metadata is invalid: {dataset_id}")
                current_entity_had_stream = True
                width = int(stream.get("width_bytes", -1))
                expected_width = {"int32": 4, "int64": 8, "float32": 4, "float64": 8}[stream["dtype"]]
                if width != expected_width:
                    raise ContractError(f"THULab stream dtype/width mismatch: {dataset_id}")
                if stream.get("conversion") == "explicit_temporal_string_to_int64_unix_us":
                    source_field = str(stream.get("source_field", ""))
                    if (
                        stream.get("dtype") != "int64"
                        or stream.get("field")
                        != f"{source_field}__timestamp_unix_us"
                        or stream.get("tsfile_type") not in {"STRING", "TEXT"}
                    ):
                        raise ContractError(f"THULab temporal stream is not int64: {dataset_id}")
                    temporal_count += 1
                elif stream.get("conversion") != "native_numeric_little_endian":
                    raise ContractError(f"THULab stream conversion is invalid: {dataset_id}")
                stream_rows = stream_valid = 0
                chunks = stream.get("chunks")
                if not isinstance(chunks, list) or not chunks:
                    raise ContractError(f"THULab stream has no chunks: {dataset_id}")
                for expected_chunk, chunk in enumerate(chunks):
                    if not isinstance(chunk, Mapping) or int(chunk.get("chunk_index", -1)) != expected_chunk:
                        raise ContractError(f"THULab chunk order mismatch: {dataset_id}")
                    rows = int(chunk.get("row_count", -1))
                    valid = int(chunk.get("valid_count", -1))
                    missing = int(chunk.get("missing_count", -1))
                    if rows <= 0 or min(valid, missing) < 0 or valid + missing != rows:
                        raise ContractError(f"THULab chunk counts are invalid: {dataset_id}")
                    if chunk.get("timestamp_unit") != "us" or chunk.get("timestamp_dtype") != "int64":
                        raise ContractError(f"THULab chunk timestamp is not int64-us: {dataset_id}")
                    expected_lengths = {
                        "timestamps": rows * 8,
                        "values": valid * width,
                        "missing_bitmap": (rows + 7) // 8,
                    }
                    chunk_payloads: dict[str, bytes] = {}
                    for name, expected_length in expected_lengths.items():
                        segment = chunk.get(f"{name}_segment")
                        if not isinstance(segment, Mapping):
                            raise ContractError(f"THULab chunk lacks segment metadata: {dataset_id}")
                        relative = _safe_relative_path(segment.get("path"), dataset_id)
                        offset = int(segment.get("offset", -1))
                        length = int(segment.get("length", -1))
                        order = shard_order.get(relative, -1)
                        if (
                            relative not in shard_sizes or offset != consumed[relative]
                            or length != expected_length or offset + length > shard_sizes[relative]
                            or order < last_shard_index
                            or length != int(chunk.get(f"{name}_bytes", -1))
                            or segment.get("sha256") != chunk.get(f"{name}_sha256")
                        ):
                            raise ContractError(f"THULab payload segment boundary mismatch: {dataset_id}")
                        if current_segment_path != relative:
                            if segment_handle is not None:
                                segment_handle.close()
                            segment_handle = shard_paths[relative].open("rb")
                            current_segment_path = relative
                        assert segment_handle is not None
                        segment_handle.seek(offset)
                        payload = segment_handle.read(length)
                        if (
                            len(payload) != length
                            or hashlib.sha256(payload).hexdigest() != segment.get("sha256")
                        ):
                            raise ContractError(f"THULab payload segment hash mismatch: {dataset_id}")
                        chunk_payloads[name] = payload
                        consumed[relative] += length
                        last_shard_index = order
                    bitmap = chunk_payloads["missing_bitmap"]
                    if (
                        sum(byte.bit_count() for byte in bitmap) != missing
                        or (rows % 8 and bitmap[-1] >> (rows % 8))
                    ):
                        raise ContractError(f"THULab missing bitmap is inconsistent: {dataset_id}")
                    first_timestamp = int.from_bytes(
                        chunk_payloads["timestamps"][:8], "little", signed=True
                    )
                    last_timestamp = int.from_bytes(
                        chunk_payloads["timestamps"][-8:], "little", signed=True
                    )
                    if (
                        first_timestamp != int(chunk.get("first_timestamp_unix_us", -(1 << 63)))
                        or last_timestamp != int(chunk.get("last_timestamp_unix_us", -(1 << 63)))
                    ):
                        raise ContractError(f"THULab timestamp boundaries are inconsistent: {dataset_id}")
                    totals["chunk_count"] += 1
                    stream_rows += rows
                    stream_valid += valid
                if (
                    stream_rows != int(stream.get("row_count", -1))
                    or stream_valid != int(stream.get("valid_count", -1))
                    or stream_rows - stream_valid != int(stream.get("missing_count", -1))
                    or stream_rows * (8 + width) != int(stream.get("logical_bytes", -1))
                ):
                    raise ContractError(f"THULab stream aggregate mismatch: {dataset_id}")
                totals["stream_count"] += 1
                totals["rows"] += stream_rows
                totals["valid_values"] += stream_valid
                totals["missing_values"] += stream_rows - stream_valid
                totals["logical_bytes"] += stream_rows * (8 + width)
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        raise ContractError(f"cannot decode THULab stream-index gzip: {dataset_id}") from error
    finally:
        if segment_handle is not None:
            segment_handle.close()
    if (
        footer is None or footer.get("totals") != totals
        or int(footer.get("stream_record_count", -1)) != totals["stream_count"]
        or int(footer.get("entity_record_count", -1)) != entity_count
        or dataset.get("totals") != totals
        or any(consumed[path] != size for path, size in shard_sizes.items())
        or record_count != int(index.get("record_count", -1))
        or byte_count != int(index.get("uncompressed_bytes", -1))
        or digest.hexdigest() != index.get("uncompressed_sha256")
        or totals["stream_count"] != int(index.get("stream_record_count", -1))
        or entity_count != int(index.get("entity_record_count", -1))
        or entity_count != int(dataset.get("selected_entity_dictionary_count", -1))
        or dataset.get("entity_dictionary_semantics")
        != "metadata_only_typed_TAG_partition_provenance_not_numeric_payload"
    ):
        raise ContractError(f"THULab sharded index totals/binding mismatch: {dataset_id}")
    actual_paths: set[str] = set()
    for child in dataset_path.parent.rglob("*"):
        mode = os.lstat(child).st_mode
        if stat.S_ISLNK(mode):
            raise ContractError(f"THULab dataset contains a symlink: {dataset_id}")
        if stat.S_ISREG(mode):
            actual_paths.add(child.relative_to(dataset_path.parent).as_posix())
        elif not stat.S_ISDIR(mode):
            raise ContractError(f"THULab dataset contains a special file: {dataset_id}")
    if actual_paths != seen_paths:
        raise ContractError(f"THULab sharded exact file set mismatch: {dataset_id}")
    return totals["stream_count"], temporal_count


def _validate_thulab_evidence(
    manifest_path: Path, *, scope: Mapping[str, Any], scope_sha256: str,
    policy: Mapping[str, Any], policy_sha256: str, producer_sha256: str,
    helper_timeseries_sha256: str, helper_tsfile_sha256: str,
) -> dict[str, Any]:
    input_path = manifest_path.absolute()
    if not stat.S_ISREG(os.lstat(input_path).st_mode):
        raise ContractError("THULab top manifest is not a regular file")
    path = input_path.resolve()
    top = _json(path)
    timestamp_ids = [str(row["id"]) for row in scope["timestamp_datasets"]]
    timestamp_scope_by_id = {
        str(row["id"]): row for row in scope["timestamp_datasets"]
    }
    raw_policy_rows = policy.get("datasets")
    if not isinstance(raw_policy_rows, list):
        raise ContractError("THULab policy dataset rows are missing")
    policy_by_id = {
        str(row.get("dataset_id", "")): row
        for row in raw_policy_rows if isinstance(row, Mapping)
    }
    if len(policy_by_id) != len(raw_policy_rows) or set(policy_by_id) != set(timestamp_ids):
        raise ContractError("THULab evidence policy rows do not equal the frozen scope")
    if top.get("schema") != "project179-thulab-numeric-collection-v1":
        raise ContractError("unsupported THULab materialization manifest")
    if top.get("status") != "complete":
        raise ContractError("THULab materialization contains blocked datasets")
    if top.get("scope_id") != scope.get("scope_id") or top.get("scope_sha256") != scope_sha256:
        raise ContractError("THULab materialization scope binding mismatch")
    if top.get("policy_file_sha256") != policy_sha256:
        raise ContractError("THULab materialization policy hash mismatch")
    expected_runtime = {
        "producer_sha256": producer_sha256,
        "helper_timeseries_sha256": helper_timeseries_sha256,
        "helper_tsfile_sha256": helper_tsfile_sha256,
    }
    if top.get("runtime_code_bindings") != expected_runtime:
        raise ContractError("THULab top-level runtime code binding mismatch")
    if int(top.get("full_scope_dataset_count", -1)) != 160 or int(top.get("selected_dataset_count", -1)) != 160:
        raise ContractError("THULab materialization is not a full 160-dataset run")
    if top.get("status_counts") != {"materialized": 160, "blocked": 0}:
        raise ContractError("THULab status counts are not 160 materialized and zero blocked")
    if int(top.get("dataset_byte_cap", -1)) != ONE_GIB:
        raise ContractError("THULab dataset cap is not 1 GiB")
    if int(top.get("metadata_compressed_byte_cap", -1)) != ONE_GIB:
        raise ContractError("THULab metadata compressed cap is not 1 GiB")
    if not 8 <= int(top.get("chunk_byte_cap", -1)) <= 64 << 20:
        raise ContractError("THULab payload shard cap is outside [8, 64 MiB]")
    storage_layout = top.get("storage_layout")
    if storage_layout != "sharded_v1":
        raise ContractError("formal THULab materialization must use sharded_v1 storage")
    rows = top.get("datasets")
    if not isinstance(rows, list) or len(rows) != 160:
        raise ContractError("THULab receipt index is incomplete")
    root = path.parent
    seen: set[str] = set()
    selected_streams = selected_bytes = timestamp_value_streams = 0
    adapter_datasets = 0
    for row in rows:
        dataset_id = str(row.get("dataset_id", ""))
        if dataset_id in seen or dataset_id not in timestamp_ids or row.get("status") != "materialized":
            raise ContractError(f"invalid THULab dataset receipt row: {dataset_id}")
        seen.add(dataset_id)
        receipt_relative = _safe_relative_path(row.get("receipt"), dataset_id)
        if receipt_relative != f"_receipts/{dataset_id}.json":
            raise ContractError(f"THULab receipt path is not canonical: {dataset_id}")
        receipt_path = _regular_file_within(root, receipt_relative, dataset_id)
        if sha256_file(receipt_path) != row.get("receipt_sha256"):
            raise ContractError(f"THULab receipt hash mismatch: {dataset_id}")
        receipt = _json(receipt_path)
        if (
            receipt.get("status") != "materialized"
            or receipt.get("dataset_id") != dataset_id
            or receipt.get("storage_layout") != storage_layout
        ):
            raise ContractError(f"THULab dataset receipt is not materialized: {dataset_id}")
        expected_scope_row = timestamp_scope_by_id[dataset_id]
        if (
            receipt.get("repo_id") != expected_scope_row.get("repo_id")
            or receipt.get("revision") != expected_scope_row.get("revision")
            or receipt.get("scope_sha256") != scope_sha256
            or receipt.get("policy_file_sha256") != policy_sha256
            or receipt.get("runtime_code_bindings") != expected_runtime
            or receipt.get("storage_layout") != storage_layout
            or receipt.get("output_directory") != dataset_id
        ):
            raise ContractError(f"THULab dataset receipt identity binding mismatch: {dataset_id}")
        dataset_path = _regular_file_within(
            root, f"{dataset_id}/dataset_manifest.json", dataset_id
        )
        if sha256_file(dataset_path) != receipt.get("manifest_sha256"):
            raise ContractError(f"THULab dataset manifest hash mismatch: {dataset_id}")
        dataset = _json(dataset_path)
        if dataset.get("schema") != "project179-thulab-numeric-materialization-v1":
            raise ContractError(f"unsupported THULab dataset manifest: {dataset_id}")
        manifest_without_provenance = dict(dataset)
        provenance_sha256 = manifest_without_provenance.pop("provenance_sha256", None)
        if provenance_sha256 != hashlib.sha256(
            _canonical_json(manifest_without_provenance)
        ).hexdigest():
            raise ContractError(f"THULab dataset provenance hash mismatch: {dataset_id}")
        expected_policy = policy_by_id[dataset_id]
        expected_dataset_policy_sha256 = hashlib.sha256(
            _canonical_json(dict(expected_policy))
        ).hexdigest()
        if (
            dataset.get("dataset_id") != dataset_id
            or dataset.get("repo_id") != expected_scope_row.get("repo_id")
            or dataset.get("revision") != expected_scope_row.get("revision")
            or dataset.get("scope_sha256") != scope_sha256
            or dataset.get("timestamp_policy", {}).get("policy_file_sha256") != policy_sha256
            or dataset.get("storage_layout") != storage_layout
            or dataset.get("runtime_code_bindings") != expected_runtime
            or dataset.get("binding_sha256") != receipt.get("binding_sha256")
            or dataset.get("source_receipt_sha256")
            != receipt.get("source_receipt_sha256")
            or dataset.get("timestamp_policy", {}).get("policy_sha256")
            != expected_dataset_policy_sha256
            or receipt.get("policy_sha256") != expected_dataset_policy_sha256
        ):
            raise ContractError(f"THULab dataset manifest identity binding mismatch: {dataset_id}")
        expected_binding_sha256 = _expected_thulab_binding_sha256(
            dataset=dataset,
            receipt=receipt,
            scope_id=str(scope.get("scope_id")),
            scope_sha256=scope_sha256,
            policy_file_sha256=policy_sha256,
            policy_sha256=expected_dataset_policy_sha256,
            runtime=expected_runtime,
        )
        if (
            dataset.get("binding_sha256") != expected_binding_sha256
            or receipt.get("binding_sha256") != expected_binding_sha256
        ):
            raise ContractError(f"THULab dataset binding hash mismatch: {dataset_id}")
        _validate_thulab_dataset_policy_evidence(
            dataset_id, dataset, expected_policy
        )
        timestamp = dataset.get("timestamp_policy", {})
        decision = timestamp.get("admission_decision")
        if decision in {"admit_exact_utc", "admit_stored_epoch_coordinate"}:
            if (
                timestamp.get("output_dtype") != "int64"
                or timestamp.get("output_unit") != "us"
                or timestamp.get("output_column") != "timestamp_unix_us"
            ):
                raise ContractError(f"THULab timestamp is not int64 Unix microseconds: {dataset_id}")
        elif decision == "admit_dataset_adapter":
            adapter_datasets += 1
            _validate_thulab_adapter_receipts(dataset_id, dataset, receipt)
        else:
            raise ContractError(f"THULab dataset has unsupported admission decision: {dataset_id}")
        selection = dataset.get("selection", {})
        if (
            int(selection.get("dataset_byte_cap", -1)) != ONE_GIB
            or int(selection.get("chunk_byte_cap", -1)) != int(top.get("chunk_byte_cap", -2))
            or int(selection.get("metadata_compressed_byte_cap", -1)) != ONE_GIB
        ):
            raise ContractError(f"THULab dataset cap binding mismatch: {dataset_id}")
        logical_bytes = int(selection.get("selected_logical_bytes", -1))
        if logical_bytes < 0 or logical_bytes > ONE_GIB:
            raise ContractError(f"THULab selected bytes exceed 1 GiB: {dataset_id}")
        candidate_count = int(selection.get("candidate_stream_count", -1))
        selected_count = int(selection.get("selected_stream_count", -1))
        witness = selection.get("longest_prefix_witness")
        if (
            candidate_count < 0 or selected_count <= 0 or selected_count > candidate_count
            or selected_count != int(receipt.get("selected_stream_count", -1))
            or candidate_count != int(receipt.get("candidate_stream_count", -1))
            or logical_bytes != int(receipt.get("selected_logical_bytes", -1))
            or not isinstance(witness, Mapping)
            or receipt.get("longest_prefix_witness") != witness
            or int(witness.get("candidate_stream_count", -1)) != candidate_count
            or int(witness.get("selected_stream_count", -1)) != selected_count
            or int(witness.get("selected_logical_bytes", -1)) != logical_bytes
        ):
            raise ContractError(f"THULab selection receipt/manifest mismatch: {dataset_id}")
        all_selected = selected_count == candidate_count
        first_unselected = witness.get("first_unselected_logical_bytes")
        if (
            witness.get("all_candidates_selected") is not all_selected
            or (
                all_selected
                and (
                    first_unselected is not None
                    or witness.get("proof_boundary") != "all_candidates_selected"
                )
            )
            or (
                not all_selected
                and (
                    not isinstance(first_unselected, int)
                    or isinstance(first_unselected, bool)
                    or first_unselected <= 0
                    or logical_bytes + first_unselected <= ONE_GIB
                    or witness.get("proof_boundary")
                    != "selected_bytes_plus_first_unselected_exceeds_cap"
                )
            )
        ):
            raise ContractError(f"THULab longest-prefix witness is invalid: {dataset_id}")
        if storage_layout == "sharded_v1":
            stream_count, temporal_count = _validate_sharded_thulab_dataset(
                dataset_id, dataset_path, dataset
            )
            selected_streams += stream_count
            timestamp_value_streams += temporal_count
            if stream_count != selected_count or dataset.get("totals", {}).get("logical_bytes") != logical_bytes:
                raise ContractError(f"THULab index/selection totals mismatch: {dataset_id}")
        else:
            streams = dataset.get("streams", [])
            if not isinstance(streams, list) or any(stream.get("dtype") not in THULAB_DTYPES for stream in streams):
                raise ContractError(f"THULab emitted a non numeric payload dtype: {dataset_id}")
            for stream in streams:
                if stream.get("conversion") == "explicit_temporal_string_to_int64_unix_us":
                    if stream.get("dtype") != "int64":
                        raise ContractError(f"THULab temporal-string stream is not int64: {dataset_id}")
                    timestamp_value_streams += 1
                for chunk in stream.get("chunks", []):
                    if chunk.get("timestamp_unit") != "us" or chunk.get("timestamp_dtype") != "int64":
                        raise ContractError(f"THULab stream coordinate is not int64 Unix microseconds: {dataset_id}")
            selected_streams += len(streams)
        selected_bytes += logical_bytes
    if seen != set(timestamp_ids):
        raise ContractError("THULab receipt identities are incomplete")
    if adapter_datasets != 3:
        raise ContractError("THULab evidence must contain exactly three mixed-dataset adapters")
    expected_children = {
        "_receipts", "MATERIALIZATION_MANIFEST.json", *timestamp_ids,
    }
    actual_children: set[str] = set()
    for child in root.iterdir():
        mode = os.lstat(child).st_mode
        should_be_directory = child.name == "_receipts" or child.name in timestamp_ids
        if (
            stat.S_ISLNK(mode)
            or (should_be_directory and not stat.S_ISDIR(mode))
            or (not should_be_directory and not stat.S_ISREG(mode))
        ):
            raise ContractError("THULab full-root path type mismatch")
        actual_children.add(child.name)
    if actual_children != expected_children:
        raise ContractError("THULab full-root exact path set mismatch")
    receipts_dir = root / "_receipts"
    expected_receipts = {f"{dataset_id}.json" for dataset_id in timestamp_ids}
    actual_receipts: set[str] = set()
    for child in receipts_dir.iterdir():
        if not stat.S_ISREG(os.lstat(child).st_mode):
            raise ContractError("THULab receipt tree contains a non-regular file")
        actual_receipts.add(child.name)
    if actual_receipts != expected_receipts:
        raise ContractError("THULab receipt exact file set mismatch")
    return {
        "state": "materialized_and_verified",
        "output_root": str(root),
        "manifest": str(path),
        "manifest_bytes": path.stat().st_size,
        "manifest_sha256": sha256_file(path),
        "dataset_count": 160,
        "selected_stream_count": selected_streams,
        "selected_logical_bytes": selected_bytes,
        "temporal_string_value_stream_count": timestamp_value_streams,
        "mixed_dataset_adapter_count": adapter_datasets,
        "storage_layout": storage_layout,
        "verification_scope": (
            "structure_hash_payload_segment_and_prefix_witness_verification; "
            "candidate selection is not recomputed from source TsFiles"
        ),
    }


def _not_attached(kind: str) -> dict[str, Any]:
    return {
        "state": "not_attached",
        "kind": kind,
        "completion_claim": False,
        "reason": "No top-level materialization manifest was supplied to this contract build.",
    }


def build_contract(
    *, scope_path: Path, core_policy_path: Path, thulab_policy_path: Path,
    core_producer_path: Path, thulab_producer_path: Path,
    core_auditor_path: Path = DEFAULT_CORE_AUDITOR,
    thulab_helper_timeseries_path: Path = DEFAULT_THULAB_HELPER_TIMESERIES,
    thulab_helper_tsfile_path: Path = DEFAULT_THULAB_HELPER_TSFILE,
    core_manifest_path: Path | None = None,
    core_audit_receipt_path: Path | None = None,
    core_active_root: Path | None = None,
    core_backup_root: Path | None = None,
    core_zstd_path: Path | None = None,
    thulab_manifest_path: Path | None = None,
    server192_state: str = "not_checked_by_this_run",
    server192_detail: str | None = None,
) -> dict[str, Any]:
    if server192_state not in SERVER192_STATES:
        raise ContractError(f"unsupported server192 state: {server192_state}")
    core_materialization_inputs = (
        core_manifest_path,
        core_audit_receipt_path,
        core_active_root,
        core_backup_root,
        core_zstd_path,
    )
    supplied_core_inputs = sum(value is not None for value in core_materialization_inputs)
    if supplied_core_inputs not in {0, len(core_materialization_inputs)}:
        raise ContractError(
            "Core19 manifest, audit receipt, active root, backup root, and zstd must be supplied together"
        )
    scope_binding = _file_binding(scope_path)
    core_policy_binding = _file_binding(core_policy_path)
    thulab_policy_binding = _file_binding(thulab_policy_path)
    core_producer_binding = _file_binding(core_producer_path)
    core_auditor_binding = _file_binding(core_auditor_path)
    if core_auditor_binding["sha256"] != EXPECTED_CORE_AUDITOR_SHA256:
        raise ContractError("Core19 auditor SHA-256 is not the hard-pinned formal auditor")
    thulab_producer_binding = _file_binding(thulab_producer_path)
    thulab_helper_timeseries_binding = _file_binding(thulab_helper_timeseries_path)
    thulab_helper_tsfile_binding = _file_binding(thulab_helper_tsfile_path)
    core_zstd_binding: Mapping[str, Any] | None = None
    if core_manifest_path is not None:
        assert core_audit_receipt_path is not None
        assert core_active_root is not None
        assert core_backup_root is not None
        assert core_zstd_path is not None
        if str(Path(os.path.abspath(core_zstd_path))) != EXPECTED_CORE_ZSTD_PATH:
            raise ContractError("Core19 zstd path is not the hard-pinned /usr/bin/zstd")
        core_zstd_binding = _regular_file_binding(core_zstd_path, role="Core19 zstd")
        if core_zstd_binding["sha256"] != EXPECTED_CORE_ZSTD_SHA256:
            raise ContractError("Core19 zstd SHA-256 is not the hard-pinned formal binary")
    scope = _json(scope_path)
    core_ids, timestamp_rows = _validate_scope(scope)
    core_policy = _json(core_policy_path)
    thulab_policy = _json(thulab_policy_path)
    core_claims = _validate_core_policy(
        core_policy, scope_id=str(scope["scope_id"]), core_ids=core_ids
    )
    thulab_claims = _validate_thulab_policy(
        thulab_policy,
        scope_id=str(scope["scope_id"]),
        scope_sha256=scope_binding["sha256"],
        timestamp_rows=timestamp_rows,
    )
    core_evidence = (
        _not_attached("core19")
        if core_manifest_path is None
        else _validate_core_evidence(
            core_manifest_path,
            core_audit_receipt_path,
            scope=scope,
            scope_path=scope_path,
            scope_binding=scope_binding,
            policy_path=core_policy_path,
            policy_binding=core_policy_binding,
            producer_path=core_producer_path,
            producer_binding=core_producer_binding,
            auditor_path=core_auditor_path,
            auditor_binding=core_auditor_binding,
            active_root=core_active_root,
            backup_root=core_backup_root,
            zstd_path=core_zstd_path,
            zstd_binding=core_zstd_binding,
        )
    )
    thulab_evidence = (
        _not_attached("thulab160")
        if thulab_manifest_path is None
        else _validate_thulab_evidence(
            thulab_manifest_path,
            scope=scope,
            scope_sha256=scope_binding["sha256"],
            policy=thulab_policy,
            policy_sha256=thulab_policy_binding["sha256"],
            producer_sha256=thulab_producer_binding["sha256"],
            helper_timeseries_sha256=thulab_helper_timeseries_binding["sha256"],
            helper_tsfile_sha256=thulab_helper_tsfile_binding["sha256"],
        )
    )
    complete_count = sum(
        evidence["state"] == "materialized_and_verified"
        for evidence in (core_evidence, thulab_evidence)
    )
    evidence_state = (
        "materialized_and_verified"
        if complete_count == 2
        else "partially_attached"
        if complete_count == 1
        else "contract_only_no_materialization_claim"
    )
    return {
        "schema": "project179-numeric-utc-contract-v1",
        "scope_id": scope["scope_id"],
        "evidence_state": evidence_state,
        "evidence_verification_boundary": (
            "Attached materialization evidence is checked by structure, SHA-256, payload "
            "segments, declared totals, and longest-prefix witness. Core19 completion also "
            "requires a cold_read_only PASS receipt from the SHA-256-bound independent "
            "auditor. The contract verifier does not rescan source TsFiles to recompute "
            "candidate selection."
        ),
        "dataset_identity_counts": {
            "core_csv": 19,
            "thulab_timestamp": 160,
            "total": 179,
            "identity_formula": "19 Core CSV identities + 160 frozen THULab repository identities",
        },
        "input_bindings": {
            "scope": scope_binding,
            "core19_policy": core_policy_binding,
            "thulab_policy": thulab_policy_binding,
            "core19_producer": core_producer_binding,
            "core19_auditor": core_auditor_binding,
            "thulab_producer": thulab_producer_binding,
            "thulab_helper_timeseries": thulab_helper_timeseries_binding,
            "thulab_helper_tsfile": thulab_helper_tsfile_binding,
        },
        "core19_cold_audit_gate": {
            "auditor_sha256": EXPECTED_CORE_AUDITOR_SHA256,
            "zstd_path": EXPECTED_CORE_ZSTD_PATH,
            "zstd_sha256": EXPECTED_CORE_ZSTD_SHA256,
            "fresh_rerun_required_for_completion": True,
            "receipt_match": "exact_except_completed_at",
        },
        "derived_view_contract": {
            "source_mutation": "forbidden; outputs are derived copies with receipt-backed identities",
            "timestamp_representation": {
                "dtype": "signed_int64",
                "unit": "unix_epoch_microseconds",
                "byte_order": "little",
                "missing_or_invalid": "producer-specific explicit drop/block receipts; no guessed fallback",
            },
            "payload_types": {
                "allowed_logical_families": ["integer", "floating_point"],
                "core19_physical_dtypes": sorted(CORE_DTYPES),
                "thulab_physical_dtypes": sorted(THULAB_DTYPES),
                "dropped_source_types": ["STRING", "TEXT", "BOOLEAN", "DATE", "BLOB"],
                "tag_boundary": (
                    "ordinary string TAG values may partition entity provenance but are not payload; "
                    "an explicitly admitted temporal TAG may additionally be emitted only as int64 Unix microseconds"
                ),
            },
            "size_cap": {
                "bytes_per_dataset": ONE_GIB,
                "label": "1 GiB",
                "core19_selection": "longest complete numeric-column prefix in original source-column order",
                "thulab_selection": "longest deterministic complete numeric-stream prefix in frozen source-schema order",
                "truncation": "never truncate a column or stream",
            },
            "thulab_storage": {
                "formal_layout": "sharded_v1",
                "payload_shard_max_bytes": 64 << 20,
                "stream_metadata": "canonical gzip JSONL with exact compressed and uncompressed hashes",
                "entity_dictionary": "typed TAG partition provenance metadata only; never numeric payload",
            },
            "timestamp_semantics": {
                "boundary": (
                    "event_utc_instant_claim=true is an exact-instant claim; false is only a deterministic "
                    "Unix-axis coordinate and must not be reported as a proven physical UTC event instant"
                ),
                "dataset_adapter_boundary": (
                    "The three mixed-axis datasets are counted separately as explicit dataset adapters; "
                    "they are not reclassified wholesale as exact UTC or stored-coordinate datasets."
                ),
                "core19": core_claims,
                "thulab": thulab_claims,
            },
            "benchmark_boundary": (
                "This contract attests preprocessing/storage inputs only; it is not a compression-ratio, "
                "compression-time, or decompression-time benchmark result."
            ),
        },
        "server166_materialization": {
            "core19": core_evidence,
            "thulab160": thulab_evidence,
            "verified_dataset_count": (
                int(core_evidence.get("dataset_count", 0))
                + int(thulab_evidence.get("dataset_count", 0))
            ),
        },
        "server192_propagation": {
            "state": server192_state,
            "detail": server192_detail,
            "completion_claim": False,
            "verification_boundary": (
                "This v1 builder has no server192 receipt-tree verifier and therefore cannot make a "
                "server192 completion claim."
            ),
        },
    }


def validate_contract(
    value: Mapping[str, Any], *, materialization_verified: bool = False,
) -> None:
    if value.get("schema") != "project179-numeric-utc-contract-v1":
        raise ContractError("unsupported numeric/UTC contract schema")
    counts = value.get("dataset_identity_counts", {})
    if counts.get("core_csv") != 19 or counts.get("thulab_timestamp") != 160 or counts.get("total") != 179:
        raise ContractError("contract dataset counts must be 19 + 160 = 179")
    bindings = value.get("input_bindings", {})
    required = {
        "scope", "core19_policy", "thulab_policy", "core19_producer", "core19_auditor",
        "thulab_producer", "thulab_helper_timeseries", "thulab_helper_tsfile",
    }
    if set(bindings) != required:
        raise ContractError("contract input binding set is incomplete")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping) or not SHA256_RE.fullmatch(str(binding.get("sha256", ""))):
            raise ContractError(f"invalid SHA-256 binding: {name}")
        if int(binding.get("bytes", -1)) < 0:
            raise ContractError(f"invalid byte binding: {name}")
    if bindings["core19_auditor"].get("sha256") != EXPECTED_CORE_AUDITOR_SHA256:
        raise ContractError("contract does not bind the hard-pinned formal Core19 auditor")
    audit_gate = value.get("core19_cold_audit_gate")
    if audit_gate != {
        "auditor_sha256": EXPECTED_CORE_AUDITOR_SHA256,
        "zstd_path": EXPECTED_CORE_ZSTD_PATH,
        "zstd_sha256": EXPECTED_CORE_ZSTD_SHA256,
        "fresh_rerun_required_for_completion": True,
        "receipt_match": "exact_except_completed_at",
    }:
        raise ContractError("contract Core19 cold-audit authority/gate drift")
    contract = value.get("derived_view_contract", {})
    timestamp = contract.get("timestamp_representation", {})
    if timestamp.get("dtype") != "signed_int64" or timestamp.get("unit") != "unix_epoch_microseconds":
        raise ContractError("contract timestamp representation drift")
    cap = contract.get("size_cap", {})
    if cap.get("bytes_per_dataset") != ONE_GIB or cap.get("truncation") != "never truncate a column or stream":
        raise ContractError("contract whole-column/stream 1 GiB cap drift")
    payload = contract.get("payload_types", {})
    if payload.get("allowed_logical_families") != ["integer", "floating_point"]:
        raise ContractError("contract payload type family drift")
    timestamp_semantics = contract.get("timestamp_semantics", {})
    thulab_semantics = timestamp_semantics.get("thulab", {})
    primary_count_keys = {
        "primary_exact_event_utc_dataset_count",
        "primary_stored_coordinate_dataset_count",
        "primary_dataset_adapter_count",
        "primary_blocked_dataset_count",
    }
    observed_primary_keys = {
        key for key in thulab_semantics if key.startswith("primary_")
    } if isinstance(thulab_semantics, Mapping) else set()
    if observed_primary_keys != primary_count_keys:
        raise ContractError("contract THULab primary decision categories are not exhaustive")
    primary_counts = [thulab_semantics.get(key) for key in primary_count_keys]
    if (
        any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in primary_counts)
        or sum(primary_counts) != 160
    ):
        raise ContractError("contract THULab primary decision counts must sum to exactly 160")
    if (
        thulab_semantics.get("primary_dataset_adapter_count") != 3
        or thulab_semantics.get("primary_blocked_dataset_count") != 0
    ):
        raise ContractError("contract THULab adapter/block decision counts drift")
    state = value.get("evidence_state")
    server166 = value.get("server166_materialization", {})
    core_evidence = server166.get("core19", {})
    thulab_evidence = server166.get("thulab160", {})
    core_state = core_evidence.get("state")
    thulab_state = thulab_evidence.get("state")
    materialized_states = (core_state, thulab_state)
    if any(item == "materialized_and_verified" for item in materialized_states) and not materialization_verified:
        raise ContractError("materialized evidence requires explicit fresh-build verification authority")
    if thulab_state not in {"not_attached", "materialized_and_verified"}:
        raise ContractError("unsupported THULab evidence state")
    if thulab_state == "materialized_and_verified" and thulab_evidence.get("dataset_count") != 160:
        raise ContractError("THULab evidence dataset count is not exactly 160")
    if core_state == "materialized_and_verified":
        receipt_binding = core_evidence.get("independent_audit_receipt")
        _validate_binding_shape(receipt_binding, role="Core19 independent audit receipt")
        if core_evidence.get("completion_claim") is not True:
            raise ContractError("Core19 completion requires an independent cold-audit PASS")
        if (
            core_evidence.get("auditor_sha256") != EXPECTED_CORE_AUDITOR_SHA256
            or core_evidence.get("auditor_sha256") != bindings["core19_auditor"]["sha256"]
        ):
            raise ContractError("Core19 evidence auditor hash differs from the contract binding")
        if (
            core_evidence.get("zstd_path") != EXPECTED_CORE_ZSTD_PATH
            or core_evidence.get("zstd_sha256") != EXPECTED_CORE_ZSTD_SHA256
        ):
            raise ContractError("Core19 evidence does not bind the hard-pinned zstd binary")
        if core_evidence.get("dataset_count") != 19:
            raise ContractError("Core19 evidence dataset count is not exactly 19")
    elif core_state == "not_attached":
        if core_evidence.get("completion_claim") is not False:
            raise ContractError("unattached Core19 evidence cannot make a completion claim")
    else:
        raise ContractError("unsupported Core19 evidence state")
    expected_verified_dataset_count = (
        (19 if core_state == "materialized_and_verified" else 0)
        + (160 if thulab_state == "materialized_and_verified" else 0)
    )
    if server166.get("verified_dataset_count") != expected_verified_dataset_count:
        raise ContractError("verified_dataset_count must be exactly 0, 19, 160, or 179 from evidence states")
    complete_count = sum(item == "materialized_and_verified" for item in materialized_states)
    expected_state = (
        "materialized_and_verified" if complete_count == 2 else
        "partially_attached" if complete_count == 1 else
        "contract_only_no_materialization_claim"
    )
    if state != expected_state:
        raise ContractError("contract evidence_state overclaims attached materialization")
    server192 = value.get("server192_propagation", {})
    if server192.get("state") not in SERVER192_STATES:
        raise ContractError("unsupported server192 propagation state")
    if server192.get("completion_claim") is not False:
        raise ContractError("server192 completion cannot be claimed without a receipt-tree verifier")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--core-policy", type=Path, default=DEFAULT_CORE_POLICY)
    parser.add_argument("--thulab-policy", type=Path, default=DEFAULT_THULAB_POLICY)
    parser.add_argument("--core-producer", type=Path, default=DEFAULT_CORE_PRODUCER)
    parser.add_argument("--core-auditor", type=Path, default=DEFAULT_CORE_AUDITOR)
    parser.add_argument("--thulab-producer", type=Path, default=DEFAULT_THULAB_PRODUCER)
    parser.add_argument(
        "--thulab-helper-timeseries",
        type=Path,
        default=DEFAULT_THULAB_HELPER_TIMESERIES,
    )
    parser.add_argument(
        "--thulab-helper-tsfile",
        type=Path,
        default=DEFAULT_THULAB_HELPER_TSFILE,
    )
    parser.add_argument("--core-manifest", type=Path)
    parser.add_argument("--core-audit-receipt", type=Path)
    parser.add_argument("--core-active-root", type=Path)
    parser.add_argument("--core-backup-root", type=Path)
    parser.add_argument("--core-zstd", type=Path)
    parser.add_argument("--thulab-manifest", type=Path)
    parser.add_argument("--server192-state", choices=sorted(SERVER192_STATES), default="not_checked_by_this_run")
    parser.add_argument("--server192-detail")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if --output differs from a fresh build")
    args = parser.parse_args(argv)
    contract = build_contract(
        scope_path=args.scope,
        core_policy_path=args.core_policy,
        thulab_policy_path=args.thulab_policy,
        core_producer_path=args.core_producer,
        core_auditor_path=args.core_auditor,
        thulab_producer_path=args.thulab_producer,
        thulab_helper_timeseries_path=args.thulab_helper_timeseries,
        thulab_helper_tsfile_path=args.thulab_helper_tsfile,
        core_manifest_path=args.core_manifest,
        core_audit_receipt_path=args.core_audit_receipt,
        core_active_root=args.core_active_root,
        core_backup_root=args.core_backup_root,
        core_zstd_path=args.core_zstd,
        thulab_manifest_path=args.thulab_manifest,
        server192_state=args.server192_state,
        server192_detail=args.server192_detail,
    )
    validate_contract(contract, materialization_verified=True)
    if args.check:
        if not args.output.is_file() or _json(args.output) != contract:
            raise ContractError(f"contract is missing or stale: {args.output}")
    else:
        _atomic_json(args.output, contract)
    print(json.dumps({
        "contract": str(args.output.resolve()),
        "evidence_state": contract["evidence_state"],
        "verified_dataset_count": contract["server166_materialization"]["verified_dataset_count"],
        "sha256": sha256_file(args.output) if args.output.is_file() else None,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
