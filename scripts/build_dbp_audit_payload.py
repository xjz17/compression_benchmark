#!/usr/bin/env python3
"""Build the public DBP native-float audit from frozen campaign receipts.

The complete path fails closed: all 169 scope datasets, all three runtimes,
every source aggregate, source preflights, artifact pins, normalized results,
and cross-runtime frame hashes must close before ``dbp-audit.json`` is written.
``--allow-partial`` is only for QA and writes explicitly named partial files
outside the deployment directory.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable

import jsonschema


SCHEMA_VERSION = "dbp-native-float-audit-v1"
RECEIPT_SCHEMA = "dbp-native-float-audit-build-v1"
FROZEN_SCOPE_SCHEMA = "tscb-screened-paper-scope-v1"
FROZEN_SCOPE_ID = "tscb-paper-realdata169-166-20261008-v1"
CAMPAIGN_BINDING_SCHEMA = "dbp-float-e2e-final-campaign-binding-v1"
GOLDEN_RECEIPT_SCHEMA = "dbp-float-e2e-169-controller-v1"
CONTROLLER_RECEIPT_SCHEMA = "dbp-float-e2e-169-controller-v1"
COLLECTION_RECEIPT_SCHEMA = "dbp-float-e2e-compact-collection-v1"
AGGREGATE_MANIFEST_SCHEMA = "dbp-float-e2e-aggregate-manifest-v1"
EXPECTED_SCOPE_DATASETS = 169
EXPECTED_GOLDEN_CASES = 7
EXPECTED_COLD_DECODER_PAIRS = 63
RUNTIMES = ("cpp", "java", "python")
WIDTH_BY_DTYPE = {"float32": 4, "float64": 8}
HEADER_BYTES_PER_BLOCK = 10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEPLOYMENT_DIR = (
    Path(__file__).resolve().parents[1] / "deploy" / "huggingface-compressionbenchmark"
)

EXACT_SOURCE_FIELDS = (
    "width",
    "blocks",
    "rows",
    "missing_count",
    "validity_bytes",
    "header_bytes",
    "valid_count",
    "original_bytes",
)
EXACT_CROSS_RUNTIME_FIELDS = (
    "width",
    "blocks",
    "rows",
    "missing_count",
    "validity_bytes",
    "header_bytes",
    "status",
    "valid_count",
    "original_bytes",
    "compressed_bytes",
    "scaled_blocks",
    "raw_blocks",
    "verified_values",
    "frame_sequence_sha256",
)
NUMERIC_RESULT_FIELDS = (
    "blocks",
    "rows",
    "missing_count",
    "validity_bytes",
    "header_bytes",
    "valid_count",
    "original_bytes",
    "compressed_bytes",
    "encode_ns",
    "decode_ns",
    "scaled_blocks",
    "raw_blocks",
    "verified_values",
)
CSV_FIELDS = (
    "status",
    "runtime",
    "dataset",
    "column",
    "source_column",
    "aggregate_id",
    "dtype",
    "width",
    "max_point",
    "rows",
    "valid_count",
    "missing_count",
    "native_bytes",
    "frame_bytes",
    "validity_bytes",
    "header_bytes",
    "compression_rate",
    "encode_ns",
    "decode_ns",
    "encode_ns_per_value",
    "decode_ns_per_value",
    "blocks",
    "scaled_blocks",
    "raw_blocks",
    "verified_values",
    "frame_sequence_sha256",
    "expected_sequence_chain_sha256",
    "result_sha256",
    "receipt_sha256",
    "source_manifest_receipt_sha256",
    "source_aggregates_sha256",
    "source_blocks_sha256",
    "source_block_catalog_sha256",
    "runtime_artifact_manifest_sha256",
    "source_column_inventory_sha256",
    "scope_sha256",
    "message",
)


class AuditBuildError(RuntimeError):
    """The requested payload is not eligible for publication."""


class Findings:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.suppressed_errors = 0
        self.failed_gates: collections.Counter[str] = collections.Counter()

    def fail(self, gate: str, message: str, dataset: str | None = None) -> None:
        self.failed_gates[gate] += 1
        row = {"gate": gate, "message": message}
        if dataset is not None:
            row["dataset"] = dataset
        if len(self.errors) < 1000:
            self.errors.append(row)
        else:
            self.suppressed_errors += 1

    def check(
        self,
        condition: bool,
        gate: str,
        message: str,
        dataset: str | None = None,
    ) -> bool:
        if not condition:
            self.fail(gate, message, dataset)
        return condition


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def public_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditBuildError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise AuditBuildError(
                    f"invalid JSONL {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise AuditBuildError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is boolean")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} is negative")
    return result


def stable_projection(value: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: value.get(key) for key in keys}


def is_lower_hex_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def parse_cpu_affinity(value: Any) -> set[int] | None:
    if isinstance(value, list):
        return {integer(item, "cpu_affinity") for item in value}
    if isinstance(value, str):
        cpus: set[int] = set()
        for token in value.replace(" ", "").split(","):
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = integer(start_text, "cpu_affinity")
                end = integer(end_text, "cpu_affinity")
                if end < start:
                    raise ValueError("cpu_affinity range is descending")
                cpus.update(range(start, end + 1))
            else:
                cpus.add(integer(token, "cpu_affinity"))
        return cpus
    return None


def resolve_column_inventory(scope_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = (
        scope_path.parent / "feeds" / "source_column_inventory.csv",
        scope_path.parent / "source_column_inventory.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AuditBuildError(
        "source_column_inventory.csv was not found beside the scope; pass "
        "--source-column-inventory"
    )


def validate_compact_collection(
    control_root: Path,
    results_root: Path,
    dataset_ids: list[str],
    findings: Findings,
) -> tuple[dict[str, Any], str, Path]:
    """Rehash the complete compact-transfer whitelist before using its files."""

    control_root = control_root.resolve(strict=True)
    results_root = results_root.resolve(strict=True)
    compact_root = control_root.parent
    findings.check(
        control_root.name == "control"
        and results_root.name == "results"
        and results_root.parent == compact_root,
        "compact_collection",
        "control/results roots do not use the fixed compact layout",
    )
    receipt_path = compact_root / "meta" / "collection_receipt.json"
    receipt = load_json(receipt_path)
    receipt_sha256 = sha256_file(receipt_path)
    findings.check(
        receipt.get("schema") == COLLECTION_RECEIPT_SCHEMA,
        "compact_collection",
        "compact collection schema differs",
    )
    findings.check(
        receipt.get("status") == "passed",
        "compact_collection",
        "compact collection receipt did not pass",
    )
    findings.check(
        receipt.get("dataset_count") == len(dataset_ids),
        "compact_collection",
        "compact collection dataset count differs",
    )
    rows = receipt.get("files", [])
    findings.check(
        isinstance(rows, list) and receipt.get("file_count") == len(rows),
        "compact_collection",
        "compact collection file count differs",
    )
    expected: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                findings.fail(
                    "compact_collection",
                    f"collection file row {index} is not an object",
                )
                continue
            relative_text = row.get("relative_path")
            try:
                relative = PurePosixPath(str(relative_text))
                safe_relative = (
                    isinstance(relative_text, str)
                    and bool(relative_text)
                    and not relative.is_absolute()
                    and ".." not in relative.parts
                    and str(relative) == relative_text
                )
            except (TypeError, ValueError):
                safe_relative = False
            if not safe_relative:
                findings.fail(
                    "compact_collection",
                    f"collection file row {index} has an unsafe relative path",
                )
                continue
            if relative_text in expected:
                findings.fail(
                    "compact_collection",
                    f"duplicate compact file {relative_text}",
                )
                continue
            expected[relative_text] = row
            local_path = compact_root / relative_text
            if not local_path.is_file() or local_path.is_symlink():
                findings.fail(
                    "compact_collection",
                    f"compact file is absent or is a symlink: {relative_text}",
                )
                continue
            findings.check(
                row.get("bytes") == local_path.stat().st_size,
                "compact_collection",
                f"compact file byte count differs: {relative_text}",
            )
            findings.check(
                is_lower_hex_sha(row.get("sha256"))
                and row.get("sha256") == sha256_file(local_path),
                "compact_collection",
                f"compact file SHA differs: {relative_text}",
            )
    findings.check(
        list(expected) == sorted(expected),
        "compact_collection",
        "compact collection files are not in relative-path order",
    )
    actual = {
        path.relative_to(compact_root).as_posix()
        for path in compact_root.rglob("*")
        if path.is_file() and path != receipt_path
    }
    fixed_paths = {
        "control/input_scope_receipt.json",
        "control/final_campaign_binding.json",
        "control/final_goldens_receipt.json",
        "control/source_commit_receipt.json",
        "control/runtime_artifacts/campaign_final_d40_cppf9.json",
        "control/build_receipts/remote_build_receipt.json",
        "results/campaign_run_receipt.json",
        "results/campaign_progress.json",
    }
    dataset_paths = {
        relative
        for dataset in dataset_ids
        for relative in (
            f"control/manifests/{dataset}/aggregates.jsonl",
            f"control/manifests/{dataset}/manifest_receipt.json",
            f"results/by_dataset/{dataset}/cpp.jsonl",
            f"results/by_dataset/{dataset}/java.jsonl",
            f"results/by_dataset/{dataset}/python.jsonl",
            f"results/by_dataset/{dataset}/run_receipt.json",
        )
    }
    allowed = fixed_paths | dataset_paths
    findings.check(
        set(expected) == actual,
        "compact_collection",
        "compact collection whitelist has missing or unlisted files",
    )
    findings.check(
        set(expected) == allowed,
        "compact_collection",
        "compact collection does not match the fixed campaign whitelist",
    )
    return receipt, receipt_sha256, compact_root


def validate_input_scope_receipt(
    control_root: Path,
    scope_sha256: str,
    dataset_count: int,
    findings: Findings,
) -> tuple[dict[str, Any], str]:
    path = control_root / "input_scope_receipt.json"
    receipt = load_json(path)
    digest = sha256_file(path)
    findings.check(
        receipt.get("schema") == CONTROLLER_RECEIPT_SCHEMA
        and receipt.get("stage") == "manifest_batch",
        "input_scope_receipt",
        "input scope receipt schema or stage differs",
    )
    findings.check(
        receipt.get("status") == "passed"
        and receipt.get("source_preflight") == "passed"
        and not receipt.get("failures"),
        "input_scope_receipt",
        "input scope receipt did not close source preflight",
    )
    findings.check(
        receipt.get("scope_sha256") == scope_sha256,
        "input_scope_receipt",
        "input scope receipt scope SHA differs",
    )
    findings.check(
        receipt.get("dataset_count") == dataset_count
        and receipt.get("requested_dataset_count") == dataset_count,
        "input_scope_receipt",
        "input scope receipt dataset count differs",
    )
    return receipt, digest


def validate_campaign_run_receipts(
    results_root: Path,
    scope_sha256: str,
    artifact_sha256: str,
    dataset_ids: list[str],
    findings: Findings,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    run_path = results_root / "campaign_run_receipt.json"
    run_receipt = load_json(run_path)
    run_sha256 = sha256_file(run_path)
    findings.check(
        run_receipt.get("schema") == CONTROLLER_RECEIPT_SCHEMA
        and run_receipt.get("stage") == "campaign_runtime",
        "campaign_run_receipt",
        "campaign run receipt schema or stage differs",
    )
    findings.check(
        run_receipt.get("status") == "passed" and not run_receipt.get("failures"),
        "campaign_run_receipt",
        "campaign run receipt did not pass",
    )
    findings.check(
        run_receipt.get("scope_sha256") == scope_sha256
        and run_receipt.get("runtime_artifact_manifest_sha256") == artifact_sha256,
        "campaign_run_receipt",
        "campaign run receipt scope or artifact SHA differs",
    )
    closed_ids = list(run_receipt.get("skipped_current", [])) + list(
        run_receipt.get("completed", [])
    )
    findings.check(
        run_receipt.get("dataset_count") == len(dataset_ids)
        and len(closed_ids) == len(set(closed_ids))
        and set(closed_ids) == set(dataset_ids),
        "campaign_run_receipt",
        "campaign run receipt does not close the exact dataset scope",
    )

    progress_path = results_root / "campaign_progress.json"
    progress = load_json(progress_path)
    progress_sha256 = sha256_file(progress_path)
    findings.check(
        progress.get("schema") == CONTROLLER_RECEIPT_SCHEMA
        and progress.get("stage") == "campaign_progress",
        "campaign_run_receipt",
        "final campaign progress schema or stage differs",
    )
    findings.check(
        progress.get("status") == "passed"
        and progress.get("pending_count") == 0
        and not progress.get("running")
        and not progress.get("failed"),
        "campaign_run_receipt",
        "final campaign progress is not closed",
    )
    findings.check(
        progress.get("scope_sha256") == scope_sha256
        and progress.get("runtime_artifact_manifest_sha256") == artifact_sha256
        and progress.get("dataset_count") == len(dataset_ids)
        and progress.get("runtime_passed_count") == len(dataset_ids),
        "campaign_run_receipt",
        "final campaign progress scope, artifact, or coverage differs",
    )
    findings.check(
        progress.get("source_preflight_passed_count") == len(dataset_ids)
        and progress.get("remaining_n_a_aggregate_count") == 0,
        "campaign_run_receipt",
        "final campaign progress source-preflight or N/A coverage differs",
    )
    return run_receipt, run_sha256, progress, progress_sha256


def load_column_inventory(
    path: Path, scope_ids: set[str], findings: Findings
) -> tuple[dict[str, dict[str, str]], str]:
    digest = sha256_file(path)
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {
            "dataset_id",
            "aggregate_id",
            "column",
            "source_column",
            "dtype",
            "width",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise AuditBuildError(
                f"column inventory lacks fields {sorted(required)}: {path}"
            )
        for row in reader:
            if row["dataset_id"] not in scope_ids or row["dtype"] not in WIDTH_BY_DTYPE:
                continue
            aggregate_id = row["aggregate_id"]
            if aggregate_id in rows:
                findings.fail(
                    "column_inventory", f"duplicate aggregate_id {aggregate_id}"
                )
            rows[aggregate_id] = row
    return rows, digest


def artifact_summary(manifest: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "sha256": manifest_sha256,
        "controller_source_sha256": manifest.get("controller_source", {}).get("sha256"),
        "cpp_executable_sha256": manifest.get("cpp_executable", {}).get("sha256"),
        "python_codec_source_sha256": manifest.get("python_codec_source", {}).get(
            "sha256"
        ),
        "python_cli_source_sha256": manifest.get("python_cli_source", {}).get("sha256"),
        "python_dependency_source_sha256": {
            row.get("path"): row.get("sha256")
            for row in manifest.get("python_dependency_sources", [])
            if isinstance(row, dict)
        },
        "java_class_count": manifest.get("java_class_count"),
        "java_classes_content_manifest_sha256": manifest.get(
            "java_classes_content_manifest_sha256"
        ),
        "status": manifest.get("status"),
    }


def public_artifact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(summary)
    dependencies = result.get("python_dependency_source_sha256", {})
    if isinstance(dependencies, dict):
        result["python_dependency_source_sha256"] = sorted(dependencies.values())
    return result


def validate_artifact_manifest(
    path: Path, findings: Findings
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    manifest = load_json(path)
    digest = sha256_file(path)
    findings.check(
        manifest.get("schema") == "dbp-float-e2e-runtime-artifacts-v1",
        "artifact_manifest",
        "runtime artifact manifest schema differs",
    )
    findings.check(
        manifest.get("status") == "passed",
        "artifact_manifest",
        "runtime artifact manifest did not pass",
    )
    summary = artifact_summary(manifest, digest)
    for key, value in summary.items():
        if key.endswith("sha256") and key != "python_dependency_source_sha256":
            findings.check(
                is_lower_hex_sha(value),
                "artifact_manifest",
                f"invalid artifact digest {key}",
            )
    return manifest, digest, summary


def validate_campaign_binding(
    control_root: Path,
    expected_binding_sha256: str,
    artifact_sha256: str,
    common_artifact: dict[str, Any],
    scope_sha256: str,
    dataset_count: int,
    findings: Findings,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Validate the caller-pinned final campaign and its cold-decoder receipt."""

    findings.check(
        is_lower_hex_sha(expected_binding_sha256),
        "campaign_binding",
        "caller did not provide a lowercase SHA-256 campaign binding pin",
    )
    binding_path = control_root / "final_campaign_binding.json"
    binding = load_json(binding_path)
    binding_sha256 = sha256_file(binding_path)
    findings.check(
        binding_sha256 == expected_binding_sha256,
        "campaign_binding",
        "final campaign binding SHA differs from the explicit caller pin",
    )
    findings.check(
        binding.get("schema") == CAMPAIGN_BINDING_SCHEMA,
        "campaign_binding",
        "final campaign binding schema differs",
    )
    findings.check(
        binding.get("status") == "passed",
        "campaign_binding",
        "final campaign binding did not pass",
    )

    binding_scope = binding.get("scope", {})
    findings.check(
        binding_scope.get("scope_sha256") == scope_sha256,
        "campaign_binding",
        "campaign binding scope SHA differs",
    )
    findings.check(
        binding_scope.get("dataset_count") == dataset_count,
        "campaign_binding",
        "campaign binding dataset count differs",
    )
    bound_artifact = binding.get("runtime_artifacts", {})
    findings.check(
        bound_artifact.get("status") == "passed"
        and bound_artifact.get("sha256") == artifact_sha256,
        "campaign_binding",
        "campaign binding runtime artifact differs",
    )
    source_commit_binding = binding.get("source_commit", {})
    source_commit_path = control_root / "source_commit_receipt.json"
    source_commit_receipt = load_json(source_commit_path)
    source_commit_sha256 = sha256_file(source_commit_path)
    standalone_sources = source_commit_receipt.get("standalone_source_sha256", {})
    findings.check(
        source_commit_binding.get("status") == "passed"
        and source_commit_binding.get("receipt_sha256") == source_commit_sha256
        and source_commit_binding.get("commit") == source_commit_receipt.get("commit")
        and source_commit_binding.get("branch") == source_commit_receipt.get("branch")
        and source_commit_receipt.get("all_standalone_sources_match_commit") is True
        and isinstance(standalone_sources, dict)
        and bool(standalone_sources)
        and all(
            isinstance(path, str)
            and bool(path)
            and not PurePosixPath(path).is_absolute()
            and ".." not in PurePosixPath(path).parts
            and is_lower_hex_sha(digest)
            for path, digest in standalone_sources.items()
        ),
        "campaign_binding",
        "campaign source-commit receipt differs from the binding",
    )
    build_binding = binding.get("remote_build_reproduction", {})
    build_path = control_root / "build_receipts" / "remote_build_receipt.json"
    build_receipt = load_json(build_path)
    build_sha256 = sha256_file(build_path)
    findings.check(
        build_binding.get("status") == "passed"
        and build_receipt.get("status") == "passed"
        and build_binding.get("sha256") == build_sha256,
        "campaign_binding",
        "campaign remote-build receipt differs from the binding",
    )

    frame_verification = binding.get("frame_verification", {})
    frame_sha256 = frame_verification.get("sha256")
    findings.check(
        frame_verification.get("status") == "passed",
        "frame_verification",
        "campaign frame verification did not pass",
    )
    findings.check(
        frame_verification.get("case_count") == EXPECTED_GOLDEN_CASES,
        "frame_verification",
        f"campaign frame verification must contain {EXPECTED_GOLDEN_CASES} cases",
    )
    findings.check(
        frame_verification.get("cold_decoder_pairs") == EXPECTED_COLD_DECODER_PAIRS,
        "frame_verification",
        "campaign cold-decoder pair count differs",
    )
    findings.check(
        is_lower_hex_sha(frame_sha256),
        "frame_verification",
        "campaign frame verification SHA is invalid",
    )

    # The local name is fixed by the campaign handoff. Never infer a local path
    # from the remote absolute path recorded inside the binding.
    golden_path = control_root / "final_goldens_receipt.json"
    golden = load_json(golden_path)
    golden_sha256 = sha256_file(golden_path)
    findings.check(
        golden_sha256 == frame_sha256,
        "frame_verification",
        "final golden receipt SHA differs from the campaign binding",
    )
    findings.check(
        golden.get("schema") == GOLDEN_RECEIPT_SCHEMA
        and golden.get("stage") == "remote_cross_runtime_goldens",
        "frame_verification",
        "final golden receipt schema or stage differs",
    )
    findings.check(
        golden.get("status") == "passed",
        "frame_verification",
        "final golden receipt did not pass",
    )
    cases = golden.get("cases", [])
    findings.check(
        isinstance(cases, list)
        and len(cases) == EXPECTED_GOLDEN_CASES
        and golden.get("case_count") == EXPECTED_GOLDEN_CASES,
        "frame_verification",
        "final golden receipt case count differs",
    )
    findings.check(
        golden.get("cold_decoder_pairs") == EXPECTED_COLD_DECODER_PAIRS,
        "frame_verification",
        "final golden receipt cold-decoder pair count differs",
    )
    case_names: set[str] = set()
    observed_pairs = 0
    if isinstance(cases, list):
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                findings.fail(
                    "frame_verification", f"golden case {index} is not an object"
                )
                continue
            case_name = str(case.get("case", ""))
            findings.check(
                bool(case_name) and case_name not in case_names,
                "frame_verification",
                f"golden case {index} has an empty or duplicate name",
            )
            case_names.add(case_name)
            findings.check(
                case.get("status") == "passed",
                "frame_verification",
                f"golden case {case_name or index} did not pass",
            )
            findings.check(
                set(case.get("encoders", [])) == set(RUNTIMES)
                and len(case.get("encoders", [])) == len(RUNTIMES),
                "frame_verification",
                f"golden case {case_name or index} encoder set differs",
            )
            findings.check(
                case.get("width") in set(WIDTH_BY_DTYPE.values()),
                "frame_verification",
                f"golden case {case_name or index} width is invalid",
            )
            for digest_key in ("frame_sha256", "input_sha256"):
                findings.check(
                    is_lower_hex_sha(case.get(digest_key)),
                    "frame_verification",
                    f"golden case {case_name or index} {digest_key} is invalid",
                )
            try:
                observed_pairs += integer(
                    case.get("cold_decoder_pairs"), "cold_decoder_pairs"
                )
            except (TypeError, ValueError) as error:
                findings.fail(
                    "frame_verification", f"golden case {case_name or index}: {error}"
                )
    findings.check(
        observed_pairs == EXPECTED_COLD_DECODER_PAIRS,
        "frame_verification",
        "sum of golden cold-decoder pairs differs",
    )
    findings.check(
        receipt_artifact_matches(
            golden.get("common_runtime_artifact_manifest"), common_artifact
        ),
        "frame_verification",
        "golden receipt runtime artifact differs from the pinned artifact",
    )
    findings.check(
        golden.get("runtime_artifact_identities_before_after", {}).get("status")
        == "passed",
        "frame_verification",
        "golden runtime artifact identity gate did not pass",
    )
    return binding, binding_sha256, golden, golden_sha256


def load_environment(path: Path, findings: Findings) -> tuple[dict[str, Any], str]:
    source = load_json(path)
    digest = sha256_file(path)
    findings.check(
        source.get("schema") == "dbp-float-e2e-environment-v1",
        "environment",
        "environment receipt schema differs",
    )
    findings.check(
        str(source.get("status", "")).lower() in {"passed", "complete", "ok"},
        "environment",
        "environment receipt is not successful",
    )
    public_keys = (
        "hardware",
        "cpu",
        "os",
        "load",
        "clock",
        "cpu_affinity",
        "concurrent_load",
        "shared_server",
        "captured_at",
        "runtime_versions",
        "notes",
        "repetitions",
        "repeats",
        "warmup_blocks",
        "runtime_artifact_manifest_sha256",
    )
    public = {key: source.get(key) for key in public_keys if key in source}
    source_versions = source.get("runtime_versions", {})
    if isinstance(source_versions, dict):
        public["runtime_versions"] = {
            runtime: runtime_version_text(source_versions, runtime)
            for runtime in RUNTIMES
        }
    public["environment_receipt_sha256"] = digest
    findings.check(
        bool(public.get("hardware") or public.get("cpu")),
        "environment",
        "hardware/CPU metadata is absent",
    )
    findings.check(bool(public.get("clock")), "environment", "clock metadata is absent")
    findings.check(
        public.get("concurrent_load", public.get("load")) is not None,
        "environment",
        "concurrent-load metadata is absent",
    )
    findings.check(
        public.get("shared_server") is True,
        "environment",
        "shared_server must be explicitly true",
    )
    findings.check(
        public.get("cpu_affinity") is not None,
        "environment",
        "CPU affinity metadata is absent",
    )
    findings.check(
        bool(public.get("captured_at")),
        "environment",
        "environment capture time is absent",
    )
    findings.check(
        public.get("repetitions", public.get("repeats")) is not None,
        "environment",
        "repetition count is absent",
    )
    findings.check(
        public.get("warmup_blocks") is not None,
        "environment",
        "warmup block count is absent",
    )
    cpu = source.get("cpu", {})
    findings.check(
        isinstance(cpu, dict)
        and cpu.get("passed_receipts_at_capture") == EXPECTED_SCOPE_DATASETS,
        "environment",
        "environment receipt was not captured after all dataset receipts passed",
    )
    return public, digest


def runtime_version_text(versions: dict[str, Any], runtime: str) -> str | None:
    key = "g++" if runtime == "cpp" else runtime
    value = versions.get(runtime, versions.get(key))
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, dict):
        return None
    for stream in ("stdout", "stderr"):
        text = value.get(stream)
        if isinstance(text, str) and text.strip():
            return text.strip().splitlines()[0]
    return None


def receipt_artifact_matches(observed: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(observed, dict):
        return False
    return all(
        observed.get(key) == value for key, value in expected.items() if key != "path"
    )


def runtime_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: collections.Counter[str] = collections.Counter()
    for row in rows:
        for field in NUMERIC_RESULT_FIELDS:
            totals[field] += integer(row[field], field)
    return dict(totals)


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    def safe(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: safe(row.get(field)) for field in CSV_FIELDS})
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def build_payload(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    findings = Findings()
    scope_path = arguments.scope.resolve(strict=True)
    scope = load_json(scope_path)
    scope_sha256 = sha256_file(scope_path)
    dataset_ids = list(scope.get("dataset_ids", []))
    findings.check(
        scope.get("schema") == FROZEN_SCOPE_SCHEMA
        and scope.get("scope_id") == FROZEN_SCOPE_ID,
        "scope",
        "frozen scope schema or scope ID differs",
    )
    findings.check(
        len(dataset_ids) == EXPECTED_SCOPE_DATASETS,
        "scope",
        f"scope has {len(dataset_ids)} datasets; expected {EXPECTED_SCOPE_DATASETS}",
    )
    findings.check(
        len(set(dataset_ids)) == len(dataset_ids),
        "scope",
        "scope dataset IDs are not unique",
    )
    findings.check(
        scope.get("dataset_identity_count") == len(dataset_ids),
        "scope",
        "frozen scope identity count differs from dataset IDs",
    )
    findings.check(
        all(
            isinstance(dataset, str)
            and bool(dataset)
            and PurePosixPath(dataset).name == dataset
            for dataset in dataset_ids
        ),
        "scope",
        "scope contains an unsafe dataset ID",
    )
    collection_receipt, collection_sha256, compact_root = validate_compact_collection(
        arguments.control_root,
        arguments.results_root,
        dataset_ids,
        findings,
    )
    input_scope_receipt, input_scope_receipt_sha256 = validate_input_scope_receipt(
        arguments.control_root, scope_sha256, len(dataset_ids), findings
    )
    scope_ids = set(dataset_ids)
    inventory_path = resolve_column_inventory(
        scope_path, arguments.source_column_inventory
    )
    inventory, inventory_sha256 = load_column_inventory(
        inventory_path, scope_ids, findings
    )
    artifact_manifest, artifact_sha256, common_artifact = validate_artifact_manifest(
        arguments.artifact_manifest.resolve(strict=True), findings
    )
    findings.check(
        arguments.artifact_manifest.resolve()
        == compact_root / "control/runtime_artifacts/campaign_final_d40_cppf9.json",
        "artifact_manifest",
        "artifact manifest is not the fixed compact campaign artifact",
    )
    (
        campaign_binding,
        campaign_binding_sha256,
        golden_receipt,
        golden_receipt_sha256,
    ) = validate_campaign_binding(
        arguments.control_root.resolve(strict=True),
        arguments.campaign_binding_sha256,
        artifact_sha256,
        common_artifact,
        scope_sha256,
        len(dataset_ids),
        findings,
    )
    (
        campaign_run_receipt,
        campaign_run_receipt_sha256,
        campaign_progress,
        campaign_progress_sha256,
    ) = validate_campaign_run_receipts(
        arguments.results_root,
        scope_sha256,
        artifact_sha256,
        dataset_ids,
        findings,
    )
    environment, environment_sha256 = load_environment(
        arguments.environment.resolve(strict=True), findings
    )
    findings.check(
        environment.get("runtime_artifact_manifest_sha256") == artifact_sha256,
        "environment",
        "environment receipt runtime artifact SHA differs",
    )

    source_aggregates: dict[str, dict[str, Any]] = {}
    manifest_info: dict[str, dict[str, Any]] = {}
    authority_projections: list[dict[str, Any]] = []
    source_chain = hashlib.sha256()
    input_files: list[dict[str, Any]] = []

    for dataset in dataset_ids:
        directory = arguments.control_root / "manifests" / dataset
        receipt_path = directory / "manifest_receipt.json"
        aggregates_path = directory / "aggregates.jsonl"
        blocks_path = directory / "blocks.tsv"
        if not all(path.is_file() for path in (receipt_path, aggregates_path)):
            findings.fail(
                "source_manifest_coverage",
                "source manifest files are incomplete",
                dataset,
            )
            continue
        try:
            receipt = load_json(receipt_path)
            receipt_sha = sha256_file(receipt_path)
            aggregates_sha = sha256_file(aggregates_path)
            recorded_blocks_sha = receipt.get("blocks_tsv", {}).get("sha256")
            blocks_sha = (
                sha256_file(blocks_path)
                if blocks_path.is_file()
                else recorded_blocks_sha
            )
            rows = read_jsonl(aggregates_path)
        except (OSError, ValueError, AuditBuildError, json.JSONDecodeError) as error:
            findings.fail("source_manifest_coverage", repr(error), dataset)
            continue
        source_chain.update(bytes.fromhex(receipt_sha))
        source_chain.update(bytes.fromhex(aggregates_sha))
        if is_lower_hex_sha(blocks_sha):
            source_chain.update(bytes.fromhex(blocks_sha))
        else:
            findings.fail(
                "source_manifest",
                "block manifest receipt SHA is invalid",
                dataset,
            )
        input_files.append(
            {
                "dataset": dataset,
                "manifest_receipt_sha256": receipt_sha,
                "aggregates_sha256": aggregates_sha,
                "blocks_sha256": blocks_sha,
                "blocks_locally_rehashed": blocks_path.is_file(),
            }
        )
        findings.check(
            receipt.get("dataset_id") == dataset,
            "source_manifest",
            "dataset ID differs",
            dataset,
        )
        findings.check(
            receipt.get("schema") == CONTROLLER_RECEIPT_SCHEMA
            and receipt.get("stage") == "dataset_manifest",
            "source_manifest",
            "manifest receipt schema or stage differs",
            dataset,
        )
        findings.check(
            receipt.get("status") == "passed",
            "source_manifest",
            "manifest receipt failed",
            dataset,
        )
        findings.check(
            receipt.get("scope_sha256") == scope_sha256,
            "source_manifest",
            "scope SHA differs",
            dataset,
        )
        findings.check(
            receipt.get("source_preflight", {}).get("status") == "passed",
            "source_preflight",
            "source preflight did not pass",
            dataset,
        )
        findings.check(
            receipt.get("aggregates_jsonl", {}).get("sha256") == aggregates_sha,
            "source_manifest",
            "aggregate manifest SHA differs",
            dataset,
        )
        findings.check(
            receipt.get("blocks_tsv", {}).get("sha256") == blocks_sha,
            "source_manifest",
            "block manifest SHA differs",
            dataset,
        )
        for digest_key in (
            "source_block_catalog_sha256",
            "column_statistics_sha256",
            "dataset_manifest_sha256",
            "ready_sha256",
        ):
            findings.check(
                is_lower_hex_sha(receipt.get(digest_key)),
                "source_manifest",
                f"manifest receipt {digest_key} is invalid",
                dataset,
            )
        findings.check(
            integer(receipt.get("float_aggregate_count", -1), "float_aggregate_count")
            == len(rows),
            "source_manifest",
            "float aggregate count differs",
            dataset,
        )
        try:
            manifest_blocks = sum(integer(row.get("blocks"), "blocks") for row in rows)
            manifest_segment_bytes = sum(
                integer(row.get("original_bytes"), "original_bytes")
                + integer(row.get("validity_bytes"), "validity_bytes")
                for row in rows
            )
            preflight = receipt.get("source_preflight", {})
            findings.check(
                integer(receipt.get("float_block_records"), "float_block_records")
                == manifest_blocks
                and integer(
                    preflight.get("verified_block_count"), "verified_block_count"
                )
                == manifest_blocks
                and integer(
                    preflight.get("verified_segment_count"), "verified_segment_count"
                )
                == 2 * manifest_blocks
                and integer(
                    preflight.get("verified_segment_bytes"), "verified_segment_bytes"
                )
                == manifest_segment_bytes,
                "source_preflight",
                "source preflight block/segment accounting differs from aggregates",
                dataset,
            )
            findings.check(
                integer(
                    receipt.get("source_block_catalog_records"),
                    "source_block_catalog_records",
                )
                >= manifest_blocks,
                "source_manifest",
                "source catalog record count is smaller than float block coverage",
                dataset,
            )
        except (TypeError, ValueError) as error:
            findings.fail("source_preflight", str(error), dataset)
        authority = receipt.get("input_authority", {})
        authority_projection = {
            key: value for key, value in authority.items() if key.endswith("_sha256")
        }
        findings.check(
            authority_projection.get("source_scope_sha256")
            == scope.get("source_scope_sha256")
            and authority_projection.get("input_inventory_sha256")
            == scope.get("benchmarkhost_input_inventory_sha256"),
            "source_authority",
            "source authority differs from the frozen paper scope",
            dataset,
        )
        findings.check(
            bool(authority_projection)
            and all(is_lower_hex_sha(value) for value in authority_projection.values()),
            "source_authority",
            "source authority hashes are absent or invalid",
            dataset,
        )
        authority_projections.append(authority_projection)
        seen: set[str] = set()
        for row in rows:
            aggregate_id = str(row.get("aggregate_id", ""))
            if (
                not aggregate_id
                or aggregate_id in seen
                or aggregate_id in source_aggregates
            ):
                findings.fail(
                    "source_aggregate_uniqueness",
                    f"duplicate/empty aggregate ID {aggregate_id!r}",
                    dataset,
                )
                continue
            seen.add(aggregate_id)
            dtype = str(row.get("dtype", ""))
            width = WIDTH_BY_DTYPE.get(dtype)
            try:
                observed_width = integer(row.get("width"), "width")
                valid_count = integer(row.get("valid_count"), "valid_count")
                original_bytes = integer(row.get("original_bytes"), "original_bytes")
                blocks = integer(row.get("blocks"), "blocks")
                header_bytes = integer(row.get("header_bytes"), "header_bytes")
                rows_count = integer(row.get("rows"), "rows")
                missing_count = integer(row.get("missing_count"), "missing_count")
            except (TypeError, ValueError) as error:
                findings.fail(
                    "source_aggregate_shape", f"{aggregate_id}: {error}", dataset
                )
                continue
            findings.check(
                width is not None and observed_width == width,
                "dtype_width",
                f"{aggregate_id}: dtype/width mismatch",
                dataset,
            )
            findings.check(
                original_bytes == observed_width * valid_count,
                "native_bytes",
                f"{aggregate_id}: native bytes mismatch",
                dataset,
            )
            findings.check(
                header_bytes == HEADER_BYTES_PER_BLOCK * blocks,
                "header_bytes",
                f"{aggregate_id}: header bytes mismatch",
                dataset,
            )
            findings.check(
                valid_count + missing_count == rows_count,
                "source_counts",
                f"{aggregate_id}: row counts mismatch",
                dataset,
            )
            findings.check(
                row.get("schema") == AGGREGATE_MANIFEST_SCHEMA,
                "source_aggregate_shape",
                f"{aggregate_id}: aggregate schema differs",
                dataset,
            )
            findings.check(
                row.get("status") == ("ready" if valid_count else "n_a"),
                "source_aggregate_shape",
                f"{aggregate_id}: aggregate status differs",
                dataset,
            )
            for digest_key in (
                "expected_sequence_chain_sha256",
                "manifest_expected_sequence_chain_sha256",
            ):
                findings.check(
                    is_lower_hex_sha(row.get(digest_key)),
                    "source_aggregate_shape",
                    f"{aggregate_id}: invalid {digest_key}",
                    dataset,
                )
            mapping = inventory.get(aggregate_id)
            findings.check(
                mapping is not None,
                "column_inventory",
                f"{aggregate_id}: column mapping absent",
                dataset,
            )
            if mapping is not None:
                findings.check(
                    mapping["dataset_id"] == dataset,
                    "column_inventory",
                    f"{aggregate_id}: mapped dataset differs",
                    dataset,
                )
                findings.check(
                    mapping["dtype"] == dtype,
                    "column_inventory",
                    f"{aggregate_id}: mapped dtype differs",
                    dataset,
                )
                findings.check(
                    int(mapping["width"]) == observed_width,
                    "column_inventory",
                    f"{aggregate_id}: mapped width differs",
                    dataset,
                )
                statistics_sha = mapping.get("statistics_sha256")
                if statistics_sha:
                    findings.check(
                        statistics_sha == receipt.get("column_statistics_sha256"),
                        "column_inventory",
                        f"{aggregate_id}: statistics SHA differs",
                        dataset,
                    )
            source_aggregates[aggregate_id] = {**row, "_dataset": dataset}
        manifest_info[dataset] = {
            "receipt": receipt,
            "receipt_sha256": receipt_sha,
            "aggregates_sha256": aggregates_sha,
            "blocks_sha256": blocks_sha,
            "aggregate_ids": seen,
        }

    findings.check(
        len(authority_projections) == len(dataset_ids)
        and len({canonical_json_bytes(row) for row in authority_projections}) == 1,
        "source_authority",
        "source authority hashes are absent or differ across datasets",
    )
    input_scope_authority = {
        key: value
        for key, value in input_scope_receipt.get("input_authority", {}).items()
        if key.endswith("_sha256")
    }
    findings.check(
        bool(authority_projections)
        and input_scope_authority == authority_projections[0],
        "source_authority",
        "input scope and per-dataset source authority hashes differ",
    )
    findings.check(
        set(source_aggregates) == set(inventory),
        "column_inventory",
        "float aggregate IDs differ between manifests and source-column inventory",
    )

    dataset_receipts: list[dict[str, Any]] = []
    columns: list[dict[str, Any]] = []
    result_chains = {runtime: hashlib.sha256() for runtime in RUNTIMES}
    runtime_versions = environment.get("runtime_versions", {})
    observed_repeats: set[int] = set()
    observed_warmups: set[int] = set()
    observed_cpu_affinity: set[int] = set()
    runtime_result_file_count = 0
    complete_dataset_status: dict[str, str] = {}
    run_receipt_hashes: dict[str, str] = {}

    for dataset in dataset_ids:
        start_errors = len(findings.errors) + findings.suppressed_errors
        source = manifest_info.get(dataset)
        result_directory = arguments.results_root / "by_dataset" / dataset
        run_receipt_path = result_directory / "run_receipt.json"
        if source is None:
            for runtime in RUNTIMES:
                dataset_receipts.append(
                    {
                        "dataset": dataset,
                        "runtime": runtime,
                        "status": "pending",
                        "reason": "Source manifest is unavailable.",
                    }
                )
            continue
        if not run_receipt_path.is_file():
            findings.fail(
                "dataset_runtime_receipts", "run_receipt.json is absent", dataset
            )
            for runtime in RUNTIMES:
                dataset_receipts.append(
                    {
                        "dataset": dataset,
                        "runtime": runtime,
                        "status": "pending",
                        "reason": "Runtime receipt is unavailable.",
                    }
                )
            continue
        try:
            run_receipt = load_json(run_receipt_path)
            run_receipt_sha = sha256_file(run_receipt_path)
        except (OSError, AuditBuildError, json.JSONDecodeError) as error:
            findings.fail("dataset_runtime_receipts", repr(error), dataset)
            for runtime in RUNTIMES:
                dataset_receipts.append(
                    {
                        "dataset": dataset,
                        "runtime": runtime,
                        "status": "failed",
                        "reason": "Runtime receipt is invalid.",
                    }
                )
            continue
        run_receipt_hashes[dataset] = run_receipt_sha
        findings.check(
            run_receipt.get("dataset_id") == dataset,
            "dataset_runtime_receipts",
            "run receipt dataset differs",
            dataset,
        )
        findings.check(
            run_receipt.get("schema") == CONTROLLER_RECEIPT_SCHEMA
            and run_receipt.get("stage") == "dataset_runtime",
            "dataset_runtime_receipts",
            "run receipt schema or stage differs",
            dataset,
        )
        findings.check(
            run_receipt.get("status") == "passed",
            "dataset_runtime_receipts",
            "run receipt failed",
            dataset,
        )
        findings.check(
            run_receipt.get("source_preflight_status") == "passed",
            "source_preflight",
            "runtime receipt lacks passed preflight",
            dataset,
        )
        findings.check(
            run_receipt.get("cross_runtime_exact_status") == "passed",
            "cross_runtime",
            "controller cross-runtime gate failed",
            dataset,
        )
        findings.check(
            run_receipt.get("cross_runtime_exact_aggregate_count")
            == len(source["aggregate_ids"]),
            "cross_runtime",
            "controller cross-runtime aggregate count differs",
            dataset,
        )
        findings.check(
            run_receipt.get("source_manifest_receipt_sha256")
            == source["receipt_sha256"],
            "dataset_runtime_receipts",
            "run receipt does not bind local source manifest receipt",
            dataset,
        )
        findings.check(
            receipt_artifact_matches(
                run_receipt.get("common_runtime_artifact_manifest"), common_artifact
            ),
            "artifact_manifest",
            "run receipt does not bind the common artifact manifest",
            dataset,
        )
        findings.check(
            run_receipt.get("runtime_artifact_identities_before_after", {}).get(
                "status"
            )
            == "passed",
            "artifact_manifest",
            "runtime artifact before/after identity gate failed",
            dataset,
        )
        findings.check(
            run_receipt.get("source_file_identities_before_after", {}).get("status")
            == "passed",
            "source_preflight",
            "source before/after identity gate failed",
            dataset,
        )
        findings.check(
            run_receipt.get("control_input_identities_before_after", {}).get("status")
            == "passed",
            "source_manifest",
            "control-input identity gate failed",
            dataset,
        )
        control_hashes = set(
            run_receipt.get("control_input_identities_before_after", {})
            .get("sha256", {})
            .values()
        )
        findings.check(
            {source["aggregates_sha256"], source["blocks_sha256"]}.issubset(
                control_hashes
            ),
            "source_manifest",
            "run receipt does not bind both local control manifests",
            dataset,
        )
        exact_fields = set(run_receipt.get("cross_runtime_exact_fields", []))
        findings.check(
            set(EXACT_CROSS_RUNTIME_FIELDS).issubset(exact_fields),
            "cross_runtime",
            "controller exact fields omit the frame sequence or another required field",
            dataset,
        )
        runtime_receipt_rows = run_receipt.get("runtimes", [])
        runtime_receipts = {
            str(row.get("runtime")): row
            for row in runtime_receipt_rows
            if isinstance(row, dict)
        }
        findings.check(
            len(runtime_receipts) == len(runtime_receipt_rows)
            and set(runtime_receipts) == set(RUNTIMES),
            "runtime_coverage",
            "runtime receipts are not exactly cpp/java/python",
            dataset,
        )
        results_by_runtime: dict[str, dict[str, dict[str, Any]]] = {}
        for runtime in RUNTIMES:
            result_path = result_directory / f"{runtime}.jsonl"
            runtime_receipt = runtime_receipts.get(runtime)
            if runtime_receipt is None or not result_path.is_file():
                findings.fail(
                    "runtime_coverage", f"{runtime} result/receipt is absent", dataset
                )
                continue
            try:
                result_sha = sha256_file(result_path)
                rows = read_jsonl(result_path)
            except (OSError, AuditBuildError, json.JSONDecodeError) as error:
                findings.fail("runtime_coverage", f"{runtime}: {error!r}", dataset)
                continue
            runtime_result_file_count += 1
            result_chains[runtime].update(bytes.fromhex(result_sha))
            findings.check(
                runtime_receipt.get("status") == "passed",
                "runtime_coverage",
                f"{runtime} receipt failed",
                dataset,
            )
            findings.check(
                runtime_receipt.get("returncode", 0) == 0,
                "runtime_coverage",
                f"{runtime} process return code differs",
                dataset,
            )
            raw_result_sha = runtime_receipt.get("raw_runtime_result_sha256")
            if raw_result_sha is not None:
                findings.check(
                    is_lower_hex_sha(raw_result_sha),
                    "runtime_coverage",
                    f"{runtime} raw result SHA is invalid",
                    dataset,
                )
            findings.check(
                runtime_receipt.get("result_sha256") == result_sha,
                "runtime_coverage",
                f"{runtime} result SHA differs",
                dataset,
            )
            findings.check(
                integer(runtime_receipt.get("aggregate_count", -1), "aggregate_count")
                == len(rows),
                "runtime_coverage",
                f"{runtime} aggregate count differs",
                dataset,
            )
            try:
                observed_repeats.add(integer(runtime_receipt.get("repeats"), "repeats"))
                for cpu in runtime_receipt.get("cpu_affinity") or ():
                    observed_cpu_affinity.add(integer(cpu, "cpu_affinity"))
                match = re.search(
                    r"first_min_(\d+)_blocks",
                    str(runtime_receipt.get("warmup_policy", "")),
                )
                if match:
                    observed_warmups.add(int(match.group(1)))
                else:
                    findings.fail(
                        "environment",
                        f"{runtime} warmup policy is not parseable",
                        dataset,
                    )
            except (TypeError, ValueError) as error:
                findings.fail("environment", f"{runtime}: {error}", dataset)
            row_map = {str(row.get("aggregate_id", "")): row for row in rows}
            findings.check(
                len(row_map) == len(rows),
                "runtime_coverage",
                f"{runtime} duplicate aggregate result",
                dataset,
            )
            findings.check(
                set(row_map) == source["aggregate_ids"],
                "runtime_coverage",
                f"{runtime} aggregate coverage differs",
                dataset,
            )
            results_by_runtime[runtime] = row_map
            if isinstance(runtime_receipt.get("totals"), dict):
                try:
                    findings.check(
                        runtime_totals(rows)
                        == {
                            key: integer(value, key)
                            for key, value in runtime_receipt["totals"].items()
                        },
                        "runtime_totals",
                        f"{runtime} totals differ",
                        dataset,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    findings.fail("runtime_totals", f"{runtime}: {error}", dataset)

        for aggregate_id in sorted(source["aggregate_ids"]):
            reference = source_aggregates[aggregate_id]
            runtime_rows = {
                runtime: results_by_runtime.get(runtime, {}).get(aggregate_id)
                for runtime in RUNTIMES
            }
            if any(row is None for row in runtime_rows.values()):
                continue
            for runtime, row in runtime_rows.items():
                assert row is not None
                try:
                    for field in NUMERIC_RESULT_FIELDS:
                        integer(row.get(field), field)
                    for field in EXACT_SOURCE_FIELDS:
                        findings.check(
                            integer(row[field], field)
                            == integer(reference[field], field),
                            "source_result_exactness",
                            f"{runtime}/{aggregate_id}: {field} differs",
                            dataset,
                        )
                    findings.check(
                        row.get("status")
                        == (
                            "ok"
                            if integer(reference["valid_count"], "valid_count")
                            else "n_a"
                        ),
                        "source_result_exactness",
                        f"{runtime}/{aggregate_id}: status differs",
                        dataset,
                    )
                    findings.check(
                        integer(row["verified_values"], "verified_values")
                        == integer(reference["valid_count"], "valid_count"),
                        "bitwise_verification",
                        f"{runtime}/{aggregate_id}: verified count differs",
                        dataset,
                    )
                    findings.check(
                        integer(row["header_bytes"], "header_bytes")
                        == HEADER_BYTES_PER_BLOCK * integer(row["blocks"], "blocks"),
                        "header_bytes",
                        f"{runtime}/{aggregate_id}: header count differs",
                        dataset,
                    )
                    findings.check(
                        integer(row["scaled_blocks"], "scaled_blocks")
                        + integer(row["raw_blocks"], "raw_blocks")
                        == integer(row["blocks"], "blocks"),
                        "mode_counts",
                        f"{runtime}/{aggregate_id}: mode counts differ",
                        dataset,
                    )
                    if integer(reference["valid_count"], "valid_count") == 0:
                        blocks = integer(row["blocks"], "blocks")
                        rows_count = integer(row["rows"], "rows")
                        compressed = integer(
                            row["compressed_bytes"], "compressed_bytes"
                        )
                        scaled = integer(row["scaled_blocks"], "scaled_blocks")
                        raw = integer(row["raw_blocks"], "raw_blocks")
                        if blocks:
                            findings.check(
                                rows_count > 0
                                and compressed == HEADER_BYTES_PER_BLOCK * blocks
                                and scaled == 0
                                and raw == blocks,
                                "n_a_semantics",
                                f"{runtime}/{aggregate_id}: all-null blocks are not measured empty raw frames",
                                dataset,
                            )
                        else:
                            findings.check(
                                rows_count == 0
                                and compressed == 0
                                and scaled == 0
                                and raw == 0
                                and row.get("frame_sequence_sha256")
                                == hashlib.sha256().hexdigest(),
                                "n_a_semantics",
                                f"{runtime}/{aggregate_id}: zero-row aggregate is not canonical",
                                dataset,
                            )
                            findings.check(
                                row.get("result_origin")
                                == "controller_synthesized_zero_block_schema_aggregate",
                                "n_a_semantics",
                                f"{runtime}/{aggregate_id}: zero-row result origin differs",
                                dataset,
                            )
                    findings.check(
                        is_lower_hex_sha(row.get("frame_sequence_sha256")),
                        "frame_sequence_sha256",
                        f"{runtime}/{aggregate_id}: invalid frame hash",
                        dataset,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    findings.fail(
                        "source_result_exactness",
                        f"{runtime}/{aggregate_id}: {error}",
                        dataset,
                    )
            baseline = runtime_rows["cpp"]
            assert baseline is not None
            for runtime in ("java", "python"):
                observed = runtime_rows[runtime]
                assert observed is not None
                for field in EXACT_CROSS_RUNTIME_FIELDS:
                    findings.check(
                        observed.get(field) == baseline.get(field),
                        "cross_runtime",
                        f"{aggregate_id}/{field}: cpp != {runtime}",
                        dataset,
                    )
            mapping = inventory.get(aggregate_id, {})
            for runtime in RUNTIMES:
                row = runtime_rows[runtime]
                assert row is not None
                valid = integer(row["valid_count"], "valid_count")
                native = integer(row["original_bytes"], "original_bytes")
                frame = integer(row["compressed_bytes"], "compressed_bytes")
                encode_ns = integer(row["encode_ns"], "encode_ns")
                decode_ns = integer(row["decode_ns"], "decode_ns")
                result_sha = runtime_receipts[runtime]["result_sha256"]
                columns.append(
                    {
                        "status": row["status"],
                        "runtime": runtime,
                        "dataset": dataset,
                        "column": mapping.get("source_column")
                        or mapping.get("column")
                        or aggregate_id,
                        "source_column": mapping.get("source_column")
                        or mapping.get("column")
                        or aggregate_id,
                        "aggregate_id": aggregate_id,
                        "dtype": reference["dtype"],
                        "width": integer(reference["width"], "width"),
                        "max_point": integer(reference["max_point"], "max_point"),
                        "rows": integer(row["rows"], "rows"),
                        "valid_count": valid,
                        "missing_count": integer(row["missing_count"], "missing_count"),
                        "native_bytes": native,
                        "frame_bytes": frame,
                        "validity_bytes": integer(
                            row["validity_bytes"], "validity_bytes"
                        ),
                        "header_bytes": integer(row["header_bytes"], "header_bytes"),
                        "compression_rate": frame / native if native else None,
                        "encode_ns": encode_ns,
                        "decode_ns": decode_ns,
                        "encode_ns_per_value": encode_ns / valid if valid else None,
                        "decode_ns_per_value": decode_ns / valid if valid else None,
                        "blocks": integer(row["blocks"], "blocks"),
                        "scaled_blocks": integer(row["scaled_blocks"], "scaled_blocks"),
                        "raw_blocks": integer(row["raw_blocks"], "raw_blocks"),
                        "verified_values": integer(
                            row["verified_values"], "verified_values"
                        ),
                        "frame_sequence_sha256": row["frame_sequence_sha256"],
                        "expected_sequence_chain_sha256": reference.get(
                            "expected_sequence_chain_sha256"
                        ),
                        "result_sha256": result_sha,
                        "receipt_sha256": run_receipt_sha,
                        "source_manifest_receipt_sha256": source["receipt_sha256"],
                        "source_aggregates_sha256": source["aggregates_sha256"],
                        "source_blocks_sha256": source["blocks_sha256"],
                        "source_block_catalog_sha256": source["receipt"].get(
                            "source_block_catalog_sha256"
                        ),
                        "runtime_artifact_manifest_sha256": artifact_sha256,
                        "source_column_inventory_sha256": inventory_sha256,
                        "scope_sha256": scope_sha256,
                        "message": "",
                    }
                )

        end_errors = len(findings.errors) + findings.suppressed_errors
        source_rows = [source_aggregates[item] for item in source["aggregate_ids"]]
        nonempty = sum(
            integer(row["valid_count"], "valid_count") > 0 for row in source_rows
        )
        n_a = len(source_rows) - nonempty
        if end_errors == start_errors and set(results_by_runtime) == set(RUNTIMES):
            dataset_status = "ok" if nonempty else "n_a"
            complete_dataset_status[dataset] = dataset_status
            reason = (
                ""
                if nonempty
                else (
                    "No float32/float64 aggregate in the frozen schema."
                    if not source_rows
                    else "Float schema has no nonempty valid values."
                )
            )
        else:
            dataset_status = "failed"
            reason = "One or more evidence gates failed."
        for runtime in RUNTIMES:
            dataset_receipts.append(
                {
                    "dataset": dataset,
                    "runtime": runtime,
                    "status": dataset_status,
                    "reason": reason,
                    "receipt_sha256": run_receipt_sha,
                    "schema_aggregate_count": len(source_rows),
                    "nonempty_aggregate_count": nonempty,
                    "n_a_aggregate_count": n_a,
                }
            )

    findings.check(
        len(complete_dataset_status) == len(dataset_ids),
        "dataset_runtime_receipts",
        "not every scope dataset has a closed three-runtime receipt",
    )
    findings.check(
        len(columns) == len(source_aggregates) * len(RUNTIMES),
        "runtime_coverage",
        "column result rows do not equal source aggregates times runtimes",
    )
    findings.check(
        len(observed_repeats) == 1, "environment", "runtime repetitions differ"
    )
    findings.check(
        len(observed_warmups) == 1, "environment", "runtime warmup policies differ"
    )
    declared_affinity = parse_cpu_affinity(environment.get("cpu_affinity"))
    if declared_affinity is not None:
        findings.check(
            declared_affinity == observed_cpu_affinity,
            "environment",
            "environment CPU affinity differs from runtime receipts",
        )
    else:
        findings.fail(
            "environment", "environment CPU affinity must be an explicit list"
        )

    declared_repeats = environment.get("repetitions", environment.get("repeats"))
    declared_warmup = environment.get("warmup_blocks")
    repeats = next(
        iter(observed_repeats),
        int(declared_repeats) if declared_repeats is not None else None,
    )
    warmup_blocks = next(
        iter(observed_warmups),
        int(declared_warmup) if declared_warmup is not None else None,
    )
    if declared_repeats is not None:
        findings.check(
            int(declared_repeats) == repeats,
            "environment",
            "declared repetitions differ",
        )
    if declared_warmup is not None:
        findings.check(
            int(declared_warmup) == warmup_blocks,
            "environment",
            "declared warmup differs",
        )

    measured_dataset_count = sum(
        status == "ok" for status in complete_dataset_status.values()
    )
    n_a_dataset_count = sum(
        status == "n_a" for status in complete_dataset_status.values()
    )
    nonempty_aggregate_count = sum(
        integer(row["valid_count"], "valid_count") > 0
        for row in source_aggregates.values()
    )
    n_a_aggregate_count = len(source_aggregates) - nonempty_aggregate_count
    native_values_bytes = sum(
        integer(row["original_bytes"], "original_bytes")
        for row in source_aggregates.values()
    )
    source_totals = {
        "float_block_records": sum(
            integer(row["blocks"], "blocks") for row in source_aggregates.values()
        ),
        "row_count": sum(
            integer(row["rows"], "rows") for row in source_aggregates.values()
        ),
        "valid_count": sum(
            integer(row["valid_count"], "valid_count")
            for row in source_aggregates.values()
        ),
        "missing_count": sum(
            integer(row["missing_count"], "missing_count")
            for row in source_aggregates.values()
        ),
        "validity_bytes": sum(
            integer(row["validity_bytes"], "validity_bytes")
            for row in source_aggregates.values()
        ),
        "header_bytes": sum(
            integer(row["header_bytes"], "header_bytes")
            for row in source_aggregates.values()
        ),
        "original_bytes": native_values_bytes,
    }
    float32_aggregates = [
        row for row in source_aggregates.values() if row.get("dtype") == "float32"
    ]
    float64_aggregates = [
        row for row in source_aggregates.values() if row.get("dtype") == "float64"
    ]
    float_schema_dataset_count = sum(
        bool(source["aggregate_ids"]) for source in manifest_info.values()
    )
    zero_row_aggregate_count = sum(
        integer(row["rows"], "rows") == 0 for row in source_aggregates.values()
    )
    all_null_nonempty_aggregate_count = sum(
        integer(row["valid_count"], "valid_count") == 0
        and integer(row["rows"], "rows") > 0
        for row in source_aggregates.values()
    )
    summary = {
        "dataset_receipt_count": len(dataset_ids),
        "measured_dataset_count": measured_dataset_count,
        "n_a_dataset_count": n_a_dataset_count,
        "schema_aggregate_count": len(source_aggregates),
        "nonempty_aggregate_count": nonempty_aggregate_count,
        "n_a_aggregate_count": n_a_aggregate_count,
        "datasets_with_float_schema_count": float_schema_dataset_count,
        "no_float_schema_dataset_count": len(dataset_ids) - float_schema_dataset_count,
        "float32_aggregate_count": len(float32_aggregates),
        "float64_aggregate_count": len(float64_aggregates),
        "all_null_nonempty_aggregate_count": all_null_nonempty_aggregate_count,
        "zero_row_aggregate_count": zero_row_aggregate_count,
        "valid_value_count": source_totals["valid_count"],
        "native_values_bytes": native_values_bytes,
        "float32_native_values_bytes": sum(
            integer(row["original_bytes"], "original_bytes")
            for row in float32_aggregates
        ),
        "float64_native_values_bytes": sum(
            integer(row["original_bytes"], "original_bytes")
            for row in float64_aggregates
        ),
        "validity_bytes": source_totals["validity_bytes"],
        "logical_blocks": source_totals["float_block_records"],
        "header_bytes": source_totals["header_bytes"],
    }
    binding_scope = campaign_binding.get("scope", {})
    expected_binding_counts = {
        "expected_float_aggregate_count": len(source_aggregates),
        "expected_measured_aggregate_count": nonempty_aggregate_count,
        "expected_n_a_aggregate_count": n_a_aggregate_count,
        "expected_native_values_bytes": native_values_bytes,
    }
    for key, observed in expected_binding_counts.items():
        findings.check(
            binding_scope.get(key) == observed,
            "campaign_binding",
            f"campaign binding {key} differs from source-derived value {observed}",
        )
    input_scope_expectations = {
        "float_aggregate_count": len(source_aggregates),
        "measured_aggregate_count": nonempty_aggregate_count,
        "n_a_aggregate_count": n_a_aggregate_count,
        "measured_dataset_count": measured_dataset_count,
        "float_dataset_count": float_schema_dataset_count,
        "float_block_records": source_totals["float_block_records"],
        "row_count": source_totals["row_count"],
        "valid_count": source_totals["valid_count"],
        "missing_count": source_totals["missing_count"],
        "validity_bytes": source_totals["validity_bytes"],
        "original_bytes": source_totals["original_bytes"],
    }
    for key, observed in input_scope_expectations.items():
        findings.check(
            input_scope_receipt.get(key) == observed,
            "input_scope_receipt",
            f"input scope receipt {key} differs from source-derived value {observed}",
        )
    complete = not findings.errors and not findings.suppressed_errors
    status = "complete" if complete and not arguments.allow_partial else "partial"
    runtime_rows = []
    for runtime in RUNTIMES:
        runtime_statuses = [
            row["status"] for row in dataset_receipts if row["runtime"] == runtime
        ]
        runtime_rows.append(
            {
                "runtime": runtime,
                "status": "complete" if status == "complete" else "partial",
                "version": (
                    runtime_versions.get(runtime)
                    if isinstance(runtime_versions, dict)
                    else None
                ),
                "result_sha256": result_chains[runtime].hexdigest(),
                "message": f"{sum(item in {'ok', 'n_a'} for item in runtime_statuses)} of {len(dataset_ids)} dataset receipts closed.",
            }
        )
    environment["artifact_manifest_sha256"] = artifact_sha256
    environment["observed_cpu_affinity"] = sorted(observed_cpu_affinity)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": utc_now(),
        "summary": summary,
        "measurement": {
            "context": "Frozen typed native-float DBF1/DBF2 full-wire diagnostic",
            "repeats": repeats,
            "repetitions": repeats,
            "warmup_blocks": warmup_blocks,
            "header_bytes_per_block": HEADER_BYTES_PER_BLOCK,
            "encode_boundary": "scale selection, native conversion, DBF header and integer-body serialization",
            "decode_boundary": "DBF header parse, integer-body decode, and inverse scaling or IEEE-word reconstruction",
            "ranking_integration": "separate",
            "notes": [
                "Frame bytes include each 10-byte DBF header and integer body; validity sidecars are separate.",
                "Timing excludes source I/O, source-hash checks, and post-decode bitwise verification.",
                "This diagnostic does not replace historical leaderboard rows or participate in Overall ranking.",
            ],
        },
        "environment": environment,
        "runtimes": runtime_rows,
        "datasets": dataset_receipts,
        "columns": sorted(
            columns,
            key=lambda row: (
                dataset_ids.index(row["dataset"]),
                row["aggregate_id"],
                RUNTIMES.index(row["runtime"]),
            ),
        ),
        "evidence": {
            "scope_sha256": scope_sha256,
            "source_column_inventory_sha256": inventory_sha256,
            "runtime_artifact_manifest_sha256": artifact_sha256,
            "campaign_binding_sha256": campaign_binding_sha256,
            "frame_verification_receipt_sha256": golden_receipt_sha256,
            "compact_collection_receipt_sha256": collection_sha256,
            "input_scope_receipt_sha256": input_scope_receipt_sha256,
            "campaign_run_receipt_sha256": campaign_run_receipt_sha256,
            "campaign_progress_sha256": campaign_progress_sha256,
            "environment_receipt_sha256": environment_sha256,
            "source_evidence_chain_sha256": source_chain.hexdigest(),
            "runtime_result_chain_sha256": {
                runtime: digest.hexdigest() for runtime, digest in result_chains.items()
            },
            "source_authority": (
                authority_projections[0] if authority_projections else {}
            ),
            "runtime_artifacts": public_artifact_summary(common_artifact),
            "campaign_binding": {
                "status": campaign_binding.get("status"),
                "source_commit": campaign_binding.get("source_commit", {}).get(
                    "commit"
                ),
                "source_commit_receipt_sha256": campaign_binding.get(
                    "source_commit", {}
                ).get("receipt_sha256"),
                "remote_build_receipt_sha256": campaign_binding.get(
                    "remote_build_reproduction", {}
                ).get("sha256"),
                "frame_verification": {
                    "status": golden_receipt.get("status"),
                    "case_count": golden_receipt.get("case_count"),
                    "cold_decoder_pairs": golden_receipt.get("cold_decoder_pairs"),
                },
            },
        },
    }
    csv_payload = csv_bytes(payload["columns"])
    gate_names = (
        "scope",
        "compact_collection",
        "input_scope_receipt",
        "source_manifest_coverage",
        "source_manifest",
        "source_preflight",
        "source_authority",
        "source_aggregate_uniqueness",
        "source_aggregate_shape",
        "source_counts",
        "column_inventory",
        "dtype_width",
        "native_bytes",
        "header_bytes",
        "dataset_runtime_receipts",
        "runtime_coverage",
        "runtime_totals",
        "source_result_exactness",
        "bitwise_verification",
        "mode_counts",
        "n_a_semantics",
        "frame_sequence_sha256",
        "cross_runtime",
        "campaign_run_receipt",
        "artifact_manifest",
        "campaign_binding",
        "frame_verification",
        "environment",
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed" if status == "complete" else "incomplete",
        "created_at_utc": payload["generated_at"],
        "mode": "complete" if status == "complete" else "qa_partial",
        "summary": summary,
        "gates": {
            gate: {
                "status": "failed" if findings.failed_gates[gate] else "passed",
                "failure_count": findings.failed_gates[gate],
            }
            for gate in gate_names
        },
        "errors": findings.errors,
        "suppressed_error_count": findings.suppressed_errors,
        "inputs": {
            "scope_sha256": scope_sha256,
            "source_column_inventory_sha256": inventory_sha256,
            "artifact_manifest_sha256": artifact_sha256,
            "campaign_binding_sha256": campaign_binding_sha256,
            "frame_verification_receipt_sha256": golden_receipt_sha256,
            "compact_collection_receipt_sha256": collection_sha256,
            "input_scope_receipt_sha256": input_scope_receipt_sha256,
            "campaign_run_receipt_sha256": campaign_run_receipt_sha256,
            "campaign_progress_sha256": campaign_progress_sha256,
            "environment_sha256": environment_sha256,
            "source_evidence_chain_sha256": source_chain.hexdigest(),
            "runtime_result_chain_sha256": {
                runtime: digest.hexdigest() for runtime, digest in result_chains.items()
            },
            "dataset_run_receipt_chain_sha256": sha256_bytes(
                canonical_json_bytes(
                    [run_receipt_hashes.get(dataset) for dataset in dataset_ids]
                )
            ),
            "source_file_count": sum(
                2 + int(row["blocks_locally_rehashed"]) for row in input_files
            ),
            "recorded_block_manifest_count": len(input_files),
            "runtime_result_file_count": runtime_result_file_count,
            "compact_collection_file_count": collection_receipt.get("file_count"),
        },
    }
    return payload, csv_payload, receipt


def write_outputs(
    arguments: argparse.Namespace,
    payload: dict[str, Any],
    csv_payload: bytes,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    output_dir = arguments.output_dir.resolve()
    if arguments.allow_partial:
        if inside(output_dir, DEPLOYMENT_DIR):
            raise AuditBuildError(
                "--allow-partial output must be outside the deployment directory"
            )
        json_name = "dbp-audit.partial.json"
        csv_name = "dbp-audit.partial-columns.csv"
    else:
        if payload["status"] != "complete" or receipt["status"] != "passed":
            raise AuditBuildError("complete publication gates did not pass")
        validate_public_payload(payload)
        json_name = "dbp-audit.json"
        csv_name = "dbp-audit-columns.csv"
    json_payload = public_json_bytes(payload)
    receipt["outputs"] = {
        json_name: {"bytes": len(json_payload), "sha256": sha256_bytes(json_payload)},
        csv_name: {"bytes": len(csv_payload), "sha256": sha256_bytes(csv_payload)},
    }
    receipt_payload = public_json_bytes(receipt)
    atomic_write(output_dir / json_name, json_payload)
    atomic_write(output_dir / csv_name, csv_payload)
    atomic_write(output_dir / "dbp-audit-build-receipt.json", receipt_payload)
    return {
        "status": payload["status"],
        "json": json_name,
        "csv": csv_name,
        "receipt": "dbp-audit-build-receipt.json",
        "json_sha256": sha256_bytes(json_payload),
        "csv_sha256": sha256_bytes(csv_payload),
    }


def validate_public_payload(payload: dict[str, Any]) -> None:
    schema_path = Path(__file__).resolve().parent / "dbp_measurement.schema.json"
    schema = load_json(schema_path)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(payload, schema)
    except (jsonschema.SchemaError, jsonschema.ValidationError) as error:
        raise AuditBuildError(
            f"public DBP audit schema validation failed: {error.message}"
        ) from error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument(
        "--campaign-binding-sha256",
        required=True,
        help="Expected SHA-256 of control_root/final_campaign_binding.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-column-inventory", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        payload, csv_payload, receipt = build_payload(arguments)
        if not arguments.allow_partial and receipt["status"] != "passed":
            details = json.dumps(
                {
                    "status": "failed",
                    "errors": receipt["errors"][:20],
                    "suppressed_error_count": receipt["suppressed_error_count"],
                },
                ensure_ascii=False,
            )
            raise AuditBuildError(details)
        result = write_outputs(arguments, payload, csv_payload, receipt)
    except (
        AuditBuildError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        jsonschema.SchemaError,
        jsonschema.ValidationError,
    ) as error:
        print(f"DBP audit build failed: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
