#!/usr/bin/env python3
"""Fail-closed launcher for the formal weighted-GRPO campaign.

This file is orchestration only.  It does not implement compression or relax
any admission rule.  Before it creates a campaign process, it reopens the
complete 16-source prepared corpus, the formal zlib6 reference table, and the
four-receipt gate bundle with the same loaders used by formal training.  It
also binds those receipts to the currently imported runtime identity.

The first two stages *verify* the already-prepared data and the already-frozen
formal calibration.  They do not regenerate either artifact: a gate bundle
cannot remain valid if its data or calibration is replaced.  Later stages run
the training-only Beam30 seed measurement/library build, the fixed three-seed
training campaign, and explicit frozen-test/comparison/ablation commands.

Every external command is an argv array (never a shell string), has explicit
expected outputs, is logged, and receives single-thread native-library limits.
Stage receipts make ``--resume`` safe: a stage is skipped only when its command
fingerprint and every output hash still match.  The launcher never scans,
signals, renices, or terminates unrelated processes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import build_weighted_grpo_solution_library_v1 as solution_library
import aggregate_formal_weighted_grpo_results_v1 as aggregation
import calibrate_weighted_grpo_references_v1 as calibration
import formal_weighted_grpo_training_v1 as training
import measure_weighted_grpo_solution_seeds_v1 as seed_measurement
import prepare_weighted_grpo_timeseries_v1 as preparation
import weighted_grpo_experiment_v1 as experiment


SCHEMA = "formal-weighted-grpo-campaign-launcher-v1"
SPEC_SCHEMA = "formal-weighted-grpo-campaign-spec-v1"
PLAN_SCHEMA = "formal-weighted-grpo-campaign-plan-v1"
STATE_SCHEMA = "formal-weighted-grpo-campaign-state-v1"
STAGE_RECEIPT_SCHEMA = "formal-weighted-grpo-campaign-stage-receipt-v1"

FORMAL_DATASET_COUNT = 16
FORMAL_SEEDS = (0, 1, 2)
FORMAL_SEED_BUDGET_SECONDS = 16 * 60 * 60
MAX_MEASUREMENT_WORKERS = 4
FORMAL_WEIGHTS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.4, 0.3, 0.3),
)
TASK_FACTORY = "weighted_grpo_graph_task_factory_v1:task_factory"
REQUIRED_COMMAND_PROTOCOLS = {
    "frozen_six_weight_test": "frozen_six_weight_test_v1",
    "beam30_comparison": "beam30_shared_budget_v1",
    "baseline86_native_comparison": "native_baseline86_v1",
}
REQUIRED_ABLATIONS = {
    "no_target_pruning_ranking": "no_target_pruning_ranking_v1",
    "no_block_reuse": "no_block_reuse_v1",
}

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
FORBIDDEN_ENV_FRAGMENTS = (
    "PASSWORD", "PASSWD", "TOKEN", "SECRET", "CREDENTIAL", "PRIVATE_KEY",
)
SOLUTION_LIBRARY_ENV = "WEIGHTED_GRPO_SOLUTION_LIBRARY_MANIFEST"
SOLUTION_LIBRARY_SHA256_ENV = "WEIGHTED_GRPO_SOLUTION_LIBRARY_SHA256"
POLICY_INFERENCE_DEVICE_ENV = "WEIGHTED_GRPO_POLICY_INFERENCE_DEVICE"
SOLUTION_LIBRARY_STAGE_NAMES = frozenset({
    "three_seed_training",
    "frozen_six_weight_test",
    "beam30_comparison",
    "ablations/no_target_pruning_ranking",
    "ablations/no_block_reuse",
})


class CampaignAdmissionError(RuntimeError):
    """Formal evidence or an immutable campaign choice is invalid."""


class CampaignExecutionError(RuntimeError):
    """A stage failed or a resume receipt no longer matches."""


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2,
                allow_nan=False,
            ).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignAdmissionError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise CampaignAdmissionError(f"{label} must be a JSON object")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignAdmissionError(f"{label} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise CampaignAdmissionError(f"{label} must be absolute: {value}")
    return path.resolve()


def _inside(root: Path, path: Path, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CampaignAdmissionError(
            f"{label} must be inside isolated campaign_root: {path}"
        ) from error
    return path


def _exact_weights(value: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list):
        raise CampaignAdmissionError("weights must be a list")
    try:
        weights = tuple(tuple(float(part) for part in row) for row in value)
    except (TypeError, ValueError) as error:
        raise CampaignAdmissionError("weights must contain numeric triples") from error
    if weights != FORMAL_WEIGHTS:
        raise CampaignAdmissionError(
            "formal campaign must use exactly weights 100,010,001,550,505,433"
        )
    return weights


@dataclass(frozen=True)
class CommandStage:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    expected_outputs: tuple[Path, ...]
    environment: tuple[tuple[str, str], ...] = ()

    @property
    def fingerprint(self) -> str:
        return _sha256_bytes(_canonical_json({
            "name": self.name,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "expected_outputs": [str(path) for path in self.expected_outputs],
            "environment": [list(item) for item in self.environment],
        }))

    def as_plan(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": "command",
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "expected_outputs": [str(path) for path in self.expected_outputs],
            "environment": dict(self.environment),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class CampaignSpec:
    spec_path: Path
    repository_root: Path
    campaign_root: Path
    prepared: tuple[Path, ...]
    test_prepared: tuple[Path, ...]
    reference_table: Path
    gate_bundle: Path
    measurement_receipt: Path
    measurement_payload_dir: Path
    solution_library_manifest: Path
    training_output: Path
    payload_audit_receipt: Path
    final_detail_output: Path
    final_summary_output: Path
    task_factory: str
    max_measurement_workers: int
    per_seed_budget_seconds: int
    seeds: tuple[int, int, int]
    weights: tuple[tuple[float, float, float], ...]
    commands: tuple[CommandStage, ...]
    spec_sha256: str
    policy_device: str = "cpu"
    policy_inference_device: str = "cpu"


def _expand_tokens(text: str, values: Mapping[str, str]) -> str:
    pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise CampaignAdmissionError(f"unknown command token: {name}")
        return values[name]

    return pattern.sub(replace, text)


def _parse_command_stage(
    name: str, raw: Any, *, tokens: Mapping[str, str], repository_root: Path,
    campaign_root: Path,
) -> CommandStage:
    if not isinstance(raw, Mapping):
        raise CampaignAdmissionError(f"commands.{name} must be an object")
    argv_raw = raw.get("argv")
    if not isinstance(argv_raw, list) or not argv_raw or not all(
        isinstance(item, str) and item for item in argv_raw
    ):
        raise CampaignAdmissionError(f"commands.{name}.argv must be a non-empty string list")
    argv = tuple(_expand_tokens(item, tokens) for item in argv_raw)
    executable = Path(argv[0]).expanduser()
    if not executable.is_absolute():
        raise CampaignAdmissionError(
            f"commands.{name}.argv[0] must be an absolute executable path"
        )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise CampaignAdmissionError(
            f"commands.{name}.argv[0] is not an executable file: {executable}"
        )
    cwd = _absolute_path(
        _expand_tokens(str(raw.get("cwd", repository_root)), tokens),
        f"commands.{name}.cwd",
    )
    if not cwd.is_dir():
        raise CampaignAdmissionError(f"commands.{name}.cwd is not a directory: {cwd}")
    outputs_raw = raw.get("expected_outputs")
    if not isinstance(outputs_raw, list) or not outputs_raw:
        raise CampaignAdmissionError(
            f"commands.{name}.expected_outputs must be non-empty"
        )
    outputs = tuple(
        _inside(
            campaign_root,
            _absolute_path(_expand_tokens(str(value), tokens),
                           f"commands.{name}.expected_outputs"),
            f"commands.{name}.expected_outputs",
        )
        for value in outputs_raw
    )
    environment_raw = raw.get("environment", {})
    if not isinstance(environment_raw, Mapping):
        raise CampaignAdmissionError(f"commands.{name}.environment must be an object")
    environment: list[tuple[str, str]] = []
    for key, value in sorted(environment_raw.items()):
        key_text = str(key)
        if any(fragment in key_text.upper() for fragment in FORBIDDEN_ENV_FRAGMENTS):
            raise CampaignAdmissionError(
                f"commands.{name}.environment may not contain credentials: {key_text}"
            )
        if key_text in {SOLUTION_LIBRARY_ENV, SOLUTION_LIBRARY_SHA256_ENV,
                        POLICY_INFERENCE_DEVICE_ENV}:
            raise CampaignAdmissionError(
                f"commands.{name}.environment may not override launcher-owned "
                "solution-library bindings"
            )
        environment.append((key_text, _expand_tokens(str(value), tokens)))
    return CommandStage(name, argv, cwd, outputs, tuple(environment))


def load_spec(path: Path) -> CampaignSpec:
    resolved = path.expanduser().resolve()
    raw_bytes = resolved.read_bytes()
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignAdmissionError(f"invalid campaign spec JSON: {resolved}") from error
    if not isinstance(document, Mapping) or document.get("schema") != SPEC_SCHEMA:
        raise CampaignAdmissionError("unsupported campaign spec schema")
    from weighted_grpo_torch_backend_v1 import normalize_device
    try:
        policy_device = normalize_device(document.get("policy_device", "cpu"))
        inference_device = normalize_device(document.get("policy_inference_device", "cpu"))
    except (TypeError, ValueError) as error:
        raise CampaignAdmissionError("policy devices must name cpu or one cuda:N device") from error
    cuda_devices = {v for v in (policy_device, inference_device) if v.startswith("cuda:")}
    if len(cuda_devices) > 1:
        raise CampaignAdmissionError("training and inference must use the same single CUDA device")
    repository_root = _absolute_path(document.get("repository_root"), "repository_root")
    campaign_root = _absolute_path(document.get("campaign_root"), "campaign_root")
    if campaign_root == repository_root or campaign_root in repository_root.parents:
        raise CampaignAdmissionError("campaign_root must not contain repository_root")
    prepared_raw = document.get("prepared")
    if not isinstance(prepared_raw, list) or len(prepared_raw) != FORMAL_DATASET_COUNT:
        raise CampaignAdmissionError("formal campaign requires exactly 16 prepared datasets")
    prepared = tuple(_absolute_path(value, "prepared") for value in prepared_raw)
    if len(set(prepared)) != FORMAL_DATASET_COUNT:
        raise CampaignAdmissionError("prepared dataset paths must be unique")
    test_prepared_raw = document.get("test_prepared")
    if not isinstance(test_prepared_raw, list) or not test_prepared_raw:
        raise CampaignAdmissionError("formal campaign requires a non-empty frozen test cohort")
    test_prepared = tuple(
        _absolute_path(value, "test_prepared") for value in test_prepared_raw
    )
    if len(set(test_prepared)) != len(test_prepared):
        raise CampaignAdmissionError("test prepared dataset paths must be unique")
    reference = _absolute_path(document.get("reference_table"), "reference_table")
    gate = _absolute_path(document.get("gate_bundle"), "gate_bundle")
    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping):
        raise CampaignAdmissionError("outputs must be an object")
    measurement_receipt = _inside(
        campaign_root, _absolute_path(outputs.get("measurement_receipt"),
                                      "outputs.measurement_receipt"),
        "outputs.measurement_receipt",
    )
    measurement_payload_dir = _inside(
        campaign_root, _absolute_path(outputs.get("measurement_payload_dir"),
                                      "outputs.measurement_payload_dir"),
        "outputs.measurement_payload_dir",
    )
    library_manifest = _inside(
        campaign_root, _absolute_path(outputs.get("solution_library_manifest"),
                                      "outputs.solution_library_manifest"),
        "outputs.solution_library_manifest",
    )
    training_output = _inside(
        campaign_root, _absolute_path(outputs.get("training_output"),
                                      "outputs.training_output"),
        "outputs.training_output",
    )
    payload_audit_receipt = _inside(
        campaign_root,
        _absolute_path(outputs.get("payload_audit_receipt"),
                       "outputs.payload_audit_receipt"),
        "outputs.payload_audit_receipt",
    )
    final_detail_output = _inside(
        campaign_root,
        _absolute_path(outputs.get("final_detail_output"),
                       "outputs.final_detail_output"),
        "outputs.final_detail_output",
    )
    final_summary_output = _inside(
        campaign_root,
        _absolute_path(outputs.get("final_summary_output"),
                       "outputs.final_summary_output"),
        "outputs.final_summary_output",
    )
    task_factory = str(document.get("task_factory", ""))
    if task_factory != TASK_FACTORY:
        raise CampaignAdmissionError(
            f"formal campaign task_factory must be {TASK_FACTORY}"
        )
    max_workers = int(document.get("max_measurement_workers", 0))
    if not 1 <= max_workers <= MAX_MEASUREMENT_WORKERS:
        raise CampaignAdmissionError("max_measurement_workers must be in [1,4]")
    seed_budget = int(document.get("per_seed_budget_seconds", 0))
    if seed_budget != FORMAL_SEED_BUDGET_SECONDS:
        raise CampaignAdmissionError("each formal seed must retain the 16-hour budget")
    seeds_raw = document.get("seeds")
    if not isinstance(seeds_raw, list) or tuple(seeds_raw) != FORMAL_SEEDS:
        raise CampaignAdmissionError("formal campaign seeds must be exactly [0,1,2]")
    weights = _exact_weights(document.get("weights"))
    tokens = {
        "repository_root": str(repository_root),
        "campaign_root": str(campaign_root),
        "reference_table": str(reference),
        "gate_bundle": str(gate),
        "measurement_receipt": str(measurement_receipt),
        "measurement_payload_dir": str(measurement_payload_dir),
        "solution_library_manifest": str(library_manifest),
        "training_output": str(training_output),
        "delivery_checkpoint": str(training_output / "delivery_policy.json"),
        "campaign_plan": str(campaign_root / "campaign_plan.json"),
        "python": str(Path(sys.executable).resolve()),
    }
    commands_raw = document.get("commands")
    if not isinstance(commands_raw, Mapping):
        raise CampaignAdmissionError("commands must be an object")
    required_commands = (
        "frozen_six_weight_test", "beam30_comparison",
        "baseline86_native_comparison",
    )
    if set(commands_raw) != set(required_commands) | {"ablations"}:
        raise CampaignAdmissionError(
            "commands must contain exactly the three comparisons and ablations"
        )
    for name in required_commands:
        if name not in commands_raw:
            raise CampaignAdmissionError(f"commands.{name} is required")
        raw = commands_raw[name]
        if (not isinstance(raw, Mapping)
                or raw.get("protocol") != REQUIRED_COMMAND_PROTOCOLS[name]):
            raise CampaignAdmissionError(
                f"commands.{name}.protocol must be "
                f"{REQUIRED_COMMAND_PROTOCOLS[name]}"
            )
    stages = [
        _parse_command_stage(
            name, commands_raw[name], tokens=tokens,
            repository_root=repository_root, campaign_root=campaign_root,
        )
        for name in required_commands
    ]
    if any(len(stage.expected_outputs) != 1 for stage in stages):
        raise CampaignAdmissionError(
            "each formal comparison command must declare exactly one semantic output"
        )
    ablations = commands_raw.get("ablations")
    if not isinstance(ablations, list) or len(ablations) != len(REQUIRED_ABLATIONS):
        raise CampaignAdmissionError(
            "commands.ablations must contain exactly the two formal ablations"
        )
    seen_ablations: set[str] = set()
    for raw in ablations:
        if not isinstance(raw, Mapping):
            raise CampaignAdmissionError("each ablation command must be an object")
        ablation_id = str(raw.get("id", ""))
        if (ablation_id not in REQUIRED_ABLATIONS
                or raw.get("protocol") != REQUIRED_ABLATIONS.get(ablation_id)):
            raise CampaignAdmissionError(
                "ablations must explicitly declare no_target_pruning_ranking_v1 "
                "and no_block_reuse_v1"
            )
        if ablation_id in seen_ablations:
            raise CampaignAdmissionError(f"duplicate ablation: {ablation_id}")
        seen_ablations.add(ablation_id)
        stages.append(_parse_command_stage(
            f"ablations/{ablation_id}", raw, tokens=tokens,
            repository_root=repository_root, campaign_root=campaign_root,
        ))
    if seen_ablations != set(REQUIRED_ABLATIONS):
        raise CampaignAdmissionError("formal ablation set is incomplete")
    return CampaignSpec(
        spec_path=resolved, repository_root=repository_root,
        campaign_root=campaign_root, prepared=prepared,
        test_prepared=test_prepared,
        reference_table=reference, gate_bundle=gate,
        measurement_receipt=measurement_receipt,
        measurement_payload_dir=measurement_payload_dir,
        solution_library_manifest=library_manifest,
        training_output=training_output,
        payload_audit_receipt=payload_audit_receipt,
        final_detail_output=final_detail_output,
        final_summary_output=final_summary_output,
        task_factory=task_factory,
        max_measurement_workers=max_workers,
        per_seed_budget_seconds=seed_budget, seeds=FORMAL_SEEDS,
        weights=weights, commands=tuple(stages),
        spec_sha256=_sha256_bytes(raw_bytes),
        policy_device=policy_device, policy_inference_device=inference_device,
    )


def _gate_receipts(bundle_path: Path) -> dict[str, dict[str, Any]]:
    bundle = _load_json(bundle_path, "formal gate bundle")
    if bundle.get("schema") != training.GATE_BUNDLE_SCHEMA:
        raise CampaignAdmissionError("unsupported formal gate-bundle schema")
    descriptors = bundle.get("receipts")
    required = set(training.REQUIRED_GATE_ASSERTIONS)
    if not isinstance(descriptors, Mapping) or set(descriptors) != required:
        raise CampaignAdmissionError("gate bundle must contain exactly four receipts")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        descriptor = descriptors[name]
        if not isinstance(descriptor, Mapping):
            raise CampaignAdmissionError(f"invalid {name} gate descriptor")
        receipt_path = (bundle_path.parent / str(descriptor.get("path", ""))).resolve()
        expected = str(descriptor.get("sha256", ""))
        if not receipt_path.is_file() or _sha256_file(receipt_path) != expected:
            raise CampaignAdmissionError(f"{name} gate receipt is absent or changed")
        result[name] = _load_json(receipt_path, f"{name} gate receipt")
    return result


def admit_campaign(spec: CampaignSpec) -> dict[str, Any]:
    """Revalidate the complete corpus, frozen calibration, and four receipts."""
    if not spec.repository_root.is_dir():
        raise CampaignAdmissionError("repository_root does not exist")
    for path in (*spec.prepared, *spec.test_prepared,
                 spec.reference_table, spec.gate_bundle):
        if not path.exists():
            raise CampaignAdmissionError(f"formal input is missing: {path}")
    try:
        corpus = training._load_corpus(spec.prepared, spec.reference_table)
        gate = training._load_gate_bundle(spec.gate_bundle, corpus)
    except (training.TrainingAdmissionError, experiment.ExperimentContractError,
            preparation.PreparationError, OSError, ValueError) as error:
        raise CampaignAdmissionError(str(error)) from error
    if len(corpus.dataset_provenance_sha256) != FORMAL_DATASET_COUNT:
        raise CampaignAdmissionError("reduced-cohort formal training is forbidden")
    training_provenance = dict(corpus.dataset_provenance_sha256)
    test_provenance: dict[str, str] = {}
    test_block_count = 0
    test_block_identities: list[dict[str, Any]] = []
    cohort_manifests = [
        _load_json(
            path / "dataset_manifest.json" if path.is_dir() else path,
            "frozen training prepared manifest",
        )
        for path in spec.prepared
    ]
    for prepared in spec.test_prepared:
        try:
            dataset = experiment._load_prepared_dataset(
                prepared, verify_source=True, expected_role="test_candidate",
            )
        except experiment.ExperimentContractError as error:
            raise CampaignAdmissionError(str(error)) from error
        manifest = dataset.manifest
        cohort_manifests.append(manifest)
        dataset_id = str(manifest.get("dataset_id", ""))
        provenance_sha256 = str(manifest.get("provenance_sha256", ""))
        if (not dataset_id or len(provenance_sha256) != 64
                or dataset_id in test_provenance):
            raise CampaignAdmissionError("frozen test cohort identity is invalid/duplicate")
        if dataset_id in training_provenance or provenance_sha256 in training_provenance.values():
            raise CampaignAdmissionError(
                "frozen test cohort overlaps the training/validation cohort"
            )
        blocks = manifest.get("blocks")
        if not isinstance(blocks, list):
            raise CampaignAdmissionError("frozen test manifest blocks must be a list")
        selected = sum(
            isinstance(row, Mapping) and row.get("split") == "test"
            for row in blocks
        )
        if selected == 0:
            raise CampaignAdmissionError(
                f"frozen test dataset {dataset_id} has no test blocks"
            )
        test_block_count += selected
        test_provenance[dataset_id] = provenance_sha256
        for row in blocks:
            if not isinstance(row, Mapping) or row.get("split") != "test":
                continue
            test_block_identities.append({
                "dataset_id": dataset_id,
                "dataset_provenance_sha256": provenance_sha256,
                "block_id": str(row.get("block_id", "")),
                "entity_id": str(row.get("entity_id", "")),
                "split": "test",
                "column": str(row.get("column", "")),
                "dtype": str(row.get("dtype", "")),
                "block_index": int(row.get("block_index", -1)),
                "original_numeric_bytes": int(
                    row.get("original_numeric_bytes", -1)
                ),
                "prepared_block_bytes": int(row.get("prepared_block_bytes", -1)),
            })
    try:
        cohort_verification = preparation.validate_prepared_cohort_manifests(
            cohort_manifests, preparation.load_cohort_manifest(),
            require_complete_train=True,
        )
    except preparation.PreparationError as error:
        raise CampaignAdmissionError(str(error)) from error
    if any(
        not row["block_id"] or not row["entity_id"] or not row["column"]
        or not row["dtype"] or row["block_index"] < 0
        or row["original_numeric_bytes"] <= 0
        or row["prepared_block_bytes"] < row["original_numeric_bytes"]
        for row in test_block_identities
    ):
        raise CampaignAdmissionError("frozen test cohort contains an invalid block contract")
    reference = _load_json(spec.reference_table, "formal zlib6 reference table")
    if reference.get("schema") != calibration.SCHEMA:
        raise CampaignAdmissionError("smoke or non-formal zlib6 calibration is forbidden")
    if reference.get("calibration_sha256") != corpus.calibration_sha256:
        raise CampaignAdmissionError("calibration content hash differs from the corpus")
    if int(reference.get("training_block_count", -1)) != len(corpus.train_blocks):
        raise CampaignAdmissionError("formal calibration does not cover every training block")
    receipts = _gate_receipts(spec.gate_bundle)
    data_receipt = receipts["data"]
    provenance = data_receipt.get("dataset_provenance_sha256")
    if not isinstance(provenance, Mapping) or len(provenance) != FORMAL_DATASET_COUNT:
        raise CampaignAdmissionError("data receipt is not a complete 16-source cohort")
    admission = data_receipt.get("dataset_admission")
    source_gate = admission.get("source_spec_gate") if isinstance(admission, Mapping) else None
    if (not isinstance(source_gate, Mapping)
            or source_gate.get("complete_cohort_trainable") is not True
            or source_gate.get("blocked") != 0):
        raise CampaignAdmissionError("blocked dataset sources cannot enter formal training")
    live_identity: dict[str, Any] = {}
    try:
        import build_formal_weighted_grpo_gate_bundle_v1 as gate_builder
        identity = gate_builder.discover_runtime_identity()
        if identity.source_fingerprint != gate.source_fingerprint:
            raise CampaignAdmissionError("runtime source fingerprint changed after gate assembly")
        if identity.action_registry_fingerprint != gate.action_registry_fingerprint:
            raise CampaignAdmissionError("action registry changed after gate assembly")
        live_identity = {
            "source_fingerprint": identity.source_fingerprint,
            "action_registry_fingerprint": identity.action_registry_fingerprint,
            "action_count": identity.action_count,
        }
    except ImportError as error:
        raise CampaignAdmissionError("cannot import live gate identity checker") from error
    return {
        "schema": "formal-weighted-grpo-campaign-admission-v1",
        "passed": True,
        "dataset_count": len(corpus.dataset_provenance_sha256),
        "train_block_count": len(corpus.train_blocks),
        "validation_block_count": len(corpus.validation_blocks),
        "test_dataset_count": len(test_provenance),
        "test_block_count": test_block_count,
        "test_dataset_provenance_sha256": test_provenance,
        "combined_cohort_verification": cohort_verification,
        "test_cohort_sha256": _sha256_bytes(_canonical_json(sorted(
            test_block_identities,
            key=lambda row: (row["dataset_id"], row["block_id"]),
        ))),
        "corpus_sha256": corpus.corpus_sha256,
        "calibration_sha256": corpus.calibration_sha256,
        "gate_bundle_sha256": gate.bundle_sha256,
        "gate_receipt_sha256": [list(row) for row in gate.receipt_sha256],
        **live_identity,
    }


def _generated_stages(spec: CampaignSpec) -> tuple[CommandStage, ...]:
    scripts = spec.repository_root / "scripts"
    python = str(Path(sys.executable).resolve())
    prepared_args = tuple(
        argument
        for path in spec.prepared
        for argument in ("--prepared", str(path))
    )
    measure = CommandStage(
        "training_seed_measurement",
        (
            python, str(scripts / "measure_weighted_grpo_solution_seeds_v1.py"),
            *prepared_args,
            "--reference-table", str(spec.reference_table),
            "--output-directory", str(spec.measurement_payload_dir),
            "--output", str(spec.measurement_receipt),
        ),
        spec.repository_root,
        (spec.measurement_receipt, spec.measurement_payload_dir),
    )
    library = CommandStage(
        "training_solution_library",
        (
            python, str(scripts / "build_weighted_grpo_solution_library_v1.py"),
            *prepared_args,
            "--receipt", str(spec.measurement_receipt),
            "--reference-table", str(spec.reference_table),
            "--output", str(spec.solution_library_manifest),
        ),
        spec.repository_root,
        (spec.solution_library_manifest,),
    )
    training_stage = CommandStage(
        "three_seed_training",
        (
            python, str(scripts / "formal_weighted_grpo_training_v1.py"),
            *prepared_args,
            "--reference", str(spec.reference_table),
            "--admission-bundle", str(spec.gate_bundle),
            "--task-factory", spec.task_factory,
            "--output", str(spec.training_output),
            "--policy-device", spec.policy_device,
            "--measurement-workers", str(spec.max_measurement_workers),
        ),
        spec.repository_root,
        (
            spec.training_output / "campaign_receipt.json",
            spec.training_output / "delivery_policy.json",
        ),
        (
            (SOLUTION_LIBRARY_ENV, str(spec.solution_library_manifest)),
        ),
    )
    external = {stage.name: stage for stage in spec.commands}
    for name in (
        "frozen_six_weight_test", "beam30_comparison",
        "ablations/no_target_pruning_ranking", "ablations/no_block_reuse",
    ):
        stage = external[name]
        external[name] = CommandStage(
            stage.name, stage.argv, stage.cwd, stage.expected_outputs,
            tuple(sorted((*stage.environment,
                          (SOLUTION_LIBRARY_ENV, str(spec.solution_library_manifest)),
                          (POLICY_INFERENCE_DEVICE_ENV, spec.policy_inference_device)))),
        )
    ordered: list[CommandStage] = [measure, library, training_stage]
    ordered.extend([
        external["frozen_six_weight_test"],
        external["beam30_comparison"],
        external["baseline86_native_comparison"],
    ])
    ordered.extend(sorted(
        (stage for name, stage in external.items()
         if name.startswith("ablations/")),
        key=lambda stage: stage.name,
    ))
    frozen_output = external["frozen_six_weight_test"].expected_outputs[0]
    beam_output = external["beam30_comparison"].expected_outputs[0]
    baseline_output = external["baseline86_native_comparison"].expected_outputs[0]
    ablation_outputs = {
        stage.name.split("/", 1)[1]: stage.expected_outputs[0]
        for stage in spec.commands if stage.name.startswith("ablations/")
    }
    common_aggregate_argv = (
        "--campaign-plan", str(spec.campaign_root / "campaign_plan.json"),
        "--reference-table", str(spec.reference_table),
        "--training-receipt", str(spec.training_output / "campaign_receipt.json"),
        "--frozen-grpo", str(frozen_output),
        "--beam30", str(beam_output),
        "--baseline86", str(baseline_output),
        "--no-target-pruning-ranking",
        str(ablation_outputs["no_target_pruning_ranking"]),
        "--no-block-reuse", str(ablation_outputs["no_block_reuse"]),
    )
    audit = CommandStage(
        "payload_artifact_audit",
        (
            python, str(scripts / "aggregate_formal_weighted_grpo_results_v1.py"),
            "audit-payloads", *common_aggregate_argv,
            "--output", str(spec.payload_audit_receipt),
        ),
        spec.repository_root,
        (spec.payload_audit_receipt,),
    )
    aggregate = CommandStage(
        "final_result_aggregation",
        (
            python, str(scripts / "aggregate_formal_weighted_grpo_results_v1.py"),
            "aggregate", *common_aggregate_argv,
            "--payload-audit", str(spec.payload_audit_receipt),
            "--detail-output", str(spec.final_detail_output),
            "--summary-output", str(spec.final_summary_output),
        ),
        spec.repository_root,
        (spec.final_detail_output, spec.final_summary_output),
    )
    ordered.extend((audit, aggregate))
    return tuple(ordered)


def build_plan(spec: CampaignSpec, admission: Mapping[str, Any]) -> dict[str, Any]:
    stages: list[dict[str, Any]] = [
        {
            "name": "prepared_data_verification",
            "mode": "verify_existing_formal_artifact",
            "dataset_count": admission["dataset_count"],
            "corpus_sha256": admission["corpus_sha256"],
        },
        {
            "name": "formal_zlib6_calibration",
            "mode": "verify_existing_frozen_reference",
            "calibration_sha256": admission["calibration_sha256"],
            "note": "recalibration would invalidate the pre-existing gate bundle",
        },
    ]
    stages.extend(stage.as_plan() for stage in _generated_stages(spec))
    plan = {
        "schema": PLAN_SCHEMA,
        "spec_path": str(spec.spec_path),
        "spec_sha256": spec.spec_sha256,
        "campaign_root": str(spec.campaign_root),
        "training_prepared_manifests": [str(path) for path in spec.prepared],
        "test_prepared_manifests": [str(path) for path in spec.test_prepared],
        "delivery_checkpoint": str(spec.training_output / "delivery_policy.json"),
        "solution_library_manifest": str(spec.solution_library_manifest),
        "solution_library_binding": {
            "path": str(spec.solution_library_manifest),
            "file_sha256_source": "recomputed_after_training_solution_library_stage",
            "action_registry_fingerprint_source": "validated_manifest",
        },
        "formal_seeds": list(spec.seeds),
        "per_seed_budget_seconds": spec.per_seed_budget_seconds,
        "max_measurement_workers": spec.max_measurement_workers,
        "policy_update_device": spec.policy_device,
        "policy_rollout_device": "cpu",
        "policy_inference_device": spec.policy_inference_device,
        "weights": [list(row) for row in spec.weights],
        "admission": dict(admission),
        "stage_order": [stage["name"] for stage in stages],
        "stages": stages,
        "resource_policy": {
            "native_library_threads": 1,
            "max_measurement_workers": spec.max_measurement_workers,
            "unrelated_process_actions": "none",
            "campaign_lock": str(spec.campaign_root / ".campaign.lock"),
        },
    }
    plan["plan_sha256"] = _sha256_bytes(_canonical_json(plan))
    return plan


def _path_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "path": str(path), "kind": "file", "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    if path.is_dir():
        rows = []
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            rows.append({
                "path": child.relative_to(path).as_posix(),
                "sha256": _sha256_file(child), "size": child.stat().st_size,
            })
        return {
            "path": str(path), "kind": "directory", "files": rows,
            "tree_sha256": _sha256_bytes(_canonical_json(rows)),
        }
    raise CampaignExecutionError(f"expected stage output is missing: {path}")


def _solution_library_binding(path: Path) -> dict[str, str]:
    """Reopen and validate the exact library bytes used by a formal stage."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CampaignExecutionError(f"solution library is missing: {resolved}")
    file_sha256 = _sha256_file(resolved)
    try:
        identity = solution_library.validate_library_manifest(
            resolved, expected_file_sha256=file_sha256,
        )
    except (solution_library.SolutionLibraryBuildError, OSError, ValueError) as error:
        raise CampaignExecutionError(str(error)) from error
    return {
        "path": str(resolved),
        "file_sha256": file_sha256,
        "action_registry_fingerprint": str(
            identity["action_registry_fingerprint"]
        ),
    }


def _validate_resume_receipt(
    stage: CommandStage, receipt_path: Path,
    *, solution_library_binding: Mapping[str, str] | None = None,
) -> bool:
    if not receipt_path.is_file():
        return False
    receipt = _load_json(receipt_path, f"{stage.name} stage receipt")
    if receipt.get("schema") != STAGE_RECEIPT_SCHEMA:
        raise CampaignExecutionError(f"invalid stage receipt schema: {receipt_path}")
    if receipt.get("stage_fingerprint") != stage.fingerprint:
        raise CampaignExecutionError(
            f"receipt for {stage.name} does not match current plan"
        )
    if solution_library_binding is not None:
        if receipt.get("solution_library_binding") != dict(solution_library_binding):
            raise CampaignExecutionError(
                f"solution library for {stage.name} changed after receipt"
            )
    if receipt.get("status") == "failed":
        return False
    if receipt.get("status") != "complete":
        raise CampaignExecutionError(f"invalid stage status for {stage.name}")
    expected = receipt.get("outputs")
    actual = [_path_record(path) for path in stage.expected_outputs]
    if expected != actual:
        raise CampaignExecutionError(
            f"completed outputs for {stage.name} changed after receipt"
        )
    return True


def _validate_core_stage_output(stage: CommandStage, spec: CampaignSpec) -> None:
    if stage.name == "training_seed_measurement":
        receipt = _load_json(spec.measurement_receipt, "training seed measurement")
        if receipt.get("schema") != seed_measurement.SCHEMA:
            raise CampaignExecutionError("training seed receipt has the wrong schema")
        selected = receipt.get("selected_training_blocks")
        measurements = receipt.get("measurements")
        if not isinstance(selected, list) or not selected or not isinstance(measurements, list):
            raise CampaignExecutionError("training seed receipt is empty or incomplete")
        if any(
            not isinstance(row, Mapping) or row.get("source_split") != "train"
            for row in measurements
        ):
            raise CampaignExecutionError("solution seeds include non-training data")
    elif stage.name == "training_solution_library":
        try:
            solution_library.validate_library_manifest(
                spec.solution_library_manifest,
                expected_file_sha256=_sha256_file(spec.solution_library_manifest),
            )
        except (solution_library.SolutionLibraryBuildError, OSError, ValueError) as error:
            raise CampaignExecutionError(str(error)) from error
    elif stage.name == "three_seed_training":
        campaign = _load_json(
            spec.training_output / "campaign_receipt.json", "training campaign receipt",
        )
        if campaign.get("schema") != training.CAMPAIGN_SCHEMA:
            raise CampaignExecutionError("training campaign receipt has the wrong schema")
        if tuple(campaign.get("seeds", ())) != FORMAL_SEEDS:
            raise CampaignExecutionError("training campaign did not cover seeds 0,1,2")
        if campaign.get("per_seed_budget_seconds") != FORMAL_SEED_BUDGET_SECONDS:
            raise CampaignExecutionError("training campaign changed the per-seed budget")
        if not 1 <= int(campaign.get("measurement_workers_per_seed", 0)) <= MAX_MEASUREMENT_WORKERS:
            raise CampaignExecutionError("training campaign exceeded four measurement workers")
        delivery = spec.training_output / "delivery_policy.json"
        if campaign.get("delivery_file_sha256") != _sha256_file(delivery):
            raise CampaignExecutionError("delivery checkpoint hash differs from campaign receipt")
        library_binding = _solution_library_binding(spec.solution_library_manifest)
        if (campaign.get("solution_library_identity") != library_binding
                or campaign.get("action_registry_fingerprint")
                != library_binding["action_registry_fingerprint"]):
            raise CampaignExecutionError(
                "training campaign receipt has a different solution-library binding"
            )
        loaded_delivery = training.grpo.load_checkpoint(delivery)
        run_identity = loaded_delivery.extra.get("run_identity")
        if (loaded_delivery.extra.get("solution_library_identity") != library_binding
                or not isinstance(run_identity, Mapping)
                or run_identity.get("solution_library_identity") != library_binding
                or run_identity.get("action_registry_fingerprint")
                != library_binding["action_registry_fingerprint"]):
            raise CampaignExecutionError(
                "delivery checkpoint has a different solution-library binding"
            )
    elif stage.name == "payload_artifact_audit":
        audit = _load_json(spec.payload_audit_receipt, "complete payload audit")
        if (audit.get("schema") != aggregation.PAYLOAD_AUDIT_SCHEMA
                or audit.get("status") != "passed"
                or audit.get("failures") != []
                or audit.get("audit_sha256") != _sha256_bytes(_canonical_json({
                    key: value for key, value in audit.items()
                    if key != "audit_sha256"
                }))):
            raise CampaignExecutionError("complete payload audit did not pass cleanly")
    elif stage.name == "final_result_aggregation":
        detail = _load_json(spec.final_detail_output, "formal result details")
        summary = _load_json(spec.final_summary_output, "formal result summary")
        if (detail.get("schema") != "formal-weighted-grpo-result-details-v1"
                or detail.get("detail_sha256") != _sha256_bytes(_canonical_json({
                    key: value for key, value in detail.items()
                    if key != "detail_sha256"
                }))):
            raise CampaignExecutionError("formal detail output has the wrong schema")
        if (summary.get("schema") != "formal-weighted-grpo-result-summary-v1"
                or summary.get("status") != "passed_fail_closed_aggregation"
                or summary.get("detail_sha256") != detail.get("detail_sha256")
                or summary.get("summary_sha256") != _sha256_bytes(_canonical_json({
                    key: value for key, value in summary.items()
                    if key != "summary_sha256"
                }))):
            raise CampaignExecutionError("formal summary is not bound to passing details")


def _receipt_name(stage_name: str) -> str:
    return stage_name.replace("/", "__") + ".json"


def _run_stage(
    stage: CommandStage, *, campaign_root: Path, resume: bool,
    max_measurement_workers: int,
    solution_library_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    receipt_path = campaign_root / "stage_receipts" / _receipt_name(stage.name)
    if resume and _validate_resume_receipt(
        stage, receipt_path,
        solution_library_binding=solution_library_binding,
    ):
        return {"name": stage.name, "status": "resumed_complete"}
    if receipt_path.exists() and not resume:
        raise CampaignExecutionError(
            f"stage receipt already exists; use --resume: {receipt_path}"
        )
    for output in stage.expected_outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    log_path = campaign_root / "logs" / (stage.name.replace("/", "__") + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment["WEIGHTED_GRPO_MAX_MEASUREMENT_WORKERS"] = str(max_measurement_workers)
    if solution_library_binding is not None:
        expected_path = str(solution_library_binding["path"])
        expected_sha = str(solution_library_binding["file_sha256"])
        stage_environment = dict(stage.environment)
        if stage_environment.get(SOLUTION_LIBRARY_ENV) != expected_path:
            raise CampaignExecutionError(
                f"stage {stage.name} is not bound to the planned solution library path"
            )
        for key, expected in (
            (SOLUTION_LIBRARY_ENV, expected_path),
            (SOLUTION_LIBRARY_SHA256_ENV, expected_sha),
        ):
            inherited = os.environ.get(key)
            if inherited is not None and inherited != expected:
                raise CampaignExecutionError(
                    f"conflicting inherited {key} for stage {stage.name}"
                )
        environment[SOLUTION_LIBRARY_ENV] = expected_path
        environment[SOLUTION_LIBRARY_SHA256_ENV] = expected_sha
    environment.update(dict(stage.environment))
    if solution_library_binding is not None:
        # The hash is launcher-owned and is injected only after the manifest
        # exists; no command-spec environment can override it.
        environment[SOLUTION_LIBRARY_SHA256_ENV] = str(
            solution_library_binding["file_sha256"]
        )
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.run(
            stage.argv, cwd=stage.cwd, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            check=False, preexec_fn=_lower_child_priority,
        )
    if process.returncode != 0:
        failure = {
            "schema": STAGE_RECEIPT_SCHEMA,
            "stage": stage.name,
            "status": "failed",
            "stage_fingerprint": stage.fingerprint,
            "returncode": process.returncode,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "log": str(log_path),
        }
        _atomic_json(receipt_path, failure)
        raise CampaignExecutionError(
            f"stage {stage.name} failed with return code {process.returncode}; "
            f"see {log_path}"
        )
    outputs = [_path_record(path) for path in stage.expected_outputs]
    receipt = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "stage": stage.name,
        "status": "complete",
        "stage_fingerprint": stage.fingerprint,
        "returncode": 0,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path),
        "outputs": outputs,
    }
    if solution_library_binding is not None:
        receipt["solution_library_binding"] = dict(solution_library_binding)
    elif stage.name == "training_solution_library":
        receipt["solution_library_binding"] = _solution_library_binding(
            stage.expected_outputs[0]
        )
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_json(receipt))
    _atomic_json(receipt_path, receipt)
    return {"name": stage.name, "status": "complete", "receipt": str(receipt_path)}


def _lower_child_priority() -> None:
    try:
        os.nice(19)
    except OSError:
        pass


def _mark_stage_semantic_failure(
    stage: CommandStage, campaign_root: Path, error: BaseException,
) -> None:
    path = campaign_root / "stage_receipts" / _receipt_name(stage.name)
    current = _load_json(path, f"{stage.name} stage receipt")
    current["status"] = "failed"
    current["semantic_validation_error"] = str(error)
    current["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    current.pop("receipt_sha256", None)
    current["receipt_sha256"] = _sha256_bytes(_canonical_json(current))
    _atomic_json(path, current)


@contextmanager
def _campaign_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".campaign.lock"
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignExecutionError(
                f"another launcher owns this campaign: {root}"
            ) from error
        yield


def execute(spec: CampaignSpec, plan: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    with _campaign_lock(spec.campaign_root):
        fresh_admission = admit_campaign(spec)
        planned_admission = plan.get("admission")
        if not isinstance(planned_admission, Mapping) or any(
            fresh_admission.get(key) != planned_admission.get(key)
            for key in (
                "corpus_sha256", "calibration_sha256", "gate_bundle_sha256",
                "source_fingerprint", "action_registry_fingerprint",
            )
        ):
            raise CampaignExecutionError(
                "formal admission changed between planning and the locked campaign"
            )
        state_path = spec.campaign_root / "campaign_state.json"
        if state_path.exists() and not resume:
            raise CampaignExecutionError("campaign already exists; use --resume")
        if resume and state_path.exists():
            prior = _load_json(state_path, "campaign state")
            if (prior.get("schema") != STATE_SCHEMA
                    or prior.get("spec_sha256") != spec.spec_sha256
                    or prior.get("plan_sha256") != plan["plan_sha256"]):
                raise CampaignExecutionError(
                    "existing campaign state does not match this immutable plan"
                )
        state: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "status": "running",
            "spec_sha256": spec.spec_sha256,
            "plan_sha256": plan["plan_sha256"],
            "campaign_root": str(spec.campaign_root),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stages": [
                {"name": "prepared_data_verification", "status": "complete"},
                {"name": "formal_zlib6_calibration", "status": "complete"},
            ],
        }
        _atomic_json(spec.campaign_root / "campaign_plan.json", dict(plan))
        _atomic_json(state_path, state)
        try:
            for stage in _generated_stages(spec):
                # Reopen the immutable gate immediately before training and
                # again before frozen testing.  This catches source/artifact
                # drift without repeatedly hashing every prepared byte.
                if stage.name in {"three_seed_training", "frozen_six_weight_test"}:
                    admit_campaign(spec)
                library_binding = (
                    _solution_library_binding(spec.solution_library_manifest)
                    if stage.name in SOLUTION_LIBRARY_STAGE_NAMES else None
                )
                if (library_binding is not None
                        and library_binding["action_registry_fingerprint"]
                        != fresh_admission["action_registry_fingerprint"]):
                    raise CampaignExecutionError(
                        "solution library action registry differs from campaign admission"
                    )
                result = _run_stage(
                    stage, campaign_root=spec.campaign_root, resume=resume,
                    max_measurement_workers=spec.max_measurement_workers,
                    solution_library_binding=library_binding,
                )
                try:
                    _validate_core_stage_output(stage, spec)
                except BaseException as error:
                    _mark_stage_semantic_failure(stage, spec.campaign_root, error)
                    raise
                state["stages"].append(result)
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(state_path, state)
        except BaseException:
            state["status"] = "failed"
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(state_path, state)
            raise
        state["status"] = "complete"
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["state_sha256"] = _sha256_bytes(_canonical_json(state))
        _atomic_json(state_path, state)
        return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume are mutually exclusive")
    try:
        spec = load_spec(args.spec)
        admission = admit_campaign(spec)
        plan = build_plan(spec, admission)
        if args.plan_output is not None:
            output = args.plan_output.expanduser().resolve()
            _atomic_json(output, plan)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
            return 0
        state = execute(spec, plan, resume=args.resume)
    except (CampaignAdmissionError, CampaignExecutionError) as error:
        parser.exit(2, f"formal campaign refused: {error}\n")
    print(json.dumps({
        "schema": state["schema"], "status": state["status"],
        "campaign_root": state["campaign_root"],
        "state_sha256": state["state_sha256"],
    }, sort_keys=True))
    return 0


__all__ = [
    "CampaignAdmissionError", "CampaignExecutionError", "CampaignSpec",
    "CommandStage", "FORMAL_WEIGHTS", "SPEC_SCHEMA", "admit_campaign",
    "build_plan", "execute", "load_spec",
]


if __name__ == "__main__":
    raise SystemExit(main())
