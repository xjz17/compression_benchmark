#!/usr/bin/env python3
"""Warm-start pretraining admitted by a six-weight train-oracle hard gate.

The union is an immutable coverage requirement, not a claim that all its paths
are executable. This independent v7 controller keeps prior campaign sources and
ledgers immutable, measures every candidate separately for all six weights, and
refuses to start GPU work unless the finite train-only oracle strictly beats the
same-run direct Zstd baseline at every weight.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import run_sixweight_campaign_helpers_v2 as base


TOTAL_TRAINING_CAP_SECONDS = 1200.0
TRAINING_OUTER_HARD_CAP_SECONDS = 350.0
TRAINING_INTERNAL_SECONDS = 310.0
CLEANUP_MARGIN_SECONDS = 2.0
COVERAGE_STATUS = "partial_current_runtime_only"
STRONG_PATH_SCHEMA = "native-online-offline-combo-strong-registered-paths-v2"
STRONG_PATH_CANDIDATES = [
    "strong_registered_byte_shuffle4+lz4_exact_stages",
    "strong_registered_byte_shuffle8+lz4_exact_stages",
]
STRONG_REFERENCE_CHAIN_CANDIDATES = [
    "strong_registered_delta2+lz4_exact_stages",
    "strong_registered_delta_streamvbyte+lz4_exact_stages",
    "strong_registered_xor_previous+lz4_exact_stages",
    "strong_registered_xor_gorilla+lz4_exact_stages",
    "strong_registered_dictionary_words64+each_lz4_exact_stages",
    "strong_registered_for_minimum+each_lz4_exact_stages",
    "strong_registered_xor_semantic_ports+each_lz4_exact_stages",
    "strong_registered_bzip2_reference_chain",
]
TEACHER_TEMPLATE_COUNT = 102
EXPECTED_TRAIN_SAMPLES = 40
EXPECTED_VALIDATION_SAMPLES = 32
CHECKPOINT_MIN_RELATIVE_IMPROVEMENT = 0.005
TEACHER_WALL_DEADLINE_SECONDS = 2400.0
TEACHER_OUTER_TIMEOUT_SECONDS = 2460.0
E0 = 15.159782608695652
D0 = 7.023168103448276
SIX_WEIGHTS = [
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.4, 0.3, 0.3],
]
WEIGHT_KEYS = {
    (1.0, 0.0, 0.0): "100", (0.0, 1.0, 0.0): "010",
    (0.0, 0.0, 1.0): "001", (0.5, 0.5, 0.0): "550",
    (0.5, 0.0, 0.5): "505", (0.4, 0.3, 0.3): "433",
}


def objective(row, weight):
    ratio = float(row["compression_ratio"])
    encode = float(row["compression_ns_per_byte"])
    decode = float(row["decompression_ns_per_byte"])
    base.require(math.isfinite(ratio) and ratio > 0 and
                 math.isfinite(encode) and encode >= 0 and
                 math.isfinite(decode) and decode >= 0,
                 "non-finite or negative complete-frame metric")
    a, b, c = (float(value) for value in weight)
    result = a * ratio + b * encode / (encode + E0) + c * decode / (decode + D0)
    base.require(math.isfinite(result), "non-finite six-weight objective")
    return result


def _jsonl(path: Path):
    base.require(path.is_file(), f"required teacher artifact absent: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    base.require(rows, f"empty teacher artifact: {path}")
    return rows


def validate_sixweight_oracle_vs_zstd(teacher: Path,
                                      train_manifest: Path | None = None):
    """Require the finite train oracle to beat same-run Zstd at all weights.

    This is a necessary pre-GPU admission bound, not a model-quality result and
    not validation/test evidence. Each sample contributes in proportion to its
    original numeric bytes, matching the project-wide B_raw denominator.
    """
    candidate_path = teacher / "candidate_measurements.jsonl"
    baseline_path = teacher / "direct_baselines.jsonl"
    teacher_path = teacher / "teacher.jsonl"
    search_path = teacher / "search_space.json"
    candidates = _jsonl(candidate_path)
    baselines = _jsonl(baseline_path)
    base.require(teacher_path.is_file() and search_path.is_file(),
                 "teacher/search-space identity artifact absent")
    expected_weights = {tuple(weight) for weight in SIX_WEIGHTS}
    grouped = {}
    sample_bytes = {}
    for row in candidates:
        if row.get("status") != "ok" or row.get("split") != "train":
            continue
        weight = tuple(float(value) for value in row.get("weight", ()))
        if weight not in expected_weights:
            continue
        base.require(row.get("approximate_teacher_timing") is False and
                     row.get("time_reused_across_weights") is False and
                     tuple(float(value) for value in
                           row.get("measured_timing_weight", ())) == weight,
                     "six-weight oracle row reused or approximated timing")
        raw = int(row.get("original_numeric_bytes", 0))
        sample = str(row.get("sample_id", ""))
        base.require(sample and raw > 0, "invalid candidate sample/B_raw")
        if sample in sample_bytes:
            base.require(sample_bytes[sample] == raw,
                         "candidate B_raw changed within sample")
        sample_bytes[sample] = raw
        measured = objective(row, weight)
        recorded = float(row.get("objective", math.nan))
        base.require(math.isfinite(recorded) and
                     math.isclose(measured, recorded, rel_tol=1e-9, abs_tol=1e-9),
                     "candidate objective does not match measured R/E/D")
        grouped.setdefault((sample, weight), []).append((measured, row))
    base.require(sample_bytes, "no successful train candidate rows")
    manifest_binding = {}
    if train_manifest is not None:
        manifest = json.loads(train_manifest.read_bytes())
        rows = manifest.get("samples")
        base.require(manifest.get("split") == "train" and isinstance(rows, list) and
                     rows, "invalid train manifest for oracle gate")
        expected = {}
        for row in rows:
            sample = str(row.get("sample_id", ""))
            raw = int(row.get("B_raw", 0))
            base.require(sample and raw > 0 and sample not in expected,
                         "invalid or duplicate train-manifest sample/B_raw")
            expected[sample] = raw
        base.require(expected == sample_bytes,
                     "finite oracle sample IDs/B_raw differ from train manifest")
        canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        manifest_binding = {
            "train_manifest_sha256": base.sha(train_manifest),
            "expected_sample_ids_and_B_raw_sha256":
                hashlib.sha256(canonical).hexdigest(),
        }
    for sample in sample_bytes:
        for weight in expected_weights:
            base.require((sample, weight) in grouped,
                         f"finite oracle lacks successful row: {sample} {weight}")

    zstd = {}
    for row in baselines:
        if row.get("method") != "Zstd" or row.get("status") != "ok":
            continue
        sample = str(row.get("sample_id", ""))
        base.require(sample and sample not in zstd,
                     "duplicate or invalid same-run Zstd baseline")
        base.require(row.get("independent_decode") is True and
                     row.get("timing_scope") ==
                     "direct_native_complete_method_no_mlp",
                     "Zstd baseline lacks direct complete-frame cold decode")
        raw = int(row.get("original_numeric_bytes", 0))
        base.require(sample in sample_bytes and sample_bytes[sample] == raw,
                     "Zstd sample coverage/B_raw differs from train oracle")
        zstd[sample] = row
    base.require(set(zstd) == set(sample_bytes),
                 "same-run Zstd baseline does not exactly cover train oracle")

    total_raw = sum(sample_bytes.values())
    comparisons = {}
    all_passed = True
    for weight_list in SIX_WEIGHTS:
        weight = tuple(weight_list)
        oracle_weighted = 0.0
        zstd_weighted = 0.0
        per_sample_wins = 0
        winners = {}
        for sample in sorted(sample_bytes):
            winner_value, winner = min(
                grouped[(sample, weight)],
                key=lambda item: (item[0], float(item[1]["compression_ratio"]),
                                  str(item[1].get("candidate", ""))))
            baseline_value = objective(zstd[sample], weight)
            raw = sample_bytes[sample]
            oracle_weighted += raw * winner_value
            zstd_weighted += raw * baseline_value
            per_sample_wins += winner_value < baseline_value
            winners[sample] = str(winner.get("candidate", ""))
        oracle_value = oracle_weighted / total_raw
        zstd_value = zstd_weighted / total_raw
        base.require(zstd_value > 0, "same-run Zstd objective is not positive")
        passed = oracle_value < zstd_value
        all_passed = all_passed and passed
        key = WEIGHT_KEYS[weight]
        comparisons[key] = {
            "weight": list(weight),
            "finite_oracle_J": oracle_value,
            "zstd_J": zstd_value,
            "oracle_over_zstd": oracle_value / zstd_value,
            "strictly_better": passed,
            "per_sample_wins": per_sample_wins,
            "winner_candidates": winners,
        }
    receipt = {
        "schema": "train-only-sixweight-finite-oracle-zstd-gate-v1",
        "status": "complete",
        "passed": all_passed,
        "selection_scope": "train_only_finite_candidate_oracle_pre_gpu",
        "aggregation": "original_numeric_bytes_weighted_mean_J",
        "objective": "a*R+b*E/(E+E0)+c*D/(D+D0)",
        "E0": E0, "D0": D0,
        "sample_count": len(sample_bytes),
        "original_numeric_bytes": total_raw,
        "exact_separate_timing_per_weight": True,
        "same_run_direct_zstd": True,
        "teacher_sha256": base.sha(teacher_path),
        "candidate_measurements_sha256": base.sha(candidate_path),
        "direct_baselines_sha256": base.sha(baseline_path),
        "search_space_sha256": base.sha(search_path),
        **manifest_binding,
        "comparisons": comparisons,
    }
    return receipt


def admit_previous_budget(document):
    previous_seconds = float(document["combined_training_wall_seconds"])
    base.require(document.get("status") == "complete" and
                 0 < previous_seconds < 848 and
                 previous_seconds + TRAINING_OUTER_HARD_CAP_SECONDS +
                 CLEANUP_MARGIN_SECONDS < TOTAL_TRAINING_CAP_SECONDS,
                 "prior cumulative budget does not permit this bounded continuation")
    return previous_seconds


def union_identity(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    base.require(document.get("schema") ==
                 "operator86-pdf102-candidate-union-v1" and
                 isinstance(document.get("payload"), dict),
                 "unexpected union catalog schema")
    canonical = json.dumps(document["payload"], sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False,
                           allow_nan=False).encode("utf-8")
    space_sha256 = hashlib.sha256(canonical).hexdigest()
    base.require(space_sha256 == document.get("space_sha256"),
                 "union catalog payload hash drift")
    return {"path": path.resolve(), "file_sha256": base.sha(path),
            "space_sha256": space_sha256}


def validate_initialization(path: Path, training: Path):
    document = json.loads(path.read_bytes())
    model = Path(str(document.get("model", ""))).resolve()
    checkpoint = model.with_suffix(".json")
    root = training.resolve()
    contract = document.get("contract", {})
    base.require(document.get("schema") == "native-combo-quality-training-v6" and
                 document.get("test_used") is False and
                 document.get("architecture") == [49, 128, 128, 6373] and
                 contract.get("registry_fingerprint") == base.REGISTRY and
                 int(contract.get("action_count", -1)) == 6373 and
                 model.is_relative_to(root) and model.is_file() and
                 checkpoint.is_relative_to(root) and checkpoint.is_file() and
                 base.sha(model) == document.get("model_sha256"),
                 "fresh initialization identity is absent or changed")
    return document, model, checkpoint


def validate_union_teacher_binding(teacher: Path, union):
    expected_space = union["space_sha256"]
    expected_file = union["file_sha256"]
    expected_path = str(union["path"])
    json_files = (teacher / "search_space.json", teacher / "summary.json",
                  teacher / "teacher.receipt.json")
    for path in json_files:
        document = json.loads(path.read_bytes())
        extension = document.get("schema_extension", {})
        base.require(document.get("union_space_sha256") == expected_space and
                     document.get("coverage_status") == COVERAGE_STATUS and
                     document.get("full_union_coverage_claimed") is False and
                     extension.get("union_space_sha256") == expected_space and
                     extension.get("union_catalog") == expected_path and
                     extension.get("union_catalog_sha256") == expected_file and
                     extension.get("coverage_status") == COVERAGE_STATUS and
                     extension.get("full_union_training_ready") is False and
                     extension.get("schema") == "native-online-offline-combo-sixweight-typed-multiflow-v2" and
                     extension.get("candidate_counts") == {
                         "all_76_v6_templates": 76, "registered_specialist_seeds": 7,
                         "typed_multiflow_registered_connections": 9, "total": 92} and
                     document.get("strong_path_schema") == STRONG_PATH_SCHEMA and
                     document.get("strong_path_candidates") == STRONG_PATH_CANDIDATES and
                     document.get("strong_path_candidate_count") == 10 and
                     document.get("new_reference_chain_candidate_count") == 8 and
                     document.get("strong_path_total_template_count") ==
                     TEACHER_TEMPLATE_COUNT and
                     document.get("strong_reference_chain_candidates") ==
                     STRONG_REFERENCE_CHAIN_CANDIDATES and
                     document.get("strong_path_action_registry_changed") is False and
                     document.get("standard_bzip2_claimed") is False and
                     document.get("standard_zstd_claimed") is False and
                     document.get("standard_brotli_claimed") is False and
                     document.get("upstream_wire_equivalence_claimed") is False,
                     f"teacher union binding mismatch: {path}")
        if path.name == "search_space.json":
            base.require(int(document.get("candidate_limit", -1)) ==
                         TEACHER_TEMPLATE_COUNT and
                         int(document.get("materialized_candidate_count", -1)) ==
                         TEACHER_TEMPLATE_COUNT and
                         len(document.get("plans", ())) == TEACHER_TEMPLATE_COUNT and
                         document.get("reuse_candidate_time_weights") is False and
                         document.get("approximate_teacher_timing") is False and
                         document.get("measured_candidate_timing_weights") == SIX_WEIGHTS,
                         "strong teacher lacks exact six-weight timing for 102 templates")
    counts = {}
    for filename in ("candidate_measurements.jsonl", "teacher.jsonl"):
        rows = [json.loads(line) for line in
                (teacher / filename).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        base.require(rows and all(
            row.get("union_space_sha256") == expected_space and
            row.get("coverage_status") == COVERAGE_STATUS and
            row.get("strong_path_schema") == STRONG_PATH_SCHEMA for row in rows),
            f"teacher JSONL union binding mismatch: {filename}")
        counts[filename] = len(rows)
    return {"union_space_sha256": expected_space,
            "union_catalog_sha256": expected_file,
            "coverage_status": COVERAGE_STATUS,
            "strong_path_schema": STRONG_PATH_SCHEMA,
            "teacher_template_count": TEACHER_TEMPLATE_COUNT,
            "action_registry_changed": False, "row_counts": counts,
            "artifact_sha256": {path.name: base.sha(path) for path in json_files}}


def choose_phase_candidate(candidates, phase, chooser):
    eligible = [dict(row) for row in candidates if row.get("phase") == phase]
    base.require(eligible, f"no completed eligible {phase} validation candidate")
    selected = chooser(eligible)
    base.require(selected in eligible, f"{phase} chooser escaped eligible candidates")
    return selected, eligible


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--oracle-gate-only":
        receipt = validate_sixweight_oracle_vs_zstd(
            Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
        base.write(Path(sys.argv[4]).resolve(), receipt)
        if not receipt["passed"]:
            raise SystemExit(2)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--gpu-worker":
        base.gpu_worker(Path(sys.argv[2]), sys.argv[3:])
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--previous-state", type=Path, required=True)
    p.add_argument("--union", type=Path, required=True)
    p.add_argument("--input-plan", type=Path, required=True)
    p.add_argument("--baseline-cli", type=Path, required=True)
    p.add_argument("--initial-checkpoint", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "campaign_state.json"
    base.require(not state_path.exists(), "fresh run directory required")
    previous = json.loads(a.previous_state.read_bytes())
    previous_seconds = admit_previous_budget(previous)
    union = union_identity(a.union)
    code, build = root / "code", root / "build"
    train, valid = (root / "data" / f"{split}_blocks_manifest.json"
                    for split in ("train", "validation"))
    reference = root / "reference/reference.json"
    catalog = code / "cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json"
    profile = code / "cpp/native_pipeline_v1/experiment_no_cmix_nncp_v1.json"
    trainer = code / "scripts/train_combo_policy_sixweight_v3.py"
    scorer = code / "scripts/score_sixweight_validation_dual_view_v1.py"
    teacher_script = code / "scripts/offline_combo_teacher_strong_paths_v2.py"
    training, teacher = root / "training", root / "teacher"
    train_samples = len(base.manifest_rows(train))
    validation_samples = len(base.manifest_rows(valid))
    base.require(train_samples == EXPECTED_TRAIN_SAMPLES and
                 validation_samples == EXPECTED_VALIDATION_SAMPLES,
                 "v7 requires the admitted full 40-train/32-validation cohort")
    for path in (trainer, scorer, teacher_script):
        base.require(path.is_file(), f"required v7 source absent: {path}")
    base.require(not training.exists() and not teacher.exists(), "output already exists")
    state = {"schema": "sixweight-pretrain-campaign-v7-strong-path-oracle-gated", "status": "preparing",
        "started_at": base.now(), "scope": COVERAGE_STATUS,
        "full_union_native_admitted": False, "fresh_random_initialization": False,
        "warm_start_checkpoint": str(a.initial_checkpoint.resolve()),
        "warm_start_checkpoint_sha256": base.sha(a.initial_checkpoint),
        "registry_fingerprint": base.REGISTRY, "action_count": 6373,
        "gpu_uuid": base.GPU, "cuda_allocator_fraction": 0.1, "cpu": 25,
        "training_outer_hard_cap_seconds": TRAINING_OUTER_HARD_CAP_SECONDS,
        "training_internal_seconds": TRAINING_INTERNAL_SECONDS,
        "previous_combined_training_wall_seconds": previous_seconds,
        "previous_campaign": str(a.previous_state.resolve()),
        "previous_campaign_sha256": base.sha(a.previous_state),
        "union_path": str(union["path"]),
        "union_sha256": union["file_sha256"],
        "union_space_sha256": union["space_sha256"],
        "teacher_schema": STRONG_PATH_SCHEMA,
        "teacher_template_count": TEACHER_TEMPLATE_COUNT,
        "teacher_exact_sixweight_timing": True,
        "teacher_train_oracle_gate":
        "strict_all_six_B_raw_weighted_J_below_same_run_direct_Zstd",
        "teacher_wall_deadline_seconds": TEACHER_WALL_DEADLINE_SECONDS,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "selection_rule": "max_six_weight_relative_J_large_and_all_domain_macro_plus_0.01_mean_v1",
        "checkpoint_min_relative_improvement": CHECKPOINT_MIN_RELATIVE_IMPROVEMENT,
        "trainer_sha256": base.sha(trainer), "dual_view_scorer_sha256": base.sha(scorer),
        "test_used_for_training_or_selection": False, "phases": []}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=base.GPU, OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    active = None

    def persist(status):
        state.update(status=status, updated_at=base.now())
        base.write(state_path, state)

    def run(name, argv, timeout):
        nonlocal active
        def started(process):
            nonlocal active
            active = process
            state["active_phase"] = {"name": name, "pid": process.pid,
                                     "started_at": base.now()}
            persist(name)
        phase = base.execute_process_group(name, argv, timeout, cwd=code,
            environment=env, log=root / f"{name}.log", on_start=started)
        active = None
        state.pop("active_phase", None)
        state["phases"].append(phase)
        persist(name + "_finished")
        return phase

    def must_run(name, argv, timeout):
        phase = run(name, argv, timeout)
        base.require(phase["returncode"] == 0 and not phase["timed_out"],
                     f"{name} failed or timed out")
        return phase

    def interrupted(signum, _frame):
        raise InterruptedError(f"controller signal {signum}")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)

    common = ["--data", str(train), "--data", str(valid), "--reference", str(reference),
        "--server", str(build / "native_online_training_server"),
        "--catalog", str(catalog), "--profile", str(profile)]
    try:
        persist("preparing")
        must_run("validation_baseline", ["taskset", "-c", "25", sys.executable,
            str(root / "driver/prepare_sixweight_validation_v1.py"), "--code", str(code),
            "--data", str(train), "--data", str(valid), "--reference", str(reference),
            "--baseline-cli", str(a.baseline_cli),
            "--output", str(root / "validation_baseline")], 90)
        validation_baseline = root / "validation_baseline/reference.json"
        trainer_extra = ["--validation-baseline", str(validation_baseline),
                         "--initial-checkpoint", str(a.initial_checkpoint)]
        must_run("initialization", ["taskset", "-c", "25", sys.executable,
            str(trainer), *common, *trainer_extra, "--output", str(training), "--phase", "initialize",
            "--device", "cpu", "--seed", "0"], 90)
        initialization = training / "initialization.json"
        init, initial_model, initial_checkpoint = validate_initialization(
            initialization, training)
        warm_native = a.initial_checkpoint.with_suffix(".wgpn")
        warm_payload = json.loads(a.initial_checkpoint.read_bytes())["payload"]
        init_payload = json.loads(initial_checkpoint.read_bytes())["payload"]
        base.require(warm_native.is_file() and
                     warm_payload["identity"] == init_payload["identity"] and
                     warm_payload["parameters"] == init_payload["parameters"],
                     "initialization changed warm-start policy identity or parameters")
        state["warm_start_parameter_guard"] = {
            "policy_identity_equal": True, "parameters_equal": True,
            "source_native_sha256": base.sha(warm_native),
            "reexported_native_sha256": init["model_sha256"],
            "file_hash_difference_expected": "WGPN binds checkpoint metadata SHA"}

        must_run("teacher", ["taskset", "-c", "25", sys.executable,
            str(teacher_script), *common, "--model", str(initial_model),
            "--baseline-cli", str(a.baseline_cli),
            "--cold-cli", str(build / "native_graph/native_graph_request_cli"),
            "--output", str(teacher), "--cpu", "25", "--candidate-limit",
            str(TEACHER_TEMPLATE_COUNT), "--max-train-blocks", str(train_samples),
            "--deadline-seconds", str(TEACHER_WALL_DEADLINE_SECONDS),
            "--warmups", "0", "--repetitions", "1",
            "--require-quality-gate", "--union-catalog", str(a.union)],
            TEACHER_OUTER_TIMEOUT_SECONDS)
        union_teacher = validate_union_teacher_binding(teacher, union)
        oracle_gate = validate_sixweight_oracle_vs_zstd(teacher, train)
        base.write(root / "train_oracle_zstd_gate.json", oracle_gate)
        state["sixweight_teacher_oracle_gate"] = oracle_gate
        persist("teacher_oracle_gate_finished")
        base.require(oracle_gate["passed"],
                     "finite train oracle does not strictly beat same-run Zstd "
                     "for all six weights; GPU training refused")
        trainer_module = base.import_trainer(trainer)
        validation_baseline_document = json.loads(validation_baseline.read_bytes())
        trainer_module.VALIDATION_BASELINE = {row["sample_id"]: row for row in
            validation_baseline_document["rows"]}
        trainer_module.VALIDATION_BASELINE_DOC = validation_baseline_document
        base.require(trainer_module.SELECTION == state["selection_rule"],
                     "trainer dual-view selection rule drift")
        base.SELECTION = trainer_module.SELECTION
        admission = base.admit_teacher(trainer_module, teacher / "teacher.jsonl",
                                      train, valid, reference)
        base.require(admission["matrix"].get("teacher_model_sha256") ==
                     init["model_sha256"],
                     "teacher was not measured with this warm-start initialization")
        state["teacher_admission"] = {k: admission[k] for k in
            ("matrix", "evidence", "summary_sha256", "contract_sha256")}
        state["union_teacher_binding"] = union_teacher
        frozen = base.freeze([Path(__file__), Path(base.__file__), trainer, scorer,
            teacher_script, a.previous_state,
            a.initial_checkpoint.parent / "transfer_receipt.json",
            a.union, a.input_plan, a.baseline_cli, a.initial_checkpoint, warm_native,
            root / "driver/prepare_sixweight_validation_v1.py", validation_baseline,
            initialization, initial_model, initial_checkpoint,
            root / "train_oracle_zstd_gate.json",
            build / "native_online_training_server", build / "native_online_policy_server",
            build / "native_graph/native_graph_request_cli", *base.files_under(root / "data"),
            *base.files_under(root / "reference"), *base.files_under(teacher),
            *base.files_under(code / "scripts", suffixes={".py"}),
            *base.files_under(code / "cpp", suffixes={".cpp", ".hpp", ".h", ".json", ".cmake"}),
            *admission["search_artifact_files"]])
        base.write(root / "frozen_inputs.json", frozen)
        state["frozen_inputs_sha256"] = base.sha(root / "frozen_inputs.json")
        base.verify_frozen(frozen)
        lines = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free",
            "--format=csv,noheader,nounits"], text=True).splitlines()
        free = [int(line.split(",")[1]) for line in lines
                if line.split(",")[0].strip() == base.GPU]
        # Full 845029-parameter, batch128 Adam probe used 36 MiB tensors and
        # about 1 GiB context. Reserve at least 3 GiB before this shared run.
        base.require(len(free) == 1 and free[0] >= 3072,
                     "GPU7 needs 3 GiB free for this measured small-model pilot")
        state["gpu_free_mib_at_training_start"] = free[0]
        phase = run("training", [sys.executable, str(Path(__file__).resolve()),
            "--gpu-worker", str(trainer), *common, *trainer_extra, "--output", str(training),
            "--teacher", str(teacher / "teacher.jsonl"), "--phase", "train",
            "--cpu", "25", "--seed", "0", "--device", "cuda:0",
            "--budget-seconds", str(TRAINING_INTERNAL_SECONDS),
            "--sft-seconds", "480", "--sft-steps", "1800",
            "--batch-size", "128", "--grpo-iterations", "0"], TRAINING_OUTER_HARD_CAP_SECONDS)
        state["training_wall_seconds"] = phase["wall_seconds"]
        state["combined_training_wall_seconds"] = previous_seconds + phase["wall_seconds"]
        base.require(state["combined_training_wall_seconds"] <= 1200,
                     "cumulative training budget exceeded")
        selection, status, source = base.select_trainer_output(training,
            phase["returncode"], phase["timed_out"], trainer_module)
        verified = base.validate_selected_candidate(selection, training, valid,
            trainer_module, expected_data_fingerprints=admission["policy_data_fingerprints"],
            expected_root_fingerprints=admission["policy_root_fingerprints"])
        base.verify_frozen(frozen)
        def export_delivery(destination, label, selected, selected_verified,
                            selection_source, training_status):
            base.require(not destination.exists(), f"delivery exists: {destination}")
            destination.mkdir()
            shutil.copy2(selected_verified["model"], destination / "policy.wgpn")
            shutil.copy2(selected_verified["checkpoint"], destination / "policy.json")
            base.require(base.sha(destination / "policy.wgpn") ==
                         selected_verified["model_sha256"] and
                         base.sha(destination / "policy.json") ==
                         selected_verified["checkpoint_sha256"],
                         "delivery copy changed selected checkpoint")
            export = {"schema": "partial-union-pretrain-export-v2-strong-dual-view",
                "checkpoint_label": label, "scope": state["scope"],
                "full_union_native_admitted": False,
                "union_sha256": state["union_sha256"],
                "union_space_sha256": state["union_space_sha256"],
                "selected": selected, "selection_rule": base.SELECTION,
                "selected_validation": str(selected_verified["validation"]),
                "selected_validation_sha256": selected_verified["validation_sha256"],
                "validation_summary": selected_verified["validation_summary"],
                "training_status": training_status,
                "selection_source": str(selection_source),
                "selection_source_sha256": base.sha(selection_source),
                "native_model_sha256": base.sha(destination / "policy.wgpn"),
                "source_checkpoint_sha256": base.sha(destination / "policy.json"),
                "test_used_for_selection": False}
            base.write(destination / "export_receipt.json", export)
            receipt = {"schema": "partial-union-pretrain-training-receipt-v2-strong-dual-view",
                "checkpoint_label": label, "fresh_random_initialization": False,
                "warm_start_checkpoint": str(a.initial_checkpoint.resolve()),
                "warm_start_checkpoint_sha256": base.sha(a.initial_checkpoint),
                "seed": 0, "scope": state["scope"],
                "teacher_admission": state["teacher_admission"],
                "union_teacher_binding": state["union_teacher_binding"],
                "teacher_sha256": base.sha(teacher / "teacher.jsonl"),
                "frozen_inputs_sha256": state["frozen_inputs_sha256"],
                "training_wall_seconds": phase["wall_seconds"],
                "combined_training_wall_seconds": state["combined_training_wall_seconds"],
                "status": training_status, "selection": export,
                "test_used_for_selection": False}
            base.write(destination / "training_receipt.json", receipt)
            base.write(destination / "matched_credential.json", base.credential(
                base.sha(destination / "policy.wgpn"), base.sha(destination / "policy.json"),
                base.sha(catalog), base.sha(profile),
                base.sha(destination / "training_receipt.json"),
                base.sha(destination / "export_receipt.json"),
                selected_verified["data_fingerprints"]))
            return export

        all_candidates = base.completed_validation_candidates(training)
        sft_selection, sft_candidates = choose_phase_candidate(
            all_candidates, "sft", trainer_module.choose_candidate)
        sft_source = root / "sft_selection.json"
        base.write(sft_source, {"schema": "partial-union-sft-selection-v2-dual-view",
            "selection_rule": base.SELECTION, "selected": sft_selection,
            "candidates": sft_candidates, "test_used_for_selection": False})
        sft_verified = base.validate_selected_candidate(
            sft_selection, training, valid, trainer_module,
            expected_data_fingerprints=admission["policy_data_fingerprints"],
            expected_root_fingerprints=admission["policy_root_fingerprints"])
        sft_export = export_delivery(root / "delivery_sft", "sft_only",
                                     sft_selection, sft_verified, sft_source,
                                     "completed_sft_validation")
        export = export_delivery(root / "delivery", "final_selected",
                                 selection, verified, source, status)
        state["sft_selection"] = sft_export
        state["selection"] = export
        persist("model_selected")
        # Frozen-test comparisons are a separate invocation after this receipt.
        # They cannot alter the selected checkpoint or the optimizer.
        base.verify_frozen(frozen)
        persist("complete" if status == "complete" else "complete_from_partial_timeout_checkpoint")
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        persist("failed_or_interrupted")
        raise
    finally:
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            active.wait()


if __name__ == "__main__":
    main()
