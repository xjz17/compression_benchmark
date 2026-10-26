#!/usr/bin/env python3
"""Bounded strong-teacher distillation and matched native GRPO refinement.

The online action IDs are unchanged; the semantic mask is versioned.  This driver never
discovers recipes during validation and never uses held-out test inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import online_matched_train_v6 as core
import score_sixweight_validation_dual_view_v1 as dual_view

SCHEMA = "native-combo-quality-training-v6"
PRIMARY_WEIGHTS = tuple(tuple(w) for w in core.FIXED_WEIGHTS)
SELECTION = dual_view.SELECTION_RULE
VALIDATION_BASELINE = {}
VALIDATION_BASELINE_DOC = {}
WEIGHT_NAMES = {
    (1.0, 0.0, 0.0): "100",
    (0.0, 1.0, 0.0): "010",
    (0.0, 0.0, 1.0): "001",
    (0.5, 0.5, 0.0): "550",
    (0.5, 0.0, 0.5): "505",
    (0.4, 0.3, 0.3): "433",
}


def write(path, value):
    core._atomic_json(Path(path), value)


def hierarchy_masses(keys):
    """Equal mass at each occupied level of a tuple-key hierarchy."""
    keys = tuple(dict.fromkeys(keys))
    core.require(keys, "empty balancing hierarchy")
    children = defaultdict(set)
    for key in keys:
        for index, value in enumerate(key):
            children[key[:index]].add(value)
    return {key: math.prod(1.0 / len(children[key[:index]])
                          for index in range(len(key))) for key in keys}


def supervised_weights(records, blocks):
    """Balance occupied sizes first, then domains and substantive decisions."""
    by_id = {block.sample_id: block for block in blocks}
    trajectories = defaultdict(list)
    for index, row in enumerate(records):
        block = by_id[row["sample_id"]]
        key = (block.bucket, block.domain, block.dataset_id,
               tuple(row["weight"]), row["sample_id"])
        trajectories[key].append(index)
    masses = hierarchy_masses(trajectories)
    result = np.zeros(len(records), dtype=np.float64)
    for key, indices in trajectories.items():
        ordered = sorted(indices, key=lambda i: int(records[i]["step"]))
        # Every admitted teacher has numeric root0 and bitmap root1. The
        # mandatory first action closes the bitmap, not the numeric input.
        core.require(len(ordered) >= 2 and
                     int(records[ordered[0]]["action_id"]) == core.FLOW_STOP_BEGIN + 1,
                     "numeric-decision weighting requires the declared two-root teacher")
        result[ordered[0]] = masses[key] * .02
        if len(ordered) == 2:
            result[ordered[1]] = masses[key] * .98
        else:
            result[ordered[1]] = masses[key] * .70
            result[ordered[2:]] = masses[key] * .28 / (len(ordered) - 2)
    core.require(np.all(result > 0) and math.isclose(float(result.sum()), 1),
                 "invalid trajectory weight mass")
    result *= len(result)
    return result, {"schema": "size-domain-numeric-decision-balanced-ce-v1",
                    "domains": sorted({key[1] for key in trajectories}),
                    "size_buckets": sorted({key[0] for key in trajectories}),
                    "bitmap_prefix_mass": .02,
                    "first_numeric_mass_with_continuations": .70,
                    "records": len(records), "trajectories": len(trajectories),
                    "weight_sum": float(result.sum())}


def validate_strong_teacher(records, blocks):
    """Reject partial, old-quick, duplicated, or internally broken trajectories."""
    expected_samples = {block.sample_id for block in blocks}
    expected_weights = set(WEIGHT_NAMES)
    groups = defaultdict(list)
    model_hashes = set()
    for row in records:
        sample = str(row["sample_id"])
        weight = tuple(float(value) for value in row["weight"])
        core.require(sample in expected_samples and weight in expected_weights,
                     "teacher contains an unexpected sample or weight")
        core.require(isinstance(row.get("offline_candidate"), str) and
                     row["offline_candidate"], "teacher is not a strong v4 offline label")
        recipe = tuple(int(value) for value in row["recipe_action_ids"])
        metrics = row.get("teacher_metrics", {})
        core.require(recipe and all(math.isfinite(float(metrics[key])) for key in
                     ("compression_ratio", "compression_ns_per_byte",
                      "decompression_ns_per_byte")), "invalid teacher recipe/metrics")
        model_hashes.add(str(row.get("teacher_model_sha256", "")))
        groups[sample, weight].append(row)
    core.require(len(model_hashes) == 1 and len(next(iter(model_hashes))) == 64,
                 "teacher model identity drift")
    core.require(set(groups) == {(sample, weight) for sample in expected_samples
                                 for weight in expected_weights},
                 "teacher lacks the complete train-sample by six-weight matrix")
    ordinary_lengths = Counter()
    for trajectory in groups.values():
        ordered = sorted(trajectory, key=lambda row: int(row["step"]))
        recipe = tuple(int(value) for value in ordered[0]["recipe_action_ids"])
        core.require([int(row["step"]) for row in ordered] == list(range(len(ordered))) and
                     all(tuple(int(value) for value in row["recipe_action_ids"]) == recipe
                         for row in ordered) and
                     tuple(int(row["action_id"]) for row in ordered) == recipe,
                     "teacher trajectory is incomplete or inconsistent")
        ordinary_lengths[sum(0 < action < core.FLOW_STOP_BEGIN for action in recipe)] += 1
    return {"schema": "strong-offline-teacher-matrix-v4",
            "samples": len(expected_samples), "weights": len(expected_weights),
            "trajectories": len(groups), "records": len(records),
            "teacher_model_sha256": next(iter(model_hashes)),
            "ordinary_length_histogram": {str(key): value
                                          for key, value in sorted(ordinary_lengths.items())}}


def load_strong_teacher_evidence(teacher, records, blocks, contract):
    """Bind SFT to the completed train-only v4 search and its quality proof."""
    receipt_path = teacher.with_name(teacher.stem + ".receipt.json")
    summary_path = teacher.parent / "summary.json"
    search_path = teacher.parent / "search_space.json"
    cold_path = teacher.parent / "winner_cold_decode_receipts.jsonl"
    for path in (receipt_path, summary_path, search_path, cold_path):
        core.require(path.is_file(), f"strong teacher evidence absent: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    search = json.loads(search_path.read_text(encoding="utf-8"))
    sample_count = len({block.sample_id for block in blocks})
    core.require(receipt.get("schema") ==
                 "native-online-offline-combo-teacher-receipt-v6" and
                 receipt.get("teacher_sha256") == core.digest(teacher.read_bytes()) and
                 int(receipt.get("records", -1)) == len(records) and
                 int(receipt.get("train_samples", -1)) == sample_count and
                 int(receipt.get("validation_records", -1)) == 0 and
                 receipt.get("contract_sha256") == contract.fingerprint and
                 receipt.get("search_space_sha256") == core.digest(search_path.read_bytes()),
                 "strong teacher receipt mismatch")
    proof = summary.get("quality_gate", {})
    core.require(summary.get("schema") == "native-online-offline-combo-teacher-v6" and
                 summary.get("completed") is True and
                 summary.get("validation_evaluated") is False and
                 int(summary.get("teacher_records", -1)) == len(records) and
                 proof.get("status") == "complete" and proof.get("passed") is True and
                 int(proof.get("common_samples", -1)) == sample_count,
                 "strong teacher quality/coverage gate failed")
    core.require(search.get("schema") == "native-online-offline-combo-space-v6" and
                 search.get("train_only_teacher") is True and
                 search.get("validation_evaluation_labels_written") is False and
                 search.get("online_candidate_search") is False,
                 "strong teacher search boundary mismatch")
    cold = [json.loads(line) for line in cold_path.read_text().splitlines() if line.strip()]
    expected_keys = {(str(row["sample_id"]), tuple(row["recipe_action_ids"]))
                     for row in records}
    cold_keys = {(str(row["sample_id"]), tuple(row["recipe_action_ids"])) for row in cold}
    raw_by_id = {block.sample_id: block.original_numeric_bytes for block in blocks}
    core.require(receipt.get("winner_cold_decode_receipts_sha256") ==
                 core.digest(cold_path.read_bytes()) and
                 int(receipt.get("unique_winner_frames_cold_decoded", -1)) == len(cold) and
                 len(cold) == len(expected_keys) and cold_keys == expected_keys and
                 all(row.get("schema") == "native-online-winner-cold-decode-v4" and
                     row.get("independent_process") is True and
                     row.get("bit_exact_roots") is True and
                     row.get("original_numeric_bytes") == raw_by_id[row["sample_id"]]
                     for row in cold), "strong teacher cold-decode coverage mismatch")
    for row in cold:
        frame = Path(row["frame_path"])
        core.require(frame.is_file() and frame.stat().st_size == row["frame_bytes"] and
                     core.digest(frame.read_bytes()) == row["frame_sha256"],
                     "strong teacher cold-decoded frame changed")
    return {"receipt": str(receipt_path.resolve()),
            "receipt_sha256": core.digest(receipt_path.read_bytes()),
            "summary": str(summary_path.resolve()),
            "summary_sha256": core.digest(summary_path.read_bytes()),
            "search_space": str(search_path.resolve()),
            "search_space_sha256": core.digest(search_path.read_bytes()),
            "cold_decode_receipts": str(cold_path.resolve()),
            "cold_decode_receipts_sha256": core.digest(cold_path.read_bytes())}


class DomainFirstSampler:
    """Domains -> datasets -> occupied sizes -> blocks, each level uniform."""
    def __init__(self, blocks):
        self.groups = defaultdict(list)
        for block in blocks:
            self.groups[block.domain, block.dataset_id, block.bucket].append(block)
        self.keys = sorted(self.groups)
        masses = hierarchy_masses(self.keys)
        self.probabilities = np.asarray([masses[key] for key in self.keys])

    def sample(self, rng):
        key = self.keys[int(rng.choice(len(self.keys), p=self.probabilities))]
        rows = self.groups[key]
        return rows[int(rng.integers(len(rows)))]

    def receipt(self):
        return {"schema": "domain-dataset-size-block-sampler-v1", "groups": [
            {"domain": key[0], "dataset": key[1], "size_bucket": key[2],
             "blocks": len(self.groups[key]), "mass": float(mass)}
            for key, mass in zip(self.keys, self.probabilities)]}


def macro_mean(rows, field):
    groups = defaultdict(list)
    for row in rows:
        key = (str(row["domain"]), str(row["dataset_id"]), str(row["size_bucket"]))
        groups[key].append(float(row[field]))
    masses = hierarchy_masses(groups)
    return float(sum(masses[key] * np.mean(values) for key, values in groups.items()))


def choose_candidate(candidates):
    """Require 0.5% improvement; retain the earlier checkpoint within noise."""
    core.require(candidates, "no eligible validation candidate")
    chosen = candidates[0]
    for candidate in candidates[1:]:
        threshold = max(abs(chosen["score"]) * 0.005, 1e-9)
        if candidate["score"] < chosen["score"] - threshold:
            chosen = candidate
    return chosen


def summarize(rows):
    core.require(rows, "validation summary is empty")
    expected_weights = set(WEIGHT_NAMES)
    by_pair = {}
    sample_bytes = {}
    for row in rows:
        sample = str(row["sample_id"])
        weight = tuple(float(value) for value in row["weight"])
        raw = int(row["B_raw"])
        core.require(weight in expected_weights and raw > 0 and
                     math.isfinite(float(row["J_total"])),
                     "invalid validation row")
        core.require((sample, weight) not in by_pair,
                     "duplicate validation sample/weight row")
        core.require(sample not in sample_bytes or sample_bytes[sample] == raw,
                     "validation B_raw drift across weights")
        by_pair[sample, weight] = row
        sample_bytes[sample] = raw
    core.require(set(by_pair) == {(sample, weight) for sample in sample_bytes
                                  for weight in expected_weights},
                 "validation lacks the complete six-weight matrix")
    large_samples = {sample for sample, raw in sample_bytes.items() if raw >= 65536}
    core.require(large_samples, "primary validation requires real >=64KiB blocks")
    primary_by_weight = {}
    for weight in PRIMARY_WEIGHTS:
        selected = [by_pair[sample, weight] for sample in sorted(large_samples)]
        successful = [row for row in selected if row["status"] == "ok"]
        primary_by_weight[WEIGHT_NAMES[weight]] = {
            "rows": len(selected),
            "failed_rows": len(selected) - len(successful),
            "mean_J_total": macro_mean(selected, "J_total"),
            "raw_stop_rate": float(np.mean([row["operator_count"] == 0
                                             for row in selected])),
            "mean_compression_ratio": (float(np.mean([
                row["compression_ratio"] for row in successful])) if successful else None),
            "mean_compression_ns_per_byte": (float(np.mean([
                row["compression_ns_per_byte"] for row in successful]))
                if successful else None),
            "mean_decompression_ns_per_byte": (float(np.mean([
                row["decompression_ns_per_byte"] for row in successful]))
                if successful else None),
        }
    relative = {}
    for weight in PRIMARY_WEIGHTS:
        selected = [by_pair[sample, weight] for sample in sorted(large_samples)]
        baseline_rows = []
        for row in selected:
            core.require(row["sample_id"] in VALIDATION_BASELINE,
                         "validation Zstd baseline is incomplete")
            baseline_rows.append({**row, "J_total":
                VALIDATION_BASELINE[row["sample_id"]]["J_by_weight"][WEIGHT_NAMES[weight]]})
        baseline_J = macro_mean(baseline_rows, "J_total")
        relative[WEIGHT_NAMES[weight]] = (
            primary_by_weight[WEIGHT_NAMES[weight]]["mean_J_total"] / baseline_J)
    failed = sum(row["status"] != "ok" for row in rows)
    # The score is predeclared before training. A failed candidate remains in
    # the receipt but cannot be selected or hide a bad domain/weight row.
    if failed:
        dual_result = None
        score = 1e9
    else:
        core.require(VALIDATION_BASELINE_DOC, "frozen Zstd baseline document is absent")
        dual_result = dual_view.score(VALIDATION_BASELINE_DOC, {
            "schema": "native-combo-greedy-validation-v6",
            "greedy_only_checkpoint_selection": True,
            "validation_candidate_evaluations": 0,
            "rows": rows})
        score = dual_result["selection_score"]
    by_weight = {}
    for weight in core.FIXED_WEIGHTS:
        selected = [by_pair[sample, tuple(weight)] for sample in sorted(sample_bytes)]
        successful = [row for row in selected if row["status"] == "ok"]
        by_weight[WEIGHT_NAMES[tuple(weight)]] = {
            "rows": len(selected),
            "failed_rows": len(selected) - len(successful),
            "mean_J_total": macro_mean(selected, "J_total"),
            "raw_stop_rate": float(np.mean([row["operator_count"] == 0
                                             for row in selected])),
            "mean_compression_ratio": (float(np.mean([
                row["compression_ratio"] for row in successful])) if successful else None),
            "mean_compression_ns_per_byte": (float(np.mean([
                row["compression_ns_per_byte"] for row in successful]))
                if successful else None),
            "mean_decompression_ns_per_byte": (float(np.mean([
                row["decompression_ns_per_byte"] for row in successful]))
                if successful else None),
        }
    return {"selection_rule": SELECTION, "selection_score": score,
            "relative_J_by_weight": relative,
            "worst_relative_J": (dual_result["worst_relative_J"]
                                  if dual_result else None),
            "dual_view_validation": dual_result,
            "row_count": len(rows), "failed_rows": failed,
            "selection_eligible": failed == 0,
            "primary_rows": len(large_samples) * len(PRIMARY_WEIGHTS),
            "primary_samples": len(large_samples),
            "primary_domains": sorted({by_pair[sample, PRIMARY_WEIGHTS[0]]["domain"]
                                       for sample in large_samples}),
            "primary_by_weight": primary_by_weight, "by_weight": by_weight,
            "raw_stop_rate": float(np.mean([row["operator_count"] == 0 for row in rows])),
            "primary_raw_stop_rate": float(np.mean([
                by_pair[sample, weight]["operator_count"] == 0
                for sample in large_samples for weight in PRIMARY_WEIGHTS])),
            "mean_J_total_all_rows": float(np.mean([row["J_total"] for row in rows]))}


def validate(backend, policy, blocks, contract, rng, output, iteration, phase):
    rows = []
    for block in blocks:
        for weight in core.FIXED_WEIGHTS:
            episode = core.rollout(backend, policy, block, weight, contract, rng, greedy=True)
            measurement, error = None, episode["failure"]
            if error is None:
                try:
                    measurement = core.replay_measure(
                        backend, block, weight, contract, episode["recipe"], warmups=1, repetitions=5)
                except (core.OnlineRequestTimeout, core.OnlineNativeFailure, ValueError) as failure:
                    error = f"{type(failure).__name__}: {failure}"
            recipe = measurement.recipe if measurement else episode["recipe"]
            rows.append({"sample_id": block.sample_id, "dataset_id": block.dataset_id,
                "domain": block.domain, "B_raw": block.original_numeric_bytes,
                "size_bucket": block.bucket, "weight": list(weight),
                "status": "ok" if measurement else "failed", "error": error,
                "model_sha256": backend.model_sha256, "recipe_action_ids": list(recipe),
                "operator_count": sum(0 < action < core.FLOW_STOP_BEGIN for action in recipe),
                "J_total": core.objective(measurement, block, weight) if measurement else
                    core.constraint_failure_objective(block, weight),
                "compression_ratio": measurement.compression_ratio if measurement else None,
                "compression_ns_per_byte": measurement.compression_ns_per_byte if measurement else None,
                "decompression_ns_per_byte": measurement.decompression_ns_per_byte if measurement else None})
    receipt = {"schema": "native-combo-greedy-validation-v6", "phase": phase,
        "iteration": iteration, "model_sha256": backend.model_sha256,
        "greedy_only_checkpoint_selection": True, "validation_candidate_evaluations": 0,
        "native_timing_repeats": 5, "rows": rows, **summarize(rows)}
    path = output / f"validation_{phase}_{iteration:06d}.json"
    write(path, receipt)
    print(json.dumps({"event": "validation", "path": str(path),
                      **{key: receipt[key] for key in ("selection_score", "failed_rows",
                                                       "raw_stop_rate", "primary_raw_stop_rate")}}), flush=True)
    return receipt, path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, action="append", required=True)
    for name in ("reference", "server", "catalog", "profile", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--initial-checkpoint", type=Path)
    p.add_argument("--validation-baseline", type=Path, required=True)
    p.add_argument("--phase", choices=("initialize", "train"), required=True)
    p.add_argument("--cpu", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budget-seconds", type=float, default=880)
    p.add_argument("--sft-seconds", type=float, default=480)
    p.add_argument("--sft-steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grpo-iterations", type=int, default=10)
    return p


def main():
    global VALIDATION_BASELINE, VALIDATION_BASELINE_DOC
    args = parser().parse_args()
    core.require(len(args.data) == 2 and 90 < args.budget_seconds <= 920,
                 "two manifests and <=920-second continuation budget required")
    core.require(0 < args.sft_seconds <= 480 and 1 <= args.sft_steps <= 1800 and
                 1 <= args.batch_size <= 4096 and 0 <= args.grpo_iterations <= 10,
                 "SFT/GRPO limits exceed the bounded v4 contract")
    args.output.mkdir(parents=True, exist_ok=True)
    blocks = core.load_training_data(args.data, reference_path=args.reference)
    train = tuple(block for block in blocks if block.split == "train")
    validation = tuple(block for block in blocks if block.split == "validation")
    baseline_document = json.loads(args.validation_baseline.read_bytes())
    core.require(baseline_document.get("schema") == "sixweight-validation-zstd-v1" and
                 baseline_document.get("reference_sha256") == core.digest(args.reference.read_bytes()) and
                 baseline_document.get("validation_manifest_sha256") == core.digest(args.data[1].read_bytes()) and
                 baseline_document.get("test_used") is False,
                 "validation baseline identity drift")
    VALIDATION_BASELINE = {row["sample_id"]: row for row in baseline_document["rows"]}
    VALIDATION_BASELINE_DOC = baseline_document
    core.require(set(VALIDATION_BASELINE) == {block.sample_id for block in validation},
                 "validation baseline sample coverage differs")
    core.require(sum(block.original_numeric_bytes >= 65536 for block in train) >= 8 and
                 sum(block.original_numeric_bytes >= 65536 for block in validation) >= 4,
                 "insufficient large-block training/validation coverage")
    contract = core.OnlineContract()
    config = core.grpo.GRPOConfig(groups_per_iteration=4)
    policy = core.grpo.TwoHiddenMLP(core.policy_identity(train, contract), np.random.default_rng(args.seed))
    if args.initial_checkpoint is not None:
        loaded = core.grpo.load_checkpoint(args.initial_checkpoint,
            expected_registry_fingerprint=contract.registry_fingerprint,
            expected_data_fingerprints=dict(policy.identity.data_fingerprints))
        core.require(loaded.policy.identity == policy.identity,
                     "warm-start checkpoint identity differs from this data/runtime")
        policy = loaded.policy
    rng = np.random.default_rng(args.seed + 1_000_003)
    frozen = {str(path.resolve()): core.digest(path.read_bytes())
              for path in (*args.data, args.reference, args.catalog, args.profile,
                           args.server, args.validation_baseline, Path(__file__), Path(core.__file__))}
    if args.initial_checkpoint is not None:
        frozen[str(args.initial_checkpoint.resolve())] = core.digest(args.initial_checkpoint.read_bytes())
    identity = {"schema": SCHEMA, "train_samples": len(train), "validation_samples": len(validation),
        "schema_extension": "sixweight-dual-view-pretrain-v3",
        "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
        "frozen_files": frozen, "selection_rule": SELECTION,
        "primary_validation_sample_ids": sorted(
            block.sample_id for block in validation
            if block.original_numeric_bytes >= 65536),
        "sampling": "occupied size bucket, domain, dataset, weight, trajectory; equal mass at each level",
        "checkpoint_min_relative_improvement": 0.005,
        "contract": asdict(contract), "architecture": [49, 128, 128, 6373],
        "test_used": False, "budget_seconds": args.budget_seconds}
    if args.phase == "initialize":
        model, model_sha = core.export_current_model(policy, config, args.seed, 0,
            args.output, contract, phase="teacher_init_v6")
        write(args.output / "initialization.json", {**identity, "model": str(model), "model_sha256": model_sha})
        return
    core.require(args.teacher is not None and args.teacher.is_file(), "train-only teacher required")
    core.require(not (args.output / "training_started.json").exists(), "training output already started")
    records = core.load_teacher(args.teacher, train, contract)
    teacher_matrix = validate_strong_teacher(records, train)
    teacher_evidence = load_strong_teacher_evidence(args.teacher, records, train, contract)
    for label in ("receipt", "summary", "search_space", "cold_decode_receipts"):
        path = Path(teacher_evidence[label])
        frozen[str(path)] = core.digest(path.read_bytes())
    frozen[str(args.teacher.resolve())] = core.digest(args.teacher.read_bytes())
    features, legal, actions = core.teacher_arrays(records)
    weights, weight_receipt = supervised_weights(records, train)
    identity.update(teacher_sha256=core.digest(args.teacher.read_bytes()),
                    strong_teacher_matrix=teacher_matrix,
                    strong_teacher_evidence=teacher_evidence,
                    supervised_weighting=weight_receipt,
                    frozen_files=frozen)
    write(args.output / "training_started.json", identity)
    optimizer = core.grpo.Adam(policy.params, config.learning_rate, device=args.device)
    started = time.monotonic()
    end = started + args.budget_seconds
    trace, candidates = [], []
    epoch = steps = 0

    def checkpoint(backend, phase, iteration):
        model, model_sha = core.export_current_model(policy, config, args.seed,
            iteration, args.output, contract, phase=phase)
        core.require(backend.load_model(model) == model_sha, "checkpoint load mismatch")
        result, path = validate(backend, policy, validation, contract, rng,
                                args.output, iteration, phase)
        if result["selection_eligible"]:
            candidates.append({"score": result["selection_score"], "model": str(model),
                "model_sha256": model_sha, "validation": str(path), "phase": phase,
                "iteration": iteration})
            write(args.output / "best_so_far.json", choose_candidate(candidates))
        return result

    with core.OnlineTrainingServer(args.server, args.catalog, args.profile,
                                  cpu=args.cpu, timeout_seconds=5) as backend:
        # Retain the warm-start unless validation proves a material improvement.
        # This is a real zero-update candidate, not a relabeled SFT checkpoint.
        if args.initial_checkpoint is not None:
            checkpoint(backend, "warm_start", 0)
        sft_end = min(end - 90, started + args.sft_seconds)
        last_validated_steps = 0
        epoch_steps = math.ceil(len(records) / args.batch_size)
        core.require(epoch_steps <= args.sft_steps, "SFT step limit is smaller than one epoch")
        while steps + epoch_steps <= args.sft_steps and time.monotonic() < sft_end:
            update = core.masked_cross_entropy_update(policy, optimizer, features, legal, actions,
                batch_size=args.batch_size, rng=rng, device=args.device, sample_weights=weights)
            epoch += 1
            steps += update["optimizer_steps"]
            trace.append({"phase": "sft", "epoch": epoch, "steps": steps, **update})
            if steps - last_validated_steps >= 400:
                from numeric_first_teacher_accuracy_v1 import numeric_first_accuracy
                trace[-1]["numeric_first_accuracy"] = numeric_first_accuracy(
                    policy, records, sample_weights=weights)
                trace[-1].update(core.masked_teacher_accuracy(policy, records))
                checkpoint(backend, "sft", steps)
                last_validated_steps = steps
                write(args.output / "trace.json", trace)
        core.require(trace, "SFT budget expired before one optimizer epoch")
        if steps != last_validated_steps:
            trace[-1].update(core.masked_teacher_accuracy(policy, records))
            checkpoint(backend, "sft", steps)
        core.require(candidates, "no valid supervised checkpoint")
        # Start RL from validation-selected SFT weights, not an unverified tail.
        selected_sft = choose_candidate(candidates)
        loaded = core.grpo.load_checkpoint(Path(selected_sft["model"]).with_suffix(".json"),
            expected_registry_fingerprint=contract.registry_fingerprint,
            expected_data_fingerprints=dict(policy.identity.data_fingerprints))
        policy = loaded.policy
        config = replace(config, learning_rate=config.learning_rate * 0.1)
        optimizer = core.grpo.Adam(policy.params, config.learning_rate, device=args.device)
        core.require(backend.load_model(Path(selected_sft["model"])) ==
                     selected_sft["model_sha256"], "selected SFT model load mismatch")
        sampler = DomainFirstSampler(train)
        for iteration in range(1, args.grpo_iterations + 1):
            if end - time.monotonic() < 60:
                break
            row = core.grpo_iteration(backend, policy, optimizer, sampler, contract, rng, config,
                                     warmups=0, repetitions=1, group_size=4)
            trace.append({"phase": "grpo", "iteration": iteration, **row})
            model, model_sha = core.export_current_model(policy, config, args.seed, iteration,
                args.output, contract, phase="grpo_v6")
            core.require(backend.load_model(model) == model_sha,
                         "GRPO model load mismatch")
            if iteration % 2 == 0 or iteration == args.grpo_iterations:
                checkpoint(backend, "grpo_v6", iteration)
            write(args.output / "trace.json", trace)
        for path, sha in frozen.items():
            core.require(core.digest(Path(path).read_bytes()) == sha, "frozen training input changed")
        for candidate in candidates:
            core.require(core.digest(Path(candidate["model"]).read_bytes()) ==
                         candidate["model_sha256"], "validated candidate model changed")
        chosen = choose_candidate(candidates)
        write(args.output / "completion.json", {**identity, "selected": chosen,
            "candidates": candidates, "sft_epochs": epoch, "sft_steps": steps,
            "grpo_iterations": sum(row["phase"] == "grpo" for row in trace),
            "grpo_learning_rate": optimizer.learning_rate,
            "grpo_sampler": sampler.receipt(),
            "training_wall_seconds": time.monotonic() - started,
            "status": "complete", "trace": trace})
        print(json.dumps({"event": "complete", "selected": chosen}), flush=True)


if __name__ == "__main__":
    main()
