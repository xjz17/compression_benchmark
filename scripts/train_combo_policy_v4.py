#!/usr/bin/env python3
"""Bounded strong-teacher distillation and matched native GRPO refinement.

The online action space and C++ runtime are unchanged.  This driver never
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
import online_matched_train_v3_fit3 as core

SCHEMA = "native-combo-quality-training-v4"
PRIMARY_WEIGHTS = ((1.0, 0.0, 0.0), (0.4, 0.3, 0.3))
SELECTION = "mean_J_total_on_ge64KiB_validation_for_100_and_433_equal_weight"
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


def supervised_weights(records, blocks):
    """Equal size buckets, datasets within buckets, weights, then trajectories.

    Every trajectory has equal mass within its stratum, regardless of how many
    actions it contains. Unlike fit2, this does not force STOP/non-STOP balance.
    """
    by_id = {block.sample_id: block for block in blocks}
    trajectories = defaultdict(list)
    for index, row in enumerate(records):
        block = by_id[row["sample_id"]]
        key = (block.bucket, block.dataset_id, tuple(row["weight"]), row["sample_id"])
        trajectories[key].append(index)
    buckets = sorted({key[0] for key in trajectories})
    datasets = {bucket: {key[1] for key in trajectories if key[0] == bucket}
                for bucket in buckets}
    weights = defaultdict(set)
    sources = defaultdict(set)
    for bucket, dataset, weight, sample in trajectories:
        weights[bucket, dataset].add(weight)
        sources[bucket, dataset, weight].add(sample)
    result = np.zeros(len(records), dtype=np.float64)
    for key, indices in trajectories.items():
        bucket, dataset, weight, _ = key
        mass = 1 / len(buckets) / len(datasets[bucket])
        mass /= len(weights[bucket, dataset]) * len(sources[bucket, dataset, weight])
        result[indices] = mass / len(indices)
    core.require(np.all(result > 0) and math.isclose(float(result.sum()), 1),
                 "invalid trajectory weight mass")
    result *= len(result)
    return result, {"schema": "size-dataset-trajectory-balanced-ce-v1",
                    "buckets": buckets, "records": len(records),
                    "trajectories": len(trajectories), "weight_sum": float(result.sum())}


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
                 "native-online-offline-combo-teacher-receipt-v4" and
                 receipt.get("teacher_sha256") == core.digest(teacher.read_bytes()) and
                 int(receipt.get("records", -1)) == len(records) and
                 int(receipt.get("train_samples", -1)) == sample_count and
                 int(receipt.get("validation_records", -1)) == 0 and
                 receipt.get("contract_sha256") == contract.fingerprint and
                 receipt.get("search_space_sha256") == core.digest(search_path.read_bytes()),
                 "strong teacher receipt mismatch")
    proof = summary.get("quality_gate", {})
    core.require(summary.get("schema") == "native-online-offline-combo-teacher-v4" and
                 summary.get("completed") is True and
                 summary.get("validation_evaluated") is False and
                 int(summary.get("teacher_records", -1)) == len(records) and
                 proof.get("status") == "complete" and proof.get("passed") is True and
                 int(proof.get("common_samples", -1)) == sample_count,
                 "strong teacher quality/coverage gate failed")
    core.require(search.get("schema") == "native-online-offline-combo-space-v4" and
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


class SizeFirstSampler:
    """Equal occupied size buckets, then domains/datasets inside each bucket."""
    def __init__(self, blocks):
        groups = defaultdict(list)
        for block in blocks:
            groups[block.bucket].append(block)
        self.samplers = {key: core.StratifiedSampler(rows) for key, rows in groups.items()}
        self.buckets = sorted(self.samplers)

    def sample(self, rng):
        return self.samplers[self.buckets[int(rng.integers(len(self.buckets)))]].sample(rng)

    def receipt(self):
        return {key: self.samplers[key].receipt() for key in self.buckets}


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
            "mean_J_total": float(np.mean([row["J_total"] for row in selected])),
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
    score = float(np.mean([primary_by_weight[WEIGHT_NAMES[weight]]["mean_J_total"]
                           for weight in PRIMARY_WEIGHTS]))
    failed = sum(row["status"] != "ok" for row in rows)
    by_weight = {}
    for weight in core.FIXED_WEIGHTS:
        selected = [by_pair[sample, tuple(weight)] for sample in sorted(sample_bytes)]
        successful = [row for row in selected if row["status"] == "ok"]
        by_weight[WEIGHT_NAMES[tuple(weight)]] = {
            "rows": len(selected),
            "failed_rows": len(selected) - len(successful),
            "mean_J_total": float(np.mean([row["J_total"] for row in selected])),
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
            "row_count": len(rows), "failed_rows": failed,
            "selection_eligible": failed == 0,
            "primary_rows": len(large_samples) * len(PRIMARY_WEIGHTS),
            "primary_samples": len(large_samples),
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
                        backend, block, weight, contract, episode["recipe"], warmups=1, repetitions=3)
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
    receipt = {"schema": "native-combo-greedy-validation-v4", "phase": phase,
        "iteration": iteration, "model_sha256": backend.model_sha256,
        "greedy_only_checkpoint_selection": True, "validation_candidate_evaluations": 0,
        "native_timing_repeats": 3, "rows": rows, **summarize(rows)}
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
    p.add_argument("--phase", choices=("initialize", "train"), required=True)
    p.add_argument("--cpu", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budget-seconds", type=float, default=900)
    p.add_argument("--sft-seconds", type=float, default=480)
    p.add_argument("--sft-steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grpo-iterations", type=int, default=10)
    return p


def main():
    args = parser().parse_args()
    core.require(len(args.data) == 2 and 90 < args.budget_seconds <= 1000,
                 "two manifests and <=1000-second continuation budget required")
    core.require(0 < args.sft_seconds <= 480 and 1 <= args.sft_steps <= 1800 and
                 1 <= args.batch_size <= 4096 and 0 <= args.grpo_iterations <= 10,
                 "SFT/GRPO limits exceed the bounded v4 contract")
    args.output.mkdir(parents=True, exist_ok=True)
    blocks = core.load_training_data(args.data, reference_path=args.reference)
    train = tuple(block for block in blocks if block.split == "train")
    validation = tuple(block for block in blocks if block.split == "validation")
    core.require(sum(block.original_numeric_bytes >= 65536 for block in train) >= 8 and
                 sum(block.original_numeric_bytes >= 65536 for block in validation) >= 4,
                 "insufficient large-block training/validation coverage")
    contract = core.OnlineContract()
    config = core.grpo.GRPOConfig(groups_per_iteration=4)
    policy = core.grpo.TwoHiddenMLP(core.policy_identity(train, contract), np.random.default_rng(args.seed))
    rng = np.random.default_rng(args.seed + 1_000_003)
    frozen = {str(path.resolve()): core.digest(path.read_bytes())
              for path in (*args.data, args.reference, args.catalog, args.profile,
                           args.server, Path(__file__), Path(core.__file__))}
    identity = {"schema": SCHEMA, "train_samples": len(train), "validation_samples": len(validation),
        "frozen_files": frozen, "selection_rule": SELECTION,
        "primary_validation_sample_ids": sorted(
            block.sample_id for block in validation
            if block.original_numeric_bytes >= 65536),
        "sampling": "size bucket first, then domain/dataset within bucket",
        "contract": asdict(contract), "architecture": [49, 128, 128, 6373],
        "test_used": False, "budget_seconds": args.budget_seconds}
    if args.phase == "initialize":
        model, model_sha = core.export_current_model(policy, config, args.seed, 0,
            args.output, contract, phase="teacher_init_v4")
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
            write(args.output / "best_so_far.json", min(candidates, key=lambda row: row["score"]))
        return result

    with core.OnlineTrainingServer(args.server, args.catalog, args.profile,
                                  cpu=args.cpu, timeout_seconds=5) as backend:
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
        selected_sft = min(candidates, key=lambda row: row["score"])
        loaded = core.grpo.load_checkpoint(Path(selected_sft["model"]).with_suffix(".json"),
            expected_registry_fingerprint=contract.registry_fingerprint,
            expected_data_fingerprints=dict(policy.identity.data_fingerprints))
        policy = loaded.policy
        config = replace(config, learning_rate=config.learning_rate * 0.1)
        optimizer = core.grpo.Adam(policy.params, config.learning_rate, device=args.device)
        core.require(backend.load_model(Path(selected_sft["model"])) ==
                     selected_sft["model_sha256"], "selected SFT model load mismatch")
        sampler = SizeFirstSampler(train)
        for iteration in range(1, args.grpo_iterations + 1):
            if end - time.monotonic() < 60:
                break
            row = core.grpo_iteration(backend, policy, optimizer, sampler, contract, rng, config,
                                     warmups=0, repetitions=1, group_size=4)
            trace.append({"phase": "grpo", "iteration": iteration, **row})
            model, model_sha = core.export_current_model(policy, config, args.seed, iteration,
                args.output, contract, phase="grpo_v4")
            core.require(backend.load_model(model) == model_sha,
                         "GRPO model load mismatch")
            if iteration % 2 == 0 or iteration == args.grpo_iterations:
                checkpoint(backend, "grpo_v4", iteration)
            write(args.output / "trace.json", trace)
        for path, sha in frozen.items():
            core.require(core.digest(Path(path).read_bytes()) == sha, "frozen training input changed")
        for candidate in candidates:
            core.require(core.digest(Path(candidate["model"]).read_bytes()) ==
                         candidate["model_sha256"], "validated candidate model changed")
        chosen = min(candidates, key=lambda row: row["score"])
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
