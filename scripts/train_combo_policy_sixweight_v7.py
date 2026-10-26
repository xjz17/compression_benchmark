#!/usr/bin/env python3
"""Bounded v7 soft-target distillation and unchanged native GRPO refinement.

The online model and C++ contract remain 49 -> 128 -> 128 -> 6373.  This
trainer replaces winner-only supervised CE with the train-only, exact-timing
top-K regret target in ``train_combo_policy_sixweight_v7_soft_targets`` and
uses the deterministic dual-view v2 scorer for checkpoint selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np

import online_matched_train_v6 as core
import score_sixweight_validation_dual_view_v2 as dual_view
import train_combo_policy_sixweight_v3 as inherited
import train_combo_policy_sixweight_v7_soft_targets as soft


SCHEMA = "native-combo-quality-training-v7-soft-target"
SELECTION = dual_view.SELECTION_RULE
VALIDATION_BASELINE: dict[str, dict] = {}
VALIDATION_BASELINE_DOC: dict = {}

write = inherited.write
supervised_weights = inherited.supervised_weights
validate_strong_teacher = inherited.validate_strong_teacher
load_strong_teacher_evidence = inherited.load_strong_teacher_evidence
DomainFirstSampler = inherited.DomainFirstSampler
choose_candidate = inherited.choose_candidate


def _configure_validation_globals() -> None:
    inherited.dual_view = dual_view
    inherited.SELECTION = SELECTION
    inherited.VALIDATION_BASELINE = VALIDATION_BASELINE
    inherited.VALIDATION_BASELINE_DOC = VALIDATION_BASELINE_DOC


def summarize(rows):
    _configure_validation_globals()
    return inherited.summarize(rows)


def validate(backend, policy, blocks, contract, rng, output, iteration, phase):
    _configure_validation_globals()
    return inherited.validate(backend, policy, blocks, contract, rng,
                              output, iteration, phase)


def admit_soft_teacher(records, candidate_measurements: Path,
                       search_space: Path, *, min_teacher_templates: int = 102,
                       top_k: int = soft.DEFAULT_TOP_K,
                       regret_temperature: float =
                       soft.DEFAULT_RELATIVE_REGRET_TEMPERATURE):
    search = json.loads(search_space.read_bytes())
    materialized = int(search.get("materialized_candidate_count", -1))
    plans = search.get("plans", ())
    core.require(min_teacher_templates >= 102 and
                 materialized >= min_teacher_templates and
                 len(plans) == materialized and
                 int(search.get("candidate_limit", -1)) >= materialized and
                 search.get("online_action_space_unchanged") is True and
                 search.get("strong_path_action_registry_changed") is False,
                 "v7 requires an admitted 102+ unchanged-registry teacher inventory")
    data = soft.load_topk_regret_training_data(
        records, candidate_measurements, search_space_path=search_space,
        top_k=top_k, relative_regret_temperature=regret_temperature)
    core.require(data.receipt["candidate_matrix"]["samples"] > 0,
                 "soft teacher has no admitted samples")
    data.receipt.update({"minimum_teacher_templates": min_teacher_templates,
                         "materialized_teacher_templates": materialized})
    return data


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, action="append", required=True)
    for name in ("reference", "server", "catalog", "profile", "output"):
        value.add_argument("--" + name, type=Path, required=True)
    value.add_argument("--teacher", type=Path)
    value.add_argument("--candidate-measurements", type=Path)
    value.add_argument("--teacher-search-space", type=Path)
    value.add_argument("--initial-checkpoint", type=Path)
    value.add_argument("--validation-baseline", type=Path, required=True)
    value.add_argument("--phase", choices=("initialize", "train"), required=True)
    value.add_argument("--cpu", type=int, default=25)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--budget-seconds", type=float, default=290)
    value.add_argument("--sft-seconds", type=float, default=200)
    value.add_argument("--sft-steps", type=int, default=1800)
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--grpo-iterations", type=int, default=0)
    value.add_argument("--min-teacher-templates", type=int, default=102)
    value.add_argument("--top-k", type=int, default=soft.DEFAULT_TOP_K)
    value.add_argument("--regret-temperature", type=float,
                       default=soft.DEFAULT_RELATIVE_REGRET_TEMPERATURE)
    value.add_argument("--stop-bce-weight", type=float,
                       default=soft.DEFAULT_STOP_BCE_WEIGHT)
    return value


def main() -> None:
    global VALIDATION_BASELINE, VALIDATION_BASELINE_DOC
    args = parser().parse_args()
    core.require(len(args.data) == 2 and 90 < args.budget_seconds <= 920 and
                 0 < args.sft_seconds <= args.budget_seconds - 90 and
                 1 <= args.sft_steps <= 1800 and 1 <= args.batch_size <= 4096 and
                 0 <= args.grpo_iterations <= 10 and
                 args.min_teacher_templates >= 102,
                 "v7 SFT/GRPO limits exceed the bounded continuation contract")
    args.output.mkdir(parents=True, exist_ok=True)
    blocks = core.load_training_data(args.data, reference_path=args.reference)
    train = tuple(block for block in blocks if block.split == "train")
    validation = tuple(block for block in blocks if block.split == "validation")
    baseline_document = json.loads(args.validation_baseline.read_bytes())
    core.require(baseline_document.get("schema") == "sixweight-validation-zstd-v1" and
                 baseline_document.get("reference_sha256") ==
                 core.digest(args.reference.read_bytes()) and
                 baseline_document.get("validation_manifest_sha256") ==
                 core.digest(args.data[1].read_bytes()) and
                 baseline_document.get("test_used") is False,
                 "validation baseline identity drift")
    VALIDATION_BASELINE = {row["sample_id"]: row
                           for row in baseline_document["rows"]}
    VALIDATION_BASELINE_DOC = baseline_document
    _configure_validation_globals()
    core.require(set(VALIDATION_BASELINE) == {block.sample_id for block in validation},
                 "validation baseline sample coverage differs")
    core.require(sum(block.original_numeric_bytes >= 65536 for block in train) >= 8 and
                 sum(block.original_numeric_bytes >= 65536 for block in validation) >= 4,
                 "insufficient large-block training/validation coverage")
    contract = core.OnlineContract()
    config = core.grpo.GRPOConfig(groups_per_iteration=4)
    policy = core.grpo.TwoHiddenMLP(
        core.policy_identity(train, contract), np.random.default_rng(args.seed))
    if args.initial_checkpoint is not None:
        loaded = core.grpo.load_checkpoint(
            args.initial_checkpoint,
            expected_registry_fingerprint=contract.registry_fingerprint,
            expected_data_fingerprints=dict(policy.identity.data_fingerprints))
        core.require(loaded.policy.identity == policy.identity,
                     "warm-start checkpoint identity differs from this data/runtime")
        policy = loaded.policy
    rng = np.random.default_rng(args.seed + 1_000_003)
    source_files = (Path(__file__), Path(core.__file__), Path(inherited.__file__),
                    Path(soft.__file__), Path(dual_view.__file__),
                    Path(dual_view.v1.__file__))
    frozen = {str(path.resolve()): core.digest(path.read_bytes())
              for path in (*args.data, args.reference, args.catalog, args.profile,
                           args.server, args.validation_baseline, *source_files)}
    if args.initial_checkpoint is not None:
        frozen[str(args.initial_checkpoint.resolve())] = core.digest(
            args.initial_checkpoint.read_bytes())
    identity = {
        "schema": SCHEMA,
        "schema_extension": "sixweight-topk-regret-soft-target-pretrain-v7",
        "train_samples": len(train), "validation_samples": len(validation),
        "initial_checkpoint": (str(args.initial_checkpoint)
                               if args.initial_checkpoint else None),
        "frozen_files": frozen, "selection_rule": SELECTION,
        "primary_validation_sample_ids": sorted(
            block.sample_id for block in validation
            if block.original_numeric_bytes >= 65536),
        "sampling": "occupied size bucket, domain, dataset, weight, trajectory",
        "checkpoint_min_relative_improvement": 0.005,
        "contract": asdict(contract), "architecture": [49, 128, 128, 6373],
        "action_registry_changed": False, "online_candidate_search": False,
        "test_used": False, "budget_seconds": args.budget_seconds,
    }
    if args.phase == "initialize":
        model, model_sha = core.export_current_model(
            policy, config, args.seed, 0, args.output, contract,
            phase="teacher_init_v7")
        write(args.output / "initialization.json",
              {**identity, "model": str(model), "model_sha256": model_sha})
        return

    for path in (args.teacher, args.candidate_measurements,
                 args.teacher_search_space):
        core.require(path is not None and path.is_file(),
                     "train phase requires teacher, candidates, and search metadata")
    assert args.teacher is not None
    assert args.candidate_measurements is not None
    assert args.teacher_search_space is not None
    core.require(not (args.output / "training_started.json").exists(),
                 "training output already started")
    records = core.load_teacher(args.teacher, train, contract)
    teacher_matrix = validate_strong_teacher(records, train)
    teacher_evidence = load_strong_teacher_evidence(
        args.teacher, records, train, contract)
    soft_data = admit_soft_teacher(
        records, args.candidate_measurements, args.teacher_search_space,
        min_teacher_templates=args.min_teacher_templates,
        top_k=args.top_k, regret_temperature=args.regret_temperature)
    for path in (args.teacher, args.candidate_measurements,
                 args.teacher_search_space):
        frozen[str(path.resolve())] = core.digest(path.read_bytes())
    for label in ("receipt", "summary", "search_space", "cold_decode_receipts"):
        path = Path(teacher_evidence[label])
        frozen[str(path)] = core.digest(path.read_bytes())
    weights, weight_receipt = supervised_weights(records, train)
    identity.update(
        teacher_sha256=core.digest(args.teacher.read_bytes()),
        candidate_measurements_sha256=core.digest(
            args.candidate_measurements.read_bytes()),
        strong_teacher_matrix=teacher_matrix,
        strong_teacher_evidence=teacher_evidence,
        soft_target_training=soft_data.receipt,
        supervised_weighting=weight_receipt,
        frozen_files=frozen,
    )
    write(args.output / "training_started.json", identity)
    optimizer = core.grpo.Adam(policy.params, config.learning_rate,
                               device=args.device)
    started = time.monotonic()
    end = started + args.budget_seconds
    trace, candidates = [], []
    epoch = steps = 0

    def checkpoint(backend, phase, iteration):
        model, model_sha = core.export_current_model(
            policy, config, args.seed, iteration, args.output, contract, phase=phase)
        core.require(backend.load_model(model) == model_sha,
                     "checkpoint load mismatch")
        result, path = validate(backend, policy, validation, contract, rng,
                                args.output, iteration, phase)
        if result["selection_eligible"]:
            candidates.append({"score": result["selection_score"],
                "model": str(model), "model_sha256": model_sha,
                "validation": str(path), "phase": phase,
                "iteration": iteration})
            write(args.output / "best_so_far.json", choose_candidate(candidates))
        return result

    with core.OnlineTrainingServer(args.server, args.catalog, args.profile,
                                   cpu=args.cpu, timeout_seconds=5) as backend:
        if args.initial_checkpoint is not None:
            checkpoint(backend, "warm_start", 0)
        sft_end = min(end - 90, started + args.sft_seconds)
        last_validated_steps = 0
        epoch_steps = math.ceil(len(records) / args.batch_size)
        core.require(epoch_steps <= args.sft_steps,
                     "SFT step limit is smaller than one epoch")
        while steps + epoch_steps <= args.sft_steps and time.monotonic() < sft_end:
            update = soft.soft_sft_epoch(
                policy, optimizer, soft_data, batch_size=args.batch_size,
                rng=rng, device=args.device, sample_weights=weights,
                stop_bce_weight=args.stop_bce_weight)
            epoch += 1
            steps += int(update["optimizer_steps"])
            trace.append({"phase": "sft", "epoch": epoch, "steps": steps, **update})
            if steps - last_validated_steps >= 400:
                trace[-1].update(core.masked_teacher_accuracy(policy, records))
                trace[-1].update(soft.masked_soft_target_metrics(
                    policy, soft_data.features, soft_data.legal, soft_data.targets))
                checkpoint(backend, "sft", steps)
                last_validated_steps = steps
                write(args.output / "trace.json", trace)
        core.require(trace, "SFT budget expired before one optimizer epoch")
        if steps != last_validated_steps:
            trace[-1].update(core.masked_teacher_accuracy(policy, records))
            trace[-1].update(soft.masked_soft_target_metrics(
                policy, soft_data.features, soft_data.legal, soft_data.targets))
            checkpoint(backend, "sft", steps)
        core.require(candidates, "no valid supervised checkpoint")
        selected_sft = choose_candidate(candidates)
        loaded = core.grpo.load_checkpoint(
            Path(selected_sft["model"]).with_suffix(".json"),
            expected_registry_fingerprint=contract.registry_fingerprint,
            expected_data_fingerprints=dict(policy.identity.data_fingerprints))
        policy = loaded.policy
        config = replace(config, learning_rate=config.learning_rate * 0.1)
        optimizer = core.grpo.Adam(policy.params, config.learning_rate,
                                   device=args.device)
        core.require(backend.load_model(Path(selected_sft["model"])) ==
                     selected_sft["model_sha256"], "selected SFT model load mismatch")
        sampler = DomainFirstSampler(train)
        for iteration in range(1, args.grpo_iterations + 1):
            if end - time.monotonic() < 60:
                break
            row = core.grpo_iteration(
                backend, policy, optimizer, sampler, contract, rng, config,
                warmups=0, repetitions=1, group_size=4)
            trace.append({"phase": "grpo", "iteration": iteration, **row})
            if iteration % 2 == 0 or iteration == args.grpo_iterations:
                checkpoint(backend, "grpo_v7", iteration)
            write(args.output / "trace.json", trace)
        for path, sha256 in frozen.items():
            core.require(core.digest(Path(path).read_bytes()) == sha256,
                         "frozen v7 training input changed")
        for candidate in candidates:
            core.require(core.digest(Path(candidate["model"]).read_bytes()) ==
                         candidate["model_sha256"],
                         "validated candidate model changed")
        chosen = choose_candidate(candidates)
        write(args.output / "completion.json", {
            **identity, "selected": chosen, "candidates": candidates,
            "sft_epochs": epoch, "sft_steps": steps,
            "grpo_iterations": sum(row["phase"] == "grpo" for row in trace),
            "grpo_learning_rate": optimizer.learning_rate,
            "grpo_sampler": sampler.receipt(),
            "training_wall_seconds": time.monotonic() - started,
            "status": "complete", "trace": trace,
        })
        print(json.dumps({"event": "complete", "selected": chosen}), flush=True)


if __name__ == "__main__":
    main()
