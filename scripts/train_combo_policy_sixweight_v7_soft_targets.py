#!/usr/bin/env python3
"""Top-K regret distillation prototype for the frozen v13 online policy.

This module is deliberately isolated from the v6 campaign driver.  It changes
only the supervised training target and loss: the policy architecture, action
IDs, legal masks, native runtime, greedy online inference, and GRPO code remain
unchanged.  Candidate measurements are train-only offline evidence; they are
never evaluated by the online policy.

At every state visited by the exact winner trajectory, successful candidate
recipes that share the visited prefix are grouped by their next action.  The
best terminal J for each next action defines its regret.  The lowest-regret K
legal actions form a soft target.  A small auxiliary binary loss calibrates the
total probability of per-flow STOP versus ordinary actions whenever both are
legal.  It adds no output head and therefore does not alter the C++ contract.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import online_matched_train_v6 as core


SCHEMA = "sixweight-topk-regret-soft-targets-v1"
LOSS_SCHEMA = "masked-soft-ce-plus-stop-bce-v1"
DEFAULT_TOP_K = 4
DEFAULT_RELATIVE_REGRET_TEMPERATURE = 0.05
DEFAULT_STOP_BCE_WEIGHT = 0.20
EXPECTED_WEIGHTS = {tuple(float(value) for value in weight)
                    for weight in core.FIXED_WEIGHTS}


@dataclass(frozen=True)
class SoftTargetTrainingData:
    features: np.ndarray
    legal: np.ndarray
    targets: np.ndarray
    hard_actions: np.ndarray
    receipt: dict[str, Any]


def _weight(value: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(item) for item in core._weights(value))


def _candidate_rows(path: Path) -> tuple[dict[str, Any], ...]:
    core.require(path.is_file(), f"candidate measurement file absent: {path}")
    rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip())
    core.require(rows, "candidate measurement file is empty")
    return rows


def validate_exact_search_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    """Reject the old cross-weight timing reuse before target construction."""

    measured = {_weight(value)
                for value in document.get("measured_candidate_timing_weights", ())}
    core.require(document.get("schema") == "native-online-offline-combo-space-v6" and
                 document.get("train_only_teacher") is True and
                 document.get("validation_evaluation_labels_written") is False and
                 document.get("online_candidate_search") is False,
                 "candidate search is not train-only offline evidence")
    core.require(document.get("reuse_candidate_time_weights") is False and
                 document.get("approximate_teacher_timing") is False and
                 measured == EXPECTED_WEIGHTS,
                 "top-K regret targets require exact timing for all six weights")
    return {"schema": "exact-sixweight-candidate-search-gate-v1",
            "measured_candidate_timing_weights": [list(weight)
                                                    for weight in sorted(measured)],
            "approximate_teacher_timing": False}


def _validate_candidate_matrix(
        records: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        ) -> tuple[dict[tuple[str, tuple[float, float, float]], list[dict[str, Any]]],
                   dict[str, Any]]:
    expected_samples = {str(row["sample_id"]) for row in records}
    teacher_groups = {(str(row["sample_id"]), _weight(row["weight"]))
                      for row in records}
    core.require(teacher_groups == {(sample, weight) for sample in expected_samples
                                    for weight in EXPECTED_WEIGHTS},
                 "teacher must contain every train sample by six-weight group")
    groups: dict[tuple[str, tuple[float, float, float]], list[dict[str, Any]]] = \
        defaultdict(list)
    deadline_rows = 0
    explicit_failures = 0
    duplicates = 0
    for source in candidates:
        row = dict(source)
        core.require(row.get("schema") ==
                     "native-online-offline-combo-measurement-v6",
                     "unexpected candidate measurement schema")
        sample = str(row.get("sample_id", ""))
        core.require(sample in expected_samples and row.get("split") == "train",
                     "candidate measurement escaped the train cohort")
        status = str(row.get("status"))
        if status == "not_run_deadline":
            deadline_rows += 1
            continue
        if status == "failed":
            explicit_failures += 1
            continue
        if status == "duplicate_recipe":
            duplicates += 1
            continue
        core.require(status == "ok", "unknown candidate measurement status")
        weight = _weight(row["weight"])
        measured_weight = _weight(row["measured_timing_weight"])
        objective = float(row["objective"])
        recipe = tuple(int(action) for action in row["recipe_action_ids"])
        core.require(weight in EXPECTED_WEIGHTS and measured_weight == weight and
                     row.get("time_reused_across_weights") is False and
                     row.get("approximate_teacher_timing") is False and
                     math.isfinite(objective) and objective >= 0.0 and recipe,
                     "candidate is not an exact finite six-weight measurement")
        row["weight"] = weight
        row["recipe_action_ids"] = recipe
        row["objective"] = objective
        groups[sample, weight].append(row)
    core.require(deadline_rows == 0,
                 "candidate matrix contains deadline skips")
    core.require(set(groups) == teacher_groups and all(groups.values()),
                 "candidate matrix lacks successful sample by weight coverage")
    for sample in sorted(expected_samples):
        recipe_sets = [{row["recipe_action_ids"] for row in groups[sample, weight]}
                       for weight in sorted(EXPECTED_WEIGHTS)]
        core.require(all(value == recipe_sets[0] for value in recipe_sets[1:]),
                     "successful recipe inventory differs across weights")
    return groups, {
        "successful_measurements": sum(map(len, groups.values())),
        "explicit_failures": explicit_failures,
        "duplicate_recipes": duplicates,
        "deadline_skips": deadline_rows,
        "sample_weight_groups": len(groups),
        "samples": len(expected_samples),
    }


def build_topk_regret_targets(
        records: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]], *,
        top_k: int = DEFAULT_TOP_K,
        relative_regret_temperature: float = DEFAULT_RELATIVE_REGRET_TEMPERATURE,
        ) -> SoftTargetTrainingData:
    """Build legal soft targets on winner-visited states only.

    This is policy distillation over a finite measured candidate set.  It is not
    a claim that unvisited states or unmeasured actions have known costs.
    """

    core.require(records and 1 <= top_k <= 32 and
                 math.isfinite(relative_regret_temperature) and
                 relative_regret_temperature > 0.0,
                 "invalid top-K regret target configuration")
    groups, matrix_receipt = _validate_candidate_matrix(records, candidates)
    features, legal, hard_actions = core.teacher_arrays(records)
    targets = np.zeros_like(legal, dtype=np.float64)
    trajectory_rows: dict[tuple[str, tuple[float, float, float]], list[int]] = \
        defaultdict(list)
    for index, row in enumerate(records):
        trajectory_rows[str(row["sample_id"]), _weight(row["weight"])].append(index)

    support_sizes = []
    winner_masses = []
    relative_regrets = []
    one_hot_rows = 0
    mixed_stop_rows = 0
    for key, indices in trajectory_rows.items():
        ordered = sorted(indices, key=lambda index: int(records[index]["step"]))
        steps = [int(records[index]["step"]) for index in ordered]
        core.require(steps == list(range(len(ordered))),
                     "teacher trajectory steps are not contiguous")
        winner = tuple(int(value) for value in records[ordered[0]]["recipe_action_ids"])
        core.require(tuple(int(records[index]["action_id"]) for index in ordered) == winner,
                     "teacher actions do not reproduce the winner recipe")
        measured = groups[key]
        measured_best = min(float(row["objective"]) for row in measured)
        teacher_objective = float(records[ordered[0]]["teacher_objective"])
        tolerance = max(1e-12, abs(measured_best) * 1e-10)
        core.require(abs(teacher_objective - measured_best) <= tolerance and
                     any(row["recipe_action_ids"] == winner and
                         abs(float(row["objective"]) - measured_best) <= tolerance
                         for row in measured),
                     "teacher winner is not the exact best measured candidate")
        for step, index in enumerate(ordered):
            prefix = winner[:step]
            legal_actions = set(int(action) for action in records[index]["legal_action_ids"])
            best_by_action: dict[int, float] = {}
            for candidate in measured:
                recipe = candidate["recipe_action_ids"]
                if len(recipe) <= step or recipe[:step] != prefix:
                    continue
                action = int(recipe[step])
                core.require(action in legal_actions,
                             "measured candidate next action is illegal at winner prefix")
                best_by_action[action] = min(best_by_action.get(action, math.inf),
                                             float(candidate["objective"]))
            teacher_action = int(records[index]["action_id"])
            core.require(teacher_action in best_by_action,
                         "winner action has no measured continuation")
            ranked = sorted(best_by_action.items(),
                            key=lambda item: (item[1], item[0] != teacher_action, item[0]))
            chosen = ranked[:top_k]
            core.require(any(action == teacher_action for action, _ in chosen),
                         "top-K target excluded the exact winner action")
            best = chosen[0][1]
            scale = max(abs(best), 1e-12)
            regrets = np.asarray([(cost - best) / scale for _, cost in chosen],
                                 dtype=np.float64)
            logits = -regrets / relative_regret_temperature
            logits -= float(logits.max())
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            for (action, _), probability in zip(chosen, probabilities):
                targets[index, action] = probability
            support_sizes.append(len(chosen))
            winner_masses.append(float(targets[index, teacher_action]))
            relative_regrets.extend(float(value) for value in regrets)
            one_hot_rows += len(chosen) == 1
            selected_actions = {action for action, _ in chosen}
            mixed_stop_rows += (any(action >= core.FLOW_STOP_BEGIN
                                    for action in selected_actions) and
                                any(action < core.FLOW_STOP_BEGIN
                                    for action in selected_actions))
    core.require(np.isfinite(targets).all() and
                 np.allclose(targets.sum(axis=1), 1.0, rtol=0.0, atol=1e-12) and
                 not np.any(targets[~legal]),
                 "constructed soft targets violate the legal action simplex")
    receipt = {
        "schema": SCHEMA,
        "action_count": core.ACTION_COUNT,
        "flow_stop_begin": core.FLOW_STOP_BEGIN,
        "registry_fingerprint": core.REGISTRY_FINGERPRINT,
        "feature_schema": core.FEATURE_SCHEMA,
        "architecture_changed": False,
        "online_contract_changed": False,
        "online_candidate_search": False,
        "train_only_candidate_evidence": True,
        "top_k": top_k,
        "relative_regret_temperature": relative_regret_temperature,
        "records": len(records),
        "trajectories": len(trajectory_rows),
        "one_hot_records": one_hot_rows,
        "multi_action_records": len(records) - one_hot_rows,
        "mixed_stop_target_records": mixed_stop_rows,
        "mean_target_support": float(np.mean(support_sizes)),
        "max_target_support": max(support_sizes),
        "mean_winner_probability": float(np.mean(winner_masses)),
        "max_relative_regret_in_targets": max(relative_regrets),
        "candidate_matrix": matrix_receipt,
    }
    return SoftTargetTrainingData(features, legal, targets, hard_actions, receipt)


def load_topk_regret_training_data(
        records: Sequence[Mapping[str, Any]], candidate_path: Path, *,
        search_space_path: Path | None = None,
        top_k: int = DEFAULT_TOP_K,
        relative_regret_temperature: float = DEFAULT_RELATIVE_REGRET_TEMPERATURE,
        ) -> SoftTargetTrainingData:
    """Load and hash-bound candidate evidence for an integrating driver."""

    if search_space_path is None:
        search_space_path = candidate_path.with_name("search_space.json")
    core.require(search_space_path.is_file(), "candidate search metadata absent")
    search = json.loads(search_space_path.read_text(encoding="utf-8"))
    gate = validate_exact_search_metadata(search)
    result = build_topk_regret_targets(
        records, _candidate_rows(candidate_path), top_k=top_k,
        relative_regret_temperature=relative_regret_temperature)
    result.receipt.update({
        "candidate_measurements": str(candidate_path.resolve()),
        "candidate_measurements_sha256": core.digest(candidate_path.read_bytes()),
        "search_space": str(search_space_path.resolve()),
        "search_space_sha256": core.digest(search_space_path.read_bytes()),
        "exact_search_gate": gate,
    })
    return result


def _validate_soft_arrays(features: np.ndarray, legal: np.ndarray,
                          targets: np.ndarray) -> None:
    core.require(features.ndim == 2 and
                 features.shape == (legal.shape[0], core.OBSERVATION_DIM) and
                 legal.shape == targets.shape == (features.shape[0], core.ACTION_COUNT),
                 "soft target batch shape mismatch")
    core.require(np.isfinite(features).all() and np.isfinite(targets).all() and
                 legal.any(axis=1).all() and np.all(targets >= 0.0) and
                 np.allclose(targets.sum(axis=1), 1.0, rtol=0.0, atol=1e-12) and
                 not np.any(targets[~legal]),
                 "invalid masked soft target data")


def masked_soft_target_metrics(policy: Any, features: np.ndarray,
                               legal: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """NumPy diagnostic using the same masked probability definition as training."""

    _validate_soft_arrays(features, legal, targets)
    logits, _ = policy.forward(features)
    masked = np.where(legal, logits, -np.inf)
    maxima = np.max(masked, axis=1, keepdims=True)
    exponentials = np.where(legal, np.exp(masked - maxima), 0.0)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    safe = np.maximum(probabilities, np.finfo(np.float64).tiny)
    cross_entropy = -np.sum(targets * np.log(safe), axis=1)
    predicted = np.argmax(masked, axis=1)
    target_top = np.argmax(targets, axis=1)
    stop_slice = slice(core.FLOW_STOP_BEGIN, core.ACTION_COUNT)
    return {
        "mean_masked_soft_cross_entropy": float(np.mean(cross_entropy)),
        "soft_target_top1_agreement": float(np.mean(predicted == target_top)),
        "mean_target_stop_mass": float(np.mean(targets[:, stop_slice].sum(axis=1))),
        "mean_predicted_stop_mass": float(np.mean(
            probabilities[:, stop_slice].sum(axis=1))),
    }


def masked_soft_cross_entropy_update(
        policy: Any, optimizer: Any, features: np.ndarray, legal: np.ndarray,
        targets: np.ndarray, *, batch_size: int, rng: np.random.Generator,
        gradient_max_norm: float = 1.0, device: str | None = None,
        sample_weights: np.ndarray | None = None,
        stop_bce_weight: float = DEFAULT_STOP_BCE_WEIGHT,
        ) -> dict[str, float | int | str | bool]:
    """One shuffled epoch of legal soft CE plus aggregate STOP BCE."""

    from weighted_grpo_torch_backend_v1 import torch_device
    _validate_soft_arrays(features, legal, targets)
    core.require(1 <= batch_size and math.isfinite(stop_bce_weight) and
                 stop_bce_weight >= 0.0, "invalid soft target optimizer configuration")
    torch, target_device = torch_device(device or optimizer.execution_device)
    if sample_weights is None:
        sample_weights = np.ones(features.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
    core.require(sample_weights.shape == (features.shape[0],) and
                 np.isfinite(sample_weights).all() and (sample_weights > 0).all(),
                 "invalid soft target sample weights")
    order = rng.permutation(features.shape[0])
    params = [torch.tensor(value, dtype=torch.float64, device=target_device)
              for value in policy.params]
    first = [torch.tensor(value, dtype=torch.float64, device=target_device)
             for value in optimizer.first]
    second = [torch.tensor(value, dtype=torch.float64, device=target_device)
              for value in optimizer.second]
    steps = optimizer.steps
    total_losses: list[float] = []
    ce_losses: list[float] = []
    stop_losses: list[float] = []
    correct = total = stop_eligible = 0
    action_ids = torch.arange(core.ACTION_COUNT, device=target_device)
    stop_columns = action_ids >= core.FLOW_STOP_BEGIN
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        x = torch.tensor(features[batch], dtype=torch.float64, device=target_device)
        masks = torch.tensor(legal[batch], dtype=torch.bool, device=target_device)
        soft = torch.tensor(targets[batch], dtype=torch.float64, device=target_device)
        weights = torch.tensor(sample_weights[batch], dtype=torch.float64,
                               device=target_device)
        work = [value.detach().requires_grad_(True) for value in params]
        w1, b1, w2, b2, w3, b3 = work
        logits = torch.tanh(torch.tanh(x @ w1 + b1) @ w2 + b2) @ w3 + b3
        masked = logits.masked_fill(~masks, -torch.inf)
        log_probabilities = torch.nn.functional.log_softmax(masked, dim=1)
        safe_log_probabilities = torch.where(
            masks, log_probabilities, torch.zeros_like(log_probabilities))
        per_record_ce = -(soft * safe_log_probabilities).sum(dim=1)

        legal_stop = masks & stop_columns.unsqueeze(0)
        legal_ordinary = masks & (~stop_columns).unsqueeze(0)
        stop_rows = legal_stop.any(dim=1) & legal_ordinary.any(dim=1)
        per_record_stop = torch.zeros_like(per_record_ce)
        if bool(stop_rows.any()):
            stop_lse = torch.logsumexp(logits[stop_rows].masked_fill(
                ~legal_stop[stop_rows], -torch.inf), dim=1)
            ordinary_lse = torch.logsumexp(logits[stop_rows].masked_fill(
                ~legal_ordinary[stop_rows], -torch.inf), dim=1)
            binary_logit = stop_lse - ordinary_lse
            target_stop = soft[stop_rows][:, core.FLOW_STOP_BEGIN:].sum(dim=1)
            per_record_stop[stop_rows] = torch.nn.functional.binary_cross_entropy_with_logits(
                binary_logit, target_stop, reduction="none")
        per_record = per_record_ce + float(stop_bce_weight) * per_record_stop
        loss = (per_record * weights).mean()
        core.require(bool(torch.isfinite(loss)), "non-finite masked soft target loss")
        loss.backward()
        gradients = [value.grad for value in work]
        core.require(all(value is not None and bool(torch.isfinite(value).all())
                         for value in gradients),
                     "non-finite masked soft target gradient")
        norm = torch.sqrt(sum(value.square().sum() for value in gradients
                              if value is not None))
        scale = torch.clamp(float(gradient_max_norm) / norm.clamp_min(1e-30), max=1.0)
        steps += 1
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(zip(params, gradients)):
                assert gradient is not None
                gradient = gradient * scale
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient.square()
                moment = first[index] / (1.0 - 0.9 ** steps)
                variance = second[index] / (1.0 - 0.999 ** steps)
                parameter -= optimizer.learning_rate * moment / (
                    torch.sqrt(variance) + 1e-8)
        total_losses.append(float(loss.detach().cpu()))
        ce_losses.append(float((per_record_ce * weights).mean().detach().cpu()))
        stop_losses.append(float((per_record_stop * weights).mean().detach().cpu()))
        correct += int((masked.argmax(dim=1) == soft.argmax(dim=1)).sum().detach().cpu())
        total += len(batch)
        stop_eligible += int(stop_rows.sum().detach().cpu())
    core.require(all(bool(torch.isfinite(value).all())
                     for value in (*params, *first, *second)),
                 "non-finite masked soft target optimizer state")
    for destination, source in zip(policy.params, params):
        destination[:] = source.detach().cpu().numpy()
    optimizer.first = [value.detach().cpu().numpy().copy() for value in first]
    optimizer.second = [value.detach().cpu().numpy().copy() for value in second]
    optimizer.steps = steps
    return {
        "loss_schema": LOSS_SCHEMA,
        "mean_masked_soft_cross_entropy": float(np.mean(ce_losses)),
        "mean_stop_binary_cross_entropy": float(np.mean(stop_losses)),
        "mean_total_distillation_loss": float(np.mean(total_losses)),
        "soft_target_top1_agreement": correct / total,
        "records": total,
        "stop_auxiliary_eligible_records": stop_eligible,
        "optimizer_steps": len(total_losses),
        "weighted_training": not np.all(sample_weights == 1.0),
        "sample_weight_sum": float(sample_weights.sum()),
        "stop_bce_weight": float(stop_bce_weight),
        "policy_update_device": str(target_device),
        "policy_update_backend": "pytorch_float64",
        "architecture_changed": False,
    }


def soft_sft_epoch(policy: Any, optimizer: Any,
                   data: SoftTargetTrainingData, *, batch_size: int,
                   rng: np.random.Generator, device: str | None = None,
                   sample_weights: np.ndarray | None = None,
                   stop_bce_weight: float = DEFAULT_STOP_BCE_WEIGHT,
                   ) -> dict[str, Any]:
    """Narrow integration seam for a future bounded campaign driver."""

    return masked_soft_cross_entropy_update(
        policy, optimizer, data.features, data.legal, data.targets,
        batch_size=batch_size, rng=rng, device=device,
        sample_weights=sample_weights, stop_bce_weight=stop_bce_weight)
