#!/usr/bin/env python3
"""Offline accuracy for the first substantive numeric-flow teacher action.

The v6 two-root runtime emits a mandatory bitmap-flow STOP at teacher step 0.
Consequently, step 1 is the first action that selects a numeric transform or
stops the raw numeric flow.  This module reports that decision separately from
the historical step-0 metric and does not rewrite old training traces.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


FLOW_STOP_BEGIN = 6341
BITMAP_FLOW_STOP = FLOW_STOP_BEGIN + 1


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def numeric_first_accuracy(
    policy: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    sample_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Measure masked top-1 accuracy at the first numeric-flow decision.

    ``sample_weights``, when provided, is aligned with all teacher records; only
    the entries for numeric-first rows contribute to weighted metrics.
    """

    if not records:
        raise ValueError("numeric-first accuracy requires teacher records")
    weights = (np.ones(len(records), dtype=np.float64) if sample_weights is None
               else np.asarray(sample_weights, dtype=np.float64))
    if (weights.shape != (len(records),) or not np.isfinite(weights).all()
            or not (weights > 0).all()):
        raise ValueError("invalid numeric-first sample weights")

    trajectories: dict[tuple[str, tuple[float, ...]], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        key = (str(row["sample_id"]), tuple(float(x) for x in row["weight"]))
        trajectories[key].append(index)

    numeric_indices = []
    for indices in trajectories.values():
        ordered = sorted(indices, key=lambda i: int(records[i]["step"]))
        steps = [int(records[i]["step"]) for i in ordered]
        if len(ordered) < 2 or steps != list(range(len(steps))):
            raise ValueError("teacher trajectory is incomplete")
        if int(records[ordered[0]]["action_id"]) != BITMAP_FLOW_STOP:
            raise ValueError("teacher step 0 is not the declared bitmap-flow STOP")
        numeric_indices.append(ordered[1])

    observations = np.asarray(
        [records[index]["observation"] for index in numeric_indices],
        dtype=np.float64,
    )
    if observations.ndim != 2 or not np.isfinite(observations).all():
        raise ValueError("invalid numeric-first observations")
    forward = policy.forward(observations)
    logits = np.asarray(forward[0] if isinstance(forward, tuple) else forward,
                        dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != len(numeric_indices):
        raise ValueError("policy returned invalid numeric-first logits")

    rows = []
    for position, record_index in enumerate(numeric_indices):
        row = records[record_index]
        legal = tuple(int(value) for value in row["legal_action_ids"])
        target = int(row["action_id"])
        if not legal or target not in legal or not all(0 <= value < logits.shape[1]
                                                       for value in legal):
            raise ValueError("invalid numeric-first legal action set")
        predicted = legal[int(np.argmax(logits[position, list(legal)]))]
        rows.append({
            "weight": tuple(float(value) for value in row["weight"]),
            "target": target,
            "predicted": predicted,
            "target_stop": target >= FLOW_STOP_BEGIN,
            "predicted_stop": predicted >= FLOW_STOP_BEGIN,
            "sample_weight": float(weights[record_index]),
        })

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        mass = sum(float(row["sample_weight"]) for row in selected)
        correct_mass = sum(float(row["sample_weight"])
                           for row in selected
                           if row["predicted"] == row["target"])
        target_stop = [row for row in selected if row["target_stop"]]
        target_operator = [row for row in selected if not row["target_stop"]]
        return {
            "records": len(selected),
            "action_accuracy": sum(row["predicted"] == row["target"]
                                   for row in selected) / len(selected),
            "weighted_action_accuracy": correct_mass / mass,
            "target_stop_support": len(target_stop),
            "target_operator_support": len(target_operator),
            "stop_action_accuracy": _ratio(
                sum(row["predicted"] == row["target"] for row in target_stop),
                len(target_stop)),
            "operator_action_accuracy": _ratio(
                sum(row["predicted"] == row["target"] for row in target_operator),
                len(target_operator)),
            "stop_vs_operator_accuracy": sum(
                row["predicted_stop"] == row["target_stop"] for row in selected
            ) / len(selected),
            "sample_weight_sum": mass,
        }

    by_weight = []
    for value in sorted({row["weight"] for row in rows}):
        by_weight.append({"weight": list(value),
                          **summarize([row for row in rows
                                      if row["weight"] == value])})
    return {
        "schema": "numeric-first-teacher-accuracy-v1",
        "numeric_first_step": 1,
        "historical_step0_excluded": True,
        **summarize(rows),
        "by_weight": by_weight,
    }
