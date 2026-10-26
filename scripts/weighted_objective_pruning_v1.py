"""Certified and heuristic pruning for the weighted compression objective.

Only ``certified_prune`` removes a subtree.  The heuristic score is an ordering
hint and must never be treated as proof that a continuation cannot win.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


SCHEMA = "weighted-objective-pruning-v1"


def validate_weight(weight: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in weight)
    if (len(values) != 3 or any(not math.isfinite(value) or value < 0.0 for value in values)
            or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-12)):
        raise ValueError("weight must contain three finite non-negative values summing to one")
    return values


@dataclass(frozen=True)
class BoundState:
    """Information whose bytes every legal continuation must retain."""

    original_numeric_bytes: int
    fixed_frame_bytes: int
    mandatory_metadata_bytes: int
    exhausted_leaf_logical_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.original_numeric_bytes <= 0:
            raise ValueError("original_numeric_bytes must be positive")
        if self.fixed_frame_bytes < 0 or self.mandatory_metadata_bytes < 0:
            raise ValueError("frame and mandatory metadata byte counts must be non-negative")
        if any(type(value) is not int or value < 0 for value in self.exhausted_leaf_logical_bits):
            raise ValueError("exhausted leaf logical bit counts must be non-negative integers")

    @property
    def unavoidable_bytes(self) -> int:
        return (self.fixed_frame_bytes + self.mandatory_metadata_bytes
                + (sum(self.exhausted_leaf_logical_bits) + 7) // 8)


@dataclass(frozen=True)
class PruningDecision:
    prune: bool
    objective_lower_bound: float
    incumbent_objective: float
    comparison: str
    reason: str


def objective_lower_bound(state: BoundState, weight: Sequence[float]) -> float:
    """Return ``a * L(s) / B_raw``; time costs only contribute zero bounds."""
    a, _b, _c = validate_weight(weight)
    return a * state.unavoidable_bytes / state.original_numeric_bytes


def certified_prune(
    state: BoundState,
    weight: Sequence[float],
    incumbent_objective: float,
    *,
    keep_ties: bool = True,
) -> PruningDecision:
    if not math.isfinite(incumbent_objective) or incumbent_objective < 0.0:
        raise ValueError("incumbent objective must be finite and non-negative")
    lower = objective_lower_bound(state, weight)
    prune = lower > incumbent_objective if keep_ties else lower >= incumbent_objective
    comparison = ">" if keep_ties else ">="
    reason = ("certified_weighted_lower_bound" if prune
              else "bound_does_not_rule_out_improvement")
    return PruningDecision(prune, lower, incumbent_objective, comparison, reason)


def ordering_score(
    *,
    weight: Sequence[float],
    estimated_ratio: float,
    estimated_compression_ns_per_byte: float,
    estimated_decompression_ns_per_byte: float,
    compression_reference_ns_per_byte: float,
    decompression_reference_ns_per_byte: float,
    exploration_bonus: float = 0.0,
) -> float:
    """Objective-aware search order only; a low score is explored first."""
    a, b, c = validate_weight(weight)
    values = (estimated_ratio, estimated_compression_ns_per_byte,
              estimated_decompression_ns_per_byte,
              compression_reference_ns_per_byte,
              decompression_reference_ns_per_byte, exploration_bonus)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("ordering inputs must be finite")
    if min(values[:5]) < 0.0 or compression_reference_ns_per_byte <= 0.0 or decompression_reference_ns_per_byte <= 0.0:
        raise ValueError("estimated costs must be non-negative and references positive")
    ehat = estimated_compression_ns_per_byte / (
        estimated_compression_ns_per_byte + compression_reference_ns_per_byte)
    dhat = estimated_decompression_ns_per_byte / (
        estimated_decompression_ns_per_byte + decompression_reference_ns_per_byte)
    return a * estimated_ratio + b * ehat + c * dhat - exploration_bonus


def feature_priority(weight: Sequence[float], features: Mapping[str, float]) -> dict[str, float]:
    """Transparent heuristic priorities; all legal groups retain nonzero mass."""
    a, b, c = validate_weight(weight)
    smooth = min(1.0, max(0.0, float(features.get("smoothness", 0.0))))
    repeated = min(1.0, max(0.0, float(features.get("repeat_fraction", 0.0))))
    floating = min(1.0, max(0.0, float(features.get("floating_fraction", 0.0))))
    priorities = {
        "delta_zigzag_pack": 1.0 + 2.0 * a * smooth,
        "dictionary_run_length": 1.0 + 2.0 * a * repeated,
        "xor_window_scaling_exception": 1.0 + 2.0 * a * floating,
        "measured_fast_encode": 1.0 + 2.0 * b,
        "measured_fast_decode": 1.0 + 2.0 * c,
        "other_legal": 1.0,
    }
    total = sum(priorities.values())
    return {key: value / total for key, value in priorities.items()}


def objective_guided_ordering_score(
    *,
    weight: Sequence[float],
    estimated_cost: tuple[float, float, float] | None,
    compression_reference_ns_per_byte: float,
    decompression_reference_ns_per_byte: float,
    feature_priority_mass: float,
    measurement_count: int,
) -> float:
    """Rank one legal action using measurements while preserving exploration.

    A complete-recipe measurement containing the action supplies rho/E/D.  An
    unseen action receives neutral physical estimates and a larger deterministic
    exploration bonus.  This function is an ordering heuristic only: neither an
    unseen action nor a poorly measured action is certified away.
    """
    if (isinstance(measurement_count, bool) or not isinstance(measurement_count, int)
            or measurement_count < 0):
        raise ValueError("measurement_count must be a non-negative integer")
    if not math.isfinite(feature_priority_mass) or feature_priority_mass <= 0.0:
        raise ValueError("feature_priority_mass must be finite and positive")
    if estimated_cost is None:
        ratio = 1.0
        compression = float(compression_reference_ns_per_byte)
        decompression = float(decompression_reference_ns_per_byte)
    else:
        if len(estimated_cost) != 3:
            raise ValueError("estimated_cost must contain rho/E/D")
        ratio, compression, decompression = (
            float(value) for value in estimated_cost
        )
    # Every legal group has positive feature mass.  The count term makes unseen
    # actions win a bounded share of ties without converting the heuristic into
    # a pruning rule.
    exploration_bonus = (
        0.10 / math.sqrt(measurement_count + 1.0)
        + 0.05 * feature_priority_mass
    )
    return ordering_score(
        weight=weight,
        estimated_ratio=ratio,
        estimated_compression_ns_per_byte=compression,
        estimated_decompression_ns_per_byte=decompression,
        compression_reference_ns_per_byte=compression_reference_ns_per_byte,
        decompression_reference_ns_per_byte=decompression_reference_ns_per_byte,
        exploration_bonus=exploration_bonus,
    )
