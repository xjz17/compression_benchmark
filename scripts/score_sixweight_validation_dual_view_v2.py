#!/usr/bin/env python3
"""Deterministic dual-view six-weight validation scorer for v7 prototypes.

The mathematical rule is the v1 rule.  Every set/dict reduction is explicitly
ordered and uses ``math.fsum``.  ``numeric_tree_close`` is intended for a
postflight recomputation: identities remain exact, while finite floating-point
leaves tolerate harmless process-order rounding.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import score_sixweight_validation_dual_view_v1 as v1


WEIGHTS = v1.WEIGHTS
E0 = v1.E0
D0 = v1.D0
MIN_LARGE_BYTES = v1.MIN_LARGE_BYTES
SELECTION_RULE = (
    "max_six_weight_relative_J_large_and_all_domain_macro_plus_0.01_mean_"
    "deterministic_v2"
)
require = v1.require
metric = v1.metric
objective = v1.objective
near = v1.near


def hierarchy_masses(keys: set[tuple[str, str, str]]
                     ) -> dict[tuple[str, str, str], float]:
    ordered = tuple(sorted(keys))
    require(ordered, "empty macro hierarchy")
    children: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for key in ordered:
        for index, part in enumerate(key):
            children[key[:index]].add(part)
    masses = {key: math.prod(1.0 / len(children[key[:index]])
                             for index in range(len(key)))
              for key in ordered}
    require(near(math.fsum(masses[key] for key in ordered), 1.0),
            "macro masses do not sum to one")
    return masses


def macro(values: dict[str, float],
          metadata: dict[str, tuple[str, str, str]]) -> float:
    groups: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
    for sample in sorted(values):
        groups[metadata[sample]].append((sample, float(values[sample])))
    masses = hierarchy_masses(set(groups))
    terms = []
    for key in sorted(groups):
        rows = sorted(groups[key])
        group_mean = math.fsum(value for _sample, value in rows) / len(rows)
        terms.append(masses[key] * group_mean)
    return math.fsum(terms)


def numeric_tree_close(left: Any, right: Any, *,
                       rel_tol: float = 1e-12,
                       abs_tol: float = 1e-15) -> bool:
    """Compare receipts exactly except for finite real-valued leaves."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int) or isinstance(right, int):
        return (type(left) is int and type(right) is int and left == right)
    if isinstance(left, float) and isinstance(right, float):
        a, b = float(left), float(right)
        return math.isfinite(a) and math.isfinite(b) and math.isclose(
            a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            numeric_tree_close(left[key], right[key], rel_tol=rel_tol,
                               abs_tol=abs_tol) for key in sorted(left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            numeric_tree_close(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def require_recomputed_summary(stored: dict[str, Any],
                               recomputed: dict[str, Any]) -> None:
    require(numeric_tree_close(stored, recomputed),
            "stored validation summary differs materially from deterministic recomputation")


def score(baseline: dict, validation: dict,
          expected_validation_manifest_sha256: str | None = None) -> dict:
    require(baseline.get("schema") == "sixweight-validation-zstd-v1" and
            baseline.get("test_used") is False and baseline.get("cpu") == 25 and
            baseline.get("objective") == "a*R+b*E/(E+E0)+c*D/(D+D0)",
            "frozen same-panel Zstd baseline contract missing")
    if expected_validation_manifest_sha256 is not None:
        require(baseline.get("validation_manifest_sha256") ==
                expected_validation_manifest_sha256,
                "baseline does not bind the admitted validation manifest")
    require(validation.get("schema") == "native-combo-greedy-validation-v6" and
            validation.get("greedy_only_checkpoint_selection") is True and
            validation.get("validation_candidate_evaluations") == 0,
            "greedy C++ validation contract missing")
    baseline_by_sample: dict[str, dict] = {}
    for row in baseline.get("rows", []):
        sample = str(row["sample_id"])
        require(sample not in baseline_by_sample and int(row["B_raw"]) > 0 and
                row.get("independent_cold_decode") is True,
                "duplicate or unverified Zstd baseline sample")
        require(set(row["J_by_weight"]) == set(WEIGHTS),
                "incomplete Zstd six-weight vector")
        for name, weight in WEIGHTS.items():
            require(near(float(row["J_by_weight"][name]), objective(row, weight)),
                    "Zstd objective differs from frozen scale")
        baseline_by_sample[sample] = row
    require(baseline_by_sample, "empty Zstd baseline")
    model: dict[tuple[str, str], dict] = {}
    metadata: dict[str, tuple[str, str, str]] = {}
    names_by_weight = {weight: name for name, weight in WEIGHTS.items()}
    for row in validation.get("rows", []):
        sample = str(row["sample_id"])
        weight = tuple(float(part) for part in row["weight"])
        require(sample in baseline_by_sample and weight in names_by_weight,
                "model validation sample or weight outside frozen panel")
        name = names_by_weight[weight]
        require((sample, name) not in model and
                int(row["B_raw"]) == int(baseline_by_sample[sample]["B_raw"]),
                "duplicate validation row or B_raw mismatch")
        require(row.get("status") == "ok" and row.get("error") is None,
                "failed validation row cannot be selected")
        require(near(float(row["J_total"]), objective(row, weight)),
                "model objective differs from frozen scale")
        key = tuple(str(row[field]) for field in
                    ("domain", "dataset_id", "size_bucket"))
        require(len(key) == 3 and all(key) and
                (sample not in metadata or metadata[sample] == key),
                "sample macro hierarchy changed across weights")
        metadata[sample] = key
        model[sample, name] = row
    require(set(metadata) == set(baseline_by_sample) and
            set(model) == {(sample, name) for sample in metadata for name in WEIGHTS},
            "validation is not a complete same-panel sample by weight matrix")
    large = tuple(sorted(sample for sample, row in baseline_by_sample.items()
                         if int(row["B_raw"]) >= MIN_LARGE_BYTES))
    require(large, "no >=64KiB validation sample")
    all_samples = tuple(sorted(metadata))
    result: dict[str, Any] = {}
    relative: list[float] = []
    for panel_name, samples in (("large", large), ("all", all_samples)):
        panel = {}
        panel_metadata = {sample: metadata[sample] for sample in samples}
        for name in WEIGHTS:
            model_j = macro({sample: float(model[sample, name]["J_total"])
                             for sample in samples}, panel_metadata)
            zstd_j = macro({sample: float(
                baseline_by_sample[sample]["J_by_weight"][name])
                            for sample in samples}, panel_metadata)
            require(zstd_j > 0 and math.isfinite(model_j) and math.isfinite(zstd_j),
                    "nonfinite or nonpositive macro objective")
            ratio = model_j / zstd_j
            relative.append(ratio)
            panel[name] = {"model_macro_J": model_j, "zstd_macro_J": zstd_j,
                           "relative_J": ratio}
        result[panel_name] = {
            "samples": len(samples),
            "domains": sorted({metadata[sample][0] for sample in samples}),
            "by_weight": panel,
        }
    mean_relative = math.fsum(relative) / len(relative)
    return {
        "schema": "sixweight-dual-view-validation-score-v2",
        "selection_rule": SELECTION_RULE,
        "selection_score": max(relative) + 0.01 * mean_relative,
        "worst_relative_J": max(relative),
        "mean_relative_J": mean_relative,
        "views": result,
        "validation_row_count": len(model),
        "test_used": False,
        "E0": E0,
        "D0": D0,
        "deterministic_reduction": "sorted_keys_and_math_fsum_v1",
        "postflight_comparison": "exact_identity_numeric_isclose_1e-12_v1",
    }
