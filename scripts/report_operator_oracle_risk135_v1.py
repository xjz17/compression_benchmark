#!/usr/bin/env python3
"""Independently recompute the development-only risk-selector report.

This program accepts no held-out cost input.  It joins the frozen development
cost matrix to the selected and frozen-control OOF rankings, recomputes every
byte and risk metric, and refuses to replace existing report outputs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TASKS = {
    "algorithm_space": "algorithm",
    "operator_zstd3_space": "operator_zstd3",
    "observed_pipeline_space": "observed_pipeline",
}
FULL_REFERENCE_COUNTS = {
    "algorithm": 85,
    "operator_zstd3": 97,
    "observed_pipeline": 182,
}
ALLOWED_COUNTS = {
    "algorithm": 84,
    "operator_zstd3": 97,
    "observed_pipeline": 181,
}
EXPECTED_CONTROL_BYTES = {
    "algorithm_space": 2_941_000_129,
    "operator_zstd3_space": 2_662_068_813,
    "observed_pipeline_space": 2_678_889_909,
}
EXPECTED_ROWS = 3433
EXPECTED_SELECTOR_ROWS = EXPECTED_ROWS * len(TASKS)


class ReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path} is not an object")
    if schema is not None:
        require(value.get("schema") == schema, f"unexpected schema in {path}")
    return value


def jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as source:
        for number, line in enumerate(source, 1):
            require(line.endswith("\n"), f"{path}:{number} lacks final newline")
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{number} is not an object")
            yield number, value


def write_new_text(path: Path, value: str) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    require(not temporary.exists(), f"temporary output exists: {temporary}")
    with temporary.open("x") as target:
        target.write(value)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def write_new_json(path: Path, value: Any) -> None:
    write_new_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity(row: dict[str, Any]) -> tuple[str, str, str]:
    dataset = row.get("dataset_id")
    column = row.get("column_id", row.get("sample_id"))
    task = row.get("task_id")
    require(all(isinstance(item, str) and item for item in (dataset, column, task)),
            "selector row lacks identity")
    return dataset, column, task


def cost_identity(row: dict[str, Any]) -> tuple[str, str]:
    dataset = row.get("dataset_id")
    column = row.get("column_id", row.get("sample_id"))
    require(isinstance(dataset, str) and dataset and
            isinstance(column, str) and column, "cost row lacks identity")
    return dataset, column


def load_selectors(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for number, row in jsonl(path):
        require(row.get("schema") == "operator-oracle-risk135-oof-row-v1",
                f"{path}:{number} schema mismatch")
        key = identity(row)
        require(key[2] in TASKS and key not in result,
                f"{path}:{number} invalid/duplicate identity {key}")
        require(row.get("query_costs_used_for_selection") is False,
                f"{path}:{number} says query costs entered selection")
        ranking = row.get("ranking_candidate_ids")
        scores = row.get("ranking_predicted_log_costs")
        pool = row.get("candidate_pool")
        require(isinstance(ranking, list) and isinstance(scores, list) and
                isinstance(pool, list) and ranking and
                len(ranking) == len(scores) == len(pool) and
                len(ranking) == len(set(ranking)) and set(ranking) == set(pool),
                f"{path}:{number} ranking is not an exact pool permutation")
        require(all(isinstance(score, (int, float)) and not isinstance(score, bool)
                    and math.isfinite(float(score)) for score in scores),
                f"{path}:{number} ranking contains a nonfinite score")
        selected = row.get("selected_candidate_id")
        require(isinstance(selected, str) and ranking[0] == selected,
                f"{path}:{number} selected ID is not rank 1")
        result[key] = row
    require(len(result) == EXPECTED_SELECTOR_ROWS,
            f"{path} must contain exactly {EXPECTED_SELECTOR_ROWS} rows")
    return result


def load_baseline_ids(path: Path | None) -> tuple[dict[tuple[str, str, str], str], dict[str, Any]]:
    if path is None:
        return {}, {"status": "not_supplied", "row_count": 0, "sha256": None}
    result: dict[tuple[str, str, str], str] = {}
    for number, row in jsonl(path):
        require(row.get("schema") == "operator-oracle-quality135-oof-row-v1",
                f"{path}:{number} old baseline schema mismatch")
        key = identity(row)
        if key[2] not in TASKS:
            continue
        selected = row.get("selected_candidate_id")
        require(isinstance(selected, str), f"{path}:{number} old baseline selected ID missing")
        require(key not in result, f"{path}:{number} duplicate old baseline identity")
        result[key] = selected
    require(len(result) == EXPECTED_SELECTOR_ROWS,
            "old baseline must contain exactly three space tasks x 3433 rows")
    return result, {"status": "verified_input", "row_count": len(result),
                    "sha256": sha256_file(path), "path": str(path.resolve())}


def candidate_lookup(row: dict[str, Any], domain: str) -> dict[str, dict[str, Any]]:
    domains = row.get("domains")
    require(isinstance(domains, dict), "cost row lacks domains")
    values = domains.get(domain)
    require(isinstance(values, list) and values, f"cost row lacks {domain}")
    result: dict[str, dict[str, Any]] = {}
    for candidate in values:
        candidate_id = candidate.get("candidate_id")
        require(isinstance(candidate_id, str) and candidate_id and
                candidate_id not in result, f"invalid candidate in {domain}")
        result[candidate_id] = candidate
    require(len(result) == ALLOWED_COUNTS[domain],
            f"{domain} candidate count differs from frozen allowed scope")
    return result


def byte_value(candidate: dict[str, Any] | None) -> int | None:
    if (not isinstance(candidate, dict) or candidate.get("eligible") is not True or
            not isinstance(candidate.get("space_cost"), (int, float)) or
            isinstance(candidate.get("space_cost"), bool) or
            not math.isfinite(float(candidate["space_cost"])) or
            float(candidate["space_cost"]) <= 0):
        return None
    value = candidate.get("compressed_bytes")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def full_oracle_bytes(row: dict[str, Any], domain: str) -> int:
    names = {
        "algorithm": "algorithm_compressed_bytes",
        "operator_zstd3": "operator_compressed_bytes",
        "observed_pipeline": "observed_pipeline_compressed_bytes",
    }
    frozen = row.get("full_frozen_oracle")
    require(isinstance(frozen, dict), "cost row lacks full_frozen_oracle")
    value = frozen.get(names[domain])
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"invalid full oracle bytes for {domain}")
    return value


def nearest_rank_p90(values: list[int]) -> int:
    require(values, "empty training byte population")
    values = sorted(values)
    return values[max(1, math.ceil(0.9 * len(values))) - 1]


def fold_thresholds(costs: list[dict[str, Any]], confirm_legacy: set[str]) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for fold in range(5):
        heldout_legacy = {row["legacy_group"] for row in costs if row["dev_fold"] == fold}
        excluded = confirm_legacy | heldout_legacy
        train = [row for row in costs
                 if row["dev_fold"] != fold and row["legacy_group"] not in excluded]
        values = [row["original_numeric_bytes"] for row in train]
        result[fold] = {"threshold": nearest_rank_p90(values),
                        "training_rows": len(train),
                        "training_datasets": len({row["dataset_id"] for row in train}),
                        "training_source_groups": len({row["source_group"] for row in train})}
    return result


@dataclass
class Aggregate:
    rows: int = 0
    selected_bytes: int = 0
    full_oracle_bytes: int = 0
    allowed_oracle_bytes: int = 0
    dtype_oracle_bytes: int = 0
    lz4_bytes: int = 0
    zstd3_bytes: int = 0
    raw_bytes: int = 0
    unsupported: int = 0
    log_regret_by_dataset: dict[str, list[float]] = field(default_factory=dict)
    source_bytes: dict[str, list[int]] = field(default_factory=dict)
    excess_bytes: list[int] = field(default_factory=list)
    selected_counts: collections.Counter[str] = field(default_factory=collections.Counter)

    def add(self, *, dataset: str, source: str, selected_id: str | None,
            selected: int | None, full: int, allowed: int, dtype_oracle: int,
            lz4: int, zstd3: int, raw: int) -> None:
        self.rows += 1
        self.full_oracle_bytes += full
        self.allowed_oracle_bytes += allowed
        self.dtype_oracle_bytes += dtype_oracle
        self.lz4_bytes += lz4
        self.zstd3_bytes += zstd3
        self.raw_bytes += raw
        if selected is None:
            self.unsupported += 1
            return
        self.selected_bytes += selected
        self.selected_counts[selected_id] += 1
        state = self.log_regret_by_dataset.setdefault(dataset, [0.0, 0.0])
        state[0] += math.log(selected / allowed)
        state[1] += 1.0
        group = self.source_bytes.setdefault(source, [0, 0])
        group[0] += selected
        group[1] += allowed
        self.excess_bytes.append(selected - allowed)

    def summary(self) -> dict[str, Any]:
        if self.unsupported:
            dataset_geo = worst_source = cvar = None
        else:
            dataset_geo = math.exp(sum(total / count for total, count in
                                           self.log_regret_by_dataset.values()) /
                                   len(self.log_regret_by_dataset))
            worst_source_id, worst_source_values = max(
                self.source_bytes.items(), key=lambda item: item[1][0] / item[1][1])
            worst_source = worst_source_values[0] / worst_source_values[1]
            tail = math.ceil(0.05 * len(self.excess_bytes))
            cvar = sum(sorted(self.excess_bytes, reverse=True)[:tail]) / tail
        if self.unsupported:
            worst_source_id = None
            worst_source_values = [None, None]
            tail = None
        ratio = lambda numerator, denominator: numerator / denominator if denominator else None
        return {
            "rows": self.rows,
            "unsupported_rows": self.unsupported,
            "selected_bytes": self.selected_bytes if not self.unsupported else None,
            "full_oracle_bytes": self.full_oracle_bytes,
            "allowed_oracle_bytes": self.allowed_oracle_bytes,
            "dtype_pool_oracle_bytes": self.dtype_oracle_bytes,
            "fixed_LZ4_bytes": self.lz4_bytes,
            "fixed_Zstd3_bytes": self.zstd3_bytes,
            "raw_bytes": self.raw_bytes,
            "selected_to_full_oracle_ratio": ratio(self.selected_bytes, self.full_oracle_bytes)
                if not self.unsupported else None,
            "allowed_oracle_to_full_oracle_ratio":
                ratio(self.allowed_oracle_bytes, self.full_oracle_bytes),
            "dtype_pool_oracle_to_allowed_oracle_ratio":
                ratio(self.dtype_oracle_bytes, self.allowed_oracle_bytes),
            "dtype_pool_oracle_to_full_oracle_ratio":
                ratio(self.dtype_oracle_bytes, self.full_oracle_bytes),
            "selected_to_allowed_oracle_ratio": ratio(self.selected_bytes, self.allowed_oracle_bytes)
                if not self.unsupported else None,
            "selected_to_dtype_pool_oracle_ratio": ratio(self.selected_bytes, self.dtype_oracle_bytes)
                if not self.unsupported else None,
            "selected_to_LZ4_ratio": ratio(self.selected_bytes, self.lz4_bytes)
                if not self.unsupported else None,
            "selected_to_Zstd3_ratio": ratio(self.selected_bytes, self.zstd3_bytes)
                if not self.unsupported else None,
            "dataset_equal_geomean_regret": dataset_geo,
            "worst_official_source_group_aggregate_byte_regret": worst_source,
            "worst_official_source_group": worst_source_id,
            "worst_source_group_selected_bytes": worst_source_values[0],
            "worst_source_group_allowed_oracle_bytes": worst_source_values[1],
            "cvar95_absolute_excess_bytes": cvar,
            "cvar95_tail_rows": tail,
            "selected_candidate_counts": dict(sorted(self.selected_counts.items())),
        }


def close_number(left: Any, right: Any, context: str) -> None:
    require(isinstance(left, (int, float)) and not isinstance(left, bool) and
            isinstance(right, (int, float)) and not isinstance(right, bool) and
            math.isfinite(float(left)) and math.isfinite(float(right)) and
            math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12),
            f"numeric mismatch: {context}: {left!r} != {right!r}")


def policy_feasible(metrics: dict[str, Any], control: dict[str, Any]) -> bool:
    return (
        metrics.get("unsupported_validation_rows") == 0 and
        float(metrics["dataset_equal_geo_regret"]) <=
            1.05 * float(control["dataset_equal_geo_regret"]) + 1e-12 and
        float(metrics["worst_official_source_group_aggregate_byte_regret"]) <=
            float(control["worst_official_source_group_aggregate_byte_regret"]) + 1e-12 and
        float(metrics["cvar95_absolute_excess_bytes"]) <=
            float(control["cvar95_absolute_excess_bytes"]) + 1e-12
    )


def policy_diagnostics(cv_task: dict[str, Any]) -> dict[str, Any]:
    policies = cv_task.get("policies")
    require(isinstance(policies, list), "CV task policies are absent")
    task_id = cv_task.get("task_id")
    expected_count = 19 if task_id == "algorithm_space" else 18
    require(len(policies) == expected_count,
            f"{task_id} must contain {expected_count} policy summaries")
    by_id: dict[str, dict[str, Any]] = {}
    for item in policies:
        require(isinstance(item, dict), f"{task_id} policy summary is not an object")
        policy = item.get("policy")
        require(isinstance(policy, dict), f"{task_id} policy metadata is absent")
        policy_id = policy.get("id")
        require(isinstance(policy_id, str) and policy_id and policy_id not in by_id,
                f"{task_id} has an invalid/duplicate policy ID")
        require(policy.get("feature_variant") in {"prefix24", "multi3"},
                f"{task_id}/{policy_id} has an unknown feature variant")
        require(policy.get("weight_mode") in
                {"dataset_equal", "sqrt_bytes", "mixed_bytes"},
                f"{task_id}/{policy_id} has an unknown weight mode")
        require(isinstance(item.get("total_selected_bytes"), int) and
                item["total_selected_bytes"] > 0 and
                item.get("oof_rows") == EXPECTED_ROWS,
                f"{task_id}/{policy_id} has invalid compact metrics")
        by_id[policy_id] = item

    control_policy = cv_task.get("frozen_prefix_control_policy")
    selected_policy = cv_task.get("selected_policy")
    unconstrained_policy = cv_task.get("unconstrained_minimum_byte_policy")
    require(all(isinstance(value, dict) and isinstance(value.get("id"), str)
                for value in (control_policy, selected_policy, unconstrained_policy)),
            f"{task_id} lacks policy identities")
    control_id = control_policy["id"]
    require(control_id in by_id, f"{task_id} control policy is absent from policies")
    control = by_id[control_id]
    normalized: list[dict[str, Any]] = []
    for policy_id, item in by_id.items():
        independently_feasible = policy_feasible(item, control)
        require(item.get("risk_feasible_against_frozen_control") is independently_feasible,
                f"{task_id}/{policy_id} native feasibility flag mismatch")
        strict = (item.get("unsupported_validation_rows") == 0 and
                  item["total_selected_bytes"] < control["total_selected_bytes"])
        require(item.get("strict_total_byte_improvement_over_control") is strict,
                f"{task_id}/{policy_id} native strict-improvement flag mismatch")
        normalized.append({
            "policy": item["policy"],
            "total_selected_bytes": item["total_selected_bytes"],
            "total_allowed_oracle_bytes": item.get("total_allowed_oracle_bytes"),
            "aggregate_byte_regret": item.get("aggregate_byte_regret"),
            "dataset_equal_geo_regret": item.get("dataset_equal_geo_regret"),
            "worst_official_source_group_aggregate_byte_regret":
                item.get("worst_official_source_group_aggregate_byte_regret"),
            "cvar95_absolute_excess_bytes": item.get("cvar95_absolute_excess_bytes"),
            "unsupported_validation_rows": item.get("unsupported_validation_rows"),
            "risk_feasible_against_frozen_control": independently_feasible,
            "strict_total_byte_improvement_over_control": strict,
            "fold_metrics": item.get("fold_metrics"),
        })
    normalized.sort(key=lambda item: item["policy"]["id"])

    supported = [item for item in normalized
                 if item["unsupported_validation_rows"] == 0]
    require(supported, f"{task_id} has no supported policy")
    stable_min = lambda values: min(
        values, key=lambda item: (item["total_selected_bytes"], item["policy"]["id"]))
    unconstrained = stable_min(supported)
    prefix = stable_min([item for item in supported
                         if item["policy"]["feature_variant"] == "prefix24"])
    multi = stable_min([item for item in supported
                        if item["policy"]["feature_variant"] == "multi3"])
    strict_feasible = [item for item in normalized
                       if item["risk_feasible_against_frozen_control"] and
                       item["strict_total_byte_improvement_over_control"]]
    independently_selected = stable_min(strict_feasible) if strict_feasible else next(
        item for item in normalized if item["policy"]["id"] == control_id)
    expected_reason = (
        "minimum_total_selected_bytes_among_feasible_strict_improvements"
        if strict_feasible else
        "no_feasible_strict_total_byte_improvement_choose_frozen_control")
    require(unconstrained["policy"]["id"] == unconstrained_policy["id"],
            f"{task_id} unconstrained minimum is not the stable byte minimum")
    require(independently_selected["policy"]["id"] == selected_policy["id"],
            f"{task_id} selected policy is not the stable feasible byte minimum")
    require(cv_task.get("selection_reason") == expected_reason,
            f"{task_id} selection reason disagrees with independent selection")
    return {
        "policy_count": len(normalized),
        "stable_tie_break": "minimum (total_selected_bytes, policy_id)",
        "all_policy_summaries": normalized,
        "diagnostic_policies": {
            "unconstrained_minimum": unconstrained,
            "best_prefix24": prefix,
            "best_multi3": multi,
            "selected": independently_selected,
            "control": next(item for item in normalized
                            if item["policy"]["id"] == control_id),
        },
        "independent_selected_policy_id": independently_selected["policy"]["id"],
        "independent_selection_reason": expected_reason,
        "selected_is_stable_minimum_feasible_strict_improvement_or_control": True,
    }


def recompute(root: Path, baseline_oof: Path | None) -> dict[str, Any]:
    costs_path = root / "costs" / "DEVELOPMENT_COSTS.jsonl"
    cv_path = root / "models" / "CV_SUMMARY.json"
    index_path = root / "models" / "MODEL_INDEX.json"
    selected_path = root / "models" / "SELECTED_OOF_FULL_RANKS.jsonl"
    control_path = root / "models" / "CONTROL_OOF_FULL_RANKS.jsonl"
    tune_complete_path = root / "models" / "COMPLETE.json"
    protocol_path = root / "PROTOCOL.json"
    for path in (costs_path, cv_path, index_path, selected_path, control_path,
                 tune_complete_path, protocol_path):
        require(path.is_file(), f"missing required input {path}")
    protocol = read_json(protocol_path, "operator-oracle-risk135-protocol-v1")
    cv = read_json(cv_path, "operator-oracle-risk135-development-cv-v1")
    index = read_json(index_path, "operator-oracle-risk135-model-index-v1")
    tune_complete = read_json(
        tune_complete_path, "operator-oracle-risk135-tune-complete-v1")
    require(tune_complete.get("status") == "complete" and
            tune_complete.get("confirm_rows_read") == 0 and
            tune_complete.get("development_rows") == EXPECTED_ROWS,
            "native tune completion receipt is incomplete")
    tune_hashes = {
        "model_index_sha256": sha256_file(index_path),
        "cv_summary_sha256": sha256_file(cv_path),
        "selected_oof_sha256": sha256_file(selected_path),
        "control_oof_sha256": sha256_file(control_path),
    }
    for name, observed in tune_hashes.items():
        require(tune_complete.get(name) == observed,
                f"native tune completion SHA mismatch: {name}")
    costs_sha256 = sha256_file(costs_path)
    source_bindings = protocol.get("source_bindings")
    require(isinstance(source_bindings, dict) and
            source_bindings.get("development_costs_sha256") == costs_sha256,
            "development costs SHA disagrees with frozen protocol")
    require(cv.get("confirm_used_for_tuning") is False and
            index.get("confirm_used_for_fit_or_selection") is False,
            "development artifacts claim held-out labels entered tuning")
    require(index.get("NNCP") == "OFF" and
            index.get("joint_cross_product_measured") is False and
            index.get("excluded_algorithm_methods") == ["PPMD8"] and
            index.get("excluded_action_ids") == [106, 107, 108, 109, 110],
            "model index exclusion or Cartesian-product contract mismatch")
    selected = load_selectors(selected_path)
    control = load_selectors(control_path)
    baseline, baseline_receipt = load_baseline_ids(baseline_oof)
    tasks = {item["task_id"]: item for item in index.get("tasks", [])}
    require(set(tasks) == set(TASKS), "model index must contain exactly three space tasks")
    cv_tasks = {item["task_id"]: item for item in cv.get("tasks", [])}
    require(set(cv_tasks) == set(TASKS), "CV summary must contain exactly three space tasks")

    # The source matrix is roughly 496 MiB.  Keep only lightweight, label-free
    # metadata in the first pass; the candidate arrays are aggregated by a
    # second streaming pass so the report stays inside the 2 GiB run cap.
    metadata: list[dict[str, Any]] = []
    cost_keys: list[tuple[str, str]] = []
    for number, row in jsonl(costs_path):
        require(row.get("split") == "development",
                f"{costs_path}:{number} is not development")
        key = cost_identity(row)
        cost_keys.append(key)
        item = {field: row.get(field) for field in (
            "dataset_id", "sample_id", "column_id", "dtype", "dev_fold",
            "source_group", "legacy_group", "original_numeric_bytes")}
        require(item["dtype"] in {"float32", "float64", "int64"} and
                isinstance(item["dev_fold"], int) and 0 <= item["dev_fold"] < 5 and
                isinstance(item["source_group"], str) and item["source_group"] and
                isinstance(item["legacy_group"], str) and item["legacy_group"] and
                isinstance(item["original_numeric_bytes"], int) and
                item["original_numeric_bytes"] > 0,
                f"{costs_path}:{number} invalid lightweight metadata")
        metadata.append(item)
    require(len(metadata) == EXPECTED_ROWS,
            "development cost row count is not 3433")
    require(len(set(cost_keys)) == EXPECTED_ROWS, "duplicate development cost identity")
    confirm_legacy = set(protocol.get("confirm_legacy_groups", []))
    require(confirm_legacy and all(isinstance(value, str) and value
                                   for value in confirm_legacy),
            "protocol confirm legacy groups are invalid")
    thresholds = fold_thresholds(metadata, confirm_legacy)

    aggregate: dict[str, dict[str, Aggregate]] = {
        task: {"selected": Aggregate(), "control": Aggregate(),
               "selected_long": Aggregate(), "control_long": Aggregate()}
        for task in TASKS
    }
    baseline_mismatches: list[dict[str, str]] = []
    streamed_rows = 0
    for _, row in jsonl(costs_path):
        streamed_rows += 1
        dataset, column = cost_identity(row)
        dtype = row.get("dtype")
        fold = row.get("dev_fold")
        source = row.get("source_group")
        raw = row.get("original_numeric_bytes")
        require(dtype in {"float32", "float64", "int64"} and
                isinstance(fold, int) and 0 <= fold < 5 and
                isinstance(source, str) and source and
                isinstance(raw, int) and raw > 0,
                f"invalid metadata for {dataset}/{column}")
        is_long = raw >= thresholds[fold]["threshold"]
        fixed = row.get("fixed_baselines")
        require(isinstance(fixed, dict), "cost row lacks fixed baselines")
        lz4 = byte_value(fixed.get("LZ4"))
        zstd3 = byte_value(fixed.get("Zstd"))
        require(lz4 is not None and zstd3 is not None, "fixed baseline bytes unavailable")
        for task_id, domain in TASKS.items():
            key = (dataset, column, task_id)
            lookup = candidate_lookup(row, domain)
            allowed_values = [byte_value(value) for value in lookup.values()]
            allowed = min(value for value in allowed_values if value is not None)
            full = full_oracle_bytes(row, domain)
            require(full <= allowed, "full oracle exceeds allowed-pool oracle")
            for role, selectors in (("selected", selected), ("control", control)):
                require(key in selectors, f"missing {role} selector {key}")
                selector = selectors[key]
                require(selector.get("dtype") == dtype and selector.get("fold") == fold and
                        selector.get("source_group") == source and
                        selector.get("original_numeric_bytes") == raw,
                        f"{role} selector metadata mismatch {key}")
                require(selector.get("long_column_training_p90_bytes") ==
                        thresholds[fold]["threshold"] and
                        selector.get("is_long_column") is is_long,
                        f"{role} long-column contract mismatch {key}")
                task_policy = (tasks[task_id]["selected_policy"] if role == "selected"
                               else tasks[task_id]["frozen_prefix_control_policy"])
                require(selector.get("policy_id") == task_policy["id"],
                        f"{role} policy mismatch {key}")
                selected_id = selector.get("selected_candidate_id")
                candidate = lookup.get(selected_id)
                selected_bytes = byte_value(candidate)
                require(selected_bytes is not None and
                        selector.get("selected_supported_on_validation_row") is True and
                        selector.get("selected_compressed_bytes") == selected_bytes and
                        selector.get("allowed_oracle_compressed_bytes") == allowed and
                        selector.get("absolute_excess_bytes") == selected_bytes - allowed,
                        f"{role} byte lookup mismatch {key}")
                pool = selector["candidate_pool"]
                require(set(pool) <= set(lookup), f"{role} pool escapes domain {key}")
                pool_values = [byte_value(lookup[candidate_id]) for candidate_id in pool]
                dtype_oracle = min(value for value in pool_values if value is not None)
                observed_loss = math.log(selected_bytes / allowed)
                require(math.isclose(float(selector["loss_log_regret"]), observed_loss,
                                     rel_tol=1e-12, abs_tol=1e-12),
                        f"{role} log regret mismatch {key}")
                aggregate[task_id][role].add(
                    dataset=dataset, source=source, selected_id=selected_id,
                    selected=selected_bytes, full=full, allowed=allowed,
                    dtype_oracle=dtype_oracle, lz4=lz4, zstd3=zstd3, raw=raw)
                if is_long:
                    aggregate[task_id][role + "_long"].add(
                        dataset=dataset, source=source, selected_id=selected_id,
                        selected=selected_bytes, full=full, allowed=allowed,
                        dtype_oracle=dtype_oracle, lz4=lz4, zstd3=zstd3, raw=raw)
            if baseline:
                expected_id = baseline.get(key)
                actual_id = control[key]["selected_candidate_id"]
                if expected_id != actual_id:
                    baseline_mismatches.append({"dataset_id": dataset,
                                                "column_id": column,
                                                "task_id": task_id,
                                                "expected": expected_id,
                                                "actual": actual_id})

    require(streamed_rows == EXPECTED_ROWS and
            len(selected) == len(control) == EXPECTED_SELECTOR_ROWS,
            "selected/control selector coverage mismatch")
    task_reports: dict[str, Any] = {}
    for task_id in TASKS:
        selected_summary = aggregate[task_id]["selected"].summary()
        control_summary = aggregate[task_id]["control"].summary()
        require(selected_summary["unsupported_rows"] == 0 and
                control_summary["unsupported_rows"] == 0,
                f"{task_id} selected/control OOF contains an unsupported row")
        require(control_summary["selected_bytes"] == EXPECTED_CONTROL_BYTES[task_id],
                f"{task_id} frozen control byte total mismatch")
        cv_task = cv_tasks[task_id]
        policy_grid = policy_diagnostics(cv_task)
        require(tasks[task_id].get("selected_policy") == cv_task.get("selected_policy") and
                tasks[task_id].get("frozen_prefix_control_policy") ==
                    cv_task.get("frozen_prefix_control_policy") and
                tasks[task_id].get("unconstrained_minimum_byte_policy") ==
                    cv_task.get("unconstrained_minimum_byte_policy") and
                tasks[task_id].get("selection_reason") == cv_task.get("selection_reason"),
                f"{task_id} model index and CV policy metadata disagree")
        for role, independent in (("selected", selected_summary),
                                  ("control", control_summary)):
            native = cv_task[f"{role}_metrics"]
            require(native["total_selected_bytes"] == independent["selected_bytes"] and
                    native["total_allowed_oracle_bytes"] ==
                        independent["allowed_oracle_bytes"] and
                    native["unsupported_validation_rows"] ==
                        independent["unsupported_rows"] and
                    native["oof_rows"] == independent["rows"],
                    f"{task_id} native {role} count/byte metrics disagree")
            close_number(native["aggregate_byte_regret"],
                         independent["selected_to_allowed_oracle_ratio"],
                         f"{task_id}/{role}/aggregate_byte_regret")
            close_number(native["dataset_equal_geo_regret"],
                         independent["dataset_equal_geomean_regret"],
                         f"{task_id}/{role}/dataset_equal_geo_regret")
            close_number(native["worst_official_source_group_aggregate_byte_regret"],
                         independent["worst_official_source_group_aggregate_byte_regret"],
                         f"{task_id}/{role}/worst_source_group_regret")
            close_number(native["cvar95_absolute_excess_bytes"],
                         independent["cvar95_absolute_excess_bytes"],
                         f"{task_id}/{role}/cvar95_absolute_excess_bytes")
            folds = native.get("fold_metrics")
            require(isinstance(folds, list) and len(folds) == 5 and
                    {item.get("fold") for item in folds} == set(range(5)),
                    f"{task_id}/{role} native fold metrics invalid")
            for fold_item in folds:
                fold = fold_item["fold"]
                require(fold_item.get("training_bytes_p90_threshold") ==
                            thresholds[fold]["threshold"],
                        f"{task_id}/{role}/fold{fold} P90 threshold mismatch")
        constraints = {
            "zero_unsupported": selected_summary["unsupported_rows"] == 0,
            "dataset_equal_geo_regret_lte_1_05x_control":
                selected_summary["dataset_equal_geomean_regret"] <=
                1.05 * control_summary["dataset_equal_geomean_regret"] + 1e-12,
            "worst_source_group_regret_lte_control":
                selected_summary["worst_official_source_group_aggregate_byte_regret"] <=
                control_summary["worst_official_source_group_aggregate_byte_regret"] + 1e-12,
            "cvar95_absolute_excess_lte_control":
                selected_summary["cvar95_absolute_excess_bytes"] <=
                control_summary["cvar95_absolute_excess_bytes"] + 1e-12,
        }
        require(all(constraints.values()), f"{task_id} selected policy violates risk constraints")
        task_reports[task_id] = {
            "domain": TASKS[task_id],
            "allowed_candidate_count": ALLOWED_COUNTS[TASKS[task_id]],
            "full_reference_candidate_count": FULL_REFERENCE_COUNTS[TASKS[task_id]],
            "selected_policy": tasks[task_id]["selected_policy"],
            "frozen_prefix_control_policy": tasks[task_id]["frozen_prefix_control_policy"],
            "unconstrained_minimum_byte_policy": tasks[task_id]["unconstrained_minimum_byte_policy"],
            "selection_reason": tasks[task_id]["selection_reason"],
            "policy_grid": policy_grid,
            "selected": selected_summary,
            "control": control_summary,
            "selected_long_columns": aggregate[task_id]["selected_long"].summary(),
            "control_long_columns": aggregate[task_id]["control_long"].summary(),
            "constraints": constraints,
            "selected_minus_control_bytes":
                selected_summary["selected_bytes"] - control_summary["selected_bytes"],
            "selected_minus_full_oracle_bytes":
                selected_summary["selected_bytes"] - selected_summary["full_oracle_bytes"],
        }
    baseline_status = {
        **baseline_receipt,
        "expected_rows": EXPECTED_SELECTOR_ROWS,
        "mismatch_count": len(baseline_mismatches),
        "first_mismatches": baseline_mismatches[:20],
        "all_selected_ids_equal": bool(baseline) and not baseline_mismatches,
    }
    if baseline:
        require(not baseline_mismatches, "frozen control differs from old baseline selected IDs")

    return {
        "schema": "operator-oracle-risk135-report-v1",
        "status": "complete_development_only",
        "scope": {
            "development_rows": EXPECTED_ROWS,
            "development_datasets": len({row["dataset_id"] for row in metadata}),
            "tasks": list(TASKS),
            "confirm_costs_accessed": False,
            "candidate_scope": {
                "algorithm": "84 selectable; full frozen reference retains 85",
                "operator_zstd3": "97 selectable and full reference",
                "observed_pipeline": "181 actual measured slots; full frozen reference retains 182",
                "joint_cross_product_measured": False,
            },
            "NNCP": "OFF",
            "excluded_algorithm_methods": ["PPMD8"],
            "excluded_action_ids": [106, 107, 108, 109, 110],
        },
        "fold_long_column_thresholds": thresholds,
        "baseline_selected_id_parity": baseline_status,
        "tasks": task_reports,
        "inputs": {
            "development_costs": {"path": str(costs_path.resolve()),
                                  "sha256": costs_sha256},
            "cv_summary": {"path": str(cv_path.resolve()),
                           "sha256": tune_hashes["cv_summary_sha256"]},
            "model_index": {"path": str(index_path.resolve()),
                            "sha256": tune_hashes["model_index_sha256"]},
            "selected_oof": {"path": str(selected_path.resolve()),
                             "sha256": tune_hashes["selected_oof_sha256"]},
            "control_oof": {"path": str(control_path.resolve()),
                            "sha256": tune_hashes["control_oof_sha256"]},
            "tune_complete": {"path": str(tune_complete_path.resolve()),
                              "sha256": sha256_file(tune_complete_path)},
            "protocol": {"path": str(protocol_path.resolve()),
                         "sha256": sha256_file(protocol_path)},
            "quality135_v2_baseline_oof": baseline_receipt,
        },
    }


def pct_change(ratio: float | None) -> str:
    if ratio is None:
        return "不可用"
    return f"{(ratio - 1.0) * 100:+.2f}%"


def metric_text(value: Any, digits: int = 6) -> str:
    if value is None:
        return "不可用"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# 长列风险约束推荐：development-only 复核",
        "",
        "本报告只读取冻结 development 成本和两组 OOF 排名；未读取 confirm 成本。",
        "候选范围分别为 84 个算法、97 个算子和 181 个已观测条目。已观测条目是实际测量并集，不是算法与算子的笛卡尔积。",
        "",
        "| 任务 | 选择策略 | 总字节 | 相对控制 | 相对完整 oracle | 相对 LZ4 | 相对 Zstd3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"algorithm_space": "算法", "operator_zstd3_space": "算子+Zstd3",
              "observed_pipeline_space": "已观测组合"}
    for task_id, label in labels.items():
        task = report["tasks"][task_id]
        selected = task["selected"]
        control = task["control"]
        lines.append(
            f"| {label} | `{task['selected_policy']['id']}` | "
            f"{selected['selected_bytes']:,} | "
            f"{pct_change(selected['selected_bytes'] / control['selected_bytes'])} | "
            f"{pct_change(selected['selected_to_full_oracle_ratio'])} | "
            f"{pct_change(selected['selected_to_LZ4_ratio'])} | "
            f"{pct_change(selected['selected_to_Zstd3_ratio'])} |"
        )
    lines += ["", "完整 oracle 是冻结的 85/97/182 候选参考；allowed oracle 是本轮可选的 84/97/181 候选，dtype oracle 进一步受训练派生的 dtype 全支持池约束。", "",
              "| 任务 | Selected / dtype oracle | dtype / allowed oracle | allowed / full oracle |",
              "|---|---:|---:|---:|"]
    for task_id, label in labels.items():
        selected = report["tasks"][task_id]["selected"]
        lines.append(
            f"| {label} | {pct_change(selected['selected_to_dtype_pool_oracle_ratio'])} | "
            f"{pct_change(selected['dtype_pool_oracle_to_allowed_oracle_ratio'])} | "
            f"{pct_change(selected['allowed_oracle_to_full_oracle_ratio'])} |")

    lines += ["", "## 全部 CV 策略与独立择优检查", "",
              "每个任务按 `(total_selected_bytes, policy_id)` 稳定排序。`可行` 由本报告根据 control 的三个风险上限和零 unsupported 独立重算。", ""]
    for task_id, label in labels.items():
        grid = report["tasks"][task_id]["policy_grid"]
        diagnostic = grid["diagnostic_policies"]
        roles: dict[str, list[str]] = collections.defaultdict(list)
        for role, item in diagnostic.items():
            roles[item["policy"]["id"]].append(role)
        lines += [f"### {label}（{grid['policy_count']} 项）", "",
                  "| Policy | 特征 | 权重 | 总字节 | 相对 control | 可行 | 角色 |",
                  "|---|---|---|---:|---:|:---:|---|"]
        control_bytes = diagnostic["control"]["total_selected_bytes"]
        for item in grid["all_policy_summaries"]:
            policy = item["policy"]
            policy_id = policy["id"]
            role_text = ", ".join(roles.get(policy_id, [])) or "-"
            lines.append(
                f"| `{policy_id}` | {policy['feature_variant']} | "
                f"{policy['weight_mode']} | {item['total_selected_bytes']:,} | "
                f"{pct_change(item['total_selected_bytes'] / control_bytes)} | "
                f"{'是' if item['risk_feasible_against_frozen_control'] else '否'} | "
                f"{role_text} |")
        lines += ["",
                  f"独立稳定择优结果：`{grid['independent_selected_policy_id']}`；"
                  f"原因：`{grid['independent_selection_reason']}`。", ""]

    lines += ["## 长列与风险约束", "",
              "每折只用该折训练列的 `original_numeric_bytes` 计算 nearest-rank P90；heldout 列字节数大于等于阈值即记为长列。"]
    for task_id, label in labels.items():
        task = report["tasks"][task_id]
        selected = task["selected"]
        control = task["control"]
        long_selected = task["selected_long_columns"]
        long_control = task["control_long_columns"]
        lines += [
            "",
            f"- {label}：长列 {long_selected['rows']} 项，总字节 "
            f"{long_selected['selected_bytes']:,}（控制 {long_control['selected_bytes']:,}）；"
            f"全列最差来源组 `{selected['worst_official_source_group']}` 的 regret="
            f"{metric_text(selected['worst_official_source_group_aggregate_byte_regret'])}，"
            f"CVaR95 绝对超额={metric_text(selected['cvar95_absolute_excess_bytes'], 2)} bytes。"
        ]
    parity = report["baseline_selected_id_parity"]
    lines += ["", "## 审计边界", "",
              f"- control OOF 固定字节总量和 10,299 条覆盖已通过；quality135 v2 旧模型逐项 ID 对照状态：`{parity['status']}`，差异 {parity['mismatch_count']}。",
              "- 三项风险约束均由成本矩阵独立重算后通过；native 汇总只作为交叉检查。",
              "- 完整 oracle 保留 85/97/182 的冻结参考；可选池排除 PPMD8，因此算法与已观测组合的 allowed-oracle 差距单独保留。",
              "- 算子和已观测组合没有可比的历史时间排名；本轮只优化和报告空间目标。",
              "- NNCP、PPMd/DMC/PAQ8/CMIX/LSTM 及 A106–A110 继续禁用。",
              ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True,
                        help="frozen quality135 v2 chosen OOF for strict 10,299-ID parity")
    parser.add_argument("--output", type=Path,
                        help="default: ROOT/report")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = (arguments.output.resolve() if arguments.output else root / "report")
    require(root.is_dir(), f"ROOT is not a directory: {root}")
    require(not output.exists(), f"report output already exists: {output}")
    report = recompute(root, arguments.baseline_oof)
    write_new_json(output / "REPORT.json", report)
    write_new_text(output / "RESULTS.zh.md", markdown_report(report))
    complete = {
        "schema": "operator-oracle-risk135-report-complete-v1",
        "status": "complete_development_only",
        "confirm_costs_accessed": False,
        "report_sha256": sha256_file(output / "REPORT.json"),
        "results_zh_sha256": sha256_file(output / "RESULTS.zh.md"),
        "reporter_source": str(Path(__file__).resolve()),
        "reporter_source_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {name: item["sha256"] for name, item in report["inputs"].items()
                         if isinstance(item, dict) and isinstance(item.get("sha256"), str)},
    }
    write_new_json(output / "COMPLETE.json", complete)
    print(f"risk report complete: {output}")


if __name__ == "__main__":
    main()
