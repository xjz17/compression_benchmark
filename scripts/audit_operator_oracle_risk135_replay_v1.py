#!/usr/bin/env python3
"""Independently audit all development-only risk-policy OOF replays.

The only outcome matrix read by this program is DEVELOPMENT_COSTS.jsonl.  It
never discovers or opens a confirm-cost path.  Candidate byte costs are packed
into flat int64 arrays so all 55 replay files can be checked under the 2 GiB
address-space limit without retaining their JSON rows.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, cast


TASK_DOMAINS = {
    "algorithm_space": "algorithm",
    "operator_zstd3_space": "operator_zstd3",
    "observed_pipeline_space": "observed_pipeline",
}
EXPECTED_CANDIDATES = {
    "algorithm_space": 84,
    "operator_zstd3_space": 97,
    "observed_pipeline_space": 181,
}
EXPECTED_POLICIES = {
    "algorithm_space": 19,
    "operator_zstd3_space": 18,
    "observed_pipeline_space": 18,
}
EXPECTED_CONTROL_BYTES = {
    "algorithm_space": 2_941_000_129,
    "operator_zstd3_space": 2_662_068_813,
    "observed_pipeline_space": 2_678_889_909,
}
EXPECTED_ROWS = 3433
UNSUPPORTED_PENALTY = 1000.0
EPSILON = 1e-12
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path} is not a JSON object")
    if schema is not None:
        require(value.get("schema") == schema,
                f"{path} has unexpected schema {value.get('schema')!r}")
    return value


def jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as source:
        for number, line in enumerate(source, 1):
            require(line.endswith("\n"), f"{path}:{number} lacks final newline")
            require(bool(line.strip()), f"{path}:{number} is blank")
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{number} is not an object")
            yield number, value


def write_new_json(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    require(not temporary.exists(), f"temporary output exists: {temporary}")
    with temporary.open("x") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def column_id(row: dict[str, Any], context: str) -> str:
    value = row.get("column_id", row.get("sample_id"))
    require(bool(isinstance(value, str) and value),
            f"{context} lacks column identity")
    return cast(str, value)


def finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)) and float(value) > 0.0)


def candidate_bytes(value: Any) -> int:
    if not isinstance(value, dict) or value.get("eligible") is not True:
        return -1
    if not finite_positive(value.get("space_cost")):
        return -1
    compressed = value.get("compressed_bytes")
    if (not isinstance(compressed, int) or isinstance(compressed, bool) or
            compressed <= 0 or compressed > (1 << 63) - 1):
        return -1
    return compressed


@dataclass
class TaskCosts:
    candidate_ids: tuple[str, ...] = ()
    candidate_index: dict[str, int] = field(default_factory=dict)
    # Row-major signed int64.  -1 denotes unsupported/unmeasured.
    bytes_flat: array = field(default_factory=lambda: array("q"))
    allowed_oracle: array = field(default_factory=lambda: array("Q"))


@dataclass
class CostCache:
    keys: list[tuple[str, str]] = field(default_factory=list)
    key_to_row: dict[tuple[str, str], int] = field(default_factory=dict)
    datasets: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    legacy_groups: list[str] = field(default_factory=list)
    folds: array = field(default_factory=lambda: array("B"))
    original_bytes: array = field(default_factory=lambda: array("Q"))
    tasks: dict[str, TaskCosts] = field(
        default_factory=lambda: {task: TaskCosts() for task in TASK_DOMAINS})


def load_cost_cache(path: Path) -> CostCache:
    cache = CostCache()
    for number, row in jsonl(path):
        context = f"{path}:{number}"
        require(row.get("split") == "development", f"{context} is not development")
        dataset = row.get("dataset_id")
        require(bool(isinstance(dataset, str) and dataset),
                f"{context} lacks dataset_id")
        dataset = cast(str, dataset)
        column = column_id(row, context)
        key = (dataset, column)
        require(key not in cache.key_to_row, f"{context} duplicates {key}")
        dtype = row.get("dtype")
        source = row.get("source_group")
        legacy = row.get("legacy_group")
        fold = row.get("dev_fold")
        original = row.get("original_numeric_bytes")
        require(dtype in {"float32", "float64", "int64"},
                f"{context} has invalid dtype")
        require(bool(isinstance(source, str) and source and
                isinstance(legacy, str) and legacy),
                f"{context} has invalid source/legacy group")
        require(isinstance(fold, int) and not isinstance(fold, bool) and 0 <= fold < 5,
                f"{context} has invalid development fold")
        require(isinstance(original, int) and not isinstance(original, bool) and
                0 < original <= (1 << 64) - 1,
                f"{context} has invalid original_numeric_bytes")
        domains = row.get("domains")
        require(isinstance(domains, dict), f"{context} lacks domains")
        dtype = cast(str, dtype)
        source = cast(str, source)
        legacy = cast(str, legacy)
        fold = cast(int, fold)
        original = cast(int, original)
        domains = cast(dict[str, Any], domains)

        row_index = len(cache.keys)
        cache.key_to_row[key] = row_index
        cache.keys.append(key)
        cache.datasets.append(dataset)
        cache.dtypes.append(dtype)
        cache.sources.append(source)
        cache.legacy_groups.append(legacy)
        cache.folds.append(fold)
        cache.original_bytes.append(original)

        for task_id, domain in TASK_DOMAINS.items():
            values = domains.get(domain)
            require(isinstance(values, list) and
                    len(values) == EXPECTED_CANDIDATES[task_id],
                    f"{context}/{domain} has wrong candidate count")
            values = cast(list[Any], values)
            ids: list[str] = []
            packed: list[int] = []
            seen: set[str] = set()
            for candidate in values:
                require(isinstance(candidate, dict),
                        f"{context}/{domain} candidate is not an object")
                candidate_id = candidate.get("candidate_id")
                require(bool(isinstance(candidate_id, str) and candidate_id and
                        candidate_id not in seen),
                        f"{context}/{domain} has invalid candidate ID")
                candidate_id = cast(str, candidate_id)
                seen.add(candidate_id)
                ids.append(candidate_id)
                packed.append(candidate_bytes(candidate))
            task = cache.tasks[task_id]
            if not task.candidate_ids:
                task.candidate_ids = tuple(ids)
                task.candidate_index = {value: index for index, value in enumerate(ids)}
            else:
                require(tuple(ids) == task.candidate_ids,
                        f"{context}/{domain} candidate order/catalog changed")
            supported = [value for value in packed if value > 0]
            require(bool(supported),
                    f"{context}/{domain} has no supported byte cost")
            task.bytes_flat.extend(packed)
            task.allowed_oracle.append(min(supported))

    require(len(cache.keys) == EXPECTED_ROWS,
            f"development cost matrix has {len(cache.keys)} rows, expected {EXPECTED_ROWS}")
    require(len(set(cache.datasets)) == 127,
            "development cost matrix must contain 127 datasets")
    for task_id, task in cache.tasks.items():
        expected = EXPECTED_ROWS * EXPECTED_CANDIDATES[task_id]
        require(len(task.bytes_flat) == expected,
                f"{task_id} packed cost shape mismatch")
        require(len(task.allowed_oracle) == EXPECTED_ROWS,
                f"{task_id} allowed-oracle shape mismatch")
    return cache


@dataclass
class MetricAccumulator:
    rows: int = 0
    unsupported: int = 0
    selected_bytes: int = 0
    allowed_oracle_bytes: int = 0
    dataset_losses: dict[str, list[float | int]] = field(default_factory=dict)
    source_bytes: dict[str, list[int]] = field(default_factory=dict)
    excess_bytes: list[int] = field(default_factory=list)

    def add(self, dataset: str, source: str, selected: int | None,
            oracle: int, loss: float) -> None:
        self.rows += 1
        state = self.dataset_losses.setdefault(dataset, [0.0, 0])
        state[0] = float(state[0]) + loss
        state[1] = int(state[1]) + 1
        if selected is None:
            self.unsupported += 1
            return
        self.selected_bytes += selected
        self.allowed_oracle_bytes += oracle
        group = self.source_bytes.setdefault(source, [0, 0])
        group[0] += selected
        group[1] += oracle
        self.excess_bytes.append(selected - oracle)

    def summary(self, *, null_risk_when_unsupported: bool) -> dict[str, Any]:
        dataset_geo = math.exp(
            sum(float(total) / int(count)
                for total, count in self.dataset_losses.values()) /
            len(self.dataset_losses))
        if self.unsupported and null_risk_when_unsupported:
            aggregate = worst_source = cvar = None
        elif self.unsupported:
            aggregate = worst_source = cvar = None
        else:
            aggregate = self.selected_bytes / self.allowed_oracle_bytes
            worst_source = max(selected / oracle
                               for selected, oracle in self.source_bytes.values())
            tail = math.ceil(0.05 * len(self.excess_bytes))
            require(tail > 0, "CVaR tail is empty")
            cvar = sum(sorted(self.excess_bytes, reverse=True)[:tail]) / tail
        return {
            "total_selected_bytes": self.selected_bytes,
            "total_allowed_oracle_bytes": self.allowed_oracle_bytes,
            "aggregate_byte_regret": aggregate,
            "dataset_equal_geo_regret": dataset_geo,
            "worst_official_source_group_aggregate_byte_regret": worst_source,
            "cvar95_absolute_excess_bytes": cvar,
            "unsupported_validation_rows": self.unsupported,
            "oof_rows": self.rows,
        }


@dataclass
class FoldAccumulator:
    aggregate: MetricAccumulator = field(default_factory=MetricAccumulator)
    threshold: int | None = None
    long_rows: int = 0


def equal_number(observed: Any, expected: Any, context: str) -> None:
    if observed is None or expected is None:
        require(observed is expected, f"{context}: {observed!r} != {expected!r}")
        return
    require(isinstance(observed, (int, float)) and not isinstance(observed, bool) and
            isinstance(expected, (int, float)) and not isinstance(expected, bool) and
            math.isfinite(float(observed)) and math.isfinite(float(expected)) and
            math.isclose(float(observed), float(expected),
                         rel_tol=EPSILON, abs_tol=EPSILON),
            f"{context}: {observed!r} != {expected!r}")


OVERALL_FIELDS = (
    "total_selected_bytes",
    "total_allowed_oracle_bytes",
    "aggregate_byte_regret",
    "dataset_equal_geo_regret",
    "worst_official_source_group_aggregate_byte_regret",
    "cvar95_absolute_excess_bytes",
    "unsupported_validation_rows",
    "oof_rows",
)


def compare_metrics(independent: dict[str, Any], native: dict[str, Any],
                    context: str) -> None:
    require(isinstance(native, dict), f"{context} native metrics are absent")
    for name in OVERALL_FIELDS:
        if name in {"total_selected_bytes", "total_allowed_oracle_bytes",
                    "unsupported_validation_rows", "oof_rows"}:
            require(native.get(name) == independent[name],
                    f"{context}/{name}: {native.get(name)!r} != {independent[name]!r}")
        else:
            equal_number(native.get(name), independent[name], f"{context}/{name}")


def fold_summary(fold: int, value: FoldAccumulator) -> dict[str, Any]:
    base = value.aggregate.summary(null_risk_when_unsupported=True)
    # Native compact fold metrics use null for every risk metric, including the
    # dataset geomean, whenever that fold has an unsupported row.
    dataset_geo = (None if value.aggregate.unsupported else
                   base["dataset_equal_geo_regret"])
    return {
        "fold": fold,
        "oof_rows": base["oof_rows"],
        "unsupported_rows": base["unsupported_validation_rows"],
        "selected_bytes": base["total_selected_bytes"],
        "allowed_oracle_bytes": base["total_allowed_oracle_bytes"],
        "aggregate_byte_regret": base["aggregate_byte_regret"],
        "dataset_equal_geo_regret": dataset_geo,
        "worst_official_source_group_aggregate_byte_regret":
            base["worst_official_source_group_aggregate_byte_regret"],
        "cvar95_absolute_excess_bytes": base["cvar95_absolute_excess_bytes"],
        "training_bytes_p90_threshold": value.threshold,
        "long_column_rows": value.long_rows,
    }


def compare_fold_metrics(independent: list[dict[str, Any]], native: Any,
                         context: str) -> None:
    require(isinstance(native, list) and len(native) == 5,
            f"{context} native fold metrics are invalid")
    by_fold = {item.get("fold"): item for item in native if isinstance(item, dict)}
    require(set(by_fold) == set(range(5)), f"{context} native folds are incomplete")
    exact = ("oof_rows", "unsupported_rows", "selected_bytes",
             "allowed_oracle_bytes", "training_bytes_p90_threshold",
             "long_column_rows")
    numeric = ("aggregate_byte_regret", "dataset_equal_geo_regret",
               "worst_official_source_group_aggregate_byte_regret",
               "cvar95_absolute_excess_bytes")
    for item in independent:
        fold = item["fold"]
        observed = by_fold[fold]
        for name in exact:
            require(observed.get(name) == item[name],
                    f"{context}/fold{fold}/{name}: "
                    f"{observed.get(name)!r} != {item[name]!r}")
        for name in numeric:
            equal_number(observed.get(name), item[name],
                         f"{context}/fold{fold}/{name}")


def audit_replay(path: Path, task_id: str, policy_id: str,
                 costs: CostCache, native: dict[str, Any]) -> dict[str, Any]:
    task_costs = costs.tasks[task_id]
    candidate_count = len(task_costs.candidate_ids)
    seen = bytearray(EXPECTED_ROWS)
    aggregate = MetricAccumulator()
    folds = [FoldAccumulator() for _ in range(5)]

    for number, row in jsonl(path):
        context = f"{path}:{number}"
        require(row.get("schema") == "operator-oracle-risk135-oof-row-v1",
                f"{context} has unexpected schema")
        require(row.get("task_id") == task_id and row.get("policy_id") == policy_id,
                f"{context} task/policy mismatch")
        require(row.get("query_costs_used_for_selection") is False,
                f"{context} says query costs entered selection")
        require(all(name not in row for name in
                    ("ranking_candidate_ids", "ranking_predicted_log_costs",
                     "candidate_pool")),
                f"{context} unexpectedly contains full ranking data")
        dataset = row.get("dataset_id")
        require(bool(isinstance(dataset, str) and dataset),
                f"{context} lacks dataset_id")
        dataset = cast(str, dataset)
        key = (dataset, column_id(row, context))
        require(key in costs.key_to_row, f"{context} identity is outside development")
        index = costs.key_to_row[key]
        require(seen[index] == 0, f"{context} duplicates development identity {key}")
        seen[index] = 1
        fold = costs.folds[index]
        original = costs.original_bytes[index]
        require(row.get("fold") == fold and row.get("dtype") == costs.dtypes[index] and
                row.get("source_group") == costs.sources[index] and
                row.get("legacy_group") == costs.legacy_groups[index] and
                row.get("original_numeric_bytes") == original,
                f"{context} metadata disagrees with development costs")

        threshold = row.get("long_column_training_p90_bytes")
        require(isinstance(threshold, int) and not isinstance(threshold, bool) and
                threshold > 0,
                f"{context} has invalid long-column threshold")
        fold_state = folds[fold]
        if fold_state.threshold is None:
            fold_state.threshold = threshold
        else:
            require(fold_state.threshold == threshold,
                    f"{context} changes the fold's long-column threshold")
        is_long = original >= threshold
        require(row.get("is_long_column") is is_long,
                f"{context} has incorrect long-column classification")
        if is_long:
            fold_state.long_rows += 1

        selected_id = row.get("selected_candidate_id")
        selected: int | None = None
        if isinstance(selected_id, str):
            candidate_index = task_costs.candidate_index.get(selected_id)
            require(candidate_index is not None,
                    f"{context} selects candidate outside {task_id}")
            candidate_index = cast(int, candidate_index)
            packed = task_costs.bytes_flat[index * candidate_count + candidate_index]
            if packed > 0:
                selected = packed
        else:
            require(selected_id is None, f"{context} has invalid selected candidate")
        oracle = task_costs.allowed_oracle[index]
        if selected is None:
            loss = math.log(UNSUPPORTED_PENALTY)
            expected_penalty: float | None = UNSUPPORTED_PENALTY
            expected_excess: int | None = None
        else:
            require(selected >= oracle,
                    f"{context} selected bytes are below the allowed oracle")
            loss = math.log(selected / oracle)
            expected_penalty = None
            expected_excess = selected - oracle
        require(row.get("selected_supported_on_validation_row") is
                    (selected is not None) and
                row.get("selected_compressed_bytes") == selected and
                row.get("allowed_oracle_compressed_bytes") == oracle and
                row.get("absolute_excess_bytes") == expected_excess and
                row.get("unsupported_penalty") == expected_penalty,
                f"{context} row cost/support fields disagree with cost matrix")
        equal_number(row.get("loss_log_regret"), loss,
                     f"{context}/loss_log_regret")
        aggregate.add(dataset, costs.sources[index], selected, oracle, loss)
        fold_state.aggregate.add(dataset, costs.sources[index], selected, oracle, loss)

    require(all(seen), f"{path} does not cover all {EXPECTED_ROWS} development rows")
    independent = aggregate.summary(null_risk_when_unsupported=False)
    independent_folds = [fold_summary(index, value)
                         for index, value in enumerate(folds)]
    compare_metrics(independent, native, f"{task_id}/{policy_id}")
    compare_fold_metrics(independent_folds, native.get("fold_metrics"),
                         f"{task_id}/{policy_id}")
    return {
        "policy_id": policy_id,
        "replay": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "metrics": {**independent, "fold_metrics": independent_folds},
        "native_cv_summary_match": True,
    }


def risk_feasible(candidate: dict[str, Any], control: dict[str, Any]) -> bool:
    return (
        candidate["unsupported_validation_rows"] == 0 and
        candidate["dataset_equal_geo_regret"] <=
            1.05 * control["dataset_equal_geo_regret"] + EPSILON and
        candidate["worst_official_source_group_aggregate_byte_regret"] <=
            control["worst_official_source_group_aggregate_byte_regret"] + EPSILON and
        candidate["cvar95_absolute_excess_bytes"] <=
            control["cvar95_absolute_excess_bytes"] + EPSILON
    )


def policy_id(value: Any, context: str) -> str:
    require(isinstance(value, dict), f"{context} is not a policy object")
    result = value.get("id")
    require(isinstance(result, str) and SAFE_ID.fullmatch(result) is not None,
            f"{context} has unsafe/invalid policy ID")
    return result


def audit(root: Path) -> dict[str, Any]:
    costs_path = root / "costs" / "DEVELOPMENT_COSTS.jsonl"
    cv_path = root / "models" / "CV_SUMMARY.json"
    replay_root = root / "verification" / "replay"
    require(costs_path.is_file(), f"missing {costs_path}")
    require(cv_path.is_file(), f"missing {cv_path}")
    require(replay_root.is_dir(), f"missing {replay_root}")
    cv = read_json(cv_path, "operator-oracle-risk135-development-cv-v1")
    require(cv.get("status") == "complete" and
            cv.get("scope") == "development_only_model_selection_not_external_evaluation" and
            cv.get("confirm_used_for_tuning") is False,
            "CV summary has unsafe or incomplete scope")
    costs_sha256 = sha256_file(costs_path)
    require(cv.get("inputs", {}).get("development_costs", {}).get("sha256") ==
                costs_sha256,
            "CV summary is not bound to the development cost matrix")
    costs = load_cost_cache(costs_path)

    cv_tasks_input = cv.get("tasks")
    require(isinstance(cv_tasks_input, list), "CV summary lacks tasks")
    cv_tasks_input = cast(list[Any], cv_tasks_input)
    cv_tasks: dict[str, dict[str, Any]] = {}
    for item in cv_tasks_input:
        require(isinstance(item, dict), "CV task is not an object")
        task_id = item.get("task_id")
        require(task_id in TASK_DOMAINS and task_id not in cv_tasks,
                f"CV summary has invalid/duplicate task {task_id!r}")
        task_id = cast(str, task_id)
        cv_tasks[task_id] = item
    require(set(cv_tasks) == set(TASK_DOMAINS),
            "CV summary must contain exactly the three space tasks")

    report_tasks: dict[str, Any] = {}
    all_replay_hashes: dict[str, str] = {}
    affected_tie_policies: dict[str, list[str]] = {}
    total_policies = 0
    for task_id in TASK_DOMAINS:
        cv_task = cv_tasks[task_id]
        policies_input = cv_task.get("policies")
        require(isinstance(policies_input, list) and
                len(policies_input) == EXPECTED_POLICIES[task_id],
                f"{task_id} has wrong policy count")
        policies_input = cast(list[Any], policies_input)
        native_by_id: dict[str, dict[str, Any]] = {}
        policy_objects: dict[str, dict[str, Any]] = {}
        for native in policies_input:
            require(isinstance(native, dict), f"{task_id} policy summary is invalid")
            policy = native.get("policy")
            current_id = policy_id(policy, f"{task_id} policy")
            require(current_id not in native_by_id,
                    f"{task_id} duplicates policy {current_id}")
            native_by_id[current_id] = native
            policy_objects[current_id] = cast(dict[str, Any], policy)
        task_replay_dir = replay_root / task_id
        require(task_replay_dir.is_dir(), f"missing replay directory {task_replay_dir}")
        expected_names = {value + ".jsonl" for value in native_by_id}
        observed_names = {path.name for path in task_replay_dir.glob("*.jsonl")}
        require(observed_names == expected_names,
                f"{task_id} replay files differ from CV policy set: "
                f"missing={sorted(expected_names - observed_names)}, "
                f"extra={sorted(observed_names - expected_names)}")

        audited: dict[str, dict[str, Any]] = {}
        for current_id in sorted(native_by_id):
            path = task_replay_dir / (current_id + ".jsonl")
            require(path.is_file(), f"missing replay {path}")
            result = audit_replay(path, task_id, current_id, costs,
                                  native_by_id[current_id])
            metrics = result["metrics"]
            audited[current_id] = result
            relative = str(path.relative_to(root))
            all_replay_hashes[relative] = result["replay"]["sha256"]
            if int(policy_objects[current_id].get("knn_k", 0)) > 0:
                affected_tie_policies.setdefault(task_id, []).append(current_id)

        control_id = policy_id(cv_task.get("frozen_prefix_control_policy"),
                               f"{task_id} frozen control")
        selected_id = policy_id(cv_task.get("selected_policy"),
                                f"{task_id} selected policy")
        unconstrained_id = policy_id(cv_task.get("unconstrained_minimum_byte_policy"),
                                     f"{task_id} unconstrained policy")
        require(control_id in audited and selected_id in audited and
                unconstrained_id in audited,
                f"{task_id} role policy is outside audited grid")
        control_metrics = audited[control_id]["metrics"]
        require(control_metrics["unsupported_validation_rows"] == 0,
                f"{task_id} frozen control contains unsupported rows")
        require(control_metrics["total_selected_bytes"] ==
                    EXPECTED_CONTROL_BYTES[task_id],
                f"{task_id} frozen control byte total differs from baseline")

        for current_id, result in audited.items():
            metrics = result["metrics"]
            feasible = risk_feasible(metrics, control_metrics)
            strict = (metrics["unsupported_validation_rows"] == 0 and
                      metrics["total_selected_bytes"] <
                          control_metrics["total_selected_bytes"])
            native = native_by_id[current_id]
            require(native.get("risk_feasible_against_frozen_control") is feasible,
                    f"{task_id}/{current_id} native feasibility flag mismatch")
            require(native.get("strict_total_byte_improvement_over_control") is strict,
                    f"{task_id}/{current_id} native strict-improvement flag mismatch")
            result["risk_feasible_against_frozen_control"] = feasible
            result["strict_total_byte_improvement_over_control"] = strict

        supported = [value for value in audited.values()
                     if value["metrics"]["unsupported_validation_rows"] == 0]
        require(bool(supported), f"{task_id} has no fully supported policy")
        stable_key = lambda value: (
            value["metrics"]["total_selected_bytes"], value["policy_id"])
        independent_unconstrained = min(supported, key=stable_key)["policy_id"]
        feasible_strict = [value for value in audited.values()
                           if value["risk_feasible_against_frozen_control"] and
                           value["strict_total_byte_improvement_over_control"]]
        if feasible_strict:
            independent_selected = min(feasible_strict, key=stable_key)["policy_id"]
            independent_reason = (
                "minimum_total_selected_bytes_among_feasible_strict_improvements")
        else:
            independent_selected = control_id
            independent_reason = (
                "no_feasible_strict_total_byte_improvement_choose_frozen_control")
        require(independent_unconstrained == unconstrained_id,
                f"{task_id} native unconstrained minimum is incorrect")
        require(independent_selected == selected_id,
                f"{task_id} native selected policy is incorrect")
        require(cv_task.get("selection_reason") == independent_reason,
                f"{task_id} native selection reason is incorrect")
        compare_metrics(audited[selected_id]["metrics"],
                        cast(dict[str, Any], cv_task.get("selected_metrics")),
                        f"{task_id}/selected_metrics")
        compare_metrics(audited[control_id]["metrics"],
                        cast(dict[str, Any], cv_task.get("control_metrics")),
                        f"{task_id}/control_metrics")

        report_tasks[task_id] = {
            "policy_count": len(audited),
            "frozen_control_policy_id": control_id,
            "frozen_control_bytes": control_metrics["total_selected_bytes"],
            "unconstrained_minimum_policy_id": independent_unconstrained,
            "selected_policy_id": independent_selected,
            "selection_reason": independent_reason,
            "all_native_policy_metrics_match": True,
            "all_replay_rows_cost_joined": True,
            "policies": [audited[value] for value in sorted(audited)],
        }
        total_policies += len(audited)

    require(total_policies == 55, f"audited {total_policies} policies, expected 55")
    return {
        "schema": "operator-oracle-risk135-replay-audit-v1",
        "status": "pass",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": {
            "development_rows": EXPECTED_ROWS,
            "development_datasets": 127,
            "task_count": len(TASK_DOMAINS),
            "policy_count": total_policies,
            "confirm_costs_accessed": False,
            "external_evaluation_performed": False,
        },
        "method": {
            "cost_join": "single-pass packed int64 candidate-byte matrix",
            "replay_processing": "one JSONL policy at a time",
            "metrics": [
                "total selected and allowed-oracle bytes",
                "dataset-equal geometric regret",
                "worst official source-group aggregate byte regret",
                "CVaR95 absolute excess bytes",
                "unsupported rows",
                "five fold compact metrics",
            ],
            "selection": "stable minimum (total_selected_bytes, policy_id) among independently feasible strict improvements; otherwise frozen control",
            "independence": "Python cost join and aggregation do not consume native aggregate values",
        },
        "checks": {
            "all_55_policy_replays_present": True,
            "all_55_policy_rows_match_development_costs": True,
            "all_55_policy_native_summaries_match": True,
            "all_three_control_byte_totals_match_frozen_baseline": True,
            "all_three_unconstrained_minima_match": True,
            "all_three_risk_feasible_selections_match": True,
        },
        "limitations": [
            {
                "id": "development_oof_used_for_model_selection",
                "detail": "This audit verifies development-only model selection; it is not an unbiased external estimate.",
            },
            {
                "id": "knn_equal_distance_identifier_tiebreak",
                "detail": "The frozen native implementation orders equal-distance neighbors by sample_id, which contains dataset_id. This is a deterministic ordering tie-break, not a numeric model feature, but kth-boundary ties can affect blend scores. The replay lacks neighbor distances, so this audit discloses rather than quantifies that limitation.",
                "affected_policies": affected_tie_policies,
                "semantic_change_deferred": True,
            },
        ],
        "inputs": {
            "development_costs": {
                "path": str(costs_path.resolve()),
                "sha256": costs_sha256,
            },
            "cv_summary": {
                "path": str(cv_path.resolve()),
                "sha256": sha256_file(cv_path),
            },
            "replay_files": all_replay_hashes,
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "tasks": report_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    require(root.is_dir(), f"ROOT is not a directory: {root}")
    output = root / "verification" / "REPLAY_AUDIT.json"
    result = audit(root)
    write_new_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "policies": result["scope"]["policy_count"],
        "output": str(output),
        "output_sha256": sha256_file(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
