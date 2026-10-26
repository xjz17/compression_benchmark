#!/usr/bin/env python3
"""Offline quality report for the frozen current-135 recommendation study.

The default command performs the one final confirm evaluation.  It validates
LOCK.json and CONFIRM_GATE.json, including all declared hashes, before it opens
CONFIRM_COSTS.jsonl.  ``--development-only`` never stats, hashes, or opens the
confirm cost file and is intended for pre-gate review and local fixtures.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


COMMON_TIME_CONTRACT = "benchmark_host-current135-valid-value-w1-r3-t1-v1"
UNSUPPORTED_PENALTY_RATIO = 1000.0
FIXED_BASELINES = (("LZ4", "LZ4"), ("Zstd3", "Zstd"))
TASKS = {
    "algorithm_space": ("algorithm", "space_cost"),
    "algorithm_encode": ("algorithm", "encode_cost"),
    "algorithm_decode": ("algorithm", "decode_cost"),
    "algorithm_balanced": ("algorithm", "balanced_cost"),
    "operator_zstd3_space": ("operator_zstd3", "space_cost"),
    "observed_pipeline_space": ("observed_pipeline", "space_cost"),
}
DEFAULT_PATHS = {
    "development_costs": "costs/DEVELOPMENT_COSTS.jsonl",
    "confirm_costs": "costs/CONFIRM_COSTS.jsonl",
    "model_index": "models/MODEL_INDEX.json",
    "chosen_oof": "models/CHOSEN_OOF_FULL_RANKS.jsonl",
    "all_predictions": "predictions/ALL_PREDICTIONS.jsonl",
    "prediction_receipt": "predictions/ALL_PREDICTIONS.jsonl.receipt.json",
    "candidate_catalog": "costs/CANDIDATES.json",
    "features_receipt": "features/FEATURES_RECEIPT.json",
    "cv_summary": "models/CV_SUMMARY.json",
}


class ReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_new_text(path: Path, value: str) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    require(not temporary.exists(), f"temporary output already exists {temporary}")
    with temporary.open("x") as target:
        target.write(value)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def write_new_json(path: Path, value: Any) -> None:
    write_new_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path} is not a JSON object")
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


def key(dataset_id: str, column_id: str, task_id: str) -> tuple[str, str, str]:
    return dataset_id, column_id, task_id


def artifact_path(root: Path, lock: dict[str, Any], name: str) -> tuple[Path, str]:
    artifacts = lock.get("artifacts")
    require(isinstance(artifacts, dict), "LOCK.json lacks artifacts object")
    entry = artifacts.get(name)
    require(isinstance(entry, dict), f"LOCK.json lacks artifact {name}")
    raw_path = entry.get("path", DEFAULT_PATHS.get(name))
    expected = entry.get("sha256")
    require(isinstance(raw_path, str) and raw_path, f"artifact {name} lacks path")
    require(valid_sha(expected), f"artifact {name} lacks lowercase SHA256")
    path = Path(raw_path)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path == root or root in path.parents,
            f"artifact {name} escapes ROOT: {path}")
    return path, expected


def verify_artifact(root: Path, lock: dict[str, Any], name: str) -> Path:
    path, expected = artifact_path(root, lock, name)
    require(path.is_file(), f"artifact {name} is missing: {path}")
    require(sha256_file(path) == expected, f"artifact {name} SHA256 mismatch")
    return path


def load_lock(root: Path, development_only: bool) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    lock_path = root / "LOCK.json"
    lock = read_json(lock_path, "operator-oracle-quality135-lock-v1")
    require(lock.get("status") == "locked", "LOCK.json status is not locked")
    require(lock.get("confirm_used_for_fit_or_selection") is False,
            "LOCK says confirm entered fitting or selection")
    require(lock.get("confirm_rows_read_before_lock") == 0,
            "LOCK says confirm rows were read before locking")
    required = [
        "development_costs", "model_index", "chosen_oof",
        "candidate_catalog", "features_receipt", "cv_summary",
    ]
    if not development_only:
        # all_predictions is label-free, so validate it before opening confirm.
        required.extend(["all_predictions", "prediction_receipt"])
    paths = {name: verify_artifact(root, lock, name) for name in required}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hashes["lock"] = sha256_file(lock_path)
    return lock, paths, hashes


def unlock_confirm(root: Path, lock: dict[str, Any], hashes: dict[str, str]) -> tuple[Path, dict[str, Any], str]:
    """Validate both gates before touching the confirm-cost payload."""
    gate_path = root / "CONFIRM_GATE.json"
    gate = read_json(gate_path, "operator-oracle-quality135-confirm-gate-v1")
    require(gate.get("status") == "open_for_single_final_evaluation",
            "CONFIRM_GATE status does not permit final evaluation")
    require(gate.get("confirm_used_for_fit_or_selection") is False,
            "CONFIRM_GATE says confirm entered fitting or selection")
    require(gate.get("lock_sha256") == hashes["lock"],
            "CONFIRM_GATE does not bind the current LOCK.json")
    for gate_key, artifact_name in [
        ("model_index_sha256", "model_index"),
        ("all_predictions_sha256", "all_predictions"),
        ("prediction_receipt_sha256", "prediction_receipt"),
    ]:
        require(gate.get(gate_key) == hashes[artifact_name],
                f"CONFIRM_GATE {gate_key} mismatch")
    confirm_path, declared = artifact_path(root, lock, "confirm_costs")
    require(gate.get("confirm_costs_sha256") == declared,
            "CONFIRM_GATE and LOCK disagree on confirm-cost SHA256")
    # Gate validation is complete.  This is the first access to confirm costs.
    require(confirm_path.is_file(), f"confirm costs are missing: {confirm_path}")
    actual = sha256_file(confirm_path)
    require(actual == declared, "confirm-cost SHA256 mismatch after gate open")
    return confirm_path, gate, sha256_file(gate_path)


def task_index(model_index: dict[str, Any], catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(model_index.get("schema") == "operator-oracle-quality135-model-index-v1",
            "unexpected model index schema")
    require(model_index.get("status") == "locked_development_only",
            "model index is not development-locked")
    require(model_index.get("confirm_used_for_fit_or_selection") is False,
            "model index used confirm labels")
    require(model_index.get("NNCP") == "OFF", "NNCP is not OFF")
    require(model_index.get("joint_cross_product_measured") is False,
            "model index claims an unmeasured Cartesian product")
    require("PPMD8" in model_index.get("excluded_algorithm_methods", []),
            "model index does not exclude PPMD8")
    expected_catalog = {
        "algorithm": set(catalog["algorithm"]),
        "operator_zstd3": set(catalog["operator_zstd3"]),
        "observed_pipeline": set(catalog["observed_pipeline"]),
    }
    require(len(expected_catalog["algorithm"]) == 84,
            "formal catalog must contain 84 algorithms")
    require(len(expected_catalog["operator_zstd3"]) == 97,
            "formal catalog must contain 97 operators")
    require(len(expected_catalog["observed_pipeline"]) == 181,
            "formal catalog must contain 181 observed candidates")
    require("PPMD8" not in expected_catalog["algorithm"], "PPMD8 is selectable")
    require(catalog.get("excluded_algorithm_methods") == ["PPMD8"],
            "formal catalog exclusions differ from frozen PPMD8-only algorithm exclusion")
    require(catalog.get("excluded_action_ids") == [106, 107, 108, 109, 110],
            "formal catalog action exclusions differ from A106--A110")
    require(catalog.get("NNCP") == "OFF", "formal catalog does not disable NNCP")
    require(catalog.get("oracle_reference_algorithm_count") == 85 and
            catalog.get("oracle_reference_operator_count") == 97,
            "formal catalog does not retain the full 85/97 oracle reference")
    require(catalog.get("joint_cross_product_measured") is False,
            "catalog claims an unmeasured Cartesian product")
    result: dict[str, dict[str, Any]] = {}
    for item in model_index.get("tasks", []):
        task_id = item.get("task_id")
        require(task_id in TASKS and task_id not in result,
                f"invalid/duplicate model task {task_id}")
        domain, _ = TASKS[task_id]
        require(item.get("domain") == domain, f"model task {task_id} domain mismatch")
        require(item.get("objective") == TASKS[task_id][1].removesuffix("_cost"),
                f"model task {task_id} objective mismatch")
        declared = item.get("declared_candidate_ids")
        require(isinstance(declared, list) and set(declared) == expected_catalog[domain],
                f"model task {task_id} declared pool differs from catalog")
        pools = item.get("dtype_candidate_pools")
        require(isinstance(pools, dict) and pools, f"model task {task_id} lacks dtype pools")
        for dtype, pool in pools.items():
            require(dtype in {"float32", "float64", "int64"}, "unknown dtype pool")
            require(isinstance(pool, list) and len(pool) == len(set(pool)) and pool,
                    f"invalid candidate pool for {task_id}/{dtype}")
            require(set(pool) <= expected_catalog[domain],
                    f"candidate pool escapes catalog for {task_id}/{dtype}")
        result[task_id] = item
    require(set(result) == set(TASKS), "model index must contain exactly six tasks")
    return result


def load_selectors(path: Path, model_sha: str, tasks: dict[str, dict[str, Any]],
                   oof: bool) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    expected_schema = (
        "operator-oracle-quality135-oof-row-v1" if oof
        else "operator-oracle-quality135-prediction-v1"
    )
    for number, row in jsonl(path):
        require(row.get("schema") == expected_schema, f"{path}:{number} schema mismatch")
        dataset = row.get("dataset_id")
        column = row.get("column_id", row.get("sample_id"))
        task_id = row.get("task_id")
        require(all(isinstance(value, str) and value for value in [dataset, column, task_id]),
                f"{path}:{number} lacks identity")
        require(task_id in tasks, f"{path}:{number} unknown task")
        if oof:
            require(row.get("query_costs_used_for_selection") is False,
                    f"{path}:{number} OOF selection used query costs")
            expected_policy = tasks[task_id]["selected_policy"]["id"]
            require(row.get("policy_id") == expected_policy,
                    f"{path}:{number} is not chosen-config OOF")
            require(isinstance(row.get("selected_supported_on_validation_row"), bool) and
                    (finite_positive(row.get("loss_log_regret")) or
                     row.get("loss_log_regret") == 0),
                    f"{path}:{number} invalid OOF support/loss fields")
        else:
            require(row.get("query_costs_read") is False,
                    f"{path}:{number} prediction read query costs")
            require(row.get("model_index_sha256") == model_sha,
                    f"{path}:{number} model-index binding mismatch")
            require(row.get("selection_cpu_repeats") == 5 and
                    row.get("selection_warm_repeats") == 1 and
                    row.get("timing_excludes_model_setup") is True and
                    row.get("timing_excludes_feature_extraction") is True,
                    f"{path}:{number} prediction timing boundary mismatch")
            samples = row.get("selection_cpu_ns_samples")
            require(isinstance(samples, list) and len(samples) == 5 and
                    all(isinstance(value, int) and value >= 0 for value in samples) and
                    row.get("selection_cpu_ns_median") == sorted(samples)[2],
                    f"{path}:{number} invalid five-repeat CPU samples")
        pool = row.get("candidate_pool")
        ranking = row.get("ranking_candidate_ids")
        scores = row.get("ranking_predicted_log_costs")
        require(isinstance(pool, list) and pool and len(pool) == len(set(pool)),
                f"{path}:{number} invalid candidate_pool")
        require(isinstance(ranking, list) and isinstance(scores, list) and
                len(ranking) == len(scores) and set(ranking) == set(pool) and
                len(ranking) == len(set(ranking)),
                f"{path}:{number} ranking is not an exact ordering of candidate_pool")
        expected_policy = tasks[task_id]["selected_policy"]
        policy_kind = expected_policy.get("kind")
        require(policy_kind in {"model", "constant", "fixed"},
                f"{path}:{number} has an unknown locked policy kind")
        if oof:
            require(row.get("policy_id") == expected_policy.get("id"),
                    f"{path}:{number} selected policy ID mismatch")
        else:
            require(row.get("selected_policy") == expected_policy,
                    f"{path}:{number} selected policy metadata mismatch")
        if policy_kind == "model":
            require(all(isinstance(value, (int, float)) and
                        not isinstance(value, bool) and math.isfinite(float(value))
                        for value in scores),
                    f"{path}:{number} model ranking lacks finite learned scores")
        else:
            require(all(value is None for value in scores),
                    f"{path}:{number} non-model ranking claims learned scores")
        selected = row.get("selected_candidate_id")
        require(selected is None or (isinstance(selected, str) and
                                     ranking and ranking[0] == selected),
                f"{path}:{number} selected candidate is not ranking top1")
        identity = key(dataset, column, task_id)
        require(identity not in result, f"duplicate selector {identity}")
        result[identity] = row
    require(result, f"empty selector file {path}")
    return result


def accepted(candidate: dict[str, Any], cost_field: str, timed_task: bool) -> bool:
    if candidate.get("eligible") is not True or not finite_positive(candidate.get(cost_field)):
        return False
    if timed_task and candidate.get("timing_contract") != COMMON_TIME_CONTRACT:
        return False
    return True


def byte_value(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("compressed_bytes")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


@dataclass
class Ratio:
    count: int = 0
    numerator_sum: float = 0.0
    denominator_sum: float = 0.0
    log_sum: float = 0.0
    datasets: dict[str, list[float]] = field(default_factory=dict)

    def add(self, numerator: Any, denominator: Any, dataset: str) -> None:
        if not finite_positive(numerator) or not finite_positive(denominator):
            return
        numerator = float(numerator)
        denominator = float(denominator)
        self.count += 1
        self.numerator_sum += numerator
        self.denominator_sum += denominator
        value = math.log(numerator / denominator)
        self.log_sum += value
        state = self.datasets.setdefault(dataset, [0.0, 0.0])
        state[0] += value
        state[1] += 1.0

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {
                "columns": 0, "datasets": 0, "column_equal_geomean_ratio": None,
                "dataset_equal_geomean_ratio": None, "total_ratio": None,
                "numerator_total": None, "denominator_total": None,
            }
        dataset_log = sum(total / count for total, count in self.datasets.values())
        return {
            "columns": self.count,
            "datasets": len(self.datasets),
            "column_equal_geomean_ratio": math.exp(self.log_sum / self.count),
            "dataset_equal_geomean_ratio": math.exp(dataset_log / len(self.datasets)),
            "total_ratio": self.numerator_sum / self.denominator_sum,
            "numerator_total": self.numerator_sum,
            "denominator_total": self.denominator_sum,
        }


@dataclass
class TaskAggregate:
    split: str
    task_id: str
    rows: int = 0
    datasets: set[str] = field(default_factory=set)
    available: int = 0
    failed: int = 0
    selected_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    failure_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    fallback_counts: collections.Counter[str] = field(default_factory=collections.Counter)
    ranking_unavailable: int = 0
    original_numeric_bytes_total: int = 0
    selected_frame_bytes_total: int = 0
    hits: collections.Counter[str] = field(default_factory=collections.Counter)
    hit_denominators: collections.Counter[int] = field(
        default_factory=collections.Counter)
    selection_cpu_ns: list[int] = field(default_factory=list)
    ratios: dict[str, Ratio] = field(default_factory=lambda: collections.defaultdict(Ratio))
    timing: dict[str, dict[str, Ratio]] = field(default_factory=lambda: {
        baseline: {"compression": Ratio(), "decompression": Ratio()}
        for baseline, _ in FIXED_BASELINES
    })
    timing_joint_comparable: collections.Counter[str] = field(
        default_factory=collections.Counter)

    def summary(self) -> dict[str, Any]:
        domain, objective = TASKS[self.task_id]
        timing_report: dict[str, Any]
        if domain != "algorithm":
            timing_report = {
                "status": "unavailable_for_operator_and_observed_domains",
                "timing_comparable_columns": 0,
                "full_scope_ratios": None,
                "common_contract_subset": None,
            }
        else:
            timing_report = {"status": "common_contract_intersection_only", "baselines": {}}
            for baseline, metrics in self.timing.items():
                compression = metrics["compression"].summary()
                decompression = metrics["decompression"].summary()
                comparable = self.timing_joint_comparable[baseline]
                timing_report["baselines"][baseline] = {
                    "timing_contract": COMMON_TIME_CONTRACT,
                    "timing_comparable_columns": comparable,
                    "full_scope_compression_ratio":
                        compression["total_ratio"] if compression["columns"] == self.rows else None,
                    "full_scope_decompression_ratio":
                        decompression["total_ratio"] if decompression["columns"] == self.rows else None,
                    "common_contract_subset": {
                        "compression": compression, "decompression": decompression,
                    },
                }
        overhead = {
            "boundary": "selection only; five CPU repeats after one warm repeat",
            "columns": len(self.selection_cpu_ns),
            "median_of_row_medians_cpu_ns": (
                statistics.median(self.selection_cpu_ns) if self.selection_cpu_ns else None
            ),
            "model_setup_cpu_ns": None,
            "model_setup_status": "reported separately when prediction receipt provides it",
            "included_in_codec_time": False,
        }
        ratio_summaries = {
            name: ratio.summary() for name, ratio in sorted(self.ratios.items())
        }
        hit_names = [
            f"{kind}_top{topk}"
            for kind in ["allowed_objective", "allowed_byte", "full_frozen_byte"]
            for topk in [1, 3, 5]
        ]
        hit_summary = {
            name: {
                "count": self.hits[name],
                "denominator": self.hit_denominators[topk],
                "rate": (self.hits[name] / self.hit_denominators[topk]
                         if self.hit_denominators[topk] else None),
            }
            for name in hit_names
            for topk in [int(name.rsplit("top", 1)[1])]
        }
        def extra(name: str) -> float | None:
            value = ratio_summaries[name]
            if value["numerator_total"] is None:
                return None
            return value["numerator_total"] - value["denominator_total"]
        return {
            "split": self.split,
            "task_id": self.task_id,
            "domain": domain,
            "objective": objective.removesuffix("_cost"),
            "coverage": {
                "rows": self.rows, "datasets": len(self.datasets),
                "selected_available": self.available,
                "selected_unavailable_failures": self.failed,
                "coverage_fraction": self.available / self.rows if self.rows else None,
                "failure_reasons": dict(sorted(self.failure_counts.items())),
            },
            "selected_candidate_top1_counts": dict(sorted(self.selected_counts.items())),
            "byte_totals": {
                "scope_original_numeric_bytes": self.original_numeric_bytes_total,
                "selected_frame_bytes_on_available_rows": self.selected_frame_bytes_total,
            },
            "byte_gap_decomposition_totals": {
                "full_to_allowed_extra_bytes":
                    extra("allowed_oracle_vs_full_byte_oracle"),
                "allowed_to_dtype_pool_extra_bytes":
                    extra("dtype_pool_oracle_vs_allowed_oracle_bytes"),
                "dtype_pool_to_selected_extra_bytes_on_available_rows":
                    extra("selected_vs_dtype_pool_oracle_bytes"),
            },
            "operator_raw_fallback_class_counts": dict(sorted(self.fallback_counts.items())),
            "ranking_unavailable_candidate_occurrences": self.ranking_unavailable,
            "exact_hits": {
                "topk_denominators": {
                    "top1_all_rows_including_failed": self.hit_denominators[1],
                    "top3_model_policy_rows": self.hit_denominators[3],
                    "top5_model_policy_rows": self.hit_denominators[5],
                },
                "metrics": hit_summary,
            },
            "ratios": ratio_summaries,
            "timing": timing_report,
            "prediction_overhead": overhead,
        }


def full_oracle_bytes(row: dict[str, Any], domain: str) -> int:
    frozen = row.get("full_frozen_oracle")
    require(isinstance(frozen, dict), "cost row lacks full_frozen_oracle")
    field = {
        "algorithm": "algorithm_compressed_bytes",
        "operator_zstd3": "operator_compressed_bytes",
        "observed_pipeline": "observed_pipeline_compressed_bytes",
    }[domain]
    value = frozen.get(field)
    require(isinstance(value, int) and value > 0, f"invalid {field}")
    return value


def fallback_class(domain: str, candidate_id: str | None,
                   candidate: dict[str, Any] | None) -> str | None:
    is_operator = domain == "operator_zstd3" or (
        domain == "observed_pipeline" and isinstance(candidate_id, str)
        and candidate_id.startswith("operator_zstd3:")
    )
    if not is_operator or candidate is None:
        return None
    value = candidate.get("raw_fallback_block_count")
    if value is None:
        return "null"
    require(isinstance(value, int) and value >= 0, "invalid raw_fallback_block_count")
    return "zero" if value == 0 else "positive"


def evaluate_split(cost_path: Path, split: str,
                   selectors: dict[tuple[str, str, str], dict[str, Any]],
                   tasks: dict[str, dict[str, Any]], catalog: dict[str, Any],
                   target, formal: bool) -> tuple[dict[str, TaskAggregate], int]:
    aggregates = {task_id: TaskAggregate(split, task_id) for task_id in TASKS}
    selector_seen: set[tuple[str, str, str]] = set()
    row_count = 0
    catalog_sets = {domain: set(catalog[domain]) for domain, _ in TASKS.values()}
    for number, row in jsonl(cost_path):
        require(row.get("split") == split, f"{cost_path}:{number} split mismatch")
        dataset = row.get("dataset_id")
        column = row.get("column_id")
        dtype = row.get("dtype")
        original = row.get("original_numeric_bytes")
        require(isinstance(dataset, str) and isinstance(column, str) and
                dtype in {"float32", "float64", "int64"} and
                isinstance(original, int) and original > 0,
                f"{cost_path}:{number} invalid identity/shape")
        domains = row.get("domains")
        require(isinstance(domains, dict), f"{cost_path}:{number} lacks domains")
        row_count += 1
        for task_id, (domain, cost_field) in TASKS.items():
            identity = key(dataset, column, task_id)
            require(identity in selectors, f"missing {split} selector {identity}")
            require(identity not in selector_seen, f"duplicate {split} cost identity {identity}")
            selector_seen.add(identity)
            selector = selectors[identity]
            require(selector.get("dtype") == dtype, f"selector dtype mismatch {identity}")
            candidates_raw = domains.get(domain)
            require(isinstance(candidates_raw, list) and candidates_raw,
                    f"cost row lacks domain {domain}")
            lookup: dict[str, dict[str, Any]] = {}
            for candidate in candidates_raw:
                candidate_id = candidate.get("candidate_id")
                require(isinstance(candidate_id, str) and candidate_id and
                        candidate_id not in lookup and candidate_id in catalog_sets[domain],
                        f"invalid candidate in {identity}")
                lookup[candidate_id] = candidate
            require(set(lookup) == catalog_sets[domain],
                    f"cost candidates differ from catalog for {identity}")
            timed_task = domain == "algorithm" and cost_field != "space_cost"
            accepted_lookup = {
                candidate_id: candidate for candidate_id, candidate in lookup.items()
                if accepted(candidate, cost_field, timed_task)
            }
            require(accepted_lookup, f"no accepted candidate for {identity}")
            allowed_oracle_id, allowed_oracle = min(
                accepted_lookup.items(), key=lambda item: (float(item[1][cost_field]), item[0])
            )
            pool = selector["candidate_pool"]
            require(set(pool) <= catalog_sets[domain], f"selector pool escapes catalog {identity}")
            if timed_task:
                require(all(lookup[candidate_id].get("timing_contract") ==
                            COMMON_TIME_CONTRACT for candidate_id in pool),
                        f"timed-task pool is not exact-contract restricted {identity}")
            if split == "confirm":
                expected_pool = tasks[task_id]["dtype_candidate_pools"].get(dtype)
                require(pool == expected_pool, f"confirm pool differs from locked model {identity}")
            ranking_ids = selector["ranking_candidate_ids"]
            accepted_ranking = [candidate_id for candidate_id in ranking_ids
                                if candidate_id in accepted_lookup]
            ranking_unavailable = len(ranking_ids) - len(accepted_ranking)
            pool_oracle_items = [(candidate_id, accepted_lookup[candidate_id])
                                 for candidate_id in pool if candidate_id in accepted_lookup]
            require(pool_oracle_items, f"dtype pool has no accepted candidate {identity}")
            pool_oracle_id, pool_oracle = min(
                pool_oracle_items, key=lambda item: (float(item[1][cost_field]), item[0])
            )
            selected_id = selector.get("selected_candidate_id")
            selected = accepted_lookup.get(selected_id)
            failure = None
            if selected_id is None:
                failure = "null_selection"
            elif selected_id not in lookup:
                failure = "selection_absent_from_domain"
            elif selected is None:
                failure = "selection_unavailable_or_nonfinite"
            selected_bytes = byte_value(selected) if selected is not None else None
            full_bytes = full_oracle_bytes(row, domain)
            allowed_byte_values = [byte_value(candidate) for candidate in lookup.values()
                                   if candidate.get("eligible") is True]
            allowed_byte_values = [value for value in allowed_byte_values if value is not None]
            require(allowed_byte_values, f"no allowed byte oracle {identity}")
            allowed_bytes = min(allowed_byte_values)
            pool_byte_values = [byte_value(lookup[candidate_id]) for candidate_id in pool
                                if lookup[candidate_id].get("eligible") is True]
            pool_byte_values = [value for value in pool_byte_values if value is not None]
            require(pool_byte_values, f"no pool byte oracle {identity}")
            pool_bytes = min(pool_byte_values)
            baseline_output: dict[str, Any] = {}
            fixed = row.get("fixed_baselines")
            require(isinstance(fixed, dict), f"{identity} lacks fixed baselines")
            aggregate = aggregates[task_id]
            aggregate.rows += 1
            aggregate.datasets.add(dataset)
            aggregate.ranking_unavailable += ranking_unavailable
            aggregate.original_numeric_bytes_total += original
            if failure is None:
                aggregate.available += 1
                aggregate.selected_counts[selected_id] += 1
                require(selected_bytes is not None,
                        f"available selection lacks complete-frame bytes {identity}")
                aggregate.selected_frame_bytes_total += selected_bytes
            else:
                aggregate.failed += 1
                aggregate.failure_counts[failure] += 1
            fallback = fallback_class(domain, selected_id, selected)
            if fallback is not None:
                aggregate.fallback_counts[fallback] += 1
            selected_cost = float(selected[cost_field]) if selected is not None else None
            allowed_cost = float(allowed_oracle[cost_field])
            pool_cost = float(pool_oracle[cost_field])
            if split == "development":
                expected_loss = (
                    math.log(selected_cost / allowed_cost)
                    if selected_cost is not None
                    else math.log(UNSUPPORTED_PENALTY_RATIO)
                )
                require(selector.get("selected_supported_on_validation_row") is
                        (failure is None),
                        f"OOF support flag differs from frozen costs {identity}")
                require(math.isclose(float(selector["loss_log_regret"]), expected_loss,
                                     rel_tol=1e-12, abs_tol=1e-12),
                        f"OOF log-regret differs from frozen costs {identity}")
                expected_penalty = None if failure is None else UNSUPPORTED_PENALTY_RATIO
                require(selector.get("unsupported_penalty") == expected_penalty,
                        f"OOF unsupported penalty differs from protocol {identity}")
            aggregate.ratios["allowed_oracle_vs_full_byte_oracle"].add(
                allowed_bytes, full_bytes, dataset)
            aggregate.ratios["dtype_pool_oracle_vs_allowed_oracle_bytes"].add(
                pool_bytes, allowed_bytes, dataset)
            aggregate.ratios["selected_vs_raw_bytes"].add(selected_bytes, original, dataset)
            aggregate.ratios["selected_vs_full_frozen_oracle_bytes"].add(
                selected_bytes, full_bytes, dataset)
            aggregate.ratios["selected_vs_allowed_oracle_bytes"].add(
                selected_bytes, allowed_bytes, dataset)
            aggregate.ratios["selected_vs_dtype_pool_oracle_bytes"].add(
                selected_bytes, pool_bytes, dataset)
            aggregate.ratios["selected_vs_allowed_objective_oracle"].add(
                selected_cost, allowed_cost, dataset)
            aggregate.ratios["selected_vs_dtype_pool_objective_oracle"].add(
                selected_cost, pool_cost, dataset)

            # Top-k is defined on the selector's original ranking positions.
            # Unavailable candidates retain their slots: filtering them before
            # slicing would silently promote later candidates and overstate hits.
            policy_kind = tasks[task_id]["selected_policy"]["kind"]
            scored_topk = [1] + ([3, 5] if policy_kind == "model" else [])
            for topk in scored_topk:
                aggregate.hit_denominators[topk] += 1
                # Top-1 is the policy decision itself. A null decision remains
                # a miss even though constant/fixed serialization retains the
                # pool in lexical order after its (possibly absent) selection.
                prefix = (
                    [selected_id] if topk == 1 and selected_id is not None
                    else ([] if topk == 1 else ranking_ids[:topk])
                )
                if any(candidate_id in accepted_lookup and
                       float(accepted_lookup[candidate_id][cost_field]) == allowed_cost
                       for candidate_id in prefix):
                    aggregate.hits[f"allowed_objective_top{topk}"] += 1
                if any(candidate_id in lookup and
                       lookup[candidate_id].get("eligible") is True and
                       byte_value(lookup[candidate_id]) == allowed_bytes
                       for candidate_id in prefix):
                    aggregate.hits[f"allowed_byte_top{topk}"] += 1
                if any(candidate_id in lookup and
                       lookup[candidate_id].get("eligible") is True and
                       byte_value(lookup[candidate_id]) == full_bytes
                       for candidate_id in prefix):
                    aggregate.hits[f"full_frozen_byte_top{topk}"] += 1

            if not selector.get("query_costs_read", selector.get(
                    "query_costs_used_for_selection", False)):
                cpu = selector.get("selection_cpu_ns_median")
                if isinstance(cpu, int) and cpu >= 0:
                    aggregate.selection_cpu_ns.append(cpu)
            for baseline_label, source_candidate_id in FIXED_BASELINES:
                baseline = fixed.get(source_candidate_id)
                require(isinstance(baseline, dict),
                        f"{identity} lacks fixed {source_candidate_id}")
                baseline_bytes = byte_value(baseline) if baseline.get("eligible") is True else None
                baseline_output[baseline_label] = {
                    "source_candidate_id": source_candidate_id,
                    "compressed_bytes": baseline_bytes,
                    "compression_ns": baseline.get("compression_ns"),
                    "decompression_ns": baseline.get("decompression_ns"),
                    "timing_contract": baseline.get("timing_contract"),
                    "selected_to_fixed_byte_ratio": (
                        selected_bytes / baseline_bytes
                        if selected_bytes is not None and baseline_bytes is not None else None
                    ),
                    "fixed_to_full_oracle_byte_ratio": (
                        baseline_bytes / full_bytes if baseline_bytes is not None else None
                    ),
                }
                aggregate.ratios[f"selected_vs_fixed_{baseline_label}_bytes"].add(
                    selected_bytes, baseline_bytes, dataset)
                aggregate.ratios[f"fixed_{baseline_label}_vs_full_oracle_bytes"].add(
                    baseline_bytes, full_bytes, dataset)
                if domain == "algorithm" and selected is not None:
                    if (baseline.get("eligible") is True and
                            selected.get("timing_contract") == COMMON_TIME_CONTRACT and
                            baseline.get("timing_contract") == COMMON_TIME_CONTRACT):
                        aggregate.timing[baseline_label]["compression"].add(
                            selected.get("compression_ns"), baseline.get("compression_ns"), dataset)
                        aggregate.timing[baseline_label]["decompression"].add(
                            selected.get("decompression_ns"), baseline.get("decompression_ns"), dataset)
                        if all(finite_positive(value) for value in [
                                selected.get("compression_ns"),
                                baseline.get("compression_ns"),
                                selected.get("decompression_ns"),
                                baseline.get("decompression_ns")]):
                            aggregate.timing_joint_comparable[baseline_label] += 1

            result = {
                "schema": "operator-oracle-quality135-result-row-v1",
                "split": split,
                "evaluation_scope": (
                    "chosen_config_development_oof_not_external_generalization"
                    if split == "development" else "single_locked_confirm_final"
                ),
                "dataset_id": dataset,
                "column_id": column,
                "dtype": dtype,
                "original_numeric_bytes": original,
                "task_id": task_id,
                "domain": domain,
                "objective": cost_field.removesuffix("_cost"),
                "selected_policy": selector.get("selected_policy", selector.get("policy_id")),
                "selected_candidate_id": selected_id,
                "selection_available": failure is None,
                "selection_failure": failure,
                "candidate_pool_count": len(pool),
                "ranking_candidate_count": len(ranking_ids),
                "ranking_unavailable_candidate_count": ranking_unavailable,
                "selected_objective_cost": selected_cost,
                "selected_compressed_bytes": selected_bytes,
                "allowed_oracle_candidate_id": allowed_oracle_id,
                "allowed_oracle_objective_cost": allowed_cost,
                "allowed_oracle_compressed_bytes": allowed_bytes,
                "dtype_pool_oracle_candidate_id": pool_oracle_id,
                "dtype_pool_oracle_objective_cost": pool_cost,
                "dtype_pool_oracle_compressed_bytes": pool_bytes,
                "full_frozen_oracle_compressed_bytes": full_bytes,
                "byte_gap_decomposition": {
                    "full_to_allowed_extra_bytes": allowed_bytes - full_bytes,
                    "allowed_to_dtype_pool_extra_bytes": pool_bytes - allowed_bytes,
                    "dtype_pool_to_selected_extra_bytes": (
                        selected_bytes - pool_bytes if selected_bytes is not None else None
                    ),
                },
                "exact_hit": {
                    "selected_allowed_objective_tie":
                        selected_cost == allowed_cost if selected_cost is not None else False,
                    "selected_allowed_byte_tie": selected_bytes == allowed_bytes,
                    "selected_full_frozen_byte_tie": selected_bytes == full_bytes,
                },
                "fixed_baselines": baseline_output,
                "operator_raw_fallback_class": fallback,
                "selection_cpu_ns_median": selector.get("selection_cpu_ns_median"),
                "query_costs_used_for_selection": False,
            }
            target.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
        # Drop the full cost-row DOM before reading the next line.
        del row
    expected = 3433 if split == "development" else 86
    if formal:
        require(row_count == expected, f"{split} cost row count {row_count} != {expected}")
    require(len(selector_seen) == row_count * len(TASKS),
            f"{split} selector coverage mismatch")
    if split == "development":
        require(len(selectors) == len(selector_seen), "OOF contains non-development identities")
    return aggregates, row_count


def oof_dataset_equal_loss(
        selectors: dict[tuple[str, str, str], dict[str, Any]], task_id: str) -> float:
    datasets: dict[str, list[float]] = collections.defaultdict(list)
    for (dataset, _column, selected_task), row in selectors.items():
        if selected_task == task_id:
            datasets[dataset].append(float(row["loss_log_regret"]))
    require(datasets, f"chosen OOF lacks task {task_id}")
    return sum(sum(values) / len(values) for values in datasets.values()) / len(datasets)


def cv_baselines(cv: dict[str, Any], tasks: dict[str, dict[str, Any]],
                 selectors: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    require(cv.get("schema") == "operator-oracle-quality135-development-cv-v1",
            "unexpected CV summary schema")
    require(cv.get("scope") == "development_only_model_selection_not_external_evaluation" and
            cv.get("confirm_used_for_tuning") is False,
            "CV summary is not development-only")
    result: dict[str, Any] = {}
    by_task = {item["task_id"]: item for item in cv.get("tasks", [])}
    require(set(by_task) == set(TASKS), "CV summary does not contain six tasks")
    for task_id, item in by_task.items():
        selected = item["selected_policy"]["id"]
        require(selected == tasks[task_id]["selected_policy"]["id"],
                f"CV/model selected policy mismatch for {task_id}")
        policies = {policy["policy"]["id"]: policy for policy in item["policies"]}
        selected_row = policies[selected]
        recomputed_loss = oof_dataset_equal_loss(selectors, task_id)
        require(math.isclose(float(selected_row["dataset_equal_mean_log_regret"]),
                             recomputed_loss, rel_tol=1e-12, abs_tol=1e-12),
                f"CV selected loss differs from chosen OOF rows for {task_id}")
        baselines: dict[str, Any] = {}
        for policy_id in ["constant_best_per_dtype", "fixed_LZ4", "fixed_Zstd3"]:
            if policy_id not in policies:
                continue
            baseline = policies[policy_id]
            baselines[policy_id] = {
                "dataset_equal_mean_log_regret": baseline["dataset_equal_mean_log_regret"],
                "dataset_equal_geomean_regret": math.exp(
                    baseline["dataset_equal_mean_log_regret"]),
                "unsupported_validation_rows": baseline["unsupported_validation_rows"],
                "selected_minus_baseline_mean_log_regret":
                    selected_row["dataset_equal_mean_log_regret"] -
                    baseline["dataset_equal_mean_log_regret"],
            }
        result[task_id] = {
            "scope": "development OOF; each fold uses its own training-derived dtype pool, which differs from the final confirm pool",
            "selected_policy_id": selected,
            "selected_dataset_equal_mean_log_regret":
                selected_row["dataset_equal_mean_log_regret"],
            "selected_dataset_equal_geomean_regret": math.exp(
                selected_row["dataset_equal_mean_log_regret"]),
            "baselines": baselines,
        }
    return result


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Operator oracle quality report", "",
        f"Status: `{report['status']}`", "",
        "Development quality uses the selected configuration's source-group OOF ranks. "
        "It is model-selection evidence, not independent external generalization.", "",
    ]
    if report["confirm_evaluated"]:
        lines += [
            "Confirm is the single final evaluation after LOCK and CONFIRM_GATE hash validation.", "",
        ]
    else:
        lines += ["Confirm costs were not opened in this development-only report.", ""]
    lines += [
        "Operator and observed-pipeline time is unavailable. Algorithm timing is reported "
        "only on the named common-contract intersection; null full-scope ratios mean the "
        "intersection did not cover every column.", "",
        "Prediction CPU measures selector execution only (five repeats after one warm call). "
        "Feature extraction and model setup remain separate, and these values are not codec "
        "end-to-end measurements.", "",
        "Top-1 uses the actual policy decision and includes failed selections in its denominator. "
        "Top-3/5 are reported only for learned model rankings; constant/fixed policies retain "
        "their serialized pool order for audit but receive null Top-3/5 rates.", "",
    ]
    for split, tasks in report["quality"].items():
        lines += [f"## {split}", "", "| Task | Coverage | Selected/full byte total | Selected/allowed objective geomean |", "|---|---:|---:|---:|"]
        for task_id, item in tasks.items():
            coverage = item["coverage"]
            byte_ratio = item["ratios"]["selected_vs_full_frozen_oracle_bytes"]["total_ratio"]
            objective = item["ratios"]["selected_vs_allowed_objective_oracle"]["column_equal_geomean_ratio"]
            lines.append(
                f"| {task_id} | {coverage['selected_available']}/{coverage['rows']} | "
                f"{byte_ratio if byte_ratio is not None else 'null'} | "
                f"{objective if objective is not None else 'null'} |"
            )
        lines.append("")
    lines += [
        "Candidate IDs in `observed_pipeline` are a measured namespaced union. The report "
        "does not compose independently optimal operators and algorithms.", "",
    ]
    return "\n".join(lines)


def run(root: Path, development_only: bool = False) -> dict[str, Any]:
    root = root.resolve(strict=True)
    output = root / "report"
    require(not output.exists(), f"report output already exists: {output}")
    output.mkdir()
    try:
        lock, paths, hashes = load_lock(root, development_only)
        formal = lock.get("require_frozen_current135", True) is True
        catalog = read_json(paths["candidate_catalog"],
                            "operator-oracle-quality135-catalog-v1")
        model_index = read_json(paths["model_index"])
        tasks = task_index(model_index, catalog) if formal else task_index_fixture(model_index, catalog)
        model_sha = hashes["model_index"]
        oof = load_selectors(paths["chosen_oof"], model_sha, tasks, True)
        predictions = None
        gate = None
        gate_sha = None
        confirm_path = None
        if not development_only:
            # Load predictions only after their LOCK hash has been checked.
            predictions = load_selectors(paths["all_predictions"], model_sha, tasks, False)
            confirm_path, gate, gate_sha = unlock_confirm(root, lock, hashes)

        results_path = output / "RESULTS.jsonl"
        all_aggregates: dict[str, dict[str, TaskAggregate]] = {}
        split_counts: dict[str, int] = {}
        with results_path.open("x") as target:
            aggregates, count = evaluate_split(
                paths["development_costs"], "development", oof, tasks,
                catalog, target, formal)
            all_aggregates["development"] = aggregates
            split_counts["development"] = count
            if not development_only:
                assert predictions is not None and confirm_path is not None
                aggregates, count = evaluate_split(
                    confirm_path, "confirm", predictions, tasks,
                    catalog, target, formal)
                all_aggregates["confirm"] = aggregates
                split_counts["confirm"] = count
            target.flush()
            os.fsync(target.fileno())

        cv = read_json(paths["cv_summary"])
        feature_receipt = read_json(paths["features_receipt"])
        require(feature_receipt.get("schema") ==
                "operator-oracle-quality135-features-receipt-v1" and
                feature_receipt.get("oracle_labels_read") is False,
                "unsafe feature receipt")
        prediction_receipt = None
        if not development_only:
            prediction_receipt = read_json(paths["prediction_receipt"])
            require(prediction_receipt.get("schema") ==
                    "operator-oracle-quality135-prediction-receipt-v1",
                    "unexpected prediction receipt schema")
            require((prediction_receipt.get("output") or {}).get("sha256") ==
                    hashes["all_predictions"] and
                    ((prediction_receipt.get("inputs") or {}).get(
                        "model_index") or {}).get("sha256") == hashes["model_index"] and
                    prediction_receipt.get("query_costs_read") is False,
                    "prediction receipt hash binding mismatch")
        quality = {
            split: {task_id: aggregate.summary()
                    for task_id, aggregate in aggregates.items()}
            for split, aggregates in all_aggregates.items()
        }
        failures = sum(
            task["coverage"]["selected_unavailable_failures"]
            for split in quality.values() for task in split.values()
        )
        report = {
            "schema": "operator-oracle-quality135-report-v1",
            "status": "complete" if failures == 0 else "complete_with_selection_failures",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": {
                "algorithm_candidates": len(catalog["algorithm"]),
                "operator_zstd3_candidates": len(catalog["operator_zstd3"]),
                "observed_pipeline_candidates": len(catalog["observed_pipeline"]),
                "full_frozen_oracle_reference": {
                    "algorithm_candidates": catalog.get(
                        "oracle_reference_algorithm_count"),
                    "operator_zstd3_candidates": catalog.get(
                        "oracle_reference_operator_count"),
                    "observed_pipeline_measured_slots": (
                        catalog.get("oracle_reference_algorithm_count") +
                        catalog.get("oracle_reference_operator_count")
                        if isinstance(catalog.get("oracle_reference_algorithm_count"), int) and
                        isinstance(catalog.get("oracle_reference_operator_count"), int)
                        else None
                    ),
                    "role": "hindsight space-only gap reference",
                },
                "PPMD8_selectable": False,
                "NNCP": "OFF",
                "joint_cross_product_measured": False,
                "runtime_execution": "offline historical matrix only; no 181-candidate runtime execution",
            },
            "confirm_evaluated": not development_only,
            "confirm_used_for_fit_or_selection": False,
            "split_column_counts": split_counts,
            "quality": quality,
            "development_cv_baselines": cv_baselines(cv, tasks, oof),
            "feature_extraction_timing": {
                "timing_contract": feature_receipt.get("timing_contract"),
                "timings": feature_receipt.get("timings"),
                "separate_from_prediction_and_codec_timing": True,
            },
            "prediction_phase_timing": (
                {
                    "timing": prediction_receipt.get("timing"),
                    "model_setup_is_separate": True,
                    "feature_load_is_separate": True,
                    "included_in_codec_time": False,
                }
                if prediction_receipt is not None else {
                    "status": "not_run_in_development_only_report"
                }
            ),
            "timing_limits": {
                "codec_timing_is_unified_new_e2e": False,
                "selection_cpu_included_in_historical_codec_wall_time": False,
                "operator_and_observed_time": None,
                "algorithm_common_contract": COMMON_TIME_CONTRACT,
            },
            "inputs": {
                **{name: {"path": str(path), "sha256": hashes[name]}
                   for name, path in paths.items()},
                "lock": {"path": str(root / "LOCK.json"), "sha256": hashes["lock"]},
                "confirm_gate": (
                    {"path": str(root / "CONFIRM_GATE.json"), "sha256": gate_sha}
                    if gate is not None else None
                ),
            },
        }
        write_new_json(output / "REPORT.json", report)
        write_new_text(output / "REPORT.md", markdown_report(report))
        source_path = Path(__file__).resolve()
        complete = {
            "schema": "operator-oracle-quality135-report-complete-v1",
            "status": report["status"],
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "confirm_evaluated": not development_only,
            "confirm_used_for_fit_or_selection": False,
            "outputs": {
                name: {"path": str(output / name), "sha256": sha256_file(output / name),
                       "bytes": (output / name).stat().st_size}
                for name in ["RESULTS.jsonl", "REPORT.json", "REPORT.md"]
            },
            "lock_sha256": hashes["lock"],
            "confirm_gate_sha256": gate_sha,
            "inputs": report["inputs"],
        }
        write_new_json(output / "COMPLETE.json", complete)
        return report
    except Exception as error:
        failure = output / "FAILURE.json"
        if not failure.exists():
            write_new_json(failure, {
                "schema": "operator-oracle-quality135-report-failure-v1",
                "status": "failed",
                "error": str(error),
                "source_sha256": sha256_file(Path(__file__).resolve()),
            })
        raise


def task_index_fixture(model_index: dict[str, Any], catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Same safety contracts as formal mode, without frozen candidate counts."""
    require(model_index.get("schema") == "operator-oracle-quality135-model-index-v1",
            "fixture model index schema mismatch")
    require(model_index.get("status") == "locked_development_only" and
            model_index.get("confirm_used_for_fit_or_selection") is False and
            model_index.get("NNCP") == "OFF" and
            model_index.get("joint_cross_product_measured") is False,
            "unsafe fixture model index")
    require("PPMD8" in model_index.get("excluded_algorithm_methods", []) and
            "PPMD8" not in catalog.get("algorithm", []),
            "fixture permits PPMD8 selection")
    result = {item["task_id"]: item for item in model_index["tasks"]}
    require(set(result) == set(TASKS), "fixture lacks six tasks")
    for task_id, item in result.items():
        domain, _ = TASKS[task_id]
        require(item["domain"] == domain and
                set(item["declared_candidate_ids"]) == set(catalog[domain]),
                "fixture task/catalog mismatch")
    return result


def selftest() -> None:
    root = Path(tempfile.mkdtemp(prefix="operator-oracle-quality135-report-test-"))
    try:
        for directory in ["costs", "models", "features", "predictions"]:
            (root / directory).mkdir()
        algorithms = ["LZ4", "Zstd"]
        operators = ["A001", "A002"]
        observed = [f"algorithm:{value}" for value in algorithms] + [
            f"operator_zstd3:{value}" for value in operators
        ]
        catalog = {
            "schema": "operator-oracle-quality135-catalog-v1",
            "algorithm": algorithms, "operator_zstd3": operators,
            "observed_pipeline": observed, "NNCP": "OFF",
            "joint_cross_product_measured": False,
        }
        write_new_json(root / "costs/CANDIDATES.json", catalog)
        tasks = []
        for task_id, (domain, _) in TASKS.items():
            pools = {dtype: catalog[domain] for dtype in ["float32", "float64", "int64"]}
            policy_kind = "constant" if task_id == "operator_zstd3_space" else "model"
            policy_id = (
                "constant_best_per_dtype" if policy_kind == "constant"
                else "fixture_model"
            )
            tasks.append({
                "task_id": task_id, "domain": domain,
                "objective": task_id.rsplit("_", 1)[-1],
                "selected_policy": {"id": policy_id, "kind": policy_kind},
                "dtype_candidate_pools": pools,
                "declared_candidate_ids": catalog[domain],
            })
        model_index = {
            "schema": "operator-oracle-quality135-model-index-v1",
            "status": "locked_development_only",
            "confirm_used_for_fit_or_selection": False,
            "NNCP": "OFF", "excluded_algorithm_methods": ["PPMD8"],
            "joint_cross_product_measured": False, "tasks": tasks,
        }
        write_new_json(root / "models/MODEL_INDEX.json", model_index)
        model_sha = sha256_file(root / "models/MODEL_INDEX.json")
        rows = []
        selectors = []
        for index, dtype in enumerate(["float64", "int64"]):
            dataset = f"dataset-{index}"
            column = hashlib.sha256(f"column-{index}".encode()).hexdigest()
            def candidate(candidate_id: str, compressed: int, fallback: int = 0) -> dict[str, Any]:
                return {
                    "candidate_id": candidate_id, "eligible": True,
                    "space_cost": compressed / 1000,
                    "encode_cost": (compressed + 100) / 1000,
                    "decode_cost": (compressed + 50) / 1000,
                    "balanced_cost": compressed / 1000,
                    "compressed_bytes": compressed,
                    "compression_ns": compressed + 100,
                    "decompression_ns": compressed + 50,
                    "raw_fallback_block_count": fallback,
                    "timing_contract": COMMON_TIME_CONTRACT,
                }
            alg = [candidate("LZ4", 600), candidate("Zstd", 500)]
            ops = [candidate("A001", 450), candidate("A002", 470, 1)]
            if index == 0:
                # The frozen selector must not silently replace this choice;
                # the report records an explicit unavailable-selection failure.
                ops[1]["eligible"] = False
            obs = [dict(value, candidate_id="algorithm:" + value["candidate_id"])
                   for value in alg] + [
                dict(value, candidate_id="operator_zstd3:" + value["candidate_id"])
                for value in ops
            ]
            rows.append({
                "sample_id": column, "dataset_id": dataset, "column_id": column,
                "dtype": dtype, "split": "development", "original_numeric_bytes": 1000,
                "domains": {"algorithm": alg, "operator_zstd3": ops,
                            "observed_pipeline": obs},
                "fixed_baselines": {
                    "LZ4": dict(alg[0], timing_contract=(
                        COMMON_TIME_CONTRACT if index == 0 else "mismatched-contract")),
                    "Zstd": dict(alg[1]),
                },
                "full_frozen_oracle": {
                    "algorithm_compressed_bytes": 500,
                    "operator_compressed_bytes": 450,
                    "observed_pipeline_compressed_bytes": 450,
                },
            })
            for task_id, (domain, _) in TASKS.items():
                pool = catalog[domain]
                selected_policy = next(
                    item["selected_policy"] for item in tasks
                    if item["task_id"] == task_id)
                if task_id == "operator_zstd3_space" and index == 0:
                    selected = "A002"
                elif domain == "algorithm":
                    selected = "Zstd"
                elif domain == "operator_zstd3":
                    selected = "A001"
                else:
                    selected = "operator_zstd3:A001"
                ranking = [selected] + [value for value in pool if value != selected]
                selectors.append({
                    "schema": "operator-oracle-quality135-oof-row-v1",
                    "task_id": task_id, "policy_id": selected_policy["id"], "fold": index,
                    "sample_id": column, "column_id": column, "dataset_id": dataset,
                    "dtype": dtype, "source_group": dataset, "legacy_group": dataset,
                    "selected_candidate_id": selected, "predicted_log_cost": 0.0,
                    "selected_supported_on_validation_row": not (
                        task_id == "operator_zstd3_space" and index == 0),
                    "loss_log_regret": (
                        math.log(UNSUPPORTED_PENALTY_RATIO)
                        if task_id == "operator_zstd3_space" and index == 0
                        else 0.0
                    ),
                    "unsupported_penalty": (
                        UNSUPPORTED_PENALTY_RATIO
                        if task_id == "operator_zstd3_space" and index == 0
                        else None
                    ),
                    "query_costs_used_for_selection": False,
                    "ranking_candidate_ids": ranking,
                    "ranking_predicted_log_costs": (
                        list(range(len(pool)))
                        if selected_policy["kind"] == "model"
                        else [None] * len(pool)
                    ),
                    "candidate_pool": pool,
                })
        write_new_text(root / "costs/DEVELOPMENT_COSTS.jsonl",
                       "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        write_new_text(root / "models/CHOSEN_OOF_FULL_RANKS.jsonl",
                       "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selectors))
        cv_tasks = []
        for task_id in TASKS:
            selected_policy = next(
                item["selected_policy"] for item in tasks
                if item["task_id"] == task_id)
            selected_loss = (
                math.log(UNSUPPORTED_PENALTY_RATIO) / 2
                if task_id == "operator_zstd3_space" else 0.0
            )
            cv_tasks.append({
                "task_id": task_id,
                "selected_policy": selected_policy,
                "policies": [
                    {"policy": {"id": "fixture_model"},
                     "dataset_equal_mean_log_regret": (
                         selected_loss if selected_policy["id"] == "fixture_model" else 0.1),
                     "unsupported_validation_rows": 0},
                    {"policy": {"id": "constant_best_per_dtype"},
                     "dataset_equal_mean_log_regret": (
                         selected_loss
                         if selected_policy["id"] == "constant_best_per_dtype"
                         else 0.2),
                     "unsupported_validation_rows": 0},
                ],
            })
        write_new_json(root / "models/CV_SUMMARY.json", {
            "schema": "operator-oracle-quality135-development-cv-v1",
            "scope": "development_only_model_selection_not_external_evaluation",
            "confirm_used_for_tuning": False, "tasks": cv_tasks,
        })
        write_new_json(root / "features/FEATURES_RECEIPT.json", {
            "schema": "operator-oracle-quality135-features-receipt-v1",
            "oracle_labels_read": False, "timing_contract": {}, "timings": {},
        })
        artifact_files = {
            "development_costs": "costs/DEVELOPMENT_COSTS.jsonl",
            "model_index": "models/MODEL_INDEX.json",
            "chosen_oof": "models/CHOSEN_OOF_FULL_RANKS.jsonl",
            "candidate_catalog": "costs/CANDIDATES.json",
            "features_receipt": "features/FEATURES_RECEIPT.json",
            "cv_summary": "models/CV_SUMMARY.json",
        }
        lock = {
            "schema": "operator-oracle-quality135-lock-v1", "status": "locked",
            "require_frozen_current135": False,
            "confirm_used_for_fit_or_selection": False,
            "confirm_rows_read_before_lock": 0,
            "artifacts": {
                name: {"path": path, "sha256": sha256_file(root / path)}
                for name, path in artifact_files.items()
            },
        }
        lock["artifacts"]["confirm_costs"] = {
            "path": "costs/NONEXISTENT_CONFIRM_COSTS.jsonl",
            "sha256": "0" * 64,
        }
        write_new_json(root / "LOCK.json", lock)
        report = run(root, development_only=True)
        require(report["confirm_evaluated"] is False and
                report["split_column_counts"] == {"development": 2} and
                (root / "report/COMPLETE.json").is_file(),
                "selftest report contract mismatch")
        quality = report["quality"]["development"]
        require(
            quality["operator_zstd3_space"]["timing"]["full_scope_ratios"] is None and
            quality["observed_pipeline_space"]["timing"]["full_scope_ratios"] is None,
            "operator/observed domains exposed invented timing")
        lz4_timing = quality["algorithm_encode"]["timing"]["baselines"]["LZ4"]
        require(lz4_timing["timing_comparable_columns"] == 1 and
                lz4_timing["full_scope_compression_ratio"] is None and
                lz4_timing["full_scope_decompression_ratio"] is None,
                "partial timing coverage did not null full-scope ratios")
        failure = quality["operator_zstd3_space"]["coverage"]
        require(failure["selected_unavailable_failures"] == 1 and
                failure["failure_reasons"] == {
                    "selection_unavailable_or_nonfinite": 1},
                "unsupported selected ID was silently replaced or dropped")
        operator_hits = quality["operator_zstd3_space"]["exact_hits"]
        require(operator_hits["topk_denominators"] == {
                    "top1_all_rows_including_failed": 2,
                    "top3_model_policy_rows": 0,
                    "top5_model_policy_rows": 0} and
                operator_hits["metrics"]["allowed_objective_top1"] == {
                    "count": 1, "denominator": 2, "rate": 0.5} and
                operator_hits["metrics"]["allowed_objective_top3"] == {
                    "count": 0, "denominator": 0, "rate": None},
                "unavailable slot or non-model Top-k semantics were misreported")
        algorithm_hits = quality["algorithm_space"]["exact_hits"]
        require(algorithm_hits["topk_denominators"] == {
                    "top1_all_rows_including_failed": 2,
                    "top3_model_policy_rows": 2,
                    "top5_model_policy_rows": 2},
                "learned model Top-k denominators were not retained")
        require(report["scope"]["PPMD8_selectable"] is False and
                all("PPMD8" not in row["candidate_pool"] for row in selectors),
                "PPMD8 escaped its full-oracle-reference-only boundary")

        # A bad gate must be rejected before the nonexistent confirm payload is
        # opened.  Guard Path.open so the test observes any accidental access.
        write_new_json(root / "CONFIRM_GATE.json", {
            "schema": "operator-oracle-quality135-confirm-gate-v1",
            "status": "open_for_single_final_evaluation",
            "confirm_used_for_fit_or_selection": False,
            "lock_sha256": "f" * 64,
            "model_index_sha256": model_sha,
            "all_predictions_sha256": "1" * 64,
            "prediction_receipt_sha256": "2" * 64,
            "confirm_costs_sha256": "0" * 64,
        })
        confirm = (root / "costs/NONEXISTENT_CONFIRM_COSTS.jsonl").resolve()
        original_open = Path.open
        confirm_open_count = 0
        def guarded_open(path: Path, *args, **kwargs):
            nonlocal confirm_open_count
            if path.resolve() == confirm:
                confirm_open_count += 1
                raise AssertionError("confirm payload opened before gate validation")
            return original_open(path, *args, **kwargs)
        Path.open = guarded_open
        try:
            try:
                unlock_confirm(root, lock, {
                    "lock": sha256_file(root / "LOCK.json"),
                    "model_index": model_sha,
                    "all_predictions": "1" * 64,
                    "prediction_receipt": "2" * 64,
                })
                raise AssertionError("mismatched confirm gate was accepted")
            except ReportError as error:
                require("does not bind the current LOCK" in str(error),
                        "gate mismatch failed at the wrong boundary")
        finally:
            Path.open = original_open
        require(confirm_open_count == 0,
                "confirm payload was opened before bad-gate rejection")
        print("report_operator_oracle_quality135_v1 selftest: PASS")
    finally:
        shutil.rmtree(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        require(args.root is None, "--selftest does not accept --root")
        selftest()
        return
    require(args.root is not None, "--root is required")
    report = run(args.root, args.development_only)
    print(json.dumps({
        "status": report["status"],
        "confirm_evaluated": report["confirm_evaluated"],
        "split_column_counts": report["split_column_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        raise SystemExit(1)
