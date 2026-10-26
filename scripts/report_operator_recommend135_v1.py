#!/usr/bin/env python3
"""Build the frozen operator-recommend135 v1 report from immutable run artifacts.

The script reads experiment artifacts only and writes exactly four files below
RUN_ROOT/report.  It deliberately keeps offline Top-K set quality separate from
the deployable Top-1 decision and reports CPU and wall clocks independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


TARGETS = ("space", "encode", "decode", "balanced")
TOPKS = (1, 3, 5)
KINDS = ("raw", "algorithm", "operator", "operator_tree", "algorithm_combination")
EXPECTED_DATASETS = 135
REFERENCE_VALUE_COLUMNS = 3519
OUTPUT_NAMES = (
    "REPORT.md",
    "SUMMARY.json",
    "RECOMMENDATIONS.csv",
    "DATASET_CANDIDATE_RESULTS.csv",
)


class ReportError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def positive_number(value: Any) -> float | None:
    result = finite_number(value)
    return result if result is not None and result > 0 else None


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def geomean(values: Sequence[float]) -> float | None:
    usable = [value for value in values if math.isfinite(value) and value > 0]
    return math.exp(sum(math.log(value) for value in usable) / len(usable)) if usable else None


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_number(value)
    if number is None:
        return "NA"
    if abs(number) >= 1e9 or (0 < abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = finite_number(value)
    return "NA" if number is None else f"{int(number):,}"


def resolve_file(
    explicit: str | None,
    roots: Sequence[Path],
    relative_candidates: Sequence[str],
    basename: str,
    required: bool = True,
) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ReportError(f"explicit input is not a file: {path}")
        return path
    hits: list[Path] = []
    for root in roots:
        for relative in relative_candidates:
            path = (root / relative).resolve()
            if path.is_file() and path not in hits:
                hits.append(path)
    if not hits:
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob(basename):
                try:
                    depth = len(path.relative_to(root).parts)
                except ValueError:
                    continue
                if depth <= 5 and "report" not in path.parts and path.resolve() not in hits:
                    hits.append(path.resolve())
    if not hits:
        if required:
            raise ReportError(f"cannot detect {basename}; pass its explicit argument")
        return None
    if len(hits) > 1:
        rendered = "\n  ".join(str(path) for path in sorted(hits))
        raise ReportError(f"ambiguous {basename}; pass its explicit argument:\n  {rendered}")
    return hits[0]


def as_array(document: Any, key: str, label: str) -> list[dict[str, Any]]:
    value = document.get(key) if isinstance(document, dict) else document
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ReportError(f"{label} must be an array of objects")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def status_class(status: str) -> str:
    normalized = status.lower()
    if normalized == "strict_verified":
        return "strict"
    if normalized in {"lossless_bypass", "wrapper_verified", "bypass", "fallback"}:
        return "bypass"
    if normalized == "domain_rejected":
        return "domain"
    return "fail"


def case_status_by_layer(
    case_index: Sequence[dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    rows: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in case_index:
        sample_id = str(case.get("sample_id", ""))
        category = status_class(str(case.get("status", "failed")))
        for layer in layer_of(samples.get(sample_id, {}), rows.get(sample_id, {})):
            counts[layer][category] += 1
            counts[layer]["total"] += 1
    result: dict[str, dict[str, int]] = {}
    for layer in ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096"):
        result[layer] = {key: counts[layer][key] for key in ("strict", "bypass", "domain", "fail", "total")}
    return result


def flatten_predictions(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds = evaluation.get("folds")
    if not isinstance(folds, list):
        raise ReportError("EVALUATION.json has no folds array")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in folds:
        if not isinstance(fold, dict) or not isinstance(fold.get("predictions"), list):
            raise ReportError("evaluation fold has no predictions array")
        for prediction in fold["predictions"]:
            sample_id = prediction.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in seen:
                raise ReportError(f"evaluation prediction sample_id is missing or duplicated: {sample_id!r}")
            seen.add(sample_id)
            output.append(prediction)
    return output


def ranking_ids(prediction: Mapping[str, Any], target: str) -> list[str]:
    ranking = prediction.get("targets", {}).get(target, {}).get("ranking", [])
    if not isinstance(ranking, list):
        return []
    return [str(entry.get("candidate_id")) for entry in ranking if isinstance(entry, dict) and isinstance(entry.get("candidate_id"), str)]


def validate_compression_replay(
    replay: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    predictions: Sequence[dict[str, Any]],
    rows_sha256: str,
    evaluation_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if replay.get("schema") != "operator-recommend135-compression-replay-v1":
        raise ReportError("unsupported COMPRESSION_REPLAY schema")
    if replay.get("profile") != "compression_only_no_raw":
        raise ReportError("unsupported COMPRESSION_REPLAY profile")
    status = replay.get("status")
    if status not in {"completed", "complete", "verified", "strict_verified", "success", "matched_frozen_rankings"}:
        raise ReportError(f"COMPRESSION_REPLAY.json status is not complete: {status!r}")
    if replay.get("source_rows_sha256") != rows_sha256:
        raise ReportError("COMPRESSION_REPLAY source_rows_sha256 mismatch")
    if replay.get("source_evaluation_sha256") != evaluation_sha256:
        raise ReportError("COMPRESSION_REPLAY source_evaluation_sha256 mismatch")
    expected_catalog = evaluation.get("candidate_scope", {}).get("catalog_sha256")
    if not isinstance(expected_catalog, str) or replay.get("catalog_sha256") != expected_catalog:
        raise ReportError("COMPRESSION_REPLAY catalog_sha256 mismatch")
    folds = evaluation.get("folds", [])
    if replay.get("fold_count") != len(folds):
        raise ReportError("COMPRESSION_REPLAY fold_count mismatch")
    offline_fits = replay.get("offline_fits")
    if not isinstance(offline_fits, list) or len(offline_fits) != len(folds):
        raise ReportError("COMPRESSION_REPLAY offline_fits mismatch")
    frozen_folds = {str(fold.get("held_out_group_id", "")): fold for fold in folds if isinstance(fold, dict)}
    seen_fit_groups: set[str] = set()
    for fit in offline_fits:
        if not isinstance(fit, dict):
            raise ReportError("COMPRESSION_REPLAY offline fit receipt is not an object")
        held_out = str(fit.get("held_out_group_id", ""))
        frozen_fold = frozen_folds.get(held_out)
        if frozen_fold is None or held_out in seen_fit_groups:
            raise ReportError(f"COMPRESSION_REPLAY unknown or duplicate offline fit: {held_out}")
        seen_fit_groups.add(held_out)
        if fit.get("training_row_count") != frozen_fold.get("training_row_count") or fit.get("training_group_ids") != frozen_fold.get("training_group_ids"):
            raise ReportError(f"COMPRESSION_REPLAY offline fit scope mismatch: {held_out}")
        for key in ("offline_fit_cpu_ns", "offline_fit_wall_ns"):
            value = finite_number(fit.get(key))
            if value is None or value < 0:
                raise ReportError(f"COMPRESSION_REPLAY invalid {key}: {held_out}")
    replay_predictions = replay.get("predictions")
    if not isinstance(replay_predictions, list) or replay.get("matched_count") != len(predictions) or len(replay_predictions) != len(predictions):
        raise ReportError("COMPRESSION_REPLAY must match every held-out prediction")
    original_by_id = {str(item.get("sample_id", "")): item for item in predictions}
    result: dict[str, dict[str, Any]] = {}
    for item in replay_predictions:
        if not isinstance(item, dict):
            raise ReportError("COMPRESSION_REPLAY prediction entry is not an object")
        sample_id = str(item.get("sample_id", ""))
        original = original_by_id.get(sample_id)
        if original is None or sample_id in result:
            raise ReportError(f"COMPRESSION_REPLAY unknown or duplicate sample_id: {sample_id}")
        if str(item.get("group_id", "")) != str(original.get("group_id", "")):
            raise ReportError(f"COMPRESSION_REPLAY group mismatch for {sample_id}")
        if item.get("rankings_match_frozen") is not True:
            raise ReportError(f"COMPRESSION_REPLAY ranking receipt is not true for {sample_id}")
        raw_timings = item.get("raw_timings")
        if not isinstance(raw_timings, list) or len(raw_timings) != 5:
            raise ReportError(f"COMPRESSION_REPLAY requires exactly 5 raw timings for {sample_id}")
        for key in ("filter_cpu_ns", "filter_wall_ns", "model_cpu_ns", "model_wall_ns"):
            value = finite_number(item.get(key))
            if value is None or value < 0:
                raise ReportError(f"COMPRESSION_REPLAY invalid {key} for {sample_id}")
        raw_cpu: list[float] = []
        raw_wall: list[float] = []
        repeats: set[int] = set()
        for timing in raw_timings:
            if not isinstance(timing, dict) or not isinstance(timing.get("repeat"), int):
                raise ReportError(f"COMPRESSION_REPLAY invalid raw timing receipt for {sample_id}")
            repeats.add(timing["repeat"])
            cpu = finite_number(timing.get("model_cpu_ns"))
            wall = finite_number(timing.get("model_wall_ns"))
            if cpu is None or wall is None or cpu < 0 or wall < 0 or not isinstance(timing.get("internal_online_cost"), dict):
                raise ReportError(f"COMPRESSION_REPLAY invalid raw model timing for {sample_id}")
            raw_cpu.append(cpu)
            raw_wall.append(wall)
        if repeats != set(range(5)):
            raise ReportError(f"COMPRESSION_REPLAY repeat indices mismatch for {sample_id}")
        if float(item["model_cpu_ns"]) != sorted(raw_cpu)[2] or float(item["model_wall_ns"]) != sorted(raw_wall)[2]:
            raise ReportError(f"COMPRESSION_REPLAY model median mismatch for {sample_id}")
        replay_prediction = item.get("prediction")
        if not isinstance(replay_prediction, dict):
            raise ReportError(f"COMPRESSION_REPLAY missing prediction for {sample_id}")
        if replay_prediction.get("schema") != "operator-recommend135-prediction-v1" or replay_prediction.get("profile") != "compression_only_no_raw":
            raise ReportError(f"COMPRESSION_REPLAY prediction schema/profile mismatch for {sample_id}")
        original_eligible = original.get("prediction", {}).get("eligible_candidates", original.get("eligible_candidates", []))
        expected_eligible = [candidate_id for candidate_id in original_eligible if candidate_id != "raw"]
        if replay_prediction.get("eligible_candidates") != expected_eligible or "raw" in replay_prediction.get("eligible_candidates", []):
            raise ReportError(f"COMPRESSION_REPLAY eligibility mismatch for {sample_id}")
        if replay_prediction.get("candidate_catalog_sha256") != expected_catalog:
            raise ReportError(f"COMPRESSION_REPLAY prediction catalog mismatch for {sample_id}")
        replay_decision = item.get("decision_cost")
        if not isinstance(replay_decision, dict):
            raise ReportError(f"COMPRESSION_REPLAY missing decision_cost for {sample_id}")
        original_decision = original.get("decision_cost", {})
        for key in ("feature_cpu_ns", "feature_wall_ns", "eligibility_cpu_ns", "eligibility_wall_ns"):
            if finite_number(replay_decision.get(key)) != finite_number(original_decision.get(key)):
                raise ReportError(f"COMPRESSION_REPLAY decision cost source mismatch for {sample_id}/{key}")
        for key in ("filter_cpu_ns", "filter_wall_ns", "model_cpu_ns", "model_wall_ns"):
            if finite_number(replay_decision.get(key)) != finite_number(item.get(key)):
                raise ReportError(f"COMPRESSION_REPLAY decision cost receipt mismatch for {sample_id}/{key}")
        expected_cpu = sum(float(replay_decision[key]) for key in ("feature_cpu_ns", "eligibility_cpu_ns", "filter_cpu_ns", "model_cpu_ns"))
        expected_wall = sum(float(replay_decision[key]) for key in ("feature_wall_ns", "eligibility_wall_ns", "filter_wall_ns", "model_wall_ns"))
        if finite_number(replay_decision.get("online_decision_cpu_ns")) != expected_cpu or finite_number(replay_decision.get("online_decision_wall_ns")) != expected_wall:
            raise ReportError(f"COMPRESSION_REPLAY decision cost total mismatch for {sample_id}")
        if replay_decision.get("probe_used") is not False:
            raise ReportError(f"COMPRESSION_REPLAY probe_used must be false for {sample_id}")
        original_prediction = original.get("prediction", {})
        for target in TARGETS:
            original_target = original_prediction.get("targets", {}).get(target, {})
            replay_target = replay_prediction.get("targets", {}).get(target, {})
            original_ranking = original_target.get("ranking", [])
            replay_ranking = replay_target.get("ranking", [])
            expected_ranking = [
                entry for entry in original_ranking
                if isinstance(entry, dict) and entry.get("candidate_id") != "raw"
            ]
            if replay_ranking != expected_ranking:
                raise ReportError(f"COMPRESSION_REPLAY filtered ranking/score mismatch for {sample_id}/{target}")
            expected_ids = [str(entry["candidate_id"]) for entry in expected_ranking]
            for k in TOPKS:
                if replay_target.get(f"top{k}") != expected_ids[:k]:
                    raise ReportError(f"COMPRESSION_REPLAY top{k} mismatch for {sample_id}/{target}")
        result[sample_id] = item
    if set(result) != set(original_by_id):
        raise ReportError("COMPRESSION_REPLAY sample set mismatch")
    return result, {
        "status": status,
        "schema": replay["schema"],
        "profile": replay["profile"],
        "fold_count": len(folds),
        "matched_count": len(result),
        "model_prediction_repeats": 5,
        "filter_timing_repeats": 1,
        "rows_sha256_verified": True,
        "evaluation_sha256_verified": True,
        "catalog_sha256_verified": True,
        "all_rankings_and_scores_match_frozen_filter_exactly": True,
    }


def layer_of(sample: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    scale = str(sample.get("scale", row.get("scale", "unknown")))
    count = finite_number(sample.get("element_count"))
    layers = ["all"]
    if scale == "u64":
        layers.append("u64")
    elif scale == "primary":
        layers.append("primary_all")
        layers.append("primary_ge4096" if count is not None and count >= 4096 else "primary_lt4096")
    else:
        layers.append(scale)
    return layers


def weighted_stats(records: Sequence[tuple[str, float]], mode: str) -> dict[str, Any]:
    usable = [(dataset, value) for dataset, value in records if math.isfinite(value)]
    if mode == "geomean":
        usable = [(dataset, value) for dataset, value in usable if value > 0]
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for dataset, value in usable:
        by_dataset[dataset].append(value)
    reducer: Callable[[Sequence[float]], float | None] = geomean if mode == "geomean" else mean
    dataset_values = [reducer(values) for values in by_dataset.values()]
    clean_dataset_values = [value for value in dataset_values if value is not None]
    return {
        "sample_equal": reducer([value for _, value in usable]),
        "dataset_equal": reducer(clean_dataset_values),
        "sample_count": len(usable),
        "dataset_count": len(by_dataset),
    }


def score_metrics(
    predictions: Sequence[dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    rows: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    layer_names = ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096")
    for layer in layer_names:
        layer_predictions = [
            prediction
            for prediction in predictions
            if layer in layer_of(samples.get(prediction["sample_id"], {}), rows.get(prediction["sample_id"], {}))
        ]
        layer_result: dict[str, Any] = {"prediction_count": len(layer_predictions), "targets": {}}
        for target in TARGETS:
            target_result: dict[str, Any] = {}
            for k in TOPKS:
                cpu_regrets: list[tuple[str, float]] = []
                wall_regrets: list[tuple[str, float]] = []
                cpu_hits: list[tuple[str, float]] = []
                wall_hits: list[tuple[str, float]] = []
                for prediction in layer_predictions:
                    dataset = str(prediction.get("dataset_id", ""))
                    score = prediction.get("scores", {}).get(target, {}).get("topk", {}).get(str(k), {})
                    cpu = positive_number(score.get("oracle_best_within_topk_multiplicative_regret"))
                    wall = positive_number(score.get("wall_oracle_best_within_topk_multiplicative_regret"))
                    if cpu is not None:
                        cpu_regrets.append((dataset, cpu))
                    if wall is not None:
                        wall_regrets.append((dataset, wall))
                    if isinstance(score.get("tie_aware_hit"), bool):
                        cpu_hits.append((dataset, float(score["tie_aware_hit"])))
                    if isinstance(score.get("wall_tie_aware_hit"), bool):
                        wall_hits.append((dataset, float(score["wall_tie_aware_hit"])))
                cpu_stats = weighted_stats(cpu_regrets, "geomean")
                wall_stats = weighted_stats(wall_regrets, "geomean")
                cpu_hit_stats = weighted_stats(cpu_hits, "mean")
                wall_hit_stats = weighted_stats(wall_hits, "mean")
                target_result[f"top{k}"] = {
                    "semantics": "online_top1_no_probe" if k == 1 else "offline_set_best_quality_only",
                    "cpu": {
                        **cpu_stats,
                        "sample_equal_oracle_gap_percent": None if cpu_stats["sample_equal"] is None else 100 * (cpu_stats["sample_equal"] - 1),
                        "dataset_equal_oracle_gap_percent": None if cpu_stats["dataset_equal"] is None else 100 * (cpu_stats["dataset_equal"] - 1),
                        "sample_equal_tie_aware_hit_rate": cpu_hit_stats["sample_equal"],
                        "dataset_equal_tie_aware_hit_rate": cpu_hit_stats["dataset_equal"],
                    },
                    "wall": {
                        **wall_stats,
                        "sample_equal_oracle_gap_percent": None if wall_stats["sample_equal"] is None else 100 * (wall_stats["sample_equal"] - 1),
                        "dataset_equal_oracle_gap_percent": None if wall_stats["dataset_equal"] is None else 100 * (wall_stats["dataset_equal"] - 1),
                        "sample_equal_tie_aware_hit_rate": wall_hit_stats["sample_equal"],
                        "dataset_equal_tie_aware_hit_rate": wall_hit_stats["dataset_equal"],
                    },
                }
            layer_result["targets"][target] = target_result
        output[layer] = layer_result
    return output


def baseline_metrics(
    predictions: Sequence[dict[str, Any]],
    rows: Mapping[str, dict[str, Any]],
    candidate_kind: Mapping[str, str],
    fixed_recipe_label: str = "fixed_full_recipe",
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target in TARGETS:
        target_output: dict[str, Any] = {}
        for baseline_key, label in (("fixed_algorithm_baseline", "fixed_algorithm"), ("fixed_candidate_baseline", fixed_recipe_label)):
            clocks: dict[str, Any] = {}
            for clock in ("cpu", "wall"):
                records: list[tuple[str, float]] = []
                invalid_algorithm_ids: set[str] = set()
                for item in predictions:
                    score = item.get("scores", {}).get(target, {})
                    baseline = score.get(baseline_key)
                    if not isinstance(baseline, dict) or baseline.get("available") is not True:
                        continue
                    candidate_id = str(baseline.get("candidate_id", ""))
                    if label == "fixed_algorithm" and candidate_kind.get(candidate_id) != "algorithm":
                        invalid_algorithm_ids.add(candidate_id)
                        continue
                    oracle_object = score.get("oracle" if clock == "cpu" else "wall_oracle")
                    oracle = positive_number(oracle_object.get("cost") if isinstance(oracle_object, dict) else None)
                    selected_regret = positive_number(
                        score.get("topk", {}).get("1", {}).get(
                            "online_selected_multiplicative_regret" if clock == "cpu" else "wall_online_selected_multiplicative_regret"
                        )
                    )
                    baseline_cost = positive_number(baseline.get("cost" if clock == "cpu" else "wall_cost"))
                    if oracle is None or selected_regret is None or baseline_cost is None:
                        continue
                    selected_cost = oracle * selected_regret
                    records.append((str(item.get("dataset_id", "")), baseline_cost / selected_cost))
                stats = weighted_stats(records, "geomean")
                clocks[clock] = {
                    **stats,
                    "sample_equal_benefit_percent": None if stats["sample_equal"] is None else 100 * (stats["sample_equal"] - 1),
                    "dataset_equal_benefit_percent": None if stats["dataset_equal"] is None else 100 * (stats["dataset_equal"] - 1),
                    "wins": sum(value > 1.000000001 for _, value in records),
                    "ties": sum(abs(value - 1) <= 1e-9 for _, value in records),
                    "losses": sum(value < 0.999999999 for _, value in records),
                    "invalid_fixed_algorithm_ids": sorted(invalid_algorithm_ids),
                }
            target_output[label] = clocks
        output[target] = target_output

    # Encode is the deployable end-to-end comparison: feature + eligibility +
    # model are added to the selected candidate encode time.
    end_to_end: dict[str, Any] = {}
    for baseline_key, label in (("fixed_algorithm_baseline", "fixed_algorithm"), ("fixed_candidate_baseline", fixed_recipe_label)):
        end_to_end[label] = {}
        for clock in ("cpu", "wall"):
            records: list[tuple[str, float]] = []
            for item in predictions:
                row = rows.get(str(item.get("sample_id", "")), {})
                input_bytes = positive_number(row.get("input_bytes"))
                encode_score = item.get("scores", {}).get("encode", {})
                baseline = encode_score.get(baseline_key)
                if input_bytes is None or not isinstance(baseline, dict) or baseline.get("available") is not True:
                    continue
                candidate_id = str(baseline.get("candidate_id", ""))
                if label == "fixed_algorithm" and candidate_kind.get(candidate_id) != "algorithm":
                    continue
                baseline_per_byte = positive_number(baseline.get("cost" if clock == "cpu" else "wall_cost"))
                selected_total = positive_number(encode_score.get(f"selected_total_encode_{clock}_ns"))
                if baseline_per_byte is not None and selected_total is not None:
                    records.append((str(item.get("dataset_id", "")), baseline_per_byte * input_bytes / selected_total))
            stats = weighted_stats(records, "geomean")
            end_to_end[label][clock] = {
                **stats,
                "sample_equal_benefit_percent": None if stats["sample_equal"] is None else 100 * (stats["sample_equal"] - 1),
                "dataset_equal_benefit_percent": None if stats["dataset_equal"] is None else 100 * (stats["dataset_equal"] - 1),
                "wins": sum(value > 1.000000001 for _, value in records),
                "ties": sum(abs(value - 1) <= 1e-9 for _, value in records),
                "losses": sum(value < 0.999999999 for _, value in records),
            }
    output["encode_end_to_end_with_feature_eligibility_model"] = end_to_end
    return output


def strict_candidate_costs(row: Mapping[str, Any], candidate_id: str) -> dict[str, float] | None:
    metric = row.get("metrics", {}).get(candidate_id)
    if not isinstance(metric, dict) or status_class(str(metric.get("status", ""))) != "strict":
        return None
    input_bytes = positive_number(row.get("input_bytes"))
    frame_bytes = positive_number(metric.get("frame_bytes"))
    encode_cpu = finite_number(metric.get("encode_cpu_ns"))
    decode_cpu = finite_number(metric.get("decode_cpu_ns"))
    encode_wall = finite_number(metric.get("encode_wall_ns"))
    decode_wall = finite_number(metric.get("decode_wall_ns"))
    if input_bytes is None or frame_bytes is None or any(value is None or value < 0 for value in (encode_cpu, decode_cpu, encode_wall, decode_wall)):
        return None
    space = frame_bytes / input_bytes
    encode = max(float(encode_cpu), 1.0) / input_bytes
    decode = max(float(decode_cpu), 1.0) / input_bytes
    encode_wall_cost = max(float(encode_wall), 1.0) / input_bytes
    decode_wall_cost = max(float(decode_wall), 1.0) / input_bytes
    return {
        "space": space,
        "encode": encode,
        "decode": decode,
        "balanced": (space * encode * decode) ** (1.0 / 3.0),
        "space_wall": space,
        "encode_wall": encode_wall_cost,
        "decode_wall": decode_wall_cost,
        "balanced_wall": (space * encode_wall_cost * decode_wall_cost) ** (1.0 / 3.0),
    }


def choose_training_baseline(
    training_rows: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    target: str,
) -> str | None:
    best: tuple[float, str] | None = None
    for candidate_id in candidate_ids:
        costs: list[float] = []
        for row in training_rows:
            candidate_costs = strict_candidate_costs(row, candidate_id)
            if candidate_costs is None:
                costs = []
                break
            costs.append(candidate_costs[target])
        if len(costs) != len(training_rows) or not costs:
            continue
        score = sum(math.log(value) for value in costs) / len(costs)
        if best is None or score < best[0] - 1e-12 or (abs(score - best[0]) <= 1e-12 and candidate_id < best[1]):
            best = (score, candidate_id)
    return best[1] if best else None


def heldout_baseline_score(
    row: Mapping[str, Any],
    candidate_id: str | None,
    target: str,
    oracle_cpu: float,
    oracle_wall: float,
) -> dict[str, Any] | None:
    if candidate_id is None:
        return None
    costs = strict_candidate_costs(row, candidate_id)
    if costs is None:
        return {"candidate_id": candidate_id, "available": False}
    cpu = costs[target]
    wall = costs[target if target == "space" else f"{target}_wall"]
    return {
        "candidate_id": candidate_id,
        "available": True,
        "cost": cpu,
        "multiplicative_regret": cpu / oracle_cpu,
        "wall_cost": wall,
        "wall_multiplicative_regret": wall / oracle_wall,
    }


def raw_selection_frequency(
    predictions: Sequence[dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    rows: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer in ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096"):
        layer_predictions = [
            item for item in predictions
            if layer in layer_of(samples.get(str(item.get("sample_id", "")), {}), rows.get(str(item.get("sample_id", "")), {}))
        ]
        output[layer] = {}
        for target in TARGETS:
            result: dict[str, Any] = {}
            for k in TOPKS:
                records: list[tuple[str, float]] = []
                for item in layer_predictions:
                    target_prediction = item.get("prediction", {}).get("targets", {}).get(target, {})
                    top_ids = target_prediction.get(f"top{k}")
                    if not isinstance(top_ids, list):
                        ranking = target_prediction.get("ranking", [])
                        top_ids = [entry.get("candidate_id") for entry in ranking[:k] if isinstance(entry, dict)]
                    selected_raw = bool(top_ids) and (top_ids[0] == "raw" if k == 1 else "raw" in top_ids)
                    records.append((str(item.get("dataset_id", "")), float(selected_raw)))
                stats = weighted_stats(records, "mean")
                result[f"top{k}"] = {
                    "meaning": "top1_selected_raw" if k == 1 else f"raw_present_in_original_top{k}",
                    "sample_equal_rate": stats["sample_equal"],
                    "dataset_equal_rate": stats["dataset_equal"],
                    "sample_count": stats["sample_count"],
                    "dataset_count": stats["dataset_count"],
                }
            output[layer][target] = result
    return output


def compression_only_analysis(
    evaluation: Mapping[str, Any],
    predictions: Sequence[dict[str, Any]],
    rows: Mapping[str, dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    candidate_kind: Mapping[str, str],
    replay_by_sample: Mapping[str, dict[str, Any]] | None,
    replay_validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    nonraw_candidates = sorted(candidate_id for candidate_id in candidate_kind if candidate_id != "raw")
    algorithm_candidates = sorted(candidate_id for candidate_id in nonraw_candidates if candidate_kind[candidate_id] == "algorithm")
    prediction_by_id = {str(item.get("sample_id", "")): item for item in predictions}
    derived: list[dict[str, Any]] = []
    fold_baselines: list[dict[str, Any]] = []

    for fold in evaluation.get("folds", []):
        held_out_group = str(fold.get("held_out_group_id", ""))
        training_groups = fold.get("training_group_ids")
        if isinstance(training_groups, list):
            training_group_set = {str(group) for group in training_groups}
            training_rows = [row for row in rows.values() if str(row.get("group_id", "")) in training_group_set]
        else:
            training_rows = [row for row in rows.values() if str(row.get("group_id", "")) != held_out_group]
        choices: dict[str, dict[str, str | None]] = {}
        for target in TARGETS:
            choices[target] = {
                "fixed_algorithm": choose_training_baseline(training_rows, algorithm_candidates, target),
                "fixed_nonraw_recipe": choose_training_baseline(training_rows, nonraw_candidates, target),
            }
        fold_baselines.append({"held_out_group_id": held_out_group, "training_row_count": len(training_rows), "targets": choices})

        for fold_prediction in fold.get("predictions", []):
            sample_id = str(fold_prediction.get("sample_id", ""))
            original = prediction_by_id.get(sample_id)
            row = rows.get(sample_id)
            if original is None or row is None:
                raise ReportError(f"compression_only cannot bind held-out prediction {sample_id}")
            original_decision = original.get("decision_cost", {})
            feature_cpu = finite_number(original_decision.get("feature_cpu_ns"))
            feature_wall = finite_number(original_decision.get("feature_wall_ns"))
            eligibility_cpu = finite_number(original_decision.get("eligibility_cpu_ns"))
            eligibility_wall = finite_number(original_decision.get("eligibility_wall_ns"))
            if any(value is None or value < 0 for value in (feature_cpu, feature_wall, eligibility_cpu, eligibility_wall)):
                raise ReportError(f"compression_only invalid feature/eligibility cost for {sample_id}")
            if replay_by_sample is not None:
                replay_item = replay_by_sample[sample_id]
                filter_cpu = float(replay_item["filter_cpu_ns"])
                filter_wall = float(replay_item["filter_wall_ns"])
                model_cpu = float(replay_item["model_cpu_ns"])
                model_wall = float(replay_item["model_wall_ns"])
                decision_source = "verified_compression_replay_5x"
            else:
                filter_cpu = 0.0
                filter_wall = 0.0
                model_cpu_value = finite_number(original_decision.get("model_cpu_ns"))
                model_wall_value = finite_number(original_decision.get("model_wall_ns"))
                if model_cpu_value is None or model_wall_value is None:
                    raise ReportError(f"compression_only invalid reference model cost for {sample_id}")
                model_cpu = model_cpu_value
                model_wall = model_wall_value
                decision_source = "reference_original_24_candidate_model_filter_unmeasured"
            compression_decision = {
                "feature_cpu_ns": feature_cpu,
                "feature_wall_ns": feature_wall,
                "eligibility_cpu_ns": eligibility_cpu,
                "eligibility_wall_ns": eligibility_wall,
                "filter_cpu_ns": filter_cpu,
                "filter_wall_ns": filter_wall,
                "model_cpu_ns": model_cpu,
                "model_wall_ns": model_wall,
                "online_decision_cpu_ns": feature_cpu + eligibility_cpu + filter_cpu + model_cpu,
                "online_decision_wall_ns": feature_wall + eligibility_wall + filter_wall + model_wall,
                "source": decision_source,
            }
            derived_scores: dict[str, Any] = {}
            for target in TARGETS:
                actual: dict[str, dict[str, float]] = {}
                for candidate_id in nonraw_candidates:
                    costs = strict_candidate_costs(row, candidate_id)
                    if costs is not None:
                        actual[candidate_id] = costs
                if not actual:
                    derived_scores[target] = {
                        "oracle": None,
                        "wall_oracle": None,
                        "topk": {},
                        "fixed_algorithm_baseline": None,
                        "fixed_candidate_baseline": None,
                        "selected_total_encode_cpu_ns": None,
                        "selected_total_encode_wall_ns": None,
                    }
                    continue
                wall_key = target if target == "space" else f"{target}_wall"
                oracle_cpu = min(cost[target] for cost in actual.values())
                oracle_wall = min(cost[wall_key] for cost in actual.values())
                tied_cpu = {candidate_id for candidate_id, cost in actual.items() if abs(cost[target] - oracle_cpu) <= 1e-12 * max(1.0, abs(cost[target]), abs(oracle_cpu))}
                tied_wall = {candidate_id for candidate_id, cost in actual.items() if abs(cost[wall_key] - oracle_wall) <= 1e-12 * max(1.0, abs(cost[wall_key]), abs(oracle_wall))}

                # Candidate IDs are selected solely by stable filtering of the
                # frozen held-out ranking.  Actual held-out metrics are read
                # only below, after selection, for oracle/scoring.
                ranking = original.get("prediction", {}).get("targets", {}).get(target, {}).get("ranking", [])
                filtered_ids = [
                    str(entry.get("candidate_id")) for entry in ranking
                    if isinstance(entry, dict) and entry.get("candidate_id") != "raw"
                ]
                topk: dict[str, Any] = {}
                for k in TOPKS:
                    selected_ids = filtered_ids[:k]
                    observed_cpu = [actual[candidate_id][target] for candidate_id in selected_ids if candidate_id in actual]
                    observed_wall = [actual[candidate_id][wall_key] for candidate_id in selected_ids if candidate_id in actual]
                    score: dict[str, Any] = {
                        "candidates": selected_ids,
                        "tie_aware_hit": any(candidate_id in tied_cpu for candidate_id in selected_ids),
                        "wall_tie_aware_hit": any(candidate_id in tied_wall for candidate_id in selected_ids),
                        "oracle_best_within_topk_multiplicative_regret": min(observed_cpu) / oracle_cpu if observed_cpu else None,
                        "wall_oracle_best_within_topk_multiplicative_regret": min(observed_wall) / oracle_wall if observed_wall else None,
                        "semantics": "online_top1_static_raw_filter_no_probe" if k == 1 else "offline_set_quality_static_raw_filter_no_probe",
                    }
                    if k == 1:
                        score["online_selected_multiplicative_regret"] = score["oracle_best_within_topk_multiplicative_regret"]
                        score["wall_online_selected_multiplicative_regret"] = score["wall_oracle_best_within_topk_multiplicative_regret"]
                    topk[str(k)] = score
                selected_id = filtered_ids[0] if filtered_ids else None
                selected_metric = row.get("metrics", {}).get(selected_id, {}) if selected_id else {}
                encode_cpu = finite_number(selected_metric.get("encode_cpu_ns")) if isinstance(selected_metric, dict) else None
                encode_wall = finite_number(selected_metric.get("encode_wall_ns")) if isinstance(selected_metric, dict) else None
                online_cpu = compression_decision["online_decision_cpu_ns"]
                online_wall = compression_decision["online_decision_wall_ns"]
                selected_strict = selected_id in actual
                derived_scores[target] = {
                    "oracle": {"cost": oracle_cpu, "candidates": sorted(tied_cpu)},
                    "wall_oracle": {"cost": oracle_wall, "candidates": sorted(tied_wall)},
                    "topk": topk,
                    "fixed_algorithm_baseline": heldout_baseline_score(row, choices[target]["fixed_algorithm"], target, oracle_cpu, oracle_wall),
                    "fixed_candidate_baseline": heldout_baseline_score(row, choices[target]["fixed_nonraw_recipe"], target, oracle_cpu, oracle_wall),
                    "selected_total_encode_cpu_ns": float(encode_cpu) + float(online_cpu) if selected_strict and encode_cpu is not None and online_cpu is not None else None,
                    "selected_total_encode_wall_ns": float(encode_wall) + float(online_wall) if selected_strict and encode_wall is not None and online_wall is not None else None,
                }
            derived.append({
                "sample_id": sample_id,
                "dataset_id": original.get("dataset_id"),
                "group_id": original.get("group_id"),
                "scale": original.get("scale"),
                "dtype": original.get("dtype"),
                "scores": derived_scores,
                "decision_cost": compression_decision,
            })

    if len(derived) != len(predictions) or {item["sample_id"] for item in derived} != set(prediction_by_id):
        raise ReportError("compression_only did not derive exactly one result per held-out prediction")
    layers = ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096")
    decision_keys = (
        "feature_cpu_ns", "feature_wall_ns", "eligibility_cpu_ns", "eligibility_wall_ns",
        "filter_cpu_ns", "filter_wall_ns", "model_cpu_ns", "model_wall_ns",
        "online_decision_cpu_ns", "online_decision_wall_ns",
    )
    online_cost_by_layer: dict[str, Any] = {}
    baselines_by_layer: dict[str, Any] = {}
    encode_total_by_layer: dict[str, Any] = {}
    for layer in layers:
        layer_items = [
            item for item in derived
            if layer in layer_of(samples.get(str(item.get("sample_id", "")), {}), rows.get(str(item.get("sample_id", "")), {}))
        ]
        online_cost = {
            key: int(sum(float(item["decision_cost"][key]) for item in layer_items))
            for key in decision_keys
        }
        online_cost["prediction_count"] = len(layer_items)
        online_cost["source"] = "verified_compression_replay_5x" if replay_by_sample is not None else "reference_original_24_candidate_model_filter_unmeasured"
        online_cost_by_layer[layer] = online_cost
        baselines_by_layer[layer] = baseline_metrics(layer_items, rows, candidate_kind, fixed_recipe_label="fixed_nonraw_recipe")
        encode_cpu_records: list[tuple[str, float]] = []
        encode_wall_records: list[tuple[str, float]] = []
        for item in layer_items:
            score = item["scores"]["encode"]
            cpu = positive_number(score.get("selected_total_encode_cpu_ns"))
            wall = positive_number(score.get("selected_total_encode_wall_ns"))
            dataset = str(item.get("dataset_id", ""))
            if cpu is not None:
                encode_cpu_records.append((dataset, cpu))
            if wall is not None:
                encode_wall_records.append((dataset, wall))
        encode_total_by_layer[layer] = {
            "cpu_ns": {
                **weighted_stats(encode_cpu_records, "mean"),
                "sum": sum(value for _, value in encode_cpu_records),
            },
            "wall_ns": {
                **weighted_stats(encode_wall_records, "mean"),
                "sum": sum(value for _, value in encode_wall_records),
            },
            "contract": "feature + eligibility + filter + model + selected nonraw encode",
        }
    return {
        "schema": "operator-recommend135-compression-only-static-filter-v1",
        "selection_contract": {
            "candidate_filter": "candidate_id != 'raw'",
            "selection_source": "frozen held-out prediction ranking, preserving order",
            "heldout_metrics_used_for_selection": False,
            "retrained": False,
            "retuned": False,
            "probe_used": False,
            "model_cost_scope": "verified replay with frozen 24-candidate fold model and nonraw eligibility" if replay_by_sample is not None else "original complete 24-candidate model cost retained",
            "static_filter_cost": "verified single outer measurement" if replay_by_sample is not None else "not separately measured; excluded from numeric decision cost",
        },
        "candidate_ids": nonraw_candidates,
        "fold_baselines": fold_baselines,
        "metrics": score_metrics(derived, samples, rows),
        "baselines": baselines_by_layer["all"],
        "baselines_by_layer": baselines_by_layer,
        "raw_selection_frequency": raw_selection_frequency(predictions, samples, rows),
        "online_decision_cost": online_cost_by_layer["all"],
        "online_decision_cost_by_layer": online_cost_by_layer,
        "selected_total_encode_cost_by_layer": encode_total_by_layer,
        "compression_replay_validation": replay_validation,
    }


def aggregate_dataset_candidates(
    rows: Sequence[dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    candidate_kind: Mapping[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        sample = samples.get(str(row.get("sample_id", "")), {})
        for candidate_id, metric in row.get("metrics", {}).items():
            if isinstance(metric, dict):
                grouped[(str(row.get("dataset_id", "")), str(row.get("scale", "")), candidate_id)].append((row, sample, metric))
    output: list[dict[str, Any]] = []
    for (dataset, scale, candidate_id), items in sorted(grouped.items()):
        statuses = Counter(status_class(str(metric.get("status", "failed"))) for _, _, metric in items)
        strict = [(row, sample, metric) for row, sample, metric in items if status_class(str(metric.get("status", ""))) == "strict"]
        numeric = sum(finite_number(sample.get("tail", {}).get("values_bytes")) or 0 for _, sample, _ in strict)
        full_input = sum(finite_number(row.get("input_bytes")) or 0 for row, _, _ in strict)
        frame = sum(finite_number(metric.get("frame_bytes")) or 0 for _, _, metric in strict)
        output.append({
            "dataset_id": dataset,
            "group_id": str(items[0][0].get("group_id", "")),
            "scale": scale,
            "candidate_id": candidate_id,
            "kind": candidate_kind.get(candidate_id, str(items[0][2].get("kind", "unknown"))),
            "raw_control": candidate_kind.get(candidate_id) == "raw",
            "sample_count": len(items),
            "strict_count": statuses["strict"],
            "bypass_count": statuses["bypass"],
            "domain_rejected_count": statuses["domain"],
            "failed_count": statuses["fail"],
            "numeric_bytes": int(numeric),
            "values_plus_mask_bytes": int(full_input),
            "full_frame_bytes": int(frame),
            "numeric_bytes_per_full_frame_byte": numeric / frame if frame else None,
            "values_plus_mask_bytes_per_full_frame_byte": full_input / frame if frame else None,
            "full_frame_bytes_per_numeric_byte": frame / numeric if numeric else None,
            "full_frame_bytes_per_values_plus_mask_byte": frame / full_input if full_input else None,
            "mean_encode_cpu_ns": mean([float(metric["encode_cpu_ns"]) for _, _, metric in strict if finite_number(metric.get("encode_cpu_ns")) is not None]),
            "mean_decode_cpu_ns": mean([float(metric["decode_cpu_ns"]) for _, _, metric in strict if finite_number(metric.get("decode_cpu_ns")) is not None]),
            "mean_encode_wall_ns": mean([float(metric["encode_wall_ns"]) for _, _, metric in strict if finite_number(metric.get("encode_wall_ns")) is not None]),
            "mean_decode_wall_ns": mean([float(metric["decode_wall_ns"]) for _, _, metric in strict if finite_number(metric.get("decode_wall_ns")) is not None]),
        })
    return output


def compression_ratio_metrics(
    rows: Sequence[dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    candidate_kind: Mapping[str, str],
) -> dict[str, Any]:
    records: dict[str, dict[str, dict[str, list[tuple[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: {"numeric": [], "values_plus_mask": []})
    )
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        dataset = str(row.get("dataset_id", ""))
        sample = samples.get(sample_id, {})
        numeric_bytes = positive_number(sample.get("tail", {}).get("values_bytes"))
        input_bytes = positive_number(row.get("input_bytes"))
        for candidate_id, metric in row.get("metrics", {}).items():
            if not isinstance(metric, dict) or status_class(str(metric.get("status", ""))) != "strict":
                continue
            frame_bytes = positive_number(metric.get("frame_bytes"))
            if frame_bytes is None:
                continue
            for layer in layer_of(sample, row):
                if numeric_bytes is not None:
                    records[layer][candidate_id]["numeric"].append((dataset, numeric_bytes / frame_bytes))
                if input_bytes is not None:
                    records[layer][candidate_id]["values_plus_mask"].append((dataset, input_bytes / frame_bytes))
    output: dict[str, Any] = {}
    for layer in ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096"):
        output[layer] = {}
        for candidate_id in sorted(candidate_kind):
            numeric = weighted_stats(records[layer][candidate_id]["numeric"], "mean")
            full = weighted_stats(records[layer][candidate_id]["values_plus_mask"], "mean")
            output[layer][candidate_id] = {
                "kind": candidate_kind[candidate_id],
                "strict_sample_count": numeric["sample_count"],
                "dataset_count": numeric["dataset_count"],
                "numeric_bytes_per_full_frame_byte": {
                    "sample_equal": numeric["sample_equal"],
                    "dataset_equal": numeric["dataset_equal"],
                },
                "values_plus_mask_bytes_per_full_frame_byte": {
                    "sample_equal": full["sample_equal"],
                    "dataset_equal": full["dataset_equal"],
                },
            }
    return output


def recommendation_rows(
    predictions: Sequence[dict[str, Any]],
    rows: Mapping[str, dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    candidate_kind: Mapping[str, str],
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        sample_id = str(item.get("sample_id", ""))
        dataset = str(item.get("dataset_id", ""))
        scale = str(item.get("scale", rows.get(sample_id, {}).get("scale", "")))
        row = rows.get(sample_id, {})
        sample = samples.get(sample_id, {})
        for target in TARGETS:
            ranking = item.get("prediction", {}).get("targets", {}).get(target, {}).get("ranking", [])
            for kind in KINDS:
                choice = next((entry for entry in ranking if candidate_kind.get(str(entry.get("candidate_id", ""))) == kind), None)
                if choice is None:
                    continue
                candidate_id = str(choice["candidate_id"])
                metric = row.get("metrics", {}).get(candidate_id, {})
                selected[(dataset, scale, target, kind)].append({
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "predicted_log_regret": finite_number(choice.get("predicted_log_regret")),
                    "status": status_class(str(metric.get("status", "failed"))),
                    "metric": metric,
                    "numeric_bytes": finite_number(sample.get("tail", {}).get("values_bytes")) or 0,
                    "input_bytes": finite_number(row.get("input_bytes")) or 0,
                })
    output: list[dict[str, Any]] = []
    for (dataset, scale, target, kind), choices in sorted(selected.items()):
        distribution = Counter(choice["candidate_id"] for choice in choices)
        mode_id, mode_count = sorted(distribution.items(), key=lambda pair: (-pair[1], pair[0]))[0]
        strict = [choice for choice in choices if choice["status"] == "strict"]
        frame = sum(finite_number(choice["metric"].get("frame_bytes")) or 0 for choice in strict)
        numeric = sum(choice["numeric_bytes"] for choice in strict)
        full_input = sum(choice["input_bytes"] for choice in strict)
        output.append({
            "dataset_id": dataset,
            "group_id": str(rows.get(choices[0]["sample_id"], {}).get("group_id", "")),
            "scale": scale,
            "target": target,
            "candidate_family_kind": kind,
            "raw_control": kind == "raw",
            "representative_sample_count": len(choices),
            "strict_actual_count": len(strict),
            "mode_candidate_id": mode_id,
            "mode_selection_count": mode_count,
            "candidate_selection_counts_json": json.dumps(dict(sorted(distribution.items())), sort_keys=True, separators=(",", ":")),
            "mean_predicted_log_regret": mean([choice["predicted_log_regret"] for choice in choices if choice["predicted_log_regret"] is not None]),
            "numeric_bytes_per_full_frame_byte": numeric / frame if frame else None,
            "values_plus_mask_bytes_per_full_frame_byte": full_input / frame if frame else None,
            "mean_actual_encode_cpu_ns": mean([float(choice["metric"]["encode_cpu_ns"]) for choice in strict if finite_number(choice["metric"].get("encode_cpu_ns")) is not None]),
            "mean_actual_decode_cpu_ns": mean([float(choice["metric"]["decode_cpu_ns"]) for choice in strict if finite_number(choice["metric"].get("decode_cpu_ns")) is not None]),
            "mean_actual_encode_wall_ns": mean([float(choice["metric"]["encode_wall_ns"]) for choice in strict if finite_number(choice["metric"].get("encode_wall_ns")) is not None]),
            "mean_actual_decode_wall_ns": mean([float(choice["metric"]["decode_wall_ns"]) for choice in strict if finite_number(choice["metric"].get("decode_wall_ns")) is not None]),
        })
    return output


def decision_and_offline_costs(predictions: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    strict_metrics = [
        metric
        for row in rows
        for metric in row.get("metrics", {}).values()
        if isinstance(metric, dict) and status_class(str(metric.get("status", ""))) == "strict"
    ]
    median_equivalent = {
        key: int(sum(finite_number(metric.get(key)) or 0 for metric in strict_metrics))
        for key in ("encode_cpu_ns", "decode_cpu_ns", "encode_wall_ns", "decode_wall_ns")
    }
    components = ("feature_cpu_ns", "eligibility_cpu_ns", "model_cpu_ns", "feature_wall_ns", "eligibility_wall_ns", "model_wall_ns", "online_decision_cpu_ns", "online_decision_wall_ns")
    online = {
        key: int(sum(finite_number(item.get("decision_cost", {}).get(key)) or 0 for item in predictions))
        for key in components
    }
    folds = evaluation.get("folds", [])
    fit = {
        "offline_fit_cpu_ns": int(sum(finite_number(fold.get("offline_fit_cpu_ns")) or 0 for fold in folds if isinstance(fold, dict))),
        "offline_fit_wall_ns": int(sum(finite_number(fold.get("offline_fit_wall_ns")) or 0 for fold in folds if isinstance(fold, dict))),
    }
    return {
        "offline_full_matrix_median_equivalent": {
            **median_equivalent,
            "strict_case_count": len(strict_metrics),
            "interpretation": "sum of per-case medians; excludes warmup, repeated trials, process startup, verification, fsync and failures",
        },
        "offline_fold_fit": fit,
        "online_decision": {
            **online,
            "prediction_count": len(predictions),
            "probe_used": False,
            "interpretation": "feature + dtype/shape eligibility + model; excludes chosen codec execution",
        },
    }


def recursively_find_key(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(recursively_find_key(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursively_find_key(child, key))
    return found


def source_scope(inputs: Mapping[str, Any]) -> dict[str, Any]:
    samples = inputs.get("samples", [])
    datasets = sorted({str(sample.get("dataset_id", "")) for sample in samples})
    scales = Counter(str(sample.get("scale", "unknown")) for sample in samples)
    dtypes = Counter(str(sample.get("dtype", "unknown")) for sample in samples)
    plan_path_raw = inputs.get("plan_columns_path")
    plan_path = Path(plan_path_raw) if isinstance(plan_path_raw, str) else None
    plan_verified = False
    plan_value_columns: int | None = None
    plan_datasets: int | None = None
    if plan_path is not None and plan_path.is_file():
        plan = load_json(plan_path)
        counts = plan.get("counts", {}) if isinstance(plan, dict) else {}
        plan_value_columns = counts.get("value_columns")
        plan_datasets = counts.get("datasets")
        declared_sha = inputs.get("plan_columns_sha256")
        plan_verified = isinstance(declared_sha, str) and sha256_file(plan_path) == declared_sha
    return {
        "dataset_count": len(datasets),
        "dataset_ids": datasets,
        "reference_value_column_count": REFERENCE_VALUE_COLUMNS,
        "reference_plan_value_column_count_observed": plan_value_columns,
        "reference_plan_dataset_count_observed": plan_datasets,
        "reference_plan_path_accessible_and_sha256_verified": plan_verified,
        "representative_sample_count": len(samples),
        "representative_series_selection": inputs.get("selection_scope"),
        "scale_counts": dict(sorted(scales.items())),
        "dtype_counts": dict(sorted(dtypes.items())),
        "max_values": inputs.get("max_values"),
        "minimum_values_per_window": inputs.get("minimum_values_per_window"),
        "statement": "135 datasets / 3519 value columns define the frozen reference; this campaign measures selected representative series and is not all 3519 columns",
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(summary: Mapping[str, Any], candidates: Sequence[dict[str, Any]]) -> str:
    scope = summary["scope"]
    status = summary["case_status"]
    audit = summary["audit"]
    env = summary["host_context"].get("env_spec")
    pre_state = summary["host_context"].get("pre_state")
    host = env.get("host") if isinstance(env, dict) else "unknown"
    worker_count = env.get("worker_count") if isinstance(env, dict) else "unknown"
    loadavg = pre_state.get("loadavg") if isinstance(pre_state, dict) else "unknown"
    compression_only = summary["compression_only"]
    lines = [
        "# Operator Recommend 135 v1 报告",
        "",
        "## 范围与口径",
        "",
        f"冻结参考范围为 **{scope['dataset_count']} 个数据集、{scope['reference_value_column_count']} 个 value columns**。本轮输入只选择每个 dataset×dtype 的最长同列同实体代表序列，并取不重叠 head/tail；因此结果是代表序列评估，**不是 3,519 列全量测量**。代表 sample 共 {scope['representative_sample_count']} 个，scale 分布为 `{json.dumps(scope['scale_counts'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        f"输入计划文件现场 SHA-256 验证：`{scope['reference_plan_path_accessible_and_sha256_verified']}`；计划中观察到的 dataset/value-column 数为 `{scope['reference_plan_dataset_count_observed']}` / `{scope['reference_plan_value_column_count_observed']}`。所有压缩率都以完整 NGF1 frame 为分母：`numeric_bytes/full_frame_bytes` 只计 packed values；`(values_bytes+missing_bitmap_bytes)/full_frame_bytes` 计 values+mask。数值越大表示压缩越好。",
        "",
        "## 候选池与有效性",
        "",
        f"catalog 共 {summary['candidate_scope']['candidate_count']} 个候选，五类 kind 的数量如下。raw 是对照，单独列示，不计作学习式方案的能力声明。",
        "",
        markdown_table(["kind", "count", "candidate_ids"], [[kind, summary['candidate_scope']['kind_counts'].get(kind, 0), ", ".join(c['candidate_id'] for c in candidates if c.get('kind') == kind)] for kind in KINDS]),
        "",
        markdown_table(["状态", "数量", "占比"], [[key, status[key], fmt(status[key] / status['total'], 6) if status['total'] else "NA"] for key in ("strict", "bypass", "domain", "fail")]),
        "",
        "strict 表示 frame-only 独立 decode 后逐位验证通过；bypass、domain rejection 与 failure 均不进入 oracle 排名。domain rejection 是冻结 dtype/shape 规则的结果，不是试压筛选。",
        "",
        markdown_table(
            ["层", "strict", "bypass", "domain", "fail", "total"],
            [[layer, *[summary["case_status_by_layer"][layer][key] for key in ("strict", "bypass", "domain", "fail", "total")]] for layer in ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096")],
        ),
        "",
        "### 完整 frame 压缩率",
        "",
        "下表对每个候选分别给出 sample 等权与 dataset 等权均值。两项都使用完整 frame；raw 仍作为对照单列，不并入能力声明。",
        "",
        markdown_table(
            ["candidate", "kind", "strict samples", "datasets", "numeric/frame sample", "numeric/frame dataset", "(values+mask)/frame sample", "(values+mask)/frame dataset"],
            [[
                candidate["candidate_id"], candidate["kind"],
                summary["compression_ratios"]["all"][candidate["candidate_id"]]["strict_sample_count"],
                summary["compression_ratios"]["all"][candidate["candidate_id"]]["dataset_count"],
                fmt(summary["compression_ratios"]["all"][candidate["candidate_id"]]["numeric_bytes_per_full_frame_byte"]["sample_equal"]),
                fmt(summary["compression_ratios"]["all"][candidate["candidate_id"]]["numeric_bytes_per_full_frame_byte"]["dataset_equal"]),
                fmt(summary["compression_ratios"]["all"][candidate["candidate_id"]]["values_plus_mask_bytes_per_full_frame_byte"]["sample_equal"]),
                fmt(summary["compression_ratios"]["all"][candidate["candidate_id"]]["values_plus_mask_bytes_per_full_frame_byte"]["dataset_equal"]),
            ] for candidate in candidates],
        ),
        "",
        "## 主分析：compression_only（去除 raw 对照）",
        "",
        "主结论采用 compression_only：对同一冻结 leave-source-group-out 模型的 held-out ranking 按原顺序静态过滤 `candidate_id == 'raw'`，没有重训、调参或在线 probe，也没有读取 held-out metric 来选择推荐 ID。held-out metric 只在 ID 冻结后用于重算 nonraw oracle 和评分。Top-1 是线上静态选择；Top-3/5 仍是离线 set-best 质量，不能称为线上收益。",
        "",
    ]
    nonraw_metric_rows: list[list[Any]] = []
    for target in TARGETS:
        for k in TOPKS:
            for clock in ("cpu", "wall"):
                item = compression_only["metrics"]["all"]["targets"][target][f"top{k}"][clock]
                nonraw_metric_rows.append([
                    target, f"Top-{k}", clock,
                    fmt(item["sample_equal_oracle_gap_percent"]),
                    fmt(item["dataset_equal_oracle_gap_percent"]),
                    fmt(item["sample_equal_tie_aware_hit_rate"]),
                    fmt(item["dataset_equal_tie_aware_hit_rate"]),
                    item["sample_count"], item["dataset_count"],
                ])
    lines.extend([
        markdown_table(["目标", "K", "时钟", "sample等权 gap%", "dataset等权 gap%", "sample等权命中", "dataset等权命中", "samples", "datasets"], nonraw_metric_rows),
        "",
        "### compression_only 分层 Top-1",
        "",
    ])
    nonraw_layer_rows: list[list[Any]] = []
    for layer, layer_value in compression_only["metrics"].items():
        for target in TARGETS:
            cpu = layer_value["targets"][target]["top1"]["cpu"]
            wall = layer_value["targets"][target]["top1"]["wall"]
            nonraw_layer_rows.append([layer, target, layer_value["prediction_count"], fmt(cpu["sample_equal_oracle_gap_percent"]), fmt(cpu["dataset_equal_oracle_gap_percent"]), fmt(wall["sample_equal_oracle_gap_percent"]), fmt(wall["dataset_equal_oracle_gap_percent"])])
    lines.extend([
        markdown_table(["层", "目标", "predictions", "CPU sample gap%", "CPU dataset gap%", "wall sample gap%", "wall dataset gap%"], nonraw_layer_rows),
        "",
        "### raw 在原冻结 ranking 中的频率",
        "",
    ])
    raw_frequency_rows: list[list[Any]] = []
    for layer in ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096"):
        for target in TARGETS:
            item = compression_only["raw_selection_frequency"][layer][target]
            raw_frequency_rows.append([
                layer, target,
                fmt(item["top1"]["sample_equal_rate"]), fmt(item["top1"]["dataset_equal_rate"]),
                fmt(item["top3"]["sample_equal_rate"]), fmt(item["top5"]["sample_equal_rate"]),
            ])
    lines.extend([
        markdown_table(["层", "目标", "Top-1 raw sample频率", "Top-1 raw dataset频率", "Top-3含raw sample频率", "Top-5含raw sample频率"], raw_frequency_rows),
        "",
        "### compression_only 固定 baseline 与 encode 端到端成本",
        "",
        "fixed algorithm 仍严格要求 `kind == algorithm`；fixed nonraw recipe 从各 fold 的 training rows 在 23 个 nonraw 候选中选择，held-out labels 不参与选择。收益为 `baseline/selected - 1`；负值表示静态 nonraw 推荐更差。",
        "",
    ])
    nonraw_baseline_rows: list[list[Any]] = []
    for target in TARGETS:
        for baseline in ("fixed_algorithm", "fixed_nonraw_recipe"):
            for clock in ("cpu", "wall"):
                item = compression_only["baselines"][target][baseline][clock]
                nonraw_baseline_rows.append([target, baseline, clock, fmt(item["sample_equal_benefit_percent"]), fmt(item["dataset_equal_benefit_percent"]), item["wins"], item["ties"], item["losses"]])
    lines.extend([
        markdown_table(["目标", "baseline", "时钟", "sample等权收益%", "dataset等权收益%", "win", "tie", "loss"], nonraw_baseline_rows),
        "",
    ])
    nonraw_e2e_rows: list[list[Any]] = []
    for baseline in ("fixed_algorithm", "fixed_nonraw_recipe"):
        for clock in ("cpu", "wall"):
            item = compression_only["baselines"]["encode_end_to_end_with_feature_eligibility_model"][baseline][clock]
            nonraw_e2e_rows.append([baseline, clock, fmt(item["sample_equal_benefit_percent"]), fmt(item["dataset_equal_benefit_percent"]), item["wins"], item["ties"], item["losses"]])
    nonraw_e2e_layer_rows: list[list[Any]] = []
    for layer in ("all", "u64", "primary_all", "primary_ge4096", "primary_lt4096"):
        for baseline in ("fixed_algorithm", "fixed_nonraw_recipe"):
            for clock in ("cpu", "wall"):
                benefit = compression_only["baselines_by_layer"][layer]["encode_end_to_end_with_feature_eligibility_model"][baseline][clock]
                selected = compression_only["selected_total_encode_cost_by_layer"][layer][f"{clock}_ns"]
                nonraw_e2e_layer_rows.append([
                    layer, baseline, clock,
                    fmt(benefit["sample_equal_benefit_percent"]),
                    fmt(benefit["dataset_equal_benefit_percent"]),
                    fmt(selected["sample_equal"]), fmt(selected["dataset_equal"]),
                    selected["sample_count"],
                ])
    replay = compression_only["compression_replay_validation"]
    replay_text = (
        f"COMPRESSION_REPLAY 严格绑定通过：folds={replay['fold_count']}，matched={replay['matched_count']}；每 sample 的完整 predict 外层计时重复 5 次取中位数，静态 filter 外层计时 1 次；全部 ranking 条目及 predicted score 与冻结过滤结果精确一致。"
        if isinstance(replay, dict)
        else "未提供 COMPRESSION_REPLAY；encode 总成本保守沿用原 24 项模型成本，静态过滤开销未测，不能称为精确成本或上界。"
    )
    decision = compression_only["online_decision_cost"]
    lines.extend([
        markdown_table(["encode端到端 baseline", "时钟", "sample等权收益%", "dataset等权收益%", "win", "tie", "loss"], nonraw_e2e_rows),
        "",
        markdown_table(["层", "baseline", "时钟", "sample等权收益%", "dataset等权收益%", "selected总encode sample均值ns", "selected总encode dataset均值ns", "samples"], nonraw_e2e_layer_rows),
        "",
        replay_text,
        "",
        f"compression_only decision 总成本：CPU `{fmt_int(decision['online_decision_cpu_ns'])}` ns，wall `{fmt_int(decision['online_decision_wall_ns'])}` ns；其中 filter CPU/wall 为 `{fmt_int(decision['filter_cpu_ns'])}` / `{fmt_int(decision['filter_wall_ns'])}` ns，model CPU/wall 为 `{fmt_int(decision['model_cpu_ns'])}` / `{fmt_int(decision['model_wall_ns'])}` ns。来源=`{decision['source']}`。encode 端到端比较按 feature+eligibility+filter+model+selected encode 计费。",
        "",
        "## 补充分析：raw-inclusive Leave-source-group-out",
        "",
        "下表保留原 24 候选（含 raw 对照）的结果，用于完整性和解释 raw 主导程度，不作为“压缩速度推荐强大”的主要证据。CPU 与 wall 独立统计；sample equal 与 dataset equal 同时给出。Top-1 是无 probe 的线上选择。Top-3/5 是离线集合中事后取最优的集合质量，只衡量候选覆盖，不能当作线上收益。",
        "",
    ])
    metric_rows: list[list[Any]] = []
    all_metrics = summary["evaluation_metrics"]["all"]["targets"]
    for target in TARGETS:
        for k in TOPKS:
            for clock in ("cpu", "wall"):
                item = all_metrics[target][f"top{k}"][clock]
                metric_rows.append([
                    target, f"Top-{k}", clock,
                    fmt(item["sample_equal_oracle_gap_percent"]),
                    fmt(item["dataset_equal_oracle_gap_percent"]),
                    fmt(item["sample_equal_tie_aware_hit_rate"]),
                    fmt(item["dataset_equal_tie_aware_hit_rate"]),
                    item["sample_count"], item["dataset_count"],
                ])
    lines.extend([
        markdown_table(["目标", "K", "时钟", "sample等权 gap%", "dataset等权 gap%", "sample等权命中", "dataset等权命中", "samples", "datasets"], metric_rows),
        "",
        "### 64 与 primary 层",
        "",
    ])
    layer_rows: list[list[Any]] = []
    for layer, layer_value in summary["evaluation_metrics"].items():
        for target in TARGETS:
            cpu = layer_value["targets"][target]["top1"]["cpu"]
            wall = layer_value["targets"][target]["top1"]["wall"]
            layer_rows.append([layer, target, layer_value["prediction_count"], fmt(cpu["sample_equal_oracle_gap_percent"]), fmt(cpu["dataset_equal_oracle_gap_percent"]), fmt(wall["sample_equal_oracle_gap_percent"]), fmt(wall["dataset_equal_oracle_gap_percent"])])
    lines.extend([
        markdown_table(["层", "目标", "predictions", "CPU sample gap%", "CPU dataset gap%", "wall sample gap%", "wall dataset gap%"], layer_rows),
        "",
        "`u64` 是 64 个有效值层；`primary_ge4096` 是 primary 且至少 4,096 个有效值；`primary_lt4096` 单独保留，避免把短 primary 混入长序列结论。",
        "",
        "## 补充：raw-inclusive 固定 baseline 与线上 encode 成本",
        "",
        "fixed algorithm 只允许 catalog 中 `kind == algorithm` 的候选；fixed full recipe 从完整冻结候选池选择。收益定义为 `baseline_cost / selected_cost - 1`，正值表示推荐器更好，负值如实表示亏损。普通四目标比较不加入决策开销；encode 端到端表把 feature、eligibility、model 全部加入 Top-1 encode。",
        "",
    ])
    baseline_rows: list[list[Any]] = []
    for target in TARGETS:
        for baseline in ("fixed_algorithm", "fixed_full_recipe"):
            for clock in ("cpu", "wall"):
                item = summary["baselines"][target][baseline][clock]
                baseline_rows.append([target, baseline, clock, fmt(item["sample_equal_benefit_percent"]), fmt(item["dataset_equal_benefit_percent"]), item["wins"], item["ties"], item["losses"]])
    lines.extend([
        markdown_table(["目标", "baseline", "时钟", "sample等权收益%", "dataset等权收益%", "win", "tie", "loss"], baseline_rows),
        "",
    ])
    e2e_rows: list[list[Any]] = []
    for baseline in ("fixed_algorithm", "fixed_full_recipe"):
        for clock in ("cpu", "wall"):
            item = summary["baselines"]["encode_end_to_end_with_feature_eligibility_model"][baseline][clock]
            e2e_rows.append([baseline, clock, fmt(item["sample_equal_benefit_percent"]), fmt(item["dataset_equal_benefit_percent"]), item["wins"], item["ties"], item["losses"]])
    lines.extend([
        markdown_table(["encode端到端 baseline", "时钟", "sample等权收益%", "dataset等权收益%", "win", "tie", "loss"], e2e_rows),
        "",
        "## 成本与产物",
        "",
        f"离线全矩阵的严格 case 中位数等价和：encode CPU `{fmt_int(summary['costs']['offline_full_matrix_median_equivalent']['encode_cpu_ns'])}` ns，decode CPU `{fmt_int(summary['costs']['offline_full_matrix_median_equivalent']['decode_cpu_ns'])}` ns，encode wall `{fmt_int(summary['costs']['offline_full_matrix_median_equivalent']['encode_wall_ns'])}` ns，decode wall `{fmt_int(summary['costs']['offline_full_matrix_median_equivalent']['decode_wall_ns'])}` ns。该值是 per-case median 的求和，不含 warmup、重复轮次、进程启动、独立验证、fsync 和失败开销，因此不冒充 campaign 实际耗时。",
        "",
        f"线上 decision 总成本（{summary['costs']['online_decision']['prediction_count']} 次）：CPU `{fmt_int(summary['costs']['online_decision']['online_decision_cpu_ns'])}` ns，wall `{fmt_int(summary['costs']['online_decision']['online_decision_wall_ns'])}` ns；无在线 codec probe。",
        "",
        "`RECOMMENDATIONS.csv` 按 dataset、scale、四目标和 candidate kind 汇总 family 内预测最优方案，并附实际完整-frame CR 与 CPU/wall encode/decode。`DATASET_CANDIDATE_RESULTS.csv` 给出每个 dataset×scale×candidate 的 strict/bypass/domain/fail 和完整 frame 结果。",
        "",
        "## 限制与审计",
        "",
        f"AUDIT.json present=`{audit['path_present']}`；`fallback_used=true/false/non-boolean` 观察数为 `{audit['fallback_used_true_count']}/{audit['fallback_used_false_count']}/{audit['fallback_used_non_boolean_count']}`；`unified_internal_fallback_audit={audit['unified_internal_fallback_audit']}`。共享主机 `{host}`，worker_count=`{worker_count}`，运行前 loadavg=`{loadavg}`。",
        "",
    ])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "输入文件均在 `SUMMARY.json.sources` 中记录绝对路径与 SHA-256。指标无论好坏均按观察值报告。", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="new run root; outputs go to RUN_ROOT/report")
    parser.add_argument("--inputs")
    parser.add_argument("--groups")
    parser.add_argument("--campaign", help="campaign directory used as a detection root")
    parser.add_argument("--rows")
    parser.add_argument("--case-index")
    parser.add_argument("--evaluation")
    parser.add_argument("--compression-replay")
    parser.add_argument("--audit")
    parser.add_argument("--candidates")
    parser.add_argument("--env-spec")
    parser.add_argument("--pre-state")
    parser.add_argument("--force", action="store_true", help="atomically replace the four report outputs")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise ReportError(f"run root is not a directory: {run_root}")
    campaign_root = Path(args.campaign).expanduser().resolve() if args.campaign else run_root / "campaign"
    roots = [run_root]
    if campaign_root.is_dir() and campaign_root not in roots:
        roots.insert(0, campaign_root)

    paths = {
        "inputs": resolve_file(args.inputs, roots, ("INPUTS.json", "inputs/INPUTS.json"), "INPUTS.json"),
        "groups": resolve_file(args.groups, roots, ("GROUPS.json", "groups/GROUPS.json"), "GROUPS.json"),
        "rows": resolve_file(args.rows, roots, ("ROWS.json", "campaign/ROWS.json"), "ROWS.json"),
        "case_index": resolve_file(args.case_index, roots, ("CASE_INDEX.json", "campaign/CASE_INDEX.json"), "CASE_INDEX.json"),
        "evaluation": resolve_file(args.evaluation, roots, ("EVALUATION.json", "evaluation/EVALUATION.json"), "EVALUATION.json"),
        "compression_replay": resolve_file(args.compression_replay, roots, ("COMPRESSION_REPLAY.json", "model/COMPRESSION_REPLAY.json"), "COMPRESSION_REPLAY.json", required=False),
        "audit": resolve_file(args.audit, roots, ("AUDIT.json", "audit/AUDIT.json"), "AUDIT.json", required=False),
        "candidates": resolve_file(args.candidates, roots, ("CANDIDATES.json", "campaign/CANDIDATES.json"), "CANDIDATES.json"),
        "env_spec": resolve_file(args.env_spec, roots, ("ENV_SPEC.json",), "ENV_SPEC.json", required=False),
        "pre_state": resolve_file(args.pre_state, roots, ("PRE_STATE.json",), "PRE_STATE.json", required=False),
    }

    inputs = load_json(paths["inputs"])
    groups = load_json(paths["groups"])
    rows_list = as_array(load_json(paths["rows"]), "rows", "ROWS.json")
    case_index = as_array(load_json(paths["case_index"]), "cases", "CASE_INDEX.json")
    evaluation = load_json(paths["evaluation"])
    compression_replay = load_json(paths["compression_replay"]) if paths["compression_replay"] else None
    catalog = load_json(paths["candidates"])
    audit = load_json(paths["audit"]) if paths["audit"] else None
    env_spec = load_json(paths["env_spec"]) if paths["env_spec"] else None
    pre_state = load_json(paths["pre_state"]) if paths["pre_state"] else None

    if not isinstance(inputs, dict) or inputs.get("schema") != "operator-recommend135-inputs-v1":
        raise ReportError("unsupported INPUTS.json schema")
    if not isinstance(groups, dict) or not isinstance(groups.get("dataset_to_group"), dict):
        raise ReportError("GROUPS.json has no dataset_to_group object")
    if not isinstance(evaluation, dict) or evaluation.get("schema") != "operator-recommend135-evaluation-v1":
        raise ReportError("unsupported EVALUATION.json schema")
    candidates = catalog.get("candidates") if isinstance(catalog, dict) else None
    if not isinstance(candidates, list) or any(not isinstance(candidate, dict) for candidate in candidates):
        raise ReportError("CANDIDATES.json has no candidates array")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if len(candidates) != 24 or len(set(candidate_ids)) != 24 or any(not isinstance(value, str) for value in candidate_ids):
        raise ReportError(f"frozen catalog must contain 24 unique candidate IDs, observed {len(candidates)}")
    candidate_kind = {str(candidate["candidate_id"]): str(candidate.get("kind", "unknown")) for candidate in candidates}
    unexpected_kinds = sorted(set(candidate_kind.values()) - set(KINDS))
    if unexpected_kinds:
        raise ReportError(f"catalog contains unexpected kind values: {unexpected_kinds}")
    raw_ids = sorted(candidate_id for candidate_id, kind in candidate_kind.items() if kind == "raw")
    if raw_ids != ["raw"]:
        raise ReportError(f"compression_only contract requires exactly one raw control with candidate_id='raw', observed {raw_ids}")

    samples_list = as_array(inputs, "samples", "INPUTS.json.samples")
    samples = {str(sample.get("sample_id")): sample for sample in samples_list}
    if len(samples) != len(samples_list):
        raise ReportError("INPUTS.json contains duplicate sample_id")
    input_datasets = {str(sample.get("dataset_id")) for sample in samples_list}
    if len(input_datasets) != EXPECTED_DATASETS:
        raise ReportError(f"report contract requires exactly 135 input datasets, observed {len(input_datasets)}")
    if set(groups["dataset_to_group"]) != input_datasets:
        missing = sorted(input_datasets - set(groups["dataset_to_group"]))
        extra = sorted(set(groups["dataset_to_group"]) - input_datasets)
        raise ReportError(f"GROUPS/input dataset mismatch; missing={missing}, extra={extra}")

    rows = {str(row.get("sample_id")): row for row in rows_list}
    if len(rows) != len(rows_list):
        raise ReportError("ROWS.json contains duplicate sample_id")
    predictions = flatten_predictions(evaluation)
    prediction_ids = {str(item.get("sample_id")) for item in predictions}
    if prediction_ids != set(rows):
        raise ReportError(f"evaluation/ROWS sample mismatch: evaluation={len(prediction_ids)}, rows={len(rows)}")
    eval_scope = evaluation.get("candidate_scope", {}).get("candidate_ids")
    if not isinstance(eval_scope, list) or set(eval_scope) != set(candidate_ids):
        raise ReportError("evaluation candidate scope does not equal the 24-candidate catalog")
    replay_by_sample: dict[str, dict[str, Any]] | None = None
    replay_validation: dict[str, Any] | None = None
    if compression_replay is not None:
        if not isinstance(compression_replay, dict):
            raise ReportError("COMPRESSION_REPLAY.json must be an object")
        replay_by_sample, replay_validation = validate_compression_replay(
            compression_replay,
            evaluation,
            predictions,
            sha256_file(paths["rows"]),
            sha256_file(paths["evaluation"]),
        )

    status_counts = Counter(status_class(str(item.get("status", "failed"))) for item in case_index)
    status_summary = {key: status_counts[key] for key in ("strict", "bypass", "domain", "fail")}
    status_summary["total"] = len(case_index)
    layered_status = case_status_by_layer(case_index, samples, rows)
    kind_counts = Counter(candidate_kind.values())
    evaluation_metrics = score_metrics(predictions, samples, rows)
    baselines = baseline_metrics(predictions, rows, candidate_kind)
    compression_only = compression_only_analysis(
        evaluation, predictions, rows, samples, candidate_kind,
        replay_by_sample, replay_validation,
    )
    dataset_results = aggregate_dataset_candidates(rows_list, samples, candidate_kind)
    compression_ratios = compression_ratio_metrics(rows_list, samples, candidate_kind)
    recommendations = recommendation_rows(predictions, rows, samples, candidate_kind)
    costs = decision_and_offline_costs(predictions, rows_list, evaluation)

    unresolved = groups.get("unresolved_provenance", [])
    if not isinstance(unresolved, list):
        unresolved = [unresolved]
    unified_values = recursively_find_key(audit, "unified_internal_fallback_audit") if audit is not None else []
    fallback_values = recursively_find_key(audit, "fallback_used") if audit is not None else []
    unified_audit = all(value is True for value in unified_values) if unified_values else None
    limitations = [
        f"source-group provenance 仍有 {len(unresolved)} 条 unresolved 记录；leave-source-group-out 只能按当前保守合并表约束已知来源泄漏，不能保证未知别名完全消失。",
        "campaign 在共享主机上运行；CPU affinity、单 worker 和 CPU/wall 双时钟不能消除同机负载、频率和缓存干扰。",
        "native codec 内部 literal/raw 路径没有统一的跨库 fallback 计数，因此 unified_internal_fallback_audit=false/unknown；这不等同于 harness 新增旁路。",
        "离线全矩阵成本是 per-case median 等价和，不是端到端 campaign 实际资源账单。",
        "Top-3/5 需要事后查看集合内真值，仅作离线覆盖指标；线上无 probe 时只执行 Top-1。",
        "135 数据集上的代表序列不能替代 3,519 value columns 的逐列全量评估。",
        "raw-inclusive 的 encode/decode/balanced 命中可能由 raw 对照主导；压缩推荐结论以 compression_only 静态过滤分析为主。",
    ]
    if replay_by_sample is None:
        limitations.append("compression_only 未提供严格绑定的 5 次 COMPRESSION_REPLAY；沿用原 24 项模型成本，额外静态过滤成本未测，因此该成本不是精确值也不能宣称为上界。")
    else:
        limitations.append("compression_only 在线成本采用严格绑定的 COMPRESSION_REPLAY：完整 predict 为 5 次外层计时中位数，静态 filter 为 1 次外层计时；输入 kernel 与原 campaign 测量保持不变。")
    if paths["audit"] is None:
        limitations.append("未检测到 AUDIT.json；strict/domain/fail 来自 CASE_INDEX，fallback 审计字段记为 unknown。")

    source_files = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
        if path is not None
    }
    summary = {
        "schema": "operator-recommend135-report-summary-v1",
        "scope": source_scope(inputs),
        "candidate_scope": {
            "candidate_count": len(candidates),
            "candidate_ids": candidate_ids,
            "kind_counts": {kind: kind_counts[kind] for kind in KINDS},
            "raw_control_candidate_ids": sorted(candidate_id for candidate_id, kind in candidate_kind.items() if kind == "raw"),
            "non_raw_candidate_count": sum(kind != "raw" for kind in candidate_kind.values()),
        },
        "case_status": status_summary,
        "case_status_by_layer": layered_status,
        "evaluation_protocol": groups.get("protocol", "leave-source-group-out"),
        "evaluation_metrics": evaluation_metrics,
        "compression_only": compression_only,
        "compression_ratios": compression_ratios,
        "baselines": baselines,
        "costs": costs,
        "compression_ratio_contract": {
            "frame": "complete NGF1 frame including metadata, values root and missing root",
            "numeric_bytes_per_full_frame_byte": "packed valid value bytes / complete frame bytes; higher is better",
            "values_plus_mask_bytes_per_full_frame_byte": "(packed valid value bytes + missing bitmap bytes) / complete frame bytes; higher is better",
        },
        "source_groups": {
            "group_count": len(groups.get("groups", [])) if isinstance(groups.get("groups"), list) else None,
            "unresolved_provenance_count": len(unresolved),
            "unresolved_provenance": unresolved,
        },
        "audit": {
            "path_present": paths["audit"] is not None,
            "unified_internal_fallback_audit": unified_audit,
            "fallback_used_true_count": sum(value is True for value in fallback_values),
            "fallback_used_false_count": sum(value is False for value in fallback_values),
            "fallback_used_non_boolean_count": sum(not isinstance(value, bool) for value in fallback_values),
            "native_internal_fallback_limit": "native codec internal literal/raw modes are not uniformly counted",
        },
        "host_context": {
            "env_spec": env_spec,
            "pre_state": pre_state,
            "shared_host_limitation": True,
        },
        "csv_rows": {
            "recommendations": len(recommendations),
            "dataset_candidate_results": len(dataset_results),
        },
        "limitations": limitations,
        "sources": source_files,
    }

    report_dir = run_root / "report"
    existing = [report_dir / name for name in OUTPUT_NAMES if (report_dir / name).exists()]
    if existing and not args.force:
        raise ReportError("report outputs already exist; use a new run root or --force: " + ", ".join(map(str, existing)))
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_fields = list(dataset_results[0]) if dataset_results else [
        "dataset_id", "group_id", "scale", "candidate_id", "kind", "raw_control",
    ]
    recommendation_fields = list(recommendations[0]) if recommendations else [
        "dataset_id", "group_id", "scale", "target", "candidate_family_kind", "raw_control",
    ]
    atomic_csv(report_dir / "DATASET_CANDIDATE_RESULTS.csv", dataset_fields, dataset_results)
    atomic_csv(report_dir / "RECOMMENDATIONS.csv", recommendation_fields, recommendations)
    atomic_json(report_dir / "SUMMARY.json", summary)
    atomic_text(report_dir / "REPORT.md", build_report(summary, candidates))
    print(json.dumps({"status": "completed", "report_dir": str(report_dir), "outputs": list(OUTPUT_NAMES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReportError as exc:
        print(f"report_operator_recommend135_v1.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
