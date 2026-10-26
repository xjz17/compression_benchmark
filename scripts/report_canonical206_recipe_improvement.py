#!/usr/bin/env python3
"""Read-only experiment audit and Chinese reporting; never executes a codec.

Default output is RUN/analysis_final, isolated from the live runner's REPORT.
Without --allow-partial, all 20 columns, both new methods and all three seeds,
the old GRPO baseline, and the runner's successful COMPLETE marker are required.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis_outputs/grpo_recipe_search_20260906_v3"
METHODS = ("grpo_recipe", "frozen_recipe")
LABELS = {"fixed": "固定算法", "old_grpo": "原 GRPO", "grpo_recipe": "新 GRPO",
          "frozen_recipe": "新 frozen（不学习）"}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法完整读取 {path}: {exc}") from exc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["dataset"]), int(row["numeric_index"])


def mark(left: int, right: int) -> str:
    return "win" if left < right else "tie" if left == right else "loss"


def finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"非法非负数值 {name}: {value}")
    return number


def verify_row(row: dict[str, Any], column: dict[str, Any], source: Path,
               frame_root: Path, registry: dict[int, str]) -> None:
    if (row["column"] != column["name"] or row["input_sha256"] != column["input_sha256"]
            or int(row["original_bytes"]) != int(column["raw_bytes"])):
        raise ValueError(f"输入不匹配: {source}")
    size = row["compressed_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"非法完整文件字节数: {source}")
    expected_ratio = int(column["raw_bytes"]) / size
    if not math.isclose(float(row["compression_ratio"]), expected_ratio, rel_tol=1e-12):
        raise ValueError(f"压缩比不是 raw / complete bytes: {source}")
    if row.get("lossless") is not True:
        raise ValueError(f"缺少成功的上游无损校验记录: {source}")
    ids = row["selected_ids"]
    if row["selected_pipeline"] != [registry[int(action)] for action in ids]:
        raise ValueError(f"算子 ID 与名字不一致: {source}")
    for field in ("compress_sec", "compose_sec", "fixed_algorithm_selection_sec"):
        if field in row:
            finite(row[field], f"{source.name}:{field}")
    frame = (frame_root / row["frame_path"]).resolve()
    if not frame.is_relative_to(frame_root.resolve()):
        raise ValueError(f"完整文件路径越界: {source}")
    if (not frame.is_file() or frame.stat().st_size != size
            or sha256(frame) != row["frame_sha256"]):
        raise ValueError(f"完整文件大小 / SHA256 不匹配: {source}")


def normalized(row: dict[str, Any], source: Path, fixed: bool = False) -> dict[str, Any]:
    result = {name: row.get(name) for name in (
        "seed", "compressed_bytes", "compression_ratio", "selected_ids", "selected_pipeline",
        "compress_sec", "compose_sec", "input_sha256", "frame_sha256", "lossless",
        "selected_option_ids", "selected_provenance", "selected_is_registered_fixed_recipe",
        "fixed_backend_completion_ids", "fixed_backend_completion_label", "search_train_sec",
        "seed_verification_sec", "completed_update_iterations", "weights_changed", "stop_reason",
        "search_cli", "all_cli", "completed_rollouts_lower_bound", "search_stats",
        "strict_new_composition_win", "byte_saving_vs_fixed")}
    result["source"] = str(source)
    if fixed:
        result.update(algorithm=row["algorithm"],
                      compose_sec=row["fixed_algorithm_selection_sec"],
                      compose_definition="prior_fixed_algorithm_selection_sec",
                      candidate_unique_chains=row["candidate_unique_chains"],
                      successful_chains=row["successful_chains"],
                      rejected_chains=row["rejected_chains"])
    else:
        result["compose_definition"] = ("prefix_search_plus_full_rerank; excludes prior training"
                                        if row["method"] == "grpo"
                                        else "fixed_winner_verification_plus_online_search_training")
    return result


def best_of(rows: list[dict[str, Any]], expected_seeds: list[int]) -> dict[str, Any] | None:
    if not rows:
        return None
    winner = min(rows, key=lambda row: (row["compressed_bytes"], row["seed"]))
    result = dict(winner)
    result.update(seeds_observed=sorted(row["seed"] for row in rows),
                  seeds_expected=expected_seeds,
                  all_seeds_available=len(rows) == len(expected_seeds),
                  best_of_n=len(rows),
                  all_seed_compose_sec=sum(row["compose_sec"] for row in rows),
                  all_seed_compress_sec=sum(row["compress_sec"] for row in rows))
    return result


def aggregate(rows: list[dict[str, Any]], raw_by_key: dict[tuple[str, int], int],
              expected: int) -> dict[str, Any]:
    raw = sum(raw_by_key[key(row)] for row in rows)
    size = sum(row["compressed_bytes"] for row in rows)
    return {"columns": len(rows), "expected_columns": expected, "raw_bytes": raw,
            "compressed_bytes": size, "compression_ratio": raw / size if size else None,
            "compress_sec": sum(row["compress_sec"] for row in rows),
            "compose_sec": sum(row["compose_sec"] for row in rows),
            "all_seed_compose_sec": sum(row.get("all_seed_compose_sec", row["compose_sec"]) for row in rows)}


def compare(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter(mark(left["compressed_bytes"], right["compressed_bytes"]) for left, right in pairs)
    left = sum(row[0]["compressed_bytes"] for row in pairs)
    right = sum(row[1]["compressed_bytes"] for row in pairs)
    return {"paired_columns": len(pairs), "wins": counts["win"], "ties": counts["tie"],
            "losses": counts["loss"], "left_bytes": left, "right_bytes": right,
            "bytes_saved": right - left, "byte_saving_fraction": 1 - left / right if right else None}


def collect(run: Path, parent_override: Path | None, allow_partial: bool) -> dict[str, Any]:
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    parent = (parent_override or Path(manifest["parent"])).resolve()
    parent_manifest_path = parent / "manifest.json"
    if sha256(parent_manifest_path) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent manifest SHA256 已改变")
    parent_manifest = read_json(parent_manifest_path)
    registry_path = parent / "action_registry.json"
    if sha256(registry_path) != manifest["registry_sha256"]:
        raise ValueError("206 算子注册表 SHA256 已改变")
    registry = {int(row["policy_action_id"]): row["name"]
                for row in read_json(registry_path)["policy_actions"]}
    adopted_columns: dict[str, dict[str, Any]] = {}
    resumption: dict[str, Any] = {"resumed_from": manifest.get("resumed_from"),
                                  "adopted_completed_rows": int(manifest.get("adopted_completed_rows", 0))}
    if manifest.get("resumed_from"):
        prior = Path(manifest["resumed_from"]).resolve()
        prior_manifest_path = run / "prior_manifest.json"
        if sha256(prior_manifest_path) != manifest["prior_manifest_sha256"]:
            raise ValueError("续跑保存的 prior_manifest SHA256 不匹配")
        prior_manifest = read_json(prior_manifest_path)
        if prior_manifest["config"] != manifest["config"]:
            raise ValueError("继承结果与续跑的模型/预算 config 不同，不能无标记合并")
        if sha256(run / "option_catalog.json") != sha256(prior / "option_catalog.json"):
            raise ValueError("续跑 option catalog 与继承结果不一致")
        for source_name, digest in prior_manifest["source_hashes"].items():
            if sha256(run / "prior_code" / Path(source_name).name) != digest:
                raise ValueError(f"继承结果的旧代码快照改变: {source_name}")
        adopted_records = read_json(run / "ADOPTED_RESULTS.json")
        for record in adopted_records:
            relative = Path(record["relative_path"])
            target = (run / relative).resolve()
            origin = Path(record["origin"]).resolve()
            if not target.is_relative_to(run) or not origin.is_relative_to(prior):
                raise ValueError("ADOPTED_RESULTS 文件路径不在对应实验目录内")
            if sha256(target) != record["sha256"] or sha256(origin) != record["sha256"]:
                raise ValueError(f"继承文件或其来源 SHA256 改变: {relative}")
            if relative.parts[0] == "columns":
                if str(relative) in adopted_columns:
                    raise ValueError("重复继承结果条目")
                adopted_columns[str(relative)] = record
        all_prior_rows = {str(path.relative_to(prior)) for path in (prior / "columns").glob("*.json")}
        if (set(adopted_columns) != all_prior_rows
                or len(adopted_columns) != resumption["adopted_completed_rows"]):
            raise ValueError("继承行不是原运行所有已完成行的无筛选全集")
        resumption.update(prior_manifest_sha256=manifest["prior_manifest_sha256"],
                          adoption_policy=manifest.get("adoption_policy"),
                          source_change_reason=manifest.get("source_change_reason"),
                          adopted_file_count=len(adopted_records),
                          adopted_files_sha256_verified=True,
                          complete_prior_row_set_adopted=True,
                          prior_source_hashes=prior_manifest["source_hashes"],
                          current_source_hashes=manifest["source_hashes"])
    columns = {key(row): row for row in parent_manifest["columns"]}
    if len(columns) != 20:
        raise ValueError(f"此报告要求原两数据集共 20 列，实际 {len(columns)}")
    seeds = sorted(set(int(seed) for seed in manifest["config"]["seeds"]))
    old_seeds = sorted(set(int(seed) for seed in parent_manifest["config"]["seeds"]))
    if len(seeds) != 3 or len(old_seeds) != 3:
        raise ValueError("此报告的 best-of-3 要求新旧实验分别具有三个 seeds")
    expected = {(method, seed, *column_key) for method in METHODS for seed in seeds for column_key in columns}
    option_catalog = {int(option["option_id"]): option for option in read_json(run / "option_catalog.json")}
    new_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    fixed_rows: dict[tuple[str, int], dict[str, Any]] = {}
    old_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for column_key, column in columns.items():
        source = parent / "fixed_algorithm_baseline_v1/columns" / f"{column_key[0]}__{column_key[1]:02d}.json"
        row = read_json(source)
        verify_row(row, column, source, parent / "fixed_algorithm_baseline_v1", registry)
        fixed_rows[column_key] = normalized(row, source, fixed=True)
    for source in sorted((parent / "columns").glob("grpo__*.json")):
        row = read_json(source)
        row_key = (int(row["seed"]), *key(row))
        if row_key[0] not in old_seeds or key(row) not in columns or row["method"] != "grpo":
            raise ValueError(f"非预期原 GRPO 结果: {source}")
        if row_key in old_rows:
            raise ValueError(f"重复原 GRPO composite key: {row_key}")
        verify_row(row, columns[key(row)], source, parent, registry)
        old_rows[row_key] = normalized(row, source)
    for source in sorted((run / "columns").glob("*.json")):
        row = read_json(source)
        row_key = (row["method"], int(row["seed"]), *key(row))
        if row_key not in expected or row_key in new_rows:
            raise ValueError(f"非预期 / 重复新结果 key: {row_key}")
        verify_row(row, columns[key(row)], source, run, registry)
        fixed = fixed_rows[key(row)]
        if (row["fixed_bytes"] != fixed["compressed_bytes"]
                or row["fixed_ids"] != fixed["selected_ids"]
                or row["fixed_algorithm"] != fixed["algorithm"]):
            raise ValueError(f"新结果固定基线与 parent 不同: {source}")
        expanded = [action for option in row["selected_option_ids"] for action in option_catalog[option]["ids"]]
        if expanded != row["selected_ids"]:
            raise ValueError(f"option 展开与最终链不一致: {source}")
        strict = row["compressed_bytes"] < fixed["compressed_bytes"]
        if bool(row["strict_win_vs_fixed"]) != strict:
            raise ValueError(f"严格胜出标志不符合文件字节: {source}")
        if row["method"] == "frozen_recipe" and (row["weights_changed"] or row["completed_update_iterations"]):
            raise ValueError(f"frozen 对照发生策略更新: {source}")
        new_rows[row_key] = normalized(row, source)
        relative = str(source.relative_to(run))
        adopted = adopted_columns.get(relative)
        new_rows[row_key]["source_generation"] = "adopted_prior_run" if adopted else "current_run"
        new_rows[row_key]["origin_result_path"] = adopted["origin"] if adopted else str(source)
    missing = sorted(expected - new_rows.keys())
    missing_old = sorted({(seed, *column_key) for seed in old_seeds for column_key in columns} - old_rows.keys())
    complete_path = run / "COMPLETE.json"
    complete = read_json(complete_path) if complete_path.exists() else None
    marker_ok = (isinstance(complete, dict) and complete.get("status") == "PASS"
                 and complete.get("expected_rows") == len(expected)
                 and complete.get("actual_rows") == len(new_rows)
                 and complete.get("new_sources_unchanged") is True
                 and complete.get("frozen_after", {}).get("status") == "PASS")
    is_complete = not missing and not missing_old and marker_ok
    if not is_complete and not allow_partial:
        raise ValueError(f"拒绝写最终报告：新结果 {len(new_rows)}/{len(expected)}，"
                         f"原 GRPO 缺 {len(missing_old)}，COMPLETE 有效={marker_ok}。"
                         "如确需查看部分结果，显式传 --allow-partial。")
    for source_name, digest in manifest["source_hashes"].items():
        snapshot = run / "code" / Path(source_name).name
        if sha256(snapshot) != digest:
            raise ValueError(f"新实验冻结代码快照已改变: {snapshot}")
    per_column = []
    for column_key, column in sorted(columns.items()):
        fixed = fixed_rows[column_key]
        result = {"dataset": column_key[0], "numeric_index": column_key[1], "column": column["name"],
                  "raw_bytes": column["raw_bytes"], "input_sha256": column["input_sha256"], "fixed": fixed}
        old_pool = [old_rows[(seed, *column_key)] for seed in old_seeds if (seed, *column_key) in old_rows]
        result["old_grpo"] = best_of(old_pool, old_seeds)
        result["old_grpo_all_seeds"] = old_pool
        for method in METHODS:
            pool = [new_rows[(method, seed, *column_key)] for seed in seeds if (method, seed, *column_key) in new_rows]
            result[method] = best_of(pool, seeds)
            result[method + "_all_seeds"] = pool
            counts = Counter(mark(row["compressed_bytes"], fixed["compressed_bytes"]) for row in pool)
            result[method + "_stability"] = {
                "observed_seeds": len(pool), "expected_seeds": len(seeds),
                "wins_vs_fixed": counts["win"], "ties_vs_fixed": counts["tie"], "losses_vs_fixed": counts["loss"],
                "strict_all_seeds_vs_fixed": len(pool) == len(seeds) and counts["win"] == len(seeds),
                "updated_seeds": sum(bool(row["weights_changed"]) for row in pool),
                "provenance_counts": dict(Counter(str(row["selected_provenance"]) for row in pool)),
            }
        paired = [(new_rows[(METHODS[0], seed, *column_key)], new_rows[(METHODS[1], seed, *column_key)])
                  for seed in seeds if all((method, seed, *column_key) in new_rows for method in METHODS)]
        result["grpo_vs_frozen_paired_seeds"] = compare(paired)
        result["grpo_strict_all_seeds_vs_frozen"] = len(paired) == len(seeds) and all(
            left["compressed_bytes"] < right["compressed_bytes"] for left, right in paired)
        # Archive delivery is separate from the raw new-GRPO result. Prefer an
        # existing fixed/old artifact on equal bytes, never award a tied archive
        # result to the new learner. No frozen result enters this portfolio.
        choices = [("fixed", result["fixed"]), ("old_grpo", result["old_grpo"]),
                   ("grpo_recipe", result["grpo_recipe"])]
        choices = [(source, value) for source, value in choices if value is not None]
        priority = {"fixed": 0, "old_grpo": 1, "grpo_recipe": 2}
        selected_source, selected = min(choices, key=lambda item: (
            item[1]["compressed_bytes"], priority[item[0]], item[1]["seed"] or 0))
        result["portfolio"] = {
            **selected, "selected_source": selected_source,
            "is_learning_contribution_claim": False,
            "selection_definition": "min(fixed, old_grpo_best_of3, new_grpo_best_of3); excludes frozen",
            "candidate_sources_available": [source for source, _ in choices],
            "recorded_candidate_selection_cost_sec": sum(
                value.get("all_seed_compose_sec", value["compose_sec"]) for _, value in choices),
        }
        per_column.append(result)
    raw_by_key = {column_key: int(column["raw_bytes"]) for column_key, column in columns.items()}
    dataset_best, dataset_seeds = [], []
    for dataset in sorted({column_key[0] for column_key in columns}):
        dataset_columns = [row for row in per_column if row["dataset"] == dataset]
        dataset_keys = [key(row) for row in dataset_columns]
        for method in ("fixed", "old_grpo", *METHODS):
            selected = [{**row[method], "dataset": row["dataset"], "numeric_index": row["numeric_index"]}
                        for row in dataset_columns if row[method] is not None]
            dataset_best.append({"dataset": dataset, "method": method,
                                 "selection": "fixed" if method == "fixed" else "per_column_best_of_available_seeds; mixed_seed_upper_envelope",
                                 **aggregate(selected, raw_by_key, len(dataset_keys))})
        for seed in seeds:
            row = {"dataset": dataset, "seed": seed, "expected_columns": len(dataset_keys)}
            for method in METHODS:
                pool = [{**new_rows[(method, seed, *column_key)], "dataset": column_key[0], "numeric_index": column_key[1]}
                        for column_key in dataset_keys if (method, seed, *column_key) in new_rows]
                row[method] = aggregate(pool, raw_by_key, len(dataset_keys))
                row[method]["vs_fixed"] = compare([(value, fixed_rows[key(value)]) for value in pool])
                row[method]["search_cli_calls"] = sum(value["search_cli"]["calls"] for value in pool)
                row[method]["search_cli_timeouts"] = sum(value["search_cli"]["timeouts"] for value in pool)
                row[method]["search_unique_atomic_edges"] = sum(value["search_stats"]["search_unique_atomic_edges"] for value in pool)
                row[method]["updated_columns"] = sum(bool(value["weights_changed"]) for value in pool)
            pairs = [(new_rows[(METHODS[0], seed, *column_key)], new_rows[(METHODS[1], seed, *column_key)])
                     for column_key in dataset_keys if all((method, seed, *column_key) in new_rows for method in METHODS)]
            row["grpo_vs_frozen"] = compare(pairs)
            dataset_seeds.append(row)
    best_pairs = [(row[METHODS[0]], row[METHODS[1]]) for row in per_column
                  if all(row[method] is not None for method in METHODS)]
    best_vs_fixed = [(row[METHODS[0]], row["fixed"]) for row in per_column if row[METHODS[0]]]
    best_vs_old = [(row[METHODS[0]], row["old_grpo"]) for row in per_column if row[METHODS[0]] and row["old_grpo"]]
    paired_all = [(new_rows[(METHODS[0], seed, *column_key)], new_rows[(METHODS[1], seed, *column_key)])
                  for seed in seeds for column_key in columns if all((method, seed, *column_key) in new_rows for method in METHODS)]
    portfolio_datasets = []
    for dataset in sorted({row["dataset"] for row in per_column}):
        pool = [{**row["portfolio"], "dataset": row["dataset"], "numeric_index": row["numeric_index"]}
                for row in per_column if row["dataset"] == dataset]
        portfolio_datasets.append({"dataset": dataset, **aggregate(pool, raw_by_key, len(pool)),
                                   "selected_source_counts": dict(Counter(row["selected_source"] for row in pool)),
                                   "recorded_candidate_selection_cost_sec": sum(row["recorded_candidate_selection_cost_sec"] for row in pool)})
    return {
        "schema_version": 1, "status": "COMPLETE" if is_complete else "PARTIAL",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run": str(run), "parent": str(parent), "seeds": seeds, "old_seeds": old_seeds,
        "coverage": {"new_rows": len(new_rows), "expected_new_rows": len(expected),
                     "missing_new": missing, "missing_old_grpo": missing_old,
                     "complete_marker_valid": marker_ok, "complete_marker": complete},
        "audit": {"same_input_hashes": True, "ratio_recomputed_from_bytes": True,
                  "selected_frame_sizes_and_sha256": True, "selected_ids_names_and_options_match": True,
                  "upstream_lossless_flags": True, "codec_executed_by_reporter": False,
                  "manifest_sha256": sha256(manifest_path), "parent_manifest_sha256": sha256(parent_manifest_path),
                  "report_script_sha256": sha256(Path(__file__)), "registry_sha256": sha256(registry_path)},
        "definitions": {
            "compression_ratio": "sum(original typed numerical bytes) / sum(complete C206+UAR1 frame bytes)",
            "winner": "minimum compressed_bytes, ties broken by minimum seed; all chain/time fields from that same row",
            "fixed_scope": "previous registered full-recipe successful candidates only; not a global codec oracle; Camel excluded",
            "best_of_three": "per-column upper envelope across three completed seeds, not a same-seed run or stable win",
            "strict": "integer complete-file compressed bytes strictly smaller; never rounded ratios",
            "claim_boundary": "incumbent protection ensures non-regression only; cannot guarantee every column is strictly better",
            "non_rl_components": [manifest["prior"], manifest["completion"], manifest["baseline_cost"]],
            "training_objective": manifest["objective"], "experiment_scope": manifest["scope"]},
        "config": manifest["config"], "columns": per_column,
        "resumption": resumption,
        "training_coverage": {
            "grpo_rows": sum(row_key[0] == METHODS[0] for row_key in new_rows),
            "rows_with_weight_updates": sum(bool(row["weights_changed"]) for row_key, row in new_rows.items()
                                            if row_key[0] == METHODS[0]),
            "update_iteration_histogram": dict(Counter(str(row["completed_update_iterations"])
                for row_key, row in new_rows.items() if row_key[0] == METHODS[0])),
            "rows_without_weight_updates": [
                {"seed": row_key[1], "dataset": row_key[2], "numeric_index": row_key[3],
                 "column": columns[(row_key[2], row_key[3])]["name"], "stop_reason": row["stop_reason"]}
                for row_key, row in new_rows.items()
                if row_key[0] == METHODS[0] and not row["weights_changed"]],
        },
        "dataset_best_of_three": dataset_best, "dataset_seed_results": dataset_seeds,
        "delivery_portfolio": {
            "claim": "system archive non-regression only; not a new GRPO discovery or learning-attribution claim",
            "columns": [{"dataset": row["dataset"], "column": row["column"], "numeric_index": row["numeric_index"],
                         "raw_bytes": row["raw_bytes"], **row["portfolio"]} for row in per_column],
            "datasets": portfolio_datasets,
            "selected_source_counts": dict(Counter(row["portfolio"]["selected_source"] for row in per_column)),
            "vs_fixed": compare([(row["portfolio"], row["fixed"]) for row in per_column]),
            "vs_old_grpo": compare([(row["portfolio"], row["old_grpo"]) for row in per_column if row["old_grpo"]]),
        },
        "comparisons": {"new_grpo_best_vs_fixed": compare(best_vs_fixed),
                        "new_grpo_best_vs_old_grpo_best": compare(best_vs_old),
                        "new_grpo_best_vs_frozen_best": compare(best_pairs),
                        "grpo_vs_frozen_same_seed_column_pairs": compare(paired_all),
                        "grpo_stable_strict_columns_vs_fixed": sum(row["grpo_recipe_stability"]["strict_all_seeds_vs_fixed"] for row in per_column),
                        "frozen_stable_strict_columns_vs_fixed": sum(row["frozen_recipe_stability"]["strict_all_seeds_vs_fixed"] for row in per_column),
                        "grpo_stable_strict_columns_vs_frozen": sum(row["grpo_strict_all_seeds_vs_frozen"] for row in per_column)},
        "gain_attribution": {
            "strict_vs_both_fixed_and_old_grpo_columns": [
                {"dataset": row["dataset"], "column": row["column"],
                 "new_bytes": row[METHODS[0]]["compressed_bytes"],
                 "fixed_bytes": row["fixed"]["compressed_bytes"],
                 "old_grpo_bytes": row["old_grpo"]["compressed_bytes"]}
                for row in per_column if row[METHODS[0]] and row["old_grpo"]
                and row[METHODS[0]]["compressed_bytes"] < min(row["fixed"]["compressed_bytes"], row["old_grpo"]["compressed_bytes"])],
            "baseline_gain_from_zero_update_winning_rows": sum(
                max(0, row["fixed"]["compressed_bytes"] - row[METHODS[0]]["compressed_bytes"])
                for row in per_column if row[METHODS[0]] and not row[METHODS[0]]["weights_changed"]),
        },
    }


def ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2%}"


def wtl(row: dict[str, Any]) -> str:
    return f"{row['wins']}/{row['ties']}/{row['losses']}"


def chain(row: dict[str, Any] | None) -> str:
    return "—" if row is None else " → ".join(str(action) for action in row["selected_ids"])


def metric_cell(row: dict[str, Any] | None, fixed: bool = False) -> str:
    if row is None:
        return "待齐"
    count = "固定" if fixed else f"best-of-{row['best_of_n']}；seed {row['seed']}"
    return f"{row['compression_ratio']:.3f}× / {row['compressed_bytes']:,} B；{count}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(map(escape, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(escape, row)) + " |" for row in rows]]


def markdown_report(summary: dict[str, Any]) -> str:
    partial = summary["status"] != "COMPLETE"
    best_label = "已完成 seeds 的最好值（尚非完整 best-of-3）" if partial else "best-of-3"
    comparisons = summary["comparisons"]
    fixed = comparisons["new_grpo_best_vs_fixed"]
    control = comparisons["new_grpo_best_vs_frozen_best"]
    paired = comparisons["grpo_vs_frozen_same_seed_column_pairs"]
    title = "【部分结果，不能作为最终结论】" if partial else "完整结果"
    lines = [f"# GRPO 完整算子组合改进：{title}", "",
             f"生成时间：{summary['created_at']}。新结果 {summary['coverage']['new_rows']}/"
             f"{summary['coverage']['expected_new_rows']}；COMPLETE 校验：{summary['coverage']['complete_marker_valid']}。", "",
             "## 结论先说清楚", "",
             f"新 GRPO 的逐列 {best_label} 相对固定算法：胜/平/负 = **{wtl(fixed)}**；"
             f"相对同样按已完成 seeds 选优的 frozen 对照：**{wtl(control)}**，对应完整字节节省 {percent(control['byte_saving_fraction'])}。",
             f"真正按同列、同 seed 配对，新 GRPO 相对 frozen：**{wtl(paired)}**（{paired['paired_columns']} 个已齐配对）。"
             f"三个 seeds 全部严格胜固定算法的列：**{comparisons['grpo_stable_strict_columns_vs_fixed']}/20**；"
             f"三个 seeds 全部严格胜 frozen 的列：**{comparisons['grpo_stable_strict_columns_vs_frozen']}/20**。",
             f"实际策略更新覆盖：**{summary['training_coverage']['rows_with_weight_updates']}/"
             f"{summary['training_coverage']['grpo_rows']}** 个 GRPO 列×seed 运行发生了权重更新。"
             "若墙钟截止时连一组轨迹都未完成，该次只有采样搜索，不能声称其结果来自已完成的 GRPO 学习。",
             f"同时严格超过固定算法和原 GRPO 最好结果的列有 **"
             f"{len(summary['gain_attribution']['strict_vs_both_fixed_and_old_grpo_columns'])}** 个。"
             f"对固定算法节省的 {fixed['bytes_saved']:,} B 中，"
             f"{summary['gain_attribution']['baseline_gain_from_zero_update_winning_rows']:,} B 来自胜出 seed 未发生更新的采样运行。"
             "总字节改善尤其不能直接作为 RL 训练收益。具体旧结果恢复、新组合和不学习对照见下表。",
             "**不能保证每列严格胜出。** 保留已知固定胜者只能保证“不退步”；找到更小完整文件是实验事实，"
             "不能靠保底规则保证。best-of-3 只是三个 seeds 中挑出的最好结果，不是稳定胜利。", "",
             "## 这次究竟改了什么：简单说", "",
             "1. **少猜几次，但不换底层算法。** 把完整的小段算子链作为一次选择，实际执行仍展开成原 206 库的真实算子。"
             "例如 BUFF 的 3 步 `[150,151,152]` 接完整 Bzip2 的 5 步 `[165,166,167,74,181]`，"
             "合计 8 个原子算子，如今只需两段选择；旧版 7 步上限无法容纳这条新组合。旧 Bzip2 本身并非不可达，"
             "问题是旧 Beam29/GRPO/PPO 的搜索轨迹没有执行其完整 5 步（固定算法对照执行过），"
             "而默认补全偏向缺少最后 residual core 的短链。",
             "2. **让一组里相对好的探索路线得到正确鼓励。** 新 GRPO 只做一次组内奖励标准化，"
             "不再将不同长度路线的逐步奖励混在一起二次标准化；每条路线总权重相等。"
             "例如组内本来略好的路线，不应仅因别的好路线更长，就被第二次标准化翻成负奖励。"
             "这属于 GRPO 更新实现的修正。单纯把奖励换成“相对固定算法的比例”只是同组常数平移，本身不带来这一改进。",
             "3. **奖励这条路线实际找到过的最好完整文件。** 一条路线先得到 100 字节、追加后变成 120 字节，"
             "仍保留 100 字节的成绩；后续失败只加小惩罚。奖励不采用未封装中间数据，也不借用其他路线的最好成绩。"
             "每个有效小段后还试接一个事先指定的通用后端，所有测量计入预算；GRPO 和不学习对照共享这一补全规则。",
             "4. **新结果必须真的更小才替换保底。** 固定算法先独立复核，搜索没找到更小文件时保留它。"
             "这只能保证同一输入和封装下不退步，不能制造严格提升。若当前候选没有更短编码，或有限搜索没找到，结果就是持平。", "",
             "GRPO 真正学习的是：比较一组路线的完整文件大小，增加好路线的选择概率，让后续测量更多尝试有希望的组合。"
             "完整小段、固定后端补全和保底属于共享的搜索设计；只有对不学习对照的额外收益，才是学习是否有用的证据。"
             "本实现采用组相对优势和 PPO 式裁剪更新，未启用参考策略 KL 项（beta=0），不能视作完整复现某个大模型训练配方。"
             "GRPO 原始方法参考：[DeepSeekMath](https://arxiv.org/abs/2402.03300)。", "",
             "## 口径与来源", "",
             "- 压缩比 CR = 原始 typed 数值字节 / 完整 C206+UAR1 文件字节；越大越好。胜负按整数文件字节判定，不能用四舍五入后的 CR 判断。",
             "- 固定算法是在旧实验已注册、成功完成的完整固定链中选最小文件，不是全球最优压缩算法；Camel 按原协议排除，超时候选未当成已测成功。",
             "- 新 GRPO 与 frozen 都拿到同一固定胜者作为候选和保底；初始网络相同，固定胜者的 logit 都加 1。这些不是 GRPO 学习贡献。",
             "- 每成功执行一个 option，都会尝试预先选好的通用固定后端闭包。两方法共享规则并计入预算；标记 fixed_backend_completion 的胜者必须归因为“搜索候选 + 确定性闭包”，不能说全链由 GRPO 独立生成。",
             "- 新方法在当前完整列上在线学习；原 GRPO 使用前缀搜索和完整列重排。与原结果的改善混合了候选设计、测量范围和学习方法变化，不能只归因于 GRPO。",
             "- 表中编码时间和组合时间均来自该列胜出 seed 的同一条记录；best-of-3 的总探索开销另记为三个 seeds 的 compose_sec 之和。",
             "- 新 compose_sec = 固定胜者复核 + 在线搜索/训练；此前固定算法筛选成本单列，不能隐藏。原 compose_sec = 前缀搜索 + 全列重排，不含此前策略训练，两者不是同预算的训练速度对比。",
             f"- 本轮搜索单次子进程调用上限 {summary['config']['search_timeout']} 秒、每列搜索墙钟上限 {summary['config']['max_seconds']} 秒、最多 {summary['config']['max_edges']} 次新增原子尝试。短 timeout 会删失慢候选，超时不代表其压缩比不好或不能无损；胜者独立重放仍使用 30 秒调用上限。",
             "- 固定与新结果的完整文件大小、SHA256、输入哈希、算子 ID/名字均由此脚本复核；无损解码沿用上游独立复编码/解码 PASS 记录，后处理不运行 codec。", "",
             "## 数据集总体：逐列 best-of-3 的上界汇总", "",
             "这里对每列分别挑 seed，再以 Σraw / Σcompressed 汇总；不是平均各列 CR，也不是某一个 seed 的实际运行结果。"
              "部分结果必须按“已齐列数”阅读，不可拿不完整数据集的比值冒充全表结果。", ""]
    if summary["resumption"]["resumed_from"]:
        resumed = summary["resumption"]
        warning = (f"**续跑来源披露：** {resumed['adopted_completed_rows']} 个已完成结果从 "
                   f"`{resumed['resumed_from']}` 无筛选继承；其余结果使用当前代码。不能声称全部 120 行由同一新源码生成。"
                   "变更仅将候选完整帧解码超时/失败记录为拒绝，而非终止整轮；模型、奖励和预算 config 未变。"
                   "继承结果已有独立 cold encode/decode 校验，后处理再次核对全部继承文件及来源哈希、"
                   "旧代码快照和原已完成行全集。每个结果在 JSON 中保留 source_generation 与 origin_result_path。")
        insert = lines.index("## 口径与来源")
        lines[insert:insert] = [warning, ""]
    lines += table(["数据集", "方法", "已齐列", "Σ原始 B", "Σ压缩 B", "整体 CR", "胜者编码 s", "胜者组合 s", "所有 seeds 组合 s"], [
        [row["dataset"], LABELS[row["method"]], f"{row['columns']}/{row['expected_columns']}",
         f"{row['raw_bytes']:,}", f"{row['compressed_bytes']:,}", ratio(row["compression_ratio"]),
         f"{row['compress_sec']:.3f}", f"{row['compose_sec']:.3f}", f"{row['all_seed_compose_sec']:.3f}"]
        for row in summary["dataset_best_of_three"]])
    lines += ["", "## 每个 seed 的真实整体结果与胜平负", "",
              "“对固定”和“GRPO 对 frozen”的胜/平/负均按同列完整文件字节。CLI 次数与唯一原子尝试数量公开；相同上限不意味着实际调用数严格相等。", ""]
    seed_table = []
    for row in summary["dataset_seed_results"]:
        for method in METHODS:
            value = row[method]
            seed_table.append([row["dataset"], str(row["seed"]), LABELS[method],
                               f"{value['columns']}/{value['expected_columns']}",
                               ratio(value["compression_ratio"]), f"{value['compressed_bytes']:,}",
                               wtl(value["vs_fixed"]),
                               wtl(row["grpo_vs_frozen"]) if method == METHODS[0] else "—",
                               f"{value['compress_sec']:.3f}", f"{value['compose_sec']:.3f}",
                               str(value["search_cli_calls"]), str(value["search_cli_timeouts"]), str(value["search_unique_atomic_edges"]),
                               str(value["updated_columns"])])
    lines += table(["数据集", "seed", "方法", "列数", "Σraw/Σcompressed", "Σ压缩 B", "对固定 胜/平/负",
                    "GRPO 对 frozen 胜/平/负", "编码 s", "组合 s", "搜索 CLI", "CLI 超时", "原子尝试", "更新列数"], seed_table)
    for dataset in sorted({row["dataset"] for row in summary["columns"]}):
        column_rows = [row for row in summary["columns"] if row["dataset"] == dataset]
        lines += ["", f"## {dataset}：逐列压缩结果", "",
                  "每个方法的 CR、字节、seed、链、时间绑定同一条胜出记录；算子链按原 206 policy ID 展示，完整名字在后面的逐列明细和 JSON 中。", ""]
        lines += table(["数据列", "固定算法", "固定 CR / B", "原 GRPO CR / B / seed",
                        "新 GRPO CR / B / seed", "新 frozen CR / B / seed", "新 GRPO 对固定稳定性", "新 GRPO 对 frozen 稳定性"], [
            [row["column"], row["fixed"]["algorithm"], metric_cell(row["fixed"], fixed=True),
             metric_cell(row["old_grpo"]), metric_cell(row[METHODS[0]]), metric_cell(row[METHODS[1]]),
             f"{row['grpo_recipe_stability']['wins_vs_fixed']}/{row['grpo_recipe_stability']['observed_seeds']} 已测 seed 严格胜；3/3={row['grpo_recipe_stability']['strict_all_seeds_vs_fixed']}",
             f"{wtl(row['grpo_vs_frozen_paired_seeds'])}；3/3={row['grpo_strict_all_seeds_vs_frozen']}"] for row in column_rows])
        lines += ["", f"### {dataset}：同 seed 的链与耗时", ""]
        detail_rows = []
        for row in column_rows:
            for method in ("fixed", "old_grpo", *METHODS):
                value = row[method]
                if value is None:
                    continue
                detail_rows.append([row["column"], LABELS[method], str(value["seed"] or "—"), chain(value),
                                    f"{value['compress_sec']:.6f}", f"{value['compose_sec']:.6f}",
                                    f"{value.get('all_seed_compose_sec', value['compose_sec']):.6f}",
                                    str(value.get("selected_provenance") or ("固定" if method == "fixed" else "原搜索")),
                                    str(value.get("fixed_backend_completion_label") or "—"),
                                    str(value.get("source_generation") or "原实验")])
        lines += table(["数据列", "方法", "胜者 seed", "原 206 ID 完整链", "编码 s", "同次组合/固定筛选 s",
                        "全部 seeds 组合 s", "首次发现来源", "共享固定闭包后端", "生成代码来源"], detail_rows)
    lines += ["", "## 每列跨 seed 稳定性：不能用最好一次替代", ""]
    stable_rows = []
    for row in summary["columns"]:
        for method in METHODS:
            state = row[method + "_stability"]
            seed_values = "; ".join(f"{value['seed']}: {value['compressed_bytes']} B ({value['compression_ratio']:.3f}×)"
                                    for value in row[method + "_all_seeds"])
            stable_rows.append([row["dataset"], row["column"], LABELS[method], seed_values or "待齐",
                                f"{state['wins_vs_fixed']}/{state['ties_vs_fixed']}/{state['losses_vs_fixed']}",
                                str(state["strict_all_seeds_vs_fixed"]),
                                str(state["updated_seeds"]), json.dumps(state["provenance_counts"], ensure_ascii=False)])
    lines += table(["数据集", "列", "方法", "各 seed 文件字节 / CR", "对固定 胜/平/负", "3/3 严格胜固定", "权重更新 seed 数", "首次来源计数"], stable_rows)
    portfolio = summary["delivery_portfolio"]
    lines += ["", "## 独立交付 portfolio：系统保底，不是新 GRPO 成果", "",
              "最终归档取 min（固定胜者，原 GRPO 逐列 best-of-3，新 GRPO 逐列 best-of-3）。不使用 frozen 结果。"
              "相同字节优先保留固定/原结果，避免把既有结果算成新算法发现。这个后处理仅可声称系统不退步，不能算新 GRPO 学到的提升。",
              f"portfolio 相对固定算法胜/平/负：{wtl(portfolio['vs_fixed'])}；相对原 GRPO：{wtl(portfolio['vs_old_grpo'])}。"
              f"来源计数：{json.dumps(portfolio['selected_source_counts'], ensure_ascii=False)}。", ""]
    lines += table(["数据集", "整体 CR = Σraw/Σcompressed", "Σ压缩 B", "胜者编码 s", "归档来源", "已记录候选筛选成本 s"], [
        [row["dataset"], ratio(row["compression_ratio"]), f"{row['compressed_bytes']:,}",
         f"{row['compress_sec']:.3f}", json.dumps(row["selected_source_counts"], ensure_ascii=False),
         f"{row['recorded_candidate_selection_cost_sec']:.3f}"] for row in portfolio["datasets"]])
    lines += ["", "候选筛选成本包含已记录的固定筛选、原三个 seeds 组合、新三个 seeds 组合；不含原离线训练等未包含在源字段中的成本。"
              "下表“来源运行组合时间”是所选结果原记录的耗时，不是假造 portfolio 筛选运行时间。", ""]
    lines += table(["数据集", "数据列", "系统交付 CR", "完整 B", "selected_source", "seed", "206 ID 链", "编码 s", "来源运行组合/固定筛选 s"], [
        [row["dataset"], row["column"], ratio(row["compression_ratio"]), str(row["compressed_bytes"]),
         row["selected_source"], str(row["seed"] if row["seed"] is not None else "—"), chain(row),
         f"{row['compress_sec']:.6f}", f"{row['compose_sec']:.6f}"] for row in portfolio["columns"]])
    lines += ["", "## 逐列完整算子名与源文件", ""]
    for row in summary["columns"]:
        lines += [f"### {row['dataset']} / {row['column']}", ""]
        for method in ("fixed", "old_grpo", *METHODS):
            value = row[method]
            if value is None:
                continue
            lines += [f"- {LABELS[method]}：`{chain(value)}`；" + " → ".join(f"`{name}`" for name in value["selected_pipeline"])
                      + f"。源文件：[JSON]({Path(value['source']).as_uri()})。"]
        lines.append("")
    lines += ["## 审计与复核", "",
              f"原结果：[parent manifest]({Path(summary['parent']).joinpath('manifest.json').as_uri()})；"
              f"新实验：[manifest]({Path(summary['run']).joinpath('manifest.json').as_uri()})。",
              f"脚本 SHA256：`{summary['audit']['report_script_sha256']}`。", ""]
    if partial:
        lines += ["**该报告为显式 --allow-partial 生成的部分结果。缺失行清单见 final_summary.json，不能作为完整实验结论。**", ""]
    return "\n".join(lines)


def html_report(markdown: str, summary: dict[str, Any]) -> str:
    """Small stdlib renderer for this script's own headings/tables/lists only."""
    import re

    def inline(text: str) -> str:
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
        return escaped

    source = markdown.splitlines()
    output, index = [], 0
    while index < len(source):
        line = source[index]
        if line.startswith("| ") and index + 1 < len(source) and source[index + 1].startswith("| ---"):
            cells = lambda text: [inline(value.strip().replace("\\|", "|")) for value in re.split(r"(?<!\\)\|", text.strip("| "))]
            output.append('<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{value}</th>" for value in cells(line)) + "</tr></thead><tbody>")
            index += 2
            while index < len(source) and source[index].startswith("| "):
                output.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells(source[index])) + "</tr>")
                index += 1
            output.append("</tbody></table></div>")
            continue
        if line.startswith("#"):
            level = min(3, len(line) - len(line.lstrip("#")))
            output.append(f"<h{level}>" + inline(line[level:].strip()) + f"</h{level}>")
        elif line.startswith("- "):
            output.append('<p class="bullet">• ' + inline(line[2:]) + "</p>")
        elif line:
            output.append("<p>" + inline(line) + "</p>")
        index += 1
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GRPO 完整算子组合改进结果</title>
<style>body{font:16px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif;color:#182733;background:#f5f7fa;margin:0}main{max-width:1700px;margin:auto;padding:34px}h1{font-size:32px}h2{margin-top:40px;border-bottom:2px solid #29435c;padding-bottom:8px}h3{margin-top:26px}p{max-width:1300px}strong{color:#8a3614}.table-wrap{overflow:auto;margin:20px 0;background:white;border:1px solid #ccd5de}table{border-collapse:collapse;min-width:100%;font-size:13px;line-height:1.5}th,td{padding:9px 12px;border:1px solid #d9e0e6;text-align:left;vertical-align:top}th{background:#263e54;color:white;white-space:nowrap}tbody tr:nth-child(even){background:#f0f4f7}code{font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere;background:#e9eef3;padding:2px 4px}a{color:#175d97}.bullet{margin:9px 0}@media print{body{background:white}main{padding:10px}.table-wrap{overflow:visible}table{font-size:9px}h2{break-before:page}}</style>
</head><body><main>""" + "\n".join(output) + "</main></body></html>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--parent", type=Path, help="默认读取新 manifest 中的 parent")
    parser.add_argument("--output", type=Path, help="默认 RUN/analysis_final")
    parser.add_argument("--allow-partial", action="store_true", help="显式允许部分结果，报告醒目标记 PARTIAL")
    parser.add_argument("--check-only", action="store_true", help="仅只读校验，不生成报告文件")
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        summary = collect(run, args.parent, args.allow_partial)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"报告拒绝生成：{exc}\n")
    if args.check_only:
        print(json.dumps({"status": summary["status"], "coverage": summary["coverage"],
                          "comparisons": summary["comparisons"]}, ensure_ascii=False, indent=2))
        return
    output = (args.output or run / "analysis_final").resolve()
    if output == run:
        parser.exit(2, "拒绝输出到正在运行的实验根目录；请使用独立 analysis_final 子目录。\n")
    output.mkdir(parents=True, exist_ok=True)
    markdown = markdown_report(summary)
    (output / "FINAL_REPORT.md").write_text(markdown, encoding="utf-8")
    (output / "FINAL_REPORT.html").write_text(html_report(markdown, summary), encoding="utf-8")
    (output / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(output),
                      "rows": summary["coverage"]["new_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
