#!/usr/bin/env python3
"""Nested-CV Pareto test for Beam-2 plus PPO-guided operator branches.

The policy is trained only on outer-training columns.  Hyperparameters are
chosen on a disjoint inner-validation fold, then the selected configuration is
retrained and evaluated once on the outer-test fold.  The final comparison
uses the standalone, self-decoding normalized-int64 Python codec from
``measure_selected_codec_cost.py`` for both the hybrid selector and Beam-4.

This is a falsification experiment, not a production-codec benchmark.  PPO is
allowed to add branches to Beam-2, but it never reads the exhaustive rates of
an outer-test column during standalone-codec search.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STOP = 28
MAX_DEPTH = 3


@dataclass(frozen=True)
class SearchConfig:
    name: str
    demonstrations_per_column: int
    bc_epochs: int
    cost_lambda: float
    policy_width: int
    policy_branch: int
    rate_weight: float


CONFIGS = (
    SearchConfig("d1_w4_b2", 1, 50, 0.0, 4, 2, 1.0),
    SearchConfig("d4_w4_b3", 4, 60, 0.0, 4, 3, 1.0),
    SearchConfig("d8_w6_b3", 8, 70, 0.0, 6, 3, 1.0),
    SearchConfig("d4_cost2_w4_b3", 4, 60, 0.002, 4, 3, 1.0),
    SearchConfig("d8_cost2_w6_b3", 8, 70, 0.002, 6, 3, 1.0),
    SearchConfig("d8_cost5_w6_b4", 8, 70, 0.005, 6, 4, 1.0),
)


def import_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pipeline_text(operators: tuple[str, ...], pipeline: tuple[int, ...]) -> str:
    return " -> ".join(operators[action] for action in pipeline)


def median_ms(samples_ns: list[int]) -> float:
    return statistics.median(samples_ns) / 1_000_000.0


def softmax_probs(actor, feature: np.ndarray, depth: int) -> np.ndarray:
    logits = actor.forward(feature[None, :])[0][0].copy()
    if depth == 0:
        logits[STOP] = -1.0e9
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def feature_stats(env, train_columns: np.ndarray) -> dict[str, np.ndarray]:
    raw_mean = env.raw_features[train_columns].mean(axis=0)
    raw_std = env.raw_features[train_columns].std(axis=0)
    raw_std[raw_std < 1.0e-8] = 1.0
    probe_mean = env.single_probe_features[train_columns].mean(axis=0)
    probe_std = env.single_probe_features[train_columns].std(axis=0)
    probe_std[probe_std < 1.0e-8] = 1.0
    return {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "probe_mean": probe_mean,
        "probe_std": probe_std,
    }


def online_feature(
    env,
    column: int,
    prefix: tuple[int, ...],
    current_rate: float,
    single_rates: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    best_single = float(np.min(single_rates))
    probes = np.clip(np.log(single_rates / best_single), 0.0, 10.0)
    raw = (env.raw_features[column] - stats["raw_mean"]) / stats["raw_std"]
    probes = (probes - stats["probe_mean"]) / stats["probe_std"]
    depth = np.zeros(3, dtype=np.float64)
    depth[len(prefix)] = 1.0
    prefix_one_hot = np.zeros(56, dtype=np.float64)
    for slot, action in enumerate(prefix):
        prefix_one_hot[slot * 28 + action] = 1.0
    result = np.concatenate(
        (
            raw,
            probes,
            np.asarray([np.clip(math.log(current_rate / best_single), -10.0, 10.0)]),
            depth,
            prefix_one_hot,
        )
    )
    if not np.all(np.isfinite(result)):
        raise AssertionError("non-finite online state feature")
    return np.clip(result, -10.0, 10.0)


def cost_aware_environment(env, train_columns: np.ndarray, cost_lambda: float):
    if cost_lambda <= 0:
        return env, 0.0
    observed_costs = [
        elapsed
        for col in train_columns
        for elapsed in env.eval_ms[env.columns[int(col)].key].values()
        if math.isfinite(elapsed) and elapsed > 0
    ]
    scale = float(np.median(np.asarray(observed_costs, dtype=np.float64)))
    adjusted = copy.copy(env)
    adjusted.rewards = env.rewards.copy()
    train_set = set(int(value) for value in train_columns)
    for sid, (col, prefix) in enumerate(zip(env.state_col, env.state_prefix)):
        if int(col) not in train_set:
            continue
        key = env.columns[int(col)].key
        for action in range(28):
            cost = env.eval_ms[key][prefix + (action,)]
            adjusted.rewards[sid, action] -= cost_lambda * min(cost / scale, 10.0)
    return adjusted, scale


def demonstration_examples(env, train_columns: np.ndarray, top_n: int):
    states: list[int] = []
    actions: list[int] = []
    for col_value in train_columns:
        col = int(col_value)
        key = env.columns[col].key
        ranked = sorted(
            env.rates[key].items(),
            key=lambda item: (item[1], len(item[0]), item[0]),
        )[:top_n]
        for pipeline, _rate in ranked:
            prefix: tuple[int, ...] = ()
            for action in pipeline:
                states.append(env.state_id[(col, prefix)])
                actions.append(action)
                prefix += (action,)
            if len(pipeline) < MAX_DEPTH:
                states.append(env.state_id[(col, pipeline)])
                actions.append(STOP)
    return np.asarray(states, dtype=np.int32), np.asarray(actions, dtype=np.int16)


def behavior_clone(cc, env, features, actor, optimizer, train_columns, config, rng) -> float:
    states, actions = demonstration_examples(
        env, train_columns, config.demonstrations_per_column
    )
    indices = np.arange(states.size)
    losses: list[float] = []
    for _epoch in range(config.bc_epochs):
        rng.shuffle(indices)
        for offset in range(0, indices.size, 128):
            take = indices[offset : offset + 128]
            x, hidden, probabilities = cc.masked_distribution(
                actor, features, env, states[take]
            )
            selected = probabilities[np.arange(take.size), actions[take]]
            losses.append(float(-np.mean(np.log(np.maximum(selected, 1.0e-30)))))
            grad = probabilities.copy()
            grad[np.arange(take.size), actions[take]] -= 1.0
            grad /= take.size
            optimizer.step(actor.backward(x, hidden, grad))
    return losses[-1]


def train_actor(cc, env, train_columns, config: SearchConfig, seed: int, args):
    features = cc.build_state_features(env, train_columns)
    stats = feature_stats(env, train_columns)
    rng = np.random.default_rng(seed)
    actor = cc.MLP(features.shape[1], args.hidden, 29, rng)
    critic = cc.MLP(features.shape[1], args.hidden, 1, rng)
    actor_optimizer = cc.Adam(actor.params, args.actor_lr)
    critic_optimizer = cc.Adam(critic.params, args.critic_lr)
    bc_loss = behavior_clone(
        cc, env, features, actor, actor_optimizer, train_columns, config, rng
    )
    training_env, cost_scale = cost_aware_environment(
        env, train_columns, config.cost_lambda
    )
    started = time.perf_counter()
    for _iteration in range(args.iterations):
        rollout, _sampling = cc.collect_ppo(
            training_env,
            features,
            actor,
            critic,
            rng,
            train_columns,
            args.episodes_per_iteration,
            args.gae_lambda,
        )
        cc.actor_update(
            training_env,
            features,
            actor,
            actor_optimizer,
            rollout,
            rng,
            args.policy_epochs,
            args.minibatch,
            args.clip_ratio,
            args.clip_ratio,
            args.entropy_coef,
        )
        cc.critic_update(
            features,
            critic,
            critic_optimizer,
            rollout,
            rng,
            args.value_epochs,
            args.minibatch,
        )
    return actor, features, stats, {
        "bc_final_loss": bc_loss,
        "cost_scale_ms": cost_scale,
        "train_seconds": time.perf_counter() - started,
        "demonstration_transitions": int(
            demonstration_examples(
                env, train_columns, config.demonstrations_per_column
            )[0].size
        ),
    }


def frontier_score(rate: float, best_single: float, log_probability: float, rate_weight: float):
    return log_probability - rate_weight * math.log(rate / best_single)


def proxy_hybrid_search(cc, env, features, actor, column: int, config: SearchConfig):
    key = env.columns[column].key
    rates = env.rates[key]
    evaluated: set[tuple[int, ...]] = {(action,) for action in range(28)}
    inference_started = time.perf_counter_ns()

    # Stable Beam-2 branch set.
    beam_frontier = sorted(evaluated, key=lambda p: (rates[p], p))[:2]
    for _depth in (2, 3):
        expanded = {prefix + (action,) for prefix in beam_frontier for action in range(28)}
        evaluated.update(expanded)
        beam_frontier = sorted(expanded, key=lambda p: (rates[p], p))[:2]

    # PPO keeps branches by a mixture of policy probability and observed prefix rate.
    root_state = int(env.root_state[column])
    root_probabilities = cc.masked_distribution(
        actor, features, env, np.asarray([root_state], dtype=np.int32)
    )[2][0]
    best_single = float(env.best_single[column])
    policy_frontier = [
        ((action,), math.log(max(float(root_probabilities[action]), 1.0e-30)))
        for action in range(28)
    ]
    policy_frontier.sort(
        key=lambda item: (
            -frontier_score(rates[item[0]], best_single, item[1], config.rate_weight),
            item[0],
        )
    )
    policy_frontier = policy_frontier[: config.policy_width]
    for _depth in (2, 3):
        expanded_policy: list[tuple[tuple[int, ...], float]] = []
        for prefix, cumulative_logp in policy_frontier:
            state = env.state_id[(column, prefix)]
            probabilities = cc.masked_distribution(
                actor, features, env, np.asarray([state], dtype=np.int32)
            )[2][0]
            actions = np.argsort(-probabilities[:28], kind="stable")[: config.policy_branch]
            for action_value in actions:
                action = int(action_value)
                pipeline = prefix + (action,)
                logp = cumulative_logp + math.log(max(float(probabilities[action]), 1.0e-30))
                evaluated.add(pipeline)
                expanded_policy.append((pipeline, logp))
        expanded_policy.sort(
            key=lambda item: (
                -frontier_score(rates[item[0]], best_single, item[1], config.rate_weight),
                item[0],
            )
        )
        policy_frontier = expanded_policy[: config.policy_width]
    inference_ms = (time.perf_counter_ns() - inference_started) / 1_000_000.0
    best = min(evaluated, key=lambda p: (rates[p], len(p), p))
    proxy_cost_ms = sum(env.eval_ms[key][pipeline] for pipeline in evaluated)
    return best, rates[best], len(evaluated), proxy_cost_ms, inference_ms


def actual_beam_search(values, codec, module, operators: tuple[str, ...], width: int):
    cache: dict[tuple[int, ...], bytes] = {}

    def encode(pipeline: tuple[int, ...]):
        if pipeline not in cache:
            names = tuple(operators[action] for action in pipeline)
            cache[pipeline] = codec.encode_stream(values, names, module)
        return cache[pipeline]

    frontier: list[tuple[int, ...]] = [()]
    for _depth in range(1, MAX_DEPTH + 1):
        expanded = [prefix + (action,) for prefix in frontier for action in range(28)]
        for pipeline in expanded:
            encode(pipeline)
        frontier = sorted(expanded, key=lambda p: (len(cache[p]), p))[:width]
    best = min(cache, key=lambda p: (len(cache[p]), len(p), p))
    return best, cache[best], len(cache)


def actual_hybrid_search(values, codec, module, env, actor, stats, column: int, config):
    cache: dict[tuple[int, ...], bytes] = {}

    def encode(pipeline: tuple[int, ...]):
        if pipeline not in cache:
            names = tuple(env.operators[action] for action in pipeline)
            cache[pipeline] = codec.encode_stream(values, names, module)
        return cache[pipeline]

    singles = [(action,) for action in range(28)]
    for pipeline in singles:
        encode(pipeline)
    original_bytes = max(int(values.size) * 8, 1)
    single_rates = np.asarray([len(cache[p]) / original_bytes for p in singles])
    best_single = float(np.min(single_rates))

    # Stable Beam-2 branch set, sharing the 28 single probes with PPO.
    beam_frontier = sorted(singles, key=lambda p: (len(cache[p]), p))[:2]
    for _depth in (2, 3):
        expanded = [prefix + (action,) for prefix in beam_frontier for action in range(28)]
        for pipeline in expanded:
            encode(pipeline)
        beam_frontier = sorted(expanded, key=lambda p: (len(cache[p]), p))[:2]

    root_feature = online_feature(env, column, (), best_single, single_rates, stats)
    root_probabilities = softmax_probs(actor, root_feature, 0)
    policy_frontier = [
        ((action,), math.log(max(float(root_probabilities[action]), 1.0e-30)))
        for action in range(28)
    ]
    policy_frontier.sort(
        key=lambda item: (
            -frontier_score(
                len(cache[item[0]]) / original_bytes,
                best_single,
                item[1],
                config.rate_weight,
            ),
            item[0],
        )
    )
    policy_frontier = policy_frontier[: config.policy_width]
    for _depth in (2, 3):
        expanded_policy: list[tuple[tuple[int, ...], float]] = []
        for prefix, cumulative_logp in policy_frontier:
            prefix_rate = len(encode(prefix)) / original_bytes
            feature = online_feature(
                env, column, prefix, prefix_rate, single_rates, stats
            )
            probabilities = softmax_probs(actor, feature, len(prefix))
            actions = np.argsort(-probabilities[:28], kind="stable")[: config.policy_branch]
            for action_value in actions:
                action = int(action_value)
                pipeline = prefix + (action,)
                encode(pipeline)
                logp = cumulative_logp + math.log(max(float(probabilities[action]), 1.0e-30))
                expanded_policy.append((pipeline, logp))
        expanded_policy.sort(
            key=lambda item: (
                -frontier_score(
                    len(cache[item[0]]) / original_bytes,
                    best_single,
                    item[1],
                    config.rate_weight,
                ),
                item[0],
            )
        )
        policy_frontier = expanded_policy[: config.policy_width]
    best = min(cache, key=lambda p: (len(cache[p]), len(p), p))
    return best, cache[best], len(cache)


def timed_search(callable_search, repetitions: int):
    warm = callable_search()
    samples: list[int] = []
    results = []
    for _repeat in range(repetitions):
        started = time.perf_counter_ns()
        result = callable_search()
        samples.append(time.perf_counter_ns() - started)
        results.append(result)
    reference = (warm[0], warm[1], warm[2])
    if any((result[0], result[1], result[2]) != reference for result in results):
        raise AssertionError("search is not deterministic")
    return warm, median_ms(samples), min(samples) / 1_000_000.0


def verify_stream(values, pipeline, encoded, codec, operators: tuple[str, ...]):
    restored, decoded_pipeline = codec.decode_stream(encoded)
    names = tuple(operators[action] for action in pipeline)
    if decoded_pipeline != names or not np.array_equal(restored, values):
        raise AssertionError(f"standalone round trip failed for {names}")


def actual_result(
    values,
    codec,
    module,
    env,
    actor,
    stats,
    column: int,
    config: SearchConfig,
    repetitions: int,
    beam_cache: dict[int, dict[str, Any]],
):
    if column not in beam_cache or int(beam_cache[column]["timing_repetitions"]) < repetitions:
        beam, beam_ms, beam_min_ms = timed_search(
            lambda: actual_beam_search(values, codec, module, env.operators, 4), repetitions
        )
        verify_stream(values, beam[0], beam[1], codec, env.operators)
        beam_cache[column] = {
            "pipeline": beam[0],
            "encoded_bytes": len(beam[1]),
            "evaluations": beam[2],
            "search_ms_median": beam_ms,
            "search_ms_min": beam_min_ms,
            "timing_repetitions": repetitions,
        }
    hybrid, hybrid_ms, hybrid_min_ms = timed_search(
        lambda: actual_hybrid_search(
            values, codec, module, env, actor, stats, column, config
        ),
        repetitions,
    )
    verify_stream(values, hybrid[0], hybrid[1], codec, env.operators)
    return {
        "pipeline": hybrid[0],
        "encoded_bytes": len(hybrid[1]),
        "evaluations": hybrid[2],
        "search_ms_median": hybrid_ms,
        "search_ms_min": hybrid_min_ms,
    }, beam_cache[column]


def choose_config(
    validation_rows: list[dict[str, Any]], configs: tuple[SearchConfig, ...]
) -> SearchConfig:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in validation_rows:
        grouped.setdefault(row["config"], []).append(row)
    summaries = []
    for config in configs:
        rows = grouped[config.name]
        hybrid_bytes = sum(int(row["hybrid_bytes"]) for row in rows)
        beam_bytes = sum(int(row["beam4_bytes"]) for row in rows)
        summaries.append(
            (
                hybrid_bytes >= beam_bytes,
                hybrid_bytes,
                statistics.median(float(row["hybrid_search_ms"]) for row in rows),
                sum(int(row["hybrid_evaluations"]) for row in rows),
                config.name,
                config,
            )
        )
    return min(summaries)[-1]


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cc = import_path(repo_root / "scripts" / "cross_column_rl_compare.py", "pareto_crosscol")
    codec = import_path(repo_root / "scripts" / "measure_selected_codec_cost.py", "pareto_codec")
    env, data_inputs, operator_source = cc.load_environment(
        args.data_repo.resolve(),
        args.measurement_root.resolve(),
        args.summary_csv.resolve(),
        args.baseline_csv.resolve(),
    )
    if set(env.operators) != set(codec.OPERATOR_IDS):
        raise AssertionError("operator sets differ between policy and codec")
    codec_module = codec.load_operator_module(args.data_repo.resolve())
    input_rows = [
        {
            "dataset": column.dataset,
            "column_index": str(column.index),
            "column_name": column.name,
        }
        for column in env.columns
    ]
    values_by_key = codec.load_values(args.data_repo.resolve(), input_rows, codec_module)
    folds = cc.assign_folds(env.columns, args.folds)
    if set(folds.tolist()) != set(range(args.folds)):
        raise AssertionError("outer fold coverage is incomplete")

    beam_cache: dict[int, dict[str, Any]] = {}
    validation_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[int, SearchConfig] = {}

    # Nested selection uses one fixed tuning seed and never accesses the outer test fold.
    for outer_fold in range(args.folds):
        validation_fold = (outer_fold + 1) % args.folds
        inner_train = np.flatnonzero((folds != outer_fold) & (folds != validation_fold))
        validation = np.flatnonzero(folds == validation_fold)
        outer_test = np.flatnonzero(folds == outer_fold)
        if set(inner_train) & set(validation) or set(inner_train) & set(outer_test):
            raise AssertionError("nested split leakage")
        fold_validation_rows: list[dict[str, Any]] = []
        for config_index, config in enumerate(CONFIGS[: args.configs]):
            actor, _features, stats, train_info = train_actor(
                cc,
                env,
                inner_train,
                config,
                args.tuning_seed + outer_fold * 1009 + config_index * 100_003,
                args,
            )
            training_rows.append(
                {
                    "stage": "inner_tuning",
                    "outer_fold": outer_fold,
                    "seed": args.tuning_seed,
                    "config": config.name,
                    "train_columns": len(inner_train),
                    **train_info,
                }
            )
            for col_value in validation:
                col = int(col_value)
                values = values_by_key[env.columns[col].key]
                hybrid, beam = actual_result(
                    values,
                    codec,
                    codec_module,
                    env,
                    actor,
                    stats,
                    col,
                    config,
                    args.validation_repetitions,
                    beam_cache,
                )
                row = {
                    "outer_fold": outer_fold,
                    "validation_fold": validation_fold,
                    "config": config.name,
                    "dataset": env.columns[col].dataset,
                    "column_index": env.columns[col].index,
                    "column_name": env.columns[col].name,
                    "hybrid_bytes": hybrid["encoded_bytes"],
                    "beam4_bytes": beam["encoded_bytes"],
                    "hybrid_evaluations": hybrid["evaluations"],
                    "beam4_evaluations": beam["evaluations"],
                    "hybrid_search_ms": hybrid["search_ms_median"],
                    "beam4_search_ms": beam["search_ms_median"],
                }
                validation_rows.append(row)
                fold_validation_rows.append(row)
        available = tuple(CONFIGS[: args.configs])
        selected = choose_config(fold_validation_rows, available)
        selected_by_fold[outer_fold] = selected
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                **asdict(selected),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "selected_config",
                    "outer_fold": outer_fold,
                    "config": selected.name,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Retrain on every non-test column.  Each outer-test column is evaluated once per seed.
    for seed in args.seeds:
        for outer_fold in range(args.folds):
            config = selected_by_fold[outer_fold]
            outer_train = np.flatnonzero(folds != outer_fold)
            outer_test = np.flatnonzero(folds == outer_fold)
            actor, features, stats, train_info = train_actor(
                cc,
                env,
                outer_train,
                config,
                seed + outer_fold * 1009,
                args,
            )
            training_rows.append(
                {
                    "stage": "outer_retrain",
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "config": config.name,
                    "train_columns": len(outer_train),
                    **train_info,
                }
            )
            for col_value in outer_test:
                col = int(col_value)
                column = env.columns[col]
                values = values_by_key[column.key]
                hybrid, beam = actual_result(
                    values,
                    codec,
                    codec_module,
                    env,
                    actor,
                    stats,
                    col,
                    config,
                    args.test_repetitions,
                    beam_cache,
                )
                proxy_pipeline, proxy_rate, proxy_evaluations, proxy_ms, inference_ms = (
                    proxy_hybrid_search(cc, env, features, actor, col, config)
                )
                original_bytes = int(values.size) * 8
                test_rows.append(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "config": config.name,
                        "dataset": column.dataset,
                        "column_index": column.index,
                        "column_name": column.name,
                        "logical_type": column.logical_type,
                        "sample_values": int(values.size),
                        "original_bytes": original_bytes,
                        "hybrid_pipeline": pipeline_text(env.operators, hybrid["pipeline"]),
                        "hybrid_encoded_bytes": hybrid["encoded_bytes"],
                        "hybrid_codec_rate": hybrid["encoded_bytes"] / original_bytes,
                        "hybrid_evaluations": hybrid["evaluations"],
                        "hybrid_search_ms_median": hybrid["search_ms_median"],
                        "hybrid_search_ms_min": hybrid["search_ms_min"],
                        "beam4_pipeline": pipeline_text(env.operators, beam["pipeline"]),
                        "beam4_encoded_bytes": beam["encoded_bytes"],
                        "beam4_codec_rate": beam["encoded_bytes"] / original_bytes,
                        "beam4_evaluations": beam["evaluations"],
                        "beam4_search_ms_median": beam["search_ms_median"],
                        "beam4_search_ms_min": beam["search_ms_min"],
                        "gain_vs_beam4_pct": 100.0
                        * (beam["encoded_bytes"] - hybrid["encoded_bytes"])
                        / beam["encoded_bytes"],
                        "search_speedup_vs_beam4": beam["search_ms_median"]
                        / hybrid["search_ms_median"],
                        "proxy_pipeline": pipeline_text(env.operators, proxy_pipeline),
                        "proxy_rate": proxy_rate,
                        "proxy_evaluations": proxy_evaluations,
                        "proxy_eval_ms_sum": proxy_ms,
                        "proxy_policy_inference_ms": inference_ms,
                        "lossless": 1,
                    }
                )
            print(
                json.dumps(
                    {
                        "stage": "outer_test_complete",
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "test_columns": len(outer_test),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    write_csv(output_dir / "validation_dse.csv", validation_rows)
    write_csv(output_dir / "selected_config.csv", selected_rows)
    write_csv(output_dir / "training_summary.csv", training_rows)
    write_csv(output_dir / "per_seed_test.csv", test_rows)

    per_column_rows = []
    for col, column in enumerate(env.columns):
        rows = [
            row
            for row in test_rows
            if row["dataset"] == column.dataset
            and int(row["column_index"]) == column.index
        ]
        hybrid_bytes = int(statistics.median(int(row["hybrid_encoded_bytes"]) for row in rows))
        beam_bytes = int(rows[0]["beam4_encoded_bytes"])
        per_column_rows.append(
            {
                "dataset": column.dataset,
                "column_index": column.index,
                "column_name": column.name,
                "seeds": len(rows),
                "median_hybrid_bytes": hybrid_bytes,
                "beam4_bytes": beam_bytes,
                "median_gain_vs_beam4_pct": 100.0 * (beam_bytes - hybrid_bytes) / beam_bytes,
                "strict_win": int(hybrid_bytes < beam_bytes),
                "tie": int(hybrid_bytes == beam_bytes),
                "loss": int(hybrid_bytes > beam_bytes),
                "median_hybrid_evaluations": statistics.median(
                    int(row["hybrid_evaluations"]) for row in rows
                ),
                "beam4_evaluations": int(rows[0]["beam4_evaluations"]),
                "median_hybrid_search_ms": statistics.median(
                    float(row["hybrid_search_ms_median"]) for row in rows
                ),
                "beam4_search_ms": float(rows[0]["beam4_search_ms_median"]),
            }
        )
    write_csv(output_dir / "per_column_summary.csv", per_column_rows)

    hybrid_total = sum(int(row["median_hybrid_bytes"]) for row in per_column_rows)
    beam_total = sum(int(row["beam4_bytes"]) for row in per_column_rows)
    hybrid_times = [float(row["median_hybrid_search_ms"]) for row in per_column_rows]
    beam_times = [float(row["beam4_search_ms"]) for row in per_column_rows]
    rate_pass = hybrid_total < beam_total
    time_pass = statistics.median(hybrid_times) < statistics.median(beam_times)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if rate_pass and time_pass else "FAIL",
        "rate_gate_pass": rate_pass,
        "time_gate_pass": time_pass,
        "columns": len(per_column_rows),
        "seeds": list(args.seeds),
        "strict_wins": sum(int(row["strict_win"]) for row in per_column_rows),
        "ties": sum(int(row["tie"]) for row in per_column_rows),
        "losses": sum(int(row["loss"]) for row in per_column_rows),
        "hybrid_total_bytes": hybrid_total,
        "beam4_total_bytes": beam_total,
        "aggregate_gain_vs_beam4_pct": 100.0 * (beam_total - hybrid_total) / beam_total,
        "median_hybrid_search_ms": statistics.median(hybrid_times),
        "median_beam4_search_ms": statistics.median(beam_times),
        "median_search_speedup": statistics.median(beam_times)
        / statistics.median(hybrid_times),
        "median_hybrid_evaluations": statistics.median(
            float(row["median_hybrid_evaluations"]) for row in per_column_rows
        ),
        "beam4_evaluations": 252,
        "split": "5 outer test folds; next fold is inner validation; remaining 3 folds train",
        "selector": "Beam-2 plus PPO-guided branches",
        "codec": "standalone self-decoding normalized-int64 Python codec",
        "production_claim_allowed": False,
        "claim_boundary": (
            "Two datasets and 20 columns only. Final rates and selection wall times are real "
            "for the standalone Python codec, not production CompressionBench kernels."
        ),
    }
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = f"""# PPO-guided Beam Pareto test

Verdict: **{summary['verdict']}**.  The frozen gate requires both aggregate
standalone-codec bytes below Beam-4 and median online selection time below
Beam-4 on outer-test columns.

| Metric | PPO-guided Beam-2 | Beam-4 |
|---|---:|---:|
| Total encoded bytes | {hybrid_total} | {beam_total} |
| Aggregate gain vs Beam-4 | {summary['aggregate_gain_vs_beam4_pct']:.6f}% | 0% |
| Median search time | {summary['median_hybrid_search_ms']:.3f} ms | {summary['median_beam4_search_ms']:.3f} ms |
| Median codec evaluations | {summary['median_hybrid_evaluations']:.0f} | 252 |
| Win / tie / loss columns | {summary['strict_wins']} / {summary['ties']} / {summary['losses']} | — |

The split is nested: each outer fold is test-only, the next fold selects the
configuration using standalone codec bytes/cost, and the remaining three folds
train PPO.  The chosen configuration is then retrained on all four non-test
folds.  Test columns are never used for normalization, reward sampling, or
configuration selection.

Claim boundary: this is a standalone self-decoding normalized-int64 Python
codec on Air-Quality and Metro-Traffic, not a production codec/kernel result.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        **summary,
        "arguments": vars(args),
        "inputs": {
            "operator_source": str(operator_source),
            "operator_source_sha256": hashlib.sha256(Path(operator_source).read_bytes()).hexdigest(),
            "measurement_root": str(args.measurement_root.resolve()),
            "summary_csv": str(args.summary_csv.resolve()),
            "baseline_csv": str(args.baseline_csv.resolve()),
            "data_inputs": data_inputs,
        },
    }
    manifest["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in manifest["arguments"].items()}
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--configs", type=int, default=len(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    parser.add_argument("--tuning-seed", type=int, default=20260825)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--episodes-per-iteration", type=int, default=192)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--value-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=0.001)
    parser.add_argument("--critic-lr", type=float, default=0.001)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--validation-repetitions", type=int, default=1)
    parser.add_argument("--test-repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.folds != 5:
        parser.error("this experiment is preregistered for five folds")
    if args.configs <= 0 or args.configs > len(CONFIGS):
        parser.error(f"--configs must be in [1, {len(CONFIGS)}]")
    if min(args.validation_repetitions, args.test_repetitions) <= 0:
        parser.error("timing repetitions must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
