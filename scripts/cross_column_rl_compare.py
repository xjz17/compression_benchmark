#!/usr/bin/env python3
"""Cross-column PPO/GRPO/DAPO comparison for depth-3 operator search.

Every reported column is held out from policy training.  The environment uses
the existing exhaustive proxy-rate table, so this evaluates policy
generalization without rerunning codecs during optimization.  DAPO is adapted
to the fixed three-action horizon: Clip-Higher, dynamic sampling, and
action-level loss apply; overlong reward shaping does not.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STOP = 28
MAX_DEPTH = 3
DYNAMIC_SAMPLING_MAX_MULTIPLIER = 32
RAW_FEATURE_NAMES = (
    "log_n",
    "is_float",
    "decimal_places_scaled",
    "signed_log_mean",
    "log_std",
    "log_range",
    "zero_fraction",
    "negative_fraction",
    "unique_fraction",
    "repeat_fraction",
    "nondecreasing_fraction",
    "nonincreasing_fraction",
    "delta_zero_fraction",
    "log_mean_abs_delta",
    "log_std_delta",
    "lag1_correlation",
    "byte_entropy_scaled",
    "log_max_abs",
)


@dataclass(frozen=True)
class Column:
    dataset: str
    index: int
    name: str
    logical_type: str
    sample_values: int
    sample_sha256: str
    baseline_algorithm: str
    baseline_rate: float

    @property
    def key(self) -> tuple[str, int]:
        return self.dataset, self.index


@dataclass
class Environment:
    columns: list[Column]
    operators: tuple[str, ...]
    rates: dict[tuple[str, int], dict[tuple[int, ...], float]]
    eval_ms: dict[tuple[str, int], dict[tuple[int, ...], float]]
    best_single: np.ndarray
    oracle: np.ndarray
    beam4: np.ndarray
    raw_features: np.ndarray
    single_probe_features: np.ndarray
    state_col: np.ndarray
    state_depth: np.ndarray
    state_prefix: list[tuple[int, ...]]
    state_rate_ratio_log: np.ndarray
    next_state: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    valid: np.ndarray
    root_state: np.ndarray
    state_id: dict[tuple[int, tuple[int, ...]], int]


class NoInformativeGroups(RuntimeError):
    def __init__(self, target_groups: int, sampled_groups: int, filtered_groups: int):
        super().__init__(
            f"dynamic sampling could not fill a {target_groups}-group batch after {sampled_groups} sampled groups"
        )
        self.sampled_groups = sampled_groups
        self.filtered_groups = filtered_groups


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_operator_module(repo_root: Path):
    path = repo_root / "scripts" / "generate_air_operator_pair_heatmap.py"
    spec = importlib.util.spec_from_file_location("cross_column_operator_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load operator implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def column_features(values: np.ndarray, logical_type: str, decimal_places: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.int64)
    xf = x.astype(np.float64)
    n = x.size
    delta = np.diff(xf)
    if n > 1 and float(np.std(xf[:-1])) > 0 and float(np.std(xf[1:])) > 0:
        corr = float(np.corrcoef(xf[:-1], xf[1:])[0, 1])
    else:
        corr = 0.0
    raw_bytes = x.astype("<i8", copy=False).view(np.uint8)
    counts = np.bincount(raw_bytes, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    byte_entropy = float(-np.sum(probs * np.log2(probs)) / 8.0)
    values_out = np.asarray(
        [
            math.log1p(n),
            float(logical_type in {"float", "double"}),
            min(max(decimal_places, 0), 14) / 14.0,
            signed_log1p(float(np.mean(xf))),
            math.log1p(float(np.std(xf))),
            math.log1p(float(np.ptp(xf))),
            float(np.mean(x == 0)),
            float(np.mean(x < 0)),
            float(np.unique(x).size / n),
            float(np.mean(x[1:] == x[:-1])) if n > 1 else 1.0,
            float(np.mean(delta >= 0)) if delta.size else 1.0,
            float(np.mean(delta <= 0)) if delta.size else 1.0,
            float(np.mean(delta == 0)) if delta.size else 1.0,
            math.log1p(float(np.mean(np.abs(delta)))) if delta.size else 0.0,
            math.log1p(float(np.std(delta))) if delta.size else 0.0,
            corr,
            byte_entropy,
            math.log1p(float(np.max(np.abs(xf)))),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values_out)):
        raise AssertionError("non-finite raw feature")
    return values_out


def locate_dataset(repo_root: Path, dataset: str) -> tuple[Path, Path]:
    data_candidates = (
        repo_root / "backend" / "dataset_csv" / f"{dataset}.csv",
        repo_root / "orignial_datasets" / f"{dataset}.csv",
    )
    info_candidates = (
        repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv",
        repo_root / "dataset" / dataset / f"{dataset}_column_info.csv",
    )
    data = next((path for path in data_candidates if path.is_file()), None)
    info = next((path for path in info_candidates if path.is_file()), None)
    if data is None or info is None:
        raise FileNotFoundError(f"missing data or metadata for {dataset}")
    return data, info


def load_raw_features(repo_root: Path, columns: list[Column], module) -> tuple[np.ndarray, list[dict]]:
    result = np.zeros((len(columns), len(RAW_FEATURE_NAMES)), dtype=np.float64)
    inputs = []
    by_dataset: dict[str, list[tuple[int, Column]]] = defaultdict(list)
    for position, column in enumerate(columns):
        by_dataset[column.dataset].append((position, column))
    for dataset, items in sorted(by_dataset.items()):
        data_path, info_path = locate_dataset(repo_root, dataset)
        metadata = {int(row["column_index"]): row for row in read_csv(info_path)}
        usecols = [column.name for _position, column in items]
        frame = pd.read_csv(data_path, usecols=usecols, low_memory=False, dtype=str)
        for position, column in items:
            row = metadata[column.index]
            if row["column_name"] != column.name:
                raise AssertionError((dataset, column.index, row["column_name"], column.name))
            values = module.column_values(frame[column.name], row)[: column.sample_values]
            digest = hashlib.sha256(module.i64_bytes(values)).hexdigest()
            if values.size != column.sample_values or digest != column.sample_sha256:
                raise AssertionError(
                    f"sample mismatch {column.key}: n={values.size}/{column.sample_values} "
                    f"sha={digest}/{column.sample_sha256}"
                )
            result[position] = column_features(
                values,
                column.logical_type,
                int(row.get("max_point_number", 0) or 0),
            )
        inputs.append(
            {
                "dataset": dataset,
                "data_path": str(data_path),
                "data_size": data_path.stat().st_size,
                "data_mtime_ns": data_path.stat().st_mtime_ns,
                "metadata_path": str(info_path),
                "metadata_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
            }
        )
    return result, inputs


def load_environment(
    repo_root: Path,
    measurement_root: Path,
    summary_csv: Path,
    baseline_csv: Path,
    datasets: tuple[str, ...] | None = None,
) -> tuple[Environment, list[dict], Path]:
    detail_paths = sorted(measurement_root.glob("*/pipeline_measurements.csv"))
    if not detail_paths:
        raise AssertionError(f"no measurement shards under {measurement_root}")
    text_rows = []
    operator_names: set[str] = set()
    column_names: dict[tuple[str, int], str] = {}
    for path in detail_paths:
        for row in read_csv(path):
            if datasets is not None and row["dataset"] not in datasets:
                continue
            if int(row["lossless"]) != 1 or row["status"] != "ok":
                continue
            pipeline = tuple(part.strip() for part in row["pipeline"].split(" -> "))
            operator_names.update(pipeline)
            key = row["dataset"], int(row["column_index"])
            column_names[key] = row["column_name"]
            text_rows.append((key, pipeline, float(row["rate"]), float(row["eval_ms"])))
    operators = tuple(sorted(operator_names))
    if len(operators) != 28:
        raise AssertionError(f"expected 28 operators, got {len(operators)}")
    operator_to_id = {name: index for index, name in enumerate(operators)}
    rates: dict[tuple[str, int], dict[tuple[int, ...], float]] = defaultdict(dict)
    eval_ms: dict[tuple[str, int], dict[tuple[int, ...], float]] = defaultdict(dict)
    for key, pipeline_text, rate, elapsed in text_rows:
        pipeline = tuple(operator_to_id[name] for name in pipeline_text)
        rates[key][pipeline] = rate
        eval_ms[key][pipeline] = elapsed
    if not rates:
        raise AssertionError(f"no columns selected for datasets={datasets}")
    for key, values in rates.items():
        if len(values) != 22764:
            raise AssertionError(f"{key} has {len(values)} pipelines, expected 22764")

    summaries = {
        (row["dataset"], int(row["column_index"])): row
        for row in read_csv(summary_csv)
        if datasets is None or row["dataset"] in datasets
    }
    baselines = {
        (row["dataset"], int(row["column_index"])): row
        for row in read_csv(baseline_csv)
        if datasets is None or row["dataset"] in datasets
    }
    keys = sorted(rates)
    if set(keys) != set(summaries) or set(keys) != set(baselines):
        raise AssertionError("summary/baseline coverage differs from measurement coverage")
    columns = []
    for key in keys:
        summary = summaries[key]
        baseline = baselines[key]
        columns.append(
            Column(
                dataset=key[0],
                index=key[1],
                name=column_names[key],
                logical_type=summary["logical_type"],
                sample_values=int(summary["sample_values"]),
                sample_sha256=summary["sample_sha256"],
                baseline_algorithm=baseline["baseline_algorithm"],
                baseline_rate=float(baseline["baseline_rate"]),
            )
        )
    module, operator_source = load_operator_module(repo_root)
    raw_features, data_inputs = load_raw_features(repo_root, columns, module)
    best_single = np.asarray(
        [min(rate for pipeline, rate in rates[column.key].items() if len(pipeline) == 1) for column in columns]
    )
    oracle = np.asarray([min(rates[column.key].values()) for column in columns])
    beam4 = np.asarray([float(summaries[column.key]["beam4_rate"]) for column in columns])
    single_probe = np.zeros((len(columns), len(operators)), dtype=np.float64)
    for col, column in enumerate(columns):
        for action in range(len(operators)):
            single_probe[col, action] = np.clip(
                math.log(rates[column.key][(action,)] / best_single[col]), 0.0, 10.0
            )

    state_col = []
    state_depth = []
    state_prefix = []
    state_ratio = []
    state_id: dict[tuple[int, tuple[int, ...]], int] = {}
    root_state = np.zeros(len(columns), dtype=np.int32)
    for col, column in enumerate(columns):
        prefixes: Iterable[tuple[int, ...]] = [()]
        prefixes = list(prefixes) + [(a,) for a in range(28)] + [
            (a, b) for a in range(28) for b in range(28)
        ]
        for prefix in prefixes:
            sid = len(state_col)
            state_id[(col, prefix)] = sid
            if not prefix:
                root_state[col] = sid
                rate = best_single[col]
            else:
                rate = rates[column.key][prefix]
            state_col.append(col)
            state_depth.append(len(prefix))
            state_prefix.append(prefix)
            state_ratio.append(np.clip(math.log(rate / best_single[col]), -10.0, 10.0))
    states = len(state_col)
    next_state = np.full((states, 29), -1, dtype=np.int32)
    rewards = np.zeros((states, 29), dtype=np.float64)
    done = np.ones((states, 29), dtype=bool)
    valid = np.ones((states, 29), dtype=bool)
    for sid, (col, depth, prefix) in enumerate(zip(state_col, state_depth, state_prefix)):
        column = columns[col]
        current_rate = best_single[col] if depth == 0 else rates[column.key][prefix]
        if depth == 0:
            valid[sid, STOP] = False
        for action in range(28):
            new_prefix = prefix + (action,)
            next_rate = rates[column.key][new_prefix]
            rewards[sid, action] = math.log(current_rate / next_rate)
            if len(new_prefix) < MAX_DEPTH:
                done[sid, action] = False
                next_state[sid, action] = state_id[(col, new_prefix)]
        if depth >= 1:
            rewards[sid, STOP] = 0.0

    env = Environment(
        columns=columns,
        operators=operators,
        rates=dict(rates),
        eval_ms=dict(eval_ms),
        best_single=best_single,
        oracle=oracle,
        beam4=beam4,
        raw_features=raw_features,
        single_probe_features=single_probe,
        state_col=np.asarray(state_col, dtype=np.int16),
        state_depth=np.asarray(state_depth, dtype=np.int8),
        state_prefix=state_prefix,
        state_rate_ratio_log=np.asarray(state_ratio, dtype=np.float64),
        next_state=next_state,
        rewards=rewards,
        done=done,
        valid=valid,
        root_state=root_state,
        state_id=state_id,
    )
    return env, data_inputs, operator_source


def assign_folds(columns: list[Column], folds: int) -> np.ndarray:
    if folds == len(columns):
        # Explicit leave-one-column-out evaluation.  Sorting in load_environment
        # makes this deterministic and every held-out fold has exactly one column.
        return np.arange(len(columns), dtype=np.int16)
    assignment = np.full(len(columns), -1, dtype=np.int8)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, column in enumerate(columns):
        groups[(column.dataset, column.logical_type)].append(position)
    for group, positions in sorted(groups.items()):
        positions.sort(key=lambda pos: (columns[pos].index, columns[pos].name))
        offset = hashlib.sha256("|".join(group).encode()).digest()[0] % folds
        for rank, position in enumerate(positions):
            assignment[position] = (offset + rank) % folds
    if np.any(assignment < 0) or set(assignment.tolist()) != set(range(folds)):
        raise AssertionError(f"invalid fold coverage: {assignment.tolist()}")
    return assignment


def build_state_features(env: Environment, train_columns: np.ndarray) -> np.ndarray:
    raw_mean = env.raw_features[train_columns].mean(axis=0)
    raw_std = env.raw_features[train_columns].std(axis=0)
    raw_std[raw_std < 1.0e-8] = 1.0
    probe_mean = env.single_probe_features[train_columns].mean(axis=0)
    probe_std = env.single_probe_features[train_columns].std(axis=0)
    probe_std[probe_std < 1.0e-8] = 1.0
    normalized_raw = (env.raw_features - raw_mean) / raw_std
    normalized_probe = (env.single_probe_features - probe_mean) / probe_std
    state_count = env.state_col.size
    depth_one_hot = np.zeros((state_count, 3), dtype=np.float64)
    depth_one_hot[np.arange(state_count), env.state_depth] = 1.0
    prefix_one_hot = np.zeros((state_count, 2 * 28), dtype=np.float64)
    for sid, prefix in enumerate(env.state_prefix):
        for slot, action in enumerate(prefix):
            prefix_one_hot[sid, slot * 28 + action] = 1.0
    features = np.concatenate(
        (
            normalized_raw[env.state_col],
            normalized_probe[env.state_col],
            env.state_rate_ratio_log[:, None],
            depth_one_hot,
            prefix_one_hot,
        ),
        axis=1,
    )
    if not np.all(np.isfinite(features)):
        raise AssertionError("non-finite state feature")
    return np.clip(features, -10.0, 10.0)


class MLP:
    def __init__(self, input_dim: int, hidden: int, output_dim: int, rng: np.random.Generator):
        self.params = [
            rng.normal(0.0, math.sqrt(2.0 / input_dim), (input_dim, hidden)),
            np.zeros(hidden, dtype=np.float64),
            rng.normal(0.0, 0.01, (hidden, output_dim)),
            np.zeros(output_dim, dtype=np.float64),
        ]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.params[0] + self.params[1])
        return hidden @ self.params[2] + self.params[3], hidden

    def backward(self, x: np.ndarray, hidden: np.ndarray, output_grad: np.ndarray) -> list[np.ndarray]:
        grad_w2 = hidden.T @ output_grad
        grad_b2 = output_grad.sum(axis=0)
        hidden_grad = (output_grad @ self.params[2].T) * (1.0 - hidden * hidden)
        return [x.T @ hidden_grad, hidden_grad.sum(axis=0), grad_w2, grad_b2]


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]
        self.step_number = 0

    def step(self, grads: list[np.ndarray], max_norm: float = 1.0) -> None:
        norm = math.sqrt(sum(float(np.sum(grad * grad)) for grad in grads))
        if norm > max_norm:
            grads = [grad * (max_norm / norm) for grad in grads]
        self.step_number += 1
        for index, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[index] = 0.9 * self.m[index] + 0.1 * grad
            self.v[index] = 0.999 * self.v[index] + 0.001 * (grad * grad)
            m_hat = self.m[index] / (1.0 - 0.9**self.step_number)
            v_hat = self.v[index] / (1.0 - 0.999**self.step_number)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1.0e-8)


def masked_distribution(actor: MLP, features: np.ndarray, env: Environment, states: np.ndarray):
    x = features[states]
    logits, hidden = actor.forward(x)
    logits = logits.copy()
    logits[env.state_depth[states] == 0, STOP] = -1.0e9
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return x, hidden, probs


def sample_actions(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    uniforms = rng.random(probs.shape[0])
    return np.sum(np.cumsum(probs, axis=1) < uniforms[:, None], axis=1).astype(np.int16)


def collect_episodes(
    env: Environment,
    features: np.ndarray,
    actor: MLP,
    rng: np.random.Generator,
    episode_columns: np.ndarray,
    critic: MLP | None,
):
    episode_count = episode_columns.size
    current = env.root_state[episode_columns].copy()
    active = np.ones(episode_count, dtype=bool)
    state_steps = np.full((episode_count, MAX_DEPTH), -1, dtype=np.int32)
    action_steps = np.full((episode_count, MAX_DEPTH), -1, dtype=np.int16)
    logp_steps = np.zeros((episode_count, MAX_DEPTH), dtype=np.float64)
    reward_steps = np.zeros((episode_count, MAX_DEPTH), dtype=np.float64)
    value_steps = np.zeros((episode_count, MAX_DEPTH), dtype=np.float64)
    mask = np.zeros((episode_count, MAX_DEPTH), dtype=bool)
    for step in range(MAX_DEPTH):
        episodes = np.flatnonzero(active)
        if episodes.size == 0:
            break
        states = current[episodes]
        x, _hidden, probs = masked_distribution(actor, features, env, states)
        actions = sample_actions(rng, probs)
        state_steps[episodes, step] = states
        action_steps[episodes, step] = actions
        logp_steps[episodes, step] = np.log(np.maximum(probs[np.arange(episodes.size), actions], 1.0e-30))
        reward_steps[episodes, step] = env.rewards[states, actions]
        mask[episodes, step] = True
        if critic is not None:
            value_steps[episodes, step] = critic.forward(x)[0][:, 0]
        terminal = env.done[states, actions]
        continuing = episodes[~terminal]
        current[continuing] = env.next_state[states[~terminal], actions[~terminal]]
        active[episodes[terminal]] = False
    if np.any(active):
        raise AssertionError("episode exceeded fixed horizon")
    lengths = mask.sum(axis=1)
    returns = reward_steps.sum(axis=1)
    flat_episode, flat_step = np.nonzero(mask)
    return {
        "states": state_steps[mask],
        "actions": action_steps[mask],
        "old_logp": logp_steps[mask],
        "rewards": reward_steps[mask],
        "old_values": value_steps[mask],
        "episode": flat_episode.astype(np.int32),
        "step": flat_step.astype(np.int8),
        "lengths": lengths,
        "episode_returns": returns,
        "mask": mask,
        "reward_matrix": reward_steps,
        "value_matrix": value_steps,
    }


def collect_ppo(
    env: Environment,
    features: np.ndarray,
    actor: MLP,
    critic: MLP,
    rng: np.random.Generator,
    train_columns: np.ndarray,
    episodes: int,
    gae_lambda: float,
):
    episode_columns = rng.choice(train_columns, size=episodes, replace=True)
    rollout = collect_episodes(env, features, actor, rng, episode_columns, critic)
    advantage_matrix = np.zeros_like(rollout["reward_matrix"])
    return_matrix = np.zeros_like(rollout["reward_matrix"])
    for episode in range(episodes):
        next_advantage = 0.0
        next_value = 0.0
        for step in range(int(rollout["lengths"][episode]) - 1, -1, -1):
            value = rollout["value_matrix"][episode, step]
            delta = rollout["reward_matrix"][episode, step] + next_value - value
            next_advantage = delta + gae_lambda * next_advantage
            advantage_matrix[episode, step] = next_advantage
            return_matrix[episode, step] = next_advantage + value
            next_value = value
    rollout["advantages"] = advantage_matrix[rollout["mask"]]
    rollout["targets"] = return_matrix[rollout["mask"]]
    rollout["weights"] = np.ones(rollout["states"].size, dtype=np.float64)
    return rollout, {"sampled_groups": 0, "filtered_groups": 0}


def collect_group_relative(
    env: Environment,
    features: np.ndarray,
    actor: MLP,
    rng: np.random.Generator,
    train_columns: np.ndarray,
    groups: int,
    group_size: int,
    dynamic_sampling: bool,
):
    pieces = []
    sampled_groups = 0
    filtered_groups = 0
    while len(pieces) < groups:
        needed = groups - len(pieces)
        pool_groups = needed * (2 if dynamic_sampling else 1)
        group_columns = rng.choice(train_columns, size=pool_groups, replace=True)
        episode_columns = np.repeat(group_columns, group_size)
        rollout = collect_episodes(env, features, actor, rng, episode_columns, None)
        sampled_groups += pool_groups
        for group in range(pool_groups):
            start = group * group_size
            stop = start + group_size
            group_returns = rollout["episode_returns"][start:stop]
            std = float(np.std(group_returns))
            if std < 1.0e-8:
                filtered_groups += 1
                if dynamic_sampling:
                    continue
                normalized = np.zeros(group_size, dtype=np.float64)
            else:
                normalized = (group_returns - float(np.mean(group_returns))) / (std + 1.0e-8)
            selected = (rollout["episode"] >= start) & (rollout["episode"] < stop)
            local_episode = rollout["episode"][selected] - start
            pieces.append(
                {
                    "states": rollout["states"][selected],
                    "actions": rollout["actions"][selected],
                    "old_logp": rollout["old_logp"][selected],
                    "advantages": normalized[local_episode],
                    "lengths": rollout["lengths"][start:stop],
                    "local_episode": local_episode,
                }
            )
            if len(pieces) == groups:
                break
        if (
            dynamic_sampling
            and sampled_groups >= groups * DYNAMIC_SAMPLING_MAX_MULTIPLIER
            and len(pieces) < groups
        ):
            raise NoInformativeGroups(groups, sampled_groups, filtered_groups)
    states = np.concatenate([piece["states"] for piece in pieces])
    actions = np.concatenate([piece["actions"] for piece in pieces])
    old_logp = np.concatenate([piece["old_logp"] for piece in pieces])
    advantages = np.concatenate([piece["advantages"] for piece in pieces])
    if dynamic_sampling:
        weights = np.ones(states.size, dtype=np.float64)
    else:
        weights = np.concatenate(
            [1.0 / piece["lengths"][piece["local_episode"]] for piece in pieces]
        )
    return {
        "states": states,
        "actions": actions,
        "old_logp": old_logp,
        "advantages": advantages,
        "targets": None,
        "weights": weights,
    }, {"sampled_groups": sampled_groups, "filtered_groups": filtered_groups}


def actor_update(
    env: Environment,
    features: np.ndarray,
    actor: MLP,
    optimizer: Adam,
    rollout,
    rng: np.random.Generator,
    epochs: int,
    minibatch: int,
    clip_low: float,
    clip_high: float,
    entropy_coef: float,
) -> float:
    states = rollout["states"]
    actions = rollout["actions"]
    old_logp = rollout["old_logp"]
    advantages = rollout["advantages"].astype(np.float64, copy=True)
    weights = rollout["weights"].astype(np.float64, copy=True)
    if float(np.std(advantages)) > 1.0e-8:
        advantages = (advantages - float(np.mean(advantages))) / (float(np.std(advantages)) + 1.0e-8)
    indices = np.arange(states.size)
    clipped = []
    for _epoch in range(epochs):
        rng.shuffle(indices)
        for offset in range(0, indices.size, minibatch):
            take = indices[offset : offset + minibatch]
            x, hidden, probs = masked_distribution(actor, features, env, states[take])
            selected = probs[np.arange(take.size), actions[take]]
            ratio = np.exp(np.log(np.maximum(selected, 1.0e-30)) - old_logp[take])
            advantage = advantages[take]
            active = ((advantage >= 0) & (ratio <= 1.0 + clip_high)) | (
                (advantage < 0) & (ratio >= 1.0 - clip_low)
            )
            local_weight = weights[take]
            local_weight /= float(np.sum(local_weight))
            coefficient = np.where(active, -advantage * ratio * local_weight, 0.0)
            grad_logits = -coefficient[:, None] * probs
            grad_logits[np.arange(take.size), actions[take]] += coefficient
            log_probs = np.log(np.maximum(probs, 1.0e-30))
            entropy = -np.sum(probs * log_probs, axis=1, keepdims=True)
            grad_logits += entropy_coef * local_weight[:, None] * probs * (log_probs + entropy)
            optimizer.step(actor.backward(x, hidden, grad_logits))
            clipped.append(float(np.mean(~active)))
    return float(np.mean(clipped)) if clipped else 0.0


def critic_update(
    features: np.ndarray,
    critic: MLP,
    optimizer: Adam,
    rollout,
    rng: np.random.Generator,
    epochs: int,
    minibatch: int,
) -> None:
    states = rollout["states"]
    targets = rollout["targets"]
    indices = np.arange(states.size)
    for _epoch in range(epochs):
        rng.shuffle(indices)
        for offset in range(0, indices.size, minibatch):
            take = indices[offset : offset + minibatch]
            x = features[states[take]]
            output, hidden = critic.forward(x)
            grad_output = 2.0 * (output[:, 0] - targets[take])[:, None] / take.size
            optimizer.step(critic.backward(x, hidden, grad_output))


def greedy_pipeline(env: Environment, features: np.ndarray, actor: MLP, column: int):
    state = int(env.root_state[column])
    pipeline: tuple[int, ...] = ()
    while len(pipeline) < MAX_DEPTH:
        _x, _hidden, probs = masked_distribution(
            actor, features, env, np.asarray([state], dtype=np.int32)
        )
        action = int(np.argmax(probs[0]))
        if action == STOP:
            break
        pipeline += (action,)
        if len(pipeline) == MAX_DEPTH:
            break
        state = int(env.next_state[state, action])
    if not pipeline:
        raise AssertionError("STOP must be masked at root")
    rate = env.rates[env.columns[column].key][pipeline]
    cost_ms = sum(
        env.eval_ms[env.columns[column].key][pipeline[:depth]]
        for depth in range(1, len(pipeline) + 1)
    )
    return pipeline, rate, cost_ms


def evaluate_columns(
    env: Environment,
    features: np.ndarray,
    actor: MLP,
    validation_columns: np.ndarray,
    algorithm: str,
    seed: int,
    fold: int,
) -> list[dict]:
    rows = []
    for col in validation_columns:
        column = env.columns[int(col)]
        pipeline, rate, proxy_eval_ms = greedy_pipeline(env, features, actor, int(col))
        baseline = column.baseline_rate
        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "fold": fold,
                "dataset": column.dataset,
                "column_index": column.index,
                "column_name": column.name,
                "logical_type": column.logical_type,
                "pipeline": " -> ".join(env.operators[action] for action in pipeline),
                "depth": len(pipeline),
                "proxy_rate": rate,
                "baseline_algorithm": column.baseline_algorithm,
                "baseline_rate": baseline,
                "gain_vs_baseline_pct": 100.0 * (baseline - rate) / baseline,
                "strictly_better_than_baseline": int(rate < baseline),
                "oracle_rate": env.oracle[col],
                "oracle_regret_pct": 100.0 * (rate - env.oracle[col]) / env.oracle[col],
                "beam4_rate": env.beam4[col],
                "gain_vs_beam4_pct": 100.0 * (env.beam4[col] - rate) / env.beam4[col],
                "proxy_search_eval_ms": proxy_eval_ms,
            }
        )
    return rows


def train_one(
    env: Environment,
    features: np.ndarray,
    train_columns: np.ndarray,
    validation_columns: np.ndarray,
    algorithm: str,
    seed: int,
    fold: int,
    args: argparse.Namespace,
):
    base_seed = seed + 1009 * fold
    init_rng = np.random.default_rng(base_seed)
    rng = np.random.default_rng(
        base_seed + {"ppo": 0, "grpo": 100_003, "dapo": 200_003}[algorithm]
    )
    # Paired seeds start every algorithm from the same actor parameters.
    actor = MLP(features.shape[1], args.hidden, 29, init_rng)
    actor_optimizer = Adam(actor.params, args.actor_lr)
    critic = None
    critic_optimizer = None
    if algorithm == "ppo":
        critic = MLP(features.shape[1], args.hidden, 1, init_rng)
        critic_optimizer = Adam(critic.params, args.critic_lr)
    training_rows = []
    started = time.perf_counter()
    completed_iterations = 0
    early_stop_reason = ""
    early_stop_sampled_groups = 0
    early_stop_filtered_groups = 0
    for iteration in range(1, args.iterations + 1):
        try:
            if algorithm == "ppo":
                rollout, sampling = collect_ppo(
                    env,
                    features,
                    actor,
                    critic,
                    rng,
                    train_columns,
                    args.episodes_per_iteration,
                    args.gae_lambda,
                )
                clip_low = args.clip_ratio
                clip_high = args.clip_ratio
            else:
                rollout, sampling = collect_group_relative(
                    env,
                    features,
                    actor,
                    rng,
                    train_columns,
                    args.groups_per_iteration,
                    args.group_size,
                    dynamic_sampling=algorithm == "dapo",
                )
                clip_low = args.clip_ratio
                clip_high = args.dapo_clip_high if algorithm == "dapo" else args.clip_ratio
        except NoInformativeGroups as exc:
            early_stop_reason = "dynamic_sampling_no_informative_groups"
            early_stop_sampled_groups = exc.sampled_groups
            early_stop_filtered_groups = exc.filtered_groups
            print(
                json.dumps(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "fold": fold,
                        "iteration": iteration,
                        "status": early_stop_reason,
                        "sampled_groups": exc.sampled_groups,
                        "filtered_groups": exc.filtered_groups,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break
        clip_fraction = actor_update(
            env,
            features,
            actor,
            actor_optimizer,
            rollout,
            rng,
            args.policy_epochs,
            args.minibatch,
            clip_low,
            clip_high,
            args.entropy_coef,
        )
        if algorithm == "ppo":
            critic_update(
                features,
                critic,
                critic_optimizer,
                rollout,
                rng,
                args.value_epochs,
                args.minibatch,
            )
        completed_iterations = iteration
        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            train_eval = evaluate_columns(env, features, actor, train_columns, algorithm, seed, fold)
            gains = np.asarray([row["gain_vs_baseline_pct"] for row in train_eval])
            regrets = np.asarray([row["oracle_regret_pct"] for row in train_eval])
            row = {
                "algorithm": algorithm,
                "seed": seed,
                "fold": fold,
                "iteration": iteration,
                "transitions": int(rollout["states"].size),
                "train_median_gain_vs_baseline_pct": float(np.median(gains)),
                "train_columns_better_than_baseline": int(np.sum(gains > 0)),
                "train_median_oracle_regret_pct": float(np.median(regrets)),
                "clip_fraction": clip_fraction,
                "sampled_groups": sampling["sampled_groups"],
                "filtered_groups": sampling["filtered_groups"],
                "elapsed_sec": time.perf_counter() - started,
            }
            training_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    evaluation = evaluate_columns(
        env, features, actor, validation_columns, algorithm, seed, fold
    )
    return evaluation, training_rows, actor, {
        "algorithm": algorithm,
        "seed": seed,
        "fold": fold,
        "requested_iterations": args.iterations,
        "completed_iterations": completed_iterations,
        "early_stop_reason": early_stop_reason,
        "early_stop_sampled_groups": early_stop_sampled_groups,
        "early_stop_filtered_groups": early_stop_filtered_groups,
    }


def self_check(env: Environment, folds: np.ndarray, args: argparse.Namespace) -> None:
    assert len(env.columns) >= 2
    assert len(env.operators) == 28
    assert env.state_col.size == len(env.columns) * (1 + 28 + 28 * 28)
    assert np.all(~env.valid[env.root_state, STOP])
    assert np.all(np.isfinite(env.rewards))
    for fold in range(args.folds):
        train = np.flatnonzero(folds != fold)
        validation = np.flatnonzero(folds == fold)
        assert train.size and validation.size and not set(train) & set(validation)
        features = build_state_features(env, train)
        rng = np.random.default_rng(7 + fold)
        actor = MLP(features.shape[1], 8, 29, rng)
        critic = MLP(features.shape[1], 8, 1, rng)
        rollout, _sampling = collect_ppo(env, features, actor, critic, rng, train, 16, 0.95)
        assert rollout["states"].size >= 16
        assert np.all(np.isfinite(rollout["advantages"]))
        # The dense reward telescopes to log(best-single/final-rate).
        for episode in range(16):
            selected = rollout["episode"] == episode
            states = rollout["states"][selected]
            actions = rollout["actions"][selected]
            col = int(env.state_col[states[0]])
            pipeline = tuple(int(action) for action in actions if action != STOP)
            final_rate = env.rates[env.columns[col].key][pipeline]
            expected = math.log(env.best_single[col] / final_rate)
            assert math.isclose(float(rollout["episode_returns"][episode]), expected, abs_tol=1.0e-10)
        states = rollout["states"]
        actions = rollout["actions"]
        assert np.all(env.valid[states, actions])
    print(
        f"SELF_CHECK ok columns={len(env.columns)} operators={len(env.operators)} "
        f"states={env.state_col.size} features={build_state_features(env, np.flatnonzero(folds != 0)).shape[1]}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="optional exact dataset names; omitted keeps the original all-dataset behavior",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--algorithms", nargs="+", choices=("ppo", "grpo", "dapo"), default=("ppo", "grpo", "dapo"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260825, 20260826, 20260827))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--episodes-per-iteration", type=int, default=512)
    parser.add_argument("--groups-per-iteration", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--value-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=3.0e-3)
    parser.add_argument("--critic-lr", type=float, default=3.0e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--dapo-clip-high", type=float, default=0.28)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2 or args.group_size < 2:
        raise ValueError("folds and group-size must be at least 2")
    env, data_inputs, operator_source = load_environment(
        args.repo_root.resolve(),
        args.measurement_root.resolve(),
        args.summary_csv.resolve(),
        args.baseline_csv.resolve(),
        tuple(args.datasets) if args.datasets else None,
    )
    if args.folds > len(env.columns):
        raise ValueError(f"folds={args.folds} exceeds selected columns={len(env.columns)}")
    folds = assign_folds(env.columns, args.folds)
    self_check(env, folds, args)
    if args.self_check_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --self-check-only is used")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    all_evaluation = []
    all_training = []
    run_status_rows = []
    fold_rows = []
    for position, column in enumerate(env.columns):
        fold_rows.append(
            {
                "dataset": column.dataset,
                "column_index": column.index,
                "column_name": column.name,
                "logical_type": column.logical_type,
                "fold": int(folds[position]),
            }
        )
    write_csv(output_dir / "fold_assignments.csv", fold_rows)
    started = time.perf_counter()
    for algorithm in args.algorithms:
        for seed in args.seeds:
            for fold in range(args.folds):
                train_columns = np.flatnonzero(folds != fold)
                validation_columns = np.flatnonzero(folds == fold)
                features = build_state_features(env, train_columns)
                evaluation, training, actor, run_status = train_one(
                    env,
                    features,
                    train_columns,
                    validation_columns,
                    algorithm,
                    seed,
                    fold,
                    args,
                )
                all_evaluation.extend(evaluation)
                all_training.extend(training)
                run_status_rows.append(run_status)
                write_csv(output_dir / "evaluation.csv", all_evaluation)
                write_csv(output_dir / "training_curve.csv", all_training)
                write_csv(output_dir / "run_status.csv", run_status_rows)
                np.savez_compressed(
                    output_dir / f"policy_{algorithm}_seed{seed}_fold{fold}.npz",
                    W1=actor.params[0],
                    b1=actor.params[1],
                    W2=actor.params[2],
                    b2=actor.params[3],
                )
    script_path = Path(__file__).resolve()
    manifests = sorted(args.measurement_root.resolve().glob("*/manifest.json"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "script": str(script_path),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "operator_source": str(operator_source),
        "operator_source_sha256": hashlib.sha256(operator_source.read_bytes()).hexdigest(),
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "protocol": {
            "split": (
                f"{args.folds}-fold held-out-column cross-validation; each column evaluated only by a policy "
                "trained without that column"
            ),
            "column_identity_features": False,
            "feature_normalization": "training columns only within each fold",
            "state_features": list(RAW_FEATURE_NAMES)
            + ["28 single-operator proxy probes", "current prefix rate/best-single", "depth", "prefix actions"],
            "reward": "sum_t log(previous_proxy_rate / next_proxy_rate) = log(best_single/final_proxy_rate)",
            "ppo": "clipped actor-critic with GAE",
            "grpo": "critic-free group-relative outcome advantages and sample-level loss",
            "dapo": "short-horizon adaptation: group-relative advantages, Clip-Higher 0.2/0.28, dynamic sampling, action-level loss",
            "dapo_overlong_reward": "not applicable because horizon is fixed at at most 3 actions",
            "dapo_dynamic_sampling_stop": (
                "evaluate the current policy if a 64-group informative batch cannot be filled "
                f"within {DYNAMIC_SAMPLING_MAX_MULTIPLIER}x the target group count"
            ),
            "evaluation": "greedy policy on held-out columns",
        },
        "claim_boundary": {
            "candidate_rate": "proxy conceptual transforms plus zlib, not a deployable codec",
            "proxy_search_eval_ms": "measured Python proxy evaluation time, not production encode/decode latency",
            "real_codec_cost": "not available until forward metadata and inverse decoding are implemented as a standalone codec",
            "baseline_rate": "best aligned lossless per-column CompressionBench result",
        },
        "inputs": {
            "measurement_manifests": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests
            },
            "summary_csv": {
                "path": str(args.summary_csv.resolve()),
                "sha256": hashlib.sha256(args.summary_csv.resolve().read_bytes()).hexdigest(),
            },
            "baseline_csv": {
                "path": str(args.baseline_csv.resolve()),
                "sha256": hashlib.sha256(args.baseline_csv.resolve().read_bytes()).hexdigest(),
            },
            "datasets": data_inputs,
        },
        "counts": {
            "columns": len(env.columns),
            "operators": len(env.operators),
            "states": int(env.state_col.size),
            "evaluation_rows": len(all_evaluation),
            "expected_evaluation_rows": len(args.algorithms) * len(args.seeds) * len(env.columns),
            "training_runs": len(run_status_rows),
            "early_stopped_runs": sum(bool(row["early_stop_reason"]) for row in run_status_rows),
        },
        "wall_sec": time.perf_counter() - started,
    }
    if manifest["counts"]["evaluation_rows"] != manifest["counts"]["expected_evaluation_rows"]:
        raise AssertionError(manifest["counts"])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT {output_dir}", flush=True)


if __name__ == "__main__":
    main()
