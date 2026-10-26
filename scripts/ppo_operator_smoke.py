#!/usr/bin/env python3
"""Tabular PPO smoke test on the exact depth-3 operator-search environment.

This is an implementation gate, not a generalization result.  The policy and
value function are tabular over (dataset column, operator prefix), and train and
evaluation use the same 20 columns.  Rates come from the previously measured
lossless exhaustive table, so PPO does not rerun codecs during optimization.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


STOP = 28


@dataclass
class Environment:
    columns: list[tuple[str, int, str]]
    operators: tuple[str, ...]
    rates: dict[tuple[str, int], dict[tuple[int, ...], float]]
    state_id: dict[tuple[tuple[str, int], tuple[int, ...]], int]
    state_depth: np.ndarray
    best_single: dict[tuple[str, int], float]
    oracle: dict[tuple[str, int], float]
    beam4: dict[tuple[str, int], float]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_environment(measurement_root: Path, summary_csv: Path) -> Environment:
    detail_paths = sorted(measurement_root.glob("*/pipeline_measurements.csv"))
    if len(detail_paths) != 4:
        raise AssertionError(f"expected four measurement shards, got {len(detail_paths)}")
    operator_names: set[str] = set()
    text_rows = []
    column_names = {}
    for path in detail_paths:
        for row in read_csv(path):
            if int(row["lossless"]) != 1 or row["status"] != "ok":
                continue
            pipeline = tuple(part.strip() for part in row["pipeline"].split(" -> "))
            operator_names.update(pipeline)
            key = (row["dataset"], int(row["column_index"]))
            column_names[key] = row["column_name"]
            text_rows.append((key, pipeline, float(row["rate"])))
    operators = tuple(sorted(operator_names))
    if len(operators) != 28:
        raise AssertionError(f"expected 28 operators, got {len(operators)}")
    operator_to_id = {name: index for index, name in enumerate(operators)}
    rates: dict[tuple[str, int], dict[tuple[int, ...], float]] = {}
    for key, pipeline_text, rate in text_rows:
        pipeline = tuple(operator_to_id[name] for name in pipeline_text)
        rates.setdefault(key, {})[pipeline] = rate
    if len(rates) != 20:
        raise AssertionError(f"expected 20 columns, got {len(rates)}")
    for key, values in rates.items():
        if len(values) != 22764:
            raise AssertionError(f"{key} has {len(values)} pipelines, expected 22764")

    summaries = {
        (row["dataset"], int(row["column_index"])): row
        for row in read_csv(summary_csv)
    }
    columns = [
        (dataset, column_index, column_names[(dataset, column_index)])
        for dataset, column_index in sorted(rates)
    ]
    best_single = {
        key: min(rate for pipeline, rate in values.items() if len(pipeline) == 1)
        for key, values in rates.items()
    }
    oracle = {key: min(values.values()) for key, values in rates.items()}
    beam4 = {key: float(summaries[key]["beam4_rate"]) for key in rates}

    state_id = {}
    depths = []
    for key in sorted(rates):
        prefixes = {()}
        prefixes.update(
            pipeline
            for pipeline in rates[key]
            if len(pipeline) in (1, 2)
        )
        for prefix in sorted(prefixes, key=lambda item: (len(item), item)):
            state_id[(key, prefix)] = len(depths)
            depths.append(len(prefix))
    expected_states = 20 * (1 + 28 + 28 * 28)
    if len(depths) != expected_states:
        raise AssertionError(f"expected {expected_states} states, got {len(depths)}")
    return Environment(
        columns=columns,
        operators=operators,
        rates=rates,
        state_id=state_id,
        state_depth=np.asarray(depths, dtype=np.int8),
        best_single=best_single,
        oracle=oracle,
        beam4=beam4,
    )


def masked_probs(logits: np.ndarray, depths: np.ndarray) -> np.ndarray:
    work = logits.copy()
    work[depths == 0, STOP] = -1.0e9
    work -= np.max(work, axis=1, keepdims=True)
    probs = np.exp(work)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


def sample_action(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(probs.size, p=probs))


def collect_rollouts(
    env: Environment,
    logits: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    episodes: int,
    gamma: float,
    gae_lambda: float,
):
    states = []
    actions = []
    old_logp = []
    old_values = []
    rewards = []
    episode_ranges = []
    column_keys = [(dataset, column_index) for dataset, column_index, _name in env.columns]

    for _episode in range(episodes):
        key = column_keys[int(rng.integers(len(column_keys)))]
        prefix: tuple[int, ...] = ()
        start = len(states)
        while True:
            sid = env.state_id[(key, prefix)]
            probs = masked_probs(logits[sid : sid + 1], env.state_depth[sid : sid + 1])[0]
            action = sample_action(rng, probs)
            states.append(sid)
            actions.append(action)
            old_logp.append(math.log(max(float(probs[action]), 1.0e-30)))
            old_values.append(float(values[sid]))
            if action == STOP:
                rewards.append(0.0)
                break
            new_prefix = prefix + (action,)
            previous_rate = env.best_single[key] if not prefix else env.rates[key][prefix]
            next_rate = env.rates[key][new_prefix]
            rewards.append(math.log(previous_rate / next_rate))
            prefix = new_prefix
            if len(prefix) == 3:
                break
        episode_ranges.append((start, len(states)))

    states_a = np.asarray(states, dtype=np.int32)
    actions_a = np.asarray(actions, dtype=np.int16)
    old_logp_a = np.asarray(old_logp, dtype=np.float64)
    old_values_a = np.asarray(old_values, dtype=np.float64)
    rewards_a = np.asarray(rewards, dtype=np.float64)
    advantages = np.zeros_like(rewards_a)
    returns = np.zeros_like(rewards_a)
    for start, end in episode_ranges:
        next_value = 0.0
        next_advantage = 0.0
        for index in range(end - 1, start - 1, -1):
            delta = rewards_a[index] + gamma * next_value - old_values_a[index]
            next_advantage = delta + gamma * gae_lambda * next_advantage
            advantages[index] = next_advantage
            returns[index] = next_advantage + old_values_a[index]
            next_value = old_values_a[index]
    return states_a, actions_a, old_logp_a, old_values_a, advantages, returns


def ppo_update(
    env: Environment,
    logits: np.ndarray,
    values: np.ndarray,
    rollout,
    rng: np.random.Generator,
    *,
    epochs: int,
    minibatch: int,
    clip_ratio: float,
    actor_lr: float,
    critic_lr: float,
    entropy_coef: float,
):
    states, actions, old_logp, _old_values, advantages, returns = rollout
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    indices = np.arange(states.size)
    clip_fraction_values = []
    for _epoch in range(epochs):
        rng.shuffle(indices)
        for offset in range(0, indices.size, minibatch):
            take = indices[offset : offset + minibatch]
            sid = states[take]
            action = actions[take]
            advantage = advantages[take]
            target = returns[take]
            depths = env.state_depth[sid]
            probs = masked_probs(logits[sid], depths)
            new_logp = np.log(np.maximum(probs[np.arange(take.size), action], 1.0e-30))
            ratio = np.exp(new_logp - old_logp[take])
            active = ((advantage >= 0) & (ratio <= 1.0 + clip_ratio)) | (
                (advantage < 0) & (ratio >= 1.0 - clip_ratio)
            )
            coefficient = np.where(active, -advantage * ratio, 0.0) / take.size
            sample_grad = -coefficient[:, None] * probs
            sample_grad[np.arange(take.size), action] += coefficient
            log_probs = np.log(np.maximum(probs, 1.0e-30))
            entropy = -np.sum(probs * log_probs, axis=1, keepdims=True)
            sample_grad += (
                entropy_coef
                * probs
                * (log_probs + entropy)
                / take.size
            )
            unique, inverse = np.unique(sid, return_inverse=True)
            gradient = np.zeros((unique.size, logits.shape[1]), dtype=np.float64)
            np.add.at(gradient, inverse, sample_grad)
            logits[unique] -= actor_lr * np.clip(gradient, -1.0, 1.0)
            logits[unique] = np.clip(logits[unique], -20.0, 20.0)

            value_error = 2.0 * (values[sid] - target) / take.size
            value_gradient = np.zeros(unique.size, dtype=np.float64)
            np.add.at(value_gradient, inverse, value_error)
            values[unique] -= critic_lr * np.clip(value_gradient, -1.0, 1.0)
            clip_fraction_values.append(float(np.mean(~active)))
    return float(np.mean(clip_fraction_values)) if clip_fraction_values else 0.0


def evaluate(env: Environment, logits: np.ndarray):
    rows = []
    for dataset, column_index, column_name in env.columns:
        key = (dataset, column_index)
        prefix: tuple[int, ...] = ()
        while len(prefix) < 3:
            sid = env.state_id[(key, prefix)]
            probs = masked_probs(logits[sid : sid + 1], env.state_depth[sid : sid + 1])[0]
            action = int(np.argmax(probs))
            if action == STOP:
                break
            prefix = prefix + (action,)
        if not prefix:
            raise AssertionError("STOP is masked at the root")
        rate = env.rates[key][prefix]
        oracle = env.oracle[key]
        beam4 = env.beam4[key]
        rows.append(
            {
                "dataset": dataset,
                "column_index": column_index,
                "column_name": column_name,
                "ppo_pipeline": " -> ".join(env.operators[action] for action in prefix),
                "ppo_depth": len(prefix),
                "ppo_rate": rate,
                "oracle_rate": oracle,
                "ppo_oracle_regret_pct": 100.0 * (rate - oracle) / oracle,
                "beam4_rate": beam4,
                "ppo_gain_vs_beam4_pct": 100.0 * (beam4 - rate) / beam4,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=4096)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=0.5)
    parser.add_argument("--critic-lr", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = load_environment(args.measurement_root.resolve(), args.summary_csv.resolve())
    rng = np.random.default_rng(args.seed)
    logits = np.zeros((len(env.state_depth), 29), dtype=np.float64)
    values = np.zeros(len(env.state_depth), dtype=np.float64)
    training_rows = []
    for iteration in range(1, args.iterations + 1):
        rollout = collect_rollouts(
            env,
            logits,
            values,
            rng,
            args.episodes_per_iteration,
            gamma=1.0,
            gae_lambda=0.95,
        )
        clip_fraction = ppo_update(
            env,
            logits,
            values,
            rollout,
            rng,
            epochs=args.ppo_epochs,
            minibatch=args.minibatch,
            clip_ratio=args.clip_ratio,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            entropy_coef=args.entropy_coef,
        )
        if iteration == 1 or iteration % 5 == 0 or iteration == args.iterations:
            evaluation = evaluate(env, logits)
            regrets = np.asarray([row["ppo_oracle_regret_pct"] for row in evaluation])
            gains = np.asarray([row["ppo_gain_vs_beam4_pct"] for row in evaluation])
            row = {
                "iteration": iteration,
                "episodes": iteration * args.episodes_per_iteration,
                "transitions_last_batch": len(rollout[0]),
                "median_oracle_regret_pct": float(np.median(regrets)),
                "max_oracle_regret_pct": float(np.max(regrets)),
                "columns_better_than_beam4": int(np.sum(gains > 0)),
                "median_gain_vs_beam4_pct": float(np.median(gains)),
                "clip_fraction": clip_fraction,
                "elapsed_sec": time.perf_counter() - started,
            }
            training_rows.append(row)
            write_csv(output_dir / "training_curve.csv", training_rows)
            print(json.dumps(row, sort_keys=True), flush=True)

    evaluation = evaluate(env, logits)
    write_csv(output_dir / "evaluation.csv", evaluation)
    np.savez_compressed(output_dir / "policy.npz", logits=logits, values=values)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": vars(args) | {
            "measurement_root": str(args.measurement_root.resolve()),
            "summary_csv": str(args.summary_csv.resolve()),
            "output_dir": str(output_dir),
        },
        "environment": {
            "columns": len(env.columns),
            "operators": len(env.operators),
            "states": len(env.state_depth),
            "horizon": 3,
            "stop_action": True,
            "reward": "log(previous_rate / next_rate); start reference is best single rate",
            "train_eval_split": "same 20 columns; smoke test only",
        },
        "wall_sec": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
