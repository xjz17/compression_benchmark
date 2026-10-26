#!/usr/bin/env python3
"""Small PPO / GRPO-style policies for the executable canonical206 environment.

The environment owns real codec execution, legality, caching, and timing.  This
module only chooses action IDs and optimizes the policy.  The GRPO-style variant
uses within-environment group-relative rewards and the PPO clipped objective,
with no reference-policy KL term (beta=0).  It is not a pretrained codec selector.

Required environment interface::

    max_steps: int
    action_count: int
    stop_id: int
    features(prefix: tuple[int, ...]) -> one-dimensional numpy array
    legal(prefix) -> tuple[int, ...]       # active IDs in reserved slots, including legal STOP
    transition(prefix, action) -> new_prefix | None  # None means invalid
    reward(prefix) -> float                # terminal reward, larger is better

The caller maintains any incumbent in the environment and decides whether to
report the proposal or the best pipeline measured during the search.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Protocol, Sequence

import numpy as np

from search_physical_operator_variants_v2 import (
    Adam,
    MLP,
    actor_update,
    critic_update,
    distribution,
)


import primitive_operator_action_runtime as primitive
import canonical156_parameter_policy as conditional
from canonical182_policy_checkpoint import bind, validate, save as save_policy, load as load_policy

STOP = primitive.STOP_ACTION_ID
ACTION_COUNT = STOP + 1
INVALID_REWARD = -10.0
Prefix = tuple[int, ...]


class Environment(Protocol):
    max_steps: int

    def features(self, prefix: Prefix) -> np.ndarray: ...
    def legal(self, prefix: Prefix) -> tuple[int, ...]: ...
    def transition(self, prefix: Prefix, action: int) -> Prefix | None: ...
    def reward(self, prefix: Prefix) -> float: ...


def _features(env: Environment, prefix: Prefix, expected_dim: int) -> np.ndarray:
    value = np.asarray(env.features(prefix), dtype=np.float64)
    if value.shape != (expected_dim,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid features for {prefix}: shape={value.shape}")
    return value


def _mask(env: Environment, prefix: Prefix) -> np.ndarray:
    legal = tuple(env.legal(prefix))
    if not legal or any(not 0 <= action < env.action_count for action in legal):
        raise ValueError(f"invalid legal action IDs for {prefix}: {legal}")
    result = np.zeros(env.action_count, dtype=bool)
    result[list(legal)] = True
    return result


def _reward(env: Environment, prefix: Prefix) -> float:
    reward = float(env.reward(prefix))
    if not np.isfinite(reward):
        raise ValueError(f"non-finite terminal reward for {prefix}")
    return float(np.clip(reward, -10.0, 10.0))


def _sample(probabilities: np.ndarray, masks: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """Sample only legal IDs, including under floating-point roundoff."""
    actions = np.empty(probabilities.shape[0], dtype=np.int16)
    for row in range(probabilities.shape[0]):
        legal = np.flatnonzero(masks[row])
        weights = probabilities[row, legal]
        actions[row] = int(rng.choice(legal, p=weights / weights.sum()))
    return actions


def _collect_rollouts(
    actor: MLP,
    critic: MLP | None,
    envs: Sequence[Environment],
    env_indices: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    episode_count = len(env_indices)
    for env in envs:
        validate(actor, env)
    feature_dim = actor.params[0].shape[0]
    horizon = max(int(envs[int(index)].max_steps) for index in env_indices)
    feature_tensor = np.zeros((episode_count, horizon, feature_dim), dtype=np.float64)
    legal_tensor = np.zeros((episode_count, horizon, actor.params[2].shape[1]), dtype=bool)
    action_tensor = np.full((episode_count, horizon), -1, dtype=np.int16)
    logp_tensor = np.zeros((episode_count, horizon), dtype=np.float64)
    parameter_tensor = np.full((episode_count, horizon), -1, dtype=np.int16)
    parameter_legal_tensor = np.zeros((episode_count, horizon, len(conditional.CANDIDATES)), dtype=bool)
    reward_tensor = np.zeros((episode_count, horizon), dtype=np.float64)
    value_tensor = np.zeros((episode_count, horizon), dtype=np.float64)
    active_tensor = np.zeros((episode_count, horizon), dtype=bool)
    active = np.ones(episode_count, dtype=bool)
    prefixes: list[Prefix] = [() for _ in range(episode_count)]
    invalid = np.zeros(episode_count, dtype=bool)
    stopped = np.zeros(episode_count, dtype=bool)

    for step in range(horizon):
        episodes = np.flatnonzero(active)
        if not episodes.size:
            break
        features = np.vstack([
            _features(envs[int(env_indices[e])], prefixes[e], feature_dim)
            for e in episodes
        ])
        masks = np.vstack([
            _mask(envs[int(env_indices[e])], prefixes[e]) for e in episodes
        ])
        _hidden, probabilities, _logits = distribution(actor, features, masks)
        actions = _sample(probabilities, masks, rng)
        feature_tensor[episodes, step] = features
        legal_tensor[episodes, step] = masks
        action_tensor[episodes, step] = actions
        logp_tensor[episodes, step] = np.log(np.maximum(
            probabilities[np.arange(episodes.size), actions], 1.0e-30
        ))
        active_tensor[episodes, step] = True
        if critic is not None:
            value_tensor[episodes, step] = critic.forward(features)[0][:, 0]

        for local, episode in enumerate(episodes):
            env = envs[int(env_indices[episode])]
            action = int(actions[local])
            if action == env.stop_id:
                stopped[episode] = True
                reward_tensor[episode, step] = _reward(env, prefixes[episode])
                active[episode] = False
                continue
            parameter_id, parameter_logp, parameters = conditional.choose(actor, env, prefixes[episode], action, features[local], rng)
            parameter_tensor[episode, step] = parameter_id
            if parameter_id >= 0:
                parameter_legal_tensor[episode, step] = conditional.legal_mask(env, prefixes[episode], action)
            logp_tensor[episode, step] += parameter_logp
            next_prefix = (env.transition(prefixes[episode], action, parameters=parameters) if parameters is not None else env.transition(prefixes[episode], action))
            if next_prefix is None:
                invalid[episode] = True
                reward_tensor[episode, step] = INVALID_REWARD
                active[episode] = False
                continue
            prefixes[episode] = next_prefix
            if step + 1 >= env.max_steps:
                reward_tensor[episode, step] = _reward(env, prefixes[episode])
                active[episode] = False

    if np.any(active):
        raise AssertionError("rollout did not terminate within its environment horizon")
    flat_episode, _flat_step = np.nonzero(active_tensor)
    return {
        "features": feature_tensor[active_tensor],
        "legal": legal_tensor[active_tensor],
        "actions": action_tensor[active_tensor],
        "old_logp": logp_tensor[active_tensor],
        "parameter_actions": parameter_tensor[active_tensor],
        "parameter_legal": parameter_legal_tensor[active_tensor],
        "old_values": value_tensor[active_tensor],
        "rewards": reward_tensor[active_tensor],
        "episode": flat_episode.astype(np.int32),
        "lengths": active_tensor.sum(axis=1),
        "episode_returns": reward_tensor.sum(axis=1),
        "active": active_tensor,
        "reward_matrix": reward_tensor,
        "value_matrix": value_tensor,
        "pipelines": prefixes,
        "invalid": invalid,
        "stopped": stopped,
        "weights": np.ones(int(active_tensor.sum()), dtype=np.float64),
    }


def _ppo_advantages(rollout: dict[str, Any], gae_lambda: float) -> None:
    advantages = np.zeros_like(rollout["reward_matrix"])
    targets = np.zeros_like(advantages)
    for episode, length in enumerate(rollout["lengths"]):
        next_advantage = next_value = 0.0
        for step in range(int(length) - 1, -1, -1):
            value = rollout["value_matrix"][episode, step]
            delta = rollout["reward_matrix"][episode, step] + next_value - value
            next_advantage = delta + gae_lambda * next_advantage
            advantages[episode, step] = next_advantage
            targets[episode, step] = next_advantage + value
            next_value = value
    rollout["advantages"] = advantages[rollout["active"]]
    rollout["targets"] = targets[rollout["active"]]


def _grpo_advantages(rollout: dict[str, Any], group_size: int) -> int:
    returns = rollout["episode_returns"].reshape(-1, group_size)
    deviations = returns.std(axis=1, keepdims=True)
    normalized = np.where(
        deviations > 1.0e-8,
        (returns - returns.mean(axis=1, keepdims=True)) / (deviations + 1.0e-8),
        0.0,
    ).reshape(-1)
    rollout["advantages"] = normalized[rollout["episode"]]
    rollout["targets"] = None
    return int(np.sum(deviations[:, 0] > 1.0e-8))


def train(
    method: str,
    envs: Sequence[Environment],
    seed: int,
    iterations: int = 12,
    episodes: int = 64,
    group_size: int = 8,
    hidden: int = 64,
    lr: float = 0.001,
    epochs: int = 4,
    *,
    minibatch: int = 256,
    clip_ratio: float = 0.2,
    entropy_coefficient: float = 0.01,
    gae_lambda: float = 0.95,
    value_epochs: int | None = None,
    verbose: bool = False,
) -> tuple[MLP, list[dict[str, Any]]]:
    """Train with exactly ``iterations * episodes`` terminal rollouts.

    PPO samples environments independently. GRPO samples one environment for
    each ``group_size`` trajectories so its comparisons use the same source.
    Actor initialization is identical across methods given the same seed.
    Environment executions, including feature computation, occur inside the
    reported training wall time. No evaluation rollouts are added implicitly.
    """
    method = method.lower()
    if method not in {"ppo", "grpo"}:
        raise ValueError("method must be 'ppo' or 'grpo'")
    if not envs or min(iterations, episodes, group_size, hidden, epochs, minibatch) < 1:
        raise ValueError("non-empty environments and positive budgets are required")
    if method == "grpo" and episodes % group_size:
        raise ValueError("GRPO episodes must be divisible by group_size")
    if not 0 <= gae_lambda <= 1 or not 0 < clip_ratio < 1 or lr <= 0:
        raise ValueError("invalid optimization hyperparameters")
    value_epochs = epochs if value_epochs is None else value_epochs
    if value_epochs < 1 or any(env.max_steps < 1 for env in envs):
        raise ValueError("value_epochs and environment horizons must be positive")

    started = time.perf_counter()
    initial = np.asarray(envs[0].features(()), dtype=np.float64)
    if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("root features must be a finite one-dimensional array")
    for env in envs:
        _features(env, (), initial.size)
    initialization_rng = np.random.default_rng(seed)
    sampling_rng = np.random.default_rng(seed + 1_000_003)
    update_rng = np.random.default_rng(seed + 2_000_003)
    actor = bind(MLP(initial.size, hidden, envs[0].action_count, initialization_rng), envs[0])
    for env in envs:
        validate(actor, env)
    actor_optimizer = conditional.optimizers(actor, lr)
    critic = MLP(initial.size, hidden, 1, initialization_rng) if method == "ppo" else None
    critic_optimizer = Adam(critic.params, lr) if critic is not None else None
    trace: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        iteration_started = time.perf_counter()
        if method == "grpo":
            groups = sampling_rng.integers(len(envs), size=episodes // group_size)
            env_indices = np.repeat(groups, group_size)
        else:
            env_indices = sampling_rng.integers(len(envs), size=episodes)
        rollout_started = time.perf_counter()
        rollout = _collect_rollouts(actor, critic, envs, env_indices, sampling_rng)
        rollout_seconds = time.perf_counter() - rollout_started
        informative_groups = 0
        if method == "ppo":
            _ppo_advantages(rollout, gae_lambda)
        else:
            informative_groups = _grpo_advantages(rollout, group_size)
        update_started = time.perf_counter()
        parameter_metrics = conditional.update(actor, actor_optimizer, rollout, epochs,
            clip_ratio, entropy_coefficient, rng=update_rng, minibatch=minibatch,
            normalize_advantages=True)
        clip_fraction = parameter_metrics['clip_fraction']
        if critic is not None and critic_optimizer is not None:
            critic_update(
                critic, critic_optimizer, rollout, update_rng, value_epochs, minibatch,
            )
        update_seconds = time.perf_counter() - update_started
        for model in (actor, getattr(actor, "parameter_head", None), critic):
            if model is not None and any(not np.all(np.isfinite(p)) for p in model.params):
                raise FloatingPointError(f"non-finite {method} parameters at iteration {iteration}")
        returns = rollout["episode_returns"]
        row = {
            "method": method,
            "seed": seed,
            "iteration": iteration,
            "episodes": episodes,
            "cumulative_episodes": iteration * episodes,
            "transitions": int(rollout["actions"].size),
            "mean_return": float(returns.mean()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "std_return": float(returns.std()),
            "invalid_episodes": int(rollout["invalid"].sum()),
            "stop_episodes": int(rollout["stopped"].sum()),
            "unique_proposals": len(set(zip(map(int, env_indices), rollout["pipelines"]))),
            "informative_groups": informative_groups,
            "clip_fraction": clip_fraction,
            "parameter_decisions": parameter_metrics["parameter_decisions"],
            "parameter_catalog_size": len(conditional.CANDIDATES),
            "rollout_seconds": rollout_seconds,
            "update_seconds": update_seconds,
            "iteration_seconds": time.perf_counter() - iteration_started,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(row)
        if verbose:
            print(json.dumps(row, sort_keys=True), flush=True)
    return actor, trace


def propose(actor: MLP, env: Environment, rng: np.random.Generator,
            greedy: bool = False) -> Prefix:
    """Execute one sampled or greedy proposal and return its last valid prefix.

    An invalid action ends the attempt. STOP is policy control and is never
    passed to the codec. The caller may compare this prefix with env.best_prefix.
    """
    validate(actor, env)
    prefix: Prefix = ()
    for _step in range(env.max_steps):
        features = _features(env, prefix, actor.params[0].shape[0])[None, :]
        masks = _mask(env, prefix)[None, :]
        _hidden, probabilities, _logits = distribution(actor, features, masks)
        action = (int(np.argmax(probabilities[0])) if greedy
                  else int(_sample(probabilities, masks, rng)[0]))
        if action == env.stop_id:
            return prefix
        _pid, _plogp, parameters = conditional.choose(actor, env, prefix, action, features[0], rng, greedy)
        next_prefix = env.transition(prefix, action, parameters=parameters) if parameters is not None else env.transition(prefix, action)
        if next_prefix is None:
            return prefix
        prefix = next_prefix
    return prefix
