#!/usr/bin/env python3
"""On-policy GRPO for one column's dynamic, executable recipe/option menu.

This is online per-column adaptation, not held-out generalization. The caller
owns codecs, measurement budgets and the best measured incumbent. A terminal
sample is NOT guaranteed to beat that incumbent. Codec/budget exceptions are
deliberately propagated, including during an incomplete rollout group.

Unlike the older shared actor_update, this implementation never re-centres
group-relative advantages across tokens: unequal trajectory lengths must not
turn a positive within-group advantage negative. Each trajectory contributes
total weight one, with 1 / length assigned to each real action. Full-batch
updates preserve these exact episode weights.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from search_physical_operator_variants_v2 import Adam, MLP, distribution
import canonical156_parameter_policy as conditional
from canonical182_policy_paths import PolicyPath


from canonical182_policy_checkpoint import bind, validate, save as save_policy, load as load_policy

Prefix = tuple[int, ...]
INVALID_REWARD = -10.0


class Environment(Protocol):
    action_count: int
    stop_id: int
    max_steps: int

    def features(self, prefix: Prefix) -> np.ndarray: ...
    def legal(self, prefix: Prefix) -> tuple[int, ...]: ...
    def transition(self, prefix: Prefix, action: int) -> Prefix | None: ...
    def reward(self, prefix: Prefix) -> float: ...


def _stop_id(env: Environment) -> int:
    for name in ("stop_id", "stop", "STOP"):
        if hasattr(env, name):
            value = int(getattr(env, name))
            if 0 <= value < int(env.action_count):
                return value
            raise ValueError("STOP is outside the dynamic action space")
    raise ValueError("environment must expose stop_id (or stop / STOP)")


def _features(env: Environment, prefix: Prefix, expected_dim: int | None = None) -> np.ndarray:
    value = np.asarray(env.features(prefix), dtype=np.float64)
    if (value.ndim != 1 or not value.size or not np.all(np.isfinite(value))
            or (expected_dim is not None and value.size != expected_dim)):
        raise ValueError(f"invalid features for {prefix}: shape={value.shape}")
    return value


def _mask(env: Environment, prefix: Prefix) -> np.ndarray:
    legal = tuple(env.legal(prefix))
    count = int(env.action_count)
    if (not legal or len(set(legal)) != len(legal)
            or any(not isinstance(a, (int, np.integer)) or not 0 <= a < count for a in legal)):
        raise ValueError(f"invalid legal action menu for {prefix}: {legal}")
    result = np.zeros(count, dtype=bool)
    result[list(legal)] = True
    return result


def _reward(env: Environment, prefix: Prefix) -> float:
    value = float(env.reward(prefix))
    if not np.isfinite(value):
        raise ValueError(f"non-finite reward for {prefix}")
    # Do not clip absolute rewards: that would destroy translation invariance.
    return value


def initialize_policy(env: Environment, seed: int, hidden: int = 64) -> MLP:
    if int(env.action_count) < 1 or int(env.max_steps) < 1 or hidden < 1:
        raise ValueError("positive action count, horizon and hidden size required")
    _stop_id(env)
    initial = _features(env, ())
    return bind(MLP(initial.size, hidden, int(env.action_count), np.random.default_rng(seed)), env)


def _validate_actor(actor: MLP, env: Environment) -> None:
    validate(actor, env)
    if int(env.max_steps) < 1:
        raise ValueError("environment horizon must be positive")
    _stop_id(env)
    if actor.params[2].shape[1] != int(env.action_count):
        raise ValueError("actor output count does not match environment")
    _features(env, (), actor.params[0].shape[0])


def group_advantages(returns: Sequence[float] | np.ndarray, group_size: int,
                     epsilon: float = 1.0e-8) -> np.ndarray:
    """Population-std normalized returns, independently for each same-root group."""
    values = np.asarray(returns, dtype=np.float64)
    if (values.ndim != 1 or not values.size or group_size < 2
            or values.size % group_size or not np.all(np.isfinite(values))):
        raise ValueError("finite returns and complete groups of at least two required")
    grouped = values.reshape(-1, group_size)
    centered = grouped - grouped.mean(axis=1, keepdims=True)
    std = grouped.std(axis=1, keepdims=True)
    return np.divide(centered, std + epsilon,
                     out=np.zeros_like(centered), where=std > epsilon).reshape(-1)


def _collect(env: Environment, actor: MLP | None, episodes: int,
             rng: np.random.Generator) -> dict[str, Any]:
    """Sample fixed-policy trajectories; do not catch environment exceptions."""
    stop = _stop_id(env)
    feature_dim = actor.params[0].shape[0] if actor is not None else None
    xs, masks, actions, logps, episode_ids = [], [], [], [], []
    parameter_actions = []
    parameter_legal = []
    lengths, returns, prefixes, invalids, stopped = [], [], [], [], []
    for episode in range(episodes):
        prefix: Prefix = ()
        invalid = did_stop = False
        length = 0
        for _step in range(int(env.max_steps)):
            x = _features(env, prefix, feature_dim)
            mask = _mask(env, prefix)
            legal = np.flatnonzero(mask)
            if actor is None:
                probabilities = mask.astype(np.float64) / len(legal)
            else:
                probabilities = distribution(actor, x[None, :], mask[None, :])[1][0]
            weights = probabilities[legal]
            action = int(rng.choice(legal, p=weights / weights.sum()))
            xs.append(x)
            masks.append(mask)
            actions.append(action)
            logps.append(float(np.log(max(probabilities[action], 1.0e-30))))
            episode_ids.append(episode)
            length += 1
            if action == stop:
                parameter_actions.append(-1)
                parameter_legal.append(np.zeros(len(conditional.CANDIDATES), dtype=bool))
                did_stop = True
                break
            parameter_id, parameter_logp, parameters = conditional.choose(actor, env, prefix, action, x, rng)
            parameter_actions.append(parameter_id)
            parameter_legal.append(conditional.legal_mask(env, prefix, action))
            logps[-1] += parameter_logp
            next_prefix = env.transition(prefix, action, parameters=parameters) if parameters is not None else env.transition(prefix, action)
            if next_prefix is None:
                invalid = True
                break
            prefix = next_prefix
        if invalid:
            invalid_hook = getattr(env, "reward_on_invalid", None)
            terminal_reward = (float(invalid_hook(prefix)) if callable(invalid_hook)
                               else float(getattr(env, "invalid_reward", INVALID_REWARD)))
        else:
            terminal_reward = _reward(env, prefix)
        if not np.isfinite(terminal_reward):
            raise ValueError("invalid terminal reward")
        lengths.append(length)
        returns.append(terminal_reward)
        prefixes.append(prefix)
        invalids.append(invalid)
        stopped.append(did_stop)
    length_array = np.asarray(lengths, dtype=np.int32)
    ep = np.asarray(episode_ids, dtype=np.int32)
    return {
        "features": np.asarray(xs, dtype=np.float64),
        "legal": np.asarray(masks, dtype=bool),
        "actions": np.asarray(actions, dtype=np.int32),
        "old_logp": np.asarray(logps, dtype=np.float64),
        "parameter_actions": np.asarray(parameter_actions, dtype=np.int32),
        "parameter_legal": np.asarray(parameter_legal, dtype=bool),
        "episode": ep,
        "lengths": length_array,
        "weights": 1.0 / length_array[ep],
        "episode_returns": np.asarray(returns, dtype=np.float64),
        "pipelines": prefixes,
        "invalid": np.asarray(invalids, dtype=bool),
        "stopped": np.asarray(stopped, dtype=bool),
    }


def _objective(actor: MLP, rollout: dict[str, Any], clip_ratio: float,
               entropy_coefficient: float) -> dict[str, float]:
    if hasattr(actor, 'parameter_head'):
        return conditional.objective(actor, rollout, clip_ratio, entropy_coefficient)
    probabilities = distribution(actor, rollout["features"], rollout["legal"])[1]
    selected = probabilities[np.arange(len(rollout["actions"])), rollout["actions"]]
    ratios = np.exp(np.log(np.maximum(selected, 1.0e-30)) - rollout["old_logp"])
    advantages = rollout["advantages"]
    weights = rollout["weights"] / rollout["weights"].sum()
    surrogate = np.minimum(ratios * advantages,
                           np.clip(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-30)), axis=1)
    policy_loss = float(-np.sum(weights * surrogate))
    entropy_mean = float(np.sum(weights * entropy))
    return {"loss": policy_loss - entropy_coefficient * entropy_mean,
            "policy_loss": policy_loss, "entropy": entropy_mean,
            "approx_kl": float(np.sum(weights * ((ratios - 1.0) - np.log(ratios))))}


def _actor_update(actor: MLP, optimizer: Adam, rollout: dict[str, Any], epochs: int,
                  clip_ratio: float, entropy_coefficient: float) -> dict[str, Any]:
    """Clipped on-policy GRPO, no second/token-level advantage normalization."""
    if hasattr(actor, 'parameter_head'):
        return conditional.update(actor, optimizer, rollout, epochs, clip_ratio, entropy_coefficient)
    before = _objective(actor, rollout, clip_ratio, entropy_coefficient)
    features, masks, actions = rollout["features"], rollout["legal"], rollout["actions"]
    advantages = rollout["advantages"]
    weights = rollout["weights"] / rollout["weights"].sum()
    clip_fractions = []
    for _epoch in range(epochs):
        hidden, probabilities, _ = distribution(actor, features, masks)
        selected = probabilities[np.arange(actions.size), actions]
        ratios = np.exp(np.log(np.maximum(selected, 1.0e-30)) - rollout["old_logp"])
        active = (((advantages >= 0) & (ratios <= 1.0 + clip_ratio))
                  | ((advantages < 0) & (ratios >= 1.0 - clip_ratio)))
        coefficient = np.where(active, -advantages * ratios * weights, 0.0)
        gradient = -coefficient[:, None] * probabilities
        gradient[np.arange(actions.size), actions] += coefficient
        log_probabilities = np.log(np.maximum(probabilities, 1.0e-30))
        entropy = -np.sum(probabilities * log_probabilities, axis=1, keepdims=True)
        gradient += (entropy_coefficient * weights[:, None] * probabilities
                     * (log_probabilities + entropy))
        optimizer.step(actor.backward(features, hidden, gradient))
        clip_fractions.append(float(np.sum(weights * ~active)))
    after = _objective(actor, rollout, clip_ratio, entropy_coefficient)
    return {"loss_before": before["loss"], "loss_after": after["loss"],
            "policy_loss_before": before["policy_loss"],
            "policy_loss_after": after["policy_loss"],
            "entropy_before": before["entropy"], "entropy_after": after["entropy"],
            "approx_kl_after": after["approx_kl"],
            "clip_fraction": float(np.mean(clip_fractions)), "optimizer_steps": epochs}


def _warm_start(actor: MLP, optimizer: Adam, env: Environment,
                paths: Sequence[Prefix], epochs: int) -> dict[str, Any]:
    started = time.perf_counter()
    xs, masks, actions = [], [], []
    parameter_targets = []
    parameter_masks = []
    stop = _stop_id(env)
    for path in paths:
        if len(path) > env.max_steps or stop in path:
            raise ValueError("warm-start paths must contain only non-STOP option IDs")
        prefix: Prefix = ()
        path = PolicyPath(path)
        target_actions = list(path)
        if len(path) < env.max_steps:
            target_actions.append(stop)
        for step, action in enumerate(target_actions):
            mask = _mask(env, prefix)
            if not 0 <= action < env.action_count or not mask[action]:
                raise ValueError(f"illegal warm-start action {action} after {prefix}")
            xs.append(_features(env, prefix, actor.params[0].shape[0]))
            masks.append(mask)
            actions.append(action)
            supplied = path.parameters[step] if step < len(path) else {}
            candidates = tuple(getattr(env, 'parameter_choices', lambda *_: ())(prefix, action))
            target = (conditional.CANDIDATES.index(supplied)
                      if supplied in conditional.CANDIDATES and supplied in candidates else -1)
            parameter_targets.append(target)
            parameter_masks.append(conditional.legal_mask(env, prefix, action))
            if action != stop:
                following = env.transition(prefix, action, parameters=supplied) if supplied else env.transition(prefix, action)
                if following is None:
                    raise ValueError(f"invalid warm-start path {path}")
                prefix = PolicyPath(following)
    if not actions:
        raise ValueError("non-empty warm-start token set required")
    xs_array, masks_array = np.asarray(xs), np.asarray(masks)
    targets = np.asarray(actions, dtype=np.int32)
    parameter_targets = np.asarray(parameter_targets, dtype=np.int32)
    rows = np.flatnonzero(parameter_targets >= 0)
    base_optimizer, parameter_optimizer = optimizer if isinstance(optimizer, tuple) else (optimizer, None)
    for _epoch in range(epochs):
        hidden, probabilities, _ = distribution(actor, xs_array, masks_array)
        gradient = probabilities.copy()
        gradient[np.arange(len(targets)), targets] -= 1.0
        base_optimizer.step(actor.backward(xs_array, hidden, gradient / len(targets)))
        if len(rows) and parameter_optimizer is not None:
            hidden, probabilities, _ = distribution(actor.parameter_head, xs_array[rows], np.asarray(parameter_masks)[rows])
            gradient = probabilities.copy()
            gradient[np.arange(len(rows)), parameter_targets[rows]] -= 1.0
            parameter_optimizer.step(actor.parameter_head.backward(xs_array[rows], hidden, gradient / len(targets)))
    return {"paths": len(paths), "tokens": len(actions), "optimizer_steps": epochs,
            "parameter_tokens": len(rows), "seconds": time.perf_counter() - started}


def train(env: Environment, seed: int, iterations: int = 12, episodes: int = 32,
          group_size: int = 8, hidden: int = 64, lr: float = 0.001, epochs: int = 4,
          *, clip_ratio: float = 0.2, entropy_coefficient: float = 0.03,
          max_rollouts: int | None = None,
          on_iteration: Callable[[MLP, dict[str, Any]], Any] | None = None,
          actor: MLP | None = None, warm_start_paths: Sequence[Prefix] = (),
          warm_start_epochs: int = 1, verbose: bool = False) -> tuple[MLP, list[dict[str, Any]]]:
    """Adapt a policy online on one column; callback False stops after an update.

    Each group consists of independently sampled trajectories from the same
    empty root and unchanged actor. A rollout budget truncates only at complete
    group boundaries. Environment budget exceptions propagate immediately;
    incomplete groups receive no update and no completion trace row. Pass an
    already initialized actor to retain access to weights after such an error.
    Warm-start imitation, when requested, is separately accounted, never called
    a GRPO optimizer step. GRPO updates use full batches, not token minibatches.
    """
    if (min(iterations, episodes, hidden, epochs, warm_start_epochs) < 1
            or group_size < 2 or episodes % group_size):
        raise ValueError("positive budgets and episodes divisible by group_size >= 2 required")
    if (not 0 < clip_ratio < 1 or not np.isfinite(lr) or lr <= 0
            or not np.isfinite(entropy_coefficient) or entropy_coefficient < 0
            or (max_rollouts is not None and max_rollouts < 0)):
        raise ValueError("invalid optimizer or rollout budget settings")
    actor = initialize_policy(env, seed, hidden) if actor is None else actor
    _validate_actor(actor, env)
    optimizer = conditional.optimizers(actor, lr) if hasattr(actor, "parameter_head") else Adam(actor.params, lr)
    sampling_rng = np.random.default_rng(seed + 1_000_003)
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    warm = {"paths": 0, "tokens": 0, "optimizer_steps": 0, "seconds": 0.0}
    if warm_start_paths:
        warm = _warm_start(actor, optimizer, env, warm_start_paths, warm_start_epochs)
    cumulative = updates = 0
    for iteration in range(1, iterations + 1):
        count = episodes
        if max_rollouts is not None:
            count = min(count, ((max_rollouts - cumulative) // group_size) * group_size)
        if count < group_size:
            break
        iteration_started = time.perf_counter()
        rollout = _collect(env, actor, count, sampling_rng)
        rollout_seconds = time.perf_counter() - iteration_started
        ep_advantages = group_advantages(rollout["episode_returns"], group_size)
        rollout["advantages"] = ep_advantages[rollout["episode"]]
        informative = int(np.count_nonzero(
            rollout["episode_returns"].reshape(-1, group_size).std(axis=1) > 1.0e-8))
        all_parameters = actor.params + (actor.parameter_head.params if hasattr(actor, 'parameter_head') else [])
        before = [parameter.copy() for parameter in all_parameters]
        update_started = time.perf_counter()
        metrics = _actor_update(actor, optimizer, rollout, epochs, clip_ratio, entropy_coefficient)
        update_seconds = time.perf_counter() - update_started
        if any(not np.all(np.isfinite(parameter)) for parameter in all_parameters):
            raise FloatingPointError(f"non-finite GRPO weights at iteration {iteration}")
        cumulative += count
        updates += metrics["optimizer_steps"]
        returns = rollout["episode_returns"]
        row: dict[str, Any] = {
            "method": "recipe_grpo", "evaluation_scope": "online_per_column",
            "seed": seed, "iteration": iteration, "episodes": count,
            "cumulative_episodes": cumulative, "group_size": group_size,
            "transitions": int(rollout["actions"].size),
            "parameter_decisions": int(np.sum(rollout["parameter_actions"] >= 0)),
            "parameter_catalog_size": len(conditional.CANDIDATES),
            "mean_return": float(returns.mean()), "min_return": float(returns.min()),
            "max_return": float(returns.max()), "std_return": float(returns.std()),
            "invalid_episodes": int(rollout["invalid"].sum()),
            "invalid_fraction": float(rollout["invalid"].mean()),
            "stop_episodes": int(rollout["stopped"].sum()),
            "unique_proposals": len(set(rollout["pipelines"])),
            "informative_groups": informative,
            "advantage_min": float(ep_advantages.min()),
            "advantage_max": float(ep_advantages.max()),
            "advantage_normalization": "within_same_root_group_only",
            "trajectory_weighting": "one_over_length_full_batch",
            "cumulative_grpo_optimizer_steps": updates,
            "parameter_change_l2": float(np.sqrt(sum(
                np.sum((old - new) ** 2) for old, new in zip(before, all_parameters)))),
            "warm_start": dict(warm),
            "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
            "iteration_seconds": time.perf_counter() - iteration_started,
            "elapsed_seconds": time.perf_counter() - started, **metrics,
        }
        trace.append(row)
        if verbose:
            print(json.dumps(row, sort_keys=True), flush=True)
        if on_iteration is not None and on_iteration(actor, row) is False:
            break
    return actor, trace


def propose(actor: MLP, env: Environment, rng: np.random.Generator,
            greedy: bool = False) -> Prefix:
    """Execute one policy proposal; caller separately measures terminal reward."""
    _validate_actor(actor, env)
    stop = _stop_id(env)
    prefix: Prefix = ()
    for _step in range(env.max_steps):
        x = _features(env, prefix, actor.params[0].shape[0])
        mask = _mask(env, prefix)
        probabilities = distribution(actor, x[None, :], mask[None, :])[1][0]
        legal = np.flatnonzero(mask)
        weights = probabilities[legal]
        action = (int(legal[np.argmax(weights)]) if greedy
                  else int(rng.choice(legal, p=weights / weights.sum())))
        if action == stop:
            break
        _pid, _plogp, parameters = conditional.choose(actor, env, prefix, action, x, rng, greedy)
        following = env.transition(prefix, action, parameters=parameters) if parameters is not None else env.transition(prefix, action)
        if following is None:
            break
        prefix = PolicyPath(following)
    return prefix


def sample_uniform(env: Environment, seed: int, episodes: int,
                   on_episode: Callable[[dict[str, Any]], Any] | None = None) -> list[dict[str, Any]]:
    """No-learning control over exactly the same state-dependent legal menus."""
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    rng = np.random.default_rng(seed + 1_000_003)
    rows = []
    for episode in range(episodes):
        started = time.perf_counter()
        rollout = _collect(env, None, 1, rng)
        row = {"method": "recipe_uniform", "episode": episode + 1,
               "prefix": list(rollout["pipelines"][0]),
               "option_parameters": list(PolicyPath(rollout["pipelines"][0]).parameters),
               "reward": float(rollout["episode_returns"][0]),
               "length": int(rollout["lengths"][0]),
               "invalid": bool(rollout["invalid"][0]),
               "stopped": bool(rollout["stopped"][0]),
               "optimizer_steps": 0, "seconds": time.perf_counter() - started}
        rows.append(row)
        if on_episode is not None and on_episode(row) is False:
            break
    return rows

