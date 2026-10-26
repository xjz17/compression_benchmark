#!/usr/bin/env python3
"""Weight-conditioned, critic-free GRPO for compression operator graphs.

The graph environment owns operator legality, native compression/decompression,
complete framing, and cold lossless validation.  This module owns only policy
sampling and optimization.  Its observation is always::

    concat(task_features, state_features, [a, b, c])

where ``a + b + c == 1``.  A rollout group contains eight trajectories from
the same task/root and exact weight vector.  Non-terminal transitions have no
reward; selecting STOP evaluates a complete representation and receives ``-J``.

The module deliberately does not provide a Python compression fallback.  A
real environment must return the complete compressed payload, including all
metadata needed by its independent decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


CHECKPOINT_SCHEMA = "weight-conditioned-grpo-policy-v1"
FEATURE_SCHEMA = "task-state-plus-simplex-weights-v1"
FIXED_WEIGHTS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.4, 0.3, 0.3),
)


class GraphContractError(ValueError):
    """The graph task or a policy checkpoint violates this module's contract."""


@dataclass(frozen=True)
class CompressionMetrics:
    """Physical terminal measurements using the project-wide denominator."""

    original_numeric_bytes: int
    compression_time_ns: int
    decompression_time_ns: int
    compression_reference_ns_per_byte: float
    decompression_reference_ns_per_byte: float

    def __post_init__(self) -> None:
        if self.original_numeric_bytes <= 0:
            raise GraphContractError("terminal original_numeric_bytes must be positive")
        if self.compression_time_ns < 0 or self.decompression_time_ns < 0:
            raise GraphContractError("terminal elapsed times must be non-negative")
        references = (
            self.compression_reference_ns_per_byte,
            self.decompression_reference_ns_per_byte,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in references):
            raise GraphContractError("time-normalization references must be finite and positive")

    @property
    def compression_ns_per_byte(self) -> float:
        return self.compression_time_ns / self.original_numeric_bytes

    @property
    def decompression_ns_per_byte(self) -> float:
        return self.decompression_time_ns / self.original_numeric_bytes

    @property
    def normalized_compression_time(self) -> float:
        value = self.compression_ns_per_byte
        return value / (value + self.compression_reference_ns_per_byte)

    @property
    def normalized_decompression_time(self) -> float:
        value = self.decompression_ns_per_byte
        return value / (value + self.decompression_reference_ns_per_byte)


@dataclass(frozen=True)
class TerminalOutcome:
    """A measured complete graph plus its actual self-contained byte payload."""

    operator_graph: Any
    compressed_payload: bytes
    metrics: CompressionMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.compressed_payload, bytes):
            raise GraphContractError("compressed_payload must be the complete immutable bytes object")

    @property
    def compression_ratio(self) -> float:
        return len(self.compressed_payload) / self.metrics.original_numeric_bytes


@dataclass(frozen=True)
class TerminalFailure:
    """An explicitly infeasible candidate, with no fabricated frame or metrics.

    The constraint cost exceeds every admissible weighted J: frame bytes are
    bounded by max_frame_bytes and both normalized time components are < 1.
    This cost is only a training penalty, never a compression measurement.
    """
    operator_graph: Any
    original_numeric_bytes: int
    max_frame_bytes: int
    error: str

    def __post_init__(self) -> None:
        if self.original_numeric_bytes <= 0 or self.max_frame_bytes <= 0 or not self.error:
            raise GraphContractError("invalid infeasible-candidate receipt")


class GraphCompressionTask(Protocol):
    """Adapter contract for an executable compression-operator graph root."""

    task_id: str
    registry_fingerprint: str
    data_fingerprint: str
    root_fingerprint: str
    action_count: int
    stop_action: int
    max_steps: int

    def task_features(self) -> np.ndarray: ...
    def initial_state(self) -> Any: ...
    def state_features(self, state: Any) -> np.ndarray: ...
    def legal_actions(self, state: Any) -> Sequence[int]: ...
    def transition(self, state: Any, action: int) -> Any | None: ...
    def evaluate_terminal(self, state: Any) -> TerminalOutcome | TerminalFailure: ...


@dataclass(frozen=True)
class GRPOConfig:
    hidden_sizes: tuple[int, int] = (128, 128)
    learning_rate: float = 3.0e-4
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    epochs: int = 4
    group_size: int = 8
    groups_per_iteration: int = 8
    dirichlet_alpha: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gradient_max_norm: float = 1.0

    def __post_init__(self) -> None:
        if len(self.hidden_sizes) != 2 or min(self.hidden_sizes) < 1:
            raise ValueError("exactly two positive hidden-layer sizes are required")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")
        if not np.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be finite and non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.group_size != 8:
            raise ValueError("v1 uses exactly eight same-task/root/weight trajectories per group")
        if self.groups_per_iteration < 2 or self.groups_per_iteration % 2:
            raise ValueError("groups_per_iteration must be positive, even, and at least two")
        if (len(self.dirichlet_alpha) != 3
                or any(not np.isfinite(x) or x <= 0.0 for x in self.dirichlet_alpha)):
            raise ValueError("Dirichlet alpha must contain three finite positive values")
        if not np.isfinite(self.gradient_max_norm) or self.gradient_max_norm <= 0.0:
            raise ValueError("gradient_max_norm must be finite and positive")


@dataclass(frozen=True)
class PolicyIdentity:
    registry_fingerprint: str
    data_fingerprints: tuple[tuple[str, str], ...]
    root_fingerprints: tuple[tuple[str, str], ...]
    action_count: int
    stop_action: int
    task_feature_dim: int
    state_feature_dim: int
    hidden_sizes: tuple[int, int]
    feature_schema: str = FEATURE_SCHEMA

    @property
    def input_dim(self) -> int:
        return self.task_feature_dim + self.state_feature_dim + 3

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["data_fingerprints"] = [list(item) for item in self.data_fingerprints]
        value["root_fingerprints"] = [list(item) for item in self.root_fingerprints]
        value["hidden_sizes"] = list(self.hidden_sizes)
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "PolicyIdentity":
        try:
            result = cls(
                registry_fingerprint=str(value["registry_fingerprint"]),
                data_fingerprints=tuple((str(a), str(b)) for a, b in value["data_fingerprints"]),
                root_fingerprints=tuple((str(a), str(b)) for a, b in value["root_fingerprints"]),
                action_count=int(value["action_count"]),
                stop_action=int(value["stop_action"]),
                task_feature_dim=int(value["task_feature_dim"]),
                state_feature_dim=int(value["state_feature_dim"]),
                hidden_sizes=tuple(int(x) for x in value["hidden_sizes"]),
                feature_schema=str(value["feature_schema"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GraphContractError("invalid policy identity") from error
        _validate_identity(result)
        return result


class TwoHiddenMLP:
    """A small float64 tanh policy with two independently sized hidden layers."""

    def __init__(self, identity: PolicyIdentity, rng: np.random.Generator):
        _validate_identity(identity)
        h1, h2 = identity.hidden_sizes
        self.identity = identity
        self.params = [
            rng.normal(0.0, math.sqrt(1.0 / identity.input_dim), (identity.input_dim, h1)),
            np.zeros(h1, dtype=np.float64),
            rng.normal(0.0, math.sqrt(1.0 / h1), (h1, h2)),
            np.zeros(h2, dtype=np.float64),
            rng.normal(0.0, 0.01, (h2, identity.action_count)),
            np.zeros(identity.action_count, dtype=np.float64),
        ]

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.identity.input_dim:
            raise ValueError("policy inputs have the wrong shape")
        w1, b1, w2, b2, w3, b3 = self.params
        h1 = np.tanh(values @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        return h2 @ w3 + b3, (values, h1, h2)

    def backward(self, cache: tuple[np.ndarray, ...], output_gradient: np.ndarray) -> list[np.ndarray]:
        inputs, h1, h2 = cache
        w1, _b1, w2, _b2, w3, _b3 = self.params
        gradient = np.asarray(output_gradient, dtype=np.float64)
        if gradient.shape != (inputs.shape[0], w3.shape[1]):
            raise ValueError("policy output gradient has the wrong shape")
        grad_w3 = h2.T @ gradient
        grad_b3 = gradient.sum(axis=0)
        grad_h2 = (gradient @ w3.T) * (1.0 - h2 * h2)
        grad_w2 = h1.T @ grad_h2
        grad_b2 = grad_h2.sum(axis=0)
        grad_h1 = (grad_h2 @ w2.T) * (1.0 - h1 * h1)
        grad_w1 = inputs.T @ grad_h1
        grad_b1 = grad_h1.sum(axis=0)
        return [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]


class FrozenPolicy:
    """Deep-copied, read-only inference parameters."""

    def __init__(self, policy: TwoHiddenMLP, *, device: str = "cpu"):
        from weighted_grpo_torch_backend_v1 import normalize_device, frozen_parameters
        self.identity = policy.identity
        self.execution_device = normalize_device(device)
        self.params = tuple(np.array(value, copy=True) for value in policy.params)
        for value in self.params:
            value.flags.writeable = False
        self._device_params = (None if self.execution_device == "cpu" else
                               frozen_parameters(self.params, self.execution_device))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.identity.input_dim:
            raise ValueError("policy inputs have the wrong shape")
        if self._device_params is not None:
            from weighted_grpo_torch_backend_v1 import frozen_forward
            return frozen_forward(values, self._device_params, self.execution_device)
        w1, b1, w2, b2, w3, b3 = self.params
        return np.tanh(np.tanh(values @ w1 + b1) @ w2 + b2) @ w3 + b3


class Adam:
    def __init__(self, params: Sequence[np.ndarray], learning_rate: float, *, device: str = "cpu"):
        from weighted_grpo_torch_backend_v1 import normalize_device, device_receipt
        self.execution_device = normalize_device(device)
        self.execution_receipt = device_receipt(self.execution_device)
        self.params = list(params)
        self.learning_rate = float(learning_rate)
        self.first = [np.zeros_like(value) for value in self.params]
        self.second = [np.zeros_like(value) for value in self.params]
        self.steps = 0

    def step(self, gradients: Sequence[np.ndarray], max_norm: float) -> None:
        if len(gradients) != len(self.params):
            raise ValueError("gradient/parameter count mismatch")
        norm = math.sqrt(sum(float(np.sum(value * value)) for value in gradients))
        scale = min(1.0, max_norm / max(norm, 1.0e-30))
        self.steps += 1
        for index, (parameter, gradient) in enumerate(zip(self.params, gradients)):
            gradient = np.asarray(gradient, dtype=np.float64) * scale
            self.first[index] = 0.9 * self.first[index] + 0.1 * gradient
            self.second[index] = 0.999 * self.second[index] + 0.001 * gradient * gradient
            first = self.first[index] / (1.0 - 0.9**self.steps)
            second = self.second[index] / (1.0 - 0.999**self.steps)
            parameter -= self.learning_rate * first / (np.sqrt(second) + 1.0e-8)


@dataclass(frozen=True)
class TrainingResult:
    policy: TwoHiddenMLP
    trace: tuple[dict[str, Any], ...]
    config: GRPOConfig
    seed: int
    completed_iterations: int

    def freeze(self) -> FrozenPolicy:
        return FrozenPolicy(self.policy)


@dataclass(frozen=True)
class LoadedCheckpoint:
    policy: TwoHiddenMLP
    config: GRPOConfig
    seed: int
    completed_iterations: int
    extra: Mapping[str, Any]
    payload_sha256: str


@dataclass(frozen=True)
class ProposalResult:
    actions: tuple[int, ...]
    weight: tuple[float, float, float]
    outcome: TerminalOutcome
    objective: float
    feature_time_ns: int
    model_inference_time_ns: int
    environment_transition_time_ns: int
    terminal_evaluation_time_ns: int
    prediction_time_ns: int

    @property
    def operator_graph(self) -> Any:
        return self.outcome.operator_graph

    @property
    def compressed_payload(self) -> bytes:
        return self.outcome.compressed_payload


def _validate_identity(identity: PolicyIdentity) -> None:
    if (not identity.registry_fingerprint or identity.action_count < 1
            or not 0 <= identity.stop_action < identity.action_count
            or min(identity.task_feature_dim, identity.state_feature_dim) < 1
            or len(identity.hidden_sizes) != 2 or min(identity.hidden_sizes) < 1
            or identity.feature_schema != FEATURE_SCHEMA):
        raise GraphContractError("invalid policy identity fields")
    for name, values in (("data", identity.data_fingerprints), ("root", identity.root_fingerprints)):
        if not values or tuple(sorted(values)) != values or len(dict(values)) != len(values):
            raise GraphContractError(f"{name} fingerprints must be non-empty, sorted, and unique")
        if any(not task_id or not fingerprint for task_id, fingerprint in values):
            raise GraphContractError(f"{name} fingerprints cannot contain empty strings")


def _vector(value: Any, label: str, expected_dim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 1 or not result.size or not np.all(np.isfinite(result))
            or (expected_dim is not None and result.size != expected_dim)):
        raise GraphContractError(f"invalid {label}: shape={result.shape}")
    return result


def validate_weight(weight: Sequence[float]) -> np.ndarray:
    result = np.asarray(weight, dtype=np.float64)
    if (result.shape != (3,) or not np.all(np.isfinite(result))
            or np.any(result < 0.0)
            or not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1.0e-12)):
        raise ValueError("weights must be three finite non-negative values summing to one")
    return result


def objective_value(outcome: TerminalOutcome | TerminalFailure, weight: Sequence[float]) -> float:
    weights = validate_weight(weight)
    if isinstance(outcome, TerminalFailure):
        return float(1.0 + weights[0] * outcome.max_frame_bytes / outcome.original_numeric_bytes
                     + weights[1] + weights[2])
    components = np.asarray([
        outcome.compression_ratio,
        outcome.metrics.normalized_compression_time,
        outcome.metrics.normalized_decompression_time,
    ])
    value = float(weights @ components)
    if not np.isfinite(value):
        raise GraphContractError("non-finite terminal objective")
    return value


def sample_training_weights(
    group_count: int,
    rng: np.random.Generator,
    dirichlet_alpha: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[tuple[np.ndarray, str], ...]:
    """Return exactly half fixed-six and half continuous-simplex weights."""

    if group_count < 2 or group_count % 2:
        raise ValueError("group_count must be positive and even")
    alpha = np.asarray(dirichlet_alpha, dtype=np.float64)
    if alpha.shape != (3,) or np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("Dirichlet alpha must contain three finite positive values")
    half = group_count // 2
    fixed_indices = rng.integers(0, len(FIXED_WEIGHTS), size=half)
    rows: list[tuple[np.ndarray, str]] = [
        (np.asarray(FIXED_WEIGHTS[int(index)], dtype=np.float64), "fixed")
        for index in fixed_indices
    ]
    rows.extend((row.astype(np.float64), "dirichlet") for row in rng.dirichlet(alpha, size=half))
    order = rng.permutation(group_count)
    return tuple(rows[int(index)] for index in order)


def _inspect_tasks(
    tasks: Sequence[GraphCompressionTask], hidden_sizes: tuple[int, int]
) -> PolicyIdentity:
    if not tasks:
        raise ValueError("at least one graph task is required")
    task_ids: set[str] = set()
    registries: set[str] = set()
    data: list[tuple[str, str]] = []
    roots: list[tuple[str, str]] = []
    first = tasks[0]
    action_count = int(first.action_count)
    stop_action = int(first.stop_action)
    task_dim = state_dim = None
    for task in tasks:
        task_id = str(task.task_id)
        if not task_id or task_id in task_ids:
            raise GraphContractError("task_id values must be non-empty and unique")
        task_ids.add(task_id)
        registry = str(task.registry_fingerprint)
        if not registry or not str(task.data_fingerprint) or not str(task.root_fingerprint):
            raise GraphContractError("registry, data, and root fingerprints are required")
        registries.add(registry)
        if (int(task.action_count) != action_count or int(task.stop_action) != stop_action
                or int(task.max_steps) < 1):
            raise GraphContractError("shared tasks must use the same positive action contract")
        static = _vector(task.task_features(), f"task features for {task_id}", task_dim)
        root = task.initial_state()
        dynamic = _vector(task.state_features(root), f"state features for {task_id}", state_dim)
        task_dim = static.size
        state_dim = dynamic.size
        data.append((task_id, str(task.data_fingerprint)))
        roots.append((task_id, str(task.root_fingerprint)))
    if len(registries) != 1:
        raise GraphContractError("a shared policy cannot mix registry fingerprints")
    identity = PolicyIdentity(
        registry_fingerprint=registries.pop(),
        data_fingerprints=tuple(sorted(data)),
        root_fingerprints=tuple(sorted(roots)),
        action_count=action_count,
        stop_action=stop_action,
        task_feature_dim=int(task_dim),
        state_feature_dim=int(state_dim),
        hidden_sizes=hidden_sizes,
    )
    _validate_identity(identity)
    return identity


def initialize_policy(
    tasks: Sequence[GraphCompressionTask], seed: int, hidden_sizes: tuple[int, int] = (128, 128)
) -> TwoHiddenMLP:
    return TwoHiddenMLP(_inspect_tasks(tasks, hidden_sizes), np.random.default_rng(seed))


def _validate_task_for_policy(policy: TwoHiddenMLP | FrozenPolicy, task: GraphCompressionTask) -> None:
    identity = policy.identity
    if (str(task.registry_fingerprint) != identity.registry_fingerprint
            or int(task.action_count) != identity.action_count
            or int(task.stop_action) != identity.stop_action
            or int(task.max_steps) < 1):
        raise GraphContractError("task action/registry contract does not match the policy")
    _vector(task.task_features(), "task features", identity.task_feature_dim)
    root = task.initial_state()
    _vector(task.state_features(root), "root state features", identity.state_feature_dim)


def _observation(
    task: GraphCompressionTask,
    state: Any,
    static_features: np.ndarray,
    weight: np.ndarray,
    identity: PolicyIdentity,
) -> np.ndarray:
    dynamic = _vector(task.state_features(state), "state features", identity.state_feature_dim)
    return np.concatenate((static_features, dynamic, weight))


def _legal_mask(task: GraphCompressionTask, state: Any, force_stop: bool = False) -> np.ndarray:
    count = int(task.action_count)
    legal = tuple(task.legal_actions(state))
    if (not legal or len(set(legal)) != len(legal)
            or any(isinstance(action, bool) or not isinstance(action, (int, np.integer))
                   or not 0 <= int(action) < count for action in legal)):
        raise GraphContractError(f"invalid legal action menu: {legal}")
    if force_stop:
        if int(task.stop_action) not in legal:
            raise GraphContractError("STOP must be legal at the final policy step")
        legal = (int(task.stop_action),)
    mask = np.zeros(count, dtype=bool)
    mask[list(legal)] = True
    return mask


def _distribution(
    policy: TwoHiddenMLP | FrozenPolicy, inputs: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
    if masks.ndim != 2 or masks.shape != (inputs.shape[0], policy.identity.action_count):
        raise ValueError("legal masks have the wrong shape")
    if not np.all(masks.any(axis=1)):
        raise GraphContractError("every policy row must have at least one legal action")
    if isinstance(policy, TwoHiddenMLP):
        logits, cache = policy.forward(inputs)
    else:
        logits, cache = policy.forward(inputs), None
    masked = np.where(masks, logits, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probabilities = np.where(masks, np.exp(shifted), 0.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, cache


def group_advantages(returns: Sequence[float], group_size: int = 8) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if (group_size != 8 or values.ndim != 1 or not values.size
            or values.size % group_size or not np.all(np.isfinite(values))):
        raise ValueError("finite returns in complete groups of eight are required")
    groups = values.reshape(-1, group_size)
    centered = groups - groups.mean(axis=1, keepdims=True)
    deviation = groups.std(axis=1, keepdims=True)
    return np.divide(
        centered,
        deviation + 1.0e-8,
        out=np.zeros_like(centered),
        where=deviation > 1.0e-8,
    ).reshape(-1)


def _rollout(
    policy: TwoHiddenMLP | FrozenPolicy,
    task: GraphCompressionTask,
    weight: np.ndarray,
    rng: np.random.Generator,
    *,
    greedy: bool,
    timing: bool,
) -> dict[str, Any]:
    identity = policy.identity
    state = task.initial_state()
    static = _vector(task.task_features(), "task features", identity.task_feature_dim)
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    log_probabilities: list[float] = []
    feature_ns = inference_ns = transition_ns = terminal_ns = 0
    outcome: TerminalOutcome | None = None
    objective = math.nan
    for step in range(int(task.max_steps)):
        started = time.perf_counter_ns()
        observation = _observation(task, state, static, weight, identity)
        mask = _legal_mask(task, state, force_stop=step + 1 == int(task.max_steps))
        if timing:
            feature_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        probabilities, _cache = _distribution(policy, observation[None, :], mask[None, :])
        legal = np.flatnonzero(mask)
        legal_probabilities = probabilities[0, legal]
        action = (int(legal[np.argmax(legal_probabilities)]) if greedy
                  else int(rng.choice(legal, p=legal_probabilities / legal_probabilities.sum())))
        if timing:
            inference_ns += time.perf_counter_ns() - started
        features.append(observation)
        masks.append(mask)
        actions.append(action)
        log_probabilities.append(float(np.log(max(probabilities[0, action], 1.0e-300))))
        if action == int(task.stop_action):
            started = time.perf_counter_ns()
            outcome = task.evaluate_terminal(state)
            if not isinstance(outcome, (TerminalOutcome, TerminalFailure)):
                raise GraphContractError("evaluate_terminal must return a terminal outcome or explicit failure")
            objective = objective_value(outcome, weight)
            if timing:
                terminal_ns += time.perf_counter_ns() - started
            break
        started = time.perf_counter_ns()
        following = task.transition(state, action)
        if timing:
            transition_ns += time.perf_counter_ns() - started
        if following is None:
            raise GraphContractError("a legal non-STOP action produced an invalid transition")
        state = following
    if outcome is None:
        raise GraphContractError("rollout reached its horizon without a terminal outcome")
    return {
        "features": np.asarray(features, dtype=np.float64),
        "legal": np.asarray(masks, dtype=bool),
        "actions": np.asarray(actions, dtype=np.int64),
        "old_logp": np.asarray(log_probabilities, dtype=np.float64),
        "return": -objective,
        "objective": objective,
        "outcome": outcome,
        "final_state": state,
        "feature_time_ns": feature_ns,
        "model_inference_time_ns": inference_ns,
        "environment_transition_time_ns": transition_ns,
        "terminal_evaluation_time_ns": terminal_ns,
    }


def _update(
    policy: TwoHiddenMLP,
    optimizer: Adam,
    rollout: Mapping[str, np.ndarray],
    config: GRPOConfig,
) -> dict[str, float | int]:
    if optimizer.execution_device != "cpu":
        from weighted_grpo_torch_backend_v1 import update_policy
        return update_policy(policy, optimizer, rollout, config)
    features = rollout["features"]
    masks = rollout["legal"]
    actions = rollout["actions"]
    old_logp = rollout["old_logp"]
    advantages = rollout["advantages"]
    weights = rollout["token_weights"]
    weights = weights / weights.sum()
    clip_fractions: list[float] = []
    for _epoch in range(config.epochs):
        probabilities, cache = _distribution(policy, features, masks)
        assert cache is not None
        selected = probabilities[np.arange(actions.size), actions]
        logp = np.log(np.maximum(selected, 1.0e-300))
        ratios = np.exp(logp - old_logp)
        active = (((advantages >= 0.0) & (ratios <= 1.0 + config.clip_ratio))
                  | ((advantages < 0.0) & (ratios >= 1.0 - config.clip_ratio)))
        coefficient = np.where(active, -advantages * ratios * weights, 0.0)
        gradient = -coefficient[:, None] * probabilities
        gradient[np.arange(actions.size), actions] += coefficient
        log_probabilities = np.log(np.maximum(probabilities, 1.0e-300))
        entropy = -np.sum(probabilities * log_probabilities, axis=1, keepdims=True)
        gradient += (config.entropy_coefficient * weights[:, None] * probabilities
                     * (log_probabilities + entropy))
        optimizer.step(policy.backward(cache, gradient), config.gradient_max_norm)
        clip_fractions.append(float(np.sum(weights * ~active)))
    probabilities, _cache = _distribution(policy, features, masks)
    selected = probabilities[np.arange(actions.size), actions]
    ratios = np.exp(np.log(np.maximum(selected, 1.0e-300)) - old_logp)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-300)), axis=1)
    surrogate = np.minimum(
        ratios * advantages,
        np.clip(ratios, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages,
    )
    return {
        "policy_loss": float(-np.sum(weights * surrogate)),
        "entropy": float(np.sum(weights * entropy)),
        "approx_kl": float(np.sum(weights * ((ratios - 1.0) - np.log(ratios)))),
        "clip_fraction": float(np.mean(clip_fractions)),
        "optimizer_steps": config.epochs,
    }


def train_grpo(
    tasks: Sequence[GraphCompressionTask],
    *,
    seed: int,
    iterations: int,
    config: GRPOConfig = GRPOConfig(),
    policy: TwoHiddenMLP | None = None,
) -> TrainingResult:
    """Train one shared policy; all expensive terminal calls belong to tasks."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    identity = _inspect_tasks(tasks, config.hidden_sizes)
    if policy is None:
        policy = TwoHiddenMLP(identity, np.random.default_rng(seed))
    elif policy.identity != identity:
        raise GraphContractError("initial policy identity does not match the training tasks")
    optimizer = Adam(policy.params, config.learning_rate)
    rng = np.random.default_rng(seed + 1_000_003)
    trace: list[dict[str, Any]] = []
    training_started = time.perf_counter_ns()
    for iteration in range(1, iterations + 1):
        iteration_started = time.perf_counter_ns()
        sampled_weights = sample_training_weights(
            config.groups_per_iteration, rng, config.dirichlet_alpha
        )
        group_tasks = rng.integers(0, len(tasks), size=config.groups_per_iteration)
        episodes: list[dict[str, Any]] = []
        group_rows: list[dict[str, Any]] = []
        for group_index, ((weight, source), task_index) in enumerate(
            zip(sampled_weights, group_tasks)
        ):
            task = tasks[int(task_index)]
            start = len(episodes)
            for _ in range(config.group_size):
                episodes.append(_rollout(policy, task, weight, rng, greedy=False, timing=False))
            returns = [episodes[index]["return"] for index in range(start, len(episodes))]
            group_rows.append({
                "group": group_index,
                "task_id": str(task.task_id),
                "data_fingerprint": str(task.data_fingerprint),
                "root_fingerprint": str(task.root_fingerprint),
                "weight": [float(x) for x in weight],
                "weight_source": source,
                "episodes": config.group_size,
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
            })
        returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float64)
        episode_advantages = group_advantages(returns, config.group_size)
        features = np.concatenate([episode["features"] for episode in episodes])
        legal = np.concatenate([episode["legal"] for episode in episodes])
        actions = np.concatenate([episode["actions"] for episode in episodes])
        old_logp = np.concatenate([episode["old_logp"] for episode in episodes])
        lengths = np.asarray([episode["actions"].size for episode in episodes], dtype=np.int64)
        episode_ids = np.repeat(np.arange(len(episodes)), lengths)
        batch = {
            "features": features,
            "legal": legal,
            "actions": actions,
            "old_logp": old_logp,
            "advantages": episode_advantages[episode_ids],
            "token_weights": 1.0 / lengths[episode_ids],
        }
        before = [value.copy() for value in policy.params]
        update_started = time.perf_counter_ns()
        update_metrics = _update(policy, optimizer, batch, config)
        update_ns = time.perf_counter_ns() - update_started
        if any(not np.all(np.isfinite(value)) for value in policy.params):
            raise FloatingPointError("non-finite GRPO policy parameters")
        trace.append({
            "iteration": iteration,
            "groups": config.groups_per_iteration,
            "group_size": config.group_size,
            "episodes": len(episodes),
            "transitions": int(actions.size),
            "fixed_weight_groups": sum(row["weight_source"] == "fixed" for row in group_rows),
            "dirichlet_weight_groups": sum(
                row["weight_source"] == "dirichlet" for row in group_rows
            ),
            "mean_terminal_reward": float(returns.mean()),
            "min_terminal_reward": float(returns.min()),
            "max_terminal_reward": float(returns.max()),
            "informative_groups": sum(row["std_return"] > 1.0e-8 for row in group_rows),
            "parameter_change_l2": float(math.sqrt(sum(
                float(np.sum((old - new) ** 2)) for old, new in zip(before, policy.params)
            ))),
            "reward_definition": "terminal_-J_nonterminal_0",
            "advantage_normalization": "same_task_root_weight_group_of_8",
            "trajectory_weighting": "one_over_trajectory_length",
            "groups_detail": group_rows,
            "update_time_ns": update_ns,
            "iteration_time_ns": time.perf_counter_ns() - iteration_started,
            "elapsed_time_ns": time.perf_counter_ns() - training_started,
            **update_metrics,
        })
    return TrainingResult(policy, tuple(trace), config, seed, iterations)


def propose_frozen(
    policy: FrozenPolicy,
    task: GraphCompressionTask,
    weight: Sequence[float],
    *,
    greedy: bool = True,
    seed: int = 0,
) -> ProposalResult:
    """Return an operator graph and its measured complete compressed bytes."""

    if not isinstance(policy, FrozenPolicy):
        raise TypeError("propose_frozen requires a FrozenPolicy snapshot")
    _validate_task_for_policy(policy, task)
    weights = validate_weight(weight)
    started = time.perf_counter_ns()
    row = _rollout(
        policy,
        task,
        weights,
        np.random.default_rng(seed),
        greedy=greedy,
        timing=True,
    )
    total = time.perf_counter_ns() - started
    return ProposalResult(
        actions=tuple(int(x) for x in row["actions"]),
        weight=tuple(float(x) for x in weights),
        outcome=row["outcome"],
        objective=float(row["objective"]),
        feature_time_ns=int(row["feature_time_ns"]),
        model_inference_time_ns=int(row["model_inference_time_ns"]),
        environment_transition_time_ns=int(row["environment_transition_time_ns"]),
        terminal_evaluation_time_ns=int(row["terminal_evaluation_time_ns"]),
        prediction_time_ns=total,
    )


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f8")
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _array_from_record(value: Mapping[str, Any]) -> np.ndarray:
    try:
        if value["dtype"] != "<f8":
            raise GraphContractError("checkpoint arrays must use little-endian float64")
        shape = tuple(int(x) for x in value["shape"])
        raw = base64.b64decode(str(value["base64"]), validate=True)
        result = np.frombuffer(raw, dtype="<f8").reshape(shape).astype(np.float64, copy=True)
    except (KeyError, TypeError, ValueError) as error:
        raise GraphContractError("invalid checkpoint array") from error
    if not np.all(np.isfinite(result)):
        raise GraphContractError("checkpoint contains non-finite parameters")
    return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must be finite JSON data") from error


def save_checkpoint(
    result: TrainingResult,
    path: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Atomically write deterministic policy bytes and return their SHA-256."""

    payload = {
        "identity": result.policy.identity.to_json(),
        "config": asdict(result.config),
        "seed": int(result.seed),
        "completed_iterations": int(result.completed_iterations),
        "parameters": [_array_record(value) for value in result.policy.params],
        "extra": dict(extra or {}),
    }
    payload_bytes = _canonical_json(payload)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    document = _canonical_json({
        "schema": CHECKPOINT_SCHEMA,
        "payload_sha256": digest,
        "payload": payload,
    }) + b"\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return digest


def load_checkpoint(
    path: str | Path,
    *,
    expected_registry_fingerprint: str | None = None,
    expected_data_fingerprints: Mapping[str, str] | None = None,
) -> LoadedCheckpoint:
    """Load exact weights, rejecting changed registries or training data identities."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraphContractError("cannot read checkpoint") from error
    if document.get("schema") != CHECKPOINT_SCHEMA or not isinstance(document.get("payload"), dict):
        raise GraphContractError("unsupported or unversioned checkpoint")
    payload = document["payload"]
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if document.get("payload_sha256") != digest:
        raise GraphContractError("checkpoint payload digest mismatch")
    identity = PolicyIdentity.from_json(payload.get("identity", {}))
    if (expected_registry_fingerprint is not None
            and identity.registry_fingerprint != expected_registry_fingerprint):
        raise GraphContractError("checkpoint registry fingerprint mismatch")
    if expected_data_fingerprints is not None:
        expected = tuple(sorted((str(key), str(value))
                                for key, value in expected_data_fingerprints.items()))
        if identity.data_fingerprints != expected:
            raise GraphContractError("checkpoint training-data fingerprints mismatch")
    try:
        config_value = dict(payload["config"])
        config_value["hidden_sizes"] = tuple(config_value["hidden_sizes"])
        config_value["dirichlet_alpha"] = tuple(config_value["dirichlet_alpha"])
        config = GRPOConfig(**config_value)
        seed = int(payload["seed"])
        completed = int(payload["completed_iterations"])
        extra = dict(payload["extra"])
        parameters = [_array_from_record(value) for value in payload["parameters"]]
    except (KeyError, TypeError, ValueError) as error:
        raise GraphContractError("invalid checkpoint payload") from error
    if config.hidden_sizes != identity.hidden_sizes or completed < 0 or len(parameters) != 6:
        raise GraphContractError("checkpoint configuration/parameter count mismatch")
    expected_shapes = [
        (identity.input_dim, identity.hidden_sizes[0]),
        (identity.hidden_sizes[0],),
        (identity.hidden_sizes[0], identity.hidden_sizes[1]),
        (identity.hidden_sizes[1],),
        (identity.hidden_sizes[1], identity.action_count),
        (identity.action_count,),
    ]
    if [value.shape for value in parameters] != expected_shapes:
        raise GraphContractError("checkpoint parameter shapes do not match its identity")
    policy = TwoHiddenMLP(identity, np.random.default_rng(0))
    policy.params = parameters
    return LoadedCheckpoint(policy, config, seed, completed, extra, digest)


__all__ = [
    "CHECKPOINT_SCHEMA", "FEATURE_SCHEMA", "FIXED_WEIGHTS", "Adam",
    "CompressionMetrics", "FrozenPolicy", "GRPOConfig", "GraphCompressionTask",
    "GraphContractError", "LoadedCheckpoint", "PolicyIdentity", "ProposalResult",
    "TerminalOutcome", "TerminalFailure", "TrainingResult", "TwoHiddenMLP", "group_advantages",
    "initialize_policy", "load_checkpoint", "objective_value", "propose_frozen",
    "sample_training_weights", "save_checkpoint", "train_grpo", "validate_weight",
]
