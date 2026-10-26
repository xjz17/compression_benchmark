#!/usr/bin/env python3
"""Run Beam-29, PPO, and GRPO on one shared PV2F physical-byte table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from enumerate_physical_operator_variants_v2 import load_column
from physical_operator_runtime_v2 import COMPRESSORS, PACKERS, TRANSFORMS, registry


TRANSFORM_NAMES = tuple(spec.name for spec in TRANSFORMS)
PACKER_NAMES = tuple(spec.name for spec in PACKERS)
COMPRESSOR_NAMES = tuple(spec.name for spec in COMPRESSORS)
ACTION_NAMES = TRANSFORM_NAMES + PACKER_NAMES + COMPRESSOR_NAMES
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
TRANSFORM_SET = set(TRANSFORM_NAMES)
PACKER_SET = set(PACKER_NAMES)
COMPRESSOR_SET = set(COMPRESSOR_NAMES)
MAX_TRANSFORMS = 2
MAX_STEPS = MAX_TRANSFORMS + 2
INVALID_RATE = 10.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ColumnTable:
    dataset: str
    column_index: int
    column_name: str
    logical_type: str
    original_bytes: int
    rates: dict[tuple[str, ...], float]
    sizes: dict[tuple[str, ...], int]
    failures: int
    source_csv: Path

    @property
    def key(self) -> tuple[str, int]:
        return self.dataset, self.column_index


def parse_pipeline(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(" -> ") if part.strip())


def load_tables(measurement_root: Path) -> list[ColumnTable]:
    manifests = sorted(measurement_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no per-column manifests under {measurement_root}")
    tables: list[ColumnTable] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        csv_path = manifest_path.with_name("pipeline_measurements.csv")
        rows = read_csv(csv_path)
        rates: dict[tuple[str, ...], float] = {}
        sizes: dict[tuple[str, ...], int] = {}
        failures = 0
        for row in rows:
            pipeline = parse_pipeline(row["pipeline"])
            if row["status"] == "ok" and int(row["lossless"]) == 1:
                if pipeline in sizes:
                    raise AssertionError(f"duplicate pipeline {pipeline} in {csv_path}")
                sizes[pipeline] = int(row["final_bytes"])
                rates[pipeline] = float(row["compression_rate"])
            else:
                failures += 1
        expected = int(manifest["expected_candidates"])
        if len(rows) != expected or len(sizes) + failures != expected:
            raise AssertionError((csv_path, len(rows), len(sizes), failures, expected))
        tables.append(
            ColumnTable(
                dataset=manifest["dataset"],
                column_index=int(manifest["column_index"]),
                column_name=manifest["column_name"],
                logical_type=manifest["logical_type"],
                original_bytes=int(manifest["original_bytes"]),
                rates=rates,
                sizes=sizes,
                failures=failures,
                source_csv=csv_path,
            )
        )
    tables.sort(key=lambda table: (table.dataset, table.column_index))
    if len({table.key for table in tables}) != len(tables):
        raise AssertionError("duplicate dataset/column table")
    return tables


def split_prefix(prefix: tuple[str, ...]) -> tuple[tuple[str, ...], str | None, str | None]:
    transforms: list[str] = []
    packer: str | None = None
    compressor: str | None = None
    for name in prefix:
        if packer is None and name in TRANSFORM_SET:
            transforms.append(name)
        elif packer is None and name in PACKER_SET:
            packer = name
        elif packer is not None and compressor is None and name in COMPRESSOR_SET:
            compressor = name
        else:
            raise ValueError(f"illegal prefix {prefix}")
    if len(transforms) > MAX_TRANSFORMS:
        raise ValueError(f"too many transforms in {prefix}")
    return tuple(transforms), packer, compressor


def legal_action_names(prefix: tuple[str, ...]) -> tuple[str, ...]:
    transforms, packer, compressor = split_prefix(prefix)
    if compressor is not None:
        return ()
    if packer is not None:
        return COMPRESSOR_NAMES
    available_transforms = tuple(name for name in TRANSFORM_NAMES if name not in transforms)
    if len(transforms) >= MAX_TRANSFORMS:
        return PACKER_NAMES
    return available_transforms + PACKER_NAMES


def is_terminal(prefix: tuple[str, ...]) -> bool:
    return split_prefix(prefix)[2] is not None


def default_completion(prefix: tuple[str, ...]) -> tuple[str, ...]:
    transforms, packer, compressor = split_prefix(prefix)
    if compressor is not None:
        return prefix
    if packer is None:
        return transforms + ("raw.i64", "none")
    return transforms + (packer, "none")


def lookup_rate(table: ColumnTable, pipeline: tuple[str, ...]) -> float:
    return table.rates.get(pipeline, INVALID_RATE)


def lookup_size(table: ColumnTable, pipeline: tuple[str, ...]) -> int:
    return table.sizes.get(pipeline, int(table.original_bytes * INVALID_RATE))


def beam29_select(table: ColumnTable, width: int = 29) -> dict[str, Any]:
    """Run only the online Beam selector over the frozen candidate-byte table."""

    frontier: list[tuple[str, ...]] = [()]
    evaluated: set[tuple[str, ...]] = set()
    terminals: set[tuple[str, ...]] = set()
    level_rows: list[dict[str, int]] = []
    for step in range(1, MAX_STEPS + 1):
        expanded: list[tuple[str, ...]] = []
        for prefix in frontier:
            for action in legal_action_names(prefix):
                candidate = prefix + (action,)
                if is_terminal(candidate):
                    terminals.add(candidate)
                    evaluated.add(candidate)
                else:
                    probe = default_completion(candidate)
                    evaluated.add(probe)
                    expanded.append(candidate)
        if not expanded:
            level_rows.append({"step": step, "expanded_partial": 0, "retained": 0, "terminal_seen": len(terminals)})
            break
        frontier = sorted(
            set(expanded),
            key=lambda prefix: (lookup_size(table, default_completion(prefix)), prefix),
        )[:width]
        level_rows.append(
            {
                "step": step,
                "expanded_partial": len(set(expanded)),
                "retained": len(frontier),
                "terminal_seen": len(terminals),
            }
        )
    if not terminals:
        raise AssertionError("Beam-29 found no terminal pipeline")
    best = min(terminals, key=lambda pipeline: (lookup_size(table, pipeline), pipeline))
    return {
        "pipeline": best,
        "bytes": lookup_size(table, best),
        "rate": lookup_rate(table, best),
        "candidate_evaluations": len(evaluated),
        "invalid_evaluations": sum(pipeline not in table.sizes for pipeline in evaluated),
        "levels": level_rows,
    }


def beam29(table: ColumnTable, width: int = 29) -> dict[str, Any]:
    """Run Beam and add the offline oracle fields used by result analysis."""

    result = beam29_select(table, width)
    oracle = min(table.sizes, key=lambda pipeline: (table.sizes[pipeline], pipeline))
    return {
        **result,
        "oracle_pipeline": oracle,
        "oracle_bytes": table.sizes[oracle],
        "oracle_rate": table.rates[oracle],
    }


def signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def column_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    floating = values.astype(np.float64)
    deltas = np.diff(floating)
    delta2 = np.diff(deltas)
    unique_ratio = np.unique(values).size / max(1, values.size)
    run_ratio = float(np.mean(values[1:] == values[:-1])) if values.size > 1 else 1.0
    monotonic = float(np.mean(deltas >= 0)) if deltas.size else 1.0
    span = int(values.max()) - int(values.min()) if values.size else 0
    bit_width = max(0, int(span).bit_length()) / 64.0
    p01, p99 = np.percentile(floating, [1.0, 99.0]) if values.size else (0.0, 0.0)
    counts = np.unique(values, return_counts=True)[1]
    probabilities = counts.astype(np.float64) / max(1, values.size)
    entropy = float(-np.sum(probabilities * np.log2(probabilities))) / max(1.0, math.log2(max(2, values.size)))
    return np.asarray(
        [
            math.log1p(values.size),
            signed_log1p(float(np.mean(floating))) if values.size else 0.0,
            math.log1p(float(np.std(floating))) if values.size else 0.0,
            signed_log1p(float(np.min(floating))) if values.size else 0.0,
            signed_log1p(float(np.max(floating))) if values.size else 0.0,
            unique_ratio,
            run_ratio,
            monotonic,
            float(np.mean(deltas == 0)) if deltas.size else 1.0,
            math.log1p(float(np.mean(np.abs(deltas)))) if deltas.size else 0.0,
            math.log1p(float(np.std(deltas))) if deltas.size else 0.0,
            math.log1p(float(np.std(delta2))) if delta2.size else 0.0,
            bit_width,
            float(np.mean(values == 0)) if values.size else 0.0,
            float(np.mean(values < 0)) if values.size else 0.0,
            math.log1p(max(0.0, float(p99 - p01))),
            entropy,
        ],
        dtype=np.float64,
    )


def load_feature_matrix(repo_root: Path, tables: list[ColumnTable]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for table in tables:
        values, metadata, _data, _info = load_column(
            repo_root,
            table.dataset,
            table.column_index,
            0,
        )
        if values.size * 8 != table.original_bytes:
            raise AssertionError((table.key, values.size * 8, table.original_bytes))
        feature = column_features(values)
        features.append(feature)
        rows.append(
            {
                "dataset": table.dataset,
                "column_index": table.column_index,
                "column_name": table.column_name,
                "logical_type": table.logical_type,
                **{f"feature_{index}": float(value) for index, value in enumerate(feature)},
            }
        )
    return np.vstack(features), rows


def prefix_features(prefix: tuple[str, ...]) -> np.ndarray:
    transforms, packer, compressor = split_prefix(prefix)
    phase = np.zeros(3, dtype=np.float64)
    if compressor is not None:
        phase[2] = 1.0
    elif packer is not None:
        phase[1] = 1.0
    else:
        phase[0] = 1.0
    depth = np.zeros(MAX_TRANSFORMS + 1, dtype=np.float64)
    depth[len(transforms)] = 1.0
    transform_history = np.zeros(MAX_TRANSFORMS * len(TRANSFORM_NAMES), dtype=np.float64)
    for position, name in enumerate(transforms):
        transform_history[position * len(TRANSFORM_NAMES) + TRANSFORM_NAMES.index(name)] = 1.0
    packer_history = np.zeros(len(PACKER_NAMES), dtype=np.float64)
    if packer is not None:
        packer_history[PACKER_NAMES.index(packer)] = 1.0
    return np.concatenate((phase, depth, transform_history, packer_history))


class FeatureBuilder:
    def __init__(self, raw_features: np.ndarray, train_indices: np.ndarray):
        self.raw_features = raw_features
        self.mean = raw_features[train_indices].mean(axis=0)
        self.std = raw_features[train_indices].std(axis=0)
        self.std[self.std < 1.0e-8] = 1.0
        self.prefix_cache: dict[tuple[str, ...], np.ndarray] = {}

    @property
    def input_dim(self) -> int:
        return self.raw_features.shape[1] + prefix_features(()).size

    def one(self, column: int, prefix: tuple[str, ...]) -> np.ndarray:
        if prefix not in self.prefix_cache:
            self.prefix_cache[prefix] = prefix_features(prefix)
        normalized = np.clip((self.raw_features[column] - self.mean) / self.std, -10.0, 10.0)
        return np.concatenate((normalized, self.prefix_cache[prefix]))


SPARSE19_RAW_INDICES = tuple(index for index in range(17) if index != 8)
Sparse19Observation = tuple[np.ndarray, int, int, int]


def legal_nonterminal_prefixes() -> tuple[tuple[str, ...], ...]:
    """Enumerate the 2,030 legal PV2F prefixes that still need an action."""

    prefixes: list[tuple[str, ...]] = []
    frontier: list[tuple[str, ...]] = [()]
    while frontier:
        prefix = frontier.pop(0)
        prefixes.append(prefix)
        for action in legal_action_names(prefix):
            candidate = prefix + (action,)
            if not is_terminal(candidate):
                frontier.append(candidate)
    if len(prefixes) != 2030 or len(set(prefixes)) != len(prefixes):
        raise AssertionError((len(prefixes), len(set(prefixes))))
    return tuple(prefixes)


def sparse19_prefix_ids(prefix: tuple[str, ...]) -> tuple[int, int, int]:
    transforms, packer, _compressor = split_prefix(prefix)
    return (
        TRANSFORM_NAMES.index(transforms[0]) if len(transforms) >= 1 else -1,
        TRANSFORM_NAMES.index(transforms[1]) if len(transforms) >= 2 else -1,
        PACKER_NAMES.index(packer) if packer is not None else -1,
    )


class Sparse19FeatureBuilder:
    """Build the 19 logical fields used by the folded inference policy.

    The categorical values are lookup-table indices, never ordinal numeric
    inputs.  f8 can be removed only when it is identical to f6 on every column
    supplied to this bounded builder.
    """

    logical_field_count = 19
    dense_equivalent_dim = 53

    def __init__(self, raw_features: np.ndarray, train_indices: np.ndarray):
        if raw_features.shape[1] != 17:
            raise AssertionError(raw_features.shape)
        if np.any(raw_features[:, 6] != raw_features[:, 8]):
            raise AssertionError("sparse19 requires feature_6 == feature_8 on all supplied columns")
        self.raw_features = raw_features[:, SPARSE19_RAW_INDICES]
        self.mean = self.raw_features[train_indices].mean(axis=0)
        self.std = self.raw_features[train_indices].std(axis=0)
        self.std[self.std < 1.0e-8] = 1.0
        self.normalized_continuous = np.clip(
            (self.raw_features - self.mean) / self.std, -10.0, 10.0
        )
        self.prefixes = legal_nonterminal_prefixes()
        self.prefix_index = {prefix: index for index, prefix in enumerate(self.prefixes)}
        self.prefix_category_ids = np.asarray(
            [sparse19_prefix_ids(prefix) for prefix in self.prefixes], dtype=np.int16
        )
        self.prefix_legal_masks = np.vstack([legal_mask(prefix) for prefix in self.prefixes])
        self.prefix_next_ids = np.full(
            (len(self.prefixes), len(ACTION_NAMES)), -2, dtype=np.int32
        )
        for prefix_id, prefix in enumerate(self.prefixes):
            for action in np.flatnonzero(self.prefix_legal_masks[prefix_id]):
                candidate = prefix + (ACTION_NAMES[int(action)],)
                self.prefix_next_ids[prefix_id, int(action)] = (
                    -1 if is_terminal(candidate) else self.prefix_index[candidate]
                )
        self.prefix_next_id_rows = tuple(
            tuple(int(value) for value in row) for row in self.prefix_next_ids
        )

    def continuous(self, columns: np.ndarray) -> np.ndarray:
        return self.normalized_continuous[columns]

    def one(self, column: int, prefix: tuple[str, ...]) -> Sparse19Observation:
        first, second, packer = self.prefix_category_ids[self.prefix_index[prefix]]
        return self.normalized_continuous[column], int(first), int(second), int(packer)


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


class Sparse19MLP:
    """MLP inference with 16 continuous fields and three categorical lookups."""

    def __init__(self, hidden: int, output_dim: int, rng: np.random.Generator):
        self.params = [
            rng.normal(0.0, math.sqrt(2.0 / 53), (53, hidden)),
            np.zeros(hidden, dtype=np.float64),
            rng.normal(0.0, 0.01, (hidden, output_dim)),
            np.zeros(output_dim, dtype=np.float64),
        ]

    def forward(self, observation: Sparse19Observation) -> tuple[np.ndarray, np.ndarray]:
        continuous, transform_slot_1, transform_slot_2, packer = observation
        w1, b1, w2, b2 = self.params
        preactivation = continuous @ w1[:16] + b1
        if transform_slot_1 >= 0:
            preactivation += w1[16 + transform_slot_1]
        if transform_slot_2 >= 0:
            preactivation += w1[28 + transform_slot_2]
        if packer >= 0:
            preactivation += w1[40 + packer]
        hidden = np.tanh(preactivation)
        return hidden @ w2 + b2, hidden

    def forward_columns(
        self,
        builder: Sparse19FeatureBuilder,
        columns: np.ndarray,
        prefix: tuple[str, ...],
    ) -> np.ndarray:
        transform_slot_1, transform_slot_2, packer = builder.prefix_category_ids[
            builder.prefix_index[prefix]
        ]
        w1, b1, w2, b2 = self.params
        preactivation = builder.continuous(columns) @ w1[:16] + b1
        if transform_slot_1 >= 0:
            preactivation += w1[16 + int(transform_slot_1)]
        if transform_slot_2 >= 0:
            preactivation += w1[28 + int(transform_slot_2)]
        if packer >= 0:
            preactivation += w1[40 + int(packer)]
        return np.tanh(preactivation) @ w2 + b2


class Sparse19InferenceCache:
    """Actor-specific preactivations for cached scalar and batched inference."""

    def __init__(self, actor: Sparse19MLP, builder: Sparse19FeatureBuilder):
        self.builder = builder
        w1, b1, self.w2, self.b2 = actor.params
        self.column_preactivations = builder.normalized_continuous @ w1[:16] + b1
        self.prefix_preactivations = np.zeros(
            (len(builder.prefixes), b1.size), dtype=np.float64
        )
        for prefix_index, (first, second, packer) in enumerate(builder.prefix_category_ids):
            if first >= 0:
                self.prefix_preactivations[prefix_index] += w1[16 + int(first)]
            if second >= 0:
                self.prefix_preactivations[prefix_index] += w1[28 + int(second)]
            if packer >= 0:
                self.prefix_preactivations[prefix_index] += w1[40 + int(packer)]

    def logits(self, column: int, prefix: tuple[str, ...]) -> np.ndarray:
        prefix_index = self.builder.prefix_index[prefix]
        hidden = np.tanh(
            self.column_preactivations[column] + self.prefix_preactivations[prefix_index]
        )
        return hidden @ self.w2 + self.b2

    def logits_prefix_ids(self, column: int, prefix_ids: np.ndarray) -> np.ndarray:
        hidden = np.tanh(
            self.column_preactivations[column] + self.prefix_preactivations[prefix_ids]
        )
        return hidden @ self.w2 + self.b2


def fold_full60_first_layer(
    full_w1: np.ndarray,
    full_b1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Algebraically fold phase/depth one-hots and duplicate f8 into 53 weights."""

    if full_w1.shape[0] != 60:
        raise AssertionError(full_w1.shape)
    compact = np.zeros((53, full_w1.shape[1]), dtype=full_w1.dtype)
    for new_index, old_index in enumerate(SPARSE19_RAW_INDICES):
        compact[new_index] = full_w1[old_index]
    compact[SPARSE19_RAW_INDICES.index(6)] += full_w1[8]

    phase0, phase1 = full_w1[17], full_w1[18]
    depth0, depth1, depth2 = full_w1[20], full_w1[21], full_w1[22]
    for action in range(len(TRANSFORM_NAMES)):
        compact[16 + action] = full_w1[23 + action] - depth0 + depth1
        compact[28 + action] = full_w1[35 + action] - depth1 + depth2
    for action in range(len(PACKER_NAMES)):
        compact[40 + action] = full_w1[47 + action] - phase0 + phase1
    return compact, full_b1 + phase0 + depth0


def fold_full60_actor(actor: MLP) -> Sparse19MLP:
    """Create an exactly equivalent Sparse19 inference actor for bounded data."""

    hidden = int(actor.params[1].shape[0])
    output_dim = int(actor.params[3].shape[0])
    compact = Sparse19MLP(hidden, output_dim, np.random.default_rng(0))
    compact_w1, compact_b1 = fold_full60_first_layer(actor.params[0], actor.params[1])
    compact.params = [compact_w1, compact_b1, actor.params[2].copy(), actor.params[3].copy()]
    return compact


class Adam:
    def __init__(self, params: list[np.ndarray], learning_rate: float):
        self.params = params
        self.learning_rate = learning_rate
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
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1.0e-8)


def legal_mask(prefix: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(len(ACTION_NAMES), dtype=bool)
    for name in legal_action_names(prefix):
        mask[ACTION_INDEX[name]] = True
    return mask


def distribution(actor: MLP, features: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits, hidden = actor.forward(features)
    logits = np.where(masks, logits, -1.0e9)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return hidden, probabilities, logits


def sample_actions(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    uniforms = rng.random(probabilities.shape[0])
    return np.sum(np.cumsum(probabilities, axis=1) < uniforms[:, None], axis=1).astype(np.int16)


def collect_episodes(
    tables: list[ColumnTable],
    builder: FeatureBuilder,
    actor: MLP,
    rng: np.random.Generator,
    episode_columns: np.ndarray,
    critic: MLP | None,
) -> dict[str, Any]:
    episode_count = episode_columns.size
    feature_tensor = np.zeros((episode_count, MAX_STEPS, builder.input_dim), dtype=np.float64)
    legal_tensor = np.zeros((episode_count, MAX_STEPS, len(ACTION_NAMES)), dtype=bool)
    action_tensor = np.full((episode_count, MAX_STEPS), -1, dtype=np.int16)
    logp_tensor = np.zeros((episode_count, MAX_STEPS), dtype=np.float64)
    reward_tensor = np.zeros((episode_count, MAX_STEPS), dtype=np.float64)
    value_tensor = np.zeros((episode_count, MAX_STEPS), dtype=np.float64)
    active_tensor = np.zeros((episode_count, MAX_STEPS), dtype=bool)
    prefixes: list[tuple[str, ...]] = [() for _ in range(episode_count)]
    active = np.ones(episode_count, dtype=bool)
    for step in range(MAX_STEPS):
        episodes = np.flatnonzero(active)
        if episodes.size == 0:
            break
        features = np.vstack([builder.one(int(episode_columns[index]), prefixes[index]) for index in episodes])
        masks = np.vstack([legal_mask(prefixes[index]) for index in episodes])
        hidden, probabilities, _logits = distribution(actor, features, masks)
        actions = sample_actions(rng, probabilities)
        feature_tensor[episodes, step] = features
        legal_tensor[episodes, step] = masks
        action_tensor[episodes, step] = actions
        logp_tensor[episodes, step] = np.log(
            np.maximum(probabilities[np.arange(episodes.size), actions], 1.0e-30)
        )
        active_tensor[episodes, step] = True
        if critic is not None:
            value_tensor[episodes, step] = critic.forward(features)[0][:, 0]
        for local, episode in enumerate(episodes):
            name = ACTION_NAMES[int(actions[local])]
            prefix = prefixes[int(episode)] + (name,)
            prefixes[int(episode)] = prefix
            if is_terminal(prefix):
                reward_tensor[int(episode), step] = -lookup_rate(tables[int(episode_columns[int(episode)])], prefix)
                active[int(episode)] = False
    if np.any(active):
        raise AssertionError("episode exceeded typed horizon")
    lengths = active_tensor.sum(axis=1)
    flat_episode, _flat_step = np.nonzero(active_tensor)
    return {
        "features": feature_tensor[active_tensor],
        "legal": legal_tensor[active_tensor],
        "actions": action_tensor[active_tensor],
        "old_logp": logp_tensor[active_tensor],
        "rewards": reward_tensor[active_tensor],
        "old_values": value_tensor[active_tensor],
        "episode": flat_episode.astype(np.int32),
        "lengths": lengths,
        "episode_returns": reward_tensor.sum(axis=1),
        "active": active_tensor,
        "reward_matrix": reward_tensor,
        "value_matrix": value_tensor,
        "pipelines": prefixes,
    }


def collect_ppo(
    tables: list[ColumnTable],
    builder: FeatureBuilder,
    actor: MLP,
    critic: MLP,
    rng: np.random.Generator,
    train_columns: np.ndarray,
    episodes: int,
    gae_lambda: float,
) -> dict[str, Any]:
    episode_columns = rng.choice(train_columns, size=episodes, replace=True)
    rollout = collect_episodes(tables, builder, actor, rng, episode_columns, critic)
    advantages = np.zeros_like(rollout["reward_matrix"])
    targets = np.zeros_like(rollout["reward_matrix"])
    for episode in range(episodes):
        next_advantage = 0.0
        next_value = 0.0
        for step in range(int(rollout["lengths"][episode]) - 1, -1, -1):
            value = rollout["value_matrix"][episode, step]
            delta = rollout["reward_matrix"][episode, step] + next_value - value
            next_advantage = delta + gae_lambda * next_advantage
            advantages[episode, step] = next_advantage
            targets[episode, step] = next_advantage + value
            next_value = value
    rollout["advantages"] = advantages[rollout["active"]]
    rollout["targets"] = targets[rollout["active"]]
    rollout["weights"] = np.ones(rollout["actions"].size, dtype=np.float64)
    return rollout


def collect_grpo(
    tables: list[ColumnTable],
    builder: FeatureBuilder,
    actor: MLP,
    rng: np.random.Generator,
    train_columns: np.ndarray,
    groups: int,
    group_size: int,
) -> dict[str, Any]:
    group_columns = rng.choice(train_columns, size=groups, replace=True)
    episode_columns = np.repeat(group_columns, group_size)
    rollout = collect_episodes(tables, builder, actor, rng, episode_columns, None)
    normalized = np.zeros(groups * group_size, dtype=np.float64)
    informative = 0
    for group in range(groups):
        start = group * group_size
        stop = start + group_size
        returns = rollout["episode_returns"][start:stop]
        standard_deviation = float(np.std(returns))
        if standard_deviation > 1.0e-8:
            normalized[start:stop] = (returns - float(np.mean(returns))) / (standard_deviation + 1.0e-8)
            informative += 1
    rollout["advantages"] = normalized[rollout["episode"]]
    rollout["targets"] = None
    rollout["weights"] = np.ones(rollout["actions"].size, dtype=np.float64)
    rollout["informative_groups"] = informative
    return rollout


def actor_update(
    actor: MLP,
    optimizer: Adam,
    rollout: dict[str, Any],
    rng: np.random.Generator,
    epochs: int,
    minibatch: int,
    clip_ratio: float,
    entropy_coefficient: float,
) -> float:
    features = rollout["features"]
    legal = rollout["legal"]
    actions = rollout["actions"]
    old_logp = rollout["old_logp"]
    advantages = rollout["advantages"].astype(np.float64, copy=True)
    weights = rollout["weights"].astype(np.float64, copy=True)
    if float(np.std(advantages)) > 1.0e-8:
        advantages = (advantages - float(np.mean(advantages))) / (float(np.std(advantages)) + 1.0e-8)
    indices = np.arange(actions.size)
    clipped: list[float] = []
    for _ in range(epochs):
        rng.shuffle(indices)
        for offset in range(0, indices.size, minibatch):
            take = indices[offset : offset + minibatch]
            hidden, probabilities, _logits = distribution(actor, features[take], legal[take])
            selected = probabilities[np.arange(take.size), actions[take]]
            ratio = np.exp(np.log(np.maximum(selected, 1.0e-30)) - old_logp[take])
            advantage = advantages[take]
            active = ((advantage >= 0) & (ratio <= 1.0 + clip_ratio)) | (
                (advantage < 0) & (ratio >= 1.0 - clip_ratio)
            )
            local_weight = weights[take].copy()
            local_weight /= max(float(np.sum(local_weight)), 1.0e-30)
            coefficient = np.where(active, -advantage * ratio * local_weight, 0.0)
            grad_logits = -coefficient[:, None] * probabilities
            grad_logits[np.arange(take.size), actions[take]] += coefficient
            log_probabilities = np.log(np.maximum(probabilities, 1.0e-30))
            entropy = -np.sum(probabilities * log_probabilities, axis=1, keepdims=True)
            grad_logits += entropy_coefficient * local_weight[:, None] * probabilities * (
                log_probabilities + entropy
            )
            optimizer.step(actor.backward(features[take], hidden, grad_logits))
            clipped.append(float(np.mean(~active)))
    return float(np.mean(clipped)) if clipped else 0.0


def critic_update(
    critic: MLP,
    optimizer: Adam,
    rollout: dict[str, Any],
    rng: np.random.Generator,
    epochs: int,
    minibatch: int,
) -> None:
    features = rollout["features"]
    targets = rollout["targets"]
    indices = np.arange(features.shape[0])
    for _ in range(epochs):
        rng.shuffle(indices)
        for offset in range(0, indices.size, minibatch):
            take = indices[offset : offset + minibatch]
            output, hidden = critic.forward(features[take])
            grad_output = 2.0 * (output[:, 0] - targets[take])[:, None] / max(1, take.size)
            optimizer.step(critic.backward(features[take], hidden, grad_output))


def propose_pipeline(
    actor: MLP,
    builder: FeatureBuilder,
    column: int,
    rng: np.random.Generator | None,
) -> tuple[str, ...]:
    prefix: tuple[str, ...] = ()
    for _ in range(MAX_STEPS):
        feature = builder.one(column, prefix)[None, :]
        mask = legal_mask(prefix)[None, :]
        _hidden, probabilities, _logits = distribution(actor, feature, mask)
        if rng is None:
            action = int(np.argmax(probabilities[0]))
        else:
            action = int(sample_actions(rng, probabilities)[0])
        prefix += (ACTION_NAMES[action],)
        if is_terminal(prefix):
            return prefix
    raise AssertionError("proposal exceeded typed horizon")


def propose_pipeline_sparse19(
    actor: Sparse19MLP,
    builder: Sparse19FeatureBuilder,
    column: int,
    rng: np.random.Generator | None,
) -> tuple[str, ...]:
    """Sample one legal pipeline directly from the 19 logical fields."""

    prefix: tuple[str, ...] = ()
    for _ in range(MAX_STEPS):
        logits, _hidden = actor.forward(builder.one(column, prefix))
        mask = legal_mask(prefix)
        masked_logits = np.where(mask, logits, -1.0e9)
        masked_logits -= np.max(masked_logits)
        probabilities = np.exp(masked_logits)
        probabilities /= np.sum(probabilities)
        if rng is None:
            action = int(np.argmax(probabilities))
        else:
            action = int(sample_actions(rng, probabilities[None, :])[0])
        prefix += (ACTION_NAMES[action],)
        if is_terminal(prefix):
            return prefix
    raise AssertionError("sparse19 proposal exceeded typed horizon")


def sparse19_probabilities(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_logits = np.where(mask, logits, -1.0e9)
    masked_logits -= np.max(masked_logits, axis=-1, keepdims=True)
    probabilities = np.exp(masked_logits)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def propose_pipeline_sparse19_cached(
    cache: Sparse19InferenceCache,
    column: int,
    rng: np.random.Generator | None,
    probability_cache: dict[int, np.ndarray] | None = None,
) -> tuple[str, ...]:
    """Scalar sampler with the original RNG schedule and no hot-path objects."""

    prefix: tuple[str, ...] = ()
    for _ in range(MAX_STEPS):
        prefix_id = cache.builder.prefix_index[prefix]
        probabilities = (
            probability_cache.get(prefix_id) if probability_cache is not None else None
        )
        if probabilities is None:
            probabilities = sparse19_probabilities(
                cache.logits(column, prefix), cache.builder.prefix_legal_masks[prefix_id]
            )
            if probability_cache is not None:
                probability_cache[prefix_id] = probabilities
        if rng is None:
            action = int(np.argmax(probabilities))
        else:
            uniform = float(rng.random())
            action = int(np.sum(np.cumsum(probabilities) < uniform))
        prefix += (ACTION_NAMES[action],)
        if is_terminal(prefix):
            return prefix
    raise AssertionError("cached sparse19 proposal exceeded typed horizon")


def propose_pipelines_sparse19_batched(
    cache: Sparse19InferenceCache,
    column: int,
    uniforms: np.ndarray,
) -> tuple[set[tuple[str, ...]], dict[str, int]]:
    """Sample rollouts by step and evaluate each repeated prefix only once."""

    if uniforms.ndim != 2 or uniforms.shape[1] != MAX_STEPS:
        raise ValueError(f"expected [rollouts,{MAX_STEPS}] uniforms, got {uniforms.shape}")
    rollout_count = int(uniforms.shape[0])
    prefix_ids = np.zeros(rollout_count, dtype=np.int32)
    active = np.ones(rollout_count, dtype=bool)
    pipelines: list[tuple[str, ...] | None] = [None] * rollout_count
    trajectory_prefix_evaluations = 0
    unique_prefix_evaluations = 0

    for step in range(MAX_STEPS):
        episodes = np.flatnonzero(active)
        if episodes.size == 0:
            break
        active_prefix_ids = prefix_ids[episodes]
        unique_ids, inverse = np.unique(active_prefix_ids, return_inverse=True)
        trajectory_prefix_evaluations += int(episodes.size)
        unique_prefix_evaluations += int(unique_ids.size)
        probabilities = sparse19_probabilities(
            cache.logits_prefix_ids(column, unique_ids),
            cache.builder.prefix_legal_masks[unique_ids],
        )
        episode_probabilities = probabilities[inverse]
        actions = np.sum(
            np.cumsum(episode_probabilities, axis=1) < uniforms[episodes, step, None],
            axis=1,
        ).astype(np.int16)
        for local, episode in enumerate(episodes):
            prefix = cache.builder.prefixes[int(active_prefix_ids[local])]
            candidate = prefix + (ACTION_NAMES[int(actions[local])],)
            if is_terminal(candidate):
                pipelines[int(episode)] = candidate
                active[int(episode)] = False
            else:
                prefix_ids[int(episode)] = cache.builder.prefix_index[candidate]

    if np.any(active) or any(pipeline is None for pipeline in pipelines):
        raise AssertionError("batched sparse19 proposal exceeded typed horizon")
    proposal_set = {pipeline for pipeline in pipelines if pipeline is not None}
    return proposal_set, {
        "rollouts": rollout_count,
        "trajectory_prefix_evaluations": trajectory_prefix_evaluations,
        "unique_prefix_evaluations": unique_prefix_evaluations,
        "merged_prefix_evaluations": trajectory_prefix_evaluations - unique_prefix_evaluations,
    }


def propose_pipelines_full60_batched(
    actor: MLP,
    builder: FeatureBuilder,
    column: int,
    uniforms: np.ndarray,
) -> tuple[set[tuple[str, ...]], dict[str, int]]:
    """Full60 semantic reference for the fixed-uniform batched sampler."""

    if uniforms.ndim != 2 or uniforms.shape[1] != MAX_STEPS:
        raise ValueError(f"expected [rollouts,{MAX_STEPS}] uniforms, got {uniforms.shape}")
    rollout_count = int(uniforms.shape[0])
    prefixes: list[tuple[str, ...]] = [()] * rollout_count
    active = np.ones(rollout_count, dtype=bool)
    trajectory_prefix_evaluations = 0
    unique_prefix_evaluations = 0

    for step in range(MAX_STEPS):
        episodes = np.flatnonzero(active)
        if episodes.size == 0:
            break
        grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for episode in episodes:
            grouped[prefixes[int(episode)]].append(int(episode))
        unique_prefixes = tuple(grouped)
        features = np.vstack([builder.one(column, prefix) for prefix in unique_prefixes])
        masks = np.vstack([legal_mask(prefix) for prefix in unique_prefixes])
        _hidden, probabilities, _logits = distribution(actor, features, masks)
        trajectory_prefix_evaluations += int(episodes.size)
        unique_prefix_evaluations += len(unique_prefixes)
        for unique_index, prefix in enumerate(unique_prefixes):
            selected_episodes = np.asarray(grouped[prefix], dtype=np.int32)
            selected_probabilities = probabilities[unique_index]
            actions = np.sum(
                np.cumsum(selected_probabilities)[None, :]
                < uniforms[selected_episodes, step, None],
                axis=1,
            ).astype(np.int16)
            for local, episode in enumerate(selected_episodes):
                candidate = prefix + (ACTION_NAMES[int(actions[local])],)
                prefixes[int(episode)] = candidate
                if is_terminal(candidate):
                    active[int(episode)] = False

    if np.any(active):
        raise AssertionError("batched full60 proposal exceeded typed horizon")
    return set(prefixes), {
        "rollouts": rollout_count,
        "trajectory_prefix_evaluations": trajectory_prefix_evaluations,
        "unique_prefix_evaluations": unique_prefix_evaluations,
        "merged_prefix_evaluations": trajectory_prefix_evaluations - unique_prefix_evaluations,
    }


def evaluate_policy(
    algorithm: str,
    seed: int,
    fold: int,
    actor: MLP,
    builder: FeatureBuilder,
    tables: list[ColumnTable],
    test_columns: np.ndarray,
    proposal_budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in test_columns:
        column = int(column)
        table = tables[column]
        greedy = propose_pipeline(actor, builder, column, None)
        proposal_rng = np.random.default_rng(seed + 10_000_019 * (fold + 1) + 1009 * (column + 1))
        proposals = {greedy, ("raw.i64", "none")}
        for _ in range(proposal_budget):
            proposals.add(propose_pipeline(actor, builder, column, proposal_rng))
        valid = [pipeline for pipeline in proposals if pipeline in table.sizes]
        if not valid:
            raise AssertionError(f"no valid policy proposal for {table.key}")
        best = min(valid, key=lambda pipeline: (table.sizes[pipeline], pipeline))
        oracle = min(table.sizes, key=lambda pipeline: (table.sizes[pipeline], pipeline))
        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "fold": fold,
                "train_dataset": ",".join(sorted({tables[index].dataset for index in range(len(tables)) if index not in set(test_columns.tolist())})),
                "test_dataset": table.dataset,
                "column_index": table.column_index,
                "column_name": table.column_name,
                "logical_type": table.logical_type,
                "pipeline": " -> ".join(best),
                "encoded_bytes": table.sizes[best],
                "compression_rate": table.rates[best],
                "greedy_pipeline": " -> ".join(greedy),
                "greedy_bytes": lookup_size(table, greedy),
                "greedy_rate": lookup_rate(table, greedy),
                "proposal_rollouts": proposal_budget,
                "unique_proposals": len(proposals),
                "valid_unique_proposals": len(valid),
                "invalid_unique_proposals": len(proposals) - len(valid),
                "oracle_pipeline": " -> ".join(oracle),
                "oracle_bytes": table.sizes[oracle],
                "oracle_rate": table.rates[oracle],
                "oracle_regret_pct": 100.0 * (table.sizes[best] - table.sizes[oracle]) / table.sizes[oracle],
            }
        )
    return rows


def train_algorithm(
    algorithm: str,
    seed: int,
    fold: int,
    tables: list[ColumnTable],
    raw_features: np.ndarray,
    train_columns: np.ndarray,
    test_columns: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], MLP]:
    builder = FeatureBuilder(raw_features, train_columns)
    initialization_rng = np.random.default_rng(seed + 1009 * fold)
    sampling_rng = np.random.default_rng(seed + 1009 * fold + (0 if algorithm == "ppo" else 100_003))
    actor = MLP(builder.input_dim, args.hidden, len(ACTION_NAMES), initialization_rng)
    actor_optimizer = Adam(actor.params, args.actor_lr)
    critic: MLP | None = None
    critic_optimizer: Adam | None = None
    if algorithm == "ppo":
        critic = MLP(builder.input_dim, args.hidden, 1, initialization_rng)
        critic_optimizer = Adam(critic.params, args.critic_lr)
    training_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        if algorithm == "ppo":
            assert critic is not None and critic_optimizer is not None
            rollout = collect_ppo(
                tables,
                builder,
                actor,
                critic,
                sampling_rng,
                train_columns,
                args.ppo_episodes,
                args.gae_lambda,
            )
        else:
            rollout = collect_grpo(
                tables,
                builder,
                actor,
                sampling_rng,
                train_columns,
                args.grpo_groups,
                args.group_size,
            )
        clip_fraction = actor_update(
            actor,
            actor_optimizer,
            rollout,
            sampling_rng,
            args.policy_epochs,
            args.minibatch,
            args.clip_ratio,
            args.entropy_coefficient,
        )
        if algorithm == "ppo":
            assert critic is not None and critic_optimizer is not None
            critic_update(
                critic,
                critic_optimizer,
                rollout,
                sampling_rng,
                args.value_epochs,
                args.minibatch,
            )
        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            returns = rollout["episode_returns"]
            row = {
                "algorithm": algorithm,
                "seed": seed,
                "fold": fold,
                "iteration": iteration,
                "transitions": int(rollout["actions"].size),
                "mean_episode_return": float(np.mean(returns)),
                "median_episode_return": float(np.median(returns)),
                "invalid_episodes": int(np.sum(returns <= -INVALID_RATE)),
                "clip_fraction": clip_fraction,
                "informative_groups": int(rollout.get("informative_groups", 0)),
                "elapsed_sec": time.perf_counter() - started,
            }
            training_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    evaluation = evaluate_policy(
        algorithm,
        seed,
        fold,
        actor,
        builder,
        tables,
        test_columns,
        args.proposal_budget,
    )
    return evaluation, training_rows, actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=29)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260825, 20260826, 20260827))
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ppo-episodes", type=int, default=256)
    parser.add_argument("--grpo-groups", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--value-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=0.001)
    parser.add_argument("--critic-lr", type=float, default=0.001)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--proposal-budget", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if min(args.beam_width, args.iterations, args.hidden, args.ppo_episodes, args.grpo_groups, args.group_size, args.policy_epochs, args.value_epochs, args.minibatch, args.proposal_budget, args.log_every) <= 0:
        parser.error("all count and budget arguments must be positive")
    return args


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_root = args.measurement_root.resolve()
    tables = load_tables(measurement_root)
    datasets = sorted({table.dataset for table in tables})
    if datasets != ["Air-Quality", "Metro-Traffic"]:
        raise AssertionError(f"expected Air-Quality and Metro-Traffic tables, got {datasets}")
    raw_features, feature_rows = load_feature_matrix(repo_root, tables)
    write_csv(output_dir / "column_features.csv", feature_rows)
    beam_rows: list[dict[str, Any]] = []
    beam_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    level_rows: list[dict[str, Any]] = []
    for table in tables:
        result = beam29(table, args.beam_width)
        beam_by_key[table.key] = result
        beam_rows.append(
            {
                "dataset": table.dataset,
                "column_index": table.column_index,
                "column_name": table.column_name,
                "logical_type": table.logical_type,
                "original_bytes": table.original_bytes,
                "beam_width": args.beam_width,
                "pipeline": " -> ".join(result["pipeline"]),
                "encoded_bytes": result["bytes"],
                "compression_rate": result["rate"],
                "candidate_evaluations": result["candidate_evaluations"],
                "invalid_evaluations": result["invalid_evaluations"],
                "oracle_pipeline": " -> ".join(result["oracle_pipeline"]),
                "oracle_bytes": result["oracle_bytes"],
                "oracle_rate": result["oracle_rate"],
                "oracle_regret_pct": 100.0 * (result["bytes"] - result["oracle_bytes"]) / result["oracle_bytes"],
                "complete_valid_pipelines": len(table.sizes),
                "complete_failed_pipelines": table.failures,
            }
        )
        for level in result["levels"]:
            level_rows.append(
                {
                    "dataset": table.dataset,
                    "column_index": table.column_index,
                    "column_name": table.column_name,
                    **level,
                }
            )
    write_csv(output_dir / "beam29_results.csv", beam_rows)
    write_csv(output_dir / "beam29_levels.csv", level_rows)

    evaluation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    policy_dir = output_dir / "policies"
    policy_dir.mkdir()
    for fold, test_dataset in enumerate(datasets):
        test_columns = np.asarray([index for index, table in enumerate(tables) if table.dataset == test_dataset], dtype=np.int32)
        train_columns = np.asarray([index for index, table in enumerate(tables) if table.dataset != test_dataset], dtype=np.int32)
        if not train_columns.size or not test_columns.size:
            raise AssertionError("empty cross-dataset fold")
        for seed in args.seeds:
            for algorithm in ("ppo", "grpo"):
                evaluation, training, actor = train_algorithm(
                    algorithm,
                    seed,
                    fold,
                    tables,
                    raw_features,
                    train_columns,
                    test_columns,
                    args,
                )
                for row in evaluation:
                    beam = beam_by_key[(row["test_dataset"], int(row["column_index"]))]
                    row["beam29_pipeline"] = " -> ".join(beam["pipeline"])
                    row["beam29_bytes"] = beam["bytes"]
                    row["beam29_rate"] = beam["rate"]
                    row["gain_vs_beam29_pct"] = 100.0 * (beam["bytes"] - int(row["encoded_bytes"])) / beam["bytes"]
                evaluation_rows.extend(evaluation)
                training_rows.extend(training)
                np.savez_compressed(
                    policy_dir / f"{algorithm}_seed{seed}_fold{fold}.npz",
                    *actor.params,
                )
    write_csv(output_dir / "rl_evaluation.csv", evaluation_rows)
    write_csv(output_dir / "training_curve.csv", training_rows)

    per_algorithm: list[dict[str, Any]] = []
    for algorithm in ("ppo", "grpo"):
        rows = [row for row in evaluation_rows if row["algorithm"] == algorithm]
        per_seed_total: dict[int, tuple[int, int, int]] = {}
        for seed in args.seeds:
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            per_seed_total[seed] = (
                sum(int(row["encoded_bytes"]) for row in seed_rows),
                sum(int(row["beam29_bytes"]) for row in seed_rows),
                sum(int(row["oracle_bytes"]) for row in seed_rows),
            )
        rl_totals = [values[0] for values in per_seed_total.values()]
        beam_totals = [values[1] for values in per_seed_total.values()]
        oracle_totals = [values[2] for values in per_seed_total.values()]
        per_algorithm.append(
            {
                "algorithm": algorithm,
                "test_columns_per_seed": len(rows) // len(args.seeds),
                "seeds": len(args.seeds),
                "median_total_bytes": statistics.median(rl_totals),
                "median_beam29_total_bytes": statistics.median(beam_totals),
                "median_oracle_total_bytes": statistics.median(oracle_totals),
                "median_gain_vs_beam29_pct": 100.0 * (statistics.median(beam_totals) - statistics.median(rl_totals)) / statistics.median(beam_totals),
                "median_oracle_regret_pct": 100.0 * (statistics.median(rl_totals) - statistics.median(oracle_totals)) / statistics.median(oracle_totals),
                "median_unique_proposals": statistics.median(int(row["unique_proposals"]) for row in rows),
                "median_invalid_unique_proposals": statistics.median(int(row["invalid_unique_proposals"]) for row in rows),
            }
        )
    write_csv(output_dir / "algorithm_summary.csv", per_algorithm)
    beam_total = sum(int(row["encoded_bytes"]) for row in beam_rows)
    oracle_total = sum(int(row["oracle_bytes"]) for row in beam_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "real_gt_normalized_int64_physical_prototype",
        "datasets": datasets,
        "columns": len(tables),
        "action_counts": {
            "transforms": len(TRANSFORM_NAMES),
            "packers": len(PACKER_NAMES),
            "compressors": len(COMPRESSOR_NAMES),
            "total": len(ACTION_NAMES),
        },
        "complete_candidates_per_column": (1 + len(TRANSFORM_NAMES) + len(TRANSFORM_NAMES) * (len(TRANSFORM_NAMES) - 1)) * len(PACKER_NAMES) * len(COMPRESSOR_NAMES),
        "beam29": {
            "total_bytes": beam_total,
            "oracle_total_bytes": oracle_total,
            "aggregate_oracle_regret_pct": 100.0 * (beam_total - oracle_total) / oracle_total,
            "median_column_oracle_regret_pct": statistics.median(float(row["oracle_regret_pct"]) for row in beam_rows),
            "total_candidate_evaluations": sum(int(row["candidate_evaluations"]) for row in beam_rows),
        },
        "algorithms": per_algorithm,
        "split": "two cross-dataset folds: train one of Metro-Traffic/Air-Quality and test the other",
        "seeds": list(args.seeds),
        "proposal_budget": args.proposal_budget,
        "registry": registry()["counts"],
        "claim_boundary": "standalone normalized-int64 physical prototype; RL test selection is frozen cross-dataset but only two datasets are in scope",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": summary["created_utc"],
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "runtime": sha256_file(Path(__file__).with_name("physical_operator_runtime_v2.py")),
            "enumerator": sha256_file(Path(__file__).with_name("enumerate_physical_operator_variants_v2.py")),
            "beam29_results": sha256_file(output_dir / "beam29_results.csv"),
            "rl_evaluation": sha256_file(output_dir / "rl_evaluation.csv"),
            "summary": sha256_file(output_dir / "summary.json"),
        },
        "measurement_inputs": [str(table.source_csv) for table in tables],
        "output_rows": {
            "beam": len(beam_rows),
            "rl_evaluation": len(evaluation_rows),
            "training": len(training_rows),
        },
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"finished": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
