#!/usr/bin/env python3
"""Leakage-aware PPO feature ablation for the bounded PV2F search task.

The script reuses the frozen real-byte tables and PPO implementation from
``search_physical_operator_variants_v2.py``.  It changes only the policy
observation.  Typed legality, pipeline lookup, rewards, and candidate bytes are
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RAW_FEATURE_NAMES = (
    "log_length",
    "signed_log_mean",
    "log_std",
    "signed_log_min",
    "signed_log_max",
    "unique_ratio",
    "run_ratio_int64",
    "monotonic_ratio",
    "delta_zero_ratio_float_duplicate",
    "log_mean_abs_delta",
    "log_std_delta",
    "log_std_delta2",
    "span_bit_width",
    "zero_ratio",
    "negative_ratio",
    "log_p99_p01_range",
    "normalized_entropy",
)

# Blocks follow the actual computation DAG.  Feature 8 is deliberately absent:
# on the observed columns it duplicates feature 6 and is less safe at int64
# precision edges because it first converts values to float64.
RAW_BLOCKS = (
    ("length", (0,)),
    ("moments_range_sign", (1, 2, 3, 4, 12, 13, 14)),
    ("cardinality", (5, 16)),
    ("runs", (6,)),
    ("delta_order", (7, 9, 10, 11)),
    ("percentile", (15,)),
)
RAW_BLOCK_BY_NAME = dict(RAW_BLOCKS)


def load_module(path: Path, name: str):
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_raw_dims(group_mask: int, include_duplicate_feature8: bool) -> tuple[int, ...]:
    dims: set[int] = set()
    for bit, (_name, members) in enumerate(RAW_BLOCKS):
        if group_mask & (1 << bit):
            dims.update(members)
    if include_duplicate_feature8:
        dims.add(8)
    return tuple(sorted(dims))


def signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def extract_selected(values: np.ndarray, dims: tuple[int, ...]) -> np.ndarray:
    """Compute only requested coordinates, skipping unused expensive blocks."""
    values = np.asarray(values, dtype=np.int64)
    wanted = set(dims)
    out: dict[int, float] = {}
    floating: np.ndarray | None = None
    deltas: np.ndarray | None = None
    unique_counts: np.ndarray | None = None

    if 0 in wanted:
        out[0] = math.log1p(values.size)

    floating_dims = {1, 2, 3, 4, 7, 8, 9, 10, 11, 15}
    if wanted & floating_dims:
        floating = values.astype(np.float64)
    assert floating is not None or not (wanted & floating_dims)

    if 1 in wanted:
        out[1] = signed_log1p(float(np.mean(floating))) if values.size else 0.0
    if 2 in wanted:
        out[2] = math.log1p(float(np.std(floating))) if values.size else 0.0
    if 3 in wanted:
        out[3] = signed_log1p(float(np.min(floating))) if values.size else 0.0
    if 4 in wanted:
        out[4] = signed_log1p(float(np.max(floating))) if values.size else 0.0

    if wanted & {5, 16}:
        unique_counts = np.unique(values, return_counts=True)[1]
    if 5 in wanted:
        out[5] = unique_counts.size / max(1, values.size)
    if 16 in wanted:
        probabilities = unique_counts.astype(np.float64) / max(1, values.size)
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        out[16] = entropy / max(1.0, math.log2(max(2, values.size)))

    if 6 in wanted:
        out[6] = float(np.mean(values[1:] == values[:-1])) if values.size > 1 else 1.0

    if wanted & {7, 8, 9, 10, 11}:
        deltas = np.diff(floating)
    if 7 in wanted:
        out[7] = float(np.mean(deltas >= 0)) if deltas.size else 1.0
    if 8 in wanted:
        out[8] = float(np.mean(deltas == 0)) if deltas.size else 1.0
    if 9 in wanted:
        out[9] = math.log1p(float(np.mean(np.abs(deltas)))) if deltas.size else 0.0
    if 10 in wanted:
        out[10] = math.log1p(float(np.std(deltas))) if deltas.size else 0.0
    if 11 in wanted:
        delta2 = np.diff(deltas)
        out[11] = math.log1p(float(np.std(delta2))) if delta2.size else 0.0

    if 12 in wanted:
        span = int(values.max()) - int(values.min()) if values.size else 0
        out[12] = max(0, int(span).bit_length()) / 64.0
    if 13 in wanted:
        out[13] = float(np.mean(values == 0)) if values.size else 0.0
    if 14 in wanted:
        out[14] = float(np.mean(values < 0)) if values.size else 0.0
    if 15 in wanted:
        p01, p99 = np.percentile(floating, [1.0, 99.0]) if values.size else (0.0, 0.0)
        out[15] = math.log1p(max(0.0, float(p99 - p01)))
    return np.asarray([out[index] for index in dims], dtype=np.float64)


class AblationFeatureBuilder:
    def __init__(
        self,
        base,
        raw_features: np.ndarray,
        train_indices: np.ndarray,
        raw_dims: tuple[int, ...],
        prefix_mode: str,
        embedding_dim: int,
    ):
        self.base = base
        self.raw_dims = raw_dims
        self.prefix_mode = prefix_mode
        self.embedding_dim = embedding_dim
        selected = raw_features[:, raw_dims] if raw_dims else np.zeros((raw_features.shape[0], 0))
        self.raw_features = selected
        self.mean = selected[train_indices].mean(axis=0) if raw_dims else np.zeros(0)
        self.std = selected[train_indices].std(axis=0) if raw_dims else np.zeros(0)
        self.std[self.std < 1.0e-8] = 1.0
        self.prefix_cache: dict[tuple[str, ...], np.ndarray] = {}

    @property
    def input_dim(self) -> int:
        prefix_dims = {
            "full": 3 + (self.base.MAX_TRANSFORMS + 1) + self.base.MAX_TRANSFORMS * len(self.base.TRANSFORM_NAMES) + len(self.base.PACKER_NAMES),
            "history": self.base.MAX_TRANSFORMS * len(self.base.TRANSFORM_NAMES) + len(self.base.PACKER_NAMES),
            "transform": self.base.MAX_TRANSFORMS * len(self.base.TRANSFORM_NAMES),
            "packer": len(self.base.PACKER_NAMES),
            "none": 0,
            "embedding": 3,
        }[self.prefix_mode]
        total = len(self.raw_dims) + prefix_dims
        return max(1, total)

    def prefix(self, prefix: tuple[str, ...]) -> np.ndarray:
        transforms, packer, compressor = self.base.split_prefix(prefix)
        if self.prefix_mode == "full":
            return self.base.prefix_features(prefix)
        transform_history = np.zeros(self.base.MAX_TRANSFORMS * len(self.base.TRANSFORM_NAMES))
        for position, name in enumerate(transforms):
            transform_history[position * len(self.base.TRANSFORM_NAMES) + self.base.TRANSFORM_NAMES.index(name)] = 1.0
        packer_history = np.zeros(len(self.base.PACKER_NAMES))
        if packer is not None:
            packer_history[self.base.PACKER_NAMES.index(packer)] = 1.0
        if self.prefix_mode == "history":
            return np.concatenate((transform_history, packer_history))
        if self.prefix_mode == "transform":
            return transform_history
        if self.prefix_mode == "packer":
            return packer_history
        if self.prefix_mode == "none":
            return np.zeros(0)
        if self.prefix_mode == "embedding":
            tokens = [0, 0, 0]
            for position, name in enumerate(transforms):
                tokens[position] = self.base.TRANSFORM_NAMES.index(name) + 1
            if packer is not None:
                tokens[2] = self.base.PACKER_NAMES.index(packer) + 1
            return np.asarray(tokens, dtype=np.float64)
        raise AssertionError(self.prefix_mode)

    def one(self, column: int, prefix: tuple[str, ...]) -> np.ndarray:
        if prefix not in self.prefix_cache:
            self.prefix_cache[prefix] = self.prefix(prefix)
        raw = self.raw_features[column]
        normalized = np.clip((raw - self.mean) / self.std, -10.0, 10.0) if raw.size else raw
        result = np.concatenate((normalized, self.prefix_cache[prefix]))
        return result if result.size else np.zeros(1, dtype=np.float64)


def embedding_mlp_class(base, raw_count: int, embedding_dim: int):
    class EmbeddingMLP:
        def __init__(self, input_dim: int, hidden: int, output_dim: int, rng: np.random.Generator):
            if input_dim != raw_count + 3:
                raise AssertionError((input_dim, raw_count))
            self.raw_count = raw_count
            self.embedding_dim = embedding_dim
            dense_dim = raw_count + 3 * embedding_dim
            self.params = [
                rng.normal(0.0, math.sqrt(2.0 / max(1, dense_dim)), (dense_dim, hidden)),
                np.zeros(hidden, dtype=np.float64),
                rng.normal(0.0, 0.01, (hidden, output_dim)),
                np.zeros(output_dim, dtype=np.float64),
                rng.normal(0.0, 0.05, (len(base.TRANSFORM_NAMES) + 1, embedding_dim)),
                rng.normal(0.0, 0.05, (len(base.TRANSFORM_NAMES) + 1, embedding_dim)),
                rng.normal(0.0, 0.05, (len(base.PACKER_NAMES) + 1, embedding_dim)),
            ]

        def expanded(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
            token_matrix = np.rint(x[:, self.raw_count : self.raw_count + 3]).astype(np.int64)
            tokens = (token_matrix[:, 0], token_matrix[:, 1], token_matrix[:, 2])
            dense = np.concatenate(
                (x[:, : self.raw_count], self.params[4][tokens[0]], self.params[5][tokens[1]], self.params[6][tokens[2]]),
                axis=1,
            )
            return dense, tokens

        def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dense, _tokens = self.expanded(x)
            hidden = np.tanh(dense @ self.params[0] + self.params[1])
            return hidden @ self.params[2] + self.params[3], hidden

        def backward(self, x: np.ndarray, hidden: np.ndarray, output_grad: np.ndarray) -> list[np.ndarray]:
            dense, tokens = self.expanded(x)
            grad_w2 = hidden.T @ output_grad
            grad_b2 = output_grad.sum(axis=0)
            hidden_grad = (output_grad @ self.params[2].T) * (1.0 - hidden * hidden)
            grad_dense = hidden_grad @ self.params[0].T
            grad_embeddings = [np.zeros_like(self.params[index]) for index in (4, 5, 6)]
            offset = self.raw_count
            for slot in range(3):
                block = grad_dense[:, offset : offset + self.embedding_dim]
                np.add.at(grad_embeddings[slot], tokens[slot], block)
                offset += self.embedding_dim
            return [dense.T @ hidden_grad, hidden_grad.sum(axis=0), grad_w2, grad_b2, *grad_embeddings]

    return EmbeddingMLP


def split_indices(tables, protocol: str):
    if protocol == "air_loco":
        air = np.asarray([index for index, table in enumerate(tables) if table.dataset == "Air-Quality"], dtype=np.int32)
        return [
            (np.asarray([index for index in air if index != held], dtype=np.int32), np.asarray([held], dtype=np.int32))
            for held in air
        ]
    if protocol == "air_to_metro":
        train = np.asarray([index for index, table in enumerate(tables) if table.dataset == "Air-Quality"], dtype=np.int32)
        test = np.asarray([index for index, table in enumerate(tables) if table.dataset == "Metro-Traffic"], dtype=np.int32)
        return [(train, test)]
    if protocol == "cross_dataset":
        datasets = sorted({table.dataset for table in tables})
        return [
            (
                np.asarray([index for index, table in enumerate(tables) if table.dataset != test_dataset], dtype=np.int32),
                np.asarray([index for index, table in enumerate(tables) if table.dataset == test_dataset], dtype=np.int32),
            )
            for test_dataset in datasets
        ]
    raise AssertionError(protocol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--group-mask", type=int, default=63)
    parser.add_argument("--raw-dims", type=int, nargs="*", default=None)
    parser.add_argument("--prefix-mode", choices=("full", "history", "transform", "packer", "none", "embedding"), default="history")
    parser.add_argument("--embedding-dim", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--include-duplicate-feature8", action="store_true")
    parser.add_argument("--protocol", choices=("air_loco", "air_to_metro", "cross_dataset"), default="air_loco")
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260825,))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ppo-episodes", type=int, default=128)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--value-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--proposal-budget", type=int, default=256)
    parser.add_argument("--timing-repeats", type=int, default=11)
    args = parser.parse_args()
    if args.group_mask < 0 or args.group_mask >= 1 << len(RAW_BLOCKS):
        parser.error(f"group-mask must be in [0,{(1 << len(RAW_BLOCKS)) - 1}]")
    if args.raw_dims is not None:
        if args.raw_dims != sorted(set(args.raw_dims)) or any(dim < 0 or dim >= len(RAW_FEATURE_NAMES) for dim in args.raw_dims):
            parser.error("raw-dims must be unique sorted integers in [0,16]")
        if 8 in args.raw_dims and not args.include_duplicate_feature8:
            parser.error("raw dimension 8 requires --include-duplicate-feature8")
    if args.prefix_mode != "embedding" and args.embedding_dim != 1:
        parser.error("embedding-dim only applies to prefix-mode=embedding")
    return args


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    base_path = script_path.with_name("search_physical_operator_variants_v2.py")
    script_dir = script_path.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    base = load_module(base_path, "physical_policy_ablation_base")
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tables = base.load_tables(args.measurement_root.resolve())
    raw_features, _rows = base.load_feature_matrix(repo_root, tables)
    raw_dims = (
        tuple(args.raw_dims)
        if args.raw_dims is not None
        else selected_raw_dims(args.group_mask, args.include_duplicate_feature8)
    )

    # Verify the selective extractor exactly reproduces every retained baseline coordinate.
    loaded_values: list[np.ndarray] = []
    for index, table in enumerate(tables):
        values, _metadata, _data, _info = base.load_column(repo_root, table.dataset, table.column_index, 0)
        loaded_values.append(values)
        selected = extract_selected(values, raw_dims)
        if not np.allclose(selected, raw_features[index, raw_dims], rtol=0.0, atol=1.0e-12, equal_nan=True):
            raise AssertionError((table.key, raw_dims, selected, raw_features[index, raw_dims]))

    timing_samples: list[float] = []
    for _ in range(args.timing_repeats):
        started = time.perf_counter_ns()
        for values in loaded_values:
            extract_selected(values, raw_dims)
        timing_samples.append((time.perf_counter_ns() - started) / 1.0e6)

    original_builder = base.FeatureBuilder
    original_mlp = base.MLP
    base.FeatureBuilder = lambda features, train: AblationFeatureBuilder(
        base, features, train, raw_dims, args.prefix_mode, args.embedding_dim
    )
    if args.prefix_mode == "embedding":
        base.MLP = embedding_mlp_class(base, len(raw_dims), args.embedding_dim)

    folds = split_indices(tables, args.protocol)
    beam_by_key = {table.key: base.beam29(table, 29) for table in tables}
    evaluation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    policy_dir = output_dir / "policies"
    policy_dir.mkdir()
    train_args = argparse.Namespace(
        hidden=args.hidden,
        actor_lr=0.001,
        critic_lr=0.001,
        iterations=args.iterations,
        ppo_episodes=args.ppo_episodes,
        grpo_groups=32,
        group_size=8,
        gae_lambda=0.95,
        policy_epochs=args.policy_epochs,
        value_epochs=args.value_epochs,
        minibatch=args.minibatch,
        clip_ratio=0.2,
        entropy_coefficient=0.01,
        proposal_budget=args.proposal_budget,
        log_every=max(1, args.iterations),
    )
    for seed in args.seeds:
        for fold, (train_columns, test_columns) in enumerate(folds):
            started = time.perf_counter()
            evaluation, training, actor = base.train_algorithm(
                "ppo", seed, fold, tables, raw_features, train_columns, test_columns, train_args
            )
            elapsed = time.perf_counter() - started
            for row in evaluation:
                key = (row["test_dataset"], int(row["column_index"]))
                beam = beam_by_key[key]
                row.update(
                    {
                        "variant": args.variant,
                        "group_mask": args.group_mask,
                        "raw_dimensions": len(raw_dims),
                        "prefix_mode": args.prefix_mode,
                        "embedding_dim": args.embedding_dim if args.prefix_mode == "embedding" else 0,
                        "beam29_bytes": beam["bytes"],
                        "byte_equivalent_oracle_hit": int(int(row["encoded_bytes"]) == int(row["oracle_bytes"])),
                        "acc_at_1pct": int(int(row["encoded_bytes"]) <= 1.01 * int(row["oracle_bytes"])),
                    }
                )
            for row in training:
                row.update({"variant": args.variant, "protocol": args.protocol})
            evaluation_rows.extend(evaluation)
            training_rows.extend(training)
            run_rows.append(
                {
                    "variant": args.variant,
                    "seed": seed,
                    "fold": fold,
                    "train_columns": len(train_columns),
                    "test_columns": len(test_columns),
                    "elapsed_sec": elapsed,
                }
            )
            np.savez_compressed(policy_dir / f"ppo_seed{seed}_fold{fold}.npz", *actor.params)

    base.FeatureBuilder = original_builder
    base.MLP = original_mlp
    write_csv(output_dir / "evaluation.csv", evaluation_rows)
    write_csv(output_dir / "training_curve.csv", training_rows)
    write_csv(output_dir / "run_status.csv", run_rows)

    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows = [row for row in evaluation_rows if int(row["seed"]) == seed]
        total = sum(int(row["encoded_bytes"]) for row in rows)
        oracle = sum(int(row["oracle_bytes"]) for row in rows)
        beam = sum(int(row["beam29_bytes"]) for row in rows)
        per_seed.append(
            {
                "variant": args.variant,
                "seed": seed,
                "columns": len(rows),
                "total_bytes": total,
                "oracle_bytes": oracle,
                "beam29_bytes": beam,
                "oracle_regret_pct": 100.0 * (total - oracle) / oracle,
                "gain_vs_beam29_pct": 100.0 * (beam - total) / beam,
                "byte_equivalent_oracle_hits": sum(int(row["byte_equivalent_oracle_hit"]) for row in rows),
                "acc_at_1pct_hits": sum(int(row["acc_at_1pct"]) for row in rows),
                "median_unique_proposals": statistics.median(int(row["unique_proposals"]) for row in rows),
            }
        )
    write_csv(output_dir / "per_seed_summary.csv", per_seed)

    probe_builder = AblationFeatureBuilder(base, raw_features, folds[0][0], raw_dims, args.prefix_mode, args.embedding_dim)
    dense_model_dim = (
        len(raw_dims) + 3 * args.embedding_dim
        if args.prefix_mode == "embedding"
        else probe_builder.input_dim
    )
    logical_prefix_fields = {"full": 5, "history": 3, "transform": 2, "packer": 1, "none": 0, "embedding": 3}[args.prefix_mode]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "protocol": args.protocol,
        "raw_group_mask": args.group_mask,
        "enabled_raw_blocks": (
            ["custom_dims"]
            if args.raw_dims is not None
            else [name for bit, (name, _dims) in enumerate(RAW_BLOCKS) if args.group_mask & (1 << bit)]
        ),
        "raw_dimensions": list(raw_dims),
        "raw_feature_names": [RAW_FEATURE_NAMES[index] for index in raw_dims],
        "prefix_mode": args.prefix_mode,
        "embedding_dim": args.embedding_dim if args.prefix_mode == "embedding" else 0,
        "logical_observation_fields": len(raw_dims) + logical_prefix_fields,
        "dense_model_input_dim": dense_model_dim,
        "feature_extraction_ms_20_columns_median": statistics.median(timing_samples),
        "feature_extraction_ms_20_columns_p25": float(np.percentile(timing_samples, 25)),
        "feature_extraction_ms_20_columns_p75": float(np.percentile(timing_samples, 75)),
        "seeds": list(args.seeds),
        "folds": len(folds),
        "median_total_bytes": statistics.median(row["total_bytes"] for row in per_seed),
        "median_oracle_regret_pct": statistics.median(row["oracle_regret_pct"] for row in per_seed),
        "median_byte_equivalent_oracle_hits": statistics.median(row["byte_equivalent_oracle_hits"] for row in per_seed),
        "median_acc_at_1pct_hits": statistics.median(row["acc_at_1pct_hits"] for row in per_seed),
        "median_unique_proposals": statistics.median(row["median_unique_proposals"] for row in per_seed),
        "training_wall_sec": time.perf_counter() - run_started,
        "claim_boundary": "bounded PV2F normalized-int64 development/lockbox ablation; policy observations only; grammar and byte tables unchanged",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": summary["created_utc"],
        "script_sha256": sha256_file(script_path),
        "base_script_sha256": sha256_file(base_path),
        "measurement_manifests": {
            str(path): sha256_file(path)
            for path in sorted(args.measurement_root.resolve().glob("*/manifest.json"))
        },
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "evaluation_sha256": sha256_file(output_dir / "evaluation.csv"),
        "counts": {"tables": len(tables), "folds": len(folds), "evaluation_rows": len(evaluation_rows), "training_rows": len(training_rows)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
