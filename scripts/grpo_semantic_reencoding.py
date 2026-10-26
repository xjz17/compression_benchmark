#!/usr/bin/env python3
"""Fresh-dataset GRPO ablation for identifiable depth/prefix semantics.

The actor always remains 106->hidden->29. C0 is the original encoding; E1-E3
are information-equivalent encodings with function-matched initialization;
D1-D4 remove explicit depth, prefix order, or prefix identity. Policies train
only on the declared development datasets and are evaluated without updates on
the declared external-test datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEPTH = slice(47, 50)
PREFIX = slice(50, 106)
REFERENCE_OPERATOR = 0


@dataclass(frozen=True)
class Encoding:
    name: str
    family: str
    depth: str
    prefix: str
    what_it_tests: str
    expected_if_matters: str


ENCODINGS = (
    Encoding("C0", "reference", "one_hot", "positional", "original 106D state", "paired reference"),
    Encoding("E1", "equivalent", "thermometer", "positional", "depth coding sensitivity", "equivalent to C0"),
    Encoding("E2", "equivalent", "one_hot", "treatment", "prefix PAD/reference coding sensitivity", "equivalent to C0"),
    Encoding("E3", "equivalent", "thermometer", "treatment", "joint coding sensitivity", "equivalent to C0"),
    Encoding("D1", "deletion", "zero", "treatment", "explicit depth beyond prefix occupancy", "near E3 if depth cue is redundant"),
    Encoding("D2", "deletion", "thermometer", "count", "prefix order beyond operator multiset", "worse if order matters"),
    Encoding("D3", "deletion", "thermometer", "zero", "prefix identity and order", "worse if operator history matters"),
    Encoding("D4", "deletion", "zero", "zero", "all explicit depth/prefix cues", "largest deletion cost if history matters"),
)
ENCODING_BY_NAME = {item.name: item for item in ENCODINGS}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def encoded_features(base, env, train_columns: np.ndarray, encoding: Encoding) -> np.ndarray:
    features = base.build_state_features(env, train_columns)
    if features.shape[1] != 106:
        raise AssertionError(features.shape)

    depth = np.zeros((env.state_col.size, 3), dtype=np.float64)
    if encoding.depth == "one_hot":
        depth[np.arange(env.state_col.size), env.state_depth] = 1.0
    elif encoding.depth == "thermometer":
        depth[:, 0] = env.state_depth >= 1
        depth[:, 1] = env.state_depth >= 2
    elif encoding.depth != "zero":
        raise ValueError(encoding.depth)

    prefix = np.zeros((env.state_col.size, 56), dtype=np.float64)
    if encoding.prefix == "positional":
        for sid, actions in enumerate(env.state_prefix):
            for slot, action in enumerate(actions):
                prefix[sid, slot * 28 + action] = 1.0
    elif encoding.prefix == "treatment":
        for sid, actions in enumerate(env.state_prefix):
            for slot in range(2):
                start = slot * 28
                if slot >= len(actions):
                    prefix[sid, start] = 1.0  # PAD is explicit.
                elif actions[slot] != REFERENCE_OPERATOR:
                    prefix[sid, start + actions[slot]] = 1.0
    elif encoding.prefix == "count":
        for sid, actions in enumerate(env.state_prefix):
            for action in actions:
                prefix[sid, action] += 1.0
    elif encoding.prefix != "zero":
        raise ValueError(encoding.prefix)

    features[:, DEPTH] = depth
    features[:, PREFIX] = prefix
    if not np.all(np.isfinite(features)):
        raise AssertionError("non-finite encoded features")
    return features


def initialize_actor(base, hidden: int, seed: int, encoding: Encoding):
    actor = base.MLP(106, hidden, 29, np.random.default_rng(seed))
    weights, bias = actor.params[0], actor.params[1]

    if encoding.depth in {"thermometer", "zero"}:
        old = weights[DEPTH].copy()
        bias += old[0]
        weights[47] = old[1] - old[0]
        weights[48] = old[2] - old[1]
        weights[49] = 0.0

    if encoding.prefix == "treatment":
        for slot in range(2):
            start = 50 + 28 * slot
            old = weights[start : start + 28].copy()
            reference = old[REFERENCE_OPERATOR]
            bias += reference
            weights[start] = -reference  # explicit PAD
            for action in range(1, 28):
                weights[start + action] = old[action] - reference
    return actor


def function_match_error(base, env, train_columns: np.ndarray, hidden: int, seed: int, encoding: Encoding) -> float:
    reference_features = encoded_features(base, env, train_columns, ENCODING_BY_NAME["C0"])
    variant_features = encoded_features(base, env, train_columns, encoding)
    reference_actor = initialize_actor(base, hidden, seed, ENCODING_BY_NAME["C0"])
    variant_actor = initialize_actor(base, hidden, seed, encoding)
    reference_logits, _ = reference_actor.forward(reference_features)
    variant_logits, _ = variant_actor.forward(variant_features)
    return float(np.max(np.abs(reference_logits - variant_logits)))


def history_code_cardinality(base, env, train_columns: np.ndarray, encoding: Encoding) -> int:
    column = int(train_columns[0])
    selected = np.flatnonzero(env.state_col == column)
    history = encoded_features(base, env, train_columns, encoding)[selected, 47:106]
    return len({tuple(row) for row in history})


def evaluate(ablation, base, env, features, actor, columns, encoding, seed):
    variant = ablation.Variant(encoding.name, encoding.family)
    rows = ablation.evaluate(base, env, features, actor, columns, variant, seed, 0)
    for row in rows:
        row["encoding_family"] = encoding.family
        row["depth_encoding"] = encoding.depth
        row["prefix_encoding"] = encoding.prefix
    return rows


def train_one(ablation, base, env, train_columns, test_columns, encoding, seed, args):
    features = encoded_features(base, env, train_columns, encoding)
    actor = initialize_actor(base, args.hidden, seed, encoding)
    optimizer = base.Adam(actor.params, args.actor_lr)
    rng = np.random.default_rng(seed + 100_003)
    variant = ablation.Variant(encoding.name, encoding.family)
    training_rows: list[dict[str, object]] = []
    scale_reference = 1.0e-3
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        rollout, sampling = ablation.collect_group_relative(
            base, env, features, actor, rng, train_columns,
            args.groups_per_iteration, args.group_size, variant, scale_reference,
        )
        update = ablation.actor_update(
            base, env, features, actor, optimizer, rollout, rng,
            args.policy_epochs, args.minibatch, args.entropy_coef, True,
        )
        if sampling["batch_nonzero_dispersion_median"] > 0:
            scale_reference = 0.9 * scale_reference + 0.1 * sampling["batch_nonzero_dispersion_median"]
        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            row = {
                "variant": encoding.name,
                "encoding_family": encoding.family,
                "seed": seed,
                "iteration": iteration,
                "transitions": int(rollout["states"].size),
                "elapsed_sec": time.perf_counter() - started,
                **update,
                **sampling,
            }
            training_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return evaluate(ablation, base, env, features, actor, test_columns, encoding, seed), training_rows, actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--development-datasets", nargs="+", required=True)
    parser.add_argument("--test-datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--encodings", nargs="+", choices=tuple(ENCODING_BY_NAME), default=tuple(ENCODING_BY_NAME))
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260825, 20260826, 20260827))
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--groups-per-iteration", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=0.003)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base = load_module(script_dir / "cross_column_rl_compare.py", "semantic_base")
    ablation = load_module(script_dir / "grpo_task_ablation.py", "semantic_ablation")
    datasets = tuple(args.development_datasets + args.test_datasets)
    env, data_inputs, operator_source = base.load_environment(
        args.repo_root.resolve(), args.measurement_root.resolve(),
        args.summary_csv.resolve(), args.baseline_csv.resolve(), datasets,
    )
    development = np.asarray([
        i for i, column in enumerate(env.columns) if column.dataset in set(args.development_datasets)
    ], dtype=np.int32)
    test = np.asarray([
        i for i, column in enumerate(env.columns) if column.dataset in set(args.test_datasets)
    ], dtype=np.int32)
    if not development.size or not test.size or set(development) & set(test):
        raise AssertionError("invalid dataset-held-out split")

    cardinalities = {
        name: history_code_cardinality(base, env, development, ENCODING_BY_NAME[name])
        for name in ENCODING_BY_NAME
    }
    expected_cardinalities = {
        "C0": 813, "E1": 813, "E2": 813, "E3": 813,
        "D1": 813, "D2": 435, "D3": 3, "D4": 1,
    }
    if cardinalities != expected_cardinalities:
        raise AssertionError(f"history coding cardinality mismatch: {cardinalities}")

    errors = {
        name: max(
            function_match_error(base, env, development, args.hidden, seed, ENCODING_BY_NAME[name])
            for seed in args.seeds
        )
        for name in ("E1", "E2", "E3")
    }
    if max(errors.values()) > 1.0e-12:
        raise AssertionError(f"function matching failed: {errors}")
    print(
        "SEMANTIC_SELF_CHECK "
        + json.dumps({"function_match_error": errors, "history_code_cardinality": cardinalities}, sort_keys=True),
        flush=True,
    )
    if args.self_check_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --self-check-only")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluation_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for name in args.encodings:
        encoding = ENCODING_BY_NAME[name]
        for seed in args.seeds:
            evaluation, training, actor = train_one(
                ablation, base, env, development, test, encoding, seed, args,
            )
            evaluation_rows.extend(evaluation)
            training_rows.extend(training)
            status_rows.append({
                "variant": name,
                "seed": seed,
                "requested_iterations": args.iterations,
                "completed_iterations": args.iterations,
                "function_match_error": errors.get(name, ""),
            })
            write_csv(output_dir / "evaluation.csv", evaluation_rows)
            write_csv(output_dir / "training_curve.csv", training_rows)
            write_csv(output_dir / "run_status.csv", status_rows)
            np.savez_compressed(
                output_dir / f"policy_{name}_seed{seed}.npz",
                W1=actor.params[0], b1=actor.params[1], W2=actor.params[2], b2=actor.params[3],
            )

    expected = len(args.encodings) * len(args.seeds) * len(test)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_script_sha256": hashlib.sha256((script_dir / "cross_column_rl_compare.py").read_bytes()).hexdigest(),
        "ablation_script_sha256": hashlib.sha256((script_dir / "grpo_task_ablation.py").read_bytes()).hexdigest(),
        "operator_source": str(operator_source),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "encodings": [asdict(ENCODING_BY_NAME[name]) for name in args.encodings],
        "protocol": {
            "split": "dataset-held-out: development datasets train; external-test datasets never update policy/scaler",
            "actor": f"106->{args.hidden}->29 for every variant",
            "initialization": "E1-E3 first layer function-matched to C0 over every legal state; D-family paired base seed",
            "reference_operator": REFERENCE_OPERATOR,
            "reference_operator_name": env.operators[REFERENCE_OPERATOR],
            "reward": "development-only proxy rate; external-test evaluation is frozen greedy selection",
            "codec_followup": "selected pipelines require disjoint evaluation-suffix byte-only lossless replay",
        },
        "function_match_max_abs_logit_error": errors,
        "history_code_cardinality": cardinalities,
        "inputs": data_inputs,
        "counts": {
            "development_columns": int(development.size),
            "external_test_columns": int(test.size),
            "variants": len(args.encodings),
            "seeds": len(args.seeds),
            "evaluation_rows": len(evaluation_rows),
            "expected_evaluation_rows": expected,
            "training_runs": len(status_rows),
        },
        "claim_boundary": {
            "probe_conditioned": True,
            "production_claim_allowed": False,
            "test_method_selection_allowed": False,
            "one_hot_bit_necessity_claim_allowed": False,
        },
        "wall_sec": time.perf_counter() - started,
    }
    if len(evaluation_rows) != expected or len(status_rows) != len(args.encodings) * len(args.seeds):
        raise AssertionError(manifest["counts"])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT {output_dir}", flush=True)


if __name__ == "__main__":
    main()
