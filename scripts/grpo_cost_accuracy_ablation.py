#!/usr/bin/env python3
"""External-dataset GRPO LOFO runs with genuinely reduced actor inputs."""

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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SEEDS = (20260825, 20260826, 20260827)


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


def feature_catalog(base, env) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(base.RAW_FEATURE_NAMES):
        rows.append({"dimension": index, "family": "raw", "feature_name": name, "independently_testable": 1})
    for index, name in enumerate(env.operators):
        rows.append({"dimension": 18 + index, "family": "probe", "feature_name": f"probe:{name}", "independently_testable": 1})
    rows.append({"dimension": 46, "family": "rate", "feature_name": "current_prefix_log_rate_ratio", "independently_testable": 1})
    for depth in range(3):
        rows.append({"dimension": 47 + depth, "family": "depth", "feature_name": f"depth_one_hot:{depth}", "independently_testable": 0})
    for slot in range(2):
        for index, name in enumerate(env.operators):
            rows.append({"dimension": 50 + slot * 28 + index, "family": f"prefix_slot_{slot + 1}", "feature_name": f"prefix_slot_{slot + 1}:{name}", "independently_testable": 0})
    if len(rows) != 106:
        raise AssertionError(len(rows))
    return rows


def paired_actor(base, hidden: int, seed: int, disabled: tuple[int, ...]):
    full = base.MLP(106, hidden, 29, np.random.default_rng(seed))
    if not disabled:
        return full
    actor = base.MLP(106 - len(disabled), hidden, 29, np.random.default_rng(0))
    actor.params = [
        np.delete(full.params[0], disabled, axis=0),
        full.params[1].copy(),
        full.params[2].copy(),
        full.params[3].copy(),
    ]
    return actor


def reduced_features(base, env, development: np.ndarray, disabled: tuple[int, ...]) -> np.ndarray:
    full = base.build_state_features(env, development)
    if full.shape[1] != 106:
        raise AssertionError(full.shape)
    return np.delete(full, disabled, axis=1) if disabled else full


def train_one(base, ablation, env, development, test, disabled, name, seed, args):
    features = reduced_features(base, env, development, disabled)
    actor = paired_actor(base, args.hidden, seed, disabled)
    optimizer = base.Adam(actor.params, args.actor_lr)
    rng = np.random.default_rng(seed + 100_003)
    variant = ablation.Variant(name, "cost_accuracy")
    training_rows: list[dict[str, object]] = []
    scale_reference = 1.0e-3
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        rollout, sampling = ablation.collect_group_relative(
            base, env, features, actor, rng, development,
            args.groups_per_iteration, args.group_size, variant, scale_reference,
        )
        update = ablation.actor_update(
            base, env, features, actor, optimizer, rollout, rng,
            args.policy_epochs, args.minibatch, args.entropy_coef, True,
        )
        if sampling["batch_nonzero_dispersion_median"] > 0:
            scale_reference = 0.9 * scale_reference + 0.1 * sampling["batch_nonzero_dispersion_median"]
        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            training_rows.append({
                "variant": name,
                "seed": seed,
                "iteration": iteration,
                "input_dimensions": features.shape[1],
                "elapsed_sec": time.perf_counter() - started,
                **update,
                **sampling,
            })
    evaluation = ablation.evaluate(base, env, features, actor, test, variant, seed, 0)
    for row in evaluation:
        row["disabled_dimensions"] = "+".join(map(str, disabled)) or "none"
        row["input_dimensions"] = features.shape[1]
        row["exact_oracle"] = int(abs(float(row["proxy_rate"]) - float(row["oracle_rate"])) <= 1.0e-15)
        row["within_1pct_oracle"] = int(float(row["oracle_regret_pct"]) <= 1.0 + 1.0e-12)
    return evaluation, training_rows, actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--development-datasets", nargs="+", required=True)
    parser.add_argument("--test-datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dims", nargs="+", type=int, default=[])
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--joint-disabled-dims", nargs="+", type=int, default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--groups-per-iteration", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=0.003)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(dim < 0 or dim > 46 for dim in args.dims):
        raise ValueError("independent --dims must be in [0,46]")
    if args.dims != sorted(set(args.dims)):
        raise ValueError("--dims must be sorted and unique")
    joint = tuple(sorted(set(args.joint_disabled_dims)))
    if any(dim < 0 or dim > 105 for dim in joint):
        raise ValueError("joint dimensions must be in [0,105]")
    script_dir = Path(__file__).resolve().parent
    base = load_module(script_dir / "cross_column_rl_compare.py", "cost_accuracy_base")
    ablation = load_module(script_dir / "grpo_task_ablation.py", "cost_accuracy_objective")
    datasets = tuple(args.development_datasets + args.test_datasets)
    env, data_inputs, operator_source = base.load_environment(
        args.repo_root.resolve(), args.measurement_root.resolve(),
        args.summary_csv.resolve(), args.baseline_csv.resolve(), datasets,
    )
    development = np.asarray([i for i, column in enumerate(env.columns) if column.dataset in set(args.development_datasets)], dtype=np.int32)
    test = np.asarray([i for i, column in enumerate(env.columns) if column.dataset in set(args.test_datasets)], dtype=np.int32)
    if development.size != 20 or test.size != 18 or set(development) & set(test):
        raise AssertionError((development.size, test.size))
    catalog = feature_catalog(base, env)

    full_features = base.build_state_features(env, development)
    witness_rng = np.random.default_rng(12345)
    for disabled in ((0,), (18,), (46,), (0, 18, 46)):
        full_actor = base.MLP(106, 8, np.int64(29), np.random.default_rng(321))
        reduced_actor = paired_actor(base, 8, 321, disabled)
        zeroed = full_features.copy()
        zeroed[:, disabled] = 0.0
        picked = witness_rng.choice(zeroed.shape[0], size=min(1000, zeroed.shape[0]), replace=False)
        full_logits, _ = full_actor.forward(zeroed[picked])
        reduced_logits, _ = reduced_actor.forward(np.delete(full_features[picked], disabled, axis=1))
        if float(np.max(np.abs(full_logits - reduced_logits))) > 1.0e-12:
            raise AssertionError(f"paired initialization mismatch {disabled}")
    print("COST_ACCURACY_SELF_CHECK paired_reduced_initialization=PASS", flush=True)
    if args.self_check_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")

    variants: list[tuple[str, tuple[int, ...]]] = []
    if args.include_full:
        variants.append(("full_106", ()))
    variants.extend((f"lofo_{dim:03d}", (dim,)) for dim in args.dims)
    if joint:
        variants.append(("joint_reduced", joint))
    if not variants:
        raise ValueError("no variants requested")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "feature_catalog.csv", catalog)
    evaluation_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for name, disabled in variants:
        for seed in args.seeds:
            evaluation, training, actor = train_one(
                base, ablation, env, development, test, disabled, name, seed, args,
            )
            evaluation_rows.extend(evaluation)
            training_rows.extend(training)
            status_rows.append({
                "variant": name,
                "seed": seed,
                "disabled_dimensions": "+".join(map(str, disabled)) or "none",
                "input_dimensions": 106 - len(disabled),
                "completed_iterations": args.iterations,
            })
            write_csv(output_dir / "evaluation.csv", evaluation_rows)
            write_csv(output_dir / "training_curve.csv", training_rows)
            write_csv(output_dir / "run_status.csv", status_rows)
            np.savez_compressed(
                output_dir / f"policy_{name}_seed{seed}.npz",
                disabled_dimensions=np.asarray(disabled, dtype=np.int16),
                W1=actor.params[0], b1=actor.params[1], W2=actor.params[2], b2=actor.params[3],
            )

    expected = len(variants) * len(args.seeds) * len(test)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_script_sha256": hashlib.sha256((script_dir / "cross_column_rl_compare.py").read_bytes()).hexdigest(),
        "objective_script_sha256": hashlib.sha256((script_dir / "grpo_task_ablation.py").read_bytes()).hexdigest(),
        "operator_source": str(operator_source),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "variants": [{"name": name, "disabled_dimensions": disabled} for name, disabled in variants],
        "protocol": {
            "split": "Metro/Air development only; Bird/meteoblue frozen external test",
            "reduction": "delete coordinate from state and paired 106D actor first layer",
            "accuracy": "tie-aware exact exhaustive-oracle and <=1% oracle regret",
            "test_selection": "forbidden before frozen independent screen",
        },
        "inputs": data_inputs,
        "counts": {
            "development_columns": int(development.size),
            "external_test_columns": int(test.size),
            "variants": len(variants),
            "seeds": len(args.seeds),
            "evaluation_rows": len(evaluation_rows),
            "expected_evaluation_rows": expected,
            "training_runs": len(status_rows),
        },
        "claim_boundary": {"probe_conditioned": True, "production_claim_allowed": False},
        "wall_sec": time.perf_counter() - started,
    }
    if len(evaluation_rows) != expected:
        raise AssertionError(manifest["counts"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_dir), "counts": manifest["counts"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
