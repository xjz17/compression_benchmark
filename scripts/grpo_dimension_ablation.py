#!/usr/bin/env python3
"""Retrain GRPO after replacing each selected 106D coordinate with zero.

This is a leave-one-feature-out sensitivity experiment.  A disabled coordinate
is set to zero after training-fold normalization in training and evaluation.
The actor shape and random initialization remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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
        rows.append({"dimension": index, "family": "raw", "feature_name": name, "inference_class": "continuous"})
    for operator_index, operator in enumerate(env.operators):
        rows.append({"dimension": 18 + operator_index, "family": "probe", "feature_name": f"probe:{operator}", "inference_class": "continuous"})
    rows.append({"dimension": 46, "family": "rate", "feature_name": "current_prefix_log_rate_ratio", "inference_class": "continuous"})
    for depth in range(3):
        rows.append({"dimension": 47 + depth, "family": "depth", "feature_name": f"depth_one_hot:{depth}", "inference_class": "one_hot_non_identifiable"})
    for slot in range(2):
        for operator_index, operator in enumerate(env.operators):
            rows.append({
                "dimension": 50 + slot * 28 + operator_index,
                "family": f"prefix_slot_{slot + 1}",
                "feature_name": f"prefix_slot_{slot + 1}:{operator}",
                "inference_class": "one_hot_non_identifiable",
            })
    if len(rows) != 106 or [int(row["dimension"]) for row in rows] != list(range(106)):
        raise AssertionError("invalid 106D catalog")
    return rows


def support_row(env, features: np.ndarray, train_columns: np.ndarray, validation_columns: np.ndarray, item, fold: int):
    train_states = np.isin(env.state_col, train_columns)
    validation_states = np.isin(env.state_col, validation_columns)
    dim = int(item["dimension"])
    train_values = features[train_states, dim]
    validation_values = features[validation_states, dim]
    return {
        "dataset": env.columns[int(validation_columns[0])].dataset,
        "fold": fold,
        **item,
        "train_state_count": int(train_values.size),
        "validation_state_count": int(validation_values.size),
        "train_std": float(np.std(train_values)),
        "train_nonzero_fraction": float(np.mean(train_values != 0.0)),
        "validation_std": float(np.std(validation_values)),
        "validation_nonzero_fraction": float(np.mean(validation_values != 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dims", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260825, 20260826, 20260827))
    parser.add_argument("--folds", type=int, required=True)
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
    parser.set_defaults(episodes_per_iteration=512, gae_lambda=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    ablation_path = script_dir / "grpo_task_ablation.py"
    ablation = load_module(ablation_path, "dimension_ablation_impl")
    base, base_path = ablation.load_base(script_dir)
    env, data_inputs, operator_source = base.load_environment(
        args.repo_root.resolve(), args.measurement_root.resolve(),
        args.summary_csv.resolve(), args.baseline_csv.resolve(), tuple(args.datasets),
    )
    if args.folds != len(env.columns):
        raise ValueError("dimension ablation requires leave-one-column-out within one dataset")
    dims = sorted(set(args.dims))
    if dims != args.dims or any(dim < 0 or dim >= 106 for dim in dims):
        raise ValueError("--dims must be unique sorted integers in [0,105]")
    folds = base.assign_folds(env.columns, args.folds)
    ablation.self_check(base, env, folds, args)
    catalog = feature_catalog(base, env)
    selected = [dict(catalog[dim]) for dim in dims]
    for item in selected:
        key = f"coordinate_{int(item['dimension']):03d}"
        ablation.FEATURE_SLICES[key] = slice(int(item["dimension"]), int(item["dimension"]) + 1)
        item["mask_key"] = key
        item["algorithm"] = f"dim_{int(item['dimension']):03d}"
    train = np.flatnonzero(folds != 0)
    witness = base.build_state_features(env, train)
    for item in selected:
        variant = ablation.Variant(item["algorithm"], "dimension", (item["mask_key"],))
        masked = ablation.mask_features(witness, variant)
        dim = int(item["dimension"])
        if np.any(masked[:, dim]) or not np.array_equal(masked[:, :dim], witness[:, :dim]) or not np.array_equal(masked[:, dim + 1 :], witness[:, dim + 1 :]):
            raise AssertionError(f"mask witness failed for dimension {dim}")
    print(f"DIMENSION_ABLATION_SELF_CHECK dimensions={len(selected)} status=PASS", flush=True)
    if args.self_check_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --self-check-only")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "feature_catalog.csv", catalog)
    write_csv(output_dir / "fold_assignments.csv", [{
        "dataset": column.dataset,
        "column_index": column.index,
        "column_name": column.name,
        "logical_type": column.logical_type,
        "fold": int(folds[position]),
    } for position, column in enumerate(env.columns)])

    evaluation_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for item in selected:
        variant = ablation.Variant(
            item["algorithm"], "dimension", (item["mask_key"],),
            what_it_tests=f"conditional contribution of coordinate {item['feature_name']}",
            expected_if_component_matters="actual codec rate worsens after reference-zero substitution and full retraining",
        )
        for seed in args.seeds:
            for fold in range(args.folds):
                train_columns = np.flatnonzero(folds != fold)
                validation_columns = np.flatnonzero(folds == fold)
                full_features = base.build_state_features(env, train_columns)
                if seed == args.seeds[0]:
                    support_rows.append(support_row(env, full_features, train_columns, validation_columns, item, fold))
                evaluation, training, actor, status = ablation.train_one(
                    base, env, full_features, train_columns, validation_columns,
                    variant, seed, fold, args,
                )
                for row in evaluation:
                    row.update({"disabled_dimension": item["dimension"], "disabled_feature_name": item["feature_name"], "feature_family": item["family"], "inference_class": item["inference_class"]})
                for row in training:
                    row.update({"disabled_dimension": item["dimension"], "disabled_feature_name": item["feature_name"], "feature_family": item["family"]})
                status.update({"disabled_dimension": item["dimension"], "disabled_feature_name": item["feature_name"], "feature_family": item["family"]})
                evaluation_rows.extend(evaluation)
                training_rows.extend(training)
                status_rows.append(status)
                write_csv(output_dir / "evaluation.csv", evaluation_rows)
                write_csv(output_dir / "training_curve.csv", training_rows)
                write_csv(output_dir / "run_status.csv", status_rows)
                write_csv(output_dir / "feature_support.csv", support_rows)

    script_path = Path(__file__).resolve()
    expected = len(selected) * len(args.seeds) * len(env.columns)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "script": str(script_path),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "ablation_script": str(ablation_path),
        "ablation_script_sha256": hashlib.sha256(ablation_path.read_bytes()).hexdigest(),
        "base_script": str(base_path),
        "base_script_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "operator_source": str(operator_source),
        "operator_source_sha256": hashlib.sha256(operator_source.read_bytes()).hexdigest(),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "features": selected,
        "protocol": {
            "name": "LOFO-retrain reference-zero ablation",
            "mask": "set one post-normalization coordinate to zero in training and evaluation",
            "split": f"{args.folds}-fold leave-one-column-out within one dataset; scaler uses training columns only",
            "paired_controls": "same 106D actor shape, initialization seed, optimizer, rollout budget, and random-number setup",
            "metric_gate": "selected pipelines require standalone codec replay before compression claims",
            "continuous_inference_family": "47 dimensions: raw18 + probe28 + current-rate1",
            "one_hot_boundary": "depth3 and prefix56 are coordinate sensitivity only, not independently identifiable necessity tests",
        },
        "claim_boundary": {
            "probe_conditioned": True,
            "production_claim_allowed": False,
            "all_106_necessary_claim_allowed": False,
            "reason": "correlated inputs, one-hot reparameterization, two exploratory datasets, and only 20 columns",
        },
        "inputs": {
            "measurement_manifests": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.measurement_root.resolve().glob("*/manifest.json"))},
            "summary_csv": hashlib.sha256(args.summary_csv.resolve().read_bytes()).hexdigest(),
            "baseline_csv": hashlib.sha256(args.baseline_csv.resolve().read_bytes()).hexdigest(),
            "datasets": data_inputs,
        },
        "counts": {
            "columns": len(env.columns), "dimensions": len(selected), "seeds": len(args.seeds),
            "evaluation_rows": len(evaluation_rows), "expected_evaluation_rows": expected,
            "training_runs": len(status_rows), "support_rows": len(support_rows),
        },
        "wall_sec": time.perf_counter() - started,
    }
    if len(evaluation_rows) != expected or len(status_rows) != expected:
        raise AssertionError(manifest["counts"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_dir), "counts": manifest["counts"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
