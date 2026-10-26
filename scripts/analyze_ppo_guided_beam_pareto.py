#!/usr/bin/env python3
"""Recompute PPO-guided Beam results and add a standalone Beam-2 ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def import_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline_text(operators: tuple[str, ...], pipeline: tuple[int, ...]) -> str:
    return " -> ".join(operators[action] for action in pipeline)


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    pareto = import_path(repo_root / "scripts" / "ppo_guided_beam_pareto.py", "audit_pareto")
    cc = import_path(repo_root / "scripts" / "cross_column_rl_compare.py", "audit_crosscol")
    codec = import_path(repo_root / "scripts" / "measure_selected_codec_cost.py", "audit_codec")
    env, _data_inputs, _operator_source = cc.load_environment(
        args.data_repo.resolve(),
        args.measurement_root.resolve(),
        args.summary_csv.resolve(),
        args.baseline_csv.resolve(),
    )
    test_rows = read_csv(experiment_dir / "per_seed_test.csv")
    column_rows = read_csv(experiment_dir / "per_column_summary.csv")
    validation_rows = read_csv(experiment_dir / "validation_dse.csv")
    selected_rows = read_csv(experiment_dir / "selected_config.csv")
    training_rows = read_csv(experiment_dir / "training_summary.csv")
    stored_summary = json.loads((experiment_dir / "SUMMARY.json").read_text())

    keys = [(row["dataset"], int(row["column_index"])) for row in test_rows]
    if len(test_rows) != 60 or len(set(keys)) != 20:
        raise AssertionError("expected 60 test rows over 20 columns")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        grouped[(row["dataset"], int(row["column_index"]))].append(row)
    if any(len(rows) != 3 for rows in grouped.values()):
        raise AssertionError("each column must have three seeds")
    if {int(row["seed"]) for row in test_rows} != {7, 19, 43}:
        raise AssertionError("seed coverage differs")
    if any(int(row["lossless"]) != 1 for row in test_rows):
        raise AssertionError("non-lossless test row")
    if any(int(row["hybrid_evaluations"]) >= int(row["beam4_evaluations"]) for row in test_rows):
        raise AssertionError("hybrid did not use fewer codec evaluations")

    folds = cc.assign_folds(env.columns, 5)
    position_by_key = {column.key: position for position, column in enumerate(env.columns)}
    for row in test_rows:
        position = position_by_key[(row["dataset"], int(row["column_index"]))]
        if int(row["outer_fold"]) != int(folds[position]):
            raise AssertionError("outer-test fold label mismatch")
    for row in training_rows:
        outer_fold = int(row["outer_fold"])
        validation_fold = (outer_fold + 1) % 5
        if row["stage"] == "inner_tuning":
            expected = int(((folds != outer_fold) & (folds != validation_fold)).sum())
        else:
            expected = int((folds != outer_fold).sum())
        if int(row["train_columns"]) != expected:
            raise AssertionError("training column count mismatch")

    config_by_name = {config.name: config for config in pareto.CONFIGS}
    for selected in selected_rows:
        outer_fold = int(selected["outer_fold"])
        rows = [row for row in validation_rows if int(row["outer_fold"]) == outer_fold]
        chosen = pareto.choose_config(rows, pareto.CONFIGS)
        if chosen.name != selected["name"]:
            raise AssertionError(f"DSE selection mismatch for fold {outer_fold}")
        if selected["name"] not in config_by_name:
            raise AssertionError("unknown selected configuration")

    recomputed_hybrid_bytes = sum(int(row["median_hybrid_bytes"]) for row in column_rows)
    recomputed_beam4_bytes = sum(int(row["beam4_bytes"]) for row in column_rows)
    recomputed_gain = 100.0 * (recomputed_beam4_bytes - recomputed_hybrid_bytes) / recomputed_beam4_bytes
    hybrid_times = [float(row["median_hybrid_search_ms"]) for row in column_rows]
    beam4_times = [float(row["beam4_search_ms"]) for row in column_rows]
    recomputed_hybrid_time = statistics.median(hybrid_times)
    recomputed_beam4_time = statistics.median(beam4_times)
    exact_checks = {
        "hybrid_total_bytes": recomputed_hybrid_bytes == int(stored_summary["hybrid_total_bytes"]),
        "beam4_total_bytes": recomputed_beam4_bytes == int(stored_summary["beam4_total_bytes"]),
        "aggregate_gain": abs(recomputed_gain - float(stored_summary["aggregate_gain_vs_beam4_pct"])) < 1e-12,
        "median_hybrid_time": abs(recomputed_hybrid_time - float(stored_summary["median_hybrid_search_ms"])) < 1e-12,
        "median_beam4_time": abs(recomputed_beam4_time - float(stored_summary["median_beam4_search_ms"])) < 1e-12,
    }
    if not all(exact_checks.values()):
        raise AssertionError(exact_checks)

    module = codec.load_operator_module(args.data_repo.resolve())
    inputs = [
        {"dataset": column.dataset, "column_index": str(column.index), "column_name": column.name}
        for column in env.columns
    ]
    values_by_key = codec.load_values(args.data_repo.resolve(), inputs, module)
    ablation_rows = []
    fixed_names = ("zigzag", "zigzag", "delta")
    for position, column in enumerate(env.columns):
        values = values_by_key[column.key]
        pipeline, encoded, evaluations = pareto.actual_beam_search(
            values, codec, module, env.operators, 2
        )
        restored, decoded_pipeline = codec.decode_stream(encoded)
        expected_pipeline = tuple(env.operators[action] for action in pipeline)
        if decoded_pipeline != expected_pipeline or not (restored == values).all():
            raise AssertionError(f"Beam-2 round trip failed for {column.key}")
        fixed_encoded = codec.encode_stream(values, fixed_names, module)
        fixed_restored, fixed_decoded_pipeline = codec.decode_stream(fixed_encoded)
        if fixed_decoded_pipeline != fixed_names or not (fixed_restored == values).all():
            raise AssertionError(f"fixed-pipeline round trip failed for {column.key}")
        summary_row = next(
            row
            for row in column_rows
            if row["dataset"] == column.dataset and int(row["column_index"]) == column.index
        )
        hybrid_bytes = int(summary_row["median_hybrid_bytes"])
        beam4_bytes = int(summary_row["beam4_bytes"])
        ablation_rows.append(
            {
                "dataset": column.dataset,
                "column_index": column.index,
                "column_name": column.name,
                "beam2_pipeline": pipeline_text(env.operators, pipeline),
                "beam2_bytes": len(encoded),
                "beam2_evaluations": evaluations,
                "fixed_pipeline": " -> ".join(fixed_names),
                "fixed_pipeline_bytes": len(fixed_encoded),
                "beam2_plus_fixed_bytes": min(len(encoded), len(fixed_encoded)),
                "median_hybrid_bytes": hybrid_bytes,
                "beam4_bytes": beam4_bytes,
                "hybrid_gain_vs_beam2_pct": 100.0 * (len(encoded) - hybrid_bytes) / len(encoded),
                "beam2_gain_vs_beam4_pct": 100.0 * (beam4_bytes - len(encoded)) / beam4_bytes,
                "ppo_branch_strictly_improves_beam2": int(hybrid_bytes < len(encoded)),
                "beam2_plus_fixed_matches_hybrid": int(
                    min(len(encoded), len(fixed_encoded)) == hybrid_bytes
                ),
            }
        )
    write_csv(experiment_dir / "beam2_ablation.csv", ablation_rows)

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "test_rows": len(test_rows),
        "columns": len(grouped),
        "seeds": sorted({int(row["seed"]) for row in test_rows}),
        "lossless_rows": sum(int(row["lossless"]) for row in test_rows),
        "exact_summary_checks": exact_checks,
        "recomputed": {
            "hybrid_total_bytes": recomputed_hybrid_bytes,
            "beam4_total_bytes": recomputed_beam4_bytes,
            "aggregate_gain_vs_beam4_pct": recomputed_gain,
            "median_hybrid_search_ms": recomputed_hybrid_time,
            "median_beam4_search_ms": recomputed_beam4_time,
            "time_faster_columns": sum(hybrid < beam for hybrid, beam in zip(hybrid_times, beam4_times)),
        },
        "beam2_ablation": {
            "columns": len(ablation_rows),
            "beam2_evaluations_each": sorted({int(row["beam2_evaluations"]) for row in ablation_rows}),
            "ppo_branch_strict_improvement_columns": sum(
                int(row["ppo_branch_strictly_improves_beam2"]) for row in ablation_rows
            ),
            "hybrid_total_bytes": sum(int(row["median_hybrid_bytes"]) for row in ablation_rows),
            "beam2_total_bytes": sum(int(row["beam2_bytes"]) for row in ablation_rows),
            "beam2_plus_fixed_total_bytes": sum(
                int(row["beam2_plus_fixed_bytes"]) for row in ablation_rows
            ),
            "beam2_plus_fixed_matches_hybrid_columns": sum(
                int(row["beam2_plus_fixed_matches_hybrid"]) for row in ablation_rows
            ),
            "beam2_plus_fixed_max_evaluations": 141,
        },
        "input_hashes": {
            name: sha256(experiment_dir / name)
            for name in (
                "SUMMARY.json",
                "per_seed_test.csv",
                "per_column_summary.csv",
                "validation_dse.csv",
                "selected_config.csv",
                "training_summary.csv",
            )
        },
        "claim_boundary": (
            "Independent arithmetic and Beam-2 ablation for the standalone normalized-int64 "
            "Python codec; not a production-codec audit."
        ),
    }
    (experiment_dir / "RECOMPUTE.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
