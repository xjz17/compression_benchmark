#!/usr/bin/env python3
"""Frozen external evaluation of Full60 versus a minimal PV2F PPO observation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXTERNAL_COLUMNS = (
    ("Bird-migration", 0),
    ("Bird-migration", 1),
    ("Bird-migration", 2),
    ("Bird-migration", 3),
    ("Bird-migration", 4),
    ("meteoblue", 0),
    ("meteoblue", 5),
    ("meteoblue", 10),
    ("meteoblue", 15),
    ("meteoblue", 20),
    ("meteoblue", 25),
    ("meteoblue", 30),
    ("meteoblue", 35),
    ("meteoblue", 40),
    ("meteoblue", 45),
    ("meteoblue", 50),
    ("meteoblue", 55),
    ("meteoblue", 60),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        hidden=64,
        actor_lr=0.001,
        critic_lr=0.001,
        iterations=args.iterations,
        ppo_episodes=256,
        grpo_groups=32,
        group_size=8,
        gae_lambda=0.95,
        policy_epochs=4,
        value_epochs=4,
        minibatch=256,
        clip_ratio=0.2,
        entropy_coefficient=0.01,
        proposal_budget=args.proposal_budget,
        log_every=args.iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-raw-dims", type=int, nargs="+", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--proposal-budget", type=int, default=512)
    args = parser.parse_args()
    if args.candidate_raw_dims != sorted(set(args.candidate_raw_dims)) or 8 in args.candidate_raw_dims:
        parser.error("candidate raw dims must be sorted, unique, and exclude duplicate dimension 8")

    script = Path(__file__).resolve()
    script_dir = script.parent
    sys.path.insert(0, str(script_dir))
    base_path = script_dir / "search_physical_operator_variants_v2.py"
    ablation_path = script_dir / "ablate_physical_policy_features.py"
    runtime_path = script_dir / "physical_operator_runtime_v2.py"
    base = load_module(base_path, "physical_external_base")
    ablation = load_module(ablation_path, "physical_external_ablation")
    runtime = load_module(runtime_path, "physical_external_runtime")
    repo_root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "policies").mkdir()

    dev_tables = base.load_tables(args.measurement_root.resolve())
    dev_raw, _dev_rows = base.load_feature_matrix(repo_root, dev_tables)
    train_columns = np.arange(len(dev_tables), dtype=np.int32)
    external_values: list[np.ndarray] = []
    external_metadata: list[dict[str, str]] = []
    external_raw: list[np.ndarray] = []
    input_rows: list[dict[str, Any]] = []
    for dataset, column in EXTERNAL_COLUMNS:
        values, metadata, data_path, info_path = base.load_column(repo_root, dataset, column, 0)
        feature_started = time.perf_counter_ns()
        raw = base.column_features(values)
        full_feature_ms = (time.perf_counter_ns() - feature_started) / 1.0e6
        candidate_started = time.perf_counter_ns()
        selected = ablation.extract_selected(values, tuple(args.candidate_raw_dims))
        candidate_feature_ms = (time.perf_counter_ns() - candidate_started) / 1.0e6
        if not np.allclose(selected, raw[args.candidate_raw_dims], rtol=0.0, atol=1.0e-12):
            raise AssertionError((dataset, column))
        external_values.append(values)
        external_metadata.append(metadata)
        external_raw.append(raw)
        input_rows.append(
            {
                "dataset": dataset,
                "column_index": column,
                "column_name": metadata["column_name"],
                "logical_type": metadata["logical_type"],
                "values": values.size,
                "original_bytes": values.size * 8,
                "input_sha256": runtime.sha256_bytes(runtime.raw_i64(values)),
                "full60_feature_ms": full_feature_ms,
                "candidate_feature_ms": candidate_feature_ms,
                "data_path": str(data_path),
                "info_path": str(info_path),
            }
        )
    write_csv(output / "external_inputs.csv", input_rows)
    combined_raw = np.vstack((dev_raw, np.vstack(external_raw)))

    variants = (
        {"name": "full60", "raw_dims": tuple(range(17)), "prefix": "full", "embedding": 0},
        {"name": args.candidate_name, "raw_dims": tuple(args.candidate_raw_dims), "prefix": "embedding", "embedding": 1},
    )
    evaluation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    codec_cache: dict[tuple[int, tuple[str, ...]], tuple[int, str, float]] = {}
    original_builder = base.FeatureBuilder
    original_mlp = base.MLP
    started = time.perf_counter()
    for variant in variants:
        raw_dims = variant["raw_dims"]
        prefix = variant["prefix"]
        embedding = variant["embedding"]
        base.FeatureBuilder = lambda features, train, rd=raw_dims, pm=prefix, ed=embedding: ablation.AblationFeatureBuilder(
            base, features, train, rd, pm, ed
        )
        base.MLP = original_mlp if prefix != "embedding" else ablation.embedding_mlp_class(base, len(raw_dims), embedding)
        for seed in args.seeds:
            trained_at = time.perf_counter()
            _evaluation, curve, actor = base.train_algorithm(
                "ppo", seed, 0, dev_tables, dev_raw, train_columns, np.asarray([], dtype=np.int32), train_args(args)
            )
            train_sec = time.perf_counter() - trained_at
            for row in curve:
                row.update({"variant": variant["name"], "train_all_dev": 1})
            training_rows.extend(curve)
            np.savez_compressed(output / "policies" / f"{variant['name']}_seed{seed}.npz", *actor.params)
            builder = ablation.AblationFeatureBuilder(base, combined_raw, train_columns, raw_dims, prefix, embedding)
            for external_index, ((dataset, column), values, metadata) in enumerate(
                zip(EXTERNAL_COLUMNS, external_values, external_metadata)
            ):
                global_column = len(dev_tables) + external_index
                greedy = base.propose_pipeline(actor, builder, global_column, None)
                rng = np.random.default_rng(seed + 90_000_019 + 1009 * (external_index + 1))
                proposal_started = time.perf_counter_ns()
                proposals = {greedy, ("raw.i64", "none")}
                for _ in range(args.proposal_budget):
                    proposals.add(base.propose_pipeline(actor, builder, global_column, rng))
                proposal_ms = (time.perf_counter_ns() - proposal_started) / 1.0e6
                valid: list[tuple[tuple[str, ...], int, str]] = []
                for pipeline in sorted(proposals):
                    cache_key = (external_index, pipeline)
                    if cache_key not in codec_cache:
                        codec_started = time.perf_counter_ns()
                        try:
                            transforms, packer, compressor = base.split_prefix(pipeline)
                            if packer is None or compressor is None:
                                raise AssertionError(pipeline)
                            encoded = runtime.encode_pipeline(values, transforms, packer, compressor)
                            restored, decoded_pipeline = runtime.decode_frame(encoded.data)
                            if decoded_pipeline != pipeline or not np.array_equal(restored, values):
                                raise AssertionError("external round-trip mismatch")
                            codec_cache[cache_key] = (
                                len(encoded.data),
                                runtime.sha256_bytes(encoded.data),
                                (time.perf_counter_ns() - codec_started) / 1.0e6,
                            )
                            codec_rows.append(
                                {
                                    "dataset": dataset,
                                    "column_index": column,
                                    "column_name": metadata["column_name"],
                                    "pipeline": " -> ".join(pipeline),
                                    "final_bytes": len(encoded.data),
                                    "compression_rate": len(encoded.data) / float(values.size * 8),
                                    "encoded_sha256": runtime.sha256_bytes(encoded.data),
                                    "codec_ms": codec_cache[cache_key][2],
                                    "lossless": 1,
                                    "status": "ok",
                                }
                            )
                        except Exception as exc:
                            codec_cache[cache_key] = (-1, f"{type(exc).__name__}:{str(exc)[:180]}", 0.0)
                            codec_rows.append(
                                {
                                    "dataset": dataset,
                                    "column_index": column,
                                    "column_name": metadata["column_name"],
                                    "pipeline": " -> ".join(pipeline),
                                    "final_bytes": "",
                                    "compression_rate": "",
                                    "encoded_sha256": "",
                                    "codec_ms": "",
                                    "lossless": 0,
                                    "status": codec_cache[cache_key][1],
                                }
                            )
                    size, digest, _codec_ms = codec_cache[cache_key]
                    if size >= 0:
                        valid.append((pipeline, size, digest))
                if not valid:
                    raise AssertionError((variant["name"], dataset, column, seed))
                best_pipeline, best_bytes, best_digest = min(valid, key=lambda item: (item[1], item[0]))
                evaluation_rows.append(
                    {
                        "variant": variant["name"],
                        "seed": seed,
                        "dataset": dataset,
                        "column_index": column,
                        "column_name": metadata["column_name"],
                        "pipeline": " -> ".join(best_pipeline),
                        "encoded_bytes": best_bytes,
                        "compression_rate": best_bytes / float(values.size * 8),
                        "encoded_sha256": best_digest,
                        "proposal_rollouts": args.proposal_budget,
                        "unique_proposals": len(proposals),
                        "valid_unique_proposals": len(valid),
                        "invalid_unique_proposals": len(proposals) - len(valid),
                        "proposal_ms": proposal_ms,
                        "training_sec": train_sec,
                    }
                )

    base.FeatureBuilder = original_builder
    base.MLP = original_mlp
    write_csv(output / "evaluation.csv", evaluation_rows)
    write_csv(output / "training_curve.csv", training_rows)
    write_csv(output / "codec_replay.csv", codec_rows)
    paired_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        full = [row for row in evaluation_rows if row["variant"] == "full60" and row["seed"] == seed]
        candidate = [row for row in evaluation_rows if row["variant"] == args.candidate_name and row["seed"] == seed]
        full_total = sum(int(row["encoded_bytes"]) for row in full)
        candidate_total = sum(int(row["encoded_bytes"]) for row in candidate)
        paired_rows.append(
            {
                "seed": seed,
                "full60_bytes": full_total,
                "candidate_bytes": candidate_total,
                "candidate_penalty_pct": 100.0 * (candidate_total - full_total) / full_total,
                "full60_median_unique_proposals": statistics.median(int(row["unique_proposals"]) for row in full),
                "candidate_median_unique_proposals": statistics.median(int(row["unique_proposals"]) for row in candidate),
            }
        )
    write_csv(output / "paired_seed_summary.csv", paired_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate_name,
        "candidate_raw_dims": args.candidate_raw_dims,
        "development_datasets": sorted({table.dataset for table in dev_tables}),
        "external_datasets": sorted({dataset for dataset, _column in EXTERNAL_COLUMNS}),
        "external_columns": len(EXTERNAL_COLUMNS),
        "seeds": args.seeds,
        "median_full60_bytes": statistics.median(row["full60_bytes"] for row in paired_rows),
        "median_candidate_bytes": statistics.median(row["candidate_bytes"] for row in paired_rows),
        "median_candidate_penalty_pct": statistics.median(row["candidate_penalty_pct"] for row in paired_rows),
        "codec_unique_rows": len(codec_rows),
        "codec_lossless_rows": sum(int(row["lossless"]) for row in codec_rows),
        "codec_failed_rows": sum(not int(row["lossless"]) for row in codec_rows),
        "wall_sec": time.perf_counter() - started,
        "claim_boundary": "frozen Bird-migration/meteoblue external proposal replay with actual PV2F frames; no external oracle enumeration",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": summary["created_utc"],
        "source_sha256": {"runner": sha256(script), "base": sha256(base_path), "ablation": sha256(ablation_path), "runtime": sha256(runtime_path)},
        "measurement_manifests": {str(path): sha256(path) for path in sorted(args.measurement_root.resolve().glob("*/manifest.json"))},
        "summary_sha256": sha256(output / "summary.json"),
        "evaluation_sha256": sha256(output / "evaluation.csv"),
        "codec_replay_sha256": sha256(output / "codec_replay.csv"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
