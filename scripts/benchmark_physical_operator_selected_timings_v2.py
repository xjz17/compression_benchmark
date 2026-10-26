#!/usr/bin/env python3
"""Benchmark selection lookup and full encoding for reported PV2F pipelines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from enumerate_physical_operator_variants_v2 import load_column
from physical_operator_runtime_v2 import encode_pipeline
from search_physical_operator_variants_v2 import (
    FeatureBuilder,
    MLP,
    Sparse19FeatureBuilder,
    Sparse19MLP,
    beam29,
    load_feature_matrix,
    load_tables,
    parse_pipeline,
    propose_pipeline,
    propose_pipeline_sparse19,
    split_prefix,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no timing rows")
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


def representative(rows: list[dict[str, str]]) -> dict[str, str]:
    median_bytes = statistics.median(int(row["encoded_bytes"]) for row in rows)
    return min(
        rows,
        key=lambda row: (
            abs(int(row["encoded_bytes"]) - median_bytes),
            int(row["encoded_bytes"]),
            int(row["seed"]),
            row["pipeline"],
        ),
    )


def distribution_summary(samples_ns: list[int]) -> dict[str, float]:
    samples_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    return {
        "median": float(np.median(samples_ms)),
        "p25": float(np.percentile(samples_ms, 25)),
        "p75": float(np.percentile(samples_ms, 75)),
        "min": float(np.min(samples_ms)),
        "max": float(np.max(samples_ms)),
    }


def load_actor(policy_path: Path, input_dim: int) -> MLP:
    with np.load(policy_path) as archive:
        params = [archive[f"arr_{index}"].copy() for index in range(4)]
    hidden = int(params[1].shape[0])
    output_dim = int(params[3].shape[0])
    actor = MLP(input_dim, hidden, output_dim, np.random.default_rng(0))
    if any(actual.shape != expected.shape for actual, expected in zip(params, actor.params)):
        raise AssertionError(f"policy shape mismatch: {policy_path}")
    actor.params = params
    return actor


def load_sparse19_actor(policy_path: Path) -> Sparse19MLP:
    with np.load(policy_path) as archive:
        params = [archive[f"arr_{index}"].copy() for index in range(4)]
    if params[0].shape[0] != 53:
        raise AssertionError(f"sparse19 policy shape mismatch: {policy_path} {params[0].shape}")
    actor = Sparse19MLP(int(params[1].shape[0]), int(params[3].shape[0]), np.random.default_rng(0))
    if any(actual.shape != expected.shape for actual, expected in zip(params, actor.params)):
        raise AssertionError(f"sparse19 policy shape mismatch: {policy_path}")
    actor.params = params
    return actor


def policy_select(
    actor: MLP,
    builder: FeatureBuilder,
    table_index: int,
    table: Any,
    seed: int,
    fold: int,
    proposal_budget: int,
) -> tuple[tuple[str, ...], int]:
    greedy = propose_pipeline(actor, builder, table_index, None)
    rng = np.random.default_rng(seed + 10_000_019 * (fold + 1) + 1009 * (table_index + 1))
    proposals = {greedy, ("raw.i64", "none")}
    for _ in range(proposal_budget):
        proposals.add(propose_pipeline(actor, builder, table_index, rng))
    valid = [pipeline for pipeline in proposals if pipeline in table.sizes]
    if not valid:
        raise AssertionError(f"no valid proposal for {table.key}")
    return min(valid, key=lambda pipeline: (table.sizes[pipeline], pipeline)), len(proposals)


def sparse19_policy_select(
    actor: Sparse19MLP,
    builder: Sparse19FeatureBuilder,
    table_index: int,
    table: Any,
    seed: int,
    fold: int,
    proposal_budget: int,
) -> tuple[tuple[str, ...], int]:
    greedy = propose_pipeline_sparse19(actor, builder, table_index, None)
    rng = np.random.default_rng(seed + 10_000_019 * (fold + 1) + 1009 * (table_index + 1))
    proposals = {greedy, ("raw.i64", "none")}
    for _ in range(proposal_budget):
        proposals.add(propose_pipeline_sparse19(actor, builder, table_index, rng))
    valid = [pipeline for pipeline in proposals if pipeline in table.sizes]
    if not valid:
        raise AssertionError(f"no valid sparse19 proposal for {table.key}")
    return min(valid, key=lambda pipeline: (table.sizes[pipeline], pipeline)), len(proposals)


def is_single_codec(pipeline: tuple[str, ...]) -> bool:
    """One non-trivial transform, packer, or compressor in a complete PV2F frame."""
    transforms, packer, compressor = split_prefix(pipeline)
    if packer is None or compressor is None:
        return False
    return len(transforms) + int(packer != "raw.i64") + int(compressor != "none") == 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path)
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation", choices=("full60", "sparse19"), default="full60")
    parser.add_argument("--selection-repeats", type=int, default=31)
    parser.add_argument("--encode-repeats", type=int, default=9)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if min(args.selection_repeats, args.encode_repeats) <= 0:
        raise ValueError("repeat counts must be positive")
    repo_root = args.repo_root.resolve()
    measurement_root = args.measurement_root.resolve()
    search_dir = args.search_dir.resolve()
    policy_dir = args.policy_dir.resolve() if args.policy_dir else search_dir / "policies"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tables = load_tables(measurement_root)
    table_index_by_key = {table.key: index for index, table in enumerate(tables)}
    if args.feature_csv:
        feature_rows = read_csv(args.feature_csv.resolve())
        if len(feature_rows) != len(tables):
            raise AssertionError((args.feature_csv, len(feature_rows), len(tables)))
        for table, row in zip(tables, feature_rows):
            if (
                row["dataset"] != table.dataset
                or int(row["column_index"]) != table.column_index
                or row["column_name"] != table.column_name
            ):
                raise AssertionError((table.key, row))
        raw_features = np.asarray(
            [[float(row[f"feature_{index}"]) for index in range(17)] for row in feature_rows],
            dtype=np.float64,
        )
    else:
        raw_features, _feature_rows = load_feature_matrix(repo_root, tables)
    beam_rows = read_csv(search_dir / "beam29_results.csv")
    rl_rows = read_csv(search_dir / "rl_evaluation.csv")
    proposal_budget = int(rl_rows[0]["proposal_rollouts"])

    grouped_rl: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rl_rows:
        grouped_rl[(row["test_dataset"], int(row["column_index"]), row["algorithm"])].append(row)

    builders: dict[int, FeatureBuilder | Sparse19FeatureBuilder] = {}
    actors: dict[tuple[str, int, int], MLP | Sparse19MLP] = {}
    for fold, test_dataset in enumerate(sorted({table.dataset for table in tables})):
        train_indices = np.asarray(
            [index for index, table in enumerate(tables) if table.dataset != test_dataset],
            dtype=np.int32,
        )
        builders[fold] = (
            Sparse19FeatureBuilder(raw_features, train_indices)
            if args.observation == "sparse19"
            else FeatureBuilder(raw_features, train_indices)
        )

    method_specs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for beam in beam_rows:
        key = (beam["dataset"], int(beam["column_index"]))
        table = tables[table_index_by_key[key]]
        single_candidates = [pipeline for pipeline in table.sizes if is_single_codec(pipeline)]
        if not single_candidates:
            raise AssertionError((key, "no single-codec candidate"))
        single_pipeline = min(single_candidates, key=lambda pipeline: (table.sizes[pipeline], pipeline))
        single_bytes = int(table.sizes[single_pipeline])
        method_specs[key].append(
            {
                "method": "Bounded oracle",
                "seed": "",
                "fold": "",
                "pipeline": beam["oracle_pipeline"],
                "encoded_bytes": int(beam["oracle_bytes"]),
                "compression_rate": float(beam["oracle_rate"]),
                "candidate_count": int(beam["complete_valid_pipelines"]),
            }
        )
        method_specs[key].append(
            {
                "method": "Best single codec",
                "seed": "",
                "fold": "",
                "pipeline": " -> ".join(single_pipeline),
                "encoded_bytes": single_bytes,
                "compression_rate": single_bytes / int(beam["original_bytes"]),
                "candidate_count": len(single_candidates),
            }
        )
        method_specs[key].append(
            {
                "method": "Beam-29",
                "seed": "",
                "fold": "",
                "pipeline": beam["pipeline"],
                "encoded_bytes": int(beam["encoded_bytes"]),
                "compression_rate": float(beam["compression_rate"]),
                "candidate_count": int(beam["candidate_evaluations"]),
            }
        )
        for algorithm, label in (("ppo", "PPO"), ("grpo", "GRPO-style")):
            selected = representative(grouped_rl[(key[0], key[1], algorithm)])
            method_specs[key].append(
                {
                    "method": label,
                    "algorithm": algorithm,
                    "seed": int(selected["seed"]),
                    "fold": int(selected["fold"]),
                    "pipeline": selected["pipeline"],
                    "encoded_bytes": int(selected["encoded_bytes"]),
                    "compression_rate": float(selected["compression_rate"]),
                    "candidate_count": int(selected["unique_proposals"]),
                }
            )

    result_rows: list[dict[str, Any]] = []
    for table in tables:
        key = table.key
        table_index = table_index_by_key[key]
        values, _metadata, _data_path, _info_path = load_column(
            repo_root, table.dataset, table.column_index, 0
        )
        if values.size * 8 != table.original_bytes:
            raise AssertionError((key, values.size * 8, table.original_bytes))
        specs = method_specs[key]

        selection_samples: dict[str, list[int]] = {spec["method"]: [] for spec in specs}
        for spec in specs:
            if spec["method"] in {"Bounded oracle", "Best single codec"}:
                continue
            expected_pipeline = parse_pipeline(spec["pipeline"])
            for _ in range(args.selection_repeats):
                started = time.perf_counter_ns()
                if spec["method"] == "Beam-29":
                    selected_pipeline = beam29(table, 29)["pipeline"]
                else:
                    algorithm = spec["algorithm"]
                    seed = int(spec["seed"])
                    fold = int(spec["fold"])
                    actor_key = (algorithm, seed, fold)
                    if actor_key not in actors:
                        policy_path = policy_dir / f"{algorithm}_seed{seed}_fold{fold}.npz"
                        actor = (
                            load_sparse19_actor(policy_path)
                            if args.observation == "sparse19"
                            else load_actor(policy_path, builders[fold].input_dim)
                        )
                        actors[actor_key] = actor
                    if args.observation == "sparse19":
                        selected_pipeline, proposal_count = sparse19_policy_select(
                            actors[actor_key], builders[fold], table_index, table,
                            seed, fold, proposal_budget,
                        )
                    else:
                        selected_pipeline, proposal_count = policy_select(
                            actors[actor_key], builders[fold], table_index, table,
                            seed, fold, proposal_budget,
                        )
                    if proposal_count != int(spec["candidate_count"]):
                        raise AssertionError((key, spec["method"], proposal_count, spec["candidate_count"]))
                elapsed = time.perf_counter_ns() - started
                if selected_pipeline != expected_pipeline:
                    raise AssertionError((key, spec["method"], selected_pipeline, expected_pipeline))
                selection_samples[spec["method"]].append(elapsed)

        encode_samples: dict[str, list[int]] = {spec["method"]: [] for spec in specs}
        parsed: dict[str, tuple[tuple[str, ...], str, str]] = {}
        for spec in specs:
            transforms, packer, compressor = split_prefix(parse_pipeline(spec["pipeline"]))
            if packer is None or compressor is None:
                raise AssertionError(spec["pipeline"])
            parsed[spec["method"]] = (transforms, packer, compressor)
            frame = encode_pipeline(values, transforms, packer, compressor)
            if len(frame.data) != int(spec["encoded_bytes"]):
                raise AssertionError((key, spec["method"], len(frame.data), spec["encoded_bytes"]))
        order_rng = np.random.default_rng(20260826 + table_index)
        for _ in range(args.encode_repeats):
            for position in order_rng.permutation(len(specs)):
                spec = specs[int(position)]
                transforms, packer, compressor = parsed[spec["method"]]
                started = time.perf_counter_ns()
                frame = encode_pipeline(values, transforms, packer, compressor)
                elapsed = time.perf_counter_ns() - started
                if len(frame.data) != int(spec["encoded_bytes"]):
                    raise AssertionError((key, spec["method"], len(frame.data), spec["encoded_bytes"]))
                encode_samples[spec["method"]].append(elapsed)

        for spec in specs:
            selection = (
                distribution_summary(selection_samples[spec["method"]])
                if selection_samples[spec["method"]]
                else None
            )
            encoding = distribution_summary(encode_samples[spec["method"]])
            result_rows.append(
                {
                    "dataset": table.dataset,
                    "column_index": table.column_index,
                    "column_name": table.column_name,
                    "logical_type": table.logical_type,
                    "sample_values": values.size,
                    "original_bytes": table.original_bytes,
                    "method": spec["method"],
                    "representative_seed": spec["seed"],
                    "fold": spec["fold"],
                    "pipeline": spec["pipeline"],
                    "encoded_bytes": spec["encoded_bytes"],
                    "compression_rate": spec["compression_rate"],
                    "candidate_count": spec["candidate_count"],
                    "selection_repeats": args.selection_repeats if selection else "",
                    "selection_ms_median": selection["median"] if selection else "",
                    "selection_ms_p25": selection["p25"] if selection else "",
                    "selection_ms_p75": selection["p75"] if selection else "",
                    "selection_ms_min": selection["min"] if selection else "",
                    "selection_ms_max": selection["max"] if selection else "",
                    "encode_repeats": args.encode_repeats,
                    "encode_ms_median": encoding["median"],
                    "encode_ms_p25": encoding["p25"],
                    "encode_ms_p75": encoding["p75"],
                    "encode_ms_min": encoding["min"],
                    "encode_ms_max": encoding["max"],
                }
            )
        print(json.dumps({"finished_column": key}, ensure_ascii=False), flush=True)

    csv_path = output_dir / "selected_pipeline_timings.csv"
    write_csv(csv_path, result_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "columns": len(tables),
        "rows": len(result_rows),
        "methods": ["Bounded oracle", "Best single codec", "Beam-29", "PPO", "GRPO-style"],
        "selection_repeats": args.selection_repeats,
        "encode_repeats": args.encode_repeats,
        "observation": args.observation,
        "logical_observation_fields": 19 if args.observation == "sparse19" else 60,
        "policy_dir": str(policy_dir),
        "selection_definition": "Frozen byte tables and loaded policies/features; Beam table-lookup search or PPO/GRPO-style 512-rollout proposal generation plus test-table best selection. Sparse19 uses 16 continuous fields and three categorical lookups. Bounded oracle and best-single-codec are offline baselines with no reported online selection time. Excludes dataset I/O, feature extraction, policy loading, and real candidate compression.",
        "encode_definition": "Full encode_pipeline from normalized values through transforms, packer, compressor and PV2F framing. Excludes decode, lossless comparison, hashing, dataset I/O, and operator selection.",
        "source_sha256": {
            "timing_script": sha256_file(Path(__file__).resolve()),
            "runtime": sha256_file(Path(__file__).with_name("physical_operator_runtime_v2.py")),
            "search": sha256_file(Path(__file__).with_name("search_physical_operator_variants_v2.py")),
            "rl_evaluation": sha256_file(search_dir / "rl_evaluation.csv"),
            "beam29_results": sha256_file(search_dir / "beam29_results.csv"),
            "feature_csv": sha256_file(args.feature_csv.resolve()) if args.feature_csv else None,
            "timing_csv": sha256_file(csv_path),
            "policy_files": {
                path.name: sha256_file(path) for path in sorted(policy_dir.glob("*.npz"))
            },
        },
        "claim_boundary": "Post-hoc benchmark timing of saved results; lookup-selection time is not end-to-end online codec search, and encode time is prototype Python wall clock rather than a production kernel benchmark.",
    }
    (output_dir / "TIMING_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
