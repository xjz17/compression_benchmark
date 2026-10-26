#!/usr/bin/env python3
"""Deterministically audit PV2F measurements and Beam/PPO/GRPO result claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from enumerate_physical_operator_variants_v2 import load_column
from physical_operator_runtime_v2 import (
    COMPRESSORS,
    PACKERS,
    TRANSFORMS,
    decode_frame,
    encode_pipeline,
    raw_i64,
    sha256_bytes,
)
from search_physical_operator_variants_v2 import beam29, load_tables, parse_pipeline


TRANSFORM_SET = {spec.name for spec in TRANSFORMS}
PACKER_SET = {spec.name for spec in PACKERS}
COMPRESSOR_SET = {spec.name for spec in COMPRESSORS}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_complete_pipeline(pipeline: tuple[str, ...]) -> tuple[tuple[str, ...], str, str]:
    if len(pipeline) < 2:
        raise AssertionError(f"short complete pipeline {pipeline}")
    transforms = pipeline[:-2]
    packer = pipeline[-2]
    compressor = pipeline[-1]
    if any(name not in TRANSFORM_SET for name in transforms):
        raise AssertionError(f"invalid transform in {pipeline}")
    if packer not in PACKER_SET or compressor not in COMPRESSOR_SET:
        raise AssertionError(f"invalid terminal actions in {pipeline}")
    return transforms, packer, compressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    measurement_root = args.measurement_root.resolve()
    search_dir = args.search_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(measurement_root)
    table_by_key = {table.key: table for table in tables}
    measurement_rows: dict[tuple[str, int], dict[tuple[str, ...], dict[str, str]]] = {}
    checked_component_rows = 0
    failed_rows = 0
    input_hashes: dict[tuple[str, int], str] = {}
    for manifest_path in sorted(measurement_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = (manifest["dataset"], int(manifest["column_index"]))
        csv_path = manifest_path.with_name("pipeline_measurements.csv")
        if sha256_file(csv_path) != manifest["source_sha256"]["pipeline_measurements"]:
            raise AssertionError(f"measurement hash mismatch {csv_path}")
        rows = read_csv(csv_path)
        if len(rows) != int(manifest["expected_candidates"]):
            raise AssertionError(f"row count mismatch {csv_path}")
        by_pipeline: dict[tuple[str, ...], dict[str, str]] = {}
        for row in rows:
            pipeline = parse_pipeline(row["pipeline"])
            if pipeline in by_pipeline:
                raise AssertionError(f"duplicate pipeline {pipeline} in {csv_path}")
            by_pipeline[pipeline] = row
            if row["status"] != "ok":
                failed_rows += 1
                continue
            outer_sum = (
                int(row["outer_header_bytes"])
                + int(row["action_table_bytes"])
                + int(row["metadata_length_prefix_bytes"])
                + int(row["transform_metadata_bytes"])
                + int(row["checksum_bytes"])
            )
            if outer_sum != int(row["outer_frame_bytes"]):
                raise AssertionError(f"outer frame byte sum mismatch {key} {pipeline}")
            if outer_sum + int(row["compressed_payload_bytes"]) != int(row["final_bytes"]):
                raise AssertionError(f"final byte sum mismatch {key} {pipeline}")
            expected_rate = int(row["final_bytes"]) / int(row["original_bytes"])
            if not math.isclose(expected_rate, float(row["compression_rate"]), rel_tol=0.0, abs_tol=1.0e-15):
                raise AssertionError(f"rate mismatch {key} {pipeline}")
            if int(row["lossless"]) != 1:
                raise AssertionError(f"ok row is not lossless {key} {pipeline}")
            checked_component_rows += 1
        ok_rows = [row for row in rows if row["status"] == "ok"]
        best = min(ok_rows, key=lambda row: (int(row["final_bytes"]), row["pipeline"]))
        if best["pipeline"] != manifest["best"]["pipeline"] or int(best["final_bytes"]) != int(manifest["best"]["final_bytes"]):
            raise AssertionError(f"manifest best mismatch {key}")
        input_hashes[key] = str(manifest["input_sha256"])
        measurement_rows[key] = by_pipeline

    beam_rows = read_csv(search_dir / "beam29_results.csv")
    if len(beam_rows) != len(tables):
        raise AssertionError("Beam result coverage mismatch")
    claimed: dict[tuple[str, int], set[tuple[str, ...]]] = defaultdict(set)
    for row in beam_rows:
        key = (row["dataset"], int(row["column_index"]))
        table = table_by_key[key]
        recomputed = beam29(table, int(row["beam_width"]))
        pipeline = parse_pipeline(row["pipeline"])
        oracle = parse_pipeline(row["oracle_pipeline"])
        if pipeline != recomputed["pipeline"] or oracle != recomputed["oracle_pipeline"]:
            raise AssertionError(f"Beam pipeline mismatch {key}")
        numeric_pairs = (
            (int(row["encoded_bytes"]), recomputed["bytes"]),
            (int(row["oracle_bytes"]), recomputed["oracle_bytes"]),
            (int(row["candidate_evaluations"]), recomputed["candidate_evaluations"]),
            (int(row["invalid_evaluations"]), recomputed["invalid_evaluations"]),
        )
        if any(left != right for left, right in numeric_pairs):
            raise AssertionError(f"Beam numeric mismatch {key}: {numeric_pairs}")
        claimed[key].update((pipeline, oracle))

    rl_rows = read_csv(search_dir / "rl_evaluation.csv")
    expected_rl_rows = len(tables) * 2 * 3
    if len(rl_rows) != expected_rl_rows:
        raise AssertionError(f"RL row coverage mismatch: {len(rl_rows)} != {expected_rl_rows}")
    coverage: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for row in rl_rows:
        key = (row["test_dataset"], int(row["column_index"]))
        algorithm = row["algorithm"]
        seed = int(row["seed"])
        coverage[(key[0], key[1], algorithm)].add(seed)
        table = table_by_key[key]
        pipeline = parse_pipeline(row["pipeline"])
        greedy = parse_pipeline(row["greedy_pipeline"])
        oracle = parse_pipeline(row["oracle_pipeline"])
        if pipeline not in table.sizes:
            raise AssertionError(f"RL selected missing pipeline {key} {pipeline}")
        if int(row["encoded_bytes"]) != table.sizes[pipeline] or not math.isclose(float(row["compression_rate"]), table.rates[pipeline], rel_tol=0.0, abs_tol=1.0e-15):
            raise AssertionError(f"RL selected metric mismatch {key} {pipeline}")
        if oracle not in table.sizes or int(row["oracle_bytes"]) != table.sizes[oracle]:
            raise AssertionError(f"RL oracle mismatch {key}")
        if int(row["greedy_bytes"]) != table.sizes.get(greedy, int(table.original_bytes * 10.0)):
            raise AssertionError(f"RL greedy mismatch {key}")
        claimed[key].update((pipeline, oracle))
        if greedy in table.sizes:
            claimed[key].add(greedy)
    expected_seeds = {20260825, 20260826, 20260827}
    if len(coverage) != len(tables) * 2 or any(seeds != expected_seeds for seeds in coverage.values()):
        raise AssertionError("RL algorithm/seed coverage mismatch")

    replayed = 0
    replay_rows: list[dict[str, Any]] = []
    for key, pipelines in sorted(claimed.items()):
        table = table_by_key[key]
        values, _metadata, _data, _info = load_column(
            repo_root, table.dataset, table.column_index, 0
        )
        if sha256_bytes(raw_i64(values)) != input_hashes[key]:
            raise AssertionError(f"input hash mismatch during replay {key}")
        for pipeline in sorted(pipelines):
            transforms, packer, compressor = split_complete_pipeline(pipeline)
            encoded = encode_pipeline(values, transforms, packer, compressor)
            restored, decoded_pipeline = decode_frame(encoded.data)
            if decoded_pipeline != pipeline or not np.array_equal(restored, values):
                raise AssertionError(f"decoder replay mismatch {key} {pipeline}")
            row = measurement_rows[key][pipeline]
            if encoded.components["final_bytes"] != int(row["final_bytes"]):
                raise AssertionError(f"replay bytes mismatch {key} {pipeline}")
            if sha256_bytes(encoded.data) != row["encoded_sha256"]:
                raise AssertionError(f"replay encoded hash mismatch {key} {pipeline}")
            replay_rows.append(
                {
                    "dataset": key[0],
                    "column_index": key[1],
                    "pipeline": " -> ".join(pipeline),
                    "final_bytes": encoded.components["final_bytes"],
                    "encoded_sha256": sha256_bytes(encoded.data),
                    "lossless": 1,
                }
            )
            replayed += 1

    summary = json.loads((search_dir / "summary.json").read_text(encoding="utf-8"))
    beam_total = sum(int(row["encoded_bytes"]) for row in beam_rows)
    oracle_total = sum(int(row["oracle_bytes"]) for row in beam_rows)
    if beam_total != int(summary["beam29"]["total_bytes"]) or oracle_total != int(summary["beam29"]["oracle_total_bytes"]):
        raise AssertionError("summary Beam totals mismatch")
    recomputed_algorithms: list[dict[str, Any]] = []
    for algorithm in ("ppo", "grpo"):
        rows = [row for row in rl_rows if row["algorithm"] == algorithm]
        totals = []
        beam_totals = []
        oracle_totals = []
        for seed in sorted(expected_seeds):
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            totals.append(sum(int(row["encoded_bytes"]) for row in seed_rows))
            beam_totals.append(sum(int(row["beam29_bytes"]) for row in seed_rows))
            oracle_totals.append(sum(int(row["oracle_bytes"]) for row in seed_rows))
        recomputed_algorithms.append(
            {
                "algorithm": algorithm,
                "median_total_bytes": statistics.median(totals),
                "median_beam29_total_bytes": statistics.median(beam_totals),
                "median_oracle_total_bytes": statistics.median(oracle_totals),
            }
        )
    published_by_algorithm = {row["algorithm"]: row for row in summary["algorithms"]}
    for recomputed in recomputed_algorithms:
        published = published_by_algorithm[recomputed["algorithm"]]
        for key in ("median_total_bytes", "median_beam29_total_bytes", "median_oracle_total_bytes"):
            if float(recomputed[key]) != float(published[key]):
                raise AssertionError(f"algorithm summary mismatch {recomputed['algorithm']} {key}")

    replay_path = output_dir / "claimed_pipeline_replay.csv"
    with replay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(replay_rows)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "evaluation_type": "real_gt_normalized_int64_physical_prototype",
        "measurement_columns": len(tables),
        "checked_component_rows": checked_component_rows,
        "recorded_failed_rows": failed_rows,
        "beam_rows": len(beam_rows),
        "rl_rows": len(rl_rows),
        "replayed_unique_claimed_pipelines": replayed,
        "checks": {
            "measurement_hashes": "PASS",
            "candidate_counts": "PASS",
            "component_byte_sums": "PASS",
            "manifest_oracle": "PASS",
            "beam_recompute": "PASS",
            "rl_table_lookup_and_coverage": "PASS",
            "decoder_only_claimed_pipeline_replay": "PASS",
            "aggregate_recompute": "PASS",
        },
        "recomputed_algorithms": recomputed_algorithms,
        "source_sha256": {
            "audit_script": sha256_file(Path(__file__).resolve()),
            "search_summary": sha256_file(search_dir / "summary.json"),
            "beam_results": sha256_file(search_dir / "beam29_results.csv"),
            "rl_evaluation": sha256_file(search_dir / "rl_evaluation.csv"),
            "replay_csv": sha256_file(replay_path),
        },
        "claim_boundary": "deterministic integrity of the bounded PV2F prototype only",
    }
    (output_dir / "DETERMINISTIC_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "DETERMINISTIC_AUDIT.md").write_text(
        "# Deterministic experiment audit\n\n"
        f"Status: **PASS**\n\n"
        f"- Checked physical byte component sums: {checked_component_rows:,} rows.\n"
        f"- Recorded failed candidates: {failed_rows:,} rows.\n"
        f"- Recomputed Beam rows: {len(beam_rows)}.\n"
        f"- Verified PPO/GRPO rows and seed coverage: {len(rl_rows)}.\n"
        f"- Re-encoded and decoder-replayed claimed pipelines: {replayed}.\n"
        "- Claim boundary: bounded normalized-int64 PV2F prototype.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run(parse_args())
